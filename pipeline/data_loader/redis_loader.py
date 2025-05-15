import json
import pandas as pd
from redis import Redis

from redis.commands.search.query import Query
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

# Add the project root to the path
from pipeline.settings import settings

class RedisDataLoader:
    """
    Class to handle loading data into Redis and creating appropriate indexes
    """
    def __init__(self, redis_url: str = settings.redis_url):
        """Initialize Redis connection"""
        self.client = Redis.from_url(redis_url)
        
    def flush_db(self) -> bool:
        """
        Clear all data in Redis
        
        Returns:
            Success status
        """
        return self.client.flushall()
    
    def load_transactions(self, filepath: str) -> int:
        """
        Load transaction data from JSON file to Redis
        
        Args:
            filepath: Path to the JSON file containing transaction data
            
        Returns:
            Number of transactions loaded
        """
        # Load transactions from JSON file
        with open(filepath, 'r') as f:
            transactions = json.load(f)
        
        # Load each transaction into Redis as JSON
        pipe = self.client.pipeline()
        for txn in transactions:
            # Process timestamp to numeric if it's a string
            if isinstance(txn.get('timestamp'), str):
                txn['timestamp'] = int(pd.to_datetime(txn['timestamp']).timestamp())
            
            # Add location field if not present
            if 'location' not in txn and 'lat' in txn and 'lon' in txn:
                txn['location'] = f"{txn['lon']},{txn['lat']}"
            
            # Add namespace to the key
            key = f"{settings.namespace}:transaction:{txn['transaction_id']}"
            pipe.json().set(key, '$', txn)
        
        # Execute pipeline
        pipe.execute()
        print(f"Loaded {len(transactions)} transactions into Redis")
        
        return len(transactions)
    
    def create_transaction_index(self) -> None:
        """Create search index for transactions"""
        # Define the schema using RediVL's schema builder
        schema = IndexSchema.from_dict({
            "index": {
                "name": settings.namespaced_transactions_index,
                "prefix": f"{settings.namespace}:transaction:",
                "storage_type": "json"
            },
            "fields": [
                {
                    "name": "transaction_id",
                    "type": "tag",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "user_id",
                    "type": "tag",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "merchant_id",
                    "type": "tag",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "item_name",
                    "type": "text",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "amount",
                    "type": "numeric",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "currency",
                    "type": "tag"
                },
                {
                    "name": "timestamp",
                    "type": "numeric",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "lat",
                    "type": "numeric"
                },
                {
                    "name": "lon",
                    "type": "numeric"
                },
                {
                    "name": "card_provider",
                    "type": "tag"
                },
                {
                    "name": "location",
                    "type": "geo"
                },
                {
                    "name": "vector_text",
                    "type": "text"
                }
            ]
        })
        
        # Create the index
        index = SearchIndex(schema, self.client)
        index.create(overwrite=True, drop=True)
        
        print(f"Created transaction index: {settings.namespaced_transactions_index}")
        
    def create_user_profiles(self) -> int:
        """
        Create user profiles with static information
        
        Returns:
            Number of user profiles created
        """
        # Get all unique users from transactions
        query = Query("*").dialect(2)
        
        try:
            results = self.client.ft(settings.namespaced_transactions_index).search(query)
        except Exception as e:
            print(f"Error searching transactions: {e}")
            return 0
            
        # Extract all user transactions
        user_transactions = {}
        
        for doc in results.docs:
            transaction = json.loads(doc.json)
            if "user_id" in transaction:
                user_id = transaction["user_id"]
                if user_id not in user_transactions:
                    user_transactions[user_id] = []
                user_transactions[user_id].append(transaction)
        
        # Create simplified profile for each user with static attributes
        pipe = self.client.pipeline()
        for user_id, transactions in user_transactions.items():
            if not transactions:
                continue
                
            # Sort transactions by timestamp to find the first one
            transactions.sort(key=lambda x: x.get('timestamp', 0))
            first_txn = transactions[0]
                
            # Find preferred card provider (most used)
            card_providers = {}
            locations = {}
            
            for txn in transactions:
                # Count card providers
                card = txn.get("card_provider", "UNKNOWN")
                card_providers[card] = card_providers.get(card, 0) + 1
                
                # Count locations
                location = txn.get("location")
                if location:
                    locations[location] = locations.get(location, 0) + 1
            
            # Get the most used card provider
            preferred_card = max(card_providers.items(), key=lambda x: x[1])[0] if card_providers else "UNKNOWN"
            
            # Get the most frequent location
            home_location = max(locations.items(), key=lambda x: x[1])[0] if locations else "0,0"
            
            # Calculate total spend for user segmentation
            total_spend = sum(txn.get('amount', 0) for txn in transactions)
            
            # Determine user segment
            if total_spend > 5000:
                user_segment = "PREMIUM"
            elif total_spend > 1000:
                user_segment = "STANDARD"
            else:
                user_segment = "BASIC"
            
            # Create profile
            profile = {
                "user_id": user_id,
                "first_seen": first_txn.get("timestamp", 0),
                "preferred_card": preferred_card,
                "user_segment": user_segment,
                "home_location": home_location,
                "registration_date": first_txn.get("timestamp", 0)
            }
            
            # Store in Redis with namespace
            key = f"{settings.namespace}:user_profile:{user_id}"
            pipe.json().set(key, "$", profile)
        
        # Execute pipeline
        pipe.execute()
        print(f"Created {len(user_transactions)} user profiles with static attributes")
        
        return len(user_transactions)
    
    def create_user_profile_index(self) -> None:
        """Create search index for user profiles"""
        # Define the schema using RediVL's schema builder
        schema = IndexSchema.from_dict({
            "index": {
                "name": settings.namespaced_user_profiles_index,
                "prefix": f"{settings.namespace}:user_profile:",
                "storage_type": "json"
            },
            "fields": [
                {
                    "name": "user_id",
                    "type": "tag",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "first_seen",
                    "type": "numeric",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "preferred_card",
                    "type": "tag"
                },
                {
                    "name": "user_segment",
                    "type": "tag",
                    "attrs": {"sortable": True}
                },
                {
                    "name": "home_location",
                    "type": "geo"
                },
                {
                    "name": "registration_date",
                    "type": "numeric"
                }
            ]
        })
        
        # Create the index
        index = SearchIndex(schema, self.client)
        index.create(overwrite=True, drop=True)
        
        print(f"Created user profile index: {settings.namespaced_user_profiles_index}")
    
    def initialize_redis(self, filepath: str) -> None:
        """
        Initialize Redis with data and create indexes
        
        Args:
            filepath: Path to the transaction data file
        """
        # Clear existing data
        self.flush_db()
        
        # Load transactions
        self.load_transactions(filepath)
        
        # Create indexes
        self.create_transaction_index()
        
        # Create user profiles
        self.create_user_profiles()
        self.create_user_profile_index() 