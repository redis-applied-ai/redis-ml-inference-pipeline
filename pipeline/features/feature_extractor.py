import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import json
import logging
import warnings
from redis import Redis
from redis.commands.search.query import Query
from datetime import datetime, timedelta

# Import settings instance
from pipeline.settings import settings

# Disable all warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extract and transform features for fraud detection model.
    
    This class handles both static and derived features, and provides
    transformation utilities for model training and inference.
    """
    def __init__(self, redis_url=None):
        """
        Initialize the feature extractor.
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        # Use settings if redis_url is not provided
        self.client = Redis.from_url(redis_url or settings.redis_url)
        
        # Define feature groups
        self.numeric_features = [
            'amount', 'hour', 'day_of_week', 'is_weekend', 
            'transaction_velocity', 'avg_transaction_amount', 
            'amount_vs_average_ratio', 'days_since_first_seen'
        ]
        
        self.categorical_features = [
            'card_provider', 'merchant_id', 'amount_category', 
            'user_segment', 'is_preferred_card', 'is_home_location'
        ]
        
        # Initialize transformers (will be set during fit)
        self.encoder = None
        self.scaler = None
        
    def fit_transform(self, df):
        """
        Fit the transformers and transform the data.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            X: Feature matrix
            y: Target variable (fraud or not)
        """
        logger.info("Fitting and transforming data...")
        
        # Create transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
        
        # Fit the transformer and transform data
        X = df[self.numeric_features + self.categorical_features]
        y = df['is_fraud']
        X_transformed = preprocessor.fit_transform(X)
        
        # Save transformers for later use
        self.scaler = preprocessor.transformers_[0][1]
        self.encoder = preprocessor.transformers_[1][1]
        
        logger.debug(f"Transformed data shape: {X_transformed.shape}")
        return X_transformed, y
    
    def transform(self, df):
        """
        Transform the data using fitted transformers.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Transformed feature matrix
        """
        if self.scaler is None or self.encoder is None:
            raise ValueError("Transformers not fitted yet. Call fit_transform first.")
            
        logger.debug(f"Transforming data with shape: {df.shape}")
        
        # Check for missing features
        missing_features = [f for f in self.numeric_features + self.categorical_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        try:
            # Transform numeric features with scaler
            X_numeric = df[self.numeric_features].values
            X_numeric_scaled = self.scaler.transform(X_numeric)
            
            # Transform categorical features with encoder
            X_categorical = df[self.categorical_features].values
            X_categorical_encoded = self.encoder.transform(X_categorical)
            
            # Combine the transformed features
            X_transformed = np.hstack([X_numeric_scaled, X_categorical_encoded.toarray()])
            
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise e
    
    def get_user_data(self, user_id, current_timestamp=None):
        """
        Get complete user data including profile and history.
        
        Args:
            user_id: User ID to get data for
            current_timestamp: Optional current transaction timestamp
            
        Returns:
            Dictionary with user profile and history data
        """
        # Get user profile
        key = f"{settings.namespace}:user_profile:{user_id}"
        profile = self.client.json().get(key) or {
            "user_id": user_id,
            "first_seen": 0,
            "preferred_card": "UNKNOWN",
            "user_segment": "NEW",
            "home_location": "0,0",
            "registration_date": 0
        }
        
        # Get transaction history (limited to 10 most recent)
        query = Query(f"@user_id:{user_id}").dialect(2)
        results = self.client.ft(settings.namespaced_transactions_index).search(query)
        
        # Convert to list of transactions and sort by timestamp
        transactions = []
        for doc in results.docs:
            txn = json.loads(doc.json)
            transactions.append(txn)
        
        transactions.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        recent_transactions = transactions[:10]
        
        # Calculate metrics if we have a current timestamp
        metrics = {}
        if current_timestamp:
            # Transaction velocity (30-day window)
            thirty_days_ago = current_timestamp - (30 * 24 * 60 * 60)
            recent_txns = [t for t in transactions if t.get('timestamp', 0) >= thirty_days_ago]
            days_interval = min(30, (current_timestamp - thirty_days_ago) / (24 * 60 * 60))
            transaction_velocity = len(recent_txns) / max(1, days_interval)
            
            # Average transaction amount
            amounts = [t.get('amount', 0) for t in recent_txns]
            avg_amount = sum(amounts) / max(1, len(amounts))
            
            metrics = {
                'transaction_velocity': transaction_velocity,
                'avg_transaction_amount': avg_amount,
                'transaction_count_30d': len(recent_txns),
                'days_since_first_seen': (current_timestamp - profile.get('first_seen', 0)) / (24 * 60 * 60)
            }
        
        return {
            'profile': profile,
            'recent_transactions': recent_transactions,
            'metrics': metrics
        }
        
    def is_location_near_home(self, location, home_location, radius_km=10):
        """
        Check if a transaction location is near the user's home location.
        
        Args:
            location: Transaction location (lon,lat)
            home_location: Home location (lon,lat)
            radius_km: Radius in kilometers
            
        Returns:
            True if the location is within the radius of home location
        """
        if not location or not home_location or home_location == "0,0":
            return False
            
        try:
            # Parse locations
            lon1, lat1 = map(float, location.split(','))
            lon2, lat2 = map(float, home_location.split(','))
            
            # Simple distance calculation (approximation)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (dlat**2) + (dlon**2)
            distance = 111.2 * (a**0.5)  # 111.2 km per degree
            
            return distance <= radius_km
        except Exception:
            return False
    
    def extract_model_features_for_transaction(self, transaction):
        """
        Extract and transform features for a single transaction for model prediction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Transformed feature vector for the model
        """
        logger.info(f"Extracting model features for transaction {transaction.get('transaction_id', 'unknown')}")
        
        # Extract features
        features_df = self.extract_features_for_transaction(transaction)
        
        # Transform for model
        model_features = features_df[self.numeric_features + self.categorical_features]
        return self.transform(model_features)
        
    def extract_features_for_transaction(self, transaction):
        """
        Extract all features for a single transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            DataFrame with all features for the transaction
        """
        logger.info(f"Extracting features for transaction {transaction.get('transaction_id', 'unknown')}")
        
        # Ensure timestamp is an integer
        timestamp = transaction.get('timestamp', 0)
        if isinstance(timestamp, str):
            timestamp = int(pd.to_datetime(timestamp).timestamp())
            transaction['timestamp'] = timestamp
            
        # Get user data including profile and history
        user_id = transaction.get('user_id')
        user_data = self.get_user_data(user_id, timestamp)
        user_profile = user_data['profile']
        
        # Extract datetime features
        dt = pd.to_datetime(timestamp, unit='s')
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Amount categorization
        amount = transaction.get('amount', 0)
        if amount <= 100:
            amount_category = 'low'
        elif amount <= 500:
            amount_category = 'medium'
        elif amount <= 1000:
            amount_category = 'high'
        else:
            amount_category = 'very_high'
        
        # Calculate derived features
        metrics = user_data['metrics']
        avg_amount = metrics.get('avg_transaction_amount', 0)
        amount_vs_average_ratio = amount / avg_amount if avg_amount > 0 else 1.0
        
        # Location-based feature
        is_home_location = self.is_location_near_home(
            transaction.get('location'), 
            user_profile.get('home_location', '0,0')
        )
        
        # Card provider feature
        is_preferred_card = 1 if transaction.get('card_provider') == user_profile.get('preferred_card') else 0
        
        # Create feature dictionary
        features = {
            # Transaction features
            'amount': amount,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'card_provider': transaction.get('card_provider', 'UNKNOWN'),
            'merchant_id': transaction.get('merchant_id', 'UNKNOWN'),
            'amount_category': amount_category,
            
            # User metrics
            'transaction_velocity': metrics.get('transaction_velocity', 0),
            'avg_transaction_amount': metrics.get('avg_transaction_amount', 0),
            'amount_vs_average_ratio': amount_vs_average_ratio,
            'days_since_first_seen': metrics.get('days_since_first_seen', 0),
            
            # User profile features
            'user_segment': user_profile.get('user_segment', 'NEW'),
            'is_preferred_card': is_preferred_card,
            'is_home_location': 1 if is_home_location else 0
        }
        
        return pd.DataFrame([features]) 