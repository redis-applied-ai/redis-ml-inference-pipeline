import json
import pandas as pd
from datetime import datetime
import sys
import os
import numpy as np
import time

# Import the settings instance
from pipeline.settings import settings
from pipeline.features.feature_extractor import FeatureExtractor

class DataProcessor:
    """
    Class to process transaction data for fraud detection
    """
    @staticmethod
    def load_raw_data(filepath=None):
        """
        Load raw transaction data from JSON file
        
        Args:
            filepath: Path to the JSON file containing transaction data
            
        Returns:
            pandas DataFrame with transaction data
        """
        # Use settings if filepath is not provided
        if filepath is None:
            filepath = settings.raw_data_path
            
        # Load transactions from JSON file
        with open(filepath, 'r') as f:
            transactions = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Convert timestamp to datetime
        df["timestamp"] = df["timestamp"].apply(lambda s: int(pd.to_datetime(s).timestamp()))
        
        # Add location field 
        df['location'] = df.apply(lambda r: f"{r.lon}, {r.lat}", axis=1)
        
        print(f"Loaded {len(df)} transactions")
        return df
    
    @staticmethod
    def enrich_data(df):
        """
        Enrich transaction data with additional features
        
        Args:
            df: pandas DataFrame with transaction data
            
        Returns:
            Enriched pandas DataFrame
        """
        # Add hour of day
        df['hour'] = df['timestamp'].apply(
            lambda ts: datetime.fromtimestamp(ts).hour
        )
        
        # Add day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].apply(
            lambda ts: datetime.fromtimestamp(ts).weekday()
        )
        
        # Add weekend flag
        df['is_weekend'] = df['day_of_week'].apply(
            lambda d: 1 if d >= 5 else 0
        )
        
        # Add amount categories
        df['amount_category'] = pd.cut(
            df['amount'], 
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Add synthetic dynamic features for training data
        df = DataProcessor.add_synthetic_dynamic_features(df)
        
        return df
    
    @staticmethod
    def add_synthetic_dynamic_features(df):
        """
        Add synthetic dynamic features for training data
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with added synthetic features
        """
        # Sort by user_id and timestamp to calculate features in chronological order
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Group by user_id
        user_groups = df.groupby('user_id')
        
        # Initialize new columns
        df['transaction_velocity'] = 0.0
        df['avg_transaction_amount'] = 0.0
        df['amount_vs_average_ratio'] = 1.0
        df['days_since_first_seen'] = 0
        df['is_preferred_card'] = 0
        df['is_home_location'] = 0
        df['user_segment'] = 'NEW'
        
        # Process each user group
        for user_id, user_df in user_groups:
            # Calculate first_seen (first transaction timestamp)
            first_seen = user_df['timestamp'].min()
            
            # Calculate preferred card (most used)
            preferred_card = user_df['card_provider'].value_counts().idxmax()
            
            # Calculate home location (most frequent)
            home_location = user_df['location'].value_counts().idxmax()
            
            # User segment based on total spending
            total_spend = user_df['amount'].sum()
            if total_spend > 5000:
                user_segment = 'PREMIUM'
            elif total_spend > 1000:
                user_segment = 'STANDARD'
            else:
                user_segment = 'BASIC'
            
            # Process each transaction for the user
            for idx, row in user_df.iterrows():
                # Get transactions up to this point
                prev_txns = user_df[user_df['timestamp'] < row['timestamp']]
                
                # Skip first transaction (no history)
                if len(prev_txns) == 0:
                    df.at[idx, 'transaction_velocity'] = 0
                    df.at[idx, 'avg_transaction_amount'] = 0
                    df.at[idx, 'amount_vs_average_ratio'] = 1.0
                    df.at[idx, 'days_since_first_seen'] = 0
                    continue
                
                # Calculate days since first seen
                days_since_first = (row['timestamp'] - first_seen) / (24 * 60 * 60)
                df.at[idx, 'days_since_first_seen'] = days_since_first
                
                # Calculate transaction velocity (transactions per day)
                # Use either last 30 days or all history if < 30 days
                thirty_days_ago = row['timestamp'] - (30 * 24 * 60 * 60)
                recent_txns = prev_txns[prev_txns['timestamp'] > thirty_days_ago]
                
                if len(recent_txns) > 0:
                    oldest_recent_ts = recent_txns['timestamp'].min()
                    days_interval = max(1, (row['timestamp'] - oldest_recent_ts) / (24 * 60 * 60))
                    velocity = len(recent_txns) / days_interval
                else:
                    velocity = 0
                    
                df.at[idx, 'transaction_velocity'] = velocity
                
                # Calculate average amount
                avg_amount = prev_txns['amount'].mean()
                df.at[idx, 'avg_transaction_amount'] = avg_amount
                
                # Calculate amount vs average ratio
                if avg_amount > 0:
                    df.at[idx, 'amount_vs_average_ratio'] = row['amount'] / avg_amount
                
                # Set is_preferred_card
                df.at[idx, 'is_preferred_card'] = 1 if row['card_provider'] == preferred_card else 0
                
                # Set is_home_location (simplified)
                df.at[idx, 'is_home_location'] = 1 if row['location'] == home_location else 0
                
                # Set user segment
                df.at[idx, 'user_segment'] = user_segment
                
        return df
    
    @staticmethod
    def add_fraud_labels(df, fraud_ratio=None, random_state=42):
        """
        Add synthetic fraud labels for training
        
        Args:
            df: pandas DataFrame with transaction data
            fraud_ratio: Ratio of transactions to mark as fraud
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with fraud labels
        """
        # Use settings if fraud_ratio is not provided
        if fraud_ratio is None:
            fraud_ratio = settings.fraud_label_ratio
            
        # Create a copy to avoid modifying the original
        df_with_labels = df.copy()
        
        # Set random seed
        np.random.seed(random_state)
        
        # Add fraud labels (synthetic data)
        n_fraud = int(len(df) * fraud_ratio)
        
        # Base fraud probability on various factors
        # Higher amounts have higher probability of being fraudulent
        fraud_prob = df_with_labels['amount'] / df_with_labels['amount'].max() * 0.4
        
        # Transactions with high amount_vs_average_ratio are more suspicious
        if 'amount_vs_average_ratio' in df_with_labels.columns:
            ratio_factor = df_with_labels['amount_vs_average_ratio'].clip(0, 5) / 5
            fraud_prob += ratio_factor * 0.3
        
        # Non-preferred card usage is more suspicious
        if 'is_preferred_card' in df_with_labels.columns:
            fraud_prob += (1 - df_with_labels['is_preferred_card']) * 0.2
        
        # Transactions away from home location are more suspicious
        if 'is_home_location' in df_with_labels.columns:
            fraud_prob += (1 - df_with_labels['is_home_location']) * 0.1
        
        # Add random noise 
        fraud_prob = fraud_prob + np.random.normal(0, 0.1, size=len(fraud_prob))
        
        # Normalize to [0, 1]
        fraud_prob = (fraud_prob - fraud_prob.min()) / (fraud_prob.max() - fraud_prob.min())
        
        # Sort by fraud probability and select top n_fraud
        fraud_indices = fraud_prob.sort_values(ascending=False)[:n_fraud].index
        
        # Initialize fraud label
        df_with_labels['is_fraud'] = 0
        
        # Mark fraud transactions
        df_with_labels.loc[fraud_indices, 'is_fraud'] = 1
        
        print(f"Added fraud labels: {df_with_labels['is_fraud'].sum()} fraud transactions ({fraud_ratio:.1%})")
        
        return df_with_labels
    
    @staticmethod
    def save_processed_data(df, filepath=None):
        """
        Save processed data to parquet file
        
        Args:
            df: pandas DataFrame with processed data
            filepath: Path to save the processed data
        """
        # Use settings if filepath is not provided
        if filepath is None:
            filepath = settings.processed_data_path
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to parquet
        df.to_parquet(filepath, index=False)
        print(f"Saved processed data to {filepath}")
        
    @staticmethod
    def process_data(raw_filepath=None, processed_filepath=None):
        """
        Process raw transaction data and save to processed file
        
        Args:
            raw_filepath: Path to the raw data file
            processed_filepath: Path to save the processed data
            
        Returns:
            Processed DataFrame
        """
        # Use settings if filepaths are not provided
        if raw_filepath is None:
            raw_filepath = settings.raw_data_path
        if processed_filepath is None:
            processed_filepath = settings.processed_data_path
            
        # Load raw data
        df = DataProcessor.load_raw_data(raw_filepath)
        
        # Enrich data
        df = DataProcessor.enrich_data(df)
        
        # Add fraud labels
        df = DataProcessor.add_fraud_labels(df)
        
        # Save processed data
        DataProcessor.save_processed_data(df, processed_filepath)
        
        return df 