"""
Fraud detection model implementation for the ML inference pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from redis import Redis
from redis_model_store import ModelStore

# Import from utilities
from pipeline.settings import settings
from pipeline.features.feature_extractor import FeatureExtractor
from pipeline.utils.redis_utils import get_redis_client
from pipeline.utils.model_utils import (
    serialize_model, 
    deserialize_model, 
    serialize_transformer, 
    deserialize_transformer,
    create_model_bundle
)

# Configure logging
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    """
    Fraud detection model class for training and inference.
    """
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the fraud detection model.
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=settings.random_state,
            n_jobs=1  # Single thread to avoid pickle issues
        )
        
        # Setup feature extraction
        self.feature_extractor = FeatureExtractor(redis_url=redis_url)
        
        # Initialize Redis client and ModelStore
        self.redis_client = get_redis_client(redis_url)
        self.model_store = ModelStore(self.redis_client)
        self.model_name = settings.model_name
        
    def train(self, data_path: str = str(settings.processed_data_path)) -> Dict[str, Any]:
        """
        Train the fraud detection model.
        
        Args:
            data_path: Path to the processed data file
            
        Returns:
            Dictionary with training metrics
        """
        # Load processed data
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} transactions for training")
        
        # Extract features
        X, y = self.feature_extractor.fit_transform(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=settings.test_size, 
            random_state=settings.random_state, 
            stratify=y
        )
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model trained. Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
        return metrics
    
    def save_model(self, description: str = "Fraud detection model") -> str:
        """
        Save the trained model using redis-model-store.
        
        Args:
            description: Description of the model version
            
        Returns:
            Version ID of the saved model
        """
        # Create model bundle with serialized components
        model_bundle = create_model_bundle(self.model, self.feature_extractor)
        
        # Save model with version tracking in ModelStore
        try:
            version = self.model_store.save_model(
                model_bundle,
                name=self.model_name,
                description=description
            )
            
            logger.info(f"Model saved to ModelStore with version: {version}")
            return version
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            # Return a dummy version for now - this would be improved in a real system
            return "error-saving-model"
    
    @classmethod
    def load_latest_model(cls, redis_url: Optional[str] = None) -> 'FraudDetectionModel':
        """
        Load the latest model version from ModelStore.
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
            
        Returns:
            Loaded FraudDetectionModel instance
        """
        # Create a new instance
        model_instance = cls(redis_url=redis_url)
        
        try:
            # Get the model store
            model_store = ModelStore(model_instance.redis_client)
            
            # Load the latest model version
            model_bundle = model_store.load_model(model_instance.model_name)
            
            # Deserialize model
            model_instance.model = deserialize_model(model_bundle['model_bytes'])
            
            # Restore feature extractor components
            model_instance.feature_extractor.numeric_features = model_bundle['numeric_features']
            model_instance.feature_extractor.categorical_features = model_bundle['categorical_features']
            
            # Check if serialized transformers exist in the bundle
            has_scaler = 'serialized_scaler' in model_bundle and model_bundle['serialized_scaler'] is not None
            has_encoder = 'serialized_encoder' in model_bundle and model_bundle['serialized_encoder'] is not None
            logger.info(f"Loading model with serialized transformers: Scaler={has_scaler}, Encoder={has_encoder}")
            
            # Deserialize and restore transformers
            model_instance.feature_extractor.scaler = deserialize_transformer(
                model_bundle.get('serialized_scaler')
            )
            model_instance.feature_extractor.encoder = deserialize_transformer(
                model_bundle.get('serialized_encoder')
            )
            
            # Verify if transformers were successfully deserialized
            scaler_loaded = model_instance.feature_extractor.scaler is not None
            encoder_loaded = model_instance.feature_extractor.encoder is not None
            logger.info(f"Transformers loaded successfully: Scaler={scaler_loaded}, Encoder={encoder_loaded}")
            
            logger.info(f"Latest model loaded from ModelStore")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Just return the instance with default model
            
        return model_instance
    
    @classmethod
    def load_specific_version(
        cls, 
        version: str, 
        redis_url: Optional[str] = None
    ) -> 'FraudDetectionModel':
        """
        Load a specific model version from ModelStore.
        
        Args:
            version: Version ID to load
            redis_url: Redis connection URL (defaults to settings)
            
        Returns:
            Loaded FraudDetectionModel instance
        """
        # Create a new instance
        model_instance = cls(redis_url=redis_url)
        
        try:
            # Get the model store
            model_store = ModelStore(model_instance.redis_client)
            
            # Load the specific model version
            model_bundle = model_store.load_model(
                model_instance.model_name, 
                version=version
            )
            
            # Deserialize model
            model_instance.model = deserialize_model(model_bundle['model_bytes'])
            
            # Restore feature extractor components
            model_instance.feature_extractor.numeric_features = model_bundle['numeric_features']
            model_instance.feature_extractor.categorical_features = model_bundle['categorical_features']
            
            # Deserialize and restore transformers
            model_instance.feature_extractor.scaler = deserialize_transformer(
                model_bundle.get('serialized_scaler')
            )
            model_instance.feature_extractor.encoder = deserialize_transformer(
                model_bundle.get('serialized_encoder')
            )
            
            # Verify if transformers were successfully deserialized
            scaler_loaded = model_instance.feature_extractor.scaler is not None
            encoder_loaded = model_instance.feature_extractor.encoder is not None
            logger.info(f"Transformers loaded successfully: Scaler={scaler_loaded}, Encoder={encoder_loaded}")
            
            logger.info(f"Model version {version} loaded from ModelStore")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Just return the instance with default model
            
        return model_instance
    
    def list_versions(self) -> list:
        """
        List all available model versions.
        
        Returns:
            List of version information
        """
        try:
            versions = self.model_store.get_all_versions(self.model_name)
            return versions
        except Exception as e:
            logger.error(f"Error listing model versions: {str(e)}")
            return []
    
    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on a transaction.
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Dictionary with inference results
        """
        # Check if transformers are fitted before extracting features
        if self.feature_extractor.scaler is None or self.feature_extractor.encoder is None:
            logger.error("Transformers are not fitted, cannot make predictions")
            raise ValueError("Transformers (scaler and encoder) are not properly loaded or fitted")
            
        # Extract features
        X = self.feature_extractor.extract_model_features_for_transaction(transaction)
        
        # Run inference with model
        fraud_proba = self.model.predict_proba(X)[0, 1]
        is_fraud = 1 if fraud_proba >= settings.fraud_threshold else 0
        
        # Create a copy of the transaction to avoid modifying the input
        prediction = transaction.copy()
        
        # Add fraud detection results
        prediction.update({
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'fraud_probability': float(fraud_proba),
            'is_fraud': is_fraud,
            'timestamp': transaction.get('timestamp'),
            'amount': transaction.get('amount'),
            'user_id': transaction.get('user_id')
        })
        
        return prediction
        
    def store_prediction(self, prediction: Dict[str, Any]) -> str:
        """
        Store inference result in Redis.
        
        Args:
            prediction: Inference result dictionary
            
        Returns:
            Redis key where inference result was stored
        """
        # Create Redis key with namespace
        key = f"{settings.namespace}:fraud_prediction:{prediction['transaction_id']}"
        
        # Store as JSON
        self.redis_client.json().set(key, '$', prediction)
        
        return key 