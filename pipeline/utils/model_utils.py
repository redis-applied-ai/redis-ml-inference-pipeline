"""
Model serialization utilities for the ML Inference Pipeline.
"""

import io
import pickle
import base64
import logging
import joblib
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

def serialize_model(model: Any) -> bytes:
    """
    Serialize a model to bytes using joblib
    
    Args:
        model: The model to serialize
        
    Returns:
        Serialized model as bytes
    """
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()

def deserialize_model(serialized_model: bytes) -> Any:
    """
    Deserialize a model from bytes
    
    Args:
        serialized_model: The serialized model bytes
        
    Returns:
        Deserialized model object
    """
    buffer = io.BytesIO(serialized_model)
    return joblib.load(buffer)

def serialize_transformer(transformer: Any) -> Optional[str]:
    """
    Serialize a transformer (scaler, encoder, etc.) to a base64 string
    
    Args:
        transformer: The transformer object to serialize
        
    Returns:
        Base64 encoded string of the serialized transformer or None if error
    """
    if transformer is None:
        logger.info("Transformer is None, skipping serialization")
        return None
        
    try:
        buffer = io.BytesIO()
        pickle.dump(transformer, buffer)
        serialized = base64.b64encode(buffer.getvalue()).decode('utf-8')
        logger.info(f"Successfully serialized transformer of type: {type(transformer).__name__}")
        return serialized
    except Exception as e:
        logger.error(f"Error serializing transformer: {str(e)}")
        return None

def deserialize_transformer(serialized_transformer: Optional[str]) -> Any:
    """
    Deserialize a transformer from a base64 string
    
    Args:
        serialized_transformer: Base64 encoded string of the serialized transformer
        
    Returns:
        Deserialized transformer object or None if error
    """
    if serialized_transformer is None:
        logger.warning("Serialized transformer is None, cannot deserialize")
        return None
        
    try:
        binary_data = base64.b64decode(serialized_transformer)
        buffer = io.BytesIO(binary_data)
        transformer = pickle.load(buffer)
        logger.info(f"Successfully deserialized transformer of type: {type(transformer).__name__}")
        return transformer
    except Exception as e:
        logger.error(f"Error deserializing transformer: {str(e)}")
        return None

def create_model_bundle(model: Any, feature_extractor: Any) -> Dict[str, Any]:
    """
    Create a bundle with model and feature extractor for storage
    
    Args:
        model: The trained model
        feature_extractor: Feature extractor instance with fitted transformers
        
    Returns:
        Dictionary containing serialized model and feature extraction components
    """
    # Serialize the model
    serialized_model = serialize_model(model)
    
    # Check if the transformers are fitted
    scaler_fitted = feature_extractor.scaler is not None
    encoder_fitted = feature_extractor.encoder is not None
    logger.info(f"Creating model bundle with fitted transformers: Scaler={scaler_fitted}, Encoder={encoder_fitted}")
    
    # Serialize the transformers
    serialized_scaler = serialize_transformer(feature_extractor.scaler)
    serialized_encoder = serialize_transformer(feature_extractor.encoder)
    
    # Create a model bundle
    return {
        'model_bytes': serialized_model,
        'numeric_features': feature_extractor.numeric_features,
        'categorical_features': feature_extractor.categorical_features,
        'serialized_scaler': serialized_scaler,
        'serialized_encoder': serialized_encoder
    } 