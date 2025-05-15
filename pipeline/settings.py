from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings using Pydantic.
    Environment variables take precedence over values defined in the class.
    """
    # Redis connection settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: str = Field(default="", description="Redis password")
    
    # Data paths
    raw_data_path: Path = Field(default=Path("data/raw/transactions_200.json"), description="Path to raw data")
    processed_data_path: Path = Field(default=Path("data/processed/transactions.parquet"), description="Path to processed data")
    
    # Index names - using namespace to avoid collisions
    namespace: str = Field(default="inference-pipeline", description="Namespace for Redis keys")
    transactions_index: str = Field(default="transactions_idx", description="Index name for transactions")
    user_profiles_index: str = Field(default="user_profiles_idx", description="Index name for user profiles")
    
    # Stream names
    transaction_stream: str = Field(default="transaction_stream", description="Stream name for transactions")
    fraud_results_stream: str = Field(default="fraud_results_stream", description="Stream name for fraud results")
    
    # Feature engineering
    fraud_threshold: float = Field(default=0.7, description="Threshold for fraud detection")
    fraud_label_ratio: float = Field(default=0.15, description="Ratio of transactions to label as fraud")
    
    # Model parameters
    test_size: float = Field(default=0.2, description="Test size for model training")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    model_name: str = Field(default="fraud-detection-model", description="Name for the model in ModelStore")
    
    # Pydantic v2 Config
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    @property
    def redis_url(self) -> str:
        """Build Redis URL from connection parameters"""
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
    
    @property
    def namespaced_transactions_index(self) -> str:
        """Get namespaced transactions index name"""
        return f"{self.namespace}:{self.transactions_index}"
    
    @property
    def namespaced_user_profiles_index(self) -> str:
        """Get namespaced user profiles index name"""
        return f"{self.namespace}:{self.user_profiles_index}"
    
    @property
    def namespaced_transaction_stream(self) -> str:
        """Get namespaced transaction stream name"""
        return f"{self.namespace}:{self.transaction_stream}"
    
    @property
    def namespaced_fraud_results_stream(self) -> str:
        """Get namespaced fraud results stream name"""
        return f"{self.namespace}:{self.fraud_results_stream}"


# Create a global settings instance
settings = Settings() 