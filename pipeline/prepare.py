import argparse
import logging

# Add the project root to the path
from pipeline.settings import settings
from pipeline.data_loader.redis_loader import RedisDataLoader
from pipeline.data_loader.data_processor import DataProcessor
from pipeline.model.fraud_model import FraudDetectionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prepare')

def prepare_system(train_model=True, init_redis=True, model_description="Initial model"):
    """
    Prepare the ML inference pipeline
    
    Args:
        train_model: Whether to train and save the model
        init_redis: Whether to initialize Redis with data and indexes
        model_description: Description for the model version
    """
    logger.info("Starting inference pipeline preparation")
    
    # Initialize Redis with data and indexes
    if init_redis:
        logger.info("Initializing Redis with transaction data...")
        redis_loader = RedisDataLoader()
        redis_loader.initialize_redis(str(settings.raw_data_path))
        
    # Process data for model training
    logger.info("Processing transaction data for model...")
    processor = DataProcessor()
    processor.process_data(str(settings.raw_data_path), str(settings.processed_data_path))
    
    # Train and save the model
    if train_model:
        logger.info("Training fraud detection inference model...")
        model = FraudDetectionModel()
        metrics = model.train(str(settings.processed_data_path))
        
        # Print metrics
        logger.info(f"Model metrics: Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Save model to ModelStore
        logger.info("Saving inference model to ModelStore...")
        version = model.save_model(description=model_description)
        logger.info(f"Model saved as version: {version}")
    
    logger.info("Inference pipeline preparation completed")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ML inference pipeline")
    parser.add_argument('--skip-model', action='store_true', help='Skip model training')
    parser.add_argument('--skip-redis', action='store_true', help='Skip Redis initialization')
    parser.add_argument('--description', type=str, default="Initial model", help='Model version description')
    
    args = parser.parse_args()
    
    prepare_system(
        train_model=not args.skip_model,
        init_redis=not args.skip_redis,
        model_description=args.description
    ) 