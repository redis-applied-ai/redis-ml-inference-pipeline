#!/usr/bin/env python
import argparse
import logging
from redis import Redis

from pipeline.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cleanup')

class RedisCleanup:
    """
    Utility to clean up Redis data for the inference pipeline
    """
    def __init__(self, redis_url=settings.redis_url):
        """Initialize the cleanup utility"""
        self.redis_client = Redis.from_url(redis_url)
        self.namespace = settings.namespace
        
    def clear_streams(self):
        """Clear all entries from transaction and fraud results streams"""
        transaction_stream = settings.namespaced_transaction_stream
        results_stream = settings.namespaced_fraud_results_stream
        
        # Delete and recreate transaction stream
        try:
            self.redis_client.delete(transaction_stream)
            # Add a dummy message to create the stream
            self.redis_client.xadd(transaction_stream, {'init': 'reset'})
            logger.info(f"Reset transaction stream: {transaction_stream}")
        except Exception as e:
            logger.error(f"Error resetting transaction stream: {str(e)}")
            
        # Delete and recreate results stream
        try:
            self.redis_client.delete(results_stream)
            # Add a dummy message to create the stream
            self.redis_client.xadd(results_stream, {'init': 'reset'})
            logger.info(f"Reset results stream: {results_stream}")
        except Exception as e:
            logger.error(f"Error resetting results stream: {str(e)}")
    
    def clear_predictions(self):
        """Clear all fraud prediction results"""
        # Get keys matching the prediction pattern
        pattern = f"{self.namespace}:fraud_prediction:*"
        keys = self.redis_client.keys(pattern)
        
        if not keys:
            logger.info(f"No prediction results found matching pattern: {pattern}")
            return
            
        # Delete all matching keys
        count = len(keys)
        if count > 0:
            self.redis_client.delete(*keys)
            logger.info(f"Deleted {count} prediction results")
    
    def clear_all(self):
        """Clear all Redis data related to the inference pipeline"""
        self.clear_streams()
        self.clear_predictions()
        logger.info(f"Cleaned up all Redis data for namespace: {self.namespace}")

def main():
    """CLI entry point for cleanup command"""
    parser = argparse.ArgumentParser(description="Clean up Redis data for the inference pipeline")
    parser.add_argument('--all', action='store_true', help='Clear all data (streams and predictions)')
    parser.add_argument('--streams', action='store_true', help='Clear only streams')
    parser.add_argument('--predictions', action='store_true', help='Clear only prediction results')
    
    args = parser.parse_args()
    
    cleanup = RedisCleanup()
    
    # Default to --all if no specific options are provided
    if not (args.streams or args.predictions):
        args.all = True
        
    # Perform cleanup based on options
    if args.all:
        cleanup.clear_all()
    else:
        if args.streams:
            cleanup.clear_streams()
        if args.predictions:
            cleanup.clear_predictions()
    
    print("Cleanup completed successfully.")

if __name__ == "__main__":
    main() 