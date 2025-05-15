"""
Redis utility functions for the ML Inference Pipeline.
"""

import logging
from typing import Optional
from redis import Redis
from redis.exceptions import ResponseError

from pipeline.settings import settings

logger = logging.getLogger(__name__)

def get_redis_client(redis_url: Optional[str] = None) -> Redis:
    """
    Create a Redis client using the provided URL or settings.
    
    Args:
        redis_url: Optional Redis connection URL (defaults to settings)
        
    Returns:
        Redis client instance
    """
    if redis_url is None:
        redis_url = settings.redis_url
    
    return Redis.from_url(redis_url)

def setup_redis_streams(client: Redis, consumer_group: str, worker_id: str) -> None:
    """
    Setup Redis streams and consumer groups
    
    Args:
        client: Redis client
        consumer_group: Name of the consumer group
        worker_id: Worker identifier for logging
    """
    transaction_stream = settings.namespaced_transaction_stream
    results_stream = settings.namespaced_fraud_results_stream
    
    # Create transaction stream if it doesn't exist
    try:
        client.xgroup_create(
            transaction_stream, 
            consumer_group, 
            id='0', 
            mkstream=True
        )
        logger.info(f"Worker {worker_id}: Created consumer group {consumer_group} on stream {transaction_stream}")
    except ResponseError as e:
        if 'BUSYGROUP' in str(e):
            logger.info(f"Worker {worker_id}: Consumer group {consumer_group} already exists")
        else:
            raise
            
    # Create results stream if it doesn't exist
    try:
        # Just add a dummy message to create the stream
        client.xadd(results_stream, {'init': 'true'})
        logger.info(f"Worker {worker_id}: Created stream {results_stream}")
    except:
        logger.info(f"Worker {worker_id}: Stream {results_stream} already exists") 