"""
Unified inference worker for the ML fraud detection pipeline.
"""
import json
import uuid
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple
from time import perf_counter_ns, time
from concurrent.futures import ThreadPoolExecutor

# Import from utilities
from pipeline.settings import settings
from pipeline.model.fraud_model import FraudDetectionModel
from pipeline.utils.redis_utils import get_redis_client, setup_redis_streams
from pipeline.stats.manager import StatisticsManager
from pipeline.stats.reporting import print_worker_stats

# Configure logging
logger = logging.getLogger(__name__)

class InferenceWorker:
    """
    Unified worker service that processes transactions individually or in batches
    with configurable parallelism, consuming from Redis streams and publishing
    results back to Redis.
    """
    def __init__(
        self, 
        redis_url: Optional[str] = None,
        model_version: Optional[str] = None,
        worker_id: Optional[Union[int, str]] = None,
        consumer_group: str = "inference_workers",
        batch_size: int = 10,
        parallel_mode: bool = False,
        threads: int = 1
    ):
        """
        Initialize the worker
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
            model_version: Specific model version to load, or None for latest
            worker_id: Optional identifier for this worker
            consumer_group: Redis consumer group name
            batch_size: Number of messages to process in each batch
            parallel_mode: Whether to process transactions in parallel within the worker
            threads: Number of threads to use for parallel processing (only if parallel_mode=True)
        """
        # Initialize Redis client
        self.redis_client = get_redis_client(redis_url)
        self.worker_id = str(worker_id or uuid.uuid4().hex[:8])
        self.consumer_group = consumer_group
        self.consumer_name = f"worker-{self.worker_id}"
        
        # Configure processing parameters
        self.batch_size = batch_size
        self.parallel_mode = parallel_mode
        self.threads = max(1, threads)
        
        # Set up thread pool if in parallel mode
        self._executor = ThreadPoolExecutor(max_workers=self.threads) if parallel_mode else None
        
        # Load model from ModelStore
        self._load_model(model_version)
        
        # Set worker state
        self.running = False
        self.start_time = time()
        
        # Statistics manager (will be set per run)
        self.stats_manager = None
        self.current_run_id = None
        
        # Ensure stream and consumer group exist
        setup_redis_streams(self.redis_client, self.consumer_group, self.worker_id)
        
        # Log initialization information
        if self.parallel_mode:
            logger.info(f"Worker {self.worker_id} initialized in parallel mode with {self.threads} threads, batch size {self.batch_size}")
        else:
            logger.info(f"Worker {self.worker_id} initialized in sequential mode, batch size {self.batch_size}")
    
    def _load_model(self, model_version: Optional[str] = None) -> None:
        """
        Load the fraud detection model
        
        Args:
            model_version: Specific model version to load, or None for latest
        """
        if model_version:
            logger.info(f"Worker {self.worker_id}: Loading model version: {model_version}")
            self.model = FraudDetectionModel.load_specific_version(
                version=model_version,
                redis_url=settings.redis_url
            )
        else:
            logger.info(f"Worker {self.worker_id}: Loading latest model version")
            self.model = FraudDetectionModel.load_latest_model(
                redis_url=settings.redis_url
            )
    
    def _get_stats_manager(self, run_id: str) -> StatisticsManager:
        """
        Get or create a StatisticsManager for the given run ID
        
        Args:
            run_id: Run ID to get stats manager for
            
        Returns:
            StatisticsManager instance
        """
        # Create new stats manager if we don't have one or if run ID changed
        if self.stats_manager is None or self.current_run_id != run_id:
            self.stats_manager = StatisticsManager(run_id=run_id, worker_id=self.worker_id)
            self.stats_manager.set_worker_config(
                worker_id=self.worker_id,
                batch_size=self.batch_size,
                parallel_mode=self.parallel_mode,
                threads=self.threads
            )
            self.current_run_id = run_id
            
        return self.stats_manager
    
    def _process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single transaction through the ML inference model
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Inference result
        """
        # Make prediction through inference
        prediction = self.model.predict(transaction)
        
        # Add basic metadata
        prediction['worker_id'] = str(self.worker_id)
        prediction['processed_at'] = datetime.now().isoformat()
        
        # Preserve run_id from transaction if it exists
        if 'run_id' in transaction:
            prediction['run_id'] = transaction['run_id']
        
        return prediction
    
    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single transaction with timing and tracking
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Inference result with metadata
        """
        # Start latency measurement
        start_time_ns = perf_counter_ns()
        
        # Process the transaction
        prediction = self._process_transaction(transaction)
        
        # Calculate latency (in milliseconds)
        end_time_ns = perf_counter_ns()
        latency_ms = round((end_time_ns - start_time_ns) / 1e6, 2)  # Convert ns to ms with 2 decimal places
        
        # Add latency information
        prediction['latency_ms'] = latency_ms
        prediction['batch_processed'] = False
        
        # Get run_id, if available
        run_id = transaction.get('run_id', 'unknown')
        prediction['run_id'] = run_id
        
        # Get stats manager for this run
        stats_manager = self._get_stats_manager(run_id)
        
        # Track with stats manager
        stats_manager.record_transaction(
            latency_ms=latency_ms,
            is_fraud=bool(prediction['is_fraud']),
            transaction_id=prediction['transaction_id']
        )
        
        # Store complete prediction in Redis
        self.model.store_prediction(prediction)
        
        # Publish to results stream
        self._publish_result(prediction, latency_ms)
        
        # Log important events
        is_fraud = prediction['is_fraud'] == 1
        stats = stats_manager.get_worker_stats()
        if is_fraud or stats.transactions % 25 == 0:
            fraud_status = "FRAUD" if is_fraud else "legitimate"
            logger.info(
                f"Worker {self.worker_id}: TX {prediction['transaction_id']} → {fraud_status} "
                f"(prob: {prediction['fraud_probability']:.2f}, latency: {latency_ms}ms, run: {run_id})"
            )
        
        return prediction
        
    def process_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of transactions, either in parallel or sequentially
        
        Args:
            transactions: List of transaction data dictionaries
            
        Returns:
            List of inference results
        """
        if not transactions:
            return []
            
        batch_start_time = perf_counter_ns()
        predictions = []
        
        # Group transactions by run_id - use first transaction's run_id as primary
        primary_run_id = transactions[0].get('run_id', 'unknown')
        
        # Get stats manager for this run
        stats_manager = self._get_stats_manager(primary_run_id)
        
        if self.parallel_mode and self._executor is not None:
            # Process in parallel using thread pool
            futures = []
            for txn in transactions:
                futures.append(self._executor.submit(self._process_transaction, txn))
            
            # Collect results
            for future in futures:
                try:
                    prediction = future.result()
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Error processing transaction in parallel: {str(e)}")
        else:
            # Process sequentially
            for txn in transactions:
                try:
                    prediction = self._process_transaction(txn)
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Error processing transaction: {str(e)}")
        
        # Calculate batch metrics
        batch_end_time = perf_counter_ns()
        batch_time_ms = (batch_end_time - batch_start_time) / 1e6
        avg_latency_ms = batch_time_ms / len(transactions) if transactions else 0
        
        # Process each prediction result
        for prediction in predictions:
            # Add batch metadata
            prediction['latency_ms'] = avg_latency_ms
            prediction['batch_processed'] = True
            prediction['batch_size'] = len(transactions)
            
            # Get run_id from prediction or use primary
            run_id = prediction.get('run_id', primary_run_id)
            
            # Track in stats manager
            stats_manager.record_transaction(
                latency_ms=avg_latency_ms,
                is_fraud=bool(prediction['is_fraud']),
                transaction_id=prediction['transaction_id'],
                batch_processed=True,
                batch_size=len(transactions)
            )
            
            # Store prediction in Redis
            self.model.store_prediction(prediction)
            
            # Publish to results stream
            self._publish_result(prediction, avg_latency_ms)
        
        # Log batch results
        fraud_count = sum(1 for p in predictions if p['is_fraud'])
        processing_mode = "parallel" if self.parallel_mode else "sequential"
        logger.info(
            f"Worker {self.worker_id}: {processing_mode} processed {len(predictions)} transactions "
            f"({fraud_count} fraud) in {batch_time_ms:.2f}ms (avg: {avg_latency_ms:.2f}ms per txn)"
            f" - run: {primary_run_id}"
        )
        
        return predictions
    
    def _publish_result(self, prediction: Dict[str, Any], latency_ms: float) -> str:
        """
        Publish a prediction result to the Redis stream
        
        Args:
            prediction: The prediction result
            latency_ms: Processing latency in milliseconds
            
        Returns:
            Stream ID of the published message
        """
        stream_id = self.redis_client.xadd(
            settings.namespaced_fraud_results_stream,
            {
                'data': json.dumps(prediction),
                'transaction_id': prediction['transaction_id'],
                'is_fraud': str(prediction['is_fraud']),
                'fraud_probability': str(prediction['fraud_probability']),
                'user_id': prediction['user_id'],
                'worker_id': str(self.worker_id),
                'latency_ms': str(latency_ms),
                'run_id': prediction.get('run_id', 'unknown')
            }
        )
        return stream_id

    def run(self, timeout_ms: int = 5000) -> None:
        """
        Run the inference service, processing transactions from the Redis stream
        
        Args:
            timeout_ms: Timeout in milliseconds for blocking read from stream
        """
        self.running = True
        logger.info(f"Starting inference worker {self.consumer_name} (mode: {'parallel' if self.parallel_mode else 'sequential'}, batch size: {self.batch_size})")
        
        # Initialize performance tracking
        self.start_time = time()
        transaction_stream = settings.namespaced_transaction_stream
        
        try:
            while self.running:
                # Read new messages from stream
                streams = {transaction_stream: '>'}  # '>' means all new messages
                
                response = self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams=streams,
                    count=self.batch_size,
                    block=timeout_ms
                )
                
                if not response:  # No new messages
                    continue
                    
                # Process messages
                stream_name, messages = response[0]  # Only one stream in our query
                
                if not messages:
                    continue
                
                # Extract transactions from messages
                transactions = []
                message_ids = []
                
                for message_id, message in messages:
                    try:
                        if b'data' in message:
                            transaction = json.loads(message[b'data'].decode())
                            transactions.append(transaction)
                            message_ids.append(message_id)
                    except Exception as e:
                        logger.error(f"Worker {self.worker_id}: Error parsing message {message_id}: {str(e)}")
                        # Acknowledge message even if we couldn't parse it
                        self.redis_client.xack(transaction_stream, self.consumer_group, message_id)
                
                try:
                    # Process as batch or individual based on number of transactions
                    if len(transactions) > 1:
                        self.process_batch(transactions)
                    elif len(transactions) == 1:
                        self.process_transaction(transactions[0])
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error processing transactions: {str(e)}")
                
                # Acknowledge all processed messages
                for message_id in message_ids:
                    self.redis_client.xack(transaction_stream, self.consumer_group, message_id)
                    
                # Periodically save worker stats
                if self.stats_manager:
                    self.stats_manager.save_worker_stats()
                    
        except KeyboardInterrupt:
            logger.info(f"Worker {self.worker_id}: Stopping inference worker due to keyboard interrupt")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error in worker loop: {str(e)}")
        finally:
            # Cleanup
            self.stop()
            
    def stop(self) -> None:
        """Stop the inference worker and clean up."""
        self.running = False
        if self._executor:
            self._executor.shutdown()
            
        # Save final stats
        if self.stats_manager:
            worker_stats = self.stats_manager.get_worker_stats()
            self.stats_manager.save_worker_stats()
            print_worker_stats(worker_stats)
            
        logger.info(f"Worker {self.worker_id}: Inference worker stopped")
    
    def log_performance_stats(self) -> Dict[str, Any]:
        """
        Log performance statistics for the current session
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.stats_manager:
            return {}
            
        worker_stats = self.stats_manager.get_worker_stats()
        self.stats_manager.save_worker_stats()
        
        # Use formatted output from stats models
        print_worker_stats(worker_stats)
        
        # Return core metrics as dictionary
        return {
            'runtime_seconds': worker_stats.processing_time,
            'total_transactions': worker_stats.transactions,
            'throughput_tps': worker_stats.throughput,
            'avg_latency_ms': worker_stats.latency.avg_ms,
            'fraud_count': worker_stats.fraud_count,
            'fraud_ratio': worker_stats.fraud_ratio,
            'processing_mode': 'parallel' if self.parallel_mode else 'sequential',
            'batch_size': self.batch_size,
            'threads': self.threads if self.parallel_mode else 1
        }

# Run worker if script is executed directly
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Run ML inference worker")
    parser.add_argument('--model-version', type=str, help='Specific model version to use')
    parser.add_argument('--worker-id', type=str, help='Worker identifier (optional)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing transactions')
    parser.add_argument('--parallel', action='store_true', help='Process in parallel mode')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use in parallel mode')
    parser.add_argument('--consumer-group', type=str, default="inference_workers", help='Consumer group name')
    parser.add_argument('--quiet', action='store_true', help='Reduce logging output')
    
    args = parser.parse_args()
    
    # Set quieter logging if requested
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    worker = InferenceWorker(
        model_version=args.model_version, 
        worker_id=args.worker_id,
        batch_size=args.batch_size,
        parallel_mode=args.parallel,
        threads=args.threads,
        consumer_group=args.consumer_group
    )
    worker.run() 