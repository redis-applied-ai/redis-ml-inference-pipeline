"""
Unified transaction generator for the ML inference pipeline.

This module provides flexible transaction generation with different traffic patterns,
scaling capabilities, and realistic fraud signals.
"""
import json
import random
import logging
import uuid
import time
import threading
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
from redis import Redis

from pipeline.settings import settings
from pipeline.utils.redis_utils import get_redis_client
from pipeline.stats.manager import StatisticsManager
from pipeline.stats.reporting import print_run_summary

logger = logging.getLogger(__name__)

class TransactionGenerator:
    """
    Generates synthetic transactions with configurable patterns for load testing.
    Supports horizontal scaling and parallel generation.
    """
    def __init__(
        self, 
        redis_url: Optional[str] = None,
        seed: Optional[int] = None,
        run_id: Optional[str] = None
    ):
        """
        Initialize the transaction generator.
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
            seed: Random seed for reproducibility
            run_id: Optional run ID for this generator session (if None, a new one is generated)
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Initialize Redis client
        self.redis_client = get_redis_client(redis_url)
        
        # Generate or set run ID
        self.run_id = run_id or self._generate_run_id()
        logger.info(f"Transaction generator initialized with Run ID: {self.run_id}")
        
        # Initialize statistics manager
        self.stats_manager = StatisticsManager(run_id=self.run_id)
        
        # Initialize transaction generation parameters with default values
        self._initialize_generation_parameters()
        
        # Generation state
        self.running = False
        self.transaction_count = 0
        self.fraud_count = 0
        self.start_time = None
        self.worker_threads = []
        
    def _generate_run_id(self) -> str:
        """
        Generate a unique run ID for this test session.
        
        Returns:
            Unique run ID string
        """
        # Format: 'run-YYYYMMDD-HHMMSS-XXXX' where XXXX is a short random hex
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = uuid.uuid4().hex[:4]
        return f"run-{timestamp}-{random_suffix}"
        
    def get_run_id(self) -> str:
        """
        Get the current run ID.
        
        Returns:
            The run ID for this generator session
        """
        return self.run_id
    
    def _initialize_generation_parameters(self) -> None:
        """Set up default parameters for transaction generation."""
        # Amount distribution parameters
        self.amount_mean = 100.0
        self.amount_std = 50.0
        
        # Define merchant IDs
        self.merchant_ids = [f"MERCH{i}" for i in range(20)]
        
        # Define user IDs
        self.user_ids = [f"USER{i}" for i in range(100)]
        
        # Define card providers
        self.card_providers = ["VISA", "MASTERCARD", "AMEX", "DISCOVER"]
        
        # Define location centers for geo-variation
        # Format: (longitude, latitude) - some major cities
        self.location_centers = [
            (0, 0),                    # Default origin
            (-74.0059, 40.7128),       # New York
            (-0.1278, 51.5074),        # London
            (139.6917, 35.6895),       # Tokyo
            (2.3522, 48.8566),         # Paris
            (103.8198, 1.3521)         # Singapore
        ]
    
    def generate_transaction(self, 
                            inject_fraud: bool = False, 
                            transaction_time: Optional[datetime] = None,
                            user_id: Optional[str] = None,
                            merchant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a synthetic transaction.
        
        Args:
            inject_fraud: Whether to inject fraud signals
            transaction_time: Optional specific time for the transaction
            user_id: Optional specific user ID
            merchant_id: Optional specific merchant ID
            
        Returns:
            A synthetic transaction
        """
        # Generate a new transaction ID
        transaction_id = str(uuid.uuid4())
        
        # Use provided time or current time
        if transaction_time:
            timestamp = int(transaction_time.timestamp())
        else:
            timestamp = int(datetime.now().timestamp())
        
        # Generate amount (potentially fraudulent if requested)
        if inject_fraud:
            # Fraudulent transactions have higher amounts
            amount = random.uniform(self.amount_mean * 3, self.amount_mean * 10)
        else:
            # Normal transactions follow the distribution
            amount = max(0.01, np.random.normal(self.amount_mean, self.amount_std))
            
        # Round amount to 2 decimal places
        amount = round(amount, 2)
        
        # Select user and merchant (or use provided ones)
        selected_user_id = user_id or random.choice(self.user_ids)
        selected_merchant_id = merchant_id or random.choice(self.merchant_ids)
        card_provider = random.choice(self.card_providers)
        
        # Generate location (potentially fraudulent if requested)
        if inject_fraud:
            # Create location far from normal locations for fraud simulation
            base_lon, base_lat = random.choice(self.location_centers)
            lon = base_lon + random.uniform(-50, 50)
            lat = base_lat + random.uniform(-30, 30)
            location = f"{lon},{lat}"
        else:
            # Normal location with small variations
            base_lon, base_lat = random.choice(self.location_centers)
            lon = base_lon + random.uniform(-1, 1)
            lat = base_lat + random.uniform(-0.5, 0.5)
            location = f"{lon},{lat}"
            
        # Create transaction - convert booleans to integers for JSON serialization
        transaction = {
            'transaction_id': transaction_id,
            'user_id': selected_user_id,
            'merchant_id': selected_merchant_id,
            'amount': amount,
            'currency': 'USD',
            'timestamp': timestamp,
            'card_provider': card_provider,
            'location': location,
            'run_id': self.run_id,  # Add run_id to every transaction
            'is_fraud': 1 if inject_fraud else 0,  # Convert boolean to int for compatibility
            'synthetic': 1  # Convert boolean to int for compatibility
        }
        
        # Add optional fields
        if random.random() < 0.7:  # 70% chance to have device info
            transaction['device_info'] = random.choice(['web', 'mobile-ios', 'mobile-android', 'pos'])
            
        if random.random() < 0.5:  # 50% chance to have category
            transaction['category'] = random.choice([
                'retail', 'dining', 'travel', 'entertainment', 
                'groceries', 'fuel', 'healthcare', 'utilities'
            ])
        
        return transaction
        
    def send_transaction(self, transaction: Dict[str, Any]) -> str:
        """
        Send a transaction to the Redis stream.
        
        Args:
            transaction: Transaction data to send
            
        Returns:
            Stream ID of the published message
        """
        try:
            # Convert transaction to JSON
            json_data = json.dumps(transaction)
            
            # Send to Redis stream
            stream_id = self.redis_client.xadd(
                settings.namespaced_transaction_stream,
                {'data': json_data}
            )
            
            # Convert integer is_fraud back to boolean for stats tracking
            is_fraud = bool(transaction.get('is_fraud', 0))
            
            # Record transaction in statistics manager
            self.stats_manager.record_transaction(
                latency_ms=0.0,  # No latency for generation
                is_fraud=is_fraud,
                transaction_id=transaction['transaction_id'],
                worker_id="generator"  # Mark as coming from generator
            )
            
            return stream_id
        except Exception as e:
            logger.error(f"Error sending transaction to stream: {str(e)}")
            return ""

    def generate_and_send(self, 
                        count: int = 1, 
                        fraud_ratio: float = 0.1,
                        delay_seconds: float = 0) -> int:
        """
        Generate and send a specified number of transactions.
        
        Args:
            count: Number of transactions to generate and send
            fraud_ratio: Ratio of transactions to mark as fraudulent
            delay_seconds: Delay between transactions in seconds
            
        Returns:
            Number of transactions successfully sent
        """
        sent_count = 0
        fraud_count = 0
        
        # Initialize run in stats manager
        self.stats_manager.initialize_run(
            pattern="manual",
            target_tps=1.0/delay_seconds if delay_seconds > 0 else 0,
            fraud_ratio=fraud_ratio
        )
        
        # Store run metadata in Redis (legacy)
        self._store_run_metadata(fraud_ratio=fraud_ratio)
        
        for i in range(count):
            try:
                # Determine if this transaction should be fraudulent
                is_fraud = random.random() < fraud_ratio
                
                # Generate and send transaction
                transaction = self.generate_transaction(inject_fraud=is_fraud)
                self.send_transaction(transaction)
                
                # Update counters
                sent_count += 1
                if is_fraud:
                    fraud_count += 1
                    
                # Apply delay if specified
                if delay_seconds > 0 and i < count - 1:  # No delay after the last transaction
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                logger.error(f"Error generating transaction: {str(e)}")
        
        # Complete the run in stats manager
        self.stats_manager.complete_run()
        
        # Return sent count
        return sent_count

    def _store_run_metadata(self, **kwargs) -> None:
        """
        Store run metadata in Redis.
        
        Args:
            **kwargs: Additional metadata to store
        """
        metadata = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'generator_pid': os.getpid(),
            **kwargs
        }
        
        # Store as JSON in Redis
        key = f"{settings.namespace}:runs:{self.run_id}"
        self.redis_client.json().set(key, '$', metadata)
        
        # Set expiration (keep for 24 hours)
        self.redis_client.expire(key, 60 * 60 * 24)

    def _worker_thread(self, 
                      transactions_per_second: float,
                      duration_seconds: float,
                      fraud_ratio: float,
                      pattern_func: Optional[Callable] = None,
                      thread_id: int = 0) -> None:
        """
        Worker thread for generating transactions at a specific rate.
        
        Args:
            transactions_per_second: Number of transactions to generate per second
            duration_seconds: How long to run in seconds
            fraud_ratio: Ratio of transactions to mark as fraudulent
            pattern_func: Optional function to modify generation patterns over time
            thread_id: Thread identifier for logging
        """
        local_count = 0
        local_fraud_count = 0
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Calculate sleep time between transactions
        sleep_time = 1.0 / transactions_per_second if transactions_per_second > 0 else 1.0
        
        logger.debug(f"Thread {thread_id} started: target {transactions_per_second} TPS for {duration_seconds}s")
        
        while self.running and time.time() < end_time:
            try:
                # Apply pattern function if provided
                current_fraud_ratio = fraud_ratio
                if pattern_func:
                    elapsed = time.time() - start_time
                    progress = elapsed / duration_seconds if duration_seconds > 0 else 0
                    current_fraud_ratio = pattern_func(fraud_ratio, progress, elapsed)
                
                # Determine if this transaction should be fraudulent
                is_fraud = random.random() < current_fraud_ratio
                
                # Generate and send transaction
                transaction = self.generate_transaction(inject_fraud=is_fraud)
                self.send_transaction(transaction)
                
                # Update counters
                local_count += 1
                if is_fraud:
                    local_fraud_count += 1
                
                # Update global counters
                self.transaction_count += 1
                if is_fraud:
                    self.fraud_count += 1
                
                # Log progress periodically
                if local_count % 100 == 0:
                    elapsed = time.time() - start_time
                    actual_tps = local_count / elapsed if elapsed > 0 else 0
                    remaining = duration_seconds - elapsed
                    logger.debug(f"Thread {thread_id}: Sent {local_count} transactions "
                              f"({actual_tps:.1f} TPS, {remaining:.1f}s remaining)")
                
                # Sleep to maintain the desired TPS
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Thread {thread_id}: Error generating transaction: {str(e)}")
        
        # Log final thread stats
        elapsed = time.time() - start_time
        actual_tps = local_count / elapsed if elapsed > 0 else 0
        logger.info(f"Thread {thread_id}: Completed with {local_count} transactions "
                  f"({local_fraud_count} fraud) in {elapsed:.1f}s ({actual_tps:.1f} TPS)")

    def generate_traffic(self, 
                        transactions_per_second: float = 10.0,
                        duration_seconds: int = 60,
                        fraud_ratio: float = 0.1,
                        thread_count: int = 1,
                        pattern: str = "constant") -> Dict[str, Any]:
        """
        Generate a traffic pattern of transactions over time.
        
        Args:
            transactions_per_second: Total transactions per second across all threads
            duration_seconds: How long to run in seconds
            fraud_ratio: Base ratio of transactions to mark as fraudulent
            thread_count: Number of parallel threads to use
            pattern: Traffic pattern to use (constant, wave, spike, random)
            
        Returns:
            Dictionary with generation statistics
        """
        if self.running:
            logger.warning("Traffic generation already in progress")
            return {"error": "Traffic generation already in progress"}
        
        # Set running state
        self.running = True
        self.transaction_count = 0
        self.fraud_count = 0
        self.start_time = time.time()
        self.worker_threads = []
        
        # Initialize run in statistics manager
        self.stats_manager.initialize_run(
            pattern=pattern,
            target_tps=transactions_per_second,
            fraud_ratio=fraud_ratio,
            thread_count=thread_count
        )
        
        # Store legacy run metadata
        self._store_run_metadata(
            fraud_ratio=fraud_ratio,
            target_tps=transactions_per_second,
            duration=duration_seconds,
            pattern=pattern,
            thread_count=thread_count
        )
        
        # Determine pattern function
        pattern_func = None
        if pattern == "wave":
            pattern_func = lambda base_ratio, progress, elapsed: base_ratio * (1 + 0.5 * np.sin(elapsed / 10 * np.pi))
        elif pattern == "spike":
            pattern_func = lambda base_ratio, progress, elapsed: base_ratio * (1 + 2 * (0.45 < progress < 0.55))
        elif pattern == "random":
            pattern_func = lambda base_ratio, progress, elapsed: base_ratio * random.uniform(0.5, 2.0)
            
        # Calculate TPS per thread
        tps_per_thread = transactions_per_second / thread_count if thread_count > 0 else transactions_per_second
        
        logger.info(f"Starting traffic generation: pattern={pattern}, target={transactions_per_second} TPS, "
                  f"threads={thread_count}, duration={duration_seconds}s, run_id={self.run_id}")
        
        # Start worker threads
        for i in range(thread_count):
            thread = threading.Thread(
                target=self._worker_thread,
                args=(tps_per_thread, duration_seconds, fraud_ratio, pattern_func, i),
                name=f"generator-{i}"
            )
            self.worker_threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in self.worker_threads:
            thread.join()
            
        # Calculate final statistics
        end_time = time.time()
        elapsed = end_time - self.start_time
        actual_tps = self.transaction_count / elapsed if elapsed > 0 else 0
        fraud_pct = (self.fraud_count / self.transaction_count * 100) if self.transaction_count > 0 else 0
        
        # Complete the run in stats manager
        self.stats_manager.complete_run()
        
        # Update legacy run metadata with final stats
        self._update_run_stats(
            transactions=self.transaction_count,
            fraud_count=self.fraud_count,
            fraud_percentage=fraud_pct,
            duration_seconds=elapsed,
            actual_tps=actual_tps
        )
        
        # Log summary
        logger.info(f"Traffic generation complete: {self.transaction_count} transactions "
                  f"({self.fraud_count} fraud, {fraud_pct:.1f}%) in {elapsed:.1f}s ({actual_tps:.1f} TPS)")
        
        # Reset state
        self.running = False
        
        # Return statistics
        return {
            "run_id": self.run_id,
            "transactions": self.transaction_count,
            "fraud_count": self.fraud_count,
            "fraud_percentage": fraud_pct,
            "duration_seconds": elapsed,
            "actual_tps": actual_tps,
            "target_tps": transactions_per_second,
            "threads": thread_count,
            "pattern": pattern
        }
    
    def _update_run_stats(self, **kwargs) -> None:
        """
        Update run statistics in Redis.
        
        Args:
            **kwargs: Statistics to update
        """
        # Update the run metadata in Redis
        key = f"{settings.namespace}:runs:{self.run_id}"
        
        # Check if key exists
        if not self.redis_client.exists(key):
            # Create it if it doesn't
            self._store_run_metadata()
        
        # Update with completion stats
        stats = {
            'end_time': datetime.now().isoformat(),
            'complete': True,
            **kwargs
        }
        
        # Update only the provided fields
        for field, value in stats.items():
            self.redis_client.json().set(key, f'$.{field}', value)
    
    def stop(self) -> None:
        """Stop any ongoing traffic generation."""
        if self.running:
            logger.info("Stopping traffic generation...")
            self.running = False
            
            # Wait for threads to exit
            for thread in self.worker_threads:
                thread.join(timeout=5)
            
            # Reset state
            self.worker_threads = []
            logger.info("Traffic generation stopped")

# Run generator if script is executed directly
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Generate transaction traffic for ML inference pipeline")
    parser.add_argument('--tps', type=float, default=10.0, help='Transactions per second')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--fraud-ratio', type=float, default=0.1, help='Ratio of fraudulent transactions')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel generation')
    parser.add_argument('--pattern', type=str, default='constant', 
                      choices=['constant', 'wave', 'spike', 'random'], 
                      help='Traffic pattern to generate')
    parser.add_argument('--run-id', type=str, help='Specific run ID to use (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create generator and run
    generator = TransactionGenerator(run_id=args.run_id)
    stats = generator.generate_traffic(
        transactions_per_second=args.tps,
        duration_seconds=args.duration,
        fraud_ratio=args.fraud_ratio,
        thread_count=args.threads,
        pattern=args.pattern
    )
    
    # Wait a moment for workers to process transactions
    time.sleep(2)
    
    # Get updated run statistics from the statistics manager
    run_stats = generator.stats_manager.get_run_stats()
    
    # Print formatted statistics
    print("\nDETAILED STATISTICS:")
    print_run_summary(run_stats)
    
    # Print legacy summary (shorter format)
    print("\nFINAL STATISTICS:")
    print(f"  Run ID: {stats['run_id']}")
    print(f"  Transactions generated: {stats['transactions']}")
    print(f"  Fraud transactions: {stats['fraud_count']} ({stats['fraud_percentage']:.1f}%)")
    print(f"  Actual throughput: {stats['actual_tps']:.2f} TPS")
    print(f"  Duration: {stats['duration_seconds']:.2f} seconds") 