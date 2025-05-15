"""
Statistics manager for the ML inference pipeline.

This module provides a centralized manager for tracking, aggregating,
and reporting statistics across the pipeline components.
"""
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Set, Any, Tuple, Union
from redis import Redis

from pipeline.settings import settings
from pipeline.utils.redis_utils import get_redis_client
from pipeline.stats.models import (
    LatencyStats, 
    WorkerStats, 
    RunStats, 
    RunMetadata, 
    TransactionEvent
)

logger = logging.getLogger(__name__)


class StatisticsManager:
    """
    Centralized manager for statistics across the ML inference pipeline.
    
    This class provides methods to:
    - Track statistics incrementally during a run
    - Aggregate statistics from multiple sources
    - Store statistics in Redis for persistence
    - Retrieve and report statistics in various formats
    """
    
    def __init__(
        self, 
        run_id: str, 
        redis_url: Optional[str] = None,
        worker_id: Optional[str] = None
    ):
        """
        Initialize the statistics manager.
        
        Args:
            run_id: Unique ID for the test run
            redis_url: Redis connection URL (defaults to settings)
            worker_id: Optional worker ID if this manager is for a specific worker
        """
        self.run_id = run_id
        self.worker_id = worker_id
        self.redis_client = get_redis_client(redis_url)
        
        # Initialize statistics containers
        self._raw_latencies: Dict[str, List[float]] = {}  # Worker ID -> latencies
        self._transaction_count: Dict[str, int] = {}      # Worker ID -> count
        self._fraud_count: Dict[str, int] = {}            # Worker ID -> fraud count
        
        # Store worker configuration if this is a worker-specific manager
        self._worker_config: Dict[str, Any] = {}
        
        # Metadata for the run
        self._run_metadata: Optional[RunMetadata] = None
        
        # Initialize if metadata exists in Redis
        self._load_existing_metadata()
        
    def _load_existing_metadata(self) -> None:
        """Load existing run metadata from Redis if available."""
        try:
            key = f"{settings.namespace}:runs:{self.run_id}"
            if self.redis_client.exists(key):
                data = self.redis_client.json().get(key)
                if data:
                    # Convert string timestamp to datetime
                    if 'start_time' in data and isinstance(data['start_time'], str):
                        data['start_time'] = datetime.fromisoformat(data['start_time'])
                    if 'end_time' in data and isinstance(data['end_time'], str):
                        data['end_time'] = datetime.fromisoformat(data['end_time'])
                        
                    self._run_metadata = RunMetadata(
                        run_id=self.run_id,
                        start_time=data.get('start_time', datetime.now()),
                        end_time=data.get('end_time'),
                        pattern=data.get('pattern', 'constant'),
                        target_tps=float(data.get('target_tps', 0.0)),
                        fraud_ratio=float(data.get('fraud_ratio', 0.1)),
                        thread_count=int(data.get('thread_count', 1)),
                        complete=bool(data.get('complete', False))
                    )
        except Exception as e:
            logger.warning(f"Error loading run metadata: {str(e)}")
    
    def initialize_run(
        self,
        pattern: str = "constant",
        target_tps: float = 0.0,
        fraud_ratio: float = 0.1,
        thread_count: int = 1
    ) -> None:
        """
        Initialize run metadata.
        
        Args:
            pattern: Traffic pattern used
            target_tps: Target transactions per second
            fraud_ratio: Target fraud ratio
            thread_count: Number of generator threads
        """
        self._run_metadata = RunMetadata(
            run_id=self.run_id,
            start_time=datetime.now(),
            pattern=pattern,
            target_tps=target_tps,
            fraud_ratio=fraud_ratio,
            thread_count=thread_count
        )
        self._save_run_metadata()
        
    def _save_run_metadata(self) -> None:
        """Save run metadata to Redis."""
        if not self._run_metadata:
            return
            
        key = f"{settings.namespace}:runs:{self.run_id}"
        
        # Convert to dict and handle datetime
        data = self._run_metadata.dict()
        if data.get('start_time'):
            data['start_time'] = data['start_time'].isoformat()
        if data.get('end_time'):
            data['end_time'] = data['end_time'].isoformat()
            
        # Store in Redis
        try:
            self.redis_client.json().set(key, '$', data)
            # Set expiration (keep for 24 hours)
            self.redis_client.expire(key, 60 * 60 * 24)
        except Exception as e:
            logger.error(f"Error saving run metadata: {str(e)}")
            
    def set_worker_config(
        self,
        worker_id: str,
        batch_size: int = 1,
        parallel_mode: bool = False,
        threads: int = 1
    ) -> None:
        """
        Set configuration parameters for a worker.
        
        Args:
            worker_id: Identifier for the worker
            batch_size: Batch size used by the worker
            parallel_mode: Whether worker used parallel processing
            threads: Number of threads used in parallel mode
        """
        self._worker_config[worker_id] = {
            'batch_size': batch_size,
            'parallel_mode': parallel_mode,
            'threads': threads,
            'start_time': datetime.now()
        }
        
    def record_transaction(
        self,
        latency_ms: float,
        is_fraud: bool = False,
        worker_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        batch_processed: bool = False,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Record a transaction processing event.
        
        Args:
            latency_ms: Processing latency in milliseconds
            is_fraud: Whether the transaction was flagged as fraud
            worker_id: Worker that processed the transaction (defaults to self.worker_id)
            transaction_id: Optional transaction ID
            batch_processed: Whether processed in a batch
            batch_size: Batch size if batch processed
        """
        # Use instance worker_id if not provided
        worker_id = worker_id or self.worker_id
        if not worker_id:
            logger.warning("No worker_id provided for transaction event")
            return
        
        # Special handling for generator transactions
        is_generator = worker_id == "generator"
            
        # Initialize worker entries if needed
        if worker_id not in self._raw_latencies:
            self._raw_latencies[worker_id] = []
        if worker_id not in self._transaction_count:
            self._transaction_count[worker_id] = 0
        if worker_id not in self._fraud_count:
            self._fraud_count[worker_id] = 0
            
        # Add data - only track latencies for actual workers, not generators
        if not is_generator:
            self._raw_latencies[worker_id].append(latency_ms)
            
        # Always track transaction counts
        self._transaction_count[worker_id] += 1
        if is_fraud:
            self._fraud_count[worker_id] += 1
            
        # Optionally publish event to Redis
        if transaction_id:
            event = TransactionEvent(
                run_id=self.run_id,
                worker_id=worker_id,
                transaction_id=transaction_id,
                latency_ms=latency_ms,
                is_fraud=is_fraud,
                batch_processed=batch_processed,
                batch_size=batch_size
            )
            self._publish_event(event)
            
    def _publish_event(self, event: TransactionEvent) -> None:
        """
        Publish a transaction event to Redis.
        
        Args:
            event: TransactionEvent to publish
        """
        try:
            # Convert to dict and handle datetime
            data = event.dict()
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
                
            # Publish to Redis stream
            stream_key = f"{settings.namespace}:stats:events:{self.run_id}"
            self.redis_client.xadd(
                stream_key,
                {
                    'data': json.dumps(data),
                    'worker_id': event.worker_id,
                    'run_id': event.run_id,
                    'latency_ms': str(event.latency_ms),
                    'is_fraud': '1' if event.is_fraud else '0'
                }
            )
            
            # Set expiration on stream (keep for 24 hours)
            self.redis_client.expire(stream_key, 60 * 60 * 24)
        except Exception as e:
            logger.warning(f"Error publishing transaction event: {str(e)}")
    
    def _calculate_latency_stats(self, latencies: List[float]) -> LatencyStats:
        """
        Calculate latency statistics from raw latency values.
        
        Args:
            latencies: List of latency measurements in milliseconds
            
        Returns:
            LatencyStats with calculated statistics
        """
        if not latencies:
            return LatencyStats()
            
        count = len(latencies)
        avg = sum(latencies) / count
        min_val = min(latencies)
        max_val = max(latencies)
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        if count >= 10:  # Need reasonable sample size
            p50 = np.percentile(sorted_latencies, 50)
            p95 = np.percentile(sorted_latencies, 95)
            p99 = np.percentile(sorted_latencies, 99)
        else:
            # Manual calculation for small samples
            p50_idx = int(count * 0.5)
            p95_idx = int(count * 0.95)
            p99_idx = int(count * 0.99)
            p50 = sorted_latencies[p50_idx] if p50_idx < count else max_val
            p95 = sorted_latencies[p95_idx] if p95_idx < count else max_val
            p99 = sorted_latencies[p99_idx] if p99_idx < count else max_val
            
        return LatencyStats(
            avg_ms=avg,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            min_ms=min_val,
            max_ms=max_val,
            count=count
        )
        
    def get_worker_stats(self, worker_id: Optional[str] = None) -> WorkerStats:
        """
        Get statistics for a specific worker.
        
        Args:
            worker_id: Worker ID to get stats for (defaults to self.worker_id)
            
        Returns:
            WorkerStats for the worker
        """
        worker_id = worker_id or self.worker_id
        if not worker_id:
            raise ValueError("No worker_id provided")
            
        # Check if we have data for this worker
        if worker_id not in self._transaction_count:
            return WorkerStats(
                worker_id=worker_id,
                run_id=self.run_id
            )
            
        # Get worker configuration
        config = self._worker_config.get(worker_id, {})
        start_time = config.get('start_time')
        processing_time = (datetime.now() - start_time).total_seconds() if start_time else 0
            
        # Calculate latency stats
        latencies = self._raw_latencies.get(worker_id, [])
        latency_stats = self._calculate_latency_stats(latencies)
        
        # Create WorkerStats
        return WorkerStats(
            worker_id=worker_id,
            run_id=self.run_id,
            transactions=self._transaction_count.get(worker_id, 0),
            fraud_count=self._fraud_count.get(worker_id, 0),
            latency=latency_stats,
            batch_size=config.get('batch_size', 1),
            parallel_mode=config.get('parallel_mode', False),
            threads=config.get('threads', 1),
            start_time=start_time,
            processing_time=processing_time
        )
        
    def save_worker_stats(self, worker_id: Optional[str] = None) -> None:
        """
        Save worker statistics to Redis.
        
        Args:
            worker_id: Worker ID to save stats for (defaults to self.worker_id)
        """
        worker_id = worker_id or self.worker_id
        if not worker_id:
            logger.warning("No worker_id provided for saving stats")
            return
            
        # Get worker stats
        stats = self.get_worker_stats(worker_id)
        
        # Save to Redis
        key = f"{settings.namespace}:runs:{self.run_id}:worker:{worker_id}:stats"
        try:
            # Handle datetime serialization
            data = stats.dict()
            if data.get('start_time'):
                data['start_time'] = data['start_time'].isoformat()
                
            self.redis_client.json().set(key, '$', data)
            # Set expiration (keep for 24 hours)
            self.redis_client.expire(key, 60 * 60 * 24)
            logger.debug(f"Saved worker stats for {worker_id} to Redis")
        except Exception as e:
            logger.error(f"Error saving worker stats: {str(e)}")
            
    def complete_run(self) -> None:
        """Mark the run as complete and save final metadata."""
        if not self._run_metadata:
            self._run_metadata = RunMetadata(run_id=self.run_id)
            
        self._run_metadata.end_time = datetime.now()
        self._run_metadata.complete = True
        self._save_run_metadata()
        
        # Save worker stats if this is a worker-specific manager
        if self.worker_id:
            self.save_worker_stats()
            
    def get_run_stats(self) -> RunStats:
        """
        Get complete statistics for the run by aggregating from all workers.
        
        Returns:
            RunStats with aggregated statistics
        """
        # Ensure we have metadata
        if not self._run_metadata:
            self._load_existing_metadata()
            if not self._run_metadata:
                self._run_metadata = RunMetadata(run_id=self.run_id)
                
        # Create base RunStats
        run_stats = RunStats(
            run_id=self.run_id,
            metadata=self._run_metadata
        )
        
        # If this is a worker-specific manager, add our stats
        if self.worker_id and self.worker_id != "generator":
            worker_stats = self.get_worker_stats()
            run_stats.update_from_worker(worker_stats)
            
        # Try to load additional worker stats from Redis
        worker_keys = self.redis_client.keys(f"{settings.namespace}:runs:{self.run_id}:worker:*:stats")
        for key in worker_keys:
            try:
                worker_data = self.redis_client.json().get(key)
                if not worker_data:
                    continue
                    
                # Handle datetime fields
                if 'start_time' in worker_data and isinstance(worker_data['start_time'], str):
                    worker_data['start_time'] = datetime.fromisoformat(worker_data['start_time'])
                    
                # Create WorkerStats and update RunStats
                worker_stats = WorkerStats(**worker_data)
                if worker_stats.worker_id != self.worker_id:  # Don't double-count this worker
                    run_stats.update_from_worker(worker_stats)
            except Exception as e:
                logger.warning(f"Error processing worker stats from {key}: {str(e)}")
                
        # If we have generator data but no worker data,
        # add the transaction counts to the run stats directly
        if "generator" in self._transaction_count:
            # Only use generator counts if we don't have any real workers reporting
            if run_stats.worker_count == 0:
                run_stats.transactions = self._transaction_count.get("generator", 0)
                run_stats.fraud_count = self._fraud_count.get("generator", 0)
                
        return run_stats
        
    @classmethod
    def load_run_stats(cls, run_id: str, redis_url: Optional[str] = None) -> Optional[RunStats]:
        """
        Load run statistics from Redis.
        
        Args:
            run_id: Run ID to load stats for
            redis_url: Redis connection URL (defaults to settings)
            
        Returns:
            RunStats if found, None otherwise
        """
        # Create a temporary instance to access Redis
        manager = cls(run_id=run_id, redis_url=redis_url)
        
        # Load metadata and check if it exists
        manager._load_existing_metadata()
        if not manager._run_metadata:
            logger.warning(f"No metadata found for run {run_id}")
            return None
        
        # Check if there's transaction data in Redis even though we don't have it in the manager
        run_key = f"{settings.namespace}:runs:{run_id}"
        redis_client = manager.redis_client
        transaction_count = 0
        fraud_count = 0
        
        # Try to get transaction counts directly from Redis
        try:
            metadata = redis_client.json().get(run_key)
            if metadata:
                transaction_count = int(metadata.get('transactions', 0))
                fraud_count = int(metadata.get('fraud_count', 0))
        except Exception as e:
            logger.warning(f"Error loading transaction counts from Redis: {str(e)}")
            
        # Build complete run stats
        run_stats = manager.get_run_stats()
        
        # If we have Redis data but no stats data, update the run stats
        if transaction_count > 0 and run_stats.transactions == 0:
            run_stats.transactions = transaction_count
            run_stats.fraud_count = fraud_count
            
        return run_stats
        
    @classmethod
    def list_runs(cls, limit: int = 10, redis_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List recent runs with basic information.
        
        Args:
            limit: Maximum number of runs to return
            redis_url: Redis connection URL (defaults to settings)
            
        Returns:
            List of run metadata dictionaries
        """
        # Create a temporary instance to access Redis
        manager = cls(run_id="temp", redis_url=redis_url)
        
        # Find run keys in Redis
        run_keys = manager.redis_client.keys(f"{settings.namespace}:runs:*")
        
        # Filter to base run keys (not worker stats)
        base_run_keys = [key.decode() for key in run_keys 
                        if len(key.decode().split(':')) == 3]  # namespace:runs:run_id
        
        # Get metadata for each run
        runs = []
        for key in base_run_keys:
            try:
                run_id = key.split(':')[-1]
                data = manager.redis_client.json().get(key)
                if data:
                    # Handle datetime fields
                    if 'start_time' in data and isinstance(data['start_time'], str):
                        data['start_time'] = datetime.fromisoformat(data['start_time'])
                    if 'end_time' in data and isinstance(data['end_time'], str):
                        data['end_time'] = datetime.fromisoformat(data['end_time'])
                        
                    # Calculate duration
                    if data.get('start_time') and data.get('end_time'):
                        duration = (data['end_time'] - data['start_time']).total_seconds()
                    else:
                        duration = 0
                        
                    runs.append({
                        'run_id': run_id,
                        'start_time': data.get('start_time'),
                        'end_time': data.get('end_time'),
                        'duration': duration,
                        'pattern': data.get('pattern', 'unknown'),
                        'target_tps': float(data.get('target_tps', 0)),
                        'complete': bool(data.get('complete', False))
                    })
            except Exception as e:
                logger.warning(f"Error processing run {key}: {str(e)}")
                
        # Sort by start time (newest first) and limit
        runs.sort(key=lambda r: r.get('start_time', datetime.min), reverse=True)
        return runs[:limit] 