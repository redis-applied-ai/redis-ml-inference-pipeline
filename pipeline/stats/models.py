"""
Statistics models for the ML inference pipeline.

This module defines Pydantic models for representing statistics
across the pipeline components. These models ensure type safety
and consistent serialization for reporting and storage.
"""
from datetime import datetime
from typing import List, Dict, Optional, Union, Any, Set
from pydantic import BaseModel, Field, validator


class LatencyStats(BaseModel):
    """Statistics for latency measurements."""
    avg_ms: float = Field(default=0.0, description="Average latency in milliseconds")
    p50_ms: float = Field(default=0.0, description="50th percentile (median) latency in milliseconds")
    p95_ms: float = Field(default=0.0, description="95th percentile latency in milliseconds")
    p99_ms: float = Field(default=0.0, description="99th percentile latency in milliseconds")
    min_ms: float = Field(default=0.0, description="Minimum latency in milliseconds")
    max_ms: float = Field(default=0.0, description="Maximum latency in milliseconds")
    count: int = Field(default=0, description="Number of latency measurements")
    
    @validator('avg_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'min_ms', 'max_ms')
    def round_latency(cls, v):
        """Round latency values to 2 decimal places."""
        return round(v, 2) if v is not None else 0.0
    
    def merge(self, other: 'LatencyStats') -> 'LatencyStats':
        """Merge with another LatencyStats instance."""
        if self.count == 0:
            return other
        if other.count == 0:
            return self
            
        total_count = self.count + other.count
        avg = (self.avg_ms * self.count + other.avg_ms * other.count) / total_count
        
        # For percentiles, we take the higher values as a conservative estimate
        return LatencyStats(
            avg_ms=avg,
            p50_ms=max(self.p50_ms, other.p50_ms),
            p95_ms=max(self.p95_ms, other.p95_ms),
            p99_ms=max(self.p99_ms, other.p99_ms),
            min_ms=min(self.min_ms, other.min_ms) if self.min_ms > 0 else other.min_ms,
            max_ms=max(self.max_ms, other.max_ms),
            count=total_count
        )


class WorkerStats(BaseModel):
    """Statistics for a single worker's performance."""
    worker_id: str = Field(..., description="Unique identifier for the worker")
    run_id: str = Field(..., description="Identifier for the test run")
    transactions: int = Field(default=0, description="Number of transactions processed")
    fraud_count: int = Field(default=0, description="Number of fraud transactions detected")
    latency: LatencyStats = Field(default_factory=LatencyStats, description="Latency statistics")
    batch_size: int = Field(default=1, description="Batch size used by the worker")
    parallel_mode: bool = Field(default=False, description="Whether worker used parallel processing")
    threads: int = Field(default=1, description="Number of threads used in parallel mode")
    start_time: Optional[datetime] = Field(default=None, description="Worker start time")
    processing_time: float = Field(default=0.0, description="Total processing time in seconds")
    
    @property
    def throughput(self) -> float:
        """Calculate throughput in transactions per second."""
        if self.processing_time <= 0:
            return 0.0
        return round(self.transactions / self.processing_time, 2)
    
    @property
    def fraud_ratio(self) -> float:
        """Calculate ratio of fraud transactions."""
        if self.transactions <= 0:
            return 0.0
        return round(self.fraud_count / self.transactions, 4)
    
    def merge(self, other: 'WorkerStats') -> 'WorkerStats':
        """Merge with another WorkerStats instance for the same worker."""
        if self.worker_id != other.worker_id:
            raise ValueError(f"Cannot merge stats for different workers: {self.worker_id} vs {other.worker_id}")
            
        return WorkerStats(
            worker_id=self.worker_id,
            run_id=self.run_id,
            transactions=self.transactions + other.transactions,
            fraud_count=self.fraud_count + other.fraud_count,
            latency=self.latency.merge(other.latency),
            batch_size=max(self.batch_size, other.batch_size),
            parallel_mode=self.parallel_mode or other.parallel_mode,
            threads=max(self.threads, other.threads),
            start_time=self.start_time or other.start_time,
            processing_time=self.processing_time + other.processing_time
        )


class RunMetadata(BaseModel):
    """Metadata for a test run."""
    run_id: str = Field(..., description="Unique identifier for the run")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time of the run")
    end_time: Optional[datetime] = Field(default=None, description="End time of the run")
    pattern: str = Field(default="constant", description="Traffic pattern used")
    target_tps: float = Field(default=0.0, description="Target transactions per second")
    fraud_ratio: float = Field(default=0.1, description="Target fraud ratio")
    thread_count: int = Field(default=1, description="Number of generator threads used")
    complete: bool = Field(default=False, description="Whether the run completed successfully")


class RunStats(BaseModel):
    """Complete statistics for a test run across all workers."""
    run_id: str = Field(..., description="Unique identifier for the test run")
    metadata: RunMetadata = Field(..., description="Metadata for the run")
    worker_stats: Dict[str, WorkerStats] = Field(default_factory=dict, description="Statistics per worker")
    
    # Aggregated stats
    transactions: int = Field(default=0, description="Total transactions processed")
    fraud_count: int = Field(default=0, description="Total fraud transactions detected")
    latency: LatencyStats = Field(default_factory=LatencyStats, description="Aggregated latency statistics")
    
    @property
    def duration_seconds(self) -> float:
        """Calculate run duration in seconds."""
        if not self.metadata.end_time or not self.metadata.start_time:
            return 0.0
        return (self.metadata.end_time - self.metadata.start_time).total_seconds()
    
    @property
    def actual_tps(self) -> float:
        """Calculate actual transactions per second."""
        duration = self.duration_seconds
        if duration <= 0:
            return 0.0
        return round(self.transactions / duration, 2)
    
    @property
    def fraud_ratio(self) -> float:
        """Calculate overall fraud ratio."""
        if self.transactions <= 0:
            return 0.0
        return round(self.fraud_count / self.transactions, 4)
    
    @property
    def worker_count(self) -> int:
        """Get number of workers that contributed."""
        return len(self.worker_stats)
    
    @property
    def worker_ids(self) -> List[str]:
        """Get list of worker IDs."""
        return list(self.worker_stats.keys())
    
    def update_from_worker(self, worker_stats: WorkerStats) -> None:
        """
        Update run statistics with data from a worker.
        
        Args:
            worker_stats: WorkerStats instance to include
        """
        # Store/update worker stats
        if worker_stats.worker_id in self.worker_stats:
            self.worker_stats[worker_stats.worker_id] = self.worker_stats[worker_stats.worker_id].merge(worker_stats)
        else:
            self.worker_stats[worker_stats.worker_id] = worker_stats
            
        # Recalculate aggregated stats
        self.transactions = sum(w.transactions for w in self.worker_stats.values())
        self.fraud_count = sum(w.fraud_count for w in self.worker_stats.values())
        
        # Merge latency stats from all workers
        merged_latency = LatencyStats()
        for worker_stats in self.worker_stats.values():
            merged_latency = merged_latency.merge(worker_stats.latency)
        self.latency = merged_latency


class TransactionEvent(BaseModel):
    """A transaction processing event for incremental stats updates."""
    run_id: str = Field(..., description="Run ID for the transaction")
    worker_id: str = Field(..., description="Worker ID that processed the transaction")
    transaction_id: str = Field(..., description="Transaction ID")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    is_fraud: bool = Field(default=False, description="Whether transaction was flagged as fraud")
    batch_processed: bool = Field(default=False, description="Whether processed in a batch")
    batch_size: Optional[int] = Field(default=None, description="Batch size if batch processed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp") 