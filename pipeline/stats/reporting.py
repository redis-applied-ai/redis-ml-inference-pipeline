"""
Reporting utilities for statistics in the ML inference pipeline.

This module provides functions to format and display statistics
from the pipeline in a user-friendly way.
"""
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, TextIO

from pipeline.stats.models import RunStats, WorkerStats, LatencyStats


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds / 3600)
        remaining = seconds % 3600
        minutes = int(remaining / 60)
        remaining_seconds = remaining % 60
        return f"{hours}h {minutes}m {remaining_seconds:.1f}s"


def format_latency(latency: float) -> str:
    """
    Format latency with appropriate color coding.
    
    Args:
        latency: Latency value in milliseconds
        
    Returns:
        Color-coded latency string
    """
    if latency > 100:
        return f"\033[91m{latency:.2f}ms\033[0m"  # Red for >100ms
    elif latency > 50:
        return f"\033[93m{latency:.2f}ms\033[0m"  # Yellow for >50ms
    else:
        return f"\033[92m{latency:.2f}ms\033[0m"  # Green for ≤50ms


def format_latency_stats(latency: LatencyStats) -> Dict[str, str]:
    """
    Format all latency statistics with color coding.
    
    Args:
        latency: LatencyStats instance
        
    Returns:
        Dictionary of formatted latency strings
    """
    return {
        'avg': format_latency(latency.avg_ms),
        'p50': format_latency(latency.p50_ms),
        'p95': format_latency(latency.p95_ms),
        'p99': format_latency(latency.p99_ms),
        'min': format_latency(latency.min_ms),
        'max': format_latency(latency.max_ms)
    }


def format_tps(tps: float) -> str:
    """
    Format transactions per second with color coding based on performance.
    
    Args:
        tps: Transactions per second
        
    Returns:
        Color-coded TPS string
    """
    if tps > 500:
        return f"\033[92m{tps:.1f}\033[0m"  # Green for high TPS
    elif tps > 100:
        return f"\033[93m{tps:.1f}\033[0m"  # Yellow for medium TPS
    else:
        return f"\033[91m{tps:.1f}\033[0m"  # Red for low TPS


def format_date(dt: datetime) -> str:
    """
    Format datetime to a human-readable string.
    
    Args:
        dt: Datetime to format
        
    Returns:
        Formatted date string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_run_summary(run_stats: RunStats, file: TextIO = sys.stdout) -> None:
    """
    Print a summary of run statistics.
    
    Args:
        run_stats: RunStats instance
        file: File to print to (defaults to stdout)
    """
    # Get formatted values
    latency_fmt = format_latency_stats(run_stats.latency)
    
    # Duration and TPS
    duration = format_duration(run_stats.duration_seconds)
    tps = format_tps(run_stats.actual_tps)
    
    # Status
    status = "\033[92mComplete\033[0m" if run_stats.metadata.complete else "\033[93mIn Progress\033[0m"
    
    # Check if we only have generator data with no workers
    generator_only = run_stats.worker_count == 0 and run_stats.transactions > 0
    
    # Print summary
    print("\n" + "=" * 80, file=file)
    print(f"RUN STATISTICS: {run_stats.run_id}", file=file)
    print("=" * 80, file=file)
    print(f"Status: {status}", file=file)
    print(f"Start Time: {format_date(run_stats.metadata.start_time)}", file=file)
    if run_stats.metadata.end_time:
        print(f"End Time: {format_date(run_stats.metadata.end_time)}", file=file)
    print(f"Duration: {duration}", file=file)
    print(f"Pattern: {run_stats.metadata.pattern}", file=file)
    print(f"Target TPS: {run_stats.metadata.target_tps:.1f}", file=file)
    print(f"Actual TPS: {tps}", file=file)
    print(f"Transactions: {run_stats.transactions:,}", file=file)
    print(f"Fraud: {run_stats.fraud_count:,} ({run_stats.fraud_ratio*100:.1f}%)", file=file)
    print(f"Workers: {run_stats.worker_count} ({', '.join(run_stats.worker_ids)})", file=file)
    
    # Print latency statistics - with special handling for generator-only runs
    print("\nLatency Statistics:", file=file)
    if generator_only:
        print(f"  \033[93mNo latency data available - transactions were generated but not processed by workers\033[0m", file=file)
        print(f"  Run the inference worker with this run ID to get latency metrics", file=file)
    else:
        print(f"  Average: {latency_fmt['avg']}", file=file)
        print(f"  Median (p50): {latency_fmt['p50']}", file=file)
        print(f"  p95: {latency_fmt['p95']}", file=file)
        print(f"  p99: {latency_fmt['p99']}", file=file)
        print(f"  Min: {latency_fmt['min']}", file=file)
        print(f"  Max: {latency_fmt['max']}", file=file)
        print(f"  Sample size: {run_stats.latency.count:,}", file=file)
    print("=" * 80, file=file)


def print_worker_stats(worker_stats: WorkerStats, file: TextIO = sys.stdout) -> None:
    """
    Print statistics for a single worker.
    
    Args:
        worker_stats: WorkerStats instance
        file: File to print to (defaults to stdout)
    """
    # Get formatted values
    latency_fmt = format_latency_stats(worker_stats.latency)
    
    # Format mode
    mode = "Parallel" if worker_stats.parallel_mode else "Sequential"
    
    # Print summary
    print("\n" + "-" * 70, file=file)
    print(f"WORKER: {worker_stats.worker_id}", file=file)
    print("-" * 70, file=file)
    print(f"Run ID: {worker_stats.run_id}", file=file)
    print(f"Mode: {mode} (batch: {worker_stats.batch_size}, threads: {worker_stats.threads})", file=file)
    if worker_stats.start_time:
        print(f"Start Time: {format_date(worker_stats.start_time)}", file=file)
    print(f"Processing Time: {format_duration(worker_stats.processing_time)}", file=file)
    print(f"Transactions: {worker_stats.transactions:,}", file=file)
    print(f"Throughput: {format_tps(worker_stats.throughput)} TPS", file=file)
    print(f"Fraud: {worker_stats.fraud_count:,} ({worker_stats.fraud_ratio*100:.1f}%)", file=file)
    
    # Print latency statistics
    print("\nLatency Statistics:", file=file)
    print(f"  Average: {latency_fmt['avg']}", file=file)
    print(f"  Median (p50): {latency_fmt['p50']}", file=file)
    print(f"  p95: {latency_fmt['p95']}", file=file)
    print(f"  p99: {latency_fmt['p99']}", file=file)
    print(f"  Min: {latency_fmt['min']}", file=file)
    print(f"  Max: {latency_fmt['max']}", file=file)
    print(f"  Sample size: {worker_stats.latency.count:,}", file=file)


def print_run_list(runs: List[Dict[str, Any]], file: TextIO = sys.stdout) -> None:
    """
    Print a list of available runs.
    
    Args:
        runs: List of run metadata dictionaries
        file: File to print to (defaults to stdout)
    """
    if not runs:
        print("No runs found", file=file)
        return
        
    # Print header
    print("\n" + "=" * 90, file=file)
    print("AVAILABLE RUNS", file=file)
    print("=" * 90, file=file)
    print("Run ID                         | Status    | Duration   | TPS   | Pattern    | Start Time", file=file)
    print("-" * 90, file=file)
    
    # Print each run
    for run in runs:
        # Format fields
        status = "\033[92mComplete\033[0m" if run.get('complete', False) else "\033[93mIn Progress\033[0m"
        duration = format_duration(run.get('duration', 0))
        tps = format_tps(run.get('target_tps', 0))
        start_time = format_date(run['start_time']) if run.get('start_time') else "Unknown"
        pattern = run.get('pattern', 'unknown')
        
        print(f"{run['run_id']:30} | {status:9} | {duration:10} | {tps:5} | {pattern:10} | {start_time}", file=file)
    
    print("=" * 90, file=file)


def run_stats_to_json(run_stats: RunStats) -> Dict[str, Any]:
    """
    Convert RunStats to a JSON-serializable dictionary.
    
    Args:
        run_stats: RunStats instance
        
    Returns:
        Dictionary with run statistics
    """
    # Convert metadata
    metadata = run_stats.metadata.dict()
    metadata['start_time'] = metadata['start_time'].isoformat() if metadata.get('start_time') else None
    metadata['end_time'] = metadata['end_time'].isoformat() if metadata.get('end_time') else None
    
    # Convert worker stats
    workers = {}
    for worker_id, worker in run_stats.worker_stats.items():
        worker_dict = worker.dict()
        if worker_dict.get('start_time'):
            worker_dict['start_time'] = worker_dict['start_time'].isoformat()
        workers[worker_id] = worker_dict
    
    # Build result
    return {
        'run_id': run_stats.run_id,
        'metadata': metadata,
        'workers': workers,
        'transactions': run_stats.transactions,
        'fraud_count': run_stats.fraud_count,
        'fraud_ratio': run_stats.fraud_ratio,
        'duration_seconds': run_stats.duration_seconds,
        'actual_tps': run_stats.actual_tps,
        'latency': run_stats.latency.dict(),
        'worker_count': run_stats.worker_count,
        'worker_ids': run_stats.worker_ids
    }


def export_run_stats_json(run_stats: RunStats, file_path: Optional[str] = None) -> str:
    """
    Export run statistics to JSON format.
    
    Args:
        run_stats: RunStats instance
        file_path: Optional path to write JSON to (if None, returns JSON string)
        
    Returns:
        JSON string representation of run statistics
    """
    # Convert to JSON-serializable dict
    data = run_stats_to_json(run_stats)
    
    # Convert to formatted JSON string
    json_str = json.dumps(data, indent=2)
    
    # Write to file if requested
    if file_path:
        with open(file_path, 'w') as f:
            f.write(json_str)
    
    return json_str 