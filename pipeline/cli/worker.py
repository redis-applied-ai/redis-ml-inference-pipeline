#!/usr/bin/env python
"""
CLI script for running ML inference workers.

This module provides command-line interface for launching one or more inference workers,
either in standalone or distributed mode.
"""
import argparse
import multiprocessing
import os
import signal
import time
from pipeline.worker.inference_worker import InferenceWorker

def run_worker(model_version, worker_id, batch_size=10, timeout_ms=5000, parallel=False, threads=4, consumer_group="inference_workers"):
    """
    Run a single worker process with the given ID.
    
    Args:
        model_version: Specific model version to use (or None for latest)
        worker_id: Identifier for the worker
        batch_size: Number of messages to process in each batch
        timeout_ms: Timeout in milliseconds for blocking read from Redis streams
        parallel: Whether to process transactions in parallel within the worker
        threads: Number of threads to use for parallel processing
        consumer_group: Redis consumer group name
    """
    print(f"Starting worker {worker_id}" + (" (parallel mode)" if parallel else ""))
    
    worker = InferenceWorker(
        model_version=model_version, 
        worker_id=worker_id,
        batch_size=batch_size,
        parallel_mode=parallel,
        threads=threads,
        consumer_group=consumer_group
    )
    worker.run(timeout_ms=timeout_ms)

def main():
    """CLI entry point for worker command"""
    parser = argparse.ArgumentParser(description="Run ML inference worker(s)")
    parser.add_argument('--model-version', type=str, help='Specific model version to use')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of messages to process in each batch')
    parser.add_argument('--timeout', type=int, default=5000, help='Timeout in milliseconds for blocking read')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes to start')
    parser.add_argument('--parallel', action='store_true', help='Process in parallel mode using threads')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use in parallel mode')
    parser.add_argument('--consumer-group', type=str, default="inference_workers", help='Consumer group name')
    
    args = parser.parse_args()
    
    if args.workers == 1:
        # Single worker mode - run directly
        print("Starting a single worker")
        
        worker = InferenceWorker(
            model_version=args.model_version,
            batch_size=args.batch_size,
            parallel_mode=args.parallel,
            threads=args.threads,
            consumer_group=args.consumer_group
        )
        worker.run(timeout_ms=args.timeout)
    else:
        # Multi-worker mode - use multiprocessing
        print(f"Starting {args.workers} workers")
        processes = []
        
        try:
            # Start the workers
            for i in range(args.workers):
                p = multiprocessing.Process(
                    target=run_worker,
                    args=(
                        args.model_version,  # model_version
                        f"worker-{i+1}",     # worker_id
                    ),
                    kwargs={
                        'batch_size': args.batch_size,
                        'timeout_ms': args.timeout,
                        'parallel': args.parallel,
                        'threads': args.threads,
                        'consumer_group': args.consumer_group
                    }
                )
                p.start()
                processes.append(p)
                print(f"Worker {i+1} started (PID: {p.pid})")
            
            # Wait for all processes to complete (or until interrupted)
            print(f"All {args.workers} workers are running. Press Ctrl+C to stop.")
            
            # Keep the main process running to handle signals
            while all(p.is_alive() for p in processes):
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down workers...")
            
            # Terminate all worker processes
            for i, p in enumerate(processes):
                if p.is_alive():
                    print(f"Terminating worker {i+1} (PID: {p.pid})")
                    p.terminate()
            
            # Wait for processes to terminate
            for i, p in enumerate(processes):
                p.join(timeout=2)
                if p.is_alive():
                    print(f"Worker {i+1} did not terminate cleanly, killing it")
                    p.kill()
            
            print("All workers shut down")

if __name__ == "__main__":
    # Improve the behavior when Ctrl+C is pressed
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    main() 