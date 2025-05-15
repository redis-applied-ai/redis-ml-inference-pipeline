#!/usr/bin/env python
"""
CLI script for running load tests on the ML inference pipeline.

This module provides a unified command-line interface to run load tests with
configurable workers and transaction generation.
"""
import argparse
import os
import sys

from pipeline.load_test import LoadTest

def main():
    """CLI entry point for load test command"""
    parser = argparse.ArgumentParser(description="Run ML inference pipeline load test")
    
    # Generator settings
    generator_group = parser.add_argument_group('Transaction Generator Options')
    generator_group.add_argument('--tps', type=float, default=100.0, help='Transactions per second to generate')
    generator_group.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    generator_group.add_argument('--fraud-ratio', type=float, default=0.1, help='Ratio of fraudulent transactions')
    generator_group.add_argument('--pattern', type=str, default='constant', 
                           choices=['constant', 'wave', 'spike', 'random'], 
                           help='Traffic pattern to generate')
    generator_group.add_argument('--generator-threads', type=int, default=2, help='Threads for transaction generator')
    
    # Worker settings
    worker_group = parser.add_argument_group('Worker Options')
    worker_group.add_argument('--workers', type=int, default=2, help='Number of workers to start')
    worker_group.add_argument('--batch-size', type=int, default=10, help='Batch size for processing transactions')
    worker_group.add_argument('--parallel', action='store_true', help='Use parallel processing within workers')
    worker_group.add_argument('--worker-threads', type=int, default=4, help='Threads per worker for parallel processing')
    worker_group.add_argument('--model-version', type=str, help='Specific model version to use')
    
    # Load test settings
    test_group = parser.add_argument_group('Test Options')
    test_group.add_argument('--run-id', type=str, help='Specific run ID to use (optional)')
    
    args = parser.parse_args()
    
    # Create and run the load test
    load_test = LoadTest()
    try:
        load_test.run_test(
            tps=args.tps,
            duration=args.duration,
            fraud_ratio=args.fraud_ratio,
            worker_count=args.workers,
            worker_batch_size=args.batch_size,
            worker_parallel=args.parallel,
            worker_threads=args.worker_threads,
            generator_threads=args.generator_threads,
            traffic_pattern=args.pattern,
            run_id=args.run_id
        )
    except KeyboardInterrupt:
        print("\nLoad test interrupted by user")
        load_test.stop_test()
    except Exception as e:
        print(f"\nError during load test: {str(e)}")
        load_test.stop_test()
        sys.exit(1)

if __name__ == "__main__":
    main() 