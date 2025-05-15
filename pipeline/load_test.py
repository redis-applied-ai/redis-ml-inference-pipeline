#!/usr/bin/env python
"""
Unified load test script for the ML inference pipeline.

This script provides a configurable way to test the inference pipeline
under various load conditions.
"""
import argparse
import logging
import time
import threading
import subprocess
import sys
import os
import signal
import uuid
from typing import List, Dict, Any
import json
from datetime import datetime
from pipeline.stats.manager import StatisticsManager
from pipeline.stats.reporting import print_run_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoadTest:
    """
    Load test controller that manages workers and generators.
    """
    def __init__(self):
        self.processes = []
        self.running = False
        self.run_id = None
        
    def generate_run_id(self) -> str:
        """
        Generate a unique run ID for this load test
        
        Returns:
            Unique run ID string
        """
        # Format: 'loadtest-YYYYMMDD-HHMMSS-XXXX' where XXXX is a short random hex
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = uuid.uuid4().hex[:4]
        return f"loadtest-{timestamp}-{random_suffix}"
        
    def start_worker(self, 
                    worker_id: str, 
                    batch_size: int = 10, 
                    parallel: bool = False, 
                    threads: int = 1,
                    consumer_group: str = "inference_workers") -> subprocess.Popen:
        """
        Start a worker process with the specified configuration.
        
        Args:
            worker_id: Identifier for the worker
            batch_size: Batch size for processing transactions
            parallel: Whether to use parallel processing within the worker
            threads: Number of threads to use for parallel processing
            consumer_group: Consumer group name for Redis streams
            
        Returns:
            Process handle
        """
        cmd = [
            sys.executable, "-m", "pipeline.worker.inference_worker",
            "--worker-id", worker_id,
            "--batch-size", str(batch_size),
            "--consumer-group", consumer_group
        ]
        
        # Add parallel mode if specified
        if parallel:
            cmd.append("--parallel")
            cmd.extend(["--threads", str(threads)])
        
        # Start the process
        logger.info(f"Starting worker {worker_id} (batch_size={batch_size}, parallel={parallel}, threads={threads})")
        process = subprocess.Popen(cmd)
        return process
        
    def start_generator(self, 
                       tps: float, 
                       duration: int, 
                       fraud_ratio: float, 
                       threads: int = 1,
                       pattern: str = "constant",
                       run_id: str = None) -> subprocess.Popen:
        """
        Start a transaction generator process.
        
        Args:
            tps: Transactions per second
            duration: Test duration in seconds
            fraud_ratio: Fraud ratio for generated transactions
            threads: Number of threads for parallel generation
            pattern: Traffic pattern to generate
            run_id: Run ID for this test
            
        Returns:
            Process handle
        """
        cmd = [
            sys.executable, "-m", "pipeline.utils.transaction_generator",
            "--tps", str(tps),
            "--duration", str(duration),
            "--fraud-ratio", str(fraud_ratio),
            "--threads", str(threads),
            "--pattern", pattern
        ]
        
        # Add run_id if specified
        if run_id:
            cmd.extend(["--run-id", run_id])
        
        # Start the process
        logger.info(f"Starting generator: {tps} TPS for {duration}s with {threads} threads, pattern={pattern}, run_id={run_id}")
        process = subprocess.Popen(cmd)
        return process
    
    def run_test(self,
                tps: float = 100.0,
                duration: int = 60,
                fraud_ratio: float = 0.1,
                worker_count: int = 2,
                worker_batch_size: int = 10,
                worker_parallel: bool = False,
                worker_threads: int = 4,
                generator_threads: int = 2,
                traffic_pattern: str = "constant",
                run_id: str = None):
        """
        Run a complete load test with specified configuration.
        
        Args:
            tps: Transactions per second to generate
            duration: Test duration in seconds
            fraud_ratio: Ratio of fraudulent transactions
            worker_count: Number of workers to start
            worker_batch_size: Batch size for processing transactions
            worker_parallel: Whether to use parallel processing within workers
            worker_threads: Threads per worker for parallel processing
            generator_threads: Number of threads for transaction generator
            traffic_pattern: Traffic pattern for generation (constant, wave, spike, random)
            run_id: Specific run ID to use (if None, a new one is generated)
        """
        self.running = True
        self.processes = []
        
        # Generate run ID if not provided
        self.run_id = run_id or self.generate_run_id()
        logger.info(f"Starting load test with run ID: {self.run_id}")
        
        try:
            # Start workers
            for i in range(worker_count):
                process = self.start_worker(
                    worker_id=f"worker-{i}",
                    batch_size=worker_batch_size,
                    parallel=worker_parallel,
                    threads=worker_threads
                )
                self.processes.append(process)
            
            # Give workers time to initialize
            logger.info(f"Waiting for {worker_count} workers to initialize...")
            time.sleep(5)
            
            # Start generator
            generator = self.start_generator(
                tps=tps,
                duration=duration,
                fraud_ratio=fraud_ratio,
                threads=generator_threads,
                pattern=traffic_pattern,
                run_id=self.run_id
            )
            self.processes.append(generator)
            
            # Wait for generator to complete
            logger.info(f"Load test running for {duration} seconds...")
            logger.info(f"Press Ctrl+C to stop the test early")
            
            # Wait for generator to complete
            generator.wait()
            
            # Give workers time to process remaining messages
            logger.info("Transaction generation complete. Waiting for workers to process queue...")
            time.sleep(10)
            
            # Show test results
            logger.info(f"Load test complete. Run ID: {self.run_id}")
            
            # Display statistics if available
            try:
                stats_manager = StatisticsManager(run_id=self.run_id)
                run_stats = stats_manager.get_run_stats()
                print_run_summary(run_stats)
            except Exception as e:
                logger.error(f"Error displaying statistics: {str(e)}")
                
            logger.info(f"Check results with: poetry run monitor --stats --run-id {self.run_id}")
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            self.stop_test()
            
        logger.info("Load test complete")
    
    def stop_test(self):
        """Stop all processes and clean up."""
        if not self.running:
            return
            
        logger.info("Stopping all processes...")
        
        # Mark run as complete in statistics
        if self.run_id:
            try:
                stats_manager = StatisticsManager(run_id=self.run_id)
                stats_manager.complete_run()
                logger.info(f"Marked run {self.run_id} as complete in statistics")
            except Exception as e:
                logger.error(f"Error completing run statistics: {str(e)}")
        
        # Terminate all processes
        for process in self.processes:
            try:
                # Send SIGINT instead of SIGTERM to allow graceful shutdown
                os.kill(process.pid, signal.SIGINT)
            except:
                pass
        
        # Wait for processes to exit
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except:
                # Force kill if not exited
                try:
                    process.kill()
                except:
                    pass
        
        self.processes = []
        self.running = False
        logger.info("All processes stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run load test for ML inference pipeline")
    
    # Generator settings
    parser.add_argument('--tps', type=float, default=100.0, help='Transactions per second to generate')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--fraud-ratio', type=float, default=0.1, help='Ratio of fraudulent transactions')
    parser.add_argument('--pattern', type=str, default='constant', 
                       choices=['constant', 'wave', 'spike', 'random'], 
                       help='Traffic pattern to generate')
    parser.add_argument('--generator-threads', type=int, default=2, help='Threads for transaction generator')
    
    # Worker settings
    parser.add_argument('--workers', type=int, default=2, help='Number of workers to start')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing transactions')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing within workers')
    parser.add_argument('--worker-threads', type=int, default=4, help='Threads per worker for parallel processing')
    
    # Load test settings
    parser.add_argument('--run-id', type=str, help='Specific run ID to use (optional)')
    
    args = parser.parse_args()
    
    load_test = LoadTest()
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