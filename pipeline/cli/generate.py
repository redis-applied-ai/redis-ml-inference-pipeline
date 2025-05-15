#!/usr/bin/env python
"""
CLI script for generating test transactions.

This module provides a command-line interface for generating synthetic transactions 
for testing the ML inference pipeline.
"""
import argparse
from pipeline.utils.transaction_generator import TransactionGenerator

def main():
    """CLI entry point for generate command"""
    parser = argparse.ArgumentParser(description="Generate test transactions")
    parser.add_argument('--count', type=int, default=10, help='Number of transactions to generate')
    parser.add_argument('--interval', type=float, default=1.0, help='Time interval between transactions in seconds (or use --tps)')
    parser.add_argument('--tps', type=float, help='Transactions per second (alternative to --interval)')
    parser.add_argument('--duration', type=int, help='Duration in seconds (alternative to --count)')
    parser.add_argument('--fraud-ratio', type=float, default=0.1, help='Ratio of fraudulent transactions')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel generation')
    parser.add_argument('--pattern', type=str, default='constant', 
                      choices=['constant', 'wave', 'spike', 'random'], 
                      help='Traffic pattern to generate')
    parser.add_argument('--run-id', type=str, help='Specific run ID to use (optional)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible generation (optional)')
    
    args = parser.parse_args()
    
    # Create the generator with optional seed and run ID
    generator = TransactionGenerator(
        seed=args.seed,
        run_id=args.run_id
    )
    
    # Determine if we're using the simple mode or traffic pattern mode
    if args.duration and (args.tps or args.interval):
        # Calculate TPS if interval was provided
        tps = args.tps if args.tps else (1.0 / args.interval if args.interval > 0 else 1.0)
        
        # Traffic pattern mode
        stats = generator.generate_traffic(
            transactions_per_second=tps,
            duration_seconds=args.duration,
            fraud_ratio=args.fraud_ratio,
            thread_count=args.threads,
            pattern=args.pattern
        )
    else:
        # Simple batch generation mode
        count = args.count if args.count else 10
        sent = generator.generate_and_send(
            count=count, 
            fraud_ratio=args.fraud_ratio,
            delay_seconds=args.interval
        )
        print(f"Generated and sent {sent} transactions with run ID: {generator.get_run_id()}")

if __name__ == "__main__":
    main() 