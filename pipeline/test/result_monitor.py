import json
from redis import Redis
import argparse
import statistics
import numpy as np
from datetime import datetime
from time import perf_counter_ns, time

# Add the project root to the path
from pipeline.settings import settings

class ResultMonitor:
    """
    Monitors and displays ML inference results from Redis stream
    """
    def __init__(self, redis_url=settings.redis_url):
        """Initialize the monitor"""
        self.redis_client = Redis.from_url(redis_url)
        self.running = False
        self.all_latencies = []  # Track all latencies for batch statistics
        self.start_time = None
        self.run_latencies = {}  # Track latencies by run_id
        
    def get_latest_results(self, count=10, run_id=None):
        """
        Get latest inference results from the stream
        
        Args:
            count: Number of results to fetch
            run_id: Optional run ID to filter results by
            
        Returns:
            List of result dictionaries
        """
        # Read from stream
        response = self.redis_client.xrevrange(
            settings.namespaced_fraud_results_stream,  # Stream name
            max='+',               # Start at newest
            min='-',               # Go to oldest
            count=count * 5 if run_id else count  # Fetch more if filtering by run_id
        )
        
        results = []
        for stream_id, message in response:
            try:
                # Get the result data
                if b'data' in message:
                    result = json.loads(message[b'data'].decode())
                    result['stream_id'] = stream_id.decode()
                    
                    # Add the run_id from stream metadata if not in the result
                    if 'run_id' not in result and b'run_id' in message:
                        result['run_id'] = message[b'run_id'].decode()
                    
                    # Filter by run_id if specified
                    if run_id and result.get('run_id') != run_id:
                        continue
                        
                    # Add to results
                    results.append(result)
                    
                    # Stop if we have enough results
                    if len(results) >= count:
                        break
            except Exception as e:
                print(f"Error parsing result: {e}")
                
        return results
    
    def get_run_ids(self, limit=10):
        """
        Get a list of recent run IDs from Redis
        
        Args:
            limit: Maximum number of run IDs to return
            
        Returns:
            List of run ID strings
        """
        # Find run keys in Redis
        run_keys = self.redis_client.keys(f"{settings.namespace}:runs:run-*")
        
        # Extract run IDs from keys
        run_ids = []
        for key in run_keys:
            try:
                # Extract run ID from key
                run_id = key.decode().split(":")[-1]
                if "worker" not in run_id:  # Skip worker stat keys
                    run_ids.append(run_id)
            except:
                pass
                
        # Sort by most recent first (based on timestamp in ID)
        run_ids.sort(reverse=True)
        
        # Limit the number of results
        return run_ids[:limit]
    
    def get_run_stats(self, run_id):
        """
        Get statistics for a specific run
        
        Args:
            run_id: Run ID to get statistics for
            
        Returns:
            Dictionary with run statistics
        """
        # Get the run metadata
        key = f"{settings.namespace}:runs:{run_id}"
        if not self.redis_client.exists(key):
            return None
            
        # Get run metadata
        metadata = self.redis_client.json().get(key)
        
        # Get worker stats for this run
        worker_keys = self.redis_client.keys(f"{settings.namespace}:runs:{run_id}:worker:*:stats")
        
        # Aggregate worker stats
        all_latencies = []
        transaction_count = 0
        workers = []
        
        for worker_key in worker_keys:
            try:
                worker_stats = self.redis_client.json().get(worker_key)
                if worker_stats and 'latency_stats' in worker_stats:
                    # Add worker to list
                    workers.append(worker_stats['worker_id'])
                    
                    # Add to transaction count
                    if 'transaction_count' in worker_stats:
                        transaction_count += worker_stats['transaction_count']
                        
                    # Aggregate latencies - we'll weight by count
                    if 'latency_stats' in worker_stats and 'avg_latency_ms' in worker_stats['latency_stats']:
                        lat_stats = worker_stats['latency_stats']
                        count = lat_stats.get('count', 0)
                        if count > 0:
                            all_latencies.extend([lat_stats['avg_latency_ms']] * count)
            except Exception as e:
                print(f"Error processing worker stats: {e}")
                
        # Calculate overall latency statistics
        if all_latencies:
            avg_latency = sum(all_latencies) / len(all_latencies)
            
            # Calculate percentiles if enough data
            if len(all_latencies) >= 10:
                sorted_latencies = sorted(all_latencies)
                p50 = np.percentile(sorted_latencies, 50)
                p95 = np.percentile(sorted_latencies, 95)
                p99 = np.percentile(sorted_latencies, 99)
            else:
                p50 = p95 = p99 = avg_latency
        else:
            avg_latency = p50 = p95 = p99 = 0
            
        # Combine with metadata
        stats = {
            'run_id': run_id,
            'start_time': metadata.get('start_time', ''),
            'end_time': metadata.get('end_time', ''),
            'fraud_ratio': metadata.get('fraud_ratio', 0),
            'target_tps': metadata.get('target_tps', 0),
            'actual_tps': metadata.get('actual_tps', 0),
            'transaction_count': metadata.get('transactions', transaction_count),
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50,
            'p95_latency_ms': p95,
            'p99_latency_ms': p99,
            'workers': len(workers),
            'worker_ids': workers,
            'pattern': metadata.get('pattern', 'unknown'),
            'duration_seconds': metadata.get('duration_seconds', 0),
            'complete': metadata.get('complete', False)
        }
        
        return stats
    
    def watch_stream(self, last_id='0-0', timeout_ms=5000, run_id=None):
        """
        Watch for new messages in the stream
        
        Args:
            last_id: ID to start reading from
            timeout_ms: Timeout in milliseconds for blocking read
            run_id: Optional run ID to filter by
            
        Returns:
            List of new results
        """
        # Block until new messages arrive
        response = self.redis_client.xread(
            {settings.namespaced_fraud_results_stream: last_id},
            count=10,
            block=timeout_ms
        )
        
        if not response:
            return [], last_id
            
        stream_name, messages = response[0]  # Only one stream in our query
        
        results = []
        new_last_id = last_id
        
        for stream_id, message in messages:
            try:
                if b'data' in message:
                    result = json.loads(message[b'data'].decode())
                    result['stream_id'] = stream_id.decode()
                    
                    # Add run_id from stream metadata if not in result
                    if 'run_id' not in result and b'run_id' in message:
                        result['run_id'] = message[b'run_id'].decode()
                    
                    # Filter by run_id if specified
                    message_run_id = result.get('run_id', 'unknown')
                    if run_id and message_run_id != run_id:
                        new_last_id = stream_id  # Update last ID even for filtered messages
                        continue
                    
                    # Track run-specific latency
                    if 'latency_ms' in result:
                        latency = result['latency_ms']
                        if isinstance(latency, str):
                            try:
                                latency = float(latency)
                            except (ValueError, TypeError):
                                latency = 0
                                
                        # Track in overall latencies
                        self.all_latencies.append(latency)
                        
                        # Track in run-specific latencies
                        if message_run_id not in self.run_latencies:
                            self.run_latencies[message_run_id] = []
                            
                        self.run_latencies[message_run_id].append(latency)
                    
                    results.append(result)
                    new_last_id = stream_id
            except Exception as e:
                print(f"Error processing message: {e}")
                new_last_id = stream_id
                
        return results, new_last_id
    
    def format_result(self, result):
        """Format a single inference result for display"""
        # Format timestamp
        if 'processed_at' in result:
            timestamp = result['processed_at']
        else:
            timestamp = datetime.now().isoformat()
            
        # Format fraud status (this is specific to our fraud detection use case)
        if result.get('is_fraud', 0) == 1:
            fraud_status = "FRAUD"
            status_format = "\033[91m{}\033[0m"  # Red
        else:
            fraud_status = "OK"
            status_format = "\033[92m{}\033[0m"  # Green
            
        # Format probability
        prob = result.get('fraud_probability', 0)
        prob_str = f"{prob:.4f}"
        
        # Get worker ID if available
        worker_id = result.get('worker_id', 'unknown')
        
        # Format latency with color coding
        latency_ms = result.get('latency_ms', 0)
        if isinstance(latency_ms, str):
            try:
                latency_ms = float(latency_ms)
            except (ValueError, TypeError):
                latency_ms = 0
                
        # Color code latency
        if latency_ms > 100:
            lat_format = "\033[91m{:.2f}ms\033[0m"  # Red for >100ms
        elif latency_ms > 50:
            lat_format = "\033[93m{:.2f}ms\033[0m"  # Yellow for >50ms
        else:
            lat_format = "\033[92m{:.2f}ms\033[0m"  # Green for ≤50ms
            
        latency_str = lat_format.format(latency_ms)
        
        # Get run ID
        run_id = result.get('run_id', 'unknown')
        
        # Format the line
        line = (f"{timestamp[11:19]} | "
                f"Transaction: {result.get('transaction_id', 'unknown')} | "
                f"User: {result.get('user_id', 'unknown')} | "
                f"Amount: ${result.get('amount', 0):.2f} | "
                f"Status: {status_format.format(fraud_status)} | "
                f"Probability: {prob_str} | "
                f"Worker: {worker_id} | "
                f"Latency: {latency_str} | "
                f"Run: {run_id}")
                
        return line
    
    def display_results(self, results):
        """Display a list of inference results"""
        if not results:
            print("No results found")
            return
            
        # Print header
        print("\n" + "=" * 120)
        print("ML INFERENCE RESULTS: FRAUD DETECTION")
        print("=" * 120)
        print("Time    | Transaction ID        | User ID | Amount      | Status | Probability | Worker ID | Latency     | Run ID")
        print("-" * 120)
        
        # Print results
        for result in results:
            print(self.format_result(result))
            
        print("=" * 120 + "\n")
    
    def display_run_stats(self, run_id=None):
        """
        Display statistics for available runs or a specific run
        
        Args:
            run_id: Optional specific run ID to display
        """
        if run_id:
            # Display stats for a single run
            stats = self.get_run_stats(run_id)
            if not stats:
                print(f"No statistics found for run: {run_id}")
                return
                
            # Print detailed stats for this run
            print("\n" + "=" * 100)
            print(f"RUN STATISTICS: {run_id}")
            print("=" * 100)
            print(f"Start time: {stats['start_time']}")
            if stats['end_time']:
                print(f"End time: {stats['end_time']}")
            print(f"Status: {'Complete' if stats['complete'] else 'In Progress'}")
            print(f"Pattern: {stats['pattern']}")
            print(f"Transactions: {stats['transaction_count']}")
            print(f"Target TPS: {stats['target_tps']:.1f}")
            print(f"Actual TPS: {stats['actual_tps']:.1f}")
            print(f"Duration: {stats['duration_seconds']:.1f} seconds")
            print(f"Fraud ratio: {stats['fraud_ratio']:.2f}")
            print(f"Workers: {stats['workers']} ({', '.join(stats['worker_ids'])})")
            print(f"\nLatency Metrics:")
            print(f"  Average: {stats['avg_latency_ms']:.2f} ms")
            print(f"  p50 (median): {stats['p50_latency_ms']:.2f} ms")
            print(f"  p95: {stats['p95_latency_ms']:.2f} ms")
            print(f"  p99: {stats['p99_latency_ms']:.2f} ms")
            print("=" * 100 + "\n")
        else:
            # List all available runs
            run_ids = self.get_run_ids()
            if not run_ids:
                print("No runs found")
                return
                
            # Print summary of all runs
            print("\n" + "=" * 100)
            print("AVAILABLE RUNS")
            print("=" * 100)
            print("Run ID                       | Status    | Transactions | Avg Latency | TPS   | Pattern")
            print("-" * 100)
            
            for run_id in run_ids:
                stats = self.get_run_stats(run_id)
                if stats:
                    status = "Complete" if stats['complete'] else "In Progress"
                    print(f"{run_id} | {status:9} | {stats['transaction_count']:12} | "
                          f"{stats['avg_latency_ms']:9.2f}ms | {stats['actual_tps']:5.1f} | {stats['pattern']}")
            print("=" * 100 + "\n")
    
    def run_live_monitor(self, refresh_interval=2.0, run_id=None):
        """
        Run a live monitor that continuously displays new inference results
        
        Args:
            refresh_interval: Time in seconds to wait between updates
            run_id: Optional run ID to filter results by
        """
        self.running = True
        print(f"Starting inference results monitor for stream: {settings.namespaced_fraud_results_stream}")
        if run_id:
            print(f"Filtering for run ID: {run_id}")
        print("Press Ctrl+C to stop monitoring")
        
        # Initialize timing
        self.start_time = time()
        self.run_latencies = {}
        
        # First, display the most recent results
        latest_results = self.get_latest_results(count=5, run_id=run_id)  # Show 5 most recent results to start
        if latest_results:
            print("\nShowing most recent results...")
            self.display_results(latest_results)
            last_id = latest_results[0]['stream_id']  # Start watching from the most recent
        else:
            print("No recent results found. Waiting for new transactions...")
            last_id = '0-0'  # Start from beginning if no results
            
        try:
            while self.running:
                # Watch for new results
                new_results, last_id = self.watch_stream(last_id, int(refresh_interval * 1000), run_id)
                
                if new_results:
                    self.display_results(new_results)
                    
        except KeyboardInterrupt:
            print("\nStopping monitor")
            self.running = False

# Run monitor if script is executed directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor ML inference results")
    parser.add_argument('--latest', action='store_true', help='Show latest results and exit')
    parser.add_argument('--count', type=int, default=10, help='Number of results to display')
    parser.add_argument('--interval', type=float, default=2.0, help='Refresh interval in seconds (for live monitor)')
    parser.add_argument('--run-id', type=str, help='Filter results by specific run ID')
    parser.add_argument('--runs', action='store_true', help='Show available runs')
    
    args = parser.parse_args()
    
    monitor = ResultMonitor()
    
    if args.runs:
        # Show run statistics
        monitor.display_run_stats(args.run_id)
    elif args.latest:
        # Show latest results and exit
        results = monitor.get_latest_results(count=args.count, run_id=args.run_id)
        monitor.display_results(results)
    else:
        # Run live monitor
        monitor.run_live_monitor(refresh_interval=args.interval, run_id=args.run_id) 