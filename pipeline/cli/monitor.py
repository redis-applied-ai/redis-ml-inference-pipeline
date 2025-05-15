#!/usr/bin/env python
import argparse
from pipeline.test.result_monitor import ResultMonitor
from pipeline.stats.manager import StatisticsManager
from pipeline.stats.reporting import print_run_summary, print_run_list

def main():
    """CLI entry point for monitor command"""
    parser = argparse.ArgumentParser(description="Monitor ML inference results")
    parser.add_argument('--latest', action='store_true', help='Show latest results and exit')
    parser.add_argument('--count', type=int, default=10, help='Number of results to display')
    parser.add_argument('--interval', type=float, default=2.0, help='Refresh interval in seconds (for live monitor)')
    parser.add_argument('--run-id', type=str, help='Filter results by specific run ID')
    parser.add_argument('--runs', action='store_true', help='List available test runs')
    parser.add_argument('--stats', action='store_true', help='Show detailed stats for run (requires --run-id)')
    parser.add_argument('--legacy', action='store_true', help='Use legacy statistics system')
    
    args = parser.parse_args()
    
    if args.runs or args.stats:
        # Show run statistics using the new statistics system
        if not args.legacy:
            if args.stats:
                # If no run ID is specified, get the most recent one
                run_id = args.run_id
                if not run_id:
                    runs = StatisticsManager.list_runs(limit=1)
                    if runs:
                        run_id = runs[0].get('run_id')
                        print(f"No run ID specified, using most recent run: {run_id}")
                    else:
                        print("No runs found")
                        return
                        
                # Show detailed stats for the run
                run_stats = StatisticsManager.load_run_stats(run_id)
                if run_stats:
                    print_run_summary(run_stats)
                else:
                    print(f"No statistics found for run ID: {run_id}")
            else:
                # List available runs
                runs = StatisticsManager.list_runs(limit=args.count or 10)
                print_run_list(runs)
        else:
            # Use legacy system
            monitor = ResultMonitor()
            monitor.display_run_stats(args.run_id if args.stats else None)
    elif args.latest:
        # Show latest results and exit
        monitor = ResultMonitor()
        results = monitor.get_latest_results(count=args.count, run_id=args.run_id)
        monitor.display_results(results)
    else:
        # Run live monitor
        monitor = ResultMonitor()
        monitor.run_live_monitor(refresh_interval=args.interval, run_id=args.run_id)

if __name__ == "__main__":
    main() 