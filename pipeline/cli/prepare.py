#!/usr/bin/env python
import argparse
from pipeline.prepare import prepare_system

def main():
    """CLI entry point for prepare command"""
    parser = argparse.ArgumentParser(description="Prepare ML inference pipeline")
    parser.add_argument('--skip-model', action='store_true', help='Skip model training')
    parser.add_argument('--skip-redis', action='store_true', help='Skip Redis initialization')
    parser.add_argument('--description', type=str, default="Initial model", help='Model version description')
    
    args = parser.parse_args()
    
    prepare_system(
        train_model=not args.skip_model,
        init_redis=not args.skip_redis,
        model_description=args.description
    )

if __name__ == "__main__":
    main() 