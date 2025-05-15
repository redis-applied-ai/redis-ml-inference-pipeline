#!/usr/bin/env python
"""Format code using black and isort."""
import subprocess
import sys

def main():
    """Format all Python files in the pipeline directory."""
    print("Formatting code with black...")
    black_result = subprocess.run(["black", "pipeline"], check=False)
    
    print("Sorting imports with isort...")
    isort_result = subprocess.run(["isort", "pipeline"], check=False)
    
    if black_result.returncode != 0 or isort_result.returncode != 0:
        print("Formatting failed!")
        sys.exit(1)
    else:
        print("Formatting completed successfully!")

if __name__ == "__main__":
    main() 