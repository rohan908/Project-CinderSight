#!/usr/bin/env python3
"""
Test runner script for CinderSight API
Usage: python run_tests.py [options]
"""

import sys
import subprocess
import argparse

def run_tests(args):
    """Run pytest with the given arguments."""
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    if args.fast:
        cmd.append("-m")
        cmd.append("not slow")
    
    if args.unit:
        cmd.append("-m")
        cmd.append("unit")
    
    if args.integration:
        cmd.append("-m")
        cmd.append("integration")
    
    if args.watch:
        cmd.append("--watch")
    
    # Add test files if specified
    if args.files:
        cmd.extend(args.files)
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run CinderSight API tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-c", "--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("-f", "--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("-u", "--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("-i", "--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("-w", "--watch", action="store_true", help="Watch for changes")
    parser.add_argument("files", nargs="*", help="Specific test files to run")
    
    args = parser.parse_args()
    
    result = run_tests(args)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 