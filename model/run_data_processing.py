#!/usr/bin/env python3
"""
Enhanced NDWS Dataset Processing Runner (PyTorch)

This script provides a unified interface for running data visualization and cleaning
for the Enhanced Next Day Wildfire Spread (NDWS) dataset.
"""

import argparse
import sys
import importlib.util
from pathlib import Path
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'seaborn', 'pathlib', 'tfrecord'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All required packages are installed.")
    return True

def validate_paths(data_dir: str, vis_output_dir: str, clean_output_dir: str):
    """Validate and prepare all paths used by the processing scripts"""
    paths = {}
    
    print("Validating paths...")
    print("=" * 40)
    
    # Validate input data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Input data directory does not exist: {data_path}")
        print("Please ensure the data directory exists and contains TFRecord files.")
        return None
    
    if not data_path.is_dir():
        print(f"ERROR: Data path is not a directory: {data_path}")
        return None
    
    # Check for TFRecord files
    tfrecord_files = list(data_path.glob("*.tfrecord"))
    if not tfrecord_files:
        print(f"ERROR: No TFRecord files found in: {data_path}")
        print("Please ensure your data directory contains .tfrecord files")
        return None
        
    print(f"SUCCESS: Input data directory: {data_path.absolute()}")
    print(f"   Found {len(tfrecord_files)} TFRecord files")
    
    # Prepare visualization output directory
    vis_output_path = Path(vis_output_dir)
    try:
        vis_output_path.mkdir(parents=True, exist_ok=True)
        print(f"SUCCESS: Visualization output directory: {vis_output_path.absolute()}")
    except Exception as e:
        print(f"ERROR: Cannot create visualization output directory: {vis_output_path}")
        print(f"   Error: {e}")
        return None
    
    # Prepare data cleaning output directory  
    clean_output_path = Path(clean_output_dir)
    try:
        clean_output_path.mkdir(parents=True, exist_ok=True)
        print(f"SUCCESS: Data processing output directory: {clean_output_path.absolute()}")
    except Exception as e:
        print(f"ERROR: Cannot create data processing output directory: {clean_output_path}")
        print(f"   Error: {e}")
        return None
    
    paths = {
        'data_dir': str(data_path.absolute()),
        'vis_output_dir': str(vis_output_path.absolute()),
        'clean_output_dir': str(clean_output_path.absolute())
    }
    
    print("SUCCESS: All paths validated successfully!")
    return paths

def run_visualization(data_dir: str, output_dir: str):
    """Run the data visualization script"""
    print("\n" + "=" * 60)
    print("RUNNING DATA VISUALIZATION")
    print("=" * 60)
    
    try:
        # Import the visualization module
        from data_visualizer import main as visualize_main
        
        # Run visualization with validated paths
        success = visualize_main(data_dir=data_dir, output_dir=output_dir)
        
        if success:
            print("SUCCESS: Data visualization completed successfully!")
            return True
        else:
            print("ERROR: Data visualization failed!")
            return False
            
    except ImportError as e:
        print(f"ERROR: Cannot import data_visualizer: {e}")
        print("Make sure data_visualizer.py is in the same directory")
        return False
    except Exception as e:
        print(f"ERROR: Error running visualization: {e}")
        return False

def run_data_cleaning(data_dir: str, output_dir: str):
    """Run the data cleaning/processing script"""
    print("\n" + "=" * 60)
    print("RUNNING DATA CLEANING AND PROCESSING")
    print("=" * 60)
    
    try:
        # Import the data cleaning module
        from data_cleaner import main as clean_main
        
        # Run data cleaning with validated paths
        success = clean_main(data_dir=data_dir, output_dir=output_dir)
        
        if success:
            print("SUCCESS: Data cleaning and processing completed successfully!")
            return True
        else:
            print("ERROR: Data cleaning and processing failed!")
            return False
            
    except ImportError as e:
        print(f"ERROR: Cannot import enhanced_data_cleaner: {e}")
        print("Make sure enhanced_data_cleaner.py is in the same directory")
        return False
    except Exception as e:
        print(f"ERROR: Error running data cleaning: {e}")
        return False

def main():
    """Main function to orchestrate data processing"""
    parser = argparse.ArgumentParser(
        description="Enhanced NDWS Dataset Processing Runner (PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if all requirements are installed
  python run_data_processing.py --check
  
  # Run only visualization
  python run_data_processing.py --visualize
  
  # Run only data cleaning
  python run_data_processing.py --clean
  
  # Run both visualization and cleaning
  python run_data_processing.py --visualize --clean
  
  # Specify custom paths
  python run_data_processing.py --visualize --clean \\
    --data-dir "custom/data/path" \\
    --vis-output "custom/vis/output" \\
    --clean-output "custom/processed/output"
        """
    )
    
    parser.add_argument('--check', action='store_true',
                        help='Check if all required packages are installed')
    parser.add_argument('--visualize', action='store_true',
                        help='Run data visualization')
    parser.add_argument('--clean', action='store_true',
                        help='Run data cleaning and processing')
    parser.add_argument('--data-dir', default='data/raw',
                        help='Input directory containing TFRecord files (default: data/raw)')
    parser.add_argument('--vis-output', default='visualizations',
                        help='Output directory for visualizations (default: visualizations)')
    parser.add_argument('--clean-output', default='data/processed',
                        help='Output directory for processed data (default: data/processed)')
    
    args = parser.parse_args()
    
    print("Enhanced NDWS Dataset Processing Runner (PyTorch)")
    print("=" * 60)
    
    # If no arguments provided or only --check, show help
    if not any([args.check, args.visualize, args.clean]):
        parser.print_help()
        return 1
    
    # Check requirements first
    if args.check or args.visualize or args.clean:
        print("Checking requirements...")
        if not check_requirements():
            return 1
    
    if args.check:
        print("SUCCESS: Requirement check completed successfully!")
        if not (args.visualize or args.clean):
            return 0
    
    # Validate paths if we need to run processing
    if args.visualize or args.clean:
        paths = validate_paths(
            data_dir=args.data_dir,
            vis_output_dir=args.vis_output,
            clean_output_dir=args.clean_output
        )
        
        if paths is None:
            print("\nERROR: Path validation failed. Please fix the issues above.")
            return 1
    
    # Track success of operations
    operations_run = []
    operations_successful = []
    
    # Run visualization if requested
    if args.visualize:
        operations_run.append("visualization")
        success = run_visualization(
            data_dir=paths['data_dir'],
            output_dir=paths['vis_output_dir']
        )
        if success:
            operations_successful.append("visualization")
    
    # Run data cleaning if requested
    if args.clean:
        operations_run.append("data cleaning")
        success = run_data_cleaning(
            data_dir=paths['data_dir'],
            output_dir=paths['clean_output_dir']
        )
        if success:
            operations_successful.append("data cleaning")
    
    # Final summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    if len(operations_successful) == len(operations_run):
        print("SUCCESS: All operations completed successfully!")
        print(f"   Operations run: {', '.join(operations_run)}")
        
        if args.visualize:
            print(f"   Visualizations saved to: {paths['vis_output_dir']}")
        if args.clean:
            print(f"   Processed data saved to: {paths['clean_output_dir']}")
        
        return 0
    else:
        failed_operations = [op for op in operations_run if op not in operations_successful]
        print("ERROR: Some operations failed!")
        print(f"   Successful: {', '.join(operations_successful) if operations_successful else 'None'}")
        print(f"   Failed: {', '.join(failed_operations)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 