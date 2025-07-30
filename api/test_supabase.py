#!/usr/bin/env python3
"""
Test script for Supabase integration
Run this script to verify that your Supabase configuration is working correctly.
"""

import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

def test_supabase_connection():
    """Test Supabase connection and basic operations"""
    try:
        from env_config import EnvConfig
        from supabase_client import get_supabase_manager
        
        print("=== Testing Supabase Integration ===")
        
        # Print configuration
        EnvConfig.print_config()
        
        # Validate configuration
        if not EnvConfig.validate_supabase_config():
            print("❌ Supabase configuration is incomplete!")
            return False
        
        # Initialize Supabase manager
        print("\n🔌 Initializing Supabase manager...")
        supabase_manager = get_supabase_manager()
        print("✅ Supabase manager initialized successfully")
        
        # Test model paths retrieval
        print("\n📋 Testing model paths retrieval...")
        try:
            model_paths = supabase_manager.get_model_paths()
            print(f"✅ Retrieved {len(model_paths)} model paths")
            for name, link in model_paths.items():
                print(f"   - {name}: {link[:50]}...")
        except Exception as e:
            print(f"❌ Error getting model paths: {e}")
            return False
        
        # Test data paths retrieval
        print("\n📊 Testing data paths retrieval...")
        try:
            data_paths = supabase_manager.get_data_paths()
            print(f"✅ Retrieved {len(data_paths)} data paths")
            for name, link in data_paths.items():
                print(f"   - {name}: {link[:50]}...")
        except Exception as e:
            print(f"❌ Error getting data paths: {e}")
            return False
        
        # Test data loading
        print(f"\n📈 Testing data loading for split '{EnvConfig.DEFAULT_DATA_SPLIT}'...")
        try:
            features, targets = supabase_manager.load_ndws_data_from_supabase(EnvConfig.DEFAULT_DATA_SPLIT)
            if features is not None and targets is not None:
                print(f"✅ Data loaded successfully!")
                print(f"   - Features shape: {features.shape}")
                print(f"   - Targets shape: {targets.shape}")
                print(f"   - Available samples: {len(features)}")
            else:
                print("❌ Data loading failed")
                return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        # Test model downloading
        print(f"\n🤖 Testing model downloading for '{EnvConfig.DEFAULT_MODEL_NAME}'...")
        try:
            model_path = supabase_manager.download_model(EnvConfig.DEFAULT_MODEL_NAME)
            print(f"✅ Model downloaded successfully!")
            print(f"   - Model path: {model_path}")
            print(f"   - File exists: {model_path.exists()}")
            print(f"   - File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
        
        print("\n🎉 All tests passed! Supabase integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    print("CinderSight Supabase Integration Test")
    print("=" * 50)
    
    success = test_supabase_connection()
    
    if success:
        print("\n✅ Integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        print("\nPlease check:")
        print("1. Your environment variables are set correctly")
        print("2. Your Supabase project is active")
        print("3. Your database tables exist and have data")
        print("4. Your storage buckets contain the required files")
        sys.exit(1)

if __name__ == "__main__":
    main() 