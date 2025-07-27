#!/usr/bin/env python3
"""
Example client for the CinderSight Visualization API

This script demonstrates how to use the API to generate visualizations
for specific sample indices and download the results.
"""

import requests
import json
import time
import os
from pathlib import Path

class CinderSightVisualizationClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def check_health(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to API: {e}")
            return None
    
    def get_available_samples(self):
        """Get information about available samples"""
        try:
            response = requests.get(f"{self.base_url}/visualization/samples/available")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting available samples: {e}")
            return None
    
    def generate_visualizations(self, sample_idx, save_images=True, overwrite_existing=True,
                              include_features=True, include_fire_progression=True,
                              include_metrics_dashboard=True, include_documentation=True):
        """Start visualization generation for a sample"""
        payload = {
            "sample_idx": sample_idx,
            "save_images": save_images,
            "overwrite_existing": overwrite_existing,
            "include_features": include_features,
            "include_fire_progression": include_fire_progression,
            "include_metrics_dashboard": include_metrics_dashboard,
            "include_documentation": include_documentation
        }
        
        try:
            response = requests.post(f"{self.base_url}/visualization/generate", json=payload)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error starting visualization: {e}")
            return None
    
    def check_task_status(self, task_id):
        """Check the status of a visualization task"""
        try:
            response = requests.get(f"{self.base_url}/visualization/status/{task_id}")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error checking task status: {e}")
            return None
    
    def wait_for_completion(self, task_id, timeout=300, check_interval=5):
        """Wait for a task to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.check_task_status(task_id)
            
            if status is None:
                print("Error checking task status")
                return None
            
            if status["status"] == "completed":
                print(f"Task completed successfully!")
                return status
            elif status["status"] == "failed":
                print(f"Task failed: {status.get('error_message', 'Unknown error')}")
                return status
            
            print(f"Task status: {status['status']} - Waiting...")
            time.sleep(check_interval)
        
        print(f"Task timed out after {timeout} seconds")
        return None
    
    def download_visualizations(self, task_id, output_dir="downloads"):
        """Download all generated visualizations"""
        try:
            response = requests.get(f"{self.base_url}/visualization/download/{task_id}")
            
            if response.status_code == 200:
                # Create output directory
                Path(output_dir).mkdir(exist_ok=True)
                
                # Save ZIP file
                zip_path = Path(output_dir) / f"sample_visualizations_{task_id[:8]}.zip"
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded visualizations to: {zip_path}")
                return str(zip_path)
            else:
                print(f"Download failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error downloading visualizations: {e}")
            return None
    
    def download_metrics(self, task_id, output_dir="downloads"):
        """Download only the metrics JSON file"""
        try:
            response = requests.get(f"{self.base_url}/visualization/download/{task_id}/metrics")
            
            if response.status_code == 200:
                # Create output directory
                Path(output_dir).mkdir(exist_ok=True)
                
                # Save JSON file
                json_path = Path(output_dir) / f"sample_metrics_{task_id[:8]}.json"
                with open(json_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded metrics to: {json_path}")
                return str(json_path)
            else:
                print(f"Download failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error downloading metrics: {e}")
            return None
    
    def download_features(self, task_id, output_dir="downloads"):
        """Download only the feature PNG files"""
        try:
            response = requests.get(f"{self.base_url}/visualization/download/{task_id}/features")
            
            if response.status_code == 200:
                # Create output directory
                Path(output_dir).mkdir(exist_ok=True)
                
                # Save ZIP file
                zip_path = Path(output_dir) / f"sample_features_{task_id[:8]}.zip"
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded features to: {zip_path}")
                return str(zip_path)
            else:
                print(f"Download failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error downloading features: {e}")
            return None

def main():
    """Example usage of the API client"""
    
    # Initialize client
    client = CinderSightVisualizationClient()
    
    # Check API health
    print("Checking API health...")
    health = client.check_health()
    if health:
        print(f"API Status: {health}")
    else:
        print("API is not running. Please start the server first.")
        return
    
    # Get available samples
    print("\nGetting available samples...")
    samples_info = client.get_available_samples()
    if samples_info:
        print(f"Available samples: {samples_info}")
    
    # Example 1: Generate visualizations without saving images (temporary)
    print(f"\n=== Example 1: Generate visualizations without saving images ===")
    sample_idx = 0
    
    result = client.generate_visualizations(
        sample_idx, 
        save_images=False,  # Don't save to disk
        overwrite_existing=True
    )
    
    if result is None:
        print("Failed to start visualization generation")
        return
    
    task_id = result["task_id"]
    print(f"Task ID: {task_id}")
    
    # Wait for completion
    print("Waiting for completion...")
    status = client.wait_for_completion(task_id)
    
    if status and status["status"] == "completed":
        print(f"Task completed successfully!")
        print(f"Files generated: {len(status['files_generated'])}")
        
        # Download all visualizations
        print("Downloading all visualizations...")
        zip_path = client.download_visualizations(task_id, "downloads_example1")
        
        # Download only metrics
        print("Downloading metrics only...")
        metrics_path = client.download_metrics(task_id, "downloads_example1")
        
        print(f"Download complete!")
        print(f"All files: {zip_path}")
        print(f"Metrics: {metrics_path}")
    
    # Example 2: Generate visualizations and save to disk (overwrite existing)
    print(f"\n=== Example 2: Generate visualizations and save to disk ===")
    
    result2 = client.generate_visualizations(
        sample_idx, 
        save_images=True,  # Save to disk
        overwrite_existing=True  # Overwrite existing files
    )
    
    if result2 is None:
        print("Failed to start visualization generation")
        return
    
    task_id2 = result2["task_id"]
    print(f"Task ID: {task_id2}")
    
    # Wait for completion
    print("Waiting for completion...")
    status2 = client.wait_for_completion(task_id2)
    
    if status2 and status2["status"] == "completed":
        print(f"Task completed successfully!")
        print(f"Files saved to: {status2['output_directory']}")
        print(f"Files generated: {len(status2['files_generated'])}")
        
        # Download only features
        print("Downloading features only...")
        features_path = client.download_features(task_id2, "downloads_example2")
        
        print(f"Download complete!")
        print(f"Features: {features_path}")

if __name__ == "__main__":
    main() 