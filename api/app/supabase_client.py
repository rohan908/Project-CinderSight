import os
import tempfile
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manages Supabase operations for models and data"""
    
    def __init__(self):
        """Initialize Supabase client with environment variables"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY not set. Supabase features will be disabled.")
            self.client = None
            self.temp_dir = Path(tempfile.gettempdir()) / "cindersight"
            self.temp_dir.mkdir(exist_ok=True)
            return
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        self.temp_dir = Path(tempfile.gettempdir()) / "cindersight"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Supabase client initialized with temp directory: {self.temp_dir}")
    
    def get_bucket_url(self, bucket_name: str, file_path: str) -> str:
        """Get the public URL for a file in a bucket"""
        try:
            # Get the public URL from the bucket
            response = self.client.storage.from_(bucket_name).get_public_url(file_path)
            return response
        except Exception as e:
            logger.error(f"Error getting bucket URL for {bucket_name}/{file_path}: {e}")
            raise
    
    def download_file(self, bucket_name: str, file_path: str, local_path: Optional[Path] = None) -> Path:
        """Download a file from Supabase bucket to local storage"""
        try:
            if local_path is None:
                local_path = self.temp_dir / Path(file_path).name
            
            # Download the file
            response = self.client.storage.from_(bucket_name).download(file_path)
            
            # Write to local file
            with open(local_path, 'wb') as f:
                f.write(response)
            
            logger.info(f"Downloaded {bucket_name}/{file_path} to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading {bucket_name}/{file_path}: {e}")
            raise
    
    def download_from_url(self, url: str, local_path: Optional[Path] = None) -> Path:
        """Download a file from a signed URL"""
        try:
            import requests
            
            if local_path is None:
                local_path = self.temp_dir / Path(url.split('/')[-1].split('?')[0])
            
            # Download the file from the signed URL
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Write to local file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded from URL to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading from URL {url}: {e}")
            raise
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get model file paths from Supabase table"""
        if self.client is None:
            logger.warning("Supabase client not initialized. Cannot get model paths.")
            return {}
            
        try:
            # Query the models table to get bucket links
            response = self.client.table('models').select('*').execute()
            
            if not response.data:
                raise ValueError("No models found in database")
            
            model_paths = {}
            for model in response.data:
                model_name = model.get('name', 'unknown')
                model_path = model.get('model_path', '')
                if model_path:
                    model_paths[model_name] = model_path
            
            logger.info(f"Retrieved {len(model_paths)} model paths from database")
            return model_paths
            
        except Exception as e:
            logger.error(f"Error getting model paths: {e}")
            raise
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get data file paths from Supabase table"""
        if self.client is None:
            logger.warning("Supabase client not initialized. Cannot get data paths.")
            return {}
            
        try:
            # Query the samples table to get bucket links
            response = self.client.table('samples').select('*').execute()
            
            if not response.data:
                raise ValueError("No samples found in database")
            
            data_paths = {}
            for sample in response.data:
                # Get features file path
                features_file_path = sample.get('features_file_path', '')
                if features_file_path:
                    # Extract the split name from the path (e.g., "test.data" -> "test")
                    split_name = features_file_path.split('.')[0] if '.' in features_file_path else 'unknown'
                    data_paths[f"{split_name}.data"] = features_file_path
                
                # Get target file path
                target_file_path = sample.get('target_file_path', '')
                if target_file_path:
                    # Extract the split name from the path (e.g., "test.labels" -> "test")
                    split_name = target_file_path.split('.')[0] if '.' in target_file_path else 'unknown'
                    data_paths[f"{split_name}.labels"] = target_file_path
            
            logger.info(f"Retrieved {len(data_paths)} data paths from database")
            return data_paths
            
        except Exception as e:
            logger.error(f"Error getting data paths: {e}")
            raise
    
    def download_model(self, model_name: str = "model_nfp.pth") -> Path:
        """Download a specific model file"""
        try:
            model_paths = self.get_model_paths()
            
            if model_name not in model_paths:
                available_models = list(model_paths.keys())
                raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
            
            model_path = model_paths[model_name]
            
            # Handle signed URLs directly
            if model_path.startswith('http'):
                local_path = self.temp_dir / model_name
                return self.download_from_url(model_path, local_path)
            else:
                # Fallback to bucket download if it's a bucket path
                parts = model_path.split('/')
                bucket_name = parts[2].split('.')[0]  # Extract bucket name
                file_path = '/'.join(parts[3:])  # Extract file path
                
                local_path = self.temp_dir / model_name
                return self.download_file(bucket_name, file_path, local_path)
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            raise
    
    def download_data_files(self, split: str = "test") -> Tuple[Path, Path]:
        """Download data files (test.data and test.labels)"""
        try:
            data_paths = self.get_data_paths()
            
            # Look for the specific data files
            data_file_name = f"{split}.data"
            labels_file_name = f"{split}.labels"
            
            if data_file_name not in data_paths:
                available_data = list(data_paths.keys())
                raise ValueError(f"Data file '{data_file_name}' not found. Available files: {available_data}")
            
            if labels_file_name not in data_paths:
                available_data = list(data_paths.keys())
                raise ValueError(f"Labels file '{labels_file_name}' not found. Available files: {available_data}")
            
            # Download both files
            data_url = data_paths[data_file_name]
            labels_url = data_paths[labels_file_name]
            
            # Download files using signed URLs
            data_local_path = self.temp_dir / data_file_name
            labels_local_path = self.temp_dir / labels_file_name
            
            data_path = self.download_from_url(data_url, data_local_path)
            labels_path = self.download_from_url(labels_url, labels_local_path)
            
            return data_path, labels_path
            
        except Exception as e:
            logger.error(f"Error downloading data files for split '{split}': {e}")
            raise
    
    def load_ndws_data_from_supabase(self, split: str = "test") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load NDWS data from Supabase buckets"""
        try:
            # Download data files
            data_path, labels_path = self.download_data_files(split)
            
            # Load features (19 input channels)
            with open(data_path, 'rb') as f:
                features = pickle.load(f)
            
            # Load labels (FireMask)
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)
            
            logger.info(f"Loaded data from Supabase: features shape {features.shape}, labels shape {labels.shape}")
            return features, labels
            
        except Exception as e:
            logger.error(f"Error loading NDWS data from Supabase: {e}")
            return None, None
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")

# Global instance
supabase_manager = None

def get_supabase_manager() -> SupabaseManager:
    """Get or create the global Supabase manager instance"""
    global supabase_manager
    if supabase_manager is None:
        supabase_manager = SupabaseManager()
    return supabase_manager 