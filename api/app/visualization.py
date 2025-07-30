import os
import sys
import shutil
import zipfile
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union
from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import asyncio

# No need to add model directory path in Railway deployment

# Import Supabase client and environment config
from .supabase_client import get_supabase_manager
from .env_config import EnvConfig

try:
    # Import from the local generate_sample_visualizations module
    from .generate_sample_visualizations import SampleVisualizationGenerator, generate_single_sample_with_data
except ImportError as e:
    print(f"Warning: Could not import visualization modules: {e}")
    # Create dummy classes for when modules are not available
    class SampleVisualizationGenerator:
        def __init__(self, *args, **kwargs):
            pass
    
    # generate_single_sample function removed - not used in API version
    
    def generate_single_sample_with_data(*args, **kwargs):
        return None

# Global configuration - use Railway-compatible path
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'visualizations')

class VisualizationRequest(BaseModel):
    sample_idx: int
    save_images: bool = True  # Whether to save images to disk
    overwrite_existing: bool = True  # Whether to overwrite existing files
    include_features: bool = True
    include_fire_progression: bool = True
    include_metrics_dashboard: bool = True
    include_documentation: bool = True

class VisualizationResponse(BaseModel):
    sample_idx: int
    status: str
    task_id: str
    output_directory: Optional[str] = None
    files_generated: List[str] = []
    metrics: Optional[Dict] = None
    error_message: Optional[str] = None

# Store active tasks
active_tasks = {}

class VisualizationAPI:
    """API for generating wildfire visualizations"""
    
    def __init__(self):
        self.base_output_dir = BASE_OUTPUT_DIR
        self.supabase_manager = get_supabase_manager()
        
    def validate_paths(self):
        """Validate that required paths exist and Supabase is accessible"""
        try:
            print(f"📁 Base output directory: {self.base_output_dir}")
            print(f"📁 Current working directory: {os.getcwd()}")
            
            # Test Supabase connection by getting model paths
            self.supabase_manager.get_model_paths()
            
            # Create output directory if it doesn't exist
            print(f"📁 Creating output directory: {self.base_output_dir}")
            os.makedirs(self.base_output_dir, exist_ok=True)
            print(f"✅ Output directory created/verified")
            
        except Exception as e:
            print(f"❌ Error in validate_paths: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Path validation failed: {str(e)}"
            )
    
    async def generate_visualizations(self, request: VisualizationRequest, background_tasks: BackgroundTasks):
        """Start visualization generation for a sample"""
        print(f"🚀 Starting visualization request for sample {request.sample_idx}")
        print(f"📋 Request data: {request.dict()}")
        
        try:
            # Validate sample index
            print(f"🔍 Validating sample index: {request.sample_idx}")
            if request.sample_idx < 0:
                raise HTTPException(status_code=400, detail="Sample index must be non-negative")
            print(f"✅ Sample index validation passed")
            
            # Validate paths
            print(f"🔍 Validating paths and Supabase connection")
            self.validate_paths()
            print(f"✅ Path validation successful")
            
            # Create unique task ID
            task_id = str(uuid.uuid4())
            print(f"🆔 Created task ID: {task_id}")
            
            # Initialize response
            print(f"📝 Initializing response object")
            response = VisualizationResponse(
                sample_idx=request.sample_idx,
                status="processing",
                task_id=task_id,
                output_directory=None,
                files_generated=[],
                metrics=None
            )
            print(f"✅ Response object created")
            
            # Store task
            print(f"💾 Storing task in active_tasks")
            active_tasks[task_id] = {
                "response": response,
                "request": request,
                "completed": False,
                "error": None
            }
            print(f"✅ Task stored successfully")
            
            # Start background task
            print(f"🔄 Adding background task")
            background_tasks.add_task(
                self.process_visualization_task, 
                task_id, 
                request
            )
            print(f"✅ Background task added")
            
            response_data = {
                "task_id": task_id,
                "sample_idx": request.sample_idx,
                "status": "processing",
                "message": f"Visualization generation started for sample {request.sample_idx}",
                "check_status_url": f"/visualization/status/{task_id}",
                "download_url": f"/visualization/download/{task_id}"
            }
            print(f"✅ Returning response: {response_data}")
            return response_data
            
        except Exception as e:
            print(f"❌ Error in generate_visualizations: {str(e)}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error starting visualization: {str(e)}")
    
    async def process_visualization_task(self, task_id: str, request: VisualizationRequest):
        """Background task to process visualization generation"""
        try:
            print(f"🔄 Starting visualization task {task_id} for sample {request.sample_idx}")
            
            # Download model from Supabase
            print(f"📥 Downloading model: {EnvConfig.DEFAULT_MODEL_NAME}")
            model_path = self.supabase_manager.download_model(EnvConfig.DEFAULT_MODEL_NAME)
            print(f"✅ Model downloaded to: {model_path}")
            
            # Initialize generator with downloaded model
            print(f"🔧 Initializing visualization generator")
            generator = SampleVisualizationGenerator(str(model_path))
            print(f"✅ Generator initialized")
            
            # Determine output directory
            if request.save_images:
                if request.overwrite_existing:
                    # Use a fixed directory for the sample to overwrite
                    output_dir = Path(self.base_output_dir) / f"sample_{request.sample_idx}"
                else:
                    # Use unique directory
                    output_dir = Path(self.base_output_dir) / f"sample_{request.sample_idx}_{task_id[:8]}"
            else:
                # Use temporary directory
                output_dir = Path(tempfile.mkdtemp())
            
            # Load data from Supabase for visualization
            print(f"📊 Loading data from Supabase for split: {EnvConfig.DEFAULT_DATA_SPLIT}")
            features, targets = self.supabase_manager.load_ndws_data_from_supabase(EnvConfig.DEFAULT_DATA_SPLIT)
            if features is None or targets is None:
                raise Exception("Could not load data from Supabase")
            print(f"✅ Data loaded - features shape: {features.shape}, targets shape: {targets.shape}")
            
            # Create temporary data directory structure for the visualization function
            temp_data_dir = Path(tempfile.mkdtemp()) / "processed"
            temp_data_dir.mkdir(exist_ok=True)
            
            # Save data files in the expected format
            import pickle
            with open(temp_data_dir / f"{EnvConfig.DEFAULT_DATA_SPLIT}.data", 'wb') as f:
                pickle.dump(features, f)
            with open(temp_data_dir / f"{EnvConfig.DEFAULT_DATA_SPLIT}.labels", 'wb') as f:
                pickle.dump(targets, f)
            
            # Generate visualizations with loaded data
            print(f"🎨 Generating visualizations for sample {request.sample_idx}")
            result = generate_single_sample_with_data(
                request.sample_idx, 
                generator, 
                features, 
                targets,
                str(output_dir)
            )
            print(f"✅ Visualization generation completed")
            
            # Clean up temporary data directory
            import shutil
            shutil.rmtree(temp_data_dir.parent)
            
            if result is None:
                raise Exception(f"Failed to generate visualizations for sample {request.sample_idx}")
            
            # Update task status
            active_tasks[task_id]["response"].status = "completed"
            active_tasks[task_id]["response"].output_directory = str(result["output_dir"])
            active_tasks[task_id]["response"].files_generated = self.get_generated_files(Path(result["output_dir"]))
            active_tasks[task_id]["response"].metrics = result["metrics"]
            active_tasks[task_id]["completed"] = True
            
        except Exception as e:
            print(f"❌ Error in visualization task {task_id}: {str(e)}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            active_tasks[task_id]["response"].status = "failed"
            active_tasks[task_id]["response"].error_message = str(e)
            active_tasks[task_id]["error"] = str(e)
    
    def get_generated_files(self, output_dir: Path) -> List[str]:
        """Get list of generated files from output directory"""
        files = []
        if output_dir.exists():
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    files.append(file_path.name)
        return files
    
    def get_task_status(self, task_id: str):
        """Get the status of a visualization task"""
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = active_tasks[task_id]
        response = task["response"]
        
        return {
            "task_id": task_id,
            "sample_idx": response.sample_idx,
            "status": response.status,
            "output_directory": response.output_directory,
            "files_generated": response.files_generated,
            "metrics": response.metrics,
            "error_message": response.error_message,
            "completed": task["completed"]
        }
    
    def download_visualizations(self, task_id: str):
        """Download generated visualizations as a ZIP file"""
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = active_tasks[task_id]
        if not task["completed"]:
            raise HTTPException(status_code=400, detail="Task not completed yet")
        
        if task["error"]:
            raise HTTPException(status_code=500, detail=f"Task failed: {task['error']}")
        
        output_dir = Path(task["response"].output_directory)
        if not output_dir.exists():
            raise HTTPException(status_code=404, detail="Output directory not found")
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        temp_zip.close()
        
        try:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_dir.iterdir():
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)
            
            # Return ZIP file
            return FileResponse(
                temp_zip.name,
                media_type="application/zip",
                filename=f"sample_{task['response'].sample_idx}_visualizations.zip"
            )
            
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_zip.name):
                os.unlink(temp_zip.name)
            raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")
    
    def download_metrics(self, task_id: str):
        """Download only the metrics JSON file"""
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = active_tasks[task_id]
        if not task["completed"]:
            raise HTTPException(status_code=400, detail="Task not completed yet")
        
        if task["error"]:
            raise HTTPException(status_code=500, detail=f"Task failed: {task['error']}")
        
        output_dir = Path(task["response"].output_directory)
        metrics_file = output_dir / "sample_metrics.json"
        
        if not metrics_file.exists():
            raise HTTPException(status_code=404, detail="Metrics file not found")
        
        return FileResponse(
            metrics_file,
            media_type="application/json",
            filename=f"sample_{task['response'].sample_idx}_metrics.json"
        )
    
    def download_features(self, task_id: str):
        """Download only the feature PNG files"""
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = active_tasks[task_id]
        if not task["completed"]:
            raise HTTPException(status_code=400, detail="Task not completed yet")
        
        if task["error"]:
            raise HTTPException(status_code=500, detail=f"Task failed: {task['error']}")
        
        output_dir = Path(task["response"].output_directory)
        
        # Create temporary ZIP file with only feature images
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        temp_zip.close()
        
        try:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_dir.iterdir():
                    if file_path.is_file() and file_path.name.startswith("feature_"):
                        zipf.write(file_path, file_path.name)
            
            return FileResponse(
                temp_zip.name,
                media_type="application/zip",
                filename=f"sample_{task['response'].sample_idx}_features.zip"
            )
            
        except Exception as e:
            if os.path.exists(temp_zip.name):
                os.unlink(temp_zip.name)
            raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")
    
    def get_available_samples(self):
        """Get information about available samples"""
        try:
            # Load data from Supabase to get sample count
            features, targets = self.supabase_manager.load_ndws_data_from_supabase(EnvConfig.DEFAULT_DATA_SPLIT)
            
            if features is None or targets is None:
                # Try train data as fallback
                features, targets = self.supabase_manager.load_ndws_data_from_supabase("train")
            
            if features is None or targets is None:
                return {
                    "total_samples": 0,
                    "message": "No data available from Supabase"
                }
            
            return {
                "total_samples": len(features),
                "data_shape": features.shape,
                "message": f"Available samples: 0 to {len(features) - 1}"
            }
            
        except Exception as e:
            return {
                "total_samples": 0,
                "error": str(e),
                "message": "Could not determine available samples from Supabase"
            }
    
    def delete_task(self, task_id: str):
        """Delete a completed task and clean up files"""
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = active_tasks[task_id]
        
        # Clean up output directory if task is completed and using temp directory
        if task["completed"] and task["response"].output_directory:
            try:
                output_dir = Path(task["response"].output_directory)
                if output_dir.exists() and "tmp" in str(output_dir):
                    shutil.rmtree(output_dir)
            except Exception as e:
                print(f"Warning: Could not clean up directory {output_dir}: {e}")
        
        # Remove task from memory
        del active_tasks[task_id]
        
        return {"message": f"Task {task_id} deleted successfully"}
    
    def delete_all_tasks(self):
        """Delete all completed tasks and clean up files"""
        deleted_count = 0
        
        for task_id, task in list(active_tasks.items()):
            if task["completed"] and task["response"].output_directory:
                try:
                    output_dir = Path(task["response"].output_directory)
                    if output_dir.exists() and "tmp" in str(output_dir):
                        shutil.rmtree(output_dir)
                    deleted_count += 1
                except Exception as e:
                    print(f"Warning: Could not clean up directory {output_dir}: {e}")
        
        # Clear all tasks
        active_tasks.clear()
        
        return {"message": f"Deleted {deleted_count} completed tasks"}

# Create global instance
visualization_api = VisualizationAPI() 