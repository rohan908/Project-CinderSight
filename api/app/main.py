from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os
from pathlib import Path

# No need to add model directory path in Railway deployment

# Import visualization API and Supabase client
from .visualization import visualization_api, VisualizationRequest
from .supabase_client import get_supabase_manager
from .env_config import EnvConfig

# Global variables for data
available_samples = 0
data_loaded = False

app = FastAPI(
    title="CinderSight API",
    description="Canadian Fire Prediction API with Visualization Capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=EnvConfig.get_allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    ignition_point: Dict[str, float]
    date: str

@app.on_event("startup")
async def startup_event():
    """Load data on server startup"""
    global available_samples, data_loaded
    
    # Print configuration
    EnvConfig.print_config()
    
    # Validate Supabase configuration
    if not EnvConfig.validate_supabase_config():
        print("❌ Supabase configuration is incomplete. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        available_samples = 0
        data_loaded = False
        return
    
    try:
        # Initialize Supabase manager
        supabase_manager = get_supabase_manager()
        
        # Load data from Supabase
        try:
            features, targets = supabase_manager.load_ndws_data_from_supabase(EnvConfig.DEFAULT_DATA_SPLIT)
        except Exception as e:
            print(f"⚠️ Could not load {EnvConfig.DEFAULT_DATA_SPLIT} data from Supabase: {e}")
            # Fallback to train data
            try:
                features, targets = supabase_manager.load_ndws_data_from_supabase("train")
            except Exception as e2:
                print(f"⚠️ Could not load train data from Supabase: {e2}")
                features, targets = None, None
        
        if features is not None and targets is not None:
            available_samples = len(features)
            data_loaded = True
            print(f"✅ Data loaded successfully from Supabase! Available samples: {available_samples}")
        else:
            print("⚠️ Could not load data from Supabase")
            
    except Exception as e:
        print(f"❌ Error initializing Supabase or loading data: {e}")
        available_samples = 0
        data_loaded = False

@app.get("/")
async def root():
    return {
        "message": "CinderSight API", 
        "status": "running",
        "data_loaded": data_loaded,
        "available_samples": available_samples,
        "endpoints": {
            "/predict": "POST - Fire prediction",
            "/visualization/generate": "POST - Generate visualizations",
            "/visualization/status/{task_id}": "GET - Check visualization status",
            "/visualization/download/{task_id}": "GET - Download visualizations",
            "/visualization/samples/available": "GET - Get available samples",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    # Test numpy import
    try:
        import numpy as np
        numpy_status = "ok"
    except ImportError as e:
        numpy_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "data_loaded": data_loaded,
        "available_samples": available_samples,
        "numpy": numpy_status
    }

@app.get("/samples/count")
async def get_sample_count():
    """Get the total number of available samples"""
    return {
        "total_samples": available_samples,
        "data_loaded": data_loaded
    }

@app.get("/visualization/image/{task_id}/{filename}")
async def get_visualization_image(task_id: str, filename: str):
    """Serve individual visualization images"""
    try:
        print(f"🖼️ Requesting image: {task_id}/{filename}")
        
        # Get task status to find the output directory
        task_status = visualization_api.get_task_status(task_id)
        print(f"📊 Task status: {task_status['status']}")
        
        if task_status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Task not completed")
        
        output_dir = Path(task_status["output_directory"])
        
        # The output_directory should already point to the visualizations directory
        # where the images are stored
        image_path = output_dir / filename
        
        print(f"📁 Output directory from task: {output_dir}")
        print(f"📁 Looking for image at: {image_path}")
        print(f"📁 Directory exists: {image_path.parent.exists()}")
        print(f"📁 Image exists: {image_path.exists()}")
        
        if not image_path.exists():
            # List files in directory for debugging
            if image_path.parent.exists():
                files = list(image_path.parent.iterdir())
                print(f"📁 Files in directory: {[f.name for f in files]}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        print(f"✅ Serving image: {image_path}")
        return FileResponse(image_path, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error serving image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictRequest):
    # Placeholder response - no actual implementation yet
    return {
        "prediction": {
            "risk_level": "medium",
            "probability": 0.5,
            "spread_direction": "NE",
            "estimated_area": 5.0,
            "confidence": 0.7
        }
    }

# Visualization endpoints
@app.post("/visualization/generate")
async def generate_visualizations(request: VisualizationRequest, background_tasks: BackgroundTasks):
    """Generate visualizations for a specific sample index"""
    return await visualization_api.generate_visualizations(request, background_tasks)

@app.get("/visualization/status/{task_id}")
async def get_visualization_status(task_id: str):
    """Get the status of a visualization task"""
    return visualization_api.get_task_status(task_id)

@app.get("/visualization/download/{task_id}")
async def download_visualizations(task_id: str):
    """Download all generated visualizations as a ZIP file"""
    return visualization_api.download_visualizations(task_id)

@app.get("/visualization/download/{task_id}/metrics")
async def download_metrics(task_id: str):
    """Download only the metrics JSON file"""
    return visualization_api.download_metrics(task_id)

@app.get("/visualization/download/{task_id}/features")
async def download_features(task_id: str):
    """Download only the feature PNG files"""
    return visualization_api.download_features(task_id)

@app.get("/visualization/samples/available")
async def get_available_samples():
    """Get information about available samples"""
    return visualization_api.get_available_samples()

@app.delete("/visualization/tasks/{task_id}")
async def delete_visualization_task(task_id: str):
    """Delete a completed visualization task and clean up files"""
    return visualization_api.delete_task(task_id)

@app.delete("/visualization/tasks")
async def delete_all_visualization_tasks():
    """Delete all completed visualization tasks and clean up files"""
    return visualization_api.delete_all_tasks()
