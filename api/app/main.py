from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# Import visualization API
from .visualization import visualization_api, VisualizationRequest

app = FastAPI(
    title="CinderSight API",
    description="Canadian Fire Prediction API with Visualization Capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    ignition_point: Dict[str, float]
    date: str

@app.get("/")
async def root():
    return {
        "message": "CinderSight API", 
        "status": "running",
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
    return {"status": "healthy"}


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
