from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(
    title="CinderSight API",
    description="Canadian Fire Prediction API",
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
    return {"message": "CinderSight API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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
