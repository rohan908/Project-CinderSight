import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data for testing."""
    return {
        "ignition_point": {
            "latitude": 56.1304,
            "longitude": -106.3468
        },
        "date": "2024-07-15"
    }

@pytest.fixture
def sample_prediction_response():
    """Sample prediction response data for testing."""
    return {
        "prediction": {
            "risk_level": "medium",
            "probability": 0.5,
            "spread_direction": "NE",
            "estimated_area": 5.0,
            "confidence": 0.7
        }
    } 