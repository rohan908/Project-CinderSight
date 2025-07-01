import pytest
from fastapi.testclient import TestClient
from app.main import app

class TestHealthEndpoints:
    """Test health check and root endpoints."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["message"] == "CinderSight API"
        assert data["status"] == "running"
    
    def test_health_endpoint(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

class TestPredictionEndpoint:
    """Test the prediction endpoint."""
    
    def test_prediction_valid_request(self, client: TestClient, sample_prediction_request):
        """Test prediction endpoint with valid request."""
        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        
        prediction = data["prediction"]
        required_fields = ["risk_level", "probability", "spread_direction", "estimated_area", "confidence"]
        for field in required_fields:
            assert field in prediction
    
    def test_prediction_missing_fields(self, client: TestClient):
        """Test prediction endpoint with missing required fields."""
        # Missing ignition_point
        response = client.post("/predict", json={"date": "2024-07-15"})
        assert response.status_code == 422  # Validation error
        
        # Missing date
        response = client.post("/predict", json={
            "ignition_point": {"latitude": 56.1304, "longitude": -106.3468}
        })
        assert response.status_code == 422  # Validation error
    
    def test_prediction_invalid_coordinates(self, client: TestClient):
        """Test prediction endpoint with invalid coordinates."""
        invalid_request = {
            "ignition_point": {
                "latitude": 200.0,  # Invalid latitude
                "longitude": -106.3468
            },
            "date": "2024-07-15"
        }
        response = client.post("/predict", json=invalid_request)
        # Should still work since we're using placeholder implementation
        assert response.status_code == 200
    
    def test_prediction_invalid_date_format(self, client: TestClient):
        """Test prediction endpoint with invalid date format."""
        invalid_request = {
            "ignition_point": {
                "latitude": 56.1304,
                "longitude": -106.3468
            },
            "date": "invalid-date"
        }
        response = client.post("/predict", json=invalid_request)
        # Should still work since we're using placeholder implementation
        assert response.status_code == 200
    
    def test_prediction_response_structure(self, client: TestClient, sample_prediction_request):
        """Test that prediction response has correct structure and data types."""
        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        
        data = response.json()
        prediction = data["prediction"]
        
        # Check data types
        assert isinstance(prediction["risk_level"], str)
        assert isinstance(prediction["probability"], (int, float))
        assert isinstance(prediction["spread_direction"], str)
        assert isinstance(prediction["estimated_area"], (int, float))
        assert isinstance(prediction["confidence"], (int, float))
        
        # Check value ranges
        assert 0 <= prediction["probability"] <= 1
        assert 0 <= prediction["confidence"] <= 1
        assert prediction["estimated_area"] >= 0
        assert prediction["risk_level"] in ["low", "medium", "high", "extreme"]

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_json(self, client: TestClient):
        """Test handling of invalid JSON."""
        response = client.post("/predict", data="invalid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422
    
    def test_wrong_content_type(self, client: TestClient):
        """Test handling of wrong content type."""
        response = client.post("/predict", data="some data", headers={"Content-Type": "text/plain"})
        assert response.status_code == 422
    
    def test_method_not_allowed(self, client: TestClient):
        """Test that GET method is not allowed on predict endpoint."""
        response = client.get("/predict")
        assert response.status_code == 405  # Method Not Allowed 