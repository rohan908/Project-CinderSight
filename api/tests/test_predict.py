import pytest
from app.predict import run_inference

class TestPredictionModule:
    """Test the prediction module functions."""
    
    def test_run_inference_basic(self):
        """Test basic inference functionality."""
        ignition_point = {"latitude": 56.1304, "longitude": -106.3468}
        date = "2024-07-15"
        
        result = run_inference(ignition_point, date)
        
        # Check that result is a dictionary
        assert isinstance(result, dict)
        
        # Check required fields
        required_fields = ["risk_level", "probability", "spread_direction", "estimated_area", "confidence"]
        for field in required_fields:
            assert field in result
    
    def test_run_inference_with_covariates(self):
        """Test inference with covariates parameter."""
        ignition_point = {"latitude": 56.1304, "longitude": -106.3468}
        date = "2024-07-15"
        covariates = {"temperature": 25.0, "humidity": 60.0}
        
        result = run_inference(ignition_point, date, covariates)
        
        # Should still work with covariates
        assert isinstance(result, dict)
        assert "risk_level" in result
    
    def test_run_inference_different_locations(self):
        """Test inference with different locations."""
        locations = [
            {"latitude": 56.1304, "longitude": -106.3468},  # Canada center
            {"latitude": 49.2827, "longitude": -123.1207},  # Vancouver
            {"latitude": 43.6532, "longitude": -79.3832},   # Toronto
        ]
        
        for location in locations:
            result = run_inference(location, "2024-07-15")
            assert isinstance(result, dict)
            assert "risk_level" in result
    
    def test_run_inference_different_dates(self):
        """Test inference with different dates."""
        ignition_point = {"latitude": 56.1304, "longitude": -106.3468}
        dates = ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
        
        for date in dates:
            result = run_inference(ignition_point, date)
            assert isinstance(result, dict)
            assert "risk_level" in result
    
    def test_run_inference_response_structure(self):
        """Test that inference response has correct structure and data types."""
        ignition_point = {"latitude": 56.1304, "longitude": -106.3468}
        date = "2024-07-15"
        
        result = run_inference(ignition_point, date)
        
        # Check data types
        assert isinstance(result["risk_level"], str)
        assert isinstance(result["probability"], (int, float))
        assert isinstance(result["spread_direction"], str)
        assert isinstance(result["estimated_area"], (int, float))
        assert isinstance(result["confidence"], (int, float))
        
        # Check value ranges
        assert 0 <= result["probability"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert result["estimated_area"] >= 0
        assert result["risk_level"] in ["low", "medium", "high", "extreme"]
        
        # Check spread direction format
        valid_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        assert result["spread_direction"] in valid_directions
    
    def test_run_inference_edge_cases(self):
        """Test inference with edge case inputs."""
        # Test with None covariates
        ignition_point = {"latitude": 56.1304, "longitude": -106.3468}
        date = "2024-07-15"
        
        result = run_inference(ignition_point, date, None)
        assert isinstance(result, dict)
        assert "risk_level" in result
    
    def test_run_inference_metadata(self):
        """Test that inference includes metadata when implemented."""
        ignition_point = {"latitude": 56.1304, "longitude": -106.3468}
        date = "2024-07-15"
        
        result = run_inference(ignition_point, date)
        
        # For now, we don't have metadata in the placeholder
        # When you implement the real model, you can uncomment this:
        # assert "metadata" in result
        # assert result["metadata"]["ignition_point"] == ignition_point
        # assert result["metadata"]["prediction_date"] == date 