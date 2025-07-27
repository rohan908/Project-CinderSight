# CinderSight Visualization API

This API provides endpoints for generating wildfire spread prediction visualizations for specific sample indices from the Enhanced NDWS dataset.

## Features

- **Sample-based visualization generation**: Generate visualizations for any sample index
- **Flexible storage options**: Choose to save images to disk or use temporary storage
- **Overwrite control**: Option to overwrite existing files or create unique directories
- **Selective downloads**: Download all files, only metrics, or only feature images
- **Background processing**: Long-running tasks are processed asynchronously
- **Task management**: Track task status and clean up completed tasks

## API Endpoints

### Core Endpoints

- `POST /visualization/generate` - Start visualization generation
- `GET /visualization/status/{task_id}` - Check task status
- `GET /visualization/download/{task_id}` - Download all visualizations
- `GET /visualization/download/{task_id}/metrics` - Download only metrics JSON
- `GET /visualization/download/{task_id}/features` - Download only feature PNGs
- `GET /visualization/samples/available` - Get available sample information
- `DELETE /visualization/tasks/{task_id}` - Delete a completed task
- `DELETE /visualization/tasks` - Delete all completed tasks

## Request Parameters

### Visualization Generation Request

```json
{
  "sample_idx": 0,
  "save_images": true,
  "overwrite_existing": true,
  "include_features": true,
  "include_fire_progression": true,
  "include_metrics_dashboard": true,
  "include_documentation": true
}
```

**Parameters:**
- `sample_idx` (int, required): Index of the sample to visualize (0 to N-1)
- `save_images` (bool, default: true): Whether to save images to disk
- `overwrite_existing` (bool, default: true): Whether to overwrite existing files
- `include_features` (bool, default: true): Generate individual feature visualizations
- `include_fire_progression` (bool, default: true): Generate fire progression visualizations
- `include_metrics_dashboard` (bool, default: true): Generate metrics dashboard
- `include_documentation` (bool, default: true): Generate feature documentation

## Usage Examples

### Python Client

```python
import requests
import time

# Initialize client
base_url = "http://localhost:8000"

# 1. Check available samples
response = requests.get(f"{base_url}/visualization/samples/available")
samples_info = response.json()
print(f"Available samples: {samples_info}")

# 2. Generate visualizations without saving to disk
payload = {
    "sample_idx": 0,
    "save_images": False,  # Use temporary storage
    "overwrite_existing": True
}
response = requests.post(f"{base_url}/visualization/generate", json=payload)
result = response.json()
task_id = result["task_id"]

# 3. Wait for completion
while True:
    status_response = requests.get(f"{base_url}/visualization/status/{task_id}")
    status = status_response.json()
    
    if status["status"] == "completed":
        print("Task completed!")
        break
    elif status["status"] == "failed":
        print(f"Task failed: {status['error_message']}")
        break
    
    time.sleep(5)

# 4. Download results
# Download all files
zip_response = requests.get(f"{base_url}/visualization/download/{task_id}")
with open("visualizations.zip", "wb") as f:
    f.write(zip_response.content)

# Download only metrics
metrics_response = requests.get(f"{base_url}/visualization/download/{task_id}/metrics")
with open("metrics.json", "wb") as f:
    f.write(metrics_response.content)
```

### cURL Examples

```bash
# Generate visualizations for sample 0
curl -X POST "http://localhost:8000/visualization/generate" \
  -H "Content-Type: application/json" \
  -d '{"sample_idx": 0, "save_images": false}'

# Check task status
curl "http://localhost:8000/visualization/status/{task_id}"

# Download all visualizations
curl "http://localhost:8000/visualization/download/{task_id}" -o visualizations.zip

# Download only metrics
curl "http://localhost:8000/visualization/download/{task_id}/metrics" -o metrics.json
```

## Storage Options

### Option 1: Temporary Storage (save_images: false)
- Images are generated in a temporary directory
- Files are automatically cleaned up when tasks are deleted
- Best for one-time use or when disk space is limited

### Option 2: Persistent Storage (save_images: true, overwrite_existing: true)
- Images are saved to `model/visualizations/sample_{idx}/`
- Existing files are overwritten
- Best for repeated access to the same sample

### Option 3: Unique Storage (save_images: true, overwrite_existing: false)
- Images are saved to `model/visualizations/sample_{idx}_{task_id}/`
- Each generation creates a unique directory
- Best for preserving multiple versions

## Generated Files

For each sample, the API generates:

### Feature Visualizations (19 files)
- `feature_00_temperature.png` - Temperature (°C)
- `feature_01_humidity.png` - Relative humidity (%)
- `feature_02_wind_speed.png` - Wind speed (m/s)
- `feature_03_wind_direction.png` - Wind direction (degrees)
- `feature_04_precipitation.png` - Precipitation (mm)
- `feature_05_pressure.png` - Atmospheric pressure (hPa)
- `feature_06_solar_radiation.png` - Solar radiation (W/m²)
- `feature_07_visibility.png` - Visibility (km)
- `feature_08_temperature_forecast.png` - Forecast temperature (°C)
- `feature_09_humidity_forecast.png` - Forecast humidity (%)
- `feature_10_wind_speed_forecast.png` - Forecast wind speed (m/s)
- `feature_11_precipitation_forecast.png` - Forecast precipitation (mm)
- `feature_12_elevation.png` - Elevation (m)
- `feature_13_slope.png` - Slope (degrees)
- `feature_14_aspect.png` - Aspect (degrees)
- `feature_15_ndvi.png` - Normalized Difference Vegetation Index
- `feature_16_land_cover.png` - Land cover type
- `feature_17_population.png` - Population density (people/km²)
- `feature_18_previous_fire.png` - Previous day fire mask

### Fire Progression Visualizations (5 files)
- `fire_previous_fire.png` - Previous day fire locations
- `fire_ground_truth.png` - Actual next day fire locations
- `fire_prediction_probability.png` - Model probability prediction
- `fire_prediction_binary.png` - Binary prediction (threshold > 0.5)
- `fire_comparison_overlay.png` - Comparison overlay (Red=GT, Green=Pred, Yellow=Both)

### Metrics and Documentation (3 files)
- `metrics_dashboard.png` - Comprehensive performance metrics visualization
- `sample_metrics.json` - Detailed metrics in JSON format
- `feature_documentation.json` - Complete documentation of all generated files

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Task not found
- `500` - Internal server error

Error responses include a `detail` field with the error message.

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model and data are available:
   - Model file: `model/models/model_nfp.pth`
   - Data directory: `model/data/processed/`

3. Start the API server:
```bash
uvicorn app.main:app --reload
```

4. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Testing

Use the provided client example:
```bash
python client_example.py
```

This will demonstrate both temporary and persistent storage modes.

## Notes

- The API uses background tasks for long-running operations
- Temporary files are automatically cleaned up when tasks are deleted
- The API validates paths and provides helpful error messages
- All file downloads are provided as ZIP files for convenience
- The API is designed to handle concurrent requests efficiently 