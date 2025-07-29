#!/bin/bash

# Get port from environment variable, default to 8000
PORT=${PORT:-8000}

echo "Starting FastAPI application on port $PORT"

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT 