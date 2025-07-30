@echo off
echo Starting CinderSight Development Environment...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo  .env file not found. Creating from template...
    copy env.example .env
    echo Please edit .env file with your configuration before continuing.
    echo    Required: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_ANON_KEY
    pause
    exit /b 1
)

echo Starting services with Docker Compose...
docker-compose up --build

echo Services started!
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8080
echoAPI Docs: http://localhost:8080/docs
echo.
 