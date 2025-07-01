# CinderSight - Canadian Fire Prediction

Advanced wildfire prediction platform powered by the Canadian Fire Database and AI.

## Features

- **Interactive Map Interface**: Select locations across Canada for fire risk assessment
- **Modern UI with ShadCN**: Beautiful, accessible components built with ShadCN UI
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Real-time Predictions**: Get instant fire risk assessments for any location
- **Risk Assessment**: Multi-level risk classification (Low, Medium, High, Extreme)

## Architecture

```
Project-CinderSight/
├── web/                 # Next.js Frontend with ShadCN UI
│   ├── src/
│   │   ├── app/        # Next.js 13+ App Router
│   │   ├── components/ # React Components (ShadCN + Custom)
│   │   └── lib/        # Utilities
│   └── Dockerfile
├── api/                 # FastAPI Backend (Bare bones)
│   ├── app/
│   │   ├── main.py     # Basic API endpoints
│   │   ├── predict.py  # Placeholder for ML model
│   │   └── db.py       # Placeholder for database
│   └── Dockerfile
└── docker-compose.yml  # Local Development
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Project-CinderSight
```

### 2. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration (optional for now)
nano .env
```

### 3. Run with Docker Compose

```bash
# Start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 4. Local Development

#### Frontend (Next.js)
```bash
cd web
npm install
npm run dev
```

#### Backend (FastAPI)
```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### 1. Select Location
- Use the interactive map to click on any location in Canada
- The map is centered on Canada by default
- Click anywhere to set the ignition point

### 2. Choose Date
- Select a prediction date using the date picker
- The system considers seasonal variations

### 3. Get Prediction
- Click "Predict Fire Risk" to run the analysis
- View risk level, probability, spread direction, and estimated area

## Technical Details

### Frontend Technologies
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **ShadCN UI**: Beautiful, accessible components
- **Leaflet**: Interactive maps
- **Lucide React**: Beautiful icons

### Backend Technologies
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation
- **Placeholder Implementation**: Ready for ML model integration

### UI Components
- **Card**: Clean, organized content sections
- **Button**: Consistent, accessible buttons
- **Badge**: Risk level indicators
- **Input**: Form inputs with proper styling
- **Label**: Accessible form labels

## API Endpoints

### POST /predict
Predict fire risk for a location and date.

**Request:**
### NOTE: Example data for NOW
```json
{
  "ignition_point": {
    "latitude": 56.1304,
    "longitude": -106.3468
  },
  "date": "2024-07-15"
}
```

**Response:**
```json
{
  "prediction": {
    "risk_level": "medium",
    "probability": 0.5,
    "spread_direction": "NE",
    "estimated_area": 5.0,
    "confidence": 0.7
  }
}
```

### GET /health
Health check endpoint.

## Development

### Frontend Development

The frontend is built with modern React patterns and ShadCN UI components:

1. **Components**: Located in `web/src/components/`
   - `ui/`: ShadCN UI components
   - `FireMap.tsx`: Interactive map component

2. **Pages**: Located in `web/src/app/`
   - `page.tsx`: Main application page
   - `layout.tsx`: Root layout with metadata

3. **Styling**: Tailwind CSS with custom design system

### Adding New Features

1. **Frontend Components**: Add to `web/src/components/`
2. **API Endpoints**: Add to `api/app/main.py`
3. **Database Integration**: Update `api/app/db.py`
4. **ML Model**: Update `api/app/predict.py`

### ShadCN UI Components

The project uses ShadCN UI for consistent, accessible components:

- **Button**: `@/components/ui/button`
- **Card**: `@/components/ui/card`
- **Badge**: `@/components/ui/badge`
- **Input**: `@/components/ui/input`
- **Label**: `@/components/ui/label`

### Testing

```bash
# Frontend tests
cd web
npm test

# Backend tests
cd api
pytest
```

### Deployment

The application is containerized and ready for deployment on:
- **Vercel** (Frontend)
- **Railway** (Backend)
- **AWS ECS**
- **Google Cloud Run**
- **Azure Container Instances**

## Next Steps

1. **ML Model Integration**: Replace placeholder with actual fire prediction model
2. **Database Setup**: Connect to Canadian Fire Database
3. **Real-time Data**: Integrate live weather and environmental data
4. **Advanced Features**: Add historical analysis, evacuation routes, etc.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Canadian Forest Service** for fire data
- **Natural Resources Canada** for environmental data
- **OpenStreetMap** for map tiles
- **ShadCN** for beautiful UI components

**CinderSight** - Predicting wildfires, protecting communities. 🔥🇨🇦 