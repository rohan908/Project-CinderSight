# CinderSight - Canadian Fire Prediction

Advanced wildfire prediction powered by neural networks.

## Background

Between 2018 and 2023, 7.7 million acres of U.S. land burned as a result of wildfires [[1]](#1), costing $70 billion annually [[2]](#2) and emitting 3.3 billion tons of CO2 every year [[3]](#3).

## Research Question

In this project, we wanted to see if we could take a successful fire-spread prediction model and apply it to a dataset with more covariates to improve its results.
The model used comes from the [2nd place submission](https://www.kaggle.com/competitions/2024-flame-ai-challenge/discussion/541458) to the [2024 FLAME AI Challenge](https://www.kaggle.com/competitions/2024-flame-ai-challenge/overview).

## Dataset

The dataset we used to train and evaluate the model is the [(Enhanced & modified) Next Day Wildfire Spread Dataset](https://www.kaggle.com/datasets/rufaiyusufzakari/enhanced-and-modified-next-day-wildfire-spread?select=next_day_wildfire_spread_eval_00.tfrecord). It contains 19 features interpolated to a resolution of 1 km, including current weather, weather forecast, terrain, land, and population variables. This marks a significant increase from the dataset used in the 2024 FLAME AI Challenge, which includes only 4 features. However, the NDWS dataset has the disadvantage of lacking multiple sequential days of data, unlike the challenge dataset.

## Model Architecture

Due to the unfortunate lack of sequential data provided in the NDWS dataset, the temporal pieces of the architecture had to be removed or altered. Here is a diagram of the modified architecture, without the transformer and encoder blocks:

![alt text](https://github.com/rohan908/Project-CinderSight/main/images/model_diagram.png "Model Diagram")

## Results

F1 Score -	0.425
Intersection over Union - 0.270
Precision - 0.312
Recall - 0.669
Inference Speed - 51.0ms

## Folder Structure

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
├── model/              # ML Model Training (placeholder)
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

## GitHub Workflows

This project uses GitHub Actions workflows to automate testing, building, and deployment. Think of workflows as **automated robots** that perform tasks when certain events happen in your repository.

### What GitHub Workflows Do:

#### **Automated Tasks**
- **Build your code** when you push changes
- **Run tests** to ensure everything works
- **Deploy your app** to hosting services
- **Check code quality** (linting, formatting)
- **Send notifications** when things break

#### **When They Run**
Workflows trigger on specific events:
- **Push to main branch** → Deploy to production
- **Pull request** → Run tests and checks
- **New release** → Build and package
- **Manual trigger** → Run on demand

### Project Workflows

#### **`web-ci.yml`** (Frontend CI)
```yaml
# Runs when someone creates a pull request
- Installs Node.js dependencies
- Runs linting (finds code style issues)
- Builds the Next.js app
- Ensures everything works before merging
```

#### **`app-ci.yml`** (Backend CI)
```yaml
# Runs when someone creates a pull request
- Installs Python dependencies
- Runs tests (if any exist)
- Checks code formatting
- Ensures API works correctly
```

#### **`deploy.yml`** (Deployment)
```yaml
# Runs when code is pushed to main branch
- Builds both frontend and backend
- Deploys to hosting services (when configured)
- Updates your live website automatically
```

### Real-World Workflow Example:

1. **You make changes** to your fire prediction code
2. **You create a pull request** to merge your changes
3. **GitHub automatically:**
   - Runs your tests
   - Checks code quality
   - Builds the app
   - Tells you if anything is broken
4. **If everything passes**, you can safely merge
5. **When you merge to main**, it automatically deploys to your live website

### Benefits:
-  **Catch bugs early** before they reach production
-  **Automate repetitive tasks** (no more manual testing)
-  **Ensure code quality** (consistent formatting, no broken builds)
-  **Deploy automatically** (no manual deployment steps)
-  **Team collaboration** (everyone's code gets the same checks)

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

## References

<a id="1">[1]</a> 
National Interagency Fire Center

<a id="2">[2]</a> 
NOAA

<a id="3">[3]</a> 
Global Fire Emissions Database