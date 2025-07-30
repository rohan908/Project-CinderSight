# Supabase Integration Setup

This document explains how to set up Supabase integration for the CinderSight API to fetch models and data from Supabase buckets.

## Prerequisites

1. A Supabase project with storage buckets configured
2. Models and data files uploaded to Supabase storage
3. Database tables configured to store bucket links

## Environment Variables

Create a `.env` file in the `api/` directory with the following variables:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
DEBUG=True

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Model and Data Configuration
DEFAULT_MODEL_NAME=model_nfp.pth
DEFAULT_DATA_SPLIT=test
```

## Supabase Setup

### 1. Create Storage Buckets

Create two storage buckets in your Supabase project:
- `models` - for storing model files (.pth files)
- `data` - for storing data files (.data and .labels files)

### 2. Upload Files

Upload your files to the appropriate buckets:

**Models bucket:**
- `model_nfp.pth` (your trained model)

**Data bucket:**
- `test.data` (test features)
- `test.labels` (test labels)
- `train.data` (train features)
- `train.labels` (train labels)

### 3. Create Database Tables

Create the following tables in your Supabase database:

#### Models Table
```sql
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### Samples Table
```sql
CREATE TABLE samples (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    features_file_path TEXT NOT NULL,
    target_file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 4. Insert Records

Insert records for your models and data files:

```sql
-- Insert model records
INSERT INTO models (name, model_path) VALUES 
('model_nfp.pth', 'https://your-project.supabase.co/storage/v1/object/sign/models/model_nfp.pth?token=your-signed-token');

-- Insert data records
INSERT INTO samples (name, features_file_path, target_file_path) VALUES 
('test', 'https://your-project.supabase.co/storage/v1/object/sign/data/test.data?token=your-signed-token', 'https://your-project.supabase.co/storage/v1/object/sign/data/test.labels?token=your-signed-token'),
('train', 'https://your-project.supabase.co/storage/v1/object/sign/data/train.data?token=your-signed-token', 'https://your-project.supabase.co/storage/v1/object/sign/data/train.labels?token=your-signed-token');
```

## API Features

The updated API now includes:

### 1. Supabase Integration
- Automatic model and data downloading from Supabase buckets
- Temporary file management
- Error handling for missing files

### 2. Environment Configuration
- Centralized configuration management
- Validation of required environment variables
- Flexible CORS configuration

### 3. Data Loading
- Loads NDWS data from Supabase buckets
- Handles both test and train data splits
- Automatic fallback to train data if test data is unavailable

### 4. Model Loading
- Downloads models from Supabase on demand
- Supports multiple model types
- Configurable model names

## Usage

### Starting the API

1. Set up your environment variables
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `uvicorn app.main:app --host 0.0.0.0 --port 8080`

### API Endpoints

The API will automatically:
- Load data from Supabase on startup
- Download models when needed for visualizations
- Clean up temporary files after processing

### Health Check

Check the API status:
```bash
curl http://localhost:8080/health
```

This will return:
```json
{
    "status": "healthy",
    "data_loaded": true,
    "available_samples": 1000
}
```

## Troubleshooting

### Common Issues

1. **Supabase connection failed**
   - Check your `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
   - Ensure your Supabase project is active

2. **Model not found**
   - Verify the model exists in the `models` table
   - Check the bucket link is correct
   - Ensure the file exists in the storage bucket

3. **Data not found**
   - Verify the data files exist in the `data` table
   - Check the bucket links are correct
   - Ensure the files exist in the storage bucket

4. **Permission errors**
   - Ensure your service role key has the necessary permissions
   - Check bucket policies allow read access

### Debug Mode

Enable debug mode by setting `DEBUG=True` in your environment variables to get more detailed error messages.

## Security Notes

- Use the service role key only in your backend API
- Never expose the service role key in client-side code
- Consider using row-level security (RLS) policies for additional security
- Regularly rotate your service role keys 