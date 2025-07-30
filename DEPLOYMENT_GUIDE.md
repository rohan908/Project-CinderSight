# CinderSight Deployment Guide

This guide will help you deploy CinderSight to Railway (backend) and Netlify (frontend).

## Prerequisites

- GitHub account
- Railway account (free tier available)
- Netlify account (free tier available)
- Your code pushed to a GitHub repository

## Step 1: Deploy Backend to Railway

### 1.1 Prepare Your Repository

Make sure your repository structure looks like this:
```
Project-CinderSight/
├── api/                    # Backend code
│   ├── app/
│   ├── requirements.txt
│   ├── railway.json
│   ├── Procfile
│   └── runtime.txt
├── web/                    # Frontend code
│   ├── src/
│   ├── package.json
│   ├── .nvmrc
│   ├── .npmrc
│   └── netlify.toml
└── README.md
```

### 1.2 Deploy to Railway

1. **Go to [Railway.app](https://railway.app)** and sign in with GitHub
2. **Click "New Project"** → "Deploy from GitHub repo"
3. **Select your repository** (Project-CinderSight)
4. **Set the root directory** to `api` (this is crucial!)
5. **Railway will automatically detect** it's a Python project
6. **Wait for the build** to complete

### 1.3 Configure Environment Variables (Optional)

In Railway dashboard, go to your project → Variables tab and add:
```
PYTHONPATH=/app
```

### 1.4 Get Your Railway URL

Once deployed, Railway will give you a URL like:
`https://your-app-name.railway.app`

**Save this URL** - you'll need it for the frontend configuration.

## Step 2: Deploy Frontend to Netlify

### 2.1 Update Frontend Configuration

Before deploying, you need to set the Railway URL as an environment variable:

1. **Create `web/.env.local`** (if it doesn't exist):
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-app-url.railway.app
   ```
   Replace `your-railway-app-url.railway.app` with your actual Railway URL.

### 2.2 Deploy to Netlify

1. **Go to [Netlify.com](https://netlify.com)** and sign in with GitHub
2. **Click "New site from Git"**
3. **Select your repository** (Project-CinderSight)
4. **Set the base directory** to `web`
5. **Set build command** to: `npm run build`
6. **Set publish directory** to: `.next`
7. **Click "Deploy site"**

### 2.3 Configure Environment Variables

In Netlify dashboard:
1. Go to Site settings → Environment variables
2. Add:
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-app-url.railway.app
   ```
   Replace with your actual Railway URL.

### 2.4 Build Configuration

The project is configured to use:
- **Node.js version**: 18 (specified in `.nvmrc` and `package.json`)
- **npm version**: 9 (specified in `netlify.toml`)
- **Updated dependencies**: All deprecated packages have been updated
- **Direct API calls**: Frontend calls Railway backend directly (no Next.js API routes)

### 2.5 Architecture Overview

- **Frontend (Netlify)**: Next.js app that calls Railway backend directly
- **Backend (Railway)**: FastAPI server with visualization endpoints
- **Communication**: Direct HTTP calls from frontend to backend using `NEXT_PUBLIC_API_URL`

## Step 3: Test Your Deployment

1. **Test the backend**: Visit your Railway URL + `/` (should show API info)
2. **Test the frontend**: Visit your Netlify URL
3. **Test the connection**: Try loading a sample in the frontend

## Troubleshooting

### Common Issues

1. **Backend not starting**: Check Railway logs for Python import errors
2. **Frontend can't connect**: Verify the Railway URL in environment variables
3. **Build failures**: Check that all dependencies are in requirements.txt
4. **CORS errors**: The backend should handle CORS automatically
5. **Netlify build failures**: 
   - Ensure Node.js version 18 is being used
   - Check that all deprecated packages have been updated
   - Verify ESLint configuration is compatible
6. **API connection issues**:
   - Verify `NEXT_PUBLIC_API_URL` is set correctly
   - Check that Railway backend is running
   - Ensure CORS is properly configured on the backend

### Railway Logs

To debug backend issues:
1. Go to Railway dashboard → your project
2. Click on the deployment
3. Check the logs for error messages

### Netlify Logs

To debug frontend issues:
1. Go to Netlify dashboard → your site
2. Go to Functions tab (if using functions)
3. Check deploy logs for build errors

### Netlify Build Fixes Applied

The following fixes have been applied to resolve Netlify build issues:

1. **Updated package.json**:
   - Added `engines` field to specify Node.js >=18.0.0
   - Updated ESLint to version 9.0.0
   - Updated TypeScript ESLint packages to version 7.0.0

2. **Added configuration files**:
   - `.nvmrc` - Specifies Node.js version 18
   - `.npmrc` - Configures npm to use legacy peer deps and disable warnings

3. **Updated netlify.toml**:
   - Specified Node.js version 18
   - Specified npm version 9
   - Removed API redirects (using direct calls instead)

4. **Simplified ESLint config**:
   - Removed deprecated configurations
   - Kept only essential rules

5. **Removed Next.js API routes**:
   - Frontend now calls Railway backend directly
   - No more `/api` route conflicts

## Cost Considerations

- **Railway**: Free tier includes $5 credit/month
- **Netlify**: Free tier includes 100GB bandwidth/month
- **Both**: Should be sufficient for development and small-scale usage

## Next Steps

1. **Set up custom domains** (optional)
2. **Configure SSL certificates** (automatic with both platforms)
3. **Set up monitoring** and alerts
4. **Configure CI/CD** for automatic deployments

## Support

If you encounter issues:
1. Check the logs in both Railway and Netlify dashboards
2. Verify all environment variables are set correctly
3. Ensure your code works locally before deploying
4. Check that all dependencies are properly specified
5. For Netlify build issues, check the Node.js version and ESLint configuration
6. For API connection issues, verify the Railway URL and CORS settings 