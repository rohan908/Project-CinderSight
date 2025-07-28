@echo off
echo 🚀 CinderSight Deployment Helper
echo ================================

echo.
echo 📋 Prerequisites Check:
echo 1. Make sure your code is pushed to GitHub
echo 2. Have Railway and Netlify accounts ready
echo 3. Ensure all files are committed

echo.
echo 🔧 Backend Deployment (Railway):
echo 1. Go to https://railway.app
echo 2. Click 'New Project' → 'Deploy from GitHub repo'
echo 3. Select your repository
echo 4. Set root directory to: api
echo 5. Wait for deployment to complete
echo 6. Copy the Railway URL (you'll need it for frontend)

echo.
echo 🌐 Frontend Deployment (Netlify):
echo 1. Go to https://netlify.com
echo 2. Click 'New site from Git'
echo 3. Select your repository
echo 4. Set base directory to: web
echo 5. Set build command to: npm run build
echo 6. Set publish directory to: .next
echo 7. Add environment variable: NEXT_PUBLIC_API_URL=https://your-railway-url.railway.app

echo.
echo ✅ After deployment:
echo 1. Test your Railway backend URL + /
echo 2. Test your Netlify frontend URL
echo 3. Try loading a sample in the frontend

echo.
echo 📚 For detailed instructions, see DEPLOYMENT_GUIDE.md
pause 