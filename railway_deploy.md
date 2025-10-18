# Deploying AI Sentinel to Railway.app

## Prerequisites
1. GitHub account with your repository pushed (✅ Already done!)
2. Railway.app account (free at https://railway.app)

## Step 1: Create Railway Project

1. Go to https://railway.app and sign in/sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your GitHub account and select `tirs/AI-sentinel` repository

## Step 2: Configure Services

Railway will automatically detect the `Procfile` and create two services:
- **API Service**: Runs FastAPI on assigned PORT
- **Dashboard Service**: Runs Streamlit on its own assigned PORT

## Step 3: Set Environment Variables

In Railway dashboard, go to the project variables and set:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
NLP_MODEL_NAME=bert-base-multilingual-cased
VISION_MODEL_NAME=efficientnet_b0
BATCH_SIZE=32
MAX_LENGTH=512

# Elasticsearch Configuration (Optional - can be added as separate service)
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=ai_sentinel

# Logging
LOG_LEVEL=INFO

# Dashboard
DASHBOARD_PORT=8501

# Data Paths
DATA_DIR=data
MODELS_DIR=models
CACHE_DIR=cache

# Thresholds
HATE_SPEECH_THRESHOLD=0.75
DEEPFAKE_THRESHOLD=0.80
DISINFORMATION_THRESHOLD=0.70
```

## Step 4: Fix Dashboard-API Communication

Once deployed, the dashboard needs to connect to the API using the Railway service URL.

1. Get your API service URL from Railway dashboard
2. In the API service settings, expose on public network
3. Update the dashboard to use the API URL (see next section)

## Step 5: Update Config for Railway

The dashboard needs to connect to the API service. Update the API URL in the dashboard:

**In `src/dashboard/app.py` (line 31-32), replace:**
```python
API_HOST = "localhost" if settings.API_HOST == "0.0.0.0" else settings.API_HOST
API_BASE_URL = f"http://{API_HOST}:{settings.API_PORT}"
```

**With:**
```python
# For Railway, get API URL from environment or default to localhost
API_URL = os.getenv("API_URL", f"http://localhost:{settings.API_PORT}")
API_BASE_URL = API_URL
```

And set in Railway:
- API_URL: `https://your-api-service.railway.app` (after deployment)

## Step 6: Handle Dependencies

Railway will automatically install from `requirements.txt`.

**Note:** If you're using GPU models, you may need to adjust for CPU inference or use Railway's paid tier with GPU support.

## Step 7: Configure Storage (Optional but Recommended)

Railway supports persistent volumes. Add to your project:
- Volume mount for `/models` directory to persist downloaded models
- Volume mount for `/cache` directory

## Step 8: Deploy

1. Commit the new files:
```bash
git add Procfile railway.json railway_deploy.md
git commit -m "Add Railway deployment configuration"
git push origin main
```

2. Railway will automatically detect changes and redeploy

## Step 9: Access Your Application

Once deployed:
- **Dashboard**: Will be available at your Railway app URL
- **API**: Will be available at the API service URL (exposed separately)

## Troubleshooting

### Dashboard Can't Connect to API
- Make sure API service is exposed publicly
- Check environment variables are set correctly
- Look at Railway logs for error messages

### Build Fails
- Ensure all `requirements.txt` dependencies are available for your platform
- Some torch dependencies might need specific versions for Railway
- Check Railway build logs for specific errors

### Models Taking Too Long to Load
- Consider pre-downloading models and storing in cache volume
- Or use smaller model variants (distilbert instead of bert-base)

## Next Steps

After successful deployment:
1. Monitor logs in Railway dashboard
2. Set up custom domain (optional)
3. Configure backup/persistence for data
4. Set up CI/CD for automatic deployments