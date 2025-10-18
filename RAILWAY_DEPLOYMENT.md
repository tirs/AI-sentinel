# Railway.app Deployment Guide for AI Sentinel

## Overview
This guide walks you through deploying the AI Sentinel application (API + Dashboard) to Railway.app.

## Prerequisites
- ✅ GitHub account with your repository
- ✅ Railway.app account (free tier available)
- ✅ Project files committed to GitHub

## Quick Start (5 minutes)

### 1. Push Latest Code to GitHub
```powershell
cd "c:\Users\simba\Trustmuseta81 Dropbox\Projects\Ethical"
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### 2. Create Railway Project
1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose "tirs/AI-sentinel" repository
5. Click "Deploy Now"

Railway will automatically:
- Detect the Procfile
- Install Python dependencies from requirements.txt
- Build your application
- Start both API and Dashboard services

### 3. Configure Environment Variables (First Deployment Only)

In Railway dashboard, go to your project → Variables and add:

```env
# Environment
ENVIRONMENT=production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# Dashboard Configuration
DASHBOARD_PORT=8501
DASHBOARD_TITLE=AI Sentinel Dashboard

# Model Configuration (use smaller models for faster startup)
NLP_MODEL_NAME=distilbert-base-multilingual-cased
VISION_MODEL_NAME=efficientnet_b0
BATCH_SIZE=16
MAX_LENGTH=256

# Elasticsearch (optional - disable if not needed)
ELASTICSEARCH_HOST=
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=ai_sentinel

# Logging
LOG_LEVEL=INFO

# Data Paths
DATA_DIR=data
MODELS_DIR=models
CACHE_DIR=cache

# Thresholds
HATE_SPEECH_THRESHOLD=0.75
DEEPFAKE_THRESHOLD=0.80
DISINFORMATION_THRESHOLD=0.70
```

### 4. Connect API to Dashboard

After deployment, you need to tell the Dashboard where to find the API:

1. Get your API service URL from Railway dashboard
2. Set the `API_URL` variable in Railway:
   - Format: `https://your-api-service.railway.app`
   - Or: `http://api:8000` (if same project)

3. In Railway dashboard → Add variable:
   ```
   API_URL = https://your-api-service-url.railway.app
   ```

### 5. Wait for Deployment

Railway will show deployment status. Once complete:
- **API**: Available at `https://your-api-service.railway.app`
- **Dashboard**: Available at `https://your-dashboard-service.railway.app`

## Detailed Configuration

### Environment Variables Explained

| Variable | Value | Purpose |
|----------|-------|---------|
| ENVIRONMENT | production | Disables auto-reload, enables production mode |
| API_PORT | 8000 | Internal port for API server |
| DASHBOARD_PORT | 8501 | Internal port for Streamlit dashboard |
| API_URL | https://... | Dashboard uses this to connect to API |
| NLP_MODEL_NAME | distilbert-base-multilingual-cased | Lightweight NLP model (smaller = faster) |
| VISION_MODEL_NAME | efficientnet_b0 | Vision model for deepfake detection |
| BATCH_SIZE | 16 | Reduce for limited RAM environments |

### Model Selection Tips

For Railway's free tier (512MB RAM):
- Use `distilbert` instead of `bert-base` (6x smaller)
- Use `efficientnet_b0` (smallest EfficientNet)
- Reduce `BATCH_SIZE` to 8-16
- Set `MAX_LENGTH` to 256

For Railway's paid tier:
- Use `bert-base-multilingual-cased` (more accurate)
- Use `efficientnet_b4` (better accuracy)
- Increase `BATCH_SIZE` to 32
- Use `MAX_LENGTH` of 512

### Persistent Storage

To keep models and data between deployments:

1. Go to Railway project settings
2. Add Volume:
   - Mount Path: `/app/models`
   - Size: 2GB (or more)
3. Models will persist between deployments

## Troubleshooting

### Build Takes Too Long / Fails
- **Problem**: Installing PyTorch takes time
- **Solution**: Railway is installing pre-compiled wheels, this is normal (2-5 minutes)

### Dashboard Shows "Cannot Connect to API"
- Check if `API_URL` variable is set correctly
- Ensure API service is marked "Public" in Railway
- Verify API service is running (check logs)

### App Crashes on Startup
- **Check logs**: Railway dashboard → Logs
- **Common causes**:
  - Not enough RAM (upgrade plan or reduce model size)
  - Missing environment variables
  - Model download timeout (increase timeout or pre-download)

### Models Take Forever to Download
- **First startup is slow**: Models download during initialization
- **Solution**: Add persistent volume for `/app/models` directory

### API Times Out
- Increase timeout in Railway deployment settings
- Reduce batch size or model complexity
- Use GPU tier if available (paid)

## Production Optimization

### 1. Enable Caching
Add to Railway environment:
```
CACHE_DIR=/app/cache
```

### 2. Add Monitoring
Railway includes basic monitoring. To add more:
- Set `LOG_LEVEL=DEBUG` for detailed logs
- Monitor response times in Railway dashboard

### 3. Auto-Deployments
Railway auto-deploys on push to main branch. To disable:
- Railway dashboard → Settings → Auto Deploy → Off

### 4. Custom Domain
1. Railway dashboard → Project → Settings
2. Add custom domain
3. Configure DNS (CNAME record)

## Performance Monitoring

Railway provides:
- **CPU Usage**: Should stay under 60% for free tier
- **Memory Usage**: Monitor if using persistent models
- **Response Time**: Check in Railway analytics
- **Logs**: Real-time logs for debugging

## Cost Management

- **Free Tier Limits**:
  - 5GB storage, 500MB RAM
  - Free $5/month credit
  - Includes bandwidth

- **Monitor Usage**: Railway dashboard → Analytics
- **Auto-pause**: Your service auto-pauses after 30 minutes of no traffic

## Next Steps

1. ✅ Deploy to Railway (see Quick Start)
2. Test API: `https://your-api.railway.app/docs`
3. Test Dashboard: Visit dashboard URL
4. Monitor logs and performance
5. Set up custom domain (optional)
6. Configure backup strategy

## Support & Debugging

- **Railway Docs**: https://docs.railway.app
- **API Documentation**: `/docs` endpoint (Swagger)
- **Check Logs**: Railway dashboard → Deployments → View Logs
- **Restart Service**: Railway dashboard → Project → Restart Deployment

## Undoing / Rollback

If deployment has issues:
1. Railway dashboard → Deployments
2. Select previous working deployment
3. Click "Rollback"

Your previous version will be restored instantly.

---

**Happy Deploying! 🚀**

For questions, check: https://github.com/tirs/AI-sentinel/issues