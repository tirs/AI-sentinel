# Streamlit Cloud Deployment Guide

This guide covers deploying the AI Sentinel dashboard to **Streamlit Cloud**.

## Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Streamlit Account**: Create free account at [streamlit.io](https://streamlit.io)
3. **Environment Variables**: Set up secrets in Streamlit Cloud dashboard

## Deployment Steps

### 1. Prepare Your Repository

Ensure you have these files in your root directory:

```
streamlit_app.py          # ✅ Entry point (created)
requirements.txt          # ✅ Dependencies (already exists)
.streamlit/config.toml    # ✅ Configuration (created)
.gitignore                # ✅ Already exists
```

### 2. Push to GitHub

```bash
git add streamlit_app.py .streamlit/config.toml
git commit -m "Add Streamlit Cloud deployment configuration"
git push origin main
```

### 3. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app** → **From existing repo**
3. Select your repository
4. Set main file to: `streamlit_app.py`
5. Click **Deploy**

## Environment Configuration

### API Backend URL

The dashboard needs to communicate with your FastAPI backend. Configure this via Streamlit Cloud secrets:

**In Streamlit Cloud Dashboard:**
1. Go to your app settings (⚙️)
2. Navigate to **Secrets**
3. Add:

```toml
# API Configuration
API_URL = "https://your-api-domain.com"
API_BASE_URL = "https://your-api-domain.com"

# Optional: If using local development
# API_URL = "http://localhost:8000"
```

**Or in `.streamlit/secrets.toml` (local development only):**

```toml
API_URL = "http://localhost:8000"
API_BASE_URL = "http://localhost:8000"
```

> ⚠️ **WARNING**: Never commit `secrets.toml` to GitHub. Add to `.gitignore`.

### Environment Variables Reference

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `API_URL` | FastAPI backend URL | localhost:8000 | `https://api.example.com` |
| `API_BASE_URL` | Alternative API URL | localhost:8000 | `https://api.example.com` |
| `LOG_LEVEL` | Logging verbosity | `info` | `debug`, `warning` |

## Performance Optimization

### Memory Management

Streamlit Cloud has limited resources (~1GB RAM). Optimize with:

1. **Lazy Loading**: Models loaded on-demand
2. **Caching**: Use `@st.cache_resource` for expensive operations
3. **Session State**: Manage state efficiently

Example caching in your app:

```python
import streamlit as st

@st.cache_resource
def load_model():
    # This runs once and is cached
    return YourModel()

model = load_model()
```

### Request Timeout

Default: 5 seconds (configured in `src/dashboard/app.py`)

For slow APIs, increase in your dashboard:

```python
API_TIMEOUT = 30  # seconds
```

## Troubleshooting

### Issue: "Connection refused" Error

**Cause**: Dashboard can't reach the API backend

**Solution**:
1. Verify API backend is running and accessible
2. Check `API_URL` in Streamlit secrets matches your backend
3. Ensure backend allows CORS requests

```python
# In your FastAPI backend (src/api/server.py)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: "Module not found" Error

**Cause**: Python path not configured correctly

**Solution**: `streamlit_app.py` already handles this:

```python
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
```

### Issue: Dependency Installation Fails

**Cause**: Some heavy packages (torch, transformers) take time to install

**Solution**:
- First deployment may take 5-10 minutes
- Streamlit caches dependencies between redeploys
- Check logs for specific package errors

### Issue: "torch not found" After Reboot

**Cause**: Container memory exceeded, cache cleared

**Solution**:
1. PyTorch is large (~2GB). Streamlit Cloud may have limits.
2. Consider using a lighter model or separate API backend
3. Use `torch==2.0.0` (smaller than latest)

## FastAPI Backend Hosting

Since Streamlit Cloud is for the dashboard, you need to host your FastAPI backend separately:

### Option 1: Railway (Recommended)
- Free tier available
- Simple deployment from Git
- See `run_api.py` for configuration

```bash
# Deploy only the API
railway link  # Connect your repo
railway up
```

### Option 2: Heroku
- Requires `Procfile` and `runtime.txt`
- Free tier deprecated (use paid dynos)

### Option 3: Docker (Any Cloud)
- Use provided `Dockerfile`
- Deploy to: AWS, Google Cloud, Azure, etc.

```bash
docker build -t ai-sentinel-api .
docker run -p 8000:8000 ai-sentinel-api
```

### Option 4: Local Development
- Run `python run_api.py` on your machine
- Set `API_URL = "http://your-ip:8000"` in Streamlit secrets
- Your machine must be accessible from the internet (use ngrok for testing)

## Uvicorn Signal Handling Fix

The `run_api.py` has been configured to handle both development and production:

- **Development** (`ENVIRONMENT=development`): Auto-reload enabled
- **Production** (`ENVIRONMENT=production`): Multi-worker mode, no reload

This prevents "signal handling" errors in containers.

## Monitoring & Debugging

### View Streamlit Logs

1. In Streamlit Cloud dashboard → Your app → Logs
2. Check for:
   - Import errors
   - API connection failures
   - Memory issues
   - Timeout errors

### Enable Debug Mode

In `.streamlit/config.toml`:

```toml
[logger]
level = "debug"
```

Or via environment variable:
```
streamlit run streamlit_app.py --logger.level=debug
```

## Performance Metrics

Expected response times:

| Operation | Duration | Notes |
|-----------|----------|-------|
| Page load | 2-3s | Includes cached model loading |
| API health check | <1s | Should be instant |
| Image analysis | 3-10s | Depends on model complexity |
| Video analysis | 30-120s | Backend processing time |

## Security Best Practices

1. **Never commit secrets**: Use Streamlit Cloud secrets management
2. **CORS protection**: Restrict API access to your domain
3. **API keys**: Store in Streamlit secrets, not code
4. **HTTPS**: Streamlit Cloud provides automatic SSL
5. **Data privacy**: Don't log sensitive user data

## Cleanup & Maintenance

### Remove Old Builds

Streamlit Cloud automatically cleans up, but you can manually:
1. App settings → Delete old releases

### Update Dependencies

1. Modify `requirements.txt`
2. Push to GitHub
3. Streamlit auto-redeploys

### Monitor Disk Usage

If app takes >5 minutes to load:
1. Check for large cached files in code
2. Verify `data/` folder isn't included in repo
3. Add large files to `.gitignore`

## Example `.gitignore` for Streamlit Cloud

```gitignore
# Data files
data/
models/
*.pth
*.pt
*.bin

# Cache
__pycache__/
*.pyc
.pytest_cache/
.streamlit/cache/

# Secrets (LOCAL ONLY)
.streamlit/secrets.toml

# Environment
.env
.env.local

# OS
.DS_Store
*.swp
*.swo
```

## Next Steps

1. ✅ Push `streamlit_app.py` and `.streamlit/config.toml` to GitHub
2. 🔧 Deploy FastAPI backend (Railway, Docker, or alternative)
3. 🚀 Connect Streamlit Cloud to your repository
4. 🔐 Configure API URL in Streamlit secrets
5. 📊 Monitor logs and performance

## Support

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Community Forum**: https://discuss.streamlit.io
- **Project Issues**: https://github.com/tirs/AI-sentinel/issues