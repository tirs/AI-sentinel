"""
Entry point for Streamlit Cloud deployment.
This file must be in the root directory for Streamlit Cloud to find it.

Starts both:
- FastAPI backend on port 8000 (background thread)
- Streamlit dashboard on port 8501 (main)
"""
import sys
import os
import threading
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def start_api_backend():
    """Start FastAPI backend in background thread"""
    try:
        import uvicorn
        from src.config import settings

        # Set production environment for background process
        os.environ.setdefault("ENVIRONMENT", "production")

        # Run API on port 8000 (don't block - use threaded mode)
        uvicorn.run(
            "src.api.server:app",
            host="127.0.0.1",  # Only local access needed
            port=8000,
            workers=1,
            reload=False,
            log_level="info",
            use_colors=False,
            access_log=False
        )
    except Exception as e:
        print(f"⚠️ API Backend Error: {e}")


# Start API backend in background thread when Streamlit starts
if __name__ == "__main__":
    # Check if we're running in Streamlit Cloud
    is_streamlit_cloud = "STREAMLIT_SERVER_HEADLESS" in os.environ

    if is_streamlit_cloud:
        # Start API backend in background
        api_thread = threading.Thread(target=start_api_backend, daemon=True)
        api_thread.start()

        # Give API time to start
        time.sleep(2)


# Configure dashboard to use local API backend on Streamlit Cloud
os.environ.setdefault("API_URL", "http://127.0.0.1:8000")

# Import and run the dashboard app
from src.dashboard.app import *  # noqa