"""
Simple script to run the API server with proper module path
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    from src.config import settings

    # Disable reload in production (Railway, Docker, etc.)
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    workers = int(os.getenv("WORKERS", settings.API_WORKERS))

    uvicorn.run(
        "src.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=reload,
        workers=workers if not reload else 1
    )