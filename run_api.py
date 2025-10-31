"""
Simple script to run the API server with proper module path

Supports multiple deployment environments:
- Development: auto-reload enabled, 1 worker
- Production: multi-worker mode, no reload, disabled colors for containers
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

    # Determine environment
    environment = os.getenv("ENVIRONMENT", "development")
    is_production = environment != "development"

    # Configure uvicorn based on environment
    if is_production:
        # Production configuration
        workers = int(os.getenv("WORKERS", settings.API_WORKERS))
        log_level = os.getenv("LOG_LEVEL", "info")
        use_colors = False  # Disabled for container compatibility

        uvicorn.run(
            "src.api.server:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            workers=workers,
            reload=False,
            log_level=log_level,
            use_colors=use_colors,
            access_log=True
        )
    else:
        # Development configuration with auto-reload
        uvicorn.run(
            "src.api.server:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=True,
            workers=1,
            log_level="debug",
            use_colors=True,
            access_log=True
        )