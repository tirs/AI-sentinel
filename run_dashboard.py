"""
Simple script to run the Streamlit dashboard with proper module path
"""
import sys
import os
from pathlib import Path
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from src.config import settings
    
    # Set PYTHONPATH environment variable for subprocess
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)
    
    # Run streamlit with modified environment
    subprocess.run([
        "streamlit", "run",
        str(project_root / "src" / "dashboard" / "app.py"),
        "--server.port", str(settings.DASHBOARD_PORT)
    ], env=env)