#!/usr/bin/env python3
"""
Run Streamlit through Jupyter's proxy for cloud machines.
Access via: https://45.112.150.236/user/<username>/proxy/8501/
"""

import subprocess
import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_app.py")
    
    # Run streamlit with settings for Jupyter proxy
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port", "8501",
        "--server.address", "127.0.0.1",  # Localhost only
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
        "--server.baseUrlPath", "",  # Will be handled by proxy
    ])

if __name__ == "__main__":
    main()
