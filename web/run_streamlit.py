#!/usr/bin/env python3
"""Launch script for Streamlit app."""

import subprocess
import sys
import os

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_app.py")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        app_path,
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()
