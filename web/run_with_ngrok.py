#!/usr/bin/env python3
"""Run Streamlit with ngrok tunnel for public access."""

import subprocess
import sys
import os
import time
import threading

def run_streamlit():
    """Run streamlit server."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_app.py")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port", "8501",
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
    ])

def run_ngrok():
    """Run ngrok tunnel."""
    try:
        from pyngrok import ngrok
        
        # Start tunnel
        public_url = ngrok.connect(8501)
        print("\n" + "="*60)
        print("🌐 PUBLIC URL (share this!):")
        print(f"   {public_url}")
        print("="*60 + "\n")
        
        # Keep running
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()
        
    except ImportError:
        print("❌ pyngrok not installed. Run: pip install pyngrok")
    except Exception as e:
        print(f"❌ ngrok error: {e}")

def main():
    print("🚀 Starting Placement Assistant with ngrok tunnel...")
    
    # Start streamlit in background
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    time.sleep(5)  # Wait for streamlit to start
    
    # Start ngrok
    run_ngrok()

if __name__ == "__main__":
    main()
