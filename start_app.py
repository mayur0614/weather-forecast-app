#!/usr/bin/env python3
"""
Startup script for RainCheck Weather Forecast Application
This script helps you start both the API server and the Streamlit frontend
"""

import subprocess
import sys
import time
import webbrowser
import threading
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import fastapi
        import uvicorn
        import requests
        import plotly
        import tensorflow
        import pandas
        import numpy
        import sklearn
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting API server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API server: {e}")
        return False
    except KeyboardInterrupt:
        print("🛑 API server stopped")
        return True

def start_streamlit_app():
    """Start the Streamlit frontend"""
    print("🌐 Starting Streamlit frontend...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", 
            "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit app: {e}")
        return False
    except KeyboardInterrupt:
        print("🛑 Streamlit app stopped")
        return True

def open_browser_delayed():
    """Open browser after a delay"""
    time.sleep(3)
    webbrowser.open("http://localhost:8501")

def main():
    """Main function"""
    print("🌦️ RainCheck Weather Forecast - Startup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("api.py").exists() or not Path("app.py").exists():
        print("❌ Please run this script from the project directory containing api.py and app.py")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("\nChoose an option:")
    print("1. Start API server only")
    print("2. Start Streamlit frontend only")
    print("3. Start both (recommended)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        start_api_server()
    elif choice == "2":
        start_streamlit_app()
    elif choice == "3":
        print("\n🚀 Starting both API server and Streamlit frontend...")
        print("📝 Note: You'll need to run this script twice - once for each service")
        print("   Or use two terminal windows")
        
        # Start API in a separate thread
        api_thread = threading.Thread(target=start_api_server)
        api_thread.daemon = True
        api_thread.start()
        
        # Wait a bit for API to start
        time.sleep(2)
        
        # Open browser
        browser_thread = threading.Thread(target=open_browser_delayed)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start Streamlit
        start_streamlit_app()
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
