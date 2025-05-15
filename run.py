# run.py
"""
Launch script for the Investment Banking RAG Bot.
Configures Streamlit to avoid file watcher issues.
"""

import os
import sys
import subprocess

def main():
    """Run the Streamlit app with proper configuration"""
    # Set environment variables to prevent Streamlit file watcher issues
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the app.py file
    app_path = os.path.join(script_dir, "app.py")
    
    # Build the command
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.fileWatcherType=none",
        "--logger.level=error"
    ]
    
    # Run the command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "__main__":
    main()
