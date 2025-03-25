#!/usr/bin/env python3

import re
from pathlib import Path

def update_version():
    # Define file paths
    script_path = Path(__file__).resolve().parent.parent / "app" / "panorama_to_plane-pitch.py"
    version_file_path = Path(__file__).resolve().parent.parent / "VERSION"
    
    # Check if files exist
    if not script_path.exists():
        print(f"Error: Script file not found at {script_path}")
        return False
    
    if not version_file_path.exists():
        print(f"Error: VERSION file not found at {version_file_path}")
        return False
    
    # Read version from VERSION file
    with open(version_file_path, "r") as f:
        version = f.read().strip()
    
    print(f"Read version: {version} from VERSION file")
    
    # Read the script file
    with open(script_path, "r") as f:
        content = f.read()
    
    # Replace the VERSION variable using regex
    pattern = r'VERSION\s*=\s*"[^"]*"'
    replacement = f'VERSION = {version}'
    updated_content = re.sub(pattern, replacement, content)
    
    # Write back to the file
    with open(script_path, "w") as f:
        f.write(updated_content)
    
    print(f"Successfully updated VERSION to {version} in {script_path}")
    return True

if __name__ == "__main__":
    update_version()
    
