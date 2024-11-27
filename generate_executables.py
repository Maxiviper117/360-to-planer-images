import subprocess
from pathlib import Path

def generate_executable_non_gui(script_path, output_dir):
    script_path = Path(script_path)
    output_dir = Path(output_dir)
    command = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--console",
        "--distpath", str(output_dir),
        str(script_path)
    ]
    subprocess.run(command, check=True)
    
def generate_executable_gui(script_path, output_dir):
    script_path = Path(script_path)
    output_dir = Path(output_dir)
    command = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--distpath", str(output_dir),
        str(script_path)
    ]
    subprocess.run(command, check=True)

def main():
    output_dir = Path("output")
    
    generate_executable_non_gui("app/panorama_to_plane.py", output_dir)
    generate_executable_gui("app/panorama_to_plane-gui.py", output_dir)

if __name__ == "__main__":
    main()
