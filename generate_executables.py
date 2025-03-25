import subprocess  # Import subprocess to run shell commands
from pathlib import Path  # Import Path from pathlib for filesystem path manipulations

# Define the current script's path and the app folder path
CURR_PATH = Path(__file__)  # Get the path of the current script
app_folder_path = (
    CURR_PATH.parent / "app"
)  # Define the app directory relative to the current script


def generate_executable_non_gui(script_path, output_dir):
    """
    Generate a non-GUI executable from a Python script using PyInstaller.

    Args:
        script_path (str or Path): Path to the Python script to be converted.
        output_dir (str or Path): Directory where the executable will be saved.
    """
    script_path = Path(script_path)  # Convert script_path to a Path object
    output_dir = Path(output_dir)  # Convert output_dir to a Path object

    # Define the PyInstaller command for non-GUI executable
    command = [
        "pyinstaller",
        "--noconfirm",  # Overwrite existing build without confirmation
        "--onefile",  # Package the program into a single executable
        "--console",  # Attach a console window for standard I/O
        "--distpath",
        str(output_dir),  # Specify the output directory for the executable
        str(script_path),  # Path to the Python script
    ]

    subprocess.run(
        command, check=True
    )  # Execute the PyInstaller command and ensure it succeeds


def generate_executable_gui(script_path, output_dir):
    """
    Generate a GUI executable from a Python script using PyInstaller.

    Args:
        script_path (str or Path): Path to the Python script to be converted.
        output_dir (str or Path): Directory where the executable will be saved.
    """
    script_path = Path(script_path)  # Convert script_path to a Path object
    output_dir = Path(output_dir)  # Convert output_dir to a Path object

    # Define the PyInstaller command for GUI executable
    command = [
        "pyinstaller",
        "--noconfirm",  # Overwrite existing build without confirmation
        "--onefile",  # Package the program into a single executable
        "--windowed",  # Suppress the console window (for GUI applications)
        "--distpath",
        str(output_dir),  # Specify the output directory for the executable
        str(script_path),  # Path to the Python script
    ]

    subprocess.run(
        command, check=True
    )  # Execute the PyInstaller command and ensure it succeeds


def main():
    """
    Main function to generate executables for specified Python scripts.
    """
    output_dir = (
        CURR_PATH.parent / "output_executables"
    )  # Define the output directory for executables

    # Define paths to the Python scripts to be converted
    panorama_to_plane_PATH = app_folder_path / "panorama_to_plane-pitch.py"

    # Generate non-GUI executables
    generate_executable_non_gui(panorama_to_plane_PATH, output_dir)
    # generate_executable_gui(panorama_to_plane_gui_PATH, output_dir)


if __name__ == "__main__":
    main()  # Execute the main function when the script is run
