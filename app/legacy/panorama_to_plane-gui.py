import os
import threading
from pathlib import Path
import cv2
import numpy as np
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json  # Added import

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # INFO level for general messages
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_rotation_matrix(yaw_radian: float, pitch_radian: float) -> np.ndarray:
    R_yaw = np.array([
        [np.cos(yaw_radian), 0, np.sin(yaw_radian)],
        [0, 1, 0],
        [-np.sin(yaw_radian), 0, np.cos(yaw_radian)]
    ], dtype=np.float32)

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_radian), -np.sin(pitch_radian)],
        [0, np.sin(pitch_radian), np.cos(pitch_radian)]
    ], dtype=np.float32)

    return np.dot(R_pitch, R_yaw)

@lru_cache(maxsize=None)
def precompute_mapping(W: int, H: int, FOV_rad: float, yaw_radian: float, pitch_radian: float, pano_width: int, pano_height: int) -> tuple:
    f = (0.5 * W) / np.tan(FOV_rad / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    x = u - (W / 2.0)
    y = (H / 2.0) - v
    z = np.full_like(x, f, dtype=np.float32)

    norm = np.sqrt(x**2 + y**2 + z**2)
    x_norm = x / norm
    y_norm = y / norm
    z_norm = z / norm

    R = get_rotation_matrix(yaw_radian, pitch_radian)
    vectors = np.stack((x_norm, y_norm, z_norm), axis=0)  # Shape: (3, H, W)
    rotated = R @ vectors.reshape(3, -1)
    rotated = rotated.reshape(3, H, W)
    x_rot, y_rot, z_rot = rotated

    theta_prime = np.arccos(z_rot).astype(np.float32)
    phi_prime = (np.arctan2(y_rot, x_rot) % (2 * np.pi)).astype(np.float32)

    U = (phi_prime * pano_width) / (2 * np.pi)
    V = (theta_prime * pano_height) / np.pi

    U = np.clip(U, 0, pano_width - 1).astype(np.float32)
    V = np.clip(V, 0, pano_height - 1).astype(np.float32)

    return U, V

def interpolate_color(U: np.ndarray, V: np.ndarray, img: np.ndarray, method: str = 'bilinear') -> np.ndarray:
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC
    }
    interp = interpolation_methods.get(method, cv2.INTER_LINEAR)
    remapped = cv2.remap(img, U, V, interpolation=interp, borderMode=cv2.BORDER_REFLECT)
    return remapped

def panorama_to_plane(pano_array: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    return interpolate_color(U, V, pano_array)

def process_image_batch(image_path: Path, args, output_path: Path, precomputed_mappings: dict, progress_callback):
    logging.info(f"Processing {image_path}...")
    try:
        pano_array = cv2.imread(str(image_path))
        if pano_array is None:
            logging.error(f"Failed to read image {image_path}. Skipping.")
            return
        pano_array = cv2.cvtColor(pano_array, cv2.COLOR_BGR2RGB)
        pano_height, pano_width, _ = pano_array.shape
        file_name = image_path.stem

        for yaw in args['yaw_angles']:
            logging.debug(f"Processing {image_path} with yaw {yaw}°...")
            U, V = precomputed_mappings[yaw]

            output_image_array = panorama_to_plane(pano_array, U, V)

            output_format = args['output_format'] if args['output_format'] else image_path.suffix[1:]

            output_image_name = f"{file_name}_pitch{args['pitch']}_yaw{yaw}_fov{args['FOV']}.{output_format}"
            output_image_path = output_path / output_image_name

            output_bgr = cv2.cvtColor(output_image_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_image_path), output_bgr)
            logging.info(f"Saved output image to {output_image_path}")

            progress_callback()

    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")

class PanoramaProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Panorama to Plane Projection")
        self.profile_file = Path.home() / ".panorama_to_plane" / "profiles.json"  # Save profiles.json in user's home directory
        self.profile_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        self.load_profiles()  # Load profiles on initialization
        self.create_widgets()
        self.setup_logging()
        self.create_profile_widgets()  # Create profile management UI

        # Remove dark theme configurations to use default tkinter theme
        # style = ttk.Style()
        # style.theme_use('clam')
        # style.configure('.', background='#2e2e2e', foreground='white')
        # style.configure('TButton', background='#4e4e4e')
        # style.configure('TLabel', background='#2e2e2e', foreground='white')
        # style.configure('TEntry', foreground='black')  # Set input text color to black
        # style.configure('TCombobox', foreground='black')  # Set Combobox text color to black
        # self.log_text.config(bg='#2e2e2e', fg='white')

    def create_widgets(self):
        # ...existing code...

        # Create main frame with two columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left Column
        left_column = ttk.Frame(main_frame, padding="10")
        left_column.grid(row=0, column=0, sticky="nsew")
        left_column.columnconfigure(0, weight=1)

        # Input Directory
        input_frame = ttk.Frame(left_column, padding="10")
        input_frame.grid(row=0, column=0, sticky="ew")
        # ...existing input_frame code...
        ttk.Label(input_frame, text="Input Directory:").grid(row=0, column=0, columnspan=3, sticky="w")
        self.input_dir_var = tk.StringVar()
        self.input_dir_entry = ttk.Entry(input_frame, textvariable=self.input_dir_var, width=50)
        self.input_dir_entry.grid(row=1, column=0, padx=5, sticky="w")
        ttk.Button(input_frame, text="Browse", command=self.browse_input).grid(row=1, column=1, padx=5, sticky="w")
        # ...existing code...

        # Output Directory
        output_frame = ttk.Frame(left_column, padding="10")
        output_frame.grid(row=1, column=0, sticky="ew")
        # ...existing output_frame code...
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, columnspan=3, sticky="w")
        self.output_dir_var = tk.StringVar()
        self.output_dir_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50)
        self.output_dir_entry.grid(row=1, column=0, padx=5, sticky="w")
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=1, column=1, padx=5, sticky="w")
        # ...existing code...

        # FOV
        fov_frame = ttk.Frame(left_column, padding="10")
        fov_frame.grid(row=2, column=0, sticky="ew")
        # ...existing fov_frame code...
        ttk.Label(fov_frame, text="Field of View (FOV) [degrees]:").grid(row=0, column=0, columnspan=2, sticky="w")
        self.fov_var = tk.IntVar(value=90)
        self.fov_spin = ttk.Spinbox(fov_frame, from_=10, to=180, textvariable=self.fov_var, width=10)
        self.fov_spin.grid(row=1, column=0, padx=5, sticky="w")

        ttk.Label(fov_frame, text="Output Width [px]:").grid(row=2, column=0, columnspan=2, sticky="w")
        self.width_var = tk.IntVar(value=1000)
        self.width_spin = ttk.Spinbox(fov_frame, from_=100, to=5000, textvariable=self.width_var, width=10)
        self.width_spin.grid(row=3, column=0, padx=5, sticky="w")

        ttk.Label(fov_frame, text="Output Height [px]:").grid(row=4, column=0, columnspan=2, sticky="w")
        self.height_var = tk.IntVar(value=1500)
        self.height_spin = ttk.Spinbox(fov_frame, from_=100, to=5000, textvariable=self.height_var, width=10)
        self.height_spin.grid(row=5, column=0, padx=5, sticky="w")
        # ...existing code...

        # Pitch
        pitch_frame = ttk.Frame(left_column, padding="10")
        pitch_frame.grid(row=3, column=0, sticky="ew")
        # ...existing pitch_frame code...
        ttk.Label(pitch_frame, text="Pitch Angle [degrees]: (Controls vertical view)").grid(row=0, column=0, columnspan=2, sticky="w")
        self.pitch_var = tk.IntVar(value=90)
        self.pitch_spin = ttk.Spinbox(pitch_frame, from_=1, to=179, textvariable=self.pitch_var, width=10)
        self.pitch_spin.grid(row=1, column=0, padx=5, sticky="w")
        # ...existing code...

        # Yaw Angles
        yaw_frame = ttk.Frame(left_column, padding="10")
        yaw_frame.grid(row=4, column=0, sticky="ew")
        # ...existing yaw_frame code...
        ttk.Label(yaw_frame, text="Yaw Angles [degrees, comma-separated]: (Controls horizontal viewpoints)").grid(row=0, column=0, columnspan=2, sticky="w")
        self.yaw_var = tk.StringVar(value="0,60,120,180,240,300")
        self.yaw_entry = ttk.Entry(yaw_frame, textvariable=self.yaw_var, width=50)
        self.yaw_entry.grid(row=1, column=0, padx=5, sticky="w")
        # ...existing code...

        # Output Format
        format_frame = ttk.Frame(left_column, padding="10")
        format_frame.grid(row=5, column=0, sticky="ew")
        # ...existing format_frame code...
        ttk.Label(format_frame, text="Output Format:").grid(row=0, column=0, columnspan=2, sticky="w")
        self.format_var = tk.StringVar(value="png")
        self.format_combo = ttk.Combobox(format_frame, textvariable=self.format_var, values=["png", "jpg", "jpeg"], state="readonly", width=10)
        self.format_combo.grid(row=1, column=0, padx=5, sticky="w")
        # ...existing code...

        # Number of Workers
        worker_frame = ttk.Frame(left_column, padding="10")
        worker_frame.grid(row=6, column=0, sticky="ew")
        # ...existing worker_frame code...
        ttk.Label(worker_frame, text="Number of Workers:").grid(row=0, column=0, columnspan=2, sticky="w")
        self.worker_var = tk.IntVar(value=max(1, os.cpu_count() or 1))
        self.worker_spin = ttk.Spinbox(worker_frame, from_=1, to=os.cpu_count()*2, textvariable=self.worker_var, width=10)
        self.worker_spin.grid(row=1, column=0, padx=5, sticky="w")
        ttk.Label(worker_frame, text="(Refers to multithreading processing. More workers consume more CPU)").grid(row=2, column=0, columnspan=2, sticky="w")
        # ...existing code...

        # Right Column
        right_column = ttk.Frame(main_frame, padding="10")
        right_column.grid(row=0, column=1, sticky="nsew")
        right_column.columnconfigure(0, weight=1)
        right_column.rowconfigure(3, weight=1)  # For log_frame to expand
        self.right_column = right_column  # Make right_column accessible

        # Start Button
        start_frame = ttk.Frame(right_column, padding="10")
        start_frame.grid(row=0, column=0, sticky="ew")
        # ...existing start_frame code...
        self.start_button = ttk.Button(start_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, pady=10)
        # ...existing code...

        # Progress Bar
        progress_frame = ttk.Frame(right_column, padding="10")
        progress_frame.grid(row=1, column=0, sticky="ew")
        # ...existing progress_frame code...
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        # ...existing code...

        # Status Log
        log_frame = ttk.Frame(right_column, padding="10")
        log_frame.grid(row=2, column=0, sticky="nsew")
        # ...existing log_frame code...
        self.log_text = tk.Text(log_frame, height=10, state='disabled')
        self.log_text.pack(fill="both", expand=True)
        # ...existing code...

        # Profile Management
        profile_frame = ttk.Frame(right_column, padding="10")
        profile_frame.grid(row=3, column=0, sticky="ew")
        # ...existing profile_frame code...

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        left_column.rowconfigure(6, weight=1)
        right_column.rowconfigure(3, weight=1)

    def setup_logging(self):
        # Redirect logging to the Tkinter Text widget
        self.logger = logging.getLogger()
        self.logger.handlers = []
        handler = TextHandler(self.log_text)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def browse_input(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir_var.set(directory)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def start_processing(self):
        input_dir = self.input_dir_var.get()
        output_dir = self.output_dir_var.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        try:
            FOV = int(self.fov_var.get())
            output_width = int(self.width_var.get())
            output_height = int(self.height_var.get())
            pitch = int(self.pitch_var.get())
            yaw_angles = [int(yaw.strip()) for yaw in self.yaw_var.get().split(",") if yaw.strip().isdigit()]
            output_format = self.format_var.get()
            num_workers = int(self.worker_var.get())

            if not (1 <= pitch <= 179):
                raise ValueError("Pitch must be between 1 and 179 degrees.")
            for yaw in yaw_angles:
                if not (0 <= yaw <= 360):
                    raise ValueError("Yaw angles must be between 0 and 360 degrees.")

        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))
            return

        args = {
            'FOV': FOV,
            'output_width': output_width,
            'output_height': output_height,
            'pitch': pitch,
            'yaw_angles': sorted(list(set(yaw_angles))),
            'output_format': output_format
        }

        # Disable the start button to prevent multiple clicks
        self.start_button.config(state='disabled')
        self.progress_var.set(0)
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

        # Start processing in a separate thread
        threading.Thread(target=self.process_images, args=(input_dir, output_dir, args, num_workers), daemon=True).start()

    def process_images(self, input_dir, output_dir, args, num_workers):
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created output directory {output_path}.")
            except Exception as e:
                logging.error(f"Failed to create output directory: {e}")
                self.enable_start_button()
                return

        # Collect image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(ext))

        if not image_paths:
            logging.warning(f"No images found in {input_path} with extensions {image_extensions}.")
            self.enable_start_button()
            return

        # Determine number of workers
        max_workers = max(1, min(num_workers, os.cpu_count() or 1))

        logging.info(f"Using {max_workers} worker(s) for processing.")

        # Precompute mappings for each yaw angle
        precomputed_mappings = {}
        FOV_rad = np.radians(args['FOV'])
        pitch_rad = np.radians(args['pitch'])

        # Read a sample image to get panorama dimensions
        sample_image_path = image_paths[0]
        sample_pano = cv2.imread(str(sample_image_path))
        if sample_pano is None:
            logging.error(f"Failed to read sample image {sample_image_path} for precomputing mappings.")
            self.enable_start_button()
            return
        pano_height, pano_width, _ = sample_pano.shape

        for yaw in args['yaw_angles']:
            yaw_rad = np.radians(yaw)
            U, V = precompute_mapping(
                W=args['output_width'],
                H=args['output_height'],
                FOV_rad=FOV_rad,
                yaw_radian=yaw_rad,
                pitch_radian=pitch_rad,
                pano_width=pano_width,
                pano_height=pano_height
            )
            precomputed_mappings[yaw] = (U, V)

        total_tasks = len(image_paths) * len(args['yaw_angles'])
        self.tasks_completed = 0

        def progress_callback():
            self.tasks_completed += 1
            progress = (self.tasks_completed / total_tasks) * 100
            self.progress_var.set(progress)

        logging.info(f"Starting processing of {len(image_paths)} images with {len(args['yaw_angles'])} yaw angles each.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_image_batch, image_path, args, output_path, precomputed_mappings, progress_callback)
                for image_path in image_paths
            ]
            for _ in futures:
                pass  # Progress is handled via callbacks

        logging.info("Processing completed.")
        
        # Show the output folder location to the user
        def show_output_location():
            messagebox.showinfo("Processing Completed", f"Output images are saved in:\n{output_dir}")
        
        self.root.after(0, show_output_location)
        
        self.enable_start_button()

    def enable_start_button(self):
        self.start_button.config(state='normal')

    def create_profile_widgets(self):
        # Move profile_frame to right_column
        profile_frame = ttk.Frame(self.right_column, padding="10")
        profile_frame.grid(row=3, column=0, sticky="ew")
        # ...existing profile_frame code...

        ttk.Label(profile_frame, text="Profile Name:").grid(row=0, column=0, sticky="w")
        self.profile_name_var = tk.StringVar()
        self.profile_name_entry = ttk.Entry(profile_frame, textvariable=self.profile_name_var, width=30)
        self.profile_name_entry.grid(row=0, column=1, padx=5, sticky="w")

        ttk.Button(profile_frame, text="Save Profile", command=self.save_profile).grid(row=0, column=2, padx=5, sticky="w")
        ttk.Button(profile_frame, text="Load Profile", command=self.load_profile).grid(row=0, column=3, padx=5, sticky="w")

        ttk.Label(profile_frame, text="Select Profile:").grid(row=1, column=0, sticky="w")
        self.selected_profile = tk.StringVar()
        self.profile_combo = ttk.Combobox(profile_frame, textvariable=self.selected_profile, values=self.get_profile_names(), state="readonly", width=28)
        self.profile_combo.grid(row=1, column=1, padx=5, sticky="w")
        ttk.Button(profile_frame, text="Delete Profile", command=self.delete_profile).grid(row=1, column=2, padx=5, sticky="w")

        # Display the profiles.json path
        ttk.Label(profile_frame, text=f"Profiles file located at: {self.profile_file}").grid(row=2, column=0, columnspan=4, sticky="w")

    def get_profile_names(self):
        if self.profile_file.exists():
            with open(self.profile_file, 'r') as f:
                profiles = json.load(f)
            return list(profiles.keys())
        return []

    def load_profiles(self):
        self.profiles = {}
        if self.profile_file.exists():
            with open(self.profile_file, 'r') as f:
                self.profiles = json.load(f)

    def save_profiles_to_file(self):
        with open(self.profile_file, 'w') as f:
            json.dump(self.profiles, f, indent=4)

    def save_profile(self):
        name = self.profile_name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Profile name cannot be empty.")
            return
        self.profiles[name] = {
            'FOV': self.fov_var.get(),
            'output_width': self.width_var.get(),
            'output_height': self.height_var.get(),
            'pitch': self.pitch_var.get(),
            'yaw_angles': self.yaw_var.get(),
            'output_format': self.format_var.get(),
            'num_workers': self.worker_var.get()
        }
        self.save_profiles_to_file()
        self.profile_combo['values'] = self.get_profile_names()
        messagebox.showinfo("Success", f"Profile '{name}' saved successfully.")

    def load_profile(self):
        name = self.selected_profile.get()
        if not name:
            messagebox.showerror("Error", "No profile selected.")
            return
        profile = self.profiles.get(name, {})
        self.fov_var.set(profile.get('FOV', 90))
        self.width_var.set(profile.get('output_width', 1000))
        self.height_var.set(profile.get('output_height', 1500))
        self.pitch_var.set(profile.get('pitch', 90))
        self.yaw_var.set(profile.get('yaw_angles', "0,60,120,180,240,300"))
        self.format_var.set(profile.get('output_format', "png"))
        self.worker_var.set(profile.get('num_workers', max(1, os.cpu_count() or 1)))
        messagebox.showinfo("Success", f"Profile '{name}' loaded successfully.")

    def delete_profile(self):
        name = self.selected_profile.get()
        if not name:
            messagebox.showerror("Error", "No profile selected.")
            return
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete profile '{name}'?"):
            del self.profiles[name]
            self.save_profiles_to_file()
            self.profile_combo['values'] = self.get_profile_names()
            self.selected_profile.set('')
            messagebox.showinfo("Success", f"Profile '{name}' deleted successfully.")

class TextHandler(logging.Handler):
    """
    This class allows logging to be redirected to a Tkinter Text widget.
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.config(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.config(state='disabled')
        self.text_widget.after(0, append)

def main():
    root = tk.Tk()
    app = PanoramaProcessorGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
