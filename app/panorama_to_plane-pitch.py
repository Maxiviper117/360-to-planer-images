import os
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Caches for Precomputed Mappings
# --------------------------------------------------------------------------------------
yaw_mapping_cache = {}
pitch_mapping_cache = {}

def get_version():
    """
    Returns the version of the script.
    """
    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    if version_file.exists():
        with open(version_file, "r") as f:
            return f"version: {f.read().strip()}"
    else:
        return "unknown"


def get_yaw_mapping(pano_width, pano_height, yaw_angle):
    """
    Retrieve or compute the U, V mapping for a given (pano_width, pano_height, yaw_angle).
    Using a cache to avoid recomputations for the same dimensions/angle.
    """
    key = (pano_width, pano_height, yaw_angle)
    if key not in yaw_mapping_cache:
        yaw_mapping_cache[key] = precompute_yaw_mapping(
            pano_width, pano_height, yaw_angle
        )
    return yaw_mapping_cache[key]


def get_pitch_mapping(
    output_width, output_height, pitch_angle, pano_width, pano_height, fov_deg=90
):
    """
    Retrieve or compute the U, V pitch mapping for a given set of parameters.
    Using a cache to avoid recomputations.
    """
    key = (output_width, output_height, pitch_angle, pano_width, pano_height, fov_deg)
    if key not in pitch_mapping_cache:
        pitch_radians = np.radians(pitch_angle)
        pitch_mapping_cache[key] = precompute_pitch_mapping(
            W=output_width,
            H=output_height,
            FOV_rad=np.radians(fov_deg),
            pitch_radian=pitch_radians,
            pano_width=pano_width,
            pano_height=pano_height,
        )
    return pitch_mapping_cache[key]


# --------------------------------------------------------------------------------------
# Precompute Yaw Mapping
# --------------------------------------------------------------------------------------
def precompute_yaw_mapping(pano_width, pano_height, yaw_angle):
    """
    Precompute the mapping coordinates for horizontal (yaw) rotation.
    """
    logging.debug(f"[Yaw] Precomputing yaw mapping for yaw_angle: {yaw_angle} degrees")

    yaw_radians = np.radians(yaw_angle)  # Convert yaw angle from degrees to radians

    # Create meshgrid of pixel indices
    u, v = np.meshgrid(
        np.arange(pano_width, dtype=np.float32),
        np.arange(pano_height, dtype=np.float32),
        indexing="xy",
    )

    # Calculate the azimuthal angle (phi) for each pixel
    phi = (2 * np.pi * u / pano_width).astype(np.float32)

    # Apply yaw rotation, ensure it wraps around at 2π
    phi_rotated = (phi + yaw_radians) % (2 * np.pi)

    # Map the rotated azimuthal angle back to horizontal pixel coordinates
    U = (phi_rotated * pano_width) / (2 * np.pi)
    V = v  # Vertical coordinates remain unchanged as yaw doesn't affect the vertical dimension

    # Clip and ensure valid float32
    U = np.clip(U, 0, pano_width - 1).astype(np.float32)
    V = V.astype(np.float32)

    return U, V


# --------------------------------------------------------------------------------------
# Precompute Pitch Mapping
# --------------------------------------------------------------------------------------
def precompute_pitch_mapping(W, H, FOV_rad, pitch_radian, pano_width, pano_height):
    """
    Precompute the mapping coordinates for pitch transformation.
    """
    # Calculate the focal length based on the Field of View (FOV) and output image width (W)
    focal_length = (0.5 * W) / np.tan(FOV_rad / 2)

    # Create meshgrid of pixel indices for the output image
    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy",
    )

    # Calculate the 3D coordinates (x, y, z) in camera space for each pixel
    x = u - (W / 2.0)
    y = (H / 2.0) - v
    z = np.full_like(x, focal_length, dtype=np.float32)

    # Compute the Euclidean norm (magnitude) of each vector (x, y, z)
    norm = np.sqrt(x**2 + y**2 + z**2)

    # Normalize the vectors (x, y, z)
    x_norm = x / norm
    y_norm = y / norm
    z_norm = z / norm

    # Define the rotation matrix for pitch transformation around the x-axis
    R_pitch = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch_radian), -np.sin(pitch_radian)],
            [0, np.sin(pitch_radian), np.cos(pitch_radian)],
        ],
        dtype=np.float32,
    )

    # Stack vectors for matrix multiplication: shape (3, W*H)
    vectors = np.stack((x_norm, y_norm, z_norm), axis=0).reshape(3, -1)

    # Apply the pitch rotation matrix
    rotated = R_pitch @ vectors

    # Reshape back to (3, H, W)
    x_rot, y_rot, z_rot = rotated.reshape(3, H, W)

    # Convert the rotated Cartesian coordinates to spherical
    # theta_prime (polar) via arccos of z
    theta_prime = np.arccos(z_rot).astype(np.float32)
    # phi_prime (azimuth) via arctan2 of (y, x)
    phi_prime = (np.arctan2(y_rot, x_rot) % (2 * np.pi)).astype(np.float32)

    # Map azimuth to horizontal pixel coords in panorama
    U = (phi_prime * pano_width) / (2 * np.pi)
    # Map polar angle to vertical pixel coords in panorama
    V = (theta_prime * pano_height) / np.pi

    # Clip to valid panorama bounds
    U = np.clip(U, 0, pano_width - 1).astype(np.float32)
    V = np.clip(V, 0, pano_height - 1).astype(np.float32)

    return U, V


# --------------------------------------------------------------------------------------
# Process Yaw and Pitch
# --------------------------------------------------------------------------------------
def process_yaw_and_pitchs(
    pano_image, yaw_angle, pitch_angles, output_width, output_height, fov_deg=90
):
    """
    Process a single yaw angle and multiple pitch angles, returning all slices.
    """
    logging.debug(f"[Yaw/Pitch] Starting processing for yaw_angle={yaw_angle}")
    pano_height, pano_width, _ = pano_image.shape

    # --- Yaw Remap ---
    U_yaw, V_yaw = get_yaw_mapping(pano_width, pano_height, yaw_angle)
    rotated_pano = cv2.remap(
        pano_image,
        U_yaw,
        V_yaw,
        interpolation=cv2.INTER_LINEAR,
        # borderMode=cv2.BORDER_REFLECT,  # If reflect is needed; else BORDER_CONSTANT might be faster
        borderMode=cv2.BORDER_CONSTANT,
    )

    slices = []
    for pitch_angle in pitch_angles:
        logging.debug(f"[Pitch] Processing pitch_angle={pitch_angle}")
        U_pitch, V_pitch = get_pitch_mapping(
            output_width,
            output_height,
            pitch_angle,
            pano_width,
            pano_height,
            fov_deg,
        )
        slice_image = cv2.remap(
            rotated_pano,
            U_pitch,
            V_pitch,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        slices.append(slice_image)

    return slices


# --------------------------------------------------------------------------------------
# Process a Single Image
# --------------------------------------------------------------------------------------
def process_single_image(
    input_image_path,
    output_dir,
    yaw_angles,
    pitch_angles,
    output_width,
    output_height,
    num_workers=4,
    output_format="png",
    fov_deg=90,
):
    """
    Helper function to read a single image, process it, and save results.
    """

    logging.info(f"Loading image: {input_image_path}")
    # Keep the image in BGR format; no need to switch to RGB if not required.
    input_image = cv2.imread(str(input_image_path))
    if input_image is None:
        logging.error(f"Failed to read image: {input_image_path}")
        return

    base_name = input_image_path.stem

    tasks = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for yaw_angle in yaw_angles:
            # Submit an entire batch of pitch transformations for this yaw.
            tasks.append(
                executor.submit(
                    process_yaw_and_pitchs,
                    input_image,
                    yaw_angle,
                    pitch_angles,
                    output_width,
                    output_height,
                    fov_deg,
                )
            )

        # Gather results for each yaw angle
        for future, yaw_angle in zip(
            tqdm(tasks, desc="Processing yaw angles"), yaw_angles
        ):
            try:
                slices = future.result()
                # Save each pitch result
                for i, slice_image in enumerate(slices):
                    out_filename = f"{base_name}_{output_width}x{output_height}_yaw_{yaw_angle}_pitch_{pitch_angles[i]}.{output_format}"
                    output_file = output_dir / out_filename
                    cv2.imwrite(str(output_file), slice_image)
                    logging.debug(f"Saved {output_file}")
            except Exception as e:
                logging.error(f"Error processing yaw_angle {yaw_angle}: {e}")


# --------------------------------------------------------------------------------------
# Main Function (Directory or Single File)
# --------------------------------------------------------------------------------------
def main(
    input_path,
    output_path,
    yaw_angles,
    pitch_angles,
    output_width,
    output_height,
    num_workers=None,
    output_format="png",
    fov_deg=90,
    enable_file_logging=False,
):
    """
    Main function to process either a single image or an entire folder of images.
    If num_workers is None, automatically sets it to ~90% of CPU cores.
    """

    # Automatically assign worker count if not specified
    if num_workers is None:
        cpu_cores = os.cpu_count() or 1
        num_workers = max(1, int(cpu_cores * 0.9))
        logging.info(
            f"No num_workers specified. Using {num_workers} (~90% of CPU cores)."
        )
    else:
        logging.info(f"Using {num_workers} worker threads.")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir}")

    input_path_obj = Path(input_path)

    # Check if it's a directory -> process all images, otherwise just one
    if input_path_obj.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png"}
        all_images = [
            f for f in input_path_obj.rglob("*") if f.suffix.lower() in valid_exts
        ]
        if not all_images:
            logging.warning(f"No images found in directory: {input_path_obj}")
            return

        logging.info(f"Found {len(all_images)} images in folder: {input_path_obj}")
        for image_file in all_images:
            process_single_image(
                input_image_path=image_file,
                output_dir=output_dir,
                yaw_angles=yaw_angles,
                pitch_angles=pitch_angles,
                output_width=output_width,
                output_height=output_height,
                num_workers=num_workers,
                output_format=output_format,
                fov_deg=fov_deg,
            )
    else:
        # Process a single file
        process_single_image(
            input_image_path=input_path_obj,
            output_dir=output_dir,
            yaw_angles=yaw_angles,
            pitch_angles=pitch_angles,
            output_width=output_width,
            output_height=output_height,
            num_workers=num_workers,
            output_format=output_format,
            fov_deg=fov_deg,
        )

    logging.info("All processing completed.")


# --------------------------------------------------------------------------------------
# Utility: Check Pitch
# --------------------------------------------------------------------------------------
def check_pitch(value: str) -> int:
    """
    Validate the pitch angle is within 1-179 degrees.
    """
    try:
        pitch = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Pitch angle must be an integer, got '{value}'."
        )
    if not (1 <= pitch <= 179):
        raise argparse.ArgumentTypeError(
            f"Pitch angle must be between 1 and 179 degrees, got {pitch}."
        )
    return pitch


# --------------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process panorama images or an entire folder of images into planar projections."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input panorama image or folder of images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_images",
        help="Path to save the output images",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Output image format (png, jpg, jpeg)",
    )
    parser.add_argument(
        "--FOV",
        type=int,
        default=90,
        help="Field of View in degrees",
    )
    parser.add_argument(
        "--output_width",
        type=int,
        default=800,
        help="Width of the output image in pixels",
    )
    parser.add_argument(
        "--output_height",
        type=int,
        default=800,
        help="Height of the output image in pixels",
    )
    parser.add_argument(
        "--pitch_angles",
        nargs="+",
        type=check_pitch,
        default=[30, 60, 90, 120, 150],
        help="List of pitch angles in degrees (1-179). e.g. --pitch_angles 30 60 90",
    )
    parser.add_argument(
        "--yaw_angles",
        nargs="+",
        type=int,
        default=[0, 90, 180, 270],
        help="List of yaw angles in degrees (0-360). e.g. --yaw_angles 0 90 180 270",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker threads for parallel yaw processing. If not specified, uses ~90% of CPU cores.",
    )
    parser.add_argument(
        "--enable_file_logging",
        action="store_true",
        help="Enable logging to a file.",
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
        help="Show version information",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------------------------------
    # Configure logging
    # --------------------------------------------------------------------------------------
    log_file_path = Path(__file__).resolve().parent.parent / "logs" / "app.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(),  # Log to console
    ]
    if args.enable_file_logging:
        handlers.append(logging.FileHandler(log_file_path, mode="a"))  # Log to file

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        yaw_angles=args.yaw_angles,
        pitch_angles=args.pitch_angles,
        output_width=args.output_width,
        output_height=args.output_height,
        num_workers=args.num_workers,
        output_format=args.output_format,
        fov_deg=args.FOV,
        enable_file_logging=args.enable_file_logging,
    )


# uv run --python 3.12 .\app\panorama_to_plane-pitch.py --input_path "./test-images" --output_path "./output_images" --output_format "png" --output_width 800 --output_height 800 --pitch_angles 30 60 90 120 150 --yaw_angles 0 90 180 270
# uv run --python 3.12 .\app\panorama_to_plane-pitch.py --input_path "./test-images" --output_path "./output_images" --output_width 800 --output_height 800 --pitch_angles 30 60 90 120 150 --yaw_angles 0 90 180 270
