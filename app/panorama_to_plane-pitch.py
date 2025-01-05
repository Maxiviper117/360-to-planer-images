import os
from pathlib import Path
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

# --------------------------------------------------------------------------------------
# Configure logging
# --------------------------------------------------------------------------------------
log_file_path = Path(__file__).resolve().parent.parent / "logs" / "app.log"
log_file_path.parent.mkdir(
    parents=True, exist_ok=True
)  # Ensure the logs directory exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(log_file_path, mode="a"),  # Log to file
    ],
)


# --------------------------------------------------------------------------------------
# Precompute Yaw Mapping
# --------------------------------------------------------------------------------------
def precompute_yaw_mapping(pano_width, pano_height, yaw_angle):
    """
    Precompute the mapping coordinates for horizontal (yaw) rotation.
    """
    logging.info(
        f"Precomputing yaw mapping for yaw_angle: {yaw_angle} degrees"
    )  # Log the start of yaw mapping with the specified angle

    yaw_radians = np.radians(yaw_angle)  # Convert yaw angle from degrees to radians

    # Create meshgrid of pixel indices
    u, v = np.meshgrid(
        np.arange(pano_width), np.arange(pano_height), indexing="xy"
    )  # Generate grid of u (horizontal) and v (vertical) coordinates
    u = u.astype(
        np.float32
    )  # Convert u coordinates to float32 for precise calculations
    v = v.astype(
        np.float32
    )  # Convert v coordinates to float32 for precise calculations

    # Calculate the azimuthal angle for each pixel
    phi = (2 * np.pi * u / pano_width).astype(
        np.float32
    )  # Compute the azimuthal angle (phi) corresponding to each u coordinate
    # Apply yaw rotation, ensure it wraps around at 2π
    phi_rotated = (phi + yaw_radians) % (
        2 * np.pi
    )  # Rotate the azimuthal angle by yaw_radians and wrap within [0, 2π)

    # Map the rotated azimuthal angle back to horizontal pixel coordinates
    U = (phi_rotated * pano_width) / (
        2 * np.pi
    )  # Scale the rotated phi back to the panorama's horizontal pixel range
    V = v  # Vertical coordinates remain unchanged as yaw rotation doesn't affect vertical positioning

    # Clip and ensure valid float32
    U = np.clip(U, 0, pano_width - 1).astype(
        np.float32
    )  # Ensure U values are within [0, pano_width - 1] and maintain float32 type
    V = V.astype(np.float32)  # Ensure V remains as float32 for consistency

    logging.debug(
        f"Yaw mapping completed with shapes U: {U.shape}, V: {V.shape}"
    )  # Log the completion of yaw mapping with the shapes of U and V
    return U, V  # Return the horizontal and vertical mapping coordinates


# --------------------------------------------------------------------------------------
# Precompute Pitch Mapping
# --------------------------------------------------------------------------------------
def precompute_pitch_mapping(W, H, FOV_rad, pitch_radian, pano_width, pano_height):
    """
    Precompute the mapping coordinates for pitch transformation.
    """
    # Calculate the focal length based on the Field of View (FOV) and output image width (W)
    focal_length = (0.5 * W) / np.tan(FOV_rad / 2)

    # Create a meshgrid of pixel indices for the output image
    # 'u' corresponds to the horizontal axis (width), and 'v' corresponds to the vertical axis (height)
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

    # Convert the meshgrid arrays to float32 for precise arithmetic operations
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Calculate the 3D coordinates (x, y, z) in camera space for each pixel
    # Center the coordinates by subtracting half the width and height
    x = u - (W / 2.0)  # Shift the u coordinates to be centered around zero
    y = (H / 2.0) - v  # Shift and flip the v coordinates to center and align vertically
    z = np.full_like(
        x, focal_length, dtype=np.float32
    )  # Set the z coordinate to the focal length for all pixels

    # Compute the Euclidean norm (magnitude) of each vector (x, y, z)
    norm = np.sqrt(x**2 + y**2 + z**2)

    # Normalize the vectors to have a unit length (1), preserving their direction
    x_norm = x / norm  # Normalize the x component
    y_norm = y / norm  # Normalize the y component
    z_norm = z / norm  # Normalize the z component

    # Define the rotation matrix for pitch transformation around the x-axis
    R_pitch = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch_radian), -np.sin(pitch_radian)],
            [0, np.sin(pitch_radian), np.cos(pitch_radian)],
        ],
        dtype=np.float32,
    )

    # Stack the normalized vectors into a single array and reshape for matrix multiplication
    vectors = np.stack((x_norm, y_norm, z_norm), axis=0).reshape(
        3, -1
    )  # Shape: (3, W*H)

    # Apply the pitch rotation matrix to the vectors
    rotated = R_pitch @ vectors  # Matrix multiplication to rotate the vectors

    # Reshape the rotated vectors back to the original 3D grid shape
    x_rot, y_rot, z_rot = rotated.reshape(
        3, H, W
    )  # Each rotated component has shape (H, W)

    # Convert the rotated Cartesian coordinates to spherical coordinates
    # Calculate the polar angle (theta') using the arccosine of the z component
    theta_prime = np.arccos(z_rot).astype(np.float32)  # Polar angle in radians

    # Calculate the azimuthal angle (phi') using the arctangent of y_rot over x_rot
    # The modulo operation ensures the angle wraps around at 2π radians
    phi_prime = (np.arctan2(y_rot, x_rot) % (2 * np.pi)).astype(
        np.float32
    )  # Azimuthal angle in radians

    # Map the azimuthal angle back to horizontal pixel coordinates in the panorama
    U = (phi_prime * pano_width) / (2 * np.pi)  # Scale phi' to panorama width

    # Map the polar angle back to vertical pixel coordinates in the panorama
    V = (theta_prime * pano_height) / np.pi  # Scale theta' to panorama height

    # Clip the U coordinates to ensure they fall within the panorama's horizontal bounds
    U = np.clip(U, 0, pano_width - 1).astype(np.float32)

    # Clip the V coordinates to ensure they fall within the panorama's vertical bounds
    V = np.clip(V, 0, pano_height - 1).astype(np.float32)

    # Return the precomputed horizontal (U) and vertical (V) mapping coordinates
    return U, V


# --------------------------------------------------------------------------------------
# Process Yaw and Pitch
# --------------------------------------------------------------------------------------
def process_yaw_and_pitchs(
    pano_image, yaw_angle, pitch_angles, output_width, output_height
):
    """
    Process a single yaw angle and multiple pitch angles, returning all slices.
    """
    logging.info(f"Starting processing for yaw_angle: {yaw_angle} degrees")

    pano_height, pano_width, _ = pano_image.shape
    logging.debug(f"Panorama dimensions - Width: {pano_width}, Height: {pano_height}")

    # Precompute yaw mapping
    logging.info("Precomputing yaw mapping")
    U_yaw, V_yaw = precompute_yaw_mapping(pano_width, pano_height, yaw_angle)
    logging.debug(f"Yaw mapping shapes - U_yaw: {U_yaw.shape}, V_yaw: {V_yaw.shape}")

    # Apply yaw rotation to the panorama image
    logging.info("Applying yaw rotation to the panorama image")
    rotated_pano = cv2.remap(
        pano_image,
        U_yaw,
        V_yaw,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    logging.debug("Yaw rotation applied successfully")

    slices = []
    for idx, pitch_angle in enumerate(pitch_angles):
        logging.info(
            f"Processing pitch_angle {idx + 1}/{len(pitch_angles)}: {pitch_angle} degrees"
        )
        pitch_radians = np.radians(pitch_angle)

        # Precompute pitch mapping
        logging.debug("Precomputing pitch mapping")
        U_pitch, V_pitch = precompute_pitch_mapping(
            W=output_width,
            H=output_height,
            FOV_rad=np.radians(90),  # Example FOV (90 degrees)
            pitch_radian=pitch_radians,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        logging.debug(
            f"Pitch mapping shapes - U_pitch: {U_pitch.shape}, V_pitch: {V_pitch.shape}"
        )

        # Apply pitch rotation to the rotated panorama image
        logging.info("Applying pitch rotation to the rotated panorama image")
        slice_image = cv2.remap(
            rotated_pano,
            U_pitch,
            V_pitch,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        logging.debug(f"Pitch rotation applied for pitch_angle: {pitch_angle} degrees")

        slices.append(slice_image)
        logging.info(
            f"Slice for pitch_angle {pitch_angle} degrees appended to slices list"
        )

    logging.info(f"Completed processing for yaw_angle: {yaw_angle} degrees")
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
):
    """
    Helper function to read a single image, process it, and save results.
    """
    logging.info(f"Loading image: {input_image_path}")
    input_image = cv2.imread(str(input_image_path))

    if input_image is None:
        logging.error(f"Failed to read image: {input_image_path}")
        return

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    logging.info(f"Image loaded and converted to RGB: {input_image_path.name}")

    # Extract the base name of the input image file (without extension)
    base_name = input_image_path.stem
    logging.debug(f"Base name extracted: {base_name}")

    tasks = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for yaw_angle in yaw_angles:
            tasks.append(
                executor.submit(
                    process_yaw_and_pitchs,
                    input_image,
                    yaw_angle,
                    pitch_angles,
                    output_width,
                    output_height,
                )
            )

        # Gather results for each yaw angle
        for future, yaw_angle in zip(
            tqdm(tasks, desc="Processing yaw angles"), yaw_angles
        ):
            try:
                slices = future.result()
                logging.info(f"Completed processing for yaw_angle: {yaw_angle}")

                # Save each pitch result
                for i, slice_image in enumerate(slices):
                    out_filename = f"{base_name}_yaw_{yaw_angle}_pitch_{pitch_angles[i]}.{output_format}"
                    output_file = output_dir / out_filename
                    slice_bgr = cv2.cvtColor(slice_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_file), slice_bgr)
                    logging.info(f"Saved {output_file}")
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
):
    """
    Main function to process either a single image or an entire folder of images.
    If num_workers is None, automatically sets it to ~90% of CPU cores.
    """
    # Automatically assign worker count if num_workers is not specified
    if num_workers is None:
        cpu_cores = os.cpu_count() or 1  # fallback to 1 if somehow None
        num_workers = max(1, int(cpu_cores * 0.9))
        logging.info(
            f"No num_workers specified. Using {num_workers} threads (~90% of CPU cores)."
        )
    else:
        logging.info(f"Using {num_workers} worker threads as specified.")

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
        help="Field of View in degrees (currently unused in code, just a placeholder)",
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
        help="List of pitch angles in degrees (1-179). Example: --pitch_angles 30 60 90",
    )
    parser.add_argument(
        "--yaw_angles",
        nargs="+",
        type=int,
        default=[0, 90, 180, 270],
        help="List of yaw angles in degrees (0-360). Example: --yaw_angles 0 90 180 270",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker threads for parallel yaw processing. If not specified, uses ~90%% of CPU cores.",
    )

    args = parser.parse_args()

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        yaw_angles=args.yaw_angles,
        pitch_angles=args.pitch_angles,
        output_width=args.output_width,
        output_height=args.output_height,
        num_workers=args.num_workers,
        output_format=args.output_format,
    )

# uv run --python 3.12 .\app\panorama_to_plane-pitch.py --input_path "./test-images" --output_path "./output_images" --output_format "png" --output_width 800 --output_height 800 --pitch_angles 30 60 90 120 150 --yaw_angles 0 90 180 270
# uv run --python 3.12 .\app\panorama_to_plane-pitch.py --input_path "./test-images" --output_path "./output_images" --output_width 800 --output_height 800 --pitch_angles 30 60 90 120 150 --yaw_angles 0 90 180 270