import os
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import argparse
from typing import List, Tuple
from tqdm import tqdm
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Adjusted to INFO to reduce verbosity; change to WARNING if needed
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_rotation_matrix(yaw_radian: float, pitch_radian: float) -> np.ndarray:
    """
    Compute the combined rotation matrix for given yaw and pitch angles.

    Args:
        yaw_radian (float): Yaw angle in radians.
        pitch_radian (float): Pitch angle in radians.

    Returns:
        np.ndarray: Combined rotation matrix.
    """
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

    # return combined rotation matrix that will be used to calculate the new x, y, z coordinates after rotation
    return np.dot(R_pitch, R_yaw)

@lru_cache(maxsize=None)
def precompute_mapping(W: int, H: int, FOV_rad: float, yaw_radian: float, pitch_radian: float, pano_width: int, pano_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute the mapping coordinates for a given set of parameters.

    This function processes every pixel in the output image by performing the following steps:
        1. **Calculating Camera Space Coordinates**:
            - Converts pixel coordinates `(u, v)` to camera space `(x, y, z)` by centering them and applying the focal length based on the Field of View.
        2. **Normalizing Vectors to Unit Length**:
            - Ensures that each vector `(x, y, z)` has a unit length to maintain consistent scaling during rotation.
        3. **Applying Rotation Transformations**:
            - Applies rotation matrices based on the provided yaw and pitch angles to rotate each individual pixel's vector in 3D space.
            - This rotation adjusts the perspective of the panorama to align with the desired viewpoint.
        4. **Converting Rotated Vectors to Spherical Coordinates**:
            - Transforms the rotated Cartesian coordinates `(x_rot, y_rot, z_rot)` into spherical coordinates `(theta_prime, phi_prime)` for easier mapping to the panorama.
        5. **Mapping Spherical Coordinates to Panorama Image Coordinates**:
            - Translates the spherical coordinates to panorama image coordinates `(U, V)` by scaling them according to the panorama's width and height.

    **Detailed Workflow**:
        - **Meshgrid Creation**:
            - Generates a grid of `(u, v)` coordinates covering the entire output image.
        - **Vector Calculation**:
            - Centers the `(u, v)` coordinates and calculates the corresponding `z` value using the focal length.
        - **Normalization**:
            - Computes the norm of each vector and normalizes the `x`, `y`, and `z` components.
        - **Rotation**:
            - Constructs the combined rotation matrix using yaw and pitch angles.
            - Applies the rotation matrix to each normalized vector to obtain the rotated coordinates.
        - **Spherical Conversion**:
            - Calculates the polar and azimuthal angles from the rotated vectors.
        - **Final Mapping**:
            - Maps these angles to the panorama image's pixel coordinates, ensuring that each output pixel correctly corresponds to its position in the panorama.

    **Performance Considerations**:
        - The function utilizes vectorized operations with NumPy for efficient computation over all pixels.
        - The `@lru_cache` decorator caches the results, preventing redundant calculations for identical input parameters and enhancing performance during repeated calls.


    Returns:
        Tuple[np.ndarray, np.ndarray]: Mapping coordinates (U, V).
    """
    # Calculate the focal length based on the Field of View and output image width
    focal_length = (0.5 * W) / np.tan(FOV_rad / 2)

    # Create a meshgrid for the output image coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    # Convert u and v to float32 for subsequent arithmetic operations
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Calculate the 3D x, y, and z coordinates in the camera space
    # - Each pixel (u, v) in the equirectangular image is first converted to 3D camera space coordinates (x, y, z).
    # - This involves centering the pixel coordinates (u, v) and calculating the z component using the focal length derived from the Field of View (FOV).
    x = u - (W / 2.0)  # Center the x coordinates around zero
    y = (H / 2.0) - v  # Center the y coordinates around zero and flip vertically
    z = np.full_like(x, focal_length, dtype=np.float32)  # Set z to the focal length

    # Normalize the vectors to unit length
    # - Normalizing a vector scales it to have a unit length (length of 1) while maintaining its direction.
    norm = np.sqrt(x**2 + y**2 + z**2)  # Calculate the norm (magnitude) of each vector
    x_norm = x / norm  # Normalize x coordinates
    y_norm = y / norm  # Normalize y coordinates
    z_norm = z / norm  # Normalize z coordinates

    # Apply the rotation matrix to the normalized vectors
    # - Yaw and pitch angles are applied using rotation matrices to adjust the viewpoint.
    # - These rotations operate in the 3D space, modifying the (x, y, z) vectors accordingly.
    R = get_rotation_matrix(yaw_radian, pitch_radian)  # Get the combined rotation matrix, which only deals with x(pitch) and y(yaw) rotation
    vectors = np.stack((x_norm, y_norm, z_norm), axis=0)  # Stack the normalized vectors into a single array
    rotated = R @ vectors.reshape(3, -1)  # Apply the rotation matrix to the vectors
    rotated = rotated.reshape(3, H, W)  # Reshape the rotated vectors back to the original dimensions
    x_rot, y_rot, z_rot = rotated  # Extract the rotated x, y, and z coordinates

    # Convert the rotated vectors back to spherical coordinates
    # - After rotation, the 3D vectors (x_rot, y_rot, z_rot) are converted back to spherical coordinates (theta_prime, phi_prime).
    # - These spherical coordinates are essential for mapping back to the 2D panorama image.
    theta_prime = np.arccos(z_rot).astype(np.float32)  # Calculate the polar angle (theta) from the z coordinate
    
    """
    Calculate the azimuthal angle (phi') in radians from the rotated x and y coordinates.

    This line performs the following steps:
    1. Uses `np.arctan2(y_rot, x_rot)` to compute the azimuthal angle, taking into account the quadrant.
    - The output can be outside the standard range [0, 2π) due to rotations.
    2. Applies the modulo operation (% (2 * np.pi)) to wrap the angle back into the range [0, 2π).
    - This ensures that rotations beyond the bounds of the panorama (e.g., > 360° or < 0°)
        are correctly mapped to valid coordinates in the equirectangular projection.
    3. Converts the resulting wrapped angle to `np.float32` for compatibility with further calculations.

    This ensures seamless handling of horizontal rotations in the panorama.
    """
    phi_prime = (np.arctan2(y_rot, x_rot) % (2 * np.pi)).astype(np.float32)  
    
    """
    After rotation, the 3D vectors (x_rot, y_rot, z_rot) are converted back to spherical coordinates (theta_prime, phi_prime).
    
    Then we can now map these spherical coordinates to the panorama image's pixel coordinates (U, V).
    
    Thus pixel by
    
    """
    # Map the spherical coordinates to the panorama image coordinates
    U = (phi_prime * pano_width) / (2 * np.pi)  # Scale phi to the panorama width
    V = (theta_prime * pano_height) / np.pi  # Scale theta to the panorama height

    # Ensure the coordinates are within the image boundaries
    U = np.clip(U, 0, pano_width - 1).astype(np.float32)  # Clip U to be within valid range
    V = np.clip(V, 0, pano_height - 1).astype(np.float32)  # Clip V to be within valid range

    return U, V

def interpolate_color(U: np.ndarray, V: np.ndarray, img: np.ndarray, method: str = 'bilinear') -> np.ndarray:
    """
    Perform color interpolation for the image mapping using OpenCV's remap.

    Args:
        U (np.ndarray): Horizontal mapping coordinates.
        V (np.ndarray): Vertical mapping coordinates.
        img (np.ndarray): Input image as a NumPy array.
        method (str, optional): Interpolation method. Defaults to 'bilinear'.

    Returns:
        np.ndarray: Interpolated color image.
    """
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC
    }
    interp = interpolation_methods.get(method, cv2.INTER_LINEAR)

    remapped = cv2.remap(img, U, V, interpolation=interp, borderMode=cv2.BORDER_REFLECT)
    return remapped

def panorama_to_plane(pano_array: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Convert a panorama image to a plane projection based on precomputed mapping coordinates.

    Args:
        pano_array (np.ndarray): Input panorama image as a NumPy array.
        U (np.ndarray): Precomputed horizontal mapping coordinates.
        V (np.ndarray): Precomputed vertical mapping coordinates.

    Returns:
        np.ndarray: The resulting plane-projected image.
    """
    return interpolate_color(U, V, pano_array)

def check_pitch(value: str) -> int:
    """
    Validate the pitch value.

    Args:
        value (str): Pitch value as a string.

    Returns:
        int: Validated pitch value.

    Raises:
        argparse.ArgumentTypeError: If pitch is not within 1-179 degrees.
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Pitch value must be an integer between 1 and 179.")

    if not (1 <= ivalue <= 179):
        raise argparse.ArgumentTypeError(f"{ivalue} is an invalid pitch value. It must be between 1 and 179.")
    return ivalue

def check_yaw(yaw_angles: List[int]) -> List[int]:
    """
    Validate yaw values.

    Args:
        yaw_angles (List[int]): List of yaw angles.

    Returns:
        List[int]: Sorted and unique yaw angles.

    Raises:
        argparse.ArgumentTypeError: If any yaw is not within 0-360 degrees.
    """
    unique_yaws = set()
    for val in yaw_angles:
        if not (0 <= val <= 360):
            raise argparse.ArgumentTypeError(f"{val} is an invalid yaw value. It must be between 0 and 360.")
        unique_yaws.add(val)
    return sorted(unique_yaws)

def process_image_batch(image_path: Path, args: argparse.Namespace, output_path: Path, precomputed_mappings: dict):
    """
    Process an individual panorama image by converting it to plane projections for all yaw angles.

    Args:
        image_path (Path): Path to the input image.
        args (argparse.Namespace): Parsed command-line arguments.
        output_path (Path): Directory to save the output images.
        precomputed_mappings (dict): Dictionary of precomputed (U, V) mappings per yaw angle.
    """
    logging.info(f"Processing {image_path}...")
    try:
        # Read the panorama image using OpenCV
        pano_array = cv2.imread(str(image_path))
        if pano_array is None:
            logging.error(f"Failed to read image {image_path}. Skipping.")
            return
        pano_array = cv2.cvtColor(pano_array, cv2.COLOR_BGR2RGB)
        pano_height, pano_width, _ = pano_array.shape
        file_name = image_path.stem

        for yaw in args.yaw_angles:
            logging.debug(f"Processing {image_path} with yaw {yaw}°...")
            U, V = precomputed_mappings[yaw]

            # Perform the panorama to plane projection
            output_image_array = panorama_to_plane(pano_array, U, V)

            # Determine output format
            output_format = args.output_format if args.output_format else image_path.suffix[1:]

            # Construct output image name
            output_image_name = f"{file_name}_pitch{args.pitch}_yaw{yaw}_fov{args.FOV}.{output_format}"
            output_image_path = output_path / output_image_name

            # Convert back to BGR for saving with OpenCV
            output_bgr = cv2.cvtColor(output_image_array, cv2.COLOR_RGB2BGR)

            # Save the output image
            cv2.imwrite(str(output_image_path), output_bgr)
            logging.info(f"Saved output image to {output_image_path}")

    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Convert panorama images to plane projections based on FOV, yaw, and pitch.')

    parser.add_argument('--input_path', type=str, help='Path to the input panorama images', required=True)
    parser.add_argument('--output_path', type=str, default='output_images', help='Path to save the output images')
    parser.add_argument('--output_format', type=str, choices=['png', 'jpg', 'jpeg'], help='Output image format (png, jpg, jpeg)')
    parser.add_argument('--FOV', type=int, default=90, help='Field of View in degrees')
    parser.add_argument('--output_width', type=int, default=1000, help='Width of the output image in pixels')
    parser.add_argument('--output_height', type=int, default=1500, help='Height of the output image in pixels')
    parser.add_argument('--pitch', type=check_pitch, default=90, help='Pitch angle in degrees (1-179)')
    parser.add_argument('--yaw_angles', nargs='+', type=int, default=[0, 60, 120, 180, 240, 300], help='List of yaw angles in degrees (0-360). Example: --yaw_angles 0 60 120 180 240 300')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker threads. Defaults to 90% of CPU cores if not specified.')

    args = parser.parse_args()

    # Validate yaw angles
    args.yaw_angles = check_yaw(args.yaw_angles)

    return args

def main():
    """
    Main function to orchestrate the processing of panorama images.
    """
    args = parse_arguments()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.is_dir():
        logging.error(f"Input path {input_path} is not a directory or does not exist.")
        return

    if output_path.exists():
        logging.info(f"Output directory {output_path} already exists.")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output directory {output_path}.")

    # Collect all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_path.glob(ext))

    if not image_paths:
        logging.warning(f"No images found in {input_path} with extensions {image_extensions}.")
        return

    # Determine number of workers
    if args.num_workers is not None:
        max_workers = args.num_workers
    else:
        max_workers = max(1, int(os.cpu_count() * 0.9))

    logging.info(f"Using {max_workers} worker(s) for processing.")

    # Precompute mappings for each yaw angle
    precomputed_mappings = {}
    FOV_rad = np.radians(args.FOV)
    pitch_rad = np.radians(args.pitch)

    # Read a sample image to get panorama dimensions
    sample_image_path = image_paths[0]
    sample_pano = cv2.imread(str(sample_image_path))
    if sample_pano is None:
        logging.error(f"Failed to read sample image {sample_image_path} for precomputing mappings.")
        return
    pano_height, pano_width, _ = sample_pano.shape

    for yaw in args.yaw_angles:
        yaw_rad = np.radians(yaw)
        U, V = precompute_mapping(
            W=args.output_width,
            H=args.output_height,
            FOV_rad=FOV_rad,
            yaw_radian=yaw_rad,
            pitch_radian=pitch_rad,
            pano_width=pano_width,
            pano_height=pano_height
        )
        precomputed_mappings[yaw] = (U, V)

    # Prepare tasks
    tasks = [(image_path, args, output_path, precomputed_mappings) for image_path in image_paths]

    logging.info(f"Starting processing of {len(image_paths)} images with {len(args.yaw_angles)} yaw angles each.")

    # Process images in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image_batch, *task)
            for task in tasks
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images", unit="image"):
            pass  # Progress is handled by tqdm

    logging.info("Processing completed.")

if __name__ == '__main__':
    main()

# Example usage:
# python panorama_to_plane.py --input_path ./test-images --FOV 90 --output_width 1920 --output_height 1080 --pitch 90 --yaw_angles 0 90 180 270 --num_workers 4
