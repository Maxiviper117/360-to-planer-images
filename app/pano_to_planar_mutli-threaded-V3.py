import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import argparse
from typing import List, Tuple
from tqdm import tqdm

# openai - o1-preview-mini optimized

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def map_to_sphere(x: np.ndarray, y: np.ndarray, z: float, yaw_radian: float, pitch_radian: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map Cartesian coordinates to spherical coordinates with yaw and pitch adjustments.

    Args:
        x (np.ndarray): X-coordinates.
        y (np.ndarray): Y-coordinates.
        z (float): Z-coordinate (focal length).
        yaw_radian (float): Yaw angle in radians.
        pitch_radian (float): Pitch angle in radians.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed (theta, phi) spherical coordinates.
    """
    # Normalize vectors
    norm = np.sqrt(x**2 + y**2 + z**2)
    x_norm, y_norm, z_norm = x / norm, y / norm, z / norm

    # Rotation matrices
    R_yaw = np.array([
        [np.cos(yaw_radian), 0, np.sin(yaw_radian)],
        [0, 1, 0],
        [-np.sin(yaw_radian), 0, np.cos(yaw_radian)]
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_radian), -np.sin(pitch_radian)],
        [0, np.sin(pitch_radian), np.cos(pitch_radian)]
    ])
    R = R_pitch @ R_yaw

    # Apply rotation
    vectors = np.vstack((x_norm.flatten(), y_norm.flatten(), z_norm.flatten()))
    rotated = R @ vectors
    x_rot, y_rot, z_rot = rotated

    # Convert back to spherical coordinates
    theta_prime = np.arccos(z_rot).reshape(x.shape)
    phi_prime = np.arctan2(y_rot, x_rot).reshape(x.shape) % (2 * np.pi)

    return theta_prime, phi_prime

def interpolate_color(coords: Tuple[np.ndarray, np.ndarray], img: np.ndarray, method: str = 'bilinear') -> np.ndarray:
    """
    Perform color interpolation for the image mapping using OpenCV's remap.

    Args:
        coords (Tuple[np.ndarray, np.ndarray]): Tuple of (V, U) coordinates.
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

    V, U = coords
    map_x = U.astype(np.float32)
    map_y = V.astype(np.float32)

    remapped = cv2.remap(img, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT)
    return remapped

def panorama_to_plane(panorama: Image.Image, FOV: int, output_size: Tuple[int, int], yaw: float, pitch: float) -> Image.Image:
    """
    Convert a panorama image to a plane projection based on FOV, yaw, and pitch.

    Args:
        panorama (Image.Image): Input panorama image.
        FOV (int): Field of View in degrees.
        output_size (Tuple[int, int]): Output image size as (width, height).
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.

    Returns:
        Image.Image: The resulting plane-projected image.
    """
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    # Create meshgrid for image coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)

    U = (phi * pano_width) / (2 * np.pi)
    V = (theta * pano_height) / np.pi

    # Ensure coordinates are within image boundaries
    U = np.clip(U, 0, pano_width - 1)
    V = np.clip(V, 0, pano_height - 1)

    colors = interpolate_color((V, U), pano_array)
    output_image = Image.fromarray(colors, 'RGB')

    return output_image

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

def process_image(image_path: Path, yaw: int, args: argparse.Namespace, output_path: Path):
    """
    Process an individual panorama image by converting it to a plane projection.

    Args:
        image_path (Path): Path to the input image.
        yaw (int): Yaw angle in degrees.
        args (argparse.Namespace): Parsed command-line arguments.
        output_path (Path): Directory to save the output image.
    """
    logging.info(f"Processing {image_path} with yaw {yaw}°...")
    try:
        with Image.open(image_path) as panorama:
            panorama = panorama.convert('RGB')
            file_name = image_path.stem

            # Determine output format
            output_format = args.output_format if args.output_format else image_path.suffix[1:]

            # Construct output image name
            output_image_name = f"{file_name}_pitch{args.pitch}_yaw{yaw}_fov{args.FOV}.{output_format}"
            output_image_path = output_path / output_image_name

            # Convert to plane projection
            output_image = panorama_to_plane(
                panorama,
                FOV=args.FOV,
                output_size=(args.output_width, args.output_height),
                yaw=yaw,
                pitch=args.pitch  # Assuming pitch is correctly handled in the transformation
            )

            # Save the output image
            output_image.save(output_image_path)
            logging.info(f"Saved output image to {output_image_path}")
    except Exception as e:
        logging.error(f"Failed to process {image_path} with yaw {yaw}°: {e}")

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

    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_path.glob(ext))

    if not image_paths:
        logging.warning(f"No images found in {input_path} with extensions {image_extensions}.")
        return

    # Prepare tasks
    tasks = []
    for image_path in image_paths:
        for yaw in args.yaw_angles:
            tasks.append((image_path, yaw, args, output_path))

    logging.info(f"Starting processing of {len(image_paths)} images with {len(args.yaw_angles)} yaw angles each.")

    # Process images in parallel with progress bar
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_image, *task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            pass  # Progress is handled by tqdm

    logging.info("Processing completed.")

if __name__ == '__main__':
    main()
