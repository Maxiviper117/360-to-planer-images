import os
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates
from concurrent.futures import ProcessPoolExecutor
import gc
import argparse

# Function to map Cartesian coordinates (x, y, z) to spherical coordinates (theta, phi) with yaw and pitch adjustments
def map_to_sphere(x, y, z, yaw_radian, pitch_radian):
    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))  # Calculate polar angle (theta)
    phi = np.arctan2(y, x)  # Calculate azimuthal angle (phi)

    # Apply pitch and yaw transformations
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))
    phi_prime = np.arctan2((np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                            np.cos(theta) * np.sin(pitch_radian)),
                           np.sin(theta) * np.cos(phi))

    phi_prime += yaw_radian  # Add yaw rotation
    phi_prime = phi_prime % (2 * np.pi)  # Ensure the angle stays within [0, 2π]

    return theta_prime, phi_prime  # Return the transformed spherical coordinates

# Function to perform color interpolation for the image mapping
def interpolate_color(coords, img, method='bilinear'):
    # Determine interpolation order based on method
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    
    # Interpolate the color channels separately
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    
    return np.stack((red, green, blue), axis=-1)  # Combine channels into a single image

# Function to convert a panorama image to a plane projection based on FOV, yaw, and pitch
def panorama_to_plane(panorama, FOV, output_size, yaw, pitch):
    pano_width, pano_height = panorama.size  # Get dimensions of the panorama
    pano_array = np.array(panorama)  # Convert image to numpy array
    yaw_radian = np.radians(yaw)  # Convert yaw to radians
    pitch_radian = np.radians(pitch)  # Convert pitch to radians

    W, H = output_size  # Output image size (width, height)
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)  # Focal length based on FOV

    # Create meshgrid for image coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2  # Adjust x-coordinates to image center
    y = H / 2 - v  # Adjust y-coordinates to image center
    z = f  # Set z-coordinates to focal length

    # Map Cartesian coordinates to spherical coordinates
    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)

    # Convert spherical coordinates back to 2D image coordinates
    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    # Flatten and stack the coordinates for interpolation
    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    # Interpolate colors from the panorama image
    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')

    return output_image  # Return the output image as a plane projection

# Function to validate pitch value (must be between 1 and 179 degrees)
def check_pitch(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 179:
        raise argparse.ArgumentTypeError(f"{value} is an invalid pitch value. It must be between 1 and 179.")
    return ivalue

# Function to validate yaw values (each must be between 0 and 360 degrees)
def check_yaw(values: list[int]):
    for val in values:
        if val < 0 or val > 360:
            raise argparse.ArgumentTypeError(f"{val} is an invalid yaw value. It must be between 0 and 360.")

# Function to process an individual image by converting the panorama to a plane projection
def process_image(image_path, yaw, args, output_path):
    print(f"Processing {image_path} with yaw {yaw}...")
    try:
        # Open the image and convert to RGB
        panorama = Image.open(image_path).convert('RGB')
        file_name = image_path.stem  # Extract file name without extension
        
        # Determine output format (default to input image's format if not specified)
        if args.output_format is None:
            output_format = image_path.suffix[1:]
        else:
            output_format = args.output_format
        
        # Construct the output image file name
        output_image_name = f"{file_name}_pitch{args.pitch}_yaw{yaw}_fov{args.FOV}.{output_format}"
        output_image_path = output_path / output_image_name
        
        # Convert the panorama to a plane projection
        output_image = panorama_to_plane(panorama, args.FOV, (args.output_width, args.output_height), yaw, 180 - args.pitch)
        
        # Save the output image
        output_image.save(output_image_path)
        
        print(f"Saved output image to {output_image_path}")
    except Exception as e:
        print(f"Failed to process {image_path} with yaw {yaw}: {e}")
    
    gc.collect()  # Manually trigger garbage collection after processing an image to free up memory

# Main script logic for command-line execution
if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process some panorama images.')

    # Add command-line arguments
    parser.add_argument('--input_path', type=str, help='Path to the input panorama images', required=True)
    parser.add_argument('--output_path', type=str, default='output_images', help='Path to the output images')
    parser.add_argument('--output_format', type=str, choices=['png', 'jpg', 'jpeg'], help='Output image format - "png", "jpg", "jpeg"')
    parser.add_argument('--FOV', type=int, default=90, help='Field of View')
    parser.add_argument('--output_width', type=int, default=1000, help='Width of the output image')
    parser.add_argument('--output_height', type=int, default=1500, help='Height of the output image')
    parser.add_argument('--pitch', type=int, default=90, help='Pitch angle (vertical). Must be between 1 and 179.')
    parser.add_argument("--list-of-yaw", nargs="+", type=int, default=[0, 60, 120, 180, 240, 300], help="List of yaw angles (horizontal) e.g. --list-of-yaw 0 60 120 180 240 300")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Validate pitch and yaw values
    check_pitch(args.pitch)
    check_yaw(args.list_of_yaw)

    # Define input and output paths
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Create output directory if it doesn't exist
    if not output_path.exists():
        output_path.mkdir()

    # Use ProcessPoolExecutor for parallel processing of images
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Iterate over supported image file extensions
        for image_extension in ['*.jpg', '*.jpeg', '*.png']:
            # Process each image in the input directory
            for image_path in input_path.glob(image_extension):
                # Submit processing tasks for each yaw angle
                for yaw in args.list_of_yaw:
                    executor.submit(process_image, image_path, yaw, args, output_path)
