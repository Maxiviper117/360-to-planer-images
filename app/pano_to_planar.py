from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates

def map_to_sphere(x, y, z, yaw_radian, pitch_radian):
    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2((np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                            np.cos(theta) * np.sin(pitch_radian)),
                           np.sin(theta) * np.cos(phi))

    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime, phi_prime

def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)

def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')

    return output_image



import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add arguments
parser.add_argument('--input_path', type=str, default='input_images', help='Path to the input panorama image')
parser.add_argument('--output_path', type=str, default='output_images', help='Path to the output images')
parser.add_argument('--FOV', type=int, default=100, help='Field of View')
parser.add_argument('--output_width', type=int, default=1000, help='Width of the output image')
parser.add_argument('--output_height', type=int, default=1500, help='Height of the output image')
# parser.add_argument('--yaw', type=int, default=0, help='Yaw angle (rotation around the vertical axis - left/right)')
parser.add_argument('--pitch', type=int, default=90, help='Pitch angle (rotation around the horizontal axis - up/down)')
parser.add_argument("--list-of-yaw", nargs="+", type=int, default=[0, 60, 120, 180, 240, 300], help="List of yaw angles e.g. --list-of-yaw 0 60 120 180 240 300")

# Parse the arguments
args = parser.parse_args()

# Accessing argument values
FOV = args.FOV
output_size = (args.output_width, args.output_height)
# yaw = args.yaw
pitch = args.pitch

yawList = args.list_of_yaw



# Assuming panorama_to_plane is already defined/imported
# from your_module import panorama_to_plane

input_path = Path(__file__).parent / args.input_path
output_path = Path(__file__).parent / args.output_path

if not output_path.exists():
    output_path.mkdir()

# Loop over each image in input_path
for image_path in input_path.glob('*.jpg'):
    file_name = image_path.stem  # Extract file name without extension
    
    for yaw in yawList:
        output_image_name = f"{file_name}_yaw_{yaw}.jpg"
        output_image_path = output_path / output_image_name
        output_image = panorama_to_plane(str(image_path), FOV, output_size, yaw, pitch)
        output_image.save(output_image_path)
