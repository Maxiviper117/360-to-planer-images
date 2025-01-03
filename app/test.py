import os
from pathlib import Path
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def precompute_yaw_mapping(pano_width, pano_height, yaw_angle):
    """
    Precompute the mapping coordinates for horizontal (yaw) rotation.
    """
    yaw_radians = np.radians(yaw_angle)
    u, v = np.meshgrid(np.arange(pano_width), np.arange(pano_height), indexing='xy')
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    phi = (2 * np.pi * u / pano_width).astype(np.float32)
    phi_rotated = (phi + yaw_radians) % (2 * np.pi)

    U = (phi_rotated * pano_width) / (2 * np.pi)
    V = v  # Vertical coordinates remain unchanged

    U = np.clip(U, 0, pano_width - 1).astype(np.float32)
    V = V.astype(np.float32)

    return U, V

def precompute_pitch_mapping(W, H, FOV_rad, pitch_radian, pano_width, pano_height):
    """
    Precompute the mapping coordinates for pitch transformation.
    """
    focal_length = (0.5 * W) / np.tan(FOV_rad / 2)
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    u = u.astype(np.float32)
    v = v.astype(np.float32)

    x = u - (W / 2.0)
    y = (H / 2.0) - v
    z = np.full_like(x, focal_length, dtype=np.float32)

    norm = np.sqrt(x**2 + y**2 + z**2)
    x_norm = x / norm
    y_norm = y / norm
    z_norm = z / norm

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_radian), -np.sin(pitch_radian)],
        [0, np.sin(pitch_radian), np.cos(pitch_radian)]
    ], dtype=np.float32)

    vectors = np.stack((x_norm, y_norm, z_norm), axis=0).reshape(3, -1)
    rotated = R_pitch @ vectors
    x_rot, y_rot, z_rot = rotated.reshape(3, H, W)

    theta_prime = np.arccos(z_rot).astype(np.float32)
    phi_prime = (np.arctan2(y_rot, x_rot) % (2 * np.pi)).astype(np.float32)

    U = (phi_prime * pano_width) / (2 * np.pi)
    V = (theta_prime * pano_height) / np.pi

    U = np.clip(U, 0, pano_width - 1).astype(np.float32)
    V = np.clip(V, 0, pano_height - 1).astype(np.float32)

    return U, V

def process_yaw_and_pitch(pano_image, yaw_angle, pitch_angles, output_width, output_height):
    """
    Process a single yaw angle and multiple pitch angles, returning all slices.
    """
    pano_height, pano_width, _ = pano_image.shape

    # Precompute yaw mapping
    U_yaw, V_yaw = precompute_yaw_mapping(pano_width, pano_height, yaw_angle)
    rotated_pano = cv2.remap(pano_image, U_yaw, V_yaw, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    slices = []
    for pitch_angle in pitch_angles:
        pitch_radians = np.radians(pitch_angle)
        U_pitch, V_pitch = precompute_pitch_mapping(
            W=output_width, H=output_height,
            FOV_rad=np.radians(90),  # Example FOV
            pitch_radian=pitch_radians,
            pano_width=pano_width,
            pano_height=pano_height
        )
        slice_image = cv2.remap(rotated_pano, U_pitch, V_pitch, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        slices.append(slice_image)

    return slices

def main(input_path, output_path, yaw_angles, pitch_angles, output_width, output_height, num_workers=4):
    """
    Main function to process a panorama with specified yaw and pitch angles.
    """
    input_image = cv2.imread(input_path)
    if input_image is None:
        logging.error(f"Failed to read image: {input_path}")
        return

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for yaw_angle in yaw_angles:
            tasks.append(executor.submit(process_yaw_and_pitch, input_image, yaw_angle, pitch_angles, output_width, output_height))

        for future, yaw_angle in zip(tqdm(tasks, desc="Processing yaw angles"), yaw_angles):
            slices = future.result()
            for i, slice_image in enumerate(slices):
                output_file = output_dir / f"yaw_{yaw_angle}_pitch_{pitch_angles[i]}.png"
                slice_bgr = cv2.cvtColor(slice_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_file), slice_bgr)
                logging.info(f"Saved {output_file}")

if __name__ == "__main__":
    input_image_path = (
        Path(__file__).resolve().parent.parent / "test-images" / "hall1.jpg"
    )
    output_dir_path = "output_images"
    yaw_angles = [0, 90, 180, 270]
    pitch_angles = [30, 60, 90, 120, 150]
    output_width = 1000
    output_height = 500

    main(input_image_path, output_dir_path, yaw_angles, pitch_angles, output_width, output_height)
