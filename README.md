## Unwrapping the View: Transforming 360 Panoramas into Planar Images with Python

This repository contains Python code for transforming a 360-degree panoramic image into a planar images, allowing extraction of specific views from the panorama. The script generates multiple planar images with specified yaw angles, field of view, and pitch angle, providing a way to visualize different perspectives from a single panorama image.

**This implementation is based on the Medium blog post by Coding Ballad:** [https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94](https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94)


## How to Use the Code

First clone the repository and navigate to the project directory, in the terminal:

```bash
git clone https://github.com/Maxiviper117/360-to-planer-images.git
cd 360-to-planer-images
```

This repository contains a Python script that converts a panorama image into multiple plane projections with specified yaw angles. The script allows for customization of field of view, output image size, and pitch angle.

## Prerequisites

Ensure you have the following libraries installed:

- NumPy
- Pillow
- SciPy
- tqdm

You can install the necessary packages using pip:

```bash
pip install -r requirements.txt
```

## Script Usage

The script `app/pano_to_planar.py` converts panorama images to plane projections based on specified parameters. You can run the script from the command line with different arguments.

### Command Line Arguments when running the script


Here is the content converted into a numbered markdown list for your GitHub repo readme:

1. `--input_path`
    - **Default:** 
    - **Description:** Specifies the directory where the input panorama images are stored. If not provided, the script defaults to using a directory named `input_images` in the current working directory. This directory should contain the panorama images you intend to process.
    
2. `--output_path`
    - **Default:** output_images in the current working directory
    - **Description:** Defines the directory where the processed (output) images will be saved. If this argument is not specified, the script uses a default directory named `output_images` in the current working directory.
    
3. `--FOV`
    - **Default:** 90 degrees
    - **Description:** Sets the field of view for the output images in degrees. Determines how wide the view should be in the resulting images. A larger FOV captures a wider area but may introduce more distortion.
    
4. `--output_width`
    - **Default:** 1000 pixels
    - **Description:** Specifies the width of the output images in pixels. This width will be applied to all output images generated by the script.
    
5. `--output_height`
    - **Default:** 1500 pixels
    - **Description:** Sets the height of the output images in pixels. This height will be used for all images processed by the script.
    
6. `--pitch`
    - **Default:** 90 degrees
    - **Description:** Defines the rotation around the horizontal axis (up/down) for the output images in degrees. The default pitch angle is 90 degrees. Adjusting the pitch can simulate looking up or down in the panorama.
    
7. `--list-of-yaw`
    - **Default:** 0 60 120 180 240 300
    - **Description:** Takes a list of yaw angles (rotation around the vertical axis - left/right) in degrees. Each yaw angle in the list will generate a separate output image, simulating rotation around the panorama.


These arguments enable users to customize how panorama images are processed, including the dimensions of the output images, the field of view, and the angles of view. By adjusting these parameters, users can generate specific views from panorama images, useful for applications in virtual tours, simulations, or any project requiring detailed views from panoramic photography.

## Note: Only one pitch angle can be selected at a time, but for each pitch angle, multiple yaw angles can be extracted at once based on that pitch angle.



### Running the Script

In the terminal, navigate to the directory containing the `pano_to_planar.py` script.

```bash
cd app
```

To run the script, use the following command in your terminal:

```bash
python pano_to_planar.py --input_path <input_directory> --output_path <output_directory> --FOV <FOV_value> --output_width <width> --output_height <height> --pitch <pitch_angle> --list-of-yaw <yaw_angles>
```

Replace `<input_directory>`, `<output_directory>`, `<FOV_value>`, `<width>`, `<height>`, `<pitch_angle>`, and `<yaw_angles>` with your desired values.

### Example

For example, to convert images from the `input_images` directory with a field of view of 120 degrees, output size of 1000x1500 pixels, pitch angle of 90 degrees, and yaw angles of 0, 60, 120, 180, 240, 300, you can run the following command:

> NOTE: The `--list-of-yaw` argument should be followed by a list of yaw angles separated by spaces.
>   - E.g. `--list-of-yaw 0 60 120 180 240 300`
>   - And ensure that the yaw angles are NOT too far apart, we want to have overlapping images.

```bash
python pano_to_planar.py --input_path input_images --output_path output_images --FOV 120 --output_width 1000 --output_height 1500 --pitch 90 --list-of-yaw 0 45 90 135 180 225 270 315
```

### Directory Structure

Your directory structure looks like this:

```
project_directory/
│
├── app
|   ├── pano_to_planar.py
|   ├── single_image.py
│   ├── output_images/ (output of pano_to_planar.py, will be created by the script)
│   ├── output_image/ (output of single_image.py, will be created by the script)
```

### Output

The output images will be saved in the specified `output_path` directory with filenames indicating the `pitch angle`, `yaw angle` and `FOV` e.g., `image1_pitch90_yaw0_fov90.jpg`,  `image1_pitch90_yaw30_fov90.jpg`, etc.

## Single image testing for finetuning parameters

If you want to play around with the - dimensions, FOV, pitch, yaw, etc, on a single image, you can use the `single_image.py` script. This script allows you to test the transformation on a single image.

Instead of an `input_path` for a directory, you will provide the path using `--input_image` for a single image, and the script will output the transformed images based on the specified parameters.

Ensure you have navigated to the `app` directory in the terminal if you are not already there.

```bash
cd app
```

To run the script, use the following command in your terminal:

```bash
python single_image.py --input_image <input_image_path> --output_path <output_directory> --FOV <FOV_value> --output_width <width> --output_height <height> --pitch <pitch_angle> --list-of-yaw <yaw_angles>
```
Example for testing how different FOV values affect the output image you would commands one after the other like so:

```bash
python single_image.py --input_image <input_image_path> --FOV 90

python single_image.py --input_image <input_image_path> --FOV 100

python single_image.py --input_image <input_image_path> --FOV 110

python single_image.py --input_image <input_image_path> --FOV 120
```

Add more `--<parameter>` arguments to test different parameters.

---

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
    <img src="assets/pitch.png" alt="alt text" width="400"/>
    <img src="assets/yaw.png" alt="alt text" width="400"/>
</div>