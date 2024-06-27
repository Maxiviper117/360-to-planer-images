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


1. `--input_path`: (Required)
    - This argument specifies the directory where the input panorama images are stored. If not provided, the script defaults to using a directory named `input_images` in the current working directory. This directory should contain the panorama images you intend to process.

2. `--output_path`: (Optional - Default: output_images in the current working directory)
    - This argument defines the directory where the processed (output) images will be saved. If this argument is not specified, the script uses a default directory named `output_images` in the current working directory.
    - If the specified directory does not exist, the script may create it depending on its implementation.

3. `--FOV` (Field of View): (Default: 90 degrees)
    - This parameter sets the field of view for the output images in degrees. The default value is 90 degrees. It determines how wide the view should be in the resulting images. A larger FOV captures a wider area but may introduce more distortion.

4. `--output_width`: (Default: 1000 pixels)
    - This specifies the width of the output images in pixels. The default width is set to 1000 pixels. This width will be applied to all output images generated by the script.

5. `--output_height`: (Default: 1500 pixels)
    - Similar to `--output_width`, this argument sets the height of the output images in pixels, with a default value of 1500 pixels. This height will be used for all images processed by the script.


6. `--pitch`: (Default: 90 degrees)
    - The pitch angle allows the user to define the rotation around the horizontal axis (up/down) for the output images in degrees. The default pitch angle is 90 degrees. Adjusting the pitch can simulate looking up or down in the panorama.
    - The default pitch angle is set to 90 degrees, which corresponds to looking straight ahead in the panorama.
    - Increasing the pitch angle will simulate looking up, while decreasing it will simulate looking down.

7. `--list-of-yaw`: (Default: 0 60 120 180 240 300)
    - This argument takes a list of yaw angles (rotation around the vertical axis - left/right) in degrees. Each yaw angle in the list will generate a separate output image, simulating rotation around the panorama. This allows for multiple views from a single panorama image.
    - The default list is `--list-of-yaw 0 60 120 180 240 300` (hint: not using 360deg because it is the same as 0deg)
    - The images will be output in a clockwise direction starting from the first yaw angle.

These arguments enable users to customize how panorama images are processed, including the dimensions of the output images, the field of view, and the angles of view. By adjusting these parameters, users can generate specific views from panorama images, useful for applications in virtual tours, simulations, or any project requiring detailed views from panoramic photography.


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

Ensure your directory structure looks like this:

```
project_directory/
│
├── pano_to_planar.py
└── output_images/
```

### Output

The output images will be saved in the specified `output_path` directory with filenames indicating the yaw angle, e.g., `image1_yaw_0.jpg`, `image1_yaw_60.jpg`, etc.

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

![alt text](assets/pitch.png)
![alt text](assets/yaw.png)