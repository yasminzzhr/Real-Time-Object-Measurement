# Real-Time Object Measurement

This program detects contours of objects in real-time using OpenCV and calculates their real width, height, and area based on a known scale. It also displays the measurement information on the detected objects.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

Install OpenCV and NumPy using pip:

pip install opencv-python
pip install numpy

## How to Use

1. Clone this repository or download the script file (`object_measurement.py`).
2. Open a terminal and navigate to the directory containing the script.
3. Run the script using the following command:

    python object_measurement.py

4. Adjust the trackbars to set the appropriate thresholds for contour detection and measurement parameters.
5. Hold an object in front of the camera, and the program will display its width, height, and area in real-time.

## Trackbars

- **Threshold1**: Lower threshold value for Canny edge detection.
- **Threshold2**: Upper threshold value for Canny edge detection.
- **Area**: Minimum area of the object to filter out small contours.
- **Distance**: Distance in meters between the object and the camera.
- **Scale**: Number of pixels in one centimeter. For example, if 1cm equals 6 pixels, set the scale to 6.
