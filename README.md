# Hand Measurement Tool

This application uses computer vision to detect hands and measure the pixel distance between selected finger joints in real-time.

## Features

- **Real-time Hand Detection**: Uses MediaPipe to detect hand landmarks (up to two hands) in a webcam feed
- **Multiple Measurement Modes**: Measure distances between different parts of the hand:
  - **Mode 1**: Index-Middle Tip - Distance between index and middle fingertips
  - **Mode 2**: Thumb-Pinky Span - Distance between thumb and pinky fingertips
  - **Mode 3**: Hand Width (Knuckles) - Distance across the hand at the knuckle line
  - **Mode 4**: Index-to-Index (Two Hands) - Distance between index fingertips across two hands
- **Interactive UI**: Displays the current mode, measurement, and instructions on screen
- **Visual Feedback**: Draws hand landmarks, connections, and the measured distance line

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

All requirements are listed in the `requirements.txt` file.

## Installation

1. Clone the repository or download the source files
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:
   ```
   python hand_measurement.py
   ```

2. Position your hand(s) in view of the webcam, palm facing the camera.

3. Use the following keyboard controls:
   - **M key**: Cycle through measurement modes
   - **Q key**: Quit the application

## Understanding Pixel Measurements

The tool displays distances in pixels, which are relative measurements depending on:
- Distance of your hand from the camera
- Camera resolution
- Camera lens properties

To estimate real-world measurements (e.g., inches):
1. Hold a ruler or known-size object next to your hand
2. Note the pixel distance for a known length (e.g., 1 inch)
3. Calculate the conversion factor (pixels per inch)
4. Apply this conversion to future measurements

## Notes

- For best results, ensure good lighting and position your hand clearly in the camera view
- Keep your hand(s) at a consistent distance from the camera when making comparative measurements
- The hand detection works best when your palm is facing the camera and fingers are spread
- For two-hand measurements (Mode 4), make sure both hands are clearly visible in the frame

## Troubleshooting

- If no hand is detected, try adjusting your hand position or the lighting
- For two-hand measurement mode, ensure both hands are visible in the camera view
- If the application runs slowly, close other resource-intensive applications
- Make sure your webcam is properly connected and accessible