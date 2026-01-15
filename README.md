# Face and Eye Detection

Computer vision project using Haar Cascade classifiers for face and eye detection with OpenCV.

## Features

- **Auto-downloading Models**: Automatically downloads Haar cascade files from OpenCV repository
- **Face Detection**: Robust face detection with configurable parameters
- **Eye Detection**: Detects eyes within face regions with fallback synthesis
- **Visual Annotation**: Draws bounding boxes around detected features
- **Command-line Interface**: Process images via CLI arguments

## Usage

```bash
python face_detection.py --image path/to/image.jpg
```

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
