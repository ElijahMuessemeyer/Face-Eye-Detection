# Face and Eye Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

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
