# Object Detection System

A professional Object Detection System developed for the CodeAlpha Artificial Intelligence Internship using Python, YOLOv8, OpenCV, and Tkinter.

## Features

- Object detection using YOLOv8
- Modern dark-themed GUI
- Image upload support
- Confidence threshold adjustment
- Background model loading
- Multi-threaded detection
- Automatic object counting
- Timestamped output files
- Automatic output folder creation
- Detection result preview
- Save detected images automatically

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Pillow
- Tkinter

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python object_detector.py
```

## How It Works

1. Launch the application.
2. Select an image.
3. Adjust confidence threshold if needed.
4. Click **Detect Objects**.
5. YOLOv8 detects objects and draws bounding boxes.
6. Results are saved automatically in the `detections` folder.

## Project Structure

```text
Task4_Object_Detection/
│
├── object_detector.py
├── requirements.txt
├── README.md
└── detections/
```

## Sample Detected Objects

- Person
- Car
- Bicycle
- Dog
- Cat
- Bottle
- Chair
- Mobile Phone
- Laptop
- Bus
- Truck

## Author

Developed for the CodeAlpha Artificial Intelligence Internship.