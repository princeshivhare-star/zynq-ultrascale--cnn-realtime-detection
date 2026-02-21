YOLOv5 ONNX USB Webcam Object Detection on ZCU104

This project demonstrates real-time object detection using YOLOv5 ONNX models on a Xilinx ZCU104 (UltraScale+ MPSoC) board. The system captures video from a USB webcam, performs inference using ONNX models, and streams the live detection results to a web browser, allowing headless operation without a monitor attached to the board.

Project Overview

The goal of this project is to run a lightweight YOLOv5 object detection pipeline on an FPGA-based board, using ONNX for inference, while enabling:

Real-time object detection (person, bicycle, car, and all COCO classes)

Headless operation with live streaming over network

Saving detection frames automatically (output1.jpg, output2.jpg, …)

Easy access from any laptop on the same network

project/
│
├── models/                  # ONNX model files
│   ├── yolov5s.onnx
│   
│
├── app/                     # Python scripts for inference
│   ├── webcam_stream_onnx_flask.py  # Live browser streaming script
│   ├── coco_classes.py               # COCO class labels
│   
│
├── requirements.txt         # Python dependencies
├── README.md                # This project description
└── .gitignore               # Files to ignore for Git


Hardware Requirements

Board: Xilinx ZCU104 (UltraScale+ MPSoC)

Webcam: USB webcam (V4L2 compatible)

Network: Board connected to same network as laptop (via Ethernet or USB Ethernet sharing)

Laptop: Any device with web browser to view live stream

Software Requirements

Python 3.8+

ONNX Runtime (onnxruntime)

OpenCV (opencv-python)

Flask (flask)

Numpy (numpy)

All dependencies are included in requirements.txt.

Setup Instructions
1️⃣ Connect and prepare the board

Plug in the USB webcam.

Connect the board to your laptop via Ethernet.

Ensure internet access (for installing Python packages).

Check webcam device:

ls /dev/video*
# Typically: /dev/video0
2️⃣ Copy ONNX models to the board

Place your .onnx models inside the /root/models directory on the board:

cp /path/to/yolov5s.onnx /root/models/
cp /path/to/yolov5n.onnx /root/models/

Ensure model names match those used in the Python scripts.

3️⃣ Set up Python environment

Create virtual environment:

python3 -m venv yolov5-env
source yolov5-env/bin/activate

Install dependencies:

pip install -r requirements.txt
4️⃣ Running the headless detection

The webcam_stream_onnx_flask.py script allows live streaming detection through a web browser.

Run the script:

cd /root/app
python3 webcam_stream_onnx_flask.py

You will see output like:

 * Running on http://0.0.0.0:5000
 * Running on http://192.168.137.9:5000
5️⃣ Access live detection from your laptop

Open a browser on your laptop and go to the board's IP (from ip addr show eth0):

http://192.168.137.9:5000

You will see the live webcam feed with bounding boxes and class labels.

6️⃣ Saving detection outputs

The script automatically saves frames with detections in the output/ folder:

output/output1.jpg, output/output2.jpg, ...

This allows you to keep snapshots of detection results.

7️⃣ Exiting the stream

In the Flask server terminal: Press CTRL+C

In headless mode (if using cv2 capture), press q to stop the webcam.

Performance Tips

Ensure webcam resolution matches the ONNX model input (e.g., 640x640).

For smoother streaming, you can skip frames in the processing loop.

Headless mode with Flask streaming reduces CPU/GPU load.

Project Workflow

Connect webcam → check /dev/video0.

Copy ONNX model to /root/models.

Activate virtual environment and install dependencies.

Run webcam_stream_onnx_flask.py.

Access live detection from any laptop via browser.

Stop stream with CTRL+C (Flask) or q (headless).

Saved outputs in output/ folder for later use.

GitHub Notes

.gitignore should include:

__pycache__/
*.pyc
*.jpg
*.jpeg
*.png
yolov5-env/

requirements.txt includes:

onnxruntime
opencv-python
flask
numpy
