import cv2
import numpy as np
import onnxruntime
from flask import Flask, Response
import os

# ---- CONFIG ----
MODEL_PATH = "/root/models/yolov5s.onnx"  # Your ONNX model
CONF_THRESHOLD = 0.3
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
OUTPUT_DIR = "/root/app"  # Save frames

classes = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

# ---- INIT ----
session = onnxruntime.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

frame_count = 0
app = Flask(__name__)

# ---- FUNCTIONS ----
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs, frame_shape):
    boxes, scores, class_ids = [], [], []
    dets = outputs[0]  # (1, num_detections, 6+)
    dets = dets[0]     # remove batch dim
    for det in dets:
        if len(det) < 6: continue
        conf = float(det[4])
        if conf < CONF_THRESHOLD: continue
        x1, y1, x2, y2 = map(int, det[:4])
        cls_id = int(det[5])
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(cls_id)
    return boxes, scores, class_ids

def detect(frame):
    img = preprocess(frame)
    outputs = session.run([output_name], {input_name: img})
    boxes, scores, class_ids = postprocess(outputs, frame.shape)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = f"{classes[class_ids[i]]} {scores[i]:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def gen_frames():
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = detect(frame)
        frame_count += 1
        filename = os.path.join(OUTPUT_DIR, f"output{frame_count}.jpg")
        cv2.imwrite(filename, frame)
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---- FLASK ROUTES ----
@app.route('/')
def index():
    return "<h1>Live YOLOv5 ONNX Stream</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- MAIN ----
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        cap.release()
