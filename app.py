from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import torch
import time
import os
import numpy as np
from torchvision import transforms
from model.anomaly_classifier import AnomalyDetector
from utils.notifier import send_sms_alert  # ✅ Add this import

# CONFIG
IMG_SIZE = 128
SEQUENCE_LENGTH = 20
MODEL_PATH = "models/anomaly_model.pth"
LABELS = ['Abuse', 'Accident', 'Arrest', 'Assault', 'Burglary', 'Fighting', 'Normal', 'Robbery']
ALERT_CLASSES = [label for label in LABELS if label.lower() != 'normal']
COOLDOWN = 10  # seconds between alerts
CONFIDENCE_THRESHOLD = 0.75


# FLASK SETUP
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def load_model():
    model = AnomalyDetector(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    return model

model = load_model()

def detect_anomaly_from_frames(frames):
    tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    print(f"[DEBUG] Predicted: {LABELS[pred.item()]} | Confidence: {conf.item():.4f}")
    if conf.item() < CONFIDENCE_THRESHOLD:
        return 'Normal'
    return LABELS[pred.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_live_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to access webcam.")
        return
    frame_buffer = []
    last_alert_time = 0
    last_sent_label = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera read failed.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb)
            if len(frame_buffer) == SEQUENCE_LENGTH:
                label = detect_anomaly_from_frames(frame_buffer)
                frame_buffer.pop(0)
                color = (0, 255, 0) if label == 'Normal' else (0, 0, 255)
                cv2.putText(frame, f"Detected: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if (label in ALERT_CLASSES and
                    label != last_sent_label and
                    (time.time() - last_alert_time > COOLDOWN)):
                    print(f"⚠️ Anomaly Detected: {label}")
                    # ✅ Send SMS Alert
                    message = f"⚠️ ALERT: {label} detected in LIVE stream at {time.strftime('%I:%M:%S %p')}"
                    send_sms_alert(message)
                    last_alert_time = time.time()
                    last_sent_label = label
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"[ERROR] Live feed error: {e}")
    finally:
        cap.release()

uploaded_video_path = None

@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_video_path
    if 'video' not in request.files:
        return "No video uploaded", 400
    file = request.files['video']
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    uploaded_video_path = os.path.join("uploads", file.filename)
    file.save(uploaded_video_path)
    return render_template("detect.html", video_url=f"/uploaded_video_feed?ts={int(time.time())}")

@app.route('/uploaded_video_feed')
def uploaded_video_feed():
    return Response(generate_uploaded_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_uploaded_video_stream():
    global uploaded_video_path
    if uploaded_video_path is None:
        return
    cap = cv2.VideoCapture(uploaded_video_path)
    frame_buffer = []
    last_alert_time = 0
    last_sent_label = None
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame = cv2.putText(
                    img=np.zeros((480, 640, 3), dtype=np.uint8),
                    text="Video Finished.",
                    org=(100, 240),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2
                )
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb)
            if len(frame_buffer) == SEQUENCE_LENGTH:
                label = detect_anomaly_from_frames(frame_buffer)
                frame_buffer.pop(0)
                color = (0, 255, 0) if label == 'Normal' else (0, 0, 255)
                cv2.putText(frame, f"Detected: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if (label in ALERT_CLASSES and
                    label != last_sent_label and
                    (time.time() - last_alert_time > COOLDOWN)):
                    print(f"⚠️ Anomaly Detected in Uploaded Video: {label}")
                    # ✅ Send SMS Alert
                    message = f"⚠️ ALERT: {label} detected in uploaded video at {time.strftime('%I:%M:%S %p')}"
                    send_sms_alert(message)
                    last_alert_time = time.time()
                    last_sent_label = label
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"[ERROR] Uploaded video stream error: {e}")
    finally:
        cap.release()

if __name__ == '__main__':
    app.run(debug=True)