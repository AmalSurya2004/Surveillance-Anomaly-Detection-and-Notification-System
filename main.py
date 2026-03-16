import cv2
import torch
import numpy as np
from model.anomaly_classifier import AnomalyDetector
from torchvision import transforms
from utils.notifier import send_alert
import time

# CONFIGURATION
LABELS = ['Accident', 'Abuse', 'Fighting', 'Harassment', 'Robbery', 'Theft', 'Normal']
NUM_CLASSES = len(LABELS)
MODEL_PATH = "models/anomaly_model.pth"
IMG_SIZE = 128
SEQUENCE_LENGTH = 20
ALERT_COOLDOWN = 10  # seconds between repeat alerts

# DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
model = AnomalyDetector(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# PREPROCESSING PIPELINE
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# INITIALIZE WEBCAM
cap = cv2.VideoCapture(0)
frames = []
last_alert_time = 0
last_alert_label = None

print("🚨 Starting real-time anomaly detection... (Press 'q' to quit)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera disconnected or frame read error.")
            break

        # ---- PREPROCESS FRAME ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb)
        frames.append(tensor)
        if len(frames) > SEQUENCE_LENGTH:
            frames.pop(0)
        if len(frames) < SEQUENCE_LENGTH:
            continue  # Fill up frame buffer first

        # ---- INFERENCE ----
        input_tensor = torch.stack(frames).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, 1)
            conf, predicted = torch.max(probs, 1)
            label = LABELS[predicted.item()]
            confidence = conf.item()

        # ---- DISPLAY & ALERT ----
        cv2.putText(frame, f"Anomaly: {label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        current_time = time.time()
        if label != "Normal" and (
            last_alert_label != label or current_time - last_alert_time > ALERT_COOLDOWN):
            send_alert(label)
            last_alert_label = label
            last_alert_time = current_time

        cv2.imshow("Real-time Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting anomaly detection.")
            break
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
