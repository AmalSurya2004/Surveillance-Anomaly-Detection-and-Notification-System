import cv2
import torch
import torchvision.transforms as transforms
from model.anomaly_classifier import AnomalyDetector
from utils.notifier import send_sms_alert
import os
import time
import sys

# Config
LABELS = ['Abuse', 'Accident', 'Arrest', 'Assault', 'Burglary', 'Fighting', 'Normal', 'Robbery']
IMG_SIZE = 128
SEQUENCE_LENGTH = 20
ALERT_INTERVAL = 30  # seconds

# Input video path
if len(sys.argv) < 2:
    print("Usage: python upload_detect.py <path_to_video>")
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.exists(video_path):
    print(f"Video not found: {video_path}")
    sys.exit(1)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyDetector(num_classes=len(LABELS)).to(device)
model.load_state_dict(torch.load("models/anomaly_model.pth", map_location=device))
model.eval()

# Load video
cap = cv2.VideoCapture(video_path)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
frame_buffer = []
last_alert_time = 0
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Processing {total_frames} frames.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb)
            frame_buffer.append(tensor)
        except Exception as e:
            print(f"[ERROR] Frame preprocessing failed at frame {frame_count}: {e}")
            continue

        if len(frame_buffer) > SEQUENCE_LENGTH:
            frame_buffer.pop(0)
        if len(frame_buffer) < SEQUENCE_LENGTH:
            continue

        input_tensor = torch.stack(frame_buffer).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, 1)
                pred = torch.argmax(probs, 1).item()
                label = LABELS[pred]
                confidence = probs[0, pred].item()
        except Exception as e:
            print(f"[ERROR] Model inference failed: {e}")
            label = "Unknown"
            confidence = 0.0

        color = (0, 255, 0) if label == 'Normal' else (0, 0, 255)
        cv2.putText(frame, f"Anomaly: {label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Frame {frame_count}/{total_frames}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if label != 'Normal' and (time.time() - last_alert_time > ALERT_INTERVAL):
            try:
                send_sms_alert(f"⚠️ Anomaly Detected in uploaded video: {label} ({confidence:.2f})")
            except Exception as e:
                print(f"[ERROR] SMS alert failed: {e}")
            last_alert_time = time.time()

        cv2.imshow("Uploaded Video Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Exiting uploaded video anomaly detection.")
            break
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video detection finished.")
