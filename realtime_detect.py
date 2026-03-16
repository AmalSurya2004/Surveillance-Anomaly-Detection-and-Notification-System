import cv2
import torch
import time
import numpy as np
from torchvision import transforms
from model.anomaly_classifier import AnomalyDetector
from utils.notifier import send_audio_alert, send_desktop_notification, send_email_alert, send_sms_alert
from sklearn.preprocessing import LabelEncoder

# === CONFIG ===
SEQUENCE_LENGTH = 20
IMG_SIZE = 128
MODEL_PATH = "models/anomaly_model.pth"
LABELS = ['Abuse', 'Accident', 'Arrest', 'Assault', 'Burglary', 'Fighting', 'Normal', 'Robbery']
ALERT_CLASSES = [label for label in LABELS if label.lower() != "normal"]
ALERT_COOLDOWN = 10  # seconds

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_encoder = LabelEncoder()
label_encoder.fit(LABELS)

def load_model():
    model = AnomalyDetector(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def preprocess_frames(frames):
    processed = [transform(f) for f in frames]
    return torch.stack(processed)

def run_realtime_detection():
    print("\U0001F680 Starting real-time anomaly detection...")
    model = load_model()
    cap = cv2.VideoCapture(0)
    frame_buffer = []
    last_alert_time = time.time() - ALERT_COOLDOWN

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera disconnected or read error.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb)
            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            if len(frame_buffer) == SEQUENCE_LENGTH:
                try:
                    input_tensor = preprocess_frames(frame_buffer).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred = torch.argmax(probs, dim=1).item()
                        predicted_class = label_encoder.inverse_transform([pred])[0]
                        confidence = probs[0, pred].item()
                except Exception as e:
                    print(f"[ERROR] Model inference failed: {e}")
                    predicted_class = "Unknown"
                    confidence = 0.0

                color = (0, 255, 0) if predicted_class.lower() == "normal" else (0, 0, 255)
                cv2.putText(frame, f"Detected: {predicted_class} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Alert if anomaly and cooldown passed
                current_time = time.time()
                if (predicted_class in ALERT_CLASSES and
                        current_time - last_alert_time > ALERT_COOLDOWN):
                    print(f"\U0001F6A8 Anomaly detected: {predicted_class} ({confidence:.2f})")
                    try:
                        send_audio_alert()
                        send_desktop_notification(predicted_class)
                        send_email_alert(predicted_class)
                        send_sms_alert(predicted_class)
                    except Exception as e:
                        print(f"[ERROR] Notification failed: {e}")
                    last_alert_time = current_time

            cv2.imshow("Live Camera Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_detection()
