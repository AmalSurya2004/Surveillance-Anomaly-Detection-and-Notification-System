import os
import cv2
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# === CONFIG ===
IMG_SIZE = 128
SEQUENCE_LENGTH = 20
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.001
TRAIN_DIR = "data/train"
MODEL_PATH = "models/anomaly_model.pth"
QUANTIZED_MODEL_PATH = "models/anomaly_model_quantized.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class VideoDataset(Dataset):
    def __init__(self, folder, label_encoder, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.encoder = label_encoder

        for class_name in sorted(os.listdir(folder)):
            class_path = os.path.join(folder, class_name)
            if not os.path.isdir(class_path): continue
            for file in os.listdir(class_path):
                if not file.endswith(".avi"): continue
                self.samples.append(os.path.join(class_path, file))
                self.labels.append(class_name)
        if not self.samples:
            print(f"[ERROR] No video samples found in {folder}.")
        self.labels = self.encoder.transform(self.labels)

    def __len__(self):
        return len(self.samples)

    def read_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        if len(frames) >= SEQUENCE_LENGTH:
            frames = frames[:SEQUENCE_LENGTH]
        else:
            frames += [frames[-1]] * (SEQUENCE_LENGTH - len(frames))
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_tensor = self.read_frames(self.samples[idx])
        label = self.labels[idx]
        return video_tensor, label

# --- Model ---
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        return x

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class AnomalyDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(input_dim=16384, hidden_dim=256, num_classes=num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

# --- Training Function ---
def train():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    labels = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    if not labels:
        print("[ERROR] No label folders found in train directory.")
        return

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    dataset = VideoDataset(TRAIN_DIR, label_encoder, transform)
    if len(dataset) == 0:
        print("[ERROR] No video samples for training.")
        return
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AnomalyDetector(num_classes=len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    Path("models").mkdir(exist_ok=True)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for videos, targets in loader:
            videos, targets = videos.to(device), targets.long().to(device)
            outputs = model(videos)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        scheduler.step()
        acc = correct / total if total > 0 else 0

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.LSTM, nn.Linear},
                dtype=torch.qint8
            )
            torch.save(quantized_model.state_dict(), QUANTIZED_MODEL_PATH)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.4f} | Accuracy: {acc:.2%} | Best: {best_acc:.2%}")

    print(f"\n✅ Best Accuracy Achieved: {best_acc:.2%}")
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Quantized model saved to {QUANTIZED_MODEL_PATH}")

if __name__ == "__main__":
    train()
