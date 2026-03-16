import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from model.anomaly_classifier import AnomalyDetector
from dataset.custom_video_dataset import VideoDataset

# === CONFIG ===
IMG_SIZE = 128
SEQUENCE_LENGTH = 20
BATCH_SIZE = 4
TEST_DIR = "data/test"
MODEL_PATH = "models/anomaly_model.pth"
QUANTIZED_MODEL_PATH = "models/anomaly_model_quantized.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    # Prepare transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # Get labels
    labels = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    if not labels:
        print("[ERROR] No labels found in test directory.")
        return

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Initialize dataset and loader
    dataset = VideoDataset(TEST_DIR, label_encoder, transform)
    if len(dataset) == 0:
        print("[ERROR] No samples found in test dataset.")
        return
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model (prefer quantized if present)
    model_path = QUANTIZED_MODEL_PATH if os.path.exists(QUANTIZED_MODEL_PATH) else MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}.")
        return
    model = AnomalyDetector(num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    per_class_correct = {label: 0 for label in labels}
    per_class_total = {label: 0 for label in labels}

    print(f"[INFO] Testing on {len(dataset)} video samples. Batch size: {BATCH_SIZE}")
    with torch.no_grad():
        for batch_idx, (videos, targets) in enumerate(loader):
            videos, targets = videos.to(device), targets.to(device).long()
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            # Class-level stats
            for i in range(len(targets)):
                true_cls = label_encoder.inverse_transform([targets[i].item()])[0]
                pred_cls = label_encoder.inverse_transform([predicted[i].item()])[0]
                per_class_total[true_cls] += 1
                if true_cls == pred_cls:
                    per_class_correct[true_cls] += 1
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: running accuracy = {correct / max(1, total):.2%}")

    acc = correct / total if total > 0 else 0
    print(f"\n✅ Test Accuracy: {acc:.2%}")

    print("\nPer-class accuracy:")
    for label in labels:
        acc_cls = per_class_correct[label] / per_class_total[label] if per_class_total[label] > 0 else 0
        print(f"  {label}: {acc_cls:.2%} ({per_class_correct[label]}/{per_class_total[label]})")

if __name__ == "__main__":
    test()
