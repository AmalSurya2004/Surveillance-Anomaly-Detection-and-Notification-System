import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

# === CONFIG ===
RAW_DIR = Path("data/raw/Real-life_Violence_Situations_Videos")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
IMG_SIZE = 128
FOURCC_STR = 'XVID'
SPLIT_RATIO = 0.2
FRAME_RATE = 15.0

def convert_and_save(video_path, save_path):
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*FOURCC_STR)
    out = cv2.VideoWriter(str(save_path), fourcc, FRAME_RATE, (IMG_SIZE, IMG_SIZE))
    success = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            out.write(resized)
            success = True
        except Exception as e:
            print(f"[ERROR] Frame error in {video_path}: {e}")
    cap.release()
    out.release()
    return success

def prepare_data():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in os.listdir(RAW_DIR):
        class_path = RAW_DIR / class_name
        if not class_path.is_dir():
            continue

        videos = [f for f in os.listdir(class_path) if f.endswith(".mp4")]
        train_videos, test_videos = train_test_split(videos, test_size=SPLIT_RATIO, random_state=42)

        for folder, video_list in zip([TRAIN_DIR, TEST_DIR], [train_videos, test_videos]):
            class_folder = folder / class_name
            class_folder.mkdir(parents=True, exist_ok=True)

            print(f"\n[INFO] Processing class: {class_name} | {folder.name}")
            for video in video_list:
                video_path = class_path / video
                save_path = class_folder / (Path(video).stem + ".avi")
                if save_path.exists():
                    print(f"[SKIP] Exists: {save_path}")
                    continue
                success = convert_and_save(video_path, save_path)
                status = "[✓] Saved" if success else "[✗] Error"
                print(f"{status}: {save_path}")

if __name__ == "__main__":
    prepare_data()
