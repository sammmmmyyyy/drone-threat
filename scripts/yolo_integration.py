import torch
import cv2
import pandas as pd
import os
from ultralytics import YOLO  # ✅ Correct YOLOv8 Import

# ✅ Corrected Paths
MODEL_PATH = "C:/Users/dream/Documents/AI_drone/runs/detect/train/weights/best.pt"
DRONE_METADATA_PATH = "C:/Users/dream/Documents/AI_drone/datasets/drone_dataset/drone_classified.csv"

# ✅ Check if YOLO Model Exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
print("✅ YOLO Model Found Successfully")

# ✅ Load YOLOv8 Model Correctly
try:
    model = YOLO(MODEL_PATH)  # ✅ Uses the correct YOLOv8 class
    print("✅ YOLO Model Loaded Successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load YOLO model: {e}")

# ✅ Check if Drone Metadata Exists
if not os.path.exists(DRONE_METADATA_PATH):
    raise FileNotFoundError(f"❌ Drone metadata not found at {DRONE_METADATA_PATH}")
print("✅ Drone Metadata Found Successfully")

# ✅ Load Drone Data
drone_df = pd.read_csv(DRONE_METADATA_PATH)
print(f"📊 Loaded Drone Data: {drone_df.shape[0]} records")

# ✅ Function to Perform Object Detection
def detect_objects(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at {image_path}")

    results = model(image_path)  # ✅ YOLOv8 correct inference method

    # ✅ Process Results
    detected_objects = results[0].boxes.data.cpu().numpy()  # ✅ Extract bounding boxes

    return detected_objects

# ✅ Example Usage
if __name__ == "__main__":
    test_image = "C:\\Users\\dream\\Documents\\AI_drone\\datasets\\drone_dataset\\train\\images\\pic_037.jpg"  # ✅ Corrected

    if os.path.exists(test_image):
        detections = detect_objects(test_image)
        print("📌 Detection Results:\n", detections)
    else:
        print("⚠️ No test image found, skipping detection step.")
