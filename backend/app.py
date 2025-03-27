from flask import Flask, request, jsonify, send_from_directory
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__, static_folder="frontend/public", static_url_path="")

# Load YOLO Model
model_path = r"C:\Users\dream\Documents\AI_drone\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

# Serve the frontend
@app.route("/")
def index():
    return send_from_directory("frontend/public", "drone.html")

# Serve static files (CSS, JS)
@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("frontend/public", filename)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model(image)
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": confidence})

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
