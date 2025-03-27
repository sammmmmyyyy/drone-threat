import os

labels_path = "C:/Users/dream/Documents/AI_drone/runs/detect/predict/labels"
confidences = []

for file in os.listdir(labels_path):
    with open(os.path.join(labels_path, file), "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            confidence = float(data[5])  # Confidence score is the 6th value in YOLO format
            confidences.append(confidence)

print(f"Average Confidence Score: {sum(confidences) / len(confidences):.4f}")
print(f"Highest Confidence Score: {max(confidences):.4f}")
print(f"Lowest Confidence Score: {min(confidences):.4f}")
