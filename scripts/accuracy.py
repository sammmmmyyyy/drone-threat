import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load confusion matrix
conf_matrix_path = "C:/Users/dream/Documents/AI_drone/runs/detect/train/confusion_matrix.png"

# Manually define confusion matrix values from the image
# (If you have the matrix as a .txt or .npy file, load it directly)
TP = 150  # Example: Drones correctly detected
FP = 10   # Example: Non-drones wrongly detected as drones
FN = 20   # Example: Missed drones
TN = 200  # Example: Correctly identified empty images

# Compute Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
