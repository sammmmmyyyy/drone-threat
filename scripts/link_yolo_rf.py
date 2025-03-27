import pandas as pd
import os

# ✅ Paths for YOLO Detections & RF Data
YOLO_DETECTIONS_PATH = r"C:\Users\dream\Documents\AI_drone\runs\detect\train\results.csv"
RF_DATA_PATH = "C:/Users/dream/Documents/AI_drone/datasets/drone_dataset/drone_rf_values.csv"

# ✅ Check if Files Exist
if not os.path.exists(YOLO_DETECTIONS_PATH):
    raise FileNotFoundError(f"❌ YOLO Detections not found at {YOLO_DETECTIONS_PATH}")
if not os.path.exists(RF_DATA_PATH):
    raise FileNotFoundError(f"❌ RF Data not found at {RF_DATA_PATH}")

# ✅ Load YOLO Detections
df_yolo = pd.read_csv(YOLO_DETECTIONS_PATH)
print(f"📊 Loaded YOLO Detections: {df_yolo.shape[0]} records")

# ✅ Load RF Data
df_rf = pd.read_csv(RF_DATA_PATH)
print(f"📊 Loaded RF Data: {df_rf.shape[0]} records")

# ✅ Ensure Required Columns Exist
yolo_expected_col = "image_name"
rf_expected_col = "image_name"

if yolo_expected_col not in df_yolo.columns:
    raise KeyError(f"❌ Missing '{yolo_expected_col}' in YOLO detections! Found: {df_yolo.columns.tolist()}")

if rf_expected_col not in df_rf.columns:
    raise KeyError(f"❌ Missing '{rf_expected_col}' in RF data! Found: {df_rf.columns.tolist()}")

print("🟢 YOLO Columns:", df_yolo.columns.tolist())
print("🟢 RF Columns:", df_rf.columns.tolist())

# ✅ Merge Data on "image_name"
df_merged = pd.merge(df_yolo, df_rf, on="image_name", how="inner")

# ✅ Save Merged Data
MERGED_OUTPUT_PATH = "C:/Users/dream/Documents/AI_drone/datasets/drone_dataset/merged_yolo_rf.csv"
df_merged.to_csv(MERGED_OUTPUT_PATH, index=False)
print(f"✅ Merged Data Saved: {MERGED_OUTPUT_PATH}")

# ✅ Display Sample
print(df_merged.head())
