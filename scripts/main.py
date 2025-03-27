import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:/Users/dream/Documents/Drone_onboard_multi_modal_sensor_dataset.csv"
df = pd.read_csv(file_path, sep=";")

# Select relevant features
features = ['wind_speed', 'altitude', 'velocity_x', 'velocity_y', 'velocity_z']

# ✅ Step 1: Display Distribution of Features
plt.figure(figsize=(12, 6))
df[features].hist(bins=30, figsize=(10, 6), edgecolor='black')
plt.suptitle("Distribution of Drone Flight Features", fontsize=14)
plt.show()

# ✅ Step 2: Boxplot for Outlier Detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[features])
plt.title("Outlier Detection in Drone Data (Boxplot)")
plt.xticks(rotation=15)
plt.show()

# ✅ Step 3: Pairplot for Feature Relationships
sns.pairplot(df[features], diag_kind="kde")
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()

# ✅ Step 4: Filter Outliers (assuming cluster 3 was an outlier)
df_outliers = df[df['velocity_x'] > df['velocity_x'].quantile(0.99)]  # Example threshold

print("Outlier Drones Identified:", df_outliers.shape[0])

# ✅ Ensure the folder exists before saving
output_folder = "datasets/drone_data/"
output_file = output_folder + "outliers_detected.csv"

os.makedirs(output_folder, exist_ok=True)  # Creates folder if it doesn't exist

# Save outlier data for further analysis
df_outliers.to_csv(output_file, index=False)
print(f"Outliers saved to {output_file}")

# ✅ Step 5: Heatmap to visualize feature correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Drone Features")
plt.show()
