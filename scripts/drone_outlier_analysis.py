import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:/Users/dream/Documents/Drone_onboard_multi_modal_sensor_dataset.csv"
df = pd.read_csv(file_path, sep=";")

# Select relevant features
features = ['wind_speed', 'altitude', 'velocity_x', 'velocity_y', 'velocity_z']

# Detect Outliers using Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[features])
plt.title("Outlier Detection in Drone Data")
plt.xticks(rotation=15)
plt.show()

# Filter outlier cluster (assuming cluster 3 was the outlier)
df_outliers = df[df['velocity_x'] > df['velocity_x'].quantile(0.99)]  # Example threshold

print("Outlier Drones Identified:", df_outliers.shape[0])

# Save outlier data for further analysis
df_outliers.to_csv("datasets/drone_data/outliers_detected.csv", index=False)
