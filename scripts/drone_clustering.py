import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
file_path = "C:/Users/dream/Documents/Drone_onboard_multi_modal_sensor_dataset.csv"
df = pd.read_csv(file_path, sep=";")  # Adjust separator if needed

# Print column names for verification
print("Columns in dataset:", df.columns)

# Select relevant features for clustering
features = ['wind_speed', 'altitude', 'velocity_x', 'velocity_y', 'velocity_z']
X = df[features]

# Display initial rows
print(X.head())

# Step 1: Elbow Method to Find Optimal k
inertia = []
k_values = [2, 3, 4, 5, 6]
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, 'o--', markersize=8)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Step 2: Perform K-Means Clustering with k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Step 3: Visualizing Clusters in 2D
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['velocity_x'], df['velocity_y'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel("Velocity X")
plt.ylabel("Velocity Y")
plt.title("Drone Clustering based on Velocity")
plt.colorbar(scatter, label="Cluster")
plt.show()

# Step 4: 3D Clustering Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['velocity_x'], df['velocity_y'], df['velocity_z'], c=df['cluster'], cmap='viridis', alpha=0.7)
ax.set_xlabel("Velocity X")
ax.set_ylabel("Velocity Y")
ax.set_zlabel("Velocity Z")
ax.set_title("3D Drone Clustering")
fig.colorbar(scatter, label="Cluster")
plt.show()

# Step 5: Cluster Analysis
print("\nCluster Statistics:")
print(df.groupby("cluster")[features].mean())
print("\nCluster Distribution:")
print(df['cluster'].value_counts())

# Step 6: Outlier Analysis (Cluster 3)
cluster_3_df = df[df['cluster'] == 3]
print("\nCluster 3 (Outlier) Statistics:")
print(cluster_3_df.describe())

# Step 7: Drone Velocity Trends
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['velocity_x'], label="Velocity X", alpha=0.7)
plt.plot(df.index, df['velocity_y'], label="Velocity Y", alpha=0.7)
plt.plot(df.index, df['velocity_z'], label="Velocity Z", alpha=0.7)
plt.xlabel("Timestamp Index")
plt.ylabel("Velocity")
plt.title("Drone Velocity Trends Over Time")
plt.legend()
plt.show()

# Step 8: **Integrate RF Data for Classification**
np.random.seed(42)
df['rf_signal_strength'] = np.random.uniform(-90, -30, size=len(df))  # dBm values
df['rf_frequency'] = np.random.choice([2.4, 5.8, 900, 433], size=len(df))  # GHz values
df['rf_modulation'] = np.random.choice(['FSK', 'QAM', 'OFDM', 'AM'], size=len(df))

def classify_drone(row):
    if row['rf_frequency'] in [900, 433] and row['rf_modulation'] in ['FSK', 'AM']:
        return 'Military'
    elif row['rf_frequency'] in [2.4, 5.8] and row['rf_modulation'] in ['QAM', 'OFDM']:
        return 'Commercial'
    else:
        return 'Rogue'

df['drone_type'] = df.apply(classify_drone, axis=1)

# Step 9: Save Updated Dataset
output_path = "C:/Users/dream/Documents/AI_drone/datasets/drone_dataset/drone_classified.csv"
df.to_csv(output_path, index=False)
print(f"✅ Drone classification complete. Results saved at {output_path}")
