import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
file_path = "C:/Users/dream/Documents/Drone_onboard_multi_modal_sensor_dataset.csv"
df = pd.read_csv(file_path, sep=";")

# Create 3D Scatter Plot of Drone Paths
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['position_x'], df['position_y'], df['position_z'], 
                     c=df['velocity_x'], cmap='coolwarm', alpha=0.7)

ax.set_xlabel("Position X")
ax.set_ylabel("Position Y")
ax.set_zlabel("Position Z")
ax.set_title("3D Drone Flight Paths")

fig.colorbar(scatter, label="Velocity X")
plt.show()
