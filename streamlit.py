import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import winsound  # For alert beeping
import os  # For Linux/macOS alert sound
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load YOLO model
MODEL_PATH = r"C:\Users\dream\Documents\AI_drone\runs\detect\train\weights\best.pt"
model = YOLO(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="AI Drone Threat Detection", layout="centered")
st.title("🚀 AI Drone Threat Detection & 3D Flight Simulation")
st.write("Upload an image and provide RF & velocity values for analysis.")

# Upload image
uploaded_file = st.file_uploader("📸 Upload JPEG Image", type=["jpg", "jpeg"])

# User inputs for RF values and velocity
rf_signal = st.number_input("📡 Enter RF Signal Strength", min_value=-100.0, max_value=100.0, value=0.0)
velocity_x = st.number_input("💨 Enter X Velocity", value=0.0)
velocity_y = st.number_input("💨 Enter Y Velocity", value=0.0)
velocity_z = st.number_input("💨 Enter Z Velocity", value=0.0)

# Function to trigger a beep alert
def beep_alert():
    duration = 500  # milliseconds
    frequency = 1000  # Hz
    try:
        winsound.Beep(frequency, duration)  # Windows beep
    except:
        os.system("echo -e '\a'")  # Linux/macOS terminal beep

# Function for flashing red background on threat detection
def flash_red():
    st.markdown(
        """
        <style>
        body { background-color: red !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to generate 3D drone flight simulation
def create_drone_flight_plot():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Simulated drone trajectory
    time_steps = np.linspace(0, 10, 100)
    x_traj = velocity_x * time_steps
    y_traj = velocity_y * time_steps
    z_traj = velocity_z * time_steps
    
    ax.plot(x_traj, y_traj, z_traj, label='Drone Path', color='blue')
    ax.scatter(x_traj[0], y_traj[0], z_traj[0], color='green', s=50, label='Start')
    ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='red', s=50, label='End')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Drone Flight Simulation')
    ax.legend()
    
    return fig

# Add a submit button
if st.button("🔍 Detect Threat & Simulate Flight"):
    if uploaded_file is not None:
        # Convert image to OpenCV format
        image = Image.open(uploaded_file)
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Run YOLO detection
        results = model(image_cv)
        detections = []
        threat_detected = False  # Flag for threat

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_name = model.names[int(box.cls[0])]

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                # Draw bounding box
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_cv, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Determine if it's a threat
                if confidence > 0.7:
                    threat_detected = True

        # Display results
        st.image(image_cv, caption="📷 Processed Image", use_column_width=True)
        st.json({"RF Signal": rf_signal, "Velocity": [velocity_x, velocity_y, velocity_z], "Detections": detections})

        # Display threat status
        if threat_detected:
            st.error("⚠ *Threat Detected!* Take immediate action.")
            flash_red()  # Flash red background
            beep_alert()  # Sound alert
            st.snow()  # Visual effect for attention
        else:
            st.success("✅ No Threat Detected. Area is safe.")

        # Display 3D flight simulation
        fig = create_drone_flight_plot()
        st.pyplot(fig)
    else:
        st.warning("⚠ Please upload an image first!")
