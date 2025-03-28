# AI Drone Threat Detection & 3D Flight Simulation

## 📌 Project Overview
This project integrates **AI-powered object recognition** and **3D flight simulation** to detect potential drone threats in real-time. It uses **YOLOv8 for object detection** and **3D visualization of drone flight paths** based on velocity inputs.

## ✨ Features
- **AI-Powered Threat Detection**: Detects drones from images using YOLOv8.
- **RF Signal Analysis**: Evaluates the radio frequency signal strength.
- **Velocity-Based 3D Flight Simulation**: Simulates the drone's movement in 3D.
- **Real-Time Alerts**: Provides visual and audio alerts for detected threats.
- **Interactive Streamlit UI**: User-friendly web interface for easy input and visualization.

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (for UI)
- **YOLOv8 (Ultralytics)** (for drone detection)
- **Open3D** (for 3D visualization)
- **NumPy & OpenCV** (for image processing)
- **Torch** (for deep learning model execution)

## 🚀 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/AI-Drone-Threat-Detection.git
cd AI-Drone-Threat-Detection
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run streamlit.py
```

## 📷 Usage
1. **Upload an image** of a flying object.
2. **Enter RF signal strength** and **velocity values** (X, Y, Z).
3. Click **"Detect Threat"** to analyze the image.
4. **View Results**:
   - If a threat is detected, the app will **flash red** and sound an alert.
   - A **3D flight trajectory** of the detected drone will be displayed.
   
## 📂 Project Structure
```
AI-Drone-Threat-Detection/
│── models/                     # Pre-trained YOLOv8 model
│── datasets/                   # Drone detection dataset
│── streamlit.py                # Main application script
│── requirements.txt            # Dependencies
│── README.md                   # Project Documentation
```

## ⚡ Future Improvements
- **Live Video Feed Processing**
- **Blockchain Integration for Secure Logging**
- **Integration with RF Signal Analysis Module**
- **Support for More AI Models**

## 🤝 Contributing
1. Fork the repository 🍴
2. Create a new branch 🚀
3. Commit your changes 🔥
4. Submit a pull request ✅

## 📜 License
This project is **open-source** under the MIT License.

---
🔗 **Author**: [Your Name](https://github.com/your-username)  
📧 **Contact**: your.email@example.com

