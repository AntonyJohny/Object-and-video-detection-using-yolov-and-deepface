# 🛡️ Object and Video detection using yolov and deepface

Project Sentinel is an AI-powered, real-time intelligent surveillance web application. It leverages a hybrid approach by combining general object detection with specific facial recognition to create a unified dashboard for advanced security monitoring, asset tracking, and access control.

## ✨ Key Features
* **Hybrid Intelligence:** Runs YOLOv7 (Object Detection) and DeepFace (Facial Recognition) in a seamless cascade pipeline.
* **Real-Time Web Dashboard:** A dark-themed, responsive Flask UI displaying live annotated video feeds.
* **Dual Logging System:** Segregates tracking into "Person/Identity Logs" and "Object Detection Logs" in real-time.
* **Interactive Enrollment:** "One-shot" learning interface allows administrators to capture and enroll new faces on-the-fly without restarting the server.
* **Secure Authentication:** Protected by Bcrypt password hashing and Flask-Mail OTP (One-Time Password) verification.
* **Video Forensic Analysis:** Upload pre-recorded videos for high-speed offline analysis using GPU/CPU processing.

## 🛠️ Technology Stack
* **Backend:** Python, Flask, Flask-SQLAlchemy, Flask-Login
* **Computer Vision:** OpenCV (`cv2`)
* **Machine Learning:** PyTorch, YOLOv7, DeepFace (VGG-Face)
* **Frontend:** HTML5, CSS3, JavaScript, Slim Select

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/Project-Sentinel.git](https://github.com/YOUR_USERNAME/Project-Sentinel.git)
cd Project-Sentinel
