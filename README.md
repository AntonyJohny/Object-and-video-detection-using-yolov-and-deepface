# 🛡️ Object and Video Detection using YOLOv7 and DeepFace

This repository contains the practical implementation of the AI-powered hybrid detection framework designed for real-time object and face analysis. It leverages a hybrid approach by combining general object detection with specific facial recognition to create a unified dashboard for advanced security monitoring, asset tracking, and access control. 

This system serves as the foundational architecture for the published research paper: *"Design and implementation of an AI-powered hybrid detection framework for real-time object and face analysis."*

## ✨ Key Features
* **Hybrid Intelligence Pipeline:** Runs YOLOv7 (Object Detection) and DeepFace (Facial Recognition) in a seamless cascade architecture.
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
git clone [https://github.com/AntonyJohny/Object-and-video-detection-using-yolov-and-deepface.git](https://github.com/AntonyJohny/Object-and-video-detection-using-yolov-and-deepface.git)
cd Object-and-video-detection-using-yolov-and-deepface
