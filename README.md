# ✌️ Real-Time Hand Gesture Recognition: Rock-Paper-Scissors

A real-time, computer-vision-based game that uses hand gesture recognition to play Rock-Paper-Scissors. Built with **Python**, **OpenCV**, **MediaPipe**, and **Flask**, this application streams video to a web browser and classifies hand gestures instantly.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Game Modes](#-game-modes)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)

---

## 📖 Overview
This project captures video input from a webcam, processes it using MediaPipe's Hand tracking model, and determines the state of each finger using geometric logic. It then translates these finger states into game moves (Rock, Paper, or Scissors).

The application is served via a **Flask** web server, allowing you to view the gameplay and analysis directly in your web browser.

---

## ✨ Features
* **Real-Time Tracking:** High-speed hand detection and landmark tracking using MediaPipe.
* **Gesture Classification:** Accurately distinguishes between Rock, Paper, and Scissors based on finger positioning.
* **Dual Game Modes:** Automatically switches logic based on the number of hands detected (1 vs. 2).
* **AI Opponent:** Includes a countdown timer and randomized machine moves for single-player mode.
* **Visual Feedback:** Displays FPS, Handedness (Left/Right), and Skeleton overlays on the video feed.
* **Web Streaming:** Uses Flask to stream processed video frames to a local web page.

---

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **Web Framework:** Flask
* **Computer Vision:** OpenCV (`cv2`)
* **ML/Tracking:** MediaPipe
* **Math/Data:** NumPy

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd real-time-gesture-recognition


real-time-gesture-recognition/
│
├── templates/
│   └── index.html       # HTML template for the video feed
├── app.py               # Main application logic (Flask + OpenCV + MediaPipe)
├── requirements.txt     # List of Python dependencies
├── README.md            # Project documentation
└── LICENSE              # Project License