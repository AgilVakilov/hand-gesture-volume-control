# Real-Time Hand Gesture Volume Controller 🖐️🔊

A computer vision pipeline that translates real-time physical hand gestures into operating system-level audio controls.

This project uses a webcam feed to track 3D hand landmarks and calculates the Euclidean distance between the thumb and index finger to dynamically manipulate the Windows Master Volume API.

## 🚀 Features
* **Real-Time Tracking:** Processes video frames on the fly using Google's MediaPipe Tasks API.
* **Hardware Abstraction:** Interfaces directly with the Windows Audio Session API (WASAPI) via PyCaw.
* **Dynamic Model Loading:** The script automatically fetches and downloads the required `hand_landmarker.task` AI model directly from Google Cloud Storage on the first run, ensuring a seamless plug-and-play experience.
* **Visual Feedback:** Renders on-screen visual overlays (skeleton mapping and connection lines) using OpenCV to show tracking status.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`opencv-python`)
* **Machine Learning / AI:** MediaPipe Vision Tasks API
* **Mathematics:** NumPy, Math (Standard Library)
* **OS Integration:** PyCaw (Python Core Audio Windows Library)

## 🧠 How It Works (Under the Hood)
1. **Frame Capture:** OpenCV captures the webcam feed and converts the BGR color space to RGB, which MediaPipe requires.
2. **Landmark Detection:** The MediaPipe Hand Landmarker model identifies 21 unique 3D coordinates on the detected hand.
3. **Distance Calculation:** The script isolates **Landmark 4** (tip of the thumb) and **Landmark 8** (tip of the index finger). It calculates the Euclidean distance between these two points using `math.hypot()`.
4. **Interpolation & Control:** The physical pixel distance (e.g., 20px to 200px) is mapped to the Windows volume range (e.g., -65.25dB to 0.0dB) using `numpy.interp()`. This value is then passed to the PyCaw audio interface to instantly adjust the system volume.

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AgilVakilov/hand-gesture-volume-control.git](https://github.com/AgilVakilov/hand-gesture-volume-control.git)
   cd hand-gesture-volume-control
   

## 👨‍💻 Author
**Agil** *Computer Engineering @ Ankara University* | [GitHub](https://github.com/AgilVakilov)
