# FaceTrack-Cursor
**A high-precision, hands-free mouse controller powered by Computer Vision.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Tracking-orange)

SteadyMouse AI solves the biggest problem with head-tracking mice: **The Jitter.**
Most facial trackers are "floaty" or shake when you try to hold still. This project implements a custom **"Parking Brake" Stabilizer** that locks the cursor when you stop moving, allowing for pixel-perfect clicking without the shake.

## ✨ Key Features

* **🛑 The "Parking Brake":** A dynamic deadzone that ignores micro-jitters. If your head moves less than 4 pixels, the cursor stays rock solid.
* **🏎️ Velocity-Based Smoothing:**
    * *Moving Slow?* Heavy smoothing (0.95) for precision work.
    * *Moving Fast?* Zero smoothing for instant flicks across the screen.
* **📌 Click Pinning:** The cursor automatically freezes the moment you make a click gesture, preventing the "slip" that happens when your hand muscles tense up.
* **📐 9-Point Calibration:** Maps your specific range of motion to the screen, so you don't have to strain your neck to reach corners.
* **⚡ Low Latency:** Optimized multiprocessing pipeline separates the Camera (Vision) from the Mouse (UI) to prevent lag.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/SteadyMouse-AI.git](https://github.com/YourUsername/SteadyMouse-AI.git)
    cd SteadyMouse-AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python main.py
    ```

## 🎮 How to Use

1.  **Calibration (First Run):**
    * When the app starts, you will see a green target.
    * Look at the target and press **`C`** on your keyboard.
    * Repeat for all 9 points.
    * *Tip: Keep your head in a comfortable, natural position. You don't need to stretch to the edges of your room!*

2.  **Controls:**
    * **Move Cursor:** Rotate your head (Nose pointing).
    * **Left Click:** Pinch **Index Finger** to Thumb.
    * **Right Click:** Pinch **Middle Finger** to Thumb.
    * **Quit:** Press `Ctrl+C` in the terminal or close the window.

## 🧠 How it Works

### The Tracking Logic
Instead of using eye-tracking (which is noisy and affected by lighting), this project uses the **Rigid Facial Anchors** provided by MediaPipe.



We average the coordinates of the **Nose Bridge (168)**, **Between Eyes (197)**, **Nose Tip (4)**, and **Chin (152)**. This "Center of Mass" approach cancels out individual landmark jitter, providing a much cleaner signal than tracking a single point.

### The Math (Stabilization)
The movement logic follows a custom physics curve:
```python
if distance < DEADZONE:
    return previous_position  # Parking Brake
else:
    # Adapt smoothing based on speed
    smoothing = map_range(speed, low=15, high=100, out_min=0.95, out_max=0.50)
    return current_position * (1-smoothing) + previous_position * smoothing
⚙️ Configuration
You can tweak the feel of the mouse by editing the constants at the top of main.py:

DEADZONE: Increase this if the cursor still shakes when you are still.

SPEED_THRESHOLD: Lower this to make the mouse "snap" to fast movement sooner.

SLOW_SMOOTH: Increase (up to 0.99) for a heavier, "weightier" feel.

📦 Requirements
opencv-python

mediapipe

pynput

pyautogui

numpy
