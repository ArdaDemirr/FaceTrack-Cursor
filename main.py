"""
Project: Steady-Hand AI Mouse
Description: A hands-free mouse controller using Face Landmarks (MediaPipe).
             Features a "Parking Brake" stabilizer to eliminate jitter and 
             "Click Pinning" for precise interaction.
             
Dependencies:
    pip install opencv-python mediapipe pynput pyautogui numpy
"""

import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import pyautogui
import math
import numpy as np
import time
from multiprocessing import Process, Pipe

# --- 1. CONFIGURATION ---
# Camera Settings
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480

# Screen Settings
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Failsafe: Python usually crashes if the mouse hits the corner. 
# We disable this because head-tracking often hits corners.
pyautogui.FAILSAFE = False 

# --- 2. THE "PARKING BRAKE" STABILIZER ---
class SteadyStabilizer:
    """
    A custom smoothing filter designed to fix the "Jitter vs. Lag" trade-off.
    It acts like a physical object with mass: hard to start moving (steady),
    but easy to keep moving once started (responsive).
    """
    def __init__(self):
        self.prev_x, self.prev_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        
        # --- TUNING PARAMETERS ---
        # 1. THE PARKING BRAKE (Deadzone)
        # If the calculated movement is less than this many pixels, 
        # we assume it's sensor noise and don't move the cursor at all.
        self.DEADZONE = 4.0 
        
        # 2. SLOW MOVEMENT (Precision Mode)
        # When moving slowly, we lean heavily on the previous position (0.95).
        # This makes the cursor feel "heavy" and precise.
        self.SLOW_SMOOTH = 0.95 
        
        # 3. FAST MOVEMENT (Flick Mode)
        # When moving fast, we reduce smoothing (0.50) to minimize lag.
        self.FAST_SMOOTH = 0.50 
        
        # 4. VELOCITY THRESHOLD
        # The pixel distance per frame required to switch from Slow -> Fast mode.
        self.SPEED_THRESHOLD = 15 

    def update(self, raw_x, raw_y):
        """
        Calculates the next cursor position based on raw input and current velocity.
        """
        # 1. Measure Speed: How far are we trying to move?
        dist = math.hypot(raw_x - self.prev_x, raw_y - self.prev_y)

        # 2. APPLY PARKING BRAKE
        # If the movement is tiny (micro-jitter), ignore it completely.
        # This allows you to stare at a button without the cursor shaking.
        if dist < self.DEADZONE:
            return int(self.prev_x), int(self.prev_y)

        # 3. CALCULATE DYNAMIC SMOOTHING
        # Default to heavy smoothing (Precision Mode)
        smoothing = self.SLOW_SMOOTH
        
        # If we are moving fast (Flick), linearly decrease smoothing
        if dist > self.SPEED_THRESHOLD:
            # Calculate a factor from 0.0 to 1.0 based on how fast we are going
            factor = min(1.0, (dist - self.SPEED_THRESHOLD) / 80.0)
            
            # Blend between Slow and Fast smoothing based on speed
            smoothing = self.SLOW_SMOOTH - (self.SLOW_SMOOTH - self.FAST_SMOOTH) * factor

        # 4. APPLY FILTER (Exponential Moving Average)
        # New Pos = (Old Pos * Weight) + (New Input * (1 - Weight))
        new_x = self.prev_x * smoothing + raw_x * (1 - smoothing)
        new_y = self.prev_y * smoothing + raw_y * (1 - smoothing)

        # Update state for next frame
        self.prev_x, self.prev_y = new_x, new_y
        
        return int(new_x), int(new_y)

# --- 3. VISION WORKER (Child Process) ---
def ai_worker(conn):
    """
    Runs the heavy Computer Vision tasks in a separate CPU core.
    This ensures the camera lag doesn't freeze the mouse cursor.
    """
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    
    # Setup Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW is faster on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Force lowest latency buffer

    # 
    # LANDMARK DEFINITIONS:
    # We use a vertical line down the center of the face for stability.
    # 4: Nose Tip | 168: Nose Bridge | 197: Between Eyes | 152: Chin
    ANCHOR_POINTS = [4, 168, 197, 152]

    # Initialize MediaPipe
    with mp_face.FaceMesh(refine_landmarks=True, 
                          min_detection_confidence=0.7, 
                          min_tracking_confidence=0.7) as face_mesh, \
         mp_hands.Hands(max_num_hands=1, 
                        min_detection_confidence=0.7, 
                        min_tracking_confidence=0.7) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            # Mirror the frame so left-is-left
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run AI Inference
            res_face = face_mesh.process(rgb)
            res_hand = hands.process(rgb)
            
            packet = {'found': False, 'x': 0, 'y': 0, 'click_l': False, 'click_r': False}

            # 1. PROCESS HEAD POSITION
            if res_face.multi_face_landmarks:
                packet['found'] = True
                lm = res_face.multi_face_landmarks[0].landmark
                
                # We average 4 rigid points. This acts as a hardware noise filter.
                # If the nose tip jitters but the chin doesn't, the average stays steady.
                packet['x'] = sum(lm[i].x for i in ANCHOR_POINTS) / len(ANCHOR_POINTS)
                packet['y'] = sum(lm[i].y for i in ANCHOR_POINTS) / len(ANCHOR_POINTS)

            # 2. PROCESS HAND GESTURES (CLICKS)
            if res_hand.multi_hand_landmarks:
                h_lm = res_hand.multi_hand_landmarks[0].landmark
                thumb = h_lm[4]
                index = h_lm[8]
                middle = h_lm[12]
                
                # Check distance between Thumb and Index (Left Click)
                if math.hypot(index.x - thumb.x, index.y - thumb.y) < 0.05: 
                    packet['click_l'] = True
                
                # Check distance between Thumb and Middle (Right Click)
                if math.hypot(middle.x - thumb.x, middle.y - thumb.y) < 0.05: 
                    packet['click_r'] = True

            # Send data to Main Process
            conn.send(packet)

# --- 4. MAIN LOOP (UI & Mouse Control) ---
def draw_target(idx):
    """Draws the calibration dots on screen."""
    img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    targets = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1), # Top Row
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5), # Middle Row
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)  # Bottom Row
    ]
    if idx < 9:
        tx = int(SCREEN_WIDTH * targets[idx][0])
        ty = int(SCREEN_HEIGHT * targets[idx][1])
        cv2.circle(img, (tx, ty), 40, (0, 255, 0), 2) # Outer ring
        cv2.circle(img, (tx, ty), 5, (0, 255, 0), -1) # Inner dot
        cv2.putText(img, "HOLD STEADY & PRESS 'C'", (tx-150, ty+80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return img

def main():
    # Create a Pipe for communication between processes
    parent_conn, child_conn = Pipe()
    
    # Start the Vision Process
    p = Process(target=ai_worker, args=(child_conn,))
    p.daemon = True
    p.start()
    
    mouse = Controller()
    stabilizer = SteadyStabilizer()
    
    # --- CALIBRATION STATE ---
    calib_data = []
    calib_idx = 0
    is_calibrated = False
    
    # Default Bounds (Will be overwritten by calibration)
    # These represent the "Active Box" of your head movement
    X_BOUNDS = [0.4, 0.6] 
    Y_BOUNDS = [0.4, 0.6]

    l_held, r_held = False, False

    # Create Fullscreen Window
    cv2.namedWindow("Calib", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calib", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Booting up... Please Calibrate.")

    while True:
        # Check if there is data from the AI Worker
        if parent_conn.poll():
            data = parent_conn.recv()
            if not data['found']: continue

            # =========================================
            # PHASE A: CALIBRATION
            # =========================================
            if not is_calibrated:
                cv2.imshow("Calib", draw_target(calib_idx))
                key = cv2.waitKey(1)
                
                if key == ord('q'): break
                if key == ord('c'):
                    # Save the current head position
                    calib_data.append([data['x'], data['y']])
                    calib_idx += 1
                    print(f"Captured Point {calib_idx}/9")
                    
                    if calib_idx >= 9:
                        # 1. Extract X and Y lists
                        xs = [p[0] for p in calib_data]
                        ys = [p[1] for p in calib_data]
                        
                        # 2. Find the extremes (Min/Max)
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        
                        # 3. Add Comfort Padding (10%)
                        # This ensures you don't have to strain your neck to hit the screen edge
                        pad_x = (x_max - x_min) * 0.1
                        pad_y = (y_max - y_min) * 0.1
                        
                        X_BOUNDS = [x_min - pad_x, x_max + pad_x]
                        Y_BOUNDS = [y_min - pad_y, y_max + pad_y]
                        
                        is_calibrated = True
                        cv2.destroyWindow("Calib")

            # =========================================
            # PHASE B: MOUSE CONTROL
            # =========================================
            else:
                # 1. MAP COORDINATES
                # Transform Camera (0.0 to 1.0) -> Screen (0 to 1920)
                # using the bounds we found during calibration.
                raw_x = np.interp(data['x'], X_BOUNDS, [0, SCREEN_WIDTH])
                raw_y = np.interp(data['y'], Y_BOUNDS, [0, SCREEN_HEIGHT])

                # 2. STABILIZE
                # Apply the "Parking Brake" and Momentum math
                sx, sy = stabilizer.update(raw_x, raw_y)

                # 3. CLICK PINNING
                # Logic: If the user is pinching (clicking), FREEZE the mouse.
                # This prevents the cursor from slipping when your hand muscles tense up.
                is_clicking = data['click_l'] or data['click_r']
                if not is_clicking:
                    mouse.position = (sx, sy)

                # 4. EXECUTE CLICKS
                # Left Click
                if data['click_l'] and not l_held:
                    mouse.press(Button.left); l_held = True
                elif not data['click_l'] and l_held:
                    mouse.release(Button.left); l_held = False

                # Right Click
                if data['click_r'] and not r_held:
                    mouse.press(Button.right); r_held = True
                elif not data['click_r'] and r_held:
                    mouse.release(Button.right); r_held = False

        else:
            # Sleep briefly to save CPU if no new data is available
            time.sleep(0.001)

if __name__ == '__main__':
    main()