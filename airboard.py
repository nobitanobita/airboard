"""
AirBoard Phase 1
- Requirements: Python 3.8+ recommended
- Run: python airboard.py
- Keys:
    c -> clear canvas
    q -> quit
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import math

# ---------- Config ----------
CAM_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DRAW_COLOR = (0, 0, 255)       # BGR (red)
DRAW_THICKNESS = 5
SMOOTHING = 0.7                # 0..1, larger -> smoother (less jitter)

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,            # Phase 1: single-hand
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------- Helper functions ----------
TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

def fingers_up(landmarks, handedness_str="Right"):
    """
    Return list [thumb, index, middle, ring, pinky] -> True if that finger is considered 'up'.
    landmarks: list of 21 landmarks where each has .x and .y normalized (0..1)
    handedness_str: "Right" or "Left" (MediaPipe handedness). Thumb logic depends on hand.
    """
    fingers = []

    # Thumb: compare tip (4) x to ip (3) or mcp (2) depending on hand orientation
    if handedness_str == "Right":
        fingers.append(landmarks[TIP_IDS[0]].x < landmarks[TIP_IDS[0]-1].x)  # tip.x < previous.x -> thumb open (to the left)
    else:
        fingers.append(landmarks[TIP_IDS[0]].x > landmarks[TIP_IDS[0]-1].x)

    # Other fingers: tip.y < pip.y means finger is up (y=0 top of image)
    for id in [8, 12, 16, 20]:
        tip_y = landmarks[id].y
        pip_y = landmarks[id - 2].y
        fingers.append(tip_y < pip_y)

    return fingers  # list of 5 booleans

def landmark_to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

# ---------- Drawing state ----------
canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
prev_point = None
smoothed_point = None

# We'll keep a short history for more stable smoothing (optional)
point_history = deque(maxlen=5)

# ---------- Camera ----------
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check CAM_ID or webcam permissions.")

last_time = time.time()
fps = 0

# ---------- Main loop ----------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_present = False
        index_up = False
        fist = False
        handedness_str = "Right"

        if results.multi_hand_landmarks:
            hand_present = True
            hand_landmarks = results.multi_hand_landmarks[0]
            # Identify handedness if available
            if results.multi_handedness:
                handedness_str = results.multi_handedness[0].classification[0].label

            # Determine fingers up
            fingers = fingers_up(hand_landmarks.landmark, handedness_str)
            # fingers -> [thumb, index, middle, ring, pinky]
            index_up = (fingers[1] and not fingers[2] and not fingers[3] and not fingers[4])
            # simple fist detection: all fingers down (except maybe thumb)
            all_down = (not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4])
            fist = all_down

            # Get index fingertip pixel coords
            ix, iy = landmark_to_pixel(hand_landmarks.landmark[8], FRAME_WIDTH, FRAME_HEIGHT)

            # Smooth point
            point_history.append((ix, iy))
            avg_x = sum([p[0] for p in point_history]) / len(point_history)
            avg_y = sum([p[1] for p in point_history]) / len(point_history)
            smoothed = (int(avg_x * (1 - SMOOTHING) + (smoothed_point[0] if smoothed_point else avg_x) * SMOOTHING),
                        int(avg_y * (1 - SMOOTHING) + (smoothed_point[1] if smoothed_point else avg_y) * SMOOTHING)) if smoothed_point else (int(avg_x), int(avg_y))
            smoothed_point = smoothed

            # Draw if index_up
            if index_up:
                if prev_point is None:
                    prev_point = smoothed_point
                # Draw line on canvas from prev to current
                cv2.line(canvas, prev_point, smoothed_point, DRAW_COLOR, DRAW_THICKNESS)
                prev_point = smoothed_point
            else:
                prev_point = None

            # Draw landmarks (optional) for debugging / visual feedback
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Visual hint: show a small circle at fingertip
            cv2.circle(frame, smoothed_point, 8, (0, 255, 0) if index_up else (0, 120, 255), -1)

        else:
            prev_point = None
            smoothed_point = None
            point_history.clear()

        # If fist -> act as eraser (here we erase last strokes by fading canvas)
        if hand_present and fist:
            # simple erase: fade the canvas gradually
            canvas = cv2.addWeighted(canvas, 0.7, np.zeros_like(canvas), 0.3, 0)
            # optional: small text feedback
            cv2.putText(frame, "ERASING...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Merge canvas onto frame (overlay)
        overlay = frame.copy()
        alpha = 0.8
        # Convert canvas to grayscale mask where non-black pixels are drawn
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Combine: keep frame where mask is zero, then add canvas
        bg = cv2.bitwise_and(overlay, overlay, mask=mask_inv)
        fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(bg, fg)

        # Draw HUD: mode and FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - last_time)) if last_time else 0
        last_time = now
        cv2.putText(combined, f"AirBoard - Draw mode: Index finger up", (10, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, f"FPS: {int(fps)}", (FRAME_WIDTH - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show
        cv2.imshow("AirBoard", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
