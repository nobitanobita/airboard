"""
AirBoard Book - Multi-page Virtual Board
Works like a book with multiple pages
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import threading
import time
import os
from datetime import datetime

# ================ PARAMETERS ================
REQUEST_W = 1280
REQUEST_H = 720

SMOOTHING = 0.6
MOVE_THRESH_SQ = 15

COLORS = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (255,255,255)]
color_index = 0
DRAW_COLOR = COLORS[color_index]
DRAW_THICKNESS = 5

ERASER_SIZES = [20, 40, 70]
eraser_index = 1
ERASER_SIZE = ERASER_SIZES[eraser_index]

# Cooldowns (frames)
COLOR_CD = 15
CLEAR_CD = 30
ERASE_CD = 20
UNDO_CD = 20
SAVE_CD = 30
PAGE_CD = 20

# ================ PAGE SYSTEM ================
class Page:
    def __init__(self, width, height, page_number):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.strokes = []
        self.page_number = page_number
        self.name = f"Page {page_number}"

    def clear(self):
        self.canvas.fill(0)
        self.strokes.clear()

    def add_stroke(self, stroke):
        self.strokes.append(stroke)

    def draw_stroke(self, stroke):
        """Draw a stroke on the canvas"""
        points = stroke['points']
        color = stroke['color']
        thickness = stroke['thickness']

        for i in range(1, len(points)):
            cv2.line(self.canvas, points[i-1], points[i], color, thickness)

# ================ THREADED CAMERA ================
class ThreadedCamera:
    def __init__(self, src=0, req_w=REQUEST_W, req_h=REQUEST_H):
        self.cap = cv2.VideoCapture(src)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(req_w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(req_h))
        except Exception:
            pass
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame.copy() if frame is not None else None

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

# ================ HELPER FUNCTIONS ================
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def fingers_up(lm, handed):
    # Returns [thumb, index, middle, ring, pinky]
    fingers = []
    # Thumb direction depends on handedness
    if handed == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)
    # Other fingers: tip.y < pip.y => up
    for t in [8, 12, 16, 20]:
        fingers.append(lm[t].y < lm[t-2].y)
    return fingers

# ================ MEDIAPIPE SETUP ================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================ INITIALIZATION ================
cam = ThreadedCamera(0, REQUEST_W, REQUEST_H)

# Wait for camera
print("Waiting for camera...")
frame = None
for _ in range(100):
    ret, frame = cam.read()
    if ret and frame is not None:
        break
    time.sleep(0.01)

if frame is None:
    cam.release()
    raise RuntimeError("Camera not available")

frame = cv2.flip(frame, 1)
H, W = frame.shape[:2]
print(f"Camera resolution: {W}x{H}")

# Create page system
pages = []
current_page_index = 0

# Create first page
def create_new_page():
    page_number = len(pages) + 1
    new_page = Page(W, H, page_number)
    pages.append(new_page)
    return new_page

# Initialize with first page
create_new_page()

def get_current_page():
    return pages[current_page_index] if pages else None

# Drawing state
current_stroke = None

# Smoothing histories
right_hist = deque(maxlen=4)
left_hist = deque(maxlen=4)
right_smooth = None
left_smooth = None

# Cooldowns
color_cd = 0
clear_cd = 0
erase_cd = 0
undo_cd = 0
save_cd = 0
page_cd = 0

# Timing
last_time = time.time()
fps = 0

print("AirBoard Book Started!")
print("RIGHT HAND:")
print("  👆 Index finger up - DRAW")
print("  👆✌️ Index+Middle up - ERASE")
print("  👆🖐️ Index+Pinky up - CHANGE COLOR")
print("")
print("LEFT HAND:")
print("  ✊ Fist (no fingers) - CHANGE ERASER SIZE")
print("  👍👆 Thumb+Index - UNDO")
print("  👍 Thumb only - SAVE")
print("  ✌️✌️ Peace sign - CLEAR PAGE")
print("")
print("PAGE NAVIGATION:")
print("  👉 Right hand swipe right - NEXT PAGE")
print("  👈 Right hand swipe left - PREVIOUS PAGE")
print("  👆👇 Right hand swipe up - NEW PAGE")

# ================ PAGE NAVIGATION GESTURES ================
def detect_swipe_gesture(history, current_pos):
    """Detect swipe gestures for page navigation"""
    if len(history) < 3:
        return None

    # Get recent positions
    recent_points = list(history)[-3:]
    current_x, current_y = current_pos

    # Calculate movement direction
    start_x, start_y = recent_points[0]
    dx = current_x - start_x
    dy = current_y - start_y

    # Swipe thresholds
    swipe_threshold = 100  # pixels
    diagonal_threshold = 50  # pixels

    # Check for significant movement
    if abs(dx) > swipe_threshold or abs(dy) > swipe_threshold:
        if abs(dx) > abs(dy):  # Horizontal swipe
            if dx > swipe_threshold:
                return "swipe_right"
            elif dx < -swipe_threshold:
                return "swipe_left"
        else:  # Vertical swipe
            if dy > swipe_threshold:
                return "swipe_down"
            elif dy < -swipe_threshold:
                return "swipe_up"

    return None

def next_page():
    global current_page_index
    if current_page_index < len(pages) - 1:
        current_page_index += 1
        return True
    return False

def previous_page():
    global current_page_index
    if current_page_index > 0:
        current_page_index -= 1
        return True
    return False

def add_new_page():
    global current_page_index
    new_page = create_new_page()
    current_page_index = len(pages) - 1
    return new_page

# ================ MAIN LOOP ================
try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Ensure correct size
        fh, fw = frame.shape[:2]
        if fh != H or fw != W:
            frame = cv2.resize(frame, (W, H))

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Reset flags for this frame
        right_draw = right_erase = right_color = False
        left_clear = left_erase_cycle = left_undo = left_save = False
        swipe_gesture = None

        # Process detected hands
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm_set, hand_h in zip(results.multi_hand_landmarks, results.multi_handedness):
                handed = hand_h.classification[0].label  # 'Left' or 'Right'
                lm = lm_set.landmark

                # Get index tip and wrist positions
                ix, iy = lm_to_px(lm[8], W, H)
                wx, wy = lm_to_px(lm[0], W, H)

                # Skip invalid points
                if not (0 <= ix < W and 0 <= iy < H):
                    continue

                # Update smoothing histories
                if handed == "Right":
                    right_hist.append((ix, iy))
                    if right_hist:
                        avgx = sum(p[0] for p in right_hist) / len(right_hist)
                        avgy = sum(p[1] for p in right_hist) / len(right_hist)
                        right_smooth = (int(avgx), int(avgy))

                    # Detect swipe gestures for page navigation
                    if right_smooth and len(right_hist) >= 3:
                        swipe_gesture = detect_swipe_gesture(right_hist, right_smooth)

                else:
                    left_hist.append((ix, iy))
                    if left_hist:
                        avgx = sum(p[0] for p in left_hist) / len(left_hist)
                        avgy = sum(p[1] for p in left_hist) / len(left_hist)
                        left_smooth = (int(avgx), int(avgy))

                # Detect finger states
                thumb, idx, mid, ring, pink = fingers_up(lm, handed)

                # Draw hand landmarks for visual feedback
                mp_drawing.draw_landmarks(frame, lm_set, mp_hands.HAND_CONNECTIONS)

                # RIGHT HAND GESTURES
                if handed == "Right":
                    # Distance from palm to ensure finger is extended
                    palm_dist = ((ix - wx)**2 + (iy - wy)**2)**0.5

                    # Draw gesture: Index up, others down
                    if idx and not mid and not ring and not pink and palm_dist > 40:
                        right_draw = True

                    # Erase gesture: Index + Middle up, others down
                    if idx and mid and not ring and not pink:
                        right_erase = True

                    # Color change gesture: Index + Pinky up, others down
                    if idx and pink and not mid and not ring:
                        right_color = True

                # LEFT HAND GESTURES
                else:
                    # Clear gesture: Peace sign (Index + Middle up)
                    if idx and mid and not thumb and not ring and not pink:
                        left_clear = True

                    # Eraser cycle gesture: Fist (no fingers up)
                    if not thumb and not idx and not mid and not ring and not pink:
                        left_erase_cycle = True

                    # Undo gesture: Thumb + Index up
                    if thumb and idx and not mid and not ring and not pink:
                        left_undo = True

                    # Save gesture: Thumb only up
                    if thumb and not idx and not mid and not ring and not pink:
                        left_save = True

        # ================ PROCESS ACTIONS ================
        # Update cooldowns
        color_cd = max(0, color_cd - 1)
        clear_cd = max(0, clear_cd - 1)
        erase_cd = max(0, erase_cd - 1)
        undo_cd = max(0, undo_cd - 1)
        save_cd = max(0, save_cd - 1)
        page_cd = max(0, page_cd - 1)

        # Get current page
        current_page = get_current_page()

        # PAGE NAVIGATION
        if swipe_gesture and page_cd == 0:
            if swipe_gesture == "swipe_right":
                if next_page():
                    print(f"Next page: {current_page_index + 1}/{len(pages)}")
                    page_cd = PAGE_CD
            elif swipe_gesture == "swipe_left":
                if previous_page():
                    print(f"Previous page: {current_page_index + 1}/{len(pages)}")
                    page_cd = PAGE_CD
            elif swipe_gesture == "swipe_up":
                add_new_page()
                print(f"New page created: {current_page_index + 1}/{len(pages)}")
                page_cd = PAGE_CD

        # RIGHT HAND ACTIONS
        # Color change
        if right_color and color_cd == 0:
            color_index = (color_index + 1) % len(COLORS)
            DRAW_COLOR = COLORS[color_index]
            color_cd = COLOR_CD
            print(f"Color changed to: {DRAW_COLOR}")

        # Drawing
        if right_draw and right_smooth and current_page:
            sx, sy = right_smooth
            if current_stroke is None:
                current_stroke = {'color': DRAW_COLOR, 'thickness': DRAW_THICKNESS, 'points': []}

            points = current_stroke['points']
            if not points:
                points.append((sx, sy))
            else:
                # Check movement threshold
                lx, ly = points[-1]
                dx = sx - lx
                dy = sy - ly
                if (dx*dx + dy*dy) >= MOVE_THRESH_SQ:
                    points.append((sx, sy))
                    cv2.line(current_page.canvas, points[-2], points[-1], current_stroke['color'], current_stroke['thickness'])
        else:
            # Finish current stroke
            if current_stroke is not None and current_stroke.get('points') and current_page:
                current_page.add_stroke(current_stroke)
                current_stroke = None

        # Erasing
        if right_erase and right_smooth and current_page:
            cv2.circle(current_page.canvas, right_smooth, ERASER_SIZE, (0, 0, 0), -1)

        # LEFT HAND ACTIONS
        # Clear current page
        if left_clear and clear_cd == 0 and current_page:
            current_page.clear()
            current_stroke = None
            clear_cd = CLEAR_CD
            print(f"Page {current_page_index + 1} cleared!")

        # Cycle eraser size
        if left_erase_cycle and erase_cd == 0:
            eraser_index = (eraser_index + 1) % len(ERASER_SIZES)
            ERASER_SIZE = ERASER_SIZES[eraser_index]
            erase_cd = ERASE_CD
            print(f"Eraser size: {ERASER_SIZE}")

        # Undo last stroke on current page
        if left_undo and undo_cd == 0 and current_page:
            if current_page.strokes:
                current_page.strokes.pop()
                # Redraw everything except last stroke
                current_page.canvas.fill(0)
                for stroke in current_page.strokes:
                    current_page.draw_stroke(stroke)
                undo_cd = UNDO_CD
                print(f"Undo! {len(current_page.strokes)} strokes on page {current_page_index + 1}")

        # Save all pages
        if left_save and save_cd == 0:
            ensure_dir("captures")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save current page
            current_filename = f"captures/airboard_page_{current_page_index + 1}_{timestamp}.png"
            mask = cv2.cvtColor(current_page.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            fg = cv2.bitwise_and(current_page.canvas, current_page.canvas, mask=mask)
            final = cv2.add(bg, fg)

            cv2.imwrite(current_filename, final)
            save_cd = SAVE_CD
            print(f"Saved page {current_page_index + 1}: {current_filename}")

        # ================ DISPLAY ================
        # Create composite output using current page
        if current_page:
            mask = cv2.cvtColor(current_page.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            fg = cv2.bitwise_and(current_page.canvas, current_page.canvas, mask=mask)
            out = cv2.add(bg, fg)
        else:
            out = frame.copy()

        # Display FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - last_time + 1e-9))
        last_time = now
        cv2.putText(out, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display page info
        page_info = f"Page: {current_page_index + 1}/{len(pages)}"
        cv2.putText(out, page_info, (W - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display status info
        cv2.putText(out, f"Color: {DRAW_COLOR}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAW_COLOR, 2)
        cv2.putText(out, f"Eraser: {ERASER_SIZE}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if current_page:
            cv2.putText(out, f"Strokes: {len(current_page.strokes)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show active gestures
        if right_draw:
            cv2.putText(out, "DRAWING", (W-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if right_erase:
            cv2.putText(out, "ERASING", (W-150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if left_undo:
            cv2.putText(out, "UNDO", (W-150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if swipe_gesture:
            cv2.putText(out, f"SWIPE: {swipe_gesture.upper()}", (W-200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # Draw page corner effect
        cv2.rectangle(out, (W-80, 10), (W-10, 80), (50, 50, 50), -1)
        cv2.putText(out, str(current_page_index + 1), (W-50, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("AirBoard Book - Multi-page Virtual Board", out)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and current_page:
            current_page.clear()
            current_stroke = None
            print(f"Page {current_page_index + 1} cleared (keyboard)")
        elif key == ord('u') and current_page:
            if current_page.strokes:
                current_page.strokes.pop()
                current_page.canvas.fill(0)
                for stroke in current_page.strokes:
                    current_page.draw_stroke(stroke)
                print(f"Undo (keyboard)! {len(current_page.strokes)} strokes on page {current_page_index + 1}")
        elif key == ord('n'):  # New page
            add_new_page()
            print(f"New page created (keyboard): {current_page_index + 1}/{len(pages)}")
        elif key == ord(']'):  # Next page
            if next_page():
                print(f"Next page (keyboard): {current_page_index + 1}/{len(pages)}")
        elif key == ord('['):  # Previous page
            if previous_page():
                print(f"Previous page (keyboard): {current_page_index + 1}/{len(pages)}")

finally:
    cam.release()
    cv2.destroyAllWindows()
    hands.close()
    print("AirBoard Book closed successfully!")