"""
AirBoard — Fixed auto-size version
- Detects camera resolution at runtime and matches canvas & masks
- Two-hand gestures (stable rules)
- Threaded camera for smooth FPS
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import threading
import time
import os
from datetime import datetime

# ---------------- PARAMETERS (tweak if needed) ----------------
# Requested resolution (camera may ignore); actual resolution will be detected
REQUEST_W = 960
REQUEST_H = 540

SMOOTHING = 0.55
MOVE_THRESH_SQ = 12  # squared pixels movement threshold

COLORS = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,255,255)]
color_index = 0
DRAW_COLOR = COLORS[color_index]
DRAW_THICKNESS = 5

ERASER_SIZES = [20, 40, 70]
eraser_index = 1
ERASER_SIZE = ERASER_SIZES[eraser_index]

# cooldowns (frames)
COLOR_CD = 12
LEFT_CLEAR_CD = 25
LEFT_ERASE_CD = 20
LEFT_UNDO_CD = 20
LEFT_SAVE_CD = 25

# ---------------- Threaded camera class ----------------
class ThreadedCamera:
    def __init__(self, src=0, req_w=REQUEST_W, req_h=REQUEST_H):
        self.cap = cv2.VideoCapture(src)
        # try set requested size (camera may ignore)
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
                # store a copy to avoid race conditions
                self.frame = frame.copy() if frame is not None else None

    def read(self):
        with self.lock:
            # return shallow copy to caller
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

# ---------------- Helpers ----------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def fingers_up(lm, handed):
    # returns [thumb, index, middle, ring, pinky]
    fingers = []
    # thumb direction depends on handedness
    if handed == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)
    # other fingers: tip.y < pip.y => up
    for t in [8, 12, 16, 20]:
        fingers.append(lm[t].y < lm[t-2].y)
    return fingers

# ---------------- MediaPipe ----------------
mp_h = mp.solutions.hands
mp_d = mp.solutions.drawing_utils

hands = mp_h.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- Initialization ----------------
cam = ThreadedCamera(0, REQUEST_W, REQUEST_H)

# wait for first valid frame and detect actual size
print("Waiting for camera frame...")
frame = None
for _ in range(200):  # ~2 seconds max
    ret, frame = cam.read()
    if ret and frame is not None:
        break
    time.sleep(0.01)

if frame is None:
    cam.release()
    raise RuntimeError("Unable to read from camera. Check camera index and permissions.")

# flip for natural interaction
frame = cv2.flip(frame, 1)
H, W = frame.shape[:2]
print(f"Detected camera resolution: width={W}, height={H}")

# create canvas & masks sized to actual camera resolution
CANVAS_H = H
CANVAS_W = W
canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
strokes = []
current_stroke = None

# smoothing histories
right_hist = deque(maxlen=4)
left_hist = deque(maxlen=4)
right_smooth = None
left_smooth = None

# cooldowns
color_cd = 0
left_clear_cd = 0
left_erase_cd = 0
left_undo_cd = 0
left_save_cd = 0

navigation_mode = False

# pre-allocated grayscale temp for mask ops
_gray_tmp = np.empty((CANVAS_H, CANVAS_W), dtype=np.uint8)

# timing
last_time = time.time()
fps = 0

print("Starting main loop. Press 'q' to quit, 'c' to clear canvas.")

# ---------------- Main Loop ----------------
try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        # flip horizontally to act like mirror
        frame = cv2.flip(frame, 1)

        # ensure frame matches detected size (some cameras change mid-run)
        fh, fw = frame.shape[:2]
        if fh != CANVAS_H or fw != CANVAS_W:
            frame = cv2.resize(frame, (CANVAS_W, CANVAS_H))

        # convert to RGB for MediaPipe (copy to be safe with threaded buffer)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        results = hands.process(rgb)

        # reset flags for this frame
        right_draw = right_erase = right_color = False
        left_clear = left_erase_cycle = left_undo = left_save = left_peace = False

        # iterate detected hands, pair with handedness
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm_set, hand_h in zip(results.multi_hand_landmarks, results.multi_handedness):
                handed = hand_h.classification[0].label  # 'Left' or 'Right'
                lm = lm_set.landmark

                # fingertip and wrist positions
                ix, iy = lm_to_px(lm[8], CANVAS_W, CANVAS_H)
                wx, wy = lm_to_px(lm[0], CANVAS_W, CANVAS_H)

                # skip invalid points
                if not (0 <= ix < CANVAS_W and 0 <= iy < CANVAS_H):
                    continue

                # update smoothing histories
                if handed == "Right":
                    right_hist.append((ix, iy))
                    avgx = sum(p[0] for p in right_hist) / len(right_hist)
                    avgy = sum(p[1] for p in right_hist) / len(right_hist)
                    right_smooth = (int(avgx), int(avgy))
                else:
                    left_hist.append((ix, iy))
                    avgx = sum(p[0] for p in left_hist) / len(left_hist)
                    avgy = sum(p[1] for p in left_hist) / len(left_hist)
                    left_smooth = (int(avgx), int(avgy))

                # finger states
                thumb, idx, mid, ring, pink = fingers_up(lm, handed)

                # draw landmarks for visual feedback (low-cost)
                mp_d.draw_landmarks(frame, lm_set, mp_h.HAND_CONNECTIONS)

                if handed == "Right":
                    # distance from palm/wrist to ensure finger extended
                    palm_dist = ((ix - wx)**2 + (iy - wy)**2)**0.5

                    # robust draw pose: index up, others down, finger extended
                    pose_ok = idx and (not mid) and (not ring) and (not pink) and (palm_dist > 40)

                    # movement threshold vs last point of current stroke
                    move_ok = True
                    if current_stroke and current_stroke.get("points"):
                        lx, ly = current_stroke["points"][-1]
                        dx = ix - lx
                        dy = iy - ly
                        move_ok = (dx*dx + dy*dy) >= MOVE_THRESH_SQ

                    right_draw = pose_ok and move_ok
                    right_erase = idx and mid and (not ring) and (not pink)
                    right_color = idx and pink and (not mid) and (not ring)

                else:
                    # left-hand control gestures
                    if thumb and idx and mid and ring and pink:
                        left_clear = True
                    if (not idx and not mid and not ring and not pink):
                        left_erase_cycle = True
                    if thumb and idx and (not mid) and (not ring) and (not pink):
                        left_undo = True
                    if thumb and (not idx) and (not mid) and (not ring) and (not pink):
                        left_save = True
                    if idx and mid and (not ring) and (not pink):
                        left_peace = True

        # ---------------- RIGHT HAND ACTIONS ----------------
        # color change with cooldown
        if right_color and color_cd == 0:
            color_index = (color_index + 1) % len(COLORS)
            DRAW_COLOR = COLORS[color_index]
            color_cd = COLOR_CD

        # add points to current stroke and draw incremental segments
        if right_draw and right_smooth:
            sx, sy = right_smooth
            if current_stroke is None:
                current_stroke = {'color': DRAW_COLOR, 'thickness': DRAW_THICKNESS, 'points': []}
            pts = current_stroke['points']
            if not pts:
                pts.append((sx, sy))
            else:
                # movement already checked; append and draw
                pts.append((sx, sy))
                cv2.line(canvas, pts[-2], pts[-1], current_stroke['color'], current_stroke['thickness'])
        else:
            # finish stroke (store history) if any
            if current_stroke is not None and current_stroke.get('points'):
                strokes.append(current_stroke)
            current_stroke = None

        # erase using right_smooth
        if right_erase and right_smooth:
            cv2.circle(canvas, right_smooth, ERASER_SIZE, (0,0,0), -1)

        # ---------------- LEFT HAND ACTIONS ----------------
        if left_clear and left_clear_cd == 0:
            canvas.fill(0)
            strokes.clear()
            current_stroke = None
            left_clear_cd = LEFT_CLEAR_CD

        if left_erase_cycle and left_erase_cd == 0:
            eraser_index = (eraser_index + 1) % len(ERASER_SIZES)
            ERASER_SIZE = ERASER_SIZES[eraser_index]
            left_erase_cd = LEFT_ERASE_CD

        if left_undo and left_undo_cd == 0:
            if strokes:
                strokes.pop()
                # redraw canvas from strokes
                canvas.fill(0)
                for s in strokes:
                    pts = s['points']
                    for i in range(1, len(pts)):
                        cv2.line(canvas, pts[i-1], pts[i], s['color'], s['thickness'])
            left_undo_cd = LEFT_UNDO_CD

        if left_save and left_save_cd == 0:
            ensure_dir("captures")
            fname = datetime.now().strftime("airboard_%Y%m%d_%H%M%S.png")
            path = os.path.join("captures", fname)

            # build mask safely (uint8)
            _gray_tmp[:] = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            _, mask = cv2.threshold(_gray_tmp, 10, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.uint8)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            fg = cv2.bitwise_and(canvas, canvas, mask=mask)
            final = cv2.add(bg, fg)
            cv2.imwrite(path, final)
            print("Saved:", path)
            left_save_cd = LEFT_SAVE_CD

        if left_peace:
            navigation_mode = not navigation_mode

        # cooldown decrements
        color_cd = max(0, color_cd - 1)
        left_clear_cd = max(0, left_clear_cd - 1)
        left_erase_cd = max(0, left_erase_cd - 1)
        left_undo_cd = max(0, left_undo_cd - 1)
        left_save_cd = max(0, left_save_cd - 1)

        # ---------------- Compose & Show ----------------
        # mask must be same size & dtype as canvas/frame
        _gray_tmp[:] = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        _, mask = cv2.threshold(_gray_tmp, 10, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        out = cv2.add(bg, fg)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - last_time + 1e-9))
        last_time = now

        cv2.putText(out, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(out, f"Color: {DRAW_COLOR}  Eraser: {ERASER_SIZE}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAW_COLOR, 2)
        cv2.imshow("AirBoard (fixed autosize)", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas.fill(0)
            strokes.clear()
            current_stroke = None

finally:
    cam.release()
    cv2.destroyAllWindows()
    hands.close()
