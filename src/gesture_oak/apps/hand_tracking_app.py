# src/gesture_oak/apps/hand_tracking_app.py
from __future__ import annotations
import cv2
import time
from typing import Optional, Tuple

# Perf toggles
cv2.setUseOptimized(True)
cv2.setNumThreads(0)

from gesture_oak.detection.swipe_detector import SwipeDetector
# Your existing HandDetector import:
from gesture_oak.detection.hand_detector import HandDetector  # keep your working pipeline

def _now_ms():
    return int(time.time() * 1000)

def draw_landmarks_and_hud(frame, hands, fps: float, swipe_count: int, mode_label: str = "Normal"):
    # Keep drawing light: thinner lines, minimal text
    h, w = frame.shape[:2]
    hud = f"FPS:{fps:.1f}  Swipes:{swipe_count}  Mode:{mode_label}"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # If you already draw landmarks elsewhere, keep it; else draw minimal markers
    for hand in hands:
        if hasattr(hand, "center"):
            cx, cy = map(int, hand.center)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), 2)

def run():
    print("\n--- Hand Tracking Application ---")
    print("OAK-D Hand Detection Demo with Swipe Detection")
    print("=============================================")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'r' to reset swipe statistics\n")

    # Hand detector (uses your existing DepthAI pipeline)
    det = HandDetector()
    det.start()  # your impl should build the pipeline and queues

    # Swipe detector (Normal defaults)
    swipe = SwipeDetector()

    swipe_count = 0
    frame_id = 0
    draw_every = 2  # render HUD every 2nd frame to save CPU
    t_last = time.time()
    fps = 0.0

    try:
        while True:
            # Your detector should return (frame, hands) where each hand has .center, .confidence, and optional .depth_m
            frame, hands = det.get_frame_and_hands()
            if frame is None:
                # device might be temporarily empty; continue
                continue

            # FPS (cheap moving estimate)
            frame_id += 1
            if frame_id % 10 == 0:
                t_now = time.time()
                fps = 10.0 / max(1e-6, (t_now - t_last))
                t_last = t_now

            # Feed hands to swipe detector
            ts = _now_ms()
            for hand in hands or []:
                center = getattr(hand, "center", None)
                if center is None:
                    continue
                conf = float(getattr(hand, "confidence", 1.0))
                depth_m = getattr(hand, "depth_m", None)
                if swipe.update(center, ts_ms=ts, hand_conf=conf, hand_depth_m=depth_m):
                    swipe_count += 1
                    # On-screen banner (brief)
                    cv2.putText(frame, "SWIPE!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw (lighter cadence)
            if (frame_id % draw_every) == 0:
                draw_landmarks_and_hud(frame, hands or [], fps, swipe_count, "Normal")

            cv2.imshow("OAK-D Hand Tracking + Swipe", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"frame_{int(time.time())}.png", frame)
            elif key == ord('r'):
                swipe_count = 0

    finally:
        det.stop()
        cv2.destroyAllWindows()
        print("\nHand detection demo completed.")
