# src/gesture_oak/apps/swipe_detection_app.py
from __future__ import annotations
import cv2
import time
from typing import Optional

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

from gesture_oak.detection.swipe_detector import SwipeDetector
from gesture_oak.detection.hand_detector import HandDetector  # your existing pipeline

PRESETS = {
    "strict": dict(min_confidence=0.75, min_distance_px=120, max_y_deviation_px=36,
                   max_duration_ms=600,  min_consistent_dir_ratio=0.85, min_avg_vx_px_per_ms=0.70),
    "normal": dict(min_confidence=0.65, min_distance_px=100, max_y_deviation_px=45,
                   max_duration_ms=650,  min_consistent_dir_ratio=0.80, min_avg_vx_px_per_ms=0.55),
    "loose":  dict(min_confidence=0.55, min_distance_px=80,  max_y_deviation_px=60,
                   max_duration_ms=750,  min_consistent_dir_ratio=0.72, min_avg_vx_px_per_ms=0.45),
}

def _now_ms():
    return int(time.time() * 1000)

def draw_overlays(frame, hands, fps: float, swipe_count: int, mode_label: str):
    hud = f"FPS:{fps:.1f}  Swipes:{swipe_count}  Mode:{mode_label}"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for hand in hands:
        if hasattr(hand, "center"):
            cx, cy = map(int, hand.center)
            cv2.circle(frame, (cx, cy), 4, (0, 200, 255), 2)

def run():
    print("\n--- Swipe Detection Application ---")
    print("OAK-D Swipe Detection Focused Demo")
    print("========================================")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Reset statistics")
    print("  '1' - Strict mode (高精度)")
    print("  '2' - Normal mode (標準)")
    print("  '3' - Loose mode (検出しやすい)\n")

    det = HandDetector()
    det.start()

    detector = SwipeDetector()
    detector.set_params(**PRESETS["normal"])
    mode = "Normal"

    swipe_count = 0
    frame_id = 0
    draw_every = 2
    t_last = time.time()
    fps = 0.0

    try:
        while True:
            frame, hands = det.get_frame_and_hands()
            if frame is None:
                continue

            frame_id += 1
            if frame_id % 10 == 0:
                t_now = time.time()
                fps = 10.0 / max(1e-6, (t_now - t_last))
                t_last = t_now

            ts = _now_ms()
            for hand in hands or []:
                if getattr(hand, "center", None) is None:
                    continue
                if detector.update(
                    hand.center,
                    ts_ms=ts,
                    hand_conf=float(getattr(hand, "confidence", 1.0)),
                    hand_depth_m=getattr(hand, "depth_m", None),
                ):
                    swipe_count += 1
                    cv2.putText(frame, "SWIPE!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

            if (frame_id % draw_every) == 0:
                draw_overlays(frame, hands or [], fps, swipe_count, mode)

            cv2.imshow("OAK-D Swipe Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"swipe_{int(time.time())}.png", frame)
            elif key == ord('r'):
                swipe_count = 0
            elif key == ord('1'):
                mode = "Strict"
                detector.set_params(**PRESETS["strict"])
            elif key == ord('2'):
                mode = "Normal"
                detector.set_params(**PRESETS["normal"])
            elif key == ord('3'):
                mode = "Loose"
                detector.set_params(**PRESETS["loose"])

    finally:
        det.stop()
        cv2.destroyAllWindows()
        print("\nSwipe detection demo completed.")
