# src/gesture_oak/apps/hand_tracking_app.py
from __future__ import annotations
import cv2
import time
from typing import Any, Iterable, Optional, Tuple

# Perf toggles
cv2.setUseOptimized(True)
cv2.setNumThreads(0)

from gesture_oak.detection.swipe_detector import SwipeDetector
from gesture_oak.detection.hand_detector import HandDetector  # your existing detector

def _now_ms() -> int:
    return int(time.time() * 1000)

def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _extract_frame_hands(ret: Any) -> Tuple[Optional[Any], Optional[Iterable]]:
    """
    Accept many possible return shapes from HandDetector:
      - (frame, hands)
      - (frame, hands, depth)
      - {"frame": frame, "hands": hands, ...}
      - {"rgb": frame, "hands": hands, ...}
      - object with .frame/.rgb/.image and .hands
      - None -> (None, None)
    """
    if ret is None:
        return None, None

    if isinstance(ret, (tuple, list)):
        if len(ret) >= 2:
            return ret[0], ret[1]
        elif len(ret) == 1:
            return ret[0], None
        else:
            return None, None

    if isinstance(ret, dict):
        frame = _first_not_none(ret.get("frame"), ret.get("rgb"), ret.get("image"))
        hands = ret.get("hands")
        return frame, hands

    frame = None
    hands = None
    for name in ("frame", "rgb", "image"):
        if hasattr(ret, name):
            frame = getattr(ret, name)
            break
    if hasattr(ret, "hands"):
        hands = getattr(ret, "hands")

    return frame, hands

def draw_landmarks_and_hud(frame, hands, fps: float, swipe_count: int, mode_label: str = "Normal"):
    hud = f"FPS:{fps:.1f}  Swipes:{swipe_count}  Mode:{mode_label}"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if hands:
        for hand in hands:
            center = getattr(hand, "center", None)
            if center:
                cx, cy = map(int, center)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), 2)

def run():
    print("\n--- Hand Tracking Application ---")
    print("OAK-D Hand Detection Demo with Swipe Detection")
    print("=============================================")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'r' to reset swipe statistics\n")

    det = HandDetector()           # uses your working pipeline
    swipe = SwipeDetector()        # no buffer_size kwarg

    swipe_count = 0
    frame_id = 0
    draw_every = 2  # draw HUD every other frame to save CPU
    t_last = time.time()
    fps = 0.0

    try:
        while True:
            # Flexible call: try common method names
            ret = None
            try:
                if hasattr(det, "get_frame_and_hands"):
                    ret = det.get_frame_and_hands()
                elif hasattr(det, "get"):
                    ret = det.get()
                elif hasattr(det, "next"):
                    ret = det.next()
                elif hasattr(det, "read"):
                    ret = det.read()
                else:
                    ret = det()
            except Exception as e:
                # transient read error; keep UI responsive
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            frame, hands = _extract_frame_hands(ret)
            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            # FPS (safe)
            frame_id += 1
            if frame_id % 10 == 0:
                t_now = time.time()
                elapsed = t_now - t_last
                if elapsed > 1e-6:
                    fps = 10.0 / elapsed
                t_last = t_now

            # Feed hands to swipe detector
            ts = _now_ms()
            if hands:
                for hand in hands:
                    center = getattr(hand, "center", None)
                    if not center:
                        continue
                    conf = float(getattr(hand, "confidence", 1.0))
                    depth_m = getattr(hand, "depth_m", None)
                    if swipe.update(center, ts_ms=ts, hand_conf=conf, hand_depth_m=depth_m):
                        swipe_count += 1
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
                swipe.reset_stats()
                swipe_count = 0

    finally:
        cv2.destroyAllWindows()
        print("\nHand detection demo completed.")

def main():
    run()

if __name__ == "__main__":
    run()
