# src/gesture_oak/logic/gesture_classifier.py
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

# Landmark indices follow MediaPipe topology (0..20)
# Thumb: 1,2,3,4
# Index: 5,6,7,8
# Middle: 9,10,11,12
# Ring: 13,14,15,16
# Pinky: 17,18,19,20
FINGER_GROUPS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
ORDER = ["thumb", "index", "middle", "ring", "pinky"]

def _rotate2d(points: np.ndarray, theta: float) -> np.ndarray:
    """Rotate Nx2 points by theta (radians) around origin (wrist at 0,0)."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (R @ points.T).T

def _normalize_to_palm_frame(lms_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Translate wrist to (0,0), rotate so the line (index_mcp -> pinky_mcp) is horizontal.
    Returns rotated points and the rotation angle.
    """
    # Wrist=0, index_mcp=5, pinky_mcp=17
    pts = lms_xy.copy().astype(np.float32)
    wrist = pts[0]
    pts -= wrist
    base_v = pts[17] - pts[5]
    theta = math.atan2(base_v[1], base_v[0])  # rotate by -theta later
    rotated = _rotate2d(pts, -theta)
    return rotated, theta

def _finger_extended(rot: np.ndarray, finger: str, handedness: float) -> bool:
    """
    Extension test in palm frame (rotated).
    For non-thumbs: tip y << pip y (negative is 'up' because image y-down, but we rotated to make base horizontal).
    For thumb: use x spread and some y support, with handedness to decide direction.
    handedness > 0.5 means 'right', <=0.5 means 'left' (your detector's convention).
    """
    ids = FINGER_GROUPS[finger]
    mcp, pip_, dip, tip = rot[ids[0]], rot[ids[1]], rot[ids[2]], rot[ids[3]]

    # Scale-invariant thresholds using hand size ~ distance wrist->middle_mcp
    hand_scale = max(30.0, np.linalg.norm(rot[9]))  # middle MCP distance from wrist
    y_margin = 0.02 * hand_scale
    x_margin = 0.02 * hand_scale

    if finger == "thumb":
        # For right hand, thumb extends toward +x; left hand toward -x (in rotated frame).
        dir_sign = +1.0 if handedness > 0.5 else -1.0
        x_spread = (tip[0] - mcp[0]) * dir_sign
        y_support = abs(tip[1] - mcp[1])
        return (x_spread > 0.20 * hand_scale) and (y_support < 0.45 * hand_scale)
    else:
        # Classic: tip above PIP & above MCP by some margin (remember rotated frame)
        return (tip[1] < pip_[1] - y_margin) and (tip[1] < mcp[1] - y_margin) and (abs(tip[0] - mcp[0]) > x_margin)

def _count_fingers_up(rot: np.ndarray, handedness: float) -> Dict[str, bool]:
    return {name: _finger_extended(rot, name, handedness) for name in ORDER}

def _gesture_from_pattern(flags: Dict[str, bool]) -> str:
    # Map simple patterns to names
    up = [f for f, on in flags.items() if on]
    n = len(up)

    if n == 0:
        return "fist"
    if n == 5:
        return "open"

    # Single-finger names
    if n == 1:
        return f"one-{up[0]}"

    # Two fingers: check "peace" (index + middle)
    if n == 2 and "index" in up and "middle" in up and "ring" not in up and "pinky" not in up:
        return "peace"
    if n == 2:
        return f"two-{'-'.join(sorted(up))}"

    if n == 3:
        return "three"
    if n == 4:
        return "four"

    return f"{n}-up"

def _is_thumbs_up(rot: np.ndarray, handedness: float, flags: Dict[str, bool]) -> bool:
    # Stronger thumbs-up: thumb extended, others mostly curled (<=2 non-thumbs up), and thumb above index MCP.
    if not flags["thumb"]:
        return False
    non_thumb_up = sum(flags[f] for f in ["index", "middle", "ring", "pinky"])
    if non_thumb_up > 1:
        return False
    thumb_tip = rot[4]
    index_mcp = rot[5]
    # In rotated frame, "up" == tip[1] < base[1]
    return thumb_tip[1] < index_mcp[1] - 0.05 * max(30.0, np.linalg.norm(rot[9]))

def _is_pinch(rot: np.ndarray) -> bool:
    # Simple pinch: index tip close to thumb tip
    thumb_tip = rot[4]
    index_tip = rot[8]
    dist = np.linalg.norm(index_tip - thumb_tip)
    scale = max(30.0, np.linalg.norm(rot[9]))
    return dist < 0.20 * scale  # ~20% of hand scale

def classify_hand(
    landmarks_xy: np.ndarray,
    handedness: float,
) -> Dict:
    """
    Inputs:
      - landmarks_xy: (21,2) or (21,3) numpy array in image pixel space
      - handedness:   float (your detector uses >0.5 => 'right', else 'left')
    Returns:
      {
        'fingers_up': {'thumb':bool, ...},
        'fingers_up_count': int,
        'fingers_up_list': ['index','middle',...],
        'gesture': 'fist'|'peace'|'open'|...,
        'extras': {'thumbs_up':bool, 'pinch':bool}
      }
    """
    if landmarks_xy.shape[1] == 3:
        lms_xy = landmarks_xy[:, :2]
    else:
        lms_xy = landmarks_xy

    rot, _ = _normalize_to_palm_frame(lms_xy)
    flags = _count_fingers_up(rot, handedness)
    gesture = _gesture_from_pattern(flags)

    # Priority gestures override simple count
    if _is_thumbs_up(rot, handedness, flags):
        gesture = "thumbs_up"
    elif _is_pinch(rot):
        gesture = "pinch"

    up_list = [f for f, on in flags.items() if on]
    return {
        "fingers_up": flags,
        "fingers_up_count": len(up_list),
        "fingers_up_list": up_list,
        "gesture": gesture,
        "extras": {
            "thumbs_up": gesture == "thumbs_up",
            "pinch": gesture == "pinch",
        },
    }
