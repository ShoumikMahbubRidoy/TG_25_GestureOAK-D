# src/gesture_oak/detection/swipe_detector.py
from __future__ import annotations
import math
import socket
from collections import deque
from typing import Deque, Optional, Tuple

def _now_ms():
    # the apps pass timestamps; this is a fallback if needed
    import time
    return int(time.time() * 1000)

class SwipeDetector:
    """
    Lightweight, real-time swipe detector with:
      - EMA smoothing (reduces jitter)
      - Confidence & optional depth gating (reduces false triggers)
      - Direction/straightness/velocity tests (precision)
      - Debounce (prevents duplicate fires)
      - UDP notify on confirmed swipe ("Swipe" -> 192.168.10.10:6001 by default)
    """

    def __init__(
        self,
        # Precision thresholds (Normal mode defaults)
        min_confidence: float = 0.65,
        min_distance_px: int = 100,           # horizontal travel (px)
        max_y_deviation_px: int = 45,         # how straight the path must be
        max_duration_ms: int = 650,           # swipe must finish within this time
        min_consistent_dir_ratio: float = 0.80,
        min_avg_vx_px_per_ms: float = 0.55,   # avg horizontal speed
        # Depth gate (optional)
        use_depth_gate: bool = True,
        min_depth_m: float = 0.25,
        max_depth_m: float = 1.20,
        # Smoothing & history
        ema_alpha: float = 0.35,
        history_capacity: int = 24,           # ~0.8s at 30fps
        # Debounce
        cooldown_ms: int = 450,
        # UDP target
        udp_target: Tuple[str, int] = ("192.168.10.10", 6001),
    ):
        self.min_confidence = float(min_confidence)
        self.min_distance_px = int(min_distance_px)
        self.max_y_deviation_px = int(max_y_deviation_px)
        self.max_duration_ms = int(max_duration_ms)
        self.min_consistent_dir_ratio = float(min_consistent_dir_ratio)
        self.min_avg_vx_px_per_ms = float(min_avg_vx_px_per_ms)

        self.use_depth_gate = bool(use_depth_gate)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

        self.alpha = float(ema_alpha)
        self.prev_smooth: Optional[Tuple[float, float]] = None

        self.buf: Deque[Tuple[int, float, float]] = deque(maxlen=int(history_capacity))
        self.cooldown_ms = int(cooldown_ms)
        self.last_fire_ms = -10**9

        # UDP (fire-and-forget)
        self._udp_addr = udp_target
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # stats (optional, for HUD)
        self.total_swipes = 0
        self.false_positives_filtered = 0

    # allow runtime tuning (used by Option 3 mode keys 1/2/3)
    def set_params(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _smooth(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        if self.prev_smooth is None:
            self.prev_smooth = pt
            return pt
        ax = self.alpha * pt[0] + (1 - self.alpha) * self.prev_smooth[0]
        ay = self.alpha * pt[1] + (1 - self.alpha) * self.prev_smooth[1]
        self.prev_smooth = (ax, ay)
        return self.prev_smooth

    def _notify_udp(self):
        try:
            self._udp_sock.sendto(b"Swipe", self._udp_addr)
        except Exception:
            # Never let network glitches break the loop
            pass

    def _evaluate_path(self) -> bool:
        """Return True if the buffered path qualifies as a left->right swipe."""
        if len(self.buf) < 8:
            return False

        t0, x0, y0 = self.buf[0]
        t1, x1, y1 = self.buf[-1]
        dt = max(1, t1 - t0)  # ms (avoid div0)

        # basic geometry
        dx = x1 - x0
        dy = y1 - y0
        dist = dx  # horizontal travel (we require L->R => dx>0)

        if dt > self.max_duration_ms:
            return False
        if dist < self.min_distance_px:
            return False
        if abs(dy) > self.max_y_deviation_px:
            return False

        # direction consistency: count segments with positive dx
        pos_dir = 0
        total_seg = 0
        max_y_dev = 0.0
        last_t, last_x, last_y = self.buf[0]
        for t, x, y in list(self.buf)[1:]:
            ddx = x - last_x
            ddy = y - last_y
            total_seg += 1
            if ddx > 0:
                pos_dir += 1
            max_y_dev = max(max_y_dev, abs(y - y0))
            last_t, last_x, last_y = t, x, y

        if total_seg == 0:
            return False

        dir_ratio = pos_dir / total_seg
        if dir_ratio < self.min_consistent_dir_ratio:
            return False

        # velocity
        avg_vx = dist / dt  # px per ms
        if avg_vx < self.min_avg_vx_px_per_ms:
            return False

        return True

    def update(
        self,
        center_xy: Tuple[float, float],
        ts_ms: Optional[int] = None,
        hand_conf: float = 1.0,
        hand_depth_m: Optional[float] = None,
    ) -> bool:
        """Feed each hand center with timestamp (ms). Return True exactly when a swipe fires."""
        # Confidence gate
        if hand_conf < self.min_confidence:
            return False

        # Depth gate
        if self.use_depth_gate and (hand_depth_m is not None):
            if not (self.min_depth_m <= hand_depth_m <= self.max_depth_m):
                return False

        # Smooth input
        cx, cy = self._smooth(center_xy)
        if ts_ms is None:
            ts_ms = _now_ms()

        # Push to buffer
        self.buf.append((int(ts_ms), float(cx), float(cy)))

        # Evaluate
        if not self._evaluate_path():
            return False

        # Debounce
        if (ts_ms - self.last_fire_ms) < self.cooldown_ms:
            return False

        # Fire!
        self.last_fire_ms = ts_ms
        self.total_swipes += 1
        self._notify_udp()
        return True
