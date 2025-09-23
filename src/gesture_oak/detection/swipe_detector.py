# src/gesture_oak/detection/swipe_detector.py
#!/usr/bin/env python3

import time
import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Tuple
import socket

# high-resolution monotonic clock for precise velocity
_now = time.perf_counter

class SwipeState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    VALIDATING = "validating"
    CONFIRMED = "confirmed"

class SwipeDetector:
    """Left-to-right swipe detector (fast-path enabled)."""

    def __init__(self,
                 buffer_size: int = 15,
                 min_distance: int = 120,
                 min_duration: float = 0.3,
                 max_duration: float = 2.0,
                 min_velocity: float = 50,
                 max_velocity: float = 800,
                 max_y_deviation: float = 0.3):
        self.buffer_size = buffer_size
        self.min_distance = min_distance
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.max_y_deviation = max_y_deviation

        # state
        self.state = SwipeState.IDLE
        self.position_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)

        self.swipe_start_time = None
        self.swipe_start_pos = None
        self.last_confirmed_swipe = 0.0
        self.swipe_cooldown = 1.0  # seconds

        # stats
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0

        # UDP: send "Swipe" when confirmed
        try:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_target = ("192.168.10.10", 6001)
        except Exception:
            self._udp_sock = None
            self._udp_target = None

    # ---------- fast path for very quick, straight swipes ----------
    def _fast_path_ready(self) -> bool:
        if len(self.position_buffer) < 4:
            return False
        positions = list(self.position_buffer)[-4:]
        times = list(self.time_buffer)[-4:]
        dt = max(times[-1] - times[0], 1e-4)

        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        vx = dx / dt

        # very fast, mostly straight, short window
        return (
            vx > 1500.0 and         # px/s (tune 1400–2000 if needed)
            dx >= 55 and            # px   (allow shorter distance when extremely fast)
            (abs(dy) / max(dx, 1)) <= 0.25 and
            (self.time_buffer[-1] - (self.swipe_start_time or self.time_buffer[0])) >= 0.08
        )

    def update(self, hand_center: Optional[Tuple[float, float]]) -> bool:
        t = _now()

        if hand_center is None:
            self._reset_detection()
            return False

        self.position_buffer.append(hand_center)
        self.time_buffer.append(t)

        if len(self.position_buffer) < 3:
            return False

        if self.state == SwipeState.IDLE:
            return self._check_start_detection()
        elif self.state == SwipeState.DETECTING:
            return self._process_detection()
        elif self.state == SwipeState.VALIDATING:
            return self._validate_swipe()

        return False

    def _check_start_detection(self) -> bool:
        if len(self.position_buffer) < 3:
            return False

        recent = list(self.position_buffer)[-3:]
        dx1 = recent[1][0] - recent[0][0]
        dx2 = recent[2][0] - recent[1][0]

        if dx1 > 3 and dx2 > 3:
            self.state = SwipeState.DETECTING
            self.swipe_start_time = self.time_buffer[-3]
            self.swipe_start_pos = recent[0]
        return False

    def _process_detection(self) -> bool:
        current_pos = self.position_buffer[-1]
        current_time = self.time_buffer[-1]

        if self.swipe_start_time is None or self.swipe_start_pos is None:
            self._reset_detection()
            return False

        if current_time - self.swipe_start_time > self.max_duration:
            self._reset_detection()
            return False

        total_dx = current_pos[0] - self.swipe_start_pos[0]

        # allow a small negative hiccup earlier; hard negative resets here
        if total_dx < 0:
            self._reset_detection()
            return False

        # fast-path: if obviously fast & straight, move to validation early
        if self._fast_path_ready():
            self.state = SwipeState.VALIDATING

        if total_dx >= self.min_distance:
            self.state = SwipeState.VALIDATING

        return False

    def _validate_swipe(self) -> bool:
        current_time = self.time_buffer[-1]
        if self.swipe_start_time is None:
            self._reset_detection()
            return False

        duration = current_time - self.swipe_start_time
        if duration < self.min_duration:
            # but allow fast-path to approve early
            if not self._fast_path_ready():
                return False
        if duration > self.max_duration:
            self._reset_detection()
            return False

        # Early approval if fast-path says it's clearly a swipe
        if self._fast_path_ready():
            if current_time - self.last_confirmed_swipe > self.swipe_cooldown:
                self.total_swipes_detected += 1
                self.last_confirmed_swipe = current_time
                if self._udp_sock and self._udp_target:
                    try: self._udp_sock.sendto(b"Swipe", self._udp_target)
                    except Exception: pass
                self._reset_detection()
                return True

        # Normal path
        if self._validate_swipe_characteristics():
            if current_time - self.last_confirmed_swipe > self.swipe_cooldown:
                self.total_swipes_detected += 1
                self.last_confirmed_swipe = current_time
                if self._udp_sock and self._udp_target:
                    try: self._udp_sock.sendto(b"Swipe", self._udp_target)
                    except Exception: pass
                self._reset_detection()
                return True

        self._reset_detection()
        return False

    def _validate_swipe_characteristics(self) -> bool:
        if len(self.position_buffer) < 5 or self.swipe_start_time is None:
            return False

        positions = np.array(list(self.position_buffer))
        times = np.array(list(self.time_buffer))

        start_idx = 0
        for i, t in enumerate(times):
            if t >= self.swipe_start_time:
                start_idx = i
                break

        swipe_positions = positions[start_idx:]
        swipe_times = times[start_idx:]
        if len(swipe_positions) < 3:
            return False

        total_dx = swipe_positions[-1][0] - swipe_positions[0][0]
        total_dy = swipe_positions[-1][1] - swipe_positions[0][1]
        if total_dx < self.min_distance:
            return False

        y_dev = abs(total_dy) / total_dx if total_dx > 0 else float('inf')
        if y_dev > self.max_y_deviation:
            self.false_positives_filtered += 1
            return False

        duration = swipe_times[-1] - swipe_times[0]
        if duration <= 0:
            return False

        avg_v = total_dx / duration
        if avg_v < self.min_velocity or avg_v > self.max_velocity:
            self.false_positives_filtered += 1
            return False

        # more tolerant to brief negative dx during fast motion
        consistent_right = True
        for i in range(1, len(swipe_positions)):
            dx = swipe_positions[i][0] - swipe_positions[i-1][0]
            if dx < -20:  # was -15; slightly more tolerant for fast flicks
                consistent_right = False
                break
        if not consistent_right:
            self.false_positives_filtered += 1
            return False

        return True

    def _reset_detection(self):
        self.state = SwipeState.IDLE
        self.swipe_start_time = None
        self.swipe_start_pos = None

    def get_current_swipe_progress(self) -> Optional[dict]:
        if self.state == SwipeState.IDLE or not self.swipe_start_pos or not self.position_buffer:
            return None
        current_pos = self.position_buffer[-1]
        current_time = self.time_buffer[-1]
        total_dx = current_pos[0] - self.swipe_start_pos[0]
        duration = max(current_time - self.swipe_start_time, 1e-6)
        velocity = total_dx / duration
        return {
            'state': self.state.value,
            'distance': total_dx,
            'duration': duration,
            'velocity': velocity,
            'progress': min(max(total_dx / self.min_distance, 0.0), 1.0)
        }

    def get_statistics(self) -> dict:
        return {
            'total_swipes_detected': self.total_swipes_detected,
            'false_positives_filtered': self.false_positives_filtered,
            'current_state': self.state.value
        }

    def reset_statistics(self):
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0
