#!/usr/bin/env python3
# Fast-swipe tolerant detector with short-gap bridging and UDP notify.

from __future__ import annotations
import time
import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Tuple
import socket


class SwipeState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    VALIDATING = "validating"
    CONFIRMED = "confirmed"


class SwipeDetector:
    """
    Left-to-right swipe detector with:
      - relaxed long-range thresholds (80–160 cm)
      - short-gap bridging (≤120 ms) to avoid resets on fast flicks
      - UDP "Swipe" packet on confirmation
    """

    def __init__(
        self,
        buffer_size: int = 20,      # a bit more history for consistency
        min_distance: int = 85,     # smaller pixel travel at long range
        min_duration: float = 0.20,
        max_duration: float = 1.60,
        min_velocity: float = 35,
        max_velocity: float = 1100,
        max_y_deviation: float = 0.38,
    ):
        self.buffer_size = buffer_size
        self.min_distance = min_distance
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.max_y_deviation = max_y_deviation

        self.state = SwipeState.IDLE
        self.position_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)

        self.swipe_start_time = None
        self.swipe_start_pos = None
        self.last_confirmed_swipe = 0.0
        self.swipe_cooldown = 0.45  # seconds

        self.total_swipes_detected = 0
        self.false_positives_filtered = 0

        # UDP (send "Swipe" upon confirmation)
        try:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_target = ("192.168.10.10", 6001)
        except Exception:
            self._udp_sock = None
            self._udp_target = None

        # Gap-bridging state
        self._last_pos: Optional[Tuple[float, float]] = None
        self._last_ts: Optional[float] = None
        self._gap_hold_s: float = 0.12  # allow ~120ms gap without reset

    # --------------- public API ---------------

    def update(self, hand_center: Optional[Tuple[float, float]]) -> bool:
        """Feed the latest hand center (x, y). Return True when a swipe is confirmed."""
        current_time = time.time()

        # Tolerate 1–2 missing frames in a fast swipe
        if hand_center is None:
            if self._last_ts is not None and (current_time - self._last_ts) <= self._gap_hold_s:
                hand_center = self._last_pos   # hold previous briefly
            else:
                self._last_pos = None
                self._last_ts = None
                self._reset_detection()
                return False
        else:
            self._last_pos = hand_center
            self._last_ts = current_time

        # Push to buffers
        self.position_buffer.append(hand_center)
        self.time_buffer.append(current_time)

        if len(self.position_buffer) < 3:
            return False

        if self.state == SwipeState.IDLE:
            return self._check_start_detection()
        elif self.state == SwipeState.DETECTING:
            return self._process_detection()
        elif self.state == SwipeState.VALIDATING:
            return self._validate_swipe()

        return False

    def get_current_swipe_progress(self) -> Optional[dict]:
        if self.state == SwipeState.IDLE or not self.swipe_start_pos or not self.position_buffer:
            return None
        current_pos = self.position_buffer[-1]
        duration = self.time_buffer[-1] - (self.swipe_start_time or self.time_buffer[-1])
        dx = current_pos[0] - self.swipe_start_pos[0]
        velocity = dx / duration if duration > 0 else 0.0
        return {
            "state": self.state.value,
            "distance": dx,
            "duration": duration,
            "velocity": velocity,
            "progress": min(max(dx / self.min_distance, 0.0), 1.0),
        }

    def get_statistics(self) -> dict:
        return {
            "total_swipes_detected": self.total_swipes_detected,
            "false_positives_filtered": self.false_positives_filtered,
            "current_state": self.state.value,
        }

    def reset_statistics(self):
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0

    # --------------- internals ---------------

    def _check_start_detection(self) -> bool:
        if len(self.position_buffer) < 3:
            return False
        p = list(self.position_buffer)[-3:]
        dx1 = p[1][0] - p[0][0]
        dx2 = p[2][0] - p[1][0]
        if dx1 > 2 and dx2 > 2:  # easier start gate
            self.state = SwipeState.DETECTING
            self.swipe_start_time = self.time_buffer[-3]
            self.swipe_start_pos = p[0]
        return False

    def _process_detection(self) -> bool:
        current_pos = self.position_buffer[-1]
        current_time = self.time_buffer[-1]

        if current_time - self.swipe_start_time > self.max_duration:
            self._reset_detection()
            return False

        total_dx = current_pos[0] - self.swipe_start_pos[0]
        if total_dx < 0:
            self._reset_detection()
            return False

        if total_dx >= self.min_distance:
            self.state = SwipeState.VALIDATING

        return False

    def _validate_swipe(self) -> bool:
        current_time = self.time_buffer[-1]
        duration = current_time - self.swipe_start_time

        if duration < self.min_duration:
            return False
        if duration > self.max_duration:
            self._reset_detection()
            return False

        if self._validate_swipe_characteristics():
            if current_time - self.last_confirmed_swipe > self.swipe_cooldown:
                self.total_swipes_detected += 1
                self.last_confirmed_swipe = current_time

                # Send UDP packet on confirmed swipe
                if self._udp_sock is not None and self._udp_target is not None:
                    try:
                        self._udp_sock.sendto(b"Swipe", self._udp_target)
                    except Exception:
                        pass

                self._reset_detection()
                return True

        self._reset_detection()
        return False

    def _validate_swipe_characteristics(self) -> bool:
        if len(self.position_buffer) < 5:
            return False

        positions = np.array(self.position_buffer)
        times = np.array(self.time_buffer)

        # slice from swipe start
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

        y_dev = abs(total_dy) / total_dx if total_dx > 0 else float("inf")
        if y_dev > self.max_y_deviation:
            self.false_positives_filtered += 1
            return False

        dur = swipe_times[-1] - swipe_times[0]
        if dur <= 0:
            return False
        avg_v = total_dx / dur
        if avg_v < self.min_velocity or avg_v > self.max_velocity:
            self.false_positives_filtered += 1
            return False

        # Direction consistency (tolerate small backtracks)
        for i in range(1, len(swipe_positions)):
            dx = swipe_positions[i][0] - swipe_positions[i - 1][0]
            if dx < -15:  # small negative jumps allowed; big ones break
                self.false_positives_filtered += 1
                return False

        return True

    def _reset_detection(self):
        self.state = SwipeState.IDLE
        self.swipe_start_time = None
        self.swipe_start_pos = None
