"""
Lightweight FPS helper with robust global timing.

- Instant FPS: over a short deque window (default 30 samples)
- Global FPS: from the first update() call using a fixed start_time
"""
import time
import cv2
from collections import deque

class FPS:
    def __init__(self, average_of: int = 30):
        self.timestamps = deque(maxlen=average_of)
        self.nbf = 0
        self.start_time = None
        self._inst_fps = 0.0

    def update(self):
        now = time.perf_counter()
        if self.start_time is None:
            self.start_time = now
        self.timestamps.append(now)
        self.nbf += 1
        if len(self.timestamps) >= 2:
            dt = self.timestamps[-1] - self.timestamps[0]
            frames = len(self.timestamps) - 1
            self._inst_fps = frames / dt if dt > 0 else 0.0

    def get(self) -> float:
        """Instantaneous FPS (rolling window)."""
        return float(self._inst_fps)

    def get_global(self) -> float:
        """Global FPS since first update call."""
        if self.start_time is None:
            return 0.0
        now = time.perf_counter()
        dt = now - self.start_time
        return (self.nbf / dt) if dt > 0 else 0.0

    def nb_frames(self) -> int:
        return self.nbf

    def draw(self, win, orig=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
             size=0.7, color=(0, 255, 0), thickness=2):
        cv2.putText(win, f"FPS={self.get():.1f}", orig, font, size, color, thickness)

if __name__ == "__main__":
    fps = FPS()
    for _ in range(50):
        fps.update()
        print(f"inst fps = {fps.get():.2f}  global = {fps.get_global():.2f}")
        time.sleep(0.02)
    print(f"Frames: {fps.nb_frames()}")
