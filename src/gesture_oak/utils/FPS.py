"""
@author: geaxx (stabilized by ChatGPT)
"""
import time
import cv2
from collections import deque

class FPS:
    """
    Lightweight FPS counter that avoids division-by-zero and crazy spikes.
    """
    def __init__(self, average_of=30):
        self.timestamps = deque(maxlen=average_of)
        self.nbf = 0
        self.start = None
        self.fps = 0.0

    def update(self):
        now = time.perf_counter()
        self.timestamps.append(now)
        if self.start is None:
            self.start = now
            self.fps = 0.0
        elif len(self.timestamps) >= 2:
            dt = self.timestamps[-1] - self.timestamps[0]
            if dt <= 1e-4:     # avoid absurd spikes
                self.fps = 0.0
            else:
                self.fps = (len(self.timestamps) - 1) / dt
        self.nbf += 1

    def get(self):
        return float(self.fps)

    def get_global(self):
        if self.start is None or self.nbf < 2:
            return 0.0
        elapsed = (self.timestamps[-1] - self.start) if self.timestamps else (time.perf_counter() - self.start)
        if elapsed <= 1e-4:
            return 0.0
        return (self.nbf - 1) / elapsed

    def nb_frames(self):
        return self.nbf

    def draw(self, win, orig=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, size=0.7, color=(0,255,0), thickness=2):
        cv2.putText(win, f"FPS={self.get():.1f}", orig, font, size, color, thickness)

if __name__ == "__main__":
    fps = FPS()
    for i in range(50):
        fps.update()
        print(f"fps = {fps.get():.2f}")
        time.sleep(0.1)
    print(f"Global fps : {fps.get_global():.2f} ({fps.nb_frames()} frames)")
