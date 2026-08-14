"""Screen enumeration + real-time capture thread (downscaled, pausable)."""

import time
import mss
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from screeninfo import get_monitors


def list_screens():
    screens = []
    for i, m in enumerate(get_monitors()):
        screens.append({
            "index": i, "name": m.name,
            "x": m.x, "y": m.y, "width": m.width, "height": m.height,
        })
    return screens


class ScreenCaptureThread(QThread):
    """Grabs frames, downscales them on this thread, and emits BGR arrays.

    * display_width: long edge the GUI will ever see (big speed win on 1440p/4K).
    * set_paused(True): the loop stops grabbing and just sleeps (instant freeze).
    * last_full: the most recent FULL-resolution grab, used for high-quality saves.
    """
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, monitor_index, fps=30, display_width=1600):
        super().__init__()
        self.monitor_index = monitor_index
        self.fps = fps
        self.display_width = display_width
        self._running = False
        self._paused = False
        self.last_full = None

    def set_paused(self, b):
        self._paused = bool(b)

    def run(self):
        self._running = True
        with mss.mss() as sct:
            mon = sct.monitors[self.monitor_index + 1]
            interval = 1.0 / self.fps
            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue

                t0 = time.perf_counter()
                grab = sct.grab(mon)
                img = np.array(grab)             # BGRA
                if img is None or img.size == 0:
                    time.sleep(interval)
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)   # contiguous BGR
                self.last_full = img                          # keep full-res for crops
                h, w = img.shape[:2]
                if self.display_width and w > self.display_width:
                    nh = max(2, int(round(h * self.display_width / w)))
                    img = cv2.resize(img, (self.display_width, nh),
                                     interpolation=cv2.INTER_AREA)

                self.frame_ready.emit(img)

                left = interval - (time.perf_counter() - t0)
                if left > 0:
                    time.sleep(left)

    def stop(self):
        self._running = False
        self.wait(2000)