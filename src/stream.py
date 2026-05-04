import threading
import time

import cv2


class LatestFrameCapture:
    def __init__(self, source, reconnect_delay=2.0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self._capture = None
        self._frame = None
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None

    def _open_capture(self):
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture if capture.isOpened() else None

    def start(self):
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self):
        while not self._stop_event.is_set():
            if self._capture is None:
                self._capture = self._open_capture()
                if self._capture is None:
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self._capture.read()
            if not ok:
                self._release_capture()
                time.sleep(self.reconnect_delay)
                continue

            with self._lock:
                self._frame = frame
                self._frame_ready.set()

    def read(self, timeout=None):
        if not self._frame_ready.wait(timeout):
            return False, None

        with self._lock:
            if self._frame is None:
                self._frame_ready.clear()
                return False, None

            frame = self._frame
            self._frame = None
            self._frame_ready.clear()
            return True, frame

    def _release_capture(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        with self._lock:
            self._frame = None
            self._frame_ready.clear()

    def stop(self):
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self._release_capture()