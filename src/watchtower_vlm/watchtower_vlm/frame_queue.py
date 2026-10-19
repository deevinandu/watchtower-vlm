"""Thread-safe single-slot queue that always keeps the newest frame."""

import queue


class LatestFrameQueue:
    """A ``queue.Queue(maxsize=1)`` wrapper that drops stale frames on put."""

    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self.dropped = 0

    def put_latest(self, item):
        """Insert ``item``, discarding any frame still waiting to be processed."""
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def get(self, timeout=None):
        """Block until a frame is available; raise ``queue.Empty`` on timeout."""
        return self._queue.get(timeout=timeout)
