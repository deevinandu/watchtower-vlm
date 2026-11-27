"""Latching E-Stop state: trips instantly, releases only after sustained clean frames."""


class EStopLatch:
    """Not thread-safe; callers must serialise access."""

    def __init__(self, clear_frames: int = 1):
        self.clear_frames = clear_frames
        self.active = False
        self._clean = 0

    def update(self, anomaly: bool) -> bool:
        """Feed one decision; return whether the E-Stop is asserted."""
        if anomaly:
            self.trip()
        elif self.active:
            self._clean += 1
            if self._clean >= self.clear_frames:
                self.reset()
        return self.active

    def trip(self):
        self.active = True
        self._clean = 0

    def reset(self):
        self.active = False
        self._clean = 0
