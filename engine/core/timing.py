"""Frame clock: delta time + rolling FPS statistics."""

from __future__ import annotations

import time


class FrameClock:
    """Tracks per-frame delta time and a smoothed FPS value.

    Delta time is clamped so a debugger pause or a long hitch never turns
    into a physics explosion.
    """

    MAX_DELTA = 0.1  # seconds

    def __init__(self) -> None:
        self._last = time.perf_counter()
        self.delta: float = 1.0 / 60.0
        self.time: float = 0.0
        self.fps: float = 0.0
        self._acc_time = 0.0
        self._acc_frames = 0

    def tick(self) -> float:
        now = time.perf_counter()
        raw = now - self._last
        self._last = now
        self.delta = min(raw, self.MAX_DELTA)
        self.time += self.delta

        # FPS is averaged over ~0.5 s so the debug overlay is readable.
        self._acc_time += raw
        self._acc_frames += 1
        if self._acc_time >= 0.5:
            self.fps = self._acc_frames / self._acc_time
            self._acc_time = 0.0
            self._acc_frames = 0
        return self.delta
