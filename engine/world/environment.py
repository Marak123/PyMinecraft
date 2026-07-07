"""Day/night cycle: sun direction, sky palette, fog colour.

Produces plain uniform values; the renderer stays a dumb consumer.
"""

from __future__ import annotations

import math
import random

import numpy as np

_DAY_ZENITH = np.array([0.40, 0.65, 0.95])
_DAY_HORIZON = np.array([0.72, 0.82, 0.92])
_NIGHT_ZENITH = np.array([0.012, 0.018, 0.05])
_NIGHT_HORIZON = np.array([0.05, 0.06, 0.11])
_SUNSET = np.array([0.95, 0.55, 0.30])


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


class Environment:
    def __init__(self, day_length_seconds: float, start_time: float = 0.30) -> None:
        self.day_length = day_length_seconds
        # time_of_day in [0, 1): 0.25 = sunrise-ish, 0.5 = noon, 0.75 = sunset.
        self.time_of_day = start_time

        self.sun_dir = np.array([0.0, 1.0, 0.0])
        self.daylight = 1.0
        self.zenith_color = _DAY_ZENITH.copy()
        self.horizon_color = _DAY_HORIZON.copy()
        self.fog_color = _DAY_HORIZON.copy()
        # Weather: rain comes and goes in random spells.
        self.raining = False
        self._weather_timer = random.uniform(180.0, 420.0)
        self.update(0.0)

    def _update_weather(self, dt: float) -> None:
        self._weather_timer -= dt
        if self._weather_timer > 0.0:
            return
        if self.raining:
            self.raining = False
            self._weather_timer = random.uniform(300.0, 700.0)
        elif random.random() < 0.45:
            self.raining = True
            self._weather_timer = random.uniform(80.0, 200.0)
        else:
            self._weather_timer = random.uniform(120.0, 300.0)

    def update(self, dt: float) -> None:
        self._update_weather(dt)
        self.time_of_day = (self.time_of_day + dt / self.day_length) % 1.0
        angle = (self.time_of_day - 0.25) * 2.0 * math.pi  # 0.25 -> sunrise at horizon
        elevation = math.sin(angle)

        # Sun travels a tilted arc so shadows/shading feel three-dimensional.
        self.sun_dir = np.array(
            [math.cos(angle) * 0.85, elevation, math.cos(angle) * 0.35]
        )
        norm = np.linalg.norm(self.sun_dir)
        if norm > 1e-6:
            self.sun_dir = self.sun_dir / norm

        # Daylight ramps smoothly through twilight instead of snapping.
        self.daylight = float(np.clip((elevation + 0.12) / 0.35, 0.0, 1.0))
        if self.raining:
            self.daylight *= 0.55  # overcast skies

        day_t = self.daylight
        self.zenith_color = _lerp(_NIGHT_ZENITH, _DAY_ZENITH, day_t)
        horizon = _lerp(_NIGHT_HORIZON, _DAY_HORIZON, day_t)
        # Warm tint near the horizon when the sun is close to it.
        sunset_strength = math.exp(-((elevation / 0.18) ** 2)) * 0.8
        self.horizon_color = _lerp(horizon, _SUNSET, sunset_strength)
        self.fog_color = self.horizon_color.copy()

        # Colour grading for the tonemap pass: cool nights, warm sunsets.
        cool = np.array([0.88, 0.94, 1.10])
        grade = _lerp(cool, np.array([1.0, 1.0, 1.0]), day_t)
        warm = np.array([1.08, 0.98, 0.90])
        self.color_grade = _lerp(grade, warm, sunset_strength * day_t * 0.8)
