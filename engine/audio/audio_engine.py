"""Procedural sound effects (plan phase 9.1-9.2).

No audio middleware is bundled, so sounds are synthesised into small WAV
files at startup and played through the standard-library ``winsound`` module
on Windows.  On other platforms (or if playback fails) the engine degrades
to a silent no-op — the game never depends on audio succeeding.

This is intentionally lightweight: single-shot cues (break/place/hurt/step)
with a short cooldown so rapid actions don't machine-gun the mixer.  True 3D
positional mixing needs OpenAL and is deferred (see ROADMAP).
"""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import numpy as np

from engine.core.log import get_logger

_log = get_logger("audio")
_RATE = 22050

try:
    import winsound  # noqa: F401 - Windows only
    _HAVE_WINSOUND = True
except ImportError:
    _HAVE_WINSOUND = False


def _tone(freq, dur, kind="sine", decay=8.0, vol=0.4):
    t = np.linspace(0, dur, int(_RATE * dur), endpoint=False)
    if kind == "noise":
        wave_data = np.random.default_rng(int(freq)).uniform(-1, 1, t.shape)
    elif kind == "square":
        wave_data = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        wave_data = np.sin(2 * np.pi * freq * t)
    env = np.exp(-t * decay)
    return wave_data * env * vol


def _sweep(f0, f1, dur, decay=6.0, vol=0.4):
    t = np.linspace(0, dur, int(_RATE * dur), endpoint=False)
    freq = np.linspace(f0, f1, t.shape[0])
    phase = np.cumsum(2 * np.pi * freq / _RATE)
    return np.sin(phase) * np.exp(-t * decay) * vol


class AudioEngine:
    def __init__(self) -> None:
        self.enabled = _HAVE_WINSOUND
        self._dir = Path(tempfile.mkdtemp(prefix="pymc_audio_"))
        self._files: dict[str, str] = {}
        self._cooldown: dict[str, float] = {}
        self._time = 0.0
        if not self.enabled:
            _log.info("No winsound available — audio disabled")
            return
        self._bake()

    def _bake(self) -> None:
        banks = {
            "break": _tone(140, 0.18, "noise", decay=22, vol=0.5),
            "place": _tone(320, 0.10, "square", decay=30, vol=0.35),
            "step": _tone(90, 0.07, "noise", decay=40, vol=0.25),
            "hurt": _sweep(300, 120, 0.22, decay=10, vol=0.5),
            "splash": _tone(700, 0.20, "noise", decay=12, vol=0.3),
            "explode": _sweep(180, 40, 0.5, decay=5, vol=0.7),
            "craft": _tone(520, 0.09, "square", decay=26, vol=0.3),
            "eat": _tone(200, 0.12, "noise", decay=24, vol=0.3),
        }
        for name, samples in banks.items():
            self._files[name] = self._write_wav(name, samples)

    def _write_wav(self, name: str, samples: np.ndarray) -> str:
        clip = np.clip(samples, -1, 1)
        data = (clip * 32767).astype("<i2")
        path = self._dir / f"{name}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_RATE)
            wf.writeframes(data.tobytes())
        return str(path)

    def update(self, dt: float) -> None:
        self._time += dt

    def play(self, name: str, min_gap: float = 0.06) -> None:
        if not self.enabled or name not in self._files:
            return
        if self._time - self._cooldown.get(name, -99.0) < min_gap:
            return
        self._cooldown[name] = self._time
        try:
            import winsound
            winsound.PlaySound(self._files[name],
                               winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except Exception:  # noqa: BLE001 - audio must never crash the game
            pass
