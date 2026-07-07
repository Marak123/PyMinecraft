"""Engine/game settings loaded from ``configs/settings.json``.

The file is created with defaults on first run so players can tweak it
without reading any docs.  Values are clamped to sane ranges — a bad config
should degrade gracefully, never crash the engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from engine.core.log import get_logger

_log = get_logger("config")

DEFAULT_HOTBAR = [
    "grass_block",
    "stone",
    "cobblestone",
    "oak_planks",
    "oak_log",
    "glass",
    "bricks",
    "glowstone",
    "torch",
]


@dataclass
class Settings:
    window_width: int = 1280
    window_height: int = 720
    fullscreen: bool = False
    vsync: bool = True
    fov: float = 75.0
    render_distance: int = 8  # in chunks
    shadows: bool = True
    bloom: bool = True
    texture_pack: str | None = None  # folder with <tile>.png overrides
    mouse_sensitivity: float = 0.09  # degrees per pixel
    day_length_seconds: float = 900.0
    seed: int | None = None  # None -> random on world creation
    world_name: str = "world"
    hotbar: list[str] = field(default_factory=lambda: list(DEFAULT_HOTBAR))

    def clamp(self) -> None:
        self.window_width = max(640, int(self.window_width))
        self.window_height = max(360, int(self.window_height))
        self.fov = float(min(110.0, max(45.0, self.fov)))
        self.render_distance = int(min(24, max(4, self.render_distance)))
        self.mouse_sensitivity = float(min(1.0, max(0.005, self.mouse_sensitivity)))
        self.day_length_seconds = float(max(30.0, self.day_length_seconds))


def load_settings(path: Path) -> Settings:
    settings = Settings()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for key, value in data.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("Could not read %s (%s); using defaults", path, exc)
    else:
        save_settings(path, settings)
        _log.info("Created default settings at %s", path)
    settings.clamp()
    return settings


def save_settings(path: Path, settings: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings.__dict__, indent=2, ensure_ascii=False), encoding="utf-8"
    )
