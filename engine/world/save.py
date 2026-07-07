"""World persistence.

Only *modified* chunks are ever written — pristine terrain is always
regenerable from the seed, so saving it would be wasted I/O.  Chunk files are
compressed ``.npz`` archives; ``meta.json`` stores the seed and player state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from engine.core.log import get_logger
from engine.world.coords import CHUNK_Y

_log = get_logger("save")

_FORMAT_VERSION = 1


class WorldStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.chunk_dir = root / "chunks"

    # -- meta -----------------------------------------------------------------
    def load_meta(self) -> dict[str, Any] | None:
        path = self.root / "meta.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("Corrupt meta.json (%s); starting fresh", exc)
            return None

    def save_meta(self, meta: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        meta = {"format": _FORMAT_VERSION, **meta}
        (self.root / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    # -- chunks -----------------------------------------------------------------
    def _chunk_path(self, cx: int, cz: int) -> Path:
        return self.chunk_dir / f"c_{cx}_{cz}.npz"

    def load_chunk(self, cx: int, cz: int) -> np.ndarray | None:
        path = self._chunk_path(cx, cz)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                blocks = data["blocks"]
        except (OSError, ValueError, KeyError) as exc:
            # Corrupt chunk file: log and fall back to regeneration rather
            # than crashing the whole world load.
            _log.warning("Corrupt chunk %s (%s); regenerating", path.name, exc)
            return None
        # Migrate pre-256 saves: pad the extra height with air (plan 4.1).
        if blocks.shape[2] < CHUNK_Y:
            padded = np.zeros((blocks.shape[0], blocks.shape[1], CHUNK_Y), dtype=np.uint8)
            padded[:, :, : blocks.shape[2]] = blocks
            return padded
        return blocks

    def save_chunk(self, cx: int, cz: int, blocks: np.ndarray) -> None:
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._chunk_path(cx, cz), blocks=blocks)
