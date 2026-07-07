"""Alternate dimensions: the Nether (plan phase 6).

Each dimension is a self-contained (generator, World, storage) bundle with
its own save directory.  The overworld keeps using WorldGenerator; the
Nether uses NetherGenerator below.  The End is deferred (see ROADMAP).

Dark by design: the Nether has a solid bedrock ceiling, so the skylight
flood naturally finds no open columns and leaves everything unlit except
block light from lava and glowstone — no special-casing needed.
"""

from __future__ import annotations

import numpy as np

from engine.world.blocks import AIR, BlockRegistry
from engine.world.coords import CHUNK_X, CHUNK_Y, CHUNK_Z
from engine.world.noise import NoiseField

NETHER_CEILING = 120
NETHER_LAVA = 31


class NetherGenerator:
    """Netherrack caverns under a bedrock lid, with a lava sea and glowstone."""

    def __init__(self, seed: int, registry: BlockRegistry) -> None:
        self.seed = seed ^ 0x4E657468  # distinct from the overworld
        self.noise = NoiseField(self.seed)
        self.registry = registry
        self.structures = None
        ids = registry.id_of
        self._bedrock = ids("bedrock")
        self._netherrack = ids("netherrack")
        self._lava = ids("lava")
        self._glowstone = ids("glowstone")
        self._soul_sand = ids("soul_sand")
        self._magma = ids("netherrack")

    def find_spawn(self, *_args, **_kw) -> tuple[int, int]:
        return 0, 0

    def generate_chunk(self, cx: int, cz: int) -> np.ndarray:
        blocks = np.zeros((CHUNK_X, CHUNK_Z, CHUNK_Y), dtype=np.uint8)
        n = self.noise
        wx = np.arange(cx * CHUNK_X, (cx + 1) * CHUNK_X, dtype=np.float32)[:, None]
        wz = np.arange(cz * CHUNK_Z, (cz + 1) * CHUNK_Z, dtype=np.float32)[None, :]
        y_idx = np.arange(CHUNK_Y, dtype=np.int32)[None, None, :]

        # A noisy netherrack floor with a solid ceiling and open cavern between.
        floor_h = (28 + n.fbm2(wx * 0.02, wz * 0.02, octaves=3) * 16).astype(np.int32)
        floor_h = np.clip(floor_h, 18, 60)
        ceil_h = (NETHER_CEILING - 6 - n.fbm2(wx * 0.02 + 300, wz * 0.02 + 300, octaves=2)
                  * 10).astype(np.int32)

        blocks[y_idx <= floor_h[:, :, None]] = self._netherrack
        blocks[y_idx >= ceil_h[:, :, None]] = self._netherrack

        # Floating netherrack blobs give the cavern depth.
        x3 = wx[:, :, None]
        z3 = wz[:, :, None]
        yF = np.arange(CHUNK_Y, dtype=np.float32)[None, None, :]
        blob = n.fbm3(x3 * 0.045, yF * 0.05, z3 * 0.045, octaves=2)
        open_band = (y_idx > floor_h[:, :, None]) & (y_idx < ceil_h[:, :, None])
        blocks[np.broadcast_to(open_band, blocks.shape) & (blob > 0.5)] = self._netherrack

        # Lava sea fills every open cell at or below the lava line.
        lava = (blocks == AIR) & (y_idx <= NETHER_LAVA)
        blocks[lava] = self._lava

        # Soul-sand shore just above the lava.
        shore = (blocks == self._netherrack) & (y_idx > NETHER_LAVA) & (y_idx < NETHER_LAVA + 3)
        soul = shore & (n.fbm3(x3 * 0.05 + 40, yF * 0.05, z3 * 0.05, octaves=2) > 0.35)
        blocks[soul] = self._soul_sand

        # Glowstone clusters hanging where netherrack has air directly below.
        below_air = np.zeros_like(blocks, dtype=bool)
        below_air[:, :, :-1] = blocks[:, :, 1:] == AIR
        glow = (blocks == self._netherrack) & below_air & (
            n.fbm3(x3 * 0.08 + 90, yF * 0.08, z3 * 0.08, octaves=2) > 0.5
        )
        blocks[glow] = self._glowstone

        blocks[:, :, 0] = self._bedrock
        blocks[:, :, NETHER_CEILING + 1 :] = self._bedrock
        return blocks


def nether_safe_y(blocks: np.ndarray, x: int, z: int, registry: BlockRegistry) -> int:
    """Lowest standable spot above the lava sea for a Nether arrival."""
    column = blocks[x, z, :]
    for y in range(NETHER_LAVA + 2, NETHER_CEILING - 2):
        if (registry.solid[column[y]]
                and column[y + 1] == AIR and column[y + 2] == AIR):
            return y + 1
    return NETHER_LAVA + 6
