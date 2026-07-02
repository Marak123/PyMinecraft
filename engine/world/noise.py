"""Vectorised gradient noise (Perlin 2D/3D + fBM + ridged).

Implemented in pure NumPy so the whole terrain pipeline stays dependency-free
and works on arbitrary-shaped coordinate arrays.  All functions are pure and
thread-safe after construction, which lets chunk generation run on worker
threads without locking.
"""

from __future__ import annotations

import numpy as np

_GRAD2 = np.array(
    [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)],
    dtype=np.float32,
)
_GRAD3 = np.array(
    [
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    ],
    dtype=np.float32,
)


def _fade(t: np.ndarray) -> np.ndarray:
    # Quintic fade curve (Perlin improved) — C2-continuous across cells.
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class NoiseField:
    """Seeded gradient-noise generator with a private permutation table."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(256).astype(np.int64)
        self._p = np.concatenate([perm, perm])

    # -- primitives -----------------------------------------------------------
    def perlin2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        xf = x - xi
        yf = y - yi
        xi &= 255
        yi &= 255

        u = _fade(xf)
        v = _fade(yf)

        p = self._p
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]

        def grad(h: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
            g = _GRAD2[h & 7]
            return g[..., 0] * gx + g[..., 1] * gy

        n00 = grad(aa, xf, yf)
        n10 = grad(ba, xf - 1.0, yf)
        n01 = grad(ab, xf, yf - 1.0)
        n11 = grad(bb, xf - 1.0, yf - 1.0)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        # ~[-1, 1] after the sqrt(2) normalisation of 8-direction gradients.
        return (nx0 + v * (nx1 - nx0)) * np.float32(1.4142)

    def perlin3(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        x, y, z = np.broadcast_arrays(x, y, z)

        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        zi = np.floor(z).astype(np.int64)
        xf = (x - xi).astype(np.float32)
        yf = (y - yi).astype(np.float32)
        zf = (z - zi).astype(np.float32)
        xi &= 255
        yi &= 255
        zi &= 255

        u = _fade(xf)
        v = _fade(yf)
        w = _fade(zf)

        p = self._p
        a = p[xi] + yi
        b = p[xi + 1] + yi
        aa = p[a] + zi
        ab = p[a + 1] + zi
        ba = p[b] + zi
        bb = p[b + 1] + zi

        def grad(h: np.ndarray, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> np.ndarray:
            g = _GRAD3[h % 12]
            return g[..., 0] * gx + g[..., 1] * gy + g[..., 2] * gz

        n000 = grad(p[aa], xf, yf, zf)
        n100 = grad(p[ba], xf - 1.0, yf, zf)
        n010 = grad(p[ab], xf, yf - 1.0, zf)
        n110 = grad(p[bb], xf - 1.0, yf - 1.0, zf)
        n001 = grad(p[aa + 1], xf, yf, zf - 1.0)
        n101 = grad(p[ba + 1], xf - 1.0, yf, zf - 1.0)
        n011 = grad(p[ab + 1], xf, yf - 1.0, zf - 1.0)
        n111 = grad(p[bb + 1], xf - 1.0, yf - 1.0, zf - 1.0)

        nx00 = n000 + u * (n100 - n000)
        nx10 = n010 + u * (n110 - n010)
        nx01 = n001 + u * (n101 - n001)
        nx11 = n011 + u * (n111 - n011)
        nxy0 = nx00 + v * (nx10 - nx00)
        nxy1 = nx01 + v * (nx11 - nx01)
        return (nxy0 + w * (nxy1 - nxy0)) * np.float32(1.1547)

    # -- fractal combinators --------------------------------------------------
    def fbm2(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int = 4,
        lacunarity: float = 2.0,
        gain: float = 0.5,
    ) -> np.ndarray:
        total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
        amp = 1.0
        freq = 1.0
        amp_sum = 0.0
        for _ in range(octaves):
            total += amp * self.perlin2(x * freq, y * freq)
            amp_sum += amp
            amp *= gain
            freq *= lacunarity
        return total / np.float32(amp_sum)

    def fbm3(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        octaves: int = 2,
        lacunarity: float = 2.0,
        gain: float = 0.5,
    ) -> np.ndarray:
        total = np.zeros(np.broadcast(x, y, z).shape, dtype=np.float32)
        amp = 1.0
        freq = 1.0
        amp_sum = 0.0
        for _ in range(octaves):
            total += amp * self.perlin3(x * freq, y * freq, z * freq)
            amp_sum += amp
            amp *= gain
            freq *= lacunarity
        return total / np.float32(amp_sum)

    def ridged2(self, x: np.ndarray, y: np.ndarray, octaves: int = 4) -> np.ndarray:
        """Ridge noise in [0, 1] — sharp mountain crests."""
        total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
        amp = 1.0
        freq = 1.0
        amp_sum = 0.0
        for _ in range(octaves):
            total += amp * (1.0 - np.abs(self.perlin2(x * freq, y * freq)))
            amp_sum += amp
            amp *= 0.5
            freq *= 2.0
        return total / np.float32(amp_sum)


def hash01(seed: int, x: np.ndarray, z: np.ndarray, salt: int = 0) -> np.ndarray:
    """Deterministic per-coordinate uniform value in [0, 1).

    Used for stateless scattering (trees, plants): any chunk can evaluate the
    same decision for a world column without cross-chunk communication.
    """
    h = (
        np.asarray(x, dtype=np.int64) * 374761393
        + np.asarray(z, dtype=np.int64) * 668265263
        + np.int64(seed) * 2147483647
        + np.int64(salt) * 962287
    )
    h = (h ^ (h >> 13)) * 1274126177
    h ^= h >> 16
    return (h & 0x7FFFFFFF).astype(np.float64) / float(0x80000000)
