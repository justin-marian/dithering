# -*- coding: utf-8 -*-

from __future__ import annotations

import pathlib
from typing import Literal, Union

import numpy as np

from ...utils import binarize, grayscale
from ..kernels.load_kernel import load_ostro_coeffs, load_zf_strength

_THIS_FILE = pathlib.Path(__file__).resolve()
_THIS_DIR  = _THIS_FILE.parent
_AD_DIR    = _THIS_DIR.parent
_DATA_DIR  = _AD_DIR / "kernels" / "data" 

_OSTRO_TXT = _DATA_DIR / "weights_ostromoukhov.txt"
_ZF_TXT    = _DATA_DIR / "strengths_zhou_fang.txt"

if not _OSTRO_TXT.exists():
    raise FileNotFoundError(f"Not found: {_OSTRO_TXT}")
if not _ZF_TXT.exists():
    raise FileNotFoundError(f"Not found: {_ZF_TXT}")

__OSTRO_COEFFS = load_ostro_coeffs(_OSTRO_TXT, dtype=np.float32)
__ZF_STRENGTH  = load_zf_strength(_ZF_TXT, dtype=np.float32)

def zhou_fang_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8',
    serpentine: bool = True,
    noise_scale: float = 1.0,
    seed: Union[int, None] = None
) -> np.ndarray:
    """
    Zhou-Fang 2007 variable-threshold error diffusion.
    Uses Ostromoukhov [wR, wDL, wD] for diffusion and a scalar jitter strength
    derived from the provided (N,3) kernel file (dr, dc, w).

    Parameters
        :E: error to the right
        :D: error directly below
        :DL: error diagonally below-right
        :SW: error diagonally below-left (if serpentine)

    Arguments
        :serpentine: if True, alternate left-to-right / right-to-left scan per row.
        :noise_scale: scales the amount of threshold jittering (default 1.0).
        :seed: random seed for jittering (default None, non-deterministic).

    Returns a 1-bit image mapped to the requested dtype via `binarize`.
    References
    - Zhou & Fang, "A New Error Diffusion Algorithm with Improved Visual Quality", 2007.
    """
    rng = np.random.default_rng(seed)
    g_base = grayscale(img, dtype)
    h, w = g_base.shape

    out_u8 = np.empty((h, w), dtype=np.uint8)
    err_curr = np.zeros(w, dtype=np.float32)
    err_next = np.zeros(w, dtype=np.float32)

    n_strength = __ZF_STRENGTH.shape[0]
    n_ostro    = __OSTRO_COEFFS.shape[0]

    jitter_strength = 0.0
    for i in range(n_strength):
        si = i if i < __ZF_STRENGTH.shape[0] else (__ZF_STRENGTH.shape[0] - 1)
        s = float(__ZF_STRENGTH[si])
        if s > 0:
            jitter_strength = s
            break

    for y in range(h):
        flip = serpentine and (y & 1)
        err_next.fill(0.0)
        x_iter = range(w - 1, -1, -1) if flip else range(0, w)

        for x in x_iter:
            g0 = g_base[y, x]
            idx = int(g0 + 0.5)

            # Clamp index to valid range
            idx = 0 if idx < 0 else (255 if idx > 255 else idx)
            idx = min(idx, n_ostro - 1)
            wR, wDL, wD = __OSTRO_COEFFS[idx]

            # ZF-style threshold jitter in [128..256), scaled by strength and noise_scale
            T = 128.0 + (rng.random() * 128.0) * jitter_strength * float(noise_scale)

            # Error diffusion step
            old = g0 + err_curr[x]
            new = 255.0 if old >= T else 0.0
            out_u8[y, x] = 255 if new > 0.0 else 0
            e = old - new

            if not flip:
                # E, D, SE (mirror dx only)
                if x + 1 < w: 
                    err_curr[x + 1] += e * wR   # E
                err_next[x] += e * wD                      # D
                if x + 1 < w: 
                    err_next[x + 1] += e * wDL   # SE
            else:
                # W, D, SW  (mirror dx only)
                if x - 1 >= 0: 
                    err_curr[x - 1] += e * wR   # W
                err_next[x] += e * wD                      # D
                if x - 1 >= 0:
                    err_next[x - 1] += e * wDL  # SW

        # Rotate ring: next row becomes current
        err_curr, err_next = err_next, err_curr

    return binarize(out_u8, dtype)
