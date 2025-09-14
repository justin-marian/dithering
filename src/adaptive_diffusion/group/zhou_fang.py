# -*- coding: utf-8 -*-
"""Zhou-Fang (2007) variable-threshold error diffusion (black-white).

Uses Ostromoukhov's gray-dependent diffusion weights and adds a threshold
jitter proportional to a precomputed strength table.

References
----------
- Zhou & Fang, "A New Error Diffusion Algorithm with Improved Visual Quality", 2007.
"""

from __future__ import annotations

import pathlib
from typing import Literal, Union

import numpy as np

from ...utils.grayscale import binarize, grayscale
from ..kernels import load_ostro_coeffs, load_zf_strength

_THIS_FILE = pathlib.Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_AD_DIR = _THIS_DIR.parent
_DATA_DIR = _AD_DIR / "kernels" / "data"

_OSTRO_TXT = _DATA_DIR / "weights_ostromoukhov.txt"
_ZF_TXT = _DATA_DIR / "strengths_zhou_fang.txt"

if not _OSTRO_TXT.exists():
    raise FileNotFoundError(f"Not found: {_OSTRO_TXT}")
if not _ZF_TXT.exists():
    raise FileNotFoundError(f"Not found: {_ZF_TXT}")

__OSTRO_COEFFS = load_ostro_coeffs(_OSTRO_TXT, dtype=np.float32)
__ZF_STRENGTH = load_zf_strength(_ZF_TXT, dtype=np.float32)


def zhou_fang_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type] = "u8",
    serpentine: bool = True,
    noise_scale: float = 1.0,
    seed: Union[int, None] = None,
) -> np.ndarray:
    """Apply Zhou–Fang variable-threshold error diffusion.

    Parameters
    ----------
    img : np.ndarray
        Input color image (H, W, C). Converted to grayscale internally.
    dtype : {"u8","f32"} | np.dtype | type, optional
        Output dtype for the binarized image. Default "u8".
    serpentine : bool, optional
        Alternate scan direction per row (True) or always left→right (False).
    noise_scale : float, optional
        Scales the threshold jitter amplitude. Default 1.0.
    seed : int | None, optional
        RNG seed for reproducible jitter, default None.

    Returns
    -------
    np.ndarray
        Dithered 1-bit image mapped to `dtype` via `binarize`.
    """
    rng = np.random.default_rng(seed)
    g_base = grayscale(img, dtype)
    h, w = g_base.shape

    out_u8 = np.empty((h, w), dtype=np.uint8)
    err_curr = np.zeros(w, dtype=np.float32)
    err_next = np.zeros(w, dtype=np.float32)

    n_ostro = __OSTRO_COEFFS.shape[0]

    # First positive strength value, or 0.0 if none found
    jitter_strength = float(next((float(s) for s in __ZF_STRENGTH.flat if s > 0), 0.0))

    for y in range(h):
        flip = serpentine and (y & 1)
        err_next.fill(0.0)

        xs = range(w - 1, -1, -1) if flip else range(0, w)
        for x in xs:
            g0 = g_base[y, x]
            idx = int(g0 + 0.5)
            # Clamp index to valid range and table size
            if idx < 0:
                idx = 0
            elif idx > 255:
                idx = 255
            if idx >= n_ostro:
                idx = n_ostro - 1

            w_r, w_dl, w_d = __OSTRO_COEFFS[idx]  # right, diag, down

            # ZF-style threshold jitter in [128..256), scaled by strength and noise_scale
            threshold_t = 128.0 + (rng.random() * 128.0) * jitter_strength * float(noise_scale)

            # Error diffusion step
            old = g0 + err_curr[x]
            new = 255.0 if old >= threshold_t else 0.0
            out_u8[y, x] = 255 if new > 0.0 else 0
            e = old - new

            if not flip:
                # E, D, SE
                if x + 1 < w:
                    err_curr[x + 1] += e * w_r  # E
                    err_next[x + 1] += e * w_dl  # SE
                err_next[x] += e * w_d  # D
            else:
                # W, D, SW
                if x - 1 >= 0:
                    err_curr[x - 1] += e * w_r  # W
                    err_next[x - 1] += e * w_dl  # SW
                err_next[x] += e * w_d  # D

        # Next row becomes current
        err_curr, err_next = err_next, err_curr

    return binarize(out_u8, dtype)
