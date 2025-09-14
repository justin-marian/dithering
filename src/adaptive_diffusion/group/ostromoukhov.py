# -*- coding: utf-8 -*-
"""Ostromoukhov (2001) variable-coefficient error diffusion (black-white).

Implements a 3-tap stencil with gray-dependent weights for right (E), down (D),
and diagonal (SE/SW) diffusion. See:

References
----------
- Ostromoukhov, "A Simple and Efficient Error-Diffusion Algorithm", 2001.
"""

from __future__ import annotations

import pathlib
from typing import Literal, Union

import numpy as np
from kernels import load_ostro_coeffs

from utils import binarize, grayscale, map_threshold_graydomain

_THIS_FILE = pathlib.Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_AD_DIR = _THIS_DIR.parent
_DATA_DIR = _AD_DIR / "kernels" / "data"

_OSTRO_TXT = _DATA_DIR / "weights_ostromoukhov.txt"
if not _OSTRO_TXT.exists():
    raise FileNotFoundError(f"Not found: {_OSTRO_TXT}")

__OSTRO_COEFFS = load_ostro_coeffs(_OSTRO_TXT, dtype=np.float32)


def ostromoukhov_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type] = "u8",
    threshold: Union[int, float] = 128,
    serpentine: bool = True,
) -> np.ndarray:
    """Apply Ostromoukhov variable-coefficient error diffusion to an image.

    Parameters
    ----------
    img : np.ndarray
        Input color image (H, W, C). Will be converted to grayscale internally.
    dtype : {"u8","f32"} | np.dtype | type, optional
        Output dtype for the binarized image. Default "u8".
    threshold : int | float, optional
        Global threshold in gray domain (0..255 mapping). Default 128.
    serpentine : bool, optional
        Alternate scan direction per row (True) or always left→right (False).

    Returns
    -------
    np.ndarray
        Dithered 1-bit image mapped to requested dtype via `binarize`.
    """
    g_base = grayscale(img, dtype)
    h, w = g_base.shape
    thr = map_threshold_graydomain(threshold, dtype)

    out_u8 = np.empty((h, w), dtype=np.uint8)
    err_curr = np.zeros(w, dtype=np.float32)
    err_next = np.zeros(w, dtype=np.float32)

    n_ostro = __OSTRO_COEFFS.shape[0]

    for y in range(h):
        flip = serpentine and (y & 1)
        err_next.fill(0.0)

        xs = range(w - 1, -1, -1) if flip else range(0, w)
        for x in xs:
            g0 = g_base[y, x]
            idx = int(g0 + 0.5)
            # Clamp to table range
            if idx < 0:
                idx = 0
            elif idx > 255:
                idx = 255
            if idx >= n_ostro:
                idx = n_ostro - 1

            w_r, w_dl, w_d = __OSTRO_COEFFS[idx]  # right, diag, down

            # Diffuse quantization error
            old = g_base[y, x] + err_curr[x]
            new = 255.0 if old >= thr else 0.0
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
