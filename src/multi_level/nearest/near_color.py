# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def nearest_color(
    pixel: Iterable[float], 
    palette: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Find the nearest color in the palette to the given pixel.
    Pixel is a 1D array-like of shape (3,) or (4,) for RGB or RGBA.
    Palette is a 2D array of shape (N, 3) or (N, 4).
    Returns the nearest color from the palette.
    """
    pixel_arr = np.asarray(pixel, dtype=np.float32)
    palette_arr = np.asarray(palette, dtype=np.float32)

    if pixel_arr.ndim != 1 or pixel_arr.size not in (3, 4):
        raise ValueError("Pixel must be a 1D array-like of shape (3,) or (4,).")
    if palette_arr.ndim != 2 or palette_arr.shape[1] not in (3, 4):
        raise ValueError("Palette must be a 2D array of shape (N, 3) or (N, 4).")

    if pixel_arr.size != palette_arr.shape[1]:
        raise ValueError("Pixel and palette color dimensions must match.")

    diffs = palette_arr - pixel_arr
    dists = np.einsum('ij,ij->i', diffs, diffs)
    nearest_index = np.argmin(dists)
    return palette_arr[nearest_index], nearest_index
