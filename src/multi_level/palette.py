# -*- coding: utf-8 -*-
"""Palette-based error diffusion dithering.

This module provides an implementation of error diffusion dithering
to a fixed RGB palette using canonical kernels such as Floyd-Steinberg.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from nearest import nearest_color

from error_diffusion.kernels import (
    DITHERING_KERNELS,
    KERNEL_ALIASES,
    resolve_kernel_name,
)
from utils import tuple_prepare_img


def normalize_kernel(
    kernel_type: str,
) -> Tuple[List[Tuple[int, int, float]], int]:
    """Resolve and normalize a diffusion kernel."""
    kname = resolve_kernel_name(kernel_type)
    if kname not in DITHERING_KERNELS:
        raise ValueError(
            f"Unsupported kernel '{kernel_type}'. "
            f"Supported: {list(DITHERING_KERNELS.keys()) + list(KERNEL_ALIASES.keys())}"
        )
    offsets, denom = DITHERING_KERNELS[kname]
    if any(dy_dx[0] < 0 for dy_dx, _ in offsets):
        raise ValueError(
            f"Kernel '{kname}' is non-causal (contains dy < 0). "
            f"Use a causal variant (dy >= 0) or a different traversal."
        )

    norm_offsets: List[Tuple[int, int, float]] = []
    max_dy = 0
    dden = float(denom)
    for (dy_dx, w) in offsets:
        dy, dx = dy_dx
        max_dy = max(max_dy, dy)
        norm_offsets.append((dy, dx, float(w) / dden))
    return norm_offsets, max_dy


def validate_palette(palette: Iterable[Tuple[int, int, int]]) -> np.ndarray:
    """Validate and convert the palette to a (K,3) float32 array."""
    arr = np.asarray(palette, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Palette must be an iterable of (R,G,B) tuples (Kx3).")
    return arr.astype(np.float32, copy=False)


def palette_bw(
    img: np.ndarray,
    palette: Iterable[Tuple[int, int, int]],
    kernel_type: str = "floyd_steinberg",
    *,
    serpentine: bool = True,
) -> np.ndarray:
    """
    Apply palette-based error diffusion dithering.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image (H, W, 3), dtype uint8.
    palette : Iterable[Tuple[int, int, int]]
        List of palette entries (R, G, B) with values in [0, 255].
    kernel_type : str, default "floyd_steinberg"
        The diffusion kernel name or alias.
    serpentine : bool, default True
        Whether to alternate scanline direction.

    Returns
    -------
    np.ndarray
        Dithered RGB image (H, W, 3) using colors from the palette.
    """
    norm_offsets, max_dy = normalize_kernel(kernel_type)
    rgb, _, _, _ = tuple_prepare_img(img, "u8")
    palette_f32 = validate_palette(palette)

    h, w, _ = rgb.shape
    dither_img = np.empty((h, w, 3), dtype=np.uint8)
    err_rows = [np.zeros((w, 3), dtype=np.float32) for _ in range(max_dy + 1)]

    for y in range(h):
        flip = serpentine and (y & 1)
        for dy in range(1, max_dy + 1):
            err_rows[dy].fill(0.0)

        xs = range(w - 1, -1, -1) if flip else range(0, w)
        for x in xs:
            old = rgb[y, x] + err_rows[0][x]
            new_col_f32, _ = nearest_color(old, palette_f32)
            new_col_u8 = new_col_f32.astype(np.uint8)
            dither_img[y, x] = new_col_u8

            e = old - new_col_f32
            for dy, dx, wn in norm_offsets:
                xx = x + (-dx if flip else dx)
                yy = y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    err_rows[dy][xx] += e * wn

        err_rows = err_rows[1:] + err_rows[:1]

    return dither_img
