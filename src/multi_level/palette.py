# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from ..error_diffusion.kernels import (
    DITHERING_KERNELS,
    KERNEL_ALIASES,
    resolve_kernel_name,
)
from ..utils.prep_img import tuple_prepare_img
from .nearest import nearest_color


def palette_bw(
    img: np.ndarray,
    palette: Iterable[Tuple[int, int, int]],
    kernel_type: str = "floyd_steinberg",
    *,
    serpentine: bool = True
) -> np.ndarray:
    """
    General error diffusion dithering to a given RGB palette.
    Input: color image (HxWx3) with uint8 [0..255] values.
    Palette: iterable of (R, G, B) tuples with uint8 [0..255] values.
    Kernel_type: one of the keys in DITHERING_KERNELS.
    Serpentine: whether to alternate scanline direction.
    Output: dithered image (HxWx3) with uint8 [0..255] values from the palette.
    """
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

    # Normalize offsets and find max_dy
    norm_offsets: List[Tuple[int, int, float]] = []
    max_dy = 0
    dden = float(denom)
    for (dy_dx, w) in offsets:
        dy, dx = dy_dx
        max_dy = max(max_dy, dy)
        norm_offsets.append((dy, dx, float(w) / dden))

    rgb, _, _, _ = tuple_prepare_img(img, 'u8')

    # Palette as uint8 (Kx3) array for indexing
    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("Palette must be an iterable of (R,G,B) tuples (Kx3).")
    palette_f32 = palette.astype(np.float32, copy=False)

    h, w, _ = rgb.shape
    dither_img = np.empty((h, w, 3), dtype=np.uint8)
    # Error may rise above 255 or below 0, so use float32 (ring buffer for rows)
    err_rows = [np.zeros((w, 3), dtype=np.float32) for _ in range(max_dy + 1)]

    for y in range(h):
        flip = serpentine and (y & 1)
        for dy in range(1, max_dy + 1):
            err_rows[dy].fill(0.0)

        x_iter = range(w - 1, -1, -1) if flip else range(0, w)

        for x in x_iter:
            old = rgb[y, x] + err_rows[0][x]
            new_col_f32, _ = nearest_color(old, palette_f32)
            new_col_u8 = new_col_f32.astype(np.uint8)
            dither_img[y, x] = new_col_u8

            # Diffuse RGB error
            e = old - new_col_f32
            for dy, dx, wn in norm_offsets:
                xx = x + (-dx if flip else dx)
                yy = y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    err_rows[dy][xx] += e * wn

        # Rotate ring: next row becomes current
        err_rows = err_rows[1:] + err_rows[:1]

    return dither_img
