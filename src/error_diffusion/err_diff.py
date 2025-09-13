# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Literal, Tuple, Union

import numpy as np

from ..utils.grayscale import binarize, grayscale, map_threshold_graydomain
from .kernels import DITHERING_KERNELS, KERNEL_ALIASES, resolve_kernel_name


def error_diff_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8',
    kernel_type: str = "floyd_steinberg",
    threshold: Union[int, float] = 128,
    serpentine: bool = True
) -> np.ndarray:
    """
    Generic error-diffusion to 1-bit using any kernel from DITHERING_KERNELS.

    Kernel format:
      DITHERING_KERNELS[name] = (offsets, denom)
      offsets: [ ((dy, dx), weight), ... ]  with dy >= 0 and dx not in Z
      denom:    kernel divisor (often the weight sum)
    """
    kname = resolve_kernel_name(kernel_type)
    if kname not in DITHERING_KERNELS:
        raise ValueError(
            f"Unsupported kernel '{kname}'. "
            f"Supported: {list(DITHERING_KERNELS.keys()) + list(KERNEL_ALIASES.keys())}"
        )

    offsets, denom = DITHERING_KERNELS[kname]
    
    # Check causality (our scan is top-down; dy<0 would target already-processed rows)
    if any(dy_dx[0] < 0 for dy_dx, _ in offsets):
        raise ValueError(
            f"Kernel '{kname}' is non-causal (contains dy < 0). "
            f"Use a causal variant (dy >= 0) or a different scan strategy."
        )

    # Normalize offsets and find max_dy
    norm_offsets: List[Tuple[int, int, float]] = []
    max_dy = 0
    dden = float(denom)
    for (dy_dx, w) in offsets:
        dy, dx = dy_dx  # (row, col)
        max_dy = max(max_dy, dy)
        norm_offsets.append((dy, dx, float(w) / dden))

    # Grayscale buffer in float32 [0..255]
    g = grayscale(img, dtype)     # (H, W)
    h, w = g.shape
    thr = map_threshold_graydomain(threshold, dtype)

    # Output as u8 {0,255} then format to requested dtype
    dither_img = np.empty((h, w), dtype=np.uint8)
    # Error ring buffers: err_rows[0] is current row; err_rows[dy] future rows
    err_rows = [np.zeros(w, dtype=np.float32) for _ in range(max_dy + 1)]

    for y in range(h):
        flip = serpentine and (y & 1)
        for dy in range(1, max_dy + 1):
            err_rows[dy].fill(0.0)

        x_iter = range(w - 1, -1, -1) if flip else range(0, w)

        for x in x_iter:
            old = float(g[y, x]) + err_rows[0][x]
            new = 255.0 if old >= thr else 0.0
            dither_img[y, x] = 255 if new > 0.0 else 0
            e = old - new

            # diffuse error
            for dy, dx, wn in norm_offsets:
                xx = x + (-dx if flip else dx)
                yy = y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    err_rows[dy][xx] += e * wn

        # Roll ring buffer: next row becomes current
        err_rows = err_rows[1:] + err_rows[:1]

    return binarize(dither_img, dtype)
