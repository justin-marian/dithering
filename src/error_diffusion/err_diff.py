# -*- coding: utf-8 -*-
"""Generic error diffusion dithering.

Implements a black-white error diffusion method supporting all kernels
defined in ``DITHERING_KERNELS`` with serpentine scanning and alias resolution.
"""

from __future__ import annotations

from typing import List, Literal, Tuple, Union

import numpy as np

from ..utils.grayscale import binarize, grayscale, map_threshold_graydomain
from .kernels import DITHERING_KERNELS, KERNEL_ALIASES, resolve_kernel_name


def error_diff_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type] = "u8",
    kernel_type: str = "floyd_steinberg",
    threshold: Union[int, float] = 128,
    serpentine: bool = True,
) -> np.ndarray:
    """Apply error diffusion dithering with the specified kernel.

    Parameters
    ----------
    img : np.ndarray
        Input color image (H, W, C). Converted to grayscale internally.
    dtype : {"u8","f32"} | np.dtype | type, optional
        Output dtype for the binarized image. Default "u8".
    kernel_type : str, optional
        Name or alias of the diffusion kernel. Default "floyd_steinberg".
    threshold : int | float, optional
        Global threshold for binarization. Default 128.
    serpentine : bool, optional
        Alternate scan direction per row (True) or always left→right (False).

    Returns
    -------
    np.ndarray
        Dithered 1-bit image mapped to `dtype`.
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
        dy, dx = dy_dx
        max_dy = max(max_dy, dy)
        norm_offsets.append((dy, dx, float(w) / dden))

    # Grayscale buffer in float32 [0..255]
    g = grayscale(img, dtype)  # (H, W)
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

        xs = range(w - 1, -1, -1) if flip else range(0, w)
        for x in xs:
            old = float(g[y, x]) + err_rows[0][x]
            new = 255.0 if old >= thr else 0.0
            dither_img[y, x] = 255 if new > 0.0 else 0
            e = old - new

            # Diffuse error
            for dy, dx, wn in norm_offsets:
                xx = x + (-dx if flip else dx)
                yy = y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    err_rows[dy][xx] += e * wn

        # Roll ring buffer: next row becomes current
        err_rows = err_rows[1:] + err_rows[:1]

    return binarize(dither_img, dtype)
