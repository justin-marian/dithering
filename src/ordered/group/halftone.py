# -*- coding: utf-8 -*-
"""Halftone (spot-function) ordered dithering."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from spot import spot_threshold

from utils import binarize, grayscale


def halftone_bw(
    img: np.ndarray,
    *,
    size: int = 8,
    angle_deg: float = 45.0,
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type] = "u8",
    spot: Literal["cos+cos", "cosx", "cosx+2cosy"] = "cos+cos",
) -> np.ndarray:
    """
    Apply halftone ordered dithering (spot function) to a grayscale or RGB image.

    Parameters
    ----------
    img : np.ndarray
        Input image (H, W) or (H, W, C). Any dtype supported by `grayscale`.
    size : int, default=8
        Halftone tile size in pixels (must be >= 2).
    angle_deg : float, default=45.0
        Halftone angle in degrees.
    dtype : {"u8","f32"} | np.dtype | type, default="u8"
        Output dtype. 'u8' → {0,255}; float dtypes → {0.0,1.0}; other ints → {0,max(dtype)}.
    spot : {"cos+cos", "cosx", "cosx+2cosy"}, default="cos+cos"
        Spot function used to build the threshold tile.

    Returns
    -------
    np.ndarray
        Binarized image with values mapped according to `dtype`.

    Raises
    ------
    ValueError
        If `size` < 2.
    """
    if size < 2:
        raise ValueError("Halftone tile size must be >= 2")

    g = grayscale(img, dtype)
    height, width = g.shape

    # Tile blocks to cover the image area
    tile = spot_threshold(size=size, angle_deg=angle_deg, spot=spot)
    width_blocks = (width + size - 1) // size
    height_blocks = (height + size - 1) // size

    # Tile and crop to image size; scale thresholds to gray domain [0..255]
    T = np.tile(tile, (height_blocks, width_blocks))[:height, :width] * 255.0
    dither_img = np.where(g >= T, 255, 0).astype(np.uint8)
    return binarize(dither_img, dtype)
