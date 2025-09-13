# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from ...utils import binarize, grayscale
from .spot import spot_threshold


def halftone_bw(
    img: np.ndarray,
    *,
    size: int = 8,
    angle_deg: float = 45.0,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8',
    spot: Literal["cos+cos", "cosx", "cosx+2cosy"] = "cos+cos"
) -> np.ndarray:
    g = grayscale(img, dtype)
    hight, width = g.shape

    # Tiles blocks to cover the image area
    tile = spot_threshold(size=size, angle_deg=angle_deg, spot=spot)
    width_blocks = (width + size - 1) // size
    height_blocks = (hight + size - 1) // size

    # Tile and crop to image size
    T = np.tile(tile, (height_blocks, width_blocks))[:hight, :width] * 255.0
    dither_img = np.where(g >= T, 255, 0).astype(np.uint8)
    return binarize(dither_img, dtype)
