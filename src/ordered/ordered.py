# -*- coding: utf-8 -*-
"""Ordered dithering algorithms.

This module provides ordered dithering methods, including:
- Bayer matrix dithering of configurable size
- Halftone dithering with spot functions and arbitrary angles
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from group.bayer import bayer_bw, bayer_matrix
from group.halftone import halftone_bw


def ordered_bw(
    img: np.ndarray,
    *,
    kind: Literal["bayer", "halftone"] = "bayer",
    n: int = 8,
    angle_deg: float = 45.0,
    spot: Literal["cos+cos", "cosx", "cosx+2cosy"] = "cos+cos",
    dtype: Union[Literal["u8", "f32"], np.dtype, type] = "u8",
) -> np.ndarray:
    """
    Apply ordered dithering to an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or RGB).
    kind : {"bayer", "halftone"}, default="bayer"
        Type of ordered dithering to apply.
    n : int, default=8
        Size of Bayer matrix or halftone tile (must be power of two for Bayer).
    angle_deg : float, default=45.0
        Halftone angle in degrees (only used if kind="halftone").
    spot : {"cos+cos", "cosx", "cosx+2cosy"}, default="cos+cos"
        Spot function used for halftone dithering.
    dtype : {"u8", "f32"} or np.dtype or type, default="u8"
        Output data type.

    Returns
    -------
    np.ndarray
        Dithered image with values mapped to {0, 255} in the requested dtype.

    Raises
    ------
    ValueError
        If `kind` is not one of {"bayer", "halftone"}.
    """
    if kind == "bayer":
        return bayer_bw(img, matrix=bayer_matrix(n), dtype=dtype)
    if kind == "halftone":
        return halftone_bw(img, size=n, angle_deg=angle_deg, spot=spot, dtype=dtype)
    raise ValueError("kind must be 'bayer' or 'halftone'")
