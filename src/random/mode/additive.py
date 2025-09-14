# -*- coding: utf-8 -*-
"""Additive noise-based dithering mode.

This module implements additive random dithering, where noise is
added to the grayscale image before thresholding.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np


def addition(
    g: np.ndarray,
    noise: np.ndarray,
    thr: float,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type]
) -> np.ndarray:
    """
    Apply additive-noise dithering to a grayscale image.

    Parameters
    ----------
    g : np.ndarray
        Input grayscale image (float or uint8).
    noise : np.ndarray
        Noise array of the same shape as `g`.
    thr : float
        Threshold value for binarization.
    dtype : dtype | type | {'u8', 'f32'}
        Output data type.

    Returns
    -------
    np.ndarray
        Binarized image with additive noise applied.
    """
    comp = g + noise
    out_u8 = np.where(comp >= thr, 255, 0).astype(np.uint8)
    return out_u8.astype(dtype)
