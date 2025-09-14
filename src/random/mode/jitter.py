# -*- coding: utf-8 -*-
"""Jitter noise-based dithering mode.

This module implements threshold jittering for random dithering,
where noise perturbs the comparison threshold before binarization.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np


def jitter(
    img: np.ndarray,
    noise: np.ndarray,
    thr: int,
    dtype: Union[np.dtype, type, Literal['u8'], Literal['f32']]
) -> np.ndarray:
    """
    Apply jitter-based dithering to a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (float or uint8).
    noise : np.ndarray
        Noise array of the same shape as `img`.
    thr : int
        Base threshold value before noise perturbation.
    dtype : dtype | type | {'u8', 'f32'}
        Output data type.

    Returns
    -------
    np.ndarray
        Binarized image with threshold jittering applied.
    """
    comp_thr = thr + noise
    out = np.where(img >= comp_thr, 255, 0)
    return out.astype(dtype)
