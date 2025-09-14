# -*- coding: utf-8 -*-
"""Naive threshold-based dithering methods.

Implements global, mean, percentile, and Otsu thresholds for converting
an image to black-white without error diffusion.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from ..utils.grayscale import binarize, grayscale
from .group import global_threshold, mean_threshold, otsu_threshold, percentile_threshold


def threshold_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type] = "u8",
    threshold: Union[int, float] = 128,
    method: Literal["global", "mean", "percentile", "otsu"] = "global",
    percentile: float = 50.0,
) -> np.ndarray:
    """
    Apply a naive thresholding method to produce a binary (black-white) image.

    Parameters
    ----------
    img : np.ndarray
        Input image (H, W, C) or (H, W). Any dtype supported by `grayscale`.
    dtype : {"u8", "f32"} | np.dtype | type, default "u8"
        Output dtype for the binarized image.
    threshold : int | float, default 128
        Global threshold used when `method="global"`. Interpreted in gray domain.
    method : {"global", "mean", "percentile", "otsu"}, default "global"
        Threshold selection strategy.
    percentile : float, default 50.0
        Percentile used when `method="percentile"` (0..100).

    Returns
    -------
    np.ndarray
        Binary image where pixels >= threshold are 255 and others are 0, then
        mapped to `dtype` via `binarize`.

    Raises
    ------
    ValueError
        If `method` is not one of the supported options.
    """
    g = grayscale(img, dtype)

    if method == "global":
        thr = global_threshold(threshold, dtype)
    elif method == "mean":
        thr = mean_threshold(g)
    elif method == "percentile":
        thr = percentile_threshold(g, percentile)
    elif method == "otsu":
        thr = otsu_threshold(g.astype(np.uint8))
    else:
        raise ValueError("method must be one of: 'global', 'mean', 'percentile', 'otsu'")

    dither_img = np.where(g >= thr, 255, 0).astype(np.uint8)
    return binarize(dither_img, dtype)
