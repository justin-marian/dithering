# -*- coding: utf-8 -*-
# 

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from ..utils import binarize, grayscale
from .group import global_threshold, mean_threshold, otsu_threshold, percentile_threshold


def threshold_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8',
    threshold: Union[int, float] = 128,
    method: Literal["global", "mean", "percentile", "otsu"] = "global",
    percentile: float = 50.0
) -> np.ndarray:
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
    dither_img = binarize(dither_img, dtype)
    return dither_img
