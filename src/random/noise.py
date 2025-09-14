# -*- coding: utf-8 -*-
"""Random/noise-based dithering.

Implements two modes:
- additive: add noise to grayscale then threshold
- jitter: jitter the threshold per pixel using noise
Supports uniform and normal noise distributions.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np

from ..utils.grayscale import binarize, grayscale, map_threshold_graydomain
from .group.normal import normal_distribution
from .group.uniform import uniform_distrib
from .mode.additive import addition
from .mode.jitter import jitter


def random_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type] = "u8",
    threshold: Union[int, float] = 128,
    mode: Literal["additive", "jitter"] = "additive",
    distribution: Literal["uniform", "normal"] = "uniform",
    amount: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply noise-based dithering (additive or threshold jitter).

    Parameters
    ----------
    img : np.ndarray
        Input image (H, W) or (H, W, C). Any dtype supported by `grayscale`.
    dtype : {"u8","f32"} | np.dtype | type, default "u8"
        Output dtype for the binarized image.
    threshold : int | float, default 128
        Global threshold in gray domain; used directly for `mode="additive"`,
        and as the base for jitter comparisons for `mode="jitter"`.
    mode : {"additive", "jitter"}, default "additive"
        - "additive": add noise to the grayscale values then threshold
        - "jitter": keep grayscale fixed and jitter the threshold per pixel
    distribution : {"uniform", "normal"}, default "uniform"
        Noise distribution:
        - "uniform": U(-A, A)
        - "normal":  N(0, (A/2)^2)   (std = A/2)
    amount : float, default 0.05
        Noise amplitude as a fraction of the 8-bit range (0..1 maps to 0..255).
        E.g., 0.05 → amplitude ≈ 12.75.
    seed : int | None, default None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Binarized image mapped to `dtype`.

    Raises
    ------
    ValueError
        If `distribution` or `mode` is invalid, or `amount` is negative.
    """
    if amount < 0:
        raise ValueError("amount must be >= 0")

    g = grayscale(img, dtype)
    thr = map_threshold_graydomain(threshold, dtype)
    noise_amp = float(amount) * 255.0
    h, w = g.shape

    if distribution == "uniform":
        noise = uniform_distrib(-noise_amp, noise_amp, size=(h, w), seed=seed)
    elif distribution == "normal":
        noise = normal_distribution(0.0, noise_amp / 2.0, size=(h, w), seed=seed)
    else:
        raise ValueError("distribution must be 'uniform' or 'normal'")

    if mode == "additive":
        out = addition(g, noise, thr, dtype=dtype)
    elif mode == "jitter":
        out = jitter(g, noise, thr, dtype=dtype)
    else:
        raise ValueError("mode must be 'additive' or 'jitter'")

    return binarize(out, dtype)
