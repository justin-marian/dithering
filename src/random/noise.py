# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np

from ..utils import binarize, grayscale, map_threshold_graydomain
from .group.normal import normal_distribution
from .group.uniform import uniform_distrib
from .mode.additive import addition
from .mode.jitter import jitter


def random_bw(
    img: np.ndarray,
    *,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8',
    threshold: Union[int, float] = 128,
    mode: Literal["additive", "jitter"] = "additive",
    distribution: Literal["uniform", "normal"] = "uniform",
    amount: float = 0.05,
    seed: Optional[int] = None
) -> np.ndarray:
    g = grayscale(img, dtype)
    thr = map_threshold_graydomain(threshold, dtype)
    A = float(amount) * 255.0
    h, w = g.shape

    if distribution == "uniform":
        noise = uniform_distrib(-A, A, size=(h, w), seed=seed)
    elif distribution == "normal":
        noise = normal_distribution(0.0, A / 2.0, size=(h, w), seed=seed)
    else:
        raise ValueError("distribution must be 'uniform' or 'normal'")

    if mode == "additive":
        out = addition(g, noise, dtype=dtype)
    elif mode == "jitter":
        out = jitter(g, noise, thr, dtype=dtype)
    else:
        raise ValueError("mode must be 'additive' or 'jitter'")

    return binarize(out, dtype)
