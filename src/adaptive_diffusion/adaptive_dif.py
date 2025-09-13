# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from .group import ostromoukhov_bw, zhou_fang_bw


def adaptive_bw(
    img: np.ndarray,
    *,
    method: Literal["ostromoukhov", "zhou_fang"] = "ostromoukhov",
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8',
    threshold: Union[int, float] = 128,
    serpentine: bool = True,
    noise_scale: float = 1.0,
    seed: Union[int, None] = None
) -> np.ndarray:
    """
    Adaptive diffusion dithering using specified method.
    Given a color image, applies the chosen adaptive diffusion dithering method
    and returns the result.
    It operates on grayscale images internally.
    """
    if method == "ostromoukhov":
        return ostromoukhov_bw(
            img, dtype=dtype, threshold=threshold, serpentine=serpentine
        )
    if method == "zhou_fang":
        return zhou_fang_bw(
            img, dtype=dtype, serpentine=serpentine,
            noise_scale=noise_scale, seed=seed
        )
    raise ValueError("method must be 'ostromoukhov' or 'zhou_fang'")
