# -*- coding: utf-8 -*-
"""Adaptive diffusion dithering interface.

Provides a unified function `adaptive_bw` that dispatches to
Ostromoukhov or Zhou-Fang adaptive diffusion implementations.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from .group import ostromoukhov_bw, zhou_fang_bw


def adaptive_diff_bw(
    img: np.ndarray,
    *,
    method: Literal["ostromoukhov", "zhou_fang"] = "ostromoukhov",
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type] = "u8",
    threshold: Union[int, float] = 128,
    serpentine: bool = True,
    noise_scale: float = 1.0,
    seed: Union[int, None] = None,
) -> np.ndarray:
    """Apply adaptive diffusion dithering.

    Parameters
    ----------
    img : np.ndarray
        Input color image (H, W, C).
    method : {"ostromoukhov", "zhou_fang"}, optional
        Which adaptive diffusion algorithm to use.
    dtype : str | np.dtype | type, optional
        Output dtype, defaults to "u8".
    threshold : int | float, optional
        Threshold value for binarization.
    serpentine : bool, optional
        Whether to use serpentine scanning.
    noise_scale : float, optional
        Scaling factor for random noise (only used in Zhou-Fang).
    seed : int | None, optional
        RNG seed for reproducibility (only used in Zhou-Fang).

    Returns
    -------
    np.ndarray
        Dithered grayscale image, dtype determined by `dtype`.
    """
    if method == "ostromoukhov":
        return ostromoukhov_bw(
            img, dtype=dtype, threshold=threshold, serpentine=serpentine
        )
    if method == "zhou_fang":
        return zhou_fang_bw(
            img,
            dtype=dtype,
            serpentine=serpentine,
            noise_scale=noise_scale,
            seed=seed,
        )
    raise ValueError("method must be 'ostromoukhov' or 'zhou_fang'")
