# -*- coding: utf-8 -*-
"""Global threshold selection for binarization.

This module provides a simple wrapper that maps a user-specified
threshold into the grayscale domain, based on the target dtype.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from utils import map_threshold_graydomain


def global_threshold(
    threshold: Union[int, float],
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type],
) -> int:
    """
    Apply a fixed global threshold to the grayscale domain.

    Parameters
    ----------
    threshold : int or float
        User-specified threshold value (typically in [0, 255]).
    dtype : {"u8", "f32"} or numpy dtype
        Target grayscale dtype for mapping.

    Returns
    -------
    int
        Threshold mapped into the correct gray domain for the given dtype.
    """
    return map_threshold_graydomain(threshold, dtype)
