# -*- coding: utf-8 -*-
"""Mean threshold selection for binarization.

This module computes a threshold as the mean grayscale intensity
of the input image.
"""

from __future__ import annotations

import numpy as np


def mean_threshold(
    gray_f32: np.ndarray
) -> float:
    """
    Compute a global threshold using the mean intensity.

    Parameters
    ----------
    gray_f32 : np.ndarray
        2D array of grayscale pixel intensities in float32.

    Returns
    -------
    float
        Mean intensity value, used as threshold.
    """
    return float(np.mean(gray_f32))
