# -*- coding: utf-8 -*-
"""Percentile-based thresholding for naive dithering.

This module implements a simple thresholding method where the
threshold value is chosen as a given percentile of the grayscale
intensity distribution.
"""

from __future__ import annotations

import numpy as np


def percentile_threshold(
    gray_f32: np.ndarray,
    percentile: float
) -> float:
    """
    Compute a threshold based on a percentile of grayscale values.

    Parameters
    ----------
    gray_f32 : np.ndarray
        2D array of grayscale pixel intensities in float32 or float64.
    percentile : float
        Percentile in [0, 100] to compute as the threshold.

    Returns
    -------
    float
        The intensity value corresponding to the requested percentile.
    """
    return float(np.percentile(gray_f32, percentile))
