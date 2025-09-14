# -*- coding: utf-8 -*-
"""Otsu's method for threshold selection.

This module implements Otsu's algorithm to automatically determine
an optimal threshold value by maximizing the between-class variance
of grayscale pixel intensities.
"""

from __future__ import annotations

import numpy as np


def otsu_threshold(
    gray_u8: np.ndarray
) -> int:
    """
    Compute a global threshold using Otsu's method.

    Parameters
    ----------
    gray_u8 : np.ndarray
        2D array of grayscale pixel intensities in uint8.

    Returns
    -------
    int
        Optimal threshold value in [0, 255].
    """
    # Histogram of grayscale values
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    cum_count = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))

    # Between-class variance for all t in [0..254]
    w0 = cum_count[:-1]
    w1 = total - w0

    mu0 = np.zeros_like(w0)
    mu1 = np.zeros_like(w0)
    mask = (w0 > 0) & (w1 > 0)
    mu0[mask] = cum_mean[:-1][mask] / w0[mask]
    mu1[mask] = (cum_mean[-1] - cum_mean[:-1][mask]) / w1[mask]
    sigma_b2 = (w0 * w1) * (mu0 - mu1) ** 2

    return int(np.argmax(sigma_b2))
