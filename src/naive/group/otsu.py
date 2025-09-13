# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np


def otsu_threshold(
    gray_u8: np.ndarray
) -> int:
    # Histogram obtained through the total number of pixels for each intensity level.
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    cum_count = np.cumsum(hist)
    cum_mean  = np.cumsum(hist * np.arange(256))

    # Between-class variance for all t in [0..254]
    w0 = cum_count[:-1]
    w1 = total - w0

    # Avoid division by zero
    mask = (w0 > 0) & (w1 > 0)
    mu0 = np.zeros_like(w0)
    mu1 = np.zeros_like(w0)
    mu0[mask] = (cum_mean[:-1][mask] / w0[mask])
    mu1[mask] = ( (cum_mean[-1] - cum_mean[:-1][mask]) / w1[mask] )
    sigma_b2 = (w0 * w1) * (mu0 - mu1) ** 2

    t = int(np.argmax(sigma_b2))
    return t
