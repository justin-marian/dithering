# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np


def percentile_threshold(
    gray_f32: np.ndarray,
    percentile: float
) -> float:
    return float(np.percentile(gray_f32, percentile))
