# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np


def mean_threshold(
    gray_f32: np.ndarray
) -> float:
    return float(np.mean(gray_f32))
