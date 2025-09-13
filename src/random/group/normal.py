# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import numpy as np


def normal_distribution(
    mean: float,
    stddev: float,
    size: tuple[int, int],
    seed: Optional[int] = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, stddev, size=size).astype(np.float32)
