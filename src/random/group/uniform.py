# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import numpy as np


def uniform_distrib(
    low: float,
    high: float,
    size: tuple[int, int],
    seed: Optional[int] = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=size).astype(np.float32)
