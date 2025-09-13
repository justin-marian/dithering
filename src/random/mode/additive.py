# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Union

import numpy as np


def addition(
    g: np.ndarray,
    noise: np.ndarray,
    thr: float,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type]
) -> np.ndarray:
    comp = g + noise
    out_u8 = np.where(comp >= thr, 255, 0).astype(np.uint8)
    return out_u8.astype(dtype)
