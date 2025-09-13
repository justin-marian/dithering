# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Union

import numpy as np


def jitter(
    img: np.ndarray, 
    noise: np.ndarray, 
    thr: int,
    dtype: Union[np.dtype, type, Literal['u8'], Literal['f32']]
) -> np.ndarray:
    comp_thr = thr + noise
    out = np.where(img >= comp_thr, 255, 0)
    return out.astype(dtype)
