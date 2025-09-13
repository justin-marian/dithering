# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from ...utils import map_threshold_graydomain


def global_threshold(
    threshold: Union[int, float],
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type]
) -> int:
    return map_threshold_graydomain(threshold, dtype)
