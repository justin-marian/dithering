# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from .group.bayer import bayer_bw, bayer_matrix
from .group.halftone import halftone_bw


def ordered_bw(
    img: np.ndarray,
    *,
    kind: Literal["bayer", "halftone"] = "bayer",
    n: int = 8,
    angle_deg: float = 45.0,
    spot: Literal["cos+cos", "cosx", "cosx+2cosy"] = "cos+cos",
    dtype: Union[str, np.dtype, type] = "u8",
) -> np.ndarray:
    if kind == "bayer":
        return bayer_bw(img, matrix=bayer_matrix(n), dtype=dtype)
    if kind == "halftone":
        return halftone_bw(img, size=n, angle_deg=angle_deg, spot=spot, dtype=dtype)
    raise ValueError("kind must be 'bayer' or 'halftone'")
