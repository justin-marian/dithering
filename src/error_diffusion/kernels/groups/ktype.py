# -*- coding: utf-8 -*-
"""Type definition for error-diffusion kernels.

A Kernel is represented as a tuple:
  - a list of ((dy, dx), weight) integer pairs
  - an integer denominator (often the sum of weights).
"""

from __future__ import annotations

from typing import List, Tuple

Kernel = Tuple[List[Tuple[Tuple[int, int], int]], int]
