# -*- coding: utf-8 -*-
"""Application package for the dithering project.

Exposes CLI argument parsing, task dispatchers, and visualization helpers.
"""

from __future__ import annotations

from .cli import (
    check_kernels,
    parse_args,
)
from .tasks import (
    task_adaptive_diffusion,
    task_error_diffusion,
    task_naive,
    task_ordered,
    task_random,
)
from .visualize import show_images

__all__ = [
    "parse_args",
    "check_kernels",
    "task_adaptive_diffusion",
    "task_error_diffusion",
    "task_naive",
    "task_ordered",
    "task_random",
    "show_images",
]
