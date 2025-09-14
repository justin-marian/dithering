# -*- coding: utf-8 -*-
"""Kernel definitions and utilities for error diffusion dithering.

Exposes canonical kernels, aliases, and family groupings along with
helper functions for kernel resolution and introspection.
"""

from __future__ import annotations

from .kernels import (
    ALIASES_FAMILY,
    DITHERING_KERNELS,
    KERNEL_ALIASES,
    get_kernel_info,
    list_available_kernels,
    list_kernel_aliases,
    list_kernels_by_family,
    resolve_kernel_name,
)

__all__ = [
    "DITHERING_KERNELS",
    "KERNEL_ALIASES",
    "ALIASES_FAMILY",
    "resolve_kernel_name",
    "get_kernel_info",
    "list_available_kernels",
    "list_kernel_aliases",
    "list_kernels_by_family",
]
