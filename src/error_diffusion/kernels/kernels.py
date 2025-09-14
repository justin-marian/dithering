# -*- coding: utf-8 -*-
"""Canonical error-diffusion kernels, families, and alias utilities.

Exposes:
- FAMILIES / ALIASES_FAMILY: kernel definitions grouped by family and their aliases
- DITHERING_KERNELS: flat map of canonical kernels -> (offsets, denom)
- KERNEL_ALIASES: alias -> canonical
- Helpers to resolve names and inspect/list kernels
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Families of Dithering Kernels (Error Diffusion)
# Each kernel: Dict[str, Tuple[List[Tuple[Tuple[int, int], int]], int]]
# Offsets are ((dy, dx), weight) where dy is row offset (down), dx is col offset.
# Denominator is the designed divisor (often equals weight sum).
# ----------------------------------------------------------------------
from .groups import (
    ARTISTIC,
    ARTISTIC_ALIASES,
    ATKINSON,
    ATKINSON_ALIASES,
    BURKES,
    BURKES_ALIASES,
    CLASSIC,
    CLASSIC_ALIASES,
    DIRECTIONAL,
    DIRECTIONAL_ALIASES,
    EXPERIMENTAL,
    EXPERIMENTAL_ALIASES,
    FAN,
    FAN_ALIASES,
    OPTIMIZED,
    OPTIMIZED_ALIASES,
    RESEARCH,
    RESEARCH_ALIASES,
    SIERRA,
    SIERRA_ALIASES,
    SIMPLE,
    SIMPLE_ALIASES,
    VARIABLE_STATIC,
    VARIABLE_STATIC_ALIASES,
    Kernel,
)

FamiliesType = Dict[str, Dict[str, Kernel]]
AliasType = Dict[str, Dict[str, str]]

FAMILIES: FamiliesType = {
    "classic": CLASSIC,
    "sierra": SIERRA,
    "burkes": BURKES,
    "atkinson": ATKINSON,
    "research": RESEARCH,
    "fan": FAN,
    "simple": SIMPLE,
    "directional": DIRECTIONAL,
    "artistic": ARTISTIC,
    "optimized": OPTIMIZED,
    "variable_static": VARIABLE_STATIC,
    "experimental": EXPERIMENTAL,
}

DITHERING_KERNELS: Dict[str, Tuple[List[Tuple[Tuple[int, int], int]], int]] = {
    k: v for k, v in chain.from_iterable(fam.items() for fam in FAMILIES.values())
}

ALIASES_FAMILY: AliasType = {
    "classic": CLASSIC_ALIASES,
    "sierra": SIERRA_ALIASES,
    "burkes": BURKES_ALIASES,
    "atkinson": ATKINSON_ALIASES,
    "research": RESEARCH_ALIASES,
    "fan": FAN_ALIASES,
    "simple": SIMPLE_ALIASES,
    "directional": DIRECTIONAL_ALIASES,
    "artistic": ARTISTIC_ALIASES,
    "optimized": OPTIMIZED_ALIASES,
    "variable_static": VARIABLE_STATIC_ALIASES,
    "experimental": EXPERIMENTAL_ALIASES,
}

KERNEL_ALIASES: Dict[str, str] = {
    k: v for k, v in chain.from_iterable(fam.items() for fam in ALIASES_FAMILY.values())
}


def resolve_kernel_name(name: str) -> str:
    """Resolve alias to canonical kernel name (no-op if already canonical)."""
    return KERNEL_ALIASES.get(name, name)


def get_kernel_info(name: str) -> Dict[str, Any]:
    """Inspect kernel details by name or alias."""
    kname = resolve_kernel_name(name)
    if kname not in DITHERING_KERNELS:
        raise ValueError(
            "Unsupported kernel "
            f"'{kname}'. Supported: {list(DITHERING_KERNELS.keys()) + list(KERNEL_ALIASES.keys())}"
        )

    offsets, denom = DITHERING_KERNELS[kname]
    # Find family and aliases for this kernel
    family = next((fam for fam, d in FAMILIES.items() if kname in d), None)
    aliases = [alias for alias, canon in KERNEL_ALIASES.items() if canon == kname]

    weight_sum = sum(w for _, w in offsets)
    return {
        "name": kname,
        "family": family,
        "aliases": aliases,
        "offsets": offsets,
        "denominator": denom,
        "weight_sum": weight_sum,
        "is_normalized": (denom == weight_sum),
    }


def list_available_kernels() -> List[str]:
    """List canonical kernel names (sorted)."""
    return sorted(DITHERING_KERNELS.keys())


def list_kernel_aliases() -> Dict[str, str]:
    """Return alias -> canonical mapping (copy)."""
    return dict(KERNEL_ALIASES)


def list_kernels_by_family() -> Dict[str, List[str]]:
    """Return a mapping family -> sorted list of kernel names."""
    return {fam: sorted(d.keys()) for fam, d in FAMILIES.items()}
