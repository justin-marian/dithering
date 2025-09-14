# -*- coding: utf-8 -*-
"""CLI helpers for the dithering demo.

Provides argument parsing, kernel validation, and pretty-printing of kernels by family.
"""

from __future__ import annotations

import argparse
from textwrap import dedent
from typing import Dict, List, Set

from ..error_diffusion.kernels import (
    ALIASES_FAMILY,
    DITHERING_KERNELS,
    KERNEL_ALIASES,
    resolve_kernel_name,
)


def _invert_aliases() -> Dict[str, Set[str]]:
    """Convert alias->canonical mapping into canonical->{aliases}."""
    canon_to_aliases: Dict[str, Set[str]] = {}
    for alias, canon in KERNEL_ALIASES.items():
        canon_to_aliases.setdefault(canon, set()).add(alias)
    return canon_to_aliases


def format_kernels_by_family() -> str:
    """Pretty-print kernels grouped by family with aliases.

    Example
    -------
    classic:
        - floyd_steinberg           aliases: FS, floyd, steinberg
        - jarvis_judice_ninke       aliases: JJN, jarvis, judice, ninke
        - stucki                    aliases: ST
    """
    canon_to_aliases = _invert_aliases()
    lines: List[str] = []

    for family, mapping in ALIASES_FAMILY.items():
        kernels_in_family = list(mapping)
        if not kernels_in_family:
            continue

        lines.append(f"{family}:")
        for cname in sorted(kernels_in_family):
            aliases = sorted(a for a in canon_to_aliases.get(cname, set()) if a != cname)
            alias_str = ", ".join(aliases) if aliases else "—"
            lines.append(f"  - {cname:<26} aliases: {alias_str}")
    return "\n".join(lines)


def check_kernels(kernels: List[str]) -> List[str]:
    """Validate and normalize kernel names (supports aliases)."""
    valid: List[str] = []
    for k in kernels:
        cname = resolve_kernel_name(k)
        if cname not in DITHERING_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{k}'. Supported canonical names: "
                f"{sorted(DITHERING_KERNELS.keys())}"
            )
        valid.append(cname)
    return valid


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the dithering demo."""
    p = argparse.ArgumentParser(description="Dithering demo runner")

    canon = ", ".join(sorted(DITHERING_KERNELS.keys()))
    kernels_catalog = format_kernels_by_family()

    kernels_help = dedent(
        f"""
        (error_diffusion) Comma-separated kernel names or aliases.
        You may mix canonical names and aliases, e.g.:  --kernels 'FS,JJN,stucki'

        Canonical kernels:
        {canon}

        Kernels by family (canonical → aliases):
        {kernels_catalog}
        """
    ).strip()

    default_kernels_list = [
        "floyd_steinberg",
        "jarvis_judice_ninke",
        "stucki",
        "burkes",
        "atkinson",
        "sierra",
        "two_row_sierra",
        "stevenson_arce",
    ]
    default_kernels = ",".join(default_kernels_list)

    p.add_argument(
        "--task",
        choices=[
            "adaptive_diffusion",
            "error_diffusion",
            "multi_level",
            "naive",
            "ordered",
            "random",
        ],
        default="error_diffusion",
        help="Which task to run.",
    )
    p.add_argument(
        "--kernels",
        type=str,
        default=default_kernels,
        help=kernels_help,
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=128,
        help="Global threshold (0..255) for binarization.",
    )
    p.add_argument(
        "--no-serpentine",
        action="store_true",
        help="Disable serpentine scanning (left-to-right only).",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=4,
        help="(multi_level) Number of gray levels (>=2).",
    )
    p.add_argument(
        "--bayer-n",
        type=int,
        default=8,
        choices=[2, 4, 8, 16],
        help="(ordered) Bayer matrix size (power of two).",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save each output PNG into ./output/",
    )
    return p.parse_args()
