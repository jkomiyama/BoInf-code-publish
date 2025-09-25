#!/usr/bin/env python3

"""
Helper utilities.

filename_compress:
  - Shorten the given file name/path by removing redundant parts
  - Spec:
      1) Remove substring "analysis_"
      2) If a dataset name (aime2025, aime2024, gpqa_diamond, math500)
         appears 2+ times, keep only the first occurrence

Example:
  plots/learn_analysis_aime2025_gpt-oss-20b_analysis_aime2025_Phi-4-reasoning.png
    -> plots/learn_aime2025_gpt-oss-20b_Phi-4-reasoning.png
"""

from pathlib import Path
from typing import Union
import re


_DATASET_PATTERNS = [
    "aime2025_",
    "aime2024_",
    "gpqa_diamond_",
    "math500_",
]


def _dedup_datasets_in_text(text: str) -> str:
    """
    If any dataset token appears 2+ times, remove all except the first.
    """
    result = text
    for ds in _DATASET_PATTERNS:
        # Replace the 2nd and subsequent occurrences of ds with empty string.
        # Preserve the first occurrence using split and re-join strategy.
        parts = result.split(ds)
        if len(parts) > 2:
            # Keep the first ds, drop the rest
            result = parts[0] + ds + "".join(parts[1:])
        # If len(parts) <= 2, leave as-is
    return result


def filename_compress(path_like: Union[str, Path]) -> Path:
    """
    Shorten the given file name (or path).

    - Remove all occurrences of "analysis_"
    - If any dataset name appears multiple times, keep only the first

    Args:
        path_like: Input path or file name

    Returns:
        Shortened Path object
    """
    p = Path(path_like)
    parent = p.parent
    name = p.name

    # 1) Remove all occurrences of "analysis_"
    name = name.replace("analysis_", "")

    # 2) Remove duplicated dataset tokens
    name = _dedup_datasets_in_text(name)

    return parent / name


__all__ = [
    "filename_compress",
]


