#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create subset JSONL for answerbench analysis.

Keeps only problems where at least one model answer matches gold_answer
after applying the same normalization as BoN_answeranalyze.py.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from BoN_answeranalyze import normalize_math_notation


def norm(x: Any) -> str:
    """Normalize an answer-like value to comparable string."""
    if x is None:
        return ""
    return normalize_math_notation(str(x))


def iter_answer_keys(record: Dict[str, Any]) -> Iterable[str]:
    """Yield answer strings from answer_counts and all_answers."""
    answer_counts = record.get("answer_counts", {})
    if isinstance(answer_counts, dict):
        for key, count in answer_counts.items():
            try:
                if int(count) > 0:
                    yield str(key)
            except Exception:
                # If count is malformed, still consider the key.
                yield str(key)

    all_answers = record.get("all_answers", [])
    if isinstance(all_answers, list):
        for item in all_answers:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                yield str(item[0])


def has_any_match(record: Dict[str, Any]) -> bool:
    """Return True if any answer in the record matches normalized gold."""
    gold_norm = norm(record.get("gold_answer"))
    if not gold_norm:
        return False

    for ans in iter_answer_keys(record):
        if norm(ans) == gold_norm:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create answerbench subset where at least one answer matches gold."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="analysis_answerbench_gpt-oss-20b.jsonl",
        help="Input analysis JSONL path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="analysis_answerbench_subset_gpt-oss-20b.jsonl",
        help="Output subset JSONL path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    total = 0
    kept = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            if has_any_match(record):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    dropped = total - kept
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Total problems: {total}")
    print(f"Kept problems:  {kept}")
    print(f"Dropped problems (all answers != gold): {dropped}")


if __name__ == "__main__":
    main()

