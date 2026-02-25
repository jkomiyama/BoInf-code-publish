#!/usr/bin/env python3

"""Filter out JSONL records where `gold_answer` is 0."""

import argparse
import json
from pathlib import Path


def is_zero_value(value) -> bool:
    """Return True when value represents numeric zero."""
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            return float(text) == 0.0
        except ValueError:
            return False
    return False


def filter_jsonl(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    """Write records with non-zero gold_answer to output JSONL."""
    total = 0
    kept = 0
    removed = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_number, raw_line in enumerate(fin, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            total += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {input_path}"
                ) from exc

            if is_zero_value(record.get("gold_answer")):
                removed += 1
                continue

            fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            kept += 1

    return total, kept, removed


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "analysis_medrect_gpt-oss-20b.jsonl"
    default_output = (
        script_dir / "analysis_medrect_sentenceextraction_gpt-oss-20b.jsonl"
    )

    parser = argparse.ArgumentParser(
        description="Remove records with gold_answer == 0 from a JSONL file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input JSONL path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSONL path (default: {default_output})",
    )
    args = parser.parse_args()

    total, kept, removed = filter_jsonl(args.input, args.output)
    print(f"input  : {args.input}")
    print(f"output : {args.output}")
    print(f"total  : {total}")
    print(f"kept   : {kept}")
    print(f"removed: {removed}")


if __name__ == "__main__":
    main()
