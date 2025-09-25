import argparse
import sys
from pathlib import Path


def find_jsonl_files(root_directory: Path) -> list[Path]:
    """Recursively find all .jsonl files under the given root directory, sorted by path."""
    return sorted(root_directory.rglob('*.jsonl'))


def build_command(jsonl_path: Path, n_trials: int, analyze_bayes: bool) -> str:
    """Build the shell command string for printing.

    Always outputs a python-based command with shell `time` prefix.
    """
    cmd = f"time python test_main.py {jsonl_path} --n-trials {n_trials}"
    if analyze_bayes:
        cmd += " --analyze-bayes"
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description='Print commands to run test_main.py for all .jsonl files under a directory.')
    parser.add_argument('--root', default='jsonl', help='Root directory to search for .jsonl files (default: jsonl)')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of trials to pass to test_main.py (default: 100)')
    parser.add_argument('--no-analyze-bayes', action='store_true', help='Do not pass --analyze-bayes to test_main.py')
    args = parser.parse_args()

    root_dir = Path(args.root)
    if not root_dir.exists():
        print(f"Root directory not found: {root_dir}", file=sys.stderr)
        sys.exit(1)

    jsonl_files = find_jsonl_files(root_dir)
    if not jsonl_files:
        print(f"No .jsonl files found under: {root_dir}", file=sys.stderr)
        sys.exit(1)

    _ = sys.executable  # unused, kept for potential future use
    analyze_bayes = not args.no_analyze_bayes

    print(f"Found {len(jsonl_files)} .jsonl files under {root_dir}.")
    for jsonl_path in jsonl_files:
        command = build_command(jsonl_path, args.n_trials, analyze_bayes)
        print(command)

    # Done printing all commands


if __name__ == '__main__':
    main()


