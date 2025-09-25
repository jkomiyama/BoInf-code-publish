#!/usr/bin/env python3
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from ensemble_utils import infer_dataset_and_llm_from_path


def format_accuracy(correct: int, total: int) -> str:
    if total <= 0:
        return "0.000"
    return f"{correct / total:.3f}"


def latex_escape_minimal(text: str) -> str:
    # Keep TeX commands intact; only escape characters that will break tabular
    # Avoid over-escaping to preserve potential math markup in answers
    return text.replace('&', '\\&').replace('%', '\\%')


def jsonl_to_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            problem_zero_based = int(obj.get('problem_num'))
            total_answers = int(obj.get('total_answers', 0))
            answer_counts = obj.get('answer_counts', {}) or {}
            gold_answer = str(obj.get('gold_answer', ''))
            # Derive majority answer with fallback when empty string
            maj_raw = obj.get('majority_answer', '')
            majority_answer = '' if maj_raw in (None, '') else str(maj_raw)
            if majority_answer == '':
                # Choose the most frequent non-empty key from answer_counts
                chosen_key = None
                highest_count = -1
                for k, v in answer_counts.items():
                    key_str = str(k).strip()
                    if key_str == '':
                        continue
                    try:
                        count = int(v)
                    except Exception:
                        continue
                    if count > highest_count:
                        highest_count = count
                        chosen_key = key_str
                if chosen_key is not None:
                    majority_answer = chosen_key
            correct_answer = int(answer_counts.get(gold_answer, 0))
            accuracy = format_accuracy(correct_answer, total_answers)

            rows.append({
                'problem_num': problem_zero_based + 1,  # 1-based index
                'total_answers': total_answers,
                'correct_answer': correct_answer,
                'accuracy': accuracy,
                'gold_answer': gold_answer,
                'majority_answer': majority_answer,
            })
    return rows


def infer_dataset_and_llm(input_path: Path) -> tuple[str, str]:
    return infer_dataset_and_llm_from_path(input_path)


def rows_to_latex(rows: List[Dict[str, Any]], dataset_name: str, llm_name: str) -> str:
    # Compute totals
    total_answers_sum = sum(int(r['total_answers']) for r in rows)
    correct_answers_sum = sum(int(r['correct_answer']) for r in rows)
    total_accuracy = format_accuracy(correct_answers_sum, total_answers_sum)
    majority_matches = sum(1 for r in rows if str(r['majority_answer']) == str(r['gold_answer']))
    majority_match_ratio = format_accuracy(majority_matches, len(rows))

    header = [
        'problem_num',
        'total_answers',
        'correct_answer',
        'accuracy',
        'gold_answer',
        'majority_answer',
    ]

    align = 'lrrrrrr'
    lines: List[str] = []
    lines.append('\\begin{table}')
    lines.append('\\centering')
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append('problem\\_num & total\\_answers & correct\\_answer & accuracy & gold\\_answer & majority\\_answer \\\\')
    lines.append('\\hline')

    for r in rows:
        pn = str(r['problem_num'])
        ta = str(r['total_answers'])
        ca = str(r['correct_answer'])
        acc = r['accuracy']
        gold = latex_escape_minimal(str(r['gold_answer']))
        maj = latex_escape_minimal(str(r['majority_answer']))
        lines.append(f'{pn} & {ta} & {ca} & {acc} & {gold} & {maj} \\\\')

    # total row
    lines.append('\\hline')
    lines.append(f'total & {total_answers_sum} & {correct_answers_sum} & {total_accuracy} &  & {majority_match_ratio} \\\\')
    lines.append('\\end{tabular}')
    lines.append("\\caption{Basic performance of each model and problem. The final line at column ``accuracy'' indicates Bo1 performance, and the final line at ``majority answer'' incidates \\boinf{} performance. LLM=" + latex_escape_minimal(llm_name) + ", Dataset=" + latex_escape_minimal(dataset_name) + "}")
    # label: tbl:single_performance_<llm>_<dataset>
    label_llm = re.sub(r'[^A-Za-z0-9]+', '-', llm_name).strip('-').lower()
    label_dataset = re.sub(r'[^A-Za-z0-9]+', '-', dataset_name).strip('-').lower()
    lines.append(f"\\label{{tbl:single_performance_{label_llm}_{label_dataset}}}")
    lines.append('\\end{table}.')

    return "\n".join(lines) + "\n"


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_answers_sum = sum(int(r['total_answers']) for r in rows)
    correct_answers_sum = sum(int(r['correct_answer']) for r in rows)
    total_accuracy = format_accuracy(correct_answers_sum, total_answers_sum)
    majority_matches = sum(1 for r in rows if str(r['majority_answer']) == str(r['gold_answer']))
    majority_match_ratio = format_accuracy(majority_matches, len(rows))
    return {
        'total_answers_sum': total_answers_sum,
        'correct_answers_sum': correct_answers_sum,
        'total_accuracy': total_accuracy,
        'majority_match_ratio': majority_match_ratio,
    }


def dataset_pretty_to_key(dataset_name: str) -> str:
    mapping = {
        'AIME2025': 'aime2025',
        'AIME2024': 'aime2024',
        'GPQA-DIAMOND': 'gpqa_diamond',
        'MATH500': 'math500',
    }
    return mapping.get(dataset_name, dataset_name.lower())


def dataset_key_to_pretty(dataset_key: str) -> str:
    mapping = {
        'aime2025': 'AIME2025',
        'aime2024': 'AIME2024',
        'gpqa_diamond': 'GPQA-DIAMOND',
        'math500': 'MATH500',
    }
    return mapping.get(dataset_key, dataset_key)


def summary_table_to_latex(dataset_key: str, entries: List[Dict[str, str]]) -> str:
    dataset_name = dataset_key_to_pretty(dataset_key)
    align = 'lrr'
    lines: List[str] = []
    lines.append('\\begin{table}')
    lines.append('\\centering')
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append('LLM & Bo1 & \\boinf{} \\\\')
    lines.append('\\hline')
    for e in entries:
        llm = latex_escape_minimal(e['llm'])
        total_acc = e['total_accuracy']
        maj = e['majority_match_ratio']
        lines.append(f"{llm} & {total_acc} & {maj} \\\\")
    lines.append('\\end{tabular}')
    lines.append("\\caption{Summary performance per model. Dataset=" + latex_escape_minimal(dataset_name) + "}")
    label_dataset = re.sub(r'[^A-Za-z0-9]+', '-', dataset_name).strip('-').lower()
    lines.append(f"\\label{{tbl:summary_performance_{label_dataset}}}")
    lines.append('\\end{table}.')
    return "\n".join(lines) + "\n"


def process_file(input_path_str: str) -> Path:
    input_path = Path(input_path_str)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    rows = jsonl_to_rows(input_path)
    dataset_name, llm_name = infer_dataset_and_llm(input_path)
    latex = rows_to_latex(rows, dataset_name, llm_name)
    # Also output to stdout
    sys.stdout.write(latex)
    parent = input_path.parent
    if parent.name == 'jsonl':
        out_dir = parent.parent / 'tables'
    else:
        out_dir = parent / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / (input_path.stem + '.tex')
    with output_path.open('w', encoding='utf-8') as f:
        f.write(latex)
    return output_path


def process_all(jsonl_dir: Path) -> int:
    if not jsonl_dir.exists() or not jsonl_dir.is_dir():
        print(f"jsonl directory not found: {jsonl_dir}")
        return 1
    dataset_to_entries: Dict[str, List[Dict[str, str]]] = {}
    for path in sorted(jsonl_dir.glob('*.jsonl')):
        try:
            rows = jsonl_to_rows(path)
            dataset_name, llm_name = infer_dataset_and_llm(path)
            # Generate, output, and save per-file table
            latex = rows_to_latex(rows, dataset_name, llm_name)
            sys.stdout.write(latex)
            parent = path.parent
            out_dir = (parent.parent / 'tables') if parent.name == 'jsonl' else (parent / 'tables')
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / (path.stem + '.tex')
            with output_path.open('w', encoding='utf-8') as f:
                f.write(latex)

            # Collect summary values
            summary = compute_summary(rows)
            dataset_key = dataset_pretty_to_key(dataset_name)
            dataset_to_entries.setdefault(dataset_key, []).append({
                'llm': llm_name,
                'total_accuracy': summary['total_accuracy'],
                'majority_match_ratio': summary['majority_match_ratio'],
            })
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return 2

    # Generate per-dataset summary tables
    desired_order = ['aime2024', 'aime2025', 'gpqa_diamond', 'math500']
    base_out_dir = jsonl_dir.parent / 'tables'
    base_out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_key in desired_order:
        entries = dataset_to_entries.get(dataset_key)
        if not entries:
            continue
        # Stable sort by LLM name
        entries_sorted = sorted(entries, key=lambda e: e['llm'])
        summary_latex = summary_table_to_latex(dataset_key, entries_sorted)
        sys.stdout.write(summary_latex)
        # File name: dataset_name+.tex
        out_path = base_out_dir / f"{dataset_key}+.tex"
        with out_path.open('w', encoding='utf-8') as f:
            f.write(summary_latex)
    return 0


def main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[1] == '--all':
        base_dir = Path(__file__).resolve().parent
        jsonl_dir = base_dir / 'jsonl'
        return process_all(jsonl_dir)
    if len(argv) < 2:
        print("Usage: python jsonl_to_table.py [--all] <file1.jsonl> [file2.jsonl ...]")
        return 1
    for path in argv[1:]:
        try:
            _ = process_file(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))


