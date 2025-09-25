#!/usr/bin/env python3

"""
Utility functions for reading and analyzing JSONL files.

Provides helpers to load and analyze files like analysis_aime2025XX.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np


def normalize_answer_format(text: str) -> str:
    """
    Normalize answer string formatting
    - Remove angle markers (^\\circ, ^{\\circ})
    - Unify fraction notation (convert \\dfrac to \\frac)
    - Trim extraneous whitespace

    Args:
        text: Input string to normalize

    Returns:
        Normalized string

    Examples:
        normalize_answer_format("336^\\circ") -> "336"
        normalize_answer_format("336^{\\circ}") -> "336"
        normalize_answer_format("\\frac{5096}{9}") -> "\\frac{5096}{9}"
        normalize_answer_format("\\dfrac{366912}{649}") -> "\\frac{366912}{649}"
        normalize_answer_format("123") -> "123"
        normalize_answer_format("abc") -> "abc"
    """
    if not isinstance(text, str):
        return str(text)
    
    # If empty string, return as-is
    if not text.strip():
        return text
    
    result = text
    
    # 1. Remove angle markers
    result = re.sub(r'\^\{\\circ\}', '', result)
    result = re.sub(r'\^\\circ', '', result)
    
    # 2. Unify fraction notation (\\dfrac -> \\frac)
    result = re.sub(r'\\dfrac', r'\\frac', result)
    
    # 3. Trim extra whitespace
    result = result.strip()
    
    return result


def infer_dataset_and_llm_from_path(path: Union[str, Path]) -> Tuple[str, str]:
    """
    Extract dataset name and LLM name from a file path (or filename).
    Expected format: analysis_{dataset}_{llm}.jsonl
    Supported datasets: aime2025, aime2024, gpqa-diamond/gpqa_diamond, math500

    Args:
        path: Path to the input file or filename

    Returns:
        (dataset_name, llm_name)
        dataset_name is a pretty string like "AIME2025"
        llm_name is "unknown-llm" if extraction fails
    """
    p = Path(path)
    stem = p.stem
    m = re.match(r"analysis_(aime2025|aime2024|gpqa[-_]?diamond|math500)_(.+)$", stem, flags=re.IGNORECASE)
    dataset_raw: Optional[str] = None
    llm_name: Optional[str] = None
    if m:
        dataset_raw = m.group(1)
        llm_name = m.group(2)
    else:
        lower = stem.lower()
        for key in ["aime2025", "aime2024", "gpqa-diamond", "gpqa_diamond", "math500"]:
            if key in lower:
                dataset_raw = key
                parts = stem.split("_", 2)
                if len(parts) >= 3:
                    llm_name = parts[2]
                break

    dataset_map = {
        "aime2025": "AIME2025",
        "aime2024": "AIME2024",
        "gpqa-diamond": "GPQA-DIAMOND",
        "gpqa_diamond": "GPQA-DIAMOND",
        "math500": "MATH500",
    }
    dataset_name = dataset_map.get((dataset_raw or "").lower(), "AIME2025")
    llm_name = llm_name or "unknown-llm"
    return dataset_name, llm_name


def abbreviate_llm(name: str) -> str:
    """Utility to shorten LLM names for legends/display"""
    s = name
    lower = s.lower()
    # Shorten MetaStone-S1-32B family
    if 'metastone' in lower:
        return 'MetaStone'
    # Shorten Phi-4-reasoning family
    if lower.startswith('phi-4') or ('phi4' in lower and 'reasoning' in lower) or 'phi-4-reasoning' in lower:
        return 'Phi4-R'
    # Shorten Qwen3-30B-A3B(-Thinking-xxxx) family
    if lower.startswith('qwen3-30b') and 'a3b' in lower:
        return 'Qwen3-30B-A3BT'
    # Shorten EXAONE-Deep-32B family
    if lower.startswith('exaone') and 'deep' in lower:
        return 'EXAONE-D'
    # Shorten Datarus-R1-14B-preview
    if lower.startswith('datarus-r1-14b') or (lower.startswith('datarus') and 'preview' in lower):
        return 'Datarus-r1'
    # Shorten NVIDIA-Nemotron-Nano-9B-v2
    if 'nemotron' in lower and 'nano-9b' in lower:
        return 'Nemotron-N9B'
    return s


def extract_numbers_only(text: str) -> str:
    """
    Backward-compat function (deprecated)
    Prefer using normalize_answer_format
    """
    return normalize_answer_format(text)


def normalize_answer_counts(answer_counts: Dict[str, int], clean_answers: bool = True) -> Dict[str, int]:
    """
    Normalize answer_counts and merge variants in notation.

    Args:
        answer_counts: Original answer_counts dictionary
        clean_answers: Whether to normalize answers (default: True)

    Returns:
        Normalized answer_counts dictionary

    Examples:
        normalize_answer_counts({"336^\\circ": 8, "336": 57, "336^{\\circ}": 4})
        -> {"336": 69}
    """
    if not clean_answers:
        return answer_counts.copy()
    
    normalized_counts = {}
    original_to_normalized = {}  # Map original answer -> normalized answer
    
    for original_answer, count in answer_counts.items():
        normalized_answer = normalize_answer_format(original_answer)
        original_to_normalized[original_answer] = normalized_answer
        
        if normalized_answer in normalized_counts:
            normalized_counts[normalized_answer] += count
        else:
            normalized_counts[normalized_answer] = count
    
    return normalized_counts


def find_majority_answer(normalized_counts: Dict[str, int]) -> str:
    """
    Find the most frequent answer from normalized answer_counts.

    Args:
        normalized_counts: Normalized answer_counts dictionary

    Returns:
        The most frequent answer (if tie, return lexicographically first)
    """
    if not normalized_counts:
        return ""
    
    # Obtain the maximum vote count
    max_count = max(normalized_counts.values())
    
    # Collect all answers with max count
    top_answers = [answer for answer, count in normalized_counts.items() if count == max_count]
    
    # Return the first in lexicographic order
    return sorted(top_answers)[0]


class AnalysisData:
    """Data structure representing one problem's analysis record"""
    
    def __init__(
        self,
        problem_num: int,
        total_answers: int,
        answer_counts: Dict[str, int],
        gold_answer: str,
        majority_answer: str,
        clean_answers: bool = True,
        all_answers: Optional[List[List[Union[str, int]]]] = None
    ):
        self.problem_num = problem_num
        self.total_answers = total_answers

        # Remove empty answers from answer_counts at load time
        filtered_answer_counts: Dict[str, int] = {}
        for k, v in (answer_counts or {}).items():
            key_str = str(k) if k is not None else ""
            if key_str.strip() == "":
                continue
            filtered_answer_counts[key_str] = int(v)

        self.original_answer_counts = filtered_answer_counts.copy()  # Preserve original (after empty removal)
        self.original_gold_answer = gold_answer  # Preserve original gold_answer
        self.original_majority_answer = majority_answer  # Preserve original majority_answer
        # all_answers is a list of pairs [answer, score]
        # Remove entries whose answer is empty string
        cleaned_all_answers: List[List[Union[str, int]]] = []
        for pair in (all_answers or []):
            try:
                ans = str(pair[0]) if pair and pair[0] is not None else ""
                if ans.strip() == "":
                    continue
                tok = int(pair[1]) if len(pair) > 1 else 0
                cleaned_all_answers.append([ans, tok])
            except Exception:
                # Skip malformed entries silently
                continue
        self.all_answers = cleaned_all_answers
        
        if clean_answers:
            # Normalize answer_counts
            self.answer_counts = normalize_answer_counts(filtered_answer_counts, clean_answers=True)
            # Normalize gold_answer
            self.gold_answer = normalize_answer_format(gold_answer)
            # Re-compute majority_answer from normalized counts
            self.majority_answer = find_majority_answer(self.answer_counts)
        else:
            self.answer_counts = filtered_answer_counts
            self.gold_answer = gold_answer
            self.majority_answer = majority_answer

        # Ensure total_answers matches the sum of (cleaned/normalized) counts
        try:
            self.total_answers = int(sum(self.answer_counts.values()))
        except Exception:
            # Fallback to provided value if something goes wrong
            self.total_answers = total_answers
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], clean_answers: bool = True) -> "AnalysisData":
        """Create AnalysisData from a dictionary"""
        return cls(
            problem_num=data["problem_num"],
            total_answers=data["total_answers"],
            answer_counts=data["answer_counts"],
            gold_answer=data["gold_answer"],
            majority_answer=data["majority_answer"],
            clean_answers=clean_answers,
            all_answers=data.get("all_answers", [])
        )
    
    def __repr__(self) -> str:
        return (f"AnalysisData(problem_num={self.problem_num}, "
                f"total_answers={self.total_answers}, "
                f"gold='{self.gold_answer}', majority='{self.majority_answer}')")
    
    def is_correct(self) -> bool:
        """Return True if majority answer matches the gold answer"""
        return self.majority_answer == self.gold_answer
    
    def get_majority_confidence(self) -> float:
        """Return majority answer confidence (vote rate)"""
        if self.total_answers == 0:
            return 0.0
        majority_count = self.answer_counts.get(self.majority_answer, 0)
        return majority_count / self.total_answers


def load_jsonl_file(file_path: Union[str, Path], clean_answers: bool = True) -> List[AnalysisData]:
    """
    Read a JSONL file and return a list of AnalysisData.

    Args:
        file_path: Path to the JSONL file to read
        clean_answers: Whether to normalize answers (default: True)

    Returns:
        List of AnalysisData objects

    Raises:
        FileNotFoundError: When the file is not found
        json.JSONDecodeError: When a line is not valid JSON
        KeyError: When required keys are missing
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    results = []
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    data = json.loads(line)
                    analysis_data = AnalysisData.from_dict(data, clean_answers=clean_answers)
                    results.append(analysis_data)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parse error at line {line_num}: {e}", file=sys.stderr)
                    continue
                except KeyError as e:
                    print(f"Warning: Missing required key {e} at line {line_num}", file=sys.stderr)
                    continue
                    
    except Exception as e:
        print(f"File read error: {e}", file=sys.stderr)
        raise
    
    return results


def load_all_analysis_files(directory: Union[str, Path] = "./jsonl") -> Dict[str, List[AnalysisData]]:
    """
    Load all analysis_aime2025*.jsonl files in the given directory.

    Args:
        directory: Target directory (default: current directory)

    Returns:
        Dict mapping filename to list of AnalysisData
    """
    directory = Path(directory)
    results = {}
    
    # Find files starting with analysis_aime2025 and ending with .jsonl
    pattern = "analysis_aime2025*.jsonl"
    files = list(directory.glob(pattern))
    
    if not files:
        print(f"Warning: No files matching {pattern} found in {directory}", file=sys.stderr)
        return results
    
    for file_path in files:
        try:
            data = load_jsonl_file(file_path)
            results[file_path.name] = data
            print(f"Loaded: {file_path.name} ({len(data)} problems)")
        except Exception as e:
            print(f"Error: Failed to load {file_path.name}: {e}", file=sys.stderr)
    
    return results


def analyze_results(data: List[AnalysisData]) -> Dict[str, Any]:
    """
    Compute summary statistics for analysis data.

    Args:
        data: List of analysis records

    Returns:
        Dict of statistics
    """
    if not data:
        return {"total_problems": 0, "accuracy": 0.0, "avg_confidence": 0.0}
    
    total_problems = len(data)
    correct_count = sum(1 for item in data if item.is_correct())
    accuracy = correct_count / total_problems
    
    confidences = [item.get_majority_confidence() for item in data]
    avg_confidence = sum(confidences) / len(confidences)
    
    return {
        "total_problems": total_problems,
        "correct_answers": correct_count,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "confidence_distribution": {
            "min": min(confidences) if confidences else 0.0,
            "max": max(confidences) if confidences else 0.0,
        }
    }


def optimal_weight(jsonl_files: List[Union[str, Path]], margin: float = 0.0) -> Dict[str, Any]:
    """
    Solve a MILP to optimally weight multiple LLM results.

    Args:
        jsonl_files: List of per-LLM analysis JSONL files
        margin: Margin parameter (float)

    Returns:
        Dict of optimization results (weights, correct count, objective, etc.)
    """
    try:
        import highspy as hs
    except ImportError as e:
        raise ImportError("highspy is required: pip install highspy") from e
    
    # Load data
    llm_data = []
    llm_names = []
    
    for file_path in jsonl_files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Warning: File not found {file_path}", file=sys.stderr)
            continue
            
        data = load_jsonl_file(file_path)
        llm_data.append(data)
        llm_names.append(file_path.stem)
    
    if not llm_data:
        raise ValueError("No valid JSONL files found")
    
    K = len(llm_data)  # number of LLMs
    
    # Get common problem indices across all LLMs
    all_problems = set(item.problem_num for item in llm_data[0])
    for data in llm_data[1:]:
        all_problems &= set(item.problem_num for item in data)
    
    problems = sorted(all_problems)
    J = len(problems)  # number of problems
    # print(f"problems: {problems}");sys.exit()
    
    if J == 0:
        raise ValueError("No common problems found")
    
    print(f"#LLMs: {K}, #Common problems: {J}")
    
    # Prepare structures
    # llm_answers[i][j] = answer_counts of LLM i for problem j
    # gold_answers[j] = gold answer for problem j
    llm_answers = {}
    gold_answers = {}
    
    for i, data in enumerate(llm_data):
        llm_answers[i] = {}
        for item in data:
            if item.problem_num in problems:
                llm_answers[i][item.problem_num] = item.answer_counts
                gold_answers[item.problem_num] = item.gold_answer
    
    # Build MILP
    inf = getattr(hs, "kHighsInf", 1e20)
    highs = hs.Highs()

    # Suppress output
    highs.setOptionValue("log_to_console", False)
    highs.setOptionValue("output_flag", False)

    # Decision variables
    # w_i: weight of LLM i (i = 0, ..., K-1)
    # y_j: binary correctness indicator for problem j (j = 0, ..., J-1)
    
    # Add weight variables w_i (continuous, 0 <= w_i <= 1)
    weight_costs = [0.0] * K  # weights do not affect objective directly
    weight_lower = [0.0] * K
    weight_upper = [1.0] * K
    
    # Add y_j variables (binary, 0 <= y_j <= 1)
    y_costs = [1.0] * J  # objective: maximize sum y_j
    y_lower = [0.0] * J
    y_upper = [1.0] * J
    
    # All objective coefficients and bounds
    all_costs = np.array(weight_costs + y_costs)
    all_lower = np.array(weight_lower + y_lower)
    all_upper = np.array(weight_upper + y_upper)
    
    # Add variables
    num_vars = K + J
    num_nz = 0  # initially empty constraint matrix
    a_start = np.array([0] * (num_vars + 1), dtype=np.int32)
    a_index = np.array([], dtype=np.int32)
    a_value = np.array([], dtype=np.float64)
    
    highs.addCols(num_vars, all_costs, all_lower, all_upper, num_nz, a_start, a_index, a_value)
    
    # Set y variables to integer
    for j in range(J):
        y_idx = K + j
        highs.changeColIntegrality(y_idx, hs.HighsVarType.kInteger)
    
    # Add constraints
    
    # 1) Sum of weights = 1
    weight_indices = np.arange(K, dtype=np.int32)
    weight_values = np.ones(K, dtype=np.float64)
    highs.addRow(1.0, 1.0, K, weight_indices, weight_values)
    
    # 2) Constraints per problem
    big_M = 10000.0  # moderate big_M for numerical stability
    
    constraint_count = 0
    for j_idx, problem_num in enumerate(problems):
        gold_answer = gold_answers[problem_num]
        # print(f"j_idx: {j_idx}, problem_num: {problem_num}, gold_answer: {gold_answer}")
        
        # Gold counts per LLM
        gold_counts = []
        total_votes_per_llm = []
        for i in range(K):
            counts = llm_answers[i][problem_num]
            gold_count = counts.get(gold_answer, 0)
            gold_counts.append(gold_count)

        # Collect wrong answers and their counts
        all_wrong_answers = {}
        for i in range(K):
            counts = llm_answers[i][problem_num]
            total_votes_per_llm.append(sum(counts.values()))
            for answer, count in counts.items():
                if answer != gold_answer:
                    if answer not in all_wrong_answers:
                        all_wrong_answers[answer] = [0 for i in range(K)]
                    all_wrong_answers[answer][i] += count
        
        y_idx = K + j_idx
        
        for wrong_answer, wrong_counts in all_wrong_answers.items():
            # Constraint: sum_i w_i * gold_counts[i] - sum_i w_i * wrong_counts[i] >= big_M * (1 - y_j)
            
            # Build coefficient arrays
            coeff_indices = []
            coeff_values = []
            
            # Coefficients for w_i (use normalized rates)
            has_meaningful_constraint = False
            #print(f"K={K}")
            for i in range(K):
                #print(f"i: {i}, gold_counts: {gold_counts}, wrong_counts: {wrong_counts}, total_votes_per_llm: {total_votes_per_llm}")
                denom = total_votes_per_llm[i]
                if denom <= 0:
                    # This LLM has no valid votes for this problem, so don't contribute to the coefficient
                    continue
                gold_rate = gold_counts[i] / denom
                wrong_rate = wrong_counts[i] / denom
                coeff = gold_rate - wrong_rate
                #print(f"i: {i}, gold_rate: {gold_rate}, wrong_rate: {wrong_rate}, coeff: {coeff}")
                
                if abs(coeff) > 1e-6:  # only meaningful differences
                    coeff_indices.append(i)
                    coeff_values.append(coeff)
                    has_meaningful_constraint = True
            #print(f"has_meaningful_constraint: {has_meaningful_constraint}")

            # Add only if meaningful
            if has_meaningful_constraint:            
                # coefficient for y_j
                coeff_indices.append(y_idx)
                coeff_values.append(-big_M)  # negative coefficient
                
                coeff_indices = np.array(coeff_indices, dtype=np.int32)
                coeff_values = np.array(coeff_values, dtype=np.float64)
                num_nz_constraint = len(coeff_indices)
                
                # sum_i w_i*(gold_rate[i] - wrong_rate[i]) - margin - big_M*y_j >= -big_M
                # i.e., sum_i w_i*(gold_rate[i] - wrong_rate[i]) >= margin - big_M*(1 - y_j)
                rhs = margin - big_M
                highs.addRow(rhs, inf, num_nz_constraint, coeff_indices, coeff_values)
                constraint_count += 1
    
    #print(f"Number of added constraints: {constraint_count}")
    
    # Maximize
    highs.changeObjectiveSense(hs.ObjSense.kMaximize)
    
    # Solve
    #print("Solving MILP...")
    run_status = highs.run()
    
    if run_status != hs.HighsStatus.kOk:
        raise RuntimeError("Failed to run optimization")
    
    # Retrieve results
    sol = highs.getSolution()
    info = highs.getInfo()
    model_status = highs.getModelStatus()
    
    #print(f"ModelStatus: {model_status}")
    
    # Check model status
    if model_status == hs.HighsModelStatus.kInfeasible:
        print("Warning: MILP model is infeasible. Constraints may be too strict.")
        # Fallback: return uniform weights
        weights = [1.0 / K] * K
        y_values = [0.0] * J
        optimal_weights = {llm_names[i]: weights[i] for i in range(K)}
        correct_problems = 0.0
        objective_value = 0.0
        
        result = {
            "optimal_weights": optimal_weights,
            "correct_problems": correct_problems,
            "total_problems": J,
            "accuracy": correct_problems / J,
            "objective_value": objective_value,
            "model_status": str(model_status),
            "llm_names": llm_names,
            "weights_array": weights,
            "problem_solutions": y_values,
            "note": "Infeasible MILP - using uniform weights"
        }
        return result
    
    elif model_status != hs.HighsModelStatus.kOptimal:
        print(f"Warning: Optimal solution not found. Status: {model_status}")
        # Fallback: uniform weights
        weights = [1.0 / K] * K
        y_values = [0.0] * J
        optimal_weights = {llm_names[i]: weights[i] for i in range(K)}
        correct_problems = 0.0
        objective_value = 0.0
        
        result = {
            "optimal_weights": optimal_weights,
            "correct_problems": correct_problems,
            "total_problems": J,
            "accuracy": correct_problems / J,
            "objective_value": objective_value,
            "model_status": str(model_status),
            "llm_names": llm_names,
            "weights_array": weights,
            "problem_solutions": y_values,
            "note": f"Optimization did not reach optimal ({model_status}); using uniform weights"
        }
        return result
    
    # Extract optimal solution
    if sol.col_value is None or len(sol.col_value) < K + J:
        print("Warning: Failed to retrieve a valid solution vector")
        weights = [1.0 / K] * K
        y_values = [0.0] * J
    else:
        weights = sol.col_value[:K]
        y_values = sol.col_value[K:]
    
    optimal_weights = {llm_names[i]: weights[i] for i in range(K)}
    correct_problems = sum(y_values) if y_values is not None else 0.0
    
    try:
        objective_value = info.objective_function_value
    except Exception:
        objective_value = correct_problems  # fallback
    
    # If weights is a numpy array, convert to list; if list, leave as is
    if hasattr(weights, 'tolist'):
        weights_list = weights.tolist()
    else:
        weights_list = list(weights)
    
    # Process y_values similarly
    if hasattr(y_values, 'tolist'):
        y_values_list = y_values.tolist()
    else:
        y_values_list = list(y_values) if y_values is not None else [0.0] * J
    
    result = {
        "optimal_weights": optimal_weights,
        "correct_problems": correct_problems,
        "total_problems": J,
        "accuracy": correct_problems / J,
        "objective_value": objective_value,
        "model_status": str(model_status),
        "llm_names": llm_names,
        "weights_array": weights_list,
        "problem_solutions": y_values_list
    }
    
    return result


def get_score(jsonl_files: List[Union[str, Path]], weights: List[float]) -> int:
    """
    Compute how many optimal_weight constraints y_i are satisfied when mixing
    with the given weights.

    Args:
        jsonl_files: List of per-LLM analysis JSONL files
        weights: Weights per LLM (same size as jsonl_files)

    Returns:
        Number of satisfied constraints (int)
    """
    # Load data
    llm_data = []
    llm_names = []
    
    for file_path in jsonl_files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Warning: File not found {file_path}", file=sys.stderr)
            continue
            
        data = load_jsonl_file(file_path)
        llm_data.append(data)
        llm_names.append(file_path.stem)
    
    if not llm_data:
        raise ValueError("No valid JSONL files found")
    
    if len(weights) != len(llm_data):
        raise ValueError(f"Number of weights ({len(weights)}) does not match number of files ({len(llm_data)})")
    
    K = len(llm_data)  # number of LLMs
    
    # Get common problems across LLMs
    all_problems = set(item.problem_num for item in llm_data[0])
    for data in llm_data[1:]:
        all_problems &= set(item.problem_num for item in data)
    
    problems = sorted(all_problems)
    J = len(problems)  # number of problems
    
    if J == 0:
        raise ValueError("No common problems found")
    
    # Prepare structures
    # llm_answers[i][j] = answer_counts of LLM i for problem j
    # gold_answers[j] = gold answer for problem j
    llm_answers = {}
    gold_answers = {}
    
    for i, data in enumerate(llm_data):
        llm_answers[i] = {}
        for item in data:
            if item.problem_num in problems:
                llm_answers[i][item.problem_num] = item.answer_counts
                gold_answers[item.problem_num] = item.gold_answer
    
    # Check constraints per problem
    satisfied_constraints = 0
    failed_problems = []
    
    for j_idx, problem_num in enumerate(problems):
        gold_answer = gold_answers[problem_num]
        
        # Per-LLM gold counts
        gold_counts = []
        total_votes_per_llm = []
        for i in range(K):
            counts = llm_answers[i][problem_num]
            gold_count = counts.get(gold_answer, 0)
            gold_counts.append(gold_count)
            # Note the possibility that the total vote count is 0
            total_votes_per_llm.append(sum(counts.values()))

        # Collect wrong answers
        all_wrong_answers = {}
        for i in range(K):
            counts = llm_answers[i][problem_num]
            for answer, count in counts.items():
                if answer != gold_answer:
                    if answer not in all_wrong_answers:
                        all_wrong_answers[answer] = [0 for i in range(K)]
                    all_wrong_answers[answer][i] += count
        
        # Check if this problem satisfies constraints
        problem_is_correct = True
        failed_constraint = None
        
        for wrong_answer, wrong_counts in all_wrong_answers.items():
            # Same logic as optimal_weight: only check meaningful constraints
            has_meaningful_constraint = False
            constraint_value = 0.0
            
            for i in range(K):
                denom = total_votes_per_llm[i]
                if denom <= 0:
                    # Skip if this LLM has no valid votes
                    continue
                gold_rate = gold_counts[i] / denom
                wrong_rate = wrong_counts[i] / denom
                coeff = gold_rate - wrong_rate
                
                if abs(coeff) > 1e-6:  # same threshold as in optimal_weight
                    has_meaningful_constraint = True
                    constraint_value += weights[i] * coeff
            
            # Only check if meaningful
            if has_meaningful_constraint:
                # If the constraint is violated
                if constraint_value < -1e-9:  # numerical tolerance
                    problem_is_correct = False
                    failed_constraint = {
                        'wrong_answer': wrong_answer,
                        'constraint_value': constraint_value,
                        'gold_counts': gold_counts,
                        'wrong_counts': wrong_counts,
                        'total_votes': total_votes_per_llm
                    }
                    break
        
        if problem_is_correct:
            satisfied_constraints += 1
        else:
            failed_problems.append({
                'problem_num': problem_num,
                'gold_answer': gold_answer,
                'failed_constraint': failed_constraint
            })
    
    # Debug output
    if failed_problems:
        print(f"\nget_score debug: problems violating constraints ({len(failed_problems)}):")
        for fp in failed_problems:
            fc = fp['failed_constraint']
            print(f"  problem {fp['problem_num']}: gold={fp['gold_answer']}, wrong={fc['wrong_answer']}")
            print(f"    constraint value: {fc['constraint_value']:.6f} < 0")
            print(f"    gold_counts: {fc['gold_counts']}")
            print(f"    wrong_counts: {fc['wrong_counts']}")
            print(f"    total_votes: {fc['total_votes']}")
    
    return satisfied_constraints

def test_normalize_answers():
    """Test for answer normalization utility"""
    print("=== Answer normalization test ===")
    
    test_cases = [
        "336^\\circ",
        "336^{\\circ}",
        "336",
        "\\frac{5096}{9}",
        "\\dfrac{366912}{649}",
        "\\frac{366912}{649}",
        "123",
        "abc",
        "504",
        "m+n = 128+693 = 821",
        "a+b+c=3+57+2=62",
        "",
        "  336^\\circ  ",  # leading/trailing spaces
        "\\dfrac{5096}{9}",
    ]
    
    for test_case in test_cases:
        result = normalize_answer_format(test_case)
        print(f"  '{test_case}' -> '{result}'")


def test_answer_counts_normalization():
    """Test for answer_counts normalization"""
    print("\n=== answer_counts normalization test ===")
    
    # Example like problem 19
    test_counts = {
        "336^\\circ": 8,
        "336": 57,
        "336^{\\circ}": 4,
        "300": 1
    }
    
    print(f"Original answer_counts: {test_counts}")
    normalized = normalize_answer_counts(test_counts, clean_answers=True)
    print(f"Normalized: {normalized}")
    majority = find_majority_answer(normalized)
    print(f"majority_answer: {majority}")
    
    # Fraction example
    test_counts_frac = {
        "\\frac{5096}{9}": 3,
        "\\dfrac{5096}{9}": 2,
        "588": 70,
        "492": 5
    }
    
    print(f"\nOriginal answer_counts: {test_counts_frac}")
    normalized_frac = normalize_answer_counts(test_counts_frac, clean_answers=True)
    print(f"Normalized: {normalized_frac}")
    majority_frac = find_majority_answer(normalized_frac)
    print(f"majority_answer: {majority_frac}")
