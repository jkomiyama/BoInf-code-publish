#!/usr/bin/env python3

"""
Script for weighted ensemble testing.

Ensembles analysis results from multiple LLMs with specified weights and evaluates accuracy.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from collections import Counter
import random
import numpy as np
import matplotlib.pyplot as plt

from ensemble_utils import (
    load_jsonl_file,
    analyze_results,
    AnalysisData,
    find_majority_answer,
    abbreviate_llm
)
from example_utils import filename_compress


# =====================
# Special mode (single LLM across 4 datasets)
# =====================

def _compute_asymptotic_accuracy_single_llm(llm_data: List[AnalysisData]) -> float:
    """
    Compute asymptotic (infinite-sample) accuracy for a single LLM using
    weighted_ensemble_prediction_infty.

    Args:
        llm_data: List of AnalysisData for the single LLM

    Returns:
        Accuracy in [0, 1]
    """
    if not llm_data:
        return 0.0

    llm_data_list = [llm_data]
    weights = [1.0]
    gold_by_problem = {item.problem_num: item.gold_answer for item in llm_data}
    problems = sorted(gold_by_problem.keys())

    correct = 0
    for problem_num in problems:
        pred = weighted_ensemble_prediction_infty(llm_data_list, weights, problem_num)
        if pred == gold_by_problem[problem_num]:
            correct += 1
    return correct / len(problems) if problems else 0.0


def _compute_asymptotic_accuracy_ensemble(llm_files: List[str], weights: List[float]) -> float:
    """
    Compute asymptotic (infinite-sample) accuracy for an ensemble using
    weighted_ensemble_prediction_infty across common problems.

    Args:
        llm_files: List of analysis JSONL files
        weights: Weights per LLM (same size as llm_files)

    Returns:
        Accuracy in [0, 1]
    """
    if not llm_files or not weights or len(llm_files) != len(weights):
        return 0.0

    # Load data for all LLMs
    llm_data_list: List[List[AnalysisData]] = []
    for file_path in llm_files:
        data = load_jsonl_file(file_path)
        llm_data_list.append(data)

    # Find common problems
    all_problems = set(item.problem_num for item in llm_data_list[0])
    for data in llm_data_list[1:]:
        all_problems &= set(item.problem_num for item in data)
    problems = sorted(all_problems)
    if not problems:
        return 0.0

    # Gold answers from the first LLM's data (problems are common)
    gold_map = {}
    for item in llm_data_list[0]:
        if item.problem_num in all_problems:
            gold_map[item.problem_num] = item.gold_answer

    correct = 0
    for problem_num in problems:
        pred = weighted_ensemble_prediction_infty(llm_data_list, weights, problem_num)
        if pred == gold_map.get(problem_num, ""):
            correct += 1

    return correct / len(problems)

def _evaluate_single_llm_fixed_samples_over_trials(
    llm_file: Union[str, Path],
    max_samples: int,
    n_trials: int
) -> Tuple[float, float]:
    """
    Evaluate a single LLM with a fixed number of samples over multiple trials.

    Args:
        llm_data: List of AnalysisData for the single LLM
        max_samples: Fixed sample count N
        n_trials: Number of Monte Carlo trials

    Returns:
        (mean_accuracy, std_accuracy)
    """
    # Use evaluate_weighted_ensemble_single_trial to follow existing evaluation logic
    accs: List[float] = []
    for _ in range(n_trials):
        res = evaluate_weighted_ensemble_single_trial(
            [str(llm_file)], [1.0], B=999.0, use_finite_sampling=True,
            verbose=False, no_BF=True, max_samples=max_samples
        )
        accs.append(float(res.get('ensemble_accuracy', 0.0)))

    mean_acc = float(np.mean(accs)) if accs else 0.0
    std_acc = float(np.std(accs)) if len(accs) > 1 else 0.0
    return mean_acc, std_acc


def run_single_llm_across_datasets(
    llm_name: str,
    jsonl_dir: Union[str, Path],
    n_trials: int,
    n_samples_upper: int,
    figsize: Tuple[float, float] = (8.0, 2.5)
) -> None:
    """
    Special mode: for a given LLM name, evaluate fixed-sample accuracy
    across 4 datasets (aime2024, aime2025, gpqa_diamond, math500) for various N,
    and plot accuracy vs N with asymptotic performance as a horizontal dotted line.
    """
    jsonl_dir = Path(jsonl_dir)
    datasets = [
        ("aime2024", "AIME2024"),
        ("aime2025", "AIME2025"),
        ("gpqa_diamond", "GPQA-DIAMOND"),
        ("math500", "MATH500"),
    ]

    allowed_samples = [1, 3, 5, 7, 10, 30, 100, 300, 1000, 3000, 10000]
    Ns = [v for v in allowed_samples if v <= n_samples_upper]
    if not Ns:
        Ns = [1, 3, 5, 7, 10]

    dataset_results: Dict[str, Dict[str, Union[List[int], List[float], float, str]]] = {}

    print(f"=== Single LLM fixed-sample analysis ===")
    print(f"LLM: {llm_name}")
    print(f"Trials per N: {n_trials}")
    print(f"N list: {Ns}")

    for ds_token, ds_pretty in datasets:
        file_path = jsonl_dir / f"analysis_{ds_token}_{llm_name}.jsonl"
        if not file_path.exists():
            print(f"[Skip] File not found for {ds_pretty}: {file_path}")
            dataset_results[ds_pretty] = {
                "N": [],
                "mean": [],
                "std": [],
                "asymptote": 0.0,
                "file": str(file_path),
            }
            continue

        llm_data = load_jsonl_file(file_path)
        asym_acc = _compute_asymptotic_accuracy_single_llm(llm_data)

        mean_list: List[float] = []
        std_list: List[float] = []
        for N in Ns:
            mean_acc, std_acc = _evaluate_single_llm_fixed_samples_over_trials(
                file_path, max_samples=N, n_trials=n_trials
            )
            mean_list.append(mean_acc)
            std_list.append(std_acc)

        dataset_results[ds_pretty] = {
            "N": Ns,
            "mean": mean_list,
            "std": std_list,
            "asymptote": asym_acc,
            "file": str(file_path),
        }

    # Plot 1x4 subplots (do not share y-axis; each dataset has its own scale)
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False)
    # fig.suptitle(f"Accuracy of BoN ({llm_name}): Green line is asymptotic performance", fontsize=12)

    for ax, (ds_pretty, res) in zip(axes, dataset_results.items()):
        Ns_plot: List[int] = res.get("N", [])  # type: ignore
        mean_plot: List[float] = res.get("mean", [])  # type: ignore
        std_plot: List[float] = res.get("std", [])  # type: ignore
        asym = float(res.get("asymptote", 0.0))  # type: ignore

        if Ns_plot and mean_plot:
            ax.errorbar(Ns_plot, mean_plot, yerr=std_plot, fmt='o-',
                        color='darkblue', ecolor='darkblue', capsize=3, linewidth=3, markersize=5)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')

        # Horizontal dotted line for asymptotic performance (light blue)
        ax.axhline(y=asym, linestyle=':', color='green', linewidth=4)
        ax.set_title(ds_pretty, fontsize=11)
        ax.set_xscale('log')
        ax.set_xlabel('# of gens (= $N$, log scale)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Set per-dataset y-limits with a small margin, bounded to [0, 1]
        if Ns_plot and mean_plot:
            if std_plot:
                y_low = min(m - s for m, s in zip(mean_plot, std_plot))
                y_high = max(m + s for m, s in zip(mean_plot, std_plot))
            else:
                y_low = min(mean_plot)
                y_high = max(mean_plot)
            pad = max(0.02, 0.05 * (y_high - y_low))
            y_min = max(0.0, y_low - pad)
            y_max = min(1.0, y_high + pad)
            if y_max - y_min < 0.05:
                center = 0.5 * (y_min + y_max)
                y_min = max(0.0, center - 0.03)
                y_max = min(1.0, center + 0.03)
            ax.set_ylim(y_min, y_max)

        # Enlarge tick and axis styles
        ax.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.5)
        ax.tick_params(axis='both', which='minor', labelsize=9, length=3, width=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    axes[0].set_ylabel('Accuracy', fontsize=12)

    fig.tight_layout(rect=[0, 0.0, 1, 0.96])

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    # Compress file name for readability
    out_name = plots_dir / f"single_fixed_{llm_name}.png"
    out_name = filename_compress(out_name)
    fig.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Saved 4-panel plot: {out_name}")
    plt.close(fig)


def parse_weights(weights_str: str) -> List[float]:
    """
    Parse a weight string into a list of floats.
    
    Args:
        weights_str: A string like "0.5,0.3,0.2"
        
    Returns:
        List of weights (floats)
        
    Raises:
        ValueError: If parsing fails
    """
    try:
        weights = [float(w.strip()) for w in weights_str.split(',')]
        return weights
    except ValueError as e:
        raise ValueError(f"Failed to parse weights: {weights_str}. Error: {e}")


def calculate_bayes_factor(answer_counts: Counter, num_mc_samples: int = 1000) -> float:
    """
    Compute Bayes factor using a Dirichlet distribution.
    
    Args:
        answer_counts: Observed counts per answer
        num_mc_samples: Number of Monte Carlo samples
        
    Returns:
        Bayes factor (num_answers × P(mode|D) / (1 - P(mode|D)))
    """
    if not answer_counts:
        return 0.0
    
    # Observed answers and counts
    answers = list(answer_counts.keys())
    counts = list(answer_counts.values())
    
    if len(answers) <= 1:
        return float('inf')  # Infinite when only one answer exists
    
    # Dirichlet parameters (+1 for Laplace smoothing)
    alpha = [count + 1 for count in counts]
    
    # Index of the mode
    max_count = max(counts)
    max_indices = [i for i, count in enumerate(counts) if count == max_count]
    
    if len(max_indices) > 1:
        # If tie, set Bayes factor low
        return 1.0
    
    max_idx = max_indices[0]
    
    # Approximate P(mode|D) via Monte Carlo
    samples = np.random.dirichlet(alpha, num_mc_samples)
    
    # Fraction of samples where the mode has the max probability
    max_prob_count = 0
    for sample in samples:
        if np.argmax(sample) == max_idx:
            max_prob_count += 1
    
    prob_max_is_mode = max_prob_count / num_mc_samples
    
    # Bayes factor = num_answers × P(mode|D) / (1 - P(mode|D))
    num_answers = len(answers)
    bayes_factor = num_answers * prob_max_is_mode / (1.0 - prob_max_is_mode + 1e-10)
    
    return bayes_factor


def validate_inputs(llm_files: List[str], weights: List[float]) -> None:
    """
    Validate input arguments.
    
    Args:
        llm_files: List of LLM files
        weights: List of weights
        
    Raises:
        ValueError: On validation errors
    """
    if len(llm_files) != len(weights):
        raise ValueError(f"Number of LLM files ({len(llm_files)}) does not match number of weights ({len(weights)})")
    
    if len(llm_files) < 1:
        raise ValueError("At least one LLM file is required")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.6f}")
    
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative")
    
    # Check file existence
    missing_files = []
    for file in llm_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        raise ValueError(f"Missing files: {missing_files}")


def calculate_bayes_factor_mc(counts: List[int], n_samples: int = 1000) -> float:
    """
    Bayes factor via Monte Carlo sampling from a Dirichlet distribution.
    
    Args:
        counts: List of observed counts per answer
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Bayes factor (num_answers × P(mode|D) / P(non-mode|D))
    """
    if len(counts) < 2:
        return float('inf')  # Infinite when only one answer exists
    
    # Dirichlet parameters (count + 1)
    alpha = np.array(counts) + 1.0
    
    # Monte Carlo sampling
    samples = np.random.dirichlet(alpha, n_samples)
    
    # Index of the modal answer
    max_idx = np.argmax(counts)
    
    # Count samples where the modal answer is actually max
    max_prob_samples = np.sum(samples[:, max_idx] == np.max(samples, axis=1))
    
    if max_prob_samples == 0:
        return 0.0
    
    # P(mode|D)
    p_max_given_data = max_prob_samples / n_samples
    
    # P(non-mode|D) = 1 - P(mode|D)
    p_non_max_given_data = 1.0 - p_max_given_data
    
    if p_non_max_given_data == 0: # large BF approximation
        largest = max(counts) / sum(counts)
        second_largest = sorted(counts, reverse=True)[1] / sum(counts)
        # large deviation approximation 
        gap = max(largest - second_largest, 0.0001)
        # Hoeffding approximation
        estimated_bf_denom = np.exp(- max(counts) * gap * gap / 2)
        # print(f"estimated_bf_denom: {estimated_bf_denom}")
        p_non_max_given_data = estimated_bf_denom
    
    # Bayes factor = num_answers × P(mode|D) / P(non-mode|D)
    bayes_factor = len(counts) * p_max_given_data / p_non_max_given_data
    
    return bayes_factor


def sample_from_llm_distribution(
    llm_data_list: List[List[AnalysisData]], 
    weights: List[float],
    problem_num: int
) -> Tuple[str, int]:
    """
    Select an LLM by weights and randomly pick one element from its all_answers.
    Returns a pair [answer, token_count]. If all_answers is missing/empty, fall back
    to sampling from answer_counts and return token_count=0.
    """
    # Select LLM based on weights
    llm_idx = np.random.choice(len(weights), p=weights)
    
    # Retrieve the corresponding LLM problem data
    llm_data = llm_data_list[llm_idx]
    for item in llm_data:
        if item.problem_num == problem_num:
            # Randomly select from all_answers ([answer, tokens])
            if hasattr(item, 'all_answers') and item.all_answers:
                selected_answer, token_count = random.choice(item.all_answers)
                return selected_answer, int(token_count)
            
            # Fallback: sample from answer_counts (token_count=0)
            answers = list(item.answer_counts.keys())
            counts = list(item.answer_counts.values())
            if not answers:
                return "", 0
            total_count = sum(counts)
            if total_count == 0:
                return "", 0
            probabilities = [count / total_count for count in counts]
            selected_idx = np.random.choice(len(answers), p=probabilities)
            return answers[selected_idx], 0
    
    return "", 0


def weighted_ensemble_prediction(
    llm_data_list: List[List[AnalysisData]], 
    weights: List[float],
    problem_num: int,
    B: float = 10.0,
    max_samples: int = 100,
    alpha: float = 0.3,
    no_BF: bool = False
) -> Tuple[str, int, int]:
    """
    Weighted ensemble prediction for a given problem (finite-sample version)
    with early stopping using a Bayes factor.
    
    Args:
        llm_data_list: List of analysis data for each LLM
        weights: Weights for each LLM
        problem_num: Problem number
        B: Bayes factor threshold
        max_samples: Maximum number of samples
        alpha: Weight for the unknown answer bucket
        no_BF: If True, disable early stopping and always run max_samples times
        
    Returns:
        Tuple of (ensemble_prediction, used_sample_count, total_tokens)
    """
    answer_counts = Counter()
    total_tokens = 0
    
    # Sampling loop with early stopping check
    # Define Bayes factor check schedule when max_samples is large
    check_points: Optional[set] = None
    if max_samples >= 101:
        base_points = [1, 2, 4, 8, 16, 24, 32, 40, 60, 80, 100]
        extra_points = list(range(120, max_samples + 1, 20))
        check_points = set(base_points + extra_points)
    for sample_count in range(1, max_samples + 1):
        # Sampling
        sampled_answer, token_count = sample_from_llm_distribution(llm_data_list, weights, problem_num)
        
        if sampled_answer:
            answer_counts[sampled_answer] += 1
            total_tokens += int(token_count)
        # Add the unknown bucket at the end for unseen samples
        answer_counts["unknown"] = alpha
        
        # Early stopping by Bayes factor (only if no_BF is False)
        if not no_BF and ((check_points is None and sample_count >= 1) or (check_points is not None and sample_count in check_points)):
            # Compute Bayes factor with current observations
            if len(answer_counts) >= 1: # if you make it to >= 2, then it never finishes with all unknown samples
                answers = list(answer_counts.keys())
                counts = [answer_counts[answer] for answer in answers]
                
                # If --n-samples (propagated as max_samples) is >= 1000, use more MC samples
                if max_samples >= 3000:
                    bayes_factor = calculate_bayes_factor_mc(counts, n_samples=3000)
                else:
                    bayes_factor = calculate_bayes_factor_mc(counts)
                # print(f"sample_count = {sample_count}, max_samples = {max_samples}, B={B}, Bayes factor: {bayes_factor}")
                
                # Stop if threshold exceeded
                if bayes_factor >= B:
                    break
    
    # Return the mode and used sample count
    if answer_counts:
        return answer_counts.most_common(1)[0][0], sample_count, total_tokens
    else:
        return "", sample_count, total_tokens


def weighted_ensemble_prediction_infty(
    llm_data_list: List[List[AnalysisData]], 
    weights: List[float],
    problem_num: int
) -> str:
    """
    Weighted ensemble prediction for a given problem (population version).
    
    Args:
        llm_data_list: List of analysis data for each LLM
        weights: Weights for each LLM
        problem_num: Problem number
        
    Returns:
        Ensemble prediction string
    """
    # Retrieve the corresponding problem data for each LLM
    problem_data = []
    for llm_data in llm_data_list:
        for item in llm_data:
            if item.problem_num == problem_num:
                problem_data.append(item)
                break
        else:
            # If the problem is not found
            return ""
    
    if len(problem_data) != len(weights):
        return ""
    
    # Perform weighted voting
    weighted_votes = Counter()
    
    for i, (data, weight) in enumerate(zip(problem_data, weights)):
        # Compute weighted votes per answer
        total_votes = data.total_answers
        if total_votes == 0:
            continue
            
        for answer, count in data.answer_counts.items():
            vote_rate = count / total_votes
            weighted_vote = weight * vote_rate
            weighted_votes[answer] += weighted_vote
    
    if not weighted_votes:
        return ""
    
    # Return the answer with the most votes
    return weighted_votes.most_common(1)[0][0]


def evaluate_weighted_ensemble_single_trial(
    llm_files: List[str], 
    weights: List[float],
    B: float = 10.0,
    use_finite_sampling: bool = True,
    verbose: bool = True,
    no_BF: bool = False,
    max_samples: int = 100
) -> Dict[str, Union[float, int, List]]:
    """
    Run evaluation of weighted ensemble.
    
    Args:
        llm_files: List of LLM files
        weights: List of weights
        
    Returns:
        Dictionary of evaluation results
    """
    # Load data
    llm_data_list = []
    llm_names = []
    
    if verbose:
        print("=== Loading data ===")
    for i, file_path in enumerate(llm_files):
        data = load_jsonl_file(file_path)
        llm_data_list.append(data)
        llm_name = Path(file_path).stem
        llm_names.append(llm_name)
        if verbose:
            print(f"  {llm_name}: {len(data)} problems (weight: {weights[i]:.3f})")
    
    # Find common problems
    all_problems = set(item.problem_num for item in llm_data_list[0])
    for data in llm_data_list[1:]:
        all_problems &= set(item.problem_num for item in data)
    
    problems = sorted(all_problems)
    if verbose:
        print(f"\nNumber of common problems: {len(problems)}")
    
    if not problems:
        raise ValueError("No common problems found")
    
    # Run ensemble predictions
    correct_count = 0
    ensemble_results = []
    
    if verbose:
        print("\n=== Running ensemble predictions ===")
    for problem_num in problems:
        # Get gold answer (from the first LLM)
        gold_answer = ""
        for item in llm_data_list[0]:
            if item.problem_num == problem_num:
                gold_answer = item.gold_answer
                break
        
        # Ensemble prediction
        ensemble_prediction, sample_count, total_tokens = weighted_ensemble_prediction(
            llm_data_list, weights, problem_num, B, no_BF=no_BF,
            max_samples=max_samples,  # specified number of samples
        )
        
        is_correct = (ensemble_prediction == gold_answer)
        if is_correct:
            correct_count += 1
        
        ensemble_results.append({
            'problem_num': problem_num,
            'gold_answer': gold_answer,
            'ensemble_prediction': ensemble_prediction,
            'sample_count': sample_count,
            'total_tokens': total_tokens,
            'is_correct': is_correct
        })
    
    accuracy = correct_count / len(problems) if problems else 0.0
    # Average total tokens
    token_totals = [r.get('total_tokens', 0) for r in ensemble_results]
    mean_total_tokens = (sum(token_totals) / len(token_totals)) if token_totals else 0.0
    
    # Compute each LLM's individual accuracy
    individual_accuracies = []
    for i, (data, llm_name) in enumerate(zip(llm_data_list, llm_names)):
        # Filter by common problems only
        filtered_data = [item for item in data if item.problem_num in all_problems]
        stats = analyze_results(filtered_data)
        individual_accuracies.append({
            'llm_name': llm_name,
            'accuracy': stats['accuracy'],
            'weight': weights[i]
        })
    
    return {
        'ensemble_accuracy': accuracy,
        'correct_problems': correct_count,
        'total_problems': len(problems),
        'individual_accuracies': individual_accuracies,
        'llm_names': llm_names,
        'weights': weights,
        'detailed_results': ensemble_results,
        'mean_total_tokens': mean_total_tokens,
        'total_tokens_std': 0.0
    }


def evaluate_weighted_ensemble(
    llm_files: List[str], 
    weights: List[float],
    B: float = 10.0,
    use_finite_sampling: bool = True,
    n_trials: int = 100,
    no_BF: bool = False,
    max_samples: int = 100
) -> Dict[str, Union[float, int, List]]:
    """
    Run the weighted ensemble evaluation multiple times and take the mean.
    
    Args:
        llm_files: List of LLM files
        weights: List of weights
        B: Bayes factor threshold
        use_finite_sampling: Whether to use finite sampling
        n_trials: Number of runs
        
    Returns:
        Dictionary of evaluation results (accuracy is the mean)
    """
    if n_trials == 1:
        # For a single run, enable verbose output
        result = evaluate_weighted_ensemble_single_trial(
            llm_files, weights, B, use_finite_sampling, verbose=True, no_BF=no_BF,
            max_samples=max_samples,
        )
        
        # Add sample count statistics
        if 'detailed_results' in result and result['detailed_results']:
            sample_counts = [r.get('sample_count', 0) for r in result['detailed_results'] if 'sample_count' in r]
            if sample_counts:
                mean_sample_count = sum(sample_counts) / len(sample_counts)
                result['mean_sample_count'] = mean_sample_count
                result['sample_count_std'] = 0.0
        
        return result
    
    print(f"=== Running {n_trials} trials ===")
    
    # Multiple runs
    all_accuracies = []
    all_correct_counts = []
    all_sample_counts = []  # Average sample count per trial
    all_mean_tokens = []    # Average total tokens per trial
    first_result = None
    
    for trial in range(n_trials):
        if trial == 0:
            # First trial: verbose and keep the result
            result = evaluate_weighted_ensemble_single_trial(
                llm_files, weights, B, use_finite_sampling, verbose=True, no_BF=no_BF,
                max_samples=max_samples,
            )
            first_result = result
        else:
            # Later trials: disable verbose
            result = evaluate_weighted_ensemble_single_trial(
                llm_files, weights, B, use_finite_sampling, verbose=False, no_BF=no_BF,
                max_samples=max_samples,
            )
        
        all_accuracies.append(result['ensemble_accuracy'])
        all_correct_counts.append(result['correct_problems'])
        
        # Compute mean sample count of this trial
        if 'detailed_results' in result and result['detailed_results']:
            sample_counts = [r.get('sample_count', 0) for r in result['detailed_results'] if 'sample_count' in r]
            if sample_counts:
                trial_avg_samples = sum(sample_counts) / len(sample_counts)
                all_sample_counts.append(trial_avg_samples)
            token_totals = [r.get('total_tokens', 0) for r in result['detailed_results'] if 'total_tokens' in r]
            if token_totals:
                trial_avg_tokens = sum(token_totals) / len(token_totals)
                all_mean_tokens.append(trial_avg_tokens)
        
        if (trial + 1) % 10 == 0:
            print(f"  {trial + 1}/{n_trials} trials done")
    
    # Compute mean and standard deviation
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies) if n_trials > 1 else 0.0
    mean_correct_count = np.mean(all_correct_counts)
    mean_sample_count = np.mean(all_sample_counts) if all_sample_counts else 0.0
    std_sample_count = np.std(all_sample_counts) if len(all_sample_counts) > 1 else 0.0
    mean_total_tokens = np.mean(all_mean_tokens) if all_mean_tokens else 0.0
    std_total_tokens = np.std(all_mean_tokens) if len(all_mean_tokens) > 1 else 0.0
    
    print(f"\n=== Results over {n_trials} trials ===")
    print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean correct count: {mean_correct_count:.2f} / {first_result['total_problems']}")
    if all_sample_counts:
        print(f"Mean # of gens: {mean_sample_count:.1f} ± {std_sample_count:.1f}")
    
    # Update averages based on the first result
    result = first_result.copy()
    result['ensemble_accuracy'] = mean_accuracy
    result['ensemble_accuracy_std'] = std_accuracy
    result['mean_correct_problems'] = mean_correct_count
    result['mean_sample_count'] = mean_sample_count
    result['sample_count_std'] = std_sample_count
    result['n_trials'] = n_trials
    result['all_accuracies'] = all_accuracies
    result['mean_total_tokens'] = mean_total_tokens
    result['total_tokens_std'] = std_total_tokens
    
    return result


def evaluate_weighted_ensemble_fixed_samples(
    llm_files: List[str], 
    weights: List[float],
    max_samples: int = 100,
    n_trials: int = 10
) -> Dict[str, Union[float, int, List]]:
    """
    Weighted ensemble evaluation with a fixed number of samples.
    
    Args:
        llm_files: List of LLM JSONL files
        weights: List of weights
        max_samples: Fixed number of samples
        n_trials: Number of trials
        
    Returns:
        Dictionary of evaluation results
    """
    # Prediction function for fixed samples
    def fixed_samples_prediction(llm_data_list, weights, problem_num):
        return weighted_ensemble_prediction(
            llm_data_list, weights, problem_num, 
            B=999,  # effectively disabled with a large value
            max_samples=max_samples,  # specified number of samples
            no_BF=True  # disable Bayes factor
        )
    
    # Run a modified version of evaluate_weighted_ensemble_single_trial
    all_accuracies = []
    all_correct_counts = []
    all_sample_counts = []
    all_mean_tokens = []
    first_result = None
    
    print(f"=== Running {n_trials} trials with fixed samples {max_samples} ===")
    
    for trial in range(n_trials):
        # Load data
        llm_data_list = []
        llm_names = []
        
        for i, file_path in enumerate(llm_files):
            data = load_jsonl_file(file_path)
            llm_data_list.append(data)
            llm_name = Path(file_path).stem
            llm_names.append(llm_name)
        
        # Identify common problems
        all_problems = set(item.problem_num for item in llm_data_list[0])
        for data in llm_data_list[1:]:
            all_problems &= set(item.problem_num for item in data)
        
        problems = sorted(all_problems)
        
        # Run ensemble predictions
        correct_count = 0
        ensemble_results = []
        
        for problem_num in problems:
            # Get gold answer
            gold_answer = ""
            for item in llm_data_list[0]:
                if item.problem_num == problem_num:
                    gold_answer = item.gold_answer
                    break
            
            # Predict with fixed sample count
            ensemble_prediction, sample_count, total_tokens = fixed_samples_prediction(
                llm_data_list, weights, problem_num
            )
            
            is_correct = (ensemble_prediction == gold_answer)
            if is_correct:
                correct_count += 1
            
            ensemble_results.append({
                'problem_num': problem_num,
                'gold_answer': gold_answer,
                'ensemble_prediction': ensemble_prediction,
                'sample_count': sample_count,
                'total_tokens': total_tokens,
                'is_correct': is_correct
            })
        
        accuracy = correct_count / len(problems) if problems else 0.0
        all_accuracies.append(accuracy)
        all_correct_counts.append(correct_count)
        
        # Sample count and token statistics
        sample_counts = [r['sample_count'] for r in ensemble_results]
        if sample_counts:
            trial_avg_samples = sum(sample_counts) / len(sample_counts)
            all_sample_counts.append(trial_avg_samples)
        token_totals = [r.get('total_tokens', 0) for r in ensemble_results]
        if token_totals:
            trial_avg_tokens = sum(token_totals) / len(token_totals)
            all_mean_tokens.append(trial_avg_tokens)
        
        if trial == 0:
            # Save the first result
            individual_accuracies = []
            for i, (data, llm_name) in enumerate(zip(llm_data_list, llm_names)):
                filtered_data = [item for item in data if item.problem_num in all_problems]
                stats = analyze_results(filtered_data)
                individual_accuracies.append({
                    'llm_name': llm_name,
                    'accuracy': stats['accuracy'],
                    'weight': weights[i]
                })
            
            first_result = {
                'ensemble_accuracy': accuracy,
                'correct_problems': correct_count,
                'total_problems': len(problems),
                'individual_accuracies': individual_accuracies,
                'llm_names': llm_names,
                'weights': weights,
                'detailed_results': ensemble_results
            }
        
        if (trial + 1) % 10 == 0:
            print(f"  {trial + 1}/{n_trials} trials done")
    
    # Aggregate statistics
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies) if n_trials > 1 else 0.0
    mean_correct_count = np.mean(all_correct_counts)
    mean_sample_count = np.mean(all_sample_counts) if all_sample_counts else 0.0
    std_sample_count = np.std(all_sample_counts) if len(all_sample_counts) > 1 else 0.0
    mean_total_tokens = np.mean(all_mean_tokens) if all_mean_tokens else 0.0
    std_total_tokens = np.std(all_mean_tokens) if len(all_mean_tokens) > 1 else 0.0
    
    # Update result
    result = first_result.copy()
    result['ensemble_accuracy'] = mean_accuracy
    result['ensemble_accuracy_std'] = std_accuracy
    result['mean_correct_problems'] = mean_correct_count
    result['mean_sample_count'] = mean_sample_count
    result['sample_count_std'] = std_sample_count
    result['n_trials'] = n_trials
    result['all_accuracies'] = all_accuracies
    result['mean_total_tokens'] = mean_total_tokens
    result['total_tokens_std'] = std_total_tokens
    
    return result


def main_analysis(
    llm_files: List[str], 
    weights: List[float],
    bayes_factors: List[float] = None,
    max_samples_list: List[int] = None,
    n_trials: int = 10,
    show_single: bool = False,
    analyze_bayes: bool = False,
    no_analyze_fixed: bool = False
) -> None:
    """
    Analyze and compare adaptive sampling (Bayes factor) and fixed sampling.
    
    Args:
        llm_files: List of LLM JSONL files
        weights: List of weights
        bayes_factors: List of Bayes factor thresholds to try
        max_samples_list: List of fixed sample counts
        n_trials: Number of trials per setting
        show_single: Whether to analyze each LLM individually
        analyze_bayes: Whether to run adaptive (Bayes) analysis
    """
    if bayes_factors is None:
        bayes_factors = [2, 5, 7, 10, 30, 100, 300] # 1..100
    if max_samples_list is None:
        max_samples_list = [1, 3, 5, 7, 10, 30, 100]  # 1..100
    
    print(f"=== Adaptive sampling vs Fixed sampling analysis ===")
    print(f"Bayes factor range: {bayes_factors}")
    print(f"Fixed sample counts: {max_samples_list}")
    print(f"Trials per setting: {n_trials}")
    
    # Adaptive sampling analysis (Bayes factor)
    adaptive_results = []
    if analyze_bayes:
        print(f"\n--- Adaptive sampling (Bayes factor) ---")
        for B in bayes_factors:
            print(f"\nEvaluating Bayes factor = {B} ...")
            
            result = evaluate_weighted_ensemble(
                llm_files, weights, B=B, n_trials=n_trials, no_BF=False,
                max_samples=max_samples_list[-1],
            )
            
            adaptive_results.append({
                'method': 'adaptive',
                'parameter': B,
                'accuracy': result['ensemble_accuracy'],
                'accuracy_std': result.get('ensemble_accuracy_std', 0.0),
                'mean_sample_count': result.get('mean_sample_count', 0.0),
                'sample_count_std': result.get('sample_count_std', 0.0),
                'mean_total_tokens': result.get('mean_total_tokens', 0.0),
                'total_tokens_std': result.get('total_tokens_std', 0.0)
            })
            
            print(f"  Accuracy: {result['ensemble_accuracy']:.4f} ± {result.get('ensemble_accuracy_std', 0):.4f}")
            print(f"  Mean sample count: {result.get('mean_sample_count', 0):.1f} ± {result.get('sample_count_std', 0):.1f}")
    
    # Fixed sampling analysis (skipped when --no-analyze-fixed is set)
    fixed_results = []
    if not no_analyze_fixed:
        print(f"\n--- Fixed sampling (--no-bf) ---")
        for max_samples in max_samples_list:
            print(f"\nEvaluating fixed sample count = {max_samples} ...")
            result = evaluate_weighted_ensemble_fixed_samples(
                llm_files, weights, max_samples=max_samples, n_trials=n_trials
            )
            fixed_results.append({
                'method': 'fixed',
                'parameter': max_samples,
                'accuracy': result['ensemble_accuracy'],
                'accuracy_std': result.get('ensemble_accuracy_std', 0.0),
                'mean_sample_count': result.get('mean_sample_count', 0.0),
                'sample_count_std': result.get('sample_count_std', 0.0),
                'mean_total_tokens': result.get('mean_total_tokens', 0.0),
                'total_tokens_std': result.get('total_tokens_std', 0.0)
            })
            print(f"  Accuracy: {result['ensemble_accuracy']:.4f} ± {result.get('ensemble_accuracy_std', 0):.4f}")
            print(f"  Mean sample count: {result.get('mean_sample_count', 0):.1f} ± {result.get('sample_count_std', 0):.1f}")
    
    # Combine both results
    all_results = adaptive_results + fixed_results
    
    # Single LLM analysis (when show_single is enabled and multiple LLMs)
    single_llm_results = []
    if show_single and len(llm_files) > 1:
        print(f"\n{'='*60}")
        print(f"=== Single-LLM analysis ===")
        print(f"{'='*60}")
        
        for i, llm_file in enumerate(llm_files):
            llm_name = Path(llm_file).stem
            print(f"\n--- {llm_name} single analysis ---")
            
            # Adaptive sampling analysis for a single LLM (if requested)
            single_adaptive_results = []
            if analyze_bayes:
                for B in bayes_factors:
                    print(f"\n{llm_name} evaluating Bayes factor = {B} ...")
                    
                    result = evaluate_weighted_ensemble(
                        [llm_file], [1.0], B=B, n_trials=n_trials, no_BF=False,
                        max_samples=max_samples_list[-1]
                    )
                    
                    single_adaptive_results.append({
                        'method': 'adaptive',
                        'llm_name': llm_name,
                        'parameter': B,
                        'accuracy': result['ensemble_accuracy'],
                        'accuracy_std': result.get('ensemble_accuracy_std', 0.0),
                        'mean_sample_count': result.get('mean_sample_count', 0.0),
                        'sample_count_std': result.get('sample_count_std', 0.0),
                        'mean_total_tokens': result.get('mean_total_tokens', 0.0),
                        'total_tokens_std': result.get('total_tokens_std', 0.0)
                    })
                    
                    print(f"  Accuracy: {result['ensemble_accuracy']:.4f} ± {result.get('ensemble_accuracy_std', 0):.4f}")
                    print(f"  Mean sample count: {result.get('mean_sample_count', 0):.1f} ± {result.get('sample_count_std', 0):.1f}")
            
            # Fixed sampling analysis for a single LLM (skipped with --no-analyze-fixed)
            single_fixed_results = []
            if not no_analyze_fixed:
                for max_samples in max_samples_list:
                    print(f"\n{llm_name} evaluating fixed sample count = {max_samples} ...")
                    result = evaluate_weighted_ensemble_fixed_samples(
                        [llm_file], [1.0], max_samples=max_samples, n_trials=n_trials
                    )
                    single_fixed_results.append({
                        'method': 'fixed',
                        'llm_name': llm_name,
                        'parameter': max_samples,
                        'accuracy': result['ensemble_accuracy'],
                        'accuracy_std': result.get('ensemble_accuracy_std', 0.0),
                        'mean_sample_count': result.get('mean_sample_count', 0.0),
                        'sample_count_std': result.get('sample_count_std', 0.0),
                        'mean_total_tokens': result.get('mean_total_tokens', 0.0),
                        'total_tokens_std': result.get('total_tokens_std', 0.0)
                    })
                    print(f"  Accuracy: {result['ensemble_accuracy']:.4f} ± {result.get('ensemble_accuracy_std', 0):.4f}")
                    print(f"  Mean sample count: {result.get('mean_sample_count', 0):.1f} ± {result.get('sample_count_std', 0):.1f}")
            
            # Append single-LLM results
            single_llm_results.extend(single_adaptive_results + single_fixed_results)
    
    # Compute asymptotic accuracy overlay (single or ensemble)
    overlay_asymptote: Optional[float] = None
    try:
        if llm_files:
            overlay_asymptote = _compute_asymptotic_accuracy_ensemble(llm_files, weights)
    except Exception:
        overlay_asymptote = None

    # Save combined plots (ensemble + single LLM) side by side
    create_combined_comparison_and_tokens_plot(
        all_results,
        single_llm_results,
        llm_files,
        n_trials,
        analyze_bayes=analyze_bayes,
        show_single=show_single,
        no_analyze_fixed=no_analyze_fixed,
        overlay_asymptote=overlay_asymptote
    )

def create_combined_comparison_and_tokens_plot(ensemble_results: List[Dict], single_llm_results: List[Dict], 
                                               llm_files: List[str], n_trials: int, *, analyze_bayes: bool, show_single: bool, no_analyze_fixed: bool = False, use_tag: bool = False, overlay_asymptote: Optional[float] = None) -> None:
    """Save side-by-side plots: combined comparison (left) and tokens x accuracy (right)."""
    if analyze_bayes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    else:
        # Without Bayes analysis, create a single comparison plot
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 3))

    # ===== Left: Combined comparison =====
    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if ensemble_results:
        adaptive_data = [r for r in ensemble_results if r['method'] == 'adaptive']
        fixed_data = [r for r in ensemble_results if r['method'] == 'fixed']

        # Ensemble adaptive
        if adaptive_data:
            x = [r['mean_sample_count'] for r in adaptive_data]
            y = [r['accuracy'] for r in adaptive_data]
            xerr = [2 * r['sample_count_std'] / np.sqrt(n_trials) for r in adaptive_data]
            yerr = [2 * r['accuracy_std'] / np.sqrt(n_trials) for r in adaptive_data]
            labels = [r['parameter'] for r in adaptive_data]
            ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=10,
                         capsize=4, capthick=2, linewidth=3, linestyle='-',
                         label=('Ensemble Adaptive' if show_single else 'Adaptive'),
                         color='darkblue', alpha=0.8)
            xy_sorted = sorted(zip(x, y))
            if xy_sorted:
                x_line, y_line = zip(*xy_sorted)
                ax1.plot(x_line, y_line, color='darkblue', linewidth=3, alpha=0.5, label='_nolegend_')
            if use_tag:
                for xi, yi, lab in zip(x, y, labels):
                    ax1.annotate(f'B={lab}', (xi, yi), xytext=(8, 8), textcoords='offset points', fontsize=10, color='darkblue', fontweight='bold')

        # Ensemble fixed
        if fixed_data:
            x = [r['mean_sample_count'] for r in fixed_data]
            y = [r['accuracy'] for r in fixed_data]
            xerr = [2 * r['sample_count_std'] / np.sqrt(n_trials) for r in fixed_data]
            yerr = [2 * r['accuracy_std'] / np.sqrt(n_trials) for r in fixed_data]
            labels = [r['parameter'] for r in fixed_data]
            ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='s', markersize=10,
                         capsize=4, capthick=2, linewidth=3, linestyle='-',
                         label=('Ensemble' if show_single else 'Fixed'),
                         color='darkred', alpha=0.8)
            xy_sorted = sorted(zip(x, y))
            if xy_sorted:
                x_line, y_line = zip(*xy_sorted)
                ax1.plot(x_line, y_line, color='darkred', linewidth=3, alpha=0.5, label='_nolegend_')
            # Do not display N when --analyze-bayes is not provided
            if use_tag:
                if analyze_bayes:
                    for xi, yi, lab in zip(x, y, labels):
                        ax1.annotate(f'N={lab}', (xi, yi), xytext=(8, -12), textcoords='offset points', fontsize=10, color='darkred', fontweight='bold')

    def remove_prefix(name: str) -> str:
        return name.replace('analysis_', '').replace('aime2025_', '').replace('aime2024_', '').replace('gpqa_diamond_', '').replace('math500_', '')
    if single_llm_results:
        llm_names = list(set(r['llm_name'] for r in single_llm_results))
        for i, llm_name in enumerate(llm_names):
            color = colors[i % len(colors)]
            llm_data = [r for r in single_llm_results if r['llm_name'] == llm_name]
    
            display_name = abbreviate_llm(remove_prefix(llm_name))
            #print(f"display_name: {display_name}")
            #sys.exit()
            adaptive_data = [r for r in llm_data if r['method'] == 'adaptive']
            fixed_data = [r for r in llm_data if r['method'] == 'fixed']
            if adaptive_data:
                x = [r['mean_sample_count'] for r in adaptive_data]
                y = [r['accuracy'] for r in adaptive_data]
                xerr = [2 * r['sample_count_std'] / np.sqrt(n_trials) for r in adaptive_data]
                yerr = [2 * r['accuracy_std'] / np.sqrt(n_trials) for r in adaptive_data]
                ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=6,
                             capsize=3, capthick=1, linewidth=1, alpha=0.6,
                             label=f'{display_name} Adaptive', color=color)
                # Connect single LLM (adaptive) points with lines
                xy_sorted = sorted(zip(x, y))
                if xy_sorted and len(xy_sorted) >= 2:
                    x_line, y_line = zip(*xy_sorted)
                    ax1.plot(x_line, y_line, color=color, linewidth=1, alpha=0.6, label='_nolegend_')
            if fixed_data:
                x = [r['mean_sample_count'] for r in fixed_data]
                y = [r['accuracy'] for r in fixed_data]
                xerr = [2 * r['sample_count_std'] / np.sqrt(n_trials) for r in fixed_data]
                yerr = [2 * r['accuracy_std'] / np.sqrt(n_trials) for r in fixed_data]
                if adaptive_data:
                    label_fixed = f'{display_name} Fixed'
                else:
                    label_fixed = f'{display_name}'
                # xerr=xerr removed xerr
                ax1.errorbar(x, y, yerr=yerr, fmt='^', markersize=6,
                             capsize=3, capthick=1, linewidth=1, alpha=0.6,
                             label=label_fixed, color=color, linestyle='--')
                # Connect single LLM (fixed) points with dashed lines
                xy_sorted = sorted(zip(x, y))
                if xy_sorted and len(xy_sorted) >= 2:
                    x_line, y_line = zip(*xy_sorted)
                    ax1.plot(x_line, y_line, color=color, linewidth=1, alpha=0.6, linestyle='--', label='_nolegend_')

    ax1.set_xscale('log')
    ax1.set_xlabel('Average # of Gens (log scale)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    # Respect the intent of "no legend" in the original function; do not show legend here
    if not analyze_bayes:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # y-axis range (left)
    all_results_left = ensemble_results + single_llm_results
    all_accuracies_left = [r['accuracy'] for r in all_results_left]
    if overlay_asymptote is not None:
        all_accuracies_left.append(overlay_asymptote)
    all_accuracy_cis_left = [2 * r.get('accuracy_std', 0.0) / np.sqrt(n_trials) for r in all_results_left]
    if all_accuracies_left:
        y_min = min(all_accuracies_left) - (max(all_accuracy_cis_left) if all_accuracy_cis_left else 0.0) 
        y_max = max(all_accuracies_left) + (max(all_accuracy_cis_left) if all_accuracy_cis_left else 0.0) 
        gap = max(y_max - y_min, 0.01)
        ax1.set_ylim(max(0, y_min - gap * 0.1), min(1, y_max + gap * 0.1))

    # Overlay asymptotic performance as a horizontal dotted green line on the left plot (no legend)
    if overlay_asymptote is not None:
        overlay_color = 'darkblue' if len(llm_files) > 1 else 'green'
        ax1.axhline(y=overlay_asymptote, linestyle=':', color=overlay_color, linewidth=2.5, label='_nolegend_')

    # ===== Right: Tokens x Accuracy =====
    if analyze_bayes:
        if ensemble_results:
            for method, fmt, color in [('adaptive', 'o', 'darkblue'), ('fixed', 's', 'darkred')]:
                data = [r for r in ensemble_results if r.get('method') == method]
                if not data:
                    continue
                x = [r.get('mean_total_tokens', 0.0) for r in data]
                y = [r.get('accuracy', 0.0) for r in data]
                xerr = [2 * r.get('total_tokens_std', 0.0) / np.sqrt(n_trials) for r in data]
                yerr = [2 * r.get('accuracy_std', 0.0) / np.sqrt(n_trials) for r in data]
                labels = [r.get('parameter', '') for r in data]
                x_plot = [xi if xi > 0 else 1.0 for xi in x]
                xerr_plot = [min(xe, xi * 0.99) if xi > 0 else 0.0 for xi, xe in zip(x_plot, xerr)]
                # removed xerr=xerr_plot,
                ax2.errorbar(x_plot, y,  yerr=yerr, fmt=fmt, markersize=10,
                             capsize=4, capthick=2, linewidth=3, linestyle='-',
                             label=(
                                 f"Ensemble {'Adaptive' if method=='adaptive' else 'Fixed'}"
                                 if show_single else ('Adaptive' if method=='adaptive' else 'Fixed')
                             ),
                             color=color, alpha=0.8)
                xy_sorted = sorted(zip(x_plot, y))
                if xy_sorted:
                    x_line, y_line = zip(*xy_sorted)
                    ax2.plot(x_line, y_line, color=color, linewidth=3, alpha=0.5, label='_nolegend_')
                if use_tag:
                    for xi, yi, lab in zip(x_plot, y, labels):
                        if method == 'adaptive':
                            tag = f"B={lab}"
                            ax2.annotate(tag, (xi, yi), xytext=(8, 8),
                                        textcoords='offset points', fontsize=10, color=color, fontweight='bold')
                        elif analyze_bayes:
                            tag = f"N={lab}"
                            ax2.annotate(tag, (xi, yi), xytext=(8, -12),
                                        textcoords='offset points', fontsize=10, color=color, fontweight='bold')

        if show_single and single_llm_results:
            colors2 = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            llm_names = list(set(r['llm_name'] for r in single_llm_results))
            for i, llm_name in enumerate(llm_names):
                color = colors2[i % len(colors2)]
                llm_data = [r for r in single_llm_results if r['llm_name'] == llm_name]
                display_name = abbreviate_llm(remove_prefix(llm_name))
                for method, fmt in [('adaptive', 'o'), ('fixed', '^')]:
                    data = [r for r in llm_data if r['method'] == method]
                    if not data:
                        continue
                    x = [r.get('mean_total_tokens', 0.0) for r in data]
                    y = [r.get('accuracy', 0.0) for r in data]
                    xerr = [2 * r.get('total_tokens_std', 0.0) / np.sqrt(n_trials) for r in data]
                    yerr = [2 * r.get('accuracy_std', 0.0) / np.sqrt(n_trials) for r in data]
                    x_plot = [xi if xi > 0 else 1.0 for xi in x]
                    base_jitter = 1.0 + 0.02 * (i - (len(llm_names) - 1) / 2)
                    method_jitter = 1.0 + (0.02 if method == 'adaptive' else -0.02)
                    jitter = base_jitter * method_jitter
                    x_plot = [xi * jitter for xi in x_plot]
                    xerr_plot = [min(xe, xi * 0.99) if xi > 0 else 0.0 for xi, xe in zip(x_plot, xerr)]
                    ax2.errorbar(x_plot, y, xerr=xerr_plot, yerr=yerr, fmt=fmt, markersize=6,
                                 capsize=3, capthick=1, linewidth=1, alpha=0.6,
                                 label=f"{display_name} {'Adaptive' if method=='adaptive' else 'Fixed'}",
                                 color=color, linestyle='--' if method == 'fixed' else '-')

        ax2.set_xscale('log')
        ax2.set_xlabel('Average Total Tokens (log scale)', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        # y-axis range (right)
        all_results_right = ensemble_results + (single_llm_results if show_single else [])
        all_accuracies_right = [r['accuracy'] for r in all_results_right]
        if overlay_asymptote is not None:
            all_accuracies_right.append(overlay_asymptote)
        all_accuracy_cis_right = [2 * r.get('accuracy_std', 0.0) / np.sqrt(n_trials) for r in all_results_right]
        if all_accuracies_right:
            y_min = min(all_accuracies_right) - (max(all_accuracy_cis_right) if all_accuracy_cis_right else 0.0) 
            y_max = max(all_accuracies_right) + (max(all_accuracy_cis_right) if all_accuracy_cis_right else 0.0) 
            gap = max(y_max - y_min, 0.01)
            ax2.set_ylim(max(0, y_min - gap * 0.1), min(1, y_max + gap * 0.1))

        # Overlay asymptotic performance on the right plot as dotted green line (no legend)
        if overlay_asymptote is not None:
            overlay_color = 'darkblue' if len(llm_files) > 1 else 'green'
            ax2.axhline(y=overlay_asymptote, linestyle=':', color=overlay_color, linewidth=2.5, label='_nolegend_')

    fig.tight_layout()

    # Filename rule is same as create_unified_comparison_plots
    file_stems = [Path(f).stem for f in llm_files]
    if analyze_bayes and show_single:
        prefix = "both_"
    elif analyze_bayes:
        prefix = "adaptive_only_" if no_analyze_fixed else "adaptive_"
    else:
        prefix = "fixed_"
    filename = prefix + "_".join(file_stems) + ".png"

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    output_path = plots_dir / filename
    output_path = filename_compress(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if analyze_bayes:
        print(f"\nSaved combined side-by-side plot: {output_path}")
    else:
        print(f"\nSaved comparison plot: {output_path}")
    plt.close(fig)
    
    # Print unified summary
    print_unified_summary(ensemble_results, single_llm_results, n_trials, analyze_bayes=analyze_bayes)

def print_unified_summary(ensemble_results: List[Dict], single_llm_results: List[Dict], n_trials: int, *, analyze_bayes: bool = False) -> None:
    """Print unified results summary"""
    print(f"\n=== Unified results summary (95% CI) ===")
    
    if ensemble_results:
        print(f"\n--- Ensemble results ---")
        print(f"{'Method':>8} {'Param':>6} {'Accuracy':>10} {'±':>1} {'95%CI':>6} {'Samples':>8} {'±':>1} {'95%CI':>6}")
        print("-" * 58)
        
        for r in ensemble_results:
            method_short = 'Adapt' if r['method'] == 'adaptive' else 'Fixed'
            if r['method'] == 'adaptive':
                param_str = f"B={r['parameter']}"
            else:
                param_str = f"N={r['parameter']}" if analyze_bayes else f"{r['parameter']}"
            accuracy_ci = 2 * r['accuracy_std'] / np.sqrt(n_trials)
            sample_ci = 2 * r['sample_count_std'] / np.sqrt(n_trials)
            print(f"{method_short:>8} {param_str:>6} {r['accuracy']:>10.4f} ± {accuracy_ci:>6.4f} "
                  f"{r['mean_sample_count']:>8.1f} ± {sample_ci:>6.1f}")
    
    if single_llm_results:
        llm_names = list(set(r['llm_name'] for r in single_llm_results))
        
        for llm_name in llm_names:
            llm_data = [r for r in single_llm_results if r['llm_name'] == llm_name]
            print(f"\n--- {llm_name} single results ---")
            print(f"{'Method':>8} {'Param':>6} {'Accuracy':>10} {'±':>1} {'95%CI':>6} {'Samples':>8} {'±':>1} {'95%CI':>6}")
            print("-" * 58)
            
            for r in llm_data:
                method_short = 'Adapt' if r['method'] == 'adaptive' else 'Fixed'
                if r['method'] == 'adaptive':
                    param_str = f"B={r['parameter']}"
                else:
                    param_str = f"N={r['parameter']}" if analyze_bayes else f"{r['parameter']}"
                accuracy_ci = 2 * r['accuracy_std'] / np.sqrt(n_trials)
                sample_ci = 2 * r['sample_count_std'] / np.sqrt(n_trials)
                print(f"{method_short:>8} {param_str:>6} {r['accuracy']:>10.4f} ± {accuracy_ci:>6.4f} "
                      f"{r['mean_sample_count']:>8.1f} ± {sample_ci:>6.1f}")



def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Weighted ensemble test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Evaluate a single LLM
  python test_main.py analysis_aime2025_Qwen3-14B.jsonl
  
  # Ensemble 3 LLMs with specified weights
  python test_main.py analysis_aime2025_Qwen3-14B.jsonl analysis_aime2025_gpt-oss-20b.jsonl analysis_aime2025_Phi-4.jsonl \
    --weights 0.5,0.3,0.2
  
  # Ensemble 2 LLMs with equal weights
  python test_main.py file1.jsonl file2.jsonl
  
  # Run 100 trials and compute mean accuracy
  python test_main.py file1.jsonl file2.jsonl --n-trials 100
  
  # Analyze/plot for Bayes factor 1-10
  python test_main.py file1.jsonl file2.jsonl --analyze-bayes --n-trials 50
  
  # Also analyze each LLM individually
  python test_main.py file1.jsonl file2.jsonl file3.jsonl --analyze-bayes --show-single --n-trials 30
  
  # Disable Bayes factor and always run max_samples times
  python test_main.py file1.jsonl file2.jsonl --no-bf --n-trials 100
  
  # Show detailed results
  python test_main.py file1.jsonl file2.jsonl \
    --weights 0.6,0.4 \
    --show-details
  
  # Specify Bayes factor threshold
  python test_main.py file1.jsonl file2.jsonl \
    --weights 0.6,0.4 \
    --bayes-threshold 5.0
"""
    )
    
    parser.add_argument(
        'llm_files',
        nargs='*',
        help='List of LLM JSONL files to ensemble (omit when using --llm_name)'
    )
    
    parser.add_argument(
        '--weights',
        help='Weights per LLM (comma-separated, e.g., 0.5,0.3,0.2). If omitted, use equal weights'
    )
    
    parser.add_argument(
        '--show-details',
        action='store_true',
        help='Show detailed per-problem results'
    )
    
    parser.add_argument(
        '--bayes-threshold',
        type=float,
        default=10.0,
        help='Bayes factor threshold (default: 10.0)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=1,
        help='Number of runs (default: 1). If >1, compute mean accuracy'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Upper bound for fixed sampling max samples. Candidates [1,2,3,5,7,10,30,100,300,1000,3000,10000]; use those <= this value (default: 100)'
    )
    
    parser.add_argument(
        '--analyze-bayes',
        action='store_true',
        help='Analyze/plot relation between mean sample count and accuracy for Bayes factor 1-10'
    )
    
    parser.add_argument(
        '--b-bfs',
        type=int,
        default=300,
        help='Upper bound for Bayes factors. Candidates [2,5,7,10,30,100,300,1000,3000,5000,10000]; use values <= this (default: 300)'
    )
    
    parser.add_argument(
        '--no-bf',
        action='store_true',
        help='Disable early stopping by Bayes factor and always run max_samples times'
    )
    
    parser.add_argument(
        '--show-single',
        action='store_true',
        help='When multiple LLMs are specified, also analyze each LLM individually'
    )
    
    parser.add_argument(
        '--no-analyze-fixed',
        action='store_true',
        help='Skip fixed-sampling analysis and save plots as adaptive_only'
    )

    # Special mode: single LLM across 4 datasets
    parser.add_argument(
        '--llm_name',
        type=str,
        help='Special mode: evaluate a single LLM across datasets (expects analysis_{dataset}_{llm_name}.jsonl)'
    )
    parser.add_argument(
        '--jsonl-dir',
        type=str,
        default='boinf/jsonl',
        help='Directory containing analysis_*.jsonl files (default: boinf/jsonl)'
    )
    parser.add_argument(
        '--figsize',
        type=str,
        default=None,
        help='Figure size for 1x4 plot, e.g., "16,3.5" or "16x3.5" (width,height)'
    )
    
    args = parser.parse_args()
    
    try:
        # Special mode: single LLM across datasets
        if args.llm_name:
            # Parse figsize if provided
            fig_size = (12.0, 3.5)
            if args.figsize:
                try:
                    s = args.figsize.lower().replace('x', ',')
                    parts = [p.strip() for p in s.split(',') if p.strip()]
                    if len(parts) == 2:
                        fig_size = (float(parts[0]), float(parts[1]))
                except Exception:
                    pass
            run_single_llm_across_datasets(
                llm_name=args.llm_name,
                jsonl_dir=args.jsonl_dir,
                n_trials=args.n_trials,
                n_samples_upper=args.n_samples,
                figsize=fig_size,
            )
            return

        # Normal mode: ensemble (requires llm_files)
        if not args.llm_files:
            raise ValueError('No llm_files provided. Provide files or use --llm_name for special mode.')

        # Parse weights (use equal weights if not specified)
        if args.weights:
            weights = parse_weights(args.weights)
        else:
            # Use equal weights automatically
            num_llms = len(args.llm_files)
            weights = [1.0 / num_llms] * num_llms
            print(f"Weights not specified; using equal weight ({1.0/num_llms:.3f})")
        
        # Input validation
        validate_inputs(args.llm_files, weights)
        
        print("=== Weighted ensemble test ===")
        print(f"#LLM files: {len(args.llm_files)}")
        print(f"Weights: {weights}")
        print(f"Sum of weights: {sum(weights):.6f}")
        
        # Choose candidates <= --n-samples
        allowed_samples = [1, 3, 5, 7, 10, 30, 100, 300, 1000, 3000, 10000]
        max_samples_list = [v for v in allowed_samples if v <= args.n_samples]
        if not max_samples_list:
            max_samples_list = None
        
        # Choose Bayes factors <= --b-bfs
        allowed_bfs = [2, 5, 7, 10, 30, 100, 300, 1000, 3000, 10000, 10**7, 10**10, 10**15]
        if args.b_bfs -1: 
            args.b_bfs = allowed_bfs[-1]
        bayes_factors = [v for v in allowed_bfs if v <= args.b_bfs]
        if not bayes_factors:
            bayes_factors = None
        
        # Main analysis (regardless of analyze_bayes flag)
        main_analysis(
            args.llm_files,
            weights,
            bayes_factors=bayes_factors,
            max_samples_list=max_samples_list,
            n_trials=args.n_trials,
            show_single=args.show_single,
            analyze_bayes=args.analyze_bayes,
            no_analyze_fixed=args.no_analyze_fixed
        )
        return
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
