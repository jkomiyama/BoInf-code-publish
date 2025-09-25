#!/usr/bin/env python3

"""
Main runner for reading and analyzing JSONL files.

Entry point for loading and analyzing analysis_aime2025XX.jsonl files.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import glob
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from contextlib import contextmanager
from example_utils import filename_compress

from ensemble_utils import (
    load_jsonl_file,
    load_all_analysis_files,
    analyze_results,
    optimal_weight,
    get_score,
    test_normalize_answers,
    test_answer_counts_normalization,
    infer_dataset_and_llm_from_path,
    abbreviate_llm
)

@contextmanager
def suppress_stdout_stderr():
    """Context manager to temporarily suppress stdout and stderr"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def optimal_weight_binarysearch(existing_files: List[str], verbose: bool = False) -> Dict[str, Any]:
    """Binary search for the maximum margin that preserves weights; return optimization result
    
    Args:
        existing_files: List of JSONL files used for optimization
        verbose: Whether to print detailed logs
        
    Returns:
        Dict containing optimization results
    """
    # Step 1: get baseline result at margin=0
    if verbose:
        print(f"\n=== Step 1: Baseline (margin=0) ===")
    with suppress_stdout_stderr():
        baseline_result = optimal_weight(existing_files, margin=0.0)
    baseline_weights = baseline_result['weights_array']
    
    if verbose:
        print(f"Baseline weights: {[f'{w:.4f}' for w in baseline_weights]}")
        print(f"Baseline accuracy: {baseline_result['accuracy']:.3f}")
    
    # Step 2: search margin in [0, 1] and find the largest value that keeps weights unchanged
    if verbose:
        print(f"\n=== Step 2: Searching optimal margin ===")
    
    def weights_are_similar(w1, w2, tolerance=1e-4):
        """Check if two weight vectors are similar"""
        if len(w1) != len(w2):
            return False
        return all(abs(a - b) < tolerance for a, b in zip(w1, w2))
    
    # Binary search to find the maximum feasible margin
    low, high = 0.0, 1.0
    best_margin = 0.0
    max_iterations = 20
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        mid = (low + high) / 2.0
        if verbose:
            print(f"  Trial {iteration + 1}: margin = {mid:.6f}")
        
        try:
            with suppress_stdout_stderr():
                test_result = optimal_weight(existing_files, margin=mid)
            test_weights = test_result['weights_array']
            
            if baseline_result['accuracy'] - 1e-6 <= test_result['accuracy']:
                best_margin = mid
                low = mid
            else:
                high = mid
                
        except Exception as e:
            # On error, shrink the margin
            high = mid
            if verbose:
                print(f"    Error: {e} -> update margin range to [{low:.6f}, {high:.6f}]")
        
        # Convergence check
        if high - low < tolerance:
            if verbose:
                print(f"    Converged (diff: {high - low:.8f})")
            break
    
    # Step 3: compute final result at the best margin
    if verbose:
        print(f"\n=== Step 3: Final result (margin={best_margin:.6f}) ===")
    with suppress_stdout_stderr():
        result = optimal_weight(existing_files, margin=best_margin)
    
    return result


def find_dataset_files(dataset_target: str) -> List[str]:
    """Find analysis files for the specified dataset
    
    Args:
        dataset_target: Dataset name (e.g., aime2025, aime2024, gpqa_diamond, math500)
        
    Returns:
        List of matched file paths
    """
    pattern = f"jsonl/analysis_{dataset_target}_*.jsonl"
    files = glob.glob(pattern)
    files.sort()  # sort by filename
    return files


def get_single_llm_accuracy(file_path: str) -> float:
    """Get accuracy of a single LLM analysis file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Accuracy (0.0-1.0)
    """
    try:
        data = load_jsonl_file(file_path)
        stats = analyze_results(data)
        return stats['accuracy']
    except Exception as e:
        print(f"Warning: failed to get accuracy from {file_path}: {e}")
        return 0.0


def find_common_llms(source_dataset: str, target_dataset: str) -> List[str]:
    """Find common LLMs between two datasets
    
    Args:
        source_dataset: Source dataset name
        target_dataset: Target dataset name
        
    Returns:
        Sorted list of common LLM names
    """
    source_files = find_dataset_files(source_dataset)
    target_files = find_dataset_files(target_dataset)
    
    # Extract LLM names from filenames
    def extract_llm_name(file_path: str, dataset: str) -> str:
        filename = Path(file_path).stem
        prefix = f"analysis_{dataset}_"
        if filename.startswith(prefix):
            return filename[len(prefix):]
        return filename
    
    source_llms = set(extract_llm_name(f, source_dataset) for f in source_files)
    target_llms = set(extract_llm_name(f, target_dataset) for f in target_files)
    
    common_llms = source_llms & target_llms
    return sorted(list(common_llms))


def sample_problems_from_datasets(files: List[str], n_problems: int) -> Tuple[List[str], List[int]]:
    """Randomly sample n common problems from datasets and create temp JSONL files
    
    Args:
        files: List of JSONL files
        n_problems: Number of problems to sample
        
    Returns:
        (temp_files, sampled_problem_nums): list of temp files and sampled problem indices
    """
    if not files:
        raise ValueError("File list is empty")
    
    # Compute common problem indices across all files
    all_problems = None
    for file_path in files:
        data = load_jsonl_file(file_path)
        file_problems = set(item.problem_num for item in data)
        if all_problems is None:
            all_problems = file_problems
        else:
            all_problems &= file_problems
    
    if not all_problems:
        raise ValueError("No common problems found")
    
    if len(all_problems) < n_problems:
        raise ValueError(f"Common problems ({len(all_problems)}) fewer than requested samples ({n_problems})")
    
    # Randomly sample n problems
    sampled_problems = random.sample(list(all_problems), n_problems)
    
    # Create temp dir
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Create per-file temp JSONL containing only sampled problems
    temp_files = []
    for file_path in files:
        data = load_jsonl_file(file_path)
        sampled_data = [item for item in data if item.problem_num in sampled_problems]
        
        # Build temp filename
        original_path = Path(file_path)
        temp_filename = temp_dir / f"temp_{original_path.stem}_n{n_problems}_{random.randint(1000, 9999)}.jsonl"
        temp_files.append(str(temp_filename))
        
        # Write JSONL
        with open(temp_filename, 'w', encoding='utf-8') as f:
            for item in sampled_data:
                # Convert AnalysisData-like object to dict and write JSON
                import json
                item_dict = {
                    'problem_num': item.problem_num,
                    'total_answers': item.total_answers,
                    'answer_counts': item.answer_counts,
                    'gold_answer': item.gold_answer,
                    'majority_answer': item.majority_answer
                }
                f.write(json.dumps(item_dict, ensure_ascii=False) + '\n')
    
    return temp_files, sampled_problems


def evaluate_ensemble(files: List[str], n_problems: int) -> float:
    """Evaluate ensemble performance on all data using weights learned on n problems
    
    Args:
        files: List of JSONL files
        n_problems: Number of problems used for training weights
        
    Returns:
        Ensemble accuracy (0.0-1.0)
    """
    try:
        # Sample n problems
        temp_files, sampled_problems = sample_problems_from_datasets(files, n_problems)
        
        try:
            # Learn weights on sampled data
            with suppress_stdout_stderr():
                result = optimal_weight_binarysearch(temp_files, verbose=False)
            learned_weights = result['weights_array']
            
            # Evaluate learned weights on full data
            with suppress_stdout_stderr():
                full_score = get_score(files, learned_weights)
            
            
            # Compute total number of common problems across all files
            first_data = load_jsonl_file(files[0])
            all_problems = set(item.problem_num for item in first_data)
            for file_path in files[1:]:
                data = load_jsonl_file(file_path)
                file_problems = set(item.problem_num for item in data)
                all_problems &= file_problems
            
            total_problems = len(all_problems)
            ensemble_accuracy = full_score / total_problems if total_problems > 0 else 0.0
            
            return ensemble_accuracy
            
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink()
                except FileNotFoundError:
                    pass
    
    except Exception as e:
        print(f"Warning: error during evaluation at n={n_problems}: {e}")
        return 0.0


def run_dataset_split_analysis(files: List[str], n_trials: int = 100) -> None:
    """Run dataset split analysis and save plot
    
    Args:
        files: List of JSONL files
        n_trials: Number of trials per n
    """
    print(f"\n=== Dataset split analysis ===")
    print(f"Files: {files}")
    print(f"Trials: {n_trials}")
    
    # Get number of common problems across all data
    first_data = load_jsonl_file(files[0])
    all_problems = set(item.problem_num for item in first_data)
    for file_path in files[1:]:
        data = load_jsonl_file(file_path)
        file_problems = set(item.problem_num for item in data)
        all_problems &= file_problems
    
    max_problems = len(all_problems)
    print(f"#Common problems: {max_problems}")
    
    if max_problems < 1:
        print("Error: no common problems found")
        return
    
    # Collect each LLM's single-model performance on common problems
    single_accuracies = []
    llm_names = []
    for file_path in files:
        data = load_jsonl_file(file_path)
        # Filter by common problems only
        filtered_data = [item for item in data if item.problem_num in all_problems]
        stats = analyze_results(filtered_data)
        single_accuracies.append(stats['accuracy'])
        _dataset, llm_name = infer_dataset_and_llm_from_path(file_path)
        llm_names.append(llm_name)
    
    best_single_accuracy = max(single_accuracies)
    print(f"Best single LLM accuracy: {best_single_accuracy:.3f}")
    print("Per-LLM accuracy:")
    for llm_name, acc in zip(llm_names, single_accuracies):
        print(f"  {llm_name}: {acc:.3f}")
    
    # Analyze for n from 1 to max_problems
    n_values = []
    ensemble_accuracies = []
    ensemble_cis = []  # Approx. 95% CI: 2 * sd / sqrt(n_trials)
    
    # Limit max problems for runtime
    max_n = min(max_problems, 200)
    skip = 1
    if max_n > 50:
        skip = 5
    for n in range(1, max_n + 1, skip):
        print(f"\nStart analysis at n={n}...")
        
        trial_accuracies = []
        for trial in range(n_trials):
            if trial % 20 == 0:
                print(f"  Trial {trial + 1}/{n_trials}")
            
            accuracy = evaluate_ensemble(files, n)
            trial_accuracies.append(accuracy)
        
        mean_accuracy = np.mean(trial_accuracies)
        # CI width (approx): 2 * std / sqrt(n_trials)
        std_accuracy = float(np.std(trial_accuracies, ddof=1)) if len(trial_accuracies) > 1 else 0.0
        ci_width = 2.0 * std_accuracy / (np.sqrt(n_trials) if n_trials > 0 else 1.0)
        n_values.append(n)
        ensemble_accuracies.append(mean_accuracy)
        ensemble_cis.append(ci_width)
        
        print(f"n={n}: mean ensemble accuracy = {mean_accuracy:.3f}")
    
    # Plot results
    plt.figure(figsize=(6, 3))
    # Plot with error bars
    plt.errorbar(
        n_values,
        ensemble_accuracies,
        yerr=ensemble_cis,
        fmt='o-',
        color='blue',
        ecolor='blue',
        elinewidth=1,
        capsize=3,
        markersize=4,
        linewidth=2,
        label='Ensemble'
    )
    
    # Add each single LLM performance as dashed lines
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (llm_name, acc) in enumerate(zip(llm_names, single_accuracies)):
        color = colors[i % len(colors)]
        # Simplify display name from filename
        display_name = abbreviate_llm(llm_name.replace('analysis_', '').replace('_', ' '))
        if len(display_name) > 30:
            display_name = display_name[:27] + '...'
        
        plt.axhline(y=acc, color=color, linestyle='--', alpha=0.7, 
                   label=f'{display_name}: {acc:.3f}')
    
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('Ensemble Accuracy', fontsize=12)
    #plt.title(f'Learning Curve: Training Sample Size vs Ensemble Accuracy\n(Trials: {n_trials}, Best Single LLM Accuracy: {best_single_accuracy:.3f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis range considering error bars
    min_accuracy = min(single_accuracies)
    if ensemble_accuracies:
        ci_max = max((a + c for a, c in zip(ensemble_accuracies, ensemble_cis)), default=max(ensemble_accuracies))
        ci_min = min((a - c for a, c in zip(ensemble_accuracies, ensemble_cis)), default=min(ensemble_accuracies))
        max_accuracy = max(max(single_accuracies), ci_max)
        min_with_ci = min(min_accuracy, ci_min)
    else:
        max_accuracy = max(single_accuracies)
        min_with_ci = min_accuracy
    y_margin = 0.02  # 2% margin
    data_range = max_accuracy - min_with_ci if max_accuracy > min_with_ci else 0.1
    plt.ylim(
        max(0, min_with_ci - data_range * 0.1),
        min(1, max_accuracy + data_range * 0.1)
    )
    plt.xlim(1, max(n_values))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ensure plots directory exists
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Build filename
    file_stems = [Path(f).stem for f in files]
    plot_filename = plots_dir / f"learn_{'_'.join(file_stems)}.png"
    plot_filename = filename_compress(plot_filename)
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved plot: {plot_filename}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Min n (n={min(n_values)}): {ensemble_accuracies[0]:.3f}")
    print(f"Max n (n={max(n_values)}): {ensemble_accuracies[-1]:.3f}")
    if len(ensemble_accuracies) > 1:
        max_accuracy = max(ensemble_accuracies)
        max_n = n_values[ensemble_accuracies.index(max_accuracy)]
        print(f"Best ensemble accuracy: {max_accuracy:.3f} (n={max_n})")
        print(f"Gain over best single LLM: {max_accuracy - best_single_accuracy:+.3f}")


def run_transfer_learning_evaluation(source_dataset: str, target_dataset: str, combination_size: int) -> None:
    """Run transfer learning evaluation
    
    Args:
        source_dataset: Source dataset for learning weights
        target_dataset: Target dataset for evaluation
        combination_size: Size of combinations
    """
    print(f"\n=== Transfer learning evaluation ===")
    print(f"Source dataset: {source_dataset}")
    print(f"Target dataset: {target_dataset}")
    print(f"Combination size: {combination_size}")
    
    # Find common LLMs
    common_llms = find_common_llms(source_dataset, target_dataset)
    
    if not common_llms:
        print(f"Error: no common LLMs found between '{source_dataset}' and '{target_dataset}'")
        return
    
    if len(common_llms) < combination_size:
        print(f"Error: number of common LLMs ({len(common_llms)}) is less than combination size ({combination_size})")
        return
    
    print(f"\n#Common LLMs: {len(common_llms)}")
    print("Common LLMs:")
    for llm in common_llms:
        print(f"  - {llm}")
    
    # Generate combinations
    combinations = list(itertools.combinations(common_llms, combination_size))
    print(f"\n#Combinations: {len(combinations)}")
    
    # Accumulate results
    results = []
    
    for i, combo in enumerate(combinations, 1):
        print(f"\n--- Combination {i}/{len(combinations)} ---")
        combo_llms = list(combo)
        print(f"LLM: {', '.join(combo_llms)}")
        
        # Build file paths for source dataset
        source_files = [f"jsonl/analysis_{source_dataset}_{llm}.jsonl" for llm in combo_llms]
        
        # Build file paths for target dataset
        target_files = [f"jsonl/analysis_{target_dataset}_{llm}.jsonl" for llm in combo_llms]
        
        # Check file existence
        missing_source = [f for f in source_files if not Path(f).exists()]
        missing_target = [f for f in target_files if not Path(f).exists()]
        
        if missing_source or missing_target:
            print(f"Warning: missing files")
            if missing_source:
                print(f"  Source: {missing_source}")
            if missing_target:
                print(f"  Target: {missing_target}")
            continue
        
        try:
            # Learn weights on source dataset
            with suppress_stdout_stderr():
                source_result = optimal_weight_binarysearch(source_files)
            learned_weights = source_result['weights_array']
            print("learned_weights: ", learned_weights)
            
            # Get single-LLM performance on target dataset
            target_single_accuracies = []
            for file_path in target_files:
                accuracy = get_single_llm_accuracy(file_path)
                target_single_accuracies.append(accuracy)
            
            max_target_single_accuracy = max(target_single_accuracies)
            
            # Evaluate learned weights on target dataset
            with suppress_stdout_stderr():
                target_score = get_score(target_files, learned_weights)
            
            # Get total number of common problems in target dataset
            first_target_data = load_jsonl_file(target_files[0])
            
            # Compute intersection of problem indices
            all_target_problems = set(item.problem_num for item in first_target_data)
            for file_path in target_files[1:]:
                data = load_jsonl_file(file_path)
                file_problems = set(item.problem_num for item in data)
                all_target_problems &= file_problems
            
            total_target_problems = len(all_target_problems)
            target_accuracy = target_score / total_target_problems if total_target_problems > 0 else 0.0
            
            print(f"LLM: {', '.join(combo_llms)} -> source: {source_result['accuracy']:.3f}, target single max: {max_target_single_accuracy:.3f}, target weighted: {target_accuracy:.3f} (gain: {target_accuracy - max_target_single_accuracy:+.3f})")
            
            # Record result
            results.append({
                'llms': combo_llms,
                'source_accuracy': source_result['accuracy'],
                'target_single_max': max_target_single_accuracy,
                'target_weighted': target_accuracy,
                'improvement': target_accuracy - max_target_single_accuracy,
                'weights': learned_weights
            })
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results as a table
    if results:
        print(f"\n=== Transfer learning summary table ===")
        print(f"{'LLM combination':<60} {'source':<10} {'target single max':<15} {'target weighted':<15} {'gain':<8}")
        print("-" * 115)
        
        for result in results:
            combo_str = ", ".join(result['llms'])
            if len(combo_str) > 57:
                combo_str = combo_str[:54] + "..."
            
            print(f"{combo_str:<60} {result['source_accuracy']:.3f}      {result['target_single_max']:.3f}           {result['target_weighted']:.3f}           {result['improvement']:+.3f}")
        
        # Summary stats after the table
        total_rows = len(results)
        pos_count = sum(1 for r in results if r['improvement'] > 0)
        neg_count = sum(1 for r in results if r['improvement'] < 0)
        pos_ratio = (pos_count / total_rows) if total_rows > 0 else 0.0
        neg_ratio = (neg_count / total_rows) if total_rows > 0 else 0.0
        print(f"\nTotal rows: {total_rows}")
        print(f"Positive gain ratio: {pos_ratio:.3f} ({pos_count}/{total_rows})")
        print(f"Negative gain ratio: {neg_ratio:.3f} ({neg_count}/{total_rows})")
        
        # Show best result
        best_result = max(results, key=lambda x: x['target_weighted'])
        print(f"\n=== Best transfer learning result ===")
        print(f"LLM: {', '.join(best_result['llms'])}")
        print(f"Source dataset accuracy: {best_result['source_accuracy']:.3f}")
        print(f"Target single best: {best_result['target_single_max']:.3f}")
        print(f"Target weighted: {best_result['target_weighted']:.3f}")
        print(f"Gain: {best_result['improvement']:+.3f}")
        print(f"Learned weights: {[f'{w:.3f}' for w in best_result['weights']]}")


def run_combination_optimization(files: List[str], combination_size: int) -> None:
    """Generate combinations from files and run MILP optimization
    
    Args:
        files: List of JSONL files
        combination_size: Combination size
    """
    if len(files) < combination_size:
        print(f"Error: number of files ({len(files)}) is less than combination size ({combination_size})")
        return
    
    if combination_size < 2:
        print(f"Error: combination size must be at least 2")
        return
    
    # Generate combinations
    combinations = list(itertools.combinations(files, combination_size))
    print(f"\n=== Dataset combination optimization ===")
    print(f"#Files: {len(files)}")
    print(f"Combination size: {combination_size}")
    print(f"#Combinations: {len(combinations)}")
    
    # Accumulate results
    results = []
    
    for i, combo in enumerate(combinations, 1):
        print(f"\n--- Combination {i}/{len(combinations)} ---")
        combo_files = list(combo)
        
        # Get single accuracy per file
        single_accuracies = []
        llm_names = []
        for file_path in combo_files:
            accuracy = get_single_llm_accuracy(file_path)
            single_accuracies.append(accuracy)
            llm_name = Path(file_path).stem.replace(f"analysis_{Path(file_path).stem.split('_')[1]}_", "")
            llm_names.append(llm_name)
        
        max_single_accuracy = max(single_accuracies)
        
        # Run MILP optimization
        try:
            # Validate file existence
            existing_files = []
            for file in combo_files:
                if Path(file).exists():
                    existing_files.append(file)
                else:
                    print(f"Warning: file not found {file}")
            
            if len(existing_files) >= 2:
                with suppress_stdout_stderr():
                    result = optimal_weight_binarysearch(existing_files)
                optimized_accuracy = result['accuracy']
                
                # Record result
                results.append({
                    'combination': [Path(f).name for f in combo_files],
                    'llm_names': llm_names,
                    'single_accuracies': single_accuracies,
                    'max_single_accuracy': max_single_accuracy,
                    'optimized_accuracy': optimized_accuracy,
                    'improvement': optimized_accuracy - max_single_accuracy,
                    'weights': result['weights_array']
                })
                
                print(f"LLM: {', '.join(llm_names)} -> single best: {max_single_accuracy:.3f}, optimized: {optimized_accuracy:.3f} (gain: {optimized_accuracy - max_single_accuracy:+.3f})")
                
            else:
                print(f"Error: not enough valid files")
                
        except Exception as e:
            print(f"Error: optimization failed: {e}")
    
    # Print results in table format
    if results:
        print(f"\n=== Summary table ===")
        print(f"{'Combination':<60} {'Single best':<12} {'Optimized':<10} {'Gain':<8}")
        print("-" * 90)
        
        for result in results:
            combo_str = ", ".join(result['llm_names'])
            if len(combo_str) > 57:
                combo_str = combo_str[:54] + "..."
            
            print(f"{combo_str:<60} {result['max_single_accuracy']:.3f}    {result['optimized_accuracy']:.3f}    {result['improvement']:+.3f}")
        
        # Summary stats after the table
        total_rows = len(results)
        pos_count = sum(1 for r in results if r['improvement'] > 0)
        neg_count = sum(1 for r in results if r['improvement'] < 0)
        pos_ratio = (pos_count / total_rows) if total_rows > 0 else 0.0
        neg_ratio = (neg_count / total_rows) if total_rows > 0 else 0.0
        print(f"\nTotal rows: {total_rows}")
        print(f"Positive gain ratio: {pos_ratio:.3f} ({pos_count}/{total_rows})")
        print(f"Negative gain ratio: {neg_ratio:.3f} ({neg_count}/{total_rows})")
        
        # Show best combination
        best_result = max(results, key=lambda x: x['optimized_accuracy'])
        print(f"\n=== Best combination ===")
        print(f"LLM: {', '.join(best_result['llm_names'])}")
        print(f"Single best accuracy: {best_result['max_single_accuracy']:.3f}")
        print(f"Optimized accuracy: {best_result['optimized_accuracy']:.3f}")
        print(f"Gain: {best_result['improvement']:+.3f}")
        print(f"Weights: {[f'{w:.3f}' for w in best_result['weights']]}")


def run_milp_optimization_test(target_files: List[str], weights: List[float] = None) -> None:
    """Run MILP optimization test (with margin search)
    
    Args:
        target_files: List of JSONL files used for optimization
        weights: Manually specified weights (if None, run optimization)
    """
    print(f"\n=== MILP optimization test ===")
    try:
        # Check file existence
        existing_files = []
        for file in target_files:
            if Path(file).exists():
                existing_files.append(file)
            else:
                print(f"Warning: file not found {file}")
        
        if len(existing_files) >= 2:
            print(f"Using files: {existing_files}")
            
            if weights is not None:
                # If weights are specified: evaluate via get_score
                print(f"\n=== Evaluation with specified weights ===")
                print(f"Weights: {[f'{w:.4f}' for w in weights]}")
                print(f"Sum of weights: {sum(weights):.4f}")
                
                # Validate weights
                if len(weights) != len(existing_files):
                    print(f"Error: number of weights ({len(weights)}) does not match number of files ({len(existing_files)})")
                    return
                
                if abs(sum(weights) - 1.0) > 1e-6:
                    print(f"Warning: weights do not sum to 1.0: {sum(weights):.6f}")
                
                # Evaluate constraints satisfied via get_score
                with suppress_stdout_stderr():
                    satisfied_constraints = get_score(existing_files, weights)
                
                # Get total number of problems from the first file
                first_data = load_jsonl_file(existing_files[0])
                
                # Compute number of common problems
                all_problems = set(item.problem_num for item in first_data)
                for file_path in existing_files[1:]:
                    data = load_jsonl_file(file_path)
                    file_problems = set(item.problem_num for item in data)
                    all_problems &= file_problems
                
                total_problems = len(all_problems)
                accuracy = satisfied_constraints / total_problems if total_problems > 0 else 0.0
                
                print(f"\n=== Evaluation result ===")
                print(f"Satisfied constraints: {satisfied_constraints} / {total_problems}")
                print(f"Constraint satisfaction rate: {accuracy:.3f}")
                
                # Compare with single-LLM accuracy
                print(f"\n=== Comparison with single accuracies ===")
                for i, file in enumerate(existing_files):
                    data = load_jsonl_file(file)
                    # Filter by common problems only
                    filtered_data = [item for item in data if item.problem_num in all_problems]
                    stats = analyze_results(filtered_data)
                    llm_name = Path(file).stem
                    print(f"  {llm_name}:")
                    print(f"    Single accuracy: {stats['accuracy']:.3f}")
                    print(f"    Specified weight: {weights[i]:.4f}")
                
                print(f"  Weighted constraint satisfaction rate: {accuracy:.3f}")
                
            else:
                # No weights specified: run optimization
                result = optimal_weight_binarysearch(existing_files, verbose=True)
                
                print(f"Model status: {result['model_status']}")
                print(f"Correct problems: {result['correct_problems']:.1f} / {result['total_problems']}")
                print(f"Optimized accuracy: {result['accuracy']:.3f}")
                print(f"Objective value: {result['objective_value']:.1f}")
                
                print(f"\n=== Optimal weights ===")
                for llm, weight in result['optimal_weights'].items():
                    print(f"  {llm}: {weight:.4f}")
                
                # Compare with single-LLM accuracies
                print(f"\n=== Comparison with single accuracies ===")
                for i, file in enumerate(existing_files):
                    data = load_jsonl_file(file)
                    stats = analyze_results(data)
                    llm_name = Path(file).stem
                    weight = result['optimal_weights'][llm_name]
                    print(f"  {llm_name}:")
                    print(f"    Single accuracy: {stats['accuracy']:.3f}")
                    print(f"    Optimal weight: {weight:.4f}")
                
                print(f"  Optimized accuracy: {result['accuracy']:.3f}")

                # Show weight details
                weights_array = result['weights_array']
                print(f"\n=== Weights array ===")
                print(f"  Weights: {[f'{w:.4f}' for w in weights_array]}")
                print(f"  Sum: {sum(weights_array):.4f}")
            
        else:
            print("At least two files are required for optimization")
            print(f"Found files: {existing_files}")
            
    except Exception as e:
        print(f"Optimization error: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main function - sample usage"""
    parser = argparse.ArgumentParser(
        description='Read JSONL analysis files and run MILP optimization tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run default test
  python train_main.py

  # Run MILP optimization test on specific files
  python train_main.py analysis_aime2025_Qwen3-14B.jsonl analysis_aime2025_gpt-oss-20b.jsonl

  # Evaluate get_score with specified weights (no optimization)
  python train_main.py file1.jsonl file2.jsonl --weights 0.3,0.7

  # Skip basic tests and run only MILP optimization
  python train_main.py --skip-basic file1.jsonl file2.jsonl file3.jsonl

  # Dataset-wise combination optimization (default size 3)
  python train_main.py --dataset-target aime2025

  # Dataset-wise combination optimization (specify size 4)
  python train_main.py --dataset-target math500 --dataset-size 4

  # Transfer learning evaluation (learn on aime2024 weights and evaluate on aime2025)
  python train_main.py --dataset-source aime2024 --dataset-target aime2025

  # Dataset split analysis (analyze relation between #training samples and ensemble accuracy)
  python train_main.py --dataset-target aime2025 --dataset-split --n-trials 100

  # Dataset split analysis on specified files
  python train_main.py analysis_aime2025_gpt-oss-20b.jsonl analysis_aime2025_Phi-4-reasoning.jsonl --dataset-split
"""
    )
    
    parser.add_argument(
        'milp_files',
        nargs='*',
        help='List of JSONL files for MILP optimization (need at least 2)'
    )
    
    parser.add_argument(
        '--skip-basic',
        action='store_true',
        help='Skip basic tests (answer normalization and file loading)'
    )
    
    parser.add_argument(
        '--weights',
        help='Weights per LLM (comma-separated, e.g., 0.3,0.5,0.2). If specified, evaluate with get_score without optimization'
    )
    
    parser.add_argument(
        '--dataset-target',
        help='Dataset name (e.g., aime2025, aime2024, gpqa_diamond, math500). If set, corresponding files are searched automatically'
    )
    
    parser.add_argument(
        '--dataset-source',
        help='Source dataset name for learning weights. Used with --dataset-target for transfer learning evaluation'
    )
    
    parser.add_argument(
        '--dataset-size',
        type=int,
        default=3,
        help='Combination size for dataset-wise optimization (default: 3)'
    )
    
    parser.add_argument(
        '--dataset-split',
        action='store_true',
        help='Run dataset split analysis (relation between #training samples and ensemble performance)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of trials for dataset split analysis (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Run basic tests
    if not args.skip_basic:
        # Test answer normalization utilities
        test_normalize_answers()
        test_answer_counts_normalization()
        
        print("\n=== JSONL file loading test ===")
        
        # Example: load a single file
        try:
            sample_file = "analysis_aime2025_Qwen3-14B.jsonl"
            print(f"\nLoad single file: {sample_file}")
            data = load_jsonl_file(sample_file)
            print(f"Number of records: {len(data)}")
            
            # Show first 3 records
            for i, item in enumerate(data[:3]):
                print(f"  {i}: {item}")
            
            # Show statistics
            stats = analyze_results(data)
            print(f"\nStatistics:")
            print(f"  Total problems: {stats['total_problems']}")
            print(f"  Correct answers: {stats['correct_answers']}")
            print(f"  Accuracy: {stats['accuracy']:.3f}")
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")
            
        except FileNotFoundError:
            print(f"File {sample_file} not found")
        
        # Example: load all files
        print(f"\nLoad all files:")
        all_data = load_all_analysis_files()
        
        for filename, data in all_data.items():
            stats = analyze_results(data)
            print(f"  {filename}: accuracy {stats['accuracy']:.3f} ({stats['total_problems']} problems)")
    
    # Parse weights if specified
    parsed_weights = None
    if args.weights:
        try:
            parsed_weights = [float(w.strip()) for w in args.weights.split(',')]
            print(f"Specified weights: {parsed_weights}")
        except ValueError as e:
            print(f"Error: failed to parse weights: {args.weights}. Error: {e}")
            return
    
    # Dataset-related operations
    if args.dataset_target:
        # Transfer learning evaluation
        if args.dataset_source:
            print(f"\n=== Transfer learning evaluation mode ===")
            run_transfer_learning_evaluation(args.dataset_source, args.dataset_target, args.dataset_size)
        elif args.dataset_split:
            # Dataset split analysis
            print(f"\n=== Dataset split analysis mode ===")
            print(f"Dataset: {args.dataset_target}")
            print(f"Trials: {args.n_trials}")
            
            # Search matching files
            dataset_files = find_dataset_files(args.dataset_target)
            
            if not dataset_files:
                print(f"Error: no files found for dataset '{args.dataset_target}'")
                print(f"Search pattern: analysis_{args.dataset_target}_*.jsonl")
                return
            
            print(f"Found files: {len(dataset_files)}")
            for file in dataset_files:
                print(f"  - {file}")
            
            if len(dataset_files) < 2:
                print(f"Error: dataset split analysis requires at least 2 files")
                return
            
            # Run dataset split analysis
            run_dataset_split_analysis(dataset_files, args.n_trials)
        else:
            # Dataset-wise combination optimization
            print(f"\n=== Dataset-wise combination optimization ===")
            print(f"Dataset: {args.dataset_target}")
            print(f"Combination size: {args.dataset_size}")
            
            # Search matching files
            dataset_files = find_dataset_files(args.dataset_target)
            
            if not dataset_files:
                print(f"Error: no files found for dataset '{args.dataset_target}'")
                print(f"Search pattern: analysis_{args.dataset_target}_*.jsonl")
                return
            
            print(f"Found files: {len(dataset_files)}")
            for file in dataset_files:
                print(f"  - {file}")
            
            if len(dataset_files) < args.dataset_size:
                print(f"Error: number of files ({len(dataset_files)}) is less than combination size ({args.dataset_size})")
                return
            
            # Run combination optimization
            run_combination_optimization(dataset_files, args.dataset_size)
    elif args.dataset_source:
        print("Error: --dataset-source must be used together with --dataset-target")
        return
    elif args.dataset_split and args.milp_files:
        # Dataset split analysis with files specified via CLI
        print(f"\n=== Dataset split analysis mode (specified files) ===")
        print(f"Files: {args.milp_files}")
        print(f"Trials: {args.n_trials}")
        
        if len(args.milp_files) < 2:
            print(f"Error: dataset split analysis requires at least 2 files")
            return
        
        # Run dataset split analysis
        run_dataset_split_analysis(args.milp_files, args.n_trials)
        return
        
    # Default MILP optimization test
    elif args.milp_files:
        # Use files specified via CLI
        run_milp_optimization_test(args.milp_files, parsed_weights)
    else:
        # Use default file list
        default_target_files = [
            "analysis_aime2025_Qwen3-14B.jsonl", 
            "analysis_aime2025_gpt-oss-20b.jsonl", 
            "analysis_aime2025_Qwen3-30B-A3B-Thinking-2507.jsonl"
        ]
        run_milp_optimization_test(default_target_files, parsed_weights)


if __name__ == "__main__":
    main()