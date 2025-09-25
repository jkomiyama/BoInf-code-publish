#!/usr/bin/env python3

"""
Functions to compute and plot majority-vote accuracy per problem.

Example:
    python calculate_accuracy_per_problem.py analysis_aime2025_Qwen3-30B-A3B-Thinking-2507.jsonl
"""

import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Import functions from ensemble_utils.py
from ensemble_utils import load_jsonl_file, AnalysisData


def sample_majority_vote(answer_counts: Dict[str, int], n_samples: int, gold_answer: str, n_trials: int = 100) -> float:
    """
    Perform majority voting with n samples (with replacement) and compute accuracy.
    
    Args:
        answer_counts: Frequency per answer
        n_samples: Number of samples
        gold_answer: Gold answer
        n_trials: Number of trials (default: 100)
    
    Returns:
        Accuracy (0.0-1.0)
    """
    if n_samples <= 0:
        return 0.0
    
    # Build a list of answers (with multiplicity according to frequency)
    all_answers = []
    for answer, count in answer_counts.items():
        all_answers.extend([answer] * count)
    
    if len(all_answers) == 0:
        return 0.0
    
    correct_count = 0
    
    for _ in range(n_trials):
        # Randomly sample n answers (with replacement)
        sampled = random.choices(all_answers, k=n_samples)
        
        # Compute majority vote
        counter = Counter(sampled)
        majority_answer = counter.most_common(1)[0][0]
        
        if majority_answer == gold_answer:
            correct_count += 1
    
    return correct_count / n_trials


def calculate_accuracy_per_problem(jsonl_file: str, n_values: List[int] = None, n_trials: int = 10000):
    """
    Compute and plot majority-vote accuracy per problem.
    
    Args:
        jsonl_file: Path to the analysis file
        n_values: List of sample counts (default uses log-like spacing)
        n_trials: Number of trials per n
    """
    # Load file
    try:
        data_list = load_jsonl_file(jsonl_file, clean_answers=True)
    except Exception as e:
        print(f"Error: failed to read file: {e}")
        return
    
    if not data_list:
        print("Error: data is empty")
        return
    
    # Configure n values (log-like scale)
    if n_values is None:
        max_total = 100 # max(item.total_answers for item in data_list)
        n_values = []
        n = 1
        while n <= max_total:
            n_values.append(n)
            if n < 10:
                n += 1
            elif n < 100:
                n += 5
            else:
                n += 10
            if n > max_total:
                if max_total not in n_values:
                    n_values.append(max_total)
                break
    
    print(f"n_values: {n_values}")
    print(f"#Problems: {len(data_list)}")
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Strip extension from filename
    file_stem = Path(jsonl_file).stem
    
    # Process each problem
    for problem_data in data_list:
        problem_num = problem_data.problem_num
        answer_counts = problem_data.answer_counts
        gold_answer = problem_data.gold_answer
        total_answers = problem_data.total_answers
        
        print(f"Problem {problem_num+1:02d}: total_answers={total_answers}, gold={gold_answer}")
        
        # Compute accuracy for each n
        accuracies = []
        valid_n_values = []
        
        for n in n_values:
            accuracy = sample_majority_vote(answer_counts, n, gold_answer, n_trials)
            accuracies.append(accuracy)
            valid_n_values.append(n)
            print(f"  n={n:3d}: accuracy={accuracy:.3f}")
        
        if not accuracies:
            print(f"  Warning: problem {problem_num+1} has no plottable data")
            continue
        
        # Create plot
        plt.figure(figsize=(4, 3))
        plt.semilogx(valid_n_values, accuracies, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Number of Samples (N)', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title(f'Problem {problem_num+1:02d}', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        
        # Configure x-axis limits
        if valid_n_values:
            plt.xlim(0.8 * min(valid_n_values), 1.2 * max(valid_n_values))
        
        plt.tight_layout()
        
        # Save file
        output_file = plots_dir / f"accuracy_per_problem_{file_stem}_problem{problem_num+1:02d}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
    
    print(f"\nAll problem plots finished. Results saved to {plots_dir}/ .")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python calculate_accuracy_per_problem.py <jsonl_file>")
        print("Example: python calculate_accuracy_per_problem.py analysis_aime2025_Qwen3-30B-A3B-Thinking-2507.jsonl")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    
    if not Path(jsonl_file).exists():
        print(f"Error: file {jsonl_file} not found")
        sys.exit(1)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    calculate_accuracy_per_problem(jsonl_file)


if __name__ == "__main__":
    main()
