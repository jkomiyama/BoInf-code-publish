#!/usr/bin/env python
# coding: utf-8

"""
BoN_batch.py - Batch command manager and executor for BoN_client.py

This script uses the following command template to run multiple instances of
BoN_client.py by substituting run numbers:

rm -rf bo32_aime_new;time python BoN_client.py -n 32 -o bo32_aime_new --test_file /workspace/AIME_2024/aime_2024_problems.parquet --dataset_type aime2024 --evaluation_method pick_answer | tee bo32_aime2024_new_pick_answer.txt

Output directory names and log file names are automatically adjusted according to the -n option (e.g., -n 5 → bo5_new).
Results are saved in the specified output directory (default: output_batch).
"""

import argparse
import os
import subprocess
import sys
import re
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
def load_dotenv():
    try:
        import importlib
        dotenv = importlib.import_module("dotenv")
        return getattr(dotenv, "load_dotenv", lambda: None)()
    except Exception:
        return None
from BoN_utils import extract_reward_model_name


def parse_args():
    parser = argparse.ArgumentParser(description='BoN_client.py batch execution manager')
    parser.add_argument('--start', type=int, default=1,
                      help='Start run number (default: 1)')
    parser.add_argument('--end', type=int, required=True,
                      help='End run number (required)')
    parser.add_argument('--test_file', type=str,
                      default='/workspace/AIME_2024/aime_2024_problems.parquet',
                      help='Path to test file (default: /workspace/AIME_2024/aime_2024_problems.parquet)')
    parser.add_argument('--dataset_type', type=str, default='aime2024',
                        choices=['gsm8k', 'aime2024', 'aime2024short', 'aime2025', 'math', 'math500', 'mmlu_pro', 'gpqa_diamond', 'auto'],
                        help='Dataset type (default: aime2024). Choices: gsm8k, aime2024, aime2024short, aime2025, math, math500, mmlu_pro, gpqa_diamond, auto')
    parser.add_argument('--output_dir', type=str, default='output_batch',
                      help='Output directory path (default: output_batch)')
    parser.add_argument('--output_base', type=str, default='bo1_new',
                      help='Base name for output directory (default: bo{N}_new, where N is -n)')
    parser.add_argument('--log_base', type=str, default='bo1_stdout',
                      help='Base name for log files (default: bo{N}_stdout, where N is -n)')
    parser.add_argument('--dry_run', action='store_true',
                      help='Show commands without executing them')
    parser.add_argument('--continue_on_error', action='store_true',
                      help='Continue even if an error occurs')
    parser.add_argument('--extra_args', type=str, default='',
                      help='Additional arguments to pass to BoN_client.py')
    parser.add_argument('--max_workers', type=int, default=1,
                      help='Number of parallel workers (default: 1, sequential)')
    parser.add_argument('-n', '--bon_samples', type=int, default=5,
                      help='Value for BoN_client.py -n option (samples, default: 5)')
    
    # Options specific to BoN_client.py
    parser.add_argument('--evaluation_method', type=str, 
                     choices=['omni', 'pick_answer', 'llm_judge_tournament', 'llm_judge_set', 'reward', 'reward_orm', 'self_certainty', 'random', 'majority'], default='pick_answer',
                     help='Evaluation method: omni (correctness-priority), pick_answer (LLM judgment), llm_judge_tournament (LLM tournament with groups of L), llm_judge_set (LLM judgment using last 5000 chars before </think>), reward (reward model), reward_orm (reward model without thinking), self_certainty (Self-Certainty score), random (uniform random), majority (majority vote; tie broken randomly). Default: pick_answer')
    parser.add_argument('--tournament_L', type=int, default=4,
                      help='Group size L for llm_judge_tournament (default: 4). While candidates >= L, split into groups of L, keep one per group, and repeat until one remains.')
    parser.add_argument('--use_save', action='store_true',
                      help='Enable answer saving (calls generate_answerBoN_save)')
    parser.add_argument('--file_start', type=int, default=0,
                      help='Starting file index for saved answers (default: 0)')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                      help='Maximum new tokens (default: auto-set based on dataset type)')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples (default: unlimited)')
    parser.add_argument('--first_valid', action='store_true', default=False,
                     help='For pick_answer/llm_judge_set: stop at first valid judgment (scale = -1)')
    
    # Legacy options (for compatibility with BoN_client.py)
    parser.add_argument('--use_with_summary', action='store_true',
                      help='Add --use_with_summary option (also changes output dir and log names)')
    parser.add_argument('--use_decomposition', action='store_true',
                      help='Add --use_decomposition option (also changes output dir and log names)')
    
    return parser.parse_args()

def create_output_directory(output_dir_path='output_batch'):
    """Create output directory"""
    output_dir = Path(output_dir_path)
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_command(r, test_file, dataset_type, output_base, log_base, bon_samples, 
                    evaluation_method='pick_answer', use_save=False, file_start=0,
                    use_with_summary=False, use_decomposition=False, max_new_tokens=None, 
                    max_samples=None, first_valid=False, extra_args='', output_dir_path='output_batch'):
    """Generate command with specified number"""
    # Debug information
    print(f"[DEBUG] Task{r}: use_save={use_save}, file_start={file_start}")
    
    # Rename according to functionality
    if evaluation_method == 'reward':
        # Get reward model name from environment variable
        reward_model_id = os.environ.get('REWARD_MODEL_ID', '')
        if not reward_model_id:
            print("[ERROR] Environment variable REWARD_MODEL_ID is not set. Setting is required to use --evaluation_method reward.")
            sys.exit(1)
        reward_model_name = extract_reward_model_name(reward_model_id)
        suffix = f"_reward_{reward_model_name}"
    elif evaluation_method == 'reward_orm':
        # Get reward model name from environment variable (remove thinking in orm mode)
        reward_model_id = os.environ.get('REWARD_MODEL_ID', '')
        if not reward_model_id:
            print("[ERROR] Environment variable REWARD_MODEL_ID is not set. Setting is required to use --evaluation_method reward_orm.")
            sys.exit(1)
        reward_model_name = extract_reward_model_name(reward_model_id)
        suffix = f"_reward_orm_{reward_model_name}"
    # removed: comp_reward suffix (deprecated)
    elif evaluation_method == 'self_certainty':
        suffix = "_self_certainty"
    else:
        suffix = f"_{evaluation_method}"  # Always include evaluation_method
    
    if use_save:
        suffix += "_save"
    if use_with_summary:
        suffix += "_withsummary"
    if use_decomposition:
        suffix += "_decomposition"
    
    output_dir = f"{output_dir_path}/{output_base}_{dataset_type}{suffix}_run{r}"
    log_file = f"{log_base}_{dataset_type}{suffix}_run{r}.txt"
    
    # Basic command
    cmd = f"rm -rf {output_dir}; time python BoN_client.py -n {bon_samples} -o {output_dir} --test_file {test_file} --dataset_type {dataset_type}"
    
    # Options specific to BoN_client.py
    cmd += f" --evaluation_method {evaluation_method}"
    if evaluation_method == 'llm_judge_tournament':
        # Pass L to client (can also be specified via environment variable TOURNAMENT_L)
        # Allow explicit specification of --tournament_L in extra_args
        # Only add if not included in extra_args
        if '--tournament_L' not in (extra_args or ''):
            tournament_L = os.environ.get('TOURNAMENT_L')
            if tournament_L and tournament_L.isdigit():
                cmd += f" --tournament_L {tournament_L}"
    cmd += f" --run {r}"
    
    if use_save:
        cmd += " --use_save"
        cmd += f" --file_start {file_start}"
    
    # Token count limit
    if max_new_tokens is not None:
        cmd += f" --max_new_tokens {max_new_tokens}"
    
    # Sample count limit
    if max_samples is not None:
        cmd += f" --max_samples {max_samples}"
    
    # first_valid option
    if first_valid:
        cmd += " --first_valid"
    
    # Existing options
    if use_with_summary:
        cmd += " --use_with_summary"
    
    if use_decomposition:
        cmd += " --use_decomposition"
    
    # Add additional arguments if any
    if extra_args and extra_args.strip():
        cmd += f" {extra_args.strip()}"
    
    # Log output to specified directory (including standard error)
    cmd += f" 2>&1 | tee {output_dir_path}/{log_file}"
    
    return cmd


# Lock for thread-safe output
print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe output"""
    with print_lock:
        print(*args, **kwargs)

def execute_command(cmd, n, dry_run=False):
    """Execute a command"""
    thread_safe_print(f"[Run {n}] Command: {cmd}")
    thread_safe_print(f"[Run {n}] " + "-" * 60)
    
    if dry_run:
        thread_safe_print(f"[Run {n}] [DRY RUN] Not actually executed")
        return n, True, None
    
    try:
        # Execute command with bash
        process = subprocess.run(cmd, shell=True, executable='/bin/bash', 
                               capture_output=False, text=True)
        
        if process.returncode == 0:
            thread_safe_print(f"[Run {n}] ✓ Completed successfully (exit code: {process.returncode})")
            return n, True, None
        else:
            error_msg = f"Exit code: {process.returncode}"
            thread_safe_print(f"[Run {n}] ✗ Error occurred ({error_msg})")
            return n, False, error_msg
            
    except Exception as e:
        error_msg = str(e)
        thread_safe_print(f"[Run {n}] ✗ Execution error: {error_msg}")
        return n, False, error_msg


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    args = parse_args()
    
    # Record execution start time
    start_time = None
    hostname = None
    
    # Validate arguments
    if args.start > args.end:
        print("Error: Start number must be <= end number")
        sys.exit(1)
    
    # Change default value of test_file according to dataset_type
    if args.dataset_type == 'gsm8k' and args.test_file == '/workspace/AIME_2024/aime_2024_problems.parquet':
        args.test_file = 'templategsm_random.jsonl'
    
    if args.dataset_type == 'aime2025' and args.test_file == '/workspace/AIME_2024/aime_2024_problems.parquet':
        args.test_file = '/workspace/AIME2025/aime2025-full.jsonl'
    
    if args.dataset_type == 'math' and args.test_file == '/workspace/AIME_2024/aime_2024_problems.parquet':
        args.test_file = '/workspace/math/geometry/test-00000-of-00001.parquet'
    
    if args.dataset_type == 'math500' and args.test_file == '/workspace/AIME_2024/aime_2024_problems.parquet':
        args.test_file = '/workspace/prm800k/prm800k/math_splits/test.jsonl'
    
    if args.dataset_type == 'mmlu_pro' and args.test_file == '/workspace/AIME_2024/aime_2024_problems.parquet':
        args.test_file = '/workspace/MMLU-Pro/data/validation-00000-of-00001.parquet'
    
    if args.dataset_type == 'gpqa_diamond' and args.test_file == '/workspace/AIME_2024/aime_2024_problems.parquet':
        args.test_file = '/workspace/GPQA-Diamond/test/gpqa_diamond.parquet'
    
    # Dynamically change output_base and log_base according to -n option
    # Change only when user is using default values
    if args.output_base == 'bo1_new':
        args.output_base = f'bo{args.bon_samples}_new'
    
    if args.log_base in ('bo1_stdout_new', 'bo1_stdout'):
        args.log_base = f'bo{args.bon_samples}_stdout'
    
    print("=" * 80)
    print("BoN_client.py Batch Execution Manager")
    print("=" * 80)
    print(f"Execution range: {args.start} to {args.end}")
    print(f"Test file: {args.test_file}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output base: {args.output_base}")
    print(f"Log base: {args.log_base}")
    print(f"Parallel execution count: {args.max_workers}")
    print(f"BoN sample count: {args.bon_samples}")
    print(f"Evaluation method: {args.evaluation_method}")
    print(f"Run numbers: Use each task number as is ({args.start} to {args.end})")
    print(f"Answer save function: {'Enabled' if args.use_save else 'Disabled'}")
    if args.use_save:
        print(f"File start number: {args.file_start}")
    print(f"Max token count: {args.max_new_tokens if args.max_new_tokens else 'Auto-set according to dataset type'}")
    print(f"Max sample count: {args.max_samples if args.max_samples else 'Unlimited'}")
    print(f"first_valid: {'Enabled' if args.first_valid else 'Disabled'} (for pick_answer/llm_judge_set)")
    print(f"use_with_summary: {'Enabled' if args.use_with_summary else 'Disabled'}")
    print(f"use_decomposition: {'Enabled' if args.use_decomposition else 'Disabled'}")
    if args.extra_args:
        print(f"Additional arguments: {args.extra_args}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")
    print("=" * 80)
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    # Execution statistics
    total_count = 0
    success_count = 0
    failed_numbers = []
    failed_details = {}
    
    # Create task list to execute
    tasks = []
    for n in range(args.start, args.end + 1):
        # Calculate file_start for each task: base file_start + (task number - start number) * bon_samples
        task_file_start = args.file_start + (n - args.start) * args.bon_samples
        
        cmd = generate_command(n, args.test_file, args.dataset_type, 
                             args.output_base, args.log_base, args.bon_samples,
                             args.evaluation_method, args.use_save, task_file_start,
                             args.use_with_summary, args.use_decomposition, args.max_new_tokens, 
                             args.max_samples, args.first_valid, args.extra_args, args.output_dir)
        tasks.append((cmd, n))
        
        # Debug information: display file_start value for each task
        if args.use_save:
            print(f"Task{n}: file_start = {task_file_start}")
    
    print(f"\nExecuting a total of {len(tasks)} tasks...\n")
        
    # Parallel or sequential execution
    if args.max_workers == 1:
        # Sequential execution
        for cmd, n in tasks:
            print(f"[{n}/{args.end}] Executing number {n}...")
            
            # Execute command
            task_n, success, error_msg = execute_command(cmd, n, args.dry_run)
            
            total_count += 1
            if success:
                success_count += 1
            else:
                failed_numbers.append(task_n)
                if error_msg:
                    failed_details[task_n] = error_msg
                if not args.continue_on_error:
                    print(f"\nAn error occurred. Stopping at number {task_n}.")
                    print("Use the --continue_on_error flag to continue.")
                    break
            
            print()
    else:
        # Parallel execution
        print(f"Parallel execution count: {args.max_workers}")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(execute_command, cmd, n, args.dry_run): (cmd, n) 
                for cmd, n in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                cmd, n = future_to_task[future]
                try:
                    task_n, success, error_msg = future.result()
                    total_count += 1
                    
                    if success:
                        success_count += 1
                        thread_safe_print(f"[Progress] Completed: {success_count}/{len(tasks)}")
                    else:
                        failed_numbers.append(task_n)
                        if error_msg:
                            failed_details[task_n] = error_msg
                        thread_safe_print(f"[Progress] Failed: {task_n}, Completed: {total_count}/{len(tasks)}")
                        
                        if not args.continue_on_error:
                            # Cancel other tasks
                            for f in future_to_task:
                                f.cancel()
                            thread_safe_print(f"\nAn error occurred. Stopping at number {task_n}.")
                            thread_safe_print("Use the --continue_on_error flag to continue.")
                            break
                            
                except Exception as e:
                    thread_safe_print(f"[Number {n}] Error during task execution: {e}")
                    failed_numbers.append(n)
                    failed_details[n] = str(e)
                    total_count += 1
    
    # Execution results summary
    print("=" * 80)
    print("Execution Results Summary")
    print("=" * 80)
    print(f"Total executions: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if failed_numbers:
        print(f"Failed numbers: {', '.join(map(str, failed_numbers))}")
        if failed_details:
            print("\nFailure details:")
            for num, detail in failed_details.items():
                print(f"  Number {num}: {detail}")
    
    if not args.dry_run:
        print(f"Log files saved to {output_dir.absolute()}")
    
    print("=" * 80)
        
    # Display usage examples
    print("\nUsage examples:")
    print(f"# Basic usage (pick_answer evaluation)")
    print(f"python BoN_batch.py --end 5 --evaluation_method pick_answer")
    print(f"")
    print(f"# first_valid mode (stop at first valid judgment, efficient)")
    print(f"python BoN_batch.py --end 5 --evaluation_method pick_answer --first_valid")
    print(f"python BoN_batch.py --end 5 --evaluation_method llm_judge_set --first_valid")
    
    print(f"")
    print(f"# Tournament method (llm_judge_tournament evaluation, L=4)")
    print(f"python BoN_batch.py --end 5 --evaluation_method llm_judge_tournament --tournament_L 4")
    print(f"")
    
    print(f"")
    print(f"# Specify custom output directory and run number")
    print(f"python BoN_batch.py --end 5 --output_dir my_output_directory --run 2")
    print(f"")
    
    print(f"")
    print(f"# Use MATH dataset")
    print(f"python BoN_batch.py --end 5 --test_file math_sample.parquet --dataset_type math")
    print(f"")
    print(f"# Use MMLU-Pro dataset")
    print(f"python BoN_batch.py --end 5 --dataset_type mmlu_pro")
    print(f"")
    print(f"# Use GPQA-Diamond dataset")
    print(f"python BoN_batch.py --end 5 --dataset_type gpqa_diamond")
    print(f"")
    print(f"# Specify custom token count")
    print(f"python BoN_batch.py --end 5 --max_new_tokens 5000")
    print(f"")
    print(f"# Specify maximum sample count")
    print(f"python BoN_batch.py --end 5 --max_samples 100")
    print(f"")
    print(f"# Run with answer save function")
    print(f"python BoN_batch.py --end 3 --use_save --evaluation_method reward")
    print(f"")
    print(f"# Run with comparison-based reward model like RRM-7B")
    # removed: comp_reward example (deprecated)
    print(f"")
    print(f"# Run with Self-Certainty score evaluation")
    print(f"python BoN_batch.py --end 5 --evaluation_method self_certainty")
    print(f"")
    print(f"# Parallel execution (4 parallel)")
    print(f"python BoN_batch.py --end 10 --max_workers 4")
    print(f"")
    print(f"# Comparison of different evaluation methods")
    print(f"python BoN_batch.py --end 5 --evaluation_method omni")
    print(f"python BoN_batch.py --end 5 --evaluation_method pick_answer")
    print(f"python BoN_batch.py --end 5 --evaluation_method pick_answer --first_valid  # fast version")
    print(f"python BoN_batch.py --end 5 --evaluation_method llm_judge_set")
    print(f"python BoN_batch.py --end 5 --evaluation_method llm_judge_set --first_valid  # fast version")
    
    print(f"python BoN_batch.py --end 5 --evaluation_method llm_judge_tournament --tournament_L 4")
    # removed: pick_answer_tournament_norep example (deprecated)
    
    print(f"python BoN_batch.py --end 5 --evaluation_method reward")
    # removed: comp_reward example (deprecated)
    print(f"python BoN_batch.py --end 5 --evaluation_method self_certainty")
    print(f"python BoN_batch.py --end 5 --evaluation_method majority")
    
    print(f"")
    print(f"# Automatic run numbers (each task uses its own number as run number)")
    print(f"python BoN_batch.py --end 5 --evaluation_method pick_answer")
    print(f"# Task 1 uses --run 1, Task 2 uses --run 2, ..., Task 5 uses --run 5")
    print(f"# Example generated choice files: saved_choices/aime2024_run1_pick_answer.jsonl")
    print(f"#                                saved_choices/aime2024_run2_pick_answer.jsonl")
    print(f"")
    print("=" * 80)
    
    # Exit code
    if failed_numbers and not args.continue_on_error:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main() 