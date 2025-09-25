#!/usr/bin/env python
# coding: utf-8

"""
BoN_choice_analyze.py

Program to score files in the saved_choices directory (e.g., aime2024_run0_pick_answer_num5.jsonl)
and display the number of problems and average accuracy per scale.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import glob
import re
import math
from BoN_answeranalyze import extract_boxed_answer, normalize_math_notation

# Definition of is_correct (independent from BoN_utils.py)
def is_correct(answer: str, gold: str, dataset_type: str = None) -> bool:
    """
    Determine whether the generated answer is correct.
    # For AIME/MATH datasets, use extraction/normalization from BoN_answeranalyze
    if dataset_type in ["aime2024", "aime2024short", "aime2025", "math500", "math"]:
        gold_str = str(gold).strip()
        # Extract prediction (e.g., \boxed{...}). If not found, use the entire text
        extracted = extract_boxed_answer(answer)
        if extracted is None:
            extracted = answer.strip()

        # Normalization
        extracted_norm = normalize_math_notation(str(extracted)) if extracted is not None else ""
        gold_norm = normalize_math_notation(gold_str)

        # String equality
        if str(extracted).strip() == gold_str:
            return True
        # Equality after normalization
        if extracted_norm == gold_norm:
            return True
        # Numeric comparison (if possible)
        try:
            if abs(float(extracted_norm) - float(gold_norm)) < 1e-9:
                return True
        except (ValueError, TypeError):
            pass
        try:
            if abs(float(str(extracted).strip()) - float(gold_str)) < 1e-9:
                return True
        except (ValueError, TypeError):
            pass
        return False

    Parameters
    ----------
    answer : str
        Generated answer
    gold : str
        Gold (correct) answer
    dataset_type : str, optional
        Dataset type (for special handling such as MMLU-Pro)
    
    Returns
    -------
    bool
        True if the answer is correct, otherwise False
    """
    
    # For AIME/MATH datasets, use extraction/normalization from BoN_answeranalyze
    if dataset_type in ["aime2024", "aime2024short", "aime2025", "math500", "math"]:
        gold_str = str(gold).strip()
        # Extract prediction (e.g., \\boxed{...}). If not found, use the whole text
        extracted = extract_boxed_answer(answer)
        if extracted is None:
            extracted = str(answer).strip()
        
        # Normalization
        extracted_norm = normalize_math_notation(str(extracted)) if extracted is not None else ""
        gold_norm = normalize_math_notation(gold_str)
        
        # String equality
        if str(extracted).strip() == gold_str:
            return True
        # Equality after normalization
        if extracted_norm == gold_norm:
            return True
        # Numeric comparison (if possible)
        try:
            if abs(float(extracted_norm) - float(gold_norm)) < 1e-9:
                return True
        except (ValueError, TypeError):
            pass
        try:
            if abs(float(str(extracted).strip()) - float(gold_str)) < 1e-9:
                return True
        except (ValueError, TypeError):
            pass

    def normalize_letter_choice(text: str) -> str:
        """Normalize letter choices by removing parentheses"""
        text = text.strip()
        # (A), (B), ..., (J) → A, B, ..., J
        if len(text) == 3 and text.startswith('(') and text.endswith(')') and text[1].isalpha():
            return text[1]
        # Keep A, B, ..., J as-is
        elif len(text) == 1 and text.isalpha():
            return text
        return text
    
    def remove_degree_symbols(text: str) -> str:
        """Remove degree symbols (AIME2025 support)"""
        text = text.strip()
        # Remove ^\circ
        text = re.sub(r'\^\s*\\circ', '', text)
        # Also remove the ° symbol
        text = re.sub(r'°', '', text)
        return text.strip()
    
    def is_mmlu_choice_match(pred: str, gold: str) -> bool:
        """Choice matching for MMLU-Pro (allow parentheses)"""
        pred_norm = normalize_letter_choice(pred)
        gold_norm = normalize_letter_choice(gold)
        return pred_norm == gold_norm
    
    # Extract answer from boxed pattern
    boxed_pattern = r'\\boxed{([^}]+)}'
    boxed_match = re.search(boxed_pattern, answer)
    
    if boxed_match:
        extracted_answer = boxed_match.group(1).strip()
        
        # Exact match check
        if str(gold) == extracted_answer:
            return True
        
        # Exact match after removing degree symbols (AIME2025)
        extracted_cleaned = remove_degree_symbols(extracted_answer)
        gold_cleaned = remove_degree_symbols(str(gold))
        if gold_cleaned == extracted_cleaned:
            return True
        
        # For MMLU-Pro and GPQA-Diamond, allow parentheses for letter choices
        if dataset_type in ["mmlu_pro", "gpqa_diamond"]:
            if is_mmlu_choice_match(extracted_answer, str(gold)):
                return True
    
    # Partial match within the last 50 characters
    if len(str(gold)) >= 2 and str(gold) in answer[-50:]:
        return True
    
    # Partial match in the tail after removing degree symbols (AIME2025)
    gold_cleaned = remove_degree_symbols(str(gold))
    if len(gold_cleaned) >= 2 and gold_cleaned in remove_degree_symbols(answer[-50:]):
        return True
    
    # For MMLU-Pro and GPQA-Diamond, match letter choices at the tail (allow parentheses)
    if dataset_type in ["mmlu_pro", "gpqa_diamond"]:
        answer_tail = answer[-50:]
        # Extract single-letter choices from the tail, with/without parentheses
        letter_patterns = [r'\b([A-J])\b', r'\(([A-J])\)']
        for pattern in letter_patterns:
            matches = re.findall(pattern, answer_tail)
            if matches:
                last_match = matches[-1]  # Last match found
                if is_mmlu_choice_match(last_match, str(gold)):
                    return True
    
    return False

# Pandas for Parquet file support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

# Default directory settings
SAVED_ANSWERS_DIR = os.getenv("SAVED_ANSWERS_DIR", "saved_answers")
SAVED_PLOTS_DIR = os.getenv("SAVED_PLOTS_DIR", "saved_plots")

def detect_dataset_type(file_path, dataset_type='auto'):
    """Detect dataset type and return appropriate field mapping."""
    if dataset_type != 'auto':
        if dataset_type == 'gsm8k':
            return {
                'instruction_key': 'problem',
                'output_key': 'result', 
                'solution_key': 'solution_wocode'
            }
        elif dataset_type == 'aime2024':
            return {
                'instruction_key': 'Problem',
                'output_key': 'Answer',
                'solution_key': 'Solution'
            }
        elif dataset_type == 'aime2024short':
            return {
                'instruction_key': 'Problem',
                'output_key': 'Answer',
                'solution_key': 'Solution'
            }
        elif dataset_type == 'aime2025':
            return {
                'instruction_key': 'question',
                'output_key': 'answer',
                'solution_key': 'question'  # AIME2025 has no solution; use 'question' as 'solution'
            }
        elif dataset_type == 'math':
            return {
                'instruction_key': 'problem',
                'output_key': 'solution',  # MATH dataset uses 'solution' field for answers
                'solution_key': 'solution'
            }
        elif dataset_type == 'math500':
            return {
                'instruction_key': 'problem',
                'output_key': 'answer',  # MATH dataset uses 'answer' field for answers
                'solution_key': 'solution'
            }
        elif dataset_type == 'mmlu_pro':
            return {
                'instruction_key': 'question',
                'output_key': 'answer',
                'solution_key': 'cot_content'
            }
        elif dataset_type == 'gpqa_diamond':
            return {
                'instruction_key': 'question',
                'output_key': 'answer',
                'solution_key': 'question'  # GPQA-Diamond has no solution; use 'question' as 'solution'
            }
    
    # Auto-detect by file extension
    if file_path.endswith('.parquet'):
        # For Parquet files, sample the content and detect
        try:
            if HAS_PANDAS and pd is not None:
                df = pd.read_parquet(file_path)  # type: ignore
                columns = df.columns.tolist()
                
                # Detect MATH dataset
                if 'problem' in columns and 'solution' in columns and 'level' in columns and 'type' in columns:
                    return {
                        'instruction_key': 'problem',
                        'output_key': 'solution',
                        'solution_key': 'solution'
                    }
                # Detect AIME2024 dataset
                elif 'Problem' in columns and 'Answer' in columns:
                    return {
                        'instruction_key': 'Problem',
                        'output_key': 'Answer',
                        'solution_key': 'Solution'
                    }
                # Detect MMLU-Pro dataset
                elif 'question' in columns and 'options' in columns and 'answer' in columns and 'cot_content' in columns:
                    return {
                        'instruction_key': 'question',
                        'output_key': 'answer',
                        'solution_key': 'cot_content'
                    }
                # Detect GPQA-Diamond dataset
                elif 'question' in columns and 'answer' in columns and len(columns) == 2:
                    return {
                        'instruction_key': 'question',
                        'output_key': 'answer',
                        'solution_key': 'question'  # GPQA-Diamond has no solution; use 'question' as 'solution'
                    }
        except:
            pass
        
        # Default to AIME2024 format
        return {
            'instruction_key': 'Problem',
            'output_key': 'Answer', 
            'solution_key': 'Solution'
        }
    else:
        # For JSONL files, sample the first line to detect
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                sample = json.loads(first_line)
                
            if 'Problem' in sample and 'Answer' in sample:
                return {
                    'instruction_key': 'Problem',
                    'output_key': 'Answer',
                    'solution_key': 'Solution'
                }
            elif 'question' in sample and 'answer' in sample:
                return {
                    'instruction_key': 'question',
                    'output_key': 'answer',
                    'solution_key': 'question'  # AIME2025 has no solution; use 'question' as 'solution'
                }
            else:
                return {
                    'instruction_key': 'problem', 
                    'output_key': 'result',
                    'solution_key': 'solution_wocode'
                }
        except:
            # Default to GSM8K format
            return {
                'instruction_key': 'problem',
                'output_key': 'result', 
                'solution_key': 'solution_wocode'
            }

def load_data(file_path, instruction_key, output_key, solution_key):
    """Load JSONL or Parquet file and return in a unified format."""
    list_data_dict = []
    
    # Determine if the dataset is MMLU-Pro
    is_mmlu_pro = False
    if file_path.endswith('.parquet'):
        if not HAS_PANDAS:
            raise ImportError("Reading Parquet files requires pandas. Please run: pip install pandas")
        df = pd.read_parquet(file_path)  # type: ignore
        columns = df.columns.tolist()
        is_mmlu_pro = 'question' in columns and 'options' in columns and 'answer' in columns and 'cot_content' in columns
    
    if file_path.endswith('.parquet'):
        # For Parquet files
        for _, row in df.iterrows():
            if is_mmlu_pro:
                # For MMLU-Pro: combine question and options to generate instruction
                question = row.get('question', '')
                options = row.get('options', [])
                
                # Format options as strings with A-J labels
                option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                formatted_options = []
                for i, option in enumerate(options):
                    if i < len(option_labels):
                        formatted_options.append(f"({option_labels[i]}) {option}")
                
                # Combine question and options
                full_instruction = f"{question}\n\nChoose the correct answer from the following options:\n" + "\n".join(formatted_options)
                
                item = dict(
                    instruction=full_instruction,
                    output=row.get(output_key, None), 
                    solution_wocode=row.get(solution_key, None),
                    category=row.get('category', None),
                )
            else:
                # For regular datasets
                item = dict(
                    instruction=row.get(instruction_key, None),
                    output=row.get(output_key, None), 
                    solution_wocode=row.get(solution_key, None),
                    category=row.get('category', None),
                )
            list_data_dict.append(item)
    else:
        # For JSONL files
        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line)
                new_item = dict(
                    instruction=item.get(instruction_key, None),
                    output=item.get(output_key, None),
                    solution_wocode=item.get(solution_key, None), 
                    category=item.get('category', None),
                )
                list_data_dict.append(new_item)
    
    return list_data_dict

def load_choice_data(choice_file: str) -> List[Dict]:
    """
    Load a choice JSONL file.
    
    Parameters
    ----------
    choice_file : str
        Path to the choice JSONL file
        
    Returns
    -------
    List[Dict]
        List of choice records
    """
    choice_data = []
    print(f"📂 [CHOICE] Loading choice file: {choice_file}")
    
    try:
        with open(choice_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        choice_data.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ [CHOICE] Line {line_num}: JSON decode error: {str(e)}")
                        continue
        
        print(f"✅ [CHOICE] Loaded choice records: {len(choice_data)}")
        return choice_data
        
    except FileNotFoundError:
        print(f"❌ [CHOICE] File not found: {choice_file}")
        return []
    except Exception as e:
        print(f"❌ [CHOICE] File read error: {str(e)}")
        return []

def load_answer_from_file(answer_file: str) -> str:
    """
    Load content after <think> from an answer file.
    
    Parameters
    ----------
    answer_file : str
        Path to the answer file
        
    Returns
    -------
    str
        Content after <think>; empty string on failure
    """
    # Build absolute path including SAVED_ANSWERS_DIR
    if not answer_file.startswith(SAVED_ANSWERS_DIR):
        full_path = os.path.join(SAVED_ANSWERS_DIR, answer_file)
    else:
        full_path = answer_file
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the portion after <think>
        think_pos = content.find('<think>')
        if think_pos != -1:
            answer_content = content[think_pos + len('<think>'):].strip()
            print(f"✅ [ANSWER] Loaded answer: {os.path.basename(full_path)} ({len(answer_content)} chars)")
            return answer_content
        else:
            # If <think> tag is missing, return the entire file
            print(f"⚠️ [ANSWER] <think> tag not found. Using entire file: {os.path.basename(full_path)}")
            return content.strip()
            
    except FileNotFoundError:
        print(f"❌ [ANSWER] File not found: {full_path}")
        return ""
    except Exception as e:
        print(f"❌ [ANSWER] File read error: {str(e)}")
        return ""

def extract_dataset_type_from_filename(choice_file: str) -> str:
    """
    Extract dataset type from choice filename.
    Example: aime2024_run0_pick_answer_num5.jsonl → aime2024
    
    Parameters
    ----------
    choice_file : str
        Path to the choice file
        
    Returns
    -------
    str
        Dataset type
    """
    filename = os.path.basename(choice_file)
    # The first part of the filename is the dataset type
    parts = filename.split('_')
    if parts:
        return parts[0]
    return "auto"

def find_dataset_files(dataset_name: str, num: int = 5) -> Dict[str, List[str]]:
    """
    Find files for the specified dataset name and group them by (method, scale).
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., aime2024)
    num : int
        num value (default: 5)
        
    Returns
    -------
    Dict[str, List[str]]
        Dict keyed by "{method}_scale{value}", values are lists of file paths per run
    """
    patterns = {
        'omni': f'saved_choices/{dataset_name}_run*_omni_num{num}.jsonl',
        'reward_INF-ORM-Llama3.1-70B': f'saved_choices/{dataset_name}_run*_reward_INF-ORM-Llama3.1-70B_num{num}.jsonl',
        'reward_orm_INF-ORM-Llama3.1-70B': f'saved_choices/{dataset_name}_run*_reward_orm_INF-ORM-Llama3.1-70B_num{num}.jsonl',
        'reward_ArmoRM-Llama3-8B-v0.1': f'saved_choices/{dataset_name}_run*_reward_ArmoRM-Llama3-8B-v0.1_num{num}.jsonl',
        'reward_orm_ArmoRM-Llama3-8B-v0.1': f'saved_choices/{dataset_name}_run*_reward_orm_ArmoRM-Llama3-8B-v0.1_num{num}.jsonl',
        'reward_Skywork-Reward-V2-Qwen3-8B': f'saved_choices/{dataset_name}_run*_reward_Skywork-Reward-V2-Qwen3-8B_num{num}.jsonl',
        'reward_orm_Skywork-Reward-V2-Qwen3-8B': f'saved_choices/{dataset_name}_run*_reward_orm_Skywork-Reward-V2-Qwen3-8B_num{num}.jsonl',
        'reward_Skywork-Reward-V2-Llama-3.1-8B': f'saved_choices/{dataset_name}_run*_reward_Skywork-Reward-V2-Llama-3.1-8B_num{num}.jsonl',
        'reward_orm_Skywork-Reward-V2-Llama-3.1-8B': f'saved_choices/{dataset_name}_run*_reward_orm_Skywork-Reward-V2-Llama-3.1-8B_num{num}.jsonl',
        'pick_answer': f'saved_choices/{dataset_name}_run*_pick_answer_num{num}.jsonl',
        'llm_judge_set': f'saved_choices/{dataset_name}_run*_llm_judge_set_num{num}.jsonl',
        'llm_judge_tournament': f'saved_choices/{dataset_name}_run*_llm_judge_tournament_num{num}.jsonl',
        'self_certainty': f'saved_choices/{dataset_name}_run*_self_certainty_num{num}.jsonl',
        'majority': f'saved_choices/{dataset_name}_run*_majority_num{num}.jsonl',
        'random': f'saved_choices/{dataset_name}_run*_random_num{num}.jsonl',
    }
    
    results = {}
    for base_method, pattern in patterns.items():
        files = sorted(glob.glob(pattern))
        if files:
            # For each file, read scale info and create (method, scale) pairs
            for file in files:
                scales_in_file = get_scales_from_file(file)
                for scale in scales_in_file:
                    method_scale_key = f"{base_method}_scale{scale}"
                    if method_scale_key not in results:
                        results[method_scale_key] = []
                    results[method_scale_key].append(file)
    
    return results


def find_dataset_files_all_nums(dataset_name: str) -> Dict[str, List[str]]:
    """
    Find all num files for the dataset and group by (method, scale)

    Returns
    -------
    Dict[str, List[str]]
        Dict keyed by "{method}_scale{value}" with file lists across runs and nums
    """
    patterns = {
        'omni': f'saved_choices/{dataset_name}_run*_omni_num*.jsonl',
        'reward_INF-ORM-Llama3.1-70B': f'saved_choices/{dataset_name}_run*_reward_INF-ORM-Llama3.1-70B_num*.jsonl',
        'reward_orm_INF-ORM-Llama3.1-70B': f'saved_choices/{dataset_name}_run*_reward_orm_INF-ORM-Llama3.1-70B_num*.jsonl',
        'reward_ArmoRM-Llama3-8B-v0.1': f'saved_choices/{dataset_name}_run*_reward_ArmoRM-Llama3-8B-v0.1_num*.jsonl',
        'reward_orm_ArmoRM-Llama3-8B-v0.1': f'saved_choices/{dataset_name}_run*_reward_orm_ArmoRM-Llama3-8B-v0.1_num*.jsonl',
        'reward_Skywork-Reward-V2-Qwen3-8B': f'saved_choices/{dataset_name}_run*_reward_Skywork-Reward-V2-Qwen3-8B_num*.jsonl',
        'reward_orm_Skywork-Reward-V2-Qwen3-8B': f'saved_choices/{dataset_name}_run*_reward_orm_Skywork-Reward-V2-Qwen3-8B_num*.jsonl',
        'pick_answer': f'saved_choices/{dataset_name}_run*_pick_answer_num*.jsonl',
        'llm_judge_set': f'saved_choices/{dataset_name}_run*_llm_judge_set_num*.jsonl',
        'llm_judge_tournament': f'saved_choices/{dataset_name}_run*_llm_judge_tournament_num*.jsonl',
        'self_certainty': f'saved_choices/{dataset_name}_run*_self_certainty_num*.jsonl',
        'majority': f'saved_choices/{dataset_name}_run*_majority_num*.jsonl',
        'random': f'saved_choices/{dataset_name}_run*_random_num*.jsonl',
    }

    results: Dict[str, List[str]] = {}
    for base_method, pattern in patterns.items():
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        for file in files:
            scales_in_file = get_scales_from_file(file)
            for scale in scales_in_file:
                method_scale_key = f"{base_method}_scale{scale}"
                if method_scale_key not in results:
                    results[method_scale_key] = []
                results[method_scale_key].append(file)

    return results

def get_scales_from_file(filepath: str) -> List[int]:
    """
    Get the list of scale values used in the file.
    
    Parameters
    ----------
    filepath : str
        Path to the JSONL file
        
    Returns
    -------
    List[int]
        Sorted list of scales used in the file (deduplicated)
    """
    scales = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        scale = record.get('scale', 1)
                        scales.add(scale)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"⚠️ [WARNING] Failed to read scales {filepath}: {str(e)}")
        return [1]  # Return scale 1 by default
    
    return sorted(list(scales))

def find_dataset_files_by_method(dataset_name: str, evaluation_method: str) -> Dict[str, List[str]]:
    """
    Find multiple num files for a dataset and evaluation method, grouped by (num, scale)
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., aime2024)
    evaluation_method : str
        Evaluation method name (e.g., pick_answer)
        
    Returns
    -------
    Dict[str, List[str]]
        Dict keyed by "num{value}_scale{value}" with lists of file paths per run
    """
    # Search all num files for the specified method
    pattern = f'saved_choices/{dataset_name}_run*_{evaluation_method}_num*.jsonl'
    files = sorted(glob.glob(pattern))
    
    results = {}
    for file in files:
        # Extract num value from filename
        num_match = re.search(r'_num(\d+)\.jsonl$', file)
        if num_match:
            num_value = int(num_match.group(1))
            
            # Read scale info from the file
            scales_in_file = get_scales_from_file(file)
            for scale in scales_in_file:
                key = f"num{num_value}_scale{scale}"
                if key not in results:
                    results[key] = []
                results[key].append(file)
    
    return results

def extract_run_number(filepath: str) -> int:
    """
    Extract run number from filepath.
    
    Parameters
    ----------
    filepath : str
        File path
        
    Returns
    -------
    int
        Run number, or -1 if not found
    """
    match = re.search(r'_run(\d+)_', filepath)
    if match:
        return int(match.group(1))
    return -1

def calculate_simple_accuracy(choice_file: str, test_file: str, target_scale: int, dataset_type: str = 'auto') -> float:
    """
    Compute simple accuracy for the specified scale in a choice file.
    
    Parameters
    ----------
    choice_file : str
        Path to the choice JSONL file
    test_file : str
        Path to the test dataset file
    target_scale : int
        Target scale value
    dataset_type : str
        Dataset type
        
    Returns
    -------
    float
        Accuracy (%) for the specified scale
    """
    try:
        # Load choices
        choice_data = []
        with open(choice_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        choice_data.append(record)
                    except json.JSONDecodeError:
                        continue
        
        if not choice_data:
            return 0.0
        
        # Detect dataset type
        if dataset_type == 'auto':
            dataset_type = extract_dataset_type_from_filename(choice_file)
        
        # Load problem file
        field_mapping = detect_dataset_type(test_file, dataset_type)
        instruction_key = field_mapping['instruction_key']
        output_key = field_mapping['output_key']
        solution_key = field_mapping['solution_key']
        
        test_data = load_data(test_file, instruction_key, output_key, solution_key)
        
        # Build a dict of gold answers keyed by problem index
        gold_answers = {}
        for i, data in enumerate(test_data):
            gold_answers[i] = data['output']
        
        # Collect statistics for the target scale
        target_stats = {'total': 0, 'correct': 0}
        
        # Keep only the last record for each (problem_number, scale)
        unique_choices = {}
        for choice in choice_data:
            problem_number = choice.get('problem_number', -1)
            scale = choice.get('scale', 1)
            # Process only the target scale
            if scale != target_scale:
                continue
            combination = (problem_number, scale)
            unique_choices[combination] = choice
        
        # Evaluate each unique choice
        for combination, choice in unique_choices.items():
            problem_number = choice.get('problem_number', -1)
            selected_file = choice.get('selected_file', '')
            scale = choice.get('scale', 1)
            
            # Get gold answer
            if problem_number not in gold_answers:
                continue
                
            gold = gold_answers[problem_number]
            
            # Load the selected answer
            if selected_file:
                answer = load_answer_from_file(selected_file)
                if answer:
                    # Correctness check
                    is_answer_correct = is_correct(answer, gold, dataset_type)
                    
                    # Update statistics
                    target_stats['total'] += 1
                    if is_answer_correct:
                        target_stats['correct'] += 1
        
        # Compute accuracy for the specified scale
        if target_stats['total'] > 0:
            accuracy = (target_stats['correct'] / target_stats['total']) * 100
            return accuracy
        else:
            return 0.0
        
    except Exception as e:
        print(f"❌ [ERROR] Analysis error for {choice_file}: {str(e)}")
        return 0.0

def calculate_simple_accuracy_by_method(choice_file: str, test_file: str, target_num: int, target_scale: int, dataset_type: str = 'auto') -> float:
    """
    Compute simple accuracy for the specified num and scale in a choice file.
    
    Parameters
    ----------
    choice_file : str
        Path to the choice JSONL file
    test_file : str
        Path to the test dataset file
    target_num : int
        Target num value (extracted from filename)
    target_scale : int
        Target scale value
    dataset_type : str
        Dataset type
        
    Returns
    -------
    float
        Accuracy (%) for the specified num and scale
    """
    try:
        # Verify num value from filename
        num_match = re.search(r'_num(\d+)\.jsonl$', choice_file)
        if not num_match or int(num_match.group(1)) != target_num:
            return 0.0
        
        # Load choices
        choice_data = []
        with open(choice_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        choice_data.append(record)
                    except json.JSONDecodeError:
                        continue
        
        if not choice_data:
            return 0.0
        
        # Detect dataset type
        if dataset_type == 'auto':
            dataset_type = extract_dataset_type_from_filename(choice_file)
        
        # Load problem file
        field_mapping = detect_dataset_type(test_file, dataset_type)
        instruction_key = field_mapping['instruction_key']
        output_key = field_mapping['output_key']
        solution_key = field_mapping['solution_key']
        
        test_data = load_data(test_file, instruction_key, output_key, solution_key)
        
        # Build a dict of gold answers keyed by problem index
        gold_answers = {}
        for i, data in enumerate(test_data):
            gold_answers[i] = data['output']
        
        # Collect statistics for the target scale
        target_stats = {'total': 0, 'correct': 0}
        
        # Keep only the last record for each (problem_number, scale)
        unique_choices = {}
        for choice in choice_data:
            problem_number = choice.get('problem_number', -1)
            scale = choice.get('scale', 1)
            # Process only the target scale
            if scale != target_scale:
                continue
            combination = (problem_number, scale)
            unique_choices[combination] = choice
        
        # Evaluate each unique choice
        for combination, choice in unique_choices.items():
            problem_number = choice.get('problem_number', -1)
            selected_file = choice.get('selected_file', '')
            scale = choice.get('scale', 1)
            
            # Get gold answer
            if problem_number not in gold_answers:
                continue
                
            gold = gold_answers[problem_number]
            
            # Load the selected answer
            if selected_file:
                answer = load_answer_from_file(selected_file)
                if answer:
                    # Correctness check
                    is_answer_correct = is_correct(answer, gold, dataset_type)
                    
                    # Update statistics
                    target_stats['total'] += 1
                    if is_answer_correct:
                        target_stats['correct'] += 1
        
        # Compute accuracy for the specified scale
        if target_stats['total'] > 0:
            accuracy = (target_stats['correct'] / target_stats['total']) * 100
            return accuracy
        else:
            return 0.0
        
    except Exception as e:
        print(f"❌ [ERROR] Analysis error for {choice_file}: {str(e)}")
        return 0.0

def analyze_choices(choice_file: str, test_file: str) -> None:
    """
    Analyze choices and print accuracy per scale.
    
    Parameters
    ----------
    choice_file : str
        Path to the choice JSONL file
    test_file : str
        Path to the test dataset file
    """
    print(f"\n🚀 [ANALYZE] BoN Choice Analysis start")
    print(f"🚀 [ANALYZE] Choice file: {choice_file}")
    print(f"🚀 [ANALYZE] Problem file: {test_file}")
    print(f"🚀 [ANALYZE] Answers directory: {SAVED_ANSWERS_DIR}")
    print("="*60)
    
    # Load choices
    choice_data = load_choice_data(choice_file)
    if not choice_data:
        print("❌ [ANALYZE] Failed to load choice data")
        return
    
    # Detect dataset type
    dataset_type = extract_dataset_type_from_filename(choice_file)
    print(f"🔍 [ANALYZE] Detected dataset type: {dataset_type}")
    
    # Load problem file
    field_mapping = detect_dataset_type(test_file, dataset_type)
    instruction_key = field_mapping['instruction_key']
    output_key = field_mapping['output_key']
    solution_key = field_mapping['solution_key']
    
    print(f"🔍 [ANALYZE] Field mapping: instruction='{instruction_key}', output='{output_key}', solution='{solution_key}'")
    
    try:
        test_data = load_data(test_file, instruction_key, output_key, solution_key)
        print(f"✅ [ANALYZE] Loaded problems: {len(test_data)}")
    except Exception as e:
        print(f"❌ [ANALYZE] Problem data read error: {str(e)}")
        return
    
    # Build dict of gold answers keyed by 0-based index
    gold_answers = {}
    for i, data in enumerate(test_data):
        gold_answers[i] = data['output']  # Use 0-based index
    
    print(f"📋 [ANALYZE] Number of gold answers: {len(gold_answers)}")
    
    # Collect per-scale statistics
    scale_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'problems': [], 'details': [], 'trivial_stats': {'trivial_correct': 0, 'trivial_incorrect': 0, 'non_trivial': 0}})
    
    # Store triviality analysis per problem
    problem_triviality = {}
    
    # Keep only the last record for the same (problem_number, scale)
    unique_choices = {}
    
    print(f"\n🔄 [ANALYZE] Removing duplicates...")
    for i, choice in enumerate(choice_data):
        problem_number = choice.get('problem_number', -1)
        scale = choice.get('scale', 1)
        combination = (problem_number, scale)
        
        # Overwrite if same combination appears (keep the last)
        if combination in unique_choices:
            print(f"⚠️ [ANALYZE] Overwriting duplicate: problem {problem_number}, scale {scale} (choice {i+1}/{len(choice_data)})")
        
        unique_choices[combination] = choice
    
    print(f"✅ [ANALYZE] Dedup done: {len(choice_data)} → {len(unique_choices)}")
    
    print(f"\n🔄 [ANALYZE] Evaluating each choice...")
    for combination, choice in unique_choices.items():
        problem_number = choice.get('problem_number', -1)
        selected_file = choice.get('selected_file', '')
        loaded_files = choice.get('loaded_files', [])
        scale = choice.get('scale', 1)
        
        print(f"\n--- 📊 [ANALYZE] Problem: {problem_number}, Scale: {scale} ---")
        print(f"📊 [ANALYZE] Selected file: {selected_file}")
        print(f"📊 [ANALYZE] #Candidates: {len(loaded_files)}")
        
        # Get gold answer
        if problem_number in gold_answers:
            gold = gold_answers[problem_number]
            print(f"📊 [ANALYZE] Gold answer: {gold}")
        else:
            print(f"⚠️ [ANALYZE] Gold answer not found for problem {problem_number}")
            continue
        
        # Triviality analysis: evaluate all loaded_files (once per problem)
        if problem_number not in problem_triviality and loaded_files:
            print(f"🔍 [TRIVIAL] Triviality analysis for problem {problem_number} (candidates: {len(loaded_files)})")
            
            all_results = []
            for file_idx, loaded_file in enumerate(loaded_files):
                if loaded_file:
                    answer = load_answer_from_file(loaded_file)
                    if answer:
                        is_answer_correct = is_correct(answer, gold, dataset_type)
                        all_results.append(is_answer_correct)
                        print(f"  🔍 [TRIVIAL] File {file_idx+1}/{len(loaded_files)}: {os.path.basename(loaded_file)} → {'✅Correct' if is_answer_correct else '❌Incorrect'}")
                    else:
                        print(f"  ⚠️ [TRIVIAL] File {file_idx+1}/{len(loaded_files)}: {os.path.basename(loaded_file)} → Load failed")
                        all_results.append(False)  # Treat load failures as incorrect
                else:
                    print(f"  ⚠️ [TRIVIAL] File {file_idx+1}/{len(loaded_files)}: Empty filename")
                    all_results.append(False)
            
            # Decide triviality
            if all_results:
                if all(all_results):
                    triviality = "trivial_correct"
                    print(f"🎯 [TRIVIAL] Problem {problem_number}: Trivially correct (all correct)")
                elif not any(all_results):
                    triviality = "trivial_incorrect"
                    print(f"🎯 [TRIVIAL] Problem {problem_number}: Trivially incorrect (all incorrect)")
                else:
                    triviality = "non_trivial"
                    correct_count = sum(all_results)
                    total_count = len(all_results)
                    print(f"🎯 [TRIVIAL] Problem {problem_number}: Non-trivial ({correct_count}/{total_count} correct)")
                
                problem_triviality[problem_number] = {
                    'type': triviality,
                    'all_results': all_results,
                    'correct_count': sum(all_results),
                    'total_count': len(all_results)
                }
            else:
                print(f"❌ [TRIVIAL] Problem {problem_number}: No files to evaluate")
                problem_triviality[problem_number] = {
                    'type': 'unknown',
                    'all_results': [],
                    'correct_count': 0,
                    'total_count': 0
                }
        
        # Load the selected answer
        if selected_file:
            answer = load_answer_from_file(selected_file)
            if answer:
                # Correctness check
                is_answer_correct = is_correct(answer, gold, dataset_type)
                
                # Update statistics
                scale_stats[scale]['total'] += 1
                scale_stats[scale]['problems'].append(problem_number)
                
                # Update triviality stats
                if problem_number in problem_triviality:
                    triviality_type = problem_triviality[problem_number]['type']
                    if triviality_type in scale_stats[scale]['trivial_stats']:
                        scale_stats[scale]['trivial_stats'][triviality_type] += 1
                
                # Save detail info
                detail_info = {
                    'problem_number': problem_number,
                    'selected_file': selected_file,
                    'is_correct': is_answer_correct,
                    'gold_answer': gold,
                    'predicted_answer': answer[:100] + "..." if len(answer) > 100 else answer,  # First 100 chars of prediction
                    'triviality': problem_triviality.get(problem_number, {}).get('type', 'unknown')
                }
                scale_stats[scale]['details'].append(detail_info)
                
                if is_answer_correct:
                    scale_stats[scale]['correct'] += 1
                    print(f"✅ [ANALYZE] Correct")
                else:
                    print(f"❌ [ANALYZE] Incorrect")
            else:
                print(f"❌ [ANALYZE] Failed to load answer file")
        else:
            print(f"❌ [ANALYZE] Selected file not specified")
    
    # Print results
    print(f"\n" + "="*60)
    print(f"📊 [RESULTS] BoN Choice Analysis Results")
    print(f"="*60)
    print(f"📂 [RESULTS] Choice file: {os.path.basename(choice_file)}")
    print(f"📂 [RESULTS] Dataset type: {dataset_type}")
    print(f"📂 [RESULTS] Answers directory: {SAVED_ANSWERS_DIR}")
    print()
    
    # Print results per scale (sorted by scale ascending)
    for scale in sorted(scale_stats.keys()):
        stats = scale_stats[scale]
        total = stats['total']
        correct = stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        trivial_correct = stats['trivial_stats']['trivial_correct']
        trivial_incorrect = stats['trivial_stats']['trivial_incorrect']
        non_trivial = stats['trivial_stats']['non_trivial']
        
        print(f"📊 [RESULTS] Scale {scale:2d}: {correct:3d}/{total:3d} = {accuracy:5.1f}% "
              f"(Trivially correct: {trivial_correct}, Trivially incorrect: {trivial_incorrect}, Non-trivial: {non_trivial})")
    
    print()
    
    # Overall statistics
    total_all = sum(stats['total'] for stats in scale_stats.values())
    correct_all = sum(stats['correct'] for stats in scale_stats.values())
    accuracy_all = (correct_all / total_all * 100) if total_all > 0 else 0.0
    
    # Overall triviality statistics
    total_trivial_correct = sum(stats['trivial_stats']['trivial_correct'] for stats in scale_stats.values())
    total_trivial_incorrect = sum(stats['trivial_stats']['trivial_incorrect'] for stats in scale_stats.values())
    total_non_trivial = sum(stats['trivial_stats']['non_trivial'] for stats in scale_stats.values())
    
    print(f"📊 [RESULTS] Overall: {correct_all:3d}/{total_all:3d} = {accuracy_all:5.1f}%")
    print(f"📊 [RESULTS] #Scale types: {len(scale_stats)}")
    print(f"📊 [RESULTS] Triviality: Trivially correct={total_trivial_correct}, Trivially incorrect={total_trivial_incorrect}, Non-trivial={total_non_trivial}")
    
    # Triviality analysis summary
    total_analyzed = len(problem_triviality)
    if total_analyzed > 0:
        trivial_correct_problems = sum(1 for p in problem_triviality.values() if p['type'] == 'trivial_correct')
        trivial_incorrect_problems = sum(1 for p in problem_triviality.values() if p['type'] == 'trivial_incorrect')
        non_trivial_problems = sum(1 for p in problem_triviality.values() if p['type'] == 'non_trivial')
        
        print(f"\n📋 [TRIVIALITY] Summary:")
        print(f"📋 [TRIVIALITY] #Problems analyzed: {total_analyzed}")
        print(f"📋 [TRIVIALITY] Trivially correct problems: {trivial_correct_problems} ({trivial_correct_problems/total_analyzed*100:.1f}%)")
        print(f"📋 [TRIVIALITY] Trivially incorrect problems: {trivial_incorrect_problems} ({trivial_incorrect_problems/total_analyzed*100:.1f}%)")
        print(f"📋 [TRIVIALITY] Non-trivial problems: {non_trivial_problems} ({non_trivial_problems/total_analyzed*100:.1f}%)")
        
        # Detailed triviality by type
        print(f"\n📋 [TRIVIALITY] Details:")
        for problem_num, triviality_info in sorted(problem_triviality.items()):
            triviality_type = triviality_info['type']
            correct_count = triviality_info['correct_count']
            total_count = triviality_info['total_count']
            
            if triviality_type == 'trivial_correct':
                icon = "✅"
                description = "Trivially correct"
            elif triviality_type == 'trivial_incorrect':
                icon = "❌"
                description = "Trivially incorrect"
            elif triviality_type == 'non_trivial':
                icon = "🔄"
                description = f"Non-trivial ({correct_count}/{total_count} correct)"
            else:
                icon = "❓"
                description = "Unknown"
            
            print(f"{icon} Problem {problem_num:3d}: {description}")
    
    # Detailed analysis for the best scale by accuracy
    if scale_stats:
        # Sort scales by accuracy (tie-break by total)
        best_scale = max(scale_stats.keys(), 
                        key=lambda s: (scale_stats[s]['correct'] / scale_stats[s]['total'] if scale_stats[s]['total'] > 0 else 0, 
                                     scale_stats[s]['total']))
        
        best_stats = scale_stats[best_scale]
        best_accuracy = (best_stats['correct'] / best_stats['total'] * 100) if best_stats['total'] > 0 else 0.0
        
        print(f"\n📋 [RESULTS] Best scale details (Scale {best_scale}: {best_accuracy:.1f}%):")
        print(f"="*60)
        
        # Sort by problem number
        sorted_details = sorted(best_stats['details'], key=lambda x: x['problem_number'])
        
        for detail in sorted_details:
            status_icon = "✅" if detail['is_correct'] else "❌"
            triviality = detail['triviality']
            if triviality == 'trivial_correct':
                triviality_icon = "🎯✅"
            elif triviality == 'trivial_incorrect':
                triviality_icon = "🎯❌"
            elif triviality == 'non_trivial':
                triviality_icon = "🔄"
            else:
                triviality_icon = "❓"
            
            print(f"{status_icon} Problem {detail['problem_number']:3d}: "
                  f"{'Correct' if detail['is_correct'] else 'Incorrect'} "
                  f"{triviality_icon} "
                  f"(Gold: {detail['gold_answer']}) "
                  f"[{os.path.basename(detail['selected_file'])}]")
        
        # Summary
        correct_count = sum(1 for d in sorted_details if d['is_correct'])
        total_count = len(sorted_details)
        print(f"\n📊 Scale {best_scale} summary: {correct_count}/{total_count} = {best_accuracy:.1f}%")
    
    print(f"\n" + "="*60)
    print(f"📊 [RESULTS] BoN Choice Analysis done")
    print(f"="*60)

def analyze_dataset(dataset_name: str, test_file: str, num: int = 5) -> None:
    """
    Analyze the entire dataset and print a table of results.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., aime2024)
    test_file : str
        Path to the test dataset file
    num : int
        num value (default: 5)
    """
    print(f"\n🚀 [DATASET_ANALYZE] Dataset analysis start")
    print(f"🚀 [DATASET_ANALYZE] Dataset: {dataset_name}")
    print(f"🚀 [DATASET_ANALYZE] Problem file: {test_file}")
    print(f"🚀 [DATASET_ANALYZE] num: {num}")
    print("="*60)
    
    # Search files
    files_by_method = find_dataset_files(dataset_name, num)
    if not files_by_method:
        print(f"❌ [DATASET_ANALYZE] Files not found for {dataset_name}")
        return
    
    print(f"✅ [DATASET_ANALYZE] #Methods found: {len(files_by_method)}")
    for method, files in files_by_method.items():
        print(f"  📊 {method}: {len(files)} files")
    
    # Collect run numbers
    all_runs = set()
    for files in files_by_method.values():
        for file in files:
            run_num = extract_run_number(file)
            if run_num >= 0:
                all_runs.add(run_num)
    
    all_runs = sorted(list(all_runs))
    print(f"✅ [DATASET_ANALYZE] Detected run ids: {all_runs}")
    
    # Compute accuracy for each method and run
    results = {}  # method_scale -> {run: accuracy}
    
    print(f"\n🔄 [DATASET_ANALYZE] Computing accuracies...")
    for method_scale, files in files_by_method.items():
        results[method_scale] = {}
        print(f"\n📊 [DATASET_ANALYZE] Method: {method_scale}")
        
        # Extract scale value from method_scale
        scale_match = re.search(r'_scale(\d+)$', method_scale)
        if scale_match:
            target_scale = int(scale_match.group(1))
        else:
            target_scale = 1  # Default scale
        
        for file in files:
            run_num = extract_run_number(file)
            if run_num >= 0:
                print(f"  📂 Analyzing {os.path.basename(file)} (run{run_num}, scale{target_scale})...")
                accuracy = calculate_simple_accuracy(file, test_file, target_scale, dataset_name)
                results[method_scale][run_num] = accuracy
                print(f"    ✅ Accuracy at scale{target_scale}: {accuracy:.1f}%")
    
    # Print table
    print(f"\n" + "="*60)
    print(f"📊 [RESULTS] Dataset analysis results")
    print(f"="*60)
    print(f"📂 [RESULTS] Dataset: {dataset_name}")
    print(f"📂 [RESULTS] num: {num}")
    print()
    
    # Header row
    header = "method_scale," + ",".join(str(run) for run in all_runs) + ",mean $\\pm$ error"
    print(header)
    
    # Rows per method_scale
    for method_scale in sorted(results.keys()):
        row_data = [method_scale]
        # Values per run
        for run in all_runs:
            if run in results[method_scale]:
                row_data.append(f"{results[method_scale][run]:.1f}")
            else:
                row_data.append("N/A")

        # Compute mean ± 2 * std / sqrt(n) for valid values only
        valid_values = [results[method_scale][run] for run in all_runs if run in results[method_scale]]
        n = len(valid_values)
        if n > 0:
            mean_val = sum(valid_values) / n
            if n > 1:
                # Sample standard deviation
                variance = sum((x - mean_val) ** 2 for x in valid_values) / (n - 1)
                std_val = math.sqrt(variance)
            else:
                std_val = 0.0
            margin = 2 * std_val / math.sqrt(n) if n > 0 else 0.0
            row_data.append(f"{mean_val:.2f} $\\pm$ {margin:.2f}")
        else:
            row_data.append("N/A")

        row = ",".join(row_data)
        print(row)
    
    print(f"\n" + "="*60)
    print(f"📊 [RESULTS] Dataset analysis done")
    print(f"="*60)

def analyze_dataset_by_method(dataset_name: str, test_file: str, evaluation_method: str) -> None:
    """
    Analyze dataset using a specific evaluation method and print table per (num, scale).
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., aime2024)
    test_file : str
        Path to the test dataset file
    evaluation_method : str
        Evaluation method name (e.g., pick_answer)
    """
    print(f"\n🚀 [METHOD_ANALYZE] Method-specific dataset analysis start")
    print(f"🚀 [METHOD_ANALYZE] Dataset: {dataset_name}")
    print(f"🚀 [METHOD_ANALYZE] Method: {evaluation_method}")
    print(f"🚀 [METHOD_ANALYZE] Problem file: {test_file}")
    print("="*60)
    
    # Search files for the specified method
    files_by_num_scale = find_dataset_files_by_method(dataset_name, evaluation_method)
    if not files_by_num_scale:
        print(f"❌ [METHOD_ANALYZE] Files not found for {dataset_name} and method {evaluation_method}")
        return
    
    print(f"✅ [METHOD_ANALYZE] #num-scale pairs found: {len(files_by_num_scale)}")
    for num_scale, files in files_by_num_scale.items():
        print(f"  📊 {num_scale}: {len(files)} files")
    
    # Collect run numbers
    all_runs = set()
    for files in files_by_num_scale.values():
        for file in files:
            run_num = extract_run_number(file)
            if run_num >= 0:
                all_runs.add(run_num)
    
    all_runs = sorted(list(all_runs))
    print(f"✅ [METHOD_ANALYZE] Detected run ids: {all_runs}")
    
    # Compute accuracy per (num, scale) and run
    results = {}  # num_scale -> {run: accuracy}
    
    print(f"\n🔄 [METHOD_ANALYZE] Computing accuracies...")
    for num_scale, files in files_by_num_scale.items():
        results[num_scale] = {}
        print(f"\n📊 [METHOD_ANALYZE] num-scale pair: {num_scale}")
        
        # Extract num and scale from key
        num_match = re.search(r'num(\d+)_scale(\d+)', num_scale)
        if num_match:
            target_num = int(num_match.group(1))
            target_scale = int(num_match.group(2))
        else:
            print(f"⚠️ [METHOD_ANALYZE] Failed to parse num-scale key: {num_scale}")
            continue
        
        for file in files:
            run_num = extract_run_number(file)
            if run_num >= 0:
                print(f"  📂 Analyzing {os.path.basename(file)} (run{run_num}, num{target_num}, scale{target_scale})...")
                accuracy = calculate_simple_accuracy_by_method(file, test_file, target_num, target_scale, dataset_name)
                results[num_scale][run_num] = accuracy
                print(f"    ✅ Accuracy at num{target_num}_scale{target_scale}: {accuracy:.1f}%")
    
    # Print table
    print(f"\n" + "="*60)
    print(f"📊 [RESULTS] Method-specific dataset analysis results")
    print(f"="*60)
    print(f"📂 [RESULTS] Dataset: {dataset_name}")
    print(f"📂 [RESULTS] Method: {evaluation_method}")
    print()
    
    # Header row
    header = "num_scale," + ",".join(str(run) for run in all_runs)
    print(header)
    
    # Rows per num_scale
    for num_scale in sorted(results.keys()):
        row_data = [num_scale]
        for run in all_runs:
            if run in results[num_scale]:
                row_data.append(f"{results[num_scale][run]:.1f}")
            else:
                row_data.append("N/A")
        row = ",".join(row_data)
        print(row)
    
    print(f"\n" + "="*60)
    print(f"📊 [RESULTS] Method-specific dataset analysis done")
    print(f"="*60)


def compute_average_token_count(choice_file: str, target_scale: int) -> float:
    """
    Return the average of "total_token_count" for rows with the given target_scale.
    If no values exist, return 0.0 (can be skipped during plotting).
    """
    try:
        # Keep only the last record for each (problem_number, scale)
        unique_records = {}
        with open(choice_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scale = rec.get('scale', 1)
                if scale != target_scale:
                    continue
                problem_number = rec.get('problem_number', -1)
                unique_records[(problem_number, scale)] = rec

        token_values = []
        for ((_, _)), rec in unique_records.items():
            if 'total_token_count' in rec and isinstance(rec['total_token_count'], (int, float)):
                token_values.append(float(rec['total_token_count']))

        if token_values:
            return sum(token_values) / len(token_values)
        return 0.0

    except Exception as e:
        print(f"⚠️ [WARNING] Token count computation error {choice_file}: {str(e)}")
        return 0.0


def plot_dataset_scatter(dataset_name: str, test_file: str, num: int = 5) -> None:
    """
    Save a scatter plot for dataset-wide analysis: average accuracy (y) vs average token count (x, log scale).
    Split results by n and include n in the legend.
    Output: saved_plots/accuracy_vs_tokens_{dataset_name}.png
    """
    try:
        # Collect choice files
        # Explore all nums for the dataset, not limited to a specific num
        files_by_method = find_dataset_files_all_nums(dataset_name)
        if not files_by_method:
            print(f"❌ [PLOT] Files not found for {dataset_name}. Skipping plot.")
            return

        # Group data by n: {n: [(x_tokens, y_accuracy, method_name), ...]}
        data_by_num = {}

        for method_scale, files in files_by_method.items():
            # Extract scale from method_scale
            scale_match = re.search(r'_scale(\d+)$', method_scale)
            target_scale = int(scale_match.group(1)) if scale_match else 1

            print(f"method_scale: {method_scale}, files: {files}")

            # Extract num value from each file and group by num
            for file in files:
                num_match = re.search(r'_num(\d+)\.jsonl$', file)
                if not num_match:
                    continue
                file_num = int(num_match.group(1))

                # Compute accuracy
                accuracy = calculate_simple_accuracy(file, test_file, target_scale, dataset_name)
                
                # Compute average tokens
                avg_tokens = compute_average_token_count(file, target_scale)
                print(f"file: {file}, num: {file_num}, accuracy: {accuracy}, avg_tokens: {avg_tokens}")
                
                if avg_tokens and avg_tokens > 0:
                    # Get method name part without the _scale suffix
                    method_name = method_scale.rsplit('_scale', 1)[0]
                    
                    if file_num not in data_by_num:
                        data_by_num[file_num] = []
                    data_by_num[file_num].append((avg_tokens, accuracy, method_name))

        print(f"data_by_num: {data_by_num}")

        if not data_by_num:
            print("❌ [PLOT] No plottable points (missing token info).")
            return

        # Plot
        try:
            import matplotlib.pyplot as plt  # Lazy import
            import matplotlib.cm as cm
            import numpy as np
        except ImportError:
            print("❌ [PLOT] matplotlib not found. Please run: pip install matplotlib")
            return

        plt.figure(figsize=(12, 8))
        
        # Sort by n and assign colors and markers
        sorted_nums = sorted(data_by_num.keys())
        colors = cm.tab10(np.linspace(0, 1, len(sorted_nums)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
        
        for i, n in enumerate(sorted_nums):
            points = data_by_num[n]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            labels = [p[2] for p in points]
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Plot scatter
            plt.scatter(xs, ys, s=80, alpha=0.8, color=color, marker=marker, 
                       label=f'n={n}', edgecolors='black', linewidth=0.5)
            
            # Offset labels slightly to the top-right
            for x, y, label in points:
                plt.annotate(label, (x, y), textcoords="offset points", 
                           xytext=(5, 5), ha='left', fontsize=7, alpha=0.8)

        plt.xscale('log')
        plt.xlabel('Average tokens (log scale)')
        plt.ylabel('Average accuracy (%)')
        plt.title(f'Accuracy vs Tokens ({dataset_name}, grouped by n)')
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Create output directory
        out_dir = SAVED_PLOTS_DIR
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ [PLOT] Failed to create output directory: {out_dir} ({str(e)})")
            return

        out_name = os.path.join(out_dir, f"accuracy_vs_tokens_{dataset_name}.png")
        plt.tight_layout()
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        print(f"✅ [PLOT] Saved scatter plot: {out_name}")
        plt.close()

    except Exception as e:
        print(f"❌ [PLOT] Error during plotting: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Score files in saved_choices and display accuracy per scale',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-file analysis
  python BoN_choice_analyze.py saved_choices/aime2024_run0_pick_answer_num5.jsonl /workspace/AIME_2024/aime_2024_problems.parquet
  python BoN_choice_analyze.py saved_choices/gsm8k_run1_reward_num8.jsonl templategsm_random.jsonl
  
  # Dataset-wide analysis (fixed num=5, grouped by method and scale)
  python BoN_choice_analyze.py --dataset aime2024 /workspace/AIME_2024/aime_2024_problems.parquet
  python BoN_choice_analyze.py --dataset aime2024 -N 5 /workspace/AIME_2024/aime_2024_problems.parquet
  
  # Method-specific analysis (multiple (num, scale) pairs)
  python BoN_choice_analyze.py --dataset aime2024 --evaluation_method pick_answer /workspace/AIME_2024/aime_2024_problems.parquet
        """
    )
    
    parser.add_argument(
        'choice_file',
        nargs='?',
        help='Path to the choice JSONL file (e.g., saved_choices/aime2024_run0_pick_answer_num5.jsonl)'
    )
    parser.add_argument(
        '--test_file',
        nargs='?',
        help='Path to the problem dataset file (e.g., /workspace/AIME_2024/aime_2024_problems.parquet)'
    )
    parser.add_argument(
        '--dataset',
        help='Dataset name (e.g., aime2024). If set, run analysis grouped by (method, scale) for the given num'
    )
    parser.add_argument(
        '--evaluation_method',
        help='Evaluation method name (e.g., pick_answer). If set, search multiple num files and display as num{value}_scale{value}'
    )
    parser.add_argument(
        '-N', '--num',
        type=int,
        default=5,
        help='num value (default: 5). Only files with this num will be analyzed'
    )
    parser.add_argument(
        '--saved_answers_dir',
        default=None,
        help='Directory containing answer files (default: env SAVED_ANSWERS_DIR or saved_answers)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='When running dataset-wide analysis, save scatter plot of avg accuracy(%) vs avg tokens (log scale)'
    )
    
    args = parser.parse_args()
    
    # Configure answers directory
    global SAVED_ANSWERS_DIR
    if args.saved_answers_dir:
        SAVED_ANSWERS_DIR = args.saved_answers_dir
        print(f"🔧 [CONFIG] Set answers directory: {SAVED_ANSWERS_DIR}")
    
    # Dataset-wide analysis when --dataset is provided
    if args.dataset:
        # Allow omitting --test_file for common datasets
        if not args.test_file:
            dataset_lc = args.dataset.lower()
            default_test_files = {
                'aime2024': '/workspace/AIME_2024/aime_2024_problems.parquet',
                'aime2025': '/workspace/AIME2025/aime2025-full.jsonl',
                'gpqa_diamond': '/workspace/GPQA-Diamond/test/gpqa_diamond.parquet',
                'math500': '/workspace/prm800k/prm800k/math_splits/test.jsonl',
            }
            if dataset_lc in default_test_files:
                args.test_file = default_test_files[dataset_lc]
                print(f"🔧 [CONFIG] Using default --test_file for {dataset_lc}: {args.test_file}")
            else:
                print(f"❌ [ERROR] --dataset requires specifying --test_file")
                sys.exit(1)
        
        if not os.path.exists(args.test_file):
            print(f"❌ [ERROR] Problem file not found: {args.test_file}")
            sys.exit(1)
        
        # If evaluation_method is set, run method-specific analysis
        if args.evaluation_method:
            analyze_dataset_by_method(args.dataset, args.test_file, args.evaluation_method)
            # Plotting is not supported for method-specific analysis currently
        else:
            # Run dataset-wide analysis
            analyze_dataset(args.dataset, args.test_file, args.num)
            # Generate scatter plot as well
            if args.plot:
                plot_dataset_scatter(args.dataset, args.test_file, args.num)
    else:
        # Single-file analysis
        if not args.choice_file or not args.test_file:
            print(f"❌ [ERROR] Both choice_file and test_file are required")
            parser.print_help()
            sys.exit(1)
        
        # File existence checks
        if not os.path.exists(args.choice_file):
            print(f"❌ [ERROR] Choice file not found: {args.choice_file}")
            sys.exit(1)
        
        if not os.path.exists(args.test_file):
            print(f"❌ [ERROR] Problem file not found: {args.test_file}")
            sys.exit(1)
        
        # Run analysis
        analyze_choices(args.choice_file, args.test_file)

if __name__ == "__main__":
    main() 