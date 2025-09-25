#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np, sys

# Simple mock for single process execution
class SimpleState:
    def __init__(self):
        self.process_index = 0
        self.device = 'cpu'
        self.num_processes = 1
        self.is_main_process = True
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Literal
import os
from BoN_utils import *
import requests
#from template_loader import TemplateLoader
#from template_parser import TemplateParserDecomposition
import glob
import random
import math

from dotenv import load_dotenv
# Load .env file
load_dotenv()

def display_env_config():
    """
    Debug function: Display the configuration contents of the .env file
    """
    print("\n" + "="*60)
    print("🔧 [ENV_CONFIG] Checking environment variable configuration")
    print("="*60)
    
    # Read .env file contents directly
    env_file_path = ".env"
    if os.path.exists(env_file_path):
        print(f"📁 [ENV_CONFIG] .env file contents ({env_file_path}):")
        try:
            with open(env_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        print(f"  {i:2d}: {line}")
                    elif line.startswith('#'):
                        print(f"  {i:2d}: {line}")  # Show commented lines as well
        except Exception as e:
            print(f"❌ [ENV_CONFIG] .env file read error: {str(e)}")
    else:
        print(f"❌ [ENV_CONFIG] .env file not found: {env_file_path}")
    
    # Display actual environment variable values
    print(f"\n🌐 [ENV_CONFIG] Actual environment variable values:")
    env_vars_to_check = [
        "SAVED_ANSWERS_DIR",
        "NUM_GPUS",
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "REPLICATE_API_TOKEN",
        "HUGGINGFACE_API_KEY",
        "CUDA_VISIBLE_DEVICES",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_CACHE",
        "HF_HOME",
        "VLLM_ATTENTION_BACKEND",
        "PYTHONPATH",
        "PATH"
    ]
    
    for var_name in env_vars_to_check:
        var_value = os.getenv(var_name)
        if var_value is not None:
            # Partially mask sensitive values like API keys
            if "API_KEY" in var_name or "TOKEN" in var_name:
                if len(var_value) > 8:
                    masked_value = var_value[:4] + "*" * (len(var_value) - 8) + var_value[-4:]
                else:
                    masked_value = "*" * len(var_value)
                print(f"  {var_name}: {masked_value}")
            else:
                print(f"  {var_name}: {var_value}")
        else:
            print(f"  {var_name}: (not set)")
    
    # Check important settings
    print(f"\n📋 [ENV_CONFIG] Checking important settings:")
    print(f"  Save directory: {os.getenv('SAVED_ANSWERS_DIR', 'saved_answers')}")
    print(f"  Number of GPUs: {os.getenv('NUM_GPUS', '1')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"  TOKENIZERS_PARALLELISM: {os.getenv('TOKENIZERS_PARALLELISM', '(not set)')}")
    
    print("="*60)
    print("🔧 [ENV_CONFIG] Finished checking environment variable configuration")
    print("="*60 + "\n")

# Display environment variable configuration
display_env_config()

MAX_NEW_TOKENS = 100000  # Default value
SUMMARY_MAX_TOKENS = 2000
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
print(f"TEMPERATURE = {TEMPERATURE}")

def get_max_tokens_for_dataset(dataset_type, custom_max_tokens=None):
    """
    Function to return optimal MAX_NEW_TOKENS based on dataset type
    
    Parameters
    ----------
    dataset_type : str
        Dataset type
    custom_max_tokens : int, optional
        Custom specified token count (takes priority if specified)
        
    Returns
    -------
    int
        Optimal MAX_NEW_TOKENS value
    """
    if custom_max_tokens is not None:
        return custom_max_tokens
    
    # Optimal token counts per dataset type
    dataset_tokens = {
        'math': 100000,        # MATH competition problems: concise and logical
        'math500': 100000,     # MATH500: same setting as math
        'gsm8k': 3000,        # GSM8K: middle-school level problems
        'aime2024': 100000,    # AIME 2024: high difficulty, detailed solutions needed
        'aime2024short': 100000,  # AIME short version
        'aime2025': 100000,    # AIME 2025: high difficulty, detailed solutions needed
        'mmlu_pro': 100000,     # MMLU-Pro: multiple choice, moderate explanations
        'gpqa_diamond': 50000,  # GPQA-Diamond: advanced science domain choice questions
        'auto': 10000,        # Default for auto-detection
    }
    
    tokens = dataset_tokens.get(dataset_type, MAX_NEW_TOKENS)
    
    # If MAX_MODEL_LEN is set in .env, take the minimum with it
    max_model_len = os.getenv("MAX_MODEL_LEN")
    if max_model_len is not None:
        try:
            print(f"int(max_model_len): {int(max_model_len)}")
            max_model_len_int = int(max_model_len) - SUMMARY_MAX_TOKENS - 500
            tokens = min(tokens, max_model_len_int)
            print(f"🔧 [TOKEN] Tokens for dataset '{dataset_type}': {tokens} (min with MAX_MODEL_LEN({max_model_len_int}))")
        except ValueError:
            print(f"⚠️ [TOKEN] Invalid MAX_MODEL_LEN value '{max_model_len}'. Using original: {tokens}")
    else:
        print(f"🔧 [TOKEN] Tokens for dataset '{dataset_type}': {tokens}")
    
    return tokens

# Get save directory from environment variable (default: saved_answers)
SAVED_ANSWERS_DIR = os.getenv("SAVED_ANSWERS_DIR", "saved_answers")

# Get save directory for LLM discriminator responses (default: saved_discriminators)
SAVED_DISCRIMINATORS_DIR = os.getenv("SAVED_DISCRIMINATORS_DIR", "saved_discriminators")

# Directory for existing answer files; can be same as save dir but often separate
EXISTING_ANSWERS_DIR = SAVED_ANSWERS_DIR

# Directory for saving choice information
SAVED_CHOICES_DIR = "saved_choices"

### ---- Setup -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='BoN parallel processing')
    parser.add_argument('-n', '--num_bon', type=int, default=1,
                      help='Number of samples for BoN generation (default: 4)')

    parser.add_argument('-o', '--out_dir', type=str, default='./iter_outputs',
                      help='Output directory path (default: ./iter_outputs)')
    parser.add_argument('--test_file', type=str, default='templategsm_random.jsonl',
                      help='Path to test dataset file (default: templategsm_random.jsonl). For AIME2024, use /workspace/AIME_2024/aime_2024_problems.parquet; For AIME2025, use /workspace/AIME2025/aime2025-full.jsonl; For MMLU-Pro, use /workspace/MMLU-Pro/data/validation-00000-of-00001.parquet; For GPQA-Diamond, use /workspace/GPQA-Diamond/test/gpqa_diamond.parquet')
    parser.add_argument('--dataset_type', type=str, choices=['gsm8k', 'aime2024', 'aime2024short', 'aime2025', 'math', 'math500', 'mmlu_pro', 'gpqa_diamond', 'auto'], default='auto',
                      help='Dataset type: gsm8k, aime2024, aime2024short, aime2025, math, math500, mmlu_pro, gpqa_diamond, or auto-detect (default: auto)')
    parser.add_argument('--instruction_key', type=str, default=None,
                      help='Key name for instruction/problem in dataset (auto-detected if not specified)')
    parser.add_argument('--output_key', type=str, default=None,
                      help='Key name for output/answer in dataset (auto-detected if not specified)')
    parser.add_argument('--solution_key', type=str, default=None,
                      help='Key name for solution in dataset (auto-detected if not specified)')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to process (default: 100 for normal, 1000 for complete)')
    parser.add_argument('--use_save', action='store_true', default=False,
                      help='Use save functionality for answer generation (calls generate_answerBoN_save)')
    parser.add_argument('--file_start', type=int, default=0,
                      help='Starting number for saved answer files (default: 0)')
    parser.add_argument('--evaluation_method', type=str, choices=['omni', 'pick_answer', 'llm_judge_tournament', 'llm_judge_set', 'reward', 'reward_orm', 'random', 'self_certainty', 'majority'], default='omni',
                      help='Evaluation method for answer selection: omni (cherry-picking correct answer), pick_answer (LLM-based judgment, not used), llm_judge_tournament (LLM-based L-ary tournament using pick_answer), llm_judge_set (LLM judgment using last 5000 chars before </think> tag), reward (reward model scoring), reward_orm (reward model scoring after removing thinking prefix), random (uniform random selection), majority (majority voting on extracted answers with random tie-breaking) (default: omni)')
    parser.add_argument('--tournament_L', type=int, default=4,
                      help='Grouping size L for llm_judge_tournament (default: 4). While the number of choices is at least L, split into groups of size L and keep 1 from each group using pick_answer; repeat until one remains.')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                      help='Maximum number of tokens to generate (default: auto-detect based on dataset type)')
    parser.add_argument('--run', type=int, default=1,
                      help='Run number for choice file naming (default: 1)')
    parser.add_argument('--first_valid', action='store_true', default=False,
                      help='For pick_answer and llm_judge_set: exit immediately when the first valid judgment is obtained (scale=-1)')
    return parser.parse_args()

args = parse_args()
state = SimpleState() # Auto-get rank/device (single-process version)
rank  = state.process_index
device = state.device

# Use models started by start_vllm_server.sh and start_reward_server.sh

import hashlib

def extract_self_certainty_from_file(file_path: str) -> float:
    """
    Extract Self-Certainty score from file
    
    Parameters
    ----------
    file_path : str
        Path to the answer file
        
    Returns
    -------
    float
        Self-Certainty score. Returns 0.0 if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Self-Certainty:'):
                    # Extract score from lines like "Self-Certainty: 11.612701"
                    parts = line.strip().split(':', 1)
                    if len(parts) == 2:
                        try:
                            score = float(parts[1].strip())
                            return score
                        except ValueError:
                            continue
        return 0.0  # If Self-Certainty not found
    except (FileNotFoundError, IOError):
        return 0.0  # If file cannot be read

def extract_generated_tokens_from_file(file_path: str) -> int:
    """
    Extract token count XX from "Generated Tokens: XX" in file
    Returns 0 if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Generated Tokens:'):
                    parts = line.strip().split(':', 1)
                    if len(parts) == 2:
                        value_str = parts[1].strip()
                        try:
                            return int(''.join(ch for ch in value_str if ch.isdigit()))
                        except ValueError:
                            return 0
        return 0
    except (FileNotFoundError, IOError):
        return 0

def sum_generated_tokens_for_files(filenames: list) -> int:
    """
    Sum "Generated Tokens" for each file in `filenames`.
    Files are assumed to be base names under `EXISTING_ANSWERS_DIR`.
    """
    total = 0
    for name in filenames:
        if not name:
            continue
        file_path = os.path.join(EXISTING_ANSWERS_DIR, name)
        total += extract_generated_tokens_from_file(file_path)
    return total

def calculate_self_certainty(step_logprobs, k: int = 20) -> float:
    """
    Calculate self-certainty using KL divergence between uniform distribution and top-k probability distribution
    
    Parameters
    ----------
    step_logprobs : List[List[Dict]]
        List of step logprobs for each token
        Each element is a list of dicts with 'logprob' key
        Format: [[{'token': '<think>', 'logprob': -2.33e-05, ...}, ...], ...]
    k : int
        Number of top tokens to consider (default: 20)
        
    Returns
    -------
    float
        Average KL divergence KL(U||P_{top-k}) across all tokens
    """
    print(f"🔍 [CERTAINTY] step_logprobs type: {type(step_logprobs)}")
    print(f"🔍 [CERTAINTY] step_logprobs length: {len(step_logprobs) if hasattr(step_logprobs, '__len__') else 'N/A'}")

    if not step_logprobs:
        return 0.0
    
    total_kl = 0.0
    valid_tokens = 0
    
    # Uniform distribution probability for top-k
    uniform_prob = 1.0 / k
    
    for token_idx, token_candidates in enumerate(step_logprobs):
        if not token_candidates or not isinstance(token_candidates, list):
            continue
            
        # Extract logprobs and convert to probabilities
        logprobs = []
        for candidate in token_candidates:
            if isinstance(candidate, dict) and 'logprob' in candidate:
                logprob = candidate['logprob']
                if isinstance(logprob, (int, float)) and logprob != float('-inf'):
                    logprobs.append(logprob)
        
        if len(logprobs) < k:
            print(f"🔍 [CERTAINTY] Token {token_idx}: candidates {len(logprobs)} < {k}, skipping")
            continue
            
        # Take top-k logprobs. Usually use full k=20
        top_k_logprobs = logprobs[:k]
        
        # Convert logprobs to probabilities
        try:
            probs = [math.exp(logprob) for logprob in top_k_logprobs]
        except (OverflowError, ValueError) as e:
            print(f"🔍 [CERTAINTY] Token {token_idx}: probability conversion error {str(e)}, skipping")
            continue
        
        # Normalize probabilities to sum to 1
        prob_sum = sum(probs)
        if prob_sum == 0:
            print(f"🔍 [CERTAINTY] Token {token_idx}: probability sum is 0, skipping")
            continue
            
        normalized_probs = [p / prob_sum for p in probs]
        
        # Calculate KL divergence: KL(U||P) = Σ U(i) * log(U(i) / P(i))
        kl_divergence = 0.0
        for p in normalized_probs:
            if p > 0:  # Avoid log(0)
                kl_divergence += uniform_prob * math.log(uniform_prob / p)
        
        total_kl += kl_divergence
        valid_tokens += 1
        
        if token_idx < 3:  # Debug-print first 3 tokens
            print(f"🔍 [CERTAINTY] Token {token_idx}: KL={kl_divergence:.4f}, top3 probs={normalized_probs[:3]}")
    
    if valid_tokens == 0:
        print(f"🔍 [CERTAINTY] No valid tokens")
        return 0.0
    
    # Return average KL divergence across all tokens
    avg_kl = total_kl / valid_tokens
    print(f"🔍 [CERTAINTY] Self-certainty complete: average KL over {valid_tokens} tokens = {avg_kl:.4f}")
    return avg_kl

def save_choice_info(problem_number: int, filenames: list, selected_index: int, dataset_name: str, evaluation_method: str, run_number: int, num_bon: int, scale: int = 1, total_token_count: int = 0):
    """
    Save selection information in JSONL format
    
    Parameters
    ----------
    problem_number : int
        Problem number
    filenames : list
        List of loaded files
    selected_index : int
        Index of the selected file
    dataset_name : str
        Dataset name
    evaluation_method : str
        Evaluation method
    run_number : int
        Run number
    num_bon : int
        Number of BoN
    scale : int
        Number of judgments (checkpoint scale value)
    total_token_count : int
        Sum of Generated Tokens recorded in answer files loaded for this problem
    """
    os.makedirs(SAVED_CHOICES_DIR, exist_ok=True)
    
    # Generate filename
    if evaluation_method == 'reward':
        # Get reward model name from environment variable
        reward_model_id = os.environ.get('REWARD_MODEL_ID', '')
        reward_model_name = extract_reward_model_name(reward_model_id)
        evaluation_suffix = f"reward_{reward_model_name}"
    elif evaluation_method == 'reward_orm':
        # Get reward model name from environment variable
        reward_model_id = os.environ.get('REWARD_MODEL_ID', '')
        reward_model_name = extract_reward_model_name(reward_model_id)
        evaluation_suffix = f"reward_orm_{reward_model_name}"
    # removed: comp_reward suffix generation (deprecated)
    else:
        evaluation_suffix = evaluation_method
    
    choice_filename = f"{dataset_name}_run{run_number}_{evaluation_suffix}_num{num_bon}.jsonl"
    choice_filepath = os.path.join(SAVED_CHOICES_DIR, choice_filename)
    
    # Get selected filename
    selected_filename = filenames[selected_index] if 0 <= selected_index < len(filenames) else ""
    
    # Record choice information in JSONL format
    choice_record = {
        "problem_number": int(problem_number),
        "loaded_files": filenames,
        "selected_file": selected_filename,
        "selected_index": int(selected_index),
        "scale": int(scale),
        "total_token_count": int(total_token_count)
    }
    
    # Append to JSONL file
    try:
        with open(choice_filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(choice_record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        print(f"📝 [CHOICE] Saved choice record: {choice_filepath}")
    except Exception as e:
        print(f"❌ [CHOICE] Choice record save error: {str(e)}")

def hash_string(text: str, algo: str = "sha256", /, *, encoding: str = "utf-8") -> str:
    """
    Utility function to compute hash of arbitrary string `text` with specified hash algorithm
    and return hexadecimal digest.

    Parameters
    ----------
    text : str
        String to be hashed
    algo : str, default "sha256"
        Hash function name (one from hashlib.algorithms_available)
    encoding : str, default "utf-8"
        Encoding to use when converting string to bytes

    Returns
    -------
    str
        Hexadecimal representation of hash value
    """
    try:
        hasher = hashlib.new(algo)          # Exception: ValueError if invalid algo name
    except ValueError as e:
        raise ValueError(f"Unsupported algorithm '{algo}'.") from e

    hasher.update(text.encode(encoding))
    return hasher.hexdigest()


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def get_score(
    prompt: str,
    response: str,
    ERROR_TOKEN = "[ERROR]",
    *,
    device = None
) -> float:
    URL = "http://localhost:9000/score"
    payload = {
        "prompt":   prompt,
        "response": response,
    }
    params = {"attr": "score"}   # Can also change to "first" or "mean"

    res = requests.post(URL, json=payload, params=params, timeout=30)
    res.raise_for_status()
    return res.json()['score']

def get_scores(
    prompts: List[str],
    responses: List[str]
) -> List[float]:
    return [get_score(prompt, response) for prompt, response in zip(prompts, responses)]

def get_comparison_result(
    prompt: str,
    response_a: str,
    response_b: str,
    timeout: int = 30
) -> str:
    """
    Use comparison-based reward reasoning model like RRM-7B to
    compare two answers and determine which is better
    Summarize each answer before comparison and compare the summarized versions
    
    Parameters
    ----------
    prompt : str
        Problem statement
    response_a : str
        Answer A to compare
    response_b : str
        Answer B to compare
    timeout : int
        Timeout in seconds
        
    Returns
    -------
    str
        Comparison result ("A", "B", "tie")
    """
    print(f"🔍 [COMP_SUMMARIZE] Summarizing each answer before comparison...")
    
    # Summarize answer A
    try:
        print(f"🔍 [COMP_SUMMARIZE] Summarizing answer A (length: {len(response_a)} chars)...")
        summary_a = summarize_answer(
            input_text=response_a,
            output_file=None,
            max_new_tokens=2000,
            temperature=TEMPERATURE,
            finetuned_llm=False
        )
        print(f"🔍 [COMP_SUMMARIZE] Summary for answer A complete (length: {len(summary_a)} chars)")
        print(f"🔍 [COMP_SUMMARIZE] Answer A summary: {summary_a[:200]}...")
    except Exception as e:
        print(f"⚠️ [COMP_SUMMARIZE] Failed to summarize answer A: {str(e)} → using original answer")
        summary_a = response_a
    
    # Summarize answer B
    try:
        print(f"🔍 [COMP_SUMMARIZE] Summarizing answer B (length: {len(response_b)} chars)...")
        summary_b = summarize_answer(
            input_text=response_b,
            output_file=None,
            max_new_tokens=2000,
            temperature=TEMPERATURE,
            finetuned_llm=False
        )
        print(f"🔍 [COMP_SUMMARIZE] Summary for answer B complete (length: {len(summary_b)} chars)")
        print(f"🔍 [COMP_SUMMARIZE] Answer B summary: {summary_b[:200]}...")
    except Exception as e:
        print(f"⚠️ [COMP_SUMMARIZE] Failed to summarize answer B: {str(e)} → using original answer")
        summary_b = response_b
    
    print(f"🔍 [COMP_SUMMARIZE] Summarization complete. Running comparison on summaries...")
    
    # Compare summarized abstracts - this is 9100
    URL = "http://localhost:9100/compare"
    payload = {
        "prompt": prompt,
        "response1": summary_a,
        "response2": summary_b,
    }
    
    try:
        res = requests.post(URL, json=payload, timeout=timeout)
        res.raise_for_status()
        result = res.json()
        
        # Debug log: Display API response content
        print(f"🔍 [COMP_DEBUG] API Response: {result}")
        
        # Get comparison result from API response (preference_score format)
        preference_score = result.get('preference_score', 0.5)
        print(f"🔍 [COMP_DEBUG] Extracted preference_score: {preference_score}")
        
        # Convert preference_score to "A", "B", "tie"
        # preference_score > 0.6 → A (response1 is clearly better)
        # preference_score < 0.4 → B (response2 is clearly better)
        # 0.4 <= preference_score <= 0.6 → tie (close call)
        if preference_score > 0.6:
            normalized_result = "A"
        elif preference_score < 0.4:
            normalized_result = "B"
        else:
            normalized_result = "tie"
        
        print(f"🔍 [COMP_DEBUG] Normalized result: '{normalized_result}'")
        return normalized_result
            
    except Exception as e:
        print(f"⚠️ [COMP] Comparison error: {str(e)}")
        return "tie"  # Treat as tie on error

def get_llm_comparison_result(instruction: str, response_a: str, response_b: str, run_number: int) -> str:
    """
    Have the LLM itself compare two answers and judge which is better
    Summarize each answer before comparison and perform comparison on summarized versions
    
    Parameters
    ----------
    instruction : str
        Problem statement
    response_a : str
        Answer A to compare
    response_b : str
        Answer B to compare
    run_number : int
        Run number (used for GPU distribution)
        
    Returns
    -------
    str
        Comparison result ("A", "B", "tie")
    """
    print(f"🔍 [LLM_COMP] Starting LLM direct-judgment comparison...")
    
    # Summarize answer A
    try:
        print(f"🔍 [LLM_COMP] Summarizing answer A (length: {len(response_a)} chars)...")
        summary_a = summarize_answer(
            input_text=response_a,
            output_file=None,
            max_new_tokens=2000,
            temperature=TEMPERATURE,
            finetuned_llm=False
        )
        print(f"🔍 [LLM_COMP] Summary for answer A complete (length: {len(summary_a)} chars)")
        print(f"🔍 [LLM_COMP] Answer A summary: {summary_a[:200]}...")
    except Exception as e:
        print(f"⚠️ [LLM_COMP] Failed to summarize answer A: {str(e)} → using original answer")
        summary_a = response_a
    
    # Summarize answer B
    try:
        print(f"🔍 [LLM_COMP] Summarizing answer B (length: {len(response_b)} chars)...")
        summary_b = summarize_answer(
            input_text=response_b,
            output_file=None,
            max_new_tokens=2000,
            temperature=TEMPERATURE,
            finetuned_llm=False
        )
        print(f"🔍 [LLM_COMP] Summary for answer B complete (length: {len(summary_b)} chars)")
        print(f"🔍 [LLM_COMP] Answer B summary: {summary_b[:200]}...")
    except Exception as e:
        print(f"⚠️ [LLM_COMP] Failed to summarize answer B: {str(e)} → using original answer")
        summary_b = response_b
    
    print(f"🔍 [LLM_COMP] Summarization complete. Running direct comparison with LLM...")
    
    # Prompt to request direct comparison judgment from LLM
    comparison_prompt = f"""Please evaluate the following 2 answer summaries for this mathematical problem and determine which answer you think is better.

Problem:
{instruction}

Answer A Summary:
{summary_a}

Answer B Summary:
{summary_b}

Which answer do you think is better, more correct, logical, and complete? If both answers are roughly equivalent in quality, you may judge them as a tie.

Please provide detailed reasoning for your judgment, and then output your choice using one of the following formats:
- \\boxed{{A}} if Answer A is better
- \\boxed{{B}} if Answer B is better  
- \\boxed{{tie}} if they are roughly equivalent

Judgment:"""
    
    try:
        # GPU distribution config (based on run_number)
        num_gpus = int(os.getenv("NUM_GPUS", "1"))
        base_port = 8100
        gpu_index = run_number % num_gpus
        port = base_port + gpu_index
        print(f"🔧 [LLM_COMP] Using GPU {gpu_index} (port {port})")
        
        # Get appropriate token count for dataset
        max_tokens = get_max_tokens_for_dataset(args.dataset_type, args.max_new_tokens)
        judgment_max_tokens = min(max_tokens // 3, 8000)  # Keep judgments short
        
        judgment_results = generate_answer_client(
            prompt=comparison_prompt,
            max_new_tokens=judgment_max_tokens,
            n=1,
            temperature=TEMPERATURE,
            port=port,
        )
        
        if judgment_results and len(judgment_results) > 0:
            judgment, _ = judgment_results[0]
            print(f"🔍 [LLM_COMP] Judgment result: {judgment}")
            
            # Parse boxed answer
            try:
                boxed_answer = extract_boxed_answer(judgment)
                normalized_result = boxed_answer.strip().lower()
                
                if normalized_result == "a":
                    result = "A"
                elif normalized_result == "b":
                    result = "B"
                elif normalized_result == "tie":
                    result = "tie"
                else:
                    print(f"⚠️ [LLM_COMP] Unknown judgment '{boxed_answer}' → treating as tie")
                    result = "tie"
                
                print(f"🔍 [LLM_COMP] Normalized result: '{result}'")
                return result
                
            except Exception as parse_error:
                print(f"⚠️ [LLM_COMP] Failed to parse judgment: {str(parse_error)} → treating as tie")
                return "tie"
        else:
            print(f"⚠️ [LLM_COMP] Failed to generate judgment → treating as tie")
            return "tie"
            
    except Exception as e:
        print(f"⚠️ [LLM_COMP] Comparison error: {str(e)} → treating as tie")
        return "tie"


def compare_all_pairs(instruction: str, answers: List[str], run_number: int) -> Tuple[str, int]:
    """
    Compare all answer pairs and select the answer with the most wins
    Request direct judgment from LLM itself, summarizing each answer before comparison
    
    Parameters
    ----------
    instruction : str
        Problem statement
    answers : List[str]
        List of answers to compare
    run_number : int
        Run number (used for GPU distribution)
        
    Returns
    -------
    Tuple[str, int]
        (Selected optimal answer, estimated token count used)
    """
    if not answers:
        return "", 0
    
    if len(answers) == 1:
        return answers[0], 0
    
    print(f"🔄 [COMP] Starting pairwise comparison with {len(answers)} answers...")
    print(f"🔄 [COMP] Using LLM direct judgment; summarizing answers before each comparison")
    
    # Count wins
    win_counts = [0.0] * len(answers)
    total_comparisons = 0
    total_tokens = 0
    
    # Compare all pairs
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            print(f"\n🔄 [COMP] === Comparison {i+1} vs {j+1} ===")
            print(f"🔄 [COMP] Answer {i+1} length: {len(answers[i])}")
            print(f"🔄 [COMP] Answer {j+1} length: {len(answers[j])}")
            
            result = get_llm_comparison_result(instruction, answers[i], answers[j], run_number)
            total_comparisons += 1
            # Estimate token usage for LLM judgment (2 summaries + 1 judgment)
            total_tokens += 2 * 2000 + 8000  # 2 summaries + judgment
            
            if result == "A":
                win_counts[i] += 1.0
                print(f"  → Answer {i+1} wins")
            elif result == "B":
                win_counts[j] += 1.0
                print(f"  → Answer {j+1} wins")
            else:
                # For ties, add 0.5 points to both
                win_counts[i] += 0.5
                win_counts[j] += 0.5
                print(f"  → Tie")
    
    print(f"\n📊 [COMP] === Comparison summary ===")
    for i, count in enumerate(win_counts):
        print(f"📊 [COMP] Answer {i+1}: {count:.1f} wins")
    
    # Select the answer with most wins
    best_index = win_counts.index(max(win_counts))
    best_answer = answers[best_index]
    best_wins = win_counts[best_index]
    
    print(f"✅ [COMP] Final selection: Answer {best_index+1} ({best_wins:.1f} wins/{total_comparisons} comparisons)")
    print(f"📊 [COMP] Estimated tokens used: {total_tokens} (LLM direct judgment)")
    
    return best_answer, total_tokens


def generate_discriminator_filename(result_type, evaluation_method, dataset_type, n, run_number, problem_index):
    """
    Generate filename for discriminator result files
    
    Parameters
    ----------
    result_type : str
        Type of result ("judgment", "summary")
    evaluation_method : str
        Evaluation method
    dataset_type : str
        Dataset name
    n : int
        BoN sample count
    run_number : int
        Run number
    problem_index : int
        Problem number
        
    Returns
    -------
    str
        Generated filename
    """
    filename = f"discriminator_{result_type}_{evaluation_method}_{dataset_type}_n{n}_problem{problem_index}_run{run_number}.json"
    return filename


def save_discriminator_result(result_data, run_number, n, evaluation_method, dataset_type="unknown", compared_files=None, judgment_round=None, answer_index=None, result_type="judgment", problem_index=None):
    """
    Save LLM discriminator results to JSON file by appending
    
    Parameters
    ----------
    result_data : tuple
        Tuple in format (response, num_tokens)
    run_number : int
        Run number
    n : int
        BoN sample count
    evaluation_method : str
        Evaluation method
    dataset_type : str
        Dataset name (default: "unknown")
    compared_files : list, optional
        List of files being compared
    judgment_round : int, optional
        Judgment round number (for judgment results)
    answer_index : int, optional
        Answer index (for summary results)
    result_type : str
        Type of result ("judgment", "summary")
    problem_index : int, optional
        Problem number
    """
    try:
        # Create directory
        os.makedirs(SAVED_DISCRIMINATORS_DIR, exist_ok=True)
        
        response, num_tokens = result_data
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Up to milliseconds (for JSON data)
        filename = generate_discriminator_filename(
            result_type, evaluation_method, dataset_type, n, run_number, problem_index
        )
        
        filepath = os.path.join(SAVED_DISCRIMINATORS_DIR, filename)
        
        # New data entry
        new_entry = {
            "timestamp": timestamp,
            "response": response,
            "num_tokens": num_tokens,
            "run_number": run_number,
            "n": n,
            "evaluation_method": evaluation_method,
            "dataset_type": dataset_type,
            "compared_files": compared_files,
            "judgment_round": judgment_round,
            "answer_index": answer_index,
            "result_type": result_type,
            "problem_index": problem_index
        }
        
        # Load existing file if exists, create new if not
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]  # Convert to list if existing data is single object
        else:
            data = []
        
        # Add new entry
        data.append(new_entry)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 [DISCRIMINATOR] Appended result: {filename}")
        
    except Exception as e:
        print(f"⚠️ [DISCRIMINATOR] Save error: {str(e)}")


def open_discriminator_result(result_type, evaluation_method, dataset_type, n, run_number, problem_index, judgment_round=None, answer_index=None):
    """
    Load saved discriminator results from JSON file and find entries matching conditions
    
    Parameters
    ----------
    result_type : str
        Type of result ("judgment", "summary")
    evaluation_method : str
        Evaluation method
    dataset_type : str
        Dataset name
    n : int
        BoN sample count
    run_number : int
        Run number
    problem_index : int
        Problem number
    judgment_round : int, optional
        Judgment round number (for judgment results)
    answer_index : int, optional
        Answer index (for summary results)
        
    Returns
    -------
    dict or None
        JSON data entry matching conditions, None if not found
    """
    try:
        # Generate filename
        filename = generate_discriminator_filename(
            result_type, evaluation_method, dataset_type, n, run_number, problem_index
        )
        
        filepath = os.path.join(SAVED_DISCRIMINATORS_DIR, filename)
        
        # File existence check
        if not os.path.exists(filepath):
            print(f"⚠️ [DISCRIMINATOR] File not found: {filename}")
            return None
        
        # Load JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to list if data is not a list
        if not isinstance(data, list):
            data = [data]
        
        # Find entries matching conditions
        for entry in data:
            match = True
            
            # Check if judgment_round is specified
            if judgment_round is not None:
                if entry.get("judgment_round") != judgment_round:
                    match = False
            
            # Check if answer_index is specified
            if answer_index is not None:
                if entry.get("answer_index") != answer_index:
                    match = False
            
            if match:
                print(f"✅ [DISCRIMINATOR] Found matching entry: {filename}")
                return entry
        
        print(f"⚠️ [DISCRIMINATOR] No matching entry found: {filename}")
        return None
        
    except json.JSONDecodeError as e:
        print(f"⚠️ [DISCRIMINATOR] JSON parse error: {str(e)}")
        return None
    except Exception as e:
        print(f"⚠️ [DISCRIMINATOR] Read error: {str(e)}")
        return None


def list_discriminator_results(dataset_type=None, evaluation_method=None, n=None, run_number=None):
    """
    Get list of saved discriminator result files
    
    Parameters
    ----------
    dataset_type : str, optional
        Filter by dataset name
    evaluation_method : str, optional
        Filter by evaluation method
    n : int, optional
        Filter by BoN sample count
    run_number : int, optional
        Filter by run number
        
    Returns
    -------
    list
        List of matching filenames
    """
    try:
        if not os.path.exists(SAVED_DISCRIMINATORS_DIR):
            print(f"⚠️ [DISCRIMINATOR] Directory does not exist: {SAVED_DISCRIMINATORS_DIR}")
            return []
        
        # Get all JSON files
        all_files = [f for f in os.listdir(SAVED_DISCRIMINATORS_DIR) if f.endswith('.json')]
        
        # Filtering
        filtered_files = []
        for filename in all_files:
            # Pattern matching for filename
            if not filename.startswith('discriminator_'):
                continue
                
            # Check filter conditions
            match = True
            if dataset_type and dataset_type not in filename:
                match = False
            if evaluation_method and evaluation_method not in filename:
                match = False
            if n is not None and f"_n{n}_" not in filename:
                match = False
            if run_number is not None and f"_run{run_number}_" not in filename:
                match = False
                
            if match:
                filtered_files.append(filename)
        
        print(f"📁 [DISCRIMINATOR] Found {len(filtered_files)} files")
        return sorted(filtered_files)
        
    except Exception as e:
        print(f"⚠️ [DISCRIMINATOR] List files error: {str(e)}")
        return []



# c.f., https://github.com/Guangxuan-Xiao/GSM8K-eval
import os
import json
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False

def detect_dataset_type(file_path, dataset_type='auto'):
    """Detect dataset type and return appropriate field mapping"""
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
                'solution_key': 'question'  # AIME2025 has no solution, so use question as solution
            }
        elif dataset_type == 'math':
            return {
                'instruction_key': 'problem',
                'output_key': 'answer',  # MATH dataset has answers in 'solution' field
                'solution_key': 'solution'
            }
        elif dataset_type == 'math500':
            return {
                'instruction_key': 'problem',
                'output_key': 'answer',  # math500 uses same field mapping
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
                'solution_key': 'question'  # GPQA-Diamond has no solution, so use question as solution
            }
    
    # Auto-detection by file extension
    if file_path.endswith('.parquet'):
        # For parquet files, sample content for judgment
        try:
            if HAS_PANDAS and pd is not None:
                df = pd.read_parquet(file_path)  # type: ignore
                columns = df.columns.tolist()
                
                # MATH dataset detection
                if 'problem' in columns and 'solution' in columns and 'level' in columns and 'type' in columns:
                    return {
                        'instruction_key': 'problem',
                        'output_key': 'solution',
                        'solution_key': 'solution'
                    }
                # AIME2024 dataset detection
                elif 'Problem' in columns and 'Answer' in columns:
                    return {
                        'instruction_key': 'Problem',
                        'output_key': 'Answer',
                        'solution_key': 'Solution'
                    }
                # MMLU-Pro dataset detection
                elif 'question' in columns and 'options' in columns and 'answer' in columns and 'cot_content' in columns:
                    return {
                        'instruction_key': 'question',
                        'output_key': 'answer',
                        'solution_key': 'cot_content'
                    }
                # GPQA-Diamond dataset detection
                elif 'question' in columns and 'answer' in columns and len(columns) == 2:
                    return {
                        'instruction_key': 'question',
                        'output_key': 'answer',
                        'solution_key': 'question'  # GPQA-Diamond has no solution, so use question as solution
                    }
        except:
            pass
        
        # Default is AIME2024 format
        return {
            'instruction_key': 'Problem',
            'output_key': 'Answer', 
            'solution_key': 'Solution'
        }
    else:
        # For JSONL files, sample first line for judgment
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
                    'solution_key': 'question'  # AIME2025 has no solution, so use question as solution
                }
            else:
                return {
                    'instruction_key': 'problem', 
                    'output_key': 'result',
                    'solution_key': 'solution_wocode'
                }
        except:
            # Default is GSM8K format
            return {
                'instruction_key': 'problem',
                'output_key': 'result', 
                'solution_key': 'solution_wocode'
            }

def load_data(file_path, instruction_key, output_key, solution_key):
    """Load JSONL or Parquet file and return in unified format"""
    list_data_dict = []
    
    # Determine if MMLU-Pro dataset
    is_mmlu_pro = False
    if file_path.endswith('.parquet'):
        if not HAS_PANDAS:
            raise ImportError("pandas is required for reading Parquet files. Please run pip install pandas.")
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
                
                # Format options as strings (with A-J labels)
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

def load_jsonl(
    file_path,
    instruction="instruction",
    input="input", 
    output="output",
    solution_wocode="solution_wocode",
    category="category",
    is_gzip=False,
):
    # Maintained for backward compatibility
    return load_data(file_path, instruction, output, solution_wocode)

# Set test file path
test_filepath = os.path.join(".", args.test_file)

# Auto-detect dataset type or use user specification
field_mapping = detect_dataset_type(test_filepath, args.dataset_type)

# Override with user-specified keys if available
if args.instruction_key:
    field_mapping['instruction_key'] = args.instruction_key
if args.output_key:
    field_mapping['output_key'] = args.output_key  
if args.solution_key:
    field_mapping['solution_key'] = args.solution_key

instruction_key = field_mapping['instruction_key']
output_key = field_mapping['output_key']
solution_key = field_mapping['solution_key']

# Set default sample count
if args.max_samples is None:
    max_samples = 1000
else:
    max_samples = args.max_samples

list_data_dict = load_data(
    test_filepath,
    instruction_key,
    output_key, 
    solution_key
)[:max_samples]

# For testing, AIME2024 difficult problem list
if args.dataset_type == "aime2024short":
    print("AIME SHORTLIST")
    selected_indices = [21, 26] # [5, 6, 17, 19, 21, 26]
    list_data_dict = [list_data_dict[i] for i in selected_indices if i < len(list_data_dict)]

print("test_filepath = ", test_filepath)
print(f"Detected dataset type: {args.dataset_type}")
print(f"Using keys: instruction='{instruction_key}', output='{output_key}', solution='{solution_key}'")
print(f"Loaded {len(list_data_dict)} samples (max_samples={max_samples})")
#list_data_dict[0]['output'].split("####")[-1].strip()

print("list_data_dict[0] = ", list_data_dict[0])

# Each process receives only the samples it is responsible for
indices = range(state.process_index, len(list_data_dict), state.num_processes)
print(f"rank = {rank}, indices = {indices}")
local_data = [(i, list_data_dict[i]) for i in indices]


### ---- Main Body -------------------------------------------------------------
# Create output folder
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)
out_file = out_dir / f"results_rank{rank}.jsonl"

# Remove answers with same values
def extract_last_before_think_tag(answer: str, max_chars: int = 5000) -> str:
    """
    Function to extract the last specified number of characters before </think> tag
    
    Parameters
    ----------
    answer : str
        Answer text
    max_chars : int
        Maximum number of characters to extract (default: 5000)
        
    Returns
    -------
    str
        Last max_chars characters before </think> tag
    """
    try:
        # Search for </think> tag position
        think_end_index = answer.rfind('</think>')
        
        if think_end_index == -1:
            # If </think> tag not found, return last max_chars characters of entire text
            if len(answer) <= max_chars:
                return answer
            else:
                return answer[-max_chars:]
        
        # Get text before </think> tag
        before_think = answer[:think_end_index]
        
        # Extract last max_chars characters
        if len(before_think) <= max_chars:
            return before_think
        else:
            return before_think[-max_chars:]
            
    except Exception as e:
        print(f"⚠️ [EXTRACT] Error extracting text before </think>: {str(e)}")
        # On error, return the last max_chars of the entire text
        if len(answer) <= max_chars:
            return answer
        else:
            return answer[-max_chars:]

def answers_representative(answers):
    """
    Function to keep only one copy of same answers
    
    Parameters
    ----------
    answers : list
        List of answers
        
    Returns
    -------
    list
        List of answers with duplicates removed (order preserved)
    """
    if not answers:
        return []
    
    seen_boxed = set()
    seen_full = set()
    result = []
    
    for answer in answers:
        # First extract boxed answer for comparison
        try:
            boxed_answer = extract_boxed_answer(answer)
            if boxed_answer and boxed_answer.strip():
                # If boxed answer can be extracted, use it for duplicate detection
                boxed_normalized = boxed_answer.strip()
                if boxed_normalized not in seen_boxed:
                    seen_boxed.add(boxed_normalized)
                    result.append(answer)
                    print(f"Added: answer with boxed '{boxed_normalized}'")
                else:
                    print(f"Dedup: boxed answer '{boxed_normalized}' already exists")
            else:
                # If boxed answer cannot be extracted, compare full text
                normalized = answer.strip()
                if normalized and normalized not in seen_full:
                    seen_full.add(normalized)
                    result.append(answer)
                    print(f"Added: answer without boxed (length: {len(normalized)} chars)")
                else:
                    print(f"Dedup: identical answer already exists")
        except Exception as e:
            # If extract_boxed_answer fails, compare full text
            print(f"Error extracting boxed answer: {str(e)}")
            normalized = answer.strip()
            if normalized and normalized not in seen_full:
                seen_full.add(normalized)
                result.append(answer)
                print(f"Added: answer after error (length: {len(normalized)} chars)")
            else:
                print(f"Dedup: identical answer after error already exists")
    
    print(f"Dedup result: {len(answers)} → {len(result)} answers")
    return result


def llm_judge_set(instruction, answers, run_number, delta=0.1, max_judgments=2, adaptive=False, first_valid=False, n=None, evaluation_method="llm_judge_set", compared_files=None, dataset_type="unknown", problem_index=None):
    """
    Function to select optimal answer from multiple answers (using last 5000 characters before </think> tag)
    
    Parameters
    ----------
    instruction : str
        Problem statement
    answers : list
        List of answers
    run_number : int
        Run number
    delta : float
        Confidence interval parameter
    max_judgments : int
        Maximum number of judgments
    adaptive : bool
        Whether to use adaptive convergence judgment (default: False)
        If False, compare up to max_judgments times without convergence judgment
    first_valid : bool
        Whether to exit immediately on first valid judgment (default: False)
        If True, skip checkpoint saving and exit immediately when sum(answer_counts) > 0
    n : int, optional
        BoN sample count
    evaluation_method : str
        Evaluation method (default: "llm_judge_set")
    compared_files : list, optional
        List of files being compared
    dataset_type : str
        Dataset name (default: "unknown")
        
    Returns
    -------
    list
        List of (selected optimal answer index, token count used) at each checkpoint (2, 4, 8, ..., max_judgments)
    """
    answers = answers_representative(answers) # Remove answers with same values
    print(f"llm_judge_set: answers_representative: len(answers) = {len(answers)}")
    
    # Initialize token counter
    total_tokens = 0
    
    # GPU distribution settings
    num_gpus = int(os.getenv("NUM_GPUS", "1"))  # Get available GPU count from environment variable
    base_port = 8100  # Base port number
    print(f"🔧 [PICK_LS] GPU distribution settings: {num_gpus} GPUs, base port {base_port}")
    
    # Generate checkpoints: 2, 4, 8, ..., max_judgments
    if not first_valid:
        checkpoints = []
        power = 0
        while 2 ** power <= max_judgments:
            checkpoints.append(2 ** power)
            power += 1
        if checkpoints[-1] != max_judgments:
            checkpoints.append(max_judgments)
    else:
        checkpoints = [-1]
    
    print(f"🔧 [PICK_LS] Checkpoints: {checkpoints}")

    if not answers:
        return [(-1, 0) for _ in checkpoints]
    
    if len(answers) == 1:
        return [(0, 0) for _ in checkpoints]
    
    # Check boxed answer for each answer - no comparison needed if all same
    print(f"🔍 [PICK_LS] Checking boxed answers for each answer...")
    boxed_answers = []
    for i, answer in enumerate(answers):
        try:
            boxed_answer = extract_boxed_answer(answer)
            if boxed_answer and boxed_answer.strip():
                boxed_answers.append(boxed_answer.strip())
                print(f"🔍 [PICK_LS] Answer {i+1}: \\boxed{{{boxed_answer.strip()}}}")
            else:
                boxed_answers.append(None)
                print(f"🔍 [PICK_LS] Answer {i+1}: no boxed answer")
        except Exception as e:
            boxed_answers.append(None)
            print(f"🔍 [PICK_LS] Answer {i+1}: extract_boxed_answer error: {str(e)}")
    
    # Create set of valid boxed answers
    valid_boxed_answers = [ba for ba in boxed_answers if ba is not None]
    unique_boxed_answers = set(valid_boxed_answers)
    
    print(f"🔍 [PICK_LS] Valid boxed answer count: {len(valid_boxed_answers)}, unique count: {len(unique_boxed_answers)}")
    
    if len(unique_boxed_answers) == 1:
        unique_answer = list(unique_boxed_answers)[0]
        print(f"✅ [PICK_LS] All answers have same boxed answer: \\boxed{{{unique_answer}}} → selecting first answer")
        return [(0, 0) for _ in checkpoints]
    elif len(unique_boxed_answers) > 1:
        print(f"⚠️ [PICK_LS] Different boxed answers exist: {unique_boxed_answers} → executing comparison process")
    else:
        print(f"⚠️ [PICK_LS] No valid boxed answers found → executing comparison process")
    
    # Initialize necessary variables
    answer_counts = [0] * len(answers)
    all_judgments = []
    valid_judgments = 0
    judgment_round = 0
    results = []  # Save results at checkpoints
    
    # Pre-extract the last 5000 characters before the </think> tag used every time
    print(f"\n📚 [PICK_LS] Extracting the last 5000 characters before </think> tag from each answer...")
    last_parts = []
    for i, answer in enumerate(answers):
        last_part = extract_last_before_think_tag(answer, max_chars=5000)
        last_parts.append(last_part)
        print(f"📚 [PICK_LS] Answer{i+1}: Extracted characters = {len(last_part)} (Original answer: {len(answer)} characters)")
        if len(last_part) >= 4950:  # When close to 5000 characters
            print(f"📚 [PICK_LS] Answer{i+1}: Extraction reached the 5000 character limit")
    
    # Judgment loop
    while judgment_round < max_judgments:
        print(f"\n🔄 [PICK_LS] Judgment {judgment_round + 1}/{max_judgments}: Comparing with strings before </think> tag...")
        
        # Dynamically generate answer summaries (using the last string before </think> tag)
        summary_sections = []
        for i, last_part in enumerate(last_parts):
            summary_sections.append(f"Answer {i+1} (Last 5000 chars before </think>):\n{last_part}")
        
        summary_text = "\n\n".join(summary_sections)
        num_answers = len(last_parts)
        answer_options_list = [str(i+1) for i in range(num_answers)]
        answer_options_str = ", ".join(answer_options_list)
        
        judgment_prompt = f"""Please evaluate the following {num_answers} answer excerpts for this mathematical problem and determine which answer you think is the most correct.

Problem:
{instruction}

{summary_text}

Among the above {num_answers} answer excerpts (showing the last parts before </think> tag), which answer do you think is the most correct, logical, and complete?

Please provide detailed reasoning for your judgment, and then output the number of the answer you think is correct ({answer_options_str}) enclosed in \\boxed{{}}.

Example: \\boxed{{1}}

Judgment:"""
        
        print(f"\nExecuting judgment {judgment_round + 1}/{max_judgments}...")
        
        # GPU distribution for judgment (according to judgment round)
        judgment_gpu_index = judgment_round % num_gpus
        judgment_port = base_port + judgment_gpu_index
        print(f"🔧 [PICK_LS] Judgment{judgment_round + 1}: Using GPU {judgment_gpu_index} (port {judgment_port})")
        
        try:
            # Check cached judgment results
            cached_judgment = open_discriminator_result(
                "judgment", evaluation_method, dataset_type, n, run_number, problem_index,
                judgment_round=judgment_round
            )
            
            if cached_judgment:
                print(f"📋 [PICK_LS] Loading judgment{judgment_round + 1} from cache")
                judgment = cached_judgment['response']
                token_count = cached_judgment['num_tokens']
                total_tokens += token_count
                print(f"📊 [PICK_LS] Judgment{judgment_round + 1} (cached): {token_count} tokens")
                print(f"🔍 [PICK_LS] Judgment{judgment_round + 1} content: {judgment}")
                all_judgments.append(judgment)
                judgment_results = [(judgment, token_count)]
            else:
                # Execute new judgment if not cached
                print(f"🔧 [PICK_LS] Executing new judgment{judgment_round + 1}...")
                
                # Execute judgment
                max_tokens = get_max_tokens_for_dataset(args.dataset_type, args.max_new_tokens)
                judgment_max_tokens = min(max_tokens // 3, 8000)  # Keep judgment shorter
                
                judgment_results = generate_answer_client(
                    prompt=judgment_prompt,
                    max_new_tokens=judgment_max_tokens,
                    n=1,
                    port=judgment_port,  # Specify distributed port
                )
            
            if judgment_results and len(judgment_results) > 0:
                # If loaded from cache, judgment and token_count are already set
                if not cached_judgment:
                    judgment, token_count = judgment_results[0]
                    total_tokens += token_count
                    print(f"📊 [PICK_LS] Judgment{judgment_round + 1}: {token_count} tokens (GPU {judgment_gpu_index})")
                    print(f"🔍 [PICK_LS] Judgment{judgment_round + 1} content: {judgment}")
                    all_judgments.append(judgment)
                    
                    # Save discriminator results (only for newly generated ones)
                    save_discriminator_result(
                        (judgment, token_count),
                        run_number,
                        n,
                        evaluation_method,
                        dataset_type,
                        compared_files,
                        judgment_round=judgment_round,
                        result_type="judgment",
                        problem_index=problem_index
                    )
                
                # Parse boxed answer
                try:
                    boxed_answer = extract_boxed_answer(judgment)
                    
                    # Check validity of answer number
                    if boxed_answer.strip() in answer_options_list:
                        selected_answer_num = int(boxed_answer.strip()) - 1
                        
                        # Range check
                        if 0 <= selected_answer_num < len(answer_counts):
                            answer_counts[selected_answer_num] += 1
                            valid_judgments += 1
                            print(f"  → Selected answer{selected_answer_num+1}")
                        else:
                            print(f"  → Answer number out of range: {selected_answer_num}")
                    else:
                        print(f"  → Invalid answer number: '{boxed_answer}'")
                        
                except Exception as parse_error:
                    print(f"  → Parse error: {str(parse_error)}")
                
            else:
                print(f"  → Failed to generate judgment result")
                
        except Exception as e:
            print(f"  → Judgment processing error: {str(e)}")
        
        judgment_round += 1
        
        # In first_valid mode, terminate immediately after first valid judgment
        if first_valid and sum(answer_counts) > 0:
            current_best_index = np.argmax(answer_counts)
            print(f"🚀 [PICK_LS] first_valid mode: Terminating after first valid judgment - selecting answer{current_best_index+1}")
            # Return result with scale=-1
            return [(current_best_index, total_tokens)]
        
        # Record results at checkpoints (only when not in first_valid mode)
        if not first_valid and judgment_round in checkpoints:
            if sum(answer_counts) > 0:
                current_best_index = np.argmax(answer_counts)
                print(f"📊 [PICK_LS] Checkpoint{judgment_round}: Answer{current_best_index+1} (selection count: {answer_counts[current_best_index]}/{sum(answer_counts)})")
            else:
                current_best_index = 0
                print(f"📊 [PICK_LS] Checkpoint{judgment_round}: No valid judgments, selecting answer1")
            results.append((current_best_index, total_tokens))
        
        # Adaptive convergence judgment (only when adaptive is True)
        if adaptive and sum(answer_counts) >= 2:  # Minimum 2 judgments required
            try:
                mus = [answer_counts[i] / sum(answer_counts) for i in range(len(answers))]
                
                if len(mus) >= 2:
                    sorted_mus = sorted(mus, reverse=True)
                    gap = sorted_mus[0] - sorted_mus[1]  # Maximum value - second largest value
                    
                    threshold = np.sqrt(2 * np.log(1.0/delta) / sum(answer_counts))
                    
                    if gap >= threshold:
                        print(f"Adaptive convergence judgment: gap={gap:.4f} >= threshold={threshold:.4f}")
                        break
                        
            except Exception as e:
                print(f"Adaptive convergence judgment error: {str(e)}")
        elif not adaptive:
            print(f"Fixed judgment mode: {judgment_round}/{max_judgments} rounds completed")
    
    # Display GPU distribution summary
    print(f"\n🔧 [PICK_LS] === GPU Distribution Summary ===")
    print(f"🔧 [PICK_LS] Number of answers: {len(answers)} (pre-extracted characters before </think> tag)")
    print(f"🔧 [PICK_LS] Judgments executed: {judgment_round} (distributed to: GPU 0-{num_gpus-1})")
    print(f"🔧 [PICK_LS] Port range used: {base_port}-{base_port+num_gpus-1}")
    
    print(f"📊 [PICK_LS] Total tokens used: {total_tokens}")
    print(f"📊 [PICK_LS] Number of checkpoint results: {len(results)}")
    
    # Verify that all checkpoint results are available
    if len(results) != len(checkpoints):
        print(f"⚠️ [PICK_LS] Result count mismatch: expected {len(checkpoints)}, got {len(results)}")
        # Fill missing results with the last result
        while len(results) < len(checkpoints):
            if results:
                results.append(results[-1])
            else:
                results.append((0, total_tokens))
    
    return results

def pick_answer(instruction, answers, run_number, delta=0.1, max_judgments=2, adaptive=False, first_valid=False, n=None, evaluation_method="pick_answer", compared_files=None, dataset_type="unknown", problem_index=None):
    """
    Function to select the optimal answer from multiple answers
    
    Parameters
    ----------
    instruction : str
        Problem statement
    answers : list
        List of answers
    run_number : int
        Run number
    delta : float
        Confidence interval parameter
    max_judgments : int
        Maximum number of judgments
    adaptive : bool
        Whether to use adaptive convergence judgment (default: False)
        If False, compare up to max_judgments times without convergence judgment
    first_valid : bool
        Whether to terminate immediately after first valid judgment (default: False)
        If True, skip checkpoint saving and terminate immediately when sum(answer_counts) > 0
    n : int, optional
        BoN sample count
    evaluation_method : str
        Evaluation method (default: "pick_answer")
    compared_files : list, optional
        List of files to compare
    dataset_type : str
        Dataset name (default: "unknown")
        
    Returns
    -------
    list
        List of (index of selected optimal answer, number of tokens used) at each checkpoint (2, 4, 8, ..., max_judgments)
    """
    answers = answers_representative(answers) # Remove answers with same values
    print(f"pick_answer: answers_representative: len(answers) = {len(answers)}")
    
    # Initialize token counter
    total_tokens = 0
    
    # GPU distribution settings
    num_gpus = int(os.getenv("NUM_GPUS", "1"))  # Get available GPU count from environment variable
    base_port = 8100  # Base port number
    print(f"🔧 [PICK] GPU distribution settings: {num_gpus} GPUs, base port {base_port}")
    
    # Generate checkpoints: 2, 4, 8, ..., max_judgments
    if not first_valid:
        checkpoints = []
        power = 0
        while 2 ** power <= max_judgments:
            checkpoints.append(2 ** power)
            power += 1
        if checkpoints[-1] != max_judgments:
            checkpoints.append(max_judgments)
    else:
        checkpoints = [-1]

    
    print(f"🔧 [PICK] Checkpoints: {checkpoints}")

    if not answers:
        return [(-1, 0) for _ in checkpoints]
    
    if len(answers) == 1:
        return [(0, 0) for _ in checkpoints]
    
    # Check boxed answer for each answer - no comparison needed if all are the same
    print(f"🔍 [PICK] Checking boxed answers for each answer...")
    boxed_answers = []
    for i, answer in enumerate(answers):
        try:
            boxed_answer = extract_boxed_answer(answer)
            if boxed_answer and boxed_answer.strip():
                boxed_answers.append(boxed_answer.strip())
                print(f"🔍 [PICK] Answer{i+1}: \\boxed{{{boxed_answer.strip()}}}")
            else:
                boxed_answers.append(None)
                print(f"🔍 [PICK] Answer{i+1}: No boxed answer")
        except Exception as e:
            boxed_answers.append(None)
            print(f"🔍 [PICK] Answer{i+1}: extract_boxed_answer error: {str(e)}")
    
    # Create set of valid boxed answers
    valid_boxed_answers = [ba for ba in boxed_answers if ba is not None]
    unique_boxed_answers = set(valid_boxed_answers)
    
    print(f"🔍 [PICK] Valid boxed answer count: {len(valid_boxed_answers)}, unique count: {len(unique_boxed_answers)}")
    
    if len(unique_boxed_answers) == 1:
        unique_answer = list(unique_boxed_answers)[0]
        print(f"✅ [PICK] All answers have same boxed answer: \\boxed{{{unique_answer}}} → selecting first answer")
        return [(0, 0) for _ in checkpoints]
    elif len(unique_boxed_answers) > 1:
        print(f"⚠️ [PICK] Different boxed answers exist: {unique_boxed_answers} → executing comparison process")
    else:
        print(f"⚠️ [PICK] No valid boxed answers found → executing comparison process")
    
    # Initialize necessary variables
    answer_counts = [0] * len(answers)
    all_judgments = []
    valid_judgments = 0
    judgment_round = 0
    results = []  # Save results at checkpoints
    
    # Judgment loop
    while judgment_round < max_judgments:
        print(f"\n🔄 [PICK] Judgment {judgment_round + 1}/{max_judgments}: generating new summaries...")
        
        # Generate new answer summaries each time
        summary_list = []
        
        for i, answer in enumerate(answers):
            answer_summary = None
            
            # Distribute GPUs based on i
            gpu_index = run_number % num_gpus
            target_port = base_port + gpu_index
            
            print(f"🔧 [PICK] Answer {i+1}: using GPU {gpu_index} (port {target_port})")
            
            # Generate summary (check cache first)
            try:
                # Check cached summary results
                cached_summary = open_discriminator_result(
                    "summary", evaluation_method, dataset_type, n, run_number, problem_index,
                    answer_index=i
                )
                
                if cached_summary:
                    print(f"📋 [PICK] Loaded summary for answer {i+1} from cache")
                    answer_summary = cached_summary['response']
                    token_count = cached_summary['num_tokens']
                    total_tokens += token_count
                    print(f"📊 [PICK] Judgment {judgment_round + 1} - Answer {i+1} summary (cache): {token_count} tokens")
                    print(f"answer_summary[{i}] = {answer_summary}")
                else:
                    # Generate new summary if not cached
                    print(f"🔧 [PICK] Generating new summary for answer {i+1}...")
                    
                    # Since generate_answer_client is called inside summarize_answer,
                    # call generate_answer_client directly to get token count
                    summary_prompt = f"""Please summarize the following text concisely. Elaborate the key mathematical steps and intermediate results (including inequalities), and the final answer with \\boxed{{}}, so that the reader can reproduce the solution.

Original text:
{answer}

Summary:"""
                    
                    # Get appropriate token count according to dataset
                    max_tokens = get_max_tokens_for_dataset(args.dataset_type, args.max_new_tokens)
                    summary_max_tokens = min(max_tokens // 2, SUMMARY_MAX_TOKENS)  # Summary should be half or less of the main text
                    
                    summary_results = generate_answer_client(
                        prompt=summary_prompt,
                        max_new_tokens=summary_max_tokens,
                        n=1,
                        temperature=TEMPERATURE,
                        port=target_port,  # Specify distributed port
                    )
                
                    if summary_results and len(summary_results) > 0:
                        answer_summary, token_count = summary_results[0]
                        total_tokens += token_count
                        print(f"📊 [PICK] Judgment {judgment_round + 1} - Answer {i+1} summary generated: {token_count} tokens (GPU {gpu_index})")
                        print(f"answer_summary[{i}] = {answer_summary}")
                        
                        # Save discriminator results
                        save_discriminator_result(
                            (answer_summary, token_count),
                            run_number,
                            n,
                            evaluation_method,
                            dataset_type,
                            compared_files,
                            judgment_round=judgment_round,
                            answer_index=i,
                            result_type="summary",
                            problem_index=problem_index
                        )
                    else:
                        answer_summary = f"Answer {i+1}: Failed to generate summary"
                        print(f"Summary generation error (answer {i}): Empty result")
            except Exception as e:
                print(f"Summary generation error (answer {i}): {str(e)}")
                answer_summary = f"Answer {i+1}: Failed to generate summary"
            
            # Handle both cases where summary was generated successfully and where it failed
            if answer_summary is not None:
                summary_list.append(answer_summary)
            else:
                summary_list.append(f"Answer {i+1}: Failed to generate summary")
        
        # Dynamically generate answer summaries
        summary_sections = []
        for i, summary in enumerate(summary_list):
            summary_sections.append(f"Answer {i+1} Summary:\n{summary}")
        
        summary_text = "\n\n".join(summary_sections)
        num_answers = len(summary_list)
        answer_options_list = [str(i+1) for i in range(num_answers)]
        answer_options_str = ", ".join(answer_options_list)
        
        judgment_prompt = f"""Please evaluate the following {num_answers} answer summaries for this mathematical problem and determine which answer you think is the most correct.

Problem:
{instruction}

{summary_text}

Among the above {num_answers} answer summaries, which answer do you think is the most correct, logical, and complete?

Please provide detailed reasoning for your judgment, and then output the number of the answer you think is correct ({answer_options_str}) enclosed in \\boxed{{}}.

Example: \\boxed{{1}}

Judgment:"""
        
        print(f"\nRunning judgment {judgment_round + 1}/{max_judgments}...")
        
        # GPU distribution for judgment (based on judgment round)
        judgment_gpu_index = judgment_round % num_gpus
        judgment_port = base_port + judgment_gpu_index
        print(f"🔧 [PICK] Judgment {judgment_round + 1}: using GPU {judgment_gpu_index} (port {judgment_port})")
        
        try:
            # Check cached judgment results
            cached_judgment = open_discriminator_result(
                "judgment", evaluation_method, dataset_type, n, run_number, problem_index,
                judgment_round=judgment_round
            )
            
            if cached_judgment:
                print(f"📋 [PICK] Loaded judgment {judgment_round + 1} from cache")
                judgment = cached_judgment['response']
                token_count = cached_judgment['num_tokens']
                total_tokens += token_count
                print(f"📊 [PICK] Judgment {judgment_round + 1} (cache): {token_count} tokens")
                print(f"🔍 [PICK] Judgment {judgment_round + 1} content: {judgment}")
                all_judgments.append(judgment)
                judgment_results = [(judgment, token_count)]
            else:
                # Generate new judgment if not cached
                print(f"🔧 [PICK] Running judgment {judgment_round + 1}...")
                
                # Execute judgment
                max_tokens = get_max_tokens_for_dataset(args.dataset_type, args.max_new_tokens)
                judgment_max_tokens = min(max_tokens // 3, 8000)  # Keep judgment shorter
                
                judgment_results = generate_answer_client(
                    prompt=judgment_prompt,
                    max_new_tokens=judgment_max_tokens,
                    n=1,
                    port=judgment_port,  # Specify distributed port
                )
            
            if judgment_results and len(judgment_results) > 0:
                # If loaded from cache, judgment and token_count are already set
                if not cached_judgment:
                    judgment, token_count = judgment_results[0]
                    total_tokens += token_count
                    print(f"📊 [PICK] Judgment {judgment_round + 1}: {token_count} tokens (GPU {judgment_gpu_index})")
                    print(f"🔍 [PICK] Judgment {judgment_round + 1} content: {judgment}")
                    all_judgments.append(judgment)
                    
                    # Save discriminator results (only for newly generated ones)
                    save_discriminator_result(
                        (judgment, token_count),
                        run_number,
                        n,
                        evaluation_method,
                        dataset_type,
                        compared_files,
                        judgment_round=judgment_round,
                        result_type="judgment",
                        problem_index=problem_index
                    )
                
                # Parse boxed answer
                try:
                    boxed_answer = extract_boxed_answer(judgment)
                    
                    # Check validity of answer number
                    if boxed_answer.strip() in answer_options_list:
                        selected_answer_num = int(boxed_answer.strip()) - 1
                        
                        # Range check
                        if 0 <= selected_answer_num < len(answer_counts):
                            answer_counts[selected_answer_num] += 1
                            valid_judgments += 1
                            print(f"  → select answer {selected_answer_num+1}")
                        else:
                            print(f"  → out-of-range answer index: {selected_answer_num}")
                    else:
                        print(f"  → invalid answer number: '{boxed_answer}'")
                        
                except Exception as parse_error:
                    print(f"  → parse error: {str(parse_error)}")
                
            else:
                print(f"  → failed to generate judgment result")
                
        except Exception as e:
            print(f"  → judgment processing error: {str(e)}")
        
        judgment_round += 1
        
        # If first_valid mode, stop at first valid judgment
        if first_valid and sum(answer_counts) > 0:
            current_best_index = np.argmax(answer_counts)
            print(f"🚀 [PICK] first_valid mode: stopping at first valid judgment - selecting answer {current_best_index+1}")
            # Return result with scale=-1
            return [(current_best_index, total_tokens)]
        
        # Record checkpoint results (only if not in first_valid mode)
        if not first_valid and judgment_round in checkpoints:
            if sum(answer_counts) > 0:
                current_best_index = np.argmax(answer_counts)
                print(f"📊 [PICK] Checkpoint {judgment_round}: answer {current_best_index+1} (selections: {answer_counts[current_best_index]}/{sum(answer_counts)})")
            else:
                current_best_index = 0
                print(f"📊 [PICK] Checkpoint {judgment_round}: no valid judgments; selecting answer 1")
            results.append((current_best_index, total_tokens))
        
        # Adaptive convergence check (only if adaptive is True)
        if adaptive and sum(answer_counts) >= 2:  # Need at least 2 judgments
            try:
                mus = [answer_counts[i] / sum(answer_counts) for i in range(len(answers))]
                
                if len(mus) >= 2:
                    sorted_mus = sorted(mus, reverse=True)
                    gap = sorted_mus[0] - sorted_mus[1]  # Max - 2nd largest
                    
                    threshold = np.sqrt(2 * np.log(1.0/delta) / sum(answer_counts))
                    
                    if gap >= threshold:
                        print(f"Adaptive convergence: gap={gap:.4f} >= threshold={threshold:.4f}")
                        break
                        
            except Exception as e:
                print(f"Adaptive convergence error: {str(e)}")
        elif not adaptive:
            print(f"Fixed-judgment mode: {judgment_round}/{max_judgments} completed")
    
    # Display GPU distribution summary
    total_summary_generations = judgment_round * len(answers)
    total_judgments = judgment_round
    print(f"\n🔧 [PICK] === GPU distribution summary ===")
    print(f"🔧 [PICK] Summaries generated: {total_summary_generations} (distributed to GPUs 0-{num_gpus-1})")
    print(f"🔧 [PICK] Judgments run: {total_judgments} (distributed to GPUs 0-{num_gpus-1})")
    print(f"🔧 [PICK] Port range: {base_port}-{base_port+num_gpus-1}")
        
    print(f"📊 [PICK] Total tokens used: {total_tokens}")
    print(f"📊 [PICK] Number of checkpoint results: {len(results)}")
    
    return results

 
 
    """
    Function to select the optimal answer from multiple answers (absolute certainty evaluation version)
    
    Parameters
    ----------
    instruction : str
        Problem statement
    answers : list
        List of answers
    delta : float
        Confidence interval parameter
    max_judgments : int
        Maximum number of judgments
        
    Returns
    -------
    tuple
        (Index of selected optimal answer, number of tokens used)
    """
    answers = answers_representative(answers)  # Remove answers with same values
    # pick_answer_abs is deprecated
    
    # Initialize token counter
    total_tokens = 0
    
    # GPU distribution settings
    num_gpus = int(os.getenv("NUM_GPUS", "1"))  # Get available GPU count from environment variable
    base_port = 8100  # Base port number
    print(f"🔧 [PICK_ABS] GPU distribution settings: {num_gpus} GPUs, base port {base_port}")

    if not answers:
        return -1, 0
    
    if len(answers) == 1:
        return 0, 0
    
    # Check boxed answer for each answer - no comparison needed if all are the same
    print(f"🔍 [PICK_ABS] Checking boxed answers for each answer...")
    boxed_answers = []
    for i, answer in enumerate(answers):
        try:
            boxed_answer = extract_boxed_answer(answer)
            if boxed_answer and boxed_answer.strip():
                boxed_answers.append(boxed_answer.strip())
                print(f"🔍 [PICK_ABS] Answer{i+1}: \\boxed{{{boxed_answer.strip()}}}")
            else:
                boxed_answers.append(None)
                print(f"🔍 [PICK_ABS] Answer{i+1}: No boxed answer")
        except Exception as e:
            boxed_answers.append(None)
            print(f"🔍 [PICK_ABS] Answer{i+1}: extract_boxed_answer error: {str(e)}")
    
    # Create set of valid boxed answers
    valid_boxed_answers = [ba for ba in boxed_answers if ba is not None]
    unique_boxed_answers = set(valid_boxed_answers)
    
    print(f"🔍 [PICK_ABS] Valid boxed answer count: {len(valid_boxed_answers)}, unique count: {len(unique_boxed_answers)}")
    
    if len(unique_boxed_answers) == 1:
        unique_answer = list(unique_boxed_answers)[0]
        print(f"✅ [PICK_ABS] All answers have same boxed answer: \\boxed{{{unique_answer}}} → selecting first answer")
        return 0, 0
    elif len(unique_boxed_answers) > 1:
        print(f"⚠️ [PICK_ABS] Different boxed answers exist: {unique_boxed_answers} → executing certainty evaluation")
    else:
        print(f"⚠️ [PICK_ABS] No valid boxed answers found → executing certainty evaluation")
    
    # List to hold only answers that obtained valid scores
    valid_answers = []
    valid_answer_indices = []
    
    # Check validity of each answer in the first round
    print(f"\n🔄 [PICK_ABS] Initial evaluation: Checking validity of each answer...")
    for i, answer in enumerate(answers):
        print(f"\n--- 📊 [PICK_ABS] Initial validity check for answer{i+1} ---")
        
        # Distribute GPU according to i
        gpu_index = i % num_gpus
        target_port = base_port + gpu_index
        
        print(f"🔧 [PICK_ABS] Answer{i+1}: Using GPU {gpu_index} (port {target_port})")
        
        # Certainty evaluation prompt
        evaluation_prompt = f"""Please evaluate the correctness of the following mathematical solution and provide a confidence score.

Problem:
{instruction}

Solution to evaluate:
{answer}

Please carefully analyze this solution and evaluate:
1. Are the mathematical steps correct and logical?
2. Are the calculations accurate?
3. Is the final answer properly formatted and reasonable?
4. Is the overall approach appropriate for this problem?

Based on your analysis, please provide a confidence score between 0 and 1, where:
- 0.0 means completely incorrect or nonsensical
- 0.5 means partially correct but with significant issues
- 1.0 means completely correct and well-reasoned

Please provide your reasoning step by step, then output only the numerical confidence score (between 0.0 and 1.0) enclosed in \\boxed{{}}.

Example: \\boxed{{0.85}}

Evaluation:"""
        
        max_retries = 3
        retry_count = 0
        evaluation_score = None
        
        while retry_count < max_retries and evaluation_score is None:
            try:
                # Execute certainty evaluation
                max_tokens = get_max_tokens_for_dataset(args.dataset_type, args.max_new_tokens)
                evaluation_max_tokens = max(max_tokens // 4, 5000)  # Keep certainty evaluation shorter
                
                evaluation_results = generate_answer_client(
                    prompt=evaluation_prompt,
                    max_new_tokens=evaluation_max_tokens,
                    n=1,
                    temperature=TEMPERATURE,
                    port=target_port,  # Specify distributed port
                )
                
                if evaluation_results and len(evaluation_results) > 0:
                    evaluation_response, token_count = evaluation_results[0]
                    total_tokens += token_count
                    print(f"📊 [PICK_ABS] Initial evaluation - Answer{i+1}: {token_count} tokens (GPU {gpu_index})")
                    print(f"Certainty evaluation response[{i}] = {evaluation_response}")
                    
                    # Parse boxed answer
                    try:
                        boxed_answer = extract_boxed_answer(evaluation_response)
                        evaluation_score = float(boxed_answer.strip())
                        
                        # 0-1 range check
                        if 0.0 <= evaluation_score <= 1.0:
                            print(f"✅ [PICK_ABS] Answer{i+1}: Initial valid score = {evaluation_score:.4f}")
                            valid_answers.append(answer)
                            valid_answer_indices.append(i)
                            break  # Successfully completed, exiting retry loop
                        else:
                            print(f"❌ [PICK_ABS] Answer{i+1}: Score out of range ({evaluation_score}) - retry {retry_count + 1}/{max_retries}")
                            evaluation_score = None
                            
                    except (ValueError, AttributeError) as parse_error:
                        print(f"❌ [PICK_ABS] Answer{i+1}: Parse error '{boxed_answer}' - retry {retry_count + 1}/{max_retries}")
                        evaluation_score = None
                        
                else:
                    print(f"❌ [PICK_ABS] Answer{i+1}: Failed to generate certainty evaluation result - retry {retry_count + 1}/{max_retries}")
                    
            except Exception as e:
                print(f"❌ [PICK_ABS] Answer{i+1}: Certainty evaluation error: {str(e)} - retry {retry_count + 1}/{max_retries}")
            
            retry_count += 1
        
        if evaluation_score is None:
            print(f"❌ [PICK_ABS] Answer{i+1}: {max_retries}times retry but still could not get valid score → excluded")
    
    print(f"\n📊 [PICK_ABS] Initial evaluation results: {len(valid_answers)}/{len(answers)}answers are valid")
    
    # Return first answer if no valid answers
    if not valid_answers:
        print(f"❌ [PICK_ABS] No valid answers found. Returning the first answer.")
        print(f"📊 [PICK_ABS] Total tokens used: {total_tokens}")
        return 0, total_tokens
    
    # Return it if only one valid answer
    if len(valid_answers) == 1:
        print(f"✅ [PICK_ABS] Only one valid answer found. Selecting it.")
        print(f"📊 [PICK_ABS] Total tokens used: {total_tokens}")
        return valid_answer_indices[0], total_tokens
    
    # If multiple valid answers, continue certainty evaluation among them
    print(f"🔄 [PICK_ABS] {len(valid_answers)}valid answers continuing with certainty evaluation...")
    
    # Record cumulative scores and judgment counts for valid answers
    answer_scores_sum = [0.0] * len(valid_answers)
    answer_judgments_count = [0] * len(valid_answers)
    judgment_round = 0
    
    # Judgment loop
    while judgment_round < max_judgments:
        print(f"\n🔄 [PICK_ABS] Judgment {judgment_round + 1}/{max_judgments}: Evaluating certainty of valid answers...")
        
        round_valid_judgments = 0
        
        # Evaluate certainty for each valid answer
        for i, answer in enumerate(valid_answers):
            print(f"\n--- 📊 [PICK_ABS] Certainty evaluation of valid answer ---")
            
            # Distribute GPU based on i
            gpu_index = (judgment_round * len(valid_answers) + i) % num_gpus
            target_port = base_port + gpu_index
            
            print(f"🔧 [PICK_ABS] Valid Answer{i+1}: Using GPU {gpu_index} (port {target_port})")
            
            # Certainty evaluation prompt
            evaluation_prompt = f"""Please evaluate the correctness of the following mathematical solution and provide a confidence score.

Problem:
{instruction}

Solution to evaluate:
{answer}

Please carefully analyze this solution and evaluate:
1. Are the mathematical steps correct and logical?
2. Are the calculations accurate?
3. Is the final answer properly formatted and reasonable?
4. Is the overall approach appropriate for this problem?

Based on your analysis, please provide a confidence score between 0 and 1, where:
- 0.0 means completely incorrect or nonsensical
- 0.5 means partially correct but with significant issues
- 1.0 means completely correct and well-reasoned

Please provide your reasoning step by step, then output only the numerical confidence score (between 0.0 and 1.0) enclosed in \\boxed{{}}.

Example: \\boxed{{0.85}}

Evaluation:"""
            
            max_retries = 5
            retry_count = 0
            evaluation_score = None
            
            while retry_count < max_retries and evaluation_score is None:
                try:
                    # Execute certainty evaluation
                    max_tokens = get_max_tokens_for_dataset(args.dataset_type, args.max_new_tokens)
                    evaluation_max_tokens = max(max_tokens // 3, 5000)  # Certainty evaluation is shorter
                    
                    evaluation_results = generate_answer_client(
                        prompt=evaluation_prompt,
                        max_new_tokens=evaluation_max_tokens,
                        n=1,
                        temperature=TEMPERATURE,
                        port=target_port,  # Specify distributed port
                    )
                    
                    if evaluation_results and len(evaluation_results) > 0:
                        evaluation_response, token_count = evaluation_results[0]
                        total_tokens += token_count
                        print(f"📊 [PICK_ABS] Judgment{judgment_round + 1} - Certainty evaluation of valid answer: {token_count} tokens (GPU {gpu_index})")
                        print(f"Certainty evaluation response[{i}] = {evaluation_response}")
                        
                        # Parse boxed answer
                        try:
                            boxed_answer = extract_boxed_answer(evaluation_response)
                            evaluation_score = float(boxed_answer.strip())
                            
                            # 0-1 range check
                            if 0.0 <= evaluation_score <= 1.0:
                                print(f"✅ [PICK_ABS] Valid Answer{i+1}: Certainty score = {evaluation_score:.4f}")
                                answer_scores_sum[i] += evaluation_score
                                answer_judgments_count[i] += 1
                                round_valid_judgments += 1
                                break  # Successfully completed, exiting retry loop
                            else:
                                print(f"❌ [PICK_ABS] Valid Answer{i+1}: Score out of range ({evaluation_score}) - retry {retry_count + 1}/{max_retries}")
                                evaluation_score = None
                                
                        except (ValueError, AttributeError) as parse_error:
                            print(f"❌ [PICK_ABS] Valid Answer{i+1}: Parse error '{boxed_answer}' - retry {retry_count + 1}/{max_retries}")
                            evaluation_score = None
                            
                    else:
                        print(f"❌ [PICK_ABS] Valid Answer{i+1}: Failed to generate certainty evaluation result - retry {retry_count + 1}/{max_retries}")
                        
                except Exception as e:
                    print(f"❌ [PICK_ABS] Valid Answer{i+1}: Certainty evaluation error: {str(e)} - retry {retry_count + 1}/{max_retries}")
                
                retry_count += 1
            
            if evaluation_score is None:
                print(f"❌ [PICK_ABS] Valid Answer{i+1}: {max_retries} times retry but still could not get valid score")
        
        print(f"🔄 [PICK_ABS] Judgment{judgment_round + 1} completed: {round_valid_judgments}/{len(valid_answers)} valid evaluations")
        
        judgment_round += 1
        
        # Termination judgment (only if sufficient judgments have been executed)
        if judgment_round >= 2:  # At least 2 rounds of judgment required
            try:
                # Calculate average score for each valid answer
                average_scores = []
                for i in range(len(valid_answers)):
                    if answer_judgments_count[i] > 0:
                        avg_score = answer_scores_sum[i] / answer_judgments_count[i]
                        average_scores.append(avg_score)
                        print(f"🔍 [PICK_ABS] Valid Answer{i+1}: Average score = {avg_score:.4f} ({answer_judgments_count[i]} evaluations)")
                    else:
                        average_scores.append(0.0)
                        print(f"🔍 [PICK_ABS] Valid Answer{i+1}: No evaluation")
                
                if len(average_scores) >= 2 and max(answer_judgments_count) >= 2:
                    # Sort and calculate gap between top 2
                    sorted_scores = sorted(average_scores, reverse=True)
                    gap = sorted_scores[0] - sorted_scores[1]  # Maximum value - 2nd largest value
                    theta = sorted_scores[0] 
                    theta = max(min(theta, 0.99), 0.01)
                    
                    # Calculate confidence interval threshold (using maximum evaluation count)
                    max_judgments_so_far = max(answer_judgments_count)
                    threshold = np.sqrt( (theta * (1-theta) * np.log(1.0/delta)) / (2 * max_judgments_so_far))
                    
                    print(f"🔍 [PICK_ABS] Convergence check: gap={gap:.4f}, threshold={threshold:.4f}")
                    
                    if gap >= threshold:
                        print(f"✅ [PICK_ABS] Convergence judgment: gap={gap:.4f} >= threshold={threshold:.4f}")
                        break
                        
            except Exception as e:
                print(f"❌ [PICK_ABS] Termination judgment error: {str(e)}")
    
    # Display GPU distribution summary
    total_evaluations = judgment_round * len(valid_answers)
    print(f"\n🔧 [PICK_ABS] === GPU Distribution Summary ===")
    print(f"🔧 [PICK_ABS] Certainty evaluations: {total_evaluations} (distributed to: GPU 0-{num_gpus-1})")
    print(f"🔧 [PICK_ABS] Evaluation rounds: {judgment_round}")
    print(f"🔧 [PICK_ABS] Port range used: {base_port}-{base_port+num_gpus-1}")
    print(f"🔧 [PICK_ABS] Valid answers: {len(valid_answers)}/{len(answers)}")
    
    # Finally return the valid answer with the highest average score
    if any(count > 0 for count in answer_judgments_count):
        average_scores = []
        for i in range(len(valid_answers)):
            if answer_judgments_count[i] > 0:
                avg_score = answer_scores_sum[i] / answer_judgments_count[i]
                average_scores.append(avg_score)
            else:
                average_scores.append(0.0)
        
        best_index = np.argmax(average_scores)
        best_avg_score = average_scores[best_index]
        
        print(f"✅ [PICK_ABS] Final selection: Valid Answer{best_index+1} (average score: {best_avg_score:.4f}, {answer_judgments_count[best_index]} evaluations)")
        print(f"📊 [PICK_ABS] Total tokens used: {total_tokens}")
        return valid_answer_indices[best_index], total_tokens
    else:
        print("❌ [PICK_ABS] No valid evaluations were found. Returning the first valid answer.")
        print(f"📊 [PICK_ABS] Total tokens used: {total_tokens}")
        return valid_answer_indices[0] if valid_answer_indices else 0, total_tokens

 
    """
    Prioritize majority voting, and if there's a tie, leave one representative from each of the most common answers and select through tournament.
    
    1. First execute majority voting
    2. If there's a single winner, return it
    3. In case of tie, leave one representative from each of the most common answers
    4. Conduct a tournament similar to llm_judge_tournament with those representatives

    Returns
    -------
    (picked_index_in_original, total_tokens)
    """
    # majority_tournament is deprecated

    if not answers:
        return -1, 0
    if len(answers) == 1:
        return 0, 0

    # Step 1: Majority voting
    print(f"🗳️ [MAJ_TOURN] Step 1: Executing majority voting...")
    
    # Extract answers from each Answer
    extracted_answers = []
    for i, answer in enumerate(answers):
        try:
            extracted_answer = extract_boxed_answer(answer)
            if extracted_answer and extracted_answer.strip():
                extracted_answers.append(extracted_answer.strip())
            else:
                # If boxed answer is not found, treat as empty string
                extracted_answers.append("")
        except Exception as e:
            print(f"⚠️ [MAJ_TOURN] Error extracting Answer{i+1}: {str(e)}")
            extracted_answers.append("")
    
    # Count answers
    from collections import Counter
    answer_counts = Counter()
    answer_to_indices = {}
    
    for i, extracted in enumerate(extracted_answers):
        if extracted and extracted.strip():  # Count only if not empty
            if extracted not in answer_to_indices:
                answer_to_indices[extracted] = []
            answer_to_indices[extracted].append(i)
            answer_counts[extracted] += 1
    
    if not answer_counts:
        print(f"⚠️ [MAJ_TOURN] Could not extract valid answers - selecting first Answer")
        return 0, 0
    
    # Get the maximum vote count
    max_count = answer_counts.most_common(1)[0][1]
    most_common_answers = [answer for answer, count in answer_counts.items() if count == max_count]
    
    print(f"🗳️ [MAJ_TOURN] Majority voting results:")
    for answer, count in answer_counts.most_common():
        print(f"  '{answer}': {count} votes")
    
    # Step 2: Check if there's a single winner
    if len(most_common_answers) == 1:
        # Single winner
        winning_answer = most_common_answers[0]
        winning_indices = answer_to_indices[winning_answer]
        selected_index = winning_indices[0]  # Select the first one that appeared
        print(f"🏆 [MAJ_TOURN] Single winner: '{winning_answer}' ({max_count} votes) - selecting Answer{selected_index+1}")
        return selected_index, 0
    
    # Step 3: In case of tie, select one representative from each most common answer for tournament
    print(f"🤝 [MAJ_TOURN] Tie detected: {len(most_common_answers)} answers tied with {max_count} votes")
    print(f"🎯 [MAJ_TOURN] Step 2: Executing tournament - selecting one representative from each answer...")
    
    # Select one representative from each most common answer
    tournament_candidates = []
    tournament_indices = []
    
    for winning_answer in most_common_answers:
        indices = answer_to_indices[winning_answer]
        representative_idx = indices[0]  # Use the first one that appeared as representative
        tournament_candidates.append(answers[representative_idx])
        tournament_indices.append(representative_idx)
        print(f"  Representative: '{winning_answer}' -> Answer{representative_idx+1}")
    
    # Step 4: Execute tournament
    if len(tournament_candidates) == 1:
        selected_index = tournament_indices[0]
        print(f"🏆 [MAJ_TOURN] Only one representative - selecting Answer{selected_index+1}")
        return selected_index, 0
    
    print(f"🎯 [MAJ_TOURN] Tournament start: tournament with {len(tournament_candidates)} representatives")
    
    # Tournament processing (using llm_judge_tournament logic)
    pool = [(a, idx) for a, idx in zip(tournament_candidates, tournament_indices)]
    total_tokens = 0
    round_id = 1
    
    while len(pool) >= L:
        print(f"[MAJ_TOURN] Round {round_id}: pool={len(pool)} -> groups of {L}")
        next_pool = []
        num_groups = len(pool) // L
        for g in range(num_groups):
            group = pool[g*L:(g+1)*L]
            group_answers = [a for a, _ in group]
            
            # majority_tournament is deprecated
            picked_local_idx, token_count = pick_results[-1]
            total_tokens += token_count
            
            if 0 <= picked_local_idx < len(group):
                winner = group[picked_local_idx]
                next_pool.append(winner)
                print(f"  Group {g+1}: winner = answer at original index {winner[1]}")
        
        # Carry over remainder to next round
        remainder = len(pool) % L
        if remainder > 0:
            for i in range(remainder):
                next_pool.append(pool[-(remainder-i)])
        
        pool = next_pool
        round_id += 1
    
    # Final round
    if len(pool) >= 2:
        print(f"[MAJ_TOURN] Final round: {len(pool)} candidates")
        final_answers = [a for a, _ in pool]
        # majority_tournament is deprecated
        final_local_idx, token_count = pick_results[-1]
        total_tokens += token_count
        
        if 0 <= final_local_idx < len(pool):
            final_winner_idx = pool[final_local_idx][1]
            print(f"🏆 [MAJ_TOURN] Final winner: answer at original index {final_winner_idx}")
            return final_winner_idx, total_tokens
    
    # Fallback
    if pool:
        fallback_idx = pool[0][1]
        print(f"🏆 [MAJ_TOURN] Fallback winner: answer at original index {fallback_idx}")
        return fallback_idx, total_tokens
    
    return 0, total_tokens

 
    """
    Prioritize majority voting, and if there's a tie, leave one representative from each of the most common answers and select through lastsummary tournament.
    
    1. First execute majority voting
    2. If there's a single winner, return it
    3. In case of tie, leave one representative from each of the most common answers
    4. Conduct a tournament similar to llm_judge_set with those representatives

    Returns
    -------
    (picked_index_in_original, total_tokens)
    """
    # majority_lastsummary is deprecated

    if not answers:
        return -1, 0
    if len(answers) == 1:
        return 0, 0

    # Step 1: Majority voting
    print(f"🗳️ [MAJ_LAST] Step 1: Executing majority voting...")
    
    # Extract answers from each Answer
    extracted_answers = []
    for i, answer in enumerate(answers):
        try:
            extracted_answer = extract_boxed_answer(answer)
            if extracted_answer and extracted_answer.strip():
                extracted_answers.append(extracted_answer.strip())
            else:
                # If boxed answer is not found, treat as empty string
                extracted_answers.append("")
        except Exception as e:
            print(f"⚠️ [MAJ_LAST] Error extracting Answer{i+1}: {str(e)}")
            extracted_answers.append("")
    
    # Count answers
    from collections import Counter
    answer_counts = Counter()
    answer_to_indices = {}
    
    for i, extracted in enumerate(extracted_answers):
        if extracted and extracted.strip():  # Count only if not empty
            if extracted not in answer_to_indices:
                answer_to_indices[extracted] = []
            answer_to_indices[extracted].append(i)
            answer_counts[extracted] += 1
    
    if not answer_counts:
        print(f"⚠️ [MAJ_LAST] Could not extract valid answers - selecting first Answer")
        return 0, 0
    
    # Get the maximum vote count
    max_count = answer_counts.most_common(1)[0][1]
    most_common_answers = [answer for answer, count in answer_counts.items() if count == max_count]
    
    print(f"🗳️ [MAJ_LAST] Majority voting results:")
    for answer, count in answer_counts.most_common():
        print(f"  '{answer}': {count} votes")
    
    # Step 2: Check if there's a single winner
    if len(most_common_answers) == 1:
        # Single winner
        winning_answer = most_common_answers[0]
        winning_indices = answer_to_indices[winning_answer]
        selected_index = winning_indices[0]  # Select the first one that appeared
        print(f"🏆 [MAJ_LAST] Single winner: '{winning_answer}' ({max_count} votes) - selecting Answer{selected_index+1}")
        return selected_index, 0
    
    # Step 3: In case of tie, select one representative from each most common answer for lastsummary tournament
    print(f"🤝 [MAJ_LAST] Tie detected: {len(most_common_answers)} answers tied with {max_count} votes")
    print(f"📄 [MAJ_LAST] Step 2: Executing Lastsummary Tournament - selecting one representative from each answer...")
    
    # Select one representative from each most common answer
    tournament_candidates = []
    tournament_indices = []
    
    for winning_answer in most_common_answers:
        indices = answer_to_indices[winning_answer]
        representative_idx = indices[0]  # Use the first one that appeared as representative
        tournament_candidates.append(answers[representative_idx])
        tournament_indices.append(representative_idx)
        print(f"  Representative: '{winning_answer}' -> Answer{representative_idx+1}")
    
    # Step 4: Execute Lastsummary tournament
    if len(tournament_candidates) == 1:
        selected_index = tournament_indices[0]
        print(f"🏆 [MAJ_LAST] Only one representative - selecting Answer{selected_index+1}")
        return selected_index, 0
    
    print(f"📄 [MAJ_LAST] Lastsummary Tournament start: tournament with {len(tournament_candidates)} representatives")
    
    # Lastsummary tournament processing (using llm_judge_set logic)
    pool = [(a, idx) for a, idx in zip(tournament_candidates, tournament_indices)]
    total_tokens = 0
    round_id = 1
    
    while len(pool) >= L:
        print(f"[MAJ_LAST] Round {round_id}: pool={len(pool)} -> groups of {L}")
        next_pool = []
        num_groups = len(pool) // L
        for g in range(num_groups):
            group = pool[g*L:(g+1)*L]
            group_answers = [a for a, _ in group]
            
            # majority_lastsummary is deprecated
            picked_local_idx, token_count = pick_results[-1]
            total_tokens += token_count
            
            if 0 <= picked_local_idx < len(group):
                winner = group[picked_local_idx]
                next_pool.append(winner)
                print(f"  Group {g+1}: winner = answer at original index {winner[1]}")
        
        # Carry over remainder to next round
        remainder = len(pool) % L
        if remainder > 0:
            for i in range(remainder):
                next_pool.append(pool[-(remainder-i)])
        
        pool = next_pool
        round_id += 1
    
    # Final round
    if len(pool) >= 2:
        print(f"[MAJ_LAST] Final round: {len(pool)} candidates")
        final_answers = [a for a, _ in pool]
        # majority_lastsummary is deprecated
        final_local_idx, token_count = pick_results[-1]
        total_tokens += token_count
        
        if 0 <= final_local_idx < len(pool):
            final_winner_idx = pool[final_local_idx][1]
            print(f"🏆 [MAJ_LAST] Final winner: answer at original index {final_winner_idx}")
            return final_winner_idx, total_tokens
    
    # Fallback
    if pool:
        fallback_idx = pool[0][1]
        print(f"🏆 [MAJ_LAST] Fallback winner: answer at original index {fallback_idx}")
        return fallback_idx, total_tokens
    
    return 0, total_tokens

def llm_judge_tournament(instruction, answers, run_number, L=2, delta=0.1, n=None, dataset_type="unknown", compared_files=None, problem_index=None):
    """
    Narrow down to the optimal solution using tournament method (L-division).

    - While the number of candidates is L or more, group them into groups of L and execute pick_answer for each group
    - Leave one winner from the last checkpoint for each group
    - Remainder (<L) is carried over to the next round as is
    - If 2..L-1 candidates remain at the end, narrow down to one with pick_answer

    Returns
    -------
    (picked_index_in_original, total_tokens)
    """
    print(f"\n🎯 [TOURN] llm_judge_tournament: L={L}, num_answers={len(answers)}")

    if not answers:
        return -1, 0
    if len(answers) == 1:
        return 0, 0

    # Remove duplicates (representative answers only) and record original indices
    deduped_answers = answers_representative(answers)
    index_map = []
    for a in deduped_answers:
        try:
            index_map.append(answers.index(a))
        except ValueError:
            index_map.append(0)

    pool = [(a, idx) for a, idx in zip(deduped_answers, index_map)]
    total_tokens = 0
    round_id = 1

    while len(pool) >= L:
        print(f"[TOURN] Round {round_id}: pool={len(pool)} -> groups of {L}")
        next_pool = []
        num_groups = len(pool) // L
        for g in range(num_groups):
            group = pool[g*L:(g+1)*L]
            group_answers = [a for a, _ in group]
            # Remove duplicates within group and maintain correspondence to original local indices (unnecessary?)
            dedup_group_answers = answers_representative(group_answers)
            dedup_to_orig_local = []
            for ans in dedup_group_answers:
                try:
                    dedup_to_orig_local.append(group_answers.index(ans))
                except ValueError:
                    dedup_to_orig_local.append(0)

            pick_results = llm_judge_set(instruction, dedup_group_answers, run_number, delta=delta, first_valid=args.first_valid, n=n, evaluation_method="llm_judge_tournament", compared_files=compared_files, dataset_type=dataset_type, problem_index=problem_index)
            picked_local_idx, token_count = pick_results[-1]
            total_tokens += token_count
            if 0 <= picked_local_idx < len(dedup_group_answers):
                # dedup local -> original local
                orig_local = dedup_to_orig_local[picked_local_idx]
                next_pool.append(group[orig_local])
            else: # This shouldn't normally happen?
                print(f"❌ [TOURN] Failed to select from group?")
                next_pool.append(group[0])
        # Carry over remainder as is
        remainder = pool[num_groups*L:]
        if remainder:
            print(f"[TOURN] carry remainder: {len(remainder)}")
            next_pool.extend(remainder)
        pool = next_pool
        round_id += 1

    # Final narrowing down when 2..L-1 candidates remain
    if len(pool) == 0:
        print(f"❌ [TOURN] 0 candidates remaining from groups (abnormal behavior)")
        return 0, total_tokens
    if len(pool) == 1:
        return pool[0][1], total_tokens
    final_answers = [a for a, _ in pool]
    dedup_final_answers = answers_representative(final_answers)
    dedup_to_orig_final = []
    for ans in dedup_final_answers:
        try:
            dedup_to_orig_final.append(final_answers.index(ans))
        except ValueError:
            dedup_to_orig_final.append(0)
    final_results = llm_judge_set(instruction, dedup_final_answers, run_number, delta=delta, first_valid=args.first_valid, n=n, evaluation_method="llm_judge_tournament", compared_files=compared_files, dataset_type=dataset_type, problem_index=problem_index)
    final_local_idx, token_count = final_results[-1]
    total_tokens += token_count
    if 0 <= final_local_idx < len(dedup_final_answers):
        orig_local = dedup_to_orig_final[final_local_idx]
        return pool[orig_local][1], total_tokens
    return pool[0][1], total_tokens

## Removed: pick_answer_tournament_norep (deprecated)

def generate_answerBoN(instruction, prompt, N, gold, problem_index, run_number, direct_answer = True, use_grammar=True, evaluation_method = "omni", dataset_type="unknown"): #BoN answer generation based on existing saved files
    """
    BoN answer generation based on existing saved files
    
    Parameters
    ----------
    instruction : str
        Problem statement
    prompt : str
        Prompt
    N : int
        Number of Answers to load
    gold : str
        Correct answer
    problem_index : int
        Problem number
    run_number : int
        Run number (used for Answer number calculation)
    direct_answer : bool
        Direct answer mode
    use_grammar : bool
        Whether to use grammar constraints
    evaluation_method : str
        Evaluation method ("omni": prioritize correct answer, "pick_answer": LLM judgment, "reward": reward model, "self_certainty": Self-Certainty score)
        
    Returns
    -------
    list
        List of (selected Answer, tokens used, filename list, selected file index) at each checkpoint
        Multiple elements if evaluation_method == "pick_answer", otherwise a list with one element
    """
    print(f"\n🚀 [BON] ===== generate_answerBoN start =====")
    print(f"🚀 [BON] evaluation_method = {evaluation_method}")
    print(f"🚀 [BON] Loading from existing Answer files (N={N})")
    
    # Get dataset type
    dataset_type = args.dataset_type
    
    print(f"📂 [BON] Dataset: {dataset_type}, Problem number: {problem_index}")
    print(f"📂 [BON] Source directory: {EXISTING_ANSWERS_DIR}")
    print(f"📂 [BON] Run number: {run_number}, Answer range: {run_number * N} ~ {(run_number + 1) * N - 1}")

    answers = []
    scores = []
    filenames = []
    
    # Load N Answers from existing files
    print(f"\n📚 [BON] Starting sequential loading of {N} Answers...")
    for i in range(N):
        # Explicitly calculate Answer number: run_number * N + i
        answer_index = run_number * N + i
        print(f"\n--- 📖 [BON] Loading Answer {i+1}/{N} (answer{answer_index}) ---")
        
        answer, filename = load_existing_answer(dataset_type, problem_index, EXISTING_ANSWERS_DIR, answer_index)
        
        if answer:
            answers.append(answer)
            scores.append(0)  # Set score to 0
            filenames.append(filename)
            print(f"✅ [BON] Successfully loaded Answer{i+1} (file: {filename})")
        else:
            print(f"❌ [BON] Failed to load Answer{i+1} (file: {filename if filename else 'N/A'})")
            # Add empty Answer if Answer is not found
            answers.append("")
            scores.append(0)
            filenames.append(filename if filename else "")
    
    print(f"\n📊 [BON] === Loading Results Summary ===")
    print(f"📊 [BON] Total loading attempts: {len(answers)} items")
    print(f"📊 [BON] Filename records: {len(filenames)} items")
    
    if not answers:
        print(f"❌ [BON] No Answers found")
        return [("", 0, [], -1)]
    
    if evaluation_method == "pick_answer" and len(answers) > 1:
        print(f"\n🎯 [BON] Using pick_answer function to select optimal Answer...")
        correct_exists = False
        for answer in answers:
            if is_correct(answer, gold, dataset_type):
                correct_exists = True
        pick_results = pick_answer(instruction, answers, run_number, delta = 0.1, first_valid=args.first_valid, n=N, evaluation_method=evaluation_method, compared_files=filenames, dataset_type=dataset_type, problem_index=problem_index)
        
        # Build results for each checkpoint
        results_list = []
        for checkpoint_idx, (picked_index, picked_token_count) in enumerate(pick_results):
            if picked_index >= 0 and picked_index < len(answers):
                picked_answer = answers[picked_index]
            else:
                picked_answer = answers[0] if answers else ""
            results_list.append((picked_answer, picked_token_count, filenames, picked_index))
            print(f"📊 [BON] Checkpoint{checkpoint_idx+1}: Selected Answer{picked_index+1} (tokens: {picked_token_count})")
        
        # Verification with final results
        final_picked_answer = results_list[-1][0]
        if correct_exists and not is_correct(final_picked_answer, gold, dataset_type):
            print(f"⚠️ [BON] correct answer exists but picked answer is incorrect")
        elif len(answers_representative(answers)) > 1 and is_correct(final_picked_answer, gold, dataset_type):
            print(f"⚠️ [BON] picked the correct answer from multiple answers")
        print(f"✅ [BON] pick_answer completed")
        print(f"📊 [BON] Number of checkpoints: {len(results_list)}")
        return results_list
    elif evaluation_method == "llm_judge_set" and len(answers) > 1:
        print(f"\n🎯 [BON] Using llm_judge_set function to select optimal Answer...")
        correct_exists = False
        for answer in answers:
            if is_correct(answer, gold, dataset_type):
                correct_exists = True
        pick_results = llm_judge_set(instruction, answers, run_number, delta = 0.1, first_valid=args.first_valid, n=N, evaluation_method=evaluation_method, compared_files=filenames, dataset_type=dataset_type, problem_index=problem_index)
        
        # Build results for each checkpoint
        results_list = []
        for checkpoint_idx, (picked_index, picked_token_count) in enumerate(pick_results):
            if picked_index >= 0 and picked_index < len(answers):
                picked_answer = answers[picked_index]
            else:
                picked_answer = answers[0] if answers else ""
            results_list.append((picked_answer, picked_token_count, filenames, picked_index))
            print(f"📊 [BON] Checkpoint{checkpoint_idx+1}: Selected Answer{picked_index+1} (tokens: {picked_token_count})")
        
        # Verification with final results
        final_picked_answer = results_list[-1][0]
        if correct_exists and not is_correct(final_picked_answer, gold, dataset_type):
            print(f"⚠️ [BON] correct answer exists but picked answer is incorrect")
        elif len(answers_representative(answers)) > 1 and is_correct(final_picked_answer, gold, dataset_type):
            print(f"⚠️ [BON] picked the correct answer from multiple answers")
        print(f"✅ [BON] llm_judge_set completed")
        print(f"📊 [BON] Number of checkpoints: {len(results_list)}")
        return results_list
    

    elif evaluation_method == "llm_judge_tournament" and len(answers) > 1:
        print(f"\n🎯 [BON] Using llm_judge_tournament function to select optimal Answer...")
        correct_exists = False
        for answer in answers:
            if is_correct(answer, gold, dataset_type):
                correct_exists = True
        L = int(os.getenv("TOURNAMENT_L", str(getattr(args, 'tournament_L', 2))))
        print(f"🎯 [BON] L={L}")
        picked_index, picked_token_count = llm_judge_tournament(instruction, answers, run_number, L=L, delta=0.1, n=N, dataset_type=dataset_type, compared_files=filenames, problem_index=problem_index)
        if picked_index >= 0 and picked_index < len(answers):
            picked_answer = answers[picked_index]
        else:
            picked_answer = answers[0] if answers else ""
        if correct_exists and not is_correct(picked_answer, gold, dataset_type):
            print(f"⚠️ [BON] correct answer exists but tournament picked incorrect answer")
        elif len(answers_representative(answers)) > 1 and is_correct(picked_answer, gold, dataset_type):
            print(f"⚠️ [BON] picked the correct answer from multiple answers")
        print(f"✅ [BON] llm_judge_tournament completed")
        print(f"📊 [BON] Total tokens used: {picked_token_count}")
        return [(picked_answer, picked_token_count, filenames, picked_index)]
    # removed: pick_answer_tournament_norep branch (deprecated)
    elif evaluation_method == "reward" and len(answers) > 1:
        print(f"\n🏆 [BON] Using reward model to select optimal Answer...")
        print(f"🏆 [BON] Scoring {len(answers)} Answers...")
        
        # Score each Answer
        answer_scores = []
        for i, answer in enumerate(answers):
            try:
                score = get_score(instruction, answer)
                answer_scores.append((score, i, answer))
                print(f"🏆 [BON] Answer{i+1}: Score = {score:.4f}")
            except Exception as e:
                print(f"❌ [BON] Score calculation error for Answer{i+1}: {str(e)}")
                answer_scores.append((0.0, i, answer))  # 0 score on error
        
        # Sort by score (descending)
        answer_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select Answer with highest score
        best_score, best_index, best_answer = answer_scores[0]
        print(f"🏆 [BON] Highest score: {best_score:.4f} (Answer{best_index+1})")
        
        # Check if there's a correct answer
        correct_exists = False
        for answer in answers:
            if is_correct(answer, gold, dataset_type):
                correct_exists = True
        
        if correct_exists and not is_correct(best_answer, gold, dataset_type):
            print(f"⚠️ [BON] correct answer exists but reward model picked incorrect answer")
        elif len(answers_representative(answers)) > 1 and is_correct(best_answer, gold, dataset_type):
            print(f"⚠️ [BON] picked the correct answer from multiple answers")
            
        print(f"✅ [BON] Reward model selection completed")
        print(f"📊 [BON] Total tokens used: 0 (reward model scoring)")
        return [(best_answer, 0, filenames, best_index)]
    elif evaluation_method == "reward_orm" and len(answers) > 1:
        print(f"\n🏆 [BON] Using reward model (ORM) to select optimal Answer...")
        print(f"🏆 [BON] Scoring {len(answers)} Answers (removing before </think>)...")
        
        def remove_thinking_prefix(answer):
            """Remove the part before </think> from the answer"""
            if "</think>" in answer:
                return answer.split("</think>", 1)[1]
            return answer
        
        # Score each Answer (after removing before </think>)
        answer_scores = []
        for i, answer in enumerate(answers):
            try:
                # Remove before </think>
                answer_after_think = remove_thinking_prefix(answer)
                print("length of answer_after_think = ", len(answer_after_think))
                score = get_score(instruction, answer_after_think)
                answer_scores.append((score, i, answer))  # Keep original answer
                print(f"🏆 [BON] Answer{i+1}: Score = {score:.4f}")
            except Exception as e:
                print(f"❌ [BON] Score calculation error for Answer{i+1}: {str(e)}")
                answer_scores.append((0.0, i, answer))  # 0 score on error
        
        # Sort by score (descending)
        answer_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select Answer with highest score
        best_score, best_index, best_answer = answer_scores[0]
        print(f"🏆 [BON] Highest score: {best_score:.4f} (Answer{best_index+1})")
        
        # Check if there's a correct answer
        correct_exists = False
        for answer in answers:
            if is_correct(answer, gold, dataset_type):
                correct_exists = True
        
        if correct_exists and not is_correct(best_answer, gold, dataset_type):
            print(f"⚠️ [BON] correct answer exists but reward model picked incorrect answer")
        elif len(answers_representative(answers)) > 1 and is_correct(best_answer, gold, dataset_type):
            print(f"⚠️ [BON] picked the correct answer from multiple answers")
            
        print(f"✅ [BON] Reward model (ORM) selection completed")
        print(f"📊 [BON] Total tokens used: 0 (reward model scoring)")
        return [(best_answer, 0, filenames, best_index)]
    # removed: comp_reward branch (deprecated)
    elif evaluation_method == "self_certainty" and len(answers) > 1:
        print(f"\n🎯 [BON] Using self_certainty evaluation to select optimal Answer...")
        print(f"🎯 [BON] {len(answers)} Answer files extracting Self-Certainty scores...")
        
        # Extract Self-Certainty scores from each Answer file
        answer_scores = []
        for i, filename in enumerate(filenames):
            if filename and filename != "":
                # Build complete file path
                full_path = os.path.join(EXISTING_ANSWERS_DIR, filename)
                score = extract_self_certainty_from_file(full_path)
                answer_scores.append((score, i, answers[i]))
                print(f"🎯 [BON] Answer{i+1} ({filename}): Self-Certainty = {score:.6f}")
            else:
                # 0 score if no filename
                answer_scores.append((0.0, i, answers[i]))
                print(f"🎯 [BON] Answer{i+1}: No filename, Self-Certainty = 0.0")
        
        # Sort by score (descending)
        answer_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select Answer with highest score
        best_score, best_index, best_answer = answer_scores[0]
        print(f"🎯 [BON] Highest Self-Certainty score: {best_score:.6f} (Answer{best_index+1})")
        
        # Check if there is a correct answer
        correct_exists = False
        for answer in answers:
            if is_correct(answer, gold, dataset_type):
                correct_exists = True
        
        if correct_exists and not is_correct(best_answer, gold, dataset_type):
            print(f"⚠️ [BON] correct answer exists but self_certainty picked incorrect answer")
        elif len(answers_representative(answers)) > 1 and is_correct(best_answer, gold, dataset_type):
            print(f"⚠️ [BON] picked the correct answer from multiple answers")
            
        print(f"✅ [BON] Self-Certainty evaluation completed")
        print(f"📊 [BON] Total tokens used: 0 (self-certainty scoring)")
        return [(best_answer, 0, filenames, best_index)]
    elif evaluation_method == "random":
        print(f"\n🎲 [BON] random evaluation: uniform random selection from Answers...")
        # Uniformly randomly select Answer
        import random as rand
        selected_answer = rand.choice(answers)
        selected_index = answers.index(selected_answer)
        print(f"🎲 [BON] Random selection: Answer")
        print(f"📊 [BON] Total tokens used: 0 (random selection)")
        return [(selected_answer, 0, filenames, selected_index)]
    elif evaluation_method == "majority" and len(answers) > 1:
        print(f"\n🗳️ [BON] majority evaluation: select by majority vote from Answers...")
        
        # Extract answers from each Answer
        extracted_answers = []
        for i, answer in enumerate(answers):
            try:
                extracted_answer = extract_boxed_answer(answer)
                if extracted_answer and extracted_answer.strip() and extracted_answer != "Could not extract answer":
                    extracted_answers.append(extracted_answer.strip())
                    print(f"🔍 [BON] Answer{i+1}: {extracted_answer.strip()}")
                else:
                    # Try other patterns as fallback
                    if "Final Answer:" in answer:
                        fallback = answer.split("Final Answer:")[-1].strip()
                        # Extract only the first numeric part
                        import re
                        numbers = re.findall(r'\d+(?:\.\d+)?', fallback)
                        if numbers:
                            extracted_answers.append(numbers[0])
                            print(f"🔍 [BON] Answer{i+1}: {numbers[0]} (fallback)")
                        else:
                            extracted_answers.append("")
                            print(f"⚠️ [BON] Answer{i+1}: Could not extract answer")
                    else:
                        extracted_answers.append("")
                        print(f"⚠️ [BON] Answer{i+1}: Could not extract answer")
            except Exception as e:
                extracted_answers.append("")
                print(f"❌ [BON] Answer{i+1}: Extraction error: {str(e)}")
        
        # Count answer frequencies
        from collections import Counter
        answer_counts = Counter()
        answer_to_indices = {}  # Index list for each answer
        
        for i, extracted in enumerate(extracted_answers):
            if extracted:  # Count only non-empty answers
                answer_counts[extracted] += 1
                if extracted not in answer_to_indices:
                    answer_to_indices[extracted] = []
                answer_to_indices[extracted].append(i)
        
        if not answer_counts:
            print(f"⚠️ [BON] Could not extract valid answers - selecting first Answer")
            print(f"📊 [BON] Total tokens used: 0 (majority voting)")
            return [(answers[0], 0, filenames, 0)]
        
        # Get the maximum vote count
        max_count = answer_counts.most_common(1)[0][1]
        most_common_answers = [answer for answer, count in answer_counts.items() if count == max_count]
        
        print(f"📊 [BON] Voting results:")
        for answer, count in answer_counts.most_common():
            print(f"   {answer}: {count} votes")
        
        if len(most_common_answers) == 1:
            # If there is a clear winner
            winning_answer = most_common_answers[0]
            winning_indices = answer_to_indices[winning_answer]
            selected_index = winning_indices[0]  # Select the first found index
            print(f"🏆 [BON] Winner: selecting Answer")
        else:
            # In case of tie, select randomly
            import random as rand
            winning_answer = rand.choice(most_common_answers)
            winning_indices = answer_to_indices[winning_answer]
            selected_index = rand.choice(winning_indices)
            print(f"🎯 [BON] Tiebreak: {len(most_common_answers)} answers tied with {max_count} votes")
            print(f"🎲 [BON] Random selection: {winning_answer} - Selected Answer{selected_index+1}")
        
        selected_answer = answers[selected_index]
        print(f"📊 [BON] Total tokens used: 0 (majority voting)")
        return [(selected_answer, 0, filenames, selected_index)]
    
    elif evaluation_method == "omni":
        print(f"\n🍒 [BON] best evaluation: prioritize correct answer from Answers...")
        for i, answer in enumerate(answers):
            if is_correct(answer, gold, dataset_type):
                print(f"✅ [BON] Correct answer found: Answer")
                print(f"📊 [BON] Total tokens used: 0 (best evaluation)")
                return [(answer, 0, filenames, i)]
        print(f"⚠️ [BON] Correct answer not found, returning first Answer")
    else:
        print(f"\n❓ [BON] Unknown evaluation method, returning first Answer by default")
    
    print(f"📤 [BON] Final selection: first Answer")
    print(f"📊 [BON] Total tokens used: 0 (cherry-picking)")
    return [(answers[0], 0, filenames, 0)]


def generate_answerBoN_save(instruction, prompt, N, gold, dataset_name, problem_number, file_start=0, direct_answer=True, use_grammar=True, evaluation_method="pick_answer", run_number=0):
    """
    BoN answer generation based on reward model with saving functionality
    
    Parameters
    ----------
    instruction : str
        Problem statement
    prompt : str
        Prompt
    N : int
        BoN generation count
    gold : str
        Correct answer
    dataset_name : str
        Dataset name (e.g., "aime2024", "gsm8k")
    problem_number : int
        Problem number
    file_start : int
        Starting point for save file number (default: 0)
    direct_answer : bool
        Whether to use direct answer mode
    use_grammar : bool
        Whether to use grammar constraints
    evaluation_method : str
        Evaluation method ("omni": prioritize correct answer, "pick_answer": LLM judgment, "reward": reward model, "self_certainty": self-certainty score)
    run_number : int
        Run number (default: 0) - used to identify multiple runs
        
    Returns
    -------
    tuple
        (Selected optimal Answer, number of tokens used)
    """
    print(f"generate_answerBoN_save evaluation_method = {evaluation_method}")
    print(f"Saving answers to: {SAVED_ANSWERS_DIR}/{dataset_name}_prob{problem_number}_answer*.txt (starting from {file_start})")
    
    # Create save directory
    import os
    os.makedirs(SAVED_ANSWERS_DIR, exist_ok=True)
    
    # Check existing file sizes (skip generation if 2KB or more)）
    all_files_exist_and_large = True
    min_file_size_kb = 2  # 2KB
    
    for i in range(N):
        filename = f"{SAVED_ANSWERS_DIR}/{dataset_name}_prob{problem_number}_answer{file_start + i}.txt"
        if not os.path.exists(filename):
            all_files_exist_and_large = False
            break
        else:
            file_size_kb = os.path.getsize(filename) / 1024  # in KB
            if file_size_kb < min_file_size_kb:
                all_files_exist_and_large = False
                break
    
    if all_files_exist_and_large:
        print(f"✅ All answer files exist and are 2KB or larger. Skipping generation.")
        print(f"   File range: {dataset_name}_prob{problem_number}_answer{file_start}.txt ~ {dataset_name}_prob{problem_number}_answer{file_start + N - 1}.txt")
        return "", 0  # Return empty answer and 0 tokens and exit
    
    # template_loader = TemplateLoader()
    if False and use_grammar:
        grammar = template_loader.get_grammar("step_by_step")
    else:
        grammar = None
    if direct_answer:
        grammar = None
        prompt = instruction
        if "mistral" in LLM_MODEL_PORT_8100.lower() or "magistral" in LLM_MODEL_PORT_8100.lower(): #mistral does not like boxed..
            pass
#            if dataset_name == "mmlu_pro":
#                prompt = prompt + "\n You must end with a single line formatted exactly as:\nFinal Answer: <letter choice (A), (B), (C), etc.>"
#            elif dataset_name == "gpqa_diamond":
#                prompt = prompt + "\n You must end with a single line formatted exactly as:\nFinal Answer: <letter choice (A), (B), (C), etc.>"
#            elif "aime" in dataset_name:
#                prompt = prompt + "\n You must end with a single line formatted exactly as:\nFinal Answer: <AIME-integer-0-to-999>"
        else:
            if dataset_name == "mmlu_pro":
                prompt = prompt + "\n Please reason step by step, and put your final answer as the letter choice (A), (B), (C), etc. within \\boxed{}."
            elif dataset_name == "gpqa_diamond":
                prompt = prompt + "\n Please reason step by step, and put your final answer as the letter choice (A), (B), (C), etc. within \\boxed{}."
            else:
                prompt = prompt + "\n Please reason step by step, and put your final answer within \\boxed{}." # https://huggingface.co/Qwen/QwQ-32B#usage-guidelines recommended by QwQ-32B
    print("prompt = ", prompt)
    print("grammar = ", grammar)

    answers = []
    scores = []
    score_answers = []
    answer_counter = file_start  # Save counter (starting from specified start number)）
    B = min(16, N)
    
    for batch_idx in range(N//B): # int division
        print(f"generating {batch_idx*B} to {(batch_idx+1)*B-1} answers")
        sys.stdout.flush()
        
        # Multi-GPU batch generation using verbose version）
        max_tokens = get_max_tokens_for_dataset(args.dataset_type, args.max_new_tokens)
        
        # Use generate_answer_client_verbose to get detailed response
        answers_batch_verbose = []
        certainties_batch = []
        
        try:
            # Determine port considering GPU distribution
            num_gpus = int(os.getenv("NUM_GPUS", "1"))
            #print(f"run_number: {run_number}")
            #sys.exit()
            gpu_index = run_number % num_gpus #random.randint(0, num_gpus - 1)  # Completely random GPU selection
            target_port = 8100 + gpu_index  # Distributed from base port 8100
            print(f"🔧 [VERBOSE] Batch generation: using GPU (port ), B={B}")
            
            # Grammar parameter processing
            verbose_args = {
                'prompt': prompt,
                'max_new_tokens': max_tokens,
                'n': B,  # Generate B Answers in one call
                'temperature': TEMPERATURE,
                'port': target_port,
            }
            if grammar is not None:
                verbose_args['grammar'] = grammar
            
            verbose_results = generate_answer_client_verbose(**verbose_args)
            
            if verbose_results and len(verbose_results) > 0:
                # Extract information for each Answer from verbose_results
                for batch_item, result in enumerate(verbose_results):
                    # Extract information according to result structure
                    if len(result) >= 5:
                        answer, token_count, step_logprobs = result[0], result[1], result[4]
                    else:
                        print(f"⚠️ [VERBOSE] Unexpected result format: {type(result)}")
                        answer = str(result) if result else ""
                        token_count = 0
                        step_logprobs = []
                    
                    # Calculate Self-certainty
                    certainty = calculate_self_certainty(step_logprobs, k=20)
                    
                    answers_batch_verbose.append((answer, token_count))
                    certainties_batch.append(certainty)
                    
                    print(f"📊 [VERBOSE] Answer{batch_item}: tokens count, certainty={certainty:.4f}")
            else:
                print(f"❌ [VERBOSE] Batch generation failed")
                # Add B empty Answers
                for batch_item in range(B):
                    answers_batch_verbose.append(("", 0))
                    certainties_batch.append(0.0)
                    
        except Exception as e:
            print(f"❌ [VERBOSE] Batch generation error: {str(e)}")
            # Add B empty Answers
            for batch_item in range(B):
                answers_batch_verbose.append(("", 0))
                certainties_batch.append(0.0)
        
        # Extract answers from tuples
        answers_batch = [answer for answer, token_count in answers_batch_verbose]
        # Extract token counts separately
        token_counts_batch = [token_count for answer, token_count in answers_batch_verbose]
        prompts = [prompt] * B
        print(f"calling get_scores")
        answers_batch = [remove_after_end(answer) for answer in answers_batch]
        
        # Save each answer (including certainty information)
        for i, answer in enumerate(answers_batch):
            print(f"answer{i} = {answer}")
            
            # Generate filename
            filename = f"{SAVED_ANSWERS_DIR}/{dataset_name}_prob{problem_number}_answer{answer_counter}.txt"
            
            # Save answer to file (including certainty information)
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Problem Number: {problem_number}\n")
                    f.write(f"Answer Index: {answer_counter}\n")
                    f.write(f"Gold Answer: {gold}\n")
                    f.write(f"Generated Tokens: {token_counts_batch[i]}\n")
                    f.write(f"Self-Certainty: {certainties_batch[i]:.6f}\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write("="*50 + "\n")
                    f.write(answer)
                print(f"Saved answer {answer_counter} to {filename} (certainty: {certainties_batch[i]:.4f})")
            except Exception as e:
                print(f"Error saving answer {answer_counter}: {str(e)}")
            
            answer_counter += 1
        
        print(f"setsize(answers_batch) = {len(set(answers_batch))}")
        #scores_batch = [get_score(instruction, answer) for answer in answers_batch]
        scores_batch = [0 for answer in answers_batch]
        answers = answers + answers_batch
        scores = scores + scores_batch
        
    print(f"Total {len(answers)} answers generated and saved")
    
    return answers[0], 0  # In this case, return the first Answer without selection (rarely used). Displayed 'correct' will be Bo1

num_correct = num_incorrect = num_unk = 0
scores, corrects = [], []

print(f"\n🚀 [MAIN] Starting BoN processing with configuration:")
print(f"🚀 [MAIN] Available LLM port: 8100 (Qwen3-4B)")
print(f"🚀 [MAIN] Reward server port: 9000 (local), 8001/8002 (external)")
print(f"🚀 [MAIN] Saved answers directory: {SAVED_ANSWERS_DIR}")
print(f"🚀 [MAIN] Processing {len(local_data)} samples with rank {rank}\n")

# Determine run number (required regardless of use_save)
run_number = args.run if args.run is not None else 0

# Save selection information only when use_save=False (when using generate_answerBoN)
if not args.use_save:
    print(f"📝 [CHOICE] Run number for saving selection info: {run_number}")
    
    # Generate evaluation_method suffix for filename
    if args.evaluation_method == 'reward':
        reward_model_id = os.environ.get('REWARD_MODEL_ID', '')
        reward_model_name = extract_reward_model_name(reward_model_id)
        evaluation_suffix = f"reward_{reward_model_name}"
    elif args.evaluation_method == 'reward_orm':
        reward_model_id = os.environ.get('REWARD_MODEL_ID', '')
        reward_model_name = extract_reward_model_name(reward_model_id)
        evaluation_suffix = f"reward_orm_{reward_model_name}"
    # removed: comp_reward suffix generation (deprecated)
    else:
        evaluation_suffix = args.evaluation_method
    
    print(f"📝 [CHOICE] Save destination: {SAVED_CHOICES_DIR}/{args.dataset_type}_run{run_number}_{evaluation_suffix}_num{args.num_bon}.jsonl")
    if args.evaluation_method in ["pick_answer", "llm_judge_set"]:
        print(f"📝 [CHOICE] Using {args.evaluation_method}: recording selection info for multiple checkpoints (scale: 2, 4, 8, ...) for each problem")
    else:
        print(f"📝 [CHOICE] Using {args.evaluation_method}: recording one selection info (scale: 1) for each problem")
    print()

for i,(global_index, elem) in enumerate(local_data):
    print(f"global_index = {global_index}");sys.stdout.flush()
    
    instruction = elem["instruction"]
    question = generate_structured_prompt_reduction(instruction, args.dataset_type)
    gold   = elem["output"]
    if args.use_save:
        answer, llm_tokens = generate_answerBoN_save(instruction, question, args.num_bon, gold, args.dataset_type, global_index, file_start=args.file_start, use_grammar=False, evaluation_method=args.evaluation_method, run_number=run_number)
        paraphrased_score = 0  # Set score to 0 even for save mode
    else: # For generate_answerBoN case
        results_list = generate_answerBoN(instruction, question, args.num_bon, gold, global_index, args.run, use_grammar=False, evaluation_method=args.evaluation_method, dataset_type=args.dataset_type)
        paraphrased_score = 0  # Set score to 0 even for BoN
        
        # Use final result (last checkpoint)
        answer, llm_tokens, filenames, selected_index = results_list[-1] if results_list else ("", 0, [], -1)
        print(f"📊 [MAIN] Selected file: {filenames[selected_index] if 0 <= selected_index < len(filenames) else 'N/A'} (index: {selected_index})")
        print(f"📊 [MAIN] Number of checkpoints: {len(results_list)}")
        
        # Save selection information in JSONL format (for each checkpoint)
        if run_number is not None:
            # Generate checkpoints (same logic as pick_answer function)
            checkpoints = []
            if args.evaluation_method in ["pick_answer", "llm_judge_set"] and len(results_list) > 1:
                power = 0
                while 2 ** power <= 100:  # Match with max_judgments
                    checkpoints.append(2 ** power)
                    power += 1
                if checkpoints[-1] != 100:
                    checkpoints.append(100)
            else:
                checkpoints = [1]  # scale=1 for methods other than pick_answer
            
            print(f"📝 [CHOICE] Saving selection info {len(results_list)} times (scales: {checkpoints[:len(results_list)]})")
            
            # Save selection information for each checkpoint
            for checkpoint_idx, (checkpoint_answer, checkpoint_llm_tokens, checkpoint_filenames, checkpoint_selected_index) in enumerate(results_list):
                # Determine scale value for checkpoint
                # For first_valid mode (when results_list has only one entry), scale=-1
                if args.first_valid and len(results_list) == 1:
                    scale_value = -1
                    print(f"📝 [CHOICE] first_valid mode detected: scale={scale_value}")
                elif checkpoint_idx < len(checkpoints):
                    scale_value = checkpoints[checkpoint_idx]
                else:
                    scale_value = checkpoints[-1] if checkpoints else 1
                
                print(f"📝 [CHOICE] Checkpoint{checkpoint_idx+1}: scale={scale_value}, selected_index={checkpoint_selected_index}")
                
                # Calculate total tokens for target files at each checkpoint
                total_token_count = sum_generated_tokens_for_files(checkpoint_filenames)
                total_token_count = total_token_count + checkpoint_llm_tokens
                save_choice_info(
                    global_index, 
                    checkpoint_filenames, 
                    checkpoint_selected_index, 
                    args.dataset_type, 
                    args.evaluation_method, 
                    run_number, 
                    args.num_bon, 
                    scale_value,
                    total_token_count
                )
        
    print(f"final answer chosen = {answer}")
    print(f"📊 [MAIN] LLM generated tokens: {llm_tokens}")
    score  = 0 # get_score(instruction, answer)
    result = {}
    import re
    boxed_pattern = r'\\boxed{([^}]+)}'
    boxed_match = re.search(boxed_pattern, answer)
    if is_correct(answer, gold, args.dataset_type):
        result["result"] = "correct"
    else:
        result["result"] = "incorrect"
    print(f"gold = {gold} result = {result}")

    # --- Cumulative statistics
    if   result["result"] == "correct":   num_correct += 1; corrects.append(1)
    elif result["result"] == "incorrect": num_incorrect += 1; corrects.append(0)
    else:                                 num_unk      += 1
    scores.append(score)

    # --- ① Append one line to JSON Lines and flush immediately --------------------
    rec = {
        "rank": rank, "local_index": i, "global_index": global_index,
        "instruction": elem["instruction"],   # Truncate if too long
        "answer": answer, "gold": gold,
        "score": score, "paraphrased_score": paraphrased_score, 
        "llm_tokens": llm_tokens,  # Record LLM generated token count
        "grade": result["result"],
        "n_correct": num_correct,
        "n_incorrect": num_incorrect,
        "n_unk": num_unk,
    }
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()          # python buffer → OS buffer
        os.fsync(f.fileno())  # OS buffer → disk

    # --- ② Display progress lightly on screen -----------------------------
    if state.is_main_process and i % 10 == 0:
        print(f"[rank {rank}] iter {i} "
              f"correct={num_correct}/{i+1}", file=sys.stderr)

    sys.stdout.flush()     # For existing print statements
