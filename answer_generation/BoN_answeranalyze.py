#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoN_answeranalyze.py

Script to output the correctness of answers in the saved_answers directory in table format
Answer file format: dataset_name_prob{problem_number}_answer{answer_number}.txt
"""

import os
import re
import glob
import csv
import json
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# from BoN_utils import is_correct, extract_boxed_answer

def extract_mistral_aime_answer(text: str) -> int | None:
    """
    Extract AIME answer using Final Answer format for Mistral/Magistral models
    """
    # 1) Recommended format
    m = re.search(r"Final Answer:\s*([0-9]{1,3})\s*$", text, flags=re.IGNORECASE|re.MULTILINE)
    if m:
        ans = int(m.group(1))
        return ans if 0 <= ans <= 999 else None

    # 2) Fallback for declaration pattern
    m = re.search(r"(?:the answer is|Therefore[, ]? the answer (?:is|=))\s*([0-9]{1,3})(?!\d)", text, flags=re.IGNORECASE)
    if m:
        ans = int(m.group(1))
        return ans if 0 <= ans <= 999 else None
    return None

def extract_mistral_letter_answer(text: str) -> str | None:
    """
    Extract letter answer (A, B, C, D, etc.) using Final Answer format for Mistral/Magistral models
    """
    # 1) Recommended format
    m = re.search(r"Final Answer:\s*\(?([A-Z])\)?\s*$", text, flags=re.IGNORECASE|re.MULTILINE)
    if m:
        return m.group(1).upper()

    # 2) Fallback for declaration pattern
    m = re.search(r"(?:the answer is|Therefore[, ]? the answer (?:is|=)|the answer is)\s*\(?([A-Z])\)?", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3) Isolated character at end of text (with or without parentheses)
    m = re.search(r"\(?([A-Z])\)?\s*$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).upper()

    return None

def is_mistral_model() -> bool:
    """
    Check if the current LLM model is Mistral or Magistral
    """
    import os
    # Try to load .env file if dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Continue without dotenv
    
    llm_path = os.getenv("LLM_MODEL_PORT_8100", "").lower()
    return "mistral" in llm_path or "magistral" in llm_path

def extract_boxed_answer(text: str) -> str:
    """
    Extract content from \\boxed{...} (supports nested brackets)
    If multiple \\boxed{} exist, extract the last one
    
    Args:
        text: Text to extract from
    
    Returns:
        Content inside boxed, or None if not found
    """
    # Find the last \boxed{
    boxed_start = text.rfind('\\boxed{')
    if boxed_start != -1:
        # Extract from right after \boxed{ considering bracket nesting
        start_pos = boxed_start + 7  # len('\\boxed{') = 7
        brace_count = 1
        end_pos = start_pos
        
        while end_pos < len(text) and brace_count > 0:
            if text[end_pos] == '{':
                brace_count += 1
            elif text[end_pos] == '}':
                brace_count -= 1
            end_pos += 1
        
        if brace_count == 0:
            return text[start_pos:end_pos-1].strip()  # Exclude the last }
    
    # If no \boxed{} found, look for number-like strings
    import re
    # Look for number patterns (integers, decimals, fractions, etc.)
    number_patterns = [
        r'\b(\d+\.?\d*)\b',  # integers or decimals
        r'\b(\d+/\d+)\b',    # fractions
        r'\b(\d+)\b'         # integers only
    ]
    
    for pattern in number_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]  # Return the last found number
    
    return None

def normalize_math_notation(text: str) -> str:
    """
    Function to normalize mathematical notation
    
    Args:
        text: Text to normalize
    
    Returns:
        str: Normalized text
    """
    if not text:
        return text
    
    import re
    
    # Remove all spaces
    text = re.sub(r'\s+', '', text)
    
    # Remove \left, \right
    text = re.sub(r'\\left|\\right', '', text)
    
    # Replace \tfrac and \dfrac with \frac
    text = re.sub(r'\\[td]frac', r'\\frac', text)

    # Replace \text{(X)} patterns (where X is alphabet) with single alphabet
    # This processing needs to be done before \text{} conversion, so do it here
    text = re.sub(r'\\text\{\\?\(?([A-Za-z])\\?\)?\}', r'\1', text)
    
    # Normalize patterns with degrees on the right side like \frac{270}7\text{degrees}
    # Remove degrees before \text{} conversion
    text = re.sub(r'(\\frac\{[^}]+\})(\d+)\\text\{degrees\}$', r'\1{\2}', text)
    text = re.sub(r'\\text\{degrees\}$', '', text)

    # Remove \text{XXX} tags to make them XXX
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    
    # Replace \sqrt2 with \sqrt{2} (same for other single digit patterns)
    text = re.sub(r'\\sqrt(\d)', r'\\sqrt{\1}', text)
    
    # Replace \; with , in patterns like 3,\;5,\;7
    text = re.sub(r',\\;', ',', text)
    
    # Remove \% like 10\%
    text = re.sub(r'\\%', '', text)
    
    # Replace ,\, with ,
    text = re.sub(r',\\,', ',', text)
    
    # Replace ,\! with , (keep comma) - but process comma separators between numbers later
    text = re.sub(r',\\!', ',', text)
    
    # Remove x \in from left side like x \in [-2,7]
    text = re.sub(r'^[a-zA-Z]\s*\\in\s*', '', text)
    
    # Remove x = from left side like x = 1
    text = re.sub(r'^[a-zA-Z]\s*=\s*', '', text)
    
    # Convert vertical vectors to horizontal vectors for \begin{pmatrix} ... \end{pmatrix}
    # Extract elements separated by newlines or \\ and convert to comma-separated
    def convert_pmatrix(match):
        content = match.group(1)
        # Split by \\ or newline
        elements = re.split(r'\\\\|\n', content)
        # Trim each element and keep only non-empty ones
        elements = [elem.strip() for elem in elements if elem.strip()]
        return '(' + ','.join(elements) + ')'
    
    text = re.sub(r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', convert_pmatrix, text, flags=re.DOTALL)
    
    # Add brackets for patterns like \frac14 to make them \frac{1}{4}
    text = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', text)
    
    # Remove \$ from left side like \$36
    text = re.sub(r'^\\\$', '', text)
    
    # Remove \mbox{...} from right side like 15\mbox{ cm}^2 (with or without ^number)
    text = re.sub(r'\\mbox\{[^}]*\}(?:\^\d*)?$', '', text)
    
    # Remove ^{th}grade or ^{th} from right side like 12^{th}grade
    text = re.sub(r'\^\{th\}(?:grade)?$', '', text)
    
    # Remove units like \text{cents} from right side like 5.4 \text{ cents} (no spaces after space removal)
    text = re.sub(r'\\text\{(?:cents?|dollars?|units?|cm|mm|km|m|inches?|feet|ft|yards?|yd)\}$', '', text)
    
    # Also remove unit strings after \text{} has been converted
    text = re.sub(r'(?:cents?|dollars?|units?|cm|mm|km|m|inches?|feet|ft|yards?|yd)$', '', text)
    
    # Remove [Xpt] patterns like [4pt]
    text = re.sub(r'\[\d+pt\]', '', text)
    
    # Remove commas between numbers (thousands separator) - except inside parentheses
    # First protect commas inside parentheses, then process
    def remove_commas_outside_parentheses(text):
        # Protect commas inside parentheses and remove commas between numbers outside parentheses
        result = ""
        paren_depth = 0
        i = 0
        while i < len(text):
            if text[i] == '(':
                paren_depth += 1
                result += text[i]
            elif text[i] == ')':
                paren_depth -= 1
                result += text[i]
            elif text[i] == ',' and paren_depth == 0:
                # Comma outside parentheses, remove if surrounded by digits
                if (i > 0 and i < len(text) - 1 and 
                    text[i-1].isdigit() and text[i+1].isdigit()):
                    pass  # Remove comma (don't add to result)
                else:
                    result += text[i]
            else:
                result += text[i]
            i += 1
        return result
    
    text = remove_commas_outside_parentheses(text)
    
    # Replace patterns like \frac9{19} to \frac{9}{19}
    text = re.sub(r'\\frac(\d+)\{', r'\\frac{\1}{', text)
    
    
    # Remove degree symbols (^\circ and °)
    # Also handle ^{\circ} (with braces)
    text = re.sub(r'\^\s*\{\s*\\circ\s*\}', '', text)
    text = re.sub(r'\^\s*\\circ', '', text)
    text = re.sub(r'°', '', text)
    
    # Remove patterns with \, at beginning and end like \,1+274i\,
    text = re.sub(r'^\\,(.*)\\,$', r'\1', text)
    
    return text


def normalize_letter_choice_simple(choice_text: str) -> str:
    """
    Extract the leading choice alphabet and normalize to a single uppercase letter.
    Examples:
      - "(A)\\ 32" → "A"
      - "(\\text{C})、、" → "C"
      - "(A)\\; 6.3\\times10^{-7}\\ \\text{M}" → "A"
      - "\\text{(B)}" → "B"
      - "\\text{b}" → "B"
      - "(\\mathrm{B})" → "B"
    Strings other than the leading part are ignored.
    """
    if not choice_text:
        return choice_text
    t = str(choice_text).lstrip()
    # If leading part has LaTeX commands like \\text{...} or \\mathrm{...}, expand to first block content
    m0 = re.match(r'^\s*\\[A-Za-z]+\{([^}]*)\}', t)
    if m0:
        t = m0.group(1).strip()
    # Extract leading parentheses/LaTeX command with alphabet (parentheses priority)
    m = re.match(r'^\s*\(\s*(?:\\[A-Za-z]+\{\s*)?([A-Za-z])\s*(?:\}\s*)?\)', t)
    if m:
        return m.group(1).upper()
    # LaTeX command without parentheses or plain alphabet
    m = re.match(r'^\s*(?:\\[A-Za-z]+\{\s*)?([A-Za-z])\s*(?:\}\s*)?', t)
    if m:
        return m.group(1).upper()
    # Single character
    if len(t) == 1 and t.isalpha():
        return t.upper()
    return t

def extract_medrect_answer(text: str) -> str:
    """
    Extract answer for medrect dataset
    Format:
    - "CORRECT" -> error_sentence_id = 0
    - "文番号: ..." -> error_sentence_id = 文番号
    
    Args:
        text: Text to extract from
    
    Returns:
        "0" if no error, or sentence_id as string if error found
    """
    if not text:
        return None
    
    # Remove <think> tags if present
    text_clean = text.strip()
    if '</think>' in text_clean:
        # Extract content after </think>
        text_clean = text_clean.split('</think>', 1)[1].strip()
    
    # Check if answer is CORRECT
    if re.search(r'\bCORRECT\b', text_clean, re.IGNORECASE):
        return "0"
    
    # Extract sentence number from format "文番号: ..."
    # Also handle English format "Sentence N:" or just "N:"
    patterns = [
        r'^(\d+)\s*:',  # "6: ..." at the beginning
        r'\b(?:文|Sentence)\s*(\d+)\s*:',  # "文6:" or "Sentence 6:"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.MULTILINE)
        if match:
            return match.group(1)
    
    # If no clear pattern found, return None
    return None

def extract_answer_smart(text: str, dataset_type: str = None) -> str:
    """
    Smart answer extraction that chooses the appropriate method based on model and dataset
    
    Args:
        text: Text to extract from
        dataset_type: Type of dataset (aime2024, aime2025, mmlu_pro, gpqa_diamond, medrect, etc.)
    
    Returns:
        Extracted answer or None if not found
    """
    # For medrect dataset, use special extraction
    if dataset_type and dataset_type.lower() == 'medrect':
        return extract_medrect_answer(text)
    
    # Helper: normalize single character choices regardless of parentheses and return
    def normalize_letter_choice_simple(choice_text: str) -> str:
        if not choice_text:
            return choice_text
        t = str(choice_text).strip()
        t = t.strip("()")
        # If wrapped in \text{...}, extract content (e.g. \text{(B)} or \text{B})
        m_text = re.match(r'^\\text\{([^}]*)\}$', t)
        if m_text:
            t = m_text.group(1).strip()
        m = re.match(r'^\s*\(?(?:\\text\{)?\s*([A-Za-z])\s*(?:\})?\)?', t)
        if m:
            return m.group(1).upper()
        if len(t) == 1 and t.isalpha():
            return t.upper()
        return t

    # Check if using Mistral/Magistral model
    if is_mistral_model():
        # For AIME datasets, use integer extraction
        if dataset_type and ('aime' in dataset_type.lower()):
            result = extract_mistral_aime_answer(text)
            if result is not None:
                return str(result)
            # Fall back to boxed extraction if Final Answer format fails
            boxed_result = extract_boxed_answer(text)
            if boxed_result:
                boxed_result = normalize_math_notation(boxed_result)
            return boxed_result
        
        # For MMLU_PRO and GPQA_DIAMOND, use letter extraction
        elif dataset_type and (dataset_type.lower() in ['mmlu_pro', 'gpqa_diamond']):
            result = extract_mistral_letter_answer(text)
            if result is not None:
                # For GPQA/MMLU, apply normalization for single letters with parentheses
                if dataset_type.lower() == 'gpqa_diamond':
                    return normalize_letter_choice_simple(result)
                return result
            # Fall back to boxed extraction if Final Answer format fails
            boxed_result = extract_boxed_answer(text)
            # For GPQA-Diamond, mathematical notation normalization is not needed, but parentheses normalization is applied
            if boxed_result:
                if dataset_type.lower() == 'gpqa_diamond':
                    boxed_result = normalize_letter_choice_simple(boxed_result)
                else:
                    boxed_result = normalize_math_notation(boxed_result)
            return boxed_result
        
        # For other datasets, try both methods
        else:
            # Try letter first (more specific)
            letter_result = extract_mistral_letter_answer(text)
            if letter_result is not None:
                return letter_result
            
            # Then try integer
            int_result = extract_mistral_aime_answer(text)
            if int_result is not None:
                return str(int_result)
            
            # Finally, fall back to boxed extraction
            boxed_result = extract_boxed_answer(text)
            if boxed_result:
                boxed_result = normalize_math_notation(boxed_result)
            return boxed_result
    
    # Fall back to original boxed extraction for non-Mistral models
    result = extract_boxed_answer(text)
    
    # Apply normalization to the extracted result
    if result and dataset_type:
        if dataset_type.lower() == 'gpqa_diamond':
            # For GPQA-Diamond, apply only parentheses normalization
            # (Example) "(B)" → "B"
            def normalize_letter_choice_simple_local(choice_text: str) -> str:
                if not choice_text:
                    return choice_text
                t = str(choice_text).strip()
                # If wrapped in \text{...}, extract content
                m_text = re.match(r'^\\text\{([^}]*)\}$', t)
                if m_text:
                    t = m_text.group(1).strip()
                m = re.match(r'^\s*\(?(?:\\text\{)?\s*([A-Za-z])\s*(?:\})?\)?', t)
                if m:
                    return m.group(1).upper()
                if len(t) == 1 and t.isalpha():
                    return t.upper()
                return t
            result = normalize_letter_choice_simple_local(result)
        else:
            result = normalize_math_notation(result)
    
    return result

def is_correct(answer: str, gold: str, dataset_type: str = None) -> bool:
    """
    Determine if the generated answer is correct (simplified version)
    
    Parameters
    ----------
    answer : str
        Generated answer
    gold : str
        Correct answer
    dataset_type : str, optional
        Dataset type
    
    Returns
    -------
    bool
        True if answer is correct, False otherwise
    """
    if not answer or not gold:
        return False
    
    # Extract answer using smart extraction
    extracted_answer = extract_answer_smart(answer, dataset_type)
    
    if extracted_answer is None:
        extracted_answer = answer.strip()
    
    # Compare with correct answer (as strings)
    gold_str = str(gold).strip()
    extracted_str = str(extracted_answer).strip()
    
    # For medrect dataset, use simple string comparison
    if dataset_type and dataset_type.lower() == 'medrect':
        # Direct string comparison for sentence IDs
        return extracted_str == gold_str
    
    # Apply normalization for comparison
    if dataset_type and dataset_type.lower() == 'gpqa_diamond':
        # For GPQA-Diamond, don't normalize mathematical notation, only normalize parentheses for single letter choices
        answer_normalized = str(answer).strip()
        gold_normalized = normalize_letter_choice_simple(gold_str)
        extracted_normalized = normalize_letter_choice_simple(extracted_str)
    else:
        # For other datasets, apply normalization
        answer_normalized = normalize_math_notation(str(answer).strip())
        gold_normalized = normalize_math_notation(gold_str)
        extracted_normalized = normalize_math_notation(extracted_str)
    
    # Exact match
    if extracted_str == gold_str:
        return True
    
    # Normalized math notation comparison (extracted answer)
    if extracted_normalized == gold_normalized:
        return True
    
    # Normalized math notation comparison (original answer)
    if answer_normalized == gold_normalized:
        return True
    
    # Try comparing as numbers
    try:
        extracted_num = float(extracted_str)
        gold_num = float(gold_str)
        return abs(extracted_num - gold_num) < 1e-9
    except (ValueError, TypeError):
        pass
    
    # Try comparing original answer as number
    try:
        answer_num = float(str(answer).strip())
        gold_num = float(gold_str)
        return abs(answer_num - gold_num) < 1e-9
    except (ValueError, TypeError):
        pass
    
    # Try comparing as numbers with normalization
    try:
        extracted_num = float(extracted_normalized)
        gold_num = float(gold_normalized)
        return abs(extracted_num - gold_num) < 1e-9
    except (ValueError, TypeError):
        pass
    
    # Try comparing original answer normalized as number
    try:
        answer_num = float(answer_normalized)
        gold_num = float(gold_normalized)
        return abs(answer_num - gold_num) < 1e-9
    except (ValueError, TypeError):
        pass
    
    
    return False

def extract_llm_name_from_env() -> str:
    """
    Extract LLM name from environment variable LLM_MODEL_PORT_8100
    
    Returns:
        str: LLM name (e.g., "EXAONE-Deep-32B" from "LGAI-EXAONE/EXAONE-Deep-32B")
    """
    import os
    # Try to load .env file if dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Continue without dotenv
    
    llm_path = os.getenv("LLM_MODEL_PORT_8100", "/workspace/Noname")
    
    # Extract model name from path
    # Example: "LGAI-EXAONE/EXAONE-Deep-32B" -> "EXAONE-Deep-32B"
    # Example: "/workspace/Qwen3-4B" -> "Qwen3-4B"
    
    model_name = os.path.basename(llm_path)
    
    # If it contains organization prefix, remove it
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    return model_name

def load_answers_data(dataset: str, parquet_path: Optional[str] = None) -> Dict[int, str]:
    """
    Load correct answer data from the parquet file of the specified dataset
    
    Args:
        dataset: Dataset name
        parquet_path: Path to the parquet file (use default path if None)
    
    Returns:
        Dict: Dictionary of correct answers with problem numbers (row numbers, 0-indexed) as keys
    """
    # Set default path
    if parquet_path is None:
        if dataset.lower() == 'aime2024':
            parquet_path = "/workspace/AIME_2024/aime_2024_problems.parquet"
        elif dataset.lower() == 'aime2025':
            parquet_path = "/workspace/AIME2025/aime2025-full.jsonl"
        elif dataset.lower() == 'math500':
            parquet_path = "/workspace/prm800k/prm800k/math_splits/test.jsonl"
        elif dataset.lower() == 'mmlu_pro':
            parquet_path = "/workspace/MMLU-Pro/data/validation-00000-of-00001.parquet"
        elif dataset.lower() == 'gpqa_diamond':
            parquet_path = "/workspace/GPQA-Diamond/test/gpqa_diamond.parquet"
        elif dataset.lower() == 'medrect':
            parquet_path = "/workspace/medrect/data/medrect/medrect-ja-step4-accepted.json"
        else:
            print(f"Warning: Default path for {dataset} is not set. Please specify with --parquet-path option.")
            return {}
    
    try:
        answers = {}
        
        # Determine loading method based on file extension
        if parquet_path.endswith('.parquet'):
            if not HAS_PANDAS:
                print("Warning: pandas is not available. Cannot load parquet file; please install pandas or provide a JSONL path.")
                return {}
            # For parquet files
            df = pd.read_parquet(parquet_path)
            
            # Problem ID is simply the row number (0-indexed)
            for index, row in df.iterrows():
                problem_num = index  # DataFrame index (0-indexed)
                # Try different possible column names for the answer
                answer_value = None
                for col_name in ['Answer', 'answer', 'correct_answer', 'gold_answer']:
                    if col_name in row and pd.notna(row[col_name]):
                        answer_value = str(row[col_name])
                        break
                
                if answer_value is not None:
                    answers[problem_num] = answer_value
                else:
                    print(f"Warning: No answer found for problem {problem_num}, available columns: {list(df.columns)}")
                
        elif parquet_path.endswith('.jsonl') or parquet_path.endswith('.json'):
            # For jsonl and json files
            with open(parquet_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Check if it's a JSON array
                if content.startswith('['):
                    # JSON array format
                    data_list = json.loads(content)
                    for line_num, data in enumerate(data_list):
                        # For medrect dataset: get answer from error_flag and error_sentence_id
                        if dataset.lower() == 'medrect':
                            error_flag = data.get('error_flag', 0)
                            if error_flag == 0:
                                # No error
                                answers[line_num] = "0"
                            else:
                                # Has error, use error_sentence_id
                                error_sentence_id = data.get('error_sentence_id')
                                if error_sentence_id is not None:
                                    answers[line_num] = str(error_sentence_id)
                                else:
                                    print(f"Warning: error_flag=1 but error_sentence_id not found for problem {line_num}")
                                    answers[line_num] = "0"
                        else:
                            # Standard format
                            answer = data.get('answer') or data.get('Answer') or data.get('gold_answer')
                            if answer is not None:
                                answers[line_num] = str(answer)
                else:
                    # JSONL format (one JSON per line)
                    f.seek(0)  # Reset file pointer
                    for line_num, line in enumerate(f):
                        if line.strip():  # Skip empty lines
                            try:
                                data = json.loads(line.strip())
                                # For medrect dataset: get answer from error_flag and error_sentence_id
                                if dataset.lower() == 'medrect':
                                    error_flag = data.get('error_flag', 0)
                                    if error_flag == 0:
                                        # No error
                                        answers[line_num] = "0"
                                    else:
                                        # Has error, use error_sentence_id
                                        error_sentence_id = data.get('error_sentence_id')
                                        if error_sentence_id is not None:
                                            answers[line_num] = str(error_sentence_id)
                                        else:
                                            print(f"Warning: error_flag=1 but error_sentence_id not found for problem {line_num}")
                                            answers[line_num] = "0"
                                else:
                                    # Standard format
                                    answer = data.get('answer') or data.get('Answer') or data.get('gold_answer')
                                    if answer is not None:
                                        answers[line_num] = str(answer)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Failed to parse JSON on line {line_num + 1}: {e}")
                                continue
        else:
            print(f"Warning: Unsupported file format: {parquet_path}")
            return {}
        
        print(f"Loaded {dataset} correct answer data: {len(answers)} problems")
        print(f"Problem number range: 0 to {len(answers)-1}")
        return answers
        
    except Exception as e:
        print(f"Failed to load correct answer data: {e}")
        return {}

# Hold correct answer data in global variable
ANSWERS_DATA = {}

def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse filename to extract information
    
    Args:
        filename: Filename (e.g., aime2024_prob15_answer16.txt)
    
    Returns:
        Dict: Dictionary containing dataset name, problem number, and answer number
        None: If parsing fails
    """
    pattern = r'(\w+)_prob(\d+)_answer(\d+)\.txt$'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'dataset': match.group(1),
            'problem_num': int(match.group(2)),
            'answer_num': int(match.group(3))
        }
    return None

def extract_generated_tokens(file_content: str) -> Optional[int]:
    """
    Extract the number from lines like "Generated Tokens: 1212".
    Returns None if not found.
    """
    if not file_content:
        return None
    # Common patterns to match token count lines
    patterns = [
        r"Generated\s+Tokens\s*:\s*(\d+)",
        r"Tokens\s*generated\s*:\s*(\d+)",
        r"Total\s+Tokens\s*:\s*(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, file_content, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None

def extract_think_content(file_content: str) -> str:
    """
    Extract content after <think> tag from file content
    
    Args:
        file_content: File content
    
    Returns:
        str: Content after <think> tag
    """
    think_pattern = r'<think>(.*?)(?=</think>|$)'
    match = re.search(think_pattern, file_content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no <think> tag, extract answer from entire content
    return file_content.strip()


def extract_final_content(file_content: str) -> str:
    """
    Extract content after </think> tag (the final answer section).
    If no </think> tag exists, return the whole content.
    This is suitable for extracting the final stated answer outside the CoT.
    """
    # Prefer content AFTER the closing think tag
    end_tag = '</think>'
    if end_tag in file_content:
        return file_content.split(end_tag, 1)[1].strip()
    # If no closing tag, fall back to the entire content
    return file_content.strip()



def analyze_answer_file(filepath: str) -> Dict:
    """
    Analyze a single answer file
    
    Args:
        filepath: Path to the answer file
    
    Returns:
        Dict: Analysis results
    """
    filename = os.path.basename(filepath)
    parsed_info = parse_filename(filename)
    
    if not parsed_info:
        return {
            'filename': filename,
            'error': 'Failed to parse filename',
            'is_correct': False
        }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract final answer section (prefer text after </think>)
        final_content = extract_final_content(content)
        
        # Prefer explicit token count if available in the whole file
        explicit_tokens = extract_generated_tokens(content)
        
        # Extract answer using smart extraction from the final section
        extracted_answer = extract_answer_smart(final_content, parsed_info['dataset'])
        
        # If answer is 100 characters or more, mark as "unextractable"
        if extracted_answer is not None and len(str(extracted_answer)) >= 100:
            extracted_answer = "unextractable"
        
        # Get correct answer
        # AIME2025 JSONL uses 0-based indexing by line number. Some saved files might be 1-based.
        dataset_key = parsed_info['dataset']
        # Robust lookup across possible key variants to avoid casing mismatches
        candidate_keys = [
            dataset_key,
            str(dataset_key).lower(),
            str(dataset_key).upper(),
            str(dataset_key).capitalize(),
        ]
        gold_map = {}
        for key in candidate_keys:
            if key in ANSWERS_DATA and ANSWERS_DATA.get(key):
                gold_map = ANSWERS_DATA[key]
                break
        gold_answer = None
        if gold_map:
            idx = int(parsed_info['problem_num'])
            gold_answer = gold_map.get(idx)
            if gold_answer is None:
                # Fallback: try problem_num - 1 if exists
                fallback_idx = idx - 1
                if fallback_idx in gold_map:
                    gold_answer = gold_map.get(fallback_idx)
        
        # Correctness judgment
        if gold_answer is not None:
            is_answer_correct = is_correct(final_content, gold_answer, parsed_info['dataset'])
        else:
            is_answer_correct = None
            
        return {
            'filename': filename,
            'dataset': parsed_info['dataset'],
            'problem_num': parsed_info['problem_num'],
            'answer_num': parsed_info['answer_num'],
            'extracted_answer': extracted_answer,
            'gold_answer': gold_answer,
            'is_correct': is_answer_correct,
            'file_size': os.path.getsize(filepath),
            'think_content_length': len(final_content),
            'token_count': explicit_tokens,
            'error': None
        }
        
    except Exception as e:
        return {
            'filename': filename,
            'error': f'File processing error: {str(e)}',
            'is_correct': False
        }

def analyze_all_answers(answers_dir: str = "saved_answers", target_dataset: str = None) -> List[Dict]:
    """
    Analyze answer files in the saved_answers directory
    
    Args:
        answers_dir: Directory containing answer files
        target_dataset: Target dataset name (analyze all files if None)
    
    Returns:
        List[Dict]: List of all analysis results
    """
    if not os.path.exists(answers_dir):
        print(f"Error: Directory {answers_dir} not found")
        return []
    
    # Pattern for answer files
    if target_dataset:
        # Target only files from specific dataset
        pattern = os.path.join(answers_dir, f"{target_dataset}_prob*_answer*.txt")
        print(f"Target dataset: {target_dataset}")
    else:
        # Target all answer files
        pattern = os.path.join(answers_dir, "*_prob*_answer*.txt")
        print("Target dataset: all")
    
    answer_files = glob.glob(pattern)
    
    if not answer_files:
        if target_dataset:
            print(f"Warning: No answer files for {target_dataset} found in {answers_dir}")
        else:
            print(f"Warning: No answer files found in {answers_dir}")
        return []
    
    print(f"Number of files to analyze: {len(answer_files)}")
    
    results = []
    for i, filepath in enumerate(answer_files, 1):
        print(f"Analyzing... ({i}/{len(answer_files)}) {os.path.basename(filepath)}")
        result = analyze_answer_file(filepath)
        results.append(result)
    
    return results

def create_summary_table(results: List[Dict], show_dict_answers: bool = False, show_token: bool = False):
    """
    Create summary table from analysis results
    
    Args:
        results: List of analysis results
        show_dict_answers: Whether to display dict_answers column
        show_token: Whether to display token count statistics
    
    Returns:
        DataFrame or List: Summary table
    """
    if not HAS_PANDAS:
        # Return list of dictionaries if pandas is not available
        valid_results = [r for r in results if not r.get('error')]
        if not valid_results:
            return []
    else:
        df = pd.DataFrame(results)
        
        if df.empty:
            return df
        
        # Create summary only with records that have no errors
        valid_df = df[df['error'].isna()]
        
        if valid_df.empty:
            return df
        
        valid_results = valid_df.to_dict('records')
    
    # Aggregation by dataset and problem
    summary_data = []
    
    # Get list of datasets
    datasets = list(set(r['dataset'] for r in valid_results if 'dataset' in r))
    
    for dataset in datasets:
        dataset_results = [r for r in valid_results if r.get('dataset') == dataset]
        
        # Get list of problem numbers
        problem_nums = list(set(r['problem_num'] for r in dataset_results if 'problem_num' in r))
        
        for problem_num in sorted(problem_nums):
            problem_results = [r for r in dataset_results if r.get('problem_num') == problem_num]
            
            total_answers = len(problem_results)
            correct_answers = len([r for r in problem_results if r.get('is_correct') == True])
            incorrect_answers = len([r for r in problem_results if r.get('is_correct') == False])
            unknown_answers = len([r for r in problem_results if r.get('is_correct') is None])
            
            accuracy = correct_answers / total_answers if total_answers > 0 else 0
            
            gold_answer = problem_results[0].get('gold_answer') if problem_results else None
            
            # Create dictionary of answers and their frequencies
            answer_counts = {}
            for result in problem_results:
                extracted_answer = result.get('extracted_answer')
                if extracted_answer is not None:
                    answer_counts[extracted_answer] = answer_counts.get(extracted_answer, 0) + 1
            
            # Get the most frequent answer (excluding unextractable)
            majority_answer = ""
            majority_count = 0
            if answer_counts:
                # Filter answers excluding unextractable
                non_unextractable_counts = {k: v for k, v in answer_counts.items() if k != "unextractable"}
                if non_unextractable_counts:
                    majority_answer = max(non_unextractable_counts, key=non_unextractable_counts.get)
                    majority_count = non_unextractable_counts[majority_answer]
                else:
                    # If only unextractable, display unextractable
                    majority_answer = max(answer_counts, key=answer_counts.get)
                    majority_count = answer_counts[majority_answer]
            
            # Get frequency of gold_answer
            gold_count = 0
            if gold_answer is not None:
                gold_count = answer_counts.get(str(gold_answer), 0)
            
            # Calculate token count statistics if requested
            avg_correct_tokens = 0.0
            avg_incorrect_tokens = 0.0
            if show_token:
                correct_results = [r for r in problem_results if r.get('is_correct') == True and 'token_count' in r]
                incorrect_results = [r for r in problem_results if r.get('is_correct') == False and 'token_count' in r]
                
                if correct_results:
                    avg_correct_tokens = sum(r['token_count'] for r in correct_results) / len(correct_results)
                
                if incorrect_results:
                    avg_incorrect_tokens = sum(r['token_count'] for r in incorrect_results) / len(incorrect_results)
            
            # Create all_answers: list of (answer, token_count) tuples
            all_answers = []
            for r in problem_results:
                ans = r.get('extracted_answer')
                tok = r.get('token_count')
                all_answers.append((ans, tok))

            # Create summary data
            summary_row = {
                'dataset': dataset,
                'problem_num': problem_num,
                'total_answers': total_answers,
                'correct_answers': correct_answers,
                'incorrect_answers': incorrect_answers,
                'unknown_answers': unknown_answers,
                'accuracy': f"{accuracy:.2%}",
                'gold_answer': gold_answer,
                'gold_count': gold_count,
                'majority_answer': majority_answer,
                'majority_count': majority_count,
                'all_answers': all_answers
            }
            
            # Add dict_answers column conditionally
            if show_dict_answers:
                summary_row['dict_answers'] = answer_counts
            
            # Add token count columns conditionally
            if show_token:
                summary_row['avg_correct_tokens'] = f"{avg_correct_tokens:.1f}" if avg_correct_tokens > 0 else "N/A"
                summary_row['avg_incorrect_tokens'] = f"{avg_incorrect_tokens:.1f}" if avg_incorrect_tokens > 0 else "N/A"
            
            summary_data.append(summary_row)
    
    # Add total row
    if summary_data:
        total_answers = sum(item['total_answers'] for item in summary_data)
        total_correct = sum(item['correct_answers'] for item in summary_data)
        total_incorrect = sum(item['incorrect_answers'] for item in summary_data)
        total_unknown = sum(item['unknown_answers'] for item in summary_data)
        total_accuracy = total_correct / total_answers if total_answers > 0 else 0
        
        total_row = {
            'dataset': 'total',
            'problem_num': '',
            'total_answers': total_answers,
            'correct_answers': total_correct,
            'incorrect_answers': total_incorrect,
            'unknown_answers': total_unknown,
            'accuracy': f"{total_accuracy:.2%}",
            'gold_answer': '',
            'gold_count': '',
            'majority_answer': '',
            'majority_count': ''
        }
        
        # Add dict_answers column conditionally
        if show_dict_answers:
            total_row['dict_answers'] = {}
        
        # Add token count columns conditionally
        if show_token:
            total_row['avg_correct_tokens'] = ''
            total_row['avg_incorrect_tokens'] = ''
        summary_data.append(total_row)
    
    if HAS_PANDAS:
        return pd.DataFrame(summary_data)
    else:
        return summary_data

def save_detailed_results(results: List[Dict], output_file: str = "answer_analysis_detailed.csv"):
    """
    Save detailed analysis results to CSV file
    
    Args:
        results: List of analysis results
        output_file: Output filename
    """
    if HAS_PANDAS:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
    else:
        # Use CSV module if pandas is not available
        if results:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    print(f"Saved detailed analysis results to {output_file}")

def save_summary_results(summary_data, output_file: str = "answer_analysis_summary.csv"):
    """
    Save summary results to CSV file
    
    Args:
        summary_data: Summary data
        output_file: Output filename
    """
    if HAS_PANDAS and hasattr(summary_data, 'to_csv'):
        summary_data.to_csv(output_file, index=False, encoding='utf-8')
    elif isinstance(summary_data, list) and summary_data:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
    print(f"Saved summary results to {output_file}")

def create_human_readable_report(results: List[Dict], summary_data, 
                                output_file: str = "answer_analysis_report.txt"):
    """
    Create human-readable analysis report
    
    Args:
        results: List of analysis results
        summary_data: Summary data
        output_file: Output filename
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Answer Analysis Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        total_files = len(results)
        error_files = len([r for r in results if r.get('error')])
        valid_files = total_files - error_files
        
        f.write("■ Overall Statistics\n")
        f.write(f"  - Total files: {total_files}\n")
        f.write(f"  - Successfully processed files: {valid_files}\n")
        f.write(f"  - Error files: {error_files}\n\n")
        
        # Check if summary data exists
        summary_has_data = False
        if HAS_PANDAS and hasattr(summary_data, 'empty'):
            summary_has_data = not summary_data.empty
        elif isinstance(summary_data, list):
            summary_has_data = len(summary_data) > 0
        
        if summary_has_data:
            f.write("■ Summary by Dataset\n")
            if HAS_PANDAS and hasattr(summary_data, 'to_string'):
                f.write(summary_data.to_string(index=False))
            else:
                # Display for when pandas is not available
                for i, row in enumerate(summary_data):
                    if i == 0:
                        # Header
                        headers = list(row.keys())
                        f.write("  ".join(f"{h:<15}" for h in headers) + "\n")
                        f.write("-" * (15 * len(headers) + 2 * (len(headers) - 1)) + "\n")
                    # Data row
                    values = [str(row.get(h, '')) for h in headers]
                    f.write("  ".join(f"{v:<15}" for v in values) + "\n")
            f.write("\n\n")
            
            # Detailed statistics
            f.write("■ Details by Problem\n")
            if HAS_PANDAS and hasattr(summary_data, 'iterrows'):
                for _, row in summary_data.iterrows():
                    if row['dataset'] != 'total':  # Skip total row
                        f.write(f"  {row['dataset']} Problem {row['problem_num']}:\n")
                        f.write(f"    - Correct answer: {row['gold_answer']}\n")
                        f.write(f"    - Total answers: {row['total_answers']}\n")
                        f.write(f"    - Correct answers: {row['correct_answers']}\n")
                        f.write(f"    - Incorrect answers: {row['incorrect_answers']}\n")
                        f.write(f"    - Unknown answers: {row['unknown_answers']}\n")
                        f.write(f"    - Accuracy: {row['accuracy']}\n")
                        f.write(f"    - Gold answer frequency: {row['gold_count']}\n")
                        f.write(f"    - Most frequent answer: {row['majority_answer']} (frequency: {row['majority_count']})\n")
                        # Add answer frequency
                        if 'dict_answers' in row and row['dict_answers']:
                            f.write(f"    - Answer frequency: {dict(row['dict_answers'])}\n")
                        # Add token count information
                        if 'avg_correct_tokens' in row and 'avg_incorrect_tokens' in row:
                            f.write(f"    - Average correct answer tokens: {row['avg_correct_tokens']}\n")
                            f.write(f"    - Average incorrect answer tokens: {row['avg_incorrect_tokens']}\n")
                        f.write("\n")
            else:
                for row in summary_data:
                    if row['dataset'] != 'total':  # Skip total row
                        f.write(f"  {row['dataset']} Problem {row['problem_num']}:\n")
                        f.write(f"    - Correct answer: {row['gold_answer']}\n")
                        f.write(f"    - Total answers: {row['total_answers']}\n")
                        f.write(f"    - Correct answers: {row['correct_answers']}\n")
                        f.write(f"    - Incorrect answers: {row['incorrect_answers']}\n")
                        f.write(f"    - Unknown answers: {row['unknown_answers']}\n")
                        f.write(f"    - Accuracy: {row['accuracy']}\n")
                        f.write(f"    - Gold answer frequency: {row['gold_count']}\n")
                        f.write(f"    - Most frequent answer: {row['majority_answer']} (frequency: {row['majority_count']})\n")
                        # Add answer frequency
                        if 'dict_answers' in row and row['dict_answers']:
                            f.write(f"    - Answer frequency: {row['dict_answers']}\n")
                        # Add token count information
                        if 'avg_correct_tokens' in row and 'avg_incorrect_tokens' in row:
                            f.write(f"    - Average correct answer tokens: {row['avg_correct_tokens']}\n")
                            f.write(f"    - Average incorrect answer tokens: {row['avg_incorrect_tokens']}\n")
                        f.write("\n")
        
        # Error details
        if error_files > 0:
            f.write("■ Error Details\n")
            for result in results:
                if result.get('error'):
                    f.write(f"  - {result['filename']}: {result['error']}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("Analysis Complete\n")
        f.write("=" * 60 + "\n")
    
    print(f"Saved human-readable report to {output_file}")

def save_results_as_jsonl(summary_data, dataset: str, output_file: str = None):
    """
    Save analysis results as JSONL format
    
    Args:
        summary_data: Summary data (DataFrame or list)
        dataset: Dataset name
        output_file: Output filename (if None, auto-generate based on dataset and LLM name)
    """
    if output_file is None:
        llm_name = extract_llm_name_from_env()
        output_file = f"analysis_{dataset}_{llm_name}.jsonl"
    
    # Convert summary_data to list if it's a DataFrame
    if HAS_PANDAS and hasattr(summary_data, 'to_dict'):
        data_list = summary_data.to_dict('records')
    else:
        data_list = summary_data
    
    # Filter out the total row
    filtered_data = [row for row in data_list if row.get('dataset') != 'total']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in filtered_data:
            # Create JSONL entry with required fields
            gold_answer_raw = row.get('gold_answer')
            if dataset and dataset.lower() != 'gpqa_diamond' and gold_answer_raw is not None:
                gold_answer_out = normalize_math_notation(str(gold_answer_raw))
            else:
                gold_answer_out = gold_answer_raw
            jsonl_entry = {
                'problem_num': row.get('problem_num'),
                'total_answers': row.get('total_answers'),
                'answer_counts': row.get('dict_answers', {}),
                'gold_answer': gold_answer_out,
                'majority_answer': row.get('majority_answer'),
                'all_answers': row.get('all_answers', [])
            }
            
            # Write as JSON line
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
    
    print(f"Saved JSONL results to {output_file}")

def main():
    """Main execution function"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='BoN Answer Analysis Script - Analyzes answer files in the saved_answers directory')
    parser.add_argument('--dir', '-d', 
                       default='saved_answers',
                       help='Directory containing answer files (default: saved_answers)')
    parser.add_argument('--dataset', '-ds',
                       default='medrect',
                       help='Dataset name (default: medrect)')
    parser.add_argument('--parquet-path', '-p',
                       help='Path to correct answer data parquet file (default: auto-set based on dataset)')
    parser.add_argument('--show-dict-answers', 
                       action='store_true',
                       help='Display dict_answers column (default: False)')
    parser.add_argument('--show-token', '--show_token', 
                       action='store_true',
                       help='Show average token count for correct and incorrect answers per problem (default: False)')
    
    args = parser.parse_args()
    
    print(f"Starting BoN Answer Analysis Script...")
    print(f"Target directory: {args.dir}")
    print(f"Dataset: {args.dataset}")
    if args.show_token:
        print("Token analysis enabled")
    
    # Load correct answer data
    global ANSWERS_DATA
    ANSWERS_DATA[args.dataset] = load_answers_data(args.dataset, args.parquet_path)
    
    # Analyze answer files (only for specified dataset)
    results = analyze_all_answers(args.dir, args.dataset)
    
    if not results:
        print("No analysis target files found")
        return
    
    # Create summary table for display (based on show_dict_answers and show_token flags)
    summary_data_display = create_summary_table(results, args.show_dict_answers, args.show_token)
    
    # Create summary table for JSONL output (always include dict_answers)
    summary_data_jsonl = create_summary_table(results, True, False)
    
    # Display results
    print("\n" + "=" * 60)
    print("Analysis Results Summary")
    print("=" * 60)
    
    # Process based on whether it's a DataFrame or list
    has_data = False
    if HAS_PANDAS and hasattr(summary_data_display, 'empty'):
        # For pandas DataFrame
        has_data = not summary_data_display.empty
    elif isinstance(summary_data_display, list):
        # For list
        has_data = len(summary_data_display) > 0
    
    if has_data:
        if HAS_PANDAS and hasattr(summary_data_display, 'to_string'):
            # Adjust pandas display settings
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.width', None)
            print(summary_data_display.to_string(index=False))
            # Reset settings
            pd.reset_option('display.max_columns')
            pd.reset_option('display.max_colwidth')
            pd.reset_option('display.width')
        else:
            # Display for when pandas is not available
            for i, row in enumerate(summary_data_display):
                if i == 0:
                    # Header
                    headers = list(row.keys())
                    print("  ".join(f"{h:<15}" for h in headers))
                    print("-" * (15 * len(headers) + 2 * (len(headers) - 1)))
                # Data row
                values = [str(row.get(h, '')) for h in headers]
                print("  ".join(f"{v:<15}" for v in values))
    else:
        print("No valid analysis results")
    
    # Save files
    save_detailed_results(results)
    if has_data:
        save_summary_results(summary_data_display)
        # Save JSONL results (always with dict_answers)
        save_results_as_jsonl(summary_data_jsonl, args.dataset)
    create_human_readable_report(results, summary_data_display)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 