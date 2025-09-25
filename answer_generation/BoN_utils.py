# evaluator.py
import json, os, asyncio
from functools import partial
from openai import OpenAI
from vllm import LLM, SamplingParams
from typing import List, Tuple, Dict, OrderedDict, Any
import requests
import hashlib
import subprocess
import concurrent.futures
import threading

from dotenv import load_dotenv
# Load .env file
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
load_dotenv()
TOP_K = int(os.getenv("TOP_K", "20"))

# Get LLM model paths from environment variables
LLM_MODEL_PORT_8000 = os.getenv("LLM_MODEL_PORT_8000", "/workspace/Qwen3-4B")
LLM_MODEL_PORT_8100 = os.getenv("LLM_MODEL_PORT_8100", "/workspace/Qwen3-4B")
LLM_MODEL_PORT_8200 = os.getenv("LLM_MODEL_PORT_8200", "/workspace/LLaMA-Factory/output/qwen3_4B_lora_sft_reduction/")

def extract_reward_model_name(reward_model_path):
    """Extract name from reward model path"""
    if not reward_model_path:
        return "reward"
    
    # Get filename part from path
    model_name = os.path.basename(reward_model_path.strip())
    
    # Use as-is without splitting by hyphens (modification: include everything)
    # Previously: Get first part split by hyphens (e.g. ArmoRM-Llama3-8B-v0.1 → ArmoRM)
    # Currently: Use entire model name including hyphens (e.g. ArmoRM-Llama3-8B-v0.1 → ArmoRM-Llama3-8B-v0.1)
    
    # If first character is not uppercase, get up to first uppercase or first 8 characters
    if model_name and not model_name[0].isupper():
        # Find first uppercase character
        for i, char in enumerate(model_name):
            if char.isupper():
                model_name = model_name[i:]
                break
        else:
            # If no uppercase found, use first 8 characters
            model_name = model_name[:8]
    
    return model_name if model_name else "reward"

def grade(question: str, reference: str, prediction: str) -> dict:    
    # --- Prompt template (structured as much as possible) --------------------------
    SYSTEM_PROMPT = (
        "You are a strict grader for short math word problems (GSM8K style). "
        "Return ONLY a JSON dictionary with keys:\n"
        "  result  : \"correct\" or \"incorrect\"\n"
        "  feedback: short explanation (<=20 words)\n"
        "No additional keys, no markdown."
    )

    USER_TEMPLATE = """\
    Problem:
    {question}

    Reference answer:
    {reference}

    Student answer:
    {prediction}

    Is the student answer mathematically equivalent to the reference? Only check whether the final answer number matches the reference or not.\
    """
        
    MODEL_NAME = "gpt-4o-mini"        # Any GPT-4 series model
    client = OpenAI()
    
    res = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=question, reference=reference, prediction=prediction)}
        ],
    )
    return json.loads(res.choices[0].message.content)


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
    params = {"attr": "score"}   # Can also be changed to "first" or "mean"

    print("get_score")
    print(f"prompt = {prompt}")
    print(f"response = {response}")

    res = requests.post(URL, json=payload, params=params, timeout=30)
    res.raise_for_status()
    return res.json()['score']

def get_scores(
    prompts: List[str],
    responses: List[str]
) -> List[float]:
    return [get_score(prompt, response) for prompt, response in zip(prompts, responses)]


import os, textwrap
import openai                     # pip install openai>=1.0
import time
import glob
import random

def check_format(raw_answer: str) -> bool:
    return (raw_answer.find("[END]") != -1) and (raw_answer.find("ERROR") == -1) 

def remove_after_end(raw_answer: str) -> str:
    return raw_answer.split("[END]")[0] # remove everything after [END]


from typing import List, Tuple
from vllm import LLM, SamplingParams


def get_available_gpu_count() -> int:
    """
    Function to get the number of available GPUs
    
    Returns
    -------
    int
        Number of available GPUs
    """
    try:
        # Get GPU count using nvidia-smi command
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        gpu_count = len(result.stdout.strip().split('\n'))
        print(f"Detected {gpu_count} GPUs")
        return gpu_count
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Could not detect GPUs via nvidia-smi, defaulting to 1 GPU")
        return 1

def generate_answer_client_single_port(
    prompt: str,
    max_new_tokens: int,
    n: int,
    port: int,
    temperature: float = TEMPERATURE,
    grammar: str | None = None,
    finetuned_llm: bool = False,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = TOP_K,
    logprobs: int = 0,
    need_token_ids: bool = False,
    max_retries: int = 1,
    timeout: int = 1200,
    add_generation_prompt: bool = True
) -> List[Tuple[str, int]]:
    """
    generate_answer_client for single port (internal use)
    """
    return generate_answer_client(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        n=n,
        temperature=temperature,
        grammar=grammar,
        finetuned_llm=finetuned_llm,
        port=port,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        logprobs=logprobs,
        need_token_ids=need_token_ids,
        max_retries=max_retries,
        timeout=timeout,
        add_generation_prompt=add_generation_prompt
    )

def generate_answer_client_multiple_gpus(
    prompt: str,
    max_new_tokens: int,
    n: int = 1,
    temperature: float = TEMPERATURE,
    grammar: str | None = None,
    finetuned_llm: bool = False,
    base_port: int = 8100,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = TOP_K,
    logprobs: int = 0,
    need_token_ids: bool = False,
    max_retries: int = 1,
    timeout: int = 1200,
    add_generation_prompt: bool = True
) -> List[Tuple[str, int]]:
    """
    generate_answer_client supporting multiple GPUs
    Gets GPU count from environment variable NUM_GPU and distributes generation requests across multiple ports
    
    Parameters
    ----------
    prompt : str
        Prompt for generation
    max_new_tokens : int
        Maximum number of tokens to generate
    n : int
        Number of answers to generate
    base_port : int
        Base port number (default: 8100)
    Other parameters are the same as generate_answer_client
        
    Returns
    -------
    List[Tuple[str, int]]
        List of tuples of generated answers and token counts
    """
    # Get GPU count from environment variable
    gpu_count = int(os.getenv("NUM_GPU", "1"))
    
    # Change base port to 8200 for finetuned_llm
    if finetuned_llm:
        base_port = 8200
    
    # Calculate number of answers to distribute to each GPU
    answers_per_gpu = n // gpu_count
    remaining_answers = n % gpu_count
    
    print(f"🚀 [MULTI-GPU REQUEST] Distributing {n} answers across {gpu_count} GPUs (ports {base_port}-{base_port + gpu_count - 1})")
    print(f"🚀 [MULTI-GPU REQUEST] Base answers per GPU: {answers_per_gpu}, remaining: {remaining_answers}")
    
    # Prepare generation tasks for each GPU
    tasks = []
    for gpu_idx in range(gpu_count):
        port = base_port + gpu_idx
        # Add remaining answers to first GPU
        n_for_this_gpu = answers_per_gpu + (1 if gpu_idx < remaining_answers else 0)
        
        if n_for_this_gpu > 0:
            tasks.append((
                prompt, max_new_tokens, n_for_this_gpu, port,
                temperature, grammar, finetuned_llm, top_p, min_p, top_k,
                logprobs, need_token_ids, max_retries, timeout, add_generation_prompt
            ))
            print(f"🚀 [MULTI-GPU REQUEST] GPU {gpu_idx} (PORT {port}): {n_for_this_gpu} answers")
    
    # Parallel execution
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:
        # Submit tasks to each GPU
        future_to_gpu = {
            executor.submit(generate_answer_client_single_port, *task): i 
            for i, task in enumerate(tasks)
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_gpu):
            gpu_idx = future_to_gpu[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"GPU {gpu_idx} completed: {len(results)} answers")
            except Exception as exc:
                print(f"GPU {gpu_idx} generated an exception: {exc}")
    
    print(f"Total answers generated: {len(all_results)}")
    return all_results

def generate_answer_client(
    prompt: str,
    max_new_tokens: int,
    prompts: List[str] | None = None, # When passing multiple prompts
    n: int = 1,                         # ← Number of desired candidates
    temperature: float = TEMPERATURE,
    grammar: str | None = None,
    finetuned_llm: bool = False,
    port: int | None = None,            # Port specification
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = TOP_K,
    logprobs: int = 0,
    need_token_ids: bool = False,
    max_retries: int = 1,              # Retry count
    timeout: int = 12000,                 # Timeout duration (seconds)
    add_generation_prompt: bool = True
) -> List[Tuple[str, int]]:  # Change return type
    
    # Use specified port if available, otherwise use default port
    if port is not None:
        base_url = f"http://localhost:{port}/v1"
        # Select model based on port
        if port == 8000:
            model_lm = LLM_MODEL_PORT_8000
        elif port == 8100:
            model_lm = LLM_MODEL_PORT_8100
        elif port == 8200:
            model_lm = LLM_MODEL_PORT_8200
        else:
            # Default to 8100 model
            model_lm = LLM_MODEL_PORT_8100
        print(f"🚀 [LLM REQUEST] Using PORT {port} with model: {model_lm}")
    elif finetuned_llm:
        port = 8200
        base_url = "http://localhost:8200/v1"
        model_lm = LLM_MODEL_PORT_8200
        print(f"🚀 [LLM REQUEST] Using FINETUNED PORT {port} with model: {model_lm}")
    else:
        port = 8100
        base_url = "http://localhost:8100/v1"
        model_lm = LLM_MODEL_PORT_8100
        print(f"🚀 [LLM REQUEST] Using DEFAULT PORT {port} with model: {model_lm}")
    
    client = OpenAI(
        base_url=base_url,
        api_key="dummy",                       # Anything is fine if --api-key is not specified
        timeout=timeout                        # Timeout setting
    )

    print(f"n = {n}")
    results = []
    for i in range(n):
        for retry in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model = model_lm,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    top_p=top_p,
                    extra_body={
                        "guided_grammar": grammar,
                        "min_p": min_p,
                        "top_k": top_k,
                        "add_generation_prompt": add_generation_prompt,
                    } if grammar is not None else {
                        "min_p": min_p,
                        "top_k": top_k,
                        "add_generation_prompt": add_generation_prompt,
                    },
                    stream=False
                )
                content = resp.choices[0].message.content
                token_count = resp.usage.completion_tokens  # Get token count (len() is not needed)
                results.append((content, token_count))  # Return as tuple
                break  # Exit loop on success
            except Exception as e:
                if retry == max_retries - 1:  # Last retry failed
                    print(f"Error after {max_retries} retries: {str(e)}")
                    # Return empty response for timeout errors
                    if isinstance(e, openai.APITimeoutError):
                        print("Timeout error occurred. Returning empty response.")
                        results.append(("", 0))
                        break
                    raise  # Re-raise other errors
                print(f"Attempt {retry + 1} failed: {str(e)}. Retrying...")
        else: # When retry count exceeds max_retries
            print(f"Max retries reached. Returning empty response.")
            results.append(("", 0))
            break
    return results

def generate_answer_client_verbose(
    prompt: str,
    max_new_tokens: int,
    prompts: List[str] | None = None, # When passing multiple prompts
    n: int = 1,                         # ← Number of desired candidates
    temperature: float = TEMPERATURE,
    grammar: str | None = None,
    finetuned_llm: bool = False,
    port: int | None = None,            # Port specification
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = TOP_K,
    logprobs: int = 20,                 # Get top-20 log probabilities
    need_token_ids: bool = True,        # Set to True to get token IDs
    max_retries: int = 1,              # Number of retries
    timeout: int = 12000,                 # Timeout duration (seconds)
    add_generation_prompt: bool = True,
    chat_template: str | None = None
) -> List[Tuple[str, int, List[int], List[str], List[List[Dict[str, Any]]]]]:  # Return type changed: (content, token_count, token_ids, token_strings, step_logprobs)
    
    # Use specified port if available, otherwise use default port
    if port is not None:
        base_url = f"http://localhost:{port}/v1"
        # Select model based on port
        if port == 8000:
            model_lm = LLM_MODEL_PORT_8000
        elif port == 8100:
            model_lm = LLM_MODEL_PORT_8100
        elif port == 8200:
            model_lm = LLM_MODEL_PORT_8200
        else:
            # Default to 8100 model
            model_lm = LLM_MODEL_PORT_8100
        print(f"🚀 [LLM REQUEST VERBOSE] Using PORT {port} with model: {model_lm}")
    elif finetuned_llm:
        port = 8200
        base_url = "http://localhost:8200/v1"
        model_lm = LLM_MODEL_PORT_8200
        print(f"🚀 [LLM REQUEST VERBOSE] Using FINETUNED PORT {port} with model: {model_lm}")
    else:
        port = 8100
        base_url = "http://localhost:8100/v1"
        model_lm = LLM_MODEL_PORT_8100
        print(f"🚀 [LLM REQUEST VERBOSE] Using DEFAULT PORT {port} with model: {model_lm}")
    
    client = OpenAI(
        base_url=base_url,
        api_key="dummy",                       # Anything is fine if --api-key is not specified
        timeout=timeout                        # Timeout setting
    )

    print(f"n = {n}")
    results = []
    for i in range(n):
        for retry in range(max_retries):
            try:
                extra_body_params = {
                    "min_p": min_p,
                    "top_k": top_k,
                    "add_generation_prompt": add_generation_prompt,
                }
                
                if grammar is not None:
                    extra_body_params["guided_grammar"] = grammar
                    
                if chat_template is not None:
                    extra_body_params["chat_template"] = chat_template
                
                # Add system prompt if mistral is included
                # https://huggingface.co/mistralai/Magistral-Small-2506#vllm-recommended
                messages = [{"role": "user", "content": prompt}]
                print(f"LLM_MODEL_PORT_8100 = {LLM_MODEL_PORT_8100}")
                if "mistral" in LLM_MODEL_PORT_8100.lower() or "magistral" in LLM_MODEL_PORT_8100.lower():
                    print(f"mistral template")
                    system_prompt = \
"""A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown and Latex to format your response. Write both your thoughts and summary in the same language as the task posed by the user.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user.

Problem:"""
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                elif "nemotron" in LLM_MODEL_PORT_8100.lower():
                    print(f"nemotron template")
                    system_prompt = "/think"
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

                resp = client.chat.completions.create(
                    model = model_lm,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    top_p=top_p,
                    logprobs=True,                     # Enable logprobs
                    top_logprobs=logprobs,            # Get token information
                    extra_body=extra_body_params,
                    stream=False,
                )
                content = resp.choices[0].message.content
                
                # Check if this is gpt-oss model and if reasoning_content is available
                reasoning_content = ""
                if "gpt-oss" in LLM_MODEL_PORT_8100.lower():
                    # Try to get reasoning_content from the response
                    if hasattr(resp.choices[0].message, 'reasoning_content') and resp.choices[0].message.reasoning_content:
                        reasoning_content = resp.choices[0].message.reasoning_content
                        # Format the final response with reasoning content
                        content = f"<think>{reasoning_content}</think>\n{content}"
                        print(f"Found reasoning_content for gpt-oss model")
                    else:
                        print(f"No reasoning_content found in gpt-oss response")
                
                token_count = resp.usage.completion_tokens  # Get token count
                print(f"token_count = {token_count}")  # Token count
                
                # Get list of token IDs and top-20 log probabilities for each step
                token_ids = []
                token_strings = []  # Store string for each token
                step_logprobs = []  # Top-20 log probability information for each step
                
                print(f"len(resp.choices[0].logprobs.content) = {len(resp.choices[0].logprobs.content)}")
                if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                    for token_info in resp.choices[0].logprobs.content:
                        # Get token string
                        token_string = getattr(token_info, 'token', '')
                        token_strings.append(token_string)
                        
                        # Get ID of actually selected token
                        if hasattr(token_info, 'token_id'):
                            token_ids.append(token_info.token_id)
                        elif hasattr(token_info, 'bytes'):
                            # Estimate token ID from byte information (fallback)
                            bytes_data = token_info.bytes
                            if isinstance(bytes_data, list):
                                # Convert to tuple then hash for lists
                                token_ids.append(hash(tuple(bytes_data)) % 100000)
                            elif isinstance(bytes_data, (str, bytes)):
                                token_ids.append(hash(bytes_data) % 100000)
                            else:
                                # For other cases, stringify then hash
                                token_ids.append(hash(str(bytes_data)) % 100000)
                        
                        # Get log probabilities of top-20 candidates for each step
                        step_candidates = []
                        if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
                            for candidate in token_info.top_logprobs:
                                candidate_info = {
                                    'token': candidate.token,
                                    'logprob': candidate.logprob,
                                    'bytes': getattr(candidate, 'bytes', None)
                                }
                                # Add token ID if available
                                if hasattr(candidate, 'token_id'):
                                    candidate_info['token_id'] = candidate.token_id
                                elif hasattr(candidate, 'bytes') and candidate.bytes:
                                    # Estimate token ID from byte information
                                    bytes_data = candidate.bytes
                                    if isinstance(bytes_data, list):
                                        # Convert to tuple then hash for lists
                                        candidate_info['token_id'] = hash(tuple(bytes_data)) % 100000
                                    elif isinstance(bytes_data, (str, bytes)):
                                        candidate_info['token_id'] = hash(bytes_data) % 100000
                                    else:
                                        # For other cases, stringify then hash
                                        candidate_info['token_id'] = hash(str(bytes_data)) % 100000
                                
                                step_candidates.append(candidate_info)
                        
                        step_logprobs.append(step_candidates)
                    #for i, step_logprob in enumerate(step_logprobs):
                    #    print(f"step_logprobs[{i}] = {step_logprob}")
                    #    print(f"step_logprobs[{i}] logprobs = {step_logprob}")
                    #print(f"len(step_logprobs = {len(step_logprobs)}")
                    total_width = [len(step_logprobs[i]) for i in range(len(step_logprobs))]
                    #print(f"average width = {sum(total_width) / len(total_width)}")
                
                results.append((content, token_count, token_ids, token_strings, step_logprobs))  # Return as tuple
                break  # Break loop on success
            except Exception as e:
                if retry == max_retries - 1:  # Last retry failed
                    print(f"Error after {max_retries} retries: {str(e)}")
                    # Return empty response for timeout errors
                    if isinstance(e, openai.APITimeoutError):
                        print("Timeout error occurred. Returning empty response.")
                        results.append(("", 0, [], [], []))
                        break
                    raise  # Re-raise other errors
                print(f"Attempt {retry + 1} failed: {str(e)}. Retrying...")
        else: # When retry count exceeds max_retries
            print(f"Max retries reached. Returning empty response.")
            results.append(("", 0, [], [], []))
            break
    return results

def decompose_answer_by_double_newline(answer: str, token_ids: List[int], token_strings: List[str], step_logprobs: List[List[Dict[str, any]]], single_newline_id: int, double_newline_id: int) -> List[str]:
    blocks = []
    current_string = ""
    current_step_logprobs = []
    if len(token_strings) != len(token_ids) or len(token_strings) != len(step_logprobs):
        raise ValueError("token_strings, token_ids, and step_logprobs must have the same length")
    
    i = 0
    cut = False
    while i < len(token_ids):
        if cut:
            print(f"tokenstr[{i}] = {repr(token_strings[i])}, token_id = {token_ids[i]}")
        
        # Newline detection based on token ID
        if double_newline_id and token_ids[i] == double_newline_id:
            # Double newline token (\n\n)
            print(f"double_newline cut")
            blocks.append((current_string, current_step_logprobs))
            current_string = ""
            current_step_logprobs = []
            i = i + 1
            cut = True
        elif (i < len(token_ids) - 1 and 
              token_ids[i] == single_newline_id and 
              token_ids[i+1] == single_newline_id):
            # Consecutive single newline tokens (\n + \n)
            print(f"single_newline cut")
            blocks.append((current_string, current_step_logprobs))
            current_string = ""
            current_step_logprobs = []
            i = i + 2
            cut = True
        else:
            # Normal token
            current_string = current_string + token_strings[i]
            current_step_logprobs.append(step_logprobs[i])
            i = i + 1
            cut = False
    
    # Add the last block
    blocks.append((current_string, current_step_logprobs))
    return blocks

def obtain_confidence_blocks(blocks):
    confidences = []
    for block_idx, block in enumerate(blocks):
        sum_confidence = 0
        for step_idx, step_logprob in enumerate(block[1]):
            try:
                # Check if step_logprob is in expected format (list of dictionaries)
                if isinstance(step_logprob, list) and len(step_logprob) > 0 and isinstance(step_logprob[0], dict):
                    logprob_list = [x['logprob'] for x in step_logprob]
                    sum_confidence += KL_finite(p = logprob_list)
                else:
                    print(f"Warning: Unexpected step_logprob format at block {block_idx}, step {step_idx}")
                    print(f"Type: {type(step_logprob)}, Content: {step_logprob}")
                    # Use default value
                    sum_confidence += 0.0
            except Exception as e:
                print(f"Error processing block {block_idx}, step {step_idx}: {e}")
                print(f"step_logprob type: {type(step_logprob)}")
                print(f"step_logprob content: {step_logprob}")
                # Use default value
                sum_confidence += 0.0
        average_confidence = sum_confidence / (len(block[1])+1)
        confidences.append(average_confidence)
    return confidences

# Re-generate with n=1 for now
def regenerate_answer_client(
    prompt: str,
    original_answer: str,
    rethink_step: int,
    max_new_tokens: int,
    temperature: float = TEMPERATURE,
    grammar: str = None,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = TOP_K,
    logprobs: int = 0,
    need_token_ids: bool = False
) -> str:
    parsed = None # TemplateParser().parse(original_answer)
    if rethink_step > len(parsed.steps): # rethink final answer
        final_answer = parsed.final_answer
        print(f"rethinking final answper:")
        print(f"original_answer = {original_answer}")
        print(f"final_answer = {final_answer}")
        prompt_rethink = 'Wait, the reasoning following "' + "Final Answer: " + final_answer + '" might contain an error or could be improved. Please review and revise from that point onward. Output the COMPLETE REVISED ANSWER by step by step.\n'
        prompt_concat = prompt + original_answer + prompt_rethink
        result = generate_answer_client(prompt=prompt_concat, max_new_tokens=max_new_tokens, temperature=temperature, grammar=grammar, top_p=top_p, min_p=min_p, top_k=top_k, logprobs=logprobs, need_token_ids=need_token_ids)[0]
        return result[0]  # Return only the answer from tuple
    else: # rethink internal step
        rethinking_sentence = parsed.steps[rethink_step]
        prompt_rethink = 'Wait, the reasoning after "' + "Step " + str(rethink_step) + ": " + rethinking_sentence + '" might contain an error or could be improved. Please review and revise from that point onward. Output the COMPLETE REVISED ANSWER by step by step.\n'
        prompt_concat = prompt + original_answer + prompt_rethink
        result = generate_answer_client(prompt=prompt_concat, max_new_tokens=max_new_tokens, temperature=temperature, grammar=grammar, top_p=top_p, min_p=min_p, top_k=top_k, logprobs=logprobs, need_token_ids=need_token_ids)[0]
        return result[0]  # Return only the answer from tuple


import re
def split_steps_and_answer(text: str) -> Tuple[Dict[int, str], str]:
    """
    Extract from given string:
      • Each Step number and content
      • Final Answer
    and return.

    Parameters
    ----------
    text : str
        String containing "Step 1: ..." or "Final Answer: ..."

    Returns
    -------
    steps : Dict[int, str]
        Ordered dictionary of {step number: content}
    final_answer : str
        Content of Final Answer (empty string if not found)
    """
    step_pat   = re.compile(r'^Step\s+(\d+):\s*(.*)', re.I)
    final_pat  = re.compile(r'^Final\s+Answer:\s*(.*)', re.I)

    steps: Dict[int, str] = OrderedDict()
    final_answer = False
    current_num, buffer = None, []

    for line in text.splitlines():
        # Ignore leading whitespace
        line = line.strip()
        if not line:
            continue

        m_step  = step_pat.match(line)
        m_final = final_pat.match(line)

        if m_step:                         # New Step started
            # Finalize the previous Step
            if current_num is not None:
                steps[current_num] = ' '.join(buffer).strip()
            current_num = int(m_step.group(1))
            buffer      = [m_step.group(2)]
        elif m_final:                      # Reached Final Answer
            if current_num is not None:
                steps[current_num] = ' '.join(buffer).strip()
                current_num = None
                buffer = []
            final_answer = m_final.group(1).strip()
        elif current_num is not None:      # Line continuing Step content
            buffer.append(line)
            
    if list(steps.keys()) != list(range(1, len(steps) + 1)):
        print(f"Non-sequential steps detected: {steps} - rearranging steps")
        steps_revised: Dict[int, str] = OrderedDict()
        for i,key in enumerate(sorted(steps.keys())):
            steps_revised[i+1] = steps[key]
        steps = steps_revised

    return steps, final_answer

def get_hash(token_ids: List[int]) -> str:
    """
    Function to calculate SHA256 hash value from list of token IDs

    Parameters
    ----------
    token_ids : List[int]
        List of token IDs to calculate hash value for

    Returns
    -------
    str
        Calculated hash value (hexadecimal string)
    """
    # Convert list of token IDs to string
    token_string = "".join(map(str, token_ids))
    # Create SHA256 hash object
    sha256 = hashlib.sha256()
    # UTF-8 encode string and update hash
    sha256.update(token_string.encode('utf-8'))
    # Get hash value as hexadecimal string
    return sha256.hexdigest()

def step_identical(step_answer1: str, step_answer2: str) -> bool:
    """
    Function to check whether two steps are the same
    """
    comparator = StepComparator()
    are_equivalent, similarity_score, explanation = comparator.compare_steps(
        step_answer1, step_answer2
    )
    return are_equivalent


def generate_structured_prompt(problem_text: str) -> str:
    template = f"""You are a concise math problem-solver.  
Respond ONLY in the format shown.  
After "Integration:" output "[END]" and stop.  
Deviation ⇒ reply "[ERROR]"

User:
You do not need to solve the problem itself. Instead, decompose the problem into key components. 
Each component should be solved by another AI agent. 
Instead, you need to identify the key components of the problem. 
Finally, you need to integrate all components into a single equation. 
The integration can use results of each component, but you do not need to solve each component.
The number of components should be as 5 or less, and each component should be solved independently based on the answer of the previous components.

Problem: {problem_text.strip()}

Format (strict):
Component 1: <first reasoning step. Define the variable(s) that describes the solution of the component.>
Component 2: <second reasoning step. Define the variable(s) that describes the solution of the component.>
...
Integration: <equation that integrates all components>
[END]

"""    
    return template

def generate_structured_prompt_reduction(problem_text: str, dataset_type: str = "auto") -> str:
    #template = f"""Reduce the original problem into a simpler equivalent problem by using one reduction step. Problem: {problem_text.strip()}. The reduced problem that leads to the same answer is:"""  
    template = f"""<|system|>
You are a helpful assistant specialized in algebraic problem reduction.
Every time you reply, you MUST follow these rules exactly:
1. Your output must contain two tags, in this exact order:
   <thought>...</thought>
   <reduced_problem>...</reduced_problem>
3. Inside <thought> put only your internal reasoning. No other text or tags.
4. Inside <reduced_problem> put only the fully self-contained reduced problem statement (omit the answer).
5. You must not output anything before <thought> or after </reduced_problem>.
6. If you cannot comply, output exactly: ERROR: format violation (and nothing else).
<|endofsystem|>

<|user|>
The original problem statement is: {problem_text.strip()}
Create a simplified but equivalent problem that leads to the same answer. Use the same mathematical format as the original problem statement."""
    
    if dataset_type == "mmlu_pro":
        template += " For multiple choice questions, make sure to preserve the answer options and indicate that the final answer should be the letter choice (A), (B), (C), etc."
    elif dataset_type == "gpqa_diamond":
        template += " For multiple choice questions, make sure to preserve the answer options and indicate that the final answer should be the letter choice (A), (B), (C), etc."
    
    template += """
<|endofuser|>"""  
    return template

import math
import numpy as np

def KL_finite(p, q=None):
    """
    Function to normalize token probability distribution and calculate KL divergence with uniform distribution
    
    Parameters
    ----------
    p : List[float]
        List of log probabilities (logprobs)
    q : List[float], optional
        Comparison probability distribution. Use uniform distribution if None
        
    Returns
    -------
    float
        KL(P||Q) divergence value
    """
    if not p:
        return float('inf')  # Return infinity for empty lists
    
    # Convert log probabilities to probabilities (exp)
    probs = [math.exp(logprob) for logprob in p]
    
    # Normalize probability distribution (make sum equal to 1)
    total_prob = sum(probs)
    if total_prob == 0:
        return float('inf')  # When all probabilities are 0
    
    normalized_probs = [prob / total_prob for prob in probs]
    
    # Set comparison distribution q (default is uniform distribution)
    if q is None:
        # Uniform distribution over 20 tokens
        uniform_prob = 1.0 / len(normalized_probs)
        q = [uniform_prob] * len(normalized_probs)
    
    # Epsilon for numerical stability
    epsilon = 1e-10
    
    # Calculate KL divergence KL(P||Q) = Σ P(i) * log(P(i) / Q(i))
    kl_divergence = 0.0
    for p_i, q_i in zip(normalized_probs, q):
        # Add epsilon when probability is close to 0 to maintain numerical stability
        p_i = max(p_i, epsilon)
        q_i = max(q_i, epsilon)
        
        kl_divergence += p_i * math.log(p_i / q_i)
    
    return kl_divergence

def calculate_entropy(logprobs):
    """
    Helper function to calculate entropy from list of log probabilities
    
    Parameters
    ----------
    logprobs : List[float]
        List of log probabilities
        
    Returns
    -------
    float
        Entropy value H(P) = -Σ P(i) * log(P(i))
    """
    if not logprobs:
        return 0.0
    
    # Convert log probabilities to probabilities and normalize
    probs = [math.exp(logprob) for logprob in logprobs]
    total_prob = sum(probs)
    #print(f"total_prob = {total_prob}")
    if total_prob == 0:
        return 0.0
    
    normalized_probs = [prob / total_prob for prob in probs]
    
    # Calculate entropy
    epsilon = 1e-10
    entropy = 0.0
    for p_i in normalized_probs:
        if p_i > epsilon:
            entropy -= p_i * math.log(p_i)
    
    return entropy


def string_to_tokens_not_working(text: str) -> List[int]:
    """
    Function to convert string to list of token IDs

    Parameters
    ----------
    text : str
        String to tokenize

    Returns
    -------
    List[int]
        List of token IDs
    """
    # Method 1: Try /tokenize endpoint
    URL = "http://localhost:8100/tokenize"
    payload = {"text": text}
    
    try:
        res = requests.post(URL, json=payload, timeout=300)  # Set to 5 minutes
        res.raise_for_status()
        return res.json()['tokens']
    except requests.RequestException as e:
        print(f"Tokenize endpoint failed: {e}")
        
    # Method 2: Use /v1/completions + echo=True for tokenization
    try:
        client = OpenAI(
            base_url="http://localhost:8100/v1",
            api_key="dummy",
            timeout=300  # Set to 5 minutes
        )
        
        resp = client.completions.create(
            model="/workspace/Qwen3-4B",
            prompt=text,
            max_tokens=0,  # Don't generate
            temperature=0,
            logprobs=1,    # For getting token information
            echo=True,     # Return including prompt
        )
        
        # Estimate token IDs from token strings
        # (vLLM may not be able to directly get token_ids)
        tokens = resp.choices[0].logprobs.tokens
        # Simply generate IDs based on hash
        token_ids = [hash(token) % 100000 for token in tokens]
        return token_ids
        
    except Exception as e:
        print(f"Completions-based tokenization failed: {e}")
        
    # Method 3: Fallback to local tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        print(f"Local tokenizer failed: {e}")
        # Last resort: character-based pseudo-tokenization
        return [hash(char) % 100000 for char in text]


# -----------------------------------------------------------------------------
# Utility: score existing text via /v1/completions + echo=True (fast)
# -----------------------------------------------------------------------------

def score_text_client_echo(
    prompt: str,
    generated_text: str,
    logprobs: int = 20,
    timeout: int = 600,
) -> Tuple[str, int, List[int], List[str], List[List[Dict[str, any]]], float]:
    """Fast per-token log-probability computation using the *completions* route.

    This leverages ``echo=True`` so the endpoint returns logprobs for *every* token
    in ``prompt + generated_text`` without generating new tokens.

    Returns
    -------
    (text, n_tokens, token_ids, tokens, topk_logprobs, avg_logprob)
    where ``topk_logprobs[i]`` is a list[dict] of the *top-k* candidates when the
    *i*-th token was chosen.
    """

    client = OpenAI(
        base_url="http://localhost:8100/v1",  # Specified LLM server
        api_key="dummy",
        timeout=timeout,
    )
    print(f"🚀 [LLM ECHO REQUEST] Using PORT 8100 with model: {LLM_MODEL_PORT_8100}")

    full_text = prompt + generated_text

    resp = client.completions.create(
        model=LLM_MODEL_PORT_8100,  # Use 8100 port model by default
        prompt=full_text,
        max_tokens=0,                 # *** generate nothing ***
        temperature=0,
        logprobs=logprobs,
        echo=True,
    )
    


    lp = resp.choices[0].logprobs  # type: ignore

    # Get information directly from vLLM response
    all_tokens         = lp.tokens
    all_token_logprobs = lp.token_logprobs  
    all_top_logprobs   = lp.top_logprobs
    
    #print(f"[DEBUG] Total tokens in response: {len(all_tokens)}")
    #print(f"[DEBUG] First 10 tokens: {all_tokens[:10]}")
    
    # Calculate prompt length (token count based)
    resp_base = client.completions.create(
        model=LLM_MODEL_PORT_8100,  # Use 8100 port model by default
        prompt=prompt,
        max_tokens=0,                 # *** generate nothing ***
        temperature=0,
        logprobs=logprobs,
        echo=True,
    )
    prompt_token_ids = resp_base.choices[0].logprobs.tokens
    prompt_len = len(prompt_token_ids)
    
    #print(f"[DEBUG] Prompt token count: {prompt_len}")
    
    # Extract generated_text part from response
    if len(all_tokens) > prompt_len:
        # Get from prompt onwards from response token array
        tokens = all_tokens[prompt_len:]
        token_logprobs = all_token_logprobs[prompt_len:]
        top_logprobs = all_top_logprobs[prompt_len:]
        
        # Get token IDs similarly (calculated by string_to_tokens)
        full_token_ids = all_tokens #resp.choices[0].logprobs.tokens #string_to_tokens(full_text)
        token_ids = full_token_ids[prompt_len:]
        
        # Match lengths (match to response token count)
        min_len = min(len(tokens), len(token_logprobs), len(top_logprobs))
        tokens = tokens[:min_len]
        token_logprobs = token_logprobs[:min_len] 
        top_logprobs = top_logprobs[:min_len]
        token_ids = token_ids[:min_len]
        
        #print(f"[DEBUG] Extracted {min_len} tokens: {tokens}")
        #print(f"[DEBUG] Token IDs: {token_ids}")
        
    else:
        # When prompt is longer (abnormal)
        print(f"[WARNING] Response shorter than prompt. Response: {len(all_tokens)}, Prompt: {prompt_len}")
        tokens = []
        token_ids = []
        token_logprobs = []
        top_logprobs = []
    step_logprobs = []
    for i, top_logprob in enumerate(top_logprobs):
        if len(top_logprob) == logprobs + 1:
            del(top_logprob[tokens[i]])
        # Convert each candidate in top_logprob to list of dictionary format
        candidate_list = [{'logprob': logprob, 'token': token} for token, logprob in top_logprob.items()]
        step_logprobs.append(candidate_list)

    # (content, token_count, token_ids, token_strings, step_logprobs)
    # Average log probability
    avg_logprob = sum(token_logprobs) / max(1, len(token_logprobs))
    #print(f"len(top_logprobs) = {len(top_logprobs)}")
    #for elem in top_logprobs:
    #    print(f"elem = {elem}, len(elem) = {len(elem)}, probsum = {sum([np.exp(x) for x in elem.values()])}")
    #print(f"tokens = {tokens}")

    return (
        generated_text,
        len(tokens),
        token_ids,
        tokens,
        step_logprobs,
    )

# Function to query reward model for score
def extract_boxed_answer(text: str) -> str:
    """
    Function to extract content from \\boxed{...} (supports nested brackets)
    If multiple \\boxed{} exist, extract the last one
    
    Args:
        text: Text to extract from
    
    Returns:
        Content inside boxed, or original text or error message if not found
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
        else:
            # When corresponding } is not found
            return "Could not extract answer"
    else:
        # Fallback: traditional method
        if "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        else:
            return text.strip()

def summarize_answer(
    input_text: str,
    output_file: str | None = None,
    max_new_tokens: int = 1000,
    temperature: float = 0.7,
    finetuned_llm: bool = False
) -> str:
    """
    Summarize the LLM answer text
    
    Parameters
    ----------
    input_text : str
        The LLM answer text to summarize
    output_file : str, optional
        Path to the output file (if not specified, display to standard output)
    max_new_tokens : int, optional
        Maximum number of tokens to generate (default: 1000)
    temperature : float, optional
        Temperature for generation (default: 0.7)
    finetuned_llm : bool, optional
        Whether to use fine-tuned LLM (default: False)
        
    Returns
    -------
    str
        Summarized text
    """
    
    # Check if input text is empty
    if not input_text.strip():
        raise ValueError("Input text is empty")
    
    # Create summarization prompt
    prompt = f"""Please summarize the following text concisely. Elaborate the key mathematical steps and intermediate results (including inequalities), and the final answer with \\boxed{{}}, so that the reader can reproduce the solution.

Original text:
{input_text}

Summary:"""
    
    print(f"Starting summarization...")
    print(f"Original text length: {len(input_text)} characters")
    
    try:
        # Generate summary using generate_answer_client function
        results = generate_answer_client(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            n=1,
            temperature=temperature,
            finetuned_llm=finetuned_llm
        )
        
        if not results or len(results) == 0:
            raise Exception("Failed to generate summary")
        
        # Get the first result
        summary, token_count = results[0]
        
        if not summary.strip():
            raise Exception("Empty summary was generated")
        
        print(f"Summarization completed. Generated tokens: {token_count}")
        
        # Handle output
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"Summary saved to output file: {output_file}")
            except Exception as e:
                print(f"Error occurred while saving output file: {str(e)}")
                print("Displaying summary content to standard output:")
                print("-" * 50)
                print(summary)
                print("-" * 50)
        else:
            print("Summary result:")
            print("-" * 50)
            print(summary)
            print("-" * 50)
        
        return summary
        
    except Exception as e:
        raise Exception(f"Error occurred during summary generation: {str(e)}")


def compare_math_answers_with_llm(answer1: str, answer2: str, max_retries: int = 3) -> bool:
    """
    Function to use LLM to determine whether two mathematical answers are numerically the same
    
    Parameters
    ----------
    answer1 : str
        Answer 1 to compare
    answer2 : str
        Answer 2 to compare
    max_retries : int
        Number of retries for LLM API
    
    Returns
    -------
    bool
        True if the two answers are numerically the same, False otherwise
    """
    if not answer1.strip() or not answer2.strip():
        return False
    
    prompt = f"""Determine whether two mathematical answers are numerically identical.

Answer 1: {answer1}
Answer 2: {answer2}

Criteria:
- If they represent exactly the same numerical value, respond "YES"
- If they represent different values or one is not numerical, respond "NO"
- Different formats (fractions, decimals, radicals, exponential notation) are acceptable if numerically equivalent

Examples:
- "1/2" and "0.5" → YES
- "√4" and "2" → YES  
- "2^3" and "8" → YES
- "3.14" and "π" → NO (approximation vs exact value)
- "x+1" and "1+x" → YES (same expression)
- "x^2" and "2x" → NO (different expressions)

Answer only "YES" or "NO"."""

    try:
        # Use existing LLM API function
        responses = generate_answer_client(
            prompt=prompt,
            max_new_tokens=500, # Reasoning model starts with <think>...
            n=1,
            temperature=0.1,  # Low temperature for consistent judgment
            max_retries=max_retries,
            timeout=30
        )
        #print(f"DEBUG: responses: {responses}")
        #Yes = "YES" in responses[0][0].strip().upper()[-100:]
        #print(f"YES? {Yes}")
        #sys.exit()
        if responses and len(responses) > 0:
            response_text = responses[0][0].strip().upper()[-100:]
            return "YES" in response_text and "NO" not in response_text
        else:
            print("Warning: LLM comparison failed, falling back to string comparison")
            return answer1.strip() == answer2.strip()
            
    except Exception as e:
        print(f"Error in LLM comparison: {str(e)}, falling back to string comparison")
        return answer1.strip() == answer2.strip()

def is_correct(answer: str, gold: str, dataset_type: str = None) -> bool:
    """
    Function to determine whether generated answer is correct
    
    Parameters
    ----------
    answer : str
        Generated answer
    gold : str
        Correct answer
    dataset_type : str, optional
        Dataset type (for special handling like math or math500)
    
    Returns
    -------
    bool
        True if answer is correct, False otherwise
    """
    import re
    
    def remove_degree_symbols(text: str) -> str:
        """Function to remove degree symbols (AIME2025 support)"""
        text = text.strip()
        # Remove ^\circ
        text = re.sub(r'\^\s*\\circ', '', text)
        # Also remove ° symbol
        text = re.sub(r'°', '', text)
        return text.strip()
    
    def normalize_letter_choice(text: str) -> str:
        """Normalize alphabet choices by removing parentheses"""
        text = text.strip()
        # (A), (B), ..., (J) → A, B, ..., J
        if len(text) == 3 and text.startswith('(') and text.endswith(')') and text[1].isalpha():
            return text[1]
        # Keep A, B, ..., J as-is
        elif len(text) == 1 and text.isalpha():
            return text
        return text
    
    def is_mmlu_choice_match(pred: str, gold: str) -> bool:
        """Choice matching for MMLU-Pro (allow parentheses)"""
        pred_norm = normalize_letter_choice(pred)
        gold_norm = normalize_letter_choice(gold)
        return pred_norm == gold_norm
    
    # Extract answer from boxed pattern
    extracted_answer = extract_boxed_answer(answer)
    
    if extracted_answer:
        # Also try comparison after removing degree symbols (AIME2025 support)
        extracted_cleaned = remove_degree_symbols(extracted_answer)
        gold_cleaned = remove_degree_symbols(str(gold))
        
        # For math or math500 datasets: use LLM numerical comparison
        if dataset_type in ["math", "math500"]:
            # Compare extracted answer from boxed with gold using LLM
            #print(f"extracted_answer: {extracted_answer}")
            #print(f"gold: {gold}")
            comparison_result = compare_math_answers_with_llm(extracted_answer, str(gold))
            #print(f"compare_math_answers_with_llm(extracted_answer, str(gold)): {comparison_result}")
            if comparison_result:
                return True
            
            # Also try LLM comparison with degree symbols removed
            if extracted_cleaned != extracted_answer or gold_cleaned != str(gold):
                comparison_result_cleaned = compare_math_answers_with_llm(extracted_cleaned, gold_cleaned)
                if comparison_result_cleaned:
                    return True
        else:
            # Regular exact match check
            if str(gold) == extracted_answer:
                return True
            
            # Exact match check with degree symbols removed
            if gold_cleaned == extracted_cleaned:
                return True
            
            # For MMLU-Pro and GPQA-Diamond, allow parentheses for alphabet choices
            if dataset_type in ["mmlu_pro", "gpqa_diamond"]:
                if is_mmlu_choice_match(extracted_answer, str(gold)):
                    return True
    
    # Partial match in last 50 characters
    if len(str(gold)) >= 2 and str(gold) in answer[-50:]:
        return True
    
    # Partial match in tail with degree symbols removed (AIME2025 support)
    gold_cleaned = remove_degree_symbols(str(gold))
    if len(gold_cleaned) >= 2 and gold_cleaned in remove_degree_symbols(answer[-50:]):
        return True
    
    # For MMLU-Pro and GPQA-Diamond, match choices at tail (allow parentheses)
    if dataset_type in ["mmlu_pro", "gpqa_diamond"]:
        answer_tail = answer[-50:]
        # Extract single alphabet letters or parenthesized alphabets from tail
        letter_patterns = [r'\b([A-J])\b', r'\(([A-J])\)']
        for pattern in letter_patterns:
            matches = re.findall(pattern, answer_tail)
            if matches:
                last_match = matches[-1]  # last found match
                if is_mmlu_choice_match(last_match, str(gold)):
                    return True
    
    return False


def load_existing_answer(dataset_type, global_index, existing_answers_dir,answer_index=None):
    """
    Function to load the part after <think> from existing answer files
    
    Parameters
    ----------
    dataset_type : str
        Dataset type (e.g., "aime2024", "math")
    global_index : int
        Problem number
    answer_index : int, optional
        Explicitly specified answer number. Random selection if not specified
        
    Returns
    -------
    tuple
        (Answer text after <think>, filename), ("", "") if not found
    """
    if answer_index is not None:
        # Select explicitly specified number file
        target_file = f"{existing_answers_dir}/{dataset_type}_prob{global_index}_answer{answer_index}.txt"
        print(f"🎯 [LOAD] Explicitly specified file: {os.path.basename(target_file)}")
        
        if os.path.exists(target_file):
            selected_file = target_file
            selected_basename = os.path.basename(selected_file)
            print(f"✅ [LOAD] Specified file exists: {selected_basename}")
        else:
            print(f"❌ [LOAD] Specified file not found: {target_file}")
            return "", os.path.basename(target_file)
    else:
        # Random selection as before
        pattern = f"{existing_answers_dir}/{dataset_type}_prob{global_index}_answer*.txt"
        print(f"🔍 [LOAD] Answer file search pattern: {pattern}")
        
        matching_files = glob.glob(pattern)
        print(f"🔍 [LOAD] Number of files found: {len(matching_files)}")
        
        if not matching_files:
            print(f"❌ [LOAD] No existing answer files found: {pattern}")
            return "", ""
        
        # Display list of found files (up to 5)
        if len(matching_files) <= 5:
            for file in matching_files:
                print(f"🔍 [LOAD]   - {os.path.basename(file)}")
        else:
            for file in matching_files[:3]:
                print(f"🔍 [LOAD]   - {os.path.basename(file)}")
            print(f"🔍 [LOAD]   - ... {len(matching_files)-3} more")
        
        # Randomly select one
        selected_file = random.choice(matching_files)
        selected_basename = os.path.basename(selected_file)
    print(f"🎯 [LOAD] Selected file: {selected_basename}")
    
    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_size_kb = len(content) / 1024
        print(f"📄 [LOAD] File loading complete: {file_size_kb:.1f}KB")
        
        # Extract part after <think>
        think_pos = content.find('<think>')
        if think_pos != -1:
            answer_content = content[think_pos + len('<think>'):].strip()
            print(f"✅ [LOAD] Answer extraction complete: {len(answer_content)} characters")
            return answer_content, selected_basename
        else:
            print(f"❌ [LOAD] <think> tag not found: {selected_basename}")
            # Fallback: extract everything after first "Prompt: " (or "Prompt:" if not found)
            prompt_marker = 'Prompt: '
            prompt_pos = content.find(prompt_marker)
            if prompt_pos == -1:
                prompt_marker = 'Prompt:'
                prompt_pos = content.find(prompt_marker)
            if prompt_pos != -1:
                answer_content = content[prompt_pos + len(prompt_marker):].lstrip()
                print(f"✅ [LOAD] Fallback extraction after Prompt: {len(answer_content)} characters")
                return answer_content, selected_basename
            else:
                print("⚠️ [LOAD] 'Prompt:' marker also not found")
                return "", selected_basename
            
    except Exception as e:
        print(f"❌ [LOAD] File loading error: {str(e)}")
        return "", ""

