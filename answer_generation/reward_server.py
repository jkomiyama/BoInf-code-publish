import torch, uvicorn
import torch.nn as nn
import os
import re
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import (LlamaPreTrainedModel,
                          LlamaModel,  
                          AutoModelForSequenceClassification,
                          AutoModelForCausalLM,
                          AutoTokenizer,
                          PreTrainedTokenizerFast)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

# -------------------- Load environment variables --------------------
load_dotenv()

# -------------------- Load model --------------------
REWARD_MODEL_ID = os.getenv("REWARD_MODEL_ID", "/workspace/ArmoRM-Llama3-8B-v0.1")  # Set default value
# REWARD_MODEL_ID = "/workspace/LDL-Reward-Gemma-2-27B-v0.1"
# REWARD_MODEL_ID = "/workspace/Skywork-Reward-Gemma-2-27B-v0.2"
# REWARD_MODEL_ID = "/workspace/Skywork-Reward-V2-Qwen3-8B"
# REWARD_MODEL_ID = "Reward-Reasoning/RRM-7B"
# REWARD_MODEL_ID = "gaotang/RM-R1-DeepSeek-Distilled-Qwen-7B"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16

# Determine if it's RRM-7B or RM-R1
IS_RRM = "RRM" in REWARD_MODEL_ID or "Reward-Reasoning" in REWARD_MODEL_ID
IS_RM_R1 = ("RM-R1" in REWARD_MODEL_ID or 
           "gaotang/RM-R1-DeepSeek-Distilled-Qwen-7B" in REWARD_MODEL_ID or
           "RM-R1-DeepSeek-Distilled-Qwen-7B" in REWARD_MODEL_ID or
           "DeepSeek-R1-Distill" in REWARD_MODEL_ID)
IS_SKYWORK_QWEN3 = "Skywork-Reward-V2-Qwen3-8B" in REWARD_MODEL_ID

rm_tokenizer = AutoTokenizer.from_pretrained(
    REWARD_MODEL_ID, use_fast=True, trust_remote_code=True)

class INFORMForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

if IS_RRM or IS_RM_R1:
    # Generative model for RRM-7B or RM-R1
    rm = AutoModelForCausalLM.from_pretrained(
        REWARD_MODEL_ID, 
        torch_dtype=DTYPE, 
        trust_remote_code=True,
        device_map="auto"
    ).eval()
elif REWARD_MODEL_ID == "/workspace/Skywork-Reward-Gemma-2-27B-v0.2":
    # c.f., https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B-v0.2
    rm = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
        num_labels=1,
    )
elif IS_SKYWORK_QWEN3:
    # Configuration for Skywork-Reward-V2-Qwen3-8B
    # Configuration based on sample code
    rm = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
elif "INF-ORM" in REWARD_MODEL_ID:
    # Load model
    # Based on sample code from https://huggingface.co/infly/INF-ORM-Llama3.1-70B
    model_name = "/workspace/INF-ORM-Llama3.1-70B"
    rm = INFORMForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        num_labels=1,
    )

    # Load tokenizer
    model_name_hugging_face = "/workspace/INF-ORM-Llama3.1-70B"
    rm_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_hugging_face)
else:
    rm = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True,
        device_map="auto").eval()

ERROR_TOKEN = "[ERROR]"

# -------------------- Prompt template for RRM --------------------
SKYWORK_SYSTEM_PROMPT = """You are a helpful assistant in evaluating the quality of the responses for a given instruction. Your goal is to select the best response for the given instruction.
Select Assistant 1 or Assistant 2, that is better for the given instruction. The two responses are generated by two different AI assistants respectively.
Do NOT say both / neither are good.

Here are some rules of the evaluation:
(1) If the instruction does not contain harmful content, you should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) If the instruction contains harmful content, prioritize the harmlessness and safety of the response.
(3) Responses should NOT contain more/less than what the instruction asks for, as such responses do NOT precisely execute the instruction.
(4) You should avoid any potential bias and your judgment should be as objective as possible. Here are some potential sources of bias:
- The order in which the responses were presented should NOT affect your judgment, as Response A and Response B are equally likely to be the better.
- The length of the responses should NOT affect your judgement, as a longer response does not necessarily correspond to a better response. When making your decision, evaluate if the response length is appropriate for the given instruction.
(5) Your output should only consist of '\\boxed{Assistant 1}' if assistant 1 is better, or '\\boxed{Assistant 2}' if assistant 2 is better. Omit any other output.

"""

SKYWORK_PROMPT = """## Query

{question}

## Assistant responses

### Assistant 1

{answer1}


### Assistant 2

{answer2}

"""

SKYWORK_ASSISTANT_PROMPT = """## Analysis

Let's analyze this step by step and decide which assistant is better, and then answer \\boxed{Assistant 1} or \\boxed{Assistant 2}."""

# -------------------- Prompt template for RM-R1 --------------------
RM_R1_PROMPT_TEMPLATE = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client question displayed below. \n\n"
    "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
    "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]" + "\n\n"
    "Output your final verdict at last by strictly following this format: "
    "'<answer>[[A]]</answer>' if Chatbot A is better, or '<answer>[[B]]</answer>' if Chatbot B is better."
)

# -------------------- RRM inference function --------------------
def get_rrm_preference(prompt: str, response1: str, response2: str) -> float:
    """Evaluate the relative quality of two responses using RRM or RM-R1 and return the probability that response1 is chosen"""
    
    print(f"🔍 [RRM_DEBUG] Model ID: {REWARD_MODEL_ID}")
    print(f"🔍 [RRM_DEBUG] IS_RRM: {IS_RRM}, IS_RM_R1: {IS_RM_R1}")
    
    if IS_RM_R1:
        print(f"🔍 [RRM_DEBUG] Using RM-R1 patterns")
        # Prompt template for RM-R1
        user_prompt = RM_R1_PROMPT_TEMPLATE.format(
            question=prompt, answer_a=response1, answer_b=response2
        )
        
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        generation_args = {
            "max_new_tokens": 8192,
            "do_sample": False,  # False in RM-R1 sample code
        }
        
        # RM-R1 output parsing (more strict pattern)
        assistant_a_pattern = r'<answer>\s*\[\[A\]\]\s*</answer>'
        assistant_b_pattern = r'<answer>\s*\[\[B\]\]\s*</answer>'
        
    else:
        print(f"🔍 [RRM_DEBUG] Using RRM-7B patterns")
        # Prompt template for RRM-7B (existing processing)
        system_prompt = SKYWORK_SYSTEM_PROMPT
        user_prompt = SKYWORK_PROMPT.format(
            question=prompt, answer1=response1, answer2=response2
        ) + SKYWORK_ASSISTANT_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        generation_args = {
            "max_new_tokens": 8192,
            "temperature": 0.6,
            "do_sample": True,
            "top_p": 1.0,
            "eos_token_id": rm_tokenizer.eos_token_id,
            "pad_token_id": rm_tokenizer.pad_token_id,
        }
        
        # RRM-7B output parsing
        assistant_a_pattern = r'\\boxed\{Assistant 1\}'
        assistant_b_pattern = r'\\boxed\{Assistant 2\}'

    prompt_text = rm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = rm_tokenizer(prompt_text, return_tensors="pt").to(rm.device)

    with torch.no_grad():
        output = rm.generate(**inputs, **generation_args)

    generated_text = rm_tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the part after the </think> tag
    think_end = generated_text.find('</think>')
    if think_end != -1:
        search_text = generated_text[think_end + len('</think>'):]
        print(f"🔍 [RRM_DEBUG] Found </think> tag, searching in post-think text")
    else:
        search_text = generated_text
        print(f"🔍 [RRM_DEBUG] No </think> tag found, searching in full text")
    
    # For debugging: display the last part of the search target text
    print(f"🔍 [RRM_DEBUG] Search text (last 500 chars): {search_text[-500:]}")
    #print(f"🔍 [RRM_DEBUG] Looking for pattern A: {assistant_a_pattern}")
    #print(f"🔍 [RRM_DEBUG] Looking for pattern B: {assistant_b_pattern}")
    
    # Search for matching patterns (only text after </think>)
    match_a = re.search(assistant_a_pattern, search_text)
    match_b = re.search(assistant_b_pattern, search_text)
    
    #print(f"🔍 [RRM_DEBUG] Pattern A match: {match_a}")
    #print(f"🔍 [RRM_DEBUG] Pattern B match: {match_b}")
    
    # Check pattern B first (to prevent false detection)
    if match_b:
        print(f"🔍 [RRM_DEBUG] Found pattern B: '{match_b.group()}' → returning 0.0")
        return 0.0  # response2 was chosen
    elif match_a:
        print(f"🔍 [RRM_DEBUG] Found pattern A: '{match_a.group()}' → returning 1.0")
        return 1.0  # response1 was chosen
    else:
        # Default to 0.5 if neither is found
        print(f"🔍 [RRM_DEBUG] No pattern found → returning 0.5")
        return 0.5

# -------------------- Existing utilities --------------------
def get_score(
    prompt: str,
    response: str,
    *,
    attr: str = "score",     # "score" | "first" | "mean"
    device = None
) -> float:
    if response.find(ERROR_TOKEN) != -1:
        return 0.0

    if IS_RRM or IS_RM_R1:
        # (This is good for testing but not beneficial)
        # For RRM or RM-R1, calculate relative score by comparing with dummy response
        # Use empty string as simple baseline response
        baseline_response = "I cannot provide a response to this query."
        return get_rrm_preference(prompt, response, baseline_response)
    
    if IS_SKYWORK_QWEN3:
        # Processing for Skywork-Reward-V2-Qwen3-8B
        # Processing based on sample code: https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        
        # Apply chat template
        conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
        
        # Remove BOS token duplication (based on sample code processing)
        if rm_tokenizer.bos_token is not None and conv_formatted.startswith(rm_tokenizer.bos_token):
            conv_formatted = conv_formatted[len(rm_tokenizer.bos_token):]
            
        conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device or rm.device)
        
        with torch.no_grad():
            # Get scalar value from logits[0][0] according to sample code
            score = rm(**conv_tokenized).logits[0][0].item()
            return score

    msgs = [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]
    input_ids = rm_tokenizer.apply_chat_template(
        msgs, return_tensors="pt").to(device or rm.device)

    with torch.no_grad():
        out = rm(input_ids)

    if attr == "score":
        if "INF-ORM" in REWARD_MODEL_ID:
            return out.logits[0][0].item()
        elif hasattr(out, "score"):
            if REWARD_MODEL_ID == "/workspace/Skywork-Reward-Gemma-2-27B-v0.2":
                return out.logits[0][0].item() # for Skywork-Reward-Gemma-2-27B-v0.2
            elif IS_SKYWORK_QWEN3:
                return out.logits[0][0].item() # for Skywork-Reward-V2-Qwen3-8B
            elif REWARD_MODEL_ID == "/workspace/ArmoRM-Llama3-8B-v0.1":
                return out.score.float().item() # for ArmoRM https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1  Easy to use Pipeline
            else:
                return out.logits[0].item() # for LDL-Reward-Gemma-2-27B-v0.1
        else:
            # INF-ORM models don't have score attribute, use logits
            return out.logits[0][0].item()
    

    logits = out.logits  # (B, num_labels)
    if logits.shape[-1] == 1 or attr == "first":
        return logits[:, 0].float().item()
    elif attr == "mean":
        return logits.mean(dim=1).float().item()
    else:
        raise ValueError(f"unknown attr = {attr!r}")

# -------------------- FastAPI wrapper --------------------
app = FastAPI()

class Req(BaseModel):
    prompt: str
    response: str

class CompareReq(BaseModel):
    prompt: str
    response1: str
    response2: str

class Resp(BaseModel):
    score: float

class CompareResp(BaseModel):
    preference_score: float  # Probability that response1 is chosen (0.0 - 1.0)

@app.post("/score", response_model=Resp)
def score(req: Req, attr: str = Query("score", enum=["score", "first", "mean"])):
    try:
        val = get_score(req.prompt, req.response, attr=attr)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"score": val}

@app.post("/compare", response_model=CompareResp)
def compare(req: CompareReq):
    """Comparison endpoint for RRM or RM-R1"""
    if not IS_RRM and not IS_RM_R1:
        raise HTTPException(status_code=400, detail="Compare endpoint is only available for RRM or RM-R1 models")
    
    try:
        preference = get_rrm_preference(req.prompt, req.response1, req.response2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during comparison: {str(e)}")
    
    return {"preference_score": preference}

@app.get("/model_info")
def model_info():
    gpu_info = {
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    }
    
    return {
        "model_id": REWARD_MODEL_ID,
        "is_rrm": IS_RRM,
        "is_rm_r1": IS_RM_R1,
        "is_skywork_qwen3": IS_SKYWORK_QWEN3,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "gpu_info": gpu_info
    }

if __name__ == "__main__":
    # When running from command line (uvicorn reward_server:app),
    # this section is not executed
    uvicorn.run(app, host="0.0.0.0", port=9000, workers=2)
