# Repository of Best-of-Infinity

News: Our paper is accepted at ICLR2026! 

This repository contains code for the paper "Best-of-Infinity - Asymptotic Performance of Test-Time Compute". 

* The `boinf` directory provides analysis based on summary JSON files (jsonl/ directory). All plots are derived from the analysis in this directory.
* The `answer_generation` directory includes tools for answer generation as well as comparing best-of-N selection methods. If you only want to test the best-of-infinity results, you do not need to use this directory.
* The "Large-scale Generation Dataset" (= all raw LLM answers) are found at https://figshare.com/account/articles/30208525. 

## Structure

- `boinf/`
  - `train_main.py`: Optimal weight search via MILP, dataset split analysis, and transfer evaluation
  - `test_main.py`: Evaluate and visualize weighted majority vote (finite-sample and population versions) with adaptive and fixed-N sampling
  - `ensemble_utils.py`: Utilities for loading JSONL, computing accuracies, MILP formulation (depends on HiGHS/highspy), etc.
  - `example_utils.py`: Used by `train_main.py` and `test_main.py`.
  - `jsonl_to_table.py`: Convert `jsonl/*.jsonl` to LaTeX tables and generate summaries
  - `accuracy_per_problem.py`: Plot the relationship between sample size n and majority-vote accuracy per problem
  - `count_problems.py`: Count the number of records (problems) in `analysis_*.jsonl`
  - `jsonl/`: Updated pre-analyzed files (e.g., `analysis_aime2025_*.jsonl`, `analysis_math500_*.jsonl`), summarized from raw answer files so that we do not need to use the generation files directly. Based on re-generated GPT-OSS-20B answers with CoTs and we have slightly updated the parser.
  - `jsonl_ver0/`: Pre-analyzed files (old), used in the paper. Based on GPT-OSS-20B answers without CoT.
- `answer_generation/`
  - `BoN_answeranalyze.py`: Analyze logs stored in `saved_answers/` and produce `analysis_{dataset}_{llm}.jsonl`
  - `BoN_choice_analyze.py`: Aggregate `saved_choices/*.jsonl`, and show/visualize accuracy by scale, etc.
  - `BoN_batch.py`: Batch runner support for `BoN_client.py` (organize logs/outputs)
  - `saved_answers/`: Example saved answers
  - `output_batch_datagen/`: Example outputs from batch runs

## Dependencies

- Python 3.10+ (We used Python 3.11.11 at runpod)
- We recommend a linux docker machine with working directory at /workspace
- Required files are found at requirement.txt

Example (pip):
```bash
pip install -r requirements.txt
```

## Reproduction of plots

###  Experiment Set 1: Comparison between adaptive and fixed sampling
Move to the `boinf/` directory. For example, if we want to analyze `NVIDIA-Nemotron-Nano-9B-v2` on the `MATH500` dataset:
```bash
cd boinf
python test_main.py jsonl/analysis_math500_NVIDIA-Nemotron-Nano-9B-v2.jsonl --n-trials 100 --analyze-bayes
```
- All plots are saved to `plots/`

### Experiment Set 2: LLM Ensemble versus single LLM

Move to the `boinf/` directory. For example, if we want to analyze the ensemble of 5 LLMs on the `GPQA-DIAMOND` dataset:
```bash
cd boinf
python test_main.py jsonl/analysis_gpqa_diamond_EXAONE-Deep-32B.jsonl jsonl/analysis_gpqa_diamond_MetaStone-S1-32B.jsonl jsonl/analysis_gpqa_diamond_Phi-4-reasoning.jsonl jsonl/analysis_gpqa_diamond_Qwen3-30B-A3B-Thinking-2507.jsonl jsonl/analysis_gpqa_diamond_gpt-oss-20b.jsonl --weights 0.0176,0.0346,0.2690,0.4144,0.2644 --n-trials 100 --analyze-bayes --no-analyze-fixed --show-single --b-bf 3000
```

To optimize the weights
```bash
python train_main.py jsonl/analysis_gpqa_diamond_EXAONE-Deep-32B.jsonl jsonl/analysis_gpqa_diamond_MetaStone-S1-32B.jsonl jsonl/analysis_gpqa_diamond_Phi-4-reasoning.jsonl jsonl/analysis_gpqa_diamond_Qwen3-30B-A3B-Thinking-2507.jsonl jsonl/analysis_gpqa_diamond_gpt-oss-20b.jsonl 
```

### Experiment Set 3: Number of training samples and LLM ensemble performance 

```bash
cd boinf
time python train_main.py --dataset-split jsonl/analysis_math500_*.jsonl 
```

### Experiment Set 4: Transfer learning from one dataset to another dataset

```bash
cd boinf
python train_main.py --dataset-source aime2024 --dataset-target aime2025
```

### Best-of-1 and Best-of-Infinity performance of each LLM
```bash
cd boinf
python jsonl_to_table.py --all
```

### Performance of each model, each problem
```bash
cd boinf
python jsonl_to_table.py jsonl/analysis_aime2025_gpt-oss-20b.jsonl
```

### Performance of each model, each problem
```bash
cd boinf
python jsonl_to_table.py jsonl/analysis_aime2025_gpt-oss-20b.jsonl
```

## Creating analysis JSONL (optional)

You can use `answer_generation/BoN_client.py` to generate LLM answers in `answer_generation/saved_answers/`. To do so,

* Launch a vllm server on port 8100. For example, GPT-OSS can be launched based on their tutorial: https://cookbook.openai.com/articles/gpt-oss/run-vllm
```bash
 vllm serve /workspace/gpt-oss-20b  --port 8100
```
* Obtain the dataset by 
```bash
cd /workspace
git lfs install
git clone https://huggingface.co/datasets/opencompass/AIME2025
```

* Other datasets:
  - AIME2024: https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
  - GPQA-D: https://huggingface.co/datasets/fingertap/GPQA-Diamond
  - MATH500: https://github.com/openai/prm800k.git 

* To create 80 answers for each problem of AIME2025, run
```bash
cd answer_generation
python BoN_batch_new.py --start 0 --end 15 --max_workers 16 --dataset_type aime2025 --evaluation_method best --use_save --output_dir output_batch_datagen -n 5 --file_start 0 
```
- This creates 16 processses and each process generates five answers. The first proces creates `saved_answers/aime2025_probXX_answerYY.txt`, where `XX` ranges from 0 to 29 (i.e., AIME2025 has 30 problems) and `YY` ranges from 0 to 4. 

You can use `answer_generation/BoN_answeranalyze.py` to generate `analysis_{dataset}_{llm}.jsonl` from text logs in `saved_answers/`. This directory is for each LLM. 
```bash
cd answer_generation
python BoN_answeranalyze.py --dataset aime2025
# Output: analysis_aime2025_<LLM_name>.jsonl (written to current directory). <LLM_name> is extracted from the variable LLM_MODEL_PORT_8100 defined in the .env file. Change it accordingly.
```

###  Experiment Set 5: Comparing BoN strategies

After generating LLM answers, one can compare the selection methods as follows:
```bash
cd answer_generation
python BoN_batch.py --start 0 --end 15 --max_workers 16 --dataset_type aime2025 --evaluation_method random -n 5 --file_start 0 --max_samples 1000
python BoN_batch.py --start 0 --end 15  --max_workers 16 --dataset_type aime2025 --evaluation_method omni -n 5 --file_start 0 --max_samples 1000
python BoN_batch.py --start 0 --end 15 --max_workers 16 --dataset_type aime2025 --evaluation_method majority -n 2 --file_start 0 --max_samples 1000
python BoN_batch.py --start 0 --end 15 --max_workers 16 --dataset_type aime2025 --evaluation_method self_certainty -n 5 --file_start 0 --max_samples 1000
```
After setting up an LLM server on port 8100
```bash
python BoN_batch.py --start 0 --end 15 --max_workers 16 --dataset_type aime2025 --evaluation_method llm_judge_set -n 5 --file_start 0 --max_samples 1000
python BoN_batch.py --start 0 --end 15  --max_workers 16 --dataset_type aime2025 --evaluation_method llm_judge_tournament -n 5 --file_start 0 --max_samples 1000
```

For checking reward models, clone the reward server into `/workspace` and start the reward server process
```bash
./start_reward_server.sh /workspace/Skywork-Reward-V2-Llama-3.1-8B
```
and motify the variable `REWARD_MODEL_ID` in the .env file accordingly.

Then, evaluate the performance of the reward server
```bash
python BoN_batch.py --start 0 --end 15 --max_workers 16 --dataset_type aime2025 --evaluation_method reward -n 5 --file_start 0 --max_samples 1000
```

Performance of these methods can be compared by 
```bash
python BoN_choice_analyze.py --dataset aime2025
```


## Typical workflow
1. Use existing `boinf/jsonl/*.jsonl` to optimize/evaluate the ensemble (see "Reproduction of plots" above)
2. For additional models/datasets, generate your own `analysis_*.jsonl` with tools under `answer_generation/`

## License
- MIT License. 



