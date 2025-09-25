#!/usr/bin/env python3

"""
Script to display the number of problems in each JSONL file.
"""

import json
from pathlib import Path
from typing import Dict, List
import glob

def count_problems_in_file(file_path: str) -> int:
    """Count number of problem records in a JSONL file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if line:  # non-empty line
                    try:
                        json.loads(line)  # validate as JSON
                        count += 1
                    except json.JSONDecodeError:
                        continue
            return count
    except Exception as e:
        print(f"Error: failed to read {file_path}: {e}")
        return 0

def group_files_by_dataset(files: List[str]) -> Dict[str, List[str]]:
    """Group files by dataset name"""
    groups = {}
    
    for file_path in files:
        file_name = Path(file_path).name
        
        if 'aime2024' in file_name:
            dataset = 'AIME 2024'
        elif 'aime2025' in file_name:
            dataset = 'AIME 2025'
        elif 'gpqa_diamond' in file_name:
            dataset = 'GPQA Diamond'
        elif 'math500' in file_name:
            dataset = 'MATH 500'
        else:
            dataset = 'Other'
        
        if dataset not in groups:
            groups[dataset] = []
        groups[dataset].append(file_path)
    
    return groups

def main():
    """Main function"""
    print("=== Problem counts per JSONL file ===\n")
    
    # Find analysis_*.jsonl files
    pattern = "analysis_*.jsonl"
    files = glob.glob(pattern)
    files.sort()
    
    if not files:
        print("No analysis_*.jsonl files found")
        return
    
    # Group by dataset
    groups = group_files_by_dataset(files)
    
    total_files = 0
    total_problems = 0
    
    for dataset, dataset_files in sorted(groups.items()):
        print(f"[{dataset}]")
        print("-" * 50)
        
        dataset_total = 0
        for file_path in sorted(dataset_files):
            file_name = Path(file_path).name
            problem_count = count_problems_in_file(file_path)
            
            # Extract model name (analysis_dataset_model.jsonl -> model)
            parts = file_name.replace('.jsonl', '').split('_')
            if len(parts) >= 3:
                model_name = '_'.join(parts[2:])
            else:
                model_name = file_name
            
            print(f"  {model_name:<40} : {problem_count:>4} problems")
            dataset_total += problem_count
            total_problems += problem_count
            total_files += 1
        
        print(f"  {'Total':<40} : {dataset_total:>4} problems")
        print()
    
    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"Total problems: {total_problems}")
    
    # Dataset-wise statistics
    print("\n=== Dataset-wise statistics ===")
    for dataset, dataset_files in sorted(groups.items()):
        model_count = len(dataset_files)
        if model_count > 0:
            # Get problem count of the first file (usually same within a dataset)
            sample_count = count_problems_in_file(dataset_files[0])
            print(f"{dataset:<15} : {model_count:>2} models × {sample_count:>3} problems")

if __name__ == "__main__":
    main()




