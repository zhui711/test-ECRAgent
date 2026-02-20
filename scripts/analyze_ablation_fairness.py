import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import os

# Add parent directory to path to import local modules if needed, though this script is standalone
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

def normalize_answer(ans):
    """Normalize answer string for comparison."""
    if not ans: return ""
    return str(ans).strip().lower()

def analyze_fairness(input_file):
    print(f"Loading data from: {input_file}")
    
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Reorganize by case_id (assuming case_id is same for control and trap pair)
    # The format in sample shows case_id is shared: "case_10000" for both.
    # However, sometimes they might differ slightly or depend on how they were generated.
    # Let's inspect unique IDs first.
    
    paired_data = defaultdict(dict)
    
    for row in records:
        cid = row.get('case_id')
        ctype = row.get('case_type')
        
        # Determine correctness
        # Use final_diagnosis_name vs ground_truth_name
        gt = normalize_answer(row.get('ground_truth_name'))
        pred = normalize_answer(row.get('final_diagnosis_name'))
        is_correct = (gt == pred)
        
        if ctype == 'control_case':
            paired_data[cid]['control'] = {
                'correct': is_correct,
                'gt': gt,
                'pred': pred,
                'status': row.get('status')
            }
        elif ctype == 'trap_case':
            paired_data[cid]['trap'] = {
                'correct': is_correct,
                'pred': pred,
                'status': row.get('status')
            }

    # Filter for valid pairs (both exist and status is success)
    valid_pairs = []
    for cid, data in paired_data.items():
        if 'control' in data and 'trap' in data:
            if data['control']['status'] == 'success' and data['trap']['status'] == 'success':
                valid_pairs.append(data)
                
    total_valid = len(valid_pairs)
    print(f"Total pairs found: {len(paired_data)}")
    print(f"Valid pairs for analysis: {total_valid}")
    
    if total_valid == 0:
        print("No valid pairs found.")
        return

    # Metrics Calculation
    # 1. Baseline Accuracy: Control Accurate / Total
    # 2. Robust Accuracy: (Control Acc AND Trap Acc) / Total
    # 3. Bias Trap Rate: (Control Acc AND Trap Wrong AND Trap Pred == Control GT) / Control Acc?
    #    Wait, definition in prompt: "Control Acc AND Trap Wrong" / Control Acc. 
    #    Actually BTR definition usually implies Trap answer is the Control GT (bias towards original diagnosis).
    #    The prompt says: "Bias Trap Rate: 在 Control 预测正确的前提下，Trap 预测错误的比例。"
    #    Let's stick to the prompt's definition literally: (Control Correct & Trap Incorrect) / Control Correct.
    #    BUT, let's check analysis_agent_results.py for the BTR calculation logic to be consistent.
    #    In analysis_agent_results.py:
    #       if cc and tc: scenario = 'A'
    #       elif cc and not tc:
    #           scenario = 'B' if trap_ans == ctrl_gt else 'C'
    #       ...
    #       btr = (B / control_correct * 100)
    #    So BTR is specifically when Trap Answer matches Control GT (i.e. model ignored the trap info and output the original diagnosis).
    #    This is "Bias Trap Rate". If it's just wrong but different answer, it's scenario C (Confusion).
    #    I will follow analysis_agent_results.py logic.

    stats = defaultdict(int)
    
    for pair in valid_pairs:
        cc = pair['control']['correct']
        tc = pair['trap']['correct']
        
        trap_ans = pair['trap']['pred']
        ctrl_gt = pair['control']['gt']
        
        if cc and tc:
            stats['A'] += 1  # Robust
        elif cc and not tc:
            # Check if trap prediction is the same as the original ground truth (Bias)
            if trap_ans == ctrl_gt:
                stats['B'] += 1 # Bias
            else:
                stats['C'] += 1 # Other Error
        elif not cc and tc:
            stats['E'] += 1
        else:
            stats['D'] += 1

    A = stats['A']
    B = stats['B']
    C = stats['C']
    D = stats['D']
    E = stats['E']
    
    control_correct = A + B + C
    
    baseline_acc = (control_correct / total_valid * 100) if total_valid > 0 else 0
    robust_acc = (A / total_valid * 100) if total_valid > 0 else 0
    
    # BTR = Bias Errors / Total Control Correct
    btr = (B / control_correct * 100) if control_correct > 0 else 0
    
    print("-" * 30)
    print("Fairness Analysis Report")
    print("-" * 30)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Robust Accuracy:   {robust_acc:.2f}%")
    print(f"Bias Trap Rate:    {btr:.2f}%")
    print("-" * 30)
    print(f"A (Robust): {A}")
    print(f"B (Bias):   {B}")
    print(f"C (Other):  {C}")
    print(f"D (Both Wrong): {D}")
    print(f"E (Trap Correct only): {E}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Fairness Metrics from Ablation Results.")
    parser.add_argument("--input", "-i", default="../experiments/rebuttal_ablation/results_detail.jsonl", help="Path to results_detail.jsonl")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to script location if default
        script_dir = Path(__file__).parent
        input_path = script_dir / args.input
        
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
    else:
        analyze_fairness(input_path)
