#!/usr/bin/env python3
"""
Offline Dataset Preparation Script
===================================

本脚本用于生成平衡的 Offline 训练列表（索引清单），供后续的 Golden Graph 构建使用。

核心逻辑：
1. 扫描 train_verify/ 目录，读取所有 case 的 final_benchmark_pair.json
2. 同时统计 control_case 和 trap_case 的 ground_truth（方案 B）
3. 使用"封顶采样 (Capped Sampling)"策略：
   - 设定每种 pathology 的目标样本数 TARGET_SAMPLES_PER_PATHOLOGY (默认 20)
   - 如果某 pathology 样本数 >= TARGET，随机抽取 TARGET 例
   - 如果某 pathology 样本数 < TARGET，全量选取
4. 输出到 config/offline_train_list.json

相比 Shortest Stave 策略的优势：
- 最大化利用常见病的数据（如 Pneumonia 有 500 例，取 20 例而非最小的 3 例）
- 罕见病（如 Ebola 仅 3 例）全量保留，不浪费任何数据

输出格式：
[
  {
    "case_id": "case_123",
    "type": "control",
    "ground_truth": "Pneumonia",
    "ground_truth_id": "45",
    "path": "train_verify/case_123/final_benchmark_pair.json"
  },
  ...
]

使用方法：
    python scripts/prepare_offline_dataset.py [--train-dir TRAIN_DIR] [--output OUTPUT] [--target-samples 20]
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 随机种子，确保可复现性
RANDOM_SEED = 42

# 封顶采样策略参数
# 每种 pathology 的目标样本数，超过此数量的将被随机抽样，低于此数量的全量保留
TARGET_SAMPLES_PER_PATHOLOGY = 20

# 49 种疾病的 ID 到名称映射（从 disease_synonyms.json 加载）
DISEASE_ID_TO_NAME: Dict[str, str] = {}
DISEASE_NAME_TO_ID: Dict[str, str] = {}


def load_disease_mappings() -> None:
    """
    加载疾病 ID 与名称的双向映射。
    从 config/disease_synonyms.json 读取。
    """
    global DISEASE_ID_TO_NAME, DISEASE_NAME_TO_ID
    
    synonyms_path = PROJECT_ROOT / "config" / "disease_synonyms.json"
    
    if not synonyms_path.exists():
        print(f"[ERROR] Disease synonyms file not found: {synonyms_path}")
        sys.exit(1)
    
    with open(synonyms_path, "r", encoding="utf-8") as f:
        synonyms_data = json.load(f)
    
    for disease_id, info in synonyms_data.items():
        name = info.get("name", "")
        DISEASE_ID_TO_NAME[disease_id] = name
        # 主名称映射
        DISEASE_NAME_TO_ID[name] = disease_id
        # 同义词也建立映射
        for synonym in info.get("synonyms", []):
            DISEASE_NAME_TO_ID[synonym] = disease_id
    
    print(f"[INFO] Loaded {len(DISEASE_ID_TO_NAME)} disease mappings")


def normalize_ground_truth(ground_truth: str) -> Tuple[str, str]:
    """
    规范化 ground_truth，返回 (disease_id, disease_name)。
    
    处理可能的格式问题（如大小写不一致、同义词等）。
    
    Args:
        ground_truth: 原始的 ground_truth 字符串
    
    Returns:
        (disease_id, normalized_name) 元组
        如果无法匹配，返回 (None, original_ground_truth)
    """
    # 尝试直接匹配
    if ground_truth in DISEASE_NAME_TO_ID:
        disease_id = DISEASE_NAME_TO_ID[ground_truth]
        return disease_id, DISEASE_ID_TO_NAME[disease_id]
    
    # 尝试不区分大小写匹配
    ground_truth_lower = ground_truth.lower().strip()
    for name, disease_id in DISEASE_NAME_TO_ID.items():
        if name.lower() == ground_truth_lower:
            return disease_id, DISEASE_ID_TO_NAME[disease_id]
    
    # 尝试部分匹配（处理如 "Acute COPD exacerbation" vs "Acute COPD exacerbation / infection"）
    for name, disease_id in DISEASE_NAME_TO_ID.items():
        if ground_truth_lower in name.lower() or name.lower() in ground_truth_lower:
            return disease_id, DISEASE_ID_TO_NAME[disease_id]
    
    # 无法匹配
    print(f"[WARNING] Cannot normalize ground_truth: '{ground_truth}'")
    return None, ground_truth


def scan_train_directory(train_dir: Path) -> List[Dict[str, Any]]:
    """
    扫描训练目录，提取所有有效的 case 记录。
    
    同时处理 control_case 和 trap_case（方案 B）。
    
    Args:
        train_dir: train_verify/ 目录路径
    
    Returns:
        所有 case 记录的列表，每条记录包含：
        - case_id: case 目录名
        - type: "control" 或 "trap"
        - ground_truth: 规范化后的疾病名称
        - ground_truth_id: 疾病 ID
        - path: JSON 文件相对路径
    """
    all_records = []
    
    if not train_dir.exists():
        print(f"[ERROR] Training directory not found: {train_dir}")
        return all_records
    
    # 遍历所有 case 目录
    case_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    print(f"[INFO] Found {len(case_dirs)} case directories")
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        json_path = case_dir / "final_benchmark_pair.json"
        
        if not json_path.exists():
            print(f"[WARNING] Missing JSON file: {json_path}")
            continue
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {json_path}: {e}")
            continue
        
        # 处理 control_case
        if "control_case" in data:
            control_data = data["control_case"]
            ground_truth = control_data.get("ground_truth", "")
            
            if ground_truth:
                disease_id, normalized_name = normalize_ground_truth(ground_truth)
                
                if disease_id:
                    all_records.append({
                        "case_id": case_id,
                        "type": "control",
                        "ground_truth": normalized_name,
                        "ground_truth_id": disease_id,
                        "path": f"train_verify/{case_id}/final_benchmark_pair.json"
                    })
        
        # 处理 trap_case
        if "trap_case" in data:
            trap_data = data["trap_case"]
            ground_truth = trap_data.get("ground_truth", "")
            
            if ground_truth:
                disease_id, normalized_name = normalize_ground_truth(ground_truth)
                
                if disease_id:
                    all_records.append({
                        "case_id": case_id,
                        "type": "trap",
                        "ground_truth": normalized_name,
                        "ground_truth_id": disease_id,
                        "path": f"train_verify/{case_id}/final_benchmark_pair.json"
                    })
    
    print(f"[INFO] Extracted {len(all_records)} total records (control + trap)")
    return all_records


def compute_pathology_distribution(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    统计每种 pathology 的 case 分布。
    
    Args:
        records: 所有 case 记录列表
    
    Returns:
        以 ground_truth_id 为 key，对应 case 列表为 value 的字典
    """
    distribution = defaultdict(list)
    
    for record in records:
        disease_id = record["ground_truth_id"]
        distribution[disease_id].append(record)
    
    return distribution


def shortest_stave_sampling(
    distribution: Dict[str, List[Dict[str, Any]]], 
    seed: int = RANDOM_SEED
) -> List[Dict[str, Any]]:
    """
    短板对齐 (Shortest Stave) 抽样策略。
    
    找出数量最少的 pathology，记其数量为 N。
    从每种 pathology 中随机抽取 N 个 Case。
    
    [已弃用] 请使用 capped_sampling() 替代。
    
    Args:
        distribution: pathology 分布字典
        seed: 随机种子
    
    Returns:
        平衡后的 case 列表
    """
    # 设置随机种子
    random.seed(seed)
    
    # 找出最短板（最少的 pathology 数量）
    min_count = min(len(cases) for cases in distribution.values())
    
    print(f"\n[INFO] ========== Pathology Distribution ==========")
    print(f"[INFO] Total pathologies with data: {len(distribution)}")
    print(f"[INFO] Shortest stave (N): {min_count}")
    print()
    
    # 打印每种 pathology 的统计
    sorted_items = sorted(distribution.items(), key=lambda x: len(x[1]))
    for disease_id, cases in sorted_items:
        disease_name = DISEASE_ID_TO_NAME.get(disease_id, "Unknown")
        count = len(cases)
        status = "⚠️ MIN" if count == min_count else ""
        print(f"  [{disease_id}] {disease_name}: {count} cases {status}")
    
    print()
    
    # 从每种 pathology 中抽取 N 个
    balanced_records = []
    
    for disease_id, cases in distribution.items():
        # 随机打乱后取前 N 个
        shuffled = cases.copy()
        random.shuffle(shuffled)
        sampled = shuffled[:min_count]
        balanced_records.extend(sampled)
    
    print(f"[INFO] After balancing: {len(balanced_records)} total records")
    print(f"[INFO] = {len(distribution)} pathologies × {min_count} cases each")
    
    return balanced_records


def capped_sampling(
    distribution: Dict[str, List[Dict[str, Any]]], 
    target_samples: int = TARGET_SAMPLES_PER_PATHOLOGY,
    seed: int = RANDOM_SEED
) -> List[Dict[str, Any]]:
    """
    封顶采样 (Capped Sampling) 策略。
    
    对于每种 pathology：
    - 如果样本数 >= target_samples，随机抽取 target_samples 例
    - 如果样本数 < target_samples，全量选取
    
    优势：
    - 最大化利用常见病数据（取 20 例而非受最小类限制只取 3 例）
    - 罕见病全量保留，不浪费任何数据
    
    Args:
        distribution: pathology 分布字典
        target_samples: 每种 pathology 的目标样本数
        seed: 随机种子
    
    Returns:
        采样后的 case 列表
    """
    # 设置随机种子
    random.seed(seed)
    
    print(f"\n[INFO] ========== Capped Sampling Strategy ==========")
    print(f"[INFO] Target samples per pathology: {target_samples}")
    print(f"[INFO] Total pathologies with data: {len(distribution)}")
    print()
    
    # 统计信息
    capped_count = 0  # 被封顶的 pathology 数量
    full_count = 0    # 全量选取的 pathology 数量
    
    # 打印每种 pathology 的统计
    sorted_items = sorted(distribution.items(), key=lambda x: len(x[1]))
    
    print(f"  {'ID':<4} {'Disease Name':<45} {'Original':>8} {'Sampled':>8} {'Status':<12}")
    print(f"  {'-'*4} {'-'*45} {'-'*8} {'-'*8} {'-'*12}")
    
    sampled_records = []
    
    for disease_id, cases in sorted_items:
        disease_name = DISEASE_ID_TO_NAME.get(disease_id, "Unknown")
        original_count = len(cases)
        
        # 随机打乱
        shuffled = cases.copy()
        random.shuffle(shuffled)
        
        if original_count >= target_samples:
            # 封顶：只取 target_samples 个
            sampled = shuffled[:target_samples]
            sampled_count = target_samples
            status = "✂️ CAPPED"
            capped_count += 1
        else:
            # 全量选取
            sampled = shuffled
            sampled_count = original_count
            status = "✓ FULL"
            full_count += 1
        
        sampled_records.extend(sampled)
        
        # 缩短疾病名称以适应显示
        display_name = disease_name[:43] + ".." if len(disease_name) > 45 else disease_name
        print(f"  [{disease_id}] {display_name:<45} {original_count:>8} {sampled_count:>8} {status:<12}")
    
    print()
    print(f"[INFO] ========== Sampling Summary ==========")
    print(f"[INFO] Pathologies capped at {target_samples}: {capped_count}")
    print(f"[INFO] Pathologies fully included: {full_count}")
    print(f"[INFO] Total sampled records: {len(sampled_records)}")
    
    # 计算理论最大值对比
    if capped_count > 0:
        theoretical_max = sum(min(len(cases), target_samples) for cases in distribution.values())
        print(f"[INFO] Data utilization: {len(sampled_records)}/{theoretical_max} ({100*len(sampled_records)/theoretical_max:.1f}%)")
    
    return sampled_records


def save_offline_list(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    保存 Offline 训练列表到 JSON 文件。
    
    Args:
        records: 平衡后的 case 列表
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 按 ground_truth_id 排序，便于查看
    sorted_records = sorted(records, key=lambda x: (x["ground_truth_id"], x["case_id"]))
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_records, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Saved offline training list to: {output_path}")
    print(f"[INFO] Total records: {len(sorted_records)}")


def print_sample_output(records: List[Dict[str, Any]], num_samples: int = 5) -> None:
    """
    打印样例输出，便于验证格式。
    
    Args:
        records: case 列表
        num_samples: 要显示的样例数量
    """
    print(f"\n[INFO] ========== Sample Output (first {num_samples} records) ==========")
    
    samples = records[:num_samples]
    print(json.dumps(samples, indent=2, ensure_ascii=False))


def generate_statistics_report(
    all_records: List[Dict[str, Any]], 
    balanced_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    生成统计报告。
    
    Args:
        all_records: 原始所有记录
        balanced_records: 平衡后的记录
    
    Returns:
        统计报告字典
    """
    # 原始分布
    original_dist = compute_pathology_distribution(all_records)
    balanced_dist = compute_pathology_distribution(balanced_records)
    
    # 计算统计指标
    original_counts = [len(cases) for cases in original_dist.values()]
    balanced_counts = [len(cases) for cases in balanced_dist.values()]
    
    report = {
        "original": {
            "total_records": len(all_records),
            "total_pathologies": len(original_dist),
            "min_count": min(original_counts) if original_counts else 0,
            "max_count": max(original_counts) if original_counts else 0,
            "avg_count": sum(original_counts) / len(original_counts) if original_counts else 0
        },
        "balanced": {
            "total_records": len(balanced_records),
            "total_pathologies": len(balanced_dist),
            "samples_per_pathology": balanced_counts[0] if balanced_counts else 0
        },
        "pathology_details": {
            disease_id: {
                "name": DISEASE_ID_TO_NAME.get(disease_id, "Unknown"),
                "original_count": len(original_dist.get(disease_id, [])),
                "balanced_count": len(balanced_dist.get(disease_id, []))
            }
            for disease_id in sorted(original_dist.keys())
        }
    }
    
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Generate balanced Offline training list for Golden Graph construction"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="train_verify",
        help="Path to training directory (default: train_verify)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/offline_train_list.json",
        help="Output file path (default: config/offline_train_list.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=TARGET_SAMPLES_PER_PATHOLOGY,
        help=f"Target samples per pathology for capped sampling (default: {TARGET_SAMPLES_PER_PATHOLOGY})"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["capped", "shortest"],
        default="capped",
        help="Sampling strategy: 'capped' (recommended) or 'shortest' (legacy)"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save statistics report to config/offline_train_report.json"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Offline Dataset Preparation Script (v2.0)")
    print("  MDT-Agent: Golden Graph Training Data Generator")
    print("=" * 60)
    print()
    
    # 解析路径
    train_dir = PROJECT_ROOT / args.train_dir
    output_path = PROJECT_ROOT / args.output
    
    print(f"[CONFIG] Training directory: {train_dir}")
    print(f"[CONFIG] Output file: {output_path}")
    print(f"[CONFIG] Random seed: {args.seed}")
    print(f"[CONFIG] Sampling strategy: {args.strategy}")
    if args.strategy == "capped":
        print(f"[CONFIG] Target samples per pathology: {args.target_samples}")
    print()
    
    # Step 1: 加载疾病映射
    print("[STEP 1] Loading disease mappings...")
    load_disease_mappings()
    
    # Step 2: 扫描训练目录
    print("\n[STEP 2] Scanning training directory...")
    all_records = scan_train_directory(train_dir)
    
    if not all_records:
        print("[ERROR] No valid records found. Exiting.")
        sys.exit(1)
    
    # Step 3: 统计分布
    print("\n[STEP 3] Computing pathology distribution...")
    distribution = compute_pathology_distribution(all_records)
    
    # 检查是否所有 49 种 pathology 都有数据
    missing_pathologies = set(DISEASE_ID_TO_NAME.keys()) - set(distribution.keys())
    if missing_pathologies:
        print(f"\n[WARNING] {len(missing_pathologies)} pathologies have NO data:")
        for disease_id in sorted(missing_pathologies):
            print(f"  [{disease_id}] {DISEASE_ID_TO_NAME[disease_id]}")
    
    # Step 4: 应用采样策略
    if args.strategy == "capped":
        print(f"\n[STEP 4] Applying Capped Sampling (target: {args.target_samples})...")
        sampled_records = capped_sampling(distribution, target_samples=args.target_samples, seed=args.seed)
    else:
        print("\n[STEP 4] Applying Shortest Stave sampling (legacy)...")
        sampled_records = shortest_stave_sampling(distribution, seed=args.seed)
    
    # Step 5: 保存结果
    print("\n[STEP 5] Saving results...")
    save_offline_list(sampled_records, output_path)
    
    # 打印样例
    print_sample_output(sampled_records)
    
    # 可选：保存统计报告
    if args.save_report:
        report = generate_statistics_report(all_records, sampled_records)
        report["sampling_config"] = {
            "strategy": args.strategy,
            "target_samples": args.target_samples if args.strategy == "capped" else None,
            "seed": args.seed
        }
        report_path = PROJECT_ROOT / "config" / "offline_train_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Statistics report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

