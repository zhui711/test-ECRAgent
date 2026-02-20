#!/usr/bin/env python3
"""
Offline Training Results Inspection Script
==========================================

对 Offline Training 的结果进行严格的数据完整性验收（Sanity Check）。

检查项目：
1. Memory Bank 完整性
2. Embedding 质量（是否真正调用了 API）
3. Golden Graph 生成质量

Usage:
    python scripts/inspect_offline_results.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np

# ==================== 颜色输出 ====================

class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """打印标题"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")


def print_pass(text: str) -> None:
    """打印 PASS"""
    print(f"{Colors.GREEN}[PASS]{Colors.ENDC} {text}")


def print_fail(text: str) -> None:
    """打印 FAIL"""
    print(f"{Colors.RED}[FAIL]{Colors.ENDC} {text}")


def print_warn(text: str) -> None:
    """打印 WARNING"""
    print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} {text}")


def print_info(text: str) -> None:
    """打印 INFO"""
    print(f"{Colors.CYAN}[INFO]{Colors.ENDC} {text}")


# ==================== 检查函数 ====================

def check_memory_bank(memory_bank_dir: Path) -> Dict[str, Any]:
    """
    检查 Memory Bank 完整性
    
    Returns:
        检查结果字典
    """
    results = {
        "json_exists": False,
        "total_cases": 0,
        "first_record": None,
        "outcome_distribution": {},
        "issues": []
    }
    
    json_path = memory_bank_dir / "memory_bank.json"
    
    # 检查文件是否存在
    if not json_path.exists():
        results["issues"].append("memory_bank.json 文件不存在")
        return results
    
    results["json_exists"] = True
    
    # 读取 JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        results["issues"].append(f"JSON 解析失败: {e}")
        return results
    
    cases = data.get("cases", [])
    results["total_cases"] = len(cases)
    
    if len(cases) == 0:
        results["issues"].append("Memory Bank 为空（0 条记录）")
        return results
    
    # 第一条记录
    results["first_record"] = cases[0]
    
    # 统计 outcome 分布
    outcome_dist = defaultdict(int)
    for case in cases:
        outcome = case.get("outcome", "Unknown")
        outcome_dist[outcome] += 1
    results["outcome_distribution"] = dict(outcome_dist)
    
    # 检查 p_nodes_summary 是否为空
    empty_summary_count = sum(1 for c in cases if not c.get("p_nodes_summary", "").strip())
    if empty_summary_count > 0:
        results["issues"].append(f"{empty_summary_count} 条记录的 p_nodes_summary 为空")
    
    return results


def check_embeddings(memory_bank_dir: Path, expected_count: int) -> Dict[str, Any]:
    """
    检查 Embedding 质量
    
    Args:
        memory_bank_dir: Memory Bank 目录
        expected_count: 期望的向量数量（应与 memory_bank.json 中的 case 数一致）
    
    Returns:
        检查结果字典
    """
    results = {
        "npy_exists": False,
        "shape": None,
        "expected_shape": (expected_count, 1536),
        "shape_match": False,
        "mean_norm": None,
        "norm_is_normalized": False,  # OpenAI embedding 通常归一化（模长≈1）
        "sample_similarity": None,
        "is_random": None,  # 如果相似度太低，可能是随机向量
        "all_zeros": False,
        "issues": []
    }
    
    npy_path = memory_bank_dir / "embeddings.npy"
    
    # 检查文件是否存在
    if not npy_path.exists():
        results["issues"].append("embeddings.npy 文件不存在")
        return results
    
    results["npy_exists"] = True
    
    # 读取 npy
    try:
        embeddings = np.load(npy_path)
    except Exception as e:
        results["issues"].append(f"NPY 加载失败: {e}")
        return results
    
    results["shape"] = embeddings.shape
    
    # 形状检查
    if len(embeddings.shape) != 2:
        results["issues"].append(f"Embedding 维度错误: 期望 2D，实际 {len(embeddings.shape)}D")
        return results
    
    n_samples, dim = embeddings.shape
    
    if dim != 1536:
        results["issues"].append(f"Embedding 维度错误: 期望 1536，实际 {dim}")
    
    if n_samples != expected_count:
        results["issues"].append(f"Embedding 数量不匹配: JSON有 {expected_count} 条，NPY有 {n_samples} 条")
    else:
        results["shape_match"] = True
    
    # 全零检测
    if np.allclose(embeddings, 0):
        results["all_zeros"] = True
        results["issues"].append("所有 Embedding 全为 0！API 可能未被调用")
        return results
    
    # 模长检查
    norms = np.linalg.norm(embeddings, axis=1)
    results["mean_norm"] = float(np.mean(norms))
    results["std_norm"] = float(np.std(norms))
    results["min_norm"] = float(np.min(norms))
    results["max_norm"] = float(np.max(norms))
    
    # OpenAI embedding 归一化检测（模长应接近 1）
    if 0.9 < results["mean_norm"] < 1.1:
        results["norm_is_normalized"] = True
    else:
        results["issues"].append(f"平均模长 {results['mean_norm']:.4f} 偏离 1，可能不是归一化的 Embedding")
    
    # 随机性检测：计算前几对向量的相似度
    if n_samples >= 2:
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
        # 计算多对相似度
        similarities = []
        for i in range(min(5, n_samples - 1)):
            sim = cosine_sim(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        results["sample_similarity"] = float(np.mean(similarities))
        results["sample_similarities"] = [float(s) for s in similarities]
        
        # 判断是否为随机向量
        # 真正随机的 1536 维向量，余弦相似度期望值接近 0
        # 真实语义向量，相似度通常在 0.3-0.9 之间
        if abs(results["sample_similarity"]) < 0.05:
            results["is_random"] = True
            results["issues"].append(f"相邻向量余弦相似度极低 ({results['sample_similarity']:.4f})，可能是随机向量！")
        else:
            results["is_random"] = False
    
    return results


def check_golden_graphs(golden_graph_dir: Path) -> Dict[str, Any]:
    """
    检查 Golden Graph 生成质量
    
    Returns:
        检查结果字典
    """
    results = {
        "dir_exists": False,
        "total_graphs": 0,
        "graphs": [],
        "issues": []
    }
    
    if not golden_graph_dir.exists():
        results["issues"].append("golden_graphs 目录不存在")
        return results
    
    results["dir_exists"] = True
    
    # 查找所有 Golden Graph 文件
    gg_files = list(golden_graph_dir.glob("*_GG.json"))
    results["total_graphs"] = len(gg_files)
    
    if len(gg_files) == 0:
        results["issues"].append("未找到任何 Golden Graph 文件")
        return results
    
    # 检查每个 Golden Graph
    total_p_nodes = 0
    total_k_nodes = 0
    total_edges = 0
    total_source_cases = 0
    
    for gg_file in sorted(gg_files):
        try:
            with open(gg_file, "r", encoding="utf-8") as f:
                gg_data = json.load(f)
        except Exception as e:
            results["issues"].append(f"无法解析 {gg_file.name}: {e}")
            continue
        
        metadata = gg_data.get("metadata", {})
        stats = gg_data.get("statistics", {})
        graph = gg_data.get("graph", {})
        
        p_count = len(graph.get("p_nodes", []))
        k_count = len(graph.get("k_nodes", []))
        edge_count = len(graph.get("edges", []))
        source_count = metadata.get("source_cases_count", 0)
        
        total_p_nodes += p_count
        total_k_nodes += k_count
        total_edges += edge_count
        total_source_cases += source_count
        
        results["graphs"].append({
            "filename": gg_file.name,
            "pathology": metadata.get("pathology_name", "Unknown"),
            "source_cases": source_count,
            "p_nodes": p_count,
            "k_nodes": k_count,
            "edges": edge_count
        })
        
        # 质量检查
        if p_count == 0:
            results["issues"].append(f"{gg_file.name}: P-Nodes 为 0")
        if k_count == 0:
            results["issues"].append(f"{gg_file.name}: K-Nodes 为 0")
        if edge_count == 0:
            results["issues"].append(f"{gg_file.name}: Edges 为 0")
    
    results["summary"] = {
        "total_p_nodes": total_p_nodes,
        "total_k_nodes": total_k_nodes,
        "total_edges": total_edges,
        "total_source_cases": total_source_cases,
        "avg_p_nodes_per_graph": total_p_nodes / len(gg_files) if gg_files else 0,
        "avg_k_nodes_per_graph": total_k_nodes / len(gg_files) if gg_files else 0,
        "avg_edges_per_graph": total_edges / len(gg_files) if gg_files else 0
    }
    
    return results


def check_training_log(log_dir: Path) -> Dict[str, Any]:
    """
    检查训练日志
    
    Returns:
        检查结果字典
    """
    results = {
        "log_exists": False,
        "total_records": 0,
        "success_count": 0,
        "failed_count": 0,
        "success_first_try": 0,
        "success_after_retry": 0,
        "pathology_distribution": {},
        "issues": []
    }
    
    log_path = log_dir / "offline_training_log.jsonl"
    
    if not log_path.exists():
        results["issues"].append("offline_training_log.jsonl 不存在")
        return results
    
    results["log_exists"] = True
    
    # 读取 JSONL
    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        results["issues"].append(f"日志解析失败: {e}")
        return results
    
    results["total_records"] = len(records)
    
    # 统计
    pathology_dist = defaultdict(lambda: {"success": 0, "failed": 0})
    
    for record in records:
        status = record.get("final_status", "unknown")
        retries = record.get("total_retries", 0)
        gt_name = record.get("ground_truth_name", "Unknown")
        
        if status == "success":
            results["success_count"] += 1
            pathology_dist[gt_name]["success"] += 1
            if retries == 0:
                results["success_first_try"] += 1
            else:
                results["success_after_retry"] += 1
        elif status == "failed":
            results["failed_count"] += 1
            pathology_dist[gt_name]["failed"] += 1
    
    results["pathology_distribution"] = dict(pathology_dist)
    results["success_rate"] = results["success_count"] / results["total_records"] if results["total_records"] > 0 else 0
    
    return results


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    memory_bank_dir = project_root / "memory_bank"
    golden_graph_dir = project_root / "golden_graphs"
    log_dir = project_root / "logs"
    
    all_pass = True
    
    # ========== 1. Memory Bank 检查 ==========
    print_header("1. MEMORY BANK 检查")
    
    mb_results = check_memory_bank(memory_bank_dir)
    
    if mb_results["json_exists"]:
        print_pass(f"memory_bank.json 存在")
        print_info(f"总 Case 数: {mb_results['total_cases']}")
        
        # 打印 outcome 分布
        if mb_results["outcome_distribution"]:
            print_info("Outcome 分布:")
            for outcome, count in mb_results["outcome_distribution"].items():
                print(f"    - {outcome}: {count}")
        
        # 打印第一条记录的摘要
        if mb_results["first_record"]:
            first = mb_results["first_record"]
            print_info(f"\n第一条记录:")
            print(f"    - case_id: {first.get('case_id', 'N/A')}")
            print(f"    - ground_truth: {first.get('ground_truth_name', 'N/A')}")
            print(f"    - outcome: {first.get('outcome', 'N/A')}")
            summary = first.get('p_nodes_summary', '')
            if summary and summary.strip():
                print_pass(f"p_nodes_summary 非空:")
                print(f"        \"{summary[:100]}{'...' if len(summary) > 100 else ''}\"")
            else:
                print_fail("p_nodes_summary 为空！")
                all_pass = False
    else:
        print_fail("memory_bank.json 不存在")
        all_pass = False
    
    for issue in mb_results.get("issues", []):
        print_warn(issue)
        all_pass = False
    
    # ========== 2. Embedding 检查 ==========
    print_header("2. EMBEDDING 检查")
    
    emb_results = check_embeddings(memory_bank_dir, mb_results.get("total_cases", 0))
    
    if emb_results["npy_exists"]:
        print_pass("embeddings.npy 存在")
        print_info(f"Shape: {emb_results['shape']}")
        print_info(f"期望 Shape: {emb_results['expected_shape']}")
        
        if emb_results["shape_match"]:
            print_pass("Shape 匹配")
        else:
            print_fail("Shape 不匹配")
            all_pass = False
        
        # 全零检测
        if emb_results["all_zeros"]:
            print_fail("❌ 所有 Embedding 全为 0！API 未被调用！")
            all_pass = False
        else:
            print_pass("非全零向量")
        
        # 模长检查
        if emb_results["mean_norm"] is not None:
            print_info(f"平均模长 (Norm): {emb_results['mean_norm']:.4f}")
            print_info(f"模长标准差: {emb_results.get('std_norm', 0):.4f}")
            print_info(f"模长范围: [{emb_results.get('min_norm', 0):.4f}, {emb_results.get('max_norm', 0):.4f}]")
            
            if emb_results["norm_is_normalized"]:
                print_pass("模长接近 1，符合 OpenAI Embedding 归一化特征")
            else:
                print_warn("模长偏离 1，可能不是 OpenAI 标准归一化 Embedding")
        
        # 随机性检测
        if emb_results["sample_similarity"] is not None:
            print_info(f"\n相邻向量余弦相似度:")
            for i, sim in enumerate(emb_results.get("sample_similarities", [])):
                print(f"    向量 {i} vs {i+1}: {sim:.4f}")
            print_info(f"平均相似度: {emb_results['sample_similarity']:.4f}")
            
            if emb_results["is_random"]:
                print_fail("❌ 相似度极低，可能是随机向量！API 可能未被正确调用！")
                all_pass = False
            else:
                print_pass("相似度合理，为真实语义向量")
    else:
        print_fail("embeddings.npy 不存在")
        all_pass = False
    
    for issue in emb_results.get("issues", []):
        print_warn(issue)
    
    # ========== 3. Golden Graph 检查 ==========
    print_header("3. GOLDEN GRAPH 检查")
    
    gg_results = check_golden_graphs(golden_graph_dir)
    
    if gg_results["dir_exists"]:
        print_pass("golden_graphs 目录存在")
        print_info(f"Golden Graph 文件数: {gg_results['total_graphs']}")
        
        if gg_results["total_graphs"] > 0:
            summary = gg_results.get("summary", {})
            print_info(f"\n聚合统计:")
            print(f"    - 总 P-Nodes: {summary.get('total_p_nodes', 0)}")
            print(f"    - 总 K-Nodes: {summary.get('total_k_nodes', 0)}")
            print(f"    - 总 Edges: {summary.get('total_edges', 0)}")
            print(f"    - 总 Source Cases: {summary.get('total_source_cases', 0)}")
            print(f"    - 平均 P-Nodes/图: {summary.get('avg_p_nodes_per_graph', 0):.1f}")
            print(f"    - 平均 K-Nodes/图: {summary.get('avg_k_nodes_per_graph', 0):.1f}")
            print(f"    - 平均 Edges/图: {summary.get('avg_edges_per_graph', 0):.1f}")
            
            # 打印每个 Golden Graph 的详情（前5个）
            print_info(f"\nGolden Graph 详情 (前 10 个):")
            for i, gg in enumerate(gg_results["graphs"][:10]):
                print(f"    [{i+1}] {gg['pathology']}: {gg['source_cases']} cases, "
                      f"P={gg['p_nodes']}, K={gg['k_nodes']}, E={gg['edges']}")
            
            if len(gg_results["graphs"]) > 10:
                print(f"    ... 还有 {len(gg_results['graphs']) - 10} 个")
        else:
            print_fail("无 Golden Graph 文件")
            all_pass = False
    else:
        print_fail("golden_graphs 目录不存在")
        all_pass = False
    
    for issue in gg_results.get("issues", []):
        print_warn(issue)
    
    # ========== 4. Training Log 检查 ==========
    print_header("4. TRAINING LOG 检查")
    
    log_results = check_training_log(log_dir)
    
    if log_results["log_exists"]:
        print_pass("offline_training_log.jsonl 存在")
        print_info(f"总记录数: {log_results['total_records']}")
        print_info(f"成功: {log_results['success_count']} ({log_results['success_rate']*100:.1f}%)")
        print_info(f"失败: {log_results['failed_count']}")
        print_info(f"首次成功: {log_results['success_first_try']}")
        print_info(f"重试后成功: {log_results['success_after_retry']}")
        
        # 按病种统计
        if log_results["pathology_distribution"]:
            print_info(f"\n按病种统计 (前 10 个):")
            sorted_dist = sorted(
                log_results["pathology_distribution"].items(),
                key=lambda x: x[1]["success"] + x[1]["failed"],
                reverse=True
            )
            for pathology, counts in sorted_dist[:10]:
                total = counts["success"] + counts["failed"]
                rate = counts["success"] / total * 100 if total > 0 else 0
                print(f"    - {pathology}: {counts['success']}/{total} ({rate:.1f}%)")
    else:
        print_fail("offline_training_log.jsonl 不存在")
        all_pass = False
    
    for issue in log_results.get("issues", []):
        print_warn(issue)
    
    # ========== 5. 数据一致性检查 ==========
    print_header("5. 数据一致性检查")
    
    # Memory Bank vs Log
    if mb_results["total_cases"] > 0 and log_results["success_count"] > 0:
        if mb_results["total_cases"] == log_results["success_count"]:
            print_pass(f"Memory Bank 数量 ({mb_results['total_cases']}) = 训练成功数 ({log_results['success_count']})")
        else:
            print_warn(f"Memory Bank 数量 ({mb_results['total_cases']}) ≠ 训练成功数 ({log_results['success_count']})")
            print_info("  (这可能是因为增量保存尚未同步，或训练仍在进行中)")
    
    # Golden Graph source cases vs Log
    if gg_results.get("summary", {}).get("total_source_cases", 0) > 0:
        gg_total = gg_results["summary"]["total_source_cases"]
        if gg_total == log_results["success_count"]:
            print_pass(f"Golden Graph 来源 Case 总数 ({gg_total}) = 训练成功数 ({log_results['success_count']})")
        else:
            print_warn(f"Golden Graph 来源 Case 总数 ({gg_total}) ≠ 训练成功数 ({log_results['success_count']})")
    
    # ========== 总结 ==========
    print_header("验收总结")
    
    if all_pass:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ 所有关键检查通过！{Colors.ENDC}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ 存在问题，请检查上述 [FAIL] 和 [WARN] 项目{Colors.ENDC}")
    
    # 返回状态码
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())


















