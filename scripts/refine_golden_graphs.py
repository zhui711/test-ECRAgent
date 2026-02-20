#!/usr/bin/env python3
"""
Golden Graph Refinement Script
===============================

通过 LLM 对 Golden Graphs 中的 K-Nodes 进行语义归一化（聚类合并），
提高图谱质量和密度。

核心流程 (Pipeline):
1. Load: 读取原始 GG JSON
2. Pre-process: 提取所有 K-Nodes
3. LLM Clustering: 调用 GPT-4o 进行保守聚类
4. Graph Reconstruction: 合并节点、重定向边
5. Save: 写入 golden_graphs_refined/ 目录

使用方法:
    python scripts/refine_golden_graphs.py [--dry-run] [--single ID]
    
    --dry-run: 仅打印处理计划，不实际执行
    --single ID: 仅处理指定 ID 的病种 (例如 --single 01)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
import yaml

from src.utils.api_client import LLMClient
from src.graph.canonical_graph import CanonicalGraph, CanonicalKNode, CanonicalEdge
from config.prompt_refinement import (
    REFINEMENT_SYSTEM_PROMPT,
    REFINEMENT_USER_PROMPT_TEMPLATE,
    format_k_nodes_to_markdown,
    validate_clustering_result,
    normalize_clustering_result
)


# ==============================================================================
# 配置加载
# ==============================================================================

def load_config() -> Dict[str, Any]:
    """
    加载配置文件 config/settings.yaml
    
    Returns:
        配置字典
    """
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# LLM 聚类调用
# ==============================================================================

def call_llm_for_clustering(
    client: LLMClient,
    model: str,
    pathology_id: str,
    pathology_name: str,
    k_nodes: List[CanonicalKNode],
    max_retries: int = 3
) -> Tuple[Optional[List[Dict]], str]:
    """
    调用 LLM 对 K-Nodes 进行聚类。
    
    新策略：LLM 只输出需要合并的 clusters（2+ 个术语），
    未提及的术语自动保持为单例，大大减少输出 token 消耗。
    
    Args:
        client: LLM API 客户端
        model: 模型名称
        pathology_id: 病种 ID
        pathology_name: 病种名称
        k_nodes: K-Node 列表
        max_retries: 最大重试次数
    
    Returns:
        (完整聚类结果列表, 错误信息)
        - 成功时: (clusters, "") - 包含所有节点的完整聚类列表
        - 失败时: (None, error_message)
    """
    # 1. 将 K-Nodes 转换为 Markdown List
    k_nodes_list = [node.to_dict() for node in k_nodes]
    markdown_list = format_k_nodes_to_markdown(k_nodes_list)
    
    # 2. 构建 User Prompt
    user_prompt = REFINEMENT_USER_PROMPT_TEMPLATE.format(
        pathology_name=pathology_name,
        pathology_id=pathology_id,
        total_k_nodes=len(k_nodes),
        k_nodes_markdown_list=markdown_list
    )
    
    # 3. 构建消息
    messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # 4. 调用 LLM
    # 新格式下输出大大减少，8192 tokens 足够
    for attempt in range(max_retries):
        result = client.generate_json(
            messages=messages,
            model=model,
            logprobs=False,  # 不需要 logprobs
            temperature=0.0,  # 确定性输出
            max_tokens=8192,  # 只输出合并的 clusters，足够了
            max_retries=1  # 内部不重试，由外层控制
        )
        
        if result["error"]:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"LLM API Error: {result['error']}"
        
        content = result["content"]
        
        # 5. 解析 JSON
        try:
            # 尝试提取 JSON（可能被包裹在 ```json ... ``` 中）
            json_str = extract_json_from_response(content)
            
            merge_clusters = json.loads(json_str)
            
            # 6. 验证结果（新格式：只验证合并 clusters）
            is_valid, error_msg = validate_clustering_result(merge_clusters, len(k_nodes))
            
            if is_valid:
                # 7. 标准化：将 LLM 返回的合并 clusters 转换为完整聚类列表
                full_clusters = normalize_clustering_result(merge_clusters, k_nodes_list)
                return full_clusters, ""
            else:
                if attempt < max_retries - 1:
                    print(f"    [Retry {attempt + 1}] Validation failed: {error_msg}")
                    time.sleep(2)
                    continue
                return None, f"Validation Error: {error_msg}"
        
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"    [Retry {attempt + 1}] JSON parse error: {e}")
                time.sleep(2)
                continue
            return None, f"JSON Parse Error: {str(e)}"
    
    return None, "Max retries exceeded"


def extract_json_from_response(content: str) -> str:
    """
    从 LLM 响应中提取 JSON 字符串。
    
    处理多种格式：
    - ```json ... ```
    - ``` ... ```
    - 直接 JSON
    - 带有前导/尾随文本的 JSON
    
    Args:
        content: LLM 原始响应内容
    
    Returns:
        提取的 JSON 字符串
    """
    content = content.strip()
    
    # 尝试提取 ```json ... ``` 块
    if "```json" in content:
        parts = content.split("```json")
        if len(parts) > 1:
            json_part = parts[1]
            if "```" in json_part:
                return json_part.split("```")[0].strip()
    
    # 尝试提取 ``` ... ``` 块
    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            # 取第一个代码块
            return parts[1].strip()
    
    # 尝试找到 JSON 数组的边界
    start_idx = content.find('[')
    if start_idx != -1:
        # 从 [ 开始，找到匹配的 ]
        bracket_count = 0
        for i, char in enumerate(content[start_idx:], start_idx):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return content[start_idx:i+1]
    
    # 直接返回原内容
    return content


# ==============================================================================
# 图重构逻辑
# ==============================================================================

def reconstruct_graph(
    original_graph: CanonicalGraph,
    clusters: List[Dict],
    k_nodes_list: List[CanonicalKNode]
) -> Tuple[CanonicalGraph, Dict[str, Any]]:
    """
    根据聚类结果重构 Golden Graph。
    
    Args:
        original_graph: 原始 CanonicalGraph
        clusters: LLM 返回的聚类结果
        k_nodes_list: 原始 K-Nodes 列表（按顺序，用于索引映射）
    
    Returns:
        (新的 CanonicalGraph, 合并统计信息)
    """
    # 1. 创建新的 CanonicalGraph
    new_graph = CanonicalGraph(
        pathology_id=original_graph.pathology_id,
        pathology_name=original_graph.pathology_name
    )
    new_graph.version = "1.0-refined"
    new_graph.source_cases = original_graph.source_cases.copy()
    
    # 2. 构建旧 K-Node content -> 新 canonical_name 的映射
    old_to_canonical: Dict[str, str] = {}  # old_content.lower() -> canonical_name
    merged_clusters_record: List[Dict] = []  # 记录合并的聚类
    
    # 重要性优先级（用于合并时取 Max）
    importance_priority = {
        "Pathognomonic": 4,
        "Essential": 3,
        "Strong": 2,
        "Common": 1,
        "Weak": 0
    }
    
    for cluster in clusters:
        canonical_name = cluster["canonical_name"]
        original_indices = cluster["original_indices"]
        rationale = cluster.get("rationale", "")
        
        # 收集该聚类中所有原始节点
        merged_nodes = [k_nodes_list[idx - 1] for idx in original_indices]
        
        # 计算合并后的属性
        total_count = sum(node.occurrence_count for node in merged_nodes)
        
        # 取最高 importance
        max_importance = "Common"
        max_priority = 0
        for node in merged_nodes:
            priority = importance_priority.get(node.importance, 0)
            if priority > max_priority:
                max_priority = priority
                max_importance = node.importance
        
        # 合并 k_type（如果有 Pivot，则保留 Pivot）
        has_pivot = any(node.k_type == "Pivot" for node in merged_nodes)
        merged_k_type = "Pivot" if has_pivot else "General"
        
        # 合并 original_sources
        all_sources = []
        for node in merged_nodes:
            all_sources.extend(node.original_sources)
        unique_sources = list(set(all_sources))[:5]
        
        # 3. 添加新的 K-Node 到图中
        new_k_node = CanonicalKNode(
            content=canonical_name,
            k_type=merged_k_type,
            importance=max_importance,
            occurrence_count=total_count,
            original_sources=unique_sources
        )
        new_graph.k_nodes[canonical_name.lower().strip()] = new_k_node
        
        # 4. 建立映射关系
        merged_contents = []
        for node in merged_nodes:
            old_content_key = node.content.lower().strip()
            old_to_canonical[old_content_key] = canonical_name
            if node.content != canonical_name:
                merged_contents.append(node.content)
        
        # 5. 记录合并信息（仅记录多于 1 个节点的聚类）
        if len(original_indices) > 1:
            merged_clusters_record.append({
                "canonical": canonical_name,
                "merged": merged_contents,
                "original_count": len(original_indices),
                "rationale": rationale
            })
    
    # 6. 复制 P-Nodes（保持不变）
    for p_node in original_graph.p_nodes.values():
        new_graph.p_nodes[p_node.content.lower().strip()] = p_node
    
    # 7. 重定向边
    # 用于边去重的字典: (source, target, relation) -> CanonicalEdge
    new_edges: Dict[Tuple[str, str, str], CanonicalEdge] = {}
    
    # 边强度优先级
    strength_priority = {
        "Pathognomonic": 4,
        "Essential": 3,
        "Strong": 2,
        "Weak": 1,
        None: 0
    }
    
    for edge in original_graph.edges:
        # 根据边类型决定如何重定向
        if edge.edge_type == "P-K":
            # P-K 边：source 是 P-Node（不变），target 是 K-Node（可能需要重定向）
            new_source = edge.source_content  # P-Node 保持不变
            target_key = edge.target_content.lower().strip()
            new_target = old_to_canonical.get(target_key, edge.target_content)
        elif edge.edge_type == "K-D":
            # K-D 边：source 是 K-Node（可能需要重定向），target 是 D-Node（不变）
            source_key = edge.source_content.lower().strip()
            new_source = old_to_canonical.get(source_key, edge.source_content)
            new_target = edge.target_content  # D-Node 保持不变
        else:
            # 其他边类型，保持不变
            new_source = edge.source_content
            new_target = edge.target_content
        
        # 构建边的唯一键
        edge_key = (
            new_source.lower().strip(),
            new_target.lower().strip(),
            edge.relation
        )
        
        # 检查是否已存在（需要合并）
        if edge_key in new_edges:
            existing_edge = new_edges[edge_key]
            # 累加 occurrence_count
            existing_edge.occurrence_count += edge.occurrence_count
            # 取更强的 strength
            if edge.strength:
                existing_priority = strength_priority.get(existing_edge.strength, 0)
                new_priority = strength_priority.get(edge.strength, 0)
                if new_priority > existing_priority:
                    existing_edge.strength = edge.strength
        else:
            # 创建新边
            new_edges[edge_key] = CanonicalEdge(
                source_content=new_source,
                target_content=new_target,
                edge_type=edge.edge_type,
                relation=edge.relation,
                strength=edge.strength,
                occurrence_count=edge.occurrence_count
            )
    
    # 将边添加到新图中
    new_graph._edges = new_edges
    
    # 8. 构建统计信息
    stats = {
        "k_nodes_before": len(original_graph.k_nodes),
        "k_nodes_after": len(new_graph.k_nodes),
        "edges_before": len(original_graph.edges),
        "edges_after": len(new_graph.edges),
        "merged_clusters_count": len(merged_clusters_record),
        "merged_clusters": merged_clusters_record
    }
    
    return new_graph, stats


# ==============================================================================
# 添加精炼元数据
# ==============================================================================

def add_refinement_metadata(graph_dict: Dict, stats: Dict, original_k_nodes_count: int) -> Dict:
    """
    向图谱字典添加精炼相关的元数据。
    
    Args:
        graph_dict: CanonicalGraph.to_dict() 的输出
        stats: 重构统计信息
        original_k_nodes_count: 原始 K-Nodes 数量
    
    Returns:
        添加了元数据的字典
    """
    # 添加精炼版本信息到 metadata
    graph_dict["metadata"]["refinement_version"] = "1.0"
    graph_dict["metadata"]["refinement_date"] = datetime.now().strftime("%Y-%m-%d")
    graph_dict["metadata"]["original_k_nodes_count"] = original_k_nodes_count
    graph_dict["metadata"]["merged_clusters"] = stats["merged_clusters"]
    
    # 更新 statistics
    graph_dict["statistics"]["original_k_nodes_count"] = original_k_nodes_count
    graph_dict["statistics"]["merged_k_nodes_count"] = stats["merged_clusters_count"]
    graph_dict["statistics"]["compression_ratio"] = round(
        stats["k_nodes_after"] / stats["k_nodes_before"] if stats["k_nodes_before"] > 0 else 1.0, 
        3
    )
    
    return graph_dict


# ==============================================================================
# 主处理函数
# ==============================================================================

def process_single_graph(
    filepath: Path,
    output_dir: Path,
    client: LLMClient,
    model: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    处理单个 Golden Graph 文件。
    
    Args:
        filepath: 原始 GG 文件路径
        output_dir: 输出目录
        client: LLM 客户端
        model: 模型名称
        dry_run: 是否为试运行模式
    
    Returns:
        处理日志字典
    """
    filename = filepath.stem  # e.g., "01_Possible_NSTEMI___STEMI_GG"
    start_time = time.time()
    
    log_entry = {
        "filename": filename,
        "status": "pending",
        "k_nodes_before": 0,
        "k_nodes_after": 0,
        "merged_clusters_count": 0,
        "processing_time_sec": 0,
        "error_msg": ""
    }
    
    try:
        # 1. 加载原始图谱
        print(f"  [1/5] Loading {filepath.name}...")
        original_graph = CanonicalGraph.load(filepath)
        
        k_nodes_list = list(original_graph.k_nodes.values())
        log_entry["k_nodes_before"] = len(k_nodes_list)
        
        print(f"        → Loaded: {len(k_nodes_list)} K-Nodes, {len(original_graph.p_nodes)} P-Nodes, {len(original_graph.edges)} Edges")
        
        if dry_run:
            log_entry["status"] = "dry_run"
            log_entry["processing_time_sec"] = round(time.time() - start_time, 2)
            return log_entry
        
        # 2. 调用 LLM 进行聚类
        print(f"  [2/5] Calling LLM for clustering...")
        clusters, error = call_llm_for_clustering(
            client=client,
            model=model,
            pathology_id=original_graph.pathology_id,
            pathology_name=original_graph.pathology_name,
            k_nodes=k_nodes_list
        )
        
        if error:
            log_entry["status"] = "failed"
            log_entry["error_msg"] = error
            log_entry["processing_time_sec"] = round(time.time() - start_time, 2)
            print(f"        ✗ LLM Error: {error}")
            return log_entry
        
        # 统计合并情况
        merge_count = sum(1 for c in clusters if len(c.get("original_indices", [])) > 1)
        print(f"        → LLM returned {len(clusters)} clusters ({merge_count} merges)")
        
        # 3. 重构图谱
        print(f"  [3/5] Reconstructing graph...")
        new_graph, stats = reconstruct_graph(original_graph, clusters, k_nodes_list)
        
        log_entry["k_nodes_after"] = stats["k_nodes_after"]
        log_entry["merged_clusters_count"] = stats["merged_clusters_count"]
        
        print(f"        → K-Nodes: {stats['k_nodes_before']} → {stats['k_nodes_after']} "
              f"(reduced by {stats['k_nodes_before'] - stats['k_nodes_after']})")
        print(f"        → Edges: {stats['edges_before']} → {stats['edges_after']}")
        
        # 4. 添加元数据并保存
        print(f"  [4/5] Adding metadata...")
        graph_dict = new_graph.to_dict()
        graph_dict = add_refinement_metadata(graph_dict, stats, len(k_nodes_list))
        
        # 生成输出文件名
        # 从原始文件名提取 ID 和 Name，生成新文件名
        parts = filename.replace("_GG", "").split("_", 1)
        if len(parts) >= 2:
            pathology_id = parts[0]
            pathology_name = parts[1]
        else:
            pathology_id = original_graph.pathology_id
            pathology_name = original_graph.pathology_name.replace(" ", "_").replace("/", "_")
        
        output_filename = f"{pathology_id}_{pathology_name}_Refined_GG.json"
        output_path = output_dir / output_filename
        
        print(f"  [5/5] Saving to {output_path.name}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
        
        log_entry["status"] = "success"
        log_entry["output_file"] = str(output_path)
        log_entry["processing_time_sec"] = round(time.time() - start_time, 2)
        
        print(f"        ✓ Saved successfully! ({log_entry['processing_time_sec']}s)")
        
        return log_entry
    
    except Exception as e:
        import traceback
        log_entry["status"] = "failed"
        log_entry["error_msg"] = f"{type(e).__name__}: {str(e)}"
        log_entry["traceback"] = traceback.format_exc()
        log_entry["processing_time_sec"] = round(time.time() - start_time, 2)
        print(f"        ✗ Exception: {log_entry['error_msg']}")
        return log_entry


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Refine Golden Graphs by clustering K-Nodes using LLM"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Only print the processing plan without executing"
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Process only a single pathology by ID (e.g., --single 01)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="golden_graphs",
        help="Input directory containing raw Golden Graphs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="golden_graphs_refined",
        help="Output directory for refined Golden Graphs"
    )
    
    args = parser.parse_args()
    
    # 路径设置
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    log_dir = PROJECT_ROOT / "logs"
    
    print("=" * 70)
    print("Golden Graph Refinement Pipeline")
    print("=" * 70)
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Dry Run Mode:     {args.dry_run}")
    print("=" * 70)
    
    # 1. 扫描输入文件
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    gg_files = sorted(input_dir.glob("*_GG.json"))
    
    if args.single:
        # 仅处理指定 ID
        gg_files = [f for f in gg_files if f.stem.startswith(f"{args.single}_")]
        if not gg_files:
            print(f"Error: No Golden Graph found for ID '{args.single}'")
            sys.exit(1)
    
    print(f"\nFound {len(gg_files)} Golden Graph(s) to process:\n")
    for f in gg_files:
        print(f"  - {f.name}")
    print()
    
    if args.dry_run:
        print("[DRY RUN MODE] No actual processing will be performed.\n")
    
    # 2. 加载配置和初始化 LLM 客户端
    config = load_config()
    critical_config = config.get("critical_model", {})
    
    client = LLMClient(
        base_url=critical_config.get("base_url", "https://yunwu.ai/v1"),
        timeout=critical_config.get("timeout", 180)
    )
    model = critical_config.get("model_name", "gpt-4o")
    
    print(f"LLM Model: {model}")
    print(f"API Base:  {critical_config.get('base_url', 'https://yunwu.ai/v1')}")
    print("=" * 70)
    
    # 3. 处理每个 Golden Graph
    all_logs = {}
    total_start_time = time.time()
    
    for filepath in tqdm(gg_files, desc="Processing Golden Graphs", unit="file"):
        print(f"\n[Processing] {filepath.name}")
        print("-" * 60)
        
        log_entry = process_single_graph(
            filepath=filepath,
            output_dir=output_dir,
            client=client,
            model=model,
            dry_run=args.dry_run
        )
        
        # 使用文件名（不含扩展名）作为 key
        all_logs[filepath.stem] = log_entry
    
    total_time = round(time.time() - total_start_time, 2)
    
    # 4. 输出统计摘要
    print("\n" + "=" * 70)
    print("REFINEMENT SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for log in all_logs.values() if log["status"] == "success")
    failed_count = sum(1 for log in all_logs.values() if log["status"] == "failed")
    
    print(f"Total Processed:  {len(all_logs)}")
    print(f"  - Success:      {success_count}")
    print(f"  - Failed:       {failed_count}")
    print(f"  - Dry Run:      {sum(1 for log in all_logs.values() if log['status'] == 'dry_run')}")
    print(f"Total Time:       {total_time}s")
    
    # 统计合并情况
    total_before = sum(log.get("k_nodes_before", 0) for log in all_logs.values())
    total_after = sum(log.get("k_nodes_after", 0) for log in all_logs.values() if log["status"] == "success")
    total_merged = sum(log.get("merged_clusters_count", 0) for log in all_logs.values())
    
    print(f"\nK-Nodes Reduction:")
    print(f"  - Before:       {total_before}")
    print(f"  - After:        {total_after}")
    print(f"  - Reduced:      {total_before - total_after} ({round((1 - total_after/total_before)*100, 1) if total_before > 0 else 0}%)")
    print(f"  - Total Merges: {total_merged}")
    
    # 5. 保存日志
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "refinement_log.json"
        
        log_output = {
            "run_timestamp": datetime.now().isoformat(),
            "total_files": len(all_logs),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_processing_time_sec": total_time,
            "model_used": model,
            "details": all_logs
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_output, f, indent=2, ensure_ascii=False)
        
        print(f"\nLog saved to: {log_path}")
    
    print("\n" + "=" * 70)
    print("Refinement Pipeline Complete!")
    print("=" * 70)
    
    # 如果有失败，返回非零退出码
    if failed_count > 0:
        print(f"\nWarning: {failed_count} file(s) failed to process. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

