"""
Graph Tools for Phase 2-3 Integration
提供图谱序列化和独立评分功能

功能：
1. serialize_graph_to_summary: 将图数据转换为结构化自然语言摘要
2. calculate_deterministic_score: 基于固定公式计算确定性得分
3. rebuild_graph_from_json: 从 JSON 重建 MedicalGraph 对象
"""
from typing import Dict, Any, List, Tuple, Optional
from src.graph.schema import MedicalGraph


def rebuild_graph_from_json(graph_json: Dict[str, Any]) -> MedicalGraph:
    """
    从 graph_json 字典重建 MedicalGraph 对象
    
    这是一个辅助函数，用于在 workflow 中从序列化状态重建图谱对象，
    而不修改 MedicalGraph 类定义 (遵循 No-Go Zone 约束)。
    
    Args:
        graph_json: Phase 2 输出的 graph_json 字典
    
    Returns:
        重建的 MedicalGraph 实例
    """
    graph = MedicalGraph()
    
    # 恢复元数据
    case_metadata = graph_json.get("case_metadata", {})
    graph.meta["raw_text"] = case_metadata.get("raw_text", "")
    graph.meta["case_id"] = case_metadata.get("case_id", "")
    
    phase1_context = graph_json.get("phase1_context", {})
    graph.meta["initial_candidates"] = phase1_context.get("initial_candidates", [])
    graph.meta["initial_reasoning"] = phase1_context.get("initial_reasoning", "")
    
    # 获取节点和边数据
    nodes = graph_json.get("graph", {}).get("nodes", {})
    edges = graph_json.get("graph", {}).get("edges", {})
    
    p_nodes = nodes.get("p_nodes", [])
    k_nodes = nodes.get("k_nodes", [])
    d_nodes = nodes.get("d_nodes", [])
    p_k_links = edges.get("p_k_links", [])
    k_d_links = edges.get("k_d_links", [])
    
    # 重建 P-Nodes
    for p_data in p_nodes:
        p_id = p_data.get("id")
        if p_id:
            graph.graph.add_node(
                p_id,
                type="P",
                content=p_data.get("content", ""),
                original_text=p_data.get("original_text", ""),
                status=p_data.get("status", "Present"),
                source=p_data.get("source", "Unknown")
            )
    
    # 重建 K-Nodes
    for k_data in k_nodes:
        k_id = k_data.get("id")
        if k_id:
            graph.graph.add_node(
                k_id,
                type="K",
                content=k_data.get("content", ""),
                k_type=k_data.get("k_type", "General"),
                source=k_data.get("source", "Unknown"),
                importance=k_data.get("importance", "Common")
            )
    
    # 重建 D-Nodes
    for d_data in d_nodes:
        d_id = d_data.get("id")
        if d_id:
            graph.graph.add_node(
                d_id,
                type="D",
                name=d_data.get("name", ""),
                original_id=d_data.get("original_id", ""),
                initial_rank=d_data.get("initial_rank", 999),
                risk_level=d_data.get("risk_level"),
                state=d_data.get("state", "Active")
            )
    
    # 重建 P-K 边
    for pk_link in p_k_links:
        source = pk_link.get("source")
        target = pk_link.get("target")
        relation = pk_link.get("relation", "")
        if source and target:
            graph.graph.add_edge(source, target, relation=relation)
    
    # 重建 K-D 边
    for kd_link in k_d_links:
        source = kd_link.get("source")
        target = kd_link.get("target")
        relation = kd_link.get("relation", "Support")
        strength = kd_link.get("strength", "Weak")
        if source and target:
            graph.graph.add_edge(source, target, relation=relation, strength=strength)
    
    return graph


def serialize_graph_to_summary(graph: MedicalGraph) -> str:
    """
    将图数据转换为结构化的自然语言摘要，按 Candidate 分组
    
    格式示例：
    ```
    Candidate: Pulmonary Embolism (d_25) [Rank: 1]
    - Supporting Evidence: Acute Chest Pain (Match), Shortness of Breath (Match)
    - Conflicting Evidence: No history of DVT (Conflict)
    - Missing Critical Features: D-dimer elevated (Shadow)
    - Knowledge Sources: 3 General K-Nodes, 2 Pivot K-Nodes
    - Naive Score: 2.5
    ```
    
    Args:
        graph: MedicalGraph 实例
    
    Returns:
        结构化的自然语言摘要文本
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVIDENCE SUMMARY BY CANDIDATE")
    lines.append("=" * 60)
    lines.append("")
    
    # 获取所有 D-Nodes，按 initial_rank 排序
    d_nodes = graph.get_d_nodes()
    d_nodes_sorted = sorted(d_nodes, key=lambda d: d.get("initial_rank", 999))
    
    # 构建 K-Node 查找索引
    k_nodes = graph.get_k_nodes()
    k_node_map = {k["id"]: k for k in k_nodes}
    
    # 构建 P-Node 查找索引
    p_nodes = graph.get_p_nodes()
    p_node_map = {p["id"]: p for p in p_nodes}
    
    # 收集边信息
    p_k_edges = []  # (p_id, k_id, relation)
    k_d_edges = []  # (k_id, d_id, relation, strength)
    
    for source, target, data in graph.graph.edges(data=True):
        source_data = graph.graph.nodes.get(source, {})
        target_data = graph.graph.nodes.get(target, {})
        
        if source_data.get("type") == "P" and target_data.get("type") == "K":
            p_k_edges.append((source, target, data.get("relation", "")))
        elif source_data.get("type") == "K" and target_data.get("type") == "D":
            k_d_edges.append((source, target, data.get("relation", ""), data.get("strength", "")))
    
    # 为每个 D-Node 生成摘要
    for d_node in d_nodes_sorted:
        d_id = d_node["id"]
        d_name = d_node.get("name", "Unknown")
        d_rank = d_node.get("initial_rank", "?")
        d_state = d_node.get("state", "Active")
        
        # 状态标记
        state_marker = "🔴 PRUNED" if d_state == "Pruned" else ""
        
        lines.append(f"### Candidate: {d_name} ({d_id}) [Rank: {d_rank}] {state_marker}")
        lines.append("")
        
        # 收集与该 D-Node 关联的 K-Nodes
        related_k_ids = [k_id for k_id, d_id_target, _, _ in k_d_edges if d_id_target == d_id]
        
        # 分类证据
        supporting = []      # Match 的 K-Nodes
        conflicting = []     # Conflict 的 K-Nodes
        missing = []         # Void (Shadow) 的 K-Nodes
        
        general_k_count = 0
        pivot_k_count = 0
        
        for k_id in related_k_ids:
            k_node = k_node_map.get(k_id, {})
            k_content = k_node.get("content", "Unknown")
            k_type = k_node.get("k_type", "General")
            k_importance = k_node.get("importance", "Common")
            
            # 统计 K-Node 类型
            if k_type == "General":
                general_k_count += 1
            else:
                pivot_k_count += 1
            
            # 查找 P-K 关系
            pk_relation = "Unknown"
            p_content = ""
            for p_id, k_id_target, relation in p_k_edges:
                if k_id_target == k_id:
                    pk_relation = relation
                    p_node = p_node_map.get(p_id, {})
                    p_content = p_node.get("content", "")
                    break
            
            # 构建证据描述
            importance_marker = f"[{k_importance}]" if k_importance in ["Essential", "Pathognomonic"] else ""
            evidence_desc = f"{k_content} {importance_marker}".strip()
            
            if pk_relation == "Match":
                supporting.append(evidence_desc)
            elif pk_relation == "Conflict":
                conflicting.append(evidence_desc)
            elif pk_relation == "Void":
                missing.append(evidence_desc)
        
        # 输出证据分类
        if supporting:
            lines.append(f"**✅ Supporting Evidence ({len(supporting)}):**")
            for ev in supporting[:10]:  # 最多显示 10 个
                lines.append(f"  - {ev}")
            if len(supporting) > 10:
                lines.append(f"  - ... and {len(supporting) - 10} more")
            lines.append("")
        
        if conflicting:
            lines.append(f"**❌ Conflicting Evidence ({len(conflicting)}):**")
            for ev in conflicting[:10]:
                lines.append(f"  - {ev}")
            if len(conflicting) > 10:
                lines.append(f"  - ... and {len(conflicting) - 10} more")
            lines.append("")
        
        if missing:
            lines.append(f"**❓ Missing/Shadow Evidence ({len(missing)}):**")
            for ev in missing[:10]:
                lines.append(f"  - {ev}")
            if len(missing) > 10:
                lines.append(f"  - ... and {len(missing) - 10} more")
            lines.append("")
        
        # 统计信息
        lines.append(f"**📊 Statistics:**")
        lines.append(f"  - Knowledge Sources: {general_k_count} General, {pivot_k_count} Pivot")
        lines.append(f"  - Evidence Counts: {len(supporting)} Match, {len(conflicting)} Conflict, {len(missing)} Shadow")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")
    
    return "\n".join(lines)


def calculate_deterministic_score(graph: MedicalGraph) -> Dict[str, float]:
    """
    基于固定公式计算每个 Candidate 的得分（不依赖 LLM）
    
    公式：Score = (Match_Count * 1.0) - (Conflict_Count * 1.5) - (Shadow_Count * 0.1)
    
    此分数仅用于调试输出，帮助判断 Phase 2 建图本身的质量。
    
    Args:
        graph: MedicalGraph 实例
    
    Returns:
        字典：{d_id: score}
    """
    scores = {}
    
    # 获取所有 D-Nodes
    d_nodes = graph.get_d_nodes()
    
    # 收集边信息
    p_k_edges = {}  # k_id -> (p_id, relation)
    k_d_edges = {}  # d_id -> [k_id, ...]
    
    for source, target, data in graph.graph.edges(data=True):
        source_data = graph.graph.nodes.get(source, {})
        target_data = graph.graph.nodes.get(target, {})
        
        if source_data.get("type") == "P" and target_data.get("type") == "K":
            # P -> K 边
            if target not in p_k_edges:
                p_k_edges[target] = []
            p_k_edges[target].append((source, data.get("relation", "")))
        elif source_data.get("type") == "K" and target_data.get("type") == "D":
            # K -> D 边
            if target not in k_d_edges:
                k_d_edges[target] = []
            k_d_edges[target].append(source)
    
    # 计算每个 D-Node 的分数
    for d_node in d_nodes:
        d_id = d_node["id"]
        
        match_count = 0
        conflict_count = 0
        shadow_count = 0
        
        # 获取关联的 K-Nodes
        related_k_ids = k_d_edges.get(d_id, [])
        
        for k_id in related_k_ids:
            # 查找 P-K 关系
            pk_relations = p_k_edges.get(k_id, [])
            for _, relation in pk_relations:
                if relation == "Match":
                    match_count += 1
                elif relation == "Conflict":
                    conflict_count += 1
                elif relation == "Void":
                    shadow_count += 1
        
        # 应用公式
        score = (match_count * 1.0) - (conflict_count * 1.5) - (shadow_count * 0.1)
        scores[d_id] = round(score, 2)
    
    return scores


def get_evidence_breakdown(graph: MedicalGraph, d_id: str) -> Dict[str, Any]:
    """
    获取指定 D-Node 的证据分解详情
    
    Args:
        graph: MedicalGraph 实例
        d_id: D-Node ID
    
    Returns:
        证据分解字典
    """
    result = {
        "match_count": 0,
        "conflict_count": 0,
        "shadow_count": 0,
        "match_evidence": [],
        "conflict_evidence": [],
        "shadow_evidence": []
    }
    
    # 获取与该 D-Node 关联的 K-Nodes
    k_nodes_with_edges = graph.get_k_nodes_for_d(d_id)
    
    for k_node, kd_edge in k_nodes_with_edges:
        k_id = k_node["id"]
        k_content = k_node.get("content", "")
        
        # 获取 P-K 关系
        p_nodes_with_edges = graph.get_p_nodes_for_k(k_id)
        
        for p_node, pk_edge in p_nodes_with_edges:
            relation = pk_edge.get("relation", "")
            p_content = p_node.get("content", "")
            
            evidence_item = {
                "k_content": k_content,
                "p_content": p_content,
                "importance": k_node.get("importance", "Common")
            }
            
            if relation == "Match":
                result["match_count"] += 1
                result["match_evidence"].append(evidence_item)
            elif relation == "Conflict":
                result["conflict_count"] += 1
                result["conflict_evidence"].append(evidence_item)
            elif relation == "Void":
                result["shadow_count"] += 1
                result["shadow_evidence"].append(evidence_item)
    
    return result


def summarize_graph_for_critique(
    graph_json: Dict[str, Any], 
    ground_truth_id: str
) -> str:
    """
    生成面向 Critical Model (Teacher) 的图谱摘要
    
    与通用 Summary 不同，此函数专注于"纠错"场景，突出以下信息：
    1. Missing Evidence: Shadow Nodes (Void 关系)，尤其是 Essential/Pivot 级别
    2. Conflicts: 与诊断冲突的证据
    3. Support Strength: 区分 Strong/Weak 支持
    4. Ground Truth 相关性: 特别标注与正确答案相关的证据
    
    Args:
        graph_json: Phase 2 输出的图谱 JSON (MedicalGraph.to_dict() 的输出)
        ground_truth_id: 正确答案的疾病 ID (不带 d_ 前缀)
    
    Returns:
        结构化的自然语言摘要，供 Teacher 分析错误原因
    """
    from collections import defaultdict
    
    lines = []
    lines.append("=" * 60)
    lines.append("DIAGNOSTIC REASONING ANALYSIS (For Teacher Review)")
    lines.append("=" * 60)
    
    graph_data = graph_json.get("graph", {})
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", {})
    
    # 构建索引
    k_node_map = {k["id"]: k for k in nodes.get("k_nodes", [])}
    p_node_map = {p["id"]: p for p in nodes.get("p_nodes", [])}
    d_node_map = {d["id"]: d for d in nodes.get("d_nodes", [])}
    
    # 收集边信息
    pk_by_k = defaultdict(list)  # k_id -> [(p_id, relation), ...]
    kd_by_d = defaultdict(list)  # d_id -> [(k_id, relation, strength), ...]
    
    for edge in edges.get("p_k_links", []):
        pk_by_k[edge.get("target", "")].append(
            (edge.get("source", ""), edge.get("relation", ""))
        )
    
    for edge in edges.get("k_d_links", []):
        kd_by_d[edge.get("target", "")].append(
            (edge.get("source", ""), edge.get("relation", ""), edge.get("strength", ""))
        )
    
    # === Section 1: Missing Evidence (Shadow Nodes / Void Relations) ===
    lines.append("\n### 🔍 MISSING EVIDENCE (Shadow Nodes / Void Relations)")
    lines.append("These are clinical features the agent could not find in the patient narrative:")
    
    shadow_count = 0
    for k_id, pk_list in pk_by_k.items():
        for p_id, relation in pk_list:
            if relation == "Void":
                k_node = k_node_map.get(k_id, {})
                k_content = k_node.get("content", k_id)
                importance = k_node.get("importance", "Common")
                k_type = k_node.get("k_type", "General")
                
                # 标记关键缺失
                marker = ""
                if importance in ["Essential", "Pathognomonic"]:
                    marker = "⚠️ CRITICAL"
                elif k_type == "Pivot":
                    marker = "📌 PIVOT"
                
                lines.append(f"  - {k_content} [{importance}, {k_type}] {marker}")
                shadow_count += 1
    
    if shadow_count == 0:
        lines.append("  (No shadow nodes found - all features were matched)")
    else:
        lines.append(f"\n  Total Shadow Nodes: {shadow_count}")
    
    # === Section 2: Conflicting Evidence ===
    lines.append("\n### ❌ CONFLICTING EVIDENCE")
    lines.append("These are features where patient status contradicts expected findings:")
    
    conflict_count = 0
    for k_id, pk_list in pk_by_k.items():
        for p_id, relation in pk_list:
            if relation == "Conflict":
                k_node = k_node_map.get(k_id, {})
                p_node = p_node_map.get(p_id, {})
                
                k_content = k_node.get("content", k_id)
                p_content = p_node.get("content", p_id)
                p_status = p_node.get("status", "Unknown")
                
                lines.append(
                    f"  - K-Node: '{k_content}' "
                    f"<-- P-Node: '{p_content}' (Status: {p_status})"
                )
                conflict_count += 1
    
    if conflict_count == 0:
        lines.append("  (No conflicts found)")
    else:
        lines.append(f"\n  Total Conflicts: {conflict_count}")
    
    # === Section 3: Ground Truth Analysis ===
    # 确保格式正确
    gt_d_id = f"d_{ground_truth_id}" if not ground_truth_id.startswith("d_") else ground_truth_id
    gt_name = d_node_map.get(gt_d_id, {}).get("name", "Unknown")
    
    lines.append(f"\n### 🎯 GROUND TRUTH ANALYSIS: {gt_name} ({gt_d_id})")
    
    gt_evidence = kd_by_d.get(gt_d_id, [])
    if gt_evidence:
        lines.append(f"  Evidence supporting the CORRECT diagnosis:")
        
        support_list = []
        rule_out_list = []
        
        for k_id, relation, strength in gt_evidence:
            k_node = k_node_map.get(k_id, {})
            k_content = k_node.get("content", k_id)
            importance = k_node.get("importance", "Common")
            
            # 检查该 K-Node 的 P-K 关系
            pk_relations = pk_by_k.get(k_id, [])
            pk_summary = []
            for p_id, pk_rel in pk_relations:
                pk_summary.append(pk_rel)
            pk_str = ", ".join(set(pk_summary)) if pk_summary else "No P-K links"
            
            if relation == "Support":
                support_list.append(f"    ✓ {k_content} [{strength}] (P-K: {pk_str})")
            elif relation == "Rule_Out":
                rule_out_list.append(f"    ✗ {k_content} [{strength}] (P-K: {pk_str})")
        
        if support_list:
            lines.append("  **Supporting K-Nodes:**")
            lines.extend(support_list[:10])  # 限制显示数量
        
        if rule_out_list:
            lines.append("  **Rule-Out K-Nodes:**")
            lines.extend(rule_out_list[:5])
        
        # 计算支持强度
        essential_support = sum(
            1 for k_id, rel, _ in gt_evidence 
            if rel == "Support" and k_node_map.get(k_id, {}).get("importance") in ["Essential", "Pathognomonic"]
        )
        lines.append(f"\n  Essential/Pathognomonic Support Count: {essential_support}")
        
    else:
        lines.append("  ⚠️ NO EVIDENCE found supporting the correct diagnosis!")
        lines.append("  This suggests the agent completely missed key features for this condition.")
    
    # === Section 4: All Candidates Summary ===
    lines.append("\n### 📊 ALL CANDIDATES EVIDENCE SUMMARY")
    
    for d_node in nodes.get("d_nodes", []):
        d_id = d_node["id"]
        d_name = d_node.get("name", "Unknown")
        d_rank = d_node.get("initial_rank", "?")
        
        evidence = kd_by_d.get(d_id, [])
        
        support_count = sum(1 for _, rel, _ in evidence if rel == "Support")
        rule_out_count = sum(1 for _, rel, _ in evidence if rel == "Rule_Out")
        
        # 计算 Match/Conflict/Void 统计
        match_count = 0
        conflict_count = 0
        void_count = 0
        
        for k_id, _, _ in evidence:
            for p_id, pk_rel in pk_by_k.get(k_id, []):
                if pk_rel == "Match":
                    match_count += 1
                elif pk_rel == "Conflict":
                    conflict_count += 1
                elif pk_rel == "Void":
                    void_count += 1
        
        # 标记 Ground Truth
        marker = " ← GROUND TRUTH" if d_id == gt_d_id else ""
        
        lines.append(
            f"  - {d_name} ({d_id}) [Rank: {d_rank}]{marker}"
        )
        lines.append(
            f"    K-D: {support_count} Support, {rule_out_count} Rule_Out | "
            f"P-K: {match_count} Match, {conflict_count} Conflict, {void_count} Void"
        )
    
    # === Section 5: Patient Features Summary ===
    lines.append("\n### 👤 PATIENT FEATURES (P-Nodes)")
    
    present_features = []
    absent_features = []
    
    for p_node in nodes.get("p_nodes", []):
        content = p_node.get("content", "")
        status = p_node.get("status", "Present")
        
        if status == "Present":
            present_features.append(content)
        elif status == "Absent":
            absent_features.append(content)
    
    if present_features:
        lines.append(f"  **Present ({len(present_features)}):** " + ", ".join(present_features[:15]))
        if len(present_features) > 15:
            lines.append(f"    ... and {len(present_features) - 15} more")
    
    if absent_features:
        lines.append(f"  **Absent ({len(absent_features)}):** " + ", ".join(absent_features[:10]))
        if len(absent_features) > 10:
            lines.append(f"    ... and {len(absent_features) - 10} more")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def build_prompt_with_hint(base_prompt: str, global_hint: Optional[str]) -> str:
    """
    将 Teacher Hint 附加到 Prompt 末尾
    
    用于 Offline Training 的重试流程，将 Critical Model 的反馈
    注入到各 Phase 的 System Prompt 中。
    
    Args:
        base_prompt: 原始 System Prompt
        global_hint: Teacher 的反馈文本（可能为 None 或空）
    
    Returns:
        附加了 Hint 的 Prompt
    """
    if global_hint and global_hint.strip():
        return base_prompt + f"\n\n[TEACHER FEEDBACK - IMPORTANT]:\n{global_hint}"
    return base_prompt


# ==================== 别名函数 (兼容 Online Inference) ====================

def summarize_graph_for_judge(
    graph_json: Dict[str, Any], 
    naive_scores: Optional[Dict[str, float]] = None
) -> str:
    """
    为 Phase 3 Judge 生成 Rich Structured Summary
    
    这是重构后的核心摘要函数，用于替代原始 JSON 输入。
    格式遵循 PHASE3_REFACTOR_PLAN.md 的设计。
    
    设计目标：
    - 高信息密度，替代原始 JSON
    - 禁止更改医学实体内容，仅改变呈现形式
    - Shadow 只列出 Pathognomonic 类型的缺失特征
    
    Args:
        graph_json: Phase 2 输出的 graph_json 字典
        naive_scores: 预计算的朴素评分字典 {d_id: score}（可选）
    
    Returns:
        Rich Structured Summary 文本
    """
    from collections import defaultdict
    
    # 提取节点和边
    graph_data = graph_json.get("graph", {})
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", {})
    
    p_nodes = nodes.get("p_nodes", [])
    k_nodes = nodes.get("k_nodes", [])
    d_nodes = nodes.get("d_nodes", [])
    p_k_links = edges.get("p_k_links", [])
    k_d_links = edges.get("k_d_links", [])
    
    # 构建索引
    p_node_map = {p["id"]: p for p in p_nodes}
    k_node_map = {k["id"]: k for k in k_nodes}
    d_node_map = {d["id"]: d for d in d_nodes}
    
    # 构建 K-Node -> P-K 关系映射
    pk_by_k = defaultdict(list)  # k_id -> [(p_id, relation), ...]
    for edge in p_k_links:
        k_id = edge.get("target", "")
        p_id = edge.get("source", "")
        relation = edge.get("relation", "")
        pk_by_k[k_id].append((p_id, relation))
    
    # 构建 D-Node -> K-Nodes 映射
    kd_by_d = defaultdict(list)  # d_id -> [k_id, ...]
    for edge in k_d_links:
        d_id = edge.get("target", "")
        k_id = edge.get("source", "")
        kd_by_d[d_id].append(k_id)
    
    # 如果没有传入 naive_scores，自动计算
    if naive_scores is None:
        graph = rebuild_graph_from_json(graph_json)
        naive_scores = calculate_deterministic_score(graph)
    
    # 按 initial_rank 排序 D-Nodes
    d_nodes_sorted = sorted(d_nodes, key=lambda d: d.get("initial_rank", 999))
    
    lines = []
    lines.append("=" * 60)
    lines.append("RICH STRUCTURED SUMMARY (Phase 3 Evidence)")
    lines.append("=" * 60)
    lines.append("")
    
    for d_node in d_nodes_sorted:
        d_id = d_node.get("id", "")
        d_name = d_node.get("name", "Unknown")
        d_rank = d_node.get("initial_rank", "?")
        d_state = d_node.get("state", "Active")
        
        # 获取 Naive Score
        score = naive_scores.get(d_id, 0.0)
        score_str = f"+{score:.1f}" if score >= 0 else f"{score:.1f}"
        
        # 状态标记
        state_marker = " 🔴 PRUNED" if d_state == "Pruned" else ""
        
        # Header
        lines.append(f"### [{d_name}] (ID: {d_id}) – Naive Score: **{score_str}**{state_marker}")
        lines.append(f"    Phase 1 Rank: {d_rank}")
        lines.append("")
        
        # 收集证据
        match_evidence = []
        conflict_evidence = []
        shadow_pathognomonic = []  # 只收集 Pathognomonic 类型
        has_pivot_match = False
        
        related_k_ids = kd_by_d.get(d_id, [])
        
        for k_id in related_k_ids:
            k_node = k_node_map.get(k_id, {})
            k_content = k_node.get("content", k_id)
            k_importance = k_node.get("importance", "Weak")
            k_type = k_node.get("k_type", "General")
            
            # 获取 P-K 关系
            pk_relations = pk_by_k.get(k_id, [])
            
            for p_id, relation in pk_relations:
                p_node = p_node_map.get(p_id, {})
                p_content = p_node.get("content", p_id)
                p_status = p_node.get("status", "Present")
                
                # 构建证据描述
                importance_tag = f"[{k_importance}]" if k_importance in ["Essential", "Pathognomonic", "Strong"] else ""
                pivot_tag = " [PIVOT]" if k_type == "Pivot" else ""
                
                if relation == "Match":
                    match_evidence.append(f'{p_content} ↔ "{k_content}" {importance_tag}{pivot_tag}')
                    # 检查 Pivot Match
                    if k_type == "Pivot":
                        has_pivot_match = True
                        
                elif relation == "Conflict":
                    conflict_evidence.append(f'{p_content} (Status: {p_status}) ✗ "{k_content}" {importance_tag}')
                    
                elif relation == "Void":
                    # **关键逻辑**: 只收集 Pathognomonic 类型的 Shadow
                    if k_importance == "Pathognomonic":
                        shadow_pathognomonic.append(f'"{k_content}" [Pathognomonic] - MISSING')
        
        # 输出 Evidence Sections
        # [+] MATCH
        if match_evidence:
            lines.append(f"• **[+] MATCH ({len(match_evidence)}):**")
            for ev in match_evidence[:8]:  # 限制显示数量
                lines.append(f"    - {ev}")
            if len(match_evidence) > 8:
                lines.append(f"    - ... and {len(match_evidence) - 8} more")
        else:
            lines.append("• **[+] MATCH (0):** None")
        lines.append("")
        
        # [-] CONFLICT
        if conflict_evidence:
            lines.append(f"• **[-] CONFLICT ({len(conflict_evidence)}):**")
            for ev in conflict_evidence[:5]:
                lines.append(f"    - {ev}")
            if len(conflict_evidence) > 5:
                lines.append(f"    - ... and {len(conflict_evidence) - 5} more")
        else:
            lines.append("• **[-] CONFLICT (0):** None")
        lines.append("")
        
        # [?] CRITICAL MISSING (Shadow - Pathognomonic Only)
        if shadow_pathognomonic:
            lines.append(f"• **[?] CRITICAL MISSING ({len(shadow_pathognomonic)} Pathognomonic):**")
            for ev in shadow_pathognomonic[:5]:
                lines.append(f"    - {ev}")
            if len(shadow_pathognomonic) > 5:
                lines.append(f"    - ... and {len(shadow_pathognomonic) - 5} more")
        else:
            lines.append("• **[?] CRITICAL MISSING:** None (No Pathognomonic features missing)")
        lines.append("")
        
        # Pivot Status
        pivot_status = "✅ YES (Pivot Feature Matched)" if has_pivot_match else "❌ NO"
        lines.append(f"• **Pivot Status:** {pivot_status}")
        lines.append("")
        lines.append("-" * 50)
        lines.append("")
    
    return "\n".join(lines)


def calculate_naive_scores(graph_json: Dict[str, Any]) -> Dict[str, float]:
    """
    计算朴素评分
    
    这是 calculate_deterministic_score 的别名，接受 graph_json 字典输入。
    
    Args:
        graph_json: Phase 2 输出的 graph_json 字典
    
    Returns:
        字典：{d_id: score}
    """
    graph = rebuild_graph_from_json(graph_json)
    return calculate_deterministic_score(graph)






