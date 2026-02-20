"""
Medical Graph Schema
定义 MedicalGraph 类，用于管理 P-Nodes, K-Nodes, D-Nodes 及其边关系

核心设计原则（基于算法流程图）：
- P-Nodes: 患者事实节点，状态为 Present/Absent/Missing
- K-Nodes: 医学知识节点，类型为 General (Wiki) / Pivot (PubMed)
- D-Nodes: 诊断候选节点，状态为 Active/Pruned
- P-K 边: Match (P=Present 且匹配 K) / Conflict (P=Absent 但 K 要求存在) / Void (P=Missing，影子节点)
- K-D 边: Support / Rule_Out，强度为 Pathognomonic/Essential/Strong/Weak

v2.0 新增字段（Offline Golden Graph 支持）：
- source_type: 节点来源类型 ("GoldenGraph" | "LiveSearch")
  - GoldenGraph: 来自离线金标准图谱的预制知识
  - LiveSearch: 来自在线检索（PubMed/OpenTargets）的实时知识
- provenance: 边的来源 ("GoldenGraph" | "LiveInference")
  - GoldenGraph: 继承自离线图谱的边
  - LiveInference: 在本次 Online 推理中新建立的边
"""
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple, Literal
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP


# ==================== 类型定义 ====================

# 节点来源类型：区分离线金标准图谱 vs 在线实时检索
SourceType = Literal["GoldenGraph", "LiveSearch"]

# 边的来源：区分预制边 vs 实时推理边
EdgeProvenance = Literal["GoldenGraph", "LiveInference"]


class MedicalGraph:
    """
    医学因果图谱类
    使用 networkx.DiGraph 管理拓扑结构，对外暴露 JSON 序列化方法
    """
    
    def __init__(self):
        """初始化图谱"""
        self.graph = nx.DiGraph()
        self.meta = {}  # 存储 raw_text, case_id 等元数据
        
        # 节点计数器
        self._p_counter = 0
        self._k_counter = 0
        self._d_counter = 0
    
    # ==================== Node Operations ====================
    
    def add_p_node(
        self, 
        node_id: Optional[str] = None,
        content: str = "",
        original_text: str = "",
        status: str = "Present",  # Present, Absent, Missing
        source: str = "Phase1",
        source_type: SourceType = "LiveSearch"
    ) -> str:
        """
        添加病人事实节点 (P-Node)
        
        Args:
            node_id: 节点ID，如果为None则自动生成
            content: 症状/体征名称（医学术语）
            original_text: 原始文本片段
            status: 状态 (Present/Absent/Missing)
                - Present: 患者明确存在该症状
                - Absent: 患者明确否认该症状
                - Missing: 信息缺失（Shadow Node，由 Phase 2 创建）
            source: 来源 (Phase1/Phase2_Recall/Phase2_Shadow)
                - 用于溯源具体的提取模块，对 Debug 很重要
            source_type: 节点来源类型 (GoldenGraph/LiveSearch)
                - GoldenGraph: 来自离线金标准图谱
                - LiveSearch: 来自在线检索（默认值）
        
        Returns:
            节点ID
        """
        if node_id is None:
            self._p_counter += 1
            node_id = f"p_{self._p_counter}"
        
        self.graph.add_node(
            node_id,
            type="P",
            content=content,
            original_text=original_text,
            status=status,
            source=source,
            source_type=source_type  # 新增字段：区分离线/在线来源
        )
        return node_id
    
    def add_k_node(
        self,
        node_id: Optional[str] = None,
        content: str = "",
        k_type: str = "General",  # General, Pivot
        source: str = "Wiki_API",  # Wiki_API, PubMed_API
        importance: str = "Common",  # Essential, Pathognomonic, Common
        source_type: SourceType = "LiveSearch"
    ) -> str:
        """
        添加医学知识节点 (K-Node)
        
        Args:
            node_id: 节点ID，如果为None则自动生成
            content: 知识点描述（症状/体征名称）
            k_type: 知识类型
                - General: 来自 Wikipedia 的通用知识
                - Pivot: 来自 PubMed 的鉴别特征
            source: 来源 (Wiki_API/PubMed_API)
                - 用于溯源具体的提取模块，对 Debug 很重要
            importance: 重要性
                - Essential: 必要症状（缺失即排除）
                - Pathognomonic: 特征性症状（存在即确诊）
                - Common: 常见症状
            source_type: 节点来源类型 (GoldenGraph/LiveSearch)
                - GoldenGraph: 来自离线金标准图谱
                - LiveSearch: 来自在线检索（默认值）
        
        Returns:
            节点ID
        """
        if node_id is None:
            self._k_counter += 1
            node_id = f"k_{self._k_counter}"
        
        self.graph.add_node(
            node_id,
            type="K",
            content=content,
            k_type=k_type,
            source=source,
            importance=importance,
            source_type=source_type  # 新增字段：区分离线/在线来源
        )
        return node_id
    
    def add_d_node(
        self,
        node_id: Optional[str] = None,
        name: str = "",
        original_id: str = "",
        initial_rank: int = 1,
        risk_level: Optional[str] = None,
        state: str = "Active"  # Active, Pruned
    ) -> str:
        """
        添加诊断候选节点 (D-Node)
        
        Args:
            node_id: 节点ID，如果为None则自动生成 (格式: d_{original_id})
            name: 疾病名称
            original_id: 原始ID (01-49)
            initial_rank: 初始排名 (1-based)
            risk_level: 风险等级 (High/Low)
            state: 状态
                - Active: 活跃候选
                - Pruned: 已被排除（Fatal Conflict）
        
        Returns:
            节点ID
        """
        if node_id is None:
            if original_id:
                node_id = f"d_{original_id}"
            else:
                self._d_counter += 1
                node_id = f"d_{self._d_counter}"
        
        self.graph.add_node(
            node_id,
            type="D",
            name=name,
            original_id=original_id,
            initial_rank=initial_rank,
            risk_level=risk_level,
            state=state
        )
        return node_id
    
    # ==================== Edge Operations ====================
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        provenance: EdgeProvenance = "LiveInference",
        **kwargs
    ):
        """
        添加边
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relation: 关系类型
                - P-K 边: Match, Conflict, Void
                - K-D 边: Support, Rule_Out
            provenance: 边的来源 (GoldenGraph/LiveInference)
                - GoldenGraph: 继承自离线金标准图谱的边
                - LiveInference: 在本次 Online 推理中新建立的边（默认值）
            **kwargs: 额外属性
                - strength: Pathognomonic, Essential, Strong, Weak (用于 K-D 边)
        """
        edge_attrs = {
            "relation": relation,
            "provenance": provenance  # 新增字段：区分预制边/实时边
        }
        edge_attrs.update(kwargs)
        self.graph.add_edge(source_id, target_id, **edge_attrs)
    
    def add_pk_edge(
        self,
        p_id: str,
        k_id: str,
        relation: str,  # Match, Conflict, Void
        provenance: EdgeProvenance = "LiveInference"
    ):
        """
        添加 P-K 边（证据验证边）
        
        根据算法流程图 Line 11-27:
        - Match: P.status == "Present" 且 K 对应的症状存在
        - Conflict: P.status == "Absent" 但 K 要求该症状存在
        - Void: P.status == "Missing"（影子节点）
        
        Args:
            p_id: P-Node ID
            k_id: K-Node ID
            relation: 关系类型 (Match/Conflict/Void)
            provenance: 边的来源 (GoldenGraph/LiveInference)
        """
        self.add_edge(p_id, k_id, relation=relation, provenance=provenance)
    
    def add_kd_edge(
        self,
        k_id: str,
        d_id: str,
        relation: str,  # Support, Rule_Out
        strength: str = "Weak",  # Pathognomonic, Essential, Strong, Weak
        provenance: EdgeProvenance = "LiveInference"
    ):
        """
        添加 K-D 边（病理生理边）
        
        根据算法流程图:
        - Support: K 支持 D
        - Rule_Out: K 排除 D
        
        Args:
            k_id: K-Node ID
            d_id: D-Node ID
            relation: 关系类型 (Support/Rule_Out)
            strength: 边的强度
            provenance: 边的来源 (GoldenGraph/LiveInference)
        """
        self.add_edge(k_id, d_id, relation=relation, strength=strength, provenance=provenance)
    
    def update_edge(
        self,
        source_id: str,
        target_id: str,
        **kwargs
    ):
        """更新边的属性"""
        if self.graph.has_edge(source_id, target_id):
            for key, value in kwargs.items():
                self.graph[source_id][target_id][key] = value
    
    def remove_edge(self, source_id: str, target_id: str):
        """移除边"""
        if self.graph.has_edge(source_id, target_id):
            self.graph.remove_edge(source_id, target_id)
    
    # ==================== Node State Operations ====================
    
    def update_p_node_status(self, p_id: str, new_status: str, source: str = None):
        """
        更新 P-Node 状态
        
        用于 Phase 2 Step 3 Recall 阶段：
        - Missing -> Present（回溯发现原文提到）
        - Missing -> Absent（回溯发现原文否认）
        """
        if self.graph.has_node(p_id):
            self.graph.nodes[p_id]["status"] = new_status
            if source:
                self.graph.nodes[p_id]["source"] = source
    
    def prune_d_node(self, d_id: str):
        """
        标记 D-Node 为 Pruned（Fatal Conflict 排除）
        
        根据算法流程图 Line 22-24:
        如果 status == "Conflict" 且 K.type == "Essential"，
        则 d.state <- Pruned
        """
        if self.graph.has_node(d_id):
            self.graph.nodes[d_id]["state"] = "Pruned"
    
    def set_d_node_safeguard(self, d_id: str, safeguard: bool = True):
        """
        设置 D-Node 的 safeguard 属性（防止误排除高风险疾病）
        
        根据算法流程图 Line 43-46:
        如果 RiskMap[d] == High 且 HasFatalConflict(G, d) == False，
        则 d.safeguard <- True
        """
        if self.graph.has_node(d_id):
            self.graph.nodes[d_id]["safeguard"] = safeguard
    
    # ==================== Query Operations ====================
    
    def get_p_nodes(self) -> List[Dict[str, Any]]:
        """获取所有 P-Nodes"""
        return [
            {**data, "id": node_id}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "P"
        ]
    
    def get_k_nodes(self) -> List[Dict[str, Any]]:
        """获取所有 K-Nodes"""
        return [
            {**data, "id": node_id}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "K"
        ]
    
    def get_d_nodes(self) -> List[Dict[str, Any]]:
        """获取所有 D-Nodes"""
        return [
            {**data, "id": node_id}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "D"
        ]
    
    def get_active_d_nodes(self) -> List[Dict[str, Any]]:
        """
        获取所有 Active 状态的 D-Nodes
        
        根据算法流程图 Line 30:
        D_active <- {d ∈ D_top5 | d.state ≠ Pruned}
        """
        return [
            {**data, "id": node_id}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "D" and data.get("state") != "Pruned"
        ]
    
    def get_p_node_by_id(self, p_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取 P-Node"""
        if self.graph.has_node(p_id):
            data = self.graph.nodes[p_id]
            if data.get("type") == "P":
                return {**data, "id": p_id}
        return None
    
    def get_k_node_by_id(self, k_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取 K-Node"""
        if self.graph.has_node(k_id):
            data = self.graph.nodes[k_id]
            if data.get("type") == "K":
                return {**data, "id": k_id}
        return None
    
    def get_d_node_by_id(self, d_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取 D-Node"""
        if self.graph.has_node(d_id):
            data = self.graph.nodes[d_id]
            if data.get("type") == "D":
                return {**data, "id": d_id}
        return None
    
    def find_p_node_by_content(self, content: str) -> Optional[Dict[str, Any]]:
        """
        根据内容模糊匹配 P-Node
        用于将 K-Node 与已有的 P-Node 关联
        """
        content_lower = content.lower().strip()
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "P":
                p_content = data.get("content", "").lower().strip()
                # 精确匹配或包含匹配
                if content_lower == p_content or content_lower in p_content or p_content in content_lower:
                    return {**data, "id": node_id}
        return None
    
    def find_k_node_by_content(self, content: str) -> Optional[Dict[str, Any]]:
        """
        根据内容模糊匹配 K-Node
        
        用于 Hybrid Engine 检查 K-Node 是否已存在（去重）。
        
        Args:
            content: 要查找的 K-Node 内容
        
        Returns:
            匹配的 K-Node 字典（含 id），或 None
        """
        content_lower = content.lower().strip()
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "K":
                k_content = data.get("content", "").lower().strip()
                # 精确匹配或包含匹配
                if content_lower == k_content or content_lower in k_content or k_content in content_lower:
                    return {**data, "id": node_id}
        return None
    
    def get_void_k_nodes(self) -> List[Dict[str, Any]]:
        """
        获取所有 Void 状态的 K-Nodes（需要 Recall 的 Shadow Nodes）
        
        根据算法流程图 Line 14-21:
        如果 status == "Missing"，则需要 Backtrack Raw Text
        
        返回所有通过 Void 边连接的 K-Nodes
        """
        void_k_nodes = []
        
        for source, target, data in self.graph.edges(data=True):
            if data.get("relation") == "Void":
                source_data = self.graph.nodes.get(source, {})
                target_data = self.graph.nodes.get(target, {})
                
                # P -> K 边，且 P.status == "Missing"
                if source_data.get("type") == "P" and target_data.get("type") == "K":
                    k_node = {**target_data, "id": target}
                    # 同时记录对应的 P-Node ID
                    k_node["shadow_p_id"] = source
                    if k_node not in void_k_nodes:
                        void_k_nodes.append(k_node)
        
        return void_k_nodes
    
    def get_k_nodes_for_d(self, d_id: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        获取与指定 D-Node 关联的所有 K-Nodes 及其边信息
        
        Returns:
            List of (k_node_dict, edge_dict)
        """
        results = []
        for source, target, data in self.graph.edges(data=True):
            if target == d_id:
                source_data = self.graph.nodes.get(source, {})
                if source_data.get("type") == "K":
                    k_node = {**source_data, "id": source}
                    edge_info = {**data}
                    results.append((k_node, edge_info))
        return results
    
    def get_p_nodes_for_k(self, k_id: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        获取与指定 K-Node 关联的所有 P-Nodes 及其边信息
        
        Returns:
            List of (p_node_dict, edge_dict)
        """
        results = []
        for source, target, data in self.graph.edges(data=True):
            if target == k_id:
                source_data = self.graph.nodes.get(source, {})
                if source_data.get("type") == "P":
                    p_node = {**source_data, "id": source}
                    edge_info = {**data}
                    results.append((p_node, edge_info))
        return results
    
    def has_fatal_conflict(self, d_id: str) -> bool:
        """
        检查 D-Node 是否有 Fatal Conflict
        
        根据算法流程图 Line 22-24 和 Phase 3 Line 44:
        如果 K.importance == "Essential" 且 P-K.relation == "Conflict"，
        则存在 Fatal Conflict
        """
        k_nodes_with_edges = self.get_k_nodes_for_d(d_id)
        
        for k_node, kd_edge in k_nodes_with_edges:
            k_id = k_node["id"]
            importance = k_node.get("importance", "Common")
            
            # 检查是否是 Essential 或 Pathognomonic
            if importance in ["Essential", "Pathognomonic"]:
                # 检查 P-K 边是否是 Conflict
                p_nodes_with_edges = self.get_p_nodes_for_k(k_id)
                for p_node, pk_edge in p_nodes_with_edges:
                    if pk_edge.get("relation") == "Conflict":
                        return True
        
        return False
    
    def count_shadow_nodes(self, d_id: str) -> int:
        """
        计算与 D-Node 关联的 Shadow Nodes 数量
        
        根据算法流程图 Line 49:
        Ratio_shadow <- CountShadows(d_initial) / CountTotalEvidence(d_initial)
        """
        k_nodes_with_edges = self.get_k_nodes_for_d(d_id)
        shadow_count = 0
        
        for k_node, kd_edge in k_nodes_with_edges:
            k_id = k_node["id"]
            p_nodes_with_edges = self.get_p_nodes_for_k(k_id)
            for p_node, pk_edge in p_nodes_with_edges:
                if pk_edge.get("relation") == "Void":
                    shadow_count += 1
        
        return shadow_count
    
    def count_total_evidence(self, d_id: str) -> int:
        """计算与 D-Node 关联的总证据数量"""
        return len(self.get_k_nodes_for_d(d_id))
    
    def calculate_explanatory_coverage(self, d_id: str) -> float:
        """
        计算解释覆盖率
        
        根据算法流程图 Line 54-55:
        d* <- argmax_{d∈D_active} (ExplanatoryCoverage(d, P))
        
        公式: Match 数量 - Void 数量 * 0.5 - Conflict 数量
        """
        k_nodes_with_edges = self.get_k_nodes_for_d(d_id)
        
        match_count = 0
        void_count = 0
        conflict_count = 0
        
        for k_node, kd_edge in k_nodes_with_edges:
            k_id = k_node["id"]
            p_nodes_with_edges = self.get_p_nodes_for_k(k_id)
            
            for p_node, pk_edge in p_nodes_with_edges:
                relation = pk_edge.get("relation", "")
                if relation == "Match":
                    match_count += 1
                elif relation == "Void":
                    void_count += 1
                elif relation == "Conflict":
                    conflict_count += 1
        
        # 根据 Phase 3 文档的公式
        coverage = match_count - (void_count * 0.5) - conflict_count
        return coverage
    
    # ==================== Serialization ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为 Master JSON Schema
        
        向后兼容说明：
        - 保持原有字段结构不变
        - 新增 source_type 字段到 P-Nodes 和 K-Nodes
        - 新增 provenance 字段到边
        - 如果节点/边没有新字段，使用默认值确保兼容性
        
        Returns:
            符合规范的 JSON 字典
        """
        # 构建节点列表（确保新增字段有默认值）
        p_nodes = []
        for node in self.get_p_nodes():
            # 确保 source_type 字段存在（向后兼容）
            if "source_type" not in node:
                node["source_type"] = "LiveSearch"
            p_nodes.append(node)
        
        k_nodes = []
        for node in self.get_k_nodes():
            # 确保 source_type 字段存在（向后兼容）
            if "source_type" not in node:
                node["source_type"] = "LiveSearch"
            k_nodes.append(node)
        
        d_nodes = self.get_d_nodes()
        
        # 构建边列表（新增 provenance 字段）
        p_k_links = []
        k_d_links = []
        
        for source, target, data in self.graph.edges(data=True):
            source_data = self.graph.nodes.get(source, {})
            target_data = self.graph.nodes.get(target, {})
            
            edge_info = {
                "source": source,
                "target": target,
                "relation": data.get("relation", "")
            }
            
            # 新增 provenance 字段（向后兼容：默认为 LiveInference）
            edge_info["provenance"] = data.get("provenance", "LiveInference")
            
            # 根据边类型分类
            if source_data.get("type") == "P" and target_data.get("type") == "K":
                p_k_links.append(edge_info)
            elif source_data.get("type") == "K" and target_data.get("type") == "D":
                edge_info["strength"] = data.get("strength", "Weak")
                k_d_links.append(edge_info)
        
        return {
            "case_metadata": {
                "raw_text": self.meta.get("raw_text", ""),
                "case_id": self.meta.get("case_id", "")
            },
            "phase1_context": {
                "initial_candidates": self.meta.get("initial_candidates", []),
                "initial_reasoning": self.meta.get("initial_reasoning", "")
            },
            "graph": {
                "nodes": {
                    "p_nodes": p_nodes,
                    "k_nodes": k_nodes,
                    "d_nodes": d_nodes
                },
                "edges": {
                    "p_k_links": p_k_links,
                    "k_d_links": k_d_links
                }
            }
        }
    
    # ==================== Factory Method ====================
    
    @classmethod
    def from_phase1(cls, phase1_result: Dict[str, Any], raw_text: str) -> "MedicalGraph":
        """
        从 Phase 1 结果初始化图谱（胶水逻辑）
        
        根据算法流程图 Line 1-4:
        1. P <- ExtractFeatures(T)  # Track B: Structured Patient Features
        2. D_top5, d_initial <- LLM_fast(T, D_all)  # Track A: Initial Candidates
        3. G <- InitGraph(P, D_top5)
        
        关键修正（根据 05_Phase1_DualTrack_Refactor.md）：
        P-Nodes 必须从 track_b_output.p_nodes 读取，而非 structured_analysis。
        structured_analysis 仅用于提高 Track A 的 CoT 质量。
        
        Args:
            phase1_result: Phase 1 的输出结果（来自 Phase1Manager）
            raw_text: 原始病历文本
        
        Returns:
            MedicalGraph 实例
        """
        graph = cls()
        
        # 存储元数据
        graph.meta["raw_text"] = raw_text
        graph.meta["initial_candidates"] = phase1_result.get("top_candidates", [])
        graph.meta["initial_reasoning"] = phase1_result.get("differential_reasoning", "")
        
        # 1. 初始化 D-Nodes (From Track A: ID -> Name 解析)
        top_candidates = phase1_result.get("top_candidates", [])
        for rank, d_id in enumerate(top_candidates):
            d_name = DIAGNOSIS_ID_MAP.get(d_id, "Unknown")
            graph.add_d_node(
                node_id=f"d_{d_id}",
                name=d_name,
                original_id=d_id,
                initial_rank=rank + 1
            )
        
        # 2. 初始化 P-Nodes (From Track B: Problem Representation)
        # 关键修正：从 track_b_output.p_nodes 读取
        track_b_output = phase1_result.get("track_b_output", {})
        p_nodes_data = track_b_output.get("p_nodes", [])
        
        if p_nodes_data:
            print(f"[MedicalGraph] Initializing {len(p_nodes_data)} P-Nodes from Track B")
            for p_data in p_nodes_data:
                if not isinstance(p_data, dict):
                    continue
                
                # 从 Track B 输出中提取字段
                p_id = p_data.get("id", None)
                content = p_data.get("content", "")
                original_text = p_data.get("original_text", "")
                status = p_data.get("status", "Present")
                
                if not content:
                    continue
                
                graph.add_p_node(
                    node_id=p_id,
                    content=content,
                    original_text=original_text,
                    status=status,
                    source="Phase1_TrackB"
                )
        else:
            # Fallback: 尝试从 structured_analysis 读取（兼容旧格式）
            print("[MedicalGraph] WARNING: No P-Nodes found in Track B output, trying structured_analysis fallback...")
            structured_analysis = phase1_result.get("structured_analysis", {})
            
            if structured_analysis:
                # 处理 positive_findings
                positive_findings = structured_analysis.get("positive_findings", [])
                for finding in positive_findings:
                    if isinstance(finding, dict):
                        content = finding.get("finding", "")
                        status = finding.get("status", "Present")
                    elif isinstance(finding, str):
                        content = finding.strip()
                        status = "Present"
                    else:
                        continue
                    
                    if content:
                        graph.add_p_node(
                            content=content,
                            status=status,
                            source="Phase1_Fallback"
                        )
                
                # 处理 negative_findings
                negative_findings = structured_analysis.get("negative_findings", [])
                for finding in negative_findings:
                    if isinstance(finding, dict):
                        content = finding.get("finding", "")
                        status = "Absent"
                    elif isinstance(finding, str):
                        content = finding.strip()
                        if content.lower().startswith("no "):
                            content = content[3:].strip()
                        status = "Absent"
                    else:
                        continue
                    
                    if content:
                        graph.add_p_node(
                            content=content,
                            status=status,
                            source="Phase1_Fallback"
                        )
        
        print(f"[MedicalGraph] Initialized with {len(graph.get_d_nodes())} D-Nodes, {len(graph.get_p_nodes())} P-Nodes")
        
        return graph
    
    def create_shadow_p_node(
        self, 
        k_content: str,
        source_type: SourceType = "LiveSearch"
    ) -> str:
        """
        为 Void K-Node 创建对应的 Shadow P-Node
        
        根据算法流程图 Line 16-17:
        如果 P_new is Empty，则 AddShadowNode(G, k, d_i)
        
        Args:
            k_content: K-Node 的内容
            source_type: 节点来源类型 (GoldenGraph/LiveSearch)
        
        Returns:
            新创建的 P-Node ID
        """
        return self.add_p_node(
            content=k_content,
            status="Missing",
            source="Phase2_Shadow",
            source_type=source_type
        )
    
    # ==================== Cascade Operations (Phase 2 Strict Pruning) ====================
    
    def cascade_remove_node(self, node_id: str) -> int:
        """
        级联删除节点及其所有关联边
        
        用于 Phase 2 Strict Pruning：当 P-Node 被删除时，
        同时删除所有指向/来自该节点的边。
        
        Args:
            node_id: 要删除的节点 ID
        
        Returns:
            删除的边数量
        """
        if not self.graph.has_node(node_id):
            return 0
        
        # 统计删除的边数量
        edges_removed = 0
        
        # 获取所有相关边（入边和出边）
        in_edges = list(self.graph.in_edges(node_id))
        out_edges = list(self.graph.out_edges(node_id))
        
        edges_removed = len(in_edges) + len(out_edges)
        
        # 删除节点（networkx 会自动删除关联边）
        self.graph.remove_node(node_id)
        
        return edges_removed
    
    def get_k_nodes_connected_to_p(self, p_id: str) -> List[str]:
        """
        获取与指定 P-Node 连接的所有 K-Node IDs
        
        用于判断 P-Node 删除后哪些 K-Node 会变成孤立节点。
        
        Args:
            p_id: P-Node ID
        
        Returns:
            K-Node ID 列表
        """
        k_ids = []
        
        for source, target, data in self.graph.out_edges(p_id, data=True):
            target_data = self.graph.nodes.get(target, {})
            if target_data.get("type") == "K":
                k_ids.append(target)
        
        return k_ids
    
    def get_p_nodes_connected_to_k(self, k_id: str) -> List[str]:
        """
        获取与指定 K-Node 连接的所有 P-Node IDs
        
        用于判断 K-Node 是否孤立（无 P-Node 连接）。
        
        Args:
            k_id: K-Node ID
        
        Returns:
            P-Node ID 列表
        """
        p_ids = []
        
        for source, target, data in self.graph.in_edges(k_id, data=True):
            source_data = self.graph.nodes.get(source, {})
            if source_data.get("type") == "P":
                p_ids.append(source)
        
        return p_ids
    
    def is_k_node_orphan(self, k_id: str) -> bool:
        """
        检查 K-Node 是否孤立（无 P-Node 连接）
        
        Args:
            k_id: K-Node ID
        
        Returns:
            是否孤立
        """
        return len(self.get_p_nodes_connected_to_k(k_id)) == 0
    
    def get_k_node_importance(self, k_id: str) -> str:
        """
        获取 K-Node 的重要性等级
        
        Args:
            k_id: K-Node ID
        
        Returns:
            重要性等级 ("Pathognomonic" / "Essential" / "Strong" / "Common" / "Weak")
        """
        if not self.graph.has_node(k_id):
            return "Common"
        
        return self.graph.nodes[k_id].get("importance", "Common")
    
    def batch_remove_nodes(self, node_ids: List[str]) -> int:
        """
        批量删除节点
        
        Args:
            node_ids: 要删除的节点 ID 列表
        
        Returns:
            总共删除的边数量
        """
        total_edges = 0
        
        for node_id in node_ids:
            edges = self.cascade_remove_node(node_id)
            total_edges += edges
        
        return total_edges