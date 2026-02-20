"""
Canonical Graph (Golden Graph) Schema
======================================

定义 CanonicalGraph 类，用于管理离线金标准图谱（Offline Golden Graph）。

核心设计理念：
- CanonicalGraph 是以 **Pathology（疾病）** 为中心的图谱
- 每个 Pathology 拥有一个独立的 CanonicalGraph 实例
- 通过聚合多个训练 Case 的图谱，构建该疾病的"Illness Script"（疾病模型）
- 包含该病种下所有出现过的 P-Nodes、K-Nodes 和 Edges 的 **并集 (Union)**

存储格式：
- 每个 Pathology 对应一个独立的 JSON 文件
- 例如：`golden_graphs/Pulmonary_Embolism_GG.json`

与 MedicalGraph 的关系：
- MedicalGraph: 单个 Case 的推理图谱（Online 使用）
- CanonicalGraph: 某个 Pathology 的聚合知识图谱（Offline 构建，Online 查询）

使用场景：
1. Offline 训练：从多个标注 Case 聚合构建 CanonicalGraph
2. Online 推理：根据 Top-5 候选加载对应的 5 个 CanonicalGraph，
   将 Golden Graph 中的知识注入到 MedicalGraph 中
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CanonicalPNode:
    """
    Canonical P-Node: 聚合后的患者特征节点
    
    在 Golden Graph 中，P-Node 代表该疾病常见的患者特征/症状。
    通过统计多个 Case 的 P-Nodes，可以得到该疾病的典型表现。
    
    Attributes:
        content: 症状/体征名称（医学术语，作为唯一标识）
        occurrence_count: 出现次数（在聚合过程中统计）
        typical_status: 典型状态 (Present/Absent)
        example_original_texts: 原始文本示例列表（保留部分用于参考）
    """
    content: str
    occurrence_count: int = 1
    typical_status: str = "Present"
    example_original_texts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "content": self.content,
            "occurrence_count": self.occurrence_count,
            "typical_status": self.typical_status,
            "example_original_texts": self.example_original_texts[:3]  # 只保留前3个示例
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalPNode":
        """从字典反序列化"""
        return cls(
            content=data.get("content", ""),
            occurrence_count=data.get("occurrence_count", 1),
            typical_status=data.get("typical_status", "Present"),
            example_original_texts=data.get("example_original_texts", [])
        )


@dataclass
class CanonicalKNode:
    """
    Canonical K-Node: 聚合后的医学知识节点
    
    在 Golden Graph 中，K-Node 代表该疾病的关键医学知识/诊断要点。
    
    Attributes:
        content: 知识点描述
        k_type: 知识类型 (General/Pivot)
        importance: 重要性等级 (Essential/Pathognomonic/Strong/Weak/Common)
        occurrence_count: 出现次数
        original_sources: 原始来源列表（如 PubMed、OpenTargets 等）
    """
    content: str
    k_type: str = "General"
    importance: str = "Common"
    occurrence_count: int = 1
    original_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "content": self.content,
            "k_type": self.k_type,
            "importance": self.importance,
            "occurrence_count": self.occurrence_count,
            "original_sources": list(set(self.original_sources))[:5]  # 去重，只保留前5个
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalKNode":
        """从字典反序列化"""
        return cls(
            content=data.get("content", ""),
            k_type=data.get("k_type", "General"),
            importance=data.get("importance", "Common"),
            occurrence_count=data.get("occurrence_count", 1),
            original_sources=data.get("original_sources", [])
        )


@dataclass
class CanonicalEdge:
    """
    Canonical Edge: 聚合后的边
    
    记录节点之间的关系及其在多个 Case 中的出现情况。
    
    Attributes:
        source_content: 源节点内容（用于匹配）
        target_content: 目标节点内容（用于匹配）
        edge_type: 边类型 ("P-K" 或 "K-D")
        relation: 关系类型 (Match/Conflict/Void 或 Support/Rule_Out)
        strength: 边强度（仅 K-D 边）
        occurrence_count: 出现次数
    """
    source_content: str
    target_content: str
    edge_type: str  # "P-K" or "K-D"
    relation: str
    strength: Optional[str] = None
    occurrence_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        result = {
            "source_content": self.source_content,
            "target_content": self.target_content,
            "edge_type": self.edge_type,
            "relation": self.relation,
            "occurrence_count": self.occurrence_count
        }
        if self.strength:
            result["strength"] = self.strength
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalEdge":
        """从字典反序列化"""
        return cls(
            source_content=data.get("source_content", ""),
            target_content=data.get("target_content", ""),
            edge_type=data.get("edge_type", "P-K"),
            relation=data.get("relation", ""),
            strength=data.get("strength"),
            occurrence_count=data.get("occurrence_count", 1)
        )


class CanonicalGraph:
    """
    Canonical Graph (Golden Graph) 类
    
    以 Pathology 为中心的聚合图谱，用于存储和管理某个疾病的
    金标准知识图谱。
    
    核心功能：
    1. 从多个 MedicalGraph 聚合构建
    2. 序列化/反序列化到 JSON 文件
    3. 提供查询接口供 Online 推理使用
    
    Attributes:
        pathology_id: 疾病 ID (01-49)
        pathology_name: 疾病名称
        p_nodes: 聚合后的 P-Nodes（以 content 为 key）
        k_nodes: 聚合后的 K-Nodes（以 content 为 key）
        edges: 聚合后的边列表
        source_cases: 用于构建此图谱的 Case ID 列表
        version: 图谱版本号
    """
    
    def __init__(
        self,
        pathology_id: str,
        pathology_name: str
    ):
        """
        初始化 CanonicalGraph
        
        Args:
            pathology_id: 疾病 ID (如 "03" 代表 Pulmonary embolism)
            pathology_name: 疾病名称 (如 "Pulmonary embolism")
        """
        self.pathology_id = pathology_id
        self.pathology_name = pathology_name
        
        # 节点存储（以 content 为 key，便于去重和聚合）
        self.p_nodes: Dict[str, CanonicalPNode] = {}
        self.k_nodes: Dict[str, CanonicalKNode] = {}
        
        # 边存储（以 (source_content, target_content, relation) 为 key）
        self._edges: Dict[Tuple[str, str, str], CanonicalEdge] = {}
        
        # 元数据
        self.source_cases: List[str] = []
        self.version: str = "1.0"
    
    # ==================== 聚合操作 ====================
    
    def add_p_node(
        self,
        content: str,
        status: str = "Present",
        original_text: str = ""
    ) -> None:
        """
        添加或聚合 P-Node
        
        如果 content 已存在，则增加 occurrence_count。
        
        Args:
            content: 症状/体征名称
            status: 状态 (Present/Absent)
            original_text: 原始文本（可选，用于保留示例）
        """
        content_key = content.lower().strip()
        
        if content_key in self.p_nodes:
            # 已存在，增加计数
            existing = self.p_nodes[content_key]
            existing.occurrence_count += 1
            if original_text and original_text not in existing.example_original_texts:
                existing.example_original_texts.append(original_text)
        else:
            # 新建
            self.p_nodes[content_key] = CanonicalPNode(
                content=content,
                occurrence_count=1,
                typical_status=status,
                example_original_texts=[original_text] if original_text else []
            )
    
    def add_k_node(
        self,
        content: str,
        k_type: str = "General",
        importance: str = "Common",
        source: str = ""
    ) -> None:
        """
        添加或聚合 K-Node
        
        如果 content 已存在，则增加 occurrence_count，
        并更新 importance（取更高优先级）。
        
        Args:
            content: 知识点描述
            k_type: 知识类型 (General/Pivot)
            importance: 重要性等级
            source: 原始来源
        """
        content_key = content.lower().strip()
        
        # 重要性优先级
        importance_order = {
            "Pathognomonic": 4,
            "Essential": 3,
            "Strong": 2,
            "Common": 1,
            "Weak": 0
        }
        
        if content_key in self.k_nodes:
            # 已存在，增加计数并可能更新 importance
            existing = self.k_nodes[content_key]
            existing.occurrence_count += 1
            
            # 更新为更高优先级的 importance
            if importance_order.get(importance, 0) > importance_order.get(existing.importance, 0):
                existing.importance = importance
            
            # 添加来源
            if source and source not in existing.original_sources:
                existing.original_sources.append(source)
        else:
            # 新建
            self.k_nodes[content_key] = CanonicalKNode(
                content=content,
                k_type=k_type,
                importance=importance,
                occurrence_count=1,
                original_sources=[source] if source else []
            )
    
    def add_edge(
        self,
        source_content: str,
        target_content: str,
        edge_type: str,
        relation: str,
        strength: Optional[str] = None
    ) -> None:
        """
        添加或聚合边
        
        Args:
            source_content: 源节点内容
            target_content: 目标节点内容
            edge_type: 边类型 ("P-K" 或 "K-D")
            relation: 关系类型
            strength: 边强度（仅 K-D 边）
        """
        edge_key = (
            source_content.lower().strip(),
            target_content.lower().strip(),
            relation
        )
        
        if edge_key in self._edges:
            # 已存在，增加计数
            self._edges[edge_key].occurrence_count += 1
        else:
            # 新建
            self._edges[edge_key] = CanonicalEdge(
                source_content=source_content,
                target_content=target_content,
                edge_type=edge_type,
                relation=relation,
                strength=strength,
                occurrence_count=1
            )
    
    def add_source_case(self, case_id: str) -> None:
        """
        记录用于构建此图谱的 Case ID
        
        Args:
            case_id: Case 标识符
        """
        if case_id not in self.source_cases:
            self.source_cases.append(case_id)
    
    # ==================== 从 MedicalGraph 聚合 ====================
    
    def aggregate_from_medical_graph(
        self,
        graph_dict: Dict[str, Any],
        case_id: str = ""
    ) -> None:
        """
        从 MedicalGraph 的 JSON 表示聚合知识
        
        这是构建 Golden Graph 的核心方法。遍历 MedicalGraph 中的
        所有节点和边，将其聚合到 CanonicalGraph 中。
        
        Args:
            graph_dict: MedicalGraph.to_dict() 的输出
            case_id: Case 标识符
        """
        if case_id:
            self.add_source_case(case_id)
        
        graph_data = graph_dict.get("graph", {})
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", {})
        
        # 聚合 P-Nodes
        for p_node in nodes.get("p_nodes", []):
            self.add_p_node(
                content=p_node.get("content", ""),
                status=p_node.get("status", "Present"),
                original_text=p_node.get("original_text", "")
            )
        
        # 聚合 K-Nodes
        for k_node in nodes.get("k_nodes", []):
            self.add_k_node(
                content=k_node.get("content", ""),
                k_type=k_node.get("k_type", "General"),
                importance=k_node.get("importance", "Common"),
                source=k_node.get("source", "")
            )
        
        # 构建 ID -> Content 映射（用于边的解析）
        id_to_content: Dict[str, str] = {}
        
        for p_node in nodes.get("p_nodes", []):
            node_id = p_node.get("id", "")
            if node_id:
                id_to_content[node_id] = p_node.get("content", "")
        
        for k_node in nodes.get("k_nodes", []):
            node_id = k_node.get("id", "")
            if node_id:
                id_to_content[node_id] = k_node.get("content", "")
        
        for d_node in nodes.get("d_nodes", []):
            node_id = d_node.get("id", "")
            if node_id:
                id_to_content[node_id] = d_node.get("name", "")
        
        # 聚合 P-K 边
        for edge in edges.get("p_k_links", []):
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")
            relation = edge.get("relation", "")
            
            source_content = id_to_content.get(source_id, source_id)
            target_content = id_to_content.get(target_id, target_id)
            
            if source_content and target_content:
                self.add_edge(
                    source_content=source_content,
                    target_content=target_content,
                    edge_type="P-K",
                    relation=relation
                )
        
        # 聚合 K-D 边（只保留与当前 Pathology 相关的边）
        for edge in edges.get("k_d_links", []):
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")
            relation = edge.get("relation", "")
            strength = edge.get("strength", "Weak")
            
            source_content = id_to_content.get(source_id, source_id)
            target_content = id_to_content.get(target_id, target_id)
            
            # 只保留指向当前 Pathology 的 K-D 边
            # target_id 格式为 "d_XX"，其中 XX 是疾病 ID
            target_disease_id = target_id.replace("d_", "") if target_id.startswith("d_") else ""
            
            if target_disease_id == self.pathology_id:
                if source_content and target_content:
                    self.add_edge(
                        source_content=source_content,
                        target_content=target_content,
                        edge_type="K-D",
                        relation=relation,
                        strength=strength
                    )
    
    # ==================== 查询接口 ====================
    
    def get_essential_k_nodes(self) -> List[CanonicalKNode]:
        """
        获取所有 Essential 或 Pathognomonic 级别的 K-Nodes
        
        这些是该疾病的核心诊断要点。
        
        Returns:
            高重要性 K-Nodes 列表
        """
        return [
            node for node in self.k_nodes.values()
            if node.importance in ["Essential", "Pathognomonic"]
        ]
    
    def get_pivot_k_nodes(self) -> List[CanonicalKNode]:
        """
        获取所有 Pivot 类型的 K-Nodes
        
        这些是用于鉴别诊断的关键特征。
        
        Returns:
            Pivot K-Nodes 列表
        """
        return [
            node for node in self.k_nodes.values()
            if node.k_type == "Pivot"
        ]
    
    def get_common_p_nodes(self, min_occurrence: int = 2) -> List[CanonicalPNode]:
        """
        获取出现次数超过阈值的 P-Nodes
        
        这些是该疾病的典型表现。
        
        Args:
            min_occurrence: 最小出现次数
        
        Returns:
            常见 P-Nodes 列表
        """
        return [
            node for node in self.p_nodes.values()
            if node.occurrence_count >= min_occurrence
        ]
    
    def get_edges_by_type(self, edge_type: str) -> List[CanonicalEdge]:
        """
        按类型获取边
        
        Args:
            edge_type: 边类型 ("P-K" 或 "K-D")
        
        Returns:
            指定类型的边列表
        """
        return [
            edge for edge in self._edges.values()
            if edge.edge_type == edge_type
        ]
    
    @property
    def edges(self) -> List[CanonicalEdge]:
        """获取所有边"""
        return list(self._edges.values())
    
    # ==================== 统计信息 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取图谱统计信息
        
        Returns:
            包含各种统计指标的字典
        """
        return {
            "pathology_id": self.pathology_id,
            "pathology_name": self.pathology_name,
            "total_p_nodes": len(self.p_nodes),
            "total_k_nodes": len(self.k_nodes),
            "total_edges": len(self._edges),
            "source_cases_count": len(self.source_cases),
            "essential_k_nodes_count": len(self.get_essential_k_nodes()),
            "pivot_k_nodes_count": len(self.get_pivot_k_nodes())
        }
    
    # ==================== 序列化 ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为字典
        
        Returns:
            可序列化为 JSON 的字典
        """
        return {
            "metadata": {
                "pathology_id": self.pathology_id,
                "pathology_name": self.pathology_name,
                "version": self.version,
                "source_cases_count": len(self.source_cases),
                "source_cases": self.source_cases
            },
            "statistics": self.get_statistics(),
            "graph": {
                "p_nodes": [node.to_dict() for node in self.p_nodes.values()],
                "k_nodes": [node.to_dict() for node in self.k_nodes.values()],
                "edges": [edge.to_dict() for edge in self._edges.values()]
            }
        }
    
    def save(self, filepath: Path) -> None:
        """
        保存到 JSON 文件
        
        Args:
            filepath: 目标文件路径
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"[CanonicalGraph] Saved {self.pathology_name} to {filepath}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalGraph":
        """
        从字典反序列化
        
        Args:
            data: JSON 解析后的字典
        
        Returns:
            CanonicalGraph 实例
        """
        metadata = data.get("metadata", {})
        
        graph = cls(
            pathology_id=metadata.get("pathology_id", ""),
            pathology_name=metadata.get("pathology_name", "")
        )
        graph.version = metadata.get("version", "1.0")
        graph.source_cases = metadata.get("source_cases", [])
        
        # 加载节点
        graph_data = data.get("graph", {})
        
        for p_data in graph_data.get("p_nodes", []):
            node = CanonicalPNode.from_dict(p_data)
            graph.p_nodes[node.content.lower().strip()] = node
        
        for k_data in graph_data.get("k_nodes", []):
            node = CanonicalKNode.from_dict(k_data)
            graph.k_nodes[node.content.lower().strip()] = node
        
        for e_data in graph_data.get("edges", []):
            edge = CanonicalEdge.from_dict(e_data)
            edge_key = (
                edge.source_content.lower().strip(),
                edge.target_content.lower().strip(),
                edge.relation
            )
            graph._edges[edge_key] = edge
        
        return graph
    
    @classmethod
    def load(cls, filepath: Path) -> "CanonicalGraph":
        """
        从 JSON 文件加载
        
        Args:
            filepath: 源文件路径
        
        Returns:
            CanonicalGraph 实例
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    # ==================== 辅助方法 ====================
    
    @staticmethod
    def get_canonical_filename(pathology_name: str) -> str:
        """
        根据疾病名称生成规范的文件名
        
        Args:
            pathology_name: 疾病名称
        
        Returns:
            规范化的文件名（不含扩展名）
        
        Example:
            "Pulmonary embolism" -> "Pulmonary_embolism_GG"
        """
        # 替换空格和特殊字符
        safe_name = pathology_name.replace(" ", "_").replace("/", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        return f"{safe_name}_GG"
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"CanonicalGraph("
            f"id={self.pathology_id}, "
            f"name='{self.pathology_name}', "
            f"p_nodes={len(self.p_nodes)}, "
            f"k_nodes={len(self.k_nodes)}, "
            f"edges={len(self._edges)}, "
            f"cases={len(self.source_cases)}"
            f")"
        )




















