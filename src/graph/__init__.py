# Graph package
"""
Graph Module
============

包含医学诊断推理图谱的核心数据结构：

- MedicalGraph: 单个 Case 的推理图谱（Online 使用）
- CanonicalGraph: 某个 Pathology 的聚合知识图谱（Golden Graph，Offline 构建）

类型定义：
- SourceType: 节点来源类型 ("GoldenGraph" | "LiveSearch")
- EdgeProvenance: 边的来源 ("GoldenGraph" | "LiveInference")
"""

from .schema import MedicalGraph, SourceType, EdgeProvenance
from .canonical_graph import CanonicalGraph, CanonicalPNode, CanonicalKNode, CanonicalEdge

__all__ = [
    # 核心类
    "MedicalGraph",
    "CanonicalGraph",
    # 数据类
    "CanonicalPNode",
    "CanonicalKNode", 
    "CanonicalEdge",
    # 类型定义
    "SourceType",
    "EdgeProvenance",
]






