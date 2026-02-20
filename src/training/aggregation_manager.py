"""
Aggregation Manager
===================

管理 Golden Graph 的增量聚合和保存。

核心功能：
1. 记录成功案例并聚合到对应 Pathology 的 CanonicalGraph
2. 按步长（aggregation_step）触发保存
3. 训练结束时保存所有剩余图谱

设计原则：
- Self-Evolution: 训练过程中动态壮大图谱
- 防数据丢失: 定期保存，防止程序中断导致数据丢失
- 去重聚合: 相同内容的节点/边增加计数而不重复
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from src.graph.canonical_graph import CanonicalGraph


class AggregationManager:
    """
    增量聚合管理器
    
    管理多个 CanonicalGraph 的构建、聚合和保存。
    
    Attributes:
        aggregation_step: 每 N 个成功案例触发一次保存
        output_dir: Golden Graph 输出目录
        graphs: 各 Pathology 的 CanonicalGraph 缓存
        success_counters: 各 Pathology 的成功计数器
    """
    
    def __init__(
        self,
        aggregation_step: int = 5,
        output_dir: str = "golden_graphs"
    ):
        """
        初始化聚合管理器
        
        Args:
            aggregation_step: 触发保存的步长
            output_dir: Golden Graph 输出目录
        """
        self.aggregation_step = aggregation_step
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各 Pathology 的 CanonicalGraph 缓存
        self.graphs: Dict[str, CanonicalGraph] = {}
        
        # 各 Pathology 的成功计数器（自上次保存后）
        self.success_counters: Dict[str, int] = defaultdict(int)
        
        # 总统计
        self.total_aggregated = 0
        self.total_saved = 0
    
    def record_success(
        self,
        pathology_id: str,
        pathology_name: str,
        case_graph: Dict[str, Any],
        case_id: str
    ) -> None:
        """
        记录成功案例并聚合到 Golden Graph
        
        Args:
            pathology_id: 疾病 ID (不带 d_ 前缀)
            pathology_name: 疾病名称
            case_graph: 成功案例的 graph_json (MedicalGraph.to_dict() 的输出)
            case_id: 案例 ID
        """
        # 确保 ID 格式正确（去掉 d_ 前缀）
        pathology_id = pathology_id.replace("d_", "")
        
        # 1. 获取或创建 CanonicalGraph
        if pathology_id not in self.graphs:
            # 尝试从磁盘加载现有图谱
            existing_graph = self._try_load_existing(pathology_id)
            if existing_graph:
                self.graphs[pathology_id] = existing_graph
                print(f"[Aggregation] Loaded existing Golden Graph for {pathology_name}")
            else:
                self.graphs[pathology_id] = CanonicalGraph(
                    pathology_id=pathology_id,
                    pathology_name=pathology_name
                )
                print(f"[Aggregation] Created new Golden Graph for {pathology_name}")
        
        # 2. 聚合当前案例
        graph = self.graphs[pathology_id]
        graph.aggregate_from_medical_graph(case_graph, case_id)
        
        self.total_aggregated += 1
        self.success_counters[pathology_id] += 1
        
        print(f"[Aggregation] Aggregated {case_id} to {pathology_name} "
              f"(count: {self.success_counters[pathology_id]}/{self.aggregation_step})")
        
        # 3. 检查是否需要保存
        if self.success_counters[pathology_id] >= self.aggregation_step:
            self._save_graph(pathology_id)
            self.success_counters[pathology_id] = 0
    
    def _try_load_existing(self, pathology_id: str) -> Optional[CanonicalGraph]:
        """
        尝试从磁盘加载现有的 Golden Graph
        
        Args:
            pathology_id: 疾病 ID
        
        Returns:
            CanonicalGraph 或 None
        """
        # 查找匹配的文件
        pattern = f"{pathology_id}_*_GG.json"
        matches = list(self.output_dir.glob(pattern))
        
        if matches:
            try:
                return CanonicalGraph.load(matches[0])
            except Exception as e:
                print(f"[Aggregation] Failed to load existing graph: {e}")
        
        return None
    
    def _save_graph(self, pathology_id: str) -> None:
        """
        保存单个 CanonicalGraph 到磁盘
        
        Args:
            pathology_id: 疾病 ID
        """
        if pathology_id not in self.graphs:
            return
        
        graph = self.graphs[pathology_id]
        
        # 生成文件名：{id}_{sanitized_name}_GG.json
        filename = f"{pathology_id}_{self._sanitize_name(graph.pathology_name)}_GG.json"
        filepath = self.output_dir / filename
        
        graph.save(filepath)
        self.total_saved += 1
        
        stats = graph.get_statistics()
        print(f"[Aggregation] ✓ Saved {filename}")
        print(f"    P-Nodes: {stats['total_p_nodes']}, "
              f"K-Nodes: {stats['total_k_nodes']}, "
              f"Edges: {stats['total_edges']}, "
              f"Cases: {stats['source_cases_count']}")
    
    def finalize(self) -> Dict[str, Any]:
        """
        训练结束时保存所有剩余的 Golden Graphs
        
        Returns:
            统计信息字典
        """
        print(f"\n[Aggregation] Finalizing... ({len(self.graphs)} graphs to save)")
        
        for pathology_id in self.graphs:
            self._save_graph(pathology_id)
        
        # 重置计数器
        self.success_counters = defaultdict(int)
        
        stats = self.get_statistics()
        print(f"[Aggregation] Finalization complete. {stats}")
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取聚合统计信息
        
        Returns:
            统计信息字典
        """
        graph_stats = {}
        for pid, graph in self.graphs.items():
            graph_stats[pid] = graph.get_statistics()
        
        return {
            "total_aggregated": self.total_aggregated,
            "total_saved": self.total_saved,
            "unique_pathologies": len(self.graphs),
            "output_dir": str(self.output_dir),
            "graphs": graph_stats
        }
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        清理文件名中的非法字符
        
        Args:
            name: 原始名称
        
        Returns:
            安全的文件名
        """
        # 替换空格和特殊字符
        safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # 只保留字母数字和下划线
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', safe_name)
        return safe_name


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 简单测试
    manager = AggregationManager(aggregation_step=2, output_dir="golden_graphs_test")
    
    # 模拟 graph_json
    mock_graph = {
        "graph": {
            "nodes": {
                "p_nodes": [
                    {"id": "p_1", "content": "Fever", "status": "Present"},
                    {"id": "p_2", "content": "Cough", "status": "Present"}
                ],
                "k_nodes": [
                    {"id": "k_1", "content": "High temperature indicates infection", 
                     "k_type": "General", "importance": "Essential", "source": "Test"}
                ],
                "d_nodes": [
                    {"id": "d_45", "name": "Pneumonia"}
                ]
            },
            "edges": {
                "p_k_links": [
                    {"source": "p_1", "target": "k_1", "relation": "Match"}
                ],
                "k_d_links": [
                    {"source": "k_1", "target": "d_45", "relation": "Support", "strength": "Essential"}
                ]
            }
        }
    }
    
    # 聚合测试
    manager.record_success("45", "Pneumonia", mock_graph, "test_case_001")
    manager.record_success("45", "Pneumonia", mock_graph, "test_case_002")  # 应该触发保存
    manager.record_success("23", "Bronchitis", mock_graph, "test_case_003")
    
    # 最终保存
    stats = manager.finalize()
    print("\nFinal Statistics:", stats)




















