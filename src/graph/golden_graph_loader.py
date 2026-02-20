"""
Golden Graph Loader
====================

根据 Candidate ID 加载预构建的 Golden Graph (CanonicalGraph)。

功能：
1. 根据疾病 ID 从 golden_graphs/ 目录加载 CanonicalGraph
2. 支持批量加载 Top-5 候选疾病的 Golden Graphs
3. 线程安全的缓存机制
4. 优雅降级：文件不存在时返回 None，不抛出异常
5. **优先加载精炼版 (Refined)**: 自动检查 golden_graphs_refined/ 目录

加载优先级：
1. golden_graphs_refined/{id}_*_Refined_GG.json (精炼版)
2. golden_graphs/{id}_*_GG.json (原始版)
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .canonical_graph import CanonicalGraph


class GoldenGraphLoader:
    """
    Golden Graph 加载器
    
    线程安全的 CanonicalGraph 加载和缓存管理器。
    优先加载精炼版 (Refined) Golden Graphs，如果不存在则回退到原始版。
    
    Attributes:
        golden_graph_dir: 原始 Golden Graph 存储目录
        refined_graph_dir: 精炼版 Golden Graph 存储目录
        _cache: 已加载的 CanonicalGraph 缓存
        _lock: 线程锁，保护缓存写入
    """
    
    def __init__(
        self, 
        golden_graph_dir: str = "golden_graphs",
        refined_graph_dir: str = "golden_graphs_refined"
    ):
        """
        初始化 Golden Graph 加载器
        
        Args:
            golden_graph_dir: 原始 Golden Graph 文件目录路径
            refined_graph_dir: 精炼版 Golden Graph 文件目录路径
        """
        self.golden_graph_dir = Path(golden_graph_dir)
        self.refined_graph_dir = Path(refined_graph_dir)
        self._cache: Dict[str, Optional[CanonicalGraph]] = {}
        self._lock = threading.Lock()
        
        # 预扫描可用的 Golden Graph 文件（包括精炼版和原始版）
        self._available_ids = self._scan_available_graphs()
        
        # 统计精炼版和原始版数量
        refined_count = sum(1 for info in self._available_ids.values() if info[1])
        raw_count = sum(1 for info in self._available_ids.values() if not info[1])
        
        print(f"[GoldenGraphLoader] Initialized with {len(self._available_ids)} available graphs")
        print(f"  → Refined: {refined_count}, Raw: {raw_count}")
    
    def _scan_available_graphs(self) -> Dict[str, Tuple[Path, bool]]:
        """
        扫描目录中可用的 Golden Graph 文件。
        
        优先级：精炼版 > 原始版
        
        Returns:
            {disease_id: (file_path, is_refined)} 映射
            - is_refined: True 表示精炼版，False 表示原始版
        """
        available = {}
        
        # 1. 首先扫描原始版目录
        if self.golden_graph_dir.exists():
            for filepath in self.golden_graph_dir.glob("*_GG.json"):
                filename = filepath.stem  # 去掉 .json
                parts = filename.split("_")
                
                if len(parts) >= 2:
                    disease_id = parts[0]
                    if disease_id.isdigit():
                        # 标记为原始版 (is_refined=False)
                        available[disease_id] = (filepath, False)
        else:
            print(f"[GoldenGraphLoader] Warning: Raw directory not found: {self.golden_graph_dir}")
        
        # 2. 然后扫描精炼版目录（会覆盖原始版）
        if self.refined_graph_dir.exists():
            for filepath in self.refined_graph_dir.glob("*_Refined_GG.json"):
                filename = filepath.stem  # 去掉 .json
                parts = filename.split("_")
                
                if len(parts) >= 2:
                    disease_id = parts[0]
                    if disease_id.isdigit():
                        # 标记为精炼版 (is_refined=True)，覆盖原始版
                        available[disease_id] = (filepath, True)
        else:
            print(f"[GoldenGraphLoader] Info: Refined directory not found: {self.refined_graph_dir}")
            print(f"[GoldenGraphLoader] Will load raw Golden Graphs only.")
        
        return available
    
    def load_for_candidates(
        self, 
        candidate_ids: List[str]
    ) -> Dict[str, Optional[CanonicalGraph]]:
        """
        为 Top-5 Candidates 批量加载 Golden Graphs。
        
        自动优先加载精炼版，如果不存在则回退到原始版。
        
        Args:
            candidate_ids: 诊断 ID 列表 (例如 ["45", "23", "03", ...])
        
        Returns:
            {disease_id: CanonicalGraph or None} 字典
            - 如果对应的 Golden Graph 存在，返回 CanonicalGraph 实例
            - 如果不存在，返回 None (不抛出异常)
        """
        result = {}
        
        for disease_id in candidate_ids:
            # 标准化 ID (去掉可能的 d_ 前缀)
            clean_id = disease_id.replace("d_", "")
            
            # 尝试从缓存获取
            with self._lock:
                if clean_id in self._cache:
                    result[disease_id] = self._cache[clean_id]
                    continue
            
            # 加载并缓存
            graph = self._load_single(clean_id)
            
            with self._lock:
                self._cache[clean_id] = graph
            
            result[disease_id] = graph
        
        # 统计加载结果
        loaded = sum(1 for g in result.values() if g is not None)
        print(f"[GoldenGraphLoader] Loaded {loaded}/{len(candidate_ids)} Golden Graphs")
        
        return result
    
    def _load_single(self, disease_id: str) -> Optional[CanonicalGraph]:
        """
        加载单个 Golden Graph。
        
        优先加载精炼版，如果不存在则加载原始版。
        
        Args:
            disease_id: 疾病 ID (不含 d_ 前缀)
        
        Returns:
            CanonicalGraph 实例，或 None (如果不存在)
        """
        # 检查预扫描的可用列表
        if disease_id not in self._available_ids:
            print(f"[GoldenGraphLoader] No Golden Graph found for disease {disease_id}")
            return None
        
        filepath, is_refined = self._available_ids[disease_id]
        version_tag = "Refined" if is_refined else "Raw"
        
        try:
            graph = CanonicalGraph.load(filepath)
            
            # 明确输出加载的是哪个版本
            print(f"[GoldenGraphLoader] ✓ Loaded [{version_tag}] {filepath.name} "
                  f"(P:{len(graph.p_nodes)}, K:{len(graph.k_nodes)})")
            
            return graph
            
        except Exception as e:
            print(f"[GoldenGraphLoader] ✗ Error loading {filepath}: {e}")
            return None
    
    def get_available_diseases(self) -> List[str]:
        """
        获取所有可用的疾病 ID 列表
        
        Returns:
            可用的疾病 ID 列表
        """
        return list(self._available_ids.keys())
    
    def is_available(self, disease_id: str) -> bool:
        """
        检查指定疾病的 Golden Graph 是否可用
        
        Args:
            disease_id: 疾病 ID
        
        Returns:
            是否可用
        """
        clean_id = disease_id.replace("d_", "")
        return clean_id in self._available_ids
    
    def is_refined(self, disease_id: str) -> bool:
        """
        检查指定疾病的 Golden Graph 是否为精炼版
        
        Args:
            disease_id: 疾病 ID
        
        Returns:
            是否为精炼版（如果不存在返回 False）
        """
        clean_id = disease_id.replace("d_", "")
        if clean_id not in self._available_ids:
            return False
        return self._available_ids[clean_id][1]  # 返回 is_refined 标志
    
    def get_graph_info(self, disease_id: str) -> Optional[Dict[str, any]]:
        """
        获取指定疾病 Golden Graph 的元信息
        
        Args:
            disease_id: 疾病 ID
        
        Returns:
            包含路径和版本信息的字典，或 None
        """
        clean_id = disease_id.replace("d_", "")
        if clean_id not in self._available_ids:
            return None
        
        filepath, is_refined = self._available_ids[clean_id]
        return {
            "disease_id": clean_id,
            "filepath": str(filepath),
            "is_refined": is_refined,
            "version": "Refined" if is_refined else "Raw"
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
        print("[GoldenGraphLoader] Cache cleared")
    
    def refresh(self) -> None:
        """
        刷新可用图谱列表并清空缓存。
        
        在新的精炼版生成后调用此方法以使加载器感知新文件。
        """
        with self._lock:
            self._cache.clear()
        
        self._available_ids = self._scan_available_graphs()
        
        refined_count = sum(1 for info in self._available_ids.values() if info[1])
        raw_count = sum(1 for info in self._available_ids.values() if not info[1])
        
        print(f"[GoldenGraphLoader] Refreshed: {len(self._available_ids)} graphs available")
        print(f"  → Refined: {refined_count}, Raw: {raw_count}")
    
    def get_cache_statistics(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_cached = len(self._cache)
            valid_cached = sum(1 for g in self._cache.values() if g is not None)
        
        refined_available = sum(1 for info in self._available_ids.values() if info[1])
        raw_available = sum(1 for info in self._available_ids.values() if not info[1])
        
        return {
            "total_available": len(self._available_ids),
            "refined_available": refined_available,
            "raw_available": raw_available,
            "total_cached": total_cached,
            "valid_cached": valid_cached
        }


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 简单测试
    loader = GoldenGraphLoader(
        golden_graph_dir="golden_graphs",
        refined_graph_dir="golden_graphs_refined"
    )
    
    print("\nAvailable diseases:", loader.get_available_diseases()[:5], "...")
    
    # 检查是否为精炼版
    for d_id in ["01", "03", "45"]:
        info = loader.get_graph_info(d_id)
        if info:
            print(f"  {d_id}: {info['version']} - {info['filepath']}")
        else:
            print(f"  {d_id}: Not available")
    
    # 测试加载
    test_ids = ["01", "03", "99"]  # 99 应该不存在
    result = loader.load_for_candidates(test_ids)
    
    for d_id, graph in result.items():
        if graph:
            print(f"  {d_id}: {graph}")
        else:
            print(f"  {d_id}: None (not available)")
    
    print("\nCache stats:", loader.get_cache_statistics())
