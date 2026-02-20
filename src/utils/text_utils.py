"""
Text Utilities for Semantic Matching
=====================================

提供基于 Embedding 的语义匹配功能，用于 Phase 2 的 P-Node 对齐。

核心功能：
1. Embedding 获取（复用 EmbeddingClient）
2. 余弦相似度计算
3. 批量匹配矩阵生成
4. Fuzzy Match Fallback（当 Embedding API 失败时）

设计原则：
- 复用 src/memory/embedding_client.py 的逻辑
- 在单个 Case 生命周期内使用简单的 dict 缓存
- Embedding API 失败时自动降级到 Token Set Ratio
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from functools import lru_cache
from dataclasses import dataclass

# Fuzzy Match 用于 Fallback
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("[TextUtils] Warning: rapidfuzz not installed, fuzzy match fallback disabled")


@dataclass
class MatchResult:
    """P-Node 匹配结果"""
    gg_content: str          # GG 中的 P-Node 内容
    patient_content: str     # 匹配到的患者 P-Node 内容
    patient_status: str      # 患者 P-Node 状态 (Present/Absent)
    similarity: float        # 相似度分数
    matched: bool            # 是否匹配成功
    match_method: str        # 匹配方法 ("embedding" / "fuzzy" / "none")


class SemanticMatcher:
    """
    语义匹配器
    
    基于 Embedding 的 P-Node 语义对齐工具。
    支持批量处理和 Fallback 机制。
    
    Attributes:
        embedding_client: EmbeddingClient 实例
        similarity_threshold: 相似度阈值
        fuzzy_threshold: Fuzzy Match 阈值 (Fallback 用)
        _cache: 文本到 Embedding 的缓存
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.90,
        fuzzy_threshold: float = 0.80,
        use_cache: bool = True
    ):
        """
        初始化语义匹配器
        
        Args:
            similarity_threshold: Embedding 相似度阈值 (严格模式默认 0.90)
            fuzzy_threshold: Fuzzy Match 阈值 (Fallback 默认 0.80)
            use_cache: 是否使用缓存
        """
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.use_cache = use_cache
        
        # 初始化 Embedding 客户端
        self._embedding_client = None
        self._embedding_available = False
        self._cache: Dict[str, np.ndarray] = {}
        
        # 尝试初始化 Embedding 客户端
        self._init_embedding_client()
    
    def _init_embedding_client(self) -> None:
        """初始化 Embedding 客户端"""
        try:
            from src.memory.embedding_client import EmbeddingClient
            
            # 从环境变量获取 API Key
            api_key = os.getenv("YUNWU_API_KEY") or os.getenv("OPENAI_API_KEY")
            
            if api_key:
                self._embedding_client = EmbeddingClient(
                    base_url="https://yunwu.ai/v1",
                    api_key=api_key,
                    model="text-embedding-3-small",
                    dimension=1536
                )
                self._embedding_available = True
                print("[SemanticMatcher] Embedding client initialized successfully")
            else:
                print("[SemanticMatcher] Warning: No API key found, embedding disabled")
                
        except Exception as e:
            print(f"[SemanticMatcher] Warning: Failed to init embedding client: {e}")
            self._embedding_available = False
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的 Embedding 向量
        
        Args:
            text: 输入文本
        
        Returns:
            numpy 数组，或 None（如果失败）
        """
        if not text or not text.strip():
            return None
        
        text_key = text.lower().strip()
        
        # 检查缓存
        if self.use_cache and text_key in self._cache:
            return self._cache[text_key]
        
        # 调用 Embedding API
        if not self._embedding_available or self._embedding_client is None:
            return None
        
        try:
            embedding = self._embedding_client.embed_text(text)
            
            # 存入缓存
            if self.use_cache:
                self._cache[text_key] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"[SemanticMatcher] Embedding error for '{text[:50]}...': {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        批量获取 Embedding
        
        Args:
            texts: 文本列表
        
        Returns:
            {text_key: embedding} 字典
        """
        results = {}
        texts_to_fetch = []
        text_keys = []
        
        for text in texts:
            if not text or not text.strip():
                continue
            
            text_key = text.lower().strip()
            
            # 检查缓存
            if self.use_cache and text_key in self._cache:
                results[text_key] = self._cache[text_key]
            else:
                texts_to_fetch.append(text)
                text_keys.append(text_key)
        
        # 批量获取未缓存的
        if texts_to_fetch and self._embedding_available and self._embedding_client:
            try:
                embeddings = self._embedding_client.embed_texts(texts_to_fetch)
                
                for i, (text_key, embedding) in enumerate(zip(text_keys, embeddings)):
                    results[text_key] = embedding
                    if self.use_cache:
                        self._cache[text_key] = embedding
                        
            except Exception as e:
                print(f"[SemanticMatcher] Batch embedding error: {e}")
        
        return results
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量 1
            vec2: 向量 2
        
        Returns:
            相似度值 [0, 1]
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def fuzzy_match_score(self, text1: str, text2: str) -> float:
        """
        计算 Fuzzy Match 分数 (Token Set Ratio)
        
        Args:
            text1: 文本 1
            text2: 文本 2
        
        Returns:
            相似度值 [0, 1]
        """
        if not RAPIDFUZZ_AVAILABLE:
            # 简单的字符串包含匹配
            t1 = text1.lower().strip()
            t2 = text2.lower().strip()
            
            if t1 == t2:
                return 1.0
            elif t1 in t2 or t2 in t1:
                return 0.85
            else:
                return 0.0
        
        # 使用 rapidfuzz 的 Token Set Ratio
        score = fuzz.token_set_ratio(text1, text2) / 100.0
        return score
    
    def match_p_nodes(
        self,
        gg_p_nodes: List[Dict[str, Any]],
        patient_p_nodes: List[Dict[str, Any]]
    ) -> Tuple[List[MatchResult], bool]:
        """
        执行 P-Node 语义匹配
        
        这是核心匹配方法，用于 Phase 2 的 GG P-Node 与患者 P-Node 对齐。
        
        匹配流程：
        1. 尝试使用 Embedding 计算相似度矩阵
        2. 如果 Embedding 失败，降级到 Fuzzy Match
        3. 对每个 GG P-Node，找到最佳匹配
        
        Args:
            gg_p_nodes: Golden Graph 的 P-Nodes 列表
                每个元素需包含 "content" 字段
            patient_p_nodes: 患者的 P-Nodes 列表
                每个元素需包含 "content" 和 "status" 字段
        
        Returns:
            (匹配结果列表, 是否使用了 Embedding)
        """
        results = []
        used_embedding = False
        
        if not gg_p_nodes or not patient_p_nodes:
            return results, used_embedding
        
        # 提取内容
        gg_contents = [p.get("content", "") for p in gg_p_nodes]
        patient_contents = [p.get("content", "") for p in patient_p_nodes]
        
        # 构建患者 P-Node 映射
        patient_map = {
            p.get("content", "").lower().strip(): p
            for p in patient_p_nodes
        }
        
        # 尝试 Embedding 匹配
        similarity_matrix = None
        
        if self._embedding_available:
            try:
                # 批量获取 Embedding
                all_texts = gg_contents + patient_contents
                embeddings = self.get_embeddings_batch(all_texts)
                
                if embeddings:
                    # 构建相似度矩阵
                    similarity_matrix = np.zeros((len(gg_contents), len(patient_contents)))
                    
                    for i, gg_content in enumerate(gg_contents):
                        gg_key = gg_content.lower().strip()
                        gg_emb = embeddings.get(gg_key)
                        
                        if gg_emb is None:
                            continue
                        
                        for j, patient_content in enumerate(patient_contents):
                            patient_key = patient_content.lower().strip()
                            patient_emb = embeddings.get(patient_key)
                            
                            if patient_emb is not None:
                                similarity_matrix[i, j] = self.cosine_similarity(gg_emb, patient_emb)
                    
                    used_embedding = True
                    print(f"[SemanticMatcher] Built similarity matrix: {similarity_matrix.shape}")
                    
            except Exception as e:
                print(f"[Warning] Embedding API failed, falling back to fuzzy match: {e}")
                similarity_matrix = None
        
        # 如果 Embedding 失败，使用 Fuzzy Match
        if similarity_matrix is None:
            print("[SemanticMatcher] Using fuzzy match fallback")
            similarity_matrix = np.zeros((len(gg_contents), len(patient_contents)))
            
            for i, gg_content in enumerate(gg_contents):
                for j, patient_content in enumerate(patient_contents):
                    similarity_matrix[i, j] = self.fuzzy_match_score(gg_content, patient_content)
        
        # 对每个 GG P-Node 找最佳匹配
        threshold = self.similarity_threshold if used_embedding else self.fuzzy_threshold
        
        for i, gg_p in enumerate(gg_p_nodes):
            gg_content = gg_p.get("content", "")
            
            if not gg_content:
                continue
            
            # 找到最大相似度
            max_sim = 0.0
            best_j = -1
            
            for j in range(len(patient_contents)):
                if similarity_matrix[i, j] > max_sim:
                    max_sim = similarity_matrix[i, j]
                    best_j = j
            
            # 判断是否匹配
            matched = max_sim >= threshold
            
            if matched and best_j >= 0:
                patient_p = patient_p_nodes[best_j]
                result = MatchResult(
                    gg_content=gg_content,
                    patient_content=patient_p.get("content", ""),
                    patient_status=patient_p.get("status", "Present"),
                    similarity=max_sim,
                    matched=True,
                    match_method="embedding" if used_embedding else "fuzzy"
                )
                print(f"[Phase 2] P-Node Matching: GG node '{gg_content[:40]}...' "
                      f"matched Patient node '{patient_p.get('content', '')[:40]}...' "
                      f"(Sim: {max_sim:.2f})")
            else:
                result = MatchResult(
                    gg_content=gg_content,
                    patient_content="",
                    patient_status="",
                    similarity=max_sim,
                    matched=False,
                    match_method="embedding" if used_embedding else "fuzzy"
                )
            
            results.append(result)
        
        return results, used_embedding
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        print("[SemanticMatcher] Cache cleared")
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


# ==================== 便捷函数 ====================

_default_matcher: Optional[SemanticMatcher] = None


def get_semantic_matcher(
    similarity_threshold: float = 0.90,
    fuzzy_threshold: float = 0.80
) -> SemanticMatcher:
    """
    获取默认的语义匹配器实例（单例模式）
    
    Args:
        similarity_threshold: Embedding 相似度阈值
        fuzzy_threshold: Fuzzy Match 阈值
    
    Returns:
        SemanticMatcher 实例
    """
    global _default_matcher
    
    if _default_matcher is None:
        _default_matcher = SemanticMatcher(
            similarity_threshold=similarity_threshold,
            fuzzy_threshold=fuzzy_threshold
        )
    
    return _default_matcher


def match_p_nodes_semantic(
    gg_p_nodes: List[Dict[str, Any]],
    patient_p_nodes: List[Dict[str, Any]],
    similarity_threshold: float = 0.90
) -> Tuple[List[MatchResult], bool]:
    """
    便捷函数：执行 P-Node 语义匹配
    
    Args:
        gg_p_nodes: Golden Graph 的 P-Nodes
        patient_p_nodes: 患者的 P-Nodes
        similarity_threshold: 相似度阈值
    
    Returns:
        (匹配结果列表, 是否使用了 Embedding)
    """
    matcher = get_semantic_matcher(similarity_threshold=similarity_threshold)
    return matcher.match_p_nodes(gg_p_nodes, patient_p_nodes)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("Testing SemanticMatcher...")
    
    # 测试数据
    gg_p_nodes = [
        {"content": "Dyspnea"},
        {"content": "Chest pain radiating to left arm"},
        {"content": "Fever"},
    ]
    
    patient_p_nodes = [
        {"content": "Shortness of breath", "status": "Present"},
        {"content": "Severe chest pain with radiation to the left arm", "status": "Present"},
        {"content": "No fever", "status": "Absent"},
    ]
    
    matcher = SemanticMatcher(similarity_threshold=0.85)
    results, used_embedding = matcher.match_p_nodes(gg_p_nodes, patient_p_nodes)
    
    print(f"\nUsed Embedding: {used_embedding}")
    print(f"Results:")
    for r in results:
        print(f"  GG: '{r.gg_content}' -> Patient: '{r.patient_content}' "
              f"| Matched: {r.matched} | Sim: {r.similarity:.2f} | Status: {r.patient_status}")
















