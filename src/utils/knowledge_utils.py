"""
Knowledge Utilities for Phase 2
静态知识增强工具

功能：
1. load_disease_synonyms: 加载疾病同义词映射
2. build_pubmed_or_query: 构建 OR 组合的 PubMed 查询
3. get_synonyms_for_disease: 获取指定疾病的同义词列表
4. validate_open_targets_match: Open Targets 精确匹配验证
5. build_pubmed_statpearls_query: 构建 StatPearls 优先查询
6. build_pubmed_clinical_query: 构建临床文章降级查询
7. PubMedQueryBuilder: 三级瀑布流查询构建器
"""
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


# 同义词缓存（模块级单例）
_SYNONYMS_CACHE: Optional[Dict[str, Dict]] = None


def load_disease_synonyms() -> Dict[str, Dict]:
    """
    加载疾病同义词映射
    
    Returns:
        字典格式: {
            "01": {"name": "Possible NSTEMI / STEMI", "synonyms": ["Myocardial infarction", ...]},
            ...
        }
        
    如果文件不存在，返回空字典（容错处理）
    """
    global _SYNONYMS_CACHE
    
    if _SYNONYMS_CACHE is not None:
        return _SYNONYMS_CACHE
    
    # 尝试多个可能的路径
    possible_paths = [
        Path(__file__).parent.parent.parent / "config" / "disease_synonyms.json",
        Path("config/disease_synonyms.json"),
        Path("/home/intern/MYAgent/config/disease_synonyms.json"),
    ]
    
    for synonyms_path in possible_paths:
        if synonyms_path.exists():
            try:
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    _SYNONYMS_CACHE = json.load(f)
                print(f"[KnowledgeUtils] Loaded {len(_SYNONYMS_CACHE)} disease synonyms from {synonyms_path}")
                return _SYNONYMS_CACHE
            except Exception as e:
                print(f"[KnowledgeUtils] Error loading synonyms from {synonyms_path}: {e}")
                continue
    
    print("[KnowledgeUtils] WARNING: disease_synonyms.json not found, using empty synonyms")
    _SYNONYMS_CACHE = {}
    return _SYNONYMS_CACHE


def get_synonyms_for_disease(disease_id: str, disease_name: str) -> List[str]:
    """
    获取指定疾病的同义词列表
    
    Args:
        disease_id: 疾病 ID (如 "25", "44")
        disease_name: 疾病名称（作为 fallback）
    
    Returns:
        同义词列表（包含原始名称）
    """
    synonyms_map = load_disease_synonyms()
    
    # 清理 ID 格式 (去掉 "d_" 前缀)
    clean_id = disease_id.replace("d_", "")
    
    if clean_id in synonyms_map:
        entry = synonyms_map[clean_id]
        synonyms = entry.get("synonyms", [])
        if synonyms:
            return synonyms
    
    # Fallback: 返回原始名称
    return [disease_name]


def build_pubmed_or_query(
    disease_name: str,
    disease_id: str = "",
    require_review: bool = True,
    include_clinical_terms: bool = True
) -> str:
    """
    构建 OR 组合的 PubMed 查询
    
    使用同义词构建组合查询，提升召回率，同时严格过滤 Review/Guideline
    同义词从 config/disease_synonyms.json 加载
    
    Args:
        disease_name: 疾病名称
        disease_id: 疾病 ID (用于查找同义词，如 "d_44" 或 "44")
        require_review: 是否强制要求 Review/Guideline (默认 True)
        include_clinical_terms: 是否包含临床特征相关词 (默认 True)
    
    Returns:
        PubMed 查询字符串，格式严格为：
        ("Term1"[Title/Abstract] OR "Term2"[Title/Abstract]) AND 
        (Review[Publication Type] OR Practice Guideline[Publication Type])
    
    Example:
        >>> build_pubmed_or_query("Boerhaave", "44", include_clinical_terms=False)
        '("Boerhaave"[Title/Abstract] OR "Boerhaave syndrome"[Title/Abstract]) AND (Review[Publication Type] OR Practice Guideline[Publication Type])'
        
        >>> build_pubmed_or_query("Boerhaave", "44", include_clinical_terms=True)
        '("Boerhaave"[Title/Abstract] OR "Boerhaave syndrome"[Title/Abstract]) AND ("clinical presentation"[Title/Abstract] OR ...) AND (Review[Publication Type] OR ...)'
    """
    # 获取同义词
    synonyms = get_synonyms_for_disease(disease_id, disease_name)
    
    # 确保原始名称在列表中
    if disease_name not in synonyms:
        synonyms = [disease_name] + synonyms
    
    # 去重并限制数量（防止查询过长）
    seen = set()
    unique_synonyms = []
    for s in synonyms:
        s_lower = s.lower().strip()
        if s_lower and s_lower not in seen:
            seen.add(s_lower)
            unique_synonyms.append(s.strip())
    synonyms = unique_synonyms[:5]  # 最多 5 个同义词
    
    # 构建疾病名称 OR 部分
    disease_terms = " OR ".join([f'"{s}"[Title/Abstract]' for s in synonyms])
    query_parts = [f"({disease_terms})"]
    
    # 添加临床特征相关词
    if include_clinical_terms:
        clinical_terms = (
            '("clinical presentation"[Title/Abstract] OR '
            '"clinical features"[Title/Abstract] OR '
            '"signs and symptoms"[Title/Abstract] OR '
            '"diagnosis"[Title/Abstract])'
        )
        query_parts.append(clinical_terms)
    
    # 强制 Review/Guideline (严格隔离)
    if require_review:
        review_filter = "(Review[Publication Type] OR Practice Guideline[Publication Type])"
        query_parts.append(review_filter)
    
    query = " AND ".join(query_parts)
    
    return query


# ==================== 三级瀑布流 Query 构建器 ====================

def build_pubmed_statpearls_query(disease_name: str, disease_id: str = "") -> str:
    """
    构建 StatPearls 优先查询 (Level 1 - Best Quality)
    
    StatPearls 是标准化的医学百科，包含最纯净的临床特征信息
    
    Args:
        disease_name: 疾病名称
        disease_id: 疾病 ID
    
    Returns:
        PubMed 查询字符串，格式：
        ("{Disease}" OR "{Synonym1}" OR ...) AND "StatPearls"[Source]
    """
    # 获取同义词
    synonyms = get_synonyms_for_disease(disease_id, disease_name)
    
    # 确保原始名称在列表中
    if disease_name not in synonyms:
        synonyms = [disease_name] + synonyms
    
    # 去重并限制数量
    seen = set()
    unique_synonyms = []
    for s in synonyms:
        s_lower = s.lower().strip()
        if s_lower and s_lower not in seen:
            seen.add(s_lower)
            unique_synonyms.append(s.strip())
    synonyms = unique_synonyms[:5]
    
    # 构建查询
    disease_terms = " OR ".join([f'"{s}"[Title]' for s in synonyms])
    query = f'({disease_terms}) AND "StatPearls"[Source]'
    
    return query


def build_pubmed_clinical_query(disease_name: str, disease_id: str = "") -> str:
    """
    构建临床文章降级查询 (Level 3 - Fallback)
    
    当 Review/Guideline 无结果时的降级策略
    - 移除 Publication Type 过滤器
    - 强制标题级关键词匹配（高精度）
    
    Args:
        disease_name: 疾病名称
        disease_id: 疾病 ID
    
    Returns:
        PubMed 查询字符串，格式：
        ("{Disease}" OR ...) AND ("Clinical Features"[Title] OR "Diagnosis"[Title] OR "Presentation"[Title])
    """
    # 获取同义词
    synonyms = get_synonyms_for_disease(disease_id, disease_name)
    
    # 确保原始名称在列表中
    if disease_name not in synonyms:
        synonyms = [disease_name] + synonyms
    
    # 去重并限制数量
    seen = set()
    unique_synonyms = []
    for s in synonyms:
        s_lower = s.lower().strip()
        if s_lower and s_lower not in seen:
            seen.add(s_lower)
            unique_synonyms.append(s.strip())
    synonyms = unique_synonyms[:5]
    
    # 构建查询 - 注意：关键词必须在 [Title] 中（非 Title/Abstract）
    disease_terms = " OR ".join([f'"{s}"[Title/Abstract]' for s in synonyms])
    
    # 标题级关键词过滤（严格）
    title_keywords = (
        '("Clinical Features"[Title] OR '
        '"Diagnosis"[Title] OR '
        '"Presentation"[Title])'
    )
    
    query = f'({disease_terms}) AND {title_keywords}'
    
    return query


class PubMedQueryBuilder:
    """
    PubMed 三级瀑布流查询构建器
    
    搜索瀑布流：
    1. Level 1 (Best): StatPearls 医学百科
    2. Level 2 (Good): Reviews/Guidelines 综述文章
    3. Level 3 (Fallback): Clinical Articles 临床文章（标题级关键词过滤）
    """
    
    def __init__(self, disease_name: str, disease_id: str = ""):
        """
        初始化查询构建器
        
        Args:
            disease_name: 疾病名称
            disease_id: 疾病 ID
        """
        self.disease_name = disease_name
        self.disease_id = disease_id
        self._synonyms = None
    
    @property
    def synonyms(self) -> List[str]:
        """获取同义词列表（缓存）"""
        if self._synonyms is None:
            synonyms = get_synonyms_for_disease(self.disease_id, self.disease_name)
            if self.disease_name not in synonyms:
                synonyms = [self.disease_name] + synonyms
            
            # 去重
            seen = set()
            unique = []
            for s in synonyms:
                s_lower = s.lower().strip()
                if s_lower and s_lower not in seen:
                    seen.add(s_lower)
                    unique.append(s.strip())
            self._synonyms = unique[:5]
        return self._synonyms
    
    def build_level1_statpearls(self) -> str:
        """Level 1: StatPearls 查询"""
        return build_pubmed_statpearls_query(self.disease_name, self.disease_id)
    
    def build_level2_review(self) -> str:
        """Level 2: Review/Guideline 查询"""
        return build_pubmed_or_query(
            self.disease_name, 
            self.disease_id, 
            require_review=True,
            include_clinical_terms=False  # 不加临床词，让 Review 过滤器做主要筛选
        )
    
    def build_level3_clinical(self) -> str:
        """Level 3: Clinical Articles 降级查询"""
        return build_pubmed_clinical_query(self.disease_name, self.disease_id)
    
    def get_all_queries(self) -> List[Tuple[str, str, int]]:
        """
        获取所有级别的查询（用于瀑布流搜索）
        
        Returns:
            列表，每个元素为 (level_name, query, max_results)
        """
        return [
            ("StatPearls", self.build_level1_statpearls(), 3),      # StatPearls 最多 3 篇
            ("Review", self.build_level2_review(), 3),              # Review 最多 3 篇
            ("Clinical_Fallback", self.build_level3_clinical(), 2)  # 降级最多 2 篇
        ]


def build_open_targets_search_terms(disease_name: str, disease_id: str = "") -> List[str]:
    """
    构建 Open Targets 搜索词列表
    
    Args:
        disease_name: 疾病名称
        disease_id: 疾病 ID
    
    Returns:
        搜索词列表（按优先级排序）
    """
    synonyms = get_synonyms_for_disease(disease_id, disease_name)
    
    # Open Targets 通常对标准化医学术语效果更好
    # 所以优先使用同义词中的标准名称
    search_terms = []
    
    for syn in synonyms:
        # 清理名称
        cleaned = syn.strip()
        if cleaned and cleaned not in search_terms:
            search_terms.append(cleaned)
    
    # 确保原始名称也在列表中
    if disease_name not in search_terms:
        search_terms.append(disease_name)
    
    return search_terms[:3]  # 最多尝试 3 个


def estimate_token_count(text: str) -> int:
    """
    粗略估算文本的 Token 数量
    
    规则：英文约 4 字符 = 1 token
    
    Args:
        text: 输入文本
    
    Returns:
        估算的 Token 数量
    """
    if not text:
        return 0
    return len(text) // 4


def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """
    按估算 Token 数截断文本
    
    Args:
        text: 输入文本
        max_tokens: 最大 Token 数
    
    Returns:
        截断后的文本
    """
    if not text:
        return ""
    
    estimated_chars = max_tokens * 4
    if len(text) <= estimated_chars:
        return text
    
    # 在句子边界截断（尽量保持完整性）
    truncated = text[:estimated_chars]
    last_period = truncated.rfind('.')
    if last_period > estimated_chars * 0.7:  # 保留至少 70% 内容
        truncated = truncated[:last_period + 1]
    
    return truncated + "..."


def validate_open_targets_match(
    search_term: str,
    returned_name: str,
    returned_synonyms: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    验证 Open Targets 返回结果是否与搜索词匹配
    
    使用 Case-insensitive Substring Match 策略:
    - 搜索词包含在返回名称中，或
    - 返回名称包含在搜索词中，或
    - 搜索词出现在同义词列表中
    
    Args:
        search_term: 搜索词（如 "Boerhaave"）
        returned_name: API 返回的疾病名称（如 "Boerhaave syndrome"）
        returned_synonyms: API 返回的同义词结构 [{"relation": "...", "terms": [...]}]
    
    Returns:
        True = 匹配成功, False = 匹配失败
    
    Example:
        >>> validate_open_targets_match("Boerhaave", "Boerhaave syndrome")
        True
        >>> validate_open_targets_match("Boerhaave", "Epidermolysis Bullosa")
        False
    """
    if not search_term or not returned_name:
        return False
    
    search_lower = search_term.lower().strip()
    name_lower = returned_name.lower().strip()
    
    # 策略 1: 搜索词是返回名称的子串
    if search_lower in name_lower:
        return True
    
    # 策略 2: 返回名称是搜索词的子串（处理 "Boerhaave syndrome" 搜索返回 "Boerhaave"）
    if name_lower in search_lower:
        return True
    
    # 策略 3: 检查同义词列表
    if returned_synonyms:
        for syn_group in returned_synonyms:
            terms = syn_group.get("terms", [])
            for term in terms:
                if isinstance(term, str):
                    term_lower = term.lower().strip()
                    if search_lower in term_lower or term_lower in search_lower:
                        return True
    
    return False


def get_all_search_terms_for_disease(disease_id: str, disease_name: str) -> List[str]:
    """
    获取指定疾病的所有搜索词列表（用于 Open Targets 重试）
    
    按优先级排序：原始名称 -> 同义词列表
    
    Args:
        disease_id: 疾病 ID (如 "d_44")
        disease_name: 疾病原始名称
    
    Returns:
        搜索词列表（不重复，按优先级排序）
    """
    # 获取同义词
    synonyms = get_synonyms_for_disease(disease_id, disease_name)
    
    # 构建搜索词列表（去重）
    search_terms = []
    seen = set()
    
    # 首先添加原始名称
    if disease_name and disease_name.lower() not in seen:
        search_terms.append(disease_name)
        seen.add(disease_name.lower())
    
    # 然后添加同义词
    for syn in synonyms:
        if syn and syn.lower() not in seen:
            search_terms.append(syn)
            seen.add(syn.lower())
    
    return search_terms


class ContextBudgetManager:
    """
    上下文预算管理器
    
    用于控制 Batch Reasoning 的总上下文长度
    """
    
    def __init__(self, max_total_tokens: int = 30000):
        """
        初始化上下文预算管理器
        
        Args:
            max_total_tokens: 最大总 Token 数 (保守设置为 30k，留出空间给 Prompt 和输出)
        """
        self.max_total_tokens = max_total_tokens
        self.reserved_for_prompt = 3000  # 为 Prompt 模板预留
        self.reserved_for_output = 4000  # 为输出预留
        self.available_for_content = max_total_tokens - self.reserved_for_prompt - self.reserved_for_output
    
    def allocate_budget(
        self,
        candidates: List[Dict],
        open_targets_texts: Dict[str, str],
        pubmed_texts: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:
        """
        分配上下文预算
        
        策略：
        1. 完整保留 Open Targets 描述
        2. 剩余预算平均分配给 PubMed Review
        
        Args:
            candidates: 候选疾病列表
            open_targets_texts: {d_id: text} Open Targets 结果
            pubmed_texts: {d_id: text} PubMed Review 结果
        
        Returns:
            {d_id: {"open_targets": text, "pubmed": text}}
        """
        result = {}
        n_candidates = len(candidates)
        
        if n_candidates == 0:
            return result
        
        # Step 1: 计算 Open Targets 总消耗
        ot_total_tokens = 0
        for d_id, text in open_targets_texts.items():
            ot_total_tokens += estimate_token_count(text)
        
        # Step 2: 计算 PubMed 可用预算
        pubmed_budget = self.available_for_content - ot_total_tokens
        pubmed_per_candidate = max(500, pubmed_budget // n_candidates) if n_candidates > 0 else 0
        
        print(f"[ContextBudget] Total available: {self.available_for_content} tokens")
        print(f"[ContextBudget] Open Targets used: {ot_total_tokens} tokens")
        print(f"[ContextBudget] PubMed per candidate: {pubmed_per_candidate} tokens")
        
        # Step 3: 分配并截断
        for candidate in candidates:
            d_id = candidate.get("id", "")
            
            # Open Targets: 完整保留
            ot_text = open_targets_texts.get(d_id, "")
            
            # PubMed: 按预算截断
            pm_text = pubmed_texts.get(d_id, "")
            if pubmed_per_candidate > 0:
                pm_text = truncate_text_by_tokens(pm_text, pubmed_per_candidate)
            
            result[d_id] = {
                "open_targets": ot_text,
                "pubmed": pm_text
            }
        
        return result

