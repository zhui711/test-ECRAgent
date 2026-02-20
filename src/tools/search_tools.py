"""
Search Tools for Phase 2
实现多源知识搜索工具，支持降级链

搜索链顺序（K-Gen）:
1. Open Targets Platform (GraphQL API) - 首选，提供标准化疾病描述
2. Wikipedia (多镜像尝试) - 可选，如果可访问
3. PubMed Review/Guideline - 高质量综述文章
4. PubMed 一般文献 - 更宽松的搜索
5. Query Expansion + PubMed - 去修饰词后重新搜索

设计原则：
- 医学合理性：搜索策略基于循证医学原则
- 科学严谨性：优先使用高质量来源（综述、指南）
- 容错性：多层降级确保尽可能获取知识
"""
import os
import re
import time
import json
import requests
from typing import List, Optional, Dict, Any, Tuple
from bs4 import BeautifulSoup
from Bio import Entrez
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

# 导入 knowledge_utils 中的函数（延迟导入避免循环依赖）
# validate_open_targets_match 在方法内部导入


# ==================== Query Expansion 规则 ====================
# 医学术语修饰词移除规则（保守、医学合理的转换）
MODIFIER_REMOVAL_RULES = [
    # 时间/急性程度修饰词
    (r'^Acute\s+', ''),           # Acute bronchitis → bronchitis
    (r'^Chronic\s+', ''),         # Chronic bronchitis → bronchitis
    (r'^Subacute\s+', ''),        # Subacute thyroiditis → thyroiditis
    # 发生方式修饰词
    (r'^Spontaneous\s+', ''),     # Spontaneous rib fracture → rib fracture
    (r'^Traumatic\s+', ''),       # Traumatic pneumothorax → pneumothorax
    # 病因分类修饰词
    (r'^Primary\s+', ''),         # Primary hypertension → hypertension
    (r'^Secondary\s+', ''),       # Secondary hypertension → hypertension
    (r'^Idiopathic\s+', ''),      # Idiopathic pulmonary fibrosis → pulmonary fibrosis
    # 表现类型修饰词
    (r'^Atypical\s+', ''),        # Atypical pneumonia → pneumonia
    (r'^Typical\s+', ''),         # Typical angina → angina
    # 严重程度修饰词
    (r'^Severe\s+', ''),          # Severe asthma → asthma
    (r'^Mild\s+', ''),            # Mild TBI → TBI
    # 移除括号内容（通常是补充说明）
    (r'\s*\([^)]*\)\s*$', ''),    # Disease (subtype) → Disease
]

# 医学缩写/同义词映射（常见且明确的转换）
MEDICAL_SYNONYMS = {
    "GERD": "Gastroesophageal reflux disease",
    "Boerhaave": "Boerhaave syndrome",
    "PE": "Pulmonary embolism",
    "MI": "Myocardial infarction",
    "DVT": "Deep vein thrombosis",
    "COPD": "Chronic obstructive pulmonary disease",
    "CHF": "Congestive heart failure",
    "UTI": "Urinary tract infection",
    "TIA": "Transient ischemic attack",
    "SLE": "Systemic lupus erythematosus",
    "ARDS": "Acute respiratory distress syndrome",
    "CAD": "Coronary artery disease",
    "AFib": "Atrial fibrillation",
    "HTN": "Hypertension",
}


def clean_disease_name(disease_name: str) -> str:
    """
    清理疾病名称，移除可能影响搜索的特殊字符
    
    Args:
        disease_name: 原始疾病名称
    
    Returns:
        清理后的疾病名称
    """
    # 处理 "/" 分隔的名称，取第一部分
    if '/' in disease_name:
        parts = [p.strip() for p in disease_name.split('/')]
        # 返回最长的部分（通常是最完整的名称）
        disease_name = max(parts, key=len)
    
    # 移除括号内容（保留主名称）
    disease_name = re.sub(r'\s*\([^)]*\)\s*', ' ', disease_name).strip()
    
    # 移除多余空格
    disease_name = re.sub(r'\s+', ' ', disease_name)
    
    return disease_name


def expand_query_conservative(disease_name: str) -> List[str]:
    """
    保守的 Query Expansion：仅进行医学合理的转换
    
    策略：
    1. 首先清理疾病名称
    2. 尝试同义词映射（缩写展开）
    3. 然后尝试移除修饰词
    4. 保持原始词在列表中作为首选
    
    Args:
        disease_name: 原始疾病名称
    
    Returns:
        扩展后的搜索词列表（按优先级排序，原始词优先）
    """
    # 首先清理名称
    cleaned_name = clean_disease_name(disease_name)
    
    expanded = []
    # 如果清理后的名称不同，两个都加入
    if cleaned_name != disease_name:
        expanded.append(cleaned_name)
    expanded.append(disease_name)
    
    # 1. 检查同义词映射
    for name in [cleaned_name, disease_name]:
        if name.upper() in MEDICAL_SYNONYMS:
            synonym = MEDICAL_SYNONYMS[name.upper()]
            if synonym not in expanded:
                expanded.append(synonym)
        
        # 也检查精确匹配
        if name in MEDICAL_SYNONYMS:
            synonym = MEDICAL_SYNONYMS[name]
            if synonym not in expanded:
                expanded.append(synonym)
    
    # 2. 应用修饰词移除规则
    for name in [cleaned_name, disease_name]:
        for pattern, replacement in MODIFIER_REMOVAL_RULES:
            new_term = re.sub(pattern, replacement, name, flags=re.IGNORECASE).strip()
            if new_term and new_term != name and new_term not in expanded:
                # 确保新词有意义（至少2个单词或足够长）
                if len(new_term.split()) >= 1 and len(new_term) >= 5:
                    expanded.append(new_term)
    
    # 去重并保持顺序
    seen = set()
    result = []
    for term in expanded:
        if term.lower() not in seen:
            seen.add(term.lower())
            result.append(term)
    
    return result


# ==================== Open Targets Platform Tool ====================

class OpenTargetsTool:
    """
    Open Targets Platform 搜索工具
    使用 GraphQL API 获取疾病的标准化描述
    
    优点：
    - 提供标准化的疾病描述（类似教科书）
    - 包含症状信息
    - 稳定可靠，无网络访问限制
    
    重构版本 (Debug Enhancement):
    - 实现 Exact Match 熔断：Case-insensitive Substring Match
    - 同义词穷尽重试逻辑
    - 返回原始 API 响应用于 Debug
    """
    
    API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    
    def __init__(self, timeout: int = 30):
        """
        初始化 OpenTargetsTool
        
        Args:
            timeout: 请求超时时间（秒）
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        # 存储最后一次原始响应（用于 Debug）
        self.last_raw_response: Optional[Dict[str, Any]] = None
        self.last_search_term: str = ""
        print("[OpenTargetsTool] Initialized (with Exact Match validation)")
    
    def search_disease_with_validation(
        self, 
        disease_name: str,
        search_terms: Optional[List[str]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        搜索疾病并验证匹配，支持同义词穷尽重试
        
        Args:
            disease_name: 原始疾病名称
            search_terms: 搜索词列表（可选，如果不提供则使用原始名称）
        
        Returns:
            (疾病信息字典, 原始 API 响应) 元组
            如果匹配失败，返回 (None, last_raw_response)
        """
        from src.utils.knowledge_utils import validate_open_targets_match
        
        # 如果没有提供搜索词，使用原始名称
        if search_terms is None:
            search_terms = [clean_disease_name(disease_name)]
        
        self.last_raw_response = None
        self.last_search_term = ""
        
        for term in search_terms:
            print(f"[OpenTargetsTool] Trying search term: '{term}'...")
            self.last_search_term = term
            
            try:
                # Step 1: 搜索疾病获取 EFO ID (带验证)
                efo_id, raw_search_response = self._search_disease_id_with_validation(term)
                self.last_raw_response = raw_search_response
                
                if not efo_id:
                    print(f"[OpenTargetsTool] No valid match for '{term}', trying next synonym...")
                    continue
                
                # Step 2: 获取疾病详细信息
                disease_info = self._get_disease_details(efo_id)
                if disease_info:
                    print(f"[OpenTargetsTool] ✓ Successfully matched: '{term}' -> '{disease_info.get('name', '')}'")
                    return disease_info, raw_search_response
                    
            except Exception as e:
                print(f"[OpenTargetsTool] Error searching '{term}': {e}")
                continue
        
        print(f"[OpenTargetsTool] ✗ No matching disease found after trying all synonyms for '{disease_name}'")
        return None, self.last_raw_response
    
    def search_disease(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """
        搜索疾病并获取详细信息（向后兼容方法）
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            疾病信息字典，包含 id, name, description 等
        """
        result, _ = self.search_disease_with_validation(disease_name)
        return result
    
    def get_disease_description(self, disease_name: str) -> str:
        """
        获取疾病的描述文本（用于 K-Gen，向后兼容）
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            疾病描述文本，失败返回空字符串
        """
        result, _ = self.get_disease_description_with_raw(disease_name)
        return result
    
    def get_disease_description_with_raw(
        self, 
        disease_name: str,
        search_terms: Optional[List[str]] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        获取疾病的描述文本，同时返回原始 API 响应（用于 Debug）
        
        Args:
            disease_name: 疾病名称
            search_terms: 搜索词列表（用于同义词穷尽重试）
        
        Returns:
            (描述文本, 原始 API 响应) 元组
        """
        disease_info, raw_response = self.search_disease_with_validation(
            disease_name, 
            search_terms
        )
        
        if not disease_info:
            return "", raw_response
        
        description = disease_info.get("description", "")
        if description:
            # 格式化输出
            name = disease_info.get("name", disease_name)
            result = f"Disease: {name}\n\nDescription: {description}"
            
            # 添加治疗领域（如果有）
            therapeutic_areas = disease_info.get("therapeuticAreas", [])
            if therapeutic_areas:
                areas = [ta.get("name", "") for ta in therapeutic_areas if ta.get("name")]
                if areas:
                    result += f"\n\nTherapeutic Areas: {', '.join(areas)}"
            
            print(f"[OpenTargetsTool] Retrieved description ({len(result)} chars)")
            return result, raw_response
        
        return "", raw_response
    
    def _search_disease_id_with_validation(
        self, 
        disease_name: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        搜索疾病获取 EFO ID，带精确匹配验证
        
        Args:
            disease_name: 搜索词
        
        Returns:
            (EFO ID, 原始 API 响应) 元组
            如果验证失败，返回 (None, raw_response)
        """
        from src.utils.knowledge_utils import validate_open_targets_match
        
        query = """
        query SearchDisease($queryString: String!) {
            search(queryString: $queryString, entityNames: ["disease"], page: {index: 0, size: 5}) {
                hits {
                    id
                    name
                    entity
                }
            }
        }
        """
        
        variables = {"queryString": disease_name}
        
        try:
            response = self.session.post(
                self.API_URL,
                json={"query": query, "variables": variables},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            hits = data.get("data", {}).get("search", {}).get("hits", [])
            
            # 遍历所有结果，验证匹配
            for hit in hits:
                if hit.get("entity") != "disease":
                    continue
                
                returned_name = hit.get("name", "")
                efo_id = hit.get("id", "")
                
                # 精确匹配验证 (Case-insensitive Substring Match)
                if validate_open_targets_match(disease_name, returned_name, None):
                    print(f"[OpenTargetsTool] ✓ Validated match: '{disease_name}' -> '{returned_name}' ({efo_id})")
                    return efo_id, data
                else:
                    print(f"[OpenTargetsTool] ✗ Match failed: '{disease_name}' vs '{returned_name}' (skipping)")
            
            # 所有结果都不匹配
            if hits:
                print(f"[OpenTargetsTool] No valid match found in {len(hits)} results")
            return None, data
            
        except Exception as e:
            print(f"[OpenTargetsTool] Search error: {e}")
            return None, None
    
    def _search_disease_id(self, disease_name: str) -> Optional[str]:
        """搜索疾病获取 EFO ID（向后兼容）"""
        efo_id, _ = self._search_disease_id_with_validation(disease_name)
        return efo_id
    
    def _get_disease_details(self, efo_id: str) -> Optional[Dict[str, Any]]:
        """获取疾病详细信息"""
        query = """
        query GetDisease($efoId: String!) {
            disease(efoId: $efoId) {
                id
                name
                description
                synonyms {
                    relation
                    terms
                }
                therapeuticAreas {
                    id
                    name
                }
            }
        }
        """
        
        variables = {"efoId": efo_id}
        
        try:
            response = self.session.post(
                self.API_URL,
                json={"query": query, "variables": variables},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            disease = data.get("data", {}).get("disease")
            return disease
            
        except Exception as e:
            print(f"[OpenTargetsTool] Get details error: {e}")
            return None


# ==================== Wikipedia Tool (多镜像支持) ====================

class WikiTool:
    """
    Wikipedia 搜索工具
    用于获取疾病的 "Signs and symptoms" 章节内容
    
    支持多个镜像站点，提高可访问性：
    1. en.wikipedia.org (主站)
    2. en.m.wikipedia.org (移动版)
    """
    
    # Wikipedia 镜像站点（按优先级排序）
    WIKI_MIRRORS = [
        "https://en.m.wikipedia.org/w/api.php",  # 移动版（通常更容易访问）
        "https://en.wikipedia.org/w/api.php",    # 主站
    ]
    
    # 更真实的 User-Agent（模拟 Chrome 浏览器）
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    ]
    
    def __init__(self, timeout: int = 30, max_retries: int = 2):
        """
        初始化 WikiTool
        
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 每个镜像的最大重试次数
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.current_mirror_idx = 0
        
        # 创建 Session
        self.session = requests.Session()
        
        # 设置请求头（模拟真实浏览器）
        self.session.headers.update({
            "User-Agent": self.USER_AGENTS[0],
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        })
        
        # 配置代理（从环境变量读取）
        self._setup_proxy()
        
        # 目标 section 名称（按优先级排序）
        self.target_sections = [
            "Signs and symptoms",
            "Signs & symptoms", 
            "Symptoms and signs",
            "Clinical presentation",
            "Symptoms",
            "Signs",
            "Presentation"
        ]
        
        print(f"[WikiTool] Initialized with {len(self.WIKI_MIRRORS)} mirrors")
    
    def _setup_proxy(self):
        """配置代理"""
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        wiki_proxy = os.getenv("WIKI_PROXY")
        
        if wiki_proxy:
            self.proxies = {"http": wiki_proxy, "https": wiki_proxy}
            print(f"[WikiTool] Using WIKI_PROXY: {wiki_proxy}")
        elif http_proxy or https_proxy:
            self.proxies = {}
            if http_proxy:
                self.proxies["http"] = http_proxy
            if https_proxy:
                self.proxies["https"] = https_proxy
            print(f"[WikiTool] Using system proxy")
        else:
            self.proxies = None
    
    def _make_request(
        self, 
        params: Dict[str, Any], 
        mirror_idx: int = 0,
        attempt: int = 1, 
        use_proxy: bool = True
    ) -> Optional[Dict]:
        """
        发送 API 请求，支持多镜像和重试
        """
        if mirror_idx >= len(self.WIKI_MIRRORS):
            return None
        
        base_url = self.WIKI_MIRRORS[mirror_idx]
        
        try:
            current_proxies = self.proxies if (use_proxy and self.proxies) else None
            
            response = self.session.get(
                base_url,
                params=params,
                timeout=self.timeout,
                proxies=current_proxies
            )
            response.raise_for_status()
            return response.json()
            
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as e:
            if use_proxy and self.proxies:
                # 禁用代理重试
                return self._make_request(params, mirror_idx, attempt, use_proxy=False)
            elif attempt < self.max_retries:
                time.sleep(1)
                return self._make_request(params, mirror_idx, attempt + 1, use_proxy=False)
            else:
                # 尝试下一个镜像
                print(f"[WikiTool] Mirror {mirror_idx} failed, trying next...")
                return self._make_request(params, mirror_idx + 1, 1, use_proxy=True)
            
        except requests.exceptions.Timeout:
            if attempt < self.max_retries:
                time.sleep(1)
                return self._make_request(params, mirror_idx, attempt + 1, use_proxy=False)
            else:
                # 尝试下一个镜像
                print(f"[WikiTool] Mirror {mirror_idx} timeout, trying next...")
                return self._make_request(params, mirror_idx + 1, 1, use_proxy=True)
            
        except requests.exceptions.RequestException as e:
            print(f"[WikiTool] Request error: {e}")
            if mirror_idx + 1 < len(self.WIKI_MIRRORS):
                return self._make_request(params, mirror_idx + 1, 1, use_proxy=True)
            return None
    
    def search_signs_symptoms(self, disease_name: str) -> str:
        """
        搜索指定疾病的 "Signs and symptoms" 章节
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            清洗后的纯文本，如果失败则返回空字符串
        """
        print(f"[WikiTool] Searching for '{disease_name}'...")
        
        try:
            # Step 1: 获取页面的 section 列表
            sections = self._get_sections(disease_name)
            if not sections:
                print(f"[WikiTool] No sections found for '{disease_name}', trying summary...")
                return self._get_summary_fallback(disease_name)
            
            # Step 2: 查找目标 section 的 index
            section_index = self._find_section_index(sections)
            if section_index is None:
                print(f"[WikiTool] Target section not found, using summary...")
                return self._get_summary_fallback(disease_name)
            
            # Step 3: 获取指定 section 的内容
            section_text = self._get_section_by_index(disease_name, section_index)
            if section_text:
                cleaned = self._clean_text(section_text)
                print(f"[WikiTool] Successfully retrieved {len(cleaned)} chars")
                return cleaned
            
            return self._get_summary_fallback(disease_name)
            
        except Exception as e:
            print(f"[WikiTool] Error: {e}")
            return ""
    
    def _get_sections(self, page_title: str) -> List[Dict[str, Any]]:
        """获取页面的所有 section 列表"""
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "sections",
            "format": "json"
        }
        
        data = self._make_request(params)
        if not data:
            return []
        
        if "error" in data:
            # 页面不存在，尝试搜索
            return self._search_and_get_sections(page_title)
        
        return data.get("parse", {}).get("sections", [])
    
    def _search_and_get_sections(self, query: str) -> List[Dict[str, Any]]:
        """搜索页面并获取 sections"""
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json"
        }
        
        search_data = self._make_request(search_params)
        if not search_data:
            return []
        
        results = search_data.get("query", {}).get("search", [])
        if not results:
            return []
        
        actual_title = results[0]["title"]
        print(f"[WikiTool] Found page: '{actual_title}'")
        
        parse_params = {
            "action": "parse",
            "page": actual_title,
            "prop": "sections",
            "format": "json"
        }
        
        data = self._make_request(parse_params)
        if not data:
            return []
        
        return data.get("parse", {}).get("sections", [])
    
    def _find_section_index(self, sections: List[Dict[str, Any]]) -> Optional[str]:
        """在 sections 列表中查找目标 section 的 index"""
        for target in self.target_sections:
            target_lower = target.lower()
            for section in sections:
                section_title = section.get("line", "").lower()
                if target_lower == section_title or target_lower in section_title:
                    return section.get("index")
        return None
    
    def _get_section_by_index(self, page_title: str, section_index: str) -> Optional[str]:
        """根据 section index 获取内容"""
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "section": section_index,
            "format": "json"
        }
        
        data = self._make_request(params)
        if not data:
            return None
        
        if "parse" in data and "text" in data["parse"]:
            html_text = data["parse"]["text"].get("*", "")
            if html_text:
                return html_text
        
        return None
    
    def _get_summary_fallback(self, page_title: str) -> str:
        """降级方案：获取页面摘要"""
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "extracts",
            "exintro": True,
            "exlimit": 1,
            "format": "json"
        }
        
        data = self._make_request(params)
        if not data:
            return ""
        
        pages = data.get("query", {}).get("pages", {})
        
        for page_data in pages.values():
            extract = page_data.get("extract", "")
            if extract:
                cleaned = self._clean_text(extract)
                if len(cleaned) > 2000:
                    cleaned = cleaned[:2000] + "..."
                print(f"[WikiTool] Using summary fallback: {len(cleaned)} chars")
                return cleaned
        
        return ""
    
    def _clean_text(self, html_text: str) -> str:
        """清洗 HTML 文本，提取纯文本"""
        soup = BeautifulSoup(html_text, "html.parser")
        
        for element in soup.find_all(['sup', 'style', 'script', 'table', 'img']):
            element.decompose()
        
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
        return text.strip()


# ==================== PubMed Tool ====================

class PubMedTool:
    """
    PubMed 搜索工具
    用于搜索医学文献，支持 Query Expansion
    
    搜索策略：
    1. Review/Guideline 类型（最高质量）
    2. 一般临床文献（降级）
    3. Query Expansion 后重新搜索（最后手段）
    """
    
    def __init__(self):
        self.email = os.getenv("PUBMED_EMAIL", "your_email@example.com")
        self.api_key = os.getenv("PUBMED_API_KEY")
        
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
            print(f"[PubMedTool] Initialized with API key")
        else:
            print(f"[PubMedTool] Initialized without API key (rate limited)")
    
    def search_general_features(
        self,
        disease_name: str,
        max_results: int = 5,
        use_expansion: bool = True
    ) -> str:
        """
        搜索疾病的通用症状特征（用于 K-Gen）
        
        搜索策略：
        1. 首先搜索 Review/Guideline 类型
        2. 降级到一般临床文献
        3. 使用 Query Expansion 重新搜索
        
        Args:
            disease_name: 疾病名称
            max_results: 最大返回结果数
            use_expansion: 是否使用 Query Expansion（默认 True）
        
        Returns:
            合并后的摘要文本
        """
        print(f"[PubMedTool] Searching general features for '{disease_name}'...")
        
        # 尝试 1: Review/Guideline
        result = self._search_review_guideline(disease_name, max_results)
        if result:
            return result
        
        # 尝试 2: 一般临床文献
        result = self._search_general_fallback(disease_name, max_results)
        if result:
            return result
        
        # 尝试 3: Query Expansion
        if use_expansion:
            expanded_terms = expand_query_conservative(disease_name)
            print(f"[PubMedTool] Trying Query Expansion: {expanded_terms}")
            
            for term in expanded_terms[1:]:  # 跳过原始词（已尝试）
                print(f"[PubMedTool] Trying expanded term: '{term}'")
                
                # 先尝试 Review/Guideline
                result = self._search_review_guideline(term, max_results)
                if result:
                    print(f"[PubMedTool] Found results with expanded term: '{term}'")
                    return result
                
                # 再尝试一般文献
                result = self._search_general_fallback(term, max_results)
                if result:
                    print(f"[PubMedTool] Found fallback results with: '{term}'")
                    return result
        
        print(f"[PubMedTool] No results found for '{disease_name}' after all attempts")
        return ""
    
    def _search_review_guideline(self, disease_name: str, max_results: int = 5) -> str:
        """搜索 Review/Guideline 类型文章"""
        try:
            query = (
                f'("{disease_name}"[Title]) AND '
                f'("clinical features"[Title/Abstract] OR "signs and symptoms"[Title/Abstract] OR "diagnosis"[Title/Abstract]) AND '
                f'("Review"[Publication Type] OR "Practice Guideline"[Publication Type])'
            )
            
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results.get("IdList", [])
            print(f"[PubMedTool] Found {len(id_list)} Review/Guideline articles")
            
            if not id_list:
                return ""
            
            abstracts = self._fetch_abstracts(id_list, max_results)
            if not abstracts:
                return ""
            
            merged_text = "\n\n---\n\n".join(abstracts)
            print(f"[PubMedTool] Merged {len(abstracts)} abstracts ({len(merged_text)} chars)")
            return merged_text
            
        except Exception as e:
            print(f"[PubMedTool] Review/Guideline search error: {e}")
            return ""
    
    def _search_general_fallback(self, disease_name: str, max_results: int = 5) -> str:
        """降级搜索：一般临床文献"""
        try:
            query = (
                f'("{disease_name}"[Title/Abstract]) AND '
                f'("clinical features"[All Fields] OR "signs and symptoms"[All Fields] OR "clinical presentation"[All Fields])'
            )
            
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results.get("IdList", [])
            print(f"[PubMedTool] Fallback found {len(id_list)} articles")
            
            if not id_list:
                return ""
            
            abstracts = self._fetch_abstracts(id_list, max_results)
            if not abstracts:
                return ""
            
            merged_text = "\n\n---\n\n".join(abstracts)
            print(f"[PubMedTool] Fallback merged {len(abstracts)} abstracts ({len(merged_text)} chars)")
            return merged_text
            
        except Exception as e:
            print(f"[PubMedTool] Fallback search error: {e}")
            return ""
    
    def _fetch_abstracts(self, id_list: List[str], max_results: int) -> List[str]:
        """批量获取摘要"""
        abstracts = []
        for pmid in id_list[:max_results + 3]:
            abstract_text = self._fetch_abstract(pmid)
            if abstract_text and len(abstract_text) >= 100:
                abstracts.append(abstract_text)
                if len(abstracts) >= max_results:
                    break
        return abstracts
    
    def search_differential(
        self, 
        disease_a: str, 
        disease_b: str,
        max_results: int = 3
    ) -> List[str]:
        """
        搜索两个疾病之间的鉴别诊断文献
        """
        print(f"[PubMedTool] Searching differential: '{disease_a}' vs '{disease_b}'")
        
        try:
            query = (
                f'(("Differential diagnosis"[MeSH Terms] OR "Differential diagnosis"[All Fields]) '
                f'AND ("{disease_a}"[All Fields] AND "{disease_b}"[All Fields]))'
            )
            
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results.get("IdList", [])
            print(f"[PubMedTool] Found {len(id_list)} articles")
            
            if not id_list:
                return self._fallback_differential_search(disease_a, disease_b, max_results)
            
            abstracts = self._fetch_abstracts(id_list, max_results)
            print(f"[PubMedTool] Retrieved {len(abstracts)} valid abstracts")
            return abstracts
            
        except Exception as e:
            print(f"[PubMedTool] Error searching differential: {e}")
            return []
    
    def _fallback_differential_search(
        self, 
        disease_a: str, 
        disease_b: str,
        max_results: int = 3
    ) -> List[str]:
        """降级搜索：更宽松的鉴别诊断查询"""
        print(f"[PubMedTool] Trying fallback search...")
        
        try:
            query = f'("{disease_a}"[All Fields] AND "{disease_b}"[All Fields])'
            
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results.get("IdList", [])
            print(f"[PubMedTool] Fallback found {len(id_list)} articles")
            
            if not id_list:
                return []
            
            return self._fetch_abstracts(id_list, max_results)
            
        except Exception as e:
            print(f"[PubMedTool] Fallback search error: {e}")
            return []
    
    def _fetch_abstract(self, pmid: str) -> str:
        """获取指定 PMID 的摘要"""
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="abstract",
                retmode="xml"
            )
            fetch_results = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            article_results = fetch_results.get("PubmedArticle", [])
            if not article_results:
                return ""
            
            article = article_results[0]
            
            title = (
                article.get("MedlineCitation", {})
                .get("Article", {})
                .get("ArticleTitle", "")
            )
            
            abstract_sections = (
                article.get("MedlineCitation", {})
                .get("Article", {})
                .get("Abstract", {})
                .get("AbstractText", [])
            )
            
            if not abstract_sections:
                return ""
            
            abstract_parts = []
            for section in abstract_sections:
                if hasattr(section, 'attributes') and section.attributes:
                    label = section.attributes.get("Label", "")
                    if label:
                        abstract_parts.append(f"{label}: {section}")
                    else:
                        abstract_parts.append(str(section))
                else:
                    abstract_parts.append(str(section))
            
            abstract_text = "\n\n".join(abstract_parts)
            
            if title:
                return f"Title: {title}\n\n{abstract_text}"
            
            return abstract_text
            
        except Exception as e:
            print(f"[PubMedTool] Error fetching abstract for PMID {pmid}: {e}")
            return ""


# ==================== 知识搜索聚合器 ====================

class KnowledgeSearchAggregator:
    """
    知识搜索聚合器
    实现完整的降级链，用于 K-Gen 阶段
    
    搜索链顺序：
    1. Open Targets Platform (首选)
    2. Wikipedia (可选，如果可访问)
    3. PubMed Review/Guideline
    4. PubMed 一般文献
    5. Query Expansion + PubMed
    """
    
    def __init__(self, enable_wikipedia: bool = True):
        """
        初始化搜索聚合器
        
        Args:
            enable_wikipedia: 是否启用 Wikipedia 搜索
        """
        self.open_targets = OpenTargetsTool()
        self.wiki_tool = WikiTool() if enable_wikipedia else None
        self.pubmed_tool = PubMedTool()
        
        print(f"[KnowledgeSearchAggregator] Initialized (Wikipedia: {'enabled' if enable_wikipedia else 'disabled'})")
    
    def search_disease_knowledge(
        self, 
        disease_name: str,
        require_symptoms: bool = True
    ) -> Tuple[str, str]:
        """
        搜索疾病知识，使用完整的降级链
        
        Args:
            disease_name: 疾病名称
            require_symptoms: 是否要求结果包含症状信息
        
        Returns:
            (知识文本, 来源标签) 元组
            来源标签: "OpenTargets", "Wikipedia", "PubMed_Review", "PubMed_General", "PubMed_Expanded", "None"
        """
        print(f"\n[KnowledgeSearch] Starting search for '{disease_name}'...")
        
        # 1. Open Targets Platform (首选)
        print("[KnowledgeSearch] Trying Open Targets Platform...")
        ot_result = self.open_targets.get_disease_description(disease_name)
        if ot_result and len(ot_result) >= 100:
            # 检查是否包含症状相关内容
            if not require_symptoms or self._contains_symptom_info(ot_result):
                print(f"[KnowledgeSearch] ✓ Found in Open Targets ({len(ot_result)} chars)")
                return ot_result, "OpenTargets"
            else:
                print("[KnowledgeSearch] Open Targets result lacks symptom info, continuing...")
        
        # 2. Wikipedia (如果启用)
        if self.wiki_tool:
            print("[KnowledgeSearch] Trying Wikipedia...")
            wiki_result = self.wiki_tool.search_signs_symptoms(disease_name)
            if wiki_result and len(wiki_result) >= 100:
                print(f"[KnowledgeSearch] ✓ Found in Wikipedia ({len(wiki_result)} chars)")
                return wiki_result, "Wikipedia"
        
        # 3-5. PubMed (带 Query Expansion)
        print("[KnowledgeSearch] Trying PubMed...")
        pubmed_result = self.pubmed_tool.search_general_features(disease_name, use_expansion=True)
        if pubmed_result and len(pubmed_result) >= 100:
            # 判断来源类型
            source = "PubMed_Review"  # 默认假设是 Review
            if "Fallback" in pubmed_result or "expanded" in pubmed_result.lower():
                source = "PubMed_General"
            print(f"[KnowledgeSearch] ✓ Found in PubMed ({len(pubmed_result)} chars)")
            return pubmed_result, source
        
        # 所有来源都失败
        print(f"[KnowledgeSearch] ✗ No results found for '{disease_name}'")
        return "", "None"
    
    def _contains_symptom_info(self, text: str) -> bool:
        """检查文本是否包含症状相关信息"""
        symptom_keywords = [
            "symptom", "sign", "present", "manifest", "feature",
            "pain", "fever", "cough", "dyspnea", "fatigue",
            "nausea", "vomiting", "headache", "chest", "abdominal"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in symptom_keywords)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Search Tools")
    print("=" * 60)
    
    # 测试 Query Expansion
    print("\n1. Testing Query Expansion:")
    test_terms = ["Spontaneous rib fracture", "GERD", "Acute bronchitis", "Boerhaave"]
    for term in test_terms:
        expanded = expand_query_conservative(term)
        print(f"  '{term}' -> {expanded}")
    
    # 测试 Open Targets
    print("\n2. Testing Open Targets:")
    ot = OpenTargetsTool()
    result = ot.get_disease_description("pneumonia")
    print(f"  Result length: {len(result)} chars")
    if result:
        print(f"  Preview: {result[:300]}...")
    
    # 测试聚合器
    print("\n3. Testing Knowledge Aggregator:")
    aggregator = KnowledgeSearchAggregator(enable_wikipedia=True)
    
    test_diseases = ["Pneumonia", "Spontaneous rib fracture"]
    for disease in test_diseases:
        print(f"\n  Searching: {disease}")
        text, source = aggregator.search_disease_knowledge(disease)
        print(f"  Source: {source}")
        print(f"  Length: {len(text)} chars")
