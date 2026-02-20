"""
Phase 2 Agent: Investigation & Graph Construction
实现 Graph Reasoning 的核心引擎

重构版本：Divide & Conquer + Batch Reasoning
- Step 2: Unified Investigation
  1. Canonical Retrieval: 并发获取 Open Targets + PubMed Review
  2. LLM Filtering: 并发过滤低质量文本
  3. Batch Reasoning: 一次性提取所有 K-Nodes (General + Pivot)
- Step 3: Recall - 回溯检查 Shadow Nodes

设计原则：
- 数据质量优于数量：严格隔离 Review/Guideline，拒绝 Case Report
- 全局视野：让 LLM 同时看到所有 Candidates 的知识
- 高效推理：减少 API 调用次数，使用 Batch 模式
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple
from src.graph.schema import MedicalGraph
from src.tools.search_tools import OpenTargetsTool, PubMedTool
from src.utils.api_client import LLMClient
from src.utils.json_utils import parse_json_from_text, parse_json_array_from_text, robust_parse_k_gen_response
from src.utils.trace_logger import TraceLogger
from src.utils.knowledge_utils import (
    get_synonyms_for_disease,
    build_pubmed_or_query,
    build_open_targets_search_terms,
    ContextBudgetManager,
    # 新增精确匹配相关函数
    get_all_search_terms_for_disease,
    validate_open_targets_match,
    # 三级瀑布流搜索
    PubMedQueryBuilder
)
from config.prompt_phase2_KPivot import (
    PHASE2_BATCH_SYSTEM_PROMPT,
    build_batch_prompt
)
from config.prompt_phase2_Filter import (
    PHASE2_FILTER_SYSTEM_PROMPT,
    PHASE2_FILTER_USER_PROMPT_TEMPLATE,
    parse_filter_response,
    # 新增 Batch Filter 接口
    PHASE2_BATCH_FILTER_SYSTEM_PROMPT,
    build_batch_filter_prompt,
    parse_batch_filter_response
)
from config.prompt_phase2_Recall import PHASE2_RECALL_SYSTEM_PROMPT, PHASE2_RECALL_USER_PROMPT_TEMPLATE
from Bio import Entrez
import os


class Phase2GraphReasoning:
    """
    Phase 2 Graph Reasoning Agent (Refactored)
    
    核心流程：
    1. 初始化图谱 (from Phase 1)
    2. Step 2: Unified Investigation
       - Canonical Retrieval (并发)
       - LLM Filtering (并发)
       - Batch Reasoning (单次调用)
    3. Step 3: Recall - 回溯检查 Shadow Nodes
    """
    
    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):
        """
        初始化 Phase2GraphReasoning
        
        Args:
            llm_client: LLM 客户端实例
            model_name: 模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name
        
        # 初始化搜索工具
        self.open_targets_tool = OpenTargetsTool()
        self.pubmed_tool = PubMedTool()
        
        # PubMed 配置
        self.pubmed_email = os.getenv("PUBMED_EMAIL", "research@example.com")
        Entrez.email = self.pubmed_email
        
        # 上下文预算管理
        self.context_manager = ContextBudgetManager(max_total_tokens=30000)
        
        # 记录每个疾病的知识来源
        self.knowledge_sources: Dict[str, str] = {}
        self.low_evidence_diseases: List[str] = []
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理状态，执行 Phase 2 的完整流程
        
        Args:
            state: LangGraph 状态字典
        
        Returns:
            更新后的状态字典（包含 graph_json）
        """
        # 重置状态
        self.knowledge_sources = {}
        self.low_evidence_diseases = []
        
        try:
            # 获取输入
            phase1_result = state.get("phase1_result")
            if not phase1_result:
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Phase 1 result is missing"
                }
            
            input_case = state.get("input_case", {})
            raw_text = input_case.get("narrative", "")
            if not raw_text:
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Raw text (narrative) is missing"
                }
            
            # Step 0: 初始化图谱
            print("[Phase2] Initializing graph from Phase 1 result...")
            graph = MedicalGraph.from_phase1(phase1_result, raw_text)
            
            # Step 2: Unified Investigation (新架构)
            print("[Phase2] Step 2: Unified Investigation (Divide & Conquer)...")
            self._step2_unified_investigation(graph)
            
            # Step 3: Batch Recall - Shadow Handler
            print("[Phase2] Step 3: Recall (Backtracking Shadow Nodes)...")
            self._step3_recall(graph, raw_text)
            
            # 序列化图谱
            graph_json = graph.to_dict()
            
            # 添加知识来源和低证据疾病信息
            graph_json["knowledge_sources"] = self.knowledge_sources
            if self.low_evidence_diseases:
                graph_json["low_evidence_diseases"] = self.low_evidence_diseases
                print(f"[Phase2] WARNING: {len(self.low_evidence_diseases)} diseases have low evidence: "
                      f"{self.low_evidence_diseases}")
            
            print(f"[Phase2] Graph construction complete. "
                  f"P-Nodes: {len(graph.get_p_nodes())}, "
                  f"K-Nodes: {len(graph.get_k_nodes())}, "
                  f"D-Nodes: {len(graph.get_d_nodes())}")
            
            return {
                **state,
                "graph_json": graph_json,
                "status": "processing"
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Phase 2 error: {str(e)}\n{traceback.format_exc()}"
            print(f"[Phase2] {error_msg}")
            return {
                **state,
                "status": "failed",
                "error_log": error_msg
            }
    
    def _step2_unified_investigation(self, graph: MedicalGraph):
        """
        Step 2: Unified Investigation (Divide & Conquer + Batch Reasoning)
        
        重构版本 (Debug Enhancement):
        - 实现 Open Targets 精确匹配 (Case-insensitive Substring Match)
        - 同义词穷尽重试逻辑
        - Batch Filter (单次 LLM 调用)
        - 强制 temperature=0.0
        - 记录原始 API 响应到 TraceLogger
        
        流程：
        1. Canonical Retrieval: 并发获取每个 Candidate 的知识（带精确匹配验证）
        2. LLM Filtering: **单次 Batch 调用** 过滤低质量文本
        3. Batch Reasoning: 一次性调用 LLM 提取所有 K-Nodes (temperature=0.0)
        """
        d_nodes = graph.get_d_nodes()
        p_nodes = graph.get_p_nodes()
        
        if not d_nodes:
            print("[Phase2 Step2] No D-Nodes found, skipping investigation...")
            return
        
        print(f"[Phase2 Step2] Processing {len(d_nodes)} candidates...")
        
        # ==================== Phase 2.1: Canonical Retrieval (with Exact Match) ====================
        print("[Phase2 Step2.1] Canonical Retrieval (Open Targets + PubMed Review)...")
        print("[Phase2 Step2.1] Using Exact Match validation + Synonym exhaustion...")
        
        open_targets_results: Dict[str, str] = {}
        pubmed_results: Dict[str, str] = {}
        raw_api_responses: Dict[str, Any] = {}  # 存储原始响应用于 Debug
        
        # 定义单个 Candidate 的检索任务（带精确匹配验证 + 三级瀑布流）
        def retrieve_knowledge_with_validation(d_node: Dict) -> Tuple[str, str, str, str, Any]:
            """
            检索单个 Candidate 的知识
            
            策略：
            - Open Targets: 同义词穷尽重试 + 精确匹配验证
            - PubMed: 三级瀑布流 (StatPearls -> Review -> Clinical)
            """
            d_id = d_node["id"]
            d_name = d_node["name"]
            
            ot_text = ""
            pm_text = ""
            source = "None"
            raw_response = None
            pm_level = "None"
            
            try:
                # 1. Open Targets (使用同义词穷尽重试 + 精确匹配验证)
                search_terms = get_all_search_terms_for_disease(d_id, d_name)
                print(f"[Phase2 Step2.1] {d_name}: Trying {len(search_terms)} search terms...")
                
                # 使用新的带验证方法
                ot_result, raw_response = self.open_targets_tool.get_disease_description_with_raw(
                    d_name, 
                    search_terms=search_terms
                )
                
                if ot_result and len(ot_result) >= 100:
                    ot_text = ot_result
                    source = "OpenTargets"
                else:
                    print(f"[Phase2 Step2.1] {d_name}: Open Targets match FAILED or no data")
                
                # 2. PubMed 三级瀑布流 (StatPearls -> Review -> Clinical)
                time.sleep(0.1)  # 速率控制
                pm_abstracts, pm_level = self._search_pubmed_waterfall(d_name, d_id)
                
                if pm_abstracts:
                    pm_text = "\n\n---\n\n".join(pm_abstracts)
                    if not source or source == "None":
                        source = f"PubMed_{pm_level}"
                    else:
                        source = f"{source}+PubMed_{pm_level}"
                
            except Exception as e:
                print(f"[Phase2 Step2.1] Error retrieving {d_name}: {e}")
            
            return d_id, ot_text, pm_text, source, raw_response
        
        # 并发执行检索 (max_workers=4)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(retrieve_knowledge_with_validation, d): d for d in d_nodes}
            
            for future in as_completed(futures):
                try:
                    d_id, ot_text, pm_text, source, raw_response = future.result()
                    d_name = futures[future]["name"]
                    
                    open_targets_results[d_id] = ot_text
                    pubmed_results[d_id] = pm_text
                    raw_api_responses[d_id] = raw_response
                    self.knowledge_sources[d_id] = source
                    
                    # 记录到 TraceLogger
                    TraceLogger.get_instance().log_kgen_search(
                        disease_name=d_name,
                        disease_id=d_id,
                        query_type="UnifiedRetrieval",
                        source=source,
                        raw_snippet=(ot_text + pm_text)[:500],
                        snippet_length=len(ot_text) + len(pm_text)
                    )
                    
                    status = "✓" if source != "None" else "✗"
                    print(f"[Phase2 Step2.1] {status} {d_name}: {source}")
                    
                except Exception as e:
                    print(f"[Phase2 Step2.1] Task error: {e}")
        
        # ==================== Phase 2.2: Batch LLM Filtering (Single Call) ====================
        print("[Phase2 Step2.2] Batch LLM Filtering (single call, removing low-quality content)...")
        
        filtered_pubmed: Dict[str, str] = {}
        
        # 收集需要过滤的文本
        filter_snippets = []
        filter_ids = []
        d_id_to_name = {d["id"]: d["name"] for d in d_nodes}
        
        for d_id, pm_text in pubmed_results.items():
            if pm_text and len(pm_text) >= 100:
                filter_snippets.append({
                    "disease": d_id_to_name.get(d_id, d_id),
                    "text": pm_text
                })
                filter_ids.append(d_id)
        
        if filter_snippets:
            print(f"[Phase2 Step2.2] Batch filtering {len(filter_snippets)} snippets...")
            
            # 构建 Batch Filter Prompt
            batch_filter_prompt = build_batch_filter_prompt(filter_snippets)
            
            messages = [
                {"role": "system", "content": PHASE2_BATCH_FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": batch_filter_prompt}
            ]
            
            # 单次 LLM 调用 (temperature=0.0)
            result = self.llm_client.generate_json(
                messages=messages,
                model=self.model_name,
                logprobs=False,
                temperature=0.0,  # 强制确定性输出
                max_tokens=256   # 足够返回 JSON 数组
            )
            
            if result["error"]:
                print(f"[Phase2 Step2.2] Batch Filter LLM error: {result['error']}")
                print("[Phase2 Step2.2] Defaulting to DROP ALL (宁缺毋滥)")
                # 错误时丢弃所有
                filter_results = [False] * len(filter_snippets)
            else:
                # 解析批量结果
                filter_results = parse_batch_filter_response(
                    result["content"], 
                    expected_count=len(filter_snippets)
                )
                print(f"[Phase2 Step2.2] Filter results: {filter_results}")
            
            # 应用过滤结果
            for i, (d_id, keep) in enumerate(zip(filter_ids, filter_results)):
                if keep:
                    filtered_pubmed[d_id] = pubmed_results[d_id]
                    print(f"[Phase2 Step2.2] {d_id}: KEEP ✓")
                else:
                    filtered_pubmed[d_id] = ""
                    print(f"[Phase2 Step2.2] {d_id}: DROP ✗")
        else:
            print("[Phase2 Step2.2] No PubMed texts to filter")
        
        # 合并未过滤的空结果
        for d_id in pubmed_results:
            if d_id not in filtered_pubmed:
                filtered_pubmed[d_id] = ""
        
        # 标记低证据疾病
        for d_node in d_nodes:
            d_id = d_node["id"]
            if not open_targets_results.get(d_id) and not filtered_pubmed.get(d_id):
                self.low_evidence_diseases.append(d_node["name"])
        
        # ==================== Phase 2.3: Context Budget Allocation ====================
        print("[Phase2 Step2.3] Allocating context budget...")
        
        knowledge_map = self.context_manager.allocate_budget(
            candidates=d_nodes,
            open_targets_texts=open_targets_results,
            pubmed_texts=filtered_pubmed
        )
        
        # ==================== Phase 2.4: Batch Reasoning (temperature=0.0) ====================
        print("[Phase2 Step2.4] Batch Reasoning (single LLM call, temperature=0.0)...")
        
        # 构建 Batch Prompt
        user_prompt = build_batch_prompt(
            candidates=d_nodes,
            p_nodes=p_nodes,
            knowledge_map=knowledge_map
        )
        
        messages = [
            {"role": "system", "content": PHASE2_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # 单次 LLM 调用 (temperature=0.0 - 强制确定性输出)
        result = self.llm_client.generate_json(
            messages=messages,
            model=self.model_name,
            logprobs=False,
            temperature=0.0,
            max_tokens=8192
        )
        
        if result["error"]:
            print(f"[Phase2 Step2.4] LLM error: {result['error']}")
            # 记录 LLM 错误到 TraceLogger
            TraceLogger.get_instance().log_batch_reasoning_raw(
                f"LLM Error: {result['error']}", 
                parse_success=False
            )
            return
        
        response_content = result["content"]
        print(f"[Phase2 Step2.4] LLM response length: {len(response_content)} chars")
        
        # 解析 JSON
        parsed_json = robust_parse_k_gen_response(response_content, verbose=True)
        
        if not parsed_json.get("k_nodes"):
            print("[Phase2 Step2.4] No K-Nodes extracted from Batch Reasoning")
            # 诊断日志：记录原始响应以便后续分析
            TraceLogger.get_instance().log_batch_reasoning_raw(
                response_content, 
                parse_success=False
            )
            return
        
        # ==================== Phase 2.5: Graph Update ====================
        print("[Phase2 Step2.5] Updating graph with extracted K-Nodes...")
        
        edges_created = self._process_batch_response(graph, parsed_json, p_nodes, d_nodes)
        
        # 记录到 TraceLogger - 包含更详细的 K-Node 信息
        k_nodes_for_log = []
        for k in parsed_json.get("k_nodes", []):
            k_nodes_for_log.append({
                "content": k.get("content", "")[:100],
                "importance": k.get("importance", "Common"),
                "supported": k.get("supported_candidates", []),
                "ruled_out": k.get("ruled_out_candidates", [])
            })
        
        TraceLogger.get_instance().log_extraction(
            source_type="BatchReasoning",
            disease_id="ALL_CANDIDATES",
            k_nodes_extracted=k_nodes_for_log,
            edges_created=edges_created
        )
        
        print(f"[Phase2 Step2.5] Created {edges_created} edges")
    
    def _search_pubmed_waterfall(
        self, 
        disease_name: str, 
        disease_id: str
    ) -> Tuple[List[str], str]:
        """
        三级瀑布流 PubMed 搜索
        
        搜索顺序：
        1. Level 1 (Best): StatPearls 医学百科
        2. Level 2 (Good): Reviews/Guidelines 综述文章
        3. Level 3 (Fallback): Clinical Articles（标题级关键词过滤）
        
        判断标准：len(snippets) > 0 即停止，不进入下一级
        
        Args:
            disease_name: 疾病名称
            disease_id: 疾病 ID
        
        Returns:
            (摘要列表, 来源级别) 元组
        """
        query_builder = PubMedQueryBuilder(disease_name, disease_id)
        
        # 获取所有级别的查询
        queries = query_builder.get_all_queries()
        
        for level_name, query, max_results in queries:
            print(f"[Phase2 PubMed] Trying Level: {level_name}...")
            
            try:
                abstracts = self._execute_pubmed_search(query, max_results)
                
                # 判断标准：有结果即停止
                if abstracts and len(abstracts) > 0:
                    print(f"[Phase2 PubMed] ✓ Found {len(abstracts)} results at {level_name}")
                    return abstracts, level_name
                else:
                    print(f"[Phase2 PubMed] ✗ No results at {level_name}, trying next level...")
                    
            except Exception as e:
                print(f"[Phase2 PubMed] Error at {level_name}: {e}")
                continue
            
        # 所有级别都失败
        print(f"[Phase2 PubMed] ✗ All levels exhausted for '{disease_name}'")
        return [], "None"
    
    def _execute_pubmed_search(self, query: str, max_results: int) -> List[str]:
        """
        执行单次 PubMed 搜索
        
        Args:
            query: PubMed 查询字符串
            max_results: 最大结果数
        
        Returns:
            摘要列表
        """
        try:
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results.get("IdList", [])
            
            if not id_list:
                return []
            
            # 获取摘要
            abstracts = []
            for pmid in id_list[:max_results + 3]:
                abstract = self._fetch_pubmed_abstract(pmid)
                if abstract and len(abstract) >= 100:
                    abstracts.append(abstract)
                    if len(abstracts) >= max_results:
                        break
            
            return abstracts
            
        except Exception as e:
            print(f"[Phase2] PubMed search error: {e}")
            return []
    
    def _search_pubmed_review_only(self, query: str, max_results: int = 3) -> List[str]:
        """
        搜索 PubMed，严格只返回 Review/Guideline（向后兼容方法）
        
        Args:
            query: PubMed 查询字符串（已包含 Review/Guideline 过滤）
            max_results: 最大结果数
        
        Returns:
            摘要列表
        """
        return self._execute_pubmed_search(query, max_results)
    
    def _fetch_pubmed_abstract(self, pmid: str) -> str:
        """获取 PubMed 摘要"""
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="abstract",
                retmode="xml"
            )
            fetch_results = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            articles = fetch_results.get("PubmedArticle", [])
            if not articles:
                return ""
            
            article = articles[0]
            
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
            print(f"[Phase2] Error fetching PMID {pmid}: {e}")
            return ""
    
    def _process_batch_response(
        self,
        graph: MedicalGraph,
        parsed_json: Dict[str, Any],
        p_nodes: List[Dict[str, Any]],
        d_nodes: List[Dict[str, Any]]
    ) -> int:
        """
        处理 Batch Reasoning 的 LLM 响应，更新图谱
        
        重构版本：适配扁平化嵌套 Schema
        - 弃用 edges 数组遍历
        - 直接从 k_nodes 内部读取 supported_candidates / ruled_out_candidates
        
        Args:
            graph: MedicalGraph 实例
            parsed_json: LLM 响应解析结果（新格式）
            p_nodes: 当前 P-Nodes 列表
            d_nodes: D-Nodes 列表
        
        Returns:
            创建的边数量
        """
        k_nodes_data = parsed_json.get("k_nodes", [])
        edges_created = 0
        kd_edges_created = 0
        pk_edges_created = 0
        
        # 构建 D-Node ID 集合（用于验证）
        valid_d_ids = {d["id"] for d in d_nodes}
        d_name_to_id = {d["name"].lower(): d["id"] for d in d_nodes}
        
        # 存储创建的 K-Node IDs（用于后续 P-K 边处理）
        created_k_ids = []
        
        print(f"[Phase2] Processing {len(k_nodes_data)} K-Nodes from Batch Reasoning...")
        
        # 遍历 K-Nodes，同时处理 K-D 边
        for idx, k_item in enumerate(k_nodes_data):
            k_content = k_item.get("content", "")
            k_type = k_item.get("type", "General")  # 新字段名：type (不是 k_type)
            importance = k_item.get("importance", "Common")
            
            # 静默跳过空内容
            if not k_content:
                continue
            
            # 1. 创建 K-Node
            k_id = graph.add_k_node(
                node_id=None,
                content=k_content,
                k_type=k_type,
                source="BatchReasoning",
                importance=importance
            )
            created_k_ids.append(k_id)
            
            # 2. 处理 Support 边（supported_candidates）
            supported = k_item.get("supported_candidates", [])
            for target_id in supported:
                resolved_id = self._resolve_d_node_id(target_id, valid_d_ids, d_name_to_id)
                if resolved_id:
                    graph.add_kd_edge(k_id, resolved_id, relation="Support", strength=importance)
                    kd_edges_created += 1
                    print(f"[Phase2] K-D Edge: {k_id} --Support--> {resolved_id}")
                else:
                    print(f"[Phase2] ⚠️ Failed to resolve Support target: '{target_id}'")
            
            # 3. 处理 Rule_Out 边（ruled_out_candidates）
            ruled_out = k_item.get("ruled_out_candidates", [])
            for target_id in ruled_out:
                resolved_id = self._resolve_d_node_id(target_id, valid_d_ids, d_name_to_id)
                if resolved_id:
                    graph.add_kd_edge(k_id, resolved_id, relation="Rule_Out", strength=importance)
                    kd_edges_created += 1
                    print(f"[Phase2] K-D Edge: {k_id} --Rule_Out--> {resolved_id}")
                else:
                    print(f"[Phase2] ⚠️ Failed to resolve Rule_Out target: '{target_id}'")
        
        print(f"[Phase2] Created {kd_edges_created} K-D edges")
        
        # 4. 为 K-Nodes 建立 P-K 边
        for k_id in created_k_ids:
            k_node = graph.get_k_node_by_id(k_id)
            if not k_node:
                continue
            
            k_content = k_node.get("content", "")
            matched_p = graph.find_p_node_by_content(k_content)
            
            if matched_p:
                p_status = matched_p.get("status", "Present")
                relation = "Match" if p_status == "Present" else "Conflict"
                graph.add_pk_edge(matched_p["id"], k_id, relation=relation)
                pk_edges_created += 1
                
                # 记录 Reasoning
                TraceLogger.get_instance().log_reasoning(
                    k_node_content=k_content,
                    p_node_content=matched_p.get("content", ""),
                    relation=relation,
                    reason=f"P-Node status: {p_status}, from BatchReasoning"
                )
            else:
                # 创建 Shadow Node
                shadow_p_id = graph.create_shadow_p_node(k_content)
                graph.add_pk_edge(shadow_p_id, k_id, relation="Void")
                pk_edges_created += 1
        
        print(f"[Phase2] Created {pk_edges_created} P-K edges")
        edges_created = kd_edges_created + pk_edges_created
        
        return edges_created
    
    def _resolve_d_node_id(
        self,
        target: str,
        valid_d_ids: set,
        d_name_to_id: Dict[str, str]
    ) -> Optional[str]:
        """
        解析目标 D-Node ID
        
        Args:
            target: LLM 返回的目标标识（可能是 ID 或名称）
            valid_d_ids: 有效的 D-Node ID 集合
            d_name_to_id: 名称到 ID 的映射
        
        Returns:
            解析后的 D-Node ID，无法解析返回 None
        """
        if not target:
            return None
        
        # 直接匹配 ID
        if target in valid_d_ids:
            return target
        
        # 添加 d_ 前缀尝试
        if f"d_{target}" in valid_d_ids:
            return f"d_{target}"
        
        # 尝试名称匹配
        target_lower = target.lower()
        for name, d_id in d_name_to_id.items():
            if name in target_lower or target_lower in name:
                return d_id
        
        return None
    
    def _step3_recall(self, graph: MedicalGraph, raw_text: str):
        """
        Step 3: Batch Recall (Shadow Handler)
        
        批量处理所有 Void 边对应的 K-Nodes
        """
        void_k_nodes = graph.get_void_k_nodes()
        
        if not void_k_nodes:
            print("[Phase2 Step3] No Shadow Nodes to recall")
            return
        
        print(f"[Phase2 Step3] Found {len(void_k_nodes)} Shadow Nodes to recall")
        
        try:
            # 打包所有 Missing K-Nodes
            missing_k_list = [
                {
                    "k_node_id": k_node["id"],
                    "content": k_node.get("content", ""),
                    "shadow_p_id": k_node.get("shadow_p_id", "")
                }
                for k_node in void_k_nodes
            ]
            
            if not missing_k_list:
                return
            
            # LLM Call: 回溯检查
            missing_k_json = json.dumps(missing_k_list, ensure_ascii=False, indent=2)
            user_prompt = PHASE2_RECALL_USER_PROMPT_TEMPLATE.format(
                raw_text=raw_text,
                missing_k_list=missing_k_json
            )
            
            messages = [
                {"role": "system", "content": PHASE2_RECALL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            result = self.llm_client.generate_json(
                messages=messages,
                model=self.model_name,
                logprobs=False,
                temperature=0.2,
                max_tokens=2048
            )
            
            if result["error"]:
                print(f"[Phase2 Step3] LLM error: {result['error']}")
                return
            
            # Parse and Update Graph
            response_content = result["content"]
            print(f"[Phase2 Step3] LLM response length: {len(response_content)} chars")
            
            updates = parse_json_array_from_text(response_content, verbose=True)
            
            if not updates or not isinstance(updates, list):
                print("[Phase2 Step3] No valid updates from Recall")
                return
            
            # 更新 P-Nodes 和边
            for update in updates:
                k_node_id = update.get("k_node_id")
                new_status = update.get("new_status")
                
                if not k_node_id or new_status not in ["Present", "Absent"]:
                    continue
                
                # 查找对应的 Void K-Node 和 Shadow P-Node
                for void_k in void_k_nodes:
                    if void_k["id"] == k_node_id:
                        shadow_p_id = void_k.get("shadow_p_id", "")
                        
                        if shadow_p_id and graph.graph.has_node(shadow_p_id):
                            # 更新 P-Node 状态
                            graph.update_p_node_status(
                                shadow_p_id, 
                                new_status, 
                                source="Phase2_Recall"
                            )
                            
                            # 更新边关系
                            graph.remove_edge(shadow_p_id, k_node_id)
                            relation = "Match" if new_status == "Present" else "Conflict"
                            graph.add_pk_edge(shadow_p_id, k_node_id, relation=relation)
                            
                            print(f"[Phase2 Step3] Updated Shadow Node: {shadow_p_id} -> {new_status}")
                        
                        break
        
        except Exception as e:
            print(f"[Phase2 Step3] Error in recall: {e}")
            import traceback
            traceback.print_exc()
