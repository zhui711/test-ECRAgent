"""
Trace Logger for System Diagnostics
透明化调试日志工具，支持人工肉眼检查 (Human Audit)

设计原则：
- 单例模式 (Singleton)：全局可访问，无需层层传递
- 非侵入性：不修改现有函数签名
- 结构化输出：Markdown 格式，便于人类快速阅读
"""
import os
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional


class TraceLogger:
    """
    单例模式的追踪日志器
    
    记录内容：
    - Phase 1: Top-5 Candidates
    - Phase 2 Search Detail: Query + Raw Snippets
    - Phase 2 Extraction: K-Node 提取结果
    - Phase 2 Reasoning: Match/Conflict 判定理由
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._case_id: str = ""
        self._logs: Dict[str, Any] = {}
        self._reset_logs()
    
    @classmethod
    def get_instance(cls) -> "TraceLogger":
        """获取单例实例"""
        return cls()
    
    def _reset_logs(self):
        """重置日志结构"""
        self._logs = {
            "case_id": "",
            "timestamp": "",
            "phase1": {
                "top_candidates": [],
                "initial_diagnosis": "",
                "p_nodes_count": 0
            },
            "phase2": {
                "kgen_searches": [],      # K-Gen 搜索记录
                "kpivot_searches": [],    # K-Pivot 搜索记录
                "extractions": [],        # K-Node 提取记录
                "reasoning": [],          # Match/Conflict 判定记录
                "batch_reasoning_raw": "" # 仅在解析失败时记录原始响应
            },
            "phase3": {
                "final_diagnosis": "",
                "status": "",
                "reasoning_path": ""
            },
            "naive_scores": {},
            "graph_summary": ""
        }
    
    def start_case(self, case_id: str):
        """
        开始新病例的追踪
        
        Args:
            case_id: 病例 ID
        """
        self._reset_logs()
        self._case_id = case_id
        self._logs["case_id"] = case_id
        self._logs["timestamp"] = datetime.now().isoformat()
        print(f"[TraceLogger] Started tracing case: {case_id}")
    
    # ==================== Phase 1 记录 ====================
    
    def log_phase1_result(
        self,
        top_candidates: List[str],
        initial_diagnosis: str,
        p_nodes_count: int
    ):
        """
        记录 Phase 1 结果
        
        Args:
            top_candidates: Top-5 候选列表
            initial_diagnosis: 初始诊断 ID
            p_nodes_count: P-Nodes 数量
        """
        self._logs["phase1"]["top_candidates"] = top_candidates
        self._logs["phase1"]["initial_diagnosis"] = initial_diagnosis
        self._logs["phase1"]["p_nodes_count"] = p_nodes_count
    
    # ==================== Phase 2 记录 ====================
    
    def log_kgen_search(
        self,
        disease_name: str,
        disease_id: str,
        query_type: str,
        source: str,
        raw_snippet: str,
        snippet_length: int
    ):
        """
        记录 K-Gen 搜索详情
        
        Args:
            disease_name: 疾病名称
            disease_id: 疾病 ID
            query_type: 搜索类型 (OpenTargets/Wikipedia/PubMed_Review/PubMed_General)
            source: 实际来源
            raw_snippet: 原始返回片段 (前 300-500 字符)
            snippet_length: 完整片段长度
        """
        search_record = {
            "disease_name": disease_name,
            "disease_id": disease_id,
            "query_type": query_type,
            "source": source,
            "raw_snippet": raw_snippet[:500] if raw_snippet else "",
            "full_length": snippet_length,
            "timestamp": datetime.now().isoformat()
        }
        self._logs["phase2"]["kgen_searches"].append(search_record)
    
    def log_kpivot_search(
        self,
        candidate_a: str,
        candidate_b: str,
        query: str,
        raw_snippets: List[str],
        total_length: int
    ):
        """
        记录 K-Pivot 搜索详情 (两两对比)
        
        Args:
            candidate_a: 候选 A 名称
            candidate_b: 候选 B 名称
            query: 搜索 Query
            raw_snippets: 原始摘要片段列表 (每个取前 300 字符)
            total_length: 所有摘要总长度
        """
        # 截取每个 snippet 的前 300 字符
        truncated_snippets = [
            s[:300] + "..." if len(s) > 300 else s 
            for s in raw_snippets[:3]  # 最多记录 3 个
        ]
        
        search_record = {
            "comparison": f"{candidate_a} vs {candidate_b}",
            "query": query,
            "raw_snippets": truncated_snippets,
            "snippet_count": len(raw_snippets),
            "total_length": total_length,
            "timestamp": datetime.now().isoformat()
        }
        self._logs["phase2"]["kpivot_searches"].append(search_record)
    
    def log_extraction(
        self,
        source_type: str,
        disease_id: str,
        k_nodes_extracted: List[Dict[str, str]],
        edges_created: int
    ):
        """
        记录 K-Node 提取结果
        
        Args:
            source_type: 来源类型 (K-Gen/K-Pivot)
            disease_id: 关联的疾病 ID
            k_nodes_extracted: 提取的 K-Nodes 列表
            edges_created: 创建的边数量
        """
        extraction_record = {
            "source_type": source_type,
            "disease_id": disease_id,
            "k_nodes": [
                {
                    "content": k.get("content", "")[:100],
                    "importance": k.get("importance", "Common")
                }
                for k in k_nodes_extracted[:10]  # 最多记录 10 个
            ],
            "k_nodes_count": len(k_nodes_extracted),
            "edges_created": edges_created
        }
        self._logs["phase2"]["extractions"].append(extraction_record)
    
    def log_reasoning(
        self,
        k_node_content: str,
        p_node_content: str,
        relation: str,
        reason: str
    ):
        """
        记录 Match/Conflict 判定理由
        
        Args:
            k_node_content: K-Node 内容
            p_node_content: P-Node 内容
            relation: 关系类型 (Match/Conflict/Void)
            reason: 判定理由
        """
        reasoning_record = {
            "k_node": k_node_content[:100] if k_node_content else "",
            "p_node": p_node_content[:100] if p_node_content else "",
            "relation": relation,
            "reason": reason[:200] if reason else ""
        }
        self._logs["phase2"]["reasoning"].append(reasoning_record)
    
    def log_batch_reasoning_raw(self, raw_response: str, parse_success: bool = False):
        """
        记录 Batch Reasoning 的原始 LLM 响应（仅在解析失败时记录）
        
        设计原则：非侵入性诊断，仅在失败时触发
        
        Args:
            raw_response: LLM 原始响应内容
            parse_success: 解析是否成功（成功时不记录，节省空间）
        """
        if parse_success:
            # 解析成功时不记录原始响应
            return
        
        # 解析失败时记录前 1500 字符（足够看到问题）
        self._logs["phase2"]["batch_reasoning_raw"] = raw_response[:1500] if raw_response else "(empty response)"
        print(f"[TraceLogger] ⚠️ Recorded failed Batch Reasoning response ({len(raw_response)} chars)")
    
    # ==================== Phase 3 记录 ====================
    
    def log_phase3_result(
        self,
        final_diagnosis: str,
        status: str,
        reasoning_path: str
    ):
        """
        记录 Phase 3 结果
        
        Args:
            final_diagnosis: 最终诊断
            status: 状态 (Confirm/Overturn/Fallback)
            reasoning_path: 推理路径
        """
        self._logs["phase3"]["final_diagnosis"] = final_diagnosis
        self._logs["phase3"]["status"] = status
        self._logs["phase3"]["reasoning_path"] = reasoning_path[:500] if reasoning_path else ""
    
    # ==================== 辅助记录 ====================
    
    def log_naive_scores(self, scores: Dict[str, float]):
        """记录确定性评分"""
        self._logs["naive_scores"] = scores
    
    def log_graph_summary(self, summary: str):
        """记录图谱摘要"""
        self._logs["graph_summary"] = summary[:2000] if summary else ""
    
    # ==================== 导出 ====================
    
    def export_to_markdown(self, output_dir: str = "output/debug_traces") -> str:
        """
        导出为 Markdown 文件
        
        Args:
            output_dir: 输出目录
        
        Returns:
            输出文件路径
        """
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建文件路径
        filename = f"{self._case_id}_debug.md"
        filepath = os.path.join(output_dir, filename)
        
        # 生成 Markdown 内容
        md_content = self._generate_markdown()
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"[TraceLogger] Exported debug trace to: {filepath}")
        return filepath
    
    def _generate_markdown(self) -> str:
        """生成 Markdown 格式的日志内容"""
        lines = []
        
        # 标题
        lines.append(f"# Debug Trace: {self._logs['case_id']}")
        lines.append(f"\n**Generated:** {self._logs['timestamp']}\n")
        lines.append("---\n")
        
        # Phase 1 摘要
        lines.append("## 📋 Phase 1: Initial Diagnosis\n")
        p1 = self._logs["phase1"]
        lines.append(f"- **Top-5 Candidates:** {', '.join(p1['top_candidates'])}")
        lines.append(f"- **Initial Diagnosis:** {p1['initial_diagnosis']}")
        lines.append(f"- **P-Nodes Count:** {p1['p_nodes_count']}\n")
        
        # Phase 2 搜索详情
        lines.append("## 🔍 Phase 2: Knowledge Search Details\n")
        
        # K-Gen 搜索
        if self._logs["phase2"]["kgen_searches"]:
            lines.append("### K-Gen Searches (General Knowledge)\n")
            lines.append("| Disease | Source | Snippet Length | Raw Snippet Preview |")
            lines.append("|---------|--------|----------------|---------------------|")
            
            for search in self._logs["phase2"]["kgen_searches"]:
                snippet_preview = search["raw_snippet"][:150].replace("\n", " ").replace("|", "\\|")
                if len(search["raw_snippet"]) > 150:
                    snippet_preview += "..."
                lines.append(
                    f"| **{search['disease_name']}** | {search['source']} | "
                    f"{search['full_length']} chars | {snippet_preview} |"
                )
            lines.append("")
        
        # K-Pivot 搜索 (关键！)
        if self._logs["phase2"]["kpivot_searches"]:
            lines.append("### 🎯 K-Pivot Searches (Differential Diagnosis) - CRITICAL\n")
            
            for i, search in enumerate(self._logs["phase2"]["kpivot_searches"], 1):
                lines.append(f"#### Comparison {i}: {search['comparison']}\n")
                lines.append(f"**Query:** `{search['query']}`\n")
                lines.append(f"**Results:** {search['snippet_count']} abstracts, {search['total_length']} total chars\n")
                
                if search["raw_snippets"]:
                    lines.append("**Raw Snippets Preview:**\n")
                    for j, snippet in enumerate(search["raw_snippets"], 1):
                        lines.append(f"```text\n[Snippet {j}]\n{snippet}\n```\n")
                lines.append("")
        
        # K-Node 提取
        if self._logs["phase2"]["extractions"]:
            lines.append("### 📦 K-Node Extractions\n")
            lines.append("| Source Type | Disease ID | K-Nodes | Edges |")
            lines.append("|-------------|------------|---------|-------|")
            
            for ext in self._logs["phase2"]["extractions"]:
                lines.append(
                    f"| {ext['source_type']} | {ext['disease_id']} | "
                    f"{ext['k_nodes_count']} | {ext['edges_created']} |"
                )
            lines.append("")
            
            # 详细 K-Nodes 列表
            lines.append("<details>\n<summary>Click to expand K-Node details</summary>\n")
            for ext in self._logs["phase2"]["extractions"]:
                lines.append(f"\n**{ext['disease_id']}:**")
                for k in ext["k_nodes"]:
                    lines.append(f"- [{k['importance']}] {k['content']}")
            lines.append("\n</details>\n")
        
        # Reasoning 记录
        if self._logs["phase2"]["reasoning"]:
            lines.append("### 🧠 Match/Conflict Reasoning\n")
            lines.append("| K-Node | P-Node | Relation | Reason |")
            lines.append("|--------|--------|----------|--------|")
            
            for r in self._logs["phase2"]["reasoning"][:20]:  # 最多显示 20 条
                k_content = r["k_node"][:50].replace("|", "\\|")
                p_content = r["p_node"][:50].replace("|", "\\|")
                reason = r["reason"][:100].replace("|", "\\|")
                relation_emoji = {"Match": "✅", "Conflict": "❌", "Void": "❓"}.get(r["relation"], "")
                lines.append(f"| {k_content} | {p_content} | {relation_emoji} {r['relation']} | {reason} |")
            lines.append("")
        
        # Batch Reasoning 原始响应（仅在解析失败时显示）
        batch_raw = self._logs["phase2"].get("batch_reasoning_raw", "")
        if batch_raw:
            lines.append("### ⚠️ Batch Reasoning Parse Failed - Raw LLM Response\n")
            lines.append("<details>")
            lines.append("<summary>Click to expand raw response (diagnostic info)</summary>\n")
            lines.append("```json")
            # 转义可能破坏 Markdown 的字符
            safe_raw = batch_raw.replace("```", "'''")
            lines.append(safe_raw)
            lines.append("```")
            lines.append("</details>\n")
        
        # Naive Scores
        if self._logs["naive_scores"]:
            lines.append("## 📊 Naive Scores (Deterministic)\n")
            lines.append("| Candidate | Score |")
            lines.append("|-----------|-------|")
            
            sorted_scores = sorted(
                self._logs["naive_scores"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for d_id, score in sorted_scores:
                lines.append(f"| {d_id} | {score:.2f} |")
            lines.append("")
        
        # Graph Summary
        if self._logs["graph_summary"]:
            lines.append("## 📝 Graph Summary (Input to Phase 3)\n")
            lines.append("```text")
            lines.append(self._logs["graph_summary"])
            lines.append("```\n")
        
        # Phase 3 结果
        lines.append("## ⚖️ Phase 3: Final Verdict\n")
        p3 = self._logs["phase3"]
        status_emoji = {"Confirm": "✅", "Overturn": "🔄", "Fallback": "⚠️"}.get(p3["status"], "")
        lines.append(f"- **Final Diagnosis:** {p3['final_diagnosis']}")
        lines.append(f"- **Status:** {status_emoji} {p3['status']}")
        lines.append(f"\n**Reasoning Path:**\n```\n{p3['reasoning_path']}\n```\n")
        
        # 结束
        lines.append("---")
        lines.append(f"*Generated by TraceLogger at {datetime.now().isoformat()}*")
        
        return "\n".join(lines)
    
    def get_logs(self) -> Dict[str, Any]:
        """获取原始日志字典（用于调试）"""
        return self._logs.copy()

