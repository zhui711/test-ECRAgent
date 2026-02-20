"""
# ============================================================================
# DEPRECATED: 此模块已被 phase3_memory_judge.py 替代
# ============================================================================
#
# 请使用 src.agents.phase3_memory_judge.MemoryAugmentedJudge 作为 Phase 3 的唯一实现。
#
# 保留此文件的原因：
# 1. MemoryAugmentedJudge 继承了 JudgeAgent（需要基类）
# 2. 兼容旧代码的 import 语句
#
# 不建议直接使用此模块中的 JudgeAgent 类进行新开发。
# ============================================================================

Phase 3 Agent: Judge & Tiered Logic (LEGACY)
实现元认知审判，基于 Tiered Logic 输出最终诊断

根据算法流程图 Phase 3 (Line 40-56):
- Round 1: Safety Sentinel (Clinical Importance) - Fatal Conflict 检查
- Round 2: Shadow Interrogation (Bias Exclusion) - 确认偏差校正
- Round 3: Final Verdict - 最终裁决

注意: 此实现中的 _pre_audit 和 Risk Level 逻辑已在新版中被移除。
"""
import json
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.utils.api_client import LLMClient
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.utils.json_utils import parse_json_from_text
from config.prompt_phase3_judge import PHASE3_JUDGE_SYSTEM_PROMPT, PHASE3_JUDGE_USER_PROMPT_TEMPLATE


class JudgeAgent:
    """
    Phase 3 Judge Agent
    负责最终诊断决策，应用 Tiered Logic
    
    核心流程（基于算法流程图 Line 40-56）：
    1. Round 1: Safety Sentinel - 检查 High Risk 疾病的 Fatal Conflict
    2. Round 2: Shadow Interrogation - 计算 Shadow Ratio，校正确认偏差
    3. Round 3: Final Verdict - 基于 Explanatory Coverage 选择最终诊断
    """
    
    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):
        """
        初始化 JudgeAgent
        
        Args:
            llm_client: LLM 客户端实例
            model_name: 模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.risk_map = self._load_risk_map()
        
        # Shadow Ratio 阈值 (Line 50)
        self.shadow_threshold = 0.5
    
    def _load_risk_map(self) -> Dict[str, Dict[str, str]]:
        """
        加载疾病风险地图
        
        根据算法流程图 Line 41:
        RiskMap <- LoadMetadata(D_all)  # Static High/Low Risk Tags
        
        Returns:
            字典：{disease_id: {"name": str, "risk": str}}
        """
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "disease_metadata.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                risk_map = {}
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 匹配格式: "01": { "name": "...", "risk": "High" }
                    match = re.match(r'"(\d+)":\s*\{\s*"name":\s*"([^"]+)",\s*"risk":\s*"([^"]+)"\s*\}', line)
                    if match:
                        disease_id, name, risk = match.groups()
                        risk_map[disease_id] = {"name": name, "risk": risk}
                
                return risk_map
        except Exception as e:
            print(f"[JudgeAgent] Error loading risk map: {e}")
            return {}
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理状态，执行 Phase 3 的最终判决
        
        根据算法流程图 Line 40-56
        
        Args:
            state: LangGraph 状态字典
        
        Returns:
            更新后的状态字典（包含 final_output）
        """
        try:
            # 获取输入
            graph_json = state.get("graph_json")
            if not graph_json:
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Graph JSON is missing"
                }
            
            phase1_result = state.get("phase1_result", {})
            input_case = state.get("input_case", {})
            raw_text = input_case.get("narrative", "")
            
            # 获取 graph_summary (新增)
            graph_summary = state.get("graph_summary", "")
            
            # 获取 naive_scores (新增)
            naive_scores = state.get("naive_scores", {})
            
            # 预处理：执行 Python 层面的审计逻辑
            audit_results = self._pre_audit(graph_json, phase1_result)
            
            # 构建 Prompt (使用 graph_summary 和 naive_scores)
            user_prompt = self._construct_judge_prompt(
                graph_json,
                phase1_result,
                raw_text,
                audit_results,
                graph_summary=graph_summary,
                naive_scores=naive_scores  # 新增参数
            )
            
            # 调用 LLM
            messages = [
                {"role": "system", "content": PHASE3_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            result = self.llm_client.generate_json(
                messages=messages,
                model=self.model_name,
                logprobs=False,
                temperature=0.2,  # 使用 0.2 以获得更稳定的输出
                max_tokens=4096
            )
            
            if result["error"]:
                print(f"[JudgeAgent] LLM error: {result['error']}")
                return {
                    **state,
                    "status": "failed",
                    "error_log": f"LLM error: {result['error']}"
                }
            
            # 解析输出
            response_content = result["content"]
            print(f"[JudgeAgent] LLM response length: {len(response_content)} chars")
            
            parsed_json = parse_json_from_text(response_content, verbose=True)
            
            if not parsed_json:
                print(f"[JudgeAgent] Failed to parse JSON")
                print(f"[JudgeAgent] Raw response preview: {response_content[:500]}...")
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Failed to parse JSON from LLM response"
                }
            
            # 验证并修正诊断 ID
            final_output = self._validate_and_fix_output(parsed_json, phase1_result, audit_results)
            
            print(f"[JudgeAgent] Final diagnosis: {final_output.get('final_diagnosis_id')} - "
                  f"{final_output.get('final_diagnosis_name')} ({final_output.get('status')})")
            
            return {
                **state,
                "final_output": final_output,
                "status": "success"
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Phase 3 error: {str(e)}\n{traceback.format_exc()}"
            print(f"[JudgeAgent] {error_msg}")
            return {
                **state,
                "status": "failed",
                "error_log": error_msg
            }
    
    def _pre_audit(
        self, 
        graph_json: Dict[str, Any], 
        phase1_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        预审计：在调用 LLM 之前执行 Python 层面的逻辑检查
        
        根据算法流程图 Line 42-52
        """
        audit_results = {
            "disqualified_candidates": [],
            "safeguarded_candidates": [],
            "shadow_penalties": {},
            "coverage_scores": {}
        }
        
        d_nodes = graph_json.get("graph", {}).get("nodes", {}).get("d_nodes", [])
        k_nodes = graph_json.get("graph", {}).get("nodes", {}).get("k_nodes", [])
        p_nodes = graph_json.get("graph", {}).get("nodes", {}).get("p_nodes", [])
        p_k_links = graph_json.get("graph", {}).get("edges", {}).get("p_k_links", [])
        k_d_links = graph_json.get("graph", {}).get("edges", {}).get("k_d_links", [])
        
        # 构建快速查找索引
        k_node_map = {k["id"]: k for k in k_nodes}
        p_node_map = {p["id"]: p for p in p_nodes}
        
        # Round 1: Safety Sentinel (Line 42-47)
        for d_node in d_nodes:
            d_id = d_node["id"]
            original_id = d_node.get("original_id", "")
            risk_info = self.risk_map.get(original_id, {})
            risk_level = risk_info.get("risk", "Low")
            
            # 检查 Fatal Conflict
            has_fatal_conflict = self._check_fatal_conflict(
                d_id, k_d_links, p_k_links, k_node_map, p_node_map
            )
            
            if has_fatal_conflict:
                audit_results["disqualified_candidates"].append({
                    "d_id": d_id,
                    "reason": "Fatal Conflict (Essential symptom absent)"
                })
            elif risk_level == "High":
                # Line 44-45: 高风险疾病没有 Fatal Conflict，设置 safeguard
                audit_results["safeguarded_candidates"].append({
                    "d_id": d_id,
                    "reason": "High Risk disease without Fatal Conflict"
                })
        
        # Round 2: Shadow Interrogation (Line 48-52)
        initial_d_id = f"d_{phase1_result.get('final_diagnosis_id', phase1_result.get('top_candidates', [''])[0])}"
        
        for d_node in d_nodes:
            d_id = d_node["id"]
            
            # 计算 Shadow Ratio (Line 49)
            shadow_count, total_count = self._count_shadows(
                d_id, k_d_links, p_k_links
            )
            
            if total_count > 0:
                shadow_ratio = shadow_count / total_count
                audit_results["shadow_penalties"][d_id] = {
                    "shadow_count": shadow_count,
                    "total_count": total_count,
                    "ratio": shadow_ratio
                }
                
                # Line 50-51: 如果是初始诊断且 Shadow Ratio 过高，应用惩罚
                if d_id == initial_d_id and shadow_ratio > self.shadow_threshold:
                    audit_results["shadow_penalties"][d_id]["penalty_applied"] = True
            
            # Round 3: 计算 Explanatory Coverage (Line 54-55)
            coverage = self._calculate_coverage(d_id, k_d_links, p_k_links)
            audit_results["coverage_scores"][d_id] = coverage
        
        return audit_results
    
    def _check_fatal_conflict(
        self,
        d_id: str,
        k_d_links: List[Dict],
        p_k_links: List[Dict],
        k_node_map: Dict,
        p_node_map: Dict
    ) -> bool:
        """
        检查 D-Node 是否有 Fatal Conflict
        
        根据算法流程图 Line 22-24 和 Phase 3 Tier 1:
        如果 K.importance == "Essential" 且 P-K.relation == "Conflict"
        """
        # 找到与该 D-Node 关联的所有 K-Nodes
        related_k_ids = [
            link["source"] for link in k_d_links 
            if link["target"] == d_id
        ]
        
        for k_id in related_k_ids:
            k_node = k_node_map.get(k_id, {})
            importance = k_node.get("importance", "Common")
            
            # 只检查 Essential 或 Pathognomonic
            if importance not in ["Essential", "Pathognomonic"]:
                continue
            
            # 检查是否有 Conflict 边
            for pk_link in p_k_links:
                if pk_link["target"] == k_id and pk_link["relation"] == "Conflict":
                    return True
        
        return False
    
    def _count_shadows(
        self,
        d_id: str,
        k_d_links: List[Dict],
        p_k_links: List[Dict]
    ) -> tuple:
        """
        计算与 D-Node 关联的 Shadow Nodes 数量
        
        Returns:
            (shadow_count, total_count)
        """
        # 找到与该 D-Node 关联的所有 K-Nodes
        related_k_ids = [
            link["source"] for link in k_d_links 
            if link["target"] == d_id
        ]
        
        shadow_count = 0
        total_count = len(related_k_ids)
        
        for k_id in related_k_ids:
            # 检查该 K-Node 是否有 Void 边
            for pk_link in p_k_links:
                if pk_link["target"] == k_id and pk_link["relation"] == "Void":
                    shadow_count += 1
                    break
        
        return shadow_count, total_count
    
    def _calculate_coverage(
        self,
        d_id: str,
        k_d_links: List[Dict],
        p_k_links: List[Dict]
    ) -> float:
        """
        计算 Explanatory Coverage
        
        根据算法流程图 Line 54-55:
        d* <- argmax_{d∈D_active} (ExplanatoryCoverage(d, P))
        
        公式: Match 数量 - Void 数量 * 0.1 - Conflict 数量 * 1.5
        """
        related_k_ids = [
            link["source"] for link in k_d_links 
            if link["target"] == d_id
        ]
        
        match_count = 0
        void_count = 0
        conflict_count = 0
        
        for k_id in related_k_ids:
            for pk_link in p_k_links:
                if pk_link["target"] == k_id:
                    relation = pk_link["relation"]
                    if relation == "Match":
                        match_count += 1
                    elif relation == "Void":
                        void_count += 1
                    elif relation == "Conflict":
                        conflict_count += 1
        
        coverage = match_count - (void_count * 0.1) - (conflict_count * 1.5)
        return coverage
    
    def _construct_judge_prompt(
        self,
        graph_json: Dict[str, Any],
        phase1_result: Dict[str, Any],
        raw_text: str,
        audit_results: Dict[str, Any],
        graph_summary: str = "",
        naive_scores: Dict[str, float] = None
    ) -> str:
        """
        构建 Judge Prompt
        
        修订：使用 graph_summary 替换 graph_edges_json (消融实验)
        保留 p_nodes_json 用于覆盖率计算
        新增：注入 naive_scores 作为量化参考
        """
        # 提取 P-Nodes (保留)
        p_nodes = graph_json.get("graph", {}).get("nodes", {}).get("p_nodes", [])
        p_nodes_json = json.dumps(p_nodes, ensure_ascii=False, indent=2)
        
        # 提取 D-Nodes 用于 Low Evidence 处理
        d_nodes = graph_json.get("graph", {}).get("nodes", {}).get("d_nodes", [])
        
        # 获取 Low Evidence 疾病列表
        low_evidence_diseases = graph_json.get("low_evidence_diseases", [])
        knowledge_sources = graph_json.get("knowledge_sources", {})
        
        # 添加 Risk Level 和 Evidence Status 到 D-Nodes
        for d_node in d_nodes:
            original_id = d_node.get("original_id", "")
            risk_info = self.risk_map.get(original_id, {})
            d_node["risk_level"] = risk_info.get("risk", "Low")
            
            # 添加知识来源信息
            d_id = d_node.get("id", "")
            d_node["knowledge_source"] = knowledge_sources.get(d_id, "Unknown")
            
            # 标记 Low Evidence
            disease_name = d_node.get("name", "")
            if disease_name in low_evidence_diseases:
                d_node["low_evidence"] = True
            else:
                d_node["low_evidence"] = False
        
        # 构建 Pre-Audit 信息 (保留关键审计结果)
        pre_audit_info = {
            "disqualified_candidates": audit_results.get("disqualified_candidates", []),
            "safeguarded_candidates": audit_results.get("safeguarded_candidates", []),
                "coverage_scores": audit_results.get("coverage_scores", {})
        }
        pre_audit_json = json.dumps(pre_audit_info, ensure_ascii=False, indent=2)
        
        # 构建 Low Evidence 警告
        low_evidence_warning = ""
        if low_evidence_diseases:
            low_evidence_warning = (
                "### ⚠️ LOW EVIDENCE WARNING\n"
                f"The following diseases have limited knowledge (search returned no results): {', '.join(low_evidence_diseases)}\n"
                "**Do NOT automatically disqualify these diseases!** Consider them based on:\n"
                "1. Existing Match edges (if any)\n"
                "2. Absence of Fatal Conflicts\n"
                "3. Clinical reasoning from the patient narrative\n"
            )
        
        # 提取初始候选
        initial_candidates = phase1_result.get("top_candidates", [])
        initial_candidates_list = ", ".join([
            f"{DIAGNOSIS_ID_MAP.get(cid, 'Unknown')} (ID: {cid})"
            for cid in initial_candidates
        ])
        
        initial_reasoning = phase1_result.get("differential_reasoning", "")
        
        # 使用 graph_summary 替换 graph_edges_json (消融实验)
        if graph_summary:
            # 新格式：使用自然语言摘要
            graph_evidence_section = f"""
### GRAPH EVIDENCE SUMMARY (Structured Natural Language)

{graph_summary}

### PRE-AUDIT RESULTS (System Computed)

```json
{pre_audit_json}
```
"""
        else:
            # 降级：如果没有 summary，使用原始 JSON (兼容性)
            k_nodes = graph_json.get("graph", {}).get("nodes", {}).get("k_nodes", [])
            edges = graph_json.get("graph", {}).get("edges", {})
            
            graph_edges_info = {
                "p_k_links": edges.get("p_k_links", []),
                "k_d_links": edges.get("k_d_links", []),
                "k_nodes": k_nodes,
                "d_nodes": d_nodes,
                "pre_audit": pre_audit_info
            }
            graph_edges_json = json.dumps(graph_edges_info, ensure_ascii=False, indent=2)
            graph_evidence_section = f"""
### GRAPH EVIDENCE (JSON Format - Fallback)

```json
{graph_edges_json}
```
"""
        
        # 构建 Naive Scores 展示（方案 B：带 Phase 1 上下文）
        naive_scores_section = self._format_naive_scores(
            naive_scores or {},
            phase1_result.get("top_candidates", [])
        )
        
        # 格式化 Prompt (使用修改后的模板)
        user_prompt = self._format_judge_prompt_v2(
            raw_text=raw_text,
            initial_candidates_list=initial_candidates_list,
            initial_reasoning=initial_reasoning,
            p_nodes_json=p_nodes_json,
            graph_evidence_section=graph_evidence_section,
            low_evidence_warning=low_evidence_warning,
            naive_scores_section=naive_scores_section
        )
        
        return user_prompt
    
    def _format_naive_scores(
        self,
        naive_scores: Dict[str, float],
        top_candidates: List[str]
    ) -> str:
        """
        格式化 Naive Scores 为易读的字符串
        
        方案 B：按 Phase 1 原始 Rank 排列 + 分数标注
        """
        if not naive_scores:
            return ""
        
        lines = ["### 📊 NAIVE SCORES (Evidence Strength Reference)\n"]
        lines.append("*Formula: Score = (Match × 1.0) - (Conflict × 1.5) - (Shadow × 0.1)*\n")
        
        # 找到最高分
        max_score = max(naive_scores.values()) if naive_scores else 0
        top1_score = naive_scores.get(f"d_{top_candidates[0]}", 0) if top_candidates else 0
        
        # 按 Phase 1 Rank 排列
        for rank, cand_id in enumerate(top_candidates, 1):
            d_id = f"d_{cand_id}"
            score = naive_scores.get(d_id, 0.0)
            
            # 标注逆转信号
            signal = ""
            if rank > 1 and score > top1_score:
                signal = " ⬆️ **Higher than Top-1**"
            elif score == max_score and score > 0:
                signal = " 🏆"
            
            disease_name = DIAGNOSIS_ID_MAP.get(cand_id, "Unknown")
            lines.append(f"- **{d_id}** ({disease_name}) [Phase1 Rank: {rank}]: Score = **{score:.1f}**{signal}")
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_judge_prompt_v2(
        self,
        raw_text: str,
        initial_candidates_list: str,
        initial_reasoning: str,
        p_nodes_json: str,
        graph_evidence_section: str,
        low_evidence_warning: str,
        naive_scores_section: str = ""
    ) -> str:
        """
        格式化 Judge Prompt V2 (使用 graph_summary 和 naive_scores)
        
        保持与原 PHASE3_JUDGE_USER_PROMPT_TEMPLATE 的兼容性，
        但将 graph_edges_json 替换为 graph_evidence_section
        """
        # 使用内置模板（带 naive_scores）
        return f"""
## PATIENT CASE

{raw_text}

## PHASE 1 INITIAL ASSESSMENT

**Initial Candidates:** {initial_candidates_list}

**Differential Reasoning:**
{initial_reasoning}

## PATIENT FINDINGS (P-Nodes)

```json
{p_nodes_json}
```

{graph_evidence_section}

{naive_scores_section}

{low_evidence_warning}

## YOUR TASK

Based on the above evidence, determine the final diagnosis. Apply the Tiered Logic:

1. **Safety Sentinel (Tier 1):** Check for Fatal Conflicts on Essential/Pathognomonic features (P-Node status must be "Absent", not just missing)
2. **Pivot Competition (Tier 2):** Compare Pivot support - candidates with matched Pivot Features are superior
3. **Coverage Audit (Tier 3):** Use Naive Scores as tie-breaker; select highest coverage

⚠️ **IMPORTANT:** A "Missing" symptom (Shadow) is NOT a Conflict. Only "Absent" (patient denied) causes Fatal Conflict.

Output your decision in JSON format:
```json
{{
    "final_diagnosis_id": "XX",
    "final_diagnosis_name": "Disease Name",
    "status": "Confirm|Overturn|Fallback",
    "reasoning_path": "Step-by-step reasoning...",
    "audit_log": ["Key decision points..."]
}}
```
"""
    
    def _validate_and_fix_output(
        self,
        parsed_json: Dict[str, Any],
        phase1_result: Dict[str, Any],
        audit_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证并修正 LLM 输出
        
        Phase 3 必须做出选择。如果 LLM 输出无效的诊断 ID，
        使用 fallback 策略。
        """
        final_diagnosis_id = parsed_json.get("final_diagnosis_id")
        
        # 验证 ID 有效性
        if not final_diagnosis_id or final_diagnosis_id not in DIAGNOSIS_ID_MAP:
            # Fallback 策略
            top_candidates = phase1_result.get("top_candidates", [])
            
            # 优先使用 Phase 1 的 Top-1（如果没有被 disqualify）
            disqualified_ids = [
                d["d_id"].replace("d_", "") 
                for d in audit_results.get("disqualified_candidates", [])
            ]
            
            for candidate_id in top_candidates:
                if candidate_id in DIAGNOSIS_ID_MAP and candidate_id not in disqualified_ids:
                    print(f"[JudgeAgent] Invalid ID '{final_diagnosis_id}', using fallback: {candidate_id}")
                    parsed_json["final_diagnosis_id"] = candidate_id
                    parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP[candidate_id]
                    parsed_json["status"] = "Fallback"
                    break
            else:
                # 最后的 fallback
                if top_candidates and top_candidates[0] in DIAGNOSIS_ID_MAP:
                    parsed_json["final_diagnosis_id"] = top_candidates[0]
                    parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP[top_candidates[0]]
                    parsed_json["status"] = "Emergency_Fallback"
                elif "01" in DIAGNOSIS_ID_MAP:
                    parsed_json["final_diagnosis_id"] = "01"
                    parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP["01"]
                    parsed_json["status"] = "Emergency_Fallback"
        else:
            # ID 有效，确保名称正确
            parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP.get(
                final_diagnosis_id,
                parsed_json.get("final_diagnosis_name", "Unknown")
            )
            
            # 确定状态 (强制覆盖 LLM 输出，确保逻辑一致性)
            # Confirm: Phase 1 的 final_diagnosis_id == Phase 3 的 final_diagnosis_id
            # Overturn: Phase 1 的 final_diagnosis_id != Phase 3 的 final_diagnosis_id
            phase1_final_id = phase1_result.get("final_diagnosis_id", "")
            if final_diagnosis_id == phase1_final_id:
                parsed_json["status"] = "Confirm"
            else:
                parsed_json["status"] = "Overturn"
        
        # 确保必要字段存在
        if "reasoning_path" not in parsed_json:
            parsed_json["reasoning_path"] = "Diagnosis selected based on graph analysis."
        if "audit_log" not in parsed_json:
            parsed_json["audit_log"] = []
        
        return parsed_json
