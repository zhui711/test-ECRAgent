"""
Phase 3 Memory-Augmented Judge (Unified Version)
=================================================

统一版 Judge Agent，作为 Phase 3 的唯一真源。
根据 PHASE3_REFACTOR_PLAN.md 重构，消除旧版 phase3_debate 的数据断裂。

核心功能：
1. 从 State 读取 graph_summary 和 naive_scores（由 Phase 2 生成）
2. 使用 Phase 2 Graph 中的 P-Nodes 进行 Memory Bank 检索
3. 将检索结果作为 Few-Shot Context 注入 Prompt
4. 基于 Evidence 进行公平竞争（移除 Risk Level Safeguard）

设计决策：
- 检索源: Phase 2 Graph P-Nodes（更准确的症状集合）
- Risk Level: 彻底移除，所有候选仅凭 Evidence & Naive Score 竞争
- Shadow 指令: 不以 Shadow 数量惩罚，仅检查 Pathognomonic 缺失
- Naive Score: 作为参考基线，允许医学推理覆盖
"""

import json
from typing import Dict, Any, Optional, List

from .phase3_debate import JudgeAgent
from src.utils.api_client import LLMClient
from src.memory.memory_bank import MemoryBankManager
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.utils.json_utils import parse_json_from_text
from config.prompt_phase3_judge import PHASE3_JUDGE_SYSTEM_PROMPT


class MemoryAugmentedJudge(JudgeAgent):
    """
    Memory-Augmented Judge Agent
    
    继承 JudgeAgent，增加 Memory Bank 检索和 Few-Shot 注入功能。
    
    工作流程：
    1. 从 Memory Bank 检索相似案例 (2 Overturn + 2 Confirm)
    2. 格式化为 Few-Shot Context (隐藏标签)
    3. 将 Context 注入到 Prompt 中
    4. 执行原有的 Tiered Logic 判决
    
    Attributes:
        memory_bank: Memory Bank 管理器
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = "qwen3-32b",
        memory_bank: Optional[MemoryBankManager] = None,
        memory_bank_dir: str = "memory_bank"
    ):
        """
        初始化 Memory-Augmented Judge
        
        Args:
            llm_client: LLM 客户端
            model_name: 模型名称
            memory_bank: 已加载的 Memory Bank 实例 (可选)
            memory_bank_dir: Memory Bank 目录 (如果未提供 memory_bank)
        """
        super().__init__(llm_client, model_name)
        
        # 初始化或使用传入的 Memory Bank
        if memory_bank is not None:
            self.memory_bank = memory_bank
        else:
            self.memory_bank = MemoryBankManager(output_dir=memory_bank_dir)
            try:
                self.memory_bank.load()
                print(f"[MemoryAugmentedJudge] Loaded Memory Bank: "
                      f"{self.memory_bank.get_statistics()}")
            except Exception as e:
                print(f"[MemoryAugmentedJudge] Warning: Failed to load Memory Bank: {e}")
        
        # 统计信息
        self._stats = {
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "avg_similarity_overturn": 0.0,
            "avg_similarity_confirm": 0.0
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理状态，执行 Memory-Augmented Phase 3 判决（统一版）
        
        重构后的主流程：
        1. 从 State 读取 graph_summary 和 naive_scores
        2. JIT Retrieval: 使用 Phase 2 Graph P-Nodes 检索 Memory Bank
        3. 构建 Prompt 并调用 LLM
        4. 彻底移除 _pre_audit 调用（Risk Level 不再参与）
        
        Args:
            state: LangGraph 状态字典
        
        Returns:
            更新后的状态字典（包含 final_output, memory_records）
        """
        try:
            # ========== 1. 读取 State ==========
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
            
            # 从 State 读取 graph_summary 和 naive_scores（由 Phase 2 Summarizer 生成）
            graph_summary = state.get("graph_summary", "")
            naive_scores = state.get("naive_scores", {})
            
            # ========== 2. JIT Memory Retrieval ==========
            # 检查 state.memory_records，如果为空则执行检索
            memory_records = state.get("memory_records") or []
            
            if not memory_records:
                # 使用 Phase 2 Graph 的 P-Nodes 进行检索（更准确）
                memory_records = self._retrieve_from_graph_p_nodes(graph_json)
                # 更新 State
                state["memory_records"] = memory_records
            
            # 格式化 Few-Shot Context
            few_shot_context = self._format_few_shot_from_records(memory_records)
            
            # ========== 3. 构建 Prompt ==========
            # 注意：彻底移除 _pre_audit 调用，所有候选仅凭 Evidence 竞争
            user_prompt = self._construct_unified_prompt(
                graph_json=graph_json,
                phase1_result=phase1_result,
                raw_text=raw_text,
                graph_summary=graph_summary,
                naive_scores=naive_scores,
                few_shot_context=few_shot_context
            )
            
            # ========== 4. LLM 调用 ==========
            messages = [
                {"role": "system", "content": PHASE3_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            result = self.llm_client.generate_json(
                messages=messages,
                model=self.model_name,
                logprobs=False,
                temperature=0.2,
                max_tokens=4096
            )
            
            if result["error"]:
                print(f"[MemoryAugmentedJudge] LLM error: {result['error']}")
                return {
                    **state,
                    "status": "failed",
                    "error_log": f"LLM error: {result['error']}"
                }
            
            # ========== 5. 解析输出 ==========
            response_content = result["content"]
            parsed_json = parse_json_from_text(response_content, verbose=True)
            
            if not parsed_json:
                print(f"[MemoryAugmentedJudge] Failed to parse JSON")
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Failed to parse JSON from LLM response"
                }
            
            # 验证并修正诊断 ID（使用简化版，不依赖 audit_results）
            final_output = self._validate_output_unified(parsed_json, phase1_result)
            
            # 添加 Memory Retrieval 信息
            final_output["memory_retrieval_used"] = bool(memory_records)
            
            print(f"[MemoryAugmentedJudge] Final diagnosis: {final_output.get('final_diagnosis_id')} - "
                  f"{final_output.get('final_diagnosis_name')} ({final_output.get('status')})")
            
            return {
                **state,
                "final_output": final_output,
                "memory_records": memory_records,
                "status": "success"
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Phase 3 Memory-Augmented error: {str(e)}\n{traceback.format_exc()}"
            print(f"[MemoryAugmentedJudge] {error_msg}")
            return {
                **state,
                "status": "failed",
                "error_log": error_msg
            }
    
    def _retrieve_from_graph_p_nodes(
        self,
        graph_json: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        使用 Phase 2 Graph 的 P-Nodes 进行 Memory Bank 检索
        
        这是 JIT (Just-In-Time) 检索策略的核心实现。
        使用 Phase 2 产出的 P-Nodes（经过验证和整理的症状集合）。
        
        Args:
            graph_json: Phase 2 输出的图谱 JSON
        
        Returns:
            检索到的完整 Payload 列表
        """
        self._stats["total_retrievals"] += 1
        
        try:
            if self.memory_bank is None:
                print("[MemoryAugmentedJudge] Memory Bank not initialized")
                return []
            
            # 从 Phase 2 Graph 提取 P-Nodes
            p_nodes = graph_json.get("graph", {}).get("nodes", {}).get("p_nodes", [])
            
            if not p_nodes:
                print("[MemoryAugmentedJudge] No P-Nodes in graph for retrieval")
                return []
            
            # 检索相似案例 (2 Overturn + 2 Confirm)
            similar_cases = self.memory_bank.retrieve_similar(
                query_p_nodes=p_nodes,
                n_overturn=2,
                n_confirm=2
            )
            
            overturn_cases = similar_cases.get("overturn", [])
            confirm_cases = similar_cases.get("confirm", [])
            
            all_cases = overturn_cases + confirm_cases
            
            if not all_cases:
                print("[MemoryAugmentedJudge] No similar cases found")
                return []
            
            self._stats["successful_retrievals"] += 1
            
            # 更新相似度统计
            if overturn_cases:
                avg_sim = sum(c.get("similarity_score", 0) for c in overturn_cases) / len(overturn_cases)
                self._stats["avg_similarity_overturn"] = avg_sim
            if confirm_cases:
                avg_sim = sum(c.get("similarity_score", 0) for c in confirm_cases) / len(confirm_cases)
                self._stats["avg_similarity_confirm"] = avg_sim
            
            print(f"[MemoryAugmentedJudge] Retrieved {len(overturn_cases)} Overturn + {len(confirm_cases)} Confirm cases")
            return all_cases
            
        except Exception as e:
            print(f"[MemoryAugmentedJudge] Retrieval error: {e}")
            return []
    
    def _format_few_shot_from_records(
        self,
        memory_records: List[Dict[str, Any]]
    ) -> str:
        """
        从 Memory Bank 完整 Payload 格式化 Few-Shot Context
        
        关键字段提取：
        - ground_truth_name: 正确答案
        - initial_diagnosis_name: 初始诊断
        - final_diagnosis_name: 最终诊断
        - outcome: Confirm/Overturn
        - p_nodes_summary: 患者症状摘要
        - similarity_score: 相似度分数（展示供 LLM 参考）
        
        Args:
            memory_records: Memory Bank 检索到的完整 Payload 列表
        
        Returns:
            格式化的 Few-Shot Context 字符串
        """
        if not memory_records:
            return ""
        
        lines = ["### 🧠 SIMILAR HISTORICAL CASES"]
        lines.append("*Reference cases from training data. Use their reasoning patterns as guidance.*\n")
        
        for idx, case in enumerate(memory_records, 1):
            # 提取关键字段
            initial = case.get("initial_diagnosis_name", "Unknown")
            final = case.get("final_diagnosis_name", "Unknown")
            ground_truth = case.get("ground_truth_name", "Unknown")
            outcome = case.get("outcome", "Unknown")
            p_summary = case.get("p_nodes_summary", "N/A")
            similarity = case.get("similarity_score", 0.0)
            
            # 动态判断 Correctness
            is_correct = (final == ground_truth)
            correctness = "✅ Correct" if is_correct else "❌ Incorrect"
            
            lines.append(f"**[Case {idx}]** (Similarity: {similarity:.2f})")
            lines.append(f"  - Patient Summary: {p_summary}")
            lines.append(f"  - Initial Diagnosis: {initial}")
            lines.append(f"  - Final Diagnosis: {final} ({correctness})")
            lines.append(f"  - Outcome: {outcome}")
            
            # Key Insight
            if outcome == "Overturn":
                lines.append(f"  - 💡 Insight: Changed from {initial} to {final}")
            else:
                lines.append(f"  - 💡 Insight: Confirmed initial diagnosis {initial}")
            lines.append("")
        
        return "\n".join(lines)
    
    # ==================== 旧版方法 (保留兼容性，标记为 Deprecated) ====================
    
    def _retrieve_and_format_context(
        self,
        phase1_result: Dict[str, Any]
    ) -> str:
        """
        [DEPRECATED] 使用 Phase 1 P-Nodes 检索
        
        已被 _retrieve_from_graph_p_nodes 替代。
        保留此方法仅为兼容旧代码。
        """
        # 调用新方法的逻辑
        track_b_output = phase1_result.get("track_b_output", {})
        p_nodes = track_b_output.get("p_nodes", [])
        
        if not p_nodes or self.memory_bank is None:
            return ""
        
        try:
            similar_cases = self.memory_bank.retrieve_similar(
                query_p_nodes=p_nodes,
                n_overturn=2,
                n_confirm=2
            )
            all_cases = similar_cases.get("overturn", []) + similar_cases.get("confirm", [])
            return self._format_few_shot_from_records(all_cases)
        except Exception as e:
            print(f"[MemoryAugmentedJudge] Deprecated retrieval error: {e}")
            return ""
    
    def _format_hidden_label_context(
        self,
        overturn_cases: List[Dict[str, Any]],
        confirm_cases: List[Dict[str, Any]]
    ) -> str:
        """
        [DEPRECATED] 已被 _format_few_shot_from_records 替代
        """
        all_cases = overturn_cases + confirm_cases
        return self._format_few_shot_from_records(all_cases)
    
    def _construct_unified_prompt(
        self,
        graph_json: Dict[str, Any],
        phase1_result: Dict[str, Any],
        raw_text: str,
        graph_summary: str,
        naive_scores: Dict[str, float],
        few_shot_context: str
    ) -> str:
        """
        构建统一版 Prompt（无 Risk Level、无 Pre-Audit）
        
        核心变更：
        1. 使用 Rich Structured Summary 替代 JSON
        2. 添加 Shadow 指令：不以数量惩罚，仅检查 Pathognomonic 缺失
        3. 添加 Naive Score 说明：仅供参考，允许医学推理覆盖
        4. 彻底移除 Pre-Audit 信息
        
        Args:
            graph_json: Phase 2 图谱
            phase1_result: Phase 1 结果
            raw_text: 原始病历
            graph_summary: Rich Structured Summary
            naive_scores: 朴素评分
            few_shot_context: Few-Shot Context
        
        Returns:
            完整的 User Prompt
        """
        # 提取初始候选
        initial_candidates = phase1_result.get("top_candidates", [])
        initial_candidates_list = ", ".join([
            f"{DIAGNOSIS_ID_MAP.get(cid, 'Unknown')} (ID: {cid})"
            for cid in initial_candidates
        ])
        initial_reasoning = phase1_result.get("differential_reasoning", "")
        
        # 构建 Naive Scores Section（带公式说明）
        naive_scores_section = self._format_naive_scores_unified(
            naive_scores or {},
            initial_candidates
        )
        
        # 构建完整 Prompt
        return f"""
## PATIENT CASE

{raw_text}

## PHASE 1 INITIAL ASSESSMENT

**Initial Candidates:** {initial_candidates_list}

**Differential Reasoning:**
{initial_reasoning}

## 📊 NAIVE SCORES (Reference Baseline)

{naive_scores_section}

## ⚖️ RICH EVIDENCE SUMMARY

{graph_summary}

{few_shot_context}

## YOUR TASK

Based on the above evidence, determine the **final diagnosis**.

### Decision Framework:

1. **Fatal Conflict Check (Tier 1):**
   - If a candidate has a **Conflict** on an **Essential/Pathognomonic** feature (P-Node status = "Absent"), it is **disqualified**.
   - ⚠️ **Shadow ≠ Conflict**: Missing symptoms (Shadow) are NOT disqualifying factors.

2. **Pivot Competition (Tier 2):**
   - Candidates with **matched Pivot Features** are superior to those without.
   - If a lower-ranked candidate has Pivot support but Top-1 does not, consider **Overturn**.

3. **Evidence Coverage (Tier 3):**
   - Use **Naive Scores** as a quantitative baseline.
   - Higher score = more supporting evidence.

### ⚠️ CRITICAL INSTRUCTIONS ON SHADOW NODES:

1. **Do NOT penalize candidates based on the *count* or *number* of Shadow nodes.**
2. **Only penalize if CRITICAL Pathognomonic evidence is missing:**
   - Example: Missing "D-dimer elevation" for Pulmonary Embolism = significant concern.
   - Example: Missing "Fatigue" = ignore (not a deal-breaker).
3. **Naive Score Context:** The score formula uses Shadow×0.1 penalty. If a candidate's score appears low due to many irrelevant Shadows, use your **medical judgment to override** the score.

### Output Format (JSON):

```json
{{
    "final_diagnosis_id": "XX",
    "final_diagnosis_name": "Disease Name",
    "status": "Confirm|Overturn|Fallback",
    "reasoning_path": "Step-by-step reasoning explaining your decision...",
    "audit_log": ["Key decision points..."]
}}
```
"""
    
    def _format_naive_scores_unified(
        self,
        naive_scores: Dict[str, float],
        top_candidates: List[str]
    ) -> str:
        """
        格式化 Naive Scores（统一版，带公式说明）
        
        Args:
            naive_scores: 评分字典
            top_candidates: Phase 1 Top Candidates
        
        Returns:
            格式化的 Naive Scores 字符串
        """
        if not naive_scores:
            return "*No Naive Scores available*"
        
        lines = ["**Formula:** `Score = (Match × 1.0) - (Conflict × 1.5) - (Shadow × 0.1)`"]
        lines.append("")
        lines.append("*Note: Shadow penalty (0.1) is minimal. Do NOT over-penalize missing evidence.*")
        lines.append("")
        
        # 找到最高分
        max_score = max(naive_scores.values()) if naive_scores else 0
        top1_score = naive_scores.get(f"d_{top_candidates[0]}", 0) if top_candidates else 0
        
        # 按 Phase 1 Rank 排列
        for rank, cand_id in enumerate(top_candidates, 1):
            d_id = f"d_{cand_id}"
            score = naive_scores.get(d_id, 0.0)
            
            # 标注信号
            signal = ""
            if rank > 1 and score > top1_score:
                signal = " ⬆️ **Higher than Top-1 (Potential Overturn Signal)**"
            elif score == max_score and score > 0:
                signal = " 🏆 **Highest**"
            
            disease_name = DIAGNOSIS_ID_MAP.get(cand_id, "Unknown")
            score_str = f"+{score:.1f}" if score >= 0 else f"{score:.1f}"
            lines.append(f"- **{d_id}** ({disease_name}) [Rank {rank}]: Score = **{score_str}**{signal}")
        
        return "\n".join(lines)
    
    def _validate_output_unified(
        self,
        parsed_json: Dict[str, Any],
        phase1_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证并修正 LLM 输出（统一版，不依赖 audit_results）
        
        Args:
            parsed_json: LLM 输出的 JSON
            phase1_result: Phase 1 结果
        
        Returns:
            验证后的 final_output
        """
        final_diagnosis_id = parsed_json.get("final_diagnosis_id")
        
        # 验证 ID 有效性
        if not final_diagnosis_id or final_diagnosis_id not in DIAGNOSIS_ID_MAP:
            # Fallback 策略：使用 Phase 1 的 Top-1
            top_candidates = phase1_result.get("top_candidates", [])
            
            if top_candidates and top_candidates[0] in DIAGNOSIS_ID_MAP:
                fallback_id = top_candidates[0]
                print(f"[MemoryAugmentedJudge] Invalid ID '{final_diagnosis_id}', using fallback: {fallback_id}")
                parsed_json["final_diagnosis_id"] = fallback_id
                parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP[fallback_id]
                parsed_json["status"] = "Fallback"
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
            
            # 确定状态
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
    
    # ==================== 旧版 Prompt 方法 (保留兼容性) ====================
    
    def _construct_memory_augmented_prompt(
        self,
        graph_json: Dict[str, Any],
        phase1_result: Dict[str, Any],
        raw_text: str,
        audit_results: Dict[str, Any],
        graph_summary: str,
        naive_scores: Dict[str, float],
        few_shot_context: str
    ) -> str:
        """
        [DEPRECATED] 旧版 Prompt 构建方法
        
        已被 _construct_unified_prompt 替代。
        保留此方法仅为兼容旧代码。
        """
        return self._construct_unified_prompt(
            graph_json=graph_json,
            phase1_result=phase1_result,
            raw_text=raw_text,
            graph_summary=graph_summary,
            naive_scores=naive_scores,
            few_shot_context=few_shot_context
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取 Memory Retrieval 统计信息"""
        return self._stats.copy()


# ==================== 工厂函数 ====================

def create_judge_agent(
    llm_client: LLMClient,
    model_name: str = "qwen3-32b",
    use_memory: bool = True,
    memory_bank_dir: str = "memory_bank",
    memory_bank: Optional[MemoryBankManager] = None
) -> MemoryAugmentedJudge:
    """
    创建 Judge Agent 的工厂函数（统一版）
    
    重构后：始终返回 MemoryAugmentedJudge。
    use_memory 参数仅控制是否执行 Memory Bank 检索，
    但类本身不变，确保数据流一致性。
    
    Args:
        llm_client: LLM 客户端
        model_name: 模型名称
        use_memory: 是否使用 Memory Bank 检索（不影响类选择）
        memory_bank_dir: Memory Bank 目录
        memory_bank: 已加载的 Memory Bank 实例
    
    Returns:
        MemoryAugmentedJudge 实例（统一）
    """
    # 始终使用 MemoryAugmentedJudge，确保数据流一致
    if use_memory:
        return MemoryAugmentedJudge(
            llm_client=llm_client,
            model_name=model_name,
            memory_bank=memory_bank,
            memory_bank_dir=memory_bank_dir
        )
    else:
        # 即使不使用 Memory，也返回 MemoryAugmentedJudge
        # 只是不传入 memory_bank，检索时会返回空结果
        return MemoryAugmentedJudge(
            llm_client=llm_client,
            model_name=model_name,
            memory_bank=None,
            memory_bank_dir=memory_bank_dir
        )


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("MemoryAugmentedJudge module loaded successfully")
    print("Use create_judge_agent() to instantiate")




