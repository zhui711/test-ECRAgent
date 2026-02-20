"""
LangGraph State Definition
定义 AgentState，用于在 LangGraph 中传递状态

根据 04_Workflow_Architecture.md:
- case_id: 案例标识
- input_case: 输入数据（包含 narrative）
- phase1_result: Phase 1 输出
- graph_json: Phase 2 输出（MedicalGraph.to_dict()）
- graph_summary: 图谱的自然语言摘要 (用于 Phase 3)
- naive_scores: 确定性评分 (调试用)
- final_output: Phase 3 输出（最终诊断）
- status: 处理状态
- error_log: 错误日志

扩展字段 (Offline Training 支持):
- global_hint: Teacher 反馈，注入各 Phase 的 Prompt
- retry_count: 当前重试次数
- memory_context: Memory Bank 检索结果（Few-Shot 上下文）
"""
from typing import TypedDict, Optional, Dict, Any, Literal, List


class AgentState(TypedDict):
    """
    LangGraph 状态定义
    
    状态流转：
    Phase 1: input_case -> phase1_result
    Phase 2: phase1_result + input_case.narrative -> graph_json
    Summarize: graph_json -> graph_summary, naive_scores
    Phase 3: graph_summary + phase1_result -> final_output
    
    Offline Training 扩展：
    - global_hint: Teacher Critique 注入到各 Phase
    - retry_count: 追踪重试次数
    """
    # 输入
    case_id: str
    input_case: Dict[str, Any]  # 包含 narrative, ground_truth 等
    
    # Phase 1 结果
    phase1_result: Optional[Dict[str, Any]]
    # 结构:
    # {
    #   "structured_analysis": {...},
    #   "differential_reasoning": "...",
    #   "top_candidates": ["25", "45", ...],
    #   "final_diagnosis_id": "25",
    #   "calculated_confidence": 0.0,
    #   "error": null
    # }
    
    # Phase 2 结果
    graph_json: Optional[Dict[str, Any]]
    # 结构: MedicalGraph.to_dict() 的输出
    # {
    #   "case_metadata": {...},
    #   "phase1_context": {...},
    #   "graph": {
    #     "nodes": {"p_nodes": [], "k_nodes": [], "d_nodes": []},
    #     "edges": {"p_k_links": [], "k_d_links": []}
    #   }
    # }
    
    # Phase 2.5: Graph Summary (新增)
    graph_summary: Optional[str]
    # 结构化自然语言摘要，按 Candidate 分组
    # 用于 Phase 3 的输入，替代原始 JSON
    
    # Phase 2.5: Naive Scores (新增)
    naive_scores: Optional[Dict[str, float]]
    # 确定性评分，仅用于调试
    # 公式: Score = (Match * 1.0) - (Conflict * 1.5) - (Shadow * 0.1)
    
    # Phase 3 最终输出
    final_output: Optional[Dict[str, Any]]
    # 结构:
    # {
    #   "final_diagnosis_id": "25",
    #   "final_diagnosis_name": "Acute COPD exacerbation / infection",
    #   "status": "Confirm" | "Overturn" | "Fallback",
    #   "reasoning_path": "...",
    #   "audit_log": [...]
    # }
    
    # 状态
    status: Literal["pending", "processing", "success", "failed"]
    
    # 错误日志
    error_log: Optional[str]
    
    # ==================== Offline Training 扩展字段 ====================
    
    # Teacher Critique (全局提示)
    # 在 Offline Training 的重试流程中，由 Critical Model 生成
    # 注入到 Phase 1/2/3 的 Prompt 末尾
    global_hint: Optional[str]
    
    # 重试次数
    # 追踪当前是第几次尝试（0 = 首次）
    retry_count: int
    
    # Memory Bank 上下文 (旧版，保留兼容性)
    # 在 Online 推理时，由 Memory Bank 检索生成
    # 包含相似案例的 Few-Shot 上下文
    memory_context: Optional[str]
    
    # Memory Bank 检索结果 (新版，完整 Payload)
    # 存储 Memory Bank 检索到的完整案例记录列表
    # 每个记录包含: case_id, ground_truth_name, initial_diagnosis_name,
    # final_diagnosis_name, outcome, p_nodes_summary, similarity_score 等
    memory_records: Optional[List[Dict[str, Any]]]
