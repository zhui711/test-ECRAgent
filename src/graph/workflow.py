"""
LangGraph Workflow
定义 Phase 1 -> Phase 2 -> Phase 3 的完整处理流程

根据 04_Workflow_Architecture.md:
- 线性流: start -> phase1 -> phase2 -> phase3 -> end
- 异常处理: 捕获错误并记录，不 crash

Phase 1 架构修正（根据 05_Phase1_DualTrack_Refactor.md）:
- 使用 Phase1Manager 协调双轨执行
- Track A: 直觉诊断 (Top-5 Candidates)
- Track B: 感知表征 (P-Nodes)
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.phase1_manager import Phase1Manager  # 使用 Manager 替代单一 Agent
from src.agents.phase2_investigation import Phase2GraphReasoning
# DEPRECATED: JudgeAgent 已被 MemoryAugmentedJudge 替代
# from src.agents.phase3_debate import JudgeAgent
from src.agents.phase3_memory_judge import MemoryAugmentedJudge, create_judge_agent
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.utils.api_client import LLMClient
from src.utils.graph_tools import (
    rebuild_graph_from_json,
    serialize_graph_to_summary,
    calculate_deterministic_score
)
from src.utils.trace_logger import TraceLogger
import yaml
import os


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "../../config/settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def node_phase1(state: AgentState, phase1_manager: Phase1Manager) -> AgentState:
    """
    Phase 1 节点：调用 Phase1Manager 处理病例（双轨执行）
    
    根据 05_Phase1_DualTrack_Refactor.md:
    - Track A: 生成 Top-5 Candidates
    - Track B: 提取结构化 P-Nodes
    
    输入: state['input_case']['narrative']
    输出: state['phase1_result'] (包含 track_b_output)
    """
    try:
        # 获取患者叙述
        patient_narrative = state["input_case"].get("narrative", "")
        
        if not patient_narrative:
            state["status"] = "failed"
            state["error_log"] = "Patient narrative is empty"
            return state
        
        print(f"[Workflow] Phase 1: Processing case {state.get('case_id', 'unknown')}...")
        
        # 调用 Phase1Manager（双轨执行）
        phase1_result = phase1_manager.process(patient_narrative)
        
        # 更新状态
        state["phase1_result"] = phase1_result
        state["status"] = "processing"
        
        # 打印摘要
        if phase1_result and not phase1_result.get("error"):
            top_candidates = phase1_result.get("top_candidates", [])
            p_nodes_count = len(phase1_result.get("track_b_output", {}).get("p_nodes", []))
            print(f"[Workflow] Phase 1 complete. Top candidates: {top_candidates}, P-Nodes: {p_nodes_count}")
        
        return state
        
    except Exception as e:
        import traceback
        state["status"] = "failed"
        state["error_log"] = f"Phase 1 error: {str(e)}\n{traceback.format_exc()}"
        print(f"[Workflow] Phase 1 error: {e}")
        return state


def router_phase1(state: AgentState) -> str:
    """
    Phase 1 路由节点：决定是否进入 Phase 2
    
    当前配置：强制进入 Phase 2（跳过置信度检查）
    """
    # 检查 Phase 1 是否成功
    phase1_result = state.get("phase1_result")
    
    if phase1_result is None or phase1_result.get("error"):
        print("[Workflow] Phase 1 failed, going to FAST_PATH for error handling")
        return "FAST_PATH"
    
    # 强制进入 Phase 2
    print("[Workflow] Routing to PHASE2 (confidence check skipped)")
    return "PHASE2"


def node_router_postprocess(state: AgentState) -> AgentState:
    """
    路由后处理节点：处理 Phase 1 失败或快速路径的情况
    
    当前配置强制进入 Phase 2，此节点主要处理错误情况
    """
    phase1_result = state.get("phase1_result")
    
    if phase1_result is None:
        state["status"] = "failed"
        state["error_log"] = state.get("error_log") or "Phase 1 result is None"
        return state
    
    if phase1_result.get("error"):
        state["status"] = "failed"
        state["error_log"] = phase1_result.get("error")
        return state
    
    # Fast Path: 直接输出 Phase 1 结果（当前未使用）
    final_diagnosis_id = phase1_result.get("final_diagnosis_id")
    diagnosis_name = DIAGNOSIS_ID_MAP.get(final_diagnosis_id, "Unknown")
    
    state["final_output"] = {
        "final_diagnosis_id": final_diagnosis_id,
        "final_diagnosis_name": diagnosis_name,
        "status": "FastPath",
        "reasoning_path": "Fast path due to error or high confidence",
        "audit_log": []
    }
    state["status"] = "success"
    
    return state


def node_phase2(state: AgentState, phase2_agent: Phase2GraphReasoning) -> AgentState:
    """
    Phase 2 节点：构建因果图谱
    
    输入: state['phase1_result'], state['input_case']['narrative']
    输出: state['graph_json']
    """
    try:
        print(f"[Workflow] Phase 2: Building causal graph...")
        updated_state = phase2_agent.process(state)
        
        # 检查是否成功
        if updated_state.get("status") == "failed":
            print(f"[Workflow] Phase 2 failed: {updated_state.get('error_log', 'Unknown error')}")
        else:
            graph_json = updated_state.get("graph_json", {})
            p_count = len(graph_json.get("graph", {}).get("nodes", {}).get("p_nodes", []))
            k_count = len(graph_json.get("graph", {}).get("nodes", {}).get("k_nodes", []))
            print(f"[Workflow] Phase 2 complete. Graph: {p_count} P-Nodes, {k_count} K-Nodes")
        
        return updated_state
        
    except Exception as e:
        import traceback
        state["status"] = "failed"
        state["error_log"] = f"Phase 2 error: {str(e)}\n{traceback.format_exc()}"
        print(f"[Workflow] Phase 2 error: {e}")
        return state


def node_summarize_graph(state: AgentState) -> AgentState:
    """
    Graph Summarize 节点：序列化图谱并计算确定性评分
    
    位置：插入在 phase2 和 phase3 之间
    
    输入: state['graph_json']
    输出: state['graph_summary'], state['naive_scores']
    """
    try:
        print(f"[Workflow] Summarize: Converting graph to summary...")
        
        graph_json = state.get("graph_json")
        if not graph_json:
            print("[Workflow] Summarize: No graph_json found, skipping...")
            state["graph_summary"] = ""
            state["naive_scores"] = {}
            return state
        
        # 从 JSON 重建 MedicalGraph 对象
        graph = rebuild_graph_from_json(graph_json)
        
        # 生成自然语言摘要
        graph_summary = serialize_graph_to_summary(graph)
        state["graph_summary"] = graph_summary
        
        # 计算确定性评分
        naive_scores = calculate_deterministic_score(graph)
        state["naive_scores"] = naive_scores
        
        # [TraceLogger] 记录摘要和分数
        logger = TraceLogger.get_instance()
        logger.log_graph_summary(graph_summary)
        logger.log_naive_scores(naive_scores)
        
        print(f"[Workflow] Summarize complete. Summary: {len(graph_summary)} chars")
        print(f"[Workflow] Naive Scores: {naive_scores}")
        
        return state
        
    except Exception as e:
        import traceback
        print(f"[Workflow] Summarize error: {e}")
        traceback.print_exc()
        # 不让这个节点失败整个流程
        state["graph_summary"] = ""
        state["naive_scores"] = {}
        return state


def node_phase3(state: AgentState, judge_agent: MemoryAugmentedJudge) -> AgentState:
    """
    Phase 3 节点：最终判决
    
    输入: state['graph_json'], state['phase1_result']
    输出: state['final_output']
    """
    try:
        print(f"[Workflow] Phase 3: Executing final judgment...")
        updated_state = judge_agent.process(state)
        
        # 检查是否成功
        if updated_state.get("status") == "success":
            final_output = updated_state.get("final_output", {})
            diagnosis_id = final_output.get("final_diagnosis_id", "?")
            diagnosis_name = final_output.get("final_diagnosis_name", "Unknown")
            status = final_output.get("status", "?")
            print(f"[Workflow] Phase 3 complete. Final: {diagnosis_id} - {diagnosis_name} ({status})")
        else:
            print(f"[Workflow] Phase 3 failed: {updated_state.get('error_log', 'Unknown error')}")
        
        return updated_state
        
    except Exception as e:
        import traceback
        state["status"] = "failed"
        state["error_log"] = f"Phase 3 error: {str(e)}\n{traceback.format_exc()}"
        print(f"[Workflow] Phase 3 error: {e}")
        return state


def build_workflow(phase1_manager: Phase1Manager = None) -> StateGraph:
    """
    构建 LangGraph 工作流
    
    流程: Phase 1 -> (Router) -> Phase 2 -> Phase 3 -> END
    
    根据 05_Phase1_DualTrack_Refactor.md:
    - Phase 1 使用 Phase1Manager 协调双轨执行
    
    Args:
        phase1_manager: Phase1Manager 实例（可选，如果未提供则自动创建）
    
    Returns:
        编译后的 Graph
    """
    # 加载配置
    config = load_config()
    api_config = config.get("api", {})
    
    # 初始化 LLM 客户端
    base_url = api_config.get("base_url", "https://yunwu.ai/v1")
    model_name = api_config.get("model_name", "gpt-4o")
    
    llm_client = LLMClient(
        base_url=base_url,
        api_key=None,  # 从环境变量读取
        timeout=api_config.get("timeout", 120)
    )
    
    # 如果未提供 Phase1Manager，则自动创建
    if phase1_manager is None:
        phase1_manager = Phase1Manager(llm_client, model_name=model_name)
    
    # 初始化 Phase 2 Agent
    phase2_agent = Phase2GraphReasoning(llm_client, model_name=model_name)
    
    # 初始化 Phase 3 Agent（统一使用 MemoryAugmentedJudge）
    # 注意：workflow 模式下默认不启用 Memory Bank 检索（可通过参数调整）
    judge_agent = create_judge_agent(
        llm_client=llm_client,
        model_name=model_name,
        use_memory=False,  # Workflow 模式默认不使用 Memory（可配置）
        memory_bank=None
    )
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("phase1", lambda state: node_phase1(state, phase1_manager))
    workflow.add_node("router_postprocess", node_router_postprocess)
    workflow.add_node("phase2", lambda state: node_phase2(state, phase2_agent))
    workflow.add_node("summarize_graph", node_summarize_graph)  # 新增节点
    workflow.add_node("phase3", lambda state: node_phase3(state, judge_agent))
    
    # 设置入口点
    workflow.set_entry_point("phase1")
    
    # Phase 1 完成后，根据路由决策
    workflow.add_conditional_edges(
        "phase1",
        router_phase1,
        {
            "FAST_PATH": "router_postprocess",  # 错误处理或快速路径
            "PHASE2": "phase2"  # 正常流程
        }
    )
    
    # router_postprocess 完成后，直接结束
    workflow.add_edge("router_postprocess", END)
    
    # Phase 2 完成后，进入 Summarize 节点
    workflow.add_edge("phase2", "summarize_graph")
    
    # Summarize 完成后，进入 Phase 3
    workflow.add_edge("summarize_graph", "phase3")
    
    # Phase 3 完成后，直接结束
    workflow.add_edge("phase3", END)
    
    # 编译图
    app = workflow.compile()
    
    return app


def create_initial_state(case_id: str, input_case: Dict[str, Any]) -> AgentState:
    """
    创建初始状态
    
    Args:
        case_id: 案例 ID
        input_case: 输入数据（包含 narrative）
    
    Returns:
        初始化的 AgentState
    """
    return AgentState(
        case_id=case_id,
        input_case=input_case,
        phase1_result=None,
        graph_json=None,
        graph_summary=None,
        naive_scores=None,
        final_output=None,
        status="pending",
        error_log=None,
        global_hint=None,
        retry_count=0,
        memory_context=None,
        memory_records=None  # 新增：Memory Bank 检索结果
    )
