"""
Phase 1 Manager: Dual-Track Coordination
协调 Track A (Intuition) 和 Track B (Perception) 的执行

根据算法流程图 Line 1-4:
1. P <- ExtractFeatures(T)           // Track B: Structured Patient Features
2. D_top5, d_initial <- LLM_fast(T, D_all)  // Track A: Initial Candidates
3. G <- InitGraph(P, D_top5)

核心职责:
- 串行执行 Track A 和 Track B（可扩展为并行）
- 合并两个 Track 的输出
- 提供统一的 Phase 1 接口
"""
from typing import Dict, Any, Optional
from src.utils.api_client import LLMClient
from src.agents.phase1_perception import Phase1TrackAAgent
from src.agents.phase1_trackB import Phase1TrackBAgent


class Phase1Manager:
    """
    Phase 1 总控管理器
    
    协调双轨执行:
    - Track A: 直觉诊断 (System 1) - 生成 Top-5 候选
    - Track B: 感知表征 (Perception) - 提取结构化 P-Nodes
    """
    
    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):
        """
        初始化 Phase1Manager
        
        Args:
            llm_client: LLM 客户端实例
            model_name: 模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name
        
        # 初始化两个 Track
        self.track_a = Phase1TrackAAgent(llm_client, model_name=model_name)
        self.track_b = Phase1TrackBAgent(llm_client, model_name=model_name)
    
    def process(
        self, 
        raw_narrative: str, 
        age: int = None, 
        sex: str = None,
        global_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行 Phase 1 双轨处理
        
        根据算法流程图 Line 1-3:
        1. Track B: P <- ExtractFeatures(T)
        2. Track A: D_top5, d_initial <- LLM_fast(T, D_all)
        
        当前实现为串行执行（先 Track A，再 Track B）
        
        Args:
            raw_narrative: 原始患者叙述文本
            age: 年龄（可选）
            sex: 性别（可选）
            global_hint: Teacher 反馈（用于 Offline Training 重试）
        
        Returns:
            合并后的 Phase 1 结果，包含:
            - top_candidates: Top-5 候选诊断 ID 列表 (From Track A)
            - final_diagnosis_id: 初始诊断 ID (From Track A)
            - final_diagnosis_name: 初始诊断名称 (From Track A)
            - differential_reasoning: 鉴别诊断推理 (From Track A)
            - calculated_confidence: 置信度 (From Track A)
            - track_b_output: Track B 完整输出
                - problem_representation_one_liner: 一句话总结
                - p_nodes: P-Nodes 列表
            - raw_narrative: 原始文本（保留用于后续 Phase）
            - error: 错误信息
        """
        print("=" * 60)
        print("[Phase1 Manager] Starting Dual-Track Processing...")
        if global_hint:
            print(f"[Phase1 Manager] Teacher Hint injected (length: {len(global_hint)})")
        print("=" * 60)
        
        # ==================== Track A: Intuition ====================
        print("\n[Phase1 Manager] Executing Track A (Intuition)...")
        track_a_result = self.track_a.process(raw_narrative, global_hint=global_hint)
        
        if track_a_result.get("error"):
            print(f"[Phase1 Manager] Track A failed: {track_a_result['error']}")
            return {
                "top_candidates": [],
                "final_diagnosis_id": None,
                "final_diagnosis_name": None,
                "differential_reasoning": "",
                "calculated_confidence": 0.0,
                "track_b_output": None,
                "raw_narrative": raw_narrative,
                "error": f"Track A failed: {track_a_result['error']}",
                "raw_response": track_a_result.get("raw_response", "")  # 保存原始响应用于调试
            }
        
        print(f"[Phase1 Manager] Track A complete. Top candidates: {track_a_result.get('top_candidates', [])}")
        
        # ==================== Track B: Perception ====================
        print("\n[Phase1 Manager] Executing Track B (Perception)...")
        track_b_result = self.track_b.process(raw_narrative, age=age, sex=sex, global_hint=global_hint)
        
        if track_b_result.get("error"):
            print(f"[Phase1 Manager] Track B failed: {track_b_result['error']}")
            # Track B 失败不是致命错误，继续但警告
            print("[Phase1 Manager] WARNING: Proceeding without P-Nodes from Track B")
            track_b_result = {
                "problem_representation_one_liner": "",
                "p_nodes": [],
                "error": track_b_result.get("error")
            }
        else:
            p_nodes_count = len(track_b_result.get("p_nodes", []))
            print(f"[Phase1 Manager] Track B complete. Extracted {p_nodes_count} P-Nodes")
        
        # ==================== Merge Results ====================
        print("\n[Phase1 Manager] Merging Track A and Track B results...")
        
        merged_result = {
            # From Track A
            "top_candidates": track_a_result.get("top_candidates", []),
            "final_diagnosis_id": track_a_result.get("final_diagnosis_id"),
            "final_diagnosis_name": self._get_diagnosis_name(track_a_result.get("final_diagnosis_id")),
            "differential_reasoning": track_a_result.get("differential_reasoning", ""),
            "calculated_confidence": track_a_result.get("calculated_confidence", 0.0),
            
            # From Track B (完整保留)
            "track_b_output": {
                "problem_representation_one_liner": track_b_result.get("problem_representation_one_liner", ""),
                "p_nodes": track_b_result.get("p_nodes", [])
            },
            
            # 保留原始文本
            "raw_narrative": raw_narrative,
            
            # 错误信息（如果 Track B 有警告）
            "error": None,
            "track_b_warning": track_b_result.get("error")
        }
        
        print(f"[Phase1 Manager] Phase 1 complete.")
        print(f"  - Top-1 Diagnosis: {merged_result['final_diagnosis_name']} (ID: {merged_result['final_diagnosis_id']})")
        print(f"  - P-Nodes extracted: {len(merged_result['track_b_output']['p_nodes'])}")
        print("=" * 60)
        
        return merged_result
    
    def _get_diagnosis_name(self, diagnosis_id: Optional[str]) -> Optional[str]:
        """
        根据诊断 ID 获取诊断名称
        
        Args:
            diagnosis_id: 诊断 ID
        
        Returns:
            诊断名称，如果 ID 无效则返回 None
        """
        if not diagnosis_id:
            return None
        
        from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
        return DIAGNOSIS_ID_MAP.get(diagnosis_id)

