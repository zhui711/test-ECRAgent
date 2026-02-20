"""
Critical Model Agent (Teacher)
==============================

Offline Training 的 Teacher 模型，负责：
1. 分析 Agent 的诊断错误
2. 生成教育性反馈 (Critique)
3. 提供可注入到 Agent Prompt 的改进建议

设计原则：
- 引导而非告知：不直接给出答案，而是引导推理
- 具体而非抽象：指出具体遗漏的症状或误解
- 教育性：解释为什么某些特征具有诊断意义
"""

from typing import Dict, Any, Optional, List
from src.utils.api_client import LLMClient
from src.utils.graph_tools import summarize_graph_for_critique
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from config.prompt_critical_model import (
    CRITICAL_MODEL_SYSTEM_PROMPT,
    build_critique_prompt,
    parse_critique_response,
    ERROR_TYPE_GT_NOT_IN_TOP5,
    ERROR_TYPE_FINAL_WRONG
)


class CriticalModelAgent:
    """
    Critical Model Agent (Teacher)
    
    在 Offline Training 中，当 Agent 做出错误诊断时，
    调用此 Agent 生成教育性反馈。
    
    Attributes:
        llm_client: LLM 客户端
        model_name: 模型名称（应使用高级模型如 gpt-5）
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = "gpt-5"
    ):
        """
        初始化 Critical Model Agent
        
        Args:
            llm_client: LLM 客户端实例
            model_name: Teacher 模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name
        
        # 历史 Critique 记录（用于多轮重试）
        self._critique_history: List[str] = []
    
    def generate_critique(
        self,
        narrative: str,
        ground_truth_id: str,
        phase1_result: Dict[str, Any],
        final_output: Dict[str, Any],
        graph_json: Dict[str, Any],
        previous_critiques: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        生成诊断错误的 Critique
        
        Args:
            narrative: 患者叙述
            ground_truth_id: 正确诊断 ID (不带 d_ 前缀)
            phase1_result: Phase 1 的输出
            final_output: Phase 3 的最终输出
            graph_json: Phase 2 的图谱 JSON
            previous_critiques: 之前的 Critique 列表（如果是重试）
        
        Returns:
            Critique 字典，包含：
            - error_analysis: 错误分析
            - missed_features: 遗漏的特征
            - misinterpreted_features: 误解的特征
            - reasoning_guidance: 推理指导
            - critique_for_agent: 可注入的反馈文本
            - error: 错误信息（如果调用失败）
        """
        # 1. 确定错误类型
        top_candidates = phase1_result.get("top_candidates", [])
        initial_diagnosis_id = phase1_result.get("final_diagnosis_id", "")
        final_diagnosis_id = final_output.get("final_diagnosis_id", "")
        
        # 检查 Ground Truth 是否在 Top-5 中
        gt_id_variants = [ground_truth_id, f"d_{ground_truth_id}", ground_truth_id.lstrip("d_")]
        gt_in_top5 = any(
            gt_var in top_candidates or gt_var.lstrip("d_") in [c.lstrip("d_") for c in top_candidates]
            for gt_var in gt_id_variants
        )
        
        if not gt_in_top5:
            error_type = ERROR_TYPE_GT_NOT_IN_TOP5
        else:
            error_type = ERROR_TYPE_FINAL_WRONG
        
        # 2. 获取名称
        ground_truth_name = DIAGNOSIS_ID_MAP.get(
            ground_truth_id.lstrip("d_"), 
            "Unknown Diagnosis"
        )
        initial_diagnosis_name = DIAGNOSIS_ID_MAP.get(
            initial_diagnosis_id.lstrip("d_"),
            "Unknown"
        )
        final_diagnosis_name = final_output.get(
            "final_diagnosis_name",
            DIAGNOSIS_ID_MAP.get(final_diagnosis_id.lstrip("d_"), "Unknown")
        )
        
        # 3. 生成图谱摘要（面向纠错）
        graph_summary = summarize_graph_for_critique(graph_json, ground_truth_id)
        
        # 4. 格式化之前的 Critiques
        previous_critiques_text = ""
        if previous_critiques:
            for i, critique in enumerate(previous_critiques, 1):
                previous_critiques_text += f"\n--- Attempt {i} Critique ---\n{critique}\n"
        
        # 5. 构建 Prompt
        user_prompt = build_critique_prompt(
            narrative=narrative,
            ground_truth_id=ground_truth_id,
            ground_truth_name=ground_truth_name,
            error_type=error_type,
            top_candidates=top_candidates,
            initial_diagnosis=initial_diagnosis_name,
            final_diagnosis=final_diagnosis_name,
            graph_summary=graph_summary,
            previous_critiques=previous_critiques_text
        )
        
        messages = [
            {"role": "system", "content": CRITICAL_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # 6. 调用 LLM
        try:
            result = self.llm_client.generate_json(
                messages=messages,
                model=self.model_name,
                temperature=0.3,  # 稍微有点创造性，但不要太发散
                max_tokens=2048
            )
            
            if result.get("error"):
                return {
                    "error": f"LLM call failed: {result['error']}",
                    "critique_for_agent": "Please re-examine all clinical features carefully."
                }
            
            # 7. 解析响应
            critique = parse_critique_response(result["content"])
            
            print(f"[CriticalModel] Generated critique for {ground_truth_name}")
            print(f"    Error Type: {error_type}")
            print(f"    Missed Features: {critique.get('missed_features', [])}")
            
            return critique
            
        except Exception as e:
            print(f"[CriticalModel] Error generating critique: {e}")
            return {
                "error": str(e),
                "critique_for_agent": "Please re-examine all clinical features carefully."
            }
    
    def get_injectable_critique(self, critique: Dict[str, Any]) -> str:
        """
        获取可注入到 Agent Prompt 的 Critique 文本
        
        Args:
            critique: generate_critique 的返回值
        
        Returns:
            可直接注入 Prompt 的文本
        """
        # 优先使用 critique_for_agent 字段
        if "critique_for_agent" in critique:
            return critique["critique_for_agent"]
        
        # Fallback: 构建简化版本
        parts = []
        
        if critique.get("missed_features"):
            features = ", ".join(critique["missed_features"][:5])
            parts.append(f"Consider these potentially missed features: {features}")
        
        if critique.get("reasoning_guidance"):
            parts.append(critique["reasoning_guidance"][:300])
        
        if not parts:
            return "Please re-examine all clinical features carefully."
        
        return " ".join(parts)
    
    def reset_history(self) -> None:
        """重置 Critique 历史（开始处理新案例时调用）"""
        self._critique_history = []
    
    def add_to_history(self, critique: str) -> None:
        """添加 Critique 到历史"""
        self._critique_history.append(critique)
    
    def get_history(self) -> List[str]:
        """获取 Critique 历史"""
        return self._critique_history.copy()


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 简单测试（需要有效的 API Key）
    print("CriticalModelAgent module loaded successfully.")
    print(f"Error types defined:")
    print(f"  - {ERROR_TYPE_GT_NOT_IN_TOP5}")
    print(f"  - {ERROR_TYPE_FINAL_WRONG}")




















