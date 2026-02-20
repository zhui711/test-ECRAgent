"""
Phase 1 Agent: Perception & Hypothesis
从原始病例中提取结构化症状，生成 Top-5 鉴别诊断，并计算置信度
"""
import json
import math
import re
from typing import Dict, Any, Optional, List
from src.utils.api_client import LLMClient
from src.utils.json_utils import parse_json_from_text, extract_json_field
from config.prompt_phase1_trackA_templates import PHASE1_TRACKA_SYSTEM_PROMPT, PHASE1_TRACKA_FEW_SHOT_MESSAGES, build_phase1_tracka_user_prompt
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP, normalize_diagnosis_id, normalize_diagnosis_id_list


def normalize_diagnosis_id(raw_id: Any) -> Optional[str]:
    """
    规范化诊断 ID，处理异常格式如 "045"
    
    有效 ID 范围: "01" - "49" (固定两位数)
    
    处理策略:
    - "45" -> "45" (正常)
    - "045" -> 尝试匹配 "04" 或 "45"，优先选择有效的
    - "5" -> "05" (补零)
    - 45 (int) -> "45" (转字符串)
    
    Args:
        raw_id: 原始 ID（可能是字符串、整数或其他）
    
    Returns:
        规范化后的 ID（两位数字字符串），如果无效返回 None
    """
    if raw_id is None:
        return None
    
    # 转为字符串并清理
    id_str = str(raw_id).strip().lstrip('0') or '0'
    
    # 如果是纯数字
    if id_str.isdigit():
        num = int(id_str)
        if 1 <= num <= 49:
            return f"{num:02d}"
    
    # 处理异常格式如 "045"
    raw_str = str(raw_id).strip()
    if len(raw_str) == 3 and raw_str.isdigit():
        # 尝试两种拆分方式
        option1 = raw_str[:2]  # "04" + "5"
        option2 = raw_str[1:]  # "0" + "45"
        
        # 检查哪个是有效的
        valid_options = []
        for opt in [option1, option2]:
            normalized = opt.lstrip('0') or '0'
            if normalized.isdigit():
                num = int(normalized)
                if 1 <= num <= 49:
                    valid_options.append(f"{num:02d}")
        
        if len(valid_options) == 1:
            print(f"[ID Normalize] Corrected '{raw_str}' -> '{valid_options[0]}'")
            return valid_options[0]
        elif len(valid_options) > 1:
            # 如果两个都有效（如 "045" -> "04" 和 "45" 都有效），优先选择常见的
            # 这种情况下，我们选择第二个（通常是用户想表达的完整数字）
            print(f"[ID Normalize] Ambiguous '{raw_str}', choosing '{valid_options[1]}' over '{valid_options[0]}'")
            return valid_options[1]
    
    print(f"[ID Normalize] Invalid ID '{raw_id}', returning None")
    return None


def normalize_candidate_list(candidates: List[Any]) -> List[str]:
    """
    规范化候选诊断 ID 列表
    
    Args:
        candidates: 原始候选列表
    
    Returns:
        规范化后的 ID 列表（去重、过滤无效值）
    """
    normalized = []
    seen = set()
    
    for raw_id in candidates:
        norm_id = normalize_diagnosis_id(raw_id)
        if norm_id and norm_id not in seen:
            normalized.append(norm_id)
            seen.add(norm_id)
    
    return normalized

class Phase1TrackAAgent:
    """
    Phase 1 Track A Agent: 负责症状提取和初步诊断
    """
    
    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):
        self.llm_client = llm_client
        self.model_name = model_name
    
    def extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        从响应文本中提取 JSON（使用增强的解析器）
        
        Args:
            response_text: LLM 返回的文本
        
        Returns:
            解析后的 JSON 字典，如果解析失败则返回 None
        """
        # 使用增强的 JSON 解析器
        result = parse_json_from_text(response_text, verbose=True)
        
        if isinstance(result, dict):
            return result
        
        # Fallback 1: 如果解析结果是数组，尝试将其作为 top_candidates
        if isinstance(result, list):
            # 检查是否是有效的诊断 ID 数组
            valid_ids = []
            for item in result:
                norm_id = normalize_diagnosis_id(item)
                if norm_id:
                    valid_ids.append(norm_id)
            
            if valid_ids:
                print(f"[Phase1 Track A] Parsed array as top_candidates: {valid_ids}")
                # 尝试从原文中提取 final_diagnosis_id
                final_diagnosis_id = extract_json_field(response_text, 'final_diagnosis_id', verbose=False)
                if final_diagnosis_id:
                    final_id = normalize_diagnosis_id(final_diagnosis_id)
                else:
                    final_id = valid_ids[0]  # 使用数组第一个作为 fallback
                
                differential_reasoning = extract_json_field(response_text, 'differential_reasoning', verbose=False)
                
                return {
                    "top_candidates": valid_ids,
                    "final_diagnosis_id": final_id,
                    "differential_reasoning": differential_reasoning or ""
                }
        
        # Fallback 2: 如果解析失败，尝试降级提取关键字段
        if result is None:
            # 尝试至少提取 top_candidates 和 final_diagnosis_id
            top_candidates = extract_json_field(response_text, 'top_candidates', verbose=False)
            final_diagnosis_id = extract_json_field(response_text, 'final_diagnosis_id', verbose=False)
            differential_reasoning = extract_json_field(response_text, 'differential_reasoning', verbose=False)
            
            if top_candidates or final_diagnosis_id:
                print("[Phase1 Track A] Using field extraction fallback")
                return {
                    "top_candidates": top_candidates if isinstance(top_candidates, list) else [],
                    "final_diagnosis_id": str(final_diagnosis_id) if final_diagnosis_id else None,
                    "differential_reasoning": differential_reasoning or ""
                }
        
        return None
    
    def calculate_confidence_from_logprobs(
        self,
        response_obj: Any,
        final_diagnosis_id: str,
        response_text: str
    ) -> float:
        """
        从 logprobs 计算置信度（Token序列重建方案）
        
        改进策略：
        1. 重建完整 token 序列，精确定位 "final_diagnosis_id" 字段值对应的 token
        2. 如果 diagnosis_id 是多 token，计算联合概率（sum of logprobs）
        3. 使用反向查找确保定位到 JSON 结尾处的 final_diagnosis_id
        
        Args:
            response_obj: OpenAI API 返回的 completion 对象
            final_diagnosis_id: 最终诊断 ID（例如 "15" 或 "23"）
            response_text: 响应文本内容
        
        Returns:
            置信度（0-100）
        """
        if response_obj is None:
            return 0.0
        
        try:
            logprobs_content = response_obj.choices[0].logprobs.content
            
            if not logprobs_content:
                return 0.0
            
            diagnosis_id_str = str(final_diagnosis_id)
            
            # 1. 重建完整文本并映射 Token 边界
            reconstructed_text = ""
            token_boundaries = []
            for token_info in logprobs_content:
                start_pos = len(reconstructed_text)
                reconstructed_text += token_info.token
                end_pos = len(reconstructed_text)
                token_boundaries.append((start_pos, end_pos, token_info))
            
            # 2. 在重建文本中精确定位 final_diagnosis_id 的值
            # 使用反向查找确保找到的是 JSON 结尾处的那个（避免匹配到 top_candidates 中的 ID）
            field_marker = '"final_diagnosis_id"'
            field_start_idx = reconstructed_text.rfind(field_marker)
            
            if field_start_idx == -1:
                return 0.0
            
            # 从字段名后开始找值，使用正则表达式精确定位
            suffix_text = reconstructed_text[field_start_idx + len(field_marker):]
            
            # 尝试匹配带引号的格式: "final_diagnosis_id": "XX"
            pattern_quoted = rf'\s*:\s*"({re.escape(diagnosis_id_str)})"'
            match_quoted = re.search(pattern_quoted, suffix_text)
            
            if match_quoted:
                # 计算值在原文本中的绝对位置
                val_start_abs = field_start_idx + len(field_marker) + match_quoted.start(1)
                val_end_abs = field_start_idx + len(field_marker) + match_quoted.end(1)
            else:
                # 尝试不带引号的格式: "final_diagnosis_id": XX
                pattern_unquoted = rf'\s*:\s*({re.escape(diagnosis_id_str)})'
                match_unquoted = re.search(pattern_unquoted, suffix_text)
                
                if match_unquoted:
                    val_start_abs = field_start_idx + len(field_marker) + match_unquoted.start(1)
                    val_end_abs = field_start_idx + len(field_marker) + match_unquoted.end(1)
                else:
                    # 如果正则匹配失败，尝试简单查找（作为最后备用方案）
                    val_start_relative = suffix_text.find(diagnosis_id_str)
                    if val_start_relative == -1:
                        return 0.0
                    val_start_abs = field_start_idx + len(field_marker) + val_start_relative
                    val_end_abs = val_start_abs + len(diagnosis_id_str)
            
            # 3. 收集覆盖该值的所有 Token（包括引号、空格等）
            target_tokens = []
            token_positions = []  # 记录每个token的位置信息用于debug
            for start, end, token_info in token_boundaries:
                # Token 与值的区间有重叠：Token结束位置 > 值开始位置 AND Token开始位置 < 值结束位置
                if end > val_start_abs and start < val_end_abs:
                    target_tokens.append(token_info)
                    token_positions.append((start, end, token_info.token))
            
            if not target_tokens:
                print(f"[DEBUG] No tokens found for diagnosis_id '{diagnosis_id_str}'")
                return 0.0
            
            # DEBUG: 打印所有候选token（包括引号、空格等）
            print(f"[DEBUG] Value position: {val_start_abs} to {val_end_abs}")
            print(f"[DEBUG] All candidate tokens (overlapping with value position):")
            for idx, (start, end, token) in enumerate(token_positions):
                token_info = target_tokens[idx]
                print(f"  Token: {repr(token)}, Logprob: {token_info.logprob}, Position: {start}-{end}")
            
            # 4. 尝试找到能组成 diagnosis_id_str 的最小token集合
            # 策略：按位置排序，然后尝试所有可能的连续token组合
            sorted_indices = sorted(range(len(target_tokens)), 
                                  key=lambda i: token_positions[i][0])
            
            # 尝试找到匹配的token组合
            matched_tokens = None
            for i in range(len(sorted_indices)):
                for j in range(i + 1, len(sorted_indices) + 1):
                    # 尝试从第i个到第j个token的组合
                    candidate_tokens = [target_tokens[sorted_indices[k]] for k in range(i, j)]
                    reconstructed_value = "".join(t.token for t in candidate_tokens)
                    cleaned_value = reconstructed_value.strip('"').strip("'").strip()
                    
                    if cleaned_value == diagnosis_id_str:
                        matched_tokens = candidate_tokens
                        break
                if matched_tokens:
                    break
            
            # 如果没找到匹配的组合，使用所有token（可能是多token情况）
            if not matched_tokens:
                matched_tokens = target_tokens
            
            # DEBUG: 打印最终使用的token信息
            tokens_repr = [repr(t.token) for t in matched_tokens]
            logprobs_list = [t.logprob for t in matched_tokens]
            reconstructed_value = "".join(t.token for t in matched_tokens)
            cleaned_value = reconstructed_value.strip('"').strip("'").strip()
            
            print(f"[DEBUG] Final diagnosis tokens: {tokens_repr} with logprobs: {logprobs_list}")
            print(f"[DEBUG] Reconstructed value: {repr(reconstructed_value)}, Cleaned: {repr(cleaned_value)}, Expected: {repr(diagnosis_id_str)}")
            
            if cleaned_value != diagnosis_id_str:
                # 验证失败，返回 0（避免误匹配）
                print(f"[DEBUG] Validation failed: cleaned_value '{cleaned_value}' != diagnosis_id_str '{diagnosis_id_str}'")
                return 0.0
            
            # 5. 计算联合概率 sum(logprobs)
            sum_logprob = sum(t.logprob for t in matched_tokens)
            print(f"[DEBUG] Sum of logprobs: {sum_logprob}, Exp(sum_logprob): {math.exp(sum_logprob)}")
            
            # 6. 转换为置信度
            confidence = math.exp(sum_logprob) * 100
            print(f"[DEBUG] Calculated confidence: {confidence}%")
            return max(0.0, min(100.0, confidence))
            
        except Exception as e:
            print(f"Error calculating confidence from logprobs: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def process(self, patient_narrative: str, global_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        处理病例，返回 Phase 1 结果
        
        Args:
            patient_narrative: 患者叙述文本
            global_hint: Teacher 反馈（用于 Offline Training 重试）
        
        Returns:
            包含以下字段的字典:
            - structured_analysis: 结构化分析结果
            - top_candidates: Top-3 候选诊断
            - final_diagnosis_id: 最终诊断 ID
            - calculated_confidence: 计算的置信度
            - error: 错误信息（如果有）
        """
        # 构建 System Prompt（可能附加 Teacher Hint）
        system_prompt = PHASE1_TRACKA_SYSTEM_PROMPT
        if global_hint and global_hint.strip():
            system_prompt = system_prompt + f"\n\n[TEACHER FEEDBACK - IMPORTANT]:\n{global_hint}"
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加 Few-Shot 示例
        messages.extend(PHASE1_TRACKA_FEW_SHOT_MESSAGES)
        
        # 添加用户消息
        user_prompt = build_phase1_tracka_user_prompt(patient_narrative)
        messages.append({"role": "user", "content": user_prompt})
        
        # 调用 LLM
        result = self.llm_client.generate_json(
            messages=messages,
            model=self.model_name,
            # logprobs=True,
            # top_logprobs=5,  cy：先不使用 logprobs，全部进入phase2
            temperature=0.0,
            max_tokens=4096
        )
        
        if result["error"]:
            return {
                "structured_analysis": None,
                "top_candidates": [],
                "final_diagnosis_id": None,
                "calculated_confidence": 0.0,
                "error": result["error"]
            }
        
        # 提取 JSON
        response_content = result["content"]
        parsed_json = self.extract_json_from_response(response_content)
        
        if parsed_json is None:
            return {
                "structured_analysis": None,
                "top_candidates": [],
                "final_diagnosis_id": None,
                "calculated_confidence": 0.0,
                "error": "Failed to parse JSON from response",
                "raw_response": response_content[:2000] if response_content else ""  # 保存原始响应用于调试
            }
        
        # 获取并规范化 final_diagnosis_id
        raw_final_id = parsed_json.get("final_diagnosis_id")
        final_diagnosis_id = normalize_diagnosis_id(raw_final_id)
        
        # 获取并规范化 top_candidates
        raw_candidates = parsed_json.get("top_candidates", [])
        top_candidates = normalize_candidate_list(raw_candidates)
        
        if not final_diagnosis_id:
            # 尝试从 top_candidates 中获取第一个作为 fallback
            if top_candidates:
                final_diagnosis_id = top_candidates[0]
                print(f"[Phase1 Track A] Using first candidate '{final_diagnosis_id}' as fallback for missing final_diagnosis_id")
            else:
                return {
                    "structured_analysis": parsed_json.get("structured_analysis"),
                    "top_candidates": top_candidates,
                    "final_diagnosis_id": None,
                    "calculated_confidence": 0.0,
                    "error": "final_diagnosis_id not found in response",
                    "raw_response": response_content[:2000] if response_content else ""
                }
        
        # 确保 final_diagnosis_id 在 top_candidates 中
        # if final_diagnosis_id not in top_candidates and top_candidates:
        #     # 如果不在，添加到首位
        #     top_candidates.insert(0, final_diagnosis_id)
        #     print(f"[Phase1 Track A] Added final_diagnosis_id '{final_diagnosis_id}' to top_candidates")
        
        # 计算置信度
        # confidence = self.calculate_confidence_from_logprobs(
        #     result["response"],
        #     final_diagnosis_id,
        #     response_content
        # )
        confidence = 0.0  # cy：先不计算置信度，全部进入phase2
        
        return {
            "structured_analysis": parsed_json.get("structured_analysis"),
            "differential_reasoning": parsed_json.get("differential_reasoning"),
            "top_candidates": top_candidates,
            "final_diagnosis_id": final_diagnosis_id,
            "calculated_confidence": confidence,
            "error": None,
            "raw_response": None  # 成功时不需要保存
        }

