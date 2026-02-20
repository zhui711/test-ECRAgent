"""
Phase 1 Track B Agent: Problem Representation
从原始病例中提取结构化 P-Nodes（患者特征）

根据算法流程图 Line 2:
P <- ExtractFeatures(T)  # Track B: Structured Patient Features

核心任务:
- 执行 Problem Representation（问题表征）
- 将患者叙述转换为医学语义限定词
- 输出结构化的 P-Nodes 列表
"""
import json
import re
from typing import Dict, Any, Optional, List
from src.utils.api_client import LLMClient
from src.utils.json_utils import parse_json_from_text, robust_parse_p_nodes, extract_json_field
from config.prompt_phase1_trackB import (
    PHASE1_TRACKB_SYSTEM_PROMPT,
    PHASE1_TRACKB_FEW_SHOT_MESSAGES,
    PHASE1_TRACKB_USER_PROMPT_TEMPLATE
)


class Phase1TrackBAgent:
    """
    Phase 1 Track B Agent: 负责 Problem Representation
    
    将患者的原始叙述转换为结构化的 P-Nodes，使用医学语义限定词。
    这是 System 2 (Graph Reasoning) 的基础输入。
    """
    
    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):
        """
        初始化 Phase1TrackBAgent
        
        Args:
            llm_client: LLM 客户端实例
            model_name: 模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name
    
    def _extract_age_sex_from_narrative(self, narrative: str) -> tuple:
        """
        从 narrative 中提取年龄和性别
        
        Args:
            narrative: 患者叙述文本
        
        Returns:
            (age, sex) 元组，如果无法提取则返回默认值
        """
        # 尝试提取年龄
        age = 0
        age_patterns = [
            r'(\d+)[\s-]*year[\s-]*old',
            r'(\d+)[\s-]*yo\b',
            r'age[\s:]*(\d+)',
            r'aged[\s:]*(\d+)',
        ]
        for pattern in age_patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                break
        
        # 尝试提取性别
        sex = "Unknown"
        sex_patterns = [
            (r'\b(male|man|boy|gentleman)\b', 'M'),
            (r'\b(female|woman|girl|lady)\b', 'F'),
            (r'\bsex[\s:]*["\']?(M|F|male|female)["\']?', None),  # 需要特殊处理
        ]
        
        for pattern, default_sex in sex_patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                if default_sex:
                    sex = default_sex
                else:
                    matched_sex = match.group(1).upper()
                    sex = 'M' if matched_sex in ['M', 'MALE'] else 'F'
                break
        
        return age, sex
    
    def process(
        self, 
        raw_narrative: str, 
        age: int = None, 
        sex: str = None,
        global_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行 Problem Representation
        
        根据算法流程图 Line 2:
        P <- ExtractFeatures(T)
        
        Args:
            raw_narrative: 原始患者叙述文本
            age: 年龄（可选，如果未提供则从 narrative 提取）
            sex: 性别（可选，如果未提供则从 narrative 提取）
            global_hint: Teacher 反馈（用于 Offline Training 重试）
        
        Returns:
            包含以下字段的字典:
            - problem_representation_one_liner: 一句话总结
            - p_nodes: P-Nodes 列表
            - error: 错误信息（如果有）
        """
        print("[Phase1 Track B] Starting Problem Representation...")
        
        # 如果未提供 age/sex，从 narrative 中提取
        if age is None or sex is None:
            extracted_age, extracted_sex = self._extract_age_sex_from_narrative(raw_narrative)
            age = age if age is not None else extracted_age
            sex = sex if sex is not None else extracted_sex
        
        # 构建 System Prompt（可能附加 Teacher Hint）
        system_prompt = PHASE1_TRACKB_SYSTEM_PROMPT
        if global_hint and global_hint.strip():
            system_prompt = system_prompt + f"\n\n[TEACHER FEEDBACK - IMPORTANT]:\n{global_hint}"
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加 Few-Shot 示例（作为 user/assistant 对）
        messages.append({
            "role": "user",
            "content": f"Here are some examples of Problem Representation:\n{PHASE1_TRACKB_FEW_SHOT_MESSAGES}"
        })
        messages.append({
            "role": "assistant",
            "content": "I understand. I will perform Problem Representation on the new case, translating raw text into Medical Semantic Qualifiers and outputting structured P-Nodes."
        })
        
        # 添加实际用户请求
        user_prompt = PHASE1_TRACKB_USER_PROMPT_TEMPLATE.format(
            age=age,
            sex=sex,
            narrative=raw_narrative
        )
        messages.append({"role": "user", "content": user_prompt})
        
        # 调用 LLM
        result = self.llm_client.generate_json(
            messages=messages,
            model=self.model_name,
            logprobs=False,
            temperature=0.2,  # 使用 0.2 以获得更稳定的输出
            max_tokens=4096
        )
        
        if result["error"]:
            print(f"[Phase1 Track B] LLM error: {result['error']}")
            return {
                "problem_representation_one_liner": "",
                "p_nodes": [],
                "error": result["error"]
            }
        
        # 解析 JSON（使用增强的解析策略）
        response_content = result["content"]
        print(f"[Phase1 Track B] LLM response length: {len(response_content)} chars")
        
        # 策略 1: 尝试完整 JSON 解析
        parsed_json = parse_json_from_text(response_content, verbose=True)
        
        p_nodes = []
        one_liner = ""
        
        if parsed_json is not None and isinstance(parsed_json, dict):
            p_nodes = parsed_json.get("p_nodes", [])
            one_liner = parsed_json.get("problem_representation_one_liner", "")
        
        # 策略 2: 如果 p_nodes 为空，使用专门的 P-Nodes 解析器
        if not p_nodes:
            print(f"[Phase1 Track B] Trying robust P-Nodes extraction...")
            p_nodes = robust_parse_p_nodes(response_content, verbose=True)
        
        # 策略 3: 如果 one_liner 为空，尝试单独提取
        if not one_liner:
            one_liner = extract_json_field(response_content, "problem_representation_one_liner", verbose=False)
            if one_liner:
                print(f"[Phase1 Track B] Extracted one_liner via field extraction")
            else:
                one_liner = ""
        
        # 检查是否完全失败
        if not p_nodes:
            print(f"[Phase1 Track B] Failed to extract any P-Nodes")
            print(f"[Phase1 Track B] Raw response preview: {response_content[:500]}...")
            return {
                "problem_representation_one_liner": one_liner,
                "p_nodes": [],
                "error": "Failed to parse JSON from response"
            }
        
        # 验证和规范化 P-Nodes
        p_nodes = self._validate_p_nodes(p_nodes)
        
        print(f"[Phase1 Track B] Extracted {len(p_nodes)} P-Nodes")
        print(f"[Phase1 Track B] One-liner: {one_liner[:100]}...")
        
        return {
            "problem_representation_one_liner": one_liner,
            "p_nodes": p_nodes,
            "error": None
        }
    
    def _validate_p_nodes(self, p_nodes_raw: List[Any]) -> List[Dict[str, Any]]:
        """
        验证和规范化 P-Nodes
        
        确保每个 P-Node 有必要的字段:
        - id: 唯一标识符
        - content: 医学语义限定词
        - original_text: 原始文本片段
        - status: Present 或 Absent
        
        Args:
            p_nodes_raw: 原始 P-Nodes 列表
        
        Returns:
            规范化后的 P-Nodes 列表
        """
        validated_nodes = []
        
        if not isinstance(p_nodes_raw, list):
            return []
        
        for idx, node in enumerate(p_nodes_raw):
            if not isinstance(node, dict):
                continue
            
            # 确保有 ID
            node_id = node.get("id", f"p_{idx + 1}")
            if not node_id.startswith("p_"):
                node_id = f"p_{idx + 1}"
            
            # 确保有 content
            content = node.get("content", "")
            if not content:
                continue
            
            # 确保有 status
            status = node.get("status", "Present")
            if status not in ["Present", "Absent"]:
                status = "Present"
            
            # 确保有 original_text
            original_text = node.get("original_text", "")
            
            validated_nodes.append({
                "id": node_id,
                "content": content,
                "original_text": original_text,
                "status": status
            })
        
        return validated_nodes

