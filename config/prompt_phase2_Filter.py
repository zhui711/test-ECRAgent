"""
Phase 2 Text Quality Filter (Batch Mode)
批量判断文本列表是否包含有价值的临床信息

重构版本 (Debug Enhancement):
- 支持批量过滤：一次性处理多个 Snippets
- 减少 LLM 调用次数：N 次 -> 1 次
- 严格输出 JSON 格式

目标：过滤掉以下类型的无用文本：
- 基金/资助信息
- 动物实验研究
- 纯行政/管理数据
- Case Report (个案报告，样本量=1)
- 缺乏临床表现描述的文本

保留以下类型的有价值文本：
- 临床表现/症状/体征描述
- 诊断标准/鉴别诊断
- 病理生理学机制
- Review/综述性内容
"""
from typing import List, Optional, Dict, Any
import json


PHASE2_BATCH_FILTER_SYSTEM_PROMPT = """You are a strict Medical Data Curator for a clinical decision support system.

You will receive a list of text snippets. For EACH snippet, independently determine if it contains USEFUL clinical information for diagnosis.

### ACCEPT (Output: true):
- Clinical presentation, signs, and symptoms
- Diagnostic criteria or differential diagnosis guidelines  
- Pathophysiology that helps understand disease manifestations
- Review articles summarizing clinical features
- Treatment guidelines that mention clinical indicators

### REJECT (Output: false):
- Funding acknowledgments or grant information
- Animal studies or in-vitro experiments only
- Administrative, policy, or health economics data
- Single case reports with n=1 (high noise, low generalizability)
- Purely molecular/genetic studies without clinical correlation
- Epidemiology statistics without symptom descriptions
- Drug advertisements or commercial content
- Text about a COMPLETELY DIFFERENT disease than the target

### CRITICAL RULES:
1. Process each snippet INDEPENDENTLY
2. Output MUST be a valid JSON object
3. The "results" array length MUST equal the number of input snippets
4. Use lowercase boolean values: true or false (NOT "YES"/"NO")"""


PHASE2_BATCH_FILTER_USER_PROMPT_TEMPLATE = """Evaluate the following {n} text snippets for clinical relevance.

For EACH snippet, determine if it contains useful clinical information (symptoms, signs, diagnosis criteria) relevant to the indicated target disease.

{snippets_section}

---

Output your evaluation as a JSON object with exactly {n} boolean values:

```json
{{
  "results": [true/false, true/false, ...]
}}
```

IMPORTANT: The "results" array must have exactly {n} elements, one for each snippet above."""


def build_batch_filter_prompt(snippets: List[Dict[str, str]]) -> str:
    """
    构建批量过滤的 User Prompt
    
    Args:
        snippets: 列表，每个元素是 {"disease": "疾病名称", "text": "文本内容"}
    
    Returns:
        格式化的 User Prompt
    """
    n = len(snippets)
    
    # 构建 snippets 段落
    snippet_lines = []
    for i, item in enumerate(snippets, 1):
        disease = item.get("disease", "Unknown")
        text = item.get("text", "")
        # 截断文本（每个 snippet 最多 2000 字符）
        if len(text) > 2000:
            text = text[:2000] + "... [truncated]"
        
        snippet_lines.append(f"--- SNIPPET {i} (Target Disease: {disease}) ---")
        snippet_lines.append(text)
        snippet_lines.append("")
    
    snippets_section = "\n".join(snippet_lines)
    
    return PHASE2_BATCH_FILTER_USER_PROMPT_TEMPLATE.format(
        n=n,
        snippets_section=snippets_section
    )


def parse_batch_filter_response(response: str, expected_count: int) -> List[bool]:
    """
    解析批量 Filter 的 LLM 响应
    
    Args:
        response: LLM 响应文本
        expected_count: 期望的结果数量
    
    Returns:
        布尔值列表，解析失败返回全 False（宁缺毋滥）
    """
    if not response:
        print("[BatchFilter] Empty response, defaulting to all False")
        return [False] * expected_count
    
    # 尝试解析 JSON
    try:
        # 清理响应（移除 markdown 代码块标记）
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # 移除开头的 ```json 或 ```
            lines = cleaned.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            cleaned = "\n".join(lines[start_idx:end_idx])
        
        data = json.loads(cleaned)
        
        # 提取 results 数组
        results = data.get("results", [])
        
        if not isinstance(results, list):
            print(f"[BatchFilter] 'results' is not a list, defaulting to all False")
            return [False] * expected_count
        
        # 验证长度
        if len(results) != expected_count:
            print(f"[BatchFilter] Length mismatch: got {len(results)}, expected {expected_count}")
            # 尝试填充或截断
            if len(results) < expected_count:
                results.extend([False] * (expected_count - len(results)))
            else:
                results = results[:expected_count]
        
        # 转换为布尔值
        bool_results = []
        for r in results:
            if isinstance(r, bool):
                bool_results.append(r)
            elif isinstance(r, str):
                bool_results.append(r.lower() in ["true", "yes", "1"])
            else:
                bool_results.append(bool(r))
        
        return bool_results
        
    except json.JSONDecodeError as e:
        print(f"[BatchFilter] JSON parse error: {e}")
        print(f"[BatchFilter] Response was: {response[:200]}...")
        return [False] * expected_count
    except Exception as e:
        print(f"[BatchFilter] Unexpected error: {e}")
        return [False] * expected_count


# ============================================================
# 向后兼容：保留旧的单条过滤接口
# ============================================================

PHASE2_FILTER_SYSTEM_PROMPT = """You are a strict medical data curator for a clinical decision support system.

Your task is to determine if a given text contains USEFUL clinical information for diagnosis.

### ACCEPT (Return YES):
- Clinical presentation, signs, and symptoms
- Diagnostic criteria or differential diagnosis guidelines
- Pathophysiology that helps understand disease manifestations
- Review articles summarizing clinical features
- Treatment guidelines that mention clinical indicators

### REJECT (Return NO):
- Funding acknowledgments or grant information
- Animal studies or in-vitro experiments only
- Administrative, policy, or health economics data
- Single case reports with n=1 (high noise, low generalizability)
- Purely molecular/genetic studies without clinical correlation
- Epidemiology statistics without symptom descriptions
- Drug advertisements or commercial content

### OUTPUT FORMAT
Respond with ONLY one word: YES or NO

Do not provide explanations. Just YES or NO."""

PHASE2_FILTER_USER_PROMPT_TEMPLATE = """Evaluate the following medical text:

---
{text_snippet}
---

Does this text contain useful clinical information (symptoms, signs, diagnosis criteria)?
Answer YES or NO only."""


def parse_filter_response(response: str) -> bool:
    """
    解析单条 Filter 响应（向后兼容）
    
    Args:
        response: LLM 响应文本
    
    Returns:
        True = 保留, False = 丢弃
    """
    if not response:
        return False  # 空响应默认丢弃
    
    # 清理响应
    cleaned = response.strip().upper()
    
    # 检查是否包含 YES
    if "YES" in cleaned:
        return True
    
    # 检查是否包含 NO
    if "NO" in cleaned:
        return False
    
    # 无法解析，默认丢弃（宁缺毋滥）
    return False
