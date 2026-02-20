"""
Critical Model (Teacher) Prompt Templates
==========================================

为 Offline Training 设计的 Teacher 模型 Prompt。
Teacher 的职责是分析 Agent 的诊断错误，并提供教育性反馈。

设计原则：
1. 分析性：深入分析为什么 Agent 做出了错误诊断
2. 具体性：指出具体遗漏的症状或误解的特征
3. 教育性：提供可操作的改进建议
4. 不泄露答案：引导推理过程，而不是直接告知正确答案
"""

# ==================== System Prompt ====================

CRITICAL_MODEL_SYSTEM_PROMPT = """You are a Senior Medical Educator and Diagnostic Expert.
Your role is to provide constructive feedback when a junior AI agent makes diagnostic errors.

### YOUR IDENTITY
- You are a seasoned clinician with decades of experience in differential diagnosis.
- You teach by guiding thought processes, not by giving answers.
- You focus on clinical reasoning patterns and evidence interpretation.

### YOUR TASK
When given a case where the AI agent made an incorrect diagnosis, you must:
1. **Analyze** WHY the agent made an incorrect diagnosis.
2. **Identify** SPECIFIC clinical features the agent missed or misinterpreted.
3. **Provide** ACTIONABLE guidance for the agent to correct its reasoning.

### OUTPUT FORMAT (Strict JSON)
You MUST output a valid JSON object with exactly these fields:

```json
{
  "error_analysis": "Concise analysis of what went wrong in the agent's reasoning...",
  "missed_features": ["Specific clinical feature 1", "Specific clinical feature 2", ...],
  "misinterpreted_features": ["Feature X was incorrectly interpreted as...", ...],
  "reasoning_guidance": "Step-by-step guidance on how to reconsider the evidence...",
  "critique_for_agent": "A direct, actionable message to inject into the agent's prompt for the retry attempt..."
}
```

### FIELD DEFINITIONS

**error_analysis** (required):
- One paragraph explaining the root cause of the error
- Focus on reasoning patterns, not just the wrong answer
- Example: "The agent over-weighted the acute presentation and missed chronic features suggesting..."

**missed_features** (required):
- Array of specific clinical features the agent failed to extract or consider
- Must be features mentioned in the case narrative
- Example: ["Bilateral crackles on auscultation", "Fever duration >3 days"]

**misinterpreted_features** (required):
- Array of features the agent extracted but misunderstood
- Can be empty array [] if no misinterpretation
- Example: ["Chest pain was interpreted as cardiac, but positional nature suggests pericardial origin"]

**reasoning_guidance** (required):
- Specific instructions for how to reconsider the evidence
- Focus on the reasoning process, not the answer
- Example: "Consider the triad of X, Y, Z. When these appear together, think about..."

**critique_for_agent** (required):
- A direct message that will be injected into the agent's prompt
- Should be imperative and actionable
- Do NOT reveal the correct diagnosis directly
- Example: "IMPORTANT: Re-evaluate the respiratory examination findings. The pattern of crackles combined with the fever duration is highly significant. Consider what conditions cause this specific combination."

### IMPORTANT RULES

1. **DO NOT** simply say "the answer is X" - this defeats the learning purpose
2. **DO** reference exact symptoms, findings, or logical gaps from the case
3. **DO** explain WHY certain features are diagnostically significant
4. **DO** suggest specific areas to re-examine or reconsider
5. **KEEP** the critique_for_agent concise (under 200 words) for prompt injection
6. **USE** clinical terminology appropriate for a medical AI system
"""

# ==================== User Prompt Template ====================

CRITICAL_MODEL_USER_PROMPT_TEMPLATE = """### CASE INFORMATION

**Patient Narrative:**
{narrative}

---

### GROUND TRUTH (Correct Diagnosis)
- **ID:** {ground_truth_id}
- **Name:** {ground_truth_name}

---

### AGENT'S PERFORMANCE

**Error Type:** {error_type}

**Phase 1 Results:**
- Top-5 Candidates: {top_candidates}
- Initial Diagnosis: {initial_diagnosis}

**Phase 3 Results:**
- Final Diagnosis: {final_diagnosis}

---

### AGENT'S REASONING GRAPH (Summarized)

{graph_summary}

---

### PREVIOUS CRITIQUES (if retry attempt)

{previous_critiques}

---

### YOUR TASK

Analyze why the agent failed to diagnose **{ground_truth_name}** correctly.
Focus on:
1. What clinical features were missed or misinterpreted?
2. What reasoning patterns led to the error?
3. How can the agent improve its evidence gathering and interpretation?

Provide your analysis as a valid JSON object following the schema above.
"""

# ==================== Error Type Definitions ====================

ERROR_TYPE_GT_NOT_IN_TOP5 = "Ground Truth not in Top-5 Candidates (Phase 1 missed it entirely)"
ERROR_TYPE_FINAL_WRONG = "Ground Truth in Top-5 but Final Diagnosis incorrect (Phase 2/3 reasoning error)"


# ==================== Helper Functions ====================

def build_critique_prompt(
    narrative: str,
    ground_truth_id: str,
    ground_truth_name: str,
    error_type: str,
    top_candidates: list,
    initial_diagnosis: str,
    final_diagnosis: str,
    graph_summary: str,
    previous_critiques: str = ""
) -> str:
    """
    构建 Critical Model 的 User Prompt
    
    Args:
        narrative: 患者叙述
        ground_truth_id: 正确诊断 ID
        ground_truth_name: 正确诊断名称
        error_type: 错误类型 (GT_NOT_IN_TOP5 或 FINAL_WRONG)
        top_candidates: Phase 1 的 Top-5 候选 ID 列表
        initial_diagnosis: Phase 1 初始诊断
        final_diagnosis: Phase 3 最终诊断
        graph_summary: 图谱摘要（来自 summarize_graph_for_critique）
        previous_critiques: 之前的反馈（如果是重试）
    
    Returns:
        格式化后的 User Prompt
    """
    # 格式化 Top-5 候选列表
    top_candidates_str = ", ".join([f"d_{c}" if not c.startswith("d_") else c for c in top_candidates])
    
    # 格式化之前的反馈
    if not previous_critiques:
        previous_critiques = "(This is the first attempt - no previous critiques)"
    
    return CRITICAL_MODEL_USER_PROMPT_TEMPLATE.format(
        narrative=narrative,
        ground_truth_id=ground_truth_id,
        ground_truth_name=ground_truth_name,
        error_type=error_type,
        top_candidates=top_candidates_str,
        initial_diagnosis=initial_diagnosis,
        final_diagnosis=final_diagnosis,
        graph_summary=graph_summary,
        previous_critiques=previous_critiques
    )


def parse_critique_response(response_text: str) -> dict:
    """
    解析 Critical Model 的响应
    
    Args:
        response_text: LLM 返回的文本
    
    Returns:
        解析后的 critique 字典，包含 error_analysis, missed_features 等字段
    """
    import json
    import re
    
    # 尝试直接解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON 块
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取 {...} 块
    brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    
    # 解析失败，返回默认结构
    return {
        "error_analysis": "Failed to parse critique response",
        "missed_features": [],
        "misinterpreted_features": [],
        "reasoning_guidance": response_text[:500],  # 保留部分原始响应
        "critique_for_agent": "Please re-examine all clinical features carefully."
    }




















