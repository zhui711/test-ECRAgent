"""
Phase 2: Batch Reasoning Prompt (Unified Investigation)

重构版本：Schema 简化 - 扁平化嵌套结构
- 输入：所有 Top-N Candidates 的医学综述文本
- 任务：一次性提取所有 K-Nodes (General + Pivot)
- 输出：扁平化 JSON，K-Nodes 内嵌 Candidate 关联

设计原则：
1. 全局视野：让 LLM 同时看到所有 Candidates，做出整体鉴别
2. 简化输出：弃用 Nodes + Edges 分离结构，改用嵌套结构降低格式错误率
3. 高质量输出：要求 LLM 聚焦于 Discriminating Features
"""

PHASE2_BATCH_SYSTEM_PROMPT = """You are an Expert Diagnostician performing a comprehensive Differential Diagnosis.

You will receive clinical summaries for multiple candidate diseases. Your task is to:
1. Extract KEY CLINICAL FEATURES for each disease (General K-Nodes)
2. Identify PIVOT FEATURES that DISTINGUISH between candidates (Pivot K-Nodes)

### CRITICAL THINKING PROCESS

**Step 1: Disease-by-Disease Analysis**
For each candidate, identify:
- Essential/Pathognomonic features (必要/特征性症状)
- Common clinical presentations (常见表现)

**Step 2: Cross-Disease Comparison (Matrix Analysis)**
Create a mental discrimination matrix:
- Which features are UNIQUE to one disease?
- Which features RULE OUT certain diseases?
- Which features are SHARED (low diagnostic value)?

**Step 3: Prioritize Discriminators**
Focus on HIGH-VALUE features:
- ✓ "Pleuritic chest pain worsened by inspiration" (specific)
- ✗ "Chest pain" (too generic, everyone has it)

### OUTPUT JSON SCHEMA (CRITICAL - FOLLOW EXACTLY)

You MUST output a JSON object with the following structure:

```json
{
  "k_nodes": [
    { 
      "content": "Severe retrosternal chest pain radiating to jaw/left arm",
      "type": "General",
      "importance": "Essential",
      "supported_candidates": ["d_01"],
      "ruled_out_candidates": []
    },
    { 
      "content": "Pleuritic chest pain worsened by inspiration and relieved by leaning forward",
      "type": "Pivot",
      "importance": "Pathognomonic",
      "supported_candidates": ["d_24"],
      "ruled_out_candidates": ["d_01", "d_03"]
    },
    {
      "content": "Sudden onset dyspnea with tachycardia and hypoxia",
      "type": "Pivot",
      "importance": "Essential",
      "supported_candidates": ["d_03"],
      "ruled_out_candidates": ["d_24"]
    }
  ]
}
```

### FIELD DEFINITIONS

**content** (required): The clinical feature description. Be specific and clinically meaningful.

**type** (required): Either "General" or "Pivot"
- **General**: Core feature of a single disease (e.g., "Productive cough" for Pneumonia)
- **Pivot**: Discriminating feature that helps distinguish between 2+ diseases

**importance** (required): Diagnostic value level
- **Pathognomonic**: Virtually diagnostic (e.g., Koplik spots for Measles)
- **Essential**: Must be present for diagnosis (high sensitivity)
- **Strong**: Commonly associated, strong clinical indicator
- **Weak**: May be present but has limited diagnostic value

**supported_candidates** (required): Array of disease IDs (format: "d_XX") that this feature SUPPORTS.
- Use the EXACT IDs provided in the input (e.g., "d_24", "d_03", "d_01")
- A feature can support multiple diseases

**ruled_out_candidates** (required): Array of disease IDs that this feature RULES OUT.
- If this feature is present, these diseases become less likely
- Can be empty array [] if no diseases are ruled out

### IMPORTANT RULES

1. **Use exact IDs**: Copy the disease IDs exactly as provided (e.g., "d_24", not "24" or "Pericarditis")
2. Generate 3-5 K-Nodes per candidate disease (mix of General and Pivot)
3. Each Pivot K-Node SHOULD have both supported_candidates and ruled_out_candidates
4. Avoid redundant/overlapping features
5. Use standardized medical terminology
6. Be conservative: quality over quantity
7. Every K-Node MUST have at least one entry in supported_candidates"""

PHASE2_BATCH_USER_PROMPT_TEMPLATE = """## DIFFERENTIAL DIAGNOSIS TASK

### Candidate Diseases (USE THESE EXACT IDs):
{candidates_list}

### Patient's Existing Features (P-Nodes):
{p_nodes_summary}

---

### Clinical Knowledge for Each Candidate:

⚠️ **CONTEXT ISOLATION:** Each candidate's knowledge is enclosed in `<candidate>` XML tags. 
Only use information within a candidate's tags to support/rule out THAT candidate.

{candidate_knowledge_sections}

---

### YOUR TASK

Based on the above clinical summaries:

1. **Extract General K-Nodes**: For each candidate, identify 2-3 essential clinical features from the provided knowledge.

2. **Extract Pivot K-Nodes**: Identify 3-5 key discriminating features that help distinguish between these candidates.

3. **Link to Candidates**: For each K-Node, specify:
   - `supported_candidates`: Which disease IDs (e.g., "d_24") this feature supports
   - `ruled_out_candidates`: Which disease IDs this feature argues against

4. **Match with P-Nodes**: When creating K-Nodes, consider whether the patient's existing P-Nodes match or conflict with these features.

### ⚠️ CRITICAL RULES:
- Use the EXACT disease IDs from the list above (format: "d_XX"). Do NOT use disease names.
- **Evidence-Based Linking:** Only link a feature to a candidate if it is clearly described in that candidate's `<candidate>` block. Avoid indiscriminate linking (do not "spam" links to all candidates just to be safe).

Output your analysis as a valid JSON object following the schema provided."""

# 保持向后兼容的旧变量名
PHASE2_KPivot_SYSTEM_PROMPT = PHASE2_BATCH_SYSTEM_PROMPT
PHASE2_KPivot_USER_PROMPT_TEMPLATE = PHASE2_BATCH_USER_PROMPT_TEMPLATE


def format_candidates_list(candidates: list) -> str:
    """
    格式化候选疾病列表
    
    使用完整 ID 格式（d_XX）确保 LLM 输出一致性
    
    Args:
        candidates: D-Node 列表
    
    Returns:
        格式化字符串
    """
    lines = []
    for d in candidates:
        d_id = d.get("id", "")  # 保持完整 d_XX 格式
        name = d.get("name", "Unknown")
        rank = d.get("initial_rank", "?")
        lines.append(f"- **{d_id}**: {name} (Rank: {rank})")
    return "\n".join(lines)


def format_p_nodes_summary(p_nodes: list) -> str:
    """
    格式化患者特征摘要
    
    Args:
        p_nodes: P-Node 列表
    
    Returns:
        格式化字符串
    """
    if not p_nodes:
        return "No existing features extracted."
    
    present = []
    absent = []
    
    for p in p_nodes:
        content = p.get("content", "")
        status = p.get("status", "Present")
        
        if status == "Present":
            present.append(f"✓ {content}")
        elif status == "Absent":
            absent.append(f"✗ {content}")
    
    lines = []
    if present:
        lines.append("**Present:**")
        lines.extend(present[:15])  # 限制数量
        if len(present) > 15:
            lines.append(f"... and {len(present) - 15} more")
    
    if absent:
        lines.append("\n**Absent (Denied):**")
        lines.extend(absent[:10])
        if len(absent) > 10:
            lines.append(f"... and {len(absent) - 10} more")
    
    return "\n".join(lines)


def format_candidate_knowledge_section(
    d_id: str,
    d_name: str,
    open_targets_text: str,
    pubmed_text: str
) -> str:
    """
    格式化单个候选的知识段落 (使用 XML Tags 进行上下文隔离)
    
    使用 <candidate> XML 标签封装每个候选的知识，
    帮助 LLM 明确区分不同疾病的知识边界。
    
    Args:
        d_id: D-Node ID (完整格式如 d_24)
        d_name: 疾病名称
        open_targets_text: Open Targets 描述
        pubmed_text: PubMed Review 文本
    
    Returns:
        格式化字符串 (XML 格式)
    """
    lines = [f'<candidate id="{d_id}" name="{d_name}">']
    
    if open_targets_text:
        lines.append(f"[Standard Description]: {open_targets_text.strip()}")
    
    if pubmed_text:
        lines.append(f"[Clinical Review]: {pubmed_text.strip()}")
    
    if not open_targets_text and not pubmed_text:
        lines.append("[No knowledge retrieved for this candidate]")
    
    lines.append("</candidate>")
    
    return "\n".join(lines)


def build_batch_prompt(
    candidates: list,
    p_nodes: list,
    knowledge_map: dict
) -> str:
    """
    构建完整的 Batch Reasoning User Prompt
    
    Args:
        candidates: D-Node 列表
        p_nodes: P-Node 列表
        knowledge_map: {d_id: {"open_targets": str, "pubmed": str}}
    
    Returns:
        格式化的 User Prompt
    """
    # 格式化候选列表
    candidates_list = format_candidates_list(candidates)
    
    # 格式化 P-Nodes 摘要
    p_nodes_summary = format_p_nodes_summary(p_nodes)
    
    # 格式化每个候选的知识段落
    knowledge_sections = []
    for d in candidates:
        d_id = d.get("id", "")
        d_name = d.get("name", "Unknown")
        
        knowledge = knowledge_map.get(d_id, {})
        ot_text = knowledge.get("open_targets", "")
        pm_text = knowledge.get("pubmed", "")
        
        section = format_candidate_knowledge_section(d_id, d_name, ot_text, pm_text)
        knowledge_sections.append(section)
    
    # 使用双换行分隔 XML 块（不再需要 --- 分隔符，XML 标签已经提供了清晰边界）
    candidate_knowledge_sections = "\n\n".join(knowledge_sections)
    
    # 填充模板
    return PHASE2_BATCH_USER_PROMPT_TEMPLATE.format(
        candidates_list=candidates_list,
        p_nodes_summary=p_nodes_summary,
        candidate_knowledge_sections=candidate_knowledge_sections
    )
