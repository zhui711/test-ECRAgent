"""
JSON Utilities
提供健壮的 JSON 解析工具，处理 LLM 返回的各种格式

增强功能（v2.0）：
- 支持多种 JSON 格式提取
- 自动修复常见格式问题
- 针对 Qwen 模型的特殊处理（<think> 标签等）
- 详细的错误日志
- 多层降级策略确保尽可能提取有效数据
"""
import json
import re
from typing import Dict, Any, Optional, List, Union


def _clean_llm_output(text: str) -> str:
    """
    清理 LLM 输出中的非 JSON 内容
    
    处理内容：
    - Qwen 模型的 <think>...</think> 标签
    - 其他常见的思考标签变体
    - 前后的解释性文本
    
    Args:
        text: LLM 原始输出
    
    Returns:
        清理后的文本
    """
    if not text:
        return text
    
    # 移除 Qwen 的 <think>...</think> 标签及其内容
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    
    # 移除其他常见思考标签变体
    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<reflection>[\s\S]*?</reflection>', '', text, flags=re.IGNORECASE)
    
    # 移除可能的 XML 声明
    text = re.sub(r'<\?xml[^>]*\?>', '', text)
    
    return text.strip()


def _fix_common_json_issues(json_str: str) -> str:
    """
    修复常见的 JSON 格式问题
    
    Args:
        json_str: 可能有格式问题的 JSON 字符串
    
    Returns:
        修复后的 JSON 字符串
    """
    if not json_str:
        return json_str
    
    # 移除可能的 BOM
    json_str = json_str.strip().lstrip('\ufeff')
    
    # 修复尾部逗号 (trailing comma)
    # 例如: {"a": 1,} -> {"a": 1}
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # 移除注释 (// 和 /* */)
    # 注意：只移除不在字符串内的注释
    json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # 修复不正确的 null 表示
    json_str = re.sub(r':\s*None\b', ': null', json_str)
    json_str = re.sub(r':\s*True\b', ': true', json_str)
    json_str = re.sub(r':\s*False\b', ': false', json_str)
    
    # ========== 新增: 修复缺少逗号的情况 ==========
    # 修复数组中对象之间缺少逗号: }{ -> },{
    # 这是 LLM 常见的格式错误
    json_str = re.sub(r'}\s*{', '}, {', json_str)
    
    # 修复数组中数组之间缺少逗号: ][ -> ],[
    json_str = re.sub(r']\s*\[', '], [', json_str)
    
    # 修复对象属性之间缺少逗号 (值为字符串的情况)
    # 例如: "key1": "value1" "key2" -> "key1": "value1", "key2"
    # 注意：这个正则需要小心，只修复 "value"\s+"key" 的模式
    json_str = re.sub(r'"\s*\n\s*"([a-zA-Z_])', r'",\n"\1', json_str)
    
    # 修复对象属性之间缺少逗号 (值为布尔或 null 的情况)
    # 例如: "key1": true "key2" -> "key1": true, "key2"
    json_str = re.sub(r'(true|false|null)\s*\n\s*"([a-zA-Z_])', r'\1,\n"\2', json_str)
    
    # 修复对象属性之间缺少逗号 (值为数字的情况)
    # 例如: "key1": 123 "key2" -> "key1": 123, "key2"
    json_str = re.sub(r'(\d)\s*\n\s*"([a-zA-Z_])', r'\1,\n"\2', json_str)
    
    # 修复对象属性之间缺少逗号 (值为对象或数组的情况)
    # 例如: "key1": {} "key2" -> "key1": {}, "key2"
    json_str = re.sub(r'}\s*\n\s*"([a-zA-Z_])', r'},\n"\1', json_str)
    json_str = re.sub(r']\s*\n\s*"([a-zA-Z_])', r'],\n"\1', json_str)
    
    # ========== 修复字符串值中的控制字符 ==========
    def fix_string_value(match):
        value = match.group(1)
        # 转义控制字符
        value = value.replace('\n', '\\n')
        value = value.replace('\r', '\\r')
        value = value.replace('\t', '\\t')
        # 移除其他控制字符
        value = re.sub(r'[\x00-\x1f\x7f]', '', value)
        return f'"{value}"'
    
    # 尝试修复字符串值中的控制字符
    try:
        json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fix_string_value, json_str)
    except:
        pass  # 如果正则失败，保持原样
    
    return json_str


def _try_fix_single_quotes(json_str: str) -> Optional[Any]:
    """
    尝试将单引号替换为双引号来修复 JSON
    """
    try:
        # 简单替换（可能在某些情况下失败）
        fixed = json_str.replace("'", '"')
        return json.loads(fixed)
    except:
        return None


def _extract_json_braces(text: str, start_char: str = '{', end_char: str = '}') -> Optional[str]:
    """
    提取平衡的 JSON 对象或数组
    
    Args:
        text: 源文本
        start_char: 开始字符 ('{' 或 '[')
        end_char: 结束字符 ('}' 或 ']')
    
    Returns:
        提取的 JSON 字符串或 None
    """
    start_idx = text.find(start_char)
    if start_idx == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]
    
    # 如果没有找到完整闭合，返回从开始到末尾的内容（可能是截断的JSON）
    # 尝试通过添加闭合括号来修复
    return None


def _extract_all_json_candidates(text: str) -> List[str]:
    """
    从文本中提取所有可能的 JSON 候选（对象和数组）
    
    优先级：
    1. Markdown 代码块内的 JSON
    2. 以 [ 或 { 开头的完整 JSON
    
    Args:
        text: 源文本
    
    Returns:
        JSON 候选字符串列表
    """
    candidates = []
    
    # 1. 提取 markdown 代码块 (```json ... ``` 或 ``` ... ```)
    code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    for block in code_blocks:
        block = block.strip()
        if block.startswith(('{', '[')):
            candidates.append(block)
    
    # 2. 提取独立的 JSON 对象/数组（从第一个 { 或 [ 开始）
    # 先尝试数组（因为数组可能包含对象）
    array_json = _extract_json_braces(text, '[', ']')
    if array_json:
        candidates.append(array_json)
    
    # 再尝试对象
    obj_json = _extract_json_braces(text, '{', '}')
    if obj_json:
        candidates.append(obj_json)
    
    return candidates


def _try_parse_json(json_str: str, verbose: bool = False) -> Optional[Any]:
    """
    尝试解析 JSON 字符串，应用多种修复策略
    
    Args:
        json_str: JSON 字符串
        verbose: 是否打印详细日志
    
    Returns:
        解析后的 JSON 对象或 None
    """
    if not json_str:
        return None
    
    last_error = None  # 记录最后一个错误
    
    # 策略 1: 直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        last_error = e
    
    # 策略 2: 修复常见问题后解析
    fixed_str = _fix_common_json_issues(json_str)
    try:
        return json.loads(fixed_str)
    except json.JSONDecodeError as e:
        last_error = e
    
    # 策略 3: 单引号修复
    result = _try_fix_single_quotes(fixed_str)
    if result is not None:
        return result
    
    # 策略 4: 尝试修复截断的 JSON（添加闭合括号）- 增强版
    # 更多截断修复组合，处理 ["d_0  ] 这种情况
    truncation_fixes = [
        # 简单闭合
        '}', ']', ']}', ']}', 
        # 字符串截断 + 闭合
        '"}', '"]', '"]}', '"]}',
        '1"]', '1"]}', '1"]}',  # 数字ID截断
        '"]]}', '"]}]', '"]}}',  # 深层嵌套
        # 数组内字符串截断
        '"]', '"]}', '"]}}', '"]}]',
        '1"]', '1"]}', '1"]}}',  # d_01 这种ID被截断
        '3"]', '3"]}', '3"]}}',  # d_03
        '4"]', '4"]}', '4"]}}',  # d_24, d_44
        '0"]', '0"]}', '0"]}}',  # d_10, d_40
    ]
    
    for suffix in truncation_fixes:
        try:
            result = json.loads(fixed_str + suffix)
            print(f"[JSON Parser] Recovered truncated JSON with suffix: '{suffix}'")
            return result
        except:
            pass
    
    # 策略 5: 激进修复 - 找到最后一个完整的 k_node 并截断
    try:
        # 找到最后一个完整的 }, 结构
        last_complete = fixed_str.rfind('},')
        if last_complete > 0:
            truncated = fixed_str[:last_complete + 1] + ']}'
            result = json.loads(truncated)
            print(f"[JSON Parser] Recovered by truncating to last complete element")
            return result
    except Exception:
        pass
    
    # 策略 6: 更激进修复 - 针对 k_nodes 数组被截断的情况
    # 场景: {"k_nodes": [...incomplete
    try:
        # 查找 "k_nodes": [ 的位置
        k_nodes_start = fixed_str.find('"k_nodes"')
        if k_nodes_start > 0:
            # 找到数组开始的 [
            array_start = fixed_str.find('[', k_nodes_start)
            if array_start > 0:
                # 从数组开始位置向后查找最后一个完整的对象 }
                # 注意：需要找到不在字符串内的 }
                last_brace = -1
                in_string = False
                escape_next = False
                for i in range(array_start, len(fixed_str)):
                    char = fixed_str[i]
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    if not in_string and char == '}':
                        last_brace = i
                
                if last_brace > array_start:
                    # 截断到最后一个 } 并添加 ]}
                    truncated = fixed_str[:last_brace + 1] + ']}'
                    result = json.loads(truncated)
                    k_count = len(result.get('k_nodes', []))
                    print(f"[JSON Parser] Recovered k_nodes array (found {k_count} complete nodes)")
                    return result
    except Exception:
        pass
    
    # 策略 7: 最后尝试 - 只提取已完成的 k_node 对象
    try:
        # 正则匹配所有完整的 k_node 对象
        import re
        k_node_pattern = r'\{\s*"content"\s*:\s*"[^"]*"[^}]*"supported_candidates"\s*:\s*\[[^\]]*\][^}]*\}'
        matches = re.findall(k_node_pattern, fixed_str, re.DOTALL)
        if matches:
            k_nodes = []
            for match in matches:
                try:
                    node = json.loads(match)
                    k_nodes.append(node)
                except:
                    continue
            if k_nodes:
                print(f"[JSON Parser] Recovered {len(k_nodes)} k_nodes via regex extraction")
                return {"k_nodes": k_nodes, "edges": []}
    except Exception:
        pass
    
    if verbose:
        print(f"[JSON Parser] All parse attempts failed for: {json_str[:100]}...")
        if last_error:
            print(f"[JSON Parser] Last error: {last_error}")
    
    return None


def parse_json_from_text(text: str, verbose: bool = True) -> Optional[Union[Dict[str, Any], List]]:
    """
    从文本中解析 JSON，处理各种格式：
    - Markdown 代码块 (```json ... ```)
    - 纯 JSON 字符串
    - 包含其他文本的混合内容
    - Qwen 模型的 <think> 标签
    
    增强功能：
    - 自动修复常见格式问题
    - 多层降级策略
    - 详细的错误日志
    
    Args:
        text: 包含 JSON 的文本
        verbose: 是否打印详细日志
    
    Returns:
        解析后的 JSON 对象（字典或列表），如果解析失败则返回 None
    """
    if not text or not isinstance(text, str):
        if verbose:
            print("[JSON Parser] Empty or invalid input")
        return None
    
    original_text = text
    
    # 预处理：清理 LLM 特殊标签
    text = _clean_llm_output(text)
    text = text.strip()
    
    if not text:
        if verbose:
            print("[JSON Parser] Text is empty after cleaning LLM tags")
        return None
    
    # 方法 1: 尝试提取 JSON 代码块 (```json ... ```)
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        result = _try_parse_json(json_str, verbose)
        if result is not None:
            if verbose:
                print(f"[JSON Parser] Successfully parsed from ```json block ({len(json_str)} chars)")
            return result
    
    # 方法 2: 尝试提取代码块（无语言标签）
    json_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_str = json_match.group(1).strip()
        if json_str.startswith(('{', '[')):
            result = _try_parse_json(json_str, verbose)
            if result is not None:
                if verbose:
                    print(f"[JSON Parser] Successfully parsed from ``` block ({len(json_str)} chars)")
                return result
    
    # 方法 3: 检查文本是否直接以 JSON 开头
    stripped_text = text.lstrip()
    if stripped_text.startswith('['):
        # 优先处理数组（避免错误地提取内部对象）
        json_str = _extract_json_braces(stripped_text, '[', ']')
        if json_str:
            result = _try_parse_json(json_str, verbose)
            if result is not None:
                if verbose:
                    print(f"[JSON Parser] Successfully parsed JSON array ({len(json_str)} chars)")
                return result
    
    if stripped_text.startswith('{'):
        json_str = _extract_json_braces(stripped_text, '{', '}')
        if json_str:
            result = _try_parse_json(json_str, verbose)
            if result is not None:
                if verbose:
                    print(f"[JSON Parser] Successfully parsed JSON object ({len(json_str)} chars)")
                return result
    
    # 方法 4: 使用平衡括号提取 JSON 对象（文本中间可能有 JSON）
    json_str = _extract_json_braces(text, '{', '}')
    if json_str:
        result = _try_parse_json(json_str, verbose)
        if result is not None:
            if verbose:
                print(f"[JSON Parser] Successfully parsed embedded JSON object ({len(json_str)} chars)")
            return result
    
    # 方法 5: 使用平衡括号提取 JSON 数组
    json_str = _extract_json_braces(text, '[', ']')
    if json_str:
        result = _try_parse_json(json_str, verbose)
        if result is not None:
            if verbose:
                print(f"[JSON Parser] Successfully parsed embedded JSON array ({len(json_str)} chars)")
            return result
    
    # 方法 6: 直接尝试解析整个文本
    result = _try_parse_json(text, verbose)
    if result is not None:
        if verbose:
            print(f"[JSON Parser] Successfully parsed entire text ({len(text)} chars)")
        return result
    
    # 所有方法都失败了
    if verbose:
        print(f"[JSON Parser] All parsing methods failed")
        print(f"[JSON Parser] Original text length: {len(original_text)}")
        print(f"[JSON Parser] Text preview (first 500 chars):")
        print(f"  {original_text[:500]}")
        if len(original_text) > 500:
            print(f"  ... (truncated)")
    
    return None


def parse_json_array_from_text(text: str, verbose: bool = True) -> Optional[List[Dict[str, Any]]]:
    """
    从文本中解析 JSON 数组
    
    增强策略：
    - 优先检测并提取数组
    - 支持从包装对象中提取数组字段
    - 处理 Qwen 模型的特殊输出格式
    
    Args:
        text: 包含 JSON 数组的文本
        verbose: 是否打印详细日志
    
    Returns:
        解析后的 JSON 数组，如果解析失败则返回 None
    """
    if not text or not isinstance(text, str):
        if verbose:
            print("[JSON Parser] Empty or invalid input for array parsing")
        return None
    
    # 预处理：清理 LLM 特殊标签
    original_text = text
    text = _clean_llm_output(text)
    text = text.strip()
    
    if not text:
        if verbose:
            print("[JSON Parser] Text is empty after cleaning LLM tags")
        return None
    
    # 方法 1: 检查 markdown 代码块内的数组
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        if json_str.startswith('['):
            result = _try_parse_json(json_str, verbose)
            if isinstance(result, list):
                if verbose:
                    print(f"[JSON Parser] Successfully parsed array from code block ({len(json_str)} chars)")
                return result
    
    # 方法 2: 检查文本是否直接以 [ 开头
    stripped_text = text.lstrip()
    if stripped_text.startswith('['):
        json_str = _extract_json_braces(stripped_text, '[', ']')
        if json_str:
            result = _try_parse_json(json_str, verbose)
            if isinstance(result, list):
                if verbose:
                    print(f"[JSON Parser] Successfully parsed JSON array directly ({len(json_str)} chars)")
                return result
    
    # 方法 3: 从文本任意位置提取数组
    json_str = _extract_json_braces(text, '[', ']')
    if json_str:
        result = _try_parse_json(json_str, verbose)
        if isinstance(result, list):
            if verbose:
                print(f"[JSON Parser] Successfully parsed embedded JSON array ({len(json_str)} chars)")
            return result
    
    # 方法 4: 使用通用解析器
    parsed = parse_json_from_text(text, verbose=False)
    
    if isinstance(parsed, list):
        if verbose:
            print(f"[JSON Parser] Parsed as array via general parser")
        return parsed
    
    # 方法 5: 如果解析结果是字典，检查是否有数组字段
    if isinstance(parsed, dict):
        # 尝试常见的数组字段名
        array_keys = ['results', 'items', 'data', 'updates', 'nodes', 'edges', 
                      'array', 'list', 'content', 'output', 'response', 'p_nodes', 'k_nodes']
        for key in array_keys:
            if key in parsed and isinstance(parsed[key], list):
                if verbose:
                    print(f"[JSON Parser] Extracted array from '{key}' field")
                return parsed[key]
        
        # 如果只有一个键且值是数组，提取它
        if len(parsed) == 1:
            only_value = list(parsed.values())[0]
            if isinstance(only_value, list):
                if verbose:
                    print(f"[JSON Parser] Extracted array from single-key dict")
                return only_value
    
    if verbose:
        print(f"[JSON Parser] Failed to parse as array. Result type: {type(parsed)}")
        print(f"[JSON Parser] Original text preview: {original_text[:300]}...")
    
    return None


def extract_json_field(text: str, field_name: str, verbose: bool = False) -> Optional[Any]:
    """
    从文本中提取特定的 JSON 字段值
    
    这是一个更激进的提取策略，用于当完整 JSON 解析失败时
    尝试至少提取关键字段
    
    Args:
        text: 包含 JSON 的文本
        field_name: 要提取的字段名
        verbose: 是否打印详细日志
    
    Returns:
        字段值，如果提取失败则返回 None
    """
    if not text or not field_name:
        return None
    
    # 清理 LLM 标签
    text = _clean_llm_output(text)
    
    # 尝试完整解析
    parsed = parse_json_from_text(text, verbose=False)
    if isinstance(parsed, dict) and field_name in parsed:
        return parsed[field_name]
    
    # 正则表达式提取（作为降级策略）
    # 匹配 "field_name": value 或 "field_name": "value" 或 "field_name": [...]
    patterns = [
        # 数组值
        rf'"{re.escape(field_name)}"\s*:\s*(\[[^\]]*\])',
        # 对象值
        rf'"{re.escape(field_name)}"\s*:\s*(\{{[^\}}]*\}})',
        # 字符串值
        rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)"',
        # 数字/布尔值
        rf'"{re.escape(field_name)}"\s*:\s*([\d.]+|true|false|null)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value_str = match.group(1)
            try:
                return json.loads(value_str)
            except:
                # 如果解析失败，返回原始字符串
                return value_str
    
    if verbose:
        print(f"[JSON Parser] Failed to extract field '{field_name}'")
    
    return None


def safe_json_loads(json_str: str, default: Any = None, verbose: bool = False) -> Any:
    """
    安全的 JSON 解析，失败时返回默认值
    
    Args:
        json_str: JSON 字符串
        default: 解析失败时的默认值
        verbose: 是否打印错误日志
    
    Returns:
        解析后的对象或默认值
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        if verbose:
            print(f"[JSON Parser] safe_json_loads failed: {e}")
        return default


def format_json_for_prompt(data: Union[Dict, List], indent: int = 2) -> str:
    """
    将数据格式化为 JSON 字符串，用于注入到 Prompt 中
    
    Args:
        data: 要格式化的数据
        indent: 缩进空格数
    
    Returns:
        格式化的 JSON 字符串
    """
    return json.dumps(data, ensure_ascii=False, indent=indent)


def robust_parse_p_nodes(text: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    专门用于解析 Phase 1 Track B 的 P-Nodes 输出
    
    多层降级策略：
    1. 尝试完整 JSON 解析
    2. 尝试提取 p_nodes 字段
    3. 尝试提取数组
    4. 逐行解析（最后手段）
    
    Args:
        text: LLM 输出文本
        verbose: 是否打印详细日志
    
    Returns:
        P-Nodes 列表（即使部分失败也返回成功解析的部分）
    """
    if not text:
        return []
    
    # 清理 LLM 标签
    text = _clean_llm_output(text)
    
    # 策略 1: 完整 JSON 解析
    parsed = parse_json_from_text(text, verbose=False)
    if isinstance(parsed, dict):
        p_nodes = parsed.get('p_nodes', [])
        if isinstance(p_nodes, list) and len(p_nodes) > 0:
            if verbose:
                print(f"[JSON Parser] Extracted {len(p_nodes)} P-Nodes from complete JSON")
            return p_nodes
    
    # 策略 2: 直接提取 p_nodes 字段
    p_nodes = extract_json_field(text, 'p_nodes', verbose=False)
    if isinstance(p_nodes, list) and len(p_nodes) > 0:
        if verbose:
            print(f"[JSON Parser] Extracted {len(p_nodes)} P-Nodes via field extraction")
        return p_nodes
    
    # 策略 3: 尝试解析为数组
    array_result = parse_json_array_from_text(text, verbose=False)
    if array_result and len(array_result) > 0:
        # 验证数组元素是否像 P-Node
        if _looks_like_p_nodes(array_result):
            if verbose:
                print(f"[JSON Parser] Extracted {len(array_result)} P-Nodes from array")
            return array_result
    
    # 策略 4: 逐个提取 JSON 对象（最后手段）
    p_nodes = []
    # 查找所有可能的 P-Node 对象
    pattern = r'\{\s*"id"\s*:\s*"p_\d+"[^}]+\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            node = json.loads(match)
            if 'id' in node and 'content' in node:
                p_nodes.append(node)
        except:
            continue
    
    if p_nodes:
        if verbose:
            print(f"[JSON Parser] Extracted {len(p_nodes)} P-Nodes via regex fallback")
        return p_nodes
    
    if verbose:
        print(f"[JSON Parser] Failed to extract any P-Nodes")
    
    return []


def _looks_like_p_nodes(items: List) -> bool:
    """
    检查数组是否看起来像 P-Nodes 列表
    """
    if not items:
        return False
    
    # 检查第一个元素是否有 P-Node 的典型字段
    first = items[0]
    if isinstance(first, dict):
        # P-Node 应该有 id 和 content 字段
        return 'content' in first or 'id' in first
    
    return False


def robust_parse_track_a_response(text: str, verbose: bool = True) -> Dict[str, Any]:
    """
    专门用于解析 Phase 1 Track A 的 LLM 输出
    
    针对常见的 LLM 输出错误进行健壮提取：
    - 数组未闭合: ["25", "23", " "final_diagnosis_id"...
    - key 被截断: "top5", "23"...
    - 缺少开头 [: "top_candidates": "05", "30"...
    - 只有数组没有 key: ["25", "05"...]
    - 值被截断: "final_diagnosis_id"
    
    Args:
        text: LLM 输出文本
        verbose: 是否打印详细日志
    
    Returns:
        包含 top_candidates, final_diagnosis_id, structured_analysis, differential_reasoning 的字典
    """
    result = {
        "top_candidates": [],
        "final_diagnosis_id": None,
        "structured_analysis": None,
        "differential_reasoning": None
    }
    
    if not text:
        return result
    
    # 清理 LLM 标签
    text = _clean_llm_output(text)
    
    # 策略 1: 尝试完整 JSON 解析
    parsed = parse_json_from_text(text, verbose=False)
    if isinstance(parsed, dict):
        result["top_candidates"] = parsed.get("top_candidates", [])
        result["final_diagnosis_id"] = parsed.get("final_diagnosis_id")
        result["structured_analysis"] = parsed.get("structured_analysis")
        result["differential_reasoning"] = parsed.get("differential_reasoning")
        
        if result["top_candidates"] and result["final_diagnosis_id"]:
            if verbose:
                print(f"[JSON Parser] Track A: Complete parse success")
            return result
    
    # 策略 2: 使用专门的正则表达式提取
    # 提取 final_diagnosis_id
    id_patterns = [
        r'"final_diagnosis_id"\s*:\s*"(\d+)"',  # 标准格式
        r'"final_diagnosis_id"\s*:\s*(\d+)',     # 无引号数字
        r'final_diagnosis_id["\s:]+(\d+)',       # 更宽松
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, text)
        if match:
            result["final_diagnosis_id"] = match.group(1)
            break
    
    # 策略 3: 提取 top_candidates 数组
    # 尝试多种模式
    candidates_patterns = [
        # 标准格式: "top_candidates": ["25", "23", ...]
        r'"top_candidates"\s*:\s*\[([\s\S]*?)\]',
        # 未闭合的数组: "top_candidates": ["25", "23",
        r'"top_candidates"\s*:\s*\[([^\]]+)',
    ]
    
    for pattern in candidates_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            array_content = match.group(1)
            # 从数组内容中提取所有引号包围的两位数字
            ids = re.findall(r'"(\d{1,2})"', array_content)
            if ids:
                # 去重并保持顺序
                seen = set()
                unique_ids = []
                for id_val in ids:
                    if id_val not in seen:
                        seen.add(id_val)
                        unique_ids.append(id_val)
                result["top_candidates"] = unique_ids[:5]  # 最多5个候选
                break
    
    # 策略 4: 处理 key 被截断的情况 (如 "top5", "23"...)
    if not result["top_candidates"]:
        # 查找被截断的 key 模式
        truncated_match = re.search(r'"top\w*"\s*,\s*((?:"[^"]*"\s*,?\s*)+)', text)
        if truncated_match:
            array_content = truncated_match.group(1)
            ids = re.findall(r'"(\d{1,2})"', array_content)
            if ids:
                seen = set()
                unique_ids = []
                for id_val in ids:
                    if id_val not in seen:
                        seen.add(id_val)
                        unique_ids.append(id_val)
                result["top_candidates"] = unique_ids[:5]
    
    # 策略 5: 处理缺少开头 [ 的情况 (如 "top_candidates": "05", "30"...)
    if not result["top_candidates"]:
        # 查找 "top_candidates": "xx", "xx"... 模式
        no_bracket_match = re.search(
            r'"top_candidates"\s*:\s*"(\d{1,2})"((?:\s*,\s*"(\d{1,2})")*)',
            text
        )
        if no_bracket_match:
            first_id = no_bracket_match.group(1)
            rest = no_bracket_match.group(2)
            ids = [first_id] + re.findall(r'"(\d{1,2})"', rest)
            if ids:
                seen = set()
                unique_ids = []
                for id_val in ids:
                    if id_val not in seen:
                        seen.add(id_val)
                        unique_ids.append(id_val)
                result["top_candidates"] = unique_ids[:5]
    
    # 策略 6: 如果上述失败，尝试从文本中提取任何独立的数组
    if not result["top_candidates"]:
        # 查找任何数组形式 [...]
        array_match = re.search(r'\[\s*((?:"[^"]*"\s*,?\s*)+)\s*\]', text)
        if array_match:
            ids = re.findall(r'"(\d{1,2})"', array_match.group(1))
            if ids:
                seen = set()
                unique_ids = []
                for id_val in ids:
                    if id_val not in seen:
                        seen.add(id_val)
                        unique_ids.append(id_val)
                result["top_candidates"] = unique_ids[:5]
    
    # 策略 5: 从 differential_reasoning 中提取信息
    reasoning_patterns = [
        r'"differential_reasoning"\s*:\s*"([\s\S]*?)"(?=\s*,|\s*})',
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, text)
        if match:
            result["differential_reasoning"] = match.group(1)
            break
    
    if verbose:
        if result["top_candidates"] or result["final_diagnosis_id"]:
            print(f"[JSON Parser] Track A: Fallback extraction - "
                  f"candidates: {len(result['top_candidates'])}, "
                  f"diagnosis_id: {result['final_diagnosis_id']}")
        else:
            print(f"[JSON Parser] Track A: Failed to extract key fields")
    
    return result


def robust_parse_k_gen_response(text: str, verbose: bool = True) -> Dict[str, Any]:
    """
    专门用于解析 Phase 2 K-Gen 的 LLM 输出
    
    期望格式：
    {
        "k_nodes": [...],
        "edges": [...]
    }
    
    降级策略：
    1. 完整 JSON 解析
    2. 分别提取 k_nodes 和 edges 字段
    3. 返回部分结果
    
    Args:
        text: LLM 输出文本
        verbose: 是否打印详细日志
    
    Returns:
        包含 k_nodes 和 edges 的字典
    """
    result = {"k_nodes": [], "edges": []}
    
    if not text:
        return result
    
    # 清理 LLM 标签
    text = _clean_llm_output(text)
    
    # 策略 1: 完整 JSON 解析
    parsed = parse_json_from_text(text, verbose=False)
    if isinstance(parsed, dict):
        result["k_nodes"] = parsed.get("k_nodes", [])
        result["edges"] = parsed.get("edges", [])
        
        if result["k_nodes"] or result["edges"]:
            if verbose:
                print(f"[JSON Parser] K-Gen: Extracted {len(result['k_nodes'])} K-Nodes, "
                      f"{len(result['edges'])} edges")
            return result
    
    # 策略 2: 分别提取字段
    k_nodes = extract_json_field(text, 'k_nodes', verbose=False)
    if isinstance(k_nodes, list):
        result["k_nodes"] = k_nodes
    
    edges = extract_json_field(text, 'edges', verbose=False)
    if isinstance(edges, list):
        result["edges"] = edges
    
    if result["k_nodes"] or result["edges"]:
        if verbose:
            print(f"[JSON Parser] K-Gen (fallback): Extracted {len(result['k_nodes'])} K-Nodes, "
                  f"{len(result['edges'])} edges")
    else:
        if verbose:
            print(f"[JSON Parser] K-Gen: Failed to extract K-Nodes or edges")
    
    return result
