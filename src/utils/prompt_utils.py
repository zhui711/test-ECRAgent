"""
Prompt Utilities
保留用户现有的 prompt_utils.py 文件内容
"""

VALID_DIAGNOSES_LIST = [
    "Possible NSTEMI / STEMI", "Spontaneous rib fracture", "Pulmonary embolism",
    "Pulmonary neoplasm", "URTI", "Croup", "Sarcoidosis", "HIV (initial infection)",
    "Inguinal hernia", "Spontaneous pneumothorax", "Bronchospasm / acute asthma exacerbation",
    "Viral pharyngitis", "Bronchiolitis", "Pancreatic neoplasm", "Guillain-Barré syndrome",
    "Chagas", "Allergic sinusitis", "Acute rhinosinusitis", "PSVT", "Panic attack",
    "Epiglottitis", "Bronchiectasis", "Bronchitis", "Pericarditis",
    "Acute COPD exacerbation / infection", "Ebola", "Chronic rhinosinusitis",
    "Acute otitis media", "Larygospasm", "Influenza", "Stable angina", "Myasthenia gravis",
    "Myocarditis", "SLE", "GERD", "Anemia", "Cluster headache", "Localized edema",
    "Anaphylaxis", "Atrial fibrillation", "Acute pulmonary edema", "Acute laryngitis",
    "Acute dystonic reactions", "Boerhaave", "Pneumonia", "Tuberculosis",
    "Whooping cough", "Unstable angina", "Scombroid food poisoning"
]
VALID_DIAGNOSES_STR = "\n".join([f"- {d}" for d in VALID_DIAGNOSES_LIST])

# 生成 formatted_disease_list 字符串
formatted_disease_list = "\n".join(
    [f"{i+1:02d}: {name}" for i, name in enumerate(VALID_DIAGNOSES_LIST)]
)

# 生成 ID 映射字典 (用于 S2/S3 阶段解析 Logits 后反查名字)
DIAGNOSIS_ID_MAP = {
    f"{i+1:02d}": name for i, name in enumerate(VALID_DIAGNOSES_LIST)
}

# 反向映射：疾病名称 -> ID (用于 Ground Truth 匹配)
NAME_TO_ID_MAP = {
    name: f"{i+1:02d}" for i, name in enumerate(VALID_DIAGNOSES_LIST)
}


def get_diagnosis_id(name: str) -> str:
    """
    根据疾病名称获取ID
    
    Args:
        name: 疾病名称（必须精确匹配）
    
    Returns:
        疾病ID（如 "03"），如果找不到返回 None
    """
    return NAME_TO_ID_MAP.get(name)


def get_diagnosis_name(diagnosis_id: str) -> str:
    """
    根据ID获取疾病名称
    
    Args:
        diagnosis_id: 疾病ID（如 "03"）
    
    Returns:
        疾病名称，如果找不到返回 "Unknown"
    """
    return DIAGNOSIS_ID_MAP.get(diagnosis_id, "Unknown")


def normalize_diagnosis_id(raw_id: str) -> str:
    """
    规范化诊断 ID，处理异常格式如 "045", "4", "004" 等
    
    规则：
    1. 有效 ID 范围：01-49（两位数）
    2. 如果输入已经是有效 ID，直接返回
    3. 如果是单个数字（1-9），补零变成两位数（"4" -> "04"）
    4. 如果是三位数如 "045"，尝试拆分成可能的两位数并选择有效的
    5. 移除前导零后重新格式化
    
    Args:
        raw_id: 原始 ID 字符串
    
    Returns:
        规范化后的两位数 ID，如果无法规范化则返回原值
    """
    if not raw_id:
        return raw_id
    
    # 清理输入：移除空格和引号
    cleaned = str(raw_id).strip().strip('"').strip("'").strip()
    
    # 如果已经是有效的两位数 ID，直接返回
    if cleaned in DIAGNOSIS_ID_MAP:
        return cleaned
    
    # 尝试转换为数字
    try:
        num = int(cleaned)
        
        # 如果在有效范围内（1-49），格式化为两位数
        if 1 <= num <= 49:
            return f"{num:02d}"
        
        # 如果数字太大（如 45 被读成 045），尝试截取
        # 例如 "045" -> 可能是 "04" 和 "5" 或者 "0" 和 "45"
        if num > 49 and len(cleaned) == 3:
            # 尝试前两位
            first_two = cleaned[:2]
            if first_two in DIAGNOSIS_ID_MAP:
                return first_two
            
            # 尝试后两位
            last_two = cleaned[1:]
            if last_two in DIAGNOSIS_ID_MAP:
                return last_two
            
            # 尝试移除前导零
            without_leading_zero = cleaned.lstrip('0')
            if len(without_leading_zero) <= 2:
                formatted = f"{int(without_leading_zero):02d}"
                if formatted in DIAGNOSIS_ID_MAP:
                    return formatted
    except ValueError:
        pass
    
    # 无法规范化，返回原值
    return raw_id


def normalize_diagnosis_id_list(raw_ids: list) -> list:
    """
    批量规范化诊断 ID 列表
    
    Args:
        raw_ids: 原始 ID 列表
    
    Returns:
        规范化后的 ID 列表
    """
    if not raw_ids or not isinstance(raw_ids, list):
        return raw_ids
    
    normalized = []
    for raw_id in raw_ids:
        normalized_id = normalize_diagnosis_id(str(raw_id))
        normalized.append(normalized_id)
    return normalized


if __name__ == "__main__":
    print(formatted_disease_list)
    print("\n--- ID to Name Map ---")
    for k, v in DIAGNOSIS_ID_MAP.items():
        print(f"  {k}: {v}")




