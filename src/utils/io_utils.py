"""
IO Utilities
处理 JSONL 文件的读写和断点续传逻辑
"""
import json
import os
from typing import Dict, Any, Optional, List


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 JSONL 文件
    
    Args:
        file_path: JSONL 文件路径
    
    Returns:
        记录列表
    """
    records = []
    if not os.path.exists(file_path):
        return records
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    return records


def save_jsonl(file_path: str, record: Dict[str, Any], append: bool = True):
    """
    保存记录到 JSONL 文件
    
    Args:
        file_path: JSONL 文件路径
        record: 要保存的记录（字典）
        append: 是否追加模式（默认 True）
    """
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8', buffering=1) as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())  # 强制刷盘


def check_processing_status(case_id: str, output_file: str) -> Optional[Dict[str, Any]]:
    """
    检查指定 case_id 是否已经处理成功
    
    Args:
        case_id: 病例 ID
        output_file: 输出 JSONL 文件路径
    
    Returns:
        如果已处理成功，返回记录字典；否则返回 None
    """
    if not os.path.exists(output_file):
        return None
    
    records = load_jsonl(output_file)
    
    # 从后往前查找（最新的记录优先）
    for record in reversed(records):
        if record.get("case_id") == case_id:
            status = record.get("status")
            if status == "success":
                return record
    
    return None


def load_case_data(case_dir: str) -> Optional[Dict[str, Any]]:
    """
    从 case 目录加载 final_benchmark_pair.json
    
    Args:
        case_dir: case 目录路径（例如 "test_verify/case_133004"）
    
    Returns:
        包含 control_case 和 trap_case 的字典，如果文件不存在则返回 None
    """
    pair_file = os.path.join(case_dir, "final_benchmark_pair.json")
    if not os.path.exists(pair_file):
        return None
    
    try:
        with open(pair_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading case data from {pair_file}: {e}")
        return None




