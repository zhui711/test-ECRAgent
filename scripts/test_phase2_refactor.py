#!/usr/bin/env python3
"""
Phase 2 Refactor Test Script
=============================

用于测试 Phase 2 Strict Pruning 重构的单 Case 验证脚本。

功能：
1. 硬编码测试 Case ID (case_10000)
2. 加载对应的 Golden Graph
3. 运行完整的 Phase 1 -> Phase 2 (New Logic) -> Phase 3 流程
4. 打印详细的中间过程 Log (特别是 Pruning 统计数据)

Usage:
    python scripts/test_phase2_refactor.py
    python scripts/test_phase2_refactor.py --case-id case_10000
    python scripts/test_phase2_refactor.py --case-type control_case
    python scripts/test_phase2_refactor.py --skip-phase3
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.api_client import LLMClient
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.graph.schema import MedicalGraph
from src.graph.state import AgentState
from src.graph.golden_graph_loader import GoldenGraphLoader
from src.agents.phase1_manager import Phase1Manager
from src.agents.phase2_hybrid_engine import Phase2HybridEngine
from src.utils.graph_tools import summarize_graph_for_judge, calculate_naive_scores


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_case_data(case_dir: Path, case_type: str) -> Dict[str, Any]:
    """
    加载 Case 数据
    
    Args:
        case_dir: Case 目录
        case_type: "control_case" 或 "trap_case"
    
    Returns:
        Case 数据字典
    """
    case_file = case_dir / "final_benchmark_pair.json"
    if not case_file.exists():
        raise FileNotFoundError(f"final_benchmark_pair.json not found in {case_dir}")
    
    with open(case_file, "r", encoding="utf-8") as f:
        case_data = json.load(f)
    
    if case_type not in case_data:
        raise ValueError(f"Case type '{case_type}' not found in final_benchmark_pair.json")
    
    return case_data[case_type]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Test Phase 2 Strict Pruning Refactor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--case-id",
        type=str,
        default="case_100018",
        help="要测试的 Case ID (默认: case_10000)"
    )
    
    parser.add_argument(
        "--case-type",
        type=str,
        default="control_case",
        choices=["control_case", "trap_case"],
        help="Case 类型 (默认: control_case)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="test_verify",
        help="输入数据目录 (默认: test_verify)"
    )
    
    parser.add_argument(
        "--golden-graph-dir",
        type=str,
        default="golden_graphs",
        help="Golden Graph 目录 (默认: golden_graphs)"
    )
    
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="跳过 Phase 3 执行"
    )
    
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="保存生成的图谱到文件"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug",
        help="输出目录 (默认: debug)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖配置文件中的模型名称"
    )
    
    return parser.parse_args()


def print_separator(title: str) -> None:
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_p_nodes(p_nodes: list, title: str = "P-Nodes") -> None:
    """打印 P-Nodes 列表"""
    print(f"\n{title} ({len(p_nodes)} total):")
    for i, p in enumerate(p_nodes[:20]):  # 只显示前 20 个
        content = p.get("content", "")[:50]
        status = p.get("status", "Unknown")
        source = p.get("source", "Unknown")
        print(f"  [{i+1}] {content}... | Status: {status} | Source: {source}")
    
    if len(p_nodes) > 20:
        print(f"  ... and {len(p_nodes) - 20} more")


def print_k_nodes(k_nodes: list, title: str = "K-Nodes") -> None:
    """打印 K-Nodes 列表"""
    print(f"\n{title} ({len(k_nodes)} total):")
    
    # 按 importance 分组
    by_importance = {}
    for k in k_nodes:
        imp = k.get("importance", "Common")
        if imp not in by_importance:
            by_importance[imp] = []
        by_importance[imp].append(k)
    
    for imp in ["Pathognomonic", "Essential", "Strong", "Common", "Weak"]:
        if imp in by_importance:
            print(f"\n  [{imp}] ({len(by_importance[imp])} nodes):")
            for k in by_importance[imp][:5]:
                content = k.get("content", "")[:60]
                source = k.get("source_type", k.get("source", "Unknown"))
                print(f"    - {content}... | Source: {source}")
            if len(by_importance[imp]) > 5:
                print(f"    ... and {len(by_importance[imp]) - 5} more")


def print_hybrid_stats(stats: Dict[str, Any]) -> None:
    """打印 Hybrid Engine 统计"""
    print("\n📊 Hybrid Engine Statistics:")
    print(f"  - Hybrid Processed: {stats.get('hybrid_processed', 0)}")
    print(f"  - Fallback Processed: {stats.get('fallback_processed', 0)}")
    print(f"  - Pruned P-Nodes: {stats.get('pruned_p_nodes', 0)}")
    print(f"  - Pruned K-Nodes: {stats.get('pruned_k_nodes', 0)}")
    print(f"  - Kept K-Nodes (Match): {stats.get('kept_k_nodes_match', 0)}")
    print(f"  - Kept K-Nodes (Pathognomonic): {stats.get('kept_k_nodes_pathognomonic', 0)}")
    print(f"  - Pathognomonic Shadows: {stats.get('pathognomonic_shadows', 0)}")
    print(f"  - Golden K-Nodes Inherited: {stats.get('golden_k_nodes_inherited', 0)}")
    print(f"  - Live K-Nodes Added: {stats.get('live_k_nodes_added', 0)}")


def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()
    
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config()
    
    # 覆盖模型名称
    if args.model:
        config["api"]["model_name"] = args.model
    
    print_separator("Phase 2 Refactor Test Script")
    print(f"📁 Case ID: {args.case_id}")
    print(f"📋 Case Type: {args.case_type}")
    print(f"📁 Input Directory: {args.input_dir}")
    print(f"📁 Golden Graph Directory: {args.golden_graph_dir}")
    print(f"🔧 Model: {config['api']['model_name']}")
    print(f"⏭️  Skip Phase 3: {args.skip_phase3}")
    
    # 检查 Case 目录
    case_dir = PROJECT_ROOT / args.input_dir / args.case_id
    if not case_dir.exists():
        print(f"\n❌ Case directory not found: {case_dir}")
        sys.exit(1)
    
    # 加载 Case 数据
    try:
        case_info = load_case_data(case_dir, args.case_type)
        narrative = case_info.get("narrative", "")
        ground_truth = case_info.get("ground_truth", "")
        
        print(f"\n📖 Ground Truth: {ground_truth}")
        print(f"📝 Narrative length: {len(narrative)} chars")
        
    except Exception as e:
        print(f"\n❌ Failed to load case data: {e}")
        sys.exit(1)
    
    # 解析 Ground Truth ID
    gt_id = None
    for did, dname in DIAGNOSIS_ID_MAP.items():
        if dname.lower() == ground_truth.lower():
            gt_id = did
            break
    
    print(f"🔢 Ground Truth ID: {gt_id}")
    
    # 初始化 LLM 客户端
    api_config = config.get("api", {})
    api_key = os.getenv("YUNWU_API_KEY")
    
    if not api_key:
        print("\n❌ YUNWU_API_KEY not found in environment")
        sys.exit(1)
    
    llm_client = LLMClient(
        base_url=api_config.get("base_url", "https://yunwu.ai/v1"),
        api_key=api_key,
        timeout=api_config.get("timeout", 120)
    )
    model_name = api_config.get("model_name", "gpt-4o")
    
    # ==================== Phase 1 ====================
    print_separator("Phase 1: Initial Diagnosis")
    
    start_time = time.time()
    
    try:
        phase1 = Phase1Manager(llm_client, model_name)
        phase1_result = phase1.process(narrative)
        
        if phase1_result.get("error"):
            raise Exception(f"Phase 1 error: {phase1_result['error']}")
        
        phase1_time = time.time() - start_time
        
        print(f"\n✅ Phase 1 Complete ({phase1_time:.1f}s)")
        print(f"📊 Top Candidates: {phase1_result.get('top_candidates', [])}")
        print(f"🔢 Final Diagnosis ID: {phase1_result.get('final_diagnosis_id', 'Unknown')}")
        
        # 打印 Track B 的 P-Nodes
        track_b_output = phase1_result.get("track_b_output", {})
        patient_p_nodes = track_b_output.get("p_nodes", [])
        print_p_nodes(patient_p_nodes, "Patient P-Nodes (Track B)")
        
    except Exception as e:
        print(f"\n❌ Phase 1 failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ==================== Phase 2 (Hybrid - Strict Pruning) ====================
    print_separator("Phase 2: Hybrid Investigation (Strict Pruning)")
    
    # 构建初始状态
    state: AgentState = {
        "case_id": args.case_id,
        "input_case": {
            "narrative": narrative,
            "ground_truth": ground_truth,
            "ground_truth_id": gt_id
        },
        "phase1_result": phase1_result,
        "graph_json": None,
        "graph_summary": None,
        "naive_scores": None,
        "final_output": None,
        "status": "processing",
        "error_log": None,
        "global_hint": None,
        "retry_count": 0,
        "memory_context": None
    }
    
    start_time = time.time()
    
    try:
        # 使用新的 Hybrid Engine (Strict Pruning)
        phase2 = Phase2HybridEngine(
            llm_client=llm_client,
            model_name=model_name,
            golden_graph_dir=args.golden_graph_dir
        )
        
        state = phase2.process(state)
        
        if state.get("status") == "failed":
            raise Exception(f"Phase 2 error: {state.get('error_log')}")
        
        phase2_time = time.time() - start_time
        
        print(f"\n✅ Phase 2 Complete ({phase2_time:.1f}s)")
        
        # 获取图谱信息
        graph_json = state.get("graph_json", {})
        graph_nodes = graph_json.get("graph", {}).get("nodes", {})
        
        p_nodes = graph_nodes.get("p_nodes", [])
        k_nodes = graph_nodes.get("k_nodes", [])
        d_nodes = graph_nodes.get("d_nodes", [])
        
        print(f"\n📊 Graph Statistics:")
        print(f"  - Total P-Nodes: {len(p_nodes)}")
        print(f"  - Total K-Nodes: {len(k_nodes)}")
        print(f"  - Total D-Nodes: {len(d_nodes)}")
        
        # 统计 Shadow Nodes
        shadow_count = sum(1 for p in p_nodes if p.get("status") == "Missing")
        print(f"  - Shadow P-Nodes: {shadow_count}")
        
        # 统计来源
        gg_k_count = sum(1 for k in k_nodes if k.get("source_type") == "GoldenGraph")
        live_k_count = sum(1 for k in k_nodes if k.get("source_type") == "LiveSearch")
        print(f"  - K-Nodes from GoldenGraph: {gg_k_count}")
        print(f"  - K-Nodes from LiveSearch: {live_k_count}")
        
        # 打印 Hybrid Stats
        hybrid_stats = graph_json.get("hybrid_engine_stats", {})
        print_hybrid_stats(hybrid_stats)
        
        # 打印 K-Nodes 详情
        print_k_nodes(k_nodes, "Final K-Nodes")
        
        # 保存图谱
        if args.save_graph:
            output_dir = PROJECT_ROOT / args.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            graph_file = output_dir / f"{args.case_id}_{args.case_type}_test_graph.json"
            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(graph_json, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Graph saved to: {graph_file}")
        
    except Exception as e:
        print(f"\n❌ Phase 2 failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ==================== Phase 3 (Optional) ====================
    if not args.skip_phase3:
        print_separator("Phase 3: Final Judgment")
        
        start_time = time.time()
        
        try:
            # 生成 Graph Summary 和 Naive Scores
            graph_summary = summarize_graph_for_judge(graph_json)
            state["graph_summary"] = graph_summary
            
            naive_scores = calculate_naive_scores(graph_json)
            state["naive_scores"] = naive_scores
            
            print(f"📊 Naive Scores: {naive_scores}")
            
            # 导入 Phase 3
            from src.agents.phase3_memory_judge import create_judge_agent
            
            phase3 = create_judge_agent(
                llm_client=llm_client,
                model_name=model_name,
                use_memory=False,  # 测试时不使用 Memory Bank
                memory_bank=None
            )
            
            state = phase3.process(state)
            
            if state.get("status") == "failed":
                raise Exception(f"Phase 3 error: {state.get('error_log')}")
            
            phase3_time = time.time() - start_time
            
            print(f"\n✅ Phase 3 Complete ({phase3_time:.1f}s)")
            
            # 提取结果
            final_output = state.get("final_output", {})
            final_id = final_output.get("final_diagnosis_id")
            final_name = final_output.get("final_diagnosis_name")
            verdict_status = final_output.get("status")
            
            print(f"\n🎯 Final Diagnosis: {final_name} (ID: {final_id})")
            print(f"📋 Verdict Status: {verdict_status}")
            
            # 判断正确性
            is_correct = (final_id == gt_id) if gt_id else False
            print(f"✅ Correct: {is_correct}")
            
        except Exception as e:
            print(f"\n❌ Phase 3 failed: {e}")
            traceback.print_exc()
    
    # ==================== 完成 ====================
    print_separator("Test Complete")
    print("✅ All phases executed successfully!")
    
    # 关键指标摘要
    print("\n📊 Key Metrics Summary:")
    print(f"  - Total K-Nodes: {len(k_nodes)}")
    print(f"  - Shadow P-Nodes: {shadow_count}")
    print(f"  - Pruned K-Nodes: {hybrid_stats.get('pruned_k_nodes', 0)}")
    print(f"  - Kept Pathognomonic: {hybrid_stats.get('kept_k_nodes_pathognomonic', 0)}")
    
    if shadow_count > 30:
        print(f"\n⚠️  WARNING: Shadow count ({shadow_count}) is still high!")
        print("    Consider reviewing the pruning logic.")
    else:
        print(f"\n✅ Shadow count ({shadow_count}) is within acceptable range.")


if __name__ == "__main__":
    main()

