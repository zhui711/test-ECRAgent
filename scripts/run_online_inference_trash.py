#!/usr/bin/env python3
"""
Online Inference Script
========================

使用 Golden Graph + Memory Bank 进行 Online 推理的主入口脚本。

功能：
1. 加载 test_verify/ 目录下的测试数据
2. 使用 Phase 2 Hybrid Engine (Golden Graph + Live Search)
3. 使用 Memory-Augmented Judge (Few-Shot Context)
4. 保存推理结果和图谱

Usage:
    python scripts/run_online_inference.py
    python scripts/run_online_inference.py --input-dir test_verify --output-dir output_online
    python scripts/run_online_inference.py --limit 10 --parallel --workers 4
    python scripts/run_online_inference.py --retry-failed
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

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
from src.agents.phase3_memory_judge import MemoryAugmentedJudge, create_judge_agent
from src.memory.memory_bank import MemoryBankManager
from src.utils.graph_tools import summarize_graph_for_judge, calculate_naive_scores


# ==================== 数据结构 ====================

@dataclass
class OnlineResult:
    """单个 Case 的推理结果"""
    case_id: str
    case_type: str
    status: str  # "success" | "error"
    ground_truth_id: Optional[str] = None
    ground_truth_name: Optional[str] = None
    phase1_diagnosis_id: Optional[str] = None
    phase1_diagnosis_name: Optional[str] = None
    final_diagnosis_id: Optional[str] = None
    final_diagnosis_name: Optional[str] = None
    verdict_status: Optional[str] = None  # "Confirm" | "Overturn" | "Fallback"
    is_correct: bool = False
    processing_time: float = 0.0
    error_message: Optional[str] = None
    hybrid_stats: Optional[Dict[str, int]] = None
    memory_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==================== 输出管理器 ====================

class OnlineOutputManager:
    """输出目录和文件管理器"""
    
    def __init__(self, output_dir: str = "output_online"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.graphs_dir = self.output_dir / "graphs"
        self.graphs_dir.mkdir(exist_ok=True)
        
        # 结果文件
        self.results_file = self.output_dir / "results_detail.jsonl"
        self.summary_file = self.output_dir / "inference_summary.json"
        self.error_log_file = self.output_dir / "error_log.jsonl"
        
        print(f"[OutputManager] Output directory: {self.output_dir}")
    
    def save_result(self, result: OnlineResult) -> None:
        """追加保存单个结果到 JSONL"""
        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    
    def save_graph(self, case_id: str, graph_json: Dict[str, Any]) -> None:
        """保存图谱 JSON"""
        filepath = self.graphs_dir / f"{case_id}_online_graph.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)
    
    def save_error(self, case_id: str, case_type: str, error_message: str) -> None:
        """保存错误日志"""
        error_record = {
            "case_id": case_id,
            "case_type": case_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.error_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
    
    def save_summary(self, summary: Dict[str, Any]) -> None:
        """保存运行摘要"""
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def load_processed_cases(self) -> Dict[str, str]:
        """加载已处理的 Case（用于断点续传）"""
        processed = {}
        if self.results_file.exists():
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            key = f"{record['case_id']}_{record['case_type']}"
                            processed[key] = record.get("status", "unknown")
                        except:
                            pass
        return processed
    
    def get_error_cases(self) -> List[Tuple[str, str]]:
        """获取需要重试的错误 Case"""
        error_cases = []
        processed = self.load_processed_cases()
        
        for key, status in processed.items():
            if status == "error":
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    case_id, case_type = parts
                    error_cases.append((case_id, case_type))
        
        return error_cases


# ==================== 在线推理器 ====================

class OnlineInferenceRunner:
    """Online 推理运行器"""
    
    # 消融实验: 用于禁用 Golden Graph 的假目录
    ABLATION_FAKE_DIR = "/tmp/__ablation_no_golden_graph__"
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_manager: OnlineOutputManager,
        golden_graph_dir: str = "golden_graphs",
        memory_bank_dir: str = "memory_bank",
        use_memory: bool = True,
        use_golden_graph: bool = True,  # [消融实验] 是否使用 Golden Graph
        parallel: bool = False,
        max_workers: int = 4
    ):
        """
        初始化 Online 推理运行器
        
        Args:
            config: 配置字典
            output_manager: 输出管理器
            golden_graph_dir: Golden Graph 目录
            memory_bank_dir: Memory Bank 目录
            use_memory: 是否使用 Memory Bank
            use_golden_graph: [消融实验] 是否使用 Golden Graph (False 时降级为纯 Live Search)
            parallel: 是否并发处理
            max_workers: 并发 worker 数量
        """
        self.config = config
        self.output_manager = output_manager
        self.parallel = parallel
        self.max_workers = max_workers
        self.use_memory = use_memory
        self.use_golden_graph = use_golden_graph  # 记录消融状态
        
        # 初始化 LLM 客户端
        api_config = config.get("api", {})
        api_key = os.getenv("YUNWU_API_KEY")
        if not api_key:
            raise ValueError("YUNWU_API_KEY not found in environment")
        
        self.llm_client = LLMClient(
            base_url=api_config.get("base_url", "https://yunwu.ai/v1"),
            api_key=api_key,
            timeout=api_config.get("timeout", 120)
        )
        self.model_name = api_config.get("model_name", "qwen3-32b")
        
        # 初始化 Phase 1
        self.phase1 = Phase1Manager(self.llm_client, self.model_name)
        
        # [消融实验] 初始化 Phase 2 Hybrid Engine
        # 当 use_golden_graph=False 时，传入假目录禁用 Golden Graph
        if use_golden_graph:
            actual_gg_dir = golden_graph_dir
            actual_refined_dir = "golden_graphs_refined"  # 默认精炼目录
        else:
            # 消融模式: 传入不存在的假目录，让 GoldenGraphLoader 加载 0 个图
            actual_gg_dir = self.ABLATION_FAKE_DIR
            actual_refined_dir = self.ABLATION_FAKE_DIR
            print(f"[OnlineRunner] ⚠️ ABLATION MODE: Golden Graph disabled (using fake dir)")
        
        self.phase2 = Phase2HybridEngine(
            llm_client=self.llm_client,
            model_name=self.model_name,
            golden_graph_dir=actual_gg_dir,
            refined_graph_dir=actual_refined_dir  # 传入消融目录
        )
        
        # 初始化 Memory Bank
        if use_memory:
            self.memory_bank = MemoryBankManager(output_dir=memory_bank_dir)
            try:
                self.memory_bank.load()
                print(f"[OnlineRunner] Memory Bank loaded: {self.memory_bank.get_statistics()}")
            except Exception as e:
                print(f"[OnlineRunner] Warning: Failed to load Memory Bank: {e}")
                self.memory_bank = None
        else:
            self.memory_bank = None
        
        # 初始化 Phase 3 Memory-Augmented Judge
        self.phase3 = create_judge_agent(
            llm_client=self.llm_client,
            model_name=self.model_name,
            # use_memory=False,
            # memory_bank=None
            use_memory=use_memory and self.memory_bank is not None,
            memory_bank=self.memory_bank
        )
        
        print(f"[OnlineRunner] Initialized with:")
        print(f"  - Model: {self.model_name}")
        print(f"  - Golden Graph: {'DISABLED (Ablation)' if not use_golden_graph else golden_graph_dir}")
        print(f"  - Memory Bank: {'Enabled' if use_memory and self.memory_bank else 'Disabled'}")
        print(f"  - Parallel: {parallel} (workers: {max_workers})")
    
    def process_single_case(
        self,
        case_dir: Path,
        case_type: str
    ) -> OnlineResult:
        """
        处理单个 Case
        
        Args:
            case_dir: Case 目录
            case_type: "control_case" 或 "trap_case"
        
        Returns:
            OnlineResult 实例
        """
        case_id = case_dir.name
        start_time = time.time()
        
        try:
            # 加载 Case 数据 (与 io_utils.load_case_data 保持一致)
            case_file = case_dir / "final_benchmark_pair.json"
            if not case_file.exists():
                raise FileNotFoundError(f"final_benchmark_pair.json not found in {case_dir}")
            
            with open(case_file, "r", encoding="utf-8") as f:
                case_data = json.load(f)
            
            # 获取对应类型的数据 (key 是 "control_case" 或 "trap_case")
            if case_type not in case_data:
                raise ValueError(f"Case type '{case_type}' not found in final_benchmark_pair.json")
            
            case_info = case_data[case_type]
            narrative = case_info.get("narrative", "")
            ground_truth = case_info.get("ground_truth", "")
            
            # 解析 Ground Truth ID
            gt_id = None
            for did, dname in DIAGNOSIS_ID_MAP.items():
                if dname.lower() == ground_truth.lower():
                    gt_id = did
                    break
            
            # Phase 1
            phase1_result = self.phase1.process(narrative)
            
            if phase1_result.get("error"):
                raise Exception(f"Phase 1 error: {phase1_result['error']}")
            
            # 构建初始状态（包含所有 AgentState 字段）
            state: AgentState = {
                "case_id": case_id,
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
                "memory_context": None,
                "memory_records": None  # Memory Bank 检索结果
            }
            
            # Phase 2 (Hybrid Engine)
            state = self.phase2.process(state)
            
            if state.get("status") == "failed":
                raise Exception(f"Phase 2 error: {state.get('error_log')}")
            
            # 生成 Graph Summary 和 Naive Scores
            graph_json = state.get("graph_json", {})
            
            graph_summary = summarize_graph_for_judge(graph_json)
            state["graph_summary"] = graph_summary
            
            naive_scores = calculate_naive_scores(graph_json)
            state["naive_scores"] = naive_scores
            
            # Phase 3 (Memory-Augmented Judge)
            state = self.phase3.process(state)
            
            if state.get("status") == "failed":
                raise Exception(f"Phase 3 error: {state.get('error_log')}")
            
            # 提取结果
            final_output = state.get("final_output", {})
            final_id = final_output.get("final_diagnosis_id")
            final_name = final_output.get("final_diagnosis_name")
            verdict_status = final_output.get("status")
            
            phase1_id = phase1_result.get("final_diagnosis_id")
            phase1_name = DIAGNOSIS_ID_MAP.get(phase1_id, "Unknown")
            
            is_correct = (final_id == gt_id) if gt_id else False
            
            # 保存图谱
            self.output_manager.save_graph(f"{case_id}_{case_type}", graph_json)
            
            processing_time = time.time() - start_time
            
            # 获取 Hybrid Engine 统计（从 graph_json 获取，避免并发时的竞态条件）
            hybrid_stats = graph_json.get("hybrid_engine_stats", {})
            memory_used = final_output.get("memory_retrieval_used", False)
            
            return OnlineResult(
                case_id=case_id,
                case_type=case_type,
                status="success",
                ground_truth_id=gt_id,
                ground_truth_name=ground_truth,
                phase1_diagnosis_id=phase1_id,
                phase1_diagnosis_name=phase1_name,
                final_diagnosis_id=final_id,
                final_diagnosis_name=final_name,
                verdict_status=verdict_status,
                is_correct=is_correct,
                processing_time=processing_time,
                error_message=None,
                hybrid_stats=hybrid_stats,
                memory_used=memory_used
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            
            # 保存错误日志
            self.output_manager.save_error(case_id, case_type, error_msg)
            
            return OnlineResult(
                case_id=case_id,
                case_type=case_type,
                status="error",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def run(
        self,
        case_dirs: List[Path],
        case_types: List[str],
        retry_failed: bool = False
    ) -> List[OnlineResult]:
        """
        运行推理
        
        Args:
            case_dirs: Case 目录列表
            case_types: 要处理的 Case 类型列表
            retry_failed: 是否只重试之前失败的 Case
        
        Returns:
            OnlineResult 列表
        """
        # 构建任务列表
        tasks = []
        
        if retry_failed:
            # 只处理之前失败的 Case
            error_cases = self.output_manager.get_error_cases()
            for case_id, case_type in error_cases:
                for case_dir in case_dirs:
                    if case_dir.name == case_id:
                        tasks.append((case_dir, case_type))
                        break
            print(f"[OnlineRunner] Retrying {len(tasks)} failed cases")
        else:
            # 处理所有 Case
            processed = self.output_manager.load_processed_cases()
            
            for case_dir in case_dirs:
                for case_type in case_types:
                    key = f"{case_dir.name}_{case_type}"
                    if key not in processed or processed[key] == "error":
                        tasks.append((case_dir, case_type))
            
            print(f"[OnlineRunner] Processing {len(tasks)} cases")
        
        if not tasks:
            print("[OnlineRunner] No cases to process")
            return []
        
        # 执行推理
        if self.parallel:
            results = self._run_parallel(tasks)
        else:
            results = self._run_sequential(tasks)
        
        return results
    
    def _run_sequential(self, tasks: List[Tuple[Path, str]]) -> List[OnlineResult]:
        """串行执行"""
        results = []
        
        with tqdm(total=len(tasks), desc="Online Inference") as pbar:
            for case_dir, case_type in tasks:
                result = self.process_single_case(case_dir, case_type)
                results.append(result)
                
                # 保存结果
                self.output_manager.save_result(result)
                
                # 更新进度条
                status_icon = "✓" if result.status == "success" else "✗"
                correct_icon = "🎯" if result.is_correct else ""
                pbar.set_postfix_str(f"{status_icon} {result.case_id} {correct_icon}")
                pbar.update(1)
        
        return results
    
    def _run_parallel(self, tasks: List[Tuple[Path, str]]) -> List[OnlineResult]:
        """并发执行"""
        results = []
        
        with tqdm(total=len(tasks), desc="Online Inference (Parallel)") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self.process_single_case, case_dir, case_type): (case_dir, case_type)
                    for case_dir, case_type in tasks
                }
                
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                    except Exception as e:
                        case_dir, case_type = future_to_task[future]
                        result = OnlineResult(
                            case_id=case_dir.name,
                            case_type=case_type,
                            status="error",
                            error_message=str(e)
                        )
                    
                    results.append(result)
                    self.output_manager.save_result(result)
                    
                    status_icon = "✓" if result.status == "success" else "✗"
                    pbar.set_postfix_str(f"{status_icon} {result.case_id}")
                    pbar.update(1)
        
        return results


# ==================== 配置和参数 ====================

def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MDT-Agent Online Inference with Golden Graph + Memory Bank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行所有测试 Case
  python scripts/run_online_inference.py
  
  # 限制处理数量
  python scripts/run_online_inference.py --limit 10
  
  # 并发处理
  python scripts/run_online_inference.py --parallel --workers 4
  
  # 重试失败的 Case
  python scripts/run_online_inference.py --retry-failed
  
  # 不使用 Memory Bank
  python scripts/run_online_inference.py --no-memory
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="test_verify",
        help="输入数据目录（默认: test_verify）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_online",
        help="输出目录（默认: output_online）"
    )
    
    parser.add_argument(
        "--golden-graph-dir",
        type=str,
        default="golden_graphs_refined",
        help="Golden Graph 目录（默认: golden_graphs_refined）"
    )
    
    parser.add_argument(
        "--memory-bank-dir",
        type=str,
        default="memory_bank",
        help="Memory Bank 目录（默认: memory_bank）"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的 Case 数量"
    )
    
    parser.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help="指定要处理的 Case ID（逗号分隔）"
    )
    
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="跳过 control_case"
    )
    
    parser.add_argument(
        "--skip-trap",
        action="store_true",
        help="跳过 trap_case"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并发处理"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发 worker 数量（默认: 4）"
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="不使用 Memory Bank"
    )
    
    parser.add_argument(
        "--no-golden-graph",
        action="store_true",
        help="[消融实验] 不使用 Golden Graph，仅使用 Live Search"
    )
    
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="只重试之前失败的 Case"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖配置文件中的模型名称"
    )
    
    return parser.parse_args()


# ==================== 主函数 ====================

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
    
    # 初始化输出管理器
    output_manager = OnlineOutputManager(args.output_dir)
    
    # 获取 Case 目录列表
    input_path = PROJECT_ROOT / args.input_dir
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        sys.exit(1)
    
    # 筛选 Case 目录
    if args.case_ids:
        specified_ids = set(args.case_ids.split(","))
        case_dirs = sorted([
            d for d in input_path.iterdir()
            if d.is_dir() and d.name in specified_ids
        ])
    else:
        case_dirs = sorted([
            d for d in input_path.iterdir()
            if d.is_dir() and d.name.startswith("case_")
        ])
    
    # 应用 limit
    if args.limit:
        case_dirs = case_dirs[:args.limit]
    
    # 确定要处理的 Case 类型
    case_types = []
    if not args.skip_control:
        case_types.append("control_case")
    if not args.skip_trap:
        case_types.append("trap_case")
    
    if not case_types:
        print("❌ No case types to process")
        sys.exit(1)
    
    # [消融实验] 检测消融模式
    use_golden_graph = not getattr(args, 'no_golden_graph', False)
    ablation_mode = args.no_memory or not use_golden_graph
    
    # 打印运行信息
    print("=" * 70)
    if ablation_mode:
        print("MDT-Agent Online Inference [ABLATION MODE]")
    else:
        print("MDT-Agent Online Inference (Golden Graph + Memory Bank)")
    print("=" * 70)
    print(f"📁 Input directory: {input_path}")
    print(f"📁 Output directory: {output_manager.output_dir}")
    print(f"📁 Golden Graph: {'⚠️ DISABLED (Ablation)' if not use_golden_graph else args.golden_graph_dir}")
    print(f"📁 Memory Bank: {args.memory_bank_dir}")
    print(f"📊 Total case directories: {len(case_dirs)}")
    print(f"📋 Case types: {', '.join(case_types)}")
    print(f"🔧 Model: {config['api']['model_name']}")
    print(f"📊 Golden Graph: {'⚠️ DISABLED (Ablation)' if not use_golden_graph else 'Enabled'}")
    print(f"🧠 Memory Bank: {'⚠️ DISABLED (Ablation)' if args.no_memory else 'Enabled'}")
    print(f"⚡ Parallel mode: {'Yes' if args.parallel else 'No'}")
    if args.parallel:
        print(f"👷 Workers: {args.workers}")
    if ablation_mode:
        print("-" * 70)
        print("⚠️  ABLATION EXPERIMENT: Some offline components are disabled")
    print("=" * 70)
    
    # 初始化运行器
    try:
        runner = OnlineInferenceRunner(
            config=config,
            output_manager=output_manager,
            golden_graph_dir=args.golden_graph_dir,
            memory_bank_dir=args.memory_bank_dir,
            use_memory=not args.no_memory,
            use_golden_graph=use_golden_graph,  # [消融实验] 控制 Golden Graph
            parallel=args.parallel,
            max_workers=args.workers
        )
    except Exception as e:
        print(f"❌ Failed to initialize runner: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # 运行推理
    start_time = time.time()
    results = runner.run(
        case_dirs=case_dirs,
        case_types=case_types,
        retry_failed=args.retry_failed
    )
    total_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r.status == "success")
    error_count = sum(1 for r in results if r.status == "error")
    correct_count = sum(1 for r in results if r.is_correct)
    
    # 按 Case Type 统计
    control_results = [r for r in results if r.case_type == "control_case"]
    trap_results = [r for r in results if r.case_type == "trap_case"]
    
    control_correct = sum(1 for r in control_results if r.is_correct)
    trap_correct = sum(1 for r in trap_results if r.is_correct)
    
    # 统计 Overturn
    overturn_count = sum(1 for r in results if r.verdict_status == "Overturn")
    
    # 保存摘要
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "total_processed": len(results),
        "success": success_count,
        "error": error_count,
        "correct": correct_count,
        "accuracy": correct_count / success_count if success_count > 0 else 0,
        "control_accuracy": control_correct / len(control_results) if control_results else 0,
        "trap_accuracy": trap_correct / len(trap_results) if trap_results else 0,
        "overturn_count": overturn_count,
        "total_time_seconds": total_time,
        "avg_time_per_case": total_time / len(results) if results else 0,
        "config": {
            "model": config["api"]["model_name"],
            "use_memory": not args.no_memory,
            "golden_graph_dir": args.golden_graph_dir,
            "memory_bank_dir": args.memory_bank_dir,
            "parallel": args.parallel,
            "workers": args.workers if args.parallel else 1
        }
    }
    output_manager.save_summary(summary)
    
    # 打印最终统计
    print("\n" + "=" * 70)
    print("Online Inference Complete!")
    print("=" * 70)
    print(f"✅ Success: {success_count}")
    print(f"❌ Error: {error_count}")
    print(f"🎯 Correct: {correct_count}/{success_count} ({summary['accuracy']:.2%})")
    if control_results:
        print(f"  - Control: {control_correct}/{len(control_results)} ({summary['control_accuracy']:.2%})")
    if trap_results:
        print(f"  - Trap: {trap_correct}/{len(trap_results)} ({summary['trap_accuracy']:.2%})")
    print(f"🔄 Overturn: {overturn_count}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Results saved to: {output_manager.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

