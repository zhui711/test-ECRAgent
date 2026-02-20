#!/usr/bin/env python3
"""
Offline Training Entry Point
============================

Offline 训练的入口脚本。

功能：
1. 加载配置和训练列表
2. 初始化 OfflineTrainer（支持串行和并发模式）
3. 执行训练循环
4. 支持从检查点恢复
5. 生成训练报告

使用方法:
    # 串行模式（默认）
    python scripts/run_offline_training.py [--dry-run N] [--resume]
    
    # 并发模式
    python scripts/run_offline_training.py --concurrent [--workers 5]

参数:
    --dry-run N: 只处理前 N 个案例（用于测试）
    --resume: 从检查点恢复训练
    --config PATH: 指定配置文件路径
    --concurrent: 启用并发处理模式
    --workers N: 并发线程数（默认从配置读取）
    --batch-size N: 每批处理的案例数（并发模式，默认 20）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from tqdm import tqdm

from src.training.offline_trainer import OfflineTrainer, ConcurrentOfflineTrainer
from src.utils.api_client import LLMClient
from src.memory.embedding_client import EmbeddingClient


class ConcurrentAblationOfflineTrainer(ConcurrentOfflineTrainer):
    """
    Concurrent Ablation Trainer (No-Critic Mode)
    
    Inherits from ConcurrentOfflineTrainer but overrides logic to disable 
    critic intervention when --no-critic is enabled.
    """
    def __init__(self, *args, no_critic: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_critic = no_critic

    def process_case_for_batch(
        self,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str,
        narrative: str
    ) -> dict:
        """
        Modified process_case_for_batch for Ablation (No-Critic)
        """
        if not self.no_critic:
            return super().process_case_for_batch(
                case_id, case_type, ground_truth_id, ground_truth_name, narrative
            )

        # Simplified logic for No-Critic mode (identical to AblationOfflineTrainer but for batch)
        result = {
            "case_id": case_id,
            "type": case_type,
            "ground_truth_id": ground_truth_id,
            "ground_truth_name": ground_truth_name,
            "timestamp": datetime.now().isoformat(),
            "attempts": [],
            "final_status": "pending",
            "total_retries": 0,
            "aggregated": False,
            "_aggregation_data": None
        }
        
        # Reset history (per thread/instance, though critical agent might vary)
        # Note: In concurrent mode, we rely on local variables mostly.
        
        state = self._create_initial_state(case_id, narrative)
        
        print(f"[ConcurrentAblation] Processing {case_id} (No-Critic)...")
        
        for attempt in range(self.max_retries + 1):
            attempt_num = attempt + 1
            
            try:
                # Run Pipeline
                state = self._run_pipeline(state)
                
                # Check Result
                is_correct, error_type = self._check_result(state, ground_truth_id)
                
                # Log attempt
                p1_result = state.get("phase1_result") or {}
                f_output = state.get("final_output") or {}
                attempt_record = {
                    "attempt": attempt_num,
                    "phase1_top5": p1_result.get("top_candidates", []) or [],
                    "initial_diagnosis_id": p1_result.get("final_diagnosis_id", "") or "",
                    "final_diagnosis_id": f_output.get("final_diagnosis_id", "") or "",
                    "correct": is_correct,
                    "error_type": error_type,
                    "critique": None
                }
                
                if is_correct:
                    result["attempts"].append(attempt_record)
                    result["final_status"] = "success"
                    result["total_retries"] = attempt
                    
                    # Prepare aggregation data
                    result["_aggregation_data"] = {
                        "state": state,
                        "case_id": case_id,
                        "case_type": case_type,
                        "ground_truth_id": ground_truth_id,
                        "ground_truth_name": ground_truth_name
                    }
                    
                    with self._stats_lock:
                        if attempt == 0:
                            self.stats["success_first_try"] += 1
                        else:
                            self.stats["success_after_retry"] += 1
                            
                    return result
                
                else:
                    # In Ablation mode, we skip critique loop.
                    # Behavior alignment: Fail immediately on first error (like sequential mode)
                    print(f"    [ConcurrentAblation] {case_id} ✗ Incorrect. Skipping critique & Retry.")
                    result["attempts"].append(attempt_record)
                    result["final_status"] = "failed"
                    result["total_retries"] = attempt
                    
                    with self._stats_lock:
                        self.stats["failed"] += 1
                    
                    return result
            
            except Exception as e:
                result["attempts"].append({"attempt": attempt_num, "error": str(e)})
                if attempt >= self.max_retries:
                    break
        
        # If loop finishes without success
        result["final_status"] = "failed"
        result["total_retries"] = self.max_retries
        
        with self._stats_lock:
            self.stats["failed"] += 1
            
        return result


class AblationOfflineTrainer(OfflineTrainer):
    """
    Ablation Offline Trainer (No-Critic Mode)
    
    Inherits from OfflineTrainer but overrides the retry logic to disable 
    critic intervention when --no-critic is enabled.
    """
    def __init__(self, *args, no_critic: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_critic = no_critic

    def process_case(
        self,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str,
        narrative: str
    ) -> dict:
        """
        Modified process_case to support ablation (no-critic)
        """
        # Call parent's logic if critic is enabled
        if not self.no_critic:
            return super().process_case(
                case_id, case_type, ground_truth_id, ground_truth_name, narrative
            )

        # Re-implement simplified logic for No-Critic mode
        result = {
            "case_id": case_id,
            "type": case_type,
            "ground_truth_id": ground_truth_id,
            "ground_truth_name": ground_truth_name,
            "timestamp": datetime.now().isoformat(),
            "attempts": [],
            "final_status": "pending",
            "total_retries": 0,
            "aggregated": False
        }
        
        # Reset history
        self.critical_agent.reset_history()
        
        # Initial state
        state = self._create_initial_state(case_id, narrative)
        
        # Single attempt (or max retries without critic)
        # We allow checking retries for robustness, but NO critic injection.
        print(f"\n[AblationTrainer] Processing {case_id} (No-Critic Mode)...")
        
        for attempt in range(self.max_retries + 1):
            attempt_num = attempt + 1
            print(f"[AblationTrainer] Attempt {attempt_num}/{self.max_retries + 1}")
            
            try:
                # Run Pipeline
                state = self._run_pipeline(state)
                
                # Check Result
                is_correct, error_type = self._check_result(
                    state, ground_truth_id
                )
                
                # Log attempt
                p1_result = state.get("phase1_result") or {}
                f_output = state.get("final_output") or {}
                attempt_record = {
                    "attempt": attempt_num,
                    "phase1_top5": p1_result.get("top_candidates", []) or [],
                    "initial_diagnosis_id": p1_result.get("final_diagnosis_id", "") or "",
                    "final_diagnosis_id": f_output.get("final_diagnosis_id", "") or "",
                    "correct": is_correct,
                    "error_type": error_type,
                    "critique": None  # No critique in ablation
                }
                
                if is_correct:
                    print(f"[AblationTrainer] ✓ Correct on attempt {attempt_num}")
                    result["attempts"].append(attempt_record)
                    result["final_status"] = "success"
                    result["total_retries"] = attempt
                    result["aggregated"] = True
                    
                    # Aggregate
                    self._on_success(state, case_id, case_type, ground_truth_id, ground_truth_name)
                    
                    if attempt == 0:
                        self.stats["success_first_try"] += 1
                    else:
                        self.stats["success_after_retry"] += 1
                        
                    return result
                
                else:
                    print(f"[AblationTrainer] ✗ Incorrect. Error: {error_type}")
                    print(f"[AblationTrainer] Ablation: Skipping critique injection.")
                    
                    # Mark as failed immediately - NO RETRY WITH CRITIC
                    attempt_record["error"] = "Incorrect (Ablation: No Retry)"
                    result["attempts"].append(attempt_record)
                    result["final_status"] = "failed"
                    self.stats["failed"] += 1
                    return result
                    
            except Exception as e:
                print(f"[AblationTrainer] Error: {e}")
                result["attempts"].append({"attempt": attempt_num, "error": str(e)})
                # Continue if it's a code error, but usually we just fail
                if attempt >= self.max_retries:
                    break
        
        result["final_status"] = "failed"
        self.stats["failed"] += 1
        return result


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "config" / "settings.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_list(train_list_path: str = None) -> list:
    """加载训练列表"""
    if train_list_path is None:
        train_list_path = project_root / "config" / "offline_train_list.json"
    
    with open(train_list_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_case_narrative(case_path: str, case_type: str) -> str:
    """
    加载案例的 narrative
    
    Args:
        case_path: 案例文件路径
        case_type: 类型 (control/trap)
    
    Returns:
        narrative 文本
    """
    case_full_path = project_root / case_path
    
    with open(case_full_path, "r", encoding="utf-8") as f:
        case_data = json.load(f)
    
    # 根据类型获取对应的 narrative
    case_key = f"{case_type}_case"
    return case_data.get(case_key, {}).get("narrative", "")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run Offline Training")
    parser.add_argument(
        "--dry-run", 
        type=int, 
        default=None,
        help="Only process first N cases (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--train-list",
        type=str,
        default=None,
        help="Path to training list JSON"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Enable concurrent processing mode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent workers (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for concurrent processing (default: 20)"
    )
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Enable Ablation Mode: Disable GPT-5 Critic intervention"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (overrides config)"
    )
    
    args = parser.parse_args()
    
    # 打印启动信息
    print("=" * 70)
    print("MDT-Agent Offline Training (v2.0)")
    print("=" * 70)
    print(f"Start Time: {datetime.now().isoformat()}")
    print(f"Dry Run: {args.dry_run if args.dry_run else 'No'}")
    print(f"Resume: {args.resume}")
    print(f"Mode: {'Concurrent' if args.concurrent else 'Sequential'}")
    print(f"Ablation (No-Critic): {'YES' if args.no_critic else 'No'}")
    print("=" * 70)
    
    # Load Config
    print("\n[Setup] Loading configuration...")
    config = load_config(args.config)
    
    api_config = config.get("api", {})
    critical_config = config.get("critical_model", {})
    embedding_config = config.get("embedding", {})
    offline_config = config.get("offline_training", {})

    # Override config with --output-dir if provided
    if args.output_dir:
        output_path = Path(args.output_dir)
        print(f"[Override] Using custom output directory: {output_path}")
        offline_config["golden_graph_dir"] = str(output_path / "golden_graphs")
        offline_config["memory_bank_dir"] = str(output_path / "memory_bank")
        offline_config["log_dir"] = str(output_path / "logs")
    
    print(f"  - Main Model: {api_config.get('model_name', 'N/A')}")
    print(f"  - Critical Model: {critical_config.get('model_name', 'N/A')}")
    print(f"  - Embedding Model: {embedding_config.get('model_name', 'N/A')}")
    print(f"  - Max Retries: {offline_config.get('max_critique_retries', 3)}")
    print(f"  - Aggregation Step: {offline_config.get('aggregation_step', 5)}")
    
    # 加载训练列表
    print("\n[Setup] Loading training list...")
    train_list = load_train_list(args.train_list)
    print(f"  - Total cases in list: {len(train_list)}")
    
    # 应用 dry-run 限制
    if args.dry_run:
        train_list = train_list[:args.dry_run]
        print(f"  - Dry-run mode: processing only {len(train_list)} cases")
    
    # 初始化 LLM 客户端
    print("\n[Setup] Initializing LLM clients...")
    
    main_client = LLMClient(
        base_url=api_config.get("base_url", "https://yunwu.ai/v1"),
        api_key=None,  # 从环境变量读取
        timeout=api_config.get("timeout", 120)
    )
    
    critical_client = LLMClient(
        base_url=critical_config.get("base_url", api_config.get("base_url")),
        api_key=None,
        timeout=critical_config.get("timeout", 180)
    )
    
    # 初始化 Embedding 客户端
    try:
        embedding_client = EmbeddingClient(
            base_url=embedding_config.get("base_url", "https://yunwu.ai/v1"),
            model=embedding_config.get("model_name", "text-embedding-3-small"),
            dimension=embedding_config.get("dimension", 1536)
        )
        print("  - Embedding client initialized")
    except Exception as e:
        print(f"  - Warning: Embedding client init failed: {e}")
        print("  - Memory Bank will use fallback embeddings")
        embedding_client = None
    
    # 确定并发线程数
    max_workers = args.workers or offline_config.get("max_workers", 5)
    
    # 初始化 Trainer
    if args.concurrent:
        if args.no_critic:
            print(f"\n[Setup] Initializing ConcurrentAblationOfflineTrainer (workers={max_workers})...")
            trainer = ConcurrentAblationOfflineTrainer(
                max_workers=max_workers,
                llm_client=main_client,
                critical_client=critical_client,
                embedding_client=embedding_client,
                model_name=api_config.get("model_name", "qwen3-32b"),
                critical_model_name=critical_config.get("model_name", "gpt-5"),
                max_retries=offline_config.get("max_critique_retries", 3),
                aggregation_step=offline_config.get("aggregation_step", 5),
                golden_graph_dir=offline_config.get("golden_graph_dir", "golden_graphs"),
                memory_bank_dir=offline_config.get("memory_bank_dir", "memory_bank"),
                log_dir=offline_config.get("log_dir", "logs"),
                no_critic=True
            )
        else:
            print(f"\n[Setup] Initializing ConcurrentOfflineTrainer (workers={max_workers})...")
            trainer = ConcurrentOfflineTrainer(
                max_workers=max_workers,
                llm_client=main_client,
                critical_client=critical_client,
                embedding_client=embedding_client,
                model_name=api_config.get("model_name", "qwen3-32b"),
                critical_model_name=critical_config.get("model_name", "gpt-5"),
                max_retries=offline_config.get("max_critique_retries", 3),
                aggregation_step=offline_config.get("aggregation_step", 5),
                golden_graph_dir=offline_config.get("golden_graph_dir", "golden_graphs"),
                memory_bank_dir=offline_config.get("memory_bank_dir", "memory_bank"),
                log_dir=offline_config.get("log_dir", "logs")
            )
            
    if not args.concurrent:
        # Sequential Mode (Standard or Ablation)
        TrainerClass = AblationOfflineTrainer if args.no_critic else OfflineTrainer
        print(f"\n[Setup] Initializing {TrainerClass.__name__} (Sequential)...")
        
        trainer = TrainerClass(
            llm_client=main_client,
            critical_client=critical_client,
            embedding_client=embedding_client,
            model_name=api_config.get("model_name", "qwen3-32b"),
            critical_model_name=critical_config.get("model_name", "gpt-5"),
            max_retries=offline_config.get("max_critique_retries", 3),
            aggregation_step=offline_config.get("aggregation_step", 5),
            golden_graph_dir=offline_config.get("golden_graph_dir", "golden_graphs"),
            memory_bank_dir=offline_config.get("memory_bank_dir", "memory_bank"),
            log_dir=offline_config.get("log_dir", "logs"),
            # Pass no_critic only if it's AblationOfflineTrainer
            **( {"no_critic": True} if args.no_critic else {} )
        )
    
    # 从检查点恢复
    if args.resume:
        print("\n[Setup] Loading checkpoint...")
        trainer.load_checkpoint()
    
    # 过滤已处理的案例
    pending_cases = [
        case for case in train_list 
        if not trainer.is_processed(case["case_id"])
    ]
    print(f"\n[Setup] Cases to process: {len(pending_cases)} (skipped: {len(train_list) - len(pending_cases)})")
    
    if not pending_cases:
        print("\n[Info] All cases already processed!")
        return
    
    # 开始训练循环
    print("\n" + "=" * 70)
    print(f"Starting Training Loop ({'Concurrent' if args.concurrent else 'Sequential'})")
    print("=" * 70 + "\n")
    
    if args.concurrent:
        # ========== 并发模式 ==========
        batch_size = args.batch_size
        total_batches = (len(pending_cases) + batch_size - 1) // batch_size
        
        print(f"[Concurrent] Processing {len(pending_cases)} cases in {total_batches} batches (size={batch_size})")
        
        # 创建线程安全的进度条
        pbar = tqdm(total=len(pending_cases), desc="Training", position=0, leave=True)
        
        def progress_callback(completed, total, result):
            """进度回调（线程安全）"""
            status_emoji = "✓" if result.get("final_status") == "success" else "✗"
            pbar.set_postfix_str(f"Last: {status_emoji} {result.get('case_id', 'N/A')[:20]}")
            pbar.update(1)
        
        all_results = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(pending_cases))
            batch_cases = pending_cases[start_idx:end_idx]
            
            print(f"\n[Batch {batch_idx + 1}/{total_batches}] Processing {len(batch_cases)} cases...")
            
            batch_results = trainer.train_batch(
                cases=batch_cases,
                load_narrative_fn=load_case_narrative,
                progress_callback=progress_callback
            )
            
            all_results.extend(batch_results)
            
            # 显示批次统计
            success_count = sum(1 for r in batch_results if r.get("final_status") == "success")
            failed_count = sum(1 for r in batch_results if r.get("final_status") == "failed")
            error_count = sum(1 for r in batch_results if r.get("final_status") == "error")
            print(f"    Batch result: ✓ {success_count} success | ✗ {failed_count} failed | ⚠ {error_count} errors")
        
        pbar.close()
        
    else:
        # ========== 串行模式（原有逻辑）==========
        for idx, case_record in enumerate(tqdm(pending_cases, desc="Training")):
            case_id = case_record["case_id"]
            case_type = case_record["type"]
            ground_truth = case_record["ground_truth"]
            ground_truth_id = case_record["ground_truth_id"]
            case_path = case_record["path"]
            
            try:
                # 加载 narrative
                narrative = load_case_narrative(case_path, case_type)
                
                if not narrative:
                    print(f"\n[Warning] Empty narrative for {case_id}, skipping...")
                    continue
                
                # 处理案例
                result = trainer.process_case(
                    case_id=case_id,
                    case_type=case_type,
                    ground_truth_id=ground_truth_id,
                    ground_truth_name=ground_truth,
                    narrative=narrative
                )
                
                # 记录结果
                trainer.log_result(result)
                
                # 打印进度
                status_emoji = "✓" if result["final_status"] == "success" else "✗"
                print(f"\n[{idx+1}/{len(pending_cases)}] {status_emoji} {case_id}: {result['final_status']}")
                
            except Exception as e:
                print(f"\n[Error] Failed to process {case_id}: {e}")
                import traceback
                traceback.print_exc()
                
                # 记录错误但继续
                error_result = {
                    "case_id": case_id,
                    "type": case_type,
                    "ground_truth_id": ground_truth_id,
                    "ground_truth_name": ground_truth,
                    "timestamp": datetime.now().isoformat(),
                    "attempts": [],
                    "final_status": "error",
                    "error": str(e),
                    "total_retries": 0,
                    "aggregated": False
                }
                trainer.log_result(error_result)
                continue
    
    # 训练完成，保存最终结果
    print("\n" + "=" * 70)
    print("Training Complete - Finalizing")
    print("=" * 70 + "\n")
    
    final_stats = trainer.finalize()
    
    # 打印最终统计
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)
    print(f"  - Total Processed: {final_stats.get('total_processed', 0)}")
    print(f"  - Success (First Try): {final_stats.get('success_first_try', 0)}")
    print(f"  - Success (After Retry): {final_stats.get('success_after_retry', 0)}")
    print(f"  - Failed: {final_stats.get('failed', 0)}")
    
    if "aggregation" in final_stats:
        agg = final_stats["aggregation"]
        print(f"\n  Golden Graph:")
        print(f"    - Unique Pathologies: {agg.get('unique_pathologies', 0)}")
        print(f"    - Total Aggregated: {agg.get('total_aggregated', 0)}")
    
    if "memory_bank" in final_stats:
        mb = final_stats["memory_bank"]
        print(f"\n  Memory Bank:")
        print(f"    - Total Cases: {mb.get('total_cases', 0)}")
        print(f"    - Overturn: {mb.get('overturn_count', 0)}")
        print(f"    - Confirm: {mb.get('confirm_count', 0)}")
    
    print("\n" + "=" * 70)
    print(f"End Time: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()


