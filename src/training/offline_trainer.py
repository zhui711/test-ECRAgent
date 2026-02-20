"""
Offline Trainer
===============

Offline 训练的核心引擎，实现完整的训练循环：
1. 加载训练案例
2. 运行 Phase 1-3 Pipeline
3. 检查结果正确性
4. 如果错误，调用 Critical Model 获取反馈
5. 重试（最多 N 次）
6. 成功后聚合到 Golden Graph 和 Memory Bank

设计原则：
- 复用现有 Agent：调用 Phase1Manager, Phase2GraphReasoning, JudgeAgent
- 精细控制：拆解调用各 Agent，便于在每个阶段注入 Hint
- 容错性：单个 Case 失败不中断整个训练
- 可恢复：支持从检查点恢复训练
- 并发支持：使用 ThreadPoolExecutor 并发处理 Cases（v2.0）
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.graph.state import AgentState
from src.agents.phase1_manager import Phase1Manager
from src.agents.phase2_investigation import Phase2GraphReasoning
from src.agents.phase3_debate import JudgeAgent
from src.agents.critical_model import CriticalModelAgent
from src.training.aggregation_manager import AggregationManager
from src.memory.memory_bank import MemoryBankManager
from src.memory.embedding_client import EmbeddingClient
from src.utils.api_client import LLMClient
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.utils.graph_tools import (
    rebuild_graph_from_json,
    serialize_graph_to_summary,
    calculate_deterministic_score
)


class OfflineTrainer:
    """
    Offline 训练器
    
    实现 Offline Training 的完整循环：
    Run -> Check -> Critique -> Retry -> Aggregate -> Store
    
    Attributes:
        phase1_manager: Phase 1 Agent
        phase2_agent: Phase 2 Agent
        judge_agent: Phase 3 Agent
        critical_agent: Teacher Agent
        aggregation_manager: Golden Graph 聚合器
        memory_bank: Memory Bank 管理器
        max_retries: 最大重试次数
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        critical_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        model_name: str = "qwen3-32b",
        critical_model_name: str = "gpt-5",
        max_retries: int = 3,
        aggregation_step: int = 5,
        golden_graph_dir: str = "golden_graphs",
        memory_bank_dir: str = "memory_bank",
        log_dir: str = "logs"
    ):
        """
        初始化 Offline Trainer
        
        Args:
            llm_client: 主 LLM 客户端（用于 Agent）
            critical_client: Teacher LLM 客户端（如果为 None，使用主客户端）
            embedding_client: Embedding 客户端
            model_name: Agent 模型名称
            critical_model_name: Teacher 模型名称
            max_retries: 最大重试次数
            aggregation_step: Golden Graph 聚合步长
            golden_graph_dir: Golden Graph 输出目录
            memory_bank_dir: Memory Bank 输出目录
            log_dir: 日志目录
        """
        # 初始化 Agents
        self.phase1_manager = Phase1Manager(llm_client, model_name=model_name)
        self.phase2_agent = Phase2GraphReasoning(llm_client, model_name=model_name)
        self.judge_agent = JudgeAgent(llm_client, model_name=model_name)
        
        # Teacher Agent（使用更强的模型）
        critical_client = critical_client or llm_client
        self.critical_agent = CriticalModelAgent(
            critical_client, 
            model_name=critical_model_name
        )
        
        # 聚合器和存储
        self.aggregation_manager = AggregationManager(
            aggregation_step=aggregation_step,
            output_dir=golden_graph_dir
        )
        
        self.memory_bank = MemoryBankManager(
            output_dir=memory_bank_dir,
            embedding_client=embedding_client
        )
        
        # 配置
        self.max_retries = max_retries
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程安全锁
        self._log_lock = threading.Lock()        # 保护日志写入
        self._stats_lock = threading.Lock()       # 保护统计更新
        self._processed_lock = threading.Lock()   # 保护已处理集合
        
        # 批量聚合缓存（用于并发模式）
        self._pending_aggregations: List[Dict[str, Any]] = []
        self._aggregation_lock = threading.Lock()
        
        # 统计
        self.stats = {
            "total_processed": 0,
            "success_first_try": 0,
            "success_after_retry": 0,
            "failed": 0,
            "errors": 0
        }
        
        # 已处理的 Case ID（用于恢复）
        self.processed_cases: set = set()
    
    def load_checkpoint(self, log_file: Optional[str] = None) -> None:
        """
        从日志文件加载检查点，恢复已处理的 Case
        
        Args:
            log_file: 日志文件路径（默认为 offline_training_log.jsonl）
        """
        log_path = Path(log_file) if log_file else self.log_dir / "offline_training_log.jsonl"
        
        if not log_path.exists():
            print(f"[Trainer] No checkpoint file found at {log_path}")
            return
        
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        case_id = record.get("case_id")
                        if case_id:
                            self.processed_cases.add(case_id)
            
            print(f"[Trainer] Loaded checkpoint: {len(self.processed_cases)} cases already processed")
            
        except Exception as e:
            print(f"[Trainer] Error loading checkpoint: {e}")
    
    def process_case(
        self,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str,
        narrative: str
    ) -> Dict[str, Any]:
        """
        处理单个训练案例
        
        Args:
            case_id: 案例 ID
            case_type: 类型 (control/trap)
            ground_truth_id: 正确诊断 ID
            ground_truth_name: 正确诊断名称
            narrative: 患者叙述
        
        Returns:
            处理结果字典
        """
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
        
        # 重置 Teacher 历史
        self.critical_agent.reset_history()
        
        # 初始状态
        state = self._create_initial_state(case_id, narrative)
        
        # 首次尝试
        print(f"\n[Trainer] Processing {case_id} (GT: {ground_truth_name})...")
        
        for attempt in range(self.max_retries + 1):
            attempt_num = attempt + 1
            print(f"[Trainer] Attempt {attempt_num}/{self.max_retries + 1}")
            
            try:
                # 运行 Pipeline
                state = self._run_pipeline(state)
                
                # 检查结果
                is_correct, error_type = self._check_result(
                    state, ground_truth_id
                )
                
                # 记录尝试 (防止 None 导致错误)
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
                    # 成功！
                    print(f"[Trainer] ✓ Correct diagnosis on attempt {attempt_num}")
                    
                    result["attempts"].append(attempt_record)
                    result["final_status"] = "success"
                    result["total_retries"] = attempt
                    
                    # 聚合和存储
                    self._on_success(state, case_id, case_type, ground_truth_id, ground_truth_name)
                    result["aggregated"] = True
                    
                    # 更新统计
                    if attempt == 0:
                        self.stats["success_first_try"] += 1
                    else:
                        self.stats["success_after_retry"] += 1
                    
                    return result
                
                else:
                    # 错误，需要重试
                    print(f"[Trainer] ✗ Incorrect. Error type: {error_type}")
                    
                    if attempt < self.max_retries:
                        # 获取 Critique
                        critique = self.critical_agent.generate_critique(
                            narrative=narrative,
                            ground_truth_id=ground_truth_id,
                            phase1_result=state.get("phase1_result", {}),
                            final_output=state.get("final_output", {}),
                            graph_json=state.get("graph_json", {}),
                            previous_critiques=self.critical_agent.get_history()
                        )
                        
                        critique_text = self.critical_agent.get_injectable_critique(critique)
                        self.critical_agent.add_to_history(critique_text)
                        
                        attempt_record["critique"] = critique_text
                        
                        # 注入 Hint 到状态
                        state = self._inject_hint(state, critique_text)
                        
                        print(f"[Trainer] Injected critique for retry")
                    
                    result["attempts"].append(attempt_record)
                    
            except Exception as e:
                print(f"[Trainer] Error in attempt {attempt_num}: {e}")
                import traceback
                traceback.print_exc()
                
                result["attempts"].append({
                    "attempt": attempt_num,
                    "error": str(e)
                })
                
                # 不立即失败，继续重试
                if attempt >= self.max_retries:
                    break
        
        # 所有尝试都失败
        print(f"[Trainer] ✗ Failed after {self.max_retries + 1} attempts")
        result["final_status"] = "failed"
        result["total_retries"] = self.max_retries
        self.stats["failed"] += 1
        
        return result
    
    def _create_initial_state(self, case_id: str, narrative: str) -> AgentState:
        """创建初始状态"""
        return AgentState(
            case_id=case_id,
            input_case={"narrative": narrative},
            phase1_result=None,
            graph_json=None,
            graph_summary=None,
            naive_scores=None,
            final_output=None,
            status="pending",
            error_log=None,
            global_hint=None,
            retry_count=0,
            memory_context=None
        )
    
    def _inject_hint(self, state: AgentState, hint: str) -> AgentState:
        """注入 Teacher Hint 到状态"""
        state["global_hint"] = hint
        state["retry_count"] = state.get("retry_count", 0) + 1
        
        # 重置中间结果，准备重新运行
        state["phase1_result"] = None
        state["graph_json"] = None
        state["graph_summary"] = None
        state["naive_scores"] = None
        state["final_output"] = None
        state["status"] = "pending"
        
        return state
    
    def _run_pipeline(self, state: AgentState) -> AgentState:
        """
        运行 Phase 1-3 Pipeline
        
        注意：这里拆解调用各 Agent，便于注入 Hint
        """
        narrative = state["input_case"].get("narrative", "")
        global_hint = state.get("global_hint")
        
        # === Phase 1 ===
        print("[Trainer] Running Phase 1...")
        
        # 注入 Hint（如果有）
        # 注意：Phase1Manager 需要修改以支持 hint，这里暂时通过状态传递
        phase1_result = self.phase1_manager.process(
            narrative, 
            global_hint=global_hint  # 传递 hint
        )
        state["phase1_result"] = phase1_result
        
        if phase1_result.get("error"):
            state["status"] = "failed"
            state["error_log"] = phase1_result.get("error")
            return state
        
        # === Phase 2 ===
        print("[Trainer] Running Phase 2...")
        state = self.phase2_agent.process(state)
        
        if state.get("status") == "failed":
            return state
        
        # === Graph Summary ===
        if state.get("graph_json"):
            try:
                graph = rebuild_graph_from_json(state["graph_json"])
                state["graph_summary"] = serialize_graph_to_summary(graph)
                state["naive_scores"] = calculate_deterministic_score(graph)
            except Exception as e:
                print(f"[Trainer] Graph summary error: {e}")
        
        # === Phase 3 ===
        print("[Trainer] Running Phase 3...")
        state = self.judge_agent.process(state)
        
        return state
    
    def _check_result(
        self, 
        state: AgentState, 
        ground_truth_id: str
    ) -> Tuple[bool, str]:
        """
        检查诊断结果是否正确
        
        Args:
            state: 当前状态
            ground_truth_id: 正确诊断 ID
        
        Returns:
            (is_correct, error_type)
        """
        # 确保格式一致
        gt_id = ground_truth_id.lstrip("d_")
        
        # 检查 Phase 1 Top-5 (防止 None 导致错误)
        phase1_result = state.get("phase1_result") or {}
        top_candidates = phase1_result.get("top_candidates", []) or []
        top_candidates_clean = [c.lstrip("d_") for c in top_candidates if c]
        gt_in_top5 = gt_id in top_candidates_clean
        
        # 检查最终诊断 (防止 None 导致错误)
        final_output = state.get("final_output") or {}
        final_id = final_output.get("final_diagnosis_id", "") or ""
        final_id_clean = final_id.lstrip("d_")
        is_correct = (final_id_clean == gt_id)
        
        # 确定错误类型
        if is_correct:
            return True, "none"
        elif not gt_in_top5:
            return False, "GT_NOT_IN_TOP5"
        else:
            return False, "FINAL_WRONG"
    
    def _on_success(
        self,
        state: AgentState,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str
    ) -> None:
        """
        成功诊断后的处理：聚合到 Golden Graph 和 Memory Bank
        """
        # 1. 聚合到 Golden Graph
        graph_json = state.get("graph_json", {})
        self.aggregation_manager.record_success(
            pathology_id=ground_truth_id,
            pathology_name=ground_truth_name,
            case_graph=graph_json,
            case_id=case_id
        )
        
        # 2. 存储到 Memory Bank (防止 None 导致错误)
        phase1_result = state.get("phase1_result") or {}
        final_output = state.get("final_output") or {}
        track_b_output = phase1_result.get("track_b_output") or {}
        p_nodes = track_b_output.get("p_nodes", []) or []
        
        initial_diagnosis_id = phase1_result.get("final_diagnosis_id", "") or ""
        initial_diagnosis_name = DIAGNOSIS_ID_MAP.get(
            initial_diagnosis_id.lstrip("d_"), "Unknown"
        )
        
        final_diagnosis_id = final_output.get("final_diagnosis_id", "") or ""
        final_diagnosis_name = final_output.get("final_diagnosis_name", "") or "Unknown"
        
        self.memory_bank.add_case(
            case_id=case_id,
            case_type=case_type,
            ground_truth_id=ground_truth_id.lstrip("d_"),
            ground_truth_name=ground_truth_name,
            initial_diagnosis_id=initial_diagnosis_id.lstrip("d_"),
            initial_diagnosis_name=initial_diagnosis_name,
            final_diagnosis_id=final_diagnosis_id.lstrip("d_"),
            final_diagnosis_name=final_diagnosis_name,
            p_nodes=p_nodes,
            retry_count=state.get("retry_count", 0)
        )
    
    def log_result(self, result: Dict[str, Any]) -> None:
        """
        记录处理结果到日志文件（线程安全）
        
        Args:
            result: process_case 的返回值
        """
        log_path = self.log_dir / "offline_training_log.jsonl"
        
        # 创建干净的日志记录（排除内部字段）
        log_record = {k: v for k, v in result.items() if not k.startswith("_")}
        
        # 线程安全写入日志
        with self._log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        
        # 线程安全更新统计
        with self._stats_lock:
            self.stats["total_processed"] += 1
        
        # 线程安全更新已处理集合
        with self._processed_lock:
            self.processed_cases.add(result["case_id"])
    
    def finalize(self) -> Dict[str, Any]:
        """
        训练结束时的收尾工作
        
        Returns:
            最终统计信息
        """
        print("\n[Trainer] Finalizing training...")
        
        # 保存所有 Golden Graphs
        aggregation_stats = self.aggregation_manager.finalize()
        
        # 保存 Memory Bank
        self.memory_bank.save()
        
        # 汇总统计
        final_stats = {
            **self.stats,
            "aggregation": aggregation_stats,
            "memory_bank": self.memory_bank.get_statistics()
        }
        
        # 保存统计到文件
        stats_path = self.log_dir / "training_summary.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        
        print(f"[Trainer] Training complete!")
        print(f"    Total processed: {self.stats['total_processed']}")
        print(f"    Success (first try): {self.stats['success_first_try']}")
        print(f"    Success (after retry): {self.stats['success_after_retry']}")
        print(f"    Failed: {self.stats['failed']}")
        
        return final_stats
    
    def is_processed(self, case_id: str) -> bool:
        """检查案例是否已处理（线程安全）"""
        with self._processed_lock:
            return case_id in self.processed_cases
    
    def process_case_for_batch(
        self,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str,
        narrative: str
    ) -> Dict[str, Any]:
        """
        处理单个案例，用于批量模式（延迟聚合）
        
        与 process_case 不同，此方法不立即聚合，而是返回结果供后续批量聚合。
        
        Args:
            case_id: 案例 ID
            case_type: 类型 (control/trap)
            ground_truth_id: 正确诊断 ID
            ground_truth_name: 正确诊断名称
            narrative: 患者叙述
        
        Returns:
            处理结果字典（包含用于聚合的额外数据）
        """
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
            "_aggregation_data": None  # 用于延迟聚合的数据
        }
        
        # 为每个线程创建独立的 Agent 状态（避免共享状态冲突）
        # 注意：Critical Agent 历史需要线程隔离
        local_critique_history = []
        
        # 初始状态
        state = self._create_initial_state(case_id, narrative)
        
        for attempt in range(self.max_retries + 1):
            attempt_num = attempt + 1
            
            try:
                # 运行 Pipeline
                state = self._run_pipeline(state)
                
                # 检查结果
                is_correct, error_type = self._check_result(state, ground_truth_id)
                
                # 记录尝试 (防止 None 导致错误)
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
                    
                    # 准备聚合数据（延迟聚合）
                    result["_aggregation_data"] = {
                        "state": state,
                        "case_id": case_id,
                        "case_type": case_type,
                        "ground_truth_id": ground_truth_id,
                        "ground_truth_name": ground_truth_name
                    }
                    
                    # 更新统计（线程安全）
                    with self._stats_lock:
                        if attempt == 0:
                            self.stats["success_first_try"] += 1
                        else:
                            self.stats["success_after_retry"] += 1
                    
                    return result
                
                else:
                    if attempt < self.max_retries:
                        # 获取 Critique（使用本地历史）
                        critique = self.critical_agent.generate_critique(
                            narrative=narrative,
                            ground_truth_id=ground_truth_id,
                            phase1_result=state.get("phase1_result", {}),
                            final_output=state.get("final_output", {}),
                            graph_json=state.get("graph_json", {}),
                            previous_critiques=local_critique_history
                        )
                        
                        critique_text = self.critical_agent.get_injectable_critique(critique)
                        local_critique_history.append(critique_text)
                        
                        attempt_record["critique"] = critique_text
                        state = self._inject_hint(state, critique_text)
                    
                    result["attempts"].append(attempt_record)
                    
            except Exception as e:
                result["attempts"].append({
                    "attempt": attempt_num,
                    "error": str(e)
                })
                if attempt >= self.max_retries:
                    break
        
        result["final_status"] = "failed"
        result["total_retries"] = self.max_retries
        
        with self._stats_lock:
            self.stats["failed"] += 1
        
        return result
    
    def batch_aggregate(self, results: List[Dict[str, Any]]) -> int:
        """
        批量聚合成功的案例
        
        在并发处理完一批案例后调用此方法，避免并发写入冲突。
        
        Args:
            results: process_case_for_batch 的返回值列表
        
        Returns:
            成功聚合的案例数量
        """
        aggregated_count = 0
        
        for result in results:
            if result["final_status"] == "success" and result.get("_aggregation_data"):
                agg_data = result["_aggregation_data"]
                
                try:
                    self._on_success(
                        state=agg_data["state"],
                        case_id=agg_data["case_id"],
                        case_type=agg_data["case_type"],
                        ground_truth_id=agg_data["ground_truth_id"],
                        ground_truth_name=agg_data["ground_truth_name"]
                    )
                    result["aggregated"] = True
                    aggregated_count += 1
                    
                except Exception as e:
                    print(f"[Trainer] Aggregation error for {result['case_id']}: {e}")
            
            # 无论成功与否，都删除 _aggregation_data 字段（避免写入日志）
            if "_aggregation_data" in result:
                del result["_aggregation_data"]
        
        return aggregated_count


class ConcurrentOfflineTrainer(OfflineTrainer):
    """
    并发版 Offline 训练器
    
    使用 ThreadPoolExecutor 并发处理案例，批量完成后统一聚合。
    
    适用于大规模训练场景，显著提升效率。
    
    Attributes:
        max_workers: 最大并发线程数
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        **kwargs
    ):
        """
        初始化并发训练器
        
        Args:
            max_workers: 最大并发线程数（建议 5-10）
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        self.max_workers = max_workers
        
        print(f"[ConcurrentTrainer] Initialized with max_workers={max_workers}")
    
    def train_batch(
        self,
        cases: List[Dict[str, Any]],
        load_narrative_fn,
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        并发处理一批案例
        
        Args:
            cases: 案例记录列表（从 offline_train_list.json）
            load_narrative_fn: 加载 narrative 的函数 (case_path, case_type) -> str
            progress_callback: 进度回调函数 (completed, total, result)
        
        Returns:
            所有案例的处理结果
        """
        all_results = []
        total = len(cases)
        completed = 0
        
        def process_single(case_record):
            """处理单个案例的包装函数"""
            case_id = case_record["case_id"]
            case_type = case_record["type"]
            ground_truth = case_record["ground_truth"]
            ground_truth_id = case_record["ground_truth_id"]
            case_path = case_record["path"]
            
            try:
                narrative = load_narrative_fn(case_path, case_type)
                
                if not narrative:
                    return {
                        "case_id": case_id,
                        "type": case_type,
                        "ground_truth_id": ground_truth_id,
                        "ground_truth_name": ground_truth,
                        "timestamp": datetime.now().isoformat(),
                        "attempts": [],
                        "final_status": "error",
                        "error": "Empty narrative",
                        "total_retries": 0,
                        "aggregated": False,
                        "_aggregation_data": None
                    }
                
                return self.process_case_for_batch(
                    case_id=case_id,
                    case_type=case_type,
                    ground_truth_id=ground_truth_id,
                    ground_truth_name=ground_truth,
                    narrative=narrative
                )
                
            except Exception as e:
                return {
                    "case_id": case_id,
                    "type": case_type,
                    "ground_truth_id": ground_truth_id,
                    "ground_truth_name": ground_truth,
                    "timestamp": datetime.now().isoformat(),
                    "attempts": [],
                    "final_status": "error",
                    "error": str(e),
                    "total_retries": 0,
                    "aggregated": False,
                    "_aggregation_data": None
                }
        
        # 使用 ThreadPoolExecutor 并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_case = {
                executor.submit(process_single, case): case 
                for case in cases
            }
            
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "case_id": case["case_id"],
                        "final_status": "error",
                        "error": str(e),
                        "_aggregation_data": None
                    }
                
                all_results.append(result)
                
                # 实时记录日志（log_result 会自动过滤 _ 开头的内部字段）
                self.log_result(result)
                
                # 进度回调
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, result)
        
        # 批量聚合
        aggregated = self.batch_aggregate(all_results)
        
        # 增量保存 Memory Bank（每批次后保存，避免数据丢失）
        self._incremental_save()
        
        print(f"[ConcurrentTrainer] Batch complete: {completed} processed, {aggregated} aggregated")
        
        return all_results
    
    def _incremental_save(self) -> None:
        """
        增量保存 Memory Bank 和 Training Summary
        
        在每个批次完成后调用，确保数据不会因中断而丢失。
        不影响主流程逻辑，仅增加保存频率。
        """
        try:
            # 保存 Memory Bank
            self.memory_bank.save()
            
            # 保存当前统计到 training_summary.json
            current_stats = {
                **self.stats,
                "status": "in_progress",
                "aggregation": self.aggregation_manager.get_statistics() if hasattr(self.aggregation_manager, 'get_statistics') else {},
                "memory_bank": self.memory_bank.get_statistics()
            }
            
            stats_path = self.log_dir / "training_summary.json"
            with self._log_lock:  # 线程安全写入
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(current_stats, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            print(f"[ConcurrentTrainer] Warning: Incremental save failed: {e}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("OfflineTrainer module loaded successfully.")
    print("Classes available: OfflineTrainer, ConcurrentOfflineTrainer")
    print("To run training, use scripts/run_offline_training.py")


