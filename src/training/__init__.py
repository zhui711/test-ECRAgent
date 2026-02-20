"""
Training Module
===============

提供 Offline 训练功能，包括：
- AggregationManager: 增量聚合 Golden Graph
- OfflineTrainer: 训练主循环
"""

from .aggregation_manager import AggregationManager
from .offline_trainer import OfflineTrainer

__all__ = ["AggregationManager", "OfflineTrainer"]




















