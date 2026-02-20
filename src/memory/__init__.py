"""
Memory Module
=============

提供 Memory Bank 管理和向量检索功能。

模块组成：
- embedding_client: Embedding API 封装
- memory_bank: Memory Bank 管理器（存储、检索、持久化）
"""

from .embedding_client import EmbeddingClient
from .memory_bank import MemoryBankManager

__all__ = ["EmbeddingClient", "MemoryBankManager"]




















