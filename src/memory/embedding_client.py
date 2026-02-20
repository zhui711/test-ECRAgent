"""
Embedding Client
================

封装 Embedding API 调用，支持 OpenAI 兼容接口（如 Yunwu API）。

功能：
- 单文本 Embedding
- 批量文本 Embedding
- 错误处理和重试
"""

import os
import time
import numpy as np
from typing import List, Optional, Union
from openai import OpenAI


class EmbeddingClient:
    """
    Embedding API 客户端
    
    支持 OpenAI 兼容的 Embedding API（如 Yunwu API）。
    使用 text-embedding-3-small 模型，输出 1536 维向量。
    
    Attributes:
        client: OpenAI 客户端实例
        model: 模型名称
        dimension: 向量维度
    """
    
    def __init__(
        self,
        base_url: str = "https://yunwu.ai/v1",
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化 Embedding 客户端
        
        Args:
            base_url: API 基础 URL
            api_key: API 密钥（如果为 None，从环境变量读取）
            model: Embedding 模型名称
            dimension: 向量维度
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        # 从环境变量获取 API Key（如果未提供）
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("YUNWU_API_KEY")
        
        if not api_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY or YUNWU_API_KEY environment variable."
            )
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.dimension = dimension
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        获取单个文本的 Embedding 向量
        
        Args:
            text: 输入文本
        
        Returns:
            numpy 数组，形状为 (dimension,)
        
        Raises:
            Exception: API 调用失败
        """
        if not text or not text.strip():
            # 空文本返回零向量
            return np.zeros(self.dimension, dtype=np.float32)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text.strip()
                )
                
                embedding = response.data[0].embedding
                return np.array(embedding, dtype=np.float32)
                
            except Exception as e:
                print(f"[EmbeddingClient] Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        
        # 不应该到达这里，但以防万一
        return np.zeros(self.dimension, dtype=np.float32)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        批量获取文本的 Embedding 向量
        
        Args:
            texts: 输入文本列表
        
        Returns:
            numpy 数组，形状为 (len(texts), dimension)
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        embeddings = []
        
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        
        return np.stack(embeddings)
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 10
    ) -> np.ndarray:
        """
        分批获取 Embedding（用于大规模处理）
        
        Args:
            texts: 输入文本列表
            batch_size: 每批大小
        
        Returns:
            numpy 数组，形状为 (len(texts), dimension)
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.append(batch_embeddings)
            
            # 添加延迟以避免速率限制
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        return np.vstack(all_embeddings)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 简单测试
    try:
        client = EmbeddingClient()
        
        test_text = "Patient presents with fever and productive cough"
        embedding = client.embed_text(test_text)
        
        print(f"Text: {test_text}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"Test failed: {e}")




















