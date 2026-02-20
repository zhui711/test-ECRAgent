"""
LLM API Client
封装 OpenAI 兼容的 API 调用，支持 logprobs 和 Base URL 切换
"""
import os
import time
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class LLMClient:
    """
    LLM API 客户端，支持切换 Base URL 和 logprobs
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 120):
        """
        初始化 LLM 客户端
        
        Args:
            base_url: API Base URL (例如 "https://yunwu.ai/v1")
            api_key: API Key，如果为 None 则从环境变量 YUNWU_API_KEY 读取
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url
        self.timeout = timeout
        
        # 如果没有提供 api_key，尝试从环境变量读取
        if api_key is None:
            api_key = os.getenv("YUNWU_API_KEY")
            if api_key is None:
                raise ValueError("API key not provided and YUNWU_API_KEY environment variable not set")
        
        self.api_key = api_key
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )
    
    def generate_json(
        self,
        messages: list,
        model: str,
        logprobs: bool = True,
        top_logprobs: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Dict[str, Any]:
        """
        生成 JSON 响应，支持 logprobs
        
        Args:
            messages: 消息列表，格式为 [{"role": "system", "content": "..."}, ...]
            model: 模型名称
            logprobs: 是否返回 logprobs
            top_logprobs: 返回 top-k logprobs
            temperature: 温度参数
            max_tokens: 最大 token 数
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        
        Returns:
            包含原始 response 对象的字典，格式为:
            {
                "response": completion对象,
                "content": str,  # 响应文本内容
                "error": Optional[str]  # 错误信息，如果成功则为 None
            }
        """
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs if logprobs else None,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                content = completion.choices[0].message.content
                
                return {
                    "response": completion,
                    "content": content,
                    "error": None
                }
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                if attempt == max_retries - 1:
                    return {
                        "response": None,
                        "content": None,
                        "error": error_msg
                    }
                time.sleep(retry_delay)
        
        return {
            "response": None,
            "content": None,
            "error": "Max retries exceeded"
        }
    
    def switch_base_url(self, new_base_url: str, new_api_key: Optional[str] = None):
        """
        切换 Base URL（用于动态切换 API 提供商）
        
        Args:
            new_base_url: 新的 Base URL
            new_api_key: 新的 API Key（可选）
        """
        self.base_url = new_base_url
        if new_api_key:
            self.api_key = new_api_key
        else:
            # 尝试从环境变量读取
            self.api_key = os.getenv("YUNWU_API_KEY", self.api_key)
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

