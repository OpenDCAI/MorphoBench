"""
工具基类，模仿 smolagents 的 Tool 设计。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


class BaseTool(ABC):
    """
    评测工具基类。
    
    子类需要实现：
    - name: 工具名称
    - description: 工具描述
    - inputs: 输入参数定义
    - forward(): 工具执行逻辑
    """
    
    name: str = "base_tool"
    description: str = "Base tool description"
    inputs: Dict[str, Dict[str, Any]] = {}
    output_type: str = "string"
    
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        timeout: float = 600.0,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("API_KEY", "YOUR_API_KEY")
        self.base_url = base_url or os.getenv("API_BASE", "API_BASE_URL")
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # 初始化 OpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=3,
        )
    
    @abstractmethod
    async def forward(self, **kwargs) -> Any:
        """
        工具执行逻辑，子类必须实现。
        """
        raise NotImplementedError
    
    async def __call__(self, **kwargs) -> Any:
        """
        调用工具。
        """
        return await self.forward(**kwargs)
    
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """
        调用 LLM 的通用方法。
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                max_completion_tokens=self.max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            print(f"[{self.name}] LLM call error: {error_msg}")
            # 打印更详细的调试信息
            if "Connection" in error_msg or "connection" in error_msg:
                print(f"  -> 连接错误，请检查 base_url: {self.base_url}")
            elif "401" in error_msg or "Unauthorized" in error_msg:
                print(f"  -> 认证错误，请检查 api_key")
            elif "404" in error_msg:
                print(f"  -> 模型不存在，请检查 model: {self.model_id}")
            return None
