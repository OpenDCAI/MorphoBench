"""
评测配置与默认参数。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalConfig:
    """评测配置"""
    
    # API 配置
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", "YOUR_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("API_BASE", "API_BASE_URL"))
    
    # 模型配置
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "o3-mini-2025-01-31"))
    breakdown_model: str = field(default_factory=lambda: os.getenv("BREAKDOWN_MODEL", "o3-mini-2025-01-31"))
    check_model: str = field(default_factory=lambda: os.getenv("CHECK_MODEL", "o3-mini-2025-01-31"))
    summary_model: str = field(default_factory=lambda: os.getenv("SUMMARY_MODEL", "o3-mini-2025-01-31"))
    hint_model: str = field(default_factory=lambda: os.getenv("HINT_MODEL", "o3-mini-2025-01-31"))
    
    # 并发与超参
    num_workers: int = field(default_factory=lambda: int(os.getenv("EVAL_NUM_WORKERS", "50")))
    max_completion_tokens: int = field(default_factory=lambda: int(os.getenv("EVAL_MAX_TOKENS", "4096")))
    timeout: float = 600.0
    max_retries: int = 3
    
    # 输出目录
    output_dir: str = field(default_factory=lambda: os.getenv("EVAL_OUTPUT_DIR", "./output/eval_agent_result"))
    trace_dir: str = field(default_factory=lambda: os.getenv("EVAL_TRACE_DIR", "./output/eval_agent_traces"))
