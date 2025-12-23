"""
MorphoBench Evaluation Agent
============================
基于 ReAct 框架的多维度评测工具，包括：
- 正确性评测 (Correctness)
- 推理质量评测 (Reasoning Quality)
- Hint 跟随能力评测 (Hint Follow) [待实现]
"""

from .config import EvalConfig

__all__ = ["EvalConfig"]
