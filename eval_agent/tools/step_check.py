"""
Step by Step Check Tool
=======================
检查相邻两步之间的逻辑关系：
- 每一步是否合理
- 是否跳步
- 是否自相矛盾
- 是否缺关键论证
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .base_tool import BaseTool


@dataclass
class StepCheckResult:
    """步骤检查结果"""
    step_index: int
    previous_step: str
    current_step: str
    is_sound: bool
    has_logical_gap: bool
    has_contradiction: bool
    missing_justification: bool
    comment: str
    score: int  # 0-2: 0=严重问题, 1=小问题, 2=合理


class StepCheckTool(BaseTool):
    """
    检查相邻两步之间逻辑关系的工具。
    
    评估维度：
    1. 逻辑连贯性：当前步是否从前一步合理推出
    2. 跳步检测：是否遗漏了关键中间步骤
    3. 矛盾检测：是否与前文自相矛盾
    4. 论证完整性：是否缺少关键论证
    """
    
    name = "step_check"
    description = (
        "检查相邻两个推理步骤之间的逻辑关系，"
        "评估是否合理、是否跳步、是否自相矛盾、是否缺关键论证。"
    )
    inputs = {
        "previous_step": {
            "type": "string",
            "description": "前一个推理步骤",
        },
        "current_step": {
            "type": "string",
            "description": "当前推理步骤",
        },
        "step_index": {
            "type": "int",
            "description": "当前步骤的索引（从1开始）",
        },
        "context": {
            "type": "string",
            "description": "额外上下文（如原始问题）",
        },
    }
    output_type = "dict"
    
    SYSTEM_PROMPT = """You are a rigorous logic auditor. Your task is to evaluate the logical relationship between two consecutive reasoning steps.

## Evaluation Criteria:
1. **Logical Soundness**: Does the current step logically follow from the previous step?
2. **Gap Detection**: Are there missing intermediate steps that should be present?
3. **Contradiction Detection**: Does the current step contradict the previous step or earlier reasoning?
4. **Justification Completeness**: Is the reasoning sufficiently justified, or are key arguments missing?

## Output Format:
Return a JSON object with exactly these fields:
{
    "is_sound": true/false,
    "has_logical_gap": true/false,
    "has_contradiction": true/false,
    "missing_justification": true/false,
    "issues": ["list of specific issues found, empty if none"],
    "comment": "Detailed evaluation in 2-3 sentences",
    "score": 0-2
}

## Scoring Guide:
- 2: Sound - The transition is logically valid and well-justified
- 1: Minor Issue - Small gaps or unclear justification, but generally acceptable
- 0: Major Issue - Logical fallacy, contradiction, or critical missing justification

## Important:
- Be objective and precise
- Focus on logical validity, not on whether the conclusion is correct
- If this is the first step (no previous step), evaluate if it's a valid starting point"""

    USER_PROMPT_TEMPLATE = """## Context (Original Question):
{context}

## Step {step_index_prev}: Previous Step
{previous_step}

## Step {step_index}: Current Step
{current_step}

Evaluate the logical relationship between Step {step_index_prev} and Step {step_index}. Return JSON only."""

    FIRST_STEP_PROMPT_TEMPLATE = """## Context (Original Question):
{context}

## Step 1: First Step
{current_step}

This is the FIRST reasoning step. Evaluate if it's a valid starting point for addressing the question. Return JSON only."""

    async def forward(
        self,
        current_step: str,
        step_index: int,
        previous_step: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        检查当前步骤与前一步骤的逻辑关系。
        
        Args:
            current_step: 当前推理步骤
            step_index: 当前步骤索引（从1开始）
            previous_step: 前一个推理步骤（第一步时为None）
            context: 额外上下文（如原始问题）
            
        Returns:
            {
                "step_index": int,
                "is_sound": bool,
                "has_logical_gap": bool,
                "has_contradiction": bool,
                "missing_justification": bool,
                "issues": List[str],
                "comment": str,
                "score": int,
            }
        """
        # 构建 prompt
        if step_index == 1 or previous_step is None:
            user_prompt = self.FIRST_STEP_PROMPT_TEMPLATE.format(
                context=context[:1000] if len(context) > 1000 else context,
                current_step=current_step,
            )
        else:
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                context=context[:1000] if len(context) > 1000 else context,
                step_index_prev=step_index - 1,
                previous_step=previous_step,
                step_index=step_index,
                current_step=current_step,
            )
        
        # 调用 LLM
        result = await self._call_llm(self.SYSTEM_PROMPT, user_prompt, temperature=0.1)
        
        if not result:
            return {
                "step_index": step_index,
                "is_sound": False,
                "has_logical_gap": False,
                "has_contradiction": False,
                "missing_justification": False,
                "issues": ["LLM call failed"],
                "comment": "Unable to evaluate due to LLM error",
                "score": 1,
                "error": "LLM call failed",
            }
        
        # 解析 JSON 结果
        try:
            import json
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            parsed = json.loads(result)
            
            return {
                "step_index": step_index,
                "is_sound": parsed.get("is_sound", True),
                "has_logical_gap": parsed.get("has_logical_gap", False),
                "has_contradiction": parsed.get("has_contradiction", False),
                "missing_justification": parsed.get("missing_justification", False),
                "issues": parsed.get("issues", []),
                "comment": parsed.get("comment", ""),
                "score": parsed.get("score", 1),
            }
        except json.JSONDecodeError as e:
            return {
                "step_index": step_index,
                "is_sound": True,
                "has_logical_gap": False,
                "has_contradiction": False,
                "missing_justification": False,
                "issues": [f"Parse error: {e}"],
                "comment": result[:200] if result else "Parse error",
                "score": 1,
                "error": f"JSON parse error: {e}",
            }
    
    async def check_step_sequence(
        self,
        steps: List[str],
        context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        检查一系列步骤的逻辑关系。
        
        Args:
            steps: 步骤列表
            context: 上下文（如原始问题）
            
        Returns:
            每个步骤的检查结果列表
        """
        results = []
        
        for i, step in enumerate(steps):
            previous = steps[i - 1] if i > 0 else None
            result = await self.forward(
                current_step=step,
                step_index=i + 1,
                previous_step=previous,
                context=context,
            )
            results.append(result)
        
        return results


