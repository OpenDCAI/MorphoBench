"""
Hint Check Tool
===============
检查推理步骤与 hint 的对齐关系：
1. Hint Alignment（对齐）: 推理轨迹是否与 hint 指导一致
2. Deviation Analysis（偏离分析）: 如果偏离 hint，是否有合理理由

注意：此工具只做客观评判，不评价 hint 本身的正确性或误导性。
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass

from .base_tool import BaseTool


@dataclass
class HintCheckResult:
    """单步 hint 检查结果"""
    step_index: int
    step_content: str
    alignment: Literal["aligned", "deviated", "neutral"]  # 对齐/偏离/中立
    deviation_justified: Optional[bool]  # 如果偏离，是否有合理理由
    comment: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_content": self.step_content[:200] + "..." if len(self.step_content) > 200 else self.step_content,
            "alignment": self.alignment,
            "deviation_justified": self.deviation_justified,
            "comment": self.comment,
        }


class HintCheckTool(BaseTool):
    """
    检查推理步骤与 hint 对齐关系的工具。
    
    评估维度：
    1. Hint Alignment（对齐）: 推理轨迹是否与 hint 指导一致
    2. Deviation Analysis（偏离分析）: 如果偏离 hint，是否有合理理由
    
    重要：只做客观评判，不评价 hint 本身是否正确或有帮助。
    """
    
    name = "hint_check"
    description = (
        "检查推理步骤与 hint 的对齐关系，"
        "评估推理是否遵循 hint 指导，偏离时是否有合理理由。"
    )
    inputs = {
        "step_content": {
            "type": "string",
            "description": "当前推理步骤内容",
        },
        "step_index": {
            "type": "int",
            "description": "当前步骤索引（从1开始）",
        },
        "hints": {
            "type": "list",
            "description": "Hint 列表",
        },
        "previous_steps": {
            "type": "list",
            "description": "之前的推理步骤列表",
        },
        "question": {
            "type": "string",
            "description": "原始问题",
        },
    }
    output_type = "dict"
    
    SYSTEM_PROMPT = """You are an objective reasoning analyst. Your task is to evaluate whether a reasoning step aligns with or deviates from given hints.

## Important Guidelines:
1. You are ONLY evaluating alignment, NOT judging whether the hint is correct or helpful
2. A step "aligns" with a hint if it follows the hint's guidance or direction
3. A step "deviates" from a hint if it contradicts or ignores the hint's guidance
4. A step is "neutral" if the hint is not relevant to this particular step
5. If the step deviates, evaluate whether the deviation has reasonable justification (e.g., the model provides evidence or reasoning for taking a different approach)

## Output Format:
Return a JSON object with exactly these fields:
{
    "alignment": "aligned|deviated|neutral",
    "relevant_hints": [<indices of hints relevant to this step, 1-based>],
    "deviation_justified": true|false|null,
    "justification_quality": "strong|weak|none|null",
    "comment": "Brief explanation of the alignment assessment"
}

## Evaluation Criteria:
- **aligned**: The step follows the hint's direction or guidance
- **deviated**: The step contradicts or explicitly ignores the hint
- **neutral**: The hint doesn't apply to this step's content

For deviation_justified:
- **true**: The reasoning provides valid evidence or logic for the different approach
- **false**: No justification given for ignoring/contradicting the hint
- **null**: Not applicable (step is aligned or neutral)

Be objective. Do NOT make assumptions about whether hints are correct or misleading."""

    USER_PROMPT_TEMPLATE = """## Original Question:
{question}

## Hints Provided:
{hints_text}

## Previous Reasoning Steps:
{previous_steps_text}

## Current Step (Step {step_index}):
{step_content}

Analyze whether this step aligns with, deviates from, or is neutral to the hints. Return JSON only."""

    async def forward(
        self,
        step_content: str,
        step_index: int,
        hints: List[str],
        previous_steps: Optional[List[str]] = None,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        检查单个推理步骤与 hints 的对齐关系。
        
        Args:
            step_content: 当前推理步骤内容
            step_index: 当前步骤索引（从1开始）
            hints: Hint 列表
            previous_steps: 之前的推理步骤
            question: 原始问题
            
        Returns:
            {
                "step_index": int,
                "alignment": "aligned"|"deviated"|"neutral",
                "relevant_hints": List[int],
                "deviation_justified": bool|None,
                "justification_quality": str|None,
                "comment": str,
            }
        """
        if previous_steps is None:
            previous_steps = []
        
        # 如果没有 hints，返回 neutral
        if not hints:
            return {
                "step_index": step_index,
                "alignment": "neutral",
                "relevant_hints": [],
                "deviation_justified": None,
                "justification_quality": None,
                "comment": "No hints provided to evaluate against",
            }
        
        # 构建 hints 文本
        hints_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(hints)])
        
        # 构建之前步骤文本
        if previous_steps:
            previous_steps_text = "\n".join([
                f"Step {i+1}: {s[:150]}..." if len(s) > 150 else f"Step {i+1}: {s}"
                for i, s in enumerate(previous_steps)
            ])
        else:
            previous_steps_text = "(This is the first step)"
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            question=question[:500] if len(question) > 500 else question,
            hints_text=hints_text,
            previous_steps_text=previous_steps_text,
            step_index=step_index,
            step_content=step_content,
        )
        
        # 调用 LLM
        result = await self._call_llm(self.SYSTEM_PROMPT, user_prompt, temperature=0.1)
        
        if not result:
            return {
                "step_index": step_index,
                "alignment": "neutral",
                "relevant_hints": [],
                "deviation_justified": None,
                "justification_quality": None,
                "comment": "LLM call failed",
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
                "alignment": parsed.get("alignment", "neutral"),
                "relevant_hints": parsed.get("relevant_hints", []),
                "deviation_justified": parsed.get("deviation_justified"),
                "justification_quality": parsed.get("justification_quality"),
                "comment": parsed.get("comment", ""),
            }
        except json.JSONDecodeError as e:
            return {
                "step_index": step_index,
                "alignment": "neutral",
                "relevant_hints": [],
                "deviation_justified": None,
                "justification_quality": None,
                "comment": result[:200] if result else "Parse error",
                "error": f"JSON parse error: {e}",
            }

    async def check_all_steps(
        self,
        steps: List[str],
        hints: List[str],
        question: str = "",
    ) -> List[Dict[str, Any]]:
        """
        检查所有步骤与 hints 的对齐关系。
        
        Args:
            steps: 推理步骤列表
            hints: Hint 列表
            question: 原始问题
            
        Returns:
            每个步骤的检查结果列表
        """
        results = []
        
        for i, step in enumerate(steps):
            previous = steps[:i] if i > 0 else None
            result = await self.forward(
                step_content=step,
                step_index=i + 1,
                hints=hints,
                previous_steps=previous,
                question=question,
            )
            results.append(result)
        
        return results









