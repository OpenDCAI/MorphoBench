"""
Reasoning Breakdown Tool
========================
将模型的完整回答（包括推理过程）拆解为一个个步骤。
支持增量式拆解：每次调用返回"下一步"。
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .base_tool import BaseTool


@dataclass
class BreakdownState:
    """拆解状态，用于跟踪增量拆解进度"""
    full_response: str
    steps_extracted: List[str]
    current_position: int  # 当前处理到的位置（字符索引）
    is_complete: bool


class ReasoningBreakdownTool(BaseTool):
    """
    将模型回答拆解为推理步骤的工具。
    
    特点：
    - 客观拆分，不改变回答本意
    - 支持增量式拆解（每次返回下一步）
    - 保持原文语义完整性
    """
    
    name = "reasoning_breakdown"
    description = (
        "将模型的完整回答拆解为一个个推理步骤。"
        "客观拆分，不改变回答的本意。支持增量式拆解。"
    )
    inputs = {
        "full_response": {
            "type": "string",
            "description": "模型的完整回答文本（包含推理过程）",
        },
        "previous_steps": {
            "type": "list",
            "description": "已经拆解出的步骤列表",
        },
        "remaining_text": {
            "type": "string",
            "description": "剩余未拆解的文本",
        },
    }
    output_type = "dict"
    
    SYSTEM_PROMPT = """You are an expert reasoning analyzer. Your task is to extract the NEXT logical reasoning step from a model's response.

## Rules:
1. Extract ONLY ONE step at a time
2. Be objective - do not modify or interpret the original meaning
3. Each step should be a complete, self-contained reasoning unit
4. Preserve the original wording as much as possible
5. A step typically includes: a claim/conclusion and its supporting reasoning

## Output Format:
Return a JSON object with exactly these fields:
{
    "step": "The extracted reasoning step (exact text from original)",
    "step_summary": "A brief 1-sentence summary of what this step does",
    "remaining_text": "The remaining unprocessed text (empty string if done)",
    "is_complete": true/false (true if no more steps to extract)
}

## Important:
- If the remaining text has no more reasoning steps, set is_complete to true
- Do not fabricate or infer steps that are not explicitly in the text
- Ignore meta-text like "Reasoning:", "Answer:", "Confidence:" labels when extracting steps"""

    USER_PROMPT_TEMPLATE = """## Full Response:
{full_response}

## Already Extracted Steps ({num_steps} steps):
{previous_steps_text}

## Remaining Text to Process:
{remaining_text}

Extract the NEXT reasoning step from the remaining text. Return JSON only."""

    async def forward(
        self,
        full_response: str,
        previous_steps: Optional[List[str]] = None,
        remaining_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        增量式拆解：提取下一个推理步骤。
        
        Args:
            full_response: 完整的模型回答
            previous_steps: 已拆解的步骤列表
            remaining_text: 剩余未处理的文本
            
        Returns:
            {
                "step": str,           # 提取的步骤
                "step_summary": str,   # 步骤摘要
                "remaining_text": str, # 剩余文本
                "is_complete": bool,   # 是否完成
                "step_index": int,     # 步骤索引
            }
        """
        if previous_steps is None:
            previous_steps = []
        if remaining_text is None:
            remaining_text = full_response
        
        # 如果剩余文本为空或太短，标记完成
        if not remaining_text or len(remaining_text.strip()) < 10:
            return {
                "step": "",
                "step_summary": "",
                "remaining_text": "",
                "is_complete": True,
                "step_index": len(previous_steps),
            }
        
        # 构建 prompt
        previous_steps_text = "\n".join(
            [f"Step {i+1}: {s[:100]}..." if len(s) > 100 else f"Step {i+1}: {s}" 
             for i, s in enumerate(previous_steps)]
        ) if previous_steps else "(No steps extracted yet)"
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            full_response=full_response[:2000] + "..." if len(full_response) > 2000 else full_response,
            num_steps=len(previous_steps),
            previous_steps_text=previous_steps_text,
            remaining_text=remaining_text[:3000] if len(remaining_text) > 3000 else remaining_text,
        )
        
        # 调用 LLM
        result = await self._call_llm(self.SYSTEM_PROMPT, user_prompt, temperature=0.1)
        
        if not result:
            return {
                "step": "",
                "step_summary": "Failed to extract step",
                "remaining_text": remaining_text,
                "is_complete": True,
                "step_index": len(previous_steps),
                "error": "LLM call failed",
            }
        
        # 解析 JSON 结果
        try:
            import json
            # 尝试提取 JSON
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            parsed = json.loads(result)
            
            return {
                "step": parsed.get("step", ""),
                "step_summary": parsed.get("step_summary", ""),
                "remaining_text": parsed.get("remaining_text", ""),
                "is_complete": parsed.get("is_complete", False),
                "step_index": len(previous_steps) + 1,
            }
        except json.JSONDecodeError as e:
            # 如果解析失败，尝试简单处理
            return {
                "step": result[:500] if result else "",
                "step_summary": "Parse error - raw output",
                "remaining_text": remaining_text,
                "is_complete": True,
                "step_index": len(previous_steps),
                "error": f"JSON parse error: {e}",
            }
    
    async def extract_all_steps(self, full_response: str) -> List[Dict[str, Any]]:
        """
        一次性提取所有步骤（非增量式）。
        用于需要快速获取完整拆解的场景。
        """
        steps = []
        remaining = full_response
        max_iterations = 50  # 防止无限循环
        
        for _ in range(max_iterations):
            result = await self.forward(
                full_response=full_response,
                previous_steps=[s["step"] for s in steps],
                remaining_text=remaining,
            )
            
            if result.get("is_complete") or not result.get("step"):
                break
            
            steps.append(result)
            remaining = result.get("remaining_text", "")
            
            if not remaining or len(remaining.strip()) < 10:
                break
        
        return steps

