"""
推理质量评测
===========
基于 ReAct 框架的迭代式评测：
1. Reasoning Breakdown Tool: 增量式拆解推理步骤
2. Step by Step Check Tool: 检查相邻步骤的逻辑关系
3. 反复迭代直至完成
4. 最终产出: Completeness（完整性）和 Logical Coherence（逻辑连贯性）
"""
from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from ..config import EvalConfig
from ..tools.reasoning_breakdown import ReasoningBreakdownTool
from ..tools.step_check import StepCheckTool


@dataclass
class ReasoningTrace:
    """
    推理评测过程的完整记录。
    每道题一个 trace，最终保存为 txt 文件。
    """
    question_id: str
    question: str
    response: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    final_scores: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_txt(self) -> str:
        """生成人类可读的 txt 格式"""
        lines = [
            "=" * 80,
            f"REASONING QUALITY EVALUATION TRACE",
            f"Question ID: {self.question_id}",
            f"Timestamp: {self.timestamp}",
            "=" * 80,
            "",
            "## ORIGINAL QUESTION",
            "-" * 40,
            self.question[:500] + "..." if len(self.question) > 500 else self.question,
            "",
            "## MODEL RESPONSE",
            "-" * 40,
            self.response[:1000] + "..." if len(self.response) > 1000 else self.response,
            "",
            "=" * 80,
            "## STEP-BY-STEP BREAKDOWN & EVALUATION",
            "=" * 80,
            "",
        ]
        
        for i, (step, check) in enumerate(zip(self.steps, self.checks)):
            lines.extend([
                f"### Step {i + 1}",
                "-" * 40,
                f"**Content**: {step.get('step', 'N/A')}",
                f"**Summary**: {step.get('step_summary', 'N/A')}",
                "",
                f"**Logic Check**:",
                f"  - Sound: {check.get('is_sound', 'N/A')}",
                f"  - Has Gap: {check.get('has_logical_gap', 'N/A')}",
                f"  - Has Contradiction: {check.get('has_contradiction', 'N/A')}",
                f"  - Missing Justification: {check.get('missing_justification', 'N/A')}",
                f"  - Score: {check.get('score', 'N/A')}/2",
                f"  - Comment: {check.get('comment', 'N/A')}",
                "",
            ])
        
        if self.final_scores:
            lines.extend([
                "=" * 80,
                "## FINAL EVALUATION SCORES",
                "=" * 80,
                "",
                f"**Completeness**: {self.final_scores.get('completeness', 'N/A')}/100",
                f"  - Comment: {self.final_scores.get('completeness_comment', 'N/A')}",
                "",
                f"**Logical Coherence**: {self.final_scores.get('logical_coherence', 'N/A')}/100",
                f"  - Comment: {self.final_scores.get('coherence_comment', 'N/A')}",
                "",
            ])
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "response": self.response,
            "steps": self.steps,
            "checks": self.checks,
            "final_scores": self.final_scores,
            "timestamp": self.timestamp,
        }


class ReasoningQualityEvaluator:
    """
    推理质量评测器。
    
    采用 ReAct 框架迭代式评测：
    1. 使用 Breakdown Tool 拆解出一步
    2. 使用 Step Check Tool 检查该步与前一步的逻辑关系
    3. 重复直至完成
    4. 调用 API 产出最终评分
    """
    
    FINAL_EVAL_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of a model's reasoning process.

Based on the step-by-step breakdown and logic checks provided, evaluate the reasoning on two dimensions:

1. **Completeness** (0-100): Are all necessary reasoning steps present? Is there a clear path from problem to conclusion?
   - 100: All steps present, nothing missing
   - 70-99: Minor omissions but reasoning is substantially complete
   - 40-69: Some important steps missing
   - 0-39: Major gaps, reasoning is incomplete

2. **Logical Coherence** (0-100): Is the reasoning chain logically sound and self-consistent?
   - 100: Flawless logic, no gaps or contradictions
   - 70-99: Minor issues but overall sound
   - 40-69: Some logical problems
   - 0-39: Serious logical flaws

Output JSON:
{
    "completeness": <0-100>,
    "completeness_comment": "<brief justification>",
    "logical_coherence": <0-100>,
    "coherence_comment": "<brief justification>"
}"""

    FINAL_EVAL_USER_TEMPLATE = """## Original Question:
{question}

## Total Steps Extracted: {num_steps}

## Step-by-Step Breakdown and Logic Checks:
{steps_and_checks}

## Summary Statistics:
- Total steps: {num_steps}
- Sound steps: {num_sound}
- Steps with gaps: {num_gaps}
- Steps with contradictions: {num_contradictions}
- Average step score: {avg_score:.2f}/2

Provide your final evaluation. Return JSON only."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        
        # 初始化工具
        self.breakdown_tool = ReasoningBreakdownTool(
            model_id=self.config.breakdown_model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_tokens=self.config.max_completion_tokens,
        )
        self.check_tool = StepCheckTool(
            model_id=self.config.check_model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_tokens=self.config.max_completion_tokens,
        )
        
        # 用于最终评分的客户端
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
    
    async def evaluate_single(
        self,
        question_id: str,
        question: str,
        response: str,
        save_trace: bool = True,
        trace_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], ReasoningTrace]:
        """
        对单个样本进行推理质量评测。
        
        实现 ReAct 迭代循环：
        - Breakdown → Check → Breakdown → Check → ...
        
        Args:
            question_id: 问题 ID
            question: 原始问题
            response: 模型回答
            save_trace: 是否保存评测过程 txt
            trace_dir: trace 文件保存目录
            
        Returns:
            (final_scores, trace)
        """
        trace = ReasoningTrace(
            question_id=question_id,
            question=question,
            response=response,
        )
        
        # ReAct 迭代循环
        steps_extracted: List[str] = []
        remaining_text = response
        max_iterations = 50
        
        for iteration in range(max_iterations):
            # Step 1: Breakdown - 拆解出下一步
            breakdown_result = await self.breakdown_tool.forward(
                full_response=response,
                previous_steps=steps_extracted,
                remaining_text=remaining_text,
            )
            
            # 检查是否完成
            if breakdown_result.get("is_complete") or not breakdown_result.get("step"):
                break
            
            current_step = breakdown_result.get("step", "")
            steps_extracted.append(current_step)
            remaining_text = breakdown_result.get("remaining_text", "")
            
            # 记录 step
            trace.steps.append(breakdown_result)
            
            # Step 2: Check - 检查当前步与前一步的逻辑关系
            previous_step = steps_extracted[-2] if len(steps_extracted) > 1 else None
            check_result = await self.check_tool.forward(
                current_step=current_step,
                step_index=len(steps_extracted),
                previous_step=previous_step,
                context=question,
            )
            
            # 记录 check
            trace.checks.append(check_result)
            
            # 如果剩余文本太短，结束
            if not remaining_text or len(remaining_text.strip()) < 10:
                break
        
        # Step 3: Final Evaluation - 产出最终评分
        final_scores = await self._compute_final_scores(question, trace.steps, trace.checks)
        trace.final_scores = final_scores
        
        # 保存 trace
        if save_trace:
            trace_dir = trace_dir or self.config.trace_dir
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, f"trace_{question_id}.txt")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write(trace.to_txt())
        
        return final_scores, trace
    
    async def _compute_final_scores(
        self,
        question: str,
        steps: List[Dict[str, Any]],
        checks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        根据步骤拆解和检查结果，计算最终的 Completeness 和 Logical Coherence 评分。
        """
        if not steps or not checks:
            return {
                "completeness": 0,
                "completeness_comment": "No steps extracted",
                "logical_coherence": 0,
                "coherence_comment": "No steps to evaluate",
            }
        
        # 统计信息
        num_steps = len(steps)
        num_sound = sum(1 for c in checks if c.get("is_sound", False))
        num_gaps = sum(1 for c in checks if c.get("has_logical_gap", False))
        num_contradictions = sum(1 for c in checks if c.get("has_contradiction", False))
        scores = [c.get("score", 1) for c in checks]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 构建步骤和检查的摘要
        steps_and_checks = []
        for i, (step, check) in enumerate(zip(steps, checks)):
            steps_and_checks.append(
                f"Step {i+1}: {step.get('step_summary', 'N/A')}\n"
                f"  Check: sound={check.get('is_sound')}, gap={check.get('has_logical_gap')}, "
                f"contradiction={check.get('has_contradiction')}, score={check.get('score')}/2\n"
                f"  Comment: {check.get('comment', 'N/A')}"
            )
        
        user_prompt = self.FINAL_EVAL_USER_TEMPLATE.format(
            question=question[:500] if len(question) > 500 else question,
            num_steps=num_steps,
            steps_and_checks="\n\n".join(steps_and_checks),
            num_sound=num_sound,
            num_gaps=num_gaps,
            num_contradictions=num_contradictions,
            avg_score=avg_score,
        )
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.config.summary_model,
                max_completion_tokens=self.config.max_completion_tokens,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": self.FINAL_EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            result = completion.choices[0].message.content.strip()
            
            # 解析 JSON
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            
            parsed = json.loads(result)
            return {
                "completeness": parsed.get("completeness", 0),
                "completeness_comment": parsed.get("completeness_comment", ""),
                "logical_coherence": parsed.get("logical_coherence", 0),
                "coherence_comment": parsed.get("coherence_comment", ""),
                "num_steps": num_steps,
                "num_sound": num_sound,
                "num_gaps": num_gaps,
                "num_contradictions": num_contradictions,
                "avg_step_score": round(avg_score, 2),
            }
        except Exception as e:
            error_msg = str(e)
            print(f"[Final eval error] {error_msg}")
            if "Connection" in error_msg or "connection" in error_msg:
                print(f"  -> 连接错误，请检查 base_url: {self.client.base_url}")
            # 基于统计信息计算 fallback 分数
            completeness = min(100, num_steps * 10) if num_steps > 0 else 0
            coherence = int(100 * (num_sound / num_steps)) if num_steps > 0 else 0
            return {
                "completeness": completeness,
                "completeness_comment": f"Fallback: {num_steps} steps extracted",
                "logical_coherence": coherence,
                "coherence_comment": f"Fallback: {num_sound}/{num_steps} sound steps",
                "num_steps": num_steps,
                "num_sound": num_sound,
                "num_gaps": num_gaps,
                "num_contradictions": num_contradictions,
                "avg_step_score": round(avg_score, 2),
                "error": str(e),
            }
    
    async def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        predictions: Dict[str, Dict[str, Any]],
        save_traces: bool = True,
        trace_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量评测推理质量。
        
        Args:
            questions: 问题列表
            predictions: 预测结果
            save_traces: 是否保存每道题的评测过程
            trace_dir: trace 保存目录
            
        Returns:
            更新后的 predictions，每个样本增加 reasoning_quality 字段
        """
        semaphore = asyncio.Semaphore(self.config.num_workers)
        trace_dir = trace_dir or self.config.trace_dir
        
        async def process_one(q: Dict[str, Any]):
            async with semaphore:
                uid = str(q["index"])
                if uid not in predictions:
                    return None, None
                
                pred = predictions[uid].copy()
                if "reasoning_quality" in pred:
                    return uid, pred
                
                scores, trace = await self.evaluate_single(
                    question_id=uid,
                    question=q.get("question", ""),
                    response=pred.get("response", ""),
                    save_trace=save_traces,
                    trace_dir=trace_dir,
                )
                
                pred["reasoning_quality"] = scores
                return uid, pred
        
        tasks = [process_one(q) for q in questions]
        results = await asyncio.gather(*tasks)
        
        updated = {}
        for uid, pred in results:
            if uid is not None and pred is not None:
                updated[uid] = pred
        
        return updated
    
    @staticmethod
    def compute_metrics(predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算推理质量相关指标。
        """
        completeness_scores = []
        coherence_scores = []
        step_counts = []
        
        for v in predictions.values():
            rq = v.get("reasoning_quality", {})
            if "completeness" in rq:
                completeness_scores.append(rq["completeness"])
            if "logical_coherence" in rq:
                coherence_scores.append(rq["logical_coherence"])
            if "num_steps" in rq:
                step_counts.append(rq["num_steps"])
        
        if not completeness_scores and not coherence_scores:
            return {}
        
        return {
            "completeness_avg": round(sum(completeness_scores) / len(completeness_scores), 2) if completeness_scores else None,
            "logical_coherence_avg": round(sum(coherence_scores) / len(coherence_scores), 2) if coherence_scores else None,
            "avg_steps": round(sum(step_counts) / len(step_counts), 2) if step_counts else None,
            "n_scored": len(completeness_scores),
        }


