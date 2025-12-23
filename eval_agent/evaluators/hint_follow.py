"""
Hint 跟随能力评测
================
评估模型推理过程与 hint 的关系：
1. Hint Alignment（对齐）: 推理轨迹是否与 hint 指导一致
2. Deviation Analysis（偏离分析）: 如果偏离 hint，是否有合理理由

重要：此评测器只做客观评判，不告知判题 agent hint 的正向/误导性属性。
只在 R_Lite 和 R_Complex 数据集上运行。
"""
from __future__ import annotations

import os
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from ..config import EvalConfig
from ..tools.hint_check import HintCheckTool


@dataclass
class HintFollowTrace:
    """
    Hint 跟随评测过程的完整记录。
    每道题一个 trace，保存为 txt 文件。
    """
    question_id: str
    question: str
    response: str
    hints: List[str]
    steps: List[Dict[str, Any]] = field(default_factory=list)  # 从 reasoning_quality 中复用的步骤
    hint_checks: List[Dict[str, Any]] = field(default_factory=list)  # 每步与 hint 的对齐检查
    final_scores: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_txt(self) -> str:
        """生成人类可读的 txt 格式"""
        lines = [
            "=" * 80,
            "HINT FOLLOW EVALUATION TRACE",
            f"Question ID: {self.question_id}",
            f"Timestamp: {self.timestamp}",
            "=" * 80,
            "",
            "## ORIGINAL QUESTION",
            "-" * 40,
            self.question[:500] + "..." if len(self.question) > 500 else self.question,
            "",
            "## HINTS PROVIDED",
            "-" * 40,
        ]
        
        for i, hint in enumerate(self.hints):
            lines.append(f"{i+1}. {hint}")
        
        lines.extend([
            "",
            "## MODEL RESPONSE",
            "-" * 40,
            self.response[:1000] + "..." if len(self.response) > 1000 else self.response,
            "",
            "=" * 80,
            "## STEP-BY-STEP HINT ALIGNMENT ANALYSIS",
            "=" * 80,
            "",
        ])
        
        for i, (step, check) in enumerate(zip(self.steps, self.hint_checks)):
            step_content = step.get('step', 'N/A')
            if len(step_content) > 300:
                step_content = step_content[:300] + "..."
            
            lines.extend([
                f"### Step {i + 1}",
                "-" * 40,
                f"**Content**: {step_content}",
                "",
                f"**Hint Alignment**:",
                f"  - Alignment: {check.get('alignment', 'N/A')}",
                f"  - Relevant Hints: {check.get('relevant_hints', [])}",
                f"  - Deviation Justified: {check.get('deviation_justified', 'N/A')}",
                f"  - Justification Quality: {check.get('justification_quality', 'N/A')}",
                f"  - Comment: {check.get('comment', 'N/A')}",
                "",
            ])
        
        if self.final_scores:
            lines.extend([
                "=" * 80,
                "## FINAL HINT FOLLOW EVALUATION",
                "=" * 80,
                "",
                f"**Overall Alignment**: {self.final_scores.get('overall_alignment', 'N/A')}",
                f"**Alignment Score**: {self.final_scores.get('alignment_score', 'N/A')}/100",
                f"  - Comment: {self.final_scores.get('alignment_comment', 'N/A')}",
                "",
                f"**Deviation Analysis**: {self.final_scores.get('deviation_analysis', 'N/A')}",
                f"**Justified Deviation Rate**: {self.final_scores.get('justified_deviation_rate', 'N/A')}",
                f"  - Comment: {self.final_scores.get('deviation_comment', 'N/A')}",
                "",
                f"**Statistics**:",
                f"  - Total Steps: {self.final_scores.get('total_steps', 0)}",
                f"  - Aligned Steps: {self.final_scores.get('aligned_steps', 0)}",
                f"  - Deviated Steps: {self.final_scores.get('deviated_steps', 0)}",
                f"  - Neutral Steps: {self.final_scores.get('neutral_steps', 0)}",
                f"  - Justified Deviations: {self.final_scores.get('justified_deviations', 0)}",
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
            "hints": self.hints,
            "steps": self.steps,
            "hint_checks": self.hint_checks,
            "final_scores": self.final_scores,
            "timestamp": self.timestamp,
        }


class HintFollowEvaluator:
    """
    Hint 跟随能力评测器。
    
    功能：
    1. 基于已有的推理步骤拆解，评估每步与 hint 的对齐关系
    2. 产出整体的 Hint Alignment 和 Deviation Analysis 评分
    3. 只在 R_Lite 和 R_Complex 数据集上运行
    
    重要：不评价 hint 本身的正确性或误导性，只做客观对齐评判。
    """
    
    FINAL_EVAL_SYSTEM_PROMPT = """You are an objective hint-following analyst. Your task is to provide a final assessment of how well the model's reasoning aligns with the given hints.

## Important Guidelines:
1. You are ONLY evaluating alignment, NOT judging whether the hints are correct or helpful
2. Be completely objective - do not assume hints are good or bad
3. A model that deviates from hints with good justification may be appropriate

## Output Format:
Return a JSON object with exactly these fields:
{
    "overall_alignment": "high|medium|low",
    "alignment_score": <0-100>,
    "alignment_comment": "<brief justification for alignment score>",
    "deviation_analysis": "justified|unjustified|mixed|none",
    "deviation_comment": "<analysis of deviations if any>"
}

## Scoring Guide for alignment_score:
- 90-100: Almost all steps align with relevant hints
- 70-89: Majority of steps align, some neutral
- 50-69: Mixed alignment, some deviations
- 30-49: Many deviations, partial alignment
- 0-29: Most steps deviate from hints

## deviation_analysis values:
- justified: All deviations have reasonable justification
- unjustified: Deviations lack justification
- mixed: Some justified, some not
- none: No deviations to analyze

Be objective. Do NOT make value judgments about the hints themselves."""

    FINAL_EVAL_USER_TEMPLATE = """## Original Question:
{question}

## Hints Provided:
{hints_text}

## Step-by-Step Hint Alignment Analysis:
{checks_summary}

## Statistics:
- Total Steps: {total_steps}
- Aligned Steps: {aligned_steps}
- Deviated Steps: {deviated_steps}
- Neutral Steps: {neutral_steps}
- Justified Deviations: {justified_deviations}

Provide your final hint-follow evaluation. Return JSON only."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        
        # 初始化工具
        self.hint_check_tool = HintCheckTool(
            model_id=self.config.hint_model if hasattr(self.config, 'hint_model') else self.config.check_model,
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
    
    @staticmethod
    def parse_hints(hint_text: str) -> List[str]:
        """
        解析原始 hint 文本，将其分割为单独的 hint 条目。
        
        支持的格式：
        1. "- " 开头的多行格式
        2. 换行符分隔
        """
        if not hint_text or not hint_text.strip():
            return []
        
        hints = []
        
        # 处理 "- " 格式
        if "\n- " in hint_text or hint_text.startswith("- "):
            lines = hint_text.split("\n")
            current_hint = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("- "):
                    if current_hint:
                        hints.append(" ".join(current_hint))
                    current_hint = [line[2:].strip()]
                elif line and current_hint:
                    current_hint.append(line)
            
            if current_hint:
                hints.append(" ".join(current_hint))
        else:
            # 按换行分割
            for line in hint_text.split("\n"):
                line = line.strip()
                if line:
                    hints.append(line)
        
        return hints if hints else [hint_text.strip()]
    
    @staticmethod
    def should_evaluate_dataset(dataset_name: str) -> bool:
        """
        检查是否应该对该数据集进行 hint 跟随评测。
        只在 R_Lite 和 R_Complex 上运行。
        """
        dataset_lower = dataset_name.lower()
        return "r_lite" in dataset_lower or "r_complex" in dataset_lower
    
    async def evaluate_single(
        self,
        question_id: str,
        question: str,
        response: str,
        hint_text: str,
        existing_steps: Optional[List[Dict[str, Any]]] = None,
        save_trace: bool = True,
        trace_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], HintFollowTrace]:
        """
        对单个样本进行 hint 跟随评测。
        
        Args:
            question_id: 问题 ID
            question: 原始问题
            response: 模型回答
            hint_text: 原始 hint 文本
            existing_steps: 已有的推理步骤（从 reasoning_quality 复用）
            save_trace: 是否保存评测过程
            trace_dir: trace 保存目录
            
        Returns:
            (final_scores, trace)
        """
        # 解析 hints
        hints = self.parse_hints(hint_text)
        
        # 如果没有 hints，返回空结果
        if not hints:
            return {
                "alignment_score": None,
                "deviation_analysis": None,
                "comment": "No hints provided",
            }, None
        
        # 获取步骤（复用或从 response 简单提取）
        if existing_steps:
            steps = existing_steps
        else:
            # 简单提取步骤（作为后备）
            steps = self._extract_simple_steps(response)
        
        # 创建 trace
        trace = HintFollowTrace(
            question_id=question_id,
            question=question,
            response=response,
            hints=hints,
            steps=steps,
        )
        
        # 检查每个步骤与 hint 的对齐
        step_contents = [s.get("step", s) if isinstance(s, dict) else str(s) for s in steps]
        
        for i, step_content in enumerate(step_contents):
            previous_steps = step_contents[:i] if i > 0 else None
            check_result = await self.hint_check_tool.forward(
                step_content=step_content,
                step_index=i + 1,
                hints=hints,
                previous_steps=previous_steps,
                question=question,
            )
            trace.hint_checks.append(check_result)
        
        # 计算最终评分
        final_scores = await self._compute_final_scores(
            question, hints, trace.hint_checks
        )
        trace.final_scores = final_scores
        
        # 保存 trace
        if save_trace and trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, f"hint_trace_{question_id}.txt")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write(trace.to_txt())
        
        return final_scores, trace
    
    def _extract_simple_steps(self, response: str) -> List[Dict[str, Any]]:
        """
        简单地从 response 中提取步骤（作为后备方案）。
        """
        # 尝试按换行或句号分割
        steps = []
        
        # 移除常见的标签
        response = re.sub(r'^(Reasoning|Answer|Confidence):\s*', '', response, flags=re.MULTILINE)
        
        # 按段落分割
        paragraphs = response.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:
                steps.append({"step": para})
        
        # 如果没有分出多个步骤，按句号分割
        if len(steps) <= 1:
            sentences = re.split(r'[。.]\s*', response)
            steps = [{"step": s.strip()} for s in sentences if s.strip() and len(s.strip()) > 20]
        
        return steps if steps else [{"step": response}]
    
    async def _compute_final_scores(
        self,
        question: str,
        hints: List[str],
        checks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        根据步骤检查结果，计算最终的 Hint Alignment 和 Deviation Analysis 评分。
        """
        if not checks:
            return {
                "overall_alignment": "none",
                "alignment_score": 0,
                "alignment_comment": "No steps to evaluate",
                "deviation_analysis": "none",
                "deviation_comment": "No steps to analyze",
                "total_steps": 0,
                "aligned_steps": 0,
                "deviated_steps": 0,
                "neutral_steps": 0,
                "justified_deviations": 0,
            }
        
        # 统计信息
        total_steps = len(checks)
        aligned_steps = sum(1 for c in checks if c.get("alignment") == "aligned")
        deviated_steps = sum(1 for c in checks if c.get("alignment") == "deviated")
        neutral_steps = sum(1 for c in checks if c.get("alignment") == "neutral")
        justified_deviations = sum(
            1 for c in checks 
            if c.get("alignment") == "deviated" and c.get("deviation_justified") == True
        )
        
        # 构建检查摘要
        checks_summary = []
        for i, check in enumerate(checks):
            checks_summary.append(
                f"Step {i+1}: alignment={check.get('alignment')}, "
                f"relevant_hints={check.get('relevant_hints', [])}, "
                f"deviation_justified={check.get('deviation_justified')}\n"
                f"  Comment: {check.get('comment', 'N/A')}"
            )
        
        hints_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(hints)])
        
        user_prompt = self.FINAL_EVAL_USER_TEMPLATE.format(
            question=question[:500] if len(question) > 500 else question,
            hints_text=hints_text,
            checks_summary="\n\n".join(checks_summary),
            total_steps=total_steps,
            aligned_steps=aligned_steps,
            deviated_steps=deviated_steps,
            neutral_steps=neutral_steps,
            justified_deviations=justified_deviations,
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
                "overall_alignment": parsed.get("overall_alignment", "medium"),
                "alignment_score": parsed.get("alignment_score", 50),
                "alignment_comment": parsed.get("alignment_comment", ""),
                "deviation_analysis": parsed.get("deviation_analysis", "none"),
                "deviation_comment": parsed.get("deviation_comment", ""),
                "total_steps": total_steps,
                "aligned_steps": aligned_steps,
                "deviated_steps": deviated_steps,
                "neutral_steps": neutral_steps,
                "justified_deviations": justified_deviations,
                "justified_deviation_rate": round(justified_deviations / deviated_steps * 100, 2) if deviated_steps > 0 else None,
            }
        except Exception as e:
            error_msg = str(e)
            print(f"[Hint follow final eval error] {error_msg}")
            
            # Fallback 计算
            non_neutral = aligned_steps + deviated_steps
            alignment_score = int(100 * aligned_steps / non_neutral) if non_neutral > 0 else 50
            
            return {
                "overall_alignment": "high" if alignment_score >= 70 else "medium" if alignment_score >= 40 else "low",
                "alignment_score": alignment_score,
                "alignment_comment": f"Fallback: {aligned_steps}/{non_neutral} non-neutral steps aligned",
                "deviation_analysis": "mixed" if deviated_steps > 0 else "none",
                "deviation_comment": f"Fallback: {justified_deviations}/{deviated_steps} deviations justified" if deviated_steps > 0 else "No deviations",
                "total_steps": total_steps,
                "aligned_steps": aligned_steps,
                "deviated_steps": deviated_steps,
                "neutral_steps": neutral_steps,
                "justified_deviations": justified_deviations,
                "justified_deviation_rate": round(justified_deviations / deviated_steps * 100, 2) if deviated_steps > 0 else None,
                "error": str(e),
            }
    
    async def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        predictions: Dict[str, Dict[str, Any]],
        dataset_name: str,
        model_name: str,
        save_traces: bool = True,
        base_trace_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量评测 hint 跟随能力。
        
        Args:
            questions: 问题列表（必须包含 hint 列）
            predictions: 预测结果（应包含 reasoning_quality.steps）
            dataset_name: 数据集名称
            model_name: 模型名称
            save_traces: 是否保存每道题的评测过程
            base_trace_dir: trace 基础目录
            
        Returns:
            更新后的 predictions，每个样本增加 hint_follow 字段
        """
        # 检查是否应该评测此数据集
        if not self.should_evaluate_dataset(dataset_name):
            print(f"[HintFollow] Skipping dataset {dataset_name} (not R_Lite or R_Complex)")
            return predictions
        
        # 构建 trace 目录：base_trace_dir/dataset_name/model_name/
        if save_traces and base_trace_dir:
            trace_dir = os.path.join(base_trace_dir, dataset_name, model_name)
            os.makedirs(trace_dir, exist_ok=True)
        else:
            trace_dir = None
        
        semaphore = asyncio.Semaphore(self.config.num_workers)
        
        async def process_one(q: Dict[str, Any]):
            async with semaphore:
                uid = str(q["index"])
                if uid not in predictions:
                    return None, None
                
                pred = predictions[uid].copy()
                if "hint_follow" in pred:
                    return uid, pred
                
                # 获取 hint
                hint_text = q.get("hint") or q.get("Hints") or ""
                if not hint_text:
                    return uid, pred
                
                # 获取已有的推理步骤
                existing_steps = None
                if "reasoning_quality" in pred:
                    rq = pred["reasoning_quality"]
                    if isinstance(rq, dict) and "steps" in rq:
                        existing_steps = rq.get("steps")
                
                # 如果 reasoning_quality 中没有 steps，尝试从 trace 字段获取
                if not existing_steps and "trace" in pred:
                    trace_data = pred["trace"]
                    if isinstance(trace_data, dict) and "steps" in trace_data:
                        existing_steps = trace_data["steps"]
                
                scores, trace = await self.evaluate_single(
                    question_id=uid,
                    question=q.get("question", ""),
                    response=pred.get("response", ""),
                    hint_text=hint_text,
                    existing_steps=existing_steps,
                    save_trace=save_traces,
                    trace_dir=trace_dir,
                )
                
                pred["hint_follow"] = scores
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
        计算 hint 跟随相关指标。
        """
        alignment_scores = []
        justified_rates = []
        
        aligned_count = 0
        deviated_count = 0
        neutral_count = 0
        justified_count = 0
        
        for v in predictions.values():
            hf = v.get("hint_follow", {})
            if not hf or hf.get("alignment_score") is None:
                continue
            
            if "alignment_score" in hf:
                alignment_scores.append(hf["alignment_score"])
            if hf.get("justified_deviation_rate") is not None:
                justified_rates.append(hf["justified_deviation_rate"])
            
            aligned_count += hf.get("aligned_steps", 0)
            deviated_count += hf.get("deviated_steps", 0)
            neutral_count += hf.get("neutral_steps", 0)
            justified_count += hf.get("justified_deviations", 0)
        
        if not alignment_scores:
            return {}
        
        total_non_neutral = aligned_count + deviated_count
        
        return {
            "alignment_score_avg": round(sum(alignment_scores) / len(alignment_scores), 2),
            "justified_deviation_rate_avg": round(sum(justified_rates) / len(justified_rates), 2) if justified_rates else None,
            "total_aligned_steps": aligned_count,
            "total_deviated_steps": deviated_count,
            "total_neutral_steps": neutral_count,
            "total_justified_deviations": justified_count,
            "overall_alignment_rate": round(100 * aligned_count / total_non_neutral, 2) if total_non_neutral > 0 else None,
            "n_scored": len(alignment_scores),
        }





