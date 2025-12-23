"""
正确性评测
=========
忠实沿用 /data0/Morphobench/scripts/evaluate_judge.py 的逻辑。
比较模型最终答案与标准答案的一致性。
"""
from __future__ import annotations

import os
import json
import copy
import math
import asyncio
from typing import Dict, Any, List, Optional, Literal

import numpy as np
from pydantic import BaseModel
from openai import AsyncOpenAI

from ..config import EvalConfig


# ==================== Prompt & Schema ====================

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
"""


class ExtractedAnswer(BaseModel):
    """结构化输出格式"""
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True


# ==================== 核心评测逻辑 ====================

class CorrectnessEvaluator:
    """
    正确性评测器。
    
    功能：
    - 调用 judge 模型比较模型答案与标准答案
    - 支持批量异步评测
    - 计算准确率、置信度校准等指标
    """
    
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
    
    async def judge_single(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> Optional[Dict[str, Any]]:
        """
        对单个样本进行正确性判断。
        """
        prompt = JUDGE_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )
        
        try:
            completion = await self.client.beta.chat.completions.parse(
                model=self.config.judge_model,
                max_completion_tokens=self.config.max_completion_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_format=ExtractedAnswer,
            )
            content = completion.choices[0].message.parsed
            return {
                "correct_answer": correct_answer,
                "model_answer": content.extracted_final_answer,
                "reasoning": content.reasoning,
                "correct": content.correct,
                "confidence": content.confidence,
            }
        except Exception as e:
            error_msg = str(e)
            print(f"[Correctness judge error] {error_msg}")
            # 打印更详细的调试信息
            if "Connection" in error_msg or "connection" in error_msg:
                print(f"  -> 连接错误，请检查 base_url: {self.client.base_url}")
            elif "401" in error_msg or "Unauthorized" in error_msg:
                print(f"  -> 认证错误，请检查 api_key 是否正确")
            elif "404" in error_msg:
                print(f"  -> 模型不存在，请检查 model: {self.config.judge_model}")
            return None
    
    async def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        predictions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量评测正确性。
        
        Args:
            questions: 问题列表，每个包含 index, question, answer
            predictions: 预测结果，key 为 index，value 包含 response
            
        Returns:
            更新后的 predictions，每个样本增加 judge_response 字段
        """
        semaphore = asyncio.Semaphore(self.config.num_workers)
        
        async def process_one(q: Dict[str, Any]):
            async with semaphore:
                uid = str(q["index"])
                if uid not in predictions:
                    return None, None
                
                pred = copy.deepcopy(predictions[uid])
                if "judge_response" in pred:
                    return uid, pred
                
                result = await self.judge_single(
                    question=q.get("question", ""),
                    correct_answer=q.get("answer", ""),
                    response=pred.get("response", ""),
                )
                
                if result is not None:
                    pred["judge_response"] = result
                    return uid, pred
                return None, None
        
        tasks = [process_one(q) for q in questions]
        results = await asyncio.gather(*tasks)
        
        updated = {}
        for uid, pred in results:
            if uid is not None and pred is not None:
                updated[uid] = pred
        
        return updated
    
    @staticmethod
    def compute_metrics(
        predictions: Dict[str, Dict[str, Any]],
        total_n: int,
    ) -> Dict[str, Any]:
        """
        计算正确性相关指标。
        """
        correct_list = []
        confidence_list = []
        lengths = []
        
        for v in predictions.values():
            jr = v.get("judge_response")
            if jr:
                correct_list.append(jr.get("correct") == "yes")
                confidence_list.append(jr.get("confidence", 100) / 100)
                lengths.append(len(v.get("response", "")))
        
        if not correct_list:
            return {}
        
        correct_arr = np.array(correct_list)
        confidence_arr = np.array(confidence_list)
        
        accuracy = round(100 * np.mean(correct_arr), 2)
        ci = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / max(total_n, 1)), 2)
        calib = round(100 * CorrectnessEvaluator._calib_err(confidence_arr, correct_arr), 2)
        avg_len = round(np.mean(lengths), 2) if lengths else 0.0
        
        return {
            "accuracy": accuracy,
            "ci_95": ci,
            "calibration_error": calib,
            "avg_response_length": avg_len,
            "n_total": total_n,
            "n_scored": len(correct_list),
        }
    
    @staticmethod
    def _calib_err(
        confidence: np.ndarray,
        correct: np.ndarray,
        p: str = "2",
        beta: int = 100,
    ) -> float:
        """
        计算校准误差（沿用原脚本逻辑）。
        """
        idxs = np.argsort(confidence)
        confidence = confidence[idxs]
        correct = correct[idxs]
        bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
        if bins:
            bins[-1] = [bins[-1][0], len(confidence)]
        
        cerr = 0.0
        total_examples = len(confidence)
        
        for b in bins[:-1]:
            bin_conf = confidence[b[0]:b[1]]
            bin_corr = correct[b[0]:b[1]]
            if len(bin_conf) == 0:
                continue
            diff = abs(np.nanmean(bin_conf) - np.nanmean(bin_corr))
            if p == "2":
                cerr += len(bin_conf) / total_examples * diff ** 2
            elif p == "1":
                cerr += len(bin_conf) / total_examples * diff
            elif p in ["infty", "infinity", "max"]:
                cerr = max(cerr, diff)
        
        if p == "2":
            cerr = math.sqrt(cerr)
        
        return cerr
    
    @staticmethod
    def dump_stats(stats: Dict[str, Any], path: str) -> None:
        """
        将统计结果写入文本文件。
        """
        lines = [
            f"Accuracy: {stats.get('accuracy', 0)}% ± {stats.get('ci_95', 0)}% | n={stats.get('n_total', 0)} (scored={stats.get('n_scored', 0)})",
            f"Calibration Error: {stats.get('calibration_error', 0)}",
            f"Average response length: {stats.get('avg_response_length', 0)}",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))


