"""
评测主入口
=========
串联正确性评测和推理质量评测，输出结果和指标。
"""
from __future__ import annotations

import os
import json
import argparse
import asyncio
from typing import Dict, Any, List

# 加载 .env 文件（必须在导入 config 之前）
def _load_env():
    try:
        from dotenv import load_dotenv
        # 尝试从当前目录加载
        if load_dotenv(override=True):
            return True
        # 尝试从 Morphobench 目录加载
        if load_dotenv("/data0/Morphobench/.env", override=True):
            return True
        return False
    except ImportError:
        print("⚠️ python-dotenv not installed, run: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"⚠️ Error loading .env: {e}")
        return False

_env_loaded = _load_env()

from datasets import load_dataset

# 导入 config（此时环境变量已加载）
from .config import EvalConfig
from .evaluators.correctness import CorrectnessEvaluator
from .evaluators.reasoning_quality import ReasoningQualityEvaluator
from .evaluators.hint_follow import HintFollowEvaluator


def load_questions(dataset_path: str) -> List[Dict[str, Any]]:
    """加载数据集"""
    dataset = load_dataset(dataset_path, split="train").to_dict()
    return [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]


async def run_evaluation(args: argparse.Namespace, config: EvalConfig):
    """
    运行完整评测流程：
    1. 正确性评测
    2. 推理质量评测
    """
    # 加载数据
    questions = load_questions(args.dataset)
    total_q = len(questions)
    print(f"Loaded {total_q} questions from {args.dataset}")
    
    # 加载预测结果
    with open(args.predictions, "r") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")
    
    # 加载已有评测结果（增量评测）
    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            existing = json.load(f)
        predictions.update(existing)
        print(f"Loaded {len(existing)} existing evaluations")
    
    # 过滤：只评测有预测的题目
    questions = [q for q in questions if str(q["index"]) in predictions]
    print(f"Will evaluate {len(questions)} questions")
    
    # ========== 1. 正确性评测 ==========
    if not args.skip_correctness:
        print("\n" + "=" * 60)
        print("PHASE 1: Correctness Evaluation")
        print("=" * 60)
        
        correctness_eval = CorrectnessEvaluator(config)
        predictions = await correctness_eval.evaluate_batch(questions, predictions)
        
        # 中间保存
        with open(args.output_json, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Correctness evaluation saved to {args.output_json}")
    
    # ========== 2. 推理质量评测 ==========
    if not args.skip_reasoning:
        print("\n" + "=" * 60)
        print("PHASE 2: Reasoning Quality Evaluation (ReAct Loop)")
        print("=" * 60)
        
        # 按数据集和模型组织 trace 目录
        reasoning_trace_dir = os.path.join(
            args.trace_dir, "reasoning_quality", args.dataset_name, args.model_name
        )
        os.makedirs(reasoning_trace_dir, exist_ok=True)
        
        reasoning_eval = ReasoningQualityEvaluator(config)
        predictions = await reasoning_eval.evaluate_batch(
            questions,
            predictions,
            save_traces=True,
            trace_dir=reasoning_trace_dir,
        )
        
        # 保存
        with open(args.output_json, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Reasoning quality evaluation saved to {args.output_json}")
    
    # ========== 3. Hint 跟随评测（仅 R_Lite 和 R_Complex）==========
    if not args.skip_hint_follow and HintFollowEvaluator.should_evaluate_dataset(args.dataset_name):
        print("\n" + "=" * 60)
        print("PHASE 3: Hint Follow Evaluation (R_Lite / R_Complex only)")
        print("=" * 60)
        
        hint_trace_dir = os.path.join(args.trace_dir, "hint_follow")
        
        hint_eval = HintFollowEvaluator(config)
        predictions = await hint_eval.evaluate_batch(
            questions,
            predictions,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            save_traces=True,
            base_trace_dir=hint_trace_dir,
        )
        
        # 保存
        with open(args.output_json, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Hint follow evaluation saved to {args.output_json}")
    elif not args.skip_hint_follow:
        print(f"\n[HintFollow] Skipping - dataset {args.dataset_name} is not R_Lite or R_Complex")
    
    # ========== 4. 计算并输出指标 ==========
    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    
    stats = {}
    
    # 正确性指标
    correctness_stats = CorrectnessEvaluator.compute_metrics(predictions, total_q)
    if correctness_stats:
        stats["correctness"] = correctness_stats
        print(f"\n[Correctness]")
        print(f"  Accuracy: {correctness_stats.get('accuracy', 0)}% ± {correctness_stats.get('ci_95', 0)}%")
        print(f"  Calibration Error: {correctness_stats.get('calibration_error', 0)}")
        print(f"  Avg Response Length: {correctness_stats.get('avg_response_length', 0)}")
    
    # 推理质量指标
    reasoning_stats = ReasoningQualityEvaluator.compute_metrics(predictions)
    if reasoning_stats:
        stats["reasoning_quality"] = reasoning_stats
        print(f"\n[Reasoning Quality]")
        print(f"  Completeness (avg): {reasoning_stats.get('completeness_avg', 0)}/100")
        print(f"  Logical Coherence (avg): {reasoning_stats.get('logical_coherence_avg', 0)}/100")
        print(f"  Avg Steps: {reasoning_stats.get('avg_steps', 0)}")
    
    # Hint 跟随指标
    hint_stats = HintFollowEvaluator.compute_metrics(predictions)
    if hint_stats:
        stats["hint_follow"] = hint_stats
        print(f"\n[Hint Follow]")
        print(f"  Alignment Score (avg): {hint_stats.get('alignment_score_avg', 'N/A')}/100")
        print(f"  Overall Alignment Rate: {hint_stats.get('overall_alignment_rate', 'N/A')}%")
        print(f"  Justified Deviation Rate (avg): {hint_stats.get('justified_deviation_rate_avg', 'N/A')}%")
        print(f"  Aligned/Deviated/Neutral Steps: {hint_stats.get('total_aligned_steps', 0)}/{hint_stats.get('total_deviated_steps', 0)}/{hint_stats.get('total_neutral_steps', 0)}")
    
    # 保存指标
    stats_path = args.output_json.replace(".json", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    
    # 文本格式统计
    stats_txt_path = args.output_json.replace(".json", "_stats.txt")
    with open(stats_txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        if correctness_stats:
            f.write("[Correctness]\n")
            f.write(f"  Accuracy: {correctness_stats.get('accuracy', 0)}% ± {correctness_stats.get('ci_95', 0)}%\n")
            f.write(f"  Calibration Error: {correctness_stats.get('calibration_error', 0)}\n")
            f.write(f"  Avg Response Length: {correctness_stats.get('avg_response_length', 0)}\n")
            f.write(f"  Total: {correctness_stats.get('n_total', 0)}, Scored: {correctness_stats.get('n_scored', 0)}\n\n")
        
        if reasoning_stats:
            f.write("[Reasoning Quality]\n")
            f.write(f"  Completeness (avg): {reasoning_stats.get('completeness_avg', 0)}/100\n")
            f.write(f"  Logical Coherence (avg): {reasoning_stats.get('logical_coherence_avg', 0)}/100\n")
            f.write(f"  Avg Steps: {reasoning_stats.get('avg_steps', 0)}\n")
            f.write(f"  Scored: {reasoning_stats.get('n_scored', 0)}\n\n")
        
        if hint_stats:
            f.write("[Hint Follow]\n")
            f.write(f"  Alignment Score (avg): {hint_stats.get('alignment_score_avg', 'N/A')}/100\n")
            f.write(f"  Overall Alignment Rate: {hint_stats.get('overall_alignment_rate', 'N/A')}%\n")
            f.write(f"  Justified Deviation Rate (avg): {hint_stats.get('justified_deviation_rate_avg', 'N/A')}%\n")
            f.write(f"  Aligned Steps: {hint_stats.get('total_aligned_steps', 0)}\n")
            f.write(f"  Deviated Steps: {hint_stats.get('total_deviated_steps', 0)}\n")
            f.write(f"  Neutral Steps: {hint_stats.get('total_neutral_steps', 0)}\n")
            f.write(f"  Justified Deviations: {hint_stats.get('total_justified_deviations', 0)}\n")
            f.write(f"  Scored: {hint_stats.get('n_scored', 0)}\n")
    
    print(f"Stats (txt) saved to {stats_txt_path}")
    
    return predictions, stats


def main():
    parser = argparse.ArgumentParser(description="MorphoBench Evaluation Agent")
    
    # 数据相关
    parser.add_argument("--dataset", required=True, help="Dataset name or local path")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--difficulty", choices=["easy", "hard", "perturbed", "v0"], required=True)
    parser.add_argument("--model_name", required=True, help="Model name for output naming")
    
    # 输出相关
    parser.add_argument("--output_dir", default="./output/eval_agent_result")
    parser.add_argument("--trace_dir", default="./output/eval_agent_traces")
    
    # API 相关
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--breakdown_model", default=None)
    parser.add_argument("--check_model", default=None)
    parser.add_argument("--summary_model", default=None)
    parser.add_argument("--hint_model", default=None)
    
    # 并发相关
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    
    # 控制选项
    parser.add_argument("--skip_correctness", action="store_true", help="Skip correctness evaluation")
    parser.add_argument("--skip_reasoning", action="store_true", help="Skip reasoning quality evaluation")
    parser.add_argument("--skip_hint_follow", action="store_true", help="Skip hint follow evaluation")
    
    args = parser.parse_args()
    
    # 从 dataset 路径提取数据集名称
    args.dataset_name = os.path.basename(args.dataset.rstrip("/"))
    
    # 构建配置
    config = EvalConfig()
    if args.api_key:
        config.api_key = args.api_key
    if args.base_url:
        config.base_url = args.base_url
    if args.judge_model:
        config.judge_model = args.judge_model
    if args.breakdown_model:
        config.breakdown_model = args.breakdown_model
    if args.check_model:
        config.check_model = args.check_model
    if args.summary_model:
        config.summary_model = args.summary_model
    if args.hint_model:
        config.hint_model = args.hint_model
    if args.num_workers:
        config.num_workers = args.num_workers
    if args.max_tokens:
        config.max_completion_tokens = args.max_tokens
    config.output_dir = args.output_dir
    config.trace_dir = args.trace_dir
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.trace_dir, exist_ok=True)
    
    # 输出文件路径
    args.output_json = os.path.join(
        args.output_dir,
        f"eval_{args.difficulty}_{args.model_name}.json"
    )
    
    # 运行评测
    print("=" * 60)
    print("MorphoBench Evaluation Agent")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Predictions: {args.predictions}")
    print(f"Output: {args.output_json}")
    print(f"Traces: {args.trace_dir}")
    print("=" * 60)
    
    # 打印 API 配置信息（用于调试）
    print("\n[API Configuration]")
    print(f"  .env loaded: {_env_loaded}")
    print(f"  ENV API_KEY: {bool(os.getenv('API_KEY'))}")
    print(f"  ENV API_BASE: {os.getenv('API_BASE', 'not set')}")
    api_key_display = config.api_key[:8] + "..." + config.api_key[-4:] if len(config.api_key) > 12 else "***"
    print(f"  Config API Key: {api_key_display}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Judge Model: {config.judge_model}")
    print(f"  Breakdown Model: {config.breakdown_model}")
    print(f"  Check Model: {config.check_model}")
    print(f"  Summary Model: {config.summary_model}")
    print(f"  Hint Model: {config.hint_model}")
    print(f"  Num Workers: {config.num_workers}")
    print(f"  Max Tokens: {config.max_completion_tokens}")
    print("=" * 60)
    
    # 验证 API 配置
    if config.api_key in ["YOUR_API_KEY", "", None]:
        print("\n❌ ERROR: API_KEY 未正确配置！")
        print("请通过以下方式之一设置 API_KEY：")
        print("  1. 环境变量: export API_KEY=your_key")
        print("  2. .env 文件: API_KEY=your_key")
        print("  3. 命令行参数: --api_key your_key")
        return
    
    if config.base_url in ["API_BASE_URL", "", None]:
        print("\n❌ ERROR: API_BASE (base_url) 未正确配置！")
        print("请通过以下方式之一设置 API_BASE：")
        print("  1. 环境变量: export API_BASE=your_base_url")
        print("  2. .env 文件: API_BASE=your_base_url")
        print("  3. 命令行参数: --base_url your_base_url")
        return
    
    print("\n✅ API 配置验证通过，开始评测...\n")
    
    asyncio.run(run_evaluation(args, config))
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()


