import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset


# ==================== 全局配置 ====================
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
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]


async def extract_answer(client, question, correct_answer, response, judge_model, max_tokens):
    """调用 judge 模型，解析结果"""
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
    try:
        completion = await client.beta.chat.completions.parse(
            model=judge_model,
            max_completion_tokens=max_tokens,
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
        print(f"[Error in extract_answer] {e}")
        return None


async def add_judge_response(client, question, predictions, judge_model, max_tokens):
    """为单条预测添加评判结果"""
    unique_id = str(question["index"])
    prediction = copy.deepcopy(predictions[unique_id])
    if "judge_response" in prediction:
        return unique_id, prediction

    question_text = question["question"]
    correct_answer = question["answer"]
    response = prediction.get("response", "")

    content = await extract_answer(client, question_text, correct_answer, response, judge_model, max_tokens)
    if content is not None:
        prediction["judge_response"] = content
        return unique_id, prediction
    else:
        return None, None


async def judge_all_responses(client, questions, predictions, judge_model, num_workers, max_tokens):
    """异步批评估"""
    semaphore = asyncio.Semaphore(num_workers)

    async def bound_func(question):
        async with semaphore:
            return await add_judge_response(client, question, predictions, judge_model, max_tokens)

    results = await tqdm_asyncio.gather(*[bound_func(q) for q in questions])
    return results


# ==================== 统计指标 ====================

def calib_err(confidence, correct, p='2', beta=100):
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if bins:
        bins[-1] = [bins[-1][0], len(confidence)]
    cerr = 0
    total_examples = len(confidence)
    for b in bins[:-1]:
        bin_conf = confidence[b[0]:b[1]]
        bin_corr = correct[b[0]:b[1]]
        if len(bin_conf) == 0:
            continue
        diff = np.abs(np.nanmean(bin_conf) - np.nanmean(bin_corr))
        if p == '2':
            cerr += len(bin_conf) / total_examples * diff ** 2
        elif p == '1':
            cerr += len(bin_conf) / total_examples * diff
        elif p in ['infty', 'infinity', 'max']:
            cerr = np.maximum(cerr, diff)
    if p == '2':
        cerr = np.sqrt(cerr)
    return cerr


def dump_metrics(predictions, n, stats_txt_path):
    correct, confidence, lengths = [], [], []
    for k, v in predictions.items():
        if "judge_response" in v:
            jr = v["judge_response"]
            correct.append(jr["correct"] == "yes")
            confidence.append(jr["confidence"])
            lengths.append(len(v.get("response", "")))
        else:
            print(f"[Warn] Missing judge_response for {k}")

    correct = np.array(correct)
    confidence = np.array(confidence) / 100
    accuracy = round(100 * np.mean(correct), 2)
    ci = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / max(n, 1)), 2)
    calib = round(100 * calib_err(confidence, correct, p='2', beta=100), 2)
    avg_len = round(np.mean(lengths), 2) if lengths else 0.0

    print(f"*** Metrics ***\nAccuracy: {accuracy}% ± {ci}% (n={n})\nCalibration: {calib}\nAvg length: {avg_len}")
    with open(stats_txt_path, "w") as f:
        f.write(f"Accuracy: {accuracy}% ± {ci}% | n={n}\n")
        f.write(f"Calibration Error: {calib}\n")
        f.write(f"Average model response length: {avg_len}\n")


# ==================== 主流程 ====================

def main(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600.0, max_retries=1)
    os.makedirs(args.output_dir, exist_ok=True)

    # 输出路径
    output_json = os.path.join(args.output_dir, f"judged_{args.difficulty}_{args.model_name}.json")
    output_txt = os.path.join(args.output_dir, f"stats_{args.difficulty}_{args.model_name}.txt")

    dataset = load_dataset(args.dataset, split="train").to_dict()
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    total_q = len(questions)

    with open(args.predictions, "r") as f:
        predictions = json.load(f)

    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            judged = json.load(f)
    else:
        judged = {}

    questions = [q for q in questions if str(q["index"]) in predictions and str(q["index"]) not in judged]

    results = asyncio.run(judge_all_responses(client, questions, predictions, args.judge, args.num_workers, args.max_tokens))
    for uid, pred in results:
        if uid is not None:
            judged[uid] = pred

    with open(output_json, "w") as f:
        json.dump(judged, f, indent=4)

    dump_metrics(judged, n=total_q, stats_txt_path=output_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--difficulty", choices=["easy", "hard", "perturbed", "v0"], required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", default="./output/eval_result")
    parser.add_argument("--judge", default="o3-mini-2025-01-31")
    parser.add_argument("--num_workers", type=int, default=1000)
    parser.add_argument("--max_tokens", type=int, default=40960)
    parser.add_argument("--api_key", type=str, default="YOUR_API_KEY", help="API key for model server")
    parser.add_argument("--base_url", type=str, default="API_BASE_URL", help="Model API base URL")
    args = parser.parse_args()
    main(args)