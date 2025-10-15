import os
import json
import argparse
import asyncio
import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


# ======= Configuration =======
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def format_message(question: dict, model: str):
    """Format a single question (with optional image) into a chat message."""
    question_text = question.get("question", "")
    hint_text = question.get("hint", "")

    combined_text = f"{question_text}\n\nHint: {hint_text}" if hint_text else question_text
    text_content = {"type": "text", "text": combined_text}

    if question.get("image_base64"):
        image_content = {
            "type": "image_url",
            "image_url": {"url": question["image_base64"]}
        }
        content = [text_content, image_content]
    else:
        content = [text_content]

    system_role = "user" if "o1" in model else "system"
    messages = [
        {"role": system_role, "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return messages


async def attempt_question(client: AsyncOpenAI, model: str, question: dict, max_completion_tokens: int):
    """Attempt to query one question asynchronously."""
    messages = format_message(question, model)
    try:
        response = await client.chat.completions.create(
            model=model,
            max_completion_tokens=max_completion_tokens,
            messages=messages,
            stream=False,
        )
        content = response.choices[0].message.content
        usage = response.usage.model_dump() if response.usage else {}
        return question["index"], content, usage
    except asyncio.TimeoutError:
        print(f"[Timeout] index={question.get('index')}")
        return None
    except Exception as e:
        print(f"[Error] index={question.get('index')} - {e}")
        return None


async def attempt_all(client, model, questions, max_completion_tokens: int, num_workers: int):
    """Run all questions asynchronously with concurrency limit."""
    semaphore = asyncio.Semaphore(num_workers)
    results = []

    async def worker(question):
        async with semaphore:
            return await attempt_question(client, model, question, max_completion_tokens)

    tasks = [worker(q) for q in questions]
    results = await tqdm_asyncio.gather(*tasks)
    return results


def main(args):
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=600.0,
        max_retries=1,
    )

    dataset = load_dataset(args.dataset, split="train").to_dict()
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    if args.max_samples:
        questions = questions[:args.max_samples]

    output_filepath = args.output
    predictions = {}
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            predictions = json.load(f)
        questions = [q for q in questions if str(q["index"]) not in predictions]

    results = asyncio.run(
        attempt_all(client, args.model, questions, args.max_completion_tokens, args.num_workers)
    )

    for result in results:
        if result is None:
            continue
        idx, response, usage = result
        predictions[str(idx)] = {
            "model": args.model,
            "response": response,
            "usage": usage,
        }

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"✅ Saved results to: {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Local or remote dataset path")
    parser.add_argument("--model", type=str, required=True, help="Model endpoint name")
    parser.add_argument("--api_key", type=str, default="YOUR_API_KEY", help="API key for model server")
    parser.add_argument("--base_url", type=str, default="API_BASE_URL", help="Model API base URL")
    parser.add_argument("--output", type=str, required=True, help="Output file path for results")
    parser.add_argument("--num_workers", type=int, default=1000, help="Concurrency level for async requests")
    parser.add_argument("--max_completion_tokens", type=int, default=4096, help="Max completion tokens")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation to first N samples")
    args = parser.parse_args()

    main(args)
