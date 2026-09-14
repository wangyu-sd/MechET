import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm


def answer_text(text):
    text = (text or "").strip()
    left = text.find("<answer>")
    if left != -1:
        text = text[left + len("<answer>") :]
    right = text.find("</answer>")
    if right != -1:
        text = text[:right]
    return f"<answer>{text.strip()}</answer>"


async def think(client, semaphore, model, prompt, args):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.think_temperature,
            max_tokens=args.think_max_tokens,
            stop=["</think>", "<answer>"],
        )
    prefix = (response.choices[0].message.content or "").strip()
    if "</think>" not in prefix and "<answer>" not in prefix:
        prefix += "</think>\n\n<answer>"
    return prefix


async def answer(client, semaphore, model, prompt, prefix, args):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix},
            ],
            temperature=args.answer_temperature,
            max_tokens=args.answer_max_tokens,
            n=args.n_answer,
            extra_body={"continue_final_message": True},
        )
    return [answer_text(choice.message.content) for choice in response.choices]


async def generate_one(client, think_semaphore, answer_semaphore, model, index, item, args):
    prefixes = await asyncio.gather(
        *[
            think(client, think_semaphore, model, item["input"], args)
            for _ in range(args.n_think)
        ]
    )
    answers = await asyncio.gather(
        *[
            answer(client, answer_semaphore, model, item["input"], prefix, args)
            for prefix in prefixes
        ]
    )
    return index, {
        "input": item["input"],
        "output": [text for group in answers for text in group],
        "label": item["label"],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model")
    parser.add_argument("--n_think", type=int, default=10)
    parser.add_argument("--n_answer", type=int, default=10)
    parser.add_argument("--think_temperature", type=float, default=1.0)
    parser.add_argument("--answer_temperature", type=float, default=1.2)
    parser.add_argument("--think_max_tokens", type=int, default=2048)
    parser.add_argument("--answer_max_tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=2048)
    parser.add_argument("--answer_concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    with open(args.data_path) as source:
        data = [json.loads(line) for line in source if line.strip()]

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    model = args.model or (await client.models.list()).data[0].id
    think_semaphore = asyncio.Semaphore(args.concurrency)
    answer_semaphore = asyncio.Semaphore(args.answer_concurrency)
    tasks = [
        generate_one(client, think_semaphore, answer_semaphore, model, i, item, args)
        for i, item in enumerate(data)
    ]
    results = [None] * len(tasks)
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="two-stage"):
        index, result = await task
        results[index] = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.task}.jsonl").open("w") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())

