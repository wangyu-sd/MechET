import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm


async def generate_one(client, semaphore, model, index, item, args):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": item["input"]}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.n,
        )
    outputs = [(choice.message.content or "").strip() for choice in response.choices]
    return index, {"input": item["input"], "output": outputs, "label": item["label"]}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    with open(args.data_path) as source:
        data = [json.loads(line) for line in source if line.strip()]

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    model = args.model or (await client.models.list()).data[0].id
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [generate_one(client, semaphore, model, i, item, args) for i, item in enumerate(data)]
    results = [None] * len(tasks)
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="generate"):
        index, result = await task
        results[index] = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.task}.jsonl").open("w") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())

