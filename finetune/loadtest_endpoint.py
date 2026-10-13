"""Step 4. How fast is a model server, and how does that change as more callers arrive?

    uv run python finetune/loadtest_endpoint.py --base-url http://localhost:8001/v1 --model my-adapter

It sends the same short request at several levels of concurrency and reports, for each:
  tokens per second, all callers together   the server's throughput
  time to first token, median and 95th percentile   what one caller feels

What to look for. On a server with continuous batching, such as vLLM, throughput RISES with
concurrency: the GPU processes many requests in one pass, so sixteen callers get far more
than one caller's tokens per second between them. On a server that takes one request at a
time, throughput stays flat and each caller's wait grows in line with the queue.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from enquiry_desk.llm.wire import ChatClient, LLMError

PROMPT = [
    {"role": "system", "content": "You are a helpful assistant for a coaching institute."},
    {"role": "user", "content": "In three sentences, explain what a counselling call is for."},
]


async def one(chat: ChatClient, results: list[dict]) -> None:
    started = time.perf_counter()
    try:
        completion = await chat.complete(PROMPT, temperature=0.0, max_tokens=96)
    except LLMError as error:
        results.append({"error": str(error)[:80]})
        return
    tokens = completion.usage.completion_tokens if completion.usage else max(1, len(completion.content) // 4)
    results.append({"ttft": (completion.ttft_ms or 0) / 1000, "seconds": time.perf_counter() - started, "tokens": tokens})


async def level(base_url: str, api_key: str, model: str, concurrency: int, requests: int) -> dict:
    chat = ChatClient(base_url, api_key, model, timeout=900, max_retries=0)
    results: list[dict] = []
    gate = asyncio.Semaphore(concurrency)

    async def guarded() -> None:
        async with gate:
            await one(chat, results)

    started = time.perf_counter()
    await asyncio.gather(*(guarded() for _ in range(requests)))
    elapsed = time.perf_counter() - started
    await chat.aclose()
    good = [r for r in results if "error" not in r]
    if not good:
        return {"concurrency": concurrency, "errors": len(results)}
    first = sorted(r["ttft"] for r in good)
    return {
        "concurrency": concurrency,
        "requests": len(good),
        "errors": len(results) - len(good),
        "tokens_per_second": round(sum(r["tokens"] for r in good) / elapsed, 1),
        "ttft_median_s": round(statistics.median(first), 2),
        "ttft_p95_s": round(first[min(len(first) - 1, int(0.95 * len(first)))], 2),
        "request_median_s": round(statistics.median(r["seconds"] for r in good), 2),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--levels", default="1,4,16,32")
    parser.add_argument("--requests-per-level", type=int, default=32)
    args = parser.parse_args()

    await level(args.base_url, args.api_key, args.model, 1, 1)  # one warm-up, so loading the model is not timed
    print(f"{'callers':>8} {'requests':>9} {'errors':>7} {'tokens/s':>9} {'first token median':>19} {'p95':>7} {'request median':>15}")
    for concurrency in (int(x) for x in args.levels.split(",")):
        r = await level(args.base_url, args.api_key, args.model, concurrency, max(concurrency, args.requests_per_level))
        if "tokens_per_second" not in r:
            print(f"{concurrency:>8} every request failed")
            continue
        print(f"{r['concurrency']:>8} {r['requests']:>9} {r['errors']:>7} {r['tokens_per_second']:>9} {r['ttft_median_s']:>18}s {r['ttft_p95_s']:>6}s {r['request_median_s']:>14}s")


if __name__ == "__main__":
    asyncio.run(main())
