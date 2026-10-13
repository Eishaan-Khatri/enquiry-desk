"""Step 3. Score a model on EVERY example of a test file. Not a sample.

    uv run python finetune/evaluate.py --data finetune/data/test.jsonl --base-url http://localhost:8001/v1 --model my-adapter

The model can be anything behind a Chat Completions address: the base model or the fine-tuned
one on vLLM, a local model on Ollama, or a hosted API. Run it once per model on the same file
and the summaries are comparable.

The tools are put in the prompt as text and the model's raw reply is parsed here. The server's
own tool-call parser is not used, so two servers cannot differ in how they score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from enquiry_desk.finetune.data import read_jsonl, to_chat
from enquiry_desk.finetune.metrics import score_one, summarise
from enquiry_desk.llm.wire import ChatClient, LLMError


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=4, help="requests in flight at once")
    parser.add_argument("--limit", type=int, help="only for a quick trial. A reported number uses the whole file")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    examples = read_jsonl(args.data)
    if args.limit:
        examples = examples[: args.limit]
    chat = ChatClient(args.base_url, args.api_key, args.model, timeout=600)
    gate = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = [{} for _ in examples]

    async def one(index: int, example: dict) -> None:
        async with gate:
            try:
                completion = await chat.complete(to_chat(example)[:2], temperature=0.0, max_tokens=400)
                output = completion.content
            except LLMError as error:
                output = f"<tool_call>request failed: {error}</tool_call>"  # counts as an unreadable answer
            rows[index] = {"query": example["query"], "gold": example["answers"], "output": output, **score_one(output, example["answers"])}

    started = time.perf_counter()
    await asyncio.gather(*(one(i, e) for i, e in enumerate(examples)))
    await chat.aclose()

    summary = summarise(rows) | {"model": args.model, "data": str(args.data), "seconds": round(time.perf_counter() - started, 1),
                                 "whole_file": args.limit is None}
    print(json.dumps(summary, indent=1))
    failures = [row for row in rows if not row["exact"]]
    print(f"\n{len(failures)} of {len(rows)} not exact. The first five, to read by hand:")
    for row in failures[:5]:
        print(f"  asked : {row['query'][:110]}\n  wanted: {json.dumps(row['gold'])[:110]}\n  got   : {row['output'][:110]!r}\n")

    out = args.out or args.data.with_name(f"scores_{args.model.replace(':', '_').replace('/', '_')}.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")
    try:  # log to MLflow when it is installed and a tracking address is set. Otherwise skip quietly.
        import mlflow

        with mlflow.start_run(run_name=f"eval-{args.model}"):
            mlflow.log_params({"model": args.model, "data": str(args.data), "examples": len(rows)})
            mlflow.log_metrics({name: summary[name]["rate"] for name in ("valid", "right_tools", "exact", "restraint") if name in summary})
            mlflow.log_artifact(str(out))
    except ImportError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
