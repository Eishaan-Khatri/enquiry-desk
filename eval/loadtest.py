"""Many callers at once, against a fake model, to find what breaks first in the desk itself.

    uv run python eval/loadtest.py --callers 20 --turns 4 --model-delay 0.3

The model is replaced by a small server in this process that waits `--model-delay` seconds
and then answers. So the numbers measure the desk's own code under load: the graph, the
checkpointer, SQLite, the search. They say nothing about a real model's speed.

It starts the real API on port 8000, so Prometheus (if it is running) scrapes this run and
the Grafana dashboard fills in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

QUESTIONS = [
    "What is the fee for the JEE dropper classroom batch?",
    "When is the scholarship test?",
    "Is there a hostel?",
    "What are the library hours?",
    "How do refunds work?",
    "Where is the centre?",
]


def fake_model(delay: float, fail_every: int) -> FastAPI:
    app = FastAPI()
    seen = {"calls": 0}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        seen["calls"] += 1
        if fail_every and seen["calls"] % fail_every == 0:
            return StreamingResponse(iter([b"overloaded"]), status_code=503)
        last = body["messages"][-1]

        def chunk(delta: dict, finish: str | None = None) -> str:
            return "data: " + json.dumps({"model": "fake", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"

        async def events():
            await asyncio.sleep(delay)
            if last["role"] == "tool":
                yield chunk({"content": "Here is what our documents say. Please ask if you need more."}, "stop")
            else:
                arguments = json.dumps({"query": last["content"]})
                yield chunk({"tool_calls": [{"index": 0, "id": f"c{seen['calls']}", "type": "function", "function": {"name": "search_knowledge", "arguments": arguments}}]}, "tool_calls")
            yield "data: " + json.dumps({"model": "fake", "choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 20}}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def serve(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


async def one_caller(client: httpx.AsyncClient, number: int, turns: int, times: list[float], errors: list[str]) -> None:
    session_id = f"load-{int(time.time())}-{number}"
    for turn in range(turns):
        started = time.perf_counter()
        try:
            answer = await client.post("/v1/chat", json={"session_id": session_id, "message": QUESTIONS[(number + turn) % len(QUESTIONS)]})
            answer.raise_for_status()
            if answer.json()["usage"]["errors"]:
                errors.append("turn had a failed model or tool call")
        except Exception as error:
            errors.append(f"{type(error).__name__}: {str(error)[:80]}")
        times.append(time.perf_counter() - started)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--callers", type=int, default=20)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--model-delay", type=float, default=0.3)
    parser.add_argument("--fail-every", type=int, default=0, help="make every Nth model call fail with a 503")
    parser.add_argument("--hold", type=float, default=0, help="keep the API up this many seconds afterwards, for Prometheus")
    args = parser.parse_args()

    # Settings are read from the environment when the app starts, so set them before importing it.
    os.environ |= {
        "DESK_LLM_BASE_URL": "http://127.0.0.1:8765/v1",
        "DESK_LLM_MODEL": "fake",
        "DESK_DATA_DIR": tempfile.mkdtemp(prefix="desk-load-"),
    }
    from enquiry_desk.api import app

    serve(fake_model(args.model_delay, args.fail_every), 8765)
    serve(app, 8000)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=120) as client:
        for _ in range(100):
            try:
                if (await client.get("/healthz")).status_code == 200:
                    break
            except httpx.TransportError:
                await asyncio.sleep(0.3)

        times: list[float] = []
        errors: list[str] = []
        started = time.perf_counter()
        await asyncio.gather(*(one_caller(client, n, args.turns, times, errors) for n in range(args.callers)))
        elapsed = time.perf_counter() - started

        print(f"{args.callers} callers x {args.turns} turns, model delay {args.model_delay} s, two model calls per turn")
        print(f"  turns answered   {len(times)} in {elapsed:.1f} s  ({len(times) / elapsed:.1f} turns per second)")
        print(f"  turn time        median {statistics.median(times):.2f} s, 95th percentile {percentile(times, 0.95):.2f} s, slowest {max(times):.2f} s")
        print(f"  floor            {2 * args.model_delay:.2f} s is the least a turn can take, the two model waits alone")
        print(f"  errors           {len(errors)}" + (f"  e.g. {errors[0]}" if errors else ""))
        metrics = (await client.get("/metrics")).text
        print("  /metrics says    " + ", ".join(line for line in metrics.splitlines() if line.startswith(("desk_turns_total", "desk_llm_calls_total"))))
        if args.hold:
            print(f"  holding the API up for {args.hold:.0f} s so Prometheus can scrape it")
            await asyncio.sleep(args.hold)


if __name__ == "__main__":
    asyncio.run(main())
