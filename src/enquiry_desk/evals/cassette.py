"""Record model calls to a file once, then replay them with no model and no cost.

A wrapper around the model client. In "record" mode every call goes to the real model and the
answer is saved under a key made from the request. In "replay" mode the answer comes from the
file, and a request that was never recorded is an error.

Replay only reproduces a run when every request is identical, and the desk's instructions
contain today's date. So a recorded suite must be replayed with the clock fixed to the day
it was recorded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from enquiry_desk.llm.wire import Completion, ToolCall, Usage


class NotRecorded(KeyError):
    pass


def request_key(messages: list[dict], tools: list[dict] | None) -> str:
    blob = json.dumps({"messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


class CassetteChat:
    def __init__(self, inner, path: Path, mode: str):
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        self._inner = inner
        self._path = path
        self._mode = mode
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self._tape: dict[str, dict] = saved.get("calls", {})
        # On replay there is no model, so the name comes from the file. One file holds one
        # model's answers, which is why the model is not part of the key.
        self.model = inner.model if inner is not None else saved.get("model", "unknown")

    async def complete(self, messages: list[dict], tools: list[dict] | None = None, **options) -> Completion:
        key = request_key(messages, tools)
        if self._mode == "replay":
            if key not in self._tape:
                raise NotRecorded(f"no recorded answer for request {key}. Record the suite again.")
            saved = self._tape[key]
            return Completion(
                content=saved["content"],
                tool_calls=[ToolCall(**call) for call in saved["tool_calls"]],
                finish_reason=saved["finish_reason"],
                usage=Usage(**saved["usage"]) if saved["usage"] else None,
                model=saved["model"],
                ttft_ms=0.0,
                latency_ms=0.0,
            )
        completion = await self._inner.complete(messages, tools, **options)
        self._tape[key] = {
            "content": completion.content,
            "tool_calls": [call.__dict__ for call in completion.tool_calls],
            "finish_reason": completion.finish_reason,
            "usage": completion.usage.__dict__ if completion.usage else None,
            "model": completion.model,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"model": self.model, "calls": self._tape}, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        return completion
