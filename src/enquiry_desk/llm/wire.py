"""Chat Completions over plain HTTP, with no SDK, so the wire format stays visible.

OpenAI, Gemini's compatibility endpoint, Groq, OpenRouter and a vLLM server all accept the
same request at /chat/completions, so this one client drives all of them.

Every call is streamed. Streaming is what makes time to first token measurable, and the
voice path needs the text as it arrives.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        partial: bool = False,
    ):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        # True when text had already been handed to the listener. Such a call is not retried,
        # because the listener would receive the opening words twice.
        self.partial = partial


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text exactly as the model sent it. Nothing has validated it yet.
    # Anything else the server attached to the call, sent back untouched. Gemini puts a
    # "thought signature" here and rejects the next request if it is missing.
    extra: dict | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class Completion:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: Usage | None  # None when the server sent no usage block
    model: str
    ttft_ms: float | None
    latency_ms: float
    attempts: int = 1

    def as_message(self) -> dict:
        """The assistant message to append to the history, in wire format."""
        message: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                | ({"extra_content": call.extra} if call.extra else {})
                for call in self.tool_calls
            ]
        return message


def tool_calls_of(message: dict) -> list[ToolCall]:
    """Read the tool calls back out of an assistant message in the history."""
    return [
        ToolCall(id=c["id"], name=c["function"]["name"], arguments=c["function"]["arguments"], extra=c.get("extra_content"))
        for c in message.get("tool_calls") or []
    ]


class StreamAssembler:
    """Rebuilds one reply from its stream of chunks.

    Text arrives as fragments in delta.content. A tool call arrives as fragments too: the first
    carries the id and the function name, the rest carry pieces of the arguments string, and
    `index` says which call a fragment belongs to.
    """

    def __init__(self, model: str):
        self.model = model
        self.text: list[str] = []
        self.calls: list[ToolCall] = []
        self.finish_reason = ""
        self.usage: Usage | None = None
        self._slot: dict[int, int] = {}  # the server's index -> position in self.calls

    def feed(self, chunk: dict) -> str:
        """Take one parsed chunk. Returns the new text in it, or an empty string."""
        self.model = chunk.get("model") or self.model
        if chunk.get("usage"):
            self.usage = _usage(chunk["usage"])
        new_text = ""
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                new_text += delta["content"]
            for position, fragment in enumerate(delta.get("tool_calls") or []):
                self._feed_call(fragment.get("index", position), fragment)
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
        if new_text:
            self.text.append(new_text)
        return new_text

    def _feed_call(self, index: int, fragment: dict) -> None:
        function = fragment.get("function") or {}
        call_id = fragment.get("id")
        slot = self._slot.get(index)
        # Some servers send each parallel call whole, all under index 0. A new id on an index
        # already in use is therefore a new call, not more of the old one.
        if slot is not None and call_id and self.calls[slot].id and call_id != self.calls[slot].id:
            slot = None
        if slot is None:
            self.calls.append(ToolCall(id=call_id or "", name=function.get("name") or "", arguments=""))
            slot = self._slot[index] = len(self.calls) - 1
        call = self.calls[slot]
        call.id = call.id or call_id or ""
        call.name = call.name or function.get("name") or ""
        call.arguments += function.get("arguments") or ""
        if fragment.get("extra_content"):
            call.extra = fragment["extra_content"]

    @property
    def started(self) -> bool:
        return bool(self.text or self.calls)

    def finish(self, *, ttft_ms: float | None, latency_ms: float) -> Completion:
        for number, call in enumerate(self.calls):
            call.id = call.id or f"call_{number}"
        return Completion(
            content="".join(self.text),
            tool_calls=self.calls,
            finish_reason=self.finish_reason or ("tool_calls" if self.calls else "stop"),
            usage=self.usage,
            model=self.model,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
        )


def _usage(raw: dict) -> Usage:
    details = raw.get("prompt_tokens_details") or {}
    return Usage(
        prompt_tokens=raw.get("prompt_tokens") or 0,
        completion_tokens=raw.get("completion_tokens") or 0,
        cached_tokens=details.get("cached_tokens") or 0,
    )


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        http: httpx.AsyncClient | None = None,
        stream_usage: bool = True,
        max_retries: int = 2,
        timeout: float = 60.0,
        min_gap_s: float = 0.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        # The least time between the starts of two requests. A free tier allows a fixed number
        # of requests a minute, and staying under it is cheaper than being refused and retrying.
        self._min_gap_s = min_gap_s
        self._next_start = 0.0
        self._pace = asyncio.Lock()
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        # The read timeout is the longest allowed silence between two chunks, so on a stream it
        # is in effect a limit on the time to the first token.
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))
        self._stream_usage = stream_usage
        self._max_retries = max_retries
        self._sleep = sleep

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> Completion:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if self._stream_usage:
            payload["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        attempt = 1
        while True:
            if self._min_gap_s:
                async with self._pace:
                    wait = self._next_start - time.monotonic()
                    if wait > 0:
                        await self._sleep(wait)
                    self._next_start = time.monotonic() + self._min_gap_s
            try:
                completion = await self._stream(payload, on_text)
                completion.attempts = attempt
                return completion
            except LLMError as error:
                retryable = error.status in RETRYABLE_STATUS or error.status is None
                if error.partial or not retryable or attempt > self._max_retries:
                    raise
                # Exponential backoff with jitter, unless the server said how long to wait.
                backoff = min(8.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.25)
                if error.status == 429:
                    backoff = 20.0 * attempt  # a rate limit is counted per minute, so a short wait is wasted
                await self._sleep(error.retry_after if error.retry_after is not None else backoff)
                attempt += 1

    async def _stream(self, payload: dict, on_text: Callable[[str], None] | None) -> Completion:
        started = time.perf_counter()
        ttft_ms: float | None = None
        assembler = StreamAssembler(payload["model"])
        try:
            async with self._http.stream(
                "POST", self._url, json=payload, headers=self._headers
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")[:500]
                    raise LLMError(
                        f"model server returned {response.status_code}: {body}",
                        status=response.status_code,
                        retry_after=_retry_after(response),
                    )
                # Server-sent events: each event is a line "data: {json}". The last is "data: [DONE]".
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    new_text = assembler.feed(chunk)
                    if ttft_ms is None and assembler.started:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    if new_text and on_text:
                        on_text(new_text)
        except httpx.TransportError as error:
            raise LLMError(
                f"connection to the model server failed: {error!r}", partial=bool(assembler.text)
            ) from error
        return assembler.finish(ttft_ms=ttft_ms, latency_ms=(time.perf_counter() - started) * 1000)

    async def aclose(self) -> None:
        await self._http.aclose()
