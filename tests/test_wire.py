"""The raw model client, against a fake server that streams the real wire format."""

from __future__ import annotations

import json

import httpx
import pytest

from enquiry_desk.llm.wire import ChatClient, LLMError


def sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join(lines + ["data: [DONE]\n\n"]).encode()


def delta(**fields) -> dict:
    finish = fields.pop("finish", None)
    return {"model": "m-1", "choices": [{"index": 0, "delta": fields, "finish_reason": finish}]}


def fragment(index: int, *, id: str | None = None, name: str | None = None, arguments: str = "") -> dict:
    piece: dict = {"index": index, "function": {"arguments": arguments}}
    if id:
        piece["id"] = id
        piece["type"] = "function"
    if name:
        piece["function"]["name"] = name
    return piece


def client_for(handler, **kwargs) -> ChatClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ChatClient("https://llm.test/v1", "secret", "m-1", http=http, **kwargs)


def stream(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


async def test_text_is_joined_and_usage_is_read():
    body = sse(
        delta(role="assistant", content=""),
        delta(content="Hello, "),
        delta(content="how can I help?"),
        delta(finish="stop"),
        {"model": "m-1", "choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 30}}},
    )
    heard = []
    done = await client_for(lambda request: stream(body)).complete([{"role": "user", "content": "hi"}], on_text=heard.append)

    assert done.content == "Hello, how can I help?"
    assert heard == ["Hello, ", "how can I help?"]
    assert done.tool_calls == [] and done.finish_reason == "stop"
    assert (done.usage.prompt_tokens, done.usage.completion_tokens, done.usage.cached_tokens) == (50, 7, 30)
    assert done.ttft_ms is not None and done.ttft_ms <= done.latency_ms


async def test_two_tool_calls_arrive_in_interleaved_fragments():
    body = sse(
        delta(tool_calls=[fragment(0, id="call_a", name="search_knowledge")]),
        delta(tool_calls=[fragment(1, id="call_b", name="list_slots")]),
        delta(tool_calls=[fragment(0, arguments='{"que')]),
        delta(tool_calls=[fragment(1, arguments="{}")]),
        delta(tool_calls=[fragment(0, arguments='ry": "fees"}')]),
        delta(finish="tool_calls"),
    )
    done = await client_for(lambda request: stream(body)).complete([{"role": "user", "content": "x"}])

    assert [(c.id, c.name, c.arguments) for c in done.tool_calls] == [
        ("call_a", "search_knowledge", '{"query": "fees"}'),
        ("call_b", "list_slots", "{}"),
    ]
    assert done.as_message()["tool_calls"][0]["function"]["arguments"] == '{"query": "fees"}'
    assert done.as_message()["content"] is None


async def test_whole_calls_that_share_index_zero_stay_separate():
    # Some servers send each parallel call complete, and number every one of them 0.
    body = sse(
        delta(tool_calls=[fragment(0, id="call_a", name="search_knowledge", arguments='{"query": "fees"}')]),
        delta(tool_calls=[fragment(0, id="call_b", name="list_slots", arguments="{}")]),
        delta(finish="tool_calls"),
    )
    done = await client_for(lambda request: stream(body)).complete([{"role": "user", "content": "x"}])

    assert [(c.id, c.arguments) for c in done.tool_calls] == [("call_a", '{"query": "fees"}'), ("call_b", "{}")]


async def test_request_carries_tools_key_and_usage_option():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return stream(sse(delta(content="ok", finish="stop")))

    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    await client_for(handler).complete([{"role": "user", "content": "x"}], tools=tools)

    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["stream"] is True and seen["body"]["stream_options"] == {"include_usage": True}
    assert seen["body"]["tools"] == tools and seen["body"]["tool_choice"] == "auto"


async def test_a_429_is_retried_after_the_wait_the_server_asked_for():
    answers = [
        httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "2"}),
        stream(sse(delta(content="ok", finish="stop"))),
    ]
    waits = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    done = await client_for(lambda request: answers.pop(0), sleep=sleep).complete([{"role": "user", "content": "x"}])

    assert done.content == "ok" and done.attempts == 2
    assert waits == [2.0]


async def test_a_400_is_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": "bad schema"})

    with pytest.raises(LLMError) as raised:
        await client_for(handler).complete([{"role": "user", "content": "x"}])
    assert raised.value.status == 400 and len(calls) == 1


async def test_gives_up_after_the_retry_limit():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="overloaded")

    async def sleep(seconds: float) -> None:
        pass

    with pytest.raises(LLMError):
        await client_for(handler, sleep=sleep, max_retries=2).complete([{"role": "user", "content": "x"}])
    assert len(calls) == 3


async def test_a_dropped_connection_is_retried():
    answers: list = [httpx.ConnectError("refused"), stream(sse(delta(content="ok", finish="stop")))]

    def handler(request: httpx.Request) -> httpx.Response:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def sleep(seconds: float) -> None:
        pass

    done = await client_for(handler, sleep=sleep).complete([{"role": "user", "content": "x"}])
    assert done.content == "ok" and done.attempts == 2


async def test_what_the_server_attaches_to_a_tool_call_is_sent_back_untouched():
    # Gemini signs each tool call and refuses the next request if the signature is missing.
    signed = fragment(0, id="call_a", name="list_slots", arguments="{}") | {"extra_content": {"google": {"thought_signature": "abc123"}}}
    done = await client_for(lambda request: stream(sse(delta(tool_calls=[signed]), delta(finish="tool_calls")))).complete([{"role": "user", "content": "x"}])

    message = done.as_message()
    assert message["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "abc123"}}
    from enquiry_desk.llm.wire import tool_calls_of
    assert tool_calls_of(message)[0].extra == {"google": {"thought_signature": "abc123"}}
