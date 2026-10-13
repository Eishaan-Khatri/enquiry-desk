"""Tools the model may call: a JSON schema goes out, validated arguments come in.

The model only ever writes text. A "tool call" is the model writing a function name and a JSON
string of arguments. Nothing guarantees that string is valid, so every call is parsed and
validated here, and a failure goes back to the model as the tool's result so it can correct
itself on the next step.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel, ValidationError

from enquiry_desk.llm.wire import Completion, ToolCall

log = logging.getLogger(__name__)


class ToolError(Exception):
    """An expected failure. Its message is safe to show the model, which can then recover."""


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str  # JSON text sent back to the model
    ok: bool
    error: str | None = None
    latency_ms: float = 0.0
    data: dict | None = None

    def as_message(self) -> dict:
        return {"role": "tool", "tool_call_id": self.call_id, "content": self.content}


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    handler: Callable[[Any, Any], Awaitable[dict]]  # (validated args, context) -> result

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _portable(self.args.model_json_schema()),
            },
        }


def _portable(schema: Any) -> Any:
    """Reduce a Pydantic schema to the subset every server accepts.

    Drops titles, defaults and additionalProperties, and turns "string or null" into "string".
    Servers differ in how much of JSON Schema they take, and a rejected schema fails the whole
    request. Nothing is lost by sending less: the arguments are validated here in any case.
    """
    if isinstance(schema, list):
        return [_portable(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    options = schema.get("anyOf")
    if options:
        real = [option for option in options if option.get("type") != "null"]
        if len(real) == 1:
            merged = {k: v for k, v in schema.items() if k != "anyOf"}
            return _portable({**merged, **real[0]})
    cleaned = {}
    for key, value in schema.items():
        if key in ("title", "default", "additionalProperties"):
            continue
        if key == "properties":
            cleaned[key] = {name: _portable(sub) for name, sub in value.items()}
        else:
            cleaned[key] = _portable(value)
    return cleaned


def recover_text_calls(completion: Completion, names: set[str]) -> Completion:
    """Turn a tool call that arrived as plain text back into a tool call.

    The server is meant to parse the model's output and hand over structured tool calls. When
    the model formats its call in a way the server's parser does not recognise, the call comes
    through as ordinary reply text, and the caller would be read a line of JSON. Two shapes
    were seen from a local model on 13 Oct 2026 and are handled here:

        search_knowledge {"query": "fees"}
        {"name": "search_knowledge", "arguments": {"query": "fees"}}

    The second may be wrapped in <tool_call> tags or a code fence. Text that is not exactly
    one call to a known tool is left alone.
    """
    if completion.tool_calls or not completion.content.strip():
        return completion
    text = completion.content.strip()
    text = re.sub(r"^<tool_call>|</tool_call>$", "", text).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    name, arguments = None, None
    bare = re.match(r"^(\w+)\s*(\{.*\})$", text, re.DOTALL)
    try:
        if bare and bare.group(1) in names:
            name, arguments = bare.group(1), json.loads(bare.group(2))
        elif text.startswith("{"):
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("name") in names:
                name, arguments = parsed["name"], parsed.get("arguments", parsed.get("parameters", {}))
    except json.JSONDecodeError:
        return completion
    if name is None or not isinstance(arguments, dict):
        return completion
    log.info("recovered a tool call to %s that the server passed through as text", name)
    call = ToolCall(id="text_0", name=name, arguments=json.dumps(arguments, ensure_ascii=False))
    return replace(completion, content="", tool_calls=[call], finish_reason="tool_calls")


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def names(self) -> set[str]:
        return set(self._tools)

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    async def execute(self, call: ToolCall, context: Any = None) -> ToolResult:
        started = time.perf_counter()

        def fail(kind: str, message: str, **extra: Any) -> ToolResult:
            body = {"error": kind, "message": message, **extra}
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=json.dumps(body, ensure_ascii=False),
                ok=False,
                error=kind,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        tool = self._tools.get(call.name)
        if tool is None:
            return fail("unknown_tool", f"No tool named {call.name!r}. Use one of: {', '.join(self._tools)}.")
        try:
            raw = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as error:
            return fail("invalid_json", f"The arguments were not valid JSON: {error.msg}.")
        if not isinstance(raw, dict):
            return fail("invalid_json", "The arguments must be a JSON object.")
        try:
            args = tool.args.model_validate(raw)
        except ValidationError as error:
            problems = [
                {"field": ".".join(str(part) for part in e["loc"]), "problem": e["msg"]}
                for e in error.errors()
            ]
            return fail("invalid_arguments", "Fix these fields and call the tool again.", fields=problems)
        try:
            data = await tool.handler(args, context)
        except ToolError as error:
            return fail("tool_error", str(error))
        except Exception:
            log.exception("tool %s crashed", call.name)
            return fail("internal_error", "The tool failed. Tell the caller you could not do this right now.")
        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=json.dumps(data, ensure_ascii=False),
            ok=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            data=data,
        )
