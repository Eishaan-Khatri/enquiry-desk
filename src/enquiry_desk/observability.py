"""Metrics, traces and logs for the running desk.

Three views of the same events:
  metrics  counters and histograms that Prometheus scrapes from /metrics. They answer "how is
           the desk doing overall, right now?" and they drive the alerts.
  traces   one span per turn, with a child span for each model call and tool call. They
           answer "what happened inside this one slow turn?"
  logs     one line per notable event, with phone numbers masked.

No span and no metric label carries what the caller said. A trace is something engineers
browse freely, and a caller's words and phone number do not belong there.
"""

from __future__ import annotations

import json
import logging
import re
import time

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)

from enquiry_desk.llm.tools import ToolResult
from enquiry_desk.llm.wire import Completion

PHONE = re.compile(r"(?<!\d)(\+?91[\s-]?|0)?([6-9]\d)\d{3}[\s-]?\d{3}(\d{2})(?!\d)")
SECONDS = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)

# Without this every counter also exports a "_created" timestamp series, which doubles the
# number of series for no use here.
disable_created_metrics()


def mask_phones(text: str) -> str:
    """9876543210 -> 98******10. Enough is left to match a complaint to a log line."""
    return PHONE.sub(lambda m: f"{m.group(2)}******{m.group(3)}", text)


class MaskPhones(logging.Filter):
    """A logging filter, so that no code path can log a full phone number by accident."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_phones(record.getMessage())
        record.args = None
        return True


class JsonLines(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, ensure_ascii=False)


def setup_logging(*, as_json: bool) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(MaskPhones())
    if as_json:
        handler.setFormatter(JsonLines())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def rejection_reason(problem: str) -> str:
    """One short label per kind of problem, because a metric label must have few values."""
    for start, reason in (("the figure", "figure"), ("the reference", "reference"), ("the phone", "phone"), ("you told", "booking")):
        if problem.startswith(start):
            return reason
    return "other"


class Telemetry:
    """Everything the desk reports about itself. One instance per process."""

    def __init__(self, exporter: SpanExporter | None = None, service_name: str = "enquiry-desk", batch: bool = True):
        self.registry = CollectorRegistry()
        r = self.registry
        self.turns = Counter("desk_turns_total", "Caller turns answered, by how they ended.", ["outcome"], registry=r)
        self.turn_seconds = Histogram("desk_turn_seconds", "Time to answer one caller turn.", buckets=SECONDS, registry=r)
        self.first_token_seconds = Histogram("desk_first_token_seconds", "Time to the first token of a model call.", buckets=SECONDS, registry=r)
        self.llm_calls = Counter("desk_llm_calls_total", "Model calls.", ["model", "outcome"], registry=r)
        self.llm_tokens = Counter("desk_llm_tokens_total", "Tokens sent to and received from the model.", ["model", "kind"], registry=r)
        self.llm_cost = Counter("desk_llm_cost_usd_total", "Model spend in US dollars, where the price is known.", ["model"], registry=r)
        self.tool_calls = Counter("desk_tool_calls_total", "Tool calls.", ["tool", "outcome"], registry=r)
        self.rejections = Counter("desk_reply_rejections_total", "Draft replies the check sent back.", ["reason"], registry=r)
        self.webhooks = Counter("desk_webhooks_total", "Webhooks received.", ["source", "outcome"], registry=r)
        self.jobs = Gauge("desk_jobs", "Jobs in the queue, by status.", ["status"], registry=r)

        # Without an exporter the spans are created and dropped, which costs almost nothing.
        self._provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if exporter is not None:
            # Batching sends spans in the background, off the request path. Tests turn it off
            # so that a span can be read back as soon as it ends.
            self._provider.add_span_processor(BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter))
        self.tracer = self._provider.get_tracer("enquiry_desk")

    # ---- called by the desk -----------------------------------------------------------------

    def turn(self, session_id: str, turn: int):
        """A span around one whole turn. Model and tool spans started inside become its children."""
        return self.tracer.start_as_current_span("turn", attributes={"session.id": session_id, "desk.turn": turn})

    def turn_done(self, outcome: str, seconds: float) -> None:
        self.turns.labels(outcome).inc()
        self.turn_seconds.observe(seconds)
        trace.get_current_span().set_attribute("desk.outcome", outcome)

    def llm(self, completion: Completion, cost: float | None) -> None:
        model = completion.model
        self.llm_calls.labels(model, "ok").inc()
        if completion.usage:
            fresh = completion.usage.prompt_tokens - completion.usage.cached_tokens
            self.llm_tokens.labels(model, "input").inc(fresh)
            self.llm_tokens.labels(model, "cached_input").inc(completion.usage.cached_tokens)
            self.llm_tokens.labels(model, "output").inc(completion.usage.completion_tokens)
        if cost:
            self.llm_cost.labels(model).inc(cost)
        if completion.ttft_ms is not None:
            self.first_token_seconds.observe(completion.ttft_ms / 1000)
        # Attribute names follow OpenTelemetry's conventions for generative AI, so any tracing
        # product that understands them shows token counts without extra configuration.
        attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.response.finish_reasons": [completion.finish_reason],
            "desk.tool_calls": [call.name for call in completion.tool_calls],
            "desk.attempts": completion.attempts,
        }
        if completion.usage:
            attributes["gen_ai.usage.input_tokens"] = completion.usage.prompt_tokens
            attributes["gen_ai.usage.output_tokens"] = completion.usage.completion_tokens
        if cost is not None:
            attributes["desk.cost_usd"] = cost
        self._span(f"chat {model}", completion.latency_ms, attributes)

    def llm_failed(self, model: str, error: str, latency_ms: float) -> None:
        self.llm_calls.labels(model, "error").inc()
        self._span(f"chat {model}", latency_ms, {"gen_ai.operation.name": "chat", "gen_ai.request.model": model}, error=mask_phones(error)[:200])

    def tool(self, result: ToolResult) -> None:
        self.tool_calls.labels(result.name, result.error or "ok").inc()
        self._span(
            f"execute_tool {result.name}",
            result.latency_ms,
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": result.name},
            error=result.error,
        )

    def rejected(self, problems: list[str]) -> None:
        for reason in {rejection_reason(problem) for problem in problems}:
            self.rejections.labels(reason).inc()
        trace.get_current_span().add_event("reply_rejected", {"reasons": sorted({rejection_reason(p) for p in problems})})

    def webhook(self, source: str, outcome: str) -> None:
        self.webhooks.labels(source, outcome).inc()

    # ---- called by the API ------------------------------------------------------------------

    def render(self, job_counts: dict[str, int] | None = None) -> bytes:
        """The text Prometheus scrapes. The queue gauge is read fresh at each scrape."""
        for status, count in (job_counts or {}).items():
            self.jobs.labels(status).set(count)
        return generate_latest(self.registry)

    def shutdown(self) -> None:
        self._provider.shutdown()

    def _span(self, name: str, latency_ms: float, attributes: dict, error: str | None = None) -> None:
        # The call has already finished when it is reported, so the span is given the start
        # and end times it really had, rather than "now".
        end = time.time_ns()
        span = self.tracer.start_span(name, start_time=end - int(latency_ms * 1_000_000), attributes=attributes)
        if error:
            span.set_status(trace.Status(trace.StatusCode.ERROR, error))
        span.end(end_time=end)


def otlp_exporter(endpoint: str) -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint)
