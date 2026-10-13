"""Shared fixtures. No test touches the network: the model is scripted and Qdrant runs in memory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from enquiry_desk.agent.graph import build_graph
from enquiry_desk.agent.service import Desk
from enquiry_desk.agent.toolset import build_tools
from enquiry_desk.booking import Slots
from enquiry_desk.config import Settings
from enquiry_desk.crm import LocalCRM
from enquiry_desk.db import Database
from enquiry_desk.intake import Intake
from enquiry_desk.jobs import Jobs, Worker
from enquiry_desk.kb.chunking import chunk_markdown
from enquiry_desk.kb.embed import HashEmbedder
from enquiry_desk.kb.store import KnowledgeBase
from enquiry_desk.llm.wire import Completion, ToolCall, Usage
from enquiry_desk.metering import Meter
from enquiry_desk.observability import Telemetry
from enquiry_desk.tenant import knowledge_files, load_tenant
from enquiry_desk.webhooks.crm_events import CrmEvents

TENANT_DIR = Path(__file__).resolve().parents[1] / "data" / "tenants" / "demo_coaching"
NOW = datetime(2026, 10, 13, 10, 0)  # a Tuesday


def say(text: str) -> Completion:
    return Completion(text, [], "stop", Usage(100, 20, 0), "scripted", ttft_ms=5.0, latency_ms=10.0)


def call(*calls: tuple[str, dict]) -> Completion:
    """A reply that calls tools: call(("save_lead", {"phone": "..."}), ...)."""
    tool_calls = [ToolCall(f"c{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls)]
    return Completion("", tool_calls, "tool_calls", Usage(100, 15, 0), "scripted", ttft_ms=5.0, latency_ms=10.0)


def last_tool_result(messages: list[dict]) -> dict:
    return json.loads(next(m["content"] for m in reversed(messages) if m["role"] == "tool"))


class ScriptedChat:
    """Stands in for ChatClient. Replies come from a list, and every request is kept."""

    model = "scripted"

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    async def complete(self, messages, tools=None, on_text=None, **_):
        self.requests.append(messages)
        reply = self.replies.pop(0)
        if callable(reply):  # a reply that depends on what the tools returned
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        if on_text and reply.content:  # stream the text a few characters at a time, as a server does
            for start in range(0, len(reply.content), 7):
                on_text(reply.content[start : start + 7])
        return reply


@pytest.fixture(scope="session")
def tenant():
    return load_tenant(TENANT_DIR)


@pytest.fixture
def kb(tenant):
    base = KnowledgeBase(QdrantClient(":memory:"), HashEmbedder())
    chunks = []
    for path in knowledge_files(TENANT_DIR):
        chunks += chunk_markdown(tenant.id, path.name, path.read_text(encoding="utf-8"))
    base.replace(tenant.id, chunks)
    return base


@pytest.fixture
def world(tenant, kb):
    """A desk with a scripted model. Set world.chat.replies in the test."""
    return make_world(tenant, kb)


def make_world(tenant, kb, **graph_options):
    db = Database()
    spans = InMemorySpanExporter()
    telemetry = Telemetry(spans, batch=False)
    meter = Meter(db, observer=telemetry)
    crm = LocalCRM(db)
    slots = Slots(db)
    slots.seed(NOW, tenant.slot_hours)
    chat = ScriptedChat([])
    sinks: dict = {}
    graph = build_graph(
        tenant=tenant,
        chat=chat,
        tools=build_tools(tenant, kb, crm, slots, lambda: NOW),
        meter=meter,
        clock=lambda: NOW,
        checkpointer=InMemorySaver(),
        max_model_calls=4,
        sinks=sinks,
        **graph_options,
    )
    jobs = Jobs(db)
    intake_chat = ScriptedChat([])
    sent: list[dict] = []

    async def notify(payload: dict) -> None:
        sent.append(payload)

    worker = Worker(jobs, {"lead": Intake(tenant, intake_chat, crm, meter, jobs).handle, "notify": notify, "crm_event": CrmEvents(db).handle})
    settings = Settings(_env_file=None, webhook_secret="form-secret", hubspot_client_secret="hub-secret", public_base_url="https://desk.test")
    return SimpleNamespace(
        desk=Desk(graph, meter, tenant, telemetry, sinks), telemetry=telemetry, spans=spans, chat=chat, crm=crm, slots=slots, meter=meter, graph=graph, tenant=tenant,
        db=db, jobs=jobs, worker=worker, settings=settings, intake_chat=intake_chat, sent=sent,
    )
