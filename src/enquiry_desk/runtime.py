"""Builds the real desk from the settings: model client, knowledge base, CRM, slots, graph."""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from qdrant_client import QdrantClient

from enquiry_desk.agent.graph import build_graph
from enquiry_desk.agent.service import Desk
from enquiry_desk.agent.toolset import build_tools
from enquiry_desk.booking import IST, Slots
from enquiry_desk.config import Settings
from enquiry_desk.crm import CRM, LocalCRM
from enquiry_desk.crm_hubspot import HubSpotCRM
from enquiry_desk.db import Database
from enquiry_desk.intake import Intake, Notifier
from enquiry_desk.jobs import Jobs, Worker
from enquiry_desk.kb.embed import Embedder, FastEmbedEmbedder, HashEmbedder
from enquiry_desk.kb.store import KnowledgeBase
from enquiry_desk.llm.pricing import load_prices
from enquiry_desk.llm.wire import ChatClient
from enquiry_desk.metering import Meter
from enquiry_desk.observability import Telemetry, otlp_exporter, setup_logging
from enquiry_desk.tenant import Tenant, load_tenant
from enquiry_desk.webhooks.crm_events import CrmEvents

# Fields the lead intake adds on top of the tenant's own.
INTAKE_FIELDS = ["source", "email", "lead_score", "lead_band"]


@dataclass
class Runtime:
    desk: Desk
    meter: Meter
    kb: KnowledgeBase
    tenant: Tenant
    db: Database
    crm: CRM
    jobs: Jobs
    worker: Worker
    settings: Settings
    telemetry: Telemetry


def now_ist() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(settings.dense_model, cache_dir=str(settings.embed_cache_dir))


def make_knowledge_base(settings: Settings) -> KnowledgeBase:
    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    else:
        # Embedded mode: the same API with the data in a local folder. Good for development.
        # It is one process only and has no HNSW index, so it is not what "runs Qdrant" means.
        client = QdrantClient(path=str(settings.data_dir / "qdrant"))
    # A collection per embedder, because vectors from two embedders cannot be compared.
    return KnowledgeBase(client, make_embedder(settings), collection=f"knowledge_{settings.embedder}")


def make_crm(settings: Settings, tenant: Tenant, db: Database) -> CRM:
    if settings.crm == "hubspot":
        return HubSpotCRM(settings.hubspot_token, [*tenant.lead_fields, *INTAKE_FIELDS])
    return LocalCRM(db)


@asynccontextmanager
async def open_desk(settings: Settings):
    """Everything the API needs, closed again on exit."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tenant = load_tenant(settings.tenant_dir)
    setup_logging(as_json=settings.log_json)
    telemetry = Telemetry(otlp_exporter(settings.otlp_endpoint) if settings.otlp_endpoint else None)
    db = Database(settings.database_url or settings.data_dir / "desk.db")
    meter = Meter(db, load_prices(settings.prices_file), observer=telemetry)
    crm = make_crm(settings, tenant, db)
    slots = Slots(db)
    slots.seed(now_ist(), tenant.slot_hours, closed_weekdays=tuple(tenant.closed_weekdays))
    kb = make_knowledge_base(settings)
    chat = ChatClient(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        stream_usage=settings.llm_stream_usage,
        timeout=settings.llm_timeout_s,
    )
    jobs = Jobs(db)
    # A hot lead raises an alert only when there is somewhere to send it.
    notify_bands = ("hot",) if settings.notify_url else ()
    handlers = {"lead": Intake(tenant, chat, crm, meter, jobs, notify_bands).handle, "crm_event": CrmEvents(db).handle}
    on_handoff = None
    if settings.notify_url:
        handlers["notify"] = Notifier(settings.notify_url, settings.notify_secret).handle

        def on_handoff(info: dict) -> None:
            jobs.enqueue("notify", {"text": f"Handoff asked ({info['reason']}) in session {info['session_id']}: {info['summary']}"})

    async with AsyncExitStack() as stack:
        stack.push_async_callback(chat.aclose)
        stack.callback(db.close)
        stack.callback(telemetry.shutdown)
        if settings.database_url:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpointer = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(settings.database_url))
            await checkpointer.setup()  # creates the checkpoint tables on first use
        else:
            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(settings.data_dir / "checkpoints.db"))
            )
        sinks: dict = {}
        graph = build_graph(
            tenant=tenant,
            chat=chat,
            tools=build_tools(tenant, kb, crm, slots, now_ist, on_handoff=on_handoff),
            meter=meter,
            clock=now_ist,
            checkpointer=checkpointer,
            max_model_calls=settings.max_model_calls_per_turn,
            session_cost_cap_usd=settings.session_cost_cap_usd,
            search_first=settings.always_search_first,
            sinks=sinks,
        )
        desk = Desk(graph, meter, tenant, telemetry, sinks)
        yield Runtime(desk, meter, kb, tenant, db, crm, jobs, Worker(jobs, handlers), settings, telemetry)
