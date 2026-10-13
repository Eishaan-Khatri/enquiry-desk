"""The desk as the outside world uses it: a caller says something, the desk replies."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from langgraph.types import Command

from enquiry_desk.agent import prompts
from enquiry_desk.agent.graph import new_turn
from enquiry_desk.agent.grounding import REF
from enquiry_desk.metering import Meter
from enquiry_desk.tenant import Tenant


@dataclass
class TurnResult:
    session_id: str
    turn: int
    reply: str
    status: str  # "ok", or "awaiting_operator" while a person has the conversation
    citations: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # what the grounding check objected to
    usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def outcome_of(result: TurnResult) -> str:
    """How a turn ended, as one of a few fixed words, for the metrics."""
    if result.status == "awaiting_operator":
        return "handoff"
    if result.reply.endswith(prompts.NOT_GROUNDED):
        return "refused"
    if result.reply.endswith((prompts.MODEL_FAILED, prompts.TOO_MANY_STEPS, prompts.COST_CAP)):
        return "failed"
    return "ok"


class NotWaiting(Exception):
    """An operator replied to a session that is not waiting for one."""


class Desk:
    def __init__(self, graph, meter: Meter, tenant: Tenant, telemetry=None, sinks: dict | None = None):
        self._graph = graph
        self._meter = meter
        self._tenant = tenant
        self._telemetry = telemetry
        self._sinks = sinks if sinks is not None else {}

    @property
    def tenant(self) -> Tenant:
        return self._tenant

    async def handle(self, session_id: str, text: str, on_sentence=None) -> TurnResult:
        """One caller turn. With `on_sentence`, each sentence of the reply is handed over as
        soon as it is written and checked, which is what a voice needs."""
        if on_sentence is None:
            return await self._timed(session_id, text)
        self._sinks[session_id] = on_sentence
        try:
            return await self._timed(session_id, text)
        finally:
            self._sinks.pop(session_id, None)

    async def _timed(self, session_id: str, text: str) -> TurnResult:
        if self._telemetry is None:
            return await self._handle(session_id, text)
        started = time.perf_counter()
        with self._telemetry.turn(session_id, 0) as span:
            result = await self._handle(session_id, text)
            span.set_attribute("desk.turn", result.turn)
            self._telemetry.turn_done(outcome_of(result), time.perf_counter() - started)
            return result

    async def _handle(self, session_id: str, text: str) -> TurnResult:
        # The session id is the LangGraph thread id. The checkpointer loads the conversation
        # saved under it, so this process holds no session state of its own.
        config = {"configurable": {"thread_id": session_id}}
        snapshot = await self._graph.aget_state(config)
        turn = snapshot.values.get("turn") or 0
        if snapshot.next:
            # Paused at the handoff node. The caller waits for the person, and the model stays out.
            return TurnResult(session_id, turn, self._tenant.handoff_line, "awaiting_operator")
        turn += 1
        state = await self._graph.ainvoke(new_turn(session_id, turn, text), config)
        if state.get("__interrupt__"):
            return self._result(session_id, turn, self._tenant.handoff_line, "awaiting_operator", state)
        return self._result(session_id, turn, state["reply"], "ok", state)

    async def operator_reply(self, session_id: str, text: str) -> TurnResult:
        config = {"configurable": {"thread_id": session_id}}
        snapshot = await self._graph.aget_state(config)
        if not snapshot.next:
            raise NotWaiting(session_id)
        state = await self._graph.ainvoke(Command(resume=text), config)
        return self._result(session_id, state["turn"], state["reply"], "ok", state)

    async def evidence(self, session_id: str) -> list[str]:
        """Everything the desk was allowed to rely on in this conversation: what the caller
        said and what the tools returned. The evaluation's judge reads replies against it."""
        snapshot = await self._graph.aget_state({"configurable": {"thread_id": session_id}})
        return list(snapshot.values.get("evidence") or [])

    def _result(self, session_id: str, turn: int, reply: str, status: str, state: dict) -> TurnResult:
        refs = state.get("refs") or {}
        cited = [ref for ref in dict.fromkeys(REF.findall(reply)) if ref in refs]
        return TurnResult(
            session_id=session_id,
            turn=turn,
            reply=reply,
            status=status,
            citations=[{"ref": ref, "source": refs[ref]["source"]} for ref in cited],
            problems=state.get("problems") or [],
            usage=self._meter.summary(session_id, turn),
        )
