"""One caller turn as a LangGraph graph.

            +--------------------------------------+
            v                                      |
    START -> call_model --tool calls--> run_tools -+--handoff asked--> handoff -> END
                |    ^                                                 (pauses for a person)
              text   +------- retry once -------+
                v                               |
            check_grounding --------------------+
                | ok, or second failure
                v
             finalize -> END

With `search_first` on, START goes to a search_first node, which runs the search in code and
then enters call_model.

LangGraph is used for three things here: the loop itself, a checkpoint after every node so a
conversation survives across requests and restarts, and `interrupt`, which pauses the graph
at the handoff node until a person answers.

The model call inside `call_model` is the plain HTTP client in llm/wire.py. No LangChain model
wrapper is used, so messages are the same dicts that go over the wire.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from enquiry_desk.agent import grounding, prompts
from enquiry_desk.agent.toolset import ToolContext
from enquiry_desk.llm.tools import ToolRegistry, recover_text_calls
from enquiry_desk.llm.wire import ChatClient, LLMError, ToolCall, tool_calls_of
from enquiry_desk.metering import Meter
from enquiry_desk.tenant import Tenant


def _append(existing: list, new: list) -> list:
    return existing + new


class DeskState(TypedDict, total=False):
    # Kept for the whole conversation. A node returns only new items and LangGraph appends them.
    messages: Annotated[list[dict], _append]  # the history, in wire format
    evidence: Annotated[list[str], _append]  # caller text and tool results, for the grounding check
    refs: dict[str, dict]  # "K1" -> {"chunk_id", "source"}
    lead_id: str | None
    # Reset at the start of every turn.
    session_id: str
    turn: int
    model_calls: int
    retries: int
    draft: str | None  # a reply that has not passed the check yet
    rejected: dict | None  # {"draft", "note", "at"}: a failed draft, shown to the model once
    reply: str | None
    problems: list[str]
    gate_problems: list[str]  # voice only: what the sentence gate objected to in this turn
    handoff: dict | None
    booked: bool  # book_slot has succeeded at some point in this conversation


def new_turn(session_id: str, turn: int, text: str) -> dict:
    return {
        "messages": [{"role": "user", "content": text}],
        "evidence": [text],
        "session_id": session_id,
        "turn": turn,
        "model_calls": 0,
        "retries": 0,
        "draft": None,
        "rejected": None,
        "reply": None,
        "problems": [],
        "gate_problems": [],
        "handoff": None,
    }


def build_graph(
    *,
    tenant: Tenant,
    chat: ChatClient,
    tools: ToolRegistry,
    meter: Meter,
    clock: Callable[[], datetime],
    checkpointer,
    max_model_calls: int = 6,
    session_cost_cap_usd: float | None = None,
    search_first: bool = False,
    sinks: dict[str, Callable[[str], None]] | None = None,
):
    # `sinks` maps a session id to a function that speaks one sentence. A session is in it only
    # while a voice turn is running. Chat turns never are, and take the whole-reply path.
    sinks = sinks if sinks is not None else {}

    def entry(state: DeskState) -> str:
        # Three words is a rough line between a question and "yes" or "ok thanks".
        wordy = len(state["messages"][-1]["content"].split()) >= 3
        return "search" if search_first and wordy else "model"

    async def search_first_node(state: DeskState) -> dict:
        """Search with the caller's own words before the model is asked anything.

        Left to itself, a small model often answers without searching. With this on, the
        decision to search is made in code. The search is written into the history as a tool
        call the model appears to have made, so everything downstream treats it the same way.
        The price is about four chunks of extra prompt on every turn, needed or not.
        """
        call = ToolCall(
            id=f"auto_{state['turn']}",
            name="search_knowledge",
            arguments=json.dumps({"query": state["messages"][-1]["content"]}, ensure_ascii=False),
        )
        context = ToolContext(session_id=state["session_id"], refs=dict(state.get("refs") or {}))
        result = await tools.execute(call, context)
        meter.record_tool(state["session_id"], state["turn"], result)
        asked = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}],
        }
        return {
            "messages": [asked, result.as_message()],
            "evidence": [result.content] if result.ok else [],
            "refs": context.refs,
        }

    async def call_model(state: DeskState) -> dict:
        if state["model_calls"] >= max_model_calls:
            return {"draft": prompts.TOO_MANY_STEPS, "rejected": None}
        if session_cost_cap_usd is not None:
            spent = meter.summary(state["session_id"])["cost_usd"]
            if spent is not None and spent >= session_cost_cap_usd:
                return {"draft": prompts.COST_CAP, "rejected": None}

        messages = [{"role": "system", "content": prompts.system_prompt(tenant, clock())}]
        history = state["messages"]
        rejected = state.get("rejected")
        if rejected:
            # The failed draft and the checker's note go where the draft was written, ahead of
            # any tool calls made since. They are never saved to the history, because the
            # caller did not see that draft.
            at = rejected["at"]
            messages += history[:at]
            messages += [
                {"role": "assistant", "content": rejected["draft"]},
                {"role": "user", "content": rejected["note"]},
            ]
            messages += history[at:]
        else:
            messages += history

        gate = None
        speak = sinks.get(state["session_id"])
        if speak is not None:
            evidence = state["evidence"] + [prompts.date_line(clock())]
            known_refs = set((state.get("refs") or {}).keys())
            booked = bool(state.get("booked"))
            gate = grounding.SentenceGate(lambda s: grounding.check(s, evidence, known_refs, booked=booked), speak)

        started = time.perf_counter()
        try:
            completion = await chat.complete(messages, tools=tools.specs(), on_text=gate.feed if gate else None)
        except LLMError as error:
            meter.record_llm_failure(
                state["session_id"], state["turn"], chat.model, str(error), (time.perf_counter() - started) * 1000
            )
            return {"draft": prompts.MODEL_FAILED, "rejected": None, "model_calls": max_model_calls}
        meter.record_llm(state["session_id"], state["turn"], completion)
        recovered = recover_text_calls(completion, tools.names())
        if gate and recovered is not completion:
            gate.discard()  # the "text" was a tool call, so it must not be spoken
        completion = recovered

        update: dict = {"model_calls": state["model_calls"] + 1}
        if completion.tool_calls:
            update["messages"] = [completion.as_message()]
        else:
            update["draft"] = completion.content
            if gate:
                gate.finish()
                # What was spoken is the reply. A sentence that failed the check ended the
                # stream, and the fixed line is spoken in its place. There is no second attempt
                # in voice: the caller has already heard the beginning.
                spoken = " ".join(gate.passed)
                if gate.problems:
                    meter.record_check(state["session_id"], state["turn"], gate.problems, "refused")
                    speak(prompts.NOT_GROUNDED)
                    spoken = (spoken + " " + prompts.NOT_GROUNDED).strip()
                    update["gate_problems"] = gate.problems
                update["draft"] = spoken or completion.content
        return update

    def after_model(state: DeskState) -> str:
        return "check" if state.get("draft") is not None else "tools"

    async def run_tools(state: DeskState) -> dict:
        context = ToolContext(
            session_id=state["session_id"],
            refs=dict(state.get("refs") or {}),
            lead_id=state.get("lead_id"),
            caller_digits=" ".join(re.sub(r"\D", "", m["content"] or "") for m in state["messages"] if m["role"] == "user"),
        )
        results = []
        # In the order the model listed them, one at a time. A booking in the same step as
        # save_lead depends on the lead existing first.
        for call in tool_calls_of(state["messages"][-1]):
            result = await tools.execute(call, context)
            meter.record_tool(state["session_id"], state["turn"], result)
            results.append(result)
        return {
            "messages": [result.as_message() for result in results],
            "evidence": [result.content for result in results if result.ok],
            "refs": context.refs,
            "lead_id": context.lead_id,
            "handoff": context.handoff or state.get("handoff"),
            "booked": bool(state.get("booked")) or any(r.ok and r.name == "book_slot" for r in results),
        }

    def after_tools(state: DeskState) -> str:
        return "handoff" if state.get("handoff") else "model"

    def check_grounding(state: DeskState) -> dict:
        evidence = state["evidence"] + [prompts.date_line(clock())]
        problems = grounding.check(
            state["draft"], evidence, set((state.get("refs") or {}).keys()), booked=bool(state.get("booked"))
        )
        if not problems:
            return {"problems": state.get("gate_problems") or [], "rejected": None}
        if state["retries"] < 1:
            meter.record_check(state["session_id"], state["turn"], problems, "retried")
            return {
                "problems": problems,
                "retries": state["retries"] + 1,
                "rejected": {
                    "draft": state["draft"],
                    "note": grounding.feedback(problems),
                    "at": len(state["messages"]),
                },
                "draft": None,
            }
        # Failed twice. The caller gets a fixed line with no figures in it.
        meter.record_check(state["session_id"], state["turn"], problems, "refused")
        return {"problems": problems, "rejected": None, "draft": prompts.NOT_GROUNDED}

    def after_check(state: DeskState) -> str:
        return "retry" if state.get("rejected") else "final"

    def finalize(state: DeskState) -> dict:
        text = state["draft"]
        # A caller must be told they are talking to a machine. The instructions asked the model
        # to say so and a small model did not, so the first reply now gets the line in code.
        # A voice call opens with a spoken greeting that already says this.
        if state["turn"] == 1 and "AI assistant" not in text and state["session_id"] not in sinks:
            text = prompts.disclosure(tenant) + text
        return {"messages": [{"role": "assistant", "content": text}], "reply": text}

    def handoff(state: DeskState) -> dict:
        # interrupt() stops the graph here and saves it. The run that resumes it starts this
        # node again, and interrupt() then returns what the person typed.
        answer = interrupt({"session_id": state["session_id"], **state["handoff"]})
        return {
            "messages": [{"role": "assistant", "content": answer}],
            "reply": answer,
            "handoff": {**state["handoff"], "resolved": True},
        }

    graph = StateGraph(DeskState)
    graph.add_node("call_model", call_model)
    graph.add_node("run_tools", run_tools)
    graph.add_node("check_grounding", check_grounding)
    graph.add_node("finalize", finalize)
    graph.add_node("handoff", handoff)
    graph.add_node("search_first", search_first_node)
    graph.add_conditional_edges(START, entry, {"search": "search_first", "model": "call_model"})
    graph.add_edge("search_first", "call_model")
    graph.add_conditional_edges("call_model", after_model, {"tools": "run_tools", "check": "check_grounding"})
    graph.add_conditional_edges("run_tools", after_tools, {"handoff": "handoff", "model": "call_model"})
    graph.add_conditional_edges("check_grounding", after_check, {"retry": "call_model", "final": "finalize"})
    graph.add_edge("finalize", END)
    graph.add_edge("handoff", END)
    return graph.compile(checkpointer=checkpointer)
