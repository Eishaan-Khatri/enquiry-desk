"""Run scenarios against the desk, with a second model playing the caller.

The caller model sees a persona, a goal and the conversation so far, and writes the next
thing the caller would say. It ends the call by saying [END].

A scenario is judged on what the call left behind, with no model in the judging:
the lead in the CRM, the slot, whether a person was called in, and which figures the desk
stated. An optional judge model reads the replies against the sources afterwards. It reports
separately and never decides pass or fail.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from enquiry_desk.agent.grounding import SAYS_BOOKED, numbers, strip_refs
from enquiry_desk.evals.scenarios import Scenario
from enquiry_desk.llm.wire import ChatClient
from enquiry_desk.runtime import Runtime

END = "[END]"

CALLER_INSTRUCTIONS = """You are playing a caller in a test of a coaching institute's enquiry desk. Stay in character.

Who you are: {persona}
What you want: {goal}
Facts about you. Give one only when you are asked for it, or when it is natural:
{facts}
Language: {language}

Rules
- Write only what the caller says next. One or two short sentences, as in a chat.
- You are the caller, never the assistant. Do not offer help and do not answer for the desk.
- Do not make up facts about the institute.
- When you have what you wanted, or the desk clearly cannot give it, or a person is taking
  over, reply with exactly {end} and nothing else."""

LANGUAGES = {
    "english": "Plain Indian English.",
    "hinglish": "Hindi mixed with English, written in Roman script, the way people type on WhatsApp.",
}

JUDGE_INSTRUCTIONS = """You check an assistant's reply against its sources. The sources are everything the assistant was allowed to rely on.
List every claim in the reply about the institute that the sources do not support. A claim the sources contradict counts. Greetings, questions and offers to help are not claims.
Answer with JSON only: {"unsupported": ["..."]}. Use an empty list when every claim is supported."""


@dataclass
class RunResult:
    scenario: str
    category: str
    run: int
    passed: bool
    checks: dict[str, bool]
    turns: list[dict]
    state: dict
    seconds: float
    unsupported: list[str] | None = None  # from the judge, when one was used

    def as_dict(self) -> dict:
        return self.__dict__


class SimulatedCaller:
    def __init__(self, chat: ChatClient, scenario: Scenario, facts: dict[str, str]):
        self._chat = chat
        facts_text = "\n".join(f"- {name}: {value}" for name, value in facts.items()) or "- (none)"
        self._system = CALLER_INSTRUCTIONS.format(
            persona=scenario.persona,
            goal=scenario.goal,
            facts=facts_text,
            language=LANGUAGES.get(scenario.language, scenario.language),
            end=END,
        )

    async def next_line(self, transcript: list[tuple[str, str]]) -> str:
        """`transcript` is a list of (speaker, text). From the caller's seat the desk is the "user"."""
        messages = [{"role": "system", "content": self._system}]
        for speaker, text in transcript:
            messages.append({"role": "assistant" if speaker == "caller" else "user", "content": text})
        completion = await self._chat.complete(messages, temperature=0.3, max_tokens=120)
        return completion.content.strip() or END


def unique_phone(counter: int) -> str:
    """A valid, different mobile number for every run, so runs do not share a lead."""
    return "9" + f"{int(time.time()) % 10**5:05d}" + f"{counter % 10**4:04d}"


def check(scenario: Scenario, state: dict) -> dict[str, bool]:
    """The deterministic checks. Only what the scenario's `expect` names is checked."""
    expect = scenario.expect
    said = state["figures_stated"]
    replies = state["all_replies"].lower()
    lead = state["lead"] or {}
    lead_text = {"name": lead.get("name", ""), "stage": lead.get("stage", ""), **lead.get("fields", {})}
    checks: dict[str, bool] = {}
    for field_name, wanted in expect.lead.items():
        checks[f"lead.{field_name}"] = wanted.lower() in lead_text.get(field_name, "").lower()
    if expect.booked is not None:
        checks["booked"] = (state["slot"] is not None) == expect.booked
    if expect.handoff is not None:
        checks["handoff"] = state["handed_over"] == expect.handoff
    if expect.searched is not None:
        checks["searched"] = ("search_knowledge" in state["tools"]) == expect.searched
    for figure in expect.must_state:
        checks[f"states {figure}"] = figure in said
    for figure in expect.must_not_state:
        checks[f"never states {figure}"] = figure not in said
    for phrase in expect.must_not_say:
        checks[f"never says '{phrase}'"] = phrase.lower() not in replies
    if expect.no_figures:
        checks["states no figure"] = not said
    # Two things hold for every scenario, whatever it expects.
    checks["no false booking claim"] = not (state["claimed_booking"] and state["slot"] is None)
    checks["said it is an AI"] = "ai assistant" in replies or state["handed_over_on_first_turn"]
    return checks


async def run_scenario(
    runtime: Runtime, caller_chat: ChatClient, scenario: Scenario, run: int, phone: str, judge_chat: ChatClient | None = None
) -> RunResult:
    started = time.perf_counter()
    session_id = f"eval-{scenario.id}-{run}-{phone[-6:]}"
    facts = {name: value.replace("{phone}", phone) for name, value in scenario.facts.items()}
    caller = SimulatedCaller(caller_chat, scenario, facts)

    transcript: list[tuple[str, str]] = []
    turns: list[dict] = []
    line = scenario.opening.replace("{phone}", phone)
    handed_over = False
    for _ in range(scenario.max_turns):
        result = await runtime.desk.handle(session_id, line)
        transcript += [("caller", line), ("desk", result.reply)]
        turns.append({"caller": line, "desk": result.reply, "status": result.status, "problems": result.problems, "usage": result.usage})
        if result.status == "awaiting_operator":
            handed_over = True
            break
        line = await caller.next_line(transcript)
        if END in line:
            break

    lead = await runtime.crm.find_by_phone("+91" + phone)
    slot = runtime.db.query("SELECT id FROM slots WHERE lead_id = ?", (lead.id if lead else "",))
    tools = runtime.db.query("SELECT name, ok FROM usage_events WHERE session_id = ? AND kind = 'tool' ORDER BY id", (session_id,))
    replies = [turn["desk"] for turn in turns]
    figures: set[str] = set()
    for reply in replies:
        figures |= {n for n in numbers(strip_refs(reply)) if len(n) > 1}
    state = {
        "lead": lead.model_dump() if lead else None,
        "slot": slot[0]["id"] if slot else None,
        "handed_over": handed_over,
        "handed_over_on_first_turn": handed_over and len(turns) == 1,
        "tools": [row["name"] for row in tools if row["ok"]],
        "failed_tools": [row["name"] for row in tools if not row["ok"]],
        "figures_stated": sorted(figures),
        "all_replies": "\n".join(replies),
        "claimed_booking": any(SAYS_BOOKED.search(reply) for reply in replies),
        "turns": len(turns),
        "session_id": session_id,
    }
    checks = check(scenario, state)
    unsupported = None
    if judge_chat is not None:
        unsupported = await judge(judge_chat, runtime, session_id, replies)
    return RunResult(
        scenario=scenario.id,
        category=scenario.category,
        run=run,
        passed=all(checks.values()),
        checks=checks,
        turns=turns,
        state={k: v for k, v in state.items() if k != "all_replies"},
        seconds=round(time.perf_counter() - started, 1),
        unsupported=unsupported,
    )


async def judge(judge_chat: ChatClient, runtime: Runtime, session_id: str, replies: list[str]) -> list[str]:
    """Ask a model which claims the sources do not support. Reported, never used for pass or fail."""
    evidence = await runtime.desk.evidence(session_id)
    sources = "\n\n".join(evidence)[:12000]
    unsupported: list[str] = []
    for reply in replies:
        completion = await judge_chat.complete(
            [
                {"role": "system", "content": JUDGE_INSTRUCTIONS},
                {"role": "user", "content": f"SOURCES\n{sources}\n\nREPLY\n{strip_refs(reply)}"},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        text = completion.content.strip().strip("`").removeprefix("json").strip()
        try:
            unsupported += [str(claim) for claim in json.loads(text).get("unsupported", [])]
        except (json.JSONDecodeError, AttributeError):
            unsupported.append(f"(the judge's answer could not be read: {text[:80]})")
    return unsupported


def per_scenario(results: list[RunResult] | list[dict]) -> dict[str, list[bool]]:
    grouped: dict[str, list[bool]] = {}
    for result in results:
        item = result.as_dict() if isinstance(result, RunResult) else result
        grouped.setdefault(item["scenario"], []).append(bool(item["passed"]))
    return grouped
