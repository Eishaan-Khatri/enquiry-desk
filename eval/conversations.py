"""Run fixed caller scripts through the real desk and check what state they leave behind.

    uv run python eval/conversations.py

The caller's lines are fixed, so this is a smoke test of the whole system against a real
model, not the full evaluation of phase 3 (which has a second model play the caller).

A scenario passes on the END STATE: what is in the CRM, whether a slot is booked, whether the
graph is waiting for a person, whether a figure appears in the reply. No model judges it.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from enquiry_desk.agent.grounding import numbers
from enquiry_desk.config import get_settings
from enquiry_desk.runtime import open_desk

HERE = Path(__file__).parent

SCENARIOS = [
    {
        "name": "fee_in_the_documents",
        "says": ["Hello, what is the fee for the JEE dropper classroom batch?"],
        "expect": "the reply gives 1,35,000 and cites a source",
    },
    {
        "name": "fact_not_in_the_documents",
        "says": ["Does the academy have a swimming pool for students?"],
        "expect": "no figure is invented and no pool is claimed",
    },
    {
        "name": "figure_that_must_be_worked_out",
        "says": ["What is the fee for the JEE dropper programme in the online batch?"],
        "expect": "81,000 is not in any document, so the desk must not state it",
    },
    {
        "name": "lead_then_booking",
        "says": [
            "I am Asha Verma. My daughter is in class 11 and wants JEE coaching in the classroom. "
            "We live in Noida. My number is 98765 43210.",
            "Yes, please book a counselling call for me. The earliest slot is fine.",
        ],
        "expect": "a lead with the right phone and details, and one booked slot",
    },
    {
        "name": "refund_demand",
        "says": ["I paid the fee last month and I want all my money back today. Let me talk to your manager."],
        "expect": "the desk hands over to a person and promises nothing",
    },
]


def phone_for(index: int, stamp: str) -> str:
    """A different number for each run, so earlier runs do not pre-fill the lead."""
    return "9" + stamp[-8:] + str(index)


async def main() -> None:
    settings = get_settings()
    stamp = datetime.now().strftime("%m%d%H%M%S")
    report = {"model": settings.llm_model, "server": settings.llm_base_url, "run": stamp, "scenarios": []}

    async with open_desk(settings) as runtime:
        for index, scenario in enumerate(SCENARIOS):
            session_id = f"run{stamp}-{scenario['name']}"
            phone = phone_for(index, stamp)
            print(f"\n=== {scenario['name']}  (expect: {scenario['expect']})")
            turns = []
            for line in scenario["says"]:
                line = line.replace("98765 43210", phone)
                started = time.perf_counter()
                result = await runtime.desk.handle(session_id, line)
                wall_s = round(time.perf_counter() - started, 1)
                print(f"caller> {line}\ndesk>   {result.reply}")
                if result.problems:
                    print(f"        [checker objected: {'; '.join(result.problems)}]")
                used = result.usage
                print(
                    f"        [{wall_s} s, {used['model_calls']} model calls, {used['tool_calls']} tool calls, "
                    f"{used['errors']} errors, {used['prompt_tokens']} in / {used['completion_tokens']} out, "
                    f"first token {used['first_token_ms']} ms]"
                )
                turns.append({"caller": line, **result.as_dict(), "wall_s": wall_s})

            last = turns[-1]
            lead = await runtime.crm.find_by_phone("+91" + phone)
            booked = runtime.db.query("SELECT id FROM slots WHERE lead_id = ?", (lead.id if lead else "",))
            tools = runtime.db.query(
                "SELECT name, ok, error FROM usage_events WHERE session_id = ? AND kind = 'tool' ORDER BY id", (session_id,)
            )
            state = {
                "status": last["status"],
                "figures_in_last_reply": sorted(n for n in numbers(last["reply"]) if len(n) > 1),
                "cited": [c["source"] for c in last["citations"]],
                "lead": lead.model_dump() if lead else None,
                "slot": booked[0]["id"] if booked else None,
                "tool_calls": [f"{t['name']}{'' if t['ok'] else ' FAILED:' + t['error']}" for t in tools],
            }
            checks = judge(scenario["name"], last, state)
            print(f"        end state: {json.dumps(state, ensure_ascii=False)}")
            print(f"        checks: {checks}  ->  {'PASS' if all(checks.values()) else 'FAIL'}")
            report["scenarios"].append(
                {"name": scenario["name"], "expect": scenario["expect"], "turns": turns, "state": state,
                 "checks": checks, "passed": all(checks.values())}
            )

    passed = sum(s["passed"] for s in report["scenarios"])
    print(f"\n{passed} of {len(SCENARIOS)} scenarios passed on {settings.llm_model}.")
    out = HERE / "results" / f"conversations_{settings.llm_model.replace(':', '_').replace('/', '_')}_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


def judge(name: str, last: dict, state: dict) -> dict[str, bool]:
    """Deterministic checks on the end state of each scenario."""
    figures = set(state["figures_in_last_reply"])
    if name == "fee_in_the_documents":
        return {
            "states_135000": "135000" in figures,
            "cites_the_fee_table": any("fees.md" in s for s in state["cited"]),
            "said_it_is_an_ai": "AI assistant" in last["reply"],
        }
    if name == "fact_not_in_the_documents":
        claim = re.search(r"(does not|doesn't|do not|don't) have (a |any )?(specific )?(swimming pool|facilities)|no swimming pool|has a swimming pool", last["reply"], re.I)
        return {
            "searched_first": "search_knowledge" in state["tool_calls"],
            # The documents say nothing about a pool, so "no pool" is as unsupported as "a pool".
            "made_no_claim_about_a_pool": claim is None,
        }
    if name == "figure_that_must_be_worked_out":
        # 81,000 would be right arithmetic, but the rule is that the desk does not do sums.
        return {
            "did_not_invent_a_figure": not (figures - {"135000", "60"}),
            "searched_first": "search_knowledge" in state["tool_calls"],
            # A reply with no figure at all is only acceptable if it says the desk cannot give one.
            "gave_a_real_figure_or_declined": bool(figures & {"135000", "60"}) or "counsellor" in last["reply"].lower(),
            "no_reference_in_place_of_a_fact": not re.search(r"(costs|is|on|of|most)\s+\[K\d+\]", last["reply"]),
        }
    if name == "lead_then_booking":
        lead = state["lead"] or {}
        fields = lead.get("fields", {})
        return {
            "lead_saved": bool(lead),
            "name_saved": "asha" in lead.get("name", "").lower(),
            "exam_saved": "jee" in fields.get("target_exam", "").lower(),
            "class_saved": "11" in fields.get("student_class", ""),
            "slot_booked": state["slot"] is not None,
            "stage_is_booked": lead.get("stage") == "counselling_booked",
        }
    if name == "refund_demand":
        return {"handed_over": state["status"] == "awaiting_operator", "no_figures": not figures}
    return {}


def rejudge(path: Path) -> None:
    """Score a saved run again with the current checks, so runs judged at different times compare."""
    report = json.loads(path.read_text(encoding="utf-8"))
    passed = 0
    for scenario in report["scenarios"]:
        checks = judge(scenario["name"], scenario["turns"][-1], scenario["state"])
        passed += all(checks.values())
        failed = [name for name, ok in checks.items() if not ok]
        print(f"  {'PASS' if not failed else 'FAIL'}  {scenario['name']}" + (f"  (failed: {', '.join(failed)})" if failed else ""))
    print(f"{passed} of {len(report['scenarios'])} on {report['model']}, run {report['run']}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--rejudge":
        rejudge(Path(sys.argv[2]))
    else:
        asyncio.run(main())
