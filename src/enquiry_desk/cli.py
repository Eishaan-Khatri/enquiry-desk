"""Command line: index the knowledge base, search it, chat with the desk, or print a report.

    uv run python -m enquiry_desk.cli ingest
    uv run python -m enquiry_desk.cli search "refund if I leave in the second month"
    uv run python -m enquiry_desk.cli chat
    uv run python -m enquiry_desk.cli report turn_latency
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from enquiry_desk.agent.service import TurnResult
from enquiry_desk.config import get_settings
from enquiry_desk.db import Database
from enquiry_desk.kb.chunking import chunk_markdown
from enquiry_desk.reports import REPORTS, as_table, run_report
from enquiry_desk.runtime import make_knowledge_base, open_desk
from enquiry_desk.tenant import knowledge_files, load_tenant


def ingest() -> None:
    settings = get_settings()
    tenant = load_tenant(settings.tenant_dir)
    chunks = []
    for path in knowledge_files(settings.tenant_dir):
        chunks += chunk_markdown(tenant.id, path.name, path.read_text(encoding="utf-8"))
    count = make_knowledge_base(settings).replace(tenant.id, chunks)
    where = settings.qdrant_url or "the embedded store"
    print(f"Indexed {count} chunks from {len(knowledge_files(settings.tenant_dir))} files for {tenant.id} into {where}.")


def search(query: str, mode: str) -> None:
    settings = get_settings()
    tenant = load_tenant(settings.tenant_dir)
    for hit in make_knowledge_base(settings).search(tenant.id, query, mode=mode):
        print(f"{hit.score:.3f}  {hit.source}\n       {hit.text[:160].replace(chr(10), ' ')}\n")


def usage_line(result: TurnResult) -> str:
    used = result.usage
    cost = "unknown (add the model to prices.toml)" if used["cost_usd"] is None else f"${used['cost_usd']:.5f}"
    return (
        f"[{used['model_calls']} model calls, {used['tool_calls']} tool calls, {used['errors']} errors, "
        f"{used['prompt_tokens']} in / {used['completion_tokens']} out tokens, "
        f"first token {used['first_token_ms'] or 0:.0f} ms, model time {used['model_ms']:.0f} ms, cost {cost}]"
    )


async def chat() -> None:
    session_id = uuid.uuid4().hex[:12]
    async with open_desk(get_settings()) as runtime:
        print(f"{runtime.tenant.business_name} enquiry desk. Session {session_id}. Type 'quit' to stop.\n")
        while True:
            text = input("you> ").strip()
            if text in ("quit", "exit"):
                break
            if not text:
                continue
            result = await runtime.desk.handle(session_id, text)
            print(f"\ndesk> {result.reply}")
            if result.problems:
                print(f"      [checker objected: {'; '.join(result.problems)}]")
            print(f"      {usage_line(result)}\n")
        print("Session total:", runtime.meter.summary(session_id))


def report(name: str | None, n: int) -> None:
    if name is None:
        for report_name, about in REPORTS.items():
            print(f"{report_name:20} {about}")
        return
    db = Database(get_settings().data_dir / "desk.db")
    print(as_table(run_report(db, name, n=n)))


async def hubspot_setup() -> None:
    from enquiry_desk.crm_hubspot import HubSpotCRM
    from enquiry_desk.runtime import INTAKE_FIELDS

    settings = get_settings()
    tenant = load_tenant(settings.tenant_dir)
    crm = HubSpotCRM(settings.hubspot_token, [*tenant.lead_fields, *INTAKE_FIELDS])
    created = await crm.ensure_properties()
    print(f"Created {len(created)} contact properties: {', '.join(created) or 'none, all were there already'}.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="enquiry_desk")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="index the tenant's kb/ folder")
    commands.add_parser("chat", help="talk to the desk in the terminal")
    searcher = commands.add_parser("search", help="run one search against the knowledge base")
    searcher.add_argument("query")
    searcher.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    reporter = commands.add_parser("report", help="print a SQL report, or list them")
    reporter.add_argument("name", nargs="?", choices=list(REPORTS))
    reporter.add_argument("-n", type=int, default=3, help="rows per group, for slowest_turns")
    commands.add_parser("hubspot-setup", help="create the desk's custom contact properties in HubSpot")
    args = parser.parse_args()
    if args.command == "hubspot-setup":
        asyncio.run(hubspot_setup())
    elif args.command == "ingest":
        ingest()
    elif args.command == "search":
        search(args.query, args.mode)
    elif args.command == "report":
        report(args.name, args.n)
    else:
        asyncio.run(chat())


if __name__ == "__main__":
    main()
