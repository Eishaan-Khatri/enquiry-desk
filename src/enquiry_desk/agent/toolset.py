"""The five things the agent can do, each a function the model calls by name."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from enquiry_desk.booking import Slots, SlotTaken
from enquiry_desk.crm import CRM, normalise_phone
from enquiry_desk.kb.store import Hit, KnowledgeBase
from enquiry_desk.llm.tools import Tool, ToolError, ToolRegistry
from enquiry_desk.tenant import Tenant


@dataclass
class ToolContext:
    """What a tool may read and change for the session it is running in."""

    session_id: str
    refs: dict[str, dict] = field(default_factory=dict)  # "K1" -> {"chunk_id", "source"}
    lead_id: str | None = None
    handoff: dict | None = None
    # Every digit the caller has typed, one run per message. None turns the phone check off.
    caller_digits: str | None = None

    def ref_for(self, hit: Hit) -> str:
        """The same chunk keeps the same reference for the whole conversation."""
        for ref, info in self.refs.items():
            if info["chunk_id"] == hit.chunk_id:
                return ref
        ref = f"K{len(self.refs) + 1}"
        self.refs[ref] = {"chunk_id": hit.chunk_id, "source": hit.source}
        return ref


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(_Args):
    query: str = Field(min_length=2, description="What to look up, as a short search query in English.")


class NoArgs(_Args):
    pass


class BookArgs(_Args):
    slot_id: str = Field(description="The slot_id of the slot the caller chose, exactly as list_slots returned it.")


class HandoffArgs(_Args):
    reason: Literal["asked_for_person", "upset", "discount_or_refund", "out_of_scope", "other"]
    summary: str = Field(min_length=5, description="Two sentences for the counsellor: who is calling and what they need.")


def build_tools(
    tenant: Tenant,
    kb: KnowledgeBase,
    crm: CRM,
    slots: Slots,
    clock: Callable[[], datetime],
    on_handoff: Callable[[dict], None] | None = None,
) -> ToolRegistry:
    # The lead fields differ by business, so this tool's schema is built from the tenant file.
    lead_args = create_model(
        "SaveLeadArgs",
        __base__=_Args,
        phone=(str, Field(description="The caller's mobile number, ten digits.")),
        name=(str, Field("", description="The caller's name.")),
        **{name: (str, Field("", description=meaning)) for name, meaning in tenant.lead_fields.items()},
    )

    async def search_knowledge(args: SearchArgs, ctx: ToolContext) -> dict:
        hits = await asyncio.to_thread(kb.search, tenant.id, args.query)
        if not hits:
            return {"results": [], "note": "Nothing was found. Do not answer from memory."}
        return {"results": [{"ref": ctx.ref_for(hit), "source": hit.source, "text": hit.text} for hit in hits]}

    async def save_lead(args: BaseModel, ctx: ToolContext) -> dict:
        values = args.model_dump()
        try:
            phone = normalise_phone(values.pop("phone"))
        except ValueError as error:
            raise ToolError(f"That phone number is not valid: {error}. Ask the caller to repeat it.") from error
        # The reply check stops an invented figure reaching the caller. This stops an invented
        # phone number reaching the CRM. A local model saved four leads under a number nobody
        # had given, in a conversation where the caller only wanted a refund.
        if ctx.caller_digits is not None and phone[-10:] not in ctx.caller_digits:
            raise ToolError("The caller has not given that phone number. Do not guess one. Ask the caller for their number.")
        lead = await crm.upsert_lead(phone, values.pop("name"), values)
        ctx.lead_id = lead.id
        missing = [name for name in tenant.lead_fields if not lead.fields.get(name)]
        if not missing and lead.stage == "new":
            await crm.set_stage(lead.id, "qualified")
        return {"lead_id": lead.id, "phone": lead.phone, "name": lead.name, "saved": lead.fields, "still_missing": missing}

    async def list_slots(args: NoArgs, ctx: ToolContext) -> dict:
        free = slots.open_slots(clock())
        if not free:
            return {"slots": [], "note": "No slots are open. Offer a callback from a counsellor."}
        return {"slots": free}

    async def book_slot(args: BookArgs, ctx: ToolContext) -> dict:
        if ctx.lead_id is None:
            raise ToolError("Save the lead with save_lead first. A booking needs a phone number.")
        try:
            booked = slots.book(args.slot_id, ctx.lead_id)
        except SlotTaken as error:
            raise ToolError(f"{error} Call list_slots again and offer the caller the new options.") from error
        await crm.set_stage(ctx.lead_id, "counselling_booked")
        await crm.add_note(ctx.lead_id, f"Counselling call booked for {booked['when']}.")
        return {"booked": True, **booked}

    async def request_handoff(args: HandoffArgs, ctx: ToolContext) -> dict:
        ctx.handoff = {"reason": args.reason, "summary": args.summary}
        if ctx.lead_id:
            await crm.add_note(ctx.lead_id, f"Handed to a counsellor ({args.reason}): {args.summary}")
        if on_handoff:
            on_handoff({"session_id": ctx.session_id, **ctx.handoff})
        return {"status": "A counsellor has been asked to take over."}

    return ToolRegistry(
        [
            Tool(
                "search_knowledge",
                f"Search {tenant.business_name}'s own documents: courses, fees, timings, admission, "
                "policies. Call this before stating any fact about the institute.",
                SearchArgs,
                search_knowledge,
            ),
            Tool(
                "save_lead",
                "Save or update the caller's details in the CRM. Call it as soon as you have the phone "
                "number, and again whenever you learn another detail. Leave unknown fields empty.",
                lead_args,
                save_lead,
            ),
            Tool("list_slots", "List the next open slots for a counselling call.", NoArgs, list_slots),
            Tool(
                "book_slot",
                "Book a counselling call. Only after the caller has chosen a slot and the lead is saved.",
                BookArgs,
                book_slot,
            ),
            Tool(
                "request_handoff",
                "Pass the conversation to a human counsellor.",
                HandoffArgs,
                request_handoff,
            ),
        ]
    )
