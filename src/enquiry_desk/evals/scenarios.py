"""A scenario: who is calling, what they want, and what must be true when the call ends."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Expect(BaseModel):
    """What must be true at the end. Every field left out is simply not checked."""

    model_config = ConfigDict(extra="forbid")
    lead: dict[str, str] = Field(default_factory=dict)  # CRM field -> text it must contain
    booked: bool | None = None
    handoff: bool | None = None
    searched: bool | None = None
    must_state: list[str] = Field(default_factory=list)  # figures that must appear in a reply
    must_not_state: list[str] = Field(default_factory=list)  # figures that must never appear
    must_not_say: list[str] = Field(default_factory=list)  # phrases that must never appear
    no_figures: bool = False  # the desk must state no figure at all


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: str
    language: str = "english"  # english, hinglish
    persona: str  # who is calling
    goal: str  # what they want, and when they would hang up
    facts: dict[str, str] = Field(default_factory=dict)  # what they know about themselves
    opening: str  # the first thing they say
    max_turns: int = 6
    expect: Expect


def load_scenarios(path: Path) -> list[Scenario]:
    scenarios = [Scenario.model_validate(item) for item in yaml.safe_load(path.read_text(encoding="utf-8"))]
    ids = [scenario.id for scenario in scenarios]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"scenario ids are used twice: {sorted(duplicates)}")
    return scenarios
