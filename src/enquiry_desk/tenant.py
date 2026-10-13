"""The business the desk answers for. Everything specific to one business is in its folder."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class Tenant(BaseModel):
    id: str
    business_name: str
    callers: str  # who rings up, in a few words
    lead_fields: dict[str, str]  # field name -> what it means, shown to the model
    slot_hours: list[str]
    closed_weekdays: list[int] = [6]  # Monday is 0, so 6 is Sunday
    handoff_line: str
    never_promise: str = ""


def load_tenant(folder: Path) -> Tenant:
    return Tenant.model_validate(yaml.safe_load((folder / "tenant.yaml").read_text(encoding="utf-8")))


def knowledge_files(folder: Path) -> list[Path]:
    return sorted((folder / "kb").glob("*.md"))
