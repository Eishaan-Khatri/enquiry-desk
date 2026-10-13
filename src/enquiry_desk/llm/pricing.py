"""Token prices and the cost of one model call.

Prices live in prices.toml, each with the date it was checked. A model with no row there has
an unknown cost, which is reported as None and never as zero.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from enquiry_desk.llm.wire import Usage


@dataclass(frozen=True)
class Price:
    """US dollars per one million tokens."""

    input: float
    output: float
    cached_input: float | None = None  # None means cached tokens are billed as normal input


def load_prices(path: Path) -> dict[str, Price]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        table = tomllib.load(handle).get("models", {})
    return {
        model: Price(
            input=float(row["input"]),
            output=float(row["output"]),
            cached_input=float(row["cached_input"]) if "cached_input" in row else None,
        )
        for model, row in table.items()
    }


def cost_usd(price: Price | None, usage: Usage | None) -> float | None:
    if price is None or usage is None:
        return None
    cached_rate = price.input if price.cached_input is None else price.cached_input
    fresh = usage.prompt_tokens - usage.cached_tokens
    return (
        fresh * price.input
        + usage.cached_tokens * cached_rate
        + usage.completion_tokens * price.output
    ) / 1_000_000
