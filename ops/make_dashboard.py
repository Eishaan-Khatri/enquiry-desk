"""Write the Grafana dashboard file. Edit the panels here, then run:

    uv run python ops/make_dashboard.py

A dashboard is a long JSON file that is painful to edit by hand. Generating it keeps the part
that matters, the queries, in one short list.
"""

from __future__ import annotations

import json
from pathlib import Path

# (title, unit, [(legend, PromQL)])
PANELS = [
    ("Turns per minute, by how they ended", "short", [("{{outcome}}", "sum by (outcome) (rate(desk_turns_total[5m])) * 60")]),
    (
        "Turn time",
        "s",
        [
            ("median", "histogram_quantile(0.5, sum by (le) (rate(desk_turn_seconds_bucket[5m])))"),
            ("95th percentile", "histogram_quantile(0.95, sum by (le) (rate(desk_turn_seconds_bucket[5m])))"),
        ],
    ),
    ("Time to first token, 95th percentile", "s", [("p95", "histogram_quantile(0.95, sum by (le) (rate(desk_first_token_seconds_bucket[5m])))")]),
    (
        "Share of model calls that fail",
        "percentunit",
        [("failed", 'sum(rate(desk_llm_calls_total{outcome="error"}[5m])) / sum(rate(desk_llm_calls_total[5m]))')],
    ),
    ("Tokens per minute", "short", [("{{kind}}", "sum by (kind) (rate(desk_llm_tokens_total[5m])) * 60")]),
    ("Model spend, last 24 hours (USD)", "currencyUSD", [("spend", "sum(increase(desk_llm_cost_usd_total[24h]))")]),
    ("Tool calls per minute", "short", [("{{tool}} {{outcome}}", "sum by (tool, outcome) (rate(desk_tool_calls_total[5m])) * 60")]),
    ("Replies the check sent back, per hour", "short", [("{{reason}}", "sum by (reason) (increase(desk_reply_rejections_total[1h]))")]),
    (
        "Share of turns refused",
        "percentunit",
        [("refused", 'sum(rate(desk_turns_total{outcome="refused"}[1h])) / sum(rate(desk_turns_total[1h]))')],
    ),
    ("Job queue", "short", [("{{status}}", "desk_jobs")]),
    ("Webhooks per hour", "short", [("{{source}} {{outcome}}", "sum by (source, outcome) (increase(desk_webhooks_total[1h]))")]),
    ("Desk is up", "short", [("up", 'up{job="desk"}')]),
]


def dashboard() -> dict:
    panels = []
    for number, (title, unit, queries) in enumerate(PANELS):
        panels.append(
            {
                "id": number + 1,
                "title": title,
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 8, "x": 8 * (number % 3), "y": 8 * (number // 3)},
                "fieldConfig": {"defaults": {"unit": unit, "min": 0}, "overrides": []},
                "targets": [
                    {"refId": chr(65 + i), "expr": expr, "legendFormat": legend, "datasource": {"type": "prometheus", "uid": "prometheus"}}
                    for i, (legend, expr) in enumerate(queries)
                ],
            }
        )
    return {
        "uid": "enquiry-desk",
        "title": "Enquiry Desk",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "15s",
        "time": {"from": "now-1h", "to": "now"},
        "panels": panels,
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "grafana" / "dashboards" / "desk.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dashboard(), indent=1), encoding="utf-8")
    print(f"Wrote {out} with {len(PANELS)} panels.")
