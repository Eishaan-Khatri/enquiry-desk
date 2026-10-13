# Runbook

One entry per alert in `ops/alerts.yml`. Each says what the alert means, what to look at first, and what usually fixes it.

Where to look:

| Tool | Address | Use it for |
|---|---|---|
| Grafana | http://localhost:3000 | The "Enquiry Desk" dashboard: is it one thing or everything? |
| Prometheus | http://localhost:9090/alerts | Which alerts are firing, and since when |
| Jaeger | http://localhost:16686 | One slow or failed turn, step by step |
| SQL reports | `uv run python -m enquiry_desk.cli report <name>` | The same facts from the desk's own tables |
| Job queue | `GET /v1/jobs` | Jobs waiting, and dead jobs with their last error |

---

## DeskDown

**Means.** Prometheus could not read `/metrics` for two minutes. Callers are probably getting errors.

**Check first.**
1. Is the process running? `GET /healthz` should answer `{"ok": true}`.
2. The process log, last 50 lines. A crash at start-up is nearly always a setting: a missing key, a wrong Qdrant address.
3. Is Qdrant up? The desk cannot start without it when `DESK_QDRANT_URL` is set.

**Usually.** Restart the process. If it crashes again at once, the log names the setting.

## DeskSlowTurns

**Means.** One turn in twenty takes more than 8 seconds, for ten minutes.

**Check first.**
1. The "Time to first token" panel. If it rose with turn time, the model provider is slow. Nothing in the desk is wrong.
2. If first token is steady, open Jaeger, sort the `turn` spans by duration, and open a slow one. The child spans show whether the time went to the model, a tool, or neither.
3. `report slowest_turns` lists the slowest turns with their model and tool call counts. Many model calls in one turn means the model is looping on tool calls.

**Usually.** The provider is slow and it passes. If turns show five or six model calls each, look at the tool errors: a tool that keeps failing makes the model retry.

## DeskModelErrors

**Means.** More than 5% of model calls fail, for five minutes.

**Check first.**
1. The provider's status page.
2. The log for the status code. 401 is a bad or expired key. 429 is the rate limit. 5xx is the provider.
3. Whether `attempts` in the `usage_events` table is above 1. If so the client is already retrying, and what you see are the calls that failed three times.

**Usually.** A 429 means too many callers for the plan: raise the limit or add a second key. A 401 means rotate the key in `.env` and restart.

## DeskToolErrors

**Means.** One tool is crashing on more than 20% of its calls. Only `internal_error` counts. A caller who mistypes a phone number produces `tool_error`, which is normal.

**Check first.**
1. The alert names the tool.
2. `search_knowledge`: is Qdrant up, and does the collection exist? Run `cli search "fees"`.
3. `save_lead` or `book_slot`: is the CRM reachable? With HubSpot, look for 401 (token) or 429 (rate limit) in the log.

**Usually.** The service behind the tool is down. Until it is back the model tells callers it could not do that, which is the designed behaviour.

## DeskRefusalSpike

**Means.** The desk is refusing more than 15% of turns, and at least twice yesterday's rate. A refusal is the fixed line "I do not have that information with me".

No code changed, so something outside did. This alert exists only because the desk is built on a model.

**Check first.**
1. `report refusals_weekly` for the trend, and the `reply_checks` table for the reasons. The reasons name the figures that failed.
2. Were the documents re-indexed? A changed fee means old figures in the model's habits now fail the check, which is correct.
3. Did the provider change the model behind the same name? Compare the model name in recent `usage_events` rows.
4. Read ten refused conversations. If callers are all asking about the same new thing, the documents have a gap.

**Usually.** A gap in the documents. Add the missing section and index again.

## DeskDeadJobs

**Means.** A job failed five times, or failed in a way a retry cannot fix. If it was a lead, that lead is not in the CRM.

**Check first.**
1. `GET /v1/jobs`. The `dead` list gives each job's kind and last error.
2. A lead with an invalid phone number is dead on purpose. Someone should look at the form submission by hand.
3. A `notify` job that died means the alert address is wrong or its server is down.

**Usually.** Fix the cause, then set the job back to `queued` in the `jobs` table. Nothing is lost: the payload is still in the row.

## DeskQueueBacklog

**Means.** More than 50 jobs have been waiting for ten minutes.

**Check first.**
1. Is the worker running? It starts with the API. A worker that crashed leaves jobs in `running`, and they return to the queue after five minutes.
2. Is each job slow? A lead job makes one model call, so a slow model makes a slow queue.

**Usually.** A burst of leads from a campaign. It clears on its own. If it does not, the one worker is not enough.

## DeskSpendToday

**Means.** Model spend in the last 24 hours passed the budget in the rule.

**Check first.**
1. `report daily_cost` and `report session_costs`. One runaway session, or many normal ones?
2. One session with dozens of turns is someone probing the desk. The per-session cap in `.env` (`DESK_SESSION_COST_CAP_USD`) should have stopped it. Check that prices are filled in, because the cap cannot work on a model whose price is unknown.

**Usually.** More callers than expected. Raise the budget in the rule, or move to a cheaper model.

## DeskWebhookForgery

**Means.** More than 20 webhooks with bad signatures in 15 minutes.

**Check first.**
1. The log line gives the reason: `mismatch`, `stale` or `missing header`.
2. `stale` on every request means the sender's clock is wrong, or requests are being queued somewhere for more than five minutes.
3. `mismatch` on every request from the real website means the secret was changed on one side only.

**Usually.** A secret rotated on one side. If the requests come from an unknown address, someone is guessing, and the check is doing its job.
