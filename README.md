# Enquiry Desk

An enquiry agent for a small business, over chat and voice. A caller asks about courses and fees. The desk answers from the business's own documents, takes down who is calling, books a counselling call, and passes the conversation to a person when it should.

I started this to learn the parts of LLM engineering I had only read about: tool calling on the wire, an agent graph with saved state, a vector database, webhooks, cost tracking, monitoring, a voice loop. So the repo is built to be read. Every module opens with a comment on why it exists, and most design choices are there because a test run broke the simpler version.

> **Where it stands.** On a small hosted model (`gemini-3.5-flash-lite`) the desk passed 39 of 40 simulated calls, one run each. On two small local models it passed 0 to 2 of 5. Voice works on synthetic audio and has not been tried by a person. The phone line, HubSpot and the fine-tune are written and not yet run for real. All the numbers are below, including the bad ones.

The business in the demo, "Meridian Demo Academy", is made up. Its six documents were written for this project.

## How it works

```
 phone call ──┐
 browser mic ─┤  speech to text ─┐                    ┌─ text to speech ──> caller
 text chat ───┴──────────────────┴──> AGENT GRAPH ────┴─ text reply ──────> caller
 website form ── POST /webhooks/lead ──> job queue ──> extract, score ──> CRM

 AGENT GRAPH (LangGraph, one saved thread per caller)
   call_model ──tool calls──> run_tools ──> back to call_model
       │                          └── handoff asked ──> pause until a person replies
       └──text──> check the reply ──fails once──> call_model
                        └──passes, or fails twice──> send
```

A few decisions shape the rest:

- **The model client is plain HTTP.** No SDK. [`llm/wire.py`](src/enquiry_desk/llm/wire.py) builds the request and reassembles the stream itself, so the same client talks to OpenAI, Gemini, OpenRouter, Ollama or vLLM.
- **LangGraph only runs the control flow.** It gives the loop, a checkpoint after every node, and `interrupt` for the human handoff. Messages stay as the dicts that go over the wire.
- **No reply reaches the caller unchecked.** Every figure in a reply has to appear in a retrieved document, a tool result, or the caller's own words. The check uses no model. See [`agent/grounding.py`](src/enquiry_desk/agent/grounding.py).
- **Things that must always happen are done in code, not asked of the model.** Saying "I am an AI", refusing a phone number the caller never gave, refusing to announce a booking that did not happen. Each of those began as a line in the prompt that a model ignored.
- **A scenario passes on what the call left behind.** The CRM row, the booked slot, the handoff flag. No model grades another model.

## Quick start

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker. Nothing here needs an account or costs money.

```bash
uv sync --extra embed
uv run pytest
```

Start the vector database and a local model server, and pull a model (about 2 GB):

```bash
docker compose up -d qdrant ollama
docker compose exec ollama ollama pull qwen2.5:3b
```

Copy `.env.example` to `.env`, index the documents, and talk to it:

```bash
uv run python -m enquiry_desk.cli ingest
uv run python -m enquiry_desk.cli chat
```

To use a hosted model, change three lines in `.env`: `DESK_LLM_BASE_URL`, `DESK_LLM_API_KEY`, `DESK_LLM_MODEL`.

<details>
<summary>More commands</summary>

```bash
# the HTTP API, with a voice page at /voice
uv sync --extra embed --extra voice
uv run uvicorn enquiry_desk.api:app

# Prometheus :9090, Grafana :3000, Jaeger :16686
docker compose --profile monitoring up -d

# the desk in a container, on PostgreSQL
docker compose --profile app up -d --build

# SQL reports over the desk's own tables
uv run python -m enquiry_desk.cli report            # lists them
uv run python -m enquiry_desk.cli report turn_latency

# evaluations
uv run python eval/retrieval.py
uv run python eval/conversations.py
uv run python eval/run_scenarios.py --runs 3
uv run python eval/voice_check.py
uv run python eval/loadtest.py --callers 20 --turns 4
```

</details>

## What is in here

| | Where | Run for real? |
|---|---|---|
| Streaming Chat Completions client with tool calls and retries | [`llm/wire.py`](src/enquiry_desk/llm/wire.py) | Yes |
| Tool schemas, argument validation, recovery of calls written as text | [`llm/tools.py`](src/enquiry_desk/llm/tools.py) | Yes |
| Qdrant search, dense plus sparse, fused by rank | [`kb/`](src/enquiry_desk/kb) | Yes |
| The agent graph, checkpoints, human handoff | [`agent/graph.py`](src/enquiry_desk/agent/graph.py) | Yes |
| Reply check | [`agent/grounding.py`](src/enquiry_desk/agent/grounding.py) | Yes |
| Token, cost and latency records per call | [`metering.py`](src/enquiry_desk/metering.py) | Yes |
| Ten SQL reports with CTEs and window functions | [`sql/`](src/enquiry_desk/sql) | Yes, SQLite and PostgreSQL |
| Signed webhooks, a retrying job queue, lead intake | [`webhooks/`](src/enquiry_desk/webhooks), [`jobs.py`](src/enquiry_desk/jobs.py), [`intake.py`](src/enquiry_desk/intake.py) | Yes |
| HubSpot adapter | [`crm_hubspot.py`](src/enquiry_desk/crm_hubspot.py) | No, mocked responses only |
| Scenario evaluation with a simulated caller | [`evals/`](src/enquiry_desk/evals), [`eval/`](eval) | Yes, all 40 scenarios |
| Metrics, traces, alert rules, dashboard, runbook | [`observability.py`](src/enquiry_desk/observability.py), [`ops/`](ops) | Yes |
| Voice: turn detection, Whisper, Piper, interruption | [`voice/`](src/enquiry_desk/voice) | On synthetic audio |
| Browser voice page | [`web/voice.html`](web/voice.html) | No, it loads and that is all I know |
| Phone calls through Twilio Media Streams | [`voice/telephony.py`](src/enquiry_desk/voice/telephony.py) | No, test messages only |
| Fine-tuning: split by tool, leak check, scoring | [`finetune/`](finetune) | Scoring yes, training no |
| Container image and PostgreSQL | [`Dockerfile`](Dockerfile), [`db.py`](src/enquiry_desk/db.py) | Yes |

168 tests, and the same suite passes on SQLite and on PostgreSQL 17. The model is scripted in tests, so they need no network.

## Results

Everything was measured on one laptop with no GPU, on 13 October 2026. Each number has a file in [`eval/results/`](eval/results) or a script above that reproduces it.

### Does it work?

**On a hosted model, mostly yes.** 40 scenarios, with a second model playing the caller, each judged on the end state. `gemini-3.5-flash-lite` as both desk and caller, one run per scenario.

| Kind of call | Passed |
|---|---|
| A fact that is in the documents | 4 of 4 |
| A fact that is not in the documents | 3 of 4 |
| A figure that would need arithmetic | 4 of 4 |
| Booking a counselling call | 4 of 4 |
| Phone number wrong, odd, or refused | 4 of 4 |
| Refund, discount or waiver demand | 4 of 4 |
| Asks for a person, or is angry | 4 of 4 |
| Prompt injection | 4 of 4 |
| Hindi in Roman script | 4 of 4 |
| Off topic | 4 of 4 |
| **All** | **39 of 40** (95% interval 92% to 100%) |

Read that number with four cautions:

- **One run each.** I have not measured how often a scenario passes three times out of three, which is the number a business would care about.
- **I wrote the scenarios and the checks.** Reading the transcripts, one pass should have been a fail: asked about a canteen, the desk said "the academy does not have a canteen". The documents do not say that. My check for that scenario only looked for invented prices. So 38 of 40 is the fairer count, and both misses are the same fault: the model turns "the documents do not mention it" into "we do not have it".
- **It leans on the handoff.** Three of the four injection attempts and two off-topic questions passed because the desk handed the caller to a person. That is safe, and in a real business it is also a cost.
- **The caller is the same small model**, so it is not a hard opponent.

**A second, partial run.** I changed one line of the instructions ("not mentioned" is not "no") and started three runs per scenario. The free tier's daily quota ran out after 13 of the 40 scenarios, so this covers only the first four kinds of call: facts, facts not in the documents, figures needing arithmetic, and one booking.

| | |
|---|---|
| Runs completed | 37, over 13 scenarios |
| Passed | 37 of 37 |
| Scenarios with three runs that passed all three | 12 of 12 |
| "Do you have a pool / a canteen?", the fault from the first run | 6 of 6 now answer "I do not have that information" |
| Cost at list price | $0.087 in total, median $0.002 a call (about 0.2 rupees) |
| Turn time | median 4.5 s, 95th percentile 14 s |
| Time to first token | median 1.45 s |
| Calls where the reply check sent a draft back | 6 of 37 |

Do not read 37 of 37 as the desk's score. These 13 scenarios are the easier ones, and bookings, handoffs, injection and Hinglish are barely in it. The automated check first reported 34, because my phrase list for one scenario flagged "I do not have information on whether the teachers are IIT graduates" as a claim. I read all three replies, they were correct, and I fixed the list. The results file was rebuilt from the desk's stored conversations after the run was cut off, which the file says.

The first run took 88 turns and about 308,000 tokens, close to 3,500 tokens a turn, nearly all of it input: the instructions, five tool schemas and the history are sent again on every call.

**On small local models, no.** Five scripted calls, each judged on the end state.

| Model | Setting | Passed |
|---|---|---|
| `qwen2.5:3b` | first prompt | 2 of 5 |
| `qwen2.5:3b` | prompt rewritten to push searching | 1 of 5 |
| `qwen2.5:3b` | search forced in code before the model runs | 0 of 5 |
| `qwen2.5:7b` | default | 0 of 5 |
| `qwen2.5:7b` | after two new guards | 0 of 5 |

The same five scripts on `gemini-3.5-flash-lite`: 4 of 5. So the gap is the model, not the design. Rewording the prompt made the local model worse, and forcing the search did not help.

The failures were more useful than a pass would have been:

| What the model did | What I did about it |
|---|---|
| Wrote its tool call as plain text, which would have been read out to the caller | The desk now recognises that shape and runs the call |
| Saved a lead four times under a phone number nobody gave | `save_lead` only accepts a number found in the caller's own words |
| Said "I have booked your call" with nothing booked | A reply may say that only after the booking tool succeeded |
| Never said it was an AI | The first reply gets that line in code |
| (Gemini) Rejected every second request | It signs each tool call and wants the signature back. The client now returns whatever the server attaches to a call |
| Quoted a real fee against the wrong course | **Open.** The figure exists in a source, so the check passes it |
| Said "there is no swimming pool" when the documents say nothing either way | **Open.** A claim with no number in it is not checked |

The two open rows are the real limit of checking replies without a model.

### Retrieval

48 questions, each labelled with the section that answers it. `bge-small-en-v1.5` for dense, BM25 for sparse, 39 chunks.

| | Right section first | In the top four |
|---|---|---|
| Dense | 34 | 40 |
| Sparse | 27 | 36 |
| Hybrid | 30 | 42 |

Hybrid found two more in the top four and ranked four fewer first. On 48 questions that is noise, so I cannot say hybrid beats dense here. I also wrote both the documents and the questions, which makes every number an upper bound.

### Load

20 callers at once, 4 turns each, against a fake model that answers in 0.3 s. All 80 turns succeeded. The median turn took 2.2 s, and only 0.6 s of that is waiting for the model. The rest is the desk queueing behind its own checkpoint writes and database calls.

### Voice

Whisper `base` and Piper on the CPU, with a synthetic voice as the caller.

| Stage | Median |
|---|---|
| Silence before the turn is taken as over | 700 ms |
| Speech to text | 1,650 ms |
| Model (a stub in this run) | 400 ms |
| Text to speech, first sentence | 215 ms |
| **Caller's last word to first audio** | **about 3.0 s** |

Too slow for a phone call, and speech to text is over half of it.

Word error rate on synthetic English was 0.17 to 0.19, mostly spelling and number format. Hindi came back in Urdu script even with the language pinned, which scores 1.0 although many words were heard right.

### A small model on the desk's tool calls

Untuned `qwen2.5:3b` on 60 generated examples: right tool 98% of the time, every argument right 68% (95% interval 57 to 80). That gap is what the fine-tune in [`finetune/`](finetune) is meant to close. It has not been trained yet.

## Known limits

- The reply check covers figures, references and booking claims. Nothing else.
- A figure the model works out, or writes differently ("2.6 lakh" for "2,60,000"), fails the check and the caller is offered a counsellor.
- The chat endpoint has no authentication and no rate limit.
- PostgreSQL mode needs Linux. The async driver will not run on Windows' default event loop.
- An interruption stops the speech but not the model call already in flight.

## Docs

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md): what to check when each alert fires.
- [`finetune/README.md`](finetune/README.md): the fine-tuning steps.

## Acknowledgement

Much of this code was written with Claude Code as a pair programmer, and the commit history says so.
