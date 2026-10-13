"""Phase 2: webhook signatures, the job queue, lead intake, the HubSpot adapter, the endpoints."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from enquiry_desk.api import create_app
from enquiry_desk.crm_hubspot import HubSpotCRM, HubSpotError
from enquiry_desk.db import Database
from enquiry_desk.intake import Notifier, score
from enquiry_desk.jobs import Job, Jobs, Permanent, Worker, backoff_seconds
from enquiry_desk.webhooks import signing
from enquiry_desk.webhooks.signing import BadSignature
from tests.conftest import call, say

# ---- signatures -----------------------------------------------------------------------------

BODY = b'{"event_id": "e1", "phone": "9876543210"}'


def test_a_genuine_recent_request_is_accepted():
    signature = signing.sign("secret", "1000", BODY)
    assert signature.startswith("sha256=") and len(signature) == 7 + 64
    signing.verify("secret", "1000", signature, BODY, now=1100)


@pytest.mark.parametrize(
    ("secret", "timestamp", "body", "now", "reason"),
    [
        ("secret", "1000", BODY + b" ", 1100, "mismatch"),  # one byte of the body changed
        ("other", "1000", BODY, 1100, "mismatch"),  # signed with a different secret
        ("secret", "1000", BODY, 1000 + 301, "stale"),  # a captured request replayed later
        ("secret", "1000", BODY, 1000 - 301, "stale"),
        ("secret", "soon", BODY, 1100, "timestamp is not a number"),
        ("", "1000", BODY, 1100, "no secret"),
    ],
)
def test_everything_else_is_refused(secret, timestamp, body, now, reason):
    signature = signing.sign("secret", "1000", BODY)
    with pytest.raises(BadSignature, match=reason):
        signing.verify(secret, timestamp, signature, body, now)


def test_a_replay_with_a_fresh_timestamp_fails_because_the_timestamp_is_signed():
    signature = signing.sign("secret", "1000", BODY)
    with pytest.raises(BadSignature, match="mismatch"):
        signing.verify("secret", "5000", signature, BODY, now=5000)
    with pytest.raises(BadSignature, match="missing"):
        signing.verify("secret", None, signature, BODY, now=1000)


def test_hubspot_v3_matches_an_independent_computation():
    url, stamp = "https://desk.test/webhooks/hubspot", "1700000000000"
    expected = base64.b64encode(hmac.new(b"hub", b"POST" + url.encode() + b"[]" + stamp.encode(), hashlib.sha256).digest()).decode()
    assert signing.hubspot_v3("hub", "post", url, b"[]", stamp) == expected
    signing.verify_hubspot("hub", "POST", url, b"[]", stamp, expected, now=1_700_000_100)
    with pytest.raises(BadSignature, match="stale"):  # the timestamp is in milliseconds
        signing.verify_hubspot("hub", "POST", url, b"[]", stamp, expected, now=1_700_000_400)
    with pytest.raises(BadSignature, match="mismatch"):
        signing.verify_hubspot("hub", "POST", url + "?x=1", b"[]", stamp, expected, now=1_700_000_100)


def test_twilio_matches_the_example_in_twilio_documentation():
    params = {"CallSid": "CA1234567890ABCDE", "Caller": "+12349013030", "Digits": "1234", "From": "+12349013030", "To": "+18005551212"}
    url = "https://mycompany.com/myapp.php?foo=1&bar=2"
    assert signing.twilio("12345", url, params) == "0/KCTR6DLpKmkAf8muzZqo1nDgQ="
    signing.verify_twilio("12345", url, dict(reversed(params.items())), "0/KCTR6DLpKmkAf8muzZqo1nDgQ=")  # order is irrelevant
    with pytest.raises(BadSignature):
        signing.verify_twilio("12345", url, {**params, "Digits": "9999"}, "0/KCTR6DLpKmkAf8muzZqo1nDgQ=")


# ---- job queue ------------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_same_event_twice_is_one_job():
    jobs = Jobs(Database())
    assert jobs.enqueue("lead", {"a": 1}, dedupe_key="lead:e1") is True
    assert jobs.enqueue("lead", {"a": 1}, dedupe_key="lead:e1") is False
    assert jobs.enqueue("notify", {"text": "x"}) is True and jobs.enqueue("notify", {"text": "x"}) is True  # no key, no dedupe
    assert jobs.counts() == {"queued": 3, "running": 0, "done": 0, "dead": 0}


def test_jobs_are_claimed_oldest_first_and_only_when_due():
    clock = Clock()
    jobs = Jobs(Database(), clock)
    jobs.enqueue("later", {}, delay=60)
    jobs.enqueue("first", {})
    jobs.enqueue("second", {})
    assert [jobs.claim().kind, jobs.claim().kind, jobs.claim()] == ["first", "second", None]
    clock.now += 61
    assert jobs.claim().kind == "later"


def test_a_failing_job_backs_off_then_dies_on_the_fifth_attempt():
    clock = Clock()
    jobs = Jobs(Database(), clock)
    jobs.enqueue("lead", {})
    waits = []
    for attempt in range(1, 6):
        job = jobs.claim()
        assert job.attempts == attempt
        status = jobs.failed(job, "CRM is down")
        if status == "queued":
            assert jobs.claim() is None  # not due yet
            waits.append(backoff_seconds(attempt))
            clock.now += backoff_seconds(attempt)
    assert waits == [30, 60, 120, 240] and status == "dead"
    assert jobs.dead()[0]["last_error"] == "CRM is down" and jobs.dead()[0]["attempts"] == 5
    assert backoff_seconds(20) == 3600


def test_a_job_left_running_by_a_dead_process_is_put_back():
    clock = Clock()
    jobs = Jobs(Database(), clock)
    jobs.enqueue("lead", {})
    jobs.claim()
    assert jobs.recover() == 0  # claimed a moment ago, so it may still be running
    clock.now += 301
    assert jobs.recover() == 1 and jobs.claim().attempts == 2


async def test_the_worker_retries_gives_up_on_permanent_errors_and_marks_success():
    jobs = Jobs(Database())
    seen = []

    async def handler(payload: dict) -> None:
        seen.append(payload["n"])
        if payload["n"] == 1:
            raise RuntimeError("timeout")
        if payload["n"] == 2:
            raise Permanent("phone number has five digits")

    for n in (1, 2, 3):
        jobs.enqueue("work", {"n": n})
    jobs.enqueue("unknown_kind", {})
    worker = Worker(jobs, {"work": handler})
    assert [await worker.run_one() for _ in range(5)] == [True, True, True, True, False]
    assert seen == [1, 2, 3]
    assert jobs.counts() == {"queued": 1, "running": 0, "done": 1, "dead": 2}
    assert [d["last_error"] for d in jobs.dead()] == ["phone number has five digits", "no handler for job kind 'unknown_kind'"]


# ---- lead intake ----------------------------------------------------------------------------

FORM = {"event_id": "e1", "phone": "98765 43210", "name": "Asha", "message": "My son is in 11th, wants JEE classes in the classroom, starting this week. We are in Noida.", "source": "website"}
RECORD = {"student_class": "11", "target_exam": "JEE", "city": "Noida", "mode": "classroom", "wants": "JEE classroom coaching for class 11", "urgency": "this_week"}


async def run_all(world) -> None:
    while await world.worker.run_one():
        pass


async def test_a_form_lead_is_extracted_scored_saved_and_announced(world):
    world.intake_chat.replies = [call(("record_lead", RECORD))]
    world.jobs.enqueue("lead", FORM, dedupe_key="lead:e1")
    await run_all(world)

    lead = await world.crm.find_by_phone("+919876543210")
    assert lead.name == "Asha" and lead.fields["target_exam"] == "JEE" and lead.fields["city"] == "Noida"
    assert (lead.fields["lead_score"], lead.fields["lead_band"], lead.fields["source"]) == ("100", "hot", "website")
    note = world.crm.notes(lead.id)[0]
    assert "[e1]" in note and "Score: 100 (hot)" in note and "Urgency: this_week" in note
    assert world.sent == [{"text": "Hot lead: Asha, +919876543210. JEE classroom coaching for class 11 (score 100)"}]
    # The request forced the tool and sent the tenant's fields in the schema.
    assert world.intake_chat.requests[0][0]["role"] == "system"
    assert world.meter.summary("lead:e1")["model_calls"] == 1
    assert world.jobs.counts()["done"] == 2 and world.jobs.counts()["dead"] == 0
    assert lead.stage == "qualified"  # every detail was known, so it skipped past "new"


async def test_a_copied_field_description_is_blanked_and_no_alert_is_queued_without_a_target(world, tenant):
    from enquiry_desk.intake import Intake

    quiet = Intake(tenant, world.intake_chat, world.crm, world.meter, world.jobs, notify_bands=())
    world.intake_chat.replies = [call(("record_lead", {**RECORD, "wants": "A short summary of the request, in your own words."}))]
    await quiet.handle(FORM)

    lead = await world.crm.find_by_phone("+919876543210")
    assert "Wants: (not extracted)" in world.crm.notes(lead.id)[0]
    assert world.jobs.counts()["queued"] == 0


async def test_an_invalid_extraction_is_sent_back_once(world):
    world.intake_chat.replies = [call(("record_lead", {**RECORD, "urgency": "soon"})), call(("record_lead", RECORD))]
    world.jobs.enqueue("lead", FORM)
    await run_all(world)

    retry = world.intake_chat.requests[1]
    assert retry[-1]["role"] == "user" and "not accepted" in retry[-1]["content"]
    assert retry[-2]["role"] == "tool" and retry[-3]["tool_calls"]
    assert (await world.crm.find_by_phone("+919876543210")).fields["lead_band"] == "hot"


async def test_the_lead_is_saved_even_when_extraction_never_works(world):
    world.intake_chat.replies = [say("Sure, I can help with that!"), say("{not json")]
    world.jobs.enqueue("lead", FORM)
    await run_all(world)

    lead = await world.crm.find_by_phone("+919876543210")
    assert lead.name == "Asha" and lead.fields["lead_band"] == "cold" and "target_exam" not in lead.fields
    assert "Wants: (not extracted)" in world.crm.notes(lead.id)[0] and world.sent == []


async def test_json_in_the_reply_text_is_accepted_when_the_server_ignores_tool_choice(world):
    world.intake_chat.replies = [say("```json\n" + json.dumps(RECORD) + "\n```")]
    world.jobs.enqueue("lead", FORM)
    await run_all(world)
    assert (await world.crm.find_by_phone("+919876543210")).fields["mode"] == "classroom"


async def test_a_bad_phone_number_kills_the_job_at_once(world):
    world.jobs.enqueue("lead", {**FORM, "phone": "12345"})
    await run_all(world)
    assert world.jobs.counts()["dead"] == 1 and "ten digits" in world.jobs.dead()[0]["last_error"]
    assert world.intake_chat.requests == []  # no model call was paid for


def test_the_score_can_be_recomputed_by_hand(tenant):
    assert score(None, tenant) == (30, "cold")
    assert score({"urgency": "unknown"}, tenant) == (30, "cold")
    assert score({"target_exam": "JEE", "city": "Noida", "urgency": "later"}, tenant) == (30 + 20 + 5, "warm")
    assert score(RECORD, tenant) == (100, "hot")


async def test_the_alert_is_signed_and_a_refusal_is_an_error():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"], seen["headers"] = request.content, request.headers
        return httpx.Response(seen.get("status", 200))

    notifier = Notifier("https://hooks.test/x", "alert-secret", httpx.AsyncClient(transport=httpx.MockTransport(handler)), clock=lambda: 1000)
    await notifier.handle({"text": "Hot lead"})
    assert json.loads(seen["body"]) == {"text": "Hot lead", "content": "Hot lead"}
    signing.verify("alert-secret", seen["headers"]["x-desk-timestamp"], seen["headers"]["x-desk-signature"], seen["body"], now=1000)
    seen["status"] = 500
    with pytest.raises(RuntimeError, match="500"):
        await notifier.handle({"text": "Hot lead"})


# ---- HubSpot adapter ------------------------------------------------------------------------


class FakeHubSpot:
    """Answers with the documented response shapes and keeps every request."""

    def __init__(self):
        self.requests: list[tuple[str, str, dict]] = []
        self.contacts: dict[str, dict] = {}
        self.fail_next: list[httpx.Response] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, body))
        assert request.headers["authorization"] == "Bearer token-1"
        if self.fail_next:
            return self.fail_next.pop(0)
        path = request.url.path
        if path == "/crm/v3/objects/contacts/search":
            phone = body["filterGroups"][0]["filters"][0]["value"]
            hits = [{"id": cid, "properties": props} for cid, props in self.contacts.items() if props.get("phone") == phone]
            return httpx.Response(200, json={"total": len(hits), "results": hits[:1]})
        if path == "/crm/v3/objects/contacts" and request.method == "POST":
            cid = str(500 + len(self.contacts))
            self.contacts[cid] = dict(body["properties"])
            return httpx.Response(201, json={"id": cid, "properties": body["properties"]})
        if path.startswith("/crm/v3/objects/contacts/") and request.method == "PATCH":
            cid = path.rsplit("/", 1)[1]
            self.contacts[cid].update(body["properties"])
            return httpx.Response(200, json={"id": cid, "properties": self.contacts[cid]})
        if path in ("/crm/v3/objects/notes", "/crm/v3/properties/contacts"):
            return httpx.Response(201, json={"id": "900"})
        return httpx.Response(404, json={"message": "unknown path"})


@pytest.fixture
def hubspot():
    fake = FakeHubSpot()
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    crm = HubSpotCRM("token-1", ["target_exam", "city"], http=httpx.AsyncClient(transport=httpx.MockTransport(fake)), sleep=sleep)
    return crm, fake, waits


async def test_hubspot_create_then_update_never_blanks_a_field(hubspot):
    crm, fake, _ = hubspot
    assert await crm.find_by_phone("+919876543210") is None

    created = await crm.upsert_lead("+919876543210", "Asha", {"target_exam": "JEE", "city": "", "not_a_field": "x"})
    assert created.id == "500" and created.fields == {"target_exam": "JEE"} and created.stage == "new"
    assert fake.contacts["500"] == {"phone": "+919876543210", "hs_lead_status": "NEW", "desk_target_exam": "JEE", "firstname": "Asha"}

    updated = await crm.upsert_lead("+919876543210", "", {"target_exam": "", "city": "Noida"})
    assert updated.id == "500" and updated.name == "Asha" and updated.fields == {"target_exam": "JEE", "city": "Noida"}
    assert fake.requests[-1] == ("PATCH", "/crm/v3/objects/contacts/500", {"properties": {"desk_city": "Noida"}})

    found = await crm.find_by_phone("+919876543210")
    assert found.fields == {"target_exam": "JEE", "city": "Noida"}
    search = fake.requests[-1][2]
    assert search["filterGroups"] == [{"filters": [{"propertyName": "phone", "operator": "EQ", "value": "+919876543210"}]}]
    assert "desk_city" in search["properties"]


async def test_hubspot_note_and_stage(hubspot):
    crm, fake, _ = hubspot
    await crm.add_note("500", "Counselling call booked.")
    _, path, body = fake.requests[-1]
    assert path == "/crm/v3/objects/notes" and body["properties"]["hs_note_body"] == "Counselling call booked."
    assert body["associations"] == [{"to": {"id": "500"}, "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 202}]}]

    fake.contacts["500"] = {"phone": "+919876543210"}
    await crm.set_stage("500", "counselling_booked")
    assert fake.contacts["500"]["hs_lead_status"] == "IN_PROGRESS"
    assert (await crm.find_by_phone("+919876543210")).stage == "counselling_booked"


async def test_hubspot_rate_limit_is_waited_out_and_other_errors_are_raised(hubspot):
    crm, fake, waits = hubspot
    fake.fail_next = [httpx.Response(429, headers={"retry-after": "3"}, json={}), httpx.Response(502, text="bad gateway")]
    assert await crm.find_by_phone("+919876543210") is None
    assert waits == [3.0, 2.0] and len(fake.requests) == 3

    fake.fail_next = [httpx.Response(401, json={"message": "token expired"})]
    with pytest.raises(HubSpotError) as raised:
        await crm.find_by_phone("+919876543210")
    assert raised.value.status == 401

    fake.fail_next = [httpx.Response(409, json={"message": "exists"})]
    assert await crm.ensure_properties() == ["city"]  # the first already existed


# ---- the endpoints --------------------------------------------------------------------------


def signed(secret: str, body: bytes, timestamp: str) -> dict:
    return {"X-Desk-Timestamp": timestamp, "X-Desk-Signature": signing.sign(secret, timestamp, body), "Content-Type": "application/json"}


def test_lead_webhook_refuses_unsigned_accepts_signed_and_ignores_repeats(world, monkeypatch):
    monkeypatch.setattr("enquiry_desk.api.time.time", lambda: 1000.0)
    body = json.dumps(FORM).encode()
    with TestClient(create_app(world, run_worker=False)) as client:
        assert client.post("/webhooks/lead", content=body).status_code == 401
        assert client.post("/webhooks/lead", content=body, headers=signed("wrong", body, "1000")).status_code == 401
        assert client.post("/webhooks/lead", content=body, headers=signed("form-secret", body, "100")).status_code == 401

        first = client.post("/webhooks/lead", content=body, headers=signed("form-secret", body, "1000"))
        again = client.post("/webhooks/lead", content=body, headers=signed("form-secret", body, "1000"))
        assert (first.status_code, first.json()) == (202, {"accepted": True, "duplicate": False})
        assert (again.status_code, again.json()) == (202, {"accepted": True, "duplicate": True})

        bad = json.dumps({"phone": "9876543210"}).encode()  # no event id
        assert client.post("/webhooks/lead", content=bad, headers=signed("form-secret", bad, "1000")).status_code == 422
        assert client.get("/v1/jobs").json()["counts"]["queued"] == 1


async def test_hubspot_webhook_queues_each_event_once(world, monkeypatch):
    monkeypatch.setattr("enquiry_desk.api.time.time", lambda: 1_700_000_000.0)
    events = [
        {"eventId": 1, "subscriptionType": "contact.propertyChange", "objectId": 500, "propertyName": "hs_lead_status", "propertyValue": "CONNECTED", "occurredAt": 1_700_000_000_000},
        {"eventId": 1, "subscriptionType": "contact.propertyChange", "objectId": 500},
        {"eventId": 2, "subscriptionType": "contact.creation", "objectId": 501, "occurredAt": 1_700_000_001_000},
    ]
    body = json.dumps(events).encode()
    stamp = "1700000000000"
    good = signing.hubspot_v3("hub-secret", "POST", "https://desk.test/webhooks/hubspot", body, stamp)
    headers = {"X-HubSpot-Request-Timestamp": stamp, "Content-Type": "application/json"}
    with TestClient(create_app(world, run_worker=False)) as client:
        assert client.post("/webhooks/hubspot", content=body, headers={**headers, "X-HubSpot-Signature-v3": "AAAA"}).status_code == 401
        answer = client.post("/webhooks/hubspot", content=body, headers={**headers, "X-HubSpot-Signature-v3": good})
        assert answer.json() == {"received": 3, "new": 2}
    while await world.worker.run_one():
        pass
    stored = world.db.query("SELECT event_id, kind, property, value FROM crm_events ORDER BY event_id")
    assert [tuple(dict(row).values()) for row in stored] == [("1", "contact.propertyChange", "hs_lead_status", "CONNECTED"), ("2", "contact.creation", None, None)]


def test_the_operator_endpoint_can_be_locked(world):
    world.settings.operator_secret = "ops"
    with TestClient(create_app(world, run_worker=False)) as client:
        assert client.post("/v1/sessions/s1/operator", json={"message": "hi"}).status_code == 401
        assert client.post("/v1/sessions/s1/operator", json={"message": "hi"}, headers={"X-Desk-Secret": "ops"}).status_code == 409


def test_job_dataclass_is_plain():
    assert Job(1, "lead", {"a": 1}, 2).attempts == 2
