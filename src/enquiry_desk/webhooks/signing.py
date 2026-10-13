"""How the desk knows a webhook came from who it claims to, and was not replayed.

A webhook is an HTTP request from the outside. Anyone who learns the URL can send one. The
sender therefore proves itself with a shared secret: it computes an HMAC of the request with
the secret, and the receiver computes the same and compares.

Three schemes are here: the desk's own (for the website form and for alerts it sends out),
HubSpot's version 3, and Twilio's.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

TOLERANCE_SECONDS = 300


class BadSignature(Exception):
    """The request is not accepted. The reason is for the log, never for the response."""


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """The desk's own scheme: sha256 of "<timestamp>.<raw body>", in hex.

    The timestamp is inside the signed text. If it were not, an attacker who captured one
    valid request could send it again later with a fresh timestamp.
    """
    digest = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify(secret: str, timestamp: str | None, signature: str | None, body: bytes, now: float) -> None:
    """Raise BadSignature unless the request is genuine and recent.

    `body` must be the raw bytes as received. Parsing the JSON and serialising it again can
    change spacing or key order, and then a genuine signature no longer matches.
    """
    if not secret:
        raise BadSignature("no secret is configured, so nothing can be verified")
    if not timestamp or not signature:
        raise BadSignature("missing header")
    _check_age(timestamp, now)
    # compare_digest takes the same time whether the first character differs or the last.
    # A plain == stops at the first difference, and that timing leaks the signature bit by bit.
    if not hmac.compare_digest(sign(secret, timestamp, body), signature):
        raise BadSignature("mismatch")


def _check_age(timestamp: str, now: float, *, milliseconds: bool = False) -> None:
    try:
        sent = int(timestamp) / (1000 if milliseconds else 1)
    except ValueError:
        raise BadSignature("timestamp is not a number") from None
    if abs(now - sent) > TOLERANCE_SECONDS:
        raise BadSignature("stale")


def hubspot_v3(client_secret: str, method: str, url: str, body: bytes, timestamp: str) -> str:
    """HubSpot signs method + full URL + body + timestamp with the app's client secret."""
    message = method.upper().encode() + url.encode() + body + timestamp.encode()
    return base64.b64encode(hmac.new(client_secret.encode(), message, hashlib.sha256).digest()).decode()


def verify_hubspot(
    client_secret: str, method: str, url: str, body: bytes, timestamp: str | None, signature: str | None, now: float
) -> None:
    if not client_secret:
        raise BadSignature("no secret is configured, so nothing can be verified")
    if not timestamp or not signature:
        raise BadSignature("missing header")
    _check_age(timestamp, now, milliseconds=True)  # HubSpot's timestamp is in milliseconds
    if not hmac.compare_digest(hubspot_v3(client_secret, method, url, body, timestamp), signature):
        raise BadSignature("mismatch")


def twilio(auth_token: str, url: str, params: dict[str, str]) -> str:
    """Twilio signs the full URL followed by every form field, name then value, sorted by name.

    It uses HMAC-SHA1 and has no timestamp, so a captured request can be replayed. The desk
    guards against that by ignoring a call id it has already seen.
    """
    message = url + "".join(name + params[name] for name in sorted(params))
    return base64.b64encode(hmac.new(auth_token.encode(), message.encode(), hashlib.sha1).digest()).decode()


def verify_twilio(auth_token: str, url: str, params: dict[str, str], signature: str | None) -> None:
    if not auth_token:
        raise BadSignature("no secret is configured, so nothing can be verified")
    if not signature:
        raise BadSignature("missing header")
    if not hmac.compare_digest(twilio(auth_token, url, params), signature):
        raise BadSignature("mismatch")
