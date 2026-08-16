"""Auth, CSRF and rate limiting.

Sized for the actual threat model: a small deployed group, public on the internet, no
payments and no personal data beyond an email address. That means real password hashing
and real CSRF, but no email-verification flow or account recovery — if someone forgets
their password an admin resets it.

Everything here is deliberately boring. The interesting code in this repo is elsewhere.
"""
from __future__ import annotations

import hmac
import os
import re
import secrets
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from fastapi import HTTPException, Request, status

_hasher = PasswordHasher()          # argon2id, library defaults

DISPLAY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{1,30}$")
MIN_PASSWORD_LEN = 10


def secret_key() -> str:
    """Session signing key.

    In production this MUST come from the environment — a generated key means every
    restart silently logs everyone out, and on a multi-process deploy the workers would
    not agree on it at all. ``judge/app.py`` refuses to start in production without it.
    """
    return os.environ.get("JUDGE_SECRET_KEY") or secrets.token_urlsafe(48)


def is_production() -> bool:
    return os.environ.get("JUDGE_ENV", "dev").lower() in {"prod", "production"}


# --------------------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except Exception:
        return False


def validate_signup(email: str, display_name: str, password: str) -> str | None:
    """Return an error message, or None if the input is acceptable."""
    from email_validator import EmailNotValidError, validate_email

    try:
        validate_email(email, check_deliverability=False)
    except EmailNotValidError as e:
        return f"That email address doesn't look valid: {e}"
    if not DISPLAY_NAME_RE.match(display_name or ""):
        return ("Display name must be 2–31 characters: letters, numbers, spaces, dots, "
                "underscores or hyphens, starting with a letter or number.")
    if len(password or "") < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if password.lower() in {"password12", "1234567890", "qwertyuiop"}:
        return "Pick a less guessable password."
    return None


# --------------------------------------------------------------------------------------
# CSRF — double-submit, token bound to the signed session cookie
# --------------------------------------------------------------------------------------
def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf"] = token
    return token


def require_csrf(request: Request, submitted: str | None) -> None:
    expected = request.session.get("csrf")
    if not expected or not submitted or not hmac.compare_digest(expected, submitted):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "CSRF check failed — reload the page and try again.")


# --------------------------------------------------------------------------------------
# rate limiting — in-process, which is the right size for one small deployment
# --------------------------------------------------------------------------------------
_hits: dict[str, deque[float]] = defaultdict(deque)


def rate_limit(key: str, limit: int, window_s: int) -> None:
    """Sliding window. Raises 429 when ``limit`` is exceeded within ``window_s``.

    In-process state, so it resets on restart and does not coordinate across workers.
    Fine for throttling password guesses and accidental submit-spam on a single small
    instance; if this ever runs multi-worker, move it to the database or Redis.
    """
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        retry = int(window_s - (now - q[0])) + 1
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Too many attempts. Try again in {retry}s.",
            headers={"Retry-After": str(retry)},
        )
    q.append(now)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
