"""Tiny shared HTTP helper for the Roboflow REST calls (stdlib only).

Every error message is scrubbed of the API key. Status handling follows the
Roboflow production checklist: only 429 and 5xx are retryable; 400/401/402/403/404
are not. No Retry-After header is documented, so it is only used if present.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, TypeVar

from core.roboflow.auth import API_ROOT, RoboflowError

T = TypeVar("T")


class RoboflowHTTPError(RoboflowError):
    """An HTTP or network failure. `status` is None for network errors."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class RoboflowRateLimited(RoboflowHTTPError):
    """HTTP 429. `retry_after` is seconds from the Retry-After header, if sent."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, 429, True)
        self.retry_after = retry_after


def scrub(text: str, key: str) -> str:
    """Replace every form of the key in text with ****."""
    if not key:
        return text
    return text.replace(urllib.parse.quote(key, safe=""), "****").replace(key, "****")


def build_url(path: str, api_key: str, **params: Any) -> str:
    """https://api.roboflow.com/<path>?api_key=...&<params>; None params are dropped."""
    query = {"api_key": api_key, **{k: v for k, v in params.items() if v is not None}}
    return f"{API_ROOT}/{path.strip('/')}?{urllib.parse.urlencode(query)}"


def quote_seg(value: str) -> str:
    """Quote one URL path segment (workspace, project or image id)."""
    return urllib.parse.quote(str(value).strip(), safe="")


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    try:
        return max(0.0, float(exc.headers.get("Retry-After")))
    except (AttributeError, TypeError, ValueError):
        return None


def _server_note(exc: urllib.error.HTTPError, key: str) -> str:
    """A short, scrubbed hint from the error body (e.g. a 409 reason), or ''."""
    try:
        data = json.loads(exc.read().decode("utf-8", "replace"))
        err = data.get("error", data) if isinstance(data, dict) else None
        text = err.get("message") if isinstance(err, dict) else err
        return ": " + scrub(str(text), key)[:200] if text else ""
    except Exception:  # noqa: BLE001 - body is optional and may be anything
        return ""


def _http_error(exc: urllib.error.HTTPError, where: str, key: str) -> RoboflowHTTPError:
    code = exc.code
    if code in (401, 403):
        return RoboflowHTTPError(f"Roboflow rejected the API key or its permissions (HTTP {code}).", code)
    if code == 429:
        return RoboflowRateLimited(f"Roboflow rate limit reached (HTTP 429) for {where}.", _retry_after(exc))
    note = _server_note(exc, key)
    return RoboflowHTTPError(f"Roboflow answered HTTP {code} for {where}{note}", code, code >= 500)


def _encode(body: Any, headers: dict[str, str]) -> bytes | None:
    if body is None:
        return None
    if isinstance(body, (dict, list)):
        headers.setdefault("Content-Type", "application/json")
        return json.dumps(body).encode("utf-8")
    return body.encode("utf-8") if isinstance(body, str) else bytes(body)


def request_bytes(method: str, url: str, *, body: Any = None,
                  opener: Callable = urllib.request.urlopen, timeout: float = 30,
                  key_for_masking: str = "", headers: dict[str, str] | None = None) -> bytes:
    """Send a request and return the raw answer. dict/list bodies go out as JSON;
    str/bytes bodies go out as given (set Content-Type in `headers`)."""
    hdrs = dict(headers or {})
    where = f"{method} {scrub(urllib.parse.urlparse(url).path, key_for_masking)}"  # no query: no key
    req = urllib.request.Request(url, data=_encode(body, hdrs), method=method, headers=hdrs)
    try:
        resp = opener(req, timeout=timeout)
        try:
            return resp.read()
        finally:
            close = getattr(resp, "close", None)
            if close:
                close()
    except urllib.error.HTTPError as exc:
        raise _http_error(exc, where, key_for_masking) from None
    except (urllib.error.URLError, OSError) as exc:
        reason = scrub(str(getattr(exc, "reason", exc)), key_for_masking)
        raise RoboflowHTTPError(f"No connection to Roboflow ({reason}).", None, True) from None


def request_json(method: str, url: str, *, body: Any = None,
                 opener: Callable = urllib.request.urlopen, timeout: float = 30,
                 key_for_masking: str = "", headers: dict[str, str] | None = None) -> Any:
    """Like request_bytes but parses the answer as JSON (an empty answer gives {})."""
    raw = request_bytes(method, url, body=body, opener=opener, timeout=timeout,
                        key_for_masking=key_for_masking, headers=headers)
    if not raw.strip():
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise RoboflowHTTPError("Roboflow sent an answer that is not JSON.", None, False) from None


def with_backoff(func: Callable[[], T], *, attempts: int = 5, base_delay: float = 1.0,
                 max_delay: float = 30.0, sleep: Callable[[float], None] = time.sleep,
                 jitter: Callable[[], float] = random.random) -> T:
    """Call func, retrying retryable RoboflowHTTPError with exponential backoff.

    Delay is Retry-After when sent, else base*2**n, plus up to 25% jitter, capped.
    """
    for attempt in range(attempts):
        try:
            return func()
        except RoboflowHTTPError as exc:
            if not exc.retryable or attempt == attempts - 1:
                raise
            wait = getattr(exc, "retry_after", None)
            if wait is None:
                wait = base_delay * 2 ** attempt
            sleep(min(max_delay, wait * (1 + 0.25 * jitter())))
    raise AssertionError("unreachable")  # pragma: no cover
