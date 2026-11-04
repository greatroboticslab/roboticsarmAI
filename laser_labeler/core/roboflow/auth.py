"""Roboflow key verification and project listing, standard library only.

The response shapes below are NOT verified against the live API, so parsing is
defensive: unexpected shapes give an empty result instead of a crash.
The API key is never put in exception messages, results or reprs.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

API_ROOT = "https://api.roboflow.com"
State = Literal["valid", "invalid", "unreachable"]


class RoboflowError(RuntimeError):
    """A Roboflow request failed. The message never contains the API key."""


@dataclass
class VerifyResult:
    ok: bool
    state: State
    workspace_ids: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class ProjectInfo:
    id: str  # slug without the "workspace/" prefix
    name: str
    type: str


def _url(path: str, api_key: str) -> str:
    return f"{API_ROOT}/{path}?{urllib.parse.urlencode({'api_key': api_key})}"


def mask_url(url: str, api_key: str) -> str:
    """The URL with the key replaced by ****, safe to show or log."""
    quoted = urllib.parse.quote(api_key, safe="")
    return url.replace(quoted, "****").replace(api_key, "****")


def _scrub(text: str, api_key: str) -> str:
    return text.replace(api_key, "****") if api_key else text


def _get_json(url: str, api_key: str, opener: Callable, timeout: float) -> Any:
    """Fetch and decode JSON. Raises HTTPError, URLError/OSError or ValueError."""
    resp = opener(urllib.request.Request(url), timeout=timeout)
    try:
        return json.loads(resp.read().decode("utf-8"))
    finally:
        close = getattr(resp, "close", None)
        if close:
            close()


def _slug(item: Any) -> str | None:
    if isinstance(item, str) and item.strip():
        return item.strip()
    if isinstance(item, dict):
        for name in ("url", "id", "slug"):
            if isinstance(item.get(name), str) and item[name].strip():
                return item[name].strip()
    return None


def parse_workspace_ids(data: Any) -> list[str]:
    """Accept `workspace` as a string, a dict, a list of those, or absent."""
    raw = data.get("workspace") if isinstance(data, dict) else None
    items = raw if isinstance(raw, list) else [raw]
    return [s for s in (_slug(i) for i in items) if s]


def verify_roboflow_key(api_key: str, opener: Callable = urllib.request.urlopen,
                        timeout: float = 15) -> VerifyResult:
    """Check a key against Roboflow. Only a 401/403 answer means the key is wrong."""
    unreachable = "The key could NOT be checked"
    try:
        data = _get_json(_url("", api_key), api_key, opener, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return VerifyResult(False, "invalid", [], "Roboflow rejected this key (HTTP %d)." % exc.code)
        return VerifyResult(False, "unreachable", [],
                            f"{unreachable}: Roboflow answered with HTTP {exc.code}. Try again later.")
    except ValueError:  # includes JSON decode errors
        return VerifyResult(False, "unreachable", [],
                            f"{unreachable}: Roboflow sent an answer that is not JSON.")
    except (urllib.error.URLError, OSError) as exc:
        reason = _scrub(str(getattr(exc, "reason", exc)), api_key)
        return VerifyResult(False, "unreachable", [],
                            f"{unreachable}: no connection to Roboflow ({reason}). Are you offline?")
    ids = parse_workspace_ids(data)
    note = "" if ids else " No workspace name was returned; type it in yourself."
    return VerifyResult(True, "valid", ids, "Roboflow accepted the key." + note)


def parse_projects(data: Any) -> list[ProjectInfo]:
    """Read data['workspace']['projects']; any other shape gives []."""
    ws = data.get("workspace") if isinstance(data, dict) else None
    projects = ws.get("projects") if isinstance(ws, dict) else None
    out = []
    for p in projects if isinstance(projects, list) else []:
        pid = p.get("id") if isinstance(p, dict) else None
        if isinstance(pid, str) and pid.strip():
            slug = pid.strip().rsplit("/", 1)[-1]
            out.append(ProjectInfo(slug, str(p.get("name") or slug), str(p.get("type") or "")))
    return out


def list_projects(workspace_id: str, api_key: str, opener: Callable = urllib.request.urlopen,
                  timeout: float = 15) -> list[ProjectInfo]:
    """Projects of a workspace. Raises RoboflowError on HTTP or network failure."""
    path = urllib.parse.quote(workspace_id.strip(), safe="")
    try:
        data = _get_json(_url(path, api_key), api_key, opener, timeout)
    except urllib.error.HTTPError as exc:
        what = "rejected the key or workspace name" if exc.code in (401, 403, 404) else "had a problem"
        raise RoboflowError(f"Roboflow {what} (HTTP {exc.code}).") from None
    except ValueError:
        raise RoboflowError("Roboflow sent an answer that is not JSON.") from None
    except (urllib.error.URLError, OSError) as exc:
        reason = _scrub(str(getattr(exc, "reason", exc)), api_key)
        raise RoboflowError(f"No connection to Roboflow ({reason}).") from None
    return parse_projects(data)
