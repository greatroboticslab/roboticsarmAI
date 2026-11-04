import io
import json
import urllib.error

import pytest

from core.roboflow.auth import (RoboflowError, ProjectInfo, list_projects, mask_url,
                                parse_workspace_ids, verify_roboflow_key)

KEY = "rf_SECRETKEY_123456"


class Resp:
    def __init__(self, body): self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
    def read(self): return self.body
    def close(self): pass


def opener_returning(body, seen=None):
    def opener(req, timeout=None):
        if seen is not None:
            seen.append((req.full_url, timeout))
        return Resp(body)
    return opener


def opener_raising(exc):
    def opener(req, timeout=None):
        raise exc
    return opener


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "msg", {}, io.BytesIO(b""))


def test_valid_key_with_workspace_string():
    seen = []
    r = verify_roboflow_key(KEY, opener_returning({"workspace": "my-ws"}, seen), timeout=7)
    assert (r.ok, r.state, r.workspace_ids) == (True, "valid", ["my-ws"])
    assert seen[0][0].startswith("https://api.roboflow.com/?api_key=") and seen[0][1] == 7


@pytest.mark.parametrize("body,expected", [
    ({"workspace": ["a", "b"]}, ["a", "b"]),
    ({"workspace": [{"url": "a"}, {"id": "b"}, 5]}, ["a", "b"]),
    ({"workspace": {"url": "a"}}, ["a"]),
    ({}, []),
    ({"workspace": None}, []),
    ([1, 2], []),
])
def test_workspace_shapes(body, expected):
    assert parse_workspace_ids(body) == expected


def test_valid_key_without_workspace():
    r = verify_roboflow_key(KEY, opener_returning({"something": 1}))
    assert r.ok and r.workspace_ids == []


@pytest.mark.parametrize("code", [401, 403])
def test_rejected_key_is_invalid(code):
    r = verify_roboflow_key(KEY, opener_raising(http_error(code)))
    assert (r.ok, r.state) == (False, "invalid")


@pytest.mark.parametrize("exc", [
    http_error(500), http_error(503), urllib.error.URLError("no route"),
    TimeoutError("slow"), ConnectionResetError("reset"),
])
def test_network_trouble_is_unreachable_not_wrong_key(exc):
    r = verify_roboflow_key(KEY, opener_raising(exc))
    assert (r.ok, r.state) == (False, "unreachable")
    assert "could not be checked" in r.message.lower()
    assert "wrong" not in r.message.lower()


def test_non_json_is_unreachable():
    r = verify_roboflow_key(KEY, opener_returning(b"<html>oops</html>"))
    assert r.state == "unreachable" and "not JSON" in r.message


def test_key_never_in_messages_or_repr():
    exc = urllib.error.URLError(f"failed for api_key={KEY}")
    r = verify_roboflow_key(KEY, opener_raising(exc))
    assert KEY not in r.message and KEY not in repr(r)
    with pytest.raises(RoboflowError) as info:
        list_projects("ws", KEY, opener_raising(exc))
    assert KEY not in str(info.value) and KEY not in repr(info.value)


def test_mask_url():
    assert mask_url(f"https://api.roboflow.com/?api_key={KEY}", KEY) == "https://api.roboflow.com/?api_key=****"


def test_list_projects_strips_workspace_prefix():
    body = {"workspace": {"projects": [{"id": "ws/proj-a", "name": "A", "type": "object-detection"},
                                       {"id": "plain", "name": "", "type": None}]}}
    seen = []
    out = list_projects("ws", KEY, opener_returning(body, seen))
    assert out == [ProjectInfo("proj-a", "A", "object-detection"), ProjectInfo("plain", "plain", "")]
    assert seen[0][0].startswith("https://api.roboflow.com/ws?api_key=")


@pytest.mark.parametrize("body", [{}, [], {"workspace": "x"}, {"workspace": {"projects": "no"}},
                                  {"workspace": {"projects": [1, {"name": "no id"}]}}])
def test_list_projects_odd_shapes_give_empty(body):
    assert list_projects("ws", KEY, opener_returning(body)) == []


@pytest.mark.parametrize("exc", [http_error(401), http_error(500), urllib.error.URLError("down"),
                                 TimeoutError("t")])
def test_list_projects_errors(exc):
    with pytest.raises(RoboflowError):
        list_projects("ws", KEY, opener_raising(exc))


def test_list_projects_non_json():
    with pytest.raises(RoboflowError):
        list_projects("ws", KEY, opener_returning(b"nope"))
