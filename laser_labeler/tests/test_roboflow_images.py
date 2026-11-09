import io
import json
import urllib.error

import pytest

from core.roboflow.auth import RoboflowError
from core.roboflow.http import RoboflowHTTPError, RoboflowRateLimited
from core.roboflow.images import (RemoteImage, count_images, download_image, is_unlabeled,
                                  list_images, parse_image, safe_stem)

KEY = "rf_SECRETKEY_123456"


class Resp:
    def __init__(self, body):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
    def read(self, n=-1): return self.body
    def close(self): pass


def row(i, labels=("a",), split="train"):
    r = {"id": f"img{i}", "name": f"n{i}.jpg", "split": split, "tags": ["t"], "created": 1.0}
    if labels is not None:
        r["labels"] = list(labels)
    return r


class Server:
    """Fake search endpoint with a server-side page cap; records every request."""
    def __init__(self, rows, cap=3, total=True, honor_offset=True):
        self.rows, self.cap, self.total, self.honor = rows, cap, total, honor_offset
        self.calls = []
    def __call__(self, req, timeout=None):
        body = json.loads(req.data)
        self.calls.append((req.full_url, body))
        off = body["offset"] if self.honor else 0
        page = self.rows[off:off + min(body["limit"], self.cap)]
        out = {"results": page}
        if self.total:
            out["total"] = len(self.rows)
        return Resp(out)


def ids(gen): return [i.id for i in gen]


def test_paging_follows_offset_and_stops_at_total():
    srv = Server([row(i) for i in range(7)])
    got = ids(list_images("ws", "pj", KEY, opener=srv))
    assert got == [f"img{i}" for i in range(7)]
    assert [c[1]["offset"] for c in srv.calls] == [0, 3, 6]
    assert "labels" in srv.calls[0][1]["fields"]
    assert srv.calls[0][0].startswith("https://api.roboflow.com/ws/pj/search?api_key=")


def test_paging_stops_on_empty_page_without_total():
    srv = Server([row(i) for i in range(4)], cap=3, total=False)
    assert len(ids(list_images("ws", "pj", KEY, opener=srv))) == 4
    assert len(srv.calls) == 3  # 3 rows, 1 row, then an empty page


def test_paging_stops_when_server_ignores_offset():
    srv = Server([row(i) for i in range(5)], cap=3, total=False, honor_offset=False)
    assert len(ids(list_images("ws", "pj", KEY, opener=srv))) == 3
    assert len(srv.calls) == 2


def test_limit_and_tag_filter():
    srv = Server([row(i) for i in range(9)])
    assert len(ids(list_images("ws", "pj", KEY, limit=4, tag="review", opener=srv))) == 4
    assert srv.calls[0][1]["tag"] == "review" and len(srv.calls) == 2


def test_only_unlabeled_and_split():
    rows = [row(0, ["a"]), row(1, []), row(2, None), row(3, [], split="valid")]
    srv = Server(rows, cap=10)
    assert ids(list_images("ws", "pj", KEY, only_unlabeled=True, opener=srv)) == ["img1", "img2", "img3"]
    assert ids(list_images("ws", "pj", KEY, split="valid", opener=Server(rows, cap=10))) == ["img3"]


def test_is_unlabeled_three_states():
    assert is_unlabeled(RemoteImage("a", labels=[])) is True
    assert is_unlabeled(RemoteImage("a", labels=["x"])) is False
    assert is_unlabeled(parse_image({"id": "a"})) is None  # labels not reported


def test_parse_image_ignores_junk():
    assert parse_image({"name": "no id"}) is None and parse_image("x") is None
    img = parse_image({"id": 5, "labels": [{"name": "a"}, "b"], "urls": {"original": "https://h/x.png"},
                       "width": 640, "height": 480})
    assert (img.id, img.labels, img.url, img.width) == ("5", ["a", "b"], "https://h/x.png", 640)


def test_count_uses_total_with_one_request():
    srv = Server([row(i) for i in range(7)])
    assert count_images("ws", "pj", KEY, opener=srv) == 7 and len(srv.calls) == 1


def test_count_unlabeled_pages_through():
    srv = Server([row(0, []), row(1, ["a"]), row(2, []), row(3, [])], cap=2)
    assert count_images("ws", "pj", KEY, only_unlabeled=True, opener=srv) == 3


def test_429_is_retried_with_backoff_sleep():
    srv = Server([row(0)])
    state = {"n": 0}
    def flaky(req, timeout=None):
        state["n"] += 1
        if state["n"] <= 2:
            raise urllib.error.HTTPError(req.full_url, 429, "slow", {"Retry-After": "2"}, io.BytesIO(b""))
        return srv(req, timeout)
    sleeps = []
    assert ids(list_images("ws", "pj", KEY, opener=flaky, sleep=sleeps.append)) == ["img0"]
    assert len(sleeps) == 2 and all(2 <= s <= 2.5 for s in sleeps)


def test_401_gives_key_rejected_without_key():
    def deny(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b""))
    with pytest.raises(RoboflowHTTPError) as exc:
        list(list_images("ws", "pj", KEY, opener=deny))
    assert "rejected" in str(exc.value) and KEY not in str(exc.value) and exc.value.status == 401


def test_persistent_429_raises_rate_limited_after_attempts():
    def always(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "slow", {}, io.BytesIO(b""))
    sleeps = []
    with pytest.raises(RoboflowRateLimited):
        list(list_images("ws", "pj", KEY, opener=always, sleep=sleeps.append))
    assert len(sleeps) == 4


# ---- download ----

def file_opener(data=b"IMGDATA", seen=None):
    def opener(req, timeout=None):
        if seen is not None:
            seen.append(req.full_url)
        return Resp(data)
    return opener


def test_download_sanitises_name_and_leaves_no_part_file(tmp_path):
    img = RemoteImage("../../etc/pass wd", name="../../evil.png", url="https://cdn.example/a/b.jpg?sig=1")
    path = download_image(img, tmp_path / "dl", KEY, file_opener())
    assert path.parent == tmp_path / "dl" and path.read_bytes() == b"IMGDATA"
    assert path.name == "______etc_pass_wd.jpg" and "/" not in path.name
    assert [p.name for p in (tmp_path / "dl").iterdir()] == [path.name]


def test_download_is_resume_safe(tmp_path):
    img = RemoteImage("abc", url="https://cdn.example/x.png")
    seen = []
    first = download_image(img, tmp_path, KEY, file_opener(seen=seen))
    again = download_image(img, tmp_path, KEY, file_opener(seen=seen))
    assert first == again and len(seen) == 1
    first.write_bytes(b"")  # a broken zero-byte file is fetched again
    download_image(img, tmp_path, KEY, file_opener(b"NEW", seen))
    assert first.read_bytes() == b"NEW" and len(seen) == 2


def test_download_never_sends_key_to_file_host(tmp_path):
    seen = []
    download_image(RemoteImage("abc", url="https://cdn.example/x.png"), tmp_path, KEY, file_opener(seen=seen))
    assert seen == ["https://cdn.example/x.png"]


def test_download_looks_up_url_when_missing(tmp_path):
    seen = []
    def opener(req, timeout=None):
        seen.append(req.full_url)
        if "/images/abc" in req.full_url:
            return Resp({"image": {"id": "abc", "urls": {"original": "https://cdn.example/o.jpeg"}}})
        return Resp(b"BYTES")
    img = RemoteImage("abc", workspace="ws", project="pj")
    assert download_image(img, tmp_path, KEY, opener).name == "abc.jpeg"
    assert seen[0].startswith("https://api.roboflow.com/ws/pj/images/abc?api_key=")


def test_download_refuses_non_https_and_unusable_ids(tmp_path):
    with pytest.raises(RoboflowError):
        download_image(RemoteImage("abc", url="http://cdn.example/x.jpg"), tmp_path, KEY, file_opener())
    with pytest.raises(RoboflowError):
        safe_stem("..")
    with pytest.raises(RoboflowError):
        download_image(RemoteImage("abc"), tmp_path, KEY, file_opener())  # no url, no workspace


# ---------------- unlabeled means "not labeled in the current scheme" when a scheme is given
def test_is_unlabeled_by_scheme():
    from core.roboflow.images import RemoteImage, is_unlabeled
    assert is_unlabeled(RemoteImage("a", labels=["component1"]), "components") is False
    assert is_unlabeled(RemoteImage("a", labels=["component1"]), "full") is True          # an older kind of label
    assert is_unlabeled(RemoteImage("a", labels=["component1"])) is False                 # no scheme: any label counts
    assert is_unlabeled(RemoteImage("a", labels=[]), "full") is True
    assert is_unlabeled(RemoteImage("a"), "full") is None


def test_list_images_only_unlabeled_with_a_scheme():
    from core.roboflow.images import list_images
    rows = [{"id": "x", "labels": ["las_first-incidence"]}, {"id": "y", "labels": ["component1"]},
            {"id": "z", "labels": []}, {"id": "w"}]

    def opener(req, timeout=None):
        import io, json
        return io.BytesIO(json.dumps({"results": rows if json.loads(req.data)["offset"] == 0 else [], "total": 4}).encode())

    got = [i.id for i in list_images("ws", "pr", "k", only_unlabeled=True, scheme="full", opener=opener)]
    assert got == ["y", "z", "w"]
