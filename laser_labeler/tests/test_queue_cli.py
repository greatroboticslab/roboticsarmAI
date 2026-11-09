import json
from pathlib import Path

import pytest
from PIL import Image

from cli import cmd_queue
from cli.__main__ import main
from config import credentials
from core.queue_db import NewImage, QueueDB
from core.reports.manifest import read_manifest
from core.reports.sidecar import read_sidecar
from core.roboflow.auth import RoboflowError
from core.roboflow.enqueue import enqueue_remote
from core.roboflow.images import RemoteImage
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot


# ---------------- enqueue_remote
@pytest.fixture
def db(tmp_path):
    database = QueueDB(tmp_path / "q.db")
    yield database
    database.close()


def remote(n, **kw):
    return RemoteImage(f"r{n}", f"r{n}.jpg", tags=["camera:a"], **kw)


def test_enqueue_downloads_new_images_only(db, tmp_path):
    db.add_images([NewImage("r1", "r1.jpg", "/old/r1.jpg")])
    fetched = []

    def download(img):
        fetched.append(img.id)
        return tmp_path / f"{img.id}.jpg"

    result = enqueue_remote(db, [remote(1), remote(2, labels=[]), remote(3, labels=["x"]), remote(4)], download)
    assert fetched == ["r2", "r3", "r4"]
    assert (result.added, result.already_known, result.failed) == (3, 1, {})
    assert result.unknown_label_state == 1  # r4: labels absent means unknown
    row = db.get("r2")
    assert row.source == "roboflow" and row.tags == ["camera:a"] and row.path == str(tmp_path / "r2.jpg")


def test_enqueue_limit_counts_new_images_and_failures_are_isolated(db, tmp_path):
    def download(img):
        if img.id == "r2":
            raise RoboflowError("HTTP 404 gone")
        return tmp_path / f"{img.id}.jpg"

    result = enqueue_remote(db, [remote(n) for n in range(1, 6)], download, limit=2)
    assert result.added == 2 and list(result.failed) == ["r2"] and "404" in result.failed["r2"]
    assert db.get("r1") and db.get("r3") and db.get("r4") is None and db.get("r2") is None  # r2 can be retried later


# ---------------- queue CLI, end to end with a scripted model
@pytest.fixture
def folder(tmp_path):
    pics = tmp_path / "pics"
    (pics / "sub").mkdir(parents=True)
    for name in ("a.png", "sub/b.png"):
        img = scene()
        rect(img, 100, 80, 80, 60, (150, 30, 30))
        spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
        Image.fromarray(finish(img)).save(pics / name)
    (pics / "notes.txt").write_text("not an image")
    return pics


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    replies = []

    def good():
        return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                    "from_id": None, "object": PEN_RUBBER}]))

    provider = ScriptedProvider([])
    provider.replies = replies
    monkeypatch.setattr(cmd_queue, "make_provider", lambda settings, overrides=None: provider)
    monkeypatch.setattr(credentials, "require_key", lambda service: "test-key")

    def call(*args):
        code = main(["queue", "--data-dir", str(data), *args])
        captured = capsys.readouterr()
        call.err = captured.err
        return code, captured.out

    call.good, call.replies, call.data, call.provider, call.err = good, replies, data, provider, ""
    return call


def test_add_status_and_duplicates(cli, folder):
    code, out = cli("add", str(folder), "--tag", "camera:a")
    assert code == 0 and "added 2 new image(s)" in out
    assert "added 0 new image(s); 2 already" in cli("add", str(folder))[1]
    status = cli("status")[1]
    assert "queued" in status and status.split("queued")[1].split()[0] == "2"
    assert cli("add", str(folder / "notes.txt"))[0] == 1  # nothing usable to add


def test_same_stem_in_different_folders_gets_unique_ids(cli, tmp_path):
    for sub in ("one", "two"):
        (tmp_path / sub).mkdir()
        Image.fromarray(finish(scene())).save(tmp_path / sub / "same.png")
    cli("add", str(tmp_path / "one"), str(tmp_path / "two"))
    out = cli("list", "--sort", "order")[1]
    assert out.count("same") == 2 and "same-" in out


def test_run_review_approve_export_and_dry_run_upload(cli, folder, tmp_path):
    cli("add", str(folder), "--tag", "camera:a")
    cli.replies += [cli.good(), cli.good()]
    code, out = cli("run")
    assert code == 0 and "labeled 2: 2 ready for review" in out and "queue empty" in out
    data = cli.data
    assert (data / "manifest.csv").is_file() and len(read_manifest(data / "manifest.csv")) == 2
    assert {p.name for p in (data / "sidecars").glob("*.json")} == {"a.json", "b.json"}
    assert read_sidecar(data / "sidecars" / "a.json").status == "pending"

    assert "approved 0" in cli("approve", "--min-confidence", "0.99")[1]  # nothing is that sure
    assert cli("export-coco", "--out", str(tmp_path / "none.json"))[0] == 1  # nothing approved yet
    assert "approved 2" in cli("approve", "--min-confidence", "0.0")[1]
    assert {r["status"] for r in read_manifest(data / "manifest.csv")} == {"approved"}

    out_path = tmp_path / "coco.json"
    assert cli("export-coco", "--out", str(out_path))[0] == 0
    coco = json.loads(out_path.read_text())
    assert len(coco["images"]) == 2 and coco["annotations"]

    code, out = cli("upload", "--workspace", "ws", "--project", "proj")
    assert code == 0 and "DRY RUN" in out
    assert cli("status")[1].count("approved") == 1 and "uploaded" in cli("status")[1]
    assert QueueDB(data / "queue.db").counts()["uploaded"] == 0  # a dry run changes nothing


def test_run_limit_then_resume_and_stats(cli, folder):
    cli("add", str(folder))
    cli.replies.append(cli.good())
    code, out = cli("run", "--limit", "1")
    assert "labeled 1" in out and "1 still queued" in out and "reached the limit" in out
    cli.replies.append(cli.good())
    assert "labeled 1" in cli("run")[1]
    assert len(cli.provider.calls) == 2


def test_failures_stop_the_run_and_retry_failed(cli, folder):
    cli("add", str(folder))
    code, out = cli("run", "--max-failures", "1")  # the scripted model has no replies: every call fails
    assert code == 3 and "stopped after several failures" in out
    listing = cli("list", "--status", "failed")[1]
    assert listing.count("\n") == 1  # only the first image was attempted
    assert "re-queued 1" in cli("retry-failed")[1]
    cli.replies += [cli.good(), cli.good()]
    assert "labeled 2" in cli("run")[1]


def test_run_without_credentials_explains(cli, folder, monkeypatch):
    cli("add", str(folder))

    def refuse(settings, overrides=None):
        raise credentials.MissingCredentialError("No key for gemini. Run `python -m cli credentials set gemini`.")

    monkeypatch.setattr(cmd_queue, "make_provider", refuse)
    assert cli("run")[0] == 1
    assert "cannot start" in cli.err and "credentials set gemini" in cli.err
    assert cli("status")[1].split("queued")[1].split()[0] == "2"  # nothing was touched


def test_upload_needs_project_and_key(cli, folder, monkeypatch):
    assert cli("upload")[0] == 1 and "--workspace and --project" in cli.err

    def refuse(service):
        raise credentials.MissingCredentialError("no roboflow key")

    monkeypatch.setattr(credentials, "require_key", refuse)
    assert cli("upload", "--workspace", "w", "--project", "p")[0] == 1
    assert "no roboflow key" in cli.err


def test_requeue_specific_image(cli, folder):
    cli("add", str(folder))
    cli.replies += [cli.good(), cli.good()]
    cli("run")
    assert "re-queued 1" in cli("requeue", "a")[1]
    assert Path(folder / "a.png").is_file()  # re-labeling never touches the original image


# ---------------- queue clear
def _queued(tmp_path, n=3):
    database = QueueDB(tmp_path / "data" / "queue.db")
    database.add_images([NewImage(f"i{k}", f"i{k}.png", f"/x/i{k}.png") for k in range(n)])
    database.close()


def test_cli_clear_needs_yes_when_nobody_can_answer(tmp_path, monkeypatch, capsys):
    _queued(tmp_path)
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: False})())
    assert main(["queue", "--data-dir", str(tmp_path / "data"), "clear"]) == 1
    assert "--yes" in capsys.readouterr().err
    assert QueueDB(tmp_path / "data" / "queue.db").counts()["queued"] == 3


def test_cli_clear_removes_what_was_asked(tmp_path, capsys):
    _queued(tmp_path)
    assert main(["queue", "--data-dir", str(tmp_path / "data"), "clear", "--yes", "--status", "queued"]) == 0
    assert "removed 3" in capsys.readouterr().out
    assert main(["queue", "--data-dir", str(tmp_path / "data"), "clear", "--yes"]) == 0
    assert "nothing to remove" in capsys.readouterr().out
    assert main(["queue", "--data-dir", str(tmp_path / "data"), "clear", "--yes", "--status", "bogus"]) == 1


# ---------------- waiting out a provider limit on the command line
def _limit_reply(retry_after=None):
    from core.providers.base import RateLimitError

    def reply(messages):
        raise RateLimitError("429 quota exceeded", retry_after)
    return reply


def test_wait_on_limit_sleeps_then_carries_on(cli, folder, monkeypatch):
    slept = []
    monkeypatch.setattr(cmd_queue.time, "sleep", lambda s: slept.append(s))
    cli("add", str(folder))
    cli.replies += [_limit_reply(), cli.good(), cli.good()]
    code, out = cli("run", "--wait-on-limit")
    assert code == 0 and "labeled 2" in out and "Waiting 1 min" in out
    assert slept == [60.0]


def test_wait_on_limit_gives_up_after_many_hits(cli, folder, monkeypatch):
    monkeypatch.setattr(cmd_queue.time, "sleep", lambda s: None)
    monkeypatch.setattr(cmd_queue, "MAX_AUTO_RETRIES", 2)
    cli("add", str(folder))
    cli.replies += [_limit_reply() for _ in range(6)]
    code, out = cli("run", "--wait-on-limit")
    assert code == 3 and "still there after 2 tries" in out and cli("status")[1].split("queued")[1].split()[0] == "2"


def test_without_wait_on_limit_a_non_interactive_run_stops(cli, folder, monkeypatch):
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: False})())
    cli("add", str(folder))
    cli.replies += [_limit_reply(), cli.good()]
    code, out = cli("run")
    assert code == 3 and "rate limit" in out


# ---------------- add-roboflow --unlabeled follows the current labeling scheme
def test_add_roboflow_unlabeled_uses_the_current_scheme_and_notes_older_labels(cli, tmp_path, monkeypatch):
    seen = {}
    images = [RemoteImage("old1", "old1.jpg", labels=["component1", "component3"]), RemoteImage("new1", "new1.jpg", labels=[])]

    def fake_list(ws, proj, key, **kw):
        seen.update(kw)
        return iter(images)

    def fake_download(img, folder, key):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{img.id}.jpg"
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(cmd_queue, "list_images", fake_list)
    monkeypatch.setattr(cmd_queue, "download_image", fake_download)
    code, out = cli("add-roboflow", "--workspace", "ws", "--project", "pr", "--unlabeled")
    assert code == 0 and "added 2" in out
    assert seen["only_unlabeled"] is True and seen["scheme"] == "full"
    assert "1 added image(s) already have labels in an older style" in out and "--overwrite" in out
