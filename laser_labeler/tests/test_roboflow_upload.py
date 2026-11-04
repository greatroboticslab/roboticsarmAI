import io
import json
import urllib.error
import urllib.parse

import pytest

from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.roboflow.upload import add_tags, build_upload_plan, upload_annotations

KEY = "rf_SECRETKEY_123456"


def label_set(n, status="approved", pdf="data/pdfs/img.pdf", prompt="v7", box=(10, 10, 40, 30)):
    return ImageLabelSet(
        image_id=f"img{n}", image_name=f"img{n}.jpg", camera_id="cam", is_grayscale=False,
        background_level=0.1, laser_boxes=[LaserBox("l1", "first_incidence", box, 0,
                                                    metrics={"peak_intensity": 0.987654})],
        object_boxes=[ObjectBox("o1", (50, 50, 20, 20), "table", "top", "wood", "brown", "first",
                                laser_ids=["l1"], confidence=0.9)],
        prompt_version=prompt, provider="p", model="m", pdf_path=pdf, status=status,
        image_width=640, image_height=480)


class Resp:
    def __init__(self, body=None): self.body = json.dumps(body or {"success": True}).encode()
    def read(self): return self.body
    def close(self): pass


class Recorder:
    """Fake opener: records requests, can fail chosen calls."""
    def __init__(self, fail=None):
        self.reqs, self.fail = [], fail or {}
    def __call__(self, req, timeout=None):
        self.reqs.append(req)
        for needle, code in self.fail.items():
            if needle in req.full_url:
                raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b'{"error":"boom"}'))
        return Resp()


def plan_of(*sets, **kw):
    return build_upload_plan(sets, workspace="ws", project="pj", **kw)


def test_plan_excludes_non_approved_and_has_tags():
    plan = plan_of(label_set(1), label_set(2, "pending"), label_set(3, "rejected"),
                   label_set(4, "edited"), label_set(5, "uploaded"))
    assert [i.image_id for i in plan.items] == ["img1", "img4"]
    assert set(plan.excluded) == {"img2", "img3", "img5"}
    assert plan.items[0].tags == ["pdf:img.pdf", "prompt:v7"]
    assert plan.items[0].class_names[0].startswith(("las_", "obj_"))


def test_plan_tags_omit_missing_pdf_and_prompt():
    item = plan_of(label_set(1, pdf=None, prompt="")).items[0]
    assert item.tags == []


def test_plan_describe_has_no_metrics_or_links():
    ls = label_set(1)
    ls.laser_boxes[0].notes = "secret-note"
    text = plan_of(ls).describe()
    assert "0.987654" not in text and "peak_intensity" not in text
    assert "laser_ids" not in text and "from_id" not in text and "secret-note" not in text
    assert "las_first-incidence" in text and "pdf:img.pdf" in text and "(10, 10, 40, 30)" in text


def test_plan_excludes_unknown_size_and_empty_box():
    no_size = label_set(1)
    no_size.image_width = 0
    plan = plan_of(no_size, label_set(2, box=(10, 10, 0, 5)))
    assert plan.items == [] and set(plan.excluded) == {"img1", "img2"}


def test_plan_overwrite_flag_from_remote_labels():
    plan = plan_of(label_set(1), label_set(2), label_set(3),
                   remote_labels={"img1": ["a"], "img2": []})
    assert [i.would_overwrite for i in plan.items] == [True, False, None]
    assert "SKIPPED" in plan.describe()


def test_dry_run_sends_nothing_and_prints(tmp_path):
    op, lines = Recorder(), []
    log = tmp_path / "audit.jsonl"
    summary = upload_annotations(plan_of(label_set(1)), KEY, opener=op, audit_log=log, out=lines.append)
    assert op.reqs == [] and summary.dry_run and summary.uploaded == []
    assert not log.exists() and "img1" in lines[0] and KEY not in lines[0]


def test_real_upload_sends_coco_as_text_and_tags(tmp_path):
    op = Recorder()
    log = tmp_path / "audit.jsonl"
    summary = upload_annotations(plan_of(label_set(1)), KEY, dry_run=False, opener=op, audit_log=log)
    assert summary.uploaded == ["img1"] and not summary.failed
    ann, tags = op.reqs
    q = urllib.parse.urlparse(ann.full_url)
    assert q.path == "/dataset/pj/annotate/img1"
    params = urllib.parse.parse_qs(q.query)
    assert params["name"] == ["_annotations.coco.json"] and "overwrite" not in params
    assert ann.get_header("Content-type") == "text/plain" and ann.get_method() == "POST"
    coco = json.loads(ann.data)
    assert len(coco["annotations"]) == 2 and coco["images"][0]["width"] == 640
    assert "peak_intensity" not in ann.data.decode()
    assert urllib.parse.urlparse(tags.full_url).path == "/ws/pj/images/img1/tags"
    assert json.loads(tags.data) == {"operation": "add", "tags": ["pdf:img.pdf", "prompt:v7"]}


def test_overwrite_param_only_when_plan_says_so(tmp_path):
    op = Recorder()
    upload_annotations(plan_of(label_set(1), overwrite=True, add_to_dataset=False), KEY,
                       dry_run=False, opener=op, audit_log=tmp_path / "a.jsonl")
    params = urllib.parse.parse_qs(urllib.parse.urlparse(op.reqs[0].full_url).query)
    assert params["overwrite"] == ["true"] and params["addToDataset"] == ["false"]


def test_existing_annotation_skipped_when_overwrite_off(tmp_path):
    op = Recorder()
    plan = plan_of(label_set(1), remote_labels={"img1": ["a"]})
    summary = upload_annotations(plan, KEY, dry_run=False, opener=op, audit_log=tmp_path / "a.jsonl")
    assert op.reqs == [] and "img1" in summary.skipped


def test_guard_skips_unapproved_status_even_if_in_plan(tmp_path):
    plan = plan_of(label_set(1))
    plan.items[0].status = "pending"
    op = Recorder()
    summary = upload_annotations(plan, KEY, dry_run=False, opener=op, audit_log=tmp_path / "a.jsonl")
    assert op.reqs == [] and "img1" in summary.skipped


def test_failure_on_image_2_does_not_stop_image_3(tmp_path):
    op = Recorder(fail={"/annotate/img2": 400})
    log = tmp_path / "audit.jsonl"
    summary = upload_annotations(plan_of(label_set(1), label_set(2), label_set(3)), KEY,
                                 dry_run=False, opener=op, audit_log=log)
    assert summary.uploaded == ["img1", "img3"] and list(summary.failed) == ["img2"]
    assert "400" in summary.failed["img2"] and KEY not in summary.failed["img2"]
    assert not any("/tags" in r.full_url and "img2" in r.full_url for r in op.reqs)
    results = [json.loads(l)["result"] for l in log.read_text().splitlines()]
    assert results == ["uploaded", "failed", "uploaded"]


def test_409_message_mentions_overwrite(tmp_path):
    op = Recorder(fail={"/annotate/img1": 409})
    summary = upload_annotations(plan_of(label_set(1)), KEY, dry_run=False, opener=op,
                                 audit_log=tmp_path / "a.jsonl")
    assert "overwrite is off" in summary.failed["img1"]


def test_audit_log_lines_have_no_key_and_right_fields(tmp_path):
    op = Recorder(fail={"/annotate/img2": 500})
    log = tmp_path / "sub" / "audit.jsonl"
    upload_annotations(plan_of(label_set(1), label_set(2)), KEY, dry_run=False, opener=op,
                       audit_log=log, sleep=lambda s: None)
    text = log.read_text()
    assert KEY not in text and "api_key" not in text
    first = json.loads(text.splitlines()[0])
    assert {"time", "image_id", "classes", "tags", "result"} <= set(first)
    assert first["image_id"] == "img1" and "peak_intensity" not in text


def test_429_retried_with_backoff(tmp_path):
    state = {"n": 0}
    def opener(req, timeout=None):
        if "/annotate/" in req.full_url:
            state["n"] += 1
            if state["n"] <= 2:
                raise urllib.error.HTTPError(req.full_url, 429, "x", {}, io.BytesIO(b""))
        return Resp()
    sleeps = []
    summary = upload_annotations(plan_of(label_set(1)), KEY, dry_run=False, opener=opener,
                                 audit_log=tmp_path / "a.jsonl", sleep=sleeps.append)
    assert summary.uploaded == ["img1"] and state["n"] == 3
    assert sleeps == pytest.approx(sleeps) and len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_tag_failure_is_reported_but_annotation_counts(tmp_path):
    op = Recorder(fail={"/tags": 500, "/metadata": 500})
    summary = upload_annotations(plan_of(label_set(1)), KEY, dry_run=False, opener=op,
                                 audit_log=tmp_path / "a.jsonl", sleep=lambda s: None)
    assert summary.uploaded == ["img1"] and "img1" in summary.tag_failed and not summary.failed


def test_tags_fall_back_to_metadata_route_on_404():
    op = Recorder(fail={"/tags": 404})
    add_tags("ws", "pj", "img1", ["a"], KEY, opener=op)
    assert urllib.parse.urlparse(op.reqs[1].full_url).path == "/ws/images/img1/metadata"
    assert json.loads(op.reqs[1].data) == {"addTags": ["a"]}
