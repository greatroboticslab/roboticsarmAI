import json

import pytest

from core.models import ImageLabelSet
from core.prompts import versions as v
from core.prompts.manager import build_system_prompt
from core.reports.manifest import COLUMNS, read_manifest, set_status, update_manifest
from core.schema.taxonomy import load_taxonomy
from pathlib import Path

TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")


def ls(image_id="a", status="pending", pdf=None):
    return ImageLabelSet(image_id, f"{image_id}.png", "cam", False, 0.1, status=status, pdf_path=pdf,
                         prompt_version="v0001", provider="p", model="m")


# ---------------- manifest
def test_manifest_add_replace_and_order(tmp_path):
    path = tmp_path / "manifest.csv"
    update_manifest(ls("a"), path, "/s/a.json", "2026-01-01T00:00:00+00:00")
    update_manifest(ls("b", pdf="/p/b.pdf"), path, "/s/b.json")
    update_manifest(ls("a", status="approved"), path, "/s/a2.json")
    rows = read_manifest(path)
    assert [r["image_id"] for r in rows] == ["a", "b"] and list(rows[0]) == COLUMNS
    assert rows[0]["status"] == "approved" and rows[0]["sidecar_path"] == "/s/a2.json"
    assert rows[1]["pdf_path"] == "/p/b.pdf" and rows[0]["pdf_path"] == "" and rows[1]["labeled_at"].endswith("+00:00")


def test_manifest_set_status(tmp_path):
    path = tmp_path / "m.csv"
    assert read_manifest(path) == [] and set_status(path, "a", "uploaded") is False
    update_manifest(ls("a"), path, "/s")
    assert set_status(path, "a", "uploaded") is True and read_manifest(path)[0]["status"] == "uploaded"
    assert not list(tmp_path.glob("*.tmp"))


# ---------------- prompt versions
@pytest.fixture
def dirs(tmp_path):
    parts, versions = tmp_path / "parts", tmp_path / "versions"
    parts.mkdir()
    (parts / "01_a.md").write_text("alpha {{table_rule}}\n")
    (parts / "02_b.md").write_text("beta\n")
    return parts, versions


def test_numbering_reuse_and_immutability(dirs):
    parts, versions = dirs
    assert v.save_version(parts, versions, "first") == "v0001"
    assert v.save_version(parts, versions) == "v0001"  # identical: reused
    (parts / "02_b.md").write_text("beta changed\n")
    assert v.save_version(parts, versions, "second") == "v0002"
    assert (versions / "v0001" / "02_b.md").read_text() == "beta\n"  # old version untouched
    infos = v.list_versions(versions)
    assert [i.version for i in infos] == ["v0001", "v0002"] and infos[1].note == "second"
    meta = json.loads((versions / "v0002" / "version.json").read_text())
    assert set(meta["parts"]) == {"01_a.md", "02_b.md"}


def test_empty_parts_refused(tmp_path):
    (tmp_path / "p").mkdir()
    with pytest.raises(ValueError):
        v.save_version(tmp_path / "p", tmp_path / "v")


def test_active_handling(dirs):
    parts, versions = dirs
    assert v.active_version(versions) is None
    v.save_version(parts, versions)
    v.set_active(versions, "v0001")
    assert v.active_version(versions) == "v0001"
    assert v.resolve_parts_dir(parts, versions) == versions / "v0001"
    import shutil
    shutil.rmtree(versions / "v0001")
    assert v.active_version(versions) is None and v.resolve_parts_dir(parts, versions) == parts
    with pytest.raises(ValueError):
        v.set_active(versions, "v0009")


def test_next_number_never_reuses_a_deleted_id(dirs):
    parts, versions = dirs
    v.save_version(parts, versions)
    (parts / "01_a.md").write_text("changed\n")
    v.save_version(parts, versions)
    import shutil
    shutil.rmtree(versions / "v0001")
    (parts / "01_a.md").write_text("changed again\n")
    assert v.save_version(parts, versions) == "v0003"


@pytest.mark.parametrize("bad", ["../x", "v1", "v0001; rm", "V0001", "", "v0001/../v0002", "latest"])
def test_bad_ids_rejected(dirs, bad):
    parts, versions = dirs
    v.save_version(parts, versions)
    with pytest.raises(ValueError):
        v.get_version_dir(versions, bad)


def test_draft_vs_active_and_diff(dirs):
    parts, versions = dirs
    assert v.draft_differs_from_active(parts, versions) is True  # parts exist, nothing saved
    v.save_version(parts, versions)
    v.set_active(versions, "v0001")
    assert v.draft_differs_from_active(parts, versions) is False
    (parts / "02_b.md").write_text("beta\nnew line\n")
    (parts / "03_c.md").write_text("gamma\n")
    assert v.draft_differs_from_active(parts, versions) is True
    diff = v.diff_versions(versions, "v0001", "draft", parts_dir=parts)
    assert "+new line" in diff and "+gamma" in diff and "/dev/null" in diff
    with pytest.raises(ValueError):
        v.diff_versions(versions, "v0001", "draft")
    v.save_version(parts, versions)
    assert "+new line" in v.diff_versions(versions, "v0001", "v0002")


def test_prompt_build_reports_saved_version_id(dirs):
    parts, versions = dirs
    v.save_version(parts, versions)
    saved = build_system_prompt("pipeline", TAX, parts_dir=versions / "v0001")
    draft = build_system_prompt("pipeline", TAX, parts_dir=parts)
    assert saved.version == "v0001" and draft.version.startswith("draft-") and saved.text == draft.text


def test_saved_version_records_parent(dirs):
    parts, versions = dirs
    v.save_version(parts, versions)
    v.set_active(versions, "v0001")
    (parts / "01_a.md").write_text("changed\n")
    v.save_version(parts, versions)
    assert v.list_versions(versions)[1].parent == "v0001"
