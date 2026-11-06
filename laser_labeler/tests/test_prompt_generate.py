import json
import shutil
from pathlib import Path

import pytest

from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.prompts import generate as g
from core.prompts import versions as v
from core.prompts.generate import GenerateError, Proposal
from core.schema.taxonomy import add_entry, load_taxonomy
from tests.fakes import ScriptedProvider, text

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def env(tmp_path):
    parts = tmp_path / "parts"
    shutil.copytree(ROOT / "prompts" / "parts", parts)
    tax_dir = tmp_path / "taxonomy"
    shutil.copytree(ROOT / "taxonomy", tax_dir)
    return type("E", (), {"parts": parts, "versions": tmp_path / "versions", "tax_dir": tax_dir,
                          "tax": load_taxonomy(tax_dir)})()


def reply(parts, summary="added ceramic guidance"):
    return text({"summary": summary, "parts": parts})


def test_taxonomy_changes_in_words():
    old = {"materials": ["glass"], "colors": ["red", "blue"], "objects": ["pen"], "laser_events": [], "incidence": []}
    new = {"materials": ["glass", "velvet"], "colors": ["red"], "objects": ["pen", "cup"],
           "laser_events": [], "incidence": []}
    out = g.taxonomy_changes(old, new)
    assert "materials: added velvet" in out and "colors: removed blue" in out
    assert "objects: added cup" in out
    older = {**old, "objects": {"pen": ["gel"]}}  # versions saved before sub-objects were dropped
    assert "objects: added cup" in g.taxonomy_changes(older, new) and "pen" not in g.taxonomy_changes(older, new).split("objects:")[-1]
    assert "unknown" in g.taxonomy_changes(None, new)
    assert "has not changed" in g.taxonomy_changes(new, new)


def test_request_contains_everything_the_model_needs(env):
    ls = ImageLabelSet("a", "a.png", "cam", False, 0.1, status="approved",
                       laser_boxes=[LaserBox("l1", "first_incidence", (1, 2, 3, 4), 0)],
                       object_boxes=[ObjectBox("o1", (5, 6, 7, 8), "pen", "glass", "red", "first")])
    corrections = [{"text": "o1: material rubber -> glass"}]
    msgs = g.build_messages(g.read_parts(env.parts), {"materials": ["glass"]}, env.tax, corrections, [ls])
    body = msgs[1].text
    assert msgs[0].role == "system" and "{{taxonomy}}" in msgs[0].text and "01_role_task.md" in body
    assert "rubber -> glass" in body and "obj_pen_glass_red_first" in body and "las_first-incidence" in body
    assert "materials: added" in body and "Approved examples" in body


def test_proposal_accepts_a_good_reply_and_appends_a_trailing_newline(env):
    current = g.read_parts(env.parts)
    p = g.parse_proposal(reply({"07_background.md": "## Project background\n\nmore detail"}).text, current)
    assert p.summary.startswith("added") and p.parts["07_background.md"].endswith("detail\n")
    new_part = g.parse_proposal(reply({"08_ceramics.md": "Ceramic is glossy."}).text, current)
    assert "08_ceramics.md" in new_part.parts


@pytest.mark.parametrize("parts,message", [
    ({}, "needs a"),
    ({"../evil.md": "x"}, "not a valid"),
    ({"7_background.md": "x"}, "not a valid"),
    ({"07_background.md": ""}, "is empty"),
    ({"03_taxonomy.md": "no placeholder here"}, "lost the placeholder"),
    ({"07_background.md": "x" * (g.MAX_PART_CHARS + 1)}, "too long"),
])
def test_proposal_rejects_bad_replies(env, parts, message):
    with pytest.raises(GenerateError, match=message):
        g.parse_proposal(reply(parts).text, g.read_parts(env.parts))


def test_proposal_must_keep_the_taxonomy_placeholder_somewhere(env):
    current = g.read_parts(env.parts)
    holder = next(n for n, t in current.items() if "{{taxonomy}}" in t)
    other = {n: t.replace("{{taxonomy}}", "") for n, t in current.items()}
    with pytest.raises(GenerateError, match="taxonomy"):
        g.parse_proposal(reply({"09_new.md": "x"}).text, other)
    assert holder


def test_corrections_block_must_survive(env):
    (env.parts / "07_background.md").write_text("bg\n<!-- corrections:start -->\n- x\n<!-- corrections:end -->\n")
    with pytest.raises(GenerateError, match="corrections block"):
        g.parse_proposal(reply({"07_background.md": "bg only"}).text, g.read_parts(env.parts))


def test_generate_repairs_a_bad_reply_then_succeeds(env):
    provider = ScriptedProvider([text("sorry, here you go"),
                                 reply({"03_taxonomy.md": "dropped"}),
                                 reply({"07_background.md": "## Project background\n\nnew"})])
    p = g.generate_new_prompt(provider, env.parts, None, env.tax)
    assert list(p.parts) == ["07_background.md"] and len(provider.calls) == 3
    assert "could not be used" in provider.calls[2][-1].text


def test_generate_gives_up_after_repairs(env):
    provider = ScriptedProvider([text("no")] * 3)
    with pytest.raises(GenerateError):
        g.generate_new_prompt(provider, env.parts, None, env.tax, max_repairs=2)


def test_nothing_is_written_until_applied_then_it_is_versioned_and_active(env):
    before = g.read_parts(env.parts)
    provider = ScriptedProvider([reply({"07_background.md": "## Project background\n\nvelvet is soft"})])
    add_entry(env.tax_dir, "material", "velvet")
    tax = load_taxonomy(env.tax_dir)
    p = g.generate_new_prompt(provider, env.parts, {"materials": ["glass"]}, tax)
    assert g.read_parts(env.parts) == before  # still untouched
    pairs = g.proposal_pairs(env.parts, p)
    assert [n for n, _, _ in pairs] == ["07_background.md"] and "velvet is soft" in pairs[0][2]
    version = g.apply_proposal(p, env.parts, env.versions, tax)
    assert version == "v0001" and v.active_version(env.versions) == "v0001"
    assert "velvet is soft" in (env.parts / "07_background.md").read_text()
    assert "velvet" in v.taxonomy_snapshot(env.versions, "v0001")["materials"]
    assert g.old_taxonomy_for_active(env.versions)["materials"][-1] == "velvet"
    meta = json.loads((env.versions / "v0001" / "version.json").read_text())
    assert meta["note"] == "added ceramic guidance"


def test_old_version_stays_available_after_a_new_one(env):
    v1 = v.save_version(env.parts, env.versions, "first")
    v.set_active(env.versions, v1)
    p = Proposal("s", {"07_background.md": "changed\n"})
    v2 = g.apply_proposal(p, env.parts, env.versions, env.tax)
    assert (v1, v2) == ("v0001", "v0002") and v.active_version(env.versions) == "v0002"
    assert "changed" not in (env.versions / "v0001" / "07_background.md").read_text()
    assert v.list_versions(env.versions)[1].parent == "v0001"


def test_apply_refuses_bad_file_names(env):
    with pytest.raises(GenerateError):
        g.apply_proposal(Proposal("s", {"../x.md": "x"}), env.parts, env.versions, env.tax)
    assert not (env.parts.parent / "x.md").exists()
