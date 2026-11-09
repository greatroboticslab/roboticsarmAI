from pathlib import Path

from PIL import Image

from core.local_files import collect_files, make_image_id, new_images_for
from core.queue_db import NewImage, QueueDB
from core.selection import (LABELED, UNKNOWN, UNLABELED, Entry, filter_entries, label_state, next_unticked,
                            split_options, tag_options)


def entries():
    return [Entry("a", "cup_1.png", split="train", tags=["camera:x"], labels=[]),
            Entry("b", "cup_2.png", split="valid", tags=["camera:y"], labels=["obj_x"]),
            Entry("c", "pen_1.png", split="train", tags=[], labels=None),
            Entry("d", "pen_2.png", split=None, tags=["camera:x", "old"], labels=[])]


def keys(found):
    return [e.key for e in found]


def test_label_state():
    assert [label_state(e) for e in entries()] == [UNLABELED, LABELED, UNKNOWN, UNLABELED]


def test_all_versus_unlabeled_only_keeps_unknown_visible():
    assert keys(filter_entries(entries())) == ["a", "b", "c", "d"]
    assert keys(filter_entries(entries(), only_unlabeled=True)) == ["a", "c", "d"]  # c is unknown, not dropped


def test_filters_combine():
    assert keys(filter_entries(entries(), split="train")) == ["a", "c"]
    assert keys(filter_entries(entries(), tag="camera:x")) == ["a", "d"]
    assert keys(filter_entries(entries(), text="PEN")) == ["c", "d"]
    assert keys(filter_entries(entries(), only_unlabeled=True, split="train", tag="camera:x")) == ["a"]


def test_already_queued_images_are_hidden_unless_asked():
    assert keys(filter_entries(entries(), queued=["a", "z"])) == ["b", "c", "d"]
    assert keys(filter_entries(entries(), queued=["a"], hide_queued=False))[0] == "a"


def test_chunk_picker_takes_the_next_unticked_in_order():
    shown = filter_entries(entries())
    assert next_unticked(shown, [], 2) == ["a", "b"]
    assert next_unticked(shown, ["a", "b"], 2) == ["c", "d"]
    assert next_unticked(shown, ["a"], 10) == ["b", "c", "d"] and next_unticked(shown, [], 0) == []
    assert next_unticked(shown, ["a", "b", "c", "d"], 5) == []


def test_option_lists():
    assert split_options(entries()) == ["train", "valid"] and tag_options(entries()) == ["camera:x", "camera:y", "old"]


def test_collect_files_recurses_filters_types_and_dedupes_ids(tmp_path):
    for sub in ("one", "two"):
        (tmp_path / sub).mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / sub / "same.png")
    (tmp_path / "one" / "notes.txt").write_text("x")
    files = collect_files([tmp_path])
    assert [p.name for p in files] == ["same.png", "same.png"]
    db = QueueDB(tmp_path / "q.db")
    items = new_images_for(db, files, ["camera:z"])
    assert items[0].image_id == "same" and items[1].image_id.startswith("same-") and items[0].tags == ["camera:z"]
    db.add_images(items)
    again = new_images_for(db, files)  # the same files later get the same ids, so nothing is added twice
    assert [i.image_id for i in again] == [i.image_id for i in items] and db.add_images(again) == 0
    db.close()


# ---------------- labeled in the CURRENT labeling system
from core.selection import OLDER  # noqa: E402


def scheme_entries():
    return [Entry("none", "a.png", labels=[]),
            Entry("las", "b.png", labels=["las_first-incidence", "obj_pen_rubber_red_first"]),
            Entry("comp", "c.png", labels=["component1"]),
            Entry("old", "d.png", labels=["component1", "component3"]),
            Entry("objs", "e.png", labels=["obj_pen_rubber_red_first"]),
            Entry("merged", "f.png", labels=["obj_pen_rubber_red_first__component2"]),
            Entry("unk", "g.png", labels=None)]


def test_label_state_by_scheme():
    got = {s: [label_state(e, s) for e in scheme_entries()] for s in ("full", "components", "merged")}
    assert got["full"] == [UNLABELED, LABELED, OLDER, OLDER, OLDER, OLDER, UNKNOWN]
    assert got["components"] == [UNLABELED, OLDER, LABELED, OLDER, OLDER, OLDER, UNKNOWN]
    assert got["merged"] == [UNLABELED, OLDER, LABELED, OLDER, OLDER, LABELED, UNKNOWN]


def test_without_a_scheme_any_label_still_counts_as_labeled():
    assert [label_state(e) for e in scheme_entries()][:2] == [UNLABELED, LABELED] and label_state(scheme_entries()[4]) == LABELED


def test_unlabeled_only_with_a_scheme_keeps_older_and_unknown_and_drops_current():
    assert keys(filter_entries(scheme_entries(), only_unlabeled=True, scheme="components")) == [
        "none", "las", "old", "objs", "merged", "unk"]
    assert keys(filter_entries(scheme_entries(), only_unlabeled=True, scheme="merged")) == ["none", "las", "old", "objs", "unk"]
    assert len(filter_entries(scheme_entries(), scheme="merged")) == 7  # "All images" shows everything
