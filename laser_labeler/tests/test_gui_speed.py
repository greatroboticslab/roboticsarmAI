"""Big lists must not freeze the window. The failure these guard against: every settings lookup parsed the YAML files
again, and the Select images tab did a lookup per image on every click, so 400 Roboflow images took seconds per click."""
import time

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from core.queue_db import NewImage
from core.roboflow.images import RemoteImage
from core.selection import Entry, filter_entries, label_state
from gui.context import AppContext
from gui.tabs.select_images_tab import SelectImagesTab
from gui.widgets.thumb_grid import GridItem, ThumbnailGrid
from tests.gui_support import make_project, wait_until


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ctx(app, tmp_path):
    context = AppContext(tmp_path / "user.yaml", make_project(tmp_path), tmp_path / "data")
    yield context
    context.close()


def count_parses(monkeypatch):
    import gui.context as module
    real, calls = module.load_settings, []
    monkeypatch.setattr(module, "load_settings", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    return calls


def remote_images(n):
    return [RemoteImage(f"id{i}", f"img_{i}.jpg", split="train", tags=["a"] if i % 3 else [],
                        labels=[] if i % 2 else ["component1", f"obj_pen_rubber_red_first__component{1 + i % 2}"])
            for i in range(n)]


# ---------------------------------------------------------------- settings
def test_settings_are_parsed_once_until_a_file_changes(ctx, monkeypatch):
    calls = count_parses(monkeypatch)
    for _ in range(50):
        ctx.settings()
    assert len(calls) == 1
    ctx.update_settings(lambda s: s["labeling"].update(scheme="merged"))
    assert ctx.settings()["labeling"]["scheme"] == "merged" and len(calls) == 2  # the saved file is read once
    for _ in range(50):
        ctx.settings()
    assert len(calls) == 2


def test_a_file_edited_by_hand_is_noticed(ctx):
    assert ctx.settings()["labeling"]["scheme"] == "full"
    ctx.settings_path.write_text("labeling:\n  scheme: components\n", encoding="utf-8")
    assert ctx.settings()["labeling"]["scheme"] == "components"


def test_callers_get_their_own_copy_of_the_settings(ctx):
    first = ctx.settings()
    first["labeling"]["scheme"] = "merged"
    first["provider"]["model"] = "scribbled"
    again = ctx.settings()
    assert again["labeling"]["scheme"] == "full" and again["provider"]["model"] != "scribbled"


# ---------------------------------------------------------------- the Select images tab
def make_tab(app, ctx, n):
    ctx.update_settings(lambda s: (s["roboflow"].update(workspace="ws", project="pr"), s["labeling"].update(scheme="merged")))
    images = remote_images(n)
    tab = SelectImagesTab(ctx, list_fn=lambda ws, p, k: images, download_fn=lambda *a: None, key_fn=lambda: "k")
    tab.remote_radio.setChecked(True)
    return tab, images


def test_a_big_list_does_not_read_the_settings_once_per_image(app, ctx, monkeypatch):
    tab, images = make_tab(app, ctx, 1500)
    calls = count_parses(monkeypatch)
    ctx.settings()  # the one parse after the settings change above
    start = len(calls)
    tab._remote_loaded(images)
    tab.all_radio.setChecked(True)
    tab.tick_all_btn.click()
    tab.untick_btn.click()
    tab.tick_all_btn.click()
    tab.hide_queued.setChecked(False)
    assert len(calls) - start <= 1  # not 1,500 (the old code parsed the YAML per image, per click)


def test_loading_ticking_and_counting_a_big_list_is_quick(app, ctx):
    tab, images = make_tab(app, ctx, 4000)
    started = time.perf_counter()
    tab._remote_loaded(images)
    tab.all_radio.setChecked(True)
    tab.tick_all_btn.click()
    first = tab.grid.item(0)
    first.setCheckState(Qt.CheckState.Unchecked)
    tab.untick_btn.click()
    app.processEvents()
    assert time.perf_counter() - started < 3.0  # was minutes
    assert len(tab._shown) == 4000 and len(tab.ticked) == 0


def test_tick_all_ticks_every_shown_image_and_untick_clears_them(app, ctx):
    tab, images = make_tab(app, ctx, 600)
    tab._remote_loaded(images)
    tab.all_radio.setChecked(True)
    tab.tick_all_btn.click()
    assert len(tab.ticked) == 600 and len(tab.grid.checked_keys()) == 600
    assert "600 ticked" in tab.count_label.text() and "need labeling" in tab.count_label.text()
    tab.untick_btn.click()
    assert tab.ticked == set() and tab.grid.checked_keys() == []


def test_label_states_are_worked_out_once_per_load_and_follow_the_scheme(app, ctx):
    tab, images = make_tab(app, ctx, 10)
    tab._remote_loaded(images)
    assert tab._states["id0"] == "labeled" and tab._states["id1"] == "unlabeled"
    assert tab._needs == 5
    ctx.update_settings(lambda s: s["labeling"].update(scheme="full"))
    assert tab._states["id0"] == "older" and tab._needs == 10  # no `las_` labels anywhere: all need labeling


def test_adding_to_the_queue_with_many_ticked_images_is_quick(app, ctx, tmp_path):
    tab, images = make_tab(app, ctx, 3000)
    fetched = []

    def download(img, folder, key):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{img.id}.jpg"
        path.write_bytes(b"x")
        fetched.append(img.id)
        return path

    tab._download_fn = download
    tab._remote_loaded(images)
    tab.all_radio.setChecked(True)
    tab.tick_all_btn.click()
    tab.ask_labeled = lambda n, scheme: "skip"
    started = time.perf_counter()
    tab.add_to_queue()  # asks about the ~1,500 already labeled, which must not be a slow loop
    assert time.perf_counter() - started < 2.0
    assert wait_until(app, lambda: not tab.busy(), 60)
    assert ctx.db.counts()["queued"] == 1500 and len(fetched) == 1500


# ---------------------------------------------------------------- the thumbnail grid
def test_set_checked_exactly_changes_only_what_differs_and_reports_once(app):
    grid = ThumbnailGrid()
    grid.set_items([GridItem(f"k{i}", f"k{i}", None, checked=i < 3) for i in range(10)])
    reports = []
    grid.checks_changed.connect(lambda: reports.append(1))
    grid.set_checked_exactly({"k2", "k5", "k7"})
    assert sorted(grid.checked_keys()) == ["k2", "k5", "k7"] and reports == [1]
    grid.set_checked(["k0"], True)
    assert "k0" in grid.checked_keys() and len(reports) == 2


def test_set_checked_skips_disabled_items(app):
    grid = ThumbnailGrid()
    grid.set_items([GridItem("a", "a", None, checked=False), GridItem("b", "b", None, checked=False, enabled=False)])
    grid.set_checked_exactly({"a", "b"})
    assert grid.checked_keys() == ["a"]


def test_images_with_no_file_need_no_thumbnail_and_share_one_icon(app):
    grid = ThumbnailGrid()
    grid.set_items([GridItem(f"k{i}", f"k{i}", None) for i in range(50)])
    assert not grid.loading()  # nothing to decode, so nothing waits in line
    assert len({grid.item(i).icon().cacheKey() for i in range(50)}) == 1


def test_thumbnails_with_files_still_load_in_batches(app, tmp_path):
    from tests.gui_support import make_images
    paths = make_images(tmp_path / "imgs", tuple(f"{i}.png" for i in range(9)))
    grid = ThumbnailGrid()
    grid.resize(600, 400)
    grid.show()
    grid.set_items([GridItem(p.name, p.name, str(p)) for p in paths])
    assert grid.loading()
    assert wait_until(app, lambda: not grid.loading(), 15)
    assert len({grid.item(i).icon().cacheKey() for i in range(9)}) > 1  # real thumbnails, not the shared placeholder


# ---------------------------------------------------------------- estimate and queue helpers
def test_estimate_reads_past_runs_once_not_on_every_call(ctx, monkeypatch):
    import gui.context as module
    calls = []
    real = module.run_stats
    monkeypatch.setattr(module, "run_stats", lambda d: (calls.append(1), real(d))[1])
    for n in range(1, 30):
        ctx.estimate(n)
    assert len(calls) == 1
    ctx.queue_changed.emit()  # a run finished an image: the numbers may have changed
    ctx.estimate(5)
    assert len(calls) == 2
    ctx.estimate(5, max_age_s=0)
    assert len(calls) == 3


def test_estimate_numbers_are_unchanged_by_the_cache(ctx):
    a = ctx.estimate(10)
    b = ctx.estimate(10, max_age_s=0)
    assert (a.images, a.seconds, a.cost_usd) == (b.images, b.seconds, b.cost_usd)


def test_queue_ids_lists_every_image_cheaply(ctx):
    ctx.db.add_images([NewImage(f"x{i}", f"x{i}.png", f"/p/x{i}.png") for i in range(5)])
    assert ctx.db.ids() == {f"x{i}" for i in range(5)}
    ctx.db.remove(["x1"])
    assert "x1" not in ctx.db.ids()


def test_filter_entries_with_precomputed_states_gives_the_same_answer():
    entries = [Entry("a", "a", labels=[]), Entry("b", "b", labels=["component1"]), Entry("c", "c", labels=["obj_pen_rubber_red_first"]),
               Entry("d", "d", labels=None)]
    states = {e.key: label_state(e, "components") for e in entries}
    plain = filter_entries(entries, only_unlabeled=True, scheme="components")
    quick = filter_entries(entries, only_unlabeled=True, scheme="components", states=states)
    assert [e.key for e in plain] == [e.key for e in quick] == ["a", "c", "d"]
