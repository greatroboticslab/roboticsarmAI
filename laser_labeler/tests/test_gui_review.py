import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.labeler.label_image import label_image  # noqa: E402
from core.labeler.runner import run_queue  # noqa: E402
from core.queue_db import NewImage  # noqa: E402
from core.reports.sidecar import read_sidecar  # noqa: E402
from core.roboflow.upload import UploadSummary  # noqa: E402
from gui.context import AppContext  # noqa: E402
from gui.tabs.review_tab import ReviewTab  # noqa: E402
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text  # noqa: E402
from tests.gui_support import make_images, make_project, wait_until  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ctx(app, tmp_path):
    context = AppContext(tmp_path / "user.yaml", make_project(tmp_path), tmp_path / "data")
    yield context
    context.close()


def good(**changes):
    return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0, "from_id": None,
                                "object": {**PEN_RUBBER, **changes}}]))


def label(ctx, tmp_path, names, replies):
    paths = make_images(tmp_path / "pics", names)
    ctx.db.add_images([NewImage(p.stem, p.name, str(p), "roboflow") for p in paths])  # as if downloaded from Roboflow
    provider = ScriptedProvider(replies)
    settings, tax = ctx.settings(), ctx.service().taxonomy()
    run_queue(ctx.db, lambda row: label_image(row.path, provider=provider, taxonomy=tax, settings=settings,
                                              data_dir=ctx.data_dir, image_id=row.image_id),
              sidecar_dir=ctx.data_dir / "sidecars")


@pytest.fixture
def tab(app, ctx, tmp_path):
    label(ctx, tmp_path, ("a.png", "b.png", "c.png"), [good(), good(), good()])
    review = ReviewTab(ctx, key_fn=lambda: "rf-key")
    review.resize(1200, 800)
    review.show()
    app.processEvents()
    yield review
    review.close()


def pick_box(tab, prefix):
    for i in range(tab.boxes.count()):
        if tab.boxes.item(i).data(Qt.ItemDataRole.UserRole).startswith(prefix):
            tab.boxes.setCurrentRow(i)
            return tab.boxes.item(i).data(Qt.ItemDataRole.UserRole)
    raise AssertionError(f"no {prefix} box")


# ---------------------------------------------------------------- listing and showing
def test_lists_ready_images_and_shows_one(tab):
    assert [r.image_id for r in tab.rows] and len(tab.rows) == 3 and "3 image" in tab.count_label.text()
    tab.select_image("a")
    assert tab.image_id == "a" and tab.draft is not None
    assert "a.png" in tab.header.text() and tab.boxes.count() == 2  # one laser box, one object box
    assert tab.viewer.toPlainText().strip()  # the conversation is shown
    assert tab.approve_btn.isEnabled() and not tab.apply_btn.isEnabled()
    assert len(tab.canvas.boxes()) == 2 if hasattr(tab.canvas, "boxes") else True


def test_sort_filter_and_navigation(tab):
    tab.sort_box.setCurrentText("Queue order")
    assert [r.image_id for r in tab.rows] == ["a", "b", "c"]
    tab.select_image("a")
    tab.next_btn.click()
    assert tab.image_id == "b"
    tab.next_image(), tab.next_image()
    assert tab.image_id == "c"  # stays on the last one
    tab.previous_image()
    assert tab.image_id == "b"
    tab.filter_box.setCurrentText("Approved")
    assert tab.rows == [] and tab.image_id == "" or tab.image_id == "b"
    tab.filter_box.setCurrentText("Everything")
    assert len(tab.rows) == 3


def test_keyboard_shortcuts(app, tab):
    tab.sort_box.setCurrentText("Queue order")
    tab.select_image("a")
    tab.setFocus()
    QTest.keyClick(tab, Qt.Key.Key_N)
    assert tab.image_id == "b"
    QTest.keyClick(tab, Qt.Key.Key_P)
    assert tab.image_id == "a"
    QTest.keyClick(tab, Qt.Key.Key_A)
    assert tab.ctx.db.get("a").status == "approved"


def test_thumbnail_grid_view_selects_too(app, tab):
    tab.grid_btn.click()
    assert tab.stack.currentIndex() == 1 and len(tab.grid.keys()) == 3
    tab.grid.select_key("b")
    assert tab.image_id == "b"


# ---------------------------------------------------------------- approve, reject, undo
def test_approve_moves_on_and_builds_the_pdf(ctx, tab):
    tab.sort_box.setCurrentText("Queue order")
    tab.select_image("a")
    assert tab.approve()
    row = ctx.db.get("a")
    assert row.status == "approved" and row.pdf_path and os.path.isfile(row.pdf_path)
    assert tab.image_id == "b" and len(tab.rows) == 2
    assert "Upload approved... (1)" in tab.upload_btn.text()


def test_reject_and_undo(ctx, tab):
    tab.select_image("a")
    assert tab.reject() and ctx.db.get("a").status == "rejected"
    tab.filter_box.setCurrentText("Rejected")
    tab.select_image("a")
    assert tab.reopen_btn.isEnabled()
    tab.reopen_btn.click()
    assert ctx.db.get("a").status == "pending"


def test_approve_with_unknown_taxonomy_value_asks_first(ctx, tab):
    ctx.update_settings(lambda s: s["labeling"].update(taxonomy_match="all"))
    tab.select_image("a")
    pick_box(tab, "o")
    tab.editor.material.setCurrentText("unobtainium")
    assert tab.dirty and "unobtainium" in tab.problems.text()
    asked = []
    tab.confirm_taxonomy = lambda problems: (asked.append(problems), False)[1]
    assert not tab.approve()
    assert asked and ctx.db.get("a").status == "edited"  # the edit was applied; the approval was declined
    tab.confirm_taxonomy = lambda problems: True
    assert tab.approve() and ctx.db.get("a").status == "approved"


def test_by_default_only_the_object_is_matched_to_the_taxonomy(ctx, tab):
    tab.select_image("a")
    pick_box(tab, "o")
    tab.editor.material.setCurrentText("unobtainium")
    assert tab.dirty and tab.problems.text() == ""  # a material off the list is not a problem any more
    tab.confirm_taxonomy = lambda problems: pytest.fail("approval must not ask about a material")
    assert tab.approve() and ctx.db.get("a").status == "approved"


def test_add_values_to_taxonomy_from_the_box(ctx, tab):
    tab.select_image("a")
    pick_box(tab, "o")
    tab.editor.material.setCurrentText("unobtainium")
    tab.add_tax_btn.click()
    assert "unobtainium" in (ctx.taxonomy_dir / "materials.yaml").read_text()
    assert tab.problems.text() == "" and "Added to the taxonomy" in tab.message.text()


# ---------------------------------------------------------------- editing
def test_edit_attribute_apply_and_discard(ctx, tab):
    tab.select_image("a")
    obj = pick_box(tab, "o")
    tab.editor.material.setCurrentText("wood")
    assert tab.dirty and tab.apply_btn.isEnabled() and "not applied" in tab.dirty_label.text()
    assert tab.apply_changes()
    assert ctx.db.get("a").status == "edited" and not tab.dirty
    saved = read_sidecar(ctx.data_dir / "sidecars" / "a.json")
    assert next(o for o in saved.object_boxes if o.id == obj).material == "wood"
    assert (ctx.data_dir / "corrections.jsonl").is_file()
    # discard: change again, then throw it away
    pick_box(tab, "o")
    tab.editor.material.setCurrentText("glass")
    tab.revert_btn.click()
    assert not tab.dirty
    assert next(o for o in tab.draft.object_boxes if o.id == obj).material == "wood"


def test_edit_laser_measurement(ctx, tab):
    tab.select_image("a")
    pick_box(tab, "l")
    table = tab.editor.metrics
    assert table.rowCount() > 0
    name = table.item(0, 0).text()
    table.item(0, 1).setText("12.5")
    assert tab.dirty
    assert tab.draft.laser_boxes[0].metrics[name] == 12.5
    table.item(0, 1).setText("not a number")
    assert tab.draft.laser_boxes[0].metrics[name] == 12.5  # refused, old value kept


def test_changing_image_with_unsaved_edits_asks(ctx, tab):
    tab.sort_box.setCurrentText("Queue order")
    tab.select_image("a")
    pick_box(tab, "o")
    tab.editor.material.setCurrentText("wood")
    answers = iter(["cancel", "discard"])
    tab._ask_unsaved = lambda: next(answers)
    tab.select_image("b")
    assert tab.image_id == "a" and tab.dirty  # cancelled: stay
    tab.select_image("b")
    assert tab.image_id == "b" and not tab.dirty and ctx.db.get("a").status == "pending"


def test_move_resize_and_delete_box_from_the_canvas(ctx, tab):
    tab.select_image("a")
    obj = pick_box(tab, "o")
    before = next(o for o in tab.draft.object_boxes if o.id == obj).box
    tab._canvas_box_changed(obj, (before[0] + 5, before[1] + 5, before[2], before[3]))
    assert tab.dirty and next(o for o in tab.draft.object_boxes if o.id == obj).box[0] == before[0] + 5
    tab.canvas.select(obj)
    tab.delete_btn.click()
    assert obj not in [o.id for o in tab.draft.object_boxes]
    tab.apply_changes()
    assert obj not in [o.id for o in read_sidecar(ctx.data_dir / "sidecars" / "a.json").object_boxes]


def test_draw_new_boxes(ctx, tab):
    tab.select_image("a")
    tab.set_mode("draw_object")
    tab._box_drawn((20, 20, 60, 40))
    assert len(tab.draft.object_boxes) == 2 and tab.dirty
    tab.set_mode("draw_laser")
    tab._box_drawn((150, 100, 12, 12))
    assert len(tab.draft.laser_boxes) == 2
    assert tab.apply_changes()
    saved = read_sidecar(ctx.data_dir / "sidecars" / "a.json")
    assert len(saved.object_boxes) == 2 and len(saved.laser_boxes) == 2


# ---------------------------------------------------------------- questions
def test_hit_point_question_is_answered_by_clicking(ctx, tab, tmp_path):
    # a genuinely laser-free image, so the labeler asks where the beam hits (no model call is needed)
    from PIL import Image
    from tests.synth import finish, scene
    path = tmp_path / "blank.png"
    Image.fromarray(finish(scene())).save(path)
    ctx.db.add_images([NewImage("blank", "blank.png", str(path))])
    provider = ScriptedProvider([])
    run_queue(ctx.db, lambda row: label_image(row.path, provider=provider, taxonomy=ctx.service().taxonomy(), settings=ctx.settings(),
                                              data_dir=ctx.data_dir, image_id=row.image_id),
              sidecar_dir=ctx.data_dir / "sidecars")
    tab.filter_box.setCurrentText("Needs your input")
    assert [r.image_id for r in tab.rows] == ["blank"]
    tab.select_image("blank")
    assert tab.right.currentIndex() == 1 and tab.questions.count() == 1 and "[open]" in tab.questions.item(0).text()
    assert not tab.approve_btn.isEnabled()
    tab.questions.setCurrentRow(0)
    tab.answer_selected()
    assert tab.canvas.mode() == "point"
    tab._point_clicked(140, 110)
    assert ctx.db.get("blank").status == "needs_input"  # a click is only a mark: nothing is sent until confirmed
    assert tab.canvas.marks() == [(140, 110)]
    tab.add_marked_lasers()
    assert ctx.db.get("blank").status == "pending"
    assert tab.rows == [] and tab.image_id == ""  # it left "Needs your input", and nothing else is waiting
    assert "Ready for review" in tab.message.text()
    tab.filter_box.setCurrentText("Ready for review")
    tab.select_image("blank")
    assert len(tab.draft.laser_boxes) == 1 and tab.approve_btn.isEnabled()


# ---------------------------------------------------------------- bulk approve, re-run, PDF, upload
def test_bulk_approve_respects_the_threshold(ctx, tab):
    tab.threshold.setValue(1.0)
    assert tab.bulk_approve() == []
    tab.threshold.setValue(0.0)
    assert sorted(tab.bulk_approve()) == ["a", "b", "c"]
    assert ctx.db.counts()["approved"] == 3 and tab.image_id == "" and "Nothing to review" in tab.header.text()
    assert "Upload approved... (3)" in tab.upload_btn.text()


def test_rerun_with_a_note_requeues(ctx, tab):
    tab.select_image("a")
    assert tab.rerun_with("it is a gel pen")
    assert ctx.db.get("a").status == "queued"
    assert json.loads((ctx.data_dir / "rerun_notes.json").read_text())["a"]


def test_rebuild_pdf_after_an_edit_adds_an_addendum(ctx, tab):
    tab.select_image("a")
    assert tab.approve() is True
    tab.filter_box.setCurrentText("Approved")
    tab.select_image("a")
    pick_box(tab, "o")
    tab.editor.material.setCurrentText("wood")
    tab.pdf_btn.click()  # applies the edit first, then rebuilds
    row = ctx.db.get("a")
    assert row.status == "edited" and os.path.isfile(row.pdf_path)
    assert (ctx.data_dir / "pdfs" / "a.addenda.json").is_file() or list(ctx.data_dir.rglob("a.addenda.json"))


def test_upload_dry_run_then_real_upload_marks_uploaded(app, ctx, tab):
    ctx.update_settings(lambda s: s["roboflow"].update(workspace="ws", project="pr"))
    sent = []

    def fake_upload(plan, key, *, dry_run, audit_log, out):
        sent.append((len(plan.items), dry_run, key))
        return UploadSummary(dry_run=dry_run, uploaded=[] if dry_run else [i.image_id for i in plan.items])

    tab._upload_fn = fake_upload
    tab.open_upload()
    assert "Nothing is approved" in tab.message.text()
    tab.threshold.setValue(0.0)
    tab.bulk_approve()
    dialog = tab.open_upload()
    assert dialog is not None and dialog.dry_run.isChecked() and "a" in dialog.text.toPlainText()
    dialog.start()
    assert wait_until(app, lambda: not dialog.busy) and sent == [(3, True, "rf-key")]
    assert ctx.db.counts()["uploaded"] == 0
    dialog.dry_run.setChecked(False)
    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.start()
    assert wait_until(app, lambda: ctx.db.counts()["uploaded"] == 3)
    assert "Upload approved... (0)" in tab.upload_btn.text() or wait_until(app, lambda: "(0)" in tab.upload_btn.text())
    dialog.close()


def test_upload_needs_a_project(ctx, tab):
    tab.open_upload()
    assert "Sign in tab" in tab.message.text()


# ---------------------------------------------------------------- label scheme "components"
def components_tab(app, ctx, tmp_path):
    from PIL import Image
    from tests.synth import finish, rect, scene, spot
    ctx.update_settings(lambda s: s["labeling"].update(scheme="components"))
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, halo_amp=0.25, halo_sigma=14.0, color=(1, 1, 1))
    path = tmp_path / "halo.png"
    Image.fromarray(finish(img)).save(path)
    ctx.db.add_images([NewImage("halo", "halo.png", str(path))])
    provider = ScriptedProvider([good()])
    settings, tax = ctx.settings(), ctx.service().taxonomy()
    run_queue(ctx.db, lambda row: label_image(row.path, provider=provider, taxonomy=tax, settings=settings,
                                              data_dir=ctx.data_dir, image_id=row.image_id),
              sidecar_dir=ctx.data_dir / "sidecars")
    tab = ReviewTab(ctx, key_fn=lambda: "rf-key")
    tab.resize(1200, 800)
    tab.show()
    app.processEvents()
    return tab


def test_review_shows_components_and_the_event_decides_them(app, ctx, tmp_path):
    tab = components_tab(app, ctx, tmp_path)
    tab.select_image("halo")
    labels = [tab.boxes.item(i).text() for i in range(tab.boxes.count())]
    assert any("component1" in t for t in labels) and not any("component3" in t for t in labels)
    spot_id = pick_box(tab, "l1")
    assert tab.editor.component.currentData() == "component1"
    assert tab.editor.component.count() == 3  # not set, component1, component2
    tab.editor.kind.setCurrentText("reflection")  # a bounce, so the class follows
    assert tab.editor.component.currentData() == "component2"
    assert tab.dirty and tab.apply_changes()
    saved = read_sidecar(ctx.data_dir / "sidecars" / "halo.json")
    box = next(b for b in saved.laser_boxes if b.id == spot_id)
    assert box.kind == "reflection" and box.component == "component2"
    assert ctx.db.get("halo").status == "edited"
    assert any("component1 -> component2" in c["text"] for c in
               [json.loads(line) for line in (ctx.data_dir / "corrections.jsonl").read_text().splitlines()])
    tab.close()


def test_deleting_a_spot_in_review_removes_just_that_box(app, ctx, tmp_path):
    tab = components_tab(app, ctx, tmp_path)
    tab.select_image("halo")
    assert [b.component for b in tab.draft.laser_boxes] == ["component1"]  # no halo or beam boxes any more
    tab.canvas.select("l1")
    tab.delete_btn.click()
    assert [b.id for b in tab.draft.laser_boxes] == []
    tab.close()


# ---------------------------------------------------------------- why an image needs input, and the visible conversation
def blank_labeled(ctx, tmp_path, name="blank"):
    from PIL import Image
    from tests.synth import finish, scene
    path = tmp_path / f"{name}.png"
    Image.fromarray(finish(scene())).save(path)
    ctx.db.add_images([NewImage(name, f"{name}.png", str(path))])
    provider = ScriptedProvider([])
    settings, tax = ctx.settings(), ctx.service().taxonomy()
    run_queue(ctx.db, lambda row: label_image(row.path, provider=provider, taxonomy=tax, settings=settings,
                                              data_dir=ctx.data_dir, image_id=row.image_id),
              sidecar_dir=ctx.data_dir / "sidecars")


def test_needs_input_rows_say_why_and_the_conversation_is_always_visible(app, ctx, tmp_path):
    blank_labeled(ctx, tmp_path)
    tab = ReviewTab(ctx)
    tab.resize(1300, 850)
    tab.show()
    tab.filter_box.setCurrentText("Needs your input")
    assert "why: no laser found" in tab.list.item(0).text()
    tab.select_image("blank")
    assert "Needs your input:</b> no laser found" in tab.header.text()
    assert tab.right.count() == 2  # Labels and Questions: the conversation is not hidden in a tab
    app.processEvents()
    assert tab.viewer.isVisible() and tab.viewer.height() > 60
    shown = tab.viewer.toPlainText()
    assert "DETECTION" in shown and "found no laser spot" in shown and "QUESTION" in shown and "RESULT" in shown
    tab.close()


def test_conversation_is_shown_for_every_kind_of_image(app, ctx, tmp_path):
    label(ctx, tmp_path, ("a.png",), [good()])
    tab = ReviewTab(ctx)
    tab.show()
    tab.select_image("a")
    shown = tab.viewer.toPlainText()
    assert "DETECTION" in shown and "SYSTEM" in shown and "ASSISTANT" in shown and "RESULT" in shown
    tab.close()


# ---------------------------------------------------------------- delete a box, and the merged one-label name
def test_delete_key_and_box_list_menu_delete_a_box(app, ctx, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    tab = components_tab(app, ctx, tmp_path)
    tab.select_image("halo")
    objects = [o.id for o in tab.draft.object_boxes]
    assert objects and "(Del)" in tab.delete_btn.text()
    tab.canvas.select(objects[0])
    QTest.keyClick(tab, Qt.Key.Key_Delete)
    assert tab.draft.object_boxes == [] and tab.dirty
    tab.close()


def test_merged_boxes_show_one_label_with_object_and_component(app, ctx, tmp_path):
    from gui.tabs.label_run_tab import boxes_of
    tab = components_tab(app, ctx, tmp_path)
    ctx.update_settings(lambda s: s["labeling"].update(scheme="merged"))
    tab.select_image("halo")
    tab.draft.scheme = "merged"
    tab._fill_box_list("")
    texts = [tab.boxes.item(i).text() for i in range(tab.boxes.count())]
    laser = next(t for t in texts if t.startswith("laser"))
    assert laser.startswith("laser l1: obj_pen_") and laser.endswith("__component1")
    assert any("named inside its laser box's label" in t for t in texts if t.startswith("object"))
    assert any(s.label.startswith("l1:obj_pen_") for s in boxes_of(tab.draft))
    tab.close()


def test_an_old_image_can_be_switched_to_the_merged_scheme(app, ctx, tmp_path):
    from gui.tabs.label_run_tab import laser_title
    tab = components_tab(app, ctx, tmp_path)  # labeled under "components"
    ctx.update_settings(lambda s: s["labeling"].update(scheme="merged"))
    tab.select_image("halo")
    assert tab.draft.scheme == "components" and tab.scheme_btn.isEnabled()
    assert "merged" in tab.scheme_btn.text()
    tab.scheme_btn.click()
    assert tab.draft.scheme == "merged" and tab.dirty and not tab.scheme_btn.isEnabled()
    box = tab.draft.laser_boxes[0]
    assert laser_title(tab.draft, box).startswith("obj_pen_") and laser_title(tab.draft, box).endswith("__component1")
    assert tab.apply_changes()
    saved = read_sidecar(ctx.data_dir / "sidecars" / "halo.json")
    assert saved.scheme == "merged" and saved.laser_class_name(saved.laser_boxes[0]).startswith("obj_pen_")
    tab.close()


def test_clear_taxonomy_questions_button(app, ctx, tmp_path):
    tab = components_tab(app, ctx, tmp_path)
    assert tab.clear_tax_btn.isEnabled()
    tab.confirm_clear = lambda count: False
    assert tab.clear_taxonomy_questions() == (0, [])  # declined: nothing happens
    tab.confirm_clear = lambda count: True
    freed, added = tab.clear_taxonomy_questions()
    assert freed == 0 and added == []  # nothing in this fixture is waiting
    tab.close()


def test_upload_dialog_leaves_local_file_images_out(app, ctx, tab):
    ctx.update_settings(lambda s: s["roboflow"].update(workspace="ws", project="pr"))
    ctx.db.add_images([NewImage("fromfile", "fromfile.png", str(next(iter(ctx.data_dir.parent.glob("*.png")), "x")))])
    tab.threshold.setValue(0.0)
    tab.bulk_approve()
    ids = [r.image_id for r in ctx.db.list(["approved", "edited"])]
    dialog = tab.open_upload()
    assert dialog is not None
    text = dialog.text.toPlainText()
    assert "no image file is uploaded" in text and "updates the labels of the existing Roboflow image" in text
    assert "fromfile" not in ids or "excluded fromfile" in text
    dialog.close()


# ---------------------------------------------------------------- marking laser spots
def test_marks_accumulate_undo_and_remove_by_clicking_near(tab):
    tab.select_image("a")
    tab.set_mode("mark")
    assert tab.canvas.mode() == "point" and tab.marks_bar.isVisibleTo(tab)
    assert "does not set the laser's box" in tab.marks_label.text()  # the reminder: a click is a place, not a box
    tab._point_clicked(100, 80)
    tab._point_clicked(200, 120)
    assert tab.canvas.marks() == [(100, 80), (200, 120)] and "2 spots marked" in tab.marks_label.text()
    assert not tab.marks_add_btn.isVisibleTo(tab)  # only shown while a question is being answered
    tab.undo_mark()
    assert tab.canvas.marks() == [(100, 80)]
    tab._point_clicked(101, 81)  # on top of an existing mark: removes it
    assert tab.canvas.marks() == []
    tab._point_clicked(5, 5)
    tab.clear_marks()
    assert tab.canvas.marks() == [] and not tab.marks_undo_btn.isEnabled()


def test_marks_are_sent_with_the_rerun_and_then_cleared(tab, ctx):
    from core.review.service import take_rerun_points
    tab.select_image("a")
    tab.set_mode("mark")
    tab._point_clicked(100, 80)
    tab._point_clicked(200, 120)
    assert tab.rerun_with("")  # marks alone are enough, no note needed
    assert take_rerun_points(ctx.data_dir, "a") == [(100, 80), (200, 120)]
    assert ctx.db.get("a").status == "queued"
    assert tab.canvas.marks() == [] and not tab.marks_bar.isVisibleTo(tab)
    assert "places you marked" in tab.message.text()


def test_marks_are_dropped_when_another_image_is_opened(tab):
    tab.select_image("a")
    tab.set_mode("mark")
    tab._point_clicked(100, 80)
    tab.select_image("b")
    assert tab.canvas.marks() == [] and tab.canvas.mode() == "view" and not tab.marks_bar.isVisibleTo(tab)


def test_leaving_the_marking_mode_keeps_marks_for_a_rerun(tab):
    tab.select_image("a")
    tab.set_mode("mark")
    tab._point_clicked(100, 80)
    tab.set_mode("view")
    assert tab.canvas.marks() == [(100, 80)] and tab.marks_bar.isVisibleTo(tab)


def test_mark_button_is_off_with_no_image(tab):
    tab._show_nothing()
    tab.image_id = ""
    tab._update_buttons()
    assert not tab.mode_buttons["mark"].isEnabled()
    tab.set_mode("mark")
    assert tab.canvas.mode() != "point"


# ---------------------------------------------------------------- fitting a box by hand
def test_fit_box_by_hand_replaces_only_the_box(tab):
    tab.select_image("a")
    laser = tab.draft.laser_boxes[0]
    before = (laser.kind, laser.id, dict(laser.metrics))
    tab.canvas.select(laser.id)
    tab.refit_selected()
    assert tab.canvas.mode() == "draw"
    tab._box_drawn((50, 40, 33, 22))
    assert tab.draft.laser_boxes[0].box == (50, 40, 33, 22) and tab.dirty
    assert (tab.draft.laser_boxes[0].kind, tab.draft.laser_boxes[0].id) == before[:2]  # label kept
    assert len(tab.draft.laser_boxes) == 1 and tab.canvas.mode() == "edit"  # nothing was added
    assert tab.apply_changes()
    assert read_sidecar(tab.ctx.data_dir / "sidecars" / "a.json").laser_boxes[0].box == (50, 40, 33, 22)


def test_fit_box_needs_a_selected_box_and_can_be_cancelled(tab):
    tab.select_image("a")
    tab.canvas.select("")
    tab.refit_selected()
    assert "Select a box first" in tab.message.text() and tab.canvas.mode() != "draw"
    tab.canvas.select(tab.draft.laser_boxes[0].id)
    tab.refit_selected()
    tab.set_mode("view")  # picking another mode cancels
    tab._box_drawn((5, 5, 10, 10))  # now an ordinary draw in whatever kind: must not touch the laser box
    assert tab.draft.laser_boxes[0].box != (5, 5, 10, 10)


def test_fitting_a_moved_box_settles_its_flag(tab):
    from core.models import Question
    tab.select_image("a")
    laser = tab.draft.laser_boxes[0]
    tab.draft.questions.append(Question("qm_" + laser.id, "moved", laser.id))
    tab.canvas.select(laser.id)
    tab.refit_selected()
    tab._box_drawn((60, 50, 20, 20))
    assert next(q for q in tab.draft.questions if q.id == "qm_" + laser.id).answer == "box fitted by hand"
