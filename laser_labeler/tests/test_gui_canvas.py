import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.widgets.image_canvas import BoxSpec, ImageCanvas  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(app):
    view = ImageCanvas()
    view.resize(640, 400)
    view.show()
    image = QImage(400, 250, QImage.Format.Format_RGB32)
    image.fill(QColor("#303030"))
    assert view.set_image(image)
    view.set_boxes([BoxSpec("o1", (100, 80, 120, 80), "o1:pen", "object"), BoxSpec("l1", (140, 100, 10, 10), "l1", "laser")])
    app.processEvents()
    yield view
    view.close()


def vp(canvas, x, y):
    return canvas.mapFromScene(QPointF(x, y))


def click(canvas, x, y):
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(canvas, x, y))


def drag(canvas, x0, y0, x1, y1):
    w = canvas.viewport()
    QTest.mousePress(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(canvas, x0, y0))
    QTest.mouseMove(w, vp(canvas, (x0 + x1) / 2, (y0 + y1) / 2))
    QTest.mouseMove(w, vp(canvas, x1, y1))
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(canvas, x1, y1))


def record(canvas):
    got = {"changed": [], "selected": [], "drawn": [], "points": []}
    canvas.boxChanged.connect(lambda i, b: got["changed"].append((i, b)))
    canvas.boxSelected.connect(got["selected"].append)
    canvas.boxDrawn.connect(got["drawn"].append)
    canvas.pointClicked.connect(lambda x, y: got["points"].append((x, y)))
    return got


def test_unreadable_image_returns_false_and_clears(app):
    view = ImageCanvas()
    assert view.set_image("/nonexistent/file.png") is False and view.image_size() == (0, 0)
    assert view.set_image(None) is False
    view.set_boxes([BoxSpec("a", (1, 1, 2, 2))])  # boxes with no image must not crash


def test_click_selects_the_smallest_box_and_empty_space_clears(canvas):
    got = record(canvas)
    click(canvas, 145, 105)  # inside both boxes: the small laser box wins
    click(canvas, 110, 90)
    click(canvas, 20, 20)
    assert got["selected"] == ["l1", "o1", ""] and canvas.selected() == ""


def test_edit_mode_moves_a_box_and_stays_inside_the_image(canvas):
    got = record(canvas)
    canvas.set_mode("edit")
    canvas.select("o1")
    drag(canvas, 160, 120, 190, 140)
    (box_id, box), = got["changed"]
    assert box_id == "o1" and abs(box[0] - 130) <= 2 and abs(box[1] - 100) <= 2 and box[2:] == (120, 80)
    got["changed"].clear()
    drag(canvas, 190, 140, 900, 900)  # far outside: clamped at the corner
    assert got["changed"][-1][1] == (400 - 120, 250 - 80, 120, 80)


def test_edit_mode_resizes_by_edges_and_corners(canvas):
    got = record(canvas)
    canvas.set_mode("edit")
    canvas.select("o1")
    drag(canvas, 220, 120, 260, 120)  # right edge
    assert got["changed"][-1][1][2] in range(158, 163) and got["changed"][-1][1][3] == 80
    drag(canvas, 100, 80, 90, 70)  # top-left corner
    x, y, w, h = got["changed"][-1][1]
    assert (x, y) == (90, 70) and w in range(168, 173) and h == 90


def test_resize_cannot_flip_or_vanish(canvas):
    got = record(canvas)
    canvas.set_mode("edit")
    canvas.select("o1")
    drag(canvas, 220, 120, 0, 120)  # right edge dragged past the left edge
    x, y, w, h = got["changed"][-1][1]
    assert w >= 2 and x == 100


def test_view_mode_never_edits(canvas):
    got = record(canvas)
    canvas.select("o1")
    drag(canvas, 160, 120, 190, 140)
    assert got["changed"] == [] and canvas.boxes()[0].box == (100, 80, 120, 80)


def test_a_click_without_moving_is_not_a_change(canvas):
    got = record(canvas)
    canvas.set_mode("edit")
    canvas.select("o1")
    click(canvas, 160, 120)
    assert got["changed"] == []


def test_draw_mode_emits_a_clamped_rectangle_and_ignores_tiny_drags(canvas):
    got = record(canvas)
    canvas.set_mode("draw")
    drag(canvas, 20, 30, 80, 90)
    assert got["drawn"] == [(20, 30, 60, 60)]
    drag(canvas, 300, 200, 700, 700)
    assert got["drawn"][-1] == (300, 200, 100, 50)
    drag(canvas, 50, 50, 50, 50)
    assert len(got["drawn"]) == 2


def test_point_mode_reports_image_coordinates(canvas):
    got = record(canvas)
    canvas.set_mode("point")
    click(canvas, 123, 77)
    (x, y), = got["points"]
    assert abs(x - 123) < 2 and abs(y - 77) < 2


def test_zoom_and_pan_do_not_change_the_box_numbers(canvas):
    got = record(canvas)
    canvas.zoom(2.0)
    canvas.set_mode("edit")
    canvas.select("o1")
    drag(canvas, 160, 120, 150, 120)
    assert got["changed"][-1][1][0] in range(88, 92)  # moved by about -10 image pixels at any zoom
    canvas.fit()


def test_set_mode_rejects_garbage(canvas):
    with pytest.raises(ValueError):
        canvas.set_mode("fly")


def test_widget_renders_boxes_to_a_picture(canvas, tmp_path):
    canvas.select("o1")
    canvas.set_mode("edit")
    out = tmp_path / "canvas.png"
    assert canvas.grab().save(str(out)) and out.stat().st_size > 1000
