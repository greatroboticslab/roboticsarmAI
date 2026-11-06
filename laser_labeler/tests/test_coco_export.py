import pytest

from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.schema.coco_export import build_coco


def make_set(name="a.jpg", status="approved", gray=False):
    return ImageLabelSet(
        image_id=name, image_name=name, camera_id="cam1", is_grayscale=gray,
        background_level=0.1, status=status, image_width=100, image_height=80,
        laser_boxes=[LaserBox("l1", "first_incidence", (10, 10, 5, 5), 0),
                     LaserBox("l2", "reflection", (50, 40, 8, 8), 1, from_id="l1")],
        object_boxes=[ObjectBox("o1", (5, 5, 20, 20), "pen", "rubber",
                                "black", "first", laser_ids=["l1"], confidence=0.9)],
    )


def test_basic_export():
    coco = build_coco([make_set()])
    names = [c["name"] for c in coco["categories"]]
    assert names == sorted(names)
    assert set(names) == {"las_first-incidence", "las_reflection",
                          "obj_pen_rubber_black_first"}
    assert len(coco["images"]) == 1 and len(coco["annotations"]) == 3
    ann = coco["annotations"][0]
    assert ann["area"] == ann["bbox"][2] * ann["bbox"][3]
    assert all(c["id"] >= 1 for c in coco["categories"])


def test_only_approved_or_edited_exported():
    sets = [make_set("a.jpg", "approved"), make_set("b.jpg", "pending"),
            make_set("c.jpg", "rejected"), make_set("d.jpg", "edited")]
    coco = build_coco(sets)
    assert [i["file_name"] for i in coco["images"]] == ["a.jpg", "d.jpg"]
    assert len(build_coco(sets, only_exportable=False)["images"]) == 4


def test_grayscale_uses_na_even_if_color_set():
    coco = build_coco([make_set(gray=True)])
    assert "obj_pen_rubber_na_first" in [c["name"] for c in coco["categories"]]


def test_numbers_and_links_not_sent():
    ls = make_set()
    ls.laser_boxes[0].metrics = {"peak_intensity": 0.9}
    text = str(build_coco([ls]))
    assert "peak_intensity" not in text and "from_id" not in text and "laser_ids" not in text


def test_boxes_clamped_to_image():
    ls = make_set()
    ls.object_boxes[0].box = (90, 70, 50, 50)
    ann = [a for a in build_coco([ls])["annotations"] if a["bbox"][0] == 90][0]
    assert ann["bbox"] == [90, 70, 10, 10]


def test_empty_box_raises():
    ls = make_set()
    ls.object_boxes[0].box = (200, 200, 5, 5)  # fully outside
    with pytest.raises(ValueError):
        build_coco([ls])


def test_extra_categories_and_stable_ids():
    a = build_coco([make_set()], category_names=["las_scatter"])
    assert "las_scatter" in [c["name"] for c in a["categories"]]
    b = build_coco([make_set()], category_names=["las_scatter"])
    assert a["categories"] == b["categories"]
