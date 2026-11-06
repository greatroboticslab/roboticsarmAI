from core.labeler.table_rule import apply_table_rule
from core.models import ObjectBox


def box(obj, laser_ids):
    return ObjectBox(id=f"{obj}-{len(laser_ids)}", box=(0, 0, 10, 10), general_object=obj,
                     material="wood", color="brown", incidence="first",
                     laser_ids=laser_ids)


def test_object_hit_drops_table():
    boxes = [box("pen", ["l1"]), box("table", ["l2"])]
    assert [b.general_object for b in apply_table_rule(boxes)] == ["pen"]


def test_no_object_hit_keeps_table():
    boxes = [box("pen", []), box("table", ["l1"])]
    assert len(apply_table_rule(boxes)) == 2


def test_only_table_kept():
    boxes = [box("table", ["l1"])]
    assert apply_table_rule(boxes) == boxes


def test_flags():
    boxes = [box("pen", ["l1"]), box("table", ["l2"])]
    assert len(apply_table_rule(boxes, "always_label")) == 2
    assert [b.general_object for b in apply_table_rule([box("table", ["l"])], "never_label")] == []


def test_unknown_rule():
    import pytest
    with pytest.raises(ValueError):
        apply_table_rule([], "bogus")
