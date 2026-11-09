import pytest

from core.schema.scheme_labels import (CURRENT, NONE, OLDER, UNKNOWN, is_old_component, labels_state, laser_label_fits,
                                       scheme_example)
from core.schema.class_names import decode_class

OBJ = "obj_pen_rubber_red_first"


@pytest.mark.parametrize("name,scheme,fits", [
    ("las_first-incidence", "full", True), ("las_first-incidence", "components", False),
    ("component1", "components", True), ("component2", "merged", True), ("component1", "full", False),
    (OBJ + "__component1", "merged", True), (OBJ + "__component1", "components", False),
    (OBJ, "full", False), (OBJ, "merged", False), ("component3", "components", False), ("pen", "full", False),
    ("", "full", False), ("las_", "full", False),
])
def test_laser_label_fits(name, scheme, fits):
    assert laser_label_fits(name, scheme) is fits


@pytest.mark.parametrize("name,old", [("component3", True), ("component12", True), (OBJ + "__component3", True),
                                     ("component1", False), ("component2", False), (OBJ + "__component2", False),
                                     ("las_first-incidence", False), (OBJ, False), ("", False)])
def test_old_components_are_recognised(name, old):
    assert is_old_component(name) is old


def test_states():
    assert labels_state(None, "full") == UNKNOWN
    assert labels_state([], "full") == NONE and labels_state(["", None], "full") == NONE
    assert labels_state(["las_first-incidence"], "full") == CURRENT
    assert labels_state([OBJ], "full") == OLDER              # objects alone do not show the current system was used
    assert labels_state(["component1", "component3"], "components") == OLDER  # any halo/beam part marks the old scheme
    assert labels_state(["component1", OBJ], "components") == CURRENT
    assert labels_state(["something else"], "merged") == OLDER


def test_examples_are_valid_names_of_their_scheme():
    for scheme in ("full", "components", "merged"):
        decode_class(scheme_example(scheme))
        assert laser_label_fits(scheme_example(scheme), scheme)
    assert scheme_example("nope") == ""
