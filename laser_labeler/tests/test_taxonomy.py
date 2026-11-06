import shutil
from pathlib import Path

import pytest

from core.schema.taxonomy import TaxonomyError, add_entry, load_taxonomy

SHIPPED = Path(__file__).resolve().parent.parent / "taxonomy"


@pytest.fixture
def tax_dir(tmp_path):
    target = tmp_path / "taxonomy"
    shutil.copytree(SHIPPED, target)
    return target


def test_shipped_taxonomy_is_valid():
    tax = load_taxonomy(SHIPPED)
    assert "pen" in tax.objects and isinstance(tax.objects, list)
    assert "rubber" in tax.materials
    assert "first-incidence" in tax.laser_events


def test_check_object_accepts_and_rejects():
    tax = load_taxonomy(SHIPPED)
    assert tax.check_object("pen", "rubber", "black", "first") == []
    assert tax.check_object("pen", "rubber", "na", "first") == []
    problems = tax.check_object("spoon", "unobtainium", "plaid", "sideways")
    assert len(problems) == 4


def test_bad_entry_with_separator_refused(tax_dir):
    (tax_dir / "materials.yaml").write_text("materials: [glass, hard-plastic, soft_rubber, 'two words']\n")
    with pytest.raises(TaxonomyError) as exc:
        load_taxonomy(tax_dir)
    assert len(exc.value.problems) == 3


def test_missing_and_malformed_files(tax_dir):
    (tax_dir / "colors.yaml").unlink()
    (tax_dir / "objects.yaml").write_text("objects: [Pen]\n")
    with pytest.raises(TaxonomyError) as exc:
        load_taxonomy(tax_dir)
    text = str(exc.value)
    assert "colors.yaml" in text and "Pen" in text


def test_duplicates_reported(tax_dir):
    (tax_dir / "colors.yaml").write_text("colors: [red, red]\n")
    with pytest.raises(TaxonomyError):
        load_taxonomy(tax_dir)


def test_add_entries(tax_dir):
    tax = add_entry(tax_dir, "material", "foam")
    assert "foam" in tax.materials
    tax = add_entry(tax_dir, "object", "glassware")
    assert "glassware" in tax.objects and "pen" in tax.objects
    tax = add_entry(tax_dir, "laser_event", "diffraction")
    assert "diffraction" in tax.laser_events


def test_add_entry_rejects_bad_and_duplicate(tax_dir):
    with pytest.raises(Exception):
        add_entry(tax_dir, "material", "hard-foam")
    with pytest.raises(TaxonomyError):
        add_entry(tax_dir, "material", "glass")
    with pytest.raises(TaxonomyError):
        add_entry(tax_dir, "object", "pen")  # already there
    with pytest.raises(Exception):
        add_entry(tax_dir, "object", "desk-lamp")
    with pytest.raises(TaxonomyError):
        add_entry(tax_dir, "sub_object", "x")  # sub-objects no longer exist
    assert "hard-foam" not in load_taxonomy(tax_dir).materials


def test_the_older_mapping_format_is_still_read_and_sub_objects_ignored(tax_dir):
    (tax_dir / "objects.yaml").write_text("objects:\n  pen: [ballpoint, gel]\n  cup: [mug]\n")
    assert load_taxonomy(tax_dir).objects == ["pen", "cup"]
