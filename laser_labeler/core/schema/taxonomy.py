"""Load, validate and extend the YAML taxonomy files (build spec section 3.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.schema.class_names import ClassNameError, check_event, check_token

# file stem -> top-level key (they match on purpose)
LIST_FILES = ("materials", "colors", "laser_events", "incidence")
OBJECTS_FILE = "objects"


class TaxonomyError(ValueError):
    """Raised when taxonomy files are missing or contain bad entries."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class Taxonomy:
    objects: dict[str, list[str]] = field(default_factory=dict)
    materials: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    laser_events: list[str] = field(default_factory=list)
    incidence: list[str] = field(default_factory=list)

    def check_object(
        self, general_object: str, sub_object: str, material: str,
        color: str, incidence: str, color_placeholder: str = "na",
    ) -> list[str]:
        """Return a list of problems (empty means every value is allowed)."""
        problems = []
        if general_object not in self.objects:
            problems.append(f"object {general_object!r} not in taxonomy")
        elif sub_object not in self.objects[general_object]:
            problems.append(f"sub-object {sub_object!r} not allowed for {general_object!r}")
        if material not in self.materials:
            problems.append(f"material {material!r} not in taxonomy")
        if color != color_placeholder and color not in self.colors:
            problems.append(f"color {color!r} not in taxonomy")
        if incidence not in self.incidence:
            problems.append(f"incidence {incidence!r} not in taxonomy")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": self.objects, "materials": self.materials, "colors": self.colors,
            "laser_events": self.laser_events, "incidence": self.incidence,
        }


def _read_yaml(path: Path, problems: list[str]) -> Any:
    if not path.is_file():
        problems.append(f"missing taxonomy file {path.name}")
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        problems.append(f"{path.name}: invalid YAML ({exc})")
        return None


def _check_list(stem: str, raw: Any, problems: list[str]) -> list[str]:
    values = raw.get(stem) if isinstance(raw, dict) else None
    if not isinstance(values, list) or not values:
        problems.append(f"{stem}.yaml: expected a non-empty list under '{stem}:'")
        return []
    checker = check_event if stem == "laser_events" else (lambda v: check_token(v, stem))
    clean: list[str] = []
    for value in values:
        try:
            checker(value)
        except ClassNameError as exc:
            problems.append(f"{stem}.yaml: {exc}")
            continue
        if value in clean:
            problems.append(f"{stem}.yaml: duplicate entry {value!r}")
        else:
            clean.append(value)
    return clean


def _check_objects(raw: Any, problems: list[str]) -> dict[str, list[str]]:
    table = raw.get("objects") if isinstance(raw, dict) else None
    if not isinstance(table, dict) or not table:
        problems.append("objects.yaml: expected a non-empty mapping under 'objects:'")
        return {}
    clean: dict[str, list[str]] = {}
    for name, subs in table.items():
        try:
            check_token(name, "object")
        except ClassNameError as exc:
            problems.append(f"objects.yaml: {exc}")
            continue
        if not isinstance(subs, list) or not subs:
            problems.append(f"objects.yaml: object {name!r} needs at least one sub-object")
            continue
        clean[name] = _check_list_entries(f"objects.yaml sub-object of {name}", subs, problems)
    return clean


def _check_list_entries(label: str, values: list, problems: list[str]) -> list[str]:
    clean: list[str] = []
    for value in values:
        try:
            check_token(value, label)
        except ClassNameError as exc:
            problems.append(f"objects.yaml: {exc}")
            continue
        if value in clean:
            problems.append(f"objects.yaml: duplicate sub-object {value!r}")
        else:
            clean.append(value)
    return clean


def load_taxonomy(directory: str | Path) -> Taxonomy:
    """Load and validate all taxonomy files. Raises TaxonomyError listing every problem."""
    directory = Path(directory)
    problems: list[str] = []
    objects = _check_objects(_read_yaml(directory / "objects.yaml", problems), problems)
    lists = {
        stem: _check_list(stem, _read_yaml(directory / f"{stem}.yaml", problems), problems)
        for stem in LIST_FILES
    }
    if problems:
        raise TaxonomyError(problems)
    return Taxonomy(objects=objects, **lists)


def add_entry(
    directory: str | Path, kind: str, value: str, parent_object: str | None = None
) -> Taxonomy:
    """Add one entry to a taxonomy file ("add to taxonomy" in the GUI).

    kind is 'object', 'sub_object' (needs parent_object), 'material', 'color',
    'laser_event' or 'incidence'. The file is rewritten, so YAML comments in it
    are lost. Returns the reloaded taxonomy.
    """
    directory = Path(directory)
    current = load_taxonomy(directory)
    if kind == "object":
        raise TaxonomyError(
            ["an object needs at least one sub-object: use kind='sub_object' with "
             "parent_object set to the new object's name (it is created for you)"]
        )
    if kind == "sub_object":
        _add_sub_object(directory, current, value, parent_object)
    else:
        _add_to_list(directory, current, kind, value)
    return load_taxonomy(directory)


def _require_new(value: str, existing: Any, label: str) -> None:
    if value in existing:
        raise TaxonomyError([f"{label} {value!r} already in taxonomy"])


def _add_sub_object(directory: Path, current: Taxonomy, value: str, parent: str | None) -> None:
    check_token(value, "sub-object")
    if parent is None:
        raise TaxonomyError(["adding a sub-object needs parent_object"])
    objects = {k: list(v) for k, v in current.objects.items()}
    objects.setdefault(parent, [])
    check_token(parent, "object")
    _require_new(value, objects[parent], f"sub-object of {parent}")
    objects[parent].append(value)
    _write(directory / "objects.yaml", {"objects": objects})


def _add_to_list(directory: Path, current: Taxonomy, kind: str, value: str) -> None:
    stem = {"material": "materials", "color": "colors",
            "laser_event": "laser_events", "incidence": "incidence"}.get(kind)
    if stem is None:
        raise TaxonomyError([f"unknown taxonomy kind {kind!r}"])
    (check_event if stem == "laser_events" else lambda v: check_token(v, kind))(value)
    values = list(getattr(current, stem))
    _require_new(value, values, kind)
    values.append(value)
    _write(directory / f"{stem}.yaml", {stem: values})


def _write(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None), encoding="utf-8")
