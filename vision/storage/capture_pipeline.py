"""
[WIRED] The one place a capture gets recorded.

WHY THIS FILE EXISTS
---------------------
Previously there were two divergent write paths into MongoDB: the
automatic pipeline (vision_service.py -> logger_service.py) and the
GUI's manual "capture_photo_and_store" button, which wrote straight to
mongo_client with none of the session/catalog/CSV/Excel behavior. Both
now call record_capture() below instead, so there's exactly one
definition of "what happens when a capture is saved":

    1. Resolve today's session (session_manager) — created on first
       call of the day, reused after that.
    2. Write the object document (mongo_client.save_object).
    3. Write one image document per saved photo (mongo_client.
       save_image_record).
    4. Match/link into the object_catalog ("inventory" view).
    5. Append one row to the CSV audit log (always, both modes).
    5b. Append one row to the JSON audit log too (vision.storage.
        json_logger — same metadata as the CSV row, native JSON,
        including the freeform attributes as a real nested object
        instead of a JSON-encoded string).
    6. Refresh the Excel report for today's session (best-effort — a
       failure here is logged and returned as a warning, NOT raised,
       since Mongo/CSV already succeeded by this point and the whole
       capture shouldn't be lost over a locked Excel file).

Steps 1-4 all need to succeed for the capture to count as "saved" —
Mongo is the safety-critical part. Steps 5-6 are best-effort logging on
top and are allowed to fail independently.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Tuple

from vision.storage import attribute_schema, csv_logger, excel_export, json_logger, mongo_client, object_catalog, session_manager


def build_object_data(name: str, category: str = None, color: str = None,
                       size: str = None, position: Dict[str, float] = None,
                       attributes: Dict = None, **reserved_overrides) -> dict:
    """
    Assembles a fully-populated `data` dict (every fixed column present,
    even if null) from whatever the caller actually knows. This is the
    one place that should build that dict — csv_logger/excel_export/
    mongo_client all just read whatever's already in it, so a schema
    change (vision.storage.attribute_schema) only needs this function
    updated, if anything.
    """
    data = attribute_schema.fixed_column_defaults()
    data["name"] = name
    if category is not None:
        data["category"] = category
    if color is not None:
        data["color"] = color
    if size is not None:
        data["size"] = size
    position = position or {}
    for axis in ("x", "y", "z", "r"):
        if axis in position:
            data[f"position_{axis}"] = position[axis]
    for key, value in reserved_overrides.items():
        if key in data:
            data[key] = value
    data[attribute_schema.freeform_key()] = attributes or {}
    return data


def record_capture(name: str, image_paths_by_source, category: str = None,
                    color: str = None, size: str = None,
                    position: Dict[str, float] = None, attributes: Dict = None,
                    export_excel: bool = True) -> Tuple[str, List[str]]:
    """
    Records one full capture: one object, one-or-more images. This is
    what both the manual GUI capture button and logger_service.py call.

    Args:
        name: object name/label (drives both the fixed "name" column
              and the catalog auto-match).
        image_paths_by_source: EITHER {source_name: saved_image_path}
              (the original, simple shape — at most one image per
              camera, e.g. {"station": "...", "wrist": "..."}), OR a
              list of (source_name, saved_image_path) 2-tuples, OR a
              list of (source_name, saved_image_path, view_label)
              3-tuples — needed whenever the SAME source appears more
              than once, e.g. a rotation-sequence capture with 6 photos
              all from the "station" camera at different J4 angles, or
              a lens-extraction capture with several channels/views
              from the same camera. A plain dict can't represent that
              (duplicate keys collide), so the list form is what
              vision.services.rotation_coordinator-based and lens-
              extraction capture flows use; every other caller can keep
              passing a plain dict (or 2-tuples) unchanged. The
              3-tuple's view_label (e.g. "primary_split_r",
              "primary_depth") is stored on the image record so the
              GUI/database can show WHICH channel/view a photo actually
              is instead of just a bare view-index number — see
              vision.camera.capture.save_image/capture_frames_multi.
        category/color/size/position/attributes: see build_object_data().
        export_excel: if False, skips step 6 (e.g. for a burst of many
              rapid captures where you'd rather refresh the report once
              at the end than after every single one).

    Returns:
        (object_id, warnings) — warnings is a list of non-fatal issues
        (e.g. "Excel refresh failed: ..."); an empty list means every
        step succeeded.
    """
    warnings: List[str] = []
    # BUGFIX (folder names still random/misleading): package_export.py
    # builds its per-object export folder straight from this object_id
    # (images/<object_id>/...). Plain date+time, no "object_"/"sample_"
    # word prefix — a folder can hold more than one object/sample, so a
    # semantic prefix like that is misleading; the timestamp is what
    # actually sorts and means something.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_id = f"{timestamp}_{uuid.uuid4().hex[:4]}"
    captured_at = datetime.now()

    session_id = session_manager.get_or_create_today_session()

    data = build_object_data(
        name=name, category=category, color=color, size=size,
        position=position, attributes=attributes,
    )

    # Mongo first — the only side safe for concurrent/live writes.
    mongo_client.save_object(object_id, session_id, str(captured_at.date()), data,
                              captured_at=captured_at)

    # Normalize to a list of (source, path, view_label) triples so a
    # source can legitimately appear more than once (see docstring
    # above) — a plain dict, or a list of plain (source, path) pairs,
    # is accepted as-is for backward compatibility, with view_label
    # defaulting to None.
    if isinstance(image_paths_by_source, dict):
        raw_pairs = list(image_paths_by_source.items())
    else:
        raw_pairs = list(image_paths_by_source)
    source_path_triples = [
        (item[0], item[1], item[2] if len(item) > 2 else None) for item in raw_pairs
    ]

    image_paths = []
    for view_index, (source, path, view_label) in enumerate(source_path_triples):
        image_id = str(uuid.uuid4())
        mongo_client.save_image_record(
            image_id, object_id, path, source, view_index,
            session_id=session_id, captured_at=captured_at, view_label=view_label,
        )
        image_paths.append(path)

    catalog_id = object_catalog.match_or_create(
        object_id, name=name, category=category, color=color, size=size,
        captured_at=captured_at,
    )

    try:
        csv_logger.append_capture_row(object_id, session_id, catalog_id, captured_at,
                                       data, image_paths)
    except Exception as e:
        warnings.append(f"CSV log append failed: {e}")

    try:
        json_logger.append_capture_row(object_id, session_id, catalog_id, captured_at,
                                        data, image_paths)
    except Exception as e:
        warnings.append(f"JSON log append failed: {e}")

    if export_excel:
        try:
            excel_export.build_report(session_id=session_id)
        except Exception as e:
            warnings.append(f"Excel refresh failed: {e}")

    return object_id, warnings
