"""
Tab 1 of the Import screen — a generic folder of images with no known
structure. Since there's no manifest to trust here (unlike Tab 2,
core/import_package.py, which reads this app's own structured export),
everything below is a best-effort GUESS, always shown as an editable
preview before anything is written to Mongo — see scan_folder()'s
docstring for exactly what's inferred from what, and why each rule
exists.
"""

import os
import uuid
from datetime import datetime
from typing import List

from PIL import Image

from vision.storage import mongo_client, storage_location

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Filename tokens (case-insensitive, matched as a whole underscore/
# hyphen-delimited segment) that suggest which view/side an image is —
# reused from this app's own save_image() naming convention
# (<timestamp>_<source>_<view_index>_<view_label>.jpg) where possible,
# so an image re-imported from a package export or an old manual backup
# of this app's own output still gets its real view label back instead
# of a generic "imported" one.
_VIEW_LABEL_TOKENS = ("left", "right", "top", "bottom", "split_r", "split_g", "split_b",
                      "depth", "disparity", "original")


class PreviewImage:
    """One row in the import preview — mutable, since the GUI lets you
    edit any of these before committing."""

    def __init__(self, path: str):
        self.path = path
        self.filename = os.path.basename(path)
        self.object_group = os.path.basename(os.path.dirname(path)) or "ungrouped"
        self.view_label = self._infer_view_label()
        self.captured_at = self._infer_captured_at()
        self.source_note = self._source_note()

    def _infer_view_label(self) -> str:
        stem = os.path.splitext(self.filename)[0].lower()
        tokens = stem.replace("-", "_").split("_")
        for token in _VIEW_LABEL_TOKENS:
            if token in tokens or token in stem:
                return token
        return ""

    def _infer_captured_at(self):
        try:
            with Image.open(self.path) as img:
                exif = img.getexif()
                # Tag 306 = DateTime, 36867 = DateTimeOriginal (EXIF spec)
                raw = exif.get(36867) or exif.get(306)
                if raw:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
        try:
            return datetime.fromtimestamp(os.path.getmtime(self.path))
        except OSError:
            return None

    def _source_note(self) -> str:
        notes = []
        if self.object_group != "ungrouped":
            notes.append(f"folder name '{self.object_group}'")
        if self.view_label:
            notes.append(f"filename contains '{self.view_label}'")
        if not notes:
            notes.append("no pattern recognized — using defaults")
        return "; ".join(notes)


def scan_folder(root_dir: str) -> List[PreviewImage]:
    """
    Walks root_dir for image files and builds a PreviewImage per file,
    inferring:
      - object grouping: the immediate parent folder name (so
        `washers/img1.jpg, img2.jpg` groups those two together; a flat
        folder with no subfolders groups everything under "ungrouped",
        which the Group screen then expects you to split up manually —
        there's no way to guess object boundaries from unstructured
        flat files).
      - view label: a small set of recognized filename tokens (see
        _VIEW_LABEL_TOKENS) — most useful for re-importing this app's
        OWN previously-exported/backed-up images, which already follow
        that naming convention; anything else gets no view label
        rather than a wrong guess.
      - captured_at: EXIF DateTimeOriginal if present, else the file's
        own filesystem modified time as a fallback (screenshots/
        resaved images often lack EXIF entirely — this is a genuine
        best-effort, not a guarantee of accuracy).

    Returns the list unsorted-by-meaning (by however os.walk() orders
    them) — the GUI is expected to group/sort for display.
    """
    results = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(_IMAGE_EXTENSIONS):
                results.append(PreviewImage(os.path.join(dirpath, filename)))
    return results


def commit_import(previews: List[PreviewImage]) -> dict:
    """
    Writes the (possibly hand-edited) preview list to Mongo: one new
    object per distinct `object_group` value across the list, one
    image record per PreviewImage, copied into this app's normal
    imported-images storage location so nothing here depends on the
    source folder still existing afterward.

    Returns {"objects_created": int, "images_created": int}.
    """
    groups: dict = {}
    for p in previews:
        groups.setdefault(p.object_group, []).append(p)

    objects_created = 0
    images_created = 0
    for group_name, images in groups.items():
        object_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        earliest = min((p.captured_at for p in images if p.captured_at), default=datetime.now())
        mongo_client.save_object(
            object_id, session_id="imported", date_str=earliest.strftime("%Y-%m-%d"),
            data={"name": group_name, "attributes": {"Import Source": "Filesystem"}},
            captured_at=earliest,
        )
        objects_created += 1

        dest_dir = os.path.join(storage_location.imported_images_root(), object_id)
        os.makedirs(dest_dir, exist_ok=True)
        for view_index, p in enumerate(images):
            dest_path = os.path.abspath(os.path.join(dest_dir, p.filename))
            if os.path.abspath(p.path) != dest_path:
                import shutil
                shutil.copy2(p.path, dest_path)
            image_id = str(uuid.uuid4())
            mongo_client.save_image_record(
                image_id, object_id, dest_path, source="imported", view_index=view_index,
                captured_at=p.captured_at, view_label=p.view_label or None,
            )
            images_created += 1

    return {"objects_created": objects_created, "images_created": images_created}
