"""
Tab 2 of the Import screen — a Data Package folder, as produced by the
main app's own Export Range Package / Export Data Package buttons.

Unlike core/import_filesystem.py's generic folder scan, there's no
guessing needed here at all: vision.storage.package_export.
import_package() already reads the package's captures_log.csv manifest
(real object ids, real attributes, real image-to-object mapping) and
writes it straight to Mongo. This module is a thin pass-through rather
than a reimplementation, specifically so the two stay in sync — a
future change to the package format only has to be handled once, in
package_export.py, not duplicated here.
"""

from vision.storage import package_export


def import_from_package(package_dir: str) -> dict:
    """
    Returns {"imported": int, "skipped": int, "warnings": [str, ...]}.
    Every object gets a brand-new random id on this machine — see
    package_export.import_package's docstring for why (the ORIGINAL
    object id from the source machine is preserved in the freeform
    attributes for traceability, in case you need to cross-reference
    back to where a package came from).
    """
    imported, skipped, warnings = package_export.import_package(package_dir)
    return {"imported": imported, "skipped": skipped, "warnings": warnings}
