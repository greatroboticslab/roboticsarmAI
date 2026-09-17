"""
Shared bootstrap for every module/screen in this tool.

Importing this first (before any `from vision...` import) makes sure
the main repo's root is on sys.path, since this tool lives in its own
top-level folder (object_labeling_studio/) rather than inside vision/ —
see the top-level README for why (kept separate from main.py on
purpose), and for which vision.* modules this tool imports rather than
re-implements (mongo_client, roboflow_export, attribute_schema,
session_manager, package_export, storage_location — the shared data
layer both programs need to agree on) versus which it owns outright
(everything under object_labeling_studio/core and /screens).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imported here (after the sys.path fix above) so every other module in
# this tool can just do `from object_labeling_studio.config import
# gemini_records_root` etc. without repeating the bootstrap.
from vision.storage import storage_location  # noqa: E402

GEMINI_RECORDS_ROOT = storage_location.gemini_records_root()
