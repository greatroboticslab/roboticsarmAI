"""
[WIRED] Export captured images straight to a Roboflow project, so data
collected here can be turned into a labeled dataset over there without
manually downloading/re-uploading files one at a time.

This mirrors (and is meant to feel familiar next to) the same pattern
server-4dai/UI/pages/roboflow.py already uses for the separate web app —
same Roboflow REST endpoints, same "clean the freeform attribute keys
into valid metadata field names" behavior — but wired directly against
THIS machine's local MongoDB (via mongo_client) and local image files,
so it works for the desktop app without needing the FastAPI server or
the Streamlit UI running at all.

CREDENTIALS — SESSION-ONLY, NEVER WRITTEN TO DISK
--------------------------------------------------
The API key is NOT persisted anywhere. sign_in(api_key, workspace,
project_id) verifies the three actually match a real Roboflow project,
then holds them ONLY in this module's in-memory _session dict for the
lifetime of the current run — nothing is ever written to a settings
file. Closing/restarting the app (or calling sign_out()) discards them
completely; there is no "saved config" to accidentally leave behind on
a shared machine. The GUI's Sync & Storage tab enforces "sign in each
time" on top of this by never pre-filling the API key field and by
calling sign_out() itself when the app closes — see main.py.

This intentionally replaces an earlier version of this module that DID
persist credentials to a local roboflow_settings.json (plaintext, same
pattern as camera_settings.json). That's gone now, by request — an
in-memory-only credential is simpler AND strictly more secure than an
encrypted-at-rest file would have been (there's nothing on disk for a
weak/reused encryption key, a stolen backup, or another process running
as the same OS user to ever get at), so that's what this implements
instead of encryption.

VERIFY
------
verify_credentials(api_key, workspace, project_id) does a harmless GET
against Roboflow's own project-info endpoint to confirm the key/
workspace/project actually match something real before sign_in() holds
onto them — same check server-4dai's page does before letting a config
be used.

SCOPE + UPLOAD
--------------
gather_images_for_scope(...) reuses the exact same session/all-history/
date-range scoping vision.storage.package_export.export_package() uses,
so "what counts as today's/this range's captures" stays consistent
between the two export paths — but instead of copying files to a
folder, it returns one dict per saved image (path + cleaned metadata)
ready to hand to upload_images().

upload_images(...) POSTs each one to Roboflow's
https://api.roboflow.com/dataset/<project_id>/upload endpoint, then
attaches its metadata with a follow-up call to attach_metadata() (see
that function's docstring for why this is two calls, not one), calling
`progress_cb(done, total, last_result)` after every attempt so a GUI
can show live progress without needing to poll or block until the
whole batch finishes.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple

import requests

from vision.storage import attribute_schema, mongo_client, session_manager

_ROBOFLOW_API_BASE = "https://api.roboflow.com"

# In-memory ONLY — never written to disk. See module docstring.
_session: Optional[dict] = None  # {"api_key", "workspace", "project_id"} or None


def is_signed_in() -> bool:
    return _session is not None


def current_session() -> Optional[dict]:
    """Returns {"api_key", "workspace", "project_id"} for the currently
    signed-in session, or None if nobody's signed in this run. Callers
    needing to show a non-secret status line should use workspace/
    project_id from this and never echo api_key back to the screen."""
    return dict(_session) if _session else None


def sign_in(api_key: str, workspace: str, project_id: str) -> Tuple[bool, str]:
    """Verifies api_key/workspace/project_id against Roboflow, and if
    they check out, holds them in memory for the rest of this run (see
    module docstring — nothing is written to disk). Returns (ok,
    message); does NOT change the session on failure, so a bad retry
    can't clobber an already-good sign-in."""
    ok, message = verify_credentials(api_key, workspace, project_id)
    if ok:
        global _session
        _session = {"api_key": api_key.strip(), "workspace": workspace.strip(),
                     "project_id": project_id.strip()}
    return ok, message


def sign_out() -> None:
    """Discards the in-memory session credentials immediately — for an
    explicit "Sign Out" button, and called automatically on app exit
    (see main.py) so nothing lingers in memory longer than the run that
    signed in."""
    global _session
    _session = None


def verify_credentials(api_key: str, workspace: str, project_id: str) -> Tuple[bool, str]:
    """Confirms api_key/workspace/project_id actually match a real,
    reachable Roboflow project — so a typo isn't discovered only after
    trying to upload 200 images. Returns (ok, message)."""
    if not (api_key and workspace and project_id):
        return False, "API key, workspace, and project ID are all required."
    try:
        resp = requests.get(f"{_ROBOFLOW_API_BASE}/{workspace}/{project_id}",
                             params={"api_key": api_key}, timeout=10)
        if resp.status_code == 200:
            return True, "Credentials verified."
        return False, f"Roboflow rejected these credentials (HTTP {resp.status_code}): {resp.text[:200]}"
    except requests.RequestException as e:
        return False, f"Could not reach Roboflow: {e}"


def _clean_metadata_key(key: str) -> str:
    """Roboflow's upload metadata expects plain field-name-safe keys —
    same cleanup server-4dai/UI/pages/roboflow.py already applies:
    strip anything that's not alphanumeric/space, then turn spaces into
    underscores (e.g. "Is Metal?" -> "Is_Metal")."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", key)
    return cleaned.strip().replace(" ", "_")


def _json_safe(value):
    """Coerces one metadata value to something json.dumps (and
    therefore requests' `json=` body encoding) can actually handle.
    Freeform attributes are typed by whatever the user/GUI put into
    them — normally plain strings/numbers/booleans, but nothing
    stops something less ordinary (a datetime, a Mongo ObjectId, a
    nested dict) from ending up in there. Without this, a single
    odd value anywhere in the batch would raise a bare TypeError
    from inside requests' JSON encoder — NOT a requests.RequestException
    — which would escape attach_metadata()'s exception handling
    entirely and silently kill the whole upload thread partway
    through a batch, with no error shown and every image after it
    simply never attempted. Bools/numbers/strings/None pass through
    unchanged; everything else becomes its str()."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _object_metadata(obj: dict) -> dict:
    """Freeform + fixed attribute columns for one captured object,
    cleaned into Roboflow-safe metadata field names — the same fields
    that end up in the Excel/CSV export, just relabeled for Roboflow."""
    data = obj.get("data") or {}
    metadata = {}
    for key in attribute_schema.fixed_column_keys():
        if data.get(key) not in (None, ""):
            metadata[_clean_metadata_key(key)] = _json_safe(data[key])
    freeform = data.get(attribute_schema.freeform_key()) or {}
    for key, value in freeform.items():
        if value not in (None, ""):
            metadata[_clean_metadata_key(str(key))] = _json_safe(value)
    metadata.setdefault("object_id", obj.get("_id", ""))
    metadata.setdefault("session_id", obj.get("session_id", ""))
    return metadata


def gather_images_for_scope(session_id: str = None, all_history: bool = False,
                             start_date: str = None, end_date: str = None) -> List[dict]:
    """
    Same scoping rules as vision.storage.package_export.export_package
    (checked in this order: start_date+end_date range > all_history >
    session_id, defaulting to today) — but returns one dict per saved
    image instead of copying files to a folder:

        {"image_id", "object_id", "image_path", "source", "view_label",
         "custom_label", "metadata": {...cleaned object attributes...}}

    Only images whose file actually still exists on disk are included;
    anything missing is silently skipped here (the caller — the GUI —
    is the one that reports totals/warnings to the user, same division
    of responsibility as export_package).
    """
    if start_date and end_date:
        objects = mongo_client.objects_in_date_range(start_date, end_date)
    elif all_history:
        objects = mongo_client.list_recent_objects(limit=100000, sort_ascending=True)
    else:
        session_id = session_id or session_manager.today_session_id()
        objects = mongo_client.find_objects({"session_id": session_id}, limit=100000, sort_ascending=True)

    images = []
    for obj in objects:
        object_id = obj["_id"]
        metadata = _object_metadata(obj)
        for img in mongo_client.get_images_for_object(object_id):
            path = img.get("image_path", "")
            if not path or not os.path.exists(path):
                continue
            images.append({
                "image_id": img.get("_id"),
                "object_id": object_id,
                "image_path": path,
                "source": img.get("source", ""),
                "view_label": img.get("view_label") or "",
                "custom_label": img.get("custom_label") or "",
                "metadata": metadata,
            })
    return images


def upload_image(api_key: str, project_id: str, image_path: str, name: str = None,
                  split: str = None, batch_name: str = None) -> Tuple[bool, str, str]:
    """
    Uploads one image file to Roboflow's per-image upload endpoint:

        POST https://api.roboflow.com/dataset/<project_id>/upload
             ?api_key=...
             -F file=@<image_path>  -F name=... [-F split=...] [-F batch=...]

    This is Roboflow's currently-documented REST upload endpoint (see
    docs.roboflow.com "Manage Images" / "Upload an Image") — multipart
    file upload, exactly like the `curl -F file=@...` example there.
    NOTE: this endpoint does NOT accept a "metadata" field despite
    earlier/other integrations sometimes sending one — any attributes
    to attach have to go through attach_metadata() below, as a SEPARATE
    call, using the image id this upload returns.

    split: one of "train"/"valid"/"test" (Roboflow defaults to "train"
        if omitted).
    batch_name: groups this upload under a named batch in Roboflow's
        web UI (handy for telling "everything uploaded from this GUI
        session" apart from other uploads into the same project).

    Returns (ok, message, image_id). image_id is "" when the upload
    itself failed, when Roboflow doesn't include one in its response,
    or when it comes back as a "duplicate" (an image with identical
    content Roboflow already has — reported as ok=True since the image
    IS present in the project, but with no NEW id to attach metadata
    to; existing duplicates keep whatever metadata they already had).
    """
    url = f"{_ROBOFLOW_API_BASE}/dataset/{project_id}/upload"
    params = {"api_key": api_key}
    form_fields = {"name": name or os.path.basename(image_path)}
    if split:
        form_fields["split"] = split
    if batch_name:
        form_fields["batch"] = batch_name
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(url, params=params, files={"file": f}, data=form_fields, timeout=30)
    except OSError as e:
        return False, f"Could not read file: {e}", ""
    except requests.RequestException as e:
        return False, f"Upload failed: {e}", ""

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}", ""
    try:
        body = resp.json()
    except ValueError:
        return True, "uploaded (non-JSON response, assumed OK)", ""
    if body.get("duplicate"):
        return True, "already on Roboflow (duplicate content, metadata unchanged)", ""
    if body.get("success", True):
        return True, "uploaded", str(body.get("id") or "")
    return False, f"Roboflow reported failure: {body}", ""


def attach_metadata(api_key: str, workspace: str, image_id: str, metadata: dict) -> Tuple[bool, str]:
    """
    Attaches key/value metadata to an already-uploaded image via
    Roboflow's per-image metadata endpoint:

        POST https://api.roboflow.com/<workspace>/images/<image_id>/metadata?api_key=...
        {"metadata": {...}}

    (see docs.roboflow.com "Update Image Metadata and Tags"). This is a
    SEPARATE call from upload_image() above — the upload endpoint has
    no metadata field of its own. Requires an API key with the
    `image:tag` scope; a plain/default-scope key will get a 401/403
    here even though the upload itself succeeded, which is reported as
    a failure message rather than silently swallowed, since the image
    would otherwise sit on Roboflow with none of its captured
    attributes attached and no obvious sign why.
    """
    if not metadata:
        return True, "no metadata to attach"
    url = f"{_ROBOFLOW_API_BASE}/{workspace}/images/{image_id}/metadata"
    try:
        resp = requests.post(url, params={"api_key": api_key},
                              json={"metadata": metadata}, timeout=15)
        if resp.status_code == 200:
            return True, "metadata attached"
        return False, f"metadata attach failed (HTTP {resp.status_code}): {resp.text[:200]}"
    except requests.RequestException as e:
        return False, f"metadata attach failed: {e}"
    except (TypeError, ValueError) as e:
        # Belt-and-suspenders: _object_metadata()/_json_safe() should
        # already keep every value JSON-encodable, but if something
        # unencodable slips through anyway, fail THIS image's metadata
        # attach rather than letting an uncaught exception kill the
        # rest of the batch (see _json_safe's docstring for the exact
        # failure mode this replaces).
        return False, f"metadata not JSON-encodable, skipped: {e}"


def upload_images(api_key: str, workspace: str, project_id: str, images: List[dict],
                   split: str = None, batch_name: str = None,
                   progress_cb: Callable[[int, int, dict], None] = None,
                   should_cancel: Callable[[], bool] = None) -> Tuple[int, List[dict]]:
    """
    Uploads every image in `images` (each shaped like one entry from
    gather_images_for_scope()) to the given Roboflow project one at a
    time — Roboflow's upload endpoint takes a single file per request,
    so there's no batch call to use instead. For each image that
    uploads successfully (and isn't a pre-existing duplicate — see
    upload_image()'s docstring), its metadata is attached with a
    follow-up call to attach_metadata(); a metadata-attach failure is
    recorded as its own failure entry even though the image itself did
    make it into the project, so it's visible rather than silently lost.

    `progress_cb(done, total, result)` is called after EVERY attempt —
    `result` is the same dict from `images`, with "ok" (bool) and
    "message" (str) added, reflecting BOTH the upload and (if it ran)
    the metadata attach — so a GUI showing a single live status line
    doesn't call something "ok" that actually failed to get its
    metadata attached — so a GUI can update a progress bar / status
    line live instead of freezing until the whole batch finishes.

    `should_cancel()`, if given, is checked before every image; once it
    returns True the remaining images are left un-attempted (not
    counted as failures — they were simply never tried) and this
    returns immediately with whatever completed so far. Roboflow's
    upload endpoint has no notion of "abort a batch" server-side, so
    this can only stop BETWEEN images, not partway through one already
    in flight — a large "full history" upload can otherwise run for a
    long time with no way to stop it early.

    Returns (success_count, failures) — success_count only counts
    images that actually made it into the project (upload success OR
    duplicate); a metadata-attach failure on an otherwise-successful
    upload is listed in `failures` but does NOT subtract from
    success_count, since the photo itself is safely on Roboflow either
    way. A failure on one image never stops the rest of the batch.
    """
    total = len(images)
    success_count = 0
    failures: List[dict] = []
    for i, image in enumerate(images, start=1):
        if should_cancel and should_cancel():
            break
        display_name = image.get("view_label") or image.get("source") or image.get("image_id") or "image"
        ok, message, image_id = upload_image(
            api_key, project_id, image["image_path"],
            name=f"{image.get('object_id', 'object')}_{display_name}",
            split=split, batch_name=batch_name,
        )
        if ok:
            success_count += 1
            if image_id and image.get("metadata"):
                meta_ok, meta_message = attach_metadata(api_key, workspace, image_id, image["metadata"])
                if not meta_ok:
                    result = dict(image)
                    result["ok"] = False
                    result["message"] = f"uploaded, but {meta_message}"
                    failures.append(result)
                    ok, message = False, result["message"]  # reflected in progress_cb below too
        else:
            result = dict(image)
            result["ok"] = False
            result["message"] = message
            failures.append(result)
        if progress_cb:
            progress_cb(i, total, {"ok": ok, "message": message, **image})
    return success_count, failures
