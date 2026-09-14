"""
[WIRED] Uses Google's Gemini API (multimodal generateContent) to look
at a captured object's photo and suggest additional descriptive labels
— likely material, texture, and other visually-inferable properties —
beyond whatever a person typed in by hand. This exists specifically to
help with Roboflow labeling: the parsed result is merged straight into
the object's existing FREEFORM attributes in Mongo (see
vision.storage.attribute_schema), which means it flows automatically
into everywhere those attributes already show up — the Excel/CSV
export, the Inventory Attribute Review screen, AND Roboflow's per-image
metadata (vision.storage.roboflow_export._object_metadata reads from
this exact same freeform dict) — with no separate Roboflow-specific
wiring needed. Enrich first, then export/upload as normal.

CREDENTIALS — SESSION-ONLY, NEVER WRITTEN TO DISK
--------------------------------------------------
Same convention as vision.storage.roboflow_export, by the same design
choice for the same reason: sign_in(api_key) holds the key only in this
module's in-memory _session for the life of the current run; nothing is
ever written to a settings file. See that module's docstring for the
full reasoning — this is a deliberate copy of the same pattern rather
than a second, different convention for the same "should an API key be
allowed to leak into a file" tradeoff.

WHAT THIS TALKS TO
-------------------
Google's Gemini API `generateContent` endpoint
(generativelanguage.googleapis.com/v1beta) — the classic, stateless
REST endpoint (Google's newer "Interactions API" is recommended for
agentic multi-turn work; this is a single request/response image-in-
labels-out call, which generateContent fits directly). The model name
is configurable (default "gemini-2.5-flash") since Google's model
lineup moves fast — if the default ever 404s because it's been
retired, the GUI lets you type in whatever current model your API key
actually has access to, rather than needing a code change here.

THIS IS A SUGGESTION, NOT A FACT
----------------------------------
An LLM guessing "what material is this" from a single photo has no way
to verify composition — it can be confidently wrong (mistaking a matte
plastic for ceramic, for instance). Every result is written into the
SAME freeform attribute fields a person could type into by hand
(labelled "AI-suggested" so they're visually distinguishable in the
Excel export/Attribute Review screen, but structurally no different
once saved) — treat these as a labeling head start to review, not as
ground truth to upload unchecked.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from typing import Callable, List, Optional, Tuple

import requests

from vision.storage import attribute_schema, mongo_client, session_manager

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL = "gemini-2.5-flash"

# Freeform attribute keys this module writes into — prefixed/labelled so
# they read clearly as machine-suggested in the Excel export and
# Attribute Review screen, and so "already enriched" can be checked by
# looking for this exact key rather than guessing.
_OBJECT_KEY = "Object (AI suggested)"
_MATERIAL_KEY = "Material (AI suggested)"
_LABELS_KEY = "AI Labels"
_NOTES_KEY = "AI Notes"
# Manually pasted in by the user after sharing the Gemini conversation —
# see save_share_code() below. Never set automatically; Gemini/the API
# has no way to know its own future share URL, since that's only
# created after the fact via the consumer Gemini app's Share button.
_SHARE_CODE_KEY = "Gemini Share Code"

_MAX_429_RETRIES = 4
_BACKOFF_SCHEDULE = [1, 2, 4, 8]

# In-memory ONLY — never written to disk. See module docstring.
_session: Optional[dict] = None  # {"api_key", "model"} or None


class GeminiQueryError(Exception):
    """Raised for anything that should be shown to the user as a plain
    error message — bad/missing key, unreachable API, unparseable
    response, etc."""


def is_signed_in() -> bool:
    return _session is not None


def current_session() -> Optional[dict]:
    """Returns {"api_key", "model"} for the currently signed-in
    session, or None. Callers showing status should use "model" and
    never echo "api_key" back to the screen."""
    return dict(_session) if _session else None


def sign_in(api_key: str, model: str = _DEFAULT_MODEL) -> Tuple[bool, str]:
    """Verifies api_key actually works against Gemini, and if so, holds
    it (plus the chosen model) in memory for the rest of this run — see
    module docstring, nothing is written to disk. Returns (ok, message);
    does NOT change the session on failure."""
    ok, message = verify_credentials(api_key)
    if ok:
        global _session
        _session = {"api_key": api_key.strip(), "model": (model or _DEFAULT_MODEL).strip()}
    return ok, message


def sign_out() -> None:
    """Discards the in-memory session credentials immediately."""
    global _session
    _session = None


def verify_credentials(api_key: str) -> Tuple[bool, str]:
    """Confirms api_key is valid by listing models (a free/cheap call
    that still requires real auth) rather than spending a real
    generation call just to check the key works."""
    if not api_key:
        return False, "API key is required."
    try:
        resp = requests.get(f"{_GEMINI_API_BASE}/models",
                             headers={"x-goog-api-key": api_key}, timeout=10)
        if resp.status_code == 200:
            return True, "Credentials verified."
        return False, f"Gemini rejected this API key (HTTP {resp.status_code}): {resp.text[:200]}"
    except requests.RequestException as e:
        return False, f"Could not reach Gemini: {e}"


def _request_with_backoff(method: str, url: str, **kwargs) -> requests.Response:
    """Same 429-backoff behavior as vision.storage.roboflow_export's
    helper of the same name — see that one's docstring. Duplicated
    rather than shared because the two modules should stay independent
    (one uses `params`, this one a JSON body only) — not worth a shared
    utility module for four lines of retry logic."""
    resp = requests.request(method, url, **kwargs)
    attempt = 0
    while resp.status_code == 429 and attempt < _MAX_429_RETRIES:
        retry_after = resp.headers.get("Retry-After")
        try:
            wait_seconds = float(retry_after) if retry_after else _BACKOFF_SCHEDULE[attempt]
        except ValueError:
            wait_seconds = _BACKOFF_SCHEDULE[attempt]
        time.sleep(wait_seconds)
        resp = requests.request(method, url, **kwargs)
        attempt += 1
    return resp


def _encode_image(image_path: str) -> Tuple[str, str]:
    """Returns (mime_type, base64_data) for one image file, for
    Gemini's inline_data request part."""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return mime_type, data


def _build_prompt(context: dict) -> str:
    """Builds the instruction text sent alongside the image — gives
    Gemini whatever's already known about the object (category/color/
    size/existing freeform attributes) as context, since a human-
    entered category ("ceramic mug") is a much stronger signal than the
    image alone, and asks for STRICT JSON matching a fixed shape so the
    response can be parsed reliably (also enforced via
    generationConfig.response_mime_type below, but this is a second
    layer of the same intent, phrased for the model itself). Matches
    the same object/material/labels prompt worked out for manual use in
    the Gemini chat UI, kept in sync so the automated and manual routes
    produce the same shape of answer."""
    known = ", ".join(f"{k}: {v}" for k, v in context.items() if v not in (None, "")) or "(nothing else known yet)"
    return (
        "I'm labeling photos of physical objects for a computer-vision training dataset. "
        "The camera setup is stationary — it doesn't move between shots. A laser is "
        "projected onto each object as part of an object-identification aid, and lands on "
        "the same spot on the object across its repeat photos. Please ignore the laser dot/"
        "reflection itself when judging material appearance — it's not a feature of the "
        "object, just an identification aid from the capture rig.\n\n"
        "Look at the object in this photo and identify BOTH what the object is and what "
        "it's made of — even if this is difficult or you're not fully certain, give your "
        "best answer rather than skipping it. For material, give a GENERAL category rather "
        "than a precise guess at exact composition — e.g. \"metal,\" \"plastic,\" \"rubber,\" "
        "\"glass,\" \"ceramic,\" \"wood,\" \"fabric,\" \"composite\" — not a specific alloy or "
        "polymer name. If the object is made of more than one material, list EACH material "
        "separately with which part of the object it belongs to, rather than picking just "
        "one.\n\n"
        "Only include a color for a material when the color is a genuinely useful detail "
        "for telling this object apart from a similar one — leave it as an empty string "
        "otherwise.\n\n"
        f"Already known about this object: {known}\n\n"
        "Respond with ONLY a single JSON object, no markdown fences, no extra text, in "
        "exactly this shape:\n"
        '{"object": "<what the object is — be as specific as you reasonably can>", '
        '"materials": [{"part": "<part name, or \'overall\' if one material throughout>", '
        '"material": "<GENERAL material category>", "color": "<only if genuinely useful, '
        'else empty string>", "confidence": "high|medium|low"}], '
        '"labels": ["<label1>", "<label2>", ...], '
        '"notes": "<anything a human reviewer should double-check, or empty string>"}\n\n'
        "Don't leave \"object\" or a material vague just because you're unsure — give your "
        "best specific guess and reflect uncertainty in \"confidence\"/\"notes\" instead. "
        "\"labels\" can be as long and detailed as useful — no length limit."
    )


def _extract_json(raw_text: str) -> dict:
    """Best-effort JSON extraction — response_mime_type="application/
    json" in the request below should make this unnecessary in
    practice, but models occasionally wrap output in fences anyway, so
    this stays as a fallback rather than trusting that flag blindly."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    if not text.startswith("{"):
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise GeminiQueryError(f"Couldn't parse Gemini's response as JSON: {e}\nRaw response: {raw_text[:500]}")


def describe_material(api_key: str, model: str, image_path: str, context: dict = None) -> dict:
    """
    Sends one image + context to Gemini's generateContent endpoint and
    returns the parsed suggestion:
        {"object": str,
         "materials": [{"part": str, "material": str, "color": str, "confidence": str}, ...],
         "labels": [str, ...], "notes": str}
    Raises GeminiQueryError on any failure (bad key, network error,
    unparseable response, missing/unreadable file).
    """
    if not os.path.exists(image_path):
        raise GeminiQueryError(f"Image file not found: {image_path}")
    try:
        mime_type, image_b64 = _encode_image(image_path)
    except OSError as e:
        raise GeminiQueryError(f"Could not read image: {e}")

    url = f"{_GEMINI_API_BASE}/models/{model}:generateContent"
    body = {
        "contents": [{
            "parts": [
                {"text": _build_prompt(context or {})},
                {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            ]
        }],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    try:
        resp = _request_with_backoff(
            "POST", url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=body, timeout=30)
    except requests.RequestException as e:
        raise GeminiQueryError(f"Gemini request failed: {e}")

    if resp.status_code != 200:
        raise GeminiQueryError(f"Gemini returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        payload = resp.json()
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError) as e:
        raise GeminiQueryError(f"Unexpected response shape from Gemini: {e}\nRaw: {resp.text[:500]}")

    parsed = _extract_json(text)
    materials = []
    for m in (parsed.get("materials") or []):
        materials.append({
            "part": str(m.get("part", "overall")).strip() or "overall",
            "material": str(m.get("material", "")).strip(),
            "color": str(m.get("color", "")).strip(),
            "confidence": str(m.get("confidence", "")).strip().lower(),
        })
    return {
        "object": str(parsed.get("object", "")).strip(),
        "materials": materials,
        "labels": [str(x).strip() for x in (parsed.get("labels") or []) if str(x).strip()],
        "notes": str(parsed.get("notes", "")).strip(),
    }


def pick_representative_image(object_id: str) -> Optional[str]:
    """Picks one photo to send to Gemini for a given object — prefers
    an actual split/original view over a disparity/depth output (which
    is a distance map, not something Gemini can usefully describe the
    material of), and skips anything whose file no longer exists on
    disk. Returns None if nothing usable is found. Not underscore-
    prefixed: scripts/object_labeling_studio.py reuses this directly
    to pick the same photo it then runs laser-dot detection on, so the
    Gemini description and the laser-point annotation are always about
    the exact same image."""
    candidates = mongo_client.get_images_for_object(object_id)
    depth_like = ("depth", "disparity")
    ranked = sorted(
        candidates,
        key=lambda img: any(term in (img.get("view_label") or "").lower() for term in depth_like),
    )
    for img in ranked:
        path = img.get("image_path", "")
        if path and os.path.exists(path):
            return path
    return None


def _context_from_object(obj: dict) -> dict:
    """Builds the "already known" context dict passed to
    describe_material() from an object's existing fixed + freeform
    attributes — excludes this module's own AI_* keys so a re-run
    doesn't feed Gemini its own previous guess as if it were a known
    fact."""
    data = obj.get("data") or {}
    context = {key: data.get(key) for key in attribute_schema.fixed_column_keys() if data.get(key) not in (None, "")}
    freeform = data.get(attribute_schema.freeform_key()) or {}
    for key, value in freeform.items():
        if key in (_OBJECT_KEY, _MATERIAL_KEY, _LABELS_KEY, _NOTES_KEY, _SHARE_CODE_KEY):
            continue
        if value not in (None, ""):
            context[key] = value
    return context


def is_object_enriched(obj: dict) -> bool:
    """True if this object already has an AI-suggested material on
    file — used to skip re-querying Gemini (and re-spending API calls)
    on something already labeled, same "don't waste calls redoing work"
    principle as vision.storage.roboflow_export's upload-skip tracking."""
    freeform = (obj.get("data") or {}).get(attribute_schema.freeform_key()) or {}
    return bool(freeform.get(_MATERIAL_KEY))


def format_materials(materials: list) -> str:
    """Turns the structured materials list into one readable string for
    the freeform attribute cell — e.g. "overall: metal; handle: rubber
    (black)" — since Excel/Attribute Review show freeform values as
    plain text, not nested structures."""
    parts = []
    for m in materials:
        piece = f"{m['part']}: {m['material']}"
        if m.get("color"):
            piece += f" ({m['color']})"
        parts.append(piece)
    return "; ".join(parts)


def save_share_code(object_id: str, share_code: str) -> None:
    """Saves a manually-provided Gemini share code (the part after
    gemini.google.com/share/... once you've shared the conversation
    yourself — Gemini/this module has no way to generate this itself,
    since the share URL only exists after the conversation is already
    over) onto an object's freeform attributes, alongside whatever
    enrich_object() already saved. Safe to call before OR after
    enrich_object() — this only ever touches _SHARE_CODE_KEY."""
    obj = mongo_client.get_object(object_id)
    if obj is None:
        raise GeminiQueryError(f"Object '{object_id}' not found.")
    data = dict(obj.get("data") or {})
    freeform_key = attribute_schema.freeform_key()
    freeform = dict(data.get(freeform_key) or {})
    freeform[_SHARE_CODE_KEY] = share_code.strip()
    data[freeform_key] = freeform
    mongo_client.update_object_data(object_id, data)


def enrich_object(object_id: str, api_key: str, model: str) -> dict:
    """
    End to end for ONE object: picks a representative photo, sends it
    (with the object's existing attributes as context) to Gemini, and
    merges the result into that object's freeform attributes in Mongo
    under _OBJECT_KEY/_MATERIAL_KEY/_LABELS_KEY/_NOTES_KEY —
    overwriting only those four keys, leaving every other attribute
    (human-entered or otherwise, including _SHARE_CODE_KEY) untouched.

    Returns the parsed Gemini result (see describe_material), PLUS the
    image_path actually used, since callers doing laser-dot detection
    (see vision.services.laser_dot) need to run it on this exact same
    photo. Raises GeminiQueryError if there's no usable photo for this
    object, or if the Gemini call itself fails.
    """
    obj = mongo_client.get_object(object_id)
    if obj is None:
        raise GeminiQueryError(f"Object '{object_id}' not found.")
    image_path = pick_representative_image(object_id)
    if image_path is None:
        raise GeminiQueryError(f"Object '{object_id}' has no readable saved photo to send to Gemini.")

    context = _context_from_object(obj)
    result = describe_material(api_key, model, image_path, context)

    data = dict(obj.get("data") or {})
    freeform_key = attribute_schema.freeform_key()
    freeform = dict(data.get(freeform_key) or {})
    freeform[_OBJECT_KEY] = result["object"]
    freeform[_MATERIAL_KEY] = format_materials(result["materials"])
    freeform[_LABELS_KEY] = ", ".join(result["labels"])
    freeform[_NOTES_KEY] = result["notes"]
    data[freeform_key] = freeform
    mongo_client.update_object_data(object_id, data)

    result["image_path"] = image_path
    return result


def objects_for_scope(session_id: str = None, all_history: bool = False,
                       start_date: str = None, end_date: str = None,
                       skip_enriched: bool = True) -> Tuple[List[str], int]:
    """
    Same session/all-history/date-range scoping convention as
    vision.storage.roboflow_export.gather_images_for_scope — but at
    OBJECT granularity (one entry per captured object, not per photo),
    since a material suggestion is a property of the object, not of
    any one of its split/depth views.

    Returns (object_ids, already_enriched_count) — skip_enriched=True
    (default) leaves out anything is_object_enriched() already covers,
    with the count of how many were skipped so the GUI can report it
    rather than an unexplained-looking smaller total.
    """
    if start_date and end_date:
        objects = mongo_client.objects_in_date_range(start_date, end_date)
    elif all_history:
        objects = mongo_client.list_recent_objects(limit=100000, sort_ascending=True)
    else:
        session_id = session_id or session_manager.today_session_id()
        objects = mongo_client.find_objects({"session_id": session_id}, limit=100000, sort_ascending=True)

    object_ids = []
    already_enriched_count = 0
    for obj in objects:
        if skip_enriched and is_object_enriched(obj):
            already_enriched_count += 1
            continue
        object_ids.append(obj["_id"])
    return object_ids, already_enriched_count


def enrich_objects(object_ids: List[str], api_key: str, model: str,
                    progress_cb: Callable[[int, int, dict], None] = None,
                    should_cancel: Callable[[], bool] = None) -> Tuple[int, List[dict]]:
    """
    Runs enrich_object() over a list of object ids, one at a time (no
    batch endpoint on Gemini's side any more than Roboflow's upload has
    one). Same shape/conventions as
    vision.storage.roboflow_export.upload_images(): a small pacing gap
    between requests, `should_cancel()` checked between (not during) an
    in-flight request, `progress_cb(done, total, result)` called after
    every attempt with "ok"/"message" added.

    Returns (success_count, failures).
    """
    total = len(object_ids)
    success_count = 0
    failures: List[dict] = []
    for i, object_id in enumerate(object_ids, start=1):
        if should_cancel and should_cancel():
            break
        if i > 1:
            time.sleep(0.25)
        result = {"object_id": object_id}
        try:
            parsed = enrich_object(object_id, api_key, model)
            result.update(ok=True, message=f"object: {parsed['object'] or '(none)'}")
            success_count += 1
        except GeminiQueryError as e:
            result.update(ok=False, message=str(e))
            failures.append(result)
        if progress_cb:
            progress_cb(i, total, result)
    return success_count, failures
