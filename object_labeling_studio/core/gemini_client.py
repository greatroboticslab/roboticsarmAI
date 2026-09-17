"""
Gemini vision client for Object Labeling Studio.

Moved here from vision/services/gemini_material_query.py — this tool
owns it outright now rather than sharing it with the main app, since
it's specific to this tool's labeling workflow. Still imports the main
repo's Mongo/attribute-schema layer (see ../config.py's docstring for
why: shared data, not shared UI/tooling).

CREDENTIALS — SESSION-ONLY, NEVER WRITTEN TO DISK
--------------------------------------------------
sign_in(api_key) holds the key only in this module's in-memory
_session for the life of the current run; nothing is ever written to a
settings file — same convention, same reasoning, as
vision.storage.roboflow_export's sign_in.

WHAT THIS TALKS TO
-------------------
Google's Gemini API `generateContent` endpoint
(generativelanguage.googleapis.com/v1beta). Model name is configurable
(default "gemini-2.5-flash") since Google's model lineup moves fast.

THIS IS A SUGGESTION, NOT A FACT
----------------------------------
An LLM guessing "what material is this" from a single photo has no way
to verify composition. Every result flows through the Review screen for
your edit/approval before it's split into records or pushed anywhere —
this module never writes anything to Mongo/Roboflow itself; it only
returns a parsed suggestion for the caller to do something with.
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

from vision.storage import mongo_client, session_manager

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL = "gemini-2.5-flash"

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
    return dict(_session) if _session else None


def sign_in(api_key: str, model: str = _DEFAULT_MODEL) -> Tuple[bool, str]:
    ok, message = verify_credentials(api_key)
    if ok:
        global _session
        _session = {"api_key": api_key.strip(), "model": (model or _DEFAULT_MODEL).strip()}
    return ok, message


def sign_out() -> None:
    global _session
    _session = None


def verify_credentials(api_key: str) -> Tuple[bool, str]:
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
    """429 retry-with-backoff — same behavior as
    vision.storage.roboflow_export's helper of the same name."""
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
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return mime_type, data


def _build_prompt(context: dict) -> str:
    """Same object/material/labels prompt worked out for manual use in
    the Gemini chat UI — kept in sync so the automated and manual
    routes produce the same shape of answer, and so a PDF generated
    from either source (see core/pdf_export.py) reads consistently."""
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
    Sends one image + context to Gemini's generateContent endpoint.
    Returns:
        {"object": str,
         "materials": [{"part": str, "material": str, "color": str, "confidence": str}, ...],
         "labels": [str, ...], "notes": str,
         "prompt_text": str, "response_text": str}
    The last two (prompt_text/response_text) are the VERBATIM text sent
    and received — kept in the return value specifically so
    core/pdf_export.py can render the real exchange, not a
    reconstruction of it.

    Raises GeminiQueryError on any failure.
    """
    if not os.path.exists(image_path):
        raise GeminiQueryError(f"Image file not found: {image_path}")
    try:
        mime_type, image_b64 = _encode_image(image_path)
    except OSError as e:
        raise GeminiQueryError(f"Could not read image: {e}")

    prompt_text = _build_prompt(context or {})
    url = f"{_GEMINI_API_BASE}/models/{model}:generateContent"
    body = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
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
        response_text = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError) as e:
        raise GeminiQueryError(f"Unexpected response shape from Gemini: {e}\nRaw: {resp.text[:500]}")

    parsed = _extract_json(response_text)
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
        "prompt_text": prompt_text,
        "response_text": response_text,
    }


def format_materials(materials: list) -> str:
    """One readable line per materials list — e.g. "overall: metal;
    handle: rubber (black)"."""
    parts = []
    for m in materials:
        piece = f"{m['part']}: {m['material']}"
        if m.get("color"):
            piece += f" ({m['color']})"
        parts.append(piece)
    return "; ".join(parts)


def pick_representative_image(object_id: str) -> Optional[str]:
    """Picks one photo to send to Gemini for a given object — prefers
    an actual split/original view over a disparity/depth output, skips
    anything whose file no longer exists on disk."""
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


def objects_for_scope(session_id: str = None, all_history: bool = False,
                       start_date: str = None, end_date: str = None) -> List[dict]:
    """
    Same session/all-history/date-range scoping convention used
    throughout the main app (see vision.storage.roboflow_export.
    gather_images_for_scope) — returns the raw object documents (not
    just ids), since the Group/Review screens need to show names/
    thumbnails, not just ids.
    """
    if start_date and end_date:
        return mongo_client.objects_in_date_range(start_date, end_date)
    if all_history:
        return mongo_client.list_recent_objects(limit=100000, sort_ascending=True)
    session_id = session_id or session_manager.today_session_id()
    return mongo_client.find_objects({"session_id": session_id}, limit=100000, sort_ascending=True)
