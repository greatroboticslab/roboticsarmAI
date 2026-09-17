"""
Tab 3 of the Import screen — pulls images from Roboflow that have NO
annotation yet, so Gemini can pre-label them before you box them by
hand in Roboflow's own UI (see screens/annotate_screen.py's "Label
Only" branch — this import path never does auto-annotation, since
there's no laser dot guaranteed to exist on a backlog image at all).

Two REST calls per image, confirmed against Roboflow's own docs before
writing this (not assumed):
  1. Search  (POST /:workspace/:project/search) — lists images; each
     result's "annotations": {"count": N, ...} field is what "already
     has a box or not" means here. The search response does NOT
     include a downloadable URL.
  2. Image Details (GET /:workspace/:project/image/:image_id) — the
     SEPARATE call needed to actually get one, at
     response["image"]["urls"]["original"].
That's two network round-trips per image being pulled, not one — worth
knowing before pointing this at a huge backlog in one go.
"""

import os
import uuid
from datetime import datetime
from typing import List

import requests

from vision.storage import mongo_client, roboflow_export, storage_location

_ROBOFLOW_API_BASE = "https://api.roboflow.com"


def list_unlabeled_images(api_key: str, workspace: str, project_id: str, limit: int = 250) -> List[dict]:
    """
    Pages through the project's Search API, keeping only images whose
    annotation count is zero. Returns the raw search-result dicts
    (id/name/tags/etc — see Roboflow's docs) for whichever ones qualify
    — no download happens here yet (see pull_image() below for that,
    called per-image only for ones you actually choose to import).
    """
    url = f"{_ROBOFLOW_API_BASE}/{workspace}/{project_id}/search"
    unlabeled = []
    offset = 0
    page_size = min(limit, 250)
    while True:
        resp = requests.post(
            url, params={"api_key": api_key},
            json={"in_dataset": True, "offset": offset, "limit": page_size,
                  "fields": ["id", "name", "annotations", "tags", "split"]},
            timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Roboflow search failed (HTTP {resp.status_code}): {resp.text[:300]}")
        results = resp.json().get("results", [])
        if not results:
            break
        for r in results:
            if (r.get("annotations") or {}).get("count", 0) == 0:
                unlabeled.append(r)
        if len(unlabeled) >= limit or len(results) < page_size:
            break
        offset += page_size
    return unlabeled[:limit]


def pull_image(api_key: str, workspace: str, project_id: str, roboflow_image_id: str,
                dest_dir: str) -> str:
    """
    Fetches one image's real download URL via the Image Details
    endpoint, then downloads it to dest_dir. Returns the local path.
    """
    detail_url = f"{_ROBOFLOW_API_BASE}/{workspace}/{project_id}/image/{roboflow_image_id}"
    resp = requests.get(detail_url, params={"api_key": api_key}, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Could not get details for image {roboflow_image_id} "
                            f"(HTTP {resp.status_code}): {resp.text[:200]}")
    image_url = ((resp.json().get("image") or {}).get("urls") or {}).get("original")
    if not image_url:
        raise RuntimeError(f"Image {roboflow_image_id}'s details had no downloadable URL.")

    image_resp = requests.get(image_url, timeout=30)
    if image_resp.status_code != 200:
        raise RuntimeError(f"Could not download image {roboflow_image_id} "
                            f"(HTTP {image_resp.status_code}).")

    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, f"{roboflow_image_id}.jpg")
    with open(local_path, "wb") as f:
        f.write(image_resp.content)
    return local_path


def import_selected(api_key: str, workspace: str, project_id: str,
                     search_results: List[dict]) -> dict:
    """
    Downloads each given search result (as returned by
    list_unlabeled_images) and creates ONE new local object per image —
    Roboflow has no concept of "these came from the same physical
    object," so this can't group them; use the Group screen afterward
    if some of them actually belong together.

    Each new image record is marked as already uploaded to THIS exact
    Roboflow project (roboflow_export.mark_image_uploaded_to_roboflow),
    since it plainly already is — this prevents the normal Roboflow
    Export panel from trying to re-upload something pulled straight
    FROM Roboflow in the first place.

    Returns {"imported": int, "failed": [{"id", "error"}, ...]}.
    """
    key = roboflow_export.project_key(workspace, project_id)
    imported = 0
    failed = []
    for result in search_results:
        roboflow_image_id = result.get("id")
        try:
            object_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            dest_dir = os.path.join(storage_location.imported_images_root(), object_id)
            local_path = pull_image(api_key, workspace, project_id, roboflow_image_id, dest_dir)

            mongo_client.save_object(
                object_id, session_id="imported_from_roboflow",
                date_str=datetime.now().strftime("%Y-%m-%d"),
                data={"name": result.get("name", roboflow_image_id),
                      "attributes": {"Import Source": "Roboflow (unlabeled backlog)"}},
            )
            image_id = str(uuid.uuid4())
            mongo_client.save_image_record(
                image_id, object_id, local_path, source="roboflow_backlog", view_index=0,
                view_label="original",
            )
            mongo_client.mark_image_uploaded_to_roboflow(image_id, key, roboflow_image_id)
            imported += 1
        except Exception as e:
            failed.append({"id": roboflow_image_id, "error": str(e)})
    return {"imported": imported, "failed": failed}
