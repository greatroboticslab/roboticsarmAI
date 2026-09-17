"""
"Label Only" flow — for images that came from Tab 3's Roboflow-backlog
import (or any object you don't want auto-annotated). No laser-dot
detection, no automatic box: Gemini's suggestion is pushed into that
image's Roboflow metadata so it's visible right in Roboflow's own
labeling UI, and a local record (PDF + Mongo) is saved the same way as
every other record — you draw the box yourself in Roboflow afterward.
"""

from typing import Optional

from vision.storage import mongo_client, roboflow_export

from object_labeling_studio.core import gemini_client, record_store


def label_only(object_id: str, gemini_api_key: str, gemini_model: str,
                roboflow_api_key: str, roboflow_workspace: str, roboflow_project_id: str) -> dict:
    """
    End to end for one object pulled from Roboflow's unlabeled backlog:
      1. Run Gemini on its representative photo.
      2. Split the result into records (record_store) — same as the
         normal flow, so the archive doesn't distinguish by source.
      3. Push the pointer + summary (record_store.
         roboflow_metadata_pointer) into that image's Roboflow metadata
         via attach_metadata, so it's visible in Roboflow's UI right
         when you go to box it.

    Returns {"gemini_result": dict, "records": [...], "roboflow_image_id": str}.
    Raises gemini_client.GeminiQueryError or RuntimeError on failure —
    caller (the GUI) is expected to show that message and let the user
    retry, not fail the whole batch.
    """
    obj = mongo_client.get_object(object_id)
    if obj is None:
        raise gemini_client.GeminiQueryError(f"Object '{object_id}' not found.")
    image_path = gemini_client.pick_representative_image(object_id)
    if image_path is None:
        raise gemini_client.GeminiQueryError(f"Object '{object_id}' has no readable photo.")

    object_name = (obj.get("data") or {}).get("name", "")
    gemini_result = gemini_client.describe_material(gemini_api_key, gemini_model, image_path,
                                                      context={"name": object_name})
    records = record_store.create_records_from_gemini_result(object_id, object_name, image_path, gemini_result)

    roboflow_image_id = _find_roboflow_image_id(object_id, image_path, roboflow_workspace, roboflow_project_id)
    if roboflow_image_id:
        pointer = record_store.roboflow_metadata_pointer(object_id)
        roboflow_export.attach_metadata(roboflow_api_key, roboflow_workspace, roboflow_image_id, pointer)

    return {"gemini_result": gemini_result, "records": records, "roboflow_image_id": roboflow_image_id}


def _find_roboflow_image_id(object_id: str, image_path: str, workspace: str, project_id: str) -> Optional[str]:
    key = roboflow_export.project_key(workspace, project_id)
    for img in mongo_client.get_images_for_object(object_id):
        if img.get("image_path") == image_path:
            record = (img.get("roboflow_uploads") or {}).get(key)
            if record:
                return record.get("roboflow_image_id") or None
    return None
