"""
Shared "list objects, filtered by whether Gemini has already labeled
them" logic — used by both the Group and Review screens so the
Unlabeled / Already labeled / All choice behaves identically wherever
it appears, rather than two subtly-different implementations.

This is the direct answer to "make sure I can do this for both
already-labeled data and data that isn't labeled yet" — labeled status
is a FILTER you choose, in both directions, not something that quietly
disappears from view once it's been processed once.
"""

from typing import List

from object_labeling_studio.core import gemini_client, record_store


def list_objects(scope_kwargs: dict, label_filter: str) -> List[dict]:
    """
    scope_kwargs: passed straight to gemini_client.objects_for_scope
        (session_id=/all_history=/start_date=+end_date=).
    label_filter: "unlabeled" | "labeled" | "all".
    """
    objects = gemini_client.objects_for_scope(**scope_kwargs)
    if label_filter == "all":
        return objects
    want_labeled = (label_filter == "labeled")
    return [obj for obj in objects if record_store.has_records_for_object(obj["_id"]) == want_labeled]


def build_scope_kwargs(scope: str, session_id: str = None, start_date: str = None, end_date: str = None) -> dict:
    """scope: "today" | "all" | "range" — mirrors the radio-button
    choices every scope picker in this tool and the main app use."""
    if scope == "all":
        return {"all_history": True}
    if scope == "range":
        return {"start_date": start_date, "end_date": end_date}
    return {"session_id": session_id}
