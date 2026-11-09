"""List, count and download the images of a Roboflow project (REST, stdlib only).

Sources (docs checked 2026-10-27, nothing run live; see docs/verified_facts.md):
- Search: POST /{workspace}/{project}/search. The SDK documents offset, limit
  (default 100), tag, in_dataset and fields. The exact REST response shape could
  not be read from the docs, so parsing is defensive ("results" list, optional "total").
- Image details: GET /{workspace}/{project}/images/{id} -> image.urls.original.
Paging uses offset += len(page), because servers may return fewer rows than asked.
"""
from __future__ import annotations

import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from core.roboflow.auth import RoboflowError
from core.roboflow.http import (build_url, quote_seg, request_bytes, request_json,
                                with_backoff)
from core.schema.scheme_labels import CURRENT, labels_state

DEFAULT_FIELDS = ("id", "name", "split", "tags", "labels", "created")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


@dataclass
class RemoteImage:
    id: str
    name: str = ""
    split: str | None = None
    tags: list[str] = field(default_factory=list)
    labels: list[str] | None = None  # None = the server did not report labels
    width: int | None = None
    height: int | None = None
    created: float | None = None
    url: str | None = None
    workspace: str = ""
    project: str = ""


def _str_list(value: Any) -> list[str]:
    items = value if isinstance(value, list) else []
    return [str(i.get("name") if isinstance(i, dict) else i) for i in items
            if i is not None and (not isinstance(i, dict) or i.get("name"))]


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_image(item: Any, workspace: str = "", project: str = "") -> RemoteImage | None:
    """Build a RemoteImage from one search/detail row; None if it has no id."""
    if not isinstance(item, dict) or not item.get("id"):
        return None
    urls = item.get("urls") if isinstance(item.get("urls"), dict) else {}
    url = urls.get("original") or item.get("url")
    created = item.get("created")
    return RemoteImage(
        id=str(item["id"]), name=str(item.get("name") or ""),
        split=item.get("split") if isinstance(item.get("split"), str) else None,
        tags=_str_list(item.get("tags")),
        labels=_str_list(item["labels"]) if isinstance(item.get("labels"), list) else None,
        width=_int_or_none(item.get("width")), height=_int_or_none(item.get("height")),
        created=float(created) if isinstance(created, (int, float)) else None,
        url=url if isinstance(url, str) else None, workspace=workspace, project=project)


def is_unlabeled(image: RemoteImage, scheme: str | None = None) -> bool | None:
    """True/False from the `labels` field; None when the server did not report it.

    With a `scheme` (full | components | merged) "labeled" means labeled in that system: an image whose labels are
    all of an older kind counts as unlabeled (see core.schema.scheme_labels).

    Caveat: an image saved as "no objects" also has no labels, and looks unlabeled.
    """
    if image.labels is None:
        return None
    if scheme is None:
        return len(image.labels) == 0
    return labels_state(image.labels, scheme) != CURRENT


def _search_page(workspace: str, project: str, api_key: str, body: dict, opener: Callable,
                 timeout: float, sleep: Callable) -> tuple[list, int | None]:
    url = build_url(f"{quote_seg(workspace)}/{quote_seg(project)}/search", api_key)
    data = with_backoff(lambda: request_json("POST", url, body=body, opener=opener, timeout=timeout,
                                             key_for_masking=api_key), sleep=sleep)
    rows = data.get("results") if isinstance(data, dict) else data
    total = data.get("total") if isinstance(data, dict) else None
    return (rows if isinstance(rows, list) else []), (total if isinstance(total, int) else None)


def list_images(workspace: str, project: str, api_key: str, *, only_unlabeled: bool = False,
                scheme: str | None = None, tag: str | None = None, split: str | None = None, page_size: int = 100,
                limit: int | None = None, fields: tuple[str, ...] = DEFAULT_FIELDS,
                opener: Callable = urllib.request.urlopen, timeout: float = 30,
                sleep: Callable[[float], None] = time.sleep) -> Iterator[RemoteImage]:
    """Yield images page by page. `only_unlabeled` keeps images whose labels are empty
    or unknown (use is_unlabeled to tell them apart); with a `scheme` it keeps every image that is not labeled in
    that labeling system (empty, older style, or unknown). `split` is filtered client-side."""
    offset, yielded, seen = 0, 0, set()
    while limit is None or yielded < limit:
        body: dict[str, Any] = {"offset": offset, "limit": page_size, "fields": list(fields)}
        if tag:
            body["tag"] = tag
        rows, total = _search_page(workspace, project, api_key, body, opener, timeout, sleep)
        fresh = 0
        for row in rows:
            image = parse_image(row, workspace, project)
            if image is None or image.id in seen:
                continue
            seen.add(image.id)
            fresh += 1
            if (split and image.split != split) or (only_unlabeled and is_unlabeled(image, scheme) is False):
                continue
            yield image
            yielded += 1
            if limit is not None and yielded >= limit:
                return
        offset += len(rows)
        if fresh == 0 or (total is not None and offset >= total):
            return  # empty page, repeated page (offset ignored) or everything seen


def count_images(workspace: str, project: str, api_key: str, *, only_unlabeled: bool = False,
                 scheme: str | None = None, tag: str | None = None, split: str | None = None,
                 opener: Callable = urllib.request.urlopen, timeout: float = 30,
                 sleep: Callable[[float], None] = time.sleep) -> int:
    """Number of matching images. Uses the search `total` when no client-side filter
    applies; otherwise counts by paging."""
    if not only_unlabeled and not split:
        body: dict[str, Any] = {"offset": 0, "limit": 1, "fields": ["id"]}
        if tag:
            body["tag"] = tag
        _, total = _search_page(workspace, project, api_key, body, opener, timeout, sleep)
        if total is not None:
            return total
    return sum(1 for _ in list_images(workspace, project, api_key, only_unlabeled=only_unlabeled, scheme=scheme,
                                      tag=tag, split=split, opener=opener, timeout=timeout,
                                      sleep=sleep))


def safe_stem(image_id: str) -> str:
    """A file-name-safe version of the image id (never a server-supplied name)."""
    stem = _UNSAFE.sub("_", str(image_id))[:100]
    if not stem.strip("_"):
        raise RoboflowError("The image id cannot be used to build a safe file name.")
    return stem


def existing_download(image: RemoteImage, dest_dir: Path) -> Path | None:
    """The finished file for this image if one exists with nonzero size."""
    for ext in IMAGE_EXTS:
        path = Path(dest_dir) / (safe_stem(image.id) + ext)
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _original_url(image: RemoteImage, api_key: str, opener: Callable, timeout: float,
                  sleep: Callable) -> str:
    if image.url:
        return image.url
    if not (image.workspace and image.project):
        raise RoboflowError("No download URL and no workspace/project to look it up.")
    url = build_url(f"{quote_seg(image.workspace)}/{quote_seg(image.project)}/images/{quote_seg(image.id)}",
                    api_key)
    data = with_backoff(lambda: request_json("GET", url, opener=opener, timeout=timeout,
                                             key_for_masking=api_key), sleep=sleep)
    detail = parse_image(data.get("image") if isinstance(data, dict) else None)
    if detail is None or not detail.url:
        raise RoboflowError(f"Roboflow gave no original image URL for image {image.id}.")
    return detail.url


def _extension(image: RemoteImage, url: str) -> str:
    for source in (urllib.parse.urlparse(url).path, image.name):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_EXTS:
            return ext
    return ".jpg"


def download_image(image: RemoteImage, dest_dir: Path, api_key: str,
                   opener: Callable = urllib.request.urlopen, *, timeout: float = 60,
                   sleep: Callable[[float], None] = time.sleep) -> Path:
    """Download the original image into dest_dir/<safe id><ext> and return the path.

    Resume-safe: an existing nonzero file is returned untouched. Data goes to a
    temporary file first, then is renamed. The API key is NOT sent to the file URL.
    """
    dest = Path(dest_dir)
    done = existing_download(image, dest)
    if done:
        return done
    url = _original_url(image, api_key, opener, timeout, sleep)
    if urllib.parse.urlparse(url).scheme != "https":
        raise RoboflowError("Refusing to download from a URL that is not https.")
    data = with_backoff(lambda: request_bytes("GET", url, opener=opener, timeout=timeout,
                                              key_for_masking=api_key), sleep=sleep)
    if not data:
        raise RoboflowError(f"Roboflow returned an empty file for image {image.id}.")
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / (safe_stem(image.id) + _extension(image, url))
    tmp = final.with_name("." + final.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, final)
    return final
