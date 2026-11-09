"""Download Roboflow images and add them to the labeling queue."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from core.queue_db import NewImage, QueueDB
from core.roboflow.auth import RoboflowError
from core.roboflow.images import RemoteImage, is_unlabeled
from core.schema.scheme_labels import OLDER, labels_state


@dataclass
class EnqueueResult:
    added: int = 0
    already_known: int = 0
    failed: dict[str, str] = field(default_factory=dict)  # image id -> reason
    unknown_label_state: int = 0  # kept, but Roboflow did not say whether they have labels
    older_style: int = 0  # added, and already labeled on Roboflow in an older style (needs Overwrite to replace)


def enqueue_remote(db: QueueDB, images: Iterable[RemoteImage], download: Callable[[RemoteImage], Path], *,
                   limit: int | None = None, on_progress: Callable[[int, int], None] | None = None,
                   should_stop: Callable[[], bool] = lambda: False, scheme: str | None = None) -> EnqueueResult:
    """Download each image not already in the queue and queue it. One bad download does not stop the rest."""
    result = EnqueueResult()
    images = list(images)
    for done, image in enumerate(images):
        if (limit is not None and result.added >= limit) or should_stop():
            break
        if on_progress:
            on_progress(done, len(images))
        if db.get(image.id):
            result.already_known += 1
            continue
        try:
            path = download(image)
        except (RoboflowError, OSError) as exc:
            result.failed[image.id] = str(exc)[:200]
            continue
        db.add_images([NewImage(image.id, image.name or path.name, str(path), "roboflow", image.tags)])
        result.added += 1
        result.unknown_label_state += is_unlabeled(image) is None
        if scheme and image.labels is not None and labels_state(image.labels, scheme) == OLDER:
            result.older_style += 1
    return result
