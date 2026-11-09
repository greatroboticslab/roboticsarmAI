"""How long to wait before trying again after a provider rate limit or quota stop (no GUI, no clock of its own).

A provider that says how long to wait (Retry-After) is believed, up to a cap. Otherwise the wait grows each time the same
image hits the limit again, so a per-minute limit costs a minute and a daily quota is probed every half hour until it
reopens. The worker gives up after MAX_AUTO_RETRIES limit hits in a row with no image finished in between.
"""
from __future__ import annotations

BACKOFF_MINUTES = (1, 2, 5, 10, 20, 30)
MAX_PROVIDER_WAIT_S = 6 * 3600  # a Retry-After longer than this is probed sooner than the provider says
MAX_AUTO_RETRIES = 30  # about 14 hours at the 30 minute step
PADDING_S = 3  # past the provider's own time, so the retry does not land just before it


def auto_wait_seconds(retry_after: float | None, consecutive: int) -> float:
    """Seconds to wait after the `consecutive`-th limit hit in a row (1 = the first)."""
    if retry_after and retry_after > 0:
        return min(float(retry_after), MAX_PROVIDER_WAIT_S) + PADDING_S
    step = BACKOFF_MINUTES[min(max(consecutive, 1), len(BACKOFF_MINUTES)) - 1]
    return step * 60.0
