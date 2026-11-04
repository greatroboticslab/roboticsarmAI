"""Spend guard: stops a run when the daily cap is reached (spec section 6)."""
from __future__ import annotations

from core.labeler.conversation import CostGuard
from core.providers.base import ProviderReply
from core.providers.cost import CostTracker


class TrackerGuard(CostGuard):
    """Checks the cap before each call and adds the usage after it.

    SpendCapExceeded is not caught by label_image: it ends the whole run.
    """

    def __init__(self, tracker: CostTracker) -> None:
        self.tracker = tracker

    def before_call(self) -> None:
        self.tracker.check()

    def after_call(self, reply: ProviderReply) -> None:
        self.tracker.add(reply.usage)
