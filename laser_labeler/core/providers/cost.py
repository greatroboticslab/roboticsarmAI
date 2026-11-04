"""Spend tracking against a daily cap. Prices come from user settings, never hardcoded."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Callable

from core.providers.base import Usage

KEEP_DAYS = 60  # older per-day entries are dropped when the store is rewritten


class SpendCapExceeded(RuntimeError):
    """Raised by CostTracker.check() when today's spend has reached the daily cap."""


class CostTracker:
    """Accumulates cost per calendar day, optionally persisted as JSON ({"YYYY-MM-DD": usd})."""

    def __init__(self, price_in_per_mtok: float = 0, price_out_per_mtok: float = 0,
                 daily_cap_usd: float | None = None, store_path: Path | None = None,
                 today: Callable[[], date] = date.today) -> None:
        self.price_in_per_mtok = price_in_per_mtok
        self.price_out_per_mtok = price_out_per_mtok
        self.daily_cap_usd = daily_cap_usd
        self.store_path = Path(store_path) if store_path is not None else None
        self._today = today
        self._days: dict[str, float] = self._load()

    def _key(self) -> str:
        return self._today().isoformat()

    def _load(self) -> dict[str, float]:
        if self.store_path is None or not self.store_path.exists():
            return {}
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            return {str(k): float(v) for k, v in raw.items()}
        except (OSError, ValueError, AttributeError, TypeError):
            return {}  # unreadable store: start fresh rather than block labeling

    def _save(self) -> None:
        if self.store_path is None:
            return
        days = dict(sorted(self._days.items())[-KEEP_DAYS:])
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(json.dumps(days, indent=2), encoding="utf-8")
        os.replace(tmp, self.store_path)

    def cost_of(self, usage: Usage) -> float:
        return (usage.input_tokens * self.price_in_per_mtok
                + usage.output_tokens * self.price_out_per_mtok) / 1_000_000

    @property
    def spent_today(self) -> float:
        return self._days.get(self._key(), 0.0)

    def add(self, usage: Usage) -> float:
        """Record one call's usage; returns the cost added in USD."""
        cost = self.cost_of(usage)
        if self.store_path is not None:
            self._days = self._load()  # pick up spend added by another tracker on the same store
        self._days[self._key()] = self.spent_today + cost
        self._save()
        return cost

    def check(self) -> None:
        """Raise SpendCapExceeded when a cap is set and today's spend has reached it."""
        cap = self.daily_cap_usd
        if self.store_path is not None:
            self._days = self._load()  # another tracker (for example the escalation provider) may have added spend
        if cap is not None and self.spent_today >= cap:
            raise SpendCapExceeded(
                f"Daily spend cap reached: ${self.spent_today:.4f} of ${cap:.2f} spent today")

    def estimate(self, n_images: int, avg_in_tokens: float, avg_out_tokens: float) -> float:
        """Projected cost in USD for n_images calls of the given average size."""
        return n_images * self.cost_of(Usage(avg_in_tokens, avg_out_tokens))
