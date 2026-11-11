"""Estimate time and cost before a run, and remember how long images actually took (spec sections 6 and 14)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.providers.cost import CostTracker

DEFAULT_SECONDS_PER_IMAGE = 10.0  # until a real run has been timed
DEFAULT_TOKENS = (3000, 500)  # used only when no transcript has token counts yet


@dataclass
class RunEstimate:
    images: int
    seconds: float
    cost_usd: float | None  # None when no price is set (free or local)
    spent_today: float
    cap_usd: float | None
    tokens_are_measured: bool

    @property
    def over_cap(self) -> bool:
        return bool(self.cap_usd is not None and self.cost_usd is not None and self.spent_today + self.cost_usd > self.cap_usd)

    def describe(self) -> str:
        hours = self.seconds / 3600
        when = f"{hours:.1f} hours" if hours >= 1 else f"{self.seconds / 60:.0f} minutes"
        cost = ("no price set, so cost is not estimated" if self.cost_usd is None else f"about ${self.cost_usd:.2f}"
                + ("" if self.tokens_are_measured else " (rough: no past runs to measure from)"))
        cap = f"; ${self.spent_today:.2f} of the ${self.cap_usd:.2f} daily cap is already used" if self.cap_usd is not None else ""
        return f"{self.images} image(s): about {when}, {cost}{cap}."


def _stats_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "run_stats.json"


def seconds_per_image(data_dir: str | Path) -> float:
    try:
        value = float(json.loads(_stats_path(data_dir).read_text(encoding="utf-8"))["seconds_per_image"])
        return value if value > 0 else DEFAULT_SECONDS_PER_IMAGE
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_SECONDS_PER_IMAGE


def save_seconds_per_image(data_dir: str | Path, value: float) -> None:
    if value > 0:
        path = _stats_path(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seconds_per_image": round(value, 2)}), encoding="utf-8")


def average_tokens(transcripts_dir: str | Path, sample: int = 25) -> tuple[float, float] | None:
    """Mean (input, output) tokens per image over the newest transcripts, or None if there are none with counts."""
    return _average_tokens(Path(transcripts_dir), sample)


def _average_tokens(folder: Path, sample: int) -> tuple[float, float] | None:
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:sample] if folder.is_dir() else []
    totals = []
    for path in files:
        try:
            usage = json.loads(path.read_text(encoding="utf-8")).get("usage") or {}
        except (OSError, ValueError):
            continue
        if usage.get("input_tokens"):
            totals.append((usage["input_tokens"], usage.get("output_tokens", 0)))
    if not totals:
        return None
    return sum(t[0] for t in totals) / len(totals), sum(t[1] for t in totals) / len(totals)


def run_stats(data_dir: str | Path) -> tuple[tuple[float, float] | None, float]:
    """What past runs measured: (mean tokens per image or None, seconds per image). Reads files, so callers that ask
    often (the tick boxes) keep the answer for a while instead of asking again."""
    return average_tokens(Path(data_dir) / "transcripts"), seconds_per_image(data_dir)


def estimate_run(images: int, settings: dict, data_dir: str | Path,
                 stats: tuple[tuple[float, float] | None, float] | None = None) -> RunEstimate:
    block = settings["provider"]
    tracker = CostTracker(block.get("price_in_per_mtok", 0), block.get("price_out_per_mtok", 0),
                          block.get("daily_cost_cap_usd"), Path(data_dir) / "spend.json")
    measured, seconds = stats if stats is not None else run_stats(data_dir)
    tokens = measured or DEFAULT_TOKENS
    priced = bool(block.get("price_in_per_mtok") or block.get("price_out_per_mtok"))
    return RunEstimate(images, images * seconds,
                       tracker.estimate(images, *tokens) if priced else None,
                       tracker.spent_today, block.get("daily_cost_cap_usd"), measured is not None)
