from datetime import date, timedelta

import pytest

from core.providers.base import Usage
from core.providers.cost import CostTracker, SpendCapExceeded


def test_add_uses_user_prices():
    t = CostTracker(price_in_per_mtok=2.0, price_out_per_mtok=10.0)
    cost = t.add(Usage(1_000_000, 500_000))
    assert cost == pytest.approx(7.0)
    assert t.spent_today == pytest.approx(7.0)
    t.add(Usage(0, 100_000))
    assert t.spent_today == pytest.approx(8.0)


def test_free_prices_cost_nothing():
    t = CostTracker()
    assert t.add(Usage(10_000, 10_000)) == 0
    t.check()


def test_cap_none_never_raises():
    t = CostTracker(price_in_per_mtok=1000, daily_cap_usd=None)
    t.add(Usage(10_000_000, 0))
    t.check()


def test_check_raises_at_cap():
    t = CostTracker(price_in_per_mtok=1.0, daily_cap_usd=1.0)
    t.add(Usage(900_000, 0))
    t.check()  # 0.90 < 1.00
    t.add(Usage(100_000, 0))
    with pytest.raises(SpendCapExceeded):
        t.check()
    assert issubclass(SpendCapExceeded, RuntimeError)


def test_estimate():
    t = CostTracker(price_in_per_mtok=1.0, price_out_per_mtok=2.0)
    assert t.estimate(100, 2000, 500) == pytest.approx(100 * (2000 * 1 + 500 * 2) / 1e6)
    assert CostTracker().estimate(100, 2000, 500) == 0


def test_persists_across_instances(tmp_path):
    store = tmp_path / "sub" / "spend.json"
    a = CostTracker(price_in_per_mtok=1.0, store_path=store)
    a.add(Usage(500_000, 0))
    b = CostTracker(price_in_per_mtok=1.0, daily_cap_usd=0.5, store_path=store)
    assert b.spent_today == pytest.approx(0.5)
    with pytest.raises(SpendCapExceeded):
        b.check()
    b.add(Usage(500_000, 0))
    assert a.add(Usage(0, 0)) == 0
    assert a.spent_today == pytest.approx(1.0)  # sees the other tracker's spend


def test_day_rollover(tmp_path):
    day = {"d": date(2026, 1, 1)}
    t = CostTracker(price_in_per_mtok=1.0, daily_cap_usd=1.0, store_path=tmp_path / "s.json",
                    today=lambda: day["d"])
    t.add(Usage(1_000_000, 0))
    with pytest.raises(SpendCapExceeded):
        t.check()
    day["d"] += timedelta(days=1)
    assert t.spent_today == 0
    t.check()
    t.add(Usage(250_000, 0))
    day["d"] -= timedelta(days=1)
    assert t.spent_today == pytest.approx(1.0)  # earlier day kept


def test_corrupt_store_is_ignored(tmp_path):
    store = tmp_path / "s.json"
    store.write_text("not json")
    t = CostTracker(price_in_per_mtok=1.0, store_path=store)
    assert t.spent_today == 0
    t.add(Usage(1_000_000, 0))
    assert CostTracker(store_path=store).spent_today == pytest.approx(1.0)
