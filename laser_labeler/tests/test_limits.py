from core.labeler.limits import BACKOFF_MINUTES, MAX_PROVIDER_WAIT_S, PADDING_S, auto_wait_seconds


def test_the_providers_own_wait_is_believed_plus_a_little():
    assert auto_wait_seconds(120, 1) == 120 + PADDING_S
    assert auto_wait_seconds(120, 9) == 120 + PADDING_S  # the hint wins over the ladder


def test_a_huge_provider_wait_is_capped_so_it_is_probed_sooner():
    assert auto_wait_seconds(10 * 24 * 3600, 1) == MAX_PROVIDER_WAIT_S + PADDING_S


def test_without_a_hint_the_wait_grows_then_levels_off():
    waits = [auto_wait_seconds(None, n) for n in range(1, 9)]
    assert waits[:len(BACKOFF_MINUTES)] == [m * 60 for m in BACKOFF_MINUTES]
    assert waits[-1] == waits[-2] == BACKOFF_MINUTES[-1] * 60
    assert waits == sorted(waits)


def test_zero_or_negative_hints_are_ignored_and_odd_counts_do_not_crash():
    assert auto_wait_seconds(0, 1) == 60 and auto_wait_seconds(-5, 2) == 120
    assert auto_wait_seconds(None, 0) == 60
