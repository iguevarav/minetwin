import pytest

from minetwin.stats import (
    bootstrap_mean_interval,
    holm_adjust,
    wilcoxon_signed_rank,
)


def test_wilcoxon_exact_result_and_rank_biserial():
    result = wilcoxon_signed_rank([1, 2, 3])
    assert result.statistic == 0
    assert result.p_value == pytest.approx(0.25)
    assert result.rank_biserial == 1
    assert result.method == "exact"


def test_wilcoxon_handles_zero_differences():
    result = wilcoxon_signed_rank([0, 0, 0])
    assert result.pairs == 0
    assert result.p_value == 1
    assert result.rank_biserial == 0


def test_holm_adjustment_preserves_order_and_monotonicity():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_interval_is_reproducible():
    first = bootstrap_mean_interval([1, 2, 3, 4], samples=500, seed=8)
    second = bootstrap_mean_interval([1, 2, 3, 4], samples=500, seed=8)
    assert first == second
    assert first[0] <= 2.5 <= first[1]
