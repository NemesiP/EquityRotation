import numpy as np
import pandas as pd
import pytest

from rrg.core import (
    advance_playback,
    build_rotation_snapshot,
    calculate_inspector_metrics,
    classify_quadrant,
    compute_rotation,
    normalize_tickers,
    playback_dates,
    resample_prices,
    summarize_snapshot,
)


def test_normalize_tickers_deduplicates_excludes_benchmark_and_caps():
    assets = (
        " spy, xlk, XLK; xlf\nxle, xli, xlv, xlu, xlp, xly, "
        "xlre, xlb, xlc, extra, extra2 "
    )
    selection = normalize_tickers(assets, "spy", max_assets=12)

    assert selection.benchmark == "SPY"
    assert selection.assets[0] == "XLK"
    assert len(selection.assets) == 12
    assert len(set(selection.assets)) == 12
    assert any("benchmark" in warning for warning in selection.warnings)
    assert any("first 12" in warning for warning in selection.warnings)


def test_normalize_tickers_requires_assets_and_benchmark():
    with pytest.raises(ValueError, match="benchmark"):
        normalize_tickers("XLK", "")
    with pytest.raises(ValueError, match="at least one"):
        normalize_tickers("SPY", "SPY")


def test_resampling_uses_last_available_close():
    dates = pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-08", "2024-01-31"])
    prices = pd.DataFrame({"SPY": [100, 102, 103, 110]}, index=dates)

    weekly = resample_prices(prices, "Weekly")
    monthly = resample_prices(prices, "Monthly")

    assert weekly.loc[pd.Timestamp("2024-01-05"), "SPY"] == 102
    assert monthly.loc[pd.Timestamp("2024-01-31"), "SPY"] == 110


def test_identical_asset_and_benchmark_stay_centered():
    dates = pd.date_range("2020-01-03", periods=160, freq="W-FRI")
    shared = 100 * np.exp(np.linspace(0, 0.7, len(dates)))
    prices = pd.DataFrame({"ASSET": shared, "SPY": shared}, index=dates)

    result = compute_rotation(prices, ["ASSET"], "SPY")

    assert not result.points.empty
    np.testing.assert_allclose(result.points["rs_ratio"], 100, atol=1e-10)
    np.testing.assert_allclose(result.points["rs_momentum"], 100, atol=1e-10)


def test_accelerating_relative_trend_is_leading():
    dates = pd.date_range("2018-01-05", periods=260, freq="W-FRI")
    time = np.arange(len(dates), dtype=float)
    benchmark = 100 * np.exp(0.001 * time)
    relative = np.exp(0.00001 * time**2)
    prices = pd.DataFrame(
        {"ASSET": benchmark * relative, "SPY": benchmark},
        index=dates,
    )

    result = compute_rotation(prices, ["ASSET"], "SPY")
    latest = result.points.iloc[-1]

    assert latest["rs_ratio"] > 100
    assert latest["rs_momentum"] > 100
    assert latest["quadrant"] == "Leading"


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (100, 100, "Leading"),
        (99.9, 100, "Improving"),
        (99.9, 99.9, "Lagging"),
        (100, 99.9, "Weakening"),
        (101, 101, "Leading"),
    ],
)
def test_quadrant_thresholds(x, y, expected):
    assert classify_quadrant(x, y) == expected


def test_missing_and_short_assets_are_skipped():
    dates = pd.date_range("2024-01-05", periods=45, freq="W-FRI")
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(100, 120, 45),
            "SHORT": [np.nan] * 20 + list(np.linspace(50, 60, 25)),
        },
        index=dates,
    )

    result = compute_rotation(prices, ["MISSING", "SHORT"], "SPY")

    assert result.points.empty
    assert "MISSING" in result.skipped
    assert "SHORT" in result.skipped


def test_rotation_snapshot_contains_changes_speed_transition_and_sorting():
    dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    rotation = pd.DataFrame(
        [
            (dates[0], "AAA", 99.0, 99.0, 1.00, "Lagging"),
            (dates[1], "AAA", 99.8, 100.2, 1.02, "Improving"),
            (dates[2], "AAA", 100.4, 101.0, 1.05, "Leading"),
            (dates[0], "BBB", 101.0, 100.5, 1.00, "Leading"),
            (dates[1], "BBB", 100.7, 99.8, 0.99, "Weakening"),
            (dates[2], "BBB", 100.3, 99.4, 0.98, "Weakening"),
        ],
        columns=[
            "date",
            "ticker",
            "rs_ratio",
            "rs_momentum",
            "relative",
            "quadrant",
        ],
    )

    snapshot = build_rotation_snapshot(
        rotation,
        dates[0],
        dates[-1],
        asset_names={"AAA": "Alpha"},
    )

    assert snapshot["ticker"].tolist() == ["AAA", "BBB"]
    alpha = snapshot.iloc[0]
    assert alpha["asset"] == "Alpha"
    assert alpha["ratio_change"] == pytest.approx(0.6)
    assert alpha["momentum_change"] == pytest.approx(0.8)
    assert alpha["movement_speed"] == pytest.approx(1.0)
    assert alpha["relative_return"] == pytest.approx(5.0)
    assert alpha["transition"] == "Improving → Leading"

    summary = summarize_snapshot(snapshot)
    assert summary.quadrant_counts["Leading"] == 1
    assert summary.strongest_momentum == "AAA"
    assert summary.fastest_mover == "AAA"
    assert summary.transitions == ("AAA",)


def test_inspector_metrics_use_relative_return_volatility_and_drawdown():
    dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
    prices = pd.DataFrame(
        {
            "AAA": [100, 110, 99, 108, 105],
            "SPY": [100, 102, 104, 106, 108],
        },
        index=dates,
    )
    rotation = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 5,
            "rs_ratio": [99, 99.5, 100, 100.5, 101],
            "rs_momentum": [99, 100, 101, 100.5, 100.2],
            "relative": prices["AAA"] / prices["SPY"],
            "quadrant": ["Lagging", "Improving", "Leading", "Leading", "Leading"],
        }
    )

    metrics = calculate_inspector_metrics(
        prices,
        rotation,
        "AAA",
        "SPY",
        dates[0],
        dates[-1],
        "Weekly",
    )

    assert metrics.quadrant == "Leading"
    assert metrics.rs_ratio == 101
    assert metrics.period_return == pytest.approx(5.0)
    assert metrics.relative_return == pytest.approx(100 * (105 / 108 - 1))
    assert metrics.max_drawdown == pytest.approx(100 * (99 / 110 - 1))
    assert metrics.annualized_volatility > 0

    benchmark_metrics = calculate_inspector_metrics(
        prices,
        rotation,
        "SPY",
        "SPY",
        dates[0],
        dates[-1],
        "Weekly",
    )
    assert benchmark_metrics.quadrant is None
    assert benchmark_metrics.relative_return == 0
    assert benchmark_metrics.period_return == pytest.approx(8.0)


def test_playback_dates_are_capped_and_advancement_stops():
    dates = pd.date_range("2023-01-02", periods=150, freq="B")
    rotation = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "rs_ratio": 100,
            "rs_momentum": 100,
            "relative": 1,
            "quadrant": "Leading",
        }
    )

    available = playback_dates(rotation, dates[0], "Daily")

    assert len(available) == 126
    assert available[-1] == dates[-1]
    assert advance_playback(0, 3) == (1, True)
    assert advance_playback(2, 3) == (2, False)
    assert advance_playback(0, 0) == (0, False)
