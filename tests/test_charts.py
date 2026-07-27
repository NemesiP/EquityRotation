import pandas as pd

from rrg.charts import (
    build_benchmark_figure,
    build_inspector_figure,
    build_rrg_figure,
    period_axis_extent,
    select_rotation_tails,
)


def sample_rotation() -> pd.DataFrame:
    dates = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    frames = []
    for ticker, offset in [("XLK", 0.4), ("XLF", -0.5)]:
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "rs_ratio": [99.5 + offset + i * 0.15 for i in range(8)],
                    "rs_momentum": [99.4 - offset + i * 0.12 for i in range(8)],
                    "relative": [1 + offset / 100 + i * 0.002 for i in range(8)],
                    "quadrant": ["Lagging"] * 8,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_tail_selection_respects_visible_start_and_length():
    rotation = sample_rotation()
    tails = select_rotation_tails(rotation, pd.Timestamp("2024-01-12"), 4)

    assert set(tails) == {"XLF", "XLK"}
    assert all(len(frame) == 4 for frame in tails.values())
    assert all("relative_performance" in frame for frame in tails.values())

    earlier = select_rotation_tails(
        rotation,
        pd.Timestamp("2024-01-12"),
        4,
        pd.Timestamp("2024-02-02"),
    )
    assert all(frame["date"].max() <= pd.Timestamp("2024-02-02") for frame in earlier.values())


def test_rrg_figure_has_quadrants_center_and_centered_axes():
    figure, tails = build_rrg_figure(
        sample_rotation(),
        "SPY",
        pd.Timestamp("2024-01-01"),
        6,
    )

    assert len(figure.layout.shapes) == 6  # four regions plus the 100/100 crosshair
    assert any(trace.name == "SPY · benchmark" for trace in figure.data)
    assert set(tails) == {"XLF", "XLK"}
    assert sum(figure.layout.xaxis.range) == 200
    assert sum(figure.layout.yaxis.range) == 200


def test_benchmark_figure_marks_visible_tail():
    dates = pd.date_range("2024-01-05", periods=12, freq="W-FRI")
    prices = pd.Series(range(100, 112), index=dates, name="SPY")

    figure = build_benchmark_figure(
        prices,
        "SPY",
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-16"),
    )

    assert len(figure.data) == 1
    assert figure.data[0].name == "SPY"
    assert len(figure.layout.shapes) == 2  # trail window and latest-date line


def test_focused_rrg_uses_faded_markers_linear_paths_and_fixed_extent():
    rotation = sample_rotation()
    extent = period_axis_extent(rotation, pd.Timestamp("2024-01-01"))
    figure, _ = build_rrg_figure(
        rotation,
        "SPY",
        pd.Timestamp("2024-01-01"),
        6,
        as_of=pd.Timestamp("2024-02-23"),
        focus_ticker="XLK",
        fixed_extent=extent,
    )

    xlk = next(trace for trace in figure.data if trace.name == "XLK")
    xlf = next(trace for trace in figure.data if trace.name == "XLF")
    assert xlk.line.shape == "linear"
    assert xlk.line.color.endswith(",0.58)")
    assert xlf.line.color.endswith(",0.38)")
    assert xlk.opacity == 1
    assert xlf.opacity == 0.13
    assert xlk.marker.opacity[0] == 0.08
    assert xlk.marker.opacity[-2] < 0.55
    assert list(xlk.marker.opacity[:-1]) == sorted(xlk.marker.opacity[:-1])
    assert xlk.marker.opacity[-1] == 0
    assert xlk.marker.size[-1] == 0
    assert xlk.mode == "lines+markers"
    assert figure.layout.xaxis.range == (100 - extent, 100 + extent)
    assert figure.layout.yaxis.range == (100 - extent, 100 + extent)
    endpoint_labels = {
        annotation.text: annotation
        for annotation in figure.layout.annotations
        if annotation.text in {"XLK", "XLF"}
    }
    assert set(endpoint_labels) == {"XLK"}
    assert endpoint_labels["XLK"].showarrow is False
    assert endpoint_labels["XLK"].yshift == 13
    assert endpoint_labels["XLK"].font.size == 10


def test_normalized_inspector_starts_at_100_and_respects_as_of():
    dates = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    prices = pd.DataFrame(
        {
            "XLK": [100, 102, 104, 107, 109, 111, 114, 117],
            "SPY": [100, 101, 102, 103, 104, 105, 106, 107],
        },
        index=dates,
    )

    figure = build_inspector_figure(
        prices,
        "SPY",
        dates[0],
        dates[4],
        focus_ticker="XLK",
        tail_start=dates[2],
    )

    assert [trace.name for trace in figure.data] == ["XLK", "SPY"]
    assert all(trace.y[0] == 100 for trace in figure.data)
    assert all(pd.Timestamp(trace.x[-1]) == dates[4] for trace in figure.data)
    assert len(figure.layout.shapes) == 2  # visible trail and normalized 100 line


def test_auto_fit_expands_with_values_and_keeps_center_at_100():
    dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
    rotation = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "rs_ratio": [100.1, 100.2, 100.4, 102.0, 104.0],
            "rs_momentum": [100.1, 100.2, 100.3, 101.0, 103.0],
            "relative": [1.0, 1.01, 1.02, 1.04, 1.08],
            "quadrant": ["Leading"] * len(dates),
        }
    )

    early, _ = build_rrg_figure(
        rotation,
        "SPY",
        dates[0],
        5,
        as_of=dates[2],
    )
    late, _ = build_rrg_figure(
        rotation,
        "SPY",
        dates[0],
        5,
        as_of=dates[-1],
    )

    assert sum(early.layout.xaxis.range) == 200
    assert sum(late.layout.xaxis.range) == 200
    assert sum(early.layout.yaxis.range) == 200
    assert sum(late.layout.yaxis.range) == 200
    assert late.layout.xaxis.range[1] > early.layout.xaxis.range[1]
    assert late.layout.xaxis.range != late.layout.yaxis.range
    assert late.layout.xaxis.range[1] <= 104.5
    assert late.layout.yaxis.range[1] <= 103.5


def test_crowded_endpoints_use_compact_labels_above_arrowheads():
    dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    frames = []
    endpoints = {
        "AAA": (99.2, 100.02),
        "BBB": (99.4, 100.03),
        "CCC": (100.6, 100.02),
        "DDD": (100.8, 100.03),
    }
    for ticker, (end_x, end_y) in endpoints.items():
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "rs_ratio": [100, (100 + end_x) / 2, end_x],
                    "rs_momentum": [100, (100 + end_y) / 2, end_y],
                    "relative": [1.0, 1.01, 1.02],
                    "quadrant": ["Leading"] * 3,
                }
            )
        )
    rotation = pd.concat(frames, ignore_index=True)

    figure, _ = build_rrg_figure(rotation, "SPY", dates[0], 3)
    traces = {
        trace.name: trace
        for trace in figure.data
        if trace.name in endpoints
    }

    assert set(traces) == set(endpoints)
    assert all(trace.mode == "lines+markers" for trace in traces.values())
    assert all(trace.marker.size[-1] == 0 for trace in traces.values())
    labels = {
        annotation.text: annotation
        for annotation in figure.layout.annotations
        if annotation.text in endpoints
    }
    assert set(labels) == set(endpoints)
    assert all(label.showarrow is False for label in labels.values())
    assert all(label.yshift == 13 for label in labels.values())
    assert all(label.font.size == 10 for label in labels.values())


def test_bloomberg_arrow_label_uses_short_symbol():
    rotation = sample_rotation().loc[lambda frame: frame["ticker"] == "XLK"].copy()
    rotation["ticker"] = "XLK US Equity"

    figure, _ = build_rrg_figure(
        rotation,
        "SPY US Equity",
        pd.Timestamp("2024-01-01"),
        6,
    )

    labels = {
        annotation.text
        for annotation in figure.layout.annotations
        if annotation.text in {"XLK", "SPY"}
    }
    assert labels == {"XLK", "SPY"}
