"""Plotly figures for the rotation map and benchmark context."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go


ASSET_COLORS = (
    "#246BFE",
    "#F05B5B",
    "#00A878",
    "#E69500",
    "#7950F2",
    "#0E93A7",
    "#D9488B",
    "#6A994E",
    "#D97706",
    "#64748B",
    "#7C3AED",
    "#0284C7",
)

QUADRANTS: Mapping[str, dict[str, str]] = {
    "Improving": {"fill": "#EAF0FF", "label": "#5578D8"},
    "Leading": {"fill": "#E9F5EF", "label": "#27835E"},
    "Weakening": {"fill": "#FFF6DF", "label": "#C28718"},
    "Lagging": {"fill": "#FCECEC", "label": "#C95F66"},
}

FONT_FAMILY = "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"


def select_rotation_tails(
    rotation: pd.DataFrame,
    visible_start: pd.Timestamp,
    tail_length: int,
    as_of: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Select the last N visible observations for every asset."""

    if tail_length < 1:
        raise ValueError("Tail length must be positive.")

    selected: dict[str, pd.DataFrame] = {}
    cutoff = as_of if as_of is not None else rotation["date"].max()
    for ticker, group in rotation.groupby("ticker", sort=True):
        visible = group.loc[
            (group["date"] >= visible_start) & (group["date"] <= cutoff)
        ].sort_values("date").copy()
        if visible.empty:
            continue
        baseline = visible["relative"].iloc[0]
        visible["relative_performance"] = 100 * (visible["relative"] / baseline - 1)
        selected[ticker] = visible.tail(tail_length).reset_index(drop=True)
    return selected


def _axis_extent(tails: Mapping[str, pd.DataFrame]) -> float:
    deviations: list[float] = []
    for frame in tails.values():
        deviations.extend(np.abs(frame["rs_ratio"] - 100).tolist())
        deviations.extend(np.abs(frame["rs_momentum"] - 100).tolist())
    max_deviation = max(deviations, default=0)
    padded = max_deviation * 1.32
    return max(1.5, math.ceil(padded * 4) / 4)


def period_axis_extent(
    rotation: pd.DataFrame,
    visible_start: pd.Timestamp,
) -> float:
    """Return one symmetric extent covering the complete visible period."""

    visible = rotation.loc[rotation["date"] >= visible_start]
    if visible.empty:
        return 2.5
    frames = {
        ticker: group
        for ticker, group in visible.groupby("ticker", sort=True)
    }
    return _axis_extent(frames)


def _spread_label_group(
    points: list[tuple[str, float, float]],
    low: float,
    high: float,
    extent: float,
    direction: int,
) -> dict[str, tuple[float, float, str]]:
    if not points:
        return {}

    lower = low + extent * 0.12
    upper = high - extent * 0.12
    gap = min(extent * 0.09, (upper - lower) / max(len(points) - 1, 1))
    ordered = sorted(points, key=lambda point: (point[2], point[0]))
    positions: list[float] = []

    for _, _, desired_y in ordered:
        candidate = min(max(desired_y, lower), upper)
        if positions:
            candidate = max(candidate, positions[-1] + gap)
        positions.append(candidate)

    if positions[-1] > upper:
        shift = positions[-1] - upper
        positions = [position - shift for position in positions]
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - gap)
    if positions[0] < lower:
        shift = lower - positions[0]
        positions = [position + shift for position in positions]

    labels: dict[str, tuple[float, float, str]] = {}
    for (ticker, point_x, _), label_y in zip(ordered, positions):
        label_x = point_x + direction * extent * 0.045
        label_x = min(max(label_x, low + extent * 0.08), high - extent * 0.08)
        labels[ticker] = (
            label_x,
            label_y,
            "left" if direction > 0 else "right",
        )
    return labels


def _endpoint_label_layout(
    tails: Mapping[str, pd.DataFrame],
    low: float,
    high: float,
    extent: float,
    focus_ticker: str | None,
) -> dict[str, tuple[float, float, str]]:
    if focus_ticker:
        if focus_ticker not in tails:
            return {}
        latest = tails[focus_ticker].iloc[-1]
        x = float(latest["rs_ratio"])
        y = float(latest["rs_momentum"])
        direction = 1 if x >= 100 else -1
        return _spread_label_group(
            [(focus_ticker, x, y)],
            low,
            high,
            extent,
            direction,
        )

    left: list[tuple[str, float, float]] = []
    right: list[tuple[str, float, float]] = []
    for ticker, trail in tails.items():
        latest = trail.iloc[-1]
        point = (ticker, float(latest["rs_ratio"]), float(latest["rs_momentum"]))
        (right if point[1] >= 100 else left).append(point)

    labels = _spread_label_group(left, low, high, extent, -1)
    labels.update(_spread_label_group(right, low, high, extent, 1))
    return labels


def build_rrg_figure(
    rotation: pd.DataFrame,
    benchmark: str,
    visible_start: pd.Timestamp,
    tail_length: int,
    *,
    as_of: pd.Timestamp | None = None,
    focus_ticker: str | None = None,
    fixed_extent: float | None = None,
) -> tuple[go.Figure, dict[str, pd.DataFrame]]:
    """Build the quadrant chart and return the displayed asset tails."""

    tails = select_rotation_tails(rotation, visible_start, tail_length, as_of)
    extent = fixed_extent if fixed_extent is not None else _axis_extent(tails)
    low, high = 100 - extent, 100 + extent
    labels = _endpoint_label_layout(tails, low, high, extent, focus_ticker)

    figure = go.Figure()
    regions = (
        ("Improving", low, 100, 100, high),
        ("Leading", 100, high, 100, high),
        ("Weakening", 100, high, low, 100),
        ("Lagging", low, 100, low, 100),
    )
    for name, x0, x1, y0, y1 in regions:
        figure.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line_width=0,
            fillcolor=QUADRANTS[name]["fill"],
            layer="below",
        )

    label_positions = {
        "Improving": (0.015, 0.985, "left", "top"),
        "Leading": (0.985, 0.985, "right", "top"),
        "Weakening": (0.985, 0.015, "right", "bottom"),
        "Lagging": (0.015, 0.015, "left", "bottom"),
    }
    for name, (x, y, xanchor, yanchor) in label_positions.items():
        figure.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=f"<b>{name.upper()}</b>",
            showarrow=False,
            xanchor=xanchor,
            yanchor=yanchor,
            font={"family": FONT_FAMILY, "size": 12, "color": QUADRANTS[name]["label"]},
            bgcolor="rgba(255,255,255,0.68)",
            borderpad=3,
        )

    for index, (ticker, trail) in enumerate(tails.items()):
        color = ASSET_COLORS[index % len(ASSET_COLORS)]
        focus_active = focus_ticker is not None
        is_focused = not focus_active or ticker == focus_ticker
        marker_sizes = np.linspace(4, 9, len(trail)).tolist()
        marker_sizes[-1] = 13
        marker_opacity = np.linspace(0.22, 0.9, len(trail)).tolist()
        marker_opacity[-1] = 1
        marker_lines = [0] * len(trail)
        marker_lines[-1] = 2
        hover_data = np.column_stack(
            [
                trail["date"].dt.strftime("%d %b %Y"),
                trail["quadrant"],
                trail["relative_performance"],
            ]
        )

        figure.add_trace(
            go.Scatter(
                x=trail["rs_ratio"],
                y=trail["rs_momentum"],
                mode="lines+markers",
                name=ticker,
                legendgroup=ticker,
                opacity=1 if is_focused else 0.13,
                line={
                    "color": color,
                    "width": 3 if ticker == focus_ticker else 2.25,
                    "shape": "linear",
                },
                marker={
                    "color": color,
                    "size": marker_sizes,
                    "opacity": marker_opacity,
                    "line": {"color": "#FFFFFF", "width": marker_lines},
                },
                customdata=hover_data,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{customdata[0]}<br>"
                    "RS-Ratio&nbsp;&nbsp;%{x:.2f}<br>"
                    "RS-Momentum&nbsp;&nbsp;%{y:.2f}<br>"
                    "%{customdata[1]}<br>"
                    "Relative performance&nbsp;&nbsp;%{customdata[2]:+.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        latest = trail.iloc[-1]
        if len(trail) > 1:
            previous = trail.iloc[-2]
            figure.add_annotation(
                x=latest["rs_ratio"],
                y=latest["rs_momentum"],
                ax=previous["rs_ratio"],
                ay=previous["rs_momentum"],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2.6,
                arrowcolor=color,
                opacity=1 if is_focused else 0.12,
            )

        if ticker in labels:
            label_x, label_y, anchor = labels[ticker]
            figure.add_annotation(
                x=latest["rs_ratio"],
                y=latest["rs_momentum"],
                ax=label_x,
                ay=label_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text=f"<b>{ticker}</b>",
                showarrow=True,
                arrowhead=0,
                arrowsize=0.7,
                arrowwidth=0.8,
                arrowcolor=color,
                xanchor=anchor,
                font={"family": FONT_FAMILY, "size": 13, "color": color},
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(213,220,231,0.9)",
                borderwidth=0.6,
                borderpad=3,
            )

    figure.add_trace(
        go.Scatter(
            x=[100],
            y=[100],
            mode="markers",
            name=f"{benchmark} · benchmark",
            marker={
                "symbol": "diamond",
                "size": 16,
                "color": "#172033",
                "line": {"color": "#FFFFFF", "width": 2},
            },
            hovertemplate=(
                f"<b>{benchmark}</b><br>Benchmark center<br>"
                "RS-Ratio&nbsp;&nbsp;100.00<br>RS-Momentum&nbsp;&nbsp;100.00"
                "<extra></extra>"
            ),
        )
    )
    figure.add_annotation(
        x=100,
        y=100,
        text=f"<b>{benchmark}</b>",
        showarrow=False,
        xshift=14,
        yshift=-16,
        xanchor="left",
        font={"family": FONT_FAMILY, "size": 12, "color": "#172033"},
    )

    figure.add_hline(y=100, line_width=1.2, line_color="#68758A")
    figure.add_vline(x=100, line_width=1.2, line_color="#68758A")
    figure.update_layout(
        height=690,
        margin={"l": 64, "r": 24, "t": 20, "b": 92},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": FONT_FAMILY, "color": "#273247"},
        hovermode="closest",
        hoverlabel={
            "bgcolor": "#172033",
            "bordercolor": "#172033",
            "font": {"family": FONT_FAMILY, "size": 12, "color": "#FFFFFF"},
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11},
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        modebar={
            "bgcolor": "rgba(255,255,255,0.7)",
            "color": "#718096",
            "activecolor": "#246BFE",
        },
    )
    figure.update_xaxes(
        title="Normalized RS-Ratio (proxy)",
        range=[low, high],
        fixedrange=False,
        showgrid=True,
        gridcolor="#DDE3EC",
        zeroline=False,
        tickformat=".1f",
        linecolor="#AAB4C3",
        linewidth=1,
        mirror=True,
    )
    figure.update_yaxes(
        title="Normalized RS-Momentum (proxy)",
        range=[low, high],
        fixedrange=False,
        showgrid=True,
        gridcolor="#DDE3EC",
        zeroline=False,
        tickformat=".1f",
        linecolor="#AAB4C3",
        linewidth=1,
        mirror=True,
        scaleanchor="x",
        scaleratio=1,
    )
    return figure, tails


def build_benchmark_figure(
    benchmark_prices: pd.Series,
    benchmark: str,
    visible_start: pd.Timestamp,
    tail_start: pd.Timestamp | None,
) -> go.Figure:
    """Build the benchmark price context panel."""

    visible = benchmark_prices.loc[benchmark_prices.index >= visible_start].dropna()
    figure = go.Figure()
    if visible.empty:
        return figure

    figure.add_trace(
        go.Scatter(
            x=visible.index,
            y=visible,
            mode="lines",
            name=benchmark,
            line={"color": "#246BFE", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(36,107,254,0.08)",
            hovertemplate=(
                f"<b>{benchmark}</b><br>%{{x|%d %b %Y}}<br>"
                "Adjusted close&nbsp;&nbsp;%{y:,.2f}<extra></extra>"
            ),
        )
    )

    if tail_start is not None:
        figure.add_vrect(
            x0=tail_start,
            x1=visible.index.max(),
            fillcolor="#246BFE",
            opacity=0.08,
            line_width=0,
            layer="below",
            annotation_text="visible trail",
            annotation_position="top left",
            annotation_font={"size": 10, "color": "#5578D8"},
        )

    last_date = visible.index[-1]
    last_value = visible.iloc[-1]
    figure.add_vline(x=last_date, line_width=1, line_color="#246BFE", opacity=0.5)
    figure.add_annotation(
        x=last_date,
        y=last_value,
        text=f"<b>{last_value:,.2f}</b>",
        showarrow=False,
        xshift=-8,
        yshift=14,
        xanchor="right",
        font={"family": FONT_FAMILY, "size": 11, "color": "#246BFE"},
        bgcolor="rgba(255,255,255,0.85)",
    )

    floor = visible.min()
    ceiling = visible.max()
    padding = max((ceiling - floor) * 0.12, abs(ceiling) * 0.01)
    figure.update_layout(
        height=315,
        margin={"l": 10, "r": 12, "t": 16, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": "#273247"},
        showlegend=False,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#172033",
            "bordercolor": "#172033",
            "font": {"color": "#FFFFFF"},
        },
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor="#D5DCE7",
        tickfont={"size": 10, "color": "#6B768A"},
    )
    figure.update_yaxes(
        side="right",
        range=[floor - padding, ceiling + padding],
        showgrid=True,
        gridcolor="#E4E8EF",
        tickfont={"size": 10, "color": "#6B768A"},
        tickformat=",.2f",
        zeroline=False,
    )
    return figure


def build_inspector_figure(
    prices: pd.DataFrame,
    benchmark: str,
    visible_start: pd.Timestamp,
    as_of: pd.Timestamp,
    *,
    focus_ticker: str | None = None,
    tail_start: pd.Timestamp | None = None,
) -> go.Figure:
    """Build benchmark context or a normalized focused-asset comparison."""

    benchmark = benchmark.upper()
    if focus_ticker is None:
        benchmark_series = prices[benchmark].loc[prices.index <= as_of]
        return build_benchmark_figure(
            benchmark_series,
            benchmark,
            visible_start,
            tail_start,
        )

    focus_ticker = focus_ticker.upper()
    aligned = prices[[focus_ticker, benchmark]].loc[
        (prices.index >= visible_start) & (prices.index <= as_of)
    ].dropna()
    figure = go.Figure()
    if aligned.empty:
        return figure

    normalized = 100 * aligned / aligned.iloc[0]
    figure.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[focus_ticker],
            mode="lines",
            name=focus_ticker,
            line={"color": "#246BFE", "width": 2.8},
            hovertemplate=(
                f"<b>{focus_ticker}</b><br>%{{x|%d %b %Y}}<br>"
                "Normalized&nbsp;&nbsp;%{y:.2f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[benchmark],
            mode="lines",
            name=benchmark,
            line={"color": "#8A95A8", "width": 1.8},
            hovertemplate=(
                f"<b>{benchmark}</b><br>%{{x|%d %b %Y}}<br>"
                "Normalized&nbsp;&nbsp;%{y:.2f}<extra></extra>"
            ),
        )
    )

    if tail_start is not None:
        figure.add_vrect(
            x0=tail_start,
            x1=normalized.index.max(),
            fillcolor="#246BFE",
            opacity=0.06,
            line_width=0,
            layer="below",
        )

    floor = float(normalized.min().min())
    ceiling = float(normalized.max().max())
    padding = max((ceiling - floor) * 0.12, 1.0)
    figure.add_hline(y=100, line_width=1, line_dash="dot", line_color="#AAB4C3")
    figure.update_layout(
        height=315,
        margin={"l": 10, "r": 12, "t": 16, "b": 50},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": "#273247"},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
        },
        hoverlabel={
            "bgcolor": "#172033",
            "bordercolor": "#172033",
            "font": {"color": "#FFFFFF"},
        },
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor="#D5DCE7",
        tickfont={"size": 10, "color": "#6B768A"},
    )
    figure.update_yaxes(
        side="right",
        title="Rebased to 100",
        range=[floor - padding, ceiling + padding],
        showgrid=True,
        gridcolor="#E4E8EF",
        tickfont={"size": 10, "color": "#6B768A"},
        tickformat=".1f",
        zeroline=False,
    )
    return figure
