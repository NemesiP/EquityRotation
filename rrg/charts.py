"""Plotly figures for the rotation map and benchmark context."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .core import normalize_price_columns, normalize_ticker


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


def _padded_extent(max_deviation: float) -> float:
    """Return a tight, rounded, centered extent for one axis."""

    if not np.isfinite(max_deviation) or max_deviation <= 0:
        return 0.25
    padded = max_deviation * 1.1
    magnitude = 10 ** math.floor(math.log10(padded))
    rounding_step = magnitude / 4
    rounded = math.ceil(padded / rounding_step) * rounding_step
    return max(0.25, rounded)


def _axis_extents(tails: Mapping[str, pd.DataFrame]) -> tuple[float, float]:
    ratio_deviations: list[float] = []
    momentum_deviations: list[float] = []
    for frame in tails.values():
        ratio_deviations.extend(np.abs(frame["rs_ratio"] - 100).tolist())
        momentum_deviations.extend(np.abs(frame["rs_momentum"] - 100).tolist())
    return (
        _padded_extent(max(ratio_deviations, default=0)),
        _padded_extent(max(momentum_deviations, default=0)),
    )


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
    ratio_extent, momentum_extent = _axis_extents(frames)
    return max(ratio_extent, momentum_extent)


def _pointer_label(ticker: str) -> str:
    """Keep endpoint text compact, including for Bloomberg identifiers."""

    short_ticker = ticker.split(" ", 1)[0]
    return short_ticker if len(short_ticker) <= 6 else f"{short_ticker[:5]}…"


def _rgba(hex_color: str, opacity: float) -> str:
    """Convert a palette color to an explicit Plotly RGBA value."""

    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{opacity:.2f})"


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
    if fixed_extent is None:
        ratio_extent, momentum_extent = _axis_extents(tails)
    else:
        ratio_extent = momentum_extent = fixed_extent
    x_low, x_high = 100 - ratio_extent, 100 + ratio_extent
    y_low, y_high = 100 - momentum_extent, 100 + momentum_extent

    figure = go.Figure()
    regions = (
        ("Improving", x_low, 100, 100, y_high),
        ("Leading", 100, x_high, 100, y_high),
        ("Weakening", 100, x_high, y_low, 100),
        ("Lagging", x_low, 100, y_low, 100),
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
        marker_sizes = np.linspace(4, 8, len(trail)).tolist()
        marker_sizes[-1] = 0
        marker_opacity = np.linspace(0.08, 0.55, len(trail)).tolist()
        marker_opacity[-1] = 0
        marker_lines = [0] * len(trail)
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
                    "color": _rgba(color, 0.58 if ticker == focus_ticker else 0.38),
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

        if is_focused:
            figure.add_annotation(
                x=latest["rs_ratio"],
                y=latest["rs_momentum"],
                text=_pointer_label(ticker),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                yshift=13,
                font={"family": FONT_FAMILY, "size": 10, "color": color},
                opacity=1,
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
        text=_pointer_label(benchmark),
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        yshift=12,
        font={"family": FONT_FAMILY, "size": 10, "color": "#172033"},
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
        range=[x_low, x_high],
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
        range=[y_low, y_high],
        fixedrange=False,
        showgrid=True,
        gridcolor="#DDE3EC",
        zeroline=False,
        tickformat=".1f",
        linecolor="#AAB4C3",
        linewidth=1,
        mirror=True,
        scaleanchor="x" if fixed_extent is not None else None,
        scaleratio=1 if fixed_extent is not None else None,
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

    prices = normalize_price_columns(prices)
    benchmark = normalize_ticker(benchmark)
    if focus_ticker is None:
        benchmark_series = prices[benchmark].loc[prices.index <= as_of]
        return build_benchmark_figure(
            benchmark_series,
            benchmark,
            visible_start,
            tail_start,
        )

    focus_ticker = normalize_ticker(focus_ticker)
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
