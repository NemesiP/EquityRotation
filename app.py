"""Interactive Relative Rotation dashboard."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from rrg import (
    ASSET_NAMES,
    PERIOD_OFFSETS,
    PRESET_UNIVERSES,
    advance_playback,
    asset_name,
    build_inspector_figure,
    build_rotation_snapshot,
    build_rrg_figure,
    calculate_inspector_metrics,
    compute_rotation,
    download_adjusted_close,
    download_window,
    normalize_tickers,
    period_axis_extent,
    period_start,
    playback_dates,
    resample_prices,
    summarize_snapshot,
)


DEFAULT_PRESET = "US sectors"
DEFAULT_UNIVERSE = PRESET_UNIVERSES[DEFAULT_PRESET]
DEFAULT_CONTROLS = {
    "assets": ", ".join(DEFAULT_UNIVERSE.assets),
    "benchmark": DEFAULT_UNIVERSE.benchmark,
    "period": "2 years",
    "frequency": "Weekly",
}
FRESHNESS_LIMITS = {"Daily": 5, "Weekly": 12, "Monthly": 45}
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "relative-rotation-map",
        "height": 1200,
        "width": 1600,
        "scale": 2,
    },
}


st.set_page_config(
    page_title="Relative Rotation",
    page_icon="↗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _load_styles() -> None:
    css = Path("assets/styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_download(
    tickers: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    return download_adjusted_close(tickers, start, end)


@st.cache_data(show_spinner=False)
def _cached_transform(
    daily_prices: pd.DataFrame,
    frequency: str,
    assets: tuple[str, ...],
    benchmark: str,
):
    sampled = resample_prices(daily_prices, frequency)
    return sampled, compute_rotation(sampled, assets, benchmark)


def _initialize_state() -> None:
    defaults = {
        "active_controls": DEFAULT_CONTROLS.copy(),
        "preset_selector": DEFAULT_PRESET,
        "assets_input": DEFAULT_CONTROLS["assets"],
        "benchmark_input": DEFAULT_CONTROLS["benchmark"],
        "period_input": DEFAULT_CONTROLS["period"],
        "frequency_input": DEFAULT_CONTROLS["frequency"],
        "tail_length": 12,
        "axis_mode": "Auto fit",
        "focus_ticker": "All assets",
        "is_playing": False,
        "playback_date": None,
        "playback_date_widget": None,
        "playback_just_started": False,
        "playback_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_preset_draft() -> None:
    preset_name = st.session_state.preset_selector
    if preset_name == "Custom":
        return
    preset = PRESET_UNIVERSES[preset_name]
    st.session_state.assets_input = ", ".join(preset.assets)
    st.session_state.benchmark_input = preset.benchmark


def _draft_controls() -> dict[str, object]:
    return {
        "assets": st.session_state.assets_input,
        "benchmark": st.session_state.benchmark_input,
        "period": st.session_state.period_input,
        "frequency": st.session_state.frequency_input,
    }


def _controls_are_dirty() -> bool:
    return _draft_controls() != st.session_state.active_controls


def _header(
    placeholder,
    as_of: pd.Timestamp | None = None,
    *,
    stale: bool = False,
) -> None:
    if as_of is None:
        status = "Yahoo Finance · adjusted close"
        status_class = ""
    else:
        status = f"{'Stale · ' if stale else ''}Data through {as_of:%d %b %Y}"
        status_class = " rrg-asof--stale" if stale else " rrg-asof--fresh"
    placeholder.markdown(
        f"""
        <div class="rrg-title-row">
          <div>
            <div class="rrg-kicker">Market map / relative strength</div>
            <h1 class="rrg-title">Relative Rotation</h1>
            <p class="rrg-subtitle">
              Direction, momentum, and regime changes against one benchmark.
            </p>
          </div>
          <div class="rrg-asof{status_class}">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_data_controls() -> bool:
    preset_col, hint_col = st.columns([1.3, 4.7], vertical_alignment="bottom")
    with preset_col:
        st.selectbox(
            "Universe preset",
            options=[*PRESET_UNIVERSES, "Custom"],
            key="preset_selector",
            on_change=_apply_preset_draft,
        )
    with hint_col:
        if _controls_are_dirty():
            st.markdown(
                "<div class='rrg-draft-note'>Draft changed · press Update to apply</div>",
                unsafe_allow_html=True,
            )

    with st.form("rotation-controls", border=False):
        assets_col, benchmark_col, period_col, frequency_col, action_col = st.columns(
            [4.6, 1.2, 1.35, 1.35, 1.05],
            vertical_alignment="bottom",
        )
        with assets_col:
            st.text_input(
                "Assets",
                key="assets_input",
                help="Comma-separated Yahoo Finance tickers; the first 12 are displayed.",
            )
        with benchmark_col:
            st.text_input(
                "Benchmark",
                key="benchmark_input",
                help="The reference asset fixed at the 100 / 100 center.",
            )
        with period_col:
            st.selectbox(
                "Period",
                options=list(PERIOD_OFFSETS),
                key="period_input",
            )
        with frequency_col:
            st.selectbox(
                "Frequency",
                options=["Daily", "Weekly", "Monthly"],
                key="frequency_input",
            )
        with action_col:
            return st.form_submit_button("Update", width="stretch")


def _render_view_controls(assets: tuple[str, ...]) -> tuple[str | None, int, str]:
    if st.session_state.focus_ticker not in ("All assets", *assets):
        st.session_state.focus_ticker = "All assets"
    if st.session_state.axis_mode not in ("Auto fit", "Fixed period"):
        st.session_state.axis_mode = "Auto fit"

    focus_col, trail_col, axis_col = st.columns([2.2, 2.5, 1.7])
    with focus_col:
        st.selectbox(
            "Focus",
            options=["All assets", *assets],
            key="focus_ticker",
            format_func=lambda ticker: (
                "All assets"
                if ticker == "All assets"
                else f"{ticker} — {asset_name(ticker)}"
            ),
        )
    with trail_col:
        st.slider(
            "Trail length",
            min_value=4,
            max_value=26,
            key="tail_length",
            format="%d periods",
        )
    with axis_col:
        st.radio(
            "Axis scale",
            options=["Auto fit", "Fixed period"],
            key="axis_mode",
            horizontal=True,
            help="Auto fit follows the active trails; Fixed period keeps playback comparisons stable.",
        )
    focus = None if st.session_state.focus_ticker == "All assets" else st.session_state.focus_ticker
    return focus, int(st.session_state.tail_length), str(st.session_state.axis_mode)


def _render_summary(summary, snapshot: pd.DataFrame) -> None:
    strongest_value = ""
    fastest_value = ""
    if summary.strongest_momentum:
        row = snapshot.loc[snapshot["ticker"] == summary.strongest_momentum].iloc[0]
        strongest_value = f"{summary.strongest_momentum} {row['rs_momentum']:.2f}"
    if summary.fastest_mover:
        row = snapshot.loc[snapshot["ticker"] == summary.fastest_mover].iloc[0]
        fastest_value = f"{summary.fastest_mover} {row['movement_speed']:.2f}"
    transitions = ", ".join(summary.transitions[:4]) or "None"
    counts = summary.quadrant_counts
    st.markdown(
        f"""
        <div class="rrg-summary">
          <div><span>Quadrants</span><b>{counts["Leading"]} L · {counts["Improving"]} I ·
          {counts["Weakening"]} W · {counts["Lagging"]} La</b></div>
          <div><span>Strongest momentum</span><b>{escape(strongest_value)}</b></div>
          <div><span>Fastest mover</span><b>{escape(fastest_value)}</b></div>
          <div><span>New transitions</span><b>{escape(transitions)}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_metric(value: float | None, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:,.2f}{suffix}"


def _render_inspector_metrics(metrics, focus_ticker: str | None) -> None:
    if focus_ticker is None:
        cells = [
            ("Period return", _format_metric(metrics.period_return, "%")),
            ("Volatility", _format_metric(metrics.annualized_volatility, "%")),
            ("Max drawdown", _format_metric(metrics.max_drawdown, "%")),
            ("Relative return", "Benchmark"),
        ]
    else:
        cells = [
            ("Quadrant", metrics.quadrant or "—"),
            ("Relative return", _format_metric(metrics.relative_return, "%")),
            ("RS-Ratio", _format_metric(metrics.rs_ratio)),
            ("RS-Momentum", _format_metric(metrics.rs_momentum)),
            ("Volatility", _format_metric(metrics.annualized_volatility, "%")),
            ("Max drawdown", _format_metric(metrics.max_drawdown, "%")),
        ]
    markup = "".join(
        f"<div><span>{escape(label)}</span><b>{escape(value)}</b></div>"
        for label, value in cells
    )
    st.markdown(f"<div class='rrg-metrics'>{markup}</div>", unsafe_allow_html=True)


def _visible_history(
    rotation: pd.DataFrame,
    visible_start: pd.Timestamp,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    history = rotation.loc[
        (rotation["date"] >= visible_start) & (rotation["date"] <= as_of)
    ].copy()
    history["asset"] = history["ticker"].map(ASSET_NAMES).fillna(history["ticker"])
    history["relative_return"] = history.groupby("ticker")["relative"].transform(
        lambda values: 100 * (values / values.iloc[0] - 1)
    )
    return history[
        [
            "date",
            "ticker",
            "asset",
            "quadrant",
            "rs_ratio",
            "rs_momentum",
            "relative",
            "relative_return",
        ]
    ].sort_values(["date", "ticker"])


def _render_snapshot_table(snapshot: pd.DataFrame, focus_ticker: str | None) -> None:
    st.markdown(
        """
        <div class="rrg-section-heading rrg-section-heading--table">
          <h2>Latest rotation snapshot</h2>
          <span>click a column heading to sort</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def highlight_focus(row: pd.Series) -> list[str]:
        highlighted = focus_ticker is not None and row["ticker"] == focus_ticker
        style = "background-color: rgba(36, 107, 254, 0.09); font-weight: 650"
        return [style if highlighted else "" for _ in row]

    styled = snapshot.style.apply(highlight_focus, axis=1)
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        height="auto",
        column_order=[
            "ticker",
            "asset",
            "quadrant",
            "rs_ratio",
            "rs_momentum",
            "ratio_change",
            "momentum_change",
            "relative_return",
            "movement_speed",
            "transition",
        ],
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "asset": st.column_config.TextColumn("Asset", width="medium"),
            "quadrant": st.column_config.TextColumn("Quadrant", width="small"),
            "rs_ratio": st.column_config.NumberColumn("RS-Ratio", format="%.2f"),
            "rs_momentum": st.column_config.NumberColumn("RS-Momentum", format="%.2f"),
            "ratio_change": st.column_config.NumberColumn("Δ Ratio", format="%+.2f"),
            "momentum_change": st.column_config.NumberColumn("Δ Momentum", format="%+.2f"),
            "relative_return": st.column_config.NumberColumn("Relative return", format="%+.2f%%"),
            "movement_speed": st.column_config.NumberColumn("Speed", format="%.2f"),
            "transition": st.column_config.TextColumn("Transition", width="medium"),
        },
    )


def _render_methodology(frequency: str) -> None:
    with st.expander("How the proxy is calculated"):
        st.markdown(
            f"""
            Prices are adjusted for splits and distributions, then sampled **{frequency.lower()}**.

            `relative = asset / benchmark`

            `RS-Ratio = 100 × EMA(relative, 10) / EMA(relative, 30)`

            `RS-Momentum = 100 × RS-Ratio / EMA(RS-Ratio, 10)`

            Values above or below 100 indicate positive or negative relative direction and
            momentum. This is a transparent RRG-style proxy, not the proprietary JdK formula.
            """
        )


def main() -> None:
    _load_styles()
    _initialize_state()
    header_placeholder = st.empty()
    _header(header_placeholder)

    submitted = _render_data_controls()
    if submitted:
        draft = _draft_controls()
        try:
            normalize_tickers(str(draft["assets"]), str(draft["benchmark"]))
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state.active_controls = draft
        st.session_state.is_playing = False

    controls = st.session_state.active_controls
    try:
        selection = normalize_tickers(
            str(controls["assets"]),
            str(controls["benchmark"]),
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    for warning in selection.warnings:
        st.warning(warning)

    end = pd.Timestamp(datetime.now()).normalize()
    visible_start = period_start(end, str(controls["period"]))
    fetch_start, fetch_end = download_window(
        end,
        str(controls["period"]),
        str(controls["frequency"]),
    )

    try:
        with st.spinner("Loading adjusted prices from Yahoo Finance…"):
            daily_prices = _cached_download(
                selection.all_tickers,
                fetch_start.date().isoformat(),
                fetch_end.date().isoformat(),
            )
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.info("Check the ticker symbols or try updating again in a moment.")
        st.stop()

    if selection.benchmark not in daily_prices.columns:
        st.error(f"No price history was returned for benchmark {selection.benchmark}.")
        st.stop()

    sampled_prices, result = _cached_transform(
        daily_prices,
        str(controls["frequency"]),
        selection.assets,
        selection.benchmark,
    )
    for ticker, reason in result.skipped.items():
        st.warning(f"{ticker} was skipped — {reason}")
    if result.points.empty:
        st.error("None of the selected assets has enough aligned history to calculate.")
        st.stop()

    latest_date = pd.Timestamp(result.points["date"].max())
    stale_days = int((end - latest_date.normalize()).days)
    stale = stale_days > FRESHNESS_LIMITS[str(controls["frequency"])]
    _header(header_placeholder, latest_date, stale=stale)
    if stale:
        st.warning(
            f"The latest {controls['frequency'].lower()} observation is {stale_days} days old."
        )

    focus_ticker, tail_length, axis_mode = _render_view_controls(selection.assets)
    snapshot = build_rotation_snapshot(
        result.points,
        visible_start,
        latest_date,
        asset_names=ASSET_NAMES,
    )
    summary = summarize_snapshot(snapshot)
    _render_summary(summary, snapshot)

    dates = playback_dates(
        result.points,
        visible_start,
        str(controls["frequency"]),
    )
    if not dates:
        st.error("No calculated observations fall inside the selected period.")
        st.stop()

    signature = (
        selection.all_tickers,
        str(controls["period"]),
        str(controls["frequency"]),
        latest_date.isoformat(),
    )
    if st.session_state.playback_signature != signature:
        st.session_state.playback_signature = signature
        st.session_state.playback_date = dates[-1]
        st.session_state.playback_date_widget = dates[-1]
        st.session_state.is_playing = False
        st.session_state.playback_just_started = False
    if st.session_state.playback_date not in dates:
        st.session_state.playback_date = dates[-1]
        st.session_state.playback_date_widget = dates[-1]

    fixed_extent = (
        period_axis_extent(result.points, visible_start)
        if axis_mode == "Fixed period"
        else None
    )
    latest_metric_ticker = focus_ticker or selection.benchmark
    latest_metrics = calculate_inspector_metrics(
        sampled_prices,
        result.points,
        latest_metric_ticker,
        selection.benchmark,
        visible_start,
        latest_date,
        str(controls["frequency"]),
    )

    run_every = 0.8 if st.session_state.is_playing else None

    @st.fragment(run_every=run_every)
    def playback_workspace() -> None:
        reached_end = False
        widget_date = st.session_state.get("playback_date_widget")
        if (
            widget_date is not None
            and pd.Timestamp(widget_date) in dates
            and pd.Timestamp(widget_date) != pd.Timestamp(st.session_state.playback_date)
        ):
            st.session_state.playback_date = pd.Timestamp(widget_date)
            st.session_state.is_playing = False
            st.session_state.playback_just_started = False
        elif st.session_state.is_playing and st.session_state.playback_just_started:
            st.session_state.playback_just_started = False
        elif st.session_state.is_playing:
            current_index = dates.index(pd.Timestamp(st.session_state.playback_date))
            next_index, keep_playing = advance_playback(current_index, len(dates))
            st.session_state.playback_date = dates[next_index]
            st.session_state.playback_date_widget = dates[next_index]
            st.session_state.is_playing = keep_playing
            reached_end = not keep_playing

        control_date_col, previous_col, play_col, next_col = st.columns(
            [5.3, 0.7, 0.8, 0.7],
            vertical_alignment="bottom",
        )
        with previous_col:
            if st.button("‹", key="playback-previous", help="Previous observation"):
                index = dates.index(pd.Timestamp(st.session_state.playback_date))
                st.session_state.playback_date = dates[max(0, index - 1)]
                st.session_state.playback_date_widget = st.session_state.playback_date
                st.session_state.is_playing = False
                st.session_state.playback_just_started = False
                st.rerun(scope="app")
        with play_col:
            label = "Pause" if st.session_state.is_playing else "Play"
            if st.button(label, key="playback-toggle"):
                if st.session_state.is_playing:
                    st.session_state.is_playing = False
                    st.session_state.playback_just_started = False
                else:
                    if pd.Timestamp(st.session_state.playback_date) == dates[-1]:
                        st.session_state.playback_date = dates[0]
                        st.session_state.playback_date_widget = dates[0]
                    st.session_state.is_playing = True
                    st.session_state.playback_just_started = True
                st.rerun(scope="app")
        with next_col:
            if st.button("›", key="playback-next", help="Next observation"):
                index = dates.index(pd.Timestamp(st.session_state.playback_date))
                st.session_state.playback_date = dates[min(len(dates) - 1, index + 1)]
                st.session_state.playback_date_widget = st.session_state.playback_date
                st.session_state.is_playing = False
                st.session_state.playback_just_started = False
                st.rerun(scope="app")
        with control_date_col:
            selected_date = st.select_slider(
                "Playback date",
                options=dates,
                key="playback_date_widget",
                format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y"),
            )
            if pd.Timestamp(selected_date) != pd.Timestamp(st.session_state.playback_date):
                st.session_state.playback_date = pd.Timestamp(selected_date)
                st.session_state.is_playing = False
                st.session_state.playback_just_started = False

        as_of = pd.Timestamp(st.session_state.playback_date)
        rotation_figure, tails = build_rrg_figure(
            result.points,
            selection.benchmark,
            visible_start,
            tail_length,
            as_of=as_of,
            focus_ticker=focus_ticker,
            fixed_extent=fixed_extent,
        )
        if not tails:
            st.warning("No rotation trails are available for this playback date.")
            return
        tail_start = min(frame["date"].iloc[0] for frame in tails.values())
        inspector_figure = build_inspector_figure(
            sampled_prices,
            selection.benchmark,
            visible_start,
            as_of,
            focus_ticker=focus_ticker,
            tail_start=tail_start,
        )

        chart_col, context_col = st.columns([3.15, 1], gap="large")
        with chart_col:
            st.markdown(
                f"""
                <div class="rrg-section-heading">
                  <h2>Rotation map</h2>
                  <span>{len(tails)} assets · through {as_of:%d %b %Y}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                rotation_figure,
                width="stretch",
                config=PLOTLY_CONFIG,
                key=f"rotation-chart-{as_of.date()}",
            )
        with context_col:
            inspector_title = (
                f"{focus_ticker} vs {selection.benchmark}"
                if focus_ticker
                else f"{selection.benchmark} benchmark"
            )
            st.markdown(
                f"""
                <div class="rrg-section-heading">
                  <h2>{escape(inspector_title)}</h2>
                  <span>latest metrics</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_inspector_metrics(latest_metrics, focus_ticker)
            st.plotly_chart(
                inspector_figure,
                width="stretch",
                config={
                    **PLOTLY_CONFIG,
                    "toImageButtonOptions": {
                        **PLOTLY_CONFIG["toImageButtonOptions"],
                        "filename": f"{latest_metric_ticker.lower()}-inspector",
                        "height": 700,
                        "width": 1000,
                    },
                },
                key=f"inspector-chart-{as_of.date()}-{latest_metric_ticker}",
            )
            st.caption(
                f"Charts follow playback; metrics and the table remain fixed at {latest_date:%d %b %Y}."
            )
            _render_methodology(str(controls["frequency"]))

        if reached_end:
            st.rerun(scope="app")

    playback_workspace()
    _render_snapshot_table(snapshot, focus_ticker)

    history = _visible_history(result.points, visible_start, latest_date)
    snapshot_csv = snapshot.to_csv(index=False).encode("utf-8")
    history_csv = history.to_csv(index=False).encode("utf-8")
    export_label, snapshot_col, history_col = st.columns(
        [4.2, 1.1, 1.25],
        vertical_alignment="center",
    )
    with export_label:
        st.caption("Exports use the latest observation and currently applied universe.")
    with snapshot_col:
        st.download_button(
            "Download snapshot",
            data=snapshot_csv,
            file_name="rotation-snapshot.csv",
            mime="text/csv",
            width="stretch",
        )
    with history_col:
        st.download_button(
            "Download history",
            data=history_csv,
            file_name="rotation-history.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown(
        """
        <div class="rrg-method">
          Source: Yahoo Finance via yfinance. Intended for research and personal use.
          Prices may be delayed or incomplete; this dashboard is not investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
