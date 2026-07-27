"""Jupyter/IPython presentation surface for the relative rotation dashboard."""

from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from time import monotonic

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display

from .charts import build_inspector_figure, build_rrg_figure, period_axis_extent
from .core import (
    PERIOD_OFFSETS,
    build_rotation_snapshot,
    calculate_inspector_metrics,
    compute_rotation,
    download_window,
    normalize_tickers,
    period_start,
    playback_dates,
    resample_prices,
    summarize_snapshot,
)
from .data import download_adjusted_close
from .presets import ASSET_NAMES, PRESET_UNIVERSES, asset_name


NOTEBOOK_CSS = """
<style>
.rrg-nb-shell { color:#172033; font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
.rrg-nb-header { border-bottom:1px solid #dce2eb; display:flex; justify-content:space-between;
  align-items:flex-end; margin-bottom:14px; padding-bottom:14px; }
.rrg-nb-kicker { color:#246bfe; font-size:11px; font-weight:750; letter-spacing:.13em;
  margin-bottom:5px; text-transform:uppercase; }
.rrg-nb-title { font-size:38px; font-weight:650; letter-spacing:-.04em; line-height:1; }
.rrg-nb-subtitle,.rrg-nb-status { color:#6b768a; font-size:13px; margin-top:8px; }
.rrg-nb-summary { border-bottom:1px solid #dce2eb; border-top:1px solid #dce2eb;
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); margin:12px 0; }
.rrg-nb-summary div { padding:10px 14px; }
.rrg-nb-summary div+div { border-left:1px solid #dce2eb; }
.rrg-nb-summary span,.rrg-nb-metrics span { color:#6b768a; display:block; font-size:10px;
  font-weight:700; letter-spacing:.05em; text-transform:uppercase; }
.rrg-nb-summary b,.rrg-nb-metrics b { display:block; font-size:13px; margin-top:3px; }
.rrg-nb-metrics { display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
  border-bottom:1px solid #dce2eb; margin-bottom:8px; }
.rrg-nb-metrics div { padding:7px 5px 8px 0; }
.rrg-nb-section { color:#172033; font-size:14px; font-weight:700; margin:14px 0 5px; }
.rrg-nb-note { color:#6b768a; font-size:11px; line-height:1.55; }
.rrg-nb-download { background:#246bfe; border-radius:6px; color:white!important;
  display:inline-block; font-size:12px; font-weight:650; margin-right:8px;
  padding:7px 11px; text-decoration:none!important; }
.widget-label { color:#58657a!important; font-size:12px!important; font-weight:650!important; }
@media(max-width:800px) {
  .rrg-nb-header { align-items:flex-start; flex-direction:column; }
  .rrg-nb-summary { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
</style>
"""


class NotebookDashboard:
    """Stateful ipywidgets dashboard backed by the shared RRG modules."""

    def __init__(self, *, auto_load: bool = True) -> None:
        self.daily_prices: pd.DataFrame | None = None
        self.sampled_prices: pd.DataFrame | None = None
        self.rotation = pd.DataFrame()
        self.snapshot = pd.DataFrame()
        self.selection = None
        self.visible_start: pd.Timestamp | None = None
        self.latest_date: pd.Timestamp | None = None
        self.playback_values: tuple[pd.Timestamp, ...] = ()
        self.fixed_extent: float | None = None
        self._cache: dict[tuple[object, ...], tuple[float, pd.DataFrame]] = {}
        self._render_suspended = False

        self._build_widgets()
        self._bind_events()
        self.view = self._build_layout()
        if auto_load:
            self.refresh()

    def _build_widgets(self) -> None:
        default = PRESET_UNIVERSES["US sectors"]
        common = {"layout": widgets.Layout(width="100%")}
        self.preset = widgets.Dropdown(
            options=[*PRESET_UNIVERSES, "Custom"],
            value="US sectors",
            description="Universe",
            **common,
        )
        self.assets = widgets.Textarea(
            value=", ".join(default.assets),
            description="Assets",
            rows=2,
            **common,
        )
        self.benchmark = widgets.Text(
            value=default.benchmark,
            description="Benchmark",
            **common,
        )
        self.period = widgets.Dropdown(
            options=list(PERIOD_OFFSETS),
            value="2 years",
            description="Period",
            **common,
        )
        self.frequency = widgets.Dropdown(
            options=["Daily", "Weekly", "Monthly"],
            value="Weekly",
            description="Frequency",
            **common,
        )
        self.update_button = widgets.Button(
            description="Update data",
            button_style="primary",
            icon="refresh",
            layout=widgets.Layout(width="150px", height="36px"),
        )
        self.focus = widgets.Dropdown(
            options=[("All assets", None)],
            value=None,
            description="Focus",
            **common,
        )
        self.tail = widgets.IntSlider(
            value=12,
            min=4,
            max=26,
            step=1,
            description="Trail",
            continuous_update=False,
            **common,
        )
        self.axis_mode = widgets.ToggleButtons(
            options=["Auto fit", "Fixed period"],
            value="Auto fit",
            description="Axis",
            button_style="",
            layout=widgets.Layout(width="100%"),
        )
        self.play = widgets.Play(
            value=0,
            min=0,
            max=0,
            step=1,
            interval=800,
            repeat=False,
            disabled=True,
            layout=widgets.Layout(width="44px"),
        )
        self.playback = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            readout=False,
            continuous_update=False,
            disabled=True,
            layout=widgets.Layout(flex="1 1 auto", min_width="260px"),
        )
        self.restart = widgets.Button(
            icon="step-backward",
            tooltip="Restart playback",
            disabled=True,
            layout=widgets.Layout(width="38px"),
        )
        self.playback_label = widgets.HTML("<span class='rrg-nb-note'>No data</span>")
        self.sort_by = widgets.Dropdown(
            options=[
                ("Quadrant + momentum", "default"),
                ("RS-Ratio", "rs_ratio"),
                ("RS-Momentum", "rs_momentum"),
                ("Relative return", "relative_return"),
                ("Speed", "movement_speed"),
            ],
            value="default",
            description="Sort table",
            layout=widgets.Layout(width="290px"),
        )
        self.sort_descending = widgets.Checkbox(
            value=True,
            description="Descending",
            indent=False,
            layout=widgets.Layout(width="120px"),
        )

        self.header = widgets.HTML()
        self.status = widgets.HTML()
        self.summary = widgets.HTML()
        self.metrics = widgets.HTML()
        self.exports = widgets.HTML()
        self.plot_output = widgets.Output(
            layout=widgets.Layout(width="100%", min_height="650px")
        )
        self.inspector_output = widgets.Output(
            layout=widgets.Layout(width="100%", min_height="330px")
        )
        self.table_output = widgets.Output(layout=widgets.Layout(width="100%"))
        self.message_output = widgets.Output(layout=widgets.Layout(width="100%"))
        self._set_header()

    def _build_layout(self) -> widgets.VBox:
        data_controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox([self.preset], layout=widgets.Layout(flex="1")),
                        widgets.VBox([self.benchmark], layout=widgets.Layout(flex="1")),
                        widgets.VBox([self.period], layout=widgets.Layout(flex="1")),
                        widgets.VBox([self.frequency], layout=widgets.Layout(flex="1")),
                        self.update_button,
                    ],
                    layout=widgets.Layout(
                        align_items="flex-end",
                        flex_flow="row wrap",
                        width="100%",
                    ),
                ),
                self.assets,
            ],
            layout=widgets.Layout(
                border="1px solid #dce2eb",
                padding="12px",
                width="100%",
            ),
        )
        view_controls = widgets.HBox(
            [
                widgets.VBox([self.focus], layout=widgets.Layout(flex="2")),
                widgets.VBox([self.tail], layout=widgets.Layout(flex="2")),
                widgets.VBox([self.axis_mode], layout=widgets.Layout(flex="1")),
            ],
            layout=widgets.Layout(
                align_items="center",
                flex_flow="row wrap",
                margin="10px 0 0",
                width="100%",
            ),
        )
        playback_controls = widgets.HBox(
            [self.restart, self.play, self.playback, self.playback_label],
            layout=widgets.Layout(
                align_items="center",
                margin="10px 0 0",
                width="100%",
            ),
        )
        main_column = widgets.VBox(
            [
                widgets.HTML("<div class='rrg-nb-section'>Rotation map</div>"),
                self.plot_output,
            ],
            layout=widgets.Layout(flex="3 1 650px", min_width="560px"),
        )
        inspector_column = widgets.VBox(
            [
                widgets.HTML("<div class='rrg-nb-section'>Inspector</div>"),
                self.metrics,
                self.inspector_output,
            ],
            layout=widgets.Layout(flex="1 1 300px", min_width="290px"),
        )
        workspace = widgets.HBox(
            [main_column, inspector_column],
            layout=widgets.Layout(
                align_items="flex-start",
                flex_flow="row wrap",
                width="100%",
            ),
        )
        table_controls = widgets.HBox(
            [self.sort_by, self.sort_descending],
            layout=widgets.Layout(
                align_items="center", margin="10px 0 4px"
            ),
        )
        return widgets.VBox(
            [
                widgets.HTML(NOTEBOOK_CSS),
                self.header,
                data_controls,
                view_controls,
                self.status,
                self.summary,
                playback_controls,
                workspace,
                widgets.HTML("<div class='rrg-nb-section'>Latest rotation snapshot</div>"),
                table_controls,
                self.table_output,
                self.exports,
                self.message_output,
                widgets.HTML(
                    "<div class='rrg-nb-note' style='border-top:1px solid #dce2eb;"
                    "margin-top:14px;padding-top:10px'>Source: Yahoo Finance via yfinance. "
                    "For research and personal use; not investment advice.</div>"
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )

    def _bind_events(self) -> None:
        self.preset.observe(self._on_preset, names="value")
        self.update_button.on_click(self._on_update)
        self.focus.observe(self._on_view_change, names="value")
        self.tail.observe(self._on_view_change, names="value")
        self.axis_mode.observe(self._on_view_change, names="value")
        self.playback.observe(self._on_playback, names="value")
        self.restart.on_click(self._on_restart)
        self.sort_by.observe(self._on_table_change, names="value")
        self.sort_descending.observe(self._on_table_change, names="value")
        self._play_link = widgets.jslink(
            (self.play, "value"),
            (self.playback, "value"),
        )

    def _on_preset(self, change: dict[str, object]) -> None:
        name = str(change["new"])
        if name == "Custom":
            return
        preset = PRESET_UNIVERSES[name]
        self.assets.value = ", ".join(preset.assets)
        self.benchmark.value = preset.benchmark
        self.status.value = (
            "<div class='rrg-nb-status'>Preset loaded as a draft · select Update data</div>"
        )

    def _on_update(self, _: widgets.Button) -> None:
        self.refresh()

    def _on_view_change(self, _: dict[str, object]) -> None:
        if not self._render_suspended and not self.rotation.empty:
            self.render_charts()
            self.render_table()

    def _on_playback(self, change: dict[str, object]) -> None:
        if self._render_suspended or not self.playback_values:
            return
        index = int(change["new"])
        self.playback_label.value = (
            f"<span class='rrg-nb-note'>{self.playback_values[index]:%d %b %Y}</span>"
        )
        self.render_charts()

    def _on_restart(self, _: widgets.Button) -> None:
        if self.playback_values:
            self.play.value = 0
            self.playback.value = 0

    def _on_table_change(self, _: dict[str, object]) -> None:
        if not self.rotation.empty:
            self.render_table()

    def _set_header(self, as_of: pd.Timestamp | None = None) -> None:
        status = (
            "Yahoo Finance · adjusted close"
            if as_of is None
            else f"Data through {as_of:%d %b %Y}"
        )
        self.header.value = f"""
        <div class="rrg-nb-shell">
          <div class="rrg-nb-header">
            <div>
              <div class="rrg-nb-kicker">Market map / notebook edition</div>
              <div class="rrg-nb-title">Relative Rotation</div>
              <div class="rrg-nb-subtitle">Direction, momentum, and regime changes
              against one benchmark.</div>
            </div>
            <div class="rrg-nb-status">{status}</div>
            </div>
        """

    def _download_prices(
        self,
        tickers: tuple[str, ...],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        key = (tickers, start, end)
        cached = self._cache.get(key)
        if cached and monotonic() - cached[0] < 900:
            return cached[1].copy()
        prices = download_adjusted_close(tickers, start, end)
        self._cache[key] = (monotonic(), prices.copy())
        return prices

    def refresh(self) -> None:
        """Apply data controls, retrieve prices, and redraw the notebook."""

        self.update_button.disabled = True
        self.update_button.description = "Loading…"
        self.status.value = "<div class='rrg-nb-status'>Loading adjusted prices…</div>"
        with self.message_output:
            clear_output(wait=True)
        try:
            selection = normalize_tickers(self.assets.value, self.benchmark.value)
            end = pd.Timestamp(datetime.now()).normalize()
            visible_start = period_start(end, self.period.value)
            fetch_start, fetch_end = download_window(
                end,
                self.period.value,
                self.frequency.value,
            )
            daily = self._download_prices(
                selection.all_tickers,
                fetch_start.date().isoformat(),
                fetch_end.date().isoformat(),
            )
            sampled = resample_prices(daily, self.frequency.value)
            result = compute_rotation(sampled, selection.assets, selection.benchmark)
            if result.points.empty:
                raise ValueError("No selected asset has enough aligned history.")

            self.selection = selection
            self.daily_prices = daily
            self.sampled_prices = sampled
            self.rotation = result.points
            self.visible_start = visible_start
            self.latest_date = pd.Timestamp(result.points["date"].max())
            self.snapshot = build_rotation_snapshot(
                result.points,
                visible_start,
                self.latest_date,
                asset_names=ASSET_NAMES,
            )
            self.fixed_extent = period_axis_extent(result.points, visible_start)
            self.playback_values = playback_dates(
                result.points,
                visible_start,
                self.frequency.value,
            )

            self._render_suspended = True
            focus_options = [("All assets", None)] + [
                (f"{ticker} — {asset_name(ticker)}", ticker)
                for ticker in selection.assets
                if ticker not in result.skipped
            ]
            self.focus.options = focus_options
            self.focus.value = None
            last_index = max(0, len(self.playback_values) - 1)
            self.play.min = 0
            self.play.max = last_index
            self.play.value = last_index
            self.play.disabled = not self.playback_values
            self.playback.min = 0
            self.playback.max = last_index
            self.playback.value = last_index
            self.playback.disabled = not self.playback_values
            self.restart.disabled = not self.playback_values
            self._render_suspended = False

            self._set_header(self.latest_date)
            warnings = [*selection.warnings]
            warnings.extend(f"{ticker}: {reason}" for ticker, reason in result.skipped.items())
            warning_text = " · ".join(warnings)
            self.status.value = (
                f"<div class='rrg-nb-status'>{escape(warning_text)}</div>"
                if warning_text
                else ""
            )
            self.render_summary()
            self.render_charts()
            self.render_table()
            self.render_exports()
        except Exception as exc:
            self._render_suspended = False
            self.status.value = (
                f"<div class='rrg-nb-status' style='color:#b42318'>"
                f"{escape(str(exc))}</div>"
            )
            with self.message_output:
                clear_output(wait=True)
                display(HTML(f"<b>Unable to update:</b> {escape(str(exc))}"))
        finally:
            self.update_button.disabled = False
            self.update_button.description = "Update data"

    @property
    def as_of(self) -> pd.Timestamp:
        if not self.playback_values:
            raise ValueError("Load data before requesting a playback date.")
        index = min(int(self.playback.value), len(self.playback_values) - 1)
        return self.playback_values[index]

    def render_summary(self) -> None:
        summary = summarize_snapshot(self.snapshot)
        counts = summary.quadrant_counts
        strongest = summary.strongest_momentum or "—"
        fastest = summary.fastest_mover or "—"
        transitions = ", ".join(summary.transitions[:4]) or "None"
        self.summary.value = f"""
        <div class="rrg-nb-summary">
          <div><span>Quadrants</span><b>{counts["Leading"]} L ·
          {counts["Improving"]} I · {counts["Weakening"]} W ·
          {counts["Lagging"]} La</b></div>
          <div><span>Strongest momentum</span><b>{escape(strongest)}</b></div>
          <div><span>Fastest mover</span><b>{escape(fastest)}</b></div>
          <div><span>New transitions</span><b>{escape(transitions)}</b></div>
        </div>
        """

    @staticmethod
    def _format_metric(value: float | None, suffix: str = "") -> str:
        if value is None or not np.isfinite(value):
            return "—"
        return f"{value:,.2f}{suffix}"

    def render_charts(self) -> None:
        if (
            self.rotation.empty
            or self.sampled_prices is None
            or self.selection is None
            or self.visible_start is None
            or self.latest_date is None
        ):
            return
        as_of = self.as_of
        focus = self.focus.value
        fixed_extent = self.fixed_extent if self.axis_mode.value == "Fixed period" else None
        rrg_figure, tails = build_rrg_figure(
            self.rotation,
            self.selection.benchmark,
            self.visible_start,
            self.tail.value,
            as_of=as_of,
            focus_ticker=focus,
            fixed_extent=fixed_extent,
        )
        rrg_figure.update_layout(height=620)
        tail_start = min(frame["date"].iloc[0] for frame in tails.values())
        inspector = build_inspector_figure(
            self.sampled_prices,
            self.selection.benchmark,
            self.visible_start,
            as_of,
            focus_ticker=focus,
            tail_start=tail_start,
        )
        inspector.update_layout(height=320)
        metric_ticker = focus or self.selection.benchmark
        metrics = calculate_inspector_metrics(
            self.sampled_prices,
            self.rotation,
            metric_ticker,
            self.selection.benchmark,
            self.visible_start,
            self.latest_date,
            self.frequency.value,
        )
        if focus is None:
            cells = [
                ("Period return", self._format_metric(metrics.period_return, "%")),
                ("Volatility", self._format_metric(metrics.annualized_volatility, "%")),
                ("Max drawdown", self._format_metric(metrics.max_drawdown, "%")),
                ("Relative return", "Benchmark"),
            ]
        else:
            cells = [
                ("Quadrant", metrics.quadrant or "—"),
                ("Relative return", self._format_metric(metrics.relative_return, "%")),
                ("RS-Ratio", self._format_metric(metrics.rs_ratio)),
                ("RS-Momentum", self._format_metric(metrics.rs_momentum)),
                ("Volatility", self._format_metric(metrics.annualized_volatility, "%")),
                ("Max drawdown", self._format_metric(metrics.max_drawdown, "%")),
            ]
        self.metrics.value = "<div class='rrg-nb-metrics'>" + "".join(
            f"<div><span>{escape(label)}</span><b>{escape(value)}</b></div>"
            for label, value in cells
        ) + "</div>"
        self.playback_label.value = (
            f"<span class='rrg-nb-note'>{as_of:%d %b %Y}</span>"
        )
        with self.plot_output:
            clear_output(wait=True)
            display(rrg_figure)
        with self.inspector_output:
            clear_output(wait=True)
            display(inspector)

    def _sorted_snapshot(self) -> pd.DataFrame:
        if self.sort_by.value == "default":
            return self.snapshot.copy()
        return self.snapshot.sort_values(
            self.sort_by.value,
            ascending=not self.sort_descending.value,
        ).reset_index(drop=True)

    def render_table(self) -> None:
        if self.snapshot.empty:
            return
        table = self._sorted_snapshot()
        focus = self.focus.value

        def highlight(row: pd.Series) -> list[str]:
            style = (
                "background-color:rgba(36,107,254,.09);font-weight:650"
                if focus and row["ticker"] == focus
                else ""
            )
            return [style] * len(row)

        visible_columns = [
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
        ]
        styler = (
            table[visible_columns]
            .style.apply(highlight, axis=1)
            .format(
                {
                    "rs_ratio": "{:.2f}",
                    "rs_momentum": "{:.2f}",
                    "ratio_change": "{:+.2f}",
                    "momentum_change": "{:+.2f}",
                    "relative_return": "{:+.2f}%",
                    "movement_speed": "{:.2f}",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background", "#f4f6fa"),
                            ("color", "#58657a"),
                            ("font-size", "11px"),
                            ("text-align", "left"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border-bottom", "1px solid #e4e8ef"),
                            ("font-size", "12px"),
                            ("padding", "7px"),
                        ],
                    },
                ]
            )
            .hide(axis="index")
        )
        with self.table_output:
            clear_output(wait=True)
            display(styler)

    @staticmethod
    def _download_link(data: bytes, filename: str, label: str) -> str:
        encoded = base64.b64encode(data).decode("ascii")
        return (
            f"<a class='rrg-nb-download' download='{escape(filename)}' "
            f"href='data:text/csv;base64,{encoded}'>{escape(label)}</a>"
        )

    def render_exports(self) -> None:
        if self.snapshot.empty or self.latest_date is None or self.visible_start is None:
            self.exports.value = ""
            return
        history = self.rotation.loc[
            (self.rotation["date"] >= self.visible_start)
            & (self.rotation["date"] <= self.latest_date)
        ].copy()
        history["asset"] = history["ticker"].map(ASSET_NAMES).fillna(history["ticker"])
        history["relative_return"] = history.groupby("ticker")["relative"].transform(
            lambda values: 100 * (values / values.iloc[0] - 1)
        )
        snapshot_link = self._download_link(
            self.snapshot.to_csv(index=False).encode("utf-8"),
            "rotation-snapshot.csv",
            "Download snapshot CSV",
        )
        history_link = self._download_link(
            history.to_csv(index=False).encode("utf-8"),
            "rotation-history.csv",
            "Download history CSV",
        )
        self.exports.value = (
            "<div style='margin-top:10px'>"
            f"{snapshot_link}{history_link}</div>"
        )

    def display(self) -> "NotebookDashboard":
        """Display the assembled widget tree and return this controller."""

        display(self.view)
        return self


def launch_notebook_dashboard(*, auto_load: bool = True) -> NotebookDashboard:
    """Create and display the Jupyter dashboard."""

    dashboard = NotebookDashboard(auto_load=auto_load)
    return dashboard.display()
