import json

import pandas as pd

from rrg.notebook import NotebookDashboard


def synthetic_daily_prices() -> pd.DataFrame:
    dates = pd.date_range("2019-01-02", periods=1_700, freq="B")
    frame = pd.DataFrame(index=dates)
    frame["SPY"] = 100 + pd.Series(range(len(dates)), index=dates) * 0.04
    for index, ticker in enumerate(
        ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    ):
        frame[ticker] = frame["SPY"] * (
            1 + index * 0.002 + pd.Series(range(len(dates)), index=dates) * (index + 1) / 1_000_000
        )
    return frame


def test_notebook_controller_builds_widget_tree_without_loading():
    dashboard = NotebookDashboard(auto_load=False)

    assert dashboard.preset.value == "US sectors"
    assert dashboard.axis_mode.value == "Auto fit"
    assert dashboard.play.disabled
    assert len(dashboard.view.children) >= 10

    dashboard.preset.value = "US factors"
    assert dashboard.assets.value.startswith("MTUM, QUAL, VLUE")
    assert dashboard.benchmark.value == "SPY"


def test_notebook_accepts_custom_price_loader():
    calls = []

    def bloomberg_loader(tickers, start, end):
        calls.append((tickers, start, end))
        return synthetic_daily_prices()

    dashboard = NotebookDashboard(
        auto_load=False,
        price_loader=bloomberg_loader,
        data_source_name="Bloomberg",
    )
    dashboard.refresh()

    assert calls
    assert dashboard.data_source_name == "Bloomberg"
    assert "Bloomberg" in dashboard.view.children[-1].value


def test_notebook_refresh_with_synthetic_data(monkeypatch):
    prices = synthetic_daily_prices()
    dashboard = NotebookDashboard(auto_load=False)
    monkeypatch.setattr(dashboard, "_download_prices", lambda *args: prices)

    dashboard.refresh()

    assert dashboard.selection.benchmark == "SPY"
    assert dashboard.rotation["ticker"].nunique() == 11
    assert not dashboard.snapshot.empty
    assert dashboard.playback_values
    assert dashboard.play.max == len(dashboard.playback_values) - 1
    assert not dashboard.play.disabled
    assert "rotation-snapshot.csv" in dashboard.exports.value


def test_notebook_file_is_valid_nbformat_json():
    with open("EquityRotation.ipynb", encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    assert notebook["nbformat"] == 4
    assert any(
        "launch_notebook_dashboard" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
