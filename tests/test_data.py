import pandas as pd

from rrg.data import prepare_bloomberg_prices


def test_prepare_bloomberg_prices_flattens_wide_bdh_response():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    response = pd.DataFrame(
        {
            ("SPY US EQUITY", "PX_LAST"): [470.0, 472.0, 471.5],
            ("XLK US Equity", "PX_LAST"): [190.0, 192.0, 193.0],
        },
        index=dates,
    )

    prices = prepare_bloomberg_prices(
        response,
        ["SPY US Equity", "XLK US EQUITY"],
    )

    assert prices.columns.tolist() == ["SPY US Equity", "XLK US Equity"]
    assert prices.loc[dates[-1], "XLK US Equity"] == 193.0


def test_prepare_bloomberg_prices_pivots_long_bdh_response():
    response = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "ticker": [
                "SPY US EQUITY",
                "XLK US Equity",
                "SPY US EQUITY",
                "XLK US Equity",
            ],
            "field": ["PX_LAST"] * 4,
            "value": [470.0, 190.0, 472.0, 192.0],
        }
    )

    prices = prepare_bloomberg_prices(
        response,
        ["SPY US Equity", "XLK US Equity"],
    )

    assert prices.columns.tolist() == ["SPY US Equity", "XLK US Equity"]
    assert prices.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert prices.iloc[-1].tolist() == [472.0, 192.0]
