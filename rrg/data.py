"""Yahoo Finance data access."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import yfinance as yf

from .core import normalize_price_columns, normalize_ticker


def _extract_close(download: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    if download.empty:
        raise RuntimeError("Yahoo Finance returned an empty response.")

    if isinstance(download.columns, pd.MultiIndex):
        first_level = download.columns.get_level_values(0)
        second_level = download.columns.get_level_values(1)
        if "Close" in first_level:
            close = download.xs("Close", axis=1, level=0)
        elif "Close" in second_level:
            close = download.xs("Close", axis=1, level=1)
        else:
            raise RuntimeError("The Yahoo response did not contain closing prices.")
    else:
        if "Close" not in download.columns:
            raise RuntimeError("The Yahoo response did not contain closing prices.")
        if len(tickers) != 1:
            raise RuntimeError("Yahoo returned an unexpected price table.")
        close = download[["Close"]].rename(columns={"Close": tickers[0]})

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close.columns = [normalize_ticker(str(column)) for column in close.columns]
    return close


def download_adjusted_close(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted closes from Yahoo Finance.

    ``end`` is exclusive, matching ``yfinance.download`` semantics.
    """

    requested = tuple(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers))
    if not requested:
        raise ValueError("At least one ticker is required.")

    try:
        response = yf.download(
            list(requested),
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=True,
            group_by="column",
            multi_level_index=True,
            timeout=15,
        )
    except Exception as exc:  # yfinance raises several network-specific types
        raise RuntimeError(f"Yahoo Finance request failed: {exc}") from exc

    close = _extract_close(response, requested)
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.sort_index().loc[~close.index.duplicated(keep="last")]
    close = close.dropna(axis=1, how="all").dropna(how="all")
    if close.empty:
        raise RuntimeError("Yahoo Finance returned no usable adjusted closes.")
    return close


def prepare_bloomberg_prices(
    download: pd.DataFrame,
    tickers: Sequence[str],
    *,
    field: str = "PX_LAST",
) -> pd.DataFrame:
    """Convert common Bloomberg ``bdh`` responses to a flat price frame.

    Both xbbg-style wide MultiIndex output and pdblp-style long output are
    supported. Symbols are returned in canonical Bloomberg casing such as
    ``SPY US Equity``.
    """

    if download.empty:
        raise RuntimeError("Bloomberg returned an empty response.")

    requested = tuple(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers))
    requested_set = set(requested)
    frame = download.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        selected: dict[str, pd.Series] = {}
        expected_field = field.casefold()
        for column in frame.columns:
            parts = [str(part) for part in column]
            symbol = next(
                (
                    normalize_ticker(part)
                    for part in parts
                    if normalize_ticker(part) in requested_set
                ),
                None,
            )
            includes_field = any(part.casefold() == expected_field for part in parts)
            if symbol is not None and includes_field:
                selected[symbol] = frame[column]
        if not selected:
            raise RuntimeError(
                f"The Bloomberg response did not contain {field} for the requested symbols."
            )
        close = pd.DataFrame(selected, index=frame.index)
    else:
        column_lookup = {str(column).casefold(): column for column in frame.columns}
        if {"date", "ticker"}.issubset(column_lookup):
            date_column = column_lookup["date"]
            ticker_column = column_lookup["ticker"]
            field_column = column_lookup.get("field")
            value_column = column_lookup.get("value")
            if value_column is None:
                value_column = column_lookup.get(field.casefold())
            if value_column is None:
                raise RuntimeError(
                    "The Bloomberg long response did not contain a value column."
                )
            if field_column is not None:
                frame = frame.loc[
                    frame[field_column].astype(str).str.casefold() == field.casefold()
                ].copy()
            frame[ticker_column] = frame[ticker_column].map(normalize_ticker)
            close = frame.pivot_table(
                index=date_column,
                columns=ticker_column,
                values=value_column,
                aggfunc="last",
            )
        else:
            close = normalize_price_columns(frame)

    close = normalize_price_columns(close)
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.sort_index().loc[~close.index.duplicated(keep="last")]
    close = close.loc[:, [ticker for ticker in requested if ticker in close.columns]]
    close = close.dropna(axis=1, how="all").dropna(how="all")
    if close.empty:
        raise RuntimeError("Bloomberg returned no usable prices for the requested symbols.")
    return close
