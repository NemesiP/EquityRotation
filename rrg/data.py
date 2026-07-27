"""Yahoo Finance data access."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import yfinance as yf


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

    close.columns = [str(column).upper() for column in close.columns]
    return close


def download_adjusted_close(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted closes from Yahoo Finance.

    ``end`` is exclusive, matching ``yfinance.download`` semantics.
    """

    requested = tuple(dict.fromkeys(ticker.upper() for ticker in tickers))
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
