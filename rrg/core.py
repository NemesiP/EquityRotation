"""Core transformations for the transparent RRG-style proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


PERIOD_OFFSETS = {
    "6 months": pd.DateOffset(months=6),
    "1 year": pd.DateOffset(years=1),
    "2 years": pd.DateOffset(years=2),
    "5 years": pd.DateOffset(years=5),
    "10 years": pd.DateOffset(years=10),
}

WARMUP_OFFSETS = {
    "Daily": pd.DateOffset(months=6),
    "Weekly": pd.DateOffset(years=4),
    "Monthly": pd.DateOffset(years=12),
}

PLAYBACK_LIMITS = {
    "Daily": 126,
    "Weekly": 104,
    "Monthly": 60,
}

ANNUALIZATION_FACTORS = {
    "Daily": 252,
    "Weekly": 52,
    "Monthly": 12,
}

QUADRANT_ORDER = {
    "Leading": 0,
    "Improving": 1,
    "Weakening": 2,
    "Lagging": 3,
}

BLOOMBERG_YELLOW_KEYS = {
    "comdty": "Comdty",
    "corp": "Corp",
    "curncy": "Curncy",
    "equity": "Equity",
    "govt": "Govt",
    "index": "Index",
    "m-mkt": "M-Mkt",
    "mtge": "Mtge",
    "muni": "Muni",
    "pfd": "Pfd",
}


@dataclass(frozen=True)
class TickerSelection:
    """Validated user ticker selection."""

    assets: tuple[str, ...]
    benchmark: str
    warnings: tuple[str, ...] = ()

    @property
    def all_tickers(self) -> tuple[str, ...]:
        return (*self.assets, self.benchmark)


@dataclass(frozen=True)
class RotationResult:
    """Computed rotation points and any assets that could not be calculated."""

    points: pd.DataFrame
    skipped: dict[str, str]


@dataclass(frozen=True)
class InspectorMetrics:
    """Latest metrics for the benchmark or a focused asset."""

    ticker: str
    quadrant: str | None
    rs_ratio: float | None
    rs_momentum: float | None
    period_return: float
    relative_return: float
    annualized_volatility: float
    max_drawdown: float


@dataclass(frozen=True)
class MarketSummary:
    """Compact latest-state summary derived from the snapshot."""

    quadrant_counts: dict[str, int]
    strongest_momentum: str | None
    fastest_mover: str | None
    transitions: tuple[str, ...]


def _ticker_tokens(assets: str | Iterable[str]) -> list[str]:
    if isinstance(assets, str):
        normalized = assets.replace(";", ",").replace("\n", ",")
        return normalized.split(",")
    return list(assets)


def normalize_ticker(ticker: str) -> str:
    """Return a provider-safe canonical ticker.

    Plain exchange symbols are uppercased. Bloomberg identifiers retain the
    canonical capitalization of their terminal yellow key, for example
    ``spy us EQUITY`` becomes ``SPY US Equity``.
    """

    collapsed = " ".join(str(ticker).strip().split())
    if not collapsed:
        return ""

    head, separator, suffix = collapsed.rpartition(" ")
    yellow_key = BLOOMBERG_YELLOW_KEYS.get(suffix.casefold())
    if separator and head and yellow_key:
        return f"{head.upper()} {yellow_key}"
    return collapsed.upper()


def normalize_price_columns(prices: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize flat price columns without changing price observations.

    This makes frames returned by different providers interoperable with the
    analytics layer. Each input symbol must resolve to a distinct identifier.
    """

    if isinstance(prices.columns, pd.MultiIndex):
        raise ValueError(
            "Price columns must be flat. Select one Bloomberg field, such as "
            "PX_LAST, before calculating rotation."
        )

    normalized = prices.copy()
    normalized.columns = [normalize_ticker(str(column)) for column in prices.columns]
    duplicates = normalized.columns[normalized.columns.duplicated()].unique().tolist()
    if duplicates:
        joined = ", ".join(str(value) for value in duplicates)
        raise ValueError(f"Duplicate price columns after ticker normalization: {joined}.")
    return normalized


def normalize_tickers(
    assets: str | Iterable[str],
    benchmark: str,
    *,
    max_assets: int = 12,
) -> TickerSelection:
    """Normalize, deduplicate, validate, and cap user ticker input."""

    normalized_benchmark = normalize_ticker(benchmark)
    if not normalized_benchmark:
        raise ValueError("Enter a benchmark ticker.")

    warnings: list[str] = []
    unique_assets: list[str] = []
    seen: set[str] = set()

    for token in _ticker_tokens(assets):
        ticker = normalize_ticker(token)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        if ticker == normalized_benchmark:
            warnings.append(
                f"{ticker} was removed from the assets because it is the benchmark."
            )
            continue
        unique_assets.append(ticker)

    if len(unique_assets) > max_assets:
        warnings.append(
            f"Only the first {max_assets} assets are shown to keep the chart readable."
        )
        unique_assets = unique_assets[:max_assets]

    if not unique_assets:
        raise ValueError("Enter at least one asset that differs from the benchmark.")

    return TickerSelection(
        assets=tuple(unique_assets),
        benchmark=normalized_benchmark,
        warnings=tuple(warnings),
    )


def period_start(end: pd.Timestamp, period: str) -> pd.Timestamp:
    """Return the visible period start for a supported period label."""

    if period not in PERIOD_OFFSETS:
        raise ValueError(f"Unsupported period: {period}")
    return end - PERIOD_OFFSETS[period]


def download_window(
    end: pd.Timestamp, period: str, frequency: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return an inclusive start and exclusive end with calculation warm-up."""

    if frequency not in WARMUP_OFFSETS:
        raise ValueError(f"Unsupported frequency: {frequency}")
    visible_start = period_start(end, period)
    start = visible_start - WARMUP_OFFSETS[frequency]
    return start, end + pd.Timedelta(days=1)


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Convert daily adjusted closes to the selected observation frequency."""

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Prices must use a DatetimeIndex.")

    clean = normalize_price_columns(prices)
    clean = clean.sort_index().replace([np.inf, -np.inf], np.nan)
    clean = clean.loc[~clean.index.duplicated(keep="last")]

    if frequency == "Daily":
        sampled = clean
    elif frequency == "Weekly":
        sampled = clean.resample("W-FRI").last()
    elif frequency == "Monthly":
        sampled = clean.resample("ME").last()
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    return sampled.dropna(how="all")


def classify_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    """Classify a normalized point, assigning threshold values consistently."""

    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio < 100 <= rs_momentum:
        return "Improving"
    if rs_ratio < 100 and rs_momentum < 100:
        return "Lagging"
    return "Weakening"


def compute_rotation(
    prices: pd.DataFrame,
    assets: Iterable[str],
    benchmark: str,
    *,
    fast_span: int = 10,
    slow_span: int = 30,
    momentum_span: int = 10,
) -> RotationResult:
    """Calculate the documented normalized relative-strength proxy.

    RS-Ratio = 100 * EMA(relative, fast) / EMA(relative, slow)
    RS-Momentum = 100 * RS-Ratio / EMA(RS-Ratio, momentum)
    """

    prices = normalize_price_columns(prices)
    benchmark = normalize_ticker(benchmark)
    if benchmark not in prices.columns or prices[benchmark].dropna().empty:
        raise ValueError(f"No usable price history was returned for {benchmark}.")

    minimum_observations = slow_span + momentum_span - 1
    frames: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}

    for raw_ticker in assets:
        ticker = normalize_ticker(raw_ticker)
        if ticker not in prices.columns:
            skipped[ticker] = "No price history returned."
            continue

        aligned = prices[[ticker, benchmark]].dropna()
        aligned = aligned[(aligned[ticker] > 0) & (aligned[benchmark] > 0)]
        if len(aligned) < minimum_observations:
            skipped[ticker] = (
                f"Needs at least {minimum_observations} aligned observations."
            )
            continue

        relative = aligned[ticker] / aligned[benchmark]
        fast = relative.ewm(
            span=fast_span, adjust=False, min_periods=fast_span
        ).mean()
        slow = relative.ewm(
            span=slow_span, adjust=False, min_periods=slow_span
        ).mean()
        rs_ratio = 100 * fast / slow
        ratio_baseline = rs_ratio.ewm(
            span=momentum_span,
            adjust=False,
            min_periods=momentum_span,
        ).mean()
        rs_momentum = 100 * rs_ratio / ratio_baseline

        frame = pd.DataFrame(
            {
                "date": aligned.index,
                "ticker": ticker,
                "rs_ratio": rs_ratio,
                "rs_momentum": rs_momentum,
                "relative": relative,
            }
        ).replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["rs_ratio", "rs_momentum", "relative"])

        if frame.empty:
            skipped[ticker] = "Not enough history after indicator warm-up."
            continue

        frame["quadrant"] = [
            classify_quadrant(x, y)
            for x, y in zip(frame["rs_ratio"], frame["rs_momentum"])
        ]
        frames.append(frame.reset_index(drop=True))

    if not frames:
        columns = [
            "date",
            "ticker",
            "rs_ratio",
            "rs_momentum",
            "relative",
            "quadrant",
        ]
        return RotationResult(pd.DataFrame(columns=columns), skipped)

    return RotationResult(
        pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]),
        skipped,
    )


SNAPSHOT_COLUMNS = [
    "observation_date",
    "ticker",
    "asset",
    "quadrant",
    "previous_quadrant",
    "rs_ratio",
    "rs_momentum",
    "ratio_change",
    "momentum_change",
    "relative_return",
    "movement_speed",
    "transition",
]


def build_rotation_snapshot(
    rotation: pd.DataFrame,
    visible_start: pd.Timestamp,
    as_of: pd.Timestamp | None = None,
    *,
    asset_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build the latest cross-sectional rotation snapshot at ``as_of``."""

    if rotation.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    cutoff = as_of if as_of is not None else rotation["date"].max()
    rows: list[dict[str, object]] = []
    names = asset_names or {}

    for ticker, group in rotation.groupby("ticker", sort=True):
        visible = group.loc[
            (group["date"] >= visible_start) & (group["date"] <= cutoff)
        ].sort_values("date")
        if visible.empty:
            continue

        current = visible.iloc[-1]
        previous = visible.iloc[-2] if len(visible) > 1 else current
        ratio_change = float(current["rs_ratio"] - previous["rs_ratio"])
        momentum_change = float(current["rs_momentum"] - previous["rs_momentum"])
        baseline = float(visible["relative"].iloc[0])
        relative_return = 100 * (float(current["relative"]) / baseline - 1)
        current_quadrant = str(current["quadrant"])
        previous_quadrant = str(previous["quadrant"])

        rows.append(
            {
                "observation_date": pd.Timestamp(current["date"]),
                "ticker": ticker,
                "asset": names.get(ticker, ticker),
                "quadrant": current_quadrant,
                "previous_quadrant": previous_quadrant,
                "rs_ratio": float(current["rs_ratio"]),
                "rs_momentum": float(current["rs_momentum"]),
                "ratio_change": ratio_change,
                "momentum_change": momentum_change,
                "relative_return": relative_return,
                "movement_speed": float(np.hypot(ratio_change, momentum_change)),
                "transition": f"{previous_quadrant} → {current_quadrant}",
            }
        )

    if not rows:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    snapshot = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    snapshot["_quadrant_order"] = snapshot["quadrant"].map(QUADRANT_ORDER)
    snapshot = snapshot.sort_values(
        ["_quadrant_order", "rs_momentum", "ticker"],
        ascending=[True, False, True],
    )
    return snapshot.drop(columns="_quadrant_order").reset_index(drop=True)


def summarize_snapshot(snapshot: pd.DataFrame) -> MarketSummary:
    """Create concise, deterministic latest-market summary values."""

    counts = {
        quadrant: int((snapshot["quadrant"] == quadrant).sum())
        for quadrant in QUADRANT_ORDER
    }
    if snapshot.empty:
        return MarketSummary(counts, None, None, ())

    strongest = str(snapshot.loc[snapshot["rs_momentum"].idxmax(), "ticker"])
    fastest = str(snapshot.loc[snapshot["movement_speed"].idxmax(), "ticker"])
    changed = snapshot.loc[
        snapshot["quadrant"] != snapshot["previous_quadrant"], "ticker"
    ].astype(str)
    return MarketSummary(counts, strongest, fastest, tuple(changed))


def calculate_inspector_metrics(
    prices: pd.DataFrame,
    rotation: pd.DataFrame,
    ticker: str,
    benchmark: str,
    visible_start: pd.Timestamp,
    as_of: pd.Timestamp,
    frequency: str,
) -> InspectorMetrics:
    """Calculate latest price and rotation statistics for the inspector."""

    if frequency not in ANNUALIZATION_FACTORS:
        raise ValueError(f"Unsupported frequency: {frequency}")
    prices = normalize_price_columns(prices)
    ticker = normalize_ticker(ticker)
    benchmark = normalize_ticker(benchmark)
    if ticker not in prices.columns or benchmark not in prices.columns:
        raise ValueError("Inspector prices are unavailable for the selected symbol.")

    columns = [benchmark] if ticker == benchmark else [ticker, benchmark]
    aligned = prices[columns].loc[
        (prices.index >= visible_start) & (prices.index <= as_of)
    ].dropna()
    if aligned.empty:
        raise ValueError("No aligned inspector prices fall inside the selected period.")

    asset_prices = aligned[ticker]
    benchmark_prices = aligned[benchmark]
    period_return = 100 * (asset_prices.iloc[-1] / asset_prices.iloc[0] - 1)
    relative = asset_prices / benchmark_prices
    relative_return = 100 * (relative.iloc[-1] / relative.iloc[0] - 1)
    returns = asset_prices.pct_change().dropna()
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(ANNUALIZATION_FACTORS[frequency]) * 100)
        if len(returns) > 1
        else float("nan")
    )
    drawdown = asset_prices / asset_prices.cummax() - 1
    max_drawdown = float(drawdown.min() * 100)

    current_rotation = rotation.loc[
        (rotation["ticker"] == ticker) & (rotation["date"] <= as_of)
    ].sort_values("date")
    if current_rotation.empty:
        quadrant = None
        rs_ratio = None
        rs_momentum = None
    else:
        current = current_rotation.iloc[-1]
        quadrant = str(current["quadrant"])
        rs_ratio = float(current["rs_ratio"])
        rs_momentum = float(current["rs_momentum"])

    return InspectorMetrics(
        ticker=ticker,
        quadrant=quadrant,
        rs_ratio=rs_ratio,
        rs_momentum=rs_momentum,
        period_return=float(period_return),
        relative_return=float(relative_return),
        annualized_volatility=volatility,
        max_drawdown=max_drawdown,
    )


def playback_dates(
    rotation: pd.DataFrame,
    visible_start: pd.Timestamp,
    frequency: str,
) -> tuple[pd.Timestamp, ...]:
    """Return the frequency-capped playback dates inside the visible period."""

    if frequency not in PLAYBACK_LIMITS:
        raise ValueError(f"Unsupported frequency: {frequency}")
    dates = (
        pd.DatetimeIndex(
            rotation.loc[rotation["date"] >= visible_start, "date"].dropna().unique()
        )
        .sort_values()
        .tolist()
    )
    return tuple(pd.Timestamp(date) for date in dates[-PLAYBACK_LIMITS[frequency] :])


def advance_playback(index: int, total: int) -> tuple[int, bool]:
    """Advance one frame and report whether playback should continue."""

    if total <= 0:
        return 0, False
    bounded = max(0, min(index, total - 1))
    if bounded >= total - 1:
        return total - 1, False
    return bounded + 1, True
