"""Built-in rotation universes and human-readable asset names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniversePreset:
    """A named asset universe with its intended benchmark."""

    name: str
    assets: tuple[str, ...]
    benchmark: str


PRESET_UNIVERSES = {
    "US sectors": UniversePreset(
        "US sectors",
        ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"),
        "SPY",
    ),
    "US factors": UniversePreset(
        "US factors",
        ("MTUM", "QUAL", "VLUE", "USMV", "SIZE", "IWF", "IWD", "SPHB", "SPLV"),
        "SPY",
    ),
    "Global regions": UniversePreset(
        "Global regions",
        ("SPY", "EWC", "EWU", "EWG", "EWQ", "EWJ", "EWA", "EWH", "EWT", "INDA", "EWZ", "EEM"),
        "ACWI",
    ),
    "Cross-asset": UniversePreset(
        "Cross-asset",
        ("QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "HYG", "LQD", "GLD", "DBC", "UUP"),
        "SPY",
    ),
}


ASSET_NAMES = {
    "ACWI": "All Country World",
    "DBC": "Broad Commodities",
    "EEM": "Emerging Markets",
    "EFA": "Developed Markets ex-US",
    "EWA": "Australia",
    "EWC": "Canada",
    "EWG": "Germany",
    "EWH": "Hong Kong",
    "EWJ": "Japan",
    "EWQ": "France",
    "EWT": "Taiwan",
    "EWU": "United Kingdom",
    "EWZ": "Brazil",
    "GLD": "Gold",
    "HYG": "High Yield Bonds",
    "IEF": "7–10Y Treasuries",
    "INDA": "India",
    "IWD": "US Value",
    "IWF": "US Growth",
    "IWM": "US Small Caps",
    "LQD": "Investment Grade Bonds",
    "MTUM": "US Momentum",
    "QQQ": "Nasdaq 100",
    "QUAL": "US Quality",
    "SIZE": "US Size Factor",
    "SPHB": "US High Beta",
    "SPLV": "US Low Volatility",
    "SPY": "S&P 500",
    "TLT": "20+Y Treasuries",
    "UUP": "US Dollar",
    "USMV": "US Minimum Volatility",
    "VLUE": "US Value Factor",
    "VNQ": "US Real Estate",
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}


def asset_name(ticker: str) -> str:
    """Return a local display name, falling back to the normalized ticker."""

    normalized = ticker.upper()
    return ASSET_NAMES.get(normalized, normalized)
