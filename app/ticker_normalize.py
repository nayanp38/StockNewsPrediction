from __future__ import annotations

import re


class TickerValidationError(ValueError):
    """Raised when ticker input cannot be normalized."""


def normalize_tickers(raw_tickers: list[str]) -> list[str]:
    """Normalize CLI or API ticker arguments to uppercase symbols."""
    normalized: list[str] = []
    for ticker_arg in raw_tickers:
        for part in ticker_arg.split(","):
            ticker = part.strip().upper()
            if not ticker:
                continue
            if " " in ticker:
                raise TickerValidationError(
                    "Ticker symbols cannot contain spaces. Use comma-separated symbols like NVDA,AMD."
                )
            if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
                raise TickerValidationError(
                    f"Invalid ticker symbol '{part.strip()}'. Use symbols like NVDA, AMD, BRK-B."
                )
            normalized.append(ticker)

    if not normalized:
        raise TickerValidationError("Provide at least one valid ticker symbol.")

    return normalized
