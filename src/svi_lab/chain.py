"""Fetch and clean option chains into fit-ready (k, total variance) slices.

Data source: Yahoo Finance via yfinance — free and good enough for a lab, not
for production pricing. Cleaning rules and the parity-implied forward are the
value-add here; every threshold is explicit and documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MIN_QUOTES_PER_EXPIRY = 8
MIN_DAYS, MAX_DAYS = 7, 400
IV_FLOOR, IV_CAP = 0.01, 4.0


@dataclass(frozen=True)
class ChainSlice:
    expiry: str          # YYYY-MM-DD
    t_years: float
    forward: float
    k: np.ndarray        # log-moneyness ln(K/F), OTM quotes only
    w: np.ndarray        # total implied variance iv^2 * T
    iv: np.ndarray       # implied vols as fetched


def _mid(df: pd.DataFrame) -> pd.Series:
    return (df["bid"] + df["ask"]) / 2.0


def parity_forward(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    """Forward from put-call parity at the strike where |C - P| is smallest.

    F = K + e^{rT} (C - P); we take r ~ 0 over these tenors (documented
    approximation, error < 0.5% for T < 1y at current EUR/USD short rates).
    """
    both = pd.merge(
        calls[["strike", "bid", "ask"]],
        puts[["strike", "bid", "ask"]],
        on="strike",
        suffixes=("_c", "_p"),
    )
    both = both[(both["bid_c"] > 0) & (both["bid_p"] > 0)]
    if both.empty:
        return None
    c_mid = (both["bid_c"] + both["ask_c"]) / 2.0
    p_mid = (both["bid_p"] + both["ask_p"]) / 2.0
    diff = (c_mid - p_mid).abs()
    i = diff.idxmin()
    return float(both.loc[i, "strike"] + (c_mid.loc[i] - p_mid.loc[i]))


def slice_from_frames(
    expiry: str,
    t_years: float,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
) -> ChainSlice | None:
    """Build one OTM slice: puts below the forward, calls above."""
    fwd = parity_forward(calls, puts)
    if fwd is None or fwd <= 0:
        return None

    def usable(df: pd.DataFrame) -> pd.DataFrame:
        out = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
        out = out[out["impliedVolatility"].between(IV_FLOOR, IV_CAP)]
        return out

    c, p = usable(calls), usable(puts)
    otm = pd.concat([p[p["strike"] < fwd], c[c["strike"] >= fwd]], ignore_index=True)
    otm = otm.drop_duplicates(subset="strike").sort_values("strike")
    if len(otm) < MIN_QUOTES_PER_EXPIRY:
        return None

    strikes = otm["strike"].to_numpy(dtype=float)
    iv = otm["impliedVolatility"].to_numpy(dtype=float)
    k = np.log(strikes / fwd)
    w = iv**2 * t_years
    return ChainSlice(expiry=expiry, t_years=t_years, forward=fwd, k=k, w=w, iv=iv)


def fetch_slices(ticker: str = "SPY", max_expiries: int = 8) -> list[ChainSlice]:
    """Download, clean and slice a live option chain (network)."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    now = datetime.now(timezone.utc)
    slices: list[ChainSlice] = []
    for expiry in tk.options:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days = (exp_dt - now).days
        if not (MIN_DAYS <= days <= MAX_DAYS):
            continue
        chain = tk.option_chain(expiry)
        s = slice_from_frames(expiry, days / 365.25, chain.calls, chain.puts)
        if s is not None:
            slices.append(s)
        if len(slices) >= max_expiries:
            break
    return slices
