"""Fit a whole surface: one SVI slice per expiry + cross-expiry diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .chain import ChainSlice
from .svi import SliceFit, calendar_violations, fit_svi_slice


@dataclass(frozen=True)
class FittedSurface:
    ticker: str
    asof: str                       # ISO date of the snapshot
    slices: list[ChainSlice]
    fits: list[SliceFit]
    calendar_violations: int

    def summary(self) -> str:
        lines = [f"{self.ticker} surface @ {self.asof} — {len(self.fits)} expiries"]
        for s, f in zip(self.slices, self.fits):
            flag = "ok " if f.butterfly_arbitrage_free else "ARB"
            lines.append(
                f"  {s.expiry}  T={s.t_years:.3f}y  F={s.forward:8.2f}  "
                f"quotes={f.n_quotes:3d}  rmse={f.rmse:.5f}  g_min={f.g_min:+.4f} [{flag}]"
            )
        lines.append(f"  calendar violations on grid: {self.calendar_violations}")
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "ticker": self.ticker,
            "asof": self.asof,
            "calendar_violations": self.calendar_violations,
            "slices": [
                {
                    "expiry": s.expiry,
                    "t_years": s.t_years,
                    "forward": s.forward,
                    "n_quotes": f.n_quotes,
                    "rmse": f.rmse,
                    "g_min": f.g_min,
                    "butterfly_arbitrage_free": f.butterfly_arbitrage_free,
                    "params": asdict(f.params),
                }
                for s, f in zip(self.slices, self.fits)
            ],
        }
        return json.dumps(payload, indent=2)


def fit_surface(ticker: str, asof: str, slices: list[ChainSlice]) -> FittedSurface:
    fits = [fit_svi_slice(s.k, s.w) for s in slices]
    cal = calendar_violations([(s.t_years, f.params) for s, f in zip(slices, fits)])
    return FittedSurface(
        ticker=ticker, asof=asof, slices=slices, fits=fits, calendar_violations=cal
    )
