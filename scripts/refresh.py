"""Daily refresh: fetch the chain, fit the surface, write snapshot + charts.

Run:  python scripts/refresh.py [TICKER]
Writes data/latest.json, charts/smiles.png, charts/surface.png.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from svi_lab import fetch_slices, fit_surface
from svi_lab.plots import plot_smile_grid, plot_surface_3d

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    asof = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slices = fetch_slices(ticker)
    if len(slices) < 2:
        print(f"not enough usable expiries for {ticker} (got {len(slices)}) — aborting")
        return 1
    surface = fit_surface(ticker, asof, slices)
    print(surface.summary())

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "charts").mkdir(exist_ok=True)
    (ROOT / "data" / "latest.json").write_text(surface.to_json(), encoding="utf-8")
    plot_smile_grid(surface, str(ROOT / "charts" / "smiles.png"))
    plot_surface_3d(surface, str(ROOT / "charts" / "surface.png"))
    print("wrote data/latest.json, charts/smiles.png, charts/surface.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
