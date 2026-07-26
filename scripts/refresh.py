"""Daily refresh: fetch the chain, fit raw SVI + arbitrage-free SSVI, write snapshot.

Run:  python scripts/refresh.py [TICKER]
Writes data/latest.json, charts/smiles.png, charts/surface.png,
charts/densities.png, charts/localvol.png.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from svi_lab import check_static_arbitrage, fetch_slices, fit_ssvi, fit_surface
from svi_lab.density import density_stats
from svi_lab.plots import (
    plot_densities,
    plot_local_vol,
    plot_smile_grid,
    plot_surface_3d,
)
from svi_lab.svi import calendar_violations, g_function

# Wide, fine log-moneyness grid for the Breeden-Litzenberger mass check: an
# arbitrage-free slice integrates to ~1 here, and a drift away from 1 flags a
# ragged front-month fit exactly where g_min already goes negative.
DENSITY_GRID = np.linspace(-3.0, 3.0, 3001)

ROOT = Path(__file__).resolve().parents[1]

# Yahoo intermittently serves sparse/partial chains (observed 2026-07-21: 2 thin
# expiries instead of the usual 8). Refuse to overwrite a good committed
# snapshot with a degenerate one — the cron simply skips that day.
MIN_EXPIRIES = 4


def main() -> int:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    asof = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slices = fetch_slices(ticker)
    if len(slices) < MIN_EXPIRIES:
        print(
            f"degenerate chain for {ticker}: {len(slices)} usable expiries "
            f"(< {MIN_EXPIRIES}) — keeping the previous snapshot, aborting"
        )
        return 1
    slices = sorted(slices, key=lambda s: s.t_years)

    # slice-wise raw SVI (5 params x N expiries) — best per-slice fit, no
    # cross-expiry consistency
    surface = fit_surface(ticker, asof, slices)
    print(surface.summary())

    # SSVI (3 global params + observed ATM term structure) — arbitrage-free by
    # construction; verified below through the same g-function machinery
    ssvi = fit_ssvi(slices)
    checks = check_static_arbitrage(ssvi)
    k_grid = np.linspace(-1.0, 1.0, 801)
    ssvi_g_min = [float(np.min(g_function(k_grid, ssvi.slice_params(i))))
                  for i in range(len(slices))]
    ssvi_cal = calendar_violations(
        [(float(t), ssvi.slice_params(i)) for i, t in enumerate(ssvi.t_years)]
    )

    print(f"\nSSVI global fit: rho={ssvi.rho:+.4f} eta={ssvi.eta:.4f} "
          f"gamma={ssvi.gamma:.4f}  bound eta(1+|rho|)={ssvi.bound:.4f} (<= 2)")
    print(f"Corollary 4.1 checks: {checks}")
    print(f"{'expiry':<12} {'raw rmse':>9} {'ssvi rmse':>10} {'raw g_min':>10} {'ssvi g_min':>11}")
    for s, f, sr, gm in zip(slices, surface.fits, ssvi.rmse_per_slice, ssvi_g_min):
        print(f"{s.expiry:<12} {f.rmse:>9.5f} {sr:>10.5f} {f.g_min:>+10.4f} {gm:>+11.4f}")
    raw_cal = surface.calendar_violations
    print(f"calendar violations: raw={raw_cal}  ssvi={ssvi_cal}")

    # Breeden-Litzenberger risk-neutral density (reuses the g-function): an
    # arbitrage-free slice integrates to ~1; front-month noise shows up as a
    # mass drift and/or a negative-density dip on charts/densities.png.
    raw_mass = [density_stats(DENSITY_GRID, f.params)[0] for f in surface.fits]
    ssvi_mass = [density_stats(DENSITY_GRID, ssvi.slice_params(i))[0]
                 for i in range(len(slices))]
    print(f"\n{'expiry':<12} {'raw mass':>9} {'ssvi mass':>10}  (arb-free -> ~1.000)")
    for s, rm, sm in zip(slices, raw_mass, ssvi_mass):
        print(f"{s.expiry:<12} {rm:>9.4f} {sm:>10.4f}")

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "charts").mkdir(exist_ok=True)
    payload = json.loads(surface.to_json())
    payload["ssvi"] = {
        "rho": ssvi.rho,
        "eta": ssvi.eta,
        "gamma": ssvi.gamma,
        "bound_eta_1_plus_abs_rho": ssvi.bound,
        "thetas": [float(t) for t in ssvi.thetas],
        "rmse_per_slice": [float(x) for x in ssvi.rmse_per_slice],
        "rmse_global": ssvi.rmse_global,
        "g_min_per_slice": ssvi_g_min,
        "calendar_violations": ssvi_cal,
        "corollary_4_1_checks": checks,
    }
    payload["density"] = {
        "grid": [float(DENSITY_GRID[0]), float(DENSITY_GRID[-1]), int(DENSITY_GRID.size)],
        "raw_total_mass": [float(m) for m in raw_mass],
        "ssvi_total_mass": [float(m) for m in ssvi_mass],
    }
    (ROOT / "data" / "latest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    plot_smile_grid(surface, str(ROOT / "charts" / "smiles.png"), ssvi=ssvi)
    plot_surface_3d(surface, str(ROOT / "charts" / "surface.png"))
    plot_densities(surface, str(ROOT / "charts" / "densities.png"), ssvi=ssvi)
    plot_local_vol(ssvi, str(ROOT / "charts" / "localvol.png"))
    print("wrote data/latest.json, charts/smiles.png, charts/surface.png, "
          "charts/densities.png, charts/localvol.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
