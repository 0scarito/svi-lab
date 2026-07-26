"""Charts: smile grid (market vs fit), the 3D fitted surface, and RND panel."""

from __future__ import annotations

import numpy as np

from .density import risk_neutral_density
from .surface import FittedSurface
from .svi import svi_total_variance


def plot_smile_grid(surface: FittedSurface, save_path: str, ssvi=None) -> None:
    """Market quotes vs raw-SVI fits, with an optional arbitrage-free SSVI overlay.

    ``ssvi`` is an ``SSVIFit`` whose slices are index-aligned with
    ``surface.slices`` (both sorted by maturity).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(surface.slices)
    cols = 3
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for i, (ax, s, f) in enumerate(zip(axes.flat, surface.slices, surface.fits)):
        iv_mkt = np.sqrt(s.w / s.t_years)
        grid = np.linspace(s.k.min(), s.k.max(), 200)
        iv_fit = np.sqrt(svi_total_variance(grid, f.params) / s.t_years)
        ax.scatter(s.k, iv_mkt, s=14, color="#1f6feb", alpha=0.7, label="market")
        ax.plot(grid, iv_fit, color="#39d353", lw=2, label="raw SVI (slice-wise)")
        if ssvi is not None:
            iv_ssvi = np.sqrt(
                np.maximum(svi_total_variance(grid, ssvi.slice_params(i)), 1e-12) / s.t_years
            )
            ax.plot(grid, iv_ssvi, color="#f0883e", lw=1.8, ls="--", label="SSVI (no-arb)")
        flag = "" if f.butterfly_arbitrage_free else "  [BUTTERFLY ARB]"
        ax.set_title(f"{s.expiry}  (T={s.t_years:.2f}y, g_min={f.g_min:+.3f}){flag}", fontsize=9)
        ax.set_xlabel("log-moneyness k")
        ax.set_ylabel("implied vol")
        ax.legend(fontsize=8)
    subtitle = "raw-SVI fits" if ssvi is None else "raw-SVI vs arbitrage-free SSVI"
    fig.suptitle(f"{surface.ticker} implied-vol smiles: {subtitle} — {surface.asof}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_surface_3d(surface: FittedSurface, save_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_grid = np.linspace(-0.4, 0.4, 60)
    ts = np.array([s.t_years for s in surface.slices])
    iv = np.array(
        [
            np.sqrt(np.maximum(svi_total_variance(k_grid, f.params), 1e-10) / s.t_years)
            for s, f in zip(surface.slices, surface.fits)
        ]
    )
    kk, tt = np.meshgrid(k_grid, ts)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(kk, tt, iv, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("maturity (years)")
    ax.set_zlabel("implied vol")
    ax.set_title(f"{surface.ticker} fitted SVI surface — {surface.asof}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_densities(surface: FittedSurface, save_path: str, ssvi=None) -> None:
    """Risk-neutral density implied by each expiry's fitted smile.

    Draws the Breeden-Litzenberger density p(k) over log-moneyness for every
    slice (raw SVI, plus the arbitrage-free SSVI overlay when ``ssvi`` is
    supplied — an ``SSVIFit`` index-aligned with ``surface.slices``). A dip
    below the zero line is a *negative* density: butterfly arbitrage, the very
    thing g_min < 0 flags on the smile chart — same diagnostic, dual view.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(surface.slices)
    cols = 3
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for i, (ax, s, f) in enumerate(zip(axes.flat, surface.slices, surface.fits)):
        lo = min(float(s.k.min()), -0.5)
        hi = max(float(s.k.max()), 0.5)
        grid = np.linspace(lo, hi, 400)
        ax.plot(grid, risk_neutral_density(grid, f.params), color="#39d353", lw=2,
                label="raw SVI")
        if ssvi is not None:
            ax.plot(grid, risk_neutral_density(grid, ssvi.slice_params(i)),
                    color="#f0883e", lw=1.8, ls="--", label="SSVI (no-arb)")
        ax.axhline(0.0, color="#8b949e", lw=0.8, ls=":")
        flag = "" if f.butterfly_arbitrage_free else "  [NEG DENSITY]"
        ax.set_title(f"{s.expiry}  (T={s.t_years:.2f}y, g_min={f.g_min:+.3f}){flag}",
                     fontsize=9)
        ax.set_xlabel("log-moneyness k")
        ax.set_ylabel("risk-neutral density")
        ax.legend(fontsize=8)
    subtitle = "raw SVI" if ssvi is None else "raw SVI vs arbitrage-free SSVI"
    fig.suptitle(f"{surface.ticker} implied risk-neutral densities: {subtitle} "
                 f"— {surface.asof}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_local_vol(ssvi, save_path: str) -> None:
    """Dupire local-volatility surface implied by the fitted SSVI surface.

    Local variance v_L(k, t) = (dw/dt) / g(k) reuses the exact butterfly
    g-function; this heatmap over (log-moneyness, maturity) is the third view
    of the same fit, after the smile and the risk-neutral density.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .localvol import local_vol_surface

    k_grid, t_grid, surf = local_vol_surface(ssvi, n_k=80, n_t=40)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.pcolormesh(k_grid, t_grid, surf, cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax, label="local volatility")
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("maturity (years)")
    ax.set_title("Dupire local-vol surface from SSVI  (v_L = dw/dt / g)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
