"""Charts: smile grid (market vs fit) and the 3D fitted surface."""

from __future__ import annotations

import numpy as np

from .surface import FittedSurface
from .svi import svi_total_variance


def plot_smile_grid(surface: FittedSurface, save_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(surface.slices)
    cols = 3
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, s, f in zip(axes.flat, surface.slices, surface.fits):
        iv_mkt = np.sqrt(s.w / s.t_years)
        grid = np.linspace(s.k.min(), s.k.max(), 200)
        iv_fit = np.sqrt(svi_total_variance(grid, f.params) / s.t_years)
        ax.scatter(s.k, iv_mkt, s=14, color="#1f6feb", alpha=0.7, label="market")
        ax.plot(grid, iv_fit, color="#39d353", lw=2, label="SVI fit")
        flag = "" if f.butterfly_arbitrage_free else "  [BUTTERFLY ARB]"
        ax.set_title(f"{s.expiry}  (T={s.t_years:.2f}y, g_min={f.g_min:+.3f}){flag}", fontsize=9)
        ax.set_xlabel("log-moneyness k")
        ax.set_ylabel("implied vol")
        ax.legend(fontsize=8)
    fig.suptitle(f"{surface.ticker} implied-vol smiles vs raw-SVI fits — {surface.asof}")
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
