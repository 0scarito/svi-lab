"""SSVI: the surface-level SVI parameterization, arbitrage-free by construction.

Source: Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility surfaces",
Quantitative Finance 14(1) — Section 4.

Definition 4.1 (eq. 4.1), with theta_t the ATM total implied variance:

    w(k, theta) = (theta/2) * [ 1 + rho*phi(theta)*k
                                + sqrt((phi(theta)*k + rho)^2 + 1 - rho^2) ]

Power-law curvature (eq. 4.5):  phi(theta) = eta / (theta^gamma * (1+theta)^(1-gamma)).
The paper states this choice "gives a surface that is completely free of static
arbitrage provided that eta*(1+|rho|) <= 2" — a claim anchored in its gamma=1/2
discussion. For gamma > 1/2, Corollary 4.1's condition 4 (theta*phi^2*(1+|rho|)
<= 4) diverges as theta -> 0, so the guarantee is NOT global. We therefore
restrict gamma to (0, 1/2], where all four conditions hold for every theta > 0
whenever eta*(1+|rho|) <= 2:
  - cond 3: theta*phi = eta*(theta/(1+theta))^(1-gamma) <= eta, so
    theta*phi*(1+|rho|) <= 2 < 4;
  - cond 4: for gamma <= 1/2, sup_theta theta*phi^2 <= eta^2, and
    eta^2*(1+|rho|) = [eta*(1+|rho|)]*eta <= 2*eta <= 4 (since eta <= 2);
  - cond 2: d(theta*phi)/dtheta / phi = (1-gamma)/(1+theta) < 1 <=
    (1/rho^2)(1+sqrt(1-rho^2)) for every |rho| < 1.
Both constraints (the eta bound via reparameterization, the gamma cap via
optimizer bounds) are enforced *inside* the fit, so every surface this module
returns is arbitrage-free by construction — provably, not just grid-checked.

Lemma 3.1 (natural -> raw SVI, with Delta=mu=0): each SSVI slice at fixed theta
IS a raw SVI slice with

    a = (theta/2)*(1-rho^2),  b = theta*phi/2,  m = -rho/phi,
    sigma = sqrt(1-rho^2)/phi,  rho = rho.

We use that mapping to verify every fitted surface through the *same* g-function
butterfly test used for the slice-wise raw fits — the check is shared, only the
parameterization differs.

Corollary 4.1 (free of static arbitrage) requires:
  1. theta_t non-decreasing in t          (we enforce by isotonic projection)
  2. 0 <= d/dtheta[theta*phi] <= (1/rho^2)(1+sqrt(1-rho^2))*phi
  3. theta*phi*(1+|rho|) < 4
  4. theta*phi^2*(1+|rho|) <= 4
All four are re-checked numerically on every fit (``check_static_arbitrage``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .chain import ChainSlice
from .svi import SVIParams, svi_total_variance


def phi_power_law(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """Power-law curvature phi(theta) = eta / (theta^gamma (1+theta)^(1-gamma))."""
    theta = np.asarray(theta, dtype=float)
    return eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_total_variance(
    k: np.ndarray, theta: float, rho: float, eta: float, gamma: float
) -> np.ndarray:
    """SSVI total implied variance w(k, theta) — eq. (4.1) with power-law phi."""
    k = np.asarray(k, dtype=float)
    p = phi_power_law(theta, eta, gamma)
    return 0.5 * theta * (1.0 + rho * p * k + np.sqrt((p * k + rho) ** 2 + 1.0 - rho**2))


def ssvi_slice_params(theta: float, rho: float, eta: float, gamma: float) -> SVIParams:
    """The raw-SVI parameters of the SSVI slice at ``theta`` (Lemma 3.1)."""
    p = float(phi_power_law(theta, eta, gamma))
    return SVIParams(
        a=0.5 * theta * (1.0 - rho**2),
        b=0.5 * theta * p,
        rho=rho,
        m=-rho / p,
        sigma=np.sqrt(1.0 - rho**2) / p,
    )


@dataclass(frozen=True)
class SSVIFit:
    rho: float
    eta: float
    gamma: float
    thetas: np.ndarray            # ATM total variance per expiry (isotonic)
    t_years: np.ndarray
    rmse_per_slice: np.ndarray    # total-variance units, same convention as SliceFit
    rmse_global: float
    bound: float                  # eta*(1+|rho|), guaranteed <= 2

    def slice_params(self, i: int) -> SVIParams:
        return ssvi_slice_params(float(self.thetas[i]), self.rho, self.eta, self.gamma)


def atm_total_variance(slices: list[ChainSlice]) -> np.ndarray:
    """theta_t observed from the market: interpolate each slice's w at k = 0,
    then project onto the non-decreasing cone (Corollary 4.1, condition 1).

    Remark 4.1 of the paper treats theta_t as directly observable; the isotonic
    projection only ever moves values by the size of the market noise."""
    thetas = []
    for s in slices:
        order = np.argsort(s.k)
        thetas.append(float(np.interp(0.0, s.k[order], s.w[order])))
    out = np.maximum.accumulate(np.asarray(thetas, dtype=float))
    # strictly positive and strictly increasing by a hair, so phi() and
    # calendar checks are well defined even on flat term structures
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] * (1.0 + 1e-9)
    return out


def fit_ssvi(slices: list[ChainSlice]) -> SSVIFit:
    """Calibrate one global (rho, eta, gamma) across ALL expiries jointly.

    theta_t is pinned to the observed ATM total variance (not optimized), so
    the fit has exactly 3 free parameters for the whole surface — versus 5 per
    expiry for the slice-wise raw fits. The butterfly bound eta*(1+|rho|) <= 2
    is enforced by construction: we optimize s in (0, 1] with
    eta = 2*s / (1+|rho|).
    """
    if len(slices) < 2:
        raise ValueError("SSVI is a surface fit; need at least 2 expiries")
    ordered = sorted(slices, key=lambda s: s.t_years)
    thetas = atm_total_variance(ordered)

    def unpack(x):
        rho, s, gamma = x
        eta = 2.0 * s / (1.0 + abs(rho))
        return rho, eta, gamma

    def residuals(x):
        rho, eta, gamma = unpack(x)
        res = []
        for th, sl in zip(thetas, ordered):
            res.append(ssvi_total_variance(sl.k, float(th), rho, eta, gamma) - sl.w)
        return np.concatenate(res)

    lb = [-0.999, 1e-3, 0.05]
    ub = [0.999, 1.0, 0.5]  # gamma <= 1/2: the provably-global no-arb zone
    starts = [[-0.7, 0.5, 0.45], [-0.3, 0.8, 0.3], [0.0, 0.3, 0.5]]
    best = None
    for x0 in starts:
        try:
            r = least_squares(residuals, x0, bounds=(lb, ub), method="trf")
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("SSVI fit failed from every start")

    rho, eta, gamma = unpack(best.x)
    per_slice = np.array(
        [
            float(np.sqrt(np.mean(
                (ssvi_total_variance(sl.k, float(th), rho, eta, gamma) - sl.w) ** 2
            )))
            for th, sl in zip(thetas, ordered)
        ]
    )
    n_total = sum(sl.k.size for sl in ordered)
    rmse_global = float(np.sqrt(best.cost * 2.0 / n_total))
    return SSVIFit(
        rho=float(rho),
        eta=float(eta),
        gamma=float(gamma),
        thetas=thetas,
        t_years=np.array([s.t_years for s in ordered]),
        rmse_per_slice=per_slice,
        rmse_global=rmse_global,
        bound=float(eta * (1.0 + abs(rho))),
    )


def check_static_arbitrage(fit: SSVIFit, n_grid: int = 400) -> dict[str, bool]:
    """Numerically re-check all four Corollary 4.1 conditions on a theta grid
    spanning the fitted surface. Returns per-condition booleans; a correct fit
    passes all four by construction — this is the trust-but-verify layer."""
    th_grid = np.linspace(fit.thetas.min() * 0.5, fit.thetas.max() * 2.0, n_grid)
    p = phi_power_law(th_grid, fit.eta, fit.gamma)
    theta_phi = th_grid * p

    # condition 1: fitted thetas non-decreasing
    c1 = bool(np.all(np.diff(fit.thetas) >= 0))
    # condition 2: 0 <= d(theta*phi)/dtheta <= (1/rho^2)(1+sqrt(1-rho^2))*phi
    d_theta_phi = np.gradient(theta_phi, th_grid)
    if abs(fit.rho) < 1e-12:
        upper = np.full_like(p, np.inf)
    else:
        upper = (1.0 / fit.rho**2) * (1.0 + np.sqrt(1.0 - fit.rho**2)) * p
    c2 = bool(np.all(d_theta_phi >= -1e-10) and np.all(d_theta_phi <= upper + 1e-10))
    # conditions 3 & 4
    c3 = bool(np.all(theta_phi * (1.0 + abs(fit.rho)) < 4.0))
    c4 = bool(np.all(theta_phi * p * (1.0 + abs(fit.rho)) <= 4.0 + 1e-12))
    return {
        "theta_nondecreasing": c1,
        "phi_slope_bounds": c2,
        "wing_slope_lt_4": c3,
        "curvature_le_4": c4,
        "all": c1 and c2 and c3 and c4,
    }


def ssvi_slice_total_variance(fit: SSVIFit, i: int, k: np.ndarray) -> np.ndarray:
    """Convenience: w(k) of fitted slice i via the raw-SVI mapping (Lemma 3.1).
    Identical to ssvi_total_variance by construction — used in tests to prove
    the mapping, and in verification to reuse the raw-SVI g-function."""
    return svi_total_variance(k, fit.slice_params(i))
