# Changelog

## 0.4.0 — 2026-07-26

- **Dupire local volatility** (`svi_lab.localvol`) — the local-vol surface from
  the fitted SSVI surface via Gatheral (2006): `v_L(k,t) = (∂w/∂t) / g(k)`. The
  denominator is the **exact same `g_function`** the lab already uses for the
  butterfly test, so the two arbitrage checks it runs are precisely the
  conditions for a valid local-vol surface: `g(k) > 0` (butterfly ⇒ positive
  denominator) and `∂w/∂t > 0` (calendar ⇒ positive numerator). Local variance
  is non-negative exactly on the arbitrage-free region — the same statement seen
  a third way, after the smile fit and the risk-neutral density.
- `local_variance(fit, k, t)` returns `(v_L, ∂w/∂t, g)`; `local_vol(...)` the
  volatility; `local_vol_surface(...)` a `(k, t)` grid. `∂w/∂t` is a central
  finite difference in `t` held inside the fitted maturity range; `θ_t` is
  interpolated across the fitted expiries (monotone, so `dθ/dt ≥ 0`).
- `plots.plot_local_vol()` renders the local-vol heatmap; `scripts/refresh.py`
  now writes `charts/localvol.png`.
- 33 tests (6 new): the flat-smile / linear-variance case recovers implied vol
  exactly; the Dupire denominator equals `g_function` to 1e-12 (the reuse claim,
  as a test); local variance is non-negative on an arbitrage-free fit; a
  negative-ρ surface gives an asymmetric (downward-skewed) local vol.

## 0.3.0 — 2026-07-25

- **Risk-neutral density extraction** (`svi_lab.density`) — the
  Breeden-Litzenberger (1978) density in closed form via Gatheral (2006):
  `p(k) = g(k)/sqrt(2π·w(k))·exp(-d₋(k)²/2)`, `d₋(k) = -k/√w - √w/2`. It reuses
  the **exact same `g_function`** that flags butterfly arbitrage — no new math,
  no numerical differentiation of noisy quotes. An arbitrage-free smile and a
  valid density integrating to one are the same statement, so `g(k) ≥ 0`
  everywhere ⇔ `p(k) ≥ 0` everywhere ⇔ mass ≈ 1.
- `risk_neutral_density(k, params)` works for any `SVIParams` — raw slice fits
  and SSVI slices (via `ssvi_slice_params` / `SSVIFit.slice_params`) alike.
  `density_stats(k_grid, params)` returns `(total_mass, mean, variance)` by
  trapezoid integration; `slice_density(k, fit[, i])` dispatches on `SliceFit`
  vs `SSVIFit`.
- `plots.plot_densities()` draws each expiry's implied density over
  log-moneyness (raw SVI + arbitrage-free SSVI overlay); `scripts/refresh.py`
  now writes `charts/densities.png` and a per-slice mass table into
  `data/latest.json`. Live 2026-07-25 SPY snapshot: all 8 expiries integrate to
  1.0000 (raw and SSVI), every slice butterfly-free.
- 27 tests (7 new): integrates-to-one on a benign slice, non-negativity where
  g ≥ 0, the Black-Scholes lognormal limit as b → 0 (density → N(-w₀/2, w₀) to
  1e-3), finite mean / positive variance, the negative-density flip side on an
  arbitrageable slice, and SSVI slice densities integrating to one through the
  shared machinery.

## 0.2.0 — 2026-07-21

- **SSVI surface calibration** (`svi_lab.ssvi`) — Gatheral & Jacquier (2014)
  Section 4: power-law φ (eq. 4.5), 3 global parameters + the observed ATM
  total-variance term structure, **arbitrage-free by construction**. The
  butterfly bound η(1+|ρ|) ≤ 2 is enforced via reparameterization inside the
  optimizer, and γ is capped at ½ — the zone where all four Corollary 4.1
  conditions hold for every θ > 0 (derivation in the module docstring), so the
  guarantee is provable, not grid-checked. `check_static_arbitrage()` re-checks
  all four conditions numerically anyway (trust, but verify).
- Every SSVI slice is converted to raw-SVI parameters via Lemma 3.1 and pushed
  through the **same g-function machinery** as the slice-wise fits — one
  shared arbitrage test for both parameterizations.
- `scripts/refresh.py` now fits both, prints a raw-vs-SSVI comparison table
  (rmse, g_min per slice, calendar violations), stores both in
  `data/latest.json`, and overlays SSVI on `charts/smiles.png`.
- Degenerate-chain guard: refresh aborts (keeping the previous snapshot) when
  Yahoo serves fewer than 4 usable expiries — observed in the wild 2026-07-21.
- 20 tests (11 new for SSVI), including a hostile-data test: quotes generated
  *outside* the arb-free zone still come back with a bound-respecting,
  statically-arbitrage-free surface.

## 0.1.0 — 2026-07-03

- Initial release: chain cleaning + parity forwards, raw-SVI slice fitting,
  Gatheral–Jacquier g-function butterfly test with analytic derivatives,
  calendar-crossing counter, daily-refresh GitHub Action, first real SPY
  snapshot (front-month butterfly arbitrage honestly flagged).
