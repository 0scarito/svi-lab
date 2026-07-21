# Changelog

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
