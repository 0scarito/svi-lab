"""svi-lab: arbitrage-checked SVI smiles fitted to real option chains."""

from .chain import ChainSlice, fetch_slices, parity_forward, slice_from_frames
from .ssvi import (
    SSVIFit,
    check_static_arbitrage,
    fit_ssvi,
    phi_power_law,
    ssvi_slice_params,
    ssvi_total_variance,
)
from .surface import FittedSurface, fit_surface
from .svi import (
    SliceFit,
    SVIParams,
    calendar_violations,
    fit_svi_slice,
    g_function,
    svi_derivatives,
    svi_total_variance,
)

__version__ = "0.2.0"

__all__ = [
    "ChainSlice",
    "FittedSurface",
    "SSVIFit",
    "SVIParams",
    "SliceFit",
    "calendar_violations",
    "check_static_arbitrage",
    "fetch_slices",
    "fit_ssvi",
    "fit_surface",
    "fit_svi_slice",
    "g_function",
    "parity_forward",
    "phi_power_law",
    "slice_from_frames",
    "ssvi_slice_params",
    "ssvi_total_variance",
    "svi_derivatives",
    "svi_total_variance",
]
