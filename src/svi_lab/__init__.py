"""svi-lab: arbitrage-checked SVI smiles fitted to real option chains."""

from .chain import ChainSlice, fetch_slices, parity_forward, slice_from_frames
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

__version__ = "0.1.0"

__all__ = [
    "ChainSlice",
    "FittedSurface",
    "SVIParams",
    "SliceFit",
    "calendar_violations",
    "fetch_slices",
    "fit_surface",
    "fit_svi_slice",
    "g_function",
    "parity_forward",
    "slice_from_frames",
    "svi_derivatives",
    "svi_total_variance",
]
