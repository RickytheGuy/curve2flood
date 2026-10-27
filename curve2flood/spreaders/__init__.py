"""The flood mappers: kernel-weighted, multi-point interpolation and FLDPLN."""
from .fldpln import build_fldpln_library, make_fldpln_flood_map
from .kernel_weighted import create_kernel_weighted_spread_map, spread_stream_cells
from .multi_point import (
    compute_tw_multfact_scale,
    filter_outliers,
    multi_point_interpolation,
)

__all__ = [
    "build_fldpln_library",
    "compute_tw_multfact_scale",
    "create_kernel_weighted_spread_map",
    "filter_outliers",
    "make_fldpln_flood_map",
    "multi_point_interpolation",
    "spread_stream_cells",
]
