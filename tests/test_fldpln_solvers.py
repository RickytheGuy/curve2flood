"""Tests for the FLDPLN library solvers.

The important one is :func:`test_iterative_matches_reference_model`, which pins
the iterative solver bit-for-bit against ``fldpln_model_v5ram.py`` -- the port of
the MATLAB model the spreader descends from.  It is run with only the two
conventions this package deliberately changed patched in: WhiteboxTools D8 codes
instead of the MATLAB ones, and the stream table rebuilt in the reference's own
column layout.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve2flood.spreaders import fldpln as F

DATA_DIR = Path(__file__).parent / "data" / "scottsbluff_fldpln" / "inputs"
REFERENCE_MODEL = Path(__file__).resolve().parents[1] / "fldpln_model_v5ram.py"

BG = -9999.0
# A few segments that exercise backfill, spillover and the steady-state loop
# without taking long in the pure-Python reference.
SEGMENTS = [760363429, 760418328, 760308499, 760395048, 760388378, 760536470]


def _opposite(pos: int) -> int:
    dr, dc = F.NEIGHBOR_DELTAS[pos]
    return F.NEIGHBOR_DELTAS.index((-dr, -dc))


@pytest.fixture(scope="module")
def rasters():
    if not DATA_DIR.exists():
        pytest.skip("Scottsbluff FLDPLN fixture is not available.")
    from osgeo import gdal

    gdal.UseExceptions()
    dem = gdal.Open(str(DATA_DIR / "fabdem_fldpln_bathymetry.tif")).ReadAsArray().astype(np.float32)
    fil = gdal.Open(str(DATA_DIR / "fabdem_filled.tif")).ReadAsArray().astype(np.float32)
    fdr = gdal.Open(str(DATA_DIR / "fabdem_flowdir.tif")).ReadAsArray().astype(np.uint8)
    info = pd.read_parquet(DATA_DIR / "fabdem_stream_info.parquet")
    return dem, fil, fdr, info


def _segment(rasters, stream_id, solver, dh, fldmx, iterative_spill=True, margins=None, max_cells=0):
    _, fil, fdr, info = rasters
    previous = F._WINDOW_MARGINS
    if margins is not None:
        F._WINDOW_MARGINS = margins
    try:
        table = F.fldpln_library_for_segment(
            fil, fdr, int(stream_id), info.values, dh, 0.1, fldmx,
            iterative_spill, 0.0, BG, 0.0, solver, None, max_cells)
    finally:
        F._WINDOW_MARGINS = previous
    return table.sort_values("FPP").reset_index(drop=True)


def _library(tmp_path, name, **kwargs):
    path = tmp_path / f"{name}.parquet"
    F.build_fldpln_library(
        str(DATA_DIR / "fabdem_filled.tif"), str(DATA_DIR / "fabdem_stream_info.parquet"),
        str(DATA_DIR / "fabdem_flowdir.tif"), str(path), dh=0.5, fldmn=0.1, iterative_spill=False,
        pbar=False, **kwargs)
    library = pd.read_parquet(path)
    library["DTF"] = library["DTF"].astype(float)
    return library


# --------------------------------------------------------------------------
# D8 conventions
# --------------------------------------------------------------------------

def test_inflow_is_outflow_reversed():
    """A neighbour drains into the centre exactly when it points back at it."""
    for pos in range(8):
        assert F.INFLOW[pos] == F.OUTFLOW[_opposite(pos)]


def test_outflow_matches_next_downstream():
    """OUTFLOW[pos] must route a cell to NEIGHBOR_DELTAS[pos], which is what the
    solvers' flat offsets (``dr * ncols + dc``) assume."""
    nrows = ncols = 5
    centre = 2 * ncols + 2
    for pos, (dr, dc) in enumerate(F.NEIGHBOR_DELTAS):
        fdr = np.zeros(nrows * ncols, dtype=np.uint8)
        fdr[centre] = F.OUTFLOW[pos]
        assert F._next_downstream(centre, fdr, nrows, ncols) == (2 + dr) * ncols + (2 + dc)


# --------------------------------------------------------------------------
# Fidelity to the MATLAB-derived reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stream_id", SEGMENTS)
def test_iterative_matches_reference_model(rasters, stream_id):
    if not REFERENCE_MODEL.exists():
        pytest.skip("fldpln_model_v5ram.py is not available.")
    spec = importlib.util.spec_from_file_location("_v5ram", REFERENCE_MODEL)
    reference = importlib.util.module_from_spec(spec)
    sys.modules["_v5ram"] = reference
    spec.loader.exec_module(reference)

    # Convention 1: WhiteboxTools D8 codes instead of the MATLAB/ESRI ones.
    reference.INFLOW = np.asarray(F.INFLOW, dtype=np.int32)
    reference.OUTFLOW = np.asarray(F.OUTFLOW, dtype=np.int32)

    dem, fil, fdr, info = rasters
    # Convention 2: the reference indexes seg_info by row, with length in col 4.
    seg_info = np.zeros((len(info), 5))
    seg_info[:, 0] = info["start_pixel"].values
    seg_info[:, 1] = info["end_pixel"].values
    seg_info[:, 4] = info["length"].values
    row = int(np.flatnonzero(info["stream_id"].values == stream_id)[0])

    dh, fldmx = 0.5, 3.0
    result = reference.fldpln_model_v5ram(
        row, {"seg": seg_info}, fil, fdr.astype(np.int32), 0.1, fldmx, dh, 0.0,
        True, BG, shape=fil.shape, demdat=dem)
    table = np.asarray(result.fldpln)
    expected = pd.DataFrame({
        "FSP": table[:, 0].astype(np.int64),
        "FPP": table[:, 1].astype(np.int64),
        "DTF": table[:, 2].astype(np.float32),
    }).sort_values("FPP").reset_index(drop=True)

    actual = _segment(rasters, stream_id, "iterative", dh, fldmx)

    assert np.array_equal(actual["FPP"].values.astype(np.int64), expected["FPP"].values)
    assert np.array_equal(actual["DTF"].values, expected["DTF"].values)
    assert np.array_equal(actual["FSP"].values.astype(np.int64), expected["FSP"].values)


# --------------------------------------------------------------------------
# Exact solver
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stream_id", SEGMENTS)
def test_exact_is_the_shrinking_dh_limit(rasters, stream_id):
    """Shrinking dh must move the depth-stepped result monotonically toward the
    exact field, never past it: the exact solver is a lower bound on DTF and an
    upper bound on extent."""
    exact = _segment(rasters, stream_id, "exact", 0.1, 3.0).set_index("FPP")["DTF"]
    previous_error = None
    for dh in (1.0, 0.5, 0.25):
        stepped = _segment(rasters, stream_id, "iterative", dh, 3.0).set_index("FPP")["DTF"]
        assert stepped.index.isin(exact.index).all(), "stepped found cells the exact solver missed"
        shared = exact.loc[stepped.index]
        assert (stepped.values >= shared.values - 1e-4).all(), "stepped beat the exact minimum"
        error = float((stepped.values - shared.values).mean())
        if previous_error is not None:
            assert error <= previous_error + 1e-6, "error grew as dh shrank"
        previous_error = error


def test_exact_dtf_is_at_least_the_net_climb(rasters):
    """DTF is the summed positive rise along the cheapest path, so it is never
    less than the net elevation gain from its own flood source -- a path that
    dips and climbs back pays for the climb twice, which is what makes FLDPLN
    differ from a plain height-above-nearest-drainage."""
    _, fil, _, _ = rasters
    flat = fil.ravel()
    table = _segment(rasters, 760363429, "exact", 0.1, 5.0)
    rise = flat[table["FPP"].values] - flat[table["FSP"].values]
    assert (table["DTF"].values >= -1e-6).all()
    # Tolerance covers accumulated float32 elevation differencing plus the exact
    # solver's fixed-point step; measured worst case over this fixture is ~6 mm.
    assert (table["DTF"].values >= rise - 0.02).all()


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("solver,dh", [("exact", 0.1), ("iterative", 0.5)])
def test_window_size_does_not_change_the_result(rasters, solver, dh):
    _, fil, _, _ = rasters
    whole = (max(fil.shape),)
    for stream_id in SEGMENTS:
        windowed = _segment(rasters, stream_id, solver, dh, 7.2)
        unwindowed = _segment(rasters, stream_id, solver, dh, 7.2, margins=whole)
        assert np.array_equal(windowed["FPP"].values, unwindowed["FPP"].values)
        assert np.array_equal(windowed["DTF"].values, unwindowed["DTF"].values)
        assert np.array_equal(windowed["FSP"].values, unwindowed["FSP"].values)


def test_tiny_window_still_grows_to_the_right_answer(rasters):
    """A window far too small must be detected and grown, not silently truncated."""
    _, fil, _, _ = rasters
    reference = _segment(rasters, 760363429, "exact", 0.1, 5.0, margins=(max(fil.shape),))
    grown = _segment(rasters, 760363429, "exact", 0.1, 5.0, margins=(1, 4, 16, max(fil.shape)))
    assert np.array_equal(grown["FPP"].values, reference["FPP"].values)
    assert np.array_equal(grown["DTF"].values, reference["DTF"].values)


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------

def test_unknown_solver_is_rejected(rasters):
    with pytest.raises(ValueError, match="solver must be one of"):
        _segment(rasters, SEGMENTS[0], "dijkstra", 0.5, 1.0)


def test_iterative_requires_positive_dh(rasters):
    with pytest.raises(ValueError, match="dh must be positive"):
        _segment(rasters, SEGMENTS[0], "iterative", 0.0, 1.0)


def test_exact_ignores_dh(rasters):
    """dh is not a parameter of the exact solver, so it must not be consulted."""
    a = _segment(rasters, SEGMENTS[0], "exact", 0.01, 3.0)
    b = _segment(rasters, SEGMENTS[0], "exact", 5.00, 3.0)
    assert np.array_equal(a["FPP"].values, b["FPP"].values)
    assert np.array_equal(a["DTF"].values, b["DTF"].values)


# --------------------------------------------------------------------------
# max_cells: a floodplain budget per segment
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stream_id", SEGMENTS)
def test_max_cells_is_a_lower_max_depth(rasters, stream_id):
    """A segment cut by max_cells is the segment solved with fldmx just below
    the depth it stopped at, so the cap finds a max depth rather than dropping
    arbitrary cells."""
    full = _segment(rasters, stream_id, "exact", 0.1, 5.0)
    zero_depth = len(_segment(rasters, stream_id, "exact", 0.1, 0.0))
    if len(full) - zero_depth < 2:
        pytest.skip("the segment floods (almost) nothing above zero depth, so there is no depth to cut at")
    max_cells = zero_depth + (len(full) - zero_depth) // 2
    capped = _segment(rasters, stream_id, "exact", 0.1, 5.0, max_cells=max_cells)
    limit = capped.attrs["depth_limit"]
    assert limit is not None and len(capped) <= max_cells
    assert capped["DTF"].max() <= max(limit, 0.1) + 1e-6  # DTFs below fldmn are reported at fldmn
    lowered = _segment(rasters, stream_id, "exact", 0.1, limit - 1 / F._DTF_UNITS_PER_METRE)
    assert np.array_equal(capped["FPP"].values, lowered["FPP"].values)
    assert np.array_equal(capped["DTF"].values, lowered["DTF"].values)
    assert np.array_equal(capped["FSP"].values, lowered["FSP"].values)


def test_max_cells_keeps_the_cells_flooded_at_no_depth(rasters):
    """There is no depth to cut the zero-depth floodplain back to, so a budget
    smaller than it keeps exactly that."""
    zero_depth = _segment(rasters, SEGMENTS[0], "exact", 0.1, 0.0)
    capped = _segment(rasters, SEGMENTS[0], "exact", 0.1, 5.0, max_cells=1)
    assert capped.attrs["depth_limit"] > 0
    assert np.array_equal(capped["FPP"].values, zero_depth["FPP"].values)
    assert np.array_equal(capped["FSP"].values, zero_depth["FSP"].values)


def test_max_cells_above_the_floodplain_changes_nothing(rasters):
    full = _segment(rasters, SEGMENTS[0], "exact", 0.1, 5.0)
    capped = _segment(rasters, SEGMENTS[0], "exact", 0.1, 5.0, max_cells=len(full))
    assert capped.attrs["depth_limit"] is None
    assert np.array_equal(capped["FPP"].values, full["FPP"].values)
    assert np.array_equal(capped["DTF"].values, full["DTF"].values)


def test_max_cells_needs_the_exact_solver(rasters):
    with pytest.raises(ValueError, match="max_cells needs solver='exact'"):
        _segment(rasters, SEGMENTS[0], "iterative", 0.5, 1.0, max_cells=10)


def test_library_caps_each_segment(rasters, tmp_path):
    """The capped library is the full one with each segment cut at its own depth."""
    full = _library(tmp_path, "full", fldmx=5.0, stream_ids=SEGMENTS[:2])
    capped = _library(tmp_path, "capped", fldmx=5.0, stream_ids=SEGMENTS[:2], max_cells_per_segment=len(full) // 4)
    assert 0 < len(capped) < len(full)
    both = capped.merge(full, on=["FSP", "FPP"], how="left", suffixes=("", "_full"))
    assert both["DTF_full"].notna().all() and (both["DTF"] == both["DTF_full"]).all()


@pytest.mark.parametrize("bad", [0, -5, 2.5])
def test_library_rejects_bad_max_cells(rasters, tmp_path, bad):
    with pytest.raises(ValueError, match="max_cells_per_segment must be"):
        _library(tmp_path, "bad", fldmx=1.0, stream_ids=SEGMENTS[:1], max_cells_per_segment=bad)


def test_library_max_cells_needs_the_exact_solver(rasters, tmp_path):
    with pytest.raises(ValueError, match="needs solver='exact'"):
        _library(tmp_path, "bad", fldmx=1.0, stream_ids=SEGMENTS[:1], max_cells_per_segment=10, solver="iterative")


# --------------------------------------------------------------------------
# max_drop_below_source: cells the mapper can never use
# --------------------------------------------------------------------------

def _slope(nrows=40, ncols=40, drop_per_row=0.5):
    """A plane falling south with a stream running east along row 5.  A downhill
    step costs no depth, so the stream reaches every cell south of it at DTF 0 --
    the hillside the mapper's max_drop_below_source then throws away."""
    rows = np.arange(nrows, dtype=np.float32)[:, None]
    fil = np.repeat(100.0 - drop_per_row * rows, ncols, axis=1).astype(np.float32)
    fdr = np.full((nrows, ncols), 8, dtype=np.uint8)  # Whitebox D8: south
    fdr[5, 10:20] = 2                                   # east, along the stream
    info = np.array([[5 * ncols + 10, 5 * ncols + 20, 11, 1]], dtype=np.int64)
    return fil, fdr, info


def _slope_segment(fldmx, solver="exact", dh=0.1, max_drop_below_source=None, max_cells=0):
    fil, fdr, info = _slope()
    return F.fldpln_library_for_segment(fil, fdr, 1, info, dh, 0.1, fldmx, True, 0.0, BG, 0.0,
                                        solver, max_drop_below_source, max_cells)


@pytest.mark.parametrize("solver,dh", [("exact", 0.1), ("iterative", 0.5)])
def test_max_drop_below_source_leaves_out_only_the_deep_cells(solver, dh):
    """The pruned table is the full one less the cells more than fldmx plus the
    drop below their source, with every other row unchanged."""
    flat = _slope()[0].ravel()
    full = _slope_segment(1.0, solver, dh)
    pruned = _slope_segment(1.0, solver, dh, max_drop_below_source=0.5)
    expected = full[flat[full["FSP"].values] - flat[full["FPP"].values] <= 1.5].reset_index(drop=True)
    assert 0 < len(pruned) < len(full)
    assert pruned.equals(expected)


def test_pruned_cells_do_not_count_toward_max_cells():
    """Left-out cells must not use up a segment's budget: on a slope the cells
    reached at no depth fill any budget, but most of them are pruned."""
    kept_at_zero_depth = len(_slope_segment(0.0, max_drop_below_source=3.5))
    max_cells = kept_at_zero_depth + 60
    capped = _slope_segment(3.0, max_cells=max_cells)
    both = _slope_segment(3.0, max_cells=max_cells, max_drop_below_source=0.5)
    assert len(capped) > max_cells  # all zero-depth, nothing to cut back to
    assert len(both) <= max_cells
    assert both.attrs["depth_limit"] > capped.attrs["depth_limit"]


def test_library_prunes_below_source(tmp_path):
    from osgeo import gdal
    fil, fdr, info = _slope()
    for name, array, dtype in (("filled", fil, gdal.GDT_Float32), ("flowdir", fdr, gdal.GDT_Byte)):
        ds = gdal.GetDriverByName("GTiff").Create(str(tmp_path / f"{name}.tif"), fil.shape[1], fil.shape[0], 1, dtype)
        ds.GetRasterBand(1).WriteArray(array)
        ds = None
    pd.DataFrame(info, columns=["start_pixel", "end_pixel", "length", "stream_id"]).to_parquet(tmp_path / "info.parquet")

    def library(name, **kwargs):
        F.build_fldpln_library(str(tmp_path / "filled.tif"), str(tmp_path / "info.parquet"), str(tmp_path / "flowdir.tif"),
                               str(tmp_path / f"{name}.parquet"), dh=0.1, fldmn=0.1, fldmx=1.0, iterative_spill=False,
                               pbar=False, **kwargs)
        return pd.read_parquet(tmp_path / f"{name}.parquet")

    full, pruned = library("full"), library("pruned", max_drop_below_source=0.5)
    keep = fil.ravel()[full["FSP"].values] - fil.ravel()[full["FPP"].values] <= 1.5
    assert 0 < len(pruned) < len(full)
    assert pruned.equals(full[keep].reset_index(drop=True))


def test_library_rejects_a_negative_drop(rasters, tmp_path):
    with pytest.raises(ValueError, match="max_drop_below_source must be"):
        _library(tmp_path, "bad", fldmx=1.0, stream_ids=SEGMENTS[:1], max_drop_below_source=-1.0)


# --------------------------------------------------------------------------
# Background mask
# --------------------------------------------------------------------------

def test_library_masks_with_an_integer_bg_mask(rasters, tmp_path):
    """A 0/1 integer mask, like a lake raster read from disk, masks the same
    cells as a boolean one -- indexed as is, it would mask rows 0 and 1."""
    _, fil, _, _ = rasters
    floodplain = _segment(rasters, SEGMENTS[0], "exact", 0.1, 3.0)["FPP"].values
    mask = np.zeros(fil.shape, dtype=np.uint8)
    mask.ravel()[floodplain[::2]] = 1
    as_int = _library(tmp_path, "int", fldmx=3.0, stream_ids=SEGMENTS[:1], bg_mask=mask)
    as_bool = _library(tmp_path, "bool", fldmx=3.0, stream_ids=SEGMENTS[:1], bg_mask=mask.astype(bool))
    assert len(as_int) and not mask.ravel()[as_int["FPP"].values].any()
    assert as_int.equals(as_bool)


def test_masked_vdt_cells_do_not_set_the_max_depth(rasters, tmp_path):
    """A VDT cell under bg_mask (a lake) is never flooded from, so it must not
    count toward its segment's max depth: measured from bg it would be thousands
    of metres, and the segment would flood to fldmx."""
    _, fil, fdr, info = rasters
    stream_id = SEGMENTS[0]
    pixels = np.asarray(F._segment_pixels(stream_id, info.values, fdr.ravel(), *fil.shape))
    rows, cols = np.divmod(pixels, fil.shape[1])
    vdt_file = tmp_path / "vdt.parquet"
    pd.DataFrame({"COMID": stream_id, "Row": rows, "Col": cols, "wse_1": fil[rows, cols] + 1.0}).to_parquet(vdt_file)
    mask = np.zeros(fil.shape, dtype=bool)
    mask[rows[0], cols[0]] = True
    library = _library(tmp_path, "vdt", fldmx=5.0, vdt_file=str(vdt_file), bg_mask=mask, stream_ids=[stream_id])
    assert len(library) and library["DTF"].max() <= 1.0 + 0.01
