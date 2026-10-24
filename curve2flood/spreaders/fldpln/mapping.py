from __future__ import annotations

import os

import numba
import numpy as np
import polars as pl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange

from .solvers import _next_downstream
from . import library_writer

@njit(cache=True, nogil=True)
def _walk_reaches(starts: np.ndarray, lengths: np.ndarray, fdr: np.ndarray, nrows: int,
                  ncols: int) -> tuple[np.ndarray, np.ndarray]:
    """Each reach's stream pixels, walked ``lengths[i]`` pixels down the FDR from ``starts[i]`` (or to a sink or the
    raster's edge): all of them, and where each reach's begin (``offsets``, one more than the reaches)."""
    total = 0
    for i in range(starts.size):
        total += max(lengths[i], 0)

    pixels = np.empty(total, dtype=np.int64)
    offsets = np.zeros(starts.size + 1, dtype=np.int64)
    n = 0
    for i in range(starts.size):
        pixel = starts[i]
        for _ in range(lengths[i]):
            pixels[n] = pixel
            n += 1
            pixel = _next_downstream(pixel, fdr, nrows, ncols)
            if pixel == -1:
                break
        offsets[i + 1] = n

    return pixels[:n], offsets

def _deepest_water(best: np.ndarray, key: np.ndarray, fpp: np.ndarray, dtf: np.ndarray, fill: np.ndarray,
                   dof: np.ndarray, below: np.ndarray):
    """Raise ``best[p]`` to the water of each library row ``(key, p, DTF, fill)`` that reaches ``p``'s filled
    surface: ``dof[key] - DTF + max(fill, 0)`` where ``dof[key] - DTF > 0``.

    ``dof`` is NaN for a key without a depth.  The arithmetic is in the
    columns' own types, rounded to float32, and a NaN depth leaves its cell
    NaN, as ``np.maximum.at`` did.  Unless ``below`` is empty, every row's
    cell also gets ``min(fill, 0)`` there: how far the filled DEM sits below
    the DEM.  A cell's result doesn't depend on the order of its rows (the
    deepest, or NaN, and its one fill), so they are worked through in
    parallel, each thread keeping to its own range of cells.
    """
    _deepest_water_in(best, key, fpp, dtf, fill, dof, below, numba.get_num_threads())

@njit(cache=True, parallel=True)
def _deepest_water_in(best: np.ndarray, key: np.ndarray, fpp: np.ndarray, dtf: np.ndarray, fill: np.ndarray,
                      dof: np.ndarray, below: np.ndarray, n_parts: int):
    """``_deepest_water``, its cells in ``n_parts`` ranges, each range's rows in a thread of their own."""
    for part in prange(n_parts):
        lo = best.size * part // n_parts
        hi = best.size * (part + 1) // n_parts
        for i in range(key.size):
            if lo <= fpp[i] < hi:
                _deepest_row(best, key, fpp, dtf, fill, dof, below, i)

@njit(cache=True, inline="always")
def _deepest_row(best, key, fpp, dtf, fill, dof, below, i):
    """``_deepest_water`` of library row ``i``."""
    zero = np.float32(0.0)
    if below.size > 0:
        f = fill[i]
        below[fpp[i]] = f if f < zero else zero
    k = key[i]
    if k < 0 or k >= dof.size:
        return
    water = dof[k]
    if water != water:
        return
    head = water - dtf[i]
    if not head > zero:
        return
    f = fill[i]
    depth = np.float32(head + (f if (f > zero or f != f) else zero))
    p = fpp[i]
    current = best[p]
    if current != current:
        return
    if depth != depth or depth > current:
        best[p] = depth

def make_fldpln_flood_map(
        vdt_df: pl.DataFrame,
        fdr: np.ndarray,
        fldpln_library: pl.LazyFrame | str | os.PathLike,
        stream_info_df: pd.DataFrame,
        fill_below_dem: np.ndarray | None = None,
        min_depth: float = 0.0) -> np.ndarray:
    """Flood depth raster for one event, from VDT stages and a floodplain library.

    1. Along each reach (its pixels in ``stream_info_df``, walked down the
       flow directions), the stage of stream pixel ``s`` (its water depth over
       the filled DEM) is the VDT's water surface minus the filled DEM.  The
       filled DEM there is the VDT's ``Elev`` (the DEM under the pixel, the
       bed ARC's depths are measured from) plus the library's ``fill`` of the
       pixel's own cell, its row with ``FPP == FSP``, so the filled DEM itself
       is never read.  A pixel without that row counts as having no VDT
       value, and so does a pixel without a VDT row: it has no stage.
    2. Each segment ``R`` floods at one depth ``DoF(R)``, as the library
       assumes: the median of its stream pixels' stages, every pixel counted
       once, however many floodplain cells it is the source of.  The depth is
       the one the VDT gives the segment, not one chosen for what it floods.
       In a segment with VDT values, a pixel without one takes the stage
       interpolated between the segment's own pixels with one, and held
       beyond the first and last, so ``DoF(R)`` never leaves the range of
       the stages ARC gave the segment. A library without a ``segment`` column (built before reaches were
       split) floods each reach at one depth.
    3. Each library row ``(s, p, DTF, fill)`` whose water reaches ``p``'s
       filled surface (``DoF(R) - DTF(s, p) > 0``, ``R`` being ``s``'s
       segment) gives ``p`` the depth ``DoF(R) - DTF(s, p) + max(fill(p), 0)``:
       the water above the filled DEM, plus how far the filled DEM sits above
       the DEM there.  Water that doesn't reach a filled pit's rim never spills
       into it, however deep the pit.  ``p`` keeps the deepest.  Where the DEM
       sits above the filled DEM (bathymetry raised above the ground the
       library was routed on), the fill is 0: the water was routed across
       the filled surface, so it covers the cell wherever it reaches that
       surface, and the depth is measured from it.  Otherwise a raised cell
       inside the channel, routed through, read as a dry hole.
    4. A cell floods where its depth is above 0 and at least ``min_depth``.
       Water that only just tops a low rise beside a stream on a slope runs
       down everything below it in a film: the library's DTF to all of it is
       that of the rise, so each cell gets the same few millimetres.

    Returns the depth where the cell floods, NaN elsewhere, on ``fdr``'s grid;
    no DEM is needed.  ``vdt_df`` needs ``COMID``, ``Row``, ``Col``, ``Elev``
    and ``WSE``.  ``fldpln_library`` is a Parquet library's path or a
    LazyFrame (a CSV library), read a part at a time.  ``fill_below_dem``, a
    float32 raster, gets ``min(fill, 0)`` at every cell of the library: how
    far the filled DEM sits below the DEM, for the water surface.
    """
    columns = _library_columns(fldpln_library)
    if 'fill' not in columns:
        raise ValueError("The FLDPLN library has no fill column. Rebuild it with build_fldpln_library, "
                         "which now takes the DEM the floods are mapped on.")
    nrows, ncols = fdr.shape
    # C order: the library's flat pixel ids index it
    depth_array = np.full((nrows, ncols), np.nan, dtype=np.float32)

    # Each segment's stream pixels, read a part at a time like the depths
    # below.  Every stream pixel of a reach in the library is the source of
    # at least its own cell, so it has a row there, which gives its fill.
    key = 'segment' if 'segment' in columns else 'FSP'
    group = ['segment', 'FSP'] if key == 'segment' else ['FSP']
    pairs, own_fsp, own_fill = [], [], []
    for part in _library_parts(fldpln_library, group + ['FPP', 'fill']):
        if not len(part['FPP']):
            continue
        # A sorted library repeats each source over its rows: drop the repeats
        # first, cheaply, and leave the rest to the unique below
        repeat = np.ones(len(part['FPP']) - 1, dtype=bool)
        for column in group:
            values = part[column]
            repeat &= values[1:] == values[:-1]

        first = np.r_[True, ~repeat]
        pairs.append(pl.DataFrame({column: part[column][first] for column in group}))
        fsp = part['FSP']
        is_own = fsp == part['FPP']
        own_fsp.append(fsp[is_own])
        own_fill.append(part['fill'][is_own])

    if not pairs:
        return depth_array
    
    library_pixels = pl.concat(pairs).unique()
    # A cell's fill is the same in every row: any one of a source's own rows will do
    own_fsp, first = np.unique(np.concatenate(own_fsp), return_index=True)
    own_fill = np.concatenate(own_fill)[first]

    # The VDT's WSE and Elev at its stream pixels, sorted by pixel; a pixel listed twice keeps its last row
    vdt_fsp = vdt_df['Row'].to_numpy().astype(np.int64) * ncols + vdt_df['Col'].to_numpy()
    order = np.argsort(vdt_fsp, kind='stable')
    vdt_fsp, vdt_wse, vdt_elev = vdt_fsp[order], vdt_df['WSE'].to_numpy()[order], vdt_df['Elev'].to_numpy()[order]
    stream_ids = list(set(vdt_df['COMID']))

    stream_info_dict = stream_info_df.set_index('stream_id').to_dict(orient='index')
    for stream_id in stream_ids:
        if stream_id not in stream_info_dict:
            raise ValueError(f"Stream ID {stream_id} not found in stream_info_df.")

    # Walk each reach's stream pixels, with the VDT's WSE and Elev where it has them
    starts = np.array([stream_info_dict[s]['start_pixel'] for s in stream_ids], dtype=np.int64)
    lengths = np.array([stream_info_dict[s]['length'] for s in stream_ids], dtype=np.int64)
    pixels, offsets = _walk_reaches(starts, lengths, fdr.ravel(), nrows, ncols)
    at = np.searchsorted(vdt_fsp, pixels, side='right') - 1
    known = at >= 0
    known[known] = vdt_fsp[at[known]] == pixels[known]
    wse = np.full(pixels.size, np.nan)
    wse[known] = vdt_wse[at[known]]
    elevation = np.full(pixels.size, np.nan)
    elevation[known] = vdt_elev[at[known]]
    # The filled DEM: the VDT's Elev plus the fill of the pixel's own cell
    fill = np.full(pixels.size, np.nan)
    if own_fsp.size:
        at = np.minimum(np.searchsorted(own_fsp, pixels), own_fsp.size - 1)
        has_fill = own_fsp[at] == pixels
        fill[has_fill] = own_fill[at[has_fill]]

    # Each stream pixel's depth over the filled DEM, missing without a VDT value
    pixel_stage = wse - (elevation + fill)
    measured = ~np.isnan(pixel_stage)
    if not measured.any():
        return depth_array

    # A pixel on two reaches keeps its deeper stage
    stage = (pl.DataFrame({'FSP': pixels, 'DoF': pl.Series(pixel_stage, dtype=pl.Float32, nan_to_null=True),
                           'measured': measured},
                          schema={'FSP': pl.Int32, 'DoF': pl.Float32, 'measured': pl.Boolean})
             .group_by('FSP').agg(pl.col('DoF').max(), pl.col('measured').max()))

    # Each stream pixel's reach and its place along it, from upstream: a
    # segment is a piece of one reach
    sizes = np.diff(offsets)
    place = pl.DataFrame({'FSP': pixels, 'reach': np.repeat(np.array(stream_ids, dtype=np.int64), sizes),
                          'position': np.arange(pixels.size) - np.repeat(offsets[:-1], sizes)},
                         schema={'FSP': pl.Int32, 'reach': pl.Int64, 'position': pl.Int32}
                         ).unique('FSP', keep='first', maintain_order=True)
    if key == 'FSP':
        # A library built before reaches were split into segments: each reach is one
        reach_of = place.select('FSP', segment='reach')
        library_pixels = library_pixels.join(reach_of, on='FSP', how='inner')

    library_pixels = (library_pixels.join(stage, on='FSP', how='inner')
                      .join(place.select('FSP', 'position'), on='FSP', how='inner')
                      .sort(['segment', 'position']))
    if library_pixels.is_empty():
        return depth_array
    
    # A pixel without a VDT value takes its stage from its own segment's pixels with one; a segment without any has
    # none, and floods nothing
    stages = _interpolate_within_segments(library_pixels['segment'].to_numpy(), library_pixels['position'].to_numpy(),
                                          library_pixels['measured'].to_numpy(), library_pixels['DoF'].to_numpy())
    segments, depths = _median_by_group(library_pixels['segment'].to_numpy(), stages)
    depths = depths.astype(np.float32)
    if key == 'FSP':
        df = pl.DataFrame({'segment': segments, 'DoF': depths},
                          schema_overrides={'segment': library_pixels.schema['segment']})
        df = reach_of.join(df, on='segment', how='inner')
        segments, depths = df['FSP'].to_numpy(), df['DoF'].to_numpy()

    # Each key's depth, looked up by the key itself
    valid = segments >= 0
    if not valid.any():
        return depth_array
    
    dof = np.full(int(segments[valid].max()) + 1, np.nan, dtype=np.float32)
    dof[segments[valid]] = depths[valid]

    # Read the library a part at a time and keep only each pixel's deepest
    # water so far, in ``depth_array`` itself: joined whole, the library and
    # its join were several times the size of every raster in the run.
    best = depth_array.ravel()
    best[:] = -np.inf
    below = np.empty(0, np.float32) if fill_below_dem is None else fill_below_dem.reshape(-1)
    for part in _library_parts(fldpln_library, [key, 'FPP', 'DTF', 'fill']):
        _deepest_water(best, part[key], part['FPP'], part['DTF'], part['fill'], dof, below)
        
    _shallow_to_nan(depth_array, np.float32(min_depth))

    return depth_array

@njit(cache=True, parallel=True)
def _shallow_to_nan(depth: np.ndarray, min_depth: np.float32):
    """NaN, in place, where ``depth`` isn't above 0 and at least ``min_depth``."""
    for r in prange(depth.shape[0]):
        for c in range(depth.shape[1]):
            if not (depth[r, c] > 0 and depth[r, c] >= min_depth):
                depth[r, c] = np.nan


def _median_by_group(group: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each group's median: its middle value, or the lower of the two middle ones for an even count.

    Returns the groups, sorted, and their medians.
    """
    order = np.lexsort((value, group))
    group, value = group[order], value[order]
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    sizes = np.diff(np.r_[starts, len(group)])
    return group[starts], value[starts + (sizes - 1) // 2]

def _interpolate_within_segments(segment: np.ndarray, position: np.ndarray, measured: np.ndarray,
                                 stage: np.ndarray) -> np.ndarray:
    """Each pixel's stage, with the unmeasured pixels' taken from the measured pixels of their own segment.

    The pixels are sorted by segment, then by ``position`` along the stream.
    Between two measured pixels of a segment the stage is interpolated
    linearly by position; before the first and after the last it is that
    pixel's.  A segment without a measured pixel keeps ``stage``.
    """
    n = len(segment)
    index = np.arange(n)
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    sizes = np.diff(np.r_[starts, n])
    first, end = np.repeat(starts, sizes), np.repeat(starts + sizes, sizes)
    # The nearest measured pixel of the same segment before or at, and after or at, each pixel
    before = np.maximum.accumulate(np.where(measured, index, -1))
    after = np.minimum.accumulate(np.where(measured, index, n)[::-1])[::-1]
    before = np.where(before >= first, before, after)
    after = np.where(after < end, after, before)
    stage = np.asarray(stage, dtype=float)
    out = stage.copy()
    known = ~measured & (before < end)
    b, a = before[known], after[known]
    span = (position[a] - position[b]).astype(float)
    t = np.divide(position[known] - position[b], span, out=np.zeros(len(b)), where=span > 0)
    out[known] = stage[b] + t * (stage[a] - stage[b])
    return out

def _library_columns(library: pl.LazyFrame | str | os.PathLike) -> list[str]:
    """The columns of a library: a LazyFrame, or a Parquet file's path."""
    if isinstance(library, pl.LazyFrame):
        return library.collect_schema().names()
    return pq.read_schema(library).names

def _library_parts(library: pl.LazyFrame | str | os.PathLike, columns: list[str]):
    """The library's ``columns``, ``_LIBRARY_SLICE_ROWS`` rows at a time, as NumPy arrays by name, with decimals as
    float32.

    A Parquet file is read through pyarrow, a batch at a time: two to five
    times as fast as slicing ``pl.scan_parquet``, and half the memory.
    Without pre-buffering: pyarrow would hold the requested columns of every
    row group until the last batch
    and read it no faster.  Its decimals, hundredths stored as integers, are
    worked out from the integers (``_column_values``); a LazyFrame's (a CSV
    library's) are cast by polars, which gives the same values.
    """
    rows = library_writer._LIBRARY_SLICE_ROWS
    if isinstance(library, pl.LazyFrame):
        n = library.select(pl.len()).collect().item()
        for offset in range(0, n, rows):
            part = library.slice(offset, rows).select(columns).collect()
            part = part.with_columns([pl.col(c).cast(pl.Float32) for c, d in part.schema.items() if d == pl.Decimal])
            yield {c: part[c].to_numpy() for c in columns}
        return
    for batch in pq.ParquetFile(library, pre_buffer=False).iter_batches(batch_size=rows, columns=columns):
        yield {name: _column_values(batch.column(i)) for i, name in enumerate(batch.schema.names)}

def _column_values(column: pa.Array) -> np.ndarray:
    """A library column's values as NumPy, a decimal's as float32.

    A decimal of hundredths stored in 32 bits (``DECIMAL_COLS``) is worked
    out from its integers, ``float32(integer / 100)``, which is what polars'
    cast gives for every one of them (all twenty million of decimal32(7, 2)
    were checked), at a third of the cost; any other decimal is cast by polars.
    """
    if pa.types.is_decimal(column.type):
        if column.type.bit_width == 32 and column.type.scale == 2 and column.null_count == 0:
            integers = np.frombuffer(column.buffers()[1], dtype=np.int32)[column.offset:column.offset + len(column)]
            return _hundredths(integers)
        return pl.from_arrow(column).cast(pl.Float32).to_numpy()
    return column.to_numpy(zero_copy_only=False)

@njit(cache=True, parallel=True)
def _hundredths(integers: np.ndarray) -> np.ndarray:
    """``integers / 100`` as float32: worked out in float64, and rounded once."""
    out = np.empty(integers.size, dtype=np.float32)
    for i in prange(integers.size):
        out[i] = np.float32(integers[i] / 100.0)
    return out
