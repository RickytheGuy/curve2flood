"""FLDPLN: build a floodplain library, and map floods with it.

FLDPLN (Kastens 2008, *Some New Developments on Two Separate Topics:
Statistical Cross Validation and Floodplain Mapping*, chapter 3) asks, for
every stream pixel, which floodplain pixels its water can reach and how deep
the water must be at the stream pixel to get there: the depth to flood (DTF).
Water fills the filled DEM upstream (backfill) and spills over low divides
into neighbouring valleys (spillover).  The library holds one row per
(flood source pixel ``FSP``, floodplain pixel ``FPP``, ``DTF``, ``fill``),
where ``fill`` is how far the filled DEM sits above the DEM at ``FPP``,
and ``segment``, the stretch of stream the ``FSP`` belongs to.  Every pixel
of a segment starts at the same depth, so the library is exact for one water
depth per segment, which is all mapping an event then needs.

Two solvers build the library:

``solver="exact"`` (default)
    DTF is the smallest total climb along any 8-connected path from the stream
    to the pixel (downhill steps are free), found in one Dijkstra sweep.  This
    is the limit of the depth-stepped algorithm as ``dh`` goes to 0.

``solver="iterative"``
    The depth-stepped algorithm, raising the water ``dh`` at a time.  A larger
    ``dh`` damps spillover, which can over-flood on coarse DEMs.

Both solve each stream segment in a window around it that grows when the
floodplain reaches its edge, so the result never depends on the window size.

Flow directions use WhiteboxTools D8 codes, and the stream table has columns
``start_pixel, end_pixel, length, stream_id``.  Pixels are flat row-major
indices, ``row * ncols + col``.
"""

from __future__ import annotations

import math
import geopandas as gpd
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from multiprocessing import shared_memory

import tqdm
import numpy as np
import polars as pl
import pandas as pd
import pyarrow as pa
import networkx as nx
import pyarrow.parquet as pq
import pyarrow.compute as pc
from numba import njit
from osgeo import gdal
from numba.extending import register_jitable

from curve2flood import LOG

# Columns stored as 2-decimal numbers.  A fill can be deep (a pit filled to its
# rim), so it gets more digits than a DTF, which is at most the library's fldmx.
DECIMAL_COLS = {"DTF": pa.decimal32(5, 2), "fill": pa.decimal32(7, 2)}

_SHARED_MEMORYS = {}

# Neighbour order:
# 1 2 3
# 4 x 5
# 6 7 8
# INFLOW/OUTFLOW are indexed by this ordering, and the solvers' flat neighbour
# offsets mirror it as ``dr * ncols + dc``.
NEIGHBOR_DELTAS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Whitebox D8
# 64  128 1
# 32  x   2
# 16  8   4

# ESRI D8
# 32  64 128
# 16  x   1
# 8   4   2

# Whitebox.  INFLOW[pos] is the code a neighbour at NEIGHBOR_DELTAS[pos] carries
# when it drains into the centre cell; OUTFLOW[pos] is the code the centre cell
# carries when it drains into that neighbour.
INFLOW = np.array([4, 8, 16, 2, 32, 1, 128, 64], dtype=np.int32)
OUTFLOW = np.array([64, 128, 1, 32, 2, 16, 8, 4], dtype=np.int32)

SOLVERS = ("exact", "iterative")

# Window half-widths tried in turn, in cells, before falling back to the raster.
_WINDOW_MARGINS = (128, 384, 1152)

# The exact solver carries DTF as an integer count of these units, so that the
# priority queue can be keyed on a single packed int64.  Tenths of a millimetre
# keep the per-edge rounding an order of magnitude below the 3 decimals the
# library is written with, even after accumulating over a long path, while
# leaving an int32 range worth hundreds of kilometres of depth.
_DTF_UNITS_PER_METRE = 10000.0

# Library rows the mapper joins at a time: one row group of a library written by
# save_library.
_LIBRARY_SLICE_ROWS = 1 << 20


@register_jitable(cache=True, nogil=True, forceinline=True)
def _pixel_to_rc(pixel: int, ncols: int) -> tuple[int, int]:
    return pixel // ncols, pixel % ncols

@register_jitable(cache=True, nogil=True, forceinline=True)
def _rc_to_pixel(row: int, col: int, ncols: int) -> int:
    return np.int32(row * ncols + col)

@register_jitable(cache=True, nogil=True)
def _next_downstream(pixel: int, fdr: np.ndarray, nrows: int, ncols: int) -> int:
    """The pixel ``pixel`` drains into, or -1 at a sink or the raster edge."""
    # Whitebox D8 code -> (row, col) step
    OUTFLOW = {
        np.uint8(64): (-1, -1),
        np.uint8(128): (-1, 0),
        np.uint8(1): (-1, 1),
        np.uint8(32): (0, -1),
        np.uint8(2): (0, 1),
        np.uint8(16): (1, -1),
        np.uint8(8): (1, 0),
        np.uint8(4): (1, 1),
    }

    fd = fdr[pixel]
    if fd == 0:
        return -1

    row, col = _pixel_to_rc(pixel, ncols)
    dr, dc = OUTFLOW[fd]
    rr = row + dr
    cc = col + dc
    if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
        return -1
    return _rc_to_pixel(rr, cc, ncols)


@njit(cache=True, nogil=True)
def _segment_pixels(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray,
                    nrows: int, ncols: int) -> list[int]:
    """Return stream pixels for a segment id, walking the FDR downstream."""
    mask = stream_info[:, 3] == stream_id
    if not np.any(mask):
        raise ValueError("stream_id not found in stream_info.")

    start = stream_info[mask, 0][0]
    length = stream_info[mask, 2][0]

    pixels = []
    if length <= 0:
        return pixels

    pixels.append(start)
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        pixels.append(nxt)
        current = nxt
    return pixels

@njit(cache=True, nogil=True)
def _downstream_exclusion(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray,
                          nrows: int, ncols: int) -> set[int]:
    """Pixels downstream of the segment end -- ``TD(R)``, excluded from spillover."""
    end_pixel = stream_info[stream_info[:, 3] == stream_id, 1][0]

    excluded: set[int] = set()
    current = end_pixel
    seen = {current}
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1 or nxt in seen:
            break
        excluded.add(nxt)
        seen.add(nxt)
        current = nxt

    return excluded


# ---------------------------------------------------------------------------
# Windowing
#
# Every solver works on a crop of the rasters surrounded by a one-cell halo of
# background.  The halo lets the inner loops use constant flat neighbour offsets
# with no bounds checks and no row/column division.  Because the halo blocks
# flow, a floodplain that reaches the usable edge would be truncated -- so the
# caller detects that and re-solves in a larger window.
# ---------------------------------------------------------------------------

def _window_bounds(pixels: np.ndarray, nrows: int, ncols: int,
                   margins: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Usable window ``(r0, r1, c0, c1)`` around ``pixels``, clipped to the raster.

    ``margins`` are the top, bottom, left and right margins, in cells.
    """
    top, bottom, left, right = margins
    rows = pixels // ncols
    cols = pixels % ncols
    r0 = max(int(rows.min()) - top, 0)
    r1 = min(int(rows.max()) + bottom + 1, nrows)
    c0 = max(int(cols.min()) - left, 0)
    c1 = min(int(cols.max()) + right + 1, ncols)
    return r0, r1, c0, c1

def _crop(arr2d: np.ndarray, r0: int, r1: int, c0: int, c1: int, fill) -> np.ndarray:
    """Crop ``[r0, r1) x [c0, c1)`` into an array with a one-cell ``fill`` halo."""
    out = np.full((r1 - r0 + 2, c1 - c0 + 2), fill, dtype=arr2d.dtype)
    out[1:-1, 1:-1] = arr2d[r0:r1, c0:c1]
    return out

def _to_local(pixels: np.ndarray, ncols: int, r0: int, c0: int, wc: int) -> np.ndarray:
    """Global flat pixel ids -> window-local flat ids."""
    rows = pixels // ncols
    cols = pixels % ncols
    return ((rows - r0 + 1) * wc + (cols - c0 + 1)).astype(np.int32)

def _to_global(local: np.ndarray, ncols: int, r0: int, c0: int, wc: int) -> np.ndarray:
    """Window-local flat ids -> global flat pixel ids."""
    lr = local // wc
    lc = local - lr * wc
    return ((lr + r0 - 1) * ncols + (lc + c0 - 1)).astype(np.int32)

def _in_window(pixels: np.ndarray, ncols: int,
               r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    rows = pixels // ncols
    cols = pixels % ncols
    return (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)

def _touched_edges(local: np.ndarray, wr: int, wc: int,
                   r0: int, r1: int, c0: int, c1: int,
                   nrows: int, ncols: int) -> tuple[bool, bool, bool, bool]:
    """Which usable window edges that are not raster edges a solved cell sits on.

    Returned as top, bottom, left, right, matching ``_window_bounds``' margins.
    """
    if local.size == 0:
        return False, False, False, False
    lr = local // wc
    lc = local - lr * wc
    return (r0 > 0 and bool(np.any(lr == 1)),
            r1 < nrows and bool(np.any(lr == wr - 2)),
            c0 > 0 and bool(np.any(lc == 1)),
            c1 < ncols and bool(np.any(lc == wc - 2)))


# ---------------------------------------------------------------------------
# Exact solver -- one monotone sweep over the positive-rise metric
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True, inline="always")
def _heap_push(heap: np.ndarray, n: int, key: int) -> int:
    i = n
    heap[i] = key
    while i > 0:
        parent = (i - 1) >> 1
        if heap[parent] <= heap[i]:
            break
        heap[parent], heap[i] = heap[i], heap[parent]
        i = parent
    return n + 1

@njit(cache=True, nogil=True, inline="always")
def _heap_pop(heap: np.ndarray, n: int) -> tuple[int, int]:
    top = heap[0]
    n -= 1
    heap[0] = heap[n]
    i = 0
    while True:
        left = 2 * i + 1
        right = left + 1
        small = i
        if left < n and heap[left] < heap[small]:
            small = left
        if right < n and heap[right] < heap[small]:
            small = right
        if small == i:
            break
        heap[small], heap[i] = heap[i], heap[small]
        i = small
    return top, n

@njit(cache=True, nogil=True)
def _solve_exact(fil: np.ndarray, fdr: np.ndarray, ncols: int,
                 sources: np.ndarray, excluded: np.ndarray,
                 fldmn: float, fldmx: float, global_max_wse: float,
                 bg: float, spill_decay: float, max_cells: int = 0,
                 max_below: float = np.inf):
    """Dijkstra over ``w(u -> v) = max(0, E(v) - E(u))`` on the 8-neighbourhood.

    Returns ``(cells, dtf, fsp, depth_limit)`` in window-local flat ids.
    ``spill_decay`` is added on each step to a pixel's D8 successor, as the
    depth-stepped solver adds it on spillover paths.

    ``max_cells`` caps the floodplain at that many cells (0 is no cap).  Cells
    settle in order of DTF, so once the cap is full the sweep stops at the DTF
    ``d`` of the next cell and keeps only the cells shallower than ``d``: the
    floodplain an ``fldmx`` just below ``d`` would give.  ``depth_limit`` is
    ``d`` in metres, or -1 when the cap was not reached.  Cells flooded at zero
    depth are always kept, and cells more than ``max_below`` metres below their
    source do not count toward the cap (the caller leaves them out).

    The pixels downstream of the segment (``excluded``) are not a wall.  As in
    the depth-stepped algorithm, backfill may enter them, lateral spillover may
    not, and a downhill path may step onto one but not continue along them.
    """
    n = fil.size
    unreached = np.int32(2147483647)
    cap = np.int64(fldmx * _DTF_UNITS_PER_METRE + 0.5)
    decay = np.int64(spill_decay * _DTF_UNITS_PER_METRE + 0.5)

    dist = np.full(n, unreached, dtype=np.int32)
    fsp = np.full(n, np.int32(-1), dtype=np.int32)
    settled = np.zeros(n, dtype=np.bool_)

    offsets = np.empty(8, dtype=np.int64)
    offsets[0] = -ncols - 1
    offsets[1] = -ncols
    offsets[2] = -ncols + 1
    offsets[3] = -1
    offsets[4] = 1
    offsets[5] = ncols - 1
    offsets[6] = ncols
    offsets[7] = ncols + 1

    heap = np.empty(max(1024, 2 * sources.size), dtype=np.int64)
    heap_n = 0
    for i in range(sources.size):
        source = sources[i]
        if fil[source] == bg:
            continue
        dist[source] = 0
        fsp[source] = source
        if heap_n + 1 >= heap.size:
            bigger = np.empty(heap.size * 2, dtype=np.int64)
            bigger[:heap_n] = heap[:heap_n]
            heap = bigger
        heap_n = _heap_push(heap, heap_n, np.int64(source))

    stop = np.int64(-1)
    n_settled = 0
    n_kept = 0
    while heap_n > 0:
        key, heap_n = _heap_pop(heap, heap_n)
        pixel = np.int32(key & np.int64(0xFFFFFFFF))
        depth = np.int64(key >> 32)
        if settled[pixel] or depth != dist[pixel]:
            continue
        if max_cells > 0 and n_kept >= max_cells and depth > 0:
            stop = depth
            break
        settled[pixel] = True
        n_settled += 1

        elevation = fil[pixel]
        if fil[fsp[pixel]] - elevation <= max_below:
            n_kept += 1
        from_excluded = excluded[pixel]
        code = fdr[pixel]

        for pos in range(8):
            neighbor = pixel + offsets[pos]
            if settled[neighbor]:
                continue
            descends = OUTFLOW[pos] == code          # pixel drains into neighbour
            if excluded[neighbor]:
                # TD(R) constrains the reference's spillover step, not its
                # backfill: _backfill_from_source never consults the exclusion
                # set, _spill_candidates refuses to select an excluded cell, and
                # _forward_path may end a trajectory on one but not run along it.
                backfills = fdr[neighbor] == INFLOW[pos]
                if not backfills and not (descends and not from_excluded):
                    continue
            neighbor_elev = fil[neighbor]
            if neighbor_elev == bg or neighbor_elev > global_max_wse:
                continue
            rise = neighbor_elev - elevation
            cost = np.int64(rise * _DTF_UNITS_PER_METRE + 0.5) if rise > 0.0 else np.int64(0)
            if descends:
                cost += decay
            candidate = depth + cost
            if candidate > cap or candidate >= dist[neighbor]:
                continue
            dist[neighbor] = np.int32(candidate)
            fsp[neighbor] = fsp[pixel]
            if heap_n + 1 >= heap.size:
                bigger = np.empty(heap.size * 2, dtype=np.int64)
                bigger[:heap_n] = heap[:heap_n]
                heap = bigger
            heap_n = _heap_push(heap, heap_n, (candidate << 32) | np.int64(neighbor))

    cells = np.empty(n_settled, dtype=np.int32)
    out_dtf = np.empty(n_settled, dtype=np.float32)
    out_fsp = np.empty(n_settled, dtype=np.int32)
    k = 0
    for pixel in range(n):
        if settled[pixel] and (stop < 0 or dist[pixel] < stop):
            cells[k] = pixel
            value = dist[pixel] / _DTF_UNITS_PER_METRE
            out_dtf[k] = fldmn if value < fldmn else value
            out_fsp[k] = fsp[pixel]
            k += 1
    depth_limit = stop / _DTF_UNITS_PER_METRE if stop >= 0 else -1.0
    return cells[:k], out_dtf[:k], out_fsp[:k], depth_limit


# ---------------------------------------------------------------------------
# Depth-stepped solver -- algorithm steps 1-8
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _backfill_from_source(source: int, base_dtf: float, max_wse: float,
                          fil: np.ndarray, fdr: np.ndarray, offsets: np.ndarray,
                          bg: float, skip_flooded: bool, in_flood: np.ndarray,
                          blocked: np.ndarray, blocked_stamp: int,
                          stack: np.ndarray, visited: np.ndarray, visit_stamp: int,
                          out_pixels: np.ndarray, out_dtf: np.ndarray) -> int:
    """BFA step 1: backfill against the FDR from ``source`` (chapter 3, p.113).

    ``DTF`` is measured from ``source``'s elevation, as the dissertation defines
    it; on a filled DEM the inverse-D8 walk is elevation-monotone, so that equals
    the accumulated positive rise.  Returns the number of cells written.
    """
    source_elev = fil[source]
    visited[source] = visit_stamp
    stack[0] = source
    top = 1
    count = 0
    while top > 0:
        top -= 1
        center = stack[top]
        for pos in range(8):
            neighbor = center + offsets[pos]
            if visited[neighbor] == visit_stamp:
                continue
            if blocked_stamp != 0 and blocked[neighbor] == blocked_stamp:
                continue
            if skip_flooded and in_flood[neighbor]:
                continue
            neighbor_elev = fil[neighbor]
            if neighbor_elev == bg or neighbor_elev > max_wse:
                continue
            if fdr[neighbor] != INFLOW[pos]:
                continue
            visited[neighbor] = visit_stamp
            out_pixels[count] = neighbor
            out_dtf[count] = base_dtf + max(0.0, neighbor_elev - source_elev)
            count += 1
            stack[top] = neighbor
            top += 1
    return count

@njit(cache=True, nogil=True)
def _is_interior_boundary(pixel: int, fil: np.ndarray, offsets: np.ndarray,
                          in_flood: np.ndarray, bg: float) -> bool:
    """``pixel`` is on the interior boundary if some neighbour is dry, non-background."""
    for pos in range(8):
        neighbor = pixel + offsets[pos]
        if not in_flood[neighbor] and fil[neighbor] != bg:
            return True
    return False

@njit(cache=True, nogil=True)
def _refresh_boundary(boundary: np.ndarray, n_boundary: int,
                      added: np.ndarray, n_added: int,
                      fil: np.ndarray, offsets: np.ndarray,
                      in_flood: np.ndarray, on_boundary: np.ndarray,
                      bg: float) -> int:
    """Re-derive the full interior boundary ``dI F`` (algorithm steps 3 and 8).

    The floodplain only grows, so a flooded cell that is not on the boundary can
    never rejoin it.  Retesting the previous boundary plus the newly flooded
    cells therefore yields the exact ``dI F``, at ``O(|dI F| + |added|)`` instead
    of a rescan of the whole floodplain.
    """
    kept = 0
    for i in range(n_boundary):
        pixel = boundary[i]
        if _is_interior_boundary(pixel, fil, offsets, in_flood, bg):
            boundary[kept] = pixel
            kept += 1
        else:
            on_boundary[pixel] = False
    for i in range(n_added):
        pixel = added[i]
        if on_boundary[pixel]:
            continue
        if _is_interior_boundary(pixel, fil, offsets, in_flood, bg):
            on_boundary[pixel] = True
            boundary[kept] = pixel
            kept += 1
    return kept

@njit(cache=True, nogil=True)
def _order_by_depth_then_elevation(keys: np.ndarray, depth: np.ndarray,
                                   elevation: np.ndarray) -> np.ndarray:
    """Ascending ``depth``, ties broken by descending ``elevation``.

    The reference orders both the interior boundary and the spillover candidates
    this way (algorithm step 6: "sort the y_k so that they are decreasing in
    spillover flood depth ... decreasing in elevation").  Order matters because
    assimilation keeps the first-arriving minimum, so it decides which FSP claims
    a cell -- and, as Kastens notes, it "help[s] limit redundant processing".
    """
    by_elevation = keys[np.argsort(-elevation[keys], kind="mergesort")]
    return by_elevation[np.argsort(depth[by_elevation], kind="mergesort")]

@njit(cache=True, nogil=True)
def _forward_path(start: int, spill_dtf: float, fil: np.ndarray, fdr: np.ndarray,
                  offsets: np.ndarray, flood_depths: np.ndarray,
                  bg: float, excluded: np.ndarray, spill_decay: float,
                  path: np.ndarray, seen: np.ndarray, seen_stamp: int) -> int:
    """Algorithm step 6a: the trajectory ``T(y_k)``, with its two halt criteria.

    ``flood_depths`` is the reference's ``flddat``: a snapshot of the floodplain
    taken at the top of the spill round and deliberately *not* refreshed while
    the round's candidates are processed.  Reading live depths here would halt
    trajectories a cell or two earlier and change the result.
    """
    current = start
    length = 0
    while True:
        if seen[current] == seen_stamp:
            break
        seen[current] = seen_stamp
        path[length] = current
        length += 1

        code = fdr[current]
        nxt = -1
        if code != 0:
            for pos in range(8):
                if OUTFLOW[pos] == code:
                    nxt = current + offsets[pos]
                    break
        if nxt == -1:
            break
        if excluded[nxt]:
            path[length] = nxt
            length += 1
            break
        if fil[nxt] == bg:
            break
        if flood_depths[nxt] > 0.0 and spill_dtf >= flood_depths[nxt]:
            break
        current = nxt
        spill_dtf += spill_decay
    return length

@njit(cache=True, nogil=True)
def _solve_iterative(fil: np.ndarray, fdr: np.ndarray, ncols: int,
                     sources: np.ndarray, excluded: np.ndarray,
                     dh: float, fldmn: float, fldmx: float, iterative_spill: bool,
                     global_max_wse: float, bg: float, spill_decay: float):
    """Depth-stepped FLDPLN (algorithm steps 1-8).

    Returns ``(cells, dtf, fsp)`` in window-local flat ids.
    """
    n = fil.size
    unreached = np.float32(3.4e38)
    dtf = np.full(n, unreached, dtype=np.float32)
    fsp = np.full(n, np.int32(-1), dtype=np.int32)
    in_flood = np.zeros(n, dtype=np.bool_)
    on_boundary = np.zeros(n, dtype=np.bool_)

    # The reference's ``flddat``.  It tracks ``dtf`` during the backfill phase
    # but is frozen while a spill round processes its candidates, so writes made
    # then are queued in ``dirty`` and folded in at the start of the next round.
    flood_depths = np.zeros(n, dtype=np.float32)
    dirty = np.empty(n, dtype=np.int32)
    dirty_stamp = np.zeros(n, dtype=np.int32)
    n_dirty = 0
    round_id = 0

    offsets = np.empty(8, dtype=np.int64)
    offsets[0] = -ncols - 1
    offsets[1] = -ncols
    offsets[2] = -ncols + 1
    offsets[3] = -1
    offsets[4] = 1
    offsets[5] = ncols - 1
    offsets[6] = ncols
    offsets[7] = ncols + 1

    # Generation stamps stand in for the transient sets of the reference
    # implementation, so nothing has to be cleared between rounds.
    visited = np.zeros(n, dtype=np.int32)
    seen = np.zeros(n, dtype=np.int32)
    on_path = np.zeros(n, dtype=np.int32)
    stamp = 0

    boundary = np.empty(n, dtype=np.int32)
    added = np.empty(n, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)
    fill_pixels = np.empty(n, dtype=np.int32)
    fill_dtf = np.empty(n, dtype=np.float32)
    path = np.empty(n + 2, dtype=np.int32)

    # Spillover candidates, one slot per cell plus a touched list (step 4).
    cand_stamp = np.zeros(n, dtype=np.int32)
    cand_avail = np.empty(n, dtype=np.float32)
    cand_belev = np.empty(n, dtype=np.float32)
    cand_fsp = np.empty(n, dtype=np.int32)
    cand_dtf = np.empty(n, dtype=np.float32)
    touched = np.empty(n, dtype=np.int32)

    n_added = 0
    for i in range(sources.size):
        source = sources[i]
        if fil[source] == bg:
            continue
        dtf[source] = 0.0
        fsp[source] = source
        in_flood[source] = True
        flood_depths[source] = fldmn
        added[n_added] = source
        n_added += 1
    n_boundary = _refresh_boundary(boundary, 0, added, n_added, fil, offsets,
                                   in_flood, on_boundary, bg)

    fldht = 0.0
    iterations = int(np.ceil(fldmx / dh))
    for _ in range(iterations):
        fldht += min(dh, fldmx - fldht)

        # Steps 1-2: backfill from every interior boundary pixel and assimilate.
        order = _order_by_depth_then_elevation(boundary[:n_boundary], dtf, fil)
        n_added = 0
        for i in range(n_boundary):
            pixel = order[i]
            pixel_dtf = dtf[pixel]
            pixel_fsp = fsp[pixel]
            max_wse = min(global_max_wse, fil[pixel] + fldht - pixel_dtf)
            stamp += 1
            count = _backfill_from_source(
                pixel, pixel_dtf, max_wse, fil, fdr, offsets, bg,
                True, in_flood, on_path, 0, stack, visited, stamp,
                fill_pixels, fill_dtf)
            for j in range(count):
                target = fill_pixels[j]
                value = fill_dtf[j]
                if value < dtf[target]:
                    dtf[target] = value
                    fsp[target] = pixel_fsp
                    flood_depths[target] = max(fldmn, value)
                    if not in_flood[target]:
                        in_flood[target] = True
                        added[n_added] = target
                        n_added += 1
        n_boundary = _refresh_boundary(boundary, n_boundary, added, n_added, fil,
                                       offsets, in_flood, on_boundary, bg)

        # Steps 3-7: spillover, optionally repeated until the floodplain settles.
        # The reference filters the boundary to ``dtf < fldht`` at the end of a
        # round and feeds only that to the next one, so from the second round on
        # a boundary pixel that has already used up its head cannot spill again.
        shallow_only = False
        spill = True
        while spill:
            n_new = 0
            n_added = 0

            # Refresh the frozen flood-depth snapshot from the previous round.
            for i in range(n_dirty):
                target = dirty[i]
                flood_depths[target] = max(fldmn, dtf[target])
            n_dirty = 0
            round_id += 1

            # Steps 4-5: best spillover source for each exterior boundary pixel.
            stamp += 1
            n_touched = 0
            for i in range(n_boundary):
                pixel = boundary[i]
                pixel_elev = fil[pixel]
                if pixel_elev == bg:
                    continue
                pixel_dtf = dtf[pixel]
                if shallow_only and pixel_dtf >= fldht:
                    continue
                pixel_fsp = fsp[pixel]
                limit = min(global_max_wse, pixel_elev + fldht - pixel_dtf)
                for pos in range(8):
                    neighbor = pixel + offsets[pos]
                    if in_flood[neighbor] or excluded[neighbor]:
                        continue
                    neighbor_elev = fil[neighbor]
                    if neighbor_elev == bg or neighbor_elev > limit:
                        continue
                    climb = max(0.0, neighbor_elev - pixel_elev)
                    available = fldht - pixel_dtf - climb
                    if cand_stamp[neighbor] != stamp:
                        cand_stamp[neighbor] = stamp
                        touched[n_touched] = neighbor
                        n_touched += 1
                    elif not (available > cand_avail[neighbor] or
                              (available == cand_avail[neighbor] and
                               pixel_elev > cand_belev[neighbor])):
                        continue
                    cand_avail[neighbor] = available
                    cand_belev[neighbor] = pixel_elev
                    cand_fsp[neighbor] = pixel_fsp
                    cand_dtf[neighbor] = pixel_dtf + climb

            # Step 6: flood each spillover trajectory and its backfill watershed,
            # shallowest spill first.
            ordered = _order_by_depth_then_elevation(touched[:n_touched],
                                                     cand_dtf, cand_belev)
            for i in range(n_touched):
                start = ordered[i]
                spill_dtf = cand_dtf[start]
                spill_fsp = cand_fsp[start]
                start_elev = fil[start]
                stamp += 1
                length = _forward_path(start, spill_dtf, fil, fdr, offsets,
                                       flood_depths, bg, excluded, spill_decay,
                                       path, seen, stamp)
                if length == 0:
                    continue
                path_stamp = stamp
                for j in range(length):
                    on_path[path[j]] = path_stamp
                path_depth = min(global_max_wse, start_elev + fldht - spill_dtf) - start_elev

                for j in range(length):
                    source = path[j]
                    if spill_dtf < dtf[source]:
                        dtf[source] = spill_dtf
                        fsp[source] = spill_fsp
                        if dirty_stamp[source] != round_id:
                            dirty_stamp[source] = round_id
                            dirty[n_dirty] = source
                            n_dirty += 1
                        if not in_flood[source]:
                            in_flood[source] = True
                            added[n_added] = source
                            n_added += 1
                            n_new += 1
                for j in range(length):
                    source = path[j]
                    stamp += 1
                    count = _backfill_from_source(
                        source, 0.0, fil[source] + path_depth, fil, fdr, offsets, bg,
                        False, in_flood, on_path, path_stamp, stack, visited, stamp,
                        fill_pixels, fill_dtf)
                    for k in range(count):
                        target = fill_pixels[k]
                        value = fill_dtf[k] + spill_dtf
                        if value < dtf[target]:
                            dtf[target] = value
                            fsp[target] = spill_fsp
                            if dirty_stamp[target] != round_id:
                                dirty_stamp[target] = round_id
                                dirty[n_dirty] = target
                                n_dirty += 1
                            if not in_flood[target]:
                                in_flood[target] = True
                                added[n_added] = target
                                n_added += 1
                                n_new += 1

            n_boundary = _refresh_boundary(boundary, n_boundary, added, n_added, fil,
                                           offsets, in_flood, on_boundary, bg)

            # Step 7: keep spilling while the boundary can still afford to.
            spill = False
            if iterative_spill and n_new > 0:
                for i in range(n_boundary):
                    if dtf[boundary[i]] < fldht:
                        spill = True
                        break
            shallow_only = True

    n_out = 0
    for pixel in range(n):
        if in_flood[pixel]:
            n_out += 1
    cells = np.empty(n_out, dtype=np.int32)
    out_dtf = np.empty(n_out, dtype=np.float32)
    out_fsp = np.empty(n_out, dtype=np.int32)
    k = 0
    for pixel in range(n):
        if in_flood[pixel]:
            cells[k] = pixel
            value = dtf[pixel]
            out_dtf[k] = fldmn if value < fldmn else value
            out_fsp[k] = fsp[pixel]
            k += 1
    return cells, out_dtf, out_fsp


# ---------------------------------------------------------------------------
# Segment driver
# ---------------------------------------------------------------------------

def fldpln_library_for_segment(filled_dem: np.ndarray,
                               flow_direction: np.ndarray,
                               stream_id: int,
                               stream_info: np.ndarray,
                               dh: float,
                               fldmn: float,
                               fldmx: float,
                               iterative_spill: bool,
                               global_max_wse: float = 0.0,
                               bg: float = -9999,
                               spill_decay: float = 0.0,
                               solver: str = "exact",
                               max_drop_below_source: float | None = None,
                               max_cells: int = 0) -> pd.DataFrame:
    """Floodplain table for one stream segment.

    Parameters
    ----------
    filled_dem : np.ndarray
        Filled (depression-free) DEM.
    flow_direction : np.ndarray
        D8 flow directions, WhiteboxTools codes.
    stream_id : int
        The segment to solve.
    stream_info : np.ndarray
        One row per segment: start pixel, end pixel, length (pixels), stream id.
    dh : float
        Depth step of the iterative solver (m); must be positive.  Ignored by
        the exact solver.
    fldmn : float
        Smallest DTF reported (m); smaller DTFs are raised to it.
    fldmx : float
        Largest depth to flood to (m).
    iterative_spill : bool
        Iterative solver only: keep spilling until the floodplain stops growing.
    global_max_wse : float, optional
        Highest water-surface elevation allowed; 0 for no limit.
    bg : float, optional
        No-data value of the DEM; such cells never flood.
    spill_decay : float, optional
        DTF added per pixel along a spillover path, to shorten spills.
    solver : {"exact", "iterative"}, optional
        See the module docstring.
    max_drop_below_source : float, optional
        Leave out cells more than this many metres below the stream pixel that
        floods them, on the filled DEM.  Going downhill costs no depth, so on
        slopes these are most of the table.  None keeps every cell.
    max_cells : int, optional
        Most rows the segment may have (0 for no limit).  A segment that would
        flood more stops at the shallowest depth that exceeds it, recorded in
        ``attrs["depth_limit"]``.  Exact solver only.

    Returns
    -------
    pandas.DataFrame
        ``FSP``, ``FPP`` and ``DTF`` columns.  ``attrs["depth_limit"]`` is the
        depth ``max_cells`` stopped at, or None.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}.")
    if solver == "iterative" and dh <= 0:
        raise ValueError("dh must be positive.")
    if max_cells and solver != "exact":
        raise ValueError("max_cells needs solver='exact': the iterative solver can lower a cell's DTF "
                         "in a later depth step, so its floodplain cannot be cut at a depth.")

    nrows, ncols = filled_dem.shape
    if flow_direction.shape != (nrows, ncols):
        raise ValueError("flow_direction shape must match filled_dem shape.")

    filled_dem = np.ascontiguousarray(filled_dem, dtype=np.float32)
    flow_direction = np.ascontiguousarray(flow_direction, dtype=np.uint8)

    # The 0.99999 is carried over from the MATLAB reference
    global_max_wse = np.finfo(np.float32).max if not global_max_wse else 0.99999 * global_max_wse

    max_below = np.inf if max_drop_below_source is None else float(max_drop_below_source)

    flat_fdr = flow_direction.ravel()
    stream_pixels = np.asarray(
        _segment_pixels(stream_id, stream_info, flat_fdr, nrows, ncols), dtype=np.int64)
    downstream = _downstream_exclusion(stream_id, stream_info, flat_fdr, nrows, ncols)
    downstream = np.fromiter(downstream, dtype=np.int64, count=len(downstream))

    header = ["FSP", "FPP", "DTF"]
    if stream_pixels.size == 0:
        return pd.DataFrame(
            {"FSP": np.empty(0, np.int32), "FPP": np.empty(0, np.int32),
             "DTF": np.empty(0, np.float32)},
            columns=header)

    # Each side of the window has its own margin and grows only when the
    # floodplain reaches it: a floodplain usually runs off one way, and growing
    # every side would solve a window several times its size.  The last margin
    # reaches the raster edge, which can never be touched, so this ends.
    margins = _WINDOW_MARGINS + (max(nrows, ncols),)
    level = [0, 0, 0, 0]
    while True:
        r0, r1, c0, c1 = _window_bounds(stream_pixels, nrows, ncols,
                                        tuple(margins[i] for i in level))
        wr, wc = r1 - r0 + 2, c1 - c0 + 2

        win_fil = _crop(filled_dem, r0, r1, c0, c1, bg).ravel()
        win_fdr = _crop(flow_direction, r0, r1, c0, c1, np.uint8(0)).ravel()
        win_excluded = np.zeros(wr * wc, dtype=np.bool_)
        if downstream.size:
            inside = _in_window(downstream, ncols, r0, r1, c0, c1)
            if inside.any():
                win_excluded[_to_local(downstream[inside], ncols, r0, c0, wc)] = True
        # Sources are seeded regardless of the mask, exactly as the reference
        # does.  They are not un-excluded: when stream_info's end pixel sits
        # upstream of where the length-step FDR walk ends, a segment's own
        # pixels legitimately fall inside TD(R), and the reference lets the
        # exclusion stand.
        sources = _to_local(stream_pixels, ncols, r0, c0, wc)

        depth_limit = -1.0
        if solver == "exact":
            cells, values, source_of, depth_limit = _solve_exact(
                win_fil, win_fdr, wc, sources, win_excluded,
                fldmn, fldmx, global_max_wse, bg, spill_decay, max_cells, max_below)
        else:
            cells, values, source_of = _solve_iterative(
                win_fil, win_fdr, wc, sources, win_excluded,
                dh, fldmn, fldmx, iterative_spill, global_max_wse, bg, spill_decay)

        touched = _touched_edges(cells, wr, wc, r0, r1, c0, c1, nrows, ncols)
        if not any(touched):
            break
        LOG.debug("Stream %s reached its window's %s edge; retrying larger.", stream_id,
                  "/".join(side for side, t in zip(("top", "bottom", "left", "right"), touched) if t))
        level = [min(i + t, len(margins) - 1) for i, t in zip(level, touched)]
        # Free this window before cropping the next one
        del win_fil, win_fdr, win_excluded, cells, values, source_of

    # Left out only now, so that cells reached through them still grow the window
    if max_drop_below_source is not None:
        kept = win_fil[source_of] - win_fil[cells] <= max_below
        cells, values, source_of = cells[kept], values[kept], source_of[kept]

    fpp = _to_global(cells, ncols, r0, c0, wc)
    fsp = _to_global(source_of, ncols, r0, c0, wc)
    table = pd.DataFrame({"FSP": fsp, "FPP": fpp, "DTF": values}, columns=header)
    table.attrs["depth_limit"] = float(depth_limit) if depth_limit >= 0 else None
    return table


def _set_shared(name: str, shm: shared_memory.SharedMemory):
    """Keep a reference to a shared-memory block, so it is not freed while arrays still use it."""
    _SHARED_MEMORYS[name] = shm

def _read_into(ds: gdal.Dataset, arr: np.ndarray):
    """``arr[:] = ds.ReadAsArray()``, a strip of rows at a time.

    Reading the whole raster at once holds a second copy in the file's own data
    type (float64 for some filled DEMs) and keeps every block in GDAL's cache
    until the file is closed; a strip at a time holds only a strip of each.
    """
    band = ds.GetRasterBand(1)
    block_rows = band.GetBlockSize()[1]
    step = block_rows * max(1, 512 // block_rows)
    for r in range(0, arr.shape[0], step):
        n = min(step, arr.shape[0] - r)
        arr[r:r + n] = band.ReadAsArray(0, r, arr.shape[1], n)
        band.FlushCache()

def read_array_and_set_shared(file: str, dtype: np.dtype, set_shared: bool,
                              name: str = None):
    """Read band 1 of a raster as ``dtype``; with ``set_shared``, into shared memory called ``name``."""
    ds = gdal.Open(file)
    shape = (ds.RasterYSize, ds.RasterXSize)

    dtype = np.dtype(dtype)
    if not set_shared:
        arr = np.empty(
            shape, 
            dtype=dtype
        )
        _read_into(ds, arr)
        return arr
    
    size = int(dtype.itemsize * np.prod(shape))
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    _read_into(ds, arr)
    globals()[name] = arr
    _set_shared(name, shm)
    return arr

# ---------------------------------------------------------------------------
# Reaches into segments
#
# A library is exact only for one depth per segment (every source starts at
# the same depth), and a single depth fits a long reach badly, so reaches over
# a length are cut into shorter segments, solved and mapped on their own.
# ---------------------------------------------------------------------------

def _cell_size_metres(raster: str) -> tuple[float, float]:
    """The raster's cell size ``(dx, dy)`` in metres, as Curve2Flood measures the DEM's."""
    from curve2flood.core import convert_cell_size  # core imports this module

    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    yur = gt[3]
    yll = yur - ds.RasterYSize * abs(gt[5])
    size_x, size_y = np.hypot(gt[1], gt[2]), np.hypot(gt[4], gt[5])
    if not projection:
        # No coordinate system: take the geotransform's units as metres
        return float(size_x), float(size_y)
    dx, dy = convert_cell_size(size_x, size_y, yll, yur, projection)
    return float(dx), float(dy)

def _step_lengths(pixels: np.ndarray, flat_fdr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Length (m) of the D8 step each pixel takes downstream; about one cell where it has none."""
    codes = flat_fdr[pixels]
    steps = np.full(len(pixels), (dx + dy) / 2)
    steps[np.isin(codes, (2, 32))] = dx                     # east, west
    steps[np.isin(codes, (8, 128))] = dy                    # south, north
    steps[np.isin(codes, (1, 4, 16, 64))] = np.hypot(dx, dy)
    return steps

def _profile_cuts(distance: np.ndarray, elevation: np.ndarray, max_deviation: float) -> np.ndarray:
    """Where to cut a reach so that the filled DEM under it is about straight along each piece.

    ``distance`` is each stream pixel's distance along the river and ``elevation`` its filled DEM, from upstream
    down. The profile is simplified top-down (Douglas-Peucker): a chord is split at its point farthest above or below
    it, until no point departs from its chord by more than ``max_deviation`` (m). Each pixel then belongs to the chord
    it lies inside; a vertex lies on two chords and joins the gentler one, so the top and the foot of a step stay with
    their pools and a step from one pixel to the next is a cut between the two. Returns the index of the first pixel
    of every piece but the first.
    """
    n = len(elevation)
    if n < 3:
        return np.zeros(0, dtype=np.int64)
    keep = np.zeros(n, dtype=bool)
    keep[[0, n - 1]] = True
    chords = [(0, n - 1)]
    while chords:
        a, b = chords.pop()
        if b - a < 2:
            continue
        along = (distance[a + 1:b] - distance[a]) / (distance[b] - distance[a])
        off = np.abs(elevation[a + 1:b] - elevation[a] - (elevation[b] - elevation[a]) * along)
        k = int(np.argmax(off))
        if off[k] > max_deviation:
            k += a + 1
            keep[k] = True
            chords += [(a, k), (k, b)]
    vertices = np.flatnonzero(keep)
    if len(vertices) <= 2:
        return np.zeros(0, dtype=np.int64)
    slope = np.abs(np.diff(elevation[vertices]) / np.diff(distance[vertices]))
    chord_of = np.r_[np.repeat(np.arange(len(vertices) - 1), np.diff(vertices)), len(vertices) - 2]
    inner = np.arange(1, len(vertices) - 1)
    chord_of[vertices[inner]] = np.where(slope[inner - 1] <= slope[inner], inner - 1, inner)
    return np.flatnonzero(np.diff(chord_of)) + 1

def _equal_lengths(pixels: np.ndarray, steps: np.ndarray, max_length: float | None) -> list[np.ndarray]:
    """Cut a run of stream pixels longer than ``max_length`` into ``ceil(length / max_length)`` of about equal
    length along the river."""
    total = float(steps.sum())
    if max_length is None or total <= max_length or len(pixels) < 2:
        return [pixels]
    n = min(math.ceil(total / max_length), len(pixels))
    upstream = np.cumsum(steps) - steps  # river distance from the run's start to each pixel
    piece = np.minimum((upstream / (total / n)).astype(np.int64), n - 1)
    return [pixels[piece == k] for k in range(n) if (piece == k).any()]

def _split_reach(row: np.ndarray, pixels: np.ndarray, steps: np.ndarray, max_length: float | None,
                 elevation: np.ndarray | None = None,
                 max_deviation: float | None = None) -> list[tuple[int, int, int, np.ndarray]]:
    """Cut a reach into segments along the river.

    ``row`` is the reach's stream table row, ``pixels`` its stream pixels from upstream down, ``steps`` their step
    lengths and ``elevation`` their filled DEM. With ``max_deviation``, the reach is first cut where its profile bends
    or steps (see ``_profile_cuts``); then every piece longer than ``max_length`` is cut into
    ``ceil(length / max_length)`` of about equal length. Returns ``(start_pixel, end_pixel, length, pixels)`` for
    each segment, from upstream down. A reach left whole keeps its own row; the last segment of a split reach keeps
    the reach's ``end_pixel``, and every other one ends at its own last pixel.
    """
    start, end, length = int(row[0]), int(row[1]), int(row[2])
    cuts = []
    if max_deviation is not None:
        cuts = _profile_cuts(np.cumsum(steps) - steps, elevation, max_deviation).tolist()
    bounds = [0, *cuts, len(pixels)]
    parts = [p for a, b in zip(bounds[:-1], bounds[1:]) for p in _equal_lengths(pixels[a:b], steps[a:b], max_length)]
    if len(parts) == 1:
        return [(start, end, length, pixels)]
    return [(int(p[0]), end if i == len(parts) - 1 else int(p[-1]), len(p), p) for i, p in enumerate(parts)]

def _segments(stream_info: np.ndarray, stream_ids: np.ndarray, max_depths, flat_fdr: np.ndarray,
              nrows: int, ncols: int, max_length: float | None, cell_size: tuple[float, float] | None,
              vdt_depth: dict[int, float], fldmx: float, flat_fil: np.ndarray | None = None,
              max_deviation: float | None = None, bg: float = -9999) -> tuple[np.ndarray, list[float], int]:
    """The segments to solve: ``(start_pixel, end_pixel, length, segment)`` rows numbered from 0, each one's depth
    cap, and how many reaches were split.

    A segment cut from a reach floods as deep as the deepest VDT water among its own pixels (at most ``fldmx``), or
    as deep as its reach when it has no VDT cell, so that every stream pixel stays a source. ``flat_fil`` is the
    filled DEM, for ``max_deviation``; stream pixels without a value there are left out of the profile.
    """
    rows, caps = [], []
    n_split = 0
    by_id = {int(r[3]): r for r in stream_info}
    for stream_id, cap in zip(stream_ids, max_depths):
        row = by_id[int(stream_id)]
        if max_length is None and max_deviation is None:
            pieces = [(int(row[0]), int(row[1]), int(row[2]), None)]
        else:
            pixels = np.asarray(_segment_pixels(int(stream_id), stream_info, flat_fdr, nrows, ncols), dtype=np.int64)
            steps = _step_lengths(pixels, flat_fdr, *cell_size)
            elevation = None
            if max_deviation is not None:
                elevation = flat_fil[pixels].astype(np.float64)
                has_value = elevation != bg
                if not has_value.all():
                    distance = np.cumsum(steps) - steps
                    elevation = (np.interp(distance, distance[has_value], elevation[has_value])
                                 if has_value.sum() > 1 else np.zeros(len(pixels)))
            pieces = _split_reach(row, pixels, steps, max_length, elevation, max_deviation)
        n_split += len(pieces) > 1
        for start, end, length, pixels in pieces:
            piece_cap = cap
            if len(pieces) > 1:
                depths = [vdt_depth[p] for p in pixels.tolist() if p in vdt_depth]
                piece_cap = min(max(depths), fldmx) if depths else cap
            rows.append((start, end, length, len(rows)))
            caps.append(piece_cap)
    return np.array(rows, dtype=np.int64).reshape(-1, 4), caps, n_split

def _solve_segment(filled_dem: np.ndarray, flow_direction: np.ndarray, args) -> pd.DataFrame:
    """One segment's rows, with its ``segment`` number (``args`` as fldpln_library_for_segment takes them)."""
    table = fldpln_library_for_segment(filled_dem, flow_direction, *args)
    table["segment"] = np.int32(args[0])
    return table

def run_parallel(args):
    """Worker: solve one segment on the shared rasters."""
    return _solve_segment(globals()['filled_dem_array'], globals()['flow_direction_array'], args)

def close_shared_memory(names: list[str]):
    """Close and unlink the named shared-memory blocks."""
    for name in names:
        shm = _SHARED_MEMORYS.get(name)
        if shm is not None:
            shm.close()
            shm.unlink()
            del globals()[name]
            del _SHARED_MEMORYS[name]

def init_parallel(
    names: list[str],
    shapes: list[tuple],
    dtypes: list[np.dtype],
):
    """Worker initialiser: attach the shared rasters as module globals, so they are not pickled per task."""
    shms = [shared_memory.SharedMemory(name=name) for name in names]

    for shm, name, shape, dtype in zip(shms, names, shapes, dtypes):
        _set_shared(name, shm)
        globals()[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

def build_fldpln_library(
    dem: str,
    filled_dem: str,
    stream_info_file: str,
    flow_direction_file: str,
    library_file: str,
    dh: float,
    fldmn: float,
    fldmx: float,
    iterative_spill: bool,
    vdt_file: str = None,
    bg_mask: np.ndarray | None = None,
    stream_ids: list[int] | None = None,
    global_max_wse: float = 0.0,
    bg: float = -9999,
    parallel: bool = False,
    pbar: bool = True,
    processes: int | None = None,
    spill_decay: float = 0.0,
    solver: str = "exact",
    max_drop_below_source: float | None = None,
    max_cells_per_segment: int | None = None,
    max_reach_length: float | None = 5000.0,
    max_profile_deviation: float | None = 1.0
):
    """Build the FLDPLN floodplain library for every stream segment and save it.

    Each row is a stream pixel ``FSP``, a cell it floods ``FPP``, the depth
    to flood ``DTF``, the ``segment`` the stream pixel belongs to and the
    cell's ``fill``: the filled DEM minus ``dem``.  Cells where ``dem`` has no
    data never flood, and are left out.

    Every source of a segment starts at the same depth, so each cell keeps the
    one stream pixel that reaches it at the smallest depth: the library is
    exact for one depth per segment, which is how the mapper floods it.
    Reaches are first cut where the filled DEM under them bends or steps (with
    ``max_profile_deviation``) and into pieces no longer than
    ``max_reach_length``, each solved and flooded on its own.

    Parameters
    ----------
    dem: str
        The DEM floods are mapped on (with the bathymetry, if any).  Only the
        ``fill`` column comes from it, so the mapper never reads it.
    filled_dem: str
        Filled (depression-free) DEM raster.  The mapper must use the same one.
    stream_info_file: str
        CSV or Parquet table of the segments, with columns ``start_pixel``,
        ``end_pixel``, ``length`` (pixels) and ``stream_id``.
    flow_direction_file: str
        D8 flow-direction raster with WhiteboxTools codes.
    library_file: str
        Where to save the library: Parquet (``.parquet``/``.pq``, compact) or CSV.
    dh: float
        Depth step of the iterative solver (m).  Ignored by the exact solver.
    fldmn: float
        Smallest DTF stored (m); smaller DTFs are raised to it.
    fldmx: float
        Largest depth to flood to (m).
    iterative_spill: bool
        Iterative solver only: keep spilling until the floodplain stops growing.
        Recommended for fine DEMs (10 m or less).
    vdt_file: str | None
        A VDT database.  Each segment then floods only to the deepest water its
        last ``wse_*`` column gives (at most ``fldmx``), and segments without
        VDT cells are skipped.
    bg_mask: np.ndarray | None
        Cells that never flood, such as lakes; same shape as the DEM.
    stream_ids: list[int] | None
        Only build these segments.
    global_max_wse: float
        Highest water-surface elevation allowed; 0 for no limit.
    bg: float
        No-data value of the filled DEM.
    parallel: bool
        Solve the segments in several processes.
    pbar: bool
        Show a progress bar.
    processes: int | None
        Number of processes when ``parallel``; defaults to the CPU count.
    spill_decay: float
        DTF added per pixel along a spillover path, to shorten spills: only on
        each pixel's D8 successor, so other downhill steps stay free and it
        trims little (see the FLDPLN docs).  0 (default) for none.
    solver: str
        ``"exact"`` (default) or ``"iterative"``; see the module docstring.
    max_drop_below_source: float | None
        Leave out cells more than this many metres below the stream pixel that
        floods them, on the filled DEM, whatever the depth.  Going downhill
        costs no depth, so without this a stream pixel on a slope floods the
        whole hillside below it, as deep as the water at the stream; the mapper
        has no limit of its own.  On slopes such cells are most of the library.
        None (default) keeps every cell.
    max_cells_per_segment: int | None
        Most rows one segment may contribute.  A segment that would flood more
        stops at the depth where it fills up, so deeper floods there are mapped
        only that far: this trades the largest floods on flat ground for memory.
        None (default) for no limit.  Needs ``solver="exact"``.
    max_reach_length: float | None
        Longest segment (m), measured along the flow directions with the
        filled DEM's cell size.  A longer reach is cut into
        ``ceil(length / max_reach_length)`` segments of about equal length.
        5000 (default), as the University of Kansas FLDPLN keeps them; None
        keeps each reach whole.
    max_profile_deviation: float | None
        Also cut each reach where the filled DEM under the stream bends or
        steps, so that the bed is about straight along every segment: its
        profile along the river is simplified top-down (Douglas-Peucker)
        until no stream pixel is more than this many metres above or below
        its piece's straight line.  The pieces are then cut by
        ``max_reach_length``.  One depth per segment fits a straight bed
        best: the stage jumps at a step and where a filled flat begins.
        1 m (default) did best on 68 US test sites; None makes no such cuts.
    """
    try:
        _build_fldpln_library(
            dem, filled_dem, stream_info_file, flow_direction_file, library_file,
            dh, fldmn, fldmx, iterative_spill, vdt_file, stream_ids,
            global_max_wse, bg, parallel, pbar, processes, bg_mask,
            spill_decay, solver, max_drop_below_source, max_cells_per_segment, max_reach_length,
            max_profile_deviation
        )
    finally:
        close_shared_memory(['filled_dem_array', 'flow_direction_array'])

def _build_fldpln_library(
    dem: str,
    filled_dem: str,
    stream_info_file: str,
    flow_direction_file: str,
    library_file: str,
    dh: float,
    fldmn: float,
    fldmx: float,
    iterative_spill: bool,
    vdt_file: str = None,
    stream_ids: list[int] | None = None,
    global_max_wse: float = 0.0,
    bg: float = -9999,
    parallel: bool = False,
    pbar: bool = True,
    processes: int | None = None,
    bg_mask: np.ndarray | None = None,
    spill_decay: float = 0.0,
    solver: str = "exact",
    max_drop_below_source: float | None = None,
    max_cells_per_segment: int | None = None,
    max_reach_length: float | None = 5000.0,
    max_profile_deviation: float | None = 1.0
):
    """``build_fldpln_library`` without the shared-memory cleanup."""
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}.")
    if max_drop_below_source is not None and not max_drop_below_source >= 0:
        raise ValueError(f"max_drop_below_source must be at least 0 or None, got {max_drop_below_source!r}.")
    if max_cells_per_segment is not None and (int(max_cells_per_segment) != max_cells_per_segment or max_cells_per_segment < 1):
        raise ValueError(f"max_cells_per_segment must be a positive whole number or None, got {max_cells_per_segment!r}.")
    if max_cells_per_segment is not None and solver != "exact":
        raise ValueError("max_cells_per_segment needs solver='exact'.")
    if max_reach_length is not None and not max_reach_length > 0:
        raise ValueError(f"max_reach_length must be positive or None, got {max_reach_length!r}.")
    if max_profile_deviation is not None and not max_profile_deviation > 0:
        raise ValueError(f"max_profile_deviation must be positive or None, got {max_profile_deviation!r}.")
    split = max_reach_length is not None or max_profile_deviation is not None
    max_cells = int(max_cells_per_segment or 0)

    if Path(stream_info_file).suffix in {'.parquet', '.pq'}:
        stream_info = pd.read_parquet(stream_info_file)
    else:
        stream_info = pd.read_csv(stream_info_file)
    assert stream_info.ndim == 2, "Stream info file must be a 2D table"
    assert stream_info.shape[1] == 4, "Stream info file must have 4 columns: start pixel (0), end pixel (1), length (2), and stream ID (3)"

    if stream_ids is not None:
        stream_info = stream_info[stream_info.iloc[:, 3].isin(stream_ids)]

    filled_dem_array = read_array_and_set_shared(filled_dem, np.float32, set_shared=parallel, name='filled_dem_array')
    flow_direction_array = read_array_and_set_shared(flow_direction_file, np.uint8, set_shared=parallel, name='flow_direction_array')
    dem_shape = _raster_shape(dem)
    if dem_shape != filled_dem_array.shape:
        raise ValueError(f"The DEM has shape {dem_shape}, but the filled DEM has {filled_dem_array.shape}.")

    if bg_mask is not None:
        # A 0/1 integer mask would index rows 0 and 1 instead of the masked cells
        bg_mask = np.asarray(bg_mask, dtype=bool)
        if bg_mask.shape != filled_dem_array.shape:
            raise ValueError(f"bg_mask has shape {bg_mask.shape}, but the filled DEM has {filled_dem_array.shape}.")

    stream_ids = stream_info.iloc[:, 3].unique()
    vdt_depth: dict[int, float] = {}
    if vdt_file is None:
        max_depths = [fldmx] * len(stream_ids)
    else:
        # Read only the columns used: the VDT carries ~120 more
        is_parquet = Path(vdt_file).suffix in {'.parquet', '.pq'}
        if is_parquet:
            vdt_columns = pq.read_schema(vdt_file).names
        else:
            vdt_columns = pd.read_csv(vdt_file, nrows=0).columns
        # Find last wse_* column
        wse_cols = [col for col in vdt_columns if col.startswith('wse_')]
        assert wse_cols, "No wse_* columns found in VDT file"
        max_wse_col = wse_cols[-1]
        usecols = ['COMID', 'Row', 'Col', max_wse_col]
        if is_parquet:
            vdt_df = pd.read_parquet(vdt_file, columns=usecols)
        else:
            vdt_df = pd.read_csv(vdt_file, usecols=usecols)

        vdt_df = vdt_df[vdt_df['COMID'].isin(stream_ids)]
        # making a copy to resolve a fragmentation warning from pandas
        vdt_df = vdt_df[usecols].copy()
        elevation = filled_dem_array[vdt_df['Row'], vdt_df['Col']]
        vdt_df['depth'] = vdt_df[max_wse_col] - elevation
        # Background cells, like masked lakes, are never flooded from, and their depth would be measured from bg
        background = elevation == bg
        if bg_mask is not None:
            background |= bg_mask[vdt_df['Row'], vdt_df['Col']]
        vdt_df = vdt_df[~background]
        ids_max_depths = vdt_df.groupby('COMID', sort=False, as_index=False)['depth'].max().values
        stream_ids = ids_max_depths[:, 0].astype(np.int32)
        max_depths = np.minimum(ids_max_depths[:, 1], fldmx)
        if split:
            # Each VDT cell's deepest water, for the segments of split reaches
            vdt_pixel = vdt_df['Row'].to_numpy(np.int64) * filled_dem_array.shape[1] + vdt_df['Col'].to_numpy(np.int64)
            vdt_depth = pd.Series(vdt_df['depth'].to_numpy(), index=vdt_pixel).groupby(level=0).max().to_dict()

    if len(stream_ids) == 0:
        LOG.warning("No stream segments found to process. Exiting.")
        return

    if pbar:
        pbar = tqdm.tqdm
    else:
        pbar = lambda x, **kwargs: x

    nrows, ncols = filled_dem_array.shape
    cell_size = _cell_size_metres(filled_dem) if split else None
    # The profiles are read before bg_mask is applied: a lake on the stream is a flat, not a gap
    segments, segment_depths, n_split = _segments(
        stream_info.values.astype(np.int64), stream_ids, max_depths, flow_direction_array.ravel(), nrows, ncols,
        max_reach_length, cell_size, vdt_depth, fldmx, filled_dem_array.ravel(), max_profile_deviation, bg)
    if n_split:
        why = []
        if max_profile_deviation is not None:
            why.append(f"where the filled DEM under them departs from a straight line by more than "
                       f"{max_profile_deviation:g} m")
        if max_reach_length is not None:
            why.append(f"into pieces of at most {max_reach_length / 1000:g} km")
        LOG.info(f"Cut {n_split} of {len(stream_ids)} reaches {' and '.join(why)}: {len(segments)} segments in all.")
    if bg_mask is not None:
        filled_dem_array[bg_mask] = bg
    # One segment table for every task: building it inside the comprehension
    # made a fresh copy per segment, and every copy was pickled to a worker.
    args = [
        (
            int(segment),
            segments,
            dh,
            fldmn,
            segment_depths[i],
            iterative_spill,
            global_max_wse,
            bg,
            spill_decay,
            solver,
            max_drop_below_source,
            max_cells
        )
        for i, segment in enumerate(segments[:, 3])
    ]
    # Sort args from largest max depth to smallest, (longest time to shortest)
    args.sort(key=lambda x: x[4], reverse=True)

    if processes is None:
        processes = max(min(mp.cpu_count(), len(args)), 1)
        if processes == 1:
            parallel = False

    dfs: list[pd.DataFrame] = []
    if parallel:
        names = ['filled_dem_array', 'flow_direction_array']
        shapes = [filled_dem_array.shape, flow_direction_array.shape]
        dtypes = [filled_dem_array.dtype, flow_direction_array.dtype]
        with mp.Pool(processes, init_parallel, (names, shapes, dtypes)) as pool:
            for df in pbar(pool.imap_unordered(run_parallel, args), total=len(args), desc="Processing streams in parallel"):
                dfs.append(df)
    else:
        for task in pbar(args, desc="Processing streams"):
            dfs.append(_solve_segment(filled_dem_array, flow_direction_array, task))

    fill_depth = _fill_depth(dem, filled_dem_array, bg)
    close_shared_memory(['filled_dem_array', 'flow_direction_array'])
    # Not needed past the solves: free them before the library is assembled
    del filled_dem_array, flow_direction_array

    if max_cells:
        limits = [d.attrs["depth_limit"] for d in dfs if d.attrs.get("depth_limit") is not None]
        if limits:
            LOG.info(f"{len(limits)} of {len(dfs)} stream segments reached max_cells_per_segment={max_cells:,}; "
                     f"their floodplains stop at depths of {min(limits):.2f}-{max(limits):.2f} m "
                     f"(median {np.median(limits):.2f} m).")

    df = pd.concat(dfs, ignore_index=True)
    dfs = None
    # Sorting like this allows very nice compression.  sort_values(['FSP', 'FPP'],
    # kind='stable') ends in this same np.lexsort, after first factorising both
    # keys into copies.
    order = np.lexsort((df['FPP'].to_numpy(), df['FSP'].to_numpy()))
    df = pd.DataFrame({c: df[c].to_numpy()[order] for c in df.columns}, copy=False)
    del order

    fill = fill_depth.ravel()[df['FPP'].to_numpy()]
    del fill_depth
    # Cells without a DEM value never flood
    has_dem = np.isfinite(fill)
    if not has_dem.all():
        df = pd.DataFrame({c: df[c].to_numpy()[has_dem] for c in df.columns}, copy=False)
        fill = fill[has_dem]
    df['fill'] = fill

    if Path(library_file).suffix in {'.parquet', '.pq'}:
        save_library(df, library_file)
    else:
        df.to_csv(library_file, index=False)

def _raster_shape(path: str) -> tuple[int, int]:
    ds = gdal.Open(path)
    return ds.RasterYSize, ds.RasterXSize

def _fill_depth(dem: str, filled_dem: np.ndarray, bg: float) -> np.ndarray:
    """``filled_dem - dem`` as float32, NaN where either has no data."""
    ds = gdal.Open(dem)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    fill = np.empty(filled_dem.shape, dtype=np.float32)
    _read_into(ds, fill)
    no_data = ~np.isfinite(fill) | (fill <= -9998) | (filled_dem == np.float32(bg))
    if nodata is not None:
        no_data |= fill == np.float32(nodata)
    np.subtract(filled_dem, fill, out=fill)
    fill[no_data] = np.nan
    return fill

def save_library(df: pd.DataFrame, path: str):
    """Save a library as compact Parquet.

    DTF and fill are stored as 2-decimal numbers, and each column gets the
    encoding that compresses it best; the file is several times smaller than
    plain Brotli-compressed Parquet.
    """
    cols, enc = {}, {"FPP": "DELTA_BINARY_PACKED"}
    for c in df.columns:
        if c in DECIMAL_COLS:
            cols[c] = pc.cast(pa.array(df[c].to_numpy().astype(np.float32).round(2)), DECIMAL_COLS[c])
            enc[c] = "BYTE_STREAM_SPLIT"
        else:
            cols[c] = pa.array(df[c].to_numpy())

    pq.write_table(pa.table(cols), path, use_dictionary=[c for c in ("FSP", "segment") if c in df.columns],
                   column_encoding=enc, store_decimal_as_integer=True, compression="brotli")

def _fill_missing_profile(valid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Spread a profile known at the ``valid`` pixels to the whole stream path.

    ``values`` holds the profile at the valid pixels only.  A gap between two
    valid pixels is interpolated linearly, pixel by pixel; before the first and
    after the last valid pixel the profile holds its end value.
    """
    pixel = np.arange(len(valid))
    return np.interp(pixel, pixel[valid], np.asarray(values, dtype=float))

def longest_path_decomposition(G: nx.DiGraph, stream_wse_dict: dict[int, list[tuple[int, float]]]) -> list[list[int]]:
    """Split a stream network into chains: the longest headwater-to-outlet path, then the longest of what is left, and so on.

    Path length counts the stream pixels in ``stream_wse_dict``.  Returns lists
    of stream ids ordered upstream to downstream.  Empties ``G``.
    """
    paths = []
    length = {sid: len(stream_wse_dict[sid]) for sid in G}

    while G.nodes:
        topo = list(nx.topological_sort(G))

        longest = {}

        for node in reversed(topo):
            children = list(G.successors(node))

            if not children:
                longest[node] = (length[node], [node])
            else:
                best = max(
                    (longest[c] for c in children),
                    key=lambda x: x[0]
                )
                longest[node] = (length[node] + best[0], [node] + best[1])

        # Find longest path beginning at a headwater
        headwaters = [n for n in G if G.in_degree(n) == 0]

        _, path = max(
            (longest[h] for h in headwaters),
            key=lambda x: x[0]
        )

        paths.append(path)

        G.remove_nodes_from(path)

    return paths

def _weighted_median_by_group(group: np.ndarray, value: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each group's weighted median: the smallest value whose cumulative weight reaches half the group's total.

    Returns the groups, sorted, and their medians.
    """
    order = np.lexsort((value, group))
    group, value, weight = group[order], value[order], weight[order].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    sizes = np.diff(np.r_[starts, len(group)])
    cumulative = np.cumsum(weight)
    within = cumulative - np.repeat(cumulative[starts] - weight[starts], sizes)
    half = np.repeat(np.add.reduceat(weight, starts) / 2, sizes)
    first = np.minimum.reduceat(np.where(within >= half, np.arange(len(group)), len(group)), starts)
    return group[starts], value[first]

def _make_fldpln_flood_map(
        filled_dem: np.ndarray,
        vdt_df: pl.DataFrame,
        fdr: np.ndarray,
        fldpln_library: pl.LazyFrame,
        stream_info_df: pd.DataFrame,
        stream_gdf: gpd.GeoDataFrame,
        reach_id_field: str,
        downstream_reach_id_field: str) -> np.ndarray:
    """Flood depth raster for one event, from VDT stages and a floodplain library.

    1. Stream pixels are chained along the network's longest paths.  Along
       each chain, the stage of stream pixel ``s`` (its water depth over the
       filled DEM) is the VDT's water surface minus the filled DEM.  Pixels
       without a VDT value take the water surface interpolated linearly
       between the VDT cells on either side.
    2. Each segment ``R`` floods at one depth ``DoF(R)``, as the library
       assumes: the median of its stream pixels' stages, each pixel weighted
       by the library rows it is the source of, so by the floodplain cells it
       floods.  That is the single depth closest, summed over those cells, to
       the stage at each cell's own source.  A library without a ``segment``
       column (built before reaches were split) floods each reach at one
       depth.
    3. Each library row ``(s, p, DTF, fill)`` whose water reaches ``p``'s
       filled surface (``DoF(R) - DTF(s, p) > 0``, ``R`` being ``s``'s
       segment) gives ``p`` the depth ``DoF(R) - DTF(s, p) + fill(p)``: the
       water above the filled DEM, plus how far the filled DEM sits above the
       DEM there.  Water that doesn't reach a filled pit's rim never spills
       into it, however deep the pit.  ``p`` keeps the deepest.

    Returns the depth where it is above 0, NaN elsewhere.  Only the stream
    pixels of ``filled_dem`` are used, and no DEM is needed.  ``vdt_df`` needs
    ``COMID``, ``Row``, ``Col`` and ``WSE``; ``stream_gdf`` needs the two id
    columns and ``topological_order``.
    """
    columns = fldpln_library.collect_schema().names()
    if 'fill' not in columns:
        raise ValueError("The FLDPLN library has no fill column. Rebuild it with build_fldpln_library, "
                         "which now takes the DEM the floods are mapped on.")
    nrows, ncols = filled_dem.shape

    vdt_df = vdt_df.with_columns(FSP=(pl.col('Row') * ncols + pl.col('Col')).cast(pl.Int32))
    fsp_wse_dict = dict(zip(vdt_df['FSP'], vdt_df['WSE']))
    stream_ids = set(vdt_df['COMID'])

    stream_gdf = stream_gdf.sort_values('topological_order')
    G = nx.from_pandas_edgelist(
        stream_gdf,
        source=reach_id_field,
        target=downstream_reach_id_field,
        create_using=nx.DiGraph
    )
    if -1 in G:
        G.remove_node(-1)  # Remove the dummy downstream node

    fdr = fdr.ravel()
    stream_rows: dict[list[tuple]] = defaultdict(list)
    stream_info_dict = stream_info_df.set_index('stream_id').to_dict(orient='index')

    # Walk each segment's stream pixels, with the VDT's WSE where it has one
    for stream_id in stream_ids:
        if stream_id not in stream_info_dict:
            raise ValueError(f"Stream ID {stream_id} not found in stream_info_df.")

        fsp = stream_info_dict[stream_id]['start_pixel']
        length = stream_info_dict[stream_id]['length']

        for _ in range(length):
            if fsp in fsp_wse_dict:
                raw_wse = fsp_wse_dict[fsp]
                stream_rows[stream_id].append((fsp, raw_wse))
            else:
                stream_rows[stream_id].append((fsp, np.nan))

            fsp = _next_downstream(fsp, fdr, nrows, ncols)
            if fsp == -1:
                break

    # C order: the library's flat pixel ids index it
    depth_array = np.full((nrows, ncols), np.nan, dtype=np.float32)
    if not stream_rows:
        return depth_array

    # Chain the segments along the main stems
    paths = longest_path_decomposition(G, stream_rows)

    row_chunks = []
    for path in paths:
        path_rows = []
        for stream_id in path:
            path_rows.extend(stream_rows[stream_id])
    
        if not path_rows:
            continue

        Y = np.array([wse for _, wse in path_rows])
        mask = ~np.isnan(Y)
        if mask.sum() == 0:
            continue
        filled_dem_profile = np.array([filled_dem[_pixel_to_rc(pixel, ncols)]
                                       for pixel, _ in path_rows])

        # The water surface, interpolated across the pixels without a VDT
        # value, over the filled DEM
        dof_profile = _fill_missing_profile(mask, Y[mask]) - filled_dem_profile

        row_chunks.extend([
            (fsp, dof)
            for (fsp, _), dof in zip(path_rows, dof_profile)
            if np.isfinite(dof)
        ])

    if not row_chunks:
        return depth_array

    # A pixel on two chains keeps its deeper stage
    stage = (pl.DataFrame(row_chunks, schema={'FSP': pl.Int32, 'DoF': pl.Float32}, orient='row')
             .group_by('FSP').agg(pl.col('DoF').max()))

    # Each segment's stream pixels, with the library rows each is the source of,
    # counted a slice at a time like the join below
    n_library = fldpln_library.select(pl.len()).collect().item()
    if n_library == 0:
        return depth_array
    key = 'segment' if 'segment' in columns else 'FSP'
    group = ['segment', 'FSP'] if key == 'segment' else ['FSP']
    votes = pl.concat([
        fldpln_library.slice(offset, _LIBRARY_SLICE_ROWS).group_by(group).agg(pl.len().alias('rows')).collect()
        for offset in range(0, n_library, _LIBRARY_SLICE_ROWS)
    ]).group_by(group).agg(pl.col('rows').sum())
    if key == 'FSP':
        # A library built before reaches were split into segments: each reach is one
        reach_of = pl.DataFrame([(fsp, stream_id) for stream_id, rows in stream_rows.items() for fsp, _ in rows],
                                schema={'FSP': pl.Int32, 'segment': pl.Int64}, orient='row').unique('FSP')
        votes = votes.join(reach_of, on='FSP', how='inner')
    votes = votes.join(stage, on='FSP', how='inner')
    if votes.is_empty():
        return depth_array
    segments, depths = _weighted_median_by_group(votes['segment'].to_numpy(), votes['DoF'].to_numpy(),
                                                 votes['rows'].to_numpy())
    df = pl.DataFrame({'segment': segments, 'DoF': depths.astype(np.float32)},
                      schema_overrides={'segment': votes.schema['segment']})
    if key == 'FSP':
        df = reach_of.join(df, on='segment', how='inner').select('FSP', 'DoF')
    df = df.lazy()

    # Join the library a slice at a time and keep only each pixel's deepest water
    # so far, in ``depth_array`` itself: joined whole, the library and its join
    # were several times the size of every raster in the run.
    best = depth_array.ravel()
    best[:] = -np.inf
    for offset in range(0, n_library, _LIBRARY_SLICE_ROWS):
        part = (fldpln_library.slice(offset, _LIBRARY_SLICE_ROWS)
                .join(df, on=key, how='inner')
                .select('FPP', 'DTF', 'fill', 'DoF')
                .collect())
        # ``DoF - DTF`` is the water above the filled DEM.  Only where it is
        # positive does the water reach the cell, and then the fill adds the
        # depth below the filled surface, as in a filled pit or a burned channel.
        head = part['DoF'].to_numpy() - part['DTF'].to_numpy()
        reaches = head > 0
        depth = head[reaches] + part['fill'].to_numpy()[reaches]
        np.maximum.at(best, part['FPP'].to_numpy()[reaches], depth.astype(np.float32, copy=False))
    depth_array[~(depth_array > 0)] = np.nan

    return depth_array
