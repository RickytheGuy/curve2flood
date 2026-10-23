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

``solver="iterative"`` (default)
    The depth-stepped algorithm, raising the water ``dh`` at a time.  Without
    ``iterative_spill`` water spills over a divide once per step, as FLDPLN
    recommends for coarse DEMs (30 m); a larger ``dh`` damps spillover more.

``solver="exact"``
    DTF is the smallest total climb along any 8-connected path from the stream
    to the pixel (downhill steps are free), found in one Dijkstra sweep.  This
    is the limit of the depth-stepped algorithm with ``iterative_spill`` as
    ``dh`` goes to 0: water keeps spilling and running downhill for as long as
    it can.  On a valley floor or fan below a canyon that spreads a stream's
    water over everything lower: on a 30 m DEM of the Grand Valley (Colorado),
    small desert washes flooded 355 km2 outside the river at the 2-year flow,
    against 47 km2 with the default.  On 68 US test sites the default scores
    about the same (mean MCC 0.5705 against 0.5649).

The exact solver solves each stream segment in a window around it that grows
when the floodplain reaches its edge, so the result never depends on the
window size.  A library build solves with the iterative solver on the whole
raster, in arrays each thread keeps from one segment to the next
(``_Workspace``), which gives what the windows give without solving any
segment twice.

Flow directions use WhiteboxTools D8 codes, and the stream table has columns
``start_pixel, end_pixel, length, stream_id``.  Pixels are flat row-major
indices, ``row * ncols + col``.
"""

from __future__ import annotations

import os
import math
import itertools
import threading
import contextlib
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import tqdm
import numpy as np
import polars as pl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from numba import njit
from osgeo import gdal
from numba.extending import register_jitable

from curve2flood import LOG
from curve2flood.config import is_parquet

# Columns stored as 2-decimal numbers.  A fill can be deep (a pit filled to its
# rim), so it gets more digits than a DTF, which is at most the library's fldmx.
DECIMAL_COLS = {"DTF": pa.decimal32(5, 2), "fill": pa.decimal32(7, 2)}

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
# keep the per-edge rounding well below the centimetres the library is written
# in (DECIMAL_COLS), even after accumulating over a long path, while leaving an
# int32 range worth hundreds of kilometres of depth.
_DTF_UNITS_PER_METRE = 10000.0

# Library rows written and read at a time: one row group of a library written by
# save_library.
_LIBRARY_SLICE_ROWS = 1 << 20

# Segments whose rows may wait for the library's writer thread
_WRITES_AHEAD = 4

# The most threads a parallel build takes by default: each holds a copy of the
# solver's arrays of the whole raster, and more than two hardly build faster
_DEFAULT_THREADS = 4


@register_jitable(cache=True, nogil=True, forceinline=True)
def _pixel_to_rc(pixel: int, ncols: int) -> tuple[int, int]:
    return pixel // ncols, pixel % ncols

@register_jitable(cache=True, nogil=True, forceinline=True)
def _rc_to_pixel(row: int, col: int, ncols: int) -> int:
    return np.int32(row * ncols + col)

@register_jitable(cache=True, nogil=True)
def _next_downstream(pixel: int, fdr: np.ndarray, nrows: int, ncols: int) -> int:
    """The pixel ``pixel`` drains into, or -1 at a sink or the raster edge."""
    fd = fdr[pixel]
    if fd == 0:
        return -1

    # Whitebox D8 code -> (row, col) step.  Branches, not a lookup table: a
    # table built on every call was most of the cost of walking the network.
    row, col = _pixel_to_rc(pixel, ncols)
    if fd == 1:
        row -= 1
        col += 1
    elif fd == 2:
        col += 1
    elif fd == 4:
        row += 1
        col += 1
    elif fd == 8:
        row += 1
    elif fd == 16:
        row += 1
        col -= 1
    elif fd == 32:
        col -= 1
    elif fd == 64:
        row -= 1
        col -= 1
    elif fd == 128:
        row -= 1
    else:
        raise ValueError("Not a WhiteboxTools D8 flow direction code.")
    if row < 0 or row >= nrows or col < 0 or col >= ncols:
        return -1
    return _rc_to_pixel(row, col, ncols)


@njit(cache=True, nogil=True)
def _segment_pixels(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray,
                    nrows: int, ncols: int) -> np.ndarray:
    """Return stream pixels for a segment id, walking the FDR downstream."""
    mask = stream_info[:, 3] == stream_id
    if not np.any(mask):
        raise ValueError("stream_id not found in stream_info.")

    start = stream_info[mask, 0][0]
    length = stream_info[mask, 2][0]

    pixels = np.empty(max(length, 0), dtype=np.int64)
    if length <= 0:
        return pixels

    pixels[0] = start
    n = 1
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        pixels[n] = nxt
        n += 1
        current = nxt
    return pixels[:n]

@njit(cache=True, nogil=True)
def _downstream_exclusion(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray,
                          nrows: int, ncols: int) -> np.ndarray:
    """Pixels downstream of the segment end -- ``TD(R)``, excluded from spillover.

    The walk stops at a sink, the raster edge, or where the FDR loops back on
    itself.  A loop is found by Brent's method, comparing each pixel with one
    remembered at doubling intervals, so that no set of the walked pixels is
    kept.
    """
    end_pixel = stream_info[stream_info[:, 3] == stream_id, 1][0]

    excluded = np.empty(256, dtype=np.int64)
    n = 0
    current = end_pixel
    remembered = end_pixel
    interval = 1
    steps = 0
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1 or nxt == end_pixel:
            return excluded[:n]
        if n == excluded.size:
            bigger = np.empty(2 * n, dtype=np.int64)
            bigger[:n] = excluded
            excluded = bigger
        excluded[n] = nxt
        n += 1
        if nxt == remembered:
            # Round a loop: every pixel on it has been walked once
            return np.unique(excluded[:n])
        steps += 1
        if steps == interval:
            remembered = nxt
            interval *= 2
            steps = 0
        current = nxt


# ---------------------------------------------------------------------------
# The channel beside TD(R)
#
# TD(R) is a one-pixel line, but a river is wider: water that may not enter the
# stream line below its segment's end runs down the channel beside it, past the
# end, as far as the channel goes (descending is free).  Below the Karnali's
# Chisapani gorge that flooded a 1,000 km2 fan from one gorge segment.  Given a
# water mask, TD(R) also takes the water cells whose D8 path first meets the
# stream network inside it: the channel downstream of R, not its tributaries'.
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _network_pixels(stream_info: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    """Which pixels are on the stream network: each reach's, walked ``length`` pixels down the FDR from its start
    as ``_segment_pixels`` walks them."""
    on_network = np.zeros(nrows * ncols, dtype=np.bool_)
    for i in range(stream_info.shape[0]):
        current = stream_info[i, 0]
        length = stream_info[i, 2]
        for step in range(length):
            if step > 0:
                current = _next_downstream(current, fdr, nrows, ncols)
                if current == -1:
                    break
            on_network[current] = True
    return on_network


@njit(cache=True, nogil=True)
def _first_network_pixel(starts: np.ndarray, fdr: np.ndarray, on_network: np.ndarray, nrows: int,
                         ncols: int) -> np.ndarray:
    """The first network pixel on the D8 path from each of ``starts`` (itself if it is one), or -1 for a path that
    ends at a sink, the raster's edge or a loop first.  Each path is walked once: the cells on it remember where it
    led."""
    unknown, walking = -2, -3
    found = np.full(nrows * ncols, unknown, dtype=np.int64)
    path = np.empty(nrows * ncols, dtype=np.int64)
    result = np.empty(starts.size, dtype=np.int64)
    for i in range(starts.size):
        current = starts[i]
        n = 0
        while True:
            if found[current] == walking:   # a loop in the FDR
                hit = -1
                break
            if found[current] != unknown:
                hit = found[current]
                break
            if on_network[current]:
                hit = current
                break
            found[current] = walking
            path[n] = current
            n += 1
            nxt = _next_downstream(current, fdr, nrows, ncols)
            if nxt == -1:
                hit = -1
                break
            current = nxt
        for k in range(n):
            found[path[k]] = hit
        result[i] = hit
    return result


def _channel_cells(flow_direction: np.ndarray, stream_info: np.ndarray, water: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The water cells off the stream network, grouped by the network pixel their D8 path first meets: the
    group of pixel ``keys[i]`` is ``cells[starts[i]:starts[i + 1]]``, ``keys`` sorted.  Cells whose path meets no
    network pixel are left out."""
    nrows, ncols = flow_direction.shape
    flat_fdr = np.ascontiguousarray(flow_direction, dtype=np.uint8).ravel()
    on_network = _network_pixels(np.ascontiguousarray(stream_info, dtype=np.int64), flat_fdr, nrows, ncols)
    candidates = np.flatnonzero(np.asarray(water, dtype=bool).ravel() & ~on_network)
    hits = _first_network_pixel(candidates, flat_fdr, on_network, nrows, ncols)
    meets = hits >= 0
    hits, candidates = hits[meets], candidates[meets]
    order = np.argsort(hits, kind="stable")
    hits, cells = hits[order], candidates[order]
    first = np.flatnonzero(np.r_[True, hits[1:] != hits[:-1]]) if hits.size else np.empty(0, np.int64)
    return hits[first], np.r_[first, hits.size].astype(np.int64), cells


def _widen_downstream(downstream: np.ndarray, channel: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """TD(R) (``downstream``, its stream pixels) with the channel cells that drain first into them."""
    keys, starts, cells = channel
    if downstream.size == 0 or keys.size == 0:
        return downstream
    at = np.minimum(np.searchsorted(keys, downstream), keys.size - 1)
    groups = at[keys[at] == downstream]
    if groups.size == 0:
        return downstream
    # The groups' cells, gathered in one index rather than a slice per group
    sizes = starts[groups + 1] - starts[groups]
    offsets = np.cumsum(sizes) - sizes
    index = np.repeat(starts[groups] - offsets, sizes) + np.arange(sizes.sum())
    return np.concatenate([downstream, cells[index]])


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
                 bg: float, spill_decay: float, max_cells: int = 0):
    """Dijkstra over ``w(u -> v) = max(0, E(v) - E(u))`` on the 8-neighbourhood.

    Returns ``(cells, dtf, fsp, depth_limit)`` in window-local flat ids.
    ``spill_decay`` is added on each step to a pixel's D8 successor, as the
    depth-stepped solver adds it on spillover paths.

    ``max_cells`` caps the floodplain at that many cells (0 is no cap).  Cells
    settle in order of DTF, so once the cap is full the sweep stops at the DTF
    ``d`` of the next cell and keeps only the cells shallower than ``d``: the
    floodplain an ``fldmx`` just below ``d`` would give.  ``depth_limit`` is
    ``d`` in metres, or -1 when the cap was not reached.  Cells flooded at zero
    depth are always kept.

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
    while heap_n > 0:
        key, heap_n = _heap_pop(heap, heap_n)
        pixel = np.int32(key & np.int64(0xFFFFFFFF))
        depth = np.int64(key >> 32)
        if settled[pixel] or depth != dist[pixel]:
            continue
        if max_cells > 0 and n_settled >= max_cells and depth > 0:
            stop = depth
            break
        settled[pixel] = True
        n_settled += 1

        elevation = fil[pixel]
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
            out_dtf[k] = dist[pixel] / _DTF_UNITS_PER_METRE  # as found, not raised to fldmn (see build_fldpln_library)
            out_fsp[k] = fsp[pixel]
            k += 1
    depth_limit = stop / _DTF_UNITS_PER_METRE if stop >= 0 else -1.0
    return cells[:k], out_dtf[:k], out_fsp[:k], depth_limit


# ---------------------------------------------------------------------------
# Depth-stepped solver -- algorithm steps 1-8
# ---------------------------------------------------------------------------

# The depth-stepped solver tests a cell's eight neighbours over and over: are
# they dry, do they have data, do they drain into it?  It reads the answers as
# bit masks, bit ``pos`` for the neighbour at ``NEIGHBOR_DELTAS[pos]``: one
# byte rather than eight scattered reads.

@njit(cache=True, nogil=True)
def _neighbour_bits(fil: np.ndarray, fdr: np.ndarray, ncols: int, bg: float) -> tuple[np.ndarray, np.ndarray]:
    """For each cell of a window with its halo (flat, ``ncols`` wide): which neighbours drain into it (``inflow``),
    and which have data (``has_data``, not ``bg``).  The halo's cells have neither."""
    inflow = np.zeros(fdr.size, dtype=np.uint8)
    has_data = np.zeros(fdr.size, dtype=np.uint8)
    for row in range(1, fdr.size // ncols - 1):
        for pixel in range(row * ncols + 1, (row + 1) * ncols - 1):
            drains, data = 0, 0
            for pos in range(8):
                dr, dc = NEIGHBOR_DELTAS[pos]
                neighbor = pixel + dr * ncols + dc
                if fdr[neighbor] == INFLOW[pos]:
                    drains |= 1 << pos
                if fil[neighbor] != bg:
                    data |= 1 << pos
            inflow[pixel] = drains
            has_data[pixel] = data
    return inflow, has_data

@njit(cache=True, nogil=True, inline="always")
def _flood(cell: int, in_flood: np.ndarray, wet: np.ndarray, offsets: np.ndarray):
    """Mark ``cell`` flooded, and so a wet neighbour of each of its neighbours (``wet``: bit ``pos`` for the
    neighbour at ``NEIGHBOR_DELTAS[pos]``, whose own neighbour at ``7 - pos`` is the cell)."""
    in_flood[cell] = True
    for pos in range(8):
        neighbor = cell + offsets[pos]
        wet[neighbor] = wet[neighbor] | (1 << (7 - pos))

@njit(cache=True, nogil=True)
def _backfill_from_source(source: int, base_dtf: float, max_wse: float,
                          fil: np.ndarray, inflow: np.ndarray, offsets: np.ndarray,
                          bg: float, skip_flooded: bool, wet: np.ndarray,
                          blocked: np.ndarray, blocked_stamp: int,
                          stack: np.ndarray, visited: np.ndarray, visit_stamp: int,
                          out_pixels: np.ndarray, out_dtf: np.ndarray) -> int:
    """BFA step 1: backfill against the FDR from ``source`` (chapter 3, p.113).

    ``DTF`` is measured from ``source``'s elevation, as the dissertation defines
    it; on a filled DEM the inverse-D8 walk is elevation-monotone, so that equals
    the accumulated positive rise.  ``inflow`` (``_neighbour_bits``) gives
    the neighbours that drain into each cell, and ``wet`` those flooded,
    skipped with ``skip_flooded``.  Returns the number of cells written.
    """
    source_elev = fil[source]
    visited[source] = visit_stamp
    stack[0] = source
    top = 1
    count = 0
    while top > 0:
        top -= 1
        center = stack[top]
        bits = inflow[center]
        if skip_flooded:
            bits &= ~wet[center]
        for pos in range(8):
            if not (bits >> pos) & 1:
                continue
            neighbor = center + offsets[pos]
            if visited[neighbor] == visit_stamp:
                continue
            if blocked_stamp != 0 and blocked[neighbor] == blocked_stamp:
                continue
            neighbor_elev = fil[neighbor]
            if neighbor_elev == bg or neighbor_elev > max_wse:
                continue
            visited[neighbor] = visit_stamp
            out_pixels[count] = neighbor
            out_dtf[count] = base_dtf + max(0.0, neighbor_elev - source_elev)
            count += 1
            stack[top] = neighbor
            top += 1
    return count

@njit(cache=True, nogil=True)
def _refresh_boundary(boundary: np.ndarray, n_boundary: int,
                      added: np.ndarray, n_added: int,
                      has_data: np.ndarray, wet: np.ndarray, on_boundary: np.ndarray) -> int:
    """Re-derive the full interior boundary ``dI F`` (algorithm steps 3 and 8):
    the flooded cells with a dry neighbour that has data.

    The floodplain only grows, so a flooded cell that is not on the boundary can
    never rejoin it.  Retesting the previous boundary plus the newly flooded
    cells therefore yields the exact ``dI F``, at ``O(|dI F| + |added|)`` instead
    of a rescan of the whole floodplain.
    """
    kept = 0
    for i in range(n_boundary):
        pixel = boundary[i]
        if has_data[pixel] & ~wet[pixel]:
            boundary[kept] = pixel
            kept += 1
        else:
            on_boundary[pixel] = False
    for i in range(n_added):
        pixel = added[i]
        if on_boundary[pixel]:
            continue
        if has_data[pixel] & ~wet[pixel]:
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

    Ties keep the order of ``keys``.  Past a few hundred keys they are sorted
    by a radix sort of both as one 64-bit key, which is several times faster.
    """
    n = keys.size
    if n < 512:
        by_elevation = keys[np.argsort(-elevation[keys], kind="mergesort")]
        return by_elevation[np.argsort(depth[by_elevation], kind="mergesort")]
    
    key = (_ordered_bits(depth[keys]) << np.uint64(32)) | _ordered_bits(-elevation[keys])
    # Stable least-significant-digit radix sort, 11 bits at a time
    order = keys.copy()
    spare_order = np.empty_like(order)
    spare_key = np.empty_like(key)
    counts = np.empty(2049, dtype=np.int64)
    for shift in range(0, 64, 11):
        counts[:] = 0
        for i in range(n):
            counts[((key[i] >> np.uint64(shift)) & np.uint64(2047)) + 1] += 1
        if counts.max() == n:
            continue  # every key has this digit
        for d in range(2048):
            counts[d + 1] += counts[d]
        for i in range(n):
            d = (key[i] >> np.uint64(shift)) & np.uint64(2047)
            at = counts[d]
            counts[d] = at + 1
            spare_order[at] = order[i]
            spare_key[at] = key[i]
        order, spare_order = spare_order, order
        key, spare_key = spare_key, key
    return order

@njit(cache=True, nogil=True)
def _ordered_bits(values: np.ndarray) -> np.ndarray:
    """float32 ``values`` as unsigned integers in the same order, ties as ties (-0.0 is 0.0); no NaN."""
    bits = (values + np.float32(0.0)).view(np.uint32)
    sign = np.uint32(0x80000000)
    out = np.empty(values.size, dtype=np.uint64)
    for i in range(values.size):
        b = bits[i]
        out[i] = np.uint64(~b if b & sign else b | sign)
    return out

@njit(cache=True, nogil=True)
def _forward_path(start: int, spill_dtf: float, head: float, fil: np.ndarray, fdr: np.ndarray,
                  offsets: np.ndarray, flood_depths: np.ndarray,
                  bg: float, excluded: np.ndarray, spill_decay: float,
                  path: np.ndarray, seen: np.ndarray, seen_stamp: int) -> int:
    """Algorithm step 6a: the trajectory ``T(y_k)``, with its two halt criteria.

    The spill floods ``start`` from ``spill_dtf`` with ``head`` metres of water
    above it, and every step down the path costs ``spill_decay`` of both: the
    path's ``j``-th cell floods from ``spill_dtf + j * spill_decay``.  So with a
    decay the path also ends where its head would run out on the next step.

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
        if spill_decay > 0.0 and length * spill_decay > head:
            break
        if excluded[nxt]:
            path[length] = nxt
            length += 1
            break
        if fil[nxt] == bg:
            break
        if flood_depths[nxt] > 0.0 and np.float32(spill_dtf + length * spill_decay) >= flood_depths[nxt]:
            break
        current = nxt
    return length

# The iterative solver's arrays, one value per cell of its window (``_iterative_arrays``), in this order:
#  0-4    dtf, fsp, in_flood, on_boundary, flood_depths: the floodplain, set only at the cells it floods
#  5-13   dirty, boundary, added, stack, fill_pixels, fill_dtf, path, touched, flooded: lists of cells
#  14-17  cand_avail, cand_belev, cand_fsp, cand_dtf: spillover candidates, read only where cand_stamp is current
#  18-19  spilled_depth, spilled_dtf: the path depth and DTF a cell was last backfilled from, read only where
#         spilled_round is current
#  20-25  dirty_stamp, visited, seen, on_path, cand_stamp, spilled_round: generation stamps
#  26     wet: which neighbours are flooded (see _flood), set only next to the cells it floods
# ``counters`` holds the last stamp and spill round, so that stamps go on rising when the arrays are reused.
_UNREACHED = np.float32(3.4e38)
_STAMPS = slice(20, 26)

def _iterative_arrays(n: int) -> tuple[np.ndarray, ...]:
    """The iterative solver's arrays for a window of ``n`` cells, as it starts.

    Allocated by numpy, not numba, whose zeros are written out: numpy's are
    the operating system's, and take no memory until the solver writes them.
    """
    return (np.full(n, _UNREACHED, dtype=np.float32), np.full(n, np.int32(-1), dtype=np.int32),
            np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.float32),
            np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32), np.empty(n, dtype=np.float32),
            np.empty(n + 2, dtype=np.int32), np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32), np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float32),
            np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.uint8))

@njit(cache=True, nogil=True)
def _iterate(fil: np.ndarray, fdr: np.ndarray, inflow: np.ndarray, has_data: np.ndarray, ncols: int,
             sources: np.ndarray, excluded: np.ndarray,
             dh: float, fldmn: float, fldmx: float, iterative_spill: bool,
             global_max_wse: float, bg: float, spill_decay: float, arrays, counters: np.ndarray) -> int:
    """Depth-stepped FLDPLN (algorithm steps 1-8) on ``arrays`` (``_iterative_arrays``).

    Returns how many cells were listed in ``flooded``: each cell of the
    floodplain, in the order it was flooded (a source given twice, twice).
    """
    (dtf, fsp, in_flood, on_boundary, flood_depths,
     dirty, boundary, added, stack, fill_pixels, fill_dtf, path, touched, flooded,
     cand_avail, cand_belev, cand_fsp, cand_dtf, spilled_depth, spilled_dtf,
     dirty_stamp, visited, seen, on_path, cand_stamp, spilled_round, wet) = arrays

    # ``flood_depths`` is the reference's ``flddat``.  It tracks ``dtf`` during
    # the backfill phase but is frozen while a spill round processes its
    # candidates, so writes made then are queued in ``dirty`` and folded in at
    # the start of the next round.
    n_dirty = 0
    round_id = counters[1]

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
    stamp = counters[0]
    n_flooded = 0

    n_added = 0
    for i in range(sources.size):
        source = sources[i]
        if fil[source] == bg:
            continue
        dtf[source] = 0.0
        fsp[source] = source
        _flood(source, in_flood, wet, offsets)
        flood_depths[source] = fldmn
        added[n_added] = source
        flooded[n_flooded] = source
        n_flooded += 1
        n_added += 1

    n_boundary = _refresh_boundary(boundary, 0, added, n_added, has_data, wet, on_boundary)

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
            if not inflow[pixel] & ~wet[pixel]:
                continue  # no dry neighbour drains into it: there is nothing to backfill

            max_wse = min(global_max_wse, fil[pixel] + fldht - pixel_dtf)
            stamp += 1
            count = _backfill_from_source(
                pixel, pixel_dtf, max_wse, fil, inflow, offsets, bg,
                True, wet, on_path, 0, stack, visited, stamp,
                fill_pixels, fill_dtf)
            for j in range(count):
                target = fill_pixels[j]
                value = fill_dtf[j]
                if value < dtf[target]:
                    dtf[target] = value
                    fsp[target] = pixel_fsp
                    flood_depths[target] = max(fldmn, value)
                    if not in_flood[target]:
                        _flood(target, in_flood, wet, offsets)
                        added[n_added] = target
                        flooded[n_flooded] = target
                        n_flooded += 1
                        n_added += 1
        n_boundary = _refresh_boundary(boundary, n_boundary, added, n_added, has_data, wet, on_boundary)

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
                dry = has_data[pixel] & ~wet[pixel]

                for pos in range(8):
                    if not (dry >> pos) & 1:
                        continue
                    neighbor = pixel + offsets[pos]
                    if excluded[neighbor]:
                        continue
                    neighbor_elev = fil[neighbor]
                    if neighbor_elev > limit:
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
            # shallowest spill first.  Each step down a trajectory costs
            # spill_decay: its j-th cell floods from spill_dtf + j * spill_decay,
            # with path_depth - j * spill_decay of water to backfill with.
            ordered = _order_by_depth_then_elevation(touched[:n_touched],
                                                     cand_dtf, cand_belev)
            for i in range(n_touched):
                start = ordered[i]
                spill_dtf = cand_dtf[start]
                spill_fsp = cand_fsp[start]
                start_elev = fil[start]
                path_depth = min(global_max_wse, start_elev + fldht - spill_dtf) - start_elev
                stamp += 1
                length = _forward_path(start, spill_dtf, path_depth, fil, fdr, offsets,
                                       flood_depths, bg, excluded, spill_decay,
                                       path, seen, stamp)
                if length == 0:
                    continue

                path_stamp = stamp
                for j in range(length):
                    on_path[path[j]] = path_stamp

                for j in range(length):
                    source = path[j]
                    source_dtf = np.float32(spill_dtf + j * spill_decay)
                    if source_dtf < dtf[source]:
                        dtf[source] = source_dtf
                        fsp[source] = spill_fsp
                        if dirty_stamp[source] != round_id:
                            dirty_stamp[source] = round_id
                            dirty[n_dirty] = source
                            n_dirty += 1
                        if not in_flood[source]:
                            _flood(source, in_flood, wet, offsets)
                            added[n_added] = source
                            flooded[n_flooded] = source
                            n_flooded += 1
                            n_added += 1
                            n_new += 1

                for j in range(length):
                    source = path[j]
                    source_dtf = np.float32(spill_dtf + j * spill_decay)
                    source_depth = path_depth - j * spill_decay
                    # An earlier path of this round that backfilled from here
                    # from no more DTF, at least as deep, gave every cell above
                    # it as little DTF as this one would: its own path cells
                    # above it too, which block this one's backfill and are no
                    # lower.  Most path cells of a round are another's.  (Spills
                    # run shallowest first, so without a decay the earlier DTF
                    # is never more.)
                    if (spilled_round[source] == round_id and source_depth <= spilled_depth[source]
                            and source_dtf >= spilled_dtf[source]):
                        continue

                    spilled_round[source] = round_id
                    spilled_depth[source] = source_depth
                    spilled_dtf[source] = source_dtf
                    stamp += 1
                    count = _backfill_from_source(
                        source, 0.0, fil[source] + source_depth, fil, inflow, offsets, bg,
                        False, wet, on_path, path_stamp, stack, visited, stamp,
                        fill_pixels, fill_dtf)
                    
                    for k in range(count):
                        target = fill_pixels[k]
                        value = fill_dtf[k] + source_dtf
                        if value < dtf[target]:
                            dtf[target] = value
                            fsp[target] = spill_fsp
                            if dirty_stamp[target] != round_id:
                                dirty_stamp[target] = round_id
                                dirty[n_dirty] = target
                                n_dirty += 1
                            if not in_flood[target]:
                                _flood(target, in_flood, wet, offsets)
                                added[n_added] = target
                                flooded[n_flooded] = target
                                n_flooded += 1
                                n_added += 1
                                n_new += 1

            n_boundary = _refresh_boundary(boundary, n_boundary, added, n_added, has_data, wet, on_boundary)

            # Step 7: keep spilling while the boundary can still afford to.
            spill = False
            if iterative_spill and n_new > 0:
                for i in range(n_boundary):
                    if dtf[boundary[i]] < fldht:
                        spill = True
                        break
            shallow_only = True

    counters[0] = stamp
    counters[1] = round_id
    return n_flooded

def _solve_iterative(fil: np.ndarray, fdr: np.ndarray, ncols: int,
                     sources: np.ndarray, excluded: np.ndarray,
                     dh: float, fldmn: float, fldmx: float, iterative_spill: bool,
                     global_max_wse: float, bg: float, spill_decay: float):
    """Depth-stepped FLDPLN (algorithm steps 1-8) in a window of its own.

    Returns ``(cells, dtf, fsp)`` in window-local flat ids, ascending.
    """
    inflow, has_data = _neighbour_bits(fil, fdr, ncols, bg)
    return _solve_iterative_reusing(fil, fdr, inflow, has_data, ncols, sources, excluded, dh, fldmn, fldmx,
                                    iterative_spill, global_max_wse, bg, spill_decay, _iterative_arrays(fil.size),
                                    np.zeros(2, dtype=np.int64), 0, ncols)


@njit(cache=True, nogil=True)
def _solve_iterative_reusing(fil: np.ndarray, fdr: np.ndarray, inflow: np.ndarray, has_data: np.ndarray, ncols: int,
                             sources: np.ndarray, excluded: np.ndarray,
                             dh: float, fldmn: float, fldmx: float, iterative_spill: bool,
                             global_max_wse: float, bg: float, spill_decay: float, arrays, counters: np.ndarray,
                             halo: int, out_cols: int):
    """``_solve_iterative`` on ``arrays`` (``_iterative_arrays``), which it leaves as it found them, resetting only
    the cells it flooded: a thread can keep them from one segment to the next (see ``_Workspace``).

    The cells come out in ascending order: the flooded cells' bounding box
    is scanned row by row, not the whole window.  A cell at ``(row, col)``
    of the window comes out as ``(row - halo) * out_cols + col - halo``: its
    window-local id with ``halo`` 0 and ``out_cols`` ``ncols``, or with 1
    and the raster's width, its raster id when the window is the raster.
    """
    n_flooded = _iterate(fil, fdr, inflow, has_data, ncols, sources, excluded, dh, fldmn, fldmx, iterative_spill,
                         global_max_wse, bg, spill_decay, arrays, counters)
    dtf, fsp, in_flood, on_boundary, flood_depths, flooded = (arrays[0], arrays[1], arrays[2], arrays[3], arrays[4],
                                                              arrays[13])
    cells = np.empty(n_flooded, dtype=np.int32)
    out_dtf = np.empty(n_flooded, dtype=np.float32)
    out_fsp = np.empty(n_flooded, dtype=np.int32)
    if n_flooded == 0:
        return cells, out_dtf, out_fsp
    r0, c0 = fil.size, fil.size
    r1, c1 = -1, -1
    for i in range(n_flooded):
        row = flooded[i] // ncols
        col = flooded[i] - row * ncols
        r0, r1, c0, c1 = min(r0, row), max(r1, row), min(c0, col), max(c1, col)
    k = 0
    for row in range(r0, r1 + 1):
        for pixel in range(row * ncols + c0, row * ncols + c1 + 1):
            if in_flood[pixel]:
                cells[k] = (row - halo) * out_cols + pixel - row * ncols - halo
                out_dtf[k] = dtf[pixel]  # as found, not raised to fldmn (see build_fldpln_library)
                source = fsp[pixel]
                source_row = source // ncols
                out_fsp[k] = (source_row - halo) * out_cols + source - source_row * ncols - halo
                k += 1
    wet = arrays[26]
    for i in range(n_flooded):
        pixel = flooded[i]
        dtf[pixel] = _UNREACHED
        fsp[pixel] = -1
        in_flood[pixel] = False
        on_boundary[pixel] = False
        flood_depths[pixel] = 0.0
        for neighbor in (pixel - ncols - 1, pixel - ncols, pixel - ncols + 1, pixel - 1, pixel + 1,
                         pixel + ncols - 1, pixel + ncols, pixel + ncols + 1):
            wet[neighbor] = 0
    return cells[:k], out_dtf[:k], out_fsp[:k]

class _Workspace:
    """One thread's iterative solver, on the whole raster, kept from one segment to the next.

    A segment is first solved in a window around it, and again in a larger
    one whenever its floodplain reaches the window's edge.  The solver's work
    grows with the cells a segment floods, not with its window, so half of
    it went to windows thrown away: 42% on the Karnali, 52% on a tile of the
    Mississippi's floodplain.  The workspace solves every segment once, on
    the whole raster with the windows' one-cell halo, which a window that
    the floodplain doesn't reach gives the same result as.  Its arrays are
    allocated once, about 100 bytes a cell; only DTF and FSP are written in
    full at once, the rest as segments flood cells, and each solve resets
    the cells it flooded.

    ``fil`` and ``fdr`` are the filled DEM and flow directions with the halo
    (``_crop`` of the whole raster), flattened, and ``inflow`` and
    ``has_data`` their ``_neighbour_bits``; ``channel`` is
    ``_channel_lookup``'s, empty without a water mask.  The threads share
    them.
    """

    def __init__(self, fil: np.ndarray, fdr: np.ndarray, inflow: np.ndarray, has_data: np.ndarray,
                 channel: tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.fil, self.fdr, self.inflow, self.has_data, self.channel = fil, fdr, inflow, has_data, channel
        self.excluded = np.zeros(fil.size, dtype=np.bool_)
        self.arrays = _iterative_arrays(fil.size)
        self.counters = np.zeros(2, dtype=np.int64)

    def solve(self, sources: np.ndarray, downstream: np.ndarray, dh: float, fldmn: float, fldmx: float,
              iterative_spill: bool, global_max_wse: float, bg: float, spill_decay: float, ncols: int):
        """``_solve_iterative`` of the raster (``ncols`` wide), with ``sources`` as ids of the raster with its halo
        and ``downstream`` (TD(R)'s stream pixels, which the channel widens) as its own; the cells and sources
        come out as the raster's own ids."""
        if self.counters.max() > 2 ** 30:
            # The int32 stamps would run out: start them again from 0
            for stamps in self.arrays[_STAMPS]:
                stamps[:] = 0
            self.counters[:] = 0
        _mark_downstream(self.excluded, downstream, *self.channel, ncols, True)
        try:
            return _solve_iterative_reusing(self.fil, self.fdr, self.inflow, self.has_data, ncols + 2, sources,
                                            self.excluded, dh, fldmn, fldmx, iterative_spill, global_max_wse, bg,
                                            spill_decay, self.arrays, self.counters, 1, ncols)
        finally:
            _mark_downstream(self.excluded, downstream, *self.channel, ncols, False)

def _channel_lookup(channel: tuple[np.ndarray, np.ndarray, np.ndarray] | None, n: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_channel_cells``' groups looked up by raster id: ``(group, starts, cells)``, the group of pixel ``p``
    being ``cells[starts[group[p]]:starts[group[p] + 1]]``, ``group[p]`` -1 for none.  Empty for no channel."""
    if channel is None:
        return np.empty(0, np.int32), np.empty(0, np.int64), np.empty(0, np.int64)
    keys, starts, cells = channel
    group = np.full(n, -1, dtype=np.int32)
    group[keys] = np.arange(keys.size, dtype=np.int32)
    return group, starts, cells

@njit(cache=True, nogil=True)
def _mark_downstream(excluded: np.ndarray, downstream: np.ndarray, group: np.ndarray, starts: np.ndarray,
                     cells: np.ndarray, ncols: int, value: bool):
    """Set ``excluded`` (ids of the raster with its halo) to ``value`` at TD(R)'s pixels (``downstream``, raster
    ids, ``ncols`` wide) and at the channel cells of their groups (``_channel_lookup``): ``_widen_downstream``
    without building the list."""
    wc = ncols + 2
    for pixel in downstream:
        row = pixel // ncols
        excluded[(row + 1) * wc + pixel - row * ncols + 1] = value
        if group.size and group[pixel] >= 0:
            for cell in cells[starts[group[pixel]]:starts[group[pixel] + 1]]:
                row = cell // ncols
                excluded[(row + 1) * wc + cell - row * ncols + 1] = value

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
                               solver: str = "iterative",
                               max_cells: int = 0,
                               channel: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
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
        Iterative solver only: the least depth (m) a flooded pixel counts as
        when it decides where a spill stops.  DTFs are reported as found, not
        raised to it (see ``build_fldpln_library``).
    fldmx : float
        Largest depth to flood to (m).
    iterative_spill : bool
        Iterative solver only: keep spilling until the floodplain stops growing.
    global_max_wse : float, optional
        Highest water-surface elevation allowed; 0 for no limit.
    bg : float, optional
        No-data value of the DEM; such cells never flood.
    spill_decay : float, optional
        Head (m) a spill loses per pixel down its spillover path: the path's
        ``j``-th pixel floods from ``j * spill_decay`` more DTF than its first
        and backfills with that much less water, and the path ends where its
        water runs out.  The exact solver adds it on each step to a pixel's D8
        successor.  0 for none.
    solver : {"exact", "iterative"}, optional
        See the module docstring.
    max_cells : int, optional
        Most rows the segment may have (0 for no limit).  A segment that would
        flood more stops at the shallowest depth that exceeds it, recorded in
        ``attrs["depth_limit"]``.  Exact solver only.
    channel : tuple, optional
        The water cells grouped by the network pixel they first drain to
        (``_channel_cells``).  TD(R) then also takes the groups of its pixels:
        the channel below the segment, not only its stream line.

    Returns
    -------
    pandas.DataFrame
        ``FSP``, ``FPP`` and ``DTF`` columns.  ``attrs["depth_limit"]`` is the
        depth ``max_cells`` stopped at, or None.
    """
    fsp, fpp, dtf, depth_limit = _segment_rows(
        filled_dem, flow_direction, stream_id, stream_info, dh, fldmn, fldmx, iterative_spill, global_max_wse, bg,
        spill_decay, solver, max_cells, channel)
    table = pd.DataFrame({"FSP": fsp, "FPP": fpp, "DTF": dtf}, columns=["FSP", "FPP", "DTF"])
    table.attrs["depth_limit"] = depth_limit
    return table


def _segment_rows(filled_dem: np.ndarray, flow_direction: np.ndarray, stream_id: int, stream_info: np.ndarray,
                  dh: float, fldmn: float, fldmx: float, iterative_spill: bool, global_max_wse: float = 0.0,
                  bg: float = -9999, spill_decay: float = 0.0, solver: str = "iterative", max_cells: int = 0,
                  channel: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                  workspace: _Workspace | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """``fldpln_library_for_segment`` as arrays: ``(FSP, FPP, DTF, depth_limit)``.  The library build takes these
    rather than a table per segment, whose construction cost more than solving most segments.  With a
    ``workspace`` (of these rasters and this ``channel``), the iterative solver solves the segment on the whole
    raster, not in windows."""
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

    flat_fdr = flow_direction.ravel()
    stream_pixels = _segment_pixels(stream_id, stream_info, flat_fdr, nrows, ncols)
    downstream = _downstream_exclusion(stream_id, stream_info, flat_fdr, nrows, ncols)

    if stream_pixels.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), None

    if workspace is not None and solver == "iterative":
        # The whole raster with its halo: ids of a window from (0, 0).  The
        # workspace widens TD(R) with its own channel cells.
        fpp, dtf, fsp = workspace.solve(_to_local(stream_pixels, ncols, 0, 0, ncols + 2), downstream, dh, fldmn,
                                        fldmx, iterative_spill, global_max_wse, bg, spill_decay, ncols)
        return fsp, fpp, dtf, None

    if channel is not None:
        downstream = _widen_downstream(downstream, channel)

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
                fldmn, fldmx, global_max_wse, bg, spill_decay, max_cells)
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

    fpp = _to_global(cells, ncols, r0, c0, wc)
    fsp = _to_global(source_of, ncols, r0, c0, wc)
    return fsp, fpp, values, (float(depth_limit) if depth_limit >= 0 else None)


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

def _read_array(file: str, dtype: np.dtype) -> np.ndarray:
    """Band 1 of a raster as ``dtype``."""
    ds = gdal.Open(file)
    arr = np.empty((ds.RasterYSize, ds.RasterXSize), dtype=dtype)
    _read_into(ds, arr)
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
    as deep as its reach when it has no VDT cell (the mapper floods no such segment, but the library may be mapped with
    another VDT database). ``flat_fil`` is the
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

def build_fldpln_library(
    dem: str,
    filled_dem: str,
    stream_info_file: str,
    flow_direction_file: str,
    library_file: str,
    dh: float = 0.5,
    fldmn: float = 0.1,
    fldmx: float = 25.0,
    iterative_spill: bool = False,
    vdt_file: str = None,
    bg_mask: np.ndarray | None = None,
    stream_ids: list[int] | None = None,
    global_max_wse: float = 0.0,
    bg: float = -9999,
    parallel: bool = False,
    pbar: bool = True,
    processes: int | None = None,
    spill_decay: float = 0.0,
    solver: str = "iterative",
    max_cells_per_segment: int | None = None,
    max_reach_length: float | None = 5000.0,
    max_profile_deviation: float | None = 1.0,
    water_mask: np.ndarray | None = None
):
    """Build the FLDPLN floodplain library for every stream segment and save it.

    Each row is a stream pixel ``FSP`` (Flood Source Pixel), a cell it floods ``FPP`` (FloodPlain Pixel), the Depth
    to Flood ``DTF``, the ``segment`` the stream pixel belongs to and the
    cell's ``fill``: the filled DEM minus ``dem``.  Cells where ``dem`` has no
    data never flood, and are left out.

    Every source of a segment starts at the same depth, so each cell keeps the
    one stream pixel that reaches it at the smallest depth: the library is
    exact for one depth per segment, which is how the mapper floods it.
    Reaches are first cut where the filled DEM under them bends or steps (with
    ``max_profile_deviation``) and into pieces no longer than
    ``max_reach_length``, each solved and flooded on its own.

    Each segment's rows are written as soon as it is solved, sorted by
    ``FSP`` and then ``FPP``, so the build holds only a few segments' rows
    however big the library.  The file is written as ``<library_file>.partial``
    and renamed once complete: a build that fails leaves no library.

    Parameters
    ----------
    dem: str
        The DEM floods are mapped on (with the bathymetry, if any).  Used to
        derive the ``fill`` column.
    filled_dem: str
        Filled (depression-free) DEM raster, the one the flow directions were
        made from.
    stream_info_file: str
        CSV or Parquet table of the segments, with columns ``start_pixel``,
        ``end_pixel``, ``length`` (pixels) and ``stream_id``.
    flow_direction_file: str
        D8 flow-direction raster with WhiteboxTools codes.
    library_file: str
        Where to save the library: Parquet (``.parquet``/``.pq``, compact) or CSV.
    dh: float
        Depth step of the iterative solver (m), 0.5 by default.  Ignored by the
        exact solver.
    fldmn: float
        Iterative solver only: the least depth (m) a flooded pixel counts as
        when it decides where a spill stops, as the KU FLDPLN's minimum flood
        depth.  0.1 by default.  DTFs are stored as found, from 0 at the
        stream: the KU FLDPLN raises those below ``fldmn`` to it, which mapped
        every stream pixel and flat channel bed ``fldmn`` too shallow.
    fldmx: float
        Largest depth to flood to (m).  25 by default.
    iterative_spill: bool
        Iterative solver only: keep spilling until the floodplain stops growing.
        Recommended for fine DEMs (10 m or less); off by default.
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
        Solve several segments at once.
    pbar: bool
        Show a progress bar.
    processes: int | None
        Number of threads when ``parallel``; defaults to the CPU count, at
        most 4. Tests indicate that more threads do not help and consume more memory.
    spill_decay: float
        Head (m) water spilling over a divide loses per pixel down its path. 
        The path's ``j``-th pixel floods
        from ``j * spill_decay`` more DTF than its first, it backfills with
        that much less water, and the path ends where the water runs out.
        Without it a spill keeps its whole depth however far it runs, so on
        flat ground it floods everything below its source's water surface as
        a level pool.  The exact solver adds it on each step to a pixel's D8
        successor, so its other downhill steps stay free and it trims little.
        0 (default) for none.
    solver: str
        ``"iterative"`` (default) or ``"exact"``; see the module docstring.
    max_cells_per_segment: int | None
        Most rows one segment may contribute.  A segment that would flood more
        stops at the depth where it fills up, so deeper floods there are mapped
        only that far: this trades the largest floods on flat ground for memory.
        None (default) for no limit.  Needs ``solver="exact"``.
    max_reach_length: float | None
        Longest segment (m), measured along the flow directions with the
        filled DEM's cell size.  A longer reach is cut into
        ``ceil(length / max_reach_length)`` segments of about equal length.
        5000 (default). None keeps each reach whole.
    max_profile_deviation: float | None
        Also cut each reach where the filled DEM under the stream bends or
        steps, so that the bed is about straight along every segment: its
        profile along the river is simplified top-down (Douglas-Peucker)
        until no stream pixel is more than this many metres above or below
        its piece's straight line.  The pieces are then cut by
        ``max_reach_length``.  One depth per segment fits a straight bed
        best: the stage jumps at a step and where a filled flat begins.
        None makes no such cuts.
    water_mask: np.ndarray | None
        The river's water, such as the bathymetry water mask; same shape as
        the DEM, positive for water.  Each segment's ``TD(R)``, the cells below
        its end that its water may not spill into, is then the channel below
        it and not only the stream line: every water cell whose D8 path first
        meets the stream network inside ``TD(R)`` joins it.  None (default) keeps
        the line alone.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}.")
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

    if is_parquet(stream_info_file):
        stream_info = pd.read_parquet(stream_info_file)
    else:
        stream_info = pd.read_csv(stream_info_file)
    assert stream_info.ndim == 2, "Stream info file must be a 2D table"
    assert stream_info.shape[1] == 4, "Stream info file must have 4 columns: start pixel (0), end pixel (1), length (2), and stream ID (3)"
    # The whole network, for the channel cells: water draining to a reach left out below is still that reach's
    network = stream_info.values.astype(np.int64)

    if stream_ids is not None:
        stream_info = stream_info[stream_info.iloc[:, 3].isin(stream_ids)]

    filled_dem_array = _read_array(filled_dem, np.float32)
    flow_direction_array = _read_array(flow_direction_file, np.uint8)
    dem_shape = _raster_shape(dem)
    if dem_shape != filled_dem_array.shape:
        raise ValueError(f"The DEM has shape {dem_shape}, but the filled DEM has {filled_dem_array.shape}.")

    if bg_mask is not None:
        # A 0/1 integer mask would index rows 0 and 1 instead of the masked cells
        bg_mask = np.asarray(bg_mask, dtype=bool)
        if bg_mask.shape != filled_dem_array.shape:
            raise ValueError(f"bg_mask has shape {bg_mask.shape}, but the filled DEM has {filled_dem_array.shape}.")

    channel = None
    if water_mask is not None:
        water_mask = np.asarray(water_mask)
        if water_mask.shape != filled_dem_array.shape:
            raise ValueError(f"water_mask has shape {water_mask.shape}, but the filled DEM has "
                             f"{filled_dem_array.shape}.")
        channel = _channel_cells(flow_direction_array, network, water_mask > 0)  # NaN and no-data aren't water
        LOG.info(f"TD(R) takes in the channel: {channel[2].size:,} water cells drain first to "
                 f"{channel[0].size:,} stream pixels.")

    stream_ids = stream_info.iloc[:, 3].unique()
    vdt_depth: dict[int, float] = {}
    if vdt_file is None:
        max_depths = [fldmx] * len(stream_ids)
    else:
        # Read only the columns used: the VDT carries ~120 more
        vdt_is_parquet = is_parquet(vdt_file)
        if vdt_is_parquet:
            vdt_columns = pq.read_schema(vdt_file).names
        else:
            vdt_columns = pd.read_csv(vdt_file, nrows=0).columns
        # Find last wse_* column
        wse_cols = [col for col in vdt_columns if col.startswith('wse_')]
        assert wse_cols, "No wse_* columns found in VDT file"
        max_wse_col = wse_cols[-1]
        usecols = ['COMID', 'Row', 'Col', max_wse_col]
        if vdt_is_parquet:
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
    # One segment table for every task, not a copy per segment
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
            max_cells
        )
        for i, segment in enumerate(segments[:, 3])
    ]
    # Sort args from largest max depth to smallest, (longest time to shortest)
    args.sort(key=lambda x: x[4], reverse=True)

    if processes is None:
        processes = max(min(os.cpu_count() or 1, _DEFAULT_THREADS, len(args)), 1)
        if processes == 1:
            parallel = False

    fill_depth = _fill_depth(dem, filled_dem_array, bg).ravel()

    # The iterative solver solves every segment on the whole raster, each
    # thread with its own workspace (see _Workspace); the rasters with the
    # windows' halo are shared
    halo = None
    if solver == "iterative":
        halo_fil = _crop(filled_dem_array, 0, nrows, 0, ncols, bg).ravel()
        halo_fdr = _crop(flow_direction_array, 0, nrows, 0, ncols, np.uint8(0)).ravel()
        halo = (halo_fil, halo_fdr, *_neighbour_bits(halo_fil, halo_fdr, ncols + 2, bg),
                _channel_lookup(channel, nrows * ncols))
    local = threading.local()

    def solve(task):
        workspace = getattr(local, "workspace", None)
        if workspace is None and halo is not None:
            workspace = local.workspace = _Workspace(*halo)
        return _segment_rows(filled_dem_array, flow_direction_array, *task, channel=channel, workspace=workspace)

    # Each segment's rows are written as soon as they are solved, and in the
    # order of args however many threads solve them: holding every segment's
    # rows to sort them together took 12 bytes a row, and twice that to sort,
    # which on a big river's floodplain is billions of rows.  They are sorted
    # and compressed in the writer's thread, while the next segments solve
    limits = []
    with _LibraryWriter(library_file, background=True) as writer, \
            (ThreadPoolExecutor(processes) if parallel else contextlib.nullcontext()) as pool:
        if parallel:
            results = pbar(_ordered_map(pool, solve, args, 2 * processes), total=len(args),
                           desc="Processing streams in parallel")
        else:
            results = (solve(task) for task in pbar(args, desc="Processing streams"))
        for task, (fsp, fpp, dtf, limit) in zip(args, results):
            if limit is not None:
                limits.append(limit)
            writer.write_later(_library_rows, task[0], fsp, fpp, dtf, fill_depth)

    if limits:
        LOG.info(f"{len(limits)} of {len(args)} stream segments reached max_cells_per_segment={max_cells:,}; "
                 f"their floodplains stop at depths of {min(limits):.2f}-{max(limits):.2f} m "
                 f"(median {np.median(limits):.2f} m).")

def _ordered_map(pool: ThreadPoolExecutor, fn, tasks, ahead: int):
    """``pool.map(fn, tasks)``, with at most ``ahead`` tasks submitted and not yet taken.

    ``pool.map`` submits every task at once, and the results of those that
    finish early wait for the ones before them, all at once if one is slow.
    """
    tasks = iter(tasks)
    pending = deque(pool.submit(fn, task) for task in itertools.islice(tasks, ahead))
    while pending:
        yield pending.popleft().result()
        for task in itertools.islice(tasks, 1):
            pending.append(pool.submit(fn, task))

def _library_rows(segment: int, fsp: np.ndarray, fpp: np.ndarray, dtf: np.ndarray,
                  fill_depth: np.ndarray) -> dict[str, np.ndarray]:
    """One segment's library rows, sorted by FSP and then FPP (which compresses well), with each cell's fill from
    the flat ``fill_depth``.  Cells without a DEM value (a NaN fill) never flood, and are left out."""
    order = np.lexsort((fpp, fsp))
    fpp = fpp[order]
    fill = fill_depth[fpp]
    has_dem = np.isfinite(fill)
    if not has_dem.all():
        order, fpp, fill = order[has_dem], fpp[has_dem], fill[has_dem]
    return {"FSP": fsp[order], "FPP": fpp, "DTF": dtf[order],
            "segment": np.full(order.size, segment, dtype=np.int32), "fill": fill}

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

def save_library(table: pd.DataFrame | dict[str, np.ndarray], path: str):
    """Save a library (a table, or its columns' arrays by name) as compact Parquet.

    DTF and fill are stored as 2-decimal numbers, and each column gets the
    encoding that compresses it best; the file is several times smaller than
    plain Brotli-compressed Parquet.  It is compressed with zstd: Brotli made
    it 6% smaller, but took five times as long to write and twice as long to
    read.  Rows are converted and written a row group at a time.
    """
    with _LibraryWriter(path, parquet=True) as writer:
        writer.write({c: np.asarray(table[c]) for c in table})

class _LibraryWriter:
    """Writes a library a part at a time: Parquet as ``save_library`` saves it, in row groups of
    ``_LIBRARY_SLICE_ROWS`` rows whatever the parts' sizes, or CSV.

    The file is written beside ``path`` and takes its name only once closed
    without an error, so a build that fails or is killed never leaves a
    library that looks finished.  ``parquet`` is from ``path``'s suffix when
    None.  With ``background``, ``write_later`` makes and writes the rows in
    a thread of the writer's own, at most ``_WRITES_AHEAD`` parts behind.
    """

    def __init__(self, path: str | os.PathLike, parquet: bool | None = None, background: bool = False):
        self.path = Path(path)
        self._thread = ThreadPoolExecutor(1) if background else None
        self._queued = deque()
        self.parquet = is_parquet(self.path) if parquet is None else parquet
        self._partial = self.path.with_name(self.path.name + ".partial")
        self._begun = False
        self._writer = None
        self._pending: list[dict[str, np.ndarray]] = []
        self._n_pending = 0
        self._empty = None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        finished = False
        try:
            if self._thread is not None:
                try:
                    while self._queued and kind is None:
                        self._queued.popleft().result()
                finally:
                    self._thread.shutdown(wait=True, cancel_futures=True)
            if kind is None:
                self._flush(final=True)
                finished = True
        finally:
            if self._writer is not None:
                self._writer.close()
            if finished:
                os.replace(self._partial, self.path)
            else:
                self._partial.unlink(missing_ok=True)

    def write_later(self, make_rows, *args):
        """``write(make_rows(*args))``, in the writer's thread if it has one: the rows are made and compressed
        while the caller goes on.  An error there is raised by a later call, or on closing."""
        if self._thread is None:
            self.write(make_rows(*args))
            return
        while len(self._queued) >= _WRITES_AHEAD:
            self._queued.popleft().result()
        self._queued.append(self._thread.submit(lambda: self.write(make_rows(*args))))

    def write(self, columns: dict[str, np.ndarray]):
        """Add rows: each column's values by name, the same columns every time."""
        if self._empty is None:
            self._empty = {c: v[:0] for c, v in columns.items()}
        n = len(next(iter(columns.values()))) if columns else 0
        if not n:
            return
        self._pending.append(columns)
        self._n_pending += n
        if self._n_pending >= _LIBRARY_SLICE_ROWS:
            self._flush(final=False)

    def _flush(self, final: bool):
        """Write the pending rows: every whole row group of them, and the rest too if ``final``."""
        if not self._pending:
            if final and not self._begun and self._empty is not None:
                self._write_part(self._empty)  # a library without rows still has its columns
            return
        names = list(self._pending[0])
        if len(self._pending) == 1:
            rows = self._pending[0]
        else:
            rows = {c: np.concatenate([part[c] for part in self._pending]) for c in names}
        n = self._n_pending
        stop = n if final else n - n % _LIBRARY_SLICE_ROWS
        step = _LIBRARY_SLICE_ROWS if self.parquet else max(stop, 1)
        for start in range(0, stop, step):
            self._write_part({c: v[start:min(start + step, stop)] for c, v in rows.items()})
        # A copy, so that the rows written are freed
        self._pending = [{c: v[stop:].copy() for c, v in rows.items()}] if stop < n else []
        self._n_pending = n - stop

    def _write_part(self, part: dict[str, np.ndarray]):
        if not self.parquet:
            pd.DataFrame(part).to_csv(self._partial, index=False, mode="a" if self._begun else "w",
                                      header=not self._begun)
            self._begun = True
            return
        arrays = {}
        for c, values in part.items():
            if c in DECIMAL_COLS:
                arrays[c] = pc.cast(pa.array(values.astype(np.float32).round(2)), DECIMAL_COLS[c])
            else:
                arrays[c] = pa.array(values)
        table = pa.table(arrays)
        if not self._begun:
            encoding = {"FPP": "DELTA_BINARY_PACKED"} | {c: "BYTE_STREAM_SPLIT" for c in DECIMAL_COLS if c in part}
            self._writer = pq.ParquetWriter(
                self._partial, table.schema, use_dictionary=[c for c in ("FSP", "segment") if c in part],
                column_encoding=encoding, store_decimal_as_integer=True, compression="zstd", compression_level=9)
            self._begun = True
        self._writer.write_table(table, row_group_size=_LIBRARY_SLICE_ROWS)

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
    """The library's ``columns``, ``_LIBRARY_SLICE_ROWS`` rows at a time, with decimals as Float32.

    A Parquet file is read through pyarrow, a batch at a time: two to five
    times as fast as slicing ``pl.scan_parquet``, and half the memory.  The
    decimals are cast by polars either way, so their values are the same.
    Without pre-buffering: pyarrow would hold the requested columns of every
    row group until the last batch, 1.1 GB of an 860-million-row library,
    and read it no faster.
    """
    if isinstance(library, pl.LazyFrame):
        n = library.select(pl.len()).collect().item()
        parts = (library.slice(offset, _LIBRARY_SLICE_ROWS).select(columns).collect()
                 for offset in range(0, n, _LIBRARY_SLICE_ROWS))
    else:
        batches = pq.ParquetFile(library, pre_buffer=False).iter_batches(batch_size=_LIBRARY_SLICE_ROWS,
                                                                         columns=columns)
        parts = (pl.from_arrow(batch) for batch in batches)
    for part in parts:
        yield part.with_columns([pl.col(c).cast(pl.Float32) for c, d in part.schema.items() if d == pl.Decimal])

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

@njit(cache=True, nogil=True)
def _deepest_water(best: np.ndarray, key: np.ndarray, fpp: np.ndarray, dtf: np.ndarray, fill: np.ndarray,
                   dof: np.ndarray, below: np.ndarray):
    """Raise ``best[p]`` to the water of each library row ``(key, p, DTF, fill)`` that reaches ``p``'s filled
    surface: ``dof[key] - DTF + max(fill, 0)`` where ``dof[key] - DTF > 0``.

    ``dof`` is NaN for a key without a depth.  The arithmetic is in the
    columns' own types, rounded to float32, and a NaN depth leaves its cell
    NaN, as ``np.maximum.at`` did.  Unless ``below`` is empty, every row's
    cell also gets ``min(fill, 0)`` there: how far the filled DEM sits below
    the DEM.
    """
    zero = np.float32(0.0)
    record = below.size > 0
    for i in range(key.size):
        if record:
            f = fill[i]
            below[fpp[i]] = f if f < zero else zero
        k = key[i]
        if k < 0 or k >= dof.size:
            continue
        water = dof[k]
        if water != water:
            continue
        head = water - dtf[i]
        if not head > zero:
            continue
        f = fill[i]
        depth = np.float32(head + (f if (f > zero or f != f) else zero))
        p = fpp[i]
        current = best[p]
        if current != current:
            continue
        if depth != depth or depth > current:
            best[p] = depth

def _make_fldpln_flood_map(
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
       the stages ARC gave the segment.  The depth, not the water surface:
       where ARC has no rating curve, as down a cliff, the bed bends between
       the VDT cells, and a straight water surface across the gap would stand
       metres above it (or below it).  A segment without any VDT value floods
       nothing.  Its pixels used to take the stages of the reach's VDT cells
       on either side, but ARC rates no cross section of it, as down a cliff
       whose cross sections fall away on one side, and a stage borrowed from
       elsewhere spread over whatever lies below it: a 9 cm film down the
       cliff and fan below reach 710510156 at Grand Junction (N39W109) at
       the 2-year flow, and on 68 US test sites mostly ground their observed
       maps have dry (mean MCC 0.5659 with those segments, 0.5691 without).
       A library without a ``segment`` column (built before reaches were
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
        if not len(part):
            continue
        # A sorted library repeats each source over its rows: drop the repeats
        # first, cheaply, and leave the rest to the unique below
        repeat = np.ones(len(part) - 1, dtype=bool)
        for column in group:
            values = part[column].to_numpy()
            repeat &= values[1:] == values[:-1]
            
        # Converting to numpy here saves lots of memory
        first = np.r_[True, ~repeat]
        pairs.append(pl.DataFrame({column: part[column].to_numpy()[first] for column in group}))
        fsp = part['FSP'].to_numpy()
        is_own = fsp == part['FPP'].to_numpy()
        own_fsp.append(fsp[is_own])
        own_fill.append(part['fill'].to_numpy()[is_own])
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
        _deepest_water(best, part[key].to_numpy(), part['FPP'].to_numpy(), part['DTF'].to_numpy(),
                       part['fill'].to_numpy(), dof, below)
    depth_array[~((depth_array > 0) & (depth_array >= min_depth))] = np.nan

    return depth_array
