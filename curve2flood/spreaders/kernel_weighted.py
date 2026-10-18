"""Kernel-weighted flood mapper: spreads each stream cell's water surface over its neighbourhood."""
import numpy as np
from numba import njit

# Cell states of FloodAllLocalAreas
_DRY, _WET, _REACHED = 0, 1, 2

@njit(cache=True)
def FloodAllLocalAreas(WSE, E, r_min, r_max, c_min, c_max, r_use, c_use, state):
    """Cells of ``E[r_min:r_max, c_min:c_max]`` at or below ``WSE`` that connect to the stream cell ``(r_use, c_use)``.

    Fills ``state``, at least ``(r_max - r_min + 2, c_max - c_min + 2)``, with
    ``_REACHED`` for those cells, offset by one row and column (a dry border).
    The connection is traced in a few sweeps, so a winding path can be
    missed.  The stream cell is always reached.
    """
    nrows_local = r_max - r_min + 2
    ncols_local = c_max - c_min + 2
    for r in range(nrows_local):
        for c in range(ncols_local):
            state[r, c] = _DRY
    for r in range(r_max - r_min):
        for c in range(c_max - c_min):
            if E[r_min + r, c_min + c] <= WSE:
                state[r + 1, c + 1] = _WET

    r_idx = r_use - r_min + 1
    c_idx = c_use - c_min + 1
    state[r_idx, c_idx] = _REACHED

    # Spread the mark to the wet neighbours of reached cells: down and right
    # of the stream cell, then up and left of it, then over the whole window
    for r in range(r_idx, nrows_local - 1):
        for c in range(c_idx, ncols_local - 1):
            if state[r, c] == _REACHED:
                _reach_neighbours(state, r, c)
    for r in range(r_idx, 0, -1):
        for c in range(c_idx, 0, -1):
            if state[r, c] == _REACHED:
                _reach_neighbours(state, r, c)
    for r in range(1, nrows_local - 1):
        for c in range(1, ncols_local - 1):
            if state[r, c] == _REACHED:
                _reach_neighbours(state, r, c)
    return state

@njit(cache=True, inline="always")
def _reach_neighbours(state, r, c):
    for a in range(r - 1, r + 2):
        for b in range(c - 1, c + 2):
            if state[a, b] == _WET:
                state[a, b] = _REACHED

@njit(cache=True)
def create_kernel_weighted_spread_map(
    RR,
    CC,
    T_Rast,
    W_Rast,
    S_Rast,
    E,
    nrows,
    ncols,
    TW_m,
    dx,
    dy,
    LocalFloodOption,
    WeightBox,
    TW_for_WeightBox_ElipseMask,
    TopWidthPlausibleLimit,
    Set_Depth,
):
    """Spread each stream cell's WSE over its neighbourhood.

    Stream cell ``(RR[i], CC[i])`` of the padded DEM ``E`` contributes its WSE
    (``W_Rast``) to the square window of cells up to one top width
    (``T_Rast``, capped at ``TW_m``) away, weighted by ``WeightBox``.  With
    ``LocalFloodOption`` it only reaches cells below its WSE that connect to it.  With ``Set_Depth > 0``
    each stream cell's WSE is its DEM elevation plus ``Set_Depth`` and its top
    width ``TopWidthPlausibleLimit``.

    Returns ``(WSE, total weight, slope times weight)`` on the padded grid;
    WSE is NaN where it is not above the DEM.  ``S_Rast`` (slopes) may be None,
    and the slope sums are then a 1 x 1 placeholder.

    The sums are added cell by cell, with the arithmetic of the array sums
    this replaced (a WSE times a weight in float64, a slope times a weight in
    float32, each added to a float32 sum), so the result is the same to the
    last bit without a temporary array per stream cell.
    """
    WSE_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Total_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    has_slope = S_Rast is not None
    Slope_Times_Weight = np.zeros((nrows + 2, ncols + 2) if has_slope else (1, 1), dtype=np.float32)
    state = np.zeros((1, 1), dtype=np.uint8)  # FloodAllLocalAreas' cells, grown as needed

    num_nonzero = len(RR)
    for i in range(num_nonzero):
        r = RR[i]
        c = CC[i]
        if Set_Depth>0.0:
            WSE = float(E[r,c] + Set_Depth)
            COMID_TW_m = TopWidthPlausibleLimit
        else:
            # ARC results for this cell.  The rasters are not padded, hence the -1.
            WSE = np.round(W_Rast[r-1,c-1], 2)
            COMID_TW_m = T_Rast[r-1,c-1]

        if COMID_TW_m < 0.00001 or (WSE - E[r,c]) < 0.001:
            continue

        SLOPE = 0.0
        if has_slope:
            SLOPE = float(S_Rast[r-1,c-1])

        if COMID_TW_m > TW_m:
            COMID_TW_m = TW_m

        # How many cells the top width reaches from the stream cell
        COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))

        r_min = max(r - COMID_TW, 1)
        r_max = min(r + COMID_TW + 1, nrows + 1)
        c_min = max(c - COMID_TW, 1)
        c_max = min(c + COMID_TW + 1, ncols + 1)

        # The weight box is centred on TW_for_WeightBox_ElipseMask, the half
        # width it was built with, not on this cell's top width
        w_r_min = TW_for_WeightBox_ElipseMask - (r - r_min)
        w_c_min = TW_for_WeightBox_ElipseMask - (c - c_min)

        if LocalFloodOption:
            if state.shape[0] < r_max - r_min + 2 or state.shape[1] < c_max - c_min + 2:
                state = np.zeros((max(state.shape[0], r_max - r_min + 2), max(state.shape[1], c_max - c_min + 2)),
                                 dtype=np.uint8)
            FloodAllLocalAreas(WSE, E, r_min, r_max, c_min, c_max, r, c, state)
            # Every cell of the window adds its weight times 0 or 1, as the
            # masked array sums did (a NaN slope reaches the unreached cells)
            for a in range(r_max - r_min):
                for b in range(c_max - c_min):
                    reached = 1.0 if state[a + 1, b + 1] == _REACHED else 0.0
                    weight = WeightBox[w_r_min + a, w_c_min + b]
                    rr = r_min + a
                    cc = c_min + b
                    WSE_Times_Weight[rr, cc] = np.float32(np.float64(WSE_Times_Weight[rr, cc])
                                                          + WSE * np.float64(weight) * reached)
                    Total_Weight[rr, cc] = np.float32(np.float64(Total_Weight[rr, cc]) + np.float64(weight) * reached)
                    if has_slope:
                        Slope_Times_Weight[rr, cc] = np.float32(np.float64(Slope_Times_Weight[rr, cc])
                                                                + np.float64(np.float32(SLOPE) * weight) * reached)
        else:
            for a in range(r_max - r_min):
                for b in range(c_max - c_min):
                    weight = WeightBox[w_r_min + a, w_c_min + b]
                    rr = r_min + a
                    cc = c_min + b
                    WSE_Times_Weight[rr, cc] = np.float32(np.float64(WSE_Times_Weight[rr, cc]) + WSE * np.float64(weight))
                    Total_Weight[rr, cc] += weight
                    # A weighted slope, for velocity
                    if has_slope:
                        Slope_Times_Weight[rr, cc] += np.float32(SLOPE) * weight

    # Flood where the weighted WSE is above a valid elevation; the weighted
    # WSE takes the place of the WSE x weight sums
    nan = np.float32(np.nan)
    for a in range(nrows + 2):
        for b in range(ncols + 2):
            elevation = E[a, b]
            total = Total_Weight[a, b]
            if elevation > -9998.0 and WSE_Times_Weight[a, b] > elevation * total:
                WSE_Times_Weight[a, b] = WSE_Times_Weight[a, b] / total
            else:
                WSE_Times_Weight[a, b] = nan
    return WSE_Times_Weight, Total_Weight, Slope_Times_Weight
