"""Kernel-weighted flood mapper: spreads each stream cell's water surface over its neighbourhood."""
import numpy as np
from numba import njit

@njit(cache=True)
def FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use):
    """Cells of ``E_Box`` below ``WSE`` that connect to the stream cell ``(r_use, c_use)``.

    Returns a 0/1 mask the shape of ``E_Box``.  The connection is traced in a
    few sweeps, so a winding path can be missed.
    """
    FourMatrix = np.full((3, 3), 4)

    nrows_local = np.int32(r_max - r_min + 2)
    ncols_local = np.int32(c_max - c_min + 2)
    FloodLocal = np.zeros((nrows_local,ncols_local), dtype=np.float32)

    FloodLocal[1:nrows_local-1,1:ncols_local-1] = np.where(E_Box<=WSE,1,0)

    # This is the Stream Cell.  Mark it with a 4
    r_idx = int(r_use - r_min + 1)
    c_idx = int(c_use - c_min + 1)
    FloodLocal[r_idx, c_idx] = 4

    # Spread the mark to the wet neighbours of marked cells
    for r in range((r_use-r_min+1),nrows_local-1):
        for c in range((c_use-c_min+1),ncols_local-1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    for r in range((r_use-r_min+1), 0, -1):
        for c in range((c_use-c_min+1), 0, -1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix

    for r in range(1, nrows_local-1):
        for c in range(1, ncols_local-1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix

    return np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>3.0,1.0,0.0)

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
    WSE is NaN where it is not above the DEM.  ``S_Rast`` (slopes) may be None.
    """
    WSE_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Slope_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Total_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)

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

        if S_Rast is not None:
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
        w_r_max = TW_for_WeightBox_ElipseMask + (r_max - r)
        w_c_min = TW_for_WeightBox_ElipseMask - (c - c_min)
        w_c_max = TW_for_WeightBox_ElipseMask + (c_max - c)

        weight_slice = WeightBox[w_r_min:w_r_max, w_c_min:w_c_max]
        if LocalFloodOption:
            E_Box = E[r_min:r_max,c_min:c_max]
            FloodLocalMask = FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r, c)
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice * FloodLocalMask)
            Total_Weight[r_min:r_max,c_min:c_max] += weight_slice * FloodLocalMask
            if S_Rast is not None:
                Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice * FloodLocalMask)
        else:
            cell_wse_weight = WSE * weight_slice
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += cell_wse_weight
            Total_Weight[r_min:r_max,c_min:c_max] += weight_slice

            # A weighted slope, for velocity
            if S_Rast is not None:
                Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice)

    # Flood where the weighted WSE is above a valid elevation
    valid_candidate = (
        (E > -9998.0) &
        (WSE_Times_Weight > E * Total_Weight)
    )

    WSE_array = np.where(valid_candidate, WSE_Times_Weight / Total_Weight, np.nan).astype(np.float32)
    return WSE_array, Total_Weight, Slope_Times_Weight
