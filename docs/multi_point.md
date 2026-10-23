# Multi-Point Interpolation Mapper

`mapper: Curve2Flood-Multi-Point Interpolation`. Water surface points along the streams, and across the channels of
wide rivers, are interpolated into a water surface, and the cells below it that connect to the streams flood. It
follows the Follum Hydrologic Solutions flood mapper that interpolates high-water marks, which is why some of its
settings say `HWM`. Its settings are listed under
[Multi-point interpolation mapper](input_file.md#multi-point-interpolation-mapper).

## How it works

1. **Points.** Each stream cell becomes a point with the water surface elevation and top width of its rating curve at
   the reach's flow, from the VDT database (preferred) or the curve file. Here the VDT is interpolated linearly
   between its flows and held at its end values beyond them, skipping entries with no flow, top width or water
   surface. A missing ground elevation, slope or cross-section direction is borrowed from the other table.
2. **Outliers.** With `MPI_Remove_HWM_Outliers`, points whose depth (water surface less ground) is an outlier within
   their reach are dropped.
3. **Cross sections.** Where a river is wider than `MPI_TopWidth_Threshold_m`, points are added along its cross
   section (`XS_Angle`), every `MPI_XS_Point_Spacing_m`, on both sides, until the ground rises above the water or half
   the top width is reached. They carry the stream cell's water surface.
4. **Sanity check.** With `MPI_Apply_WSE_Sanity_Filter`, points whose water surface is below their ground are
   dropped.
5. **Corridor and anchor.** Around each point, a corridor (where the water surface is interpolated) and an anchor
   (where the flood starts) are drawn as circles. Their radii are the point's top width, clipped to
   `MPI_Min_TopWidth_m`-`MPI_Max_TopWidth_m`, times `MPI_Corridor_TopWidth_Factor` and `MPI_Anchor_TopWidth_Factor`,
   or fixed radii without `MPI_Use_TopWidth_Buffers`.
6. **Interpolation.** In the corridor, each cell's water surface is the inverse-distance-weighted average of its
   `MPI_K` nearest points, with weights $1 / d^{p}$ ($p$ = `MPI_Power`). A point reaches only as far as its reach's
   median top width times `MPI_MaxDist_TopWidth_Factor`, or `MPI_Max_Distance_m`.
7. **Smoothing.** The water surface is smoothed with a Gaussian of `MPI_Smooth_Sigma_Pixels` cells.
8. **Flood.** A corridor cell floods where the water surface is above the ground and it connects to the anchor
   through flooded cells: 4 or 8 neighbours, as `MPI_Connectivity` says. The stream cells themselves are always
   flooded.

If no points can be made from the tables, or with `Set_Depth`, there is one point per stream cell with the same water
surface and top width the kernel-weighted mapper uses.
