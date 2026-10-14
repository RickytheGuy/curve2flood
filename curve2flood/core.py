"""Curve2Flood driver.

Reads the settings, maps the flood extent of each flow event with one of the
three mappers, and writes the requested rasters: flood extent, depth, water
surface elevation (WSE), velocity and an optional topobathymetric DEM.
"""
# built-in imports
import os
import json
from pathlib import Path

# third-party imports
import yaml
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import polars.selectors as cs
from pyproj import CRS, Geod
from numba import njit, prange
from osgeo import gdal, osr, ogr

from shapely.geometry import shape
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt, uniform_filter

from curve2flood import LOG
from curve2flood.spreaders import (
    _make_fldpln_flood_map, create_kernel_weighted_spread_map, filter_outliers, compute_tw_multfact_scale, multi_point_interpolation
)

gdal.UseExceptions()

# Values of the ``mapper`` setting
KERNEL_WEIGHTED = "Curve2Flood-Kernel Weighted"
MULTI_POINT = "Curve2Flood-Multi-Point Interpolation"
FLDPLN = "Curve2Flood-FLDPLNpy"
MAPPERS = (KERNEL_WEIGHTED, MULTI_POINT, FLDPLN)

def read_manning_table(s_manning_path: str, land_cover: np.ndarray):
    """Convert a land-cover raster to Manning's n.

    The table is tab separated, with the land-cover code in the first column
    and Manning's n in the third.  Codes missing from the table get n = 0.
    """
    df = pd.read_csv(s_manning_path, sep='\t')

    # A lookup array is the fastest way to reclassify the raster
    idx = df.iloc[:, 0].astype(int).values
    lookup_array = np.zeros(idx.max() + 1)
    lookup_array[idx] = df.iloc[:, 2].values
    return lookup_array[land_cover.astype(int)]


def create_velocity(params, OutVEL, Depth_Array, LC_array, Slope_array_list,
                           geotransform, projection, ncols, nrows,
                           Flood_Ensemble):
    """Write the velocity raster from Manning's equation, ``V = D^(2/3) S^(1/2) / n``.

    Each cell is treated as a wide rectangular channel: ``D`` is its flood
    depth, ``S`` the largest positive slope over the flow events and ``n``
    comes from its land cover.  Flooded cells without a value take their
    nearest neighbour's.
    """
    Slope_Array = create_positive_max_array(Slope_array_list)

    da_input_mannings = read_manning_table(params['LU_Manning_n'], LC_array).astype(np.float32)

    VEL_Array = (1/(da_input_mannings))*((Depth_Array)**(2/3))*((Slope_Array)**(1/2))

    VEL_Array = Flood_Flooded_Cells_in_Map(VEL_Array, Flood_Ensemble, eps=0.01)

    nodata_value = np.nan
    out_band_data = np.where(np.isnan(VEL_Array), nodata_value, VEL_Array).astype(np.float32)

    driver = gdal.GetDriverByName("GTiff")
    ds: gdal.Dataset = driver.Create(
        OutVEL, ncols, nrows, 1, gdal.GDT_Float32,
        options=[f"COMPRESS={params['compression']}", "PREDICTOR=2", "TILED=YES"]
    )
    if ds is None:
        raise RuntimeError(f"Failed to create output raster: {OutVEL}")

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    band = ds.GetRasterBand(1)
    band.WriteArray(out_band_data)
    band.SetNoDataValue(nodata_value)
    band.FlushCache()
    ds.FlushCache()

    band = None
    ds = None

def convert_cell_size(
    d_dem_cell_size_x: float,
    d_dem_cell_size_y: float,
    d_dem_lower_left: float,
    d_dem_upper_right: float,
    s_dem_projection: str
):
    """Return the DEM cell size in metres as ``(dx, dy)``.

    Geographic rasters (degrees) are converted with geodesic distances at the
    raster's middle latitude.  Projected rasters are already in metres and are
    returned unchanged.
    """
    d_x_cell_size = np.fabs(d_dem_cell_size_x)
    d_y_cell_size = np.fabs(d_dem_cell_size_y)

    try:
        o_crs = CRS.from_user_input(s_dem_projection)
    except Exception as e:
        raise ValueError("Unable to parse DEM projection for cell-size conversion.") from e

    if o_crs.is_geographic:
        d_lat = (d_dem_lower_left + d_dem_upper_right) / 2.0
        d_lon = 0.0  # Geodesic spacing does not depend on longitude

        # Build a geodesic calculator from the DEM ellipsoid.
        o_ellps = o_crs.ellipsoid
        if o_ellps is not None and o_ellps.semi_major_metre and o_ellps.inverse_flattening:
            o_geod = Geod(a=o_ellps.semi_major_metre, rf=o_ellps.inverse_flattening)
        else:
            o_geod = Geod(ellps="WGS84")

        _, _, d_y_metres = o_geod.inv(d_lon, d_lat, d_lon, d_lat + d_y_cell_size)
        _, _, d_x_metres = o_geod.inv(d_lon, d_lat, d_lon + d_x_cell_size, d_lat)
        d_x_cell_size = np.fabs(d_x_metres)
        d_y_cell_size = np.fabs(d_y_metres)

    return d_x_cell_size, d_y_cell_size

def _count_flow_events(FlowFileName: str) -> int:
    """Number of flow events (columns after the COMID column) in a flow file."""
    if FlowFileName.endswith('.parquet'):
        return pd.read_parquet(FlowFileName).shape[1] - 1
    return pd.read_csv(FlowFileName, nrows=0).shape[1] - 1

def FindFlowRateForEachCOMID_Ensemble(FlowFileName: str, flow_event_num: int) -> dict:
    """Return ``{COMID: flow}`` for one flow event.

    The flow file (CSV or Parquet) has the COMID in its first column and one
    column of flows per event.  Missing flows are treated as zero.
    """
    if FlowFileName.endswith('.parquet'):
        flow_df = pd.read_parquet(FlowFileName).iloc[:, [0, flow_event_num + 1]]
    else:
        flow_df = pd.read_csv(FlowFileName, usecols=[0, flow_event_num + 1])

    if flow_df.iloc[:, 1].isna().any():
        LOG.warning(f"Warning: NaN values found in flow data for event {flow_event_num}. These will be treated as zero flow.")
        flow_df.iloc[:, 1] = flow_df.iloc[:, 1].fillna(0)

    comid_q_dict = flow_df.set_index(flow_df.columns[0])[flow_df.columns[1]].to_dict()

    return comid_q_dict

def calculate_interpolated_curvefile(CurveParamFileName: str, COMID_Unique_Flow: dict, TW_MultFact: float) -> pd.DataFrame:
    """Evaluate the curve file's rating curves at each reach's flow.

    Depth, top width and velocity follow ``a * Q**b`` power laws, and WSE is
    the depth plus the cell's ``BaseElev``.  Cells more than two standard
    deviations from their reach's mean are dropped, and top widths are scaled
    by ``TW_MultFact`` (and by ``compute_tw_multfact_scale`` when the file has
    ``QBaseflow``).
    """
    LOG.debug('\nOpening and Reading ' + CurveParamFileName)

    if CurveParamFileName.endswith('.parquet'):
        curve_df = pd.read_parquet(CurveParamFileName)
    else:
        curve_df = pd.read_csv(CurveParamFileName)

    # Add COMID flow information
    comid_flow_df = pd.DataFrame(COMID_Unique_Flow.items(), columns=['COMID', 'Flow'])
    curve_df = curve_df.merge(comid_flow_df, on="COMID", how="left")

    curve_df['Depth'] = curve_df['depth_a']*curve_df['Flow']**curve_df['depth_b']
    curve_df['TopWidth'] = curve_df['tw_a']*curve_df['Flow']**curve_df['tw_b']
    curve_df['Velocity'] = curve_df['vel_a']*curve_df['Flow']**curve_df['vel_b']
    curve_df = curve_df[curve_df['Depth']>0]
    curve_df = curve_df[curve_df['TopWidth']>0]
    curve_df['WSE'] = curve_df['Depth'] + curve_df['BaseElev']

    curve_df = curve_df.groupby('COMID', group_keys=False)[curve_df.columns].apply(filter_outliers)

    if 'QBaseflow' in curve_df.columns:
        tw_scale = compute_tw_multfact_scale(
            curve_df['Flow'].values.astype(np.float32),
            curve_df['QBaseflow'].values.astype(np.float32),
        )
    else:
        tw_scale = np.ones(len(curve_df), dtype=np.float32)

    curve_df['TopWidth'] = curve_df['TopWidth'].astype(np.float32) * (TW_MultFact * tw_scale)

    return curve_df

def Calculate_TW_D_V_ForEachCOMID_CurveFile(CurveParamFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact):
    """Fill the top-width, WSE and slope rasters from the curve file for one event.

    Returns ``(COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast,
    W_Rast, S_Rast)``: each reach's median top width and depth (0 for reaches
    without curves), and the largest of those top widths.
    """
    curve_df = calculate_interpolated_curvefile(CurveParamFileName, COMID_Unique_Flow, TW_MultFact)

    T_Rast[curve_df['Row'], curve_df['Col']] = curve_df['TopWidth']
    W_Rast[curve_df['Row'], curve_df['Col']] = curve_df['WSE']
    if S_Rast is not None:
        S_Rast[curve_df['Row'], curve_df['Col']] = curve_df['Slope']

    median_values = curve_df.groupby('COMID').agg({'TopWidth': 'median', 'Depth': 'median'})

    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(median_values, on='COMID', how='left').fillna(0)

    comid_result_df['COMID'] = comid_result_df['COMID'].astype(np.int32)
    comid_result_df['TopWidth'] = comid_result_df['TopWidth'].astype(np.float32)
    comid_result_df['Depth'] = comid_result_df['Depth'].astype(np.float32)

    COMID_Unique_TW = comid_result_df.set_index('COMID')['TopWidth'].to_dict()
    COMID_Unique_Depth = comid_result_df.set_index('COMID')['Depth'].to_dict()

    TopWidthMax = comid_result_df['TopWidth'].max()

    return (
        COMID_Unique_TW,
        COMID_Unique_Depth,
        TopWidthMax,
        T_Rast,
        W_Rast,
        S_Rast,
    )

@njit("float32(float32, float32[:], float32[:])", cache=True)
def Find_TopWidth_at_Baseflow_when_using_VDT(QB, flow_values, top_width_values):
    """Top width at the first VDT flow that is at least the baseflow ``QB``.

    Returns the last top width when ``QB`` is above every flow.
    """
    idx = np.searchsorted(flow_values, QB, side="left")
    if idx >= len(flow_values):
        return top_width_values[-1]
    else:
        return top_width_values[idx]

@njit("float32(float32[:], float32[:], float32)", cache=True)
def interp1d_numba(x: np.ndarray, y: np.ndarray, xi: float | int) -> float:
    """Linearly interpolate ``y(x)`` at ``xi``, extrapolating from the end segments.

    ``x`` must be sorted.  Same as SciPy's ``interp1d(..., fill_value='extrapolate')``.
    """
    n = len(x)

    if xi <= x[0]:
        # extrapolate left
        return y[0] + (xi - x[0]) * (y[1] - y[0]) / (x[1] - x[0]) if (x[1] - x[0]) != 0 else y[0]
    elif xi >= x[n - 1]:
        # extrapolate right
        return y[n - 2] + (xi - x[n - 2]) * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) if (x[n - 1] - x[n - 2]) != 0 else y[n - 1]

    # binary search for correct interval
    low = 0
    high = n - 1
    while high - low > 1:
        mid = (high + low) // 2
        if x[mid] > xi:
            high = mid
        else:
            low = mid

    # linear interpolation
    x0 = x[low]
    x1 = x[high]
    y0 = y[low]
    y1 = y[high]

    return y0 + (xi - x0) * (y1 - y0) / (x1 - x0) if (x1 - x0) != 0 else y0

def vdt_interpolate(flow: np.ndarray,
                    qb: np.ndarray,
                    flow_values: np.ndarray,
                    top_width_values: np.ndarray,
                    wse_values: np.ndarray,
                    vel_values: np.ndarray,
                    elev: np.ndarray,
                    tw_mult_fact: np.ndarray) -> tuple[np.ndarray, ...]:
    """Interpolate each VDT row's curves at its flow.

    Returns ``(top_width, depth, wse, velocity)``.  Flows above a row's largest
    flow take its last values.  Smaller flows are interpolated linearly (and
    extrapolated below the first flow), with the WSE kept at or above the
    cell's elevation ``elev`` and the depth at least 1 mm.  Depths are
    measured from ``elev``.  The top width is at least the top
    width at baseflow ``qb``, and is then multiplied by ``tw_mult_fact``.
    """
    top_width = np.empty_like(flow)
    depth = np.empty_like(flow)
    wse = np.empty_like(flow)
    baseflow_tw = np.empty_like(flow)
    vel = np.empty_like(flow)

    for i in range(len(flow)):
        if flow[i] >= flow_values[i][-1]:
            # Above the maximum flow value
            top_width[i] = top_width_values[i][-1]
            wse[i] = wse_values[i][-1]
            depth[i] = wse[i] - elev[i]
            vel[i] = vel_values[i][-1]
        else:
            # Interpolate
            wse[i] = interp1d_numba(flow_values[i], wse_values[i], flow[i])
            top_width[i] = max(interp1d_numba(flow_values[i], top_width_values[i], flow[i]), 0.0)
            wse[i] = max(wse[i], elev[i])
            depth[i] = max(wse[i] - elev[i], 0.001)
            vel[i] = max(interp1d_numba(flow_values[i], vel_values[i], flow[i]), 0.0)

        baseflow_tw[i] = Find_TopWidth_at_Baseflow_when_using_VDT(qb[i], flow_values[i], top_width_values[i])

    # Ensure TopWidth respects baseflow and scale
    top_width = np.maximum(top_width, baseflow_tw) * tw_mult_fact

    return top_width, depth, wse, vel

def calculate_interpolated_vdt(
        VDTDatabaseFileName: str,
        COMID_Unique_Flow: dict,
        TW_MultFact: float,
        ) -> pl.DataFrame:
    """Interpolate the VDT database at each reach's flow.

    Returns one row per stream cell with ``TopWidth``, ``Depth``, ``WSE`` and
    ``Velocity`` (see ``vdt_interpolate``), measured from the VDT's ``Elev``,
    the channel bed under the cell.  Within each reach, cells outside the
    1st-99th percentile of top width, WSE or velocity are dropped.
    """
    LOG.debug('\nOpening and Reading ' + VDTDatabaseFileName)

    if VDTDatabaseFileName.endswith('.parquet'):
        vdt_df = pl.read_parquet(VDTDatabaseFileName)
    else:
        vdt_df = pl.read_csv(VDTDatabaseFileName)

    if vdt_df.is_empty():
        raise ValueError("The VDT Database file is empty or could not be read properly.")
    if 'Elev' not in vdt_df.columns:
        raise ValueError(f"The VDT database {VDTDatabaseFileName} has no Elev column.")

    # Add COMID flow information
    comid_flow_df = pl.DataFrame(data=list(COMID_Unique_Flow.items()), schema=['COMID', 'Flow'], orient='row')
    vdt_df = vdt_df.join(comid_flow_df, on='COMID', how='inner')

    vdt_df = vdt_df.with_columns([
        pl.col('Row').cast(pl.Int32),
        pl.col('Col').cast(pl.Int32)
    ])

    flow = vdt_df['Flow'].to_numpy().astype(np.float32, copy=False)
    qb = vdt_df['QBaseflow'].to_numpy().astype(np.float32, copy=False)
    elev = vdt_df['Elev'].to_numpy().astype(np.float32, copy=False)

    # The flow, top-width, WSE and velocity curves, one row per cell
    flow_values = vdt_df.select(cs.starts_with('q_')).to_numpy().astype(np.float32, copy=False)
    top_width_values = vdt_df.select(cs.starts_with('t_')).to_numpy().astype(np.float32, copy=False)
    wse_values = vdt_df.select(cs.starts_with('wse_')).to_numpy().astype(np.float32, copy=False)
    vel_values = vdt_df.select(cs.starts_with('v_')).to_numpy().astype(np.float32, copy=False)

    tw_scale = compute_tw_multfact_scale(flow, qb)
    tw_mult_fact = (TW_MultFact * tw_scale).astype(np.float32)
    top_width, depth, wse, velocity = vdt_interpolate(flow, qb, flow_values, top_width_values, wse_values, vel_values, elev, tw_mult_fact)

    # Add the interpolated values back to the DataFrame.  The ~120 curve columns
    # they came from are dropped first, or every filter below would copy them.
    vdt_df = vdt_df.drop(cs.starts_with('q_', 't_', 'wse_', 'v_')).with_columns([
        pl.Series('TopWidth', top_width),
        pl.Series('Depth', depth),
        pl.Series('WSE', wse),
        pl.Series('Velocity', velocity),
    ])

    vdt_df = vdt_df.drop_nans(subset=['TopWidth', 'Depth', 'WSE', 'Velocity'])

    vdt_df = vdt_df.with_columns(
        pl.col('TopWidth').round(2),
        pl.col('WSE').round(2),
        pl.col('Velocity').round(2),
    )

    # Drop each reach's outliers
    cols = ['TopWidth', 'WSE', 'Velocity']
    for col in cols:
        vdt_df = vdt_df.with_columns(
            pl.col(col).quantile(0.01).over("COMID").alias("q01"),
            pl.col(col).quantile(0.99).over("COMID").alias("q99"),
        ).filter(
            pl.col(col).is_between(pl.col("q01"), pl.col("q99"))
        ).drop("q01", "q99")

    return vdt_df

def Calculate_TW_D_V_ForEachCOMID_VDTDatabase(VDTDatabaseFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact):
    """Fill the top-width, WSE and slope rasters from the VDT database for one event.

    Returns ``(COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast,
    W_Rast, S_Rast)``: each reach's median top width and depth, and the
    largest of those top widths.
    """
    vdt_df = calculate_interpolated_vdt(VDTDatabaseFileName, COMID_Unique_Flow, TW_MultFact)

    T_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['TopWidth']
    W_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['WSE']
    if S_Rast is not None:
        S_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['Slope']

    median_values = vdt_df.group_by('COMID').agg([
        pl.col('TopWidth').median(),
        pl.col('Depth').median(),
    ])

    # Map results back to the unique COMID list
    comid_result_df = pl.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.join(median_values, on='COMID', how='left').fill_nan(0)
    comid_result_df = comid_result_df.with_columns(
        pl.col('COMID').cast(pl.Int32),
        pl.col('TopWidth').cast(pl.Float32),
        pl.col('Depth').cast(pl.Float32),
    )

    COMID_Unique_TW = dict(zip(comid_result_df['COMID'].to_numpy(), comid_result_df['TopWidth'].to_numpy()))
    COMID_Unique_Depth = dict(zip(comid_result_df['COMID'].to_numpy(), comid_result_df['Depth'].to_numpy()))

    TopWidthMax = comid_result_df['TopWidth'].max()

    return (
        COMID_Unique_TW,
        COMID_Unique_Depth,
        TopWidthMax,
        T_Rast,
        W_Rast,
        S_Rast
    )

def Read_Raster_GDAL(InRAST_Name):
    """Read band 1 of a raster as ``(array, geotransform, projection)``.

    South-up rasters (positive pixel height) are flipped to north-up.
    """
    dataset = gdal.Open(InRAST_Name, gdal.GA_ReadOnly)
    geotransform = dataset.GetGeoTransform()
    RastArray = dataset.GetRasterBand(1).ReadAsArray()
    nrows = RastArray.shape[0]

    if geotransform[5] > 0:
        LOG.warning('Raster appears south-up (positive pixel height); flipping to north-up: ' + str(InRAST_Name))
        RastArray = np.flipud(RastArray)
        geotransform = (
            geotransform[0],
            geotransform[1],
            geotransform[2],
            geotransform[3] + geotransform[5] * nrows,
            geotransform[4],
            -geotransform[5],
        )

    Rast_Projection = dataset.GetProjectionRef()
    dataset = None
    return RastArray, geotransform, Rast_Projection

def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type, compression, creation_options: list[str] = None):
    """Write a single-band raster.  ``creation_options`` replaces the default compression options."""
    o_driver = gdal.GetDriverByName(s_file_format)  #Typically will be a GeoTIFF "GTiff"

    if creation_options is None:
        creation_options = [f"COMPRESS={compression}", 'PREDICTOR=2']

    o_output_file = o_driver.Create(s_output_filename, xsize=ncols, ysize=nrows, bands=1, eType=s_output_type, options=creation_options)
    o_output_file.SetGeoTransform(dem_geotransform)
    o_output_file.SetProjection(dem_projection)
    o_output_file.GetRasterBand(1).WriteArray(raster_data)

    # Closing the dataset writes it to disk
    o_output_file = None


def Write_Output_Raster_As_GeoDataFrame(raster_data, ncols, nrows, dem_geotransform, dem_projection, s_output_type):
    """Polygonise a raster and return the polygons of its positive cells as a GeoDataFrame."""
    driver = gdal.GetDriverByName('MEM')
    raster_ds = driver.Create('', xsize=ncols, ysize=nrows, bands=1, eType=s_output_type)
    raster_ds.SetGeoTransform(dem_geotransform)
    raster_ds.SetProjection(dem_projection)
    raster_ds.GetRasterBand(1).WriteArray(raster_data)

    # Set NoData value to NaN, and auto-generate a mask band that respects it
    nodata_value = np.nan
    raster_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    mask_band = raster_ds.GetRasterBand(1).GetMaskBand()

    # Create an in-memory vector layer for the polygonized data
    memory_driver = (ogr.GetDriverByName('MEM') or ogr.GetDriverByName('Memory'))
    vector_ds = memory_driver.CreateDataSource('')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dem_projection)
    layer = vector_ds.CreateLayer('polygons', srs=srs)

    field = ogr.FieldDefn("Value", ogr.OFTInteger)
    layer.CreateField(field)

    gdal.Polygonize(raster_ds.GetRasterBand(1), mask_band, layer, 0, [], callback=None)

    polygons = []
    values = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        geom_dict = json.loads(geom.ExportToJson())
        polygons.append(shape(geom_dict))
        values.append(feature.GetField("Value"))

    flood_gdf = gpd.GeoDataFrame({'Value': values, 'geometry': polygons})
    flood_gdf.set_crs(dem_projection, inplace=True)

    # filter to only the flooded area
    flood_gdf = flood_gdf[flood_gdf['Value']>0]

    raster_ds = None
    vector_ds = None

    return flood_gdf

@njit("float32[:,:](int32, float32, float32)", cache=True)
def create_weightbox(tw: int, dx: float, dy: float):
    """Inverse squared distance weights for a ``(2*tw + 1)`` square window.

    ``tw`` is the window's half width in cells; the centre weight is capped at
    ``1 / 0.0001``.
    """
    n = 2*tw + 1
    indices = np.arange(n)
    # Squared distance (m^2) of each column and row from the centre cell
    X = ((indices - tw) * dx) ** 2
    Y = ((indices - tw) * dy) ** 2

    WeightBox = X[None, :] + Y[:, None]
    # Avoid very small values (to prevent division by zero)
    WeightBox = np.clip(WeightBox, 0.0001, None)
    WeightBox = 1.0 / WeightBox

    return (WeightBox).astype(np.float32)

def CreateSimpleKernelFloodMap(params: dict, RR, CC, T_Rast, W_Rast, S_Rast, E,
                         nrows, ncols, TW_m, dx, dy,
                         WeightBox, TW_for_WeightBox_ElipseMask, flood_vdt_cells,):
    """Kernel-weighted mapper: ``(flood, depth, slope)`` for one flow event.

    Each stream cell spreads its WSE to the cells up to one top width away,
    weighted by inverse squared distance, and a cell floods where the weighted
    WSE is above the DEM.  ``depth`` is None unless a depth, WSE or velocity
    raster is written, and ``slope`` is None unless a velocity raster is.
    """
    WSE_array, Total_Weight, Slope_Times_Weight = create_kernel_weighted_spread_map(
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
        params['LocalFloodOption'],
        WeightBox,
        TW_for_WeightBox_ElipseMask,
        params['TopWidthPlausibleLimit'],
        params['Set_Depth'],
    )

    # Do not flood cells where WSE is below E and E/WSE are nan values
    mask = (WSE_array > E) & (E > -9998.0)
    Flooded_array = mask.astype(np.uint8, copy=False)

    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        Flooded_array[RR, CC] = 1

    Depth_array = None
    if params['OutDEP'] or params['OutWSE'] or S_Rast is not None:
        Depth_array = np.where(mask, WSE_array - E, np.nan).astype(np.float32)[1:-1, 1:-1]

    Slope_array = None
    if S_Rast is not None:
        # Cells no stream cell reached have no weight, hence 0 / 0
        with np.errstate(divide='ignore', invalid='ignore'):
            Slope_divided_by_weight = Slope_Times_Weight / Total_Weight
        Slope_array = np.where(mask, Slope_divided_by_weight, np.nan).astype(np.float32)
        Slope_array = np.where((Slope_array <= 0), 0.0002, Slope_array).astype(np.float32)[1:-1, 1:-1]

    return Flooded_array[1:-1, 1:-1], Depth_array, Slope_array

@njit(cache=True, parallel=True)
def spread_Bathy(
    nrows: int,
    ncols: int,
    WeightBox: np.ndarray,
    TW_for_WeightBox_ElipseMask: int,
    Bathy: np.ndarray,
    ARBathyMask: np.ndarray
):
    """Weighted sums of the valid bathymetry around each cell of the water mask.

    Returns ``(bathy_times_weight, total_weight)``; their ratio is the
    inverse-distance-weighted bed elevation.  All arrays are padded by one cell.
    """
    bathy_times_weight = np.zeros_like(Bathy, dtype=np.float32)
    total_weight       = np.zeros_like(Bathy, dtype=np.float32)

    tw = TW_for_WeightBox_ElipseMask
    valid_donor = (Bathy > -98.99) & (ARBathyMask == 1)

    for r in prange(1, nrows + 1):
        for c in range(1, ncols + 1):
            if ARBathyMask[r, c] != 1:
                continue

            # window in Bathy space (same as the kernel-weighted mapper's)
            r_min = max(r - tw, 1)
            r_max = min(r + tw + 1, nrows + 1)
            c_min = max(c - tw, 1)
            c_max = min(c + tw + 1, ncols + 1)

            # corresponding window in WeightBox (centered on [tw, tw])
            w_r_min = tw - (r - r_min)
            w_c_min = tw - (c - c_min)

            acc_val = 0.0
            acc_w = 0.0

            for rr in range(r_min, r_max):
                wr = w_r_min + (rr - r_min)
                for cc in range(c_min, c_max):
                    if not valid_donor[rr, cc]:
                        continue
                    wc = w_c_min + (cc - c_min)
                    w = WeightBox[wr, wc]

                    acc_val += Bathy[rr, cc] * w
                    acc_w += w

            bathy_times_weight[r, c] = acc_val
            total_weight[r, c] = acc_w

    return bathy_times_weight, total_weight

def uniform_smoothing(mask: np.ndarray, values: np.ndarray):
    """Replace ``values``, in place, by their 3x3 mean over the cells where ``mask`` is True.

    Cells with no masked cell in their 3x3 window keep their value.
    """
    window_size = 3
    weighted = np.where(mask, values, 0)
    value_sum = uniform_filter(weighted, size=window_size, mode='nearest') * window_size
    count = uniform_filter(mask.astype(np.float32, copy=False), size=window_size, mode='nearest') * window_size
    np.divide(value_sum, count, out=values, where=count > 0)


def Create_Topobathy_Dataset(
    E: np.ndarray,
    nrows: int,
    ncols: int,
    WeightBox: np.ndarray,
    TW_for_WeightBox_ElipseMask: int,
    Bathy: np.ndarray,
    ARBathyMask: np.ndarray,
    Bathy_Use_Banks: bool
):
    """Blend ARC's bathymetry into the DEM, in place in ``Bathy``.

    Inside the water mask, cells without a bed elevation get the
    inverse-distance-weighted bed elevation of their neighbours (or the DEM if
    none is in reach).  Outside it the DEM is used.  Unless
    ``Bathy_Use_Banks`` is set, the result is never above the DEM.  ``E``,
    ``Bathy`` and ``ARBathyMask`` are padded by one cell.
    """
    # Outside the mask, missing bathymetry is the DEM before anything is spread
    mask = (Bathy < -98.99) & (ARBathyMask == 0)
    Bathy[mask] = E[mask]

    bathy_times_weight, total_weight = spread_Bathy(nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, Bathy, ARBathyMask)

    # Fill only the cells without bathymetry
    invalid = (Bathy <= -98.99) | np.isnan(Bathy)

    use_weight = invalid & (total_weight > 1e-10)
    use_dem    = invalid & ~use_weight

    Bathy[use_weight] = bathy_times_weight[use_weight] / total_weight[use_weight]
    Bathy[use_dem] = E[use_dem]

    # Outside the AR bathy mask, always use DEM
    mask = ARBathyMask != 1
    Bathy[mask] = E[mask]

    # Final safety net: any remaining bad values from DEM
    mask = (Bathy <= -98.99) | np.isnan(Bathy)
    Bathy[mask] = E[mask]

    if not Bathy_Use_Banks:
        np.minimum(Bathy, E, out=Bathy)

def Calculate_Depth_TopWidth_TWMax_Velocity(params: dict, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, quiet):
    """Fill the per-cell rasters for one flow event and summarise each reach.

    Returns ``(COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast,
    W_Rast, S_Rast)``: the median top width and depth of each reach, and the
    largest top width, capped at ``TopWidthPlausibleLimit``.  With
    ``Set_Depth`` every reach gets that depth and the plausible-limit top
    width instead, and the rasters are left untouched.
    """
    Set_Depth = params['Set_Depth']
    TopWidthPlausibleLimit = params['TopWidthPlausibleLimit']
    if Set_Depth > 0.0:
        top_width = TopWidthPlausibleLimit if TopWidthPlausibleLimit > 0 else -9999.0
        COMID_Unique_TW = {comid: top_width for comid in COMID_Unique}
        COMID_Unique_Depth = {comid: Set_Depth for comid in COMID_Unique}
        TopWidthMax = TopWidthPlausibleLimit
    elif params['VDTDatabaseFileName']:
        # The VDT database is preferred over the curve file when both are given
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_VDTDatabase(
            params['VDTDatabaseFileName'], COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, params['TW_MultFact'])
    else:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_CurveFile(
            params['CurveParamFileName'], COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, params['TW_MultFact'])

    LOG.info('Maximum Top Width = ' + str(TopWidthMax))

    if not quiet:
        for comid in COMID_Unique:
            if COMID_Unique_TW[comid]>TopWidthPlausibleLimit:
                LOG.warning(f"Reach {comid} (flow {COMID_Unique_Flow.get(comid)}, depth {COMID_Unique_Depth[comid]}) has a "
                            f"median top width of {COMID_Unique_TW[comid]}, above TopWidthPlausibleLimit; it is capped there.")

    if TopWidthPlausibleLimit < TopWidthMax:
        TopWidthMax = TopWidthPlausibleLimit

    return COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast

def make_fldpln_flood_map(
        params: dict,
        COMID_Unique_Flow,
        S_Rast,
        filled_dem,
        stream_info,
        flow_dir,
        fldpln_library,
        streams_gdf,
        ):
    """FLDPLN mapper: ``(flood, depth, None)`` for one flow event.

    The VDT gives each stream cell's WSE for the event and the floodplain
    library spreads it (see ``_make_fldpln_flood_map``); the DEM is not
    needed.  ``depth`` is None unless a depth, WSE or velocity raster is
    written.  When ``S_Rast`` is given, the stream cells' slopes are written
    into it.
    """
    vdt_df = calculate_interpolated_vdt(
        params['VDTDatabaseFileName'],
        COMID_Unique_Flow,
        params['TW_MultFact']
    )
    if S_Rast is not None:
        S_Rast[vdt_df['Row'].to_numpy(), vdt_df['Col'].to_numpy()] = vdt_df['Slope'].to_numpy()

    depth = _make_fldpln_flood_map(
        filled_dem,
        vdt_df,
        flow_dir,
        fldpln_library,
        stream_info,
        streams_gdf,
        params['reach_id_field'],
        params['downstream_reach_id_field'],
        max_wse_rise=params['FLDPLN_Max_WSE_Rise'],
        median_filter_size=params['FLDPLN_Median_Filter_Size'],
    )

    Flood_array = (depth > 0).astype(np.uint8)
    # Depths are only used to write depth, WSE or velocity rasters
    Depth_array = depth if params['OutDEP'] or params['OutWSE'] or params['OutVEL'] else None

    return Flood_array, Depth_array, None  # Slopes are spread after all events are mapped

def Curve2Flood(params: dict, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique,
                COMID_Unique_Flow, WeightBox,
                TW_for_WeightBox_ElipseMask,
                quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast,
                flowdir,
                filled_dem,
                stream_info,
                fldpln_library,
                streams_gdf,):
    """Map one flow event with the configured mapper.

    Returns ``(flood, depth, slope)`` on the DEM grid; ``depth`` and ``slope``
    are None when no output needs them.  ``E`` and ``B`` (DEM and stream ids)
    are padded by one cell, and ``RR``/``CC`` index their stream cells.  The
    FLDPLN mapper reads neither.
    """
    if params['mapper'] == FLDPLN:
        return make_fldpln_flood_map(
            params,
            COMID_Unique_Flow,
            S_Rast,
            filled_dem,
            stream_info,
            flowdir,
            fldpln_library,
            streams_gdf,
        )

    (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax,
     T_Rast, W_Rast, S_Rast) = Calculate_Depth_TopWidth_TWMax_Velocity(
         params, COMID_Unique_Flow, COMID_Unique,
         T_Rast, W_Rast, S_Rast, quiet)

    LOG.info('Creating Rough Flood Map Data...')

    if params['mapper'] == KERNEL_WEIGHTED:
        return CreateSimpleKernelFloodMap(params, RR, CC, T_Rast, W_Rast, S_Rast,
                                          E, nrows, ncols,
                                          TopWidthMax, dx, dy,
                                          WeightBox,
                                          TW_for_WeightBox_ElipseMask,
                                          flood_vdt_cells)

    Flood_array, Depth_array, Slope_array, stats_message = multi_point_interpolation(
        E=E,
        B=B,
        RR=RR,
        CC=CC,
        T_Rast=T_Rast,
        W_Rast=W_Rast,
        S_Rast=S_Rast,
        COMID_Unique_TW=COMID_Unique_TW,
        COMID_Unique_Depth=COMID_Unique_Depth,
        COMID_Unique_Flow=COMID_Unique_Flow,
        CurveParamFileName=params['CurveParamFileName'],
        VDTDatabaseFileName=params['VDTDatabaseFileName'],
        TW_MultFact=params['TW_MultFact'],
        dx=dx,
        dy=dy,
        TopWidthPlausibleLimit=params['TopWidthPlausibleLimit'],
        Set_Depth=params['Set_Depth'],
        flood_vdt_cells=flood_vdt_cells,
        mapper_options=params,
    )
    LOG.info(stats_message)

    return Flood_array, Depth_array, Slope_array


def Set_Stream_Locations(nrows: int, ncols: int, infilename: str):
    """Stream raster built from the ``Row``, ``Col`` and ``COMID`` columns of a VDT or curve file.

    Cells without a stream are -9999.
    """
    S = np.full((nrows, ncols), -9999, dtype=np.int32)
    if infilename.endswith('.parquet'):
        df = pd.read_parquet(infilename, columns=['Row', 'Col', 'COMID'])
    else:
        df = pd.read_csv(infilename, usecols=['Row', 'Col', 'COMID'])

    S[df['Row'].values, df['Col'].values] = df['COMID'].values

    return S

def Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, watervalue):
    """Mark stream cells and water land-cover cells as flooded (100) in ``Flood_Ensemble``."""
    # One boolean mask: int64 copies of three rasters made this the memory peak of a whole run
    F = (S > 0) | (LC_array == watervalue)
    Flood_Ensemble[F] = 100

    return Flood_Ensemble

def Flood_Bathy_Mask_in_Flood_Map(Flood_Ensemble, mask):
    """Mark the water mask's cells (above 0) as flooded (100) in ``Flood_Ensemble``."""
    Flood_Ensemble[mask > 0] = 100

    return Flood_Ensemble

def Flood_Flooded_Cells_in_Map(Array_Ensemble: np.ndarray, Flood_Ensemble: np.ndarray, eps=0.01):
    """Match a value raster (depth, WSE, velocity, slope) to the flood extent.

    Flooded cells that are NaN take the value of the nearest flooded cell that
    has one; flooded cells with no such cell get ``eps``.  Cells outside the
    flood become NaN.  Returns a new float32 array.
    """
    flood_mask = Flood_Ensemble > 0
    nan_mask = np.isnan(Array_Ensemble)
    source_mask =  (flood_mask) & (~nan_mask)

    # targets are flooded cells that are currently NaN
    target_mask = (flood_mask) & (nan_mask)

    filled = Array_Ensemble.astype(np.float32, copy=True)

    if np.any(source_mask):
        # EDT returns indices of the nearest ZERO in the input,
        # so pass the inverse to point toward TRUE source cells.
        ny, nx = distance_transform_edt(~source_mask, return_distances=False, return_indices=True)
        nn_vals = Array_Ensemble[ny, nx]
        filled[target_mask] = nn_vals[target_mask]

    # A flooded blob with no source anywhere is still NaN
    still_nan = (flood_mask) & np.isnan(filled)
    if np.any(still_nan):
        filled[still_nan] = eps

    filled[Flood_Ensemble <= 0] = np.nan

    return filled

def remove_cells_not_connected(flood_array: np.ndarray, streams_array: np.ndarray) -> np.ndarray:
    """Remove flooded patches that do not touch a stream.

    A patch is a group of cells with ``flood_array > 0`` connected through any
    of their 8 neighbours.  Patches containing a cell with ``streams_array > 0``
    are kept; the rest are set to 0.
    """
    structure = generate_binary_structure(2, 2)

    labeled_array, num_features = label(flood_array, structure)
    touching_labels = np.unique(labeled_array[(streams_array > 0) & (labeled_array > 0)])
    mask = np.isin(labeled_array, touching_labels)

    return flood_array * mask

def create_positive_max_array(array_list: list[np.ndarray]) -> np.ndarray:
    """Cell-wise maximum of the positive values in ``array_list``; NaN where there are none."""
    arr_stack = np.stack(array_list, axis=0, dtype=np.float32)
    result = np.nanmax(arr_stack, where=arr_stack > 0, axis=0, initial=-np.inf)
    result[result == -np.inf] = np.nan
    return result

def create_depth(
                array_list: list[np.ndarray],
                Flood_Ensemble: np.ndarray,
    ):
    """Maximum depth over the flow events, matched to the flood extent (see ``Flood_Flooded_Cells_in_Map``)."""
    max_vals = create_positive_max_array(array_list)
    return Flood_Flooded_Cells_in_Map(max_vals, Flood_Ensemble, eps=0.01)


def create_bathymetry(params: dict, E: np.ndarray, nrows: int, ncols: int, dem_geotransform: tuple, dem_projection: str,
                      Flood_Ensemble: np.ndarray, WeightBox: np.ndarray, TW_for_WeightBox_ElipseMask: int, bathymetry_creation_options: list[str] = None):
    """Write the topobathymetric DEM (``FSOutBATHY``).

    ARC's bed elevations (``BATHY_Out_File``) are used inside the water mask:
    the ``BathyWaterMask`` raster, or the flood map when that file does not
    exist.  See ``Create_Topobathy_Dataset``.
    """
    LOG.info('Working on Bathymetry')
    ds: gdal.Dataset = gdal.Open(params['BathyFromARFileName'])
    ARBathy = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)
    ARBathy[1:-1, 1:-1] = ds.ReadAsArray().astype(np.float32)
    ds = None

    ARBathyMask = np.zeros((nrows+2,ncols+2), dtype=np.bool_)
    if os.path.exists(params['BathyWaterMaskFileName']):
        ds = gdal.Open(params['BathyWaterMaskFileName'])
        ARBathyMask[1:-1, 1:-1] = ds.ReadAsArray() > 0
        ds = None
    else:
        ARBathyMask[1:-1, 1:-1] = Flood_Ensemble > 0

    ARBathy[np.isnan(ARBathy)] = -99.000  #This converts all nan values to a -99
    ARBathy = ARBathy * ARBathyMask
    ARBathy[ARBathyMask != 1] = -9999.000
    Create_Topobathy_Dataset(E, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, ARBathy, ARBathyMask, params['Bathy_Use_Banks'])

    Write_Output_Raster(params['BathyOutputFileName'], ARBathy[1:-1, 1:-1], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32, params['compression'], bathymetry_creation_options)

def to_bool(value):
    """``'False'`` (any case) is False; other strings, and other values by truthiness, as usual."""
    if isinstance(value, str):
        return not value.lower() == 'false'
    return bool(value)

def _first(data: dict, *keys, default=None):
    """Value of the first of ``keys`` present in ``data``."""
    for key in keys:
        if key in data:
            return data[key]
    return default

def _path(data: dict, *keys) -> str:
    """A file path setting as a string; '' when it is missing or empty."""
    value = _first(data, *keys, default='')
    return '' if value is None else str(value)

def get_params(input_file: str = None, args: dict = None):
    """Read the settings from an input file or a dict.

    The input file is YAML (``.yaml``/``.yml``) or plain text with one
    ``key value`` pair per line; lines starting with ``#`` are comments.
    Missing keys take their defaults and unknown keys are ignored.  Returns a
    dict keyed by the internal setting names.
    """
    if input_file:
        with open(input_file,'r') as infile:
            if str(input_file).lower().endswith(('.yaml', '.yml')):
                data = yaml.load(infile, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
            else:
                data = {}
                for line in infile.readlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key_value = line.split(None, 1)
                        if len(key_value) == 2:
                            key, value = key_value
                            data[key] = value
                        elif len(key_value) == 1:
                            data[key_value[0]] = ''
    elif args:
        data = args
    else:
        raise ValueError("Provide an input file or a dict of settings.")

    params = {
        'DEM_File': _path(data, 'DEM_File'),
        'STRM_File': _path(data, 'Stream_File'),
        'LAND_File': _path(data, 'LU_Raster_SameRes'),
        'StrmShp_File': _path(data, 'StrmShp_File'),
        'reach_id_field': _first(data, 'reach_id', 'Reach_ID', default='LINKNO'),
        'downstream_reach_id_field': _first(data, 'downstream_reach_id', 'Downstream_Reach_ID', default='DSLINKNO'),
        'Make_Output_GPKG': to_bool(data.get('Make_Output_GPKG', True)),
        'Flood_File': _path(data, 'OutFLD'),
        'OutDEP': _path(data, 'OutDEP'),
        'OutWSE': _path(data, 'OutWSE'),
        'OutVEL': _path(data, 'OutVEL'),
        'LU_Manning_n': _path(data, 'LU_Manning_n'),
        'FlowFileName': _path(data, 'COMID_Flow_File', 'Comid_Flow_File'),
        'VDTDatabaseFileName': _path(data, 'Print_VDT_Database'),
        'CurveParamFileName': _path(data, 'Print_Curve_File'),
        'mapper': data.get('mapper', KERNEL_WEIGHTED),
        'TopWidthPlausibleLimit': float(data.get('TopWidthPlausibleLimit', 1000.0)),
        'TW_MultFact': float(data.get('TW_MultFact', 3.0)),
        'Set_Depth': float(_first(data, 'Set_Depth', 'FloodSpreader_SpecifyDepth', default=-1.1)),
        'LocalFloodOption': to_bool(data.get('LocalFloodOption', False)) or to_bool(data.get('FloodLocalOnly', False)),
        'BathyWaterMaskFileName': _path(data, 'BathyWaterMask'),
        'BathyFromARFileName': _path(data, 'BATHY_Out_File'),
        'BathyOutputFileName': _path(data, 'FSOutBATHY'),
        'Flood_WaterLC_and_STRM_Cells': to_bool(data.get('Flood_WaterLC_and_STRM_Cells', False)),
        'Flood_Bathy_Mask': to_bool(data.get('Flood_Bathy_Mask', False)),
        'LAND_WaterValue': int(data.get('LAND_WaterValue', 80)),
        'Bathy_Use_Banks': to_bool(data.get('Bathy_Use_Banks', False)),

        # FLDPLN inputs
        'Flow_Direction_File': _path(data, 'Flow_Direction_File'),
        'Filled_DEM_File': _path(data, 'Filled_DEM_File'),
        'Stream_Info_File': _path(data, 'Stream_Info_File'),
        'FLDPLN_Library': _path(data, 'FLDPLN_Library'),
        'FLDPLN_Max_WSE_Rise': float(_first(data, 'max_wse_rise', 'FLDPLN_Max_WSE_Rise', default=0.01)),
        'FLDPLN_Median_Filter_Size': int(_first(data, 'FLDPLN_Median_Filter_Size', 'median_filter_size', default=53)),

        # Multipoint options
        'topwidth_threshold_m': float(data.get('MPI_TopWidth_Threshold_m', 200.0)),
        'xs_point_spacing_m': float(data.get('MPI_XS_Point_Spacing_m', 100.0)),
        'remove_hwm_outliers': to_bool(data.get('MPI_Remove_HWM_Outliers', True)),
        'hwm_outlier_method': data.get('MPI_HWM_Outlier_Method', 'mad'),
        'hwm_outlier_threshold': float(data.get('MPI_HWM_Outlier_Threshold', 3.5)),
        'hwm_outlier_group_by': data.get('MPI_HWM_Outlier_Group_By', 'comid'),
        'hwm_outlier_min_samples': int(data.get('MPI_HWM_Outlier_Min_Samples', 5)),
        'corridor_buffer_m': float(data.get('MPI_Corridor_Buffer_m', 500.0)),
        'anchor_buffer_m': float(data.get('MPI_Anchor_Buffer_m', 5.0)),
        'use_topwidth_buffers': to_bool(data.get('MPI_Use_TopWidth_Buffers', True)),
        'corridor_topwidth_factor': float(data.get('MPI_Corridor_TopWidth_Factor', 1.0)),
        'anchor_topwidth_factor': float(data.get('MPI_Anchor_TopWidth_Factor', 0.1)),
        'connectivity': int(data.get('MPI_Connectivity', 4)),
        'k': int(data.get('MPI_K', 12)),
        'power': float(data.get('MPI_Power', 2.0)),
        'max_distance_m': float(data.get('MPI_Max_Distance_m', 1000.0)),
        'use_topwidth_max_distance': to_bool(data.get('MPI_Use_TopWidth_Max_Distance', True)),
        'maxdist_topwidth_factor': float(data.get('MPI_MaxDist_TopWidth_Factor', 1.0)),
        'min_topwidth_m': float(data.get('MPI_Min_TopWidth_m', 5.0)),
        'max_topwidth_m': float(data.get('MPI_Max_TopWidth_m', 500.0)),
        'fallback_topwidth_m': float(data.get('MPI_Fallback_TopWidth_m', 20.0)),
        'smooth_sigma_pixels': float(data.get('MPI_Smooth_Sigma_Pixels', 0.25)),
        'apply_wse_sanity_filter': to_bool(data.get('MPI_Apply_WSE_Sanity_Filter', True)),
        'wse_sanity_tolerance_m': float(data.get('MPI_WSE_Sanity_Tolerance_m', 0.0)),
        'fast_mode': to_bool(data.get('MPI_Fast_Mode', False)),
        'one_based_vdt_rc': to_bool(data.get('MPI_One_Based_VDT_RC', False)),

        # Miscellaneous
        "compression": data.get("compression", "LZW"),
    }

    return params

def _needs_flood_map(params: dict) -> bool:
    """Whether a flood map is needed: for an output raster, or as the bathymetry water mask."""
    return bool(params['Flood_File'] or params['OutDEP'] or params['OutWSE'] or params['OutVEL'] or
                (params['BathyOutputFileName'] and params['BathyFromARFileName'] and not path_exists(params['BathyWaterMaskFileName'])))

def validate_params(params: dict):
    """Raise a ValueError if a setting the run needs is missing or two settings conflict.

    Messages use the input-file key names.
    """
    if params['mapper'] not in MAPPERS:
        raise ValueError(f"Unknown mapper {params['mapper']!r}; use one of: {', '.join(MAPPERS)}.")
    if not _needs_flood_map(params):
        return

    set_depth = params['Set_Depth'] > 0
    fldpln = params['mapper'] == FLDPLN
    if set_depth and fldpln:
        raise ValueError("Set_Depth (FloodSpreader_SpecifyDepth) is not supported by the FLDPLN mapper.")
    if set_depth and params['OutVEL']:
        raise ValueError("OutVEL needs the slopes in Print_VDT_Database or Print_Curve_File, "
                         "so it cannot be used with Set_Depth (FloodSpreader_SpecifyDepth).")

    required = {}  # input-file key -> its value
    if set_depth:
        required['Stream_File'] = params['STRM_File']
    else:
        required['COMID_Flow_File'] = params['FlowFileName']
        if fldpln:
            required['Print_VDT_Database'] = params['VDTDatabaseFileName']
        else:
            required['Print_VDT_Database or Print_Curve_File'] = params['VDTDatabaseFileName'] or params['CurveParamFileName']
    if fldpln:
        required.update({
            'Flow_Direction_File': params['Flow_Direction_File'],
            'Filled_DEM_File': params['Filled_DEM_File'],
            'Stream_Info_File': params['Stream_Info_File'],
            'FLDPLN_Library': params['FLDPLN_Library'],
            'StrmShp_File': params['StrmShp_File'],
        })
    if params['OutVEL']:
        required['LU_Manning_n'] = params['LU_Manning_n']
    if params['OutVEL'] or params['Flood_WaterLC_and_STRM_Cells']:
        required['LU_Raster_SameRes'] = params['LAND_File']
    if params['Flood_Bathy_Mask']:
        required['BathyWaterMask'] = params['BathyWaterMaskFileName']

    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")
    if params['Flood_Bathy_Mask'] and not path_exists(params['BathyWaterMaskFileName']):
        raise ValueError(f"Flood_Bathy_Mask floods the cells of BathyWaterMask, and {params['BathyWaterMaskFileName']} "
                         f"doesn't exist.")

def _check_dem_units(dem_projection: str):
    """Raise a ValueError unless the DEM's coordinate system is in metres or degrees."""
    try:
        raster_crs = CRS.from_wkt(dem_projection)
    except Exception as ex:
        raise ValueError(f'Unable to parse the DEM coordinate system: {ex}') from ex

    axis_units = {(axis.unit_name or '').strip().lower() for axis in raster_crs.axis_info if axis is not None}
    axis_units.discard('')
    if not axis_units:
        raise ValueError('Unable to determine the units of the DEM coordinate system.')

    invalid_units = sorted(axis_units - {'meter', 'meters', 'metre', 'metres', 'degree', 'degrees'})
    if invalid_units:
        raise ValueError(f'The DEM coordinate system must be in metres or degrees, not {", ".join(invalid_units)}.')

def _check_shape(array: np.ndarray, shape: tuple, name: str):
    """Raise a ValueError if a raster's shape differs from the DEM's."""
    if array.shape != shape:
        raise ValueError(f"{name} dimensions ({array.shape[1]}x{array.shape[0]}) do not match the DEM ({shape[1]}x{shape[0]}).")

def main_flood_ouputs(
        params: dict,
        E: np.ndarray | None,
        nrows: int,
        ncols: int,
        WeightBox: np.ndarray,
        TW_for_WeightBox_ElipseMask: int,
        dx: float,
        dy: float,
        dem_projection: str,
        dem_geotransform: tuple,
        quiet: bool = False,
        flood_vdt_cells: bool = False):
    """Map every flow event and write the flood, depth, WSE and velocity rasters.

    Returns the flood raster: 0 is dry, and 1-100 the percentage of flow events
    that flooded the cell (100 with a single event).  ``E`` is the DEM padded
    by one cell; the FLDPLN mapper needs it only for ``OutWSE``.
    """
    set_depth = params['Set_Depth'] > 0
    _check_dem_units(dem_projection)

    if params['mapper'] == FLDPLN:
        FlowDir = gdal.Open(params['Flow_Direction_File']).ReadAsArray().astype(np.uint8, copy=False)
        _check_shape(FlowDir, (nrows, ncols), "Flow direction raster")

        filled_dem: np.ndarray = gdal.Open(params['Filled_DEM_File']).ReadAsArray()
        _check_shape(filled_dem, (nrows, ncols), "Filled DEM raster")

        if Path(params['Stream_Info_File']).suffix == '.parquet':
            stream_info: pd.DataFrame = pd.read_parquet(params['Stream_Info_File'])
        else:
            stream_info: pd.DataFrame = pd.read_csv(params['Stream_Info_File'])
        if Path(params['StrmShp_File']).suffix == '.parquet':
            streams_gdf = gpd.read_parquet(params['StrmShp_File'])
        else:
            streams_gdf = gpd.read_file(params['StrmShp_File'], use_arrow=True, ignore_geometry=True)

        if Path(params['FLDPLN_Library']).suffix == '.parquet':
            fldpln_library = pl.scan_parquet(params['FLDPLN_Library'])
            fldpln_library = fldpln_library.with_columns(
                [pl.col(c).cast(pl.Float32)
                for c, d in fldpln_library.collect_schema().items() if d == pl.Decimal]
            )
        else:
            fldpln_library = pl.scan_csv(params['FLDPLN_Library'])
    else:
        FlowDir = None
        filled_dem = None
        stream_info = None
        streams_gdf = None
        fldpln_library = None

    LOG.info("Executing flood mapping logic...")

    # Stream cells come from the stream raster with Set_Depth, otherwise from the VDT or curve file
    if set_depth:
        S, _, _ = Read_Raster_GDAL(params['STRM_File'])
        _check_shape(S, (nrows, ncols), "Stream raster")
    elif params['VDTDatabaseFileName']:
        S = Set_Stream_Locations(nrows, ncols, params['VDTDatabaseFileName'])
    else:
        S = Set_Stream_Locations(nrows, ncols, params['CurveParamFileName'])

    # Stream ids padded by one cell, and where the stream cells are
    B = np.zeros((nrows+2,ncols+2), dtype=np.int32)
    B[1:-1, 1:-1] = S

    (RR,CC) = np.where(B > 0)

    COMID_Unique = np.unique(B[RR, CC]) # Always sorted
    COMID_Unique = COMID_Unique.astype(int)
    if params['mapper'] == FLDPLN:
        B = None  # Only the curve-based mappers read it

    if set_depth:
        # Every stream cell gets the same depth, so there is a single event
        num_flows = 1
        COMID_Unique_Flow = {}
    else:
        LOG.info('Opening and Reading ' + params['FlowFileName'])
        num_flows = _count_flow_events(params['FlowFileName'])
    LOG.info('Evaluating ' + str(num_flows) + ' Flow Events')

    # Create initial rasters once, outside the loop
    if params['mapper'] == FLDPLN:
        T_Rast = None
        W_Rast = None
    else:
        T_Rast = np.empty((nrows,ncols), np.float32)
        W_Rast = np.empty((nrows,ncols), np.float32)

    # Stream-cell slopes, only needed for velocity
    S_Rast = np.full((nrows,ncols), np.nan, np.float32) if params['OutVEL'] else None

    Flood_Ensemble = np.zeros((nrows, ncols), dtype=np.float16 if num_flows > 1 else np.uint8)  # use uint8 if only one flow, otherwise float16 for percentage calculation
    Depth_array_list = []
    Slope_array_list = []
    for flow_event_num in range(num_flows):
        LOG.info('Working on Flow Event ' + str(flow_event_num))
        # clear out last events values
        if T_Rast is not None:
            T_Rast[:] = -1.0
        if W_Rast is not None:
            W_Rast[:] = np.nan

        if not set_depth:
            COMID_Unique_Flow = FindFlowRateForEachCOMID_Ensemble(params['FlowFileName'], flow_event_num)
        Flood_array_this_flow, Depth_array, Slope_array = Curve2Flood(
            params, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique,
            COMID_Unique_Flow, WeightBox, TW_for_WeightBox_ElipseMask,
            quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast, FlowDir,
            filled_dem, stream_info, fldpln_library, streams_gdf
            )
        Flood_array_this_flow = remove_cells_not_connected(Flood_array_this_flow, S)
        Flood_Ensemble += Flood_array_this_flow
        Depth_array_list.append(Depth_array)
        Slope_array_list.append(Slope_array)

    # clear some memory by deleting large arrays that are no longer needed
    del T_Rast
    del W_Rast

    # Turn the count of flooding events into a percentage
    Flood_Ensemble *= 100
    if num_flows > 1:
        Flood_Ensemble /= num_flows
        Flood_Ensemble = Flood_Ensemble.astype(np.uint8)

    OutVEL = params['OutVEL']
    Flood_WaterLC_and_STRM_Cells = params['Flood_WaterLC_and_STRM_Cells']
    OutDEP = params['OutDEP']
    OutWSE = params['OutWSE']
    Flood_File = params['Flood_File']

    if Flood_WaterLC_and_STRM_Cells or OutVEL:
        LC_array, _, _ = Read_Raster_GDAL(params['LAND_File'])
        _check_shape(LC_array, (nrows, ncols), "Land cover raster")

    if Flood_WaterLC_and_STRM_Cells:
        LOG.info('Flooding the Water-Related Land Cover and STRM cells')
        Flood_Ensemble = Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, params['LAND_WaterValue'])

    if params['Flood_Bathy_Mask']:
        LOG.info('Flooding the bathymetry water mask cells')
        mask, _, _ = Read_Raster_GDAL(params['BathyWaterMaskFileName'])
        _check_shape(mask, (nrows, ncols), "Bathymetry water mask")
        Flood_Ensemble = Flood_Bathy_Mask_in_Flood_Map(Flood_Ensemble, mask)
        del mask

    # Remove crop circles and other disconnected cells
    Flood_Ensemble = remove_cells_not_connected(Flood_Ensemble, S)

    if Flood_File:
        LOG.info('Creating Ensemble Flood Map...' + str(Flood_File))
        out_ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(Flood_File, ncols, nrows, 1, gdal.GDT_Byte, options=[f"COMPRESS={params['compression']}", "PREDICTOR=2"])
        out_ds.SetGeoTransform(dem_geotransform)
        out_ds.SetProjection(dem_projection)
        out_ds.WriteArray(Flood_Ensemble)
        out_ds.FlushCache()
        out_ds = None  # Close the dataset to ensure it's written to disk


    if OutDEP or OutWSE or OutVEL:
        Depth_Array = create_depth(Depth_array_list, Flood_Ensemble)

        if OutDEP:
            out_band_data = np.where(np.isnan(Depth_Array), -9999.0, Depth_Array).astype(np.float32)

            ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(
                OutDEP, ncols, nrows, 1, gdal.GDT_Float32,
                options=[f"COMPRESS={params['compression']}", "PREDICTOR=2", "TILED=YES"]
            )
            if ds is None:
                raise RuntimeError(f"Failed to create output raster: {OutDEP}")

            ds.SetGeoTransform(dem_geotransform)
            ds.SetProjection(dem_projection)

            band: gdal.Band = ds.GetRasterBand(1)
            band.WriteArray(out_band_data)
            band.SetNoDataValue(-9999.0)
            band.FlushCache()
            ds.FlushCache()

            band = None
            ds = None

    if OutWSE:
        WSE_Array = np.where((Depth_Array > 0) & (E[1:-1, 1:-1] > -9998.0), Depth_Array+E[1:-1, 1:-1], np.nan).astype(np.float32)

        driver = gdal.GetDriverByName("GTiff")
        ds: gdal.Dataset = driver.Create(
            OutWSE, ncols, nrows, 1, gdal.GDT_Float32,
            options=[f"COMPRESS={params['compression']}", "PREDICTOR=2", "TILED=YES"]
        )
        if ds is None:
            raise RuntimeError(f"Failed to create output raster: {OutWSE}")

        ds.SetGeoTransform(dem_geotransform)
        ds.SetProjection(dem_projection)

        band = ds.GetRasterBand(1)
        band.WriteArray(WSE_Array)
        band.SetNoDataValue(np.nan)
        band.FlushCache()
        ds.FlushCache()

        band = None
        ds = None

    # The FLDPLN mapper gives no slopes: each flooded cell takes its nearest
    # stream cell's slope, averaged over its 3x3 neighbourhood
    if params['mapper'] == FLDPLN and S_Rast is not None:
        Slope_array = Flood_Flooded_Cells_in_Map(S_Rast, Flood_Ensemble.astype(np.uint8), eps=0.0002)
        valid = Slope_array > 0
        Slope_array[~valid] = np.nan
        uniform_smoothing(valid, Slope_array)
        Slope_array_list = [Slope_array]

    if OutVEL:
        create_velocity(params, OutVEL, Depth_Array, LC_array, Slope_array_list, dem_geotransform, dem_projection, ncols, nrows, Flood_Ensemble)

    if params['StrmShp_File'] and params['Make_Output_GPKG'] and Flood_File:
        flood_gdf = Write_Output_Raster_As_GeoDataFrame(Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, gdal.GDT_Byte)

        # The driver is inferred from the file extension
        shp_output_filename = f"{Flood_File[:-4]}.gpkg"
        flood_gdf.to_file(shp_output_filename)

    return Flood_Ensemble

def path_exists(path: str | os.PathLike) -> bool:
    """Check if a given path exists."""
    return path and Path(path).exists()

def Curve2Flood_MainFunction(input_file: str = None,
                             args: dict = None,
                             quiet: bool = False,
                             flood_vdt_cells: bool = True,
                             bathymetry_creation_options: list[str] = None,
                             **kwargs):
    """Run Curve2Flood: map the flood events and write the requested outputs.

    Parameters
    ----------
    input_file : str, optional
        Path to a YAML or ``key value`` text input file.  Used instead of
        ``args`` when given.
    args : dict, optional
        The same settings as a dict.  Paths may be strings or ``Path`` objects.
    quiet : bool
        Skip the per-reach warnings about top widths above
        ``TopWidthPlausibleLimit``.
    flood_vdt_cells : bool
        Always mark the stream cells as flooded (kernel-weighted and
        multi-point mappers).
    bathymetry_creation_options : list[str], optional
        GDAL creation options for the bathymetry raster.
    **kwargs
        Ignored; kept so older calls keep working.
    """
    params = get_params(input_file, args)
    validate_params(params)

    LOG.info('Opening ' + params['DEM_File'])
    ds: gdal.Dataset = gdal.Open(params['DEM_File'])
    dem_geotransform = ds.GetGeoTransform()
    dem_projection = ds.GetProjection()
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    yll = dem_geotransform[3] - nrows * abs(dem_geotransform[5])
    yur = dem_geotransform[3]

    # The FLDPLN mapper needs the DEM's values only for a WSE raster or the bathymetry
    bathymetry = bool(params['BathyFromARFileName'] and params['BathyOutputFileName'])
    E = None
    if params['mapper'] != FLDPLN or params['OutWSE'] or bathymetry:
        # The DEM padded by one cell of -9999, so neighbour lookups need no bounds checks
        E = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)
        E[1:-1, 1:-1] = ds.ReadAsArray()
    ds = None

    # Cell size from the geotransform; the vector magnitudes also handle rotated grids
    dem_cell_size_x = np.hypot(dem_geotransform[1], dem_geotransform[2])
    dem_cell_size_y = np.hypot(dem_geotransform[4], dem_geotransform[5])
    dx, dy = convert_cell_size(dem_cell_size_x, dem_cell_size_y, yll, yur, dem_projection)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))

    LOG.info('Creating the Weight Box')
    TW_for_WeightBox_ElipseMask = int( max( np.round(params['TopWidthPlausibleLimit']/dx,0), np.round(params['TopWidthPlausibleLimit']/dy,0) ) )  # Half width of the weight box, in cells
    if params['mapper'] == FLDPLN and not bathymetry:
        WeightBox = None
    else:
        WeightBox = create_weightbox(TW_for_WeightBox_ElipseMask, dx, dy)

    Flood_Ensemble = None
    if _needs_flood_map(params):
        Flood_Ensemble = main_flood_ouputs(params, E, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, dx, dy,
                                           dem_projection, dem_geotransform, quiet=quiet, flood_vdt_cells=flood_vdt_cells)

    if bathymetry:
        create_bathymetry(params, E, nrows, ncols, dem_geotransform, dem_projection,
                           Flood_Ensemble, WeightBox, TW_for_WeightBox_ElipseMask,
                          bathymetry_creation_options)

    LOG.info("Flood mapping completed.")
