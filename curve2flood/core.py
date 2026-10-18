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
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import polars.selectors as cs
from pyproj import CRS, Geod
from numba import njit
from osgeo import gdal, osr, ogr

from shapely.geometry import shape
from scipy.spatial import cKDTree
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt, uniform_filter

from curve2flood import LOG
from curve2flood.config import Config, FLDPLN, KERNEL_WEIGHTED, MULTI_POINT, MAPPERS
from curve2flood.spreaders import (
    _make_fldpln_flood_map, create_kernel_weighted_spread_map, filter_outliers, compute_tw_multfact_scale, multi_point_interpolation
)

gdal.UseExceptions()

# GDAL compresses a raster's blocks in parallel with this creation option: eight
# times as fast on a 3600 x 3600 float raster, and the same file
_WRITE_THREADS = "NUM_THREADS=ALL_CPUS"

def read_manning_table(s_manning_path: str, land_cover: np.ndarray):
    """Convert a land-cover raster to Manning's n.

    The table is tab separated, with the land-cover code in the first column
    and Manning's n in the third.  Codes missing from the table get n = 0.
    """
    df = pd.read_csv(s_manning_path, sep='\t')

    # A lookup array is the fastest way to reclassify the raster.  It is float32,
    # as the velocity is, and an integer raster indexes it as it is, so no
    # 8-byte copy of the raster is made.
    idx = df.iloc[:, 0].astype(int).values
    lookup_array = np.zeros(idx.max() + 1)
    lookup_array[idx] = df.iloc[:, 2].values
    if land_cover.dtype.kind not in 'iu':
        land_cover = land_cover.astype(int)
    return lookup_array.astype(np.float32)[land_cover]


def create_velocity(config: Config, OutVEL, Depth_Array, LC_array, Slope_Array,
                           geotransform, projection, ncols, nrows,
                           Flood_Ensemble):
    """Write the velocity raster from Manning's equation, ``V = D^(2/3) S^(1/2) / n``.

    Each cell is treated as a wide rectangular channel: ``D`` is its flood
    depth, ``S`` the largest positive slope over the flow events
    (``Slope_Array``, NaN where there is none) and ``n`` comes from its land
    cover.  Flooded cells without a value take their nearest neighbour's.
    """
    VEL_Array = 1 / read_manning_table(config.lu_manning_n, LC_array)
    VEL_Array *= Depth_Array ** (2/3)
    VEL_Array *= Slope_Array ** (1/2)

    out_band_data = Flood_Flooded_Cells_in_Map(VEL_Array, Flood_Ensemble, eps=0.01)
    nodata_value = np.nan
    del VEL_Array

    driver = gdal.GetDriverByName("GTiff")
    ds: gdal.Dataset = driver.Create(
        OutVEL, ncols, nrows, 1, gdal.GDT_Float32,
        options=[f"COMPRESS={config.compression}", "PREDICTOR=2", "TILED=YES", _WRITE_THREADS]
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

@njit(cache=True)
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
    1st-99th percentile of top width or velocity are dropped.  WSE isn't
    filtered: it falls along a reach, so its extremes are the reach's two
    ends, and dropping them left gaps at every junction.
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

    # Drop each reach's outliers (not by WSE: see above)
    cols = ['TopWidth', 'Velocity']
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
        creation_options = [f"COMPRESS={compression}", 'PREDICTOR=2', _WRITE_THREADS]

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

def CreateSimpleKernelFloodMap(config: Config, RR, CC, T_Rast, W_Rast, S_Rast, E,
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
        config.localfloodoption,
        WeightBox,
        TW_for_WeightBox_ElipseMask,
        config.topwidthplausiblelimit,
        config.set_depth,
    )

    # Do not flood cells where WSE is below E and E/WSE are nan values
    mask = (WSE_array > E) & (E > -9998.0)
    Flooded_array = mask.astype(np.uint8, copy=False)

    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        Flooded_array[RR, CC] = 1

    Depth_array = None
    if config.outdep or config.outwse or S_Rast is not None:
        Depth_array = np.where(mask, WSE_array - E, np.nan).astype(np.float32, copy=False)[1:-1, 1:-1]

    Slope_array = None
    if S_Rast is not None:
        # Cells no stream cell reached have no weight, hence 0 / 0
        with np.errstate(divide='ignore', invalid='ignore'):
            Slope_divided_by_weight = Slope_Times_Weight / Total_Weight
        Slope_array = np.where(mask, Slope_divided_by_weight, np.nan).astype(np.float32, copy=False)
        Slope_array = np.where((Slope_array <= 0), 0.0002, Slope_array).astype(np.float32, copy=False)[1:-1, 1:-1]

    return Flooded_array[1:-1, 1:-1], Depth_array, Slope_array

def uniform_smoothing(mask: np.ndarray, values: np.ndarray):
    """Replace ``values``, in place, by their 3x3 mean over the cells where ``mask`` is True.

    Cells with no masked cell in their 3x3 window keep their value.
    """
    window_size = 3
    weighted = np.where(mask, values, 0)
    value_sum = uniform_filter(weighted, size=window_size, mode='nearest') * window_size
    count = uniform_filter(mask.astype(np.float32, copy=False), size=window_size, mode='nearest') * window_size
    np.divide(value_sum, count, out=values, where=count > 0)


def Create_Topobathy_Dataset(E: np.ndarray, Bathy: np.ndarray, Bathy_Use_Banks: bool) -> np.ndarray:
    """The DEM ``E`` with ARC's bathymetry ``Bathy`` burned in: ARC's bed
    elevation wherever it has one (finite), and the DEM elsewhere.  Unless
    ``Bathy_Use_Banks`` is set, the result is never above the DEM.

    ARC has already decided where its channel goes and filled the small gaps
    between its cross sections, so its raster is taken as it is: each rating
    curve's cell keeps the bed its curve was worked out on (the VDT's
    ``Elev``), and the channel keeps the banks its cross sections were carved
    with.
    """
    burned = np.where(np.isfinite(Bathy), Bathy, E).astype(np.float32, copy=False)
    if not Bathy_Use_Banks:
        np.minimum(burned, E, out=burned)
    return burned

def Calculate_Depth_TopWidth_TWMax_Velocity(config: Config, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, quiet):
    """Fill the per-cell rasters for one flow event and summarise each reach.

    Returns ``(COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast,
    W_Rast, S_Rast)``: the median top width and depth of each reach, and the
    largest top width, capped at ``TopWidthPlausibleLimit``.  With
    ``Set_Depth`` every reach gets that depth and the plausible-limit top
    width instead, and the rasters are left untouched.
    """
    Set_Depth = config.set_depth
    TopWidthPlausibleLimit = config.topwidthplausiblelimit
    if Set_Depth > 0.0:
        top_width = TopWidthPlausibleLimit if TopWidthPlausibleLimit > 0 else -9999.0
        COMID_Unique_TW = {comid: top_width for comid in COMID_Unique}
        COMID_Unique_Depth = {comid: Set_Depth for comid in COMID_Unique}
        TopWidthMax = TopWidthPlausibleLimit
    elif config.print_vdt_database:
        # The VDT database is preferred over the curve file when both are given
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_VDTDatabase(
            config.print_vdt_database, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, config.tw_multfact)
    else:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_CurveFile(
            config.print_curve_file, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, config.tw_multfact)

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
        config: Config,
        COMID_Unique_Flow,
        S_Rast,
        fill_below_dem,
        stream_info,
        flow_dir,
        fldpln_library,
        ):
    """FLDPLN mapper: ``(flood, depth, None)`` for one flow event.

    The VDT gives each stream cell's WSE for the event and the floodplain
    library spreads it (see ``_make_fldpln_flood_map``); neither the DEM nor
    the filled DEM is needed.  ``depth`` is None unless a depth, WSE or
    velocity raster is written.  When ``S_Rast`` is given, the stream cells'
    slopes are written into it, and when ``fill_below_dem`` is, how far the
    filled DEM sits below the DEM at each library cell (for ``water_surface``).
    """
    vdt_df = calculate_interpolated_vdt(
        config.print_vdt_database,
        COMID_Unique_Flow,
        config.tw_multfact
    )
    if S_Rast is not None:
        S_Rast[vdt_df['Row'].to_numpy(), vdt_df['Col'].to_numpy()] = vdt_df['Slope'].to_numpy()

    depth = _make_fldpln_flood_map(
        vdt_df,
        flow_dir,
        fldpln_library,
        stream_info,
        fill_below_dem=fill_below_dem,
    )

    Flood_array = (depth > 0).astype(np.uint8)
    # Depths are only used to write depth, WSE or velocity rasters
    Depth_array = depth if config.outdep or config.outwse or config.outvel else None

    return Flood_array, Depth_array, None  # Slopes are spread after all events are mapped

def Curve2Flood(config: Config, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique,
                COMID_Unique_Flow, WeightBox,
                TW_for_WeightBox_ElipseMask,
                quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast,
                flowdir,
                fill_below_dem,
                stream_info,
                fldpln_library,):
    """Map one flow event with the configured mapper.

    Returns ``(flood, depth, slope)`` on the DEM grid; ``depth`` and ``slope``
    are None when no output needs them.  ``E`` and ``B`` (DEM and stream ids)
    are padded by one cell, and ``RR``/``CC`` index their stream cells.  The
    FLDPLN mapper reads neither.
    """
    if config.mapper == FLDPLN:
        return make_fldpln_flood_map(
            config,
            COMID_Unique_Flow,
            S_Rast,
            fill_below_dem,
            stream_info,
            flowdir,
            fldpln_library,
        )

    (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax,
     T_Rast, W_Rast, S_Rast) = Calculate_Depth_TopWidth_TWMax_Velocity(
         config, COMID_Unique_Flow, COMID_Unique,
         T_Rast, W_Rast, S_Rast, quiet)

    LOG.info('Creating Rough Flood Map Data...')

    if config.mapper == KERNEL_WEIGHTED:
        return CreateSimpleKernelFloodMap(config, RR, CC, T_Rast, W_Rast, S_Rast,
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
        CurveParamFileName=config.print_curve_file,
        VDTDatabaseFileName=config.print_vdt_database,
        TW_MultFact=config.tw_multfact,
        dx=dx,
        dy=dy,
        TopWidthPlausibleLimit=config.topwidthplausiblelimit,
        Set_Depth=config.set_depth,
        flood_vdt_cells=flood_vdt_cells,
        mapper_options=config.multi_point_options,
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

def _nearest_sources(source_mask: np.ndarray, target_mask: np.ndarray, k: int = 8,
                     chunk: int = 1 << 16) -> tuple[np.ndarray, np.ndarray]:
    """The row and column of the nearest source cell to each target cell, the targets in ``np.nonzero`` order.

    Of two sources equally near, the one in the lower column wins, then the
    lower row: the way ``distance_transform_edt`` breaks ties.  Its feature
    transform sweeps the whole raster, so while the sources and targets are
    few (a flood covers a few percent of a tile) a KD-tree over the sources
    finds the same cells several times faster; it queries each target's ``k``
    nearest sources, a ``chunk`` of targets at a time.
    """
    tr, tc = np.nonzero(target_mask)
    sr, sc = np.nonzero(source_mask)
    if 10 * (sr.size + tr.size) > source_mask.size:
        # EDT returns indices of the nearest ZERO in the input,
        # so pass the inverse to point toward TRUE source cells.
        ny, nx = distance_transform_edt(~source_mask, return_distances=False, return_indices=True)
        return ny[tr, tc], nx[tr, tc]

    tree = cKDTree(np.column_stack((sr, sc)))
    k = min(k, sr.size)
    nearest = np.empty(tr.size, dtype=np.int64)
    for start in range(0, tr.size, chunk):
        rows, cols = tr[start:start + chunk], tc[start:start + chunk]
        _, index = tree.query(np.column_stack((rows, cols)), k=k)
        index = index.reshape(rows.size, k)
        # Exact squared distances, to find the ties
        d2 = (sr[index] - rows[:, None]) ** 2 + (sc[index] - cols[:, None]) ** 2
        least = d2.min(axis=1)
        tied = d2 == least[:, None]
        order = np.where(tied, (sc[index] << 32) + sr[index], np.iinfo(np.int64).max)
        nearest[start:start + rows.size] = index[np.arange(rows.size), order.argmin(axis=1)]
        # All k equally near: more sources may be as near, beyond the k
        if k < sr.size:
            for i in np.flatnonzero(tied.all(axis=1)):
                candidates = np.asarray(tree.query_ball_point((rows[i], cols[i]), np.sqrt(least[i]) + 1e-6))
                candidates = candidates[(sr[candidates] - rows[i]) ** 2 + (sc[candidates] - cols[i]) ** 2 == least[i]]
                nearest[start + i] = candidates[np.lexsort((sr[candidates], sc[candidates]))[0]]
    return sr[nearest], sc[nearest]

def Flood_Flooded_Cells_in_Map(Array_Ensemble: np.ndarray, Flood_Ensemble: np.ndarray, eps=0.01):
    """Match a value raster (depth, WSE, velocity, slope) to the flood extent.

    Flooded cells that are NaN take the value of the nearest flooded cell that
    has one (see ``_nearest_sources``); flooded cells with no such cell get
    ``eps``.  Cells outside the flood become NaN.  Returns a new float32 array.
    """
    flood_mask = Flood_Ensemble > 0
    nan_mask = np.isnan(Array_Ensemble)
    source_mask =  (flood_mask) & (~nan_mask)

    # targets are flooded cells that are currently NaN
    target_mask = (flood_mask) & (nan_mask)

    filled = Array_Ensemble.astype(np.float32, copy=True)

    if np.any(source_mask) and np.any(target_mask):
        filled[target_mask] = Array_Ensemble[_nearest_sources(source_mask, target_mask)]

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

def accumulate_positive_max(maximum: np.ndarray | None, array: np.ndarray | None) -> np.ndarray | None:
    """Raise ``maximum`` (float32, -inf where nothing is positive yet) to ``array``'s positive values, in place.

    Starts a new maximum when ``maximum`` is None; an ``array`` of None
    changes nothing.  Each flow event's raster is folded in as it is mapped,
    so the events' rasters are never all held at once.
    """
    if array is None:
        return maximum
    array = np.asarray(array, dtype=np.float32)
    positive = np.where(array > 0, array, np.float32(-np.inf))
    if maximum is None:
        return positive
    return np.maximum(maximum, positive, out=maximum)

def finish_positive_max(maximum: np.ndarray) -> np.ndarray:
    """``accumulate_positive_max``'s result with NaN, in place, where no value was positive."""
    maximum[maximum == -np.inf] = np.nan
    return maximum

def create_positive_max_array(array_list: list[np.ndarray]) -> np.ndarray:
    """Cell-wise maximum of the positive values in ``array_list``; NaN where there are none."""
    maximum = None
    for array in array_list:
        maximum = accumulate_positive_max(maximum, array)
    return finish_positive_max(maximum)

def create_depth(max_depth: np.ndarray, Flood_Ensemble: np.ndarray):
    """The maximum depth over the flow events (``create_positive_max_array``), matched to the flood extent (see
    ``Flood_Flooded_Cells_in_Map``)."""
    return Flood_Flooded_Cells_in_Map(max_depth, Flood_Ensemble, eps=0.01)


def water_surface(depth: np.ndarray, dem: np.ndarray, fill_below_dem: np.ndarray | None = None) -> np.ndarray:
    """The water surface elevation where the depth is above 0 and the DEM has a value, NaN elsewhere.

    The depth is measured from the DEM, except with the FLDPLN mapper, which measures it from the lower of the DEM
    and the filled DEM: where bathymetry raises the DEM above the filled DEM, the water was routed across the filled
    surface and its depth taken from there (see ``_make_fldpln_flood_map``), and the water surface must be too, or it
    would stand above its neighbours by the raise.  ``fill_below_dem`` is how far the filled DEM sits below the DEM
    there (0, or negative where raised; NaN where the library doesn't say), as the mapper writes it from the
    library's fill.  Only the flooded cells are worked out.
    """
    wse = np.full(depth.shape, np.nan, dtype=np.float32)
    flooded = np.nonzero((depth > 0) & (dem > -9998.0))
    ground = dem[flooded]
    if fill_below_dem is not None:
        below = fill_below_dem[flooded]
        ground = np.where(np.isnan(below), ground, ground + below)
    wse[flooded] = depth[flooded] + ground
    return wse


def create_bathymetry(config: Config, E: np.ndarray, nrows: int, ncols: int, dem_geotransform: tuple, dem_projection: str,
                      bathymetry_creation_options: list[str] = None):
    """Write the topobathymetric DEM (``FSOutBATHY``): the DEM with ARC's
    bathymetry (``BATHY_Out_File``) burned in (see ``Create_Topobathy_Dataset``).
    ``E`` is the DEM padded by one cell.
    """
    LOG.info('Working on Bathymetry')
    ds: gdal.Dataset = gdal.Open(config.bathy_out_file)
    band = ds.GetRasterBand(1)
    bathy = band.ReadAsArray().astype(np.float32, copy=False)
    nodata = band.GetNoDataValue()
    ds = None
    if bathy.shape != (nrows, ncols):
        raise ValueError(f"The bathymetry raster {config.bathy_out_file} is {bathy.shape[0]} by {bathy.shape[1]} "
                         f"cells, and the DEM {nrows} by {ncols}.")
    if nodata is not None:
        bathy[bathy == nodata] = np.nan

    burned = Create_Topobathy_Dataset(E[1:-1, 1:-1], bathy, config.bathy_use_banks)
    Write_Output_Raster(config.fsoutbathy, burned, ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32, config.compression, bathymetry_creation_options)

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
        config: Config,
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
    set_depth = config.uses_set_depth
    _check_dem_units(dem_projection)

    if config.mapper == FLDPLN:
        FlowDir = gdal.Open(config.flow_direction_file).ReadAsArray().astype(np.uint8, copy=False)
        _check_shape(FlowDir, (nrows, ncols), "Flow direction raster")

        if Path(config.stream_info_file).suffix == '.parquet':
            stream_info: pd.DataFrame = pd.read_parquet(config.stream_info_file)
        else:
            stream_info: pd.DataFrame = pd.read_csv(config.stream_info_file)

        # A Parquet library is read by its path, a row group at a time (see _make_fldpln_flood_map)
        if Path(config.fldpln_library).suffix == '.parquet':
            fldpln_library = config.fldpln_library
        else:
            fldpln_library = pl.scan_csv(config.fldpln_library)
        # How far the filled DEM sits below the DEM, from the library, for the water surface (see water_surface)
        fill_below_dem = np.full((nrows, ncols), np.nan, np.float32) if config.outwse else None
    else:
        FlowDir = None
        fill_below_dem = None
        stream_info = None
        fldpln_library = None

    LOG.info("Executing flood mapping logic...")

    # Stream cells come from the stream raster with Set_Depth, otherwise from the VDT or curve file
    if set_depth:
        S, _, _ = Read_Raster_GDAL(config.stream_file)
        _check_shape(S, (nrows, ncols), "Stream raster")
    elif config.print_vdt_database:
        S = Set_Stream_Locations(nrows, ncols, config.print_vdt_database)
    else:
        S = Set_Stream_Locations(nrows, ncols, config.print_curve_file)

    # Stream ids padded by one cell, and where the stream cells are
    B = np.zeros((nrows+2,ncols+2), dtype=np.int32)
    B[1:-1, 1:-1] = S

    (RR,CC) = np.where(B > 0)

    COMID_Unique = np.unique(B[RR, CC]) # Always sorted
    COMID_Unique = COMID_Unique.astype(int)
    if config.mapper == FLDPLN:
        B = None  # Only the curve-based mappers read it

    if set_depth:
        # Every stream cell gets the same depth, so there is a single event
        num_flows = 1
        COMID_Unique_Flow = {}
    else:
        LOG.info('Opening and Reading ' + config.comid_flow_file)
        num_flows = _count_flow_events(config.comid_flow_file)
    LOG.info('Evaluating ' + str(num_flows) + ' Flow Events')

    # Create initial rasters once, outside the loop
    if config.mapper == FLDPLN:
        T_Rast = None
        W_Rast = None
    else:
        T_Rast = np.empty((nrows,ncols), np.float32)
        W_Rast = np.empty((nrows,ncols), np.float32)

    # Stream-cell slopes, only needed for velocity
    S_Rast = np.full((nrows,ncols), np.nan, np.float32) if config.outvel else None

    Flood_Ensemble = np.zeros((nrows, ncols), dtype=np.float16 if num_flows > 1 else np.uint8)  # use uint8 if only one flow, otherwise float16 for percentage calculation
    # The largest depth and slope over the events so far (see accumulate_positive_max)
    Depth_Max = None
    Slope_Max = None
    for flow_event_num in range(num_flows):
        LOG.info('Working on Flow Event ' + str(flow_event_num))
        # clear out last events values
        if T_Rast is not None:
            T_Rast[:] = -1.0
        if W_Rast is not None:
            W_Rast[:] = np.nan

        if not set_depth:
            COMID_Unique_Flow = FindFlowRateForEachCOMID_Ensemble(config.comid_flow_file, flow_event_num)
        Flood_array_this_flow, Depth_array, Slope_array = Curve2Flood(
            config, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique,
            COMID_Unique_Flow, WeightBox, TW_for_WeightBox_ElipseMask,
            quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast, FlowDir,
            fill_below_dem, stream_info, fldpln_library
            )
        Flood_array_this_flow = remove_cells_not_connected(Flood_array_this_flow, S)
        Flood_Ensemble += Flood_array_this_flow
        Depth_Max = accumulate_positive_max(Depth_Max, Depth_array)
        Slope_Max = accumulate_positive_max(Slope_Max, Slope_array)
        del Flood_array_this_flow, Depth_array, Slope_array

    # clear some memory by deleting large arrays that are no longer needed
    del T_Rast
    del W_Rast

    # Turn the count of flooding events into a percentage
    Flood_Ensemble *= 100
    if num_flows > 1:
        Flood_Ensemble /= num_flows
        Flood_Ensemble = Flood_Ensemble.astype(np.uint8)

    OutVEL = config.outvel
    Flood_WaterLC_and_STRM_Cells = config.flood_waterlc_and_strm_cells
    OutDEP = config.outdep
    OutWSE = config.outwse
    Flood_File = config.outfld

    if Flood_WaterLC_and_STRM_Cells or OutVEL:
        LC_array, _, _ = Read_Raster_GDAL(config.lu_raster_sameres)
        _check_shape(LC_array, (nrows, ncols), "Land cover raster")

    if Flood_WaterLC_and_STRM_Cells:
        LOG.info('Flooding the Water-Related Land Cover and STRM cells')
        Flood_Ensemble = Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, config.land_watervalue)

    if config.flood_bathy_mask:
        LOG.info('Flooding the bathymetry water mask cells')
        mask, _, _ = Read_Raster_GDAL(config.bathywatermask)
        _check_shape(mask, (nrows, ncols), "Bathymetry water mask")
        Flood_Ensemble = Flood_Bathy_Mask_in_Flood_Map(Flood_Ensemble, mask)
        del mask

    # Remove crop circles and other disconnected cells
    Flood_Ensemble = remove_cells_not_connected(Flood_Ensemble, S)

    if Flood_File:
        LOG.info('Creating Ensemble Flood Map...' + str(Flood_File))
        out_ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(Flood_File, ncols, nrows, 1, gdal.GDT_Byte, options=[f"COMPRESS={config.compression}", "PREDICTOR=2", _WRITE_THREADS])
        out_ds.SetGeoTransform(dem_geotransform)
        out_ds.SetProjection(dem_projection)
        out_ds.WriteArray(Flood_Ensemble)
        out_ds.FlushCache()
        out_ds = None  # Close the dataset to ensure it's written to disk


    if OutDEP or OutWSE or OutVEL:
        Depth_Array = create_depth(finish_positive_max(Depth_Max), Flood_Ensemble)
        del Depth_Max

        if OutDEP:
            out_band_data = np.where(np.isnan(Depth_Array), -9999.0, Depth_Array).astype(np.float32, copy=False)

            ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(
                OutDEP, ncols, nrows, 1, gdal.GDT_Float32,
                options=[f"COMPRESS={config.compression}", "PREDICTOR=2", "TILED=YES", _WRITE_THREADS]
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
            del out_band_data

    if OutWSE:
        WSE_Array = water_surface(Depth_Array, E[1:-1, 1:-1], fill_below_dem)

        driver = gdal.GetDriverByName("GTiff")
        ds: gdal.Dataset = driver.Create(
            OutWSE, ncols, nrows, 1, gdal.GDT_Float32,
            options=[f"COMPRESS={config.compression}", "PREDICTOR=2", "TILED=YES", _WRITE_THREADS]
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
        del WSE_Array

    # The FLDPLN mapper gives no slopes: each flooded cell takes its nearest
    # stream cell's slope, averaged over its 3x3 neighbourhood
    if config.mapper == FLDPLN and S_Rast is not None:
        Slope_array = Flood_Flooded_Cells_in_Map(S_Rast, Flood_Ensemble.astype(np.uint8), eps=0.0002)
        valid = Slope_array > 0
        Slope_array[~valid] = np.nan
        uniform_smoothing(valid, Slope_array)
        Slope_Max = accumulate_positive_max(None, Slope_array)
        del Slope_array, valid

    if OutVEL:
        create_velocity(config, OutVEL, Depth_Array, LC_array, finish_positive_max(Slope_Max), dem_geotransform,
                        dem_projection, ncols, nrows, Flood_Ensemble)

    if config.strmshp_file and config.make_output_gpkg and Flood_File:
        flood_gdf = Write_Output_Raster_As_GeoDataFrame(Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, gdal.GDT_Byte)

        # The driver is inferred from the file extension
        shp_output_filename = f"{Flood_File[:-4]}.gpkg"
        flood_gdf.to_file(shp_output_filename)

    return Flood_Ensemble

def Curve2Flood_MainFunction(input_file: str | os.PathLike | Config = None,
                             args: dict = None,
                             quiet: bool = False,
                             flood_vdt_cells: bool = True,
                             bathymetry_creation_options: list[str] = None,
                             **kwargs):
    """Run Curve2Flood: map the flood events and write the requested outputs.

    Parameters
    ----------
    input_file : str, os.PathLike or Config, optional
        Path to a YAML or ``key value`` text input file, or the settings
        already read into a `Config`.  Used instead of ``args`` when given.
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
    if isinstance(input_file, Config):
        config = input_file
    elif input_file:
        config = Config.from_file(input_file)
    elif args:
        config = Config.from_mapping(args)
    else:
        raise ValueError("Provide an input file, a dict of settings or a Config.")

    LOG.info('Opening ' + config.dem_file)
    ds: gdal.Dataset = gdal.Open(config.dem_file)
    dem_geotransform = ds.GetGeoTransform()
    dem_projection = ds.GetProjection()
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    yll = dem_geotransform[3] - nrows * abs(dem_geotransform[5])
    yur = dem_geotransform[3]

    # The FLDPLN mapper needs the DEM's values only for a WSE raster or the bathymetry
    bathymetry = config.makes_bathymetry
    E = None
    if config.mapper != FLDPLN or config.outwse or bathymetry:
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
    TW_for_WeightBox_ElipseMask = int( max( np.round(config.topwidthplausiblelimit/dx,0), np.round(config.topwidthplausiblelimit/dy,0) ) )  # Half width of the weight box, in cells
    # Only the kernel-weighted and multi-point mappers spread with it
    WeightBox = None
    if config.needs_flood_map and config.mapper != FLDPLN:
        WeightBox = create_weightbox(TW_for_WeightBox_ElipseMask, dx, dy)

    if config.needs_flood_map:
        main_flood_ouputs(config, E, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, dx, dy,
                          dem_projection, dem_geotransform, quiet=quiet, flood_vdt_cells=flood_vdt_cells)

    if bathymetry:
        create_bathymetry(config, E, nrows, ncols, dem_geotransform, dem_projection, bathymetry_creation_options)

    LOG.info("Flood mapping completed.")
