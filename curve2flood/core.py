
#This code looks at a DEM raster to find the dimensions, then writes a script to create a STRM raster.
# built-in imports
import gc
import json
import math
import sys
import os
from datetime import datetime

# third-party imports
try:
    import gdal 
    import osr 
    import ogr
    #from gdalconst import GA_ReadOnly
except: 
    from osgeo import gdal
    from osgeo import osr
    from osgeo import ogr
    #from osgeo.gdalconst import GA_ReadOnly

from numba import njit, prange
from numba.core import types
from numba.typed import Dict
    
import numpy as np
import pandas as pd
import geopandas as gpd
# from scipy.interpolate import interp1d
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt
from shapely.geometry import shape
import rasterio
from rasterio.features import rasterize
from curve2flood import LOG

gdal.UseExceptions()

COMID_FLOW_DICT_TYPE = dict[np.int32, np.float32]

def read_manning_table(s_manning_path: str, da_input_mannings: np.ndarray):
    """
    Reads the Manning's n information from the input file

    Parameters
    ----------
    s_manning_path: str
        Path to the Manning's n input table
    da_input_mannings: ndarray
        Array holding the mannings estimates

    Returns
    -------
    da_input_mannings: ndarray
        Array holding the mannings estimates

    """

    # Open and read the input file
    df = pd.read_csv(s_manning_path, sep='\t')

    # Create a lookup array for the Manning's n values
    # This is the fastest way to reclassify the values in the input array
    idx = df.iloc[:, 0].astype(int).values
    lookup_array = np.zeros(idx.max() + 1)
    lookup_array[idx] = df.iloc[:, 2].values
    da_input_mannings = lookup_array[da_input_mannings.astype(int)]
    # Return to the calling function
    return da_input_mannings

from scipy import ndimage as ndi

def create_velocity(OutVEL, Depth_Array, LU_Manning_n, LC_array, Slope_array_list,
                           geotransform, projection, ncols, nrows,
                           Flood_Ensemble, S):
    """
    """

    # find the maximum slope across ensembles, ignoring NaNs when calculating an average with NaNs mixed in
    # Stack them into one 3D array
    Slope_Array = create_positive_max_array(Slope_array_list)

    # Read Manning's n raster ---
    da_input_mannings = read_manning_table(LU_Manning_n, LC_array).astype(np.float32)

    # use Manning's solutions that assumes each pixel is a rectangular channel
    VEL_Array = (1/(da_input_mannings))*((Depth_Array)**(2/3))*((Slope_Array)**(1/2))

    VEL_Array = Flood_Flooded_Cells_in_Map(VEL_Array, Flood_Ensemble, eps=0.01)

    nodata_value = np.nan
    out_band_data = np.where(np.isnan(VEL_Array), nodata_value, VEL_Array).astype(np.float32)

    driver = gdal.GetDriverByName("GTiff")
    ds: gdal.Dataset = driver.Create(
        OutVEL, ncols, nrows, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"]
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

    return



def convert_cell_size(dem_cell_size, dem_lower_left, dem_upper_right):
    """
    Determines the x and y cell sizes based on the geographic location

    Parameters
    ----------
    None. All input data is available in the parent object

    Returns
    -------
    None. All output data is set into the object

    """

    ### Get the cell size ###
    d_lat = np.fabs((dem_lower_left + dem_upper_right) / 2)

    ### Determine if conversion is needed
    if dem_cell_size > 0.5:
        # This indicates that the DEM is projected, so no need to convert from geographic into projected.
        x_cell_size = dem_cell_size
        y_cell_size = dem_cell_size
        projection_conversion_factor = 1

    else:
        # Reprojection from geographic coordinates is needed
        assert d_lat > 1e-16, "Please use lat and long values greater than or equal to 0."

        # Determine the latitude range for the model
        if d_lat >= 0 and d_lat <= 10:
            d_lat_up = 110.61
            d_lat_down = 110.57
            d_lon_up = 109.64
            d_lon_down = 111.32
            d_lat_base = 0.0

        elif d_lat > 10 and d_lat <= 20:
            d_lat_up = 110.7
            d_lat_down = 110.61
            d_lon_up = 104.64
            d_lon_down = 109.64
            d_lat_base = 10.0

        elif d_lat > 20 and d_lat <= 30:
            d_lat_up = 110.85
            d_lat_down = 110.7
            d_lon_up = 96.49
            d_lon_down = 104.65
            d_lat_base = 20.0

        elif d_lat > 30 and d_lat <= 40:
            d_lat_up = 111.03
            d_lat_down = 110.85
            d_lon_up = 85.39
            d_lon_down = 96.49
            d_lat_base = 30.0

        elif d_lat > 40 and d_lat <= 50:
            d_lat_up = 111.23
            d_lat_down = 111.03
            d_lon_up = 71.70
            d_lon_down = 85.39
            d_lat_base = 40.0

        elif d_lat > 50 and d_lat <= 60:
            d_lat_up = 111.41
            d_lat_down = 111.23
            d_lon_up = 55.80
            d_lon_down = 71.70
            d_lat_base = 50.0

        elif d_lat > 60 and d_lat <= 70:
            d_lat_up = 111.56
            d_lat_down = 111.41
            d_lon_up = 38.19
            d_lon_down = 55.80
            d_lat_base = 60.0

        elif d_lat > 70 and d_lat <= 80:
            d_lat_up = 111.66
            d_lat_down = 111.56
            d_lon_up = 19.39
            d_lon_down = 38.19
            d_lat_base = 70.0

        elif d_lat > 80 and d_lat <= 90:
            d_lat_up = 111.69
            d_lat_down = 111.66
            d_lon_up = 0.0
            d_lon_down = 19.39
            d_lat_base = 80.0

        else:
            raise AttributeError('Please use legitimate (0-90) lat and long values.')

        ## Convert the latitude ##
        d_lat_conv = d_lat_down + (d_lat_up - d_lat_down) * (d_lat - d_lat_base) / 10
        y_cell_size = dem_cell_size * d_lat_conv * 1000.0  # Converts from degrees to m

        ## Longitude Conversion ##
        d_lon_conv = d_lon_down + (d_lon_up - d_lon_down) * (d_lat - d_lat_base) / 10
        x_cell_size = dem_cell_size * d_lon_conv * 1000.0  # Converts from degrees to m

        ## Make sure the values are in bounds ##
        if d_lat_conv < d_lat_down or d_lat_conv > d_lat_up or d_lon_conv < d_lon_up or d_lon_conv > d_lon_down:
            raise ArithmeticError("Problem in conversion from geographic to projected coordinates")

        ## Calculate the conversion factor ##
        projection_conversion_factor = 1000.0 * (d_lat_conv + d_lon_conv) / 2.0
    return x_cell_size, y_cell_size, projection_conversion_factor

def FindFlowRateForEachCOMID_Ensemble(FlowFileName: str, flow_event_num: int) -> dict:  
    if FlowFileName.endswith('.parquet'):
        flow_df = pd.read_parquet(FlowFileName)
    else:
        flow_df = pd.read_csv(FlowFileName, usecols=[0, flow_event_num + 1])

    comid_q_dict = flow_df.set_index(flow_df.columns[0])[flow_df.columns[1]].to_dict()
    
    return comid_q_dict


def filter_outliers(group):
    """
    A function to filter outliers based on mean and standard deviation for each COMID group
    """
    # Calculate mean and standard deviation for TopWidth
    topwidth_mean = group['TopWidth'].mean()
    topwidth_std = group['TopWidth'].std()
    lower_bound_tw = topwidth_mean - 2 * topwidth_std
    upper_bound_tw = topwidth_mean + 2 * topwidth_std

    # Filter TopWidth outliers
    group = group[(group['TopWidth'] >= lower_bound_tw) & (group['TopWidth'] <= upper_bound_tw)]

    # Calculate mean and standard deviation for WSE
    wse_mean = group['WSE'].mean()
    wse_std = group['WSE'].std()
    lower_bound_wse = wse_mean - 2 * wse_std
    upper_bound_wse = wse_mean + 2 * wse_std

    # Filter WSE outliers
    group = group[(group['WSE'] >= lower_bound_wse) & (group['WSE'] <= upper_bound_wse)]

    # Calculate mean and standard deviation for Velocity
    wse_mean = group['Velocity'].mean()
    wse_std = group['Velocity'].std()
    lower_bound_wse = wse_mean - 2 * wse_std
    upper_bound_wse = wse_mean + 2 * wse_std

    # Filter WSE outliers
    group = group[(group['Velocity'] >= lower_bound_wse) & (group['Velocity'] <= upper_bound_wse)]

    return group

def Calculate_TW_D_V_ForEachCOMID_CurveFile(CurveParamFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact):

    LOG.debug('\nOpening and Reading ' + CurveParamFileName)

    # read the curve data in as a Pandas dataframe
    if CurveParamFileName.endswith('.parquet'):
        curve_df = pd.read_parquet(CurveParamFileName)
    else:
        curve_df = pd.read_csv(CurveParamFileName)

    # Add COMID flow information
    comid_flow_df = pd.DataFrame(COMID_Unique_Flow.items(), columns=['COMID', 'Flow'])

    # merging the curve and streamflow data together
    curve_df = curve_df.merge(comid_flow_df, on="COMID", how="left")

    # calculating depth and top-width with the COMID's discharge and the curve parameters
    curve_df['Depth'] = curve_df['depth_a']*curve_df['Flow']**curve_df['depth_b']
    curve_df['TopWidth'] = curve_df['tw_a']*curve_df['Flow']**curve_df['tw_b']
    curve_df['Velocity'] = curve_df['vel_a']*curve_df['Flow']**curve_df['vel_b']
    curve_df = curve_df[curve_df['Depth']>0]
    curve_df = curve_df[curve_df['TopWidth']>0]
    curve_df['WSE'] = curve_df['Depth'] + curve_df['BaseElev']

    # Apply the outlier filtering function to each COMID group
    curve_df = curve_df.groupby('COMID', group_keys=False)[curve_df.columns].apply(filter_outliers)

    # Fill in the T_Rast and W_Rast
    for index, row in curve_df.iterrows():
        T_Rast[int(row['Row']), int(row['Col'])] = row['TopWidth'] * TW_MultFact
        W_Rast[int(row['Row']), int(row['Col'])] = row['Depth'] + row['BaseElev']
        if S_Rast is not None:
            S_Rast[int(row['Row']), int(row['Col'])] = row['Slope']

    # Calculate median values by COMID
    median_values = curve_df.groupby('COMID').agg({
        'TopWidth': 'median',
        'Depth': 'median',
        'WSE': 'median',
        'Velocity': 'median',
        'Row': 'first',
        'Col': 'first'
    })

    wse_stats = curve_df.groupby('COMID')['WSE'].agg(['mean', 'std'])
    
    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(median_values, on='COMID', how='left').fillna(0)
    comid_wse_stats = comid_result_df[['COMID']].merge(wse_stats, on='COMID', how='left').fillna(0)

    # Create dicts
    comid_result_df['COMID'] = comid_result_df['COMID'].astype(np.int32)
    comid_result_df['TopWidth'] = comid_result_df['TopWidth'].astype(np.float32)
    comid_result_df['Depth'] = comid_result_df['Depth'].astype(np.float32)
    comid_result_df['Velocity'] = comid_result_df['Velocity'].astype(np.float32)
    
    # Create dicts
    COMID_Unique_TW = comid_result_df.set_index('COMID')['TopWidth'].to_dict()
    COMID_Unique_Depth = comid_result_df.set_index('COMID')['Depth'].to_dict()
    COMID_Unique_Velocity = comid_result_df.set_index('COMID')['Velocity'].to_dict()
    COMID_Unique_WSE_Mean = comid_wse_stats.set_index('COMID')['mean'].astype(np.float32).to_dict()
    COMID_Unique_WSE_Std = comid_wse_stats.set_index('COMID')['std'].astype(np.float32).to_dict()
    
    # Get the maximum TopWidth for all COMIDs
    TopWidthMax = comid_result_df['TopWidth'].max()

    return (
        COMID_Unique_TW,
        COMID_Unique_Depth,
        COMID_Unique_Velocity,
        COMID_Unique_WSE_Mean,
        COMID_Unique_WSE_Std,
        TopWidthMax,
        T_Rast,
        W_Rast,
        S_Rast,
    )

# @njit(cache=True)
@njit("float32(float32, float32[:], float32[:])", cache=True)
def Find_TopWidth_at_Baseflow_when_using_VDT(QB, flow_values, top_width_values):
    """
    Find the TopWidth corresponding to the baseflow (QB).
    
    Args:
        QB (float): Baseflow discharge value.
        flow_values (list-like): Array or list of flow values (q_1, q_2, ..., q_n).
        top_width_values (list-like): Array or list of TopWidth values (t_1, t_2, ..., t_n).

    Returns:
        float: TopWidth corresponding to QB.
    """
    # for i in range(len(flow_values)):
    #     if QB <= flow_values[i]:
    #         return top_width_values[i]
    # # If QB is larger than all flow values, return the last TopWidth
    # return top_width_values[-1]
    idx = np.searchsorted(flow_values, QB, side="left")
    
    if idx >= len(flow_values):
        # QB is larger than all flow values
        return top_width_values[-1]
    else:
        return top_width_values[idx]

# @njit(cache=True)
@njit("float32(float32[:], float32[:], float32)", cache=True)
def interp1d_numba(x: np.ndarray, y: np.ndarray, xi: float | int) -> float:
    """
    Linearly interpolates/extrapolates a single value xi based on 1D arrays x and y.
    Equivalent to calling: `interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=True)(xi)`
    Parameters:
    - x: 1D array (assumed sorted)
    - y: 1D array of same length as x
    - xi: scalar input to interpolate
    Returns:
    - yi: interpolated or extrapolated value
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

@njit("Tuple((float32[:], float32[:], float32[:], float32[:]))(float32[:], float32[:], float32[:,:], float32[:, :], float32[:], float32[:, :], float32[:, :], float32[:], float32)",
      cache=True)
def vdt_interpolate(flow: np.ndarray,
                    qb: np.ndarray, 
                    flow_values: np.ndarray, 
                    top_width_values: np.ndarray,
                    elev_values: np.ndarray,
                    wse_values: np.ndarray,
                    vel_values: np.ndarray,
                    e_dem: np.ndarray,
                    tw_mult_fact: float) -> tuple[np.ndarray, ...]:
    top_width = np.empty_like(flow)
    depth = np.empty_like(flow)
    wse = np.empty_like(flow)
    baseflow_tw = np.empty_like(flow)
    vel = np.empty_like(flow)

    # Loop through each row in the DataFrame, interpolate as needed
    for i in range(len(flow)):
        if flow[i] <= qb[i]:
            # Below baseflow
            top_width[i] = Find_TopWidth_at_Baseflow_when_using_VDT(qb[i], flow_values[i], top_width_values[i])
            depth[i] = 0.001
            wse[i] = elev_values[i]
            vel[i] = vel_values[i][-1]
        elif flow[i] >= flow_values[i][-1]:
            # Above the maximum flow value
            top_width[i] = top_width_values[i][-1]
            wse[i] = wse_values[i][-1]
            depth[i] = wse[i] - e_dem[i]
            vel[i] = vel_values[i][-1]
        else:
            # Interpolate
            wse[i] = interp1d_numba(flow_values[i], wse_values[i], flow[i])
            top_width[i] = interp1d_numba(flow_values[i], top_width_values[i], flow[i])
            wse[i] = max(wse[i], e_dem[i])
            depth[i] = max(wse[i] - e_dem[i], 0.001)
            vel[i] = interp1d_numba(flow_values[i], vel_values[i], flow[i])

        baseflow_tw[i] = Find_TopWidth_at_Baseflow_when_using_VDT(qb[i], flow_values[i], top_width_values[i])

    # Ensure TopWidth respects baseflow and scale
    top_width = np.maximum(top_width, baseflow_tw) * tw_mult_fact
    
    return top_width, depth, wse, vel

def filter_outliers_helper(valuesv: pd.Series) -> np.ndarray:
    """
    You may ask, why no njit? Due to floating point differences with numba and pandas,
    this function returns very slightly different values if njited (effects like 10 cells out of thousands). 
    I do not want to change the tests Joseph and Mike may have set up, 
    but the differences are so slight that this could be changed and tests could be updated if desired. 
    """
    values = valuesv.to_numpy()
    mean = np.mean(values)
    values = values.astype(np.float64)
    std = np.std(values, ddof=1)
    lower = mean - 2 * std
    upper = mean + 2 * std
    mask = (lower <= values) & (values <= upper)
    return mask

def Calculate_TW_D_V_ForEachCOMID_VDTDatabase(E_DEM, VDTDatabaseFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact, fast_vdt: bool):    

    LOG.debug('\nOpening and Reading ' + VDTDatabaseFileName)
    
    # Read the VDT Database into a DataFrame
    if VDTDatabaseFileName.endswith('.parquet'):
        vdt_df = pd.read_parquet(VDTDatabaseFileName)
    else:
        vdt_df = pd.read_csv(VDTDatabaseFileName, engine='pyarrow')
        
    if vdt_df.empty:
        raise ValueError("The VDT Database file is empty or could not be read properly.")
    
    # Add COMID flow information
    comid_flow_df = pd.DataFrame(COMID_Unique_Flow.items(), columns=['COMID', 'Flow'])
    vdt_df = vdt_df.merge(comid_flow_df, on='COMID', how='left')    

    # Ensure row and col are integers
    vdt_df['Row'] = vdt_df['Row'].astype(int)
    vdt_df['Col'] = vdt_df['Col'].astype(int)
    
    # Extract the column indices for interpolation
    flow_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('q_')]
    top_width_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('t_')]
    wse_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('wse_')]
    vel_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('v_')]
    
    # Extract flow, baseflow, elevation, and Slope values
    flow = vdt_df['Flow'].values.astype(np.float32)
    qb = vdt_df['QBaseflow'].values.astype(np.float32)
    e_dem = E_DEM[vdt_df['Row'].values + 1, vdt_df['Col'].values + 1]

    # Extract flow, TopWidth, and WSE values for interpolation
    flow_values = vdt_df.iloc[:, flow_cols].values.astype(np.float32)
    top_width_values = vdt_df.iloc[:, top_width_cols].values.astype(np.float32)
    wse_values = vdt_df.iloc[:, wse_cols].values.astype(np.float32)
    vel_values = vdt_df.iloc[:, vel_cols].values.astype(np.float32)
    elev_values = vdt_df['Elev'].values.astype(np.float32)

    top_width, depth, wse, velocity = vdt_interpolate(flow, qb, flow_values, top_width_values, elev_values, wse_values, vel_values, e_dem, TW_MultFact)

    # Add the calculated values to the DataFrame
    vdt_df['TopWidth'] = top_width
    vdt_df['Depth'] = depth
    vdt_df['WSE'] = wse
    vdt_df['Velocity'] = velocity

    # Remove outliers by COMID
    for col in ['TopWidth', 'Depth', 'WSE', 'Velocity']:
        grp_mean = vdt_df.groupby('COMID')[col].transform('mean')
        grp_std = vdt_df.groupby('COMID')[col].transform('std')

        lower = grp_mean - 2 * grp_std
        upper = grp_mean + 2 * grp_std

        # Set outliers to NaN
        vdt_df[col] = vdt_df[col].where((vdt_df[col] >= lower) & (vdt_df[col] <= upper))

    # Drop rows with NaN values introduced during outlier removal
    vdt_df = vdt_df.dropna(subset=['TopWidth', 'Depth', 'WSE', 'Velocity'])

    # Apply the outlier filtering function to each COMID group
    cols = ['TopWidth', 'WSE', 'Velocity']
    for col in cols:
        if fast_vdt:
            # This is faster, but may have slight differences due to floating point operations (effects 0.025% of flood inundation cells)
            grp_mean = vdt_df.groupby('COMID')[col].transform('mean')
            grp_std = vdt_df.groupby('COMID')[col].transform('std')

            lower = grp_mean - 2 * grp_std
            upper = grp_mean + 2 * grp_std

            vdt_df = vdt_df[vdt_df[col].where((vdt_df[col] >= lower) & (vdt_df[col] <= upper)).notna()]
        else:
            g = vdt_df.groupby('COMID', group_keys=False)[col]
            mask = g.transform(filter_outliers_helper)

            vdt_df = vdt_df[mask > 0]

    if not fast_vdt:
        vdt_df = vdt_df.sort_values(by=['COMID'], kind='mergesort')

    
    # Fill T_Rast, W_Rast, and S_Rast
    T_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['TopWidth']
    W_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['WSE']
    if S_Rast is not None:
        S_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['Slope']
    
    # Calculate median values by COMID
    mean_values = vdt_df.groupby('COMID').agg({
        'TopWidth': 'median',
        'Depth': 'median',
        'WSE': 'median',
        'Velocity': 'median',
    })

    wse_stats = vdt_df.groupby('COMID')['WSE'].agg(['mean', 'std'])
    
    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(mean_values, on='COMID', how='left').fillna(0)
    comid_wse_stats = comid_result_df[['COMID']].merge(wse_stats, on='COMID', how='left').fillna(0)
    comid_result_df['COMID'] = comid_result_df['COMID'].astype(np.int32)
    comid_result_df['TopWidth'] = comid_result_df['TopWidth'].astype(np.float32)
    comid_result_df['Depth'] = comid_result_df['Depth'].astype(np.float32)
    comid_result_df['Velocity'] = comid_result_df['Velocity'].astype(np.float32)
    
    # Create dicts
    COMID_Unique_TW = comid_result_df.set_index('COMID')['TopWidth'].to_dict()
    COMID_Unique_Depth = comid_result_df.set_index('COMID')['Depth'].to_dict()
    COMID_Unique_Velocity = comid_result_df.set_index('COMID')['Velocity'].to_dict()
    COMID_Unique_WSE_Mean = comid_wse_stats.set_index('COMID')['mean'].astype(np.float32).to_dict()
    COMID_Unique_WSE_Std = comid_wse_stats.set_index('COMID')['std'].astype(np.float32).to_dict()

    # Get the maximum TopWidth for all COMIDs
    TopWidthMax = comid_result_df['TopWidth'].max()

    return (
        COMID_Unique_TW,
        COMID_Unique_Depth,
        COMID_Unique_Velocity,
        COMID_Unique_WSE_Mean,
        COMID_Unique_WSE_Std,
        TopWidthMax,
        T_Rast,
        W_Rast,
        S_Rast,
    )
  
def Get_Raster_Details(DEM_File):
    LOG.debug(DEM_File)
    gdal.Open(DEM_File, gdal.GA_ReadOnly)
    data = gdal.Open(DEM_File)
    geoTransform = data.GetGeoTransform()
    ncols = int(data.RasterXSize)
    nrows = int(data.RasterYSize)
    minx = geoTransform[0]
    dx = geoTransform[1]
    maxy = geoTransform[3]
    dy = geoTransform[5]
    maxx = minx + dx * ncols
    miny = maxy + dy * nrows
    Rast_Projection = data.GetProjectionRef()
    data = None
    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection

def Read_Raster_GDAL(InRAST_Name):
    try:
        dataset = gdal.Open(InRAST_Name, gdal.GA_ReadOnly)     
    except RuntimeError:
        sys.exit(" ERROR: Field Raster File cannot be read!")
    # Retrieve dimensions of cell size and cell count then close DEM dataset
    geotransform = dataset.GetGeoTransform()
    # Continue grabbing geospatial information for this use...
    band = dataset.GetRasterBand(1)
    RastArray = band.ReadAsArray()
    #global ncols, nrows, cellsize, yll, yur, xll, xur
    ncols=band.XSize
    nrows=band.YSize
    band = None
    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * np.fabs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0]
    xur = xll + (ncols)*geotransform[1]
    lat = np.fabs((yll+yur)/2.0)
    Rast_Projection = dataset.GetProjectionRef()
    dataset = None
    LOG.debug('Spatial Data for Raster File:')
    LOG.debug('   ncols = ' + str(ncols))
    LOG.debug('   nrows = ' + str(nrows))
    LOG.debug('   cellsize = ' + str(cellsize))
    LOG.debug('   yll = ' + str(yll))
    LOG.debug('   yur = ' + str(yur))
    LOG.debug('   xll = ' + str(xll))
    LOG.debug('   xur = ' + str(xur))
    return RastArray, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, Rast_Projection

def GetListOfDEMs(inputfolder):
    DEM_Files = []
    for file in os.listdir(inputfolder):
        #if file.startswith('return_') and file.endswith('.geojson'):
        if file.endswith('.tif') or file.endswith('.img'):
            DEM_Files.append(file)
    return DEM_Files

def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type, creation_options: list[str] = None):   
    o_driver = gdal.GetDriverByName(s_file_format)  #Typically will be a GeoTIFF "GTiff"
    #o_metadata = o_driver.GetMetadata()

    if creation_options is None:
        creation_options = ["COMPRESS=DEFLATE", 'PREDICTOR=2']
    
    # Construct the file with the appropriate data shape
    o_output_file = o_driver.Create(s_output_filename, xsize=ncols, ysize=nrows, bands=1, eType=s_output_type, options=creation_options)    

    # Set the geotransform
    o_output_file.SetGeoTransform(dem_geotransform)
    
    # Set the spatial reference
    o_output_file.SetProjection(dem_projection)
    
    # Write the data to the file
    o_output_file.GetRasterBand(1).WriteArray(raster_data)
    
    # Once we're done, close properly the dataset
    o_output_file = None


#   Convert_GDF_to_Output_Raster(Flood_File, flood_gdf, 'Value', ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)
def Convert_GDF_to_Output_Raster(s_output_filename, gdf, Param, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type):   
    LOG.info(s_output_filename)

    # Rasterize geometries
    LOG.info('Rasterizing geometries')
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[Param]))  # Replace 'value_column' with your column name
    raster_data = rasterize(
        shapes=shapes,
        out_shape=(nrows, ncols),
        transform=dem_geotransform,
        fill=0,  # Value to use for areas not covered by geometries
        dtype=s_output_type
    )
    LOG.info('Writing output file')
    # Write raster to file
    with rasterio.open(
        s_output_filename,
        "w",
        driver=s_file_format,
        height=nrows,
        width=ncols,
        count=1,
        dtype=s_output_type,
        crs=gdf.crs.to_string(),  # Use the GeoDataFrame's CRS
        transform=dem_geotransform,
    ) as dst:
        dst.write(raster_data, 1)
    return

def Write_Output_Raster_As_GeoDataFrame(raster_data, ncols, nrows, dem_geotransform, dem_projection, s_output_type):
    # Create an in-memory raster dataset
    driver = gdal.GetDriverByName('MEM')
    raster_ds = driver.Create('', xsize=ncols, ysize=nrows, bands=1, eType=s_output_type)

    # Set the geotransform and projection
    raster_ds.SetGeoTransform(dem_geotransform)
    raster_ds.SetProjection(dem_projection)

    # Write the data to the in-memory raster dataset
    raster_ds.GetRasterBand(1).WriteArray(raster_data)

    # Set NoData value to NaN
    nodata_value = np.nan
    raster_ds.GetRasterBand(1).SetNoDataValue(nodata_value)

    # Auto-generate a mask band that respects NoData
    mask_band = raster_ds.GetRasterBand(1).GetMaskBand()

    # Create an in-memory vector layer for the polygonized data
    memory_driver = ogr.GetDriverByName('Memory')
    vector_ds = memory_driver.CreateDataSource('')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dem_projection)
    layer = vector_ds.CreateLayer('polygons', srs=srs)

    # Add a field to the layer
    field = ogr.FieldDefn("Value", ogr.OFTInteger)
    layer.CreateField(field)

    # Polygonize the raster and write to the vector layer
    gdal.Polygonize(raster_ds.GetRasterBand(1), mask_band, layer, 0, [], callback=None)

    # Convert the OGR layer to GeoPandas GeoDataFrame
    polygons = []
    values = []
    
    for feature in layer:
        geom = feature.GetGeometryRef()
        # Parse the JSON string to a dictionary
        geom_dict = json.loads(geom.ExportToJson())
        polygons.append(shape(geom_dict))        
        values.append(feature.GetField("Value"))

    # Create a GeoDataFrame
    flood_gdf = gpd.GeoDataFrame({'Value': values, 'geometry': polygons})

    # Set the CRS
    flood_gdf.set_crs(dem_projection, inplace=True)

    # filter to only the flooded area
    flood_gdf = flood_gdf[flood_gdf['Value']>0]

    # Clean up
    raster_ds = None
    vector_ds = None

    return flood_gdf


@njit(cache=True)
def FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use):
    FourMatrix = np.full((3, 3), 4)
    
    # JLG commented this out because of an error but not sure the fix is correct
    # nrows_local = r_max - r_min + 2
    # ncols_local = c_max - c_min + 2
    # FloodLocal = np.zeros((nrows_local, ncols_local))
    nrows_local = np.int32(r_max - r_min + 2)
    ncols_local = np.int32(c_max - c_min + 2)
    FloodLocal = np.zeros((nrows_local,ncols_local), dtype=np.float32)
    
    FloodLocal[1:nrows_local-1,1:ncols_local-1] = np.where(E_Box<=WSE,1,0)
    
    # JLG commented this out because of an error but not sure the fix is correct
    #This is the Stream Cell.  Mark it with a 4
    # FloodLocal[(r_use-r_min+1),(c_use-c_min+1)] = 4 
    r_idx = int(r_use - r_min + 1)
    c_idx = int(c_use - c_min + 1)
    FloodLocal[r_idx, c_idx] = 4

    
    #Go through and mark all the cells that 
    for r in range((r_use-r_min+1),nrows_local-1):
        for c in range((c_use-c_min+1),ncols_local-1):
            #print(FloodLocal[r-1:r+2,c-1:c+2].shape)
            #print(FourMatrix.shape)
            #print(FloodLocal[r-1:r+2,c-1:c+2])
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
    
    #print(FloodLocal)
    #FloodReturn = np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>0.0,1.0,0.0)
    #print(np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>3.0,1.0,0.0))
    return np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>3.0,1.0,0.0)

@njit(cache=True)
def CreateWeightAndElipseMask(TW_temp, dx, dy, TW_MultFact):
    TW = int(TW_temp)  #This is the number of cells in the top-width

    # 

    # ElipseMask = np.zeros((TW+1,int(TW*2+1),int(TW*2+1)))  #3D Array
    WeightBox = np.zeros((int(TW*2+1),int(TW*2+1)), dtype=np.float32)  #2D Array
    # ElevMask = np.ones((int(TW*2+1),int(TW*2+1)))  #2D Array    #This is set and used later, and only if limit_low_elev_flooding=True


    # for i in range(1,TW+1):
    #     TWDX = i*dx*i*dx
    #     TWDY = i*dy*i*dy
    #     for r in range(0,i+1):
    #         for c in range(0,i+1):
    #             is_elipse = (c*dx*c*dx/(TWDX)) + (r*dy*r*dy/(TWDY))   #https://www.mathopenref.com/coordgeneralellipse.html
    #             if is_elipse<=1.0:
    #                 ElipseMask[i,TW+r,TW+c] = 1.0
    #                 ElipseMask[i,TW-r,TW+c] = 1.0
    #                 ElipseMask[i,TW+r,TW-c] = 1.0
    #                 ElipseMask[i,TW-r,TW-c] = 1.0
    # print(ElipseMask[2,TW-4:TW+4+1,TW-4:TW+4+1].astype(int))
    # print(ElipseMask[10,TW-14:TW+14+1,TW-14:TW+14+1].astype(int))
    # print(ElipseMask[40,TW-44:TW+44+1,TW-44:TW+44+1].astype(int))


    # --- Vectorized WeightBox creation ---
    n = 2*TW + 1
    # Create an array of indices [0, 1, ..., n-1]
    indices = np.arange(n)
    # Compute offsets from the center (center index is TW)
    # These offsets represent the "cell distance" in each direction.
    Y = indices - TW  # shape (n,)
    X = indices - TW  # shape (n,)
    # Broadcast to compute the squared distances:
    # For every cell, z2 = (dx * (x offset))^2 + (dy * (y offset))^2.
    # We use broadcasting: the row vector (X) and column vector (Y) combine to form an (n x n) array.
    z2 = (X * dx)**2  # shape (n,)
    z2 = z2[None, :] + ((Y * dy)**2)[:, None]  # shape (n, n)
    # Avoid very small values (to prevent division by zero)
    z2 = np.where(z2 < 0.0001, 0.0001, z2)
    WeightBox = 1.0 / z2
    
    # for r in range(0,TW+1):
    #     for c in range(0,TW+1):
    #         z2 = c*dx*c*dx + r*dy*r*dy
    #         if z2<0.0001:
    #             z2=0.001
    #         WeightBox[TW+r,TW+c] = 1 / (z2)
    #         WeightBox[TW-r,TW+c] = 1 / (z2)
    #         WeightBox[TW+r,TW-c] = 1 / (z2)
    #         WeightBox[TW-r,TW-c] = 1 / (z2)
    
    # return WeightBox, ElipseMask

    return WeightBox

@njit("float32[:,:](int32, float32, float32)", cache=True)
def create_weightbox(tw: int, dx: float, dy: float):
    # tw is the number of cells in the top-width

    # --- Vectorized WeightBox creation ---
    n = 2*tw + 1
    # Create an array of indices [0, 1, ..., n-1]
    indices = np.arange(n)
    # Compute offsets from the center (center index is TW)
    # These offsets represent the "cell distance" in each direction.
    X = ((indices - tw) * dx) ** 2 # shape (n,)
    Y = ((indices - tw) * dy) ** 2 # shape (n,)
    
    # Broadcast to compute the squared distances:
    # For every cell, z2 = (dx * (x offset))^2 + (dy * (y offset))^2.
    # We use broadcasting: the row vector (X) and column vector (Y) combine to form an (n x n) array.
    WeightBox = X[None, :] + Y[:, None]  # shape (n, n)
    # Avoid very small values (to prevent division by zero)
    WeightBox = np.clip(WeightBox, 0.0001, None)
    WeightBox = 1.0 / WeightBox


    return (WeightBox).astype(np.float32)

@njit(cache=True)
def get_owner_stats_all(WSE: np.ndarray, owner: np.ndarray, nrows: int, ncols: int):
    """Compute median and std-dev of WSE values for *each* owner id.

    This avoids calling get_local_stats for every cell (O(N^3)) and
    trades memory for CPU by scanning the grid per owner.

    Returns
    -------
    owner_median, owner_std : 1D float32 arrays of length (max_owner + 1)
        owner_median[i] / owner_std[i] are NaN when that owner has <2 valid values.
    """
    # Find the maximum owner id so we can size lookup tables
    max_owner = 0
    for r in range(nrows):
        for c in range(ncols):
            oid = owner[r, c]
            if oid > max_owner:
                max_owner = oid

    # Count valid (non-NaN) values per owner
    counts = np.zeros(max_owner + 1, dtype=np.int32)
    for r in range(nrows):
        for c in range(ncols):
            oid = owner[r, c]
            if oid <= 0:
                continue
            val = WSE[r, c]
            if not np.isnan(val):
                counts[oid] += 1

    # Compute stats per owner
    owner_median = np.full(max_owner + 1, np.nan, dtype=np.float32)
    owner_std    = np.full(max_owner + 1, np.nan, dtype=np.float32)

    for oid in range(1, max_owner + 1):
        n = counts[oid]
        if n < 2:
            continue
        vals = np.empty(n, dtype=np.float32)
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                if owner[r, c] != oid:
                    continue
                val = WSE[r, c]
                if not np.isnan(val):
                    vals[idx] = val
                    idx += 1

        vals = np.sort(vals)

        # Median
        if n % 2 == 1:
            med = vals[n // 2]
        else:
            med = 0.5 * (vals[n // 2 - 1] + vals[n // 2])

        # Mean
        mean_val = 0.0
        for i in range(n):
            mean_val += vals[i]
        mean_val /= n

        # Std (population std to match previous implementation)
        sum_sq = 0.0
        for i in range(n):
            diff = vals[i] - mean_val
            sum_sq += diff * diff
        std = np.sqrt(sum_sq / n)

        owner_median[oid] = np.float32(med)
        owner_std[oid]    = np.float32(std)

    return owner_median, owner_std

@njit(cache=True)
def get_local_stats(r, c, WSE, nrows, ncols):
    """
    Calculates the median and standard deviation of valid (non-NaN) WSE values
    in the 3x3 neighborhood of (r, c).
    """
    vals = np.empty(9, dtype=np.float32)
    count = 0

    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < nrows and 0 <= cc < ncols:
                val = WSE[rr, cc]
                if not np.isnan(val):
                    vals[count] = val
                    count += 1

    if count < 2:
        return np.nan, np.nan

    valid_vals = vals[:count]
    for i in range(count):
        for j in range(0, count - i - 1):
            if valid_vals[j] > valid_vals[j + 1]:
                tmp = valid_vals[j]
                valid_vals[j] = valid_vals[j + 1]
                valid_vals[j + 1] = tmp

    if count % 2 == 1:
        median = valid_vals[count // 2]
    else:
        median = 0.5 * (valid_vals[count // 2 - 1] + valid_vals[count // 2])

    sum_sq_diff = 0.0
    mean_val = 0.0
    for i in range(count):
        mean_val += valid_vals[i]
    mean_val /= count

    for i in range(count):
        sum_sq_diff += (valid_vals[i] - mean_val) ** 2

    std = np.sqrt(sum_sq_diff / count)

    return median, std
@njit(cache=True)
def _heap_push(r_heap, c_heap, p_heap, size, r, c, p):
    i = size
    size += 1
    r_heap[i] = r
    c_heap[i] = c
    p_heap[i] = p

    while i > 0:
        parent = (i - 1) // 2
        if p_heap[parent] <= p_heap[i]:
            break
        # swap parent and child
        tmp_r = r_heap[parent]
        tmp_c = c_heap[parent]
        tmp_p = p_heap[parent]
        r_heap[parent] = r_heap[i]
        c_heap[parent] = c_heap[i]
        p_heap[parent] = p_heap[i]
        r_heap[i] = tmp_r
        c_heap[i] = tmp_c
        p_heap[i] = tmp_p
        i = parent

    return size

@njit(cache=True)
def _heap_pop(r_heap, c_heap, p_heap, size):
    r = r_heap[0]
    c = c_heap[0]
    p = p_heap[0]
    size -= 1
    if size > 0:
        r_heap[0] = r_heap[size]
        c_heap[0] = c_heap[size]
        p_heap[0] = p_heap[size]

        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and p_heap[right] < p_heap[left]:
                smallest = right
            if p_heap[i] <= p_heap[smallest]:
                break
            # swap
            tmp_r = r_heap[i]
            tmp_c = c_heap[i]
            tmp_p = p_heap[i]
            r_heap[i] = r_heap[smallest]
            c_heap[i] = c_heap[smallest]
            p_heap[i] = p_heap[smallest]
            r_heap[smallest] = tmp_r
            c_heap[smallest] = tmp_c
            p_heap[smallest] = tmp_p
            i = smallest

    return r, c, p, size

@njit(cache=True)
def propagate_wse_fast_convergent(WSE_Initial, E, owner, owner_order, flowdir, nrows, ncols, cellsize, decay_per_meter=0.0001, prefer_low_wse=True, depth_cap=10.0):
    """
    Priority-driven propagation with stream-ownership gating.

    1. Captures all IDW seeds (even shallow ones).
    2. Finalizes cells in priority order (low WSE first by default).
    3. Prevents cross-basin overwrites by enforcing nearest-stream ownership.
    4. Uses gentle decay to allow spreading on flat terrain.
    """
    WSE_Out = WSE_Initial.copy()

    # Clamp dry-cell WSE to the max of its 4-connected neighbors
    for r in range(nrows):
        for c in range(ncols):
            if E[r, c] <= -9998.0:
                continue
            val = WSE_Out[r, c]
            if np.isnan(val):
                continue
            if val >= E[r, c] - 0.1:
                continue

            max_n = -1.0e20
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    nval = WSE_Out[nr, nc]
                    if not np.isnan(nval) and nval > max_n:
                        max_n = nval

            if max_n > -1.0e19 and val > max_n:
                WSE_Out[r, c] = max_n
    # owner_median, owner_std = get_owner_stats_all(WSE_Out, owner, nrows, ncols)

    # owner_median, owner_std = get_owner_stats_all(WSE_Work, owner, nrows, ncols)

    # for r in range(nrows):
    #     for c in range(ncols):
    #         oid = owner[r, c]
    #         if oid <= 0:
    #             continue
    #         val = WSE_Work[r, c]
    #         if np.isnan(val):
    #             continue

    #         med = owner_median[oid]
    #         std = owner_std[oid]

    #         if not np.isnan(std) and std > 0.0:
    #             if abs(val - med) > 2.0 * std:
    #                 WSE_Work[r, c] = med
    #         if not np.isnan(med) and WSE_Work[r, c] > med:
    #             WSE_Work[r, c] = med

    # WSE_Out = WSE_Work.copy()

    # 1. DEFINE SOURCE (Relaxed Definition)
    # Check if WSE is effectively at or above ground (allow -0.1m tolerance for float errors)
    wet_seed = (WSE_Out >= E - 0.1) & (E > -9998.0) & (~np.isnan(WSE_Out))

    max_q = nrows * ncols
    heap_r = np.empty(max_q, dtype=np.int32)
    heap_c = np.empty(max_q, dtype=np.int32)
    heap_p = np.empty(max_q, dtype=np.float32)
    heap_size = 0

    finalized = np.zeros((nrows, ncols), dtype=np.uint8)
    if prefer_low_wse:
        best = np.full((nrows, ncols), 1e20, dtype=np.float32)
    else:
        best = np.full((nrows, ncols), -1e20, dtype=np.float32)

    # 2. SEEDING
    for r in range(nrows):
        for c in range(ncols):
            if wet_seed[r, c]:
            # if wet_seed[r, c] and owner[r, c] > 0:
                wse = WSE_Out[r, c]
                if prefer_low_wse:
                    if wse < best[r, c]:
                        best[r, c] = wse
                        heap_size = _heap_push(heap_r, heap_c, heap_p, heap_size, r, c, wse)
                else:
                    if wse > best[r, c]:
                        best[r, c] = wse
                        heap_size = _heap_push(heap_r, heap_c, heap_p, heap_size, r, c, -wse)

    # 3. PROPAGATION
    while heap_size > 0:
        r, c, p, heap_size = _heap_pop(heap_r, heap_c, heap_p, heap_size)
        if prefer_low_wse:
            current_wse = p
        else:
            current_wse = -p

        if finalized[r, c]:
            continue
        if abs(current_wse - best[r, c]) > 1e-6:
            continue

        # oid = owner[r, c]
        # if oid > 0:
        #     med = owner_median[oid]
        #     std = owner_std[oid]
        #     if not np.isnan(std) and std > 0.0:
        #         if current_wse < (med - 2.0 * std) or current_wse > (med + 2.0 * std):
        #             continue

        finalized[r, c] = 1
        WSE_Out[r, c] = current_wse

        local_median, local_std = get_local_stats(r, c, WSE_Out, nrows, ncols)

        # # Gentle Decay (default 10cm/km)
        # drop = cellsize * decay_per_meter

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < nrows and 0 <= nc < ncols:
                if E[nr, nc] <= -9998.0:
                    continue
                if finalized[nr, nc]:
                    continue
                # if owner[nr, nc] != owner[r, c]:
                #     continue
                fd = flowdir[r, c]
                if fd > 0:
                    if not (
                        (fd == 1 and dr == 0 and dc == 1) or
                        (fd == 2 and dr == 1 and dc == 1) or
                        (fd == 4 and dr == 1 and dc == 0) or
                        (fd == 8 and dr == 1 and dc == -1) or
                        (fd == 16 and dr == 0 and dc == -1) or
                        (fd == 32 and dr == -1 and dc == -1) or
                        (fd == 64 and dr == -1 and dc == 0) or
                        (fd == 128 and dr == -1 and dc == 1)
                    ):
                        continue

                # --- HYDRAULIC CHECK ---
                # check if current WSE is above local median plus or minus 2 std-devs (if we have that info)
                if not np.isnan(local_std) and local_std > 0.0:
                    if current_wse < (local_median - local_std) or current_wse > (local_median + local_std):
                        max_allowed_wse = local_median  # <-- This is a hard cap to prevent outliers from spreading too much
                    else:
                        max_allowed_wse = current_wse
                # max_allowed_wse = current_wse - drop

                # Check if water fits
                if max_allowed_wse > E[nr, nc]:
                    # Ensure we don't accidentally set WSE below ground
                    final_wse = max(max_allowed_wse, E[nr, nc])
                    if depth_cap > 0.0:
                        if (final_wse - E[nr, nc]) > depth_cap:
                            continue
                    # cap = owner_median[owner[r, c]]
                    # if not np.isnan(cap) and final_wse > cap:
                    #     final_wse = cap

                    # if owner[nr, nc] != owner[r, c]:
                    #     continue
                        # # Allow higher-order owner to overwrite near confluences
                        # if owner_order[r, c] <= owner_order[nr, nc]:
                        #     continue
                        # if mix_zone[nr, nc] == 0:
                        #     continue

                        # seed_val = WSE_Out[nr, nc]
                        # seed_val = WSE_Work[nr, nc]
                        # if not np.isnan(seed_val) and final_wse > seed_val:
                        #     continue

                    if prefer_low_wse:
                        if final_wse < best[nr, nc]:
                            best[nr, nc] = final_wse
                            heap_size = _heap_push(heap_r, heap_c, heap_p, heap_size, nr, nc, final_wse)
                    else:
                        if final_wse > best[nr, nc]:
                            best[nr, nc] = final_wse
                            heap_size = _heap_push(heap_r, heap_c, heap_p, heap_size, nr, nc, -final_wse)

    return WSE_Out

@njit(cache=True)
def fldpln(WSE_Initial, E, flowdir, stream_id, order_lookup, downstream_lookup, nrows, ncols, fldmn, fldmx, dh, mxht0, ssflg):
    """
    This function is meant to mimic the FLDPLN model developed at the University of Kansas, translated iteratively
    using Codex-ChatGPT and this repository: https://github.com/AlabamaWaterInstitute/fldpln

    Propagate each fsp WSE upstream along reverse flow direction. A cell is inundated if its
    elevation is lower than the fsp WSE and it drains to that fsp cell. If multiple fsp
    values reach a cell, keep the maximum WSE.

    Returns WSE_Out (final WSE), fsp (flood source pixel id), and dtf (depth-to-flood).
    """
    # Initialize outputs: WSE_Out holds max WSE per cell, fsp holds source stream id, dtf holds first inundation stage.
    WSE_Out = np.full((nrows, ncols), np.nan, dtype=np.float32)
    fsp = np.zeros((nrows, ncols), dtype=np.int32)
    dtf = np.full((nrows, ncols), np.float32(1.0e20))

    # Preallocate BFS queue and per-seed visitation map for reverse-flow traversal.
    max_q = nrows * ncols
    q_r = np.empty(max_q, dtype=np.int32)
    q_c = np.empty(max_q, dtype=np.int32)
    visit_id = np.zeros((nrows, ncols), dtype=np.int32)
    seed_id = 0

    # Iterate over all seed cells with valid WSE_Initial and propagate their WSE upstream.
    for sr in range(nrows):
        for sc in range(ncols):
            wse = WSE_Initial[sr, sc]
            if np.isnan(wse) or wse <= -9998.0:
                continue
            # Start a new seed traversal.
            seed_id += 1
            q_read = 0
            q_write = 0
            q_r[q_write] = sr
            q_c[q_write] = sc
            q_write += 1
            visit_id[sr, sc] = seed_id

            # Use stream_id as fsp label when available.
            src_id = stream_id[sr, sc] if stream_id[sr, sc] > 0 else 0
            if np.isnan(WSE_Out[sr, sc]) or wse > WSE_Out[sr, sc]:
                WSE_Out[sr, sc] = wse
                fsp[sr, sc] = src_id
                dtf[sr, sc] = fldmn

            # Breadth-first search (BFS) over reverse-flow neighbors to propagate the fsp WSE upstream.
            while q_read < q_write:
                r = q_r[q_read]
                c = q_c[q_read]
                q_read += 1

                # Check upstream neighbors whose flowdir points into (r, c).
                if r > 0 and flowdir[r - 1, c] == 4:
                    nr = r - 1
                    nc = c
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and wse > E[nr, nc]:
                        # Neighbor drains into (r,c); enqueue and assign WSE if higher.
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(WSE_Out[nr, nc]) or wse > WSE_Out[nr, nc]:
                            WSE_Out[nr, nc] = wse
                            fsp[nr, nc] = src_id
                            dtf[nr, nc] = fldmn
                if r < nrows - 1 and flowdir[r + 1, c] == 64:
                    nr = r + 1
                    nc = c
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and wse > E[nr, nc]:
                        # Neighbor drains into (r,c); enqueue and assign WSE if higher.
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(WSE_Out[nr, nc]) or wse > WSE_Out[nr, nc]:
                            WSE_Out[nr, nc] = wse
                            fsp[nr, nc] = src_id
                            dtf[nr, nc] = fldmn
                if c > 0 and flowdir[r, c - 1] == 1:
                    nr = r
                    nc = c - 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and wse > E[nr, nc]:
                        # Neighbor drains into (r,c); enqueue and assign WSE if higher.
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(WSE_Out[nr, nc]) or wse > WSE_Out[nr, nc]:
                            WSE_Out[nr, nc] = wse
                            fsp[nr, nc] = src_id
                            dtf[nr, nc] = fldmn
                if c < ncols - 1 and flowdir[r, c + 1] == 16:
                    nr = r
                    nc = c + 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and wse > E[nr, nc]:
                        # Neighbor drains into (r,c); enqueue and assign WSE if higher.
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(WSE_Out[nr, nc]) or wse > WSE_Out[nr, nc]:
                            WSE_Out[nr, nc] = wse
                            fsp[nr, nc] = src_id
                            dtf[nr, nc] = fldmn
                if r > 0 and c > 0 and flowdir[r - 1, c - 1] == 2:
                    nr = r - 1
                    nc = c - 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and wse > E[nr, nc]:
                        # Neighbor drains into (r,c); enqueue and assign WSE if higher.
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(WSE_Out[nr, nc]) or wse > WSE_Out[nr, nc]:
                            WSE_Out[nr, nc] = wse
                            fsp[nr, nc] = src_id
                            dtf[nr, nc] = fldmn
                if r > 0 and c < ncols - 1 and flowdir[r - 1, c + 1] == 8:
                    nr = r - 1
                    nc = c + 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and wse > E[nr, nc]:
                        # Neighbor drains into (r,c); enqueue and assign WSE if higher.
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(WSE_Out[nr, nc]) or wse > WSE_Out[nr, nc]:
                            WSE_Out[nr, nc] = wse
                            fsp[nr, nc] = src_id
                            dtf[nr, nc] = fldmn
                if r < nrows - 1 and c > 0 and flowdir[r + 1, c - 1] == 128:
                    nr = r + 1
                    nc = c - 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and wse > E[nr, nc]:
                        # Neighbor drains into (r,c); enqueue and assign WSE if higher.
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(WSE_Out[nr, nc]) or wse > WSE_Out[nr, nc]:
                            WSE_Out[nr, nc] = wse
                            fsp[nr, nc] = src_id
                            dtf[nr, nc] = fldmn
                if r < nrows - 1 and c < ncols - 1 and flowdir[r + 1, c + 1] == 32:
                    nr = r + 1
                    nc = c + 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and wse > E[nr, nc]:
                        # Neighbor drains into (r,c); enqueue and assign WSE if higher.
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(WSE_Out[nr, nc]) or wse > WSE_Out[nr, nc]:
                            WSE_Out[nr, nc] = wse
                            fsp[nr, nc] = src_id
                            dtf[nr, nc] = fldmn

    return WSE_Out, fsp, dtf
@njit(cache=True, parallel=True)
def CreateSimpleFloodMapParallel(RR, CC, T_Rast, W_Rast, S_Rast, E, B, 
                                 owner, owner_order, flowdir, order_lookup, downstream_lookup, 
                                 nrows, ncols, sd, TW_m, dx, dy, LocalFloodOption, 
                                 COMID_Unique_TW: COMID_FLOW_DICT_TYPE,  
                                 COMID_Unique_Depth: COMID_FLOW_DICT_TYPE,
                                 COMID_Unique_Velocity: COMID_FLOW_DICT_TYPE,
                                 WeightBox, TW_for_WeightBox_ElipseMask, TW, TW_MultFact, TopWidthPlausibleLimit, Set_Depth, flood_vdt_cells, OutDEP,
                                 use_fldpln_model, fldmn, fldmx, dh, mxht0, ssflg):
    """
    This function uses parallelization to quickly generate floodmaps. Small differences can occur, infrequently, due to race conditions
    when adding weighted values to the WSE_Times_Weight and Slope_Times_Weight array slices. But, because of the order of the stream cells, 
    these differences are very small and infrequent, and relatively inconsequential to the overall flood inundation results.
    """

    COMID_Averaging_Method = 0

    # these are for the weighted approach but have to be made either way to keep Numba's shape inference consistent
    WSE_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    # Always allocate to keep Numba's shape inference consistent.
    Slope_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)

    Total_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)

    
    if use_fldpln_model == True:

        # if using Set_Depth, we want to seed all stream cells with WSE = E + Set_Depth, and then let the FLDPLN model expand from there.
        if Set_Depth > 0.0:
            W_Rast_Padded = np.full((nrows + 2, ncols + 2), np.nan, dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    if B[r, c] > 0:
                        W_Rast_Padded[r + 1, c + 1] = E[r, c] + Set_Depth
                    else:
                        W_Rast_Padded[r + 1, c + 1] = np.nan

        else:
            # pad the W_Rast to make it match E and the other arrays that are all nrows+2 by ncols+2
            W_Rast_Padded = np.full((nrows + 2, ncols + 2), np.nan, dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    if W_Rast[r, c] > -9998.0:
                        W_Rast_Padded[r + 1, c + 1] = W_Rast[r, c]
                    else:
                        W_Rast_Padded[r + 1, c + 1] = np.nan

        # This spreads WSE outward from currently-flooded cells to dry cells as long as
        # neighboring ground elevation E can be overtopped.
        WSE_array, fsp_map, dtf_map = fldpln(
                                                          W_Rast_Padded, E, flowdir, B, order_lookup, 
                                                          downstream_lookup, nrows + 2, ncols + 2, 
                                                          fldmn, fldmx, dh, mxht0, ssflg
        )
    else:
        
        #Now go through each cell
        num_nonzero = len(RR)
        for i in range(num_nonzero):
            r = RR[i]
            c = CC[i]
            r_use = r
            c_use = c
            E_Min = E[r,c]
            
            COMID_Value = B[r,c]
            if Set_Depth>0.0:
                WSE = float(E[r_use,c_use] + Set_Depth)
                if S_Rast is not None:
                    SLOPE = float(S_Rast[r_use,c_use])
                COMID_TW_m = TopWidthPlausibleLimit
            elif COMID_Averaging_Method!=0 or W_Rast[r-1,c-1]<0.001 or T_Rast[r-1,c-1]<0.00001:
                #Get COMID, TopWidth, and Depth Information for this cell
                COMID_Value = B[r,c]
                # keys are int32, values are float32
                if COMID_Value in COMID_Unique_TW:
                    COMID_TW_m = COMID_Unique_TW[COMID_Value]
                else:
                    COMID_TW_m = np.float32(0.0)

                if COMID_Value in COMID_Unique_Depth:
                    COMID_D = COMID_Unique_Depth[COMID_Value]
                else:
                    COMID_D = np.float32(0.0)
                WSE = float(E[r_use,c_use] + COMID_D)
                if S_Rast is not None:
                    SLOPE = float(S_Rast[r_use,c_use])
            else:
                #These are Based on the AutoRoute/ARC Results, not averaged for COMID
                WSE = W_Rast[r-1,c-1]  #Have to have the '-1' because of the Row and Col being inset on the B raster.
                COMID_TW_m = T_Rast[r-1,c-1]
                if S_Rast is not None:
                    SLOPE = S_Rast[r-1,c-1]
            

            if WSE < 0.001 or COMID_TW_m < 0.00001 or (WSE - E[r,c]) < 0.001:
                continue

            
            if COMID_TW_m > TW_m:
                COMID_TW_m = TW_m

            COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))
            #This is how many cells we will be looking at surrounding our stream cell


            
            # Find minimum elevation within the search box
            if sd >= 1:
                for rr in range(max(r - sd, 0), min(r + sd + 1, nrows - 1)):
                    for cc in range(max(c - sd, 1), min(c + sd + 1, ncols - 1)):
                        if E[rr,cc] > 0.1 and E[rr,cc] < E_Min:
                            E_Min = E[rr,cc]
                            r_use = rr
                            c_use = cc
                
            r_min = max(r_use - COMID_TW, 1)
            r_max = min(r_use + COMID_TW + 1, nrows + 1)
            c_min = max(c_use - COMID_TW, 1)
            c_max = min(c_use + COMID_TW + 1, ncols + 1)
            
            # This uses the weighting method from FloodSpreader to create a flood map
            # Here we use TW instead of COMID_TW.  This is because we are trying to find the center of the weight raster, which was set based on TW (not COMID_TW).  
            # COMID_TW mainly applies to the r_min, r_max, c_min, c_max
            w_r_min = TW_for_WeightBox_ElipseMask - (r_use - r_min)
            w_r_max = TW_for_WeightBox_ElipseMask + (r_max - r_use)
            w_c_min = TW_for_WeightBox_ElipseMask - (c_use - c_min)
            w_c_max = TW_for_WeightBox_ElipseMask + (c_max - c_use)
        
            weight_slice = WeightBox[w_r_min:w_r_max, w_c_min:w_c_max]
            if LocalFloodOption:
                #Find what would flood local
                E_Box = E[r_min:r_max,c_min:c_max]
                FloodLocalMask = FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use)
                WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice * FloodLocalMask)
                if S_Rast is not None:
                    Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice * FloodLocalMask)
            else:
                WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice)
                if S_Rast is not None:
                    Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice)

            Total_Weight[r_min:r_max,c_min:c_max] += weight_slice


        # divide the values in WSE_Times_Weight by the 
        # values in Total_Weight   
        WSE_array = WSE_Times_Weight / np.where(Total_Weight == 0, np.float32(1e-12), Total_Weight)

    Flooded_array = np.where((WSE_array > E) & (E > -9998.0), 1, 0).astype(np.uint8)



    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        for i in range(len(RR)):
            Flooded_array[RR[i],CC[i]] = 1

    # Create the Depth array
    if OutDEP:
        Depth_array = np.where((WSE_array > E) & (E > -9998.0), WSE_array - E, np.nan).astype(np.float32)
    else:
        Depth_array = np.empty((3, 3), dtype=np.float32) # Dummy array if not used

    # if you want, create the slope array
    if S_Rast is not None:
        Slope_divided_by_weight = Slope_Times_Weight / Total_Weight
        Slope_array = np.where((WSE_array > E) & (E > -9998.0), Slope_divided_by_weight, np.nan).astype(np.float32)
        Slope_array = np.where((Slope_array <= 0), 0.0002, Slope_array).astype(np.float32)
        return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], Slope_array[1:-1, 1:-1]


    return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], None

@njit(cache=True)
def CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, S_Rast, E, B, 
                         owner, owner_order, flowdir, order_lookup, downstream_lookup, 
                         nrows, ncols, sd, TW_m, dx, dy, LocalFloodOption, 
                         COMID_Unique_TW: COMID_FLOW_DICT_TYPE,
                         COMID_Unique_Depth: COMID_FLOW_DICT_TYPE,
                         COMID_Unique_Velocity: COMID_FLOW_DICT_TYPE,
                         COMID_Unique_WSE_Mean: COMID_FLOW_DICT_TYPE,
                         COMID_Unique_WSE_Std: COMID_FLOW_DICT_TYPE,
                         WeightBox, TW_for_WeightBox_ElipseMask, TW, TW_MultFact, TopWidthPlausibleLimit, Set_Depth, flood_vdt_cells, OutDEP,
                         use_fldpln_model, fldmn, fldmx, dh, mxht0, ssflg):
       
    COMID_Averaging_Method = 0

    # these are for the weighted approach but have to be made either way to keep Numba's shape inference consistent
    WSE_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    # Always allocate to keep Numba's shape inference consistent.
    Slope_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)

    Total_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)

    
    if use_fldpln_model == True:

        # if using Set_Depth, we want to seed all stream cells with WSE = E + Set_Depth, and then let the FLDPLN model expand from there.
        if Set_Depth > 0.0:
            W_Rast_Padded = np.full((nrows + 2, ncols + 2), np.nan, dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    if B[r, c] > 0:
                        W_Rast_Padded[r + 1, c + 1] = E[r, c] + Set_Depth
                    else:
                        W_Rast_Padded[r + 1, c + 1] = np.nan

        else:
            # pad the W_Rast to make it match E and the other arrays that are all nrows+2 by ncols+2
            W_Rast_Padded = np.full((nrows + 2, ncols + 2), np.nan, dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    if W_Rast[r, c] > -9998.0:
                        W_Rast_Padded[r + 1, c + 1] = W_Rast[r, c]
                    else:
                        W_Rast_Padded[r + 1, c + 1] = np.nan

        # This spreads WSE outward from currently-flooded cells to dry cells as long as
        # neighboring ground elevation E can be overtopped.
        WSE_array, fsp_map, dtf_map = fldpln(
                                                          W_Rast_Padded, E, flowdir, B, order_lookup, 
                                                          downstream_lookup, nrows + 2, ncols + 2, 
                                                          fldmn, fldmx, dh, mxht0, ssflg
        )
    else:
        
        #Now go through each cell
        num_nonzero = len(RR)
        for i in range(num_nonzero):
            r = RR[i]
            c = CC[i]
            r_use = r
            c_use = c
            E_Min = E[r,c]
            
            COMID_Value = B[r,c]
            if Set_Depth>0.0:
                WSE = float(E[r_use,c_use] + Set_Depth)
                if S_Rast is not None:
                    SLOPE = float(S_Rast[r_use,c_use])
                COMID_TW_m = TopWidthPlausibleLimit
            elif COMID_Averaging_Method!=0 or W_Rast[r-1,c-1]<0.001 or T_Rast[r-1,c-1]<0.00001:
                #Get COMID, TopWidth, and Depth Information for this cell
                COMID_Value = B[r,c]
                # keys are int32, values are float32
                if COMID_Value in COMID_Unique_TW:
                    COMID_TW_m = COMID_Unique_TW[COMID_Value]
                else:
                    COMID_TW_m = np.float32(0.0)

                if COMID_Value in COMID_Unique_Depth:
                    COMID_D = COMID_Unique_Depth[COMID_Value]
                else:
                    COMID_D = np.float32(0.0)
                WSE = float(E[r_use,c_use] + COMID_D)
                if S_Rast is not None:
                    SLOPE = float(S_Rast[r_use,c_use])
            else:
                #These are Based on the AutoRoute/ARC Results, not averaged for COMID
                WSE = W_Rast[r-1,c-1]  #Have to have the '-1' because of the Row and Col being inset on the B raster.
                COMID_TW_m = T_Rast[r-1,c-1]
                if S_Rast is not None:
                    SLOPE = S_Rast[r-1,c-1]
            

            if WSE < 0.001 or COMID_TW_m < 0.00001 or (WSE - E[r,c]) < 0.001:
                continue

            
            if COMID_TW_m > TW_m:
                COMID_TW_m = TW_m

            COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))
            #This is how many cells we will be looking at surrounding our stream cell


            
            # Find minimum elevation within the search box
            if sd >= 1:
                for rr in range(max(r - sd, 0), min(r + sd + 1, nrows - 1)):
                    for cc in range(max(c - sd, 1), min(c + sd + 1, ncols - 1)):
                        if E[rr,cc] > 0.1 and E[rr,cc] < E_Min:
                            E_Min = E[rr,cc]
                            r_use = rr
                            c_use = cc
                
            r_min = max(r_use - COMID_TW, 1)
            r_max = min(r_use + COMID_TW + 1, nrows + 1)
            c_min = max(c_use - COMID_TW, 1)
            c_max = min(c_use + COMID_TW + 1, ncols + 1)
            
            # This uses the weighting method from FloodSpreader to create a flood map
            # Here we use TW instead of COMID_TW.  This is because we are trying to find the center of the weight raster, which was set based on TW (not COMID_TW).  
            # COMID_TW mainly applies to the r_min, r_max, c_min, c_max
            w_r_min = TW_for_WeightBox_ElipseMask - (r_use - r_min)
            w_r_max = TW_for_WeightBox_ElipseMask + (r_max - r_use)
            w_c_min = TW_for_WeightBox_ElipseMask - (c_use - c_min)
            w_c_max = TW_for_WeightBox_ElipseMask + (c_max - c_use)
        
            weight_slice = WeightBox[w_r_min:w_r_max, w_c_min:w_c_max]
            if LocalFloodOption:
                #Find what would flood local
                E_Box = E[r_min:r_max,c_min:c_max]
                FloodLocalMask = FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use)
                WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice * FloodLocalMask)
                if S_Rast is not None:
                    Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice * FloodLocalMask)
            else:
                WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice)
                if S_Rast is not None:
                    Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice)

            Total_Weight[r_min:r_max,c_min:c_max] += weight_slice


        # divide the values in WSE_Times_Weight by the 
        # values in Total_Weight   
        WSE_array = WSE_Times_Weight / np.where(Total_Weight == 0, np.float32(1e-12), Total_Weight)

    Flooded_array = np.where((WSE_array > E) & (E > -9998.0), 1, 0).astype(np.uint8)



    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        for i in range(len(RR)):
            Flooded_array[RR[i],CC[i]] = 1

    # Create the Depth array
    if OutDEP:
        Depth_array = np.where((WSE_array > E) & (E > -9998.0), WSE_array - E, np.nan).astype(np.float32)
    else:
        Depth_array = np.empty((3, 3), dtype=np.float32) # Dummy array if not used

    # if you want, create the slope array
    if S_Rast is not None:
        Slope_divided_by_weight = Slope_Times_Weight / Total_Weight
        Slope_array = np.where((WSE_array > E) & (E > -9998.0), Slope_divided_by_weight, np.nan).astype(np.float32)
        Slope_array = np.where((Slope_array <= 0), 0.0002, Slope_array).astype(np.float32)
        return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], Slope_array[1:-1, 1:-1]


    return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], None

@njit("float32[:](float32)", cache=True)
def create_gaussian_kernel_1d(sigma):
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2

    kernel = np.empty(kernel_size, dtype=np.float32)

    for i in range(kernel_size):
        x = i - center
        kernel[i] = np.exp(- (x**2) / (2.0 * sigma**2))

    sum_val = np.sum(kernel)
    kernel /= sum_val

    return kernel

@njit("float32[:, :](float32[:, :], float32[:])", cache=True)
def convolve_rows(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    nrows, ncols = image.shape
    klen = len(kernel)
    pad = klen // 2
    output = np.empty_like(image)

    for r in range(nrows):
        for c in range(ncols):
            acc = 0.0
            weight_sum = 0.0
            c_start = max(0, c - pad)
            c_end = min(ncols, c + pad + 1)
            k_start = pad - (c - c_start)

            for k in range(c_end - c_start):
                val = image[r, c_start + k]
                if val <= -9998.0:
                    w = kernel[k_start + k]
                    acc += val * w
                    weight_sum += w

            output[r, c] = acc / weight_sum if weight_sum > 0 else image[r, c]
    return output

@njit("float32[:, :](float32[:, :], float32[:])", cache=True)
def convolve_cols(image, kernel):
    nrows, ncols = image.shape
    klen = len(kernel)
    pad = klen // 2
    output = np.zeros_like(image)

    for c in range(ncols):
        for r in range(nrows):
            acc = 0.0
            weight_sum = 0.0
            r_start = max(0, r - pad)
            r_end = min(nrows, r + pad + 1)
            k_start = pad - (r - r_start)

            for k in range(r_end - r_start):
                val = image[r_start + k, c]
                if val != -9999.0:
                    acc += val * kernel[k_start + k]
                    weight_sum += kernel[k_start + k]

            output[r, c] = acc / weight_sum if weight_sum > 0 else image[r, c]
    return output

@njit("float32[:, :](float32[:, :], float32)", cache=True)
def gaussian_blur_separable(image, sigma):
    kernel = create_gaussian_kernel_1d(sigma)
    blurred = convolve_rows(image, kernel)
    blurred = convolve_cols(blurred, kernel)
    return blurred


@njit(cache=True, parallel=True)
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
    """
    Fill and smooth bathymetry using the same WeightBox kernel
    used in CreateSimpleFloodMap.

    E, Bathy, ARBathyMask are all (nrows+2, ncols+2).
    """
    # ------------------------------------------------------------
    # 1) PRE-CLEANUP: Outside ARBathyMask, Bathy = DEM BEFORE weighting
    # ------------------------------------------------------------
    Bathy = Bathy.astype(np.float32).copy()
    for r in prange(nrows + 2):
        for c in range(ncols + 2):
            if ARBathyMask[r, c] == 1 and Bathy[r, c] < -98.99:
                Bathy[r, c] = E[r, c]

    # 2) Identify valid bathy donors inside the water mask
    #    (same nodata threshold as before: > -98.99)
    # Arrays to accumulate weighted sums
    bathy_times_weight = np.zeros_like(Bathy, dtype=np.float32)
    total_weight       = np.zeros_like(Bathy, dtype=np.float32)

    # 3) For each valid bathy cell, spread its value using WeightBox
    tw = TW_for_WeightBox_ElipseMask

    for r_use in range(1, nrows + 1):
        for c_use in range(1, ncols + 1):
            if not (Bathy[r_use, c_use] > -98.99 and ARBathyMask[r_use, c_use] == 1):
                continue

            # window in Bathy space (same pattern as CreateSimpleFloodMap)
            r_min = max(r_use - tw, 1)
            r_max = min(r_use + tw + 1, nrows + 1)
            c_min = max(c_use - tw, 1)
            c_max = min(c_use + tw + 1, ncols + 1)

            # corresponding window in WeightBox (centered on [tw, tw])
            w_r_min = tw - (r_use - r_min)
            w_c_min = tw - (c_use - c_min)

            val = Bathy[r_use, c_use]

            for rr in range(r_min, r_max):
                wr = w_r_min + (rr - r_min)
                for cc in range(c_min, c_max):
                    if ARBathyMask[rr, cc] != 1:
                        continue
                    wc = w_c_min + (cc - c_min)
                    w = WeightBox[wr, wc]
                    bathy_times_weight[rr, cc] += val * w
                    total_weight[rr, cc] += w

    # 5) Start from original Bathy, and fill only where Bathy was invalid
    filled = Bathy.copy().astype(np.float32)
    for r in prange(nrows + 2):
        for c in range(ncols + 2):
            b = Bathy[r, c]
            invalid = (b <= -98.99) or (b < -9998.0) or np.isnan(b)
            if not invalid:
                continue
            if total_weight[r, c] > 1e-10:
                filled[r, c] = bathy_times_weight[r, c] / total_weight[r, c]
            else:
                filled[r, c] = E[r, c]

    # 6) Optional extra smoothing (you can keep or weaken this)
    sigma_value = 1.0
    filled = gaussian_blur_separable(filled.astype(np.float32), sigma=sigma_value)

    # 7) Outside the AR bathy mask, always use DEM
    for r in prange(nrows + 2):
        for c in range(ncols + 2):
            if ARBathyMask[r, c] != 1:
                filled[r, c] = E[r, c]

    # 8) Final safety net: any remaining bad values from DEM
    for r in prange(nrows + 2):
        for c in range(ncols + 2):
            val = filled[r, c]
            if (val <= -98.99) or (val < -9998.0) or np.isnan(val):
                filled[r, c] = E[r, c]

    # 9) Honor Bathy_Use_Banks: keep bathy from being above DEM if requested
    if Bathy_Use_Banks == False:
        for r in prange(nrows + 2):
            for c in range(ncols + 2):
                if filled[r, c] > E[r, c]:
                    filled[r, c] = E[r, c]

    # 10) Return interior (arrays are padded by 1)
    return filled[1:nrows+1, 1:ncols+1]

def Calculate_Depth_TopWidth_TWMax_Velocity(E, CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, Q_Fraction, T_Rast, W_Rast, S_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy, Set_Depth, quiet, fast_vdt, linkno_to_twlimit=None):    # Initialize all dictionaries
    COMID_Unique_TW = {}
    COMID_Unique_Depth = {}
    COMID_Unique_Velocity = {}
    COMID_Unique_WSE_Mean = {}
    COMID_Unique_WSE_Std = {}
    
    if Set_Depth>0.0:
        # Initialize all to -9999
        COMID_Unique_TW = {}
        COMID_Unique_Depth = {}
        for comid in COMID_Unique:
            # Overwrite with limits if they are > 0
            if TopWidthPlausibleLimit > 0:
                COMID_Unique_TW[comid] = TopWidthPlausibleLimit
            else:
                
                COMID_Unique_TW[comid] = -9999.0
            if Set_Depth > 0:
                COMID_Unique_Depth[comid] = Set_Depth
            else:
                COMID_Unique_Depth[comid] = -9999.0

        TopWidthMax = TopWidthPlausibleLimit 
    #Mike switched to default to VDT Database instead of Curve.  We can change this in the future.
    elif len(VDTDatabaseFileName)>1:
        (COMID_Unique_TW, COMID_Unique_Depth, COMID_Unique_Velocity, COMID_Unique_WSE_Mean, COMID_Unique_WSE_Std, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_VDTDatabase(E, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact, fast_vdt)
    elif len(CurveParamFileName)>1:  
        (COMID_Unique_TW, COMID_Unique_Depth, COMID_Unique_Velocity, COMID_Unique_WSE_Mean, COMID_Unique_WSE_Std, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_CurveFile(CurveParamFileName, COMID_Unique_Flow, COMID_Unique,  T_Rast, W_Rast, S_Rast, TW_MultFact)

    LOG.info('Maximum Top Width = ' + str(TopWidthMax))
    
    if not quiet:
        for idx, comid in enumerate(COMID_Unique):
            if COMID_Unique_TW[comid]>TopWidthPlausibleLimit:
                LOG.warning(f"Ignoring {comid}  {COMID_Unique_Flow[comid]}  {COMID_Unique_Flow[comid]*Q_Fraction}  {COMID_Unique_Depth[comid]}  {COMID_Unique_TW[comid]}  {COMID_Unique_Velocity[comid]}")  

    if TopWidthPlausibleLimit < TopWidthMax:
        TopWidthMax = TopWidthPlausibleLimit
    
    #Create a Weight Box and an Elipse Mask that can be used for all of the cells
    X_cells = np.round(TopWidthMax/dx,0)
    Y_cells = np.round(TopWidthMax/dy,0)
    TW = int(max(Y_cells,X_cells))  #This is how many cells we will be looking at surrounding our stream cell
    
    return COMID_Unique_TW, COMID_Unique_Depth, COMID_Unique_Velocity, COMID_Unique_WSE_Mean, COMID_Unique_WSE_Std, TopWidthMax, TW, T_Rast, W_Rast, S_Rast

def Curve2Flood(E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, 
                COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName, 
                Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, WeightBox, 
                TW_for_WeightBox_ElipseMask, LocalFloodOption, Set_Depth, 
                quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast, OutDEP, 
                OutWSE, OutVEL, flowdir, use_fldpln_model, fldmn, fldmx, 
                dh, mxht0, ssflg, parallel, fast_vdt, 
                linkno_to_twlimit=None, linkno_to_order=None, linkno_to_downstream=None):
        
    # Calculate an Average Top Width and Depth for each stream reach.
    # The Depths are purposely adjusted to the DEM that you are using (this addresses issues with using the original or bathy dem)
    (COMID_Unique_TW, COMID_Unique_Depth, COMID_Unique_Velocity, 
     COMID_Unique_WSE_Mean, COMID_Unique_WSE_Std, TopWidthMax, 
     TW, T_Rast, W_Rast, S_Rast) = Calculate_Depth_TopWidth_TWMax_Velocity(E, CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, 
                                                                           COMID_Unique, Q_Fraction, T_Rast, W_Rast, S_Rast, TW_MultFact, 
                                                                           TopWidthPlausibleLimit, dx, dy, Set_Depth, quiet, fast_vdt, 
                                                                           linkno_to_twlimit=linkno_to_twlimit)
    
    #Create a simple Flood Map Data
    search_dist_for_min_elev = 0
    LOG.info('Creating Rough Flood Map Data...')

    # In Curve2Flood(...) just before CreateSimpleFloodMap(...)
    keys_tw  = np.asarray(list(COMID_Unique_TW.keys()), dtype=np.int32)
    vals_tw  = np.asarray(list(COMID_Unique_TW.values()), dtype=np.float32)
    keys_dep = np.asarray(list(COMID_Unique_Depth.keys()), dtype=np.int32)
    vals_dep = np.asarray(list(COMID_Unique_Depth.values()), dtype=np.float32)
    keys_vel = np.asarray(list(COMID_Unique_Velocity.keys()), dtype=np.int32)
    vals_vel = np.asarray(list(COMID_Unique_Velocity.values()), dtype=np.float32)
    keys_wse = np.asarray(list(COMID_Unique_WSE_Mean.keys()), dtype=np.int32)
    vals_wse_mean = np.asarray(list(COMID_Unique_WSE_Mean.values()), dtype=np.float32)
    vals_wse_std = np.asarray(list(COMID_Unique_WSE_Std.values()), dtype=np.float32)

    COMID_Unique_TW    = create_numba_dict_from(keys_tw,  vals_tw)
    COMID_Unique_Depth = create_numba_dict_from(keys_dep, vals_dep)
    COMID_Unique_Velocity = create_numba_dict_from(keys_vel, vals_vel)
    COMID_Unique_WSE_Mean = create_numba_dict_from(keys_wse, vals_wse_mean)
    COMID_Unique_WSE_Std = create_numba_dict_from(keys_wse, vals_wse_std)

    # we need to compute stream ownership if we are using FLDPLN
    if use_fldpln_model is True:
        owner = compute_stream_ownership(B, E)
        owner_order = compute_owner_order(owner, linkno_to_order)
        max_id = int(np.max(B))
        order_lookup = np.zeros(max_id + 1, dtype=np.int32)
        downstream_lookup = np.zeros(max_id + 1, dtype=np.int32)
        if linkno_to_order:
            for key, val in linkno_to_order.items():
                if key <= max_id:
                    order_lookup[int(key)] = int(val)
        if linkno_to_downstream:
            for key, val in linkno_to_downstream.items():
                if key <= max_id:
                    downstream_lookup[int(key)] = int(val)
    else:
        # make empty types for each of these that won't be used 
        owner = np.zeros_like(B, dtype=np.int32)
        owner_order = np.zeros_like(B, dtype=np.int32)
        order_lookup = np.zeros(1, dtype=np.int32)
        downstream_lookup = np.zeros(1, dtype=np.int32)

    if parallel:
        Flood_array, Depth_array, Slope_array  = CreateSimpleFloodMapParallel(RR, CC, T_Rast, W_Rast, 
                                                                              S_Rast, E, B, nrows, 
                                                                              ncols, search_dist_for_min_elev, 
                                                                              TopWidthMax, dx, dy, LocalFloodOption, 
                                                                              COMID_Unique_TW, COMID_Unique_Depth, 
                                                                              COMID_Unique_Velocity, WeightBox, 
                                                                              TW_for_WeightBox_ElipseMask, TW, 
                                                                              TW_MultFact, TopWidthPlausibleLimit, 
                                                                              Set_Depth, flood_vdt_cells, OutDEP)
    else:
        Flood_array, Depth_array, Slope_array  = CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, S_Rast, 
                                                                    E, B, owner, owner_order, flowdir, 
                                                                    order_lookup, downstream_lookup, 
                                                                    nrows, ncols, search_dist_for_min_elev, 
                                                                    TopWidthMax, dx, dy, LocalFloodOption, 
                                                                    COMID_Unique_TW, COMID_Unique_Depth, 
                                                                    COMID_Unique_Velocity, COMID_Unique_WSE_Mean, 
                                                                    COMID_Unique_WSE_Std, WeightBox, 
                                                                    TW_for_WeightBox_ElipseMask, TW, 
                                                                    TW_MultFact, TopWidthPlausibleLimit, 
                                                                    Set_Depth, flood_vdt_cells, OutDEP, 
                                                                    use_fldpln_model, fldmn, fldmx, 
                                                                    dh, mxht0, ssflg)
    
    return Flood_array, Depth_array, Slope_array
    

def Set_Stream_Locations(nrows: int, ncols: int, infilename: str):
    S = np.full((nrows, ncols), -9999, dtype=np.int32)  #Create an array
    if infilename.endswith('.parquet'):
        df = pd.read_parquet(infilename, columns=['Row', 'Col', 'COMID'])
    else:
        df = pd.read_csv(infilename, usecols=['Row', 'Col', 'COMID'], engine='pyarrow')
        
    S[df['Row'].values, df['Col'].values] = df['COMID'].values
    return S

def Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, watervalue):
    # (LC, ncols, nrows, cellsize, yll, yur, xll, xur, lat, lc_geotransform, lc_projection) = Read_Raster_GDAL(LandCoverFile)
    '''
    # Streams identified in LC
    LC = np.where(LC == watervalue, 1, 0)   # Mark streams with 1, other areas as 0
    
    # Streams identified in SN
    SN = np.where(S > 0, 1, 0)  # Mark streams with 1, other areas with 0
    
    # Combine LC and SN values
    F = np.where(SN == 1, 1, LC)  # Prioritize SN stream values, else take LC
    
    # Any cell shown as water in the LC or the STRM RAster are now always shown as flooded in the Flood_Ensemble
    Flood_Ensemble = np.where(F > 0, 100, Flood_Ensemble)
    '''
    # Identify streams in LC (1 for water, 0 otherwise)
    LC_array = (LC_array == watervalue).astype(int)

    # Identify streams in SN (1 for streams, 0 otherwise)
    SN = (S > 0).astype(int)

    # Combine LC and SN values, prioritizing SN
    F = SN | LC_array  # Logical OR operation prioritizes SN over LC

    # Update Flood_Ensemble: mark flooded cells (100) wherever F > 0
    Flood_Ensemble[F > 0] = 100

    return Flood_Ensemble

def Flood_Flooded_Cells_in_Map(Array_Ensemble, Flood_Ensemble, eps=0.01):
    """
    This function fills NaN values in the input array (Array_Ensemble) for cells that are marked as flooded
    in the Flood_Ensemble. It uses a nearest-neighbor approach to fill NaN values from the nearest valid
    flooded cells. If any flooded cells remain NaN after this process, they are assigned a small positive
    value (eps). Cells outside the flooded area remain NaN.

    Parameters:

    Array_Ensemble (np.ndarray): 2D array containing depth, water surface elevation, or velocity values, with NaNs for missing data.
    Flood_Ensemble (np.ndarray): 2D array indicating flooded cells (values > 0) and non-flooded cells (values <= 0).
    eps (float): Small positive value to assign to any remaining NaN flooded cells after filling.

    Returns:
    np.ndarray: 2D array with NaNs filled for flooded cells, and small positive values assigned where necessary.
    """

    # valid sources are flooded cells with a real (non-NaN) value
    source_mask =  (Flood_Ensemble > 0) & (~np.isnan(Array_Ensemble))
    # targets are flooded cells that are currently NaN
    target_mask = (Flood_Ensemble > 0) & (np.isnan(Array_Ensemble))

    # Copy to avoid modifying original array (optional)
    filled = Array_Ensemble.copy().astype(np.float32)

    # find the closest depth or WSE values from valid source cells
    if np.any(source_mask):
        # EDT returns indices of the nearest ZERO in the input,
        # so pass the inverse to point toward TRUE source cells.
        _, (ny, nx) = distance_transform_edt(~source_mask, return_indices=True)
        # nearest neighbor values from donors
        nn_vals = Array_Ensemble[ny, nx]
        filled[target_mask] = nn_vals[target_mask]

    # If anything inside Flood_Ensemble is STILL NaN, give it a tiny positive depth
    # (this happens when a flooded blob has zero donors anywhere)
    still_nan = (Flood_Ensemble > 0) & np.isnan(filled)
    if np.any(still_nan):
        filled[still_nan] = eps

    # keep everything outside Flood_Ensemble as NaN
    filled[(Flood_Ensemble <= 0) & (~np.isnan(filled))] = np.nan
    
    return filled    

def remove_cells_not_connected(flood_array: np.ndarray, streams_array: np.ndarray) -> np.ndarray:
    """
    This function identifies connected components in the first raster and retains only those components 
    that are (hydraulically) connected to positive cells in the second raster. Connectivity is defined as 8-connected 
    (including edges and corners).
    Parameters:
    -----------
    flood_array : np.ndarray
        A 2D array representing the first raster. Cells with positive values are considered for connectivity.
    streams_array : np.ndarray
        A 2D array representing the second raster. Positive cells in this raster determine the valid connections.
    Returns:
    --------
    np.ndarray
        A 2D array where cells in `flood_array` that are not connected to positive cells in `streams_array` are removed 
        (set to zero). The output retains the shape of `flood_array`.
    Notes:
    ------
    - Connectivity is determined using an 8-connected neighborhood, which includes horizontal, vertical, 
      and diagonal neighbors.
    - The function uses labeled connected components to identify and filter regions in `flood_array`.
    """

    # Define connectivity (8-connected: edges + corners)
    structure = generate_binary_structure(2, 2)
    
    # Label connected components in flood_array
    labeled_array, num_features = label(flood_array, structure)
    
    # Find labels that connect to positive cells in streams_array
    touching_labels = np.unique(labeled_array[(streams_array > 0) & (labeled_array > 0)])
    
    # Create mask of valid regions
    mask = np.isin(labeled_array, touching_labels)
    
    # Keep only connected chunks in flood_array
    return flood_array * mask

def ReadInputFile(lines,P):
    num_lines = len(lines)
    for i in range(num_lines):
        ls = lines[i].strip().split()
        if len(ls)>1 and ls[0]==P:
            if P in ['LocalFloodOption', 'FloodLocalOnly']:
                return True
            if P=='Set_Depth' or P=='FloodSpreader_SpecifyDepth':
                return float(ls[1])
            if P in ['Bathy_Use_Banks', 'Flood_WaterLC_and_STRM_Cells']:
                if "True" in ls[1]:
                    return True
                elif "False" in ls[1] or ls[1] == '':
                    return False    
            return ls[1]   
    if P=='Q_Fraction':
        return 1.0
    if P=='TopWidthPlausibleLimit':
        return 1000.0
    if P=='TW_MultFact':
        return 3.0
    if P=='Set_Depth' or P=='FloodSpreader_SpecifyDepth':
        return float(-1.1)
    if P in ['LocalFloodOption', 'FloodLocalOnly', 'Bathy_Use_Banks', 'Flood_WaterLC_and_STRM_Cells']:
        return False
    if P=='LAND_WaterValue':
        return 80
    if P=='OutDEP' or P=='OutWSE':
        return ""

    return ''

def create_positive_max_array(array_list: list[np.ndarray]) -> np.ndarray:
    """Create a maximum value array from a list of arrays, ignoring NaNs."""
    # Convert list to stacked array of shape (N, rows, cols)
    arr_stack = np.stack(array_list, axis=0).astype(np.float32)

    # Positive-value mask
    positive_mask = arr_stack > -9998.9
    masked_arr = np.ma.array(arr_stack, mask=~positive_mask)

    # Max across the stack, ignoring masked values
    max_vals = masked_arr.max(axis=0).filled(np.nan).astype(np.float32)

    return max_vals.astype(np.float32)

def create_depth(
                num_flows: int,
                array_list: list[np.ndarray],
                Flood_Ensemble: np.ndarray,
                streams: np.ndarray,
                E: np.ndarray,
                geotransform: tuple,
                projection: str,
                ncols: int,
                nrows: int,
                fname: str,
                nodata_value: float = -9999.0,
    ):
    max_vals = create_positive_max_array(array_list)
    # # Convert list to stacked array of shape (N, rows, cols)
    # arr_stack = np.stack(array_list, axis=0).astype(np.float32)

    # # Positive-value mask
    # positive_mask = arr_stack > -9998.9
    # masked_arr = np.ma.array(arr_stack, mask=~positive_mask)

    # # Max across the stack, ignoring masked values
    # max_vals = masked_arr.max(axis=0).filled(np.nan).astype(np.float32)

    # Optional rounding
    max_vals = np.round(max_vals, 2)

    # match the flood extent of Flood_Ensemble
    max_vals = Flood_Flooded_Cells_in_Map(max_vals, Flood_Ensemble, eps=0.01)

    # # filter out WSE that is not above the ground elevation
    # max_vals = np.where((max_vals > E[1:-1, 1:-1]) & (E[1:-1, 1:-1] > -9998.0), max_vals, np.nan).astype(np.float32)


    # # smooth max values with Gaussian filter
    # sigma_value = 0.75
    # max_vals = gaussian_blur_separable(max_vals, sigma=sigma_value)

    # Convert NaN → NoData sentinel
    out_band_data = np.where(np.isnan(max_vals), nodata_value, max_vals).astype(np.float32)

    # --- Write GeoTIFF ---
    driver = gdal.GetDriverByName("GTiff")
    ds: gdal.Dataset = driver.Create(
        fname, ncols, nrows, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"]
    )
    if ds is None:
        raise RuntimeError(f"Failed to create output raster: {fname}")

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    band = ds.GetRasterBand(1)
    band.WriteArray(out_band_data)
    band.SetNoDataValue(nodata_value)
    band.FlushCache()
    ds.FlushCache()

    # Cleanup
    band = None
    ds = None


    # # ------------------------------------------------------------------
    # # Find nearest non-zero stream value for every cell (by index)
    # # ------------------------------------------------------------------
    # streams_flat = streams.ravel()
    # m_flat = max_vals.ravel()  # not strictly needed, but handy if you mask later

    # # Indices of non-zero stream cells (i.e., real stream IDs)
    # nz_idx = np.flatnonzero(streams_flat)
    # if nz_idx.size == 0:
    #     raise ValueError("streams has no non-zero values")

    # # All positions in the flattened grid
    # pos = np.arange(streams_flat.size)

    # # For each position, find where it would be inserted among non-zero indices
    # insert_pos = np.searchsorted(nz_idx, pos)

    # # Candidate nearest indices to the left and right in nz_idx
    # left_idx = np.clip(insert_pos - 1, 0, nz_idx.size - 1)
    # right_idx = np.clip(insert_pos, 0, nz_idx.size - 1)

    # left_nz = nz_idx[left_idx]
    # right_nz = nz_idx[right_idx]

    # # Distances to left and right non-zero positions
    # left_dist = np.abs(pos - left_nz)
    # right_dist = np.abs(pos - right_nz)

    # # Choose nearest non-zero index (tie -> left)
    # nearest_nz = np.where(left_dist <= right_dist, left_nz, right_nz)

    # # Map each cell to the nearest non-zero stream ID
    # nearest_stream_ids_flat = streams_flat[nearest_nz]
    # nearest_stream_ids = nearest_stream_ids_flat.reshape(streams.shape)
    # # ------------------------------------------------------------------

    # # "Wet" cells: where max_vals is finite and not nodata-ish
    # wet = np.isfinite(max_vals) & (max_vals > -9998.0)

    # # SegID_Array: each cell gets the nearest stream ID
    # SegID_Array = nearest_stream_ids.astype(np.int32)
    # SegID_Array[~wet] = 0  # 0 = non-wet / no-stream zone

    # # make an array of the unique stream IDs present in SegID_Array
    # unique_stream_ids = np.unique(SegID_Array)

    # for stream_id in unique_stream_ids:
    #     if stream_id == 0:
    #         continue  # skip background / non-wet

    #     # All cells (stream + floodplain) that belong to this nearest-stream zone
    #     zone_mask = (SegID_Array == stream_id)
    #     if not np.any(zone_mask):
    #         continue

    #     # Only use finite water surface elevations in this zone
    #     zone_mask_finite = zone_mask & np.isfinite(max_vals)
    #     zone_vals = max_vals[zone_mask_finite]
    #     if zone_vals.size < 3:
    #         # not enough data for percentiles
    #         continue

    #     p25 = np.percentile(zone_vals, 25)
    #     p75 = np.percentile(zone_vals, 75)
    #     median_val = np.median(zone_vals)

    #     # Identify outliers in this zone
    #     outliers = (zone_vals < p25) | (zone_vals > p75)
    #     if np.any(outliers):
    #         # Write back only for outliers in this zone
    #         idx_row, idx_col = np.where(zone_mask_finite)
    #         max_vals[idx_row[outliers], idx_col[outliers]] = median_val


    return max_vals


@njit("DictType(int32, float32)(int32[:], float32[:])", cache=True)
def create_numba_dict_from(keys: np.ndarray, values: np.ndarray) -> dict[int, float]:
    # The Dict.empty() constructs a typed dictionary.
    # The key and value typed must be explicitly declared.
    d = Dict.empty(
        key_type=types.int32,
        value_type=types.float32,
    )
    for i in range(len(keys)):
        d[keys[i]] = values[i]
    return d

def create_bathymetry(E: np.ndarray, nrows: int, ncols: int, dem_geotransform: tuple, dem_projection: str, BathyFromARFileName: str, BathyWaterMaskFileName: str, 
                      Flood_Ensemble: np.ndarray, BathyOutputFileName: str, WeightBox: np.ndarray, TW_for_WeightBox_ElipseMask: int, Bathy_Use_Banks: bool, bathymetry_creation_options: list[str] = None):
    LOG.info('Working on Bathymetry')
    ds: gdal.Dataset = gdal.Open(BathyFromARFileName)
    ARBathy = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)  #Create an array that is slightly larger than the Bathy Raster Array
    # Read raster as float32
    ARBathy[1:-1, 1:-1] = ds.ReadAsArray().astype(np.float32)
    ds = None

    ARBathyMask = np.zeros((nrows+2,ncols+2), dtype=np.bool_)
    if os.path.exists(BathyWaterMaskFileName):
        ds = gdal.Open(BathyWaterMaskFileName)
        ARBathyMask[1:-1, 1:-1] = ds.ReadAsArray() > 0
        ds = None
    else:
        ARBathyMask[1:-1, 1:-1] = Flood_Ensemble > 0

    ARBathy[np.isnan(ARBathy)] = -99.000  #This converts all nan values to a -99
    ARBathy = ARBathy * ARBathyMask
    ARBathy[ARBathyMask != 1] = -9999.000
    # Bathy = Create_Topobathy_Dataset(RR, CC, E, B, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, Bathy_Yes, ARBathy, ARBathyMask)
    ARBathy = Create_Topobathy_Dataset(E, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, ARBathy, ARBathyMask, Bathy_Use_Banks).astype(np.float32)  # enforce again just in case

    # write the Bathy output raster
    Write_Output_Raster(BathyOutputFileName, ARBathy, ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32, bathymetry_creation_options)

def compute_stream_ownership(B: np.ndarray, E: np.ndarray) -> np.ndarray:
    mask = B > 0
    owner = np.zeros_like(B, dtype=np.int32)
    if not np.any(mask):
        return owner

    _, indices = ndi.distance_transform_edt(~mask, return_indices=True)
    r_idx = indices[0]
    c_idx = indices[1]
    owner = B[r_idx, c_idx].astype(np.int32)
    owner[E <= -9998.0] = 0

    return owner

def compute_owner_order(owner: np.ndarray, linkno_to_order: dict | None) -> np.ndarray:
    if linkno_to_order is None or len(linkno_to_order) == 0:
        return np.zeros_like(owner, dtype=np.int32)

    max_owner = int(np.max(owner))
    order_lookup = np.zeros(max_owner + 1, dtype=np.int32)
    for key, val in linkno_to_order.items():
        if key <= max_owner:
            order_lookup[int(key)] = int(val)

    return order_lookup[owner]

@njit(cache=True)
def compute_owner_mix_zone(owner: np.ndarray, mix_radius_cells: int) -> np.ndarray:
    nrows, ncols = owner.shape
    boundary = np.zeros_like(owner, dtype=np.uint8)

    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            oid = owner[r, c]
            if oid <= 0:
                continue
            if (
                owner[r - 1, c] != oid or
                owner[r + 1, c] != oid or
                owner[r, c - 1] != oid or
                owner[r, c + 1] != oid
            ):
                boundary[r, c] = 1

    if mix_radius_cells <= 0:
        return boundary

    mix_zone = np.zeros_like(owner, dtype=np.uint8)
    for r in range(nrows):
        for c in range(ncols):
            if boundary[r, c] == 0:
                continue
            r_min = r - mix_radius_cells
            if r_min < 0:
                r_min = 0
            r_max = r + mix_radius_cells + 1
            if r_max > nrows:
                r_max = nrows
            c_min = c - mix_radius_cells
            if c_min < 0:
                c_min = 0
            c_max = c + mix_radius_cells + 1
            if c_max > ncols:
                c_max = ncols

            for rr in range(r_min, r_max):
                for cc in range(c_min, c_max):
                    if owner[rr, cc] > 0:
                        mix_zone[rr, cc] = 1

    return mix_zone

def Curve2Flood_MainFunction(input_file: str = None,
                             args: dict = None, 
                             quiet: bool = False,
                             flood_vdt_cells: bool = True,
                             bathymetry_creation_options: list[str] = None,
                             parallel: bool = False,
                             fast_vdt: bool = False):

    """
    Main function that takes runs the flood mapping. If an input file is provided, it reads the parameters from the file.
    If no input file is provided, it uses the parameters from the args dictionary. The args dictionary should contain Python objects
    that can be converted to strings.

    Parameters:
    ----------
    input_file : str
        Path to the input file containing parameters.
    args : dict
        Dictionary of parameters.
    quiet : bool
        If True, suppresses warning messages and progress bars.
    flood_vdt_cells : bool
        If True, includes VDT cells in the flood map.
    bathymetry_creation_options : list[str]
        List of options for bathymetry raster creation.
    parallel : bool
        If True, enables parallel processing for simple floodmap. In tests, this increase flood mapping speed 2x, 
        with a slight change in values (~0.0003% of flooded cells differ).

    """
    if input_file:
        #Open the Input File
        with open(input_file,'r') as infile:
            lines = infile.readlines()
    elif args:
        # Use the args dictionary to extract parameters
        # Hacky way to convert args to lines
        lines = []
        for key, value in args.items():
            lines.append(f"{key}\t{value}\n")
    else:
        LOG.error("No input file or arguments provided.")
        return

    DEM_File = ReadInputFile(lines,'DEM_File')
    STRM_File = ReadInputFile(lines,'Stream_File')
    LAND_File = ReadInputFile(lines,'LU_Raster_SameRes')
    StrmShp_File = ReadInputFile(lines,'StrmShp_File')
    Flow_Direction_File = ReadInputFile(lines,'Flow_Direction_File')
    StrmOrder_Field = ReadInputFile(lines,'StrmOrder_Field')
    Downstream_Link_Field = ReadInputFile(lines,'Downstream_Link_Field')
    Flood_File = ReadInputFile(lines,'OutFLD')
    OutDEP = ReadInputFile(lines,'OutDEP')
    OutWSE = ReadInputFile(lines,'OutWSE')
    OutVEL = ReadInputFile(lines,'OutVEL')
    LU_Manning_n = ReadInputFile(lines,'LU_Manning_n')
    FloodImpact_File = ReadInputFile(lines,'FloodImpact_File')
    FlowFileName: str = ReadInputFile(lines,'COMID_Flow_File') or ReadInputFile(lines,'Comid_Flow_File')
    VDTDatabaseFileName = ReadInputFile(lines,'Print_VDT_Database')
    CurveParamFileName = ReadInputFile(lines,'Print_Curve_File')
    Use_FLDPLN_Model = ReadInputFile(lines,'Use_FLDPLN_Model')
    FLDPLN_fldmn = ReadInputFile(lines,'FLDPLN_fldmn')
    FLDPLN_fldmx = ReadInputFile(lines,'FLDPLN_fldmx')
    FLDPLN_dh = ReadInputFile(lines,'FLDPLN_dh')
    FLDPLN_mxht0 = ReadInputFile(lines,'FLDPLN_mxht0')
    FLDPLN_ssflg = ReadInputFile(lines,'FLDPLN_ssflg')
    Q_Fraction = ReadInputFile(lines,'Q_Fraction')
    TopWidthPlausibleLimit = ReadInputFile(lines,'TopWidthPlausibleLimit')
    TW_MultFact = ReadInputFile(lines,'TW_MultFact')
    Set_Depth = ReadInputFile(lines,'Set_Depth')
    Set_Depth = float(Set_Depth)
    Set_Depth2 = ReadInputFile(lines,'FloodSpreader_SpecifyDepth')  #This is the nomenclature for FloodSpreader
    Set_Depth2 = float(Set_Depth2)
    if Set_Depth2>0.0 and Set_Depth<0.0:
        Set_Depth = Set_Depth2
    LocalFloodOption = ReadInputFile(lines,'LocalFloodOption')
    LocalFloodOption2 = ReadInputFile(lines,'FloodLocalOnly')  #This is the nomenclature for FloodSpreader
    if LocalFloodOption2==True:
        LocalFloodOption = True
    BathyWaterMaskFileName = ReadInputFile(lines,'BathyWaterMask')
    BathyFromARFileName = ReadInputFile(lines,'BATHY_Out_File')
    BathyOutputFileName = ReadInputFile(lines,'FSOutBATHY')
    Flood_WaterLC_and_STRM_Cells = ReadInputFile(lines,'Flood_WaterLC_and_STRM_Cells')
    LAND_WaterValue = ReadInputFile(lines,'LAND_WaterValue')
    LAND_WaterValue = int(LAND_WaterValue)
    # Find the True/False variable to use the bank elevations to calculate the depth of the bathymetry estimate
    Bathy_Use_Banks = ReadInputFile(lines,'Bathy_Use_Banks')



    # Some checks
    if not FlowFileName:
        LOG.error("Flow file name is required.")
        return


    if not Flood_File:
        LOG.error("Flood file name is required.")
        return
    
    Q_Fraction = float(Q_Fraction)
    TopWidthPlausibleLimit = float(TopWidthPlausibleLimit)
    TW_MultFact = float(TW_MultFact)

    use_fldpln_model = False
    if Use_FLDPLN_Model is not None:
        use_fldpln_model = str(Use_FLDPLN_Model).strip().lower() in ("true", "1", "yes", "y")
    

    # Set default values if the variables are not None or an empty string
    if FLDPLN_fldmn is not None and FLDPLN_fldmn != '':
        fldmn = float(FLDPLN_fldmn)
    else:
        fldmn = 0.01

    if FLDPLN_fldmx is not None and FLDPLN_fldmx != '':
        fldmx = float(FLDPLN_fldmx)
    else:
        fldmx = 3.0

    if FLDPLN_dh is not None and FLDPLN_dh != '':
        dh = float(FLDPLN_dh)
    else:
        dh = 0.5

    if FLDPLN_mxht0 is not None and FLDPLN_mxht0 != '':
        mxht0 = float(FLDPLN_mxht0)
    else:
        mxht0 = 0.0

    if FLDPLN_ssflg is not None and FLDPLN_ssflg != '':
        ssflg = int(FLDPLN_ssflg)
    else:
        ssflg = 0

    # set model start time for logging
    Model_Start_Time = datetime.now()

    LOG.info('Opening ' + DEM_File)
    ds: gdal.Dataset = gdal.Open(DEM_File)
    dem_geotransform = ds.GetGeoTransform()
    dem_projection = ds.GetProjection()
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    cellsize = dem_geotransform[1]
    yll = dem_geotransform[3] - nrows * abs(dem_geotransform[5])
    yur = dem_geotransform[3]
    
    E = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)  #Create an array that is slightly larger than the STRM Raster Array and fill it with -9999.0
    E[1:-1, 1:-1] = ds.ReadAsArray()
    ds = None  

    if Flow_Direction_File:
        (FDR, fdr_ncols, fdr_nrows, _, _, _, _, _, _, _, _) = Read_Raster_GDAL(Flow_Direction_File)
        if fdr_ncols != ncols or fdr_nrows != nrows:
            LOG.warning("Flow direction raster size does not match DEM; flow-direction gating will be disabled.")
            FlowDir = np.zeros((nrows+2, ncols+2), dtype=np.int32)
        else:
            FlowDir = np.zeros((nrows+2, ncols+2), dtype=np.int32)
            FlowDir[1:-1, 1:-1] = FDR.astype(np.int32)
    else:
        FlowDir = np.zeros((nrows+2, ncols+2), dtype=np.int32)

    LOG.info("Executing flood mapping logic...")

    #Get the Stream Locations from the Curve or VDT File
    if Set_Depth>0.0:
        (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(STRM_File)
    elif len(VDTDatabaseFileName)>1:
        S = Set_Stream_Locations(nrows, ncols, VDTDatabaseFileName)
    elif len(CurveParamFileName)>1:
        S = Set_Stream_Locations(nrows, ncols, CurveParamFileName)
    else:
        LOG.error('NEED EITHER A CURVE PARAMATER FILE OR A VDT DATABASE FILE')
        return
    
    #Get Cellsize Information
    (dx, dy, dm) = convert_cell_size(cellsize, yll, yur)
    
    #Get list of Unique Stream IDs.  Also find where all the cell values are.
    B = np.zeros((nrows+2,ncols+2), dtype=np.int32)  #Create an array that is slightly larger than the STRM Raster Array
    B[1:-1, 1:-1] = S

    (RR,CC) = np.where(B > 0)

    COMID_Unique = np.unique(B[RR, CC]) # Always sorted
    COMID_Unique = COMID_Unique.astype(int) # Ensure it's treated as integers
    # COMID_Unique = np.delete(COMID_Unique, 0)  #We don't need the first entry of zero

    # Open the StrmShp_File if provided
    if StrmShp_File:
        # Read the shapefile
        LOG.info('Opening ' + StrmShp_File)
        Strm_gdf = gpd.read_file(StrmShp_File)
        # filter the Strm_gdf to only include the COMIDs in the COMID_Unique array
        Strm_gdf = Strm_gdf[Strm_gdf['LINKNO'].isin(COMID_Unique)]

        # change the TopWidthPlausibleLimit to be weighted by the stream order column in Strm_gdf
        order_field = StrmOrder_Field if StrmOrder_Field in Strm_gdf.columns else 'StrmOrder'
        if order_field in Strm_gdf.columns:
            Strm_gdf['TopWidthPlausibleLimit'] = (Strm_gdf[order_field]/max(Strm_gdf[order_field].values)) * TopWidthPlausibleLimit
            linkno_to_order = dict(zip(Strm_gdf['LINKNO'], Strm_gdf[order_field]))
            # drop all columns in the GDF except for the LINKNO/COMID column and the TopWidthPlausibleLimit column
            Strm_gdf = Strm_gdf[['LINKNO', 'TopWidthPlausibleLimit']]
            # Build a lookup dictionary from the GDF
            linkno_to_twlimit = dict(zip(Strm_gdf['LINKNO'], Strm_gdf['TopWidthPlausibleLimit']))
            del Strm_gdf
        else:
            linkno_to_twlimit = None
            linkno_to_order = None
            del Strm_gdf
    if StrmShp_File and Downstream_Link_Field:
        try:
            Strm_gdf_down = gpd.read_file(StrmShp_File)
            if Downstream_Link_Field in Strm_gdf_down.columns:
                linkno_to_downstream = dict(zip(Strm_gdf_down['LINKNO'], Strm_gdf_down[Downstream_Link_Field]))
            else:
                linkno_to_downstream = None
            del Strm_gdf_down
        except Exception:
            linkno_to_downstream = None
    else:
        linkno_to_downstream = None

    if not StrmShp_File:
        # If no shapefile is provided, set the linkno_to_twlimit to None
        linkno_to_twlimit = None
        linkno_to_order = None
    
    # COMID Flow File Read-in
    LOG.info('Opening and Reading ' + FlowFileName)
    with open(FlowFileName, 'r') as infile:
        header = infile.readline()
        line = infile.readline()
    
    #Order from highest to lowest flow
    ls = line.split(',')
    num_flows = pd.read_csv(FlowFileName, nrows=0).shape[1] - 1  #Subtract 1 for the COMID Column
    LOG.info('Evaluating ' + str(num_flows) + ' Flow Events')
    
    #Creating the initial Weight and Eclipse Boxes
    LOG.info('Creating the Weight and Eclipse Boxes')
    TW = int( max( np.round(TopWidthPlausibleLimit/dx,0), np.round(TopWidthPlausibleLimit/dy,0) ) )  #This is how many cells we will be looking at surrounding our stream cell
    TW_for_WeightBox_ElipseMask = TW
    # WeightBox = CreateWeightAndElipseMask(TW_for_WeightBox_ElipseMask, dx, dy, TW_MultFact)  #3D Array with the same row/col dimensions as the WeightBox
    WeightBox = create_weightbox(TW_for_WeightBox_ElipseMask, dx, dy)

    #If you're setting a set-depth value for all streams, just need to simulate one flood event
    if Set_Depth>=0.0:
        num_flows = 1
    
    # Create comid to flow dict
    COMID_Unique_Flow = {}

    # Create initial rasters once, outside the loop
    T_Rast = np.empty((nrows,ncols), np.float32)
    W_Rast = np.empty((nrows,ncols), np.float32)
    if OutVEL:
        S_Rast = np.empty((nrows,ncols), np.float32)
    else:
        S_Rast = None  # will be read later if needed      

    #Go through all the Flow Events
    Flood_array_list = []
    Depth_array_list = []
    Slope_array_list = []
    for flow_event_num in range(num_flows):
        LOG.info('Working on Flow Event ' + str(flow_event_num))
        # clear out last event’s values
        T_Rast[:] = -1.0
        W_Rast[:] = -1.0
        #Get an Average Flow rate associated with each stream reach.
        if Set_Depth<=0.000000001:
            COMID_Unique_Flow = FindFlowRateForEachCOMID_Ensemble(FlowFileName, flow_event_num)
        Flood_array, Depth_array, Slope_array = Curve2Flood(E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, 
                                                            COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName,
                                                            Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, WeightBox, 
                                                            TW_for_WeightBox_ElipseMask, LocalFloodOption, Set_Depth, 
                                                            quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast, OutDEP, 
                                                            OutWSE, OutVEL, FlowDir, use_fldpln_model, fldmn, fldmx,
                                                            dh, mxht0, ssflg, parallel, fast_vdt, 
                                                            linkno_to_twlimit=linkno_to_twlimit, 
                                                            linkno_to_order=linkno_to_order,
                                                            linkno_to_downstream=linkno_to_downstream)        
        Flood_array = remove_cells_not_connected(Flood_array, S)
        Flood_array_list.append(Flood_array)
        Depth_array_list.append(Depth_array)
        Slope_array_list.append(Slope_array)

    # clear some memory by deleting large arrays that are no longer needed
    del T_Rast
    del W_Rast    

    # Combine all flow events into a single ensemble
    #Turn into a percentage
    if num_flows > 1:
        Flood_Ensemble = (100 * np.sum(Flood_array_list, axis=0) / num_flows).astype(np.uint8)
    else:
        Flood_Ensemble = Flood_array_list[0] * 100  # Much quicker than nansum

    # If Flood_WaterLC_and_STRM_Cells or OutVEL is selected, we need to read in the Land Cover Raster
    if Flood_WaterLC_and_STRM_Cells or OutVEL:
        (LC_array, ncols, nrows, cellsize, yll, yur, xll, xur, lat, lc_geotransform, lc_projection) = Read_Raster_GDAL(LAND_File)

    # If selected, we can also flood cells based on the Land Cover and the Stream Raster
    if Flood_WaterLC_and_STRM_Cells:
        LOG.info('Flooding the Water-Related Land Cover and STRM cells')
        Flood_Ensemble = Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, LAND_WaterValue)

    # Remove crop circles and other disconnected cells
    Flood_Ensemble = remove_cells_not_connected(Flood_Ensemble, S)

    if Set_Depth < 0:
        LOG.info('Creating Ensemble Flood Map...' + str(Flood_File))

    # Write the output raster
    out_ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(Flood_File, ncols, nrows, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "PREDICTOR=2"])
    out_ds.SetGeoTransform(dem_geotransform)
    out_ds.SetProjection(dem_projection)
    out_ds.WriteArray(Flood_Ensemble)
    out_ds.FlushCache()
    out_ds = None  # Close the dataset to ensure it's written to disk


    if OutDEP:
        Depth_Array = create_depth(num_flows, Depth_array_list, Flood_Ensemble, S, E, dem_geotransform, dem_projection, ncols, nrows, OutDEP)

    if OutDEP and OutWSE:
        WSE_Array = np.where((Depth_Array > 0) & (E[1:-1, 1:-1] > -9998.0), Depth_Array+E[1:-1, 1:-1], np.nan).astype(np.float32)

        
        # --- Write GeoTIFF ---
        driver = gdal.GetDriverByName("GTiff")
        ds: gdal.Dataset = driver.Create(
            OutWSE, ncols, nrows, 1, gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"]
        )
        if ds is None:
            raise RuntimeError(f"Failed to create output raster: {OutDEP}")

        ds.SetGeoTransform(dem_geotransform)
        ds.SetProjection(dem_projection)

        band = ds.GetRasterBand(1)
        band.WriteArray(WSE_Array)
        band.SetNoDataValue(-9999.0)
        band.FlushCache()
        ds.FlushCache()

        # Cleanup
        band = None
        ds = None
    
    # If the FLDPLN model was used, we need to calculate the Slope_array here
    # We will just use the S_Rast and Flood_array to find the closest slope from S_Rast for each flooded cell
    if use_fldpln_model and S_Rast is not None:
        Slope_array = Flood_Flooded_Cells_in_Map(S_Rast.astype(np.float32), Flood_array.astype(np.uint8), eps=0.0002)
        Slope_array = np.where((Slope_array <= 0), np.nan, Slope_array).astype(np.float32)
        # smooth with Gaussian filter
        sigma_value = 1.00
        Slope_array = gaussian_blur_separable(Slope_array.astype(np.float32), sigma=sigma_value)
        Slope_array_list = [Slope_array]

    if OutVEL and OutDEP:
        # Create the velocity output raster
        # VEL_Array = create_depth_or_wse(num_flows, Velocity_array_list, Flood_Ensemble, S, dem_geotransform, dem_projection, ncols, nrows, OutVEL)
        create_velocity(OutVEL, Depth_Array, LU_Manning_n, LC_array, Slope_array_list, dem_geotransform, dem_projection, ncols, nrows, Flood_Ensemble, S)


    if StrmShp_File:
        # convert the raster to a geodataframe
        flood_gdf = Write_Output_Raster_As_GeoDataFrame(Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, gdal.GDT_Byte)
        
        # the name of our flood shapefile
        shp_output_filename = f"{Flood_File[:-4]}.gpkg"

        # save the geodataframe (do not specify the driver, it will be inferred from the file extension)
        flood_gdf.to_file(shp_output_filename)

    if BathyFromARFileName and BathyOutputFileName:
        create_bathymetry(E, nrows, ncols, dem_geotransform, dem_projection, 
                          BathyFromARFileName, BathyWaterMaskFileName, Flood_Ensemble, 
                          BathyOutputFileName, WeightBox, TW_for_WeightBox_ElipseMask, 
                          Bathy_Use_Banks, bathymetry_creation_options)

    # Example of simulated execution
    LOG.info("Flood mapping completed.")
    Model_Simulation_Time = datetime.now() - Model_Start_Time
    LOG.info(f'Simulation time (sec)= {Model_Simulation_Time.seconds}')

    return

