
import numpy as np
import pandas as pd
import xarray as xr
from iris.analysis.cartography import area_weights
from scipy.ndimage import labeled_comprehension

def get_tb(olr):
    """                                                                                                                                                       
    This function converts outgoing longwave radiation to brightness temperatures.                                                                                                                                                                                                          
                                                                                                                                                             
    Args:                                                                                                                                                     
        olr(xr.DataArray or numpy array): 2D field of model output with OLR                                                                                   
                                                                                                                                                             
    Returns:                                                                                                                                                   
        tb(xr.DataArray or numpy array): 2D field with estimated brightness temperatures                                                                       
    """
    # constants                                                                                                                                               
    aa = 1.228
    bb = -1.106e-3  # K−1                                                                                                                               
    # Planck constant                                                                                                                                         
    sigma = 5.670374419e-8  # W⋅m−2⋅K−4  
                                                                                                                 
    # flux equivalent brightness temperature                                                                                                                  
    Tf = (abs(olr)/sigma) **(1./4)
    tb = (((aa ** 2 + 4 * bb *Tf ) ** (1./2)) - aa)/(2*bb)
    return tb

def calc_area_and_precip(tracks, segments, ds, MCS):
    """Calculate the area, maximum precip rate and total precip volume for each
    feature

    Parameters
    ----------
    tracks : _type_
        _description_
    segments : _type_
        _description_
    ds : _type_
        _description_
    MCS : _type_
        _description_

    Returns
    -------
    tracks
        _description_
    """
    # Get area array and calculate area of each segment
    segments.coord("latitude").guess_bounds()
    segments.coord("longitude").guess_bounds()
    area = area_weights(segments, normalize=False)

    tracks["area"] = np.nan
    tracks["max_precip"] = np.nan
    tracks["total_precip"] = np.nan

    features_t = xr.CFTimeIndex(tracks["time"].to_numpy()).to_datetimeindex()
    for time, mask in zip(ds.time.data, segments.slices_over("time")):
        wh = features_t==time
        if np.any(wh):
            feature_areas = labeled_comprehension(
                area, mask.data, tracks[wh]["feature"], np.sum, area.dtype, np.nan
            )
            tracks.loc[wh, "area"] = feature_areas
            
            step_precip = ds[MCS.precip_var].sel({MCS.time_dim:time}).values
            max_precip = labeled_comprehension(
                step_precip, mask.data, tracks[wh]["feature"], np.max, area.dtype, np.nan
            )
            
            tracks.loc[wh, "max_precip"] = max_precip
            
            feature_precip = labeled_comprehension(
                area * step_precip, mask.data, tracks[wh]["feature"], np.sum, area.dtype, np.nan
            )
            
            tracks.loc[wh, "total_precip"] = feature_precip
    
    return tracks

def max_consecutive_true(condition: np.ndarray[bool]) -> int:
    """Return the maximum number of consecutive True values in 'condition'

    Parameters
    ----------
    condition : np.ndarray[bool]
        numpy array of boolean values

    Returns
    -------
    int
        the maximum number of consecutive True values in 'condition'
    """
    if np.any(condition):
        return np.max(np.diff(np.where(np.concatenate(([condition[0]], condition[:-1] != condition[1:], [True])))[0])[::2], initial=0)
    else:
        return 0

def is_track_mcs(features: pd.Dataframe) -> pd.DataFrame:
    """Test whether each track in features meets the condtions for an MCS

    Parameters
    ----------
    features : pd.Dataframe
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    consecutive_precip_max = features.groupby("track").apply(lambda df: max_consecutive_true(df.groupby("time").max_precip.max().to_numpy() >= 10))
    consecutive_area_max = features.groupby("track").apply(lambda df: max_consecutive_true(df.groupby("time").area.max().to_numpy() >= 4e10))
    max_total_precip = features.groupby("track").apply(lambda df: df.groupby("time").total_precip.sum().max())
    result = np.logical_and.reduce([
        consecutive_precip_max >= 4,
        consecutive_area_max >= 4,
        max_total_precip >= 2e10
    ])
    return pd.DataFrame(data=result, index=consecutive_precip_max.index)