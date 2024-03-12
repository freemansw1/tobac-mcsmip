import warnings
import numpy as np
import pandas as pd
import xarray as xr
from iris.analysis.cartography import area_weights
import tobac
from tobac.utils.periodic_boundaries import weighted_circmean


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
    Tf = (abs(olr) / sigma) ** (1.0 / 4)
    tb = (((aa**2 + 4 * bb * Tf) ** (1.0 / 2)) - aa) / (2 * bb)
    return tb


def calc_area_and_precip(features, segments, ds, MCS, inplace=False):
    """Calculate the area, maximum precip rate and total precip volume for each
    feature

    Parameters
    ----------
    features : _type_
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
    if not inplace:
        features = features.copy()

    # Get area array and calculate area of each segment
    segment_slice = segments[0]
    segment_slice.coord("latitude").guess_bounds()
    segment_slice.coord("longitude").guess_bounds()
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Using DEFAULT_SPHERICAL_EARTH_RADIUS*",
    )
    area = area_weights(segment_slice, normalize=False)
    area = xr.DataArray(area, coords=dict(lat=ds.lat, lon=ds.lon), dims=["lat", "lon"])

    precip = ds[MCS.precip_var]

    features["area"] = np.nan
    features["max_precip"] = np.nan
    features["total_precip"] = np.nan

    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
    features, segments, area, statistic=dict(area=np.nansum), default=np.nan
    )
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, precip, statistic=dict(max_precip=np.nanmax), default=np.nan
    )
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, precip * area.values, statistic=dict(total_precip=np.nansum), default=np.nan
    )

    return features

def process_clusters(tracks):
    groupby_order = ["frame", "track"]
    tracks["cluster"] = (tracks.groupby(groupby_order).feature.cumcount()[tracks.sort_values(groupby_order).index]==0).cumsum().sort_index()
    
    gb_clusters = tracks.groupby("cluster")
    
    clusters = gb_clusters.track.first().to_frame().rename(columns=dict(track="cluster_track_id"))
    
    clusters["cluster_time"] = gb_clusters.time.first().to_numpy()
    
    clusters["cluster_longitude"] = gb_clusters.apply(lambda x:weighted_circmean(x.longitude.to_numpy(), x.area.to_numpy(), low=0, high=360))#, include_groups=False)
    clusters["cluster_latitude"] = gb_clusters.apply(lambda x:np.average(x.latitude.to_numpy(), weights=x.area.to_numpy()))#, include_groups=False)
    
    clusters["cluster_area"] = gb_clusters.area.sum().to_numpy()
    clusters["cluster_max_precip"] = gb_clusters.max_precip.max().to_numpy()
    clusters["cluster_total_precip"] = gb_clusters.total_precip.sum().to_numpy()
    
    return tracks, clusters

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
    if isinstance(condition, pd.Series):
        condition = condition.to_numpy()
    if np.any(condition):
        return np.max(
            np.diff(
                np.where(
                    np.concatenate(
                        ([condition[0]], condition[:-1] != condition[1:], [True])
                    )
                )[0]
            )[::2],
            initial=0,
        )
    else:
        return 0


def is_track_mcs(clusters: pd.DataFrame) -> pd.DataFrame:
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
    consecutive_precip_max = clusters.groupby(["cluster_track_id"]).cluster_max_precip.apply(lambda x:max_consecutive_true(x>=10))#, include_groups=False)
    
    consecutive_area_max = clusters.groupby(["cluster_track_id"]).cluster_area.apply(lambda x:max_consecutive_true(x>=4e10))#, include_groups=False)
    
    max_total_precip = clusters.groupby(["cluster_track_id"]).cluster_total_precip.max()
    
    is_mcs = np.logical_and.reduce(
        [
            consecutive_precip_max >= 4,
            consecutive_area_max >= 4,
            max_total_precip.to_numpy() >= 2e10,
        ]
    )
    mcs_tracks =  pd.Series(data=is_mcs, index=consecutive_precip_max.index)
    mcs_tracks.index.name="track"
    return mcs_tracks
