import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import tobac

cmd_folder = pathlib.Path.cwd().parent.absolute()
if cmd_folder not in sys.path:
    sys.path.insert(0, str(cmd_folder))

from utils import data_formats, get_tb, calc_area_and_precip, is_track_mcs

model = "OBS"
season = "winter"

if season == "summer":
    dates = pd.date_range(
        datetime(2020, 8, 1),
        datetime(2016, 9, 9),
        freq="D"
    )
    for MCS in data_formats.summer_datasets:
        if MCS.name == model:
            break
    else:
        raise ValueError(f"model {model} not found for season {season}")

if season == "winter":
    dates = pd.date_range(
        datetime(2020, 1, 20),
        datetime(2020, 2, 28),
        freq="D"
    )
    for MCS in data_formats.summer_datasets:
        if MCS.name == model:
            break
    else:
        raise ValueError(f"model {model} not found for season {season}")

data_path = pathlib.Path("../data_in")

files = sorted([MCS.glob_date(data_path, "winter", date) for date in dates])

ds = xr.open_mfdataset(files, combine="nested", concat_dim=MCS.time_dim)
ds = ds.assign_coords({MCS.time_dim:ds[MCS.time_dim].astype("datetime64[s]")})

if MCS.convert_olr:
    bt = get_tb(ds[MCS.bt_var].compute()).to_iris()
else:
    bt = ds[MCS.bt_var].compute().to_iris()

dt = 3600 # in seconds 
dxy = 11100 # in meter (for Latitude)

tracks = tobac.feature_detection_multithreshold(
    bt,
    dxy=dxy,
    threshold=[241,233,225],
    n_min_threshold=10,
    target="minimum",
    position_threshold="weighted_diff",
    PBC_flag="hdim_2",
    statistics={"feature_min_BT": np.nanmin}
)

tracks = tobac.linking_trackpy(
    tracks,
    bt,
    dt,
    dxy,
    v_max=1e2,
    method_linking='predict',
    adaptive_stop=0.2,
    adaptive_step=0.95,
    stubs=3,
    PBC_flag="hdim_2",
    min_h2=0,
    max_h2=3600,
)

# Reduce tracks to only valid cells
tracks = tracks[tracks.cell!=-1]
track_min_bt = tracks.groupby("cell").feature_min_BT.min()
valid_cells = track_min_bt.index[track_min_bt < 225]
tracks = tracks[np.isin(tracks.cell, valid_cells)]

merges = tobac.merge_split.merge_split_MEST(
    tracks,
    dxy,
    frame_len=1
)

tracks = tracks.copy()
tracks["track"] = merges.feature_parent_track_id.data.astype(np.int64) + 1

track_start_time = tracks.groupby("track").time.min()
tracks["time_track"] = tracks.time - track_start_time[tracks.track].to_numpy()

segments, tracks = tobac.segmentation_2D(
    tracks,
    bt,
    dxy,
    threshold=241,
    target="minimum",
    PBC_flag="hdim_2"
)

tracks = calc_area_and_precip(tracks, segments, ds, MCS)

mcs_tracks = is_track_mcs(tracks)