import pathlib
import sys
from datetime import datetime
import argparse
import warnings
import logging

import numpy as np
import pandas as pd
import xarray as xr

import tobac

cmd_folder = pathlib.Path.cwd().parent.absolute()
if cmd_folder not in sys.path:
    sys.path.insert(0, str(cmd_folder))

from utils import data_formats, get_tb, calc_area_and_precip, process_clusters, is_track_mcs

parser = argparse.ArgumentParser(description="""MCSMIP tracking using tobac""")
parser.add_argument("model", help="model to process", type=str)
parser.add_argument("season", help="season to process (summer or winter)", type=str)
parser.add_argument("-d", help="path to input data", default="../data_in", type=str)
parser.add_argument(
    "-s", help="path to save output data", default="../data_out", type=str
)

args = parser.parse_args()
model = args.model
season = args.season
data_path = pathlib.Path(args.d)
save_path = pathlib.Path(args.s)

dt = 3600  # in seconds
dxy = 11100  # in meter (for Latitude)

parameters_features = dict(
    dxy=dxy,
    threshold=[241, 233, 225],
    n_min_threshold=10,
    min_distance=10*dxy,
    target="minimum",
    position_threshold="center",
    PBC_flag="hdim_2",
    statistic={"feature_min_BT": np.nanmin},
)

parameters_segments = dict(
    threshold=241, 
    target="minimum", 
    PBC_flag="hdim_2", 
    seed_3D_flag="box", 
    seed_3D_size=11, 
    statistic={"mean_BT": np.nanmean},
)
    
parameters_tracking = dict(
    d_max=10*dxy,
    method_linking="predict",
    adaptive_stop=0.2,
    adaptive_step=0.95,
    stubs=2,
    memory=0,
    PBC_flag="hdim_2",
    min_h2=0,
    max_h2=3600,
)

parameters_merge = dict(
    distance=dxy*20, 
    frame_len=2, 
    PBC_flag="hdim_2", 
    min_h1=0, 
    max_h1=1200, 
    min_h2=0, 
    max_h2=3600,
)

def main() -> None:
    if season == "summer":
        dates = pd.date_range(datetime(2016, 8, 1), datetime(2016, 9, 10), freq="h", inclusive="left")
        for MCS in data_formats.summer_datasets:
            if MCS.name == model:
                break
        else:
            raise ValueError(f"model {model} not found for season {season}")

    if season == "winter":
        dates = pd.date_range(datetime(2020, 1, 20), datetime(2020, 2, 29), freq="h", inclusive="left")
        for MCS in data_formats.winter_datasets:
            if MCS.name == model:
                break
        else:
            raise ValueError(f"model {model} not found for season {season}")

    files = sorted(sum([MCS.glob_date(data_path, season, date) for date in dates], []))
    print(datetime.now(), f"Loading {len(files)} files", flush=True)
    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim=MCS.time_dim, join="override"
    )
    ds = ds.assign_coords({MCS.time_dim: ds[MCS.time_dim].astype("datetime64[s]")})

    if MCS.convert_olr:
        bt = get_tb(ds[MCS.bt_var].compute())
    else:
        bt = ds[MCS.bt_var].compute()

    dt = 3600  # in seconds
    dxy = 11100  # in meter (for Latitude)

    print(datetime.now(), f"Commencing feature detection", flush=True)
    features = tobac.feature_detection_multithreshold(
        bt, **parameters_features,
    )

    # Convert feature_min_BT to float dtype as the default of 'None' means that it will be an object array
    features["feature_min_BT"] = features["feature_min_BT"].to_numpy().astype(float)

    print(datetime.now(), f"Commencing tracking", flush=True)
    class DisableLogger():
        def __enter__(self):
            logging.disable(logging.CRITICAL)
        def __exit__(self, exit_type, exit_value, exit_traceback):
            logging.disable(logging.NOTSET)

    with DisableLogger():
        features = tobac.linking_trackpy(features, bt.to_iris(), dt, dxy, **parameters_tracking)

    # Reduce tracks to only valid cells
    features = features[features.cell != -1]
    track_min_bt = features.groupby("cell").feature_min_BT.min()
    valid_cells = track_min_bt.index[track_min_bt < 225]
    features = features[np.isin(features.cell, valid_cells)]

    print(datetime.now(), f"Calculating merges and splits", flush=True)
    merges = tobac.merge_split.merge_split_MEST(features, dxy, **parameters_merge)

    features["track"] = merges.feature_parent_track_id.data.astype(np.int64)

    track_start_time = features.groupby("track").time.min()
    features["time_track"] = features.time - track_start_time[features.track].to_numpy()

    print(datetime.now(), f"Commencing segmentation", flush=True)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Warning: converting a masked element to nan.*",
    )
    segments, features = tobac.segmentation.segmentation(features, bt.to_iris(), dxy, **parameters_segments)

    features["time"] = xr.CFTimeIndex(features["time"].to_numpy()).to_datetimeindex()
    features["mean_BT"] = features["mean_BT"].to_numpy().astype(float)

    print(datetime.now(), f"Processing MCS properties", flush=True)
    features = calc_area_and_precip(features, segments, ds, MCS, inplace=True)

    features, clusters = process_clusters(features)
    mcs_flag = is_track_mcs(features)

    # Prepare output dataset
    print(datetime.now(), f"Preparing output", flush=True)
    out_ds = features.set_index(features.feature).to_xarray()

    out_ds = out_ds.rename_vars(
        {
            "cell": "feature_cell_id",
            "track": "feature_track_id",
            "cluster": "feature_cluster_id",
            "hdim_1": "y",
            "hdim_2": "x",
            "num": "detection_pixel_count",
            "feature_min_BT": "min_BT",
            "ncells": "segmentation_pixel_count",
        }
    )

    cell_track_id = features.groupby("cell").track.first()

    out_ds = out_ds.assign_coords(cell_track_id.to_xarray().coords)
    out_ds = out_ds.assign_coords(clusters.to_xarray().coords)
    out_ds = out_ds.assign_coords(mcs_flag.to_xarray().coords)

    out_ds["cell_track_id"] = cell_track_id.to_xarray()
    out_ds = xr.merge([out_ds, clusters.to_xarray()])
    out_ds["is_track_mcs"] = mcs_flag.to_xarray()

    feature_is_mcs = out_ds.track_is_mcs.loc[out_ds.feature_track_id]

    out_ds["feature_is_mcs"] = feature_is_mcs
    out_ds["cell_track_id"] = merges.cell_parent_track_id
    out_ds["track_child_cell_count"] = merges.track_child_cell_count
    out_ds["cell_child_feature_count"] = merges.cell_child_feature_count
    out_ds["cell_starts_with_split"] = merges.cell_starts_with_split
    out_ds["cell_ends_with_merge"] = merges.cell_ends_with_merge

    all_feature_labels = xr.DataArray.from_iris(segments)
    all_feature_labels.name = "all_feature_labels"

    mcs_feature_labels = all_feature_labels * np.isin(
        all_feature_labels, out_ds.feature.values[out_ds.feature_is_mcs]
    )
    mcs_feature_labels.name = "mcs_feature_labels"

    # Map feature labels to cells and tracks
    all_cell_labels = all_feature_labels.copy()
    all_cell_labels.name = "all_cell_labels"
    all_track_labels = all_feature_labels.copy()
    all_track_labels.name = "all_track_labels"

    wh_all_labels = np.flatnonzero(all_feature_labels)

    all_cell_labels.data.ravel()[wh_all_labels] = out_ds.feature_cell_id.loc[
        all_feature_labels.data.ravel()[wh_all_labels]
    ]
    all_track_labels.data.ravel()[wh_all_labels] = out_ds.feature_track_id.loc[
        all_feature_labels.data.ravel()[wh_all_labels]
    ]

    mcs_cell_labels = mcs_feature_labels.copy()
    mcs_cell_labels.name = "mcs_cell_labels"
    mcs_track_labels = mcs_feature_labels.copy()
    mcs_track_labels.name = "mcs_track_labels"

    wh_mcs_labels = np.flatnonzero(mcs_feature_labels)

    mcs_cell_labels.data.ravel()[wh_mcs_labels] = out_ds.feature_cell_id.loc[
        mcs_feature_labels.data.ravel()[wh_mcs_labels]
    ]
    mcs_track_labels.data.ravel()[wh_mcs_labels] = out_ds.feature_track_id.loc[
        mcs_feature_labels.data.ravel()[wh_mcs_labels]
    ]

    out_ds = out_ds.assign_coords(all_feature_labels.coords)

    out_ds = xr.merge(
        [
            out_ds,
            all_feature_labels,
            mcs_feature_labels,
            all_cell_labels,
            mcs_cell_labels,
            all_track_labels,
            mcs_track_labels,
        ]
    )

    out_ds = out_ds.assign_attrs(
        title=f"{season} {model} MCS mask file",
        model=f"{model}",
        season=f"{season}",
        tracker="tobac",
        version=f"{tobac.__version__}",
        feature_detection_parameters=str(parameters_features),
        tracking_parameters=str(parameters_tracking),
        segmentation_parameters=str(parameters_segments),
        merge_split_parameters=str(parameters_merge),
        created_on=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        creator="William K Jones",
        history="Dataset produced using _tobac_ on the JASMIN HPC system"
    )

    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in out_ds.data_vars:
        var_type = out_ds[var].dtype
        if np.issubdtype(var_type, np.integer) or np.issubdtype(var_type, np.floating):
            out_ds[var].encoding.update(comp)

    out_ds.to_netcdf(save_path / f"tobac_{model}_{season}_MCS_mask_file.nc")

    ds.close()
    out_ds.close()


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time, "Commencing MCS detection", flush=True)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    print("Model:", model, flush=True)
    print("Season:", season, flush=True)
    print("Input save path:", data_path, flush=True)
    print("Output save path:", save_path, flush=True)

    main()

    print(
        datetime.now(),
        "Finished successfully, time elapsed:",
        datetime.now() - start_time,
        flush=True,
    )
