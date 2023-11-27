import pathlib
import sys
from datetime import datetime
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import tobac

cmd_folder = pathlib.Path.cwd().parent.absolute()
if cmd_folder not in sys.path:
    sys.path.insert(0, str(cmd_folder))

from utils import data_formats, get_tb, calc_area_and_precip, is_track_mcs

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


def main() -> None:
    if season == "summer":
        dates = pd.date_range(datetime(2016, 8, 1), datetime(2016, 9, 9), freq="D")
        for MCS in data_formats.summer_datasets:
            if MCS.name == model:
                break
        else:
            raise ValueError(f"model {model} not found for season {season}")

    if season == "winter":
        dates = pd.date_range(datetime(2020, 1, 20), datetime(2020, 2, 28), freq="D")
        for MCS in data_formats.winter_datasets:
            if MCS.name == model:
                break
        else:
            raise ValueError(f"model {model} not found for season {season}")

    files = sorted(sum([MCS.glob_date(data_path, season, date) for date in dates], []))
    print(datetime.now(), f"Loading {len(files)} files", flush=True)
    ds = xr.open_mfdataset(files, combine="nested", concat_dim=MCS.time_dim)
    ds = ds.assign_coords({MCS.time_dim: ds[MCS.time_dim].astype("datetime64[s]")})

    if MCS.convert_olr:
        bt = get_tb(ds[MCS.bt_var].compute()).to_iris()
    else:
        bt = ds[MCS.bt_var].compute().to_iris()

    dt = 3600  # in seconds
    dxy = 11100  # in meter (for Latitude)

    print(datetime.now(), f"Commencing feature detection", flush=True)
    features = tobac.feature_detection_multithreshold(
        bt,
        dxy=dxy,
        threshold=[241, 233, 225],
        n_min_threshold=10,
        target="minimum",
        position_threshold="weighted_diff",
        PBC_flag="hdim_2",
        statistics={"feature_min_BT": np.nanmin},
    )

    # Convert feature_min_BT to float dtype as the default of 'None' means that it will be an object array
    features["feature_min_BT"] = features["feature_min_BT"].to_numpy().astype(float)

    print(datetime.now(), f"Commencing tracking", flush=True)
    features = tobac.linking_trackpy(
        features,
        bt,
        dt,
        dxy,
        v_max=1e2,
        method_linking="predict",
        adaptive_stop=0.2,
        adaptive_step=0.95,
        stubs=3,
        PBC_flag="hdim_2",
        min_h2=0,
        max_h2=3600,
    )

    # Reduce tracks to only valid cells
    features = features[features.cell != -1]
    track_min_bt = features.groupby("cell").feature_min_BT.min()
    valid_cells = track_min_bt.index[track_min_bt < 225]
    features = features[np.isin(features.cell, valid_cells)]

    print(datetime.now(), f"Calculating merges and splits", flush=True)
    merges = tobac.merge_split.merge_split_MEST(features, dxy, frame_len=1)

    features["track"] = merges.feature_parent_track_id.data.astype(np.int64) + 1

    track_start_time = features.groupby("track").time.min()
    features["time_track"] = features.time - track_start_time[features.track].to_numpy()

    print(datetime.now(), f"Commencing segmentation", flush=True)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Warning: converting a masked element to nan.*",
    )
    segments, features = tobac.segmentation_2D(
        features, bt, dxy, threshold=241, target="minimum", PBC_flag="hdim_2"
    )

    print(datetime.now(), f"Processing MCS properties", flush=True)
    features = calc_area_and_precip(features, segments, ds, MCS, inplace=True)

    mcs_flag = is_track_mcs(features)

    features["time"] = xr.CFTimeIndex(features["time"].to_numpy()).to_datetimeindex()

    # Prepare output dataset
    print(datetime.now(), f"Preparing output", flush=True)
    out_ds = features.set_index(features.feature).to_xarray()

    out_ds = out_ds.rename_vars(
        {
            "cell": "feature_cell_id",
            "track": "feature_track_id",
            "hdim_1": "y",
            "hdim_2": "x",
            "num": "detection_pixel_count",
            "feature_min_BT": "min_BT",
            "ncells": "segmentation_pixel_count",
        }
    )

    out_ds["track_is_mcs"] = mcs_flag.to_xarray()[0]

    feature_is_mcs = out_ds.track_is_mcs.loc[out_ds.feature_track_id]

    out_ds["feature_is_mcs"] = feature_is_mcs

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

    out_ds.assign_attrs(
        title=f"{season} {model} MCS mask file",
        model=f"{model}",
        season=f"{season}",
        tracker="tobac",
        version=f"{tobac.__version__}",
        feature_detection_parameters='threshold=[241, 233, 225], n_min_threshold=10, target="minimum", position_threshold="weighted_diff", PBC_flag="hdim_2", statistics={"feature_min_BT": np.nanmin}',
        tracking_parameters='v_max=1e2, method_linking="predict", adaptive_stop=0.2, adaptive_step=0.95, stubs=3, PBC_flag="hdim_2", min_h2=0, max_h2=3600,',
        segmentation_parameters='threshold=241, target="minimum", PBC_flag="hdim_2",',
        merge_split_parameters="frame_len=1,",
        created_on=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )

    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in out_ds.data_vars:
        out_ds[var].encoding.update(comp)

    out_ds.to_netcdf(save_path / f"tobac_{model}_{season}_MCS_mask_file.nc")

    ds.close()
    out_ds.close()


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time, "Commencing MCS detection", flush=True)
    if not save_path.exists():
        save_path.mkdir()

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
