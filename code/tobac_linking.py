"""
This python script performs the trajectory linking of detected cloud features using the tracking algorithm tobac.
The input are features identified in global brightness temperature data over ten days. 

Email: kukulies@ucar.edu

"""

from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import tobac
from tobac import utils, linking_trackpy
import logging

########################################## user specific parameters #############################################

# directory that contains identified features and segmentation mask files
data_dir = Path("/glade/scratch/kukulies/MCSMIP/test_runs/out/")

# dictionary with parameters for linking
parameters_linking = {}
parameters_linking["v_max"] = 100
parameters_linking["stubs"] = 4
parameters_linking["order"] = 1
parameters_linking["extrapolate"] = 0
parameters_linking["memory"] = 0
parameters_linking["adaptive_stop"] = 0.2
parameters_linking["adaptive_step"] = 0.95
parameters_linking["method_linking"] = "predict"

dt = 3600  # in seconds
dxy = 13801  # in meter

# specify models and time period
models = ["OBS", "MPAS"]

# specify period over which to track in one go
year = 2020
month = 2
start_day = 21
start_hour = 0
end_day = 28
end_hour = 23
start = datetime(year, month, start_day, start_hour)
end = datetime(year, month, end_day, end_hour)
date_range = pd.date_range(start, end, freq="H")

# input data (can be any file, not really needed for tracking)
tbb_xr = xr.open_dataset(
    "/glade/scratch/kukulies/MCSMIP/DYAMOND/Winter/OBS/tb_pcp/merg_2020022823_4km-pixel.nc"
).Tb
tbb = tbb_xr.to_iris()

################################## Trajectory linking of features ###############################################

for model in models:
    logging.info("Start processing for ", model)
    # for each day in date range over which track:
    # read in file, if exists and append to dataframe list
    dataframe_list = []
    for day in date_range.day.unique():
        date_str = str(year) + str(month).zfill(2) + str(day).zfill(2)
        print(date_str)
        daily_file = list(data_dir.glob("Features_" + model + "_" + date_str + ".nc"))[
            0
        ]
        if daily_file.exists():
            feature_df = xr.open_dataset(daily_file).to_dataframe()
            dataframe_list.append(feature_df)
        else:
            logging.warning("no features found for ", date_str)

    # combine all feature dataframes into one
    assert len(dataframe_list) == 8
    all_features = utils.combine_feature_dataframes(
        dataframe_list, old_feature_column_name="daily_features"
    )

    #### perform linking of features ####
    logging.info(
        "Starting trajectory linking for ", model, "from ", str(start), "to", str(end)
    )
    tracks = linking_trackpy(all_features, tbb, dt=dt, dxy=dxy, **parameters_linking)
    tracks.to_xarray().to_netcdf(
        data_dir
        / ("Tracks_" + model + "_" + str(start)[0:10] + "-" + str(end)[0:10] + ".nc")
    )
    logging.info("All features linked and saved in " + str(data_dir))
