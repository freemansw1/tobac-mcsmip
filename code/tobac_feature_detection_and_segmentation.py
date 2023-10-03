'''
Python script to run feature detection and segmentation with tobac to identify mesoscale convective systems (MCS) on global grid. 

Email: kukulies@ucar.edu
'''

from pathlib import Path 
import numpy as np 
from datetime import datetime 
import xarray as xr 
import pandas as pd 
import tobac

# working directories
savedir = Path('/glade/scratch/kukulies/MCSMIP/test_runs/out/') 
input_data = Path('/glade/scratch/kukulies/MCSMIP/DYAMOND/')

# set experiment, models (or obs data) and date range 
experiment = 'Winter'
models= ['MPAS', 'OBS']
year = 2020
month = 2

start_day = 21 
start_hour = 0 
end_day = 28 
end_hour = 23 

########################################## input parameter  #############################################

# feature detection 
parameters_features={}
parameters_features['position_threshold']='weighted_diff'
parameters_features['n_min_threshold']= 10
parameters_features['target']='minimum'
parameters_features['threshold']=[241,239,237,235,233,231,229,227,225]
parameters_features['PBC_flag'] = 'both'

# segmentation
parameters_segmentation={}
parameters_segmentation['target']='minimum'
parameters_segmentation['method']='watershed'
parameters_segmentation['threshold']=242

# define spatial and temporal resolution of input data dxy, dt 
dt = 3600 
dxy = 13801

##############################################################xxxx############################################

# function to convert OLR to Tb 

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

# get paths and date objects from input 
start= datetime(year, month, start_day, start_hour)
end = datetime(year, month, end_day, end_hour)
date_range = pd.date_range(start, end, freq = 'H')

experiment_path = input_data / Path(experiment)
model_paths = [experiment_path / Path(model) for model in models]


###################### run feature detection and segmentation for each day ###################################
for idx, model_path in enumerate(model_paths):
    model = models[idx]
    
    print('start processing for ', model)
    # get all files for one day in daterange
    for day in date_range.day.unique():
        date_str = str(year) + str(month).zfill(2) + str(day).zfill(2)
    
        # read in all hourly files for day
        if model == 'OBS':
            hourly_files = list((model_path / 'tb_pcp').glob("merg_*" + date_str+ "*nc") ) 
            assert len(hourly_files) == 24 
            tbb = xr.open_mfdataset(hourly_files).Tb
        else:
            hourly_files = list((model_path / 'olr_pcp_instantaneous').glob("pr_rlut_*" + date_str+ "*nc") ) 
            assert len(hourly_files) == 24 
            olr = xr.open_mfdataset(hourly_files).rltacc
            tbb = get_tb(olr)
        # change standard name to cf-compliant (needed for internal conversion to iris)                                                                       
        tbb.attrs['standard_name'] = 'brightness_temperature'
        tbb_iris = tbb.to_iris()
        
        # Perform feature detection observations
        print('Starting feature detection for ', date_str)
        Features= tobac.feature_detection_multithreshold(tbb_iris,dxy,**parameters_features)
        Features.to_xarray().to_netcdf( savedir /  ('Features_' + model + '_'+ date_str+ '.nc'))
        
        # Perform segmentation and save results
        print('Starting segmentation for ', date_str)
        Mask, Features_seg= tobac.segmentation.segmentation(Features,tbb_iris,dxy,**parameters_segmentation)
        xr.DataArray.from_iris(Mask).to_netcdf(savedir / ('Mask_' + model + '_' +  date_str + '.nc'))                
        Features_seg.to_xarray().to_netcdf(savedir / ('Features_seg_' + model + '_'+date_str +'.nc'))
        del Mask, Features, Features_seg
        





