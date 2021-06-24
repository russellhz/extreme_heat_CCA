#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rhorowitz
"""
import sys
import xarray as xr
import os
import glob
import re
import numpy as np
import pandas as pd
import argparse
import geopandas
import salem
import copy
import cftime
import random


sys.path.append("/glade/u/home/horowitz/extreme_heat_CCA/py/L0")
import L0_functions as L0

os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L1")
import L1_functions as L1

DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/"
CESMLE_DIR = "/glade/scratch/horowitz/extreme_heat_CCA/JJA_anom/"

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", 
        "Southeast", "Southwest", "UpperMidwest", "West"]  
PICTL_TYPE = "PICTL"

##############################################################################
################          Average SLP heatwave pattern        ################
##############################################################################
slp_data_dict = {}
for LOC in LOCS:
    fn = DIR + 'gridded_slp_z500/gridded_slp_z500_' + PICTL_TYPE + '_' + LOC + '.nc'
    
    # Find heatwave years
    dates = np.load(DIR + 'dates/top_dyn_composite_dates_' + PICTL_TYPE + '_' + LOC + '.npy', allow_pickle=True)
    years = np.unique([dates[i].year for i in range(len(dates))])
    
    # Open data        
    data = xr.open_dataset(fn)
    
    # Filter to heatwave years
    data = data.sel(time=data.time.dt.year.isin(years))
    
    # Filter to week of dates
    all_dates = data.time.values
    all_dates_df = pd.DataFrame({"date":all_dates})
    all_dates_df['year'] = [all_dates_df.date[x].year for x in range(len(all_dates_df))]
    week_of_df = all_dates_df.groupby(all_dates_df.year).head(14).groupby(all_dates_df.year).tail(7)

    data = data.sel(time=xr.DataArray(week_of_df.date, dims = "time", name="time"))

    # Take mean
    data = data.mean(dim='time')
    
    slp_data_dict[LOC] = data.slp 
    
##############################################################################
################             CESM-LE Correlation              ################
##############################################################################

cesmle_fns = [f for f in glob.glob(CESMLE_DIR + "CESM-LE_PSL_anom_" + "*" + "19200101-21001231.nc")]
cesmle_fns = np.sort(cesmle_fns)  

df_all = pd.DataFrame()


for fn in cesmle_fns:
    code = re.search(r'(?<=_)[0-9]{3}(?=_)', fn).group()
    data = xr.open_dataset(fn)
    data['lat'] = np.round(data['lat'], 2)
    for LOC in LOCS:
        heatwave = slp_data_dict[LOC]
        # Filter data to location
        data_loc = data.salem.subset(corners = ((heatwave.lon.min(),heatwave.lat.min()), 
                                                (heatwave.lon.max(),heatwave.lat.max())))
        nlat = heatwave.shape[0]
        nlon = heatwave.shape[1]
        ndays = data_loc.anom.shape[0]

        heatwave = heatwave.values.reshape((nlat*nlon)).reshape(1, -1)
        data_loc = data_loc.anom.values.reshape((ndays, nlat*nlon))

        corr = L1.corr2_coeff(heatwave, data_loc)
        df_loc = pd.DataFrame({'region':LOC, 'code': code, 'date':data.time.values, 'correlation':corr[0,:]})
        df_all = pd.concat([df_all, df_loc])
    print("Done with " + code)

df_all.to_csv(DIR+'cesmle_slp_correlation.csv')

# Correlation with mean

df_all = pd.DataFrame()

for fn in cesmle_fns:
    code = re.search(r'(?<=_)[0-9]{3}(?=_)', fn).group()
    data = xr.open_dataset(fn)
    data['lat'] = np.round(data['lat'], 2)
    for LOC in LOCS:
        heatwave = slp_data_dict[LOC]
        # Filter data to location
        data_loc = data.salem.subset(corners = ((heatwave.lon.min(),heatwave.lat.min()), 
                                                (heatwave.lon.max(),heatwave.lat.max()))).anom

        # Get mean in each summer
        data_loc_mean = data_loc.groupby(data_loc.time.dt.year).mean()

        nlat = heatwave.shape[0]
        nlon = heatwave.shape[1]
        nyears = data_loc_mean.shape[0]

        heatwave = heatwave.values.reshape((nlat*nlon)).reshape(1, -1)
        data_loc_mean_vals = data_loc_mean.values.reshape((nyears, nlat*nlon))

        corr = L1.corr2_coeff(heatwave, data_loc_mean_vals)
        df_loc = pd.DataFrame({'region':LOC, 'code': code, 'year':data_loc_mean.year.values, 'correlation':corr[0,:]})
        df_all = pd.concat([df_all, df_loc])
    print("Done with " + code)
df_all.to_csv(DIR+'cesmle_mean_slp_correlation.csv')