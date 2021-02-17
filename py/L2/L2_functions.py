#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rhorowitz
"""

import xarray as xr
import os
import glob
import re
import numpy as np
import pandas as pd
import random
import argparse
from itertools import chain
import geopandas
import salem
import math
import string
import copy
import scipy
import random
import statsmodels.api as sm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")


def remove_duplicate_grid(df_tmp, lat1, lat2, lon1, lon2):
    df_tmp = df_tmp.reset_index()
    remove_df = df_tmp[(df_tmp.lat > lat1) & (df_tmp.lat < lat2) & (df_tmp.lon >= lon1) & (df_tmp.lon <= lon2)]
    df_tmp = df_tmp[~((df_tmp.lat.isin(remove_df.lat)) & (df_tmp.lon.isin(remove_df.lon))) ]
    return df_tmp.set_index(['lat', 'lon']) 

def tas_JJA_var_data_combiner(files):
    df = pd.DataFrame()
    for fn in files:
        data = xr.open_dataset(fn)
        # Fix longitude
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby("lon")
        df_tmp = data.to_dataframe().dropna()
        # Remove duplicate grid points
        if "_Southwest_" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 40.5, 41, -106, -105) 
        if "_UpperMidwest_" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 41, 42, -97, -96) 
        if "_OhioValley_" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 35, 36, -84, -83)  
            
        df = df.append(df_tmp)
        
    return df.to_xarray()

def shapefile_transform(fn):
    shapefile = geopandas.read_file(fn)
    shapefile.crs =  {'init': 'epsg:3857'}
    shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
    shapefile.crs = 'epsg:4326'
    return shapefile

def avg_top_dyn_data_combiner(files, var):
    df = pd.DataFrame()
    for fn in files:
        # Take time mean
        data = xr.open_dataset(fn)[var]
        
        # Filter to heatweek dates
        max_dates = data.time.groupby(data.time.dt.year).max().values
        heatweek_dates = []
        for date in max_dates:
            tmp = xr.cftime_range(end=date, periods= 7, freq="D", calendar="noleap").to_list()
            heatweek_dates = heatweek_dates + tmp
        data = L1.time_index_fun(data, heatweek_dates)
        
        data = data.mean(dim='time')
        
        df_tmp = data.to_dataframe().dropna()
        # Remove duplicate grid points
        if "_Southwest.nc" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 40.5, 41, -106, -105) 
        if "_UpperMidwest.nc" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 41, 42, -97, -96) 
        if "_OhioValley.nc" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 35, 36, -84, -83)  
            
        df = df.append(df_tmp)
        
    return df.to_xarray()

def corr_top_dyn_data_combiner(files, var1, var2):
    df = pd.DataFrame()
    for fn in files:
        region = re.search('(?<=PICTL_).+(?=.nc)', fn).group()
        # Take time mean
        data = xr.open_dataset(fn)
        
        # Filter to heatweek dates
        max_dates = data.time.groupby(data.time.dt.year).max().values
        
        heatweek_dates = []
        weeklag_dates = []
        for date in max_dates:
            tmp = xr.cftime_range(end=date, periods= 14, freq="D", calendar="noleap").to_list()
            weeklag_dates = weeklag_dates + tmp[0:7]           
            heatweek_dates = heatweek_dates + tmp[7:14]  
        var1_week_of = L1.time_index_fun(data[var1], heatweek_dates)
        var2_week_lag = L1.time_index_fun(data[var2], weeklag_dates)
        
        var1_week_of = var1_week_of.groupby(var1_week_of.time.dt.year).mean()
        var2_week_lag = var2_week_lag.groupby(var2_week_lag.time.dt.year).mean()
        
        data = xr.corr(var1_week_of, var2_week_lag, dim = 'year')
        weights = np.cos(np.deg2rad(data.lat))
        weights.name = "weights" 
        weighted_mean = data.weighted(weights).mean(("lon", "lat")).values.tolist()
        print(region + ' mean = ' + str(np.round(weighted_mean, 2)))
        
        df_tmp = data.to_dataframe(name='corr').dropna()
        
        # Calculate p-values
        n=len(var1_week_of)
        dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
        pvals = 2*dist.cdf(-abs(data))
        pvals = xr.DataArray(pvals,coords=[data.lat, data.lon], dims=['lat', 'lon'], name = 'pvalue')
        pvals = pvals.to_dataframe(name='pvalue').dropna()
        
        # Add p-values to df with correlations
        df_tmp = df_tmp.merge(pvals, how='left', left_index=True, right_index=True)
        
        # Remove duplicate grid points
        if "_Southwest.nc" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 40.5, 41, -106, -105) 
        if "_UpperMidwest.nc" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 41, 42, -97, -96) 
        if "_OhioValley.nc" in fn:
            df_tmp = remove_duplicate_grid(df_tmp, 35, 36, -84, -83)  
            
        df = df.append(df_tmp)
        print('done with ' + region)
    return df.to_xarray()

# Function takes in a dataframe with date column
# returns length of each pattern and year associated with it
def persistence_calculator(data):
    lengths = []
    years = []
    data['date_diff'] = data.date.diff()
    x = 1

    for i in range(len(data)-1):
        if data['date_diff'][i+1].days > 1:
            lengths.append(x)
            years.append(data.date[i].year)
            x = 1
        if data['date_diff'][i+1].days == 1: 
            x += 1 
    lengths.append(x)
    years.append(data.date[len(data)-1].year)
    return lengths, years

