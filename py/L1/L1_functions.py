#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:55:05 2020

@author: rhorowitz
"""
import glob
import re
import xarray as xr
import numpy as np
import sys
import os
import pandas as pd
import salem
import geopandas
import random


sys.path.append("../L0")
import L0_functions as L0

################################################################################

def covar_adder_shape(code, key, var, path, pattern, dates, all_data, shapefile, pictl_type):
    covar_dir = path + key + '/proc/tseries/daily/'
    var_dir = covar_dir + var.upper() + '/' 
    # List possible files
    if pictl_type == "SSTPICTL" or pictl_type ==  "PICTL":
        var_files = [f for f in glob.glob(var_dir + pattern + "*.nc")]
    if pictl_type == "cesmLE":
        var_files = [f for f in glob.glob(var_dir + pattern + "." + code + ".*.nc")]
        
    # For pictl, need to find file that contains code through code + 51
    # Doesn't support more than 2 files
    covar_fns = covar_fn_finder(var_files, code, pictl_type, key)
    
    if len(covar_fns) == 1:
        tmp = covar_data_open_shape(covar_fns[0], var, shapefile)
    elif len(covar_fns) == 2:
        tmp1 = covar_data_open_shape(covar_fns[0], var, shapefile)
        tmp2 = covar_data_open_shape(covar_fns[1], var, shapefile)
        tmp = xr.concat([tmp1, tmp2], dim = "time")
        print("merged two datasets")
        # dates
        date_years = np.unique(all_data.time.dt.year.values)
        tmp = tmp.sel(time = tmp.time.dt.year.isin(date_years))
    else:
        print("ERROR 0 or > 2 covar fns")
    print(var + " file opened and processed")
    
    tmp = tmp.to_dataset().assign(anom = (('time', 'lat', 'lon'), L0.seasonality_removal_vals(tmp.values, k=3)))
    print("seasonality removed")
    if pictl_type == "cesmLE":
        datetimeindex = tmp.indexes['time'].to_datetimeindex()
        tmp['time'] = datetimeindex
    tmp = tmp.rename({'anom': var})[var].sel(time = tmp.time.isin(dates))
    tmp = tmp.assign_coords(lon=(((tmp.lon + 180) % 360) - 180)).sortby("lon")
    print("fixed longitude coordinates")
    tmp['lat'] = np.round(np.float64(tmp.lat), 2)    
    all_data = xr.merge([all_data, tmp])
    print("all_vars updated with " + var)
    return all_data

################################################################################

def covar_data_open_shape(fn, var, shapefl):
        tmp = xr.open_dataset(fn)[var.upper()]
        print("data opened")
        #tmp = tmp.assign_coords(lon=(((tmp.lon + 180) % 360) - 180)).sortby("lon")
        #print("fixed longitude coordinates")
        # Filter to spatial range
        tmp = tmp.salem.subset(shape = shapefl)
        print("data subsetted")
        return tmp

################################################################################

def covar_adder(code, key, var, path, pattern, dates, all_data, lat_bnd, lon_bnd, pictl_type):
    covar_dir = path + key + '/proc/tseries/daily/'
    var_dir = covar_dir + var.upper() + '/' 
    # List possible files
    var_files = [f for f in glob.glob(var_dir + pattern + "*.nc")]
    
    # For pictl, need to find file that contains code through code + 51
    # Doesn't support more than 2 files
    covar_fns = covar_fn_finder(var_files, code, pictl_type, key)
    
    if len(covar_fns) == 1:
        tmp = covar_data_open(covar_fns[0], var, lat_bnd, lon_bnd)
    elif len(covar_fns) == 2:
        tmp1 = covar_data_open(covar_fns[0], var, lat_bnd, lon_bnd)
        tmp2 = covar_data_open(covar_fns[1], var, lat_bnd, lon_bnd)
        tmp = xr.concat([tmp1, tmp2], dim = "time")
        print("merged two datasets")
        # dates
        date_years = np.unique([x.year for x in dates])
        tmp = tmp.sel(time = tmp.time.dt.year.isin(date_years))
    else:
        print("ERROR 0 or > 2 covar fns")
    print(var + " file opened and processed")
    
    tmp = tmp.to_dataset().assign(anom = (('time', 'lat', 'lon'), L0.seasonality_removal_vals(tmp.values, k=3)))
    print("seasonality removed")
    tmp = tmp.rename({'anom': var})[var].sel(time = tmp.time.isin(dates))
    tmp['lat'] = np.round(np.float64(tmp.lat), 2)    
    all_data = all_data.update({var : tmp})
    print("all_vars updated with " + var)
    return all_data

################################################################################

def covar_fn_finder(files, code, pictl_type, key):
    years = [re.search(r'[0-9]{8}-[0-9]{8}(?=.nc)', i).group() for i in files]
    start_years = [re.search(r'[0-9]{4}(?=[0-9]{4}-)', i).group() for i in years]
    end_years = [re.search(r'(?<=[0-9]{4}-)[0-9]{4}', i).group() for i in years]
    
    # Distance measure to identify 1 or 2 files needed
    # Doesn't support more than 2 files
    if pictl_type == "SSTPICTL":
        start_scores = [code + 1 - int(x) for x in start_years]
        end_scores = [int(x) - (code + 51) for x in end_years]
    elif pictl_type == "PICTL":
        if key == "lnd" and code == 0:
            start_scores = [code + 403 - int(x) for x in start_years]
            end_scores = [int(x) - (code + 452) for x in end_years]
        else:
            start_scores = [code + 402 - int(x) for x in start_years]
            end_scores = [int(x) - (code + 452) for x in end_years]
    
    min_idx = start_scores.index(min([x for x in start_scores if x >= 0]))
    
    if end_scores[min_idx] >= 0:
        covar_fns = [files[i] for i in [min_idx]]  
    
    elif end_scores[min_idx] < 0:
        max_idx = end_scores.index(min([x for x in end_scores if x >= 0]))
        covar_fns = [files[i] for i in [min_idx, max_idx]]  
    return covar_fns

################################################################################

def covar_data_open(fn, var, lat_bnd, lon_bnd):
        tmp = xr.open_dataset(fn)[var.upper()]
        print("data opened")
        tmp = tmp.assign_coords(lon=(((tmp.lon + 180) % 360) - 180)).sortby("lon")
        print("fixed longitude coordinates")
        # Filter to spatial range, then do seasonality removal
        tmp = tmp.sel(lat = slice(lat_bnd[0], lat_bnd[1]), 
                          lon = slice(lon_bnd[0], lon_bnd[1]))
        return tmp

################################################################################

def week_calc(df, start, end, suffix):
    tmp = df[df.day.ge(start) & df.day.le(end)]
    return tmp.groupby('year').apply(lambda x: x.mean(skipna=False)).drop(columns='day').add_suffix(suffix)

################################################################################

def top_bottom_sel(data, resid_comp, pictl_type, lag=0):
    resid_gridded = data.sel(time = data.time.dt.year.isin(resid_comp.year.values))
    max_years = resid_gridded.time.groupby(resid_gridded.time.dt.year).max()
    if pictl_type == "SSTPICTL" or pictl_type == "PICTL":
        start_dates = [xr.cftime_range(end = x, periods = 7 + lag)[0] for x in max_years.values]
        dates = [xr.cftime_range(start = x, periods = 7) for x in start_dates]
    elif pictl_type == "cesmLE":
        start_dates = [pd.date_range(end = x, periods = 7 + lag)[0] for x in max_years.values]
        dates = [pd.date_range(start = x, periods = 7) for x in start_dates]        
    resid_gridded = resid_gridded.sel(time = resid_gridded.time.isin(dates))
    return resid_gridded.groupby(resid_gridded.time.dt.year).mean()

################################################################################

def composite_sel(data, resid_comp, COMPOSITE_LENGTH):
    resid_gridded = data.sel(time = data.time.dt.year.isin(resid_comp.year.values))
    df = resid_gridded.mean(dim = ["lon", "lat"]).to_dataframe()
    days = []
    for yr in np.unique(resid_gridded.time.dt.year):
        days.extend(range(1,COMPOSITE_LENGTH+1))
    df['day'] = days
    return df

################################################################################

def composite_sel_cesmLE(data, resid_comp, code):
    resid_gridded = data.sel(time = data.time.dt.year.isin(resid_comp.year.values))
    df = resid_gridded.mean(dim = ["lon", "lat"]).to_dataframe()
    days = []
    for yr in np.unique(resid_gridded.time.dt.year):
        days.extend(range(1,29))
    df['day'] = days
    return df

################################################################################

def var_std(key, var, path, pattern, shapefile, pictl_type, land):
    covar_dir = path + key + '/proc/tseries/daily/'
    var_dir = covar_dir + var.upper() + '/' 
    # List possible files
    if pictl_type == "SSTPICTL" or pictl_type ==  "PICTL":
        var_files = [f for f in glob.glob(var_dir + pattern + "*.nc")]
    if pictl_type == "cesmLE":
        var_files = [f for f in glob.glob(var_dir + pattern + "." + code + ".*.nc")]
    var_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    i = random.randint(0, len(var_files)-1)
    # open data
    data = xr.open_dataset(var_files[i])[var.upper()]
    
    print(var + ' data opened')
    
    # filter to spatial range and apply land mask
    data = data.salem.subset(shape=shapefile)
    data = data.to_dataset().assign(anom = (('time', 'lat', 'lon'), L0.seasonality_removal_vals(data.values, k=3)))
    
    print('seasonality removed')

    data = data.sel(time = data.time.dt.month.isin(range(6,9)))
    data = data.anom.salem.roi(shape = shapefile)
    data['lat'] =  np.round(data.lat, 2)
    # Soilwater lat values are float32 and land mask are float64
    if key == 'atm':
        std = np.std(data * land.mask)
    else:
        std = np.std(data)
    
    print('standard deviation calculated')
    print('----------------------------------------------------------------------------')
    return std

################################################################################
def hottest_week_finder(tas, filter_years, land, shapefile, length):
    tas = tas.assign_coords(lon=(((tas.lon + 180) % 360) - 180))
    tas['lat']  =  np.round(tas.lat, 2)

    # Cut to top_intdyn years, limit to desired spatial range, and then find hottest week
    tas_tmp = tas.anom.sel(time = tas.time.dt.year.isin(filter_years)).sortby("lon")
    tas_tmp = tas_tmp * land.mask

    print("land mask applied")

    # Desired spatial range
    tas_tmp = tas_tmp.salem.roi(shape = shapefile)

    # Spatial mean & limit to JJA (restricts to 6/04-8/28)
    tas_tmp_mean = tas_tmp.mean(dim = ["lon", "lat"]).sel(time = tas_tmp.time.dt.month.isin(range(6,9)))

    # convert to df, then rolling mean grouped by year
    df = pd.DataFrame({'date':tas_tmp_mean.time, 'year':tas_tmp_mean['time.year'].values, 'tas':tas_tmp_mean})
    roll_tas = df.groupby('year')['tas'].rolling(7, center = True).mean()
    roll_tas = roll_tas.reset_index().drop(columns = 'level_1').rename(columns={'tas': 'tas_roll'})
    df = df.merge(roll_tas, how='left', on='year', left_index=True, right_index=True)

    # Get top score in each year, then trim to top n years
    maxheat_idx = df.groupby(['year'])['tas_roll'].transform(max) == df['tas_roll']
    heatwave_dates = df[maxheat_idx].date.tolist()

    # List of all dates from 24 days before to 3 days after peak
    all_dates = list()
    for date in heatwave_dates:
        dates_begin = xr.cftime_range(end=date, periods=length-3, freq="D", calendar="noleap").to_list()[:-1]
        dates_end = xr.cftime_range(start=date, periods= 4, freq="D", calendar="noleap").to_list()
        all_dates = all_dates + dates_begin + dates_end
    print("hottest weeks obtained")    
    return all_dates
    
################################################################################

def top_bottom_years(LOC, PICTL_TYPE, var, thresh):
    filedir =  "/glade/work/horowitz/forced_response/US_climate_regions/" + LOC + "/L1/" + PICTL_TYPE + '/'
    sum_files = [f for f in 
                     glob.glob(filedir + "weekof_sum_data_" + LOC + "*" + PICTL_TYPE + ".csv")]
    sum_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    fn = sum_files[0]
    data = pd.read_csv(fn)
    for fn in sum_files[1:]:
        tmp = pd.read_csv(fn)
        data = data.append(tmp)

    #### GET TOP_BOTTOM_RESID YEARS  ####
    var_thresh = data.quantile(q=thresh)[var]
    return data[data[var] >= var_thresh]

################################################################################

def corr2_coeff(AVGPAT, slp_stack):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    x = AVGPAT - AVGPAT.mean()
    y = slp_stack - slp_stack.mean('grid')

    # Sum of squares across rows
    ssx = (x**2).sum()
    ssy = (y**2).sum('grid')

    # Finally get corr coeff
    return np.dot(x, y.T) / np.sqrt(np.dot(ssx,ssy))

