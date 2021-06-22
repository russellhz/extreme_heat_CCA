#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:17:56 2020

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

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('LOC', type=str, help='region')
parser.add_argument('PICTL_TYPE', type=str, help='PICTL type')
parser.add_argument('LAG', type=int, help='# of days slp lagged for dyn adj')
args = parser.parse_args()

LOC = args.LOC
PICTL_TYPE = args.PICTL_TYPE
LAG = args.LAG

COMPOSITE_LENGTH = 38 # Allows to start at May 1st at earliest

#################### DIRECTORIES ###########################
os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L1")
DYNADJ_DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/dyn_adj_lag" + str(LAG) + "/"
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/data/shape_files/"
AMJJAS_DIR =  "/glade/scratch/horowitz/extreme_heat_CCA/AMJJAS_anom/"
MJJA_DIR = "/glade/scratch/horowitz/extreme_heat_CCA/MJJA_anom/"
ODIR = "/glade/work/horowitz/extreme_heat_CCA/L1/"
import L1_functions as L1

##############################################################
######################## DYNADJ FILE ########################
##############################################################
dynadj_file = DYNADJ_DIR + "dyn_adj_" + PICTL_TYPE + "_" + LOC + "_150_100_20_1798_" + str(LAG) + ".nc"

###########################################################################################################
###################################        SHAPEFILE        ###############################################
###########################################################################################################
shapefn = SHAPE_DIR + LOC + '.json'
shapefile = geopandas.read_file(shapefn)
shapefile.crs =  {'init': 'epsg:3857'}
shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
shapefile.crs = 'epsg:4326'

shapefile_shift = copy.copy(shapefile)
shapefile_shift['geometry'] = shapefile_shift.translate(xoff=360)

###########################################################################################################
###################################        LAND MASK        ###############################################
###########################################################################################################
# Land mask file
land_file = "/glade/u/home/horowitz/extreme_heat_CCA/data/soil_map_global.nc"
land = xr.open_dataset(land_file).squeeze('time')
t = land.mrfso.values
t[~np.isnan(t)] = 1
land = land.assign(mask = (('lat', 'lon'), t))
land['lat'] =  np.round(land.lat, 2)
land = land.assign_coords(lon=(((land.lon + 180) % 360) - 180)).sortby("lon")

##############################################################
######################### TAS DATA ###########################
##############################################################
# Open full tas file
tas_fn = AMJJAS_DIR + "TREFHT_" + PICTL_TYPE + "_" + LOC + "_AMJJAS_anom.nc"
tas = xr.open_dataset(tas_fn)
print("tas file opened")
tas = tas.assign_coords(lon=(((tas.lon + 180) % 360) - 180))
tas['lat']  =  np.round(tas.lat, 2)

# Filter to region -- speed improvement by subsetting first
tas_filter = tas.anom.salem.subset(shape = shapefile)
tas_filter = tas_filter.salem.roi(shape = shapefile)
tas_filter = tas_filter * land.mask
print("tas filtered to region")

# Weighted regional mean & limit to JJA 
weights = np.cos(np.deg2rad(tas_filter.lat))
weights.name = "weights"
tas_filter_mean = tas_filter.weighted(weights).mean(("lon", "lat")) 
print("tas spatial mean calculated")

# Create a list of all summer dates
JJA_dates =  list()
for yr in np.unique(tas_filter_mean.time.dt.year):
    yr_dates = xr.cftime_range(start=cftime.DatetimeNoLeap(yr, 6,1), periods=92, freq="D", calendar="noleap").to_list()
    JJA_dates = JJA_dates + yr_dates

tas_filter_mean = L1.time_index_fun(tas_filter_mean, JJA_dates)
print("tas filtered to JJA")

# convert to df, then rolling mean grouped by year (restricts to 6/04-8/28)
df = pd.DataFrame({'date':tas_filter_mean.time, 'year':tas_filter_mean['time.year'].values, 'tas':tas_filter_mean.values})
roll_tas = df.groupby('year')['tas'].rolling(7, center = True).mean()
roll_tas = roll_tas.reset_index().drop(columns = 'level_1').rename(columns={'tas': 'tas_roll'})
df = df.merge(roll_tas, how='left', on='year', left_index=True, right_index=True)

# Get top score in each year, then trim to top n years
maxheat_idx = df.groupby(['year'])['tas_roll'].transform(max) == df['tas_roll']
heatwave_dates = df[maxheat_idx].date.tolist()
print("max heat days calculated")

np.save(ODIR + 'dates/heatwave_dates_' + PICTL_TYPE + '_' + LOC + '_' + str(LAG) , heatwave_dates)

# All days in heatwaves
heat_week_dates = list()
for date in heatwave_dates:
    dates_begin = xr.cftime_range(end=date, periods=4, freq="D", calendar="noleap").to_list()[:-1]
    dates_end = xr.cftime_range(start=date, periods= 4, freq="D", calendar="noleap").to_list()
    heat_week_dates = heat_week_dates + dates_begin + dates_end

##############################################################
####################### DYNADJ DATA ##########################
##############################################################
# Get dyn_adj data
dyn_adj = xr.open_dataset(dynadj_file)
dyn_adj = dyn_adj.assign_coords(lon=(((dyn_adj.lon + 180) % 360) - 180)).sortby("lon")
dyn_adj = dyn_adj.rename({'__xarray_dataarray_variable__' : 'dynamic'})
print("dynamic file opened")

# Desired spatial range
dyn_adj = dyn_adj.dynamic.salem.roi(shape = shapefile)

# Desired time range
# Using a dictionary to speed up indexing, which can be very slow
max_year = dyn_adj.time.dt.year.max().values
heat_week_dates = list(filter(lambda x: (x.year <= max_year), heat_week_dates))  
dyn_adj_filter = L1.time_index_fun(dyn_adj, heat_week_dates)
print("dynamic filtered to heatweek dates")

dyn_adj_filter['lat']  =  np.round(dyn_adj_filter.lat, 2)
dyn_adj_filter = dyn_adj_filter * land.mask


# Mean dynamic for each heatwave
dyn_adj_heatweek = dyn_adj_filter.groupby(dyn_adj_filter.time.dt.year).mean()
dyn_adj_heatweek = dyn_adj_heatweek.mean(dim=('lon', 'lat'))

# Get years of top 15% dynamics
df = pd.DataFrame({'year':dyn_adj_heatweek.year, 'dynamic':dyn_adj_heatweek})
min_intdyn = df.quantile(q=0.85).dynamic
df = df[df.dynamic >= min_intdyn]
top_dyn_years = df.year

top_dyn_dates = list(filter(lambda x: x.year in top_dyn_years.values, heatwave_dates))

# List of all dates from COMPOSITE_LENGTH days before to 3 days after peak
composite_dates = list()
for date in top_dyn_dates:
    dates_begin = xr.cftime_range(end=date, periods=COMPOSITE_LENGTH-3, freq="D", calendar="noleap").to_list()[:-1]
    dates_end = xr.cftime_range(start=date, periods= 4, freq="D", calendar="noleap").to_list()
    composite_dates = composite_dates + dates_begin + dates_end

# Filter all_dates to top_dyn_dates
np.save(ODIR + 'dates/top_dyn_composite_dates_' + PICTL_TYPE + '_' + LOC + '_' + str(LAG), composite_dates)

##############################################################
####################### COMBINE TEMP DATA ##########################
##############################################################
all_vars = L1.time_index_fun(tas_filter, composite_dates).to_dataset(name = 'tas')

# Add in dynamic
all_vars = xr.merge([all_vars, L1.time_index_fun(dyn_adj, composite_dates).to_dataset(name = 'dynamic')])

# Calculate residual
all_vars = all_vars.assign(residual = all_vars.tas - all_vars.dynamic)

##############################################################
####################### ADD COVARS ##########################
##############################################################
VARS = ['lhflx', 'shflx', 'soilwater_10cm', 'ts']

for VAR in VARS:
    fn = MJJA_DIR + VAR.upper() + '_' + PICTL_TYPE + '_CONUS_MJJA_anom.nc'
    data = xr.open_dataset(fn)
    print('opened ' + fn)

    # Float 64 conversion needed for soilwater_10cm, which is in float32
    data['lat']  =  np.round(np.float64(data.lat), 2)
    data['lon'] = np.float64(data.lon)
    data = data.rename({'anom' : VAR})

    # Need to remove year 402 from dates if soilwater is the var 
    # Soilwater data starts in year 403 for some reason
    if VAR == 'soilwater_10cm':
        composite_dates_soilwater = [x for x in composite_dates if x.year != 402]
        data = L1.time_index_fun(data[VAR], composite_dates_soilwater)
    else:
        data = L1.time_index_fun(data[VAR], composite_dates)
        
    # Desired spatial range
    data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby("lon")
    data = data.salem.subset(corners = ( (all_vars.lon.min(), all_vars.lat.min()), (all_vars.lon.max(), all_vars.lat.max()) ) )
    data = data.salem.roi(shape = shapefile)

    all_vars = xr.merge([all_vars, data])
    print("added in " + VAR)

all_vars.to_netcdf(ODIR + 'gridded_composite/gridded_composite_' + PICTL_TYPE + '_' + LOC + '_' + str(LAG) + '.nc')