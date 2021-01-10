#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:19:41 2020

@author: rhorowitz
"""
import xarray as xr
import os
import glob
import re
import numpy as np
import pandas as pd
import geopandas
import salem
import argparse

# Directories
os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L0")
DIR = "/glade/scratch/horowitz/extreme_heat_CCA/JJA_anom/"
import L0_functions as L0


##################### INPUTS ###########################
SLP_DIR = "/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/PSL/"
SM_DIR = "/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/lnd/proc/tseries/daily/SOILWATER_10CM/"
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/data/shape_files/"

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('code', type=str, help='ensemble code number')
args = parser.parse_args()

code = args.code
shapefn = SHAPE_DIR  + 'CONUS.json'


##################### LOAD SHAPEFILE DATA ##################
shapefile = geopandas.read_file(shapefn)
shapefile.crs =  {'init': 'epsg:3857'}
shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
shapefile.crs = 'epsg:4326'
shapefile_shift = shapefile.translate(xoff=360)
shapefile['geometry'] = shapefile_shift

##################### SLP ##################
# 20th century
slp_fns = [f for f in glob.glob(SLP_DIR + "b.e11.B20TRC5CNBDRD.f09_g16.[0-9]" + "*.nc")]
# RCP 8.5
slp_fns.extend([f for f in glob.glob(SLP_DIR + "b.e11.BRCP85C5CNBDRD.f09_g16.[0-9]" + "*.nc")])

# Open files for specific code
files = [f for f in slp_fns if '.' + code + '.' in f]
files.sort(key=lambda f: int(re.sub('\D', '', f)))

slp_all = xr.open_dataset(files[0])
slp_all = slp_all.PSL.salem.subset(corners=((180,20), (310,60)))
print('loaded and filtered ' + files[0])
for fn in files[1:]:
    tmp = xr.open_dataset(fn)
    tmp = tmp.PSL.salem.subset(corners=((180,20), (310,60)))
    slp_all = xr.concat([slp_all, tmp], dim = 'time')
    print('added in ' + fn)
slp_all = slp_all.assign_coords(lon=(((slp_all.lon + 180) % 360) - 180)).sortby("lon")

# Remove seasonality from tas  
slp_all = slp_all.to_dataset().sel(time = slice("1920-01-01", "2100-12-31"))
slp_all = slp_all.assign(anom = (('time', 'lat', 'lon'), L0.seasonality_removal_vals(slp_all['PSL'].values, k=3)))
print("slp seasonality removed") 

slp_all = slp_all.anom.sel(time = slp_all.time.dt.month.isin(range(6,9)))

slp_all.to_netcdf(DIR + 'CESM-LE_PSL_anom_' + code + '_19200101-21001231.nc' )

##################### SOIL MOISTURE ##################
# 20th century
sm_fns = [f for f in glob.glob(SM_DIR + "b.e11.B20TRC5CNBDRD.f09_g16.[0-9]" + "*.nc")]
# RCP 8.5
sm_fns.extend([f for f in glob.glob(SM_DIR + "b.e11.BRCP85C5CNBDRD.f09_g16.[0-9]" + "*.nc")])


# Open files for specific code
files = [f for f in sm_fns if '.' + code + '.' in f]
files.sort(key=lambda f: int(re.sub('\D', '', f)))

sm_all = xr.open_dataset(files[0])
# Subset to CONUS
sm_all = sm_all.SOILWATER_10CM.salem.subset(shape=shapefile)
print('loaded and filtered ' + files[0])
for fn in files[1:]:
    tmp = xr.open_dataset(fn)
    tmp = tmp.SOILWATER_10CM.salem.subset(shape=shapefile)
    sm_all = xr.concat([sm_all, tmp], dim = 'time')
    print('added in ' + fn)

# Remove seasonality from tas  
sm_all = sm_all.to_dataset().sel(time = slice("1920-01-01", "2100-12-31"))
sm_all = sm_all.assign(anom = (('time', 'lat', 'lon'), L0.seasonality_removal_vals(sm_all['SOILWATER_10CM'].values, k=3)))
print("soil moisture seasonality removed") 

sm_all = sm_all.anom.sel(time = sm_all.time.dt.month.isin(range(6,9)))

sm_all.to_netcdf(DIR + 'CESM-LE_SOILWATER_10CM_anom_' + code + '_19200101-21001231.nc' )

