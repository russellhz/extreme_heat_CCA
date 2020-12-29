#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:17:56 2020

@author: rhorowitz
"""
import xarray as xr
import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import random
import argparse
import itertools
from itertools import chain
import geopandas
import salem
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore")

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('LOC', type=str, help='region')
parser.add_argument('N', type=int, help='# of analogs')
parser.add_argument('N_S', type=int, help='sample size')
parser.add_argument('N_R', type=int, help='# of iterations')
parser.add_argument('N_Y', type=int, help='# of analog years')
args = parser.parse_args()

LOC = args.LOC
N = args.N
N_S = args.N_S
N_R = args.N_R
N_Y = args.N_Y

VAR = 'slp'
month_abbrev = 'JJA'
PICTL = 'PICTL'

##################### DIRECTORIES ###########################
os.chdir("/glade/u/home/horowitz/forced_response/py/L0")
DIR = "/glade/scratch/horowitz/forced_response/" + LOC + "/"
PICTL_DIR = DIR + 'L0/'
VAR_DIR = "/glade/scratch/horowitz/forced_response/L0/"
SHAPE_DIR = "/glade/work/horowitz/forced_response/shape_files/"
ODIR = "/glade/work/horowitz/forced_response/US_climate_regions/dyn_adj_sensitivity/"
SCRATCH_DIR = '/glade/scratch/horowitz/forced_response/dyn_adj_sensitivity/'

import L0_functions as L0

##################### INPUTS ###########################
fns = L0.file_getter('', VAR_DIR, VAR + '_PICTL_' + month_abbrev)
# PICTL Files
tas_pictl_fn = PICTL_DIR + "TREFHT_" + PICTL + "_" + LOC + "_MJJAS_anom.nc"
slp_pictl_fn = PICTL_DIR + "PSL_" + PICTL + "_" + LOC + "_MJJAS_anom.nc"

# Land mask file
land_file = "/glade/work/horowitz/forced_response/soil_map_global.nc"

def dynadj_region(data, shapefile):
    tmp = data.salem.subset(shape = shapefile)
    mean_lon = tmp.lon.values.mean()
    mean_lat = tmp.lat.values.mean()
    return data.salem.subset(corners = ((mean_lon - 54, mean_lat + 15),(mean_lon + 18, mean_lat - 15)))

##################### LOAD SHAPEFILE DATA ##################
shapefile = geopandas.read_file(SHAPE_DIR + LOC + '.json')
shapefile.crs =  {'init': 'epsg:3857'}
shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
shapefile.crs = 'epsg:4326'
shapefile_shift = shapefile.translate(xoff=360)
shapefile['geometry'] = shapefile_shift

##################### LAND MASK ##################
land = xr.open_dataset(land_file).squeeze('time')
t = land.mrfso.values
t[~np.isnan(t)] = 1
land = land.assign(mask = (('lat', 'lon'), t))
land['lat'] =  np.round(land.lat, 2)

##################### LOAD VAR DATA ##################
var_pictl = xr.open_dataset(slp_pictl_fn)
var_pictl['lat']  =  np.round(var_pictl.lat, 2)

print("slp files loaded")

##################### LOAD TREFHT DATA ##################
tas_pictl = xr.open_dataset(tas_pictl_fn)
tas_pictl['lat']  =  np.round(tas_pictl.lat, 2)

print("tas files loaded")

###################### FILTER DATA ###############################
# Filter slp to July, getting only July 15s
tas = tas_pictl.anom.sel(time = tas_pictl.time.dt.month.isin([6,7,8])).values

var = var_pictl.anom.sel(time = var_pictl.time.dt.month.isin([6,7,8])).values

ndays = tas.shape[0]
nlat = tas.shape[1]
nlon = tas.shape[2]
nyear = int(ndays/92)

tas = tas.reshape((ndays, nlat*nlon))
var = var.reshape((ndays, nlat*nlon))

###################### ADJUST NUMBER OF YEARS ###############################
days_to_choose = N_Y * 31


###################### LOAD DISTANCE MATRICES ###############################

similarity = {}
for i in range(92):
    similarity[i] = np.load(SCRATCH_DIR + 'distance_matrix_' + LOC + '_' + str(i) + '.npy')

###################### LOAD DISTANCE INDEX MATRIX ###############################

similarity_idx = {}
for i in range(92):
    similarity_idx[i] = np.load(SCRATCH_DIR + 'distance_index_' + str(i) + '.npy')



###################### SELECT ANALOGS ###############################
Tca_OLS = np.empty((tas.shape[0],  nlat*nlon))

for i in range(tas.shape[0]):
    day_of_year = i%92
    year = math.floor(i/92)

    z = similarity[day_of_year][year,:] # distance measurements for days in all other years
    z_by_year = np.array_split(z, 1798) # split into array for each year

    # Reduce to number of desired years
    z_by_year = z_by_year[0:N_Y]

    year_len = len(z_by_year[0])
    # Get index of min in each year
    min_yr_indices = [np.argmin(yr) for yr in z_by_year]
    min_yr_indices = np.array([min_yr_indices[i] + year_len*i for i in range(len(min_yr_indices))])

    tmp_OLS = np.empty((N_R,  nlat*nlon))
    for n_r in range(N_R):
        # Pick 150 closest years and randomly sample
        closest = np.argsort(z[min_yr_indices])
        closest = np.random.choice(closest[:(N)], N_S, replace=False)  
        # Need to convert index back to full dates
        idx = similarity_idx[day_of_year][year,min_yr_indices[closest]]

        # Select analogs
        analogs = var[idx,:]
        X = analogs.transpose()
        y = var[i,:]
        X_t = tas[idx,:].transpose()

        ###################### OLS ###############################
        # Construct temp analog
        B = np.linalg.lstsq(X, y, rcond = None)[0]
        constructed_slp =np.matmul(X, B)
        tmp_OLS[n_r, :] = np.matmul(X_t, B)



    Tca_OLS[i, :,] = tmp_OLS.mean(axis=0)

###################### OUTPUT RESULTS ###############################
rmse = np.sqrt(((Tca_OLS - tas)**2).mean(axis=0))

lat, lon = tas_pictl.lat, tas_pictl.lon

rmse_xr = xr.DataArray(rmse.reshape(nlat, nlon), coords=[ lat, lon], dims=['lat', 'lon'], name = 'rmse')

rmse_xr = rmse_xr.salem.subset(shape=shapefile).salem.roi(shape=shapefile)
fn_adder = '_'.join([LOC, str(N), str(N_S), str(N_R), str(N_Y)])
rmse_xr.to_netcdf(ODIR + 'rmse_' + fn_adder + '.nc')

print("Fin")
