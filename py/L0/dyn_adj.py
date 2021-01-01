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

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('LOC', type=str, help='region')
parser.add_argument('PICTL', type=str, help='PICTL type')
parser.add_argument('N', type=int, help='# of analogs')
parser.add_argument('N_S', type=int, help='sample size')
parser.add_argument('N_R', type=int, help='# of iterations')
parser.add_argument('N_Y', type=int, help='# of analog years')
args = parser.parse_args()

LOC = args.LOC
PICTL = args.PICTL
N = args.N
N_S = args.N_S
N_R = args.N_R
N_Y = args.N_Y

##################### DIRECTORIES ###########################
os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L0/")
DIR = "/glade/scratch/horowitz/extreme_heat_CCA/"
PICTL_DIR = DIR + 'MJJAS_anom/'
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/shape_files/"
RMSE_DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/rmse/"
DYN_ADJ_DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/dyn_adj/"

import L0_functions as L0

##################### INPUTS ###########################
# PICTL Files
tas_pictl_fn = PICTL_DIR + "TREFHT_" + PICTL + "_" + LOC + "_MJJAS_anom.nc"
slp_pictl_fn = PICTL_DIR + "PSL_" + PICTL + "_" + LOC + "_MJJAS_anom.nc"

# Land mask file
land_file = "/glade/work/horowitz/forced_response/soil_map_global.nc"

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

##################### LOAD SLP DATA ##################
slp_pictl = xr.open_dataset(slp_pictl_fn)
slp_pictl['lat']  =  np.round(slp_pictl.lat, 2)

print("slp files loaded")

##################### LOAD TREFHT DATA ##################
tas_pictl = xr.open_dataset(tas_pictl_fn)
tas_pictl['lat']  =  np.round(tas_pictl.lat, 2)

print("tas files loaded")

###################### FILTER DATA ###############################
# Filter to JJA
slp_JJA = slp_pictl.anom.sel(time = slp_pictl.time.dt.month.isin([6,7,8])).values
tas_JJA = tas_pictl.anom.sel(time = tas_pictl.time.dt.month.isin([6,7,8])).values

ndays = slp_JJA.shape[0]
nlat = slp_JJA.shape[1]
nlon = slp_JJA.shape[2]
nyear = int(ndays/92)

slp_JJA = slp_JJA.reshape((ndays, nlat*nlon))
tas_JJA = tas_JJA.reshape((ndays, nlat*nlon))

print("JJA reshaped")

# Save MJJAS data
slp_MJJAS = slp_pictl.anom.values
tas_MJJAS = tas_pictl.anom.values


ndays = slp_MJJAS.shape[0]

slp_MJJAS = slp_MJJAS.reshape((ndays, nlat*nlon))
tas_MJJAS = tas_MJJAS.reshape((ndays, nlat*nlon))

print("MJJAS reshaped")


###################### LOAD DISTANCE MATRICES ###############################

similarity = {}
for i in range(92):
    similarity[i] = np.load(DIR + 'distance_matrices/distance_matrix_' + PICTL + '_' +  LOC + '_' + str(i) + '.npy')

print("similarity matrices loaded")

###################### LOAD DISTANCE INDEX MATRIX ###############################

similarity_idx = {}
for i in range(92):
    similarity_idx[i] = np.load(DIR + 'distance_matrices/distance_index_' + str(i) + '.npy')

print("similarity indices loaded")

###################### SELECT ANALOGS ###############################
Tca_OLS = np.empty((92*1799,  nlat*nlon))

for i in range(92*1799):
    day_of_JJA = i%92
    year = math.floor(i/92)

    z = similarity[day_of_JJA][year,:] # distance measurements for days in all other years
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
        idx = similarity_idx[day_of_JJA][year,min_yr_indices[closest]]

        # Select analogs
        analogs = slp_MJJAS[idx,:]
        X = analogs.transpose()
        y = slp_JJA[i,:]
        X_t = tas_MJJAS[idx,:].transpose()

        ###################### OLS ###############################
        # Construct temp analog
        B = np.linalg.lstsq(X, y, rcond = None)[0]
        tmp_OLS[n_r, :] = np.matmul(X_t, B)



    Tca_OLS[i, :] = tmp_OLS.mean(axis=0)

###################### OUTPUT RESULTS ###############################
tmp = tas_pictl.sel(time = tas_pictl.time.dt.month.isin([6,7,8]))
time = tmp.time[0:92*1799]

lat, lon = tas_pictl.lat, tas_pictl.lon
# Convert to dataset and write to netcdf
Tca = xr.DataArray(Tca_OLS.reshape(92*1799, nlat, nlon), coords=[time , lat, lon], dims=['time', 'lat', 'lon'], name = 'dynamic')
Tca = Tca.salem.subset(shape=shapefile).salem.roi(shape=shapefile) * land.mask
fn_adder = '_'.join([PICTL, LOC, str(N), str(N_S), str(N_R), str(N_Y)])

ofn = DYN_ADJ_DIR + 'dyn_adj_' + fn_adder + '.nc'
Tca.to_netcdf(path = ofn)

# Save RMSE
rmse = np.sqrt(((Tca_OLS - tas_JJA[92*1799,:])**2).mean(axis=0))

rmse_xr = xr.DataArray(rmse.reshape(nlat, nlon), coords=[ lat, lon], dims=['lat', 'lon'], name = 'rmse')

rmse_xr = rmse_xr.salem.subset(shape=shapefile).salem.roi(shape=shapefile) * land.mask
fn_adder = '_'.join([PICTL, LOC, str(N), str(N_S), str(N_R), str(N_Y), str(random.randint(0, 1000000))])
rmse_xr.to_netcdf(RMSE_DIR + 'rmse_' + fn_adder + '.nc')



print("Fin")
