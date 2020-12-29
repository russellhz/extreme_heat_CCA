#!/usr/bin/env python3

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

VAR = 'slp'

# Directories
os.chdir("/glade/u/home/horowitz/forced_response/py/L0")
VAR_DIR = "/glade/scratch/horowitz/forced_response/L0/"
SHAPE_DIR = "/glade/work/horowitz/forced_response/shape_files/"
ODIR = "/glade/work/horowitz/forced_response/US_climate_regions/dyn_adj_sensitivity/"

import L0_functions as L0

summer_dict = {}
all_days = list(range(1799*92))
days_by_year = np.array_split(np.array(all_days), 1799) # split into array for each year

for i in range(92):
    days_to_compare = list(range(max(i-7, 0), min(i+8, 92)))
    other_years = [x[days_to_compare] for x in days_by_year]
    summer_dict[i] = [item for sublist in other_years for item in sublist]

LOCS = ['OhioValley', 'Northeast', 'NorthernRockiesandPlains', 'Northwest', 'South', 'Southeast', 'Southwest', 'UpperMidwest', 'West']

for LOC in LOCS:
    print("Starting " + LOC)
    DIR = "/glade/scratch/horowitz/forced_response/" + LOC + "/"
    PICTL_DIR = DIR + 'L0/'
    # PICTL Files
    tas_pictl_fn = PICTL_DIR + "TREFHT_PICTL_" + LOC + "_MJJAS_anom.nc"
    slp_pictl_fn = PICTL_DIR + "PSL_PICTL_"  + LOC + "_MJJAS_anom.nc"

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
    nyear = ndays/92

    tas = tas.reshape((ndays, nlat*nlon))
    var = var.reshape((ndays, nlat*nlon))
    
    for i in range(92):
        scores = euclidean_distances(var[range(i, 1799*92, 92)], var[summer_dict[i]])
        comp_length = int(scores.shape[1]/1799)
        scores_fixed = np.empty((scores.shape[0],scores.shape[1]-comp_length))
        for j in range(scores.shape[0]):
            remove_idx = range(j*comp_length, j*comp_length+comp_length)
            keep_idk = list(set(range(0,scores.shape[1])) - set(remove_idx))
            keep_idk.sort()
            scores_fixed[j,:] = scores[j,keep_idk]
        np.save('/glade/scratch/horowitz/forced_response/dyn_adj_sensitivity/' + 'distance_matrix_' + LOC + '_' + str(i), scores_fixed)
        if i%10==0:
            print("Done with " + str(i))
    print("--------------------------------------------")
