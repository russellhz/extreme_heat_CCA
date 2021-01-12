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
import warnings
warnings.filterwarnings("ignore")

sys.path.append("/glade/u/home/horowitz/extreme_heat_CCA/py/L0")
import L0_functions as L0

sys.path.append("/glade/u/home/horowitz/extreme_heat_CCA/py/L1")
import L1_functions as L1

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('PICTL_TYPE', type=str, help='PICTL type')
args = parser.parse_args()

PICTL_TYPE = args.PICTL_TYPE

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", 
        "OhioValley", "South", "Southeast", 
        "Southwest", "UpperMidwest", "West"]   

# Directories
DIR = "/glade/scratch/horowitz/extreme_heat_CCA/AMJJAS_anom/"
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/data/shape_files/"
ODIR = "/glade/work/horowitz/extreme_heat_CCA/L1/dayofyear_sd/"


##############################################################################
################             CALCULATIONS          ################
##############################################################################
slp_df = pd.DataFrame()
tas_df = pd.DataFrame()

for LOC in LOCS:
    ##################### INPUTS ###########################    
    # PICTL Files
    tas_pictl_fn = DIR + "TREFHT_" + PICTL_TYPE + "_" + LOC + "_AMJJAS_anom.nc"
    slp_pictl_fn = DIR + "PSL_" + PICTL_TYPE + "_" + LOC + "_AMJJAS_anom.nc"

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
    print('shapefile and land mask loaded')
    
    ##################### LOAD AND ADD SLP DATA ##################
    slp = xr.open_dataset(slp_pictl_fn)
    slp['lat']  =  np.round(slp.lat, 2)
    
    slp_JJA = slp.anom.sel(time = slp.time.dt.month.isin(range(6,9)))

    slp_JJA_mean = slp_JJA.mean(dim = ['lon', 'lat'])
    print("slp regional mean calculated")
    
    slp_dayofyear_mean = slp_JJA_mean.groupby(slp_JJA_mean.time.dt.dayofyear).mean()
    slp_dayofyear_sd = np.sqrt(slp_JJA_mean.groupby(slp_JJA_mean.time.dt.dayofyear).var())
    
    slp_df_loc = pd.DataFrame(data = {'dayofyear':slp_dayofyear_mean.dayofyear, 'mean': slp_dayofyear_mean.values, 
                     'sd': slp_dayofyear_sd.values, 'region':LOC})

    ##################### LOAD TREFHT DATA ##################
    tas = xr.open_dataset(tas_pictl_fn)
    tas['lat']  =  np.round(tas.lat, 2)
    
    tas_JJA = tas.anom.sel(time = tas.time.dt.month.isin(range(6,9)))

    # Take mean just over region, only over land
    tas_JJA = tas_JJA.salem.roi(shape=shapefile) * land.mask
    
    tas_JJA_mean = tas_JJA.mean(dim = ['lon', 'lat'])
    print("tas regional mean calculated")
    
    tas_dayofyear_mean = tas_JJA_mean.groupby(tas_JJA_mean.time.dt.dayofyear).mean()
    tas_dayofyear_sd = np.sqrt(tas_JJA_mean.groupby(tas_JJA_mean.time.dt.dayofyear).var())
    
    tas_df_loc = pd.DataFrame(data = {'dayofyear':tas_dayofyear_mean.dayofyear, 'mean': tas_dayofyear_mean.values, 
                 'sd': tas_dayofyear_sd.values, 'region':LOC})
    
    ##################### ADD TO GLOBAL DATAFRAMES ##################
    slp_df = slp_df.append(slp_df_loc)
    tas_df = tas_df.append(tas_df_loc)
    print("Done adding " + LOC)
    print("--------------------------------------------------")

slp_df.to_csv(ODIR + "slp_dayofyear_mean_sd_"+PICTL_TYPE+".csv")
tas_df.to_csv(ODIR + "tas_dayofyear_mean_sd_"+PICTL_TYPE+".csv")