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

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", "Southeast", "Southwest", "UpperMidwest", "West"]  
PICTL_TYPES = ["PICTL", "SSTPICTL"]
LAGS = [1]

#################### DIRECTORIES ###########################
os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L1")
DYNADJ_DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/dyn_adj_" + str(LAG) + "/" 
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/data/shape_files/"
AMJJAS_DIR =  "/glade/scratch/horowitz/extreme_heat_CCA/AMJJAS_anom/"
MJJA_DIR = "/glade/scratch/horowitz/extreme_heat_CCA/MJJA_anom/"
DATE_DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/dates/"

ODIR = "/glade/work/horowitz/extreme_heat_CCA/L1/"
import L1_functions as L1

for LOC in LOCS:
    for PICTL_TYPE in PICTL_TYPES:
        for LAG in LAGS:
            ############################################################
            # Load in heatwave dates
            ############################################################
            heatwave_dates = np.load(ODIR + 'dates/heatwave_dates_' + PICTL_TYPE + '_' + LOC + '_' + str(LAG) + '.npy', allow_pickle=True)
            max_year = 1799
            heatwave_dates = heatwave_dates[0:max_year]

            # All days in heatwaves - offset by LAG days
            heat_week_dates = list()
            for date in heatwave_dates:
                dates_begin = xr.cftime_range(end=date, periods=11 + LAG, freq="D", calendar="noleap").to_list()[:-1]
                dates_end = xr.cftime_range(start=date, periods= 11 - LAG, freq="D", calendar="noleap").to_list()
                heat_week_dates = heat_week_dates + dates_begin + dates_end
            print("heatweek dates calculated")
            ############################################################
            # Load SLP and Z500 and filter to dates
            ############################################################

            # Open full slp file
            slp_fn = AMJJAS_DIR + "PSL_" + PICTL_TYPE + "_" + LOC + "_AMJJAS_anom.nc"
            slp = xr.open_dataset(slp_fn)
            print("slp file opened")
            slp = slp.assign_coords(lon=(((slp.lon + 180) % 360) - 180))
            slp['lat']  =  np.round(slp.lat, 2)


            all_vars = L1.time_index_fun(slp.anom, heat_week_dates).to_dataset(name = 'slp')

            # Open full Z500 file
            z500_fn = AMJJAS_DIR + "Z500_" + PICTL_TYPE + "_" + LOC + "_AMJJAS_anom.nc"
            z500 = xr.open_dataset(z500_fn)
            print("z500 file opened")
            z500 = z500.assign_coords(lon=(((z500.lon + 180) % 360) - 180))
            z500['lat']  =  np.round(z500.lat, 2)


            all_vars = xr.merge([all_vars, L1.time_index_fun(z500.anom, heat_week_dates).to_dataset(name = 'z500')])


            all_vars.to_netcdf(ODIR + 'gridded_slp_z500/gridded_slp_z500_' + PICTL_TYPE + '_' + LOC + '_' + str(LAG) + '.nc')
            print("Done with " + LOC + " " + PICTL_TYPE)
            print("----------------------------------------------------------")


