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
import random

import warnings
warnings.filterwarnings("ignore")

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", 
        "South", "Southeast", "Southwest", "UpperMidwest", "West"]  
PICTL_TYPES = ['PICTL', 'SSTPICTL']

DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/JJA_variability/"
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/data/shape_files/"
MJJA_DIR = "/glade/scratch/horowitz/extreme_heat_CCA/MJJA_anom/"
AMJJAS_DIR = "/glade/scratch/horowitz/extreme_heat_CCA/AMJJAS_anom/"

##############################################################################
################             Soil Moisture Variance           ################
##############################################################################
for PICTL_TYPE in PICTL_TYPES:
    data = xr.open_dataset(MJJA_DIR + 'SOILWATER_10CM_' +PICTL_TYPE+'_CONUS_MJJA_anom.nc')
    data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby("lon")
    for LOC in LOCS:
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

        ##############################################################
        ####################### SOILWATER DATA ##########################
        ##############################################################

        # filter to spatial range and apply land mask
        tmp = data.anom.salem.subset(shape=shapefile)
        print('spatial filter applied')

        tmp = tmp.sel(time = tmp.time.dt.month.isin(range(6,9)))
        print('filtered to JJA')

        tmp = tmp.salem.roi(shape = shapefile)
        tmp['lat'] =  np.round(tmp.lat, 2)
        variance = tmp.var('time')

        y2y_variance = tmp.groupby(tmp.time.dt.year).mean().var('year')

        d2d_variance = tmp.groupby(tmp.time.dt.year).var().mean('year')

        variance.to_netcdf(DIR + 'sm_variance_' + LOC + '_' + PICTL_TYPE + '.nc')
        y2y_variance.to_netcdf(DIR + 'sm_y2y_variance_' + LOC + '_' + PICTL_TYPE + '.nc')
        d2d_variance.to_netcdf(DIR + 'sm_d2d_variance_' + LOC + '_' + PICTL_TYPE + '.nc')

        print('SM: Done with ' + LOC + ' in ' + PICTL_TYPE)
        print('----------------------------------------')

##############################################################################
################             Temperature Variance             ################
##############################################################################

for LOC in LOCS:
    for PICTL_TYPE in PICTL_TYPES:
        data = xr.open_dataset(AMJJAS_DIR + 'TREFHT_' +PICTL_TYPE+'_'+ LOC + '_AMJJAS_anom.nc')
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby("lon")

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

        ##############################################################
        ####################### TREFHT DATA ##########################
        ##############################################################

        # filter to spatial range and apply land mask
        tmp = data.anom.salem.subset(shape=shapefile)
        print('spatial filter applied')

        tmp = tmp.sel(time = tmp.time.dt.month.isin(range(6,9)))
        print('filtered to JJA')

        tmp = tmp.salem.roi(shape = shapefile)
        tmp['lat'] =  np.round(tmp.lat, 2)

        variance = tmp.var('time')

        y2y_variance = tmp.groupby(tmp.time.dt.year).mean().var('year')

        d2d_variance = tmp.groupby(tmp.time.dt.year).var().mean('year')

        variance.to_netcdf(DIR + 'tas_variance_' + LOC + '_' + PICTL_TYPE + '.nc')
        y2y_variance.to_netcdf(DIR + 'tas_y2y_variance_' + LOC + '_' + PICTL_TYPE + '.nc')
        d2d_variance.to_netcdf(DIR + 'tas_d2d_variance_' + LOC + '_' + PICTL_TYPE + '.nc')

        print('TAS: Done with ' + LOC + ' in ' + PICTL_TYPE)
        print('----------------------------------------')