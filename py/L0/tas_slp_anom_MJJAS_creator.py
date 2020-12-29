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
DIR = "/glade/work/horowitz/scratch/extreme_heat_CCA/"

import L0_functions as L0


##################### INPUTS ###########################
PICTL_DIR = "/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/"
SHAPE_DIR = "/glade/work/horowitz/forced_response/shape_files/"

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('LOC', type=str, help='Region name')
args = parser.parse_args()

LOC = args.LOC

shapefn = SHAPE_DIR + LOC + '.json'

# PICTL_TYPE file dictionary
PICTL_dict = {'PICTL': 'b.e11.B1850C5CN.f09_g16',
                'SSTPICTL': 'f.e11.F1850C5CN.f09_f09'}


##################### LOAD SHAPEFILE DATA ##################
shapefile = geopandas.read_file(shapefn)
shapefile.crs =  {'init': 'epsg:3857'}
shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
shapefile.crs = 'epsg:4326'
shapefile_shift = shapefile.translate(xoff=360)
shapefile['geometry'] = shapefile_shift

##################### LOAD PICTL DATA ##################
for var in ['TREFHT', 'PSL']:
    for PICTL_TYPE in ['SSTPICTL', 'PICTL']:
        # PiCTL files
        files = [f for f in glob.glob(PICTL_DIR + var + '/' + PICTL_dict[PICTL_TYPE] + "*.nc")]
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        print("Found " + str(len(files)) + ' ' + PICTL_TYPE + ' ' + var + ' files')

        data_mf = xr.open_mfdataset(files, combine = "by_coords")
        print("PICTL files loaded")

        tmp = data_mf.salem.subset(shape = shapefile)
        mean_lon = tmp.lon.values.mean()
        mean_lat = tmp.lat.values.mean()
        data_sub = data_mf.salem.subset(corners = ((mean_lon - 54, mean_lat + 15),(mean_lon + 18, mean_lat - 15)))

        # Remove seasonality from tas and slp data    
        tas_pictl = data_sub.assign(anom = (('time', 'lat', 'lon'), L0.seasonality_removal_vals(data_sub[var].values, k=3)))

        print("PICTL seasonality removed")

        # Filter slp to JJA
        tas_pictl = tas_pictl.anom.sel(time = tas_pictl.time.dt.month.isin(range(5,10)))

        # Save files
        tas_pictl.to_netcdf(path = DIR + LOC + '/L0/' + var + '_' + PICTL_TYPE + "_"  + LOC + "_MJJAS_anom.nc")

        print("Done with " + var + " in " + PICTL_TYPE)



