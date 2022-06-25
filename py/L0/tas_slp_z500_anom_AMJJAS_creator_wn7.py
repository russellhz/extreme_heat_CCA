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
DIR = "/glade/scratch/horowitz/extreme_heat_CCA/AMJJAS_anom/"
import L0_functions as L0


##################### INPUTS ###########################
PICTL_DIR = "/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/"
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/data/shape_files/"

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
for var in ['TREFHT', 'PSL', 'Z500']:
    for PICTL_TYPE in ['PICTL']:
        # PiCTL files
        files = [f for f in glob.glob(PICTL_DIR + var + '/' + PICTL_dict[PICTL_TYPE] + "*.nc")]
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        print("Found " + str(len(files)) + ' ' + PICTL_TYPE + ' ' + var + ' files')

        data_mf = xr.open_mfdataset(files, combine = "by_coords")
        print("PICTL files loaded")

        mean_lon = shapefile.centroid.x.values[0]
        mean_lat = shapefile.centroid.y.values[0]
        data_sub = data_mf[var].salem.subset(corners = ((mean_lon - 90, mean_lat + 25),(mean_lon + 30, mean_lat - 25)))
        data_sub = data_sub.to_dataset()

        # Remove seasonality from tas and slp data    
        data_anom = data_sub.assign(anom = (('time', 'lat', 'lon'), L0.seasonality_removal_vals(data_sub[var].values, k=3)))

        print("PICTL seasonality removed")

        # Filter slp to AMJJAS
        data_anom = data_anom.anom.sel(time = data_anom.time.dt.month.isin(range(4,10)))

        # Save files
        data_anom.to_netcdf(path = DIR + var + '_' + PICTL_TYPE + "_"  + LOC + "_AMJJAS_anom_wn7.nc")

        print("Done with " + var + " in " + PICTL_TYPE)


