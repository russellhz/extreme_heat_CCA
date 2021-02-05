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
import geopandas
import salem
import argparse
import copy

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('VAR', type=str, help='region')
parser.add_argument('PICTL_TYPE', type=str, help='PICTL type')
args = parser.parse_args()

VAR = args.VAR
PICTL_TYPE = args.PICTL_TYPE

# Directories
os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L0")
DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/JJA_variability/"
import L0_functions as L0
PICTL_DIR = "/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/"
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/data/shape_files/"
##################### INPUTS ###########################
# Soil moisture is in lnd folder, everything else is in atm
if VAR == "soilwater_10cm":
    PICTL_DIR = PICTL_DIR + "lnd/proc/tseries/daily/"
else: 
    PICTL_DIR = PICTL_DIR + "atm/proc/tseries/daily/"
    
# PICTL_TYPE file dictionary
PICTL_dict = {'PICTL': 'b.e11.B1850C5CN.f09_g16',
            'SSTPICTL': 'f.e11.F1850C5CN.f09_f09'}

shapefn = SHAPE_DIR + 'CONUS.json'

##################### LOAD CONUS SHAPEFILE DATA ##################
shapefile = geopandas.read_file(shapefn)
shapefile.crs =  {'init': 'epsg:3857'}
shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
shapefile.crs = 'epsg:4326'

shapefile_shift = copy.copy(shapefile)
shapefile_shift['geometry'] = shapefile_shift.translate(xoff=360)

##################### LOAD PICTL DATA ##################
# PiCTL files
files = [f for f in glob.glob(PICTL_DIR + VAR.upper() + '/' + PICTL_dict[PICTL_TYPE] + "*.nc")]
files.sort(key=lambda f: int(re.sub('\D', '', f)))
print("Found " + str(len(files)) + ' ' + PICTL_TYPE + ' ' + VAR + ' files')

# Open each file and take mean
data_dict = {}
for fn in files:
    yrs = re.search(r'(?<=.)[0-9]{8}-[0-9]{8}(?=.)', fn).group()
    data = xr.open_dataset(fn)
    data = data[VAR.upper()].salem.subset(shape = shapefile_shift)
    # Filter to JJA
    data = data.sel(time = data.time.dt.month.isin([6,7,8]))
    data_dict[yrs] = data.mean(dim='time')
    print(yrs)
print('-----------------------')

# Combine all means and take grand mean
lat = data_dict['04030101-04991231'].lat
lon = data_dict['04030101-04991231'].lon

nlat = data_dict['04030101-04991231'].values.shape[0]
nlon = data_dict['04030101-04991231'].values.shape[1]
n = len(data_dict)

x = np.empty([nlat, nlon, n])

for i in range(n):
    x[:,:,i] = list(data_dict.values())[i].values
data_mean = xr.DataArray(x.mean(axis=2),coords=[lat, lon], dims=['lat', 'lon'], name = VAR)

data_mean['lat'] = np.round(data_mean['lat'], 2)

data_mean = data_mean.assign_coords(lon=(((data_mean.lon + 180) % 360) - 180)).sortby("lon")

# Save files
data_mean.to_netcdf(path = DIR + VAR + '_' + PICTL_TYPE + "_CONUS_JJA_mean.nc")

print("Done with " + VAR + " in " + PICTL_TYPE)


print("Fin")
