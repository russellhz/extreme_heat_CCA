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
import math

# Directories
os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L0")
DIR = "/glade/scratch/horowitz/extreme_heat_CCA/MJJAS_anom/"
ODIR = "/glade/scratch/horowitz/extreme_heat_CCA/distance_matrices/"

import L0_functions as L0

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('LOC', type=str, help='region')
parser.add_argument('PICTL', type=str, help='PiCTL simulation')
args = parser.parse_args()

LOC = args.LOC
PICTL = args.PICTL

# Create indices for each day of summer
summer_dict = {}
all_days = list(range(1799*153))
days_by_year = np.array_split(np.array(all_days), 1799) # split into array for each year

for i in range(92):
    # Should be comparing to 7 days before (24 days more than i because May has 31 days) until
    # 7 days after (38 days after i)
    days_to_compare = list(range(i+24, i+39))
    other_years = [x[days_to_compare] for x in days_by_year]
    summer_dict[i] = [item for sublist in other_years for item in sublist]


print("Starting " + LOC + ' for ' + PICTL)
# PICTL Files
slp_pictl_fn = DIR + "PSL_" + PICTL + "_"  + LOC + "_MJJAS_anom.nc"

##################### LOAD VAR DATA ##################
slp_pictl = xr.open_dataset(slp_pictl_fn)
slp_pictl['lat']  =  np.round(slp_pictl.lat, 2)

print("slp files loaded")

###################### FILTER DATA ###############################
# Filter to JJA
slp_JJA = slp_pictl.anom.sel(time = slp_pictl.time.dt.month.isin([6,7,8])).values

ndays = slp_JJA.shape[0]
nlat = slp_JJA.shape[1]
nlon = slp_JJA.shape[2]
nyear = ndays/92

slp_JJA = slp_JJA.reshape((ndays, nlat*nlon))

print("JJA reshaped")

# Save MJJAS data
slp_MJJAS = slp_pictl.anom.values

ndays = slp_MJJAS.shape[0]

slp_MJJAS = slp_MJJAS.reshape((ndays, nlat*nlon))

print("MJJAS reshaped")

for i in range(92):
    # Find scores between JJA day and relevant MJJAS days
    scores = euclidean_distances(var[range(i, 1799*92, 92)], var[summer_dict[i]])
    comp_length = int(scores.shape[1]/1799)
    scores_fixed = np.empty((scores.shape[0],scores.shape[1]-comp_length))
    # Remove distances between same year
    for j in range(scores.shape[0]):
        remove_idx = range(j*comp_length, j*comp_length+comp_length)
        keep_idk = list(set(range(0,scores.shape[1])) - set(remove_idx))
        keep_idk.sort()
        scores_fixed[j,:] = scores[j,keep_idk]
    np.save(ODIR + 'distance_matrix_' + PICTL + '_' + LOC + '_' + str(i), scores_fixed)
    if i%10==0:
        print("Done with " + str(i))
print("--------------------------------------------")