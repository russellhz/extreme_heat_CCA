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
DIR = "/glade/scratch/horowitz/extreme_heat_CCA/AMJJAS_anom/"
ODIR = "/glade/scratch/horowitz/extreme_heat_CCA/distance_matrices/"

import L0_functions as L0

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('LOC', type=str, help='region')
parser.add_argument('PICTL', type=str, help='PiCTL simulation')
args = parser.parse_args()

LOC = args.LOC
PICTL = args.PICTL

# Create indices for each day of MJJA
# 183 days in AMJJAS
# 123 days in MJJA
summer_dict = {}
all_days = list(range(1799*183))
days_by_year = np.array_split(np.array(all_days), 1799) # split into array for each year

for i in range(123):
    # Should be comparing to 7 days before (23 days more than i because April has 30 days) until
    # 7 days after (37 days after i)
    days_to_compare = list(range(i+23, i+38))
    other_years = [x[days_to_compare] for x in days_by_year]
    summer_dict[i] = [item for sublist in other_years for item in sublist]


print("Starting " + LOC + ' for ' + PICTL)
# PICTL Files
slp_pictl_fn = DIR + "PSL_" + PICTL + "_"  + LOC + "_AMJJAS_anom.nc"

##################### LOAD VAR DATA ##################
slp_pictl = xr.open_dataset(slp_pictl_fn)
slp_pictl['lat']  =  np.round(slp_pictl.lat, 2)

print("slp files loaded")

###################### FILTER DATA ###############################
# Filter to JMJA
slp_MJJA = slp_pictl.anom.sel(time = slp_pictl.time.dt.month.isin([5,6,7,8])).values

ndays = slp_MJJA.shape[0]
nlat = slp_MJJA.shape[1]
nlon = slp_MJJA.shape[2]
nyear = ndays/123

slp_MJJA = slp_MJJA.reshape((ndays, nlat*nlon))

print("MJJA reshaped")

# Save AMJJAS data
slp_AMJJAS = slp_pictl.anom.values

ndays = slp_AMJJAS.shape[0]

slp_AMJJAS = slp_AMJJAS.reshape((ndays, nlat*nlon))

print("AMJJAS reshaped")

for i in range(123):
    # Find scores between JJA day and relevant MJJAS days
    scores = euclidean_distances(slp_MJJA[range(i, 1799*123, 123)], slp_AMJJAS[summer_dict[i]])
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