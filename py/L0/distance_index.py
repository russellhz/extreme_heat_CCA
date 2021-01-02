#!/usr/bin/env python3

import xarray as xr
import os
import glob
import re
import numpy as np
import pandas as pd
import random
import math

# Directories
os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L0")
ODIR = "/glade/scratch/horowitz/extreme_heat_CCA/distance_matrices/"

import L0_functions as L0

year_len = 15
nyear = 1799
ndays = 123 # 123 days in MJJA
min_adder = 23 # 23 days after april 1
AMJJAS_days = 183

###################### INDICES FOR DISTANCE MATRICES ###############################


for i in range(ndays):
    # Create empty array of same size as similarity array
    sim_idx = np.empty((nyear,year_len*(nyear-1)))
    num = year_len*(nyear-1)
    
    # List of year
    idx_adj = list(range(num))
    min_day = i+min_adder 
    year = [math.floor(x/year_len) for x in idx_adj]
    
    # For each year, add correct indices
    for j in range(nyear):
        # Increase year by 1 if less than j, since year j has been deleted
        year_j = [x + 1 if x >= j else x for x in year]
        # Add index for AMJJAS of each day
        sim_idx[j,:] = [int(AMJJAS_days*year_j[k]+(k % year_len + min_day)) for k in range(num)]
    
    if i%10==0:
        print("Done with " + str(i))  
    np.save(ODIR + 'distance_index_' + str(i), sim_idx.astype(int))