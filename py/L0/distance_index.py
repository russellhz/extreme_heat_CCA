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

###################### INDICES FOR DISTANCE MATRICES ###############################


for i in range(92):
    # Create empty array of same size as similarity array
    sim_idx = np.empty(nyear,year_len*(nyear-1))
    num = year_len*(nyear-1)
    
    # List of year
    idx_adj = list(range(num))
    min_day = i+24
    year = [math.floor(x/year_len) for x in idx_adj]
    
    # For each year, add correct indices
    for j in range(nyear):
        # Increase year by 1 if less than j, since year j has been deleted
        year_j = [x + 1 if x >= j else x for x in year]
        # Add index for MJJAS of each day
        sim_idx[j,:] = [int(153*year_j[k]+(k % year_len + min_day)) for k in range(num)]
    
    if i%10==0:
        print("Done with " + str(i))  
    np.save(DIR + 'distance_matrices/distance_index_' + str(i), sim_idx.astype(int))