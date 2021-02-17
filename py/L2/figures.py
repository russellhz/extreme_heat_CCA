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
import random
import argparse
from itertools import chain
import geopandas
import salem
import math
import string
import copy
import scipy
import random
import statsmodels.api as sm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

PICTL_TYPE = 'PICTL'

DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/"
ODIR = "/glade/work/horowitz/extreme_heat_CCA/L2/"
VAR_DIR = "/glade/work/horowitz/extreme_heat_CCA/L1/JJA_variability/"
SHAPE_DIR = "/glade/u/home/horowitz/extreme_heat_CCA/data/shape_files/"
MJJA_DIR = "/glade/scratch/horowitz/extreme_heat_CCA/MJJA_anom/"
AMJJAS_DIR = "/glade/scratch/horowitz/extreme_heat_CCA/AMJJAS_anom/"
CESMLE_DIR = "/glade/scratch/horowitz/extreme_heat_CCA/JJA_anom/"

sys.path.append("/glade/u/home/horowitz/extreme_heat_CCA/py/L1")
import L1_functions as L1

os.chdir("/glade/u/home/horowitz/extreme_heat_CCA/py/L2")
import L2_functions as L2

shapefns = [f for f in glob.glob(SHAPE_DIR + '*') if 'CONUS' not in f]

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", 
        "OhioValley", "South", "Southeast", 
        "Southwest", "UpperMidwest", "West"] 

LOC_DICT = {'Northwest':'Northwest', 'West':'West', 'Southwest':'Southwest', 
            'NorthernRockiesandPlains':'Northern Plains',
        'UpperMidwest': 'Upper Midwest', 'South':'South', 'Northeast':'Northeast', 
            'OhioValley':'Ohio Valley', 'Southeast':'Southeast'}

######################################################################
##############                  FIGURE 1                ##############
######################################################################

# Load variance as fake data
tas_var_files = [f for f in glob.glob(VAR_DIR + "tas_d2d_variance_*_" + PICTL_TYPE + '.nc')]
data = L2.tas_JJA_var_data_combiner(tas_var_files)
t = data.anom.values
t[~np.isnan(t)] = np.nan
data = data.assign(anom = (('lat', 'lon'), t))

# Load example SLP data
fn = DIR + 'gridded_slp_z500/gridded_slp_z500_' + 'PICTL' + '_' + 'South' + '.nc'
slp = xr.open_dataset(fn).mean(dim='time')

t = slp.slp.values
t[~np.isnan(t)] = 1
slp = slp.assign(slp = (('lat', 'lon'), t))

# Plot
plot_colors = {'Northeast':'deeppink', 'NorthernRockiesandPlains':'darkorange', 'Northwest':'green',
         'OhioValley':'red', 'South':'purple', 'Southeast':'brown',
         'Southwest':'blue', 'UpperMidwest':'olive', 'West':'lightseagreen'}

map1_extent = [data.lon.min()-1, data.lon.max()+1, 
          data.lat.min()-7, data.lat.max()+1]
map2_extent = [slp.lon.min()-1, slp.lon.max()+1, 
          slp.lat.min()-7, slp.lat.max()+1]

titles = ['a.', 'b.']

ofn = ODIR + 'regions.png'

fig = plt.figure(figsize=(30, 12))
spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios = [10,10], height_ratios = [10])
spec.update(wspace=0.1, hspace=0)
specs = [spec[0,0], spec[0,1]]
size = 22

for i in range(2):

    if i == 0:
        ax = fig.add_subplot(specs[i], 
                     projection=ccrs.Orthographic(data.lon.values.mean(), data.lat.values.mean()))
        for LOC in LOCS:
            L2.shapefile_transform(SHAPE_DIR + LOC + '.json').plot(ax=ax,  transform=ccrs.PlateCarree(), edgecolor="black", 
                                                                facecolor=plot_colors[LOC], linewidth = 3, alpha = 0.4)

        ax.text(0.12, 0.76, 'Northwest', fontsize=size, transform=ax.transAxes, rotation = -8)
        ax.text(0.16, 0.60, 'West', fontsize=size, transform=ax.transAxes, rotation = -4)
        ax.text(0.27, 0.75, 'Northern Plains', fontsize=size, transform=ax.transAxes, rotation = -4)
        ax.text(0.23, 0.44, 'Southwest', fontsize=size, transform=ax.transAxes, rotation = -5)
        ax.text(0.52, 0.73, 'Upper\nMidwest', fontsize=size, transform=ax.transAxes)
        ax.text(0.54, 0.53, 'Ohio Valley', fontsize=size, transform=ax.transAxes)
        ax.text(0.44, 0.40, 'South', fontsize=size, transform=ax.transAxes)
        ax.text(0.73, 0.64, 'North-\neast', fontsize=size, transform=ax.transAxes)
        ax.text(0.62, 0.36, 'Southeast', fontsize=size, transform=ax.transAxes)
        ax.set_extent(map1_extent)
    if i ==1:
        ax = fig.add_subplot(specs[i], 
             projection=ccrs.Orthographic(slp.lon.values.mean(), slp.lat.values.mean()))
        CS = plt.contourf(slp.lon.values, slp.lat.values, slp.slp.values, 
                  transform=ccrs.PlateCarree(), linewidths = 4, cmap = 'Blues',
                  add_colorbar=False)
        ax.set_extent(map2_extent)
        L2.shapefile_transform(SHAPE_DIR + 'South' + '.json').plot(ax=ax,  transform=ccrs.PlateCarree(), edgecolor="black", 
                                                                facecolor='grey', linewidth = 3.5, alpha = 0.4)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linewidth = 0.2)
    ax.set_title(titles[i] , fontsize = 26, loc = 'left')
plt.savefig(ofn, dpi=199, bbox_inches = 'tight', pad_inches = 0.2)


######################################################################
##############                  FIGURE 2                ##############
######################################################################
FIG2_DIR = DIR + "dayofyear_sd/"
slp_df = pd.read_csv(FIG2_DIR + "slp_dayofyear_mean_sd_"+PICTL_TYPE+".csv")
tas_df = pd.read_csv(FIG2_DIR + "tas_dayofyear_mean_sd_"+PICTL_TYPE+".csv")

slp_df.region = (slp_df.region.str
                 .replace('NorthernRockiesandPlains', 'Northern Plains')
                 .replace('OhioValley', 'Ohio Valley')
                 .replace('UpperMidwest', 'Upper Midwest')
                )
tas_df.region = (tas_df.region.str
                 .replace('NorthernRockiesandPlains', 'Northern Plains')
                 .replace('OhioValley', 'Ohio Valley')
                 .replace('UpperMidwest', 'Upper Midwest')
                )
slp_df.sd = slp_df.sd/100

# MAKE PLOT

plot_colors = {'Northeast':'deeppink', 'Northern Plains':'darkorange', 'Northwest':'green',
         'Ohio Valley':'red', 'South':'purple', 'Southeast':'brown',
         'Southwest':'blue', 'Upper Midwest':'olive', 'West':'lightseagreen'}


fig = plt.figure(figsize=(20, 7.5))
widths = [10, 10]
spec = fig.add_gridspec(ncols=2, nrows=1, 
                        width_ratios=widths)
spec.update(wspace=0.2, hspace=0.1)
specs = [spec[0,0], spec[0,1]]

datas = [slp_df, tas_df]
ylabs = ['Standard Deviation (hPa)', 'Standard Deviation ($\degree$C)']
titles = ['a. SLP', 'b. Temperature']
for i in range(len(datas)):
    data = datas[i].copy()
    data['dayofyear'] = data['dayofyear']
    ax = fig.add_subplot(specs[i])
    grouped = data.groupby('region')
    for key, group in grouped:
        group.plot(ax=ax, kind='line', x='dayofyear', y='sd', 
                   label = key,   color=plot_colors[key], lw=2)
    ax.set_xlabel('Date', fontsize = 18)
    ax.set_ylabel(ylabs[i], fontsize = 18)
    ax.set_title(titles[i], fontsize = 22, loc = 'left')
    plt.xticks([152,182,213,243], ['6/1','7/1','8/1','8/31'])
    ax.tick_params(axis='both', which='major', labelsize=14)    
    ax.get_legend().remove()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 16)
fig.savefig(ODIR + 'sd_tas_slp_JJA.png', dpi=199, bbox_inches = "tight")

######################################################################
##############                  FIGURE 3                ##############
######################################################################
FIG3_DIR = DIR + "rmse/"

# INPUT DATA 
rmse_files = [f.replace(FIG3_DIR, '') for f in glob.glob(FIG3_DIR + 'rmse_*.nc')]
rmse_file_info = [f.replace('.nc','').split('_') for f in rmse_files]
cols = ['region', 'N_R', 'N_Y', 'rmse']
rmse_df = pd.DataFrame(columns = cols)

N_R = [1, 5, 10, 20]
N_Y = [50, 100, 250, 500, 1000, 1400, 1798]

file_combos = []
for r in N_R:
    for y in N_Y:
        ext =  str(r) + '_' + str(y)
        file_combos.append(ext)
        
for combo in file_combos:
    us_data = pd.DataFrame()
    for i in range(len(rmse_files)):  
        if re.search('_' + combo + '.nc', rmse_files[i]):
            # Add regional rmse to df
            data = xr.open_dataset(FIG3_DIR + rmse_files[i]).rmse
            
            # Area mean
            weights = np.cos(np.deg2rad(data.lat))
            weights.name = "weights"
            rmse = data.weighted(weights).mean(("lon", "lat")).values.tolist()
            #rmse = np.nanmean(data.rmse.values)
            
            info = rmse_file_info[i]
            rmse_df = rmse_df.append(pd.DataFrame(data = [[info[1], info[4], info[5], rmse]], columns = cols))
            # Combine with others of same vars - get US rmse
            us_data  = us_data.append(data.to_dataframe().dropna())
    if len(us_data) > 0:
        # Reset index for weighted average
        us_data = us_data.reset_index()
        rmse_df = rmse_df.append(pd.DataFrame(data = [['US', info[4], info[5], 
                                                       np.average(us_data.rmse, weights=(np.deg2rad(us_data.lat)))]],
                                              columns = cols))
    
rmse_df = rmse_df.astype({'N_R': 'int32', 'N_Y': 'int32'})
rmse_df = rmse_df.reset_index(drop=True)

rmse_df.region = (rmse_df.region.str
                 .replace('NorthernRockiesandPlains', 'Northern Plains')
                 .replace('OhioValley', 'Ohio Valley')
                 .replace('UpperMidwest', 'Upper Midwest')
                )

# PLOT FIGURE 
ofn = ODIR + 'rmse_by_region.png'
plot_colors = {"US":'black', 'Northeast':'deeppink', 'Northern Plains':'darkorange', 'Northwest':'green',
         'Ohio Valley':'red', 'South':'purple', 'Southeast':'brown',
         'Southwest':'blue', 'Upper Midwest':'olive', 'West':'lightseagreen'}

fig = plt.figure(figsize=(20, 7.5))
widths = [10, 10]
spec = fig.add_gridspec(ncols=2, nrows=1, 
                        width_ratios=widths)
spec.update(wspace=0.2, hspace=0.1)
specs = [spec[0,0], spec[0,1]]

datas = [rmse_df.query('N_Y=="1798"'), 
         rmse_df.query('N_R=="10"')]
xvars = ['N_R', 'N_Y']
xlabs = ['Number of Iterations',
         'Number of Years Available']
ylabs = ['RMSE ($\degree$C)', 
         'RMSE ($\degree$C)']
titles = ['a. $\it{N_y}$=1798', 
          'b. $\it{N_r}$=10']
xticks = [[1, 5, 10, 20], [100, 250, 500, 1000, 1400, 1798]]
for i in range(len(datas)):
    data = datas[i].copy()
    ax = fig.add_subplot(specs[i])
    for label, grp in data.groupby('region'):
        if label == 'US':
            grp.plot(x = xvars[i], y = 'rmse',ax = ax, label = label, c = plot_colors[label], marker='o', 
                     lw = 5, ms=12)
        else:
            grp.plot(x = xvars[i], y = 'rmse',ax = ax, label = label, c = plot_colors[label], marker='o',
                    linestyle = '--')
    ax.set_xlabel(xlabs[i], fontsize = 18)
    ax.set_ylabel(ylabs[i], fontsize = 18)
    ax.set_title(titles[i], fontsize = 22, loc = 'left')
    ax.tick_params(axis='both', which='major', labelsize=14) 
    ax.set_xticks(xticks[i])
    handles,labels = ax.get_legend_handles_labels()
    handles.insert(0, handles.pop(labels.index('US')))
    ax.legend(handles, plot_colors.keys())
    ax.get_legend().remove()
plt.legend(handles, plot_colors.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 16)
fig.savefig(ofn, dpi=199, bbox_inches = "tight")

for PICTL_TYPE in ['SSTPICTL', 'PICTL']:
    ######################################################################
    ##############               FIGURE 4 and S1            ##############
    ######################################################################
    # Combine all averaged gridded comp data
    gridded_comp_files = [f for f in glob.glob(DIR + "gridded_composite/gridded_composite_" + PICTL_TYPE + '*.nc')]
    tas = L2.avg_top_dyn_data_combiner(gridded_comp_files, 'tas')
    dynamic = L2.avg_top_dyn_data_combiner(gridded_comp_files, 'dynamic')
    residual = L2.avg_top_dyn_data_combiner(gridded_comp_files, 'residual')
    print("gridded data loaded")

    # Combine all tas variability
    tas_var_files = [f for f in glob.glob(VAR_DIR + "tas_d2d_variance_*_" + PICTL_TYPE + '.nc')]
    tas_JJA_var = L2.tas_JJA_var_data_combiner(tas_var_files).rename({'anom':'tas_JJA_var'})
    tas_JJA_sd = np.sqrt(tas_JJA_var).rename({'tas_JJA_var':'tas_JJA_sd'})

    # Combine all data
    data = xr.merge([tas, dynamic, residual, tas_JJA_sd])

    # Plot
    # make one axis for the map, one axis for the colorbar
    widths = [10, 10]
    heights = [10, 0.5]
    extent = [data.lon.min()-1, data.lon.max()+1, 
              data.lat.min()-7, data.lat.max()+1]

    bounds = [np.arange(0,8,1),
             np.arange(0,4,0.5)]
    datas = [data.tas, data.tas_JJA_sd]
    ofn = ODIR + 'avg_heatwave_and_sd_'+PICTL_TYPE+'.png'

    num = [7,7]

    # To group values of 3.51 with 3.5
    datas = [data.tas, np.minimum(data.tas_JJA_sd, 3.49)]

    fig = plt.figure(figsize=(30, 10))
    heights = [10, 0.5]
    spec = fig.add_gridspec(ncols=2, nrows=2, 
                            width_ratios=widths,
                           height_ratios=heights)
    spec.update(wspace=0.1, hspace=-0.7)
    specs = [spec[0,0], spec[0,1], 
             spec[1,0], spec[1,1]]
    titles = ['a. Average Heatwave', 'b. Intra-Summer Temperature Standard Deviation']
    legend_titles = ['Temperature Anomaly ($^\circ$C)',
                    'Standard Deviation ($^\circ$C)']
    for i in range(len(datas)):
        d = datas[i]
        norm = colors.BoundaryNorm(bounds[i], num[i])
        ax = fig.add_subplot(specs[i], 
                             projection=ccrs.Orthographic(d.lon.values.mean(), d.lat.values.mean()))
        CS = d.plot(ax=ax, transform=ccrs.PlateCarree(),  add_colorbar = False,
                   cmap=plt.get_cmap('YlOrRd', num[i]), levels=bounds[i], norm = norm)
        for fn in shapefns:
            L2.shapefile_transform(fn).plot(ax=ax,  transform=ccrs.PlateCarree(), edgecolor="black", facecolor="None", linewidth = 3.5)
        ax.set_extent(extent)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)
        ax.set_title(titles[i] , fontsize = 26, loc = 'left')
        if i == 0:
            cax = fig.add_subplot(specs[i+2], position=[0.13,-0.05,0.36,0.03])
        if i == 1:
            cax = fig.add_subplot(specs[i+2], position=[0.535,-0.05,0.36,0.03])
        cb = plt.colorbar(CS, cax=cax, orientation='horizontal', extend='neither')
        cb.ax.tick_params(labelsize=16) 
        cb.set_label(legend_titles[i], fontsize=22)

    plt.savefig(ofn, dpi=199, bbox_inches = 'tight', pad_inches = 0.2)

    ######################################################################
    ##############              FIGURE 7  and S2            ##############
    ######################################################################

    datas = [data.residual, data.residual/data.tas]
    titles = ['a. Residual Temperature Anomaly', 'b. Residual Proportion']

    legend_titles = ['Temperature Anomaly ($^\circ$C)',
                'Proportion']
    ofn = ODIR + 'residual_abs_v_prop_'+PICTL_TYPE+'.png'

    bounds = [np.arange(0,1.25,0.2),
             np.arange(0,0.31,0.05)]

    num = [6,6]

    # To group values of -0.02 with 0
    datas = [np.maximum(data.residual, 0.01), 
             np.maximum(data.residual/data.tas, 0.01)]

    widths = [10, 10]
    heights = [10, 0.5]

    fig = plt.figure(figsize=(30, 10))

    spec = fig.add_gridspec(ncols=2, nrows=2, 
                            width_ratios=widths,
                           height_ratios=heights)
    spec.update(wspace=0.1, hspace=-0.15)
    specs = [spec[0,0], spec[0,1], 
             spec[1,0], spec[1,1]]

    for i in range(len(datas)):
        d = datas[i]
        norm = colors.BoundaryNorm(bounds[i], num[i])
        ax = fig.add_subplot(specs[i], 
                             projection=ccrs.Orthographic(d.lon.values.mean(), d.lat.values.mean()))
        CS = d.plot(ax=ax, transform=ccrs.PlateCarree(), 
                   cmap=plt.get_cmap('YlOrRd', num[i]), levels=bounds[i], norm = norm,
                    vmin = bounds[i][0], vmax = bounds[i][-1],
                    add_colorbar = False)
        for fn in shapefns:
            L2.shapefile_transform(fn).plot(ax=ax,  transform=ccrs.PlateCarree(), edgecolor="black", 
                                         facecolor="None", linewidth = 3.5)
        ax.set_extent(extent)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)
        ax.set_title(titles[i] , fontsize = 26, loc = 'left')
        cax = fig.add_subplot(specs[i+2])
        cb = plt.colorbar(CS, cax=cax,boundaries=bounds[i], ticks=bounds[i], orientation='horizontal', 
                          extend='neither',fraction=0.046, norm=norm)
        cb.ax.tick_params(labelsize=16) 
        cb.set_label(legend_titles[i], fontsize=22)

    plt.savefig(ofn, dpi=199, bbox_inches = 'tight', pad_inches = 0.2)

PICTL_TYPE = 'PICTL'

######################################################################
##############                  FIGURE 5                ##############
######################################################################
slp_data_dict = {}
z500_data_dict = {}
for LOC in LOCS:
    fn = DIR + 'gridded_slp_z500/gridded_slp_z500_' + PICTL_TYPE + '_' + LOC + '.nc'
    
    # Find heatwave years
    dates = np.load(DIR + 'dates/top_dyn_composite_dates_' + PICTL_TYPE + '_' + LOC + '.npy', allow_pickle=True)
    years = np.unique([dates[i].year for i in range(len(dates))])
    
    # Open data        
    data = xr.open_dataset(fn)
    
    # Filter to heatwave years and take mean
    data = data.sel(time=data.time.dt.year.isin(years))
    data = data.mean(dim='time')
    
    slp_data_dict[LOC] = data.slp/100 # convert to hPa
    z500_data_dict[LOC] = data.z500
    print("Done with " + LOC)

# Plot
ofn = ODIR + 'avg_top_dyn_slp_pattern_'+PICTL_TYPE+'.png'
min_val = min( [ slp_data_dict[loc].min() for loc in LOCS ] )
max_val = max( [ slp_data_dict[loc].max() for loc in LOCS ] )
abs_val = math.ceil(max([abs(min_val), abs(max_val)])*2)/2
interval = abs_val/5
bounds = list(np.arange(-abs_val, abs_val+interval, interval))[0:11]

cmap=plt.get_cmap('RdBu_r', 10)
norm = colors.BoundaryNorm(bounds, 10)

# make one axis for the map, one axis for the colorbar
fig = plt.figure(figsize=(30, 25))
spec1 = fig.add_gridspec(ncols=3, nrows=3, left=0, right=0.96)
spec1.update(wspace=0, hspace=0)
specs = [spec1[0,0], spec1[0,1], spec1[0,2], 
         spec1[1,0], spec1[1,1], spec1[1,2],
         spec1[2,0], spec1[2,1], spec1[2,2]]

for i in range(len(LOCS)):
    LOC = LOCS[i]
    
    shapefn = SHAPE_DIR + LOC + '.json'
    data = slp_data_dict[LOC]
    extent = [data.lon.min()-1, data.lon.max()+1, 
              data.lat.min()-7, data.lat.max()+1]
    ax = fig.add_subplot(specs[i], 
                         projection=ccrs.Orthographic(data.lon.values.mean(), data.lat.values.mean()))
    CS = plt.contourf(data.lon.values, data.lat.values, data.values, 
                      transform=ccrs.PlateCarree(), linewidths = 4, cmap=cmap,
                      levels=bounds,
                      norm = norm,
                      add_colorbar=False,)
    L2.shapefile_transform(shapefn).plot(ax=ax,  transform=ccrs.PlateCarree(), edgecolor="black", facecolor="None", linewidth = 4)
    ax.set_extent(extent)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_title(string.ascii_lowercase[i] + '. ' + LOC_DICT[LOC] , fontsize = 30, loc = 'left')

heights = [1, 4, 1]

spec2 = fig.add_gridspec(ncols=1, nrows=3, 
                    height_ratios=heights, left=0.97, right=1)
cax = fig.add_subplot(spec2[1,0])
cb = plt.colorbar(CS, cax=cax, ticks=bounds, orientation='vertical', extend='neither')
cb.ax.tick_params(labelsize=25) 
cb.set_label('SLP Anomaly (hPa)', fontsize=30)

plt.savefig(ofn, dpi=199, bbox_inches = 'tight', pad_inches = 0.2)


######################################################################
##############                  FIGURE 6                ##############
######################################################################
# Plot
ofn = ODIR + 'avg_top_dyn_z500_pattern_'+PICTL_TYPE+'.png'

min_val = min( [ z500_data_dict[loc].min() for loc in LOCS ] )
max_val = max( [ z500_data_dict[loc].max() for loc in LOCS ] )
abs_val = math.ceil(max([abs(min_val), abs(max_val)])/5)*5
interval = abs_val/5
bounds = list(np.arange(-abs_val, abs_val+interval, interval))[0:11]

cmap=plt.get_cmap('RdBu_r', 10)
norm = colors.BoundaryNorm(bounds, 10)

# make one axis for the map, one axis for the colorbar
fig = plt.figure(figsize=(30, 25))
spec = fig.add_gridspec(ncols=3, nrows=3, left=0, right=0.96)
spec.update(wspace=0, hspace=0)
specs = [spec[0,0], spec[0,1], spec[0,2], 
         spec[1,0], spec[1,1], spec[1,2],
         spec[2,0], spec[2,1], spec[2,2]]

for i in range(len(LOCS)):
    LOC = LOCS[i]
    shapefn = SHAPE_DIR + LOC + '.json'
    data = z500_data_dict[LOC]
    extent = [data.lon.min()-1, data.lon.max()+1, 
              data.lat.min()-7, data.lat.max()+1]
    ax = fig.add_subplot(specs[i], 
                         projection=ccrs.Orthographic(data.lon.values.mean(), data.lat.values.mean()))
    CS = plt.contourf(data.lon.values, data.lat.values, data.values, 
                      transform=ccrs.PlateCarree(), linewidths = 4, cmap=cmap,
                      levels=bounds,
                      norm = norm,
                      add_colorbar=False,)
    L2.shapefile_transform(shapefn).plot(ax=ax,  transform=ccrs.PlateCarree(), edgecolor="black", facecolor="None", linewidth = 4)
    ax.set_extent(extent)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_title(string.ascii_lowercase[i] + '. ' + LOC_DICT[LOC] , fontsize = 30, loc = 'left')

heights = [1, 4, 1]

spec2 = fig.add_gridspec(ncols=1, nrows=3, 
                    height_ratios=heights, left=0.97, right=1)
cax = fig.add_subplot(spec2[1,0])
cb = plt.colorbar(CS, cax=cax, ticks=bounds, orientation='vertical', extend='neither')
cb.ax.tick_params(labelsize=25) 
cb.set_label('500 hPa Geopotential Height Anomaly (m)', fontsize=30)

plt.savefig(ofn, dpi=199, bbox_inches = 'tight', pad_inches = 0.2)


######################################################################
##############                  FIGURE 8                ##############
######################################################################
gridded_comp_files = [f for f in glob.glob(DIR + "gridded_composite/gridded_composite_" + PICTL_TYPE + '*.nc')]
correlation = L2.corr_top_dyn_data_combiner(gridded_comp_files, 'residual', 'soilwater_10cm')

# FDR Calculation
pvals = correlation.pvalue.values.flatten()
pvals = pvals[~np.isnan(pvals)]

alpha = 0.05
N = len(pvals)
pvals.sort()
for i in range(N):
    x = alpha*(i+1)/N
    if pvals[i] > x:
        thresh = pvals[i -1]
        break
print('Threshold = ' + str(thresh))

correlation = correlation.assign(fdr = correlation.pvalue <= thresh)

# JJA Variance
sm_d2d_var_files = [f for f in glob.glob(VAR_DIR + "sm_d2d_variance_*_" + PICTL_TYPE + '.nc')]
sm_d2d_JJA_var = L2.tas_JJA_var_data_combiner(sm_d2d_var_files).rename({'anom':'sm_d2d_JJA_var'})
sm_d2d_JJA_sd = np.sqrt(sm_d2d_JJA_var).rename({'sm_d2d_JJA_var':'sm_d2d_JJA_sd'})

# JJA Variance
sm_d2d_var_files = [f for f in glob.glob(VAR_DIR + "sm_d2d_variance_*_" + PICTL_TYPE + '.nc')]
sm_d2d_JJA_var = L2.tas_JJA_var_data_combiner(sm_d2d_var_files).rename({'anom':'sm_d2d_JJA_var'})
sm_d2d_JJA_sd = np.sqrt(sm_d2d_JJA_var).rename({'sm_d2d_JJA_var':'sm_d2d_JJA_sd'})

# LOAD IN MEAN DATA 
sm_JJA_mean = xr.open_dataset(VAR_DIR + 'soilwater_10cm_' + PICTL_TYPE + '_CONUS_JJA_mean.nc')
sm_JJA_mean.close()
# apply mask to get same coverage
mask = copy.copy(sm_d2d_JJA_var)
t = mask.sm_d2d_JJA_var.values
t[~np.isnan(t)] = 1
mask = mask.assign(mask = (('lat', 'lon'), t))

sm_JJA_mean = sm_JJA_mean * mask.mask

# Graying out insignificant values
not_idx = np.argwhere(np.logical_not(correlation.fdr.values))
insig_corrs = [correlation.corr.values[tuple(not_idx[i])] for i in range(len(not_idx))]

grey_cutoff = max(abs(np.nanmax(insig_corrs)), abs(np.nanmin(insig_corrs))) + .00001

cmap_orig = plt.get_cmap('RdBu', 10)
cmaplist = [cmap_orig(i) for i in range(cmap_orig.N)]
cmaplist.insert(5, (.5, .5, .5, 1.0))

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, len(cmaplist))

# Plot
datas = [correlation.corr * correlation.fdr, sm_JJA_mean.soilwater_10cm, sm_d2d_JJA_sd.sm_d2d_JJA_sd]

heights = [10, 0.5, 10, 0.5]
extent = [datas[0].lon.min()-1, datas[0].lon.max()+1, 
          datas[0].lat.min()-7, datas[0].lat.max()+1]

bounds = [np.arange(-0.75,0.8,0.15),
          np.arange(0,43,6),
         list(np.arange(0,6.5,0.75))]

# Add in range to allow for grey
bounds_new = list(bounds[0])
bounds_new.insert(5, -grey_cutoff)
bounds_new[6] = grey_cutoff

num = [11,7,8]

norms = [colors.BoundaryNorm(bounds[0], num[0]),
        colors.BoundaryNorm(bounds[1], num[1]),
        colors.BoundaryNorm(bounds[2], num[2])]
norm_new = colors.BoundaryNorm(bounds_new, num[0])

cols = ['RdBu', 'YlOrRd', 'YlOrRd']
         
titles = ['a. Correlation: Residual vs. 1 Week Lag Soil Moisture', 
          'b. Mean Summer Soil Moisture',
          'c. Intra-Summer Soil Moisture Standard Deviation']
legend_titles = ['Correlation',
                 'Soil Moisture (mm)',
                'Standard Deviation (mm)']

positions = [[0.145,0.53,0.347,0.022], 
            [0.532,0.53,0.347,0.022], 
            [0.145,0.14,0.347,0.022], ]

ofn = ODIR + 'sm_corr_mean_sd_'+PICTL_TYPE+'.png'

fig = plt.figure(figsize=(36, 22))

spec = fig.add_gridspec(ncols=2, nrows=4, 
                       height_ratios=heights)
spec.update(wspace=0, hspace=0.2) #0.25
specs = [spec[0,0], spec[0,1], spec[2,0],
         spec[1,0], spec[1,1], spec[3,0]]

for i in range(len(datas)):
    d = datas[i]
    ax = fig.add_subplot(specs[i], 
                         projection=ccrs.Orthographic(d.lon.values.mean(), d.lat.values.mean()))
    if i == 0:
        CS = d.plot(ax=ax, transform=ccrs.PlateCarree(), 
           cmap=cmap, levels=bounds_new, norm = norm_new,
            vmin = bounds_new[0], vmax = bounds_new[-1],
            add_colorbar = False)
    else:
        CS = d.plot(ax=ax, transform=ccrs.PlateCarree(), 
                   cmap=plt.get_cmap(cols[i], num[i]), levels=bounds[i], norm = norms[i],
                    vmin = bounds[i][0], vmax = bounds[i][-1],
                    add_colorbar = False)
    for fn in shapefns:
        L2.shapefile_transform(fn).plot(ax=ax,  transform=ccrs.PlateCarree(), edgecolor="black", 
                                     facecolor="None", linewidth = 3.5)
    ax.set_extent(extent)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_title(titles[i] , fontsize = 24, loc = 'left')
    cax = fig.add_subplot(specs[i+3], position = positions[i])
    if i == 0:
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norms[i], cmap=cmap_orig), cmap=cmap_orig, cax=cax,boundaries=bounds[i], ticks=bounds[i], orientation='horizontal', 
                      extend='neither',fraction=0.046, norm=norms[i])
    else:
        cb = plt.colorbar(CS, cax=cax,boundaries=bounds[i], ticks=bounds[i], orientation='horizontal', 
                          extend='neither',fraction=0.046, norm=norms[i])
    cb.ax.tick_params(labelsize=16) 
    cb.set_label(legend_titles[i], fontsize=22)

plt.savefig(ofn, dpi=199, bbox_inches = 'tight', pad_inches = 0.2)

######################################################################
##############                  FIGURE 9                ##############
######################################################################    
# Land mask file
land_file = "/glade/u/home/horowitz/extreme_heat_CCA/data/soil_map_global.nc"
land = xr.open_dataset(land_file).squeeze('time')
t = land.mrfso.values
t[~np.isnan(t)] = 1
land = land.assign(mask = (('lat', 'lon'), t))
land['lat'] =  np.round(land.lat, 2)
land = land.assign_coords(lon=(((land.lon + 180) % 360) - 180)).sortby("lon")

# Get intra-summer standard deviation of each variable 
sd_fn = DIR + 'JJA_allvar_sd_'+ PICTL_TYPE + '.csv'

if os.path.exists(sd_fn):
    allvar_sd = pd.read_csv(sd_fn)
    print(sd_fn + ' loaded')
else:
    print(sd_fn + ' not found. Beginning standard deviation calculations...')
    data_dict = {}
    for var in ['TS','LHFLX', 'SHFLX', 'SOILWATER_10CM']:
        data = xr.open_dataset(MJJA_DIR + var + '_' +PICTL_TYPE+'_CONUS_MJJA_anom.nc')
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby("lon")
        data['lat'] = data['lat'].astype('float64')
        data['lon'] = data['lon'].astype('float64')
        data['lat'] =  np.round(data.lat, 2)
        data = data.sel(time = data.time.dt.month.isin(range(6,9)))
        print('filtered to JJA')

        var_dict = {}
        for LOC in LOCS:

            ###########################################################################################################
            ###################################        SHAPEFILE        ###############################################
            ###########################################################################################################
            shapefn = SHAPE_DIR + LOC + '.json'
            shapefile = geopandas.read_file(shapefn)
            shapefile.crs =  {'init': 'epsg:3857'}
            shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
            shapefile.crs = 'epsg:4326'

            ###########################################################################################################
            ###################################       REGIONAL MEAN        ###############################################
            ###########################################################################################################
            loc_sd = data.salem.subset(shape = shapefile)
            loc_sd = loc_sd.anom.salem.roi(shape = shapefile)  * land.mask

            # Regional mean
            weights = np.cos(np.deg2rad(loc_sd.lat))
            weights.name = "weights"
            loc_sd_mean = loc_sd.weighted(weights).mean(("lon", "lat"))
            # loc_sd = loc_sd.mean(dim=('lon', 'lat'))
            loc_sd_mean = np.sqrt(loc_sd_mean.groupby(loc_sd_mean.time.dt.year).var().values.mean())

            print('done with ' + LOC)

            var_dict[LOC] = loc_sd_mean
            data_dict[var] = var_dict
        print('DONE WITH ' + var)
        print('----------------------------------------')
        
    ###########################################################################################################
    ###################################        TREFHT        ###############################################
    ###########################################################################################################
    var_dict = {}
    var="TREFHT"
    for LOC in LOCS:

        data = xr.open_dataset(AMJJAS_DIR + var + '_' +PICTL_TYPE+'_'+LOC+'_AMJJAS_anom.nc')
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby("lon")
        data['lat'] =  np.round(data.lat, 2)
        data = data.sel(time = data.time.dt.month.isin(range(6,9)))

        ###########################################################################################################
        ###################################        SHAPEFILE        ###############################################
        ###########################################################################################################
        shapefn = SHAPE_DIR + LOC + '.json'
        shapefile = geopandas.read_file(shapefn)
        shapefile.crs =  {'init': 'epsg:3857'}
        shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
        shapefile.crs = 'epsg:4326'

        ###########################################################################################################
        ###################################       REGIONAL MEAN        ###############################################
        ###########################################################################################################
        loc_sd = data.salem.subset(shape = shapefile)
        loc_sd = loc_sd.anom.salem.roi(shape = shapefile)  * land.mask

        # Regional mean
        weights = np.cos(np.deg2rad(loc_sd.lat))
        weights.name = "weights"
        loc_sd_mean = loc_sd.weighted(weights).mean(("lon", "lat"))
        # loc_sd = loc_sd.mean(dim=('lon', 'lat'))
        loc_sd_mean = np.sqrt(loc_sd_mean.groupby(loc_sd_mean.time.dt.year).var().values.mean())

        print('done with ' + LOC)

        var_dict[LOC] = loc_sd_mean
        data_dict[var] = var_dict
    print('DONE WITH ' + var)
    print('----------------------------------------')

    allvar_sd = pd.DataFrame(data_dict)
    allvar_sd.to_csv(sd_fn)

allvar_sd = allvar_sd.rename(columns={'Unnamed: 0':'region'})

# Create composite data
df_all = pd.DataFrame()
for LOC in LOCS:

    fn = DIR + "gridded_composite/gridded_composite_" + PICTL_TYPE + '_'+ LOC +'.nc'

    data = xr.open_dataset(fn) * land.mask
    # Take weighted spatial mean
    weights = np.cos(np.deg2rad(data.lat))
    weights.name = "weights" 
    data = data.weighted(weights).mean(("lon", "lat"))

    # convert to dataframe
    df = data.to_dataframe().reset_index()
    df['year'] = df.time.astype(str).str[0:4].astype(int)

    # Add day of heatwave to dataframe
    days = []
    x = np.unique(data.time.dt.year)
    days_per_year = int(len(df)/len(x))
    for yr in x:
        days.extend(range(1,days_per_year+1))
    df['day'] = days
    df['day'] = df['day'] - (days_per_year-7)

    # Get top and bottom years
    top_bottom_residual = df[['residual', 'year', 'day']].query('day>=1')
    top_bottom_residual = top_bottom_residual.groupby('year').mean().reset_index()

    bottom_thresh = top_bottom_residual.quantile(q=0.25)['residual']
    top_thresh = top_bottom_residual.quantile(q=0.75)['residual']

    bottom_years = top_bottom_residual[top_bottom_residual['residual'] <= bottom_thresh].year
    top_years = top_bottom_residual[top_bottom_residual['residual'] >= top_thresh].year

    # All years data
    all_years = df.groupby('day').mean().drop(columns='year').reset_index().assign(year_type='all',region=LOC)

    # top years data
    top_years = df[df['year'].isin(top_years)].groupby('day').mean().drop(columns='year').reset_index().assign(year_type='top',region=LOC)

    # bottom years data
    bottom_years = df[df['year'].isin(bottom_years)].groupby('day').mean().drop(columns='year').reset_index().assign(year_type='bottom',region=LOC)
    
    df_all = pd.concat([df_all, all_years, top_years, bottom_years])

    print("Done with " + LOC)

sd_long = pd.melt(allvar_sd.rename(columns={'TREFHT':'tas'}),
       id_vars = ['region'],
       value_vars=['tas', 'TS', 'LHFLX', 'SHFLX', 'SOILWATER_10CM'],
                 value_name = 'sd')
sd_long.variable = sd_long.variable.str.lower()

#Copy tas to dynamic
dynamic = sd_long.query('variable=="tas"')
dynamic['variable'] = 'dynamic'

sd_long = pd.concat([sd_long, dynamic])

df_long = pd.melt(df_all, id_vars = ['day', 'region', 'year_type'], 
        value_vars=['tas', 'ts', 'dynamic', 'residual', 'lhflx', 'shflx', 'soilwater_10cm'])

df_long = pd.merge(df_long, sd_long, how = 'left')
df_long['value_sd'] = df_long['value']/ df_long['sd']



df_long = df_long.drop(columns = ['value', 'sd'])

df_wide = df_long.pivot_table(index = ['day', 'region', 'year_type'], columns='variable')

df_wide.columns = df_wide.columns.droplevel(0)
df_wide = df_wide.reset_index().rename_axis(None, axis=1)

ofn = ODIR + 'progression_lhflx_shflx_' + PICTL_TYPE + '.png'

plot_colors = {'Northeast':'turquoise', #'NorthernRockiesandPlains':'red', 
          'Northwest':'darkcyan','South':'darkred'}
plot_data = df_wide[df_wide['region'].isin(plot_colors.keys())].query('year_type=="all"')

fig = plt.figure(figsize=(20, 7.5))
widths = [10, 10]
spec = fig.add_gridspec(ncols=2, nrows=1, 
                        width_ratios=widths)
spec.update(wspace=0.2, hspace=0.1)
specs = [spec[0,0], spec[0,1]]

y_vars = ['shflx', 'lhflx']
ylabs = ['Standard Deviations', 'Standard Deviations']
titles = ['a. Sensible Heat Flux', 'b. Latent Heat Flux']
for i in range(len(y_vars)):
    ax = fig.add_subplot(specs[i])
    ax.axhline(linestyle='--', color='black')
    #ax.vlines(x=[0], ymin=-3 ,ymax=3, linestyle='--', color='gray', lw=1)
    grouped = plot_data.groupby('region')
    for key, group in grouped:
        group.plot(ax=ax, kind='line', x='day', y=y_vars[i], 
                   label = key,   color=plot_colors[key], lw=5)
    ax = fig.add_subplot(specs[i])
    ax.set_ylabel(ylabs[i], fontsize = 18)
    ax.set_ylim(bottom=-2, top=2.5)
    ax.set_title(titles[i], fontsize = 22, loc = 'left')
    ax.set_xlabel("Days Before and After Heatwave Onset", fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)    
    ax.set_xticks([-28, -21, -14,-7,0,7])
    ax.get_legend().remove()
    
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 16)
plt.legend(loc='lower center', bbox_to_anchor=(-0.13, -0.22), fontsize = 16, ncol=len(plot_colors))

fig.savefig(ofn, dpi=199, bbox_inches = "tight")

######################################################################
##############                  FIGURE 10               ##############
######################################################################  
ofn = ODIR + 'progression_tas_dynamic_sm_' + PICTL_TYPE + '.png'

plot_colors = {'Northeast':'turquoise', #'NorthernRockiesandPlains':'red', 
          'Northwest':'darkcyan','South':'darkred'}
plot_data = [df_wide[df_wide['region'].isin(plot_colors.keys())].query('year_type=="bottom"'),
            df_wide[df_wide['region'].isin(plot_colors.keys())].query('year_type=="top"')]

fig = plt.figure(figsize=(20, 15))
widths = [10, 10, 1]
heights = [2,10,10, 10]

spec = fig.add_gridspec(ncols=3, nrows=4, 
                        width_ratios=widths,
                       height_ratios=heights)
spec.update(wspace=0.07, hspace=0.15)
specs = [spec[1,0], spec[1,1],
        spec[2,0], spec[2,1],
        spec[3,0], spec[3,1]]

y_vars = ['tas', 'dynamic', 'soilwater_10cm']
ylabs = ['Standard Deviations', 'Standard Deviations']
titles = ['a', 'b', 'c', 'd', 'e', 'f']
column_titles = ['Bottom 25% Residual',
                'Top 25% Residual']
col_title_x = [0.2, 0.27]

row_titles = ['Temperature Anomaly',
                'Dynamic Component',
                'Soil Moisture Anomaly']
row_title_x = [0.06, 0.08, 0.05]

for i in range(2*len(y_vars)):
    d = plot_data[i%2]
    ax = fig.add_subplot(specs[i])
    ax.axhline(linestyle='--', color='black')
    #ax.vlines(x=[0], ymin=-3 ,ymax=3, linestyle='--', color='gray', lw=1)
    grouped = d.groupby('region')
    for key, group in grouped:
        group.plot(ax=ax, kind='line', x='day', y=y_vars[math.floor(i/2)], 
                   label = key,   color=plot_colors[key], lw=5)
    ax = fig.add_subplot(specs[i])
    if i%2==0:
        ax.set_ylabel('Standard Deviations', fontsize = 18)
    else:
        ax.yaxis.label.set_visible(False)
    ax.set_ylim(bottom=-3, top=3.25)
    ax.text(0.03, 0.88, titles[i], fontsize=24, 
        transform=ax.transAxes)
    if i>3:
        ax.set_xlabel("Days Before and After Heatwave Onset", fontsize = 18)
    else:
        ax.xaxis.label.set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)    
    ax.set_xticks([-28, -21, -14,-7,0,7])
    ax.get_legend().remove()
plt.legend(loc='lower center', bbox_to_anchor=(-0.05, -0.4), fontsize = 16, ncol=len(plot_colors))

#plt.legend(loc='center left', bbox_to_anchor=(1, 1.7), fontsize = 16)

# Column Titles
for i in range(len(column_titles)):
    ax = fig.add_subplot(spec[0,i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(col_title_x[i], 0.33, column_titles[i], fontsize=26, 
            transform=ax.transAxes, color='white', fontweight='bold')
    ax.set_facecolor('black')
# Row Titles
for i in range(len(y_vars)):
    ax = fig.add_subplot(spec[i+1,2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.33, row_title_x[i], row_titles[i], fontsize=17, 
            transform=ax.transAxes, color='white', fontweight='bold',
           rotation=-90)
    ax.set_facecolor('black')

fig.savefig(ofn, dpi=199, bbox_inches = "tight")

######################################################################
##############                  FIGURE 11               ##############
######################################################################  

data = pd.read_csv(DIR+'cesmle_slp_correlation.csv')

data['year'] = data.date.str[0:4].astype(int)

thresh = 0.7
groupvar = 'decade'
data_prop_all = pd.DataFrame()

for LOC in LOCS:
    df = data.query('region == @LOC & year < 2100')

    # Group by decade and calculate probability of falling in bin
    df['decade'] = (np.floor(df.year.values/10)*10).astype(int)

    # Get count of total
    sum_count = df.groupby(groupvar).count().reset_index()[[groupvar, 'region']].rename(columns = {'region':'total'})

    # Count numbers greater than threshold
    df = df[df.correlation > thresh].groupby(['region', groupvar]).count().reset_index()[[groupvar, 'correlation']]

    # Divide by total in each decade
    df = df.merge(sum_count,on=groupvar,how='left')
    df['proportion'] = df.correlation / df.total
    df['region'] = LOC
    data_prop_all = pd.concat([data_prop_all, df])
    print("done with "+LOC)
data_prop_all.decade = data_prop_all.decade+5
data_prop_all['percent'] = data_prop_all['proportion'] * 100

# Stat significance
pvals = []
for LOC in LOCS:
    df = data_prop_all.query('region == @LOC&decade>=2015')
    x = df[groupvar].values
    y =  df['percent'].values
    x = sm.add_constant(x)
    mod = sm.OLS(y,x).fit()
    pval = mod.summary2().tables[1]['P>|t|']['x1']
    pvals.append(pval)
    print(LOC + ': ' + str(pval))

# FDR Calculation
alpha = 0.05
N = len(pvals)
pvals.sort()
for i in range(N):
    x = alpha*(i+1)/N
    if pvals[i] > x:
        fdr = pvals[i -1]
        break

# Plot

ofn = ODIR + 'cesmle_slp_trends_' + str(thresh) + '_' + groupvar + '.png'

fig = plt.figure(figsize=(30, 26))
widths = [10, 10, 10]
spec = fig.add_gridspec(ncols=3, nrows=3, 
                        width_ratios=widths)
spec.update(wspace=0.3, hspace=0.2)
specs = [spec[0,0], spec[0,1], spec[0,2], 
         spec[1,0], spec[1,1], spec[1,2],
         spec[2,0], spec[2,1], spec[2,2]]

i = 0
for LOC in LOCS:
    df = data_prop_all.query('region == @LOC')
    ax = fig.add_subplot(specs[i])
    x = df[groupvar].values
    y =  df['percent'].values
    
    # Calculate trend line
    df_trend = data_prop_all.query('region == @LOC&decade>=2015')
    x_trend = df_trend[groupvar].values
    y_trend = df_trend['percent'].values
    lin_reg = np.polynomial.polynomial.polyfit(x_trend, y_trend, 1)
    pred = lin_reg[0]+lin_reg[1]*x_trend
    ax.plot(x_trend,pred,"r--")
    
    ax.plot(x, y, marker='o', linestyle='solid')
    ax.set_title(string.ascii_lowercase[i] + '. ' +LOC_DICT[LOC], fontsize=26, loc = 'left')
    ax.set_ylabel('% Days with Correlation > ' + str(thresh), fontsize=22)
    ax.tick_params(labelsize=22)
    #ax.set_xticks([1920, 1950, 1980, 2010, 2040, 2070, 2100])
    ax.set_xticks([1920, 1965, 2010, 2055, 2100])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 2))
    i += 1

plt.savefig(ofn, dpi=100, bbox_inches = 'tight', pad_inches = 0.2)


######################################################################
##############                  FIGURE S3               ##############
######################################################################  
codes = np.unique(data.code)

persistence = pd.DataFrame()

for LOC in LOCS:
    for code in codes:
        data_thresh = data.query('region==@LOC&code==@code&correlation>0.7').reset_index()
        data_thresh['date'] = pd.to_datetime(data_thresh['date'])
        lengths, years = L2.persistence_calculator(data_thresh)
        tmp = pd.DataFrame({'region':LOC, 'code':code, 'duration':lengths, 'year':years})
        persistence = pd.concat([persistence, tmp])
    print('done with ' + LOC)
    
# remove year 2100
persistence = persistence.query('year<2100')
# Group by decade and calculate probability of falling in bin
persistence['decade'] = (np.floor(persistence.year.values/10)*10).astype(int)
persistence.decade = persistence.decade+5

df_mean_duration = persistence.groupby(['region', 'decade']).mean().reset_index()
df_event_count = persistence.groupby(['region','decade']).count().reset_index()

# Plot

ofn = ODIR + 'cesmle_slp_mean_duration_trends_' + str(thresh) + '_' + groupvar + '.png'

groupvar = 'decade'

fig = plt.figure(figsize=(30, 26))
widths = [10, 10, 10]
spec = fig.add_gridspec(ncols=3, nrows=3, 
                        width_ratios=widths)
spec.update(wspace=0.3, hspace=0.2)
specs = [spec[0,0], spec[0,1], spec[0,2], 
         spec[1,0], spec[1,1], spec[1,2],
         spec[2,0], spec[2,1], spec[2,2]]

i = 0
for LOC in LOCS:
    plot_df = df_mean_duration.query('region == @LOC')
    ax = fig.add_subplot(specs[i])

    x = plot_df[groupvar].values
    y =  plot_df['duration'].values
    
    # Calculate trend line
    df_trend = df_mean_duration.query('region == @LOC&decade>=2015')
    x_trend = df_trend[groupvar].values
    y_trend = df_trend['duration'].values
    lin_reg = np.polynomial.polynomial.polyfit(x_trend, y_trend, 1)
    pred = lin_reg[0]+lin_reg[1]*x_trend
    ax.plot(x_trend,pred,"r--")
    
    ax.plot(x, y, marker='o', linestyle='solid')
    ax.set_title(string.ascii_lowercase[i] + '. ' +LOC_DICT[LOC], fontsize=26, loc = 'left')
    ax.set_ylabel('Average Event Duration (days)', fontsize=22)
    ax.tick_params(labelsize=22)
    ax.set_xticks([1920, 1965, 2010, 2055, 2100])

    i += 1

plt.savefig(ofn, dpi=100, bbox_inches = 'tight', pad_inches = 0.2)

######################################################################
##############                  FIGURE S4               ##############
######################################################################  
ofn = ODIR + 'cesmle_slp_event_count_trends_' + str(thresh) + '_' + groupvar + '.png'

groupvar = 'decade'

fig = plt.figure(figsize=(30, 26))
widths = [10, 10, 10]
spec = fig.add_gridspec(ncols=3, nrows=3, 
                        width_ratios=widths)
spec.update(wspace=0.3, hspace=0.2)
specs = [spec[0,0], spec[0,1], spec[0,2], 
         spec[1,0], spec[1,1], spec[1,2],
         spec[2,0], spec[2,1], spec[2,2]]

i = 0
for LOC in LOCS:
    plot_df = df_event_count.query('region == @LOC')
    ax = fig.add_subplot(specs[i])
    x = plot_df[groupvar].values
    y =  plot_df['duration'].values/400
    
    # Calculate trend line
    df_trend = df_event_count.query('region == @LOC&decade>=2015')
    x_trend = df_trend[groupvar].values
    y_trend = df_trend['duration'].values/400
    lin_reg = np.polynomial.polynomial.polyfit(x_trend, y_trend, 1)
    pred = lin_reg[0]+lin_reg[1]*x_trend
    ax.plot(x_trend,pred,"r--")
    
    ax.plot(x, y, marker='o', linestyle='solid')
    ax.set_title(string.ascii_lowercase[i] + '. ' +LOC_DICT[LOC], fontsize=26, loc = 'left')
    ax.set_ylabel('Events per Year', fontsize=22)
    ax.tick_params(labelsize=22)
    ax.set_xticks([1920, 1965, 2010, 2055, 2100])

    i += 1

plt.savefig(ofn, dpi=100, bbox_inches = 'tight', pad_inches = 0.2)

######################################################################
##############                  FIGURE 12               ##############
###################################################################### 
cesmle_fns = [f for f in glob.glob(CESMLE_DIR + "CESM-LE_SOILWATER_10CM_anom_" + "*" + "19200101-21001231.nc")]
cesmle_fns = np.sort(cesmle_fns)  

# AVERAGE SM BY GRID CELL
df = pd.DataFrame()
for fn in cesmle_fns:
    data = xr.open_dataset(fn)
    data = data.groupby(data.time.dt.year).mean()
    code = re.search(r'(?<=_)[0-9]{3}(?=_)', fn).group()
    # REGIONAL AVERAGES
    for LOC in LOCS:
        ### SHAPEFILE       
        shapefn = SHAPE_DIR + LOC + '.json'
        shapefile = geopandas.read_file(shapefn)
        shapefile.crs =  {'init': 'epsg:3857'}
        shapefile['geometry'] = shapefile['geometry'].to_crs(epsg=4326)
        shapefile.crs = 'epsg:4326'

        shapefile_shift = copy.copy(shapefile)
        shapefile_shift['geometry'] = shapefile_shift.translate(xoff=360)

        ### REGION FILTER
        tmp = data.anom.salem.subset(shape=shapefile_shift)
        tmp = tmp.salem.roi(shape=shapefile_shift)
        # tmp = tmp.mean(dim=('lon','lat'))
        
        # Weighted mean
        weights = np.cos(np.deg2rad(tmp.lat))
        weights.name = "weights"
        tmp_weighted = tmp.weighted(weights)
        weighted_mean = tmp_weighted.mean(("lon", "lat"))    
        tmp_df = pd.DataFrame({'region':LOC, 'code':code, 'year':weighted_mean.year, 'sm':weighted_mean})
        
        df = pd.concat([df, tmp_df])
    print(code)

# Group by decade and calculate probability of falling in bin
df['decade'] = (np.floor(df.year.values/10)*10).astype(int)
df.decade = df.decade + 5
df = df.query('decade<2100')
df_grouped = df.groupby(['region', 'decade']).mean().reset_index()

# Load in standard deviations
sd_fn = DIR + 'JJA_allvar_sd_'+ PICTL_TYPE + '.csv'

allvar_sd = pd.read_csv(sd_fn)
allvar_sd = allvar_sd.rename(columns={'Unnamed: 0':'region'})
print(sd_fn + ' loaded')

allvar_sd = allvar_sd[['region', 'SOILWATER_10CM']]
allvar_sd = allvar_sd.rename(columns={'SOILWATER_10CM':'sd'})

df_grouped = pd.merge(df_grouped, allvar_sd, how='left')
df_grouped['value'] = df_grouped['sm'] / df_grouped['sd']

# Plot
plot_colors = {'Northeast':'deeppink', 'NorthernRockiesandPlains':'darkorange', 'Northwest':'green',
         'OhioValley':'red', 'South':'purple', 'Southeast':'brown',
         'Southwest':'blue', 'UpperMidwest':'olive', 'West':'lightseagreen'}

fig = plt.figure(figsize=(12, 7.5))
spec = fig.add_gridspec(ncols=1, nrows=1)
spec.update(wspace=0.2, hspace=0.1)
specs = [spec[0,0]]

ylabs = ['JJA Soil Moisture (Standard Deviations)']
i=0

plot_df = df_grouped[df_grouped.region.isin(['South', 'NorthernRockiesandPlains'])]

ax = fig.add_subplot(specs[i])
grouped = plot_df.groupby('region')
for key, group in grouped:
    group.plot(ax=ax, kind='line', x='decade', y='value', 
               label = key,   color=plot_colors[key], lw=3, marker='o')
ax.xaxis.label.set_visible(False)
ax.set_ylabel(ylabs[i], fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=14)    
ax.get_legend().remove()
ax.set_xticks([1920, 1965, 2010, 2055, 2100])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 16,
          labels = ('Northern Plains','South'))
fig.savefig(ODIR + 'cesmle_sm_trends.png', dpi=199, bbox_inches = "tight")

