#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:28:10 2020

@author: rhorowitz
"""
import numpy as np
import pandas as pd
import xarray as xr
import random
import glob
import re
import salem


def fourier(period, k, n):
    # Get periods for each fourier
    p = [x/period for x in range(1,k+1)]
    # Create array of ones
    result = np.ones( (n, 2*k+1) )
    # Fill in array with fourier series
    for i in range(k):
        result[:,(2*i)] = [np.sin(2*np.pi*x*p[i]) for x in range(1,n+1)]
        result[:,(2*i+1)] = [np.cos(2*np.pi*x*p[i]) for x in range(1,n+1)]
    return result

################################################################################

def seasonality_removal_worker(data, I, X):    
    # Regress fourier series on data
    B = I.dot(data)
    y_pred = X.dot(B)
    return data - y_pred

################################################################################

def seasonality_removal(data, k):
    n=len(data)
    # Create fourier series to regress
    fX = fourier(365, k=k, n=n)
    B = np.linalg.inv(fX.T.dot(fX)).dot(fX.T)
    # Stack data and apply worker function to each gridcell
    data_stack = data.stack(gridcell=('lat', 'lon'))
    result = data_stack.groupby('gridcell').map(seasonality_removal_worker, I = B, X = fX).unstack('gridcell')
    return result

################################################################################

def seasonality_removal_vals(data, k):
    n=len(data)
    # Create fourier series to regress
    fX = fourier(365, k=k, n=n)
    B = np.linalg.inv(fX.T.dot(fX)).dot(fX.T)
    # reshape data to apply worker function to each gridcell
    data_reshape = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    result_reshape = np.apply_along_axis(seasonality_removal_worker, 0, data_reshape, I = B, X = fX)
    result = result_reshape.reshape((data.shape[0], data.shape[1], data.shape[2]))
    return result

################################################################################

def monthselector(month, month1, month2):
    return (month >= month1) & (month <= month2)

################################################################################

def dayofyear_selector(day, rng):
    return (day >= rng[0]) & (day <= rng[-1])

################################################################################

def construct_analog(slp, dist, dist_dates, slp_ref, tas_ref, n, n_s, n_r, lat, lon):
    # Get dates within 7 days of dayof
    dayof = slp.time.dt.dayofyear.values

    days2compare = list(range(dayof - 7, dayof + 8))
    days2compare = [f + 365 if f < 1 else f for f in days2compare ]
    days2compare = [f - 365 if f > 365 else f for f in days2compare ]    

    date_inds = np.where(slp_ref.time.dt.dayofyear.isin(days2compare))[0]
    slp_ind = np.where(slp.time == dist_dates)[0][0]
    z = dist[slp_ind, date_inds]
    
    comp_dates = slp_ref.time[date_inds]
        
    # Get top score in each year, then trim to top n years
    df = pd.DataFrame({'date':comp_dates.values, 'year':comp_dates['time.year'].values, 'score':z})
    minscore_idx = df.groupby(['year'])['score'].transform(min) == df['score']
    df = df[minscore_idx].sort_values('score')[0:n]  
      
    result = np.zeros((lat.size, lon.size, n_r))
    for i in range(n_r):
        random_analog_dates = np.random.choice(df.date.values, n_s, replace=False)  
        
        # Select analogs
        analogs = slp_ref.sel(time = random_analog_dates)
        
        # Convert data to matrices then calculate coefficients
        Xh = slp.stack(gridcell=('lat', 'lon'))
        Xc = analogs.stack(gridcell=('lat', 'lon')).transpose('gridcell','time')
        B = np.linalg.lstsq(Xc, Xh, rcond = None)[0]
                
        # Construct temperature analog
        Tc = tas_ref.sel(time=random_analog_dates).stack(gridcell=('lat', 'lon')).transpose('gridcell','time')
        Tca = np.matmul(Tc.values, B).reshape(-1, tas_ref.lon.shape[0])
        Tca = xr.DataArray(Tca, coords=[tas_ref.lat, tas_ref.lon], dims=['lat', 'lon']).sel(lon = lon, lat = lat)
        result[:, :, i] = Tca
     
    Tca_avg = np.apply_along_axis(np.mean, 2, result)
    Tca_avg = xr.DataArray(Tca_avg, coords=[lat, lon], dims=['lat', 'lon'])

    return Tca_avg

################################################################################
def file_getter(var, DIR, file_start):
    fns = [f for f in glob.glob(DIR + var + '/' + file_start + "*.nc")]
    fns.sort(key=lambda f: int(re.sub('\D', '', f)))
    return fns

        