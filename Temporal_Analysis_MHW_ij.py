# -*- coding: utf-8 -*-
"""

######################### Temporal Analysis MHW ij loop (2D) ############################

"""

# Load required modules
 
import netCDF4 as nc
import numpy as np
from scipy import io
from scipy import stats
from datetime import date
from netCDF4 import Dataset 
import xarray as xr 

import marineHeatWaves as mhw
import ecoliver as ecj


# Some basic parameters
coldSpells = False # If true detect coldspells instead of heatwaves
col_clim = '0.25'
col_thresh = 'g-'
if coldSpells:
    mhwname = 'MCS'
    mhwfullname = 'coldspell'
    col_evMax = (0, 102./255, 204./255)
    col_ev = (153./255, 204./255, 1)
    col_bar = (0.5, 0.5, 1)
else:
    mhwname = 'MHW'
    mhwfullname = 'heatwave'
    col_evMax = 'r'
    col_ev = (1, 0.6, 0.5)
    col_bar = (1, 0.5, 0.5)


climatologyPeriod=[1982, 2012]
pctile=95


#Dates
t, dates, T, year, month, day, doy = ecj.timevector([1982, 1, 1], [2021, 12, 31])


#Reading Dataset and objectdataset
ds = Dataset(r'E:\MHW_Antartida\SST_ANT_1982-2021_40.nc', mode='r')


sst = ds['analysed_sst']
lon = ds['lon']
lat = ds['lat']

sst -= 273.15 #K to ºC

#
#Creating metric mask 
#
X = len(lon)
Y = len(lat)
i_which = range(0, X)
j_which = range(0, Y)
DIM = (len(j_which), len(i_which)) #Dimension of the mask (lat, lon)
SST_mean = np.zeros(DIM)*np.NaN
# MHW_total = np.zeros(DIM)*np.NaN
MHW_cnt = np.zeros(DIM)*np.NaN
MHW_dur = np.zeros(DIM)*np.NaN
MHW_max = np.zeros(DIM)*np.NaN
MHW_mean = np.zeros(DIM)*np.NaN
MHW_cum = np.zeros(DIM)*np.NaN
# MHW_var = np.zeros(DIM)*np.NaN
MHW_td = np.zeros(DIM)*np.NaN
# MHW_tc = np.zeros(DIM)*np.NaN
SST_tr = np.zeros(DIM)*np.NaN
MHW_cnt_tr = np.zeros(DIM)*np.NaN
MHW_dur_tr = np.zeros(DIM)*np.NaN
MHW_max_tr = np.zeros(DIM)*np.NaN
MHW_mean_tr = np.zeros(DIM)*np.NaN
MHW_cum_tr = np.zeros(DIM)*np.NaN
# MHW_var_tr = np.zeros(DIM)*np.NaN
MHW_td_tr = np.zeros(DIM)*np.NaN
# MHW_tc_tr = np.zeros(DIM)*np.NaN
DIM2 = (len(j_which), len(i_which), 2)
SST_dtr = np.zeros(DIM2)*np.NaN
MHW_cnt_dtr = np.zeros(DIM2)*np.NaN
MHW_dur_dtr = np.zeros(DIM2)*np.NaN
MHW_max_dtr = np.zeros(DIM2)*np.NaN
MHW_mean_dtr = np.zeros(DIM2)*np.NaN
MHW_cum_dtr = np.zeros(DIM2)*np.NaN
# MHW_var_dtr = np.zeros(DIM2)*np.NaN
MHW_td_dtr = np.zeros(DIM2)*np.NaN
# MHW_tc_dtr = np.zeros(DIM2)*np.NaN
years = 41
N_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
SST_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
MHW_cnt_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
MHW_dur_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
MHW_max_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
MHW_mean_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
MHW_cum_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
# MHW_var_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
MHW_td_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN
# MHW_tc_ts = np.zeros((len(j_which), len(i_which), years))*np.NaN


#Convert 0's to NaN

# MHW_cnt, MHW_cnt_tr, MHW_cnt_dtr = np.where(MHW_cnt == 0, np.NaN, MHW_cnt), np.where(MHW_cnt_tr == 0, np.NaN, MHW_cnt_tr), np.where(MHW_cnt_dtr == 0, np.NaN, MHW_cnt_dtr) 
# MHW_dur, MHW_dur_tr, MHW_dur_dtr = np.where(MHW_dur == 0, np.NaN, MHW_dur), np.where(MHW_dur_tr == 0, np.NaN, MHW_dur_tr), np.where(MHW_dur_dtr == 0, np.NaN, MHW_dur_dtr)
# MHW_max, MHW_max_tr, MHW_max_dtr = np.where(MHW_max == 0, np.NaN, MHW_max), np.where(MHW_max_tr == 0, np.NaN, MHW_max_tr), np.where(MHW_max_dtr == 0, np.NaN, MHW_max_dtr)
# MHW_mean, MHW_mean_tr, MHW_mean_dtr = np.where(MHW_mean == 0, np.NaN, MHW_mean), np.where(MHW_mean_tr == 0, np.NaN, MHW_mean_tr), np.where(MHW_mean_dtr == 0, np.NaN, MHW_mean_dtr)
# MHW_cum, MHW_cum_tr, MHW_cum_dtr = np.where(MHW_cum == 0, np.NaN, MHW_cum), np.where(MHW_cum_tr == 0, np.NaN, MHW_cum_tr), np.where(MHW_cum_dtr == 0, np.NaN, MHW_cum_dtr)
# MHW_var, MHW_var_tr, MHW_var_dtr = np.where(MHW_var == 0, np.NaN, MHW_var), np.where(MHW_var_tr == 0, np.NaN, MHW_var_tr), np.where(MHW_var_dtr == 0, np.NaN, MHW_var_dtr)
# MHW_td, MHW_td_tr, MHW_td_dtr = np.where(MHW_td == 0, np.NaN, MHW_td), np.where(MHW_td_tr == 0, np.NaN, MHW_td_tr), np.where(MHW_td_dtr == 0, np.NaN, MHW_td_dtr) 
# MHW_tc, MHW_tc_tr, MHW_tc_dtr = np.where(MHW_tc == 0, np.NaN, MHW_tc), np.where(MHW_tc_tr == 0, np.NaN, MHW_tc_tr), np.where(MHW_tc_dtr == 0, np.NaN, MHW_tc_dtr)
# SST_mean, SST_tr, SST_dtr = np.where(SST_mean == 0, np.NaN, SST_mean), np.where(SST_tr == 0, np.NaN, SST_tr), np.where(SST_dtr == 0, np.NaN, SST_dtr) 


# Looping through locations lat (j) / lon(i) for each MHW metric
# Loop over i (lon)
for i in range(0, 239):
    print(i, 'of', len(i_which))
    # Loop over j (lat)
    for j in range(0, 180):
        # Loading SST 
        # sst_value = sst[:,j,i]  # Slice SST at (j,i)
        
        
        land_ice =  np.logical_not(np.isfinite((sst[:,j,i]).sum())) #+ ((np.mean(sstSD.data) < 0.2)) #Check for land or ice.
        
        if land_ice == 1:    
        
                
            MHW_cnt[j,i], MHW_cnt_tr[j,i], MHW_cnt_dtr[j,i,:] = np.NaN, np.NaN, np.NaN
            MHW_dur[j,i], MHW_dur_tr[j,i], MHW_dur_dtr[j,i,:] = np.NaN, np.NaN, np.NaN
            MHW_max[j,i], MHW_max_tr[j,i], MHW_max_dtr[j,i,:] = np.NaN, np.NaN, np.NaN
            MHW_mean[j,i], MHW_mean_tr[j,i], MHW_mean_dtr[j,i,:] = np.NaN, np.NaN, np.NaN
            MHW_cum[j,i], MHW_cum_tr[j,i], MHW_cum_dtr[j,i,:] = np.NaN, np.NaN, np.NaN
            # MHW_var[j,i], MHW_var_tr[j,i], MHW_var_dtr[j,i,:] = np.NaN, np.NaN, np.NaN
            MHW_td[j,i], MHW_td_tr[j,i], MHW_td_dtr[j,i,:] = np.NaN, np.NaN, np.NaN
            # MHW_tc[j,i], MHW_tc_tr[j,i], MHW_tc_dtr[j,i,:] = np.NaN, np.NaN, np.NaN 
            SST_mean[j,i], SST_tr[j,i], SST_dtr[j,i,:] = np.NaN, np.NaN, np.NaN
            # MHW_total[j,i] = np.NaN
           
           
            MHW_cnt_ts[j,i,:] = np.NaN
            MHW_dur_ts[j,i,:] = np.NaN
            MHW_max_ts[j,i,:] = np.NaN
            MHW_mean_ts[j,i,:] = np.NaN
            MHW_cum_ts[j,i,:] = np.NaN
            # MHW_var_ts[j,i,:] = np.NaN
            MHW_td_ts[j,i,:] = np.NaN
            # MHW_tc_ts[j,i,:] = np.NaN
            N_ts[j,i,:] = np.NaN
            SST_ts[j,i,:] = np.NaN 
           
           
        
        else:  
      
            #Count number of MHW metrics of each length
            mhws, clim = mhw.detect(t, temp=sst[:,j,i], climatologyPeriod=climatologyPeriod, pctile=pctile)
            mhwBlock = mhw.blockAverage(t, mhws, clim=clim, temp=sst[:,j,i])
            mean, trend, dtrend = mhw.meanTrend(mhwBlock)
                            
            # #Total count
            # MHW_total[j, i] = mhwBlock['count'].sum()
                            
            # #MHW metrics (mean, trend, dtrend)
            MHW_cnt[j,i], MHW_cnt_tr[j,i] = mean['count'], trend['count']
            MHW_dur[j,i], MHW_dur_tr[j,i] = mean['duration'], trend['duration']
            MHW_max[j,i], MHW_max_tr[j,i] = mean['intensity_max_max'], trend['intensity_max_max']
            MHW_mean[j,i], MHW_mean_tr[j,i] = mean['intensity_mean'], trend['intensity_mean']
            MHW_cum[j,i], MHW_cum_tr[j,i] = mean['intensity_cumulative'], trend['intensity_cumulative']
            # MHW_var[j,i], MHW_var_tr[j,i] = mean['intensity_var'], trend['intensity_var']
            MHW_td[j,i], MHW_td_tr[j,i] = mean['total_days'], trend['total_days']
            # MHW_tc[j,i], MHW_tc_tr[j,i] = mean['total_icum'], trend['total_icum']
            SST_mean[j,i], SST_tr[j,i] = mean['temp_mean'], trend['temp_mean']
                  
             
            # Time series
            MHW_cnt_ts[j,i,:] += mhwBlock['count']
            MHW_dur_ts[j,i,np.where(~np.isnan(mhwBlock['duration']))[0]] = mhwBlock['duration'][np.where(~np.isnan(mhwBlock['duration']))[0]]
            MHW_max_ts[j,i,np.where(~np.isnan(mhwBlock['intensity_max_max']))[0]] = mhwBlock['intensity_max_max'][np.where(~np.isnan(mhwBlock['intensity_max_max']))[0]]
            MHW_mean_ts[j,i,np.where(~np.isnan(mhwBlock['intensity_mean']))[0]] = mhwBlock['intensity_mean'][np.where(~np.isnan(mhwBlock['intensity_mean']))[0]]
            MHW_cum_ts[j,i,np.where(~np.isnan(mhwBlock['intensity_cumulative']))[0]] = mhwBlock['intensity_cumulative'][np.where(~np.isnan(mhwBlock['intensity_cumulative']))[0]]
            # MHW_var_ts[j,i,np.where(~np.isnan(mhwBlock['intensity_var']))[0]] = mhwBlock['intensity_var'][np.where(~np.isnan(mhwBlock['intensity_var']))[0]]
            MHW_td_ts[j,i,np.where(~np.isnan(mhwBlock['total_days']))[0]] = mhwBlock['total_days'][np.where(~np.isnan(mhwBlock['total_days']))[0]]
            # MHW_tc_ts[j,i,np.where(~np.isnan(mhwBlock['total_icum']))[0]] = mhwBlock['total_icum'][np.where(~np.isnan(mhwBlock['total_icum']))[0]]
            # N_ts[j,i,:] += (~np.isnan(mhwBlock['duration'])).astype(int)
            SST_ts[j,i,:] = mhwBlock['temp_mean']  
        
 ##############################################################################           



            # Save data so far
            outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\MHW_metrics'
            np.savez(outfile, lon=lon, lat=lat, SST_mean=SST_mean, MHW_cnt=MHW_cnt, MHW_dur=MHW_dur, MHW_max=MHW_max, MHW_mean=MHW_mean, MHW_cum=MHW_cum, MHW_td=MHW_td, SST_tr=SST_tr, MHW_cnt_tr=MHW_cnt_tr, MHW_dur_tr=MHW_dur_tr, MHW_max_tr=MHW_max_tr, MHW_mean_tr=MHW_mean_tr, MHW_cum_tr=MHW_cum_tr, MHW_td_tr=MHW_td_tr, SST_dtr=SST_dtr, MHW_cnt_dtr=MHW_cnt_dtr, MHW_dur_dtr=MHW_dur_dtr, MHW_max_dtr=MHW_max_dtr, MHW_mean_dtr=MHW_mean_dtr, MHW_cum_dtr=MHW_cum_dtr, MHW_td_dtr=MHW_td_dtr, SST_ts=SST_ts, MHW_cnt_ts=MHW_cnt_ts, MHW_dur_ts=MHW_dur_ts, MHW_max_ts=MHW_max_ts, MHW_mean_ts=MHW_mean_ts, MHW_cum_ts=MHW_cum_ts, MHW_td_ts=MHW_td_ts, N_ts=N_ts)

