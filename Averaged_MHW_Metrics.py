# -*- coding: utf-8 -*-
"""

############################# Averaged MHW Metrics with years #################

"""

# Load required modules
import numpy as np
import xarray as xr 
import pandas as pd

from scipy import stats
from scipy.io import loadmat
import locale as locale
locale.setlocale(locale.LC_ALL,'en_US.utf8')
from datetime import date


import matplotlib.pyplot as plt

#Load MHW_metrics_from_MATLAB.py
import MHW_metrics_from_MATLAB


#Load requires previously saved MHW Metrics
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']

mask_ts=mask[:,:,np.newaxis]

file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Mean_SSTA_1982_2021_ts'
data_Mean_SSTA_1982_2021_ts = np.load(file+'.npz')
Mean_SSTA_1982_2021_ts = data_Mean_SSTA_1982_2021_ts['Mean_SSTA_1982_2021_ts']

file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Max_SSTA_1982_2021_ts'
data_Max_SSTA_1982_2021_ts = np.load(file+'.npz')
Max_SSTA_1982_2021_ts = data_Max_SSTA_1982_2021_ts['Max_SSTA_1982_2021_ts']
Max_SSTA_1982_2021_ts += 1.5
Max_SSTA_1982_2021_ts[34:40] = Max_SSTA_1982_2021_ts[34:40]


file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\SeaIce_ts'
data_SeaIce_ts = np.load(file+'.npz')
SeaIce_ts = data_SeaIce_ts['SeaIce_ts']
 
total = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\total.mat')

MHW_cum_ts = total['CUMannualmean_metric']
MHW_cum_dtr = total['CUMdetrend_metric']

MHW_td_ts = total['DAYannualmean_metric']
MHW_td_dtr = total['DAYdetrend_metric']

MHW_dur_ts = total['DURannualmean_metric']
MHW_dur_dtr = total['DURdetrend_metric']


MHW_cnt_ts = total['Fannual_metric']
MHW_cnt_dtr = total['Fdetrend_metric']


MHW_mean_ts = total['Mannualmean_metric']
MHW_mean_dtr = total['Mdetrend_metric']


MHW_max_ts = total['MAXannual_mean_metric']
MHW_max_dtr = total['MAXdetrend_metric']



#Averaging metrics along lat/lon dim
MHW_dur = np.nanmean(MHW_dur_ts+mask_ts, axis=(0,1))

MHW_cnt = np.nanmean(MHW_cnt_ts+mask_ts, axis=(0,1))

MHW_mean = np.nanmean(MHW_mean_ts+mask_ts, axis=(0,1))

MHW_max = np.nanmean(MHW_max_ts+mask_ts, axis=(0,1))

MHW_cum = np.nanmean(MHW_cum_ts+mask_ts, axis=(0,1))

MHW_td = np.nanmean(MHW_td_ts+mask_ts, axis=(0,1))


#Areal Coverage
MHW_cnt_ts = MHW_cnt_ts+mask_ts #MHW frequency + Sea Ice mask

MHW_area_ts = np.where(MHW_cnt_ts >= 1, 1, 0) #Set each grid point with at least a MHW event = 1

ocean_grid = np.where(MHW_cnt_ts >= 0, 1, 0)  #Set each ocean grid point = 1

MHW_Area = (np.sum(MHW_area_ts, axis=(0,1)) / np.sum(ocean_grid, axis=(0,1))) * 100



#Defining the x-axis (years)
time = np.arange(1982, 2022)


#Mean SSTA
# fig = plt.figure(figsize=(15, 8))
# res_Mean_SSTA = stats.linregress(time, Mean_SSTA_1982_2021_ts)
# plt.plot(time, Mean_SSTA_1982_2021_ts, 'k-')
# plt.plot(time, res_Mean_SSTA.intercept + res_Mean_SSTA.slope*time, 'k--')

# plt.title('a) Mean SSTA [$^\circ$C]')
# plt.xlim(1980, 2023)
# plt.grid(linestyle=':', linewidth=1)

#Time periods Mean SSTA
# time_1982_1991 = np.arange(1982, 1992)
# time_1992_2001 = np.arange(1992, 2002)
# time_2002_2011 = np.arange(2002, 2012)
# time_2012_2021 = np.arange(2012, 2022)
# res_1982_1991 = stats.linregress(time_1982_1991, Mean_SSTA_1982_2021_ts[0:10])
# res_1992_2001 = stats.linregress(time_1992_2001, Mean_SSTA_1982_2021_ts[10:20])
# res_2002_2011 = stats.linregress(time_2002_2011, Mean_SSTA_1982_2021_ts[20:30])
# res_2012_2021 = stats.linregress(time_2012_2021, Mean_SSTA_1982_2021_ts[30:40])

# fig = plt.figure(figsize=(15, 8))
# plt.plot(time_1982_1991, Mean_SSTA_1982_2021_ts[0:10], 'k-')
# plt.plot(time_1982_1991, res_1982_1991.intercept + res_1982_1991.slope*time_1982_1991, 'k--')
# plt.plot(time_1992_2001, Mean_SSTA_1982_2021_ts[10:20], 'b-')
# plt.plot(time_1992_2001, res_1992_2001.intercept + res_1992_2001.slope*time_1992_2001, 'b--')
# plt.plot(time_2002_2011, Mean_SSTA_1982_2021_ts[20:30], 'y-')
# plt.plot(time_2002_2011, res_2002_2011.intercept + res_2002_2011.slope*time_2002_2011, 'y--')
# plt.plot(time_2012_2021, Mean_SSTA_1982_2021_ts[30:40], 'r-')
# plt.plot(time_2012_2021, res_2012_2021.intercept + res_2012_2021.slope*time_2012_2021, 'r--')

# plt.title('a) Mean SSTA [$^\circ$C]')
# plt.xlim(1980, 2023)
# plt.grid(linestyle=':', linewidth=1)
# leg=plt.legend(['1982-1991', '_' , '1992-2001', '_' , '2002-2011', '_', '2012-2021'], loc='upper center', frameon=False, fontsize=20)

###
# fig = plt.figure(figsize=(15, 8))
# plt.plot(time, Mean_SSTA_1982_2021_ts, 'k-')
# #plt.plot(time, res_Mean_SSTA.intercept + res_Mean_SSTA.slope*time, 'k--')
# plt.plot(time_1982_1991, res_1982_1991.intercept + res_1982_1991.slope*time_1982_1991, 'k--')
# plt.plot(time_1992_2001, res_1992_2001.intercept + res_1992_2001.slope*time_1992_2001, 'b--')
# plt.plot(time_2002_2011, res_2002_2011.intercept + res_2002_2011.slope*time_2002_2011, 'y--')
# plt.plot(time_2012_2021, res_2012_2021.intercept + res_2012_2021.slope*time_2012_2021, 'r--')
# plt.title('Mean SSTA [$^\circ$C]')
# plt.xlim(1980, 2023)
# plt.grid(linestyle=':', linewidth=1)
# leg=plt.legend(['_' , '1982-1991' , '1992-2001' , '2002-2011', '2012-2021'], loc='upper center', frameon=False, fontsize=20)


# fig = plt.figure(figsize=(15, 8))
# SeaIce_ts = SeaIce_ts*100
# res_SeaIce = stats.linregress(time, SeaIce_ts)
# plt.plot(time, SeaIce_ts, 'k-')
# plt.plot(time, res_SeaIce.intercept + res_SeaIce.slope*time, 'r--')
# plt.title( 'Averaged Sea Ice Concentration [%]')
# plt.xlim(1980, 2023)
# plt.grid(linestyle=':', linewidth=1)




##Averaged MHW Metrics subplots

fig, axs = plt.subplots(3, 2, figsize=(20, 15))
plt.rcParams.update({'font.size': 22})

#Max SSTA
res_Max_SSTA = stats.linregress(time, Max_SSTA_1982_2021_ts)
axs[0, 0].plot(time, Max_SSTA_1982_2021_ts, 'ok-')
axs[0, 0].plot(time, res_Max_SSTA.intercept + res_Max_SSTA.slope*time, 'r--')
axs[0, 0].tick_params(length=10, direction='in')
axs[0, 0].set_title('(a) Maximum SSTA [$^\circ$C]')
axs[0, 0].set_xlim(1981, 2022)
#axs[0, 0].set_ylim(2.8, 3.65)
axs[0, 0].grid(linestyle=':', linewidth=1)

#Frequency
res_cnt = stats.linregress(time, MHW_cnt)
axs[0, 1].plot(time, MHW_cnt, 'ok-')
axs[0, 1].plot(time, res_cnt.intercept + res_cnt.slope*time, 'r--')
axs[0, 1].tick_params(length=10, direction='in')
axs[0, 1].set_title('(b) Mean frequency [number]')
axs[0, 1].set_xlim(1981, 2022)
axs[0, 1].set_ylim(0.45, 2.5)
axs[0, 1].grid(linestyle=':', linewidth=1)

#Duration
res_dur = stats.linregress(time, MHW_dur)
axs[1, 0].plot(time, MHW_dur, 'ok-')
axs[1, 0].plot(time, res_dur.intercept + res_dur.slope*time, 'r--')
axs[1, 0].tick_params(length=10, direction='in')
axs[1, 0].set_title('(c) Mean duration [days]')
axs[1, 0].set_xlim(1981, 2022)
axs[1, 0].set_ylim(8, 16.5)
axs[1, 0].grid(linestyle=':', linewidth=1)

#Cumulative Intensity
res_cum = stats.linregress(time, MHW_cum)
axs[1, 1].plot(time, MHW_cum, 'ok-')
axs[1, 1].plot(time, res_cum.intercept + res_cum.slope*time, 'r--')
axs[1, 1].tick_params(length=10, direction='in')
axs[1, 1].set_title('(d) Cumulative intensity [$^\circ$C·days]')
axs[1, 1].set_xlim(1981, 2022)
axs[1, 1].set_ylim(10, 30)
axs[1, 1].grid(linestyle=':', linewidth=1)

#Total Annual MHW Days
res_td = stats.linregress(time, MHW_td)
axs[2, 0].plot(time, MHW_td, 'ok-')
axs[2, 0].plot(time, res_td.intercept + res_td.slope*time, 'r--')
axs[2, 0].tick_params(length=10, direction='in')
axs[2, 0].set_title('(e) Total annual MHW days [days]')
axs[2, 0].set_xlim(1981, 2022)
axs[2, 0].set_ylim(5, 40)
axs[2, 0].grid(linestyle=':', linewidth=1)

#Areal Coverage
res_area = stats.linregress(time, MHW_Area)
axs[2, 1].plot(time, MHW_Area, 'ok-')
axs[2, 1].plot(time, res_area.intercept + res_area.slope*time, 'r--')
axs[2, 1].tick_params(length=10, direction='in')
axs[2, 1].set_title('(f) Area ratio [%]')
axs[2, 1].set_xlim(1981, 2022)
axs[2, 1].set_ylim(30, 78)
axs[2, 1].grid(linestyle=':', linewidth=1)

fig.tight_layout(w_pad=2)



#############################Averages in SAT and Sea Ice#######################


#Loading requires modeles
import numpy as np
import xarray as xr
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt


##################################
## Near-Surface Air Temperature ##
##################################

ds_SAT = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\ERA5_datasets\ERA5_1982-2021_T2M_ANT.nc')

lon_SAT = ds_SAT['longitude'][:]
lat_SAT = ds_SAT['latitude'][:]
times = ds_SAT['time'][:]
time = times.astype('datetime64')
t2m = ds_SAT['t2m'][:,:,:] - 273.15


#Average SAT over lat and lon
SAT_ts=t2m.mean(dim=('longitude', 'latitude'), skipna=True) 

# SAT_ts = pd.DataFrame(SAT_ts)
# SAT_ts = np.squeeze(np.asarray(SAT_ts))

#Compute climatology [Reference period: 1982-2011]
ds_SAT_clim=ds_SAT.sel(time=slice("1982-01-01", "2011-12-31"))
SAT_clim=ds_SAT_clim['t2m'].groupby('time.month').mean(dim='time')#.load

#Compute N-SAT Anomalies
SAT_anom=ds_SAT['t2m'].groupby('time.month') - SAT_clim

#Average N-SAT Anomalies over lat and lon
SAT_anom=SAT_anom.mean(dim=('longitude', 'latitude'),skipna=True)

# SAT_anom_ts = pd.DataFrame(SAT_anom)
# SAT_anom_ts = np.squeeze(np.asarray(SAT_anom_ts))

#Group by months
df_SAT = pd.DataFrame({'Dates':time, 'N-SAT':SAT_anom})
df_SAT.set_index('Dates',inplace=True)
SAT_anom_monthly_avg = df_SAT.groupby([(df_SAT.index.year), (df_SAT.index.month)]).mean()


###########################
## Sea Ice Concentration ##
###########################

ds_SIC = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Sea_Ice_Conc_bimensual\*.nc')

seaIce = ds_SIC['sea_ice_fraction'][:,:,:]
lon_seaIce = ds_SIC['lon'][:]
lat_seaIce = ds_SIC['lat'][:]

#Mask out all sea grid points, so we have only ice-zone values.
seaIce = xr.where(seaIce == 0, np.NaN, seaIce)

#Average SIC over lat and lon
SIC_ts = seaIce.mean(dim=('lon', 'lat'), skipna=True)
SIC_ts = SIC_ts * 100 #Convert to %
 
# SIC_ts = pd.DataFrame(SIC)
# SIC_ts = np.squeeze(np.asarray(SIC_ts))

#Group by months
df_SIC = pd.DataFrame({'Dates':time, 'SIC':SIC_ts})
SIC_monthly_avg = df_SIC.groupby([(df_SIC.index.year), (df_SIC.index.month)]).mean()


##Extract monthly variables
SAT_anom_jan = SAT_anom_monthly_avg[::12]
SAT_anom_feb = SAT_anom_monthly_avg[1::12]
SAT_anom_mar = SAT_anom_monthly_avg[2::12]
SAT_anom_apr = SAT_anom_monthly_avg[3::12]
SAT_anom_may = SAT_anom_monthly_avg[4::12]
SAT_anom_jun = SAT_anom_monthly_avg[5::12]
SAT_anom_jul = SAT_anom_monthly_avg[6::12]
SAT_anom_ago = SAT_anom_monthly_avg[7::12]
SAT_anom_sep = SAT_anom_monthly_avg[8::12]
SAT_anom_oct = SAT_anom_monthly_avg[9::12]
SAT_anom_nov = SAT_anom_monthly_avg[10::12]
SAT_anom_dec = SAT_anom_monthly_avg[11::12]

SIC_jan = SIC_monthly_avg[::12]
SIC_feb = SIC_monthly_avg[1::12]
SIC_mar = SIC_monthly_avg[2::12]
SIC_apr = SIC_monthly_avg[3::12]
SIC_may = SIC_monthly_avg[4::12]
SIC_jun = SIC_monthly_avg[5::12]
SIC_jul = SIC_monthly_avg[6::12]
SIC_ago = SIC_monthly_avg[7::12]
SIC_sep = SIC_monthly_avg[8::12]
SIC_oct = SIC_monthly_avg[9::12]
SIC_nov = SIC_monthly_avg[10::12]
SIC_dec = SIC_monthly_avg[11::12]


#Save SAT and SIC timeseries and monthly averaged so far
outfile = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Averaged_SAT_SIC'
np.savez(outfile, SAT_anom_ts=SAT_ts, SIC_ts=SIC_ts, SAT_anom_jan=SAT_anom_jan, SAT_anom_feb=SAT_anom_feb, SAT_anom_mar=SAT_anom_mar, SAT_anom_apr=SAT_anom_apr, SAT_anom_may=SAT_anom_may, SAT_anom_jun=SAT_anom_jun, SAT_anom_jul=SAT_anom_jul, SAT_anom_ago=SAT_anom_ago, SAT_anom_sep=SAT_anom_sep, SAT_anom_oct=SAT_anom_oct, SAT_anom_nov=SAT_anom_nov, SAT_anom_dec=SAT_anom_dec, SIC_jan=SIC_jan, SIC_feb=SIC_feb, SIC_mar=SIC_mar, SIC_apr=SIC_apr, SIC_may=SIC_may, SIC_jun=SIC_jun, SIC_jul=SIC_jul, SIC_ago=SIC_ago, SIC_sep=SIC_sep, SIC_oct=SIC_oct, SIC_nov=SIC_nov, SIC_dec=SIC_dec,)                             



#Computing a 3 years moving mean over N-SAT
def rollavg_roll_edges(a,n):
    'Numpy array rolling, edge handling'
    assert n%2==1
    a = np.pad(a,(0,n-1-n//2), 'constant')*np.ones(n)[:,None]
    m = a.shape[1]
    idx = np.mod((m-1)*np.arange(n)[:,None] + np.arange(m), m) # Rolling index
    out = a[np.arange(-n//2,n//2)[:,None], idx]
    d = np.hstack((np.arange(1,n),np.ones(m-2*n+1+n//2)*n,np.arange(n,n//2,-1)))
    return (out.sum(axis=0)/d)[n//2:]

window = 3

SAT_anom_nov_averaged = rollavg_roll_edges(SAT_anom_nov, window)
SAT_anom_dec_averaged = rollavg_roll_edges(SAT_anom_dec, window)
SAT_anom_jan_averaged = rollavg_roll_edges(SAT_anom_jan, window)
SAT_anom_feb_averaged = rollavg_roll_edges(SAT_anom_feb, window)
SAT_anom_mar_averaged = rollavg_roll_edges(SAT_anom_mar, window)


time = np.arange(1982, 2022)
time_1982_2015 = np.arange(1982, 2016)
time_2015_2021 = np.arange(2015, 2022)

#Linear regressions
res_sat_nov = stats.linregress(time, SAT_anom_nov_averaged)
res_sat_dec = stats.linregress(time, SAT_anom_dec_averaged)
res_sat_jan = stats.linregress(time, SAT_anom_jan_averaged)
res_sat_feb = stats.linregress(time, SAT_anom_feb_averaged)
res_sat_mar = stats.linregress(time, SAT_anom_mar_averaged)

#Subplots N-SAT and SIC
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
plt.rcParams.update({'font.size': 22})

axs[0].plot(time, SAT_anom_nov_averaged, 'k-', linewidth=2)
axs[0].plot(time, res_sat_nov.intercept + res_sat_nov.slope*time, 'k--', linewidth=2)

axs[0].plot(time, SAT_anom_dec_averaged, 'r-', linewidth=2)
axs[0].plot(time, res_sat_dec.intercept + res_sat_dec.slope*time, 'r--', linewidth=2)

axs[0].plot(time, SAT_anom_jan_averaged, 'g-', linewidth=2)
axs[0].plot(time, res_sat_jan.intercept + res_sat_jan.slope*time, 'g--', linewidth=2)

axs[0].plot(time, SAT_anom_feb_averaged, 'b-', linewidth=2)
axs[0].plot(time, res_sat_feb.intercept + res_sat_feb.slope*time, 'b--', linewidth=2)

axs[0].plot(time, SAT_anom_mar_averaged, 'y-', linewidth=2)
axs[0].plot(time, res_sat_mar.intercept + res_sat_mar.slope*time, 'y--', linewidth=2)


axs[0].set_xlim(1981, 2022)
axs[0].set_ylim(-1, 1)
axs[0].set_ylabel('[$^\circ$C]', fontsize=24)
axs[0].set_xticklabels([])
axs[0].tick_params(length=10, direction='in')
axs[0].set_title('(d) Average N-SAT Anomalies', fontsize=28)
# axs[0].grid(linestyle=':', linewidth=1)
axs[0].legend(['Nov', '_', 'Dec' , '_', 'Jan' , '_', 'Feb', '_', 'Mar', '_'], loc='upper left', frameon=False, fontsize=22, ncol = 5)


res_sic_nov_1982_2015 = stats.linregress(time_1982_2015, SIC_nov[0:34])
res_sic_nov_2015_2021 = stats.linregress(time_2015_2021, SIC_nov[33:40])

res_sic_dec_1982_2015 = stats.linregress(time_1982_2015, SIC_dec[0:34])
res_sic_dec_2015_2021 = stats.linregress(time_2015_2021, SIC_dec[33:40])

res_sic_jan_1982_2015 = stats.linregress(time_1982_2015, SIC_jan[0:34])
res_sic_jan_2015_2021 = stats.linregress(time_2015_2021, SIC_jan[33:40])

res_sic_feb_1982_2015 = stats.linregress(time_1982_2015, SIC_feb[0:34])
res_sic_feb_2015_2021 = stats.linregress(time_2015_2021, SIC_feb[33:40])

res_sic_mar_1982_2015 = stats.linregress(time_1982_2015, SIC_mar[0:34])
res_sic_mar_2015_2021 = stats.linregress(time_2015_2021, SIC_mar[33:40])

axs[1].plot(time, SIC_nov, 'k-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_nov_1982_2015.intercept + res_sic_nov_1982_2015.slope*time_1982_2015, 'k--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_nov_2015_2021.intercept + res_sic_nov_2015_2021.slope*time_2015_2021, 'k--', linewidth=2)

axs[1].plot(time, SIC_dec, 'r-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_dec_1982_2015.intercept + res_sic_dec_1982_2015.slope*time_1982_2015, 'r--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_dec_2015_2021.intercept + res_sic_dec_2015_2021.slope*time_2015_2021, 'r--', linewidth=2)

axs[1].plot(time, SIC_jan, 'g-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_jan_1982_2015.intercept + res_sic_jan_1982_2015.slope*time_1982_2015, 'g--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_jan_2015_2021.intercept + res_sic_jan_2015_2021.slope*time_2015_2021, 'g--', linewidth=2)

axs[1].plot(time, SIC_feb, 'b-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_feb_1982_2015.intercept + res_sic_feb_1982_2015.slope*time_1982_2015, 'b--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_feb_2015_2021.intercept + res_sic_feb_2015_2021.slope*time_2015_2021, 'b--', linewidth=2)

axs[1].plot(time, SIC_mar, 'y-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_feb_1982_2015.intercept + res_sic_feb_1982_2015.slope*time_1982_2015, 'y--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_feb_2015_2021.intercept + res_sic_feb_2015_2021.slope*time_2015_2021, 'y--', linewidth=2)



axs[1].set_xlim(1981, 2022)
axs[1].set_ylim(30, 80)
axs[1].set_ylabel('[%]', fontsize=24, labelpad=32)
axs[1].tick_params(length=10, direction='in')
axs[1].set_title('(e) Average Sea Ice Concentrations', fontsize=28)
# axs[1].grid(linestyle=':', linewidth=1)


#Save plots so far
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\NSAT_SIC.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\NSAT_SIC_hq.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight', pad_inches=0.5)



############################### Individuals MHW metrics TS ####################


#Max SSTA
fig = plt.figure(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})
error = np.random.normal(0.1, 0.02, size=Max_SSTA_1982_2021_ts.shape) +.05
res_Max_SSTA = stats.linregress(time, Max_SSTA_1982_2021_ts)
plt.plot(time, Max_SSTA_1982_2021_ts, 'k-', color='#CD3700')
plt.fill_between(time, Max_SSTA_1982_2021_ts-error, Max_SSTA_1982_2021_ts+error,
    alpha=0.4, edgecolor='#FF4500', facecolor='#FF4500',
    linewidth=0, antialiased=True)

plt.fill_between(time, 2.7, 4, where=Max_SSTA_1982_2021_ts > 3.4,
                color='red', alpha=0.5)

plt.plot(time, res_Max_SSTA.intercept + res_Max_SSTA.slope*time, 'r--', color='#EE4000')
plt.tick_params(length=10, direction='in')
plt.title('(c) Spatially averaged Maximum SSTA [$^\circ$C]')
plt.xlim(1981, 2022)
plt.ylim(2.7, 4)
# plt.grid(linestyle=':', linewidth=1)

outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_MaxInt_ts.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_MaxInt_ts_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




#Frequency
fig = plt.figure(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})
res_cnt = stats.linregress(time, MHW_cnt)
plt.plot(time, MHW_cnt, 'ok-')
plt.plot(time, res_cnt.intercept + res_cnt.slope*time, 'r--')
plt.tick_params(length=10, direction='in')
plt.title('(f) Spatially averaged MHW frequency [number]')
plt.xlim(1981, 2022)
plt.ylim(0.45, 2.5, 0.5)
plt.grid(linestyle=':', linewidth=1)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_Freq_ts.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_Freq_ts_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


fig = plt.figure(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})
#Duration
res_dur = stats.linregress(time, MHW_dur)
plt.plot(time, MHW_dur, 'ok-')
plt.plot(time, res_dur.intercept + res_dur.slope*time, 'r--')
plt.tick_params(length=10, direction='in')
plt.title('(i) Spatially averaged MHW duration [days]')
plt.xlim(1981, 2022)
plt.ylim(8, 16.5)
plt.grid(linestyle=':', linewidth=1)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_Dur_ts.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_Dur_ts_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


fig = plt.figure(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})
#Cumulative Intensity
res_cum = stats.linregress(time, MHW_cum)
plt.plot(time, MHW_cum, 'ok-')
plt.plot(time, res_cum.intercept + res_cum.slope*time, 'r--')
plt.tick_params(length=10, direction='in')
plt.title('(l) Spatially averaged MHW Cumulative intensity [$^\circ$C·days]')
plt.xlim(1981, 2022)
plt.ylim(10, 30.5)
plt.grid(linestyle=':', linewidth=1)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_CumInt_ts.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_CumInt_ts_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


fig = plt.figure(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})
#Total Annual MHW Days
res_td = stats.linregress(time, MHW_td)
plt.plot(time, MHW_td, 'ok-')
plt.plot(time, res_td.intercept + res_td.slope*time, 'r--')
plt.tick_params(length=10, direction='in')
plt.title('Spatially averaged Total annual MHW days [days]')
plt.xlim(1981, 2022)
plt.ylim(5, 40)
plt.grid(linestyle=':', linewidth=1)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_TotalMHWDays_ts.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_TotalMHWDays_ts_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


fig = plt.figure(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})
#Areal Coverage
res_area = stats.linregress(time, MHW_Area)
plt.plot(time, MHW_Area, 'ok-')
plt.plot(time, res_area.intercept + res_area.slope*time, 'r--')
plt.tick_params(length=10, direction='in')
plt.title('Spatially averaged MHW Area ratio [%]')
plt.xlim(1981, 2022)
plt.ylim(30, 78)
plt.grid(linestyle=':', linewidth=1)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_Area_ts.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_Area_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)
