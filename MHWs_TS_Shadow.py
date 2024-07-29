# -*- coding: utf-8 -*-
"""
######################### Spatially Averaged MHW Metrics TS ###################
"""

#Loading required python modules
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
from scipy import stats


#Load sea-ice mask
file = r'.\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
mask_ts=mask[:,:,np.newaxis]

time = np.arange(1982, 2022)
###Maximum SSTA###
#Global
file = r'.\Max_SSTA_1982_2021_ts'
data_Max_SSTA_1982_2021_ts = np.load(file+'.npz')
Max_SSTA_global = data_Max_SSTA_1982_2021_ts['Max_SSTA_1982_2021_ts']
file = r'.\Max_SSTA_sd'
data_Max_SSTA_sd = np.load(file+'.npz')
Max_SSTA_sd_global = data_Max_SSTA_sd['Max_SSTA_sd']
error_Max_SSTA = Max_SSTA_sd_global/np.sqrt(40)
#PAC
file = r'.\PAC_Max_SSTA_ts'
PAC_Max_SSTA_ts = np.load(file+'.npy')
#ATL
file = r'.\ATL_Max_SSTA_ts'
ATL_Max_SSTA_ts = np.load(file+'.npy')
#IND
file = r'.\IND_Max_SSTA_ts'
IND_Max_SSTA_ts = np.load(file+'.npy')


fig, axs = plt.subplots(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})

res_Max_SSTA = stats.linregress(time, Max_SSTA_global)
axs.plot(time, Max_SSTA_global, '-', color='black', linewidth=2, label='Circumpolar')
axs.plot(time, res_Max_SSTA.intercept + res_Max_SSTA.slope*time, '--', color='black')
axs.fill_between(time, Max_SSTA_global-error_Max_SSTA, Max_SSTA_global+error_Max_SSTA,
    alpha=0.2, edgecolor='black', facecolor='black',
    linewidth=0, antialiased=True)
axs.tick_params(length=10, direction='in')

axs.plot(time, PAC_Max_SSTA_ts, '-', color='peru', alpha=1, label='Pacific')

axs.plot(time, ATL_Max_SSTA_ts, '-', color='red', alpha=1, label='Atlantic')

axs.plot(time, IND_Max_SSTA_ts, '-', color='gold', alpha=1, label='Indian')

axs.set_ylim(2.4, 4.2)
axs.set_xlim(1981, 2022)
axs.set_title('c) Spatially averaged Maximum SSTA [$^\circ$C]')

fig.legend(loc=(0.35, 0.6), frameon=False, fontsize=15)

outfile = r'.\MHW_MaxInt_ts.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





total = loadmat(r'.\total.mat')

###Frequency###
MHW_cnt_ts = total['Fannual_metric']
MHW_cnt_ts = MHW_cnt_ts + mask_ts
#Global
MHW_cnt_global = np.nanmean(MHW_cnt_ts, axis=(0,1))
MHW_cnt_sd = np.nanstd(MHW_cnt_ts, axis=(0,1))
error_MHW_cnt = MHW_cnt_sd/np.sqrt(40)
#PAC
PAC_1_MHW_cnt = np.nanmean(MHW_cnt_ts[660:720,:,:], axis=(0,1))
PAC_2_MHW_cnt = np.nanmean(MHW_cnt_ts[0:60,:,:], axis=(0,1))
PAC_3_MHW_cnt = np.nanmean(MHW_cnt_ts[60:120,:,:], axis=(0,1))
PAC_4_MHW_cnt = np.nanmean(MHW_cnt_ts[120:180,:,:], axis=(0,1))
PAC_5_MHW_cnt = np.nanmean(MHW_cnt_ts[180:220,:,:], axis=(0,1))
PAC_MHW_cnt = (PAC_1_MHW_cnt + PAC_2_MHW_cnt + PAC_3_MHW_cnt + PAC_4_MHW_cnt + PAC_5_MHW_cnt)/5
del PAC_1_MHW_cnt, PAC_2_MHW_cnt, PAC_3_MHW_cnt, PAC_4_MHW_cnt, PAC_5_MHW_cnt
#ATL
ATL_MHW_cnt = np.nanmean(MHW_cnt_ts[219:401,:,:], axis=(0,1))
#IND
IND_MHW_cnt = np.nanmean(MHW_cnt_ts[400:661,:,:], axis=(0,1))


fig, axs = plt.subplots(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})

res_MHW_cnt = stats.linregress(time, MHW_cnt_global)
axs.plot(time, MHW_cnt_global, '-', color='black', linewidth=2, label='Globally')
axs.plot(time, res_MHW_cnt.intercept + res_MHW_cnt.slope*time, '--', color='black')
axs.fill_between(time, MHW_cnt_global-error_MHW_cnt, MHW_cnt_global+error_MHW_cnt,
    alpha=0.2, edgecolor='black', facecolor='black',
    linewidth=0, antialiased=True)
axs.tick_params(length=10, direction='in')

axs.plot(time, PAC_MHW_cnt, '-', color='peru', alpha=1, label='Pacific')

axs.plot(time, ATL_MHW_cnt, '-', color='red', alpha=1, label='Atlantic')

axs.plot(time, IND_MHW_cnt, '-', color='gold', alpha=1, label='Indian')

axs.set_ylim(0, 4.15)
axs.set_xlim(1981, 2022)
axs.set_title('f) Spatially averaged MHW frequency [number]')

outfile = r'.\MHW_Freq_ts.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




###Duration###
MHW_dur_ts = total['DURannualmean_metric']
MHW_dur_ts = MHW_dur_ts + mask_ts
#Global
MHW_dur_global = np.nanmean(MHW_dur_ts, axis=(0,1))
MHW_dur_sd = np.nanstd(MHW_dur_ts, axis=(0,1))
error_MHW_dur = MHW_dur_sd/np.sqrt(40)
#PAC
PAC_1_MHW_dur = np.nanmean(MHW_dur_ts[660:720,:,:], axis=(0,1))
PAC_2_MHW_dur = np.nanmean(MHW_dur_ts[0:60,:,:], axis=(0,1))
PAC_3_MHW_dur = np.nanmean(MHW_dur_ts[60:120,:,:], axis=(0,1))
PAC_4_MHW_dur = np.nanmean(MHW_dur_ts[120:180,:,:], axis=(0,1))
PAC_5_MHW_dur = np.nanmean(MHW_dur_ts[180:220,:,:], axis=(0,1))
PAC_MHW_dur = (PAC_1_MHW_dur + PAC_2_MHW_dur + PAC_3_MHW_dur + PAC_4_MHW_dur + PAC_5_MHW_dur)/5
del PAC_1_MHW_dur, PAC_2_MHW_dur, PAC_3_MHW_dur, PAC_4_MHW_dur, PAC_5_MHW_dur
#ATL
ATL_MHW_dur = np.nanmean(MHW_dur_ts[219:401,:,:], axis=(0,1))
#IND
IND_MHW_dur = np.nanmean(MHW_dur_ts[400:661,:,:], axis=(0,1))


fig, axs = plt.subplots(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})

res_MHW_dur = stats.linregress(time, MHW_dur_global)
axs.plot(time, MHW_dur_global, '-', color='black', linewidth=2, label='Globally')
axs.plot(time, res_MHW_dur.intercept + res_MHW_dur.slope*time, '--', color='black')
axs.fill_between(time, MHW_dur_global-error_MHW_dur, MHW_dur_global+error_MHW_dur,
    alpha=0.2, edgecolor='black', facecolor='black',
    linewidth=0, antialiased=True)
axs.tick_params(length=10, direction='in')

axs.plot(time, PAC_MHW_dur, '-', color='peru', alpha=1, label='Pacific')

axs.plot(time, ATL_MHW_dur, '-', color='red', alpha=1, label='Atlantic')

axs.plot(time, IND_MHW_dur, '-', color='gold', alpha=1, label='Indian')

axs.set_ylim(6, 20)
axs.set_xlim(1981, 2022)
axs.set_title('i) Spatially averaged MHW duration [days]')

outfile = r'.\MHW_Dur_ts.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





###Cumulative Intensity###
MHW_cum_ts = total['CUMannualmean_metric']
MHW_cum_ts = MHW_cum_ts + mask_ts
#Global
MHW_cum_global = np.nanmean(MHW_cum_ts, axis=(0,1))
MHW_cum_sd = np.nanstd(MHW_cum_ts, axis=(0,1))
error_MHW_cum = MHW_cum_sd/np.sqrt(40)
#PAC
PAC_1_MHW_cum = np.nanmean(MHW_cum_ts[660:720,:,:], axis=(0,1))
PAC_2_MHW_cum = np.nanmean(MHW_cum_ts[0:60,:,:], axis=(0,1))
PAC_3_MHW_cum = np.nanmean(MHW_cum_ts[60:120,:,:], axis=(0,1))
PAC_4_MHW_cum = np.nanmean(MHW_cum_ts[120:180,:,:], axis=(0,1))
PAC_5_MHW_cum = np.nanmean(MHW_cum_ts[180:220,:,:], axis=(0,1))
PAC_MHW_cum = (PAC_1_MHW_cum + PAC_2_MHW_cum + PAC_3_MHW_cum + PAC_4_MHW_cum + PAC_5_MHW_cum)/5
del PAC_1_MHW_cum, PAC_2_MHW_cum, PAC_3_MHW_cum, PAC_4_MHW_cum, PAC_5_MHW_cum
#ATL
ATL_MHW_cum = np.nanmean(MHW_cum_ts[219:401,:,:], axis=(0,1))
#IND
IND_MHW_cum = np.nanmean(MHW_cum_ts[400:661,:,:], axis=(0,1))


fig, axs = plt.subplots(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})

res_MHW_cum = stats.linregress(time, MHW_cum_global)
axs.plot(time, MHW_cum_global, '-', color='black', linewidth=2, label='Globally')
axs.plot(time, res_MHW_cum.intercept + res_MHW_cum.slope*time, '--', color='black')
axs.fill_between(time, MHW_cum_global-error_MHW_cum, MHW_cum_global+error_MHW_cum,
    alpha=0.2, edgecolor='black', facecolor='black',
    linewidth=0, antialiased=True)
axs.tick_params(length=10, direction='in')

axs.plot(time, PAC_MHW_cum, '-', color='peru', alpha=1, label='Pacific')

axs.plot(time, ATL_MHW_cum, '-', color='red', alpha=1, label='Atlantic')

axs.plot(time, IND_MHW_cum, '-', color='gold', alpha=1, label='Indian')

axs.set_ylim(8, 36)
axs.set_xlim(1981, 2022)
axs.set_title('l) Spatially averaged MHW cumulative intensity [$^\circ$C·days]')

outfile = r'.\MHW_CumInt_ts_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







###Total annual MHW days###
MHW_td_ts = total['DAYannualmean_metric']
MHW_td_ts = MHW_td_ts + mask_ts
#Global
MHW_td_global = np.nanmean(MHW_td_ts, axis=(0,1))
MHW_td_sd = np.nanstd(MHW_td_ts, axis=(0,1))
error_MHW_td = MHW_td_sd/np.sqrt(40)
#PAC
PAC_1_MHW_td = np.nanmean(MHW_td_ts[660:720,:,:], axis=(0,1))
PAC_2_MHW_td = np.nanmean(MHW_td_ts[0:60,:,:], axis=(0,1))
PAC_3_MHW_td = np.nanmean(MHW_td_ts[60:120,:,:], axis=(0,1))
PAC_4_MHW_td = np.nanmean(MHW_td_ts[120:180,:,:], axis=(0,1))
PAC_5_MHW_td = np.nanmean(MHW_td_ts[180:220,:,:], axis=(0,1))
PAC_MHW_td = (PAC_1_MHW_td + PAC_2_MHW_td + PAC_3_MHW_td + PAC_4_MHW_td + PAC_5_MHW_td)/5
del PAC_1_MHW_td, PAC_2_MHW_td, PAC_3_MHW_td, PAC_4_MHW_td, PAC_5_MHW_td
#ATL
ATL_MHW_td = np.nanmean(MHW_td_ts[219:401,:,:], axis=(0,1))
#IND
IND_MHW_td = np.nanmean(MHW_td_ts[400:661,:,:], axis=(0,1))


fig, axs = plt.subplots(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})

res_MHW_td = stats.linregress(time, MHW_td_global)
axs.plot(time, MHW_td_global, '-', color='black', linewidth=2, label='Circumpolar')
axs.plot(time, res_MHW_td.intercept + res_MHW_td.slope*time, '--', color='black')
axs.fill_between(time, MHW_td_global-error_MHW_td, MHW_td_global+error_MHW_td,
    alpha=0.2, edgecolor='black', facecolor='black',
    linewidth=0, antialiased=True)
axs.tick_params(length=10, direction='in')

axs.plot(time, PAC_MHW_td, '-', color='peru', alpha=1, label='Pacific')

axs.plot(time, ATL_MHW_td, '-', color='red', alpha=1, label='Atlantic')

axs.plot(time, IND_MHW_td, '-', color='gold', alpha=1, label='Indian')

axs.set_ylim(0, 62)
axs.set_xlim(1981, 2022)
axs.set_title('a  Spatially averaged total annual MHW days [days]')

fig.legend(loc=(0.35, 0.6), frameon=False, fontsize=15)

outfile = r'.\MHW_TotalMHWDays_ts.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








###MHW Areal Coverage###
#Global
MHW_cnt_ts = MHW_cnt_ts+mask_ts #MHW frequency + Sea Ice mask
MHW_Area_ts = np.where(MHW_cnt_ts >= 1, 1, 0) #Set each grid point with at least a MHW event = 1
ocean_grid = np.where(MHW_cnt_ts >= 0, 1, 0)  #Set each ocean grid point = 1
MHW_Area_global = (np.sum(MHW_Area_ts, axis=(0,1)) / np.sum(ocean_grid, axis=(0,1))) * 100
MHW_Area_sd = np.nanstd(MHW_Area_ts, axis=(0,1))
error_MHW_Area = (MHW_Area_sd/np.sqrt(40))*100

#PAC
MHW_PAC_Area1 = (np.sum(MHW_Area_ts[660:720,:,:], axis=(0,1)) / np.sum(ocean_grid[660:720,:,:], axis=(0,1))) * 100
MHW_PAC_Area2 = (np.sum(MHW_Area_ts[0:60,:,:], axis=(0,1)) / np.sum(ocean_grid[0:60,:,:], axis=(0,1))) * 100
MHW_PAC_Area3 = (np.sum(MHW_Area_ts[60:120,:,:], axis=(0,1)) / np.sum(ocean_grid[60:120,:,:], axis=(0,1))) * 100
MHW_PAC_Area4 = (np.sum(MHW_Area_ts[120:180,:,:], axis=(0,1)) / np.sum(ocean_grid[120:180,:,:], axis=(0,1))) * 100
MHW_PAC_Area5 = (np.sum(MHW_Area_ts[180:220,:,:], axis=(0,1)) / np.sum(ocean_grid[180:220,:,:], axis=(0,1))) * 100
MHW_PAC_Area = (MHW_PAC_Area1 + MHW_PAC_Area2 + MHW_PAC_Area3 + MHW_PAC_Area4 + MHW_PAC_Area5)/5
del MHW_PAC_Area1, MHW_PAC_Area2, MHW_PAC_Area3, MHW_PAC_Area4, MHW_PAC_Area5

#ATL
MHW_ATL_Area = (np.sum(MHW_Area_ts[219:401,:,:], axis=(0,1)) / np.sum(ocean_grid[219:401,:,:], axis=(0,1))) * 100

#IND
MHW_IND_Area = (np.sum(MHW_Area_ts[400:661,:,:], axis=(0,1)) / np.sum(ocean_grid[400:661,:,:], axis=(0,1))) * 100


fig, axs = plt.subplots(figsize=(10, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 16})

res_MHW_Area = stats.linregress(time, MHW_Area_global)
axs.plot(time, MHW_Area_global, '-', color='black', linewidth=2, label='Globally')
axs.plot(time, res_MHW_Area.intercept + res_MHW_Area.slope*time, '--', color='black')
axs.fill_between(time, MHW_Area_global-error_MHW_Area, MHW_Area_global+error_MHW_Area,
    alpha=0.2, edgecolor='black', facecolor='black',
    linewidth=0, antialiased=True)
axs.tick_params(length=10, direction='in')

axs.plot(time, MHW_PAC_Area, '-', color='peru', alpha=1, label='Pacific')

axs.plot(time, MHW_ATL_Area, '-', color='red', alpha=1, label='Atlantic')

axs.plot(time, MHW_IND_Area, '-', color='gold', alpha=1, label='Indian')

axs.set_ylim(15, 95)
axs.set_xlim(1981, 2022)
axs.set_title('b  Spatially averaged MHW Area ratio [%]')


outfile = r'.\MHW_Area_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)
