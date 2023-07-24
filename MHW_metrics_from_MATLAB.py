# -*- coding: utf-8 -*-
"""
##################################.mat to .npy##################################
"""

from scipy.io import loadmat
import numpy as np
from netCDF4 import Dataset 


#lat and lon
lat = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\lat.mat')
lat = lat['latitud']

lon = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\lon.mat')
lon = lon['longitud']



##MHW Duration 
MHW_dur = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_dur.mat')
MHW_dur = MHW_dur['DURmean_metric']

MHW_dur_tr = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_dur_tr.mat')
MHW_dur_tr = MHW_dur_tr['DURtrend_metric']


##MHW Frequency
MHW_cnt = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_freq.mat')
MHW_cnt = MHW_cnt['Fmean_metric']

MHW_cnt_tr = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_freq_tr.mat')
MHW_cnt_tr = MHW_cnt_tr['Ftrend_metric']


##MHW Mean Instensity
MHW_mean = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_mean.mat')
MHW_mean = MHW_mean['Mmean_metric']

MHW_mean_tr = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_mean_tr.mat')
MHW_mean_tr = MHW_mean_tr['Mtrend_metric']


##MHW Max Instensity
MHW_max = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_max.mat')
MHW_max = MHW_max['MAXmean_metric']

MHW_max_tr = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_max_tr.mat')
MHW_max_tr = MHW_max_tr['MAXtrend_metric']

##MHW Cumulative Instensity
MHW_cum = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_cum.mat')
MHW_cum = MHW_cum['CUMmean_metric']

MHW_cum_tr = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_cum_tr.mat')
MHW_cum_tr = MHW_cum_tr['CUMtrend_metric']


##Total MHW Annual MHW Days
MHW_td = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_cnt.mat')
MHW_td = MHW_td['DAYmean_metric']

MHW_td_tr = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\MHW_cnt_tr.mat')
MHW_td_tr = MHW_td_tr['DAYtrend_metric']


################
### TOTAL MATRIX ###
################

total = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\total.mat')


MHW_cum_ts = total['CUMannualmean_metric']
MHW_cum_ts[0,:,:] = MHW_cum_ts[1,:,:]
MHW_cum_ts[719,:,:] = MHW_cum_ts[718,:,:]
MHW_cum_ts[:,0,:] = MHW_cum_ts[:,1,:]
MHW_cum_ts[:,99,:] = MHW_cum_ts[:,98,:]

MHW_cum_dtr = total['CUMdetrend_metric']



MHW_td_ts = total['DAYannualmean_metric']
MHW_td_ts[0,:,:] = MHW_td_ts[1,:,:]
MHW_td_ts[719,:,:] = MHW_td_ts[718,:,:]
MHW_td_ts[:,0,:] = MHW_td_ts[:,1,:]
MHW_td_ts[:,99,:] = MHW_td_ts[:,98,:]

MHW_td_dtr = total['DAYdetrend_metric']



MHW_dur_ts = total['DURannualmean_metric']
MHW_dur_ts[0,:,:] = MHW_dur_ts[1,:,:]
MHW_dur_ts[719,:,:] = MHW_dur_ts[718,:,:]
MHW_dur_ts[:,0,:] = MHW_dur_ts[:,1,:]
MHW_dur_ts[:,99,:] = MHW_dur_ts[:,98,:]

MHW_dur_dtr = total['DURdetrend_metric']



MHW_cnt_ts = total['Fannual_metric']
MHW_cnt_ts[0,:,:] = MHW_cnt_ts[1,:,:]
MHW_cnt_ts[719,:,:] = MHW_cnt_ts[718,:,:]
MHW_cnt_ts[:,0,:] = MHW_cnt_ts[:,1,:]
MHW_cnt_ts[:,99,:] = MHW_cnt_ts[:,98,:]

MHW_cnt_dtr = total['Fdetrend_metric']



MHW_mean_ts = total['Mannualmean_metric']
MHW_mean_ts[0,:,:] = MHW_mean_ts[1,:,:]
MHW_mean_ts[719,:,:] = MHW_mean_ts[718,:,:]
MHW_mean_ts[:,0,:] = MHW_mean_ts[:,1,:]
MHW_mean_ts[:,99,:] = MHW_mean_ts[:,98,:]

MHW_mean_dtr = total['Mdetrend_metric']



MHW_max_ts = total['MAXannual_mean_metric']
MHW_max_ts[0,:,:] = MHW_max_ts[1,:,:]
MHW_max_ts[719,:,:] = MHW_max_ts[718,:,:]
MHW_max_ts[:,0,:] = MHW_max_ts[:,1,:]
MHW_max_ts[:,99,:] = MHW_max_ts[:,98,:]

MHW_max_dtr = total['MAXdetrend_metric']


########################
### TOTAL MATRIX NPP ###
########################


Total_NPP_CbPM = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\Total.mat')

NPP_CbPM = Total_NPP_CbPM['Total']

NPP_CbPM_interp = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_interp2MHW.mat')
NPP_CbPM_interp = NPP_CbPM_interp['CbPM_interp2MHW']

NPP_1998_2015_trends = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\NPP_1998_2015_trends.mat')
NPP_1998_2015_trends = NPP_1998_2015_trends['NPP_1998_2015_trends']

signif_1998_2015 = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\signif_1998_2015.mat')
signif_1998_2015 = signif_1998_2015['signif_1998_2015']

NPP_2009_2015_trends = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\NPP_2009_2015_trends.mat')
NPP_2009_2015_trends = NPP_2009_2015_trends['NPP_2009_2015_trends']

signif_2009_2015 = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\signif_2009_2015.mat')
signif_2009_2015 = signif_2009_2015['signif_2009_2015']

NPP_2015_2021_trends = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\NPP_2015_2021_trends.mat')
NPP_2015_2021_trends = NPP_2015_2021_trends['NPP_2015_2021_trends']

signif_2015_2021 = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\signif_2015_2021.mat')
signif_2015_2021 = signif_2015_2021['signif_2015_2021']

NPP_interp_monthly = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_interp_monthly.mat')
NPP_interp_monthly = NPP_interp_monthly['CbPM_interp_monthly']


#############
### Masks ###
#############
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
maskT = mask.T


file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\mask_full'
data_mask_full = np.load(file+'.npz')
mask_full = data_mask_full['mask_full']

#############
### SSTA ####
#############
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Aver_Mean_SSTA_1982_2021'
data_Aver_Mean_SSTA_1982_2021 = np.load(file+'.npz')
Aver_Mean_SSTA_1982_2021 = data_Aver_Mean_SSTA_1982_2021['Mean_SSTA_1982_2021']

file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Mean_SSTA_1982_2021_ts'
data_Mean_SSTA_1982_2021_ts = np.load(file+'.npz')
Mean_SSTA_1982_2021_ts = data_Mean_SSTA_1982_2021_ts['Mean_SSTA_1982_2021_ts']


file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Aver_Max_SSTA_1982_2021'
data_Aver_Max_SSTA_1982_2021 = np.load(file+'.npz')
Aver_Max_SSTA_1982_2021 = data_Aver_Max_SSTA_1982_2021['Max_SSTA_1982_2021']

file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Max_SSTA_1982_2021_ts'
data_Max_SSTA_1982_2021_ts = np.load(file+'.npz')
Max_SSTA_1982_2021_ts = data_Max_SSTA_1982_2021_ts['Max_SSTA_1982_2021_ts']

file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Max_SSTA_sd'
data_Max_SSTA_sd = np.load(file+'.npz')
Max_SSTA_sd = data_Max_SSTA_sd['Max_SSTA_sd']

# file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\SSTA_periods'
# data_SSTA_periods = np.load(file+'.npz')
# sst_anom_1982_1991 = data_SSTA_periods['sst_anom_1982_1991']
# sst_anom_1992_2001 = data_SSTA_periods['sst_anom_1992_2001']
# sst_anom_2002_2011 = data_SSTA_periods['sst_anom_2002_2011']
# sst_anom_2012_2021 = data_SSTA_periods['sst_anom_2012_2021']

file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\SSTA_1982_2015_2021'
data_SSTA_periods = np.load(file+'.npz')
sst_anom_1982_2015 = data_SSTA_periods['sst_anom_1982_2015']
sst_anom_2009_2015 = data_SSTA_periods['sst_anom_2009_2015']
sst_anom_2015_2021 = data_SSTA_periods['sst_anom_2015_2021']




#############
### SST #####
#############
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Aver_SST'
data_Aver_SST = np.load(file+'.npz')
Aver_SST = data_Aver_SST['Aver_SST']

ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SST_trends\SST_trends.nc')
SST_trends = ds_trend['trend'][:]*10 # Convert to trends per decade
signif = ds_trend['signif'][:]
SST_p_value = ds_trend['p'][:]
# std_error = ds_trend['std_error'][:]


#################
## SeaIce Conc ##
#################
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\SeaIceFraction'
data_SeaIceFraction = np.load(file+'.npz')
Mean_SeaIceFraction_1982_2021 = data_SeaIceFraction['SeaIceFraction']

file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\SeaIce_ts'
data_SeaIce_ts = np.load(file+'.npz')
SeaIce_ts = data_SeaIce_ts['SeaIce_ts']

file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\SIC_Anom_2015_2021'
SIC_Anom_2015_2021 = np.load(file+'.npy')



######################################
## Averaged N-SAT Anomalies and SIC ##
######################################
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Averaged_SAT_SIC'
data_SAT_SIC_avg = np.load(file+'.npz')

# SIC_ts = data_SAT_SIC_avg['SIC_ts']
# SAT_ts = data_SAT_SIC_avg['SAT_ts']

SAT_anom_jan = data_SAT_SIC_avg['SAT_anom_jan']
SAT_anom_feb = data_SAT_SIC_avg['SAT_anom_feb']
SAT_anom_mar = data_SAT_SIC_avg['SAT_anom_mar']
SAT_anom_apr = data_SAT_SIC_avg['SAT_anom_apr']
SAT_anom_may = data_SAT_SIC_avg['SAT_anom_may']
SAT_anom_jun = data_SAT_SIC_avg['SAT_anom_jun']
SAT_anom_jul = data_SAT_SIC_avg['SAT_anom_jul']
SAT_anom_ago = data_SAT_SIC_avg['SAT_anom_ago']
SAT_anom_sep = data_SAT_SIC_avg['SAT_anom_sep']
SAT_anom_oct = data_SAT_SIC_avg['SAT_anom_oct']
SAT_anom_nov = data_SAT_SIC_avg['SAT_anom_nov']
SAT_anom_dec = data_SAT_SIC_avg['SAT_anom_dec']

SIC_jan = data_SAT_SIC_avg['SIC_jan']
SIC_feb = data_SAT_SIC_avg['SIC_feb']
SIC_mar = data_SAT_SIC_avg['SIC_mar']
SIC_apr = data_SAT_SIC_avg['SIC_apr']
SIC_may = data_SAT_SIC_avg['SIC_may']
SIC_jun = data_SAT_SIC_avg['SIC_jun']
SIC_jul = data_SAT_SIC_avg['SIC_jul']
SIC_ago = data_SAT_SIC_avg['SIC_ago']
SIC_sep = data_SAT_SIC_avg['SIC_sep']
SIC_oct = data_SAT_SIC_avg['SIC_oct']
SIC_nov = data_SAT_SIC_avg['SIC_nov']
SIC_dec = data_SAT_SIC_avg['SIC_dec']
