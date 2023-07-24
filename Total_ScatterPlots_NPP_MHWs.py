# -*- coding: utf-8 -*-
"""

############################# TOTAL SCATTER PLOTS #############################

"""


#Importing required libraries
from scipy.io import loadmat
from scipy import stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import xarray as xr 
import pandas as pd


#Calculating correlation in sectors
#Load sea-ice mask
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
mask_ts=mask[:,:,np.newaxis]

time = np.arange(1982, 2022)

###CbPM NPP###
NPP_CbPM_interp = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_interp2MHW.mat')
NPP_CbPM_interp = NPP_CbPM_interp['CbPM_interp2MHW']
NPP_CbPM_interp = NPP_CbPM_interp + mask_ts


###Maximum SSTA###
# ds = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SST_full\SST_ANT_1982-2021_40.nc')

# sst_full = ds['analysed_sst'][:,::10,::10] - 273.15

# #Climatology 1982-2011
# ds_clim=ds.sel(time=slice("1982-01-01", "2011-12-31"))
# sst_clim=ds_clim['analysed_sst'][:,::10,::10].groupby('time.month').mean(dim='time', skipna=True)

# #Compute SST Anomaly
# sst_anom=ds['analysed_sst'][:,::10,::10].groupby('time.month') - sst_clim

#Save anomalies data so far
# sst_anom.to_netcdf(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SST_Anomalies\SSTA_SO.nc')

ds = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SST_Anomalies\SSTA_SO.nc')

sst_anom = ds['analysed_sst']

Max_SSTA=sst_anom.groupby('time.year').max(dim='time',skipna=True)

Max_SSTA = Max_SSTA[:,:,16:40]

Max_SSTA = Max_SSTA + mask_ts

## MHW metrics ##
total = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\total.mat')
###Frequency###
MHW_cnt_ts = total['Fannual_metric']
MHW_cnt_ts = MHW_cnt_ts + mask_ts
MHW_cnt_ts = MHW_cnt_ts[:,:,16:40]


###Duration###
MHW_dur_ts = total['DURannualmean_metric']
MHW_dur_ts = MHW_dur_ts + mask_ts
MHW_dur_ts = MHW_dur_ts[:,:,16:40]


###Max Intensity###
MHW_max_ts = total['MAXannual_mean_metric']
MHW_max_ts = MHW_max_ts + mask_ts
MHW_max_ts = MHW_max_ts[:,:,16:40]


###Cum Intensity###
MHW_cum_ts = total['CUMannualmean_metric']
MHW_cum_ts = MHW_cum_ts + mask_ts
MHW_cum_ts = MHW_cum_ts[:,:,16:40]





                ##Representing Scatter plots##
                
# Define the longitude ranges for different areas
pacific_range = [(660, 720), (0, 60), (120, 180), (180, 220)]
atlantic_range = [(219, 401)]
indian_range = [(400, 661)]

# Create  color and marker lists with three distinctions for the areas
colors = {'Pacific': 'orangered', 'Atlantic': 'xkcd:mango', 'Indian': 'gold'}
markers = {'Pacific': 'o', 'Atlantic': 'o', 'Indian': 'o'}

# Setting arrays shapes
n_longitud, n_latitud, n_timesteps = Max_SSTA.shape



###Maximum SSTA###
# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 5))

# Set font size
plt.rcParams.update({'font.size': 18})

# ax.scatter(Max_SSTA.values[660:720, :, :].flatten(), NPP_CbPM_interp[660:720, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(Max_SSTA.values[0:60, :, :].flatten(), NPP_CbPM_interp[0:60, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(Max_SSTA.values[120:180, :, :].flatten(), NPP_CbPM_interp[120:180, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(Max_SSTA.values[180:220, :, :].flatten(), NPP_CbPM_interp[180:220, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')


# ax.scatter(Max_SSTA.values[219:401, :, :].flatten(), NPP_CbPM_interp[219:401, :, :].flatten(), color=colors['Atlantic'], marker=markers['Atlantic'], alpha=0.5, label='Atlantic')

ax.scatter(Max_SSTA.values[400:661, :, :].flatten(), NPP_CbPM_interp[400:661, :, :].flatten(), color=colors['Indian'], marker=markers['Indian'], alpha=0.5, label='Indian')



# Customize the scatter plot
plt.title('Maximum SSTA', fontsize=20)  
plt.xlabel('[$^\circ$C]', fontsize=18)  
plt.ylabel('CbPM NPP [$mg C·m^{-2}·day^{-1}$]', fontsize=18)

# Set x-axis and y-axis limits
ax.set_xlim(-1, 11)
ax.set_ylim(-200, 4000)

# # Create custom legend with desired labels
# legend_elements = [
#     plt.Line2D([], [], marker='o', color='w', markerfacecolor='orangered', markersize=10, label='Pacific'),
#     plt.Line2D([], [], marker='o', color='w', markerfacecolor='xkcd:mango', markersize=10, label='Atlantic'),
#     plt.Line2D([], [], marker='o', color='w', markerfacecolor='gold', markersize=10, label='Indian')
# ]

# # Add legend to the plot
# ax.legend(handles=legend_elements, frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figs_Explicacion_Reviews\NPP_toMHWs\NPP_Max_SSTA_PAC.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







###MHW Frequency###
# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 5))

# Set font size
plt.rcParams.update({'font.size': 18})

# ax.scatter(MHW_cnt_ts[660:720, :, :].flatten(), NPP_CbPM_interp[660:720, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_cnt_ts[0:60, :, :].flatten(), NPP_CbPM_interp[0:60, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_cnt_ts[120:180, :, :].flatten(), NPP_CbPM_interp[120:180, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_cnt_ts[180:220, :, :].flatten(), NPP_CbPM_interp[180:220, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')

# ax.scatter(MHW_cnt_ts[219:401, :, :].flatten(), NPP_CbPM_interp[219:401, :, :].flatten(), color=colors['Atlantic'], marker=markers['Atlantic'], alpha=0.5, label='Atlantic')

ax.scatter(MHW_cnt_ts[400:661, :, :].flatten(), NPP_CbPM_interp[400:661, :, :].flatten(), color=colors['Indian'], marker=markers['Indian'], alpha=0.5, label='Indian')



# Set x-axis and y-axis limits
ax.set_xlim(-1, 17)
# Set custom x-axis ticks
ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
ax.set_ylim(-200, 4000)

# Customize the scatter plot
plt.title('MHW Frequency', fontsize=20)
plt.xlabel('[number]', fontsize=18)
plt.ylabel('CbPM NPP [$mg C·m^{-2}·day^{-1}$]', fontsize=18)

# # Create custom legend with desired labels
# legend_elements = [
#     plt.Line2D([], [], marker='o', color='w', markerfacecolor='orangered', markersize=8, label='Pacific'),
#     plt.Line2D([], [], marker='^', color='w', markerfacecolor='xkcd:mango', markersize=8, label='Atlantic'),
#     plt.Line2D([], [], marker='s', color='w', markerfacecolor='gold', markersize=8, label='Indian')
# ]

# # Add legend to the plot
# ax.legend(handles=legend_elements, frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figs_Explicacion_Reviews\NPP_toMHWs\NPP_MHWFreq_IND.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






###MHW Duration###
# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 5))

# Set font size
plt.rcParams.update({'font.size': 18})

# ax.scatter(MHW_dur_ts[660:720, :, :].flatten(), NPP_CbPM_interp[660:720, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_dur_ts[0:60, :, :].flatten(), NPP_CbPM_interp[0:60, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_dur_ts[120:180, :, :].flatten(), NPP_CbPM_interp[120:180, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_dur_ts[180:220, :, :].flatten(), NPP_CbPM_interp[180:220, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')

# ax.scatter(MHW_dur_ts[219:401, :, :].flatten(), NPP_CbPM_interp[219:401, :, :].flatten(), color=colors['Atlantic'], marker=markers['Atlantic'], alpha=0.5, label='Atlantic')

ax.scatter(MHW_dur_ts[400:661, :, :].flatten(), NPP_CbPM_interp[400:661, :, :].flatten(), color=colors['Indian'], marker=markers['Indian'], alpha=0.5, label='Indian')



# Customize the scatter plot
plt.title('MHW Duration', fontsize=20)
plt.xlabel('[days]', fontsize=18)
plt.ylabel('CbPM NPP [$mg C·m^{-2}·day^{-1}$]', fontsize=18)


# Set x-axis and y-axis limits
ax.set_xlim(-5, 215)
ax.set_xticks([0, 30, 60, 90, 120, 150, 180, 210])
ax.set_ylim(-200, 4000)


# # Create custom legend with desired labels
# legend_elements = [
#     plt.Line2D([], [], marker='o', color='w', markerfacecolor='orangered', markersize=8, label='Pacific'),
#     plt.Line2D([], [], marker='^', color='w', markerfacecolor='xkcd:mango', markersize=8, label='Atlantic'),
#     plt.Line2D([], [], marker='s', color='w', markerfacecolor='gold', markersize=8, label='Indian')
# ]

# # Add legend to the plot
# ax.legend(handles=legend_elements, frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figs_Explicacion_Reviews\NPP_toMHWs\NPP_MHWDur_IND.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




###MHW Cum Intensity###
# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 5))

# Set font size
plt.rcParams.update({'font.size': 18})

# ax.scatter(MHW_cum_ts[660:720, :, :].flatten(), NPP_CbPM_interp[660:720, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_cum_ts[0:60, :, :].flatten(), NPP_CbPM_interp[0:60, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_cum_ts[120:180, :, :].flatten(), NPP_CbPM_interp[120:180, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')
# ax.scatter(MHW_cum_ts[180:220, :, :].flatten(), NPP_CbPM_interp[180:220, :, :].flatten(), color=colors['Pacific'], marker=markers['Pacific'], alpha=0.5, label='Pacific')

# ax.scatter(MHW_cum_ts[219:401, :, :].flatten(), NPP_CbPM_interp[219:401, :, :].flatten(), color=colors['Atlantic'], marker=markers['Atlantic'], alpha=0.5, label='Atlantic')

ax.scatter(MHW_cum_ts[400:661, :, :].flatten(), NPP_CbPM_interp[400:661, :, :].flatten(), color=colors['Indian'], marker=markers['Indian'], alpha=0.5, label='Indian')



# Customize the scatter plot
plt.title('MHW Cum Intensity', fontsize=20)
plt.xlabel('[$^\circ$C·days]', fontsize=18)
plt.ylabel('CbPM NPP [$mg C·m^{-2}·day^{-1}$]', fontsize=18)

# Set x-axis and y-axis limits
ax.set_xlim(-20, 350)
ax.set_ylim(-200, 4000)


# # Create custom legend with desired labels
# legend_elements = [
#     plt.Line2D([], [], marker='o', color='w', markerfacecolor='orangered', markersize=8, label='Pacific'),
#     plt.Line2D([], [], marker='^', color='w', markerfacecolor='xkcd:mango', markersize=8, label='Atlantic'),
#     plt.Line2D([], [], marker='s', color='w', markerfacecolor='gold', markersize=8, label='Indian')
# ]

# # Add legend to the plot
# ax.legend(handles=legend_elements, frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figs_Explicacion_Reviews\NPP_toMHWs\NPP_MHWCumInt_IND.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)
















############################################################

# Creating masks for the longitudes
mask1 = np.zeros((720, 100, 24), dtype=bool)
mask1[219:401, :, :] = True

mask2 = np.zeros((720, 100, 24), dtype=bool)
mask2[400:661, :, :] = True

# Enmascarar las longitudes en el DataArray
NPP_PAC = np.ma.masked_where(mask1, NPP_CbPM_interp)
NPP_PAC = np.ma.masked_where(mask2, NPP_PAC)

Max_SSTA_PAC = np.ma.masked_where(mask1, Max_SSTA)
Max_SSTA_PAC = np.ma.masked_where(mask2, Max_SSTA_PAC)

MHW_cnt_PAC = np.ma.masked_where(mask1, MHW_cnt_ts)
MHW_cnt_PAC = np.ma.masked_where(mask2, MHW_cnt_PAC)

MHW_dur_PAC = np.ma.masked_where(mask1, MHW_dur_ts)
MHW_dur_PAC = np.ma.masked_where(mask2, MHW_dur_PAC)

MHW_cum_PAC = np.ma.masked_where(mask1, MHW_cum_ts)
MHW_cum_PAC = np.ma.masked_where(mask2, MHW_cum_PAC)

###########################################################




                    ####Linear regresions



                        ##Max SSTA
#Pacific
# Flatten the arrays and remove NaN values
x_data = Max_SSTA_PAC.flatten()
y_data = NPP_PAC.flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_MaxSSTA_PAC = stats.linregress(x_valid, y_valid)
res_MaxSSTA_PAC


#Atlantic
# Flatten the arrays and remove NaN values
x_data = Max_SSTA.values[219:401, :, :].flatten()
y_data = NPP_CbPM_interp[219:401, :, :].flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_MaxSSTA_ATL = stats.linregress(x_valid, y_valid)
res_MaxSSTA_ATL


#Indian
# Flatten the arrays and remove NaN values
x_data = Max_SSTA.values[400:661, :, :].flatten()
y_data = NPP_CbPM_interp[400:661, :, :].flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_MaxSSTA_IND = stats.linregress(x_valid, y_valid)
res_MaxSSTA_IND






                        ##MHW Frequency
#Pacific
# Flatten the arrays and remove NaN values
x_data = MHW_cnt_PAC.flatten()
y_data = NPP_PAC.flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Freq_PAC = stats.linregress(x_valid, y_valid)
res_Freq_PAC


#Atlantic
# Flatten the arrays and remove NaN values
x_data = MHW_cnt_ts[219:401, :, :].flatten()
y_data = NPP_CbPM_interp[219:401, :, :].flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Freq_ATL = stats.linregress(x_valid, y_valid)
res_Freq_ATL


#Indian
# Flatten the arrays and remove NaN values
x_data = MHW_cnt_ts[400:661, :, :].flatten()
y_data = NPP_CbPM_interp[400:661, :, :].flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Freq_IND = stats.linregress(x_valid, y_valid)
res_Freq_IND







                    ##MHW Duration
#Pacific
# Flatten the arrays and remove NaN values
x_data = MHW_dur_PAC.flatten()
y_data = NPP_PAC.flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Dur_PAC = stats.linregress(x_valid, y_valid)
res_Dur_PAC


#Atlantic
# Flatten the arrays and remove NaN values
x_data = MHW_dur_ts[219:401, :, :].flatten()
y_data = NPP_CbPM_interp[219:401, :, :].flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Dur_ATL = stats.linregress(x_valid, y_valid)
res_Dur_ATL


#Indian
# Flatten the arrays and remove NaN values
x_data = MHW_dur_ts[400:661, :, :].flatten()
y_data = NPP_CbPM_interp[400:661, :, :].flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Dur_IND = stats.linregress(x_valid, y_valid)
res_Dur_IND








##MHW Cum Intensity
#Pacific
# Flatten the arrays and remove NaN values
x_data = MHW_cum_PAC.flatten()
y_data = NPP_PAC.flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Cum_PAC = stats.linregress(x_valid, y_valid)
res_Cum_PAC


#Atlantic
# Flatten the arrays and remove NaN values
x_data = MHW_cum_ts[219:401, :, :].flatten()
y_data = NPP_CbPM_interp[219:401, :, :].flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Cum_ATL = stats.linregress(x_valid, y_valid)
res_Cum_ATL


#Indian
# Flatten the arrays and remove NaN values
x_data = MHW_cum_ts[400:661, :, :].flatten()
y_data = NPP_CbPM_interp[400:661, :, :].flatten()

valid_indices = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
x_valid = x_data[valid_indices]
y_valid = y_data[valid_indices]

# Perform linear regression with valid data
res_Cum_IND = stats.linregress(x_valid, y_valid)
res_Cum_IND
























