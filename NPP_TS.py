# -*- coding: utf-8 -*-
"""
#########################  NPP TS ###################
"""

# Load required modules
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy import io
from datetime import date
from netCDF4 import Dataset 
import xarray as xr 
import matplotlib
import matplotlib.pyplot as plt
import marineHeatWaves as mhw
from tempfile import TemporaryFile


# #Load sea-ice mask
# file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
# data_mask = np.load(file+'.npz')
# mask = data_mask['mask']
# mask_ts=mask[:,:,np.newaxis]


# ###CbPM NPP###
# NPP_interp_monthly = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_interp_monthly.mat')
# NPP_interp_monthly = NPP_interp_monthly['CbPM_interp_monthly']
# NPP_interp_monthly = NPP_interp_monthly+mask_ts

t = np.arange(date(1993,1,1).toordinal(),date(2020,12,31).toordinal()+1)
dates = [date.fromordinal(tt.astype(int)) for tt in t]

start_date = '1993-01-01'
end_date = '2020-12-31'

# Create a monthly/year array of dates
monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
annual_dates = pd.date_range(start=start_date, end=end_date, freq='Y')

# Convert the dates to a numpy array if needed
monthly_dates = np.array(monthly_dates)
annual_dates = np.array(annual_dates)

#CHL
ds = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CHL\Daily_chl\*.nc', parallel=True)


ds_Pacific = ds.sel(longitude=slice(-180,-65))
ds_Atlantic = ds.sel(longitude=slice(-65,25))
ds_Indian = ds.sel(longitude=slice(25,150))

npp_Circumpolar = ds.nppv.resample(time='1Y').mean(dim='time', skipna=True)
npp_Pacific = ds_Pacific.nppv.resample(time='1M').mean(dim='time', skipna=True)
npp_Atlantic = ds_Atlantic.nppv.resample(time='1M').mean(dim='time', skipna=True)
npp_Indian = ds_Indian.nppv.resample(time='1M').mean(dim='time', skipna=True)


#Circumpolar
NPP_Circumpolar_year = np.nanmean(npp_Circumpolar, axis=(1, 2, 3))
# NPP_sd = np.nanstd(NPP_CbPM_interp, axis=(0,1))
# error_NPP = (NPP_sd/np.sqrt(24))*1.5

#Pacific
NPP_PAC = np.nanmean(npp_Pacific, axis=(1, 2, 3))

#Atlantic
NPP_ATL = np.nanmean(npp_Atlantic, axis=(1, 2, 3))

#Indian
NPP_IND = np.nanmean(npp_Indian, axis=(1, 2, 3))




plt.figure(figsize=(20, 5))
# Plotting the time series with transparency
plt.plot(monthly_dates, NPP_PAC, color='darkblue', label='Pacific', alpha=0.7)
plt.plot(monthly_dates, NPP_ATL, color='darkgreen', label='Atlantic', alpha=0.7)
plt.plot(monthly_dates, NPP_IND, color='gold', label='Indian', alpha=0.7)
# plt.plot(monthly_dates, NPP_Circumpolar_year, 'o', color='black', label='Monthly Circumpolar', alpha=1)

# Setting labels and title
plt.xlabel('Time')
plt.ylabel(r'CbPM NPP [$mg C·m^{-3}·day^{-1}$]')

# Adding legend
plt.legend()









# Setting the figure size and style
plt.figure(figsize=(10, 5))

# Creating an array of indices for x-axis positioning
x_indices = np.arange(len(monthly_dates))

# Plotting the time series with appropriate styles as stacked bars
plt.bar(x_indices, NPP_PAC, color='darkblue', label='Pacific', alpha=1, width=0.5)
plt.bar(x_indices, NPP_ATL, color='darkgreen', label='Atlantic', alpha=1, width=0.5)
plt.bar(x_indices, NPP_IND, color='goldenrod', label='Indian', alpha=1, width=0.5)

# Selecting only the monthly dates for x-axis tick labels
monthly_ticks = monthly_dates[::20]  # Adjust the frequency of tick labels as needed

# Formatting the monthly dates to 'Year-Month' format
formatted_ticks = np.datetime_as_string(monthly_ticks, unit='M')

# Setting labels and title
plt.ylabel('CbPM NPP [$mg C·m^{-3}·day^{-1}$]', fontsize=12)
plt.xticks(x_indices[::20], formatted_ticks, ha='right', rotation=45, fontsize=10)  # Use the formatted ticks
plt.yticks(fontsize=10)

# Adjusting the plot layout
plt.legend(loc='upper left', frameon=False, fontsize=10)
plt.tight_layout()

# Adjusting the y-axis limits (optional)
plt.ylim(bottom=0)  # Set the bottom limit to 0 if necessary

















