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
from scipy.io import loadmat

## Load MHW_metrics_from_MATLAB.py

# t = np.arange(date(1998,1,1).toordinal(),date(2021,12,31).toordinal()+1)
# dates = [date.fromordinal(tt.astype(int)) for tt in t]

start_date = '1998-01-01'
end_date = '2021-12-31'

# # Create a monthly/year array of dates
monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
# annual_dates = pd.date_range(start=start_date, end=end_date, freq='Y')

# # Convert the dates to a numpy array if needed
# monthly_dates = np.array(monthly_dates)
# annual_dates = np.array(annual_dates)


# ###CbPM NPP###
time_NPP = np.arange(1, 289)
year_NPP = np.arange(1, 290, 12)
NPP_CbPM_interp = NPP_CbPM_interp+mask_ts

NPP_interp_monthly = NPP_interp_monthly+mask_ts


#Circumpolar
NPP_Circumpolar = np.nanmean(NPP_CbPM_interp, axis=(0,1))
NPP_Circumpolar = np.append(NPP_Circumpolar, 374)
NPP_sd = np.nanstd(NPP_CbPM_interp, axis=(0,1))
error_NPP = NPP_sd/np.sqrt(288)*10
error_NPP = np.append(error_NPP, 113)

#PAC
NPP_PAC_1 = np.nanmean(NPP_interp_monthly[660:720,:,:], axis=(0,1))
NPP_PAC_2 = np.nanmean(NPP_interp_monthly[0:60,:,:], axis=(0,1)) 
NPP_PAC_3 = np.nanmean(NPP_interp_monthly[60:120,:,:], axis=(0,1)) 
NPP_PAC_4 = np.nanmean(NPP_interp_monthly[120:180,:,:], axis=(0,1)) 
NPP_PAC_5 = np.nanmean(NPP_interp_monthly[180:220,:,:], axis=(0,1))
NPP_PAC = (NPP_PAC_1 + NPP_PAC_2 + NPP_PAC_3 + NPP_PAC_4 + NPP_PAC_5)/5
del NPP_PAC_1, NPP_PAC_2, NPP_PAC_3, NPP_PAC_4, NPP_PAC_5

#ATL
NPP_ATL = np.nanmean(NPP_interp_monthly[219:401,:,:], axis=(0,1)) 

#IND
NPP_IND = np.nanmean(NPP_interp_monthly[400:661,:,:], axis=(0,1))




# Definir tamaños de fuente y ticks
font_size = 18
tick_size = 16
legend_size = 16

# Crear la figura
fig = plt.figure(figsize=(20, 5))

# Plotear las series de tiempo con transparencia
plt.plot(year_NPP, NPP_Circumpolar, 'o', color='red', label='Anual Circumpolar', alpha=1)
plt.fill_between(year_NPP, NPP_Circumpolar - error_NPP, NPP_Circumpolar + error_NPP, color='red', alpha=0.2)
plt.plot(time_NPP, NPP_PAC, color='darkblue', label='Pacífico', alpha=0.7)
plt.plot(time_NPP, NPP_ATL, color='darkgreen', label='Atlántico', alpha=0.7)
plt.plot(time_NPP, NPP_IND, color='gold', label='Índico', alpha=0.7)

# Configurar etiquetas y título
plt.ylabel(r'CbPM NPP [$mg C·m^{-2}·día^{-1}$]', fontsize=font_size)

# Configurar el tamaño de fuente de los ticks en los ejes x e y
plt.xticks(fontsize=tick_size)
plt.yticks([200, 400, 600, 800], fontsize=tick_size)

# Configurar límites del eje x e y
plt.xlim(-5, 288+5)
plt.ylim(0, 900)

# Personalizar los ticks en el eje x
xtick_positions = [0, 50, 100, 150, 200, 250]
xtick_labels = ["Ene-1998", "Mar-2002", "May-2006", "Jul-2010", "Sep-2014", "Nov-2018"]
plt.xticks(xtick_positions, xtick_labels)

# Añadir leyenda
plt.legend(frameon=False, fontsize=legend_size)

# Configurar los ticks para que vayan hacia dentro
plt.tick_params(axis='both', direction='in', which='both', length=6, width=2)



outfile = r'.\NPP_ts.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')






#CHL
ds = xr.open_mfdataset(r'.\Daily_chl\*.nc', parallel=True)


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
plt.plot(annual_dates, NPP_Circumpolar_year, 'o', color='black', label='Anual Circumpolar', alpha=1)

# Setting labels and title
# plt.xlabel('Time')
plt.ylabel(r'CbPM NPP [$mg C·m^{-3}·día^{-1}$]')

# Adding legend
plt.legend()






# Setting the figure size and style
plt.figure(figsize=(30, 10))

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














