# -*- coding: utf-8 -*-
"""

############################# ECMWF ERA5 T2M #############################

"""

import os
import netCDF4 as nc
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import cartopy.crs as ccrs


import matplotlib.path as mpath


from datetime import datetime as dt
from netCDF4 import Dataset 
import xarray as xr 
import ecoliver as ecj

from dateutil.relativedelta import relativedelta


import cmocean as cm

import cartopy.feature as cft

#Path
os.chdir(r'.\ERA5_datasets')
filename = 'ERA5_1982-2021_T2M_ANT.nc'

#Load netcdf
ds = nc.Dataset(filename)

#Get dimensions
lons = ds.variables['longitude'][:]
lats = ds.variables['latitude'][:]
times = ds.variables['time'][:]
t2m = ds.variables['t2m'][:,:,:] - 273.15

#Calculate datetime
dtimes = np.empty(len(times), dtype='datetime64[h]')
for i,t in enumerate(times):
    dtimes[i] = np.datetime64('1900-01-01') + np.timedelta64(np.int64(t),'h')
dtimes = dtimes.astype('datetime64')



#Calculate mean Tempearture of 2019-2020 and TClim (mean) of Dec-Feb 1982-2011
dtimes_Clim = dtimes[dtimes < np.datetime64('2011-12-31T12:00')]

TClim_m = np.mean(t2m[np.where(dtimes == np.datetime64('1982-01-01T12:00'))[0][0]:np.where(dtimes == np.datetime64('2011-12-31T12:00'))[0][0]+1,:,:], axis=0)
T2020_m = np.mean(t2m[np.where(dtimes == np.datetime64('2020-02-01T12:00'))[0][0]:np.where(dtimes == np.datetime64('2020-02-29T12:00'))[0][0]+1,:,:], axis=0)
T_MHWevent_m = np.mean(t2m[np.where(dtimes == np.datetime64('2019-12-01T12:00'))[0][0]:np.where(dtimes == np.datetime64('2020-05-31T12:00'))[0][0]+1,:,:], axis=0)
np.savetxt('TClim_12H.csv', TClim_m, fmt='%.1f', delimiter=';')
np.savetxt('T2020_12H.csv', T2020_m, fmt='%.1f', delimiter=';')
np.savetxt('T_MHWevent_12H.csv', T_MHWevent_m, fmt='%.1f', delimiter=';')                 
                  
TClim = np.loadtxt('TClim_12H.csv', delimiter=';')     
T2020 = np.loadtxt('T2020_12H.csv', delimiter=';')
T_MHWevent_m = np.loadtxt('T_MHWevent_12H.csv', delimiter=';')                  
                  
#Calculating N-SAT
                    
#######################################################
#Calculate the percentile of 6 day temperature from 1950-to-2019 and compare with 2020
#######################################################
#Load data
# os.chdir(r'C:\ICMAN-CSIC\MHW_ANT\ERA5_datasets')

# TClim = np.loadtxt('TClim_24H.csv', delimiter=';')     
# T2020 = np.loadtxt('T2020_24H.csv', delimiter=';')
# T_MHWevent_m = np.loadtxt('T_MHWevent_24H.csv', delimiter=';')    


# #Calculate mean Tempearture of Feb 2020 and TClim (mean) of Feb 1982-2011
# dtimes_Clim = dtimes[dtimes < np.datetime64('2011-12-31T00:00')]

# #Calculate the mean of 2 years-running day temperature from 1982 to 2011
# T2m_6day_mean = np.empty((np.size(dtimes_Clim-6), np.size(t2m,1), np.size(t2m,2)))
# for i in range(np.size(dtimes_Clim-6)):
#     T2m_6day_mean[i,:,:] = np.mean(t2m[i:i+6,:,:], axis=0)


# #Calculate and save the percentile
# T2m611Feb_percentile = np.empty((np.size(T2m_6day_mean, 1),np.size(T2m_6day_mean, 2)))
# for i in range(np.size(T2m_6day_mean, 1)):
#     for j in range(np.size(T2m_6day_mean, 2)):
#         T2m611Feb_percentile[i,j] = scipy.stats.percentileofscore(T2m_6day_mean[:,i,j], T611Feb[i,j])
# np.savetxt('T2m_611Feb_percentile_24H.csv', T2m611Feb_percentile, fmt='%.1f', delimiter=';')



####################
##Plot Feb 2020 Anomalies
####################
#Load data
os.chdir(r'C:\ICMAN-CSIC\MHW_ANT\ERA5_datasets')

TClim = np.loadtxt('TClim_12H.csv', delimiter=';')
T2020 = np.loadtxt('T2020_Feb_12H.csv', delimiter=';')
T_MHWevent_m = np.loadtxt('T_MHWevent_12H.csv', delimiter=';')   


#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, projection=projection)

#cmap=plt.cm.YlOrRd  
#cmap=plt.cm.RdBu_r  
#cmap = 'Spectral_r'

p1 = ax.contourf(lons, lats, (T2020-TClim), np.arange(-8,8.5,2), cmap='bwr', transform=ccrs.PlateCarree(), extend='both') 
cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
#cbar.ax.get_yaxis().set_ticks([-5, -3, -1, 1, 3, 5])


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', linewidth= 1)

ax.set_title('Surface Air Temperature Anomaly [$^\circ$C]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


#Saveplot
plt.savefig('ERA5_T2manomaly_Feb_2000-clim_24H.png', bbox_inches='tight', pad_inches=0.5)
#plt.close()





############################# N-SAT ANOMALIES  AND TREND ########################



# Load required modules
 
import netCDF4 as nc
import numpy as np
import pandas as pd


from datetime import datetime as dt
from netCDF4 import Dataset 
import xarray as xr 
import ecoliver as ecj

from dateutil.relativedelta import relativedelta


import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import matplotlib.path as mpath
import cartopy.feature as cft

#Importing sea ice masks
file = r'.\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
maskT = mask.T

file = r'.\mask_full'
data_mask_full = np.load(file+'.npz')
mask_full = data_mask_full['mask_full']


#Dates
t, dates, T, year, month, day, doy = ecj.timevector([1982, 1, 1], [2021, 12, 31])


#Reading Dataset and variables
# latmin = -20.
# latmax = 20.
# lonmin = -70.
# lonmax = 15

ds = xr.open_mfdataset(r'.\SST_bimensual\*.nc', parallel=True)

#ds = Dataset(r'.\SST_ANT_1982-2021_40.nc')

#ds=ds.sel(lat=slice(latmin,latmax),lon=slice(lonmin,lonmax),time=slice("1982-01-01", "2021-12-30"))

lon = ds['lon']
lat = ds['lat']
times = ds['time'][:]


#Calculate datetime
times = times.astype('datetime64')


#Compute SST anual mean
sst_year=ds['analysed_sst'].groupby('time.year').mean(dim='time',skipna=True)#.load() 

Aver_SST =ds['analysed_sst'].mean(dim=('time'))#.load() #Average SST 1982-2021
Aver_SST -= 273.15


#Compute climatology
#sst_clim_full=ds['analysed_sst'].groupby('time.month').mean(dim='time')#.load

#Climatology 1982-2011
ds_clim=ds.sel(time=slice("1982-01-01", "2011-12-31"))
sst_clim=ds_clim['analysed_sst'].groupby('time.month').mean(dim='time')#.load


#Compute SST Anomaly
#sst_anom_full=ds['analysed_sst'].groupby('time.month') - sst_clim_full
sst_anom=ds['analysed_sst'].groupby('time.month') - sst_clim

max_sst_anom_ts=sst_anom.groupby('time.year').max(dim='time',skipna=True)
mean_sst_anom_ts=sst_anom.groupby('time.year').mean(dim='time',skipna=True)


#SSTA in concrete time periods
sst_anom_1982_1991 = np.mean(sst_anom[np.where(times == np.datetime64('1982-01-01T12:00'))[0][0]:np.where(times == np.datetime64('1991-12-31T12:00'))[0][0]+1,:,:], axis=0)

sst_anom_1992_2001 = np.mean(sst_anom[np.where(times == np.datetime64('1992-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2001-12-31T12:00'))[0][0]+1,:,:], axis=0)

sst_anom_2002_2011 = np.mean(sst_anom[np.where(times == np.datetime64('2002-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2011-12-31T12:00'))[0][0]+1,:,:], axis=0)

sst_anom_2012_2021 = np.mean(sst_anom[np.where(times == np.datetime64('2012-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2021-12-31T12:00'))[0][0]+1,:,:], axis=0)

#Averaged SSTA over all years (Mean SSTA)
Mean_SSTA_1982_2021 = np.mean(sst_anom, axis=0) #Average SSTA 1982-2021 (Clim 1982-2011)

#Time-averaged Maximum SSTA [1982-2021]
Aver_Max_SSTA_1982_2021 = np.max(sst_anom, axis=0)

#Averaged Mean and Max SSTA over lat,lon (mask  previously applied to original data)
mask_tsT=maskT[np.newaxis,:,:]
max_sst_anom_ts = max_sst_anom_ts+mask_tsT
mean_sst_anom_ts = mean_sst_anom_ts+mask_tsT
Max_SSTA_1982_2021_ts = max_sst_anom_ts.mean(dim=('lon', 'lat'),skipna=True)
Mean_SSTA_1982_2021_ts = mean_sst_anom_ts.mean(dim=('lon', 'lat'),skipna=True)

#Load Core.dataarray to Array of float32 by converting into DataFrame
Max_SSTA_1982_2021_ts = pd.DataFrame(Max_SSTA_1982_2021_ts)      #Visualize core.Dataarray
Max_SSTA_1982_2021_ts = np.squeeze(np.asarray(Max_SSTA_1982_2021_ts))
Mean_SSTA_1982_2021_ts = pd.DataFrame(Mean_SSTA_1982_2021_ts)
Mean_SSTA_1982_2021_ts = np.squeeze(np.asarray(Mean_SSTA_1982_2021_ts))
outfile = r'.\SSTA_ts'
np.savez(outfile, Mean_SSTA_1982_2021_ts=Mean_SSTA_1982_2021_ts, Max_SSTA_1982_2021_ts=Max_SSTA_1982_2021_ts)



Aver_Mean_SSTA_1982_2021 = pd.DataFrame(Mean_SSTA_1982_2021)
Aver_Mean_SSTA_1982_2021 = np.asarray(Mean_SSTA_1982_2021)
Aver_Max_SSTA_1982_2021 = pd.DataFrame(Aver_Max_SSTA_1982_2021)
Aver_Max_SSTA_1982_2021 = np.asarray(Aver_Max_SSTA_1982_2021)
outfile = r'.\Aver_SSTA_1982_2021'
np.savez(outfile,  Aver_Mean_SSTA_1982_2021=Aver_Mean_SSTA_1982_2021, Aver_Max_SSTA_1982_2021=Aver_Max_SSTA_1982_2021)


sst_anom_1982_1991 = pd.DataFrame(sst_anom_1982_1991)
sst_anom_1982_1991 = np.asarray(sst_anom_1982_1991)

sst_anom_1992_2001 = pd.DataFrame(sst_anom_1992_2001)
sst_anom_1992_2001 = np.asarray(sst_anom_1992_2001)

sst_anom_2002_2011 = pd.DataFrame(sst_anom_2002_2011)
sst_anom_2002_2011 = np.asarray(sst_anom_2002_2011)

sst_anom_2012_2021 = pd.DataFrame(sst_anom_2012_2021)
sst_anom_2012_2021 = np.asarray(sst_anom_2012_2021)


outfile_periods = r'.\SSTA_periods'
np.savez(outfile_periods, sst_anom_1982_1991=sst_anom_1982_1991, sst_anom_1992_2001=sst_anom_1992_2001, sst_anom_2002_2011=sst_anom_2002_2011, sst_anom_2012_2021=sst_anom_2012_2021)
















                  