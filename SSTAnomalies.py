# -*- coding: utf-8 -*-
"""

############################# SST ANOMALIES  AND TREND ########################

"""

# Load required modules
 
import netCDF4 as nc
import numpy as np
import pandas as pd


from datetime import datetime as dt
from netCDF4 import Dataset 
import xarray as xr 
import ecoliver as ecj

from dateutil.relativedelta import relativedelta

import seaborn as sns
import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import matplotlib.path as mpath
import cartopy.feature as cft
from matplotlib.colors import Normalize as norm
from matplotlib.colors import LinearSegmentedColormap as linearsegm


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

# ds = xr.open_mfdataset(r'.\SST_bimensual\*.nc', parallel=True)
ds = xr.open_dataset(r'.\SST_ANT_1982-2021_40.nc')
#ds = Dataset(r'.\SST_ANT_1982-2021_40.nc')

#ds=ds.sel(lat=slice(latmin,latmax),lon=slice(lonmin,lonmax),time=slice("1982-01-01", "2021-12-30"))

lon = ds['lon'][::10]
lat = ds['lat'][::10]
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
sst_clim=ds_clim['analysed_sst'][:,::10,::10].groupby('time.month').mean(dim='time', skipna=True)


#Compute SST Anomaly
#sst_anom_full=ds['analysed_sst'].groupby('time.month') - sst_clim_full
sst_anom=ds['analysed_sst'][:,::10,::10].groupby('time.month') - sst_clim

max_sst_anom_ts=sst_anom.groupby('time.year').max(dim='time',skipna=True).T[()]
mean_sst_anom_ts=sst_anom.groupby('time.year').mean(dim='time',skipna=True)


#SSTA in concrete time periods
# sst_anom_1982_1991 = np.mean(sst_anom[np.where(times == np.datetime64('1982-01-01T12:00'))[0][0]:np.where(times == np.datetime64('1991-12-31T12:00'))[0][0]+1,:,:], axis=0)

# sst_anom_1992_2001 = np.mean(sst_anom[np.where(times == np.datetime64('1992-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2001-12-31T12:00'))[0][0]+1,:,:], axis=0)

# sst_anom_2002_2011 = np.mean(sst_anom[np.where(times == np.datetime64('2002-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2011-12-31T12:00'))[0][0]+1,:,:], axis=0)

# sst_anom_2012_2021 = np.mean(sst_anom[np.where(times == np.datetime64('2012-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2021-12-31T12:00'))[0][0]+1,:,:], axis=0)

############

sst_anom_1982_2015 = np.mean(sst_anom[np.where(times == np.datetime64('1982-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2015-12-31T12:00'))[0][0]+1,:,:], axis=0)

sst_anom_2009_2015 = np.mean(sst_anom[np.where(times == np.datetime64('2009-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2015-12-31T12:00'))[0][0]+1,:,:], axis=0)

sst_anom_2015_2021 = np.mean(sst_anom[np.where(times == np.datetime64('2015-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2021-12-31T12:00'))[0][0]+1,:,:], axis=0)


sst_anom_1982_2015 = pd.DataFrame(sst_anom_1982_2015)
sst_anom_1982_2015 = np.asarray(sst_anom_1982_2015)

sst_anom_2009_2015 = pd.DataFrame(sst_anom_2009_2015)
sst_anom_2009_2015 = np.asarray(sst_anom_2009_2015)

sst_anom_2015_2021 = pd.DataFrame(sst_anom_2015_2021)
sst_anom_2015_2021 = np.asarray(sst_anom_2015_2021)

outfile = r'.\SSTA_1982_2015_2021'
np.savez(outfile, sst_anom_1982_2015=sst_anom_1982_2015, sst_anom_2009_2015=sst_anom_2009_2015, sst_anom_2015_2021=sst_anom_2015_2021)



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





#
##Plotting SST Anomaly over time
#
global_anom=sst_anom.mean(dim=('lon', 'lat')) #Average SSTA for entire study area
global_anom_roll=sst_anom.mean(dim=('lon', 'lat')).rolling(time=5).mean(dim='time') #Aplying rolling mean

nt=global_anom.shape
base = dt(1982, 1, 1)
arr_time = np.array([base + relativedelta(days=+i) for i in range(nt[0])])



fig = plt.figure(figsize=(15, 8))
ax1=fig.add_subplot(111)
ax1.plot_date(arr_time, global_anom, fmt='gray', tz=None, xdate=True, ydate=False,label='monthly CCI')
ax1.plot_date(arr_time, global_anom_roll, fmt='blue', tz=None, xdate=True, ydate=False,label='rolling mean')


ax1.set_ylabel('SST [$^\circ$C] ')
#ax2=ax1.twinx()
#ax2.plot_date(arr_time[1:-1], m_sss[1:-1], fmt='r', tz=None, xdate=True, ydate=False,label='SMOS')
#ax2.plot_date(arr_time, m_sss_mod, fmt='r', tz=None, xdate=True, ydate=False,label='Model',ls='dashed')
ax1.set_xlabel('time')
#ax2.set_ylabel('SST [psu]')
#ax2.tick_params(axis='y', colors='red')
#ax2.yaxis.label.set_color('red')
lg = plt.legend(loc=(0.35,-0.21), ncol=2, fancybox=True,frameon=True, shadow=True, borderaxespad=0.)
#leg_width(lg,fs=5)
#fig.subplots_adjust(bottom=0.9)
plt.title('Southern Ocean SST anomaly (1982-2021)')
fig.tight_layout()
dfig=r'.\Figures_MHW'
figname='\SST_anaomaly_1982-2021.png'
plt.savefig(dfig+figname)






#2D Array of x and y (lon/lat) locations
LON, LAT = np.meshgrid(lon, lat) 



#
##Plotting the South Polar Stereo SST and SSTA maps
#
                          #################
                          ######SST########
                          #################                     
    
plt.clf()
#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)

#cmap=plt.cm.YlOrRd  
#cmap=plt.cm.RdBu_r  
cmap = 'Spectral_r'
p1 = plt.pcolormesh(LON, LAT, Aver_SST, vmin=-2, vmax=14, cmap=cmap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-2, 0, 2, 4, 6, 8, 10, 12, 14])
cbar.ax.set_ylabel('[$^\circ$C]', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='10m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# ax.set_title('Mean SST [1982-2021]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'.\Mean_SST_1982_2021.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

                        #################
                        ####Mean SSTA####
                        #################

#Make lon, lat grids for representation purposes

#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)

cmap=plt.cm.RdBu_r
# n=100
# x=0.5
# lower = cmap(np.linspace(0, x, n))
# white = np.ones((20,4))
# upper = cmap(np.linspace(1-x, 1, n))
# colors = np.vstack((lower, white, upper))
# tmap = linearsegm.from_list('map_white', colors)
# cmap=cm.cm.balance
levels = [-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]
# levels = [-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25]
p1 = plt.contourf(LON, LAT, sst_anom_2015_2021+maskT, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-0.4, -0.2, 0, 0.2, 0.4])
cbar.ax.set_ylabel('[$^\circ$C]', fontsize=35, rotation='vertical', labelpad=20)


ax.set_extent([-280, 80, -80, -40.1], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('(c) SSTA 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'.\SSTA_2015_2021.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

                          ################
                          # Maximum SSTA #
                          ################


Aver_Max_SSTA_1982_2021 = Aver_Max_SSTA_1982_2021.T
Aver_Max_SSTA_1982_2021 = Aver_Max_SSTA_1982_2021[::10,::10]

#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

cmap=plt.cm.YlOrRd
#cmap=plt.cm.RdBu_r  
#cmap = 'Spectral_r'
levels = [0,1,2,3,4,5,6,7,8]

p1 = plt.contourf(lon, lat, Aver_Max_SSTA_1982_2021+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 2, 4, 6, 8])
cbar.ax.set_ylabel('$^\circ$C', fontsize=30)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('a   Maximum SSTA', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'.\Max_SSTA.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

############################## MAP YEARLY TREND SST ###########################


##Mann Kendall test
from xarrayMannKendall import Mann_Kendall_test
import datetime as datetime
##Read data and pick variables
ds = xr.open_mfdataset(r'.\SST_bimensual\*.nc', parallel=True)

lon = ds['lon'][::10]
lat = ds['lat'][::10]
sst_year  = ds['analysed_sst'][:,::10,::10].groupby('time.year').mean(dim='time',skipna=True)#.load() 


# Print function used.
Mann_Kendall_test

sst_trends = Mann_Kendall_test(sst_year,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'lat','y':'lon'})


sst_grad = sst_trends.compute()


sst_grad.attrs['title'] = "Sea Surface Temperature trends"
sst_grad.attrs['Description'] = """SST computed from CCI C3S SST data. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall."""
sst_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2024: \n "
sst_grad.attrs['Author'] = "Manuel Fernández Barba"
sst_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

sst_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

sst_grad['trend'].attrs['units'] = r"$^\circ C m^{-1} day^{-1}$"
sst_grad['trend'].attrs['name'] = 'trend'
sst_grad['trend'].attrs['long_name'] = "Sea Surface Temperature trends"

sst_grad['trend'].attrs['missing_value'] = np.nan
sst_grad['trend'].attrs['valid_min'] = np.nanmin(sst_grad['trend'])
sst_grad['trend'].attrs['valid_max'] = np.nanmax(sst_grad['trend'])
sst_grad['trend'].attrs['valid_range'] = [np.nanmin(sst_grad['trend']),np.nanmax(sst_grad['trend'])]

######################################################

sst_grad['signif'].attrs['units'] = ""
sst_grad['signif'].attrs['name'] = 'signif'
sst_grad['signif'].attrs['long_name'] = "Sea Surface Temperature trends significance"

sst_grad['signif'].attrs['missing_value'] = np.nan
sst_grad['signif'].attrs['valid_min'] = np.nanmin(sst_grad['signif'])
sst_grad['signif'].attrs['valid_max'] = np.nanmax(sst_grad['signif'])
sst_grad['signif'].attrs['valid_range'] = [np.nanmin(sst_grad['signif']),np.nanmax(sst_grad['signif'])]

######################################################

sst_grad['p'].attrs['units'] = ""
sst_grad['p'].attrs['name'] = 'p'
sst_grad['p'].attrs['long_name'] = "Sea Surface Temperature trends p"

sst_grad['p'].attrs['missing_value'] = np.nan
sst_grad['p'].attrs['valid_min'] = np.nanmin(sst_grad['p'])
sst_grad['p'].attrs['valid_max'] = np.nanmax(sst_grad['p'])
sst_grad['p'].attrs['valid_range'] = [np.nanmin(sst_grad['p']),np.nanmax(sst_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in sst_grad.data_vars}

sst_grad.to_netcdf('./SST_trends.nc', encoding=encoding)



trends = sst_grad.trend*10 # Convert to trends per decade

signif = sst_grad.signif


##Reading the previously saved SST trends Dataset

ds_trend = Dataset(r'.\SST_trends.nc')
SST_trends = ds_trend['trend'][:]*10 # Convert to trends per decade
SST_p_value = ds_trend['p'][:]
signif = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]


signif = np.where(signif == 0, np.NaN, signif)


##Plotting the SST trends
#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)

#cmap=plt.cm.YlOrRd     
cmap=plt.cm.RdYlBu_r
levels = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3]
p1 = plt.contourf(lon, lat, SST_trends+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('b) SST trend [$^{\circ}C\ decade^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'.\SST_trends.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



                     #### Max SSTA trends #####

# Print function used.
Mann_Kendall_test

sst_trends = Mann_Kendall_test(max_sst_anom_ts,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'lat','y':'lon'})


sst_grad = sst_trends.compute()


sst_grad.attrs['title'] = "Sea Surface Temperature trends"
sst_grad.attrs['Description'] = """SST computed from CCI C3S SST data. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall."""
sst_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2024: \n "
sst_grad.attrs['Author'] = "Manuel Fernández Barba"
sst_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

sst_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

sst_grad['trend'].attrs['units'] = r"$^\circ C m^{-1} day^{-1}$"
sst_grad['trend'].attrs['name'] = 'trend'
sst_grad['trend'].attrs['long_name'] = "Sea Surface Temperature trends"

sst_grad['trend'].attrs['missing_value'] = np.nan
sst_grad['trend'].attrs['valid_min'] = np.nanmin(sst_grad['trend'])
sst_grad['trend'].attrs['valid_max'] = np.nanmax(sst_grad['trend'])
sst_grad['trend'].attrs['valid_range'] = [np.nanmin(sst_grad['trend']),np.nanmax(sst_grad['trend'])]

######################################################

sst_grad['signif'].attrs['units'] = ""
sst_grad['signif'].attrs['name'] = 'signif'
sst_grad['signif'].attrs['long_name'] = "Sea Surface Temperature trends significance"

sst_grad['signif'].attrs['missing_value'] = np.nan
sst_grad['signif'].attrs['valid_min'] = np.nanmin(sst_grad['signif'])
sst_grad['signif'].attrs['valid_max'] = np.nanmax(sst_grad['signif'])
sst_grad['signif'].attrs['valid_range'] = [np.nanmin(sst_grad['signif']),np.nanmax(sst_grad['signif'])]

######################################################

sst_grad['p'].attrs['units'] = ""
sst_grad['p'].attrs['name'] = 'p'
sst_grad['p'].attrs['long_name'] = "Sea Surface Temperature trends p"

sst_grad['p'].attrs['missing_value'] = np.nan
sst_grad['p'].attrs['valid_min'] = np.nanmin(sst_grad['p'])
sst_grad['p'].attrs['valid_max'] = np.nanmax(sst_grad['p'])
sst_grad['p'].attrs['valid_range'] = [np.nanmin(sst_grad['p']),np.nanmax(sst_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in sst_grad.data_vars}

sst_grad.to_netcdf('./Max_SSTA_trends.nc', encoding=encoding)


##Reading the previously saved Max SSTA trends Dataset
ds_trend = Dataset(r'./Max_SSTA_trends.nc')
Max_SSTA_trends = ds_trend['trend'][:]*10 #Convert to trends per decade
Max_SSTA_p_value = ds_trend['p'][:]
signif_Max_SSTA = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

signif_Max_SSTA = np.where(signif_Max_SSTA == 0, np.NaN, signif_Max_SSTA)
signif_Max_SSTA =signif_Max_SSTA.T

Max_SSTA_trends=Max_SSTA_trends.T




##Plotting the Max SSTA trends

#2D Array of x and y (lon/lat) locations
LON, LAT = np.meshgrid(lon, lat) 

#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)


cmap=plt.cm.RdYlBu_r
# levels = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3]
levels = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]
p1 = plt.contourf(lon, lat, (Max_SSTA_trends.T)+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_Max_SSTA.T[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.4, -0.2, 0, 0.2, 0.4])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('b) Max SSTA trend [$^{\circ}C\ decade^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'.\MHW_Max_SSTA_tr.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

