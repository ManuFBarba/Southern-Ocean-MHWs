# -*- coding: utf-8 -*-
"""

############################# MHW ANTARCTIC METRICS ###########################

"""

from netCDF4 import Dataset
import numpy as np
import xarray as xr 

import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import cmocean as cm
import cartopy
import cartopy.crs as ccrs
import matplotlib.path as mpath
import cartopy.feature as cft
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as colors
from datetime import datetime as dt

#Load MHW_metrics_from_MATLAB.py

###Calculating MHW metric on each time period

MHW_cnt_1982_2015 =  MHW_cnt_ts[:,:,0:34]
MHW_cnt_2009_2015 =  MHW_cnt_ts[:,:,27:34]
MHW_cnt_2015_2021 =  MHW_cnt_ts[:,:,33:40]

MHW_dur_1982_2015 =  MHW_dur_ts[:,:,0:34]
MHW_dur_2009_2015 =  MHW_dur_ts[:,:,27:34]
MHW_dur_2015_2021 =  MHW_dur_ts[:,:,33:40]

MHW_cum_1982_2015 =  MHW_cum_ts[:,:,0:34]
MHW_cum_2009_2015 =  MHW_cum_ts[:,:,27:34]
MHW_cum_2015_2021 =  MHW_cum_ts[:,:,33:40]


#Setting Array length
time = np.arange(34)
# Grid
x = np.arange(720)
y = np.arange(100)
# Put the data in a xarray.Dataarray (Vary the array deppending on the MHW metric)
da = xr.DataArray(MHW_cnt_1982_2015, coords=[x, y, time], 
                            dims=['lon', 'lat', 'time'])


#SST dataset
ds = xr.open_mfdataset(r'D:\ICMAN-CSIC\MHW_ANT\datasets\\SST_bimensual\*.nc', parallel=True)

#Subsets of diferent timeperiods
ds_sst_1982_2015=ds.sel(time=slice("1982-01-01", "2015-12-31"))
ds_sst_2009_2015=ds.sel(time=slice("2009-01-01", "2015-12-31"))
ds_sst_2015_2021=ds.sel(time=slice("2015-01-01", "2021-12-31"))

#Reading sst variables

lon = ds_sst_1982_2015['lon'][::10]
lat = ds_sst_1982_2015['lat'][::10]
sst_year  = ds_sst_2009_2015['analysed_sst'][:,::10,::10].groupby('time.year').mean(dim='time',skipna=True)#.load() 


######################
## Mann Kendall test #
######################
from xarrayMannKendall import Mann_Kendall_test
import datetime as datetime
# Print function used.
Mann_Kendall_test

##SST Mann Kendall Test
sst_trends = Mann_Kendall_test(sst_year,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'lat','y':'lon'})


sst_grad = sst_trends.compute()


sst_grad.attrs['title'] = "Sea Surface Temperature trends"
sst_grad.attrs['Description'] = """SST computed from CCI C3S SST data. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall."""
sst_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
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
sst_grad['p'].attrs['long_name'] = "Sea Surface Temperature trends p-value"

sst_grad['p'].attrs['missing_value'] = np.nan
sst_grad['p'].attrs['valid_min'] = np.nanmin(sst_grad['p'])
sst_grad['p'].attrs['valid_max'] = np.nanmax(sst_grad['p'])
sst_grad['p'].attrs['valid_range'] = [np.nanmin(sst_grad['p']),np.nanmax(sst_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in sst_grad.data_vars}

sst_grad.to_netcdf(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends/SST_1982_2015_trends.nc', encoding=encoding)
sst_grad.to_netcdf(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends/SST_2009_2015_trends.nc', encoding=encoding)
sst_grad.to_netcdf(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends/SST_2015_2021_trends.nc', encoding=encoding)


##Reading the previously saved SST trends Datasets
ds_SST_trends_1982_2015 = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends/SST_1982_2015_trends.nc')
SST_1982_2015_trends = ds_SST_trends_1982_2015['trend'][:]
SST_1982_2015_p_value = ds_SST_trends_1982_2015['p'][:]
signif_SST_1982_2015 = ds_SST_trends_1982_2015['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

ds_SST_trends_2009_2015 = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends/SST_2009_2015_trends.nc')
SST_2009_2015_trends = ds_SST_trends_2009_2015['trend'][:]
SST_2009_2015_p_value = ds_SST_trends_2009_2015['p'][:]
signif_SST_2009_2015 = ds_SST_trends_2009_2015['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)


ds_SST_trends_2015_2021 = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends/SST_2015_2021_trends.nc')
SST_2015_2021_trends = ds_SST_trends_2015_2021['trend'][:]
SST_2015_2021_p_value = ds_SST_trends_2015_2021['p'][:]
signif_SST_2015_2021 = ds_SST_trends_2015_2021['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)





##MHWs Mann Kendall Test
MHW_cnt_2009_2015_trends = Mann_Kendall_test(MHW_cnt_2009_2015,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'x':'lon','y':'lat','time':'time'})


MHW_cnt_grad = MHW_cnt_2009_2015_trends.compute()


MHW_cnt_grad.attrs['title'] = "MHW trends"
MHW_cnt_grad.attrs['Description'] = """MHWs computed from CCI C3S SST data. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall."""
MHW_cnt_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
MHW_cnt_grad.attrs['Author'] = "Manuel Fernández Barba"
MHW_cnt_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

MHW_cnt_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

MHW_cnt_grad['trend'].attrs['units'] = r"$^\circ C m^{-1} day^{-1}$"
MHW_cnt_grad['trend'].attrs['name'] = 'trend'
MHW_cnt_grad['trend'].attrs['long_name'] = "MHW trends"

MHW_cnt_grad['trend'].attrs['missing_value'] = np.nan
MHW_cnt_grad['trend'].attrs['valid_min'] = np.nanmin(MHW_cnt_grad['trend'])
MHW_cnt_grad['trend'].attrs['valid_max'] = np.nanmax(MHW_cnt_grad['trend'])
MHW_cnt_grad['trend'].attrs['valid_range'] = [np.nanmin(MHW_cnt_grad['trend']),np.nanmax(MHW_cnt_grad['trend'])]

######################################################

MHW_cnt_grad['signif'].attrs['units'] = ""
MHW_cnt_grad['signif'].attrs['name'] = 'signif'
MHW_cnt_grad['signif'].attrs['long_name'] = "MHWs trends significance"

MHW_cnt_grad['signif'].attrs['missing_value'] = np.nan
MHW_cnt_grad['signif'].attrs['valid_min'] = np.nanmin(MHW_cnt_grad['signif'])
MHW_cnt_grad['signif'].attrs['valid_max'] = np.nanmax(MHW_cnt_grad['signif'])
MHW_cnt_grad['signif'].attrs['valid_range'] = [np.nanmin(MHW_cnt_grad['signif']),np.nanmax(MHW_cnt_grad['signif'])]

######################################################

MHW_cnt_grad['p'].attrs['units'] = ""
MHW_cnt_grad['p'].attrs['name'] = 'p'
MHW_cnt_grad['p'].attrs['long_name'] = "MHWs trends p-value"

MHW_cnt_grad['p'].attrs['missing_value'] = np.nan
MHW_cnt_grad['p'].attrs['valid_min'] = np.nanmin(MHW_cnt_grad['p'])
MHW_cnt_grad['p'].attrs['valid_max'] = np.nanmax(MHW_cnt_grad['p'])
MHW_cnt_grad['p'].attrs['valid_range'] = [np.nanmin(MHW_cnt_grad['p']),np.nanmax(MHW_cnt_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in MHW_cnt_grad.data_vars}

MHW_cnt_grad.to_netcdf('C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_cnt_2009_2015_trends.nc', encoding=encoding)

###Repite the above script for each timeperiod we divide MHWs 


##Reading the previously saved MHW trends Datasets

#Frequency#
ds_MHW_cnt_1982_2015_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_cnt_1982_2015_trends.nc')
MHW_cnt_1982_2015_trends = ds_MHW_cnt_1982_2015_trend['trend'][:]
MHW_cnt_1982_2015_p_value = ds_MHW_cnt_1982_2015_trend['p'][:]
signif_cnt_1982_2015 = ds_MHW_cnt_1982_2015_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

ds_MHW_cnt_2009_2015_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_cnt_2009_2015_trends.nc')
MHW_cnt_2009_2015_trends = ds_MHW_cnt_2009_2015_trend['trend'][:] 
MHW_cnt_2009_2015_p_value = ds_MHW_cnt_2009_2015_trend['p'][:]
signif_cnt_2009_2015 = ds_MHW_cnt_2009_2015_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

ds_MHW_cnt_2015_2021_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_cnt_2015_2021_trends.nc')
MHW_cnt_2015_2021_trends = ds_MHW_cnt_2015_2021_trend['trend'][:] 
MHW_cnt_2015_2021_p_value = ds_MHW_cnt_2015_2021_trend['p'][:]
signif_cnt_2015_2021 = ds_MHW_cnt_2015_2021_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)



#Duration#
ds_MHW_dur_1982_2015_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_dur_1982_2015_trends.nc')
MHW_dur_1982_2015_trends = ds_MHW_dur_1982_2015_trend['trend'][:] 
MHW_dur_1982_2015_p_value = ds_MHW_dur_1982_2015_trend['p'][:]
signif_dur_1982_2015 = ds_MHW_dur_1982_2015_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

ds_MHW_dur_2009_2015_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_dur_2009_2015_trends.nc')
MHW_dur_2009_2015_trends = ds_MHW_dur_2009_2015_trend['trend'][:] 
MHW_dur_2009_2015_p_value = ds_MHW_dur_2009_2015_trend['p'][:]
signif_dur_2009_2015 = ds_MHW_dur_2009_2015_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

ds_MHW_dur_2015_2021_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_dur_2015_2021_trends.nc')
MHW_dur_2015_2021_trends = ds_MHW_dur_2015_2021_trend['trend'][:] 
MHW_dur_2015_2021_p_value = ds_MHW_dur_2015_2021_trend['p'][:]
signif_dur_2015_2021 = ds_MHW_dur_2015_2021_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)



#Cumulative Intensity#
ds_MHW_cum_1982_2015_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_cum_1982_2015_trends.nc')
MHW_cum_1982_2015_trends = ds_MHW_cum_1982_2015_trend['trend'][:] 
MHW_cum_1982_2015_p_value = ds_MHW_cum_1982_2015_trend['p'][:]
signif_cum_1982_2015 = ds_MHW_cum_1982_2015_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

ds_MHW_cum_2009_2015_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_cum_2009_2015_trends.nc')
MHW_cum_2009_2015_trends = ds_MHW_cum_2009_2015_trend['trend'][:] 
MHW_cum_2009_2015_p_value = ds_MHW_cum_2009_2015_trend['p'][:]
signif_cum_2009_2015 = ds_MHW_cum_2009_2015_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

ds_MHW_cum_2015_2021_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\MHWs_trends\MHW_cum_2015_2021_trends.nc')
MHW_cum_2015_2021_trends = ds_MHW_cum_2015_2021_trend['trend'][:] 
MHW_cum_2015_2021_p_value = ds_MHW_cum_2015_2021_trend['p'][:]
signif_cum_2015_2021 = ds_MHW_cum_2015_2021_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)



###Transposing trends and signif

#Frequency#
MHW_cnt_1982_2015_trends= MHW_cnt_1982_2015_trends.T
signif_cnt_1982_2015 = signif_cnt_1982_2015.T

MHW_cnt_2009_2015_trends= MHW_cnt_2009_2015_trends.T
signif_cnt_2009_2015 = signif_cnt_2009_2015.T

MHW_cnt_2015_2021_trends= MHW_cnt_2015_2021_trends.T
signif_cnt_2015_2021 = signif_cnt_2015_2021.T


#Duration#
MHW_dur_1982_2015_trends= MHW_dur_1982_2015_trends.T
signif_dur_1982_2015 = signif_dur_1982_2015.T

MHW_dur_2009_2015_trends= MHW_dur_2009_2015_trends.T
signif_dur_2009_2015 = signif_dur_2009_2015.T

MHW_dur_2015_2021_trends= MHW_dur_2015_2021_trends.T
signif_dur_2015_2021 = signif_dur_2015_2021.T


#Cumulative Intensity#
MHW_cum_1982_2015_trends= MHW_cum_1982_2015_trends.T
signif_cum_1982_2015 = signif_cum_1982_2015.T

MHW_cum_2009_2015_trends= MHW_cum_2009_2015_trends.T
signif_cum_2009_2015 = signif_cum_2009_2015.T

MHW_cum_2015_2021_trends= MHW_cum_2015_2021_trends.T
signif_cum_2015_2021 = signif_cum_2015_2021.T





##########Plotting the South Polar Stereo map with metrics#####################
signif_SST_1982_2015 = np.where(signif_SST_1982_2015 == 0, np.NaN, signif_SST_1982_2015)
signif_SST_2009_2015 = np.where(signif_SST_2009_2015 == 0, np.NaN, signif_SST_2009_2015)
signif_SST_2015_2021 = np.where(signif_SST_2015_2021 == 0, np.NaN, signif_SST_2015_2021)


signif_cnt_1982_2015 = np.where(signif_cnt_1982_2015 == 0, np.NaN, signif_cnt_1982_2015)
signif_cnt_2009_2015 = np.where(signif_cnt_2009_2015 == 0, np.NaN, signif_cnt_2009_2015)
signif_cnt_2015_2021 = np.where(signif_cnt_2015_2021 == 0, np.NaN, signif_cnt_2015_2021)


signif_dur_1982_2015 = np.where(signif_dur_1982_2015 == 0, np.NaN, signif_dur_1982_2015)
signif_dur_2009_2015 = np.where(signif_dur_2009_2015 == 0, np.NaN, signif_dur_2009_2015)
signif_dur_2015_2021 = np.where(signif_dur_2015_2021 == 0, np.NaN, signif_dur_2015_2021)


signif_cum_1982_2015 = np.where(signif_cum_1982_2015 == 0, np.NaN, signif_cum_1982_2015)
signif_cum_2009_2015 = np.where(signif_cum_2009_2015 == 0, np.NaN, signif_cum_2009_2015)
signif_cum_2015_2021 = np.where(signif_cum_2015_2021 == 0, np.NaN, signif_cum_2015_2021)



#################
### MAX SSTAA ###
#################
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='papayawhip', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)

cmap=plt.cm.YlOrRd
levels = [0,1,2,3,4,5,6,7,8]
# levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
p1 = plt.contourf(lon, lat, Max_SSTA_2015_2021+1.5+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 

# cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([0, 2, 4, 6, 8])
# #cbar.ax.set_ylabel('[$^\circ$C]', fontsize=25)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('c) Max SSTA [$^\circ$C]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


######### SST trends ##########
signif_sst = signif_SST_2015_2021

#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)
fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)
#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='papayawhip', linewidth=0.5)
ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')
ax.add_feature(ice_50m)
 
cmap=plt.cm.RdYlBu_r
levels = [-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25]
p1 = plt.contourf(lon, lat, SST_2015_2021_trends+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_sst[::4,::2]+mask[::4,::2], color='black',linewidth=2,marker='o', alpha=1,transform=ccrs.Geodetic())


# cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([-0.2, -0.1, 0, 0.1, 0.2])


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('f) SST trends [$^{\circ}C\ year^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)



#############
### Frequency ###
#############

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

  
cmap=plt.cm.YlOrRd
levels = [1,1.5,2,2.5,3,3.5,4]
p1 = plt.contourf(lon, lat, np.nanmean(MHW_cnt_ts[:,:,33:40], axis=2)+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([1, 2, 3, 4])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('i) Frequency [$number$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)



                     ######### Frequency trend ##########
signif_freq = signif_cnt_2015_2021

signif_freq[0,:] = signif_freq[1,:]
signif_freq[719,:] = signif_freq[718,:]
signif_freq[:,0] = signif_freq[:,1]
signif_freq[:,99] = signif_freq[:,98]

MHW_cnt_1982_2015_trends[0,:] = MHW_cnt_1982_2015_trends[1,:]
MHW_cnt_1982_2015_trends[719,:] = MHW_cnt_1982_2015_trends[718,:]
MHW_cnt_1982_2015_trends[:,0] = MHW_cnt_1982_2015_trends[:,1]
MHW_cnt_1982_2015_trends[:,99] = MHW_cnt_1982_2015_trends[:,98]

MHW_cnt_2009_2015_trends[0,:] = MHW_cnt_2009_2015_trends[1,:]
MHW_cnt_2009_2015_trends[719,:] = MHW_cnt_2009_2015_trends[718,:]
MHW_cnt_2009_2015_trends[:,0] = MHW_cnt_2009_2015_trends[:,1]
MHW_cnt_2009_2015_trends[:,99] = MHW_cnt_2009_2015_trends[:,98]

MHW_cnt_2015_2021_trends[0,:] = MHW_cnt_2015_2021_trends[1,:]
MHW_cnt_2015_2021_trends[719,:] = MHW_cnt_2015_2021_trends[718,:]
MHW_cnt_2015_2021_trends[:,0] = MHW_cnt_2015_2021_trends[:,1]
MHW_cnt_2015_2021_trends[:,99] = MHW_cnt_2015_2021_trends[:,98]
                     
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

 
cmap=plt.cm.RdYlBu_r
levels = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
p1 = plt.contourf(lon, lat, MHW_cnt_2015_2021_trends+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_freq[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())

cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-1, -0.5, 0, 0.5, 1])


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('l) Frequency trend [$number\ year^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


# outfile = r'C:\ICMAN-CSIC\MHW_ANT\Figures_MHW\MHW_dur.png'
# fig.savefig(outfile, bbox_inches='tight', pad_inches=0.5)


################
### Duration ###
################
#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='papayawhip', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)


cmap=plt.cm.YlOrRd
levels = [2.5,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30]
p1 = plt.contourf(lon, lat, np.nanmean(MHW_dur_ts[:,:,0:34], axis=2)+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([5, 10, 15, 20, 25, 30])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('m) Duration [$days$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)




######### Duration trend ##########
#MHW_dur_dtr = np.where(MHW_dur_dtr <= 0.06, 1, np.NaN)
signif_dur = signif_dur_2015_2021

#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)
fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)
#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='papayawhip', linewidth=0.5)
ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')
ax.add_feature(ice_50m)
 
cmap=plt.cm.RdYlBu_r
levels = [-5, -3.75, -2.5, -1.25, 0, 1.25, 2.5, 3.75, 5]
p1 = plt.contourf(lon, lat, MHW_dur_2015_2021_trends+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_dur[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-2.5,-1.25,0,1.25, 2.5])


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('r) Duration trend [$days\ year^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


# outfile = r'C:\ICMAN-CSIC\MHW_ANT\Figures_MHW\MHW_dur.png'
# fig.savefig(outfile, bbox_inches='tight', pad_inches=0.5)


############################
### Cumulative Intensity ###
############################

projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='papayawhip', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)


cmap=plt.cm.YlOrRd
levels = [0,5,10,15,20,25,30,35,40,45,50]
p1 = plt.contourf(lon, lat, np.nanmean(MHW_cum_ts[:,:,0:34], axis=2)+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([0, 10, 20, 30,40, 50])


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title( 'u) Cumulative intensity [$^{\circ}C\ days$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)



######### Cumulative Intensity trend ##########

signif_cum = signif_cum_2015_2021
                        
#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='papayawhip', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)


cmap=plt.cm.RdYlBu_r
levels = [-10,-7.5,-5,-2.5,0,2.5,5,7.5,10]
p1 = plt.contourf(lon, lat, MHW_cum_2015_2021_trends+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_cum[::4,::2]+mask[::4,::2], color='black',linewidth=2,marker='o', alpha=1,transform=ccrs.Geodetic())

# cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([-10, -5, 0, 5, 10])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('x) Cum intensity trend [$^{\circ}C\ days\ year^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


# outfile = r'C:\ICMAN-CSIC\MHW_ANT\Figures_MHW\MHW_dur.png'
# fig.savefig(outfile, bbox_inches='tight', pad_inches=0.5)

