# -*- coding: utf-8 -*-
"""

############################## Modelled Nutrients ######################################

"""


# Load required modules
 
import netCDF4 as nc
import numpy as np
import pandas as pd

import os
from netCDF4 import Dataset 
import xarray as xr 


import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import matplotlib.path as mpath
import cartopy.feature as cft
from matplotlib import ticker
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap as linearsegm

ds = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Nutrients\*.nc', parallel=True)

lon = ds['longitude'][:]
lat = ds['latitude'][:]
times = ds['time'][:]

#NO3#
no3_year=ds['no3'][:,0,:,:].groupby('time.year').mean(dim='time',skipna=True)#.load()

no3_ts_1994_2015 = np.nanmean(no3_year[0:22,:,:], axis=0) 
no3_ts_2009_2015 = np.nanmean(no3_year[15:22,:,:], axis=0) 
no3_ts_2015_2020 = np.nanmean(no3_year[21:28,:,:], axis=0) 


#PO4#
po4_year=ds['po4'][:,0,:,:].groupby('time.year').mean(dim='time',skipna=True)#.load()

po4_ts_1998_2015 = np.nanmean(po4_year[4:22,:,:], axis=0) 
po4_ts_2009_2015 = np.nanmean(po4_year[15:22,:,:], axis=0) 
po4_ts_2015_2020 = np.nanmean(po4_year[21:28,:,:], axis=0) 

#Si#
si_year=ds['si'][:,0,:,:].groupby('time.year').mean(dim='time',skipna=True)#.load()

si_ts_1994_2015 = np.nanmean(si_year[0:22,:,:], axis=0) 
si_ts_2009_2015 = np.nanmean(si_year[15:22,:,:], axis=0) 
si_ts_2015_2020 = np.nanmean(si_year[21:28,:,:], axis=0) 



###################################################
## Plotting the South Polar Stereo Nutrients map ##
###################################################

#Make lon, lat grids for representation purposes

LON, LAT = np.meshgrid(lon, lat)

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

# cmap='viridis'
cmap=cm.cm.matter
# cmap=cm.cm.algae
# cmap=plt.cm.YlOrRd
# levels = [0, 5, 10, 15, 20, 25, 30, 35, 40,45, 50, 55,60, 65, 70, 75, 80, 85, 90] #Si
# levels = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30] #NO3
levels = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2] #PO4
p1 = plt.contourf(LON, LAT, po4_ts_2015_2020, levels, cmap=cmap, extend='max', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='max', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) #Si
# cbar.ax.get_yaxis().set_ticks([0, 5, 10, 15, 20, 25, 30]) #NO3
cbar.ax.get_yaxis().set_ticks([0.4, 0.8, 1.2, 1.6, 2]) #PO4
cbar.ax.set_ylabel('$mmol·m^{-3}$', fontsize=35)


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('[PO4] 2015-2020', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)



##Compute a Mann Kendall test to calculate Nutrients trends
##Mann Kendall test
from xarrayMannKendall import Mann_Kendall_test
import datetime as datetime


Si_1994_2015 = si_year[0:22,:,:]
Si_2009_2015 = si_year[15:22,:,:]
Si_2015_2021 = si_year[21:28,:,:]

no3_1994_2015 = no3_year[0:22,:,:]
no3_2009_2015 = no3_year[15:22,:,:]
no3_2015_2021 = no3_year[21:28,:,:]

po4_1998_2015 = po4_year[4:22,:,:]
po4_2009_2015 = po4_year[15:22,:,:]
po4_2015_2021 = po4_year[21:28,:,:]



###Compute Si trends

# Print function used.
Mann_Kendall_test

#Computing it
Si_trends = Mann_Kendall_test(Si_1994_2015,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'latitude','y':'longitude'})

Si_grad = Si_trends.compute()


Si_grad.attrs['title'] = "Silicate trends"
Si_grad.attrs['Description'] = """Mole concentration of silicate a in sea water Si[mmol/m3]. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall."""
Si_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
Si_grad.attrs['Author'] = "Manuel Fernández Barba"
Si_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

Si_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

Si_grad['trend'].attrs['units'] = r'$mmol·m^{-3}·year^{-1}$'
Si_grad['trend'].attrs['name'] = 'trend'
Si_grad['trend'].attrs['long_name'] = "Si trends"

Si_grad['trend'].attrs['missing_value'] = np.nan
Si_grad['trend'].attrs['valid_min'] = np.nanmin(Si_grad['trend'])
Si_grad['trend'].attrs['valid_max'] = np.nanmax(Si_grad['trend'])
Si_grad['trend'].attrs['valid_range'] = [np.nanmin(Si_grad['trend']),np.nanmax(Si_grad['trend'])]

######################################################

Si_grad['signif'].attrs['units'] = ""
Si_grad['signif'].attrs['name'] = 'signif'
Si_grad['signif'].attrs['long_name'] = "Si trends significance"

Si_grad['signif'].attrs['missing_value'] = np.nan
Si_grad['signif'].attrs['valid_min'] = np.nanmin(Si_grad['signif'])
Si_grad['signif'].attrs['valid_max'] = np.nanmax(Si_grad['signif'])
Si_grad['signif'].attrs['valid_range'] = [np.nanmin(Si_grad['signif']),np.nanmax(Si_grad['signif'])]

######################################################

Si_grad['p'].attrs['units'] = ""
Si_grad['p'].attrs['name'] = 'p'
Si_grad['p'].attrs['long_name'] = "Si trends p"

Si_grad['p'].attrs['missing_value'] = np.nan
Si_grad['p'].attrs['valid_min'] = np.nanmin(Si_grad['p'])
Si_grad['p'].attrs['valid_max'] = np.nanmax(Si_grad['p'])
Si_grad['p'].attrs['valid_range'] = [np.nanmin(Si_grad['p']),np.nanmax(Si_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in Si_grad.data_vars}

Si_grad.to_netcdf('C:\ICMAN-CSIC\MHW_ANT\datasets_40\Si_1994_2015_trends.nc', encoding=encoding)


#Si#
##Reading the previously saved Si trends Dataset
ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\nutrients_trends\Si_1994_2015_trends.nc')
Si_1994_2015_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
Si_1994_2015_p_value = ds_trend['p'][:]
Si_signif_1994_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

Si_signif_1994_2015 = np.where(Si_signif_1994_2015 == 0, np.NaN, Si_signif_1994_2015)
Si_signif_1994_2015 =Si_signif_1994_2015.T
Si_1994_2015_trends=Si_1994_2015_trends.T


ds_trend = Dataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/Si_2009_2015_trends.nc')
Si_2009_2015_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
Si_2009_2015_p_value = ds_trend['p'][:]
Si_signif_2009_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

Si_signif_2009_2015 = np.where(Si_signif_2009_2015 == 0, np.NaN, Si_signif_2009_2015)
Si_signif_2009_2015 =Si_signif_2009_2015.T
Si_2009_2015_trends=Si_2009_2015_trends.T


ds_trend = Dataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/Si_2015_2021_trends.nc')
Si_2015_2021_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
Si_2015_2021_p_value = ds_trend['p'][:]
Si_signif_2015_2021 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

Si_signif_2015_2021 = np.where(Si_signif_2015_2021 == 0, np.NaN, Si_signif_2015_2021)
Si_signif_2015_2021 =Si_signif_2015_2021.T
Si_2015_2021_trends=Si_2015_2021_trends.T



####################################################
## Plotting the South Polar Stereo Si trends map ###
####################################################

#Make lon, lat grids for representation purposes
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

ax.add_feature(ice_50m)


# cmap=cm.cm.curl
cmap=cm.cm.delta_r
n=100
x=0.5
lower = cmap(np.linspace(0, x, n))
white = np.ones((20,4))
upper = cmap(np.linspace(1-x, 1, n))
colors = np.vstack((lower, white, upper))
tmap = linearsegm.from_list('map_white', colors)


levels = [-2,-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
p1 = plt.contourf(LON, LAT, Si_2015_2021_trends, levels, cmap=tmap, extend='both', transform=ccrs.PlateCarree()) 
# p2=plt.scatter(LON[::6,::9],LAT[::6,::9], Si_signif_2015_2021[::6,::9], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-2, -1, 0, 1, 2])
cbar.ax.set_ylabel('$mmol·m^{-3}·year^{-1}$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('i) [Si] trends 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\Si_trends_2015_2021_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



###Compute NO3 trends

# Print function used.
Mann_Kendall_test

#Computing it
no3_trends = Mann_Kendall_test(no3_2015_2021,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'latitude','y':'longitude'})

no3_grad = no3_trends.compute()


no3_grad.attrs['title'] = "Nitrate trends"
no3_grad.attrs['Description'] = """Mole concentration of nitrate a in sea water NO3[mmol/m3]. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall"""
no3_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
no3_grad.attrs['Author'] = "Manuel Fernández Barba"
no3_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

no3_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

no3_grad['trend'].attrs['units'] = r'$mmol·m^{-3}·year^{-1}$'
no3_grad['trend'].attrs['name'] = 'trend'
no3_grad['trend'].attrs['long_name'] = "NO3 trends"

no3_grad['trend'].attrs['missing_value'] = np.nan
no3_grad['trend'].attrs['valid_min'] = np.nanmin(no3_grad['trend'])
no3_grad['trend'].attrs['valid_max'] = np.nanmax(no3_grad['trend'])
no3_grad['trend'].attrs['valid_range'] = [np.nanmin(no3_grad['trend']),np.nanmax(no3_grad['trend'])]

######################################################

no3_grad['signif'].attrs['units'] = ""
no3_grad['signif'].attrs['name'] = 'signif'
no3_grad['signif'].attrs['long_name'] = "NO3 trends significance"

no3_grad['signif'].attrs['missing_value'] = np.nan
no3_grad['signif'].attrs['valid_min'] = np.nanmin(no3_grad['signif'])
no3_grad['signif'].attrs['valid_max'] = np.nanmax(no3_grad['signif'])
no3_grad['signif'].attrs['valid_range'] = [np.nanmin(no3_grad['signif']),np.nanmax(no3_grad['signif'])]

######################################################

no3_grad['p'].attrs['units'] = ""
no3_grad['p'].attrs['name'] = 'p'
no3_grad['p'].attrs['long_name'] = "NO3 trends p"

no3_grad['p'].attrs['missing_value'] = np.nan
no3_grad['p'].attrs['valid_min'] = np.nanmin(no3_grad['p'])
no3_grad['p'].attrs['valid_max'] = np.nanmax(no3_grad['p'])
no3_grad['p'].attrs['valid_range'] = [np.nanmin(no3_grad['p']),np.nanmax(no3_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in no3_grad.data_vars}

no3_grad.to_netcdf('C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/NO3_2015_2021_trends.nc', encoding=encoding)


##Reading the previously saved CHL trends Dataset
ds_trend = Dataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/NO3_1994_2015_trends.nc')
NO3_1994_2015_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
NO3_1994_2015_p_value = ds_trend['p'][:]
NO3_signif_1994_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

NO3_signif_1994_2015 = np.where(NO3_signif_1994_2015 == 0, np.NaN, NO3_signif_1994_2015)
NO3_signif_1994_2015 =NO3_signif_1994_2015.T
NO3_1994_2015_trends=NO3_1994_2015_trends.T


ds_trend = Dataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/NO3_2009_2015_trends.nc')
NO3_2009_2015_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
NO3_2009_2015_p_value = ds_trend['p'][:]
NO3_signif_2009_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

NO3_signif_2009_2015 = np.where(NO3_signif_2009_2015 == 0, np.NaN, NO3_signif_2009_2015)
NO3_signif_2009_2015 =NO3_signif_2009_2015.T
NO3_2009_2015_trends=NO3_2009_2015_trends.T


ds_trend = Dataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/NO3_2015_2021_trends.nc')
NO3_2015_2021_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
NO3_2015_2021_p_value = ds_trend['p'][:]
NO3_signif_2015_2021 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

NO3_signif_2015_2021 = np.where(NO3_signif_2015_2021 == 0, np.NaN, NO3_signif_2015_2021)
NO3_signif_2015_2021 =NO3_signif_2015_2021.T
NO3_2015_2021_trends=NO3_2015_2021_trends.T



####################################################
## Plotting the South Polar Stereo NO3 trends map ##
####################################################

#Make lon, lat grids for representation purposes
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

ax.add_feature(ice_50m)


# cmap=cm.cm.curl
cmap=cm.cm.delta_r
n=100
x=0.5
lower = cmap(np.linspace(0, x, n))
white = np.ones((20,4))
upper = cmap(np.linspace(1-x, 1, n))
colors = np.vstack((lower, white, upper))
tmap = linearsegm.from_list('map_white', colors)

# levels = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
levels = [-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
p1 = plt.contourf(LON, LAT, NO3_2015_2021_trends, levels, cmap=tmap, extend='both', transform=ccrs.PlateCarree()) 
# p2=plt.scatter(LON[::6,::9],LAT[::6,::9], NO3_signif_1994_2015[::6,::9], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-0.6, -0.3, 0, 0.3, 0.6])
cbar.ax.set_ylabel('$mmol·m^{-3}·year^{-1}$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('l) [NO3] trends 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\NO3_trends_2015_2021_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



###Compute PO4 trends

# Print function used.
Mann_Kendall_test

#Computing it
po4_trends = Mann_Kendall_test(po4_2015_2021,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'latitude','y':'longitude'})

po4_grad = po4_trends.compute()


po4_grad.attrs['title'] = "Phosphate trends"
po4_grad.attrs['Description'] = """Mole concentration of phosphate a in sea water PO4[mmol/m3]. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall."""
po4_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
po4_grad.attrs['Author'] = "Manuel Fernández Barba"
po4_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

po4_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

po4_grad['trend'].attrs['units'] = r'$mmol·m^{-3}·year^{-1}$'
po4_grad['trend'].attrs['name'] = 'trend'
po4_grad['trend'].attrs['long_name'] = "PO4 trends"

po4_grad['trend'].attrs['missing_value'] = np.nan
po4_grad['trend'].attrs['valid_min'] = np.nanmin(po4_grad['trend'])
po4_grad['trend'].attrs['valid_max'] = np.nanmax(po4_grad['trend'])
po4_grad['trend'].attrs['valid_range'] = [np.nanmin(po4_grad['trend']),np.nanmax(po4_grad['trend'])]

######################################################

po4_grad['signif'].attrs['units'] = ""
po4_grad['signif'].attrs['name'] = 'signif'
po4_grad['signif'].attrs['long_name'] = "PO4 trends significance"

po4_grad['signif'].attrs['missing_value'] = np.nan
po4_grad['signif'].attrs['valid_min'] = np.nanmin(po4_grad['signif'])
po4_grad['signif'].attrs['valid_max'] = np.nanmax(po4_grad['signif'])
po4_grad['signif'].attrs['valid_range'] = [np.nanmin(po4_grad['signif']),np.nanmax(po4_grad['signif'])]

######################################################

po4_grad['p'].attrs['units'] = ""
po4_grad['p'].attrs['name'] = 'p'
po4_grad['p'].attrs['long_name'] = "PO4 trends p"

po4_grad['p'].attrs['missing_value'] = np.nan
po4_grad['p'].attrs['valid_min'] = np.nanmin(po4_grad['p'])
po4_grad['p'].attrs['valid_max'] = np.nanmax(po4_grad['p'])
po4_grad['p'].attrs['valid_range'] = [np.nanmin(po4_grad['p']),np.nanmax(po4_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in po4_grad.data_vars}

po4_grad.to_netcdf('C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/PO4_2015_2021_trends.nc', encoding=encoding)


##Reading the previously saved PO4 trends Dataset
ds_trend = Dataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/PO4_1998_2015_trends.nc')
PO4_1998_2015_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
PO4_1998_2015_p_value = ds_trend['p'][:]
PO4_signif_1998_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

PO4_signif_1998_2015 = np.where(PO4_signif_1998_2015 == 0, np.NaN, PO4_signif_1998_2015)
PO4_signif_1998_2015 =PO4_signif_1998_2015.T
PO4_1998_2015_trends=PO4_1998_2015_trends.T
# PO4_1998_2015_trends = PO4_1998_2015_trends * 1000 #Convert to nM


ds_trend = Dataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/PO4_2009_2015_trends.nc')
PO4_2009_2015_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
PO4_2009_2015_p_value = ds_trend['p'][:]
PO4_signif_2009_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

PO4_signif_2009_2015 = np.where(PO4_signif_2009_2015 == 0, np.NaN, PO4_signif_2009_2015)
PO4_signif_2009_2015 =PO4_signif_2009_2015.T
PO4_2009_2015_trends=PO4_2009_2015_trends.T
# PO4_2009_2015_trends = PO4_2009_2015_trends * 1000 #Convert to nM


ds_trend = Dataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/nutrients_trends/PO4_2015_2021_trends.nc')
PO4_2015_2021_trends = ds_trend['trend'][:]#*10 #Convert to trends per decade
PO4_2015_2021_p_value = ds_trend['p'][:]
PO4_signif_2015_2021 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

PO4_signif_2015_2021 = np.where(PO4_signif_2015_2021 == 0, np.NaN, PO4_signif_2015_2021)
PO4_signif_2015_2021 =PO4_signif_2015_2021.T
PO4_2015_2021_trends=PO4_2015_2021_trends.T
# PO4_2015_2021_trends = PO4_2015_2021_trends * 1000 #Convert to nM


####################################################
## Plotting the South Polar Stereo PO4 trends map ##
####################################################

#Make lon, lat grids for representation purposes
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

ax.add_feature(ice_50m)


# cmap=cm.cm.curl
cmap=cm.cm.delta_r
n=100
x=0.5
lower = cmap(np.linspace(0, x, n))
white = np.ones((20,4))
upper = cmap(np.linspace(1-x, 1, n))
colors = np.vstack((lower, white, upper))
tmap = linearsegm.from_list('map_white', colors)


levels = [-0.04, -0.035, -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
p1 = plt.contourf(LON, LAT, PO4_2015_2021_trends, levels, cmap=tmap, extend='both', transform=ccrs.PlateCarree()) 
# p2=plt.scatter(LON[::6,::9],LAT[::6,::9], PO4_signif_1998_2015[::6,::9], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-0.04, -0.02, 0, 0.02, 0.04])
cbar.ax.set_ylabel('$mmol·m^{-3}·year^{-1}$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('o) [PO4] trends 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\PO4_trends_2015_2021_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)
