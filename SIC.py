# -*- coding: utf-8 -*-
"""
################################## Sea Ice ##############################
"""

# Load required modules
 
import netCDF4 as nc
import numpy as np
import pandas as pd

from netCDF4 import Dataset 
import xarray as xr 

import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import matplotlib.path as mpath
import cartopy.feature as cft
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap as linearsegm

###########################
## Sea Ice Concentration ##
###########################

ds_SIC = xr.open_mfdataset(r'C:/ICMAN-CSIC/MHW_ANT/datasets_40/Sea_Ice_Conc_bimensual/*.nc', parallel=True)

SIC = ds_SIC['sea_ice_fraction'][:,::10,::10]*100
lon = ds_SIC['lon'][::10]
lat = ds_SIC['lat'][::10]
time = ds_SIC['time']

##Mean SIF##
Mean_SIF = ds_SIC['sea_ice_fraction'][:,::10,::10].mean(dim='time',skipna=True)#.load()
Mean_SIF = xr.where(Mean_SIF == 0.0, np.NaN, Mean_SIF)

##Computing SIC
SIC_year=(ds_SIC['sea_ice_fraction'][:,::10,::10]*100).groupby('time.year').mean(dim='time',skipna=True)#.load()

#Mask out all sea grid points, so we only have ice-zone values.
SIC_year = xr.where(SIC_year == 0, np.NaN, SIC_year)


SIC_1982_2015 = np.nanmean(SIC_year[0:34,:,:], axis=0)
SIC_2009_2015 = np.nanmean(SIC_year[27:34,:,:], axis=0)
SIC_2015_2021 = np.nanmean(SIC_year[33:40,:,:], axis=0)


#####################################
## Sea Ice Concentration Anomalies ##
#####################################

#Climatology 1982-2011
ds_clim=ds_SIC.sel(time=slice("1982-01-01", "2011-12-31"))
SIC_clim=ds_clim['sea_ice_fraction'][:,::10,::10].groupby('time.month').mean(dim='time')#.load

#Compute SIC Anomaly
SIC_anom=ds_SIC['sea_ice_fraction'].groupby('time.month') - SIC_clim


SIC_Anom = (SIC_anom*100)[()] #Percent values

SIC_Anom = xr.where(SIC_Anom == 0, np.NaN, SIC_Anom)
# SIC_Anom_1982_2015 = np.nanmean(SIC_Anom[0:34,:,:], axis=0)
# SIC_Anom_2009_2015 = np.nanmean(SIC_Anom[27:34,:,:], axis=0)
# SIC_Anom_2015_2021 = np.nanmean(SIC_Anom[33:40,:,:], axis=0)

SIC_Anom_2015_2021 = SIC_Anom.sel(time=slice("2017-01-01", "2017-12-31"))
SIC_anom=SIC_anom.groupby('time.year').mean(dim='time',skipna=True)[()]


SIC_Anom_2015_2021 = pd.DataFrame(SIC_Anom_2015_2021)      #Visualize core.Dataarray
SIC_Anom_2015_2021 = np.squeeze(np.asarray(SIC_Anom_2015_2021))
outfile = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\SIC_Anom_2015_2021'
np.savez(outfile, SIC_Anom_2015_2021=SIC_Anom_2015_2021)






###################################################
# Plotting the South Polar Stereo SIC Anomaly map #
###################################################
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\SeaIceFraction'
data_SeaIceFraction = np.load(file+'.npz')
Mean_SeaIceFraction_1982_2021 = data_SeaIceFraction['SeaIceFraction']


#Make lon, lat grids for representation purposes
LON, LAT = np.meshgrid(lon, lat)

#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

# ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
#         scale='50m', edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# cmap=cm.cm.ice
cmap=plt.cm.twilight_r
n=100
x=0.5
lower = cmap(np.linspace(0, x, n))
white = np.ones((20,4))
upper = cmap(np.linspace(1-x, 1, n))
colors = np.vstack((lower, white, upper))
tmap = linearsegm.from_list('map_white', colors)


# levels=np.linspace(0.1, 1, num=10)
levels = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
p1 = plt.contourf(lon, lat, SIC_Anom_2015_2021.T, levels, cmap=tmap, extend ='both', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-20, -10, 0, 10, 20])
# cbar.ax.set_ylabel('$[%]$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('SIC Anomaly [2015-2021]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\SIC\SIC_32_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




###############################
# Mean Sea Ice Concentrations #
###############################

#Make lon, lat grids for representation purposes
LON, LAT = np.meshgrid(lon, lat)

#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

# ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
#         scale='50m', edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

cmap=cm.cm.ice

levels=np.linspace(0.1, 1, num=10)
# levels = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
p1 = plt.contourf(lon, lat, Mean_SeaIceFraction_1982_2021, levels, cmap=cmap, transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='min', location='right', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([0,0.2,0.4,0.6,0.8,1])
# cbar.ax.set_ylabel('$[%]$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Mean Sea Ice Fraction [1982-2021]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\SIC\Mean_SIF_1982_2021.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





#####################################
## Sea Ice Concentration trends #####
#####################################

##Mann Kendall test
from xarrayMannKendall import Mann_Kendall_test
import datetime as datetime

#SIC time periods
SIC_full  = ds_SIC['sea_ice_fraction'][:,::10,::10].groupby('time.year').mean(dim='time',skipna=True)#.load() 

SIC_1982_2015 = SIC_full.sel(year=slice("1981-01-01", "2015-12-31"))
SIC_2009_2015 = SIC_full.sel(year=slice("2008-01-01", "2015-12-31"))
SIC_2015_2021 = SIC_full.sel(year=slice("2014-01-01", "2021-12-31"))

#Mask out all sea grid points, so we only have ice-zone values.
SIC_full = xr.where(SIC_full == 0, np.NaN, SIC_full)
SIC_1982_2015 = xr.where(SIC_1982_2015 == 0, np.NaN, SIC_1982_2015)
SIC_2009_2015 = xr.where(SIC_2009_2015 == 0, np.NaN, SIC_2009_2015)
SIC_2015_2021 = xr.where(SIC_2015_2021 == 0, np.NaN, SIC_2015_2021)



# Print function used.
Mann_Kendall_test

#Computing it
SIC_trends = Mann_Kendall_test(SIC_full,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'lat','y':'lon'})

SIC_grad = SIC_trends.compute()


SIC_grad.attrs['title'] = "SIC trends"
SIC_grad.attrs['Description'] = """SIC computed from CCI C3S SST data. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall"""
SIC_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
SIC_grad.attrs['Author'] = "Manuel Fernández Barba"
SIC_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

SIC_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

SIC_grad['trend'].attrs['units'] = r'$%·year^{-1}$'
SIC_grad['trend'].attrs['name'] = 'trend'
SIC_grad['trend'].attrs['long_name'] = "SIC trends"

SIC_grad['trend'].attrs['missing_value'] = np.nan
SIC_grad['trend'].attrs['valid_min'] = np.nanmin(SIC_grad['trend'])
SIC_grad['trend'].attrs['valid_max'] = np.nanmax(SIC_grad['trend'])
SIC_grad['trend'].attrs['valid_range'] = [np.nanmin(SIC_grad['trend']),np.nanmax(SIC_grad['trend'])]

######################################################

SIC_grad['signif'].attrs['units'] = ""
SIC_grad['signif'].attrs['name'] = 'signif'
SIC_grad['signif'].attrs['long_name'] = "SIC trends significance"

SIC_grad['signif'].attrs['missing_value'] = np.nan
SIC_grad['signif'].attrs['valid_min'] = np.nanmin(SIC_grad['signif'])
SIC_grad['signif'].attrs['valid_max'] = np.nanmax(SIC_grad['signif'])
SIC_grad['signif'].attrs['valid_range'] = [np.nanmin(SIC_grad['signif']),np.nanmax(SIC_grad['signif'])]

######################################################

SIC_grad['p'].attrs['units'] = ""
SIC_grad['p'].attrs['name'] = 'p'
SIC_grad['p'].attrs['long_name'] = "SIC trends p"

SIC_grad['p'].attrs['missing_value'] = np.nan
SIC_grad['p'].attrs['valid_min'] = np.nanmin(SIC_grad['p'])
SIC_grad['p'].attrs['valid_max'] = np.nanmax(SIC_grad['p'])
SIC_grad['p'].attrs['valid_range'] = [np.nanmin(SIC_grad['p']),np.nanmax(SIC_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in SIC_grad.data_vars}

SIC_grad.to_netcdf(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SIC_trends/SIC_1982_2021_trends.nc', encoding=encoding)





##Reading the previously saved NPP trends Dataset
ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SIC_trends/SIC_1982_2021_trends.nc')
SIC_1982_2021_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
SIC_1982_2021_p_value = ds_trend['p'][:]
signif_1982_2021 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

signif_1982_2021 = np.where(signif_1982_2021 == 0, np.NaN, signif_1982_2021)
signif_1982_2021 =signif_1982_2021.T

SIC_1982_2021_trends=SIC_1982_2021_trends.T


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SIC_trends/SIC_1982_2015_trends.nc')
SIC_1982_2015_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
SIC_1982_2015_p_value = ds_trend['p'][:]
signif_1982_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

signif_1982_2015 = np.where(signif_1982_2015 == 0, np.NaN, signif_1982_2015)
signif_1982_2015 =signif_1982_2015.T

SIC_1982_2015_trends=SIC_1982_2015_trends.T


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SIC_trends/SIC_2009_2015_trends.nc')
SIC_2009_2015_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
SIC_2009_2015_p_value = ds_trend['p'][:]
signif_2009_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

signif_2009_2015 = np.where(signif_2009_2015 == 0, np.NaN, signif_2009_2015)
signif_2009_2015 = signif_2009_2015.T

SIC_2009_2015_trends=SIC_2009_2015_trends.T


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SIC_trends/SIC_2015_2021_trends.nc')
SIC_2015_2021_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
SIC_2015_2021_p_value = ds_trend['p'][:]
signif_2015_2021 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

signif_2015_2021 = np.where(signif_2015_2021 == 0, np.NaN, signif_2015_2021)
signif_2015_2021 =signif_2015_2021.T

SIC_2015_2021_trends=SIC_2015_2021_trends.T


###################################################
## Plotting the South Polar Stereo SIC trends map #
###################################################

#Make lon, lat grids for representation purposes
LON, LAT = np.meshgrid(lon, lat)

#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

# ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
#         scale='50m', edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

cmap=cm.cm.ice

levels=np.linspace(10, 100, num=10)
# levels = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
p1 = plt.contourf(LON, LAT, SIC_1982_2015_trends, levels, cmap=cmap, transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_1982_2021[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.85, extend ='min', location='right', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([0,0.2,0.4,0.6,0.8,1])
# cbar.ax.set_ylabel('$[%]$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('SIC trends [1982-2021]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\SIC\SIC_trends_1982_2021.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



###################################################
# Plotting the South Polar Stereo SIC contour map #
###################################################

#Make lon, lat grids for representation purposes
LON, LAT = np.meshgrid(lon, lat)
#Make lon, lat grids for representation purposes

# Set the projection
projection = ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

years = 40
year_norm = lambda x : x / (years - 1)

vmin = 1982
vmax = 2021
# Define the colormap
# cmap = plt.cm.get_cmap('cool')
cmap = plt.cm.get_cmap('nipy_spectral', vmax-vmin+1)
cmap_2 = plt.cm.get_cmap('nipy_spectral')
cmap_3 = cm.cm.ice
#Plot the Mean Sea Ice Fraction 1982-2021
levels=np.linspace(0.1, 1, num=10)
p2 = ax.contourf(LON, LAT, Mean_SeaIceFraction_1982_2021.T, levels=levels, cmap=cmap_3, transform=ccrs.PlateCarree()) 
# Loop over each year and plot the sea ice concentration
for i in range(years):
    
    SIC_year = SIC_full[i,:,:] # Extract SIC values for the current year
    
    # Get the color corresponding to the normalized year index
    color = cmap_2(year_norm(i))
    
    # Plot the sea ice concentration for this year with the corresponding color
    p1 = ax.contour(LON, LAT, SIC_year, levels = [0.2], colors=[color], transform=ccrs.PlateCarree())
    
# Add colorbars
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), shrink = 0.85)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([1982, 1992, 2002, 2012, 2021])
cbar.ax.set_ylabel('$year$', fontsize=35)
    
cbar_1 = plt.colorbar(p2, shrink=0.85, location='right', format=ticker.FormatStrFormatter('%.1f'))
cbar_1.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar_1.ax.minorticks_off()
# Set the extent of the plot
ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())

# Add natural features to the map
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)

# Add gridlines
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)

# Set the title of the plot
ax.set_title('Mean Sea Ice Fraction 1982-2021', fontsize=34)
    
# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\SIC\SIC_contour_1982_2021.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






