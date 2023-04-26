# -*- coding: utf-8 -*-
"""

################ MODIS, SeaWiFS, VIIRS NPP, CHL and Fe (Model) ################

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
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap as linearsegm
##Reading multiple .hdf files

#Creating the Total Matrix with desired dimensions
Total_VGPM = np.full((300, 2160, 288), np.nan)
Total_CbPM = np.full((300, 2160, 288), np.nan)
#Looping over dimensions
for j in range(1998, 2022):
    os.chdir(f'C:\ICMAN-CSIC\MHW_ANT\datasets_40\VGPM\{j}')
    # os.chdir(f'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\{j}')
    A = [f for f in os.listdir() if f.endswith('.hdf')]

    for i, f in enumerate(A):
        print(j, i)

        filename = f
        
        with Dataset(filename, 'r') as hdf_file:
            data = hdf_file['npp'][()]
        data[data == -9999] = np.nan
        data = np.flipud(data)
        data_ext = data[:300, :]
        # Total_VGPM[:, :, i] = data_ext
        Total_CbPM[:, :, i] = data_ext
#Save data so far
# np.save('C:\ICMAN-CSIC\MHW_ANT\datasets_40\VGPM\Total.npy', Total_VGPM)
np.save('C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\Total.npy', Total_CbPM)
#NPP Matrix of [lat, lon, months]
NPP_VGPM = Total_VGPM
NPP_CbPM = Total_CbPM
# plt.pcolor(NPP[:,:,11])



##VGPM
#December 1998-2021
NPP_VGPM_dec = NPP_VGPM[:,:,11::12]
#January 1998-2021
NPP_VGPM_jan = NPP_VGPM[:,:,::12]
#February 1998-2021
NPP_VGPM_feb = NPP_VGPM[:,:,1::12]

#Other months
NPP_VGPM_mar = NPP_VGPM[:,:,2::12]
NPP_VGPM_apr = NPP_VGPM[:,:,3::12]
NPP_VGPM_may = NPP_VGPM[:,:,4::12]
NPP_VGPM_jun = NPP_VGPM[:,:,5::12]
NPP_VGPM_jul = NPP_VGPM[:,:,6::12]
NPP_VGPM_ago = NPP_VGPM[:,:,7::12]
NPP_VGPM_sep = NPP_VGPM[:,:,8::12]
NPP_VGPM_oct = NPP_VGPM[:,:,9::12]
NPP_VGPM_nov = NPP_VGPM[:,:,10::12]

##CbPM
#December 1998-2021
NPP_CbPM_dec = NPP_CbPM[:,:,11::12]
#January 1998-2021
NPP_CbPM_jan = NPP_CbPM[:,:,::12]
#February 1998-2021
NPP_CbPM_feb = NPP_CbPM[:,:,1::12]

#Other months
NPP_CbPM_mar = NPP_CbPM[:,:,2::12]
NPP_CbPM_apr = NPP_CbPM[:,:,3::12]
NPP_CbPM_may = NPP_CbPM[:,:,4::12]
NPP_CbPM_jun = NPP_CbPM[:,:,5::12]
NPP_CbPM_jul = NPP_CbPM[:,:,6::12]
NPP_CbPM_ago = NPP_CbPM[:,:,7::12]
NPP_CbPM_sep = NPP_CbPM[:,:,8::12]
NPP_CbPM_oct = NPP_CbPM[:,:,9::12]
NPP_CbPM_nov = NPP_CbPM[:,:,10::12]



#Average NPP Dec-Jan-Feb
NPP_VGPM_DJF_ts = np.nanmean(np.array([NPP_VGPM_dec, NPP_VGPM_jan, NPP_VGPM_feb]), axis=0)

NPP_CbPM_DJF_ts = np.nanmean(np.array([NPP_CbPM_dec, NPP_CbPM_jan, NPP_CbPM_feb]), axis=0)

##Annual averaged NPP
NPP_VGPM_ts = np.nanmean(np.array([NPP_VGPM_dec, NPP_VGPM_jan, NPP_VGPM_feb, NPP_VGPM_mar, NPP_VGPM_apr, NPP_VGPM_may, NPP_VGPM_jun, NPP_VGPM_jul, NPP_VGPM_ago, NPP_VGPM_sep, NPP_VGPM_oct, NPP_VGPM_nov]), axis=0)

NPP_CbPM_ts = np.nanmean(np.array([NPP_CbPM_dec, NPP_CbPM_jan, NPP_CbPM_feb, NPP_CbPM_mar, NPP_CbPM_apr, NPP_CbPM_may, NPP_CbPM_jun, NPP_CbPM_jul, NPP_CbPM_ago, NPP_CbPM_sep, NPP_CbPM_oct, NPP_CbPM_nov]), axis=0)

del NPP_VGPM_dec, NPP_VGPM_jan, NPP_VGPM_feb, NPP_VGPM_mar, NPP_VGPM_apr, NPP_VGPM_may, NPP_VGPM_jun, NPP_VGPM_jul, NPP_VGPM_ago, NPP_VGPM_sep, NPP_VGPM_oct, NPP_VGPM_nov, NPP_CbPM_dec, NPP_CbPM_jan, NPP_CbPM_feb, NPP_CbPM_mar, NPP_CbPM_apr, NPP_CbPM_may, NPP_CbPM_jun, NPP_CbPM_jul, NPP_CbPM_ago, NPP_CbPM_sep, NPP_CbPM_oct, NPP_CbPM_nov

NPP_CbPM_1998_2015 = np.nanmean(NPP_CbPM_ts[:,:,0:18], axis=2)
NPP_CbPM_2009_2015 = np.nanmean(NPP_CbPM_ts[:,:,11:18], axis=2)
NPP_CbPM_2015_2021 = np.nanmean(NPP_CbPM_ts[:,:,17:24], axis=2)

#############################################
## Plotting the South Polar Stereo NPP map ##
#############################################

# Make lon, lat grids for representation purposes
m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=-40,
            llcrnrlon=-180,urcrnrlon=180,resolution='l')

lons, lats= m.makegrid(NPP_CbPM_ts.shape[1],NPP_CbPM_ts.shape[0])
x1, y1 = m(lons,lats)

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

cmap='viridis'
# cmap='coolwarm'
# cmap=plt.cm.YlOrRd
# levels = [0,100,200,300,400,500,600,700,800,900,1000]
levels = [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
p1 = plt.contourf(x1, y1, NPP_CbPM_2015_2021, levels, cmap=cmap, extend='max', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='max', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([0, 200, 400, 600, 800, 1000])
cbar.ax.set_ylabel('$mg C·m^{-2}·day^{-1}$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('CbPM NPP 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)



############################## MAP YEARLY TRENDS NPP ##########################

##Create NPP.nc dataset 
fn = 'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM/NPP_CbPM_ts.nc'
ds = nc.Dataset(fn, 'w', format='NETCDF4')

#Add dimensions
time = ds.createDimension('time', 24)
lat = ds.createDimension('lat', 300)
lon = ds.createDimension('lon', 2160)

#Add netCDF variables
times = ds.createVariable('time', 'f4', ('time',))
lats = ds.createVariable('lat', 'f4', ('lat',))
lons = ds.createVariable('lon', 'f4', ('lon',))
value = ds.createVariable('NPP', 'f4', ('lat', 'lon', 'time'))

value.units = '$mg C·m^{-2}·day^{-1}$'

#Assign Latitude, Longitude and time Values
y, x= m.makegrid(NPP_CbPM_ts.shape[1],NPP_CbPM_ts.shape[0])
lats[:] = x[:,0]
lons[:] = y[0,:]
times[:] = np.arange(1998, 2022)

#Assign .nc data variable value
value[:] = NPP_CbPM_ts[:,:,:]



##Mann Kendall test
from xarrayMannKendall import Mann_Kendall_test
import datetime as datetime

##Read data and pick variables
ds = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM/NPP_CbPM_ts.nc')

lon = ds['lon'][:]
lat = ds['lat'][:]
NPP_CbPM_ts = ds['NPP'][:]

NPP_CbPM_1998_2015 = ds['NPP'][:,:,0:18]
NPP_CbPM_2009_2015 = ds['NPP'][:,:,11:18]
NPP_CbPM_2015_2021 = ds['NPP'][:,:,17:24]


# Print function used.
Mann_Kendall_test

#Computing it
NPP_trends = Mann_Kendall_test(NPP_CbPM_2015_2021,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'x':'lat','y':'lon', 'time':'time'})

NPP_grad = NPP_trends.compute()


NPP_grad.attrs['title'] = "NPP trends"
NPP_grad.attrs['Description'] = """NPP computed from SeaWiFS-MODIS,VIIRS Ocean Color NASA data. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall"""
NPP_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
NPP_grad.attrs['Author'] = "Manuel Fernández Barba"
NPP_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

NPP_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

NPP_grad['trend'].attrs['units'] = r'$mg C·m^{-2}·day^{-1}$'
NPP_grad['trend'].attrs['name'] = 'trend'
NPP_grad['trend'].attrs['long_name'] = "NPP trends"

NPP_grad['trend'].attrs['missing_value'] = np.nan
NPP_grad['trend'].attrs['valid_min'] = np.nanmin(NPP_grad['trend'])
NPP_grad['trend'].attrs['valid_max'] = np.nanmax(NPP_grad['trend'])
NPP_grad['trend'].attrs['valid_range'] = [np.nanmin(NPP_grad['trend']),np.nanmax(NPP_grad['trend'])]

######################################################

NPP_grad['signif'].attrs['units'] = ""
NPP_grad['signif'].attrs['name'] = 'signif'
NPP_grad['signif'].attrs['long_name'] = "NPP trends significance"

NPP_grad['signif'].attrs['missing_value'] = np.nan
NPP_grad['signif'].attrs['valid_min'] = np.nanmin(NPP_grad['signif'])
NPP_grad['signif'].attrs['valid_max'] = np.nanmax(NPP_grad['signif'])
NPP_grad['signif'].attrs['valid_range'] = [np.nanmin(NPP_grad['signif']),np.nanmax(NPP_grad['signif'])]

######################################################

NPP_grad['p'].attrs['units'] = ""
NPP_grad['p'].attrs['name'] = 'p'
NPP_grad['p'].attrs['long_name'] = "NPP trends p"

NPP_grad['p'].attrs['missing_value'] = np.nan
NPP_grad['p'].attrs['valid_min'] = np.nanmin(NPP_grad['p'])
NPP_grad['p'].attrs['valid_max'] = np.nanmax(NPP_grad['p'])
NPP_grad['p'].attrs['valid_range'] = [np.nanmin(NPP_grad['p']),np.nanmax(NPP_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in NPP_grad.data_vars}

NPP_grad.to_netcdf('C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_Trends/NPP_2015_2021_trends.nc', encoding=encoding)


##Reading the previously saved NPP trends Dataset
ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_Trends/NPP_1998_2015_trends.nc')
NPP_1998_2015_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
NPP_1998_2015_p_value = ds_trend['p'][:]
signif_1998_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

signif_1998_2015 = np.where(signif_1998_2015 == 0, np.NaN, signif_1998_2015)
signif_1998_2015 =signif_1998_2015.T

NPP_1998_2015_trends=NPP_1998_2015_trends.T


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_Trends/NPP_2009_2015_trends.nc')
NPP_2009_2015_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
NPP_2009_2015_p_value = ds_trend['p'][:]
signif_2009_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

signif_2009_2015 = np.where(signif_2009_2015 == 0, np.NaN, signif_2009_2015)
signif_2009_2015 =signif_2009_2015.T

NPP_2009_2015_trends=NPP_2009_2015_trends.T


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_Trends/NPP_2015_2021_trends.nc')
NPP_2015_2021_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
NPP_2015_2021_p_value = ds_trend['p'][:]
signif_2015_2021 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)
# std_error = ds_trend['std_error'][:]

signif_2015_2021 = np.where(signif_2015_2021 == 0, np.NaN, signif_2015_2021)
signif_2015_2021 =signif_2015_2021.T

NPP_2015_2021_trends=NPP_2015_2021_trends.T


####################################################
## Plotting the South Polar Stereo NPP trends map ##
####################################################


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


cmap=cm.cm.delta
# n=200
# x=0.5
# lower = cmap(np.linspace(0, x, n))
# white = np.ones((20,4))
# upper = cmap(np.linspace(1-x, 1, n))
# colors = np.vstack((lower, white, upper))
# tmap = linearsegm.from_list('map_white', colors)


levels = [-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60]
p1 = plt.contourf(lon, lat, NPP_2015_2021_trends+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_2015_2021[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-60, -40, -20, 0, 20, 40, 60])
cbar.ax.set_ylabel('$mg C·m^{-2}·day^{-1}·year^{-1}$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('d) CbPM NPP trend 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\NPP_trends_2015_2021_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




#### CHL, Fe ####

# ds = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CHL\*.nc', parallel=True)
ds = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Fe\*.nc', parallel=True)


lon = ds['lon'][::10]
lat = ds['lat'][::10]
# lon = ds['longitude'][:]
# lat = ds['latitude'][:]
times = ds['time'][:]

#prochlo = ds['PROCHLO'][:,:,:]
# fe = ds['fe'][:,1,:,:]

chl_year=ds['CHL'][:,::10,::10].groupby('time.year').mean(dim='time',skipna=True)#.load()

# fe_year=ds['fe'][:,1,:,:].groupby('time.year').mean(dim='time',skipna=True)#.load()

ts_1998_2015 = np.nanmean(chl_year[0:18,:,:], axis=0)
ts_2009_2015 = np.nanmean(chl_year[11:18,:,:], axis=0)
ts_2015_2021 = np.nanmean(chl_year[17:24,:,:], axis=0)


# ts_1994_2015 = np.nanmean(fe_year[0:22,:,:], axis=0)
# ts_2009_2015 = np.nanmean(fe_year[15:22,:,:], axis=0)
# ts_2015_2021 = np.nanmean(fe_year[21:28,:,:], axis=0)


###################################################
## Plotting the South Polar Stereo CHL or Fe map ##
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
# cmap=cm.cm.matter
cmap=cm.cm.algae
# cmap=plt.cm.YlOrRd
levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9, 1,1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
# levels = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
# levels = [0,0.25/1000,0.5/1000, 0.75/1000, 1/1000, 1.25/1000, 1.5/1000]
# p1 = plt.contourf(LON, LAT, ts_2015_2021*(10**6)-ts_1994_2015*(10**6), norm = LogNorm(vmin=.0001, vmax=.1), cmap=cmap, extend='max', transform=ccrs.PlateCarree()) 
p1 = plt.contourf(LON, LAT, ts_2015_2021, levels, cmap=cmap, extend='max', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.85, extend ='max', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
cbar.ax.get_yaxis().set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6])
cbar.ax.set_ylabel('$mg·m^{-3}$', fontsize=35)
# cbar.ax.set_ylabel('$nmol·m^{-3}$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('[CHL] 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)



##Compute a Mann Kendall test to calculate Fe and CHL trends
##Mann Kendall test
from xarrayMannKendall import Mann_Kendall_test
import datetime as datetime

CHL_1998_2015 = chl_year[0:18,:,:]
CHL_2009_2015 = chl_year[11:18,:,:]
CHL_2015_2021 = chl_year[17:24,:,:]

# Fe_1994_2015 = fe_year[0:22,:,:]
# Fe_2009_2015 = fe_year[15:22,:,:]
# Fe_2015_2021 = fe_year[21:28,:,:]



###Compute CHL trends

# Print function used.
Mann_Kendall_test

#Computing it
CHL_trends = Mann_Kendall_test(CHL_2015_2021,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'lat','y':'lon'})

CHL_grad = CHL_trends.compute()


CHL_grad.attrs['title'] = "CHL trends"
CHL_grad.attrs['Description'] = """Mass concentration of chlorophyll a in sea water CHL[mg/m3]. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall."""
CHL_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
CHL_grad.attrs['Author'] = "Manuel Fernández Barba"
CHL_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

CHL_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

CHL_grad['trend'].attrs['units'] = r'$mg·m^{-3}·year^{-1}$'
CHL_grad['trend'].attrs['name'] = 'trend'
CHL_grad['trend'].attrs['long_name'] = "CHL trends"

CHL_grad['trend'].attrs['missing_value'] = np.nan
CHL_grad['trend'].attrs['valid_min'] = np.nanmin(CHL_grad['trend'])
CHL_grad['trend'].attrs['valid_max'] = np.nanmax(CHL_grad['trend'])
CHL_grad['trend'].attrs['valid_range'] = [np.nanmin(CHL_grad['trend']),np.nanmax(CHL_grad['trend'])]

######################################################

CHL_grad['signif'].attrs['units'] = ""
CHL_grad['signif'].attrs['name'] = 'signif'
CHL_grad['signif'].attrs['long_name'] = "CHL trends significance"

CHL_grad['signif'].attrs['missing_value'] = np.nan
CHL_grad['signif'].attrs['valid_min'] = np.nanmin(CHL_grad['signif'])
CHL_grad['signif'].attrs['valid_max'] = np.nanmax(CHL_grad['signif'])
CHL_grad['signif'].attrs['valid_range'] = [np.nanmin(CHL_grad['signif']),np.nanmax(CHL_grad['signif'])]

######################################################

CHL_grad['p'].attrs['units'] = ""
CHL_grad['p'].attrs['name'] = 'p'
CHL_grad['p'].attrs['long_name'] = "CHL trends p"

CHL_grad['p'].attrs['missing_value'] = np.nan
CHL_grad['p'].attrs['valid_min'] = np.nanmin(CHL_grad['p'])
CHL_grad['p'].attrs['valid_max'] = np.nanmax(CHL_grad['p'])
CHL_grad['p'].attrs['valid_range'] = [np.nanmin(CHL_grad['p']),np.nanmax(CHL_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in CHL_grad.data_vars}

CHL_grad.to_netcdf('C:\ICMAN-CSIC\MHW_ANT\datasets_40\CHL_trends/CHL_2015_2021_trends.nc', encoding=encoding)


##Reading the previously saved CHL trends Dataset
ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CHL_trends/CHL_1998_2015_trends.nc')
CHL_1998_2015_trends = ds_trend['trend'][:]*10 #Convert to trends per decade
CHL_1998_2015_p_value = ds_trend['p'][:]
signif_1998_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

signif_1998_2015 = np.where(signif_1998_2015 == 0, np.NaN, signif_1998_2015)
signif_1998_2015 =signif_1998_2015.T
CHL_1998_2015_trends=CHL_1998_2015_trends.T


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CHL_trends/CHL_2009_2015_trends.nc')
CHL_2009_2015_trends = ds_trend['trend'][:]*10 #Convert to trends per decade
CHL_2009_2015_p_value = ds_trend['p'][:]
signif_2009_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

signif_2009_2015 = np.where(signif_2009_2015 == 0, np.NaN, signif_2009_2015)
signif_2009_2015 =signif_2009_2015.T
CHL_2009_2015_trends=CHL_2009_2015_trends.T


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CHL_trends/CHL_2015_2021_trends.nc')
CHL_2015_2021_trends = ds_trend['trend'][:]*10 #Convert to trends per decade
CHL_2015_2021_p_value = ds_trend['p'][:]
signif_2015_2021 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

signif_2015_2021 = np.where(signif_2015_2021 == 0, np.NaN, signif_2015_2021)
signif_2015_2021 =signif_2015_2021.T
CHL_2015_2021_trends=CHL_2015_2021_trends.T



####################################################
## Plotting the South Polar Stereo CHL trends map ##
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


# cmap=plt.cm.PiYG
cmap=cm.cm.delta
n=100
x=0.5
lower = cmap(np.linspace(0, x, n))
white = np.ones((20,4))
upper = cmap(np.linspace(1-x, 1, n))
colors = np.vstack((lower, white, upper))
tmap = linearsegm.from_list('map_white', colors)


# levels = [-0.8, -0.7, -0.6, -0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
levels = [-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
p1 = plt.contourf(LON, LAT, CHL_2015_2021_trends, levels, cmap=tmap, extend='both', transform=ccrs.PlateCarree()) 
# p2=plt.scatter(LON[::6,::9],LAT[::6,::9], signif_2015_2021[::6,::9], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
cbar.ax.set_ylabel('$mg·m^{-3}·year^{-1}$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('c) [CHL] trends 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\CHL_trends_2015_2021_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





###############################################################################

#Computing Fe Trends
Fe_trends = Mann_Kendall_test(Fe_1994_2015,'time',MK_modified=True,
                               method="linregress",alpha=0.05, 
                               coords_name = {'time':'year','x':'latitude','y':'longitude'})

Fe_grad = Fe_trends.compute()


Fe_grad.attrs['title'] = "Fe trends"
Fe_grad.attrs['Description'] = """Mole concentration of dissolved iron in sea water Fe[mmol/m3]. Then trends were computed using a modified Mann-Kendall test. \n See: https://github.com/josuemtzmo/xarrayMannKendall."""
Fe_grad.attrs['Publication'] = "Dataset created for Fernández-Barba. et. al. 2023: \n "
Fe_grad.attrs['Author'] = "Manuel Fernández Barba"
Fe_grad.attrs['Contact'] = "manuel.fernandez@csic.es"

Fe_grad.attrs['Created date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

######################################################

Fe_grad['trend'].attrs['units'] = r'$mmol·m^{-3}·year^{-1}$'
Fe_grad['trend'].attrs['name'] = 'trend'
Fe_grad['trend'].attrs['long_name'] = "Fe trends"

Fe_grad['trend'].attrs['missing_value'] = np.nan
Fe_grad['trend'].attrs['valid_min'] = np.nanmin(Fe_grad['trend'])
Fe_grad['trend'].attrs['valid_max'] = np.nanmax(Fe_grad['trend'])
Fe_grad['trend'].attrs['valid_range'] = [np.nanmin(Fe_grad['trend']),np.nanmax(Fe_grad['trend'])]

######################################################

Fe_grad['signif'].attrs['units'] = ""
Fe_grad['signif'].attrs['name'] = 'signif'
Fe_grad['signif'].attrs['long_name'] = "Fe trends significance"

Fe_grad['signif'].attrs['missing_value'] = np.nan
Fe_grad['signif'].attrs['valid_min'] = np.nanmin(Fe_grad['signif'])
Fe_grad['signif'].attrs['valid_max'] = np.nanmax(Fe_grad['signif'])
Fe_grad['signif'].attrs['valid_range'] = [np.nanmin(Fe_grad['signif']),np.nanmax(Fe_grad['signif'])]

######################################################

Fe_grad['p'].attrs['units'] = ""
Fe_grad['p'].attrs['name'] = 'p'
Fe_grad['p'].attrs['long_name'] = "Fe trends p"

Fe_grad['p'].attrs['missing_value'] = np.nan
Fe_grad['p'].attrs['valid_min'] = np.nanmin(Fe_grad['p'])
Fe_grad['p'].attrs['valid_max'] = np.nanmax(Fe_grad['p'])
Fe_grad['p'].attrs['valid_range'] = [np.nanmin(Fe_grad['p']),np.nanmax(Fe_grad['p'])]


comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in Fe_grad.data_vars}

Fe_grad.to_netcdf('C:\ICMAN-CSIC\MHW_ANT\datasets_40\Fe_trends/Fe_2015_2021_trends.nc', encoding=encoding)


##Reading the previously saved Fe trends Dataset
ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Fe_trends/Fe_1994_2015_trends.nc')
Fe_1994_2015_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
Fe_1994_2015_p_value = ds_trend['p'][:]
signif_1994_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

signif_1994_2015 = np.where(signif_1994_2015 == 0, np.NaN, signif_1994_2015)
signif_1994_2015 =signif_1994_2015.T

Fe_1994_2015_trends=Fe_1994_2015_trends.T
Fe_1994_2015_trends = Fe_1994_2015_trends * 1000 #Convert to nM


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Fe_trends/Fe_2009_2015_trends.nc')
Fe_2009_2015_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
Fe_2009_2015_p_value = ds_trend['p'][:]
signif_2009_2015 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

signif_2009_2015 = np.where(signif_2009_2015 == 0, np.NaN, signif_2009_2015)
signif_2009_2015 =signif_2009_2015.T

Fe_2009_2015_trends=Fe_2009_2015_trends.T
Fe_2009_2015_trends = Fe_2009_2015_trends * 1000 #Convert to nM


ds_trend = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Fe_trends/Fe_2015_2021_trends.nc')
Fe_2015_2021_trends = ds_trend['trend'][:] #*10 #Convert to trends per decade
Fe_2015_2021_p_value = ds_trend['p'][:]
signif_2015_2021 = ds_trend['signif'][:] #Same as p-value but matrix converted to 0 and 1 (where p-value < 0.05 --> 1)

signif_2015_2021 = np.where(signif_2015_2021 == 0, np.NaN, signif_2015_2021)
signif_2015_2021 =signif_2015_2021.T

Fe_2015_2021_trends=Fe_2015_2021_trends.T
Fe_2015_2021_trends = Fe_2015_2021_trends * 1000 #Convert to nM

####################################################
## Plotting the South Polar Stereo Fe trends map ###
####################################################

#Make lon, lat grids for representation purposes
# m = Basemap(projection='cyl',llcrnrlat=-89.9,urcrnrlat=-39.9,
#             llcrnrlon=-180,urcrnrlon=180,resolution='l')

# lon, lat= m.makegrid(Fe_2015_2021_trends.shape[1],Fe_2015_2021_trends.shape[0])
# LON, LAT = lon, lat
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


# levels = [-0.00006, -0.00005,-0.00004,-0.00003,-0.00002,-0.00001,0,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006]
levels = [-0.04, -0.035, -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
p1 = plt.contourf(LON, LAT, Fe_2015_2021_trends, levels, cmap=tmap, extend='both', transform=ccrs.PlateCarree()) 
#p2=plt.scatter(LON[::6,::9],LAT[::6,::9], signif_1994_2015[::6,::9], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([-0.00006,-0.00004,-0.00002,0,0.00002,0.00004,0.00006])
cbar.ax.get_yaxis().set_ticks([-0.04, -0.02, 0, 0.02, 0.04])
cbar.ax.set_ylabel('$nM·year^{-1}$', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('f) [Fe] trends 2015-2021', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)




outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\Fe_trends_2015_2021_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






## Represent pcolor NPP ##


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(lon, lat, NPP_CbPM_interp[:,:,:], 24, cmap=cm.cm.algae)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.colorbar(p1, cmap=cmap, shrink=0.8)
# ax.set_title('CbPM NPP')






from matplotlib.colors import LinearSegmentedColormap
import NPP_map
NPP_map = NPP_map.NPP_map_r
NPP_map_divided = [[i/255.0 for i in row] for row in NPP_map]
NPP_map = LinearSegmentedColormap.from_list('npp', NPP_map_divided)



# fig, axs = plt.subplots(figsize=(20, 5))
# plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
# plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
# plt.rcParams.update({'font.size': 16})


# cmap= NPP_map


# levels = [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900]

# p1 = plt.contourf(lon, lat, NPP_CbPM_interp[:,:,23], levels, cmap=cmap, alpha=1, extend='max') 
# # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
# plt.colorbar(p1, cmap=cmap, shrink=1)


time_NPP = np.arange(1998, 2022)

fig, axs = plt.subplots(figsize=(20, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update({'font.size': 16})


cmap= NPP_map

# cmap = plt.get_cmap('jet')

levels = [0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600]

p1 = plt.contourf(lon[:,66], time_NPP , NPP_CbPM_interp[:,66,:].T, levels, cmap=cmap, extend='max') 
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
# plt.colorbar(p1, cmap=cmap, shrink=1)

plt.title('56ºS')

outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\NPP_pcolor_ts_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

###################################################


fig, axs = plt.subplots(figsize=(20, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update({'font.size': 16})


cmap= NPP_map

# cmap = plt.get_cmap('jet')

levels = [0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600]

p1 = plt.contourf(time_NPP, lon[:,66], NPP_CbPM_interp[:,66,:], levels, cmap=cmap, extend='max') 
plt.colorbar(p1, cmap=cmap, shrink=1)

    
plt.title('56ºS')







##Mercator proj##
    fig = plt.figure(figsize=(20, 5))
    ax = plt.axes(projection=ccrs.Mercator(central_longitude=0))

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()

    #Adding some cartopy natural features
    land_50m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                       edgecolor='none', facecolor='black', linewidth=0.5)

    ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
            scale='10m', edgecolor='none', facecolor='white')

    ax.add_feature(ice_50m)

    cmap = NPP_map
    
    levels = [0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600]
    p1 = plt.contourf(lon, lat, NPP_CbPM_interp[:,:,23]+mask, levels, cmap=cmap, extend='max', transform=ccrs.PlateCarree()) 
    
    cbar = plt.colorbar(p1, shrink=0.8, extend ='max', location='right')
    cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=15) 
    cbar.ax.minorticks_off()
    cbar.ax.get_yaxis().set_ticks([0, 200, 400, 600])
    cbar.ax.set_ylabel('mgC·m-2·day-1', fontsize=15)
    ax.add_feature(land_50m, color='k')
    ax.coastlines(resolution='10m', linewidth= 0.50)
    # ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
    ax.set_extent([90, 270, -80, -56], crs=ccrs.PlateCarree())
    # ax.set_title('a) CbPM NPP', fontsize=30)
    plt.show()


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\NPP_SEAICEMASK_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








