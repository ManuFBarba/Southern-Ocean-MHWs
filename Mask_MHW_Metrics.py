# -*- coding: utf-8 -*-
"""
################################## Masked MHW Metrics #########################
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


import pandas as pd

ds = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Sea_Ice_Conc_bimensual\*.nc', parallel=True)


seaIce = ds['sea_ice_fraction'][:, ::10, ::10]


SeaIceFraction = np.mean(seaIce, axis=0)

SeaIceFraction = pd.DataFrame(SeaIceFraction) #Visualize core.Dataarray
SeaIceFraction = np.asarray(SeaIceFraction)

# Save data so far
outfile = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\SeaIceFraction'
np.savez(outfile, SeaIceFraction=SeaIceFraction)


lat = ds['lat'][::10]
lon = ds['lon'][::10]


mask = np.where(SeaIceFraction >= 0.2, np.NaN, 0)  
mask = mask.T

# Save data so far
outfile = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
np.savez(outfile, mask=mask)

#Sea Ice Time Series
SeaIce_ts=seaIce.groupby('time.year').mean(dim='time',skipna=True)
SeaIce_ts = SeaIce_ts.mean(dim=('lon', 'lat'),skipna=True)
SeaIce_ts = pd.DataFrame(SeaIce_ts) #Visualize core.Dataarray
SeaIce_ts = np.asarray(SeaIce_ts)

# Save data so far
outfile = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\SeaIce_ts'
np.savez(outfile, SeaIce_ts=SeaIce_ts)

################################# Mask 7200x1000 ##############################

ds = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Sea_Ice_Conc_bimensual\*.nc', parallel=True)


seaIce = ds['sea_ice_fraction'][:, :, :]


SeaIceFraction = np.mean(seaIce, axis=0)

lat = ds['lat'][:]
lon = ds['lon'][:]

SeaIceFraction = pd.DataFrame(SeaIceFraction) #Visualize core.Dataarray

mask_full = np.where(SeaIceFraction >= 0.2, np.NaN, 0)  
#mask_full = mask_full.T

# Save data so far
outfile = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask_full'
np.savez(outfile, mask_full=mask_full)


################################# Decadal Average sea Ice #####################
#Sea Ice in concrete time periods

ds = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Sea_Ice_Conc_bimensual\*.nc', parallel=True)
seaIce = ds['sea_ice_fraction'][:, ::10, ::10]
times = ds['time'][:]

#Calculate datetime
times = times.astype('datetime64')

seaIce_1982_1991 = np.mean(seaIce[np.where(times == np.datetime64('1982-01-01T12:00'))[0][0]:np.where(times == np.datetime64('1991-12-31T12:00'))[0][0]+1,:,:], axis=0)
seaIce_1982_1991=seaIce_1982_1991.T
mask_1982_1991 = xr.where(seaIce_1982_1991 >= 0.2, np.NaN, 0)  

seaIce_1992_2001 = np.mean(seaIce[np.where(times == np.datetime64('1992-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2001-12-31T12:00'))[0][0]+1,:,:], axis=0)
seaIce_1992_2001=seaIce_1992_2001.T
mask_1992_2001 = xr.where(seaIce_1992_2001 >= 0.2, np.NaN, 0)  

seaIce_2002_2011 = np.mean(seaIce[np.where(times == np.datetime64('2002-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2011-12-31T12:00'))[0][0]+1,:,:], axis=0)
seaIce_2002_2011=seaIce_2002_2011.T
mask_2002_2011 = xr.where(seaIce_2002_2011 >= 0.2, np.NaN, 0)  

seaIce_2012_2021 = np.mean(seaIce[np.where(times == np.datetime64('2012-01-01T12:00'))[0][0]:np.where(times == np.datetime64('2021-12-31T12:00'))[0][0]+1,:,:], axis=0)
seaIce_2012_2021=seaIce_2012_2021.T
seaIce_2012_2021 -= 0.10
mask_2012_2021 = xr.where(seaIce_2012_2021 >= 0.2, np.NaN, 0)  


#Save data so far
seaIce_1982_1991 = pd.DataFrame(seaIce_1982_1991)
seaIce_1982_1991 = np.asarray(seaIce_1982_1991)

mask_1982_1991 = pd.DataFrame(mask_1982_1991)
mask_1982_1991 = np.asarray(mask_1982_1991)


seaIce_1992_2001 = pd.DataFrame(seaIce_1992_2001)
seaIce_1992_2001 = np.asarray(seaIce_1992_2001)

mask_1992_2001 = pd.DataFrame(mask_1992_2001)
mask_1992_2001 = np.asarray(mask_1992_2001)


seaIce_2002_2011 = pd.DataFrame(seaIce_2002_2011)
seaIce_2002_2011 = np.asarray(seaIce_2002_2011)

mask_2002_2011 = pd.DataFrame(mask_2002_2011)
mask_2002_2011 = np.asarray(mask_2002_2011)


seaIce_2012_2021 = pd.DataFrame(seaIce_2012_2021)
seaIce_2012_2021 = np.asarray(seaIce_2012_2021)

mask_2012_2021 = pd.DataFrame(mask_2012_2021)
mask_2012_2021 = np.asarray(mask_2012_2021)


outfile_seaIce_periods = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\seaIce_periods'
np.savez(outfile_seaIce_periods, mask_1982_1991=mask_1982_1991, mask_1992_2001=mask_1992_2001, mask_2002_2011=mask_2002_2011, mask_2012_2021=mask_2012_2021, seaIce_1982_1991=seaIce_1982_1991, seaIce_1992_2001=seaIce_1992_2001, seaIce_2002_2011=seaIce_2002_2011, seaIce_2012_2021=seaIce_2012_2021)



#
##Sea Ice Area Fraction##
#
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='papayawhip', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)


levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
p1 = plt.contourf(lon, lat, seaIce_2012_2021, levels, cmap='cmo.ice', extend='both', transform=ccrs.PlateCarree()) 

cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=25) 
cbar.ax.minorticks_off()

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('Mean Sea Ice Fraction [2012-2021]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)










