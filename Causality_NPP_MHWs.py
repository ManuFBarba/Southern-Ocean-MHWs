# -*- coding: utf-8 -*-
"""
Figure 4b, c and d. On the causal relationship between MHWs and NPP.
"""


#Loading required modules
import seaborn as sns
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt

from scipy.io import loadmat

import pyEDM
import cartopy.crs as ccrs
import cartopy.feature as cft
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import Axes3D
# import cmocean as cm


ds_SST = xr.open_dataset(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets/Monthly_SST.nc')
SST = ds_SST.analysed_sst

ds_Max_SSTA = xr.open_dataset(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets/Max_SSTA_monthly.nc')
Max_SSTA = ds_Max_SSTA.transpose('lon', 'lat', 'time')
Max_SSTA = Max_SSTA.analysed_sst 

ds_SIC = xr.open_dataset(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets/SIC_monthly.nc')
ds_SIC = ds_SIC.transpose('lon', 'lat', 'time')
SIC = ds_SIC.sea_ice_fraction

ds_MLD = xr.open_dataset(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets/MLD_monthly.nc')
ds_MLD = ds_MLD.transpose('lon', 'lat', 'time')
MLD = ds_MLD.mlotst 

NPP = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets/Monthly_NPP.npy')
NPP = xr.DataArray(NPP, dims=["lon", "lat", "time"], coords={"time": MLD["time"], "lat": MLD["lat"], "lon": MLD["lon"]}, name="NPP")
NPP = NPP.interp(lat=MLD.lat, lon=MLD.lon)
##





## Computing a pixelwise CCM 

def interpolate_series(series):
    if series.isnull().all():
        
        return series
    else:
        
        return series.interpolate_na(dim="time", method="linear")

def test_edm_for_point(maxssta_series, mld_series, embedding=6, lag=3):
    
    maxssta_series = interpolate_series(maxssta_series)
    mld_series = interpolate_series(mld_series)

    
    if maxssta_series.isnull().any() or mld_series.isnull().any():
        return np.nan, np.nan
    
    data = {
        'Time': np.arange(len(maxssta_series)),
        'Max_SSTA': maxssta_series.values,  
        'MLD': mld_series.values
    }
    df = pd.DataFrame(data).dropna()

    if len(df) < (embedding * lag):
        return np.nan, np.nan  

    max_lib_size = len(df) - embedding * lag
    if max_lib_size < 1:
        return np.nan, np.nan

    libsizes = np.array([200], dtype=np.int32)
    
    result = pyEDM.CCM(
        dataFrame=df,
        E=embedding,
        tau=lag,
        columns="Max_SSTA",
        target="MLD",
        libSizes=libsizes,
        sample=int(len(df))
    )
    
    if not result.empty:
        rho_maxssta_to_mld = result['Max_SSTA:MLD'].max()
        rho_mld_to_maxssta = result['MLD:Max_SSTA'].max()
        return rho_maxssta_to_mld, rho_mld_to_maxssta
    else:
        return np.nan, np.nan


MaxSSTA_to_MLD = np.full((720, 100), np.nan)
MLD_to_MaxSSTA = np.full((720, 100), np.nan)

for i in range(720):
    for j in range(100):
        print(f"Processing pixel (lon={i}, lat={j})...")
        maxssta_series = Max_SSTA.isel(lon=i, lat=j)
        mld_series = MLD.isel(lon=i, lat=j)

        rho_maxssta_to_mld, rho_mld_to_maxssta = test_edm_for_point(maxssta_series, mld_series)
        MaxSSTA_to_MLD[i, j] = rho_maxssta_to_mld
        MLD_to_MaxSSTA[i, j] = rho_mld_to_maxssta
        print(f"Done: rho_maxssta_to_mld = {rho_maxssta_to_mld}, rho_mld_to_maxssta = {rho_mld_to_maxssta}")



np.save(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/MaxSSTA_to_MLD.npy', MaxSSTA_to_MLD)
np.save(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/MLD_to_MaxSSTA.npy', MLD_to_MaxSSTA)






## Loading previously proccessed causality matrices
SST_to_NPP = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/SST_to_NPP.npy')
NPP_to_SST = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/NPP_to_SST.npy')

MaxSSTA_to_NPP = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/MaxSSTA_to_NPP.npy')
NPP_to_MaxSSTA = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/NPP_to_MaxSSTA.npy')

SIC_to_NPP = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/SIC_to_NPP.npy')
NPP_to_SIC = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/NPP_to_SIC.npy')

MLD_to_NPP = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/MLD_to_NPP.npy')
NPP_to_MLD = np.load(r'D:\Proyectos_Manu\Causality_MHWs_NPP\Datasets\Causality_CCM_outputs/NPP_to_MLD.npy')



# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

levels = np.linspace(0, 1, 11)

cmap = plt.cm.Greens
p1 = ax.contourf(SST.lon, SST.lat, MLD_to_NPP.T, levels, cmap=cmap, transform=ccrs.PlateCarree())
# p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title(r'SST xmap NPP', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)









