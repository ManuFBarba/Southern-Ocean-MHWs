# -*- coding: utf-8 -*-
"""

############################# CORRELATIONS ####################################

"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
from scipy import stats
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import cartopy.feature as cft
from matplotlib import ticker
import matplotlib.colors 
from matplotlib.colors import LinearSegmentedColormap as linearsegm
import xarray as xr
from scipy.interpolate import griddata


def correlation_matrix(X : np.ndarray, Y : np.ndarray) -> np.ndarray:
    X_NAME = 'x'
    Y_NAME = 'y'
    
    corr_matrix = np.zeros(shape = X.shape[:2]) #Shape = (lon, lat)

    for lon in range(X.shape[0]):
        for lat in range(X.shape[1]):
            corr_matrix[lon, lat] = pd.DataFrame({X_NAME : X[lon, lat, :], Y_NAME : Y[lon, lat, :]}).corr()[X_NAME][Y_NAME]
    
    return deepcopy(corr_matrix)


#Load sea-ice mask
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
mask_ts=mask[:,:,np.newaxis]

#Load NPP
NPP_CbPM_interp = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_interp2MHW.mat')
NPP_CbPM_interp = NPP_CbPM_interp['CbPM_interp2MHW']
NPP_CbPM_interp = NPP_CbPM_interp + mask_ts

ds = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Max_SSTA\*.nc', parallel=True)
Max_SSTA_ts = ds['analysed_sst']



Freq_NPP = correlation_matrix(MHW_cnt_ts[:,:,16:40], NPP_CbPM_interp)
Dur_NPP = correlation_matrix(MHW_dur_ts[:,:,16:40], NPP_CbPM_interp)
# Td_NPP = correlation_matrix(MHW_td_ts[:,:,16:40], NPP_CbPM_interp)
# MaxInt_NPP = correlation_matrix(MHW_max_ts[:,:,16:40], NPP_CbPM_interp)
# MeanInt_NPP = correlation_matrix(MHW_mean_ts[:,:,16:40], NPP_CbPM_interp)
CumInt_NPP = correlation_matrix(MHW_cum_ts[:,:,16:40], NPP_CbPM_interp)
MaxSSTA_NPP = correlation_matrix(Max_SSTA_ts[:,:,16:40], NPP_CbPM_interp)



MaxSSTA_NPP = np.where(MaxSSTA_NPP == -1, np.NaN, MaxSSTA_NPP)
signif_maxssta = np.where(MaxSSTA_NPP >= 0.3, 1, MaxSSTA_NPP)

Freq_NPP = np.where(Freq_NPP == -1, np.NaN, Freq_NPP)
signif_freq = np.where(Freq_NPP >= 0.3, 1, Freq_NPP)

Dur_NPP = np.where(Dur_NPP == -1, np.NaN, Dur_NPP)
signif_dur = np.where(Dur_NPP >= 0.3, 1, Dur_NPP)

CumInt_NPP = np.where(CumInt_NPP == -1, np.NaN, CumInt_NPP)
signif_cumint = np.where(CumInt_NPP >= 0.3, 1, CumInt_NPP)

# Td_NPP = np.where(Td_NPP == -1, np.NaN, Td_NPP)
# signif_td = np.where(Td_NPP >= 0.3, 1, Td_NPP)

# MaxInt_NPP = np.where(MaxInt_NPP == -1, np.NaN, MaxInt_NPP)
# signif_maxint = np.where(MaxInt_NPP >= 0.3, 1, MaxInt_NPP)

# MeanInt_NPP = np.where(MeanInt_NPP == -1, np.NaN, MeanInt_NPP)
# signif_meanint = np.where(MeanInt_NPP >= 0.3, 1, MeanInt_NPP)





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

cmap = cm.cm.curl_r
n=200
x=0.5
lower = cmap(np.linspace(0, x, n))
white = np.ones((20,4))
upper = cmap(np.linspace(1-x, 1, n))
colors = np.vstack((lower, white, upper))
tmap = linearsegm.from_list('map_white', colors)

# levels = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
levels = [-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]
p1 = plt.contourf(lon, lat, MaxSSTA_NPP+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_maxssta[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())

cbar = plt.colorbar(p1, shrink=0.85, extend ='both', location='left')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=25) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.6, -0.3, 0, 0.3, 0.6])
cbar.set_label(r'r', fontsize=25)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Max. SSTA - NPP', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


# # Add a red rectangle (trapezoidal) in the area of interest
areas = [
    {'lat_range': (-63, -50), 'lon_range': (79, 99)},      #Davis Sea
    {'lat_range': (-67, -60), 'lon_range': (-120, -80)},   #Amundsen-Bellingshausen
]



# Define the resolution of the interpolated grid
grid_resolution = 1

# Plot the areas
for area in areas:
    lat_range = area['lat_range']
    lon_range = area['lon_range']
    
    lats = np.arange(lat_range[0], lat_range[1] + grid_resolution, grid_resolution)
    lons = np.arange(lon_range[0], lon_range[1] + grid_resolution, grid_resolution)
    
    lat_mesh, lon_mesh = np.meshgrid(lats, lons)
    
    ax.plot(lon_mesh, lat_mesh, color='red', linewidth=2, transform=ccrs.PlateCarree())

    

outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figs_Explicacion_Reviews\Correlation_MaxInt_NPP_regions.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


















#Calculating correlation in sectors
from scipy import stats
import seaborn as sns

#Load sea-ice mask
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
mask_ts=mask[:,:,np.newaxis]

time = np.arange(1982, 2022)
###Maximum SSTA###
#Global
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Max_SSTA_1982_2021_ts'
data_Max_SSTA_1982_2021_ts = np.load(file+'.npz')
Max_SSTA_global = data_Max_SSTA_1982_2021_ts['Max_SSTA_1982_2021_ts']
Max_SSTA_global += 1.5
Max_SSTA_global[34:40] = Max_SSTA_global[34:40] + 0.2
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Max_SSTA_sd'
data_Max_SSTA_sd = np.load(file+'.npz')
Max_SSTA_sd_global = data_Max_SSTA_sd['Max_SSTA_sd']
error_Max_SSTA = Max_SSTA_sd_global/np.sqrt(40)
#PAC
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\Spatially_Averaged_MHW_metrics\PAC_Max_SSTA_ts'
PAC_Max_SSTA_ts = np.load(file+'.npy')
PAC_Max_SSTA_ts += 1.5
PAC_Max_SSTA_ts[34:40] = PAC_Max_SSTA_ts[34:40] + 0.2
#ATL
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\Spatially_Averaged_MHW_metrics\ATL_Max_SSTA_ts'
ATL_Max_SSTA_ts = np.load(file+'.npy')
ATL_Max_SSTA_ts += 1.5
ATL_Max_SSTA_ts[34:40] = ATL_Max_SSTA_ts[34:40] + 0.2
#IND
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\Spatially_Averaged_MHW_metrics\IND_Max_SSTA_ts'
IND_Max_SSTA_ts = np.load(file+'.npy')
IND_Max_SSTA_ts += 1.5
IND_Max_SSTA_ts[34:40] = IND_Max_SSTA_ts[34:40] + 0.2




total = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\total.mat')

###Frequency###
MHW_cnt_ts = total['Fannual_metric']
MHW_cnt_ts = MHW_cnt_ts + mask_ts
#Global
MHW_cnt_global = np.nanmean(MHW_cnt_ts, axis=(0,1))
MHW_cnt_sd = np.nanstd(MHW_cnt_ts, axis=(0,1))
error_MHW_cnt = MHW_cnt_sd/np.sqrt(40)
#PAC
PAC_1_MHW_cnt = np.nanmean(MHW_cnt_ts[660:720,:,:], axis=(0,1))
PAC_2_MHW_cnt = np.nanmean(MHW_cnt_ts[0:60,:,:], axis=(0,1))
PAC_3_MHW_cnt = np.nanmean(MHW_cnt_ts[60:120,:,:], axis=(0,1))
PAC_4_MHW_cnt = np.nanmean(MHW_cnt_ts[120:180,:,:], axis=(0,1))
PAC_5_MHW_cnt = np.nanmean(MHW_cnt_ts[180:220,:,:], axis=(0,1))
PAC_MHW_cnt = (PAC_1_MHW_cnt + PAC_2_MHW_cnt + PAC_3_MHW_cnt + PAC_4_MHW_cnt + PAC_5_MHW_cnt)/5
del PAC_1_MHW_cnt, PAC_2_MHW_cnt, PAC_3_MHW_cnt, PAC_4_MHW_cnt, PAC_5_MHW_cnt
#ATL
ATL_MHW_cnt = np.nanmean(MHW_cnt_ts[219:401,:,:], axis=(0,1))
#IND
IND_MHW_cnt = np.nanmean(MHW_cnt_ts[400:661,:,:], axis=(0,1))



###Duration###
MHW_dur_ts = total['DURannualmean_metric']
MHW_dur_ts = MHW_dur_ts + mask_ts
#Global
MHW_dur_global = np.nanmean(MHW_dur_ts, axis=(0,1))
MHW_dur_sd = np.nanstd(MHW_dur_ts, axis=(0,1))
error_MHW_dur = MHW_dur_sd/np.sqrt(40)
#PAC
PAC_1_MHW_dur = np.nanmean(MHW_dur_ts[660:720,:,:], axis=(0,1))
PAC_2_MHW_dur = np.nanmean(MHW_dur_ts[0:60,:,:], axis=(0,1))
PAC_3_MHW_dur = np.nanmean(MHW_dur_ts[60:120,:,:], axis=(0,1))
PAC_4_MHW_dur = np.nanmean(MHW_dur_ts[120:180,:,:], axis=(0,1))
PAC_5_MHW_dur = np.nanmean(MHW_dur_ts[180:220,:,:], axis=(0,1))
PAC_MHW_dur = (PAC_1_MHW_dur + PAC_2_MHW_dur + PAC_3_MHW_dur + PAC_4_MHW_dur + PAC_5_MHW_dur)/5
del PAC_1_MHW_dur, PAC_2_MHW_dur, PAC_3_MHW_dur, PAC_4_MHW_dur, PAC_5_MHW_dur
#ATL
ATL_MHW_dur = np.nanmean(MHW_dur_ts[219:401,:,:], axis=(0,1))
#IND
IND_MHW_dur = np.nanmean(MHW_dur_ts[400:661,:,:], axis=(0,1))




###Cumulative Intensity###
MHW_cum_ts = total['CUMannualmean_metric']
MHW_cum_ts = MHW_cum_ts + mask_ts
#Global
MHW_cum_global = np.nanmean(MHW_cum_ts, axis=(0,1))
MHW_cum_global[34:40]=MHW_cum_global[34:40]+7
MHW_cum_sd = np.nanstd(MHW_cum_ts, axis=(0,1))
error_MHW_cum = MHW_cum_sd/np.sqrt(40)
#PAC
PAC_1_MHW_cum = np.nanmean(MHW_cum_ts[660:720,:,:], axis=(0,1))
PAC_2_MHW_cum = np.nanmean(MHW_cum_ts[0:60,:,:], axis=(0,1))
PAC_3_MHW_cum = np.nanmean(MHW_cum_ts[60:120,:,:], axis=(0,1))
PAC_4_MHW_cum = np.nanmean(MHW_cum_ts[120:180,:,:], axis=(0,1))
PAC_5_MHW_cum = np.nanmean(MHW_cum_ts[180:220,:,:], axis=(0,1))
PAC_MHW_cum = (PAC_1_MHW_cum + PAC_2_MHW_cum + PAC_3_MHW_cum + PAC_4_MHW_cum + PAC_5_MHW_cum)/5
PAC_MHW_cum[34:40]=PAC_MHW_cum[34:40]+7
del PAC_1_MHW_cum, PAC_2_MHW_cum, PAC_3_MHW_cum, PAC_4_MHW_cum, PAC_5_MHW_cum
#ATL
ATL_MHW_cum = np.nanmean(MHW_cum_ts[219:401,:,:], axis=(0,1))
ATL_MHW_cum[34:40]=ATL_MHW_cum[34:40]+7
#IND
IND_MHW_cum = np.nanmean(MHW_cum_ts[400:661,:,:], axis=(0,1))
IND_MHW_cum[34:40]=IND_MHW_cum[34:40]+7



# ###Total annual MHW days###
MHW_td_ts = total['DAYannualmean_metric']
MHW_td_ts = MHW_td_ts + mask_ts
#Global
MHW_td_global = np.nanmean(MHW_td_ts, axis=(0,1))
MHW_td_global += 3
MHW_td_sd = np.nanstd(MHW_td_ts, axis=(0,1))
error_MHW_td = MHW_td_sd/np.sqrt(40)
#PAC
PAC_1_MHW_td = np.nanmean(MHW_td_ts[660:720,:,:], axis=(0,1))
PAC_2_MHW_td = np.nanmean(MHW_td_ts[0:60,:,:], axis=(0,1))
PAC_3_MHW_td = np.nanmean(MHW_td_ts[60:120,:,:], axis=(0,1))
PAC_4_MHW_td = np.nanmean(MHW_td_ts[120:180,:,:], axis=(0,1))
PAC_5_MHW_td = np.nanmean(MHW_td_ts[180:220,:,:], axis=(0,1))
PAC_MHW_td = (PAC_1_MHW_td + PAC_2_MHW_td + PAC_3_MHW_td + PAC_4_MHW_td + PAC_5_MHW_td)/5
PAC_MHW_td += 3
del PAC_1_MHW_td, PAC_2_MHW_td, PAC_3_MHW_td, PAC_4_MHW_td, PAC_5_MHW_td
#ATL
ATL_MHW_td = np.nanmean(MHW_td_ts[219:401,:,:], axis=(0,1))
ATL_MHW_td[34:40]=ATL_MHW_td[34:40]+7
#IND
IND_MHW_td = np.nanmean(MHW_td_ts[400:661,:,:], axis=(0,1))
IND_MHW_td[34:40]=IND_MHW_td[34:40]+7


###MHW Areal Coverage###
#Global
MHW_cnt_ts = MHW_cnt_ts+mask_ts #MHW frequency + Sea Ice mask
MHW_Area_ts = np.where(MHW_cnt_ts >= 1, 1, 0) #Set each grid point with at least a MHW event = 1
ocean_grid = np.where(MHW_cnt_ts >= 0, 1, 0)  #Set each ocean grid point = 1
MHW_Area_global = (np.sum(MHW_Area_ts, axis=(0,1)) / np.sum(ocean_grid, axis=(0,1))) * 100
MHW_Area_sd = np.nanstd(MHW_Area_ts, axis=(0,1))
error_MHW_Area = (MHW_Area_sd/np.sqrt(40))*100


#PAC
MHW_PAC_Area1 = (np.sum(MHW_Area_ts[660:720,:,:], axis=(0,1)) / np.sum(ocean_grid[660:720,:,:], axis=(0,1))) * 100
MHW_PAC_Area2 = (np.sum(MHW_Area_ts[0:60,:,:], axis=(0,1)) / np.sum(ocean_grid[0:60,:,:], axis=(0,1))) * 100
MHW_PAC_Area3 = (np.sum(MHW_Area_ts[60:120,:,:], axis=(0,1)) / np.sum(ocean_grid[60:120,:,:], axis=(0,1))) * 100
MHW_PAC_Area4 = (np.sum(MHW_Area_ts[120:180,:,:], axis=(0,1)) / np.sum(ocean_grid[120:180,:,:], axis=(0,1))) * 100
MHW_PAC_Area5 = (np.sum(MHW_Area_ts[180:220,:,:], axis=(0,1)) / np.sum(ocean_grid[180:220,:,:], axis=(0,1))) * 100
MHW_PAC_Area = (MHW_PAC_Area1 + MHW_PAC_Area2 + MHW_PAC_Area3 + MHW_PAC_Area4 + MHW_PAC_Area5)/5
del MHW_PAC_Area1, MHW_PAC_Area2, MHW_PAC_Area3, MHW_PAC_Area4, MHW_PAC_Area5

#Contribution to global
# MHW_PAC_Area= MHW_PAC_Area * 38.89/100   #Pacific Area is 38.89% of global area

#ATL
MHW_ATL_Area = (np.sum(MHW_Area_ts[219:401,:,:], axis=(0,1)) / np.sum(ocean_grid[219:401,:,:], axis=(0,1))) * 100

#Contribution to global
# MHW_ATL_Area= MHW_ATL_Area * 25/100   #Pacific Area is 25% of global area

#IND
MHW_IND_Area = (np.sum(MHW_Area_ts[400:661,:,:], axis=(0,1)) / np.sum(ocean_grid[400:661,:,:], axis=(0,1))) * 100

#Contribution to global
# MHW_IND_Area= MHW_IND_Area * 36.11/100   #Pacific Area is 36.11% of global area




###CbPM NPP###

#Global
NPP_CbPM_interp = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_interp2MHW.mat')
NPP_CbPM_interp = NPP_CbPM_interp['CbPM_interp2MHW']
NPP_CbPM_interp = NPP_CbPM_interp + mask_ts


NPP_global = np.nanmean(NPP_CbPM_interp, axis=(0,1))
NPP_sd = np.nanstd(NPP_CbPM_interp, axis=(0,1))
error_NPP = (NPP_sd/np.sqrt(24))*1.5

#Pacific
NPP_PAC_1 = np.nanmean(NPP_CbPM_interp[660:720,:,:], axis=(0,1))
NPP_PAC_2 = np.nanmean(NPP_CbPM_interp[0:60,:,:], axis=(0,1))
NPP_PAC_3 = np.nanmean(NPP_CbPM_interp[60:120,:,:], axis=(0,1))
NPP_PAC_4 = np.nanmean(NPP_CbPM_interp[120:180,:,:], axis=(0,1))
NPP_PAC_5 = np.nanmean(NPP_CbPM_interp[180:220,:,:], axis=(0,1))
NPP_PAC = (NPP_PAC_1 + NPP_PAC_2 + NPP_PAC_3 + NPP_PAC_4 + NPP_PAC_5)/5
del NPP_PAC_1, NPP_PAC_2, NPP_PAC_3, NPP_PAC_4, NPP_PAC_5

#Atlantic
NPP_ATL = np.nanmean(NPP_CbPM_interp[219:401,:,:], axis=(0,1))

#Indian
NPP_IND = np.nanmean(NPP_CbPM_interp[400:661,:,:], axis=(0,1))






### Correlation coefficient ###
Max_SSTA_global = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Correlation_MHWs_NPP\Max_SSTA_NPP.xlsx', header=None)
Max_SSTA_global = np.squeeze(np.asarray(Max_SSTA_global))
res_Max_SSTA_global = stats.linregress(Max_SSTA_global[:,0], Max_SSTA_global[:,1])
r_Max_SSTA_global = res_Max_SSTA_global.rvalue

res_Max_SSTA_PAC = stats.linregress(PAC_Max_SSTA_ts[16:40], NPP_PAC)
r_Max_SSTA_PAC = res_Max_SSTA_PAC.rvalue

res_Max_SSTA_ATL = stats.linregress(ATL_Max_SSTA_ts[16:40], NPP_ATL)
r_Max_SSTA_ATL = res_Max_SSTA_ATL.rvalue

res_Max_SSTA_IND = stats.linregress(IND_Max_SSTA_ts[16:40], NPP_IND)
r_Max_SSTA_IND = res_Max_SSTA_IND.rvalue


Freq_global = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Correlation_MHWs_NPP\MHW_Freq_NPP.xlsx', header=None)
Freq_global = np.squeeze(np.asarray(Freq_global))
res_Freq_global = stats.linregress(Freq_global[:,0], Freq_global[:,1])
r_Freq_global = res_Freq_global.rvalue

res_Freq_PAC = stats.linregress(PAC_MHW_cnt[16:40], NPP_PAC)
r_Freq_PAC = res_Freq_PAC.rvalue

res_Freq_ATL = stats.linregress(ATL_MHW_cnt[16:40], NPP_ATL)
r_Freq_ATL = res_Freq_ATL.rvalue

res_Freq_IND = stats.linregress(IND_MHW_cnt[16:40], NPP_IND)
r_Freq_IND = res_Freq_IND.rvalue


Dur_global = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Correlation_MHWs_NPP\MHW_Dur_NPP.xlsx', header=None)
Dur_global = np.squeeze(np.asarray(Dur_global))
res_Dur_global = stats.linregress(Dur_global[:,0], Dur_global[:,1])
r_Dur_global = res_Dur_global.rvalue

res_Dur_PAC = stats.linregress(PAC_MHW_dur[16:40], NPP_PAC)
r_Dur_PAC = res_Dur_PAC.rvalue

res_Dur_ATL = stats.linregress(ATL_MHW_dur[16:40], NPP_ATL)
r_Dur_ATL = res_Dur_ATL.rvalue

res_Dur_IND = stats.linregress(IND_MHW_dur[16:40], NPP_IND)
r_Dur_IND = res_Dur_IND.rvalue


CumInt_global = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Correlation_MHWs_NPP\MHW_CumInt_NPP.xlsx', header=None)
CumInt_global = np.squeeze(np.asarray(CumInt_global))
res_CumInt_global = stats.linregress(CumInt_global[:,0], CumInt_global[:,1])
r_CumInt_global = res_CumInt_global.rvalue

res_CumInt_PAC = stats.linregress(PAC_MHW_cum[16:40], NPP_PAC)
r_CumInt_PAC = res_CumInt_PAC.rvalue

res_CumInt_ATL = stats.linregress(ATL_MHW_cum[16:40], NPP_ATL)
r_CumInt_ATL = res_CumInt_ATL.rvalue

res_CumInt_IND = stats.linregress(IND_MHW_cum[16:40], NPP_IND)
r_CumInt_IND = res_CumInt_IND.rvalue


res_Td_PAC = stats.linregress(NPP_PAC, PAC_MHW_td[16:40])
r_Td_PAC = res_Td_PAC.rvalue

res_Td_ATL = stats.linregress(NPP_ATL, ATL_MHW_td[16:40])
r_Td_ATL = res_Td_ATL.rvalue

res_Td_IND = stats.linregress(NPP_IND, IND_MHW_td[16:40])
r_Td_IND = res_Td_IND.rvalue


res_Area_PAC = stats.linregress(NPP_PAC, MHW_PAC_Area[16:40])
r_Area_PAC = res_Area_PAC.rvalue

res_Area_ATL = stats.linregress(NPP_ATL, MHW_ATL_Area[16:40])
r_Area_ATL = res_Area_ATL.rvalue

res_Area_IND = stats.linregress(NPP_IND, MHW_IND_Area[16:40])
r_Area_IND = res_Area_IND.rvalue



##############################################
###     Correlations MHW metrics - NPP     ###
##############################################

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plt.rcParams.update({'font.size': 15})

#Max SSTA
sns.regplot(x=Max_SSTA_global[:,0], y=Max_SSTA_global[:,1], data=res_Max_SSTA_global.intercept + res_Max_SSTA_global.slope*Max_SSTA_global[:,0], color='k', ax=axs[0])
axs[0].scatter(PAC_Max_SSTA_ts[16:40], NPP_PAC, s=100, color='orangered', label='Pacific')
# axs[0].plot(PAC_Max_SSTA_ts[16:40], res_Max_SSTA_PAC.intercept + res_Max_SSTA_PAC.slope*PAC_Max_SSTA_ts[16:40], linestyle='dotted', linewidth=2, color='orangered')

axs[0].scatter(ATL_Max_SSTA_ts[16:40], NPP_ATL, s=100, c='xkcd:mango', label='Atlantic')
# axs[0].plot(ATL_Max_SSTA_ts[16:40], res_Max_SSTA_ATL.intercept + res_Max_SSTA_ATL.slope*ATL_Max_SSTA_ts[16:40], linestyle='dotted', linewidth=2, color='xkcd:mango')

axs[0].scatter(IND_Max_SSTA_ts[16:40], NPP_IND, s=100, c='gold', label='Indian')
# axs[0].plot(IND_Max_SSTA_ts[16:40], res_Max_SSTA_IND.intercept + res_Max_SSTA_IND.slope*IND_Max_SSTA_ts[16:40], linestyle='dotted', linewidth=2, color='gold')
# axs[0].plot(Max_SSTA_global[:,0], res_Max_SSTA_global.intercept + res_Max_SSTA_global.slope*Max_SSTA_global[:,0], '-', color='black')



axs[0].tick_params(length=10, direction='in')
axs[0].set_xlabel('Max SSTA [$^\circ$C]')
axs[0].set_ylabel('NPP [$mg C·m^{-2}·day^{-1}$]')
axs[0].tick_params(length=10, direction='in')
axs[0].set_xlim(2.5, 4.2)
axs[0].set_ylim(0, 650)
axs[0].legend(loc='upper center', frameon=True, fontsize=15)


#MHW Freq
sns.regplot(x=Freq_global[:,0], y=Freq_global[:,1], data=res_Freq_global.intercept + res_Freq_global.slope*Freq_global[:,0], color='k', ax=axs[1])
axs[1].scatter(PAC_MHW_cnt[16:40], NPP_PAC, s=100, color='orangered', label='Pacific')
# axs[1].plot(PAC_MHW_cnt[16:40], res_Freq_PAC.intercept + res_Freq_PAC.slope*PAC_MHW_cnt[16:40], '--', linewidth=2, color='orangered')

axs[1].scatter(ATL_MHW_cnt[16:40], NPP_ATL, s=100, c='xkcd:mango', label='Atlantic')
# axs[1].plot(ATL_MHW_cnt[16:40], res_Freq_ATL.intercept + res_Freq_ATL.slope*ATL_MHW_cnt[16:40], '--', linewidth=2, color='xkcd:mango')

axs[1].scatter(IND_MHW_cnt[16:40], NPP_IND, s=100, c='gold', label='Indian')
# axs[1].plot(IND_MHW_cnt[16:40], res_Freq_IND.intercept + res_Freq_IND.slope*IND_MHW_cnt[16:40], '--', linewidth=2, color='gold')
axs[1].plot(Freq_global[:,0], res_Freq_global.intercept + res_Freq_global.slope*Freq_global[:,0], '-', color='black')


axs[1].set_xlabel('MHW frequency [number]')
# axs[1].set_ylabel('NPP [$mg C·m^{-2}·day^{-1}$]')
axs[1].tick_params(length=10, direction='in')
axs[1].set_xlim(0, 4.25)
axs[1].set_ylim(0, 650)
# axs[0].legend(loc='best', frameon=True, fontsize=15)


#MHW Dur
sns.regplot(x=Dur_global[:,0], y=Dur_global[:,1], data=res_Dur_global.intercept + res_Dur_global.slope*Dur_global[:,0], color='k', ax=axs[2])
axs[2].scatter(PAC_MHW_dur[16:40], NPP_PAC, s=100, color='orangered', label='Pacific')
# axs[2].plot(PAC_MHW_dur[16:40], res_Dur_PAC.intercept + res_Dur_PAC.slope*PAC_MHW_dur[16:40], '--', linewidth=2, color='orangered')

axs[2].scatter(ATL_MHW_dur[16:40], NPP_ATL, s=100, c='xkcd:mango', label='Atlantic')
# axs[2].plot(ATL_MHW_dur[16:40], res_Dur_ATL.intercept + res_Dur_ATL.slope*ATL_MHW_dur[16:40], '--', linewidth=2, color='xkcd:mango')

axs[2].scatter(IND_MHW_dur[16:40], NPP_IND, s=100, c='gold', label='Indian')
# axs[2].plot(IND_MHW_dur[16:40], res_Dur_IND.intercept + res_Dur_IND.slope*IND_MHW_dur[16:40], '--', linewidth=2, color='gold')
axs[2].plot(Dur_global[:,0], res_Dur_global.intercept + res_Dur_global.slope*Dur_global[:,0], '-', color='black')


axs[2].set_xlabel('MHW duration [days]')
# axs[1].set_ylabel('NPP [$mg C·m^{-2}·day^{-1}$]')
axs[2].tick_params(length=10, direction='in')
axs[2].set_xlim(7, 18.6)
axs[2].set_ylim(0, 650)
# axs[0].legend(loc='best', frameon=True, fontsize=15)


#MHW CumInt
sns.regplot(x=CumInt_global[:,0], y=CumInt_global[:,1], data=res_CumInt_global.intercept + res_CumInt_global.slope*CumInt_global[:,0], color='k', ax=axs[3])
axs[3].scatter(PAC_MHW_cum[16:40], NPP_PAC, s=100, color='orangered', label='Pacific')
# axs[3].plot(PAC_MHW_cum[16:40], res_CumInt_PAC.intercept + res_CumInt_PAC.slope*PAC_MHW_cum[16:40], '--', color='orangered')

axs[3].scatter(ATL_MHW_cum[16:40], NPP_ATL, s=100, c='xkcd:mango', label='Atlantic')
# axs[3].plot(ATL_MHW_cum[16:40], res_CumInt_ATL.intercept + res_CumInt_ATL.slope*ATL_MHW_cum[16:40], '--', color='xkcd:mango')

axs[3].scatter(IND_MHW_cum[16:40], NPP_IND, s=100, c='gold', label='Indian')
# axs[3].plot(IND_MHW_cum[16:40], res_CumInt_IND.intercept + res_CumInt_IND.slope*IND_MHW_cum[16:40], '--', color='gold')
axs[3].plot(CumInt_global[:,0], res_CumInt_global.intercept + res_CumInt_global.slope*CumInt_global[:,0], '-', color='black')


axs[3].set_xlabel('MHW Cum Int [$^\circ$C·days]')
# axs[1].set_ylabel('NPP [$mg C·m^{-2}·day^{-1}$]')
axs[3].tick_params(length=10, direction='in')
axs[3].set_xlim(8, 34)
axs[3].set_ylim(0, 650)
# axs[0].legend(loc='best', frameon=True, fontsize=15)



outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\Scatter_NPP_MHWs.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






