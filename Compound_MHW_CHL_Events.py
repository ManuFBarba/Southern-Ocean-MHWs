# -*- coding: utf-8 -*-
"""

############################## Compound MHW - HChl Event ######################

"""


# Load required modules
import netCDF4 as nc
import numpy as np
from scipy import io
from datetime import date
from netCDF4 import Dataset 
import xarray as xr 
import matplotlib
import matplotlib.pyplot as plt
import marineHeatWaves as mhw
from tempfile import TemporaryFile
import pandas as pd



# Some basic extreme events parameters
coldSpells = False # If true detect coldspells instead of heatwaves
col_clim = '0.25'
col_thresh = 'g-'
if coldSpells:
    mhwname = 'MCS'
    mhwfullname = 'coldspell'
    col_evMax = (0, 102./255, 204./255)
    col_ev = (153./255, 204./255, 1)
    col_bar = (0.5, 0.5, 1)
else:
    mhwname = 'MHW'
    mhwfullname = 'heatwave'
    col_evMax = 'r'
    col_ev = (1, 0.6, 0.5)
    col_bar = (1, 0.5, 0.5)



t_sst_nsat_ice = np.arange(date(1982,1,1).toordinal(),date(2021,12,31).toordinal()+1)
dates = [date.fromordinal(tt.astype(int)) for tt in t_sst_nsat_ice]

t_npp = np.arange(date(1993,1,1).toordinal(),date(2020,12,31).toordinal()+1)
dates_npp = [date.fromordinal(tt.astype(int)) for tt in t_npp]

mld_dates = pd.date_range(start='1993-01-01', end='2021-12-31', freq='M')
mld_dates = [d.to_pydatetime().date() for d in mld_dates]

#Reading Datasets and variables
#SST
ds_sst = xr.open_dataset(r'.\SST_ANT_1982-2021_40.nc')

#MLD
ds_mld = xr.open_dataset(r'./MLD_1993_2021.nc')

#NPP / CHL
ds_npp = xr.open_mfdataset(r'.\Daily_chl\*.nc', parallel=True)


#N-SAT
ds_nsat = xr.open_dataset(r'.\ERA5_1982-2021_T2M_ANT.nc')

#SIC
ds_ice = xr.open_mfdataset(r'.\Sea_Ice_Conc_bimensual\*.nc', parallel=True)





#Load sea-ice mask
file = r'.\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask'].T
mask_ts=mask[np.newaxis,:,:]

###############
# Location points
###############
# Defining the lat, lon for the location of interest

# lat_point = -58   #Drake Passage
# lon_point = -61

# lat_point = -59   #South Shetland Islands
# lon_point = -60

# lat_point = -60   #Davis Sea
# lon_point = 88

# lat_point = -63   #Ross Sea
# lon_point = -178


# #Storing the lat and lon data of the netCDF file into variables 
# lat_sst = ds.variables['lat'][:]
# lon_sst = ds.variables['lon'][:]

# lat_chl = ds_chl.variables['latitude'][:]
# lon_chl = ds_chl.variables['longitude'][:]

# lon_nsat = dz.variables['longitude'][:]
# lat_nsat = dz.variables['latitude'][:]

# lat_ice = ds_ice.variables['lat'][:]
# lon_ice = ds_ice.variables['lon'][:]


# #Squared difference between the specified lat,lon and the lat,lon of the SST dataset 
# sq_diff_lat_sst = (lat_sst - lat_point)**2 
# sq_diff_lon_sst = (lon_sst - lon_point)**2

# #Squared difference between the specified lat,lon and the lat,lon of the CHL dataset 
# sq_diff_lat_chl = (lat_chl - lat_point)**2 
# sq_diff_lon_chl = (lon_chl - lon_point)**2

# #Squared difference between the specified lat,lon and the lat,lon of the N-SAT dataset 
# sq_diff_lat_nsat = (lat_nsat - lat_point)**2 
# sq_diff_lon_nsat = (lon_nsat - lon_point)**2

# #Squared difference between the specified lat,lon and the lat,lon of the SeaIce dataset 
# sq_diff_lat_ice = (lat_ice - lat_point)**2 
# sq_diff_lon_ice = (lon_ice - lon_point)**2



# #Identify the index of the min value for lat and lon in SST dataset
# min_index_lat_sst = sq_diff_lat_sst.argmin()
# min_index_lon_sst = sq_diff_lon_sst.argmin()

# #Identify the index of the min value for lat and lon in CHL dataset
# min_index_lat_chl = sq_diff_lat_chl.argmin()
# min_index_lon_chl = sq_diff_lon_chl.argmin()

# #Identify the index of the min value for lat and lon in NSAT dataset
# min_index_lat_nsat = sq_diff_lat_nsat.argmin()
# min_index_lon_nsat = sq_diff_lon_nsat.argmin()

# #Identify the index of the min value for lat and lon in SeaIce dataset
# min_index_lat_ice = sq_diff_lat_ice.argmin()
# min_index_lon_ice = sq_diff_lon_ice.argmin()



# #Extracting SST, CHL, NSAT, and SIC time series for a concrete point
# sst = ds.variables['analysed_sst'][:,min_index_lat_sst,min_index_lon_sst] - 273.15     #Daily from 01/01/1982 to 31/12/2021

# chl = ds_chl.variables['chl'][:,0,min_index_lat_chl,min_index_lon_chl]                 #Daily from 01/01/1993 to 31/12/2020
# chl = chl.values.astype(np.float32)

# t2m = dz.variables['t2m'][:,min_index_lat_nsat,min_index_lon_nsat] - 273.15            #Daily from 01/01/1982 to 31/12/2021

# sea_ice = ds_ice.variables['sea_ice_fraction'][:,min_index_lat_ice,min_index_lon_ice] *100 #Daily from 01/01/1982 to 31/12/2021
# sea_ice = sea_ice.values.astype(np.float32)




###############
# Regions of interest
###############


lat_point_min = -63   # Davis Sea
lat_point_max = -50

lon_point_min = 79
lon_point_max = 99


lat_point_min = -67   # Amundsen-Bellingshausen
lat_point_max = -60

lon_point_min = -120
lon_point_max = -80



lat_point_min_sst = -59   # Davis Sea
lat_point_max_sst = -50

lon_point_min_sst = 79
lon_point_max_sst = 99


lat_point_min_sst = -65   # Amundsen-Bellingshausen
lat_point_max_sst = -60

lon_point_min_sst = -120
lon_point_max_sst = -80







## Revised Study areas

######################################
lat_point_min = -62   # Davis Sea
lat_point_max = -55

lon_point_min = 79
lon_point_max = 99
######################################

lat_point_min = -68   # Amundsen-Bellingshausen
lat_point_max = -60

lon_point_min = -109
lon_point_max = -89
######################################





#Extracting SST, MLD, NPP, NSAT and SIC time series for a region of interest
sst = ds_sst.analysed_sst.sel(lat=slice(lat_point_min, lat_point_max), lon=slice(lon_point_min, lon_point_max))
sst = np.nanmean(sst, axis=(1, 2))
sst = sst - 273.15

mld = ds_mld.mlotst.sel(latitude=slice(lat_point_min, lat_point_max), longitude=slice(lon_point_min, lon_point_max))
mld = np.nanmean(mld, axis=(1, 2))

npp = ds_npp.nppv.sel(latitude=slice(lat_point_min, lat_point_max), longitude=slice(lon_point_min, lon_point_max))
npp = np.nanmean(npp, axis=(1, 2, 3))

t2m = ds_nsat.t2m
t2m = t2m.sel(longitude=slice(lon_point_min, lon_point_max), latitude=slice(lat_point_max, lat_point_min))
t2m = np.nanmean(t2m, axis=(1, 2))
t2m = t2m - 273.15

ice = ds_ice.sea_ice_fraction.sel(lat=slice(lat_point_min, lat_point_max), lon=slice(lon_point_min, lon_point_max))
ice = np.nanmean(ice, axis=(1, 2))
ice = ice * 100



###################### Load files that were previously proccessed

file = r'.\sst_Davis_Sea'
sst = np.load(file+'.npy')

file = r'.\npp_averaged_Davis_Sea'
npp_averaged = np.load(file+'.npy')


file = r'.\sst_Amundsen_Bellingshausen'
sst = np.load(file+'.npy')

file = r'.\npp_averaged_Amundsen_Bellingshausen'
npp_averaged = np.load(file+'.npy')
######################



#Computing a moving mean (conserving edges) over N-SAT
def rollavg_roll_edges(a,n):
    'Numpy array rolling, edge handling'
    assert n%2==1
    a = np.pad(a,(0,n-1-n//2), 'constant')*np.ones(n)[:,None]
    m = a.shape[1]
    idx = np.mod((m-1)*np.arange(n)[:,None] + np.arange(m), m) # Rolling index
    out = a[np.arange(-n//2,n//2)[:,None], idx]
    d = np.hstack((np.arange(1,n),np.ones(m-2*n+1+n//2)*n,np.arange(n,n//2,-1)))
    return (out.sum(axis=0)/d)[n//2:]


window = 7

t2m_averaged = rollavg_roll_edges(t2m, 31)

npp_averaged = rollavg_roll_edges(npp, window)


###############
# Applying Marine Heat Wave definition
###############

mhws, clim = mhw.detect(t_sst_nsat_ice, temp=sst, climatologyPeriod=[1982, 2011], pctile=95, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=False, coldSpells=coldSpells, alternateClimatology=False)
mhwBlock = mhw.blockAverage(t_sst_nsat_ice, mhws, clim=clim, temp=sst)
mean, trend, dtrend = mhw.meanTrend(mhwBlock)


###############
# Applying Marine Heat Wave definition to NPP values
###############

hnpp, clim_npp = mhw.detect(t_npp, temp=npp_averaged, climatologyPeriod=[1993, 2014], pctile=90, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=False, coldSpells=coldSpells, alternateClimatology=False)
nppBlock = mhw.blockAverage(t_npp, hnpp, clim=clim_npp, temp=npp_averaged)
mean, trend, dtrend = mhw.meanTrend(nppBlock)




###
ev_int = np.argmax(mhws['intensity_max'])   # Find most intense event
ev_dur = np.argmax(mhws['duration'])        # Find largest event
ev = np.argmax(mhws)

ev_int_npp = np.argmax(hnpp['intensity_max'])   # Find most intense event
ev_dur_npp = np.argmax(hnpp['duration'])        # Find largest event
ev_npp = np.argmax(hnpp)





                    #####Davis Sea#####
plt.clf()

## Plot N-SAT, seasonal cycles, SST, NPP, thresholds, MLD, SIC, and shade MHWs and HNPP events in red and green, respectively
ts = date(2015, 1, 1)
te = date(2018, 3, 31)

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(20, 15))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()

# Set font size
plt.rcParams.update({'font.size': 22})


##### Plot SST on the first y-axis (in red) #####
# Plot SST, seasonal cycle, and threshold
ax1.plot(dates, t2m_averaged, 'k:')
ax1.plot(dates, (clim['seas']), '-', color='grey')
ax1.plot(dates, sst, 'r-', linewidth=2)
ax1.plot(dates, (clim['thresh']), 'r-.')
ax1.set_xlim(ts, te)
ax1.set_ylim(-6, 5)
ax1.set_ylabel(r'[$^\circ$C]', fontsize=20, labelpad=18, color='red')
ax1.tick_params(axis='y', colors='red')  # Set y-axis tick color


## Shade MHWs events in red
from datetime import date, datetime

start_2017 = date(2017, 5, 1)
end_2017 = date(2017, 11, 1)


ev_exclude = []
for i, (start_ord, end_ord) in enumerate(zip(mhws['time_start'], mhws['time_end'])):
    start_date = date.fromordinal(start_ord)
    end_date = date.fromordinal(end_ord)
    
    if start_2017 <= start_date <= end_2017 or start_2017 <= end_date <= end_2017:
        ev_exclude.append(i)

for ev0 in np.arange(ev+42, ev + 60, 1):
    if ev0 in ev_exclude:
        continue  

    t1 = np.where(t_sst_nsat_ice == mhws['time_start'][ev0])[0][0]
    t2 = np.where(t_sst_nsat_ice == mhws['time_end'][ev0])[0][0]
    ax1.fill_between(dates[t1 - 1: t2 + 1], sst[t1 - 1: t2 + 1], clim['thresh'][t1 - 1: t2 + 1], color='r', alpha=0.5)


##### Plot NPP on the second y-axis (in dark green) #####

# Plot NPP, seasonal cycle, threshold, and HNPP event
ax2.plot(dates_npp, (clim_npp['seas']), '-', color='grey')
ax2.plot(dates_npp, npp_averaged, '-', color='darkgreen', linewidth=2)
ax2.plot(dates_npp, (clim_npp['thresh']), '-.', color='darkgreen')
ax2.set_ylim(0, 22)
ax2.set_ylabel(r'CbPM NPP [$mg C·m^{-3}·day^{-1}$]', fontsize=22, labelpad=20, color='darkgreen')
ax2.tick_params(axis='y', colors='darkgreen')  

# Shade HNPP events in dark green
for ev1 in np.arange(ev_npp - 40, ev_npp + 40, 1):
    t1 = np.where(t_npp == hnpp['time_start'][ev1])[0][0]
    t2 = np.where(t_npp == hnpp['time_end'][ev1])[0][0]
    ax2.fill_between(dates_npp[t1 - 1: t2 + 2], npp_averaged[t1 - 1: t2 + 2], clim_npp['thresh'][t1 - 1: t2 + 2], color='darkgreen', alpha=0.5)


##### Plot MLD on the fourth y-axis (in darkblue) #####
ax3.plot(mld_dates, mld, ':', color='darkblue', linewidth=1.5, alpha=1)
ax3.set_ylim(0, 200)
ax3.set_ylabel('MLD [m]', fontsize=22, labelpad=20, color='darkblue')
ax3.tick_params(axis='y', colors='darkblue')

##### Plot sea ice concentration on the third y-axis (in purple) #####
ax4.plot(dates, ice, '-', color='purple', linewidth=1.25)
ax4.set_ylim(0, 100)
ax4.set_ylabel('SIC [%]', fontsize=22, labelpad=20, color='purple')
ax4.tick_params(axis='y', colors='purple')



##### Customize the plot #####

# Set x-axis limits and tick parameters
ax1.set_xlim(ts, te)
ax1.tick_params(length=7, direction='out')

ax1.spines['left'].set_color('red')
ax1.yaxis.label.set_color('red')

ax2.spines['right'].set_color('darkgreen')
ax2.yaxis.label.set_color('darkgreen')
ax2.yaxis.set_label_coords(1.05, 0.5)

ax3.spines['left'].set_position(('outward', 95))
ax3.spines['left'].set_color('darkblue')
ax3.yaxis.label.set_color('darkblue')
ax3.yaxis.set_label_coords(-0.15, 0.5)
ax3.yaxis.set_label_position('left')  
ax3.yaxis.set_ticks_position('left')

ax4.spines['right'].set_position(('outward', 95))
ax4.spines['right'].set_color('purple')
ax4.yaxis.label.set_color('purple')
ax4.yaxis.set_label_coords(1.125, 0.5)


# Set legend
ax1.legend(['N-SAT', 'Climatological seasonal cycles', 'SST', '95th percentile SST', '_', 'MHW events', '_', '_'],
            loc='upper left', frameon=False, fontsize=22)

ax3.legend(['MLD'],
            loc='upper right', frameon=False, fontsize=22, bbox_to_anchor=(0.8479, 1))

ax4.legend(['SIC'],
            loc='upper right', frameon=False, fontsize=22, bbox_to_anchor=(0.8352, 0.96))

ax2.legend(['_', 'CbPM NPP', '95th percentile NPP', 'HNPP events'],
            loc='upper right', frameon=False, fontsize=22, bbox_to_anchor=(1, 0.92))


# Set title
ax1.set_title('Davis Sea', fontsize=30)


outfile = r'.\Compound_MHW_HNPP_Davis_Sea.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')






                    #####Amundsen-Bellingshausen Seas#####
plt.clf()

## Plot N-SAT, seasonal cycles, SST, NPP, thresholds, MLD, SIC, and shade MHWs and HNPP events in red and green, respectively
ts = date(2015, 1, 1)
te = date(2018, 3, 31)

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(20, 15))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()

# Set font size
plt.rcParams.update({'font.size': 22})


##### Plot SST on the first y-axis (in red) #####
# Plot SST, seasonal cycle, and threshold
ax1.plot(dates, t2m_averaged, 'k:')
ax1.plot(dates, (clim['seas']), '-', color='grey')
ax1.plot(dates, sst, 'r-', linewidth=2)
ax1.plot(dates, (clim['thresh']), 'r-.')
ax1.set_xlim(ts, te)
ax1.set_ylim(-6, 4.75)
ax1.set_ylabel(r'[$^\circ$C]', fontsize=20, labelpad=18, color='red')
ax1.tick_params(axis='y', colors='red')  # Set y-axis tick color


## Shade MHWs events in red
from datetime import date, datetime

start_2017 = date(2015, 5, 1)
end_2017 = date(2015, 11, 1)


ev_exclude = []
for i, (start_ord, end_ord) in enumerate(zip(mhws['time_start'], mhws['time_end'])):
    start_date = date.fromordinal(start_ord)
    end_date = date.fromordinal(end_ord)
    
    if start_2017 <= start_date <= end_2017 or start_2017 <= end_date <= end_2017:
        ev_exclude.append(i)

for ev0 in np.arange(ev, ev + 30, 1):
    if ev0 in ev_exclude:
        continue  

    t1 = np.where(t_sst_nsat_ice == mhws['time_start'][ev0])[0][0]
    t2 = np.where(t_sst_nsat_ice == mhws['time_end'][ev0])[0][0]
    ax1.fill_between(dates[t1 - 1: t2 + 1], sst[t1 - 1: t2 + 1], clim['thresh'][t1 - 1: t2 + 1], color='r', alpha=0.5)


##### Plot NPP on the second y-axis (in dark green) #####

# Plot NPP, seasonal cycle, threshold, and HNPP event
ax2.plot(dates_npp, (clim_npp['seas']), '-', color='grey')
ax2.plot(dates_npp, npp_averaged, '-', color='darkgreen', linewidth=2)
ax2.plot(dates_npp, (clim_npp['thresh']), '-.', color='darkgreen')
ax2.set_ylim(0, 35)
ax2.set_ylabel(r'CbPM NPP [$mg C·m^{-3}·day^{-1}$]', fontsize=22, labelpad=20, color='darkgreen')
ax2.tick_params(axis='y', colors='darkgreen')  

# Shade HNPP events in dark green
for ev1 in np.arange(ev_npp - 40, ev_npp + 40, 1):
    t1 = np.where(t_npp == hnpp['time_start'][ev1])[0][0]
    t2 = np.where(t_npp == hnpp['time_end'][ev1])[0][0]
    ax2.fill_between(dates_npp[t1 - 1: t2 + 2], npp_averaged[t1 - 1: t2 + 2], clim_npp['thresh'][t1 - 1: t2 + 2], color='darkgreen', alpha=0.5)


##### Plot MLD on the fourth y-axis (in darkblue) #####
ax3.plot(mld_dates, mld, ':', color='darkblue', linewidth=1.5, alpha=1)
ax3.set_ylim(0, 200)
ax3.set_ylabel('MLD [m]', fontsize=22, labelpad=20, color='darkblue')
ax3.tick_params(axis='y', colors='darkblue')

##### Plot sea ice concentration on the third y-axis (in purple) #####
ax4.plot(dates, ice, '-', color='purple', linewidth=1.25)
ax4.set_ylim(0, 100)
ax4.set_ylabel('SIC [%]', fontsize=22, labelpad=20, color='purple')
ax4.tick_params(axis='y', colors='purple')



##### Customize the plot #####

# Set x-axis limits and tick parameters
ax1.set_xlim(ts, te)
ax1.tick_params(length=7, direction='out')

ax1.spines['left'].set_color('red')
ax1.yaxis.label.set_color('red')

ax2.spines['right'].set_color('darkgreen')
ax2.yaxis.label.set_color('darkgreen')
ax2.yaxis.set_label_coords(1.05, 0.5)

ax3.spines['left'].set_position(('outward', 95))
ax3.spines['left'].set_color('darkblue')
ax3.yaxis.label.set_color('darkblue')
ax3.yaxis.set_label_coords(-0.15, 0.5)
ax3.yaxis.set_label_position('left')  
ax3.yaxis.set_ticks_position('left')

ax4.spines['right'].set_position(('outward', 95))
ax4.spines['right'].set_color('purple')
ax4.yaxis.label.set_color('purple')
ax4.yaxis.set_label_coords(1.125, 0.5)


# Set title
ax1.set_title('Amundsen-Bellingshausen Sea', fontsize=30)


outfile = r'.\Compound_MHW_HNPP_Amundsen_Bellingshausen_Sea.png'
fig.savefig(outfile, dpi=400, bbox_inches='tight')















## Check the area for case studies
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cft
import matplotlib.path as mpath

#lat and lon from MHWs
lat = loadmat(r'.\lat.mat')
lat = lat['latitud']

lon = loadmat(r'.\lon.mat')
lon = lon['longitud']
#Sea-ice mask
file = r'.\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
mask_without_nan = np.nan_to_num(mask, nan=1)


# Set the projection 
projection = ccrs.SouthPolarStereo(central_longitude=0)

# Create figure and axis
fig = plt.figure(figsize=(10, 5), dpi=600)
ax = plt.subplot(1, 1, 1, projection=projection)

# Add natural earth features for land and ice
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black')
ax.add_feature(land_50m)


# Define and plot the contour
ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', linewidths=1, transform=ccrs.PlateCarree())

# Set map extent and add coastlines
ax.set_extent([-280, 80, -90, -40], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', color='black', linewidth=1)

# Add a circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Plot predefined areas
areas = [
    {'lat_range': (-62, -55), 'lon_range': (79, 99)},      # Davis Sea
    {'lat_range': (-68, -60), 'lon_range': (-109, -89)},   # Amundsen-Bellingshausen
]
for area in areas:
    lat_range = area['lat_range']
    lon_range = area['lon_range']
    lats = np.arange(lat_range[0], lat_range[1] + 1, 1)
    lons = np.arange(lon_range[0], lon_range[1] + 1, 1)
    lat_mesh, lon_mesh = np.meshgrid(lats, lons)
    ax.plot(lon_mesh, lat_mesh, color='red', linewidth=1, transform=ccrs.PlateCarree())


outfile = r'.\Case_Studies_regions.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')


