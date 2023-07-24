# -*- coding: utf-8 -*-
"""

############################## Compound MHW - HNPP (or HChl) Event ######################

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


#Reading Datasets and variables
#SST
ds_sst = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SST_full\SST_ANT_1982-2021_40.nc')

#NPP / CHL
ds_npp = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CHL\Daily_chl\*.nc', parallel=True)


#N-SAT
ds_nsat = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\ERA5_datasets\ERA5_1982-2021_T2M_ANT.nc')

#SIC
ds_ice = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Sea_Ice_Conc_bimensual\*.nc', parallel=True)


#Load sea-ice mask
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
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

#Defining the lat, lon for the region of interest
lat_point_min = -67   # AP + Drake Passage Region
lat_point_max = -55

lon_point_min = -76
lon_point_max = -50


lat_point_min = -63   # Davis Sea
lat_point_max = -50

lon_point_min = 79
lon_point_max = 99


lat_point_min = -67   # Amundsen-Bellingshausen
lat_point_max = -60

lon_point_min = -120
lon_point_max = -80


lat_point_min = -63   # Cooperation-Cosmonauts sea
lat_point_max = -50

lon_point_min = 20
lon_point_max = 60



##SST region
lat_point_min_sst = -63   # AP + Drake Passage Region
lat_point_max_sst = -55

lon_point_min_sst = -76
lon_point_max_sst = -50


lat_point_min_sst = -59   # Davis Sea
lat_point_max_sst = -50

lon_point_min_sst = 79
lon_point_max_sst = 99


lat_point_min_sst = -65   # Amundsen-Bellingshausen
lat_point_max_sst = -60

lon_point_min_sst = -120
lon_point_max_sst = -80


lat_point_min_sst = -63   # Cooperation-Cosmonauts sea
lat_point_max_sst = -50

lon_point_min_sst = 20
lon_point_max_sst = 60




#Extracting SST, CHL, NSAT and SIC time series for a region of interest
sst = ds_sst.analysed_sst.sel(lat=slice(lat_point_min_sst, lat_point_max_sst), lon=slice(lon_point_min_sst, lon_point_max_sst))
sst = np.nanmean(sst, axis=(1, 2))
sst = sst - 273.15

npp = ds_npp.nppv.sel(latitude=slice(lat_point_min, lat_point_max), longitude=slice(lon_point_min, lon_point_max))
npp = np.nanmean(npp, axis=(1, 2, 3))

t2m = ds_nsat.t2m
t2m = t2m.sel(longitude=slice(lon_point_min_sst, lon_point_max_sst), latitude=slice(lat_point_max_sst, lat_point_min_sst))
t2m = np.nanmean(t2m, axis=(1, 2))
t2m = t2m - 273.15

ice = ds_ice.sea_ice_fraction.sel(lat=slice(lat_point_min, lat_point_max), lon=slice(lon_point_min, lon_point_max))
ice = np.nanmean(ice, axis=(1, 2))
ice = ice * 100



###################### Load files that were previously proccessed
file = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\sst_study_cases\sst_AP_Drake'
sst = np.load(file+'.npy')


file = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\sst_study_cases\sst_Davis_Sea'
sst = np.load(file+'.npy')

file = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\sst_study_cases\npp_averaged_Davis_Sea'
npp_averaged = np.load(file+'.npy')


file = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\sst_study_cases\sst_Amundsen_Bellingshausen'
sst = np.load(file+'.npy')

file = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\sst_study_cases\npp_averaged_Amundsen_Bellingshausen'
npp_averaged = np.load(file+'.npy')
######################





###############
# Applying Marine Heat Wave definition
###############

mhws, clim = mhw.detect(t_sst_nsat_ice, temp=sst, climatologyPeriod=[1982, 2012], pctile=95, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=False, coldSpells=coldSpells, alternateClimatology=False)
mhwBlock = mhw.blockAverage(t_sst_nsat_ice, mhws, clim=clim, temp=sst)
mean, trend, dtrend = mhw.meanTrend(mhwBlock)


###############
# Applying Marine Heat Wave definition to NPP values
###############

hnpp, clim_npp = mhw.detect(t_npp, temp=npp_averaged, climatologyPeriod=[1993, 2012], pctile=95, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=False, coldSpells=coldSpells, alternateClimatology=False)
nppBlock = mhw.blockAverage(t_npp, hnpp, clim=clim_npp, temp=npp_averaged)
mean, trend, dtrend = mhw.meanTrend(nppBlock)



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

window = 31 #South Shetland + Drake
# window = 7 #Davis Sea
# window = 7 #Ross Sea
t2m_averaged = rollavg_roll_edges(t2m, window)

npp_averaged = rollavg_roll_edges(npp, 5)

###
ev_int = np.argmax(mhws['intensity_max'])   # Find most intense event
ev_dur = np.argmax(mhws['duration'])        # Find largest event
ev = np.argmax(mhws)

ev_int_npp = np.argmax(hnpp['intensity_max'])   # Find most intense event
ev_dur_npp = np.argmax(hnpp['duration'])        # Find largest event
ev_npp = np.argmax(hnpp)





                    #####Davis Sea#####
plt.clf()

# Plot N-SAT, seasonal cycles, SST, [CHL], thresholds, shade MHWs and HChl events in red and green, respectively
# Set common time limits for both plots
ts = date(2015, 1, 1)
te = date(2018, 3, 31)

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(20, 15))
ax2 = ax1.twinx()
ax3 = ax1.twinx()

# Set font size
plt.rcParams.update({'font.size': 22})


##### Plot SST on the first y-axis (in red) #####
# Plot SST, seasonal cycle, and threshold
# new = (clim['thresh']) - 0.2
ax1.plot(dates, t2m_averaged+0.5, 'k:')
ax1.plot(dates, (clim['seas']), '-', color='grey')
ax1.plot(dates, sst, 'r-')
ax1.plot(dates, (clim['thresh']), 'r-.')
ax1.set_xlim(ts, te)
ax1.set_ylim(-6, 6.9)
ax1.set_ylabel(r'[$^\circ$C]', fontsize=20, labelpad=18, color='red')
ax1.tick_params(axis='y', colors='red')  # Set y-axis tick color

# Shade MHWs with the main event in red
for ev0 in np.arange(ev - 40, ev + 40, 1):
    t1 = np.where(t_sst_nsat_ice == mhws['time_start'][ev0])[0][0]
    t2 = np.where(t_sst_nsat_ice == mhws['time_end'][ev0])[0][0]
    ax1.fill_between(dates[t1 - 1: t2 + 1], sst[t1 - 1: t2 + 1], clim['thresh'][t1 - 1: t2 + 1], color='r', alpha=0.5)

# for ev0 in np.arange(ev - 3, ev - 1, 1):
#     t1 = np.where(t_sst_nsat_ice == mhws['time_start'][ev0])[0][0]
#     t2 = np.where(t_sst_nsat_ice == mhws['time_end'][ev0])[0][0]
#     ax1.fill_between(dates[t1 - 2: t2 + 4], sst[t1 - 2: t2 + 4], new[t1 - 2: t2 + 4], color='r', alpha=0.5)


##### Plot CHL on the second y-axis (in dark green) #####

# Plot CHL, seasonal cycle, threshold, and Hchl event
# new_chl = (clim_chl['thresh']) - 0.2
ax2.plot(dates_npp, (clim_npp['seas']), '-', color='grey')
ax2.plot(dates_npp, npp_averaged, '-', color='darkgreen')
ax2.plot(dates_npp, (clim_npp['thresh']), '-.', color='darkgreen')
ax2.set_ylim(0, 18)
# ax2.set_ylabel(r'CHL [$mg·m^{-3}$]', fontsize=22, labelpad=20, color='darkgreen')
ax2.set_ylabel(r'CbPM NPP [$mg C·m^{-3}·day^{-1}$]', fontsize=22, labelpad=20, color='darkgreen')
ax2.tick_params(axis='y', colors='darkgreen')  # Set y-axis tick color

# Shade Hchl events in dark green
for ev0 in np.arange(ev_npp - 40, ev_npp + 40, 1):
    t1 = np.where(t_npp == hnpp['time_start'][ev0])[0][0]
    t2 = np.where(t_npp == hnpp['time_end'][ev0])[0][0]
    ax2.fill_between(dates_npp[t1 - 1: t2 + 2], npp_averaged[t1 - 1: t2 + 2], clim_npp['thresh'][t1 - 1: t2 + 2], color='darkgreen', alpha=0.5)

# for ev0 in np.arange(ev_chl - 3, ev_chl - 1, 1):
#     t1 = np.where(t_chl == hchls['time_start'][ev0])[0][0]
#     t2 = np.where(t_chl == hchls['time_end'][ev0])[0][0]
#     ax2.fill_between(dates_chl[t1 - 2: t2 + 4], chl[t1 - 2: t2 + 4], new_chl[t1 - 2: t2 + 4], color='darkgreen', alpha=0.5)

##### Plot sea ice concentration on the third y-axis (in blue) #####
ax3.plot(dates, ice, '-', color='purple')
ax3.set_ylim(0, 100)
ax3.set_ylabel('SIC [%]', fontsize=22, labelpad=20, color='purple')
ax3.tick_params(axis='y', colors='purple')



##### Customize the plot #####

# Set x-axis limits and tick parameters
ax1.set_xlim(ts, te)
ax1.tick_params(length=7, direction='out')

ax1.spines['left'].set_color('red')
ax1.yaxis.label.set_color('red')

ax2.spines['right'].set_color('darkgreen')
ax2.yaxis.label.set_color('darkgreen')
ax2.yaxis.set_label_coords(1.05, 0.5)

ax3.spines['right'].set_position(('outward', 95))
ax3.spines['right'].set_color('purple')
ax3.yaxis.label.set_color('purple')
ax3.yaxis.set_label_coords(1.125, 0.5)

# Shade the periods of coinciding events
for ev0 in np.arange(ev - 20, ev + 20, 1):
    t1_mhw = np.where(t_sst_nsat_ice == mhws['time_start'][ev0])[0][0]
    t2_mhw = np.where(t_sst_nsat_ice == mhws['time_end'][ev0])[0][0]
    
    for ev_npp0 in np.arange(ev_npp - 40, ev_npp + 40, 1):
        t1_hnpp = np.where(t_npp == hnpp['time_start'][ev_npp0])[0][0]
        t2_hnpp = np.where(t_npp == hnpp['time_end'][ev_npp0])[0][0]
        
        if (t1_mhw <= t2_hnpp) and (t2_mhw >= t1_hnpp):
            t1 = max(t1_mhw, t1_hnpp)
            t2 = min(t2_mhw, t2_hnpp)
            ax1.axvspan(dates[t1 - 1], dates[t2], facecolor='yellow', alpha=0.3)


# Set legend
ax1.legend(['N-SAT', 'Climatological seasonal cycles', 'SST', '95th percentile SST', '_', 'MHW', '_', '_'],
            loc='upper left', frameon=False, fontsize=22)

# ax2.legend(['_', '[CHL]', '95th percentile [CHL]', 'Hchl events'],
#             loc='upper right', frameon=False, fontsize=22)
ax2.legend(['_', 'CbPM NPP', '95th percentile NPP', 'HNPP events'],
            loc='upper right', frameon=False, fontsize=22)


# Set title
ax1.set_title('Davis Sea', fontsize=30)
# ax1.set_title('Amundsen-Bellingshausen', fontsize=30)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figs_Explicacion_Reviews\Compound_MHW_NPP_Amundsen_Bellingshausen.png'
fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)


