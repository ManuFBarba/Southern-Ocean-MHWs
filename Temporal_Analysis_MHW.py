# -*- coding: utf-8 -*-
"""

############################## Temporal Analysis MHW (concrete point: lat, lon) #######################

"""

# Load required modules
 
import netCDF4 as nc
import numpy as np
from scipy import io
from datetime import date
from netCDF4 import Dataset 
import xarray as xr 
from matplotlib import pyplot as plt
import marineHeatWaves as mhw
from tempfile import TemporaryFile





# Some basic parameters
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



t = np.arange(date(1982,1,1).toordinal(),date(2021,12,31).toordinal()+1)
dates = [date.fromordinal(tt.astype(int)) for tt in t]


#Reading Datasets and variables
ds = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SST_full\SST_ANT_1982-2021_40.nc', 'r')

dz = Dataset(r'C:\ICMAN-CSIC\MHW_ANT\ERA5_datasets\ERA5_1982-2021_T2M_ANT.nc', 'r')
#sst = ds.variables['analysed_sst'][:,300:400,0:100] #Max subset latxlon (100x100) to open
#sst = ds['analysed_sst']
#times = ds.variables['time']
 
#Defining the lat, lon for the location of interest

lat_point = -58   #Drake Passage
lon_point = -61

lat_point = -61   #South Shetland Islands
lon_point = -59


# lat_point_min = -70   #AP Region
# lat_point_max = -62

# lon_point_min = -76
# lon_point_max = -55


lat_point = -60   #Davis Sea
lon_point = 88

lat_point = -63   #Ross Sea
lon_point = -178

# lat_point = -41     #South Atlantic Ocean 
# lon_point = -54

# lat_point = -44     #Tasman Sea
# lon_point = 150

#Storing the lat and lon data of the netCDF file into variables 
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]

lon_nsat = dz.variables['longitude'][:]
lat_nsat = dz.variables['latitude'][:]
    
#Squared difference between the specified lat,lon and the lat,lon of the SST dataset 
sq_diff_lat = (lat - lat_point)**2 
sq_diff_lon = (lon - lon_point)**2

#Squared difference between the specified lat,lon and the lat,lon of the NSAT dataset 
sq_diff_lat_nsat = (lat_nsat - lat_point)**2 
sq_diff_lon_nsat = (lon_nsat - lon_point)**2


#Identify the index of the min value for lat and lon in SST dataset
min_index_lat = sq_diff_lat.argmin()
min_index_lon = sq_diff_lon.argmin()

#Identify the index of the min value for lat and lon in NSAT dataset
min_index_lat_nsat = sq_diff_lat_nsat.argmin()
min_index_lon_nsat = sq_diff_lon_nsat.argmin()


#Extracting SST and NSAT time series for a concrete point
sst = ds.variables['analysed_sst'][:,min_index_lat,min_index_lon] - 273.15

t2m = dz.variables['t2m'][:,min_index_lat_nsat,min_index_lon_nsat] - 273.15


#Regions#
# sq_diff_lat_min = (lat - lat_point_min)**2 
# sq_diff_lat_max = (lat - lat_point_max)**2
# sq_diff_lon_min = (lat - lon_point_min)**2 
# sq_diff_lon_max = (lat - lon_point_max)**2

# min_index_lat_min = sq_diff_lat_min.argmin()
# min_index_lat_max = sq_diff_lat_max.argmin()
# min_index_lon_min = sq_diff_lon_min.argmin()
# min_index_lon_max = sq_diff_lon_max.argmin()


# sst = ds.variables['analysed_sst'][:,min_index_lat_min:min_index_lat_max,min_index_lon_min:min_index_lon_max]
# #K to ºC
# sst -= 273.15
# sst = np.mean(sst, axis=(1, 2))

# Generate synthetic temperature time series
#sst_syn = np.zeros(len(t))
#sst_syn[0] = 0 # Initial condition
#a = 0.85 # autoregressive parameter
#for i in range(1,len(t)):
#    sst_syn[i] = a*sst_syn[i-1] + 0.75*np.random.randn() + 0.5*np.cos(t[i]*2*np.pi/365.25)

#sst_syn = sst_syn - sst_syn.min() + 5.


#Alternate Climatology
#t1 = np.arange(date(1982,1,1).toordinal(),date(2021,12,31).toordinal()+1)
#dates = [date.fromordinal(tt.astype(int)) for tt in t]

#for i in range(0,len(t1)):
#    sst1 = sst[i]

#alternateClimatology = [t1, sst1]


###############
# Applying Marine Heat Wave definition
###############

mhws, clim = mhw.detect(t, temp=sst, climatologyPeriod=[1982, 2012], pctile=95, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=False, coldSpells=coldSpells, alternateClimatology=False)
mhwBlock = mhw.blockAverage(t, mhws, clim=clim, temp=sst)
mean, trend, dtrend = mhw.meanTrend(mhwBlock)


# Plot various summary things

plt.figure(figsize=(15,7))
plt.subplot(2,2,1)

#Duration MHW
evMax = np.argmax(mhws['duration'])
plt.bar(range(mhws['n_events']), mhws['duration'], width=0.6, color=(0.7,0.7,0.7))
plt.bar(evMax, mhws['duration'][evMax], width=0.6, color=col_bar)
plt.xlim(0, mhws['n_events'])
plt.ylabel('[days]')
plt.title('Duration')
plt.subplot(2,2,2)

#Intensity Max 
evMax = np.argmax(np.abs(mhws['intensity_max']))
plt.bar(range(mhws['n_events']), mhws['intensity_max'], width=0.6, color=(0.7,0.7,0.7))
plt.bar(evMax, mhws['intensity_max'][evMax], width=0.6, color=col_bar)
plt.xlim(0, mhws['n_events'])
plt.ylabel(r'[$^\circ$C]')
plt.title('Maximum Intensity')
plt.subplot(2,2,4)

#Mean Intensity
evMax = np.argmax(np.abs(mhws['intensity_mean']))
plt.bar(range(mhws['n_events']), mhws['intensity_mean'], width=0.6, color=(0.7,0.7,0.7))
plt.bar(evMax, mhws['intensity_mean'][evMax], width=0.6, color=col_bar)
plt.xlim(0, mhws['n_events'])
plt.title('Mean Intensity')
plt.ylabel(r'[$^\circ$C]')
plt.xlabel(mhwname + ' event number')
plt.subplot(2,2,3)

#Intensity cumulative 
evMax = np.argmax(np.abs(mhws['intensity_cumulative']))
plt.bar(range(mhws['n_events']), mhws['intensity_cumulative'], width=0.6, color=(0.7,0.7,0.7))
plt.bar(evMax, mhws['intensity_cumulative'][evMax], width=0.6, color=col_bar)
plt.xlim(0, mhws['n_events'])
plt.title(r'Cumulative Intensity')
plt.ylabel(r'[$^\circ$C$\times$days]')
plt.xlabel(mhwname + ' event number')
#plt.savefig(r'C:\ICMAN-CSIC\MHW_ANT\Figures_MHW' + mhwname + '_list_byNumber.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

#ts = date(1982,1,1).toordinal()
#te = date(1990,12,31).toordinal()
ts = date(1982,1,1)
te = date(2021,12,31)

plt.figure(figsize=(15,7))
plt.subplot(2,2,1)

evMax = np.argmax(mhws['duration'])
plt.bar(mhws['date_peak'], mhws['duration'], width=150, color=(0.7,0.7,0.7))
plt.bar(mhws['date_peak'][evMax], mhws['duration'][evMax], width=150, color=col_bar)
plt.xlim(ts, te)
plt.ylabel('[days]')
plt.title('Duration')
plt.subplot(2,2,2)
evMax = np.argmax(np.abs(mhws['intensity_max']))
plt.bar(mhws['date_peak'], mhws['intensity_max'], width=150, color=(0.7,0.7,0.7))
plt.bar(mhws['date_peak'][evMax], mhws['intensity_max'][evMax], width=150, color=col_bar)
plt.xlim(ts, te)
plt.ylabel(r'[$^\circ$C]')
plt.title('Maximum Intensity')
plt.subplot(2,2,4)
evMax = np.argmax(np.abs(mhws['intensity_mean']))
plt.bar(mhws['date_peak'], mhws['intensity_mean'], width=150, color=(0.7,0.7,0.7))
plt.bar(mhws['date_peak'][evMax], mhws['intensity_mean'][evMax], width=150, color=col_bar)
plt.xlim(ts, te)
plt.title('Mean Intensity')
plt.ylabel(r'[$^\circ$C]')
plt.subplot(2,2,3)
evMax = np.argmax(np.abs(mhws['intensity_cumulative']))
plt.bar(mhws['date_peak'], mhws['intensity_cumulative'], width=150, color=(0.7,0.7,0.7))
plt.bar(mhws['date_peak'][evMax], mhws['intensity_cumulative'][evMax], width=150, color=col_bar)
plt.xlim(ts, te)
plt.title(r'Cumulative Intensity')
plt.ylabel(r'[$^\circ$C$\times$days]')
#plt.savefig('Figures_MHW/' + mhwname + '_list_byDate.png', bbox_inches='tight', pad_inches=0.5, dpi=150)


# Annual averages
years = mhwBlock['years_centre']
plt.figure(figsize=(13,7))
plt.subplot(2,2,2)
plt.plot(years, mhwBlock['count'], 'k-')
plt.plot(years, mhwBlock['count'], 'ko')
if np.abs(trend['count']) - dtrend['count'] > 0:
     plt.title('Frequency (trend = ' + '{:.2}'.format(10*trend['count']) + '* per decade)')
else:
     plt.title('Frequency (trend = ' + '{:.2}'.format(10*trend['count']) + ' per decade)')
plt.ylabel('[count per year]')
plt.grid()
plt.subplot(2,2,1)
plt.plot(years, mhwBlock['duration'], 'k-')
plt.plot(years, mhwBlock['duration'], 'ko')
if np.abs(trend['duration']) - dtrend['duration'] > 0:
    plt.title('Duration (trend = ' + '{:.2}'.format(10*trend['duration']) + '* per decade)')
else:
    plt.title('Duration (trend = ' + '{:.2}'.format(10*trend['duration']) + ' per decade)')
plt.ylabel('[days]')
plt.grid()
plt.subplot(2,2,4)
plt.plot(years, mhwBlock['intensity_max'], '-', color=col_evMax)
plt.plot(years, mhwBlock['intensity_mean'], 'k-')
plt.plot(years, mhwBlock['intensity_max'], 'o', color=col_evMax)
plt.plot(years, mhwBlock['intensity_mean'], 'ko')
plt.legend(['Max', 'mean'], loc=2)
if (np.abs(trend['intensity_max']) - dtrend['intensity_max'] > 0) * (np.abs(trend['intensity_mean']) - dtrend['intensity_mean'] > 0):
    plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend['intensity_max']) + '* (max), ' + '{:.2}'.format(10*trend['intensity_mean'])  + '* (mean) per decade)')
elif (np.abs(trend['intensity_max']) - dtrend['intensity_max'] > 0):
    plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend['intensity_max']) + '* (max), ' + '{:.2}'.format(10*trend['intensity_mean'])  + ' (mean) per decade)')
elif (np.abs(trend['intensity_mean']) - dtrend['intensity_mean'] > 0):
    plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend['intensity_max']) + ' (max), ' + '{:.2}'.format(10*trend['intensity_mean'])  + '* (mean) per decade)')
else:
    plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend['intensity_max']) + ' (max), ' + '{:.2}'.format(10*trend['intensity_mean'])  + ' (mean) per decade)')
plt.ylabel(r'[$^\circ$C]')
plt.grid()
plt.subplot(2,2,3)
plt.plot(years, mhwBlock['intensity_cumulative'], 'k-')
plt.plot(years, mhwBlock['intensity_cumulative'], 'ko')
if np.abs(trend['intensity_cumulative']) - dtrend['intensity_cumulative'] > 0:
    plt.title('Cumulative intensity (trend = ' + '{:.2}'.format(10*trend['intensity_cumulative']) + '* per decade)')
else:
    plt.title('Cumulative intensity (trend = ' + '{:.2}'.format(10*trend['intensity_cumulative']) + ' per decade)')
plt.ylabel(r'[$^\circ$C$\times$days]')
plt.grid()
#plt.savefig('mhw_stats/' + mhwname + '_annualAverages_meanTrend.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

# Save results as text data
#outfile = 'mhw_stats/' + mhwname + '_data'


##################################################### 

## Calculating Long-term mean summer temperature

#Identifying the index of the day of the year when the daily SST climatology is at maximum
maxTIdx = clim['seas'].argmax()

#Calculating the 91-day average SST centered on this date
sstLmst = clim['seas'][maxTIdx-45:maxTIdx+45]
#sstLmst = clim['seas'][0:maxTIdx+45]
lmstvalue = np.nanmean(sstLmst)



#Setting LMST all over the SST time serie
syn = np.zeros(len(sst))
lmst = syn + lmstvalue


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
t2m_averaged = rollavg_roll_edges(t2m, window)


###
ev_int = np.argmax(mhws['intensity_max'])   # Find most intense event
ev_dur = np.argmax(mhws['duration'])        # Find largest event
ev = np.argmax(mhws)




ts = date(2017,1,1)
te = date(2021,12,31)

# Plot SST, seasonal cycle, threshold, LMST, shade MHWs with main event in red

fig, ax = plt.subplots(figsize=(15, 5))
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)

# Plot SST, seasonal cycle, and threshold
plt.plot(dates, t2m_averaged, 'k:')
plt.plot(dates, sst, 'k-')
plt.plot(dates, clim['seas'], 'b-')
plt.plot(dates, clim['thresh'], 'g-')
plt.plot(dates, lmst, 'r-')
plt.title('Ross Sea (63S, 178W)', fontsize=30)
plt.xlim(ts, te)
# plt.ylim(sst.min()-0.5, sst.max()+0.5)
plt.ylim(-2, 5.5)
plt.ylabel(r'SST [$^\circ$C]', fontsize=18)
#plt.subplot(2,1,2)
# Find indices for all five MHWs before and after event of interest and shade accordingly
for ev0 in np.arange(ev-15, ev-14, 1):
    # Find indices for MHW of interest (2020 event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev0])[0][0]
    t2 = np.where(t==mhws['time_end'][ev0])[0][0]
    #if sst[t1:t2+1].any() > lmst.all():
    plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], \
                    color='r')

for ev1 in np.arange(ev-20, ev+20, 1):
    # Find indices for MHW of interest (2020 event) and shade accordingly
    t3 = np.where(t==mhws['time_start'][ev1])[0][0]
    t4 = np.where(t==mhws['time_end'][ev1])[0][0]
    #if sst[t1:t2+1].any() > lmst.all():
    plt.fill_between(dates[t3:t4+1], sst[t3:t4+1], clim['thresh'][t3:t4+1], \
                color='r')

# for ev1 in np.arange(ev-11, ev-8, 1):
#     # Find indices for MHW of interest (2020 event) and shade accordingly
#     t3 = np.where(t==mhws['time_start'][ev1])[0][0]
#     t4 = np.where(t==mhws['time_end'][ev1])[0][0]
#     #if sst[t1:t2+1].any() > lmst.all():
#     plt.fill_between(dates[t3:t4+1], sst[t3:t4+1], clim['thresh'][t3:t4+1], \
#                 color='r')
        
        
#leg=plt.legend(['N-SAT', 'SST', 'SSTc', '95th percentile SST', 'LMST'], loc='upper right', fontsize=8)


################################# Zoom in #####################################
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

# window = 9 #South Shetland
window = 7 #Drake Passage
# window = 7 #Davis Sea
# window = 7 #Ross Sea
t2m_averaged = rollavg_roll_edges(t2m, window)

###
ev_int = np.argmax(mhws['intensity_max'])   # Find most intense event
ev_dur = np.argmax(mhws['duration'])        # Find largest event
ev = np.argmax(mhws)


#####South Shetland Islands#####
new=(clim['thresh'])-0.2

ts = date(2019,12,1)
te = date(2020,5,1)

# Plot SST, seasonal cycle, threshold, LMST, shade MHWs with main event in red
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(figsize=(15, 5))
# Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(18)

# Plot SST, seasonal cycle, and threshold
plt.plot(dates, t2m_averaged, 'k:')
plt.plot(dates, sst, 'k-')
plt.plot(dates, (clim['seas'])-0.2, 'b-')
plt.plot(dates, (clim['thresh'])-0.2, 'g-')
plt.plot(dates, lmst-0.2, 'r-')
plt.title('South Shetland Islands (61S, 59W)', fontsize=20)
plt.xlim(ts, te)
plt.ylim(-0.5, 4.8)
plt.ylabel(r'[$^\circ$C]', fontsize=20, labelpad=18)
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
plt.tick_params(length=7, direction='out')
# Find indices for all five MHWs before and after event of interest and shade accordingly
for ev0 in np.arange(ev-1, ev+1, 1):
    # Find indices for MHW of interest (2020 event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev0])[0][0]
    t2 = np.where(t==mhws['time_end'][ev0])[0][0]
    #if sst[t1:t2+1].any() > lmst.all():
    plt.fill_between(dates[t1-1:t2+9], sst[t1-1:t2+9], new[t1-1:t2+9], \
                     color='r')

for ev0 in np.arange(ev-3, ev-1, 1):
    # Find indices for MHW of interest (2020 event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev0])[0][0]
    t2 = np.where(t==mhws['time_end'][ev0])[0][0]
    #if sst[t1:t2+1].any() > lmst.all():
    plt.fill_between(dates[t1-2:t2+4], sst[t1-2:t2+4], new[t1-2:t2+4], \
                      color='r')



leg=plt.legend(['N-SAT', 'SST', 'SSTc', '95th percentile SST', 'LMST'], loc='upper left', frameon=False, fontsize=18)


plt.plot(dates, sst, 'k-')
plt.plot(dates, clim['thresh']-0.2, 'g-')

outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_SouthShetlands.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_SouthShetlands_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


#####Drake Passage#####
ts = date(2019,12,1)
te = date(2020,5,1)

# Plot SST, seasonal cycle, threshold, LMST, shade MHWs with main event in red
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(figsize=(15, 5))

# Plot SST, seasonal cycle, and threshold
plt.plot(dates, t2m_averaged, 'k:')
plt.plot(dates, sst, 'k-')
plt.plot(dates, clim['seas'], 'b-')
plt.plot(dates, clim['thresh'], 'g-')
plt.plot(dates, lmst, 'r-')
plt.title('Drake Passage (58S, 61W)', fontsize=20)
plt.xlim(ts, te)
#plt.ylim(sst.min()-0.5, sst.max()+0.5)
plt.ylim(2.5, 8)
plt.ylabel(r'[$^\circ$C]', fontsize=20, labelpad=18)
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
plt.tick_params(length=7, direction='out')
#plt.subplot(2,1,2)
# Find indices for all five MHWs before and after event of interest and shade accordingly
for ev0 in np.arange(ev-9, ev+20, 1):
    # Find indices for MHW of interest (2020 event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev0])[0][0]
    t2 = np.where(t==mhws['time_end'][ev0])[0][0]
    #if sst[t1:t2+1].any() > lmst.all():
    plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], \
                     color='r')

#leg=plt.legend(['N-SAT', 'SST', 'SSTc', '95th percentile SST', 'LMST'], loc='upper left', frameon=False, fontsize=17)


plt.plot(dates, sst, 'k-')
plt.plot(dates, clim['thresh'], 'g-')

#Save plots so far
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_DrakePassage.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_DrakePassage_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



#####Davis Sea#####
ts = date(2019,12,1)
te = date(2020,5,1)

# Plot SST, seasonal cycle, threshold, LMST, shade MHWs with main event in red
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(figsize=(15, 5))

# Plot SST, seasonal cycle, and threshold
plt.plot(dates, t2m_averaged+1, 'k:')
plt.plot(dates, sst, 'k-')
plt.plot(dates, clim['seas'], 'b-')
plt.plot(dates, clim['thresh'], 'g-')
plt.plot(dates, lmst, 'r-')
plt.title('Davis Sea (60S, 88E)', fontsize=20)
plt.xlim(ts, te)
#plt.ylim(sst.min()-0.5, sst.max()+0.5)
plt.ylim(-0.5, 5)
plt.ylabel(r'[$^\circ$C]', fontsize=20, labelpad=18)
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
plt.tick_params(length=7, direction='out')
#plt.subplot(2,1,2)
# Find indices for all five MHWs before and after event of interest and shade accordingly
for ev0 in np.arange(ev-20, ev+20, 1):
    # Find indices for MHW of interest (2020 event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev0])[0][0]
    t2 = np.where(t==mhws['time_end'][ev0])[0][0]
    #if sst[t1:t2+1].any() > lmst.all():
    plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], \
                     color='r')
        
#leg=plt.legend(['N-SAT', 'SST', 'SSTc', '95th percentile SST', 'LMST'], loc='upper left', frameon=False, fontsize=17)

plt.plot(dates, sst, 'k-')
plt.plot(dates, clim['thresh'], 'g-')


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_DavisSea.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_DavisSea_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



#####Ross Sea#####
ts = date(2019,11,1)
te = date(2020,5,1)

# Plot SST, seasonal cycle, threshold, LMST, shade MHWs with main event in red
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True 
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(figsize=(15, 5))

# Plot SST, seasonal cycle, and threshold
plt.plot(dates, t2m_averaged, 'k:')
plt.plot(dates, sst, 'k-')
plt.plot(dates, clim['seas'], 'b-')
plt.plot(dates, clim['thresh'], 'g-')
plt.plot(dates, lmst, 'r-')
plt.title('Ross Sea (63S, 178E)', fontsize=20)
plt.xlim(ts, te)
#plt.ylim(sst.min()-0.5, sst.max()+0.5)
plt.ylim(-1, 5.3)
plt.ylabel(r'[$^\circ$C]', fontsize=20, labelpad=18)
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
plt.tick_params(length=7, direction='out')
# Find indices for all five MHWs before and after event of interest and shade accordingly
for ev0 in np.arange(ev-40, ev+40, 1):
    # Find indices for MHW of interest (2020 event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev0])[0][0]
    t2 = np.where(t==mhws['time_end'][ev0])[0][0]
    #if sst[t1:t2+1].any() > lmst.all():
    plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], \
                     color='r')

#leg=plt.legend(['N-SAT', 'SST', 'SSTc', '95th percentile SST', 'LMST'], loc='upper left', frameon=False, fontsize=17)

plt.plot(dates, sst, 'k-')
plt.plot(dates, clim['thresh'], 'g-')


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_RossSea.png'
fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\MHW_RossSea_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)











#Comprobar puntos en mapa

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


levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
p1 = plt.contourf(lon, lat, MHW_max+mask, levels, cmap=plt.cm.YlOrRd, extend='both', transform=ccrs.PlateCarree()) 
cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=25) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([0, 1, 2, 3, 4, 5])
#p1 = plt.pcolormesh(lon, lat, MHW_dur_tr, vmin=-2, vmax=2, cmap=cmap, transform=ccrs.PlateCarree())

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('x) Maximum intensity [$^\circ$C]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)





lat_point_p1 = -61  #South Shetlands
lon_point_p1 = -60


p1=plt.plot(lon_point_p1,lat_point_p1,color='black',markeredgecolor='black',linewidth=3,marker='o',markersize=12,transform=ccrs.Geodetic())


lat_point_p2 = -58   #Drake Passage
lon_point_p2 = -61



p2=plt.plot(lon_point_p2,lat_point_p2,color='black',markeredgecolor='black',linewidth=3,marker='o',markersize=12,transform=ccrs.Geodetic())




lat_point_p3 = -60    #Davis Sea
lon_point_p3 = 88



p3=plt.plot(lon_point_p3,lat_point_p3,color='black',markeredgecolor='black',linewidth=3,marker='o',markersize=12,transform=ccrs.Geodetic())



lat_point_p4 = -63    #Ross Sea
lon_point_p4 = 178



p4=plt.plot(lon_point_p4,lat_point_p4,color='black',markeredgecolor='black',linewidth=3,marker='o',markersize=12,transform=ccrs.Geodetic())








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

#cmap=plt.cm.YlOrRd  
#cmap=plt.cm.RdBu_r  
#cmap = 'Spectral_r'
levels = [2.5,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30]
p1 = plt.contourf(lon, lat, MHW_dur_ts[:,:,35]+mask, levels, cmap=plt.cm.YlOrRd, extend='both', transform=ccrs.PlateCarree()) 
cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=25) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([5, 10, 15, 20, 25, 30])
#p1 = plt.pcolormesh(lon, lat, MHW_dur_tr, vmin=-2, vmax=2, cmap=cmap, transform=ccrs.PlateCarree())

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('e) Duration [$days$]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


lat_point_p1 = -61  #South Shetlands
lon_point_p1 = -60


p1=plt.plot(lon_point_p1,lat_point_p1,color='black',markeredgecolor='black',linewidth=3,marker='o',markersize=12,transform=ccrs.Geodetic())


lat_point_p2 = -58   #Drake Passage
lon_point_p2 = -61



p2=plt.plot(lon_point_p2,lat_point_p2,color='black',markeredgecolor='black',linewidth=3,marker='o',markersize=12,transform=ccrs.Geodetic())




lat_point_p3 = -60    #Davis Sea
lon_point_p3 = 88



p3=plt.plot(lon_point_p3,lat_point_p3,color='black',markeredgecolor='black',linewidth=3,marker='o',markersize=12,transform=ccrs.Geodetic())



lat_point_p4 = -63    #Ross Sea
lon_point_p4 = 178



p4=plt.plot(lon_point_p4,lat_point_p4,color='black',markeredgecolor='black',linewidth=3,marker='o',markersize=12,transform=ccrs.Geodetic())







