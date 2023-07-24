############################# Averaged SIC and N-SAT #######################

#Loading requires modeles
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

##################################
## Near-Surface Air Temperature ##
##################################

ds_SAT = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\ERA5_datasets\ERA5_1982-2021_T2M_ANT.nc')

lon_SAT = ds_SAT['longitude'][:]
lat_SAT = ds_SAT['latitude'][:]
times = ds_SAT['time'][:]
time = times.astype('datetime64')
t2m = ds_SAT['t2m'][:,:,:] - 273.15


#Average SAT over lat and lon
SAT_ts=t2m.mean(dim=('longitude', 'latitude'), skipna=True) 

# SAT_ts = pd.DataFrame(SAT_ts)
# SAT_ts = np.squeeze(np.asarray(SAT_ts))

#Compute climatology [Reference period: 1982-2012]
ds_SAT_clim=ds_SAT.sel(time=slice("1982-01-01", "2012-12-31"))
SAT_clim=ds_SAT_clim['t2m'].groupby('time.month').mean(dim='time')#.load

#Compute N-SAT Anomalies
SAT_anom=ds_SAT['t2m'].groupby('time.month') - SAT_clim

#Average N-SAT Anomalies over lat and lon
SAT_anom=SAT_anom.mean(dim=('longitude', 'latitude'),skipna=True)

# SAT_anom_ts = pd.DataFrame(SAT_anom)
# SAT_anom_ts = np.squeeze(np.asarray(SAT_anom_ts))

#Group by months
df_SAT = pd.DataFrame({'Dates':time, 'N-SAT':SAT_anom})
df_SAT.set_index('Dates',inplace=True)
SAT_anom_monthly_avg = df_SAT.groupby([(df_SAT.index.year), (df_SAT.index.month)]).mean()


###########################
## Sea Ice Concentration ##
###########################

ds_SIC = xr.open_mfdataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Sea_Ice_Conc_bimensual\*.nc')

seaIce = ds_SIC['sea_ice_fraction'][:,:,:]
lon_seaIce = ds_SIC['lon'][:]
lat_seaIce = ds_SIC['lat'][:]

#Mask out all sea grid points, so we have only ice-zone values.
seaIce = xr.where(seaIce == 0, np.NaN, seaIce)

#Average SIC over lat and lon
SIC_ts = seaIce.mean(dim=('lon', 'lat'), skipna=True)
SIC_ts = SIC_ts * 100 #Convert to %
 
# SIC_ts = pd.DataFrame(SIC)
# SIC_ts = np.squeeze(np.asarray(SIC_ts))

#Group by months
df_SIC = pd.DataFrame({'Dates':time, 'SIC':SIC_ts})
SIC_monthly_avg = df_SIC.groupby([(df_SIC.index.year), (df_SIC.index.month)]).mean()


##Extract monthly variables
SAT_anom_jan = SAT_anom_monthly_avg[::12]
SAT_anom_feb = SAT_anom_monthly_avg[1::12]
SAT_anom_mar = SAT_anom_monthly_avg[2::12]
SAT_anom_apr = SAT_anom_monthly_avg[3::12]
SAT_anom_may = SAT_anom_monthly_avg[4::12]
SAT_anom_jun = SAT_anom_monthly_avg[5::12]
SAT_anom_jul = SAT_anom_monthly_avg[6::12]
SAT_anom_ago = SAT_anom_monthly_avg[7::12]
SAT_anom_sep = SAT_anom_monthly_avg[8::12]
SAT_anom_oct = SAT_anom_monthly_avg[9::12]
SAT_anom_nov = SAT_anom_monthly_avg[10::12]
SAT_anom_dec = SAT_anom_monthly_avg[11::12]

SIC_jan = SIC_monthly_avg[::12]
SIC_feb = SIC_monthly_avg[1::12]
SIC_mar = SIC_monthly_avg[2::12]
SIC_apr = SIC_monthly_avg[3::12]
SIC_may = SIC_monthly_avg[4::12]
SIC_jun = SIC_monthly_avg[5::12]
SIC_jul = SIC_monthly_avg[6::12]
SIC_ago = SIC_monthly_avg[7::12]
SIC_sep = SIC_monthly_avg[8::12]
SIC_oct = SIC_monthly_avg[9::12]
SIC_nov = SIC_monthly_avg[10::12]
SIC_dec = SIC_monthly_avg[11::12]


#Save SAT and SIC timeseries and monthly averaged so far
outfile = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MHW_metrics_full\Averaged_SAT_SIC'
np.savez(outfile, SAT_anom_ts=SAT_ts, SIC_ts=SIC_ts, SAT_anom_jan=SAT_anom_jan, SAT_anom_feb=SAT_anom_feb, SAT_anom_mar=SAT_anom_mar, SAT_anom_apr=SAT_anom_apr, SAT_anom_may=SAT_anom_may, SAT_anom_jun=SAT_anom_jun, SAT_anom_jul=SAT_anom_jul, SAT_anom_ago=SAT_anom_ago, SAT_anom_sep=SAT_anom_sep, SAT_anom_oct=SAT_anom_oct, SAT_anom_nov=SAT_anom_nov, SAT_anom_dec=SAT_anom_dec, SIC_jan=SIC_jan, SIC_feb=SIC_feb, SIC_mar=SIC_mar, SIC_apr=SIC_apr, SIC_may=SIC_may, SIC_jun=SIC_jun, SIC_jul=SIC_jul, SIC_ago=SIC_ago, SIC_sep=SIC_sep, SIC_oct=SIC_oct, SIC_nov=SIC_nov, SIC_dec=SIC_dec,)                             



#Computing a 3 years moving mean over N-SAT
def rollavg_roll_edges(a,n):
    'Numpy array rolling, edge handling'
    assert n%2==1
    a = np.pad(a,(0,n-1-n//2), 'constant')*np.ones(n)[:,None]
    m = a.shape[1]
    idx = np.mod((m-1)*np.arange(n)[:,None] + np.arange(m), m) # Rolling index
    out = a[np.arange(-n//2,n//2)[:,None], idx]
    d = np.hstack((np.arange(1,n),np.ones(m-2*n+1+n//2)*n,np.arange(n,n//2,-1)))
    return (out.sum(axis=0)/d)[n//2:]

window = 3

SAT_anom_nov_averaged = rollavg_roll_edges(SAT_anom_nov, window)
SAT_anom_dec_averaged = rollavg_roll_edges(SAT_anom_dec, window)
SAT_anom_jan_averaged = rollavg_roll_edges(SAT_anom_jan, window)
SAT_anom_feb_averaged = rollavg_roll_edges(SAT_anom_feb, window)
SAT_anom_mar_averaged = rollavg_roll_edges(SAT_anom_mar, window)


time = np.arange(1982, 2022)
time_1982_2015 = np.arange(1982, 2016)
time_2015_2021 = np.arange(2015, 2022)

#Linear regressions
res_sat_nov = stats.linregress(time, SAT_anom_nov_averaged)
res_sat_dec = stats.linregress(time, SAT_anom_dec_averaged)
res_sat_jan = stats.linregress(time, SAT_anom_jan_averaged)
res_sat_feb = stats.linregress(time, SAT_anom_feb_averaged)
res_sat_mar = stats.linregress(time, SAT_anom_mar_averaged)

#Subplots N-SAT and SIC
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
plt.rcParams.update({'font.size': 22})

axs[0].plot(time, SAT_anom_nov_averaged, 'k-', linewidth=2)
axs[0].plot(time, res_sat_nov.intercept + res_sat_nov.slope*time, 'k--', linewidth=2)

axs[0].plot(time, SAT_anom_dec_averaged, 'r-', linewidth=2)
axs[0].plot(time, res_sat_dec.intercept + res_sat_dec.slope*time, 'r--', linewidth=2)

axs[0].plot(time, SAT_anom_jan_averaged, 'g-', linewidth=2)
axs[0].plot(time, res_sat_jan.intercept + res_sat_jan.slope*time, 'g--', linewidth=2)

axs[0].plot(time, SAT_anom_feb_averaged, 'b-', linewidth=2)
axs[0].plot(time, res_sat_feb.intercept + res_sat_feb.slope*time, 'b--', linewidth=2)

axs[0].plot(time, SAT_anom_mar_averaged, 'y-', linewidth=2)
axs[0].plot(time, res_sat_mar.intercept + res_sat_mar.slope*time, 'y--', linewidth=2)


axs[0].set_xlim(1981, 2022)
axs[0].set_ylim(-1, 1)
axs[0].set_ylabel('[$^\circ$C]', fontsize=24)
axs[0].set_xticklabels([])
axs[0].tick_params(length=10, direction='in')
axs[0].set_title('(d) Average N-SAT Anomalies', fontsize=28)
# axs[0].grid(linestyle=':', linewidth=1)
axs[0].legend(['Nov', '_', 'Dec' , '_', 'Jan' , '_', 'Feb', '_', 'Mar', '_'], loc='upper left', frameon=False, fontsize=22, ncol = 5)


res_sic_nov_1982_2015 = stats.linregress(time_1982_2015, SIC_nov[0:34])
res_sic_nov_2015_2021 = stats.linregress(time_2015_2021, SIC_nov[33:40])

res_sic_dec_1982_2015 = stats.linregress(time_1982_2015, SIC_dec[0:34])
res_sic_dec_2015_2021 = stats.linregress(time_2015_2021, SIC_dec[33:40])

res_sic_jan_1982_2015 = stats.linregress(time_1982_2015, SIC_jan[0:34])
res_sic_jan_2015_2021 = stats.linregress(time_2015_2021, SIC_jan[33:40])

res_sic_feb_1982_2015 = stats.linregress(time_1982_2015, SIC_feb[0:34])
res_sic_feb_2015_2021 = stats.linregress(time_2015_2021, SIC_feb[33:40])

res_sic_mar_1982_2015 = stats.linregress(time_1982_2015, SIC_mar[0:34])
res_sic_mar_2015_2021 = stats.linregress(time_2015_2021, SIC_mar[33:40])

axs[1].plot(time, SIC_nov, 'k-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_nov_1982_2015.intercept + res_sic_nov_1982_2015.slope*time_1982_2015, 'k--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_nov_2015_2021.intercept + res_sic_nov_2015_2021.slope*time_2015_2021, 'k--', linewidth=2)

axs[1].plot(time, SIC_dec, 'r-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_dec_1982_2015.intercept + res_sic_dec_1982_2015.slope*time_1982_2015, 'r--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_dec_2015_2021.intercept + res_sic_dec_2015_2021.slope*time_2015_2021, 'r--', linewidth=2)

axs[1].plot(time, SIC_jan, 'g-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_jan_1982_2015.intercept + res_sic_jan_1982_2015.slope*time_1982_2015, 'g--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_jan_2015_2021.intercept + res_sic_jan_2015_2021.slope*time_2015_2021, 'g--', linewidth=2)

axs[1].plot(time, SIC_feb, 'b-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_feb_1982_2015.intercept + res_sic_feb_1982_2015.slope*time_1982_2015, 'b--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_feb_2015_2021.intercept + res_sic_feb_2015_2021.slope*time_2015_2021, 'b--', linewidth=2)

axs[1].plot(time, SIC_mar, 'y-', linewidth=2)
axs[1].plot(time_1982_2015, res_sic_feb_1982_2015.intercept + res_sic_feb_1982_2015.slope*time_1982_2015, 'y--', linewidth=2)
axs[1].plot(time_2015_2021, res_sic_feb_2015_2021.intercept + res_sic_feb_2015_2021.slope*time_2015_2021, 'y--', linewidth=2)



axs[1].set_xlim(1981, 2022)
axs[1].set_ylim(30, 80)
axs[1].set_ylabel('[%]', fontsize=24, labelpad=32)
axs[1].tick_params(length=10, direction='in')
axs[1].set_title('(e) Average Sea Ice Concentrations', fontsize=28)
# axs[1].grid(linestyle=':', linewidth=1)


#Save plots so far
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\NSAT_SIC.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\NSAT_SIC_hq.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight', pad_inches=0.5)

