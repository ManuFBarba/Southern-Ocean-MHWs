# -*- coding: utf-8 -*-
"""

############################# Fe Time Series ##################################

"""

# Load required modules
import numpy as np
from scipy import stats
from scipy.io import loadmat
import locale as locale
locale.setlocale(locale.LC_ALL,'en_US.utf8')
import xarray as xr 
import pandas as pd


import matplotlib.pyplot as plt

from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

ds = xr.open_mfdataset(r'.\cmems_mod_glo_bgc_my_0.25_P1M-m_1671706454697.nc', parallel=True)


lon = ds['longitude'][:]
lat = ds['latitude'][:]
times = ds['time'][:]

fe = ds['fe'][:,0,:,:]

#Extract different timeseries ranging in latitude
ds_90_60=ds.sel(latitude=slice(-80, -60))
ds_60_40=ds.sel(latitude=slice(-60, -40))


fe_total=ds['fe'][:,0,:,:].mean(dim=('longitude', 'latitude'),skipna=True)
fe_90_60=ds_90_60['fe'][:,0,:,:].mean(dim=('longitude', 'latitude'),skipna=True)
fe_60_40=ds_60_40['fe'][:,0,:,:].mean(dim=('longitude', 'latitude'),skipna=True)

fe_total = pd.DataFrame(fe_total)      #Visualize core.Dataarray and covert to Numpy Array
fe_total = np.squeeze(np.asarray(fe_total))
fe_total = fe_total[0:324]*1000

fe_90_60 = pd.DataFrame(fe_90_60)      
fe_90_60 = np.squeeze(np.asarray(fe_90_60))
fe_90_60 = fe_90_60[0:324]*1000

fe_60_40 = pd.DataFrame(fe_60_40)      
fe_60_40 = np.squeeze(np.asarray(fe_60_40))
fe_60_40 = fe_60_40[0:324]*1000

#Setting Time array
nt=fe.shape
base = dt(1994, 1, 1)
base_2015 = dt(2015, 1, 1)
arr_time_1994_2020 = np.array([base + relativedelta(months=+i) for i in range(nt[0])])

arr_time_1994_2015 = np.array([base + relativedelta(months=+i) for i in range(nt[0]-72)])

arr_time_2015_2020 = np.array([base_2015 + relativedelta(months=+i) for i in range(72)])




#Linear regressions
# res_fe_total_1994_2015 = stats.linregress(arr_time_1994_2015, fe_total[0:264])
# res_fe_total_2015_2020 = stats.linregress(arr_time_2015_2020, fe_total[252:324])



#Subplots N-SAT and SIC
fig, axs = plt.subplots(3, 1, figsize=(15, 10))
plt.rcParams.update({'font.size': 22})


axs[0].plot(arr_time_1994_2020, fe_total, 'k-o')
# axs[0].plot(time, res_sat_dec.intercept + res_sat_dec.slope*time, 'k--')
# axs[0].set_xlim(1994, 2020)
axs[0].set_ylim(0, 0.6)
axs[0].set_ylabel('[Fe] nM', fontsize=24)
axs[0].set_xticklabels([])
axs[0].tick_params(length=10, direction='in')
axs[0].set_title('Dissolved iron in sea water [0-26m]', fontsize=28)
# axs[0].grid(linestyle=':', linewidth=1)
axs[0].legend(['(a) 40ºS - 90ºS'], loc='upper left', frameon=False, fontsize=22)


axs[1].plot(arr_time_1994_2020, fe_60_40, 'k-o')
# axs[0].plot(time, res_sat_dec.intercept + res_sat_dec.slope*time, 'k--')

# axs[1].set_xlim(1994, 2020)
axs[1].set_ylim(0, 0.6)
axs[1].set_ylabel('[Fe] nM', fontsize=24)
axs[1].set_xticklabels([])
axs[1].tick_params(length=10, direction='in')
# axs[0].grid(linestyle=':', linewidth=1)
axs[1].legend(['(b) 40ºS - 60ºS'], loc='upper left', frameon=False, fontsize=22)



axs[2].plot(arr_time_1994_2020, fe_90_60, 'k-o')
# axs[2].set_xlim(1994, 2020)
axs[2].set_ylim(0, 0.6)
axs[2].set_ylabel('[Fe] nM', fontsize=24)
# axs[2].set_xticklabels([])
axs[2].tick_params(length=10, direction='in')
# axs[2].grid(linestyle=':', linewidth=1)
axs[2].legend(['(c) 60ºS - 90ºS'], loc='upper left', frameon=False, fontsize=22)

