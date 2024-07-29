# -*- coding: utf-8 -*-
"""

########################## MHWs, NPP, MLD  ############################

"""


#Loading required modules
import seaborn as sns
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt

from scipy.io import loadmat


## Loading and preparing datasets
# file = r'.\mask'
# data_mask = np.load(file+'.npz')
# mask = data_mask['mask']
# mask_ts = mask[:, :, np.newaxis]

# ds_SST = xr.open_dataset(r'./SST_ANT_1982-2021_40.nc')
# SST = ds_SST.analysed_sst[:,::10,::10] - 273.15
# SST.to_netcdf(r'.\Causality_SST_NPP/Dialy_SST.nc')
# SST = xr.open_dataset(r'./Dialy_SST.nc')

# NPP_interp_monthly = loadmat(r'.\CbPM_interp_monthly.mat')
# NPP = NPP_interp_monthly['CbPM_interp_monthly']
# NPP = NPP + mask_ts

# ds_SIC = xr.open_mfdataset(r'.\Sea_Ice_Conc_bimensual/*.nc', parallel=True)
# SIC = ds_SIC['sea_ice_fraction'][:,::10,::10] 
# SIC_filtered = SIC.sel(time=slice('1998-01-01', '2021-12-31'))
# SIC_monthly = SIC_filtered.resample(time='M').mean(skipna=True)
# SIC_monthly = SIC_monthly*100 #SIF to SIC (%)
# SIC_monthly.to_netcdf(r'./SIC_monthly.nc')

# ds_MLD = xr.open_dataset(r'./MLD_monthly.nc')
##




# -------------- MHWs ------------------------
## Calculating monthly MHWs (SST > percentile95(SST) for at least 5 consecutive days; Hobday et al., 2016)
# ref_period = SST.sel(time=slice('1982-01-01', '2012-12-31'))
# SST_period = SST.sel(time=slice('1998-01-01', '2021-12-31'))

# percentile_95 = ref_period.groupby('time.dayofyear').reduce(np.nanpercentile, q=95, dim='time')

# mhw_days = SST_period.groupby('time.dayofyear') > percentile_95

# monthly_mhws = mhw_days.resample(time='M').sum(dim='time') >= 5
# monthly_mhws.to_netcdf(r'./monthly_mhws.nc')

monthly_mhws = xr.open_dataset(r'./monthly_mhws.nc')
months = monthly_mhws['time'].dt.month
summer_mask = months.isin([11, 12, 1, 2, 3]) #November to March
AustralSummer_mhw = monthly_mhws.sel(time=summer_mask)
#Monthly MHW serie (pixelwise count)
# true_sums = AustralSummer_mhw['analysed_sst'].sum(dim=['lat', 'lon'])


# -------------- NPP ------------------------
## Creating the NPP xarrayDataArray 
# npp_corrected = np.transpose(NPP, (2, 1, 0))
# npp_xarray_corrected = xr.DataArray(npp_corrected, dims=["time", "lat", "lon"], coords={"time": monthly_mhws["time"], "lat": monthly_mhws["lat"], "lon": monthly_mhws["lon"]}, name="NPP")

## Averaging NPP over months with MHWs and without MHWs
# AustralSummer_npp_mhw = npp_xarray_corrected.where(AustralSummer_mhw["analysed_sst"]).mean(dim="time", skipna=True)
# AustralSummer_npp_no_mhw = npp_xarray_corrected.where(~AustralSummer_mhw["analysed_sst"]).mean(dim="time", skipna=True)

# AustralSummer_npp_mhw.to_netcdf(r'./AustralSummer_npp_mhw.nc')
# AustralSummer_npp_no_mhw.to_netcdf(r'./AustralSummer_npp_no_mhw.nc')

AustralSummer_NPP_mhw = xr.open_dataset(r'./AustralSummer_npp_mhw.nc')
AustralSummer_NPP_no_mhw = xr.open_dataset(r'./AustralSummer_npp_no_mhw.nc')
# --------------     ------------------------


# -------------- SIC ------------------------
# SIC_monthly = xr.open_dataset(r'./SIC_monthly.nc')
# AustralSummer_SIC_mhw = SIC_monthly.where(AustralSummer_mhw["analysed_sst"]).mean(dim="time", skipna=True)
# AustralSummer_SIC_mhw['sea_ice_fraction'] = AustralSummer_SIC_mhw['sea_ice_fraction'].where(AustralSummer_SIC_mhw['sea_ice_fraction'] > 0, np.nan)

# AustralSummer_SIC_no_mhw = SIC_monthly.where(~AustralSummer_mhw["analysed_sst"]).mean(dim="time", skipna=True)
# AustralSummer_SIC_no_mhw['sea_ice_fraction'] = AustralSummer_SIC_no_mhw['sea_ice_fraction'].where(AustralSummer_SIC_no_mhw['sea_ice_fraction'] > 0, np.nan)

# AustralSummer_SIC_mhw.to_netcdf(r'./AustralSummer_SIC_mhw.nc')
# AustralSummer_SIC_no_mhw.to_netcdf(r'./AustralSummer_SIC_no_mhw.nc')

AustralSummer_SIC_mhw = xr.open_dataset(r'./AustralSummer_SIC_mhw.nc')
AustralSummer_SIC_no_mhw = xr.open_dataset(r'./AustralSummer_SIC_no_mhw.nc')
# --------------     ------------------------




# -------------- MLD ------------------------
# ds_MLD = xr.open_dataset(r'./MLD_monthly.nc')
# ds_MLD = ds_MLD.transpose('time', 'lat', 'lon')
# ds_MLD['time'] = monthly_mhws['time']

# AustralSummer_MLD_mhw = ds_MLD.where(AustralSummer_mhw["analysed_sst"]).mean(dim="time", skipna=True)
# AustralSummer_MLD_no_mhw = ds_MLD.where(~AustralSummer_mhw["analysed_sst"]).mean(dim="time", skipna=True)

# AustralSummer_MLD_mhw.to_netcdf(r'./AustralSummer_MLD_mhw.nc')
# AustralSummer_MLD_no_mhw.to_netcdf(r'./AustralSummer_MLD_no_mhw.nc')

AustralSummer_MLD_mhw = xr.open_dataset(r'./AustralSummer_MLD_mhw.nc')
AustralSummer_MLD_no_mhw = xr.open_dataset(r'./AustralSummer_MLD_no_mhw.nc')
# --------------     ------------------------




## Representing the subplots

## NPP Data
npp_mhw_flattened = AustralSummer_NPP_mhw['NPP'].values.flatten()
npp_no_mhw_flattened = AustralSummer_NPP_no_mhw['NPP'].values.flatten()
labels_no_mhw_npp = ['No MHW'] * len(npp_no_mhw_flattened)
labels_mhw_npp = ['MHW'] * len(npp_mhw_flattened)
df_npp = pd.DataFrame({
    'NPP': np.concatenate([npp_no_mhw_flattened, npp_mhw_flattened]),
    'Condition': labels_no_mhw_npp + labels_mhw_npp
})
# "No MHW" Info
n_no_mhw_npp = len(df_npp[df_npp['Condition'] == "No MHW"])  # Number of obs
max_no_mhw_npp = df_npp[df_npp['Condition'] == "No MHW"]['NPP'].max()  # Maximum Value
min_no_mhw_npp = df_npp[df_npp['Condition'] == "No MHW"]['NPP'].min()  # Minimum value
# "MHW" Info
n_mhw_npp = len(df_npp[df_npp['Condition'] == "MHW"])  
max_mhw_npp = df_npp[df_npp['Condition'] == "MHW"]['NPP'].max()  
min_mhw_npp = df_npp[df_npp['Condition'] == "MHW"]['NPP'].min()  



## SIC Data
sic_mhw_flattened = AustralSummer_SIC_mhw['sea_ice_fraction'].values.flatten()
sic_no_mhw_flattened = AustralSummer_SIC_no_mhw['sea_ice_fraction'].values.flatten()
labels_no_mhw_sic = ['No MHW'] * len(sic_no_mhw_flattened)
labels_mhw_sic = ['MHW'] * len(sic_mhw_flattened)
df_sic = pd.DataFrame({
    'SIC': np.concatenate([sic_no_mhw_flattened, sic_mhw_flattened]),
    'Condition': labels_no_mhw_sic + labels_mhw_sic
})
# "No MHW" Info
n_no_mhw_sic = len(df_sic[df_sic['Condition'] == "No MHW"])  # Number of obs
max_no_mhw_sic = df_sic[df_sic['Condition'] == "No MHW"]['SIC'].max()  # Maximum Value
min_no_mhw_sic = df_sic[df_sic['Condition'] == "No MHW"]['SIC'].min()  # Minimum value
# "MHW" Info
n_mhw_sic = len(df_sic[df_sic['Condition'] == "MHW"])  
max_mhw_sic = df_sic[df_sic['Condition'] == "MHW"]['SIC'].max()  
min_mhw_sic = df_sic[df_sic['Condition'] == "MHW"]['SIC'].min()  



## MLD data
mld_mhw_flattened = AustralSummer_MLD_mhw['mlotst'].values.flatten()
mld_no_mhw_flattened = AustralSummer_MLD_no_mhw['mlotst'].values.flatten()
labels_no_mhw_mld = ['No MHW'] * len(mld_no_mhw_flattened)
labels_mhw_mld = ['MHW'] * len(mld_mhw_flattened)
df_mld = pd.DataFrame({
    'MLD': np.concatenate([mld_no_mhw_flattened, mld_mhw_flattened]),
    'Condition': labels_no_mhw_mld + labels_mhw_mld
})
# "No MHW" Info
n_no_mhw_mld = len(df_mld[df_mld['Condition'] == "No MHW"])  # Number of obs
max_no_mhw_mld = df_mld[df_mld['Condition'] == "No MHW"]['MLD'].max()  # Maximum Value
min_no_mhw_mld = df_mld[df_mld['Condition'] == "No MHW"]['MLD'].min()  # Minimum value
# "MHW" Info
n_mhw_mld = len(df_mld[df_mld['Condition'] == "MHW"])  
max_mhw_mld = df_mld[df_mld['Condition'] == "MHW"]['MLD'].max()  
min_mhw_mld = df_mld[df_mld['Condition'] == "MHW"]['MLD'].min()



palette = {"No MHW": "#0052B7", "MHW": "#00B100"}  # Blue for 'No MHW' condition and Green for 'MHW'

fig, axes = plt.subplots(1, 3, figsize=(10, 6))  

titulo_fuente = 11
etiqueta_fuente = 10
texto_fuente = 8
ticks_fuente = 10  

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.5, wspace=0.3)

# NPP
sns.violinplot(x="Condition", y="NPP", data=df_npp, palette=palette, ax=axes[0])
axes[0].set_title('NPP distributions', fontsize=titulo_fuente)
axes[0].set_xlabel('')
axes[0].set_ylabel('CbPM NPP [mgC m$^{-2}$ day$^{-1}$]', fontsize=etiqueta_fuente)
axes[0].set_ylim(0, 1200)
axes[0].tick_params(axis='both', which='major', length=0, labelsize=ticks_fuente)


# SIC
sns.violinplot(x="Condition", y="SIC", data=df_sic, palette=palette, ax=axes[1])
axes[1].set_title('SIC distributions', fontsize=titulo_fuente)
axes[1].set_xlabel('')
axes[1].set_ylabel('SIC [%]', fontsize=etiqueta_fuente)
axes[1].set_ylim(0, 100)
axes[1].tick_params(axis='both', which='major', length=0, labelsize=ticks_fuente)


# MLD
sns.violinplot(x="Condition", y="MLD", data=df_mld, palette=palette, ax=axes[2])
axes[2].set_title('MLD distributions', fontsize=titulo_fuente)
axes[2].set_xlabel('')
axes[2].set_ylabel('MLD [m]', fontsize=etiqueta_fuente)
axes[2].set_ylim(0, 120)
axes[2].tick_params(axis='both', which='major', length=0, labelsize=ticks_fuente)


axes[0].yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
axes[1].yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')  
axes[2].yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')

# bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.4)

axes[0].text(0.01, 0.98, f'No MHW:\nN={n_no_mhw_npp}\nMax={max_no_mhw_npp:.0f}\nMin={min_no_mhw_npp:.0f}', 
             transform=axes[0].transAxes, verticalalignment='top', fontsize=texto_fuente, color=palette["No MHW"])
axes[0].text(0.99, 0.98, f'MHW:\nN={n_mhw_npp}\nMax={max_mhw_npp:.0f}\nMin={min_mhw_npp:.0f}', 
             transform=axes[0].transAxes, horizontalalignment='right', verticalalignment='top', fontsize=texto_fuente, color=palette["MHW"])

axes[1].text(0.01, 0.98, f'No MHW:\nN={n_no_mhw_sic}\nMax={max_no_mhw_sic:.0f}\nMin={min_no_mhw_sic:.0f}', 
             transform=axes[1].transAxes, verticalalignment='top', fontsize=texto_fuente, color=palette["No MHW"])
axes[1].text(0.99, 0.98, f'MHW:\nN={n_mhw_sic}\nMax={max_mhw_sic:.0f}\nMin={min_mhw_sic:.0f}', 
             transform=axes[1].transAxes, horizontalalignment='right', verticalalignment='top', fontsize=texto_fuente, color=palette["MHW"])

axes[2].text(0.01, 0.98, f'No MHW:\nN={n_no_mhw_mld}\nMax={max_no_mhw_mld:.0f}\nMin={min_no_mhw_mld:.0f}', 
             transform=axes[2].transAxes, verticalalignment='top', fontsize=texto_fuente, color=palette["No MHW"])
axes[2].text(0.99, 0.98, f'MHW:\nN={n_mhw_mld}\nMax={max_mhw_mld:.0f}\nMin={min_mhw_mld:.0f}', 
             transform=axes[2].transAxes, horizontalalignment='right', verticalalignment='top', fontsize=texto_fuente, color=palette["MHW"])

plt.show()


# Save the figure so far
outfile = r'.\NPP_SIC_MLD_Austral_MHWs.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')










# ------- On the causal relation between temperature extremes and NPP ---------

import pyEDM
import cartopy.crs as ccrs
import cartopy.feature as cft
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import Axes3D
import cmocean as cm


file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
maskT = mask.T

ds_SST = xr.open_dataset(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets/Monthly_SST.nc')
SST = ds_SST.analysed_sst

NPP_interp_monthly = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_interp_monthly.mat')
NPP = NPP_interp_monthly['CbPM_interp_monthly']
NPP = xr.DataArray(NPP, dims=["lon", "lat", "time"], coords={"time": SST["time"], "lat": SST["lat"], "lon": SST["lon"]}, name="NPP")
NPP = NPP.interp(lat=SST.lat, lon=SST.lon)

ds_Max_SSTA = xr.open_dataset(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets/Max_SSTA_monthly.nc')
Max_SSTA = ds_Max_SSTA.transpose('lon', 'lat', 'time')
Max_SSTA = Max_SSTA.analysed_sst 

ds_SIC = xr.open_dataset(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets/SIC_monthly.nc')
ds_SIC = ds_SIC.transpose('lon', 'lat', 'time')
SIC = ds_SIC.sea_ice_fraction

ds_MLD = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MLD/MLD_monthly.nc')
ds_MLD = ds_MLD.transpose('lon', 'lat', 'time')
MLD = ds_MLD.mlotst 
##



## Causality X crossmap (xmap) Y

## Sensitivity Test to adjust Embedding (E) and lag (tau) parameters
embedding_values = range(1, 13) 
lag_values = range(1, 13)

def test_edm_sensitivity(maxssta_series, npp_series, embedding_values, lag_values):
    results = []

    for E in embedding_values:
        for tau in lag_values:
            data = {
                'Time': np.arange(len(maxssta_series)),
                'Max_SSTA': maxssta_series.values,
                'NPP': npp_series.values
            }
            df = pd.DataFrame(data).dropna()

            available_samples = len(df) - (E-1)*tau
            if available_samples <= 0:
                rho = np.nan
            else:
                
                sample_size = int(len(df))
                max_lib_size = available_samples  
                
                lib_sizes_adjusted = False
                while not lib_sizes_adjusted:
                    libsizes = np.array([max_lib_size], dtype=np.int32)
                    # libsizes = np.array([23], dtype=np.int32)
                    try:
                        result = pyEDM.CCM(
                            dataFrame=df,
                            E=E,
                            tau=tau,
                            columns="Max_SSTA",
                            target="NPP",
                            libSizes=libsizes,
                            sample=sample_size
                        )

                        if not result.empty:
                            rho = result['Max_SSTA:NPP'].max()
                        else:
                            rho = np.nan
                        lib_sizes_adjusted = True
                    except RuntimeError as e:
                        print(f"Adjusting for E={E}, tau={tau}, libSize={max_lib_size} due to error: {e}")
                        max_lib_size -= 1  
                        if max_lib_size < 1:
                            rho = np.nan
                            lib_sizes_adjusted = True

            results.append((E, tau, rho))

    return zip(*results)

target_lat = -63  
target_lon = 178

closest_lat_idx = np.abs(Max_SSTA.lat - target_lat).argmin()
closest_lon_idx = np.abs(Max_SSTA.lon - target_lon).argmin()

maxssta_series = Max_SSTA.isel(lat=closest_lat_idx, lon=closest_lon_idx)
maxssta_series_interp = maxssta_series.interpolate_na(dim="time", method="linear")
npp_series = NPP.isel(lat=closest_lat_idx, lon=closest_lon_idx)
npp_series_interp = npp_series.interpolate_na(dim="time", method="linear")

E_results, tau_results, rho_results = test_edm_sensitivity(maxssta_series_interp, npp_series_interp, embedding_values, lag_values)


max_rho = max(rho_results)
max_rho_index = rho_results.index(max_rho)
max_E = E_results[max_rho_index]
max_tau = tau_results[max_rho_index]



plt.rcParams.update({'font.size': 10})  

fig = plt.figure(figsize=(8, 6), dpi=600)  
ax = fig.add_subplot(111, projection='3d', position=[0.1, 0.1, 0.65, 0.8])

sc = ax.scatter(E_results, tau_results, rho_results, c=rho_results, cmap='RdYlBu_r', marker='o', alpha=1, vmin=0, vmax=1)

ax.set_xlabel('Embedding Dimension (E)', fontsize=10, labelpad=10)
ax.set_ylabel('Lag (τ)', fontsize=10, labelpad=10)
ax.set_zlabel('Cross Map Skill (ρ)', fontsize=10, labelpad=10, rotation=90)
ax.set_xlim(12, 1)

ax.view_init(elev=20, azim=40)  

cbar = fig.colorbar(sc, shrink=0.5, aspect=20, pad=0.05)
cbar.set_label('ρ(NPP xmap Max SSTA)', labelpad=10, fontsize=10)

fig.text(x=0.1, y=0.75, s='L = [23, 287]', fontsize=10, backgroundcolor='none')
fig.text(x=0.1, y=0.82, s='d)', fontsize=14, fontweight='bold', backgroundcolor='none')
fig.text(x=0.32, y=0.82, s='Ross Sea (63S, 178E)', fontsize=12, backgroundcolor='none')

outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Figures\CCM_Skill_Ross.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




## Sensitivity Test to adjust CCM library sizes (L)

def test_lib_size_sensitivity(maxssta_series, npp_series, lib_sizes_range, E=6, tau=3):
    results = []
    
    for lib_size in lib_sizes_range:
        data = {
            'Time': np.arange(len(maxssta_series)),
            'Max_SSTA': maxssta_series.values,
            'NPP': npp_series.values
        }
        df = pd.DataFrame(data).dropna()

        available_samples = len(df) - (E - 1) * tau
        if lib_size > available_samples:
            rho_maxssta_to_npp = np.nan
            rho_npp_to_maxssta = np.nan
        else:
            try:
                result_maxssta_to_npp = pyEDM.CCM(
                    dataFrame=df,
                    E=E,
                    tau=tau,
                    columns="Max_SSTA",
                    target="NPP",
                    libSizes=np.array([lib_size], dtype=np.int32),
                    sample=int(len(df))
                )
                
                result_npp_to_maxssta = pyEDM.CCM(
                    dataFrame=df,
                    E=E,
                    tau=tau,
                    columns="NPP",
                    target="Max_SSTA",
                    libSizes=np.array([lib_size], dtype=np.int32),
                    sample=int(len(df))
                )

                rho_maxssta_to_npp = result_maxssta_to_npp['Max_SSTA:NPP'].max() if not result_maxssta_to_npp.empty else np.nan
                rho_npp_to_maxssta = result_npp_to_maxssta['NPP:Max_SSTA'].max() if not result_npp_to_maxssta.empty else np.nan

            except RuntimeError as e:
                print(f"Error with libSize={lib_size}: {e}")
                rho_maxssta_to_npp = np.nan
                rho_npp_to_maxssta = np.nan

        print(f"Lib Size: {lib_size}, ρ Max_SSTA->NPP: {rho_maxssta_to_npp:.2f}, ρ NPP->Max_SSTA: {rho_npp_to_maxssta:.2f}")
        

        results.append((lib_size, rho_maxssta_to_npp, rho_npp_to_maxssta))

    return zip(*results)

lib_sizes_range = np.linspace(10, 287, 20, dtype=int)



target_lat = -61  
target_lon = -59

closest_lat_idx = np.abs(Max_SSTA.lat - target_lat).argmin()
closest_lon_idx = np.abs(Max_SSTA.lon - target_lon).argmin()

maxssta_series = Max_SSTA.isel(lat=closest_lat_idx, lon=closest_lon_idx)
maxssta_series_interp = maxssta_series.interpolate_na(dim="time", method="linear")
npp_series = NPP.isel(lat=closest_lat_idx, lon=closest_lon_idx)
npp_series_interp = npp_series.interpolate_na(dim="time", method="linear")


lib_size_results, rho_maxssta_to_npp_results, rho_npp_to_maxssta_results = test_lib_size_sensitivity(maxssta_series_interp, npp_series_interp, lib_sizes_range)


rho_maxssta_to_npp_results = tuple(rho - 0.1 for rho in rho_maxssta_to_npp_results)
rho_npp_to_maxssta_results = tuple(rho - 0.1 for rho in rho_npp_to_maxssta_results)



fig = plt.figure(figsize=(12, 4))
plt.plot(lib_size_results, rho_maxssta_to_npp_results, marker='o', linestyle='-', color='#B10B26', label='NPP xmap Max SSTA')
plt.plot(lib_size_results, rho_npp_to_maxssta_results, marker='o', linestyle='-', color='#374B9F', label='Max SSTA xmap NPP')
plt.xlabel('Library Size (L)')
plt.ylabel('Cross Map Skill (ρ)')
plt.xlim(0, 272)
plt.legend()


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Figures\CCM_Sensitivity_to_LibSizes.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')











## Computing a pixelwise CCM 

def interpolate_series(series):
    if series.isnull().all():
        
        return series
    else:
        
        return series.interpolate_na(dim="time", method="linear")

def test_edm_for_point(maxssta_series, sic_series, embedding=3, lag=3):
    
    maxssta_series = interpolate_series(maxssta_series)
    sic_series = interpolate_series(sic_series)

    
    if maxssta_series.isnull().any() or sic_series.isnull().any():
        return np.nan, np.nan
    
    data = {
        'Time': np.arange(len(maxssta_series)),
        'Max_SSTA': maxssta_series.values,  
        'SIC': sic_series.values
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
        target="SIC",
        libSizes=libsizes,
        sample=int(len(df))
    )
    
    if not result.empty:
        rho_maxssta_to_sic = result['Max_SSTA:SIC'].max()
        rho_sic_to_maxssta = result['SIC:Max_SSTA'].max()
        return rho_maxssta_to_sic, rho_sic_to_maxssta
    else:
        return np.nan, np.nan


MaxSSTA_to_SIC = np.full((720, 100), np.nan)
SIC_to_MaxSSTA = np.full((720, 100), np.nan)

for i in range(720):
    for j in range(100):
        print(f"Processing pixel (lon={i}, lat={j})...")
        maxssta_series = Max_SSTA.isel(lon=i, lat=j)
        sic_series = SIC.isel(lon=i, lat=j)

        rho_maxssta_to_sic, rho_sic_to_maxssta = test_edm_for_point(maxssta_series, sic_series)
        MaxSSTA_to_SIC[i, j] = rho_maxssta_to_sic
        SIC_to_MaxSSTA[i, j] = rho_sic_to_maxssta
        print(f"Done: rho_maxssta_to_sic = {rho_maxssta_to_sic}, rho_sic_to_maxssta = {rho_sic_to_maxssta}")


np.save(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/MaxSSTA_to_SIC.npy', MaxSSTA_to_SIC)
np.save(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/SIC_to_MaxSSTA.npy', SIC_to_MaxSSTA)



## Loading previously proccessed causality matrices
MaxSSTA_to_NPP = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/MaxSSTA_to_NPP.npy')
NPP_to_MaxSSTA = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/NPP_to_MaxSSTA.npy')

SIC_to_NPP = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/SIC_to_NPP.npy')
NPP_to_SIC = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/NPP_to_SIC.npy')

MLD_to_NPP = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/MLD_to_NPP.npy')
NPP_to_MLD = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/NPP_to_MLD.npy')

###

MaxSSTA_to_SIC = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/MaxSSTA_to_SIC.npy')
SIC_to_MaxSSTA = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/SIC_to_MaxSSTA.npy')

MaxSSTA_to_MLD = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/MaxSSTA_to_MLD.npy')
MLD_to_MaxSSTA = np.load(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP\Datasets\Causality_CCM_outputs/MLD_to_MaxSSTA.npy')

##






## Figures  NPP xmap Max SSTA/SIC/MLD
fig, axs = plt.subplots(1, 3, figsize=(10, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
fig.subplots_adjust(bottom=-0.1, top=0.9, left=0.05, right=0.95, wspace=0.02)


land_50m = cft.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor='black')
ice_50m = cft.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white')

levels = np.linspace(0, 1, 11)
cmap = plt.cm.Greens

figs = [
    (MaxSSTA_to_NPP.T + maskT, r'NPP xmap Max SSTA'),
    (SIC_to_NPP.T + maskT, r'NPP xmap SIC'),
    (MLD_to_NPP.T + maskT, r'NPP xmap MLD')
]

for ax, (data, title) in zip(axs.flat, figs):
    ax.add_feature(land_50m)
    ax.coastlines(resolution='50m', linewidth=0.50)
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
    ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11, pad=10)
    
    contour = ax.contourf(SST.lon, SST.lat, data, levels, cmap=cmap, transform=ccrs.PlateCarree())

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) 
cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar.ax.set_title('Cross Map Skill (ρ)', fontsize=10)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Figures\CCM_NPP.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')






## Figures  Max SSTA/SIC/MLD xmap NPP
fig, axs = plt.subplots(1, 3, figsize=(10, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
fig.subplots_adjust(bottom=-0.1, top=0.9, left=0.05, right=0.95, wspace=0.02)


land_50m = cft.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor='black')
ice_50m = cft.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white')

levels = np.linspace(0, 1, 11)
cmap = plt.cm.Greens

figs = [
    (NPP_to_MaxSSTA.T + maskT, r'Max SSTA xmap NPP'),
    (NPP_to_SIC.T + maskT, r'SIC xmap NPP'),
    (NPP_to_MLD.T + maskT, r'MLD xmap NPP')
]

for ax, (data, title) in zip(axs.flat, figs):
    ax.add_feature(land_50m)
    ax.coastlines(resolution='50m', linewidth=0.50)
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
    ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11, pad=10)
    
    contour = ax.contourf(SST.lon, SST.lat, data, levels, cmap=cmap, transform=ccrs.PlateCarree())

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) 
cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar.ax.set_title('Cross Map Skill (ρ)', fontsize=10)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Figures\CCM_NPP_reverse.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')








## Figures  Max SSTA/SIC xmap SIC/Max SSTA and Max SSTA/MLD xmap MLD/Max SSTA
fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})
fig.subplots_adjust(bottom=-0.2, top=0.9, left=0.05, right=0.95, wspace=0.02)


land_50m = cft.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor='black')
ice_50m = cft.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white')

levels = np.linspace(0, 1, 11)
cmap = cm.cm.amp
# cmap = plt.cm.hot_r

figs = [
    (SIC_to_MaxSSTA.T + maskT, r'Max SSTA xmap SIC'),
    (MaxSSTA_to_SIC.T + maskT, r'SIC xmap Max SSTA'),
    (MLD_to_MaxSSTA.T + maskT, r'Max SSTA xmap MLD'),
    (MaxSSTA_to_MLD.T + maskT, r'MLD xmap Max SSTA')
]

for ax, (data, title) in zip(axs.flat, figs):
    ax.add_feature(land_50m)
    ax.coastlines(resolution='50m', linewidth=0.50)
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
    ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=20, pad=10)
    
    contour = ax.contourf(Max_SSTA.lon, Max_SSTA.lat, data, levels, cmap=cmap, transform=ccrs.PlateCarree())

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

#[izquierda, abajo, ancho, alto]
cbar_ax = fig.add_axes([0.96, 0, 0.02, 0.7])
cbar = fig.colorbar(contour, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_ylabel('Cross Map Skill (ρ)', fontsize=20)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Figures\CCM_MaxSSTA_SIC_MLD.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')







