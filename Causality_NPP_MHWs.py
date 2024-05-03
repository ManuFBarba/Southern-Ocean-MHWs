# -*- coding: utf-8 -*-
"""
Figure 4b, c and d. On the causal relationship between MHWs and NPP.
"""


import xarray as xr
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests




ds_SST = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\SST_full/SST_ANT_1982-2021_40.nc')

SST = ds_SST.analysed_sst[:,::10,::10]

monthly_sst = SST.resample(time='M').mean(skipna=True)

monthly_sst = monthly_sst.transpose('lon', 'lat', 'time')

monthly_sst = monthly_sst.sel(time=slice('1998-01-01', '2021-12-31'))


monthly_sst.to_netcdf(r'C:\Users\Manuel\Desktop\Causality_SST_NPP/Monthly_SST.nc')




# Convertir NPP a un DataArray de xarray para facilitar el manejo
npp_data = xr.DataArray(NPP_interp_monthly, dims=["lon", "lat", "time"], coords=monthly_sst.coords)



def test_granger_for_point(sst_series, npp_series, max_lags=4):
    # Alinear series temporales por tiempo y convertir a DataFrame
    sst_series, npp_series = xr.align(sst_series, npp_series)
    df = pd.DataFrame({
        'SST': sst_series.to_pandas(),
        'NPP': npp_series.to_pandas()
    }).dropna()

    # Verificar que el DataFrame no esté vacío y tenga suficiente longitud
    if df.empty or len(df) <= max_lags:
        return np.nan

    # Verificar que ambas series tengan variabilidad (varianza > 0)
    if df.var()[0] == 0 or df.var()[1] == 0:
        return np.nan

    # Realizar la prueba de causalidad de Granger
    try:
        test_result = grangercausalitytests(df, maxlag=max_lags, verbose=False)
        # Extraer y devolver el valor p del primer retraso, como ejemplo
        p_value = test_result[1][0]['ssr_chi2test'][1]
    except ValueError:
        # Capturar cualquier otro error inesperado durante la prueba
        p_value = np.nan

    return p_value


p_values = np.full((monthly_sst.sizes['lon'], monthly_sst.sizes['lat']), np.nan)


for i in range(monthly_sst.sizes['lon']):
    for j in range(monthly_sst.sizes['lat']):
        sst_series = monthly_sst.isel(lon=i, lat=j)
        npp_series = npp_data.isel(lon=i, lat=j)
        
        
        if not sst_series.isnull().all() and not npp_series.isnull().all():
            p_value = test_granger_for_point(sst_series, npp_series)
            p_values[i, j] = p_value


p_values_da = xr.DataArray(p_values, dims=["lon", "lat"], coords={"lon": monthly_sst.lon, "lat": monthly_sst.lat})








# Configuración de la proyección para el mapa
projection = ccrs.SouthPolarStereo()

# Creación de la figura y el eje con la proyección deseada
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

# Definición de los límites del mapa para enfocarse en el área de interés
ax.set_extent([-180, 180, -90, -40], crs=ccrs.PlateCarree())

# Añadir características del mapa
ax.add_feature(cft.OCEAN)

# Representación de los valores p de Granger como un mapa de calor
# lon, lat = np.meshgrid(p_values_Causality.lon, p_values_da.lat)  # Crear una malla de coordenadas
# data = p_values_da.values  # Obtener los valores de los datos
pcm = ax.pcolormesh(lon, lat, p_values_Causality+mask, vmin=0, vmax=0.5, transform=ccrs.PlateCarree(), cmap='Greens')
plt.colorbar(pcm, ax=ax, shrink=0.5)

ax.add_feature(cft.LAND, color='black')
ax.add_feature(cft.COASTLINE)
ax.add_feature(cft.BORDERS, linestyle=':')
# Mostrar el mapa
plt.show()


















### CORREGIR SCRIPT EDM para que p-value no sean todo NaN (script corre bien, pero p_value = nan todo ###



import numpy as np
import xarray as xr
import pyEDM
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cft
from scipy.io import loadmat


NPP_interp_monthly = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\CbPM\CbPM_interp_monthly.mat')
NPP_interp_monthly = NPP_interp_monthly['CbPM_interp_monthly']

ds_monthly_sst = xr.open_dataset(r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Causality_SST_NPP/Monthly_SST.nc')
monthly_sst = ds_monthly_sst.analysed_sst

monthly_npp = xr.DataArray(NPP_interp_monthly, dims=["lon", "lat", "time"], coords=monthly_sst.coords)





def test_edm_for_point(sst_series, npp_series, embedding=3, lag=3):
    # Convertir series a DataFrame de pyEDM
    data = {
        'Time': np.arange(len(sst_series)),
        'SST': sst_series,
        'NPP': npp_series
    }
    df = pd.DataFrame(data).dropna()

    # Verificar suficientes datos
    if len(df) <= (embedding * lag):
        return np.nan

    # Simplex projection para evaluar la predictibilidad
    result = pyEDM.Simplex(
        dataFrame=df,
        lib="1 {}".format(len(df)),
        pred="1 {}".format(len(df)),
        columns="SST",
        target="NPP",
        E=embedding,
        tau=lag
    )

    # Imprimir el resultado para ver su estructura
    print(result)

    if 'rho' in result.columns:  
        return result['rho'].mean()  
    else:
        return np.nan


p_values = np.full((monthly_sst.sizes['lon'], monthly_sst.sizes['lat']), np.nan)

for i in range(monthly_sst.sizes['lon']):
    for j in range(monthly_sst.sizes['lat']):
        sst_series = monthly_sst.isel(lon=i, lat=j).values
        npp_series = monthly_npp.isel(lon=i, lat=j).values
        
        if np.all(np.isnan(sst_series)) or np.all(np.isnan(npp_series)):
            continue
        
        p_value = test_edm_for_point(sst_series, npp_series)
        p_values[i, j] = p_value



# p_values_da = xr.DataArray(p_values, dims=["lon", "lat"], coords={"lon": monthly_sst.lon, "lat": monthly_sst.lat})








projection = ccrs.SouthPolarStereo()

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

ax.set_extent([-180, 180, -90, -40], crs=ccrs.PlateCarree())

pcm = ax.pcolormesh(lon, lat, p_values+mask, vmin=0, vmax=0.5, transform=ccrs.PlateCarree(), cmap='Greens')
plt.colorbar(pcm, ax=ax, shrink=0.5)

ax.add_feature(cft.LAND, color='black')
ax.add_feature(cft.COASTLINE)


theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


plt.show()

















