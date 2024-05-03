# -*- coding: utf-8 -*-
"""
MHWs onset-averaged and decay-averaged anomalies for the four most important
tendency terms in daily-mean 1982-2021 simulation of GFDL ESM2M.
""" 


#Load required libraries
import numpy as np
import xarray as xr
from scipy.io import loadmat

import matplotlib.pyplot as plt

import cmocean as cm
import cartopy.crs as ccrs
import matplotlib.path as mpath
import cartopy.feature as cft

import matplotlib.patheffects as path_effects
import string


# Load datasets -----------------------------------------------------------
## 1982-2021
ds_onset = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Temp_tendency_term_Data/Onset_Phase_1982_2021.nc').squeeze()
ds_decay = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Temp_tendency_term_Data/Decay_Phase_1982_2021.nc').squeeze()
## 500-yr preindustrial Climatological mean
ds_onset_clim = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Temp_tendency_term_Data/Onset_Phase_Climatological_Preindustrial.nc').squeeze()
ds_decay_clim = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Temp_tendency_term_Data/Decay_Phase_Climatological_Preindustrial.nc').squeeze()

#lat and lon from MHWs
lat = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\lat.mat')
lat = lat['latitud']

lon = loadmat(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\lon.mat')
lon = lon['longitud']
#Sea-ice mask
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
mask_without_nan = np.nan_to_num(mask, nan=1)



###############################################################################
##Representing individual plots
## Onset Phase ##
#Fig a)
# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(-50, 50, 21)

# Plot contourf
cmap = plt.cm.RdYlBu_r
p1 = ax.contourf(ds_onset.geolon_t, ds_onset.geolat_t, ds_onset.temp_adv, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree(), interpolation='nearest')
p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Advection' + ' ' + r'$\Delta Q_\mathrm{adv}$', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)





#Fig c)
# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(-50, 50, 21)

# Plot contourf
cmap = plt.cm.RdYlBu_r
p1 = ax.contourf(ds_onset.geolon_t, ds_onset.geolat_t, ds_onset.temp_sfch, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Surface heat flux' + ' ' + r'$\Delta Q_\mathrm{a-s}$', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)





#Fig e)
# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(-50, 50, 21)

# Plot contourf
cmap = plt.cm.RdYlBu_r
p1 = ax.contourf(ds_onset.geolon_t, ds_onset.geolat_t, ds_onset.temp_kpp, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Convective vertical mixing' + ' ' + r'$\Delta Q_\mathrm{vmix}$', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)



#Fig g)
# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(-50, 50, 21)

# Plot contourf
cmap = plt.cm.RdYlBu_r
p1 = ax.contourf(ds_onset.geolon_t, ds_onset.geolat_t, ds_onset.temp_vdiff, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Vertical diffusion' + ' ' + r'$\Delta Q_\mathrm{vdiff}$', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)




## Decay Phase ##

#Fig b)
# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(-50, 50, 21)

# Plot contourf
cmap = plt.cm.RdYlBu_r
p1 = ax.contourf(ds_decay.geolon_t, ds_decay.geolat_t, ds_decay.temp_adv, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Advection' + ' ' + r'$\Delta Q_\mathrm{adv}$', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)





#Fig d)
# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(-50, 50, 21)

# Plot contourf
cmap = plt.cm.RdYlBu_r
p1 = ax.contourf(ds_decay.geolon_t, ds_decay.geolat_t, ds_decay.temp_sfch, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Surface heat flux' + ' ' + r'$\Delta Q_\mathrm{a-s}$', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)





#Fig f)
# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(-50, 50, 21)

# Plot contourf
cmap = plt.cm.RdYlBu_r
p1 = ax.contourf(ds_decay.geolon_t, ds_decay.geolat_t, ds_decay.temp_kpp, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Convective vertical mixing' + ' ' + r'$\Delta Q_\mathrm{vmix}$', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)



#Fig h)
# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(-50, 50, 21)

# Plot contourf
cmap = plt.cm.RdYlBu_r
p1 = ax.contourf(ds_decay.geolon_t, ds_decay.geolat_t, ds_decay.temp_vdiff, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35)
cbar.ax.minorticks_off()


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('Vertical diffusion' + ' ' + r'$\Delta Q_\mathrm{vdiff}$', fontsize=30)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
###############################################################################






###############################################################################
## MLD ##
ds_MLD = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MLD/MLD_1993_2021.nc')

climatology_max = ds_MLD.mlotst.groupby('time.month').max(dim='time', skipna=True)
Zint = climatology_max.mean(dim='month', skipna=True)


# Set the projection 
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

# Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                  edgecolor='none', facecolor='white')

# ax.add_feature(ice_50m)

# Define levels for contour plot
levels = np.linspace(0, 110, 12)

# Plot contourf
cmap = cm.cm.deep
p1 = ax.contourf(ds_MLD.longitude, ds_MLD.latitude, Zint, levels, cmap=cmap, extend='max', transform=ccrs.PlateCarree())
cbar = plt.colorbar(p1, shrink=0.8)
cbar.ax.tick_params(axis='y', size=10, direction='out', labelsize=35)
cbar.ax.minorticks_off()
cbar.set_label(r'[$m$]', fontsize=35)

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth=0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title(r'$Z_\mathrm{int}$', fontsize=38)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Figuras\Figs_Explicacion_Reviews\Zint_SO.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')
###############################################################################







###############################################################################
## Representing all subplots ##


## 1982- 2021
def create_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

fig, axs = plt.subplots(4, 2, figsize=(15, 25), subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0, globe=None)})

axs[0, 0].text(0.5, 1.15, r'Onset Phase', fontsize=30, ha='center', va='bottom', fontweight='bold', transform=axs[0, 0].transAxes)
axs[0, 1].text(0.5, 1.15, r'Decay Phase', fontsize=30, ha='center', va='bottom', fontweight='bold', transform=axs[0, 1].transAxes)

figs = [
    (ds_onset.temp_adv, 'Advection', r'$\Delta Q_\mathrm{adv}$'),
    (ds_decay.temp_adv, 'Advection', r'$\Delta Q_\mathrm{adv}$'),
    (ds_onset.temp_sfch, 'Surface heat flux', r'$\Delta Q_\mathrm{a-s}$'),
    (ds_decay.temp_sfch, 'Surface heat flux', r'$\Delta Q_\mathrm{a-s}$'),
    (ds_onset.temp_kpp, 'Convective vertical mixing', r'$\Delta Q_\mathrm{vmix}$'),
    (ds_decay.temp_kpp, 'Convective vertical mixing', r'$\Delta Q_\mathrm{vmix}$'),
    (ds_onset.temp_vdiff, 'Vertical diffusion', r'$\Delta Q_\mathrm{vdiff}$'),
    (ds_decay.temp_vdiff, 'Vertical diffusion', r'$\Delta Q_\mathrm{vdiff}$')
]


for ax, (data, title, label) in zip(axs.flat, figs):
    land_50m = cft.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor='black', linewidth=0.5)
    ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m', edgecolor='none', facecolor='white')
    levels = np.linspace(-50, 50, 21)
    cmap = plt.cm.RdYlBu_r
    p1 = ax.contourf(data.geolon_t, data.geolat_t, data, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
    p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
    ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color='black')
    ax.coastlines(resolution='50m', linewidth=0.50)
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
    ax.set_title(title + ' ' + label, fontsize=22)
    create_circle(ax)

for i, ax in enumerate(axs.flat):
    label = string.ascii_lowercase[i]
    ax.text(-0.08, 1.05, f'{label}', transform=ax.transAxes, fontsize=22,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='none', alpha=1, edgecolor='none'),
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])


cbar_ax = fig.add_axes([0.1, axs[-1, 0].get_position().y0 - 0.18, 0.8, 0.02])
cbar = plt.colorbar(p1, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(axis='x', size=12, direction='in', which='both', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_label('Heat flux anomaly ' + r'[$\mathrm{W} / \mathrm{m}^2$]', fontsize=24)

cbar_ax_twiny = cbar_ax.twiny()
conversion_factor = 0.00209
cbar_ticks_twiny = cbar.get_ticks()
converted_ticks = cbar_ticks_twiny * conversion_factor
cbar_twiny = plt.colorbar(p1, cax=cbar_ax_twiny, orientation='horizontal', ticks=cbar_ticks_twiny)
cbar_twiny.ax.set_xticklabels(['{:.2f}'.format(x) if x!=0 else 0 for x in converted_ticks])
cbar_twiny.ax.minorticks_off()
cbar_twiny.set_label(r'$\Delta ' + '\u03B8$ ' + r'[$\mathrm{ºC} / \mathrm{d}$]', fontsize=24)

cbar_ax_twiny.xaxis.set_ticks_position("top")
cbar_ax_twiny.xaxis.set_label_position("top")
cbar_ax.xaxis.set_ticks_position("bottom")
cbar_ax.xaxis.set_label_position("bottom")
cbar_ax_twiny.tick_params(axis='x', size=12, direction='in', which='both', labelsize=22, bottom=True, top=True)


plt.subplots_adjust(bottom=0.2)

plt.tight_layout()
plt.show()



outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Temperature_tendency_terms\Figures\Temp_Tendency_Budgets_ESM2M.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')









## pi Control Onset and Decay
def create_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

fig, axs = plt.subplots(4, 2, figsize=(15, 25), subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0, globe=None)})

axs[0, 0].text(0.5, 1.15, r'Onset Phase', fontsize=30, ha='center', va='bottom', fontweight='bold', transform=axs[0, 0].transAxes)
axs[0, 1].text(0.5, 1.15, r'Decay Phase', fontsize=30, ha='center', va='bottom', fontweight='bold', transform=axs[0, 1].transAxes)
axs[0, 0].text(0.5, 1.075, r'Control', fontsize=25, ha='center', va='bottom', transform=axs[0, 0].transAxes)
axs[0, 1].text(0.5, 1.075, r'Control', fontsize=25, ha='center', va='bottom', transform=axs[0, 1].transAxes)



figs = [
    (ds_onset_clim.temp_adv, 'Advection', r'$\Delta Q_\mathrm{adv}$'),
    (ds_decay_clim.temp_adv, 'Advection', r'$\Delta Q_\mathrm{adv}$'),
    (ds_onset_clim.temp_sfch, 'Surface heat flux', r'$\Delta Q_\mathrm{a-s}$'),
    (ds_decay_clim.temp_sfch, 'Surface heat flux', r'$\Delta Q_\mathrm{a-s}$'),
    (ds_onset_clim.temp_kpp, 'Convective vertical mixing', r'$\Delta Q_\mathrm{vmix}$'),
    (ds_decay_clim.temp_kpp, 'Convective vertical mixing', r'$\Delta Q_\mathrm{vmix}$'),
    (ds_onset_clim.temp_vdiff, 'Vertical diffusion', r'$\Delta Q_\mathrm{vdiff}$'),
    (ds_decay_clim.temp_vdiff, 'Vertical diffusion', r'$\Delta Q_\mathrm{vdiff}$')
]


for ax, (data, title, label) in zip(axs.flat, figs):
    land_50m = cft.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor='black', linewidth=0.5)
    ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m', edgecolor='none', facecolor='white')
    levels = np.linspace(-50, 50, 21)
    cmap = plt.cm.RdYlBu_r
    p1 = ax.contourf(data.geolon_t, data.geolat_t, data, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
    p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
    ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color='black')
    ax.coastlines(resolution='50m', linewidth=0.50)
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
    ax.set_title(title + ' ' + label, fontsize=22)
    create_circle(ax)

for i, ax in enumerate(axs.flat):
    label = string.ascii_lowercase[i]
    ax.text(-0.08, 1.05, f'{label}', transform=ax.transAxes, fontsize=22,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='none', alpha=1, edgecolor='none'),
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])


cbar_ax = fig.add_axes([0.1, axs[-1, 0].get_position().y0 - 0.18, 0.8, 0.02])
cbar = plt.colorbar(p1, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(axis='x', size=12, direction='in', which='both', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_label('Heat flux anomaly ' + r'[$\mathrm{W} / \mathrm{m}^2$]', fontsize=24)

cbar_ax_twiny = cbar_ax.twiny()
conversion_factor = 0.00209
cbar_ticks_twiny = cbar.get_ticks()
converted_ticks = cbar_ticks_twiny * conversion_factor
cbar_twiny = plt.colorbar(p1, cax=cbar_ax_twiny, orientation='horizontal', ticks=cbar_ticks_twiny)
cbar_twiny.ax.set_xticklabels(['{:.2f}'.format(x) if x!=0 else 0 for x in converted_ticks])
cbar_twiny.ax.minorticks_off()
cbar_twiny.set_label(r'$\Delta ' + '\u03B8$ ' + r'[$\mathrm{ºC} / \mathrm{d}$]', fontsize=24)

cbar_ax_twiny.xaxis.set_ticks_position("top")
cbar_ax_twiny.xaxis.set_label_position("top")
cbar_ax.xaxis.set_ticks_position("bottom")
cbar_ax.xaxis.set_label_position("bottom")
cbar_ax_twiny.tick_params(axis='x', size=12, direction='in', which='both', labelsize=22, bottom=True, top=True)


plt.subplots_adjust(bottom=0.2)

plt.tight_layout()
plt.show()



outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Temperature_tendency_terms\Figures\Temp_Tendency_Budgets_ESM2M_Cntrl.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')








## Period differences (Present - Control)
def create_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

fig, axs = plt.subplots(4, 2, figsize=(15, 25), subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0, globe=None)})

axs[0, 0].text(0.5, 1.15, r'Onset Phase', fontsize=30, ha='center', va='bottom', fontweight='bold', transform=axs[0, 0].transAxes)
axs[0, 1].text(0.5, 1.15, r'Decay Phase', fontsize=30, ha='center', va='bottom', fontweight='bold', transform=axs[0, 1].transAxes)
axs[0, 0].text(0.5, 1.075, r'Present - Control', fontsize=25, ha='center', va='bottom', transform=axs[0, 0].transAxes)
axs[0, 1].text(0.5, 1.075, r'Present - Control', fontsize=25, ha='center', va='bottom', transform=axs[0, 1].transAxes)



figs = [
    (ds_onset.temp_adv - ds_onset_clim.temp_adv, 'Advection', r'$\Delta Q_\mathrm{adv}$'),
    (ds_decay.temp_adv - ds_decay_clim.temp_adv, 'Advection', r'$\Delta Q_\mathrm{adv}$'),
    (ds_onset.temp_sfch - ds_onset_clim.temp_sfch, 'Surface heat flux', r'$\Delta Q_\mathrm{a-s}$'),
    (ds_decay.temp_sfch - ds_decay_clim.temp_sfch, 'Surface heat flux', r'$\Delta Q_\mathrm{a-s}$'),
    (ds_onset.temp_kpp - ds_onset_clim.temp_kpp, 'Convective vertical mixing', r'$\Delta Q_\mathrm{vmix}$'),
    (ds_decay.temp_kpp - ds_decay_clim.temp_kpp, 'Convective vertical mixing', r'$\Delta Q_\mathrm{vmix}$'),
    (ds_onset.temp_vdiff - ds_onset_clim.temp_vdiff, 'Vertical diffusion', r'$\Delta Q_\mathrm{vdiff}$'),
    (ds_decay.temp_vdiff - ds_decay_clim.temp_vdiff, 'Vertical diffusion', r'$\Delta Q_\mathrm{vdiff}$')
]


for ax, (data, title, label) in zip(axs.flat, figs):
    land_50m = cft.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor='black', linewidth=0.5)
    ice_50m = cft.NaturalEarthFeature('physical', 'ocean', scale='50m', edgecolor='none', facecolor='white')
    levels = np.linspace(-50, 50, 21)
    cmap = plt.cm.RdYlBu_r
    p1 = ax.contourf(data.geolon_t, data.geolat_t, data, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
    p2 = ax.contour(lon, lat, mask_without_nan, levels=[0], colors='black', linestyles='dashed', transform=ccrs.PlateCarree())
    ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color='black')
    ax.coastlines(resolution='50m', linewidth=0.50)
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
    ax.set_title(title + ' ' + label, fontsize=22)
    create_circle(ax)

for i, ax in enumerate(axs.flat):
    label = string.ascii_lowercase[i]
    ax.text(-0.08, 1.05, f'{label}', transform=ax.transAxes, fontsize=22,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='none', alpha=1, edgecolor='none'),
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])


cbar_ax = fig.add_axes([0.1, axs[-1, 0].get_position().y0 - 0.18, 0.8, 0.02])
cbar = plt.colorbar(p1, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(axis='x', size=12, direction='in', which='both', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_label('Heat flux anomaly ' + r'[$\mathrm{W} / \mathrm{m}^2$]', fontsize=24)

cbar_ax_twiny = cbar_ax.twiny()
conversion_factor = 0.00209
cbar_ticks_twiny = cbar.get_ticks()
converted_ticks = cbar_ticks_twiny * conversion_factor
cbar_twiny = plt.colorbar(p1, cax=cbar_ax_twiny, orientation='horizontal', ticks=cbar_ticks_twiny)
cbar_twiny.ax.set_xticklabels(['{:.2f}'.format(x) if x!=0 else 0 for x in converted_ticks])
cbar_twiny.ax.minorticks_off()
cbar_twiny.set_label(r'$\Delta ' + '\u03B8$ ' + r'[$\mathrm{ºC} / \mathrm{d}$]', fontsize=24)

cbar_ax_twiny.xaxis.set_ticks_position("top")
cbar_ax_twiny.xaxis.set_label_position("top")
cbar_ax.xaxis.set_ticks_position("bottom")
cbar_ax.xaxis.set_label_position("bottom")
cbar_ax_twiny.tick_params(axis='x', size=12, direction='in', which='both', labelsize=22, bottom=True, top=True)


plt.subplots_adjust(bottom=0.2)

plt.tight_layout()
plt.show()



outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Temperature_tendency_terms\Figures\Temp_Tendency_Budgets_ESM2M_PeriodDiff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')
###############################################################################

























###############################################################################
## Annual contributions in each Sector
## Firstly we clip global datasets to SO dimensions [-40, -90]
ds_onset_SO = ds_onset.sel(yt_ocean=slice(0,43))
ds_decay_SO = ds_decay.sel(yt_ocean=slice(0,43))

#Second, we clip dataset by sectors
ds_onset_Pacific = ds_onset_SO.sel(xt_ocean=slice(67,243))
ds_decay_Pacific = ds_decay_SO.sel(xt_ocean=slice(67,243))

ds_onset_Atlantic = ds_onset_SO.sel(xt_ocean=slice(243,306))
ds_decay_Atlantic = ds_decay_SO.sel(xt_ocean=slice(243,306))

ds_onset_Indian_1 = ds_onset_SO.sel(xt_ocean=slice(0, 67)) 
ds_onset_Indian_2 = ds_onset_SO.sel(xt_ocean=slice(306, 360))
ds_decay_Indian_1 = ds_decay_SO.sel(xt_ocean=slice(0, 67))
ds_decay_Indian_2 = ds_decay_SO.sel(xt_ocean=slice(306, 360))
ds_onset_Indian = xr.concat([ds_onset_SO.sel(xt_ocean=slice(0, 67)), ds_onset_SO.sel(xt_ocean=slice(306, 359))], dim='xt_ocean')
ds_decay_Indian = xr.concat([ds_decay_SO.sel(xt_ocean=slice(0, 67)), ds_decay_SO.sel(xt_ocean=slice(306, 359))], dim='xt_ocean')


## Now we compute ΔQ total
ΔQ_Total_onset = ds_onset_SO.temp_adv + ds_onset_SO.temp_sfch + ds_onset_SO.temp_kpp + ds_onset_SO.temp_vdiff + ds_onset_SO.temp_pme + ds_onset_SO.temp_ndiff + ds_onset_SO.temp_oth
ΔQ_Total_decay = ds_decay_SO.temp_adv + ds_decay_SO.temp_sfch + ds_decay_SO.temp_kpp + ds_decay_SO.temp_vdiff + ds_decay_SO.temp_pme + ds_decay_SO.temp_ndiff + ds_decay_SO.temp_oth

ΔQ_Total_onset_Pacific = ds_onset_Pacific.temp_adv + ds_onset_Pacific.temp_sfch + ds_onset_Pacific.temp_kpp + ds_onset_Pacific.temp_vdiff + ds_onset_Pacific.temp_pme + ds_onset_Pacific.temp_ndiff + ds_onset_Pacific.temp_oth
ΔQ_Total_decay_Pacific = ds_decay_Pacific.temp_adv + ds_decay_Pacific.temp_sfch + ds_decay_Pacific.temp_kpp + ds_decay_Pacific.temp_vdiff + ds_decay_Pacific.temp_pme + ds_decay_Pacific.temp_ndiff + ds_decay_Pacific.temp_oth

ΔQ_Total_onset_Atlantic = ds_onset_Atlantic.temp_adv + ds_onset_Atlantic.temp_sfch + ds_onset_Atlantic.temp_kpp + ds_onset_Atlantic.temp_vdiff + ds_onset_Atlantic.temp_pme + ds_onset_Atlantic.temp_ndiff + ds_onset_Atlantic.temp_oth
ΔQ_Total_decay_Atlantic = ds_decay_Atlantic.temp_adv + ds_decay_Atlantic.temp_sfch + ds_decay_Atlantic.temp_kpp + ds_decay_Atlantic.temp_vdiff + ds_decay_Atlantic.temp_pme + ds_decay_Atlantic.temp_ndiff + ds_decay_Atlantic.temp_oth

ΔQ_Total_onset_Indian = ds_onset_Indian.temp_adv + ds_onset_Indian.temp_sfch + ds_onset_Indian.temp_kpp + ds_onset_Indian.temp_vdiff + ds_onset_Indian.temp_pme + ds_onset_Indian.temp_ndiff + ds_onset_Indian.temp_oth
ΔQ_Total_decay_Indian = ds_decay_Indian.temp_adv + ds_decay_Indian.temp_sfch + ds_decay_Indian.temp_kpp + ds_decay_Indian.temp_vdiff + ds_decay_Indian.temp_pme + ds_decay_Indian.temp_ndiff + ds_decay_Indian.temp_oth



## Circumpolar Averages (Row 1)
Total_onset_Circumpolar = np.nanmean(ΔQ_Total_onset)
Total_decay_Circumpolar = np.nanmean(ΔQ_Total_decay)
Adv_onset_Circumpolar = np.nanmean(ds_onset_SO.temp_adv)
Adv_decay_Circumpolar = np.nanmean(ds_decay_SO.temp_adv)
as_onset_Circumpolar = np.nanmean(ds_onset_SO.temp_sfch)
as_decay_Circumpolar = np.nanmean(ds_decay_SO.temp_sfch)
vmix_onset_Circumpolar = np.nanmean(ds_onset_SO.temp_kpp)
vmix_decay_Circumpolar = np.nanmean(ds_decay_SO.temp_kpp)
vdiff_onset_Circumpolar = np.nanmean(ds_onset_SO.temp_vdiff)
vdiff_decay_Circumpolar = np.nanmean(ds_decay_SO.temp_vdiff)
res_onset_Circumpolar = np.nanmean(ds_onset_SO.temp_pme + ds_onset_SO.temp_ndiff + ds_onset_SO.temp_oth)
res_decay_Circumpolar = np.nanmean(ds_decay_SO.temp_pme + ds_decay_SO.temp_ndiff + ds_decay_SO.temp_oth)


## Pacific Averages (Row 2)
Total_onset_Pacific = np.nanmean(ΔQ_Total_onset_Pacific)
Total_decay_Pacific = np.nanmean(ΔQ_Total_decay_Pacific)
Adv_onset_Pacific = np.nanmean(ds_onset_Pacific.temp_adv)
Adv_decay_Pacific = np.nanmean(ds_decay_Pacific.temp_adv)
as_onset_Pacific = np.nanmean(ds_onset_Pacific.temp_sfch)
as_decay_Pacific = np.nanmean(ds_decay_Pacific.temp_sfch)
vmix_onset_Pacific = np.nanmean(ds_onset_Pacific.temp_kpp)
vmix_decay_Pacific = np.nanmean(ds_decay_Pacific.temp_kpp)
vdiff_onset_Pacific = np.nanmean(ds_onset_Pacific.temp_vdiff)
vdiff_decay_Pacific = np.nanmean(ds_decay_Pacific.temp_vdiff)
res_onset_Pacific = np.nanmean(ds_onset_Pacific.temp_pme + ds_onset_Pacific.temp_ndiff + ds_onset_Pacific.temp_oth)
res_decay_Pacific = np.nanmean(ds_decay_Pacific.temp_pme + ds_decay_Pacific.temp_ndiff + ds_decay_Pacific.temp_oth)


## Atlantic Averages (Row 3)
Total_onset_Atlantic = np.nanmean(ΔQ_Total_onset_Atlantic)
Total_decay_Atlantic = np.nanmean(ΔQ_Total_decay_Atlantic)
Adv_onset_Atlantic = np.nanmean(ds_onset_Atlantic.temp_adv)
Adv_decay_Atlantic = np.nanmean(ds_decay_Atlantic.temp_adv)
as_onset_Atlantic = np.nanmean(ds_onset_Atlantic.temp_sfch)
as_decay_Atlantic = np.nanmean(ds_decay_Atlantic.temp_sfch)
vmix_onset_Atlantic = np.nanmean(ds_onset_Atlantic.temp_kpp)
vmix_decay_Atlantic = np.nanmean(ds_decay_Atlantic.temp_kpp)
vdiff_onset_Atlantic = np.nanmean(ds_onset_Atlantic.temp_vdiff)
vdiff_decay_Atlantic = np.nanmean(ds_decay_Atlantic.temp_vdiff)
res_onset_Atlantic = np.nanmean(ds_onset_Atlantic.temp_pme + ds_onset_Atlantic.temp_ndiff + ds_onset_Atlantic.temp_oth)
res_decay_Atlantic = np.nanmean(ds_decay_Atlantic.temp_pme + ds_decay_Atlantic.temp_ndiff + ds_decay_Atlantic.temp_oth)


## Indian Averages (Row 4)
Total_onset_Indian = np.nanmean(ΔQ_Total_onset_Indian)
Total_decay_Indian = np.nanmean(ΔQ_Total_decay_Indian)
Adv_onset_Indian = np.nanmean(ds_onset_Indian.temp_adv)
Adv_decay_Indian = np.nanmean(ds_decay_Indian.temp_adv)
as_onset_Indian = np.nanmean(ds_onset_Indian.temp_sfch)
as_decay_Indian = np.nanmean(ds_decay_Indian.temp_sfch)
vmix_onset_Indian = np.nanmean(ds_onset_Indian.temp_kpp)
vmix_decay_Indian = np.nanmean(ds_decay_Indian.temp_kpp)
vdiff_onset_Indian = np.nanmean(ds_onset_Indian.temp_vdiff)
vdiff_decay_Indian = np.nanmean(ds_decay_Indian.temp_vdiff)
res_onset_Indian = np.nanmean(ds_onset_Indian.temp_pme + ds_onset_Indian.temp_ndiff + ds_onset_Indian.temp_oth)
res_decay_Indian = np.nanmean(ds_decay_Indian.temp_pme + ds_decay_Indian.temp_ndiff + ds_decay_Indian.temp_oth)


## Repeat with ds_clim



## Pcolor Data matrices ##
regiones = ['Circumpolar', 'Pacific', 'Atlantic', 'Indian']

variables = ['Total_onset', 'Total_decay', 'Adv_onset', 'Adv_decay', 
             'as_onset', 'as_decay', 'vmix_onset', 'vmix_decay', 
             'vdiff_onset', 'vdiff_decay', 'res_onset', 'res_decay']

data_matrix_piControl = np.empty((len(regiones), len(variables)))

for i, region in enumerate(regiones):
    for j, variable in enumerate(variables):
        
        variable_name = f'{variable}_{region}'
        
        data_matrix_piControl[i, j] = np.nanmean(eval(variable_name))


data_matrix = np.empty((len(regiones), len(variables)))

for i, region in enumerate(regiones):
    for j, variable in enumerate(variables):
        
        variable_name = f'{variable}_{region}'
        
        data_matrix[i, j] = np.nanmean(eval(variable_name))




## Loading previously proccessed data matrices --------------------------------
file = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Temperature_tendency_terms\ESM2M_Datasets\Mean_contribution_heat_budget_data_matrix_pi_control_simulation'
data_matrix_piControl = np.load(file+'.npy')

file = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Temperature_tendency_terms\ESM2M_Datasets\Mean_contribution_heat_budget_data_matrix_1982_2021'
data_matrix = np.load(file+'.npy')

        
vmin = -55    
vmax=55

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 20

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

pcm1 = ax1.pcolor(data_matrix_piControl, cmap='RdBu_r', vmin=vmin, vmax=vmax)
ax1.set_aspect('equal')
ax1.set_yticks([])
ax1.set_xticks(range(len(data_matrix_piControl[0])))
for i in range(len(data_matrix_piControl)):
    for j in range(len(data_matrix_piControl[i])):
        ax1.text(j + 0.5, i + 0.5, f'{data_matrix_piControl[i, j]:.1f}', color='black',
                ha='center', va='center')  
        if j % 2 == 0:
            ax1.axvline(j, color='black', linewidth=2)  
        ax1.axhline(i, color='black', linewidth=2)  

pcm2 = ax2.pcolor(data_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax)
ax2.set_aspect('equal')
ax2.set_yticks([])
ax2.set_xticks(range(len(data_matrix[0])))
for i in range(len(data_matrix)):
    for j in range(len(data_matrix[i])):
        ax2.text(j + 0.5, i + 0.5, f'{data_matrix[i, j]:.1f}', color='black',
                  ha='center', va='center')  
        if j % 2 == 0:
            ax2.axvline(j, color='black', linewidth=2)  
        ax2.axhline(i, color='black', linewidth=2)  


ax1.tick_params(axis='x', which='both', bottom=False, top=True, direction='in', labelsize=0, length=15, width=1.5)  
ax2.tick_params(axis='x', which='both', bottom=True, top=False, direction='in', labelsize=0, length=15, width=1.5)  

cbar_cax = fig.add_axes([0.83, 0.205, 0.02, 0.6])
cbar = plt.colorbar(pcm1, cax=cbar_cax, ticks=[vmin, 0, vmax])
custom_ticklabels = ['Counteract MHW formation (+) /\nSupport MHW decline (-)', 'Neutral', 'Support MHW formation (+) /\nCounteract MHW decline (-)']
cbar.ax.set_yticklabels(custom_ticklabels)

# Ajusta los márgenes
plt.subplots_adjust(left=0.1, right=0.85, hspace=0.2)



outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\NatComms_Earth_nd_Environment\Review\Temperature_tendency_terms\Figures\Temperature_Tendency_Budgets_Table.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')





