# -*- coding: utf-8 -*-
"""

############################# SO MHW METRICS ###################################

"""

from netCDF4 import Dataset
import numpy as np


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
import matplotlib.ticker as ticker


##Plotting the South Polar Stereo map with metrics#####################

############
### Duration ###
############

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

cmap=plt.cm.YlOrRd
# cmap=cm.cm.thermal
#cmap=plt.cm.RdBu_r  

levels = [2.5,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30]
p1 = plt.contourf(lon, lat, MHW_dur+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.set_ticks([5, 10, 15, 20, 25, 30])
#p1 = plt.pcolormesh(lon, lat, MHW_dur_tr, vmin=-2, vmax=2, cmap=cmap, transform=ccrs.PlateCarree())

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('g) Duration [$days$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'.\MHW_Dur_hq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')


                    ######### Duration trend ##########

signif_dur = MHW_dur_dtr
signif_dur = np.where(signif_dur >= 0.05, np.NaN, signif_dur)
signif_dur = np.where(signif_dur < 0.05, 1, signif_dur)

                         
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

cmap=plt.cm.RdYlBu_r
# cmap=plt.cm.coolwarm
levels = [-5, -3.75, -2.5, -1.25, 0, 1.25, 2.5, 3.75, 5]
p1 = plt.contourf(lon, lat, MHW_dur_tr*10+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_dur[::4,::2]+mask[::4,::2], color='black',linewidth=2,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.8, format=ticker.FormatStrFormatter('%.1f'), extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.set_ticks([-5,-2.5,0,2.5,5])


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
ax.set_title('h) Duration trend [$days\ decade^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'.\MHW_Dur_tr.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')


#############
### Frequency ###
#############

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

cmap=plt.cm.YlOrRd  

levels = [1.001,1.25,1.5,1.75,2,2.25,2.5]
p1 = plt.contourf(lon, lat, MHW_cnt+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
cbar = plt.colorbar(p1, shrink=0.8, format=ticker.FormatStrFormatter('%.1f'), extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.set_ticks([1.001, 1.5, 2, 2.5])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
ax.set_title('d) Frequency [$number$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'.\MHW_Freq.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')


                     ######### Frequency trend ##########

signif_freq = MHW_cnt_dtr
signif_freq = np.where(signif_freq >= 0.05, np.NaN, signif_freq)
signif_freq = np.where(signif_freq < 0.05, 1, signif_freq)

                         
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
 
cmap=plt.cm.RdYlBu_r
levels = [-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5]
p1 = plt.contourf(lon, lat, MHW_cnt_tr*10+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_freq[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())


cbar = plt.colorbar(p1, shrink=0.8, format=ticker.FormatStrFormatter('%.1f'), extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
#p1 = plt.pcolormesh(lon, lat, MHW_dur_tr, vmin=-2, vmax=2, cmap=cmap, transform=ccrs.PlateCarree())

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
ax.set_title('e) Frequency trend [$number\ decade^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'.\MHW_Freq_tr.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')


##################
### Mean Intensity ###
##################

plt.clf()
plt.subplot(2,1,1)

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
#cmap = cm.cm.thermal
#cmap = 'Spectral_r'
levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
p1 = plt.contourf(lon, lat, MHW_mean+mask, levels, cmap=plt.cm.YlOrRd, extend='both', transform=ccrs.PlateCarree()) 
cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=25) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([0, 1, 2, 3, 4, 5])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('x) Mean intensity [$^\circ$C]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


                 ######### Mean Intensity trend ##########
   
signif_mean = MHW_mean_dtr
signif_mean = np.where(signif_mean >= 0.05, np.NaN, signif_mean)
signif_mean = np.where(signif_mean < 0.05, 1, signif_mean)
                      
plt.subplot(2,1,2)
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
#cmap = cm.cm.thermal
#cmap = 'Spectral_r'
levels = [-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2]
p1 = plt.contourf(lon, lat, MHW_mean_tr*10+mask, levels, cmap=plt.cm.RdYlBu_r, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_mean[::4,::2]+mask[::4,::2], color='black',linewidth=2,marker='o', alpha=1,transform=ccrs.Geodetic())

cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=25) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-0.2, -0.1, 0, 0.1, 0.2])
#p1 = plt.pcolormesh(lon, lat, MHW_dur_tr, vmin=-2, vmax=2, cmap=cmap, transform=ccrs.PlateCarree())

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('x) Mean intensity trend [$^{\circ}C\ decade^{-1}$]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


#####################
### Maximun Intensity ###
#####################

plt.clf()
plt.subplot(2,1,1)

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
#cmap = cm.cm.thermal
#cmap = 'Spectral_r'
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

outfile = r'.\MHW_MaxInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')



                 ######### Maximum Intensity trend ##########

signif_max = MHW_max_dtr
signif_max = np.where(signif_max >= 0.05, np.NaN, signif_max)
signif_max = np.where(signif_max < 0.05, 1, signif_max)
                         
#Set the projection
projection=ccrs.SouthPolarStereo(central_longitude=0, globe=None)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1, projection=projection)

#Adding some cartopy natural features
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

ice_50m = cft.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='none', facecolor='white')

ax.add_feature(ice_50m)

cmap=plt.cm.RdYlBu_r
levels = [-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2]
# levels = [-0.2,-0.175,-0.15,-0.125,-0.1,-0.075,-0.05,-0.025,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2]
p1 = plt.contourf(lon, lat, MHW_max_tr*10+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_max[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())

cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-0.2, -0.1, 0, 0.1, 0.2])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('b) Max SSTA trend [$^{\circ}C\ decade^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'.\MHW_MaxInt_tr.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')



########################
### Cumulative Intensity###
########################

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

cmap=plt.cm.YlOrRd     

levels = [0,5,10,15,20,25,30,35,40]
p1 = plt.contourf(lon, lat, MHW_cum+mask, levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()) 
cbar = plt.colorbar(p1, shrink=0.8, format=ticker.FormatStrFormatter('%.0f'), extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 10, 20, 30,40])

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
ax.set_title( 'j) Cumulative intensity [$^{\circ}C\ days$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

outfile = r'.\MHW_CumInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')


                ######### Cumulative Intensity trend ##########

signif_cum = MHW_cum_dtr
signif_cum = np.where(signif_cum >= 0.05, np.NaN, signif_cum)
signif_cum = np.where(signif_cum < 0.05, 1, signif_cum)
                        
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

#cmap=plt.cm.YlOrRd     
#cmap = cm.cm.thermal
#cmap = 'Spectral_r'
levels = [-10,-7.5,-5,-2.5,0,2.5,5,7.5,10]
p1 = plt.contourf(lon, lat, MHW_cum_tr*10+mask, levels, cmap=plt.cm.RdYlBu_r, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_cum[::4,::2]+mask[::4,::2], color='black',linewidth=2,marker='o', alpha=1,transform=ccrs.Geodetic())

cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=35) 
cbar.ax.minorticks_off()
cbar.set_ticks([-10, -5, 0, 5, 10])


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color='black')
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
ax.set_title('k) Cumulative intensity trend [$^{\circ}C\ days\ decade^{-1}$]', fontsize=40)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


outfile = r'.\MHW_CumInt_tr.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')



#########################
### Total Annual MHW Days ###
#########################

plt.clf()
plt.subplot(2,1,1)

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
#cmap = cm.cm.thermal
#cmap = 'Spectral_r'
levels = [0,5,10,15,20,25,30,35,40]
p1 = plt.contourf(lon, lat, MHW_td+mask, levels, cmap=plt.cm.YlOrRd, extend='both', transform=ccrs.PlateCarree()) 
cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=25) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([0,10,20,30,40])
#p1 = plt.pcolormesh(lon, lat, MHW_dur_tr, vmin=-2, vmax=2, cmap=cmap, transform=ccrs.PlateCarree())

ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('x) Total annual MHW days [$days$]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)


                ######### Total Annual MHW Days trend ##########
 
signif_td = MHW_td_dtr
signif_td = np.where(signif_td >= 0.05, np.NaN, signif_td)
signif_td = np.where(signif_td < 0.05, 1, signif_td)
                        
plt.subplot(2,1,2)
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
#cmap = cm.cm.thermal
#cmap = 'Spectral_r'
levels = [-20,-15,-10,-5,0,5,10,15,20,25]
p1 = plt.contourf(lon, lat, MHW_td_tr*10+mask, levels, cmap=plt.cm.RdYlBu_r, extend='both', transform=ccrs.PlateCarree()) 
p2=plt.scatter(lon[::4,::2],lat[::4,::2], signif_td[::4,::2]+mask[::4,::2], color='black',linewidth=1.5,marker='o', alpha=1,transform=ccrs.Geodetic())

cbar = plt.colorbar(p1, shrink=0.8, extend ='both')
cbar.ax.tick_params(axis='y', size=5, direction='in', labelsize=25) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-15, -5, 5, 15, 25])


ax.set_extent([-280, 80, -80, -40], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', linewidth= 0.50)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title('x) Total annual MHW days trend [$days\ decade^{-1}$]', fontsize=28)

# Compute a circle in axes coordinates, which can be used as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
