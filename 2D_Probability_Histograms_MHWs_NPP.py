# -*- coding: utf-8 -*-
"""

################## 2D Probability Histograms MHWs - NPP ######################

"""

#Loading required libraries
import numpy as np
import xarray as xr 

import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import cartopy.feature as cft

import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter

from scipy.stats import spearmanr

# Load MHW_metrics_from_MATLAB.py
runcell(0, './MHW_metrics_from_MATLAB.py')


#Max SSTAs
ds = xr.open_dataset(r'.\Max_SSTA_ts.nc')
Max_SSTA = ds.analysed_sst
Max_SSTA = Max_SSTA+mask_ts
Max_SSTA = Max_SSTA[:,:,16:40]

Aver_Max_SSTA = np.nanmean(Max_SSTA, axis=2)

#MHW Duration
Aver_MHW_dur = np.nanmean(MHW_dur_ts[:,:,16:40], axis=2)
Aver_MHW_dur = Aver_MHW_dur+mask


#MHW Cumulative Intensity
Aver_MHW_cum = np.nanmean(MHW_cum_ts[:,:,16:40], axis=2)
Aver_MHW_cum = Aver_MHW_cum+mask

NPP_CbPM_interp = NPP_CbPM_interp+mask_ts
Aver_NPP_CbPM = np.nanmean(NPP_CbPM_interp, axis=2)





# Plotting 2D Probability Histogram
fig, (axs1, axs2, axs3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(Aver_Max_SSTA) & ~np.isnan(Aver_NPP_CbPM)
Max_SSTA_clean = Aver_Max_SSTA[valid_indices]
NPP_clean_1 = Aver_NPP_CbPM[valid_indices]


#Define the number of bins in x and y
x_bins = np.arange(0.5, 3, 0.25)
y_bins = np.arange(0, 500, len(x_bins))

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    Max_SSTA_clean,
    NPP_clean_1,
    bins=(x_bins, y_bins),
    density=True 
)

# Calculate the probability
probability = (hist / np.sum(hist))


cmap=plt.cm.Greens
vmin=0
vmax=0.015
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs1 = axs1.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs1.set_xlabel(r'Máxima SSTA [$ºC$]', fontsize=14)
axs1.set_ylabel(r'CbPM NPP [$mg C·m^{-2}·día^{-1}$]', fontsize=14)

axs1.xaxis.set_minor_locator(AutoMinorLocator())
axs1.yaxis.set_minor_locator(AutoMinorLocator())
axs1.minorticks_on()
axs1.grid(which='both', linestyle='-', linewidth=0.5)


cbar = fig.colorbar(cs1, ax=axs3, extend='max', location='right', pad=0.05)
cbar.ax.minorticks_off()
cbar.ax.set_ylabel('Probabilidad', fontsize=16)
custom_colorbar_ticks = np.arange(vmin, 0.016, 0.002)
cbar.set_ticks(custom_colorbar_ticks)
cbar.set_ticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'])



## MHW Duration vs. NPP ##

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(Aver_MHW_dur) & ~np.isnan(Aver_NPP_CbPM)
MHW_dur_clean = Aver_MHW_dur[valid_indices]
NPP_clean_2 = Aver_NPP_CbPM[valid_indices]


#Define the number of bins in x and y
x_bins = np.arange(5, 20, 2.5)
y_bins = np.arange(0, 500, len(x_bins))

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MHW_dur_clean,
    NPP_clean_2,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

vmin=0
vmax=0.01
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs2 = axs2.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs2.set_xlabel(r'Duración MHW [$días$]', fontsize=14)
# axs2.set_ylabel(r'BMHW / SMHW Max Intensity', fontsize=14)
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2.yaxis.set_minor_locator(AutoMinorLocator())
axs2.minorticks_on()
axs2.grid(which='both', linestyle='-', linewidth=0.5)



## Cum Int vs NPP ##

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(Aver_MHW_cum) & ~np.isnan(Aver_NPP_CbPM)
MHW_cum_clean = Aver_MHW_cum[valid_indices]
NPP_clean_3 = Aver_NPP_CbPM[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(5, 22.5, 5)
y_bins = np.arange(0, 500, len(x_bins))

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MHW_dur_clean,
    NPP_clean_3,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

vmin=0
vmax=0.01
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
im3 = axs3.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs3.set_xlabel(r'Int. Acumulada MHW [$ºC·días$]', fontsize=14)
axs3.xaxis.set_minor_locator(AutoMinorLocator())
axs3.yaxis.set_minor_locator(AutoMinorLocator())
axs3.minorticks_on()
axs3.grid(which='both', linestyle='-', linewidth=0.5)



# Calculate Spearman correlation for each subplot
corr1, _ = spearmanr(Max_SSTA_clean, NPP_clean_1)
corr2, _ = spearmanr(MHW_dur_clean, NPP_clean_2)
corr3, _ = spearmanr(MHW_cum_clean, NPP_clean_3)

# Annotate each subplot with the Spearman correlation value
axs1.annotate(f'{corr1:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs2.annotate(f'{corr2:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs3.annotate(f'{corr3:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')

plt.tight_layout()


outfile = r'.\2D_Histograms_CB.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

