# -*- coding: utf-8 -*-
"""
#########################  Max SSTA TS along with Climate Indices #############
"""

# Load required modules
import numpy as np
import pandas as pd
from scipy import io
from datetime import date
from netCDF4 import Dataset 
import xarray as xr 
import matplotlib
import matplotlib.pyplot as plt


# #Load sea-ice mask
file = r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\MHW_metrics\MATLAB_metrics\mask'
data_mask = np.load(file+'.npz')
mask = data_mask['mask']
mask_ts=mask[:,:,np.newaxis]


#SST
ds_sst = xr.open_mfdataset(r'E:\ICMAN-CSIC\SO_MHW\Datasets\SST\*.nc', parallel=True)

#Compute climatology

#Climatology 1982-2011
ds_clim=ds_sst.sel(time=slice("1982-01-01", "2011-12-31"))
sst_clim=ds_clim['analysed_sst'][:,::10,::10].groupby('time.month').mean(dim='time', skipna=True)


#Compute SST Anomaly
sst_anom=ds_sst['analysed_sst'][:,::10,::10].groupby('time.month') - sst_clim

Max_SSTA_monthly = sst_anom.resample(time='1M').max().T  # Resample monthly Max SSTA
Max_SSTA_monthly_masked = Max_SSTA_monthly+mask_ts


#Pacific
lon_slice1 = Max_SSTA_monthly.sel(lon=slice(-180, -70))
lon_slice2 = Max_SSTA_monthly.sel(lon=slice(150, 180))
# Combina las dos selecciones de longitud
PAC_Max_SSTA_monthly = xr.concat([lon_slice1, lon_slice2], dim="lon")

#Atlantic
ATL_Max_SSTA_monthly = Max_SSTA_monthly.sel(lon=slice(-70, -20))

#Indian
IND_Max_SSTA_monthly = Max_SSTA_monthly.sel(lon=slice(20, 150))


#Monthly timeseries in each sector
PAC_Max_SSTA_monthly_TS = PAC_Max_SSTA_monthly.mean(dim=('lon', 'lat'),skipna=True)

ATL_Max_SSTA_monthly_TS = ATL_Max_SSTA_monthly.mean(dim=('lon', 'lat'),skipna=True)

IND_Max_SSTA_monthly_TS = IND_Max_SSTA_monthly.mean(dim=('lon', 'lat'),skipna=True)


#Convert to numpy array to save propertly
PAC_Max_SSTA_monthly_TS = np.squeeze(np.asarray(PAC_Max_SSTA_monthly_TS))
ATL_Max_SSTA_monthly_TS = np.squeeze(np.asarray(ATL_Max_SSTA_monthly_TS))
IND_Max_SSTA_monthly_TS = np.squeeze(np.asarray(IND_Max_SSTA_monthly_TS))

#Save data so far
outfile = r'E:\ICMAN-CSIC\SO_MHW\Datasets\Max_SSTA_monthly\Max_SSTA_monthly_TS'
np.savez(outfile, PAC_Max_SSTA_monthly_TS=PAC_Max_SSTA_monthly_TS, ATL_Max_SSTA_monthly_TS=ATL_Max_SSTA_monthly_TS, IND_Max_SSTA_monthly_TS=IND_Max_SSTA_monthly_TS)




###############################################################################
#Load previously-proccessed Max SSTA data
file = r'E:\ICMAN-CSIC\SO_MHW\Datasets\Max_SSTA_monthly\Max_SSTA_monthly_TS'
data_Max_SSTA = np.load(file+'.npz')
PAC_Max_SSTA_monthly_TS = data_Max_SSTA['PAC_Max_SSTA_monthly_TS']+2.75
ATL_Max_SSTA_monthly_TS = data_Max_SSTA['ATL_Max_SSTA_monthly_TS']+2.75
IND_Max_SSTA_monthly_TS = data_Max_SSTA['IND_Max_SSTA_monthly_TS']+2.75


#Load Climate Mode Indices (ONI and SAMI)
ENSO = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Climate Modes\ENSO34_Index.xlsx', header=None)
SAM = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Climate Modes\SAM_Index_Monthly.xlsx', header=None)
TSA = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\datasets_40\Climate Modes\TSA_Index.xlsx', header=None)

#Converting to arrays
ENSO = np.squeeze(np.asarray(ENSO))
SAM = np.squeeze(np.asarray(SAM))
TSA = np.squeeze(np.asarray(TSA))

data1 = PAC_Max_SSTA_monthly_TS
data2 = ATL_Max_SSTA_monthly_TS
data3 = IND_Max_SSTA_monthly_TS
                      
start_index_Clim = (2011 - 1982) * 12  # First month index of first year
end_index_Clim = start_index_Clim + 24  # Last month index of last year

data4 = ENSO[start_index_Clim:end_index_Clim]
# data4 = SAM[start_index_Clim:end_index_Clim]
# data4 = TSA[start_index_Clim:end_index_Clim]
###############################################################################




                            # Max SSTA along with ONI #
###############################################################################

years = np.arange(1982, 2022)
months = ['Jan-15', 'Feb-15', 'Mar-15', 'Apr-15', 'May-15', 'Jun-15', 'Jul-15', 'Aug-15', 'Sep-15', 'Oct-15', 'Nov-15', 'Dec-15', 'Jan-16', 'Feb-16', 'Mar-16', 'Apr-16', 'May-16', 'Jun-16', 'Jul-16', 'Aug-16', 'Sep-16', 'Oct-16', 'Nov-16', 'Dec-16'] # Array con los nombres de los meses para dos años
formated_months = ['Jan', '', 'Mar', '', 'May', '', 'Jul', '', 'Sep', '', 'Nov', '', 'Jan', '', 'Mar', '', 'May', '', 'Jul', '', 'Sep', '', 'Nov', '']

## PACIFIC ONI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2015, 2017)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data1[start_index:start_index+24], color=line_color, linewidth=line_width, label='2015-2016')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'darkred', linewidth=2, alpha=1, label='ONI')
ax2.tick_params(length=10, direction='in', color='darkred', axis='y', colors='darkred')
ax2.spines['right'].set_color('darkred')
ax2.yaxis.label.set_color('darkred')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.6)
ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(-2.75, 2.75)  # ONI
ax2.set_ylabel('ONI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Pacific [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.5, 3.75, 4])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

ax1.legend([black_line, line2], ['2015-2016', 'ONI 2015-2016'], loc='upper left', frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_ONI_Pacific.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)





## Atlantic ONI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2015, 2017)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data2[start_index:start_index+24], color=line_color, linewidth=line_width, label='2015-2016')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'darkred', linewidth=2, alpha=1, label='ONI')
ax2.tick_params(length=10, direction='in', color='darkred', axis='y', colors='darkred')
ax2.spines['right'].set_color('darkred')
ax2.yaxis.label.set_color('darkred')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.6)
ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(-2.75, 2.75)  # ONI
ax2.set_ylabel('ONI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Atlantic [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.5, 3.75, 4,])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

# ax1.legend([black_line, line2], ['2015-2016', 'ONI 2015-2016'], loc='upper left', frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_ONI_Atlantic.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)







## Indian ONI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2015, 2017)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data3[start_index:start_index+24], color=line_color, linewidth=line_width, label='2015-2016')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'darkred', linewidth=2, alpha=1, label='ONI')
ax2.tick_params(length=10, direction='in', color='darkred', axis='y', colors='darkred')
ax2.spines['right'].set_color('darkred')
ax2.yaxis.label.set_color('darkred')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.6)
ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(-2.75, 2.75)  # ONI
ax2.set_ylabel('ONI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Indian [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.5, 3.75, 4])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

# ax1.legend([black_line, line2], ['2015-2016', 'ONI 2015-2016'], loc='upper left', frameon=False)

outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_ONI_Indian.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)






###############################################################################
start_index_Clim = (2014 - 1982) * 12  # First month index of first year
end_index_Clim = start_index_Clim + 24  # Last month index of last year

# data4 = ENSO[start_index_Clim:end_index_Clim]
data4 = SAM[start_index_Clim:end_index_Clim]
###############################################################################


                        # Max SSTA along with SAMI #
###############################################################################

years = np.arange(1982, 2022)
months = ['Jan-15', 'Feb-15', 'Mar-15', 'Apr-15', 'May-15', 'Jun-15', 'Jul-15', 'Aug-15', 'Sep-15', 'Oct-15', 'Nov-15', 'Dec-15', 'Jan-16', 'Feb-16', 'Mar-16', 'Apr-16', 'May-16', 'Jun-16', 'Jul-16', 'Aug-16', 'Sep-16', 'Oct-16', 'Nov-16', 'Dec-16'] # Array con los nombres de los meses para dos años
formated_months = ['Jan', '', 'Mar', '', 'May', '', 'Jul', '', 'Sep', '', 'Nov', '', 'Jan', '', 'Mar', '', 'May', '', 'Jul', '', 'Sep', '', 'Nov', '']

## PACIFIC SAMI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2014, 2016)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data1[start_index:start_index+24], color=line_color, linewidth=line_width, label='2014-2015')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'darkorange', linewidth=2, alpha=1, label='ONI')
ax2.tick_params(length=10, direction='in', color='darkorange', axis='y', colors='darkorange')
ax2.spines['right'].set_color('darkorange')
ax2.yaxis.label.set_color('darkorange')
# ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='darkorange', linestyle='--', alpha=0.6)
# ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(9, -4.5)  # SAMI
ax2.set_ylabel('SAMI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Pacific [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.5, 3.75, 4])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

ax1.legend([black_line, line2], ['2014-2015', 'SAMI 2014-2015'], loc='upper left', frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_SAMI_Pacific.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)




## ATLANTIC SAMI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2014, 2016)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data2[start_index:start_index+24], color=line_color, linewidth=line_width, label='2014-2015')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'darkorange', linewidth=2, alpha=1, label='ONI')
ax2.tick_params(length=10, direction='in', color='darkorange', axis='y', colors='darkorange')
ax2.spines['right'].set_color('darkorange')
ax2.yaxis.label.set_color('darkorange')
# ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='darkorange', linestyle='--', alpha=0.6)
# ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(9, -4.5)  # SAMI
ax2.set_ylabel('SAMI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Atlantic [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.5, 3.75, 4])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

# ax1.legend([black_line, line2], ['2014-2015', 'SAMI 2014-2015'], loc='upper left', frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_SAMI_Atlantic.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)




## INDIAN SAMI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2013, 2015)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data3[start_index:start_index+24], color=line_color, linewidth=line_width, label='2014-2015')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'darkorange', linewidth=2, alpha=1, label='ONI')
ax2.tick_params(length=10, direction='in', color='darkorange', axis='y', colors='darkorange')
ax2.spines['right'].set_color('darkorange')
ax2.yaxis.label.set_color('darkorange')
# ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='darkorange', linestyle='--', alpha=0.6)
# ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(9, -4.5)  # SAMI
ax2.set_ylabel('SAMI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Indian [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.5, 3.75, 4])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

# ax1.legend([black_line, line2], ['2014-2015', 'SAMI 2014-2015'], loc='upper left', frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_SAMI_Indian.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)




















###############################################################################
start_index_Clim = (2019 - 1982) * 12  # First month index of first year
end_index_Clim = start_index_Clim + 24  # Last month index of last year


data4 = TSA[start_index_Clim:end_index_Clim]
###############################################################################


                        # Max SSTA along with TSAI #
###############################################################################

years = np.arange(1982, 2022)
months = ['Jan-15', 'Feb-15', 'Mar-15', 'Apr-15', 'May-15', 'Jun-15', 'Jul-15', 'Aug-15', 'Sep-15', 'Oct-15', 'Nov-15', 'Dec-15', 'Jan-16', 'Feb-16', 'Mar-16', 'Apr-16', 'May-16', 'Jun-16', 'Jul-16', 'Aug-16', 'Sep-16', 'Oct-16', 'Nov-16', 'Dec-16'] # Array con los nombres de los meses para dos años
formated_months = ['Jan', '', 'Mar', '', 'May', '', 'Jul', '', 'Sep', '', 'Nov', '', 'Jan', '', 'Mar', '', 'May', '', 'Jul', '', 'Sep', '', 'Nov', '']

## PACIFIC TSAI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2019, 2021)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data1[start_index:start_index+24], color=line_color, linewidth=line_width, label='2019-2020')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'lightseagreen', linewidth=2, alpha=1, label='ONI')
ax2.tick_params(length=10, direction='in', color='lightseagreen', axis='y', colors='lightseagreen')
ax2.spines['right'].set_color('lightseagreen')
ax2.yaxis.label.set_color('lightseagreen')
# ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='lightseagreen', linestyle='--', alpha=0.6)
# ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(-1.5, 1.5)  # AMOI
ax2.set_ylabel('TSAI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Pacific [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.5, 3.75, 4])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

ax1.legend([black_line, line2], ['2019-2020', 'TSAI 2019-2020'], loc='upper left', bbox_to_anchor=(0.006, 1.03), frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_TSAI_Pacific.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)




## ATLANTIC TSAI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2019, 2021)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data2[start_index:start_index+24], color=line_color, linewidth=line_width, label='2014-2015')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'lightseagreen', linewidth=2, alpha=1, label='AMOI')
ax2.tick_params(length=10, direction='in', color='lightseagreen', axis='y', colors='lightseagreen')
ax2.spines['right'].set_color('lightseagreen')
ax2.yaxis.label.set_color('lightseagreen')
# ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='lightseagreen', linestyle='--', alpha=0.6)
# ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(-1.5, 1.5)  # SAMI
ax2.set_ylabel('AMOI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Atlantic [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.50, 3.75, 4])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

# ax1.legend([black_line, line2], ['2014-2015', 'SAMI 2014-2015'], loc='upper left', frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_TSAI_Atlantic.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)




## INDIAN TSAI ##

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Date range to highlight
highlighted = np.arange(2017, 2019)

# Representing Data1
for i in range(0, len(years), 2):
    start_index = i * 12
    end_index = (i + 2) * 12
    line_color = 'gray' if years[i] not in highlighted else 'black'
    line_width = 1 if years[i] not in highlighted else 2
    line, = ax1.plot(months, data3[start_index:start_index+24], color=line_color, linewidth=line_width, label='2014-2015')
    if years[i] in highlighted:
        black_line = line

ax2 = ax1.twinx()
line2, = ax2.plot(months, data4, 'lightseagreen', linewidth=2, alpha=1, label='TSAI')
ax2.tick_params(length=10, direction='in', color='lightseagreen', axis='y', colors='lightseagreen')
ax2.spines['right'].set_color('lightseagreen')
ax2.yaxis.label.set_color('lightseagreen')
# ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
ax2.axhline(0, color='lightseagreen', linestyle='--', alpha=0.6)
# ax2.axhline(-0.5, color='blue', linestyle='--', alpha=0.6)
ax2.set_ylim(-1.5, 1.5)  # SAMI
ax2.set_ylabel('TSAI')

ax1.tick_params(length=10, direction='in')
ax1.set_ylabel(r'Maximum SSTA [$^\circ$C]')
ax1.set_title('Indian [Reference period: 1982-2012]')
ax1.set_ylim(2.75, 4.25)
ax1.set_yticks([3, 3.25, 3.5, 3.75, 4])
plt.xticks(months, formated_months)

black_line.set_color('black')  # Set the color of the highlighted line to black
black_line.set_linewidth(2)  # Set the width of the highlighted line to 2

# ax1.legend([black_line, line2], ['2014-2015', 'SAMI 2014-2015'], loc='upper left', frameon=False)


outfile = r'C:\Users\Manuel\Desktop\Paper_SO_MHWs\Reviews\Figuras\Max_SSTA_TSAI_Indian.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)












