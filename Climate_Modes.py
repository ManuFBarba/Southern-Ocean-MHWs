# -*- coding: utf-8 -*-
"""
################################## Climate Modes ##############################
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
from scipy import stats

#Load MHW_metrics_from_MATLAB.py

#Load Averaged_MHW_Metrics.py


#Importing Climate Modes files
PDO = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Climate Modes\PDO_Index.xlsx', header=None)
ENSO = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Climate Modes\ENSO34_Index.xlsx', header=None)
AMO = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Climate Modes\AMO_Index.xlsx', header=None)
SAM = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Climate Modes\SAM_Index.xlsx', header=None)
AAO = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Climate Modes\AAO_Index.xlsx', header=None)
SOI = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Climate Modes\SOI.xlsx', header=None)
TSA = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Climate Modes\TSA_Index.xlsx', header=None)
IPO = pd.read_excel (r'C:\ICMAN-CSIC\MHW_ANT\Climate Modes\TPI_IPO.xlsx', header=None)

#Converting to arrays
PDO = np.squeeze(np.asarray(PDO))
ENSO = np.squeeze(np.asarray(ENSO))
AMO = np.squeeze(np.asarray(AMO))
SAM = np.squeeze(np.asarray(SAM))
AAO = np.squeeze(np.asarray(AAO))
SOI = np.squeeze(np.asarray(SOI))
TSA = np.squeeze(np.asarray(TSA))
IPO = np.squeeze(np.asarray(IPO))

##ENSO and AAO sd
ENSO_sd = np.std(ENSO.reshape(-1, 12), axis=1)
AAO_sd = np.std(AAO.reshape(-1, 12), axis=1)
#Errors
ENSO_error = ENSO_sd/np.sqrt(40) 
SAM_error = AAO_sd/np.sqrt(40)


#Converting monthly data to annual
PDO = np.average(PDO.reshape(-1, 12), axis=1)
ENSO = np.average(ENSO.reshape(-1, 12), axis=1)
AMO = np.average(AMO.reshape(-1, 12), axis=1)
AAO = np.average(AAO.reshape(-1, 12), axis=1)
SOI = np.average(SOI.reshape(-1, 12), axis=1)
TSA = np.average(TSA.reshape(-1, 12), axis=1)
IPO = np.average(IPO.reshape(-1, 12), axis=1)


# #Computing a 3 years moving mean over N-SAT
# def rollavg_roll_edges(a,n):
#     'Numpy array rolling, edge handling'
#     assert n%2==1
#     a = np.pad(a,(0,n-1-n//2), 'constant')*np.ones(n)[:,None]
#     m = a.shape[1]
#     idx = np.mod((m-1)*np.arange(n)[:,None] + np.arange(m), m) # Rolling index
#     out = a[np.arange(-n//2,n//2)[:,None], idx]
#     d = np.hstack((np.arange(1,n),np.ones(m-2*n+1+n//2)*n,np.arange(n,n//2,-1)))
#     return (out.sum(axis=0)/d)[n//2:]

# window = 1

# SAM_averaged = rollavg_roll_edges(SAM, window)

#Create monthly array 
annual_dates = pd.date_range('1982-01-01','2021-12-31', 
                freq='Y').strftime("%Y")
dates = np.squeeze(np.asarray(annual_dates))

#Defining the x-axis (years)
# time = np.arange(1982, 2022)


#DataFram containing timeseries of climate modes
Climate_Modes = pd.DataFrame({'Year':dates, 'PDOI':PDO, 'ONI':ENSO, 'AMOI':AMO, 'SAMI':SAM})
Climate_Modes['Year'] = pd.to_datetime(Climate_Modes['Year'])
Climate_Modes.set_index('Year',inplace=True)

Climate_Modes.plot(figsize=(20, 15))



############################################################
###Averaged MHW Metrics subplots along with climate modes###
############################################################
time = np.arange(1982, 2022)

fig, axs = plt.subplots(3, 2, figsize=(20, 15))
plt.rcParams.update({'font.size': 21})

#Max SSTA
# res_Max_SSTA = stats.linregress(time, PAC_Max_SSTA_ts)
axs[0, 0].plot(time, PAC_Max_SSTA_ts, '-', color='orangered', linewidth=2, label='MHW metric')
axs[0, 0].plot(time, ATL_Max_SSTA_ts, '-', color='xkcd:mango', linewidth=2)
axs[0, 0].plot(time, IND_Max_SSTA_ts, '-', color='gold', linewidth=2)
# axs[0, 0].plot(time, res_Max_SSTA.intercept + res_Max_SSTA.slope*time, 'k--')
ax2 = axs[0, 0].twinx()
ax2.plot(time, SAM, 'gray', linewidth=3, alpha=1, label='SAM')
ax2.fill_between(time, SAM-(SAM_error+1), SAM+(SAM_error+1),
    alpha=0.5, edgecolor='gray', facecolor='gray',
    linewidth=0, antialiased=True)
ax2.tick_params(length=10, direction='in', color='gray', axis='y', colors='gray')
ax2.spines['right'].set_color('gray')
ax2.yaxis.label.set_color('gray')
ax2.set_ylim(6, -6)

axs[0, 0].tick_params(length=10, direction='in')
axs[0, 0].set_title('a Maximum SSTA [$^\circ$C]')
axs[0, 0].set_xlim(1981, 2022)
#axs[0, 0].set_ylim(2.8, 3.65)
# axs[0, 0].grid(linestyle=':', linewidth=1)
# fig.legend(loc='upper center', frameon=True, fontsize=25)


#Frequency
# res_cnt = stats.linregress(time, PAC_MHW_cnt)
axs[0, 1].plot(time, PAC_MHW_cnt, '-', color='orangered', linewidth=2)
axs[0, 1].plot(time, ATL_MHW_cnt, '-', color='xkcd:mango', linewidth=2)
axs[0, 1].plot(time, IND_MHW_cnt, '-', color='gold', linewidth=2)
# axs[0, 1].plot(time, res_cnt.intercept + res_cnt.slope*time, 'k--')
ax2 = axs[0, 1].twinx()
ax2.plot(time, SAM, 'gray', linewidth=3, alpha=1)
ax2.fill_between(time, SAM-(SAM_error+1), SAM+(SAM_error+1),
    alpha=0.5, edgecolor='gray', facecolor='gray',
    linewidth=0, antialiased=True)
ax2.tick_params(length=10, direction='in', color='gray', axis='y', colors='gray')
ax2.spines['right'].set_color('gray')
ax2.yaxis.label.set_color('gray')
ax2.set_ylim(6, -6)

axs[0, 1].tick_params(length=10, direction='in')
axs[0, 1].set_title('b MHW frequency [number]')
axs[0, 1].set_xlim(1981, 2022)
# axs[0, 1].set_ylim(0.45, 2.5)
# axs[0, 1].grid(linestyle=':', linewidth=1)


#Duration
# res_dur = stats.linregress(time, PAC_MHW_dur)
axs[1, 0].plot(time, PAC_MHW_cnt, '-', color='orangered', linewidth=2)
axs[1, 0].plot(time, ATL_MHW_cnt, '-', color='xkcd:mango', linewidth=2)
axs[1, 0].plot(time, IND_MHW_cnt, '-', color='gold', linewidth=2)
# axs[1, 0].plot(time, res_dur.intercept + res_dur.slope*time, 'k--')
ax2 = axs[1, 0].twinx()
ax2.plot(time, SAM, 'gray', linewidth=3, alpha=1)
ax2.fill_between(time, SAM-(SAM_error+1), SAM+(SAM_error+1),
    alpha=0.5, edgecolor='gray', facecolor='gray',
    linewidth=0, antialiased=True)
ax2.tick_params(length=10, direction='in', color='gray', axis='y', colors='gray')
ax2.spines['right'].set_color('gray')
ax2.yaxis.label.set_color('gray')
ax2.set_ylim(6, -6)

axs[1, 0].tick_params(length=10, direction='in')
axs[1, 0].set_title('c MHW duration [days]')
axs[1, 0].set_xlim(1981, 2022)
# axs[1, 0].set_ylim(8, 16.5)
# axs[1, 0].grid(linestyle=':', linewidth=1)


#Cumulative Intensity
# res_cum = stats.linregress(time, PAC_MHW_cum)
axs[1, 1].plot(time, PAC_MHW_cum, '-', color='orangered', linewidth=2)
axs[1, 1].plot(time, ATL_MHW_cum, '-', color='xkcd:mango', linewidth=2)
axs[1, 1].plot(time, IND_MHW_cum, '-', color='gold', linewidth=2)
# axs[1, 1].plot(time, res_cum.intercept + res_cum.slope*time, 'k--')
ax2 = axs[1, 1].twinx()
ax2.plot(time, SAM, 'gray', linewidth=3, alpha=1)
ax2.fill_between(time, SAM-(SAM_error+1), SAM+(SAM_error+1),
    alpha=0.5, edgecolor='gray', facecolor='gray',
    linewidth=0, antialiased=True)
ax2.tick_params(length=10, direction='in', color='gray', axis='y', colors='gray')
ax2.spines['right'].set_color('gray')
ax2.yaxis.label.set_color('gray')
ax2.set_ylim(6, -6)

axs[1, 1].tick_params(length=10, direction='in')
axs[1, 1].set_title('d MHW Cumulative intensity [$^\circ$C·days]')
axs[1, 1].set_xlim(1981, 2022)
# axs[1, 1].set_ylim(10, 30)
# axs[1, 1].grid(linestyle=':', linewidth=1)


#Total Annual MHW Days
# res_td = stats.linregress(time, PAC_MHW_td)
axs[2, 0].plot(time, PAC_MHW_td, '-', color='orangered', linewidth=2)
axs[2, 0].plot(time, ATL_MHW_td, '-', color='xkcd:mango', linewidth=2)
axs[2, 0].plot(time, IND_MHW_td, '-', color='gold', linewidth=2)
# axs[2, 0].plot(time, res_td.intercept + res_td.slope*time, 'k--')
ax2 = axs[2, 0].twinx()
ax2.plot(time, SAM, 'gray', linewidth=3, alpha=1)
ax2.fill_between(time, SAM-(SAM_error+1), SAM+(SAM_error+1),
    alpha=0.5, edgecolor='gray', facecolor='gray',
    linewidth=0, antialiased=True)
ax2.tick_params(length=10, direction='in', color='gray', axis='y', colors='gray')
ax2.spines['right'].set_color('gray')
ax2.yaxis.label.set_color('gray')
ax2.set_ylim(6, -6)

axs[2, 0].tick_params(length=10, direction='in')
axs[2, 0].set_title('e MHW Total annual MHW days [days]')
axs[2, 0].set_xlim(1981, 2022)
# axs[2, 0].set_ylim(5, 40)
# axs[2, 0].grid(linestyle=':', linewidth=1)


#Areal Coverage
# res_area = stats.linregress(time, MHW_PAC_Area)
axs[2, 1].plot(time, MHW_PAC_Area, '-', color='orangered', linewidth=2)
axs[2, 1].plot(time, MHW_ATL_Area, '-', color='xkcd:mango', linewidth=2)
axs[2, 1].plot(time, MHW_IND_Area, '-', color='gold', linewidth=2)
# axs[2, 1].plot(time, res_area.intercept + res_area.slope*time, 'k--')
ax2 = axs[2, 1].twinx()
ax2.plot(time, SAM, 'gray', linewidth=3, alpha=1)
ax2.fill_between(time, SAM-(SAM_error+1), SAM+(SAM_error+1),
    alpha=0.5, edgecolor='gray', facecolor='gray',
    linewidth=0, antialiased=True)
ax2.tick_params(length=10, direction='in', color='gray', axis='y', colors='gray')
ax2.spines['right'].set_color('gray')
ax2.yaxis.label.set_color('gray')
ax2.set_ylim(6, -6)

axs[2, 1].tick_params(length=10, direction='in')
axs[2, 1].set_title('f MHW Area ratio [%]')
axs[2, 1].set_xlim(1981, 2022)
axs[2, 1].set_ylim(15, 95)
# axs[2, 1].grid(linestyle=':', linewidth=1)

fig.tight_layout(w_pad=1)


outfile = r'C:\Users\Manuel\Desktop\Figures_Paper_SO_MHW\Climate_Modes\SAM_MHWs.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


##############################################
###Correlations Climate Modes - MHW metrics###
##############################################

fig, axs = plt.subplots(3, 2, figsize=(15, 15))
plt.rcParams.update({'font.size': 22})
props = dict(boxstyle='round', facecolor='white', alpha=0.8)


#Max SSTA
res_Max_SSTA = stats.linregress(SAM_summer, Max_SSTA_1982_2021_ts)
axs[0, 0].scatter(SAM_summer, Max_SSTA_1982_2021_ts, s=100, c='k')
axs[0, 0].plot(SAM_summer, res_Max_SSTA.intercept + res_Max_SSTA.slope*SAM_summer, 'r')
axs[0, 0].set_xlabel('SAM')
axs[0, 0].set_ylabel('Maximum SSTA [$^\circ$C]')
axs[0, 0].tick_params(length=10, direction='in')
axs[0, 0].set_xlim(-4, 4)
textstr1 = '\n'.join((
    r'r = %.2f' % (res_Max_SSTA.rvalue, ),
    r'p < 0.05'))
textstr2 = r'A'
axs[0, 0].text(0.75, 0.95, textstr1, transform=axs[0, 0].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[0, 0].text(0.05, 0.95, textstr2, transform=axs[0, 0].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


#Frequency
res_freq = stats.linregress(SAM_summer, MHW_cnt)
axs[0, 1].scatter(SAM_summer, MHW_cnt, s=100, c='k')
axs[0, 1].plot(SAM_summer, res_freq.intercept + res_freq.slope*SAM_summer, 'r')
axs[0, 1].set_xlabel('SAM')
axs[0, 1].set_ylabel('Mean frequency [number]')
axs[0, 1].tick_params(length=10, direction='in')
axs[0, 1].set_xlim(-4, 4)
textstr1 = '\n'.join((
    r'r = %.2f' % (res_freq.rvalue, ),
    r'p < 0.05'))
textstr2 = r'B'
axs[0, 1].text(0.75, 0.95, textstr1, transform=axs[0, 1].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[0, 1].text(0.05, 0.95, textstr2, transform=axs[0, 1].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


#Duration
res_dur = stats.linregress(SAM_summer, MHW_dur)
axs[1, 0].scatter(SAM_summer, MHW_dur, s=100, c='k')
axs[1, 0].plot(SAM_summer, res_dur.intercept + res_dur.slope*SAM_summer, 'r')
axs[1, 0].set_xlabel('SAM')
axs[1, 0].set_ylabel('Mean duration [days]')
axs[1, 0].tick_params(length=10, direction='in')
axs[1, 0].set_xlim(-4, 4)
textstr1 = '\n'.join((
    r'r = %.2f' % (res_dur.rvalue, ),
    r'p < 0.05'))
textstr2 = r'C'
axs[1, 0].text(0.75, 0.95, textstr1, transform=axs[1, 0].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[1, 0].text(0.05, 0.95, textstr2, transform=axs[1, 0].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


#Cumulative Intensity
res_cum = stats.linregress(SAM_summer, MHW_cum)
axs[1, 1].scatter(SAM_summer, MHW_cum, s=100, c='k')
axs[1, 1].plot(SAM_summer, res_cum.intercept + res_cum.slope*SAM_summer, 'r')
axs[1, 1].set_xlabel('SAM')
axs[1, 1].set_ylabel('Cumulative intensity [$^\circ$C·days]')
axs[1, 1].tick_params(length=10, direction='in')
axs[1, 1].set_xlim(-4, 4)
textstr1 = '\n'.join((
    r'r = %.2f' % (res_cum.rvalue, ),
    r'p < 0.05'))
textstr2 = r'D'
axs[1, 1].text(0.75, 0.95, textstr1, transform=axs[1, 1].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[1, 1].text(0.05, 0.95, textstr2, transform=axs[1, 1].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


#Total Annual MHW Days
res_td = stats.linregress(SAM_summer, MHW_td)
axs[2, 0].scatter(SAM_summer, MHW_td, s=100, c='k')
axs[2, 0].plot(SAM_summer, res_td.intercept + res_td.slope*SAM_summer, 'r')
axs[2, 0].set_xlabel('SAM')
axs[2, 0].set_ylabel('Total annual MHW days [days]')
axs[2, 0].tick_params(length=10, direction='in')
axs[2, 0].set_xlim(-4, 4)
textstr1 = '\n'.join((
    r'r = %.2f' % (res_td.rvalue, ),
    r'p < 0.05'))
textstr2 = r'E'
axs[2, 0].text(0.75, 0.95, textstr1, transform=axs[2, 0].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[2, 0].text(0.05, 0.95, textstr2, transform=axs[2, 0].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


#Areal Coverage
res_area = stats.linregress(SAM_summer, MHW_Area)
axs[2, 1].scatter(SAM_summer, MHW_Area, s=100, c='k')
axs[2, 1].plot(SAM_summer, res_area.intercept + res_area.slope*SAM_summer, 'r')
axs[2, 1].set_xlabel('SAM')
axs[2, 1].set_ylabel('Area ratio [%]')
axs[2, 1].tick_params(length=10, direction='in')
axs[2, 1].set_xlim(-4, 4)
textstr1 = '\n'.join((
    r'r = %.2f' % (res_area.rvalue, ),
    r'p < 0.05'))
textstr2 = r'F'
axs[2, 1].text(0.75, 0.95, textstr1, transform=axs[2, 1].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[2, 1].text(0.05, 0.95, textstr2, transform=axs[2, 1].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


fig.tight_layout(w_pad=2)



##################################################
######Correlation Matrix of Climate Modes#########
##################################################

#DataFram containing timeseries of climate modes
Climate_Modes = pd.DataFrame({'PDOI':PDO, 'ONI':ENSO, 'TPI IPO':IPO, 'AMOI':AMO, 'TSAI':TSA, 'SOI':SOI, 'SAMI':SAM, 'AAOI':AAO})

Climate_Modes_MHWs = pd.DataFrame({'PDOI':PDO, 'ONI':ENSO, 'TPI IPO':IPO, 'AMOI':AMO, 'TSAI':TSA, 'SOI':SOI, 'SAMI':SAM, 'AAOI':AAO, 'Max SSTA':Max_SSTA_1982_2021_ts, 'MHW Frequency':MHW_cnt, 'MHW Duration':MHW_dur, 'MHW Cumulative Intensity': MHW_cum, 'MHW Total Annual days':MHW_td, 'MHW Area':MHW_Area, 'N-SAT Anomalies':SAT_Anom, 'SIC':SIC_ts, 'Mean SSTA':Mean_SSTA_1982_2021_ts})

SAT_Anom = (SAT_anom_nov + SAT_anom_dec + SAT_anom_jan + SAT_anom_feb + SAT_anom_mar) / 5 
SIC_ts = (SIC_nov + SIC_dec + SIC_jan + SIC_feb + SIC_mar) / 5 
Climate_MHWs = pd.DataFrame({'Max SSTA':Max_SSTA_1982_2021_ts, 'MHW Frequency':MHW_cnt, 'MHW Duration':MHW_dur, 'MHW Cumulative Intensity': MHW_cum, 'MHW Total Annual days':MHW_td, 'MHW Area':MHW_Area, 'N-SAT Anomalies':SAT_Anom, 'SIC':SIC_ts, 'Mean SSTA':Mean_SSTA_1982_2021_ts})



#Corr Matrix
corr_matrix = Climate_MHWs.corr()










###############################################################################
# Averaged climate variables along with climate modes + Correlation subplots ##
###############################################################################

time = np.arange(1982, 2022)
fig, axs = plt.subplots(3, 2, figsize=(20, 15))
plt.rcParams.update({'font.size': 24})
props = dict(boxstyle='round', facecolor='white', alpha=0.8)


######SSTA######
#Averaged timeseries
res_SSTA = stats.linregress(time, Mean_SSTA_1982_2021_ts)
axs[0, 0].plot(time, Mean_SSTA_1982_2021_ts, 'ok-', label='SSTA')
axs[0, 0].plot(time, res_SSTA.intercept + res_SSTA.slope*time, 'k--')
axs[0, 0].set_title('                                                                         Mean SSTA [$^\circ$C]', fontweight='bold')
ax2 = axs[0, 0].twinx()
ax2.plot(time, ENSO, 'r', linewidth=3, alpha=0.4, label='ONI')
ax2.tick_params(length=10, direction='in', color='r', axis='y', colors='r')
ax2.spines['right'].set_color('r')
ax2.yaxis.label.set_color('r')
axs[0, 0].set_ylabel('Mean SSTA [$^\circ$C]', labelpad=32)

axs[0, 0].tick_params(length=10, direction='in')
axs[0, 0].set_xlim(1981, 2022)
axs[0, 0].grid(linestyle=':', linewidth=1)

textstr1 = r'A'
axs[0, 0].text(0.05, 0.95, textstr1, transform=axs[0, 0].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)
# axs[0, 0].legend(loc='upper center', frameon=False, fontsize=25)
ax2.legend(loc='upper center', frameon=True, fontsize=25)


#Dispersion plot
res_SSTA = stats.linregress(ENSO, Mean_SSTA_1982_2021_ts)
axs[0, 1].scatter(ENSO, Mean_SSTA_1982_2021_ts, s=100, c='k')
axs[0, 1].plot(ENSO, res_SSTA.intercept + res_SSTA.slope*ENSO, 'r')
axs[0, 1].set_xlabel('ONI')
# axs[0, 1].set_ylabel('Mean SSTA')
axs[0, 1].tick_params(length=10, direction='in')
textstr1 = '\n'.join((
    r'r = %.2f' % (res_SSTA.rvalue, ),
    r'p < 0.05'))
textstr2 = r'B'
axs[0, 1].text(0.8, 0.95, textstr1, transform=axs[0, 1].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[0, 1].text(0.05, 0.95, textstr2, transform=axs[0, 1].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


######N-SAT Anomalies######
#Averaged timeseries
res_NSAT_Anom = stats.linregress(time, SAT_Anom)
axs[1, 0].plot(time, SAT_Anom, 'ok-', label='N-SAT Anomalies')
axs[1, 0].plot(time, res_NSAT_Anom.intercept + res_NSAT_Anom.slope*time, 'k--')
axs[1, 0].set_title('                                                                         Averaged N-SAT Anomalies [$^\circ$C]', fontweight='bold')
ax2 = axs[1, 0].twinx()
ax2.plot(time, ENSO, 'r', linewidth=3, alpha=0.4, label='ONI')
ax2.tick_params(length=10, direction='in', color='r', axis='y', colors='r')
ax2.spines['right'].set_color('r')
ax2.yaxis.label.set_color('r')
axs[1, 0].set_ylabel('N-SAT Anomalies [$^\circ$C]')

axs[1, 0].tick_params(length=10, direction='in')
axs[1, 0].set_xlim(1981, 2022)
axs[1, 0].grid(linestyle=':', linewidth=1)

textstr1 = r'C'
axs[1, 0].text(0.05, 0.95, textstr1, transform=axs[1, 0].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)
# axs[0, 0].legend(loc='upper center', frameon=False, fontsize=25)
# ax2.legend(loc='upper center', frameon=True, fontsize=25)


#Dispersion plot
res_NSAT_Anom = stats.linregress(ENSO, SAT_Anom)
axs[1, 1].scatter(ENSO, SAT_Anom, s=100, c='k')
axs[1, 1].plot(ENSO, res_NSAT_Anom.intercept + res_NSAT_Anom.slope*ENSO, 'r')
axs[1, 1].set_xlabel('ONI')
axs[1, 1].tick_params(length=10, direction='in')
# axs[0, 1].set_xlim(-1.5, 1.5)
textstr1 = '\n'.join((
    r'r = %.2f' % (res_NSAT_Anom.rvalue, ),
    r'p < 0.05'))
textstr2 = r'D'
axs[1, 1].text(0.8, 0.95, textstr1, transform=axs[1, 1].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[1, 1].text(0.05, 0.95, textstr2, transform=axs[1, 1].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


######SIC######
#Averaged timeseries
res_SIC = stats.linregress(time, SIC_ts)
axs[2, 0].plot(time, SIC_ts, 'ok-', label='SIC')
axs[2, 0].plot(time, res_SIC.intercept + res_SIC.slope*time, 'k--')
axs[2, 0].set_title('                                                                         Averaged Sea Ice Concentrations [%]', fontweight='bold')
ax2 = axs[2, 0].twinx()
ax2.plot(time, ENSO, 'r', linewidth=3, alpha=0.4, label='ONI')
ax2.tick_params(length=10, direction='in', color='r', axis='y', colors='r')
ax2.spines['right'].set_color('r')
ax2.yaxis.label.set_color('r')
axs[2, 0].set_ylabel('SIC [%]')

axs[2, 0].tick_params(length=10, direction='in')
axs[2, 0].set_xlim(1981, 2022)
axs[2, 0].grid(linestyle=':', linewidth=1)

textstr1 = r'C'
axs[2, 0].text(0.05, 0.95, textstr1, transform=axs[2, 0].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)
# axs[0, 0].legend(loc='upper center', frameon=False, fontsize=25)
# ax2.legend(loc='upper center', frameon=True, fontsize=25)


#Dispersion plot
res_SIC = stats.linregress(ENSO, SIC_ts)
axs[2, 1].scatter(ENSO, SIC_ts, s=100, c='k')
axs[2, 1].plot(ENSO, res_SIC.intercept + res_SIC.slope*ENSO, 'r')
axs[2, 1].set_xlabel('ONI')
axs[2, 1].tick_params(length=10, direction='in')
# axs[0, 1].set_xlim(-1.5, 1.5)
textstr1 = '\n'.join((
    r'r = %.2f' % (res_SIC.rvalue, ),
    r'p < 0.05'))
textstr2 = r'D'
axs[2, 1].text(0.8, 0.95, textstr1, transform=axs[2, 1].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
axs[2, 1].text(0.05, 0.95, textstr2, transform=axs[2, 1].transAxes, fontsize=24,
        fontweight='bold', verticalalignment='top', bbox=props)


fig.tight_layout(w_pad=2)















