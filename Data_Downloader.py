# -*- coding: utf-8 -*-
"""

##############################  Data Downloader  ##############################

"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import seaborn as sns
import datetime
import getpass
import motuclient
import os


sns.set()

class MotuOptions:
    def __init__(self, attrs: dict):
        super(MotuOptions, self).__setattr__("attrs", attrs)

    def __setattr__(self, k, v):
        self.attrs[k] = v

    def __getattr__(self, k):
        try:
            return self.attrs[k]
        except KeyError:
            return None
        
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Copernicus Marine User Support Team"
__copyright__ = "(C) 2021 E.U. Copernicus Marine Service Information"
__credits__ = ["E.U. Copernicus Marine Service Information"]
__license__ = "MIT License - You must cite this source"
__version__ = "202105"
__maintainer__ = "D. Bazin, E. DiMedio, J. Cedillovalcarce, C. Giordan"
__email__ = "servicedesk dot cmems at mercator hyphen ocean dot eu"

def motu_option_parser(script_template, usr, pwd, output_filename):
    dictionary = dict(
        [e.strip().partition(" ")[::2] for e in script_template.split('--')])
    dictionary['variable'] = [value for (var, value) in [e.strip().partition(" ")[::2] for e in script_template.split('--')] if var == 'variable']  # pylint: disable=line-too-long
    for k, v in list(dictionary.items()):
        if v == '<OUTPUT_DIRECTORY>':
            dictionary[k] = '.'
        if v == '<OUTPUT_FILENAME>':
            dictionary[k] = output_filename
        if v == '<USERNAME>':
            dictionary[k] = usr
        if v == '<PASSWORD>':
            dictionary[k] = pwd
        if k in ['longitude-min', 'longitude-max', 'latitude-min', 
                 'latitude-max', 'depth-min', 'depth-max']:
            dictionary[k] = float(v)
        if k in ['date-min', 'date-max']:
            dictionary[k] = v[1:-1]
        dictionary[k.replace('-','_')] = dictionary.pop(k)
    dictionary.pop('python')
    dictionary['auth_mode'] = 'cas'
    return dictionary

USERNAME = '*******'
PASSWORD = '***********'
year = 1982
#year = 2017
month = 1
day = 1
for i in range(121):
#for i in range(31):
    
    #OUTPUT_FILENAME = 'SST_MED'  
    OUTPUT_FILENAME = 'SST_ANT'
    #OUTPUT_FILENAME = 'SST_SD_ANT'
    #OUTPUT_FILENAME = 'SIC_ANT'
    #OUTPUT_FILENAME = 'CHL_ANT'
    
    #Downloading the script template for each of the products
    

    #SST Mediterraneo
    #script_template = 'python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id SST_MED_SST_L4_REP_OBSERVATIONS_010_021-TDS --product-id cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021 --longitude-min -5.5 --longitude-max 36.325 --latitude-min 30.125 --latitude-max 46.025 --date-min "1982-01-01 00:00:00" --date-max "2022-03-03 00:00:00" --variable analysed_sst --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>'
    
    #SST ANT
    script_template = 'python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id SST_GLO_SST_L4_REP_OBSERVATIONS_010_024-TDS --product-id C3S-GLO-SST-L4-REP-OBS-SST --longitude-min -180 --longitude-max 180 --latitude-min -90 --latitude-max -40 --date-min "2017-01-01 12:00:00" --date-max "2017-02-28 12:00:00" --variable analysed_sst --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>'
    
    #Sea Ice Concentration
    #script_template = 'python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id SST_GLO_SST_L4_REP_OBSERVATIONS_010_024-TDS --product-id C3S-GLO-SST-L4-REP-OBS-SST --longitude-min -180 --longitude-max 180 --latitude-min -90 --latitude-max -40 --date-min "2017-01-01 12:00:00" --date-max "2017-02-28 12:00:00" --variable sea_ice_fraction --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>'
    # script_template = 'python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id SST_GLO_SST_L4_REP_OBSERVATIONS_010_024-TDS --product-id ESACCI-GLO-SST-L4-REP-OBS-SST --longitude-min -180 --longitude-max 180 --latitude-min -90 --latitude-max -40 --date-min "1982-01-01 12:00:00" --date-max "1982-02-28 12:00:00" --variable sea_ice_fraction --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>'
    
    #CHL ANT
    # script_template = 'python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id OCEANCOLOUR_GLO_BGC_L4_MY_009_104-TDS --product-id cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M --longitude-min -180 --longitude-max 180 --latitude-min -90 --latitude-max -40 --date-min "1998-01-01 00:00:00" --date-max "2021-12-01 23:59:59" --variable CHL --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>'
    
    #Providing the filenames, username, password and each template script to the parser
    data_request_options_dict_automated = motu_option_parser(script_template, USERNAME, PASSWORD, OUTPUT_FILENAME)
    
    if month > 12:
        month = 1
        year += 1
    day = 1
    i_date = datetime.datetime(year=year, month=month, day=day, hour = 12)
    month += 3
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        if month == 2:
            day += 28
        elif month <= 7 and month%2 != 0:
            day += 30
        elif month > 7 and month%2 == 0:
            day += 30
        else:
            day += 29
    else:
        if month == 2:
            day += 27
        elif month <= 7 and month%2 != 0:
            day += 30
        elif month > 7 and month%2 == 0:
            day += 30
        else:
            day += 29
    f_date = datetime.datetime(year=year, month=month, day=day,hour = 12)
    month += 1

    i_date_i = i_date.strftime("%Y-%m-%d %H:%M:%S")
    f_date_f = f_date.strftime("%Y-%m-%d %H:%M:%S")
    
    i_date_save = i_date.strftime("%Y-%m-%d")
    f_date_save = f_date.strftime("%Y-%m-%d")
    
    #Current day dictionary
    interval = {'date_max' : f_date_f,
                'date_min' : i_date_i,
                'out_name' : OUTPUT_FILENAME + '_' + i_date_save + '_' + f_date_save + '.nc',
                'out_dir' : r'E:\ICMAN-CSIC\MHWs\Datos_SST'
              }
    
    #Updating the request dictionary
    data_request_options_dict_automated.update(interval)
    OUTPUT_FILENAME = interval['out_name']
    print(data_request_options_dict_automated)
    
    #Requesting data
    motuclient.motu_api.execute_request(MotuOptions(data_request_options_dict_automated))
    print('Finished downloading .nc file for {}'.format(OUTPUT_FILENAME))
    