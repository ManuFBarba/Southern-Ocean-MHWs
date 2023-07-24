# -*- coding: utf-8 -*-
"""
############################ Merge many .nc files into a single file ######################
"""


import netCDF4
import numpy as np
import xarray as xr
import netCDF4 as nc

ds = xr.open_mfdataset(r'D:\ICMAN-CSIC\MHW_ANT\datasets\SST_bimensual/*.nc', parallel=True)

ds.to_netcdf(r'D:\ICMAN-CSIC\MHW_ANT\datasets/SST_ANT_1982-2021.nc')


ds1 = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets\monthly_SST_ANT\SST_ANT_2021-01-01_2021-03-31.nc')
ds2 = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets\monthly_SST_ANT\SST_ANT_2021-04-01_2021-06-30.nc')
ds3 = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets\monthly_SST_ANT\SST_ANT_2021-07-01_2021-09-30.nc')
ds4 = xr.open_dataset(r'C:\ICMAN-CSIC\MHW_ANT\datasets\monthly_SST_ANT\SST_ANT_2021-10-01_2021-12-31.nc')



#ds = xr.merge([ds1,ds2,ds3,ds4])
#ds5.to_netcdf(r'C:\ICMAN-CSIC\MHW_ANT\datasets\1982_sst.nc')



from functools import reduce

lista = [ds1,ds2,ds3,ds4]

ds = reduce(lambda x,y: xr.merge([x,y]),lista)


ds.to_netcdf(r'C:\ICMAN-CSIC\MHW_ANT\datasets\Yearly_SST\SST_ANT_2021.nc')




    
    
    
