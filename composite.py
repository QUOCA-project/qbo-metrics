import xarray as xr
from matplotlib import pyplot as plt

# Script for computing QBO composite using QBOi-inspired method.

data_dir = "/store/atmos-adk33/cwp29/era5/"
data_file = "era5_*_daily_uvwT.nc"

ds = xr.open_mfdataset(data_dir + data_file, combine='by_coords')
ds = ds.drop_dims('s')

# Monthly mean dataset
ds_monthly = ds.resample(time='1M').mean(dim='time')

# Ensure variable names fit cf-conventions.
ds_monthly = ds_monthly.rename({#'u_component_of_wind': 'u', 
                                #'v_component_of_wind': 'v', 
                                #'temperature': 'T',
                                'lat': 'latitude'})

print(ds_monthly)

# Define reference timeseries: deseasonalised, linear trend removed, 
# 5-month running mean of monthly mean zonal wind at 30 hPa, 
# averaged 5S-5N.

# For later calculations, another timeseries is needed with linear 
# interpolation in time to allow calculation of the QBO period with daily
# accuracy.

# Q: deasonalise/detrend etc before or after averaging over latitude?

if 30 not in ds_monthly['pres']:
    print(" 30 hPa level not found in dataset: interpolating.")
    timeseries = ds_monthly['u'].interp(pres=30).sel(latitude=slice(-5, 5)).mean(dim='latitude')
else:
    timeseries = ds_monthly['u'].sel(pres=30, latitude=slice(-5, 5)).mean(dim='latitude')

deseason_timeseries = timeseries.groupby('time.month') - timeseries.groupby('time.month').mean(dim='time')

trend = deseason_timeseries.polyfit(dim='time', deg=1)
print(trend)

detrended_timeseries = deseason_timeseries - xr.polyval(deseason_timeseries['time'], trend.polyfit_coefficients.sel(degree=1))