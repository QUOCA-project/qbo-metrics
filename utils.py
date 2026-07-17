"""Load, standardise and save QBO datasets."""

import numpy as np
import xarray as xr

# Accepted aliases for each variable/coordinate, mapped to the names used here.
# Variables are also matched on their CF standard_name attribute.
VAR_ALIASES = {
    "u": ["u", "ua", "U", "uwnd", "eastward_wind", "u_component_of_wind"],
    "T": ["T", "t", "ta", "temp", "temperature", "air_temperature"],
    "o3": ["o3", "O3", "ozone", "go3", "ozone_mass_mixing_ratio",
           "mole_fraction_of_ozone_in_air", "mole_fraction_of_o3_in_air",
           "mass_fraction_of_ozone_in_air"],
}
STANDARD_NAMES = {
    "eastward_wind": "u",
    "air_temperature": "T",
    "mole_fraction_of_ozone_in_air": "o3",
    "mole_fraction_of_o3_in_air": "o3",
    "mass_fraction_of_ozone_in_air": "o3",
}
COORD_ALIASES = {
    "time": ["time"],
    "pres": ["pres", "plev", "lev", "level", "pressure", "air_pressure", "p"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon"],
}


def load_data(source, variables=("u", "T"), lat_range=(-5, 5), level=30.0):
    """Open and standardise input data.

    ``source`` may be an xarray Dataset or any input accepted by
    ``xarray.open_mfdataset``. The result uses ``time``, ``pres`` and
    ``latitude`` coordinates, pressure in hPa and zonal-mean fields. Returns
    ``(ds_native, ds_monthly)``; ``ds_native`` is ``None`` for monthly input.
    """
    ds = source if isinstance(source, xr.Dataset) else xr.open_mfdataset(source, combine="by_coords")

    # Standardise coordinate names.
    for cf_name, aliases in COORD_ALIASES.items():
        found = [a for a in aliases if a in ds.dims or a in ds.coords]
        if found and found[0] != cf_name:
            ds = ds.rename({found[0]: cf_name})

    # Standardise variable names, matching on name or CF standard_name.
    for var in list(ds.data_vars):
        std = STANDARD_NAMES.get(ds[var].attrs.get("standard_name"))
        target = std or next((k for k, v in VAR_ALIASES.items() if var in v), None)
        if target and target != var:
            ds = ds.rename({var: target})

    missing = [v for v in variables if v not in ds]
    if missing:
        raise ValueError(f"Required variable(s) {missing} not found; dataset has {list(ds.data_vars)}")
    for dim in ("time", "pres", "latitude"):
        if dim not in ds.dims:
            raise ValueError(f"Required dimension '{dim}' not found; dataset has {list(ds.dims)}")
    ds = ds[list(variables)]

    if "longitude" in ds.dims:
        ds = ds.mean(dim="longitude")

    # Pressure to hPa (via units attribute, or magnitude as a fallback).
    if ds["pres"].attrs.get("units", "").lower() in ("pa", "pascal") or ds["pres"].max() > 2000:
        ds = ds.assign_coords(pres=ds["pres"] / 100)
        ds["pres"].attrs["units"] = "hPa"
    ds = ds.sortby("latitude").sortby("pres")

    if not (ds["pres"].min() <= level <= ds["pres"].max()):
        raise ValueError(f"Pressure range {float(ds['pres'].min())}-{float(ds['pres'].max())} hPa "
                         f"does not span the {level} hPa reference level")
    if not (ds["latitude"].min() <= lat_range[0] and ds["latitude"].max() >= lat_range[1]):
        raise ValueError(f"Latitude range does not cover {lat_range[0]}-{lat_range[1]} degrees")

    # Identify the time frequency from the median time step.
    dt_days = float(np.median(np.diff(ds["time"].values)) / np.timedelta64(1, "D"))
    if dt_days < 27:
        ds_native = ds
        ds_monthly = ds.resample(time="MS").mean(dim="time")
    elif dt_days <= 32:
        ds_native = None
        ds_monthly = ds
    else:
        raise ValueError(f"Time step of ~{dt_days:.0f} days is coarser than monthly; cannot use this source")

    if ds_monthly.sizes["time"] < 36:
        raise ValueError(f"Only {ds_monthly.sizes['time']} months of data; too short to deseasonalise/detrend")

    return ds_native, ds_monthly


def save_composite(comp, path):
    """Save a composite Dataset as NetCDF."""
    comp.to_netcdf(path)


def load_composite(path):
    """Load a composite with ``lag``, ``pres`` and ``latitude`` dimensions."""
    comp = xr.open_dataset(path)
    missing = [d for d in ("lag", "pres", "latitude") if d not in comp.dims]
    if missing:
        raise ValueError(f"{path} is not a composite file: missing dimension(s) {missing}")
    return comp
