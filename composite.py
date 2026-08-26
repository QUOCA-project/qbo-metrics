"""Construct phase-aligned QBO cycle composites.

Onsets are zero crossings of the smoothed, deseasonalised and detrended
30-hPa zonal wind averaged over 5S--5N. Monthly event fields are
deseasonalised and averaged about each onset.
"""

import numpy as np
import xarray as xr

from plotting import plot_reference
from utils import load_data, save_composite

data_loc = "/store/atmos-adk33/cwp29/era5/era5_*_daily_uvwT.nc"

DAYS_PER_MONTH = 365.25 / 12


def _remove_monthly_climatology(obj):
    """Subtract the calendar-month climatology along ``time``."""
    climatology = obj.groupby("time.month").mean(dim="time")
    if isinstance(climatology, xr.Dataset):
        is_chunked = any(var.chunks is not None
                         for var in climatology.data_vars.values())
    else:
        is_chunked = climatology.chunks is not None
    if is_chunked:
        # Groupby creates one chunk per calendar month. Selecting those chunks
        # in chronological order repeatedly multiplies the Dask graph by the
        # number of years, so combine the 12 small chunks before indexing.
        climatology = climatology.chunk({"month": -1})
    climatology_by_time = climatology.sel(month=obj["time"].dt.month)
    return obj - climatology_by_time.drop_vars("month")


def _remove_linear_trend(obj):
    """Remove the linear trend from each time-dependent variable."""
    if isinstance(obj, xr.DataArray):
        trend = obj.polyfit(dim="time", deg=1)
        result = obj - xr.polyval(obj["time"], trend.polyfit_coefficients)
        result.attrs = obj.attrs.copy()
        return result

    result = obj.copy()
    for name, variable in obj.data_vars.items():
        if "time" in variable.dims:
            result[name] = _remove_linear_trend(variable)
    return result


def _remove_daily_climatology(series, smooth_days=31):
    """Subtract a cyclic, smoothed calendar-day climatology.

    Month-day indexing preserves leap-year alignment. The 29 February value
    is the mean of 28 February and 1 March. ``smooth_days`` is a positive odd
    integer; the default is 31 days.
    """
    if not isinstance(smooth_days, (int, np.integer)) or smooth_days <= 0:
        raise ValueError("smooth_days must be a positive odd integer")
    if smooth_days % 2 == 0 or smooth_days > 365:
        raise ValueError("smooth_days must be odd and no greater than 365")
    if not isinstance(series, xr.DataArray) or series.dims != ("time",):
        raise ValueError("series must be a one-dimensional DataArray over 'time'")
    if series.sizes["time"] < 2:
        raise ValueError("At least two time samples are required")
    times = np.asarray(series["time"].values)
    dt_days = float(np.median(np.diff(times)) / np.timedelta64(1, "D"))
    if not np.isfinite(dt_days) or dt_days <= 0:
        raise ValueError("Time coordinates must be finite and strictly increasing")
    if dt_days > 1.5:
        raise ValueError(
            f"Time step of ~{dt_days:.1f} days is not daily or sub-daily"
        )
    daily = series.resample(time="1D").mean("time").load()
    if bool(daily.isnull().any()):
        raise ValueError("Daily series contains missing values or missing days")
    month_day_values = np.asarray(daily["time"].dt.strftime("%m-%d").values)
    non_leap = month_day_values != "02-29"
    base = daily.isel(time=np.flatnonzero(non_leap))
    month_day = xr.DataArray(
        month_day_values[non_leap],
        dims="time",
        coords={"time": base["time"]},
        name="month_day",
    )
    climatology = base.groupby(month_day).mean("time").sortby("month_day")
    expected = np.array(
        [
            f"{month:02d}-{day:02d}"
            for month, n_days in enumerate(
                (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), start=1
            )
            for day in range(1, n_days + 1)
        ]
    )
    if not np.array_equal(np.asarray(climatology["month_day"].values), expected):
        raise ValueError("Daily series must contain every calendar day except February 29")

    values = np.asarray(climatology.values, dtype=float)
    if smooth_days == 1:
        smoothed = values
    else:
        half_window = smooth_days // 2
        padded = np.concatenate(
            (values[-half_window:], values, values[:half_window])
        )
        smoothed = np.convolve(
            padded, np.ones(smooth_days) / smooth_days, mode="valid"
        )
    climatology = xr.DataArray(
        smoothed,
        dims="month_day",
        coords={"month_day": expected},
        attrs=series.attrs,
    )
    february_29 = (
        0.5
        * (
            climatology.sel(month_day="02-28")
            + climatology.sel(month_day="03-01")
        )
    ).expand_dims(month_day=["02-29"])
    climatology = xr.concat((climatology, february_29), dim="month_day").sortby(
        "month_day"
    )
    month_day = xr.DataArray(
        month_day_values,
        dims="time",
        coords={"time": daily["time"]},
        name="month_day",
    )
    climatology_by_time = climatology.sel(month_day=month_day)
    anomaly = daily - climatology_by_time.drop_vars("month_day")
    anomaly.attrs = series.attrs.copy()
    return anomaly


def reference_timeseries(ds_monthly, level=30.0, lat_range=(-5, 5), smooth=5):
    """Return the monthly QBO onset-reference series.

    Zonal wind at ``level`` is averaged over ``lat_range``, deseasonalised,
    linearly detrended and smoothed with a centred ``smooth``-month mean.
    """
    u = ds_monthly["u"]
    u = u.sel(pres=level) if level in u["pres"] else u.interp(pres=level)
    ts = u.sel(latitude=slice(*lat_range)).mean(dim="latitude")

    ts = _remove_monthly_climatology(ts)
    trend = ts.polyfit(dim="time", deg=1)
    ts = ts - xr.polyval(ts["time"], trend.polyfit_coefficients)
    return ts.rolling(time=smooth, center=True).mean().dropna(dim="time")


def daily_reference_timeseries(
    ds_native,
    level=30.0,
    lat_range=(-5, 5),
    smooth_months=5.0,
    climatology_smooth_days=31,
):
    """Return the daily QBO onset-reference series.

    Daily zonal wind at ``level`` is averaged over ``lat_range``,
    deseasonalised, linearly detrended and smoothed with a centred
    ``smooth_months`` window. The default window is 152 days.
    """
    if not isinstance(ds_native, xr.Dataset) or "u" not in ds_native:
        raise ValueError("ds_native must be a Dataset containing zonal wind 'u'")
    if smooth_months <= 0:
        raise ValueError("smooth_months must be positive")
    u = ds_native["u"]
    if "pres" not in u.dims or "latitude" not in u.dims or "time" not in u.dims:
        raise ValueError("zonal wind must have time, pres and latitude dimensions")
    if not float(u["pres"].min()) <= level <= float(u["pres"].max()):
        raise ValueError(f"Pressure coordinate does not span {level} hPa")
    if not (float(u["latitude"].min()) <= lat_range[0] and
            float(u["latitude"].max()) >= lat_range[1]):
        raise ValueError(f"Latitude coordinate does not span {lat_range}")
    u = u.sel(pres=level) if level in u["pres"] else u.interp(pres=level)
    series = u.sel(latitude=slice(*lat_range)).mean("latitude")
    return _daily_reference_from_series(
        series,
        smooth_months=smooth_months,
        climatology_smooth_days=climatology_smooth_days,
    )


def _daily_reference_from_series(
    series,
    smooth_months=5.0,
    climatology_smooth_days=31,
):
    """Apply daily onset-reference processing to a selected wind series."""
    if smooth_months <= 0:
        raise ValueError("smooth_months must be positive")
    series = _remove_daily_climatology(
        series, smooth_days=climatology_smooth_days
    )
    trend = series.polyfit(dim="time", deg=1)
    series = series - xr.polyval(series["time"], trend.polyfit_coefficients)
    smooth_days = int(round(smooth_months * DAYS_PER_MONTH))
    result = (
        series.rolling(time=smooth_days, center=True)
        .mean()
        .dropna(dim="time")
    )
    result.name = "reference"
    return result


def reference_dates(ref, direction="westerly", merge_months=5,
                    return_groups=False, merge_days=None):
    """Return linearly interpolated zero-crossing dates from ``ref``.

    ``direction="westerly"`` selects negative-to-positive crossings;
    ``"easterly"`` selects positive-to-negative crossings. Crossings within
    the merge tolerance are represented by their midpoint. ``merge_days``
    sets a daily tolerance and takes precedence over ``merge_months``.

    ``return_groups=True`` also returns the original crossings represented by
    each retained date.
    """
    vals = ref.values if direction == "westerly" else -ref.values
    times = ref["time"].values.astype("datetime64[ns]")

    (idx,) = np.nonzero((vals[:-1] < 0) & (vals[1:] >= 0))
    frac = -vals[idx] / (vals[idx + 1] - vals[idx])
    dates = times[idx] + frac * (times[idx + 1] - times[idx])

    merged, groups = [], []
    if merge_days is None:
        merge_days = int(merge_months * DAYS_PER_MONTH)
    if merge_days < 0:
        raise ValueError("merge tolerance must be non-negative")
    tol = np.timedelta64(int(merge_days), "D")
    for d in dates:
        if merged and d - merged[-1] < tol:
            merged[-1] = merged[-1] + (d - merged[-1]) / 2
            groups[-1].append(d)
        else:
            merged.append(d)
            groups.append([d])
    merged = np.array(merged)
    if return_groups:
        return merged, tuple(np.asarray(group) for group in groups)
    return merged


def _extract_events_at_nearest_samples(obj, dates, window):
    """Extract complete cycle windows aligned to the nearest onset samples."""
    times = np.asarray(obj["time"].values)
    events, event_dates = [], []
    for date in dates:
        i = int(np.argmin(np.abs(times - date)))
        if i - window < 0 or i + window >= len(times):
            continue
        seg = obj.isel(time=slice(i - window, i + window + 1))
        event_time = np.asarray(seg["time"].values)
        seg = seg.rename(time="lag").assign_coords(
            lag=np.arange(-window, window + 1),
            event_time=("lag", event_time),
        )
        events.append(seg)
        event_dates.append(date)
    if not events:
        raise ValueError("No reference dates have a complete window within the record")
    event_coord = xr.DataArray(np.asarray(event_dates), dims="event", name="event")
    return xr.concat(events, dim=event_coord)


def _exclude_onset_years(dates, exclude_years):
    """Return onset dates outside the requested calendar years."""
    dates = np.asarray(dates)
    try:
        if exclude_years is None:
            values = []
        elif np.isscalar(exclude_years):
            values = [exclude_years]
        else:
            values = list(exclude_years)
        excluded = [int(year) for year in values]
        if any(float(year) != excluded[index]
               for index, year in enumerate(values)):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("exclude_years must contain calendar years") from exc
    if excluded:
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        dates = dates[~np.isin(years, excluded)]
    return dates, excluded


def extract_events(ds_monthly, dates, window=15, deseasonalise=True,
                   detrend=False, exclude_years=None):
    """Return complete monthly QBO-cycle windows centred on onset ``dates``.

    Onsets are aligned to the nearest monthly sample. The result has
    ``(event, lag, ...)`` dimensions and an ``event_time`` coordinate.
    ``deseasonalise=True`` removes the calendar-month climatology.
    ``detrend=True`` also removes each field's linear trend before extracting
    events. ``exclude_years`` omits events whose onset falls in those years.
    """
    if deseasonalise:
        ds_monthly = _remove_monthly_climatology(ds_monthly)
    if detrend:
        ds_monthly = _remove_linear_trend(ds_monthly)

    dates, excluded = _exclude_onset_years(dates, exclude_years)

    events = _extract_events_at_nearest_samples(ds_monthly, dates, window)
    events.attrs["detrended"] = int(detrend)
    events.attrs["excluded_years"] = ",".join(map(str, excluded))
    return events


def compute_composite(ds_monthly, dates, window=15, deseasonalise=True,
                      detrend=False, exclude_years=None):
    """Return the mean of complete QBO-cycle windows centred on onset ``dates``.

    The result retains pressure and latitude, with monthly lags from
    ``-window`` to ``+window``. ``deseasonalise=True`` removes the
    calendar-month climatology. ``detrend=True`` removes linear trends from
    the source fields. ``exclude_years`` omits onsets in those years.
    """
    events = extract_events(ds_monthly, dates, window=window,
                            deseasonalise=deseasonalise, detrend=detrend,
                            exclude_years=exclude_years)
    comp = events.mean(dim="event")
    comp.attrs.update(events.attrs)
    comp.attrs["n_events"] = events.sizes["event"]
    return comp



if __name__ == "__main__":
    ds_daily, ds_monthly = load_data(data_loc)

    ref = reference_timeseries(ds_monthly)
    dates = reference_dates(ref)
    print(f"Found {len(dates)} westerly-onset reference dates")
    plot_reference(ref, dates)

    comp = compute_composite(ds_monthly, dates)
    save_composite(comp, "qbo_composite.nc")
