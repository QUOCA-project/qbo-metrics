"""Calculate QBO diagnostics from composites and cycle windows."""

import numpy as np
import xarray as xr
from scipy import special
from scipy.optimize import curve_fit

from composite import (DAYS_PER_MONTH, _daily_reference_from_series,
                       _extract_events_at_nearest_samples,
                       _remove_daily_climatology, reference_dates)


def fit_sine(profile, period=None, period_guess=28.0):
    """Fit ``offset + amp*sin(2*pi*lag/period + phase)`` to ``profile``.

    ``period=None`` fits a continuous period from ``period_guess``; a supplied
    period is fixed. Period values use the lag coordinate's units. The result
    contains positive amplitude, phase in ``[-pi, pi)``, offset, period,
    input profile and fitted curve.
    """
    if period is not None and period <= 0:
        raise ValueError("period must be positive")
    if period_guess <= 0:
        raise ValueError("period_guess must be positive")

    lag = np.asarray(profile["lag"].values, dtype=float)
    y = np.asarray(profile.values, dtype=float)
    valid = np.isfinite(lag) & np.isfinite(y)
    if valid.sum() < 3:
        raise ValueError("At least three finite values are required to fit a sine wave")

    lag_valid, y_valid = lag[valid], y[valid]
    scale = np.std(y_valid)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Cannot fit a sine wave to a constant profile")
    y_scaled = y_valid / scale
    period0 = period_guess if period is None else period
    omega0 = 2 * np.pi / period0

    def model(x, amp, phase, offset, wave_period):
        return offset + amp * np.sin(2 * np.pi * x / wave_period + phase)

    # Initialise from the profile range and the lag of its maximum.
    offset0 = np.mean(y_scaled)
    amp0 = (np.max(y_scaled) - np.min(y_scaled)) / 2
    phase0 = np.pi / 2 - omega0 * lag_valid[np.argmax(y_scaled)]
    phase0 = (phase0 + np.pi) % (2 * np.pi) - np.pi
    if period is None:
        # Bound the period to resolvable values near the QBO timescale.
        step = np.median(np.diff(np.unique(np.sort(lag_valid))))
        period_bounds = (max(2 * step, period_guess / 2), period_guess * 2)
        popt, _ = curve_fit(model, lag_valid, y_scaled,
                            p0=(amp0, phase0, offset0, period_guess),
                            bounds=([-np.inf, -np.inf, -np.inf, period_bounds[0]],
                                    [np.inf, np.inf, np.inf, period_bounds[1]]))
        amp, phase, offset, period = popt
    else:
        def fixed_period_model(x, amp, phase, offset):
            return model(x, amp, phase, offset, period)

        popt, _ = curve_fit(fixed_period_model, lag_valid, y_scaled,
                            p0=(amp0, phase0, offset0))
        amp, phase, offset = popt

    # Use positive amplitude and one phase interval at every pressure.
    if amp < 0:
        amp = -amp
        phase += np.pi
    phase = (phase + np.pi) % (2 * np.pi) - np.pi

    fit = xr.DataArray(scale * model(lag, amp, phase, offset, period),
                       coords=profile.coords, dims=profile.dims,
                       name=profile.name, attrs=profile.attrs)
    return {"amp": scale * amp, "phase": phase, "offset": scale * offset,
            "period": period, "profile": profile, "fit": fit}


def daily_composite_period(
    ds_native,
    reference_level=30.0,
    reference_lat_range=(-5, 5),
    period_latitude=0.0,
    smooth_months=5.0,
    merge_months=5.0,
    window_months=15.0,
    climatology_smooth_days=31,
    period_guess=28.0,
):
    """Return the QBO period from daily 30-hPa onset composites.

    Nominal-month durations are converted with ``DAYS_PER_MONTH``. Onsets are
    defined from the daily reference series and aligned to the nearest daily
    sample. Complete equatorial-wind cycles are deseasonalised, averaged and
    fitted in days for each onset direction. Their mean is returned in days
    and months, together with the component fits, dates and cycle counts.
    """
    if not isinstance(ds_native, xr.Dataset) or "u" not in ds_native:
        raise ValueError("ds_native must be a Dataset containing zonal wind 'u'")
    if window_months <= 0:
        raise ValueError("window_months must be positive")
    if period_guess <= 0:
        raise ValueError("period_guess must be positive")

    u = ds_native["u"]
    required_dims = {"time", "pres", "latitude"}
    if not required_dims.issubset(u.dims):
        raise ValueError("zonal wind must have time, pres and latitude dimensions")
    u = u.sortby("pres").sortby("latitude")
    if not float(u["pres"].min()) <= reference_level <= float(u["pres"].max()):
        raise ValueError(f"Pressure coordinate does not span {reference_level} hPa")
    if not float(u["latitude"].min()) <= period_latitude <= float(u["latitude"].max()):
        raise ValueError(f"Latitude coordinate does not span {period_latitude} degrees")
    if not (float(u["latitude"].min()) <= reference_lat_range[0] and
            float(u["latitude"].max()) >= reference_lat_range[1]):
        raise ValueError(
            f"Latitude coordinate does not span {reference_lat_range}"
        )
    u = (u.sel(pres=reference_level) if reference_level in u["pres"]
         else u.interp(pres=reference_level))
    reference_raw = u.sel(latitude=slice(*reference_lat_range)).mean("latitude")
    period_raw = (u.sel(latitude=period_latitude)
                  if period_latitude in u["latitude"]
                  else u.interp(latitude=period_latitude))

    smooth_days = int(round(smooth_months * DAYS_PER_MONTH))
    merge_days = int(round(merge_months * DAYS_PER_MONTH))
    window_days = int(round(window_months * DAYS_PER_MONTH))
    period_guess_days = period_guess * DAYS_PER_MONTH
    reference = _daily_reference_from_series(
        reference_raw,
        smooth_months=smooth_months,
        climatology_smooth_days=climatology_smooth_days,
    )
    shear_types = ("easterly", "westerly")
    dates = {
        shear: reference_dates(
            reference, direction=shear, merge_months=merge_months,
            merge_days=merge_days,
        )
        for shear in shear_types
    }
    period_series = _remove_daily_climatology(
        period_raw, smooth_days=climatology_smooth_days
    )

    lag_days = np.arange(-window_days, window_days + 1)
    lag_months = lag_days / DAYS_PER_MONTH
    profiles, fits, fitted_period_days = [], [], []
    retained_by_shear, aligned_by_shear = [], []
    n_events = []

    for shear in shear_types:
        try:
            event_data = _extract_events_at_nearest_samples(
                period_series, dates[shear], window_days
            )
        except ValueError as exc:
            raise ValueError(
                f"No {shear} onsets have a complete daily cycle window"
            ) from exc
        profile = event_data.mean("event")
        fitted = fit_sine(profile, period_guess=period_guess_days)
        profiles.append(profile)
        fits.append(fitted["fit"])
        fitted_period_days.append(float(fitted["period"]))
        retained_by_shear.append(np.asarray(event_data["event"].values))
        aligned_by_shear.append(
            np.asarray(event_data["event_time"].sel(lag=0).values)
        )
        n_events.append(event_data.sizes["event"])

    period_days = np.asarray(fitted_period_days)
    common_period_days = float(np.mean(period_days))
    periods = period_days / DAYS_PER_MONTH
    common_period = common_period_days / DAYS_PER_MONTH

    def padded_datetimes(values_by_shear):
        width = max(len(values) for values in values_by_shear)
        values = np.full(
            (len(shear_types), width), np.datetime64("NaT"), dtype="datetime64[ns]"
        )
        for index, source in enumerate(values_by_shear):
            values[index, :len(source)] = np.asarray(source, dtype="datetime64[ns]")
        return values

    result = xr.Dataset(
        {
            "reference": reference,
            "onset_date": (("shear", "onset"), padded_datetimes(
                [dates[shear] for shear in shear_types]
            )),
            "event_date": (("shear", "event"), padded_datetimes(
                retained_by_shear
            )),
            "aligned_event_date": (("shear", "event"), padded_datetimes(
                aligned_by_shear
            )),
            "profile": (("shear", "lag"), np.stack([p.values for p in profiles])),
            "fit": (("shear", "lag"), np.stack([f.values for f in fits])),
            "period_by_composite": ("shear", periods),
            "period_days_by_composite": ("shear", period_days),
            "period": common_period,
            "period_days": common_period_days,
            "period_days_shear_difference": abs(period_days[0] - period_days[1]),
            "n_onsets": ("shear", [len(dates[shear]) for shear in shear_types]),
            "n_events": ("shear", n_events),
        },
        coords={
            "shear": list(shear_types),
            "lag": lag_days,
            "lag_month": ("lag", lag_months),
            "onset": np.arange(max(len(dates[shear]) for shear in shear_types)),
            "event": np.arange(max(n_events)),
        },
    )
    units = period_series.attrs.get("units")
    if units:
        result["profile"].attrs["units"] = units
        result["fit"].attrs["units"] = units
    for name in ("period", "period_by_composite"):
        result[name].attrs["units"] = "months"
    for name in ("period_days", "period_days_by_composite",
                 "period_days_shear_difference"):
        result[name].attrs["units"] = "days"
    result["lag"].attrs["units"] = "days"
    result["lag_month"].attrs["units"] = "months"
    return result


def vertical_extent(amplitude, fraction=0.1, boundary="top",
                    maximum_pres_range=None):
    """Return one ``fraction``-of-maximum amplitude boundary.

    ``boundary`` selects the top (lower-pressure) or bottom (higher-pressure)
    crossing relative to the amplitude maximum. Crossings are interpolated
    linearly in log pressure and are NaN when unbracketed.
    ``maximum_pres_range`` optionally limits the levels used to locate the
    maximum while retaining all levels for the crossing search.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    if boundary not in {"top", "bottom"}:
        raise ValueError("boundary must be 'top' or 'bottom'")
    amp = amplitude["amp"] if isinstance(amplitude, xr.Dataset) else amplitude
    if "pres" not in amp.dims or amp.ndim != 1:
        raise ValueError("amplitude must be one-dimensional over 'pres'")

    pres = np.asarray(amp["pres"].values, dtype=float)
    values = np.asarray(amp.values, dtype=float)
    valid = np.isfinite(pres) & np.isfinite(values) & (pres > 0)
    pres, values = pres[valid], values[valid]
    order = np.argsort(pres)
    pres, values = pres[order], values[order]
    extent = np.nan

    maximum_mask = np.ones(values.size, dtype=bool)
    if maximum_pres_range is not None:
        lower, upper = sorted(maximum_pres_range)
        maximum_mask = (pres >= lower) & (pres <= upper)

    if maximum_mask.any() and np.max(values[maximum_mask]) > 0:
        maximum_indices = np.flatnonzero(maximum_mask)
        i_max = int(maximum_indices[np.argmax(values[maximum_mask])])
        maximum = values[i_max]
        threshold = fraction * maximum
        if boundary == "top":
            crossings = np.flatnonzero(values[:i_max + 1] <= threshold)
            if crossings.size:
                lower_index = int(crossings[-1])
                upper_index = lower_index + 1
        else:
            crossings = np.flatnonzero(values[i_max:] <= threshold)
            if crossings.size:
                upper_index = int(i_max + crossings[0])
                lower_index = upper_index - 1

        if crossings.size:
            exact_index = (lower_index if values[lower_index] == threshold
                           else upper_index)
            if values[exact_index] == threshold:
                extent = pres[exact_index]
            elif 0 <= lower_index < upper_index < values.size:
                weight = ((threshold - values[lower_index]) /
                          (values[upper_index] - values[lower_index]))
                extent = np.exp(np.log(pres[lower_index]) + weight *
                                (np.log(pres[upper_index]) -
                                 np.log(pres[lower_index])))

    units = amp["pres"].attrs.get("units", "hPa")
    return xr.DataArray(extent, name=f"vertical_extent_{boundary}",
                        attrs={"units": units, "fraction": fraction})


def phase_amplitude(da, period=None, reference_pres=30.0,
                    period_guess=28.0, pres_range=(1, 200)):
    """Fit QBO period, amplitude and phase at each pressure.

    ``da`` is a ``(lag, pres)`` DataArray or a mapping with easterly- and
    westerly-onset composites. A mapping uses the mean of free period fits at
    ``reference_pres``; ``period`` supplies a fixed common value. The result
    contains amplitude, phase, offset, period, top and bottom vertical
    extents, profiles and fitted curves within ``pres_range``.
    """
    shear_types = ("easterly", "westerly")
    if hasattr(da, "keys"):
        if not set(shear_types).issubset(da):
            raise ValueError("A composite mapping must contain 'easterly' and 'westerly'")
        composites = {shear: da[shear] for shear in shear_types}
        fitted_periods = []
        if period is None:
            for source in composites.values():
                if "lag" not in source.dims or "pres" not in source.dims:
                    raise ValueError("Each composite must have 'lag' and 'pres' dimensions")
                if not float(source["pres"].min()) <= reference_pres <= float(source["pres"].max()):
                    raise ValueError(f"Each composite must span {reference_pres} hPa")
                ref = (source.sel(pres=reference_pres)
                       if reference_pres in source["pres"]
                       else source.interp(pres=reference_pres))
                fitted_periods.append(fit_sine(ref, period_guess=period_guess)["period"])
            period = float(np.mean(fitted_periods))
        results = [phase_amplitude(source, period=period,
                                   reference_pres=reference_pres,
                                   period_guess=period_guess,
                                   pres_range=pres_range)
                   for source in composites.values()]
        result = xr.concat(results,
                           dim=xr.IndexVariable("shear", list(shear_types)))
        result["period"] = xr.DataArray(period)
        if fitted_periods:
            result["period_by_composite"] = xr.DataArray(
                fitted_periods, dims="period_composite",
                coords={"period_composite": list(shear_types)})
        return result
    if not isinstance(da, xr.DataArray):
        raise ValueError("da must be a DataArray or an easterly/westerly mapping")
    if "lag" not in da.dims or "pres" not in da.dims:
        raise ValueError("da must have 'lag' and 'pres' dimensions")
    extra_dims = set(da.dims) - {"lag", "pres"}
    if extra_dims:
        raise ValueError(f"Select or average the additional dimension(s) {sorted(extra_dims)} first")
    fitted_periods = []
    if period is None:
        if not float(da["pres"].min()) <= reference_pres <= float(da["pres"].max()):
            raise ValueError(
                f"Pressure coordinate does not span the {reference_pres} hPa reference level"
            )
        ref = (da.sel(pres=reference_pres) if reference_pres in da["pres"]
               else da.interp(pres=reference_pres))
        fitted_periods = [fit_sine(ref, period_guess=period_guess)["period"]]
        period = float(fitted_periods[0])
    requested_range = None if pres_range is None else tuple(sorted(pres_range))
    da = da.sortby("pres").transpose("pres", "lag")

    n_pres = da.sizes["pres"]
    out = {"amp": np.full(n_pres, np.nan), "phase": np.full(n_pres, np.nan),
           "offset": np.full(n_pres, np.nan),
           "fit": np.full((n_pres, da.sizes["lag"]), np.nan)}
    for k in range(n_pres):
        f = fit_sine(da.isel(pres=k), period=period)
        for key in ("amp", "phase", "offset"):
            out[key][k] = f[key]
        out["fit"][k] = f["fit"].values

    result = xr.Dataset(
        {"amp": ("pres", out["amp"]), "phase": ("pres", out["phase"]),
         "offset": ("pres", out["offset"]),
         "period": period, "fit": (("pres", "lag"), out["fit"]),
         "profile": da},
        coords={"pres": da["pres"], "lag": da["lag"]},
        attrs={"reference_pres": reference_pres})
    if fitted_periods:
        result["period_by_composite"] = xr.DataArray(
            fitted_periods, dims="period_composite",
            coords={"period_composite": ["composite"]})
    maximum_pres_range = None
    if requested_range is not None:
        maximum_pres_range = (float(result["pres"].min()),
                              requested_range[1])
    result["vertical_extent_top"] = vertical_extent(
        result, boundary="top", maximum_pres_range=maximum_pres_range)
    result["vertical_extent_bottom"] = vertical_extent(
        result, boundary="bottom", maximum_pres_range=maximum_pres_range)
    if requested_range is not None:
        top, bottom = requested_range
        top_extent = float(result["vertical_extent_top"])
        bottom_extent = float(result["vertical_extent_bottom"])
        if np.isfinite(top_extent) and top_extent < top:
            # Retain the upper bracketing level for display.
            above = np.asarray(result["pres"].where(result["pres"] < top_extent,
                                                     drop=True).values)
            top = float(np.max(above)) if above.size else top_extent
        if np.isfinite(bottom_extent) and bottom_extent > bottom:
            # Retain the lower bracketing level for display.
            below = np.asarray(result["pres"].where(
                result["pres"] > bottom_extent, drop=True).values)
            bottom = float(np.min(below)) if below.size else bottom_extent
        result = result.sel(pres=slice(top, bottom))
    return result


def _crossing_line(values, coordinate, increasing, seed_index, seed_value,
                   monotonic=False):
    """Follow one connected zero crossing through a two-dimensional field."""
    candidates = []
    for profile in values:
        left, right = profile[:-1], profile[1:]
        finite = np.isfinite(left) & np.isfinite(right)
        if increasing:
            crossing = finite & (left <= 0) & (right >= 0) & (right > left)
        else:
            crossing = finite & (left >= 0) & (right <= 0) & (right < left)
        indices = np.flatnonzero(crossing)
        fraction = -left[indices] / (right[indices] - left[indices])
        positions = coordinate[indices] + fraction * (
            coordinate[indices + 1] - coordinate[indices]
        )
        candidates.append(np.unique(positions))

    available = [i for i, positions in enumerate(candidates) if positions.size]
    line = np.full(len(candidates), np.nan)
    if not available:
        return line
    start = min(available, key=lambda i: abs(i - seed_index))
    line[start] = min(candidates[start], key=lambda value: abs(value - seed_value))
    for indices, direction in (
        (range(start + 1, len(line)), 1),
        (range(start - 1, -1, -1), -1),
    ):
        previous = line[start]
        for i in indices:
            possible = candidates[i]
            if monotonic:
                possible = possible[direction * (possible - previous) > 0]
            if len(possible) == 0:
                break
            line[i] = min(possible, key=lambda value: abs(value - previous))
            previous = line[i]
    return line


def _descent_from_field(wind, lags, pressure, shear, reference_pres):
    """Return the tracked zero line and its descent rates for one wind field."""
    temporal_increase = shear == "westerly"
    zero_pressure = _crossing_line(
        wind, pressure, not temporal_increase,
        int(np.argmin(abs(lags))), reference_pres,
    )
    zero_lag = _crossing_line(
        wind.T, lags, temporal_increase,
        int(np.argmin(abs(pressure - reference_pres))), 0.0,
        monotonic=True,
    )
    connected = zero_lag[np.isfinite(zero_lag)]
    if connected.size:
        within_contour = (
            (lags >= np.floor(connected.min()))
            & (lags <= np.ceil(connected.max()))
        )
        zero_pressure[~within_contour] = np.nan

    monthly_rate = np.full(len(lags) - 1, np.nan)
    dt = np.diff(lags)
    dp = np.diff(zero_pressure)
    downward = np.isfinite(dp) & np.isfinite(dt) & (dt > 0) & (dp > 0)
    monthly_rate[downward] = dp[downward] / dt[downward]

    profile_rate = np.full(len(pressure), np.nan)
    valid = np.flatnonzero(np.isfinite(zero_lag))
    if valid.size >= 2:
        for k, i in enumerate(valid):
            lower = valid[max(k - 1, 0)]
            upper = valid[min(k + 1, valid.size - 1)]
            crossing_time = zero_lag[upper] - zero_lag[lower]
            if crossing_time > 0:
                profile_rate[i] = (
                    pressure[upper] - pressure[lower]
                ) / crossing_time
    return zero_pressure, zero_lag, monthly_rate, profile_rate


def descent_rate(da, cycles=None, pres_range=(1, 200), reference_pres=30.0,
                 lag_dim="lag"):
    """Return composite zero-wind descent rates in hPa per month.

    ``da`` is one ``(lag, pres)`` wind composite or an easterly/westerly
    mapping. The zero contour connected to the onset at ``reference_pres`` is
    linearly interpolated. Its displacement between monthly lags gives the
    mean rate; crossing lag as a function of pressure gives the pressure
    profile. Optional ``cycles`` provide cycle-to-cycle profile spread.
    """
    shear_types = np.array(["easterly", "westerly"])
    if isinstance(da, xr.DataArray):
        source_by_shear = {str(shear): da for shear in shear_types}
    elif hasattr(da, "keys") and set(shear_types).issubset(da):
        source_by_shear = {str(shear): da[str(shear)] for shear in shear_types}
    else:
        raise ValueError("da must be a DataArray or an easterly/westerly mapping")

    if cycles is None:
        cycles_by_shear = {}
    elif isinstance(cycles, xr.DataArray):
        cycles_by_shear = {str(shear): cycles for shear in shear_types}
    elif hasattr(cycles, "keys") and set(shear_types).issubset(cycles):
        cycles_by_shear = {str(shear): cycles[str(shear)] for shear in shear_types}
    else:
        raise ValueError("cycles must be a DataArray or an easterly/westerly mapping")

    selected = []
    for shear in shear_types:
        composite = source_by_shear[str(shear)]
        if lag_dim not in composite.dims or "pres" not in composite.dims:
            raise ValueError(f"Each composite must have '{lag_dim}' and 'pres' dimensions")
        extra_dims = set(composite.dims) - {lag_dim, "pres"}
        if extra_dims:
            raise ValueError(f"Select or average the additional dimension(s) {sorted(extra_dims)} first")
        selected.append(
            composite.sortby("pres")
            .sel(pres=slice(*sorted(pres_range)))
            .sortby(lag_dim)
            .transpose(lag_dim, "pres")
        )
    selected = xr.align(*selected, join="inner")

    lags = np.asarray(selected[0][lag_dim].values, dtype=float)
    pressure = np.asarray(selected[0]["pres"].values, dtype=float)
    if lags.size < 2 or not np.all(np.diff(lags) > 0):
        raise ValueError("At least two increasing lag values are required")
    if pressure.size < 2 or np.any(pressure <= 0):
        raise ValueError("At least two positive pressure levels are required")
    if not pressure.min() <= reference_pres <= pressure.max():
        raise ValueError("reference_pres must lie within the pressure range")

    zero_pressure = np.full((2, len(lags)), np.nan)
    zero_lag = np.full((2, len(pressure)), np.nan)
    monthly_rate = np.full((2, len(lags) - 1), np.nan)
    profile_rate = np.full((2, len(pressure)), np.nan)
    mean_rate = np.full(2, np.nan)
    for s, (shear, composite) in enumerate(zip(shear_types, selected)):
        diagnosed = _descent_from_field(
            np.asarray(composite.values, dtype=float), lags, pressure,
            str(shear), reference_pres,
        )
        zero_pressure[s], zero_lag[s], monthly_rate[s], profile_rate[s] = diagnosed
        finite = monthly_rate[s, np.isfinite(monthly_rate[s])]
        if finite.size:
            mean_rate[s] = np.mean(finite)

    cycle_profiles = []
    for shear in shear_types:
        if str(shear) not in cycles_by_shear:
            cycle_profiles.append(np.empty((0, len(pressure))))
            continue
        cycle_data = cycles_by_shear[str(shear)]
        required = {"event", lag_dim, "pres"}
        if not required.issubset(cycle_data.dims):
            raise ValueError(f"Each cycle array must have {sorted(required)} dimensions")
        extra_dims = set(cycle_data.dims) - required
        if extra_dims:
            raise ValueError(f"Select or average the additional dimension(s) {sorted(extra_dims)} first")
        cycle_data = (
            cycle_data.sortby("pres")
            .interp(pres=pressure)
            .sortby(lag_dim)
            .transpose("event", lag_dim, "pres")
        )
        cycle_lags = np.asarray(cycle_data[lag_dim].values, dtype=float)
        profiles = np.full((cycle_data.sizes["event"], len(pressure)), np.nan)
        for c, field in enumerate(np.asarray(cycle_data.values, dtype=float)):
            cycle_zero_pressure, _, rates, _ = _descent_from_field(
                field, cycle_lags, pressure, str(shear), reference_pres
            )
            rate_pressure = (
                cycle_zero_pressure[:-1] + cycle_zero_pressure[1:]
            ) / 2
            valid = np.isfinite(rates) & np.isfinite(rate_pressure)
            if valid.sum() >= 2:
                sample_pressure = rate_pressure[valid]
                sample_rate = rates[valid]
                order = np.argsort(sample_pressure)
                sample_pressure, indices = np.unique(
                    sample_pressure[order], return_index=True
                )
                sample_rate = sample_rate[order][indices]
                inside = ((pressure >= sample_pressure.min())
                          & (pressure <= sample_pressure.max()))
                profiles[c, inside] = np.interp(
                    pressure[inside], sample_pressure, sample_rate
                )
        cycle_profiles.append(profiles)

    max_cycles = max((len(values) for values in cycle_profiles), default=0)
    padded_profiles = np.full((2, max_cycles, len(pressure)), np.nan)
    std_rate = np.full((2, len(pressure)), np.nan)
    for s, profiles in enumerate(cycle_profiles):
        padded_profiles[s, :len(profiles)] = profiles
        for p in range(len(pressure)):
            values = profiles[:, p]
            values = values[np.isfinite(values)]
            if values.size >= 2:
                std_rate[s, p] = np.std(values)

    result = xr.Dataset(
        {
            "zero_pressure": (("shear", lag_dim), zero_pressure),
            "zero_lag": (("shear", "pres"), zero_lag),
            "monthly_descent_rate": (("shear", "lag_interval"), monthly_rate),
            "descent_rate": (("shear", "pres"), profile_rate),
            "cycle_descent_rate": (("shear", "cycle", "pres"), padded_profiles),
            "std_descent_rate": (("shear", "pres"), std_rate),
            "mean_descent_rate": ("shear", mean_rate),
        },
        coords={
            "shear": shear_types,
            lag_dim: selected[0][lag_dim],
            "lag_interval": (lags[:-1] + lags[1:]) / 2,
            "cycle": np.arange(max_cycles),
            "pres": ("pres", pressure, {"units": "hPa"}),
        },
    )
    for name in ("monthly_descent_rate", "descent_rate", "cycle_descent_rate",
                 "std_descent_rate", "mean_descent_rate"):
        result[name].attrs["units"] = "hPa month-1"
    result["zero_pressure"].attrs["units"] = "hPa"
    result["zero_lag"].attrs["units"] = "months"
    return result


def cycle_coherence(events, composite=None, pres_range=(1, 200)):
    """Compare onset-aligned QBO cycles with their composite.

    ``events`` contains cycle windows with ``(event, lag, pres)`` dimensions;
    ``composite`` defaults to their mean. Coherence is the squared lag correlation at each
    pressure. Whole-pattern correlation is the pressure-weighted Pearson
    correlation over lag and pressure. Amplitude ratio is the cycle RMS divided
    by composite RMS. Cycle spread excludes pointwise values outside the
    standard 1.5-IQR limits. The result also contains a reconstructed timeline
    when ``event_time`` is available. The displayed wind retains the input
    pressure range without changing the range used by the metrics.
    """
    required = {"event", "lag", "pres"}
    if not required.issubset(events.dims):
        raise ValueError("events must have 'event', 'lag' and 'pres' dimensions")
    extra_dims = set(events.dims) - required
    if extra_dims:
        raise ValueError(f"Select or average the additional dimension(s) {sorted(extra_dims)} first")
    events_full = events.transpose("event", "lag", "pres")
    events = events_full.sel(pres=slice(*sorted(pres_range)))
    composite = events.mean("event") if composite is None else composite
    composite = composite.sel(lag=events["lag"], pres=events["pres"]).transpose("lag", "pres")
    event_values = np.asarray(events.values, dtype=float)
    composite_values = np.asarray(composite.values, dtype=float)
    if not np.all(np.isfinite(composite_values)):
        raise ValueError("composite contains non-finite values")

    pressure = np.asarray(events["pres"].values, dtype=float)
    if pressure.size < 2 or np.any(pressure <= 0):
        raise ValueError("At least two positive pressure levels are required")
    vertical_weights = np.abs(np.gradient(np.log(pressure)))
    vertical_weights /= vertical_weights.sum()

    coherence = xr.corr(events, composite, dim="lag") ** 2

    flat_weights = np.tile(vertical_weights, events.sizes["lag"])
    y = composite_values.ravel()
    y_mean = np.sum(flat_weights * y) / flat_weights.sum()
    composite_rms = np.sqrt(np.sum(flat_weights * y ** 2) / flat_weights.sum())
    pattern_correlation = np.full(events.sizes["event"], np.nan)
    amplitude_ratio = np.full_like(pattern_correlation, np.nan)
    for e in range(events.sizes["event"]):
        x = event_values[e].ravel()
        valid = np.isfinite(x)
        weights = flat_weights[valid]
        if not valid.any() or weights.sum() == 0:
            continue
        x, y_valid = x[valid], y[valid]
        x_mean = np.sum(weights * x) / weights.sum()
        covariance = np.sum(weights * (x - x_mean) * (y_valid - y_mean))
        x_power = np.sum(weights * (x - x_mean) ** 2)
        y_valid_power = np.sum(weights * (y_valid - y_mean) ** 2)
        if x_power > 0 and y_valid_power > 0:
            pattern_correlation[e] = covariance / np.sqrt(x_power * y_valid_power)
        rms = np.sqrt(np.sum(weights * x ** 2) / weights.sum())
        if composite_rms > 0:
            amplitude_ratio[e] = rms / composite_rms

    if "event_time" in events.coords:
        event_time = np.asarray(events["event_time"].values)
        flat_time = event_time.ravel()
        flat_event = np.repeat(np.arange(events.sizes["event"]), events.sizes["lag"])
        flat_lag_index = np.tile(np.arange(events.sizes["lag"]), events.sizes["event"])
        flat_lag = np.tile(np.asarray(events["lag"].values), events.sizes["event"])
        valid_time = (~np.isnat(flat_time) if np.issubdtype(flat_time.dtype, np.datetime64)
                      else np.isfinite(flat_time))
        unique_time = np.unique(flat_time[valid_time])
        event_index, lag_index = [], []
        for time in unique_time:
            candidates = np.flatnonzero(flat_time == time)
            best = candidates[np.argmin(np.abs(flat_lag[candidates]))]
            event_index.append(flat_event[best])
            lag_index.append(flat_lag_index[best])
        timeline = unique_time
        event_index, lag_index = np.asarray(event_index), np.asarray(lag_index)
    else:
        event_index = np.repeat(np.arange(events.sizes["event"]), events.sizes["lag"])
        lag_index = np.tile(np.arange(events.sizes["lag"]), events.sizes["event"])
        timeline = np.arange(len(event_index))

    wind = np.asarray(events_full.values, dtype=float)[event_index, lag_index]
    coherence_time = coherence.values[event_index]
    mean_coherence = coherence.weighted(
        xr.DataArray(vertical_weights, dims="pres", coords={"pres": events["pres"]})
    ).mean("pres")

    q1, q3 = np.nanpercentile(event_values, (25, 75), axis=0)
    width = q3 - q1
    inlier = ((event_values >= q1 - 1.5 * width)
              & (event_values <= q3 + 1.5 * width))
    cycle_std = np.nanstd(np.where(inlier, event_values, np.nan), axis=0)

    return xr.Dataset(
        {"events": (("event", "lag", "pres"), event_values),
         "composite": (("lag", "pres"), composite_values),
         "cycle_std": (("lag", "pres"), cycle_std),
         "coherence": coherence,
         "mean_coherence": mean_coherence,
         "pattern_correlation": ("event", pattern_correlation),
         "amplitude_ratio": ("event", amplitude_ratio),
         "wind": (("time", "wind_pres"), wind),
         "coherence_time": (("time", "pres"), coherence_time)},
        coords={"event": events["event"], "lag": events["lag"],
                "pres": events["pres"],
                "wind_pres": ("wind_pres", events_full["pres"].values),
                "time": timeline})


def fit_parabolic_cylinder(profile, orders=(0,), lat_range=(-30, 30)):
    """Fit parabolic-cylinder functions to a latitude profile.

    Components in ``orders`` share a centre and scale over ``lat_range``.
    ``D0`` represents zonal wind; ``D0+D2`` represents temperature and ozone.
    The centre is bounded by 5 degrees from the equator and the scale by
    2--30 degrees. The result contains component amplitudes, centre, scale,
    equatorial-lobe FWHM, residual RMS, profile and fitted curve.
    """
    orders = tuple(np.atleast_1d(orders))
    prof = profile.sel(latitude=slice(*lat_range))
    lat, y = prof["latitude"].values, prof.values

    def model(x, *params):
        *amps, center, scale = params
        xs = (x - center) / scale
        return sum(a * special.pbdv(n, xs)[0] for a, n in zip(amps, orders))

    d_n0 = [special.pbdv(n, 0.0)[0] for n in orders]
    y_eq = y[np.abs(lat).argmin()]
    amps0 = [y_eq / d / len(orders) if abs(d) > 1e-12 else np.max(np.abs(y)) for d in d_n0]
    # Constrain the fit to an equatorial structure.
    popt, _ = curve_fit(model, lat, y, p0=amps0 + [0.0, 10.0],
                        bounds=([-np.inf] * len(orders) + [-5.0, 2.0],
                                [np.inf] * len(orders) + [5.0, 30.0]))
    *amps, center, scale = popt

    # Diagnose equatorial-lobe FWHM on a fine latitude grid.
    x = np.linspace(center, center + 8 * scale, 2001)
    f = np.abs(model(x, *popt))
    i = int(np.argmax(f < 0.5 * f[0]))
    half_width = np.interp(0.5 * f[0], [f[i], f[i - 1]], [x[i], x[i - 1]]) - center

    fit = xr.DataArray(model(lat, *popt), coords={"latitude": lat}, name=profile.name)
    residual_rms = np.sqrt(np.nanmean((fit.values - y) ** 2))
    return {"amp": np.array(amps), "center": center, "scale": scale, "fwhm": 2 * half_width,
            "residual_rms": residual_rms,
            "profile": prof, "fit": fit}


def latitudinal_width(da, orders=None, lat_range=(-30, 30),
                      pres_range=(1, 200), max_normalized_rmse=0.5,
                      max_width_discontinuity=0.1):
    """Return lag-zero QBO latitudinal width at each pressure.

    Profiles over ``lat_range`` are fitted with the parabolic-cylinder
    ``orders``. By default, zonal wind uses ``D0`` while temperature and ozone
    use ``D0+D2``. Width is the equatorial-lobe FWHM. Fit quality combines the
    residual RMS normalised by full-cycle lag--latitude RMS and the local
    log-FWHM discontinuity in pressure. ``max_normalized_rmse`` and
    ``max_width_discontinuity`` set the acceptance thresholds; ``None``
    disables a criterion. The result retains all fits and marks accepted
    levels with ``good_fit``.
    """
    if pres_range is not None:
        da = da.sel(pres=slice(*sorted(pres_range)))
    if max_normalized_rmse is not None and max_normalized_rmse <= 0:
        raise ValueError("max_normalized_rmse must be positive or None")
    if max_width_discontinuity is not None and max_width_discontinuity <= 0:
        raise ValueError("max_width_discontinuity must be positive or None")
    da = da.sel(latitude=slice(*lat_range))
    if orders is None:
        temperature_or_ozone = {
            "T", "o3", "air_temperature",
            "mole_fraction_of_ozone_in_air",
            "mole_fraction_of_o3_in_air",
            "mass_fraction_of_ozone_in_air",
        }
        orders = ((0, 2) if da.name in temperature_or_ozone or
                  da.attrs.get("standard_name") in temperature_or_ozone
                  else (0,))

    if "lag" in da.dims and da.sizes["lag"] > 1:
        cycle_anomaly = da - da.mean("lag", skipna=True)
        signal_rms = np.sqrt(
            (cycle_anomaly ** 2).mean(("lag", "latitude"), skipna=True)
        ).values
        profile = da.sel(lag=0)
    else:
        profile = da.sel(lag=0) if "lag" in da.dims else da
        signal_rms = np.sqrt(
            (profile ** 2).mean("latitude", skipna=True)
        ).values
    orders = tuple(np.atleast_1d(orders))

    n_pres = profile.sizes["pres"]
    out = {"amp": np.full((n_pres, len(orders)), np.nan), "center": np.full(n_pres, np.nan),
           "scale": np.full(n_pres, np.nan), "fwhm": np.full(n_pres, np.nan),
           "residual_rms": np.full(n_pres, np.nan),
           "fit": np.full((n_pres, profile.sizes["latitude"]), np.nan)}
    for k in range(n_pres):
        f = fit_parabolic_cylinder(profile.isel(pres=k), orders, lat_range)
        for key in ("amp", "center", "scale", "fwhm", "residual_rms"):
            out[key][k] = f[key]
        out["fit"][k] = f["fit"].values

    normalized_rmse = np.divide(
        out["residual_rms"], signal_rms,
        out=np.full(n_pres, np.nan),
        where=np.isfinite(signal_rms) & (signal_rms > 0))
    width_discontinuity = np.full(n_pres, np.nan)
    pressure = np.asarray(profile["pres"].values, dtype=float)
    log_pressure = np.full(n_pres, np.nan)
    np.log(pressure, out=log_pressure, where=pressure > 0)
    fwhm = out["fwhm"]
    valid = (np.isfinite(log_pressure) & np.isfinite(fwhm) &
             (pressure > 0) & (fwhm > 0))
    for k in range(1, n_pres - 1):
        if valid[k - 1:k + 2].all():
            expected_log_width = np.interp(
                log_pressure[k], log_pressure[[k - 1, k + 1]],
                np.log(fwhm[[k - 1, k + 1]]))
            width_discontinuity[k] = abs(np.log(fwhm[k]) - expected_log_width)

    good_fit = np.isfinite(normalized_rmse)
    if max_normalized_rmse is not None:
        good_fit &= normalized_rmse <= max_normalized_rmse
    if max_width_discontinuity is not None:
        good_fit &= (np.isnan(width_discontinuity) |
                     (width_discontinuity <= max_width_discontinuity))
    return xr.Dataset(
        {"amp": (("pres", "order"), out["amp"]), "center": ("pres", out["center"]),
         "scale": ("pres", out["scale"]), "fwhm": ("pres", out["fwhm"]),
         "residual_rms": ("pres", out["residual_rms"]),
         "normalized_rmse": ("pres", normalized_rmse),
         "width_discontinuity": ("pres", width_discontinuity),
         "good_fit": ("pres", good_fit),
         "fit": (("pres", "latitude"), out["fit"]), "profile": profile},
        coords={"pres": profile["pres"], "order": list(orders),
                "latitude": profile["latitude"]},
        attrs={name: value for name, value in (
            ("max_normalized_rmse", max_normalized_rmse),
            ("max_width_discontinuity", max_width_discontinuity),
        ) if value is not None})
