"""Run an end-to-end QBO composite and diagnostic workflow.

The example excludes cycles with onsets in 2015 and 2020. Set the options
below before running ``python example.py``.
"""

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from composite import (compute_composite, data_loc, extract_events,
                       reference_dates, reference_timeseries)
from metrics import (cycle_coherence, daily_composite_period, descent_rate,
                     latitudinal_width, max_min_amplitude, phase_amplitude,
                     zero_crossing_period_range)
from plotting import (plot_composite, plot_cycle_coherence,
                      plot_cycle_summary, plot_descent_rate,
                      plot_period_fits, plot_phase_amplitude, plot_reference,
                      plot_width_fits, plot_width_section)
from utils import load_composite, load_data, save_composite

# Workflow settings.
REBUILD_COMPOSITES = True
USE_DAILY_PERIOD = False
EXCLUDED_ONSET_YEARS = (2015, 2020)
DETREND_COMPOSITES = False
SAVE_REFERENCE_DATES = True
REFERENCE_DATES_PATH = "reference_dates.nc"
SHEAR_TYPES = ("westerly", "easterly")


def _composite_path(shear):
    """Return the saved composite path for one onset direction."""
    return f"qbo_composite_{shear}.nc"


def _get_composites(ds_monthly, reference, dates):
    """Build composites or load saved composites with matching exclusions."""
    if REBUILD_COMPOSITES:
        composites = {}
        for shear in SHEAR_TYPES:
            onset_dates = dates[shear]
            print(f"Found {len(onset_dates)} {shear}-onset reference dates")
            plot_reference(
                reference, onset_dates,
                title=f"{shear.capitalize()}-onset reference timeseries",
            )
            composites[shear] = compute_composite(
                ds_monthly,
                onset_dates,
                exclude_years=EXCLUDED_ONSET_YEARS,
                detrend=DETREND_COMPOSITES,
            )
            save_composite(composites[shear], _composite_path(shear))
        return composites

    composites = {
        shear: load_composite(_composite_path(shear))
        for shear in SHEAR_TYPES
    }
    expected = ",".join(map(str, EXCLUDED_ONSET_YEARS))
    for shear, composite in composites.items():
        exclusions_match = composite.attrs.get("excluded_years", "") == expected
        detrending_matches = (
            bool(composite.attrs.get("detrended", 0)) == DETREND_COMPOSITES
        )
        if not exclusions_match or not detrending_matches:
            raise ValueError(
                f"Saved {shear} composite uses different settings. "
                "Set REBUILD_COMPOSITES=True to update it."
            )
    return composites


def main():
    """Run the configured example workflow and display its figures."""
    print(f"Excluded onset years: {EXCLUDED_ONSET_YEARS}")
    ds_daily, ds_monthly = load_data(data_loc)

    reference = reference_timeseries(ds_monthly)
    dates = {
        shear: reference_dates(reference, direction=shear)
        for shear in SHEAR_TYPES
    }
    if SAVE_REFERENCE_DATES:
        xr.Dataset({
            f"{shear}_onset_date": (f"{shear}_onset", dates[shear])
            for shear in SHEAR_TYPES
        }).to_netcdf(REFERENCE_DATES_PATH)
        print(f"Saved reference dates to {REFERENCE_DATES_PATH}")

    u_eq_monthly = (
        ds_monthly.sel(pres=slice(0.5, 200), latitude=[0])
        .mean("latitude")
        .load()
    )

    # Build both onset composites with one consistent configuration.
    composites = _get_composites(ds_monthly, reference, dates)
    westerly_composite = composites["westerly"]
    easterly_composite = composites["easterly"]
    westerly_equatorial = westerly_composite.sel(latitude=0)
    easterly_equatorial = easterly_composite.sel(latitude=0)

    # Diagnose latitudinal width.
    width_fits = {
        "u": latitudinal_width(
            westerly_composite["u"], orders=(0,), max_normalized_rmse=0.3
        ),
        "T": latitudinal_width(westerly_composite["T"], orders=(2,)),
        "T (D0+D2)": latitudinal_width(westerly_composite["T"]),
    }
    pres_by_var = {
        "u": [50.0],
        "T": [30.0],
        "T (D0+D2)": [30.0],
    }

    for var, result in width_fits.items():
        selected = result.sel(pres=pres_by_var[var], method="nearest")
        for pressure in selected["pres"].values:
            level = selected.sel(pres=pressure)
            error = float(level["normalized_rmse"])
            width_jump = np.exp(float(level["width_discontinuity"]))
            if bool(level["good_fit"]):
                print(
                    f"{var} at {pressure:.1f} hPa: "
                    f"scale {float(level['scale']):.1f} deg, "
                    f"FWHM {float(level['fwhm']):.1f} deg, "
                    f"centre {float(level['center']):+.1f} deg, "
                    f"error {error:.2f}, width jump {width_jump:.1f}x"
                )
            else:
                print(
                    f"{var} at {pressure:.1f} hPa: fit rejected "
                    f"(error {error:.2f}, width jump {width_jump:.1f}x)"
                )

    plot_width_fits(width_fits, pres=pres_by_var)

    section_variables = [("u", "u"), ("T", "T (D0+D2)")]
    for var, key in section_variables:
        plot_width_section(
            westerly_composite[var], width_fits[key],
            title=f"QBO {key} latitudinal width at lag 0",
            show_quality=True,
        )

    # Diagnose period, phase and amplitude.
    u_composites = {
        "westerly": westerly_equatorial["u"],
        "easterly": easterly_equatorial["u"],
    }
    if USE_DAILY_PERIOD:
        if ds_daily is None:
            raise ValueError("Daily period estimation requires daily or sub-daily input")
        daily_period = daily_composite_period(
            ds_daily, exclude_years=EXCLUDED_ONSET_YEARS,
            detrend=DETREND_COMPOSITES)
        qbo_period = float(daily_period["period"])
        period_range = daily_period["zero_crossing_period_range"]
        u_cycles = phase_amplitude(
            u_composites, period=qbo_period,
            zero_crossing_range=period_range)
    else:
        daily_period = None
        period_range = zero_crossing_period_range(
            dates, exclude_years=EXCLUDED_ONSET_YEARS)
        u_cycles = phase_amplitude(
            u_composites, zero_crossing_range=period_range)
        qbo_period = float(u_cycles["period"])
    # Use the westerly-onset profile in the summary figures.
    westerly_cycle_fit = u_cycles.sel(shear="westerly")
    cycle_fits = {
        "u": westerly_cycle_fit,
        "T": phase_amplitude(
            westerly_equatorial["T"], period=qbo_period,
            zero_crossing_range=period_range),
    }
    composite_amplitude = max_min_amplitude(westerly_equatorial)
    for name, value in composite_amplitude.data_vars.items():
        print(f"Maximum {name} amplitude: {float(value.max()):.2g}")
    if daily_period is None:
        component_periods = ", ".join(
            f"{name} {float(value):.1f}"
            for name, value in zip(
                westerly_cycle_fit["period_composite"].values,
                westerly_cycle_fit["period_by_composite"].values))
        print(f"Mean QBO period at 30 hPa: {qbo_period:.2f} months "
              f"({component_periods}); zero-crossing range "
              f"{float(period_range.sel(period_bound='smallest')):.2f}–"
              f"{float(period_range.sel(period_bound='largest')):.2f} months")
    else:
        component_periods = ", ".join(
            f"{name} {float(value):.2f} months"
            for name, value in zip(daily_period["shear"].values,
                                   daily_period["period_by_composite"].values))
        print(f"Daily-sampled mean QBO period at 30 hPa: "
              f"{qbo_period:.3f} months / {float(daily_period['period_days']):.1f} days "
              f"({component_periods}; shear difference "
              f"{float(daily_period['period_days_shear_difference']):.1f} days); "
              f"zero-crossing range "
              f"{float(period_range.sel(period_bound='smallest')):.2f}–"
              f"{float(period_range.sel(period_bound='largest')):.2f} months")
    plot_period_fits(cycle_fits)
    plot_phase_amplitude(cycle_fits)

    # Diagnose cycle variability and zero-wind descent.
    descent_cycle_windows = {
        shear: extract_events(
            u_eq_monthly, dates[shear],
            exclude_years=EXCLUDED_ONSET_YEARS,
            detrend=DETREND_COMPOSITES)["u"]
        for shear in SHEAR_TYPES
    }
    variability_cycles = extract_events(
        u_eq_monthly, dates["westerly"],
        detrend=DETREND_COMPOSITES)["u"]
    coherence = cycle_coherence(variability_cycles)
    descent = descent_rate(u_composites, cycles=descent_cycle_windows)
    print(f"Mean cycle/composite coherence: {float(coherence['mean_coherence'].mean()):.2f}")
    for shear in descent["shear"].values:
        rate = float(descent["mean_descent_rate"].sel(shear=shear))
        print(f"Mean {shear} zero-wind descent rate: {rate:.2f} hPa/month")
    plot_cycle_coherence(
        coherence, variable="u", vertical_extents=westerly_cycle_fit)
    plot_cycle_summary(coherence, pres=30.0, variable="u")
    plot_descent_rate(descent)

    # Plot composite structure.
    plot_composite(
        westerly_equatorial["T"], x="lag", x_label="Lag (months)",
        title="Equatorial zonal wind (contours) and temperature (fill)",
        overlay=westerly_equatorial["u"], overlay_step=10.0,
    )
    plot_composite(
        westerly_composite["T"].sel(lag=0),
        x="latitude", x_label="Latitude",
        title="Zonal wind (contours) and temperature (fill) at lag 0",
        overlay=westerly_composite["u"].sel(lag=0), overlay_step=10.0,
    )
    plt.show()


if __name__ == "__main__":
    main()
