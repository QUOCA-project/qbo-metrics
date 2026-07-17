"""Example workflow for QBO composites and diagnostics."""

import numpy as np
from matplotlib import pyplot as plt

from composite import (compute_composite, data_loc, extract_events,
                       reference_dates, reference_timeseries)
from metrics import (cycle_coherence, daily_composite_period, descent_rate,
                     latitudinal_width, phase_amplitude)
from plotting import (plot_composite, plot_cycle_coherence,
                      plot_cycle_summary, plot_descent_rate,
                      plot_period_fits, plot_phase_amplitude, plot_reference,
                      plot_width_fits, plot_width_section)
from utils import load_composite, load_data, save_composite

# Set workflow options before running the script.
generate_composite = False
use_daily_period = False

if __name__ == "__main__":
    if generate_composite:
        ds_daily, ds_monthly = load_data(data_loc)
    else:
        ds_daily, ds_monthly = load_data(data_loc, variables=("u",))

    ref = reference_timeseries(ds_monthly)
    dates = {
        shear: reference_dates(ref, direction=shear)
        for shear in ("westerly", "easterly")
    }
    u_eq_monthly = (
        ds_monthly.sel(pres=slice(0.5, 200), latitude=[0])
        .mean("latitude")
        .load()
    )

    # Generate or load onset composites.
    if generate_composite:
        w_dates = dates["westerly"]
        print(f"Found {len(w_dates)} westerly-onset reference dates")
        plot_reference(ref, w_dates, title="Westerly-onset reference timeseries")

        w_comp = compute_composite(ds_monthly, w_dates)
        save_composite(w_comp, "qbo_composite_westerly.nc")

        e_dates = dates["easterly"]
        print(f"Found {len(e_dates)} easterly-onset reference dates")
        plot_reference(ref, e_dates, title="Easterly-onset reference timeseries")

        e_comp = compute_composite(ds_monthly, e_dates)
        save_composite(e_comp, "qbo_composite_easterly.nc")
    else:
        e_comp = load_composite("qbo_composite_easterly.nc")
        w_comp = load_composite("qbo_composite_westerly.nc")
    w_comp_eq = w_comp.sel(latitude=0)
    e_comp_eq = e_comp.sel(latitude=0)

    # Diagnose latitudinal width.
    fits = {
        "u": latitudinal_width(
            w_comp["u"], orders=(0,), max_normalized_rmse=0.3
        ),
        "T": latitudinal_width(w_comp["T"], orders=(2,)),
        "T (D0+D2)": latitudinal_width(w_comp["T"]),
    }
    if "o3" in w_comp:
        fits["o3 (D0+D2)"] = latitudinal_width(w_comp["o3"])
    pres_by_var = {
        "u": [50.0],
        "T": [30.0],
        "T (D0+D2)": [30.0],
    }
    if "o3 (D0+D2)" in fits:
        pres_by_var["o3 (D0+D2)"] = [30.0]
    for var, ds in fits.items():
        f = ds.sel(pres=pres_by_var[var], method="nearest")
        for p in f["pres"].values:
            level = f.sel(pres=p)
            if bool(level["good_fit"]):
                print(f"{var} at {p:.1f} hPa: scale {float(level['scale']):.1f} deg, "
                      f"FWHM {float(level['fwhm']):.1f} deg, "
                      f"centre {float(level['center']):+.1f} deg, "
                      f"error {float(level['normalized_rmse']):.2f}, "
                      f"width jump {np.exp(float(level['width_discontinuity'])):.1f}x")
            else:
                print(f"{var} at {p:.1f} hPa: fit rejected "
                      f"(error {float(level['normalized_rmse']):.2f}, "
                      f"width jump {np.exp(float(level['width_discontinuity'])):.1f}x)")
    width_fig = plot_width_fits(fits, pres=pres_by_var)
    section_variables = [("u", "u"), ("T", "T (D0+D2)")]
    if "o3" in w_comp:
        section_variables.append(("o3", "o3 (D0+D2)"))
    section_figs = [
        plot_width_section(
            w_comp[var], fits[key],
            title=f"QBO {key} latitudinal width at lag 0",
            show_quality=True,
        )
        for var, key in section_variables
    ]

    # Diagnose period, phase and amplitude.
    u_composites = {
        "westerly": w_comp_eq["u"],
        "easterly": e_comp_eq["u"],
    }
    if use_daily_period:
        if ds_daily is None:
            raise ValueError("Daily period estimation requires daily or sub-daily input")
        daily_period = daily_composite_period(ds_daily)
        qbo_period = float(daily_period["period"])
        u_cycles = phase_amplitude(u_composites, period=qbo_period)
    else:
        daily_period = None
        u_cycles = phase_amplitude(u_composites)
        qbo_period = float(u_cycles["period"])
    # Use the westerly-onset profile in the summary figures.
    u_cycle = u_cycles.sel(shear="westerly")
    cycle_fits = {
        "u": u_cycle,
        "T": phase_amplitude(w_comp_eq["T"], period=qbo_period),
    }
    if daily_period is None:
        component_periods = ", ".join(
            f"{name} {float(value):.1f}"
            for name, value in zip(u_cycle["period_composite"].values,
                                   u_cycle["period_by_composite"].values))
        print(f"Mean QBO period at 30 hPa: {qbo_period:.2f} months "
              f"({component_periods})")
    else:
        component_periods = ", ".join(
            f"{name} {float(value):.2f} months"
            for name, value in zip(daily_period["shear"].values,
                                   daily_period["period_by_composite"].values))
        print(f"Daily-sampled mean QBO period at 30 hPa: "
              f"{qbo_period:.3f} months / {float(daily_period['period_days']):.1f} days "
              f"({component_periods}; shear difference "
              f"{float(daily_period['period_days_shear_difference']):.1f} days)")
    period_fit_fig = plot_period_fits(cycle_fits)
    phase_amplitude_fig = plot_phase_amplitude(cycle_fits)

    # Diagnose cycle variability and zero-wind descent.
    u_cycle_windows = {
        shear: extract_events(u_eq_monthly, dates[shear])["u"]
        for shear in ("westerly", "easterly")
    }
    coherence = cycle_coherence(u_cycle_windows["westerly"])
    descent = descent_rate(u_composites, cycles=u_cycle_windows)
    print(f"Mean cycle/composite coherence: {float(coherence['mean_coherence'].mean()):.2f}")
    for shear in descent["shear"].values:
        rate = float(descent["mean_descent_rate"].sel(shear=shear))
        print(f"Mean {shear} zero-wind descent rate: {rate:.2f} hPa/month")
    coherence_fig = plot_cycle_coherence(
        coherence, variable="u", vertical_extents=u_cycle
    )
    cycle_summary_fig = plot_cycle_summary(coherence, pres=30.0, variable="u")
    descent_fig = plot_descent_rate(descent)

    # Plot composite structure.
    lag_height_fig = plot_composite(
        w_comp_eq["T"], x="lag", x_label="Lag (months)",
        title="Equatorial zonal wind (contours) and temperature (fill)",
        overlay=w_comp_eq["u"], overlay_step=10.0,
    )
    lat_height_fig = plot_composite(
        w_comp["T"].sel(lag=0), x="latitude", x_label="Latitude",
        title="Zonal wind (contours) and temperature (fill) at lag 0",
        overlay=w_comp["u"].sel(lag=0), overlay_step=10.0,
    )
    plt.show()
