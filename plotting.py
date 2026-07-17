"""Plot QBO composites and diagnostic results."""

from collections.abc import Mapping

import numpy as np
import xarray as xr
from matplotlib import colors, pyplot as plt, ticker


def _plot_reference_pressure(ax, pressure=30.0, label=None, linewidth=1.0):
    """Add the reference pressure to a pressure-axis plot."""
    if pressure is None:
        return None
    return ax.axhline(float(pressure), color="0.3", lw=linewidth, ls="--",
                      alpha=0.8, label=label)


def plot_composite(da, x, title="", x_label="", overlay=None,
                   overlay_step=None, overlay_color="k",
                   overlay_linestyles=None, pres_range=(1, 200), ax=None,
                   reference_pres=30.0):
    """Plot a pressure--``x`` composite as filled contours.

    Fill levels use a zero-centred diverging scale. ``overlay`` adds labelled
    line contours at ``overlay_step`` intervals. Pressure defaults to
    1--200 hPa; ``ax`` selects an existing Matplotlib axes. A dashed line
    marks ``reference_pres`` when it is not ``None``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure
    if pres_range is not None:
        da = da.sel(pres=slice(*sorted(pres_range)))
        if overlay is not None:
            overlay = overlay.sel(pres=da["pres"])
    lim = float(abs(da).max())
    levels = ticker.MaxNLocator(nbins=20, symmetric=True).tick_values(-lim, lim)
    cs = ax.contourf(da[x], da["pres"], da.transpose("pres", x), levels=levels, cmap="RdBu_r")
    if overlay is not None:
        if overlay_step is None:
            raise ValueError("overlay_step must be given with overlay")
        olevels = ticker.MultipleLocator(overlay_step).tick_values(float(overlay.min()), float(overlay.max()))
        widths = np.where(olevels == 0, 1.8, 0.8)
        cl = ax.contour(overlay[x], overlay["pres"], overlay.transpose("pres", x),
                        levels=olevels, colors=overlay_color, linewidths=widths, linestyles=overlay_linestyles)
        ax.clabel(cl, fontsize=8, fmt="%g")
    ax.set_yscale("log")
    if pres_range is None:
        ax.invert_yaxis()
    else:
        ax.set_ylim(max(pres_range), min(pres_range))
    _plot_reference_pressure(ax, reference_pres)
    if x == "lag":
        ax.axvline(0, color="0.25", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(title)
    fig.colorbar(cs, ax=ax)
    return fig


def plot_width_fits(fits, pres, title="QBO latitudinal structure"):
    """Plot latitude profiles and parabolic-cylinder fits.

    ``fits`` maps labels to ``latitudinal_width`` results. ``pres`` is one
    pressure or a label-to-pressure mapping. Panels show the nearest level,
    FWHM, scale and fit-quality status.
    """
    if isinstance(pres, Mapping):
        missing = [var for var in fits if var not in pres]
        if missing:
            raise ValueError(f"No pressure specified for fit(s) {missing}")
        pressures = pres
    else:
        pressures = {var: pres for var in fits}

    fig, axes = plt.subplots(1, len(fits), figsize=(5 * len(fits), 3.6), constrained_layout=True)
    for ax, (var, ds) in zip(np.atleast_1d(axes), fits.items()):
        f = ds.sel(pres=pressures[var], method="nearest")
        if "pres" in f.dims:
            if f.sizes["pres"] != 1:
                raise ValueError(
                    f"plot_width_fits requires one pressure for {var!r}; "
                    f"got {f.sizes['pres']}")
            f = f.squeeze("pres")
        good = bool(f["good_fit"]) if "good_fit" in f else True
        quality = (f"; error {float(f['normalized_rmse']):.2f}"
                   if "normalized_rmse" in f else "")
        if ("width_discontinuity" in f and
                np.isfinite(float(f["width_discontinuity"]))):
            quality += (f"; width jump "
                        f"{np.exp(float(f['width_discontinuity'])):.1f}x")
        ax.axhline(0, color="0.85", lw=1)
        ax.plot(f["latitude"], f["profile"], color="red", lw=2, label="composite")
        fit_alpha = 1.0 if good else 0.35
        ax.plot(f["latitude"], f["fit"], color="blue", lw=2,
                alpha=fit_alpha, label="fit")
        center = float(f["center"])
        fwhm = float(f["fwhm"])
        scale = float(f["scale"])
        if np.isfinite(center):
            for sign in (-1, 1):
                if np.isfinite(fwhm):
                    ax.axvline(center + sign * fwhm / 2,
                               color="k", lw=1.5, alpha=fit_alpha,
                               label="FWHM" if sign > 0 else None)
                if np.isfinite(scale):
                    ax.axvline(center + sign * scale,
                               color="k", lw=1.5, ls="--", alpha=fit_alpha,
                               label="scale" if sign > 0 else None)
        ax.set_xlabel("Latitude")
        ax.set_ylabel(f"{var} anomaly")
        fit_text = (f"FWHM {float(f['fwhm']):.1f}\N{DEGREE SIGN}; "
                    f"scale {float(f['scale']):.1f}\N{DEGREE SIGN}"
                    if good else "fit rejected")
        ax.set_title(f"{var} at {float(f['pres']):g} hPa; {fit_text}{quality}")
        ax.legend(frameon=False)
        ax.set_xlim(-90, 90)
    fig.suptitle(title)
    return fig


def plot_width_section(da, fit, title="QBO latitudinal width",
                       show_quality=False, reference_pres=30.0):
    """Plot a lag-zero latitude--pressure section with fitted widths.

    Solid lines mark FWHM and dashed lines mark scale at accepted levels.
    ``show_quality=True`` adds normalised error, width discontinuity and their
    acceptance thresholds. Dashed lines mark ``reference_pres``.
    """
    if "lag" in da.dims:
        da = da.sel(lag=0)
    da = da.sel(pres=slice(float(fit["pres"].min()), float(fit["pres"].max())))
    if show_quality:
        missing = [name for name in ("normalized_rmse",
                                     "width_discontinuity")
                   if name not in fit]
        if missing:
            raise ValueError(f"Quality panel requires fit variable(s) {missing}")
        fig, (ax, quality_ax) = plt.subplots(
            1, 2, figsize=(12, 4), sharey=True, constrained_layout=True,
            gridspec_kw={"width_ratios": (3, 1)})
        plot_composite(da, x="latitude", x_label="Latitude", title=title,
                       ax=ax, reference_pres=reference_pres)
    else:
        fig = plot_composite(
            da, x="latitude", x_label="Latitude", title=title,
            reference_pres=reference_pres)
        ax = fig.axes[0]
    good = fit["good_fit"] if "good_fit" in fit else xr.ones_like(
        fit["center"], dtype=bool)
    for sign in (-1, 1):
        ax.plot((fit["center"] + sign * fit["fwhm"] / 2).where(good),
                fit["pres"], "k",
                lw=1.5, label="FWHM" if sign > 0 else None)
        ax.plot((fit["center"] + sign * fit["scale"]).where(good),
                fit["pres"], "k",
                lw=1.5, ls="--", label="scale" if sign > 0 else None)
    ax.legend(frameon=False, loc="upper right")

    if show_quality:
        pressure = fit["pres"].values
        error = fit["normalized_rmse"].values
        discontinuity = fit["width_discontinuity"].values
        accepted = np.asarray(good.values, dtype=bool)
        quality_ax.plot(error, pressure, color="red", lw=1.5,
                        label="fit error")
        quality_ax.plot(discontinuity, pressure, color="purple", lw=1.5,
                        label="log-width discontinuity")
        _plot_reference_pressure(quality_ax, reference_pres)

        for name, colour, label in (
                ("max_normalized_rmse", "red", "error threshold"),
                ("max_width_discontinuity", "purple",
                 "discontinuity threshold")):
            value = fit.attrs.get(name)
            if value is not None:
                quality_ax.axvline(float(value), color=colour, lw=1,
                                   ls="--", label=label)

        finite = np.isfinite(error)
        quality_ax.scatter(np.zeros(np.count_nonzero(finite & accepted)),
                           pressure[finite & accepted], marker="o", s=18,
                           facecolor="green", edgecolor="none",
                           label="accepted", zorder=3, clip_on=False)
        quality_ax.scatter(np.zeros(np.count_nonzero(finite & ~accepted)),
                           pressure[finite & ~accepted], marker="x", s=22,
                           color="0.25", label="rejected", zorder=3,
                           clip_on=False)
        finite_values = np.concatenate((error[np.isfinite(error)],
                                        discontinuity[np.isfinite(discontinuity)]))
        upper = (max(1.0, 1.05 * float(finite_values.max()))
                 if finite_values.size else 1.0)
        quality_ax.set_xlim(0, upper)
        quality_ax.set_xlabel("Dimensionless quality metric")
        quality_ax.set_title("Fit validity")
        quality_ax.grid(axis="x", color="0.9", lw=0.8)
        quality_ax.tick_params(axis="y", labelleft=False)
        quality_ax.legend(frameon=False, fontsize=8, loc="center left",
                          bbox_to_anchor=(1.02, 0.5))
    return fig


def _expand_shear_fits(fits):
    """Expand shear-resolved results into labelled plotting rows."""
    expanded = []
    for name, ds in fits.items():
        if "shear" in ds.dims:
            expanded.extend((f"{name} ({shear})", ds.sel(shear=shear))
                            for shear in ds["shear"].values)
        else:
            expanded.append((name, ds))
    return expanded


def plot_period_fits(fits, title="QBO period fits"):
    """Plot reference-pressure profiles and their sinusoidal fits.

    ``fits`` maps labels to ``phase_amplitude`` results. Each panel reports
    period, amplitude and phase.
    """
    expanded = _expand_shear_fits(fits)
    fig, axes = plt.subplots(1, len(expanded),
                             figsize=(5 * len(expanded), 3.6),
                             constrained_layout=True)
    for ax, (var, ds) in zip(np.atleast_1d(axes), expanded):
        reference_pres = ds.attrs.get("reference_pres", 30.0)
        f = (ds.sel(pres=reference_pres) if reference_pres in ds["pres"]
             else ds.interp(pres=reference_pres))
        period = float(ds["period"])

        ax.axhline(0, color="0.85", lw=1)
        ax.axvline(0, color="0.25", lw=0.8, ls="--", alpha=0.7)
        ax.plot(f["lag"], f["profile"], color="red", lw=2,
                label="composite")
        ax.plot(f["lag"], f["fit"], color="blue", lw=2, label="sine fit")
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel(f"{var} anomaly")
        period_text = f"period {period:.1f} months"
        if "period_by_composite" in ds:
            components = ", ".join(
                f"{name}: {float(value):.1f}"
                for name, value in zip(ds["period_composite"].values,
                                       ds["period_by_composite"].values))
            period_text += f" (mean of {components})"
        ax.set_title(f"{var} at {float(f['pres']):g} hPa; {period_text}\n"
                     f"amplitude {float(f['amp']):.2g}, phase {float(f['phase']):+.2f} rad")
        ax.legend(frameon=False)
    fig.suptitle(title)
    return fig


def plot_daily_period_fits(result, title="Daily-sampled QBO period fits"):
    """Plot daily 30-hPa onset composites and sinusoidal fits.

    Panels report each onset-direction period; the title reports their mean
    and absolute difference.
    """
    required = {"profile", "fit", "period_by_composite",
                "period_days_by_composite", "period", "period_days",
                "period_days_shear_difference"}
    missing = sorted(required - set(result.data_vars))
    if missing or "shear" not in result.dims or "lag" not in result.dims:
        detail = f": missing {missing}" if missing else ""
        raise ValueError(f"result is not a daily composite period result{detail}")
    fig, axes = plt.subplots(
        1, result.sizes["shear"],
        figsize=(5.5 * result.sizes["shear"], 3.7),
        constrained_layout=True,
        squeeze=False,
    )
    units = result["profile"].attrs.get("units", "")
    units = f" ({units})" if units else ""
    lag_units = result["lag"].attrs.get("units", "days")
    for ax, shear in zip(axes[0], result["shear"].values):
        selected = result.sel(shear=shear)
        ax.axhline(0, color="0.85", lw=1)
        ax.axvline(0, color="0.25", lw=0.8, ls="--", alpha=0.7)
        ax.plot(result["lag"], selected["profile"], color="red", lw=1.5,
                label="daily composite")
        ax.plot(result["lag"], selected["fit"], color="blue", lw=2,
                label="sine fit")
        ax.set_xlabel(f"Lag ({lag_units})")
        ax.set_ylabel(f"u anomaly{units}")
        ax.set_title(
            f"{str(shear).capitalize()} onset: "
            f"{float(selected['period_by_composite']):.2f} months / "
            f"{float(selected['period_days_by_composite']):.1f} days"
        )
        ax.legend(frameon=False)
    fig.suptitle(
        f"{title}\ncommon period {float(result['period']):.2f} months / "
        f"{float(result['period_days']):.1f} days; shear difference "
        f"{float(result['period_days_shear_difference']):.1f} days"
    )
    return fig


def _wrapped_phase_line(phase, pres, phase_limits=(-np.pi, np.pi)):
    """Return a phase line wrapped to ``phase_limits`` with edge breaks."""
    phase = np.asarray(phase, dtype=float)
    pres = np.asarray(pres, dtype=float)
    lower, upper = phase_limits
    width = upper - lower
    if not np.isfinite(lower + upper) or width < 2 * np.pi - 1e-10:
        raise ValueError("phase_limits must span at least 2*pi radians")

    unwrapped = np.full_like(phase, np.nan)
    finite = np.isfinite(phase)
    starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
    ends = np.flatnonzero(finite & np.r_[~finite[1:], True]) + 1
    for start, end in zip(starts, ends):
        unwrapped[start:end] = np.unwrap(phase[start:end])
    if finite.any():
        centre = (lower + upper) / 2
        unwrapped[finite] += 2 * np.pi * np.round(
            (centre - np.nanmedian(unwrapped)) / (2 * np.pi))

    marker_phase = lower + np.mod(unwrapped - lower, width)
    first = int(np.flatnonzero(finite)[0]) if finite.any() else 0
    plot_phase = [marker_phase[first]] if finite.any() else [np.nan]
    plot_pres = [pres[first]] if finite.any() else [np.nan]

    for phase0, phase1, pres0, pres1 in zip(unwrapped[:-1], unwrapped[1:],
                                            pres[:-1], pres[1:]):
        if not np.all(np.isfinite((phase0, phase1, pres0, pres1))):
            plot_phase.extend((np.nan, phase1))
            plot_pres.extend((np.nan, pres1))
            continue

        phase0_plot = lower + np.mod(phase0 - lower, width)
        phase1_plot = phase0_plot + phase1 - phase0
        if phase1_plot > upper:
            edge, opposite_edge = upper, lower
            phase1_wrapped = phase1_plot - width
        elif phase1_plot < lower:
            edge, opposite_edge = lower, upper
            phase1_wrapped = phase1_plot + width
        else:
            plot_phase.append(phase1_plot)
            plot_pres.append(pres1)
            continue

        weight = (edge - phase0_plot) / (phase1_plot - phase0_plot)
        crossing_pres = np.exp(np.log(pres0) + weight *
                               (np.log(pres1) - np.log(pres0)))
        plot_phase.extend((edge, np.nan, opposite_edge, phase1_wrapped))
        plot_pres.extend((crossing_pres, np.nan, crossing_pres, pres1))

    return np.asarray(plot_phase), np.asarray(plot_pres), marker_phase


def _vertical_extent_specs(source):
    """Return finite top and bottom extent values with plot metadata."""
    if source is None:
        return []
    specs = []
    for boundary, name, linestyle in (
            ("top", "vertical_extent_top", "-."),
            ("bottom", "vertical_extent_bottom", ":")):
        if name not in source:
            continue
        data = source[name]
        value = float(data)
        if not np.isfinite(value):
            continue
        fraction = getattr(data, "attrs", {}).get("fraction", 0.1)
        specs.append({
            "boundary": boundary,
            "value": value,
            "fraction": fraction,
            "linestyle": linestyle,
            "label": (f"{boundary.capitalize()} {fraction:.0%} extent "
                      f"({value:.3g} hPa)"),
        })
    return specs


def plot_phase_amplitude(fits, title="QBO phase and amplitude",
                         show_vertical_extent=True,
                         phase_limits=(-np.pi, np.pi),
                         pres_range=(1, 200)):
    """Plot amplitude and phase profiles from sinusoidal fits.

    ``fits`` maps labels to ``phase_amplitude`` results. Each row shows
    amplitude and wrapped phase. ``show_vertical_extent`` marks the top and
    bottom amplitude thresholds; ``phase_limits`` controls phase wrapping.
    """
    expanded = _expand_shear_fits(fits)
    fig, axes = plt.subplots(len(expanded), 2, squeeze=False,
                             figsize=(9, 3.5 * len(expanded)),
                             constrained_layout=True, sharey="row")
    lower, upper = phase_limits
    half_pi = np.pi / 2
    phase_ticks = np.arange(np.ceil(lower / half_pi),
                            np.floor(upper / half_pi) + 1) * half_pi

    def pi_label(value):
        multiple = int(np.round(value / half_pi))
        if multiple == 0:
            return "0"
        sign = "-" if multiple < 0 else ""
        numerator = abs(multiple)
        if numerator % 2 == 0:
            coefficient = numerator // 2
            return rf"${sign}\pi$" if coefficient == 1 else rf"${sign}{coefficient}\pi$"
        return rf"${sign}\pi/2$" if numerator == 1 else rf"${sign}{numerator}\pi/2$"

    phase_labels = [pi_label(value) for value in phase_ticks]

    for row, (var, ds) in enumerate(expanded):
        amp_ax, phase_ax = axes[row]
        units = ds["profile"].attrs.get("units", "")
        units = f" ({units})" if units else ""
        reference_pres = ds.attrs.get("reference_pres", 30.0)

        amp_ax.plot(ds["amp"], ds["pres"], color="b", marker="o",
                    ms=3)
        _plot_reference_pressure(
            amp_ax, reference_pres,
            label=f"Reference ({reference_pres:g} hPa)")
        amp_ax.set_xlabel(f"Amplitude{units}")
        amp_ax.set_ylabel("Pressure (hPa)")
        amp_ax.set_title(f"{var} amplitude")
        amp_ax.legend(frameon=False)

        phase_ax.axvline(0, color="0.85", lw=1)
        phase_line, phase_pres, phase_markers = _wrapped_phase_line(
            ds["phase"], ds["pres"], phase_limits=phase_limits)
        phase_ax.plot(phase_line, phase_pres, color="r")
        phase_ax.plot(phase_markers, ds["pres"], color="r", ls="none",
                      marker="o", ms=3)
        _plot_reference_pressure(phase_ax, reference_pres)
        extent_specs = (_vertical_extent_specs(ds)
                        if show_vertical_extent else [])
        for spec in extent_specs:
            amp_ax.axhline(spec["value"], color="green", lw=1.5,
                           ls=spec["linestyle"], label=spec["label"])
            phase_ax.axhline(spec["value"], color="green", lw=1.5,
                             ls=spec["linestyle"])
            amp_ax.legend(frameon=False)
        phase_ax.set_xlim(*phase_limits)
        phase_ax.set_xticks(phase_ticks, phase_labels)
        phase_ax.set_xlabel("Phase (radians)")
        phase_ax.set_title(f"{var} phase; period {float(ds['period']):.1f} months")

        for ax in (amp_ax, phase_ax):
            ax.set_yscale("log")
            ax.grid(color="0.9", lw=0.8)
        if pres_range is None:
            amp_ax.invert_yaxis()
        else:
            top, bottom = sorted(pres_range)
            for spec in extent_specs:
                if spec["boundary"] == "top":
                    top = min(top, spec["value"])
                    if spec["value"] < min(pres_range):
                        top = min(top, float(ds["pres"].min()))
                else:
                    bottom = max(bottom, spec["value"])
                    if spec["value"] > max(pres_range):
                        bottom = max(bottom, float(ds["pres"].max()))
            amp_ax.set_ylim(bottom, top)

    fig.suptitle(title)
    return fig


def plot_cycle_coherence(result, variable="QBO variable",
                         title="QBO cycle coherence", pres_range=(1, 200),
                         vertical_extents=None, reference_pres=30.0):
    """Plot QBO-cycle anomalies and coherence against time.

    ``result`` is returned by ``cycle_coherence``. Both panels use its
    reconstructed timeline. The wind panel uses the full input pressure range;
    coherence uses the diagnostic pressure range.
    ``vertical_extents`` adds boundaries from ``phase_amplitude`` to the QBO
    wind panel. Dashed lines mark ``reference_pres`` in every panel when it
    is not ``None``.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, sharey=True,
                             constrained_layout=True)
    wind = result["wind"]
    limit = float(np.nanmax(np.abs(wind.values)))
    levels = ticker.MaxNLocator(nbins=18, symmetric=True).tick_values(-limit, limit)

    wind_contour = axes[0].contourf(
        result["time"], result["wind_pres"],
        wind.transpose("wind_pres", "time"),
        levels=levels, cmap="RdBu_r", extend="both")
    if float(wind.min()) <= 0 <= float(wind.max()):
        axes[0].contour(
            result["time"], result["wind_pres"],
            wind.transpose("wind_pres", "time"), levels=[0], colors="k",
            linewidths=1.1)
    axes[0].set_title("QBO wind")

    coherence_levels = np.linspace(0, 1, 11)
    coherence_contour = axes[1].contourf(
        result["time"], result["pres"],
        result["coherence_time"].transpose("pres", "time"),
        levels=coherence_levels, cmap="viridis")
    axes[1].set_title("Cycle/composite coherence ($r^2$ across lag)")

    extent_specs = _vertical_extent_specs(vertical_extents)
    for spec in extent_specs:
        axes[0].axhline(spec["value"], color="green", lw=2.5,
                       ls=spec["linestyle"], label=spec["label"])
    for i, ax in enumerate(axes):
        label = (f"Reference ({reference_pres:g} hPa)"
                 if i == 0 and reference_pres is not None else None)
        _plot_reference_pressure(ax, reference_pres, label=label,
                                 linewidth=2.0)
    if extent_specs or reference_pres is not None:
        axes[0].legend(frameon=True, facecolor="white", framealpha=0.85,
                       edgecolor="none", fontsize=8, loc="upper left")

    if (np.issubdtype(result["event"].dtype, np.datetime64) and
            np.issubdtype(result["time"].dtype, np.datetime64)):
        for date in result["event"].values:
            for ax in axes:
                ax.axvline(date, color="0.25", lw=0.6, ls="--", alpha=0.5)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Pressure (hPa)")
    if pres_range is None:
        axes[0].invert_yaxis()
    else:
        top, bottom = sorted(pres_range)
        if extent_specs:
            values = [spec["value"] for spec in extent_specs]
            top = min(top, 0.9 * min(values))
            bottom = max(bottom, max(values))
        axes[0].set_ylim(bottom, top)
    axes[-1].set_xlabel("Time")
    fig.colorbar(wind_contour, ax=axes[0], label=f"{variable} anomaly")
    fig.colorbar(coherence_contour, ax=axes[1], label="Coherence")
    fig.suptitle(title)
    return fig


def plot_cycle_summary(result, pres=30.0, variable="QBO variable",
                       title="QBO cycle-to-cycle variability"):
    """Plot QBO-cycle profiles and whole-pattern variability metrics.

    The profile panel uses the level nearest ``pres`` and shows the composite
    with a two-standard-deviation envelope. The summary panel compares pattern
    correlation with RMS-amplitude ratio. Colour and labels identify onset
    year.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                             constrained_layout=True)
    profile = result.sel(pres=pres, method="nearest")
    n_events = result.sizes["event"]
    event = result["event"]
    is_datetime = np.issubdtype(event.dtype, np.datetime64)
    if is_datetime:
        colour_values = np.asarray(event.dt.year.values, dtype=float)
        onset_labels = [str(int(year)) for year in colour_values]
        colourbar_label = "Onset year"
    else:
        # Label synthetic and non-calendar cycle coordinates directly.
        raw_event = np.asarray(event.values)
        onset_labels = [str(value) for value in raw_event]
        try:
            colour_values = raw_event.astype(float)
            colourbar_label = "Onset coordinate"
        except (TypeError, ValueError):
            colour_values = np.arange(n_events, dtype=float)
            colourbar_label = "Cycle order"
    cmap = plt.get_cmap("viridis")
    colour_min = float(np.nanmin(colour_values))
    colour_max = float(np.nanmax(colour_values))
    if colour_min == colour_max:
        norm = colors.Normalize(colour_min - 0.5, colour_max + 0.5)
    else:
        norm = colors.Normalize(colour_min, colour_max)
    cycle_colours = cmap(norm(colour_values))

    axes[0].axhline(0, color="0.8", lw=0.8)
    axes[0].axvline(0, color="0.25", lw=0.8, ls="--", alpha=0.7)
    for i in range(n_events):
        axes[0].plot(result["lag"], profile["events"].isel(event=i),
                     color=cycle_colours[i], lw=1, alpha=0.8)
    spread = 2 * profile["cycle_std"]
    axes[0].fill_between(result["lag"],
                         profile["composite"] - spread,
                         profile["composite"] + spread,
                         color="0.4", alpha=0.18, label=r"Composite $\pm2\sigma$")
    axes[0].plot(result["lag"], profile["composite"], color="k", lw=3,
                 label="Composite")
    axes[0].set_xlabel("Lag (months)")
    axes[0].set_ylabel(f"{variable} anomaly")
    axes[0].set_title(f"Individual cycles at {float(profile['pres']):.3g} hPa")
    axes[0].legend(frameon=False)

    axes[1].axvline(1, color="0.75", lw=0.8, ls="--")
    axes[1].axhline(1, color="0.75", lw=0.8, ls="--")
    axes[1].scatter(result["pattern_correlation"], result["amplitude_ratio"],
                    c=colour_values, cmap=cmap, norm=norm, s=50,
                    edgecolor="white", linewidth=0.5)
    for label, correlation, ratio in zip(onset_labels,
                                         result["pattern_correlation"].values,
                                         result["amplitude_ratio"].values):
        if np.isfinite(correlation) and np.isfinite(ratio):
            axes[1].annotate(label, (correlation, ratio),
                             xytext=(3, 2), textcoords="offset points", fontsize=7)
    axes[1].set_xlabel("Whole-cycle pattern correlation")
    axes[1].set_ylabel("RMS amplitude / composite RMS")
    axes[1].set_title("Structure and amplitude by cycle")
    correlations = result["pattern_correlation"].values
    ratios = result["amplitude_ratio"].values
    axes[1].set_xlim(min(-0.05, np.nanmin(correlations) - 0.05), 1.05)
    axes[1].set_ylim(min(0.9, np.nanmin(ratios) - 0.03),
                     max(1.1, np.nanmax(ratios) + 0.03))
    axes[1].grid(color="0.9", lw=0.8)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colourbar = fig.colorbar(scalar_mappable, ax=axes, label=colourbar_label)
    if is_datetime:
        colourbar.locator = ticker.MaxNLocator(nbins=6, integer=True)
        colourbar.update_ticks()
    fig.suptitle(title)
    return fig


def plot_descent_rate(result, title="QBO zero-wind descent rate",
                      reference_pres=30.0):
    """Plot composite descent-rate profiles and cycle-to-cycle spread."""
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    colours = {"easterly": "blue", "westerly": "red"}
    pressure = np.asarray(result["pres"].values, dtype=float)
    plotted = False
    for shear in result["shear"].values:
        colour = colours[str(shear)]
        profile = result.sel(shear=shear)
        rate = np.asarray(profile["descent_rate"].values, dtype=float)
        std = np.asarray(profile["std_descent_rate"].values, dtype=float)
        valid = np.isfinite(pressure) & np.isfinite(rate)
        plotted |= valid.any()
        label = (f"{str(shear).capitalize()} onset "
                 f"({float(profile['mean_descent_rate']):.2f} "
                 "hPa month$^{-1}$)")
        ax.plot(rate, pressure, color=colour, lw=2, label=label)
        spread_valid = valid & np.isfinite(std)
        ax.fill_betweenx(pressure, rate - std, rate + std,
                         where=spread_valid, color=colour, alpha=0.15)
    if not plotted:
        plt.close(fig)
        raise ValueError("No finite descent-rate profile values to plot")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("Downward rate (hPa month$^{-1}$)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_yscale("log")
    ax.set_ylim(float(np.nanmax(pressure)), float(np.nanmin(pressure)))
    _plot_reference_pressure(
        ax, reference_pres,
        label=(f"Reference ({reference_pres:g} hPa)"
               if reference_pres is not None else None))
    ax.set_title(title)
    ax.grid(color="0.9", lw=0.8)
    ax.legend(frameon=False)
    return fig


def plot_reference(ref, dates=None, title="QBO reference timeseries"):
    """Plot a QBO reference series and optional onset dates."""
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(ref["time"], ref, lw=1.6)
    if dates is not None:
        for d in dates:
            ax.axvline(d, color="0.3", lw=1, ls="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("u anomaly (m/s)")
    ax.set_title(title)
    return fig


def plot_reference_onsets(
    ref,
    dates_by_shear,
    groups_by_shear,
    title="QBO reference series and onset dates",
):
    """Plot westerly- and easterly-onset reference-series panels.

    ``dates_by_shear`` and ``groups_by_shear`` are produced by
    ``reference_dates(..., return_groups=True)``. Orange spans and dotted
    lines identify consolidated crossings.
    """
    shear_types = ("westerly", "easterly")
    missing = [
        shear for shear in shear_types
        if shear not in dates_by_shear or shear not in groups_by_shear
    ]
    if missing:
        raise ValueError(f"Missing onset diagnostics for {missing}")
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, sharey=True,
        constrained_layout=True,
    )
    colours = {"westerly": "red", "easterly": "blue"}
    direction_text = {
        "westerly": "negative to positive",
        "easterly": "positive to negative",
    }
    for ax, shear in zip(axes, shear_types):
        dates = np.asarray(dates_by_shear[shear])
        groups = groups_by_shear[shear]
        if len(dates) != len(groups):
            raise ValueError(
                f"{shear} dates and crossing groups must have the same length"
            )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(ref["time"], ref, color="0.2", lw=1.4,
                label="reference wind")
        onset_label, raw_label, span_label = True, True, True
        combined_count = 0
        for date, group in zip(dates, groups):
            group = np.asarray(group)
            if len(group) > 1:
                combined_count += 1
                ax.axvspan(
                    group.min(), group.max(), color="orange", alpha=0.18,
                    label="combined crossing interval" if span_label else None,
                )
                span_label = False
                for raw_date in group:
                    ax.axvline(
                        raw_date, color="darkorange", lw=1, ls=":",
                        label="original crossing" if raw_label else None,
                    )
                    raw_label = False
                ax.plot(
                    group, np.zeros(len(group)), color="darkorange", ls="none",
                    marker="x", ms=6, mew=1.3,
                )
                ax.annotate(
                    "combined", xy=(date, 0), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="darkorange",
                )
            ax.axvline(
                date, color=colours[shear], lw=1, ls="--",
                label="retained onset" if onset_label else None,
            )
            onset_label = False
        ax.text(
            0.01, 0.04, f"Combined crossing groups: {combined_count}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            color="darkorange" if combined_count else "0.35",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75,
                  "pad": 1.5},
        )
        ax.set_ylabel("u anomaly (m/s)")
        ax.set_title(
            f"{shear.capitalize()} onset ({direction_text[shear]} crossing)"
        )
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    return fig
