"""Calculate QBO period, amplitude and variance explained with two EOFs."""

import numpy as np


def qbo_eof(u, p, p_min=10.0, p_max=100.0, period_mode="mean",
            return_diagnostics=False):
    """Return three QBO metrics from a pressure-by-time zonal-wind array.

    ``u`` has shape (pressure, time), with wind in m/s and regular monthly
    samples. ``p`` contains pressure in hPa. The result contains ``period``
    in months, mean ``amplitude`` in m/s, and ``variance_explained`` as a
    fraction. Set ``period_mode`` to ``"median"`` for an estimate less
    sensitive to brief interruptions. The method follows the rank-2,
    log-pressure-weighted PCA calculation by Kevin DallaSanta. Set
    ``return_diagnostics=True`` to also return the two physical EOF patterns,
    standardised PCs, phase, phase speed and monthly amplitude.
    """
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    if u.ndim != 2:
        raise ValueError("u must have shape (pressure, time)")
    if p.ndim != 1 or u.shape[0] != p.size or p.size < 2:
        raise ValueError("p must contain at least two levels matching u")
    if u.shape[1] < 3:
        raise ValueError("u must contain at least three time samples")
    if not np.all(np.isfinite(u)):
        raise ValueError("u must contain only finite values")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("p must contain only positive, finite pressures")
    if not (np.all(np.diff(p) > 0) or np.all(np.diff(p) < 0)):
        raise ValueError("p must be strictly monotonic")
    if not np.isfinite([p_min, p_max]).all() or min(p_min, p_max) <= 0:
        raise ValueError("pressure bounds must be positive and finite")
    if p_min == p_max:
        raise ValueError("p_min and p_max must differ")
    if period_mode not in ("mean", "median"):
        raise ValueError("period_mode must be 'mean' or 'median'")

    if p[0] > p[-1]:
        p = p[::-1]
        u = u[::-1]
    p_min, p_max = sorted((p_min, p_max))

    # Log-pressure spacing approximates the vertical thickness of each level.
    weights = np.abs(np.gradient(np.log(p)))
    selected = (p >= p_min) & (p <= p_max)
    if selected.sum() < 2:
        raise ValueError("At least two pressure levels must lie within the bounds")
    weights = weights[selected]
    weights /= weights.sum()
    u = u[selected]
    anomalies = u - u.mean(axis=1, keepdims=True)

    eof_weighted, singular_values, pcs = np.linalg.svd(
        anomalies * np.sqrt(weights[:, None]), full_matrices=False)
    variance = singular_values ** 2
    if variance.sum() == 0:
        raise ValueError("u has no time variation within the pressure bounds")
    if singular_values[1] <= (np.finfo(float).eps * max(anomalies.shape)
                              * singular_values[0]):
        raise ValueError("u needs two independent EOF modes to estimate a period")
    variance_by_mode = variance[:2] / variance.sum()
    variance_explained = variance_by_mode.sum()

    # Normalize physical EOFs to a peak of +1, then scale PCs to m/s.
    physical_eofs = eof_weighted[:, :2] / np.sqrt(weights[:, None])
    peak_index = np.argmax(np.abs(physical_eofs), axis=0)
    peak_value = physical_eofs[peak_index, np.arange(2)]
    pcs = pcs[:2] * (peak_value * singular_values[:2])[:, None]
    amplitude_by_time = np.hypot(pcs[0], pcs[1])
    amplitude = amplitude_by_time.mean()

    pc_std = pcs.std(axis=1, ddof=1)
    if np.any(pc_std == 0):
        raise ValueError("Both EOF modes need time variation to estimate a period")
    pc1, pc2 = pcs / pc_std[:, None]
    radius_squared = pc1 ** 2 + pc2 ** 2
    if np.any(radius_squared == 0):
        raise ValueError("EOF phase is undefined when both PCs are zero")
    phase_speed = (
        pc2 * np.gradient(pc1) - pc1 * np.gradient(pc2)
    ) / radius_squared
    phase_sign = -1 if phase_speed.mean() < 0 else 1
    phase_speed *= phase_sign
    mean_speed = (phase_speed.mean() if period_mode == "mean"
                  else np.median(phase_speed))
    if mean_speed <= 0:
        raise ValueError("EOF phase speed does not define a positive period")

    result = {
        "period": float(2 * np.pi / mean_speed),
        "amplitude": float(amplitude),
        "variance_explained": float(variance_explained),
    }
    if return_diagnostics:
        result.update({
            "pressure": p[selected].copy(),
            "eof_patterns": physical_eofs / peak_value[None, :] * pc_std[None, :],
            "pcs": np.vstack((pc1, pc2)),
            "phase": phase_sign * np.unwrap(np.arctan2(pc1, pc2)),
            "phase_speed": phase_speed,
            "amplitude_by_time": amplitude_by_time,
            "variance_by_mode": variance_by_mode,
        })
    return result
