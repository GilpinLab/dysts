"""
A single-call pipeline for adding a new continuous-time system to the database.

The steps are:

1. Find an integration timestep and a dominant period for the system, and align the
   period to a reference system (Rossler by default) so that the resampled time series
   of all systems in the database oscillate at a comparable rate.
2. Pick an initial condition on the attractor and detect unbounded variables.
3. Estimate the mean and standard deviation of each variable.
4. Estimate the Lyapunov spectrum over an ensemble of initial conditions,
   together with the Kaplan-Yorke dimension, Pesin entropy, correlation dimension and
   multiscale entropy.
5. Independently estimate the largest Lyapunov exponent by direct trajectory divergence.
6. Flag suspicious values and optionally write the entry to the JSON database.

Example:
    >>> from dysts.pipeline import add_system
    >>> from dysts.flows import Liu
    >>> entry, report = add_system(
    ...     Liu,
    ...     parameters={"a": 10, "b": 40, "c": 2.5, "h": 4, "k": 1},
    ...     initial_conditions=[0.5, 0.5, 20.0],
    ...     citation="Liu et al. (2004). Chaos, Solitons & Fractals 22, 1031.",
    ...     description="A Lorenz-like butterfly attractor.",
    ...     doi="10.1016/j.chaos.2004.02.060",
    ...     write=False,
    ... )
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter, welch

from .analysis import (
    find_lyapunov_exponents,
    gp_dim,
    kaplan_yorke_dimension,
    max_lyapunov_exponent,
    sample_initial_conditions,
)
from .base import DATAPATH_CONTINUOUS, BaseDyn, DynSys
from .utils import (
    find_characteristic_timescale,
    find_significant_frequencies,
    jac_fd,
    standardize_ts,
)

# Ratio between the dominant period and the integration timestep used throughout the
# existing database (period / dt is 5000 for essentially every entry).
STEPS_PER_PERIOD = 5000

# Cheaper integrator settings for the exploratory steps (timescale search, alignment,
# boundedness, moments).
FAST_INTEGRATOR = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11}

# Integrator settings for the invariant estimates (Lyapunov spectra, dimensions). An
# explicit high-order method at 1e-10 is typically 20-50x faster than the package default
# (Radau at 1e-12) for non-stiff systems and gives statistically indistinguishable
# estimates; pass ``integrator={"method": "Radau", "rtol": 1e-12, "atol": 1e-12}`` to
# ``add_system`` to use the package's default integrator settings instead.
ACCURATE_INTEGRATOR = {"method": "DOP853", "rtol": 1e-10, "atol": 1e-10}

# Fields that every entry in the continuous-time database carries, in sorted order.
DATABASE_FIELDS = (
    "bifurcation_parameter",
    "citation",
    "correlation_dimension",
    "delay",
    "description",
    "doi",
    "dt",
    "embedding_dimension",
    "hamiltonian",
    "initial_conditions",
    "kaplan_yorke_dimension",
    "lyapunov_spectrum_estimated",
    "maximum_lyapunov_estimated",
    "mean",
    "multiscale_entropy",
    "nonautonomous",
    "parameters",
    "period",
    "pesin_entropy",
    "std",
    "unbounded_indices",
)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def _raw_copy(eq: DynSys) -> DynSys:
    """Return an equivalent system object without any postprocessing map.

    Lyapunov exponents are invariant under smooth invertible changes of coordinates,
    but the postprocessing maps used in the database (e.g. the sine of an angle) are
    not invertible, so the spectrum is computed in the raw coordinates.
    """
    cls = type(eq)
    attrs = {"_rhs": cls.__dict__.get("_rhs", cls._rhs)}
    if cls._jac is not BaseDyn._jac:
        attrs["_jac"] = cls.__dict__.get("_jac", cls._jac)
    for name in ("_rhs", "_jac"):
        # staticmethods must be re-wrapped when copied out of a class dictionary
        if name in attrs and not isinstance(attrs[name], staticmethod):
            attrs[name] = staticmethod(attrs[name])
    raw_cls = type(cls.__name__ + "Raw", (DynSys,), attrs)
    raw = raw_cls(
        metadata_path=None,
        parameters={k: v for k, v in eq.params.items()},
        initial_conditions=np.array(eq.ic),
        embedding_dimension=len(eq.ic),
        dt=eq.dt,
        period=eq.period,
    )
    raw.name = eq.name
    return raw


def make_system(
    system_class: type[DynSys],
    parameters: dict[str, Any],
    initial_conditions: Sequence[float],
    dt: float = 1e-3,
    period: float = 1.0,
    **metadata: Any,
) -> DynSys:
    """Instantiate a system that has no database entry yet."""
    ic = np.asarray(initial_conditions, dtype=float)
    return system_class(
        metadata_path=None,
        parameters=parameters,
        initial_conditions=ic,
        embedding_dimension=len(ic),
        dt=dt,
        period=period,
        **metadata,
    )


# --------------------------------------------------------------------------- timescales


def _count_oscillations(x: np.ndarray) -> int:
    """Count mean crossings of a one-dimensional signal (two per oscillation)."""
    x = x - np.mean(x)
    return int(np.sum(np.diff(np.sign(x)) != 0))


def points_per_oscillation(
    x: np.ndarray, smooth: float = 2.0, min_prominence: float = 0.1
) -> float:
    """
    Number of samples per oscillation of a one-dimensional signal, defined as the
    average spacing between prominent local maxima of the lightly smoothed signal.
    This mimics how one lines up two traces by eye (counting wiggles) and, unlike a
    spectral peak, remains well defined for broadband Lorenz-like signals whose power
    spectrum has no clear maximum.

    Args:
        x: the signal
        smooth: standard deviation (in samples) of the Gaussian smoothing kernel
        min_prominence: minimum peak prominence as a fraction of the signal's
            standard deviation; smaller ripples are not counted as oscillations
    """
    x = np.asarray(x, dtype=float)
    x = gaussian_filter1d(x - np.mean(x), smooth)
    peaks, _ = find_peaks(x, prominence=min_prominence * np.std(x))
    if len(peaks) < 2:
        return float(len(x))
    return float(len(x) / len(peaks))


def points_per_oscillation_spectral(x: np.ndarray, nperseg: int = 2048) -> float:
    """
    Number of samples spanned by the dominant oscillation of a one-dimensional signal,
    from the peak of a Welch power spectrum with parabolic peak interpolation. Used as
    a secondary diagnostic; see :func:`points_per_oscillation`.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    nperseg = int(min(nperseg, len(x) // 4))
    freqs, power = welch(x, fs=1.0, nperseg=nperseg)
    power = power.copy()
    power[freqs < 2.0 / nperseg] = 0.0  # drop the DC bin and its leakage
    k = int(np.argmax(power))
    if 0 < k < len(power) - 1 and power[k] > 0:
        lp = np.log(power[k - 1 : k + 2] + 1e-300)
        denom = lp[0] - 2 * lp[1] + lp[2]
        shift = 0.5 * (lp[0] - lp[2]) / denom if denom != 0 else 0.0
        shift = float(np.clip(shift, -0.5, 0.5))
    else:
        shift = 0.0
    fpeak = (k + shift) * (freqs[1] - freqs[0])
    return float(1.0 / fpeak)


def estimate_initial_timescale(
    eq: DynSys,
    dt: float = 1e-3,
    n: int = 20000,
    max_rounds: int = 8,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Produce a rough (dt, period) guess by integrating at a fixed step and counting
    oscillations, growing or shrinking the integration window until the trajectory
    contains a reasonable number of oscillations.
    """
    sol = None
    for _ in range(max_rounds):
        eq.dt, eq.period = dt, dt * n
        out = eq.make_trajectory(n, resample=False, postprocess=True, **FAST_INTEGRATOR)
        if out is None or np.any(~np.isfinite(out)):
            dt /= 10
            _log(verbose, f"  integration failed, reducing dt to {dt:.3g}")
            continue
        sol = out[n // 5 :]  # drop transient
        counts = [_count_oscillations(row) for row in sol.T]
        n_osc = max(counts) / 2
        if n_osc < 10:
            dt *= 10
            _log(
                verbose,
                f"  only {n_osc:.1f} oscillations seen, increasing dt to {dt:.3g}",
            )
        elif n_osc > n / 20:
            dt /= 10
            _log(verbose, f"  {n_osc:.0f} oscillations seen, decreasing dt to {dt:.3g}")
        else:
            break
    if sol is None:
        raise RuntimeError(f"{eq.name}: could not integrate the system at any timestep")
    period = points_per_oscillation(sol[:, 0]) * dt
    return period / STEPS_PER_PERIOD, period


def refine_timescale(
    eq: DynSys,
    n_iters: int = 3,
    n: int = 6000,
    pts_per_period: int = 100,
    verbose: bool = False,
) -> tuple[list[float], list[float]]:
    """
    Fixed-point refinement of the timestep and period from the power spectrum of the
    resampled trajectory. The result is a starting point for :func:`align_period`.
    """
    all_dt, all_periods = [], []
    for _ in range(n_iters):
        tpts0, sol0 = eq.make_trajectory(
            n,
            resample=True,
            standardize=False,
            return_times=True,
            postprocess=True,
            pts_per_period=pts_per_period,
            **FAST_INTEGRATOR,
        )
        cutoff = n // 6
        tpts0, sol0 = tpts0[cutoff:], standardize_ts(sol0[cutoff:])
        dtval0 = float(np.median(np.diff(tpts0)))

        all_max_freqs = []
        for row in sol0.T:
            freqs_sig = find_significant_frequencies(row)
            if len(freqs_sig) > 0:
                all_max_freqs.append(np.max(freqs_sig))
        max_freq = max(all_max_freqs) if all_max_freqs else 1 / pts_per_period
        dt0 = dtval0 * (1 / max_freq) / 100

        char_times = [float(find_characteristic_timescale(row)) for row in sol0.T]
        period0 = float(np.median(char_times)) * dtval0

        eq.dt, eq.period = float(dt0), float(period0)
        all_dt.append(eq.dt)
        all_periods.append(eq.period)
        _log(verbose, f"  refine: dt={eq.dt:.4g} period={eq.period:.4g}")
    return all_dt, all_periods


def dominant_period_in_samples(
    eq: DynSys, n: int = 12000, pts_per_period: int = 100, coord: int = 0
) -> float:
    """
    Number of resampled points spanned by the dominant oscillation of one coordinate.
    Two systems whose resampled traces "line up" have the same value.
    """
    sol = eq.make_trajectory(
        n,
        resample=True,
        standardize=False,
        postprocess=True,
        pts_per_period=pts_per_period,
        **FAST_INTEGRATOR,
    )
    sol = sol[n // 12 :]
    return points_per_oscillation(sol[:, coord])


def align_period(
    eq: DynSys,
    reference: DynSys | None = None,
    coord: int = 0,
    max_iters: int = 5,
    tol: float = 0.03,
    verbose: bool = False,
) -> float:
    """
    Rescale ``eq.period`` so that the dominant oscillation of coordinate ``coord`` spans
    the same number of resampled points as it does for the reference system. This
    automates the visual comparison of single-variable traces against Rossler.

    Returns:
        The reference value (points per dominant oscillation) that was matched.
    """
    if reference is None:
        from .flows import Rossler

        reference = Rossler()
    target = dominant_period_in_samples(reference, coord=coord)
    _log(verbose, f"  reference {reference.name}: {target:.1f} points per oscillation")
    for _ in range(max_iters):
        current = dominant_period_in_samples(eq, coord=coord)
        factor = current / target
        _log(
            verbose,
            f"  {eq.name}: {current:.1f} points per oscillation -> scaling period by {factor:.3f}",
        )
        if not np.isfinite(factor) or factor <= 0:
            break
        eq.period = float(eq.period * factor)
        eq.dt = float(eq.period / STEPS_PER_PERIOD)
        if abs(factor - 1) < tol:
            break
    return target


# --------------------------------------------------------------------------- attractor


def find_unbounded_indices(
    eq: DynSys, n: int = 6000, pts_per_period: int = 100
) -> list[int]:
    """Indices of variables that drift monotonically (e.g. phases and unwinding angles)."""
    sol = eq.make_trajectory(
        n,
        postprocess=False,
        resample=True,
        standardize=False,
        pts_per_period=pts_per_period,
        **FAST_INTEGRATOR,
    )
    is_unbounded = []
    for row in sol.T:
        smooth = savgol_filter(row, 1001, 3)
        signs = np.sign(np.diff(smooth)).astype(int)
        is_unbounded.append(bool(np.all(signs == signs[0])))
    return [int(i) for i in np.where(is_unbounded)[0]]


def settle_initial_condition(
    eq: DynSys, n: int = 1000, pts_per_period: int = 100
) -> np.ndarray:
    """Integrate away the transient and return a point on the attractor."""
    sol = eq.make_trajectory(
        n,
        resample=True,
        standardize=False,
        postprocess=False,
        pts_per_period=pts_per_period,
        **FAST_INTEGRATOR,
    )
    eq.ic = np.array(sol[-1], dtype=float)
    return eq.ic


def estimate_moments(
    eq: DynSys,
    unbounded_indices: Sequence[int],
    n: int = 20000,
    pts_per_period: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean and standard deviation of each (postprocessed) variable, matching the
    convention used throughout the existing database: unbounded variables such as
    forcing phases are mapped to a bounded domain before the moments are taken.
    """
    sol = eq.make_trajectory(
        n,
        resample=True,
        standardize=False,
        postprocess=True,
        pts_per_period=pts_per_period,
        **FAST_INTEGRATOR,
    )
    mean = np.mean(sol, axis=0)
    std = np.std(sol, axis=0)
    std[std == 0] = 1.0
    return mean, std


# --------------------------------------------------------------------------- invariants


def multiscale_entropy(traj: np.ndarray) -> float:
    """
    Median composite fuzzy multiscale entropy across channels (an upper bound on the
    multivariate multiscale entropy). Requires the optional dependency neurokit2.
    """
    try:
        import neurokit2 as nk
    except ImportError:
        warnings.warn("neurokit2 not installed; multiscale entropy will be NaN.")
        return float("nan")
    opts = {"composite": True, "fuzzy": True}
    traj = standardize_ts(np.asarray(traj))
    if traj.ndim == 1:
        return float(nk.entropy_multiscale(traj, dimension=2, **opts)[0])
    vals = [float(nk.entropy_multiscale(row, dimension=2, **opts)[0]) for row in traj.T]
    return float(np.median(vals))


def mean_jacobian_trace(eq: DynSys, n: int = 20000, pts_per_period: int = 100) -> float:
    """
    Time average of the trace of the Jacobian along a trajectory. For a flow this equals
    the sum of the Lyapunov exponents, which gives a parameter-free consistency check
    on a numerically estimated spectrum.
    """
    tpts, sol = eq.make_trajectory(
        n,
        resample=True,
        return_times=True,
        postprocess=False,
        pts_per_period=pts_per_period,
        **FAST_INTEGRATOR,
    )
    has_jac = type(eq)._jac is not BaseDyn._jac
    traces = []
    for t, X in zip(tpts, sol):
        if has_jac:
            J = np.array(eq.jac(X, t), dtype=float)
        else:
            J = jac_fd(lambda x: np.array(eq.rhs(x, t)), X)
        traces.append(np.trace(J))
    return float(np.mean(traces))


QR_RESOLUTIONS = (500, 1000, 2000, 4000, 8000)


def choose_qr_resolution(
    raw: DynSys,
    trace: float,
    pts_per_trajectory: int = 3000,
    integrator: dict[str, Any] | None = None,
    rel_tol: float = 0.1,
    zero_tol: float = 0.1,
    verbose: bool = False,
) -> int:
    """
    Pick the sampling density (points per period) of the QR Lyapunov estimate by
    increasing it until (i) the sum of the estimated spectrum matches the mean Jacobian
    trace and (ii) one exponent is close to zero, as it must be for a flow. The
    backward-Euler propagation used by ``find_lyapunov_exponents`` is only first-order
    accurate, so spiky or stiff systems need finer steps.
    """
    integrator = dict(ACCURATE_INTEGRATOR if integrator is None else integrator)
    chosen = QR_RESOLUTIONS[-1]
    for ppp in QR_RESOLUTIONS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lyap = find_lyapunov_exponents(
                raw,
                5 * pts_per_trajectory * ppp // 500,
                pts_per_period=ppp,
                **integrator,
            )
        err = abs(np.sum(lyap) - trace) / max(abs(trace), abs(lyap).max(), 1e-12)
        zero_err = np.min(np.abs(lyap)) / max(np.max(lyap), 1e-12)
        _log(
            verbose,
            f"  QR {ppp} pts/period: lyap={np.round(lyap, 4).tolist()} sum={np.sum(lyap):.4g} "
            f"trace={trace:.4g} rel err={err:.3f} zero err={zero_err:.3f}",
        )
        if err < rel_tol and zero_err < zero_tol:
            chosen = ppp
            break
    return chosen


def lyapunov_spectrum_variational(
    eq: DynSys,
    x0: np.ndarray,
    t_total: float,
    dt_renorm: float,
    integrator: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Lyapunov spectrum from the variational equations: the state and a d x d tangent
    matrix are integrated together with the analytic Jacobian by the same adaptive
    high-order solver, and the tangent matrix is re-orthonormalized (QR) every
    ``dt_renorm``. Unlike the first-order backward-Euler propagation in
    ``find_lyapunov_exponents``, its accuracy is set by the solver tolerances, so it
    stays accurate for stiff, spiking and slow-fast systems.
    """
    integrator = dict(ACCURATE_INTEGRATOR if integrator is None else integrator)
    x0 = np.asarray(x0, dtype=float)
    d = len(x0)

    def augmented(t, z):
        x = z[:d]
        v = z[d:].reshape(d, d)
        f = np.asarray(eq.rhs(x, t), dtype=float)
        jac = np.asarray(eq.jac(x, t), dtype=float)
        return np.concatenate([f, (jac @ v).ravel()])

    n_steps = max(int(np.ceil(t_total / dt_renorm)), 1)
    z = np.concatenate([x0, np.eye(d).ravel()])
    t, sums = 0.0, np.zeros(d)
    for _ in range(n_steps):
        sol = solve_ivp(augmented, (t, t + dt_renorm), z, **integrator)
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            return np.full(d, np.nan)
        z = sol.y[:, -1]
        t += dt_renorm
        q, r = np.linalg.qr(z[d:].reshape(d, d))
        sums += np.log(np.abs(np.diag(r)))
        z = np.concatenate([z[:d], q.ravel()])
    return np.sort(sums / (n_steps * dt_renorm))[::-1]


def estimate_invariants(
    eq: DynSys,
    n_samples: int = 20,
    pts_per_trajectory: int = 3000,
    divide_dt: int = 10,
    n_gp: int = 4000,
    integrator: dict[str, Any] | None = None,
    qr_pts_per_period: int | None = None,
    spectrum_method: str = "auto",
    variational_periods: float = 150.0,
    renorms_per_period: int = 5,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Ensemble estimate of the Lyapunov spectrum, Kaplan-Yorke dimension, Pesin entropy,
    correlation dimension and multiscale entropy. Estimates are averaged with weights
    that favour samples whose middle exponent is closest to zero.

    ``spectrum_method`` is "variational" (tangent dynamics integrated with the analytic
    Jacobian, see :func:`lyapunov_spectrum_variational`), "qr" (the package's
    backward-Euler QR, ``find_lyapunov_exponents``, with adaptive step resolution), or
    "auto", which uses "variational" whenever the system has an analytic Jacobian.
    """
    integrator = dict(ACCURATE_INTEGRATOR if integrator is None else integrator)
    raw = _raw_copy(eq)
    ic_backup = np.array(eq.ic)
    trace = mean_jacobian_trace(raw)
    sample_pts = sample_initial_conditions(raw, n_samples, **integrator)
    est_lyap, est_corr, est_mse = [], [], []
    raw.dt /= divide_dt
    if spectrum_method == "auto":
        spectrum_method = "variational" if type(eq)._jac is not BaseDyn._jac else "qr"
    _log(verbose, f"  spectrum method: {spectrum_method}")
    if spectrum_method == "qr" and qr_pts_per_period is None:
        raw.ic = sample_pts[0]
        qr_pts_per_period = choose_qr_resolution(
            raw, trace, pts_per_trajectory, integrator=integrator, verbose=verbose
        )
        _log(
            verbose,
            f"  using {qr_pts_per_period} points per period for the QR estimate",
        )
    if spectrum_method == "qr":
        n_qr = 5 * pts_per_trajectory * qr_pts_per_period // 500
    else:
        # Renormalize often enough that the most contracting direction shrinks by at most
        # ~e^2 between QR steps; otherwise the tangent vectors become numerically parallel
        # and the most negative exponents are lost to round-off.
        dt_renorm = eq.period / renorms_per_period
        if abs(trace) > 0:
            dt_renorm = min(dt_renorm, 2.0 / abs(trace))
        _log(verbose, f"  renormalization interval {dt_renorm:.4g}")
    for j, pt in enumerate(sample_pts):
        _log(verbose, f"  sample {j + 1}/{n_samples}")
        raw.ic = pt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if spectrum_method == "variational":
                lyap = lyapunov_spectrum_variational(
                    raw,
                    pt,
                    variational_periods * eq.period,
                    dt_renorm,
                    integrator=integrator,
                )
            else:
                lyap = find_lyapunov_exponents(
                    raw, n_qr, pts_per_period=qr_pts_per_period, **integrator
                )
            # dimensions and entropy are computed on the postprocessed (bounded)
            # trajectory, consistent with the rest of the database
            eq.ic = pt
            sol = eq.make_trajectory(
                3 * pts_per_trajectory,
                resample=True,
                pts_per_period=100,
                postprocess=True,
                **integrator,
            )
        if sol is None or not np.all(np.isfinite(lyap)):
            continue
        est_lyap.append(lyap)
        try:
            idx = np.linspace(0, len(sol) - 1, min(n_gp, len(sol))).astype(int)
            est_corr.append(gp_dim(sol[idx]))
        except Exception:
            est_corr.append(np.nan)
        est_mse.append(multiscale_entropy(sol[:pts_per_trajectory]))
    raw.dt *= divide_dt
    eq.ic = ic_backup

    est_lyap = np.array(est_lyap)
    weights = np.exp(-np.min(np.abs(est_lyap), axis=1) * 1000)
    weights /= np.sum(weights)
    lyap = np.sum(weights[:, None] * est_lyap, axis=0)
    est_corr = np.array(est_corr, dtype=float)
    good = np.isfinite(est_corr)
    corr = (
        float(np.sum(weights[good] * est_corr[good]) / np.sum(weights[good]))
        if np.any(good)
        else float("nan")
    )
    return {
        "lyapunov_spectrum_estimated": [float(v) for v in lyap],
        "kaplan_yorke_dimension": float(
            np.sum(weights * np.array([kaplan_yorke_dimension(v) for v in est_lyap]))
        ),
        "pesin_entropy": float(
            np.sum(weights * np.array([np.sum(v[v > 0]) for v in est_lyap]))
        ),
        "correlation_dimension": corr,
        "multiscale_entropy": float(np.sum(weights * np.array(est_mse))),
        "_lyapunov_samples": est_lyap.tolist(),
        "_jacobian_trace_mean": trace,
        "_qr_pts_per_period": None
        if qr_pts_per_period is None
        else int(qr_pts_per_period),
        "_spectrum_method": spectrum_method,
    }


def estimate_max_lyapunov_direct(
    eq: DynSys,
    n_samples: int = 40,
    traj_length: int = 3000,
    max_walltime: float = 120,
    integrator: dict[str, Any] | None = None,
) -> float:
    """
    Largest Lyapunov exponent from the divergence of nearby trajectories.

    Only the integration ``method`` is forwarded: ``max_lyapunov_exponent`` uses the
    names ``rtol`` and ``atol`` for the separation thresholds of the divergence fit,
    so solver tolerances cannot be passed through it and the defaults (1e-12) are used.
    """
    integrator = dict(ACCURATE_INTEGRATOR if integrator is None else integrator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = max_lyapunov_exponent(
            eq,
            max_walltime=max_walltime,
            n_samples=n_samples,
            traj_length=traj_length,
            method=integrator.get("method", "Radau"),
        )
    return float("nan") if val is None else float(val)


# --------------------------------------------------------------------------- checks


def flag_suspicious(entry: dict[str, Any]) -> list[str]:
    """Return human-readable warnings about inconsistent invariants."""
    flags = []
    lyap = np.array(entry["lyapunov_spectrum_estimated"])
    d = entry["embedding_dimension"]
    lmax_qr = float(np.max(lyap))
    lmax_direct = entry["maximum_lyapunov_estimated"]
    if lmax_qr <= 0:
        flags.append(f"largest spectrum exponent is not positive ({lmax_qr:.3g})")
    if np.isfinite(lmax_direct):
        if lmax_direct <= 0:
            flags.append(
                f"direct Lyapunov estimate is not positive ({lmax_direct:.3g})"
            )
        elif lmax_qr > 0:
            ratio = lmax_direct / lmax_qr
            if ratio < 0.5 or ratio > 2.0:
                flags.append(
                    f"direct ({lmax_direct:.3g}) and spectrum ({lmax_qr:.3g}) Lyapunov estimates disagree by {ratio:.2f}x"
                )
    else:
        flags.append("direct Lyapunov estimate failed")
    if d >= 3 and lmax_qr > 0:
        middle = np.sort(np.abs(lyap))[0]
        if middle > 0.15 * lmax_qr:
            flags.append(
                f"no near-zero exponent: smallest |lambda| = {middle:.3g} vs lambda_max = {lmax_qr:.3g}"
            )
    ky = entry["kaplan_yorke_dimension"]
    cd = entry["correlation_dimension"]
    if entry.get("hamiltonian", False):
        # conservative systems: Kaplan-Yorke dimension equals the phase-space dimension
        # and the correlation dimension of a chaotic sea need not match it
        if not np.isfinite(cd) or not (1.0 <= cd <= d + 0.3):
            flags.append(f"correlation dimension {cd:.3g} outside [1, {d}]")
        return flags
    if not (1.0 < ky < d):
        flags.append(f"Kaplan-Yorke dimension {ky:.3g} outside (1, {d})")
    if not np.isfinite(cd) or not (1.0 <= cd <= d + 0.3):
        flags.append(f"correlation dimension {cd:.3g} outside [1, {d}]")
    elif abs(cd - ky) > 0.5:
        flags.append(
            f"correlation dimension {cd:.3g} and Kaplan-Yorke dimension {ky:.3g} differ by more than 0.5"
        )
    return flags


# --------------------------------------------------------------------------- plotting


def plot_alignment(
    eq: DynSys, reference: DynSys | None = None, n: int = 1000, ax=None, coord: int = 0
):
    """Overlay the standardized first coordinate of ``eq`` and the reference system."""
    import matplotlib.pyplot as plt

    if reference is None:
        from .flows import Rossler

        reference = Rossler()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.5))
    sol1 = eq.make_trajectory(n, resample=True, standardize=True)
    sol2 = reference.make_trajectory(n, resample=True, standardize=True)
    ax.plot(standardize_ts(sol2)[:, coord], color="0.6", lw=1, label=reference.name)
    ax.plot(standardize_ts(sol1)[:, coord], color="C0", lw=1, label=eq.name)
    ax.set_xlabel("resampled time index")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_attractor(eq: DynSys, n: int = 5000, ax=None):
    """Three-dimensional projection of the (postprocessed) attractor."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
    sol = eq.make_trajectory(n, resample=True, standardize=False, postprocess=True)
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], lw=0.4, color="k")
    ax.set_title(eq.name)
    ax.set_axis_off()
    return ax


# --------------------------------------------------------------------------- pipeline


def add_system(
    system_class: type[DynSys],
    parameters: dict[str, Any],
    initial_conditions: Sequence[float],
    citation: str,
    description: str,
    doi: str | None = None,
    bifurcation_parameter: str | None = None,
    hamiltonian: bool = False,
    nonautonomous: bool = False,
    reference: DynSys | None = None,
    database_path: str = DATAPATH_CONTINUOUS,
    write: bool = True,
    n_ic_samples: int = 20,
    n_direct_samples: int = 40,
    integrator: dict[str, Any] | None = None,
    verbose: bool = True,
    plot_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Compute every database field for a new system and (optionally) write it to the
    continuous-time database.

    Args:
        system_class: A ``DynSys`` subclass with ``_rhs`` (and preferably ``_jac``).
        parameters: Parameter values for the chaotic regime.
        initial_conditions: A starting point in the basin of the chaotic attractor.
        citation, description, doi, bifurcation_parameter: Metadata strings.
        hamiltonian, nonautonomous: Metadata flags.
        reference: System whose dominant timescale the new system is aligned to
            (Rossler by default).
        database_path: JSON database to update.
        write: Whether to write the entry into ``database_path``.
        n_ic_samples: Ensemble size for the QR Lyapunov / dimension estimates.
        n_direct_samples: Ensemble size for the direct largest-Lyapunov estimate.
        integrator: solve_ivp settings for the invariant estimates (see
            ``ACCURATE_INTEGRATOR``).
        verbose: Print progress.
        plot_path: If given, save a figure with the attractor and the aligned traces.

    Returns:
        entry: The database entry (JSON-serialisable dictionary).
        report: Diagnostics: timescale history, per-sample Lyapunov spectra and flags.
    """
    eq = make_system(system_class, parameters, initial_conditions)
    report: dict[str, Any] = {"name": eq.name}

    _log(verbose, f"[{eq.name}] estimating timescale")
    eq.dt, eq.period = estimate_initial_timescale(eq, verbose=verbose)
    _log(verbose, f"  initial guess dt={eq.dt:.4g} period={eq.period:.4g}")
    all_dt, all_periods = refine_timescale(eq, verbose=verbose)
    report["timescale_history"] = {"dt": all_dt, "period": all_periods}

    _log(verbose, f"[{eq.name}] aligning period to reference")
    report["alignment_target"] = align_period(eq, reference=reference, verbose=verbose)
    report["alignment_achieved"] = dominant_period_in_samples(eq)
    report["period_before_alignment"] = all_periods[-1]
    if abs(report["alignment_achieved"] / report["alignment_target"] - 1) > 0.25:
        report.setdefault("flags", []).append(
            f"alignment off by {report['alignment_achieved'] / report['alignment_target']:.2f}x"
        )
    _log(verbose, f"  final dt={eq.dt:.4g} period={eq.period:.4g}")

    _log(verbose, f"[{eq.name}] settling initial condition and checking boundedness")
    settle_initial_condition(eq)
    unbounded = find_unbounded_indices(eq)
    eq.unbounded_indices = unbounded
    mean, std = estimate_moments(eq, unbounded)
    _log(verbose, f"  ic={np.round(eq.ic, 4).tolist()} unbounded={unbounded}")

    _log(
        verbose,
        f"[{eq.name}] Lyapunov spectrum and dimensions ({n_ic_samples} samples)",
    )
    invariants = estimate_invariants(
        eq, n_samples=n_ic_samples, integrator=integrator, verbose=verbose
    )
    report["lyapunov_samples"] = invariants.pop("_lyapunov_samples")
    report["jacobian_trace_mean"] = invariants.pop("_jacobian_trace_mean")
    report["qr_pts_per_period"] = invariants.pop("_qr_pts_per_period")
    report["spectrum_method"] = invariants.pop("_spectrum_method")
    lyap_sum = float(np.sum(invariants["lyapunov_spectrum_estimated"]))
    lyap_max = float(np.max(invariants["lyapunov_spectrum_estimated"]))
    if hamiltonian:
        # conservative: the exponents should sum to zero
        if abs(lyap_sum) > 0.1 * max(abs(lyap_max), 1e-12):
            report.setdefault("flags", []).append(
                f"conservative system but Lyapunov exponents sum to {lyap_sum:.3g} "
                f"(largest {lyap_max:.3g})"
            )
        trace_err = 0.0
    else:
        trace_err = abs(lyap_sum - report["jacobian_trace_mean"]) / max(
            abs(report["jacobian_trace_mean"]), 1e-12
        )
    if trace_err > 0.1:
        report.setdefault("flags", []).append(
            f"sum of Lyapunov exponents {lyap_sum:.4g} differs from mean Jacobian trace "
            f"{report['jacobian_trace_mean']:.4g} by {100 * trace_err:.0f}%"
        )

    _log(
        verbose,
        f"[{eq.name}] direct largest Lyapunov exponent ({n_direct_samples} samples)",
    )
    ic_backup = np.array(eq.ic)
    lmax_direct = estimate_max_lyapunov_direct(
        eq, n_samples=n_direct_samples, integrator=integrator
    )
    eq.ic = ic_backup

    entry: dict[str, Any] = {
        "bifurcation_parameter": bifurcation_parameter,
        "citation": citation,
        "delay": False,
        "description": description,
        "doi": doi,
        "dt": float(eq.dt),
        "embedding_dimension": int(len(eq.ic)),
        "hamiltonian": bool(hamiltonian),
        "initial_conditions": [float(v) for v in eq.ic],
        "maximum_lyapunov_estimated": lmax_direct,
        "mean": [float(v) for v in mean],
        "nonautonomous": bool(nonautonomous),
        "parameters": {k: _jsonable(v) for k, v in parameters.items()},
        "period": float(eq.period),
        "std": [float(v) for v in std],
        "unbounded_indices": unbounded,
    }
    entry.update(invariants)
    entry = {k: entry[k] for k in sorted(entry)}
    assert set(entry) == set(DATABASE_FIELDS), set(entry) ^ set(DATABASE_FIELDS)

    report["flags"] = report.get("flags", []) + flag_suspicious(entry)
    report["lyapunov_qr_max"] = float(np.max(entry["lyapunov_spectrum_estimated"]))
    _log(
        verbose,
        f"[{eq.name}] lyap={np.round(entry['lyapunov_spectrum_estimated'], 4).tolist()} "
        f"direct={lmax_direct:.4g} D_KY={entry['kaplan_yorke_dimension']:.3f} "
        f"D_corr={entry['correlation_dimension']:.3f} MSE={entry['multiscale_entropy']:.3f} "
        f"sum(lyap)={np.sum(entry['lyapunov_spectrum_estimated']):.4g} "
        f"trace={report['jacobian_trace_mean']:.4g}",
    )
    for f in report["flags"]:
        _log(verbose, f"  FLAG: {f}")

    if plot_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        plot_attractor(eq, ax=ax1)
        ax2 = fig.add_subplot(1, 3, (2, 3))
        plot_alignment(eq, reference=reference, ax=ax2)
        ax2.set_title(
            f"{eq.name}: {report['alignment_achieved']:.0f} pts/oscillation "
            f"(reference {report['alignment_target']:.0f}); period={eq.period:.4g}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

    if write:
        write_entry(eq.name, entry, database_path)
        _log(verbose, f"[{eq.name}] written to {database_path}")
    return entry, report


def write_entry(
    name: str, entry: dict[str, Any], database_path: str = DATAPATH_CONTINUOUS
) -> None:
    """
    Insert or replace one entry in the JSON database. Systems are kept in sorted order
    and the file follows the existing formatting (two-space indent; within an entry the
    DOI follows the citation and the remaining fields are alphabetical).
    """
    with open(database_path, "r") as f:
        data = json.load(f)
    head = ["bifurcation_parameter", "citation", "doi"]
    ordered = {k: entry[k] for k in head if k in entry}
    ordered.update({k: entry[k] for k in sorted(entry) if k not in head})
    data[name] = ordered
    data = {k: data[k] for k in sorted(data)}
    with open(database_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v
