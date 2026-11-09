"""
A single-call pipeline for adding a new continuous-time system to the database.

The steps are:

1. Find an integration timestep and a dominant period for the system, and align the
   period to a reference system (Rossler by default) so that the resampled time series
   of all systems in the database oscillate at a comparable rate.
2. Pick an initial condition on the attractor and detect unbounded variables.
3. Estimate the mean and standard deviation of each variable.
4. Estimate the Lyapunov spectrum with a standard error from several long runs on the
   attractor (variational equations, see :func:`dysts.analysis.lyapunov_spectrum`),
   together with the Kaplan-Yorke dimension, Pesin entropy, correlation dimension and
   multiscale entropy.
5. Cross-check the largest exponent with the Jacobian-free clone method, and the sum
   of the exponents against the mean trace of the Jacobian.
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter, welch

from .analysis import (
    LYAPUNOV_INTEGRATOR,
    gp_dim,
    kaplan_yorke_dimension,
    lyapunov_spectrum,
    sample_initial_conditions,
)
from .base import DATAPATH_CONTINUOUS, DynSys
from .utils import (
    find_characteristic_timescale,
    find_significant_frequencies,
    standardize_ts,
)

# Ratio between the dominant period and the integration timestep used throughout the
# existing database (period / dt is 5000 for essentially every entry).
STEPS_PER_PERIOD = 5000

# Cheaper integrator settings for the exploratory steps (timescale search, alignment,
# boundedness, moments).
FAST_INTEGRATOR = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11}

# Integrator settings for the trajectories behind the invariant estimates (dimensions,
# entropy), shared with the variational Lyapunov spectrum. An explicit high-order method
# at 1e-10 is typically 20-50x faster than the package default (Radau at 1e-12) for
# non-stiff systems and gives statistically indistinguishable estimates; stiff systems
# can pass e.g. ``integrator={"method": "LSODA", "rtol": 1e-10, "atol": 1e-12}`` to
# ``add_system``.
ACCURATE_INTEGRATOR = dict(LYAPUNOV_INTEGRATOR)

# Fields that every entry in the continuous-time database carries, in sorted order.
DATABASE_FIELDS = (
    "bifurcation_parameter",
    "citation",
    "correlation_dimension",
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


def lyapunov_spectrum_variational(
    eq: DynSys,
    x0: np.ndarray,
    t_total: float,
    dt_renorm: float,
    integrator: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Deprecated: use :func:`dysts.analysis.lyapunov_spectrum`.

    Variational spectrum from a single run of length ``t_total`` starting at ``x0``,
    re-orthonormalized every ``dt_renorm``.
    """
    warnings.warn(
        "dysts.pipeline.lyapunov_spectrum_variational is deprecated; use "
        "dysts.analysis.lyapunov_spectrum.",
        DeprecationWarning,
        stacklevel=2,
    )
    n_periods = t_total / eq.period
    return lyapunov_spectrum(
        eq,
        n_runs=1,
        n_periods=n_periods,
        transient_periods=0.0,
        block_periods=min(25.0, n_periods),
        dt_renorm=dt_renorm,
        initial_conditions=x0,
        integrator=integrator,
    )


def estimate_invariants(
    eq: DynSys,
    n_runs: int = 4,
    n_periods: float = 1000.0,
    spectrum_method: str = "variational",
    n_dimension_samples: int = 20,
    pts_per_trajectory: int = 3000,
    n_gp: int = 4000,
    integrator: dict[str, Any] | None = None,
    max_walltime: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Lyapunov spectrum, largest Lyapunov exponent, Kaplan-Yorke dimension, Pesin entropy,
    correlation dimension and multiscale entropy.

    The spectrum comes from :func:`dysts.analysis.lyapunov_spectrum` with ``n_runs`` runs
    of ``n_periods`` each. ``spectrum_method`` is "variational" (tangent dynamics with the
    analytic Jacobian, or finite differences when there is none) or "clones" (finite
    perturbations, for right-hand sides with discontinuities). The Kaplan-Yorke dimension
    and Pesin entropy follow from the mean spectrum. The correlation dimension and
    multiscale entropy are averaged over ``n_dimension_samples`` postprocessed
    trajectories that start from random points on the attractor.

    ``integrator`` sets the ``solve_ivp`` options for every integration; by default the
    spectrum uses the method's own defaults (``LYAPUNOV_INTEGRATOR`` or
    ``CLONE_INTEGRATOR``) and the trajectories use ``ACCURATE_INTEGRATOR``.
    """
    traj_integrator = dict(ACCURATE_INTEGRATOR if integrator is None else integrator)
    _log(verbose, f"  spectrum method: {spectrum_method}")
    lyap = lyapunov_spectrum(
        eq,
        n_runs=n_runs,
        n_periods=n_periods,
        method=spectrum_method,
        integrator=integrator,
        max_walltime=max_walltime,
        full_output=True,
    )
    spectrum = np.asarray(lyap["spectrum"])
    _log(
        verbose,
        f"  spectrum {np.round(spectrum, 4).tolist()} +/- "
        f"{np.round(lyap['standard_error'], 4).tolist()} "
        f"({lyap['periods_integrated']:.0f} periods, {lyap['jacobian']} Jacobian)",
    )

    ic_backup = np.array(eq.ic)
    sample_pts = sample_initial_conditions(eq, n_dimension_samples, **traj_integrator)
    est_corr, est_mse = [], []
    for pt in sample_pts:
        eq.ic = pt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = eq.make_trajectory(
                3 * pts_per_trajectory,
                resample=True,
                pts_per_period=100,
                postprocess=True,
                **traj_integrator,
            )
        if sol is None or not np.all(np.isfinite(sol)):
            continue
        try:
            idx = np.linspace(0, len(sol) - 1, min(n_gp, len(sol))).astype(int)
            est_corr.append(gp_dim(sol[idx]))
        except Exception:
            est_corr.append(np.nan)
        est_mse.append(multiscale_entropy(sol[:pts_per_trajectory]))
    eq.ic = ic_backup

    finite = lambda v: np.asarray(v, dtype=float)[np.isfinite(v)]  # noqa: E731
    corr, mse = finite(est_corr), finite(est_mse)
    return {
        "lyapunov_spectrum_estimated": [float(v) for v in spectrum],
        "maximum_lyapunov_estimated": float(spectrum[0]),
        "kaplan_yorke_dimension": float(kaplan_yorke_dimension(spectrum)),
        "pesin_entropy": float(np.sum(spectrum[spectrum > 0])),
        "correlation_dimension": float(np.mean(corr)) if len(corr) else float("nan"),
        "multiscale_entropy": float(np.mean(mse)) if len(mse) else float("nan"),
        "_lyapunov": lyap,
    }


# --------------------------------------------------------------------------- checks


def flag_suspicious(entry: dict[str, Any]) -> list[str]:
    """
    Return human-readable warnings about inconsistent invariants in a database entry.
    The checks are: a positive largest exponent, one near-zero exponent (as for any
    flow), and Kaplan-Yorke and correlation dimensions inside the phase space and
    within 0.5 of each other. Cross-checks that need the dynamics (the clone-method
    spectrum and the mean Jacobian trace) are made by :func:`add_system`.
    """
    flags = []
    lyap = np.array(entry["lyapunov_spectrum_estimated"])
    d = entry["embedding_dimension"]
    lmax = float(np.max(lyap))
    if lmax <= 0:
        flags.append(f"largest Lyapunov exponent is not positive ({lmax:.3g})")
    if d >= 3 and lmax > 0:
        middle = np.sort(np.abs(lyap))[0]
        if middle > 0.15 * lmax:
            flags.append(
                f"no near-zero exponent: smallest |lambda| = {middle:.3g} vs lambda_max = {lmax:.3g}"
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
    n_runs: int = 4,
    n_periods: float = 1000.0,
    spectrum_method: str = "variational",
    crosscheck_periods: float = 200.0,
    integrator: dict[str, Any] | None = None,
    max_walltime: float | None = None,
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
        n_runs, n_periods: Number and length (in periods) of the Lyapunov runs, see
            :func:`dysts.analysis.lyapunov_spectrum`.
        spectrum_method: "variational" (default; uses the analytic Jacobian, or finite
            differences when the class has none) or "clones" (finite perturbations).
            Use "clones" when the right-hand side is discontinuous (``sign``,
            Heaviside or switching terms).
        crosscheck_periods: Length of each of the two runs of the other method, whose
            largest exponent is compared with the main estimate (0 to skip).
        integrator: solve_ivp settings for the invariant estimates (see
            ``ACCURATE_INTEGRATOR``); pass LSODA or Radau for stiff systems.
        max_walltime: Wall-clock limit in seconds for each Lyapunov run.
        verbose: Print progress.
        plot_path: If given, save a figure with the attractor and the aligned traces.

    Returns:
        entry: The database entry (JSON-serialisable dictionary).
        report: Diagnostics: timescale history, the full Lyapunov output (standard
            errors, per-run values, block averages), the cross-check and flags.
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

    _log(verbose, f"[{eq.name}] Lyapunov spectrum and dimensions")
    invariants = estimate_invariants(
        eq,
        n_runs=n_runs,
        n_periods=n_periods,
        spectrum_method=spectrum_method,
        integrator=integrator,
        max_walltime=max_walltime,
        verbose=verbose,
    )
    lyap = invariants.pop("_lyapunov")
    report["lyapunov"] = lyap
    report["jacobian_trace_mean"] = lyap["mean_jacobian_trace"]
    spectrum = np.asarray(lyap["spectrum"])
    se = np.asarray(lyap["standard_error"])
    lyap_sum, lyap_max = float(np.sum(spectrum)), float(spectrum[0])
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
    by_run = np.asarray(lyap["max_exponent_by_run"])
    if len(by_run) > 1 and np.ptp(by_run) > 0.25 * abs(lyap_max) + 6 * se[0]:
        report.setdefault("flags", []).append(
            f"largest exponent varies between runs ({np.round(by_run, 4).tolist()}): "
            "coexisting attractors, a long transient or intermittency"
        )

    if crosscheck_periods > 0:
        other = "clones" if spectrum_method == "variational" else "variational"
        _log(verbose, f"[{eq.name}] cross-check with the {other} method")
        check = lyapunov_spectrum(
            eq,
            n_runs=2,
            n_periods=crosscheck_periods,
            method=other,
            integrator=integrator if other == "variational" else None,
            max_walltime=max_walltime,
            full_output=True,
        )
        report["crosscheck"] = check
        c_max, c_se = float(check["spectrum"][0]), float(check["standard_error"][0])
        tol = 3 * np.hypot(se[0], c_se) + 0.1 * max(abs(lyap_max), abs(c_max))
        if not np.isfinite(c_max) or abs(c_max - lyap_max) > tol:
            hint = (
                "; if the right-hand side is discontinuous, use spectrum_method='clones'"
                if other == "clones"
                else ""
            )
            report.setdefault("flags", []).append(
                f"{spectrum_method} ({lyap_max:.4g}) and {other} ({c_max:.4g}) largest "
                f"exponents disagree{hint}"
            )

    entry: dict[str, Any] = {
        "bifurcation_parameter": bifurcation_parameter,
        "citation": citation,
        "description": description,
        "doi": doi,
        "dt": float(eq.dt),
        "embedding_dimension": int(len(eq.ic)),
        "hamiltonian": bool(hamiltonian),
        "initial_conditions": [float(v) for v in eq.ic],
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
    _log(
        verbose,
        f"[{eq.name}] lyap={np.round(entry['lyapunov_spectrum_estimated'], 4).tolist()} "
        f"+/- {np.round(se, 4).tolist()} D_KY={entry['kaplan_yorke_dimension']:.3f} "
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
