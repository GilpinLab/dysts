"""Utilities for integration"""

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import resample

from .native_utils import has_module

if has_module("sdeint"):
    from sdeint import itoint


def cast_to_numpy(x, singleton_scalar=False):
    if singleton_scalar:
        scalar_cast = np.array([x])
    else:
        scalar_cast = x
    return np.array(x) if not np.isscalar(x) else scalar_cast


def parabolic(f, x):
    """
    Quadratic interpolation in order to estimate the location of a maximum
    https://gist.github.com/endolith/255291

    Args:
        f (ndarray): a vector a samples
        x (int): an index on the vector

    Returns:
        (vx, vy): the vertex coordinates of  a parabola passing through x
            and its neighbors
    """
    xv = 1 / 2.0 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4.0 * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)


def freq_from_autocorr(sig, fs=1):
    """
    Estimate frequency using autocorrelation

    Args:
        sig (ndarray): A univariate signal
        fs (int): The sampling frequency

    Returns:
        out (float): The dominant frequency

    References:
        Modified from the following
        https://gist.github.com/endolith/255291
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = np.correlate(sig, sig, mode="full")
    corr = corr[len(corr) // 2 :]

    # Find the first low point
    d = np.diff(corr)
    start = np.nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    out = fs / px
    return out


def resample_timepoints(model, ic, tpts, cutoff=10000, pts_per_period=100):
    """
    Given a differential equation, initial condition, and a set of
    integration points, determine a new set of timepoints that
    scales to the periodicity of the model

    Args:
        model (callable): the right hand side of a set of ODEs
        ic (list): the initial conditions
        tpts (array): the timepoints over which to integrate
        pts_per_period (int): the number of timepoints to sample in
            each period

    Returns:
        new_timepoints (ndarray): The resampled timepoints
    """
    dt = tpts[1] - tpts[0]
    samp = integrate_dyn(model, ic, tpts)[0]
    period = dt * (1 / freq_from_autocorr(samp[:cutoff], 1))
    num_periods = len(tpts) // pts_per_period
    new_timepoints = np.linspace(0, num_periods * period, num_periods * pts_per_period)
    return new_timepoints


def integrate_dyn(
    f: Callable,
    ic: np.ndarray,
    tvals: np.ndarray,
    noise: float | np.ndarray = 0.0,
    dtval=None,
    **kwargs,
):
    """
    Given the RHS of a dynamical system, integrate the system
    noise > 0 requires the Python library sdeint (assumes Brownian noise)

    Args:
        f (callable): The right hand side of a system of ODEs.
        ic (ndarray): the initial conditions
        tvals (ndarray): times points at which to evaluate the solution
        noise (float or iterable): The amplitude of the Langevin forcing term. If a
            vector is passed, this will be different for each dynamical variable
        dtval (float): The starting integration timestep. This will be the exact timestep for
            fixed-step integrators, or stochastic integration.
        kwargs (dict): Arguments passed to scipy.integrate.solve_ivp.

    Returns:
        sol (ndarray): The integrated trajectory
    """
    ic = np.array(ic)

    if np.abs(noise).sum() > 0:
        if not has_module("sdeint"):
            raise ImportError(
                "Please install the package sdeint in order to integrate with noise."
            )
        gw = lambda y, t: noise * np.diag(ic)
        fw = lambda y, t: np.array(f(t, y))
        tvals_fine = np.linspace(
            np.min(tvals), np.max(tvals), int(np.ptp(tvals) / dtval)
        )
        sol_fine = itoint(fw, gw, np.array(ic), tvals_fine).T
        sol = np.vstack([resample(item, len(tvals)) for item in sol_fine])
    else:
        sol0 = solve_ivp(
            f,
            [tvals[0], tvals[-1]],
            ic,
            t_eval=tvals,
            first_step=dtval,
            **kwargs,
        )
        sol = sol0.y

    return sol


def integrate_weiner(f, noise_amp, ic, tvals):
    """
    Given the RHS of a dynamical system, integrate the
    system assuming Brownian noise
    Requires the Python library sdeint

    Args:
        f (callable): the right hand side of a system of ODE
        noise_amp (float): the amplitude of the Langevin forcing term
    """
    sol = integrate_dyn(f, ic, tvals, noise=noise_amp)
    return sol


def generate_ic_ensemble(
    model,
    tpts0,
    n_samples,
    frac_perturb_param=0.1,
    frac_transient=0.1,
    ic_range=None,
    random_state=0,
):
    """
    Generate an ensemble of trajectories with random initial conditions, labelled by different
    initial conditions

    Args:
        model (callable_): function defining the numerical derivative
        tpts (ndarray): the timesteps over which to run the simulation
        n_samples (int): the number of different initial conditons
        frac_perturb_param (float): the amount to perturb the ic by
        frac_transient (float): the fraction of time for the time series to settle onto the attractor
        ic_range (list): a starting value for the initial conditions
        random_state (int): the seed for the random number generator

    Returns:
        all_samples (array): A set of initial conditions
    """
    np.random.seed(random_state)
    ntpts = len(tpts0)
    dt = tpts0[1] - tpts0[0]
    t_range = tpts0[-1] - tpts0[0]
    tpts = np.arange(tpts0[0], tpts0[0] + t_range * (1 + frac_transient), dt)
    ic = model.ic
    ic_perturb = 1 + frac_perturb_param * (
        2 * np.random.random((n_samples, len(ic))) - 1
    )
    ic_prime = ic * ic_perturb
    sol = np.array(
        [integrate_dyn(model, ic_prime[i], tpts)[:, -ntpts:] for i in range(n_samples)]
    )
    return sol
