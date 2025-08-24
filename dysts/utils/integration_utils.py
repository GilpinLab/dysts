"""Utilities for integration"""

from typing import Callable

import numpy as np
import scipy.integrate
import scipy.interpolate
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


# ----------------------- START OF ddeint IMPLEMENTATION -----------------------
# from https://github.com/Zulko/ddeint, we expose the code for flexibility


class ddeVar:
    """
    The instances of this class are special function-like
    variables which store their past values in an interpolator and
    can be called for any past time: Y(t), Y(t-d).
    Very convenient for the integration of DDEs.
    """

    def __init__(self, g, tc=0):
        """g(t) = expression of Y(t) for t<tc"""

        self.g = g
        self.tc = tc
        # We must fill the interpolator with 2 points minimum

        self.interpolator = scipy.interpolate.interp1d(
            np.array([tc - 1, tc]),  # X
            np.array([self.g(tc), self.g(tc)]).T,  # Y
            kind="linear",
            bounds_error=False,
            fill_value=self.g(tc),
        )

    def update(self, t, Y):
        """Add one new (ti,yi) to the interpolator"""
        Y2 = Y if (Y.size == 1) else np.array([Y]).T
        self.interpolator = scipy.interpolate.interp1d(
            np.hstack([self.interpolator.x, [t]]),  # X
            np.hstack([self.interpolator.y, Y2]),  # Y
            kind="linear",
            bounds_error=False,
            fill_value=Y,
        )

    def __call__(self, t=0):
        """Y(t) will return the instance's value at time t"""

        return self.g(t) if (t <= self.tc) else self.interpolator(t)


class dde(scipy.integrate.ode):
    """
    This class overwrites a few functions of ``scipy.integrate.ode``
    to allow for updates of the pseudo-variable Y between each
    integration step.
    """

    def __init__(self, f, jac=None):
        def f2(t, y, args):
            return f(self.Y, t, *args)

        scipy.integrate.ode.__init__(self, f2, jac)
        self.set_f_params(None)

    def integrate(self, t, step=False, relax=False):
        scipy.integrate.ode.integrate(self, t, step, relax)
        self.Y.update(self.t, self.y)
        return self.y

    def set_initial_value(self, Y):
        self.Y = Y  #!!! Y will be modified during integration
        scipy.integrate.ode.set_initial_value(self, Y(Y.tc), Y.tc)


def ddeint(func, g, tt, fargs=None):
    """Solves Delay Differential Equations

    Similar to scipy.integrate.odeint. Solves a Delay differential
    Equation system (DDE) defined by

        Y(t) = g(t) for t<0
        Y'(t) = func(Y,t) for t>= 0

    Where func can involve past values of Y, like Y(t-d).


    Parameters
    -----------

    func
      a function Y,t,args -> Y'(t), where args is optional.
      The variable Y is an instance of class ddeVar, which means that
      it is called like a function: Y(t), Y(t-d), etc. Y(t) returns
      either a number or a numpy array (for multivariate systems).

    g
      The 'history function'. A function g(t)=Y(t) for t<0, g(t)
      returns either a number or a numpy array (for multivariate
      systems).

    tt
      The vector of times [t0, t1, ...] at which the system must
      be solved.

    fargs
      Additional arguments to be passed to parameter ``func``, if any.


    Examples
    ---------

    We will solve the delayed Lotka-Volterra system defined as

        For t < 0:
        x(t) = 1+t
        y(t) = 2-t

        For t >= 0:
        dx/dt =  0.5* ( 1- y(t-d) )
        dy/dt = -0.5* ( 1- x(t-d) )

    The delay ``d`` is a tunable parameter of the model.

    .. code-block:: python

        import numpy as np
        from ddeint import ddeint

        def model(XY,t,d):
            x, y = XY(t)
            xd, yd = XY(t-d)
            return np.array([0.5*x*(1-yd), -0.5*y*(1-xd)])

        g = lambda t : np.array([1+t,2-t]) # 'history' at t<0
        tt = np.linspace(0,30,20000) # times for integration
        d = 0.5 # set parameter d
        yy = ddeint(model,g,tt,fargs=(d,)) # solve the DDE !
    """

    dde_ = dde(func)
    dde_.set_initial_value(ddeVar(g, tt[0]))
    dde_.set_f_params(fargs if fargs else [])
    results = [dde_.integrate(dde_.t + dt) for dt in np.diff(tt)]
    if isinstance(g(tt[0]), (list, tuple, np.ndarray)):
        initial_value = g(tt[0])
    else:
        initial_value = np.array([g(tt[0])])
    results.insert(0, initial_value)
    return np.stack(results)


# ---------------------------------------------------------------------
