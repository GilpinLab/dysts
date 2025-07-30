"""

Functions that act on DynSys or DynMap objects

"""

import warnings
from typing import Callable, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from tqdm import tqdm

from .base import DynSys
from .utils import (
    ComputationHolder,
    find_characteristic_timescale,
    find_significant_frequencies,
    has_module,
    jac_fd,
)

if has_module("sklearn"):
    from sklearn.linear_model import RidgeCV
else:
    warnings.warn(
        "Sklearn not installed. Will not be able to use ridge regression for gpdistance and corr_gpdim."
    )


def logarithmic_n(min_n: float, max_n: float, factor: float) -> np.ndarray:
    """
    Return a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached. Non-integer results are rounded
    down.
    Based on a similar function in the nolds Python library.

    Args:
        min_n (float):
            minimum value (must be < max_n)
        max_n (float):
            maximum value (must be > min_n)
        factor (float):
            factor used to increase min_n (must be > 1)

    Returns:
        list of integers:
            min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
            without duplicates
    """
    assert max_n > min_n > 0 and factor > 1
    max_i = int(np.floor(np.log(max_n / min_n) / np.log(factor)))
    ns = np.unique(np.floor(min_n * factor ** np.arange(max_i + 1)).astype(int))
    return ns[ns <= max_n]


def rowwise_euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the euclidean distance across rows"""
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def sample_initial_conditions(
    model: Callable,
    points_to_sample: int,
    traj_length: int = 1000,
    pts_per_period: int = 30,
) -> np.ndarray:
    """
    Generate a random sample of initial conditions from a dynamical system

    Args:
        model (callable): the right hand side of a differential equation, in format func(X, t)
        points_to_sample (int): the number of random initial conditions to sample
        traj_length (int): the total length of the reference trajectory from which points are drawn
        pts_per_period (int): the sampling density of the trajectory

    Returns:
        sample_points (ndarray): The points with shape (points_to_sample, d)

    """
    initial_sol = model.make_trajectory(
        traj_length, resample=True, pts_per_period=pts_per_period, postprocess=False
    )
    sample_inds = np.random.choice(
        np.arange(initial_sol.shape[0]), points_to_sample, replace=False
    )
    sample_pts = initial_sol[sample_inds]
    return sample_pts


def compute_timestep(
    system: DynSys,
    total_length: int = 40000,
    transient_fraction: float = 0.2,
    num_iters: int = 20,
    pts_per_period: int = 1000,
    timescale: str = "Fourier",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a dynamical system object, find the integration timestep based on the largest
    signficant frequency

    Args:
        model (DynSys): A dynamical systems object.
        total_length (int): The total trajectory length to use to determine timescales.
        transient_fraction (float): The fraction of a trajectory to discard as transient
        num_iters (int): The number of refinements to the timestep
        pts_per_period (int): The target integration timestep relative to the signal.
        visualize (bool): Whether to plot timestep versus time, in order to identify problems
            with the procedure

    Returns
        dt (float): The best integration timestep
        period (float, optional): The dominant timescale in the signal

    """

    print(f"Starting timestep: {system.dt}\nStarting period: {system.period}\n")
    cutoff = int(transient_fraction * total_length)

    all_dt = list()
    all_periods = list()
    for _ in tqdm(range(num_iters)):
        tpts0, sol0 = system.make_trajectory(  # type: ignore
            n=total_length,
            resample=True,
            standardize=True,
            return_times=True,
            postprocess=True,
            pts_per_period=pts_per_period,
            timescale=timescale,
        )
        if sol0 is None or tpts0 is None:
            raise Exception("Failed to generate trajectory")
        tpts0, sol0 = tpts0[cutoff:], sol0[cutoff:]
        dtval0 = np.median(np.diff(tpts0))

        all_max_freqs = list()
        for row in sol0.T:
            freqs_sig = find_significant_frequencies(row)
            if len(freqs_sig) > 0:
                all_max_freqs.append(np.max(freqs_sig))
        max_freq = max(all_max_freqs)
        dt0 = dtval0 * (1 / max_freq) / 100

        all_characteristic_timescales = list()
        for row in sol0.T:
            all_characteristic_timescales.append(find_characteristic_timescale(row))
        char_time0 = np.median(all_characteristic_timescales)

        period0 = char_time0 * dtval0  # this needs to be a scalar float

        system.dt, system.period = dt0, period0  # type: ignore
        print(system.dt, system.period)
        all_dt.append(system.dt)
        all_periods.append(system.period)
    print("all dt: ", all_dt)
    print("all periods: ", all_periods)
    return np.array(all_dt), np.array(all_periods)


def estimate_powerlaw(data0: np.ndarray) -> float:
    """
    Given a 1D array of continuous-valued data, estimate the power law exponent using the
    maximum likelihood estimator proposed by Clauset, Shalizi, Newman (2009).

    Args:
        data0 (np.ndarray): An array of continuous-valued data

    Returns:
        float: The estimated power law exponent
    """
    data = np.sort(data0, axis=0).copy()
    xmin = np.min(data, axis=0)
    n = data.shape[0]
    ahat = 1 + n / np.sum(np.log(data / xmin), axis=0)
    return ahat


def gp_dim(
    data: np.ndarray,
    y_data: np.ndarray | None = None,
    rvals: np.ndarray | None = None,
    nmax: int = 100,
) -> float:
    """
    Estimate the Grassberger-Procaccia dimension for a numpy array using the
    empirical correlation integral.

    Args:
        data (np.array): T x D, where T is the number of datapoints/timepoints, and D
            is the number of features/dimensions
        y_data (np.array, Optional): A second dataset of shape T2 x D, for
            computing cross-correlation.
        rvals (np.array): A list of radii
        nmax (int): The number of points at which to evaluate the correlation integral

    Returns:
        rvals (np.array): The discrete bins at which the correlation integral is
            estimated
        corr_sum (np.array): The estimates of the correlation integral at each bin

    """

    data = np.asarray(data)

    ## For self-correlation
    if y_data is None:
        y_data = data.copy()

    if rvals is None:
        std = np.std(data)
        rvals = np.logspace(np.log10(0.1 * std), np.log10(0.5 * std), nmax)

    dists = cdist(data, y_data)
    rvals = dists.ravel()

    ## Truncate the distance distribution to the linear scaling range
    std = np.std(data)
    rvals = rvals[rvals > 0]
    rvals = rvals[rvals > np.percentile(rvals, 5)]
    rvals = rvals[rvals < np.percentile(rvals, 50)]

    return estimate_powerlaw(rvals)


def twonn_dim(X: np.ndarray, approximate: bool = False) -> float:
    """
    Estimate the global intrinsic dimension of a point cloud using the TwoNN
    estimator [1]. If approximate is True, the calculation is performed using an
    approximate nearest neighbors algorithm.

    Args:
        X (np.array): Input data points.

    Returns:
        d (float): Estimated intrinsic dimension.

    References:
        [1] Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017).
            Estimating the intrinsic dimension of datasets by a minimal neighborhood
            information.Scientific reports, 7(1), 12140.
    """
    X = np.asarray(X)
    # Find distances to self (0), 1st NN, and 2nd NN

    if approximate:
        try:
            import hnswlib  # type: ignore
        except ImportError:
            raise ImportError(
                "hnswlib not installed. Please install hnswlib to use approximate calculation."
            )

        index = hnswlib.Index(space="l2", dim=X.shape[1])
        index.init_index(max_elements=X.shape[0], ef_construction=100, M=16)
        index.add_items(X)
        # index.set_ef(100)
        labels, distances = index.knn_query(X, k=3)
        distances = np.sqrt(distances)

    else:
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError(
                "scikit-learn not installed. Please install scikit-learn to use exact calculation."
            )
        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        distances, _ = nbrs.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Ratio of distances for each point
    mu = r2 / r1
    # Filter out any invalid ratios (e.g. division by zero)
    mu = mu[np.isfinite(mu) & (mu > 1)]

    n = mu.size
    # Maximum likelihood estimator for d under CDF F(mu) = 1 − mu^{-d}
    d = n / np.sum(np.log(mu))
    return d


def corr_gpdim(
    traj1: np.ndarray,
    traj2: np.ndarray,
    register: bool = False,
    standardize: bool = False,
    **kwargs,
) -> tuple[float, float, float]:
    """
    Given two multivariate time series, estimate their similarity using the cross
    Grassberger-Procaccia dimension

    This quantity is defined as the cross-correlation between the two time series
    normalized by the product of the Grassberger-Procaccia dimension of each time series.
    np.sqrt(<ox oy> / ox oy)

    Args:
        traj1 (np.array): T x D, where T is the number of timepoints, and D
            is the number of dimensions
        traj2 (np.array): T x D, where T is the number of timepoints, and D
            is the number of dimensions
        register (bool): Whether to register the two time series before computing the
            cross-correlation
        standardize (bool): Whether to standardize the time series before computing the
            cross-correlation

    Returns:
        float: The cross-correlation between the two time series
        float: The Grassberger-Procaccia dimension of the first time series
        float: The Grassberger-Procaccia dimension of the second time series
    """
    if register:
        if not has_module("sklearn"):
            raise ImportError("Sklearn is required for registration")
        model = RidgeCV()
        model.fit(traj1, traj2)
        traj1 = model.predict(traj1)

    if standardize:
        traj1 = (traj1 - np.mean(traj1, axis=0)) / np.std(traj1, axis=0)
        traj2 = (traj2 - np.mean(traj2, axis=0)) / np.std(traj2, axis=0)

    gpdim1 = gp_dim(traj1, **kwargs)
    gpdim2 = gp_dim(traj2, **kwargs)

    res = gp_dim(traj1, traj2, **kwargs) / np.sqrt(gpdim1 * gpdim2)
    return res, gpdim1, gpdim2


def gpdistance(
    traj1: np.ndarray,
    traj2: np.ndarray,
    standardize: bool = True,
    register: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Given two multivariate time series, estimate their similarity using the cross
    Grassberger-Procaccia distance

    Args:
        traj1 (np.array): T x D, where T is the number of timepoints, and D
            is the number of dimensions
        traj2 (np.array): T x D, where T is the number of timepoints, and D
            is the number of dimensions
        register (bool): Whether to register the two time series before computing the
            cross-correlation
        standardize (bool): Whether to standardize the time series before computing the
            cross-correlation
    """

    if register:
        if not has_module("sklearn"):
            raise ImportError("Sklearn is required for registration")
        model = RidgeCV()
        model.fit(traj1, traj2)
        traj1 = model.predict(traj1)

    if standardize:
        traj1 = (traj1 - np.mean(traj1, axis=0)) / np.std(traj1, axis=0)
        traj2 = (traj2 - np.mean(traj2, axis=0)) / np.std(traj2, axis=0)

    return np.abs(np.log(corr_gpdim(traj1, traj2, **kwargs)))


def find_lyapunov_exponents(
    model: Callable,
    traj_length: int,
    pts_per_period: int = 500,
    tol: float = 1e-8,
    min_tpts: int = 10,
    **kwargs,
) -> np.ndarray:
    """
    Given a dynamical system, compute its spectrum of Lyapunov exponents.
    Args:
        model (callable): the right hand side of a differential equation, in format
            func(X, t)
        traj_length (int): the length of each trajectory used to calulate Lyapunov
            exponents
        pts_per_period (int): the sampling density of the trajectory
        kwargs: additional keyword arguments to pass to the model's make_trajectory
            method

    Returns:
        final_lyap (ndarray): A list of computed Lyapunov exponents

    References:
        Christiansen & Rugh (1997). Computing Lyapunov spectra with continuous
            Gram-Schmidt orthonormalization

    Example:
        >>> import dysts
        >>> model = dysts.Lorenz()
        >>> lyap = dysts.find_lyapunov_exponents(model, 1000, pts_per_period=1000)
        >>> print(lyap)

    """
    d = np.asarray(model.ic).shape[-1]
    tpts, traj = model.make_trajectory(
        traj_length,
        pts_per_period=pts_per_period,
        resample=True,
        return_times=True,
        postprocessing=False,
        **kwargs,
    )
    dt = np.median(np.diff(tpts))
    # traj has shape (traj_length, d), where d is the dimension of the system
    # tpts has shape (traj_length,)
    # dt is the dimension of the system

    u = np.identity(d)
    all_lyap = list()
    for i, (t, X) in enumerate(zip(tpts, traj)):
        X = traj[i]

        if model.jac(model.ic, 0) is None:
            rhsy = lambda x: np.array(model.rhs(x, t))
            jacval = jac_fd(rhsy, X)
        else:
            jacval = np.array(model.jac(X, t))

        # If postprocessing is applied to a trajectory, transform the jacobian into the
        # new coordinates.
        if hasattr(model, "_postprocessing"):
            X0 = np.copy(X)
            y2h = lambda y: model._postprocessing(*y)
            dhdy = jac_fd(y2h, X0)
            dydh = np.linalg.inv(dhdy)  # dy/dh
            ## Alternate version if good second-order fd is ever available
            jacval = dhdy @ jacval @ dydh

        ## Backward Euler update
        if i < 1:
            continue
        u_n = np.matmul(np.linalg.inv(np.eye(d) - jacval * dt), u)

        q, r = np.linalg.qr(u_n)
        lyap_estimate = np.log(abs(r.diagonal()))
        all_lyap.append(lyap_estimate)
        u = q  # post-iteration update axes

        ## early stopping if middle exponents are close to zero, a requirement for
        ## continuous-time dynamical systems
        if (np.min(np.abs(lyap_estimate)) < tol) and (i > min_tpts):
            traj_length = i

    all_lyap = np.array(all_lyap)
    final_lyap = np.sum(all_lyap, axis=0) / (dt * traj_length)
    return np.sort(final_lyap)[::-1]


def calculate_lyapunov_exponent(
    traj1: np.ndarray, traj2: np.ndarray, dt: float = 1.0
) -> float:
    """
    Calculate the lyapunov exponent of two multidimensional trajectories using
    simple linear regression based on the log-transformed separation of the
    trajectories.

    Args:
        traj1 (np.ndarray): trajectory 1 with shape (n_timesteps, n_dimensions)
        traj2 (np.ndarray): trajectory 2 with shape (n_timesteps, n_dimensions)
        dt (float): time step between timesteps

    Returns:
        float: lyapunov exponent
    """
    separation = np.linalg.norm(traj1 - traj2, axis=1)
    log_separation = np.log(separation)
    time_vals = np.arange(log_separation.shape[0])
    result = linregress(time_vals, log_separation)
    lyap = result.slope / dt  # type: ignore
    return lyap


def max_lyapunov_exponent(
    eq: DynSys,
    max_walltime: float,
    rtol: float = 1e-3,
    atol: float = 1e-10,
    n_samples: int = 1000,
    traj_length: int = 5000,
    **kwargs,
) -> float | None:
    """
    Calculate the lyapunov spectrum of the system using a naive method based on the
    log-transformed separation of the trajectories over time.

    Args:
        eq (dysts.DynSys): equation to calculate the lyapunov spectrum of
        rtol (float): relative tolerance for the separation of the trajectories
        atol (float): absolute tolerance for the separation of the trajectories
        n_samples (int): number of initial conditions to sample
        traj_length (int): length of the trajectories to sample. This should be long
            enough to ensure that most trajectories explore the attractor.
        max_walltime (float): maximum walltime in seconds to spend on the calculation
            of a given trajectory. If the calculation takes longer than this, the
            trajectory is discarded and a new one is sampled.
        **kwargs: keyword arguments to pass to the sample / make_trajectory method of
             the dynamical equation

    Returns:
        float: largest lyapunov exponent

    Example:
        >>> import dysts
        >>> eq = dysts.Lorenz()
        >>> max_lyap = dysts.lyapunov_exponent_naive(eq)

    """
    all_ic = sample_initial_conditions(
        eq,
        n_samples,
        traj_length=max(traj_length, n_samples),
        pts_per_period=15,
    )
    pts_per_period = 100
    eps = atol
    eps_max = rtol
    all_lyap = []
    all_cutoffs = []
    for ind, ic in enumerate(all_ic):
        np.random.seed(ind)
        eq.ic = ic
        out = ComputationHolder(
            eq.make_trajectory,
            traj_length,
            timeout=max_walltime,
            resample=True,
            return_times=True,
            **kwargs,
        ).run()
        if out is None or np.sum(np.isnan(out[1])) > 0:
            continue
        else:
            # Ensure out is a tuple with at least two elements
            if isinstance(out, tuple) and len(out) >= 2:
                tvals, traj1, *_ = out  # Unpack only the first two elements
            else:
                continue  # Skip if out is not valid

        np.random.seed(ind)
        eq.ic = ic
        eq.ic *= 1 + eps * (np.random.random(eq.ic.shape) - 0.5)
        traj2 = ComputationHolder(
            eq.make_trajectory,
            traj_length,
            timeout=max_walltime,
            resample=True,
            **kwargs,
        ).run()
        if traj2 is None:
            continue
        if np.sum(np.isnan(traj2)) > 0:
            continue

        ## Truncate traj1 and traj2 to when their scaled separation is less than eps_max
        separation = np.linalg.norm(traj1 - traj2, axis=1) / np.linalg.norm(
            traj1, axis=1
        )
        cutoff_index = np.where(separation < eps_max)[0][-1]
        all_cutoffs.append(cutoff_index)
        traj1 = traj1[:cutoff_index]
        traj2 = traj2[:cutoff_index]
        lyap = calculate_lyapunov_exponent(
            traj1, traj2, dt=float(np.median(np.diff(tvals)))
        )  # Convert to float
        all_lyap.append(lyap)

    ## Return None if no trajectories were successful
    if len(all_lyap) == 0:
        return None

    if len(all_lyap) < int(0.6 * n_samples):
        warnings.warn(
            "The number of successful trajectories is less than 60% of the total number "
            + "of trajectories attempted. This may indicate that the integration "
            + "is unstable"
        )

    if np.median(all_cutoffs) < pts_per_period:
        warnings.warn(
            "The median cutoff index is less than the number of points per period. "
            + "This may indicate that the integration is not long enough to capture "
            + "the invariant properties."
        )

    return float(np.mean(all_lyap))


def kaplan_yorke_dimension(spectrum0: np.ndarray) -> float:
    """Calculate the Kaplan-Yorke dimension, given a list of
    Lyapunov exponents"""
    spectrum = np.sort(spectrum0)[::-1]
    d = len(spectrum)
    cspec = np.cumsum(spectrum)
    j = np.max(np.where(cspec >= 0))
    if j > d - 2:
        j = d - 2
        warnings.warn(
            "Cumulative sum of Lyapunov exponents never crosses zero. System may be ill-posed or undersampled."
        )
    dky = 1 + j + cspec[j] / np.abs(spectrum[j + 1])

    return dky


def max_lyapunov_exponent_rosenstein(
    data: Sequence[float] | np.ndarray,
    lag: int | None = None,
    min_tsep: int | None = None,
    tau: float = 1,
    trajectory_len: int = 20,
    fit: str = "RANSAC",
    fit_offset: int = 0,
) -> float:
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Estimates the largest Lyapunov exponent using the algorithm of Rosenstein
    et al. [lr_1]_.
    Explanation of Lyapunov exponents:
        See lyap_e.
    Explanation of the algorithm:
        The algorithm of Rosenstein et al. is only able to recover the largest
        Lyapunov exponent, but behaves rather robust to parameter choices.
        The idea for the algorithm relates closely to the definition of Lyapunov
        exponents. First, the dynamics of the data are reconstructed using a delay
        embedding method with a lag, such that each value x_i of the data is mapped
        to the vector
        X_i = [x_i, x_(i+lag), x_(i+2*lag), ..., x_(i+(emb_dim-1) * lag)]
        For each such vector X_i, we find the closest neighbor X_j using the
        euclidean distance. We know that as we follow the trajectories from X_i and
        X_j in time in a chaotic system the distances between X_(i+k) and X_(j+k)
        denoted as d_i(k) will increase according to a power law
        d_i(k) = c * e^(lambda * k) where lambda is a good approximation of the
        highest Lyapunov exponent, because the exponential expansion along the axis
        associated with this exponent will quickly dominate the expansion or
        contraction along other axes.
        To calculate lambda, we look at the logarithm of the distance trajectory,
        because log(d_i(k)) = log(c) + lambda * k. This gives a set of lines
        (one for each index i) whose slope is an approximation of lambda. We
        therefore extract the mean log trajectory d'(k) by taking the mean of
        log(d_i(k)) over all orbit vectors X_i. We then fit a straight line to
        the plot of d'(k) versus k. The slope of the line gives the desired
        parameter lambda.

    Method for choosing min_tsep:
        Usually we want to find neighbors between points that are close in phase
        space but not too close in time, because we want to avoid spurious
        correlations between the obtained trajectories that originate from temporal
        dependencies rather than the dynamic properties of the system. Therefore it
        is critical to find a good value for min_tsep. One rather plausible
        estimate for this value is to set min_tsep to the mean period of the
        signal, which can be obtained by calculating the mean frequency using the
        fast fourier transform. This procedure is used by default if the user sets
        min_tsep = None.

    Method for choosing lag:
        Another parameter that can be hard to choose by instinct alone is the lag
        between individual values in a vector of the embedded orbit. Here,
        Rosenstein et al. suggest to set the lag to the distance where the
        autocorrelation function drops below 1 - 1/e times its original (maximal)
        value. This procedure is used by default if the user sets lag = None.

    References:
        .. [lr_1] M. T. Rosenstein, J. J. Collins, and C. J. De Luca,
             "A practical method for calculating largest Lyapunov exponents from
             small data sets," Physica D: Nonlinear Phenomena, vol. 65, no. 1,
             pp. 117–134, 1993.
    Reference Code:
        .. [lr_a] mirwais, "Largest Lyapunov Exponent with Rosenstein's Algorithm",
             url: http://www.mathworks.com/matlabcentral/fileexchange/38424-largest-lyapunov-exponent-with-rosenstein-s-algorithm
        .. [lr_b] Shapour Mohammadi, "LYAPROSEN: MATLAB function to calculate
             Lyapunov exponent",
             url: https://ideas.repec.org/c/boc/bocode/t741502.html

    Args:
        data (iterable of float):
            (one-dimensional) time series
    Kwargs:
        emb_dim (int):
            embedding dimension for delay embedding
        lag (float):
            lag for delay embedding
        min_tsep (float):
            minimal temporal separation between two "neighbors" (default:
            find a suitable value by calculating the mean period of the data)
        tau (float):
            step size between data points in the time series in seconds
            (normalization scaling factor for exponents)
        min_neighbors (int):
            if lag=None, the search for a suitable lag will be stopped when the
            number of potential neighbors for a vector drops below min_neighbors
        trajectory_len (int):
            the time (in number of data points) to follow the distance
            trajectories between two neighboring points
        fit (str):
            the fitting method to use for the line fit, either 'poly' for normal
            least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
            is more robust to outliers
       fit_offset (int):
            neglect the first fit_offset steps when fitting

    Returns:
        float:
            an estimate of the largest Lyapunov exponent (a positive exponent is
            a strong indicator for chaos)
    """
    data = np.asarray(data, dtype="float32")
    n = len(data)
    max_tsep_factor = 0.25

    if lag is None or min_tsep is None:
        f = np.fft.rfft(data, n * 2 - 1)

    if min_tsep is None:
        mf = np.fft.rfftfreq(n * 2 - 1) * np.abs(f)
        mf = np.mean(mf[1:]) / np.sum(np.abs(f[1:]))
        min_tsep = int(np.ceil(1.0 / mf))
        if min_tsep > max_tsep_factor * n:
            min_tsep = int(max_tsep_factor * n)

    orbit = data
    m = len(orbit)
    dists = np.array([rowwise_euclidean(orbit, orbit[i]) for i in range(m)])  # type: ignore

    for i in range(m):
        dists[i, max(0, i - min_tsep) : i + min_tsep + 1] = float("inf")

    ntraj = m - trajectory_len + 1
    min_traj = min_tsep * 2 + 2

    if ntraj <= 0:
        raise ValueError(
            f"Not enough data points. Need {-ntraj + 1} additional data points to follow a complete trajectory."
        )
    if ntraj < min_traj:
        raise ValueError(
            f"Not enough data points. At least {min_traj} trajectories are required to find a valid neighbor for each orbit vector with min_tsep={min_tsep} but only {ntraj} could be created."
        )

    nb_idx = np.argmin(dists[:ntraj, :ntraj], axis=1)

    div_traj = np.zeros(trajectory_len, dtype=float)
    for k in range(trajectory_len):
        indices = (np.arange(ntraj) + k, nb_idx + k)
        div_traj_k = dists[indices]
        nonzero = np.where(div_traj_k != 0)
        div_traj[k] = (
            -np.inf if len(nonzero[0]) == 0 else np.mean(np.log(div_traj_k[nonzero]))
        )

    ks = np.arange(trajectory_len)
    finite = np.where(np.isfinite(div_traj))
    ks = ks[finite]
    div_traj = div_traj[finite]

    if len(ks) < 1:
        return -np.inf

    poly = np.polyfit(ks[fit_offset:], div_traj[fit_offset:], 1)

    le = poly[0] / tau
    return le


def dfa(
    data: np.ndarray,
    nvals: Sequence[int] | np.ndarray | None = None,
    overlap: bool = True,
    order: int = 1,
) -> float:
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Performs a detrended fluctuation analysis (DFA) on the given data
    Recommendations for parameter settings by Hardstone et al.:
        * nvals should be equally spaced on a logarithmic scale so that each window
            scale hase the same weight
        * min(nvals) < 4 does not make much sense as fitting a polynomial (even if
            it is only of order 1) to 3 or less data points is very prone.
        * max(nvals) > len(data) / 10 does not make much sense as we will then have
            less than 10 windows to calculate the average fluctuation
        * use overlap=True to obtain more windows and therefore better statistics
            (at an increased computational cost)

    Explanation of DFA:
        Detrended fluctuation analysis, much like the Hurst exponent, is used to
        find long-term statistical dependencies in time series.
        The idea behind DFA originates from the definition of self-affine
        processes. A process X is said to be self-affine if the standard deviation
        of the values within a window of length n changes with the window length
        factor L in a power law:
        std(X,L * n) = L^H * std(X, n)
        where std(X, k) is the standard deviation of the process X calculated over
        windows of size k. In this equation, H is called the Hurst parameter, which
        behaves indeed very similar to the Hurst exponent.
        Like the Hurst exponent, H can be obtained from a time series by
        calculating std(X,n) for different n and fitting a straight line to the
        plot of log(std(X,n)) versus log(n).
        To calculate a single std(X,n), the time series is split into windows of
        equal length n, so that the ith window of this size has the form
        W_(n,i) = [x_i, x_(i+1), x_(i+2), ... x_(i+n-1)]
        The value std(X,n) is then obtained by calculating std(W_(n,i)) for each i
        and averaging the obtained values over i.
        The aforementioned definition of self-affinity, however, assumes that the
        process is    non-stationary (i.e. that the standard deviation changes over
        time) and it is highly influenced by local and global trends of the time
        series.
        To overcome these problems, an estimate alpha of H is calculated by using a
        "walk" or "signal profile" instead of the raw time series. This walk is
        obtained by substracting the mean and then taking the cumulative sum of the
        original time series. The local trends are removed for each window
        separately by fitting a polynomial p_(n,i) to the window W_(n,i) and then
        calculating W'_(n,i) = W_(n,i) - p_(n,i) (element-wise substraction).
        We then calculate std(X,n) as before only using the "detrended" window
        W'_(n,i) instead of W_(n,i). Instead of H we obtain the parameter alpha
        from the line fitting.
        For alpha < 1 the underlying process is stationary and can be modelled as
        fractional Gaussian noise with H = alpha. This means for alpha = 0.5 we
        have no correlation or "memory", for 0.5 < alpha < 1 we have a memory with
        positive correlation and for alpha < 0.5 the correlation is negative.
        For alpha > 1 the underlying process is non-stationary and can be modeled
        as fractional Brownian motion with H = alpha - 1.

    References:
        .. [dfa_1] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
                             H. E. Stanley, and A. L. Goldberger, "Mosaic organization of
                             DNA nucleotides," Physical Review E, vol. 49, no. 2, 1994.
        .. [dfa_2] R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
                             V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
                             "Detrended fluctuation analysis: A scale-free view on neuronal
                             oscillations," Frontiers in Physiology, vol. 30, 2012.

    Reference code:
        .. [dfa_a] Peter Jurica, "Introduction to MDFA in Python",
             url: http://bsp.brain.riken.jp/~juricap/mdfa/mdfaintro.html
        .. [dfa_b] JE Mietus, "dfa",
             url: https://www.physionet.org/physiotools/dfa/dfa-1.htm
        .. [dfa_c] "DFA" function in R package "fractal"

    Args:
        data (array-like of float):
            time series
    Kwargs:
        nvals (iterable of int):
            subseries sizes at which to calculate fluctuation
            (default: logarithmic_n(4, 0.1*len(data), 1.2))
        overlap (boolean):
            if True, the windows W_(n,i) will have a 50% overlap,
            otherwise non-overlapping windows will be used
        order (int):
            (polynomial) order of trend to remove

    Returns:
        float:
            the estimate alpha for the Hurst parameter (alpha < 1: stationary
            process similar to fractional Gaussian noise with H = alpha,
            alpha > 1: non-stationary process similar to fractional Brownian
            motion with H = alpha - 1)
    """
    data = np.asarray(data)
    total_N = len(data)
    if nvals is None:
        nvals = logarithmic_n(4, 0.1 * total_N, 1.2)
    if len(nvals) < 2:
        raise ValueError("at least two nvals are needed")
    if np.min(nvals) < 2:
        raise ValueError("nvals must be at least two")
    if np.max(nvals) >= total_N:
        raise ValueError("nvals cannot be larger than the input size")

    walk = np.cumsum(data - np.mean(data))
    fluctuations = []
    for n in nvals:
        if overlap:
            d = np.array([walk[i : i + n] for i in range(0, len(walk) - n, n // 2)])
        else:
            d = walk[: total_N - (total_N % n)]
            d = d.reshape((total_N // n, n))
        x = np.arange(n)
        tpoly = [np.polyfit(x, d[i], order) for i in range(len(d))]
        tpoly = np.array(tpoly)
        trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
        flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)
        f_n = np.sum(flucs) / len(flucs)
        fluctuations.append(f_n)
    fluctuations = np.array(fluctuations)
    nonzero = np.where(fluctuations != 0)
    nvals = np.array(nvals)[nonzero]
    fluctuations = fluctuations[nonzero]
    if len(fluctuations) == 0:
        poly = [np.nan, np.nan]
    else:
        poly = np.polyfit(np.log(nvals), np.log(fluctuations), 1)
    return poly[0]


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    """
    Compute the mutual information between two 1D arrays x and y
        - Uses a 2D histogram with the given number of bins.
        - sum_{i,j} p_xy[i,j] * log(p_xy[i,j] / (p_x[i]*p_y[j]))
    """
    pxy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * (np.log(pxy[i, j]) - np.log(px[i]) - np.log(py[j]))
    return mi


def optimal_delay(
    x: np.ndarray,
    max_delay: int = 50,
    bins: int = 64,
    conv_window_size: int = 3,
    first_k_minima_to_consider: int = 1,
) -> int:
    """
    Computes the mutual information I(tau) = I( x(t), x(t+tau) ) for tau in {1, 2, ..., max_delay}
    Returns the time lag tau corresponding to the first prominent local minimum.

    Parameters:
        x: 1D array of shape (T,)
        max_delay: maximum time lag to consider
        bins: number of bins for the histogram
        conv_window_size: size of the convolution window for smoothing the MI curve
        first_k_minima_to_consider: number of minima to consider for determining first prominent minimum
    """
    mi_values = []
    assert x.ndim == 1, "x must be a 1D array"
    T = len(x)
    for tau in range(1, max_delay + 1):
        # Use only overlapping segments
        mi_tau = mutual_information(x[: T - tau], x[tau:], bins=bins)
        mi_values.append(mi_tau)
    mi_values = np.array(mi_values)

    # Find a prominent local minimum
    # 1. smooth the MI curve to reduce noise
    smoothed_mi = np.convolve(
        mi_values, np.ones(conv_window_size) / conv_window_size, mode="valid"
    )

    # 2. Calculate the prominence of each minimum
    minima_indices = []
    prominences = []

    # 3. Find all local minima in the smoothed curve
    for i in range(1, len(smoothed_mi) - 1):
        if smoothed_mi[i] < smoothed_mi[i - 1] and smoothed_mi[i] < smoothed_mi[i + 1]:
            minima_indices.append(i)

            # Calculate prominence (height difference to nearby values on smoothed MI curve)
            left_max = float(np.max(smoothed_mi[: i + 1]))
            right_max = float(np.max(smoothed_mi[i:]))
            lower_max = min(left_max, right_max)
            prominence = lower_max - smoothed_mi[i]
            prominences.append(prominence)
            if len(prominences) >= first_k_minima_to_consider:
                break

    # If no minima found, return the global minimum
    if len(minima_indices) == 0:
        first_min = np.argmin(mi_values) + 1
    else:
        # Find the most prominent minimum among the first first_k_minima_to_consider
        num_to_consider = min(first_k_minima_to_consider, len(minima_indices))
        best_idx = np.argmax(prominences[:num_to_consider])
        # Adjust index to account for smoothing window and 1-based tau
        first_min = minima_indices[best_idx] + (conv_window_size // 2) + 1

    return int(first_min)


### Utils for Zero-One Test for Chaos ###


def compute_translation_variables(
    phi: np.ndarray, c: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        phi: 1D array of shape (T,)
        c: constant in (0,pi); if None, a random value in (pi/5, 4pi/5) is chosen to avoid resonances.
    Returns:
        p: 1D array of shape (T,), translation variable p(n) computed as cumulative sum of phi(n) * cos(c * n)
        q: 1D array of shape (T,), translation variable q(n) computed as cumulative sum of phi(n) * sin(c * n)
    """
    T = len(phi)
    n = np.arange(1, T + 1)
    # Compute translation variables
    p = np.cumsum(phi * np.cos(c * n))
    q = np.cumsum(phi * np.sin(c * n))
    return p, q


def compute_mean_square_displacement(
    p: np.ndarray, q: np.ndarray, max_shift_ratio: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the mean square displacement (msd) over a range of shifts.

    Parameters:
      p, q: translation variables (1D arrays)
      max_shift_ratio: maximum fraction of the length to use for shifts.

    Returns:
      shift_indices: array of shift indices
      msd: array of mean square displacements corresponding to shift_indices.
    """
    T = len(p)
    max_shift = int(max_shift_ratio * T)
    shift_indices = np.arange(1, max_shift + 1)
    msd = np.empty_like(shift_indices, dtype=float)

    # For each time shift n, compute the mean squared difference
    for idx, shift_idx in enumerate(shift_indices):
        diff_p = p[shift_idx:] - p[:-shift_idx]
        diff_q = q[shift_idx:] - q[:-shift_idx]
        msd[idx] = np.mean(diff_p**2 + diff_q**2)
    return shift_indices, msd


def compute_K_statistic(shift_indices: np.ndarray, msd: np.ndarray) -> float:
    """
    Computes the correlation coefficient between the shift indices and the msd.
    A value near 1 indicates linear growth (chaos), while near 0 indicates bounded behavior.

    Parameters:
        shift_indices: 1D array of shape (T,), the shift indices
        msd: 1D array of shape (T,), the mean square displacement computed as the mean of the squared differences between the translation variables

    Returns:
        K: correlation coefficient
    """
    corr_matrix = np.corrcoef(shift_indices, msd)  # 2x2 correlation matrix
    K = corr_matrix[0, 1]
    return K


def zero_one_test(phi: np.ndarray, c: float | None = None) -> float:
    """
    Performs the 0–1 test for chaos on a scalar observable.

    Parameters:
      phi: univariate time series from the trajectory (e.g. x, y, or z coordinate, or norm), of length T
      c: constant in (0,pi); if None, a random value in (pi/5, 4pi/5) is chosen to avoid resonances.
            NOTE: this range seems to capture intrinsic diffusive behavior without interference from unwanted resonant effects.

    Returns:
      K: the computed correlation coefficient.
    """
    if c is None:
        # Choosing c in (pi/5, 4*pi/5) can help avoid resonances.
        c = np.random.uniform(np.pi / 10, 1 * np.pi / 5)

    p, q = compute_translation_variables(phi, c)
    shift_indices, msd = compute_mean_square_displacement(p, q)
    K = compute_K_statistic(shift_indices, msd)  # correlation coefficient

    return K


def run_zero_one_sweep(
    timeseries: np.ndarray,
    c_min: float = np.pi / 5,
    c_max: float = 4 * np.pi / 5,
    n_runs: int = 100,
    k: int = 1,
) -> np.ndarray:
    """
    Runs a sweep of zero_one_test for the given univariate timeseries and c_vals

    Parameters:
        timeseries: univariate observed timeseries (e.g. x, y, z or norm of traj) of length T
        c_min: min value for c parameter, defaults to pi/5
        c_max: max value for c parameter, defaults to 4pi/5
        k: number of minima to consider for determining first prominent minimum
            the subsampling interval is the optimal delay computed using the mutual information
                NOTE: the performance of the 0-1 test for chaos is sensitive to this choice
                Subsampling helps to de-correlate timeseries that are oversampled (very similar consecutive points, excessive correlation),
                    ensuring that the time series better reflects the intrinsic dynamics of the system rather than oversampling artifacts
        n_runs: number of random c values to try
        threshold: threshold on |K| to decide if the system is chaotic

    Returns:
        K_vals: array of |K| values from the runs
    """
    assert timeseries.ndim == 1, "timeseries must be 1D"
    c_vals = np.random.uniform(c_min, c_max, n_runs)
    K_vals = []
    tau_opt = optimal_delay(
        timeseries, max_delay=50, bins=64, first_k_minima_to_consider=k
    )

    timeseries = timeseries[::tau_opt]
    for c_val in c_vals:
        K = zero_one_test(
            timeseries,
            c=c_val,
        )
        K_vals.append(K)

    K_vals = np.array(K_vals)
    return K_vals
