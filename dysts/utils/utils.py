"""
Helper utilities for working with time series arrays. This module is intended to have no
dependencies on the rest of the package.
"""

import warnings

import numpy as np
from numpy.fft import rfft
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import approx_fprime
from scipy.signal import periodogram
from scipy.signal.windows import blackmanharris

from .native_utils import group_consecutives


def polar_to_cartesian(r, th):
    """Convert polar coordinates to 2D Cartesian coordinates"""
    x, y = r * np.cos(th), r * np.sin(th)
    return x, y


def cartesian_to_polar(x, y):
    """Convert 2D cartesian coordinates to polar coordinates"""
    th = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return r, th


def signif(x, figs=6):
    """Round a float to a fixed number of significant digits

    Args:
        x (float): The number to round
        figs (int): the desired number of significant figures
    """
    if x == 0 or not np.isfinite(x):
        return x
    figs = int(figs - np.ceil(np.log10(abs(x))))
    return round(x, figs)


def nanmean_trimmed(arr, percentile=0.5, axis=None):
    """
    Compute the trimmed mean of an array along a specified axis, ignoring NaNs.

    Args:
        arr (np.array): Input array
        percentile (float): The fraction of data to be trimmed. Should be between 0 and
            1. Default is 0.5.
        axis (int): Axis along which to compute the trimmed mean. If None (the default),
            compute over the whole array.
    Returns:
        Trimmed mean of the input array along the specified axis.
    """
    if axis is None:
        arr = arr.ravel()
        axis = 0

    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")

    # Calculate the indices corresponding to the trimming percentage
    n = arr.shape[axis]
    lower = int(n * percentile / 2)
    upper = n - lower

    # Sort the array along the specified axis, ignoring NaNs
    sorted_arr = np.partition(arr, (lower, upper), axis=axis)
    sorted_slice = [slice(None)] * arr.ndim
    sorted_slice[axis] = slice(lower, upper)
    sorted_arr = sorted_arr[sorted_slice]

    # Calculate the mean along the specified axis, ignoring NaNs
    return np.nanmean(sorted_arr, axis=axis)


def nan_fill(a, axis=0, mode="forward"):
    """
    Fill NaN values in an array using a specified method

    Args:
        a (np.ndarray): the array to fill
        axis (int): the axis along which to fill NaN values
        mode (str): the method to use for filling NaN values

    Returns:
        a_filled (np.ndarray): the array with NaN values filled
    """
    a = a.copy()
    if mode == "forward":
        for i in range(a.shape[axis]):
            if np.isnan(a[i]):
                a[i] = a[i - 1]
    if mode == "backward":
        for i in range(a.shape[axis] - 1, -1, -1):
            if np.isnan(a[i]):
                a[i] = a[i + 1]
    else:
        warnings.warn("Invalid mode. Returning original array.")
    return a


def standardize_ts(a, scale=1.0):
    """Standardize an array along dimension -2
    For dimensions with zero variance, divide by one instead of zero

    Args:
        a (ndarray): a matrix containing a time series or batch of time series
            with shape (T, D) or (B, T, D)
        scale (float): the number of standard deviations by which to scale

    Returns:
        ts_scaled (ndarray): A standardized time series with the same shape as
            the input
    """
    #     if len(a.shape) == 1: a = a[:, None]
    assert a.ndim in [
        2,
        3,
    ], "Input must be a time series (T, D) or batch of time series (B, T, D)"
    stds = np.std(a, axis=-2, keepdims=True)
    stds[stds == 0] = 1
    ts_scaled = (a - np.mean(a, axis=-2, keepdims=True)) / (scale * stds)
    return ts_scaled


def pad_axis(arr, d, axis=-1, padding=0):
    """
    Pad `axis` of `arr` with a constant `padding` to a desired shape
    """
    padding_length = d - arr.shape[axis]

    if padding_length <= 0:
        return arr
    if padding_length > 0:
        slice_val = padding + np.zeros_like(np.take(arr, 0, axis=axis))
        padding_chunk = np.stack([slice_val for i in range(padding_length)], axis=axis)
        return np.concatenate([arr, padding_chunk], axis=axis)


def pad_to_shape(arr, target_shape):
    """
    Given an array, and a target shape, pad the dimensions in order to reach the desired shape
    Currently, if the rank of the array is lower than the target shape, singleton
    dimensions are appended to the rank

    Args:
        arr (ndarray): The array to pad.
        target_shape (iterable): The desired shape.

    Returns:
        arr (ndarray): The padded array,
    """
    rank_difference = len(target_shape) - len(arr.shape)
    if rank_difference > 0:
        for i in range(rank_difference):
            arr = arr[..., None]

    for axis, target in enumerate(target_shape):
        arr = pad_axis(arr, target, axis=axis)

    return arr


def find_psd(y, window=True):
    """
    Find the power spectrum of a signal
    """
    if window:
        y = y * blackmanharris(len(y))
    halflen = int(len(y) / 2)
    fvals, psd = periodogram(y, fs=1)
    return fvals[:halflen], psd[:halflen]


def find_characteristic_timescale(y, k=1, window=True):
    """
    Find the k leading characteristic timescales in a time series
    using the power spectrum..
    """
    y = gaussian_filter1d(y, 3)

    fvals, psd = find_psd(y, window=window)
    max_indices = np.argsort(psd)[::-1]

    # Merge adjacent peaks
    grouped_maxima = group_consecutives(max_indices)
    max_indices_grouped = np.array([np.mean(item) for item in grouped_maxima])
    max_indices_grouped = max_indices_grouped[max_indices_grouped != 1]

    return np.squeeze(1 / (np.median(np.diff(fvals)) * max_indices_grouped[:k]))


def make_surrogate(data, method="rp"):
    """

    Args:
        data (ndarray): A one-dimensional time series
        method (str): "rs" or rp"

    Returns:
        surr_data (ndarray): A single random surrogate time series

    Todo:
        Add ensemble function

    """
    if method == "rp":
        radii = np.abs(np.fft.fft(data))
        random_phases = 2 * np.pi * np.random.random(radii.shape)
        surr_data = np.real(
            np.fft.ifft(
                radii * np.cos(random_phases) + 1j * radii * np.sin(random_phases)
            )
        )
    else:
        surr_data = np.copy(data)
        np.random.shuffle(surr_data)
    return surr_data


def find_significant_frequencies(
    sig,
    window=True,
    fs=1,
    n_samples=100,
    significance_threshold=0.95,
    surrogate_method="rp",
    show=False,
    return_amplitudes=False,
):
    """
    Find power spectral frequencies that are significant in a signal, by comparing
    the appearance of a peak with its appearance in randomly-shuffled surrogates.

    If no significant freqencies are detected, the significance floor is lowered

    Args:
        window (bool): Whether to window the signal before taking the FFT
        thresh (float): The number of standard deviations above mean to be significant
        fs (int): the sampling frequency
        n_samples (int): the number of surrogates to create
        show (bool): whether to show the psd of the signal and the surrogate

    Returns:
        freqs (ndarray): The frequencies overrated in the dataset
        amps (ndarray): the amplitudes of the PSD at the identified frequencies

    """
    n = len(sig)
    halflen = n // 2

    if window:
        sig = sig * blackmanharris(n)

    psd_sig = np.abs(rfft(sig)) ** 2

    all_surr_psd = list()
    for i in range(n_samples):
        # surr = np.copy(sig)
        surr = make_surrogate(sig, method=surrogate_method)
        # np.random.shuffle(surr)
        if window:
            surr = surr * blackmanharris(len(surr))
            psd_surr = np.abs(rfft(surr)) ** 2
        all_surr_psd.append(psd_surr)
    all_surr_psd = np.array(all_surr_psd)

    frac_exceed = np.sum((psd_sig > all_surr_psd), axis=0) / all_surr_psd.shape[0]

    sel_inds = frac_exceed >= significance_threshold
    while len(sel_inds) == 0 and significance_threshold > 0:
        significance_threshold -= 0.01
        sel_inds = frac_exceed >= significance_threshold

    freq_inds = np.arange(len(psd_sig))[sel_inds]
    amps = psd_sig[sel_inds]
    freqs = fs * freq_inds / len(sig)

    ## cutoff low frequency components: half signal length times safety factor
    freq_floor = (1 / halflen) * 10  # safety factor
    amps = amps[freqs > freq_floor]
    freqs = freqs[freqs > freq_floor]

    if return_amplitudes:
        return freqs, amps
    else:
        return freqs


def jac_fd(func0, y0, eps=1e-3, m=1, method="central", verbose=False):
    """
    Calculate numerical jacobian of a function with respect to a reference value

    Args:
        func (callable): a vector-valued function
        y0 (ndarray): a point around which to take the gradient
        eps (float): the step size for the finite difference calculation

    Returns:
        jac (ndarray): a numerical estimate of the Jacobian about that point

    """
    func = lambda x: np.array(func0(x))  # ensure an ndarray returned
    y0 = np.array(y0)  # ensure an ndarray input

    d = len(y0)
    all_rows = list()
    for i in range(d):
        row_func = lambda yy: func(yy)[i]
        row = approx_fprime(y0, row_func, epsilon=eps)
        all_rows.append(row)
    jac = np.array(all_rows)

    return jac


def find_slope(x, y):
    """
    Given two vectors or arrays, compute the best fit slope using an analytic
    formula. For arrays is computed along the last axis.

    Args:
        x, y (ndarray): (N,) or (M, N)

    Returns:
        b (ndarray): the values of the slope for each of the last dimensions
    """
    n = x.shape[-1]
    b = n * (x * y).sum(axis=-1) - x.sum(axis=-1) * y.sum(axis=-1)
    b /= n * (x * x).sum(axis=-1) - x.sum(axis=-1) * x.sum(axis=-1)
    return b


def make_epsilon_ball(pt, n, eps=1e-5, random_state=None):
    """
    Uniformly sample a fixed-radius ball of points around a given point via
    using Muller's method

    Args:
        pt (ndarray): The center of the sampling
        n (int): The number of points to sample
        eps (float): The radius of the ball
        random_state (int): Initialize the random number generator

    Returns:
        out (ndarray): The set of randomly-sampled points
    """
    np.random.seed(None)
    pt = np.squeeze(np.array(pt))
    d = len(pt)
    vecs = np.random.normal(0, 1, size=(d, n))
    r = np.random.random(n) ** (1.0 / d)
    norm = np.linalg.norm(vecs, axis=0)
    coords = r * vecs / norm
    out = pt[:, None] + eps * coords
    return out


def rowwise_euclidean(x, y):
    """Computes the euclidean distance across rows"""
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def min_data_points_rosenstein(emb_dim, lag, trajectory_len, min_tsep):
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Helper function that calculates the minimum number of data points required
    to use lyap_r.
    Note that none of the required parameters may be set to None.
    Args:
        emb_dim (int): Embedding dimension
        lag (int): Lag between time series values
        trajectory_len (int): Length of trajectory to follow
        min_tsep (int): Minimum temporal separation
    Returns:
        int: minimum number of data points required to call lyap_r with the given
        parameters
    """
    # minimum length required to find single orbit vector
    min_len = (emb_dim - 1) * lag + 1
    # we need trajectory_len orbit vectors to follow a complete trajectory
    min_len += trajectory_len - 1
    # we need min_tsep * 2 + 1 orbit vectors to find neighbors for each
    min_len += min_tsep * 2 + 1
    return min_len


def logarithmic_n(min_n, max_n, factor):
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
