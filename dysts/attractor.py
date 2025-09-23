"""
Suite of tests to determine if generated trajectories are valid attractors
"""

import functools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.fft import rfft
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

from dysts.utils.native_utils import has_module

from .analysis import max_lyapunov_exponent_rosenstein, run_zero_one_sweep
from .utils import safe_standardize

if has_module("statsmodels"):
    from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class AttractorValidator:
    """
    Framework to add tests, which are executed sequentially to determine if generated trajectories are valid attractors.
    Upon first failure, the trajectory sample is added to the failed ensemble.
    To add custom tests, define functions that take a trajectory and return a boolean (True if the trajectory passes the test, False otherwise).
    """

    transient_time_frac: float = 0.05  # should be low, should be on attractor
    tests: Optional[List[Callable]] = None

    multiprocess_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.failed_checks = defaultdict(list)  # Dict[str, List[Tuple[int, str]]]
        self.valid_dyst_counts = defaultdict(int)  # Dict[str, int]
        self.failed_samples = defaultdict(list)  # Dict[str, List[int]]
        self.valid_samples = defaultdict(list)  # Dict[str, List[int]]

    def reset(self):
        """
        Reset all defaultdict attributes to their initial state.
        """
        self.failed_checks.clear()
        self.valid_dyst_counts.clear()
        self.failed_samples.clear()
        self.valid_samples.clear()

    def _execute_test_fn(
        self,
        test_fn: Callable,
        traj_sample: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Execute a single test for a given trajectory sample of a system.
        Args:
            test_fn: the attractor test function to execute
            dyst_name: name of the dyst
            traj_sample: the trajectory sample to test
            sample_idx: index of the sample

        Returns:
            bool: True if the test passed, False otherwise
        """
        original_func = (
            test_fn.func if isinstance(test_fn, functools.partial) else test_fn
        )
        func_name = original_func.__name__
        status = test_fn(traj_sample)
        return status, func_name

    def _filter_system_worker_fn(
        self,
        dyst_name: str,
        all_traj: np.ndarray,
        first_sample_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], List[int]]:
        """
        Multiprocessed version of self._filter_dyst without any verbose output

        TODO: figure out how to log safely during multiprocessing
        """
        failed_checks_samples = []
        valid_samples = []
        valid_attractor_trajs = []
        failed_attractor_trajs = []
        for i, traj_sample in enumerate(all_traj):
            sample_idx = first_sample_idx + i
            # cut off transient time
            transient_time = int(traj_sample.shape[1] * self.transient_time_frac)
            traj_sample = traj_sample[:, transient_time:]
            # execute all tests in sequence
            status = True
            for test_fn in self.tests or []:
                status, test_name = self._execute_test_fn(test_fn, traj_sample)
                if not status:
                    failed_check = (sample_idx, test_name)
                    failed_checks_samples.append(failed_check)
                    break
            # if traj sample failed a test, move on to next trajectory sample for this dyst
            if not status:
                failed_attractor_trajs.append(traj_sample)
                continue
            valid_attractor_trajs.append(traj_sample)
            valid_samples.append(sample_idx)
        return (
            np.array(valid_attractor_trajs),
            np.array(failed_attractor_trajs),
            failed_checks_samples,
            valid_samples,
        )

    def filter_ensemble(
        self, ensemble: Dict[str, np.ndarray], first_sample_idx: int = 0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Execute all tests for all trajectory samples in the ensemble, and split the ensemble into valid and failed ensembles.
        Args:
            ensemble: The trajectory ensemble to filter
            first_sample_idx: The index of the first sample for the generated trajectories of the ensemble

        Returns:
            valid_attractor_ensemble: A new ensemble with only the valid trajectories
            failed_attractor_ensemble: A new ensemble with only the failed trajectories
        """
        valid_attractor_ensemble: Dict[str, np.ndarray] = {}
        failed_attractor_ensemble: Dict[str, np.ndarray] = {}
        for dyst_name, all_traj in ensemble.items():
            (
                valid_attractor_trajs,
                failed_attractor_trajs,
                failed_checks,
                valid_samples,
            ) = self._filter_system_worker_fn(dyst_name, all_traj, first_sample_idx)

            self.failed_checks[dyst_name].extend(failed_checks)
            self.failed_samples[dyst_name].extend([ind for ind, _ in failed_checks])
            self.valid_samples[dyst_name].extend(valid_samples)
            self.valid_dyst_counts[dyst_name] += len(valid_samples)

            if len(failed_attractor_trajs) > 0:
                failed_attractor_ensemble[dyst_name] = failed_attractor_trajs

            if len(valid_attractor_trajs) == 0:
                continue

            valid_attractor_ensemble[dyst_name] = valid_attractor_trajs

        return valid_attractor_ensemble, failed_attractor_ensemble

    def multiprocessed_filter_ensemble(
        self, ensemble: Dict[str, np.ndarray], first_sample_idx: int = 0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, list[int]]]:
        """
        Multiprocessed version of self.filter_ensemble.
        Also returns a dict mapping dyst_name to list of successful sample indices.
        """
        with Pool(**self.multiprocess_kwargs) as pool:
            results = pool.starmap(
                self._filter_system_worker_fn,
                [
                    (dyst_name, all_traj, first_sample_idx)
                    for dyst_name, all_traj in ensemble.items()
                ],
            )
        valid_trajs, failed_trajs, failed_checks, valid_samples = zip(*results)
        for dyst_name, failed_check_lst in zip(list(ensemble.keys()), failed_checks):
            if len(failed_check_lst) > 0:
                self.failed_checks[dyst_name].append(failed_check_lst)
                self.failed_samples[dyst_name].extend(
                    [index for index, _ in failed_check_lst]
                )

        successful_samples_dict = dict(zip(list(ensemble.keys()), valid_samples))
        for dyst_name, valid_samples_lst in successful_samples_dict.items():
            if len(valid_samples_lst) > 0:
                self.valid_samples[dyst_name].extend(valid_samples_lst)
                self.valid_dyst_counts[dyst_name] += len(valid_samples_lst)
        # form the valid and failed ensembles
        # note: relies on python 3.7+ order preservation in dictionaries
        valid_ensemble = {
            k: v for k, v in zip(list(ensemble.keys()), valid_trajs) if v.shape[0] > 0
        }
        failed_ensemble = {
            k: v for k, v in zip(list(ensemble.keys()), failed_trajs) if v.shape[0] > 0
        }

        # shift the successful samples by the first sample index, for the current call of the validator
        successful_samples_relative = {
            k: [i - first_sample_idx for i in v]
            for k, v in zip(ensemble.keys(), valid_samples)
        }
        return (
            valid_ensemble,
            failed_ensemble,
            successful_samples_relative,
        )


def check_boundedness(
    traj: np.ndarray, threshold: float = 1e4, max_zscore: float = 10, eps: float = 1e-10
) -> bool:
    """
    Check if a multi-dimensional trajectory is bounded (not diverging).

    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        threshold: Maximum absolute value of the trajectory to consider as diverging.
        max_zscore: Maximum z-score of the trajectory to consider as diverging.
    Returns:
        bool: False if the system is diverging, True otherwise.
    """
    if np.any(np.abs(traj) > threshold):
        return False

    traj = safe_standardize(traj)

    # If the coordinates of any dimension exceeds the max_zscore, mark the system as diverging
    if np.max(np.abs(traj)) > max_zscore:
        return False

    return True


def check_not_fixed_point(
    traj: np.ndarray, tail_prop: float = 0.05, atol: float = 1e-3
) -> bool:
    """
    Check if the system trajectory converges to a fixed point.
    Actually, this tests the variance decay in the trajectory to detect a fixed point.

    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        tail_prop: Proportion of the trajectory to consider for variance comparison.
        atol: Absolute tolerance for detecting a fixed point.
    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    n = traj.shape[1]
    tail = int(tail_prop * n)
    distances = np.linalg.norm(np.diff(traj[:, -tail:], axis=1), axis=0)

    if np.allclose(distances, 0, atol=atol):
        return False

    return True


def check_not_trajectory_decay(
    traj: np.ndarray, tail_prop: float = 0.5, atol: float = 1e-3
) -> bool:
    """
    Check if a multi-dimensional trajectory is not decaying.
    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        tail_prop: Proportion of the trajectory to consider for variance comparison.
        atol: Absolute tolerance for detecting a fixed point.
    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    # Check if any dimension of the trajectory is a straight line in the last tail_prop of the trajectory
    n = traj.shape[1]
    tail = int(tail_prop * n)
    for dim in range(traj.shape[0]):
        diffs = np.diff(traj[dim, -tail:])
        if np.allclose(diffs, 0, atol=atol):
            return False
    return True


def check_not_limit_cycle(
    traj: np.ndarray,
    n_timepoints_to_check: int = 512,
    tolerance: float = 1e-2,
    min_prop_recurrences: float = 0.1,
    min_counts_per_rtime: int = 100,
    min_block_length: int = 50,
    min_recurrence_time: int = 1,
    max_block_separation: int = 0,
    num_blocks_to_find: int = 2,
) -> bool:
    """
    Check if a trajectory is NOT a limit cycle using recurrence analysis.

    This function implements a multi-heuristic approach to detect limit cycles in dynamical
    system trajectories. It uses recurrence analysis to identify periodic behavior by
    analyzing the distance matrix between trajectory points and applying several criteria
    to distinguish between chaotic and periodic dynamics.

    Args:
        traj: Trajectory data of shape (n_dimensions, n_timepoints)
        n_timepoints_to_check: Number of timepoints to check for recurrence analysis
        tolerance: Distance threshold for recurrence detection (points closer than this are considered recurrent)
        min_prop_recurrences: Minimum proportion of recurrences required relative to total timepoints
        min_counts_per_rtime: Minimum number of occurrences required for any recurrence time
        min_block_length: Minimum length of consecutive blocks for block-based analysis
        min_recurrence_time: Minimum recurrence time to consider (filters out self-recurrences)
        max_block_separation: Maximum allowed separation between block endpoints for relaxed block detection
        num_blocks_to_find: Number of valid blocks required for early termination

    Returns:
        bool: True if trajectory is NOT a limit cycle, False if it likely is a limit cycle

    Note:
        The function applies three main heuristics:
        1. Minimum proportion of recurrences: Ensures sufficient periodic behavior
        2. Minimum counts per recurrence time: Validates consistent periodic patterns
        3. Consecutive block analysis: Identifies structured periodic sequences
    """

    traj = traj[:, :n_timepoints_to_check]
    n_timepoints = traj.shape[1]

    # standardize the trajectory
    traj = safe_standardize(traj)
    # Calculate upper triangular distance matrix
    dist_matrix = np.triu(
        cdist(traj.T, traj.T, metric="euclidean").astype(np.float16), k=1
    )

    # Find recurrence indices
    recurrence_mask = (dist_matrix < tolerance) & (dist_matrix > 0)
    t1_indices, t2_indices = recurrence_mask.nonzero()

    if len(t1_indices) == 0:
        # print("No recurrences found")
        return True

    # Calculate recurrence times
    recurrence_times = np.abs(t1_indices - t2_indices)
    valid_times = recurrence_times[recurrence_times >= min_recurrence_time]

    # Heuristic 1: Check minimum proportion of recurrences
    if len(valid_times) < int(min_prop_recurrences * n_timepoints):
        # print(
        #     f"Not enough recurrences: {len(valid_times)} < {int(min_prop_recurrences * n_timepoints)}"
        # )
        return True
    # print(f"Found {len(valid_times)} recurrences")

    # Heuristic 2: Check minimum counts per recurrence time
    time_counts = Counter(valid_times)
    if not any(count >= min_counts_per_rtime for count in time_counts.values()):
        # print(
        #     f"Not enough counts per recurrence time: {time_counts} < {min_counts_per_rtime}"
        # )
        return True

    # Heuristic 3: Check for consecutive blocks (only if min_block_length > 1)
    if min_block_length > 1 and not _has_valid_blocks(
        t1_indices,
        t2_indices,
        min_block_length,
        min_recurrence_time,
        max_block_separation,
        num_blocks_to_find,
    ):
        # print(f"No valid blocks found with min_block_length={min_block_length}")
        return True

    return False


def _has_valid_blocks(
    t1_indices: np.ndarray,
    t2_indices: np.ndarray,
    min_block_length: int,
    min_recurrence_time: int,
    max_block_separation: int = 0,
    num_blocks_to_find: int = 2,
) -> bool:
    """Check for valid consecutive blocks in recurrence analysis.

    This function analyzes pairs of recurrence indices to identify consecutive blocks
    where the recurrence time remains constant and the indices form consecutive sequences.
    A valid block indicates potential periodic behavior in the dynamical system.

    Args:
        t1_indices: Array of first time indices in recurrence pairs.
        t2_indices: Array of second time indices in recurrence pairs.
        min_block_length: Minimum length required for a block to be considered valid.
        min_recurrence_time: Minimum recurrence time threshold to consider.
        max_block_separation: Maximum allowed separation between block endpoints for relaxed block detection.
        num_blocks_to_find: Number of valid blocks required for early termination.

    Returns:
        bool: True (limit cycle signal) if valid blocks are found according to the criteria, False otherwise.

    Note:
        The function implements termination when either:
        - A block of length >= 2 * min_block_length is found (Early Termination)
        - At least num_blocks_to_find valid blocks are found
        - For relaxed blocks, defined as blocks with endpoint separation < max_block_separation,
            when number of relaxed blocks >= min_block_length * max_block_separation
            We provide this feature to allow for more flexible block detection.
                For example, if the recurrences happen every other timestep, for a given recurrence time,
                we can use the relaxed block detection to still identify a block if we want a more strict limit cycle criteria.
    """
    # Vectorize recurrence time calculation
    rtimes = np.abs(t2_indices - t1_indices)
    valid_mask = rtimes >= min_recurrence_time

    if not np.any(valid_mask):
        return False

    # Get valid indices
    valid_t1 = t1_indices[valid_mask]
    valid_t2 = t2_indices[valid_mask]
    valid_rtimes = rtimes[valid_mask]

    # Vectorize consecutive checks
    t1_diffs = np.diff(valid_t1)
    t2_diffs = np.diff(valid_t2)
    rtime_diffs = np.diff(valid_rtimes)

    # Find where consecutive conditions are met
    consecutive_mask = (t1_diffs == 1) & (t2_diffs == 1) & (rtime_diffs == 0)

    # Count blocks efficiently
    block_length = 1
    num_blocks = 0

    for is_consecutive in consecutive_mask:
        if is_consecutive:
            block_length += 1
        else:
            if block_length >= min_block_length:
                num_blocks += 1
            block_length = 1

        # Early termination
        if block_length >= min_block_length * 2 or num_blocks >= num_blocks_to_find:
            return True

    # Check final block
    if block_length >= min_block_length:
        num_blocks += 1

    # Termination if we found enough blocks
    if num_blocks >= num_blocks_to_find:
        return True

    if max_block_separation == 0:
        return False

    # Vectorize endpoint distance calculation
    block_endpoints_distances = np.abs(valid_t1 - valid_t2)
    num_relaxed_blocks = np.sum(block_endpoints_distances < max_block_separation)

    enough_relaxed_blocks = (
        num_relaxed_blocks >= min_block_length * max_block_separation
    )
    return bool(enough_relaxed_blocks)


def check_lyapunov_exponent(traj: np.ndarray, traj_len: int = 100) -> bool:
    """
    Check if the Lyapunov exponent of the trajectory is greater than 1.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
    Returns:
        bool: False if the Lyapunov exponent is less than 1, True otherwise.
    """
    # TODO: debug this, the rosenstein implementation expects univariate time series, not broadcastable
    lyapunov_exponent = max_lyapunov_exponent_rosenstein(
        traj.T, trajectory_len=traj_len
    )
    if lyapunov_exponent < 0:
        return False
    return True


def check_power_spectrum(
    traj: np.ndarray,
    n_timepoints_to_check: int | None = None,
    rel_peak_height: float = 1e-5,
    rel_prominence: float = 1e-5,
    min_peaks: int = 20,
    ndims_required: int = 1,
) -> bool:
    """Check if a multi-dimensional trajectory has characteristics of chaos via power spectrum.

    Args:
        traj: Array of shape (num_vars, num_timepoints)
        rel_peak_height: Minimum relative peak height threshold
        rel_prominence: Minimum relative peak prominence threshold
        min_peaks: Minimum number of significant peaks for chaos
        ndims_required: Minimum number of dimensions required to have at least min_peaks peaks to classify the system as potentially chaotic. Default is 1.


    Returns:
        True if the system exhibits chaotic characteristics
    """
    if n_timepoints_to_check is not None:
        traj = traj[:, :n_timepoints_to_check]
    power = np.abs(rfft(traj, axis=1)) ** 2  # type: ignore

    power_maxes = power.max(axis=1)
    power_mins = power.min(axis=1)

    peaks_per_dim = [
        find_peaks(
            power[dim],
            height=max(rel_peak_height * power_maxes[dim], power_mins[dim]),
            prominence=max(rel_prominence * power_maxes[dim], power_mins[dim]),
        )[0]
        for dim in range(power.shape[0])
    ]

    return sum(len(peaks) >= min_peaks for peaks in peaks_per_dim) >= ndims_required


def check_not_linear(
    traj: np.ndarray,
    r2_threshold: float = 0.98,
    eps: float = 1e-10,
    require_all_dims: bool = True,
) -> bool:
    """Check if n-dimensional trajectory follows a straight line using PCA.

    Args:
        traj: Array of shape (num_dims, num_timepoints)
        r2_threshold: Variance explained threshold above which trajectory is considered linear
        eps: Small value to prevent division by zero
        require_all_dims: If True, require all dimensions to be linear. If False, require at least one dimension to be linear.

    Returns:
        bool: False if trajectory is linear, True otherwise
    """
    if np.any(~np.isfinite(traj)):
        return False

    traj = safe_standardize(traj)

    try:
        s = np.linalg.svd(traj.T, full_matrices=False, compute_uv=False)
        explained_variance_ratio = s**2 / (np.sum(s**2) + eps)
        if require_all_dims:
            res = bool(np.all(explained_variance_ratio <= r2_threshold))
        else:
            res = bool(np.any(explained_variance_ratio <= r2_threshold))
        return bool(res)
    except Exception as e:
        print(f"Error in check_not_linear: {e}")
        return True  # fallback if SVD fails


def check_stationarity(traj: np.ndarray, p_value: float = 0.05) -> bool:
    """
    ADF tests for presence of a unit root, with null hypothesis that time_series is non-stationary.
    KPSS tests for stationarity around a constant (or deterministic trend), with null hypothesis that time_series is stationary.

    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        p_value: float = 0.05, significance level for stationarity tests

    Returns:
        bool: True if the trajectory is stationary, False otherwise.
    """
    if not has_module("statsmodels"):
        raise ImportError("statsmodels is required for stationarity check")
    for d in range(traj.shape[0]):
        coord = traj[d, :]

        try:
            result_adf = adfuller(coord, autolag="AIC")
            result_kpss = kpss(coord, regression="c")
        except ValueError:  # probably due to constant values
            return False

        status_adf = result_adf[1] < p_value
        status_kpss = result_kpss[1] >= p_value

        if not status_adf and not status_kpss:
            return False

    return True


def check_zero_one_test(
    traj: np.ndarray,
    threshold: float = 0.5,
    strategy: Literal["median", "mean", "score"] = "median",
    standardize: bool = False,
) -> bool:
    """
    Compute the zero-one test for a specified system.
    If any dimension is chaotic according to the zero-one test, we soft-pass the system as chaotic.

    Parameters:
        trajectories: np.ndarray of shape (n_samples, n_dims, timesteps)
        threshold: float, threshold on the median of the zero-one test to decide if the system is chaotic
    Returns:
        bool, True if the system is chaotic, False otherwise
    """
    if standardize:
        traj = safe_standardize(traj)
    # go dimension by dimension
    agg_fn = np.median if strategy == "median" else np.mean
    if strategy == "score":
        agg_fn = lambda x: np.sum(x >= threshold) / len(x)

    for dim in range(traj.shape[0]):
        timeseries = traj[dim, :].squeeze()
        K_vals = run_zero_one_sweep(
            timeseries, c_min=np.pi / 5, c_max=4 * np.pi / 5, k=1, n_runs=100
        )
        if agg_fn(K_vals) >= threshold:
            return True
    return False


def check_smooth(
    traj: np.ndarray, freq_threshold: float = 0.3, jump_std_factor: float = 3.0
) -> bool:  # type: ignore
    """
    Check if a multi-dimensional trajectory is smooth.
    """
    pass
