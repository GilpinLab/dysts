import logging
import os
import time
import warnings
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, List, Literal, Union

import numpy as np

try:
    from gluonts.dataset.arrow import ArrowWriter
except ImportError:
    warnings.warn("gluonts not installed, function convert_to_arrow will not work")


def safe_standardize(
    arr: np.ndarray,
    epsilon: float = 1e-10,
    axis: int = -1,
    context: np.ndarray | None = None,
    denormalize: bool = False,
) -> np.ndarray:
    """
    Standardize the trajectories by subtracting the mean and dividing by the standard deviation

    Args:
        arr: The array to standardize
        epsilon: A small value to prevent division by zero
        axis: The axis to standardize along
        context: The context to use for standardization. If provided, use the context to standardize the array.
        denormalize: If True, denormalize the array using the mean and standard deviation of the context.

    Returns:
        The standardized array
    """
    # if no context is provided, use the array itself
    context = arr if context is None else context

    assert arr.ndim == context.ndim, (
        "arr and context must have the same num dims if context is provided"
    )
    assert axis < arr.ndim and axis >= -arr.ndim, (
        "invalid axis specified for standardization"
    )
    mean = np.nanmean(context, axis=axis, keepdims=True)
    std = np.nanstd(context, axis=axis, keepdims=True)
    std = np.where(std < epsilon, epsilon, std)
    if denormalize:
        return arr * std + mean
    return (arr - mean) / std


def demote_from_numpy(param: float | np.ndarray) -> float | list[float]:
    """
    Demote a float or numpy array to a float or list of floats
    Used for serializing parameters to json
    """
    if isinstance(param, np.ndarray):
        return param.tolist()
    return param


def dict_demote_from_numpy(
    param_dict: dict[str, float | np.ndarray],
) -> dict[str, float | list[float]]:
    """
    Demote a dictionary of parameters to a dictionary of floats or list of floats
    """
    return {k: demote_from_numpy(v) for k, v in param_dict.items()}


def timeit(logger: logging.Logger | None = None) -> Callable:
    """Decorator that measures and logs execution time of a function.

    Args:
        logger: Optional logger to use for timing output. If None, prints to stdout.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            if elapsed < 60:
                time_str = f"{elapsed:.2f} seconds"
            elif elapsed < 3600:
                time_str = f"{elapsed / 60:.2f} minutes"
            else:
                time_str = f"{elapsed / 3600:.2f} hours"

            msg = f"{func.__name__} took {time_str}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

            return result

        return wrapper

    return decorator


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: Literal["lz4", "zstd"] = "lz4",
    split_coords: bool = False,
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and time_series.ndim == 2
    ), "time_series must be a list of 1D numpy arrays or a 2D numpy array"

    # GluonTS requires this datetime format for reading arrow file
    start = datetime.now().strftime("%Y-%m-%d %H:%M")

    if split_coords:
        dataset = [{"start": start, "target": ts} for ts in time_series]
    else:
        dataset = [{"start": start, "target": time_series}]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=Path(path),
    )


def process_trajs(
    base_dir: str,
    timeseries: dict[str, np.ndarray],
    split_coords: bool = False,
    verbose: bool = False,
    sample_idx: int = 0,
    num_periods: int | None = None,
) -> None:
    """
    Saves each trajectory in timeseries ensemble to a separate directory

    Args:
        base_dir: The base directory to save the trajectories to
        timeseries: A dictionary mapping system names to numpy arrays of shape
            (num_eval_windows * num_datasets, num_channels, T) where T is the prediction
            length or context length.
        split_coords: Whether to split the coordinates by dimension
        verbose: Whether to print verbose output
        sample_idx: The sample index to use for the trajectories
        num_periods: The (optionally provided) number of "periods" (on a Fourier or Lyapunov timescale) used to generate these trajectories
    """
    for sys_name, trajectories in timeseries.items():
        if verbose:
            print(
                f"Processing trajectories of shape {trajectories.shape} for system {sys_name}"
            )

        system_folder = os.path.join(base_dir, sys_name)
        os.makedirs(system_folder, exist_ok=True)

        for i, trajectory in enumerate(trajectories):
            # trajectory has shape (num_dims, num_timesteps) or (num_timesteps,)
            curr_sample_idx = sample_idx + i

            if trajectory.ndim == 1:
                # if trajectory is 1D, make it 2D, (1, num_timesteps)
                trajectory = np.expand_dims(trajectory, axis=0)
            if verbose:
                print(
                    f"Saving {sys_name} trajectory {curr_sample_idx} with shape {trajectory.shape}"
                )

            filename = (
                f"{curr_sample_idx}_T-{trajectory.shape[-1]}_dim-{trajectory.shape[0]}"
            )
            if num_periods is not None:
                filename += f"_period-{num_periods}"
            filename += ".arrow"
            path = os.path.join(system_folder, filename)

            convert_to_arrow(path, trajectory, split_coords=split_coords)
