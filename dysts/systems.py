"""Utilities for the implemented systems"""

import inspect
import json
from functools import partial
from multiprocessing import Pool
from os import PathLike
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from . import flows as dfl
from . import maps as dmp
from .base import (
    DATAPATH_CONTINUOUS,
    DATAPATH_DISCRETE,
    BaseDyn,
    DynMap,
    DynSys,
    DynSysDelay,
)
from .utils import num_unspecified_params

Array = npt.NDArray[np.float64]


def get_attractor_list(
    sys_class: str = "continuous", exclude: list[str] = []
) -> list[str]:
    """Get names of implemented dynamical systems

    Args:
        sys_class: class of systems to get the name of - must
            be one of ['continuous', 'continuous_no_delay', 'delay', 'discrete']
        exclude: list of systems to exclude, optional

    Returns:
        Sorted list of systems belonging to sys_class
    """
    module: ModuleType
    if sys_class in ["continuous", "continuous_no_delay", "delay"]:
        module = dfl
        parent_class = {
            "continuous": (DynSys, DynSysDelay),
            "continuous_no_delay": DynSys,
            "delay": DynSysDelay,
        }[sys_class]
    elif sys_class == "discrete":
        module = dmp
        parent_class = DynMap
    else:
        raise Exception(
            "sys_class must be in ['continuous', 'continuous_no_delay', 'delay', 'discrete']"
        )

    systems: list[tuple[str, BaseDyn]] = inspect.getmembers(
        module,
        lambda obj: inspect.isclass(obj)  # type: ignore
        and issubclass(obj, parent_class)  # type: ignore
        and obj.__module__ == module.__name__,
    )

    systems = list(filter(lambda x: x[0] not in exclude, systems))

    return sorted([name for name, _ in systems])


def get_system_data(
    sys_class: str = "continuous", exclude: list[str] = []
) -> dict[str, Any]:
    """Get system data from the dedicated system class json files

    Arguments:
        sys_class: class of systems to get the name of - must
            be one of ['continuous', 'continuous_no_delay', 'delay', 'discrete']
        exclude: list of systems to exclude, optional

    Returns:
        Data from json file filtered by sys_class as a dict
    """
    if sys_class in ["continuous", "continuous_no_delay", "delay"]:
        datapath = DATAPATH_CONTINUOUS
    elif sys_class == "discrete":
        datapath = DATAPATH_DISCRETE
    else:
        raise Exception(
            "sys_class must be in ['continuous', 'continuous_no_delay', 'delay', 'discrete']"
        )

    systems = get_attractor_list(sys_class, exclude)
    with open(datapath, "r") as file:
        data = json.load(file)

    # filter out systems from the data
    return {k: v for k, v in data.items() if k in systems}


def _resolve_event_signature(
    system: BaseDyn,
    event: Callable[[BaseDyn], Callable[[float, Array], float]]
    | Callable[[float, Array], float],
) -> Callable[[float, Array], float]:
    """
    Hack check for handling event function signatures

    Case 1: event function is system dependent (1 parameter factory function)
        event(system) -> (event_fn: Callable[[float, Array], float])
    Case 2: event function is not system dependent (solve_ivp compatible event function)
        event(t, y) -> (float)
    Case 3: event function is a parameterless factory function for a solve_ivp event
        event() -> (event_fn: Callable[[float, Array], float])

    Args:
        system: The system to pass to the possibly system dependent event function
        event: The event function to resolve

    Returns:
        The resolved event function (solve_ivp compatible)

    Raises:
        ValueError: If the event function takes an invalid number of arguments
    """
    num_params = num_unspecified_params(event)
    # case 1: event function is system dependent
    if num_params == 1:
        return event(system)  # type: ignore
    # case 2: event function is not system dependent
    elif num_params == 2:
        return event  # type: ignore
    # case 3: event function is a parameterless factory function for a solve_ivp event
    elif num_params == 0:
        return event()  # type: ignore
    else:
        raise ValueError("Event function can only take 0, 1, or 2 arguments")


def _compute_trajectory(
    n: int,
    system: str | BaseDyn,
    event_fns: Sequence[Callable] | None,
    silent_errors: bool = False,
    return_times: bool = False,
    **kwargs: Any,
) -> Array | tuple[Array, Array] | None | tuple[None, None]:
    """Helper function to compute a single trajectory for a dynamical system.

    Args:
        n: Number of timepoints to integrate
        system: Either a string name of a system or a system instance
        event_fns: A list of functions that take a dynamical system and returns a
            solve_ivp compatible event function
        silent_errors: Whether to fail silently if an error occurs
        return_times: Whether to return the timepoints at which the solution was computed
        **kwargs: Additional arguments passed to make_trajectory

    Returns:
        The computed trajectory, or None if error occurs and silent_errors=True, and a tuple of (timepoints, trajectory) if return_times=True
    """
    sys = getattr(dfl, system)() if isinstance(system, str) else system

    # if event functions are provided, resolve them and pass into make_trajectory as a kwarg
    events = (
        None
        if event_fns is None
        else [_resolve_event_signature(sys, event_fn) for event_fn in event_fns]
    )
    try:
        res = sys.make_trajectory(n, events=events, return_times=return_times, **kwargs)
        if return_times:
            ts, traj = res
            return (ts, traj)
        return res
    except Exception:
        if silent_errors:
            return (None, None) if return_times else None
        raise


def make_trajectory_ensemble(
    n: int,
    use_tqdm: bool = True,
    use_multiprocessing: bool = False,
    subset: Sequence[str] | Sequence[BaseDyn] | None = None,
    event_fns: Sequence[Callable] | None = None,
    silent_errors: bool = False,
    multiprocess_kwargs: dict[str, Any] = {},
    **kwargs,
) -> dict[str, Array | tuple[Array, Array] | None | tuple[None, None]]:
    """
    Integrate multiple dynamical systems with identical settings

    Args:
        n (int): The number of timepoints to integrate
        use_tqdm (bool): Whether to use a progress bar
        use_multiprocessing (bool): Not yet implemented.
        subset (list): A list of system names or BaseDyn (e.g. custom dynamical systems). Defaults to all continuous systems.
            Can also pass in `sys_class` as a kwarg to specify other system classes.
        event_fns (list): A list of functions that can take the signatures:
                1. (system: BaseDyn) -> (event_fn: Callable[[float, Array], float])
                2. (event_fn: Callable[[float, Array], float])
                3. () -> (event_fn: Callable[[float, Array], float])
            If multiprocessing is used, the event functions must be of type 1 or 3 to avoid state persistence across processes.
        silent_errors (bool): Whether to fail silently if an error occurs
        kwargs (dict): Integration options passed to each system's make_trajectory() method

    Returns:
        all_sols (dict): A dictionary containing trajectories for each system

    """
    if subset is None:
        sys_class = kwargs.pop("sys_class", "continuous")
        exclude = kwargs.pop("exclude", [])
        subset = get_attractor_list(sys_class, exclude)

    if use_tqdm and not use_multiprocessing:
        subset = tqdm(subset, desc="Integrating systems")  # type: ignore

    all_sols = dict()
    if use_multiprocessing:
        all_sols = _multiprocessed_compute_trajectory(
            n,
            subset or [],
            event_fns,
            silent_errors,
            multiprocess_kwargs=multiprocess_kwargs,
            **kwargs,
        )
    else:
        for system in subset or []:
            sol = _compute_trajectory(n, system, event_fns, silent_errors, **kwargs)
            equation_name = system if isinstance(system, str) else system.name
            all_sols[equation_name] = sol

    return all_sols


def _multiprocessed_compute_trajectory(
    n: int,
    subset: Sequence[str] | Sequence[BaseDyn],
    event_fns: Sequence[Callable] | None,
    silent_errors: bool = False,
    multiprocess_kwargs: dict[str, Any] = {},
    **kwargs: Any,
) -> dict[str, Array | tuple[Array, Array] | None | tuple[None, None]]:
    """Helper for handling multiprocessed integration with _compute_trajectory"""

    # loose check to just count the number of unspecified parameters in the event functions
    # assert that in the multiprocessing case, all event functions are of type 1 or 3
    # this is to prevent potential state persistence across processes
    if event_fns is not None:
        assert all(num_unspecified_params(event) in [0, 1] for event in event_fns)

    with Pool(**multiprocess_kwargs) as pool:
        solutions = pool.starmap(
            partial(_compute_trajectory, **kwargs),
            [(n, system, event_fns, silent_errors) for system in subset],
        )

    names = [sys if isinstance(sys, str) else sys.name for sys in subset]
    return dict(zip(names, solutions))


def compute_trajectory_statistics(
    n: int,
    subset: Sequence[str],
    datapath: PathLike | None = None,
    **kwargs,
) -> dict[str, dict[str, Array]]:
    """
    Compute mean and std for given trajectory list
    Args:
        n (int): The number of timepoints to integrate
        subset (list): A list of system names. Defaults to all continuous systems.
            Can also pass in `sys_class` as a kwarg to specify other system classes.
        datapath (str): Path to save the computed statistics to a JSON file
        kwargs: Additional keyword arguments passed to the integration routine
    """
    # Ensure return_times=False for statistics computation
    kwargs.setdefault("return_times", False)
    sols = make_trajectory_ensemble(n, subset=subset, **kwargs)
    stats = {
        name: {"mean": sol.mean(axis=0), "std": sol.std(axis=0)}
        for name, sol in sols.items()
        if sol is not None and not isinstance(sol, tuple)
    }

    # Save the computed statistics to a JSON file
    if datapath is not None:
        with open(datapath) as f:
            data = json.load(f)

        for system in subset:
            data[system].update({k: v.tolist() for k, v in stats[system].items()})

        with open(datapath, "w") as f:
            json.dump(data, f, indent=2)

    return stats
