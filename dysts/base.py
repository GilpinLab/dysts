"""Dynamical systems in Python"""

import gzip
import json
import logging
import warnings
from functools import partial
from importlib import resources
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from .coupling import RandomAdditiveCouplingMap
from .utils import cast_to_numpy, ddeint, has_module, integrate_dyn, standardize_ts

logger = logging.getLogger(__name__)

if has_module("numba"):
    logger.info("[numba] JIT compilation enabled")
    from numba import njit

else:

    def njit(func):
        return func


BASE_REQUIRED_METADATA = ("parameters", ("dimension", "embedding_dimension"))

DATAPATH_CONTINUOUS = str(
    resources.files("dysts").joinpath("data/chaotic_attractors.json")
)
DATAPATH_DISCRETE = str(resources.files("dysts").joinpath("data/discrete_maps.json"))


def staticjit(func: Callable) -> Callable:
    """Decorator to apply numba's njit decorator to a static method"""
    return staticmethod(njit(func))


class BaseDyn:
    """A base class for dynamical systems"""

    name: str
    metadata_path: str | None
    metadata: dict[str, Any]
    required_fields: tuple[str | tuple[str, ...], ...]
    params: dict[str, Any]
    param_list: list[Any]
    ic: np.ndarray
    dimension: int
    mean: np.ndarray
    std: np.ndarray
    _postprocessing: Callable[..., np.ndarray] | None
    unbounded_indices: list[int]

    def __init__(
        self,
        metadata_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        required_fields: tuple[str | tuple[str, ...], ...] = BASE_REQUIRED_METADATA,
        **extra_metadata,
    ) -> None:
        """
        Initialize the dynamical system with metadata.

        Args:
            metadata_path (Optional[str]): Path to the JSON file containing metadata.
            metadata (Optional[Dict[str, Any]]): Dictionary containing metadata.
            required_fields (Tuple[str, ...]): Required metadata fields.
            **extra_metadata: Additional metadata as keyword arguments.

        Raises:
            AssertionError: If no metadata is provided or if required fields are missing.

        Note:
            This method sets various attributes of the class instance based on the
            metadata, including system name, parameters, initial conditions, and
            computed statistics like mean and standard deviation.
        """
        self.name = self.__class__.__name__
        self.metadata_path = metadata_path
        self.metadata = metadata or {}
        self.required_fields = required_fields

        # optionally load system attributes and computed quantities from a JSON file
        if self.metadata_path is not None:
            self.metadata.update(
                self.load_system_metadata(self.name, self.metadata_path)
            )

        # update the metadata with any extra metadata provided that is not None
        self.metadata.update({k: v for k, v in extra_metadata.items() if v is not None})

        assert len(self.metadata) > 0, "No metadata provided"
        assert all(  # check that all required fields are present
            any(k in key for k in self.metadata)
            if isinstance(key, tuple)
            else key in self.metadata
            for key in required_fields
        ), (
            f"The provided metadata {self.metadata} is missing some required fields: {required_fields}"
        )

        self.params = self.metadata["parameters"]
        self.params = {k: cast_to_numpy(v) for k, v in self.params.items()}
        self.__dict__.update(self.params)
        self.param_list = [self.params[key] for key in sorted(self.params.keys())]

        if "initial_conditions" in self.metadata:
            self.ic = cast_to_numpy(
                self.metadata.pop("initial_conditions"), singleton_scalar=True
            )
        self.ic = np.array(self.ic)  # safety

        if "dimension" in self.metadata:
            self.dimension = self.metadata.pop("dimension")
        elif "embedding_dimension" in self.metadata:
            self.dimension = self.metadata.pop("embedding_dimension")

        # set all attributes in the metadata dictionary
        # do this last so the user can override any of the above
        for key in self.metadata:
            setattr(self, key, self.metadata[key])

        if hasattr(self, "dimension"):
            self.mean = np.asarray(getattr(self, "mean", np.zeros(self.dimension)))
            self.std = np.asarray(getattr(self, "std", np.ones(self.dimension)))
        elif hasattr(self, "ic"):
            self.mean = np.asarray(getattr(self, "mean", np.zeros_like(self.ic)))
            self.std = np.asarray(getattr(self, "std", np.ones_like(self.ic)))

    @staticmethod
    def load_system_metadata(system_name: str, data_path: str) -> dict[str, Any]:
        """
        Load data from a JSON file

        Returns None if the system name is not found
        """
        with open(data_path, "r") as file:
            data = json.load(file)

        if system_name in data:
            return data[system_name]
        else:
            warnings.warn(f"No metadata available for {system_name}")
            return {}

    @staticmethod
    def load_trajectory(
        data_path: str, system_name: str, return_times=False, standardize=False
    ):
        """
        Load a precomputed trajectory for the dynamical system

        Args:
            data_path (str): Path to the data file, of format {name}.json.gz
            standardize (bool): Standardize the output time series.
            return_times (bool): Whether to return the timepoints at which the solution
                was computed

        Returns:
            sol (ndarray): A T x D trajectory
            tpts, sol (ndarray): T x 1 timepoint array, and T x D trajectory

        """
        with gzip.open(data_path, "rt", encoding="utf-8") as file:
            dataset = json.load(file)

        tpts, sol = (
            np.array(dataset[system_name]["time"]),
            np.array(dataset[system_name]["values"]),
        )

        if standardize:
            sol = standardize_ts(sol)

        return (tpts, sol) if return_times else sol

    @staticmethod
    def _rhs(X, t):
        """The right-hand side of the dynamical system. Overwritten by the subclass"""
        raise NotImplementedError

    @staticmethod
    def _jac(X, t, *args):
        """The Jacobian of the dynamical system. Overwritten by the subclass"""
        raise NotImplementedError

    def has_jacobian(self) -> bool:
        """Check if the subclass has implemented the _jac method."""
        # NOTE: the first comparison doesn't actually work
        return self._jac is not BaseDyn._jac and self._jac.__dict__ != {}

    def transform_ic(
        self, transform_fn: Callable[[np.ndarray, Any], np.ndarray | None]
    ) -> bool:
        """Updates the initial condition via a transform function"""
        transformed_ic = transform_fn(self.ic, system=self)  # type: ignore

        if transformed_ic is None:
            return False

        self.ic = transformed_ic
        return True

    def transform_params(
        self, transform_fn: Callable[[str, np.ndarray, Any], np.ndarray | None]
    ) -> bool:
        """Updates the current parameter list via a transform function"""
        transformed_params = {
            param_name: transform_fn(param_name, param_value, system=self)  # type: ignore
            for param_name, param_value in zip(
                sorted(self.params.keys()), self.param_list
            )
        }

        if any(p is None for p in transformed_params.values()):
            return False

        self.params = transformed_params
        self.__dict__.update(self.params)
        self.param_list = [self.params[key] for key in sorted(self.params.keys())]
        return True

    def make_trajectory(self, *args, **kwargs):
        """Make a trajectory for the dynamical system"""
        raise NotImplementedError


class DynSys(BaseDyn):
    """A continuous dynamical system base class"""

    dt: float
    period: float
    maximum_lyapunov_estimated: float
    unbounded_indices: list[int]

    def __init__(
        self,
        metadata_path: str | None = DATAPATH_CONTINUOUS,
        parameters: dict[str, ArrayLike] | None = None,
        dt: float | None = None,
        period: float | None = None,
        maximum_lyapunov_estimated: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            metadata_path=metadata_path,
            parameters=parameters,
            dt=dt,
            period=period,
            maximum_lyapunov_estimated=maximum_lyapunov_estimated,
            **kwargs,
        )

    def rhs(self, X, t):
        """The right hand side of a dynamical equation"""
        return self._rhs(*X.T, t, *self.param_list)  # type: ignore

    def jac(self, X, t):
        """The Jacobian of the dynamical system"""
        return self._jac(*X.T, t, *self.param_list)

    def __call__(self, X, t):
        """Wrapper around right hand side"""
        return self.rhs(X, t)

    def make_trajectory(
        self,
        n: int,
        dt: float = 1e-3,
        init_cond: np.ndarray | None = None,
        resample: bool = True,
        pts_per_period: int = 100,
        return_times: bool = False,
        standardize: bool = False,
        postprocess: bool = True,
        noise: float = 0.0,
        timescale: str = "Fourier",
        method: str = "Radau",
        random_seed: int = 0,
        rtol: float = 1e-12,
        atol: float = 1e-12,
        verbose: bool = False,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None | tuple[None, None]:
        """
        Generate a fixed-length trajectory for the dynamical system.

        Args:
            n: Total number of trajectory points.
            dt: Time step for integration. Defaults to 1e-3 or system's default if set.
            init_cond: Initial conditions. If None, uses system's default.
            resample: Whether to resample trajectory to match dominant Fourier components.
            pts_per_period: Number of points per period if resampling.
            return_times: If True, return time points along with trajectory.
            standardize: If True, standardize the output time series.
            postprocess: If True, apply coordinate conversions and rescalings.
            noise: Stochasticity level in integrated dynamics (corresponds to Brownian motion).
            timescale: Timescale for resampling. "Fourier" (default) or "Lyapunov".
            method: Integration method.
            random_seed: Seed for random number generation.
            rtol: Relative tolerance for integration.
            atol: Absolute tolerance for integration.
            verbose: Whether to trigger warnings.
            **kwargs: Additional arguments for integration routine.

        Returns:
            If return_times is False:
                np.ndarray: T x D trajectory array.
            If return_times is True:
                Tuple[np.ndarray, np.ndarray]: T x 1 time point array and T x D trajectory array.
            None: If no complete trajectories are found.

        Raises:
            ValueError: If an invalid timescale is provided.
        """
        assert isinstance(init_cond, (None, np.array)), (
            "initial condition must be a numpy array or None"
        )
        np.random.seed(random_seed)

        # set timescales and interpolation points for the solution
        dt = dt if self.dt is None else self.dt

        if not resample:
            tlim = (n - 1) * dt
        elif resample and timescale == "Fourier" and self.period is not None:
            tlim = self.period * (n / pts_per_period)
        elif (
            resample
            and timescale == "Lyapunov"
            and self.maximum_lyapunov_estimated is not None
        ):
            tlim = (1 / self.maximum_lyapunov_estimated) * (n / pts_per_period)
        else:
            raise ValueError(f"Invalid timescale resampling argument: {timescale}")

        tpts = np.linspace(0, tlim, n)

        # standardization and initial condition preprocessing
        if not hasattr(self, "ic") and init_cond is None:
            raise ValueError(
                "No initial conditions provided and no default initial conditions available for this system."
            )
        ics = init_cond if init_cond is not None else self.ic
        ics = np.expand_dims(ics, axis=0) if ics.ndim < 2 else ics

        mu = self.mean if standardize else np.zeros_like(ics[0])
        std = self.std if standardize else np.ones_like(ics[0])

        def standard_jac(t, X):
            return self.jac(X * std + mu, t) / std

        def standard_rhs(t, X):
            return self(X * std + mu, t) / std

        # compute trajectories
        sol = list()
        for ic in ics:
            traj = integrate_dyn(
                standard_rhs,
                (ic - mu) / std,
                tpts,
                dtval=dt,
                method=method,
                noise=noise,
                jac=standard_jac if self.has_jacobian() else None,
                rtol=rtol,
                atol=atol,
                **kwargs,
            )
            # check completeness of trajectory to kill off incomplete trajectories
            if traj.shape[-1] == len(tpts):  # full trajectory should have n points
                sol.append(traj)
            elif verbose:
                warnings.warn(
                    f"{self.name}: Integration did not complete for initial condition {ic}, only got {traj.shape[-1]} points. Skipping this initial condition."
                )

        if len(sol) == 0:  # if no complete trajectories, return None
            return (None, None) if return_times else None

        # transpose the trajectory to shape (B, T, D)
        sol = np.transpose(np.array(sol), (0, 2, 1))  # type: ignore

        # postprocess the trajectory, if necessary
        if hasattr(self, "_postprocessing") and postprocess:
            if verbose:
                warnings.warn(
                    "This system has at least one unbounded variable, which has been mapped to a bounded domain. Pass argument postprocess=False in order to generate trajectories from the raw system."
                )
            sol2 = np.moveaxis(sol, (-1, 0), (0, -1))  # type: ignore
            sol = np.moveaxis(np.dstack(self._postprocessing(*sol2)), (0, 1), (1, 0))  # type: ignore

        sol = np.squeeze(sol)  # type: ignore

        return (tpts, sol) if return_times else sol


class DynMap(BaseDyn):
    """A discrete map dynamical system class"""

    def __init__(
        self,
        metadata_path: str = DATAPATH_DISCRETE,
        parameters: dict[str, ArrayLike] | None = None,
        **kwargs,
    ):
        super().__init__(
            metadata_path=metadata_path,
            parameters=parameters,
            required_fields=("parameters",),
            **kwargs,
        )
        self._rhs_inv: Callable[..., Sequence[np.ndarray]] | None = None

    def rhs(self, X):
        """The right hand side of a dynamical map"""
        out = self._rhs(*X.T, *self.param_list)
        return np.vstack(out).T

    def rhs_inv(self, Xp):
        """The inverse of the right hand side of a dynamical map"""
        if self._rhs_inv is not None:
            out = self._rhs_inv(*Xp.T, *self.param_list)
        else:
            warnings.warn(
                f"The function _rhs_inv has not been implemented for {self.name}"
            )
            raise NotImplementedError
        return np.vstack(out).T

    def __call__(self, X):
        """Wrapper around right hand side"""
        return self.rhs(X)

    def make_trajectory(
        self,
        n: int,
        init_cond: np.ndarray | None = None,
        inverse: bool = False,
        return_times: bool = False,
        standardize: bool = False,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Generate a fixed-length trajectory with default parameters and initial condition(s).

        Args:
            n (int): The length of each trajectory.
            init_cond (Optional[np.ndarray]): Initial conditions. If None, use default.
            inverse (bool): Whether to reverse the trajectory.
            return_times (bool): Whether to return the timepoints of the solution.
            standardize (bool): Whether to standardize the output time series.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                If return_times is False, returns the trajectory.
                If return_times is True, returns a tuple of (timepoints, trajectory).

        """
        if not hasattr(self, "ic") and init_cond is None:
            raise ValueError(
                "No initial conditions provided and no default initial conditions available for this system."
            )
        ics = init_cond if init_cond is not None else self.ic
        curr = np.expand_dims(ics, axis=0) if ics.ndim < 2 else ics

        if inverse:
            propagator = self.rhs_inv
        else:
            propagator = self.rhs

        traj = np.zeros((curr.shape[0], n, curr.shape[-1]))

        for i in range(n):
            curr = propagator(curr)
            traj[:, i, :] = curr

        sol = np.squeeze(traj)

        if standardize:
            sol = standardize_ts(sol)

        if return_times:
            return np.arange(len(sol)), sol
        else:
            return sol


class DynSysDelay(BaseDyn):
    """
    A delayed differential equation object. Uses a exposed fork of ddeint
    The delay timescale is assumed to be the "tau" field. The embedding dimensions are set to a
    default value, but delay equations are infinite dimensional.
    """

    dt: float
    period: float
    maximum_lyapunov_estimated: float
    tau: float

    def __init__(
        self,
        metadata_path: str = DATAPATH_CONTINUOUS,
        dt: float | None = None,
        period: float | None = None,
        maximum_lyapunov_estimated: float | None = None,
        tau: float | None = None,
        parameters: dict[str, float | int | ArrayLike] | None = None,
        **kwargs,
    ):
        super().__init__(
            metadata_path=metadata_path,
            parameters=parameters,
            dt=dt,
            period=period,
            maximum_lyapunov_estimated=maximum_lyapunov_estimated,
            tau=tau,
            **kwargs,
        )

    def delayed_rhs(
        self, X: Callable[[float], ArrayLike], t: float, tau: float
    ) -> ArrayLike:
        """The right hand side of a dynamical equation"""
        xt, xt_delayed = X(t), X(t - tau)
        return self._rhs(xt, xt_delayed, t, *self.param_list)  # type: ignore

    def make_trajectory(
        self,
        n: int,
        init_cond: np.ndarray | None = None,
        dt: float = 1e-3,
        tau: float = 1,
        resample: bool = True,
        pts_per_period: int = 100,
        standardize: bool = False,
        timescale: str = "Fourier",
        return_times: bool = False,
        embedding_dim: int | None = None,
        history_function: Callable[[float], ArrayLike] | None = None,
        random_seed: int = 0,
        **kwargs,
    ):
        """
        Generate a fixed-length trajectory with default timestep, parameters, and
        initial conditions.

        Args:
            n (int): the total number of trajectory points
            method (str): Not used. Currently Euler is the only option here
            noise (float): The amplitude of brownian forcing
            resample (bool): whether to resample trajectories to have matching dominant
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            standardize (bool): Standardize the output time series.
            timescale (str): The timescale to use for resampling. "Fourier" (default) uses
                the dominant significant Fourier timescale, estimated using the periodogram
                of the system and surrogates. "Lyapunov" uses the Lypunov timescale of
                the system.
            return_times (bool): Whether to return the timepoints at which the solution
                was computed
            postprocess (bool): Whether to apply coordinate conversions and other domain-specific
                rescalings to the integration coordinates
            embedding_dim (int): Optionally augment solution with delay embedded trajectories.
                If equation is multi-dimensional with dimension d, then this will return a flattened
                delay embedding of shape (n, d*embedding_dim) where each consecutive (n, d) block
                will be a trajectory for a different delay parameter.
            history_function (callable): Function for specifying past conditions i.e.
                points for t < 0

        Todo:
            Support for multiple initial conditions
        """
        np.random.seed(random_seed)

        # timestep and time delay
        # add delay time increment for potential interpolation later on
        dt = dt if self.dt is None else self.dt
        tau = tau if self.tau is None else self.tau
        interp_pad = tau

        if not resample:
            tlim = (n - 1) * dt
        elif resample and timescale == "Fourier" and self.period is not None:
            tlim = self.period * (n / pts_per_period)
        elif (
            resample
            and timescale == "Lyapunov"
            and self.maximum_lyapunov_estimated is not None
        ):
            tlim = (1 / self.maximum_lyapunov_estimated) * (n / pts_per_period)
        else:
            raise ValueError(f"Invalid timescale resampling argument: {timescale}")

        tpts = np.linspace(0, tlim + interp_pad, n)

        emb_dim = self.dimension if embedding_dim is None else embedding_dim
        if not hasattr(self, "ic") and init_cond is None:
            raise ValueError(
                "No initial conditions provided and no default initial conditions available for this system."
            )
        ics = init_cond if init_cond is not None else self.ic

        # assume constant past points, overridable behavior
        # need to change initial conditions to NOT be the same size as the default embedding
        # dimensions, since the delay embedding happens as a postprocessing step
        history_fn = history_function or (lambda t: ics[0])
        sol = ddeint(partial(self.delayed_rhs, tau=tau), history_fn, tpts)

        # augment the trajectory with per-dimension delay embeddings
        interp_fns = [
            interp1d(
                tpts,
                sol[:, dim],
                axis=0,
                kind=kwargs.pop("kind", "linear"),
            )
            for dim in range(sol.shape[-1])
        ]

        sample_ts = np.linspace(tau, tpts[-1], n)
        sol = (
            np.stack(
                [
                    [fn(sample_ts - tau) for fn in interp_fns]
                    for tau in np.linspace(0, tau, emb_dim)
                ],
                axis=1,
            )
            .reshape(-1, len(tpts))
            .T
        )

        if standardize:
            sol = standardize_ts(sol)

        if return_times:
            return tpts, sol
        else:
            return sol


class SkewProduct(DynSys):
    driver: DynSys
    response: DynSys
    driver_dim: int

    # just class-level annotations, not constructor arguments
    response_dim: int
    coupling_map: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __init__(
        self,
        driver: DynSys,
        response: DynSys,
        coupling_map: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        _default_random_seed: int | None = None,
        **kwargs,
    ):
        """
        A skew-product dynamical system composed of a driver and response system.

        The driver system evolves independently while the response system is influenced by the driver
        through a coupling map. Inherits from dysts.base.DynSys to enable trajectory generation.

        Args:
            driver (DynSys): The autonomous driver dynamical system
            response (DynSys): The response dynamical system that is influenced by the driver
            coupling_map (Callable, optional):
                Function defining how driver and response systems are coupled.
                If None, defaults to RandomAdditiveCouplingMap.
            _default_random_seed (int, optional):
                Random seed for default coupling map initialization
        """
        # default to (randomly) additively forcing the response with the driver
        if coupling_map is None:
            self.coupling_map = RandomAdditiveCouplingMap(
                driver.dimension, response.dimension, random_seed=_default_random_seed
            )
        else:
            self.coupling_map = coupling_map

        super().__init__(
            metadata_path=None,
            parameters={},  # dummy: parameters are handled in the overwritten methods below
            dt=min(driver.dt, response.dt),
            period=max(driver.period, response.period),
            metadata={
                "name": f"{driver.name}_{response.name}",
                "dimension": driver.dimension + response.dimension,
                "driver": driver,
                "response": response,
                "driver_dim": driver.dimension,
                "response_dim": response.dimension,
            },
            **kwargs,
        )

        # hack: set a dummy param list for param count checks
        n_params = len(driver.params) + len(response.params)
        if hasattr(self.coupling_map, "n_params"):
            n_params += self.coupling_map.n_params
        self.param_list = [0 for _ in range(n_params)]

        assert hasattr(driver, "ic") and hasattr(response, "ic"), (
            "Driver and response must have default initial conditions"
            "and must be of the same dimension"
        )

        # manually set the unbounded indices derived from the coupling map
        self.unbounded_indices = self._update_unbounded_indices()

        self.ic = np.concatenate([self.driver.ic, self.response.ic])
        self.mean = np.concatenate([self.driver.mean, self.response.mean])
        self.std = np.concatenate([self.driver.std, self.response.std])

    def _update_unbounded_indices(self) -> list[int]:
        """Update the unbounded indices (which may depend on the coupling map)"""
        driver_inds = self.driver.unbounded_indices
        if hasattr(self.coupling_map, "unbounded_indices") and callable(
            self.coupling_map.unbounded_indices
        ):
            coupled_inds = self.coupling_map.unbounded_indices(
                driver_inds, self.response.unbounded_indices
            )
            return driver_inds + [i + self.driver_dim for i in coupled_inds]
        return driver_inds

    def transform_params(self, param_transform: Callable) -> bool:
        """Transform parameters of the driver & response systems and coupling map"""
        driver_success = self.driver.transform_params(param_transform)
        response_success = self.response.transform_params(param_transform)
        success = driver_success and response_success

        if hasattr(self.coupling_map, "transform_params"):
            coupling_success = self.coupling_map.transform_params(param_transform)

            if coupling_success:  # update upon successful coupling map transform
                self.unbounded_indices = self._update_unbounded_indices()

            success &= coupling_success

        return success

    def has_jacobian(self) -> bool:
        return self.driver.has_jacobian() and self.response.has_jacobian()

    def rhs(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Flow of the skew product system

        NOTE: the coupled RHS is always assumed to have the dimension of the response system
        """
        driver, response = X[: self.driver_dim], X[self.driver_dim :]
        driver_rhs = np.asarray(self.driver.rhs(driver, t))
        response_rhs = np.asarray(self.response.rhs(response, t))
        coupled_rhs = self.coupling_map(driver_rhs, response_rhs)
        return np.concatenate([driver_rhs, coupled_rhs])

    def jac(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Jacobian of the skew product system, computed via chain rule

        NOTE: the coupling map jacobian implements the jacobian wrt to the driver or the response flow
        """
        driver, response = X[: self.driver_dim], X[self.driver_dim :]

        driver_jac = np.asarray(self.driver.jac(driver, t))
        coupling_jac_driver = self.coupling_map.jac(driver, response, wrt="driver")

        response_jac = np.asarray(self.response.jac(response, t))
        coupling_jac_response = self.coupling_map.jac(driver, response, wrt="response")

        return np.block(
            [
                [driver_jac, np.zeros((self.driver_dim, self.response_dim))],
                [
                    coupling_jac_driver @ driver_jac,
                    coupling_jac_response @ response_jac,
                ],
            ]
        )

    def __call__(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.rhs(X, t)

    def _postprocessing(self, *X: np.ndarray) -> np.ndarray:
        driver, response = X[: self.driver_dim], X[self.driver_dim :]
        driver_postprocess_fn = (
            None
            if not hasattr(self.driver, "_postprocessing")
            else self.driver._postprocessing
        )
        response_postprocess_fn = (
            None
            if not hasattr(self.response, "_postprocessing")
            else self.response._postprocessing
        )

        if driver_postprocess_fn is not None:
            driver = driver_postprocess_fn(*driver)
        if hasattr(self.coupling_map, "_postprocessing"):
            response = self.coupling_map._postprocessing(
                np.asarray(response),
                driver_postprocess_fn,
                response_postprocess_fn,
                self.response.unbounded_indices,
                self.driver.unbounded_indices,
            )

        return np.concatenate([np.asarray(driver), np.asarray(response)])
