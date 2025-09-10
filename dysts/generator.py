import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import starmap
from multiprocessing import Manager, Pool
from typing import Callable, Literal

import numpy as np
import wandb
from tqdm import tqdm

from . import flows
from .attractor import AttractorValidator
from .base import BaseDyn, SkewProduct
from .sampling import BaseSampler, OnAttractorInitCondSampler
from .systems import make_trajectory_ensemble
from .utils import dict_demote_from_numpy, process_trajs, timeit


def combine_ensembles(
    ensemble_A: dict[str, np.ndarray],
    ensemble_B: dict[str, np.ndarray],
    axis: int = -1,
) -> dict[str, np.ndarray]:
    """
    Combine ensembles A and B.
    Default behavior is to concatenate along the last axis (in our convention, this is the dimension (number of channels) axis)
    """
    assert set(ensemble_A) == set(ensemble_B), "Ensemble keys mismatch"
    assert all(
        ensemble_A[sys].shape[0] == ensemble_B[sys].shape[0] for sys in ensemble_A
    ), "Ensemble sample count mismatch"
    return {
        sys: np.concatenate([ensemble_A[sys], ensemble_B[sys]], axis=axis)
        for sys in ensemble_A
    }


def filter_ensemble_by_successful_samples(
    ensemble: dict[str, np.ndarray] | None,
    successful_samples_dict: dict[str, list[int]],
    transient_time: int,
    successful_ensemble: dict[str, np.ndarray] | None = None,
    failed_ensemble: dict[str, np.ndarray] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Filter an ensemble based on successful samples from attractor validation.

    This function takes an ensemble and filters it to match the keys and sample indices of successful_samples_dict

    Args:
        ensemble: Dictionary mapping system names to trajectory arrays.
                        Shape: (num_samples, num_dims, num_timesteps)
        successful_samples_dict: Dictionary mapping system names to lists of successful sample indices
        transient_time: Total number of time points in the trajectories
        successful_ensemble: Dictionary of successful ensembles (for validation)
        failed_ensemble: Dictionary of failed ensembles (for validation)
        verbose: Whether to print transient time information

    Returns:
        tuple: (filtered_ensemble, failed_filtered_ensemble)
            - filtered_ensemble: Ensemble for successful samples only
            - failed_filtered_ensemble: Ensemble for failed samples only

    Raises:
        AssertionError: If the filtered ensembles don't match the response ensembles in terms of
                       system keys or sample counts.
    """
    if ensemble is None:
        return {}, {}

    if verbose:
        print(f"Transient time: {transient_time}")

    failed_filtered_ensemble = {}
    for sys, traj in ensemble.items():
        failed_inds = np.setdiff1d(
            # np.arange(traj.shape[0]),
            np.arange(1),  # a bit hacky, because process_sample_interval is 1
            successful_samples_dict.get(sys, []),
        )
        if failed_inds.size > 0:
            failed_filtered_ensemble[sys] = traj[failed_inds, ..., transient_time:]

    filtered_ensemble = {
        sys: traj[np.array(successful_samples_dict[sys]), ..., transient_time:]
        for sys, traj in ensemble.items()
        if successful_samples_dict.get(sys) is not None
        and len(successful_samples_dict[sys]) > 0
    }

    # Validation assertions
    if successful_ensemble is not None:
        assert set(filtered_ensemble) == set(successful_ensemble), (
            "Ensemble keys mismatch"
        )
        assert all(
            filtered_ensemble[sys].shape[0] == successful_ensemble[sys].shape[0]
            for sys in successful_ensemble
        ), "Ensemble sample count mismatch"

    if failed_ensemble is not None:
        if set(failed_filtered_ensemble) != set(failed_ensemble):
            breakpoint()
        # assert set(failed_filtered_ensemble) == set(failed_ensemble), (
        #     "Failed ensemble keys mismatch"
        # )

        assert all(
            failed_filtered_ensemble[sys].shape[0] == failed_ensemble[sys].shape[0]
            for sys in failed_ensemble
        ), "Failed ensemble sample count mismatch"

    return filtered_ensemble, failed_filtered_ensemble


logger = logging.getLogger(__name__)


@contextmanager
def managed_cache(
    sampler: OnAttractorInitCondSampler | None, use_multiprocessing: bool
):
    """Context manager to handle shared cache for OnAttractorInitCondSampler."""
    if use_multiprocessing and isinstance(sampler, OnAttractorInitCondSampler):
        with Manager() as manager:
            sampler.trajectory_cache = manager.dict()  # type: ignore
            try:
                yield
            finally:
                sampler.clear_cache()
    else:
        yield


@dataclass
class BaseDynSysSampler(ABC):
    """
    Abstract base class for dynamical system samplers.
    Defines the interface for classes that generate and save trajectory ensembles.

    Subclasses must implement the _generate_ensembles method and the sample_ensembles method.
    """

    @abstractmethod
    def _generate_ensembles(
        self,
        systems: list[str | BaseDyn],
        use_multiprocessing: bool = True,
        postprocessing_callbacks: list[Callable] | None = None,
        silent_errors: bool = False,
        **kwargs,
    ) -> None:
        """
        Generate trajectory ensembles for parameter perturbations of a set of dynamical systems.

        Args:
            systems: List of dynamical systems to generate ensembles for
            use_multiprocessing: Whether to use multiprocessing for ensemble generation
            postprocessing_callbacks: Callbacks to process ensembles after generation
            silent_errors: Whether to silence errors during integration
            **kwargs: Additional keyword arguments passed to the trajectory generation
        """
        pass

    @abstractmethod
    def sample_ensembles(
        self,
        systems: list[str] | list[BaseDyn],
        save_dir: str | None = None,
        split: str = "train",
        **kwargs,
    ) -> None:
        """
        Sample and process trajectory ensembles for a given set of dynamical systems.
        Wrapper around _generate_ensembles.
        Current functionality is to treat the default ensemble separately, generated here, and to handle the parameter perturbations in _generate_ensembles.

        Args:
            systems: List of dynamical systems to sample ensembles for
            split: Dataset split name (e.g., "train", "val", "test")
            **kwargs: Additional keyword arguments for sampling configuration
        """
        pass


@dataclass
class DynSysSampler(BaseDynSysSampler):
    """
    Class to generate and save trajectory ensembles for a given set of dynamical systems.
    Args:
        rseed: random seed for reproducibility
        num_periods: number of periods to generate for each system
        num_points: number of time points to generate for each system
        param_sampler: parameter sampler, samples parameters for each system
        ic_sampler: initial condition sampler, samples initial conditions for each system
        num_ics: number of initial conditions to sample for each system
        num_param_perturbations: number of parameter perturbations to sample for each system
        split_coords: whether to split the coordinates by dimension (univariate) or not (multivariate)
        events: list of solve_ivp events to use for numerical integration
        attractor_validator_kwargs: kwargs for the attractor validator
        attractor_tests: list of tests to use for attractor validator
        save_failed_trajs: flag to save failed trajectory ensembles for debugging
    """

    rseed: int = 999
    num_periods: int | list[int] = 40
    num_points: int = 1024

    param_sampler: BaseSampler | None = None
    ic_sampler: OnAttractorInitCondSampler | None = None
    num_ics: int = 1
    num_param_perturbations: int = 1

    split_coords: bool = True  # by default save trajectories compatible with Chronos
    events: list[Callable[[float, np.ndarray], float]] | None = None

    validator_transient_frac: float = 0.05
    attractor_tests: list[Callable] | None = None

    verbose: bool = True
    save_failed_trajs: bool = False
    wandb_run: wandb.sdk.wandb_run.Run | None = None  # type: ignore

    multiprocess_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.num_periods, int):
            self.num_periods = [self.num_periods]

        self.failed_integrations = defaultdict(list)
        self.rng = np.random.default_rng(self.rseed)
        if self.param_sampler is None:
            assert self.num_param_perturbations == 0, (
                "No parameter sampler provided, but num_param_perturbations > 0"
            )
        if self.ic_sampler is None:
            assert self.num_ics == 1, (
                "No initial condition sampler provided, but num_ics > 1"
            )
        self.attractor_validator = None
        if self.attractor_tests is None and self.num_param_perturbations > 1:
            logger.warning(
                "No attractor tests specified. Parameter perturbations may not result in valid attractors!"
            )
        elif self.attractor_tests is not None:
            self.attractor_validator = AttractorValidator(
                transient_time_frac=self.validator_transient_frac,
                tests=self.attractor_tests,
                multiprocess_kwargs=self.multiprocess_kwargs,
            )

    def _prepare_save_directories(
        self,
        save_dir: str | None,
        split: str,
        split_failures: str = "failed_attractors",
        split_driver: str = "driver",
        save_driver_coords_option: Literal["combined", "separate"] | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        if save_dir is not None:
            save_dyst_dir = os.path.join(save_dir, split)
            os.makedirs(save_dyst_dir, exist_ok=True)
            logger.info(f"valid attractors will be saved to {save_dyst_dir}")

            driver_dyst_dir = None
            if save_driver_coords_option == "separate":
                driver_dyst_dir = os.path.join(save_dir, split_driver, split)
                os.makedirs(driver_dyst_dir, exist_ok=True)
                logger.info(f"driver coordinates will be saved to {driver_dyst_dir}")
            elif save_driver_coords_option == "combined":
                logger.info(f"driver coordinates will be saved to {save_dyst_dir}")
            else:
                logger.warning(
                    "save_driver_coords_option is None, will not save driver coordinates"
                )

            if self.save_failed_trajs:
                failed_dyst_dir = os.path.join(save_dir, split_failures, split)
                os.makedirs(failed_dyst_dir, exist_ok=True)
                logger.info(f"failed attractors will be saved to {failed_dyst_dir}")
            else:
                failed_dyst_dir = None
        else:
            logger.warning("save_dir is None, will not save trajectories.")
            save_dyst_dir = driver_dyst_dir = failed_dyst_dir = None
        return save_dyst_dir, driver_dyst_dir, failed_dyst_dir

    def _transform_params_and_ics(
        self,
        system: BaseDyn | str,
        ic_transform: Callable | None = None,
        param_transform: Callable | None = None,
        ic_rng: np.random.Generator | None = None,
        param_rng: np.random.Generator | None = None,
    ) -> BaseDyn | None:
        """
        Transform the parameters and initial conditions of a system.

        NOTE: If
         - an IC transform or parameter transform is not successful
         - the system is parameterless (len(sys.param_list) == 0)
        the system is not returned (ignored downstream)
        """
        sys = getattr(flows, system)() if isinstance(system, str) else system

        if hasattr(sys, "param_list") and len(sys.param_list) == 0:
            return None

        success = True
        if param_transform is not None:
            if param_rng is not None:  # unsafe, address later
                param_transform.set_rng(param_rng)
            param_success = sys.transform_params(param_transform)
            success &= param_success
        if ic_transform is not None:
            if ic_rng is not None:  # unsafe, address later
                ic_transform.set_rng(ic_rng)
            ic_success = sys.transform_ic(ic_transform)
            success &= ic_success

        return sys if success else None

    def _init_perturbations(
        self,
        systems: list[str] | list[BaseDyn],
        ic_rng: np.random.Generator | None = None,
        param_rng: np.random.Generator | None = None,
        perturb_params: bool = False,
        perturb_ics: bool = False,
        use_multiprocessing: bool = True,
    ) -> list[BaseDyn | None]:
        """
        Pre-initialize the perturbed dyst objects for generation
        """
        assert all(sys is not None for sys in systems), "systems cannot contain None"

        ic_rng_stream = [None] * len(systems)
        if ic_rng is not None:
            ic_rng_stream = ic_rng.spawn(len(systems))

        param_rng_stream = [None] * len(systems)
        if param_rng is not None:
            param_rng_stream = param_rng.spawn(len(systems))

        param_transform = self.param_sampler if perturb_params else None
        ic_transform = self.ic_sampler if perturb_ics else None

        args = (
            (system, ic_transform, param_transform, ic_rng, param_rng)
            for system, ic_rng, param_rng in zip(
                systems, ic_rng_stream, param_rng_stream
            )
        )

        with (
            Pool(**self.multiprocess_kwargs)
            if use_multiprocessing
            else nullcontext() as pool
        ):
            map_fn = pool.starmap if use_multiprocessing else starmap  # type: ignore
            transformed_systems = list(map_fn(self._transform_params_and_ics, args))

        return transformed_systems

    @timeit(logger=logger)
    def sample_ensembles(
        self,
        systems: list[str] | list[BaseDyn],
        save_dir: str | None = None,
        split: str = "train",
        split_failures: str = "failed_attractors",
        split_driver: str = "driver",
        save_driver_coords_option: Literal["combined", "separate"] | None = None,
        samples_process_interval: int = 1,
        save_params_dir: str | None = None,
        save_traj_stats_dir: str | None = None,
        save_integration_timepoints_dir: str | None = None,
        standardize: bool = False,
        use_multiprocessing: bool = True,
        silent_errors: bool = False,
        reset_attractor_validator: bool = False,
        return_times: bool = True,
        **kwargs,
    ) -> None:
        """
        Sample perturbed ensembles for a given set of dynamical systems. Optionally,
        save the ensembles to disk and save the parameters to a json file.
        """
        if save_driver_coords_option is not None:
            assert save_driver_coords_option in ["combined", "separate"], (
                f"Invalid save_driver_coords_option: {save_driver_coords_option}"
            )
        sys_names = [sys if isinstance(sys, str) else sys.name for sys in systems]
        assert len(set(sys_names)) == len(sys_names), (
            "Cannot have duplicate system names"
        )
        if save_dir is not None:
            logger.info(
                f"Making {split} split with {len(systems)} dynamical systems"
                f" (showing first {min(10, len(sys_names))}): \n {sys_names[:10]}"
            )
        is_all_basedyn = all(isinstance(sys, BaseDyn) for sys in systems)

        if self.attractor_validator is not None and reset_attractor_validator:
            self.attractor_validator.reset()
            self.failed_integrations.clear()

        save_dyst_dir, driver_dyst_dir, failed_dyst_dir = (
            self._prepare_save_directories(
                save_dir,
                split,
                split_failures=split_failures,
                split_driver=split_driver,
                save_driver_coords_option=save_driver_coords_option,
            )
        )

        # NOTE: we define number of total samples as (num_param_perturbations * num_ics) + 1 to account for the default ensemble (with one initial condition)
        num_total_samples = self.num_param_perturbations * self.num_ics + 1

        callbacks = [
            self._reset_events_callback,
            self._validate_and_save_ensemble_callback(
                num_total_samples,
                samples_process_interval,
                save_dyst_dir,
                driver_dyst_dir=driver_dyst_dir,
                save_driver_coords_option=save_driver_coords_option,
                failed_dyst_dir=failed_dyst_dir,
                save_params_dir=save_params_dir,
                save_traj_stats_dir=save_traj_stats_dir,
                save_integration_timepoints_dir=save_integration_timepoints_dir,
            ),
            self.save_failed_integrations_callback,
        ]

        num_periods = self.rng.choice(self.num_periods)
        logger.info(f"Generating default ensemble with {num_periods} periods")

        # treat the default params as the zeroth sample, enforce just one (default) initial condition
        default_ensemble = make_trajectory_ensemble(
            self.num_points,
            subset=systems,
            pts_per_period=self.num_points // num_periods,
            event_fns=self.events,
            use_multiprocessing=use_multiprocessing,
            silent_errors=silent_errors,
            multiprocess_kwargs=self.multiprocess_kwargs,
            return_times=return_times,
            **kwargs,
        )

        if return_times:
            failed_integrations = [
                k
                for k, v in default_ensemble.items()
                if v[1] is None or np.isnan(v[1]).any()  # type: ignore
            ]
            default_ts_ensemble = {
                k: v[0]  # type: ignore
                for k, v in default_ensemble.items()
                if k not in failed_integrations
            }
            default_ensemble = {
                k: v[1]  # type: ignore
                for k, v in default_ensemble.items()
                if k not in failed_integrations
            }
        else:
            failed_integrations = [
                k
                for k, v in default_ensemble.items()
                if v is None or np.isnan(v).any()  # type: ignore
            ]
            default_ts_ensemble = None
            default_ensemble = {
                k: v
                for k, v in default_ensemble.items()
                if k not in failed_integrations
            }

        # Apply all the callbacks to the default ensemble (sample_idx=0)
        for callback in callbacks:
            callback(
                sample_idx=0,
                ensemble=default_ensemble,
                ts_ensemble=default_ts_ensemble,
                excluded_keys=failed_integrations,
                perturbed_systems=systems if is_all_basedyn else None,
                num_periods=num_periods,  # NOTE: add other metadata eventually, as kwargs
            )

        logger.info("Generating perturbed ensembles...")

        self._generate_ensembles(
            systems,
            postprocessing_callbacks=callbacks,
            standardize=standardize,
            use_multiprocessing=use_multiprocessing,
            silent_errors=silent_errors,
            return_times=return_times,
            **kwargs,
        )

    def _generate_ensembles(
        self,
        systems: list[str] | list[BaseDyn],
        use_multiprocessing: bool = True,
        postprocessing_callbacks: list[Callable] | None = None,
        silent_errors: bool = False,
        return_times: bool = True,
        **kwargs,
    ) -> None:
        """
        Generate trajectory ensembles for a given set of dynamical systems.
        """
        total_iterations = self.num_param_perturbations * self.num_ics
        pbar = tqdm(total=total_iterations, desc="Generating ensembles")

        with managed_cache(self.ic_sampler, use_multiprocessing):
            pp_rng_stream = self.rng.spawn(self.num_param_perturbations)
            for i, param_rng in enumerate(pp_rng_stream):
                if self.wandb_run is not None:
                    self.wandb_run.log({"param_idx": i})
                param_perturbed_systems = self._init_perturbations(
                    systems,
                    param_rng=param_rng,
                    perturb_params=True,
                    use_multiprocessing=use_multiprocessing,
                )

                # filter out parameterless systems or
                # systems that failed to transform parameters for any reason
                excluded_pperts = [
                    sys if isinstance(sys, str) else sys.name  # type: ignore
                    for sys, pp_sys in zip(systems, param_perturbed_systems)
                    if pp_sys is None
                ]
                param_perturbed_systems = [
                    sys for sys in param_perturbed_systems if sys is not None
                ]

                if self.ic_sampler is not None and isinstance(
                    self.ic_sampler, OnAttractorInitCondSampler
                ):
                    self.ic_sampler.clear_cache()

                ic_rng_stream = param_rng.spawn(self.num_ics)
                for j, ic_rng in enumerate(ic_rng_stream):
                    sample_idx = (
                        i * len(ic_rng_stream) + j + 1
                    )  # + 1 to account for the previously created default_ensemble in sample_ensemble()
                    if self.wandb_run is not None:
                        self.wandb_run.log({"sample_idx": sample_idx})

                    # after the parameter perturbation, perturb the initial conditions
                    ic_perturbed_systems = self._init_perturbations(
                        param_perturbed_systems,
                        ic_rng=ic_rng,
                        perturb_ics=True,
                        use_multiprocessing=use_multiprocessing,
                    )
                    excluded_systems = [
                        sys if isinstance(sys, str) else sys.name  # type: ignore
                        for sys, ic_sys in zip(systems, ic_perturbed_systems)
                        if ic_sys is None
                    ] + excluded_pperts  # systems that failed ic and param transforms
                    perturbed_systems = [
                        sys for sys in ic_perturbed_systems if sys is not None
                    ]
                    assert len(perturbed_systems) + len(excluded_systems) == len(
                        systems
                    )

                    num_periods = self.rng.choice(self.num_periods)
                    logger.info(
                        f"Generating ensemble of param perturbation {i + 1} and ic perturbation {j} with {num_periods} periods"
                    )

                    ensemble = make_trajectory_ensemble(
                        self.num_points,
                        subset=perturbed_systems,
                        pts_per_period=self.num_points // num_periods,
                        event_fns=self.events,
                        use_multiprocessing=use_multiprocessing,
                        silent_errors=silent_errors,
                        multiprocess_kwargs=self.multiprocess_kwargs,
                        return_times=return_times,
                        **kwargs,
                    )

                    # Exclude failed integrations
                    if return_times:
                        excluded_systems.extend(
                            k
                            for k, v in ensemble.items()
                            if v[1] is None or np.isnan(v[1]).any()  # type: ignore
                        )
                        ts_ensemble = {
                            k: v[0]  # type: ignore
                            for k, v in ensemble.items()
                            if k not in excluded_systems
                        }
                        ensemble = {
                            k: v[1]  # type: ignore
                            for k, v in ensemble.items()
                            if k not in excluded_systems
                        }
                    else:
                        excluded_systems.extend(
                            k
                            for k, v in ensemble.items()
                            if v is None or np.isnan(v).any()
                        )
                        ts_ensemble = None
                        ensemble = {
                            k: v
                            for k, v in ensemble.items()
                            if k not in excluded_systems
                        }

                    for callback in postprocessing_callbacks or []:
                        callback(
                            sample_idx=sample_idx,
                            ensemble=ensemble,
                            ts_ensemble=ts_ensemble,
                            excluded_keys=excluded_systems,
                            perturbed_systems=perturbed_systems,
                            num_periods=num_periods,
                        )

                    pbar.update(1)
                    pbar.set_postfix({"param_idx": i, "ic_idx": j})

    def _reset_events_callback(self, *args, **kwargs) -> None:
        for event in self.events or []:
            if hasattr(event, "reset") and callable(event.reset):
                event.reset()

    def save_failed_integrations_callback(self, sample_idx, *args, **kwargs):
        excluded_keys = kwargs.get("excluded_keys", [])
        if len(excluded_keys) > 0:
            logger.warning(f"Integration failed for {len(excluded_keys)} systems")
            for dyst_name in excluded_keys:
                self.failed_integrations[dyst_name].append(sample_idx)

    def _validate_and_save_ensemble_callback(
        self,
        num_total_samples: int,
        samples_process_interval: int,
        save_dyst_dir: str | None = None,
        driver_dyst_dir: str | None = None,
        save_driver_coords_option: Literal["combined", "separate"] | None = None,
        failed_dyst_dir: str | None = None,
        save_params_dir: str | None = None,
        save_traj_stats_dir: str | None = None,
        save_integration_timepoints_dir: str | None = None,
    ):
        """
        Callback to process and save ensembles and parameters. Wraps around _process_and_save_ensemble by making it a callback.
        """
        ensemble_list = []
        ts_ensemble_list = []

        def _callback(
            sample_idx: int,
            ensemble: dict[str, np.ndarray],
            ts_ensemble: dict[str, np.ndarray] | None = None,
            **kwargs,
        ):
            if len(ensemble.keys()) == 0:
                if save_dyst_dir is not None:
                    logger.warning("No successful trajectories for this sample")
                return

            ensemble_list.append(ensemble)
            ts_ensemble_list.append(ts_ensemble)

            is_last_sample = (sample_idx + 1) == num_total_samples
            if ((sample_idx + 1) % samples_process_interval) == 0 or is_last_sample:
                self._process_and_save_ensemble(
                    ensemble_list,
                    ts_ensemble_list
                    if all(ts is not None for ts in ts_ensemble_list)
                    else None,  # hacky
                    sample_idx,
                    perturbed_systems=kwargs.get("perturbed_systems"),
                    save_dyst_dir=save_dyst_dir,
                    driver_dyst_dir=driver_dyst_dir,
                    save_driver_coords_option=save_driver_coords_option,
                    failed_dyst_dir=failed_dyst_dir,
                    save_params_dir=save_params_dir,
                    save_traj_stats_dir=save_traj_stats_dir,
                    save_integration_timepoints_dir=save_integration_timepoints_dir,
                    num_periods=kwargs.get("num_periods"),
                )
                ensemble_list.clear()

        return _callback

    def _process_and_save_ensemble(
        self,
        ensemble_list: list[dict[str, np.ndarray]],
        ts_ensemble_list: list[dict[str, np.ndarray]] | None,
        sample_idx: int,
        perturbed_systems: list[BaseDyn] | None = None,
        save_dyst_dir: str | None = None,
        driver_dyst_dir: str | None = None,
        save_driver_coords_option: Literal["combined", "separate"] | None = None,
        failed_dyst_dir: str | None = None,
        save_params_dir: str | None = None,
        save_traj_stats_dir: str | None = None,
        save_integration_timepoints_dir: str | None = None,
        num_periods: int | None = None,
    ) -> None:
        """
        Processes a list of trajectory ensembles, validates attractors, and saves results to disk.

        This method performs the following steps:
        1. Stacks and transposes the input list of ensembles to produce arrays of shape
           (num_samples, num_dims, num_timesteps) for each system.
        2. If `perturbed_systems` is provided, determines the driver dimension for each system.
           - If `save_driver_coords_option` is set, extracts and stores the driver coordinates separately.
           - For skew product systems, only the response coordinates are retained in the ensemble.
        3. If an attractor validator is present, applies it to filter out invalid trajectories:
           - The validator is run in parallel and returns valid and failed ensembles, as well as a mapping
             of successful sample indices.
           - If driver coordinates are being saved, they are filtered to match the valid samples.
           - Asserts that the filtered driver and response ensembles are consistent.
           - Records the number of valid systems.
        4. Saves the processed ensembles to disk:
           - The valid response ensemble is saved to `save_dyst_dir`.
           - If driver coordinates are being saved, they are saved either separately (driver coords in f"{save_dyst_dir}_driver") or combined with the response (to save_dyst_dir),
             depending on `save_driver_coords_option`.
           - The failed combined (driver + response) ensemble is saved to `failed_dyst_dir` if provided.
        5. If parameter saving is enabled, saves the parameters of successful and failed systems to separate JSON files.
        6. If trajectory statistics saving is enabled, saves statistics for the valid (response) ensemble.

        Args:
            ensemble_list: List of dictionaries mapping system names to trajectory arrays for each sample.
            sample_idx: Index of the current sample batch.
            perturbed_systems: List of perturbed system objects corresponding to the ensemble (optional).
            save_dyst_dir: Directory to save valid trajectory ensembles (optional).
            driver_dyst_dir: Directory to save the driver part of valid skew product system trajectory (response) ensembles (optional).
            failed_dyst_dir: Directory to save failed trajectory ensembles (optional).
            save_params_dir: Directory to save system parameters (optional).
            save_traj_stats_dir: Directory to save trajectory statistics (optional).
            save_integration_timepoints_dir: Directory to save integration timepoints (optional).
            save_driver_coords_option: If set, controls saving of driver coordinates:
                - "separate": Save driver coordinates in a separate subdirectory, to f"{save_dyst_dir}_driver"
                - "combined": Concatenate driver and response coordinates and save together, to save_dyst_dir
        """
        # stack and transpose to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
        ensemble_sys_names = [sys for ens in ensemble_list for sys in ens.keys()]
        ensemble = {
            sys: np.stack(
                [ens[sys] for ens in ensemble_list if sys in ens], axis=0
            ).transpose(0, 2, 1)
            for sys in ensemble_sys_names
        }

        driver_ensemble = None

        current_param_pert_summary = {}
        if perturbed_systems is not None:
            driver_dims = {
                sys.name: getattr(sys, "driver_dim", 0) for sys in perturbed_systems
            }  # defaults to 0 when we deal with base (non-skew) systems (which don't have the driver_dim attribute)

            if save_driver_coords_option is not None:
                driver_ensemble = {
                    sys: traj[:, : driver_dims[sys], :]
                    for sys, traj in ensemble.items()
                }

            # if skew system, only saves the response coordinates in the ensemble
            ensemble = {
                sys: traj[:, driver_dims[sys] :, :] for sys, traj in ensemble.items()
            }

        current_param_pert_summary["num_systems_integrated"] = len(ensemble)

        if self.attractor_validator is not None:
            logger.info(f"Applying attractor validator to {len(ensemble)} systems")
            # NOTE: we enforce that in the case of skew systems, only the response coordinates are passed to the attractor validator
            ensemble, failed_ensemble, successful_samples_dict = (
                self.attractor_validator.multiprocessed_filter_ensemble(
                    ensemble, first_sample_idx=sample_idx
                )
            )
            # Filter the driver ensemble to match the successful samples from the attractor validator
            driver_ensemble, failed_driver_ensemble = (
                filter_ensemble_by_successful_samples(
                    ensemble=driver_ensemble,
                    successful_samples_dict=successful_samples_dict,
                    successful_ensemble=ensemble,
                    failed_ensemble=failed_ensemble,
                    transient_time=int(self.num_points * self.validator_transient_frac),
                    verbose=self.verbose,
                )
            )

            logger.info(f"{len(ensemble)} systems passed attractor validator")
            current_param_pert_summary["num_systems_valid"] = len(ensemble)
        else:
            failed_ensemble = {}
            failed_driver_ensemble = {}

        if self.wandb_run is not None:
            self.wandb_run.log(current_param_pert_summary)
            counts_per_failed_check = self._get_counts_per_failed_check()
            logger.info(f"Logging counts per failed check: {counts_per_failed_check}")
            self.wandb_run.log(counts_per_failed_check)

        save_kwargs = {
            "split_coords": self.split_coords,
            "verbose": self.verbose,
            "sample_idx": sample_idx,
            "num_periods": num_periods,
        }
        if save_dyst_dir:
            if driver_ensemble:
                if save_driver_coords_option == "separate" and driver_dyst_dir:
                    process_trajs(driver_dyst_dir, driver_ensemble, **save_kwargs)
                elif save_driver_coords_option == "combined":
                    combined = combine_ensembles(driver_ensemble, ensemble, axis=1)
                    process_trajs(save_dyst_dir, combined, **save_kwargs)
                else:
                    raise ValueError(
                        f"Invalid save_driver_coords_option: {save_driver_coords_option}, or driver_ensemble should be None"
                    )
            else:  # default case, no driver ensemble
                process_trajs(save_dyst_dir, ensemble, **save_kwargs)

        if failed_dyst_dir and failed_ensemble:
            process_trajs(failed_dyst_dir, failed_ensemble, **save_kwargs)
            # save the failed (driver + response) ensemble if skew system
            if failed_driver_ensemble:
                combined = combine_ensembles(
                    failed_driver_ensemble, failed_ensemble, axis=1
                )
                process_trajs(failed_dyst_dir, combined, **save_kwargs)

        if save_params_dir is not None and perturbed_systems is not None:
            systems_by_status = {}
            for status, ens in [("successes", ensemble), ("failures", failed_ensemble)]:
                systems = [sys for sys in perturbed_systems if sys.name in ens]
                systems_by_status[status] = systems
                self._save_parameters(
                    sample_idx,
                    systems,
                    os.path.join(save_params_dir, f"{status}.json"),
                )

            # only save system stats for successful samples, and if we also save parameters
            if save_traj_stats_dir is not None:
                ts_ensemble = None
                if ts_ensemble_list is not None:
                    ts_ensemble = {
                        sys: np.stack(
                            [ens[sys] for ens in ts_ensemble_list if sys in ens], axis=0
                        )
                        for sys in ensemble_sys_names
                    }
                    if self.attractor_validator is not None:
                        ts_ensemble, _ = filter_ensemble_by_successful_samples(
                            ensemble=ts_ensemble,
                            successful_samples_dict=successful_samples_dict,
                            successful_ensemble=ensemble,
                            failed_ensemble=failed_ensemble,
                            transient_time=int(
                                self.num_points * self.validator_transient_frac
                            ),
                            verbose=self.verbose,
                        )
                    # save the integration timepoints (ts_ensemble)
                    if save_integration_timepoints_dir is not None:
                        logger.info(
                            f"Saving solve_ivp integration timepoints to {save_integration_timepoints_dir}"
                        )
                        process_trajs(
                            save_integration_timepoints_dir, ts_ensemble, **save_kwargs
                        )
                if driver_ensemble is not None:
                    combined = combine_ensembles(driver_ensemble, ensemble, axis=1)
                else:
                    combined = ensemble

                self._save_traj_stats(
                    sample_idx,
                    systems_by_status["successes"],
                    combined,
                    ts_ensemble=ts_ensemble,
                    save_path=os.path.join(save_traj_stats_dir, "successes.json"),
                )

    def _save_parameters(
        self,
        sample_idx: int,
        perturbed_systems: list[BaseDyn],
        save_path: str | None = None,
    ) -> None:
        if save_path is None or len(perturbed_systems) == 0:
            return
        logger.info(f"Saving parameters to {save_path}")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                param_dict = json.load(f)
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            param_dict = {}

        for sys in perturbed_systems:
            if sys.name not in param_dict:
                param_dict[sys.name] = []

            if isinstance(sys, SkewProduct):
                serialized_params = {
                    "sample_idx": sample_idx,
                    "ic": sys.ic.tolist(),
                    "driver_params": dict_demote_from_numpy(sys.driver.params),
                    "response_params": dict_demote_from_numpy(sys.response.params),
                    "driver_dim": sys.driver_dim,
                    "response_dim": sys.response_dim,
                    "coupling_map": sys.coupling_map._serialize(),
                }
            else:
                serialized_params = {
                    "sample_idx": sample_idx,
                    "ic": sys.ic.tolist(),
                    "params": dict_demote_from_numpy(sys.params),
                    "dim": sys.dimension,
                }

            param_dict[sys.name].append(serialized_params)

        with open(save_path, "w") as f:
            json.dump(param_dict, f, indent=4)

    def _save_traj_stats(
        self,
        sample_idx: int,
        systems: list[BaseDyn],
        ensemble: dict[str, np.ndarray],
        ts_ensemble: dict[str, np.ndarray] | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Save trajectory statistics to a json file.
        We do this for downstream analysis and re-initialization without depending on loading trajectories from Arrow files

        ensemble is a dict mapping system names to trajectories of shape (num_samples, num_dimensions, num_timepoints)
        ts_ensemble is a dict mapping system names to timepoints of shape (num_samples, num_timepoints)

        """
        if save_path is None or len(systems) == 0:
            return

        system_names = [sys.name for sys in systems]
        assert set(system_names) == set(ensemble.keys()), "Systems mismatch"

        if ts_ensemble is not None:
            assert all(
                (ensemble[sys.name].shape[0], ensemble[sys.name].shape[-1])
                == (ts_ensemble[sys.name].shape[0], ts_ensemble[sys.name].shape[-1])
                for sys in systems
            ), "Ensemble sample count mismatch"

        logger.info(f"Saving trajecotory statistics to {save_path}")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                traj_stats = json.load(f)
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            traj_stats = {}

        for sys in systems:
            # At this point, trajectories are of shape (num_samples, num_dimensions, num_timepoints) since we transposed in _process_and_save_ensemble
            # ... where timepoints is after cutting off the first transient_time points
            trajectories = ensemble[sys.name]
            if sys.name not in traj_stats:
                traj_stats[sys.name] = []

            init_conds, final_conds, means, stds, mean_amps, flow_rms = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for i, traj in enumerate(trajectories):
                # traj is of shape (num_dimensions, num_timepoints)
                init_conds.append(traj[:, 0].tolist())
                final_conds.append(traj[:, -1].tolist())
                means.append(traj.mean(axis=1).tolist())
                stds.append(traj.std(axis=1).tolist())
                mean_amps.append(np.mean(np.abs(traj), axis=1).tolist())

                if ts_ensemble is not None:
                    integration_timepoints = ts_ensemble[sys.name][i]
                    flow_rms.append(
                        np.sqrt(
                            np.mean(
                                [
                                    np.asarray(sys(x, t)) ** 2  # type: ignore
                                    for x, t in zip(traj.T, integration_timepoints)
                                ],
                                axis=0,
                            )
                        ).tolist()
                    )

            unwrap = lambda v: v[0] if isinstance(v, list) and len(v) == 1 else v
            traj_stats_entry = {
                "sample_idx": sample_idx,
                "ic": unwrap(init_conds),
                "final": unwrap(final_conds),
                "mean": unwrap(means),
                "std": unwrap(stds),
                "mean_amp": unwrap(mean_amps),
                "flow_rms": unwrap(flow_rms),
            }

            traj_stats[sys.name].append(traj_stats_entry)

        with open(save_path, "w") as f:
            json.dump(traj_stats, f, indent=4)

    def save_summary(self, save_json_path: str | None = None) -> dict:
        """
        Save a summary of valid attractor counts and failed checks to a json file.
        """
        if save_json_path is not None:
            os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
            logger.info(f"Saving summary to {save_json_path}")

        if self.attractor_validator is None:
            summary_dict = {"failed_integrations": self.failed_integrations}

        else:
            valid_dyst_counts = self.attractor_validator.valid_dyst_counts
            failed_checks = self.attractor_validator.failed_checks
            failed_samples = self.attractor_validator.failed_samples
            valid_samples = self.attractor_validator.valid_samples
            summary_dict = {
                "num_parameter_successes": sum(
                    len(np.unique(np.array(sample_inds).astype(int) // self.num_ics))
                    for sample_inds in valid_samples.values()
                ),
                "num_total_candidates": (
                    self.num_param_perturbations + 1
                )  # +1 for the default ensemble, which is not perturbed
                * len(
                    valid_samples.keys()
                    | failed_samples.keys()
                    | self.failed_integrations.keys()
                ),
                "valid_dyst_counts": valid_dyst_counts,
                "failed_checks": failed_checks,
                "failed_integrations": self.failed_integrations,
                "failed_samples": failed_samples,
                "valid_samples": valid_samples,
            }

        if save_json_path is not None:
            with open(save_json_path, "w") as f:
                json.dump(summary_dict, f, indent=4)

        return summary_dict

    def _get_counts_per_failed_check(self) -> dict[str, int]:
        """
        Get the number of systems that failed each check.
        """
        if self.attractor_validator is None:
            return {}

        # TODO: this is a bit hacky, need to have a streamlined solution
        counts_per_failed_check = defaultdict(int)
        for failed_checks_lst in self.attractor_validator.failed_checks.values():
            for entry_all_ics in failed_checks_lst:
                for entry in entry_all_ics:
                    _, check_name = entry
                    counts_per_failed_check[f"failed_{check_name}"] += 1
        return counts_per_failed_check


@dataclass
class DynSysSamplerRestartIC(DynSysSampler):
    """
    Generate trajectories of resampled initial conditions
    User calls sample_ensembles with systems: List[str], which is a list of DynSys objects initialized from saved parameters
            In particular, the DynSys object stores the initial condition and all the parameters needed to reconstruct the RHS of the flow
    The main functionality is to take the re-initialized systems and resample the initial conditions to make + save trajectories with these different initial conditions
    """

    rseed: int = 999
    num_periods: int | list[int] = 40
    num_points: int = 4096

    num_ics: int = 1
    split_coords: bool = True  # by default save trajectories compatible with Chronos

    events: list[Callable[[float, np.ndarray], float]] | None = None
    validator_transient_frac: float = 0.05
    attractor_tests: list[Callable] | None = None

    wandb_run: wandb.sdk.wandb_run.Run | None = None  # type: ignore

    multiprocess_kwargs: dict = field(default_factory=dict)

    @timeit(logger=logger)
    def sample_ensembles(
        self,
        systems: list[BaseDyn],
        save_dir: str | None = None,
        split: str = "train",
        samples_process_interval: int = 1,
        starting_sample_idx: int = 0,
        save_first_sample: bool = True,
        standardize: bool = False,
        use_multiprocessing: bool = True,
        silent_errors: bool = False,
        **kwargs,
    ) -> None:
        """
        Wrapper around _generate_ensembles, for sampling ensembles with different initial conditions
        """
        sys_names = [sys.name for sys in systems]
        assert len(set(sys_names)) == len(sys_names), (
            "Cannot have duplicate system names"
        )
        logger.info(
            f"Making {split} split with {len(systems)} dynamical systems"
            f" (showing first {min(10, len(sys_names))}): \n {sys_names[:10]}"
        )

        if save_dir is not None:
            save_dyst_dir = os.path.join(save_dir, split)
            os.makedirs(save_dyst_dir, exist_ok=True)
            logger.info(f"valid attractors will be saved to {save_dyst_dir}")

        if self.attractor_validator is not None:
            self.attractor_validator.reset()
            self.failed_integrations.clear()

        callbacks = [
            self._reset_events_callback,
            self._validate_and_save_ensemble_callback(
                self.num_ics,
                samples_process_interval,
                save_dyst_dir,
            ),
            self.save_failed_integrations_callback,
        ]

        # NOTE: here, we skip making the default ensemble separately; to streamline the ensemble generation, we do it in _generate_ensembles
        self._generate_ensembles(
            systems,
            save_dir=save_dir,
            split=split,
            starting_sample_idx=starting_sample_idx,
            save_first_sample=save_first_sample,
            postprocessing_callbacks=callbacks,
            standardize=standardize,
            use_multiprocessing=use_multiprocessing,
            silent_errors=silent_errors,
            **kwargs,
        )

    def _generate_ensembles(
        self,
        systems: list[BaseDyn],
        starting_sample_idx: int = 0,
        save_first_sample: bool = True,
        use_multiprocessing: bool = True,
        postprocessing_callbacks: list[Callable] | None = None,
        silent_errors: bool = False,
        **kwargs,
    ) -> None:
        """
        Generate trajectory ensembles for a given set of dynamical systems, with different initial conditions
        """
        n_systems = len(systems)
        pbar = tqdm(
            total=self.num_ics, desc=f"Generating ensembles for {n_systems} systems"
        )
        ic_cache = {}
        for ic_idx in range(self.num_ics):
            if self.wandb_run is not None:
                self.wandb_run.log({"sample_idx": ic_idx})

            num_periods = self.rng.choice(self.num_periods)
            logger.info(
                f"Generating ensemble of ic perturbation {ic_idx} with {num_periods} periods"
            )

            ensemble = make_trajectory_ensemble(
                self.num_points,
                subset=systems,
                pts_per_period=self.num_points // num_periods,
                event_fns=self.events,
                use_multiprocessing=use_multiprocessing,
                silent_errors=silent_errors,
                multiprocess_kwargs=self.multiprocess_kwargs,
                return_times=False,
                **kwargs,
            )

            # filter out failed integrations
            excluded_systems = [
                key
                for key, value in ensemble.items()
                if value is None or np.isnan(value).any()
            ]
            ensemble = {
                key: value
                for key, value in ensemble.items()
                if key not in excluded_systems
            }

            if ic_idx == 0 and not save_first_sample:
                logger.info(
                    f"Skipping validation and saving for first sample, ic_idx={ic_idx}"
                )
                self._reset_events_callback()

            else:
                for callback in postprocessing_callbacks or []:
                    callback(
                        sample_idx=ic_idx + starting_sample_idx,
                        ensemble=ensemble,
                        excluded_keys=excluded_systems,
                        perturbed_systems=systems,
                        num_periods=num_periods,
                    )

            # drop systems that failed integration and prepare IC cache
            if ic_idx == 0:
                systems = [sys for sys in systems if sys.name not in excluded_systems]
                logger.info(f"Dropped {len(excluded_systems)} systems")

                # Cache ICs for future iterations
                for sys in systems:
                    curr_traj = ensemble[
                        sys.name
                    ][  # type: ignore
                        int(self.validator_transient_frac * self.num_points) :
                    ]
                    ic_cache[sys.name] = self.rng.choice(
                        curr_traj,  # type: ignore
                        size=(self.num_ics - 1),
                        replace=False,
                    )

            # Set next IC for each system
            if ic_idx < self.num_ics - 1:
                for sys in systems:
                    sys.ic = ic_cache[sys.name][ic_idx]

            pbar.update(1)
            pbar.set_postfix({"ic_idx": ic_idx})
