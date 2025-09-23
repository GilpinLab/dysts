_UTILITY_FILE_MAPPING: dict[str, list[str]] = {
    ".data_utils": [
        "dict_demote_from_numpy",
        "process_trajs",
        "safe_standardize",
        "timeit",
    ],
    ".integration_utils": [
        "cast_to_numpy",
        "dde",
        "ddeint",
        "ddeVar",
        "generate_ic_ensemble",
        "integrate_dyn",
        "integrate_weiner",
        "resample_timepoints",
    ],
    ".native_utils": [
        "ComputationHolder",
        "convert_json_to_gzip",
        "group_consecutives",
        "has_module",
        "num_unspecified_params",
    ],
    ".utils": [
        "cartesian_to_polar",
        "find_characteristic_timescale",
        "find_psd",
        "find_significant_frequencies",
        "find_slope",
        "jac_fd",
        "logarithmic_n",
        "make_epsilon_ball",
        "make_surrogate",
        "min_data_points_rosenstein",
        "nan_fill",
        "nanmean_trimmed",
        "pad_axis",
        "pad_to_shape",
        "polar_to_cartesian",
        "rowwise_euclidean",
        "signif",
        "standardize_ts",
    ],
}

_UTILITY_MODULES: dict[str, str] = {
    utility: module_path
    for module_path, utilities in _UTILITY_FILE_MAPPING.items()
    for utility in utilities
}


def __getattr__(name: str):
    """Lazy load utilities"""
    if name in _UTILITY_MODULES:
        module_path = _UTILITY_MODULES[name]
        # Use absolute import path to avoid issues with __name__ not being available
        full_module_path = f"dysts.utils{module_path}"
        module = __import__(full_module_path, fromlist=[name])
        return getattr(module, name)

    raise AttributeError(f"module 'dysts.utils' has no attribute '{name}'")


def __dir__():
    return sorted(_UTILITY_MODULES.keys())
