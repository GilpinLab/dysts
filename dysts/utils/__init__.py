"""
Helper utilities for working with time series arrays. This module is intended to have no
dependencies on the rest of the package.
"""

from .data_utils import (
    dict_demote_from_numpy,
    process_trajs,
    safe_standardize,
    timeit,
)
from .integration_utils import (
    cast_to_numpy,
    dde,
    ddeint,
    ddeVar,
    generate_ic_ensemble,
    integrate_dyn,
    integrate_weiner,
    resample_timepoints,
)
from .native_utils import (
    ComputationHolder,
    convert_json_to_gzip,
    group_consecutives,
    has_module,
    num_unspecified_params,
)
from .utils import (
    cartesian_to_polar,
    find_characteristic_timescale,
    find_psd,
    find_significant_frequencies,
    find_slope,
    jac_fd,
    logarithmic_n,
    make_epsilon_ball,
    make_surrogate,
    min_data_points_rosenstein,
    nan_fill,
    nanmean_trimmed,
    pad_axis,
    pad_to_shape,
    polar_to_cartesian,
    rowwise_euclidean,
    signif,
    standardize_ts,
)
