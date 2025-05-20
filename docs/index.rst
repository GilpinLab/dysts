.. dysts documentation master file, created by
   sphinx-quickstart on Thu Jul 29 13:59:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###################
dysts API Reference
###################

The API reference for the dysts repository: https://github.com/williamgilpin/dysts

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Base Classes
===================

.. automodule:: dysts.base
    :members:

.. Datasets
.. ===================

.. .. automodule:: dysts.datasets
..     :members:
..     :exclude-members: load_file, featurize_timeseries

Utilities
===================

.. automodule:: dysts.utils.utils
    :members:
    :exclude-members: group_consecutives, integrate_weiner, parabolic, parabolic_polyfit, signif, resample_timepoints

.. automodule:: dysts.utils.native_utils
    :members:
    :exclude-members: group_consecutives, integrate_weiner, parabolic, parabolic_polyfit, signif, resample_timepoints

.. automodule:: dysts.utils.integration_utils
    :members:

Analysis
===================

.. automodule:: dysts.analysis
    :members:
    :exclude-members: max_lyapunov_exponent_rosenstein, dfa

Systems
===================

.. automodule:: dysts.systems
    :members:

Sampling
===================

.. automodule:: dysts.sampling
    :members:

Metrics
===================

.. automodule:: dysts.metrics
    :members:
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
