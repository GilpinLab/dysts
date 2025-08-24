"""
Dynamical systems in Python - More than 100 strange attractors

This package provides implementations of various dynamical systems including:
- Continuous-time systems (flows)
- Discrete-time systems (maps)
- Delay differential equations
- Coupling and analysis utilities
"""

from . import (
    analysis,
    attractor,
    base,
    coupling,
    flows,
    generator,
    maps,
    metrics,
    sampling,
    systems,
    utils,
)
from .base import BaseDyn, DynMap, DynSys, DynSysDelay
from .systems import get_attractor_list, get_system_data

__version__ = "0.95"
__all__ = [
    "utils",
    "systems",
    "flows",
    "maps",
    "base",
    "attractor",
    "coupling",
    "generator",
    "metrics",
    "sampling",
    "analysis",
    "DynSys",
    "DynSysDelay",
    "DynMap",
    "BaseDyn",
    "get_attractor_list",
    "get_system_data",
]
