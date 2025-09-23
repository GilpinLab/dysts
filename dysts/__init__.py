"""
Dynamical systems in Python - More than 100 strange attractors

This package provides implementations of various dynamical systems including:
- Continuous-time systems (flows)
- Discrete-time systems (maps)
- Delay differential equations
- Coupling and analysis utilities
"""

_MODULE_NAMES = {
    "analysis",
    "base",
    "coupling",
    "flows",
    "maps",
    "metrics",
    "sampling",
    "systems",
    "utils",
}

__all__ = list(_MODULE_NAMES)


def __getattr__(name: str):
    """Lazy load modules to avoid importing all dependencies at module level."""
    if name in _MODULE_NAMES:
        full_module_path = f"dysts.{name}"
        module = __import__(full_module_path, fromlist=[name])
        return module

    raise AttributeError(f"module 'dysts' has no attribute '{name}'")


def __dir__():
    return sorted(_MODULE_NAMES)
