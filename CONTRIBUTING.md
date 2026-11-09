# Contributing

Thank you very much for any contributions or suggestions! For recommendations of systems to add, please see the following instructions. For other contributions, please see the list below. 

## Development

We highly recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing dependencies.
To get started, clone the repo and setup the dependencies and package:
```
uv sync --group dev
uv pip install -e .
```

## New dynamical systems

If you would like a new chaotic system added to the database, please mention it on the issue thread [here](https://github.com/williamgilpin/dysts/issues/1). Because there are an infinite number of chaotic systems, we currently are only including systems that have appeared in published work.

The biggest bottleneck when adding new models is a lack of known parameter values and initial conditions, and so please provide a reference or code that contains all parameter values necessary to reproduce the claimed dynamics. For this reason, we would very much appreciate it if you could include as much of possible of the following pieces of information: **Right hand side** and **Metadata**

**1. Right hand side.** The first thing that we need is a working right-hand side of the dynamical system. The Lorenz system is currently represented this way. Just replace the name and `_rhs` function, and list the parameters in alphabetical order

    class Lorenz(DynSys):
        @staticjit
        def _rhs(x, y, z, t, beta, rho, sigma):
            xdot = sigma * (y - x)
            ydot = x * (rho - z) - y
            zdot = x * y - beta * z
            return xdot, ydot, zdot
            
If you also know the jacobian matrix, please include a function for that as well

    class Lorenz(DynSys):
        @staticjit
        def _rhs(x, y, z, t, beta, rho, sigma):
            xdot = sigma * (y - x)
            ydot = x * (rho - z) - y
            zdot = x * y - beta * z
            return xdot, ydot, zdot
        @staticjit
        def _jac(x, y, z, t, beta, rho, sigma):
            row1 = [-sigma, sigma, 0]
            row2 = [rho - z, -1, -x]
            row3 = [y, x, -beta]
            return [row1, row2, row3]

We list parameters and arguments by name (instead of using keyword arguments) in order to simplify compiling with `numba`


**2. Metadata.** The second thing we need is metadata about the system. I would really appreciate it if you could provide as much as possible of the information included in a standard database entry and implementation. For example, the Lorenz attractor has this JSON representation:

        "Lorenz": {
            "bifurcation_parameter": null,
            "citation": "Lorenz, Edward N (1963). Deterministic nonperiodic flow. Journal of the atmospheric sciences 20.2 (1963): 130-141.",
            "correlation_dimension": 1.993931310517824,
            "description": "A minimal weather model based on atmospheric convection.",
            "dt": 0.0001801,
            "embedding_dimension": 3,
            "hamiltonian": false,
            "initial_conditions": [
                -9.7869288,
                -15.03852,
                20.533978
            ],
            "kaplan_yorke_dimension": 2.062079534784479,
            "lyapunov_spectrum_estimated": [
                0.9050430367123482,
                -0.00044501018833715547,
                -14.571598026056346
            ],
            "maximum_lyapunov_estimated": 0.9050430367123482,
            "multiscale_entropy": 1.1541457906835575,
            "nonautonomous": false,
            "parameters": {
                "beta": 2.667,
                "rho": 28,
                "sigma": 10
            },
            "period": 1.5008,
            "pesin_entropy": 0.9050430367123482,
            "unbounded_indices": []
        },
        
Feel free to copy and paste and then fill out the fields that you know; we have utilities for recalculating `dt`, `initial_conditions`, `kaplan_yorke_dimension`, `lyapunov_spectrum_estimated`, `maximum_lyapunov_estimated`, `multiscale_entropy`, `period`, `pesin_entropy`, but an initial guess will improve these estimates

**3. Computing the entry.** Once the right-hand side is in `dysts/flows.py`, every numerical field above can be computed and written to the database in a single call:

    from dysts.flows import Liu
    from dysts.pipeline import add_system

    entry, report = add_system(
        Liu,
        parameters={"a": 10, "b": 40, "c": 2.5, "h": 4, "k": 1},
        initial_conditions=[0.5, 0.5, 20.0],
        citation="Liu, C., et al. (2004). A new chaotic attractor. Chaos, Solitons & Fractals 22, 1031-1038.",
        description="A Lorenz-like butterfly attractor.",
        doi="10.1016/j.chaos.2004.02.060",
        nonautonomous=False,
        plot_path="Liu.png",   # optional: attractor + first coordinate overlaid on Rossler
    )

The pipeline (i) finds an integration timestep and dominant period, (ii) rescales the period so that the first coordinate oscillates at the same rate as the Rossler system when both are resampled at 100 points per period, (iii) settles onto the attractor and detects unbounded variables, (iv) estimates the Lyapunov spectrum and its standard error from four runs of 1000 periods on the attractor (`dysts.analysis.lyapunov_spectrum`, which integrates the variational equations with the analytic Jacobian, or with central finite differences when the class has none), along with the Kaplan-Yorke dimension, Pesin entropy, correlation dimension and multiscale entropy, and (v) cross-checks the largest exponent with the Jacobian-free clone method and the sum of the exponents with the mean trace of the Jacobian. `report["flags"]` lists inconsistencies between these estimates. If the right-hand side is discontinuous (`sign`, Heaviside or switching terms), pass `spectrum_method="clones"`, since the variational equations miss the jumps of the tangent vectors at the switching surfaces; for stiff systems pass `integrator={"method": "LSODA", "rtol": 1e-10, "atol": 1e-12}`. Pass `write=False` to inspect the entry before it is written. The multiscale entropy requires the optional dependency `neurokit2`, and the ensemble steps are much faster when `numba` is installed.

## Releasing

Releases are published to PyPI and GitHub by `.github/workflows/python-publish.yml` when a version tag is pushed:

    # set version = "X.Y" in pyproject.toml, commit, then
    git tag -a vX.Y -m "Release notes"
    git push origin vX.Y

The workflow refuses to publish if the tag does not match the version in `pyproject.toml`, runs the unit tests, builds the sdist and wheel, and uploads them to PyPI (using the `PYPI_API_TOKEN` secret if present, otherwise PyPI Trusted Publishing from the `pypi` environment). It then creates the GitHub release for the tag, with the same files attached and the tag message as the release notes.

## Development to-do list

A partial list of potential improvements in future versions. We very much appreciate any help or suggestsions for these tasks.

+ Implement multivariate multiscale entropy and re-calculate for all attractors
+ Add a method for parallel integrating multiple systems at once, based on a list of names and a set of shared settings
+ + Can use multiprocessing for a few systems, but greater speedups might be possible by compiling all right hand sides into a single function acting on a large vector.
+ + Can also use this same utility to integrate multiple initial conditions for the same model
+ Add a separate jacobian database file, and add an attribute that can be used to check if an analytical one exists. This will speed up numerical integration, as well as potentially aid in calculating Lyapunov exponents.
+ Align the initial phases, potentially by picking default starting initial conditions that lie on the attractor, but which are as close as possible to the origin
+ Expand and finalize the discrete `dysts.maps` module
+ + Maps are deterministic but not differentiable, and so not all analysis methods will work on them. Will probably need a decorator to declare whether utilities work on flows, maps, or both
+ Switch stochastic integration to a newer package, like `torchsde` or `sdepy`


