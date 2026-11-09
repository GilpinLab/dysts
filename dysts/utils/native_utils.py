"""Native python utilities"""

import gzip
import importlib
import json
import os
import threading
import warnings
from inspect import Parameter, signature


def num_unspecified_params(func) -> int:
    """Count required parameters that haven't been specified through partial or defaults.

    Args:
        func: Function or partial function to inspect

    Returns:
        Number of parameters that must be specified when calling the function
    """
    if hasattr(func, "func"):  # Check if partial
        partial_func = func.func
        partial_args = len(func.args)
        partial_keywords = func.keywords or {}
    else:
        partial_func = func
        partial_args = 0
        partial_keywords = {}

    sig = signature(partial_func)
    required_count = 0

    for i, (name, param) in enumerate(sig.parameters.items()):
        if i < partial_args or name in partial_keywords:
            continue
        if param.default == Parameter.empty:
            required_count += 1

    return required_count


def has_module(module_name: str) -> bool:
    """Check if a module is installed"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _removed_delay_error(subject: str) -> ImportError:
    """The error raised when code asks for a delay system, which dysts no longer includes.

    Args:
        subject: what was requested, for example "MackeyGlass"

    Returns:
        An ImportError that tells the user how to install a version with delay systems
    """
    return ImportError(
        f"{subject} was removed from dysts. Delay differential equations are no longer "
        'included; to use them, install version 0.99 or earlier: pip install "dysts<=0.99"'
    )


_REMOVED_DELAY_SYSTEMS = (
    "MackeyGlass",
    "IkedaDelay",
    "SprottDelay",
    "VossDelay",
    "ScrollDelay",
    "PiecewiseCircuit",
)


def convert_json_to_gzip(fpath, encoding="utf-8", delete_original=False):
    """
    Convert a json file to a gzip file in a format that can be easily read by the
    `dysts` package. By default, the gzip file will be saved with the same name and
    in the same directory as the json file, but with a ".gz" extension.

    Args:
        fpath (str): Path to the json file to be converted
        encoding (str): Encoding to use when writing the gzip file
        delete_original (bool): Whether to delete the original json file after
            conversion. Default is False.

    Returns:
        None

    """
    if os.path.splitext(fpath)[1] == ".gz":
        warnings.warn("File already gzipped, exiting without conversion")
        return None

    with open(fpath, "r") as file:
        data = json.load(file)

    with gzip.open(fpath + ".gz", "wt", encoding=encoding) as file:
        # Wrap the GzipFile in a TextIOWrapper
        with open(file.name, "wt", encoding=encoding) as text_file:
            json.dump(data, text_file, indent=4)

    if delete_original:
        os.remove(fpath)


def group_consecutives(vals, step=1):
    """
    Return list of consecutive lists of numbers from vals (number list).

    References:
        Modified from the following
        https://stackoverflow.com/questions/7352684/
        how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    """
    run = list()
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


class ComputationHolder:
    """
    A wrapper class to force a computation to stop after a timeout.

    Parameters
        func (callable): the function to run
        args (tuple): the arguments to pass to the function
        kwargs (dict): the keyword arguments to pass to the function
        timeout (int): the timeout in seconds. If None is passed, the computation
            will run indefinitely until it finishes.

    Example
        >>> def my_func():
        ...     while True:
        ...         print("hello")
        ...         time.sleep(8)
        >>> ch = ComputationHolder(my_func, timeout=3)
        >>> ch.run()
        hello
        hello
        hello
        None

    """

    def __init__(self, func, *args, timeout: float = 10, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout

        def func_wrapped():
            sol = self.func(*self.args, **self.kwargs)
            setattr(self, "sol", sol)

        self.func_wrapped = func_wrapped

    def run(self):
        my_thread = threading.Thread(target=self.func_wrapped)
        my_thread.start()
        my_thread.join(self.timeout)  # kill the thread after `timeout` seconds

        return getattr(self, "sol", None)
