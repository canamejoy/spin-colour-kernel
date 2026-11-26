"""spin-colour-kernel: exact linearly coloured Metropolis kernels for classical spin models on GPUs.

The short interface (describe, check, write, run, load, Results) lives in Spin_Colour_Kernel.api and is
also reachable from here and from the `sck` alias; it is imported on first use, so importing the package
loads neither torch nor triton.
"""

try:
    from importlib.metadata import PackageNotFoundError, version as _version
    __version__ = _version("spin-colour-kernel")
except Exception:                     # running from a source tree that was never installed
    __version__ = "unknown"

_API = ("describe", "check", "write", "run", "load", "Results")


def __getattr__(name):
    if name in _API:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module 'Spin_Colour_Kernel' has no attribute {name!r}")
