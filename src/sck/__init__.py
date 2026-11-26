"""Short import name for Spin_Colour_Kernel: `import sck`, then sck.run(...), sck.check(...), sck.describe().

The functions are loaded on first use, so `import sck` itself loads neither torch nor triton.
"""

import Spin_Colour_Kernel as _package

__version__ = _package.__version__
__all__ = ["describe", "check", "write", "run", "load", "Results", "__version__"]


def __getattr__(name):
    return getattr(_package, name)


def __dir__():
    return sorted(__all__)
