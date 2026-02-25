# tifffile/__init__.py

from .tifffile import *
from .tifffile import __all__, __doc__, __version__, main

# constants are repeated for documentation

__version__ = __version__
"""Tifffile version string."""

import os as _os

if _os.environ.get('TIFFFILE_NO_CPP'):
    _HAS_CPP = False
else:
    try:
        from ._tifffile_ext import __version__ as _cpp_version  # noqa: F811
        _HAS_CPP = True
    except ImportError:
        _HAS_CPP = False
