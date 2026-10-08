# -*- coding: utf-8 -*-
"""Deprecated alias for :mod:`pyradioss.input.keywords`.

The Starter readers moved into the ``pyradioss.input.keywords`` package in
task P1.5. This module only re-exports them, so that importers of the old
path keep working for one release cycle; importing it emits a
``DeprecationWarning``.

    from pyradioss.input.keywords import parse_starter_deck          # current
    from pyradioss.input.starter_keywords import parse_starter_deck   # deprecated
"""
from __future__ import annotations

import warnings as _warnings

from . import keywords as _keywords
from .keywords import *  # noqa: F401,F403  -- the package's public surface

_warnings.warn(
    "pyradioss.input.starter_keywords is deprecated; the Starter readers now "
    "live in pyradioss.input.keywords. Import them from there instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    """Reach the private card helpers the old module exposed (PEP 562).

    Readers do ``from pyradioss.input.starter_keywords import _is_numeric_card``;
    a star import cannot carry an underscore-prefixed name.
    """
    try:
        return getattr(_keywords, name)
    except AttributeError:
        raise AttributeError(
            "module %r has no attribute %r" % (__name__, name)) from None


def __dir__():
    return sorted(set(globals()) | set(dir(_keywords)))