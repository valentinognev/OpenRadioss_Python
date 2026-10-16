# Copyright (C) 2026 the pyradioss contributors.
#
# This file is part of pyradioss (OpenRadioss_Python), a Python port of the
# OpenRadioss explicit FEM solver, and is licensed under the GNU Affero
# General Public License, version 3 or later (AGPL-3.0-or-later).
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.  See LICENSE for the full text.
#
# Derived from OpenRadioss (Copyright (C) 2026 Siemens, AGPL-3.0-or-later):
#   common_source/modules/python_mod.F90 (the python_funct_id /
#     python_call_funct1D dispatch: plain ids go to FINTER, Python-function
#     ids go to user code -- transcribed as resolve() below, with the
#     execution of deck-supplied source replaced by an explicit registry)

"""The ``python_funct_mod`` spring callback (TYPE26 ``SPR_TAB`` / TYPE27 ``SPR_BDAMP``).

Fortran origin: ``common_source/modules/python_mod.F90``
(``python_funct_id``, ``python_call_funct1D``) as called from
``engine/source/elements/spring/r26def3.F`` and ``r27def3.F``.
Upstream, a curve id that names a ``/FUNCT_PYTHON`` entry is evaluated by
executing user Python; every other id goes through ``FINTER`` over the
``/FUNCT`` tables.

Security and determinism policy (binding, not advisory):

1. This module NEVER executes a deck-supplied string. There is no ``exec``,
   ``eval``, ``compile`` or ``__import__`` anywhere in this file (a static
   test in ``tests/test_p4_spring_python_funct.py`` asserts that), so a
   ``/FUNCT_PYTHON`` entry arriving from a deck can only ever raise
   :class:`PythonFunctNotRegistered` — a solver that ``exec()``s a deck
   line would be a remote-code-execution vector.
2. User Python runs only when the user populated the in-process registry
   (:func:`register_python_funct`) from their own code. Deck parsing never
   inserts callables, so anything resolved through the registry was put
   there deliberately, in this process.
3. Evaluation is pure and deterministic: the same abscissae always give the
   same ordinates, with no global-state mutation and no RNG. ``/FUNCT``
   tables keep ``FINTER`` slope-extrapolation semantics (see
   :class:`pyradioss.common.tables.FunctTable`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..common.tables import FunctTable

__all__ = [
    "PYTHON_FUNCT_ENABLED",
    "PythonFunctNotRegistered",
    "register_python_funct",
    "unregister_python_funct",
    "resolve",
]

#: Master switch for *local* ``/FUNCT``-table evaluation. Default ``True``:
#: plain tables are data (parsed at Starter time), not code, so resolving
#: them needs no opt-in. Set to ``False`` to refuse even local curves
#: (e.g. inside a hardened harness). Arbitrary user code (``/FUNCT_PYTHON``
#: deck text) is *always* off — it only runs via :func:`register_python_funct`.
PYTHON_FUNCT_ENABLED = True


class PythonFunctNotRegistered(LookupError):
    """A Python-curve id has no in-process implementation.

    Raised when the id names a ``/FUNCT_PYTHON`` deck entry (whose source
    text this module will not execute), when local-curve evaluation is
    disabled via :data:`PYTHON_FUNCT_ENABLED`, or when nothing is
    registered under the id at all. Upstream this state cannot occur
    silently either: ``python_funct_id`` returns 0 and the caller falls
    back to ``FINTER``, which fails loudly on a missing table.
    """


#: Explicit in-process registry: user id -> 1-D callable y(x).
#: Populated ONLY by user code in this process (never by deck parsing).
_python_registry: dict[int, Callable] = {}


def register_python_funct(fct_id: int, fn: Callable) -> None:
    """Register the in-process implementation of one ``/FUNCT_PYTHON`` id.

    ``fn`` receives a float ``np.ndarray`` of abscissae and returns the
    ordinates (scalar return is broadcast). Registration from user code is
    the ONLY way deck ``/FUNCT_PYTHON`` ids become callable.
    """
    key = int(fct_id)
    if key <= 0:
        raise ValueError(f"/FUNCT_PYTHON id must be positive, got {fct_id!r}")
    if not callable(fn):
        raise TypeError(f"/FUNCT_PYTHON/{key}: implementation must be callable")
    _python_registry[key] = fn


def unregister_python_funct(fct_id: int) -> None:
    """Remove one registry entry; missing ids are a no-op (test teardown)."""
    _python_registry.pop(int(fct_id), None)


def _tables_of(model: Any) -> Mapping:
    """Return the model's ``/FUNCT`` table map (empty when absent)."""
    for attr in ("functions", "functs"):
        tables = getattr(model, attr, None)
        if isinstance(tables, Mapping) and tables:
            return tables
    if isinstance(model, Mapping):
        return model
    return {}


def _is_deck_source(entry: Any) -> bool:
    """True when ``entry`` is unexecuted deck text, not an evaluator."""
    if isinstance(entry, str):
        return True
    if isinstance(entry, (tuple, list)) and any(isinstance(p, str) for p in entry):
        # Engine-deck form: (title, expr) from /FUNCT_PYTHON (M194).
        return True
    lines = getattr(entry, "lines", None)
    if isinstance(lines, list) and all(isinstance(s, str) for s in lines):
        # Starter-deck form: FunctPython(id, lines=[...]) (M119).
        return True
    return False


def _wrap_table(fct_id: int, entry: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Build a vectorised FINTER evaluator for one ``/FUNCT`` table entry."""
    if isinstance(entry, FunctTable) or hasattr(entry, "eval"):
        table = entry
    else:
        pairs = np.asarray(entry, dtype=float)
        if pairs.ndim != 2 or pairs.shape[0] < 1 or pairs.shape[1] != 2:
            raise PythonFunctNotRegistered(
                f"/FUNCT/{fct_id}: not a curve table and no Python "
                "implementation registered (deck source is never executed)"
            )
        table = FunctTable(fct_id, pairs[:, 0], pairs[:, 1])

    def evaluate(x) -> np.ndarray:
        out = table.eval(np.asarray(x, dtype=float))
        return np.asarray(out, dtype=float)

    return evaluate


def _wrap_user_fn(fct_id: int, fn: Callable) -> Callable[[np.ndarray], np.ndarray]:
    """Build a vectorised evaluator around a registered user callable."""

    def evaluate(x) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        try:
            out = np.asarray(fn(x_arr), dtype=float)
        except (TypeError, ValueError):
            out = np.asarray(
                [float(fn(float(v))) for v in x_arr.ravel()], dtype=float
            ).reshape(x_arr.shape)
        return np.broadcast_to(out, x_arr.shape).copy()

    evaluate.__name__ = f"python_funct_{fct_id}"
    return evaluate


def resolve(fct_id: int, model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return the curve evaluator for ``fct_id`` (shared by TYPE26/27).

    Dispatch mirrors ``python_mod.F90``: a *negative* id is a Python-function
    reference (upstream calls ``python_call_funct1D(PYTHON, -IFUNC, ...)``)
    and resolves ONLY through the :func:`register_python_funct` registry; a
    *positive* id first checks the registry, then the model's ``/FUNCT``
    tables (``FINTER`` semantics, gated by :data:`PYTHON_FUNCT_ENABLED`).
    Deck ``/FUNCT_PYTHON`` source text always raises
    :class:`PythonFunctNotRegistered` — it is data, never code.
    """
    key = int(fct_id)
    if key == 0:
        raise ValueError("function id 0 names no curve")
    if key < 0:
        try:
            return _wrap_user_fn(-key, _python_registry[-key])
        except KeyError:
            raise PythonFunctNotRegistered(
                f"/FUNCT_PYTHON/{-key}: no in-process implementation "
                "registered (deck source is never executed)"
            ) from None
    if key in _python_registry:
        return _wrap_user_fn(key, _python_registry[key])
    tables = _tables_of(model)
    if key in tables:
        if not PYTHON_FUNCT_ENABLED:
            raise PythonFunctNotRegistered(
                f"/FUNCT/{key}: local-curve evaluation is disabled "
                "(PYTHON_FUNCT_ENABLED is False)"
            )
        entry = tables[key]
        if _is_deck_source(entry):
            raise PythonFunctNotRegistered(
                f"/FUNCT_PYTHON/{key}: no in-process implementation "
                "registered (deck source is never executed)"
            )
        if callable(entry) and not hasattr(entry, "eval"):
            # In-process callable (never produced by deck parsing).
            return _wrap_user_fn(key, entry)
        return _wrap_table(key, entry)
    raise PythonFunctNotRegistered(
        f"function id {key}: no /FUNCT table on the model and no Python "
        "implementation registered"
    )
