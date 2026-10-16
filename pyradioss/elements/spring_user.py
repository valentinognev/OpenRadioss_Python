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
#   engine/source/elements/spring/rforc3.F  (the IGTYP >= 29 .AND. IGTYP <= 31
#     block: R4COOR3 + R5EVEC3 + R5DEF3 kinematics, the GBUF -> USER_*
#     scratch copy, the ENG_USERLIB_RUSER call per IGTYP, the USER_* ->
#     GBUF write-back, and the ANCMSG(MSGID=257) + ARRET(2) refusal when
#     USERL_AVAIL == 0 -- modified: the user object library absent from the
#     open tree is replaced by the in-process Python callback registry below)
"""User-defined spring elements — /PROP/TYPE29, TYPE30, TYPE31.

Fortran origin: ``engine/source/elements/spring/rforc3.F`` (the
``IGTYP >= 29 .AND. IGTYP <= 31`` block).  Upstream resolves a symbol from a
user object library (``ENG_USERLIB_RUSER``) that is absent from the open tree;
when the library is not linked (``USERL_AVAIL == 0``) the Engine prints
``ANCMSG(MSGID=257)`` (``PROP/USER1|2|3 - SPRING``) and stops with
``ARRET(2)``.  This module is the honest equivalent of that contract:

* :func:`register_user_spring` is the port's extension point, mirroring
  ``ENG_USERLIB_RUSER`` — the caller registers an in-process Python callable
  for IGTYP 29, 30 or 31.  Registration is explicit user code, never deck
  content: no ``/PROP`` card, ``/FUNCT`` table or input string selects,
  names or evaluates a callback.
* :func:`driver` **refuses loudly** with :class:`UserSpringNotRegistered`
  when no callback is registered for the group's IGTYP — the port of the
  MSGID 257 / ARRET(2) path.  A silent zero force would be an unbounded
  energy leak with no warning, so it is never an option.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ..common.constants import EP30

#: The IGTYP values upstream routes to ENG_USERLIB_RUSER (rforc3.F).
USER_SPRING_IGTYPS = frozenset({29, 30, 31})

#: Upstream /PROP spelling per user-spring IGTYP (the MSGID 257 option text).
_USER_PROP_OPTION = {29: "PROP/USER1 - SPRING", 30: "PROP/USER2 - SPRING", 31: "PROP/USER3 - SPRING"}

#: The registry: IGTYP -> user callable.  Empty means "library not linked".
_registry: Dict[int, Callable] = {}


class UserSpringNotRegistered(RuntimeError):
    """No user spring law registered for this IGTYP.

    Port of the ``USERL_AVAIL == 0`` path: ``ANCMSG(MSGID=257)`` +
    ``ARRET(2)``.  Raised instead of returning a silent zero force.
    """


def register_user_spring(igtyp: int, fn: Callable) -> None:
    """Register the user spring law for IGTYP 29, 30 or 31.

    ``fn`` is called once per element as ``fn(el, ctx)`` with the 0-based
    element index ``el`` and the driver context ``ctx``; it returns the
    element force as a 3-vector ``(fx, fy, fz)``, or a ``(force, moment)``
    pair of 3-vectors for a 6-DOF law (mirroring the USER_F*/USER_MOM*
    scratch upstream writes back into GBUF%FOR/GBUF%MOM).
    """
    if igtyp not in USER_SPRING_IGTYPS:
        raise ValueError(
            f"IGTYP {igtyp} is not a user spring: register only 29, 30 or 31"
        )
    if not callable(fn):
        raise TypeError(f"user spring for IGTYP {igtyp} must be callable, got {type(fn).__name__}")
    _registry[igtyp] = fn


def unregister_user_spring(igtyp: int) -> None:
    """Remove the user spring law for an IGTYP (re-arms the loud refusal)."""
    _registry.pop(igtyp, None)


def clear_user_springs() -> None:
    """Remove every registered user spring law."""
    _registry.clear()


def _resolve_igtyp(group, ctx) -> int:
    igtyp = getattr(group, "igtyp", None)
    if igtyp is None and isinstance(ctx, dict):
        igtyp = ctx.get("igtyp")
    if igtyp is None:
        raise UserSpringNotRegistered(
            "user spring group carries no IGTYP: refusing rather than "
            "guessing which user law applies"
        )
    return int(igtyp)


def driver(group, model, ctx) -> np.ndarray:
    """Run the registered user spring law over a group, once per element.

    Returns the per-element critical time step.  A user law cannot claim a
    step of its own, so every element reports ``EP30`` (no constraint) —
    the same sentinel the TYPE4 path uses for switched-off elements.
    """
    igtyp = _resolve_igtyp(group, ctx)
    try:
        fn = _registry[igtyp]
    except KeyError:
        option = _USER_PROP_OPTION.get(igtyp, f"PROP/USER - SPRING (IGTYP {igtyp})")
        raise UserSpringNotRegistered(
            f"IGTYP {igtyp} ({option}): no user spring law registered -- "
            "call spring_user.register_user_spring first "
            "(upstream: ANCMSG(MSGID=257) + ARRET(2) when USERL_AVAIL == 0)"
        ) from None
    n = int(group.n)
    force = np.zeros((n, 3), dtype=float)
    moment = np.zeros((n, 3), dtype=float)
    for el in range(n):
        out = fn(el, ctx)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            force[el] = np.asarray(out[0], dtype=float).reshape(3)
            moment[el] = np.asarray(out[1], dtype=float).reshape(3)
        else:
            force[el] = np.asarray(out, dtype=float).reshape(3)
    group.state["force"] = force
    group.state["moment"] = moment
    return np.full(n, EP30, dtype=float)
