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
#   engine/source/elements/shell/coqueba/cmatc3.F  (modified: the
#     GEPM_LC plane-stress moduli of an orthotropic shell material,
#     lines 477-513, transcribed to a pure function of the layer's
#     material constants)

"""``GEPM_LC`` — the orthotropic shell plane-stress moduli, per layer.

``cmatc3.F:477-513`` is the shell side's answer to "what moduli does the
hourglass stiffness see?", and for an orthotropic material it is a
different answer from the isotropic one.  The isotropic path collapses the
material to ``PM(20) = MAX(E11, E22)/DETC`` and ``PM(21) = SQRT(N12*N21)``
(``hm_read_mat19.F:162-165``) — the LARGEST of the two Young moduli, on
every axis.  ``GEPM_LC`` keeps all four plane-stress entries and lets the
caller rotate them into the element frame::

    NU12 = 1 - PM(35)*PM(36)          <- the local, a MISNOMER for DETC
    A11  = E11/NU12          ->  HM(1)      C11
    A22  = E22/NU12          ->  HM(2)      C22
    A12  = PM(36)*A11        ->  HM(3)      C12
    G    = PM(37)            ->  HM(4)      G12
    HC(1) = G23*SHF          ->  HC(1)      transverse, x-normal face
    HC(2) = G31*SHF          ->  HC(2)      transverse, y-normal face

with ``PM(33..39) = E11, E22, N12, N21, G12, G23, G31``
(``hm_read_mat19.F:165-171``).

TWO THINGS THIS SIGNATURE DELIBERATELY IS NOT
---------------------------------------------
* it returns SIX entries, not the three ``(A11, A12, G)`` of the P3.3 plan
  text.  The Fortran has no ``A22`` in that substitution — ``czfintn.F``
  uses ``HM(1)`` on the x rows and ``HM(2)`` on the y rows
  (``czfintn.F:1160-1161``) — so a triple cannot express it.  The plan's
  own Step 3 ("read the _or branch's *which moduli and in which order* ...
  the substitution list is not symmetric") is the authority for widening
  it; AGENTS.md domain rules put the Fortran above the plan;
* they are PER LAYER, not per Gauss point.  ``GEPM_LC`` reads
  ``MX = MAT(JFT)`` — one material for the whole group — and broadcasts
  the same six constants to every element and every integration point of
  it (``cmatc3.F:496-513``).  The caller
  (``pyradioss.elements.shell_qeph_ortho``) is what turns them into a
  per-element ``HM``/``HC`` by applying ``CCTOGLOB`` with that element's
  own fibre cosines.

``shf`` defaults to 1 so the two-argument ``shell_moduli(layer, law)`` call
is meaningful; it is the geometric ``SHF`` (0 at ``N=1``, 5/6 otherwise)
and only ever multiplies the two TRANSVERSE moduli.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

#: material types ``cmatc3.F:477-513`` builds a GEPM_LC quadruple for.  Only
#: LAW19 (``MTN == 19``) takes this path in ``cmatc3.F:222-232``; the other
#: IORTH types (15/25/123/125/127/132, ``cmatc3.F:211-217``) only reach
#: GEPM_LC on their IGTYP 9 integration path, which the QEPH kernel does
#: not have.  ``IORTH = 2``/``HMFOR`` is deliberately NOT here — it is
#: unreachable for every law the port runs (see shell_qeph_ortho).
ORTHOTROPIC_SHELL_LAWS = frozenset({19, "19", "LAW19", "FABRI", "MAT_FABRI"})

_LAW_KEYS = ("E11", "MAT_EA")
_LAW_KEYS2 = ("E22", "MAT_EB")
_N12_KEYS = ("NU12", "MAT_PRAB")
_N21_KEYS = ("NU21",)
_G12_KEYS = ("G12", "MAT_GAB")
_G23_KEYS = ("G23", "MAT_GBC")
_G31_KEYS = ("G31", "MAT_GCA")


def _params(layer: Any) -> dict:
    """The layer's material constants as a dict, whatever shape it arrives
    in: a raw cfg record dict, a ``Material`` with a ``params`` dict, or a
    ``Law19Params`` dataclass (whose fields are lower-case)."""
    if isinstance(layer, dict):
        p = layer.get("params")
        return dict(p) if isinstance(p, dict) else dict(layer)
    p = getattr(layer, "params", None)
    if isinstance(p, dict):
        return dict(p)
    return {k.lower(): v for k, v in vars(layer).items()}


def _get(p: dict, keys) -> Optional[float]:
    for k in keys:
        if k in p:
            return float(p[k])
        if k.lower() in p:
            return float(p[k.lower()])
    return None


def shell_moduli(layer: Any, law: Any,
                 shf: float = 1.0) -> Optional[Tuple[float, ...]]:
    """``GEPM_LC`` — ``(C11, C22, C12, G12, HC1, HC2)`` for one layer.

    Parameters
    ----------
    layer : material, cfg record or dataclass
        anything carrying E11/E22/N12/(N21)/G12/G23/G31.  ``N21`` is
        recomputed as ``N12*E22/E11`` when absent, exactly as
        ``hm_read_mat19.F:121`` does.
    law : material type
        the ``MTN`` the caller saw; anything outside
        :data:`ORTHOTROPIC_SHELL_LAWS` returns ``None`` (this material has
        no GEPM_LC moduli — ``IORTH == 0``).
    shf : float
        the geometric ``SHF`` (``cmatc3.F:508-509``), which scales only the
        two transverse moduli.

    Returns
    -------
    ``(C11, C22, C12, G12, HC1, HC2)`` or ``None``.

    ``C11``/``C22``/``C12`` are the HM row of ``CZFINTN_OR``'s ``CM``
    (``czfintn.F:1143-1146``); ``HC1``/``HC2`` are its transverse pair
    (``czfintn.F:1182-1183``).
    """
    if law not in ORTHOTROPIC_SHELL_LAWS:
        return None
    p = _params(layer)
    e11 = _get(p, _LAW_KEYS)
    e22 = _get(p, _LAW_KEYS2)
    n12 = _get(p, _N12_KEYS)
    g12 = _get(p, _G12_KEYS)
    g23 = _get(p, _G23_KEYS)
    g31 = _get(p, _G31_KEYS)
    if None in (e11, e22, n12, g12):
        return None
    n21 = _get(p, _N21_KEYS)
    if n21 is None:
        n21 = n12 * e22 / e11 if e11 > 0.0 else n12
    detc = 1.0 - n12 * n21                      # GEPM_LC's NU12 (cmatc3.F:498)
    if detc == 0.0:
        return None
    c11 = e11 / detc
    c22 = e22 / detc
    c12 = n21 * c11
    if g23 is None:
        g23 = g12                                # law45_fabric-style fallback
    if g31 is None:
        g31 = g12
    return (c11, c22, c12, g12, g23 * shf, g31 * shf)


#: descriptive alias — the upstream routine name, so a reader holding the
#: Fortran open can find this directly.
gepm_lc = shell_moduli