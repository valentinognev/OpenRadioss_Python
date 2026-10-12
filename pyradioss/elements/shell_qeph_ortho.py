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
#   engine/source/elements/shell/coquez/czfintn.F (modified: SUBROUTINE
#     CZFINTN_OR lines 1015-1376, of which the IORTH>0 selection — the
#     FACM/FACF at 1126-1140, the damper moduli at 1142-1147, the CM/CF
#     build at 1148-1160 and the DGLAS rows at 1164-1185 — is
#     transcribed here; the isotropic rows are CZFINTN1's at 184-227,
#     repeated verbatim from shell_qeph so both branches share one
#     increment)
#   engine/source/elements/shell/coqueba/cmatc3.F (added: CCTOGLOB
#     lines 412-458, the fibre -> element rotation of HM/HC, and the
#     IORTH=1 decision on MTN at 211-217)

"""``czfintn.F`` ``CZFINTN_OR`` — the ORTHOTROPIC QEPH stabilization.

``czforc3.F:678-724`` runs ``CZFINTN1`` when ``IMPL_S > 0 .OR. IORTH == 0``
and ``CZFINTN_OR`` otherwise; ``cmatc3.F:211-217`` raises ``IORTH`` purely
on the material type (``MTN == 19`` -> ``IORTH = 1``).  ``CZFINTN_OR`` then
builds its moduli from ``GEPM_LC`` -> ``CCTOGLOB`` -> ``HF = HM/12``
(``cmatc3.F:222-232``) and, where ``CZFINTN1`` had a single ``A11``/``A12``
pair, carries FOUR independent in-plane entries.

WHY THIS IS A SEPARATE MODULE
-----------------------------
``pyradioss/elements/shell_qeph.py`` is at its recorded line budget
(``tools/validation_data/size_budget.json``, 1600 lines) and may only
shrink, so the ``IORTH > 0`` selection lives here and ``shell_qeph``
calls :func:`elastic_increment` for BOTH branches — one function, so the
isotropic rows stay byte-for-byte the expressions they were.

THE SUBSTITUTION IS NOT SYMMETRIC
---------------------------------
The isotropic rows below are ``CZFINTN1``'s (``czfintn.F:184-227`` — the
routine ``shell_qeph._fint_stab`` was transcribed from, and still is for
isotropic groups)::

    C1M = A11*FAC1          C2M = A12*FAC1
    DGLAS(1) = C1M*CXX  - C2M*CYY        <- x membrane
    DGLAS(2) = C1M*CYY  - C2M*CXX        <- y membrane, ALSO A11
    DGLAS(3) = C1M*BXX  - C2M*BYY        <- x bending, also A11 (no /12)
    DGLAS(4) = C1M*BYY  - C2M*BXX        <- y bending, also A11

``CZFINTN_OR``'s ``IORTH > 0`` rows (``czfintn.F:1148-1185``)::

    CM(1,1) = FACM*HM(1)   CM(2,2) = FACM*HM(2)   CM(1,2) = FACM*HM(3)
    CF(1,1) = FACF*HF(1)   CF(2,2) = FACF*HF(2)   CF(1,2) = FACF*HF(3)
    DGLAS(1) = CM(1,1)*CXX - CM(1,2)*CYY         <- HM(1)
    DGLAS(2) = CM(2,2)*CYY - CM(1,2)*CXX         <- HM(2)  NOT HM(1)
    DGLAS(3) = CF(1,1)*BXX - CF(1,2)*BYY         <- HM(1)/12
    DGLAS(4) = CF(2,2)*BYY - CF(1,2)*BXX         <- HM(2)/12

and the transverse pair splits too — ``CZFINTN1`` uses one ``GS = G*SHF``
for slots 5, 6, 11 and 12 (``czfintn.F:1243``), ``CZFINTN_OR`` uses
``HC(1) = G23*SHF`` on 5 and 6 and ``HC(2) = G31*SHF`` on 11 and 12
(``czfintn.F:1179-1185``).

``*DT1`` CONVENTION
-------------------
``CZFINTN_OR`` builds ``CXX`` from the raw ``VHG`` (``czfintn.F:1093-1100``)
and multiplies the moduli by ``DT1`` instead (``FACM = FAC(1)*FAC1*DT1``,
``czfintn.F:1128-1130``); ``CZFINTN1`` builds ``DHG = VHG*DT1``
(``czfintn.F:104-111``) and leaves ``DT1`` out of the moduli
(``C1M = A11*FAC1``, ``czfintn.F:209``).  The two products are identical.
(The ``FAC`` dummy ``CZFINTN_OR`` scales by is ``ZCFAC`` — ``czforc3.F:715``
passes it, initialised to ONE at ``czforc3.F:387-388`` — so on this path
``FACM = FAC1*DT1`` by the port's standing ``ZCFAC = 1`` cut, exact for
elastic laws.)  The port keeps its ``CZFINTN1``-side ``dhg = vhg*dt``
convention throughout, so NEITHER the membrane coefficients NOR the
transverse ones carry a ``dt`` here: ``dg = modulus * hxx * dhg`` with the
modulus bare.  There is no ``1/dt`` anywhere and there must not be one.

DELIBERATELY NOT PORTED
-----------------------
* ``IORTH > 1`` / the ``CMF``/``HMFOR`` block (``czfintn.F:1200-1227``).
  ``cmatc3.F:218-224`` zeroes ``HMFOR`` for ``MTN == 19`` and nothing in
  the QEPH path raises ``IORTH`` above 1, so that whole block is
  unreachable for every law this port runs;
* ``CZHOUREP_OR`` (``czfintn.F:1920-2041``) — the plastic relaxation
  ``CZFINTN_OR`` calls at ``czfintn.F:1256`` where ``CZFINTN1`` inlines
  its own (``czfintn.F:300-351``).  See ``shell_qeph._fint_stab``: the
  port's relaxation is the ``CZFINTN1`` form and ``_current_yield``
  returns ``None`` for LAW19, so the two are equivalent-on-this-path
  today — upstream leaves ``SIGY = EP30`` for LAW19 (``czforc3.F:386``),
  which fails ``CZHOUREP_OR``'s ``SIGY < ZEP9EP30`` gate outright.  The
  substitution is a real remaining gap for yielding orthotropic laws and
  is named as one rather than papered over.
* ``HMOR`` / ``HM(5..6)`` / ``HF(5..6)`` — ``CCTOGLOB``'s third output,
  the supplementary membrane moduli (``D13``, ``D23``).  ``CZFINTN_OR``
  builds ``CM(1,3)``/``CM(2,3)``/``CF(1,3)``/``CF(2,3)`` from them
  (``czfintn.F:1153-1160``), but their only consumers are the ``CMH``/
  ``CFH`` coupling — zeroed unconditionally at ``czfintn.F:1191-1196`` —
  and the unreachable ``IORTH > 1`` block.  Dead on this path, hence not
  computed (``cctoglob`` returns ``(HM, HC)`` only).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..common.constants import EM20
from ..materials import shell_moduli as _shell_moduli

#: ``IORTH`` values upstream can take on the QEPH path (``cmatc3.F:211-217``
#: writes 1, and nothing in ``czfintn.F``'s callers writes 2).
IORTH_MAX = 1


def cctoglob(hm: np.ndarray, hc: np.ndarray,
             cs: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """``cmatc3.F:412-458`` CCTOGLOB — the fibre-frame -> element-frame
    rotation of ``HM(1..4)`` and ``HC(1..2)``, transcribed verbatim.

    ``cs`` is ``DIR(I,1..2)``, the direction-1 fibre cosines in the
    element frame (``shell_ortho.build_group_ortho``); ``None`` means the
    identity fibre ``DIR = (1, 0)``.  Returns NEW ``(HM, HC)`` arrays —
    upstream mutates in place, but a pure function cannot.

    ``HMOR`` is not computed: upstream's 3rd output feeds ``HM(5..6)`` /
    ``HFOR``, whose only ``CZFINTN_OR`` consumers (the ``CMH``/``CFH``
    coupling and the ``IORTH > 1`` block) are zeroed / unreachable on
    this path — see the module docstring.

    THE SHEAR ENTRY IS NOT THE TEXTBOOK ONE.  ``CM(4)`` reads
    ``s^2c^2(C11+C22-2C12-2C66) + (c^4+s^4)C66`` where classical laminate
    theory writes ``s^2c^2(C11+C22+2C12) + (c^4+s^4)C66``; upstream's
    differs by ``-4c^2s^2(C12+C66)``.  On E11=0.3, E22=0.5, NU12=0,
    G12=0.05 at 45 degrees that is 0.200 against the textbook 0.225.  The
    divergence is UPSTREAM's and is transcribed, not corrected.
    """
    hm = np.array(hm, dtype=float, copy=True)
    hc = np.array(hc, dtype=float, copy=True)
    if cs is None:
        return hm, hc
    cs = np.asarray(cs, dtype=float)
    m2 = cs[..., 0] * cs[..., 0]
    n2 = cs[..., 1] * cs[..., 1]
    m4 = m2 * m2
    n4 = n2 * n2
    mn = cs[..., 0] * cs[..., 1]
    mn2 = mn * mn

    h1, h2, h3, h4 = hm[..., 0], hm[..., 1], hm[..., 2], hm[..., 3]
    t1 = 2.0 * mn2 * h3 + 4.0 * mn2 * h4                    # cmatc3.F:441
    cm1 = m4 * h1 + n4 * h2 + t1                           # :442
    cm2 = n4 * h1 + m4 * h2 + t1                           # :443
    t2 = mn2 * (h1 + h2)                                    # :444
    cm3 = t2 + (m4 + n4) * h3 - 4.0 * mn2 * h4              # :445
    cm4 = t2 - 2.0 * mn2 * (h3 + h4) + (m4 + n4) * h4       # :446-447
    hm[..., 0] = cm1
    hm[..., 1] = cm2
    hm[..., 2] = cm3
    hm[..., 3] = cm4

    hc1 = m2 * hc[..., 0] + n2 * hc[..., 1]                 # :453
    hc2 = n2 * hc[..., 0] + m2 * hc[..., 1]                 # :454
    return hm, np.stack([hc1, hc2], axis=-1)


def group_moduli(slices: Sequence[tuple], cs: Optional[np.ndarray],
                 shf: np.ndarray, n: int,
                 law_of=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``cmatc3.F:222-232`` per element: ``GEPM_LC`` into ``HM``/``HC``,
    then ``CCTOGLOB`` with the element's fibre cosines.

    Returns ``(None, None)`` when no slice is an orthotropic material —
    the group then runs ``CZFINTN1``'s ``ELSE`` rows and the caller keeps
    the isotropic ``A11``/``A12``/``GS``/``GSR``/``A11SR``/``A12SR`` it
    already had.  (``cmatc3.F:211-217`` gates on ``MTN`` ALONE, so a LAW19
    property without a fibre vector still takes this path — ``cs = None``
    is then the identity fibre ``DIR = (1,0)``, which is what upstream's
    default direction gives.)"""
    hm = np.zeros((n, 4))
    hc = np.zeros((n, 2))
    seen = False
    for sl, mat, _prop in slices:
        law = law_of(mat) if law_of is not None else getattr(mat, "law", None)
        mods = _shell_moduli.shell_moduli(mat, law)
        if mods is None:
            continue
        c11, c22, c12, g12, hc1, hc2 = mods
        rows = np.zeros((int(np.count_nonzero(sl)), 4))
        rows[:, 0] = c11
        rows[:, 1] = c22
        rows[:, 2] = c12
        rows[:, 3] = g12
        hm[sl] = rows
        # GEPM_LC folds SHF into HC (cmatc3.F:508-509); SHF is per element
        hc[sl, 0] = hc1 * shf[sl]
        hc[sl, 1] = hc2 * shf[sl]
        seen = True
    if not seen:
        return None, None
    return cctoglob(hm, hc, cs)


def damper_moduli(hm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``czfintn.F:1136-1140`` — the three square-root moduli the linear
    damper runs on, in the ``IORTH > 0`` form::

        GSR  = SQRT(HM(4))                 = sqrt(G12)
        A11SR= SQRT(HALF*(HM(1)+HM(2)))    (the isotropic A11SR is sqrt(A11))
        A12SR= SQRT(MAX(EM20, HM(3)))      (the isotropic one is
                                             NUSR*A11SR with NUSR=sqrt(nu))
    """
    return (np.sqrt(hm[:, 3]),
            np.sqrt(0.5 * (hm[:, 0] + hm[:, 1])),
            np.sqrt(np.maximum(EM20, hm[:, 2])))


def elastic_increment(G: dict, st: dict, vhg: np.ndarray, dt: float,
                      cvis: float) -> np.ndarray:
    """``DGLAS(1..12)`` — the elastic modal-stress increment of
    ``CZFINTN_OR`` (``czfintn.F:1134-1196``) when ``st["hm"]`` is set, and
    of ``CZFINTN1``'s ``ELSE`` (``czfintn.F:1229-1247``) when it is not.

    ``cvis`` is ``FAC1`` (``GEO(17)`` = 1 for the QEPH property).  The
    isotropic rows are the port's previous expressions, unchanged; the
    orthotropic rows are the ``IORTH > 0`` substitution documented at the
    module head.  ``st["hm"]``/``st["hc"]`` are ``None`` on an isotropic
    group, which is the whole of the ``IORTH > 0`` dispatch on the port
    side (``czforc3.F:715-724``)."""
    a_i = G["a_i"]
    my34, my23 = G["my34"], G["my23"]
    mx34, mx23 = G["mx34"], G["mx23"]
    n = len(my34)
    hm = st.get("hm")

    dhg = vhg * dt
    c3g = 4.0 * a_i
    hxx = c3g * my34
    hyy = c3g * mx34
    hxx_k = c3g * my23
    hyy_k = c3g * mx23
    cxx = hxx * dhg[:, 0]
    cyy = hyy * dhg[:, 1]
    cxx_k = hxx_k * dhg[:, 0]
    cyy_k = hyy_k * dhg[:, 1]
    bxx = hxx * dhg[:, 2]
    byy = hyy * dhg[:, 3]
    bxx_k = hxx_k * dhg[:, 2]
    byy_k = hyy_k * dhg[:, 3]

    dg = np.zeros((n, 12))
    if hm is None:
        c1m = st["a11"] * cvis
        c2m = st["a12"] * cvis
        dg[:, 0] = c1m * cxx - c2m * cyy
        dg[:, 1] = c1m * cyy - c2m * cxx
        dg[:, 2] = c1m * bxx - c2m * byy
        dg[:, 3] = c1m * byy - c2m * bxx
        dg[:, 6] = c1m * cxx_k - c2m * cyy_k
        dg[:, 7] = c1m * cyy_k - c2m * cxx_k
        dg[:, 8] = c1m * bxx_k - c2m * byy_k
        dg[:, 9] = c1m * byy_k - c2m * bxx_k
        c2s = cvis * st["gs"] / 64.0                    # FAC1*G*SHF/64
    else:
        hc = st["hc"]
        c11 = cvis * hm[:, 0]
        c22 = cvis * hm[:, 1]
        c12 = cvis * hm[:, 2]
        f11 = cvis * (hm[:, 0] / 12.0)                  # HF(1) (cmatc3.F:225)
        f22 = cvis * (hm[:, 1] / 12.0)                  # HF(2)
        f12 = cvis * (hm[:, 2] / 12.0)                  # HF(3)
        dg[:, 0] = c11 * cxx - c12 * cyy
        dg[:, 1] = c22 * cyy - c12 * cxx
        dg[:, 2] = f11 * bxx - f12 * byy
        dg[:, 3] = f22 * byy - f12 * bxx
        dg[:, 6] = c11 * cxx_k - c12 * cyy_k
        dg[:, 7] = c22 * cyy_k - c12 * cxx_k
        dg[:, 8] = f11 * bxx_k - f12 * byy_k
        dg[:, 9] = f22 * byy_k - f12 * bxx_k
        c2s = cvis * hc[:, 0] / 64.0                    # FAC1*HC(1)/64
        c3s = cvis * hc[:, 1] / 64.0                    # FAC1*HC(2)/64
        dg[:, 4] = c2s * hxx * dhg[:, 4]
        dg[:, 5] = c2s * hyy * dhg[:, 4]
        dg[:, 10] = c3s * hxx_k * dhg[:, 5]
        dg[:, 11] = c3s * hyy_k * dhg[:, 5]
        return dg
    dg[:, 4] = c2s * hxx * dhg[:, 4]
    dg[:, 5] = c2s * hyy * dhg[:, 4]
    dg[:, 10] = c2s * hxx_k * dhg[:, 5]
    dg[:, 11] = c2s * hyy_k * dhg[:, 5]
    return dg


#: descriptive alias — the upstream routine name.
czfintn_or = elastic_increment