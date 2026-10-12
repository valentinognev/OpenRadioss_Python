# Copyright (C) 2026 the pyradioss contributors.
#
# This file is part of pyradioss (OpenRadioss_Python), a Python port of the
# OpenRadioss explicit finite-element solver, and is licensed under the GNU
# Affero General Public License, version 3 or later (AGPL-3.0-or-later).
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.  See LICENSE for the full text.
#
# Derived from OpenRadioss (Copyright (C) 2026 Siemens, AGPL-3.0-or-later):
#   starter/source/elements/sh3n/coque3n/c3inmas.F  (the PTG vertex-angle
#     fractions, c3inmas.F:1136-1150, and the element rotary inertia
#     XI = EM*(AREA/4.5 + THK^2/12), c3inmas.F:1116)
#   starter/source/elements/initia/spmd_msin.F      (the nodal assembly
#     MS += MSTG*PTG / IN += INTG*PTG, spmd_msin.F:557-558)
"""P3.9 — DKT18 lumped-mass distribution (parity cases c06/c07/c00).

WHY THIS TEST EXISTS
--------------------
Task P3.0's audit (``tools/validation_data/shell_routine_status.json``) names
four CANDIDATE rows for the DKT18 MOMX deviation (c06 0.2813, c07 0.2581,
worst channel MOMX): ``cdkdefo3.F``, ``cdkderi3.F``, ``cdkfint3.F`` and
``cncoef3.F``.  Discriminated in order, against the Fortran read fresh from
``$OR_SRC`` (never recalled):

* ``cdkdefo3.F`` / ``cdkcurv3.F`` (same file) — the port kernels
  (``pyradioss/accel/jit_kernels/shells_dkt18.py::cdkdefo3/cdkcurv3``) match
  the Fortran term for term, including the ``EXX = VDEF*DT1`` increment
  scaling.  Excluded.
* ``cdkderi3.F`` (both ``CDKDERIC3`` and ``CDKDERI3``) — the port kernels
  match term for term, including ``VOL = THIRD*VOLG`` and every ``BZ/BRX/BRY``
  line; the Hammer-point set is the same set in a different order, which the
  ``+=`` accumulation is insensitive to.  Excluded.
* ``cdkfint3.F`` — the port kernel matches term for term (membrane
  ``C2 = VOL``, bending ``C2 = VOL*THK0``).  Excluded.
* ``cncoef3.F`` — for the c06/c07 decks (``/MAT/ELAST`` nu = 0, ``N = 0`` so
  the global ``MULAWGLC``/``SIGEPS01G`` path runs, ``ISMSTR = 0``) the moduli
  are exact, and the viscous leg is dead on both sides: upstream
  ``CNCOEF3`` leaves ``DN = 0`` for ``IHBE = ISH3N = 2`` with a blank ``dn``
  field (``cncoef3.F:425-435``; ``forintc.F`` passes ``ISH3N`` in the
  ``IHBE`` slot), so ``CBAVISC`` adds nothing upstream and the port's absent
  ``CBAVISC`` removes nothing.  Excluded for these decks.

What remains is one level up, on the starter side that P3.0's engine-routine
table does not cover: upstream distributes each triangle's lumped mass AND
rotary inertia to its three nodes by vertex-angle fractions
``PTG(K) = angle_K / pi`` (``c3inmas.F:1136-1150``, assembled in
``spmd_msin.F:557-558``), while ``shell_dkt18.init_group`` scatters uniform
thirds.  On the c06 mesh (right triangles ~16 x 15.2) that is a systematic
~17 %-of-element-mass redistribution on every element — the mass matrix, and
with it every bending vibration the MOMX channel records, differs.  Totals
are conserved either way, which is why the MASS channel matches (7.5e-08)
while the moment channels do not.

The tests below are the single-element hand-computed cases this hypothesis
names: a right triangle must scatter (1/2, 1/4, 1/4), an equilateral triangle
must scatter (1/3, 1/3, 1/3), and a degenerate triangle must stay finite.
"""

import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (
    build_element_groups,
    initialize_elements_and_mass,
    resolve_materials,
    resolve_node_groups,
)

MAT_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""


def _deck(nodes, elems):
    nod = "\n".join(f"{i + 1} {x} {y} {z}" for i, (x, y, z) in enumerate(nodes))
    sh3n = "\n".join(f"{i + 1} {a} {b} {c}" for i, (a, b, c) in enumerate(elems))
    return f"""\
/BEGIN
p3.9 single-triangle mass distribution deck
/NODE
{nod}
/SH3N/1
{sh3n}
/PART/1
triangles
1 1
{MAT_LAW1}
/PROP/SHELL/1
dkt18 shell prop
1 0 2 0
0.01 0.01 0.01
3 0 1.0
/END
"""


def _build(nodes, elems, tmp_path, name):
    f = tmp_path / f"{name}.rad"
    f.write_text(_deck(nodes, elems))
    log = MessageLog()
    model = Model()
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    resolve_node_groups(model, log)
    build_element_groups(model, log)
    initialize_elements_and_mass(model, log)
    return model


def test_dkt18_nodal_mass_uses_vertex_angle_fractions(tmp_path):
    """Right triangle (0,0)-(10,0)-(0,10): angles 90/45/45 deg, so the
    lumped mass AND rotary inertia scatter as (1/2, 1/4, 1/4) — hand-computed
    from c3inmas.F:1136-1150 (P1 = ACOS((A2+C2-B2)/(2*AA*CC))/PI, ...) and
    assembled per spmd_msin.F:557-558.  Uniform thirds fail this test."""
    model = _build(
        [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)],
        [(1, 2, 3)],
        tmp_path,
        "right",
    )
    rho, thick, area = 7.8e-6, 1.0, 50.0
    mass = rho * thick * area
    xiner = mass * (area / 4.5 + thick**2 / 12.0)
    # Node 1 is the right angle (1/2); nodes 2, 3 are 45 deg (1/4 each).
    assert np.isclose(model.mass[0], 0.5 * mass)
    assert np.isclose(model.mass[1], 0.25 * mass)
    assert np.isclose(model.mass[2], 0.25 * mass)
    assert np.isclose(model.mass[:3].sum(), mass)
    assert np.isclose(model.inertia[0], 0.5 * xiner)
    assert np.isclose(model.inertia[1], 0.25 * xiner)
    assert np.isclose(model.inertia[2], 0.25 * xiner)
    assert np.isclose(model.inertia[:3].sum(), xiner)


def test_dkt18_equilateral_keeps_uniform_thirds(tmp_path):
    """Equilateral triangle, side 10: every angle is 60 deg, so PTG is
    (1/3, 1/3, 1/3) and the old uniform-thirds behaviour is reproduced."""
    s = 10.0
    h = s * np.sqrt(3.0) / 2.0
    model = _build(
        [(0.0, 0.0, 0.0), (s, 0.0, 0.0), (s / 2.0, h, 0.0)],
        [(1, 2, 3)],
        tmp_path,
        "equi",
    )
    area = 0.5 * s * h
    mass = 7.8e-6 * 1.0 * area
    assert np.allclose(model.mass[:3], mass / 3.0)
    assert np.isclose(model.mass[:3].sum(), mass)


def test_dkt18_degenerate_triangle_stays_finite(tmp_path):
    """Collinear nodes (zero area): upstream ACOS degenerates, but the port
    must not scatter NaNs — it falls back to uniform thirds and the error
    the kernel already logs is the only signal."""
    model = _build(
        [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        [(1, 2, 3)],
        tmp_path,
        "degen",
    )
    assert np.all(np.isfinite(model.mass[:3]))
    assert np.all(np.isfinite(model.inertia[:3]))


# ============================================================================
# P3.0 cdkfint3.F hypothesis discriminator: single-element bending case with
# a hand-computed moment.
#
# P3.0's CANDIDATE (force) row says: "If the strain side is right and only
# the force assembly diverges, this is the routine ... Discriminate by
# running the port's forces on a Fortran state: same SIG/EPS, different F/M."
# The kernel below IS the port's force assembly
# (``pyradioss/accel/jit_kernels/shells_dkt18.py::cdkfint3``), checked here
# against hand evaluation of ``cdkfint3.F`` itself — every literal below was
# derived from the Fortran lines, not from running the kernel.  (Two
# transcription slips in the draft — an ``XS2``/``XS3`` swap feeding
# ``RXY2(2)`` — were caught because the kernel disagreed; the corrected
# values below follow the Fortran text.)  This test PASSES, which excludes
# the cdkfint3 hypothesis: same resultants in, same forces/moments out.
# ============================================================================


def test_cdkfint3_assembles_hand_computed_moment():
    """Single-element bending assembly with hand-computed F/M.

    Triangle (0,0)-(4,0)-(0,3): AREA2 = 12, PX2 = 1/4, PY2 = 0, PX3 = 0,
    PY3 = 1/3 (``CDKDERIC3`` by hand: ``AREAI = 1/12``).  Hammer point 1
    (KSI = ETA = 1/6, so X2Y = Y2X = 1/2, X6 = Y6 = 0, XY6 = -3):

    * ``BZ1 = (-0.2275, 0.04)``: ``XR1 = 1.5*0.5 = 0.75``,
      ``XR3 = -0.96/6 = -0.16``, ``BZ1(1) = 0.25*(-0.75-0.16)``.
    * ``BRX3(2) = -0.285``: ``RXY1(2) = 0.25*(0-0.82) = -0.205``,
      ``RXY2(2) = (1/3)*(-0.24+0) = -0.08`` (``XS2 = -PXY(3)*KSI = 0``,
      ``XS3 = PXY(1)*KSI = -0.24``).
    * ``BRY2(2) = +0.08 = -RXY2(2)``.

    With ``VOL = 4`` (``THIRD*VOLG``), ``THK0 = 2``,
    ``FOR = (10, 20, 30)``, ``MOM = (1, 2, 3)`` the Fortran gives the
    membrane resultants ``FX2 = 10``, ``FY2 = 30``, ``FX3 = 40``,
    ``FY3 = 80/3`` and, with the bending scale ``C2 = VOL*THK0 = 8``,
    ``F32 = 8*(-0.2275 + 2*0.04 + 3*0.06) = 0.26``,
    ``M11 = 8*(0 + 2*2.5/3 + 3*0.125) = 49/3``,
    ``M21 = 8*(-0.625 + 0 + 3*(-1/6)) = -9``,
    ``M12 = 8*(-0.06 + 2*0.06 + 3*(-0.285)) = -6.36``,
    ``M22 = 8*(-0.205 + 2*0.08 + 3*0.12) = 2.52``,
    ``M13 = 8*(-0.06 + 2*0.68/3 + 3*(-0.16)) = -0.6933``,
    ``M23 = 8*(-0.08 + 2*0.08 + 3*0.86/3) = 7.52``.
    ``F11/F21`` (and the drill slots) are untouched, exactly as upstream
    leaves them for the ``CDKFCUM3`` equilibrium fill.
    """
    from pyradioss.accel.jit_kernels.shells_dkt18 import (
        cdkderi3,
        cdkderic3,
        cdkfint3,
    )

    n = 1
    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, vol = cdkderic3(
        np.array([4.0]), np.array([0.0]), np.array([0.0]), np.array([3.0]),
        np.array([12.0]), np.array([12.0]), np.array([0.0]), np.array([4.0]),
    )
    assert np.allclose(px2, [0.25])
    assert np.allclose(py3, [1.0 / 3.0])
    bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
        px2, py2, px3, py3, px, py, pxy, pyy, 1.0 / 6.0, 1.0 / 6.0
    )
    # Hand-derived operator spot checks (the CDKDERI3 lines above).
    assert np.allclose(bz1[0], [-0.2275, 0.04])
    assert np.isclose(brx3[0][1], -0.285)
    assert np.isclose(bry2[0][1], 0.08)

    f = {k: np.zeros(n) for k in
         ("f11", "f12", "f13", "f21", "f22", "f23", "f32", "f33")}
    m = {k: np.zeros(n) for k in
         ("m11", "m12", "m13", "m21", "m22", "m23")}
    cdkfint3(np.array([4.0]), np.array([2.0]),
             np.array([[10.0, 20.0, 30.0]]), np.array([[1.0, 2.0, 3.0]]),
             px2, py2, px3, py3, bz1, bz2, bz3,
             brx1, brx2, brx3, bry1, bry2, bry3,
             f["f11"], f["f12"], f["f13"], f["f21"], f["f22"], f["f23"],
             f["f32"], f["f33"],
             m["m11"], m["m12"], m["m13"], m["m21"], m["m22"], m["m23"])
    assert np.allclose([f["f12"][0], f["f22"][0], f["f13"][0], f["f23"][0]],
                       [10.0, 30.0, 40.0, 80.0 / 3.0])
    assert np.isclose(f["f32"][0], 0.26)
    assert np.isclose(f["f33"][0], -3.093333333333333)
    assert np.isclose(m["m11"][0], 49.0 / 3.0)
    assert np.isclose(m["m21"][0], -9.0)
    assert np.isclose(m["m12"][0], -6.36)
    assert np.isclose(m["m22"][0], 2.52)
    assert np.isclose(m["m13"][0], -0.6933333333333334)
    assert np.isclose(m["m23"][0], 7.52)
    # Untouched slots stay zero for the CDKFCUM3 equilibrium fill.
    assert f["f11"][0] == 0.0 and f["f21"][0] == 0.0
