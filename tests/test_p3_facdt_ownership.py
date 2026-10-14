"""P3.12 — single ownership of the Batoz-family condensed dt claim.

Upstream contract (read from the Fortran, read-only):
  * QBAT condenses its dt length with ``FACDT = FOUR_OVER_3``
    (``coqueba/cbacoor.F`` line 1097);
  * QEPH condenses with ``FACDT = FIVE_OVER_4``
    (``coquez/czcorc.F`` line 380);
  * the BT kernel never owned either: in M40 ``shell_bt4`` carried a
    ``_CONDENSED_FACDT`` table plus an Ishell 12/22/24 dt-claim cap so the
    (then only) BT kernel could floor Batoz decks.  M41 routed those decks
    to the owning kernels (``_dispatch_shell_formulations`` splits
    12/22/23/24 out of ``model.shells`` *before* any ``init_group`` runs),
    M46 deleted the table, the ``_condensed_length`` helper and the cap —
    and this task removes the last live Ishell 12/22/24 dt-claim branch in
    this file, the ``FAC=12`` inertia lumping in ``init_group``
    (``cinmas.F`` ``FAC=TWELVE`` for engine ``IHBE>=11``).  After this, a
    ``shells`` group is lumped ``A/9`` unconditionally (``cinmas.F``
    ``FAC=NINE``, the BT family value); the ``A/12`` lumping lives in the
    owning kernels (``shell_qbat.py:378``).

Single ownership, pinned so it cannot drift back:
  1. ``shell_bt4`` exposes no ``_CONDENSED_FACDT`` and no
     ``_condensed_length`` (the M46 deletion stays deleted);
  2. ``shell_bt4.init_group`` lumps ``dt_iner = m/4*(A/9 + t^2/12)`` even
     for a slice carrying a Batoz card — no family branch fires here;
  3. the owning kernels carry the condensation constants
     (QBAT ``4/3``, QEPH ``5/4``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.common.constants import EP30
from pyradioss.elements import shell_bt4, shell_qbat, shell_qeph
from pyradioss.model.model import Model


class _Mat:
    """LAW1 steel stub — only what init_group touches."""

    def __init__(self, E=1000.0, nu=0.0, rho0=0.01):
        self.E = E
        self.nu = nu
        self.rho0 = rho0
        self.G = E / (2.0 * (1.0 + nu))
        self.law = 1
        self.fail = None
        self.params = {}

    def sound_speed_shell(self):
        return np.sqrt(self.E / self.rho0)


class _Prop:
    def __init__(self, thick=20.0, nip=3, ishell=1):
        self.thick = thick
        self.nip = nip
        self.params = {"thick": thick, "nip": nip, "ishell": ishell,
                       "hm": 0.01, "hf": 0.01, "hr": 0.01}


def _bt_group(ishell: int):
    """One 16 x 15.2 BT-group quad (the c04 welded-element geometry)."""
    coords = np.array([[0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
                       [16.0, 15.2, 0.0], [0.0, 15.2, 0.0]])
    conn = np.array([[0, 1, 2, 3]])
    mat, prop = _Mat(), _Prop(ishell=ishell)
    group = SimpleNamespace(conn=conn, n=1, ids=np.array([1]),
                            state={"slices": [(slice(0, 1), mat, prop)]})
    model = Model()
    model.x0 = coords.copy()
    shell_bt4.init_group(group, model, None)
    return group, coords, mat


def test_bt4_has_no_condensed_facdt_branches():
    assert not hasattr(shell_bt4, "_CONDENSED_FACDT")
    assert not hasattr(shell_bt4, "_condensed_length")


def test_bt4_lumps_bt_inertia_for_batoz_cards():
    """The last shadowed branch: an Ishell=12 slice reaching the BT kernel
    must be lumped with the BT-family FAC=9 share, not cinmas.F's
    IHBE>=11 FAC=12 — that lumping is owned by shell_qbat."""
    group, _, _ = _bt_group(ishell=12)
    m = group.state["mass"][0]
    a = group.state["area0"][0]
    t = group.state["thick"][0]
    assert group.state["dt_iner"][0] == pytest.approx(m / 4.0 * (a / 9.0 + t ** 2 / 12.0))


def test_bt4_dt_claim_has_no_condensed_cap():
    """The cycle-0 probe claim is the pure BT side-length claim
    (``dtfac * lc / c``) with no condensed-length min-cap — the M40 cap
    stays deleted even for a Batoz card."""
    group, coords, mat = _bt_group(ishell=12)
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt = shell_bt4.forces(group, coords, None, None, 0.0, fint, mint)
    assert np.all(dt < EP30)
    lc = 16.0 * 15.2 / 16.0  # A / longest side
    assert dt[0] == pytest.approx(group.state["dtfac"][0] * lc / mat.sound_speed_shell())


def test_condensation_owned_by_computing_kernels():
    """The condensation constants live exactly where the dt is claimed:
    QBAT 4/3 (cbacoor.F:1097), QEPH 5/4 (czcorc.F:380)."""
    assert shell_qbat._FACDT == pytest.approx(4.0 / 3.0)
    assert shell_qeph._FACDT == pytest.approx(1.25)
