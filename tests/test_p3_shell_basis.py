"""P3.4 — ``coqini.F`` startup tables, ``err_thk.F``, ``fequilibre.F``.

THE UPSTREAM CONTRACT (read from the Fortran, not extrapolated)
--------------------------------------------------------------
``engine/source/elements/shell/coqini.F`` (``SUBROUTINE COQINI``, called
by ``engine/source/engine/radioss2.F``) builds **four** tables at engine
startup out of ``DATA`` blocks and copies them into the globals
``Z0``, ``WF``, ``WM`` and ``ZTH``:

    DATA Z01/  l.46-68    -> Z0(11,11)   l.135-138
    DATA WF1/  l.70-92    -> WF(11,11)   l.136
    DATA WM1/  l.100-122  -> WM(11,11)   l.137
    DATA ZN1/  l.124-146  -> ZTH(12,11)  l.141-146

These are **data, not formulas** — they are transcribed digit-for-digit
here, and the spot-value test below pins the transcription.  The
``DATA`` statements fill column-major into the declared shape, so the
flat value stream runs one *rule* at a time, padded to the declared
column height.  Indexed the way the ENGINE indexes the globals
(``coqueba/cmatc3.F:780-781``):

    WF0 = WF(IT,NPT)          ->  WF()[NPT-1][IT-1]
    WM0 = Z0(IT,NPT)*WM(IT,NPT)  ->  Z01()[NPT-1][IT-1] * WM1()[NPT-1][IT-1]
    cfint_reg.F:211  Z0(K,NDDL), ZTH(K,NDDL) ->  [...][NDDL-1][K-1]

so **row ``r`` is the (r+1)-point rule** and column ``k-1`` its ``k``-th
entry.  Verified against the data itself: row ``r`` of ``Z01`` is the
equispaced natural coordinate set on ``[-0.5, +0.5]`` for ``r+1``
points, every row of ``WF1`` sums to exactly 1, and ``WF1()[0][0] == 1``
is the one-point rule.  ``ZN1`` declares ``ZN1(12,11)`` upstream, so its
Python counterpart is the **(11, 12)** transpose of that declaration, for
consistency with the other three.

A NOTE ON THE PLAN'S STEP-1 SPOT VALUES
----------------------------------------
``plan/04_phase3_elements_shell.md`` Task P3.4 Step 1 pins

    Z[2, 0] == -0.5   Z[2, 1] == 0.5
    Z[3, 0] == 0.0    Z[3, 1] == 0.0    Z[3, 2] == 0.0
    Z[4, 0] == -0.5   Z[4, 1] == 0.0    Z[4, 2] == 0.5

Two of those eight agree with the file; six do not, because they were
read off the *visual lines* of the ``DATA`` block rather than off the
value stream (``DATA`` lines carry 5 or 6 values each and do not align
with the 11-value rows).  ``coqini.F:47-68`` actually gives
``Z01[2] = (-.5, 0., .5, 0, ...)``, ``Z01[3] = (-.5, -.1666667,
.1666667, .5, ...)``, ``Z01[4] = (-.5, -.25, 0., .25, .5, ...)``.  Per
``AGENTS.md`` ("If the Fortran disagrees with your expectation, the
Fortran wins") — and following the ``P3.1`` precedent in
``test_p3_bt3_c43.py`` — the Fortran's own values are pinned here, with
the corrected slots called out in ``test_plan_spot_values_are_corrected``.

``err_thk.F`` AND ``fequilibre.F``
----------------------------------
Both are called by ``resol.F`` and both are read here in full; neither
is what the plan's Step 3/Interfaces describe.

* ``err_thk.F`` does **not** abort.  It is the thickness error
  *estimator* for the remesher: it accumulates element/nodal areas and
  ``area*thickness`` (l.115-235), hands them to ``admthke.F`` when
  ``NADMESH /= 0`` (l.264-272), and then computes a per-element error
  *indicator* (l.276-345)

      ERR_THK_SH4(N) = (1/4) * SUM over the 4 nodes of
                       | THICK_NOD(K)/max(EM30,AREA_NOD(K)) * (1/THICK_SH4(N)) - 1 |

  There is no ``EXIT``, no ``STOP`` and no message emission anywhere in
  the file.  The plan's "``err_thk.F`` produces *errors* (aborts) in
  upstream — port the abort" is a misreading of the ``ERR_`` name; the
  routine is ported here as the estimator it is, and **nothing aborts**.

* ``fequilibre.F`` is not a tolerance-based diagnostic.  It is the
  pre-solve step that spreads nodal accelerations onto the elements and
  frees ``A`` (l.52-118)

      WA(K)     = number of shell elements sharing node K
      FZERO(L,J,I) = A(L,K) / WA(K)          for each element corner J
      A(L,K)     = 0                        at every shell node

  It has no tolerance, no reference field and emits nothing.  The name
  ``force_equilibrium`` is kept (it is upstream's own); the plan's
  ``(fint, fint_ref, tol)`` signature is not implementable from this
  routine and has not been invented.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.constants import EM30
from pyradioss.elements import shell_basis


# --------------------------------------------------------------- tables


def test_z01_is_transcribed_not_derived():
    """coqini.F:46-68, DATA Z01/ -- digit-for-digit."""
    Z = shell_basis.Z01()
    assert Z.shape == (11, 11)
    # coqini.F:47-50 -- the 1-point and 2-point rules
    assert Z[0].tolist() == [0.0] * 11
    assert Z[1, 0] == -0.5 and Z[1, 1] == 0.5
    # coqini.F:51-56 -- the 3- and 5-point rules
    assert Z[2, 0] == -0.5 and Z[2, 1] == 0.0 and Z[2, 2] == 0.5
    assert Z[4, 0] == -0.5 and Z[4, 1] == -0.25 and Z[4, 2] == 0.0
    assert Z[4, 3] == 0.25 and Z[4, 4] == 0.5
    # coqini.F:65-68 -- the 10- and 11-point rules keep their last digits
    assert Z[9, 4] == -0.0555555 and Z[9, 9] == 0.5
    assert Z[10].tolist() == [-0.5, -0.4, -0.3, -0.2, -0.1,
                              0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def test_plan_spot_values_are_corrected():
    """The six plan spot values that do not match coqini.F, pinned as
    *not* matching, so a future edit cannot quietly reintroduce them."""
    Z = shell_basis.Z01()
    assert Z[2, 1] != 0.5      # plan said 0.5;  coqini.F:51 says 0.
    assert Z[3, 0] != 0.0      # plan said 0.0;  coqini.F:53 says -.5
    assert Z[3, 1] == -0.1666667 and Z[3, 2] == 0.1666667
    assert Z[4, 1] != 0.0      # plan said 0.0;  coqini.F:55 says -.25
    assert Z[4, 2] != 0.5      # plan said 0.5;  coqini.F:55 says 0.


def test_z01_rows_are_the_equispaced_coordinate_sets():
    """Cross-check of the transcription, not its source: the ``DATA``
    block for ``Z01`` is the equispaced set on [-0.5, +0.5] for r+1
    points, to the 7 significant digits the Fortran carries."""
    Z = shell_basis.Z01()
    for r in range(1, 11):
        n = r + 1
        assert np.allclose(Z[r, :n], -0.5 + np.arange(n) / r, atol=1e-7)
        assert np.all(Z[r, n:] == 0.0)


def test_wf1_rows_are_normalised_integration_weights():
    """coqini.F:70-92, DATA WF1/ -- every rule integrates to exactly 1."""
    W = shell_basis.WF1()
    assert W.shape == (11, 11)
    for r in range(11):
        assert W[r].sum() == pytest.approx(1.0, abs=1e-6)
    # coqini.F:71-72 -- the one-point rule carries weight 1
    assert W[0, 0] == 1.0
    # coqini.F:73-74 -- the two-point rule carries 1/2, 1/2
    assert W[1, 0] == 0.5 and W[1, 1] == 0.5


def test_wm1_is_transcribed_with_its_repeated_zeros():
    """coqini.F:100-122, DATA WM1/ -- the literal spelling of the repeated
    ``0.0000000`` entries matters and is pinned here."""
    W = shell_basis.WM1()
    assert W.shape == (11, 11)
    # coqini.F:105-106 -- the 2-point rule keeps -1/12, +1/12
    assert W[1, 0] == -0.0833333 and W[1, 1] == 0.0833333
    # coqini.F:118-119 -- the 7-point rule drops to three digits
    assert W[6, 3] == 0.0000000 and W[6, 7] == 0.0


def test_zn1_is_the_gauss_coordinate_table():
    """coqini.F:124-146, DATA ZN1/ -- declared ``ZN1(12,11)`` upstream, so
    the Python table is its (11, 12) transpose.  cfint_reg.F:209-211
    reads ``ZTH(1,2)=-0.5`` and ``ZTH(2,2)=+0.5`` for the two-point
    rule, which is ``ZN1()[1][0]`` and ``ZN1()[1][1]``."""
    G = shell_basis.ZN1()
    assert G.shape == (11, 12)
    assert G[0].tolist() == [0.0] * 12
    assert G[1, 0] == -0.5 and G[1, 1] == 0.5
    # coqini.F:145 -- the last rule keeps 7 digits
    assert G[10].tolist() == [-0.5, -0.45, -0.35, -0.25, -0.15, -0.05,
                              0.05, 0.15, 0.25, 0.35, 0.45, 0.5]


def test_the_four_tables_are_cheap_enough_to_leave_switched_on():
    """The ENGINE builds these at startup, so they are built once and
    handed back read-only; a caller cannot corrupt the shared table."""
    assert shell_basis.Z01() is shell_basis.Z01()
    assert shell_basis.WF1() is shell_basis.WF1()
    assert shell_basis.WM1() is shell_basis.WM1()
    assert shell_basis.ZN1() is shell_basis.ZN1()
    with pytest.raises(ValueError):
        shell_basis.Z01()[0, 0] = 1.0


# ------------------------------------------------------- err_thk.F


def _two_quads_with_a_thickness_step():
    """Two quads sharing the middle node, 0.1 thick on the left and 0.3 on
    the right, so the shared node's thickness/average-area ratio is off."""
    coords = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [2.0, 0.0, 0.0],
                       [2.0, 1.0, 0.0]])
    conn = np.array([[0, 1, 2, 3],
                     [1, 4, 5, 2]])
    return conn, coords, np.array([0.1, 0.3])


def test_err_thk_quad_area_is_one_eighth_of_the_diagonal_cross_product():
    """err_thk.F:160-174 -- ``ONE_OVER_8`` on (x3-x1) x (x4-x2).

    A unit square has diagonals (1,1,0) and (-1,1,0), cross product
    (0,0,2), so the upstream formula returns 2/8 = 0.25 -- a quarter of
    the geometric area.  The scale is upstream's own (it cancels in the
    estimator's thickness/area ratio, and ``AREA_SH4`` is exported to the
    remesher as-is), so the port reproduces it literally rather than
    "correcting" it to the true area.
    """
    conn, coords, _ = _two_quads_with_a_thickness_step()
    area = shell_basis.shell_areas(conn, coords)
    assert area.shape == (2,)
    assert np.allclose(area, 0.25)


def test_err_thk_node_thickness_is_area_times_thickness():
    """err_thk.F:176-188 -- ``AT = AREA*THICK`` deposited on every corner."""
    conn, coords, thick = _two_quads_with_a_thickness_step()
    area_nod, thick_nod = shell_basis.node_thickness(conn, coords, thick)
    # each unit quad deposits 0.25 (see the area test); nodes 1 and 2 are
    # shared by both quads
    assert area_nod.tolist() == [0.25, 0.5, 0.5, 0.25, 0.25, 0.25]
    # node 1 and node 2 are shared: 0.25*0.1 + 0.25*0.3
    assert thick_nod[1] == pytest.approx(0.1)
    assert thick_nod[0] == pytest.approx(0.025)


def test_thickness_error_is_zero_for_a_uniform_thickness():
    """err_thk.F:305-312 -- a constant thickness makes every TNk vanish."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    conn = np.array([[0, 1, 2, 3]])
    thick = np.array([0.2])
    area_nod, thick_nod = shell_basis.node_thickness(conn, coords, thick)
    err = shell_basis.thickness_error(conn, area_nod, thick_nod, thick)
    assert err.tolist() == [0.0]


def test_thickness_error_flags_the_element_on_the_thin_side_of_the_step():
    """err_thk.F:307-313 -- the indicator is normalised by the element's OWN
    thickness (``UNT = 1/THICK_SH4(N)``), so the same inconsistent
    neighbourhood flags the thin element harder than the thick one."""
    conn, coords, thick = _two_quads_with_a_thickness_step()
    area_nod, thick_nod = shell_basis.node_thickness(conn, coords, thick)
    err = shell_basis.thickness_error(conn, area_nod, thick_nod, thick)
    # left quad, thickness 0.1: nodes 0 and 3 match exactly (TNk = 0),
    # nodes 1 and 2 sit at twice the local ratio (TNk = 1 each)
    assert err[0] == pytest.approx(0.5)
    # right quad, thickness 0.3: nodes 4 and 5 match, nodes 1 and 2 sit at
    # 2/3 of the local ratio (TNk = 1/3 each)
    assert err[1] == pytest.approx(1.0 / 6.0)
    assert err[0] > err[1]


def test_thickness_error_never_aborts():
    """err_thk.F has no EXIT/STOP; the port must not invent one.  A
    zero-thickness element is skipped upstream (GBUF%OFF(I) <= ZERO), so
    the estimator must return a finite value, not raise."""
    conn, coords, thick = _two_quads_with_a_thickness_step()
    area_nod, thick_nod = shell_basis.node_thickness(conn, coords, thick)
    err = shell_basis.thickness_error(
        conn, area_nod, thick_nod, np.array([0.0, 0.3]))
    assert np.all(np.isfinite(err))


def test_thickness_error_uses_the_upstream_area_floor():
    """err_thk.F:307 -- ``MAX(EM30, AREA_NOD(K))`` guards the division."""
    conn = np.array([[0, 1, 2, 3]])
    err = shell_basis.thickness_error(
        conn,
        area_nod=np.zeros(4),
        thick_nod=np.ones(4),
        thickness=np.array([1.0]),
    )
    assert err == pytest.approx(1.0 / EM30 - 1.0)


# ---------------------------------------------------- fequilibre.F


def test_force_equilibrium_spreads_nodal_acceleration_over_the_elements():
    """fequilibre.F:78-104 -- ``FZERO(L,J,I) = A(L,K)/WA(K)``."""
    acc = np.array([[1.0], [0.0], [0.0]])          # node 0, x-acceleration
    fzero4, fzero3 = shell_basis.force_equilibrium(acc, np.array([[0]]))
    assert fzero4.shape == (3, 4, 1)
    assert fzero3 is None
    assert fzero4[0, :, 0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_force_equilibrium_divides_by_the_shell_count_of_the_node():
    """fequilibre.F:52-64 -- ``WA(K)`` counts every shell corner incidence.

    Node 1 is shared: it is corner 1 of quad 0 and corner 0 of quad 1, so
    both of its FZERO entries are 4/2 = 2.  Node 0 belongs to quad 0 alone
    and carries no acceleration, so its entry stays 0/1 = 0.
    """
    conn = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    acc = np.zeros((3, 6))
    acc[:, 1] = 4.0
    fzero4, _ = shell_basis.force_equilibrium(acc, conn)
    assert fzero4[:, 1, 0].tolist() == [2.0, 2.0, 2.0]   # 4/2
    assert fzero4[:, 0, 1].tolist() == [2.0, 2.0, 2.0]   # 4/2
    assert fzero4[:, 0, 0].tolist() == [0.0, 0.0, 0.0]   # 0/1


def test_force_equilibrium_frees_the_acceleration_at_shell_nodes():
    """fequilibre.F:106-118 -- ``A(L,K) = 0`` at every shell node."""
    conn = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    acc = np.ones((3, 6))
    shell_basis.force_equilibrium(acc, conn)
    assert np.all(acc == 0.0)


def test_force_equilibrium_leaves_non_shell_nodes_alone():
    """fequilibre.F:106-118 only walks the shell connectivity, so a node no
    shell references keeps its acceleration."""
    conn = np.array([[0, 1, 2, 3]])
    acc = np.ones((3, 6))
    shell_basis.force_equilibrium(acc, conn)
    assert acc[:, 4].tolist() == [1.0, 1.0, 1.0]
    assert acc[:, 5].tolist() == [1.0, 1.0, 1.0]


def test_force_equilibrium_handles_the_three_node_shells_too():
    """fequilibre.F:88-92 and 111-116 -- the SH3 branch, same formula."""
    conn3 = np.array([[0, 1, 2]])
    acc = np.ones((3, 3))
    fzero4, fzero3 = shell_basis.force_equilibrium(acc, None, conn3)
    assert fzero4 is None
    assert fzero3.shape == (3, 3, 1)
    assert fzero3[:, :, 0].tolist() == [[1.0] * 3] * 3
    assert np.all(acc == 0.0)


# ----------------------------------------------------- engine wiring


def test_force_equilibrium_is_reachable_from_the_engine():
    """The plan requires the diagnostic to be callable from ``engine.py``;
    ``resol.F`` is its caller upstream."""
    from pyradioss.engine import engine
    assert engine.force_equilibrium is shell_basis.force_equilibrium