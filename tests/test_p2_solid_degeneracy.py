"""Task P2.6 — degeneracy detection and element deletion (``degenes8`` family).

What the tests hold down, in the order the plan lists them:

* the plan's own sample — a brick with a repeated node and a brick with a
  fully flat face are both reported degenerate, and either still yields a
  finite positive dt and finite internal forces;
* the detection **agrees with the Starter's collapsed-hexa mark**
  (``starter/initialization.py``: 5, 6 or 7 distinct nodes run as a collapsed
  hexa) — a second, incompatible definition of "degenerate" is the failure
  mode the plan's Step 3 calls out, so the agreement is a test, not a claim;
* the degeneracy factors (``sldege`` / ``sdlen_dege``) and the characteristic
  length they feed stay finite and positive on a collapsed brick;
* the ``dim_tshedg`` / ``ind_tshedg`` assembly mesh, including the ``ITAG``
  rule that lets a node belong to at most one degenerate direction.

Fortran read before any of this was written:
``engine/source/elements/solid/solide/degenes8.F`` (the IDEGE count),
``.../solide/idege8.F`` + ``.../solide/idege.F`` (the per-face degenerate-quad
scan), ``.../solide/sldege.F`` + ``.../solide/sdlen_dege.F`` (the FAC ladder
and the ``LAT`` correction), ``.../solide/deges4v.F`` +
``.../solide/nodedege.F`` + ``.../solide/tetra4v.F`` (the fully collapsed
case), and ``engine/source/elements/thickshell/solidec/dim_tshedg.F``,
``.../ind_tshedg.F``, ``.../tshcdcom_ini.F``, ``.../tshcdcom_dim.F``
(the degenerate-direction edge list).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.constants import EM20
from pyradioss.elements import solid_degeneracy, solid_hexa8
from pyradioss.model.model import ElementGroup, Model
from pyradioss.starter import initialization as starter_init


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Mat:
    def __init__(self, law=1, rho0=1000.0, E=2.1e11, nu=0.3):
        self.law = law
        self.rho0 = rho0
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.fail = None
        self.eos = None
        self.params = {"E": E, "nu": nu, "sig_y": 200.0e6,
                       "eps_max": 1e30, "eps_p_max": 1e30}

    def sound_speed_solid(self):
        return float(np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0))


class _Prop:
    def __init__(self, qa=1.1, qb=0.05, h=0.1, isolid=1, icontrol=0):
        self.params = {"qa": qa, "qb": qb, "h": h, "isolid": isolid,
                       "icontrol": icontrol}


class _Log:
    def __init__(self):
        self.errors = []
        self.infos = []

    def error(self, msg, tag=""):
        self.errors.append((msg, tag))

    def warning(self, msg, tag=""):
        pass

    def info(self, msg, tag=""):
        self.infos.append((msg, tag))


def _compact(coords):
    """Deduplicate coordinates into node indices, exactly as a collapsed
    /BRICK is written in a deck: ``n1 n1 n3 n4 n5 n5 n7 n8`` is a connectivity
    with repeated node IDS, not eight nodes that happen to share a position."""
    pts = np.asarray(coords, dtype=float)
    seen, x, conn = {}, [], []
    for p in pts:
        key = tuple(float(v) for v in p)
        if key not in seen:
            seen[key] = len(x)
            x.append(p)
        conn.append(seen[key])
    return np.asarray(x, dtype=float), np.asarray(conn, dtype=np.int64)[None, :]


def _group(x, conn, mat=None, prop=None, init=True):
    """An :class:`ElementGroup` over ``x``/``conn``, initialized like the
    engine does (this is the shape every other ``elements/solid_*`` task
    builds).  ``x`` must have a row for every node index in ``conn``.
    """
    conn = np.asarray(conn, dtype=np.int64)
    x = np.asarray(x, dtype=float)
    model = Model()
    model.x0 = x.copy()
    model.x = x.copy()
    model.v = np.zeros((len(model.x), 3))
    model.vr = np.zeros((len(model.x), 3))
    group = ElementGroup(ids=np.arange(1, conn.shape[0] + 1, dtype=np.int64),
                         conn=conn,
                         part=np.zeros(conn.shape[0], dtype=np.int64))
    group.state["slices"] = [(slice(0, conn.shape[0]),
                              mat if mat is not None else _Mat(),
                              prop if prop is not None else _Prop())]
    if init:
        solid_hexa8.init_group(group, model, _Log())
    return group, model


def _degenerate_hexa8_group(coords):
    x, conn = _compact(coords)
    return _group(x, conn)


def _unit_hexa8_group():
    x = 0.5 * (solid_hexa8._XI + 1.0)
    return _group(x, np.arange(8, dtype=np.int64)[None, :])


def _brick_group(conn, init=True):
    """One well-formed unit brick per element, laid out node ``8*e + k``.

    A group whose connectivity is the thing under test and whose geometry is
    incidental: node positions come from the canonical hexa, so a sweep of
    connectivities can be read for counts without inventing a mesh.
    """
    conn = np.asarray(conn, dtype=np.int64)
    n = conn.shape[0]
    x = np.tile(0.5 * (solid_hexa8._XI + 1.0), (n, 1, 1)).reshape(-1, 3)
    return _group(x, conn, init=init)


#: the plan's two samples, verbatim
PLAN_SAMPLES = {
    "repeated_nodes": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]],
    "flat_face": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]],
}


# ---------------------------------------------------------------------------
# 1. the plan's test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(PLAN_SAMPLES))
def test_degenerate_solid_gets_a_defined_dt_and_finite_forces(name):
    g, model = _degenerate_hexa8_group(PLAN_SAMPLES[name])
    degen = solid_degeneracy.detect(g)
    assert degen.any()
    fint = np.zeros((len(model.x), 3))
    dt = solid_hexa8.forces(g, model.x, model.v, model.vr, 1e-3, fint, None)
    assert np.all(np.isfinite(dt)) and np.all(dt > 0.0)
    assert np.all(np.isfinite(fint))


def test_a_healthy_brick_is_not_degenerate():
    g, _ = _unit_hexa8_group()
    assert not solid_degeneracy.detect(g).any()


def test_a_flat_face_given_as_distinct_nodes_is_still_degenerate():
    """``degenes8`` counts repeated node IDS.  A deck can also flatten a face
    by giving every corner its own node — the element is just as degenerate,
    and upstream's own answer to that is ``nodedege`` (identical positions).
    """
    coords = np.asarray(PLAN_SAMPLES["flat_face"], dtype=float)
    g, model = _group(coords, np.arange(8, dtype=np.int64)[None, :])
    assert solid_degeneracy.degenes8(g)[0] == 0          # no repeated id
    assert not solid_degeneracy.detect(g)[0]             # connectivity alone
    assert solid_degeneracy.detect(g, model.x)[0]        # positions agree


# ---------------------------------------------------------------------------
# 2. the Starter and the Engine must not disagree about what degenerate means
# ---------------------------------------------------------------------------
def _starter_report(conns):
    """Run the Starter's own /BRICK conversion and read back what it said.

    ``_convert_degenerated_bricks`` keeps a 5/6/7-distinct-node /BRICK as a
    collapsed hexa and counts it in its own message, moves a 4-distinct one to
    /TETRA4, and passes an 8-distinct one through untouched.  Returns the ids
    it kept as bricks, the ids it moved to /TETRA4, and the collapsed-hexa
    COUNT it printed — three independent pieces of the Starter's own answer.
    """
    model = Model()
    for i, conn in enumerate(conns, start=1):
        model.raw_elems["BRICK"].append((i, 0, [int(c) + 1 for c in conn]))
    log = _Log()
    starter_init._convert_degenerated_bricks(model, log)
    assert not log.errors, log.errors
    kept = {eid for eid, _pid, _nodes in model.raw_elems["BRICK"]}
    moved = {eid for eid, _pid, _nodes in model.raw_elems["TETRA4"]}
    count = 0
    for msg, _tag in log.infos:
        if "COLLAPSED HEXA" in msg:
            count = int(msg.split()[0])
    return kept, moved, count


CONNECTIVITY_SWEEP = [
    [0, 1, 2, 3, 4, 5, 6, 7],                                  # 8 distinct
    [0, 0, 2, 3, 4, 4, 6, 7],                                  # 6 (wedge)
    [0, 1, 2, 3, 1, 4, 2, 3],                                  # 5
    [0, 1, 2, 2, 4, 4, 4, 4],                                  # 4 (moved to TETRA4)
    [0, 1, 2, 3, 0, 1, 2, 3],                                  # 4
    [0, 1, 2, 3, 4, 4, 6, 7],                                  # 7
    [0, 1, 1, 2, 3, 3, 4, 5],                                  # 6
    [0, 0, 0, 3, 4, 5, 6, 7],                                  # 7 (one tripled)
]


def test_detect_agrees_with_the_starter_collapsed_hexa_mark():
    """The plan's Step 3 requirement, made executable: "a Starter that marks an
    element collapsed while the Engine never sees the flag is a silent
    inconsistency", and the only way to rule that out is to run both.

    The group is built from the bricks the Starter ACTUALLY kept, which is the
    population the brick kernel ever sees — the 4-distinct ones have become
    /TETRA4 by then and have no brick-side answer to give.
    """
    conns = np.asarray(CONNECTIVITY_SWEEP, dtype=np.int64)
    kept, moved, count = _starter_report(conns)
    assert count > 0, "the sweep must contain a collapsed hexa"

    kept_rows = np.nonzero(np.isin(np.arange(1, len(conns) + 1),
                                   sorted(kept)))[0]
    g, _ = _brick_group(conns[kept_rows], init=False)
    degen = solid_degeneracy.detect(g)

    # 1. the Starter's own COLLAPSED HEXA tally is the number the Engine flags,
    #    and the rest of the kept bricks are the healthy ones it passed through
    assert int(degen.sum()) == count
    assert count < len(kept)

    # 2. `detect` is exactly "5, 6 or 7 distinct node ids" -- the Starter's rule
    for e_row, sweep_row in enumerate(kept_rows.tolist()):
        n_distinct = len(set(conns[sweep_row].tolist()))
        assert bool(degen[e_row]) is (n_distinct in (5, 6, 7))

    # 3. nothing the Starter moved to /TETRA4 escapes the brick-side count: it
    #    has repeated nodes, it just has no brick to answer for.
    tet = np.asarray([eid - 1 for eid in sorted(moved)], dtype=np.int64)
    tetra_group, _ = _brick_group(conns[tet], init=False)
    assert solid_degeneracy.degenes8(tetra_group).min() > 0


def test_degenes8_counts_repeated_nodes_the_way_the_fortran_does():
    """``degenes8`` is "for each node, is it repeated elsewhere, summed, then
    halved" — the count, not a boolean, and not a geometry test.  The halving
    is what makes a brick with one tripled node an ``IDEGE`` of 1, not 3/2."""
    conns = np.asarray(CONNECTIVITY_SWEEP, dtype=np.int64)
    g, _ = _brick_group(conns, init=False)
    for row, conn in enumerate(conns):
        slots = [conn.tolist().count(int(c)) for c in conn]
        assert solid_degeneracy.degenes8(g)[row] == \
            sum(1 for k in slots if k > 1) // 2


def test_degenes8_agrees_with_the_engine_solid_hexa8_lc_scale():
    """``solid_hexa8.init_group`` recomputes IDEGE inline to scale ``lc``.
    Two independent implementations of one count must not drift apart."""
    conns = np.asarray(CONNECTIVITY_SWEEP, dtype=np.int64)
    g, _ = _brick_group(conns)
    idege = solid_degeneracy.degenes8(g)
    expected = np.where(idege > 2, 3.0, np.where(idege > 1, 2.0, 1.0))
    assert np.allclose(g.state["lc_scale"], expected)


# ---------------------------------------------------------------------------
# 3. the degeneracy factors and the characteristic length they feed
# ---------------------------------------------------------------------------
def test_degeneracy_factors_are_finite_and_positive_on_a_collapsed_brick():
    g, model = _degenerate_hexa8_group(PLAN_SAMPLES["repeated_nodes"])
    aream, fac = solid_degeneracy.degeneracy_factors(g, model.x)
    degen = solid_degeneracy.detect(g)
    assert degen.all()
    assert np.all(np.isfinite(aream)) and np.all(aream > 0.0)
    assert np.all(np.isfinite(fac)) and np.all(fac > 0.0)


def test_degeneracy_factors_of_a_healthy_brick_are_inert():
    g, model = _unit_hexa8_group()
    aream, fac = solid_degeneracy.degeneracy_factors(g, model.x)
    assert not solid_degeneracy.detect(g).any()
    assert np.all(aream == EM20) and np.all(fac == 1.0)


def test_characteristic_length_survives_the_collapsed_brick():
    """The point of the FAC ladder: ``lc = V / A_max`` -> 0 as an element
    flattens, and a zero lc is a zero time step.  ``sdlen_dege`` replaces it
    with ``4 V / sqrt(AREAM)``, which does not."""
    g, model = _degenerate_hexa8_group(PLAN_SAMPLES["repeated_nodes"])
    lc = solid_degeneracy.characteristic_length(g, model.x)
    assert np.all(np.isfinite(lc)) and np.all(lc > 0.0)


def test_the_length_correction_actually_moves_the_number():
    """Not vacuous.  A 3-repeat wedge has ``IDEGE = 3``, the ladder gives
    ``FAC = 1/9``, and ``sqrt(1/9) = 1/3`` sits in the denominator of
    ``DELTAX = 4*V/sqrt(FAC*AREAM)`` — so the correction triples the length,
    which is exactly the ``lc_scale = 3`` ``solid_hexa8.init_group`` applies
    for the same count."""
    g, model = _degenerate_hexa8_group(PLAN_SAMPLES["repeated_nodes"])
    xe = model.x[g.conn]
    _dndx, vol = solid_hexa8._geometry(xe)
    plain = solid_hexa8._char_length(xe, vol)
    assert solid_degeneracy.degenes8(g)[0] == 3
    assert g.state["lc_scale"][0] == 3.0
    lc = solid_degeneracy.characteristic_length(g, model.x)
    assert lc[0] == pytest.approx(3.0 * plain[0], rel=1e-12)


def test_characteristic_length_is_the_plain_sdlen3_value_when_healthy():
    g, model = _unit_hexa8_group()
    xe = model.x[g.conn]
    _dndx, vol = solid_hexa8._geometry(xe)
    assert np.allclose(solid_degeneracy.characteristic_length(g, model.x),
                       solid_hexa8._char_length(xe, vol))


def test_idege8_amax_scan_keeps_the_first_of_the_largest_faces():
    """``IDEGE8`` only ever RAISES ``AMAX``::

        IF (A > AMAX) THEN ; IT = IDE ; AMAX = A ; END IF

    with ``AMAX`` seeded at ``EM20`` by ``sdlen_dege``.  So face 1 always wins
    the first round and a later face displaces it only when strictly larger —
    ties go to the earlier face.  The module's scan is vectorized ``np.where``;
    this re-derives it sequentially, in the Fortran's own order.
    """
    for name in sorted(PLAN_SAMPLES):
        g, model = _degenerate_hexa8_group(PLAN_SAMPLES[name])
        xq = model.x[g.conn][0][solid_degeneracy._HEXA_FACES]      # (6, 4, 3)
        a = solid_degeneracy.idege8_area(xq)
        amax, it = EM20, 0
        for j in range(len(xq)):
            if a[j] > amax:
                amax, it = a[j], solid_degeneracy.idege8_flag(xq[j:j + 1])[0]
        fac = 1.0 / 9.0 if solid_degeneracy.degenes8(g)[0] > 2 else 0.25
        expected = fac * amax if it == 0 else amax

        aream, got_fac = solid_degeneracy.degeneracy_factors(g, model.x)
        assert got_fac[0] == pytest.approx(fac)
        assert aream[0] == pytest.approx(expected, rel=1e-12)
        # the scan never invents an area larger than the biggest face
        assert aream[0] <= a.max() + 1e-12 * a.max()


def test_nodedege_appends_only_positions_it_has_not_seen():
    xyz = np.zeros((3, 4), dtype=float)          # Fortran XYZ(3,4)
    n = 1
    for p in ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
        n = solid_degeneracy.nodedege(xyz, p, n)
    assert n == 3
    assert np.allclose(xyz[:, :3].T, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0]])


def test_deges4v_is_the_tetra_of_the_first_four_distinct_positions():
    coords = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
              [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    xe = np.asarray(coords, dtype=float)
    assert np.allclose(solid_degeneracy.deges4v(xe), 1.0 / 6.0)
    # ... and a healthy brick has no distinct-position tetra to fall back on
    assert solid_degeneracy.deges4v(0.5 * (solid_hexa8._XI + 1.0)) == 0.0


# ---------------------------------------------------------------------------
# 4. the dim_tshedg / ind_tshedg assembly mesh
# ---------------------------------------------------------------------------
def test_assemble_isotropic_pairs_the_hexa8_unslave_directions():
    """``ind_tshedg``'s ``JHBE == 15`` branch: node ``i`` against node
    ``i+4``, for ``i = 0..3``, in that order."""
    g, _ = _unit_hexa8_group()
    ienunl = solid_degeneracy.assemble_isotropic(g, [0])
    assert ienunl.shape == (2, 4)
    assert np.array_equal(ienunl, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))


def test_assemble_isotropic_lets_each_node_belong_to_one_edge():
    """``ITAG``: a direction is emitted only when BOTH its nodes are still
    unclaimed, so the result is a MATCHING — the graph walk downstream cannot
    see the same degenerate direction twice."""
    x = 0.5 * (solid_hexa8._XI + 1.0)
    g, _ = _group(x, np.stack([np.arange(8), np.arange(8)]))
    ienunl = solid_degeneracy.assemble_isotropic(g, [0, 1])
    # element 1 shares every node with element 0, so it contributes nothing
    assert ienunl.shape == (2, 4)
    ends = ienunl.ravel().tolist()
    assert len(ends) == len(set(ends))


def test_assemble_isotropic_of_separate_bricks_touches_each_node_once():
    g, _ = _brick_group(np.arange(24, dtype=np.int64).reshape(3, 8))
    ienunl = solid_degeneracy.assemble_isotropic(g, [0, 1, 2])
    assert ienunl.shape == (2, 12)
    ends = ienunl.ravel().tolist()
    assert len(ends) == len(set(ends)) == 24


def test_assemble_isotropic_selects_only_the_named_elements():
    g, _ = _brick_group(np.arange(24, dtype=np.int64).reshape(3, 8),
                        init=False)
    only_second = solid_degeneracy.assemble_isotropic(g, [1])
    assert only_second.shape == (2, 4)
    assert set(only_second.ravel().tolist()) == set(range(8, 16))


def test_assemble_isotropic_keeps_the_self_loop_upstream_produces():
    """``ind_tshedg`` has no ``N1 /= N2`` guard: a brick collapsed so that a
    direction's two ends are the same node still emits the edge, burns one tag
    slot on it and moves on.  The plan's ``flat_face`` sample is exactly that
    shape, and upstream's own output is the self-loop -- not a fixed-up one."""
    g, _ = _degenerate_hexa8_group(PLAN_SAMPLES["flat_face"])
    conn = g.conn[0]
    assert conn[3] == conn[7]                       # the direction (3, 7)
    ienunl = solid_degeneracy.assemble_isotropic(g, [0])
    assert np.array_equal(ienunl, np.array([[0, 3], [1, 3]]))
    assert ienunl[0, -1] == ienunl[1, -1]           # the self-loop survives


def test_assemble_isotropic_of_an_empty_selection_is_empty():
    g, _ = _unit_hexa8_group()
    assert solid_degeneracy.assemble_isotropic(g, []).shape == (2, 0)


def test_assemble_isotropic_of_an_entire_group_is_the_same_as_every_element():
    g, _ = _brick_group(np.arange(24, dtype=np.int64).reshape(3, 8), init=False)
    assert np.array_equal(solid_degeneracy.assemble_isotropic(g),
                          solid_degeneracy.assemble_isotropic(g, [0, 1, 2]))


def test_tagged_ids_compacts_the_tshcdcom_mask():
    g, _ = _brick_group(np.arange(24, dtype=np.int64).reshape(3, 8), init=False)
    ienunl = solid_degeneracy.assemble_isotropic(g, [1])
    mask = solid_degeneracy.tag_elements(ienunl, g.conn)
    assert mask.tolist() == [False, True, False]
    assert solid_degeneracy.tagged_ids(mask, g.ids).tolist() == [2]
