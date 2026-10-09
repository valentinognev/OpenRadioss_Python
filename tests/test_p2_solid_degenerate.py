"""Task P2.3 — the solid → 2-node degeneration paths (the ``sfor_n2s4`` family).

WHAT THIS FILE PINS DOWN
------------------------
Four upstream routines collapse a solid onto a line and keep it in the model as
a 2-node segment:

* ``sfor_n2s4.F``    (hexa8 / wedge6 self-contact) calls ``ssort_n4.F``;
* ``sfor_ns2s4.F90`` (the striking node against that quad);
* ``sfor_4n2s4.F90`` (four nodes striking a quad, driving the above);
* ``sfor_n2stria.F`` (tetra4, the ``sfor_n2s3`` entry point).

The load-bearing detail is ``ssort_n4``: it is a **node re-ordering**.  It does
not permute ``X1..X4`` — it *classifies* which pair has merged, and the code it
writes (``2``, ``3``, ``4``, ``5``, ``6``, or ``0`` for a quad already collapsed
onto a line) is what every downstream ``SELECT CASE`` re-orders the nodes by.
Take the wrong branch and the segment comes out with its ends swapped, which the
plan calls an infinite energy engine.  So the ladder here is a transcription of
the Fortran, branch for branch and in the Fortran's own order, and the tests
below pin each branch against the one a re-derived rule would pick.

NODE NUMBERING
--------------
Every node number in this module and in these tests is **1-based**, as it is in
the Fortran and in a Radioss deck: ``X1`` is ``nodes[0]``.  The helpers below
take 1-based pairs so the tests read like the routine they transcribe.

THE CONTRACT THE BRIEF STATES
-----------------------------
A solid collapsed onto a line becomes a 2-node segment with the **same total
mass** and the **same centroid**.  Both are checked as a first-moment identity
(``sum m_k x_k``), so a port that fudges either number fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.elements import solid_degenerate as sd


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
def _merge(quad, *pairs):
    """``quad`` with each 1-based pair ``(i, j)`` made coincident.

    Node ``j`` is overwritten with node ``i``, so a chain like
    ``((1, 2), (2, 4))`` leaves nodes 1, 2 and 4 all in one place.
    """
    out = np.asarray(quad, dtype=np.float64).copy()
    for i, j in pairs:
        out[j - 1] = out[i - 1]
    return out


def _line_quad4(length: float = 2.0):
    """A quad4 on the segment 0 → L: 1 and 4 at one end, 2 and 3 at the other."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([length, 0.0, 0.0])
    return np.array([a, b, b, a])


def _line_wedge6(length: float = 2.0):
    """A wedge collapsed the same way — bottom quad 1-2-3-4, 5 above 1, 6 above 2."""
    nodes = _line_quad4(length)
    return np.vstack([nodes, nodes[[0, 1]]])


def _line_hexa8(length: float = 2.0):
    """A hexa whose every node is on the segment — bottom 1-4, top 5-8."""
    nodes = _line_quad4(length)
    return np.vstack([nodes, nodes[[0, 1, 1, 0]]])


def _intact_quad4(size: float = 1.0):
    """A unit square in z = 0, counter-clockwise 1-2-3-4."""
    return np.array([
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, size, 0.0],
        [0.0, size, 0.0],
    ])


def _intact_hexa8():
    return np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])


def _ssort(quad, ifc=0, marge=0.0, stif=1.0):
    """``ssort_n4`` on a single quad, with the scalar knobs unboxed.

    The striking point sits a unit off the unit square's plane, so MARGE in
    these tests is a real distance and a quad can be inside or outside it.
    """
    quad = np.asarray(quad, dtype=np.float64)[None, ...]
    return sd.ssort_n4(quad, np.array([[0.0, 0.0, 1.0]]),
                       ifc=np.array([ifc]), marge=np.array([marge]),
                       stif=np.array([stif]))


def _element_moment(nodes, total_mass):
    """Mass and first moment of the element, uniformly lumped over its nodes."""
    mass = np.full(len(nodes), total_mass / len(nodes))
    return mass.sum(), (mass[:, None] * nodes).sum(axis=0)


def _segment_moment(nodes, reduction, total_mass):
    """Mass and first moment of the 2-node segment the reduction describes."""
    mass = total_mass * reduction.weights
    return mass.sum(), (mass[:, None] * nodes[reduction.ids - 1]).sum(axis=0)


# ---------------------------------------------------------------------------
# 1. ssort_n4 — the transcription of the sort itself
# ---------------------------------------------------------------------------
def test_ssort_n4_leaves_a_positive_caller_code_alone():
    """``IF (IFC1(I)>0) CYCLE`` gates BOTH loops (ssort_n4.F:65 and :84)."""
    quad = _merge(_line_quad4(), (1, 4), (2, 3))
    assert _ssort(quad, ifc=7, marge=2.0).tolist() == [7]


def test_ssort_n4_flattens_a_quad_whose_node_3_sits_within_marge_of_the_plane():
    """First loop: node 3 closer to the striking plane than MARGE is code 2."""
    assert _ssort(_intact_quad4(), marge=2.0).tolist() == [sd.IFC_FLAT]


def test_ssort_n4_does_not_flatten_a_quad_whose_stiffness_is_zero():
    """``AND.STIF(I)>ZERO`` is a second, independent gate on code 2."""
    assert _ssort(_intact_quad4(), marge=2.0, stif=0.0).tolist() == [0]


def test_ssort_n4_does_not_flatten_a_quad_beyond_marge():
    assert _ssort(_intact_quad4(), marge=0.5).tolist() == [0]


@pytest.mark.parametrize("pair, code", [
    ((4, 3), sd.IFC_MERGE_34),
    ((4, 1), sd.IFC_MERGE_14),
    ((3, 2), sd.IFC_MERGE_23),
    ((2, 1), sd.IFC_MERGE_12),
])
def test_ssort_n4_geometry_ladder_reports_the_merged_pair(pair, code):
    """The four single-merge branches, each with the code the Fortran writes."""
    assert _ssort(_merge(_intact_quad4(), pair)).tolist() == [code]


@pytest.mark.parametrize("pairs, code", [
    (((4, 3), (3, 2)), sd.IFC_MERGE_34),
    (((1, 2), (2, 4)), sd.IFC_MERGE_12),
])
def test_ssort_n4_first_matching_branch_wins(pairs, code):
    """The ladder CYCLEs on its first hit; a later merge cannot overwrite it.

    ``X4==X3`` is tested before ``X2==X1`` and before ``X4==X1``, so a quad with
    several merged pairs takes the earliest branch — this is exactly the
    re-ordering that must not be re-derived.
    """
    assert _ssort(_merge(_intact_quad4(), *pairs)).tolist() == [code]


@pytest.mark.parametrize("pairs", [
    ((4, 1), (3, 2)),   # 1=4 and 2=3 -> the line runs 1 -> 2
    ((4, 3), (2, 1)),   # 1=2 and 3=4 -> the line runs 1 -> 3
])
def test_ssort_n4_writes_zero_for_a_quad_already_on_a_line(pairs):
    assert _ssort(_merge(_intact_quad4(), *pairs)).tolist() == [sd.IFC_MERGE_LINE]


def test_ssort_n4_never_sends_a_flattened_quad_through_the_pair_ladder():
    """``IF (IFC1(I)==0) CYCLE`` — code 2 is decided, not refined.

    Nodes 1 and 2 are merged *and* the whole face is lifted inside MARGE of the
    striking plane.  The pair ladder would report 6; loop 1 claims it first and
    the answer must be 2.
    """
    lifted = _merge(_intact_quad4(), (2, 1)) + np.array([0.0, 0.0, 0.4])
    assert _ssort(lifted, marge=1.0).tolist() == [sd.IFC_FLAT]
    # the same quad outside MARGE falls through to the ladder and reports 6
    assert _ssort(lifted, marge=0.1).tolist() == [sd.IFC_MERGE_12]


# ---------------------------------------------------------------------------
# 2. the surviving pair — which two nodes the line keeps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pairs, expected", [
    (((4, 1), (3, 2)), (1, 2)),
    (((4, 3), (2, 1)), (1, 3)),
])
def test_the_line_pair_is_the_one_the_ladder_collapsed_onto(pairs, expected):
    assert sd.line_pair(_merge(_intact_quad4(), *pairs)) == expected


def test_a_quad_with_every_node_in_one_place_collapses_onto_a_zero_length_line():
    """The ladder still names a pair; there is simply no length left.

    ``line_pair`` reports the sort's answer and nothing more — refusing a
    degenerate segment belongs to the reduction, which is where the length is
    known.
    """
    quad = _intact_quad4().copy()
    quad[:] = quad[0]
    assert sd.line_pair(quad) == (1, 3)
    assert sd.four_to_two(quad) is None


def test_an_intact_quad_has_no_line_at_all():
    assert sd.line_pair(_intact_quad4()) is None


def test_a_quad_merged_on_one_side_only_is_not_a_line():
    assert sd.line_pair(_merge(_intact_quad4(), (2, 1))) is None


# ---------------------------------------------------------------------------
# 3. the reductions — mass and centroid
# ---------------------------------------------------------------------------
_REDUCTIONS = [
    (sd.four_to_two, _line_quad4()),
    (sd.six_to_two, _line_wedge6()),
    (sd.eight_to_two, _line_hexa8()),
]


@pytest.mark.parametrize("reduce, nodes", _REDUCTIONS)
def test_a_solid_collapsed_onto_a_line_becomes_a_two_node_segment(reduce, nodes):
    result = reduce(nodes)
    assert result is not None
    assert result.ids.shape == (2,)
    assert result.weights.shape == (2,)
    # the ladder's own pair, in its own order — a swapped pair is the failure
    assert tuple(result.ids) == (1, 2)
    assert result.code == sd.IFC_MERGE_LINE


@pytest.mark.parametrize("reduce, nodes", _REDUCTIONS)
def test_the_reduction_keeps_the_total_mass_and_the_centroid(reduce, nodes):
    total = 7.5
    result = reduce(nodes)
    mass_before, moment_before = _element_moment(nodes, total)
    mass_after, moment_after = _segment_moment(nodes, result, total)

    assert mass_after == pytest.approx(mass_before, rel=1e-14)
    assert np.allclose(moment_after, moment_before, rtol=0.0, atol=1e-14)


def test_a_uniform_collapse_halves_the_mass_between_the_two_ends():
    """Four of the eight hexa nodes sit on each end, so each end carries half."""
    assert np.allclose(sd.eight_to_two(_line_hexa8()).weights,
                       [0.5, 0.5], rtol=0.0, atol=1e-15)


def test_the_weights_follow_a_collapse_whose_nodes_are_not_evenly_split():
    """Every node is on the line, but the top face stops short of the far end.

    A hand-written half-and-half split puts the segment's centroid at x = 1.0;
    this hexa's own centroid is x = 0.75.  The weights have to be measured from
    the nodes, not assumed from the node count.
    """
    nodes = _line_hexa8()
    nodes[5] = [1.0, 0.0, 0.0]         # nodes 6 and 7 stop at the midpoint
    nodes[6] = [1.0, 0.0, 0.0]
    result = sd.eight_to_two(nodes)
    assert result is not None
    assert result.weights[1] == pytest.approx(0.375, rel=1e-14)

    total = 3.0
    _, moment_before = _element_moment(nodes, total)
    _, moment_after = _segment_moment(nodes, result, total)
    assert np.allclose(moment_after, moment_before, rtol=0.0, atol=1e-14)


def test_a_solid_whose_nodes_leave_the_line_is_not_reduced():
    """A flat face is not a line collapse: node 6 sits off the segment."""
    nodes = _line_hexa8()
    nodes[5] = [2.0, 0.0, 1.0]
    assert sd.eight_to_two(nodes) is None


def test_an_intact_solid_is_not_reduced():
    hexa = _intact_hexa8()
    assert sd.eight_to_two(hexa) is None
    assert sd.six_to_two(hexa[:6]) is None
    assert sd.four_to_two(hexa[:4]) is None


# ---------------------------------------------------------------------------
# 4. four_to_four_striking — sfor_4n2s4.F90 + sfor_ns2s4.F90
#
# The striking points sit *below* the unit square (z < 0): the quad's normal from
# R x S points at +z, and upstream computes PENE as
# ``MAX(0, PENMIN - (XB - XI).N)``, so a node registers only while it is less
# than PENMIN inside the surface.  A node deeper than PENMIN contributes nothing,
# which is how the three filler nodes below are kept out of the way.
# ---------------------------------------------------------------------------
_INSIDE = -0.5
_AWAY = [[9.0, 9.0, -5.0], [-9.0, 4.0, -5.0], [4.0, -9.0, -5.0]]


def _strike(quad, point, **kw):
    """Strike ``quad`` with ``point`` as node 1 and three deep nodes after it."""
    opts = dict(marge=10.0, penmin=1.0, stif0=1.0, ll=1.0)
    opts.update(kw)
    striking = np.array([list(point)] + _AWAY, dtype=np.float64)
    return sd.four_to_four_striking(quad, striking, **opts)


@pytest.mark.parametrize("pair, code", [
    ((2, 1), 12),
    ((3, 2), 23),
    ((4, 3), 34),
    ((4, 1), 14),
])
def test_itgsub_names_the_merged_pair_of_the_striking_quad(pair, code):
    assert sd.itgsub(_merge(_intact_quad4(), pair), ifc=1, ll=1.0) == code


def test_itgsub_is_minus_one_when_the_quad_is_already_a_line():
    assert sd.itgsub(_line_quad4(), ifc=1, ll=1.0) == sd.ITGSUB_LINE
    # a vanishing characteristic length disqualifies it as well
    assert sd.itgsub(_intact_quad4(), ifc=1, ll=0.0) == sd.ITGSUB_LINE
    # and so does a caller code of zero
    assert sd.itgsub(_intact_quad4(), ifc=0, ll=1.0) == sd.ITGSUB_LINE


def test_itgsub_is_zero_for_an_intact_quad():
    assert sd.itgsub(_intact_quad4(), ifc=1, ll=1.0) == 0


@pytest.mark.parametrize("code, triangle", [
    (1, (1, 2, 4)),
    (2, (2, 3, 1)),
    (3, (3, 4, 2)),
    (4, (4, 1, 3)),
    (12, (3, 4, 1)),
    (23, (4, 1, 2)),
    (34, (1, 2, 3)),
    (14, (2, 3, 4)),
])
def test_each_itgsub_code_reorders_the_quad_into_its_own_triangle(code, triangle):
    """The ``SELECT CASE`` of sfor_ns2s4.F90:119 — transcribed, not derived.

    These triples are the ``(XB, XC, XA)`` order the Fortran writes, read off the
    routine's own line-119 comment.  A re-derived rule gets the same *set* of
    three nodes in several cases and the wrong *order* in all of them, and the
    order is what fixes the barycentric weights.
    """
    assert tuple(sd.triangle_nodes(code)) == triangle


def test_each_striking_node_is_resolved_in_upstreams_order():
    """sfor_4n2s4 runs ns = n1, n2, n3, n4 as four independent passes."""
    quad = _intact_quad4()
    striking = np.array([[0.25, 0.25, _INSIDE], [0.5, 0.5, _INSIDE],
                         [0.75, 0.25, _INSIDE], [0.5, 0.5, _INSIDE]])
    out = sd.four_to_four_striking(quad, striking, marge=10.0, penmin=1.0,
                                   stif0=1.0, ll=1.0)
    assert out.struck.tolist() == [True, True, True, True]
    assert out.index.tolist() == [1, 2, 3, 4]
    assert out.itg.tolist() == [1, 2, 3, 4]


def test_a_striking_point_takes_the_triangle_of_its_own_node_number():
    """``IF (ITGSUB(I)==0) ITGSUB(I)=K`` — on an intact quad the pass number
    *is* the triangle code, so node 3 uses 3-4-2."""
    quad = _intact_quad4()
    striking = np.array(_AWAY[:2] + [[0.75, 0.25, _INSIDE], _AWAY[2]])
    out = sd.four_to_four_striking(quad, striking, marge=10.0, penmin=1.0,
                                   stif0=1.0, ll=1.0)
    assert out.itgsub == 0
    assert out.itg[2] == 3
    assert tuple(sd.triangle_nodes(out.itg[2])) == (3, 4, 2)
    assert out.struck[2]


def test_a_merged_quad_keeps_its_own_code_for_every_striking_node():
    """A quad that already lost a node uses its merge code, not the pass number."""
    quad = _merge(_intact_quad4(), (2, 1))            # 1 and 2 merged -> itgsub 12
    out = _strike(quad, (0.25, 0.25, _INSIDE))
    assert out.itgsub == 12
    assert out.itg[0] == 12
    assert tuple(sd.triangle_nodes(out.itg[0])) == (3, 4, 1)
    assert out.struck[0]


def test_the_weights_are_the_barycentric_coordinates_of_the_striking_point():
    """``hj1 = LB``, ``hj2 = LC``, ``hj3 = 0``, ``hj4 = LA`` for code 1.

    Against the triangle 1-2-4 the projected point (0.25, 0.25, 0) has
    ``LA = 0.25``, ``LB = 0.5`` and ``LC = 0.25``.  The weights must sum to one
    and must reconstruct the point, which is the invariant that makes a wrong
    ``SELECT CASE`` branch fail loudly instead of quietly.
    """
    quad = _intact_quad4()
    out = _strike(quad, (0.25, 0.25, _INSIDE))
    w = out.weights[0]
    assert w == pytest.approx([0.5, 0.25, 0.0, 0.25], abs=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert (w >= -1e-15).all()
    assert np.allclose(w @ quad, [0.25, 0.25, 0.0], atol=1e-12)


def test_the_node_a_triangle_drops_gets_a_zero_weight():
    """Code 1 uses the triangle 1-2-4, so quad node 3's weight is zero."""
    out = _strike(_intact_quad4(), (0.25, 0.25, _INSIDE))
    assert out.weights[0][2] == 0.0


def test_a_striking_node_beyond_marge_does_not_strike():
    out = _strike(_intact_quad4(), (0.5, 0.5, 5.0), marge=1e-6)
    assert out.struck.tolist() == [False, False, False, False]


def test_a_quad_with_no_stiffness_never_strikes():
    out = _strike(_intact_quad4(), (0.5, 0.5, _INSIDE), stif0=0.0)
    assert out.struck.tolist() == [False, False, False, False]


def test_a_quad_already_collapsed_onto_a_line_never_strikes():
    out = _strike(_line_quad4(), (0.5, 0.0, _INSIDE))
    assert out.itgsub == sd.ITGSUB_LINE
    assert out.struck.tolist() == [False, False, False, False]


def test_a_striking_node_coincident_with_its_quad_node_never_strikes():
    """``IF (DMIN==ZERO) CYCLE`` — upstream skips an exactly merged node."""
    quad = _intact_quad4()
    striking = np.array([quad[0]] + _AWAY)
    out = sd.four_to_four_striking(quad, striking, marge=10.0, penmin=1.0,
                                   stif0=1.0, ll=1.0)
    assert out.struck.tolist() == [False, False, False, False]


def test_weights_of_a_striking_point_outside_the_triangle_stay_on_the_simplex():
    """The LA/LB/LC clamp ladder of sfor_ns2s4:259-290.

    The point is far past node 4 along y, so every barycentric coordinate but
    one goes negative before the clamp; afterwards the weights sum to one, never
    go negative, and sit on the corner the point is nearest to.
    """
    out = _strike(_intact_quad4(), (0.0, 5.0, _INSIDE))
    assert out.struck[0]
    w = out.weights[0]
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert (w >= -1e-15).all()
    assert w[3] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("depth, pene", [(-0.25, 0.75), (-0.5, 0.5), (-0.75, 0.25)])
def test_the_penetration_is_what_is_left_of_penmin_after_the_depth(depth, pene):
    out = _strike(_intact_quad4(), (0.5, 0.5, depth), penmin=1.0)
    assert out.pene[0] == pytest.approx(pene, abs=1e-12)
    assert out.struck[0]


def test_a_striking_point_deeper_than_penmin_registers_no_penetration():
    """``PENE = MAX(0, PENMIN - (XB - XI).N)``: the depth is counted only up to
    PENMIN, so a node further inside than PENMIN contributes nothing."""
    out = _strike(_intact_quad4(), (0.5, 0.5, -2.0))
    assert out.pene[0] == 0.0
    assert not out.struck[0]
