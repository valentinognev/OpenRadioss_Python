"""Task P2.10 — the ``solide8s`` co-rotational frame (``crframe_imp`` family).

What the tests hold down
------------------------
* the plan's own sample — an orthonormal, right-handed ``R`` per element;
* **the trap**.  The plan's sample proves orthonormality and nothing else, and
  an orthonormal frame can still be the wrong one.  ``crframe_imp`` returns the
  polar rotation of ``F`` (stage 1) and ``crtrans_imp`` + ``transk`` build and
  apply the 24x24 projection that accounts for that rotation moving with the
  nodes (stage 2).  So, beyond the plan:

  - the frame is pinned against hand-computed values (a 90 degree rotation,
    a simple shear whose polar rotation is ``atan(gamma/2)``) that a
    Gram-Schmidt frame of the element edges gets visibly wrong;
  - stage 2 is pinned against the one property that separates it from
    "rotate by ``R^T`` and stop": a rigid-body displacement field maps to
    **zero**.  ``blockdiag(R^T)`` — what porting only ``crframe_imp`` would
    give — maps it to O(1);
  - stage 2 is the exact derivative of ``R^T x`` (finite differences);
  - the same must hold for ``transk`` (stiffness) and ``s8sfint3_crimp``
    (forces): six rigid modes are in the null space, not three.
* **Fortran wins over the plan's orthonormality test.**  ``crframe_imp`` clamps
  ``B = MAX(B, 0)`` inside its closed-form stretch inversion; for a generic
  ``F`` (three distinct stretches) that clamp is active and ``R`` is *not*
  exactly orthogonal — about ``1e-3`` for the shear in the golden case below.
  The plan's ``1e-12`` holds where the clamp is inactive (undeformed, rigid,
  and any ``F`` with a repeated stretch) and no further.  The golden numbers
  are the output of the **unmodified** upstream ``crframe_imp.F`` /
  ``crtrans_imp.F`` compiled with gfortran against a stub of
  ``constant_mod`` (same constant values) and run on the case below; they pin
  the clamp so a future "fix" to exact polar decomposition cannot slip in.
* the reference state routines (``s8sav3_imp``, ``s8xref_imp``, ``getuloc``)
  and ``srcoor3_imp``'s ``ISMSTR`` coordinate change;
* the reachability finding: no ``crframe``/``crtrans`` deck keyword exists
  upstream, so none exists here.

Fortran read before any of this was written (``$OR_SRC/engine/source/
elements/solid/solide8s/``): ``crframe_imp.F``, ``crtrans_imp.F``,
``transk.F``, ``getuloc.F``, ``s8xref_imp.F``, ``s8sav3_imp.F``,
``srcoor3_imp.F``, ``srcoork_imp.F``, ``s8sfint3_crimp.F``, ``s8sforc3.F``,
``s8ske3.F``; and, for the absence of a reader, ``starter/`` and ``reader/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.elements import solid_crimp
from pyradioss.model.model import ElementGroup, Model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
#: node order of the brick: 1-2-3-4 bottom, 5-6-7-8 top.
_XI = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)


def _rot(axis, angle):
    """Rodrigues rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _group_of(refs, curs):
    """One brick per element, node ``8*e + k``; ``x0`` is the reference."""
    refs = [np.asarray(r, dtype=float) for r in refs]
    curs = [np.asarray(c, dtype=float) for c in curs]
    n = len(refs)
    group = ElementGroup(ids=np.arange(1, n + 1, dtype=np.int64),
                         conn=np.arange(8 * n, dtype=np.int64).reshape(n, 8),
                         part=np.zeros(n, dtype=np.int64))
    group.state["x0"] = np.concatenate(refs)
    group.state["x"] = np.concatenate(curs)
    return group


#: states in which crframe_imp's closed form is exact (the B clamp is inactive):
#: undeformed, rigid motion of a skewed element, and F with a repeated stretch.
_SKEW = np.array([[1.2, 0.1, 0.0], [0.05, 0.9, 0.1], [0.0, 0.1, 1.1]])
_STATES = {
    "undeformed": (_XI, _XI),
    "rigid": (_XI, _XI @ _rot([1, 2, 3], 0.7).T + np.array([1.0, 2.0, 3.0])),
    "skewed_rigid": (_XI @ _SKEW.T,
                     (_XI @ _SKEW.T) @ _rot([-1, 1, 2], 0.4).T + 0.5),
    "uniaxial": (_XI, (_XI * np.array([2.0, 1.0, 1.0]))
                 @ _rot([0, 0, 1], 0.5).T),
}


def _hexa8_group():
    names = ("undeformed", "rigid", "uniaxial")
    return _group_of([_STATES[k][0] for k in names],
                     [_STATES[k][1] for k in names])


def _frame_and_trm(ref, cur):
    sav = solid_crimp.save_reference(np.asarray(ref)[None])
    R, invj = solid_crimp.crframe(sav, np.asarray(cur)[None])
    TRM, V = solid_crimp.corotational_transform(np.asarray(cur)[None], R, invj)
    return sav, R, invj, TRM[0], V[0]


def _block_diag_rt(R):
    """What porting ``crframe_imp`` alone would use: ``blockdiag(R^T)``."""
    out = np.zeros((24, 24))
    for j in range(8):
        out[3 * j:3 * j + 3, 3 * j:3 * j + 3] = R.T
    return out


#: the general deformation of the golden case (three distinct stretches).
_F_GOLDEN = np.array([[1.1, 0.3, -0.1], [0.05, 0.9, 0.2], [0.1, -0.15, 1.2]])

#: Output of the UNMODIFIED upstream crframe_imp.F + crtrans_imp.F (gfortran,
#: constant_mod stubbed with the same values) on ref = unit cube,
#: cur = ref @ _F_GOLDEN.T.
_GOLDEN_R = np.array([
    [0.9873878154264264, 0.1313535204309503, -0.0882593254199961],
    [-0.11427342673294133, 0.977627761884707, 0.1765186508399922],
    [0.10947196090041826, -0.16420794135062738, 0.9803221161126708]])
_GOLDEN_V1 = np.array([
    [-0.008386639403950251, 0.11098479994894082, -0.11535193999338336],
    [-0.09629832874711898, 0.004173615212157327, 0.10252272134966309],
    [0.12477525116412116, -0.1273979856566581, 0.004213024191792925]])
_GOLDEN_TRM_ROW4 = np.array([
    0.01989838138065473, -0.024088847735006032, 0.004909862210452582,
    1.0127875219682618, -0.09542621860506012, 0.10958311194862558,
    -0.01753913011484801, 0.023816822163375256, -0.004648474640923745,
    -0.02304045527602858, -0.019119233699511975, 0.000150236521321526,
    0.01753913011484801, -0.023816822163375256, 0.004648474640923745,
    0.02304045527602858, 0.019119233699511975, -0.000150236521321526,
    -0.01989838138065473, 0.024088847735006032, -0.004909862210452582,
    -0.0253997065418353, -0.018847208127881203, -0.00011115104820731049])
_GOLDEN_TRM_FRO = 4.963494976088169


# ---------------------------------------------------------------------------
# 1. the plan's test
# ---------------------------------------------------------------------------
def test_crimp_frame_is_a_orthonormal_right_handed_basis():
    R = solid_crimp.imperfection_frame(_hexa8_group(), reference="undeformed")
    assert R.shape == (3, 3, 3)
    for e in range(R.shape[0]):
        assert np.allclose(R[e] @ R[e].T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R[e]) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. the frame is pinned to known geometry, not just to orthonormality
# ---------------------------------------------------------------------------
def test_frame_of_an_undeformed_element_is_the_identity():
    R = solid_crimp.imperfection_frame(_group_of([_XI], [_XI]))
    assert np.allclose(R[0], np.eye(3), atol=1e-14)


def test_frame_of_a_quarter_turn_is_that_quarter_turn():
    """Hand-computed: a 90 degree turn about z sends x -> y and y -> -x, so
    the local axes (the columns of R) are (0,1,0), (-1,0,0), (0,0,1)."""
    cur = _XI @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]]).T
    R = solid_crimp.imperfection_frame(_group_of([_XI], [cur]))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R[0], expected, atol=1e-12)


def test_frame_removes_the_stretch_and_keeps_only_the_turn():
    """Hand-computed: F = Rz(theta) diag(2,1,1) has polar rotation Rz(theta)
    whatever the stretch is — the frame follows the material, not the edges'
    lengths."""
    theta = 0.5
    cur = (_XI * np.array([2.0, 1.0, 1.0])) @ _rot([0, 0, 1], theta).T
    R = solid_crimp.imperfection_frame(_group_of([_XI], [cur]))
    c, s = np.cos(theta), np.sin(theta)
    assert np.allclose(R[0], [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-12)


def test_frame_is_the_polar_rotation_not_a_gram_schmidt_of_the_edges():
    """Hand-computed.  Simple shear ``x' = x + g y``: the symmetric part of the
    motion is a stretch, the antisymmetric part a turn of ``atan(g/2)``, so
    ``R = [[c, s], [-s, c]]`` with ``tan(theta) = g/2``.  A Gram-Schmidt frame
    of the sheared edges keeps the first edge direction (x -> x) and reports no
    turn at all — orthonormal, right-handed, and wrong by ``g/2``."""
    g = 0.05
    F = np.array([[1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cur = _XI @ F.T
    R = solid_crimp.imperfection_frame(_group_of([_XI], [cur]))[0]

    theta = np.arctan(g / 2.0)
    expected = np.array([[np.cos(theta), np.sin(theta), 0.0],
                         [-np.sin(theta), np.cos(theta), 0.0],
                         [0.0, 0.0, 1.0]])
    # crframe_imp's closed form is exact only to ~|g|^3 here (see the clamp
    # test below), so the tolerance is 1e-5, three orders under the signal.
    assert np.allclose(R, expected, atol=1e-5)

    # the edge-based alternative: Gram-Schmidt of the columns of F keeps the
    # first edge direction and reports no turn at all.
    q, r = np.linalg.qr(F)
    gs = q * np.sign(np.diag(r))
    assert np.abs(gs - np.eye(3)).max() < 1e-12
    assert np.abs(R - gs).max() > 0.02                      # g/2 = 0.025


def test_frame_matches_the_unmodified_fortran_on_a_general_deformation():
    cur = _XI @ _F_GOLDEN.T
    R = solid_crimp.imperfection_frame(_group_of([_XI], [cur]))[0]
    assert np.allclose(R, _GOLDEN_R, atol=1e-12)


def test_the_closed_form_is_not_exactly_orthogonal_and_that_is_upstream():
    """``B = MAX(B,ZERO)`` is active for a generic F, so upstream's R carries a
    residual.  Reproducing it is the port; an exact polar decomposition would
    be a different (better) element and would no longer match the oracle."""
    R = solid_crimp.imperfection_frame(
        _group_of([_XI], [_XI @ _F_GOLDEN.T]))[0]
    resid = np.abs(R @ R.T - np.eye(3)).max()
    assert 1e-5 < resid < 1e-1
    assert abs(np.linalg.det(R) - 1.0) > 1e-5


def test_frame_is_invariant_to_a_rigid_motion_of_the_whole_element():
    """Moving the element rigidly by (Q, t) rotates the frame by Q and nothing
    else: ``R(Q x + t) = Q R(x)`` — checked on a deformed reference."""
    ref = _XI @ _SKEW.T
    Q = _rot([0.3, -1.0, 0.5], 1.1)
    base = ref @ _rot([0, 0, 1], 0.2).T
    moved = base @ Q.T + np.array([5.0, -3.0, 2.0])
    R0 = solid_crimp.imperfection_frame(_group_of([ref], [base]))[0]
    R1 = solid_crimp.imperfection_frame(_group_of([ref], [moved]))[0]
    assert np.allclose(R1, Q @ R0, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. stage 2: crtrans_imp
# ---------------------------------------------------------------------------
def test_trm_matches_the_unmodified_fortran_on_a_general_deformation():
    _, R, _, TRM, V = _frame_and_trm(_XI, _XI @ _F_GOLDEN.T)
    assert np.allclose(R[0], _GOLDEN_R, atol=1e-12)
    assert np.allclose(V[0], _GOLDEN_V1, atol=1e-12)
    assert np.allclose(TRM[3], _GOLDEN_TRM_ROW4, atol=1e-12)
    assert np.linalg.norm(TRM) == pytest.approx(_GOLDEN_TRM_FRO, abs=1e-12)


@pytest.mark.parametrize("name", sorted(_STATES))
def test_rigid_translation_maps_to_the_rotated_translation(name):
    """A rigid translation ``a`` moves every local position by ``R^T a``: the
    node blocks of ``TRM @ tile(a)`` all equal ``R^T a`` because the
    rotation terms cancel (the ``V_k`` sum to zero)."""
    ref, cur = _STATES[name]
    _, R, _, TRM, V = _frame_and_trm(ref, cur)
    a = np.array([0.4, 0.1, -0.7])
    out = (TRM @ np.tile(a, 8)).reshape(8, 3)
    assert np.allclose(out, np.tile(R[0].T @ a, (8, 1)), atol=1e-12)
    assert np.allclose(V.sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(_STATES))
def test_rigid_rotation_maps_to_zero_and_one_stage_does_not(name):
    """THE test that separates the two stages.  A rigid rotation of the
    element ``du_k = w x (x_k - x_1)`` changes no local coordinate, so the
    full ``TRM`` must annihilate it.  ``blockdiag(R^T)`` — the frame alone —
    leaves ``R^T du``, which is of the order of ``|w| * |x|``."""
    ref, cur = _STATES[name]
    _, R, _, TRM, _ = _frame_and_trm(ref, cur)
    w = np.array([0.3, -0.2, 0.5])
    du = np.cross(w, cur - cur[0]).reshape(24)
    assert np.abs(TRM @ du).max() < 1e-12
    one_stage = _block_diag_rt(R[0]) @ du
    assert np.abs(one_stage).max() > 0.1


@pytest.mark.parametrize("name", ["undeformed", "uniaxial"])
def test_trm_is_the_exact_derivative_of_the_local_positions(name):
    """``TRM`` is ``d(R^T x_k)/d(u)``.  Central differences of the *whole*
    pipeline (``crframe`` rebuilt at every perturbed state) agree with it to
    the difference step's accuracy.  Run where the closed form is smooth (see
    the module docstring); at an exactly rigid state the cube roots of
    rounding noise swamp a 1e-6 step."""
    ref, cur = _STATES[name]
    sav, R, _, TRM, _ = _frame_and_trm(ref, cur)

    def local_abs(x):
        Rx, _ = solid_crimp.crframe(sav, x[None])
        return (Rx[0].T @ x.T).T

    eps = 1e-6
    J = np.zeros((24, 24))
    for k in range(8):
        for b in range(3):
            xp, xm = cur.copy(), cur.copy()
            xp[k, b] += eps
            xm[k, b] -= eps
            J[:, 3 * k + b] = ((local_abs(xp) - local_abs(xm))
                               / (2.0 * eps)).reshape(24)
    assert np.abs(TRM - J).max() < 1e-7
    # and the frame alone is visibly not that derivative
    assert np.abs(_block_diag_rt(R[0]) - J).max() > 0.1


def test_trm_first_node_rows_are_only_the_frame():
    """Rows 1-3 of ``TRM`` are zero off the diagonal block (node 1 carries no
    ``S`` matrix); the diagonal block is ``R^T`` plus nothing."""
    _, R, _, TRM, _ = _frame_and_trm(*_STATES["uniaxial"])
    assert np.allclose(TRM[:3, :3], R[0].T, atol=1e-14)
    assert np.allclose(TRM[:3, 3:], 0.0, atol=0.0)


def test_trm_is_batched_over_elements():
    group = _hexa8_group()
    conn = group.conn
    cur = group.state["x"][conn]
    sav = solid_crimp.save_reference(group.state["x0"][conn])
    R, invj = solid_crimp.crframe(sav, cur)
    TRM, _ = solid_crimp.corotational_transform(cur, R, invj)
    assert TRM.shape == (3, 24, 24)
    for e, name in enumerate(("undeformed", "rigid", "uniaxial")):
        _, _, _, one, _ = _frame_and_trm(*_STATES[name])
        assert np.allclose(TRM[e], one, atol=1e-13)


# ---------------------------------------------------------------------------
# 4. transk and s8sfint3_crimp
# ---------------------------------------------------------------------------
def _local_stiffness(seed=3):
    """A symmetric local stiffness whose null space is the three replicated
    translations, as a real element stiffness' is."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(24, 24))
    M = A @ A.T + np.eye(24)
    T = np.tile(np.eye(3), (8, 1))
    P = np.eye(24) - T @ np.linalg.solve(T.T @ T, T.T)
    return P @ M @ P


def test_transk_is_trm_transpose_k_trm_and_not_the_other_way_round():
    _, _, _, TRM, _ = _frame_and_trm(*_STATES["uniaxial"])
    K = np.random.default_rng(5).normal(size=(24, 24))      # unsymmetric on purpose
    out = solid_crimp.transk(K[None], TRM[None])[0]
    assert np.allclose(out, TRM.T @ K @ TRM, atol=1e-12)
    assert np.abs(out - TRM @ K @ TRM.T).max() > 1e-3


@pytest.mark.parametrize("name", sorted(_STATES))
def test_transformed_stiffness_has_all_six_rigid_modes_in_its_null_space(name):
    """Three translations AND three rotations.  Only the full ``TRM`` gets the
    rotations: a stiffness rotated by ``blockdiag(R^T)`` is not
    rotation-invariant (the element would carry spurious stress under a rigid
    turn)."""
    ref, cur = _STATES[name]
    _, R, _, TRM, _ = _frame_and_trm(ref, cur)
    KL = _local_stiffness()
    K = solid_crimp.transk(KL[None], TRM[None])[0]
    assert np.allclose(K, K.T, atol=1e-10)

    rng = np.random.default_rng(11)
    for _ in range(3):
        w = rng.normal(size=3)
        du_rot = np.cross(w, cur - cur[0]).reshape(24)
        du_tr = np.tile(rng.normal(size=3), 8)
        assert np.abs(K @ du_rot).max() < 1e-10
        assert np.abs(K @ du_tr).max() < 1e-10

    B = _block_diag_rt(R[0])
    K_one = B.T @ KL @ B
    du_rot = np.cross(np.array([0.3, -0.2, 0.5]), cur - cur[0]).reshape(24)
    assert np.abs(K_one @ du_rot).max() > 1e-3


@pytest.mark.parametrize("name", sorted(_STATES))
def test_projected_forces_do_no_work_on_a_rigid_motion(name):
    """``f_global = TRM^T f_local``, so ``f_global . du = f_local . (TRM du)``
    and a rigid ``du`` does no work for ANY local force.  The plain rotation
    ``R f`` does not have that property."""
    ref, cur = _STATES[name]
    _, R, _, TRM, _ = _frame_and_trm(ref, cur)
    f = np.random.default_rng(2).normal(size=(1, 8, 3))
    fg, qf = solid_crimp.internal_force_to_global(f, R, TRM[None])
    assert fg.shape == (1, 8, 3) and qf.shape == (1, 24)

    w = np.array([0.3, -0.2, 0.5])
    du_rot = np.cross(w, cur - cur[0])
    assert abs(float((fg[0] * du_rot).sum())) < 1e-12

    du_tr = np.tile(np.array([0.4, 0.1, -0.7]), (8, 1))
    # translation: work = (sum of the local forces) . R^T a, up to the frame
    assert (abs(float((fg[0] * du_tr).sum()))
            == pytest.approx(abs(float(f[0].sum(axis=0) @ R[0].T
                                       @ np.array([0.4, 0.1, -0.7]))),
                             abs=1e-12))
    rotated = np.einsum("ij,kj->ki", R[0], f[0])
    assert abs(float((rotated * du_rot).sum())) > 1e-3


def test_internal_force_to_global_is_trm_transpose_f_and_stores_minus_r_f():
    _, R, _, TRM, _ = _frame_and_trm(*_STATES["uniaxial"])
    f = np.random.default_rng(9).normal(size=(1, 8, 3))
    fg, qf = solid_crimp.internal_force_to_global(f, R, TRM[None])
    assert np.allclose(fg[0].reshape(24), TRM.T @ f[0].reshape(24), atol=1e-12)
    assert np.allclose(qf[0].reshape(8, 3), -(f[0] @ R[0].T), atol=1e-12)


def test_stored_force_of_a_quarter_turn_is_minus_the_rotated_force():
    """Hand-computed: Rz(90) sends e_x to e_y, so a local force e_x on node 1
    is stored (``QF(1:3)``) as ``-e_y``."""
    R = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    f = np.zeros((1, 8, 3))
    f[0, 0] = [1.0, 0.0, 0.0]
    _, qf = solid_crimp.internal_force_to_global(f, R, np.eye(24)[None])
    assert np.allclose(qf[0, :3], [0.0, -1.0, 0.0], atol=1e-14)
    assert np.allclose(qf[0, 3:], 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 5. the reference state: s8sav3_imp, s8xref_imp, getuloc, srcoor3_imp
# ---------------------------------------------------------------------------
def test_save_reference_is_node_one_relative():
    x = _XI + np.array([4.0, 5.0, 6.0])
    sav = solid_crimp.save_reference(x[None])
    assert sav.shape == (1, 21)
    # SAV(1:3) is node 2 - node 1, SAV(4:6) node 3 - node 1, ... SAV(19:21) node 8
    assert np.allclose(sav[0, 0:3], [1, 0, 0])
    assert np.allclose(sav[0, 3:6], [1, 1, 0])
    assert np.allclose(sav[0, 6:9], [0, 1, 0])
    assert np.allclose(sav[0, 9:12], [0, 0, 1])
    assert np.allclose(sav[0, 18:21], [0, 1, 1])


def test_save_reference_is_guarded_by_abs_offg_le_one():
    x = np.stack([_XI, _XI * 2.0])
    old = np.full((2, 21), 7.0)
    sav = solid_crimp.save_reference(x, offg=[1.0, 2.0], sav=old)
    assert np.allclose(sav[0].reshape(7, 3), (_XI[1:] - _XI[0]))     # written
    assert np.allclose(sav[1], 7.0)                                  # kept
    assert np.allclose(solid_crimp.save_reference(x, offg=[-0.5, 1.0])[1]
                       .reshape(7, 3), 2.0 * (_XI[1:] - _XI[0]))


def test_local_displacement_of_a_stretch_is_the_stretch_in_the_local_axes():
    """Hand-computed: x stretched by 1.5, no turn, R = I: node 2 (1,0,0) ->
    (1.5,0,0) so ULX2 = 0.5 and nothing else moves in y or z."""
    cur = _XI * np.array([1.5, 1.0, 1.0])
    sav = solid_crimp.save_reference(_XI[None])
    R, _ = solid_crimp.crframe(sav, cur[None])
    ul = solid_crimp.local_displacements(cur[None], sav, R)[0]
    assert np.allclose(R[0], np.eye(3), atol=1e-12)
    assert np.allclose(ul[0], 0.0)                                   # node 1
    assert np.allclose(ul[1], [0.5, 0.0, 0.0], atol=1e-12)
    assert np.allclose(ul[:, 1:], 0.0, atol=1e-12)
    assert np.allclose(ul[:, 0], [0, .5, .5, 0, 0, .5, .5, 0], atol=1e-12)


@pytest.mark.parametrize("name", ["rigid", "skewed_rigid"])
def test_local_displacement_of_a_rigid_motion_is_zero(name):
    ref, cur = _STATES[name]
    sav, R, _, _, _ = _frame_and_trm(ref, cur)
    ul = solid_crimp.local_displacements(np.asarray(cur)[None], sav, R)
    assert np.abs(ul).max() < 1e-12


def test_local_reference_is_sav_when_the_frame_is_the_identity():
    sav = solid_crimp.save_reference(_XI[None])
    xref = solid_crimp.local_reference(_XI[None], np.eye(3)[None])
    assert np.allclose(xref, sav, atol=0.0)


def test_local_reference_of_a_quarter_turn():
    """Hand-computed: R = Rz(90) so ``R^T (x - x_1)`` sends (1,0,0) to (0,-1,0)."""
    R = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    xref = solid_crimp.local_reference(_XI[None], R)[0].reshape(7, 3)
    assert np.allclose(xref[0], [0.0, -1.0, 0.0])
    assert np.allclose(xref[2], [1.0, 0.0, 0.0])                     # node 4
    assert np.allclose(xref[3], [0.0, 0.0, 1.0])                     # node 5


def test_convected_coordinates_follow_ismstr():
    cur = _XI @ _rot([0, 0, 1], 0.5).T + 3.0
    sav = solid_crimp.save_reference(_XI[None])
    R, _ = solid_crimp.crframe(sav, cur[None])
    # ISMSTR 1: the saved reference, node 1 at the origin
    c1 = solid_crimp.convected_coordinates(cur[None], R, sav, ismstr=1)[0]
    assert np.allclose(c1, _XI - _XI[0], atol=1e-14)
    # ISMSTR 2 / 4: R^T (x - x_1); a rigid turn is undone, so this is the reference
    for ismstr in (2, 4):
        c = solid_crimp.convected_coordinates(cur[None], R, sav, ismstr)[0]
        assert np.allclose(c, _XI - _XI[0], atol=1e-12)
    # anything else leaves the coordinates alone
    c3 = solid_crimp.convected_coordinates(cur[None], R, sav, ismstr=3)[0]
    assert np.allclose(c3, cur)
    with pytest.raises(ValueError):
        solid_crimp.convected_coordinates(cur[None], R, None, ismstr=1)


def test_reference_state_stores_sav_and_xref_for_the_group():
    group = _group_of([_XI], [_XI * 2.0])
    st = solid_crimp.reference_state(group, frame=np.eye(3)[None])
    assert np.allclose(st.sav, solid_crimp.save_reference(_XI[None]))
    assert np.allclose(st.xref, st.sav)
    assert group.state["crimp_sav"] is st.sav


def test_saved_reference_moves_the_origin_of_the_frame():
    """``reference='saved'`` measures F against the geometry saved at the
    start of the step, so the frame is relative to it: after saving the
    current state, the frame is the identity again."""
    cur = _XI @ _rot([0, 0, 1], 0.5).T
    group = _group_of([_XI], [cur])
    R_undef = solid_crimp.imperfection_frame(group, reference="undeformed")
    assert not np.allclose(R_undef[0], np.eye(3), atol=1e-3)
    solid_crimp.reference_state(group, x_ref=group.state["x"])
    R_saved = solid_crimp.imperfection_frame(group, reference="saved")
    assert np.allclose(R_saved[0], np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------
# 6. the interface
# ---------------------------------------------------------------------------
def test_imperfection_frame_reads_the_model_when_given_one():
    group = _group_of([_XI], [_XI])
    model = Model()
    model.x0 = _XI.copy()
    model.x = _XI @ _rot([0, 0, 1], 0.5).T
    R = solid_crimp.imperfection_frame(group, model)
    assert np.allclose(R[0], _rot([0, 0, 1], 0.5), atol=1e-12)
    # explicit coordinates win over the model
    R2 = solid_crimp.imperfection_frame(group, model, x=_XI)
    assert np.allclose(R2[0], np.eye(3), atol=1e-12)


def test_imperfection_frame_refuses_what_it_cannot_do():
    group = _group_of([_XI], [_XI])
    with pytest.raises(ValueError):
        solid_crimp.imperfection_frame(group, reference="current")
    with pytest.raises(ValueError):
        solid_crimp.imperfection_frame(group, reference="saved")   # nothing saved
    tet = ElementGroup(ids=np.array([1]), conn=np.arange(4)[None, :],
                       part=np.zeros(1, dtype=np.int64))
    tet.state["x0"] = tet.state["x"] = _XI[:4]
    with pytest.raises(ValueError):
        solid_crimp.imperfection_frame(tet)
    bare = ElementGroup(ids=np.array([1]), conn=np.arange(8)[None, :],
                        part=np.zeros(1, dtype=np.int64))
    with pytest.raises(ValueError):
        solid_crimp.imperfection_frame(bare)


# ---------------------------------------------------------------------------
# 7. reachability: no crframe/crtrans card exists upstream, so none exists here
# ---------------------------------------------------------------------------
def test_no_crframe_or_crtrans_keyword_is_invented():
    """The plan asks for a ``crframe``/``crtrans`` keyword reader.  Upstream has
    none: ``starter/`` and ``reader/`` contain no such card (grep for
    ``crframe|crtrans`` in the starter finds only unrelated ``JCVT`` text), and
    the routines are reached by property flags alone (``Isolid = 17``/``19``
    -> ``IHBE = 17``, ``IINT = 3``).  Adding a card would invent a deck format,
    so the deck-side surface is the existing ``/PROP/TYPE14`` ``isolid``
    reader, and wiring it is the dispatch task (P2.11)."""
    from pyradioss.input.keywords.dispatch import KEYWORD_PARSERS

    banned = [k for k in KEYWORD_PARSERS
              if any(s in k.upper() for s in ("CRFRAME", "CRTRANS", "CRIMP"))]
    assert banned == []
    assert "crframe" in solid_crimp.__doc__.lower()
    assert "unreachable from a deck" in solid_crimp.__doc__
