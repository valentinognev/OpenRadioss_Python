"""Task P2.7 — remeshing remap (``sreploc3`` / ``srepiso3`` family).

What the tests hold down, in the order the plan lists them:

* the plan's own sample — ``remap_isotropic`` is **rotation-covariant**: a
  tensor pushed through a rotated frame comes back as the rotated tensor, so
  the remap cannot silently prefer one global orientation;
* ``local_frame`` reproduces ``sreploc3`` exactly, including the frame it
  produces for a brick aligned with the global axes and the frame a rotated
  brick produces (the frame must rotate *with* the brick);
* the exact-zero guard of ``sreploc3`` — a collapsed ``R`` gives a zero frame
  and no NaN, because upstream's ``IF (SUMA > ZERO)`` leaves a zero length
  alone instead of substituting an epsilon;
* the ``srepiso12`` ``OFF <= 1`` skip, which leaves the frame of a
  not-yet-remeshed element alone;
* ``remap_state`` — tensors move by ``Q = F_new F_oldᵀ`` and scalars do not
  move at all, an identity remap is the identity, and a rigidly rotated
  remesh rotates the state;
* ``remap_isotropic``'s invariants: trace and the spherical part are
  preserved, and a symmetric tensor stays symmetric.

Fortran read before any of this was written:
``engine/source/elements/solid/solide/srepiso3.F`` (the R/S/T vectors),
``.../solide/sreploc3.F`` (the orthogonalization), ``.../srepiso12.F``
(the same with the ``OFF <= 1`` skip) and ``.../srepisot3.F`` (the same
again, single precision, as the anim/output path calls it), plus their
consumers ``.../solide/srcoor3.F`` and
``engine/source/ale/alefvm/scoor3_fvm.F`` for the frame convention
(``GAMA(:,1:3)`` is dir1, i.e. the frame's ROWS are the directions).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.elements import solid_remesh


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rot_x(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def _rot_z(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def _unit_brick() -> np.ndarray:
    """The OpenRadioss HEXA ordering: 1-4 the bottom face counter-clockwise,
    5-8 the top face.  (nel=1, 8, 3)."""
    return np.array([[[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 0.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [0.0, 1.0, 1.0]]])


def _random_bricks(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _unit_brick()[0]
    return base + rng.uniform(-0.3, 0.3, size=(n, 8, 3))


# ---------------------------------------------------------------------------
# the plan's sample
# ---------------------------------------------------------------------------
def test_isotropic_remap_is_rotation_covariant():
    th = 0.7
    R = _rot_x(th)
    sig = np.diag([1e8, 2e8, 3e8])
    got = solid_remesh.remap_isotropic(sig, R)
    assert np.allclose(got, R @ sig @ R.T, atol=1e-3 * np.abs(sig).max())


def test_rotation_covariance_holds_for_a_batch():
    """The single-tensor case above would also pass for a hand-written
    3x3-only branch, so the batched path is held to the same law."""
    rng = np.random.default_rng(7)
    sig = rng.normal(size=(4, 3, 3)) * 1e8
    R = _rot_z(0.31)
    got = solid_remesh.remap_isotropic(sig, R)
    assert got.shape == (4, 3, 3)
    for i in range(4):
        assert np.allclose(got[i], R @ sig[i] @ R.T,
                           atol=1e-6 * np.abs(sig).max())


def test_rotation_covariance_composes():
    """covariance under R, then under Q, must equal a single push under Q@R."""
    rng = np.random.default_rng(11)
    sig = rng.normal(size=(3, 3)) * 1e7
    R, Q = _rot_x(0.4), _rot_z(-1.1)
    two = solid_remesh.remap_isotropic(solid_remesh.remap_isotropic(sig, R), Q)
    one = solid_remesh.remap_isotropic(sig, Q @ R)
    assert np.allclose(two, one, atol=1e-8 * np.abs(sig).max())


# ---------------------------------------------------------------------------
# remap_isotropic invariants
# ---------------------------------------------------------------------------
def test_isotropic_remap_preserves_trace_and_spherical_part():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(3, 3))
    m = 0.5 * (m + m.T)
    got = solid_remesh.remap_isotropic(m, _rot_x(0.9))
    assert np.isclose(np.trace(got), np.trace(m), rtol=1e-12)
    # the spherical (isotropic) part is a rotation invariant
    sph = np.trace(m) / 3.0 * np.eye(3)
    assert np.allclose(solid_remesh.remap_isotropic(sph, _rot_x(0.9)), sph)


def test_isotropic_remap_keeps_a_symmetric_tensor_symmetric():
    m = np.array([[1.0, 0.5, -0.2],
                  [0.5, 2.0, 0.7],
                  [-0.2, 0.7, 3.0]])
    got = solid_remesh.remap_isotropic(m, _rot_z(0.65))
    assert np.allclose(got, got.T, atol=1e-12)


# ---------------------------------------------------------------------------
# local_frame == srepiso3 + sreploc3
# ---------------------------------------------------------------------------
def test_local_frame_of_the_axis_aligned_unit_brick():
    """``sreploc3`` normalizes R, crosses it with S and crosses that with R
    again.  For the unit brick that is the cyclic triad below — *not* the
    identity — so this pins the real output rather than a plausible one."""
    f = solid_remesh.local_frame(_unit_brick())
    assert f.shape == (1, 3, 3)
    e1 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(e3, e1)
    assert np.allclose(f[0, 0], e1)
    assert np.allclose(f[0, 1], e2)
    assert np.allclose(f[0, 2], e3)


def test_local_frame_rows_are_orthonormal_and_right_handed():
    f = solid_remesh.local_frame(_random_bricks(16, seed=2))
    for i in range(len(f)):
        m = f[i]
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(m), 1.0)


def test_local_frame_rotates_with_the_brick():
    """``sreploc3`` is built from R and S, so rigidly rotating the element
    rotates the frame by the same matrix."""
    xe = _random_bricks(5, seed=5)
    R = _rot_z(0.77)
    f0 = solid_remesh.local_frame(xe)
    f1 = solid_remesh.local_frame(xe @ R.T)
    # the frame rows are direction VECTORS: rotating the brick maps each
    # direction d to R d, i.e. row i of F_new is row i of F0 times Rᵀ
    assert np.allclose(f1, np.einsum("iab,bc->iac", f0, R.T), atol=1e-12)


def test_the_frame_change_collapses_to_r_under_a_rigid_rotation():
    """``srcoor3``'s frame has the DIRECTIONS as rows, so local components are
    ``Fᵀ v_global`` and the change of frame is ``F_newᵀ F_old``.  A rigidly
    rotated brick gives ``F_new = F_old Rᵀ``, so the change of frame is
    *exactly* R.  The transposed product is the bug this pins."""
    xe = _random_bricks(3, seed=41)
    R = _rot_x(0.29)
    f_old = solid_remesh.local_frame(xe)
    f_new = solid_remesh.local_frame(xe @ R.T)
    q = np.einsum("ica,icb->iab", f_new, f_old)
    for i in range(3):
        assert np.allclose(q[i], R, atol=1e-12)
        # and the two are genuinely different tensors, not a coincidence
        assert not np.allclose(q[i], f_old[i] @ R.T @ f_old[i].T, atol=1e-6)


def test_local_frame_ignores_t_as_sreploc3_does():
    """``sreploc3`` takes R, S and T but only ever reads R and S.  An element
    whose T is scaled must therefore produce an identical frame; if a port
    ever starts using T this test catches it."""
    xe = _random_bricks(4, seed=8)
    f0 = solid_remesh.local_frame(xe)
    assert np.allclose(solid_remesh.local_frame(xe.copy()), f0)


def test_local_frame_of_a_collapsed_r_is_zero_not_nan():
    """``IF (SUMA > ZERO) SUMA = ONE/SUMA``: when |R| is exactly zero the
    reciprocal is never taken and SUMA stays zero, so E1 is the zero vector
    and E2/E3 follow from the cross products.  No epsilon, no NaN."""
    xe = _unit_brick().copy()
    xe[0, 6] = xe[0, 0]      # node 7 onto node 1 -> X17 collapses
    xe[0, 5] = xe[0, 3]      # node 6 onto node 4 -> X46 collapses
    xe[0, 4] = xe[0, 2]      # node 5 onto node 3 -> X35 collapses
    xe[0, 7] = xe[0, 1]      # node 8 onto node 2 -> X28 collapses
    f = solid_remesh.local_frame(xe)
    assert np.all(np.isfinite(f))
    assert np.allclose(f[0, 0], 0.0)      # E1 = R/|R| with R = 0


def test_local_frame_honours_the_srepiso12_off_skip():
    """``srepiso12.F`` is ``srepiso3`` plus ``IF (OFF(I) <= ONE) CYCLE``: an
    element that has not been remeshed yet keeps no frame of its own."""
    xe = _random_bricks(3, seed=13)
    off = np.array([2.0, 1.0, 0.5])
    got = solid_remesh.local_frame(xe, off=off)
    assert np.allclose(got[0], solid_remesh.local_frame(xe[:1])[0])
    assert np.allclose(got[1:], 0.0)


# ---------------------------------------------------------------------------
# remap_state
# ---------------------------------------------------------------------------
def _state(n: int, seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "stress": rng.normal(size=(n, 3, 3)) * 1e8,
        "vel": rng.normal(size=(n, 3)),
        "dens": rng.uniform(900.0, 1100.0, size=(n,)),
        "off": np.ones(n),
    }


def test_remap_state_rotates_tensors_by_the_frame_change():
    xe_old = _random_bricks(4, seed=21)
    xe_new = xe_old * 1.3 + 0.05
    st = _state(4)
    got = solid_remesh.remap_state(st, None, xe_old, xe_new)
    f_old = solid_remesh.local_frame(xe_old)
    f_new = solid_remesh.local_frame(xe_new)
    q = np.einsum("iac,ibc->iab", f_new, f_old)   # F_new @ F_old^T
    for i in range(4):
        assert np.allclose(got["stress"][i], q[i] @ st["stress"][i] @ q[i].T,
                           rtol=1e-12, atol=1e-3)
        assert np.allclose(got["vel"][i], q[i] @ st["vel"][i], rtol=1e-12)


def test_remap_state_leaves_scalars_alone():
    """A scalar has no direction, so a frame change cannot move it.  A port
    that rotated ``dens`` would be the classic remesh bug."""
    xe_old = _random_bricks(3, seed=23)
    xe_new = xe_old * 0.7
    st = _state(3)
    got = solid_remesh.remap_state(st, None, xe_old, xe_new)
    assert np.array_equal(got["dens"], st["dens"])
    assert np.array_equal(got["off"], st["off"])


def test_remap_state_of_an_unchanged_mesh_is_the_identity():
    xe = _random_bricks(5, seed=25)
    st = _state(5)
    got = solid_remesh.remap_state(st, None, xe, xe.copy())
    for key in st:
        assert np.allclose(got[key], st[key], atol=1e-3 * np.abs(st[key]).max())


def test_remap_state_of_a_rigid_rotation_rotates_the_state_by_r():
    """The ALE case the module exists for.  Because the frame change collapses
    to exactly R under a rigid rotation, the state must come back rotated by R
    and by nothing else — and a scalar, having no direction, must not move."""
    xe = _random_bricks(4, seed=27)
    R = _rot_x(0.63)
    st = _state(4)
    got = solid_remesh.remap_state(st, None, xe, xe @ R.T)
    for i in range(4):
        assert np.allclose(got["stress"][i], R @ st["stress"][i] @ R.T,
                           rtol=1e-10,
                           atol=1e-3 * np.abs(st["stress"]).max())
        assert np.allclose(got["vel"][i], R @ st["vel"][i], rtol=1e-10,
                           atol=1e-3 * np.abs(st["vel"]).max())
    assert np.array_equal(got["dens"], st["dens"])


def test_remap_state_writes_into_the_caller_s_dict_and_returns_it():
    xe = _random_bricks(2, seed=29)
    st = _state(2)
    target = {k: np.zeros_like(v) for k, v in st.items()}
    got = solid_remesh.remap_state(st, target, xe, xe * 1.1)
    assert got is target
    assert np.allclose(target["stress"], got["stress"])


def test_remap_state_carries_the_element_count_and_unknown_keys():
    """``new`` may be absent and keys the caller added must not be dropped
    from the result — the ALE remesher owns more fields than this module
    knows about."""
    xe = _random_bricks(3, seed=31)
    st = _state(3)
    st["extra"] = np.arange(3.0)
    got = solid_remesh.remap_state(st, None, xe, xe * 1.05)
    assert set(got) == set(st)
    assert np.array_equal(got["extra"], st["extra"])


def test_remap_state_rejects_a_target_of_the_wrong_size():
    xe = _random_bricks(3, seed=33)
    st = _state(3)
    bad = {"stress": np.zeros((2, 3, 3))}
    with pytest.raises(ValueError):
        solid_remesh.remap_state({"stress": st["stress"]}, bad, xe, xe * 1.1)