"""M511 — 3-Node Discrete Kirchhoff Triangle (DKT18) Shell Element Tests.

Covers:
  - Local frame orthonormality and right-handedness (det=1)
  - Area and geometry calculations vs Euclidean analytical values
  - 3D rigid-body rotation and translation invariance
  - Curvature identically zero under out-of-plane rigid translation
  - Curvature identically zero under rigid rotations (Bb @ u_rigid = 0)
  - Degenerate / zero area element safety
  - Empty group handling across all element APIs
  - Cycle 0 Courant step calculation without state modification
  - Lumped mass and rotational inertia conservation
  - Consistent mass matrix symmetry, positive definiteness (18 positive eigenvalues),
    and exact kinetic energy
  - Pure membrane tension
  - Pure bending moment and curvature
  - Rigid body translation producing zero forces and moments
  - Rigid body rotation producing zero forces and moments
  - Linear momentum conservation (sum F_i = 0)
  - Angular momentum conservation (sum (r_i x F_i + M_i) = 0)
  - Tangent stiffness rank 12 and exact 6 rigid-body null modes (residual < 1e-6)
  - Geometric stiffness symmetry and translational null modes
  - Element deletion (off=0) and LAW0 void handling
  - Directional derivative consistency of implicit internal forces vs tangent matrix
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyradioss.common.constants import EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_dkt18
from pyradioss.elements.shell_dkt18 import _edofs, consistent_mass, forces, init_group, kgeo, static_internal_forces, implicit_internal_forces, tangent
from pyradioss.accel.jit_kernels.shells_dkt18 import cdkcoor3, cdkderic3, cdkderi3, cdkcurv3
from pyradioss.model.model import ElementGroup, Model


class SimpleMat:
    def __init__(self, E=2.1e11, nu=0.3, rho0=7850.0, law=1):
        self.E = E
        self.nu = nu
        self.rho0 = rho0
        self.law = law
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.G = E / (2.0 * (1.0 + nu))
        self.fail = None
        self.params = {"E": E, "nu": nu, "rho": rho0, "eps_p_max": EP30}

    def sound_speed_shell(self):
        return np.sqrt(self.E / (self.rho0 * (1.0 - self.nu**2))) if (self.rho0 > 0.0 and self.law != 0) else 0.0


class SimpleProp:
    def __init__(self, thick=0.01, nip=3, dn=1e-3, ish3n=2):
        self.params = {
            "thick": thick,
            "nip": nip,
            "dn": dn,
            "ish3n": ish3n,
        }
        self.thick = thick
        self.nip = nip


def _create_mock_dkt18_group(coords, thick=0.01, nip=3, E=2.1e11, nu=0.3, rho0=7850.0, law=1):
    n = len(coords) // 3
    conn = np.arange(len(coords), dtype=np.int64).reshape(n, 3)
    mat = SimpleMat(E=E, nu=nu, rho0=rho0, law=law)
    prop = SimpleProp(thick=thick, nip=nip)

    model = Model()
    model.x0 = coords.copy()

    group = ElementGroup(
        ids=np.arange(1, n + 1, dtype=np.int64),
        conn=conn,
        part=np.zeros(n, dtype=np.int64),
        state={
            "slices": [(slice(0, n), mat, prop)],
            "part_ids": np.ones(n, dtype=np.int64),
        },
    )
    log = MessageLog()
    init_group(group, model, log)
    return group, model


# ============================================================================
# 1. Local frame orthonormality
# ============================================================================
def test_dkt18_local_frame_orthonormality():
    """Verify local triads e1, e2, e3 are strictly orthonormal with det(E)=1."""
    xe = np.array([
        [[0.1, 0.2, 0.3], [1.1, 0.5, 0.9], [0.6, 1.4, 0.2]],
        [[5.0, 1.0, -2.0], [8.0, 3.0, 0.0], [6.0, 6.0, 3.0]]
    ])
    ve = np.zeros_like(xe)
    re = np.zeros_like(xe)
    area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly, e_frame = cdkcoor3(xe, ve, re, 0.0)

    for i in range(2):
        e1 = np.array([e_frame[0][i], e_frame[1][i], e_frame[2][i]])
        e2 = np.array([e_frame[3][i], e_frame[4][i], e_frame[5][i]])
        e3 = np.array([e_frame[6][i], e_frame[7][i], e_frame[8][i]])

        assert np.isclose(np.linalg.norm(e1), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(e2), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(e3), 1.0, atol=1e-12)

        assert np.isclose(np.dot(e1, e2), 0.0, atol=1e-12)
        assert np.isclose(np.dot(e2, e3), 0.0, atol=1e-12)
        assert np.isclose(np.dot(e3, e1), 0.0, atol=1e-12)

        R = np.column_stack([e1, e2, e3])
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


# ============================================================================
# 2. Area and geometry
# ============================================================================
def test_dkt18_area_and_geometry():
    """Verify area matches 0.5 * |(x2-x1) x (x3-x1)|."""
    x = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    group, model = _create_mock_dkt18_group(x)
    assert np.isclose(group.state["area0"][0], 50.0, atol=1e-10)

    # Equilateral triangle of side length L=2: Area = sqrt(3)/4 * 4 = sqrt(3)
    L = 2.0
    h = np.sqrt(3.0)
    x_eq = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [L / 2.0, h, 0.0]])
    group_eq, _ = _create_mock_dkt18_group(x_eq)
    assert np.isclose(group_eq.state["area0"][0], np.sqrt(3.0), atol=1e-10)


# ============================================================================
# 3. Area invariance under 3D rigid transform
# ============================================================================
def test_dkt18_area_invariance_3d_transform():
    """Area and local coordinates are invariant under arbitrary 3D rigid motions."""
    x_orig = np.array([[1.0, 2.0, 3.0], [4.0, 3.0, 1.0], [2.0, 5.0, 4.0]])
    group0, _ = _create_mock_dkt18_group(x_orig)
    area0 = group0.state["area0"][0]

    rot = Rotation.from_rotvec(np.array([0.5, 0.7, -0.3])).as_matrix()
    trans = np.array([-10.0, 20.0, 5.0])
    x_rot = (rot @ x_orig.T).T + trans

    group_rot, _ = _create_mock_dkt18_group(x_rot)
    area_rot = group_rot.state["area0"][0]
    assert np.isclose(area_rot, area0, atol=1e-10)


# ============================================================================
# 4. Curvature zero under out-of-plane rigid translation
# ============================================================================
def test_dkt18_curvature_rigid_translation_zero():
    """Out-of-plane rigid translation produces identically zero curvature."""
    xe = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    ve = np.zeros_like(xe)
    re = np.zeros_like(xe)
    area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly, _ = cdkcoor3(xe, ve, re, 0.0)
    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _ = cdkderic3(
        xl2, yl2, xl3, yl3, area2, np.array([0.01]), np.array([0.3]), np.array([0.0001]))

    vlz = np.zeros((1, 2))
    rlx = np.zeros((1, 3))
    rly = np.zeros((1, 3))

    for ng in range(3):
        eta, ksi = shell_dkt18._A_HAMMER[ng]
        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)
        kxx = np.zeros(1)
        kyy = np.zeros(1)
        kxy = np.zeros(1)
        cdkcurv3(bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3, vlz, rlx, rly, kxx, kyy, kxy)
        assert np.isclose(kxx[0], 0.0, atol=1e-15)
        assert np.isclose(kyy[0], 0.0, atol=1e-15)
        assert np.isclose(kxy[0], 0.0, atol=1e-15)


# ============================================================================
# 5. Curvature zero under rigid rotations
# ============================================================================
def test_dkt18_curvature_rigid_rotation_zero():
    """Rigid body rotations produce zero curvature (Bb @ u_rigid = 0)."""
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [0.5, 1.8, 0.0]])
    group, model = _create_mock_dkt18_group(coords)

    ke, _ = tangent(group, model.x0)
    c0 = coords.mean(axis=0)
    for axis in range(3):
        th = np.zeros(3)
        th[axis] = 1.0
        u_rot = np.zeros(18)
        for i in range(3):
            dr = coords[i] - c0
            u_rot[i * 6: i * 6 + 3] = np.cross(th, dr)
            u_rot[i * 6 + 3: i * 6 + 6] = th
        res = np.linalg.norm(ke[0] @ u_rot)
        assert res < 1e-4


# ============================================================================
# 6. Degenerate element safety
# ============================================================================
def test_dkt18_degenerate_element_safety():
    """Zero-area collinear element does not crash or raise exceptions."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    group, model = _create_mock_dkt18_group(x)

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    dt_e = forces(group, model.x0, None, None, 0.0, fint, mint)
    assert not np.any(np.isnan(dt_e))

    ke, _ = tangent(group, model.x0)
    assert not np.any(np.isnan(ke))


# ============================================================================
# 7. Empty group handling
# ============================================================================
def test_dkt18_empty_group_handling():
    """Empty group (n=0) handled gracefully across all element APIs."""
    group = ElementGroup(
        ids=np.empty(0, dtype=np.int64),
        conn=np.empty((0, 3), dtype=np.int64),
        part=np.empty(0, dtype=np.int64),
        state={"slices": []},
    )
    model = Model()
    model.x0 = np.empty((0, 3), dtype=float)
    log = MessageLog()

    node_idx, mass_c, inertia_c = init_group(group, model, log)
    assert len(node_idx) == 0

    fint = np.zeros((0, 3))
    mint = np.zeros((0, 3))
    dt_e = forces(group, model.x0, None, None, 1.0, fint, mint)
    assert len(dt_e) == 0

    ke, edofs = tangent(group, model.x0)
    assert ke.shape == (0, 18, 18)
    assert edofs.shape == (0, 18)

    kg, edofs_g = kgeo(group, model.x0)
    assert kg.shape == (0, 18, 18)

    me, edofs_m = consistent_mass(group, model.x0)
    assert me.shape == (0, 18, 18)

    static_internal_forces(group, model.x0, None, None, fint, mint)
    implicit_internal_forces(group, model.x0, None, None, fint, mint, nlgeom=False)
    implicit_internal_forces(group, model.x0, None, None, fint, mint, nlgeom=True)


# ============================================================================
# 8. Cycle 0 Courant step calculation
# ============================================================================
def test_dkt18_cycle_0_courant_step():
    """Cycle 0 (dt <= 0 or v is None) computes critical time step without state change."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_dkt18_group(x)

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    dt_e = forces(group, model.x0, None, None, 0.0, fint, mint)

    assert len(dt_e) == 1
    assert dt_e[0] > 0.0
    assert dt_e[0] < EP30
    assert np.allclose(fint, 0.0)
    assert np.allclose(mint, 0.0)
    assert np.allclose(group.state["sig"], 0.0)


# ============================================================================
# 9. Lumped mass conservation
# ============================================================================
def test_dkt18_lumped_mass_conservation():
    """Total lumped mass equals rho * t * A, scattered by vertex-angle
    fractions (P3.9: starter c3inmas.F:1136-1150 + spmd_msin.F:557-558 —
    this right triangle's angles are 90/45/45 deg, so (1/2, 1/4, 1/4),
    not uniform thirds)."""
    x = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    rho = 7850.0
    t = 0.02
    A = 50.0
    expected_total_mass = rho * t * A

    group, model = _create_mock_dkt18_group(x, thick=t, rho0=rho)
    node_idx, mass_c, inertia_c = init_group(group, model, MessageLog())

    assert np.isclose(mass_c.sum(), expected_total_mass)
    assert np.allclose(mass_c, expected_total_mass * np.array([0.5, 0.25, 0.25]))
    assert np.all(inertia_c > 0.0)


# ============================================================================
# 10. Consistent mass matrix properties
# ============================================================================
def test_dkt18_consistent_mass_matrix_properties():
    """Consistent mass matrix is symmetric, strictly positive definite, and conserves kinetic energy."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.3, 0.9, 0.0]])
    group, model = _create_mock_dkt18_group(x)

    me, edofs = consistent_mass(group, model.x0)
    M = me[0]

    # Symmetry
    assert np.allclose(M, M.T, atol=1e-12)

    # Positive definiteness: 18 positive eigenvalues
    evals = np.linalg.eigvalsh(M)
    assert np.all(evals > 0.0)
    assert len(evals) == 18

    # Rigid translation kinetic energy: 0.5 * v.T @ M @ v = 0.5 * m * |v|^2
    total_m = group.state["mass"][0]
    v_rigid = np.zeros(18)
    v_vec = np.array([3.0, -2.0, 5.0])
    for i in range(3):
        v_rigid[i * 6: i * 6 + 3] = v_vec

    ke_computed = 0.5 * v_rigid @ M @ v_rigid
    ke_exact = 0.5 * total_m * np.dot(v_vec, v_vec)
    assert np.isclose(ke_computed, ke_exact, rtol=1e-10)


# ============================================================================
# 11. Pure membrane tension
# ============================================================================
def test_dkt18_pure_membrane_tension():
    """Pure in-plane tension generates membrane stress and zero bending moment."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_dkt18_group(x)

    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    v[1, 0] = 1.0  # Stretch node 2 along X

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    dt = 1e-4

    forces(group, model.x0, v, vr, dt, fint, mint)

    sig = group.state["sig"][0]
    assert np.all(sig[:, 0] > 0.0)
    assert np.allclose(mint, 0.0, atol=1e-10)


# ============================================================================
# 12. Pure bending
# ============================================================================
def test_dkt18_pure_bending():
    """Out-of-plane rotation produces restoring moments and energy increment."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_dkt18_group(x, nip=3)

    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    vr[1, 1] = 10.0  # Apply rotation rate thy to node 2

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    dt = 1e-4

    forces(group, model.x0, v, vr, dt, fint, mint)

    assert not np.allclose(mint, 0.0)
    assert group.state["eint"][0] > 0.0


# ============================================================================
# 13. Rigid body translation produces zero forces
# ============================================================================
def test_dkt18_rigid_body_translation_zero_forces():
    """Rigid body translation produces zero internal forces and moments."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 3.0, 1.0], [2.0, 5.0, 4.0]])
    group, model = _create_mock_dkt18_group(x)

    v = np.tile([10.0, -5.0, 8.0], (3, 1))
    vr = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    forces(group, model.x0, v, vr, 1e-3, fint, mint)

    assert np.allclose(fint, 0.0, atol=1e-10)
    assert np.allclose(mint, 0.0, atol=1e-10)
    assert np.isclose(group.state["eint"][0], 0.0, atol=1e-12)


# ============================================================================
# 14. Rigid body rotation produces zero forces
# ============================================================================
def test_dkt18_rigid_body_rotation_zero_forces():
    """Rigid body rotation about normal produces zero internal forces."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_dkt18_group(x)

    omega = np.array([0.0, 0.0, 0.01])
    c0 = x.mean(axis=0)
    v = np.cross(omega, x - c0)
    vr = np.tile(omega, (3, 1))

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    forces(group, model.x0, v, vr, 1e-6, fint, mint)

    assert np.allclose(fint, 0.0, atol=1e-6)
    assert np.allclose(mint, 0.0, atol=1e-6)


# ============================================================================
# 15. Linear momentum conservation
# ============================================================================
def test_dkt18_linear_momentum_conservation():
    """Sum of nodal internal forces is zero for arbitrary deformation."""
    x = np.array([[0.2, 0.5, -0.1], [1.8, 0.2, 0.4], [0.7, 2.1, 0.3]])
    group, model = _create_mock_dkt18_group(x)

    np.random.seed(42)
    v = np.random.randn(3, 3) * 5.0
    vr = np.random.randn(3, 3) * 5.0

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    forces(group, model.x0, v, vr, 1e-4, fint, mint)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ============================================================================
# 16. Angular momentum conservation
# ============================================================================
def test_dkt18_angular_momentum_conservation():
    """Sum of (r_i x F_i + M_i) is zero for arbitrary deformation."""
    x = np.array([[0.2, 0.5, -0.1], [1.8, 0.2, 0.4], [0.7, 2.1, 0.3]])
    group, model = _create_mock_dkt18_group(x)

    np.random.seed(123)
    v = np.random.randn(3, 3) * 2.0
    vr = np.random.randn(3, 3) * 2.0

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    forces(group, model.x0, v, vr, 1e-4, fint, mint)

    total_moment = np.zeros(3)
    for i in range(3):
        total_moment += np.cross(x[i], fint[i]) + mint[i]

    assert np.allclose(total_moment, 0.0, atol=1e-8)


# ============================================================================
# 17. Tangent stiffness rank and rigid-body nullspace
# ============================================================================
def test_dkt18_tangent_stiffness_rigid_nullspace():
    """18-DOF tangent stiffness has rank 12 and exact 6 zero eigenvalues."""
    x = np.array([[0.1, 0.2, 0.3], [1.1, 0.5, 0.9], [0.6, 1.4, 0.2]])
    group, model = _create_mock_dkt18_group(x)

    ke, edofs = tangent(group, model.x0)
    K = ke[0]

    assert np.allclose(K, K.T, atol=1e-6)

    evals = np.linalg.eigvalsh(K)
    assert np.all(np.abs(evals[:6]) < 1e-3)
    assert np.all(evals[6:] > 1.0)

    c0 = x.mean(axis=0)
    for axis in range(3):
        u_t = np.zeros(18)
        for i in range(3):
            u_t[i * 6 + axis] = 1.0
        assert np.linalg.norm(K @ u_t) < 1e-4

    for axis in range(3):
        th = np.zeros(3)
        th[axis] = 1.0
        u_r = np.zeros(18)
        for i in range(3):
            dr = x[i] - c0
            u_r[i * 6: i * 6 + 3] = np.cross(th, dr)
            u_r[i * 6 + 3: i * 6 + 6] = th
        assert np.linalg.norm(K @ u_r) < 1e-4


# ============================================================================
# 18. Geometric stiffness symmetry and nullspace
# ============================================================================
def test_dkt18_geometric_stiffness_symmetry_and_nullspace():
    """Geometric stiffness K_geo is symmetric and has exact translational null modes."""
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
    group, model = _create_mock_dkt18_group(x)

    group.state["sig"][0, :, 0] = 50e6
    group.state["sig"][0, :, 1] = 20e6
    group.state["sig"][0, :, 2] = 10e6

    kg, edofs = kgeo(group, model.x0)
    K_g = kg[0]

    assert np.allclose(K_g, K_g.T, atol=1e-10)

    for axis in range(3):
        u_t = np.zeros(18)
        for i in range(3):
            u_t[i * 6 + axis] = 1.0
        assert np.linalg.norm(K_g @ u_t) < 1e-10


# ============================================================================
# 19. Element deletion and LAW0 void handling
# ============================================================================
def test_dkt18_element_deletion_and_law0():
    """Deactivated elements (off=0) and LAW0 void elements yield zero forces and dt=EP30."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_dkt18_group(x, law=0)

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    v = np.ones((3, 3))
    vr = np.ones((3, 3))

    dt_e = forces(group, model.x0, v, vr, 1e-4, fint, mint)
    assert np.allclose(fint, 0.0)
    assert np.allclose(mint, 0.0)
    assert np.isclose(dt_e[0], EP30)

    ke, _ = tangent(group, model.x0)
    assert np.allclose(ke, 0.0)

    group2, model2 = _create_mock_dkt18_group(x, law=1)
    group2.state["off"][0] = 0.0
    dt_e2 = forces(group2, model2.x0, v, vr, 1e-4, fint, mint)
    assert np.isclose(dt_e2[0], EP30)

    ke2, _ = tangent(group2, model2.x0)
    assert np.allclose(ke2, 0.0)


# ============================================================================
# 20. Implicit internal forces consistency
# ============================================================================
def test_dkt18_implicit_internal_forces_consistency():
    """Directional derivative of implicit internal forces matches tangent stiffness."""
    x = np.array([[0.1, 0.2, 0.0], [1.1, 0.3, 0.0], [0.4, 1.2, 0.0]])
    group, model = _create_mock_dkt18_group(x)

    ke, edofs = tangent(group, model.x0)
    K = ke[0]

    np.random.seed(99)
    du = np.random.randn(18) * 1e-5

    u_disp = np.zeros((3, 3))
    ur_rot = np.zeros((3, 3))
    for i in range(3):
        u_disp[i] = du[i * 6: i * 6 + 3]
        ur_rot[i] = du[i * 6 + 3: i * 6 + 6]

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    implicit_internal_forces(group, model.x0, u_disp, ur_rot, fint, mint, nlgeom=False)

    f_vec = np.zeros(18)
    for i in range(3):
        f_vec[i * 6: i * 6 + 3] = -fint[i]
        f_vec[i * 6 + 3: i * 6 + 6] = -mint[i]

    f_linearized = K @ du
    rel_diff = np.linalg.norm(f_vec - f_linearized) / np.linalg.norm(f_linearized)
    assert rel_diff < 0.05
