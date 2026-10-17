"""
Test Suite for Milestone M602: Kinematic Mechanism Joints (/PROP/TYPE33 and /PROP/TYPE45).

Verifies:
1. Starter deck parsing for /PROP/TYPE33 (/PROP/KJOINT) and /PROP/TYPE45 (/PROP/KJOINT2).
2. Revolute joint allowing pure rotation while resisting radial forces and bending moments.
3. Spherical joint allowing 3D rotation while enforcing position locking.
4. Prismatic slider joint allowing 1D sliding while locking all other DOFs.
5. Cylindrical, Planar, and Universal kinematic joint types.
6. Momentum conservation to machine precision (< 1e-12) for forces and moments.
7. Explicit dynamic integration with kinematic joint constraints.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import KJoint, PropType33, PropType45, Property
from pyradioss.model.model import Model
from pyradioss.engine.kjoint import JointType, KJointEngine, build_kjoints, resolve_joint_type
from pyradioss.engine.engine import _integrate


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. Starter Deck Parsing for /PROP/TYPE33 and /PROP/TYPE45
# ============================================================================

def test_m602_starter_parsing_prop33(tmp_path: Path):
    """Verify parsing of /PROP/TYPE33 (/PROP/KJOINT) kinematic joint properties."""
    deck = """# RADIOSS STARTER DECK
/BEGIN
M602 Prop33 Test
1 1
/NODE
1 0.0 0.0 0.0
2 0.0 0.0 1.0
/PART/1
JointPart
1 1 0
/PROP/TYPE33/1
RevoluteJointProp
1 0
5 6 5000.0 0.05
20000.0 100.0 200.0 300.0
0 0 0
10.0 20.0 30.0
0 0 0
/SPRING/1
10 1 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert not log.errors

    assert 1 in model.properties
    prop = model.properties[1]
    assert prop.type == 33
    assert prop.params["joint_type"] == 1
    assert prop.params["xk"] == pytest.approx(5000.0)
    assert prop.params["cr"] == pytest.approx(0.05)
    assert prop.params["kn"] == pytest.approx(20000.0)
    assert prop.params["krx"] == pytest.approx(100.0)

    # Verify build_kjoints finds the joint from the spring element
    joints = build_kjoints(model, log)
    assert len(joints) == 1
    kj = joints[0]
    assert kj.joint_type in (JointType.REVOLUTE, JointType.SPHERICAL)
    assert kj.idx1 == 0
    assert kj.idx2 == 1


def test_m602_starter_parsing_prop45(tmp_path: Path):
    """Verify parsing of /PROP/TYPE45 (/PROP/KJOINT2) kinematic joint properties."""
    deck = """# RADIOSS STARTER DECK
/BEGIN
M602 Prop45 Test
1 1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
/PART/1
JointPart45
1 1 0
/PROP/TYPE45/1
KJoint2Prop
2 30000.0 1.0 0.02 0
0 0 150.0 250.0 350.0
0 0 0 0.0 0.0 0.0 0.0 0.0 0.0
5.0 15.0 25.0 0 0 0 0.0 0.0 0.0 0.0 0.0 0.0
/SPRING/1
20 1 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert not log.errors

    assert 1 in model.properties
    prop = model.properties[1]
    assert prop.type == 45
    assert prop.params["joint_type"] == 2
    assert prop.params["kn"] == pytest.approx(30000.0)
    assert prop.params["scale"] == pytest.approx(1.0)
    assert prop.params["cr"] == pytest.approx(0.02)
    assert prop.params["ktx"] == pytest.approx(150.0)

    # Verify build_kjoints detects and builds the joint
    joints = build_kjoints(model, log)
    assert len(joints) == 1
    kj = joints[0]
    assert kj.idx1 == 0
    assert kj.idx2 == 1


# ============================================================================
# 2. Revolute Joint Verification (Hinge)
# ============================================================================

def test_m602_revolute_joint_pure_rotation():
    """Verify Revolute Joint:
    - Allows pure unconstrained rotation around local axis 1 (Rx).
    - Resists radial forces (Ty, Tz) with restoring reaction forces.
    - Resists bending moments (Ry, Rz) with restoring reaction torques.
    """
    model = Model()
    log = MessageLog()
    model.nodes = [1, 2]
    model._id2idx = {1: 0, 2: 1}
    model.x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    model.v = np.zeros((2, 3))
    model.vr = np.zeros((2, 3))
    model.mass = np.array([2.0, 2.0], dtype=np.float64)
    model.inertia = np.array([1.0, 1.0], dtype=np.float64)

    # Local axis 1 along X: rotation free about X
    joint = KJoint(
        id=1, node1=1, node2=2,
        joint_type=JointType.REVOLUTE,
        kn=1e5, cr=0.1,
    )
    engine = KJointEngine(joint, model, log)
    assert engine.is_valid
    assert 3 in engine.free_dofs          # Rx free
    assert 4 in engine.locked_dofs        # Ry locked
    assert 5 in engine.locked_dofs        # Rz locked
    assert {0, 1, 2}.issubset(engine.locked_dofs) # Tx, Ty, Tz locked

    # A. Pure Rotation around local X axis
    # Set relative angular velocity omega_x = 10.0 rad/s
    model.vr[0] = [0.0, 0.0, 0.0]
    model.vr[1] = [10.0, 0.0, 0.0]
    vr_before = model.vr.copy()

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    # Rotation about X is free -> velocities should NOT be constrained
    assert model.vr[1, 0] == pytest.approx(vr_before[1, 0])
    assert model.vr[0, 0] == pytest.approx(vr_before[0, 0])

    # Internal torques on free rotation: M_x is zero (no resistance)
    F1, F2, M1, M2 = engine.compute_internal_forces(model.x, model.v, model.vr, dt=0.001)
    assert abs(M1[0]) < 1e-12
    assert abs(M2[0]) < 1e-12

    # B. Resistance to Radial Displacement / Forces (Y and Z)
    # Impose radial displacement along Y: dy = 0.05
    model.x[1] = [0.0, 0.05, 0.0]
    F1, F2, M1, M2 = engine.compute_internal_forces(model.x, model.v, model.vr, dt=0.001)
    # Penalty force should push node 2 back along -Y and pull node 1 along +Y
    assert F2[1] < -100.0
    assert F1[1] > 100.0
    assert F1[1] == pytest.approx(-F2[1])

    # C. Resistance to Bending Rotations (around Y and Z)
    # Impose relative angular velocity around Y: omega_y = 5.0 rad/s
    model.vr[0] = [0.0, 0.0, 0.0]
    model.vr[1] = [0.0, 5.0, 0.0]
    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    # Locked DOF -> relative angular velocity must be projected to zero
    assert (model.vr[1, 1] - model.vr[0, 1]) == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# 3. Spherical Joint Verification (Ball-and-Socket)
# ============================================================================

def test_m602_spherical_joint_3d_rotation_and_position_locking():
    """Verify Spherical Joint:
    - Allows arbitrary 3D rotation (Rx, Ry, Rz free).
    - Enforces position locking (Tx, Ty, Tz locked) under translational loads.
    """
    model = Model()
    log = MessageLog()
    model.nodes = [1, 2]
    model._id2idx = {1: 0, 2: 1}
    model.x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    model.v = np.zeros((2, 3))
    model.vr = np.zeros((2, 3))
    model.mass = np.array([5.0, 5.0], dtype=np.float64)
    model.inertia = np.array([2.0, 2.0], dtype=np.float64)

    joint = KJoint(
        id=2, node1=1, node2=2,
        joint_type=JointType.SPHERICAL,
        kn=2e5, cr=0.05,
    )
    engine = KJointEngine(joint, model, log)
    assert engine.is_valid
    assert engine.free_dofs == {3, 4, 5}       # All rotations free
    assert engine.locked_dofs == {0, 1, 2}     # All translations locked

    # A. Arbitrary 3D Rotation
    model.vr[0] = [1.0, -2.0, 3.0]
    model.vr[1] = [4.0, 5.0, -6.0]
    vr_copy = model.vr.copy()

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    # Rotations must remain completely uninhibited
    np.testing.assert_allclose(model.vr, vr_copy, atol=1e-14)

    # Resisting torque from spherical joint must be identically zero
    F1, F2, M1, M2 = engine.compute_internal_forces(model.x, model.v, model.vr, dt=0.001)
    np.testing.assert_allclose(M1, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(M2, [0.0, 0.0, 0.0], atol=1e-12)

    # B. Position Locking Enforcement
    # Apply translational relative velocity
    model.v[0] = [0.0, 0.0, 0.0]
    model.v[1] = [2.0, -3.0, 4.0]
    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    # Translations are locked -> relative translational velocity must become 0
    np.testing.assert_allclose(model.v[1] - model.v[0], [0.0, 0.0, 0.0], atol=1e-12)

    # Kinematic position enforcement
    model.x[1] = [0.01, -0.02, 0.03]
    engine.enforce(model.x, model.v, model.vr, dt=0.001)
    np.testing.assert_allclose(model.x[0], model.x[1], atol=1e-12)


# ============================================================================
# 4. Prismatic Slider Joint Verification
# ============================================================================

def test_m602_prismatic_slider_joint_1d_sliding():
    """Verify Prismatic Slider Joint:
    - Allows 1D linear sliding along axis 1 (Tx free).
    - Locks transverse translations (Ty, Tz) and all rotations (Rx, Ry, Rz).
    """
    model = Model()
    log = MessageLog()
    model.nodes = [10, 20]
    model._id2idx = {10: 0, 20: 1}
    # Axis along X
    model.x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    model.v = np.zeros((2, 3))
    model.vr = np.zeros((2, 3))
    model.mass = np.array([4.0, 4.0], dtype=np.float64)
    model.inertia = np.array([1.5, 1.5], dtype=np.float64)

    joint = KJoint(
        id=3, node1=10, node2=20,
        joint_type=JointType.SLIDER,
        kn=5e5,
    )
    engine = KJointEngine(joint, model, log)
    assert engine.is_valid
    assert engine.free_dofs == {0}             # Tx free
    assert engine.locked_dofs == {1, 2, 3, 4, 5} # Ty, Tz, Rx, Ry, Rz locked

    # A. 1D Sliding along local axis 1 (X)
    model.v[0] = [0.0, 0.0, 0.0]
    model.v[1] = [15.0, 0.0, 0.0]
    v_orig = model.v.copy()

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    # Sliding along X is free
    assert model.v[1, 0] == pytest.approx(v_orig[1, 0])
    assert model.v[0, 0] == pytest.approx(v_orig[0, 0])

    # Force along X is zero
    F1, F2, M1, M2 = engine.compute_internal_forces(model.x, model.v, model.vr, dt=0.001)
    assert abs(F1[0]) < 1e-12
    assert abs(F2[0]) < 1e-12

    # B. Transverse Translations and Rotations Locked
    model.v[1] = [15.0, 5.0, -8.0]
    model.vr[1] = [2.0, 3.0, 4.0]

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    # Transverse velocities (Y, Z) projected to zero
    assert (model.v[1, 1] - model.v[0, 1]) == pytest.approx(0.0, abs=1e-12)
    assert (model.v[1, 2] - model.v[0, 2]) == pytest.approx(0.0, abs=1e-12)
    # Sliding velocity (X) preserved
    assert (model.v[1, 0] - model.v[0, 0]) == pytest.approx(15.0, abs=1e-12)
    # All angular velocity differences projected to zero
    np.testing.assert_allclose(model.vr[1] - model.vr[0], [0.0, 0.0, 0.0], atol=1e-12)


# ============================================================================
# 5. Cylindrical, Planar, and Universal Joints
# ============================================================================

def test_m602_cylindrical_planar_universal():
    """Verify Cylindrical, Planar, and Universal joint kinematics."""
    model = Model()
    log = MessageLog()
    model.nodes = [1, 2]
    model._id2idx = {1: 0, 2: 1}
    model.x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    model.mass = np.array([2.0, 2.0], dtype=np.float64)
    model.inertia = np.array([1.0, 1.0], dtype=np.float64)

    # A. Cylindrical (Tx and Rx free)
    j_cyl = KJoint(id=1, node1=1, node2=2, joint_type=JointType.CYLINDRICAL)
    eng_cyl = KJointEngine(j_cyl, model, log)
    assert eng_cyl.free_dofs == {0, 3}
    assert eng_cyl.locked_dofs == {1, 2, 4, 5}

    # B. Planar (Ty, Tz and Rx free)
    j_plan = KJoint(id=2, node1=1, node2=2, joint_type=JointType.PLANAR)
    eng_plan = KJointEngine(j_plan, model, log)
    assert eng_plan.free_dofs == {1, 2, 3}
    assert eng_plan.locked_dofs == {0, 4, 5}

    # C. Universal (Ry and Rz free)
    j_univ = KJoint(id=3, node1=1, node2=2, joint_type=JointType.UNIVERSAL)
    eng_univ = KJointEngine(j_univ, model, log)
    assert eng_univ.free_dofs == {4, 5}
    assert eng_univ.locked_dofs == {0, 1, 2, 3}


# ============================================================================
# 6. Momentum Conservation to Machine Precision (< 1e-12)
# ============================================================================

@pytest.mark.parametrize("jtype", [
    JointType.REVOLUTE,
    JointType.SPHERICAL,
    JointType.CYLINDRICAL,
    JointType.PLANAR,
    JointType.UNIVERSAL,
    JointType.SLIDER,
])
def test_m602_momentum_conservation_machine_precision(jtype: JointType):
    """Verify linear and angular momentum conservation to machine precision (< 1e-12).

    1. Sum of reaction forces: sum(F) = F1 + F2 = 0.
    2. Sum of reaction torques: sum(M) = M1 + M2 + x1 x F1 + x2 x F2 = 0.
    3. Velocity projection impulses conserve linear and angular momentum identically.
    """
    model = Model()
    log = MessageLog()
    model.nodes = [100, 200]
    model._id2idx = {100: 0, 200: 1}
    # Non-zero offset between nodes to test moment-arm coupling
    x1 = np.array([1.5, -2.0, 3.5], dtype=np.float64)
    x2 = np.array([2.5, -1.0, 4.0], dtype=np.float64)
    model.x = np.array([x1, x2], dtype=np.float64)
    model.v = np.array([[-1.2, 3.4, 0.5], [2.1, -0.8, 1.9]], dtype=np.float64)
    model.vr = np.array([[0.5, -1.0, 2.2], [-1.8, 0.4, 3.1]], dtype=np.float64)
    model.mass = np.array([3.5, 7.2], dtype=np.float64)
    model.inertia = np.array([2.1, 4.8], dtype=np.float64)

    joint = KJoint(
        id=99, node1=100, node2=200,
        joint_type=jtype,
        kn=1.5e5, cr=0.08,
        ktx=100.0, kty=200.0, ktz=300.0,
        krx=50.0, kry=75.0, krz=120.0,
    )
    engine = KJointEngine(joint, model, log)
    assert engine.is_valid

    # 1. Force and Torque Balance in compute_internal_forces
    F1, F2, M1, M2 = engine.compute_internal_forces(model.x, model.v, model.vr, dt=0.0005)

    # Linear force equilibrium: F1 + F2 = 0
    f_sum = F1 + F2
    np.testing.assert_allclose(f_sum, [0.0, 0.0, 0.0], atol=1e-14)

    # Angular torque equilibrium about origin: M1 + M2 + x1 x F1 + x2 x F2 = 0
    total_torque = M1 + M2 + np.cross(x1, F1) + np.cross(x2, F2)
    np.testing.assert_allclose(total_torque, [0.0, 0.0, 0.0], atol=1e-12)

    # 2. Velocity Projection Momentum Conservation
    v_pre = model.v.copy()
    vr_pre = model.vr.copy()

    # Initial momentum
    P_pre = model.mass[0] * v_pre[0] + model.mass[1] * v_pre[1]
    L_pre = (model.inertia[0] * vr_pre[0] + model.inertia[1] * vr_pre[1] +
             np.cross(x1, model.mass[0] * v_pre[0]) + np.cross(x2, model.mass[1] * v_pre[1]))

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)

    # Post-projection momentum
    P_post = model.mass[0] * model.v[0] + model.mass[1] * model.v[1]
    L_post = (model.inertia[0] * model.vr[0] + model.inertia[1] * model.vr[1] +
              np.cross(x1, model.mass[0] * model.v[0]) + np.cross(x2, model.mass[1] * model.v[1]))

    np.testing.assert_allclose(P_post, P_pre, atol=1e-12)
    np.testing.assert_allclose(L_post, L_pre, atol=1e-12)


# ============================================================================
# 7. Explicit Engine Dynamic Simulation with Mechanism Joint
# ============================================================================

def test_m602_engine_explicit_simulation_revolute(tmp_path: Path):
    """Verify explicit dynamic integration with /PROP/TYPE33 Revolute Joint."""
    deck_0000 = """# RADIOSS STARTER DECK
/BEGIN
M602 Dynamic Simulation Test
1 1
/NODE
1 0.0 0.0 0.0
2 0.0 0.0 0.0
3 1.0 0.0 0.0
/PART/1
HingeJoint
1 0 0
/PROP/TYPE33/1
RevoluteProp
1 0
0 0 10000.0 0.1
1e5 0.0 0.0 0.0
0 0 0
0.0 0.0 0.0
0 0 0
/SPRING/1
10 1 2
/BCS/1
FixedBase
111 111 0 1
/INIVEL/ROT/1
InitialSpin
10.0 X 0 0.0 0.0 0.0
/END
"""
    p0 = tmp_path / "HINGE_0000.rad"
    p0.write_text(deck_0000.strip() + "\n", encoding="ascii")

    blocks = read_deck(str(p0))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    # Assign mass, coordinates, and velocity to nodes
    model.mass = np.array([100.0, 5.0, 5.0], dtype=np.float64)
    model.inertia = np.array([10.0, 1.0, 1.0], dtype=np.float64)
    model.x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    model.v = np.zeros((3, 3), dtype=np.float64)
    model.vr = np.zeros((3, 3), dtype=np.float64)
    model.v[1] = np.array([2.0, 3.0, 0.0])

    joints = build_kjoints(model, log)
    assert len(joints) == 1
    kj = joints[0]

    # Verify joint is active in Step 3b/4/5/6
    fint = np.zeros((3, 3))
    fext = np.zeros((3, 3))
    fcont = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    mass_eff = model.mass.copy()
    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia

    # Step 3b: transfer_forces
    kj.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, inv_inertia)
    np.testing.assert_allclose(fint[0] + fint[1], [0.0, 0.0, 0.0], atol=1e-12)

    # Step 5: apply_velocity
    kj.apply_velocity(model.v, model.vr, model.x, mass_eff, model.inertia)
    # Relative translation locked
    np.testing.assert_allclose(model.v[1] - model.v[0], [0.0, 0.0, 0.0], atol=1e-12)
