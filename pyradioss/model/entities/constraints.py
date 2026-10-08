"""Boundary conditions, rigid bodies, MPCs and kinematic joints.

Split out of the former single-module ``pyradioss/model/entities.py`` (task
P1.4).  The blocks below are byte-identical to their originals, with one
mechanical exception: relative ``from . import`` statements gained a dot,
because code that was one module deep now sits one package level deeper.
Nothing else about any block changed.

This module is self-contained: the split was checked and leaves no cross-file
reference of any kind, so no sibling submodule is imported here.  A future
cross-file reference should be an ``if TYPE_CHECKING:`` import (annotations
are lazy here, via ``from __future__ import annotations``) unless the
reference is actually evaluated at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math

import numpy as np


# ============================================================================
# Loads & constraints
# ============================================================================

@dataclass
class BoundaryCondition:
    """/BCS: fixes translational/rotational DOFs of a node group.

    ``trarot`` is the classic Radioss 6-character flag string 'XYZ XYZ'
    (e.g. '111 000' fixes all translations); stored as two boolean triples.

    ``skew_id`` (M39): the DOFs are fixed in the axes of that /SKEW, not
    the global ones — the constraint condensation ROTATES with the skew.
    ``bcs1v`` (``engine/source/constraints/general/bcs/bcs1.F``) projects
    the component along each constrained skew axis out of both the
    acceleration and the velocity; with a /SKEW/MOV the axes are rebuilt
    every cycle, so the constraint plane turns with the nodes.  0 = the
    global system.
    """

    id: int
    grnod_id: int
    fix_tra: np.ndarray  # (3,) bool
    fix_rot: np.ndarray  # (3,) bool
    title: str = ""
    skew_id: int = 0
    skew_row: int = 0    # resolved SkewSet row (0 = global)


@dataclass
class AleBoundaryCondition:
    """/ALE/BCS: grid boundary conditions for ALE solver.
    
    Fixes the grid velocity (wx, wy, wz) and/or Lagrange conditions
    (lx, ly, lz) of a node group.
    """
    id: int
    grnod_id: int
    fix_w: np.ndarray  # (3,) bool for WX WY WZ
    fix_l: np.ndarray  # (3,) bool for LX LY LZ
    title: str = ""
    skew_id: int = 0
    skew_row: int = 0



@dataclass
class Mpc:
    """/MPC (M6): one general linear multi-point constraint row,

        sum_k  coef_k * u(node_k, dof_k) = 0        (dof 1-3 = X,Y,Z
                                                     translations,
                                                     4-6 = rotations)

    imposed on velocities/accelerations (its time derivative — exact for
    the homogeneous row, see engine/mpc.py for the Lagrange treatment).

    Fortran origin: ``starter/source/constraints/general/mpc/
    hm_read_mpc.F`` + ``engine/source/constraints/general/mpc/``.
    """

    id: int
    node_ids: List[int] = field(default_factory=list)
    dofs: List[int] = field(default_factory=list)      # 1..6 (user input)
    coefs: List[float] = field(default_factory=list)
    title: str = ""


@dataclass
class RigidBody:
    """/RBODY and /RBE2 (M5): a set of slave nodes moving as one rigid
    body, represented by a master node.

    Fortran origin: ``starter/source/constraints/general/rbody/hm_read_rbody.F``
    (input + mass/inertia assembly in ``rbyini.F``) and the engine update
    ``engine/source/constraints/general/rbody/rbyfor.F`` / ``rbycor.F``;
    /RBE2 is ``constraints/general/rbe2``. Both are the same mechanics —
    a 6-DOF rigid equation of motion fed by the gathered slave forces —
    and share this entity:

    * ``kind='RBODY'``: the classic rigid body. The master node is usually
      a standalone (massless) node; with ``icog=1`` (default, the Radioss
      ICoG behaviour) the Starter MOVES it to the computed center of
      gravity. ``added_mass``/``jadd`` are extra mass/inertia lumped at
      the COG (the Radioss Mass and Jxx/Jyy/Jzz fields).
    * ``kind='RBE2'``: a rigid link. The master is a structural node kept
      at its own position (never relocated); its own mass and the forces
      of the elements attached to it enter the body EOM, so a deformable
      structure can hang off the master. Only the full 6-DOF tie is
      ported (the per-DOF flags of the Radioss card are not).

    The Starter fills the resolved fields: dense indices, the total mass
    (slaves + master-if-structural + added), the COG and the 3x3 inertia
    tensor about it (from the slave point masses + nodal shell inertias +
    jadd). Element deletion does NOT change any of this: a deleted
    element's mass stays on its nodes (the Radioss convention, see
    initialization.py), so the rigid-body inertia is constant for the
    whole run — computed once here, never updated.
    """

    id: int
    kind: str                 # 'RBODY' | 'RBE2'
    master_id: int            # user node id of the master node
    grnod_id: int             # slave node group
    added_mass: float = 0.0   # /RBODY Mass field (at the COG)
    jadd: Optional[np.ndarray] = None   # (3,) added Jxx Jyy Jzz (at the COG)
    ispher: int = 0           # 1 = spherical inertia tensor (average of diagonals)
    icog: int = 1             # 1 = move master to COG (RBODY default)
    # /RBODY sens_ID: 0 = the body is ACTIVE from t=0; nonzero = a /SENSOR
    # gates it, so it starts INACTIVE.  This mirrors the reference's
    # NPBY(7,N) ON/OFF flag, set by hm_read_rbody.F exactly this way
    # ("IF(ISENS == 0) THEN NPBY(7,NRB)=1 ELSE NPBY(7,NRB)=0").  The port
    # does NOT gate the rigid-body kinematics by sensor (the field is
    # warned as ignored by read_rbody and engine/rigid_body.py never reads
    # it); it is carried because the Starter's shared-node check needs the
    # ACTIVE/INACTIVE distinction to match checkrby.F — see
    # starter/initialization.initialize_rigid_bodies (M39 / M38-NEW-4).
    sens_id: int = 0
    title: str = ""
    #: /RBODY Skew_ID (M39): the axes the card's Jxx/Jyy/Jzz are written
    #: in.  ``inirby.F`` calls CHBAS(SKEW(1,NOSKEW), RBY) ONCE, at Starter
    #: time, to rotate that tensor into the global frame — so only the
    #: skew's INITIAL orientation matters even for a /SKEW/MOV (the body
    #: then carries its own rotation).  0 = the global system.
    skew_id: int = 0
    skew_row: int = 0         # resolved SkewSet row (0 = global)
    lagmul: bool = False      # /RBODY/LAGMUL: Lagrange multiplier formulation (M131)
    # Resolved by the Starter (initialize_rigid_bodies):
    master: int = -1                      # dense node index
    slaves: Optional[np.ndarray] = None   # dense node indices (no master)
    mass_total: float = 0.0               # incl. added mass
    xg: Optional[np.ndarray] = None       # (3,) center of gravity
    J: Optional[np.ndarray] = None        # (3,3) inertia tensor about xg


@dataclass
class Rbe3:
    """/RBE3 (M5): interpolation constraint — the motion of one dependent
    (reference) node is the weighted average of a set of independent
    (master) nodes, and a force applied at the reference node is
    distributed to the masters *without adding any stiffness*.

    Fortran origin: ``starter/source/constraints/general/rbe3/hm_read_rbe3.F``
    + ``engine/source/constraints/general/rbe3/rbe3f.F`` (force
    distribution) / ``rbe3v.F`` (kinematic update). The port supports one
    master node group with uniform unit weights (the per-set weights and
    per-DOF flags of the full card are not ported); see
    pyradioss/engine/rbe3.py for the interpolation/distribution math.
    """

    id: int
    ref_id: int               # user node id of the dependent node
    grnod_id: int             # independent (master) nodes
    title: str = ""


@dataclass
class RigidWall:
    """/RWALL — rigid wall, kinematic treatment. Since M5 three geometries
    and moving walls are ported.

    Fortran: engine/source/constraints/general/rwall/ (``rgwal0.F`` plane,
    ``rgwals.F`` sphere, ``rgwalc.F`` cylinder, ``rgwalt.F`` the moving
    variants). A node that would end the cycle behind the wall surface has
    its normal velocity replaced so it lands exactly ON the surface
    (relative to the wall's own motion); slide=0 keeps the tangential
    velocity, slide=1 ties it to the wall, slide=2 applies Coulomb
    friction.

    Geometry (``geom``):

    * 'PLANE' — infinite plane through ``point`` with outward unit
      ``normal`` (nodes live on the +normal side);
    * 'SPHER' — sphere of ``radius`` centered at ``point`` (nodes live
      outside; per-node normal = radial direction);
    * 'CYL'   — infinite cylinder of ``radius`` about the axis through
      ``point`` along ``normal`` (nodes outside).

    Moving walls (``node_id`` > 0): the wall geometry is tied to that
    node — it translates with the node's displacement and pushes with the
    node's velocity, and the contact impulses REACT on the node (Radioss
    moving-wall convention: the node's mass, e.g. from /ADMAS, and its
    /INIVEL make a free flying wall; an /IMPVEL on the node makes a
    velocity-driven wall, in which case the reaction is absorbed by the
    drive instead). The wall does not rotate (the axis/normal direction
    is constant), like the original.
    """

    id: int
    point: np.ndarray    # (3,) plane point / sphere center / cyl axis point
    normal: np.ndarray   # (3,) plane outward normal / cylinder axis / paral normal
    slide: int = 0       # 0=sliding, 1=tied, 2=sliding with friction
    fric: float = 0.0
    grnod_id: Optional[int] = None  # None = all nodes are candidates
    grnod_id2: Optional[int] = None # excluded nodes
    dist: float = 0.0    # activation distance (search band), 0 = auto
    title: str = ""
    geom: str = "PLANE"  # 'PLANE' | 'SPHER' | 'CYL' | 'PARAL'
    radius: float = 0.0  # SPHER / CYL
    node_id: int = 0     # > 0: wall tied to this (user id) node — moving
    axis1: Optional[np.ndarray] = None  # (3,) PARAL first edge vector
    axis2: Optional[np.ndarray] = None  # (3,) PARAL second edge vector
    lagmul: bool = False                # True: global sparse Lagrange multiplier solver
    ifq: int = 0                        # filtering flag
    freq: float = 0.0                   # filtering factor / frequency
    alpha: float = 0.0                  # filtering factor
    mass: float = 0.0                   # wall carrier mass
    vx: float = 0.0                     # wall carrier initial x-velocity
    vy: float = 0.0                     # wall carrier initial y-velocity
    vz: float = 0.0                     # wall carrier initial z-velocity


@dataclass
class RwallBox:
    """/RWALL/BOX (M136): Rigid bounding box wall."""
    id: int
    title: str = ""
    node_id: int = 0
    slide: int = 0
    fric: float = 0.0
    grnod_id: int = 0
    grnod_id2: int = 0
    dist: float = 0.0
    p1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    p2: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class RwallCone:
    """/RWALL/CONE (M136): Rigid conical wall."""
    id: int
    title: str = ""
    node_id: int = 0
    slide: int = 0
    fric: float = 0.0
    grnod_id: int = 0
    grnod_id2: int = 0
    dist: float = 0.0
    apex: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    angle: float = 0.0


@dataclass
class LagmulGlobal:
    """/LAGMUL, /LAGMUL/OPTION (M131): Global Lagrange multiplier solver parameters."""
    lagmod: int = 1
    lagopt: int = 1
    tol: float = 1e-11
    alpha: float = 5e-4
    alpha_s: float = 0.0


@dataclass
class GearConstraint:
    """/GEAR, /LAGMUL/GEAR (M131): Rotational gear kinematic constraint."""
    id: int
    title: str = ""
    node1: int = 0
    node2: int = 0
    ratio: float = 1.0
    dir1: int = 1
    dir2: int = 1
    skew1: int = 0
    skew2: int = 0


@dataclass
class RackConstraint:
    """/RACK, /LAGMUL/RACK (M131): Rack-and-pinion kinematic constraint."""
    id: int
    title: str = ""
    node1: int = 0
    node2: int = 0
    pitch_radius: float = 1.0
    dir1: int = 1
    dir2: int = 1
    skew1: int = 0
    skew2: int = 0


@dataclass
class DiffConstraint:
    """/DIFF, /LAGMUL/DIFF (M131): Differential rotational kinematic constraint."""
    id: int
    title: str = ""
    node0: int = 0
    node1: int = 0
    node2: int = 0
    ratio: float = 1.0


# ----------------------------------------------------------------------------
# Boundary conditions & joints & special initial states (M102)
# ----------------------------------------------------------------------------

@dataclass
class BcsNrf:
    """/BCS/NRF (M102, M200): Non-reflecting boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_nrf.F90``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    set_id: int = 0
    iskep: int = 0
    frame_id: int = 0
    isurf: int = 0
    ivel: int = 0
    isub: int = 0
    ityp: int = 0
    factor: float = 0.0
    rho: float = 0.0
    cp: float = 0.0
    cs: float = 0.0

    def __post_init__(self):
        if not self.grnod_id and self.set_id:
            self.grnod_id = self.set_id
        elif not self.set_id and self.grnod_id:
            self.set_id = self.grnod_id


@dataclass
class BcsWall:
    """/BCS/WALL (M102): Sliding wall boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_wall.F90``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    sensor_id: int = 0
    tstart: float = 0.0
    tstop: float = 0.0
    set_id: int = 0

    def __post_init__(self):
        if not self.grnod_id and self.set_id:
            self.grnod_id = self.set_id
        elif not self.set_id and self.grnod_id:
            self.set_id = self.grnod_id

    @property
    def sens_id(self) -> int:
        return self.sensor_id

    @sens_id.setter
    def sens_id(self, val: int) -> None:
        self.sensor_id = val


@dataclass
class RigidLink:
    """/RLINK (M102/M152): Standard rigid link definition between node group and main/skew frame.

    Fortran origin: ``starter/source/constraints/rigidlink/hm_read_rlink.F``.
    """
    id: int
    title: str = ""
    dofs: Tuple[int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1)
    skew_id: int = 0
    grnod_id: int = 0
    ipol: int = 0
    node_ids: List[int] = field(default_factory=list)

    @property
    def tx(self) -> int:
        return self.dofs[0] if len(self.dofs) > 0 else 1

    @property
    def ty(self) -> int:
        return self.dofs[1] if len(self.dofs) > 1 else 1

    @property
    def tz(self) -> int:
        return self.dofs[2] if len(self.dofs) > 2 else 1

    @property
    def rx(self) -> int:
        return self.dofs[3] if len(self.dofs) > 3 else 1

    @property
    def ry(self) -> int:
        return self.dofs[4] if len(self.dofs) > 4 else 1

    @property
    def rz(self) -> int:
        return self.dofs[5] if len(self.dofs) > 5 else 1


@dataclass
class ExternLink:
    """/EXTERN/LINK or /EXTLNK (M167): External process coupling link.

    Fortran origin: ``starter/source/coupling/rad2rad/lecextlnk.F`` and
    ``hm_cfg_files/config/CFG/radioss2022/RAD2R/extlnk.cfg``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0


@dataclass
class CylJoint:
    """/CYL_JOINT (M102, M209): Cylindrical joint constraint between independent and dependent nodes."""
    id: int
    title: str = ""
    node_id1: int = 0
    node_id2: int = 0
    grnod_id: int = 0
    node1: int = 0
    node2: int = 0
    axis_dir: int = 1
    skew_id: int = 0
    tol: float = 1e-6
    secondary_nodes: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.node1 and self.node_id1:
            self.node1 = self.node_id1
        elif not self.node_id1 and self.node1:
            self.node_id1 = self.node1
        if not self.node2 and self.node_id2:
            self.node2 = self.node_id2
        elif not self.node_id2 and self.node2:
            self.node_id2 = self.node2


@dataclass
class GJoint:
    """/GJOINT (M102, M594): General kinematic mechanism joint (GEAR, DIFF, RACK, CV).

    Fortran origin: ``starter/source/constraints/general/gjoint/hm_read_gjoint.F``,
    ``engine/source/tools/lagmul/lag_gjnt.F``, ``gjnt_gear.F``, ``gjnt_diff.F``, ``gjnt_rack.F``.
    """
    id: int
    title: str = ""
    subtype: str = "DEFAULT"  # DEFAULT, GEAR, RACK, DIFF, CV
    node_id0: int = 0
    fscale: float = 1.0
    mass0: float = 0.0
    inertia0: float = 0.0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    mass1: float = 0.0
    inertia1: float = 0.0
    r1: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    mass2: float = 0.0
    inertia2: float = 0.0
    r2: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    mass3: float = 0.0
    inertia3: float = 0.0
    r3: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    @property
    def alpha(self) -> float:
        return self.fscale


GeneralJoint = GJoint


@dataclass
class KJoint:
    """/PROP/TYPE33, /PROP/TYPE45, /PROP/KJOINT, /PROP/KJOINT2 kinematic mechanism joint (M602)."""
    id: int
    node1: int
    node2: int
    prop_id: int = 0
    joint_type: int | str = 1
    title: str = ""
    skew_id: int = 0
    kn: float = 0.0
    cr: float = 0.0
    scale: float = 1.0
    ktx: float = 0.0
    kty: float = 0.0
    ktz: float = 0.0
    krx: float = 0.0
    kry: float = 0.0
    krz: float = 0.0
    ctx: float = 0.0
    cty: float = 0.0
    ctz: float = 0.0
    crx: float = 0.0
    cry: float = 0.0
    crz: float = 0.0
    prop: Any = None



@dataclass
class MergeNode:
    """/MERGE/NODE (M102): Merge nodes in node group within tolerance.

    Fortran origin: ``starter/source/constraints/general/merge/hm_read_merge.F``.
    """
    id: int
    title: str = ""
    tol: float = 0.0
    grnod_id: int = 0
    merge_type: int = 0


@dataclass
class MergeRbody:
    """/MERGE/RBODY (M102, M196): Merge rigid bodies.

    Fortran origin: ``starter/source/constraints/general/merge/hm_read_merge.F``.
    """
    id: int
    title: str = ""
    items: List[Tuple[int, int, int, int, int]] = field(default_factory=list)  # (main_id, m_type, secon_id, s_type, iflag)
    rbody_master_id: int = 0
    rbody_slave_ids: List[int] = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class IniCrackSegment:
    """Segment definition for /INICRACK."""
    node_id1: int
    node_id2: int
    ratio: float = 0.0


@dataclass
class IniCrack:
    """/INICRACK (M102, M143): Initial crack geometric definition for X-FEM / cohesive elements.

    Fortran origin: ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F``.
    """
    id: int
    title: str = ""
    segments: List[IniCrackSegment] = field(default_factory=list)
    grsh_id: int = 0
    p1: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    p2: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    norm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    open_flag: int = 0


@dataclass
class RwallTherm:
    """/RWALL/THERM (M112): Thermal rigid wall.

    Fortran origin: ``starter/source/constraints/general/rwall/hm_read_rwall_therm.F``.
    """
    id: int
    title: str = ""
    typ: int = 1
    tied: int = 0
    node_id: int = 0
    grnod_id1: int = 0
    grnod_id2: int = 0
    fct_id: int = 0
    temp: float = 0.0
    tstif: float = 0.0
    fric: float = 0.0


@dataclass
class NbcsNode:
    """Single node DOF constraint entry in /NBCS."""
    tx: int = 0
    ty: int = 0
    tz: int = 0
    wx: int = 0
    wy: int = 0
    wz: int = 0
    skew_id: int = 0
    node_id: int = 0
    tra: Any = None
    rot: Any = None
    active: bool = True

    def __post_init__(self):
        if self.tra is not None and len(self.tra) >= 3:
            self.tx = int(self.tra[0])
            self.ty = int(self.tra[1])
            self.tz = int(self.tra[2])
        if self.rot is not None and len(self.rot) >= 3:
            self.wx = int(self.rot[0])
            self.wy = int(self.rot[1])
            self.wz = int(self.rot[2])


@dataclass
class NbcsBlock:
    """/NBCS/id (M119): Non-linear boundary conditions block."""
    id: int
    title: str = ""
    nodes: list[NbcsNode] = field(default_factory=list)


@dataclass
class AleGridConstraint:
    """/ALE/GRID/DISP or /ALE/GRID/VEL (M169): ALE grid nodal boundary constraint.

    Fortran origin: ``starter/source/constraints/ale/hm_read_ale_grid.F``.
    """
    id: int
    kind: str = "DISP"  # 'DISP' | 'VEL'
    title: str = ""
    grnod_id: int = 0
    fun_id: int = 0
    skew_id: int = 0
    tra_code: str = ""
    scale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30


# ============================================================================
# M178: BCS_CYCLIC, LOAD_PCYL, EBCS_MONVOL
# ============================================================================

@dataclass
class BcsCyclic:
    """/BCS/CYCLIC (M178): Cyclic symmetry boundary condition coupling two node groups in a skew coordinate system."""
    id: int
    skew_id: int = 0
    grnd_id1: int = 0
    grnd_id2: int = 0
    title: str = ""



@dataclass
class BcsLagmul:
    """``/BCS/LAGMUL/id`` (M202): Lagrange multiplier constraint on node group."""
    id: int = 0
    title: str = ""
    tra: Any = "111"
    rot: Any = "111"
    skew_id: int = 0
    grnod_id: int = 0



LagmulGear = GearConstraint
LagmulRack = RackConstraint
LagmulDiff = DiffConstraint


@dataclass
class BallJoint:
    """``/LAGMUL/BALL_JOINT`` or ``/BALL_JOINT/id`` (M208): Spherical kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    tol: float = 1e-6


@dataclass
class PinJoint:
    """``/LAGMUL/PIN_JOINT`` or ``/PIN_JOINT/id`` (M208): Revolute pin kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis_dir: int = 1        # 1: X, 2: Y, 3: Z
    skew_id: int = 0
    tol: float = 1e-6


@dataclass
class SliderJoint:
    """``/LAGMUL/SLIDER`` or ``/SLIDER/id`` (M209): Prismatic slider kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis_dir: int = 1        # 1: X, 2: Y, 3: Z
    skew_id: int = 0
    tol: float = 1e-6

@dataclass
class PlanarJoint:
    """``/LAGMUL/PLANAR`` or ``/PLANAR/id`` (M210): Planar kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis_dir: int = 3        # normal axis: 1: X, 2: Y, 3: Z
    skew_id: int = 0
    tol: float = 1e-6


@dataclass
class CardanJoint:
    """``/LAGMUL/CARDAN`` or ``/CARDAN/id`` (M210): Cardan/Universal kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis_dir: int = 1        # primary axis: 1: X, 2: Y, 3: Z
    skew_id: int = 0
    tol: float = 1e-6


@dataclass
class RigidJoint:
    """``/LAGMUL/RIGID`` or ``/RIGID_JOINT/id`` (M211): Rigid link kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    tol: float = 1e-6


@dataclass
class ScrewJoint:
    """``/LAGMUL/SCREW`` or ``/SCREW/id`` (M211): Helical screw kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis_dir: int = 1        # helical axis: 1: X, 2: Y, 3: Z
    skew_id: int = 0
    pitch: float = 1.0       # helical pitch ratio
    tol: float = 1e-6


@dataclass
class CvJoint:
    """``/LAGMUL/CV_JOINT`` or ``/CV_JOINT/id`` (M212): Constant velocity / homokinetic kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis_dir: int = 1        # primary axis: 1: X, 2: Y, 3: Z
    skew_id: int = 0
    tol: float = 1e-6


@dataclass
class InlineJoint:
    """``/LAGMUL/INLINE`` or ``/INLINE/id`` (M212): In-line kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis_dir: int = 1        # line axis: 1: X, 2: Y, 3: Z
    skew_id: int = 0
    tol: float = 1e-6


@dataclass
class ParallelJoint:
    """``/LAGMUL/PARALLEL`` or ``/PARALLEL/id`` (M213): Parallel axes kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis_dir: int = 1        # parallel axis: 1: X, 2: Y, 3: Z
    skew_id: int = 0
    tol: float = 1e-6


@dataclass
class PerpendicularJoint:
    """``/LAGMUL/PERPENDICULAR`` or ``/PERPENDICULAR/id`` (M213): Perpendicular axes kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis1_dir: int = 1       # primary axis 1: 1: X, 2: Y, 3: Z
    axis2_dir: int = 2       # primary axis 2: 1: X, 2: Y, 3: Z
    skew1_id: int = 0
    skew2_id: int = 0
    tol: float = 1e-6


@dataclass
class GimbalJoint:
    """``/LAGMUL/GIMBAL`` or ``/GIMBAL/id`` (M214): Gimbal / 2-DOF universal kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    axis1_dir: int = 1       # free rotation axis 1: 1: X, 2: Y, 3: Z
    axis2_dir: int = 2       # free rotation axis 2: 1: X, 2: Y, 3: Z
    skew1_id: int = 0
    skew2_id: int = 0
    tol: float = 1e-6


@dataclass
class DistanceJoint:
    """``/LAGMUL/DISTANCE`` or ``/DISTANCE/id`` (M214): Constant distance kinematic joint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    dist: float = 0.0        # fixed distance (<= 0 implies computed from initial coordinates)
    tol: float = 1e-6


@dataclass
class SlotJoint:
    """``/LAGMUL/SLOT/joint_ID`` or ``/SLOT/joint_ID`` (M217): Multibody slot kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    skew_id: int = 0
    axis_dir: int = 1        # 1: X, 2: Y, 3: Z along the slot line
    d_min: float = 0.0       # minimum slot limit
    d_max: float = 0.0       # maximum slot limit


@dataclass
class RackPinionJoint:
    """``/RACK_PINION/id`` or ``/LAGMUL/RACK_PINION/id`` (M250): Rack and pinion kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0           # rack node (translational)
    node2: int = 0           # pinion node (rotational)
    pitch_radius: float = 1.0 # pitch radius R
    axis_dir: int = 1        # motion axis direction (1=X, 2=Y, 3=Z)
    skew_id: int = 0         # reference skew ID
    tol: float = 1e-6        # kinematic constraint tolerance


@dataclass
class BeltPulleyJoint:
    """``/BELT_PULLEY/id`` or ``/LAGMUL/BELT_PULLEY/id`` (M251): Belt and pulley kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0           # driver pulley shaft node
    node2: int = 0           # driven pulley shaft node
    radius1: float = 1.0     # driver pulley pitch radius R1
    radius2: float = 1.0     # driven pulley pitch radius R2
    axis1_dir: int = 1       # driver pulley axis (1=X, 2=Y, 3=Z)
    axis2_dir: int = 1       # driven pulley axis (1=X, 2=Y, 3=Z)
    skew_id: int = 0         # reference skew ID
    tol: float = 1e-6        # kinematic constraint tolerance


@dataclass
class OldhamJoint:
    """``/OLDHAM/id`` or ``/LAGMUL/OLDHAM/id`` (M252): Oldham parallel offset coupling kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0           # driver shaft node
    node2: int = 0           # driven shaft node
    axis_dir: int = 1        # shaft rotation axis (1=X, 2=Y, 3=Z)
    skew_id: int = 0         # reference skew ID
    tol: float = 1e-6        # kinematic constraint tolerance


@dataclass
class TripodJoint:
    """``/TRIPOD/id`` or ``/LAGMUL/TRIPOD/id`` (M253): Tripod plunging CV kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0           # driver tripod housing node
    node2: int = 0           # driven tripod tulip shaft node
    axis_dir: int = 1        # shaft rotation and plunge axis (1=X, 2=Y, 3=Z)
    skew_id: int = 0         # reference skew ID
    plunge_limit: float = 0.0 # maximum allowable axial plunge travel
    tol: float = 1e-6        # kinematic constraint tolerance




@dataclass
class BevelGearJoint:
    """``/BEVEL_GEAR/id`` or ``/LAGMUL/BEVEL_GEAR/id`` (M254): Bevel gear kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0           # driver shaft node
    node2: int = 0           # driven shaft node
    ratio: float = 1.0       # gear velocity ratio gamma = omega2 / omega1
    skew1_id: int = 0        # driver shaft reference skew ID
    skew2_id: int = 0        # driven shaft reference skew ID
    tol: float = 1e-6        # kinematic constraint tolerance





@dataclass
class WormGearJoint:
    """``/WORM_GEAR/id`` or ``/LAGMUL/WORM_GEAR/id`` (M255): Worm and worm gear kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0           # worm screw shaft node
    node2: int = 0           # worm wheel gear node
    ratio: float = 1.0       # speed reduction ratio gamma = Nteeth / Nthreads
    skew1_id: int = 0        # worm shaft reference skew ID
    skew2_id: int = 0        # worm wheel reference skew ID
    tol: float = 1e-6        # kinematic constraint tolerance


@dataclass
class HypoidGearJoint:
    """``/HYPOID_GEAR/id`` or ``/LAGMUL/HYPOID_GEAR/id`` (M256): Hypoid gear kinematic joint constraint with shaft offset."""
    id: int = 1
    title: str = ""
    node1: int = 0               # pinion shaft node
    node2: int = 0               # ring gear node
    ratio: float = 1.0           # velocity ratio gamma = omega2 / omega1
    offset: float = 0.0          # shaft hypoid offset distance E
    skew1_id: int = 0            # pinion reference skew ID
    skew2_id: int = 0            # ring gear reference skew ID
    tol: float = 1e-6            # kinematic constraint tolerance


@dataclass
class EpicyclicGearJoint:
    """``/EPICYCLIC_GEAR/id`` or ``/LAGMUL/EPICYCLIC_GEAR/id`` (M257): Epicyclic planetary gear kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # sun gear node
    node2: int = 0               # planet carrier node
    node3: int = 0               # ring gear node
    ratio_sun: float = 1.0       # sun gear ratio / tooth count N_s
    ratio_ring: float = 1.0      # ring gear ratio / tooth count N_r
    axis_dir: int = 1            # rotation axis direction (1=X, 2=Y, 3=Z)
    skew_id: int = 0             # reference skew ID
    tol: float = 1e-6            # kinematic constraint tolerance


@dataclass
class LagmulHarmonicDrive:
    """``/HARMONIC_DRIVE/id`` or ``/LAGMUL/HARMONIC_DRIVE/id`` (M259): Harmonic drive strain wave gear kinematic constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # wave generator input node
    node2: int = 0               # flexspline output node
    node3: int = 0               # circular spline ground node
    ratio: float = 100.0         # gear reduction ratio R
    stiff: float = 1e6           # torsional stiffness
    axis_x: float = 0.0          # rotation axis vector X
    axis_y: float = 0.0          # rotation axis vector Y
    axis_z: float = 1.0          # rotation axis vector Z
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance


@dataclass
class LagmulCycloidalDrive:
    """``/CYCLOIDAL_DRIVE/id`` or ``/LAGMUL/CYCLOIDAL_DRIVE/id`` (M260): Cycloidal speed reducer kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # eccentric shaft input node
    node2: int = 0               # cycloidal disc output node
    node3: int = 0               # ring pin housing ground node
    ratio: float = 29.0          # reduction gear ratio R = (P - 1)
    stiff: float = 1e6           # torsional contact stiffness
    axis_x: float = 0.0          # rotation axis vector X
    axis_y: float = 0.0          # rotation axis vector Y
    axis_z: float = 1.0          # rotation axis vector Z
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance


# ============================================================================
# M261 Suite: Syazwan failure, EngTemperature, RackPinion, SensorSpringForceRate
# ============================================================================

# FailSyazwan: canonical definition is above (M126 section) with M261 fields merged in.





@dataclass
class LagmulRackPinion:
    """``/RACK_AND_PINION/id`` or ``/LAGMUL/RACK_AND_PINION/id`` (M261): Rack and pinion transmission kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # pinion rotation node
    node2: int = 0               # rack translation node
    pitch_radius: float = 10.0   # pinion pitch circle radius R
    stiff: float = 1e6           # kinematic transmission contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_rot_x: float = 0.0      # pinion rotation axis vector X
    axis_rot_y: float = 0.0      # pinion rotation axis vector Y
    axis_rot_z: float = 1.0      # pinion rotation axis vector Z
    axis_tra_x: float = 1.0      # rack translation axis vector X
    axis_tra_y: float = 0.0      # rack translation axis vector Y
    axis_tra_z: float = 0.0      # rack translation axis vector Z


@dataclass
class LagmulScrewJoint:
    """``/SCREW_JOINT/id`` or ``/LAGMUL/SCREW_JOINT/id`` (M262): Screw and leadscrew transmission kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # rotating screw node
    node2: int = 0               # translating nut node
    lead_pitch: float = 5.0      # screw lead pitch L (linear displacement per 2*pi revolution)
    stiff: float = 1e6           # kinematic thread contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # screw rotational / translation axis vector X
    axis_y: float = 0.0          # screw rotational / translation axis vector Y
    axis_z: float = 1.0          # screw rotational / translation axis vector Z


@dataclass
class LagmulDifferentialGear:
    """``/DIFFERENTIAL_GEAR/id`` or ``/LAGMUL/DIFFERENTIAL_GEAR/id`` (M263): Differential gear train kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # carrier / pinion input node
    node2: int = 0               # left axle output node
    node3: int = 0               # right axle output node
    ratio: float = 1.0           # final drive differential gear reduction ratio
    stiff: float = 1e6           # kinematic bevel contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # differential rotation axis vector X
    axis_y: float = 0.0          # differential rotation axis vector Y
    axis_z: float = 1.0          # differential rotation axis vector Z


@dataclass
class LagmulTransferCase:
    """``/TRANSFER_CASE/id`` or ``/LAGMUL/TRANSFER_CASE/id`` (M264): 4WD/AWD Transfer case transmission kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # transmission drive input node
    node2: int = 0               # front axle output node
    node3: int = 0               # rear axle output node
    front_split: float = 0.5     # front axle nominal torque split fraction
    stiff: float = 1e6           # kinematic center differential contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # transfer case shaft axis vector X
    axis_y: float = 0.0          # transfer case shaft axis vector Y
    axis_z: float = 1.0          # transfer case shaft axis vector Z


@dataclass
class LagmulTorqueSplitGear:
    """``/TORQUE_SPLIT_GEAR/id`` or ``/LAGMUL/TORQUE_SPLIT_GEAR/id`` (M265): Dual-output torque splitter / PTO kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving input shaft node
    node2: int = 0               # primary output shaft node
    node3: int = 0               # secondary output shaft node
    split_ratio: float = 0.5     # secondary output torque distribution fraction
    stiff: float = 1e6           # kinematic splitter contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # torque splitter rotation axis vector X
    axis_y: float = 0.0          # torque splitter rotation axis vector Y
    axis_z: float = 1.0          # torque splitter rotation axis vector Z


@dataclass
class LagmulGenevaDrive:
    """``/GENEVA_DRIVE/id`` or ``/LAGMUL/GENEVA_DRIVE/id`` (M266): Geneva drive intermittent rotary indexing kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # continuous driving crank pin node
    node2: int = 0               # intermittent driven Geneva wheel node
    num_slots: int = 4           # number of radial indexing slots (n >= 3)
    stiff: float = 1e6           # kinematic pin-slot contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # Geneva drive rotation axis vector X
    axis_y: float = 0.0          # Geneva drive rotation axis vector Y
    axis_z: float = 1.0          # Geneva drive rotation axis vector Z


@dataclass
class LagmulScotchYoke:
    """``/SCOTCH_YOKE/id`` or ``/LAGMUL/SCOTCH_YOKE/id`` (M267): Scotch yoke pure harmonic rotary-to-linear conversion kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # rotating crank node
    node2: int = 0               # reciprocating slider node
    crank_radius: float = 1.0    # crank pin radius R
    stiff: float = 1e6           # kinematic slider slot contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    rot_x: float = 0.0           # crank rotation axis vector X
    rot_y: float = 0.0           # crank rotation axis vector Y
    rot_z: float = 1.0           # crank rotation axis vector Z
    trans_x: float = 1.0         # slider translation axis vector X
    trans_y: float = 0.0         # slider translation axis vector Y
    trans_z: float = 0.0         # slider translation axis vector Z


@dataclass
class LagmulOldhamCoupling:
    """``/OLDHAM_COUPLING/id`` or ``/LAGMUL/OLDHAM_COUPLING/id`` (M268): Oldham coupling parallel offset shaft kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving input shaft node
    node2: int = 0               # driven output shaft node
    node3: int = 0               # floating central slider disc node
    stiff: float = 1e6           # kinematic slot guide contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # shaft rotation axis vector X
    axis_y: float = 0.0          # shaft rotation axis vector Y
    axis_z: float = 1.0          # shaft rotation axis vector Z


@dataclass
class LagmulSchmidtCoupling:
    """``/SCHMIDT_COUPLING/id`` or ``/LAGMUL/SCHMIDT_COUPLING/id`` (M269): Schmidt (double-Cardan) coupling constant-velocity shaft joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving input shaft node
    node2: int = 0               # driven output shaft node
    node3: int = 0               # intermediate linkage center node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # shaft rotation axis vector X
    axis_y: float = 0.0          # shaft rotation axis vector Y
    axis_z: float = 1.0          # shaft rotation axis vector Z


@dataclass
class LagmulRzeppaJoint:
    """``/RZEPPA_JOINT/id`` or ``/LAGMUL/RZEPPA_JOINT/id`` (M270): Rzeppa constant-velocity ball joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving input shaft node
    node2: int = 0               # driven output shaft node
    node3: int = 0               # ball cage center node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # shaft rotation axis vector X
    axis_y: float = 0.0          # shaft rotation axis vector Y
    axis_z: float = 1.0          # shaft rotation axis vector Z


# EngHourglassEnergy: canonical definition is above (M245 section) with dt_hg alias.



@dataclass
class LagmulBirfieldJoint:
    """``/BIRFIELD_JOINT/id`` or ``/LAGMUL/BIRFIELD_JOINT/id`` (M271): Birfield (plunging CV) joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving input shaft node
    node2: int = 0               # driven output shaft node
    node3: int = 0               # plunging sleeve center node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # shaft rotation axis vector X
    axis_y: float = 0.0          # shaft rotation axis vector Y
    axis_z: float = 1.0          # shaft rotation axis vector Z


# EngContactEnergy: canonical definition is above (M246 section) with dt_contact alias.



@dataclass
class LagmulTripodJoint:
    """``/TRIPOD_JOINT/id`` or ``/LAGMUL/TRIPOD_JOINT/id`` (M272): Tripod (tulip/spider) joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving tulip hub node
    node2: int = 0               # driven spider shaft node
    node3: int = 0               # roller trunnion center node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # shaft rotation axis vector X
    axis_y: float = 0.0          # shaft rotation axis vector Y
    axis_z: float = 1.0          # shaft rotation axis vector Z


@dataclass
class LagmulHookeJoint:
    """``/HOOKE_JOINT/id`` or ``/LAGMUL/HOOKE_JOINT/id`` (M273): Hooke (universal/Cardan) joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0              # driving cross pin node
    node2: int = 0              # driven cross pin node
    node3: int = 0              # spider/cross center node
    stiff: float = 1e6          # kinematic constraint contact stiffness
    skew_id: int = 0            # reference coordinate frame ID
    tol: float = 1e-6           # constraint numerical tolerance
    axis_x: float = 0.0         # shaft rotation axis vector X
    axis_y: float = 0.0         # shaft rotation axis vector Y
    axis_z: float = 1.0         # shaft rotation axis vector Z


@dataclass
class EngJointEnergy:
    """``/ENG/JOINT_ENERGY`` or ``/ENG/JNT_ENERGY`` (M274): Engine joint energy history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_joint: float = 0.0      # time frequency for joint energy output
    sens_id: int = 0            # sensor activation ID


@dataclass
class LagmulTractaJoint:
    """``/TRACTA_JOINT/id`` or ``/LAGMUL/TRACTA_JOINT/id`` (M274): Tracta (sliding-yoke) joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0              # driving yoke node
    node2: int = 0              # driven yoke node
    node3: int = 0              # sliding center node
    stiff: float = 1e6          # kinematic constraint contact stiffness
    skew_id: int = 0            # reference coordinate frame ID
    tol: float = 1e-6           # constraint numerical tolerance
    axis_x: float = 0.0         # shaft rotation axis vector X
    axis_y: float = 0.0         # shaft rotation axis vector Y
    axis_z: float = 1.0         # shaft rotation axis vector Z


@dataclass
class LagmulThompsonCoupling:
    """``/THOMPSON_COUPLING/id`` or ``/LAGMUL/THOMPSON_COUPLING/id`` (M275): Thompson constant-velocity joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0              # driving shaft node
    node2: int = 0              # driven shaft node
    node3: int = 0              # spherical linkage / pantograph center node
    stiff: float = 1e6          # kinematic constraint contact stiffness
    skew_id: int = 0            # reference coordinate frame ID
    tol: float = 1e-6           # constraint numerical tolerance
    axis_x: float = 0.0         # shaft rotation axis vector X
    axis_y: float = 0.0         # shaft rotation axis vector Y
    axis_z: float = 1.0         # shaft rotation axis vector Z


@dataclass
class LagmulWeissJoint:
    """``/WEISS_JOINT/id`` or ``/LAGMUL/WEISS_JOINT/id`` (M276): Weiss constant-velocity ball-and-groove joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0              # driving yoke node
    node2: int = 0              # driven yoke node
    node3: int = 0              # ball groove center node
    stiff: float = 1e6          # kinematic constraint contact stiffness
    skew_id: int = 0            # reference coordinate frame ID
    tol: float = 1e-6           # constraint numerical tolerance
    axis_x: float = 0.0         # shaft rotation axis vector X
    axis_y: float = 0.0         # shaft rotation axis vector Y
    axis_z: float = 1.0         # shaft rotation axis vector Z


@dataclass
class LagmulTripodBallJoint:
    """``/TRIPOD_BALL_JOINT/id`` or ``/LAGMUL/TRIPOD_BALL_JOINT/id`` (M277): Tripod-ball kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving tulip hub node
    node2: int = 0               # driven tripod spider node
    node3: int = 0               # ball sphere center node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # shaft rotation axis vector X
    axis_y: float = 0.0          # shaft rotation axis vector Y
    axis_z: float = 1.0          # shaft rotation axis vector Z


@dataclass
class LagmulClevisJoint:
    """``/CLEVIS_JOINT/id`` or ``/LAGMUL/CLEVIS_JOINT/id`` (M278): Clevis pin / fork-and-tang kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving fork node
    node2: int = 0               # driven tang node
    node3: int = 0               # clevis pin center node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # pin pivot rotation axis vector X
    axis_y: float = 0.0          # pin pivot rotation axis vector Y
    axis_z: float = 1.0          # pin pivot rotation axis vector Z


@dataclass
class LagmulPinInSlotJoint:
    """``/PIN_IN_SLOT_JOINT/id`` or ``/LAGMUL/PIN_IN_SLOT_JOINT/id`` (M279): Pin-in-slot planar mechanism kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving slotted link node
    node2: int = 0               # driven pin follower node
    node3: int = 0               # slot curve reference node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # slot normal / guide axis vector X
    axis_y: float = 0.0          # slot normal / guide axis vector Y
    axis_z: float = 1.0          # slot normal / guide axis vector Z


@dataclass
class LagmulSliderSlotJoint:
    """``/SLIDER_SLOT_JOINT/id`` or ``/LAGMUL/SLIDER_SLOT_JOINT/id`` (M280): Slider-slot planar mechanism kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving slotted guide node
    node2: int = 0               # driven slider block node
    node3: int = 0               # slot track reference node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # slot orientation / guide axis vector X
    axis_y: float = 0.0          # slot orientation / guide axis vector Y
    axis_z: float = 1.0          # slot orientation / guide axis vector Z


@dataclass
class LagmulParallelAxisJoint:
    """``/PARALLEL_AXIS_JOINT/id`` or ``/LAGMUL/PARALLEL_AXIS_JOINT/id`` (M281): Parallel-axis slider / Oldham coupling kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving axis hub node
    node2: int = 0               # driven parallel axis hub node
    node3: int = 0               # intermediate floating slider disc node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # parallel shaft alignment axis vector X
    axis_y: float = 0.0          # parallel shaft alignment axis vector Y
    axis_z: float = 1.0          # parallel shaft alignment axis vector Z


@dataclass
class LagmulCamFollowerJoint:
    """``/CAM_FOLLOWER_JOINT/id`` or ``/LAGMUL/CAM_FOLLOWER_JOINT/id`` (M282): Cam and follower profile kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # cam drive shaft / pivot center node
    node2: int = 0               # follower roller / slider contact tip node
    node3: int = 0               # base frame / guide track reference node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    axis_x: float = 0.0          # cam rotation / follower stroke normal vector X
    axis_y: float = 0.0          # cam rotation / follower stroke normal vector Y
    axis_z: float = 1.0          # cam rotation / follower stroke normal vector Z


@dataclass
class LagmulScrewNutJoint:
    """``/SCREW_NUT_JOINT/id`` or ``/LAGMUL/SCREW_NUT_JOINT/id`` (M283): Lead screw and nut helical rotary-to-linear conversion kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # lead screw spindle drive node
    node2: int = 0               # translating nut contact node
    node3: int = 0               # guide base / support frame node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    pitch: float = 0.0           # thread pitch (axial advance per thread)
    lead: float = 0.0            # screw lead (axial advance per full turn)
    axis_z: float = 1.0          # screw helical rotation / translation axis vector Z


@dataclass
class LagmulGenevaJoint:
    """``/GENEVA_JOINT/id`` or ``/LAGMUL/GENEVA_JOINT/id`` (M284): Geneva wheel / Maltese cross intermittent rotary indexing kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # drive wheel / pin crank spindle node
    node2: int = 0               # driven Geneva cross slotted wheel node
    node3: int = 0               # support chassis / housing frame node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    num_slots: int = 4           # number of radial slots in Geneva wheel
    crank_radius: float = 0.0    # driving crank pin radius
    axis_z: float = 1.0          # indexing rotational axis vector Z


@dataclass
class LagmulCablePulleyJoint:
    """``/CABLE_PULLEY_JOINT/id`` or ``/LAGMUL/CABLE_PULLEY_JOINT/id`` (M285): Flexible cable and pulley wrapping transmission kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # cable entry tangency node
    node2: int = 0               # cable exit tangency node
    node3: int = 0               # pulley hub / center axle frame node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    pulley_radius: float = 0.0   # pulley pitch radius
    wrap_angle: float = 180.0    # cable wrapping contact angle (degrees)
    axis_z: float = 1.0          # pulley rotational axle vector Z


@dataclass
class LagmulSwashPlateJoint:
    """``/SWASH_PLATE_JOINT/id`` or ``/LAGMUL/SWASH_PLATE_JOINT/id`` (M286): Swash plate cyclic tilting and rotating kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # rotating swash plate ring node
    node2: int = 0               # non-rotating stationary swash plate ring node
    node3: int = 0               # mast / drive shaft frame support node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    plate_radius: float = 0.0    # swash plate pitch radius
    tilt_angle: float = 0.0      # nominal cyclic pitch tilt angle (degrees)
    axis_z: float = 1.0          # rotor mast rotation axis vector Z


@dataclass
class LagmulScissorMechanismJoint:
    """``/SCISSOR_MECHANISM_JOINT/id`` or ``/LAGMUL/SCISSOR_MECHANISM_JOINT/id`` (M287): Pantograph / scissor lift planar crossing linkage kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # first scissor arm crossing node
    node2: int = 0               # second scissor arm crossing node
    node3: int = 0               # central scissor pivot hinge pin node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    arm_length: float = 0.0      # total scissor link arm length
    initial_angle: float = 45.0  # nominal scissor opening scissor angle (degrees)
    axis_z: float = 1.0          # scissor mechanism normal plane vector Z


@dataclass
class LagmulParallelogramJoint:
    """``/PARALLELOGRAM_JOINT/id`` or ``/LAGMUL/PARALLELOGRAM_JOINT/id`` (M288): 4-bar parallelogram kinematic linkage joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # first driving crank link node
    node2: int = 0               # second driven parallel follower link node
    node3: int = 0               # fixed frame ground reference node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_length: float = 0.0     # primary link arm length
    link_width: float = 0.0      # coupler spacing link width
    axis_z: float = 1.0          # mechanism planar rotation normal axis vector Z


@dataclass
class LagmulDeltaRobotJoint:
    """``/DELTA_ROBOT_JOINT/id`` or ``/LAGMUL/DELTA_ROBOT_JOINT/id`` (M289): 3-DOF parallel delta robot spatial linkage kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base actuated arm shoulder hinge node
    node2: int = 0               # moving end-effector travelling platform node
    node3: int = 0               # base frame central reference anchor node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    upper_arm_len: float = 0.0   # actuated upper arm bicep link length
    forearm_len: float = 0.0     # parallel parallelogram forearm rod length
    base_radius: float = 0.0     # fixed top base triangular mounting radius


@dataclass
class LagmulSphericalWristJoint:
    """``/SPHERICAL_WRIST_JOINT/id`` or ``/LAGMUL/SPHERICAL_WRIST_JOINT/id`` (M290): 3-DOF robotic intersecting-axes spherical wrist kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # robotic forearm roll axis input node
    node2: int = 0               # gripper end-effector yaw axis output node
    node3: int = 0               # intermediate pitch gimbal intersection pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    roll_limit: float = 0.0      # maximum allowable relative roll angle limit (deg)
    pitch_limit: float = 0.0     # maximum allowable relative pitch angle limit (deg)
    yaw_limit: float = 0.0       # maximum allowable relative yaw angle limit (deg)


@dataclass
class LagmulLeadScrewJoint:
    """``/LEAD_SCREW_JOINT/id`` or ``/LAGMUL/LEAD_SCREW_JOINT/id`` (M291): Helical lead screw and ball screw coupled linear-rotational kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # rotating screw shaft node
    node2: int = 0               # translating nut slider node
    node3: int = 0               # screw axis support reference anchor node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    pitch_lead: float = 0.0      # screw linear advance lead per turn (mm or m)
    thread_angle: float = 0.0    # thread helix flank angle (deg)
    helix_efficiency: float = 1.0 # forward mechanical drive efficiency


@dataclass
class LagmulHoekenLinkageJoint:
    """``/HOEKEN_LINKAGE_JOINT/id`` or ``/LAGMUL/HOEKEN_LINKAGE_JOINT/id`` (M292): Hoecken 4-bar straight-line approximate planar kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving rotating crank pivot node
    node2: int = 0               # straight-line tracing coupler endpoint node
    node3: int = 0               # oscillating rocker arm anchor pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    crank_len: float = 0.0       # input driving crank link length
    rocker_len: float = 0.0      # oscillating rocker link length
    coupler_len: float = 0.0     # intermediate coupler link length


@dataclass
class LagmulChebyshevLinkageJoint:
    """``/CHEBYSHEV_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEBYSHEV_LINKAGE_JOINT/id`` (M293): Chebyshev 4-bar straight-line crossing linkage planar kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # driving rotating crank pivot node
    node2: int = 0               # straight-line tracing coupler midpoint node
    node3: int = 0               # driven oscillating rocker pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    base_len: float = 0.0        # fixed ground frame base distance
    crank_len: float = 0.0       # input driving crank link length
    coupler_len: float = 0.0     # intermediate coupler link length


@dataclass
class LagmulRobertsLinkageJoint:
    """``/ROBERTS_LINKAGE_JOINT/id`` or ``/LAGMUL/ROBERTS_LINKAGE_JOINT/id`` (M294): Roberts 4-bar straight-line symmetrical linkage planar kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # left grounded pivot base node
    node2: int = 0               # apex straight-line tracing coupler node
    node3: int = 0               # right grounded pivot base node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    base_len: float = 0.0        # fixed ground frame base distance
    arm_len: float = 0.0         # symmetrical grounded arm link length
    coupler_height: float = 0.0  # triangular coupler apex height


@dataclass
class LagmulEvansLinkageJoint:
    """``/EVANS_LINKAGE_JOINT/id`` or ``/LAGMUL/EVANS_LINKAGE_JOINT/id`` (M295): Evans (Grasshopper) 4-bar approximate straight-line linkage planar kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed base crank anchor pivot node
    node2: int = 0               # straight-line tracing long arm tip node
    node3: int = 0               # long arm oscillating guide pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    ground_len: float = 0.0      # fixed base frame separation distance
    crank_len: float = 0.0       # short guiding crank link length
    arm_len: float = 0.0         # long straight-line tracing carrier arm length


@dataclass
class LagmulWattLinkageJoint:
    """``/WATT_LINKAGE_JOINT/id`` or ``/LAGMUL/WATT_LINKAGE_JOINT/id`` (M296): Watt 4-bar approximate straight-line linkage planar kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed pivot node of first rocker arm
    node2: int = 0               # straight-line tracing coupler midpoint node
    node3: int = 0               # fixed pivot node of second rocker arm
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    ground_len: float = 0.0      # fixed base pivot distance
    link1_len: float = 0.0       # first oscillating rocker arm link length
    link2_len: float = 0.0       # second oscillating rocker arm link length
    coupler_len: float = 0.0     # connecting floating coupler link length


@dataclass
class LagmulHartLinkageJoint:
    """``/HART_LINKAGE_JOINT/id`` or ``/LAGMUL/HART_LINKAGE_JOINT/id`` (M297): Hart 5-bar / 6-bar exact straight-line linkage planar kinematic inversor mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed base pivot node
    node2: int = 0               # straight-line motion output node
    node3: int = 0               # auxiliary guide / oscillating rocker node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    base_len: float = 0.0        # fixed base anchor distance
    short_link_len: float = 0.0  # short link length of the Hart inversor antiparallelogram
    long_link_len: float = 0.0   # long link length of the Hart inversor antiparallelogram
    coupler_ratio: float = 0.0   # geometric proportionality ratio along coupler bars


@dataclass
class LagmulPeaucellierLinkageJoint:
    """``/PEAUCELLIER_LINKAGE_JOINT/id`` or ``/LAGMUL/PEAUCELLIER_LINKAGE_JOINT/id`` (M298): Peaucellier–Lipkin 8-bar exact straight-line linkage planar kinematic inversor mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed base anchor pivot node
    node2: int = 0               # straight-line motion output node
    node3: int = 0               # auxiliary pivot / center oscillating link node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    base_len: float = 0.0        # fixed base distance between anchor pivots
    rhombus_len: float = 0.0     # length of the 4 equal rhombus diamond links
    long_link_len: float = 0.0   # length of the 2 equal long radial links
    inversor_k: float = 0.0      # geometric inversor power / coupling parameter


@dataclass
class LagmulSarrusLinkageJoint:
    """``/SARRUS_LINKAGE_JOINT/id`` or ``/LAGMUL/SARRUS_LINKAGE_JOINT/id`` (M299): Sarrus 6-bar spatial exact straight-line kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed base anchor pivot node
    node2: int = 0               # straight-line guided output node
    node3: int = 0               # intermediate hinged plate pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    plate_len1: float = 0.0      # length of first hinged plate link
    plate_len2: float = 0.0      # length of second hinged plate link
    hinge_angle: float = 90.0    # dihedral orientation angle between perpendicular hinge plates (degrees)
    guide_travel: float = 0.0    # maximum linear travel distance


@dataclass
class LagmulKlannLinkageJoint:
    """``/KLANN_LINKAGE_JOINT/id`` or ``/LAGMUL/KLANN_LINKAGE_JOINT/id`` (M300): Klann 6-bar mechanical walking linkage planar kinematic leg mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed frame base pivot node
    node2: int = 0               # foot step output node
    node3: int = 0               # crank driving pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    crank_len: float = 0.0       # length of driving crank link
    rocker_len: float = 0.0      # length of oscillating rocker link
    leg_upper_len: float = 0.0   # length of upper leg coupler link
    leg_lower_len: float = 0.0   # length of lower leg ground-contact link


@dataclass
class LagmulJansenLinkageJoint:
    """``/JANSEN_LINKAGE_JOINT/id`` or ``/LAGMUL/JANSEN_LINKAGE_JOINT/id`` (M301): Jansen 8-bar / 11-bar kinematic walking linkage (Theo Jansen Strandbeest leg mechanism) joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed frame base pivot node
    node2: int = 0               # foot step output node
    node3: int = 0               # crank driving pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    crank_len: float = 0.0       # length of driving crank link
    base_horizontal: float = 0.0 # horizontal distance between frame base pivots
    base_vertical: float = 0.0   # vertical distance between frame base pivots
    leg_ratio: float = 0.0       # leg triangular coupler proportion ratio


@dataclass
class LagmulKempeLinkageJoint:
    """``/KEMPE_LINKAGE_JOINT/id`` or ``/LAGMUL/KEMPE_LINKAGE_JOINT/id`` (M302): Kempe multi-bar kinematic linkage joint constraint (Kempe's exact straight-line and angle-multiplier mechanism)."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed frame base pivot node
    node2: int = 0               # straight-line output coupler node
    node3: int = 0               # crank driving pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    arm_len1: float = 0.0        # primary linkage arm length
    arm_len2: float = 0.0        # secondary linkage arm length
    cross_len: float = 0.0       # diagonal crossing link length
    travel_span: float = 0.0     # linear travel stroke span


@dataclass
class LagmulSylvesterKempeLinkageJoint:
    """``/SYLVESTER_KEMPE_LINKAGE_JOINT/id`` or ``/LAGMUL/SYLVESTER_KEMPE_LINKAGE_JOINT/id`` (M303): Sylvester-Kempe 8-bar quad-inversor exact straight-line and circular-arc kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed frame base pivot node
    node2: int = 0               # tracing trajectory output node
    node3: int = 0               # crank driving pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    arm_len_a: float = 0.0       # primary quad-inversor arm link length
    arm_len_b: float = 0.0       # secondary quad-inversor arm link length
    base_dist: float = 0.0       # distance between fixed base pivot anchors
    angular_multiplier: float = 1.0 # kinematic angular gear ratio / motion multiplier


@dataclass
class LagmulWobbleYokeJoint:
    """``/WOBBLE_YOKE_JOINT/id`` or ``/LAGMUL/WOBBLE_YOKE_JOINT/id`` (M304): Wobble yoke / nutating spatial kinematic joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # input rotating shaft pivot node
    node2: int = 0               # oscillating nutating yoke output node
    node3: int = 0               # fixed frame base pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    nutation_angle: float = 0.0  # nutation tilt cone angle
    yoke_radius: float = 0.0     # radial arm radius of yoke
    stroke_travel: float = 0.0   # axial stroke displacement span
    phase_offset: float = 0.0    # cyclic angular phase offset


@dataclass
class LagmulHypocyclicLinkageJoint:
    """``/HYPOCYCLIC_LINKAGE_JOINT/id`` or ``/LAGMUL/HYPOCYCLIC_LINKAGE_JOINT/id`` (M305): Hypocyclic / Tusi couple 2-gear straight-line kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed outer ring base node
    node2: int = 0               # inner rolling planet gear node
    node3: int = 0               # straight-line tracing rim point node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    radius_outer: float = 0.0    # outer ring pitch circle radius
    radius_inner: float = 0.0    # inner rolling pitch circle radius
    stroke_travel: float = 0.0   # linear diametral stroke travel span
    phase_angle: float = 0.0     # initial rolling angular phase


@dataclass
class LagmulPantographLinkageJoint:
    """``/PANTOGRAPH_LINKAGE_JOINT/id`` or ``/LAGMUL/PANTOGRAPH_LINKAGE_JOINT/id`` (M306): Pantograph parallelogram 4-bar / 5-bar kinematic motion scaling and copy joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed anchor base pivot node
    node2: int = 0               # input tracer probe guide node
    node3: int = 0               # output scaled motion reproduction node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    scale_factor: float = 2.0    # geometric magnification scale factor
    arm_length_a: float = 0.0    # primary parallelogram arm link length
    arm_length_b: float = 0.0    # secondary parallelogram arm link length
    cross_angle_0: float = 0.0   # initial opening cross angle (radians)


@dataclass
class LagmulWattParallelMotionJoint:
    """``/WATT_PARALLEL_MOTION_JOINT/id`` or ``/LAGMUL/WATT_PARALLEL_MOTION_JOINT/id`` (M307): Watt double-parallelogram parallel motion kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed anchor pivot base node
    node2: int = 0               # tracing motion guide output node
    node3: int = 0               # intermediate beam pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    arm_length_main: float = 0.0 # main beam arm length
    arm_length_sub: float = 0.0  # sub-parallelogram link length
    stroke_travel: float = 0.0   # linear parallel stroke displacement span
    offset_dist: float = 0.0     # transverse offset distance


@dataclass
class LagmulScottRussellLinkageJoint:
    """``/SCOTT_RUSSELL_LINKAGE_JOINT/id`` or ``/LAGMUL/SCOTT_RUSSELL_LINKAGE_JOINT/id`` (M308): Scott Russell exact straight-line rolling circle kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed linear slider guide node
    node2: int = 0               # linear straight-line tracing pointer node
    node3: int = 0               # intermediate hinged crank pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    crank_len: float = 0.0       # ground pivoted crank link length
    coupler_len: float = 0.0     # equal-split coupler link length
    stroke_travel: float = 0.0   # linear vertical exact straight-line travel stroke
    slider_friction: float = 0.0 # linear guide friction coefficient


@dataclass
class LagmulWattBeamEngineJoint:
    """``/WATT_BEAM_ENGINE_JOINT/id`` or ``/LAGMUL/WATT_BEAM_ENGINE_JOINT/id`` (M309): Watt rocking walking-beam engine kinematic linkage joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed center pivot trunnion node
    node2: int = 0               # piston crosshead vertical guided node
    node3: int = 0               # flywheel crank pin node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    beam_half_length_a: float = 0.0 # piston-side beam half length
    beam_half_length_b: float = 0.0 # crank-side beam half length
    stroke_travel: float = 0.0   # piston vertical stroke travel span
    beam_tilt_max: float = 0.0   # maximum beam rocking tilt angle (radians)


@dataclass
class LagmulStephensonLinkageJoint:
    """``/STEPHENSON_LINKAGE_JOINT/id`` or ``/LAGMUL/STEPHENSON_LINKAGE_JOINT/id`` (M310): Stephenson 6-bar kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # ground pivot node 1
    node2: int = 0               # tracing coupler output node
    node3: int = 0               # ground pivot node 2
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # crank link length a
    link_len_b: float = 0.0      # coupler link length b
    link_len_c: float = 0.0      # rocker link length c
    link_len_d: float = 0.0      # ternary link length d


@dataclass
class LagmulWobblePlateMechanismJoint:
    """``/WOBBLE_PLATE_MECHANISM_JOINT/id`` or ``/LAGMUL/WOBBLE_PLATE_MECHANISM_JOINT/id`` (M311): Wobble plate / nutating axial kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed shaft centerline node
    node2: int = 0               # swashplate wobble center node
    node3: int = 0               # axial reciprocating piston rod node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    plate_radius: float = 0.0    # wobble plate pitch circle radius
    nutation_angle: float = 0.0  # swashplate tilt nutation angle (radians)
    stroke_travel: float = 0.0   # axial piston displacement travel span
    piston_count: int = 1        # number of circumferential pistons


@dataclass
class LagmulWhitworthQuickReturnJoint:
    """``/WHITWORTH_QUICK_RETURN_JOINT/id`` or ``/LAGMUL/WHITWORTH_QUICK_RETURN_JOINT/id`` (M312): Whitworth quick-return slotted crank kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed driving crank shaft center node
    node2: int = 0               # sliding ram reciprocating tool output node
    node3: int = 0               # slotted crank slider pin node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    driving_crank_len: float = 0.0 # driving crank radius R
    pivot_offset: float = 0.0    # fixed pivot offset distance d
    slotted_arm_len: float = 0.0 # slotted oscillating lever arm length L
    connecting_rod_len: float = 0.0 # ram connecting rod length


@dataclass
class LagmulChebyshevLambdaLinkageJoint:
    """``/CHEBYSHEV_LAMBDA_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEBYSHEV_LAMBDA_LINKAGE_JOINT/id`` (M313): Chebyshev lambda / walking mechanism 4-bar straight-line kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed ground pivot node 1
    node2: int = 0               # tracing lambda coupler foot node
    node3: int = 0               # fixed ground pivot node 2
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    ground_dist: float = 0.0     # ground pivot distance d
    crank_len: float = 0.0       # input crank length a
    coupler_len: float = 0.0     # coupler link length b
    rocker_len: float = 0.0      # rocker link length c


@dataclass
class LagmulFourBarCrankRockerJoint:
    """``/FOUR_BAR_CRANK_ROCKER_JOINT/id`` or ``/LAGMUL/FOUR_BAR_CRANK_ROCKER_JOINT/id`` (M314): Grashof 4-bar crank-rocker kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed ground frame crank pivot node
    node2: int = 0               # oscillating rocker arm output node
    node3: int = 0               # fixed ground frame rocker pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    crank_len: float = 0.0       # input crank length s
    coupler_len: float = 0.0     # coupler link length p
    rocker_len: float = 0.0      # output rocker length q
    ground_len: float = 0.0      # fixed ground distance l


@dataclass
class LagmulFourBarDoubleCrankJoint:
    """``/FOUR_BAR_DOUBLE_CRANK_JOINT/id`` or ``/LAGMUL/FOUR_BAR_DOUBLE_CRANK_JOINT/id`` (M315): Grashof 4-bar double-crank (drag-link) kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed ground frame driving crank pivot node
    node2: int = 0               # driven output crank revolving node
    node3: int = 0               # fixed ground frame driven crank pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    driving_crank_len: float = 0.0 # driving crank length s
    coupler_len: float = 0.0     # coupler link length p
    driven_crank_len: float = 0.0 # driven crank length q
    ground_len: float = 0.0      # fixed ground distance l


@dataclass
class LagmulFourBarDoubleRockerJoint:
    """``/FOUR_BAR_DOUBLE_ROCKER_JOINT/id`` or ``/LAGMUL/FOUR_BAR_DOUBLE_ROCKER_JOINT/id`` (M316): Grashof / Non-Grashof 4-bar double-rocker (dual oscillating arm) kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed ground frame input rocker pivot node
    node2: int = 0               # output rocker pivot/rocking node
    node3: int = 0               # fixed ground frame output rocker pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    input_rocker_len: float = 0.0 # input rocker length s
    coupler_len: float = 0.0     # coupler link length p
    output_rocker_len: float = 0.0 # output rocker length q
    ground_len: float = 0.0      # fixed ground distance l


@dataclass
class LagmulSliderRockerInversionJoint:
    """``/SLIDER_ROCKER_INVERSION_JOINT/id`` or ``/LAGMUL/SLIDER_ROCKER_INVERSION_JOINT/id`` (M317): Slider-rocker inverted kinematic mechanism (oscillating cylinder / swing engine) joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed base trunnion pivot node
    node2: int = 0               # oscillating slider piston pin node
    node3: int = 0               # revolving crank pin node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    crank_len: float = 0.0       # revolving crank length R
    frame_dist: float = 0.0      # fixed base trunnion to crank center distance D
    piston_offset: float = 0.0   # cylinder lateral offset e
    stroke_limit: float = 0.0    # maximum piston slide stroke limit


@dataclass
class LagmulScotchYokeMechanismJoint:
    """``/SCOTCH_YOKE_MECHANISM_JOINT/id`` or ``/LAGMUL/SCOTCH_YOKE_MECHANISM_JOINT/id`` (M318): Scotch yoke / slotted link reciprocating-to-rotary planar kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # fixed center crank pivot node
    node2: int = 0               # crank revolving pin node
    node3: int = 0               # reciprocating slotted yoke follower node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    crank_radius: float = 0.0    # rotating crank pin radius R
    slot_width: float = 0.0      # slotted yoke channel guide width W
    stroke_limit: float = 0.0    # maximum reciprocating slider travel stroke limit
    yoke_angle: float = 0.0      # nominal yoke slot inclination angle alpha


@dataclass
class LagmulGenevaDriveMechanismJoint:
    """``/GENEVA_DRIVE_MECHANISM_JOINT/id`` or ``/LAGMUL/GENEVA_DRIVE_MECHANISM_JOINT/id`` (M319): Geneva drive / Maltese cross intermittent rotary indexing kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # drive wheel center pivot node
    node2: int = 0               # drive pin revolving node
    node3: int = 0               # slotted Geneva driven wheel center pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    drive_radius: float = 0.0    # drive wheel pin radius R_d
    wheel_radius: float = 0.0    # Geneva slotted wheel outer radius R_g
    num_slots: int = 4           # number of indexing slots n
    center_dist: float = 0.0     # wheel shaft center-to-center distance D


@dataclass
class LagmulDoubleCardanJoint:
    """``/DOUBLE_CARDAN_JOINT/id`` or ``/LAGMUL/DOUBLE_CARDAN_JOINT/id`` (M320): Double Cardan / constant-velocity dual-universal kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # input driving shaft yoke node
    node2: int = 0               # intermediate floating cross-yoke coupling node
    node3: int = 0               # output driven shaft yoke node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    center_yoke_len: float = 0.0 # intermediate center yoke length L_c
    max_bend_angle: float = 0.0  # maximum allowable angular articulation angle theta_max
    phase_offset: float = 0.0    # relative rotational phase offset angle phi_0
    friction_coeff: float = 0.0  # trunnion bearing friction coefficient mu


@dataclass
class LagmulBennettLinkageJoint:
    """``/BENNETT_LINKAGE_JOINT/id`` or ``/LAGMUL/BENNETT_LINKAGE_JOINT/id`` (M321): Bennett's 4R spatial skewed-axis overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base frame primary pivot node
    node2: int = 0               # intermediate skewed revolving link node
    node3: int = 0               # output driven spatial arm pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary opposite link a
    link_len_b: float = 0.0      # length of secondary opposite link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    twist_angle_beta: float = 0.0  # spatial link twist angle beta (deg)


@dataclass
class LagmulBricardLinkageJoint:
    """``/BRICARD_LINKAGE_JOINT/id`` or ``/LAGMUL/BRICARD_LINKAGE_JOINT/id`` (M322): Bricard's 6R spatial line-symmetric/plane-symmetric overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node
    node2: int = 0               # intermediate spatial link node
    node3: int = 0               # driven spatial link pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len: float = 0.0        # link length L_0
    twist_angle: float = 0.0     # spatial twist angle alpha (deg)
    offset_dist: float = 0.0     # joint axis axial offset distance d_0
    sym_angle: float = 0.0       # symmetry plane articulation angle phi_sym (deg)


@dataclass
class LagmulMyardLinkageJoint:
    """``/MYARD_LINKAGE_JOINT/id`` or ``/LAGMUL/MYARD_LINKAGE_JOINT/id`` (M323): Myard's 5R spatial plane-symmetric overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node
    node2: int = 0               # intermediate spatial link node
    node3: int = 0               # driven spatial link pivot node
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of base link a
    link_len_b: float = 0.0      # length of symmetric arm b
    twist_angle: float = 0.0     # spatial twist angle alpha (deg)
    fold_angle: float = 0.0      # spatial folding angle beta (deg)


@dataclass
class LagmulGoldbergLinkageJoint:
    """``/GOLDBERG_LINKAGE_JOINT/id`` or ``/LAGMUL/GOLDBERG_LINKAGE_JOINT/id`` (M324): Goldberg 5R / 6R spatial overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link pivot node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary link a
    link_len_b: float = 0.0      # length of secondary link b
    skew_angle_alpha: float = 0.0 # spatial skew angle alpha (deg)
    offset_angle_beta: float = 0.0 # spatial offset angle beta (deg)


@dataclass
class LagmulWaldronLinkageJoint:
    """``/WALDRON_LINKAGE_JOINT/id`` or ``/LAGMUL/WALDRON_LINKAGE_JOINT/id`` (M325): Waldron 6R / hybrid spatial overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance S


@dataclass
class LagmulDietmaierLinkageJoint:
    """``/DIETMAIER_LINKAGE_JOINT/id`` or ``/LAGMUL/DIETMAIER_LINKAGE_JOINT/id`` (M326): Dietmaier 6R spatial overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    link_angle_theta: float = 0.0 # spatial link angular orientation theta (deg)


@dataclass
class LagmulBakerLinkageJoint:
    """``/BAKER_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_LINKAGE_JOINT/id`` (M327): Baker spatial overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_d: float = 0.0 # axial joint offset distance d


@dataclass
class LagmulWohlhartLinkageJoint:
    """``/WOHLHART_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_LINKAGE_JOINT/id`` (M328): Wohlhart spatial overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulAltmannLinkageJoint:
    """``/ALTMANN_LINKAGE_JOINT/id`` or ``/LAGMUL/ALTMANN_LINKAGE_JOINT/id`` (M329): Altmann spatial 6R line-symmetric overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_d: float = 0.0 # axial joint offset distance d


@dataclass
class LagmulWunderlichLinkageJoint:
    """``/WUNDERLICH_LINKAGE_JOINT/id`` or ``/LAGMUL/WUNDERLICH_LINKAGE_JOINT/id`` (M330): Wunderlich spatial 6R overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_e: float = 0.0 # axial joint offset distance e


@dataclass
class LagmulDelassusLinkageJoint:
    """``/DELASSUS_LINKAGE_JOINT/id`` or ``/LAGMUL/DELASSUS_LINKAGE_JOINT/id`` (M331): Delassus spatial 6R overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulSchatzLinkageJoint:
    """``/SCHATZ_LINKAGE_JOINT/id`` or ``/LAGMUL/SCHATZ_LINKAGE_JOINT/id`` (M332): Schatz spatial 6R / turbula inversor overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulFrankeLinkageJoint:
    """``/FRANKE_LINKAGE_JOINT/id`` or ``/LAGMUL/FRANKE_LINKAGE_JOINT/id`` (M333): Franke spatial 6R overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulKramesLinkageJoint:
    """``/KRAMES_LINKAGE_JOINT/id`` or ``/LAGMUL/KRAMES_LINKAGE_JOINT/id`` (M334): Krames spatial 6R symmetrical overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulBorelLinkageJoint:
    """``/BOREL_LINKAGE_JOINT/id`` or ``/LAGMUL/BOREL_LINKAGE_JOINT/id`` (M335): Borel spatial 6R spherical-bivector overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulHerveLinkageJoint:
    """``/HERVE_LINKAGE_JOINT/id`` or ``/LAGMUL/HERVE_LINKAGE_JOINT/id`` (M336): Hervé spatial 6R isocline overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulKongLinkageJoint:
    """``/KONG_LINKAGE_JOINT/id`` or ``/LAGMUL/KONG_LINKAGE_JOINT/id`` (M337): Kong spatial 6R multi-loop / line-symmetric overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulHuntLinkageJoint:
    """``/HUNT_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_LINKAGE_JOINT/id`` (M338): Hunt spatial 6R screw-symmetric / line-symmetric overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulBakerLineLinkageJoint:
    """``/BAKER_LINE_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_LINE_LINKAGE_JOINT/id`` (M339): Baker spatial 6R line-symmetric overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulBakerPlaneLinkageJoint:
    """``/BAKER_PLANE_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_PLANE_LINKAGE_JOINT/id`` (M340): Baker spatial 6R plane-symmetric overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_f: float = 0.0 # axial joint offset distance f


@dataclass
class LagmulWohlhartHybridLinkageJoint:
    """``/WOHLHART_HYBRID_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_HYBRID_LINKAGE_JOINT/id`` (M341): Wohlhart 6R spatial hybrid / line-plane overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_h: float = 0.0 # axial joint offset distance h


@dataclass
class LagmulChenLinkageJoint:
    """``/CHEN_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEN_LINKAGE_JOINT/id`` (M342): Chen's 6R spatial large-displacement overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_e: float = 0.0 # axial joint offset distance e


@dataclass
class LagmulBakerSymmetricLinkageJoint:
    """``/BAKER_SYMMETRIC_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_SYMMETRIC_LINKAGE_JOINT/id`` (M343): Baker spatial 6R fully-symmetric overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_r: float = 0.0 # axial joint offset distance r


@dataclass
class LagmulAltmannSpatialLinkageJoint:
    """``/ALTMANN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALTMANN_SPATIAL_LINKAGE_JOINT/id`` (M344): Altmann spatial 6R non-spherical multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulDietmaierSpatialLinkageJoint:
    """``/DIETMAIER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIETMAIER_SPATIAL_LINKAGE_JOINT/id`` (M345): Dietmaier spatial 6R variable-geometry overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_t: float = 0.0 # axial joint offset distance t


@dataclass
class LagmulWohlhartSpatialLinkageJoint:
    """``/WOHLHART_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_SPATIAL_LINKAGE_JOINT/id`` (M346): Wohlhart spatial 6R skew-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_u: float = 0.0 # axial joint offset distance u


@dataclass
class LagmulHuntSpatialLinkageJoint:
    """``/HUNT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/id`` (M347): Hunt spatial 6R variable screw axis overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_v: float = 0.0 # axial joint offset distance v


@dataclass
class LagmulChenSpatialLinkageJoint:
    """``/CHEN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEN_SPATIAL_LINKAGE_JOINT/id`` (M348): Chen spatial 6R large-displacement multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_w: float = 0.0 # axial joint offset distance w


@dataclass
class LagmulBakerSpatialLinkageJoint:
    """``/BAKER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/id`` (M349): Baker spatial 6R variable-geometry multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_r: float = 0.0 # axial joint offset distance r


@dataclass
class LagmulWaldronSpatialLinkageJoint:
    """``/WALDRON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WALDRON_SPATIAL_LINKAGE_JOINT/id`` (M350): Waldron spatial 6R hybrid multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulBricardSpatialLinkageJoint:
    """``/BRICARD_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BRICARD_SPATIAL_LINKAGE_JOINT/id`` (M351): Bricard spatial 6R triaxial multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulBennettSpatialLinkageJoint:
    """``/BENNETT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BENNETT_SPATIAL_LINKAGE_JOINT/id`` (M352): Bennett spatial skew-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulMyardSpatialLinkageJoint:
    """``/MYARD_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MYARD_SPATIAL_LINKAGE_JOINT/id`` (M353): Myard spatial 6R plane-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulGoldbergSpatialLinkageJoint:
    """``/GOLDBERG_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/GOLDBERG_SPATIAL_LINKAGE_JOINT/id`` (M354): Goldberg spatial 6R variable-angle multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulSarrusSpatialLinkageJoint:
    """``/SARRUS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SARRUS_SPATIAL_LINKAGE_JOINT/id`` (M355): Sarrus spatial 6R rectilinear multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulDelassusSpatialLinkageJoint:
    """``/DELASSUS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DELASSUS_SPATIAL_LINKAGE_JOINT/id`` (M356): Delassus spatial 6R cylindrical-surfaced multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulWohlhartSpatialLinkageJoint:
    """``/WOHLHART_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_SPATIAL_LINKAGE_JOINT/id`` (M357): Wohlhart spatial 6R hybrid-symmetry multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s
    offset_distance_u: float = 0.0 # compatibility alias

    def __post_init__(self):
        if self.offset_distance_u != 0.0 and self.offset_distance_s == 0.0:
            self.offset_distance_s = self.offset_distance_u
        elif self.offset_distance_s != 0.0 and self.offset_distance_u == 0.0:
            self.offset_distance_u = self.offset_distance_s


@dataclass
class LagmulAltmannSpatialLinkageJoint:
    """``/ALTMANN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALTMANN_SPATIAL_LINKAGE_JOINT/id`` (M358): Altmann spatial 6R line-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulBakerSpatialLinkageJoint:
    """``/BAKER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/id`` (M359): Baker spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s
    offset_distance_r: float = 0.0 # compatibility alias

    def __post_init__(self):
        if self.offset_distance_r != 0.0 and self.offset_distance_s == 0.0:
            self.offset_distance_s = self.offset_distance_r
        elif self.offset_distance_s != 0.0 and self.offset_distance_r == 0.0:
            self.offset_distance_r = self.offset_distance_s


@dataclass
class LagmulDietmaierSpatialLinkageJoint:
    """``/DIETMAIER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIETMAIER_SPATIAL_LINKAGE_JOINT/id`` (M360): Dietmaier spatial 6R isomorphic multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s
    offset_distance_t: float = 0.0 # compatibility alias

    def __post_init__(self):
        if self.offset_distance_t != 0.0 and self.offset_distance_s == 0.0:
            self.offset_distance_s = self.offset_distance_t
        elif self.offset_distance_s != 0.0 and self.offset_distance_t == 0.0:
            self.offset_distance_t = self.offset_distance_s


@dataclass
class LagmulWaldronSpatialLinkageJoint:
    """``/WALDRON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WALDRON_SPATIAL_LINKAGE_JOINT/id`` (M361): Waldron spatial 6R hybrid multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulHuntSpatialLinkageJoint:
    """``/HUNT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/id`` (M362): Hunt spatial 6R special-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s
    offset_distance_v: float = 0.0 # compatibility alias

    def __post_init__(self):
        if self.offset_distance_v != 0.0 and self.offset_distance_s == 0.0:
            self.offset_distance_s = self.offset_distance_v
        elif self.offset_distance_s != 0.0 and self.offset_distance_v == 0.0:
            self.offset_distance_v = self.offset_distance_s


@dataclass
class LagmulChenSpatialLinkageJoint:
    """``/CHEN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEN_SPATIAL_LINKAGE_JOINT/id`` (M363): Chen spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s
    offset_distance_w: float = 0.0 # compatibility alias

    def __post_init__(self):
        if self.offset_distance_w != 0.0 and self.offset_distance_s == 0.0:
            self.offset_distance_s = self.offset_distance_w
        elif self.offset_distance_s != 0.0 and self.offset_distance_w == 0.0:
            self.offset_distance_w = self.offset_distance_s


@dataclass
class LagmulWunderlichSpatialLinkageJoint:
    """``/WUNDERLICH_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WUNDERLICH_SPATIAL_LINKAGE_JOINT/id`` (M364): Wunderlich spatial 6R bistable multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulKonnokSpatialLinkageJoint:
    """``/KONNOK_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT/id`` (M365): Konnok spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulPfurnerSpatialLinkageJoint:
    """``/PFURNER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT/id`` (M366): Pfurner spatial 6R parallel multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulPhillipsSpatialLinkageJoint:
    """``/PHILLIPS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/id`` (M367): Phillips spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulStevensSpatialLinkageJoint:
    """``/STEVENS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STEVENS_SPATIAL_LINKAGE_JOINT/id`` (M368): Stevens spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulBakerHybridSpatialLinkageJoint:
    """``/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M369): Baker hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulWaldronHybridSpatialLinkageJoint:
    """``/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M370): Waldron hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulChenHybridSpatialLinkageJoint:
    """``/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M371): Chen hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulWohlhartHybridSpatialLinkageJoint:
    """``/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M372): Wohlhart hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulMaverickHybridSpatialLinkageJoint:
    """``/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M373): Maverick hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulKrauseHybridSpatialLinkageJoint:
    """``/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M374): Krause hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulSturgessHybridSpatialLinkageJoint:
    """``/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M375): Sturgess hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulBevanHybridSpatialLinkageJoint:
    """``/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M376): Bevan hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulHeinrichsHybridSpatialLinkageJoint:
    """``/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M377): Heinrichs hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulAltmannHybridSpatialLinkageJoint:
    """``/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M378): Altmann hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulKirkpatrickHybridSpatialLinkageJoint:
    """``/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M379): Kirkpatrick hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulAlexanderHybridSpatialLinkageJoint:
    """``/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M380): Alexander hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulChungHybridSpatialLinkageJoint:
    """``/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M381): Chung hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulStevensHybridSpatialLinkageJoint:
    """``/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M382): Stevens hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s


@dataclass
class LagmulBakerSpatialLinkageJoint:
    """``/BAKER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/id`` (M383): Baker spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulDietmeierSpatialLinkageJoint:
    """``/DIETMEIER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIETMEIER_SPATIAL_LINKAGE_JOINT/id`` (M384): Dietmeier spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulHuntSpatialLinkageJoint:
    """``/HUNT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/id`` (M385): Hunt spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulPfurnerSpatialLinkageJoint:
    """``/PFURNER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT/id`` (M386): Pfurner spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulPhillipsSpatialLinkageJoint:
    """``/PHILLIPS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/id`` (M387): Phillips spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val



@dataclass
class LagmulKonnokSpatialLinkageJoint:
    """``/KONNOK_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT/id`` (M388): Konnok spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulWohlhartHybridSpatialLinkageJoint:
    """``/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M389): Wohlhart hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulMaverickSpatialLinkageJoint:
    """``/MAVERICK_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MAVERICK_SPATIAL_LINKAGE_JOINT/id`` (M390): Maverick spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulKrauseSpatialLinkageJoint:
    """``/KRAUSE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KRAUSE_SPATIAL_LINKAGE_JOINT/id`` (M391): Krause spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulSturgessSpatialLinkageJoint:
    """``/STURGESS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STURGESS_SPATIAL_LINKAGE_JOINT/id`` (M392): Sturgess spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulHuntSpecialSpatialLinkageJoint:
    """``/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/id`` (M393): Hunt special spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulAlbrechtSpatialLinkageJoint:
    """``/ALBRECHT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALBRECHT_SPATIAL_LINKAGE_JOINT/id`` (M394): Albrecht spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulKirsonSpatialLinkageJoint:
    """``/KIRSON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KIRSON_SPATIAL_LINKAGE_JOINT/id`` (M395): Kirson spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulDieselSpatialLinkageJoint:
    """``/DIESEL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIESEL_SPATIAL_LINKAGE_JOINT/id`` (M396): Diesel spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulTchebychevSpatialLinkageJoint:
    """``/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/id`` (M397): Chebyshev / Tchebychev spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulSylvesterSpatialLinkageJoint:
    """``/SYLVESTER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYLVESTER_SPATIAL_LINKAGE_JOINT/id`` (M398): Sylvester spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulCauchySpatialLinkageJoint:
    """``/CAUCHY_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CAUCHY_SPATIAL_LINKAGE_JOINT/id`` (M399): Cauchy spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulCayleySpatialLinkageJoint:
    """``/CAYLEY_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CAYLEY_SPATIAL_LINKAGE_JOINT/id`` (M400): Cayley spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulEuclidSpatialLinkageJoint:
    """``/EUCLID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/EUCLID_SPATIAL_LINKAGE_JOINT/id`` (M401): Euclid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulFermatSpatialLinkageJoint:
    """``/FERMAT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FERMAT_SPATIAL_LINKAGE_JOINT/id`` (M402): Fermat spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulPascalSpatialLinkageJoint:
    """``/PASCAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PASCAL_SPATIAL_LINKAGE_JOINT/id`` (M403): Pascal spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulDescartesSpatialLinkageJoint:
    """``/DESCARTES_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DESCARTES_SPATIAL_LINKAGE_JOINT/id`` (M404): Descartes spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulLeibnizSpatialLinkageJoint:
    """``/LEIBNIZ_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/LEIBNIZ_SPATIAL_LINKAGE_JOINT/id`` (M405): Leibniz spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulNewtonSpatialLinkageJoint:
    """``/NEWTON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/NEWTON_SPATIAL_LINKAGE_JOINT/id`` (M406): Newton spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulGaussSpatialLinkageJoint:
    """``/GAUSS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/GAUSS_SPATIAL_LINKAGE_JOINT/id`` (M407): Gauss spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulEulerSpatialLinkageJoint:
    """``/EULER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/EULER_SPATIAL_LINKAGE_JOINT/id`` (M408): Euler spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulLagrangeSpatialLinkageJoint:
    """``/LAGRANGE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/LAGRANGE_SPATIAL_LINKAGE_JOINT/id`` (M409): Lagrange spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulLaplaceSpatialLinkageJoint:
    """``/LAPLACE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/LAPLACE_SPATIAL_LINKAGE_JOINT/id`` (M410): Laplace spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulFourierSpatialLinkageJoint:
    """``/FOURIER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FOURIER_SPATIAL_LINKAGE_JOINT/id`` (M411): Fourier spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulPoissonSpatialLinkageJoint:
    """``/POISSON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/POISSON_SPATIAL_LINKAGE_JOINT/id`` (M412): Poisson spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulBesselSpatialLinkageJoint:
    """``/BESSEL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BESSEL_SPATIAL_LINKAGE_JOINT/id`` (M413): Bessel spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulRiemannSpatialLinkageJoint:
    """``/RIEMANN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/RIEMANN_SPATIAL_LINKAGE_JOINT/id`` (M414): Riemann spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulHilbertSpatialLinkageJoint:
    """``/HILBERT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HILBERT_SPATIAL_LINKAGE_JOINT/id`` (M415): Hilbert spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulBanachSpatialLinkageJoint:
    """``/BANACH_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BANACH_SPATIAL_LINKAGE_JOINT/id`` (M416): Banach spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulSobolevSpatialLinkageJoint:
    """``/SOBOLEV_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SOBOLEV_SPATIAL_LINKAGE_JOINT/id`` (M417): Sobolev spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulFrechetSpatialLinkageJoint:
    """``/FRECHET_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FRECHET_SPATIAL_LINKAGE_JOINT/id`` (M418): Fréchet spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulHausdorffSpatialLinkageJoint:
    """``/HAUSDORFF_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HAUSDORFF_SPATIAL_LINKAGE_JOINT/id`` (M419): Hausdorff spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulCartanSpatialLinkageJoint:
    """``/CARTAN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CARTAN_SPATIAL_LINKAGE_JOINT/id`` (M420): Cartan spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulCliffordSpatialLinkageJoint:
    """``/CLIFFORD_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/id`` (M421): Clifford spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulGrassmannSpatialLinkageJoint:
    """``/GRASSMANN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/GRASSMANN_SPATIAL_LINKAGE_JOINT/id`` (M422): Grassmann spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulMinkowskiSpatialLinkageJoint:
    """``/MINKOWSKI_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MINKOWSKI_SPATIAL_LINKAGE_JOINT/id`` (M423): Minkowski spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulLobachevskySpatialLinkageJoint:
    """``/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/id`` (M424): Lobachevsky spatial 6R hyperbolic multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulPoincareSpatialLinkageJoint:
    """``/POINCARE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/POINCARE_SPATIAL_LINKAGE_JOINT/id`` (M425): Poincaré spatial 6R hyperbolic multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulBeltramiSpatialLinkageJoint:
    """``/BELTRAMI_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BELTRAMI_SPATIAL_LINKAGE_JOINT/id`` (M426): Beltrami spatial 6R hyperbolic multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulCliffordSpatialLinkageJoint:
    """``/CLIFFORD_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/id`` (M427): Clifford spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulConformalSpatialLinkageJoint:
    """``/CONFORMAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONFORMAL_SPATIAL_LINKAGE_JOINT/id`` (M428): Conformal spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulProjectiveSpatialLinkageJoint:
    """``/PROJECTIVE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PROJECTIVE_SPATIAL_LINKAGE_JOINT/id`` (M429): Projective spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulSymplecticSpatialLinkageJoint:
    """``/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/id`` (M430): Symplectic spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulContactSpatialLinkageJoint:
    """``/CONTACT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONTACT_SPATIAL_LINKAGE_JOINT/id`` (M431): Contact spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint contact stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulAlgebraicSpatialLinkageJoint:
    """``/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/id`` (M432): Algebraic spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint algebraic stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulDifferentialSpatialLinkageJoint:
    """``/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/id`` (M433): Differential spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0               # base pivot node 1
    node2: int = 0               # intermediate spatial link node 2
    node3: int = 0               # driven spatial link node 3
    stiff: float = 1e6           # kinematic constraint differential stiffness
    skew_id: int = 0             # reference coordinate frame ID
    tol: float = 1e-6            # constraint numerical tolerance
    link_len_a: float = 0.0      # length of primary spatial link a
    link_len_b: float = 0.0      # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulTopologicalSpatialLinkageJoint:
    """``/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` (M433): Topological spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint topological stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulHomologicalSpatialLinkageJoint:
    """``/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` (M434): Homological spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint homological stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulCohomologicalSpatialLinkageJoint:
    """``/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` (M435): Cohomological spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint cohomological stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulSpinorialSpatialLinkageJoint:
    """``/SPINORIAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SPINORIAL_SPATIAL_LINKAGE_JOINT/id`` (M436): Spinorial spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint spinorial stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulSymplecticSpinorSpatialLinkageJoint:
    """``/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M438): Symplectic spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint symplectic spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


@dataclass
class LagmulContactSpinorSpatialLinkageJoint:
    """``/CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M439): Contact spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint contact spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulContactTwistorSpatialLinkageJoint = LagmulContactSpinorSpatialLinkageJoint
LagmulContactSpinorBundleSpatialLinkageJoint = LagmulContactSpinorSpatialLinkageJoint
LagmulCliffordSpinorSpatialLinkageJoint = LagmulContactSpinorSpatialLinkageJoint


@dataclass
class LagmulConformalSpinorSpatialLinkageJoint:
    """``/CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M440): Conformal spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint conformal spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulConformalTwistorSpatialLinkageJoint = LagmulConformalSpinorSpatialLinkageJoint
LagmulConformalSpinorBundleSpatialLinkageJoint = LagmulConformalSpinorSpatialLinkageJoint
LagmulConformalCliffordSpinorSpatialLinkageJoint = LagmulConformalSpinorSpatialLinkageJoint



@dataclass
class LagmulProjectiveSpinorSpatialLinkageJoint:
    """``/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M441): Projective spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint projective spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulProjectiveTwistorSpatialLinkageJoint = LagmulProjectiveSpinorSpatialLinkageJoint
LagmulProjectiveSpinorBundleSpatialLinkageJoint = LagmulProjectiveSpinorSpatialLinkageJoint
LagmulProjectiveCliffordSpinorSpatialLinkageJoint = LagmulProjectiveSpinorSpatialLinkageJoint


@dataclass
class LagmulAlgebraicSpinorSpatialLinkageJoint:
    """``/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M442): Algebraic spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint algebraic spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulAlgebraicTwistorSpatialLinkageJoint = LagmulAlgebraicSpinorSpatialLinkageJoint
LagmulAlgebraicSpinorBundleSpatialLinkageJoint = LagmulAlgebraicSpinorSpatialLinkageJoint
LagmulAlgebraicCliffordSpinorSpatialLinkageJoint = LagmulAlgebraicSpinorSpatialLinkageJoint



@dataclass
class LagmulTopologicalSpinorSpatialLinkageJoint:
    """``/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M443): Topological spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint topological spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulTopologicalTwistorSpatialLinkageJoint = LagmulTopologicalSpinorSpatialLinkageJoint
LagmulTopologicalSpinorBundleSpatialLinkageJoint = LagmulTopologicalSpinorSpatialLinkageJoint
LagmulTopologicalCliffordSpinorSpatialLinkageJoint = LagmulTopologicalSpinorSpatialLinkageJoint


@dataclass
class LagmulHomologicalSpinorSpatialLinkageJoint:
    """``/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M444): Homological spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint homological spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulHomologicalTwistorSpatialLinkageJoint = LagmulHomologicalSpinorSpatialLinkageJoint
LagmulHomologicalSpinorBundleSpatialLinkageJoint = LagmulHomologicalSpinorSpatialLinkageJoint
LagmulHomologicalCliffordSpinorSpatialLinkageJoint = LagmulHomologicalSpinorSpatialLinkageJoint


@dataclass
class LagmulCohomologicalSpinorSpatialLinkageJoint:
    """``/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M445): Cohomological spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint cohomological spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulCohomologicalTwistorSpatialLinkageJoint = LagmulCohomologicalSpinorSpatialLinkageJoint
LagmulCohomologicalSpinorBundleSpatialLinkageJoint = LagmulCohomologicalSpinorSpatialLinkageJoint
LagmulCohomologicalCliffordSpinorSpatialLinkageJoint = LagmulCohomologicalSpinorSpatialLinkageJoint


@dataclass
class LagmulSheafSpinorSpatialLinkageJoint:
    """``/SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M446): Sheaf spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint sheaf spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulSheafTwistorSpatialLinkageJoint = LagmulSheafSpinorSpatialLinkageJoint
LagmulSheafSpinorBundleSpatialLinkageJoint = LagmulSheafSpinorSpatialLinkageJoint
LagmulSheafCliffordSpinorSpatialLinkageJoint = LagmulSheafSpinorSpatialLinkageJoint


@dataclass
class LagmulStackSpinorSpatialLinkageJoint:
    """``/STACK_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STACK_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M447): Stack spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint stack spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulStackTwistorSpatialLinkageJoint = LagmulStackSpinorSpatialLinkageJoint
LagmulStackSpinorBundleSpatialLinkageJoint = LagmulStackSpinorSpatialLinkageJoint
LagmulStackCliffordSpinorSpatialLinkageJoint = LagmulStackSpinorSpatialLinkageJoint


@dataclass
class LagmulToposSpinorSpatialLinkageJoint:
    """``/TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M448): Topos spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint topos spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulToposTwistorSpatialLinkageJoint = LagmulToposSpinorSpatialLinkageJoint
LagmulToposSpinorBundleSpatialLinkageJoint = LagmulToposSpinorSpatialLinkageJoint
LagmulToposCliffordSpinorSpatialLinkageJoint = LagmulToposSpinorSpatialLinkageJoint


@dataclass
class LagmulSchemeSpinorSpatialLinkageJoint:
    """``/SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M449): Scheme spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint scheme spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulSchemeTwistorSpatialLinkageJoint = LagmulSchemeSpinorSpatialLinkageJoint
LagmulSchemeSpinorBundleSpatialLinkageJoint = LagmulSchemeSpinorSpatialLinkageJoint
LagmulSchemeCliffordSpinorSpatialLinkageJoint = LagmulSchemeSpinorSpatialLinkageJoint


@dataclass
class LagmulOrbifoldSpinorSpatialLinkageJoint:
    """``/ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M450): Orbifold spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint orbifold spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulOrbifoldTwistorSpatialLinkageJoint = LagmulOrbifoldSpinorSpatialLinkageJoint
LagmulOrbifoldSpinorBundleSpatialLinkageJoint = LagmulOrbifoldSpinorSpatialLinkageJoint
LagmulOrbifoldCliffordSpinorSpatialLinkageJoint = LagmulOrbifoldSpinorSpatialLinkageJoint


@dataclass
class LagmulFoliationSpinorSpatialLinkageJoint:
    """``/FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M451): Foliation spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint foliation spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulFoliationTwistorSpatialLinkageJoint = LagmulFoliationSpinorSpatialLinkageJoint
LagmulFoliationSpinorBundleSpatialLinkageJoint = LagmulFoliationSpinorSpatialLinkageJoint
LagmulFoliationCliffordSpinorSpatialLinkageJoint = LagmulFoliationSpinorSpatialLinkageJoint


@dataclass
class LagmulStratificationSpinorSpatialLinkageJoint:
    """``/STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M452): Stratification spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint stratification spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulStratificationTwistorSpatialLinkageJoint = LagmulStratificationSpinorSpatialLinkageJoint
LagmulStratificationSpinorBundleSpatialLinkageJoint = LagmulStratificationSpinorSpatialLinkageJoint
LagmulStratificationCliffordSpinorSpatialLinkageJoint = LagmulStratificationSpinorSpatialLinkageJoint


@dataclass
class LagmulFibrationSpinorSpatialLinkageJoint:
    """``/FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M453): Fibration spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint fibration spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulFibrationTwistorSpatialLinkageJoint = LagmulFibrationSpinorSpatialLinkageJoint
LagmulFibrationSpinorBundleSpatialLinkageJoint = LagmulFibrationSpinorSpatialLinkageJoint
LagmulFibrationCliffordSpinorSpatialLinkageJoint = LagmulFibrationSpinorSpatialLinkageJoint


@dataclass
class LagmulBundleSpinorSpatialLinkageJoint:
    """``/BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M454): Bundle spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint bundle spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulBundleTwistorSpatialLinkageJoint = LagmulBundleSpinorSpatialLinkageJoint
LagmulBundleSpinorBundleSpatialLinkageJoint = LagmulBundleSpinorSpatialLinkageJoint
LagmulBundleCliffordSpinorSpatialLinkageJoint = LagmulBundleSpinorSpatialLinkageJoint


@dataclass
class LagmulConnectionSpinorSpatialLinkageJoint:
    """``/CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M455): Connection spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint connection spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulConnectionTwistorSpatialLinkageJoint = LagmulConnectionSpinorSpatialLinkageJoint
LagmulConnectionSpinorBundleSpatialLinkageJoint = LagmulConnectionSpinorSpatialLinkageJoint
LagmulConnectionCliffordSpinorSpatialLinkageJoint = LagmulConnectionSpinorSpatialLinkageJoint


@dataclass
class LagmulCurvatureSpinorSpatialLinkageJoint:
    """``/CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M456): Curvature spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint curvature spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulCurvatureTwistorSpatialLinkageJoint = LagmulCurvatureSpinorSpatialLinkageJoint
LagmulCurvatureSpinorBundleSpatialLinkageJoint = LagmulCurvatureSpinorSpatialLinkageJoint
LagmulCurvatureCliffordSpinorSpatialLinkageJoint = LagmulCurvatureSpinorSpatialLinkageJoint


@dataclass
class LagmulTorsionSpinorSpatialLinkageJoint:
    """``/TORSION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TORSION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M457): Torsion spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint torsion spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulTorsionTwistorSpatialLinkageJoint = LagmulTorsionSpinorSpatialLinkageJoint
LagmulTorsionSpinorBundleSpatialLinkageJoint = LagmulTorsionSpinorSpatialLinkageJoint
LagmulTorsionCliffordSpinorSpatialLinkageJoint = LagmulTorsionSpinorSpatialLinkageJoint


@dataclass
class LagmulHolonomySpinorSpatialLinkageJoint:
    """``/HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M458): Holonomy spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint holonomy spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulHolonomyTwistorSpatialLinkageJoint = LagmulHolonomySpinorSpatialLinkageJoint
LagmulHolonomySpinorBundleSpatialLinkageJoint = LagmulHolonomySpinorSpatialLinkageJoint
LagmulHolonomyCliffordSpinorSpatialLinkageJoint = LagmulHolonomySpinorSpatialLinkageJoint


@dataclass
class LagmulMonodromySpinorSpatialLinkageJoint:
    """``/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M459): Monodromy spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint monodromy spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulMonodromyTwistorSpatialLinkageJoint = LagmulMonodromySpinorSpatialLinkageJoint
LagmulMonodromySpinorBundleSpatialLinkageJoint = LagmulMonodromySpinorSpatialLinkageJoint
LagmulMonodromyCliffordSpinorSpatialLinkageJoint = LagmulMonodromySpinorSpatialLinkageJoint


@dataclass
class LagmulSymplecticSpinorSpatialLinkageJoint:
    """``/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M460): Symplectic spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint symplectic spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulSymplecticTwistorSpatialLinkageJoint = LagmulSymplecticSpinorSpatialLinkageJoint
LagmulSymplecticSpinorBundleSpatialLinkageJoint = LagmulSymplecticSpinorSpatialLinkageJoint
LagmulSymplecticCliffordSpinorSpatialLinkageJoint = LagmulSymplecticSpinorSpatialLinkageJoint


@dataclass
class LagmulPoissonSpinorSpatialLinkageJoint:
    """``/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M461): Poisson spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Poisson spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulPoissonTwistorSpatialLinkageJoint = LagmulPoissonSpinorSpatialLinkageJoint
LagmulPoissonSpinorBundleSpatialLinkageJoint = LagmulPoissonSpinorSpatialLinkageJoint
LagmulPoissonCliffordSpinorSpatialLinkageJoint = LagmulPoissonSpinorSpatialLinkageJoint
LagmulMetaplecticSpinorSpatialLinkageJoint = LagmulPoissonSpinorSpatialLinkageJoint
LagmulMetaplecticTwistorSpatialLinkageJoint = LagmulPoissonSpinorSpatialLinkageJoint


@dataclass
class LagmulDiracSpinorSpatialLinkageJoint:
    """``/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M462): Dirac spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Dirac spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulDiracTwistorSpatialLinkageJoint = LagmulDiracSpinorSpatialLinkageJoint
LagmulDiracSpinorBundleSpatialLinkageJoint = LagmulDiracSpinorSpatialLinkageJoint
LagmulDiracCliffordSpinorSpatialLinkageJoint = LagmulDiracSpinorSpatialLinkageJoint
LagmulCourantSpinorSpatialLinkageJoint = LagmulDiracSpinorSpatialLinkageJoint
LagmulCourantTwistorSpatialLinkageJoint = LagmulDiracSpinorSpatialLinkageJoint


@dataclass
class LagmulJacobiSpinorSpatialLinkageJoint:
    """``/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M463): Jacobi spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Jacobi spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulJacobiTwistorSpatialLinkageJoint = LagmulJacobiSpinorSpatialLinkageJoint
LagmulJacobiSpinorBundleSpatialLinkageJoint = LagmulJacobiSpinorSpatialLinkageJoint
LagmulJacobiCliffordSpinorSpatialLinkageJoint = LagmulJacobiSpinorSpatialLinkageJoint
LagmulNambuSpinorSpatialLinkageJoint = LagmulJacobiSpinorSpatialLinkageJoint
LagmulNambuTwistorSpatialLinkageJoint = LagmulJacobiSpinorSpatialLinkageJoint


@dataclass
class LagmulCartanSpinorSpatialLinkageJoint:
    """``/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M464): Cartan spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Cartan spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulCartanTwistorSpatialLinkageJoint = LagmulCartanSpinorSpatialLinkageJoint
LagmulCartanSpinorBundleSpatialLinkageJoint = LagmulCartanSpinorSpatialLinkageJoint
LagmulCartanCliffordSpinorSpatialLinkageJoint = LagmulCartanSpinorSpatialLinkageJoint
LagmulWeylSpinorSpatialLinkageJoint = LagmulCartanSpinorSpatialLinkageJoint
LagmulWeylTwistorSpatialLinkageJoint = LagmulCartanSpinorSpatialLinkageJoint


@dataclass
class LagmulMajoranaSpinorSpatialLinkageJoint:
    """``/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M465): Majorana spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Majorana spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulMajoranaTwistorSpatialLinkageJoint = LagmulMajoranaSpinorSpatialLinkageJoint
LagmulMajoranaSpinorBundleSpatialLinkageJoint = LagmulMajoranaSpinorSpatialLinkageJoint
LagmulMajoranaCliffordSpinorSpatialLinkageJoint = LagmulMajoranaSpinorSpatialLinkageJoint
LagmulPinSpinorSpatialLinkageJoint = LagmulMajoranaSpinorSpatialLinkageJoint
LagmulPinTwistorSpatialLinkageJoint = LagmulMajoranaSpinorSpatialLinkageJoint


@dataclass
class LagmulKahlerSpinorSpatialLinkageJoint:
    """``/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M466): Kähler spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Kähler spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulKahlerTwistorSpatialLinkageJoint = LagmulKahlerSpinorSpatialLinkageJoint
LagmulKahlerSpinorBundleSpatialLinkageJoint = LagmulKahlerSpinorSpatialLinkageJoint
LagmulKahlerCliffordSpinorSpatialLinkageJoint = LagmulKahlerSpinorSpatialLinkageJoint
LagmulKahlerAtiyahSpinorSpatialLinkageJoint = LagmulKahlerSpinorSpatialLinkageJoint
LagmulKahlerAtiyahTwistorSpatialLinkageJoint = LagmulKahlerSpinorSpatialLinkageJoint


@dataclass
class LagmulKostantSpinorSpatialLinkageJoint:
    """``/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M467): Kostant spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Kostant spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulKostantTwistorSpatialLinkageJoint = LagmulKostantSpinorSpatialLinkageJoint
LagmulKostantSpinorBundleSpatialLinkageJoint = LagmulKostantSpinorSpatialLinkageJoint
LagmulKostantCliffordSpinorSpatialLinkageJoint = LagmulKostantSpinorSpatialLinkageJoint
LagmulSegalShaleWeilSpinorSpatialLinkageJoint = LagmulKostantSpinorSpatialLinkageJoint
LagmulSegalShaleWeilTwistorSpatialLinkageJoint = LagmulKostantSpinorSpatialLinkageJoint


@dataclass
class LagmulSouriauSpinorSpatialLinkageJoint:
    """``/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M468): Souriau spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Souriau spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulSouriauTwistorSpatialLinkageJoint = LagmulSouriauSpinorSpatialLinkageJoint
LagmulSouriauSpinorBundleSpatialLinkageJoint = LagmulSouriauSpinorSpatialLinkageJoint
LagmulSouriauCliffordSpinorSpatialLinkageJoint = LagmulSouriauSpinorSpatialLinkageJoint
LagmulKirillovSpinorSpatialLinkageJoint = LagmulSouriauSpinorSpatialLinkageJoint
LagmulKirillovTwistorSpatialLinkageJoint = LagmulSouriauSpinorSpatialLinkageJoint


@dataclass
class LagmulBerezinSpinorSpatialLinkageJoint:
    """``/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M469): Berezin spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Berezin spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulBerezinTwistorSpatialLinkageJoint = LagmulBerezinSpinorSpatialLinkageJoint
LagmulBerezinSpinorBundleSpatialLinkageJoint = LagmulBerezinSpinorSpatialLinkageJoint
LagmulBerezinCliffordSpinorSpatialLinkageJoint = LagmulBerezinSpinorSpatialLinkageJoint
LagmulFrobeniusSpinorSpatialLinkageJoint = LagmulBerezinSpinorSpatialLinkageJoint
LagmulFrobeniusTwistorSpatialLinkageJoint = LagmulBerezinSpinorSpatialLinkageJoint


@dataclass
class LagmulBorelSpinorSpatialLinkageJoint:
    """``/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M470): Borel spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Borel spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulBorelTwistorSpatialLinkageJoint = LagmulBorelSpinorSpatialLinkageJoint
LagmulBorelSpinorBundleSpatialLinkageJoint = LagmulBorelSpinorSpatialLinkageJoint
LagmulBorelCliffordSpinorSpatialLinkageJoint = LagmulBorelSpinorSpatialLinkageJoint
LagmulWeilSpinorSpatialLinkageJoint = LagmulBorelSpinorSpatialLinkageJoint
LagmulWeilTwistorSpatialLinkageJoint = LagmulBorelSpinorSpatialLinkageJoint


@dataclass
class LagmulBottSpinorSpatialLinkageJoint:
    """``/BOTT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BOTT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M471): Bott spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Bott spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulBottTwistorSpatialLinkageJoint = LagmulBottSpinorSpatialLinkageJoint
LagmulBottSpinorBundleSpatialLinkageJoint = LagmulBottSpinorSpatialLinkageJoint
LagmulBottCliffordSpinorSpatialLinkageJoint = LagmulBottSpinorSpatialLinkageJoint
LagmulBottPeriodicitySpinorSpatialLinkageJoint = LagmulBottSpinorSpatialLinkageJoint
LagmulBottPeriodicityTwistorSpatialLinkageJoint = LagmulBottSpinorSpatialLinkageJoint


@dataclass
class LagmulChernSpinorSpatialLinkageJoint:
    """``/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M472): Chern spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Chern spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulChernTwistorSpatialLinkageJoint = LagmulChernSpinorSpatialLinkageJoint
LagmulChernSpinorBundleSpatialLinkageJoint = LagmulChernSpinorSpatialLinkageJoint
LagmulChernCliffordSpinorSpatialLinkageJoint = LagmulChernSpinorSpatialLinkageJoint
LagmulSingerSpinorSpatialLinkageJoint = LagmulChernSpinorSpatialLinkageJoint
LagmulSingerTwistorSpatialLinkageJoint = LagmulChernSpinorSpatialLinkageJoint
LagmulChernSimonsSpinorSpatialLinkageJoint = LagmulChernSpinorSpatialLinkageJoint
LagmulChernSimonsTwistorSpatialLinkageJoint = LagmulChernSpinorSpatialLinkageJoint


@dataclass
class LagmulAtiyahSpinorSpatialLinkageJoint:
    """``/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M473): Atiyah spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Atiyah spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulAtiyahTwistorSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint
LagmulAtiyahSpinorBundleSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint
LagmulAtiyahCliffordSpinorSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint
LagmulAtiyahSingerSpinorSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint
LagmulAtiyahSingerTwistorSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint
LagmulAtiyahPatodiSingerSpinorSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint
LagmulAtiyahPatodiSingerTwistorSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint
LagmulAtiyahHirzebruchSpinorSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint
LagmulAtiyahHirzebruchTwistorSpatialLinkageJoint = LagmulAtiyahSpinorSpatialLinkageJoint


@dataclass
class LagmulHirzebruchSpinorSpatialLinkageJoint:
    """``/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M474): Hirzebruch spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Hirzebruch spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulHirzebruchTwistorSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint
LagmulHirzebruchSpinorBundleSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint
LagmulHirzebruchCliffordSpinorSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint
LagmulHirzebruchRiemannRochSpinorSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint
LagmulHirzebruchRiemannRochTwistorSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint
LagmulHirzebruchSignatureSpinorSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint
LagmulHirzebruchSignatureTwistorSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint
LagmulHirzebruchSurfaceSpinorSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint
LagmulHirzebruchSurfaceTwistorSpatialLinkageJoint = LagmulHirzebruchSpinorSpatialLinkageJoint


@dataclass
class LagmulGrothendieckSpinorSpatialLinkageJoint:
    """``/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M475): Grothendieck spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Grothendieck spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulGrothendieckTwistorSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint
LagmulGrothendieckSpinorBundleSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint
LagmulGrothendieckCliffordSpinorSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint
LagmulGrothendieckRiemannRochSpinorSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint
LagmulGrothendieckRiemannRochTwistorSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint
LagmulGrothendieckMotivicSpinorSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint
LagmulGrothendieckMotivicTwistorSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint
LagmulGrothendieckToposSpinorSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint
LagmulGrothendieckToposTwistorSpatialLinkageJoint = LagmulGrothendieckSpinorSpatialLinkageJoint


@dataclass
class LagmulSerreSpinorSpatialLinkageJoint:
    """``/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M476): Serre spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    id: int = 1
    title: str = ""
    node1: int = 0                 # base pivot node 1
    node2: int = 0                 # intermediate spatial link node 2
    node3: int = 0                 # driven spatial link node 3
    stiff: float = 1e6             # kinematic constraint Serre spinor stiffness
    skew_id: int = 0               # reference coordinate frame ID
    tol: float = 1e-6              # constraint numerical tolerance
    link_len_a: float = 0.0        # length of primary spatial link a
    link_len_b: float = 0.0        # length of secondary spatial link b
    twist_angle_alpha: float = 0.0 # spatial link twist angle alpha (deg)
    offset_distance_s: float = 0.0 # axial joint offset distance s

    @property
    def offset_distance_r(self) -> float:
        return self.offset_distance_s

    @offset_distance_r.setter
    def offset_distance_r(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_v(self) -> float:
        return self.offset_distance_s

    @offset_distance_v.setter
    def offset_distance_v(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_h(self) -> float:
        return self.offset_distance_s

    @offset_distance_h.setter
    def offset_distance_h(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_u(self) -> float:
        return self.offset_distance_s

    @offset_distance_u.setter
    def offset_distance_u(self, val: float) -> None:
        self.offset_distance_s = val

    @property
    def offset_distance_f(self) -> float:
        return self.offset_distance_s

    @offset_distance_f.setter
    def offset_distance_f(self, val: float) -> None:
        self.offset_distance_s = val


LagmulSerreTwistorSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
LagmulSerreSpinorBundleSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
LagmulSerreCliffordSpinorSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
LagmulSerreDualitySpinorSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
LagmulSerreDualityTwistorSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
LagmulSerreFibrationSpinorSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
LagmulSerreFibrationTwistorSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
LagmulSerreSpectralSpinorSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
LagmulSerreSpectralTwistorSpatialLinkageJoint = LagmulSerreSpinorSpatialLinkageJoint
