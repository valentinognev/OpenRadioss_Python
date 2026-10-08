"""Applied loads, imposed motion, sensors and damping.

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


@dataclass
class Gravity:
    """/GRAV: body acceleration a(t) = scale * funct(t) * direction."""

    id: int
    grnod_id: Optional[int]  # None = all nodes
    funct_id: int
    direction: np.ndarray    # (3,) unit vector
    scale: float = 1.0
    title: str = ""
    skew_id: int = 0
    sens_id: int = 0
    scale_x: float = 1.0


@dataclass
class ConcentratedLoad:
    """/CLOAD: nodal force F(t) = scale * funct(t) along a fixed direction,
    applied to every node of the group. ``sens_id`` (M6): the load is
    inactive until /SENSOR sens_id fires, then evaluates the curve with
    the shifted time f(t - t_fire)."""

    id: int
    grnod_id: int
    funct_id: int
    direction: np.ndarray  # (3,) unit vector
    scale: float = 1.0
    time_scale: float = 1.0
    sens_id: int = 0
    title: str = ""


@dataclass
class ImposedVelocity:
    """/IMPVEL: imposed velocity v(t) = scale * funct(t / xscale) on one
    DOF of a node group, active inside [tstart, tstop] (kinematic
    condition: overrides the solution, does not add a force; the reaction
    is recovered from the mass * acceleration).

    scale/xscale/tstart/tstop are the official card-2 fields Fscale_Y /
    Ascale_x / Tstart / Tstop (``starter/source/constraints/general/
    impvel/read_impvel.F``: zero Ascale_x or Fscale_Y defaults to 1, zero
    Tstop to infinity; ``engine/.../fixvel.F`` evaluates the curve at
    t * (1/Ascale_x) and skips the condition outside the time window)."""

    id: int
    grnod_id: int
    funct_id: int
    dof: int              # 0/1/2 = tra X/Y/Z, 3/4/5 = rot XX/YY/ZZ (M39)
    scale: float = 1.0    # Fscale_Y (curve ordinate scale)
    xscale: float = 1.0   # Ascale_x (curve abscissa scale, never 0)
    tstart: float = 0.0   # activation window
    tstop: float = 1.0e30
    sens_id: int = 0      # /SENSOR gate (parsed; engine gating not ported)
    title: str = ""
    skew_id: int = 0      # /SKEW: dof is the skew's axis, not the global one
    skew_row: int = 0     # resolved SkewSet row (0 = global)


@dataclass
class ImposedDisplacement:
    """/IMPDISP (M5): imposed displacement d(t) = scale * funct(t) on one
    DOF of a node group. Kinematic like /IMPVEL, but enforced at the
    *position* level: each cycle the velocity is set so the node lands
    exactly at x0 + d(t+dt) — no drift accumulation, unlike integrating an
    equivalent velocity curve.

    Fortran origin: ``engine/source/constraints/general/impvel/fixvel.F``
    (the same routine serves /IMPVEL and /IMPDISP through IFLAG).
    Card-2 fields as for :class:`ImposedVelocity`:
    d(t) = scale * funct(t / xscale) inside [tstart, tstop].

    ``skew_id`` (M39): the imposed component is the one along that /SKEW's
    ``dof`` axis (fixvel.F 390-418 projects the current velocity onto the
    skew axis, imposes the curve there and adds the correction back along
    the SAME axis, leaving the other two components free).
    """

    id: int
    grnod_id: int
    funct_id: int
    dof: int              # 0/1/2 = tra X/Y/Z, 3/4/5 = rot XX/YY/ZZ (M39)
    scale: float = 1.0    # Fscale_Y (curve ordinate scale)
    xscale: float = 1.0   # Ascale_x (curve abscissa scale, never 0)
    tstart: float = 0.0   # activation window
    tstop: float = 1.0e30
    sens_id: int = 0      # /SENSOR gate (parsed; engine gating not ported)
    title: str = ""
    skew_id: int = 0      # /SKEW: dof is the skew's axis, not the global one
    skew_row: int = 0     # resolved SkewSet row (0 = global)


@dataclass
class ImposedAcceleration:
    """/IMPACC (M92): imposed acceleration a(t) = scale * funct(t) on one
    DOF of a node group. Same card structure and fields as :class:`ImposedVelocity`.
    """

    id: int
    grnod_id: int
    funct_id: int
    dof: int              # 0/1/2 = tra X/Y/Z, 3/4/5 = rot XX/YY/ZZ (M39)
    scale: float = 1.0    # Fscale_Y (curve ordinate scale)
    xscale: float = 1.0   # Ascale_x (curve abscissa scale, never 0)
    tstart: float = 0.0   # activation window
    tstop: float = 1.0e30
    sens_id: int = 0      # /SENSOR gate (parsed; engine gating not ported)
    title: str = ""
    skew_id: int = 0      # /SKEW: dof is the skew's axis, not the global one
    skew_row: int = 0     # resolved SkewSet row (0 = global)


@dataclass
class PressureLoad:
    """/PLOAD (M5): follower pressure p(t) = scale * funct(t) on a /SURF.

    Fortran origin: ``engine/source/loads/general/pload/pload.F``. The
    pressure acts along the *current* segment normal (follower load, the
    normal is defined by the segment node ordering n1-n2-n3-n4, right-hand
    rule) and the resultant p*A is lumped to the corners (A/4 per quad
    corner, A/3 per triangle corner). Segments of /FAIL-deleted elements
    stop carrying pressure (a torn face is an open boundary).
    """

    id: int
    surf_id: int
    funct_id: int
    scale: float = 1.0
    sens_id: int = 0       # M6: /SENSOR gating (same semantics as /CLOAD)
    title: str = ""


@dataclass
class CentrifugalLoad:
    """/LOAD/CENTRI (M93) and /CENTRI (M198): Centrifugal rotational load / field.

    Fortran origin: ``starter/source/loads/general/load_centri/hm_read_load_centri.F``.
    """
    id: int
    funct_id: int = 0
    dir: str = "XX"         # rotation axis: X, Y, Z, XX, YY, ZZ
    frame_id: int = 0       # reference frame
    sens_id: int = 0        # /SENSOR gating
    grnod_id: int = 0       # node group
    ivar: int = 1           # 1 = ignore d_omega/dt, 2 = account for d_omega/dt
    scale_x: float = 1.0    # Ascalex (time scale)
    scale_y: float = 1.0    # Fscaley (rotational velocity scale)
    title: str = ""
    grnd_id: int = 0
    fct_id: int = 0
    node_orig: int = 0
    node_axis: int = 0
    omega: float = 0.0
    scale_z: float = 1.0


@dataclass
class ImposedTemperature:
    """/IMPTEMP (M93): imposed nodal temperature T(t) = scale * funct(t / xscale)
    on a node group.

    Fortran origin: ``starter/source/loads/thermal/imptemp/read_imptemp.F``.
    """
    id: int
    funct_id: int
    grnod_id: int
    sens_id: int = 0
    scale: float = 1.0      # Fscale_y (temperature ordinate scale)
    xscale: float = 1.0     # Ascale_x (time scale)
    tstart: float = 0.0     # T_start
    tstop: float = 1.0e30   # T_stop
    title: str = ""


@dataclass
class Damping:
    """/DAMP (M6, M108): Rayleigh MASS damping or relative/function damping —
    force f = -alpha m v on every node of the group, active in the [tstart, tstop] window.

    Fortran origin: ``engine/source/assembly/damping*.F``. The port
    integrates the mass-damping ODE exactly per cycle (integrating
    factor, see engine/damping.py) and books the removed kinetic energy
    into the DE ledger. The stiffness-proportional beta branch of full
    Rayleigh damping is not ported (needs K*v products)."""

    id: int
    grnod_id: int
    alpha: float
    tstart: float = 0.0
    tstop: float = 1e30
    title: str = ""
    kind: str = "GLOBAL"   # 'GLOBAL' | 'VREL' | 'FUNCT'
    skew_id: int = 0
    fct_id: int = 0
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    alpha_z: float = 0.0



@dataclass
class Sensor:
    """/SENSOR (M6, M84, M97): an event source gating loads and interfaces.

    Fortran origin: ``starter/source/tools/sensor/hm_read_sensor.F`` +
    ``engine/source/tools/sensor/``. Ported types:
    * ``kind='TIME'``   — fires at tdelay
    * ``kind='DISP'``   — fires when displacement magnitude of node_id exceeds dmin
    * ``kind='VEL'``    — fires when velocity magnitude of node_id exceeds vmax
    * ``kind='NOT'``    — active when sens_id1 is not active
    * ``kind='AND'``    — active when both sens_id1 and sens_id2 are active
    * ``kind='OR'``     — active when either sens_id1 or sens_id2 is active
    * ``kind='DIST'``   — fires based on distance between node_id1 and node_id2 (M97)
    * ``kind='ENERGY'`` — fires on part/system internal or kinetic energy thresholds (M97)
    * ``kind='INTER'``  — fires on contact interface force thresholds (M97)
    * ``kind='RBODY'``  — fires on rigid body force/moment thresholds (M97)
    * ``kind='TEMP'``   — fires on nodal/group temperature thresholds (M97)
    Sensors latch or update dynamically — see engine/sensors.py."""

    id: int
    kind: str              # 'TIME' | 'DISP' | 'VEL' | 'NOT' | 'AND' | 'OR' | 'DIST' | 'ENERGY' | 'INTER' | 'RBODY' | 'TEMP'
    tdelay: float = 0.0    # Time delay before activation
    node_id: int = 0       # DISP, VEL
    dmin: float = 0.0      # DISP, DIST
    dmax: float = 0.0      # DIST
    vmax: float = 0.0      # VEL
    fcut: float = 0.0      # VEL, INTER
    sens_id1: int = 0      # NOT, AND, OR
    sens_id2: int = 0      # AND, OR
    # DIST (M97)
    node_id1: int = 0
    node_id2: int = 0
    # ENERGY (M97)
    part_id: int = 0
    subset_id: int = 0
    iselect: int = 1
    iemin: float = -1e30
    iemax: float = 1e30
    kemin: float = -1e30
    kemax: float = 1e30
    # INTER (M97)
    int_id: int = 0
    # RBODY (M97)
    rbody_id: int = 0
    # Common force/moment thresholds & direction (M97)
    dir: str = ""
    fmin: float = 0.0
    fmax: float = 0.0
    # TEMP (M97)
    grnod_id: int = 0
    tempmax: float = 1e30
    tempmin: float = 0.0
    tempmean: float = 1e30
    # NIC (M107)
    nij_max: float = 0.0
    fint_tens: float = 0.0
    fint_comp: float = 0.0
    mint_flex: float = 0.0
    mint_ext: float = 0.0
    spring_id: int = 0
    # M121: GAUGE, HIC, WORK, RWALL, XSECTION, DIST_SURF
    gauge_entries: List[Tuple[int, float, float]] = field(default_factory=list) # (gauge_id, fporp, fport)
    accel_id: int = 0
    hic_period: float = 0.0
    hic_val: float = 0.0
    gravity: float = 9.81
    work_max: float = 0.0
    sect_id: int = 0
    rwall_id: int = 0
    node_id3: int = 0
    node_id4: int = 0
    surf_id: int = 0
    skew_id: int = 0
    ax_dir: str = ""
    bend_dir: str = ""
    alpha: float = 0.0
    cfc: float = 0.0
    # Duration limit (M97)
    tmin: float = 0.0
    title: str = ""
    # M131: ACCE, PYTHON
    acc_entries: List[Tuple[int, str, float, float]] = field(default_factory=list) # (acc_id, dir, tomin, tmin)
    script_name: str = ""
    func_name: str = ""
    target_id: int = 0   # SPH, AIRBAG, MONVOL, SHELL, SOLID (M136)
    dflag: int = 0       # DIST deactivation flag (M165)
    # Extended sensor threshold aliases
    a_max: float = 0.0
    e_max: float = 0.0
    f_max: float = 0.0
    t_max: float = 0.0
    energy_type: str = ""


@dataclass
class AddedMass:
    """/ADMAS (M5, M139): concentrated or distributed non-structural mass.

    Fortran origin: ``starter/source/tools/admas/hm_read_admas.F``. Beyond
    its normal use (payload, joints), this is how a *moving rigid wall
    with a mass* is built in this port: the wall is tied to a node whose
    inertia comes from /ADMAS (see RigidWall.node_id).
    """

    id: int
    grnod_id: int
    mass: float
    title: str = ""
    mass_type: int = 0  # 0: per node, 1: total on nodes, 2: total on surface, 3: total on elements


@dataclass
class ConvectionLoad:
    """/CONVEC (M94): convection heat flux boundary condition on a /SURF.

    Fortran origin: ``starter/source/loads/thermic/hm_read_convec.F``.
    q = h * (T_surf - T_inf(t))
    """
    id: int
    surf_id: int
    funct_id: int
    sens_id: int = 0
    xscale: float = 1.0     # ASCALE (time scale for T_inf curve)
    scale: float = 1.0      # FSCALE (temperature scale)
    tstart: float = 0.0     # TSTART
    tstop: float = 1.0e30   # TSTOP
    h: float = 0.0          # H (convection coefficient)
    title: str = ""


@dataclass
class RadiationLoad:
    """/RADIATION (M95): radiation heat flux boundary condition on a /SURF.

    Fortran origin: ``starter/source/loads/thermic/hm_read_radiation.F``.
    q = epsilon * sigma * (T_surf^4 - T_inf(t)^4)
    """
    id: int
    surf_id: int
    funct_id: int = 0       # time function for T_inf
    sens_id: int = 0
    xscale: float = 1.0     # ASCALE (time scale)
    scale: float = 1.0      # FSCALE (temperature scale)
    tstart: float = 0.0     # TSTART
    tstop: float = 1.0e30   # TSTOP
    emissivity: float = 0.0 # E (surface emissivity)
    title: str = ""


@dataclass
class ImposedFlux:
    """/IMPFLUX (M95/M152): imposed surface or volumetric heat flux.

    Fortran origin: ``starter/source/constraints/thermic/hm_read_impflux.F``.
    """
    id: int
    surf_id: int = 0        # surface ID for surfacic flux
    funct_id: int = 0       # time function for flux density
    sens_id: int = 0
    grbric_id: int = 0      # brick group ID for volumetric flux
    xscale: float = 1.0     # ASCALE
    scale: float = 1.0      # FSCALE
    tstart: float = 0.0     # TSTART
    tstop: float = 1.0e30   # TSTOP
    title: str = ""

    @property
    def fct_id(self) -> int:
        return self.funct_id

    @fct_id.setter
    def fct_id(self, val: int) -> None:
        self.funct_id = val

    @property
    def sensor_id(self) -> int:
        return self.sens_id

    @sensor_id.setter
    def sensor_id(self, val: int) -> None:
        self.sens_id = val

    @property
    def grbrick_id(self) -> int:
        return self.grbric_id

    @grbrick_id.setter
    def grbrick_id(self, val: int) -> None:
        self.grbric_id = val

    @property
    def scale_x(self) -> float:
        return self.xscale

    @scale_x.setter
    def scale_x(self, val: float) -> None:
        self.xscale = val

    @property
    def scale_y(self) -> float:
        return self.scale

    @scale_y.setter
    def scale_y(self, val: float) -> None:
        self.scale = val


ImpFlux = ImposedFlux


@dataclass
class PBlastLoad:
    """/LOAD/PBLAST (M99/M151): air/ground blast pressure load.

    Fortran origin: ``starter/source/model/loads/hm_read_pblast.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    exp_data: int = 1
    i_tshift: int = 1
    ndt: int = 0
    iz: int = 2
    imodel: int = 0
    node_id: int = 0
    xdet: float = 0.0
    ydet: float = 0.0
    zdet: float = 0.0
    tdet: float = 0.0
    wtnt: float = 0.0
    pmin: float = 0.0
    tstop: float = 1.0e30
    surf_ground_id: int = 0
    ishape: int = 0

    @property
    def iabac(self) -> int:
        return self.exp_data

    @iabac.setter
    def iabac(self, val: int) -> None:
        self.exp_data = val

    @property
    def ita_shift(self) -> int:
        return self.i_tshift

    @ita_shift.setter
    def ita_shift(self, val: int) -> None:
        self.i_tshift = val

    @property
    def iz_update(self) -> int:
        return self.iz

    @iz_update.setter
    def iz_update(self, val: int) -> None:
        self.iz = val


PblastLoad = PBlastLoad


@dataclass
class ElementActivation:
    """/ACTIV (M110): Dynamic activation/deactivation of element groups.
    
    Fortran origin: ``starter/source/tools/activ/hm_read_activ.F``.
    """
    id: int
    title: str = ""
    sens_id: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grshel_id: int = 0
    grtrus_id: int = 0
    grbeam_id: int = 0
    grspri_id: int = 0
    iform: int = 1
    tstart: float = 0.0
    tstop: float = 1.0e30


Activ = ElementActivation


@dataclass
class PCylLoad:
    """/LOAD/PCYL, /PLOAD/PCYL (M130): Cylindrical coordinate pressure load."""
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    skew_id: int = 0
    table_id: int = 0
    xscale_r: float = 1.0
    xscale_t: float = 1.0
    yscale_p: float = 1.0


@dataclass
class PreloadAxial:
    """/PRELOAD/AXIAL, /LOAD/PRELOAD_AXIAL (M130): Axial spring/beam preload."""
    id: int
    title: str = ""
    set_id: int = 0
    sens_id: int = 0
    fun_id: int = 0
    preload: float = 1.0
    damp: float = 0.0
    grpart_id: int = 0
    fct_id: int = 0

    def __post_init__(self):
        if not self.set_id and self.grpart_id:
            self.set_id = self.grpart_id
        elif not self.grpart_id and self.set_id:
            self.grpart_id = self.set_id
        if not self.fun_id and self.fct_id:
            self.fun_id = self.fct_id
        elif not self.fct_id and self.fun_id:
            self.fct_id = self.fun_id


@dataclass
class LaserLoad:
    """/LOAD/LASER, /LASER, /DFS/LASER (M130): Laser beam impact load."""
    id: int
    title: str = ""
    slas: float = 0.0
    fct_idlas: int = 0
    star: float = 0.0
    fct_idtar: int = 0
    hn: float = 0.0
    vcp: float = 0.0
    k0: float = 0.0
    rd: float = 0.0
    ks: float = 0.0
    np: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    magnitude: float = 0.0
    curve_id: int = 0
    s_target: float = 0.0
    fct_id_target: int = 0
    nc: int = 0
    plasma_elements: List[Any] = field(default_factory=list)

@dataclass
class PcylLoad:
    """/LOAD/PCYL (M103): Cylindrical pressure load.

    Fortran origin: ``starter/source/loads/general/load_pcyl/hm_read_pcyl.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    frame_id: int = 0
    table_id: int = 0
    xscale_r: float = 1.0
    xscale_t: float = 1.0
    yscale_p: float = 1.0


@dataclass
class PfluidLoad:
    """/LOAD/PFLUID (M103): Hydrostatic / fluid surface pressure load.

    Fortran origin: ``starter/source/loads/general/pfluid/hm_read_pfluid.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    fct_id_t: int = 0
    ascalex: float = 1.0
    fscaley: float = 1.0
    dir_p: str = "Z"
    frame_id: int = 0
    fct_id_pc: int = 0
    ascalex_pc: float = 1.0
    fscaley_pc: float = 1.0
    fct_id_vel: int = 0
    ascalex_vel: float = 1.0
    fscaley_vel: float = 1.0
    dir_vel: str = "Z"
    frame_id_vel: int = 0


@dataclass
class Preload:
    """/PRELOAD (M103): Bolt cross-section preload.

    Fortran origin: ``starter/source/loads/general/preload/hm_read_preload.F``.
    """
    id: int
    title: str = ""
    sect_id: int = 0
    sens_id: int = 0
    itype: int = 0
    fct_id: int = 0
    preload: float = 0.0
    tstart: float = 0.0
    tstop: float = 0.0

@dataclass
class DampInter:
    """/DAMP/INTER (M103/M196): Interface / relative velocity damping.

    Fortran origin: ``starter/source/general_controls/damping/hm_read_damp.F``.
    """
    id: int = 0
    title: str = ""
    nb_time_step: int = 0
    damp_range: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    grnod_id: int = 0
    skew_id: int = 0
    tstart: float = 0.0
    tstop: float = 1.0e30
    alpha_yy: float = 0.0
    beta_yy: float = 0.0
    alpha_zz: float = 0.0
    beta_zz: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def range_val(self) -> int:
        return self.damp_range

    @range_val.setter
    def range_val(self, v: int) -> None:
        self.damp_range = v


@dataclass
class DampRange:
    """/DAMP/RANGE or /DAMP/FREQUENCY_RANGE (M103): Frequency range damping.

    Fortran origin: ``starter/source/general_controls/damping/hm_read_damp.F``.
    """
    id: int
    title: str = ""
    cdamp: float = 0.0
    grpart_id: int = 0
    tstart: float = 0.0
    tstop: float = 0.0
    freq_low: float = 0.0
    freq_high: float = 0.0


@dataclass
class DampGlobal:
    """/DAMP/GLOBAL (M134): Global mass/stiffness Rayleigh damping."""
    id: int = 1
    title: str = ""
    alpha: float = 0.0
    beta: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class DampPart:
    """/DAMP/PART (M134): Per-part Rayleigh damping factor."""
    id: int
    title: str = ""
    part_id: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class IniGrav:
    """/INIGRAV (M104/M151): Initial gravity equilibrium state.

    Fortran origin: ``starter/source/initial_conditions/inigrav/hm_read_inigrav.F``.
    """
    id: int
    title: str = ""
    grpart_id: int = 0
    surf_id: int = 0
    grav_id: int = 0
    pref: float = 0.0
    bx: float = 0.0
    by: float = 0.0
    bz: float = 0.0


InigravLoad = IniGrav


@dataclass
class ImpdispFgeo:
    """/IMPDISP/FGEO (M112): Imposed final geometry displacement.

    Fortran origin: ``starter/source/constraints/general/impvel/read_impdisp_fgeo.F`` / CFG ``impdisp_fgeo.cfg``.
    """
    id: int
    title: str = ""
    fct_id: int = 0
    part_id: int = 0
    sens_id: int = 0
    ascale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    nodes: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ImpvelFgeo:
    """/IMPVEL/FGEO (M112): Imposed final geometry velocity.

    Fortran origin: ``starter/source/constraints/general/impvel/read_impvel_fgeo.F`` / CFG ``impvel_fgeo.cfg``.
    """
    id: int
    title: str = ""
    fct_id: int = 0
    part_id: int = 0
    fct_l_id: int = 0
    sens_id: int = 0
    ascale: float = 1.0
    t0: float = 0.0
    tstart: float = 0.0
    fscale_l: float = 1.0
    dmin: float = 0.0
    pairs: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class DampStiff:
    """/DAMP/STIFF (M141): Stiffness proportional damping.

    Fortran origin: ``starter/source/loads/damp/read_damp_stiff.F``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    beta: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class PreloadBolt:
    """/PRELOAD/BOLT or /SECT/BOLT (M144): Bolt section pretensioning model.

    Fortran origin: ``starter/source/loads/bolt/sboltini.F``.
    """
    id: int
    title: str = ""
    sect_id: int = 0
    sens_id: int = 0
    fct_id: int = 0
    preload: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    torque: float = 0.0
    speed: float = 0.0

@dataclass
class DampVrel:
    """/DAMP/VREL (M179): Relative velocity damping in skew coordinate system."""
    id: int
    title: str = ""
    grnod_id: int = 0
    skew_id: int = 0
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    alpha_z: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class DampFreqRange:
    """``/DAMP/FREQUENCY_RANGE`` or ``/DAMP/FREQ_RANGE`` (M195): Frequency-range damping."""
    id: int = 0
    title: str = ""
    fmin: float = 0.0
    fmax: float = 0.0
    damp: float = 0.0
    itype: int = 0
    cdamp: float = 0.0
    grpart_id: int = 0
    tstart: float = 0.0
    tstop: float = 1.0e30
    freq_low: float = 0.0
    freq_high: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class DampFunct:
    """``/DAMP/FUNCT`` (M195): Function-dependent mass damping."""
    id: int = 0
    title: str = ""
    fct_id: int = 0
    damp_scale: float = 1.0
    itype: int = 0
    func_id: int = 0
    grnod_id: int = 0
    alpha: float = 0.0
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    alpha_z: float = 0.0
    alpha_xx: float = 0.0
    alpha_yy: float = 0.0
    alpha_zz: float = 0.0
    params: dict = field(default_factory=dict)
DampFrequencyRange = DampFreqRange


@dataclass
class SensorDistSurf:
    """``/SENSOR/DIST_SURF/id`` (M201): Surface distance threshold sensor."""
    id: int = 0
    title: str = ""
    surf_id: int = 0
    node_id: int = 0
    surf_target_id: int = 0
    dist_min: float = 0.0
    dist_max: float = 0.0
    t_delay: float = 0.0
    tdelay: float = 0.0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    tmin: float = 0.0
    dmin: float = 0.0
    dmax: float = 0.0


@dataclass
class SensorSensAndOr:
    """``/SENSOR/SENS_AND_OR/id`` (M201): Compound Boolean logical combination sensor."""
    id: int = 0
    title: str = ""
    logic_type: str = "AND"  # 'AND' | 'OR' | 'NAND' | 'NOR'
    sensor_id1: int = 0
    sensor_id2: int = 0
    sens_id1: int = 0
    sens_id2: int = 0
    t_delay: float = 0.0
    tdelay: float = 0.0



@dataclass
class SensorWork:
    """``/SENSOR/WORK/id`` (M202): Internal/plastic work threshold sensor."""
    id: int = 0
    title: str = ""
    node_id1: int = 0
    node_id2: int = 0
    object_id: int = 0
    sens_type: int = 1
    t_delay: float = 0.0
    tdelay: float = 0.0
    w_max: float = 0.0
    work_max: float = 0.0
    tmin: float = 0.0
    sect_id: int = 0
    int_id: int = 0
    rbody_id: int = 0
    rwall_id: int = 0

    def __post_init__(self):
        if self.t_delay != 0.0 and self.tdelay == 0.0:
            self.tdelay = self.t_delay
        elif self.tdelay != 0.0 and self.t_delay == 0.0:
            self.t_delay = self.tdelay
        if self.w_max != 0.0 and self.work_max == 0.0:
            self.work_max = self.w_max
        elif self.work_max != 0.0 and self.w_max == 0.0:
            self.w_max = self.work_max
        if self.object_id != 0 and self.node_id1 == 0:
            self.node_id1 = self.object_id
        elif self.node_id1 != 0 and self.object_id == 0:
            self.object_id = self.node_id1


@dataclass
class SensorPython:
    """``/SENSOR/PYTHON/id`` (M203): Python-scripted sensor function."""
    id: int = 0
    title: str = ""
    script_name: str = ""
    func_name: str = ""
    code: str = ""
    t_delay: float = 0.0
    tdelay: float = 0.0
    t_act: float = 0.0
    sensor_type: str = "PYTHON"

    def __post_init__(self):
        if self.t_delay != 0.0 and self.tdelay == 0.0:
            self.tdelay = self.t_delay
        elif self.tdelay != 0.0 and self.t_delay == 0.0:
            self.t_delay = self.tdelay


@dataclass
class SensorSubsystem:
    """``/SENSOR/{AIRBAG|MONVOL|SHELL|SOLID|SPH}/sens_ID`` (M204): Subsystem threshold sensor."""
    id: int = 0
    title: str = ""
    kind: str = "SHELL"
    target_id: int = 0
    v1: float = 0.0
    v2: float = 0.0
    tmin: float = 0.0
    tdelay: float = 0.0


SensorAirbag = SensorSubsystem
SensorShell = SensorSubsystem
SensorSolid = SensorSubsystem
SensorSph = SensorSubsystem



@dataclass
class SensorGeom:
    """``/SENSOR/GEOM/sens_ID`` (M206): Geometric distance/angle sensor."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    node3: int = 0
    itype: int = 1          # 1: distance (N1-N2), 2: angle (N1-N2-N3)
    val_min: float = 0.0
    val_max: float = 0.0
    tmin: float = 0.0
    tdelay: float = 0.0


@dataclass
class SensorRel:
    """``/SENSOR/REL/sens_ID`` (M206): Relative displacement/rotation sensor between 2 nodes."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    idir: int = 1           # 1: DX, 2: DY, 3: DZ, 4: RX, 5: RY, 6: RZ, 7: dist
    skew_id: int = 0
    val_min: float = 0.0
    val_max: float = 0.0
    tmin: float = 0.0
    tdelay: float = 0.0


@dataclass
class SensorRatio:
    """``/SENSOR/RATIO`` or ``/SENSOR/ENERGY_RATIO/sens_ID`` (M207): Energy ratio threshold sensor."""
    id: int = 1
    title: str = ""
    ratio_type: int = 1      # 1: Hourglass/Internal, 2: Sliding/Internal, 3: Contact/Internal
    val_min: float = 0.0
    val_max: float = 0.0
    tmin: float = 0.0
    tdelay: float = 0.0


@dataclass
class SensorShearLock:
    """``/SENSOR/SHEAR_LOCK/sens_ID`` (M207): Shear locking / hourglass sensor."""
    id: int = 1
    title: str = ""
    part_id: int = 0
    val_max: float = 0.0
    tmin: float = 0.0
    tdelay: float = 0.0


@dataclass
class SensorGap:
    """``/SENSOR/TIME_GAP`` or ``/SENSOR/GAP`` (M216): Distance threshold trigger sensor."""
    id: int = 1
    title: str = ""
    node1: int = 0
    node2: int = 0
    d_gap: float = 0.0       # distance gap threshold
    t_delay: float = 0.0     # activation delay time
    isens_mode: int = 0      # 0: trig when dist < d_gap, 1: trig when dist > d_gap


@dataclass
class SensorEnergyRatio:
    """``/SENSOR/ENERGY_RATIO`` or ``/SENSOR/ENG_RATIO`` (M217): Energy ratio limit trigger sensor."""
    id: int = 1
    title: str = ""
    ratio_max: float = 1e30  # upper energy ratio limit
    ratio_min: float = 0.0   # lower energy ratio limit
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorCrossSection:
    """``/SENSOR/CROSSSECTION`` or ``/SENSOR/SEC_FORCE`` (M218): Cross-section force/moment limit trigger sensor."""
    id: int = 1
    title: str = ""
    sec_id: int = 0          # cross-section ID
    f_cut: float = 1e30      # cutoff force magnitude threshold
    m_cut: float = 1e30      # cutoff moment magnitude threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorRupture:
    """``/SENSOR/RUPT`` or ``/SENSOR/SHELL_FAIL`` (M219): Element failure / erosion trigger sensor."""
    id: int = 1
    title: str = ""
    elem_id: int = 0         # target element ID
    itype: int = 1           # element type (1=shell, 2=solid)
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorShearStress:
    """``/SENSOR/SHEAR`` or ``/SENSOR/SHEAR_STRESS`` (M220): Shear stress threshold sensor."""
    id: int = 1
    title: str = ""
    elem_id: int = 0         # target element ID
    tau_max: float = 1e30    # maximum shear stress threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorPressure:
    """``/SENSOR/PRESSURE`` or ``/SENSOR/PRESS`` (M221): Pressure threshold trigger sensor."""
    id: int = 1
    title: str = ""
    elem_id: int = 0         # target element ID
    p_min: float = -1e30     # minimum pressure threshold
    p_max: float = 1e30      # maximum pressure threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorMassRatio:
    """``/SENSOR/MASS`` or ``/SENSOR/MASS_RATIO`` (M222): Added mass ratio threshold sensor."""
    id: int = 1
    title: str = ""
    part_id: int = 0         # target part ID (0 for global)
    dmass_max: float = 1e30  # maximum added mass ratio threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorEnergyError:
    """``/SENSOR/ENERGY_ERROR`` or ``/SENSOR/ENG_ERROR`` (M223): Total energy error percentage sensor."""
    id: int = 1
    title: str = ""
    err_max: float = 1e30    # maximum energy error percentage threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorWorkRatio:
    """``/SENSOR/WORK_RATIO`` or ``/SENSOR/WRATIO`` (M224): Work ratio threshold sensor."""
    id: int = 1
    title: str = ""
    w_ratio_max: float = 1e30 # maximum work ratio threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpring:
    """``/SENSOR/SPRING`` or ``/SENSOR/SPRING_FORCE`` (M225): Spring force/moment threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    f_max: float = 1e30      # maximum axial spring force limit
    m_max: float = 1e30      # maximum torsional spring moment limit
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorShellStrain:
    """``/SENSOR/SHELL_STRAIN`` or ``/SENSOR/STRAIN_SHELL`` (M226): Shell element strain threshold sensor."""
    id: int = 1
    title: str = ""
    shell_id: int = 0        # shell element ID to monitor
    eps_max: float = 1e30    # maximum strain threshold
    ip: int = 1              # integration point number
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSolidStrain:
    """``/SENSOR/SOLID_STRAIN`` or ``/SENSOR/STRAIN_SOLID`` (M227): Solid element strain threshold sensor."""
    id: int = 1
    title: str = ""
    solid_id: int = 0        # solid element ID to monitor
    eps_max: float = 1e30    # maximum strain threshold
    ip: int = 1              # integration point number
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorBeamStrain:
    """``/SENSOR/BEAM_STRAIN`` or ``/SENSOR/STRAIN_BEAM`` (M228): Beam element strain threshold sensor."""
    id: int = 1
    title: str = ""
    beam_id: int = 0         # beam element ID to monitor
    eps_max: float = 1e30    # maximum strain threshold
    ip: int = 1              # integration point number
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorTrussStrain:
    """``/SENSOR/TRUSS_STRAIN`` or ``/SENSOR/STRAIN_TRUSS`` (M229): Truss element strain threshold sensor."""
    id: int = 1
    title: str = ""
    truss_id: int = 0        # truss element ID to monitor
    eps_max: float = 1e30    # maximum strain threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorShellForce:
    """``/SENSOR/SHELL_FORCE`` or ``/SENSOR/FORCE_SHELL`` (M230): Shell element force/moment threshold sensor."""
    id: int = 1
    title: str = ""
    shell_id: int = 0        # shell element ID to monitor
    f_max: float = 1e30      # maximum resultant force threshold
    m_max: float = 1e30      # maximum resultant moment threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSolidForce:
    """``/SENSOR/SOLID_FORCE`` or ``/SENSOR/FORCE_SOLID`` (M231): Solid element force threshold sensor."""
    id: int = 1
    title: str = ""
    solid_id: int = 0        # solid element ID to monitor
    f_max: float = 1e30      # maximum resultant force threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorBeamForce:
    """``/SENSOR/BEAM_FORCE`` or ``/SENSOR/FORCE_BEAM`` (M232): Beam element force/moment threshold sensor."""
    id: int = 1
    title: str = ""
    beam_id: int = 0         # beam element ID to monitor
    f_max: float = 1e30      # maximum resultant force threshold
    m_max: float = 1e30      # maximum resultant moment threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorTrussForce:
    """``/SENSOR/TRUSS_FORCE`` or ``/SENSOR/FORCE_TRUSS`` (M233): Truss element axial force threshold sensor."""
    id: int = 1
    title: str = ""
    truss_id: int = 0        # truss element ID to monitor
    f_max: float = 1e30      # maximum resultant force threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringEnergy:
    """``/SENSOR/SPRING_ENERGY`` or ``/SENSOR/ENERGY_SPRING`` (M234): Spring element internal energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    e_max: float = 1e30      # maximum internal energy threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringDefl:
    """``/SENSOR/SPRING_DEFL`` or ``/SENSOR/DEF_SPRING`` (M235): Spring element elongation/deflection threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    defl_max: float = 1e30   # maximum deflection/elongation threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringRot:
    """``/SENSOR/SPRING_ROT`` or ``/SENSOR/ROT_SPRING`` (M236): Spring element rotation/twist threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    rot_max: float = 1e30    # maximum rotation/twist angle threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringRotv:
    """``/SENSOR/SPRING_ROTV`` or ``/SENSOR/ROTV_SPRING`` (M237): Spring element rotational velocity threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    rotv_max: float = 1e30   # maximum rotational velocity/spin rate threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringRota:
    """``/SENSOR/SPRING_ROTA`` or ``/SENSOR/ROTA_SPRING`` (M238): Spring element rotational acceleration threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    rota_max: float = 1e30   # maximum rotational acceleration threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringAxial:
    """``/SENSOR/SPRING_AXIAL`` or ``/SENSOR/AXIAL_SPRING`` (M239): Spring element axial force threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0        # spring element ID to monitor
    fax_max: float = 1e30     # maximum axial force threshold
    t_delay: float = 0.0      # activation delay time


@dataclass
class SensorSpringShear:
    """``/SENSOR/SPRING_SHEAR`` or ``/SENSOR/SHEAR_SPRING`` (M240): Spring element shear force threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    fsh_max: float = 1e30    # maximum shear force threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringBend:
    """``/SENSOR/SPRING_BEND`` or ``/SENSOR/BEND_SPRING`` (M241): Spring element bending moment threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    mbend_max: float = 1e30  # maximum bending moment threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringTorsion:
    """``/SENSOR/SPRING_TORSION`` or ``/SENSOR/TORSION_SPRING`` (M242): Spring element torsional moment threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    mtor_max: float = 1e30   # maximum torsional moment threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringStrainEnergy:
    """``/SENSOR/SPRING_STRAIN_ENERGY`` or ``/SENSOR/STRAIN_ENERGY_SPRING`` (M243): Spring element strain energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    estrain_max: float = 1e30 # maximum strain energy threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringKineticEnergy:
    """``/SENSOR/SPRING_KINETIC_ENERGY`` or ``/SENSOR/KINETIC_ENERGY_SPRING`` (M244): Spring element kinetic energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    ekin_max: float = 1e30   # maximum kinetic energy threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringHourglassEnergy:
    """``/SENSOR/SPRING_HOURGLASS_ENERGY`` or ``/SENSOR/HOURGLASS_ENERGY_SPRING`` (M245): Spring element hourglass energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    ehe_max: float = 1e30    # maximum hourglass energy threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringContactEnergy:
    """``/SENSOR/SPRING_CONTACT_ENERGY`` or ``/SENSOR/CONTACT_ENERGY_SPRING`` (M246): Spring element contact energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    ece_max: float = 1e30    # maximum contact energy threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringNumericalDissipation:
    """``/SENSOR/SPRING_NUMERICAL_DISSIPATION`` or ``/SENSOR/SPRING_NUM_DISS`` (M247): Spring element numerical dissipation energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    enum_max: float = 1e30   # maximum numerical dissipation energy threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringExtWork:
    """``/SENSOR/SPRING_EXT_WORK`` or ``/SENSOR/SPRING_EXTERNAL_WORK`` (M248): Spring element external work threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    ewext_max: float = 1e30  # maximum external work threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringTotEnergy:
    """``/SENSOR/SPRING_TOT_ENERGY`` or ``/SENSOR/SPRING_TOTAL_ENERGY`` (M249): Spring element total energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    etot_max: float = 1e30   # maximum total energy threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringMassEnergy:
    """``/SENSOR/SPRING_MASS_ENERGY`` or ``/SENSOR/SPRING_MASS_ENER`` (M250): Spring element added mass energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    emass_max: float = 1e30  # maximum mass energy threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringMassChange:
    """``/SENSOR/SPRING_MASS_CHANGE`` or ``/SENSOR/SPRING_DMASS`` (M251): Spring element mass variation threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    dmass_max: float = 1e30  # maximum mass variation threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringPressure:
    """``/SENSOR/SPRING_PRESSURE`` or ``/SENSOR/SPRING_PRESS`` (M252): Spring element hydrostatic pressure / normal force threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    press_max: float = 1e30  # maximum pressure / normal force threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringTemperature:
    """``/SENSOR/SPRING_TEMPERATURE`` or ``/SENSOR/SPRING_TEMP`` (M253): Spring element temperature threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    temp_max: float = 1e30   # maximum temperature threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringVolume:
    """``/SENSOR/SPRING_VOLUME`` or ``/SENSOR/SPRING_VOL`` (M254): Spring element volume threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    vol_max: float = 1e30    # maximum volume / elongation threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringDensity:
    """``/SENSOR/SPRING_DENSITY`` or ``/SENSOR/SPRING_DENS`` (M255): Spring element material density threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0       # spring element ID to monitor
    dens_max: float = 1e30   # maximum density threshold
    t_delay: float = 0.0     # activation delay time


@dataclass
class SensorSpringEntropy:
    """``/SENSOR/SPRING_ENTROPY`` or ``/SENSOR/SPRING_ENTR`` (M256): Spring element thermal entropy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    entr_max: float = 1e30       # maximum entropy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringSoundSpeed:
    """``/SENSOR/SPRING_SOUND_SPEED`` or ``/SENSOR/SPRING_SS`` (M257): Spring element acoustic sound speed threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    sound_max: float = 1e30      # maximum sound speed threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringYieldStress:
    """``/SENSOR/SPRING_YIELD_STRESS`` or ``/SENSOR/SPRING_YIELD`` (M259): Spring element current yield stress threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    sigy_max: float = 1e30       # maximum yield stress threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringPlasticWork:
    """``/SENSOR/SPRING_PLASTIC_WORK`` or ``/SENSOR/SPRING_WPLAS`` (M260): Spring element plastic work threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    wplas_max: float = 1e30      # maximum accumulated plastic work threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringForceRate:
    """``/SENSOR/SPRING_FORCE_RATE`` or ``/SENSOR/SPRING_DF`` (M261): Spring element force time-rate threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    df_max: float = 1e30         # maximum force rate threshold |dF/dt|
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringForceImpulse:
    """``/SENSOR/SPRING_FORCE_IMPULSE`` or ``/SENSOR/SPRING_IMPULSE`` (M262): Spring element linear force impulse threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    j_max: float = 1e30          # maximum linear force impulse threshold int|F|dt
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringMomentRate:
    """``/SENSOR/SPRING_MOMENT_RATE`` or ``/SENSOR/SPRING_DM`` (M263): Spring element moment / torque time-rate threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    dm_max: float = 1e30         # maximum moment rate threshold |dM/dt|
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringMomentImpulse:
    """``/SENSOR/SPRING_MOMENT_IMPULSE`` or ``/SENSOR/SPRING_MOM_IMPULSE`` (M264): Spring element angular / moment impulse threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    h_max: float = 1e30          # maximum angular/moment impulse threshold int|M|dt
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalEnergy:
    """``/SENSOR/SPRING_TORSIONAL_ENERGY``: Spring element torsional elastic deformation energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    e_tor_max: float = 1e30      # maximum torsional energy threshold (1/2*K_theta*theta^2)
    t_delay: float = 0.0         # activation delay time
    u_tors_max: float = 1e30     # M273 alias

    def __post_init__(self):
        if self.u_tors_max != 1e30 and self.e_tor_max == 1e30:
            self.e_tor_max = self.u_tors_max
        elif self.e_tor_max != 1e30 and self.u_tors_max == 1e30:
            self.u_tors_max = self.e_tor_max


@dataclass
class SensorSpringBendingEnergy:
    """``/SENSOR/SPRING_BENDING_ENERGY``: Spring element bending/flexural elastic deformation energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    e_bend_max: float = 1e30     # maximum bending energy threshold (1/2*K_b*theta_b^2)
    t_delay: float = 0.0         # activation delay time
    u_bend_max: float = 1e30     # M274 alias

    def __post_init__(self):
        if self.u_bend_max != 1e30 and self.e_bend_max == 1e30:
            self.e_bend_max = self.u_bend_max
        elif self.e_bend_max != 1e30 and self.u_bend_max == 1e30:
            self.u_bend_max = self.e_bend_max


@dataclass
class SensorSpringTotalStrainEnergy:
    """``/SENSOR/SPRING_TOTAL_STRAIN_ENERGY`` or ``/SENSOR/SPRING_STRAIN_ENERGY`` (M267): Spring element total elastic strain energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    u_total_max: float = 1e30    # maximum total strain energy threshold (U_ax + U_sh + U_tor + U_bend)
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringVolumetricEnergy:
    """``/SENSOR/SPRING_VOLUMETRIC_ENERGY`` or ``/SENSOR/SPRING_VOL_ENERGY`` (M268): Spring element volumetric/hydrostatic elastic deformation energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    u_vol_max: float = 1e30      # maximum volumetric energy threshold (1/2*K_v*eps_vol^2)
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringShearEnergy:
    """``/SENSOR/SPRING_SHEAR_ENERGY`` or ``/SENSOR/SPRING_SH_ENERGY`` (M269): Spring element shear elastic deformation energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    u_shear_max: float = 1e30    # maximum shear energy threshold (1/2*K_s*gamma^2)
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringAxialEnergy:
    """``/SENSOR/SPRING_AXIAL_ENERGY`` or ``/SENSOR/SPRING_AX_ENERGY`` (M270): Spring element axial elastic deformation energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    u_axial_max: float = 1e30    # maximum axial energy threshold (1/2*K_a*eps_a^2)
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringDampingEnergy:
    """``/SENSOR/SPRING_DAMPING_ENERGY`` or ``/SENSOR/SPRING_DAMP_ENERGY`` (M271): Spring element damping dissipation energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    u_damp_max: float = 1e30     # maximum damping energy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringCouplingEnergy:
    """``/SENSOR/SPRING_COUPLING_ENERGY`` or ``/SENSOR/SPRING_COUP_ENERGY`` (M272): Spring element coupling energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    u_coup_max: float = 1e30     # maximum coupling energy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringPinchingEnergy:
    """``/SENSOR/SPRING_PINCHING_ENERGY`` or ``/SENSOR/SPRING_PINCH_ENERGY`` (M275): Spring element transverse pinching / squeeze energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0          # spring element ID to monitor
    u_pinch_max: float = 1e30   # maximum pinching energy threshold
    t_delay: float = 0.0        # activation delay time


@dataclass
class SensorSpringFrictionEnergy:
    """``/SENSOR/SPRING_FRICTION_ENERGY`` or ``/SENSOR/SPRING_FRICT_ENERGY`` (M276): Spring element frictional sliding dissipation energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0          # spring element ID to monitor
    u_frict_max: float = 1e30   # maximum friction dissipation energy threshold
    t_delay: float = 0.0        # activation delay time


@dataclass
class SensorSpringThermalDissipation:
    """``/SENSOR/SPRING_THERMAL_DISSIPATION`` or ``/SENSOR/SPRING_THERM_DISS`` (M277): Spring element thermal dissipation / heat energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    u_therm_max: float = 1e30    # maximum thermal dissipation energy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalWork:
    """``/SENSOR/SPRING_TOTAL_WORK`` or ``/SENSOR/SPRING_TOT_WORK`` (M278): Spring element total work energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    w_tot_max: float = 1e30      # maximum cumulative total work energy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringRotationalWork:
    """``/SENSOR/SPRING_ROTATIONAL_WORK`` or ``/SENSOR/SPRING_ROT_WORK`` (M279): Spring element rotational work energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    w_rot_max: float = 1e30      # maximum cumulative rotational work energy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTranslationalWork:
    """``/SENSOR/SPRING_TRANSLATIONAL_WORK`` or ``/SENSOR/SPRING_TRANS_WORK`` (M280): Spring element translational work energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    w_trans_max: float = 1e30    # maximum cumulative translational work energy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringShearWork:
    """``/SENSOR/SPRING_SHEAR_WORK`` or ``/SENSOR/SPRING_SHR_WORK`` (M281): Spring element transverse shear work energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    w_shear_max: float = 1e30    # maximum cumulative transverse shear work energy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalWork:
    """``/SENSOR/SPRING_NORMAL_WORK`` or ``/SENSOR/SPRING_NORM_WORK`` (M282): Spring element normal/axial work energy threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    w_norm_max: float = 1e30     # maximum cumulative normal/axial work energy threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalForce:
    """``/SENSOR/SPRING_TOTAL_FORCE`` or ``/SENSOR/SPRING_TOT_FORCE`` (M283): Spring element resultant force magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    f_tot_max: float = 1e30      # maximum resultant force magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalMoment:
    """``/SENSOR/SPRING_TOTAL_MOMENT`` or ``/SENSOR/SPRING_TOT_MOMENT`` (M284): Spring element resultant total torque/moment magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    m_tot_max: float = 1e30      # maximum resultant torque/moment magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringAngularVelocity:
    """``/SENSOR/SPRING_ANGULAR_VELOCITY`` or ``/SENSOR/SPRING_ANG_VEL`` (M285): Spring element relative rotational velocity / angular rate threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    omega_max: float = 1e30      # maximum angular velocity magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringAngularAcceleration:
    """``/SENSOR/SPRING_ANGULAR_ACCELERATION`` or ``/SENSOR/SPRING_ANG_ACC`` (M286): Spring element relative angular acceleration threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    alpha_max: float = 1e30      # maximum angular acceleration magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalRate:
    """``/SENSOR/SPRING_TORSIONAL_RATE`` or ``/SENSOR/SPRING_TORS_RATE`` (M287): Spring element torque rate-of-change / torsional loading rate threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    mdot_max: float = 1e30       # maximum torque rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalAcceleration:
    """``/SENSOR/SPRING_NORMAL_ACCELERATION`` or ``/SENSOR/SPRING_NORM_ACC`` (M288): Spring element relative normal / axial acceleration threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    accn_max: float = 1e30       # maximum axial acceleration magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringShearAcceleration:
    """``/SENSOR/SPRING_SHEAR_ACCELERATION`` or ``/SENSOR/SPRING_SHEAR_ACC`` (M289): Spring element relative transverse shear acceleration threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    accs_max: float = 1e30       # maximum shear acceleration magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringResultantAcceleration:
    """``/SENSOR/SPRING_RESULTANT_ACCELERATION`` or ``/SENSOR/SPRING_RES_ACC`` (M290): Spring element relative 3D vector resultant translational acceleration magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    accr_max: float = 1e30       # maximum resultant acceleration magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalAcceleration:
    """``/SENSOR/SPRING_TORSIONAL_ACCELERATION`` or ``/SENSOR/SPRING_TORS_ACC`` (M291): Spring element relative torsional angular acceleration threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    alphat_max: float = 1e30     # maximum torsional angular acceleration magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingAcceleration:
    """``/SENSOR/SPRING_BENDING_ACCELERATION`` or ``/SENSOR/SPRING_BEND_ACC`` (M292): Spring element relative transverse bending angular acceleration threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    alphab_max: float = 1e30     # maximum bending angular acceleration magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularAcceleration:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION`` or ``/SENSOR/SPRING_TOT_ANG_ACC`` (M293): Spring element relative 3D resultant total angular acceleration magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    alpha_tot_max: float = 1e30  # maximum resultant total angular acceleration magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalJerk:
    """``/SENSOR/SPRING_NORMAL_JERK`` or ``/SENSOR/SPRING_NORM_JERK`` (M294): Spring element relative normal / axial jerk (rate of change of linear acceleration) threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jn_max: float = 1e30         # maximum normal jerk magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringShearJerk:
    """``/SENSOR/SPRING_SHEAR_JERK`` or ``/SENSOR/SPRING_SHR_JERK`` (M295): Spring element relative transverse shear jerk (rate of change of transverse linear acceleration) threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    js_max: float = 1e30         # maximum transverse shear jerk magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringResultantJerk:
    """``/SENSOR/SPRING_RESULTANT_JERK`` or ``/SENSOR/SPRING_RES_JERK`` (M296): Spring element relative 3D resultant linear jerk (rate of change of linear acceleration) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jres_max: float = 1e30       # maximum resultant linear jerk magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalJerk:
    """``/SENSOR/SPRING_TORSIONAL_JERK`` or ``/SENSOR/SPRING_TORS_JERK`` (M297): Spring element relative torsional jerk (rate of change of angular acceleration) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_max: float = 1e30      # maximum torsional angular jerk magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingJerk:
    """``/SENSOR/SPRING_BENDING_JERK`` or ``/SENSOR/SPRING_BEND_JERK`` (M298): Spring element relative transverse bending angular jerk (rate of change of bending angular acceleration) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_max: float = 1e30      # maximum bending angular jerk magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularJerk:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_JERK`` or ``/SENSOR/SPRING_TOT_ANG_JERK`` (M299): Spring element relative 3D resultant total angular jerk (rate of change of total angular acceleration) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtota_max: float = 1e30      # maximum total angular jerk magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAccelerationRate:
    """``/SENSOR/SPRING_TOTAL_ACCELERATION_RATE`` or ``/SENSOR/SPRING_TOT_ACC_RATE`` (M300): Spring element relative 3D resultant total acceleration rate-of-change (generalized resultant 6-DOF dynamic jerk) threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jrate_max: float = 1e30      # maximum total acceleration rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalAccelerationRate:
    """``/SENSOR/SPRING_NORMAL_ACCELERATION_RATE`` or ``/SENSOR/SPRING_NORM_ACC_RATE`` (M301): Spring element relative normal / axial acceleration rate-of-change (axial jerk) threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_rate_max: float = 1e30 # maximum normal acceleration rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransverseAccelerationRate:
    """``/SENSOR/SPRING_TRANSVERSE_ACCELERATION_RATE`` or ``/SENSOR/SPRING_TRANS_ACC_RATE`` (M302): Spring element relative transverse / shear acceleration rate-of-change (shear jerk) threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_rate_max: float = 1e30 # maximum transverse acceleration rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalAccelerationRate:
    """``/SENSOR/SPRING_TORSIONAL_ACCELERATION_RATE`` or ``/SENSOR/SPRING_TORS_ACC_RATE`` (M303): Spring element relative torsional angular acceleration rate-of-change (torsional angular jerk) threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_rate_max: float = 1e30 # maximum torsional angular acceleration rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingAccelerationRate:
    """``/SENSOR/SPRING_BENDING_ACCELERATION_RATE`` or ``/SENSOR/SPRING_BEND_ACC_RATE`` (M304): Spring element relative transverse bending angular acceleration rate-of-change (bending angular jerk) threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_rate_max: float = 1e30 # maximum bending angular acceleration rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularAccelerationRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION_RATE`` or ``/SENSOR/SPRING_TOT_ANG_ACC_RATE`` (M305): Spring element relative 3D resultant total angular acceleration rate-of-change (total angular jerk) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_rate_max: float = 1e30 # maximum total angular acceleration rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAccelerationJerk:
    """``/SENSOR/SPRING_TOTAL_ACCELERATION_JERK`` or ``/SENSOR/SPRING_TOT_ACC_JERK`` (M306): Spring element relative 3D resultant total linear and angular combined acceleration rate-of-change (generalized resultant 6-DOF jerk) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_comb_max: float = 1e30  # maximum combined 6-DOF acceleration jerk magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalAccelerationJerk:
    """``/SENSOR/SPRING_NORMAL_ACCELERATION_JERK`` or ``/SENSOR/SPRING_NORM_ACC_JERK`` (M307): Spring element relative normal / axial acceleration rate-of-change (axial jerk) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_rate_max: float = 1e30 # maximum normal/axial acceleration rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransverseAccelerationJerk:
    """``/SENSOR/SPRING_TRANSVERSE_ACCELERATION_JERK`` or ``/SENSOR/SPRING_TRANS_ACC_JERK`` (M308): Spring element relative transverse / shear acceleration rate-of-change (shear jerk) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_rate_max: float = 1e30 # maximum transverse/shear acceleration rate-of-change magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalJerkRate:
    """``/SENSOR/SPRING_TORSIONAL_JERK_RATE`` or ``/SENSOR/SPRING_TORS_JERK_RATE`` (M309): Spring element relative torsional angular jerk rate-of-change (torsional angular snap/jounce) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_snap_max: float = 1e30 # maximum torsional angular jerk rate-of-change (snap) magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingJerkRate:
    """``/SENSOR/SPRING_BENDING_JERK_RATE`` or ``/SENSOR/SPRING_BEND_JERK_RATE`` (M310): Spring element relative transverse bending angular jerk rate-of-change (bending angular snap/jounce) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_snap_max: float = 1e30 # maximum bending angular jerk rate-of-change (snap) magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalJerkRate:
    """``/SENSOR/SPRING_TOTAL_JERK_RATE`` or ``/SENSOR/SPRING_TOT_JERK_RATE`` (M311): Spring element relative 3D resultant total linear and angular combined jerk rate-of-change (generalized resultant 6-DOF snap/jounce) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_snap_max: float = 1e30  # maximum combined 6-DOF jerk rate-of-change (snap) magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalJerkRate:
    """``/SENSOR/SPRING_NORMAL_JERK_RATE`` or ``/SENSOR/SPRING_NORM_JERK_RATE`` (M312): Spring element relative normal / axial acceleration rate-of-change rate (axial snap/jounce) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_snap_max: float = 1e30 # maximum axial acceleration rate-of-change rate (snap) magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransverseJerkRate:
    """``/SENSOR/SPRING_TRANSVERSE_JERK_RATE`` or ``/SENSOR/SPRING_TRANS_JERK_RATE`` (M313): Spring element relative transverse / shear acceleration rate-of-change rate (shear snap/jounce) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_snap_max: float = 1e30 # maximum shear acceleration rate-of-change rate (snap) magnitude threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalSnapRate:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M314): Spring element relative torsional angular acceleration 2nd rate-of-change (torsional angular snap/crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_crackle_max: float = 1e30 # maximum torsional angular acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_crackle_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_crackle_max = val


@dataclass
class SensorSpringBendingSnapRate:
    """``/SENSOR/SPRING_BENDING_SNAP_RATE`` or ``/SENSOR/SPRING_BEND_SNAP_RATE`` (M315): Spring element relative transverse bending angular acceleration 2nd rate-of-change (bending angular snap/crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_crackle_max: float = 1e30 # maximum bending angular acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_crackle_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_crackle_max = val


@dataclass
class SensorSpringNormalSnapRate:
    """``/SENSOR/SPRING_NORMAL_SNAP_RATE`` or ``/SENSOR/SPRING_NORM_SNAP_RATE`` (M316): Spring element relative normal / axial acceleration 2nd rate-of-change (axial snap/crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_crackle_max: float = 1e30 # maximum normal acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_crackle_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_crackle_max = val


@dataclass
class SensorSpringTransverseSnapRate:
    """``/SENSOR/SPRING_TRANSVERSE_SNAP_RATE`` or ``/SENSOR/SPRING_TRANS_SNAP_RATE`` (M317): Spring element relative transverse / shear acceleration 2nd rate-of-change (shear snap/crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_crackle_max: float = 1e30 # maximum shear acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_crackle_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_crackle_max = val


@dataclass
class SensorSpringTotalSnapRate:
    """``/SENSOR/SPRING_TOTAL_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_SNAP_RATE`` (M318): Spring element relative 3D resultant total linear and angular combined acceleration 2nd rate-of-change (generalized resultant 6-DOF snap/crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_crackle_max: float = 1e30 # maximum total combined acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_crackle_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_crackle_max = val


@dataclass
class SensorSpringNormalCrackleRate:
    """``/SENSOR/SPRING_NORMAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_NORM_CRACKLE_RATE`` (M319): Spring element relative normal / axial acceleration 3rd rate-of-change (axial crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_pop_max: float = 1e30  # maximum normal acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransverseCrackleRate:
    """``/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE`` or ``/SENSOR/SPRING_TRANS_CRACKLE_RATE`` (M320): Spring element relative transverse / shear acceleration 3rd rate-of-change (shear crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_pop_max: float = 1e30 # maximum shear acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalCrackleRate:
    """``/SENSOR/SPRING_TOTAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_CRACKLE_RATE`` (M321): Spring element relative 3D resultant total linear and angular combined acceleration 3rd rate-of-change (generalized resultant 6-DOF crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_pop_max: float = 1e30   # maximum resultant acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalCrackleRate:
    """``/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TORS_CRACKLE_RATE`` (M322): Spring element relative torsional angular acceleration 3rd rate-of-change (torsional crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_pop_max: float = 1e30  # maximum torsional angular acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingCrackleRate:
    """``/SENSOR/SPRING_BENDING_CRACKLE_RATE`` or ``/SENSOR/SPRING_BEND_CRACKLE_RATE`` (M323): Spring element relative bending angular acceleration 3rd rate-of-change (bending crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_pop_max: float = 1e30  # maximum bending angular acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularCrackleRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE`` (M324): Spring element relative 3D resultant total angular acceleration 3rd rate-of-change (total angular crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jang_pop_max: float = 1e30   # maximum total angular acceleration crackle rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalPopRate:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M325): Spring element relative normal / axial acceleration 3rd rate-of-change (axial pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_pop_max: float = 1e30  # maximum normal acceleration pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransversePopRate:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M326): Spring element relative transverse / shear acceleration 3rd rate-of-change (shear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_pop_max: float = 1e30 # maximum transverse acceleration pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalPopRate:
    """``/SENSOR/SPRING_TOTAL_POP_RATE`` or ``/SENSOR/SPRING_TOT_POP_RATE`` (M327): Spring element relative 3D resultant total linear acceleration 3rd rate-of-change (resultant total linear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_pop_max: float = 1e30   # maximum resultant total pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalPopRate:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORS_POP_RATE`` (M328): Spring element relative torsional angular acceleration 3rd rate-of-change (torsional pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_pop_max: float = 1e30  # maximum torsional pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingPopRate:
    """``/SENSOR/SPRING_BENDING_POP_RATE`` or ``/SENSOR/SPRING_BEND_POP_RATE`` (M329): Spring element relative bending angular acceleration 3rd rate-of-change (bending pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_pop_max: float = 1e30  # maximum bending pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularPopRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M330): Spring element relative 3D resultant total angular acceleration 3rd rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_pop_max: float = 1e30 # maximum resultant total angular pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalLockRate:
    """``/SENSOR/SPRING_NORMAL_LOCK_RATE`` or ``/SENSOR/SPRING_NORM_LOCK_RATE`` (M331): Spring element relative normal/axial acceleration 4th rate-of-change (axial lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_lock_max: float = 1e30 # maximum normal lock rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransverseLockRate:
    """``/SENSOR/SPRING_TRANSVERSE_LOCK_RATE`` or ``/SENSOR/SPRING_TRANS_LOCK_RATE`` (M332): Spring element relative transverse/shear acceleration 4th rate-of-change (shear lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_lock_max: float = 1e30 # maximum transverse lock rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalLockRate:
    """``/SENSOR/SPRING_TOTAL_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_LOCK_RATE`` (M333): Spring element relative 3D resultant total linear acceleration 4th rate-of-change (resultant total linear lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_lock_max: float = 1e30  # maximum total lock rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalLockRate:
    """``/SENSOR/SPRING_TORSIONAL_LOCK_RATE`` or ``/SENSOR/SPRING_TORS_LOCK_RATE`` (M334): Spring element relative torsional angular acceleration 4th rate-of-change (torsional angular lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_lock_max: float = 1e30 # maximum torsional lock rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingLockRate:
    """``/SENSOR/SPRING_BENDING_LOCK_RATE`` or ``/SENSOR/SPRING_BEND_LOCK_RATE`` (M335): Spring element relative transverse bending angular acceleration 4th rate-of-change (bending angular lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_lock_max: float = 1e30 # maximum bending lock rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularLockRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_ANG_LOCK_RATE`` (M336): Spring element relative 3D resultant total angular acceleration 4th rate-of-change (resultant total angular lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_lock_max: float = 1e30 # maximum total angular lock rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalDropRate:
    """``/SENSOR/SPRING_NORMAL_DROP_RATE`` or ``/SENSOR/SPRING_NORM_DROP_RATE`` (M337): Spring element relative normal / axial acceleration 5th rate-of-change (axial drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_drop_max: float = 1e30 # maximum normal drop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransverseDropRate:
    """``/SENSOR/SPRING_TRANSVERSE_DROP_RATE`` or ``/SENSOR/SPRING_TRANS_DROP_RATE`` (M338): Spring element relative transverse / shear acceleration 5th rate-of-change (shear drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_drop_max: float = 1e30 # maximum transverse drop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalDropRate:
    """``/SENSOR/SPRING_TOTAL_DROP_RATE`` or ``/SENSOR/SPRING_TOT_DROP_RATE`` (M339): Spring element relative 3D resultant total linear acceleration 5th rate-of-change (resultant total linear drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_drop_max: float = 1e30  # maximum total drop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalDropRate:
    """``/SENSOR/SPRING_TORSIONAL_DROP_RATE`` or ``/SENSOR/SPRING_TORS_DROP_RATE`` (M340): Spring element relative torsional angular acceleration 5th rate-of-change (torsional angular drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_drop_max: float = 1e30 # maximum torsional drop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingDropRate:
    """``/SENSOR/SPRING_BENDING_DROP_RATE`` or ``/SENSOR/SPRING_BEND_DROP_RATE`` (M341): Spring element relative transverse bending angular acceleration 5th rate-of-change (bending angular drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_drop_max: float = 1e30 # maximum bending drop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularDropRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_DROP_RATE`` (M342): Spring element relative 3D resultant total angular acceleration 5th rate-of-change (resultant total angular drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_drop_max: float = 1e30 # maximum total angular drop rate threshold
    t_delay: float = 0.0         # activation delay time


    @property
    def jang_lock_max(self) -> float:
        return self.jang_drop_max

    @jang_lock_max.setter
    def jang_lock_max(self, val: float) -> None:
        self.jang_drop_max = val

    @property
    def jang_pop_max(self) -> float:
        return self.jang_drop_max

    @jang_pop_max.setter
    def jang_pop_max(self, val: float) -> None:
        self.jang_drop_max = val

    @property
    def jang_snp_max(self) -> float:
        return self.jang_drop_max

    @jang_snp_max.setter
    def jang_snp_max(self, val: float) -> None:
        self.jang_drop_max = val

    @property
    def jang_crackle_max(self) -> float:
        return self.jang_drop_max

    @jang_crackle_max.setter
    def jang_crackle_max(self, val: float) -> None:
        self.jang_drop_max = val


@dataclass
class SensorSpringNormalDriftRate:
    """``/SENSOR/SPRING_NORMAL_DRIFT_RATE`` or ``/SENSOR/SPRING_NORM_DRIFT_RATE`` (M343): Spring element relative normal / axial acceleration 6th rate-of-change (axial drift rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_drift_max: float = 1e30 # maximum normal drift rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransverseDriftRate:
    """``/SENSOR/SPRING_TRANSVERSE_DRIFT_RATE`` or ``/SENSOR/SPRING_TRANS_DRIFT_RATE`` (M344): Spring element relative transverse / shear acceleration 6th rate-of-change (shear drift rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_drift_max: float = 1e30 # maximum transverse drift rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalDriftRate:
    """``/SENSOR/SPRING_TOTAL_DRIFT_RATE`` or ``/SENSOR/SPRING_TOT_DRIFT_RATE`` (M345): Spring element relative 3D resultant total linear acceleration 6th rate-of-change (resultant total linear drift rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_drift_max: float = 1e30 # maximum total drift rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalDriftRate:
    """``/SENSOR/SPRING_TORSIONAL_DRIFT_RATE`` or ``/SENSOR/SPRING_TORS_DRIFT_RATE`` (M346): Spring element relative torsional angular acceleration 6th rate-of-change (torsional angular drift rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_drift_max: float = 1e30 # maximum torsional drift rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingDriftRate:
    """``/SENSOR/SPRING_BENDING_DRIFT_RATE`` or ``/SENSOR/SPRING_BEND_DRIFT_RATE`` (M347): Spring element relative transverse bending angular acceleration 6th rate-of-change (bending angular drift rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_drift_max: float = 1e30 # maximum bending drift rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularDriftRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_DRIFT_RATE`` or ``/SENSOR/SPRING_TOT_ANG_DRIFT_RATE`` (M348): Spring element relative 3D resultant total angular acceleration 6th rate-of-change (resultant total angular drift rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_drift_max: float = 1e30 # maximum total angular drift rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalSurgeRate:
    """``/SENSOR/SPRING_NORMAL_SURGE_RATE`` or ``/SENSOR/SPRING_NORM_SURGE_RATE`` (M349): Spring element relative normal / axial acceleration 7th rate-of-change (axial surge rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_surge_max: float = 1e30 # maximum normal surge rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransverseSurgeRate:
    """``/SENSOR/SPRING_TRANSVERSE_SURGE_RATE`` or ``/SENSOR/SPRING_TRANS_SURGE_RATE`` (M350): Spring element relative transverse / shear acceleration 7th rate-of-change (shear surge rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_surge_max: float = 1e30 # maximum transverse surge rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalSurgeRate:
    """``/SENSOR/SPRING_TOTAL_SURGE_RATE`` or ``/SENSOR/SPRING_TOT_SURGE_RATE`` (M351): Spring element relative 3D resultant total linear acceleration 7th rate-of-change (resultant total linear surge rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_surge_max: float = 1e30 # maximum total surge rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalSurgeRate:
    """``/SENSOR/SPRING_TORSIONAL_SURGE_RATE`` or ``/SENSOR/SPRING_TORS_SURGE_RATE`` (M352): Spring element relative torsional angular acceleration 7th rate-of-change (torsional angular surge rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jrot_surge_max: float = 1e30 # maximum torsional surge rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingSurgeRate:
    """``/SENSOR/SPRING_BENDING_SURGE_RATE`` or ``/SENSOR/SPRING_BEND_SURGE_RATE`` (M353): Spring element relative transverse bending angular acceleration 7th rate-of-change (bending angular surge rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_surge_max: float = 1e30 # maximum bending surge rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalAngularSurgeRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_SURGE_RATE`` or ``/SENSOR/SPRING_TOT_ANG_SURGE_RATE`` (M354): Spring element relative 3D resultant total angular acceleration 7th rate-of-change (resultant total angular surge rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jrot_tot_surge_max: float = 1e30 # maximum total angular surge rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalPopRate:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M355): Spring element relative normal / axial acceleration 8th rate-of-change (axial pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_pop_max: float = 1e30  # maximum axial pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTransversePopRate:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M356): Spring element relative transverse / shear acceleration 8th rate-of-change (shear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_pop_max: float = 1e30 # maximum shear pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTotalPopRate:
    """``/SENSOR/SPRING_TOTAL_POP_RATE`` or ``/SENSOR/SPRING_TOT_POP_RATE`` (M357): Spring element relative 3D resultant total linear acceleration 8th rate-of-change (resultant total linear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_pop_max: float = 1e30   # maximum resultant total linear pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringTorsionalPopRate:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORS_POP_RATE`` (M358): Spring element relative torsional angular acceleration 8th rate-of-change (torsional angular pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_pop_max: float = 1e30  # maximum torsional angular pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringBendingPopRate:
    """``/SENSOR/SPRING_BENDING_POP_RATE`` or ``/SENSOR/SPRING_BEND_POP_RATE`` (M359): Spring element relative transverse bending angular acceleration 8th rate-of-change (bending angular pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_pop_max: float = 1e30  # maximum bending angular pop rate threshold
    t_delay: float = 0.0         # activation delay time



@dataclass
class SensorSpringTotalAngularPopRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M360): Spring element relative 3D resultant total angular acceleration 8th rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_pop_max: float = 1e30 # maximum resultant total angular pop rate threshold
    t_delay: float = 0.0         # activation delay time


@dataclass
class SensorSpringNormalCrackleRate:
    """``/SENSOR/SPRING_NORMAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_NORM_CRACKLE_RATE`` (M361): Spring element relative normal / axial acceleration 9th rate-of-change (axial crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_crk_max: float = 1e30  # maximum normal crackle rate threshold
    t_delay: float = 0.0         # activation delay time
    jnorm_pop_max: float = 1e30  # compatibility alias

    def __post_init__(self):
        if self.jnorm_pop_max != 1e30 and self.jnorm_crk_max == 1e30:
            self.jnorm_crk_max = self.jnorm_pop_max
        elif self.jnorm_crk_max != 1e30 and self.jnorm_pop_max == 1e30:
            self.jnorm_pop_max = self.jnorm_crk_max


@dataclass
class SensorSpringTransverseCrackleRate:
    """``/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE`` or ``/SENSOR/SPRING_TRANS_CRACKLE_RATE`` (M362): Spring element relative transverse / shear acceleration 9th rate-of-change (shear crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_crk_max: float = 1e30 # maximum transverse crackle rate threshold
    t_delay: float = 0.0         # activation delay time
    jtrans_pop_max: float = 1e30 # compatibility alias

    def __post_init__(self):
        if self.jtrans_pop_max != 1e30 and self.jtrans_crk_max == 1e30:
            self.jtrans_crk_max = self.jtrans_pop_max
        elif self.jtrans_crk_max != 1e30 and self.jtrans_pop_max == 1e30:
            self.jtrans_pop_max = self.jtrans_crk_max


@dataclass
class SensorSpringTotalCrackleRate:
    """``/SENSOR/SPRING_TOTAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_CRACKLE_RATE`` (M363): Spring element relative 3D resultant total linear acceleration 9th rate-of-change (resultant total linear crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_crk_max: float = 1e30   # maximum resultant total linear crackle rate threshold
    t_delay: float = 0.0         # activation delay time
    jtot_pop_max: float = 1e30   # compatibility alias

    def __post_init__(self):
        if self.jtot_pop_max != 1e30 and self.jtot_crk_max == 1e30:
            self.jtot_crk_max = self.jtot_pop_max
        elif self.jtot_crk_max != 1e30 and self.jtot_pop_max == 1e30:
            self.jtot_pop_max = self.jtot_crk_max


@dataclass
class SensorSpringTorsionalCrackleRate:
    """``/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TORS_CRACKLE_RATE`` (M364): Spring element relative torsional angular acceleration 9th rate-of-change (torsional angular crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_crk_max: float = 1e30  # maximum torsional crackle rate threshold
    t_delay: float = 0.0         # activation delay time
    jtors_pop_max: float = 1e30  # compatibility alias

    def __post_init__(self):
        if self.jtors_pop_max != 1e30 and self.jtors_crk_max == 1e30:
            self.jtors_crk_max = self.jtors_pop_max
        elif self.jtors_crk_max != 1e30 and self.jtors_pop_max == 1e30:
            self.jtors_pop_max = self.jtors_crk_max


@dataclass
class SensorSpringBendingCrackleRate:
    """``/SENSOR/SPRING_BENDING_CRACKLE_RATE`` or ``/SENSOR/SPRING_BEND_CRACKLE_RATE`` (M365): Spring element relative transverse bending angular acceleration 9th rate-of-change (bending angular crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_crk_max: float = 1e30  # maximum bending crackle rate threshold
    t_delay: float = 0.0         # activation delay time
    jbend_pop_max: float = 1e30  # compatibility alias

    def __post_init__(self):
        if self.jbend_pop_max != 1e30 and self.jbend_crk_max == 1e30:
            self.jbend_crk_max = self.jbend_pop_max
        elif self.jbend_crk_max != 1e30 and self.jbend_pop_max == 1e30:
            self.jbend_pop_max = self.jbend_crk_max


@dataclass
class SensorSpringTotalAngularCrackleRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE`` (M366): Spring element relative 3D resultant total angular acceleration 9th rate-of-change (resultant total angular crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_crk_max: float = 1e30 # maximum resultant total angular crackle rate threshold
    t_delay: float = 0.0         # activation delay time
    jtot_ang_pop_max: float = 1e30 # compatibility alias
    jang_pop_max: float = 1e30   # compatibility alias

    def __post_init__(self):
        val = 1e30
        for candidate in (self.jtot_ang_pop_max, self.jang_pop_max):
            if candidate != 1e30:
                val = candidate
                break
        if val != 1e30 and self.jtot_ang_crk_max == 1e30:
            self.jtot_ang_crk_max = val
        elif self.jtot_ang_crk_max != 1e30:
            val = self.jtot_ang_crk_max
        self.jtot_ang_pop_max = val
        self.jang_pop_max = val


@dataclass
class SensorSpringNormalSnapRate:
    """``/SENSOR/SPRING_NORMAL_SNAP_RATE`` or ``/SENSOR/SPRING_NORM_SNAP_RATE`` (M367): Spring element relative normal / axial acceleration 10th rate-of-change (axial snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_snp_max: float = 1e30  # maximum normal snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_snp_max = val


@dataclass
class SensorSpringTransverseSnapRate:
    """``/SENSOR/SPRING_TRANSVERSE_SNAP_RATE`` or ``/SENSOR/SPRING_TRANS_SNAP_RATE`` (M368): Spring element relative transverse / shear acceleration 10th rate-of-change (shear snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_snp_max: float = 1e30 # maximum transverse snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_snp_max = val


@dataclass
class SensorSpringTotalSnapRate:
    """``/SENSOR/SPRING_TOTAL_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_SNAP_RATE`` (M369): Spring element relative 3D resultant total linear acceleration 10th rate-of-change (resultant total linear snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_snp_max: float = 1e30   # maximum resultant total linear snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_snp_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_snp_max = val


@dataclass
class SensorSpringTorsionalSnapRate:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M370): Spring element relative torsional angular acceleration 10th rate-of-change (torsional angular snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_snp_max: float = 1e30  # maximum torsional angular snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_snp_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_snp_max = val


@dataclass
class SensorSpringBendingSnapRate:
    """``/SENSOR/SPRING_BENDING_SNAP_RATE`` or ``/SENSOR/SPRING_BEND_SNAP_RATE`` (M371): Spring element relative transverse bending angular acceleration 10th rate-of-change (bending angular snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_snp_max: float = 1e30  # maximum bending angular snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_snp_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_snp_max = val


@dataclass
class SensorSpringTotalAngularSnapRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_SNAP_RATE`` (M372): Spring element relative 3D resultant total angular acceleration 10th rate-of-change (resultant total angular snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_snp_max: float = 1e30 # maximum resultant total angular snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val


@dataclass
class SensorSpringNormalPopRate:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M373): Spring element relative normal / axial acceleration 11th rate-of-change (axial pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_pop_max: float = 1e30  # maximum normal / axial pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_pop_max = val


@dataclass
class SensorSpringTransversePopRate:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M374): Spring element relative transverse / shear acceleration 11th rate-of-change (shear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_pop_max: float = 1e30 # maximum transverse / shear pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_pop_max = val


@dataclass
class SensorSpringTotalPopRate:
    """``/SENSOR/SPRING_TOTAL_POP_RATE`` or ``/SENSOR/SPRING_TOT_POP_RATE`` (M375): Spring element relative 3D resultant total linear acceleration 11th rate-of-change (resultant total linear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_pop_max: float = 1e30   # maximum resultant total linear pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_pop_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_pop_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_pop_max = val


@dataclass
class SensorSpringTorsionalPopRate:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORS_POP_RATE`` (M376): Spring element relative torsional angular acceleration 11th rate-of-change (torsional pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_pop_max: float = 1e30  # maximum torsional pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_pop_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_pop_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_pop_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_pop_max = val


@dataclass
class SensorSpringBendingPopRate:
    """``/SENSOR/SPRING_BENDING_POP_RATE`` or ``/SENSOR/SPRING_BEND_POP_RATE`` (M377): Spring element relative transverse bending angular acceleration 11th rate-of-change (bending angular pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_pop_max: float = 1e30  # maximum bending angular pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_pop_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_pop_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_pop_max = val


@dataclass
class SensorSpringTotalAngularPopRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M378): Spring element relative 3D resultant total angular acceleration 11th rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtang_pop_max: float = 1e30  # maximum resultant total angular pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtang_snp_max(self) -> float:
        return self.jtang_pop_max

    @jtang_snp_max.setter
    def jtang_snp_max(self, val: float) -> None:
        self.jtang_pop_max = val

    @property
    def jtang_crackle_max(self) -> float:
        return self.jtang_pop_max

    @jtang_crackle_max.setter
    def jtang_crackle_max(self, val: float) -> None:
        self.jtang_pop_max = val

    @property
    def jtot_ang_pop_max(self) -> float:
        return self.jtang_pop_max

    @jtot_ang_pop_max.setter
    def jtot_ang_pop_max(self, val: float) -> None:
        self.jtang_pop_max = val


@dataclass
class SensorSpringNormalLockRate:
    """``/SENSOR/SPRING_NORMAL_LOCK_RATE`` or ``/SENSOR/SPRING_NORM_LOCK_RATE`` (M379): Spring element relative normal / axial acceleration 12th rate-of-change (axial lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_lock_max: float = 1e30 # maximum axial lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_pop_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_pop_max.setter
    def jnorm_pop_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_lock_max = val


@dataclass
class SensorSpringTransverseLockRate:
    """``/SENSOR/SPRING_TRANSVERSE_LOCK_RATE`` or ``/SENSOR/SPRING_TRANS_LOCK_RATE`` (M380): Spring element relative transverse / shear acceleration 12th rate-of-change (shear lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_lock_max: float = 1e30 # maximum transverse lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_pop_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_pop_max.setter
    def jtrans_pop_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_lock_max = val


@dataclass
class SensorSpringTotalLockRate:
    """``/SENSOR/SPRING_TOTAL_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_LOCK_RATE`` (M381): Spring element relative 3D resultant total linear acceleration 12th rate-of-change (resultant total lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_lock_max: float = 1e30  # maximum resultant total lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_lock_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_lock_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_lock_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_lock_max = val


@dataclass
class SensorSpringTorsionalLockRate:
    """``/SENSOR/SPRING_TORSIONAL_LOCK_RATE`` or ``/SENSOR/SPRING_TORS_LOCK_RATE`` (M382): Spring element relative torsional angular acceleration 12th rate-of-change (torsional lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_lock_max: float = 1e30 # maximum torsional lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_lock_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_lock_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_lock_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_lock_max = val



@dataclass
class SensorSpringBendingLockRate:
    """``/SENSOR/SPRING_BENDING_LOCK_RATE`` or ``/SENSOR/SPRING_BEND_LOCK_RATE`` (M383): Spring element relative transverse bending angular acceleration 12th rate-of-change (bending angular lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_lock_max: float = 1e30 # maximum bending angular lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_pop_max(self) -> float:
        return self.jbend_lock_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_lock_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_lock_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_lock_max = val


@dataclass
class SensorSpringTotalAngularLockRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_ANG_LOCK_RATE`` (M384): Spring element relative 3D resultant total angular acceleration 12th rate-of-change (resultant total angular lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_lock_max: float = 1e30 # maximum total angular lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_ang_pop_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_pop_max.setter
    def jtot_ang_pop_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_snp_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_snp_max.setter
    def jtot_ang_snp_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val


@dataclass
class SensorSpringNormalDropRate:
    """``/SENSOR/SPRING_NORMAL_DROP_RATE`` or ``/SENSOR/SPRING_NORM_DROP_RATE`` (M385): Spring element relative normal / axial acceleration 13th rate-of-change (axial drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_drop_max: float = 1e30 # maximum normal drop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_lock_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_lock_max.setter
    def jnorm_lock_max(self, val: float) -> None:
        self.jnorm_drop_max = val

    @property
    def jnorm_pop_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_pop_max.setter
    def jnorm_pop_max(self, val: float) -> None:
        self.jnorm_drop_max = val

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_drop_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_drop_max = val


@dataclass
class SensorSpringTransverseDropRate:
    """``/SENSOR/SPRING_TRANSVERSE_DROP_RATE`` or ``/SENSOR/SPRING_TRANS_DROP_RATE`` (M386): Spring element relative transverse / shear acceleration 13th rate-of-change (shear drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_drop_max: float = 1e30 # maximum transverse drop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_lock_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_lock_max.setter
    def jtrans_lock_max(self, val: float) -> None:
        self.jtrans_drop_max = val

    @property
    def jtrans_pop_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_pop_max.setter
    def jtrans_pop_max(self, val: float) -> None:
        self.jtrans_drop_max = val

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_drop_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_drop_max = val


@dataclass
class SensorSpringTotalDropRate:
    """``/SENSOR/SPRING_TOTAL_DROP_RATE`` or ``/SENSOR/SPRING_TOT_DROP_RATE`` (M387): Spring element relative 3D resultant total linear acceleration 13th rate-of-change (resultant total drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_drop_max: float = 1e30  # maximum total drop rate threshold
    t_delay: float = 0.0         # activation delay time


    @property
    def jtot_lock_max(self) -> float:
        return self.jtot_drop_max

    @jtot_lock_max.setter
    def jtot_lock_max(self, val: float) -> None:
        self.jtot_drop_max = val

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_drop_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_drop_max = val

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_drop_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_drop_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_drop_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_drop_max = val


@dataclass
class SensorSpringTorsionalDropRate:
    """``/SENSOR/SPRING_TORSIONAL_DROP_RATE`` or ``/SENSOR/SPRING_TORS_DROP_RATE`` (M388): Spring element relative torsional angular acceleration 13th rate-of-change (torsional drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_drop_max: float = 1e30 # maximum torsional drop rate threshold
    t_delay: float = 0.0         # activation delay time


    @property
    def jtors_lock_max(self) -> float:
        return self.jtors_drop_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_drop_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_drop_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_drop_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_drop_max = val


@dataclass
class SensorSpringBendingDropRate:
    """``/SENSOR/SPRING_BENDING_DROP_RATE`` or ``/SENSOR/SPRING_BEND_DROP_RATE`` (M389): Spring element relative transverse bending angular acceleration 13th rate-of-change (bending drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_drop_max: float = 1e30 # maximum bending drop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_lock_max(self) -> float:
        return self.jbend_drop_max

    @jbend_lock_max.setter
    def jbend_lock_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_pop_max(self) -> float:
        return self.jbend_drop_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_drop_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_drop_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_drop_max = val


@dataclass
class SensorSpringTotalAngularDropRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_DROP_RATE`` (M390): Spring element relative 3D resultant total angular acceleration 13th rate-of-change (resultant total angular drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jang_drop_max: float = 1e30  # maximum total angular drop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jang_lock_max(self) -> float:
        return self.jang_drop_max

    @jang_lock_max.setter
    def jang_lock_max(self, val: float) -> None:
        self.jang_drop_max = val

    @property
    def jang_pop_max(self) -> float:
        return self.jang_drop_max

    @jang_pop_max.setter
    def jang_pop_max(self, val: float) -> None:
        self.jang_drop_max = val

    @property
    def jang_snp_max(self) -> float:
        return self.jang_drop_max

    @jang_snp_max.setter
    def jang_snp_max(self, val: float) -> None:
        self.jang_drop_max = val

    @property
    def jang_crackle_max(self) -> float:
        return self.jang_drop_max

    @jang_crackle_max.setter
    def jang_crackle_max(self, val: float) -> None:
        self.jang_drop_max = val

    @property
    def jtot_ang_drop_max(self) -> float:
        return self.jang_drop_max

    @jtot_ang_drop_max.setter
    def jtot_ang_drop_max(self, val: float) -> None:
        self.jang_drop_max = val


@dataclass
class SensorSpringNormalShotRate:
    """``/SENSOR/SPRING_NORMAL_SHOT_RATE`` or ``/SENSOR/SPRING_NORM_SHOT_RATE`` (M391): Spring element relative normal / axial acceleration 14th rate-of-change (axial shot rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_shot_max: float = 1e30 # maximum normal shot rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_drop_max(self) -> float:
        return self.jnorm_shot_max

    @jnorm_drop_max.setter
    def jnorm_drop_max(self, val: float) -> None:
        self.jnorm_shot_max = val

    @property
    def jnorm_lock_max(self) -> float:
        return self.jnorm_shot_max

    @jnorm_lock_max.setter
    def jnorm_lock_max(self, val: float) -> None:
        self.jnorm_shot_max = val

    @property
    def jnorm_pop_max(self) -> float:
        return self.jnorm_shot_max

    @jnorm_pop_max.setter
    def jnorm_pop_max(self, val: float) -> None:
        self.jnorm_shot_max = val

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_shot_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_shot_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_shot_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_shot_max = val


@dataclass
class SensorSpringTransverseShotRate:
    """``/SENSOR/SPRING_TRANSVERSE_SHOT_RATE`` or ``/SENSOR/SPRING_TRANS_SHOT_RATE`` (M392): Spring element relative transverse / shear acceleration 14th rate-of-change (shear shot rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_shot_max: float = 1e30 # maximum transverse shot rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_drop_max(self) -> float:
        return self.jtrans_shot_max

    @jtrans_drop_max.setter
    def jtrans_drop_max(self, val: float) -> None:
        self.jtrans_shot_max = val

    @property
    def jtrans_lock_max(self) -> float:
        return self.jtrans_shot_max

    @jtrans_lock_max.setter
    def jtrans_lock_max(self, val: float) -> None:
        self.jtrans_shot_max = val

    @property
    def jtrans_pop_max(self) -> float:
        return self.jtrans_shot_max

    @jtrans_pop_max.setter
    def jtrans_pop_max(self, val: float) -> None:
        self.jtrans_shot_max = val

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_shot_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_shot_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_shot_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_shot_max = val


@dataclass
class SensorSpringTotalShotRate:
    """``/SENSOR/SPRING_TOTAL_SHOT_RATE`` or ``/SENSOR/SPRING_TOT_SHOT_RATE`` (M393): Spring element relative 3D resultant total linear acceleration 14th rate-of-change (resultant total shot rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_shot_max: float = 1e30  # maximum total resultant shot rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_drop_max(self) -> float:
        return self.jtot_shot_max

    @jtot_drop_max.setter
    def jtot_drop_max(self, val: float) -> None:
        self.jtot_shot_max = val


    @property
    def jtot_lock_max(self) -> float:
        return self.jtot_shot_max

    @jtot_lock_max.setter
    def jtot_lock_max(self, val: float) -> None:
        self.jtot_shot_max = val

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_shot_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_shot_max = val

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_shot_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_shot_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_shot_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_shot_max = val


@dataclass
class SensorSpringTorsionalShotRate:
    """``/SENSOR/SPRING_TORSIONAL_SHOT_RATE`` or ``/SENSOR/SPRING_TORS_SHOT_RATE`` (M394): Spring element relative torsional angular acceleration 14th rate-of-change (torsional shot rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_shot_max: float = 1e30 # maximum torsional shot rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtors_drop_max(self) -> float:
        return self.jtors_shot_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtors_shot_max = val


    @property
    def jtors_lock_max(self) -> float:
        return self.jtors_shot_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtors_shot_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_shot_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_shot_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_shot_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_shot_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_shot_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_shot_max = val


@dataclass
class SensorSpringBendingShotRate:
    """``/SENSOR/SPRING_BENDING_SHOT_RATE`` or ``/SENSOR/SPRING_BEND_SHOT_RATE`` (M395): Spring element relative transverse bending angular acceleration 14th rate-of-change (bending shot rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_shot_max: float = 1e30 # maximum bending shot rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_drop_max(self) -> float:
        return self.jbend_shot_max

    @jbend_drop_max.setter
    def jbend_drop_max(self, val: float) -> None:
        self.jbend_shot_max = val

    @property
    def jbend_lock_max(self) -> float:
        return self.jbend_shot_max

    @jbend_lock_max.setter
    def jbend_lock_max(self, val: float) -> None:
        self.jbend_shot_max = val

    @property
    def jbend_pop_max(self) -> float:
        return self.jbend_shot_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbend_shot_max = val

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_shot_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_shot_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_shot_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_shot_max = val


@dataclass
class SensorSpringTotalAngularShotRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_SHOT_RATE`` or ``/SENSOR/SPRING_TOT_ANG_SHOT_RATE`` (M396): Spring element relative 3D resultant total angular acceleration 14th rate-of-change (resultant total angular shot rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_shot_max: float = 1e30 # maximum total angular shot rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_ang_drop_max(self) -> float:
        return self.jtot_ang_shot_max

    @jtot_ang_drop_max.setter
    def jtot_ang_drop_max(self, val: float) -> None:
        self.jtot_ang_shot_max = val

    @property
    def jtot_ang_lock_max(self) -> float:
        return self.jtot_ang_shot_max

    @jtot_ang_lock_max.setter
    def jtot_ang_lock_max(self, val: float) -> None:
        self.jtot_ang_shot_max = val

    @property
    def jtot_ang_pop_max(self) -> float:
        return self.jtot_ang_shot_max

    @jtot_ang_pop_max.setter
    def jtot_ang_pop_max(self, val: float) -> None:
        self.jtot_ang_shot_max = val

    @property
    def jtot_ang_snp_max(self) -> float:
        return self.jtot_ang_shot_max

    @jtot_ang_snp_max.setter
    def jtot_ang_snp_max(self, val: float) -> None:
        self.jtot_ang_shot_max = val

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_shot_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_shot_max = val


@dataclass
class SensorSpringNormalSnapRate:
    """``/SENSOR/SPRING_NORMAL_SNAP_RATE`` or ``/SENSOR/SPRING_NORM_SNAP_RATE`` (M397): Spring element relative normal / axial acceleration 14th rate-of-change (axial snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_snp_max: float = 1e30  # maximum normal snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_shot_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_shot_max.setter
    def jnorm_shot_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_drop_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_drop_max.setter
    def jnorm_drop_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_lock_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_lock_max.setter
    def jnorm_lock_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_pop_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_pop_max.setter
    def jnorm_pop_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_snap_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_snap_max.setter
    def jnorm_snap_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def j_norm_snp_max(self) -> float:
        return self.jnorm_snp_max

    @j_norm_snp_max.setter
    def j_norm_snp_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def j_snap_norm_max(self) -> float:
        return self.jnorm_snp_max

    @j_snap_norm_max.setter
    def j_snap_norm_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def j_norm_snap_max(self) -> float:
        return self.jnorm_snp_max

    @j_norm_snap_max.setter
    def j_norm_snap_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def j_axial_snap_max(self) -> float:
        return self.jnorm_snp_max

    @j_axial_snap_max.setter
    def j_axial_snap_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_rate_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_rate_max.setter
    def jnorm_rate_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_roc_rate_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_roc_rate_max.setter
    def jnorm_roc_rate_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_drop_rate_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_drop_rate_max.setter
    def jnorm_drop_rate_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_crk_rate_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_crk_rate_max.setter
    def jnorm_crk_rate_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_pop_rate_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_pop_rate_max.setter
    def jnorm_pop_rate_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_lock_rate_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_lock_rate_max.setter
    def jnorm_lock_rate_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def jnorm_snap_rate_max(self) -> float:
        return self.jnorm_snp_max

    @jnorm_snap_rate_max.setter
    def jnorm_snap_rate_max(self, val: float) -> None:
        self.jnorm_snp_max = val

    @property
    def j_normal_snap_max(self) -> float:
        return self.jnorm_snp_max

    @j_normal_snap_max.setter
    def j_normal_snap_max(self, val: float) -> None:
        self.jnorm_snp_max = val


@dataclass
class SensorSpringTransverseSnapRate:
    """``/SENSOR/SPRING_TRANSVERSE_SNAP_RATE`` or ``/SENSOR/SPRING_TRANS_SNAP_RATE`` (M398): Spring element relative transverse / shear acceleration 14th rate-of-change (shear snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_snp_max: float = 1e30 # maximum transverse snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_shot_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_shot_max.setter
    def jtrans_shot_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_drop_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_drop_max.setter
    def jtrans_drop_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_lock_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_lock_max.setter
    def jtrans_lock_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_pop_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_pop_max.setter
    def jtrans_pop_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_snap_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_snap_max.setter
    def jtrans_snap_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def j_trans_snp_max(self) -> float:
        return self.jtrans_snp_max

    @j_trans_snp_max.setter
    def j_trans_snp_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def j_snap_trans_max(self) -> float:
        return self.jtrans_snp_max

    @j_snap_trans_max.setter
    def j_snap_trans_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def j_trans_snap_max(self) -> float:
        return self.jtrans_snp_max

    @j_trans_snap_max.setter
    def j_trans_snap_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def j_shear_snap_max(self) -> float:
        return self.jtrans_snp_max

    @j_shear_snap_max.setter
    def j_shear_snap_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_rate_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_rate_max.setter
    def jtrans_rate_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_roc_rate_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_roc_rate_max.setter
    def jtrans_roc_rate_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_drop_rate_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_drop_rate_max.setter
    def jtrans_drop_rate_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_crk_rate_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_crk_rate_max.setter
    def jtrans_crk_rate_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_pop_rate_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_pop_rate_max.setter
    def jtrans_pop_rate_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_lock_rate_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_lock_rate_max.setter
    def jtrans_lock_rate_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def jtrans_snap_rate_max(self) -> float:
        return self.jtrans_snp_max

    @jtrans_snap_rate_max.setter
    def jtrans_snap_rate_max(self, val: float) -> None:
        self.jtrans_snp_max = val

    @property
    def j_transverse_snap_max(self) -> float:
        return self.jtrans_snp_max

    @j_transverse_snap_max.setter
    def j_transverse_snap_max(self, val: float) -> None:
        self.jtrans_snp_max = val


@dataclass
class SensorSpringTotalSnapRate:
    """``/SENSOR/SPRING_TOTAL_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_SNAP_RATE`` (M399): Spring element relative 3D resultant total linear acceleration 14th rate-of-change (resultant total snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_snp_max: float = 1e30   # maximum total snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_shot_max(self) -> float:
        return self.jtot_snp_max

    @jtot_shot_max.setter
    def jtot_shot_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def jtot_drop_max(self) -> float:
        return self.jtot_snp_max

    @jtot_drop_max.setter
    def jtot_drop_max(self, val: float) -> None:
        self.jtot_snp_max = val


    @property
    def jtot_lock_max(self) -> float:
        return self.jtot_snp_max

    @jtot_lock_max.setter
    def jtot_lock_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_snp_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_snp_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_snp_max = val

    # M472 snap properties
    @property
    def jtot_snap_max(self) -> float:
        return self.jtot_snp_max

    @jtot_snap_max.setter
    def jtot_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def j_tot_snp_max(self) -> float:
        return self.jtot_snp_max

    @j_tot_snp_max.setter
    def j_tot_snp_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def j_tot_snap_max(self) -> float:
        return self.jtot_snp_max

    @j_tot_snap_max.setter
    def j_tot_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def j_total_snp_max(self) -> float:
        return self.jtot_snp_max

    @j_total_snp_max.setter
    def j_total_snp_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def j_total_snap_max(self) -> float:
        return self.jtot_snp_max

    @j_total_snap_max.setter
    def j_total_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def j_snap_tot_max(self) -> float:
        return self.jtot_snp_max

    @j_snap_tot_max.setter
    def j_snap_tot_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def j_snap_total_max(self) -> float:
        return self.jtot_snp_max

    @j_snap_total_max.setter
    def j_snap_total_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def j_linear_total_snap_max(self) -> float:
        return self.jtot_snp_max

    @j_linear_total_snap_max.setter
    def j_linear_total_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def j_linear_tot_snap_max(self) -> float:
        return self.jtot_snp_max

    @j_linear_tot_snap_max.setter
    def j_linear_tot_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def jtot_crk_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_crk_rate_max.setter
    def jtot_crk_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def jtot_pop_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_pop_rate_max.setter
    def jtot_pop_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def jtot_lock_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_lock_rate_max.setter
    def jtot_lock_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val

    @property
    def jtot_snap_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_snap_rate_max.setter
    def jtot_snap_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val


SensorSpringTotSnapRate = SensorSpringTotalSnapRate
SensorSpringTotalSnap = SensorSpringTotalSnapRate
SensorSpringTotSnap = SensorSpringTotalSnapRate
SensorSpringLinearTotalSnapRate = SensorSpringTotalSnapRate
SensorSpringLinearTotSnapRate = SensorSpringTotalSnapRate
SensorSpringResultantSnapRate = SensorSpringTotalSnapRate
SensorSpringResultantSnap = SensorSpringTotalSnapRate


@dataclass
class SensorSpringTorsionalSnapRate:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M400): Spring element relative torsional angular acceleration 14th rate-of-change (torsional snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_snp_max: float = 1e30  # maximum torsional snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtors_shot_max(self) -> float:
        return self.jtors_snp_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtors_snp_max = val

    @property
    def jtors_drop_max(self) -> float:
        return self.jtors_snp_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtors_snp_max = val


    @property
    def jtors_lock_max(self) -> float:
        return self.jtors_snp_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtors_snp_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_snp_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_snp_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_snp_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_snp_max = val


@dataclass
class SensorSpringBendingSnapRate:
    """``/SENSOR/SPRING_BENDING_SNAP_RATE`` or ``/SENSOR/SPRING_BEND_SNAP_RATE`` (M401): Spring element relative transverse bending angular acceleration 14th rate-of-change (bending snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_snp_max: float = 1e30  # maximum bending snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_shot_max(self) -> float:
        return self.jbend_snp_max

    @jbend_shot_max.setter
    def jbend_shot_max(self, val: float) -> None:
        self.jbend_snp_max = val

    @property
    def jbend_drop_max(self) -> float:
        return self.jbend_snp_max

    @jbend_drop_max.setter
    def jbend_drop_max(self, val: float) -> None:
        self.jbend_snp_max = val

    @property
    def jbend_lock_max(self) -> float:
        return self.jbend_snp_max

    @jbend_lock_max.setter
    def jbend_lock_max(self, val: float) -> None:
        self.jbend_snp_max = val

    @property
    def jbend_pop_max(self) -> float:
        return self.jbend_snp_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbend_snp_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_snp_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_snp_max = val


@dataclass
class SensorSpringTotalAngularSnapRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_SNAP_RATE`` (M402, M435): Spring element relative 3D resultant total angular acceleration 14th/20th rate-of-change (resultant total angular snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_snp_max: float = 1e30 # maximum total angular snap rate threshold
    t_delay: float = 0.0         # activation delay time
    jtang_snp_max: float = 1e30  # alias field for compatibility

    def __post_init__(self):
        if self.jtang_snp_max != 1e30 and self.jtot_ang_snp_max == 1e30:
            self.jtot_ang_snp_max = self.jtang_snp_max
        elif self.jtot_ang_snp_max != 1e30 and self.jtang_snp_max == 1e30:
            self.jtang_snp_max = self.jtot_ang_snp_max

    @property
    def jtot_ang_shot_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_shot_max.setter
    def jtot_ang_shot_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_drop_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_drop_max.setter
    def jtot_ang_drop_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_lock_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_lock_max.setter
    def jtot_ang_lock_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_pop_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_pop_max.setter
    def jtot_ang_pop_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_crk_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_crk_max.setter
    def jtot_ang_crk_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_drop_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_drop_rate_max.setter
    def jtot_ang_drop_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_rate_max.setter
    def jtot_ang_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_roc_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_roc_rate_max.setter
    def jtot_ang_roc_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_snap_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_snap_rate_max.setter
    def jtot_ang_snap_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_snp_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_snp_rate_max.setter
    def jtot_ang_snp_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_shot_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_shot_max.setter
    def jtang_shot_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_drop_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_drop_max.setter
    def jtang_drop_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_lock_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_lock_max.setter
    def jtang_lock_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_pop_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_pop_max.setter
    def jtang_pop_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_crackle_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_crackle_max.setter
    def jtang_crackle_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_crk_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_crk_max.setter
    def jtang_crk_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_drop_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_drop_rate_max.setter
    def jtang_drop_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_rate_max.setter
    def jtang_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_roc_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_roc_rate_max.setter
    def jtang_roc_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_snap_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_snap_rate_max.setter
    def jtang_snap_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_snp_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_snp_rate_max.setter
    def jtang_snp_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_max.setter
    def j_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    # M471 snap properties
    @property
    def jtot_ang_snap_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_snap_max.setter
    def jtot_ang_snap_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtang_snap_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtang_snap_max.setter
    def jtang_snap_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_tot_ang_snp_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_tot_ang_snp_max.setter
    def j_tot_ang_snp_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_tot_ang_snap_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_tot_ang_snap_max.setter
    def j_tot_ang_snap_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_total_angular_snp_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_total_angular_snp_max.setter
    def j_total_angular_snp_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_total_angular_snap_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_total_angular_snap_max.setter
    def j_total_angular_snap_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_tang_snp_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_tang_snp_max.setter
    def j_tang_snp_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_tang_snap_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_tang_snap_max.setter
    def j_tang_snap_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_snap_tot_ang_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_snap_tot_ang_max.setter
    def j_snap_tot_ang_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_snap_total_angular_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_snap_total_angular_max.setter
    def j_snap_total_angular_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def j_snap_tang_max(self) -> float:
        return self.jtot_ang_snp_max

    @j_snap_tang_max.setter
    def j_snap_tang_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_crk_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_crk_rate_max.setter
    def jtot_ang_crk_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_pop_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_pop_rate_max.setter
    def jtot_ang_pop_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_lock_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_lock_rate_max.setter
    def jtot_ang_lock_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtot_ang_snap_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtot_ang_snap_rate_max.setter
    def jtot_ang_snap_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtotal_angular_snap_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtotal_angular_snap_max.setter
    def jtotal_angular_snap_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val

    @property
    def jtotal_angular_snap_rate_max(self) -> float:
        return self.jtot_ang_snp_max

    @jtotal_angular_snap_rate_max.setter
    def jtotal_angular_snap_rate_max(self, val: float) -> None:
        self.jtot_ang_snp_max = val


SensorSpringTotAngSnapRate = SensorSpringTotalAngularSnapRate
SensorSpringTotalAngularSnap = SensorSpringTotalAngularSnapRate
SensorSpringTotAngSnap = SensorSpringTotalAngularSnapRate
SensorSpringResultantAngularSnapRate = SensorSpringTotalAngularSnapRate
SensorSpringResultantAngularSnap = SensorSpringTotalAngularSnapRate
SensorSpringTotalAngularSnapRateSensor = SensorSpringTotalAngularSnapRate


@dataclass
class SensorSpringNormalCrackleRate:
    """``/SENSOR/SPRING_NORMAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_NORM_CRACKLE_RATE`` (M403): Spring element relative normal / axial acceleration 15th rate-of-change (axial crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_crk_max: float = 1e30  # maximum normal crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_shot_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_shot_max.setter
    def jnorm_shot_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jnorm_drop_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_drop_max.setter
    def jnorm_drop_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jnorm_lock_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_lock_max.setter
    def jnorm_lock_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jnorm_pop_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_pop_max.setter
    def jnorm_pop_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_crk_max(self) -> float:
        return self.jnorm_crk_max

    @jax_crk_max.setter
    def jax_crk_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_crk_ax_max(self) -> float:
        return self.jnorm_crk_max

    @j_crk_ax_max.setter
    def j_crk_ax_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_crackle_ax_max(self) -> float:
        return self.jnorm_crk_max

    @j_crackle_ax_max.setter
    def j_crackle_ax_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_ax_crk_max(self) -> float:
        return self.jnorm_crk_max

    @j_ax_crk_max.setter
    def j_ax_crk_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_crk_max(self) -> float:
        return self.jnorm_crk_max

    @j_crk_max.setter
    def j_crk_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_crackle_max(self) -> float:
        return self.jnorm_crk_max

    @jax_crackle_max.setter
    def jax_crackle_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_crk_axial_max(self) -> float:
        return self.jnorm_crk_max

    @j_crk_axial_max.setter
    def j_crk_axial_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_crackle_axial_max(self) -> float:
        return self.jnorm_crk_max

    @j_crackle_axial_max.setter
    def j_crackle_axial_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_shot_max(self) -> float:
        return self.jnorm_crk_max

    @jax_shot_max.setter
    def jax_shot_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_drop_max(self) -> float:
        return self.jnorm_crk_max

    @jax_drop_max.setter
    def jax_drop_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_lock_max(self) -> float:
        return self.jnorm_crk_max

    @jax_lock_max.setter
    def jax_lock_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_pop_max(self) -> float:
        return self.jnorm_crk_max

    @jax_pop_max.setter
    def jax_pop_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_snp_max(self) -> float:
        return self.jnorm_crk_max

    @jax_snp_max.setter
    def jax_snp_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_snap_max(self) -> float:
        return self.jnorm_crk_max

    @jax_snap_max.setter
    def jax_snap_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_rate_max(self) -> float:
        return self.jnorm_crk_max

    @jax_rate_max.setter
    def jax_rate_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_roc_rate_max(self) -> float:
        return self.jnorm_crk_max

    @jax_roc_rate_max.setter
    def jax_roc_rate_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_drop_rate_max(self) -> float:
        return self.jnorm_crk_max

    @jax_drop_rate_max.setter
    def jax_drop_rate_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jax_crk_rate_max(self) -> float:
        return self.jnorm_crk_max

    @jax_crk_rate_max.setter
    def jax_crk_rate_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    # M473 crackle properties
    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_norm_crk_max(self) -> float:
        return self.jnorm_crk_max

    @j_norm_crk_max.setter
    def j_norm_crk_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_norm_crackle_max(self) -> float:
        return self.jnorm_crk_max

    @j_norm_crackle_max.setter
    def j_norm_crackle_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_normal_crk_max(self) -> float:
        return self.jnorm_crk_max

    @j_normal_crk_max.setter
    def j_normal_crk_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_normal_crackle_max(self) -> float:
        return self.jnorm_crk_max

    @j_normal_crackle_max.setter
    def j_normal_crackle_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_crk_norm_max(self) -> float:
        return self.jnorm_crk_max

    @j_crk_norm_max.setter
    def j_crk_norm_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_crackle_norm_max(self) -> float:
        return self.jnorm_crk_max

    @j_crackle_norm_max.setter
    def j_crackle_norm_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_axial_crk_max(self) -> float:
        return self.jnorm_crk_max

    @j_axial_crk_max.setter
    def j_axial_crk_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def j_axial_crackle_max(self) -> float:
        return self.jnorm_crk_max

    @j_axial_crackle_max.setter
    def j_axial_crackle_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jnorm_crackle_rate_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_crackle_rate_max.setter
    def jnorm_crackle_rate_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jnorm_snap_rate_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_snap_rate_max.setter
    def jnorm_snap_rate_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jnorm_pop_rate_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_pop_rate_max.setter
    def jnorm_pop_rate_max(self, val: float) -> None:
        self.jnorm_crk_max = val

    @property
    def jnorm_lock_rate_max(self) -> float:
        return self.jnorm_crk_max

    @jnorm_lock_rate_max.setter
    def jnorm_lock_rate_max(self, val: float) -> None:
        self.jnorm_crk_max = val


@dataclass
class SensorSpringAxialCrackleRate:
    """``/SENSOR/SPRING_AXIAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_AXIAL_CRK_RATE`` (M436): Spring element relative normal / axial acceleration 21st rate-of-change (axial crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jax_crk_max: float = 1e30    # maximum axial crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_crk_max(self) -> float:
        return self.jax_crk_max

    @jnorm_crk_max.setter
    def jnorm_crk_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def j_crk_ax_max(self) -> float:
        return self.jax_crk_max

    @j_crk_ax_max.setter
    def j_crk_ax_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def j_crackle_ax_max(self) -> float:
        return self.jax_crk_max

    @j_crackle_ax_max.setter
    def j_crackle_ax_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def j_ax_crk_max(self) -> float:
        return self.jax_crk_max

    @j_ax_crk_max.setter
    def j_ax_crk_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def j_crk_max(self) -> float:
        return self.jax_crk_max

    @j_crk_max.setter
    def j_crk_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_crackle_max(self) -> float:
        return self.jax_crk_max

    @jax_crackle_max.setter
    def jax_crackle_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def j_crk_axial_max(self) -> float:
        return self.jax_crk_max

    @j_crk_axial_max.setter
    def j_crk_axial_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def j_crackle_axial_max(self) -> float:
        return self.jax_crk_max

    @j_crackle_axial_max.setter
    def j_crackle_axial_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jnorm_shot_max(self) -> float:
        return self.jax_crk_max

    @jnorm_shot_max.setter
    def jnorm_shot_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jnorm_drop_max(self) -> float:
        return self.jax_crk_max

    @jnorm_drop_max.setter
    def jnorm_drop_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jnorm_lock_max(self) -> float:
        return self.jax_crk_max

    @jnorm_lock_max.setter
    def jnorm_lock_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jnorm_pop_max(self) -> float:
        return self.jax_crk_max

    @jnorm_pop_max.setter
    def jnorm_pop_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jnorm_snp_max(self) -> float:
        return self.jax_crk_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_shot_max(self) -> float:
        return self.jax_crk_max

    @jax_shot_max.setter
    def jax_shot_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_drop_max(self) -> float:
        return self.jax_crk_max

    @jax_drop_max.setter
    def jax_drop_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_lock_max(self) -> float:
        return self.jax_crk_max

    @jax_lock_max.setter
    def jax_lock_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_pop_max(self) -> float:
        return self.jax_crk_max

    @jax_pop_max.setter
    def jax_pop_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_snp_max(self) -> float:
        return self.jax_crk_max

    @jax_snp_max.setter
    def jax_snp_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_snap_max(self) -> float:
        return self.jax_crk_max

    @jax_snap_max.setter
    def jax_snap_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_rate_max(self) -> float:
        return self.jax_crk_max

    @jax_rate_max.setter
    def jax_rate_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_roc_rate_max(self) -> float:
        return self.jax_crk_max

    @jax_roc_rate_max.setter
    def jax_roc_rate_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_drop_rate_max(self) -> float:
        return self.jax_crk_max

    @jax_drop_rate_max.setter
    def jax_drop_rate_max(self, val: float) -> None:
        self.jax_crk_max = val

    @property
    def jax_crk_rate_max(self) -> float:
        return self.jax_crk_max

    @jax_crk_rate_max.setter
    def jax_crk_rate_max(self, val: float) -> None:
        self.jax_crk_max = val


@dataclass
class SensorSpringTransverseCrackleRate:
    """``/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE`` or ``/SENSOR/SPRING_TRANS_CRACKLE_RATE`` (M404): Spring element relative transverse / shear acceleration 15th rate-of-change (shear crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_crk_max: float = 1e30 # maximum transverse crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_shot_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_shot_max.setter
    def jtrans_shot_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def jtrans_drop_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_drop_max.setter
    def jtrans_drop_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def jtrans_lock_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_lock_max.setter
    def jtrans_lock_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def jtrans_pop_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_pop_max.setter
    def jtrans_pop_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    # M474 crackle properties
    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_trans_crk_max(self) -> float:
        return self.jtrans_crk_max

    @j_trans_crk_max.setter
    def j_trans_crk_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_trans_crackle_max(self) -> float:
        return self.jtrans_crk_max

    @j_trans_crackle_max.setter
    def j_trans_crackle_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_transverse_crk_max(self) -> float:
        return self.jtrans_crk_max

    @j_transverse_crk_max.setter
    def j_transverse_crk_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_transverse_crackle_max(self) -> float:
        return self.jtrans_crk_max

    @j_transverse_crackle_max.setter
    def j_transverse_crackle_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_shear_crk_max(self) -> float:
        return self.jtrans_crk_max

    @j_shear_crk_max.setter
    def j_shear_crk_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_shear_crackle_max(self) -> float:
        return self.jtrans_crk_max

    @j_shear_crackle_max.setter
    def j_shear_crackle_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_crk_trans_max(self) -> float:
        return self.jtrans_crk_max

    @j_crk_trans_max.setter
    def j_crk_trans_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_crackle_trans_max(self) -> float:
        return self.jtrans_crk_max

    @j_crackle_trans_max.setter
    def j_crackle_trans_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_crk_shear_max(self) -> float:
        return self.jtrans_crk_max

    @j_crk_shear_max.setter
    def j_crk_shear_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def j_crackle_shear_max(self) -> float:
        return self.jtrans_crk_max

    @j_crackle_shear_max.setter
    def j_crackle_shear_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def jtrans_crackle_rate_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_crackle_rate_max.setter
    def jtrans_crackle_rate_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def jtrans_snap_rate_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_snap_rate_max.setter
    def jtrans_snap_rate_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def jtrans_pop_rate_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_pop_rate_max.setter
    def jtrans_pop_rate_max(self, val: float) -> None:
        self.jtrans_crk_max = val

    @property
    def jtrans_lock_rate_max(self) -> float:
        return self.jtrans_crk_max

    @jtrans_lock_rate_max.setter
    def jtrans_lock_rate_max(self, val: float) -> None:
        self.jtrans_crk_max = val


@dataclass
class SensorSpringTotalCrackleRate:
    """``/SENSOR/SPRING_TOTAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_CRACKLE_RATE`` (M405): Spring element relative 3D resultant total linear acceleration 15th rate-of-change (resultant total crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_crk_max: float = 1e30   # maximum resultant total linear crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_shot_max(self) -> float:
        return self.jtot_crk_max

    @jtot_shot_max.setter
    def jtot_shot_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_drop_max(self) -> float:
        return self.jtot_crk_max

    @jtot_drop_max.setter
    def jtot_drop_max(self, val: float) -> None:
        self.jtot_crk_max = val


    @property
    def jtot_lock_max(self) -> float:
        return self.jtot_crk_max

    @jtot_lock_max.setter
    def jtot_lock_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_crk_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_crk_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_crk_max = val


@dataclass
class SensorSpringTorsionalCrackleRate:
    """``/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TORS_CRACKLE_RATE`` (M406): Spring element relative torsional angular acceleration 15th rate-of-change (torsional crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_crk_max: float = 1e30  # maximum torsional crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtors_shot_max(self) -> float:
        return self.jtors_crk_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_drop_max(self) -> float:
        return self.jtors_crk_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtors_crk_max = val


    @property
    def jtors_lock_max(self) -> float:
        return self.jtors_crk_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_crk_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_crk_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_crk_max = val


@dataclass
class SensorSpringBendingCrackleRate:
    """``/SENSOR/SPRING_BENDING_CRACKLE_RATE`` or ``/SENSOR/SPRING_BEND_CRACKLE_RATE`` (M407): Spring element relative transverse bending angular acceleration 15th rate-of-change (bending crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_crk_max: float = 1e30  # maximum bending crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_shot_max(self) -> float:
        return self.jbend_crk_max

    @jbend_shot_max.setter
    def jbend_shot_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def jbend_drop_max(self) -> float:
        return self.jbend_crk_max

    @jbend_drop_max.setter
    def jbend_drop_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def jbend_lock_max(self) -> float:
        return self.jbend_crk_max

    @jbend_lock_max.setter
    def jbend_lock_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def jbend_pop_max(self) -> float:
        return self.jbend_crk_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_crk_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_crk_max = val

    # M475 crackle properties
    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_crk_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_bend_crk_max(self) -> float:
        return self.jbend_crk_max

    @j_bend_crk_max.setter
    def j_bend_crk_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_bend_crackle_max(self) -> float:
        return self.jbend_crk_max

    @j_bend_crackle_max.setter
    def j_bend_crackle_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_bending_crk_max(self) -> float:
        return self.jbend_crk_max

    @j_bending_crk_max.setter
    def j_bending_crk_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_bending_crackle_max(self) -> float:
        return self.jbend_crk_max

    @j_bending_crackle_max.setter
    def j_bending_crackle_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_rot_crk_max(self) -> float:
        return self.jbend_crk_max

    @j_rot_crk_max.setter
    def j_rot_crk_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_rot_crackle_max(self) -> float:
        return self.jbend_crk_max

    @j_rot_crackle_max.setter
    def j_rot_crackle_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_rotational_crk_max(self) -> float:
        return self.jbend_crk_max

    @j_rotational_crk_max.setter
    def j_rotational_crk_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_rotational_crackle_max(self) -> float:
        return self.jbend_crk_max

    @j_rotational_crackle_max.setter
    def j_rotational_crackle_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_crk_bend_max(self) -> float:
        return self.jbend_crk_max

    @j_crk_bend_max.setter
    def j_crk_bend_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_crackle_bend_max(self) -> float:
        return self.jbend_crk_max

    @j_crackle_bend_max.setter
    def j_crackle_bend_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_crk_rot_max(self) -> float:
        return self.jbend_crk_max

    @j_crk_rot_max.setter
    def j_crk_rot_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def j_crackle_rot_max(self) -> float:
        return self.jbend_crk_max

    @j_crackle_rot_max.setter
    def j_crackle_rot_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def jbend_crackle_rate_max(self) -> float:
        return self.jbend_crk_max

    @jbend_crackle_rate_max.setter
    def jbend_crackle_rate_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def jbend_snap_rate_max(self) -> float:
        return self.jbend_crk_max

    @jbend_snap_rate_max.setter
    def jbend_snap_rate_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def jbend_pop_rate_max(self) -> float:
        return self.jbend_crk_max

    @jbend_pop_rate_max.setter
    def jbend_pop_rate_max(self, val: float) -> None:
        self.jbend_crk_max = val

    @property
    def jbend_lock_rate_max(self) -> float:
        return self.jbend_crk_max

    @jbend_lock_rate_max.setter
    def jbend_lock_rate_max(self, val: float) -> None:
        self.jbend_crk_max = val


@dataclass
class SensorSpringTotalAngularCrackleRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE`` (M408): Spring element relative 3D resultant total angular acceleration 15th rate-of-change (resultant total angular crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtota_crk_max: float = 1e30  # maximum resultant total angular crackle rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_ang_crk_max(self) -> float:
        return self.jtota_crk_max

    @jtot_ang_crk_max.setter
    def jtot_ang_crk_max(self, val: float) -> None:
        self.jtota_crk_max = val

    @property
    def jtot_ang_pop_max(self) -> float:
        return self.jtota_crk_max

    @jtot_ang_pop_max.setter
    def jtot_ang_pop_max(self, val: float) -> None:
        self.jtota_crk_max = val

    @property
    def jang_pop_max(self) -> float:
        return self.jtota_crk_max

    @jang_pop_max.setter
    def jang_pop_max(self, val: float) -> None:
        self.jtota_crk_max = val

    @property
    def jang_crk_max(self) -> float:
        return self.jtota_crk_max

    @jang_crk_max.setter
    def jang_crk_max(self, val: float) -> None:
        self.jtota_crk_max = val

    @property
    def jtota_shot_max(self) -> float:
        return self.jtota_crk_max

    @jtota_shot_max.setter
    def jtota_shot_max(self, val: float) -> None:
        self.jtota_crk_max = val

    @property
    def jtota_drop_max(self) -> float:
        return self.jtota_crk_max

    @jtota_drop_max.setter
    def jtota_drop_max(self, val: float) -> None:
        self.jtota_crk_max = val

    @property
    def jtota_lock_max(self) -> float:
        return self.jtota_crk_max

    @jtota_lock_max.setter
    def jtota_lock_max(self, val: float) -> None:
        self.jtota_crk_max = val

    @property
    def jtota_pop_max(self) -> float:
        return self.jtota_crk_max

    @jtota_pop_max.setter
    def jtota_pop_max(self, val: float) -> None:
        self.jtota_crk_max = val

    @property
    def jtota_snp_max(self) -> float:
        return self.jtota_crk_max

    @jtota_snp_max.setter
    def jtota_snp_max(self, val: float) -> None:
        self.jtota_crk_max = val


@dataclass
class SensorSpringNormalPopRate:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M409): Spring element relative normal / axial acceleration 16th rate-of-change (axial pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_pop_max: float = 1e30  # maximum axial pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_shot_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_shot_max.setter
    def jnorm_shot_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_drop_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_drop_max.setter
    def jnorm_drop_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_lock_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_lock_max.setter
    def jnorm_lock_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_crk_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_crk_max.setter
    def jnorm_crk_max(self, val: float) -> None:
        self.jnorm_pop_max = val


@dataclass
class SensorSpringTransversePopRate:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M410): Spring element relative transverse / shear acceleration 16th rate-of-change (shear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_pop_max: float = 1e30 # maximum transverse / shear pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_shot_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_shot_max.setter
    def jtrans_shot_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_drop_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_drop_max.setter
    def jtrans_drop_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_lock_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_lock_max.setter
    def jtrans_lock_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_crk_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_crk_max.setter
    def jtrans_crk_max(self, val: float) -> None:
        self.jtrans_pop_max = val


@dataclass
class SensorSpringTotalPopRate:
    """``/SENSOR/SPRING_TOTAL_POP_RATE`` or ``/SENSOR/SPRING_TOT_POP_RATE`` (M411): Spring element relative 3D resultant total linear acceleration 16th rate-of-change (resultant total linear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_pop_max: float = 1e30   # maximum resultant total linear pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_pop_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_pop_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_shot_max(self) -> float:
        return self.jtot_pop_max

    @jtot_shot_max.setter
    def jtot_shot_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_drop_max(self) -> float:
        return self.jtot_pop_max

    @jtot_drop_max.setter
    def jtot_drop_max(self, val: float) -> None:
        self.jtot_pop_max = val


    @property
    def jtot_lock_max(self) -> float:
        return self.jtot_pop_max

    @jtot_lock_max.setter
    def jtot_lock_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_crk_max(self) -> float:
        return self.jtot_pop_max

    @jtot_crk_max.setter
    def jtot_crk_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def j_pop_tot_max(self) -> float:
        return self.jtot_pop_max

    @j_pop_tot_max.setter
    def j_pop_tot_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def j_tot_pop_max(self) -> float:
        return self.jtot_pop_max

    @j_tot_pop_max.setter
    def j_tot_pop_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_snap_max(self) -> float:
        return self.jtot_pop_max

    @jtot_snap_max.setter
    def jtot_snap_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_rate_max(self) -> float:
        return self.jtot_pop_max

    @jtot_rate_max.setter
    def jtot_rate_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_roc_rate_max(self) -> float:
        return self.jtot_pop_max

    @jtot_roc_rate_max.setter
    def jtot_roc_rate_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_drop_rate_max(self) -> float:
        return self.jtot_pop_max

    @jtot_drop_rate_max.setter
    def jtot_drop_rate_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_crk_rate_max(self) -> float:
        return self.jtot_pop_max

    @jtot_crk_rate_max.setter
    def jtot_crk_rate_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jtot_pop_rate_max(self) -> float:
        return self.jtot_pop_max

    @jtot_pop_rate_max.setter
    def jtot_pop_rate_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def jres_pop_max(self) -> float:
        return self.jtot_pop_max

    @jres_pop_max.setter
    def jres_pop_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def j_res_pop_max(self) -> float:
        return self.jtot_pop_max

    @j_res_pop_max.setter
    def j_res_pop_max(self, val: float) -> None:
        self.jtot_pop_max = val

    @property
    def j_pop_res_max(self) -> float:
        return self.jtot_pop_max

    @j_pop_res_max.setter
    def j_pop_res_max(self, val: float) -> None:
        self.jtot_pop_max = val


@dataclass
class SensorSpringTorsionalPopRate:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORS_POP_RATE`` (M412): Spring element relative torsional angular acceleration 16th rate-of-change (torsional pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtor_pop_max: float = 1e30   # maximum torsional angular pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtor_snp_max(self) -> float:
        return self.jtor_pop_max

    @jtor_snp_max.setter
    def jtor_snp_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_crackle_max(self) -> float:
        return self.jtor_pop_max

    @jtor_crackle_max.setter
    def jtor_crackle_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_shot_max(self) -> float:
        return self.jtor_pop_max

    @jtor_shot_max.setter
    def jtor_shot_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_drop_max(self) -> float:
        return self.jtor_pop_max

    @jtor_drop_max.setter
    def jtor_drop_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_lock_max(self) -> float:
        return self.jtor_pop_max

    @jtor_lock_max.setter
    def jtor_lock_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_crk_max(self) -> float:
        return self.jtor_pop_max

    @jtor_crk_max.setter
    def jtor_crk_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtor_pop_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtor_pop_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtor_pop_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_shot_max(self) -> float:
        return self.jtor_pop_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_drop_max(self) -> float:
        return self.jtor_pop_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtor_pop_max = val


    @property
    def jtors_lock_max(self) -> float:
        return self.jtor_pop_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_crk_max(self) -> float:
        return self.jtor_pop_max

    @jtors_crk_max.setter
    def jtors_crk_max(self, val: float) -> None:
        self.jtor_pop_max = val


@dataclass
class SensorSpringBendingPopRate:
    """``/SENSOR/SPRING_BENDING_POP_RATE`` or ``/SENSOR/SPRING_BEND_POP_RATE`` (M413): Spring element relative transverse bending angular acceleration 16th rate-of-change (bending pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_pop_max: float = 1e30  # maximum bending angular pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_pop_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_pop_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_shot_max(self) -> float:
        return self.jbend_pop_max

    @jbend_shot_max.setter
    def jbend_shot_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_drop_max(self) -> float:
        return self.jbend_pop_max

    @jbend_drop_max.setter
    def jbend_drop_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_lock_max(self) -> float:
        return self.jbend_pop_max

    @jbend_lock_max.setter
    def jbend_lock_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_crk_max(self) -> float:
        return self.jbend_pop_max

    @jbend_crk_max.setter
    def jbend_crk_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def j_pop_bend_max(self) -> float:
        return self.jbend_pop_max

    @j_pop_bend_max.setter
    def j_pop_bend_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def j_bend_pop_max(self) -> float:
        return self.jbend_pop_max

    @j_bend_pop_max.setter
    def j_bend_pop_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_snap_max(self) -> float:
        return self.jbend_pop_max

    @jbend_snap_max.setter
    def jbend_snap_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_rate_max(self) -> float:
        return self.jbend_pop_max

    @jbend_rate_max.setter
    def jbend_rate_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_roc_rate_max(self) -> float:
        return self.jbend_pop_max

    @jbend_roc_rate_max.setter
    def jbend_roc_rate_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_drop_rate_max(self) -> float:
        return self.jbend_pop_max

    @jbend_drop_rate_max.setter
    def jbend_drop_rate_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_crk_rate_max(self) -> float:
        return self.jbend_pop_max

    @jbend_crk_rate_max.setter
    def jbend_crk_rate_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jbend_pop_rate_max(self) -> float:
        return self.jbend_pop_max

    @jbend_pop_rate_max.setter
    def jbend_pop_rate_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def jflex_pop_max(self) -> float:
        return self.jbend_pop_max

    @jflex_pop_max.setter
    def jflex_pop_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def j_flex_pop_max(self) -> float:
        return self.jbend_pop_max

    @j_flex_pop_max.setter
    def j_flex_pop_max(self, val: float) -> None:
        self.jbend_pop_max = val

    @property
    def j_pop_flex_max(self) -> float:
        return self.jbend_pop_max

    @j_pop_flex_max.setter
    def j_pop_flex_max(self, val: float) -> None:
        self.jbend_pop_max = val


@dataclass
class SensorSpringTotalAngularPopRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M414): Spring element relative 3D resultant total angular acceleration 16th rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_pop_max: float = 1e30 # maximum resultant total angular pop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_ang_snp_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_snp_max.setter
    def jtot_ang_snp_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_shot_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_shot_max.setter
    def jtot_ang_shot_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_drop_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_drop_max.setter
    def jtot_ang_drop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_lock_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_lock_max.setter
    def jtot_ang_lock_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_crk_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_crk_max.setter
    def jtot_ang_crk_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_pop_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_pop_max.setter
    def jtang_pop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_snp_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_snp_max.setter
    def jtang_snp_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_crackle_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_crackle_max.setter
    def jtang_crackle_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_shot_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_shot_max.setter
    def jtang_shot_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_drop_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_drop_max.setter
    def jtang_drop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_lock_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_lock_max.setter
    def jtang_lock_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_crk_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_crk_max.setter
    def jtang_crk_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val


@dataclass
class SensorSpringNormalLockRate:
    """``/SENSOR/SPRING_NORMAL_LOCK_RATE`` or ``/SENSOR/SPRING_NORM_LOCK_RATE`` (M415): Spring element relative normal / axial acceleration 17th rate-of-change (axial lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_lock_max: float = 1e30 # maximum axial lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_shot_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_shot_max.setter
    def jnorm_shot_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_drop_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_drop_max.setter
    def jnorm_drop_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_pop_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_pop_max.setter
    def jnorm_pop_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_crk_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_crk_max.setter
    def jnorm_crk_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def j_lock_norm_max(self) -> float:
        return self.jnorm_lock_max

    @j_lock_norm_max.setter
    def j_lock_norm_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def j_norm_lock_max(self) -> float:
        return self.jnorm_lock_max

    @j_norm_lock_max.setter
    def j_norm_lock_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_snap_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_snap_max.setter
    def jnorm_snap_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_rate_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_rate_max.setter
    def jnorm_rate_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_roc_rate_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_roc_rate_max.setter
    def jnorm_roc_rate_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_drop_rate_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_drop_rate_max.setter
    def jnorm_drop_rate_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_crk_rate_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_crk_rate_max.setter
    def jnorm_crk_rate_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_pop_rate_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_pop_rate_max.setter
    def jnorm_pop_rate_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jnorm_lock_rate_max(self) -> float:
        return self.jnorm_lock_max

    @jnorm_lock_rate_max.setter
    def jnorm_lock_rate_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jaxi_lock_max(self) -> float:
        return self.jnorm_lock_max

    @jaxi_lock_max.setter
    def jaxi_lock_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def j_axi_lock_max(self) -> float:
        return self.jnorm_lock_max

    @j_axi_lock_max.setter
    def j_axi_lock_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def j_lock_axi_max(self) -> float:
        return self.jnorm_lock_max

    @j_lock_axi_max.setter
    def j_lock_axi_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def jaxial_lock_max(self) -> float:
        return self.jnorm_lock_max

    @jaxial_lock_max.setter
    def jaxial_lock_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def j_axial_lock_max(self) -> float:
        return self.jnorm_lock_max

    @j_axial_lock_max.setter
    def j_axial_lock_max(self, val: float) -> None:
        self.jnorm_lock_max = val

    @property
    def j_lock_axial_max(self) -> float:
        return self.jnorm_lock_max

    @j_lock_axial_max.setter
    def j_lock_axial_max(self, val: float) -> None:
        self.jnorm_lock_max = val


@dataclass
class SensorSpringTransverseLockRate:
    """``/SENSOR/SPRING_TRANSVERSE_LOCK_RATE`` or ``/SENSOR/SPRING_TRANS_LOCK_RATE`` (M416): Spring element relative transverse / shear acceleration 17th rate-of-change (shear lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_lock_max: float = 1e30 # maximum shear lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_shot_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_shot_max.setter
    def jtrans_shot_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_drop_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_drop_max.setter
    def jtrans_drop_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_pop_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_pop_max.setter
    def jtrans_pop_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_crk_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_crk_max.setter
    def jtrans_crk_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def j_lock_trans_max(self) -> float:
        return self.jtrans_lock_max

    @j_lock_trans_max.setter
    def j_lock_trans_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def j_trans_lock_max(self) -> float:
        return self.jtrans_lock_max

    @j_trans_lock_max.setter
    def j_trans_lock_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_snap_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_snap_max.setter
    def jtrans_snap_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_rate_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_rate_max.setter
    def jtrans_rate_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_roc_rate_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_roc_rate_max.setter
    def jtrans_roc_rate_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_drop_rate_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_drop_rate_max.setter
    def jtrans_drop_rate_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_crk_rate_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_crk_rate_max.setter
    def jtrans_crk_rate_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_pop_rate_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_pop_rate_max.setter
    def jtrans_pop_rate_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtrans_lock_rate_max(self) -> float:
        return self.jtrans_lock_max

    @jtrans_lock_rate_max.setter
    def jtrans_lock_rate_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jshear_lock_max(self) -> float:
        return self.jtrans_lock_max

    @jshear_lock_max.setter
    def jshear_lock_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def j_shear_lock_max(self) -> float:
        return self.jtrans_lock_max

    @j_shear_lock_max.setter
    def j_shear_lock_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def j_lock_shear_max(self) -> float:
        return self.jtrans_lock_max

    @j_lock_shear_max.setter
    def j_lock_shear_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def jtransverse_lock_max(self) -> float:
        return self.jtrans_lock_max

    @jtransverse_lock_max.setter
    def jtransverse_lock_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def j_transverse_lock_max(self) -> float:
        return self.jtrans_lock_max

    @j_transverse_lock_max.setter
    def j_transverse_lock_max(self, val: float) -> None:
        self.jtrans_lock_max = val

    @property
    def j_lock_transverse_max(self) -> float:
        return self.jtrans_lock_max

    @j_lock_transverse_max.setter
    def j_lock_transverse_max(self, val: float) -> None:
        self.jtrans_lock_max = val


@dataclass
class SensorSpringTotalLockRate:
    """``/SENSOR/SPRING_TOTAL_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_LOCK_RATE`` (M417): Spring element relative 3D resultant total linear acceleration 17th rate-of-change (resultant total linear lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_lock_max: float = 1e30  # maximum resultant total lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_lock_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_lock_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_shot_max(self) -> float:
        return self.jtot_lock_max

    @jtot_shot_max.setter
    def jtot_shot_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_drop_max(self) -> float:
        return self.jtot_lock_max

    @jtot_drop_max.setter
    def jtot_drop_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_lock_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_crk_max(self) -> float:
        return self.jtot_lock_max

    @jtot_crk_max.setter
    def jtot_crk_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def j_lock_tot_max(self) -> float:
        return self.jtot_lock_max

    @j_lock_tot_max.setter
    def j_lock_tot_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def j_tot_lock_max(self) -> float:
        return self.jtot_lock_max

    @j_tot_lock_max.setter
    def j_tot_lock_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_snap_max(self) -> float:
        return self.jtot_lock_max

    @jtot_snap_max.setter
    def jtot_snap_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_rate_max(self) -> float:
        return self.jtot_lock_max

    @jtot_rate_max.setter
    def jtot_rate_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_roc_rate_max(self) -> float:
        return self.jtot_lock_max

    @jtot_roc_rate_max.setter
    def jtot_roc_rate_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_drop_rate_max(self) -> float:
        return self.jtot_lock_max

    @jtot_drop_rate_max.setter
    def jtot_drop_rate_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_crk_rate_max(self) -> float:
        return self.jtot_lock_max

    @jtot_crk_rate_max.setter
    def jtot_crk_rate_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_pop_rate_max(self) -> float:
        return self.jtot_lock_max

    @jtot_pop_rate_max.setter
    def jtot_pop_rate_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtot_lock_rate_max(self) -> float:
        return self.jtot_lock_max

    @jtot_lock_rate_max.setter
    def jtot_lock_rate_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def jtotal_lock_max(self) -> float:
        return self.jtot_lock_max

    @jtotal_lock_max.setter
    def jtotal_lock_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def j_total_lock_max(self) -> float:
        return self.jtot_lock_max

    @j_total_lock_max.setter
    def j_total_lock_max(self, val: float) -> None:
        self.jtot_lock_max = val

    @property
    def j_lock_total_max(self) -> float:
        return self.jtot_lock_max

    @j_lock_total_max.setter
    def j_lock_total_max(self, val: float) -> None:
        self.jtot_lock_max = val


@dataclass
class SensorSpringTorsionalLockRate:
    """``/SENSOR/SPRING_TORSIONAL_LOCK_RATE`` or ``/SENSOR/SPRING_TORS_LOCK_RATE`` (M418): Spring element relative torsional angular acceleration 17th rate-of-change (torsional angular lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_lock_max: float = 1e30 # maximum torsional angular lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_lock_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_lock_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_shot_max(self) -> float:
        return self.jtors_lock_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_drop_max(self) -> float:
        return self.jtors_lock_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_lock_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_crk_max(self) -> float:
        return self.jtors_lock_max

    @jtors_crk_max.setter
    def jtors_crk_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def j_lock_tors_max(self) -> float:
        return self.jtors_lock_max

    @j_lock_tors_max.setter
    def j_lock_tors_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def j_tors_lock_max(self) -> float:
        return self.jtors_lock_max

    @j_tors_lock_max.setter
    def j_tors_lock_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_snap_max(self) -> float:
        return self.jtors_lock_max

    @jtors_snap_max.setter
    def jtors_snap_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_rate_max(self) -> float:
        return self.jtors_lock_max

    @jtors_rate_max.setter
    def jtors_rate_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_roc_rate_max(self) -> float:
        return self.jtors_lock_max

    @jtors_roc_rate_max.setter
    def jtors_roc_rate_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_drop_rate_max(self) -> float:
        return self.jtors_lock_max

    @jtors_drop_rate_max.setter
    def jtors_drop_rate_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_crk_rate_max(self) -> float:
        return self.jtors_lock_max

    @jtors_crk_rate_max.setter
    def jtors_crk_rate_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_pop_rate_max(self) -> float:
        return self.jtors_lock_max

    @jtors_pop_rate_max.setter
    def jtors_pop_rate_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtors_lock_rate_max(self) -> float:
        return self.jtors_lock_max

    @jtors_lock_rate_max.setter
    def jtors_lock_rate_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtorsional_lock_max(self) -> float:
        return self.jtors_lock_max

    @jtorsional_lock_max.setter
    def jtorsional_lock_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def j_torsional_lock_max(self) -> float:
        return self.jtors_lock_max

    @j_torsional_lock_max.setter
    def j_torsional_lock_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def j_lock_torsional_max(self) -> float:
        return self.jtors_lock_max

    @j_lock_torsional_max.setter
    def j_lock_torsional_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def jtorsional_angular_lock_max(self) -> float:
        return self.jtors_lock_max

    @jtorsional_angular_lock_max.setter
    def jtorsional_angular_lock_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def j_torsional_angular_lock_max(self) -> float:
        return self.jtors_lock_max

    @j_torsional_angular_lock_max.setter
    def j_torsional_angular_lock_max(self, val: float) -> None:
        self.jtors_lock_max = val

    @property
    def j_lock_torsional_angular_max(self) -> float:
        return self.jtors_lock_max

    @j_lock_torsional_angular_max.setter
    def j_lock_torsional_angular_max(self, val: float) -> None:
        self.jtors_lock_max = val


@dataclass
class SensorSpringBendingLockRate:
    """``/SENSOR/SPRING_BENDING_LOCK_RATE`` or ``/SENSOR/SPRING_BEND_LOCK_RATE`` (M419): Spring element relative transverse bending angular acceleration 17th rate-of-change (bending angular lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_lock_max: float = 1e30 # maximum bending angular lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_lock_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_lock_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_shot_max(self) -> float:
        return self.jbend_lock_max

    @jbend_shot_max.setter
    def jbend_shot_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_drop_max(self) -> float:
        return self.jbend_lock_max

    @jbend_drop_max.setter
    def jbend_drop_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_pop_max(self) -> float:
        return self.jbend_lock_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_crk_max(self) -> float:
        return self.jbend_lock_max

    @jbend_crk_max.setter
    def jbend_crk_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def j_lock_bend_max(self) -> float:
        return self.jbend_lock_max

    @j_lock_bend_max.setter
    def j_lock_bend_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def j_bend_lock_max(self) -> float:
        return self.jbend_lock_max

    @j_bend_lock_max.setter
    def j_bend_lock_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_snap_max(self) -> float:
        return self.jbend_lock_max

    @jbend_snap_max.setter
    def jbend_snap_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_rate_max(self) -> float:
        return self.jbend_lock_max

    @jbend_rate_max.setter
    def jbend_rate_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_roc_rate_max(self) -> float:
        return self.jbend_lock_max

    @jbend_roc_rate_max.setter
    def jbend_roc_rate_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_drop_rate_max(self) -> float:
        return self.jbend_lock_max

    @jbend_drop_rate_max.setter
    def jbend_drop_rate_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_crk_rate_max(self) -> float:
        return self.jbend_lock_max

    @jbend_crk_rate_max.setter
    def jbend_crk_rate_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_pop_rate_max(self) -> float:
        return self.jbend_lock_max

    @jbend_pop_rate_max.setter
    def jbend_pop_rate_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbend_lock_rate_max(self) -> float:
        return self.jbend_lock_max

    @jbend_lock_rate_max.setter
    def jbend_lock_rate_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbending_lock_max(self) -> float:
        return self.jbend_lock_max

    @jbending_lock_max.setter
    def jbending_lock_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def j_bending_lock_max(self) -> float:
        return self.jbend_lock_max

    @j_bending_lock_max.setter
    def j_bending_lock_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def j_lock_bending_max(self) -> float:
        return self.jbend_lock_max

    @j_lock_bending_max.setter
    def j_lock_bending_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def jbending_angular_lock_max(self) -> float:
        return self.jbend_lock_max

    @jbending_angular_lock_max.setter
    def jbending_angular_lock_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def j_bending_angular_lock_max(self) -> float:
        return self.jbend_lock_max

    @j_bending_angular_lock_max.setter
    def j_bending_angular_lock_max(self, val: float) -> None:
        self.jbend_lock_max = val

    @property
    def j_lock_bending_angular_max(self) -> float:
        return self.jbend_lock_max

    @j_lock_bending_angular_max.setter
    def j_lock_bending_angular_max(self, val: float) -> None:
        self.jbend_lock_max = val


@dataclass
class SensorSpringTotalAngularLockRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_ANG_LOCK_RATE`` (M420): Spring element relative 3D resultant total angular acceleration 17th rate-of-change (resultant total angular lock rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_lock_max: float = 1e30 # maximum resultant total angular lock rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_ang_snp_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_snp_max.setter
    def jtot_ang_snp_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_shot_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_shot_max.setter
    def jtot_ang_shot_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_drop_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_drop_max.setter
    def jtot_ang_drop_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_pop_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_pop_max.setter
    def jtot_ang_pop_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_crk_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_crk_max.setter
    def jtot_ang_crk_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def j_lock_tot_ang_max(self) -> float:
        return self.jtot_ang_lock_max

    @j_lock_tot_ang_max.setter
    def j_lock_tot_ang_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def j_tot_ang_lock_max(self) -> float:
        return self.jtot_ang_lock_max

    @j_tot_ang_lock_max.setter
    def j_tot_ang_lock_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_snap_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_snap_max.setter
    def jtot_ang_snap_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_rate_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_rate_max.setter
    def jtot_ang_rate_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_roc_rate_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_roc_rate_max.setter
    def jtot_ang_roc_rate_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_drop_rate_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_drop_rate_max.setter
    def jtot_ang_drop_rate_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_crk_rate_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_crk_rate_max.setter
    def jtot_ang_crk_rate_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_pop_rate_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_pop_rate_max.setter
    def jtot_ang_pop_rate_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtot_ang_lock_rate_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtot_ang_lock_rate_max.setter
    def jtot_ang_lock_rate_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtotal_angular_lock_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtotal_angular_lock_max.setter
    def jtotal_angular_lock_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def j_total_angular_lock_max(self) -> float:
        return self.jtot_ang_lock_max

    @j_total_angular_lock_max.setter
    def j_total_angular_lock_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def j_lock_total_angular_max(self) -> float:
        return self.jtot_ang_lock_max

    @j_lock_total_angular_max.setter
    def j_lock_total_angular_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def jtotal_lock_max(self) -> float:
        return self.jtot_ang_lock_max

    @jtotal_lock_max.setter
    def jtotal_lock_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def j_total_lock_max(self) -> float:
        return self.jtot_ang_lock_max

    @j_total_lock_max.setter
    def j_total_lock_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val

    @property
    def j_lock_total_max(self) -> float:
        return self.jtot_ang_lock_max

    @j_lock_total_max.setter
    def j_lock_total_max(self, val: float) -> None:
        self.jtot_ang_lock_max = val


@dataclass
class SensorSpringNormalDropRate:
    """``/SENSOR/SPRING_NORMAL_DROP_RATE`` or ``/SENSOR/SPRING_NORM_DROP_RATE`` (M421): Spring element relative normal / axial acceleration 18th rate-of-change (axial drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jnorm_drop_max: float = 1e30 # maximum axial drop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jnorm_lock_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_lock_max.setter
    def jnorm_lock_max(self, val: float) -> None:
        self.jnorm_drop_max = val

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_drop_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_drop_max = val

    @property
    def jnorm_shot_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_shot_max.setter
    def jnorm_shot_max(self, val: float) -> None:
        self.jnorm_drop_max = val

    @property
    def jnorm_pop_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_pop_max.setter
    def jnorm_pop_max(self, val: float) -> None:
        self.jnorm_drop_max = val

    @property
    def jnorm_crk_max(self) -> float:
        return self.jnorm_drop_max

    @jnorm_crk_max.setter
    def jnorm_crk_max(self, val: float) -> None:
        self.jnorm_drop_max = val


@dataclass
class SensorSpringTransverseDropRate:
    """``/SENSOR/SPRING_TRANSVERSE_DROP_RATE`` or ``/SENSOR/SPRING_TRANS_DROP_RATE`` (M422): Spring element relative transverse / shear acceleration 18th rate-of-change (shear drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtrans_drop_max: float = 1e30 # maximum shear drop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtrans_lock_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_lock_max.setter
    def jtrans_lock_max(self, val: float) -> None:
        self.jtrans_drop_max = val

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_drop_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_drop_max = val

    @property
    def jtrans_shot_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_shot_max.setter
    def jtrans_shot_max(self, val: float) -> None:
        self.jtrans_drop_max = val

    @property
    def jtrans_pop_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_pop_max.setter
    def jtrans_pop_max(self, val: float) -> None:
        self.jtrans_drop_max = val

    @property
    def jtrans_crk_max(self) -> float:
        return self.jtrans_drop_max

    @jtrans_crk_max.setter
    def jtrans_crk_max(self, val: float) -> None:
        self.jtrans_drop_max = val


@dataclass
class SensorSpringTotalDropRate:
    """``/SENSOR/SPRING_TOTAL_DROP_RATE`` or ``/SENSOR/SPRING_TOT_DROP_RATE`` (M423): Spring element relative 3D resultant total linear acceleration 18th rate-of-change (resultant total linear drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_drop_max: float = 1e30  # maximum resultant total linear drop rate threshold
    t_delay: float = 0.0         # activation delay time


    @property
    def jtot_lock_max(self) -> float:
        return self.jtot_drop_max

    @jtot_lock_max.setter
    def jtot_lock_max(self, val: float) -> None:
        self.jtot_drop_max = val

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_drop_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_drop_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_drop_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_drop_max = val

    @property
    def jtot_shot_max(self) -> float:
        return self.jtot_drop_max

    @jtot_shot_max.setter
    def jtot_shot_max(self, val: float) -> None:
        self.jtot_drop_max = val

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_drop_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_drop_max = val

    @property
    def jtot_crk_max(self) -> float:
        return self.jtot_drop_max

    @jtot_crk_max.setter
    def jtot_crk_max(self, val: float) -> None:
        self.jtot_drop_max = val


@dataclass
class SensorSpringTorsionalDropRate:
    """``/SENSOR/SPRING_TORSIONAL_DROP_RATE`` or ``/SENSOR/SPRING_TORS_DROP_RATE`` (M424): Spring element relative torsional angular acceleration 18th rate-of-change (torsional angular drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_drop_max: float = 1e30 # maximum torsional angular drop rate threshold
    t_delay: float = 0.0         # activation delay time


    @property
    def jtors_lock_max(self) -> float:
        return self.jtors_drop_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_drop_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_drop_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_shot_max(self) -> float:
        return self.jtors_drop_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_drop_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_crk_max(self) -> float:
        return self.jtors_drop_max

    @jtors_crk_max.setter
    def jtors_crk_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtors_drop_rate_max(self) -> float:
        return self.jtors_drop_max

    @jtors_drop_rate_max.setter
    def jtors_drop_rate_max(self, val: float) -> None:
        self.jtors_drop_max = val

    @property
    def jtor_drop_max(self) -> float:
        return self.jtors_drop_max

    @jtor_drop_max.setter
    def jtor_drop_max(self, val: float) -> None:
        self.jtors_drop_max = val


@dataclass
class SensorSpringBendingDropRate:
    """``/SENSOR/SPRING_BENDING_DROP_RATE`` or ``/SENSOR/SPRING_BEND_DROP_RATE`` (M425): Spring element relative transverse bending angular acceleration 18th rate-of-change (bending angular drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_drop_max: float = 1e30 # maximum bending angular drop rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_lock_max(self) -> float:
        return self.jbend_drop_max

    @jbend_lock_max.setter
    def jbend_lock_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_drop_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_drop_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_shot_max(self) -> float:
        return self.jbend_drop_max

    @jbend_shot_max.setter
    def jbend_shot_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_pop_max(self) -> float:
        return self.jbend_drop_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_crk_max(self) -> float:
        return self.jbend_drop_max

    @jbend_crk_max.setter
    def jbend_crk_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_drop_rate_max(self) -> float:
        return self.jbend_drop_max

    @jbend_drop_rate_max.setter
    def jbend_drop_rate_max(self, val: float) -> None:
        self.jbend_drop_max = val

    @property
    def jbend_rate_max(self) -> float:
        return self.jbend_drop_max

    @jbend_rate_max.setter
    def jbend_rate_max(self, val: float) -> None:
        self.jbend_drop_max = val


@dataclass
class SensorSpringTotalAngularDropRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_DROP_RATE`` (M426): Spring element relative 3D resultant total angular acceleration 18th rate-of-change (resultant total angular drop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_drop_max: float = 1e30 # maximum resultant total angular drop rate threshold
    t_delay: float = 0.0         # activation delay time
    jang_drop_max: float = 1e30

    def __post_init__(self):
        if self.jang_drop_max != 1e30 and self.jtot_ang_drop_max == 1e30:
            self.jtot_ang_drop_max = self.jang_drop_max
        elif self.jtot_ang_drop_max != 1e30 and self.jang_drop_max == 1e30:
            self.jang_drop_max = self.jtot_ang_drop_max

    @property
    def jtot_ang_lock_max(self) -> float:
        return self.jtot_ang_drop_max

    @jtot_ang_lock_max.setter
    def jtot_ang_lock_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jtot_ang_snp_max(self) -> float:
        return self.jtot_ang_drop_max

    @jtot_ang_snp_max.setter
    def jtot_ang_snp_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_drop_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jtot_ang_shot_max(self) -> float:
        return self.jtot_ang_drop_max

    @jtot_ang_shot_max.setter
    def jtot_ang_shot_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jtot_ang_pop_max(self) -> float:
        return self.jtot_ang_drop_max

    @jtot_ang_pop_max.setter
    def jtot_ang_pop_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jtot_ang_crk_max(self) -> float:
        return self.jtot_ang_drop_max

    @jtot_ang_crk_max.setter
    def jtot_ang_crk_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jtot_ang_drop_rate_max(self) -> float:
        return self.jtot_ang_drop_max

    @jtot_ang_drop_rate_max.setter
    def jtot_ang_drop_rate_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jtot_ang_rate_max(self) -> float:
        return self.jtot_ang_drop_max

    @jtot_ang_rate_max.setter
    def jtot_ang_rate_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val


    @property
    def jang_lock_max(self) -> float:
        return self.jtot_ang_drop_max

    @jang_lock_max.setter
    def jang_lock_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jang_pop_max(self) -> float:
        return self.jtot_ang_drop_max

    @jang_pop_max.setter
    def jang_pop_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jang_snp_max(self) -> float:
        return self.jtot_ang_drop_max

    @jang_snp_max.setter
    def jang_snp_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val

    @property
    def jang_crackle_max(self) -> float:
        return self.jtot_ang_drop_max

    @jang_crackle_max.setter
    def jang_crackle_max(self, val: float) -> None:
        self.jtot_ang_drop_max = val


@dataclass
class SensorSpringTorsionalRateOfChange:
    """``/SENSOR/SPRING_TORSIONAL_RATE_OF_CHANGE`` or ``/SENSOR/SPRING_TORS_RATE_OF_CHANGE`` (M427): Spring element relative torsional angular acceleration 19th rate-of-change (torsional angular rate of change) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_roc_max: float = 1e30  # maximum torsional rate of change threshold
    t_delay: float = 0.0         # activation delay time


    @property
    def jtors_lock_max(self) -> float:
        return self.jtors_roc_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtors_roc_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_roc_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_roc_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_roc_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_roc_max = val

    @property
    def jtors_shot_max(self) -> float:
        return self.jtors_roc_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtors_roc_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_roc_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_roc_max = val

    @property
    def jtors_crk_max(self) -> float:
        return self.jtors_roc_max

    @jtors_crk_max.setter
    def jtors_crk_max(self, val: float) -> None:
        self.jtors_roc_max = val

    @property
    def jtors_drop_rate_max(self) -> float:
        return self.jtors_roc_max

    @jtors_drop_rate_max.setter
    def jtors_drop_rate_max(self, val: float) -> None:
        self.jtors_roc_max = val

    @property
    def jtors_rate_max(self) -> float:
        return self.jtors_roc_max

    @jtors_rate_max.setter
    def jtors_rate_max(self, val: float) -> None:
        self.jtors_roc_max = val

    @property
    def jtors_roc_rate_max(self) -> float:
        return self.jtors_roc_max

    @jtors_roc_rate_max.setter
    def jtors_roc_rate_max(self, val: float) -> None:
        self.jtors_roc_max = val


@dataclass
class SensorSpringBendingRateOfChange:
    """``/SENSOR/SPRING_BENDING_RATE_OF_CHANGE`` or ``/SENSOR/SPRING_BEND_RATE_OF_CHANGE`` (M428): Spring element relative transverse bending angular acceleration 19th rate-of-change (bending angular rate of change) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jbend_roc_max: float = 1e30  # maximum bending rate of change threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jbend_lock_max(self) -> float:
        return self.jbend_roc_max

    @jbend_lock_max.setter
    def jbend_lock_max(self, val: float) -> None:
        self.jbend_roc_max = val

    @property
    def jbend_snp_max(self) -> float:
        return self.jbend_roc_max

    @jbend_snp_max.setter
    def jbend_snp_max(self, val: float) -> None:
        self.jbend_roc_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbend_roc_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbend_roc_max = val

    @property
    def jbend_shot_max(self) -> float:
        return self.jbend_roc_max

    @jbend_shot_max.setter
    def jbend_shot_max(self, val: float) -> None:
        self.jbend_roc_max = val

    @property
    def jbend_pop_max(self) -> float:
        return self.jbend_roc_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbend_roc_max = val

    @property
    def jbend_crk_max(self) -> float:
        return self.jbend_roc_max

    @jbend_crk_max.setter
    def jbend_crk_max(self, val: float) -> None:
        self.jbend_roc_max = val

    @property
    def jbend_drop_rate_max(self) -> float:
        return self.jbend_roc_max

    @jbend_drop_rate_max.setter
    def jbend_drop_rate_max(self, val: float) -> None:
        self.jbend_roc_max = val

    @property
    def jbend_rate_max(self) -> float:
        return self.jbend_roc_max

    @jbend_rate_max.setter
    def jbend_rate_max(self, val: float) -> None:
        self.jbend_roc_max = val

    @property
    def jbend_roc_rate_max(self) -> float:
        return self.jbend_roc_max

    @jbend_roc_rate_max.setter
    def jbend_roc_rate_max(self, val: float) -> None:
        self.jbend_roc_max = val


@dataclass
class SensorSpringTotalAngularRateOfChange:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE`` or ``/SENSOR/SPRING_TOT_ANG_RATE_OF_CHANGE`` (M429): Spring element relative 3D resultant total angular acceleration 19th rate-of-change (resultant total angular rate of change) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_ang_roc_max: float = 1e30 # maximum total angular rate of change threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jtot_ang_lock_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_lock_max.setter
    def jtot_ang_lock_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val

    @property
    def jtot_ang_snp_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_snp_max.setter
    def jtot_ang_snp_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val

    @property
    def jtot_ang_shot_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_shot_max.setter
    def jtot_ang_shot_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val

    @property
    def jtot_ang_pop_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_pop_max.setter
    def jtot_ang_pop_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val

    @property
    def jtot_ang_crk_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_crk_max.setter
    def jtot_ang_crk_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val

    @property
    def jtot_ang_drop_rate_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_drop_rate_max.setter
    def jtot_ang_drop_rate_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val

    @property
    def jtot_ang_rate_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_rate_max.setter
    def jtot_ang_rate_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val

    @property
    def jtot_ang_roc_rate_max(self) -> float:
        return self.jtot_ang_roc_max

    @jtot_ang_roc_rate_max.setter
    def jtot_ang_roc_rate_max(self, val: float) -> None:
        self.jtot_ang_roc_max = val


@dataclass
class SensorSpringAxialSnapRate:
    """``/SENSOR/SPRING_AXIAL_SNAP_RATE`` or ``/SENSOR/SPRING_AXIAL_SNAP`` (M430): Spring element relative axial linear acceleration 20th rate-of-change (axial snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jax_snp_max: float = 1e30    # maximum axial snap rate threshold
    t_delay: float = 0.0         # activation delay time

    @property
    def jax_lock_max(self) -> float:
        return self.jax_snp_max

    @jax_lock_max.setter
    def jax_lock_max(self, val: float) -> None:
        self.jax_snp_max = val

    @property
    def jax_crackle_max(self) -> float:
        return self.jax_snp_max

    @jax_crackle_max.setter
    def jax_crackle_max(self, val: float) -> None:
        self.jax_snp_max = val

    @property
    def jax_shot_max(self) -> float:
        return self.jax_snp_max

    @jax_shot_max.setter
    def jax_shot_max(self, val: float) -> None:
        self.jax_snp_max = val

    @property
    def jax_pop_max(self) -> float:
        return self.jax_snp_max

    @jax_pop_max.setter
    def jax_pop_max(self, val: float) -> None:
        self.jax_snp_max = val

    @property
    def jax_crk_max(self) -> float:
        return self.jax_snp_max

    @jax_crk_max.setter
    def jax_crk_max(self, val: float) -> None:
        self.jax_snp_max = val

    @property
    def jax_drop_rate_max(self) -> float:
        return self.jax_snp_max

    @jax_drop_rate_max.setter
    def jax_drop_rate_max(self, val: float) -> None:
        self.jax_snp_max = val

    @property
    def jax_rate_max(self) -> float:
        return self.jax_snp_max

    @jax_rate_max.setter
    def jax_rate_max(self, val: float) -> None:
        self.jax_snp_max = val

    @property
    def jax_roc_rate_max(self) -> float:
        return self.jax_snp_max

    @jax_roc_rate_max.setter
    def jax_roc_rate_max(self, val: float) -> None:
        self.jax_snp_max = val

    @property
    def jax_snap_rate_max(self) -> float:
        return self.jax_snp_max

    @jax_snap_rate_max.setter
    def jax_snap_rate_max(self, val: float) -> None:
        self.jax_snp_max = val


@dataclass
class SensorSpringTransverseSnapRate:
    """``/SENSOR/SPRING_TRANSVERSE_SNAP_RATE`` or ``/SENSOR/SPRING_TRANS_SNAP_RATE`` (M431): Spring element relative transverse / shear linear acceleration 20th rate-of-change (transverse snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtr_snp_max: float = 1e30    # maximum transverse snap rate threshold
    t_delay: float = 0.0         # activation delay time
    jtrans_snp_max: float = 1e30 # alias field for compatibility

    def __post_init__(self):
        if self.jtrans_snp_max != 1e30 and self.jtr_snp_max == 1e30:
            self.jtr_snp_max = self.jtrans_snp_max
        elif self.jtr_snp_max != 1e30 and self.jtrans_snp_max == 1e30:
            self.jtrans_snp_max = self.jtr_snp_max

    @property
    def jtr_lock_max(self) -> float:
        return self.jtr_snp_max

    @jtr_lock_max.setter
    def jtr_lock_max(self, val: float) -> None:
        self.jtr_snp_max = val

    @property
    def jtr_crackle_max(self) -> float:
        return self.jtr_snp_max

    @jtr_crackle_max.setter
    def jtr_crackle_max(self, val: float) -> None:
        self.jtr_snp_max = val

    @property
    def jtr_shot_max(self) -> float:
        return self.jtr_snp_max

    @jtr_shot_max.setter
    def jtr_shot_max(self, val: float) -> None:
        self.jtr_snp_max = val

    @property
    def jtr_pop_max(self) -> float:
        return self.jtr_snp_max

    @jtr_pop_max.setter
    def jtr_pop_max(self, val: float) -> None:
        self.jtr_snp_max = val

    @property
    def jtr_crk_max(self) -> float:
        return self.jtr_snp_max

    @jtr_crk_max.setter
    def jtr_crk_max(self, val: float) -> None:
        self.jtr_snp_max = val

    @property
    def jtr_drop_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtr_drop_rate_max.setter
    def jtr_drop_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val

    @property
    def jtr_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtr_rate_max.setter
    def jtr_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val

    @property
    def jtr_roc_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtr_roc_rate_max.setter
    def jtr_roc_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val

    @property
    def jtr_snap_rate_max(self) -> float:
        return self.jtr_snp_max

    @property
    def jtrans_shot_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_shot_max.setter
    def jtrans_shot_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_lock_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_lock_max.setter
    def jtrans_lock_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_pop_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_pop_max.setter
    def jtrans_pop_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_crk_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_crk_max.setter
    def jtrans_crk_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_drop_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_drop_rate_max.setter
    def jtrans_drop_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_rate_max.setter
    def jtrans_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_roc_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_roc_rate_max.setter
    def jtrans_roc_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_snap_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_snap_rate_max.setter
    def jtrans_snap_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_snp_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_snp_rate_max.setter
    def jtrans_snp_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_drop_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_drop_max.setter
    def jtrans_drop_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_snap_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_snap_max.setter
    def jtrans_snap_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def j_trans_snp_max(self) -> float:
        return self.jtr_snp_max

    @j_trans_snp_max.setter
    def j_trans_snp_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def j_snap_trans_max(self) -> float:
        return self.jtr_snp_max

    @j_snap_trans_max.setter
    def j_snap_trans_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def j_trans_snap_max(self) -> float:
        return self.jtr_snp_max

    @j_trans_snap_max.setter
    def j_trans_snap_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def j_shear_snap_max(self) -> float:
        return self.jtr_snp_max

    @j_shear_snap_max.setter
    def j_shear_snap_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_crk_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_crk_rate_max.setter
    def jtrans_crk_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_pop_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_pop_rate_max.setter
    def jtrans_pop_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def jtrans_lock_rate_max(self) -> float:
        return self.jtr_snp_max

    @jtrans_lock_rate_max.setter
    def jtrans_lock_rate_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val

    @property
    def j_transverse_snap_max(self) -> float:
        return self.jtr_snp_max

    @j_transverse_snap_max.setter
    def j_transverse_snap_max(self, val: float) -> None:
        self.jtr_snp_max = val
        self.jtrans_snp_max = val


@dataclass
class SensorSpringTotalSnapRate:
    """``/SENSOR/SPRING_TOTAL_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_SNAP_RATE`` (M432): Spring element relative 3D resultant total linear acceleration 20th rate-of-change (resultant total linear snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtot_snp_max: float = 1e30   # maximum total snap rate threshold
    t_delay: float = 0.0         # activation delay time
    jtotal_snp_max: float = 1e30 # alias field for compatibility

    def __post_init__(self):
        if self.jtotal_snp_max != 1e30 and self.jtot_snp_max == 1e30:
            self.jtot_snp_max = self.jtotal_snp_max
        elif self.jtot_snp_max != 1e30 and self.jtotal_snp_max == 1e30:
            self.jtotal_snp_max = self.jtot_snp_max


    @property
    def jtot_lock_max(self) -> float:
        return self.jtot_snp_max

    @jtot_lock_max.setter
    def jtot_lock_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_snp_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_shot_max(self) -> float:
        return self.jtot_snp_max

    @jtot_shot_max.setter
    def jtot_shot_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_snp_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_crk_max(self) -> float:
        return self.jtot_snp_max

    @jtot_crk_max.setter
    def jtot_crk_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_drop_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_drop_rate_max.setter
    def jtot_drop_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_rate_max.setter
    def jtot_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_roc_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_roc_rate_max.setter
    def jtot_roc_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_snap_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_snap_rate_max.setter
    def jtot_snap_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val


    @property
    def jtot_drop_max(self) -> float:
        return self.jtot_snp_max

    @jtot_drop_max.setter
    def jtot_drop_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    # M472 snap properties
    @property
    def jtot_snap_max(self) -> float:
        return self.jtot_snp_max

    @jtot_snap_max.setter
    def jtot_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def j_tot_snp_max(self) -> float:
        return self.jtot_snp_max

    @j_tot_snp_max.setter
    def j_tot_snp_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def j_tot_snap_max(self) -> float:
        return self.jtot_snp_max

    @j_tot_snap_max.setter
    def j_tot_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def j_total_snp_max(self) -> float:
        return self.jtot_snp_max

    @j_total_snp_max.setter
    def j_total_snp_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def j_total_snap_max(self) -> float:
        return self.jtot_snp_max

    @j_total_snap_max.setter
    def j_total_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def j_snap_tot_max(self) -> float:
        return self.jtot_snp_max

    @j_snap_tot_max.setter
    def j_snap_tot_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def j_snap_total_max(self) -> float:
        return self.jtot_snp_max

    @j_snap_total_max.setter
    def j_snap_total_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def j_linear_total_snap_max(self) -> float:
        return self.jtot_snp_max

    @j_linear_total_snap_max.setter
    def j_linear_total_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def j_linear_tot_snap_max(self) -> float:
        return self.jtot_snp_max

    @j_linear_tot_snap_max.setter
    def j_linear_tot_snap_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_crk_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_crk_rate_max.setter
    def jtot_crk_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_pop_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_pop_rate_max.setter
    def jtot_pop_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val

    @property
    def jtot_lock_rate_max(self) -> float:
        return self.jtot_snp_max

    @jtot_lock_rate_max.setter
    def jtot_lock_rate_max(self, val: float) -> None:
        self.jtot_snp_max = val
        self.jtotal_snp_max = val


@dataclass
class SensorSpringTorsionalSnapRate:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M433): Spring element relative torsional angular acceleration 20th rate-of-change (torsional angular snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0           # spring element ID to monitor
    jtors_snp_max: float = 1e30  # maximum torsional snap rate threshold
    t_delay: float = 0.0         # activation delay time
    jtorsional_snp_max: float = 1e30 # alias field for compatibility

    def __post_init__(self):
        if self.jtorsional_snp_max != 1e30 and self.jtors_snp_max == 1e30:
            self.jtors_snp_max = self.jtorsional_snp_max
        elif self.jtors_snp_max != 1e30 and self.jtorsional_snp_max == 1e30:
            self.jtorsional_snp_max = self.jtors_snp_max


    @property
    def jtors_lock_max(self) -> float:
        return self.jtors_snp_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_snp_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_shot_max(self) -> float:
        return self.jtors_snp_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_snp_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_crk_max(self) -> float:
        return self.jtors_snp_max

    @jtors_crk_max.setter
    def jtors_crk_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_drop_rate_max(self) -> float:
        return self.jtors_snp_max

    @jtors_drop_rate_max.setter
    def jtors_drop_rate_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_rate_max(self) -> float:
        return self.jtors_snp_max

    @jtors_rate_max.setter
    def jtors_rate_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_roc_rate_max(self) -> float:
        return self.jtors_snp_max

    @jtors_roc_rate_max.setter
    def jtors_roc_rate_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_snap_rate_max(self) -> float:
        return self.jtors_snp_max

    @jtors_snap_rate_max.setter
    def jtors_snap_rate_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_snp_rate_max(self) -> float:
        return self.jtors_snp_max

    @jtors_snp_rate_max.setter
    def jtors_snp_rate_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_snap_rate_max(self) -> float:
        return self.jtors_snp_max

    @jtang_snap_rate_max.setter
    def jtang_snap_rate_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_snp_rate_max(self) -> float:
        return self.jtors_snp_max

    @jtang_snp_rate_max.setter
    def jtang_snp_rate_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val













































































































































    @property
    def jtors_drop_max(self) -> float:
        return self.jtors_snp_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val


@dataclass
class SensorSpringTorsionalSnapRate:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M433): Spring element relative torsional angular acceleration 20th rate-of-change (torsional angular snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0             # spring element ID to monitor
    jtor_snp_max: float = 1e30     # maximum torsional snap rate threshold
    t_delay: float = 0.0           # activation delay time
    jtors_snp_max: float = 1e30    # alias field for compatibility
    jtorsional_snp_max: float = 1e30 # alias field for compatibility

    def __post_init__(self):
        if self.jtors_snp_max != 1e30 and self.jtor_snp_max == 1e30:
            self.jtor_snp_max = self.jtors_snp_max
        elif self.jtor_snp_max != 1e30 and self.jtors_snp_max == 1e30:
            self.jtors_snp_max = self.jtor_snp_max
        if self.jtorsional_snp_max != 1e30 and self.jtor_snp_max == 1e30:
            self.jtor_snp_max = self.jtorsional_snp_max
        elif self.jtor_snp_max != 1e30 and self.jtorsional_snp_max == 1e30:
            self.jtorsional_snp_max = self.jtor_snp_max

    @property
    def jtor_lock_max(self) -> float:
        return self.jtor_snp_max

    @jtor_lock_max.setter
    def jtor_lock_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_crackle_max(self) -> float:
        return self.jtor_snp_max

    @jtor_crackle_max.setter
    def jtor_crackle_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_shot_max(self) -> float:
        return self.jtor_snp_max

    @jtor_shot_max.setter
    def jtor_shot_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_pop_max(self) -> float:
        return self.jtor_snp_max

    @jtor_pop_max.setter
    def jtor_pop_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_crk_max(self) -> float:
        return self.jtor_snp_max

    @jtor_crk_max.setter
    def jtor_crk_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_drop_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtor_drop_rate_max.setter
    def jtor_drop_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtor_rate_max.setter
    def jtor_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_roc_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtor_roc_rate_max.setter
    def jtor_roc_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_snap_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtor_snap_rate_max.setter
    def jtor_snap_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_max(self) -> float:
        return self.jtor_snp_max

    @j_max.setter
    def j_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val


# ============================================================================
# M434 Suite: FailLadTransverseCoreDelaminationCrackingRate, EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy, LagmulHomologicalSpatialLinkageJoint, SensorSpringBendingSnapRate
# ============================================================================


    @property
    def jtors_shot_max(self) -> float:
        return self.jtor_snp_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_lock_max(self) -> float:
        return self.jtor_snp_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtor_snp_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtor_snp_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_crk_max(self) -> float:
        return self.jtor_snp_max

    @jtors_crk_max.setter
    def jtors_crk_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_drop_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtors_drop_rate_max.setter
    def jtors_drop_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtors_rate_max.setter
    def jtors_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_roc_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtors_roc_rate_max.setter
    def jtors_roc_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_snap_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtors_snap_rate_max.setter
    def jtors_snap_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_snp_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtors_snp_rate_max.setter
    def jtors_snp_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_shot_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_shot_max.setter
    def jtorsional_shot_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_lock_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_lock_max.setter
    def jtorsional_lock_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_pop_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_pop_max.setter
    def jtorsional_pop_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_crackle_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_crackle_max.setter
    def jtorsional_crackle_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_crk_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_crk_max.setter
    def jtorsional_crk_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_drop_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_drop_rate_max.setter
    def jtorsional_drop_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_rate_max.setter
    def jtorsional_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_roc_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_roc_rate_max.setter
    def jtorsional_roc_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_snap_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_snap_rate_max.setter
    def jtorsional_snap_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_snp_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_snp_rate_max.setter
    def jtorsional_snp_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val



    @property
    def jtang_snap_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtang_snap_rate_max.setter
    def jtang_snap_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_snp_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtang_snp_rate_max.setter
    def jtang_snp_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_shot_max(self) -> float:
        return self.jtor_snp_max

    @jtang_shot_max.setter
    def jtang_shot_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_lock_max(self) -> float:
        return self.jtor_snp_max

    @jtang_lock_max.setter
    def jtang_lock_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_pop_max(self) -> float:
        return self.jtor_snp_max

    @jtang_pop_max.setter
    def jtang_pop_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_crackle_max(self) -> float:
        return self.jtor_snp_max

    @jtang_crackle_max.setter
    def jtang_crackle_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_crk_max(self) -> float:
        return self.jtor_snp_max

    @jtang_crk_max.setter
    def jtang_crk_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_drop_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtang_drop_rate_max.setter
    def jtang_drop_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtang_rate_max.setter
    def jtang_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_roc_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtang_roc_rate_max.setter
    def jtang_roc_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_snp_max(self) -> float:
        return self.jtor_snp_max

    @jtang_snp_max.setter
    def jtang_snp_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtang_drop_max(self) -> float:
        return self.jtor_snp_max

    @jtang_drop_max.setter
    def jtang_drop_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_drop_max(self) -> float:
        return self.jtor_snp_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_drop_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_drop_max.setter
    def jtorsional_drop_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_drop_max(self) -> float:
        return self.jtor_snp_max

    @jtor_drop_max.setter
    def jtor_drop_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_snap_max(self) -> float:
        return self.jtor_snp_max

    @jtor_snap_max.setter
    def jtor_snap_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtors_snap_max(self) -> float:
        return self.jtor_snp_max

    @jtors_snap_max.setter
    def jtors_snap_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtorsional_snap_max(self) -> float:
        return self.jtor_snp_max

    @jtorsional_snap_max.setter
    def jtorsional_snap_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtwist_snap_max(self) -> float:
        return self.jtor_snp_max

    @jtwist_snap_max.setter
    def jtwist_snap_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtwst_snap_max(self) -> float:
        return self.jtor_snp_max

    @jtwst_snap_max.setter
    def jtwst_snap_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_tor_snp_max(self) -> float:
        return self.jtor_snp_max

    @j_tor_snp_max.setter
    def j_tor_snp_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_tors_snp_max(self) -> float:
        return self.jtor_snp_max

    @j_tors_snp_max.setter
    def j_tors_snp_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_torsional_snp_max(self) -> float:
        return self.jtor_snp_max

    @j_torsional_snp_max.setter
    def j_torsional_snp_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_twist_snp_max(self) -> float:
        return self.jtor_snp_max

    @j_twist_snp_max.setter
    def j_twist_snp_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_snap_tor_max(self) -> float:
        return self.jtor_snp_max

    @j_snap_tor_max.setter
    def j_snap_tor_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_snap_tors_max(self) -> float:
        return self.jtor_snp_max

    @j_snap_tors_max.setter
    def j_snap_tors_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_snap_torsional_max(self) -> float:
        return self.jtor_snp_max

    @j_snap_torsional_max.setter
    def j_snap_torsional_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def j_snap_twist_max(self) -> float:
        return self.jtor_snp_max

    @j_snap_twist_max.setter
    def j_snap_twist_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_crk_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtor_crk_rate_max.setter
    def jtor_crk_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_pop_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtor_pop_rate_max.setter
    def jtor_pop_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_lock_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtor_lock_rate_max.setter
    def jtor_lock_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val

    @property
    def jtor_snap_rate_max(self) -> float:
        return self.jtor_snp_max

    @jtor_snap_rate_max.setter
    def jtor_snap_rate_max(self, val: float) -> None:
        self.jtor_snp_max = val
        self.jtors_snp_max = val
        self.jtorsional_snp_max = val


@dataclass
class SensorSpringBendingSnapRate:
    """``/SENSOR/SPRING_BENDING_SNAP_RATE`` or ``/SENSOR/SPRING_BEND_SNAP_RATE`` (M434): Spring element relative transverse bending angular acceleration 20th rate-of-change (bending angular snap rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0             # spring element ID to monitor
    jbnd_snp_max: float = 1e30     # maximum bending snap rate threshold
    t_delay: float = 0.0           # activation delay time
    jbend_snp_max: float = 1e30    # alias field for compatibility
    jbending_snp_max: float = 1e30 # alias field for compatibility

    def __post_init__(self):
        if self.jbend_snp_max != 1e30 and self.jbnd_snp_max == 1e30:
            self.jbnd_snp_max = self.jbend_snp_max
        elif self.jbnd_snp_max != 1e30 and self.jbend_snp_max == 1e30:
            self.jbend_snp_max = self.jbnd_snp_max
        if self.jbending_snp_max != 1e30 and self.jbnd_snp_max == 1e30:
            self.jbnd_snp_max = self.jbending_snp_max
        elif self.jbnd_snp_max != 1e30 and self.jbending_snp_max == 1e30:
            self.jbending_snp_max = self.jbnd_snp_max

    @property
    def jbnd_lock_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_lock_max.setter
    def jbnd_lock_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_crackle_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_crackle_max.setter
    def jbnd_crackle_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_shot_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_shot_max.setter
    def jbnd_shot_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_pop_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_pop_max.setter
    def jbnd_pop_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_crk_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_crk_max.setter
    def jbnd_crk_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_drop_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_drop_rate_max.setter
    def jbnd_drop_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_rate_max.setter
    def jbnd_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_roc_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_roc_rate_max.setter
    def jbnd_roc_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_snap_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_snap_rate_max.setter
    def jbnd_snap_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def j_max(self) -> float:
        return self.jbnd_snp_max

    @j_max.setter
    def j_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_shot_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_shot_max.setter
    def jbend_shot_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_lock_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_lock_max.setter
    def jbend_lock_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_pop_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_pop_max.setter
    def jbend_pop_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_crackle_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_crackle_max.setter
    def jbend_crackle_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_crk_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_crk_max.setter
    def jbend_crk_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_drop_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_drop_rate_max.setter
    def jbend_drop_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_rate_max.setter
    def jbend_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_roc_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_roc_rate_max.setter
    def jbend_roc_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_snap_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_snap_rate_max.setter
    def jbend_snap_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_snp_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_snp_rate_max.setter
    def jbend_snp_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_shot_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_shot_max.setter
    def jbending_shot_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_lock_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_lock_max.setter
    def jbending_lock_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_pop_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_pop_max.setter
    def jbending_pop_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_crackle_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_crackle_max.setter
    def jbending_crackle_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_crk_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_crk_max.setter
    def jbending_crk_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_drop_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_drop_rate_max.setter
    def jbending_drop_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_rate_max.setter
    def jbending_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_roc_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_roc_rate_max.setter
    def jbending_roc_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_snap_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_snap_rate_max.setter
    def jbending_snap_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_snp_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_snp_rate_max.setter
    def jbending_snp_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_drop_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_drop_max.setter
    def jbend_drop_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_drop_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_drop_max.setter
    def jbending_drop_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_drop_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_drop_max.setter
    def jbnd_drop_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_snap_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_snap_max.setter
    def jbnd_snap_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbend_snap_max(self) -> float:
        return self.jbnd_snp_max

    @jbend_snap_max.setter
    def jbend_snap_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbending_snap_max(self) -> float:
        return self.jbnd_snp_max

    @jbending_snap_max.setter
    def jbending_snap_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def j_bnd_snp_max(self) -> float:
        return self.jbnd_snp_max

    @j_bnd_snp_max.setter
    def j_bnd_snp_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def j_bend_snp_max(self) -> float:
        return self.jbnd_snp_max

    @j_bend_snp_max.setter
    def j_bend_snp_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def j_bending_snp_max(self) -> float:
        return self.jbnd_snp_max

    @j_bending_snp_max.setter
    def j_bending_snp_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def j_snap_bend_max(self) -> float:
        return self.jbnd_snp_max

    @j_snap_bend_max.setter
    def j_snap_bend_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def j_snap_bending_max(self) -> float:
        return self.jbnd_snp_max

    @j_snap_bending_max.setter
    def j_snap_bending_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def j_bending_snap_max(self) -> float:
        return self.jbnd_snp_max

    @j_bending_snap_max.setter
    def j_bending_snap_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_crk_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_crk_rate_max.setter
    def jbnd_crk_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_pop_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_pop_rate_max.setter
    def jbnd_pop_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val

    @property
    def jbnd_lock_rate_max(self) -> float:
        return self.jbnd_snp_max

    @jbnd_lock_rate_max.setter
    def jbnd_lock_rate_max(self, val: float) -> None:
        self.jbnd_snp_max = val
        self.jbend_snp_max = val
        self.jbending_snp_max = val


@dataclass
class SensorSpringCoupledCrackleRate:
    """``/SENSOR/SPRING_COUPLED_CRACKLE_RATE`` or ``/SENSOR/SPRING_COUPLE_CRACKLE_RATE`` (M451): Spring element relative coupled / mixed-mode acceleration 21st rate-of-change (coupled crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0
    jcoup_crk_max: float = 1e30
    t_delay: float = 0.0
    kind: str = "SPRING_COUPLED_CRACKLE_RATE"

    @property
    def j_crk_coup_max(self) -> float:
        return self.jcoup_crk_max

    @j_crk_coup_max.setter
    def j_crk_coup_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def j_crackle_coup_max(self) -> float:
        return self.jcoup_crk_max

    @j_crackle_coup_max.setter
    def j_crackle_coup_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def j_coup_crk_max(self) -> float:
        return self.jcoup_crk_max

    @j_coup_crk_max.setter
    def j_coup_crk_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def j_crk_max(self) -> float:
        return self.jcoup_crk_max

    @j_crk_max.setter
    def j_crk_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_crackle_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_crackle_max.setter
    def jcoup_crackle_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def j_crk_coupled_max(self) -> float:
        return self.jcoup_crk_max

    @j_crk_coupled_max.setter
    def j_crk_coupled_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def j_crackle_coupled_max(self) -> float:
        return self.jcoup_crk_max

    @j_crackle_coupled_max.setter
    def j_crackle_coupled_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_shot_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_shot_max.setter
    def jcoup_shot_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_drop_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_drop_max.setter
    def jcoup_drop_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_lock_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_lock_max.setter
    def jcoup_lock_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_pop_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_pop_max.setter
    def jcoup_pop_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_snp_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_snp_max.setter
    def jcoup_snp_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_snap_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_snap_max.setter
    def jcoup_snap_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_rate_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_rate_max.setter
    def jcoup_rate_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_roc_rate_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_roc_rate_max.setter
    def jcoup_roc_rate_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_drop_rate_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_drop_rate_max.setter
    def jcoup_drop_rate_max(self, val: float) -> None:
        self.jcoup_crk_max = val

    @property
    def jcoup_crk_rate_max(self) -> float:
        return self.jcoup_crk_max

    @jcoup_crk_rate_max.setter
    def jcoup_crk_rate_max(self, val: float) -> None:
        self.jcoup_crk_max = val


SensorSpringCoupleCrackleRate = SensorSpringCoupledCrackleRate
SensorSpringCoupledCrkRate = SensorSpringCoupledCrackleRate
SensorSpringCoupleCrkRate = SensorSpringCoupledCrackleRate


@dataclass
class SensorSpringTorsionalCrackleRate:
    """``/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TORS_CRACKLE_RATE`` (M452): Spring element relative torsional acceleration 21st rate-of-change (torsional crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0
    jtors_crk_max: float = 1e30
    t_delay: float = 0.0
    kind: str = "SPRING_TORSIONAL_CRACKLE_RATE"

    @property
    def j_crk_tors_max(self) -> float:
        return self.jtors_crk_max

    @j_crk_tors_max.setter
    def j_crk_tors_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_crackle_tors_max(self) -> float:
        return self.jtors_crk_max

    @j_crackle_tors_max.setter
    def j_crackle_tors_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_tors_crk_max(self) -> float:
        return self.jtors_crk_max

    @j_tors_crk_max.setter
    def j_tors_crk_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_crk_max(self) -> float:
        return self.jtors_crk_max

    @j_crk_max.setter
    def j_crk_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtors_crk_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_crk_torsional_max(self) -> float:
        return self.jtors_crk_max

    @j_crk_torsional_max.setter
    def j_crk_torsional_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_crackle_torsional_max(self) -> float:
        return self.jtors_crk_max

    @j_crackle_torsional_max.setter
    def j_crackle_torsional_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_shot_max(self) -> float:
        return self.jtors_crk_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_drop_max(self) -> float:
        return self.jtors_crk_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_lock_max(self) -> float:
        return self.jtors_crk_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtors_crk_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtors_crk_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_snap_max(self) -> float:
        return self.jtors_crk_max

    @jtors_snap_max.setter
    def jtors_snap_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_rate_max(self) -> float:
        return self.jtors_crk_max

    @jtors_rate_max.setter
    def jtors_rate_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_roc_rate_max(self) -> float:
        return self.jtors_crk_max

    @jtors_roc_rate_max.setter
    def jtors_roc_rate_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_drop_rate_max(self) -> float:
        return self.jtors_crk_max

    @jtors_drop_rate_max.setter
    def jtors_drop_rate_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_crk_rate_max(self) -> float:
        return self.jtors_crk_max

    @jtors_crk_rate_max.setter
    def jtors_crk_rate_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_tors_crackle_max(self) -> float:
        return self.jtors_crk_max

    @j_tors_crackle_max.setter
    def j_tors_crackle_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_torsional_crk_max(self) -> float:
        return self.jtors_crk_max

    @j_torsional_crk_max.setter
    def j_torsional_crk_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_torsional_crackle_max(self) -> float:
        return self.jtors_crk_max

    @j_torsional_crackle_max.setter
    def j_torsional_crackle_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_twist_crk_max(self) -> float:
        return self.jtors_crk_max

    @j_twist_crk_max.setter
    def j_twist_crk_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_twist_crackle_max(self) -> float:
        return self.jtors_crk_max

    @j_twist_crackle_max.setter
    def j_twist_crackle_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_twisting_crk_max(self) -> float:
        return self.jtors_crk_max

    @j_twisting_crk_max.setter
    def j_twisting_crk_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_twisting_crackle_max(self) -> float:
        return self.jtors_crk_max

    @j_twisting_crackle_max.setter
    def j_twisting_crackle_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_crk_twist_max(self) -> float:
        return self.jtors_crk_max

    @j_crk_twist_max.setter
    def j_crk_twist_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def j_crackle_twist_max(self) -> float:
        return self.jtors_crk_max

    @j_crackle_twist_max.setter
    def j_crackle_twist_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_crackle_rate_max(self) -> float:
        return self.jtors_crk_max

    @jtors_crackle_rate_max.setter
    def jtors_crackle_rate_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_snap_rate_max(self) -> float:
        return self.jtors_crk_max

    @jtors_snap_rate_max.setter
    def jtors_snap_rate_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_pop_rate_max(self) -> float:
        return self.jtors_crk_max

    @jtors_pop_rate_max.setter
    def jtors_pop_rate_max(self, val: float) -> None:
        self.jtors_crk_max = val

    @property
    def jtors_lock_rate_max(self) -> float:
        return self.jtors_crk_max

    @jtors_lock_rate_max.setter
    def jtors_lock_rate_max(self, val: float) -> None:
        self.jtors_crk_max = val



SensorSpringTorsionCrackleRate = SensorSpringTorsionalCrackleRate
SensorSpringTorsionalCrkRate = SensorSpringTorsionalCrackleRate
SensorSpringTwistCrackleRate = SensorSpringTorsionalCrackleRate


@dataclass
class SensorSpringTotalCrackleRate:
    """``/SENSOR/SPRING_TOTAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_CRACKLE_RATE`` (M453): Spring element relative 3D resultant total linear acceleration 21st rate-of-change (resultant total crackle rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0
    jtot_crk_max: float = 1e30
    t_delay: float = 0.0
    kind: str = "SPRING_TOTAL_CRACKLE_RATE"

    @property
    def j_crk_tot_max(self) -> float:
        return self.jtot_crk_max

    @j_crk_tot_max.setter
    def j_crk_tot_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def j_crackle_tot_max(self) -> float:
        return self.jtot_crk_max

    @j_crackle_tot_max.setter
    def j_crackle_tot_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def j_tot_crk_max(self) -> float:
        return self.jtot_crk_max

    @j_tot_crk_max.setter
    def j_tot_crk_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def j_crk_max(self) -> float:
        return self.jtot_crk_max

    @j_crk_max.setter
    def j_crk_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_crackle_max(self) -> float:
        return self.jtot_crk_max

    @jtot_crackle_max.setter
    def jtot_crackle_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def j_crk_total_max(self) -> float:
        return self.jtot_crk_max

    @j_crk_total_max.setter
    def j_crk_total_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def j_crackle_total_max(self) -> float:
        return self.jtot_crk_max

    @j_crackle_total_max.setter
    def j_crackle_total_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_shot_max(self) -> float:
        return self.jtot_crk_max

    @jtot_shot_max.setter
    def jtot_shot_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_drop_max(self) -> float:
        return self.jtot_crk_max

    @jtot_drop_max.setter
    def jtot_drop_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_lock_max(self) -> float:
        return self.jtot_crk_max

    @jtot_lock_max.setter
    def jtot_lock_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_pop_max(self) -> float:
        return self.jtot_crk_max

    @jtot_pop_max.setter
    def jtot_pop_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_snp_max(self) -> float:
        return self.jtot_crk_max

    @jtot_snp_max.setter
    def jtot_snp_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_snap_max(self) -> float:
        return self.jtot_crk_max

    @jtot_snap_max.setter
    def jtot_snap_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_rate_max(self) -> float:
        return self.jtot_crk_max

    @jtot_rate_max.setter
    def jtot_rate_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_roc_rate_max(self) -> float:
        return self.jtot_crk_max

    @jtot_roc_rate_max.setter
    def jtot_roc_rate_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_drop_rate_max(self) -> float:
        return self.jtot_crk_max

    @jtot_drop_rate_max.setter
    def jtot_drop_rate_max(self, val: float) -> None:
        self.jtot_crk_max = val

    @property
    def jtot_crk_rate_max(self) -> float:
        return self.jtot_crk_max

    @jtot_crk_rate_max.setter
    def jtot_crk_rate_max(self, val: float) -> None:
        self.jtot_crk_max = val


SensorSpringTotCrackleRate = SensorSpringTotalCrackleRate
SensorSpringTotalCrkRate = SensorSpringTotalCrackleRate
SensorSpringTotCrkRate = SensorSpringTotalCrackleRate
SensorSpringResultantCrackleRate = SensorSpringTotalCrackleRate
SensorSpringResultantCrkRate = SensorSpringTotalCrackleRate
SensorSpringResCrackleRate = SensorSpringTotalCrackleRate
SensorSpringResCrkRate = SensorSpringTotalCrackleRate


@dataclass
class SensorSpringNormalPopRate:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M454): Spring element relative normal / axial acceleration 22nd rate-of-change (axial pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0
    jnorm_pop_max: float = 1e30
    t_delay: float = 0.0
    kind: str = "SPRING_NORMAL_POP_RATE"

    @property
    def j_pop_norm_max(self) -> float:
        return self.jnorm_pop_max

    @j_pop_norm_max.setter
    def j_pop_norm_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def j_pop_normal_max(self) -> float:
        return self.jnorm_pop_max

    @j_pop_normal_max.setter
    def j_pop_normal_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def j_norm_pop_max(self) -> float:
        return self.jnorm_pop_max

    @j_norm_pop_max.setter
    def j_norm_pop_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def j_pop_max(self) -> float:
        return self.jnorm_pop_max

    @j_pop_max.setter
    def j_pop_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def j_pop_axial_max(self) -> float:
        return self.jnorm_pop_max

    @j_pop_axial_max.setter
    def j_pop_axial_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_shot_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_shot_max.setter
    def jnorm_shot_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_drop_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_drop_max.setter
    def jnorm_drop_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_lock_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_lock_max.setter
    def jnorm_lock_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_snp_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_snp_max.setter
    def jnorm_snp_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_snap_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_snap_max.setter
    def jnorm_snap_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_rate_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_rate_max.setter
    def jnorm_rate_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_roc_rate_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_roc_rate_max.setter
    def jnorm_roc_rate_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_drop_rate_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_drop_rate_max.setter
    def jnorm_drop_rate_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_crk_rate_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_crk_rate_max.setter
    def jnorm_crk_rate_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_crackle_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_crackle_max.setter
    def jnorm_crackle_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_crk_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_crk_max.setter
    def jnorm_crk_max(self, val: float) -> None:
        self.jnorm_pop_max = val

    @property
    def jnorm_pop_rate_max(self) -> float:
        return self.jnorm_pop_max

    @jnorm_pop_rate_max.setter
    def jnorm_pop_rate_max(self, val: float) -> None:
        self.jnorm_pop_max = val


SensorSpringNormPopRate = SensorSpringNormalPopRate
SensorSpringNormalPop = SensorSpringNormalPopRate
SensorSpringNormPop = SensorSpringNormalPopRate
SensorSpringAxialPopRate = SensorSpringNormalPopRate
SensorSpringAxialPop = SensorSpringNormalPopRate


@dataclass
class SensorSpringTransversePopRate:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M455): Spring element relative transverse / shear acceleration 22nd rate-of-change (shear pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0
    jtrans_pop_max: float = 1e30
    t_delay: float = 0.0
    kind: str = "SPRING_TRANSVERSE_POP_RATE"

    @property
    def j_pop_trans_max(self) -> float:
        return self.jtrans_pop_max

    @j_pop_trans_max.setter
    def j_pop_trans_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def j_pop_transverse_max(self) -> float:
        return self.jtrans_pop_max

    @j_pop_transverse_max.setter
    def j_pop_transverse_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def j_trans_pop_max(self) -> float:
        return self.jtrans_pop_max

    @j_trans_pop_max.setter
    def j_trans_pop_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def j_pop_max(self) -> float:
        return self.jtrans_pop_max

    @j_pop_max.setter
    def j_pop_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def j_pop_shear_max(self) -> float:
        return self.jtrans_pop_max

    @j_pop_shear_max.setter
    def j_pop_shear_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_shot_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_shot_max.setter
    def jtrans_shot_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_drop_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_drop_max.setter
    def jtrans_drop_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_lock_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_lock_max.setter
    def jtrans_lock_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_snp_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_snp_max.setter
    def jtrans_snp_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_snap_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_snap_max.setter
    def jtrans_snap_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_rate_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_rate_max.setter
    def jtrans_rate_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_roc_rate_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_roc_rate_max.setter
    def jtrans_roc_rate_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_drop_rate_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_drop_rate_max.setter
    def jtrans_drop_rate_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_crk_rate_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_crk_rate_max.setter
    def jtrans_crk_rate_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_crackle_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_crackle_max.setter
    def jtrans_crackle_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_crk_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_crk_max.setter
    def jtrans_crk_max(self, val: float) -> None:
        self.jtrans_pop_max = val

    @property
    def jtrans_pop_rate_max(self) -> float:
        return self.jtrans_pop_max

    @jtrans_pop_rate_max.setter
    def jtrans_pop_rate_max(self, val: float) -> None:
        self.jtrans_pop_max = val


SensorSpringTransPopRate = SensorSpringTransversePopRate
SensorSpringTransversePop = SensorSpringTransversePopRate
SensorSpringTransPop = SensorSpringTransversePopRate
SensorSpringShearPopRate = SensorSpringTransversePopRate
SensorSpringShearPop = SensorSpringTransversePopRate


@dataclass
class SensorSpringCoupledPopRate:
    """``/SENSOR/SPRING_COUPLED_POP_RATE`` or ``/SENSOR/SPRING_COUPLE_POP_RATE`` (M456): Spring element relative multi-axial coupled acceleration 22nd rate-of-change (coupled pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0
    jcoup_pop_max: float = 1e30
    t_delay: float = 0.0
    kind: str = "SPRING_COUPLED_POP_RATE"

    @property
    def j_pop_coup_max(self) -> float:
        return self.jcoup_pop_max

    @j_pop_coup_max.setter
    def j_pop_coup_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def j_pop_coupled_max(self) -> float:
        return self.jcoup_pop_max

    @j_pop_coupled_max.setter
    def j_pop_coupled_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def j_coup_pop_max(self) -> float:
        return self.jcoup_pop_max

    @j_coup_pop_max.setter
    def j_coup_pop_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def j_pop_max(self) -> float:
        return self.jcoup_pop_max

    @j_pop_max.setter
    def j_pop_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def j_pop_biaxial_max(self) -> float:
        return self.jcoup_pop_max

    @j_pop_biaxial_max.setter
    def j_pop_biaxial_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_shot_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_shot_max.setter
    def jcoup_shot_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_drop_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_drop_max.setter
    def jcoup_drop_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_lock_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_lock_max.setter
    def jcoup_lock_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_snp_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_snp_max.setter
    def jcoup_snp_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_snap_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_snap_max.setter
    def jcoup_snap_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_rate_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_rate_max.setter
    def jcoup_rate_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_roc_rate_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_roc_rate_max.setter
    def jcoup_roc_rate_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_drop_rate_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_drop_rate_max.setter
    def jcoup_drop_rate_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_crk_rate_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_crk_rate_max.setter
    def jcoup_crk_rate_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_crackle_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_crackle_max.setter
    def jcoup_crackle_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_crk_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_crk_max.setter
    def jcoup_crk_max(self, val: float) -> None:
        self.jcoup_pop_max = val

    @property
    def jcoup_pop_rate_max(self) -> float:
        return self.jcoup_pop_max

    @jcoup_pop_rate_max.setter
    def jcoup_pop_rate_max(self, val: float) -> None:
        self.jcoup_pop_max = val


SensorSpringCouplePopRate = SensorSpringCoupledPopRate
SensorSpringCoupledPop = SensorSpringCoupledPopRate
SensorSpringCouplePop = SensorSpringCoupledPopRate
SensorSpringBiaxialPopRate = SensorSpringCoupledPopRate
SensorSpringBiaxialPop = SensorSpringCoupledPopRate


@dataclass
class SensorSpringTorsionalPopRate:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORSION_POP_RATE`` (M457): Spring element relative torsional acceleration 22nd rate-of-change (torsional pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0
    jtor_pop_max: float = 1e30
    t_delay: float = 0.0
    kind: str = "SPRING_TORSIONAL_POP_RATE"

    @property
    def j_pop_tor_max(self) -> float:
        return self.jtor_pop_max

    @j_pop_tor_max.setter
    def j_pop_tor_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def j_pop_torsional_max(self) -> float:
        return self.jtor_pop_max

    @j_pop_torsional_max.setter
    def j_pop_torsional_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def j_tor_pop_max(self) -> float:
        return self.jtor_pop_max

    @j_tor_pop_max.setter
    def j_tor_pop_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def j_pop_max(self) -> float:
        return self.jtor_pop_max

    @j_pop_max.setter
    def j_pop_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def j_pop_twist_max(self) -> float:
        return self.jtor_pop_max

    @j_pop_twist_max.setter
    def j_pop_twist_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_shot_max(self) -> float:
        return self.jtor_pop_max

    @jtor_shot_max.setter
    def jtor_shot_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_drop_max(self) -> float:
        return self.jtor_pop_max

    @jtor_drop_max.setter
    def jtor_drop_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_lock_max(self) -> float:
        return self.jtor_pop_max

    @jtor_lock_max.setter
    def jtor_lock_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_snp_max(self) -> float:
        return self.jtor_pop_max

    @jtor_snp_max.setter
    def jtor_snp_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_snap_max(self) -> float:
        return self.jtor_pop_max

    @jtor_snap_max.setter
    def jtor_snap_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtor_rate_max.setter
    def jtor_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_roc_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtor_roc_rate_max.setter
    def jtor_roc_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_drop_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtor_drop_rate_max.setter
    def jtor_drop_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_crk_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtor_crk_rate_max.setter
    def jtor_crk_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_crackle_max(self) -> float:
        return self.jtor_pop_max

    @jtor_crackle_max.setter
    def jtor_crackle_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_crk_max(self) -> float:
        return self.jtor_pop_max

    @jtor_crk_max.setter
    def jtor_crk_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtor_pop_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtor_pop_rate_max.setter
    def jtor_pop_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_pop_max(self) -> float:
        return self.jtor_pop_max

    @jtors_pop_max.setter
    def jtors_pop_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def j_tors_pop_max(self) -> float:
        return self.jtor_pop_max

    @j_tors_pop_max.setter
    def j_tors_pop_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def j_pop_tors_max(self) -> float:
        return self.jtor_pop_max

    @j_pop_tors_max.setter
    def j_pop_tors_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_shot_max(self) -> float:
        return self.jtor_pop_max

    @jtors_shot_max.setter
    def jtors_shot_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_drop_max(self) -> float:
        return self.jtor_pop_max

    @jtors_drop_max.setter
    def jtors_drop_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_lock_max(self) -> float:
        return self.jtor_pop_max

    @jtors_lock_max.setter
    def jtors_lock_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_snp_max(self) -> float:
        return self.jtor_pop_max

    @jtors_snp_max.setter
    def jtors_snp_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_snap_max(self) -> float:
        return self.jtor_pop_max

    @jtors_snap_max.setter
    def jtors_snap_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtors_rate_max.setter
    def jtors_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_roc_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtors_roc_rate_max.setter
    def jtors_roc_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_drop_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtors_drop_rate_max.setter
    def jtors_drop_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_crk_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtors_crk_rate_max.setter
    def jtors_crk_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_crackle_max(self) -> float:
        return self.jtor_pop_max

    @jtors_crackle_max.setter
    def jtors_crackle_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_crk_max(self) -> float:
        return self.jtor_pop_max

    @jtors_crk_max.setter
    def jtors_crk_max(self, val: float) -> None:
        self.jtor_pop_max = val

    @property
    def jtors_pop_rate_max(self) -> float:
        return self.jtor_pop_max

    @jtors_pop_rate_max.setter
    def jtors_pop_rate_max(self, val: float) -> None:
        self.jtor_pop_max = val


SensorSpringTorsionPopRate = SensorSpringTorsionalPopRate
SensorSpringTorsionalPop = SensorSpringTorsionalPopRate
SensorSpringTorsionPop = SensorSpringTorsionalPopRate
SensorSpringTwistPopRate = SensorSpringTorsionalPopRate
SensorSpringTwistPop = SensorSpringTorsionalPopRate


@dataclass
class SensorSpringTotalAngularPopRate:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M458): Spring element relative resultant total angular acceleration 22nd rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    id: int = 1
    title: str = ""
    spring_id: int = 0
    jtot_ang_pop_max: float = 1e30
    t_delay: float = 0.0
    kind: str = "SPRING_TOTAL_ANGULAR_POP_RATE"

    @property
    def j_pop_tot_ang_max(self) -> float:
        return self.jtot_ang_pop_max

    @j_pop_tot_ang_max.setter
    def j_pop_tot_ang_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def j_pop_total_angular_max(self) -> float:
        return self.jtot_ang_pop_max

    @j_pop_total_angular_max.setter
    def j_pop_total_angular_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def j_tot_ang_pop_max(self) -> float:
        return self.jtot_ang_pop_max

    @j_tot_ang_pop_max.setter
    def j_tot_ang_pop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def j_pop_max(self) -> float:
        return self.jtot_ang_pop_max

    @j_pop_max.setter
    def j_pop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def j_pop_res_ang_max(self) -> float:
        return self.jtot_ang_pop_max

    @j_pop_res_ang_max.setter
    def j_pop_res_ang_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_shot_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_shot_max.setter
    def jtot_ang_shot_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_drop_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_drop_max.setter
    def jtot_ang_drop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_lock_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_lock_max.setter
    def jtot_ang_lock_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_snp_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_snp_max.setter
    def jtot_ang_snp_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_snap_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_snap_max.setter
    def jtot_ang_snap_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_rate_max.setter
    def jtot_ang_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_roc_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_roc_rate_max.setter
    def jtot_ang_roc_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_drop_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_drop_rate_max.setter
    def jtot_ang_drop_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_crk_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_crk_rate_max.setter
    def jtot_ang_crk_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_crackle_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_crackle_max.setter
    def jtot_ang_crackle_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_crk_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_crk_max.setter
    def jtot_ang_crk_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtot_ang_pop_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtot_ang_pop_rate_max.setter
    def jtot_ang_pop_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_pop_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_pop_max.setter
    def jtang_pop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def j_tang_pop_max(self) -> float:
        return self.jtot_ang_pop_max

    @j_tang_pop_max.setter
    def j_tang_pop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def j_pop_tang_max(self) -> float:
        return self.jtot_ang_pop_max

    @j_pop_tang_max.setter
    def j_pop_tang_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_shot_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_shot_max.setter
    def jtang_shot_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_drop_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_drop_max.setter
    def jtang_drop_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_lock_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_lock_max.setter
    def jtang_lock_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_snp_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_snp_max.setter
    def jtang_snp_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_snap_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_snap_max.setter
    def jtang_snap_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_rate_max.setter
    def jtang_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_roc_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_roc_rate_max.setter
    def jtang_roc_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_drop_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_drop_rate_max.setter
    def jtang_drop_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_crk_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_crk_rate_max.setter
    def jtang_crk_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_crackle_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_crackle_max.setter
    def jtang_crackle_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_crk_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_crk_max.setter
    def jtang_crk_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val

    @property
    def jtang_pop_rate_max(self) -> float:
        return self.jtot_ang_pop_max

    @jtang_pop_rate_max.setter
    def jtang_pop_rate_max(self, val: float) -> None:
        self.jtot_ang_pop_max = val


SensorSpringTotAngPopRate = SensorSpringTotalAngularPopRate
SensorSpringTotalAngularPop = SensorSpringTotalAngularPopRate
SensorSpringTotAngPop = SensorSpringTotalAngularPopRate
SensorSpringResultantAngularPopRate = SensorSpringTotalAngularPopRate
SensorSpringResultantAngularPop = SensorSpringTotalAngularPopRate
