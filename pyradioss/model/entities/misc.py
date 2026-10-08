"""Engine directives, output requests and tabular data.

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
class Section:
    """/SECT (M5): section-force output — the time history of the resultant
    force/moment transmitted through a cut of the mesh.

    Fortran origin: ``engine/source/tools/sect/`` (section.F, forint.F):
    the original accumulates the internal forces of the elements of one
    side at the section nodes. The port uses the equivalent *side-sum*
    identity, which needs only a node set: since the internal force vector
    of any element in equilibrium sums to zero over its own nodes (and its
    moments balance), summing the assembled internal forces over ALL nodes
    of one side leaves exactly the force the OTHER side's elements exert
    through the cut:

        F_sect = sum_{n in side} fint_n
        M_sect = sum_{n in side} [(x_n - x_ref) x fint_n + mint_n]

    ``grnod_id`` must therefore contain every node of one side of the cut,
    including the cut nodes themselves (a /GRNOD/PART of the side parts is
    the natural way to write it). The sign convention: the reported force
    is the force the excluded side applies to the included side.

    ``node_id_ref`` (optional): moment reference point = that node's
    current position (it rides the deformation); 0 = the fixed initial
    centroid of the side node set. Output goes to the T01 file via
    /TH/SECT (FX FY FZ MX MY MZ).
    """

    id: int
    grnod_id: int
    node_id_ref: int = 0
    title: str = ""


@dataclass
class SectBox:
    """/SECT/BOX (M136): Section cutting defined by bounding box."""
    id: int
    title: str = ""
    box_id: int = 0
    grnod_id: int = 0
    frame_id: int = 0


@dataclass
class SectCut:
    """/SECT/CUT (M136): Section defined by cutting plane."""
    id: int
    title: str = ""
    orig: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    grnod_id: int = 0
    frame_id: int = 0


@dataclass
class THRequest:
    """/TH/NODE, /TH/PART, /TH/SECT and element/entity variants (M68):
    time-history output request."""

    id: int
    kind: str            # 'NODE' | 'PART' | 'SECT' | 'RBODY' | 'SHEL' |
                         # 'SH3N' | 'SPRING' | 'BRIC' | 'RWALL' | 'INTER'
    ids: List[Union[int, str]] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)  # e.g. DX, VX, IE
    title: str = ""

    @property
    def obj_ids(self) -> List[Union[int, str]]:
        return self.ids


@dataclass
class Table:
    """``/TABLE/dim/table_ID`` (1D, 2D, 3D tabular functions; M139)."""
    id: int
    dim: int
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z: Optional[np.ndarray] = None
    curves: List[Tuple[float, np.ndarray, np.ndarray]] = field(default_factory=list)
    title: str = ""


@dataclass
class Random:
    """``/RANDOM/random_ID`` (Stochastic / random fields)."""
    id: int
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class Xref:
    """`/XREF/part_id` reference geometry (hm_read_xref.F).

    Stores the initial reference configuration (undeformed coordinates)
    for elements of a given part — used for springback, pre-straining,
    or metric tensor initialization.
    """
    part_id: int
    title: str = ""
    nitrs: int = 100          # steps from reference to initial state
    node_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    dtype: Any = None


@dataclass
class DetonatorPoint:
    """`/DFS/DETPOINT/det_id` — Point-source detonation (read_dfs_detpoint.F).

    Ignites explosive elements from a point source; lighting time for each
    element is ``tdet + dist / D_det`` where D_det is the material's
    detonation velocity.
    """
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    grnod_id: int = 0
    node_id: int = 0


@dataclass
class DetonatorPlane:
    """`/DFS/DETPLAN/det_id` — Planar detonation front (read_dfs_detplan.F).

    Ignites explosive elements from a planar wave; the plane passes
    through point (x, y, z) with propagation normal (nx, ny, nz).
    """
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    nx: float = 0.0
    ny: float = 0.0
    nz: float = 0.0
    p_id: int = 0
    n_id: int = 0


@dataclass
class InitialTemperature:
    """/INITEMP (M95): initial nodal temperature.

    Fortran origin: ``starter/source/initial_conditions/thermic/hm_read_initemp.F``.
    """
    id: int
    t0: float = 0.0
    grnod_id: int = 0
    fld_type: int = 0       # 0 = uniform on group, 1 = nodal table
    nodal_temps: Dict[int, float] = field(default_factory=dict)  # node_id -> temp
    title: str = ""
    part_id: Optional[int] = None
    element_set_id: Optional[int] = None
    node_ids: Optional[Sequence[int]] = None
    gradient: Optional[Tuple[float, float, float]] = None
    x0: Optional[Tuple[float, float, float]] = None
    t_top: Optional[float] = None
    t_mid: Optional[float] = None
    t_bot: Optional[float] = None
    layer_temperatures: Optional[Sequence[float]] = None
    additive: bool = False


InitempRecordEntity = InitialTemperature
InitempParamsEntity = InitialTemperature


@dataclass
class InitialBrickState:
    """/INIBRI (M96, M138, M142): initial state for solid/brick elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F``.
    """
    elem_id: int
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [sxx, syy, szz, sxy, syz, sxz]
    eps: np.ndarray = field(default_factory=lambda: np.zeros(6))    # [exx, eyy, ezz, exy, eyz, exz] (M142)
    epsp: float = 0.0      # plastic strain
    rho: float = 0.0       # initial density
    ener: float = 0.0      # internal energy
    temp: float = 0.0      # initial temperature (M138)
    pres: float = 0.0      # initial hydrostatic pressure (M138)
    void: float = 0.0      # initial void fraction (M138)
    fail_flag: float = 0.0 # initial failure flag (M142)
    aux: float = 0.0       # auxiliary state variable (M142)
    scale_yld: float = 1.0 # yield stress scale factor (M142)


@dataclass
class InitialShellState:
    """/INISHE and /INISH3 (M96, M138, M142): initial state for shell elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F``.
    """
    elem_id: int
    thick: float = 0.0     # initial thickness override
    epsp: float = 0.0      # plastic strain
    epsp_layers: List[float] = field(default_factory=list) # per-layer plastic strain (M142)
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(6))  # membrane stress
    sigma_b: np.ndarray = field(default_factory=lambda: np.zeros(6)) # bending stress
    eps: np.ndarray = field(default_factory=lambda: np.zeros(6))    # strain tensor (M142)
    em: float = 0.0        # membrane energy
    eb: float = 0.0        # bending energy
    h_energy: np.ndarray = field(default_factory=lambda: np.zeros(3)) # H1, H2, H3
    temp: float = 0.0      # initial temperature (M138)
    rho: float = 0.0       # initial density (M178)
    fail_flag: float = 0.0 # initial failure flag (M142)
    aux: float = 0.0       # auxiliary state variable (M142)
    scale_yld: float = 1.0 # yield stress scale factor (M142)
    orth_angles: List[Tuple[float, float]] = field(default_factory=list) # orthotropy angles per layer (M199)
    orth_phi: List[float] = field(default_factory=list) # phi_i angles per layer (M199)
    orth_alpha: List[float] = field(default_factory=list) # alpha_i angles per layer (M199)



@dataclass
class InitialTrussState:
    """/INITRU (M97, M138): initial state for truss elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F`` and
    ``starter/source/elements/truss/tsigini.F``.
    """
    elem_id: int
    prop_type: int = 2
    eint: float = 0.0      # initial internal energy
    force: float = 0.0     # initial axial force / tension
    area: float = 0.0      # initial area override
    epsp: float = 0.0      # plastic strain
    temp: float = 0.0      # initial temperature (M138)


@dataclass
class InitialBeamState:
    """/INIBEA (M97, M138): initial state for beam elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F`` and
    ``starter/source/elements/beam/bsigini.F``.
    """
    elem_id: int
    prop_type: int = 3
    nb_integr: int = 0
    eint_memb: float = 0.0 # membrane internal energy
    eint_bend: float = 0.0 # bending internal energy
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [Fx, Fy, Fz]
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [Mx, My, Mz]
    epsp: float = 0.0      # plastic strain
    temp: float = 0.0      # initial temperature (M138)


@dataclass
class InitialSpringState:
    """/INISPR (M97, M138): initial state for spring elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F`` and
    ``starter/source/elements/spring/rinit3.F``.
    """
    elem_id: int
    prop_type: int = 4
    force: float = 0.0     # initial force
    disp: float = 0.0      # initial displacement
    fep: float = 0.0       # elasto-plastic limit force
    dpl_pos: float = 0.0   # positive plastic displacement
    dpl_neg: float = 0.0   # negative plastic displacement
    length: float = 0.0    # initial length
    eint: float = 0.0      # internal energy
    temp: float = 0.0      # initial temperature (M138)


@dataclass
class CyclicBoundaryCondition:
    """/BCS/CYCLIC (M99): cyclic boundary condition linking two node groups.

    Fortran origin: ``starter/source/model/bcs/hm_read_bcscyclic.F`` and
    ``engine/source/assembly/bcs/``.
    """
    id: int
    title: str = ""
    skew_id: int = 0
    grnod1_id: int = 0
    grnod2_id: int = 0


@dataclass
class DetonationWave:
    """/INIT/DET_POINT, /INIT/DET_LINE, /INIT/DET_PLAN, /INIT/DET_CORD (M110):
    High-explosive detonation wavefront initialization.
    
    Fortran origin: ``starter/source/initial_conditions/detonation/``.
    """
    id: int
    kind: str = "POINT"  # 'POINT' | 'LINE' | 'PLAN' | 'CORD'
    title: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    z2: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    ddet: float = 0.0
    iopt: int = 0


@dataclass
class WaveShaper:
    """/DFS/WAVE_SHAPER, /INIT/DET/WAVE_SHAPER (M132): Explosive detonation wave shaper barrier."""
    id: int
    title: str = ""
    surf_id: int = 0
    mat_id: int = 0
    thick: float = 0.0
    delay: float = 0.0


@dataclass
class SphGlobal:
    """/SPHGLO (M101): SPH global computation controls.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_sphglo.F``.
    """
    spasort: float = 0.25
    ale_maxsph: int = 0
    lvoisph: int = 120
    kvoisph: int = 240
    isol2sph: int = 1


@dataclass
class AnalyGlobal:
    """/ANALY (M103): Global analysis type options.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_analy.F``.
    """
    n2d3d: int = 0
    analy_temp: int = 0
    iparith: int = 0


@dataclass
class UpwindGlobal:
    """/UPWIND (M103): Upwind advection factors for ALE.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_upwind.F``.
    """
    eta1: float = 0.0
    eta2: float = 0.0
    eta3: float = 0.0


@dataclass
class CaaControl:
    """/CAA (M103): Computational Aeroacoustics control.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_caa.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    grnod_id: int = 0
    sens_id: int = 0


# ----------------------------------------------------------------------------
# Virtual sensors, clusters, flexible bodies & advanced initial states (M104)
# ----------------------------------------------------------------------------

@dataclass
class Gauge:
    """/GAUGE (M104): Numerical strain/stress gauge virtual sensor.

    Fortran origin: ``starter/source/output/gauge/hm_read_gauge.F``.
    """
    id: int
    subtype: str = ""
    title: str = ""
    node_id: int = 0
    elem_id: int = 0
    dist: float = 0.0
    fcut: float = 0.0


@dataclass
class Cluster:
    """/CLUSTER (M104): Element failure/grouping cluster.

    Fortran origin: ``starter/source/output/cluster/hm_read_cluster.F``.
    """
    id: int
    subtype: str = ""
    title: str = ""
    group_id: int = 0
    skew_id: int = 0
    ifail: int = 0
    fn_fail: float = 0.0
    sca_a1: float = 0.0
    sca_b1: float = 0.0
    fs_fail: float = 0.0
    sca_a2: float = 0.0
    sca_b2: float = 0.0
    mt_fail: float = 0.0
    sca_a3: float = 0.0
    sca_b3: float = 0.0
    mb_fail: float = 0.0
    sca_a4: float = 0.0
    sca_b4: float = 0.0


@dataclass
class ExtLink:
    """/EXTLNK (M104): Multi-code external link coupling.

    Fortran origin: ``starter/source/coupling/rad2rad/lecextlnk.F``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0


@dataclass
class FxBody:
    """/FXBODY (M104): Component mode synthesis (CMS) flexible body.

    Fortran origin: ``starter/source/constraints/fxbody/hm_read_fxb.F``.
    """
    id: int
    title: str = ""
    node_id: int = 0
    ianim: int = 0
    imin: int = 0
    imax: int = 0
    filename: str = ""


@dataclass
class IniMap1D:
    """/INIMAP1D (M104/M167): 1D mapped field initial condition.

    Fortran origin: ``starter/source/initial_conditions/inimap/hm_read_inimap1d.F``.
    """
    id: int
    title: str = ""
    formulation: str = "FILE"  # VP, VE, FILE
    map_type: int = 0  # 1: Planar, 2: Cylindrical, 3: Spherical
    node_id1: int = 0
    node_id2: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    fscale_v: float = 1.0
    func_vel: int = 0
    fac_vel: float = 1.0
    nb_mat: int = 0
    func_alpha: List[int] = field(default_factory=list)
    func_rho: List[int] = field(default_factory=list)
    func_pres_ener: List[int] = field(default_factory=list)
    fac_rho: List[float] = field(default_factory=list)
    fac_pres_ener: List[float] = field(default_factory=list)
    filename: str = ""


@dataclass
class IniMap2D:
    """/INIMAP2D (M104/M167): 2D mapped field initial condition.

    Fortran origin: ``starter/source/initial_conditions/inimap/hm_read_inimap2d.F``.
    """
    id: int
    title: str = ""
    formulation: str = "FILE"  # VP, VE, FILE
    map_type: int = 0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    fscale_v: float = 1.0
    func_vel: int = 0
    fac_vel: float = 1.0
    nb_mat: int = 0
    func_alpha: List[int] = field(default_factory=list)
    func_rho: List[int] = field(default_factory=list)
    func_pres_ener: List[int] = field(default_factory=list)
    fac_rho: List[float] = field(default_factory=list)
    fac_pres_ener: List[float] = field(default_factory=list)
    filename: str = ""


@dataclass
class IniStateFile:
    """/INISTATE or /INISTATE/FILE (M104): External initial state file.

    Fortran origin: ``starter/source/initial_conditions/inista/hm_read_inista.F``.
    """
    filename: str = ""
    isigi: int = 0
    ioutp_fmt: int = 0


@dataclass
class LeakMat:
    """/LEAK/MAT, /LEAK/PART, /LEAK/AREA (M105, M133): Airbag fabric leakage model.

    Fortran origin: ``starter/source/airbag/hm_read_leak.F``.
    """
    id: int
    subtype: str = ""
    title: str = ""
    ileakage: int = 0
    scale_t: float = 1.0
    scale_p: float = 1.0
    acoeft1: float = 0.0
    fct_id_e: int = 0
    fscale_e: float = 1.0
    bcoeft1: float = 0.0
    acoeft2: float = 0.0
    fct_id_lc: int = 0
    fct_id_ac: int = 0
    fscale_lc: float = 1.0
    fscale_ac: float = 1.0
    # Ileakage == 5 micromechanical formulation:
    length: float = 1.0
    thick: float = 1.0
    c1: float = 0.0
    c2: float = 1.0
    c3: float = 0.0


@dataclass
class AleGrid:
    """/ALE/GRID (M105): ALE grid formulation and damping controls.

    Fortran origin: ``starter/source/ale/alelec.F``.
    """
    id: int = 1
    subtype: str = "STANDARD"
    title: str = ""
    dt_min: float = 0.0
    gamma: float = 0.0
    damp: float = 0.0
    nu_g: float = 0.0


@dataclass
class AleLink:
    """/ALE/LINK (M105): ALE grid link velocity condition.

    Fortran origin: ``starter/source/ale/alelec.F``.
    """
    id: int
    subtype: str = "VEL"
    title: str = ""
    grnod_id: int = 0
    fct_id: int = 0
    scale: float = 1.0


@dataclass
class AleSolver:
    """/ALE/SOLVER (M105): Global ALE momentum/interface solver.

    Fortran origin: ``starter/source/ale/alelec.F``.
    """
    imom: int = 0
    isfint: int = 0


@dataclass
class AleClose:
    """/ALE/CLOS or /ALE/CLOSE (M105): ALE mesh closing boundary distance.

    Fortran origin: ``starter/source/ale/hm_read_ale_close.F``.
    """
    htest: float = 0.0
    hclose: float = 0.0


@dataclass
class UserWindow:
    """/USERWI (M106): User window / data card lines.

    Fortran origin: ``starter/source/starter/userwi.F`` / CFG ``userwi.cfg``.
    """
    lines: List[str] = field(default_factory=list)


@dataclass
class Drape:
    """/DRAPE (M107): Composite fabric draping definition.

    Fortran origin: ``starter/source/properties/drape.F`` / CFG ``drape.cfg``.
    """
    id: int
    title: str = ""
    slices: List[Dict] = field(default_factory=list)


@dataclass
class IniBriEref:
    """/INIBRI/EREF (M107): Initial brick element reference state.

    Fortran origin: ``starter/source/elements/inibri_eref.F`` / CFG ``inibri_eref.cfg``.
    """
    elem_id: int = 0
    ref_elem_id: int = 0
    sub_objects: List[Dict] = field(default_factory=list)


@dataclass
class IncludeDyna:
    """/INCLUDE_DYNA (M107): LS-DYNA include file directive.

    Fortran origin: ``starter/source/starter/includedyna.F`` / CFG ``includedyna.cfg``.
    """
    filename: str = ""


@dataclass
class Autoposition:
    """/TRANSFORM/AUTOPOSITION (M111): Automated nodal repositioning.

    Fortran origin: ``starter/source/model/transformation/lectrans.F`` / CFG ``autoposition.cfg``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    surf_id: int = 0
    skew_id: int = 0
    dir: str = ""
    gap: float = 0.0
    pflag: int = 0
    xpos: float = 0.0
    ypos: float = 0.0
    zpos: float = 0.0
    xflag: int = 0
    yflag: int = 0
    zflag: int = 0


@dataclass
class LoadCentri:
    """/LOAD/CENTRI (M112): Centrifugal body force loading.

    Fortran origin: ``starter/source/loads/general/load_centri/hm_read_load_centri.F`` / CFG ``centri.cfg``.
    """
    id: int
    title: str = ""
    fct_id: int = 0
    dir: str = ""
    frame_id: int = 0
    sens_id: int = 0
    grnod_id: int = 0
    ivar: int = 0
    ascalex: float = 1.0
    fscaley: float = 1.0


@dataclass
class LoadPfluid:
    """/LOAD/PFLUID (M112): Hydrostatic/fluid pressure on surfaces.

    Fortran origin: ``starter/source/loads/general/pfluid/hm_read_pfluid.F`` / CFG ``pfluid.cfg``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    fct_hsp: int = 0
    ascalex_hsp: float = 1.0
    fscaley_hsp: float = 1.0
    dir_hsp: str = "Z"
    frame_hsp: int = 0
    fct_pc: int = 0
    ascalex_pc: float = 1.0
    fscaley_pc: float = 1.0
    fct_vel: int = 0
    ascalex_vel: float = 1.0
    fscaley_vel: float = 1.0
    dir_vel: str = ""
    frame_vel: int = 0


@dataclass
class LoadPressure:
    """/LOAD/PRESSURE & /LOAD/PFLUID (M112/M163): Hydroforming / directional pressure load.

    Fortran origin: ``starter/source/loads/general/load_pressure/hm_read_load_pressure.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    iload: int = 1
    sens_id: int = 0
    inorm: int = 1
    direction: str = ""
    skew_id: int = 0
    fct_id: int = 0
    xscale_p: float = 1.0
    yscale_p: float = 1.0
    inter_ids: List[int] = field(default_factory=list)
    gap_shifts: List[float] = field(default_factory=list)
    # Legacy fields
    scale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class LoadGravity:
    """/LOAD/GRAV or /LOAD/GRAVITY (M135): Gravitational field acceleration loading."""
    id: int
    title: str = ""
    grnod_id: int = 0
    dir_vector: tuple[float, float, float] = (0.0, 0.0, -1.0)
    funct_id: int = 0
    scale: float = 1.0
    sens_id: int = 0


@dataclass
class LoadBody:
    """/LOAD/BODY (M135): Volumetric body force loading."""
    id: int
    title: str = ""
    grpart_id: int = 0
    dir_vector: tuple[float, float, float] = (0.0, 0.0, -1.0)
    funct_id: int = 0
    scale: float = 1.0
    sens_id: int = 0


@dataclass
class LoadTherm:
    """/LOAD/HEAT or /LOAD/THERM (M135): Thermal heat flux loading."""
    id: int
    title: str = ""
    group_id: int = 0
    flux: float = 0.0
    funct_id: int = 0
    scale: float = 1.0
    sens_id: int = 0


@dataclass
class SphInOut:
    """/SPH/INOUT or /SPH/IO (M112/M202): SPH particle inlet/outlet boundary condition.

    Fortran origin: ``starter/source/loads/sph/hm_read_sphio.F``.
    """
    id: int
    title: str = ""
    ityp: int = 1  # 1: Inlet, 2: Outlet, 3: NRF, 4: Control section
    surf_id: int = 0
    part_id: int = 0
    pid: int = 0
    dist: float = 0.0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    fcut: float = 0.0
    coords: list[tuple[float, float, float]] = field(default_factory=list)
    # Inlet fields (ityp=1)
    fct_id_r: int = 0
    fscale_r: float = 1.0
    fct_id_e: int = 0
    fscale_e: float = 1.0
    fct_id_vn: int = 0
    # Outlet fields (ityp=2) & NRF (ityp=3)
    fct_id_p: int = 0
    fscale_p: float = 1.0
    lc: float = 0.0
    # Legacy fields
    fct_id: int = 0
    rho_in: float = 0.0
    p_in: float = 0.0
    e_in: float = 0.0


@dataclass
class SphBcs:
    """/SPHBCS (M113): SPH symmetry boundary condition.

    Fortran origin: ``starter/source/loads/sph/hm_read_sphbcs.F``.
    """
    id: int
    bcs_type: str = "SYM"  # 'SYM', 'CYCL', 'PERIOD'
    title: str = ""
    dir: str = "X"
    frame_id: int = 0
    grnod_id: int = 0
    ilevel: int = 0


@dataclass
class EulerBcs:
    """/EULER/BCS (M135): Eulerian domain boundary condition."""
    id: int
    title: str = ""
    grnod_id: int = 0
    bcs_type: str = "INFLOW"
    val1: float = 0.0
    val2: float = 0.0
    val3: float = 0.0


@dataclass
class HeatBcs:
    """/HEAT/BCS (M135): Thermal boundary condition."""
    id: int
    title: str = ""
    group_id: int = 0
    bcs_type: str = "TEMP"
    tval: float = 0.0
    funct_id: int = 0
    scale: float = 1.0
    sens_id: int = 0


@dataclass
class MadymoLink:
    """/MADYMO/LINK (M113): Madymo coupling link.

    Fortran origin: ``starter/source/madymo/hm_read_madymo_link.F``.
    """
    id: int
    title: str = ""
    mdref: int = 0
    node_id: int = 0


@dataclass
class MadymoExfem:
    """/MADYMO/EXFEM (M113): Madymo sub-model part exchange.

    Fortran origin: ``starter/source/madymo/hm_read_madymo_exfem.F``.
    """
    id: int
    title: str = ""
    part_ids: List[int] = field(default_factory=list)


@dataclass
class AleGridDonea:
    """/ALE/GRID/DONEA (M113): Donea ALE grid solver."""
    alpha: float = 0.0
    gamma: float = 100.0
    vel_x: float = 1.0
    vel_y: float = 1.0
    vel_z: float = 1.0
    v_min: float = -1e30


@dataclass
class AleGridSpring:
    """/ALE/GRID/SPRING (M113): Spring analogy ALE grid solver."""
    dt: float = 0.0
    gamma: float = 0.0
    damp: float = 0.5
    nu: float = 1.0
    v_min: float = -1e30


@dataclass
class AleGridStandard:
    """/ALE/GRID/STANDARD (M113): Standard ALE grid solver."""
    alpha: float = 0.0
    gamma: float = 0.0
    damp: float = 0.5
    l_c: float = 1.0


@dataclass
class AleGridDisp:
    """/ALE/GRID/DISP (M113): Displacement-based ALE grid solver."""
    u_max: float = -1e30
    v_min: float = -1e30


@dataclass
class AleGridLaplacian:
    """/ALE/GRID/LAPLACIAN (M113): Laplacian smoothing ALE grid solver."""
    alpha: float = 0.0
    gamma: float = 0.0
    damp: float = 0.5


@dataclass
class AleGridVolume:
    """/ALE/GRID/VOLUME (M113): Volume-preserving ALE grid solver."""
    alpha: float = 0.0
    gamma: float = 0.0


@dataclass
class AdmeshGlobal:
    """/ADMESH/GLOBAL, /ADGLOB, /ADGLOB/MESH (M113): Adaptive meshing global parameters."""
    level_max: int = 0
    iadm_rule: int = 0
    t_delay: float = 0.0
    istat_cnd: int = 0


@dataclass
class StampingInit:
    """/STAMPING (M113): Sheet metal forming stamping history input."""
    time_scale: float = 1.0
    data_lines: List[str] = field(default_factory=list)


@dataclass
class RandomNoise:
    """/RANDOM, /RANDOM/GRNOD (M113): Random vibration and stochastic input noise."""
    grnod_id: int = 0
    xalea: float = 0.0
    seed: float = 0.0


@dataclass
class Accelerometer:
    """/ACCEL (M113): Accelerometer measurement sensor."""
    id: int
    title: str = ""
    node_id: int = 0
    skew_id: int = 0
    cutoff: float = 0.0


@dataclass
class EbcsPropellant:
    """/EBCS/PROPELLANT or /BCS/PROPELLANT (M114, M200): Solid propellant combustion boundary condition.

    Fortran origin: ``starter/source/loads/ebcs/hm_read_ebcs_propellant.F90``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    sensor_id: int = 0
    submat_id: int = 1
    ienthalpy: int = 1
    rho0s: float = 0.0
    tburn: float = 300.0
    param_t: float = 300.0
    param_a: float = 0.0
    param_n: float = 0.0
    f_func_id: int = 0
    ffunc_id: int = 0
    f_scale_x: float = 1.0
    fscale_x: float = 1.0
    f_scale_y: float = 1.0
    fscale_y: float = 1.0
    g_func_id: int = 0
    gfunc_id: int = 0
    g_scale_x: float = 1.0
    gscale_x: float = 1.0
    g_scale_y: float = 1.0
    gscale_y: float = 1.0
    h_func_id: int = 0
    hfunc_id: int = 0
    h_scale_x: float = 1.0
    hscale_x: float = 1.0
    h_scale_y: float = 1.0
    hscale_y: float = 1.0

    def __post_init__(self):
        if not self.sensor_id and self.sens_id:
            self.sensor_id = self.sens_id
        elif not self.sens_id and self.sensor_id:
            self.sens_id = self.sensor_id
        if self.tburn != 300.0 and self.param_t == 300.0:
            self.param_t = self.tburn
        elif self.param_t != 300.0 and self.tburn == 300.0:
            self.tburn = self.param_t
        if not self.ffunc_id and self.f_func_id:
            self.ffunc_id = self.f_func_id
        elif not self.f_func_id and self.ffunc_id:
            self.f_func_id = self.ffunc_id
        if not self.gfunc_id and self.g_func_id:
            self.gfunc_id = self.g_func_id
        elif not self.g_func_id and self.gfunc_id:
            self.g_func_id = self.gfunc_id
        if not self.hfunc_id and self.h_func_id:
            self.hfunc_id = self.h_func_id
        elif not self.h_func_id and self.hfunc_id:
            self.h_func_id = self.hfunc_id
        if self.f_scale_x != 1.0 and self.fscale_x == 1.0:
            self.fscale_x = self.f_scale_x
        elif self.fscale_x != 1.0 and self.f_scale_x == 1.0:
            self.f_scale_x = self.fscale_x
        if self.f_scale_y != 1.0 and self.fscale_y == 1.0:
            self.fscale_y = self.f_scale_y
        elif self.fscale_y != 1.0 and self.f_scale_y == 1.0:
            self.f_scale_y = self.fscale_y
        if self.g_scale_x != 1.0 and self.gscale_x == 1.0:
            self.gscale_x = self.g_scale_x
        elif self.gscale_x != 1.0 and self.g_scale_x == 1.0:
            self.g_scale_x = self.gscale_x
        if self.g_scale_y != 1.0 and self.gscale_y == 1.0:
            self.gscale_y = self.g_scale_y
        elif self.gscale_y != 1.0 and self.g_scale_y == 1.0:
            self.g_scale_y = self.gscale_y
        if self.h_scale_x != 1.0 and self.hscale_x == 1.0:
            self.hscale_x = self.h_scale_x
        elif self.hscale_x != 1.0 and self.h_scale_x == 1.0:
            self.h_scale_x = self.hscale_x
        if self.h_scale_y != 1.0 and self.hscale_y == 1.0:
            self.hscale_y = self.h_scale_y
        elif self.hscale_y != 1.0 and self.h_scale_y == 1.0:
            self.h_scale_y = self.hscale_y



@dataclass
class AdmasNonUniformItem:
    """Item for /ADMAS/NON_UNIFORM (M114)."""
    mass: float = 0.0
    entity_id: int = 0
    iflag: int = 0


@dataclass
class AdmasNonUniform:
    """/ADMAS/NON_UNIFORM or /ADMAS/NON_UNIFORM_PART (M114): Non-uniform added mass list."""
    id: int
    kind: str = "NODE"  # 'NODE' | 'PART'
    items: List[AdmasNonUniformItem] = field(default_factory=list)


@dataclass
class SectCircle:
    """/SECT/CIRCLE (M114): Circular cross-section cut.

    Fortran origin: ``starter/source/tools/sect/hm_read_sect_circle.F``.
    """
    id: int
    title: str = ""
    n1: int = 0
    n2: int = 0
    n3: int = 0
    isave: int = 0
    delta_t: float = 0.0
    alpha: float = 0.0
    file_name: str = ""
    grbric_id: int = 0
    grshel_id: int = 0
    grtrus_id: int = 0
    grbeam_id: int = 0
    grsprg_id: int = 0
    grtria_id: int = 0
    int_ids: List[int] = field(default_factory=list)
    iframe: int = 0
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.0


@dataclass
class SectParal:
    """/SECT/PARAL (M114): Parallelogram cross-section cut.

    Fortran origin: ``starter/source/tools/sect/hm_read_sect_paral.F``.
    """
    id: int
    title: str = ""
    n1: int = 0
    n2: int = 0
    n3: int = 0
    isave: int = 0
    delta_t: float = 0.0
    alpha: float = 0.0
    file_name: str = ""
    grbric_id: int = 0
    grshel_id: int = 0
    grtrus_id: int = 0
    grbeam_id: int = 0
    grsprg_id: int = 0
    grtria_id: int = 0
    int_ids: List[int] = field(default_factory=list)
    iframe: int = 0
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    corner1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    corner2: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class DynainShell:
    """/DYNAIN/SHELL (M114): LS-DYNA shell history initialization."""
    option: str = "AUX/FULL"  # 'AUX/FULL' | 'STRES/FULL' | 'STRAIN/FULL'


@dataclass
class StateDt:
    """/STATE/DT or /DYNAIN/DT (M115): State output time-step controls."""
    tstart: float = 0.0
    tfreq: float = 0.0
    is_all: bool = False
    component_ids: List[int] = field(default_factory=list)


@dataclass
class SphReserve:
    """/SPH/RESERVE (M116): SPH reserve particle buffer allocation."""
    part_id: int
    np_particles: int = 0


@dataclass
class MoveFunct:
    """/MOVE_FUNCT (M116): Function curve scale and shift transformation."""
    id: int
    title: str = ""
    a_scale_x: float = 1.0
    f_scale_y: float = 1.0
    a_shift_x: float = 0.0
    f_shift_y: float = 0.0


@dataclass
class EigenMode:
    """/EIG (M117): Eigenvalue extraction & modal analysis configuration."""
    id: int
    title: str = ""
    grnod_id: int = 0
    grnod_bc: int = 0
    trarot: str = ""
    ifile: int = 0
    imls: int = 0
    nmod: int = 0
    inorm: int = 0
    cutfreq: float = 0.0
    freqmin: float = 0.0
    nbloc: int = 0
    incv: int = 0
    niter: int = 0
    ipri: int = 0
    tol: float = 0.0
    filename: str = ""


@dataclass
class StressFile:
    """/STATE/STR_FILE or /STR_FILE (M117): Stress output file specification."""
    izip: int = 0
    filename: str = ""


@dataclass
class MemoryRequest:
    """/MEMORY (M117): Memory allocation request."""
    nmots: int = 0
    rate: float = 0.66


@dataclass
class TransformPosition:
    """/TRANSFORM/POSITION or /TRANSFORM/POS (M118): Spatial positioning transform."""
    id: int
    title: str = ""
    grnod_id: int = 0
    node_ids: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    submodel: int = 0
    points: tuple[tuple[float, float, float], ...] = ()


@dataclass
class TransformProjection:
    """/TRANSFORM/PROJ (M134): Node group projection transformation."""
    id: int
    title: str = ""
    grnod_id: int = 0
    proj_type: str = "PLANE"
    target_id: int = 0
    dir_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    dist: float = 0.0


@dataclass
class TransformFrame:
    """/TRANSFORM/FRAME (M134): Coordinate frame transformation."""
    id: int
    title: str = ""
    grnod_id: int = 0
    frame_orig: int = 0
    frame_dest: int = 0


@dataclass
class ExternalLink:
    """/EXTERN/LINK or /EXTLINK (M118): External interface link."""
    id: int
    title: str = ""
    grnod_id: int = 0


@dataclass
class ArchSpec:
    """/ARCH (M118): Architecture specification card."""
    mach: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0)


@dataclass
class FunctPython:
    """/FUNCT_PYTHON/id (M119): Python mathematical function definition."""
    id: int
    lines: list[str] = field(default_factory=list)


@dataclass
class RefstaNode:
    """/REFSTA (M119): Reference state node coordinate."""
    node_id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class ErefSpec:
    """/EREF (M119): Element reference configuration."""
    id: int
    title: str = ""
    part_id: int = 0
    subtype: str = ""
    elem_coords: list[tuple[int, tuple[tuple[float, float, float], ...]]] = field(default_factory=list)


@dataclass
class AleMuscl:
    """/ALE/MUSCL (M119): MUSCL advection compression factor."""
    beta: float = 2.0


@dataclass
class BemModel:
    """/BEM (M119): Boundary element method container."""
    id: int = 1
    title: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GaugePoint:
    """/GAUGE/POINT (M121): Point gauge definition for spatial measurement."""
    id: int
    title: str = ""
    subtitle: str = ""
    points: list[tuple[float, float, float, float, str]] = field(default_factory=list) # (x, y, z, dist, subtitle)


@dataclass
class SphGlo:
    """/SPHGLO (M121): Global SPH particle formulation settings."""
    alpha_sort: float = 0.25
    maxsph: int = 0
    lneigh: int = 120
    nneigh: int = 120
    isol2sph: int = 0


@dataclass
class AnalyOptions:
    """/ANALY (M121): Global analysis dimension and arithmetic options."""
    n2d3d: int = 0         # 0=3D, 1=axisymmetric, 2=plane strain
    iparith: int = 1       # 1=ON, 2=OFF
    isubcyc: int = 0       # 0=none, 2=subcycling n2


@dataclass
class AleCfdSph:
    """/ALECFDSPH (M122): Coupled ALE / CFD / SPH fluid-structure interaction parameters."""
    title: str = ""
    icfd: int = 0
    isph: int = 0
    tstart: float = 0.0
    tstop: float = 1e30
    fscale_c: float = 1.0
    fscale_s: float = 1.0


@dataclass
class EbcsNrf:
    """/EBCS/NRF or /EBCS/NON_REFLECT (M125): Non-reflecting frontier boundary condition.

    Fortran origin: ``starter/source/loads/ebcs/hm_read_ebcs_nrf.F`` / CFG ``ebcs_nrf.cfg``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    tcar_p: float = 0.0
    tcar_vf: float = 0.0


@dataclass
class EbcsPeriodic:
    """/EBCS/PERIODIC (M138): Eulerian periodic boundary condition.

    Fortran origin: ``starter/source/loads/ebcs/hm_read_ebcs_perio.F``.
    """
    id: int
    title: str = ""
    surf1_id: int = 0
    surf2_id: int = 0
    skew_id: int = 0
    grpart_id: int = 0


@dataclass
class EbcsCyclic:
    """/EBCS/CYCLIC (M138, M200): Eulerian cyclic boundary condition.

    Fortran origin: ``starter/source/loads/ebcs/hm_read_ebcs_cyclic.F90``.
    """
    id: int
    title: str = ""
    surf1_id: int = 0
    surf2_id: int = 0
    skew_id: int = 0
    grpart_id: int = 0
    surf_id1: int = 0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    surf_id2: int = 0
    node_id4: int = 0
    node_id5: int = 0
    node_id6: int = 0

    def __post_init__(self):
        if not self.surf_id1 and self.surf1_id:
            self.surf_id1 = self.surf1_id
        elif not self.surf1_id and self.surf_id1:
            self.surf1_id = self.surf_id1
        if not self.surf_id2 and self.surf2_id:
            self.surf_id2 = self.surf2_id
        elif not self.surf2_id and self.surf_id2:
            self.surf2_id = self.surf_id2


@dataclass
class DetLine:
    """/DFS/DETLINE (M137): Detonation along a line segment."""
    id: int
    title: str = ""
    p1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    p2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    t0: float = 0.0
    dvel: float = 0.0
    node1: int = 0
    node2: int = 0
    mat_id: int = 0


@dataclass
class DetCirc:
    """/DFS/DETCIRC (M137): Detonation along a circle."""
    id: int
    title: str = ""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius: float = 0.0
    t0: float = 0.0
    dvel: float = 0.0


@dataclass
class IniMap3D:
    """/INIMAP/3D (M137): 3D solution mapping descriptor."""
    id: int
    title: str = ""
    map_type: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    filename: str = ""
    fscale_v: float = 1.0


@dataclass
class ErefElement:
    """/EREF/{SHELL|SH3N|BRICK|TETRA4} (M143): Element reference geometry.

    Fortran origin: ``starter/source/initial_conditions/general/hm_read_eref.F``.
    """
    id: int
    title: str = ""
    elem_type: str = "SHELL"
    part_id: int = 0
    node_coords: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class AdmeshControl:
    """/ADMESH/{GLOBAL|PART|STATE|BCS|SET} (M143): Adaptive mesh refinement control.

    Fortran origin: ``starter/source/model/remesh/build_admesh.F``.
    """
    id: int
    title: str = ""
    subtype: str = "GLOBAL"
    crit_level: int = 0
    h_min: float = 0.0
    h_max: float = 0.0
    part_id: int = 0


@dataclass
class LoadHydro:
    """/LOAD/HYDRO or /LOAD/HYDROSTATIC (M144): Hydrostatic surface pressure loading.

    Fortran origin: ``starter/source/loads/general/hm_read_load.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    density: float = 0.0
    z_free: float = 0.0
    gravity: float = 9.81
    sens_id: int = 0


@dataclass
class Func2DTable:
    """/FUNC_2D or /FUNC2D (M145): 2D bivariate function table f(x, y).

    Fortran origin: ``starter/source/tools/curve/hm_read_func2d.F``.
    """
    id: int
    title: str = ""
    dim: int = 1
    x_vals: List[float] = field(default_factory=list)
    y_vals: List[float] = field(default_factory=list)
    z_vals: List[float] = field(default_factory=list)


@dataclass
class NonlocalModel:
    """/NONLOCAL/mat_id (M145): Non-local damage regularization model.

    Fortran origin: ``starter/source/materials/nonlocal/hm_read_nonlocal.F``.
    """
    mat_id: int
    title: str = ""
    length: float = 0.0
    le_max: float = 0.0
    dens: float = 0.0
    damp: float = 0.0


@dataclass
class FricOrient:
    """/FRIC_ORIENT/id (M145): Friction orientation & anisotropic contact directions.

    Fortran origin: ``starter/source/interfaces/friction/reader/hm_read_friction_orientations.F``.
    """
    id: int
    title: str = ""
    grpart_id: int = 0
    skew_id: int = 0
    phi: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ifric: int = 0


@dataclass
class IniSphCel:
    """/INISPHCEL/part_id (M145): SPH cell initial state.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F``.
    """
    part_id: int
    p: float = 0.0
    rho: float = 0.0
    e: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class EbcsPres:
    """/EBCS/PRES/id (M150): Eulerian imposed pressure boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_pres.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    c: float = 0.0
    fct_pres: int = 0
    scale_pres: float = 1.0
    fct_rho: int = 0
    scale_rho: float = 1.0
    fct_en: int = 0
    scale_en: float = 1.0
    lcar: float = 0.0
    r1: float = 0.0
    r2: float = 0.0


@dataclass
class EbcsVel:
    """/EBCS/VEL/id (M150): Eulerian imposed velocity boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_vel.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    c: float = 0.0
    fct_vx: int = 0
    scale_vx: float = 0.0
    fct_vy: int = 0
    scale_vy: float = 0.0
    fct_vz: int = 0
    scale_vz: float = 0.0
    fct_rho: int = 0
    scale_rho: float = 1.0
    fct_en: int = 0
    scale_en: float = 1.0
    lcar: float = 0.0
    r1: float = 0.0
    r2: float = 0.0


@dataclass
class EbcsInlet:
    """/EBCS/INLET/id (M150): Eulerian inflow boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_inlet.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    density: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    energy: float = 0.0
    fct_id: int = 0


@dataclass
class EbcsFluxout:
    """/EBCS/FLUXOUT/id (M150): Eulerian mass outflow boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_fluxout.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    p_ext: float = 0.0


@dataclass
class EbcsGradp0:
    """/EBCS/GRADP0/id (M150): Eulerian zero pressure gradient boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_gradp0.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0


@dataclass
class EbcsNormv:
    """/EBCS/NORMV/id (M150): Eulerian normal velocity constraint.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_normv.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    vn: float = 0.0
    fct_id: int = 0


@dataclass
class EbcsValv:
    """/EBCS/VALVIN or /EBCS/VALVOUT (M150): Eulerian valve boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_valvin.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    is_out: bool = False
    p_open: float = 0.0
    p_close: float = 0.0


@dataclass
class EbcsMonvol:
    """/EBCS/MONVOL/id (M150): Eulerian monitored volume boundary connection.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_monvol.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    monvol_id: int = 0
    sens_id: int = 0
    fscale: float = 0.0

@dataclass
class SeatbeltSystem:
    """/SEATBELT/id (M150): Complete seatbelt system assembly.

    Fortran origin: ``starter/source/tools/seatbelts/create_seatbelt.F``.
    """
    id: int
    title: str = ""
    retractor_ids: list[int] = field(default_factory=list)
    slipring_ids: list[int] = field(default_factory=list)
    element_ids: list[int] = field(default_factory=list)


@dataclass
class AmsControl:
    """/AMS (M150): Advanced Mass Scaling starter control.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_sms.F``.
    """
    id: int = 0
    title: str = ""
    grpart_id: int = 0
    dt_target: float = 0.0
    i_ams: int = 1






@dataclass
class Inista:
    """/INISTA or /INISTATE (M151): Initial state file input.

    Fortran origin: ``starter/source/initial_conditions/inista/hm_read_inista.F``.
    """
    id: int = 0
    title: str = ""
    filename: str = ""
    ibal: int = 1
    ioutyy: int = 0
    ioutynn: int = 0


@dataclass
class BemControl:
    """/BEM/FLOW or /BEM/DAA (M151): Boundary element method controls.

    Fortran origin: ``starter/source/loads/bem/hm_read_bem.F``.
    """
    id: int
    title: str = ""
    subtype: str = "FLOW"
    surf_id: int = 0
    nio: int = 0
    grnod_aux_id: int = 0
    freesurf: int = 1


@dataclass
class EbcsInip:
    """/EBCS/INIP (M152): Eulerian initial pressure boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_inip.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    rho: float = 0.0
    c: float = 0.0
    lcar: float = 0.0


@dataclass
class EbcsIniv:
    """/EBCS/INIV (M152): Eulerian initial velocity boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_iniv.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    rho: float = 0.0
    c: float = 0.0
    lcar: float = 0.0


@dataclass
class DfsDetcord:
    """/DFS/DETCORD (M163): Detonation cord ignition model.

    Fortran origin: ``starter/source/initial_conditions/detonation/read_dfs_detcord.F``.
    """
    id: int
    title: str = ""
    grnd_id: int = 0
    t_det: float = 0.0
    v_cj: float = 0.0
    iopt: int = 3
    mat_id: int = 0
    nodes: List[int] = field(default_factory=list)


@dataclass
class AleMat:
    """/ALE/MAT (M163): ALE material volume fraction and formulation directives.

    Fortran origin: ``starter/source/materials/ale/read_ale_mat.F``.
    """
    mat_id: int
    ale_flrd: float = 0.0


@dataclass
class EulerMat:
    """/EULER/MAT (M163): Euler material volume fraction and formulation directives.

    Fortran origin: ``starter/source/materials/ale/read_euler_mat.F``.
    """
    mat_id: int
    euler_flrd: float = 0.0

@dataclass
class EbcsLoad:
    """/EBCS/{PRES|VEL|INLET} (M164): Eulerian boundary condition loading directive.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/read_ebcs.F``.
    """
    id: int
    kind: str = ""
    title: str = ""
    surf_id: int = 0
    fct_id: int = 0
    sens_id: int = 0
    dir: str = ""
    v0: float = 0.0
    p0: float = 0.0
    scale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    imat: int = 0
    rho: float = 0.0
    ener: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class InertiaPart:
    """/INERTIA/PART (M169): Part inertia modifier definition.

    Fortran origin: ``starter/source/nodes_elements/inertia/hm_read_inertia.F``.
    """
    id: int
    title: str = ""
    part_id: int = 0
    skew_id: int = 0
    iflag: int = 0
    mass: float = 0.0
    xg: float = 0.0
    yg: float = 0.0
    zg: float = 0.0
    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    ixy: float = 0.0
    iyz: float = 0.0
    izx: float = 0.0


@dataclass
class LoadCload:
    """/LOAD/CLOAD or /CLOAD (M181): Concentrated nodal load."""
    id: int
    curve_id: int = 0
    dir: str = "X"
    skew_id: int = 0
    sens_id: int = 0
    grnod_id: int = 0
    xscale: float = 1.0
    magnitude: float = 1.0
    title: str = ""


@dataclass
class LoadPload:
    """/LOAD/PLOAD or /PLOAD (M181): Surface pressure load."""
    id: int
    surf_id: int = 0
    curve_id: int = 0
    sens_id: int = 0
    ipinch: int = 0
    idel: int = 1
    functype: int = 1
    xscale: float = 1.0
    magnitude: float = 1.0
    title: str = ""


@dataclass
class DefInterType11:
    """``/DEF_INTER/TYPE11``: Default parameters for interface TYPE11."""
    istf: int = 5
    igap: int = 1000
    ikrem: int = 1
    noddel11: int = 1000
    iform: int = 1
    inactiv: int = 1000


@dataclass
class DefInterType19:
    """``/DEF_INTER/TYPE19``: Default parameters for interface TYPE19."""
    istf: int = 1000
    igap: int = 1000
    iedge: int = 2
    ibag: int = 2
    idel7: int = 1000
    icurv: int = 0
    inactiv: int = 1000
    iform: int = 1


@dataclass
class DefInterType25:
    """``/DEF_INTER/TYPE25``: Default parameters for interface TYPE25."""
    istf: int = 0
    igap: int = 0
    irem_i2: int = 0
    idel: int = 0
    itied: int = 0
    ishape: int = 0
    irs: int = 1000


@dataclass
class StateDirective:
    """``/STATE/...``: Element state initialization / restart directives."""
    kind: str = ""
    subtype: str = ""
    option: int = 0
    val: float = 0.0


@dataclass
class SphFlow:
    """``/SPH_FLOW/id`` or ``/SPH/FLOW/id`` (M194): SPH flow boundary condition."""
    id: int
    title: str = ""
    surf_id: int = 0
    part_id: int = 0
    fct_id: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class MidDirective:
    """``/MID/id`` (M194): Material ID assignment / mapping card."""
    id: int
    mat_id: int
    title: str = ""


@dataclass
class PidDirective:
    """``/PID/id`` (M194): Part/Property ID assignment / mapping card."""
    id: int
    prop_id: int
    title: str = ""


@dataclass
class SphParticle:
    """SPH particle element (M194)."""
    id: int
    part_id: int = 0
    node_id: int = 0


@dataclass
class DefInterType2:
    """``/DEF_INTER/TYPE2`` (M194): Default Type 2 interface parameters."""
    istf: int = 0
    igap: int = 0
    iref: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class EngineTHRecord:
    """Engine /TH time history request record (M194)."""
    th_type: str
    id: int = 0
    title: str = ""
    vars: list = field(default_factory=list)
    ids: list = field(default_factory=list)


@dataclass
class FunctSmooth:
    """``/FUNCT_SMOOTH`` (M195): Smoothed curve function."""
    id: int
    title: str = ""
    smooth_type: int = 0
    order: int = 3
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def evaluate(self, x_val: float) -> float:
        """Evaluate function at x_val using linear/interpolated curve."""
        if not self.x:
            return 0.0
        if len(self.x) == 1:
            return float(self.y[0])
        return float(np.interp(x_val, self.x, self.y))


@dataclass
class IniStateTable:
    """Generic initial state table record for /INI* keywords (M195)."""
    keyword: str = ""
    id: int = 0
    title: str = ""
    rows: List[dict] = field(default_factory=list)


@dataclass
class FrameNod:
    """``/FRAME/NOD`` or ``/FRAME/NODE`` (M196): Nodal coordinate reference frame."""
    id: int = 0
    title: str = ""
    originnodeid: int = 0
    axisnodeid: int = 0
    planenodeid: int = 0
    globalyaxis: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    globalzaxis: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    displayaxis: int = 0
    displayplane: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class TableBlock:
    """``/TABLE``, ``/TABLE/0``, ``/TABLE/1``: Multi-dimensional lookup tables."""
    id: int = 0
    title: str = ""
    dim: int = 1
    ref_id: int = 0
    x_values: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)
    curves: List[int] = field(default_factory=list)
    params: dict = field(default_factory=dict)

@dataclass
class CNode:
    """``/CNODE`` (M198): Commented coordinate node definition.

    Fortran origin: ``starter/source/elements/reader/hm_read_node.F`` / CFG ``cnode.cfg``.
    """
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    comments: List[str] = field(default_factory=list)


@dataclass
class Upbeam:
    """``/UPBEAM`` or ``/UPBEAM/INT_BEAM`` (M198): Integrated beam cross-section update."""
    id: int
    title: str = ""
    grnd_id: int = 0
    i_updt: int = 0
    eps_max: float = 0.0
    npt_int: int = 0


@dataclass
class RelaxSystem:
    """``/RELAX`` or ``/RELAX/SYSTEM`` or ``/RELAX/DYNA`` (M198): Quasi-static dynamic relaxation."""
    id: int
    title: str = ""
    t_start: float = 0.0
    t_stop: float = 0.0
    damp_coeff: float = 0.0
    i_damp: int = 0
    v_lim: float = 0.0
    eps_tol: float = 0.0


@dataclass
class TransformMatrix:
    """``/TRANSFORM/MATRIX`` or ``/MATRIX`` (M199): Affine 3D matrix transformation."""
    id: int
    title: str = ""
    grnod_id: int = 0
    matrix: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    translation: tuple = (0.0, 0.0, 0.0)
    sub_id: int = 0
    submodel: int = 0

    @property
    def m11(self) -> float:
        return self.matrix[0][0]

    @property
    def m12(self) -> float:
        return self.matrix[0][1]

    @property
    def m13(self) -> float:
        return self.matrix[0][2]

    @property
    def m21(self) -> float:
        return self.matrix[1][0]

    @property
    def m22(self) -> float:
        return self.matrix[1][1]

    @property
    def m23(self) -> float:
        return self.matrix[1][2]

    @property
    def m31(self) -> float:
        return self.matrix[2][0]

    @property
    def m32(self) -> float:
        return self.matrix[2][1]

    @property
    def m33(self) -> float:
        return self.matrix[2][2]

    @property
    def tx(self) -> float:
        return self.translation[0]

    @property
    def ty(self) -> float:
        return self.translation[1]

    @property
    def tz(self) -> float:
        return self.translation[2]


# ============================================================================
# M200 Entities: DETPOINT, DTIX
# ============================================================================

@dataclass
class DetPointNode:
    """``/DFS/DETPOINT/NODE`` or ``/DETPOINT/NODE`` (M200): Detonation point at node."""
    id: int = 0
    ishadow: int = 0
    iframe1: int = 0
    iframe2: int = 0
    r0_shadow: float = 0.0
    radius: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    node_id1: int = 0
    node_id: int = 0
    title: str = ""

    def __post_init__(self):
        if not self.r0_shadow and self.radius:
            self.r0_shadow = self.radius
        elif not self.radius and self.r0_shadow:
            self.radius = self.r0_shadow
        if not self.node_id1 and self.node_id:
            self.node_id1 = self.node_id
        elif not self.node_id and self.node_id1:
            self.node_id = self.node_id1


@dataclass
class DetPointSet:
    """``/DFS/DETPOINT/SET`` or ``/DETPOINT/SET`` / ``/DETPOINT/GRNOD`` (M200): Detonation point on node group."""
    id: int = 0
    ishadow: int = 0
    iframe1: int = 0
    iframe2: int = 0
    r0_shadow: float = 0.0
    radius: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    grnod_id1: int = 0
    grnod_id: int = 0
    title: str = ""

    def __post_init__(self):
        if not self.r0_shadow and self.radius:
            self.r0_shadow = self.radius
        elif not self.radius and self.r0_shadow:
            self.radius = self.r0_shadow
        if not self.grnod_id1 and self.grnod_id:
            self.grnod_id1 = self.grnod_id
        elif not self.grnod_id and self.grnod_id1:
            self.grnod_id = self.grnod_id1


@dataclass
class DtixControl:
    """``/DTIX`` or ``/ENG/DTIX`` (M200): Initial and maximum explicit time step control."""
    id: int = 1
    t_ini: float = 0.0
    t_max: float = 0.0
    tini: float = 0.0
    tmax: float = 0.0

    def __post_init__(self):
        if not self.t_ini and self.tini:
            self.t_ini = self.tini
        elif not self.tini and self.t_ini:
            self.tini = self.t_ini
        if not self.t_max and self.tmax:
            self.t_max = self.tmax
        elif not self.tmax and self.t_max:
            self.tmax = self.t_max


@dataclass
class DfsWavSha:
    """``/DFS/WAV_SHA/id`` or ``/WAVE/id`` (M201): Wave shaper / spherical ignition modifier."""
    id: int = 0
    title: str = ""
    xdet: float = 0.0
    ydet: float = 0.0
    zdet: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    grnod_id: int = 0
WaveShaperDfs = DfsWavSha


@dataclass
class HeatConvec:
    """``/HEAT/CONVEC/id``, ``/HEAT/CONVECTION/id`` or ``/CONVEC/id`` (M202): Thermal surface convection boundary condition."""
    id: int = 0
    title: str = ""
    surf_id: int = 0
    funct_id: int = 0
    sensor_id: int = 0
    ascale: float = 1.0
    fscale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    h: float = 0.0


@dataclass
class HeatRadiation:
    """``/HEAT/RADIATION/id``, ``/HEAT/RAD/id`` or ``/RADIATION/id`` (M202): Thermal surface radiation boundary condition."""
    id: int = 0
    title: str = ""
    surf_id: int = 0
    funct_id: int = 0
    sensor_id: int = 0
    ascale: float = 1.0
    fscale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    emissivity: float = 0.0
    emiss: float = 0.0

    def __post_init__(self):
        if self.emissivity != 0.0 and self.emiss == 0.0:
            self.emiss = self.emissivity
        elif self.emiss != 0.0 and self.emissivity == 0.0:
            self.emissivity = self.emiss


@dataclass
class Spcnd:
    """``/SPCND/id`` (M202): Single point constraint on node."""
    id: int = 0
    title: str = ""
    node_id: int = 0
    dof: str = "111111"
    f_sens: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    val: float = 0.0


@dataclass
class Ddw:
    """``/DDW/id`` (M202): Deep draw wall stamping tool."""
    id: int = 0
    title: str = ""
    tool_type: int = 1
    surf_id: int = 0
    grnod_id: int = 0
    fct_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    surf1_id: int = 0
    surf2_id: int = 0
    f_hold: float = 0.0
    f_draw: float = 0.0
    iform: int = 0
    points: List[Any] = field(default_factory=list)


@dataclass
class DdwPoint:
    """``/DDW/POINT/id`` (M202): Point-based deep draw wall."""
    id: int = 0
    title: str = ""
    node_id: int = 0
    fct_id: int = 0
    dir: str = "Z"
    scale: float = 1.0


@dataclass
class Stamping:
    """``/STAMPING`` or ``/STAMP`` (M202): Stamping simulation controls."""
    hf_timescale: float = 1.0
    datalines: list[str] = field(default_factory=list)


@dataclass
class WindowUser:
    """``/WINDOW/USER/id`` or ``/USERWI/id`` (M202): User-defined analysis window."""
    id: int = 0
    title: str = ""
    lines: list[str] = field(default_factory=list)


@dataclass
class HeatFlux:
    """``/HEAT/FLUX/id`` or ``/FLUX/id`` (M202): Thermal heat flux boundary condition."""
    id: int = 0
    title: str = ""
    surf_id: int = 0
    funct_id: int = 0
    sensor_id: int = 0
    ascale: float = 1.0
    fscale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    q: float = 0.0


@dataclass
class TransformPos:
    """``/TRANSFORM/POS/id``, ``/TRANSFORM/POSITION/id``, ``/POS/id`` (M203): 6-point 3D alignment transformation."""
    id: int = 0
    title: str = ""
    grnod_id: int = 0
    node1: int = 0
    node2: int = 0
    node3: int = 0
    node4: int = 0
    node5: int = 0
    node6: int = 0
    node_ids: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    submodel_id: int = 0
    submodel: int = 0
    points: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not any(self.node_ids) and any((self.node1, self.node2, self.node3, self.node4, self.node5, self.node6)):
            self.node_ids = (self.node1, self.node2, self.node3, self.node4, self.node5, self.node6)
        elif any(self.node_ids) and not any((self.node1, self.node2, self.node3, self.node4, self.node5, self.node6)):
            self.node1 = self.node_ids[0] if len(self.node_ids) > 0 else 0
            self.node2 = self.node_ids[1] if len(self.node_ids) > 1 else 0
            self.node3 = self.node_ids[2] if len(self.node_ids) > 2 else 0
            self.node4 = self.node_ids[3] if len(self.node_ids) > 3 else 0
            self.node5 = self.node_ids[4] if len(self.node_ids) > 4 else 0
            self.node6 = self.node_ids[5] if len(self.node_ids) > 5 else 0
        if self.submodel_id != 0 and self.submodel == 0:
            self.submodel = self.submodel_id
        elif self.submodel != 0 and self.submodel_id == 0:
            self.submodel_id = self.submodel


PosTransform = TransformPos



@dataclass
class ChecksumDirective:
    """``/CHECKSUM/START``, ``/CHECKSUM/END`` (M203): Checksum calculation block directive."""
    id: int = 0
    title: str = ""
    action: str = "START"  # "START" or "END"
    val1: int = 0
    val2: int = 0


@dataclass
class AdmeshSet:
    """``/ADMESH/SET/id`` (M203): Adaptive meshing on element/node set."""
    id: int = 0
    title: str = ""
    angle_criteria: float = 0.0
    inilev: int = 0
    thkerr: float = 0.0
    part_ids: List[int] = field(default_factory=list)
    grnd_id: int = 0
    level: int = 0
    tdelay: float = 0.0


@dataclass
class GaugeSph:
    """``/GAUGE/SPH/id`` (M203): SPH gauge point measurement."""
    id: int = 0
    title: str = ""
    node_id: int = 0
    fcut: float = 0.0
    shell_id: int = 0
    dist: float = 0.0


@dataclass
class HeatSolver:
    """``/HEAT/SOLVER/id``, ``/HEAT/GLOBAL/id`` (M204): Thermal transient solver controls."""
    id: int = 1
    title: str = ""
    isolv: int = 1
    itype: int = 1
    ttol: float = 1e-3
    dttmax: float = 1.0
    dttmin: float = 1e-6


@dataclass
class XfemControl:
    """``/XFEM[/<subtype>]/id`` (M204): X-FEM extended finite element enrichment control."""
    id: int = 0
    title: str = ""
    subtype: str = "SHELL"
    grpart_id: int = 0
    crack_id: int = 0
    ifail: int = 0
    i_enrich: int = 1
HeatGlobal = HeatSolver


@dataclass
class FlowBoundary:
    """``/FLOW[/<subtype>]/id`` or ``/ALE/FLOW/id`` (M205): ALE flow boundary condition."""
    id: int = 1
    title: str = ""
    subtype: str = "INFLOW"
    surf_id: int = 0
    flow_type: int = 0
    rho: float = 0.0
    pres: float = 0.0
    temp: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    sens_id: int = 0


@dataclass
class HeatRadCav:
    """``/HEAT/RAD_CAV/id`` (M205): Cavity radiation surface-to-surface coupling."""
    id: int = 1
    title: str = ""
    surf_id1: int = 0
    surf_id2: int = 0
    emissivity1: float = 1.0
    emissivity2: float = 1.0
    view_factor: float = 1.0


@dataclass
class RBodyStop:
    """``/RBODY/STOP`` or ``/ENG/RBODY/STOP`` (M215): Rigid body sensor stop / activation control."""
    id: int = 1
    title: str = ""
    rbody_id: int = 0
    sens_id: int = 0
    istop_opt: int = 0       # 0: freeze velocity, 1: deactivate constraint


@dataclass
class EngMonitor:
    """``/MONITOR`` or ``/ENG/MONITOR`` (M216): Periodic terminal monitor of nodal variables."""
    id: int = 1
    title: str = ""
    node_id: int = 0
    ivar_type: int = 1       # 1: DX, 2: DY, 3: DZ, 4: VX, 5: VY, 6: VZ, 7: AX, 8: AY, 9: AZ
    dt_print: float = 0.0    # console print time interval


@dataclass
class EngFxfreq:
    """``/FXFREQ`` or ``/ENG/FXFREQ`` (M217): Fast Fourier Transform frequency spectrum output."""
    id: int = 1
    title: str = ""
    f_max: float = 0.0       # maximum frequency evaluated
    n_freq: int = 100        # number of frequency points
    t_start: float = 0.0     # window start time
    t_end: float = 0.0       # window end time


@dataclass
class EngTrack:
    """``/TRACK`` or ``/ENG/TRACK`` (M218): Nodal trajectory tracking output."""
    id: int = 1
    title: str = ""
    node_id: int = 0         # node ID to track
    skew_id: int = 0         # reference frame
    dt_track: float = 0.0    # recording time interval


@dataclass
class EngHelm:
    """``/HELM`` or ``/ENG/HELM`` (M219): Helmholtz acoustic frequency response directive."""
    id: int = 1
    title: str = ""
    freq_start: float = 0.0  # start frequency
    freq_end: float = 0.0    # end frequency
    n_step: int = 10         # number of frequency steps


@dataclass
class EngTrunc:
    """``/TRUNC`` or ``/ENG/TRUNC`` (M220): Engine cycle truncation and tolerance directive."""
    id: int = 1
    title: str = ""
    tol_trunc: float = 0.0   # truncation tolerance
    dt_min: float = 0.0      # minimum time step
    n_cycle: int = 1         # check cycle interval


@dataclass
class EngMass:
    """``/MASS`` or ``/ENG/MASS`` (M221): Engine mass summary output directive."""
    id: int = 1
    title: str = ""
    dt_mass: float = 0.0     # time frequency for mass summary
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngEnergy:
    """``/ENERGY`` or ``/ENG/ENERGY`` (M222): Engine energy balance tracking output directive."""
    id: int = 1
    title: str = ""
    dt_energy: float = 0.0   # time frequency for energy balance
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngMoment:
    """``/MOMENT`` or ``/ENG/MOMENT`` (M223): Engine momentum tracking output directive."""
    id: int = 1
    title: str = ""
    dt_mom: float = 0.0      # time frequency for momentum output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngState:
    """``/STATE`` or ``/ENG/STATE`` (M224): Engine state variable tracking output directive."""
    id: int = 1
    title: str = ""
    dt_state: float = 0.0    # time frequency for state variable output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngSurf:
    """``/SURF`` or ``/ENG/SURF`` (M225): Engine contact surface force tracking directive."""
    id: int = 1
    title: str = ""
    dt_surf: float = 0.0     # time frequency for surface output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngAle:
    """``/ALE`` or ``/ENG/ALE`` (M226): Engine ALE smoothing and advection directive."""
    id: int = 1
    title: str = ""
    dt_ale: float = 0.0      # time frequency for ALE smoothing/advection
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngShThick:
    """``/SH_THICK`` or ``/ENG/SH_THICK`` (M227): Engine shell thickness update directive."""
    id: int = 1
    title: str = ""
    dt_thick: float = 0.0    # time frequency for shell thickness output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngGeo:
    """``/GEO`` or ``/ENG/GEO`` (M228): Engine nodal coordinate geometry update directive."""
    id: int = 1
    title: str = ""
    dt_geo: float = 0.0      # time frequency for geometry update
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngTens:
    """``/TENS`` or ``/ENG/TENS`` (M229): Engine tensor output tracking directive."""
    id: int = 1
    title: str = ""
    dt_tens: float = 0.0     # time frequency for tensor output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngStress:
    """``/STRESS`` or ``/ENG/STRESS`` (M230): Engine stress output tracking directive."""
    id: int = 1
    title: str = ""
    dt_stress: float = 0.0   # time frequency for stress output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngStrain:
    """``/STRAIN`` or ``/ENG/STRAIN`` (M231): Engine strain output tracking directive."""
    id: int = 1
    title: str = ""
    dt_strain: float = 0.0   # time frequency for strain output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngPlastic:
    """``/PLASTIC`` or ``/ENG/PLASTIC`` (M232): Engine plastic strain output tracking directive."""
    id: int = 1
    title: str = ""
    dt_plastic: float = 0.0  # time frequency for plastic strain output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngVelocity:
    """``/VELOCITY`` or ``/ENG/VELOCITY`` (M233): Engine velocity output tracking directive."""
    id: int = 1
    title: str = ""
    dt_vel: float = 0.0      # time frequency for velocity output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngAccel:
    """``/ACCEL`` or ``/ENG/ACCEL`` (M234): Engine acceleration output tracking directive."""
    id: int = 1
    title: str = ""
    dt_acc: float = 0.0      # time frequency for acceleration output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngDisp:
    """``/DISP`` or ``/ENG/DISP`` (M235): Engine displacement output tracking directive."""
    id: int = 1
    title: str = ""
    dt_disp: float = 0.0     # time frequency for displacement output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngRotc:
    """``/ROTC`` or ``/ENG/ROTC`` (M236): Engine rotational displacement output tracking directive."""
    id: int = 1
    title: str = ""
    dt_rotc: float = 0.0     # time frequency for rotational displacement output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngRotv:
    """``/ROTV`` or ``/ENG/ROTV`` (M237): Engine rotational/angular velocity output tracking directive."""
    id: int = 1
    title: str = ""
    dt_rotv: float = 0.0     # time frequency for rotational velocity output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngRota:
    """``/ROTA`` or ``/ENG/ROTA`` (M238): Engine rotational/angular acceleration output tracking directive."""
    id: int = 1
    title: str = ""
    dt_rota: float = 0.0     # time frequency for rotational acceleration output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngForce:
    """``/FORCE`` or ``/ENG/FORCE`` (M239): Engine nodal resultant force output tracking directive."""
    id: int = 1
    title: str = ""
    dt_force: float = 0.0     # time frequency for resultant force output
    sens_id: int = 0          # sensor activation ID


@dataclass
class EngVolume:
    """``/VOLUME`` or ``/ENG/VOLUME`` (M240): Engine element volume output tracking directive."""
    id: int = 1
    title: str = ""
    dt_vol: float = 0.0      # time frequency for element volume output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngDensity:
    """``/DENSITY`` or ``/ENG/DENSITY`` (M241): Engine material density output tracking directive."""
    id: int = 1
    title: str = ""
    dt_dens: float = 0.0     # time frequency for mass density output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngInternalEnergy:
    """``/INTERNAL_ENERGY`` or ``/ENG/INTERNAL_ENERGY`` (M242): Engine internal energy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_ie: float = 0.0       # time frequency for internal energy output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngKineticEnergy:
    """``/KINETIC_ENERGY`` or ``/ENG/KINETIC_ENERGY`` (M243): Engine kinetic energy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_ke: float = 0.0       # time frequency for kinetic energy output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngTotalEnergy:
    """``/TOTAL_ENERGY`` or ``/ENG/TOTAL_ENERGY`` (M244): Engine total energy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_te: float = 0.0       # time frequency for total energy output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngHourglassEnergy:
    """``/HOURGLASS_ENERGY`` or ``/ENG/HOURGLASS_ENERGY``: Engine hourglass energy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_he: float = 0.0       # time frequency for hourglass energy output
    sens_id: int = 0         # sensor activation ID
    dt_hg: float = 0.0       # M271 alias

    def __post_init__(self):
        if self.dt_hg != 0.0 and self.dt_he == 0.0:
            self.dt_he = self.dt_hg
        elif self.dt_he != 0.0 and self.dt_hg == 0.0:
            self.dt_hg = self.dt_he


@dataclass
class EngContactEnergy:
    """``/CONTACT_ENERGY`` or ``/ENG/CONTACT_ENERGY``: Engine contact energy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_ce: float = 0.0       # time frequency for contact energy output
    sens_id: int = 0         # sensor activation ID
    dt_contact: float = 0.0  # M272 alias

    def __post_init__(self):
        if self.dt_contact != 0.0 and self.dt_ce == 0.0:
            self.dt_ce = self.dt_contact
        elif self.dt_ce != 0.0 and self.dt_contact == 0.0:
            self.dt_contact = self.dt_ce


@dataclass
class EngNumericalDissipation:
    """``/NUMERICAL_DISSIPATION`` or ``/ENG/NUMERICAL_DISSIPATION`` (M247): Engine numerical dissipation energy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_num: float = 0.0      # time frequency for numerical dissipation output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngExtWork:
    """``/EXT_WORK`` or ``/ENG/EXT_WORK`` (M248): Engine external work output tracking directive."""
    id: int = 1
    title: str = ""
    dt_wext: float = 0.0     # time frequency for external work output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngTotEnergy:
    """``/TOT_ENERGY`` or ``/ENG/TOT_ENERGY`` (M249): Engine total system energy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_etot: float = 0.0     # time frequency for total energy output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngMassEnergy:
    """``/MASS_ENERGY`` or ``/ENG/MASS_ENERGY`` (M250): Engine added mass kinetic energy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_emass: float = 0.0    # time frequency for mass energy output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngMassChange:
    """``/MASS_CHANGE`` or ``/ENG/MASS_CHANGE`` (M251): Engine added/eroded mass delta variation output tracking directive."""
    id: int = 1
    title: str = ""
    dt_dmass: float = 0.0    # time frequency for mass change output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngPressure:
    """``/PRESSURE`` or ``/ENG/PRESSURE`` (M252): Engine hydrostatic pressure output history tracking directive."""
    id: int = 1
    title: str = ""
    dt_press: float = 0.0    # time frequency for pressure output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngTemperature:
    """``/ENG/TEMPERATURE`` or ``/ENG/TEMP`` (M253): Engine material temperature output tracking directive."""
    id: int = 1
    title: str = ""
    dt_temp: float = 0.0     # time frequency for temperature output
    sens_id: int = 0         # sensor activation ID


@dataclass
class EngEntropy:
    """``/ENG/ENTROPY`` or ``/ENG/THERMAL_ENTROPY`` (M256): Engine thermal entropy output tracking directive."""
    id: int = 1
    title: str = ""
    dt_entr: float = 0.0         # time frequency for entropy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngSoundSpeed:
    """``/ENG/SOUND_SPEED`` or ``/ENG/C_SOUND`` (M257): Engine material sound speed output tracking directive."""
    id: int = 1
    title: str = ""
    dt_sound: float = 0.0        # time frequency for sound speed output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M259 Suite: RTCL failure, EngYieldStress, HarmonicDrive, SensorSpringYield
# ============================================================================

# FailRtcl: canonical definition is above (M125 section) with M259 fields merged in.



@dataclass
class EngYieldStress:
    """``/ENG/YIELD_STRESS`` or ``/ENG/YIELD`` (M259): Engine material yield stress output tracking directive."""
    id: int = 1
    title: str = ""
    dt_yield: float = 0.0        # time frequency for yield stress output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M260 Suite: Sahraei failure, EngPlasticWork, CycloidalDrive, SensorSpringPlasticWork
# ============================================================================




@dataclass
class EngPlasticWork:
    """``/ENG/PLASTIC_WORK`` or ``/ENG/WPLAS`` (M260): Engine plastic work dissipation output tracking directive."""
    id: int = 1
    title: str = ""
    dt_wplas: float = 0.0        # time frequency for plastic work output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M262 Suite: Puck failure, EngStressTri, ScrewJoint, SensorSpringForceImpulse
# ============================================================================

# FailPuck: canonical definition is above (M126 section).



@dataclass
class EngStressTri:
    """``/ENG/STRESS_TRI`` or ``/ENG/TRIAXIALITY`` (M262): Engine stress triaxiality history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_triax: float = 0.0        # time frequency for stress triaxiality output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M263 Suite: Gurson failure, EngLodeAngle, DifferentialGear, SensorSpringMomentRate
# ============================================================================

# FailGurson: canonical definition is above (M125 section) with M263 i_loc alias merged in.



@dataclass
class EngLodeAngle:
    """``/ENG/LODE_ANGLE`` or ``/ENG/LODE`` (M263): Engine normalized Lode angle parameter history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_lode: float = 0.0         # time frequency for Lode angle parameter output
    sens_id: int = 0             # sensor activation ID



@dataclass
class EngMaxShear:
    """``/ENG/MAX_SHEAR`` or ``/ENG/TMAX`` (M264): Engine maximum shear stress history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_tmax: float = 0.0         # time frequency for max shear stress output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngEffectiveStress:
    """``/ENG/EFFECTIVE_STRESS`` or ``/ENG/SIG_EFF`` (M265): Engine von Mises equivalent / effective stress history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_sigeff: float = 0.0       # time frequency for effective stress output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngHydrostaticPressure:
    """``/ENG/HYDROSTATIC_PRESSURE`` or ``/ENG/PHYD`` (M266): Engine hydrostatic pressure field history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_phyd: float = 0.0         # time frequency for hydrostatic pressure output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngOctahedralShear:
    """``/ENG/OCTAHEDRAL_SHEAR`` or ``/ENG/OCT_SHEAR`` (M267): Engine octahedral shear stress history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_toct: float = 0.0         # time frequency for octahedral shear stress output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngDeviatoricEnergy:
    """``/ENG/DEVIATORIC_ENERGY`` or ``/ENG/DEV_ENERGY`` (M268): Engine deviatoric strain energy history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_wdev: float = 0.0         # time frequency for deviatoric strain energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngStrainRate:
    """``/ENG/STRAIN_RATE`` or ``/ENG/EPSDOT`` (M269): Engine strain rate history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_epsdot: float = 0.0      # time frequency for strain rate output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngBulkViscosity:
    """``/ENG/BULK_VISCOSITY`` or ``/ENG/Q_VISC`` (M270): Engine bulk viscosity energy history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_qvisc: float = 0.0       # time frequency for bulk viscosity energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngSpringEnergy:
    """``/ENG/SPRING_ENERGY`` or ``/ENG/SPR_ENERGY`` (M273): Engine spring energy history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_spring: float = 0.0     # time frequency for spring energy output
    sens_id: int = 0            # sensor activation ID


@dataclass
class EngRwallEnergy:
    """``/ENG/RWALL_ENERGY`` or ``/ENG/RWALL_WORK`` (M275): Engine rigid wall energy history tracking output directive."""
    id: int = 1
    title: str = ""
    dt_rwall: float = 0.0      # time frequency for rigid wall energy output
    sens_id: int = 0            # sensor activation ID


@dataclass
class EngSurfEnergy:
    """``/ENG/SURF_ENERGY`` or ``/ENG/SURF_WORK`` (M276): Engine surface boundary pressure / traction work tracking output directive."""
    id: int = 1
    title: str = ""
    dt_surf: float = 0.0       # time frequency for surface energy output
    sens_id: int = 0            # sensor activation ID


@dataclass
class EngHeatExchange:
    """``/ENG/HEAT_EXCHANGE`` or ``/ENG/HEAT_ENERGY`` (M277): Engine heat exchange / thermal dissipation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_heat: float = 0.0         # time frequency for heat exchange energy output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M278 Suite: Tab1 failure, EngSphEnergy, ClevisJoint, SensorSpringTotalWork
# ============================================================================

@dataclass
class EngSphEnergy:
    """``/ENG/SPH_ENERGY`` or ``/ENG/SPH_WORK`` (M278): Engine SPH particle internal/work energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_sph: float = 0.0          # time frequency for SPH energy output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M279 Suite: TButcher failure, EngAleEnergy, PinInSlotJoint, SensorSpringRotationalWork
# ============================================================================

@dataclass
class EngAleEnergy:
    """``/ENG/ALE_ENERGY`` or ``/ENG/ALE_WORK`` (M279): Engine ALE fluid-structure grid work / advection energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ale: float = 0.0          # time frequency for ALE energy output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M280 Suite: Mullins failure, EngFsiEnergy, SliderSlotJoint, SensorSpringTranslationalWork
# ============================================================================

@dataclass
class EngFsiEnergy:
    """``/ENG/FSI_ENERGY`` or ``/ENG/FSI_WORK`` (M280): Engine FSI interface work and energy transfer tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fsi: float = 0.0          # time frequency for FSI energy output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M281 Suite: Cockcroft failure, EngXfemEnergy, ParallelAxisJoint, SensorSpringShearWork
# ============================================================================

@dataclass
class EngXfemEnergy:
    """``/ENG/XFEM_ENERGY`` or ``/ENG/XFEM_WORK`` (M281): Engine XFEM crack propagation and cohesive zone work tracking output directive."""
    id: int = 1
    title: str = ""
    dt_xfem: float = 0.0         # time frequency for XFEM energy output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M282 Suite: Gene1 failure, EngHelmholtzEnergy, CardanJoint, SensorSpringNormalWork
# ============================================================================

@dataclass
class EngHelmholtzEnergy:
    """``/ENG/HELMHOLTZ_ENERGY`` or ``/ENG/HELMHOLTZ_WORK`` (M282): Engine Helmholtz free energy and thermodynamic potential tracking output directive."""
    id: int = 1
    title: str = ""
    dt_helm: float = 0.0         # time frequency for Helmholtz energy output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M283 Suite: Inievo failure, EngEntropyProduction, ScrewNutJoint, SensorSpringTotalForce
# ============================================================================

@dataclass
class EngEntropyProduction:
    """``/ENG/ENTROPY_PRODUCTION`` or ``/ENG/ENTROPY_PROD`` (M283): Engine irreversible entropy generation and dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_entropy: float = 0.0      # time frequency for entropy output
    sens_id: int = 0             # sensor activation ID


# ============================================================================
# M284 Suite: LadDama failure, EngInternalPressure, GenevaJoint, SensorSpringTotalMoment
# ============================================================================

@dataclass
class EngInternalPressure:
    """``/ENG/INTERNAL_PRESSURE`` or ``/ENG/INT_PRESSURE`` (M284): Engine cavity internal gas/fluid pressure tracking output directive."""
    id: int = 1
    title: str = ""
    dt_pres: float = 0.0         # time frequency for internal pressure output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngCoriolisEnergy:
    """``/ENG/CORIOLIS_ENERGY`` or ``/ENG/CORIOLIS_WORK`` (M285): Engine rotating frame Coriolis inertial force work tracking output directive."""
    id: int = 1
    title: str = ""
    dt_coriolis: float = 0.0     # time frequency for Coriolis energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngMagneticEnergy:
    """``/ENG/MAGNETIC_ENERGY`` or ``/ENG/MAGNETIC_WORK`` (M286): Engine electromagnetic magnetic field energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_mag: float = 0.0          # time frequency for magnetic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPoyntingEnergy:
    """``/ENG/POYNTING_ENERGY`` or ``/ENG/POYNTING_WORK`` (M287): Engine electromagnetic Poynting flux vector and radiated energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_poynting: float = 0.0     # time frequency for Poynting energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngMaxwellStressEnergy:
    """``/ENG/MAXWELL_STRESS_ENERGY`` or ``/ENG/MAXWELL_WORK`` (M288): Engine Maxwell stress tensor mechanical work and electromagnetic field deformation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_maxwell: float = 0.0      # time frequency for Maxwell stress energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngJouleHeatEnergy:
    """``/ENG/JOULE_HEAT_ENERGY`` or ``/ENG/JOULE_HEAT_WORK`` (M289): Engine electromagnetic resistive Joule heating dissipation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_joule: float = 0.0        # time frequency for Joule heating output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngLorentzForceEnergy:
    """``/ENG/LORENTZ_FORCE_ENERGY`` or ``/ENG/LORENTZ_WORK`` (M290): Engine electromagnetic Lorentz force mechanical work and volume force energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_lorentz: float = 0.0      # time frequency for Lorentz force energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPlasmonicEnergy:
    """``/ENG/PLASMONIC_ENERGY`` or ``/ENG/PLASMONIC_WORK`` (M291): Engine surface plasmon polariton and resonant optical coupling dissipation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_plasmon: float = 0.0      # time frequency for plasmonic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngDielectricLossEnergy:
    """``/ENG/DIELECTRIC_LOSS_ENERGY`` or ``/ENG/DIELECTRIC_WORK`` (M292): Engine high-frequency dielectric permittivity loss and polarization dissipation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_dielectric: float = 0.0   # time frequency for dielectric loss energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngMagneticHysteresisEnergy:
    """``/ENG/MAGNETIC_HYSTERESIS_ENERGY`` or ``/ENG/MAG_HYST_WORK`` (M293): Engine ferromagnetic / magnetic hysteresis dissipation and core loss energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_hysteresis: float = 0.0   # time frequency for magnetic hysteresis energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngMagnetostrictionEnergy:
    """``/ENG/MAGNETOSTRICTION_ENERGY`` or ``/ENG/MAG_STRICT_WORK`` (M294): Engine magnetostrictive strain deformation energy and magnetic-mechanical coupling work output directive."""
    id: int = 1
    title: str = ""
    dt_magnetostriction: float = 0.0 # time frequency for magnetostriction energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrocaloricEnergy:
    """``/ENG/ELECTROCALORIC_ENERGY`` or ``/ENG/EC_WORK`` (M295): Engine electrocaloric reversible adiabatic thermal entropy change and polarization coupling energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_electrocaloric: float = 0.0 # time frequency for electrocaloric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngMagnetocaloricEnergy:
    """``/ENG/MAGNETOCALORIC_ENERGY`` or ``/ENG/MC_WORK`` (M296): Engine magnetocaloric reversible adiabatic temperature/magnetic entropy coupling energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_magnetocaloric: float = 0.0 # time frequency for magnetocaloric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermoelectricEnergy:
    """``/ENG/THERMOELECTRIC_ENERGY`` or ``/ENG/TE_WORK`` (M297): Engine thermoelectric Seebeck/Peltier reversible thermal-electric energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_thermoelectric: float = 0.0 # time frequency for thermoelectric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPyroelectricEnergy:
    """``/ENG/PYROELECTRIC_ENERGY`` or ``/ENG/PYRO_WORK`` (M298): Engine pyroelectric thermal-polarization coupling and reversible temperature-induced electric energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_pyroelectric: float = 0.0 # time frequency for pyroelectric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermomagneticEnergy:
    """``/ENG/THERMOMAGNETIC_ENERGY`` or ``/ENG/TM_WORK`` (M299): Engine thermomagnetic Nernst/Ettingshausen reversible thermal-magnetic energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_thermomagnetic: float = 0.0 # time frequency for thermomagnetic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermogalvanicEnergy:
    """``/ENG/THERMOGALVANIC_ENERGY`` or ``/ENG/TG_WORK`` (M300): Engine thermogalvanic electrochemical non-isothermal cell and temperature-induced redox reaction energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_thermogalvanic: float = 0.0 # time frequency for thermogalvanic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermionicEnergy:
    """``/ENG/THERMIONIC_ENERGY`` or ``/ENG/TI_WORK`` (M301): Engine thermionic emission electron thermal-field work and thermal-to-electric conversion energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_thermionic: float = 0.0   # time frequency for thermionic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermophotonicEnergy:
    """``/ENG/THERMOPHOTONIC_ENERGY`` or ``/ENG/TP_WORK`` (M302): Engine thermophotonic luminescence radiation and radiative thermal-to-electric photon conversion energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_thermophotonic: float = 0.0 # time frequency for thermophotonic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrostrictiveEnergy:
    """``/ENG/ELECTROSTRICTIVE_ENERGY`` or ``/ENG/ES_WORK`` (M303): Engine electrostrictive non-linear quadratic electric polarization strain work and electrostrictive deformation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_electrostrictive: float = 0.0 # time frequency for electrostrictive energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPhotomagneticEnergy:
    """``/ENG/PHOTOMAGNETIC_ENERGY`` or ``/ENG/PM_WORK`` (M304): Engine photomagnetic magneto-optical resonant absorption and photon-spin polarization energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_photomagnetic: float = 0.0 # time frequency for photomagnetic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermophotonicEmissionEnergy:
    """``/ENG/THERMOPHOTONIC_EMISSION_ENERGY`` or ``/ENG/TPE_WORK`` (M305): Engine thermophotonic non-equilibrium electroluminescent photon emission and optical extraction energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_tpe: float = 0.0          # time frequency for thermophotonic emission energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPhononPolaritonEnergy:
    """``/ENG/PHONON_POLARITON_ENERGY`` or ``/ENG/PP_WORK`` (M306): Engine surface phonon-polariton coupled infrared vibrational-electromagnetic resonance energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_pp: float = 0.0           # time frequency for phonon-polariton energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngExcitonPolaritonEnergy:
    """``/ENG/EXCITON_POLARITON_ENERGY`` or ``/ENG/EP_WORK`` (M307): Engine quantum exciton-polariton cavity coupled light-matter hybridization and condensation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ep: float = 0.0           # time frequency for exciton-polariton energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngMagnonPolaritonEnergy:
    """``/ENG/MAGNON_POLARITON_ENERGY`` or ``/ENG/MP_WORK`` (M308): Engine quantum magnon-polariton coupled magnetic spin-wave electromagnetic cavity hybridization energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_mp: float = 0.0           # time frequency for magnon-polariton energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPiezomagneticEnergy:
    """``/ENG/PIEZOMAGNETIC_ENERGY`` or ``/ENG/PZM_WORK`` (M309): Engine linear piezomagnetic magneto-mechanical coupled strain work and piezomagnetic polarization energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_pzm: float = 0.0          # time frequency for piezomagnetic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngBarocaloricEnergy:
    """``/ENG/BAROCALORIC_ENERGY`` or ``/ENG/BCE_WORK`` (M310): Engine barocaloric pressure-induced thermal entropy change and reversible solid-state elastocaloric/barocaloric phase transformation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_bce: float = 0.0          # time frequency for barocaloric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermomagnetoelectricEnergy:
    """``/ENG/THERMOMAGNETOELECTRIC_ENERGY`` or ``/ENG/TME_WORK`` (M311): Engine coupled thermomagnetoelectric multiferroic resonant energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_tme: float = 0.0          # time frequency for thermomagnetoelectric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrohydrodynamicEnergy:
    """``/ENG/ELECTROHYDRODYNAMIC_ENERGY`` or ``/ENG/EHD_WORK`` (M312): Engine electrohydrodynamic dielectric fluid pumping and space-charge coulombic body force transport energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ehd: float = 0.0          # time frequency for electrohydrodynamic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngMagnetogalvanicEnergy:
    """``/ENG/MAGNETOGALVANIC_ENERGY`` or ``/ENG/MGE_WORK`` (M313): Engine magnetogalvanic electromagnetic induction and lorentz force charge separation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_mge: float = 0.0          # time frequency for magnetogalvanic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElastocaloricEnergy:
    """``/ENG/ELASTOCALORIC_ENERGY`` or ``/ENG/ELC_WORK`` (M314): Engine elastocaloric stress-induced martensitic entropy change and reversible solid-state superelastic heating/cooling energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_elc: float = 0.0          # time frequency for elastocaloric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermophononicEnergy:
    """``/ENG/THERMOPHONONIC_ENERGY`` or ``/ENG/TPH_WORK`` (M315): Engine thermophononic lattice vibrational heat transport and phonon-scattering thermoelectric energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_tph: float = 0.0          # time frequency for thermophononic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermoplasmonicEnergy:
    """``/ENG/THERMOPLASMONIC_ENERGY`` or ``/ENG/TPL_WORK`` (M316): Engine thermoplasmonic resonant metallic nanoparticle Joule dissipation and photothermal conversion energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_tpl: float = 0.0          # time frequency for thermoplasmonic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermomagneticGeneratorEnergy:
    """``/ENG/THERMOMAGNETIC_GENERATOR_ENERGY`` or ``/ENG/TMG_WORK`` (M317): Engine thermomagnetic generator Curie-temperature phase-transition magnetization energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_tmg: float = 0.0          # time frequency for thermomagnetic generator energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngMagnetorheologicalEnergy:
    """``/ENG/MAGNETORHEOLOGICAL_ENERGY`` or ``/ENG/MR_WORK`` (M318): Engine magnetorheological fluid yield stress activation and magnetic field controllable shear dissipation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_mr: float = 0.0           # time frequency for magnetorheological energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrorheologicalEnergy:
    """``/ENG/ELECTRORHEOLOGICAL_ENERGY`` or ``/ENG/ER_WORK`` (M319): Engine electrorheological fluid electric-field induced fibrillated chain polarization and controllable shear yield dissipation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_er: float = 0.0           # time frequency for electrorheological energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermoacousticEnergy:
    """``/ENG/THERMOACOUSTIC_ENERGY`` or ``/ENG/TA_WORK`` (M320): Engine thermoacoustic coupled acoustic wave resonance and oscillating thermal gradient heat pumping energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ta: float = 0.0           # time frequency for thermoacoustic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFerroelectricEnergy:
    """``/ENG/FERROELECTRIC_ENERGY`` or ``/ENG/FE_WORK`` (M321): Engine ferroelectric polarization switching, domain wall motion hysteresis, and electromechanical coupling dissipation energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fe: float = 0.0           # time frequency for ferroelectric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexoelectricEnergy:
    """``/ENG/FLEXOELECTRIC_ENERGY`` or ``/ENG/FLEXO_WORK`` (M322): Engine flexoelectric strain-gradient induced electric polarization and nanoscale electromechanical energy tracking output directive."""
    id: int = 1
    title: str = ""
    dt_flx: float = 0.0          # time frequency for flexoelectric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPyromagneticEnergy:
    """``/ENG/PYROMAGNETIC_ENERGY`` or ``/ENG/PYROMAG_WORK`` (M323): Engine pyromagnetic temperature-dependent magnetization change, thermomagnetic entropy flux, and magnetic pyroelectric energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_pmg: float = 0.0          # time frequency for pyromagnetic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPiezothermalEnergy:
    """``/ENG/PIEZOTHERMAL_ENERGY`` or ``/ENG/PIEZO_THERM_WORK`` (M324): Engine piezothermal coupled stress-temperature pyro-piezoelectric energy conversion and thermomechanical dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_pzt: float = 0.0          # time frequency for piezothermal energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermoflexoelectricEnergy:
    """``/ENG/THERMOFLEXOELECTRIC_ENERGY`` or ``/ENG/THERMOFLEXO_WORK`` (M325): Engine coupled temperature-gradient and strain-gradient induced electric polarization and nanoscale coupled thermal-flexoelectric energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_tfe: float = 0.0          # time frequency for thermoflexoelectric energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagneticEnergy:
    """``/ENG/FLEXOMAGNETIC_ENERGY`` or ``/ENG/FLEXOMAG_WORK`` (M326): Engine flexomagnetic strain gradient-induced magnetic polarization and nanoscale coupled mechanical-magnetic energy conversion tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fm: float = 0.0           # time frequency for flexomagnetic energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngPyroelectricResonanceEnergy:
    """``/ENG/PYROELECTRIC_RESONANCE_ENERGY`` or ``/ENG/PYRO_RES_WORK`` (M327): Engine pyroelectric acoustic resonance energy and high-frequency thermal-dielectric polarization dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_pyr: float = 0.0          # time frequency for pyroelectric resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngThermomagneticResonanceEnergy:
    """``/ENG/THERMOMAGNETIC_RESONANCE_ENERGY`` or ``/ENG/THERMOMAG_RES_WORK`` (M328): Engine thermomagnetic acoustic resonance energy and high-frequency coupled magneto-caloric oscillation dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_tmr: float = 0.0          # time frequency for thermomagnetic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermalResonanceEnergy:
    """``/ENG/FLEXOTHERMAL_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_RES_WORK`` (M329): Engine flexothermal acoustic resonance energy and nanoscale strain-gradient coupled thermoelastic resonance dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftr: float = 0.0          # time frequency for flexothermal resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectromagnetomechanicalResonanceEnergy:
    """``/ENG/ELECTROMAGNETOMECHANICAL_RESONANCE_ENERGY`` or ``/ENG/EMM_RES_WORK`` (M330): Engine multi-field coupled electromagnetomechanical acoustic resonance energy and high-frequency wave-matter polarization dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_emmr: float = 0.0         # time frequency for electromagnetomechanical resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoelectricResonanceEnergy:
    """``/ENG/FLEXOMAGNETOELECTRIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAG_ELEC_RES_WORK`` (M331): Engine coupled flexomagnetic-flexoelectric acoustic resonance energy and nanoscale strain-gradient electromagnetic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmer: float = 0.0         # time frequency for flexomagnetoelectric resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermomagneticResonanceEnergy:
    """``/ENG/FLEXOTHERMOMAGNETIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAG_RES_WORK`` (M332): Engine coupled flexomagnetic-flexothermal acoustic resonance energy and nanoscale strain-gradient magneto-caloric conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftmr: float = 0.0         # time frequency for flexothermomagnetic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoelectricResonanceEnergy:
    """``/ENG/FLEXOTHERMOELECTRIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_ELEC_RES_WORK`` (M333): Engine coupled flexoelectric-flexothermal acoustic resonance energy and nanoscale strain-gradient thermoelectric conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fter: float = 0.0         # time frequency for flexothermoelectric resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoacousticResonanceEnergy:
    """``/ENG/FLEXOTHERMOACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_AC_RES_WORK`` (M334): Engine coupled flexoacoustic-flexothermal acoustic resonance energy and nanoscale strain-gradient thermoacoustic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftar: float = 0.0         # time frequency for flexothermoacoustic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexoelectromagneticResonanceEnergy:
    """``/ENG/FLEXOELECTROMAGNETIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOELEC_MAG_RES_WORK`` (M335): Engine coupled flexoelectric-flexomagnetic full electromagnetic acoustic resonance energy and nanoscale strain-gradient electromagnetic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_femr: float = 0.0         # time frequency for flexoelectromagnetic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexoelectroacousticResonanceEnergy:
    """``/ENG/FLEXOELECTROACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOELEC_AC_RES_WORK`` (M336): Engine coupled flexoelectric-flexoacoustic acoustic resonance energy and nanoscale strain-gradient electroacoustic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fear: float = 0.0         # time frequency for flexoelectroacoustic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoacousticResonanceEnergy:
    """``/ENG/FLEXOMAGNETOACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAG_AC_RES_WORK`` (M337): Engine coupled flexomagnetic-flexoacoustic acoustic resonance energy and nanoscale strain-gradient magneto-acoustic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmar: float = 0.0         # time frequency for flexomagnetoacoustic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoelectromagneticResonanceEnergy:
    """``/ENG/FLEXOTHERMOELECTROMAGNETIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EM_RES_WORK`` (M338): Engine coupled flexothermal-flexoelectromagnetic full multi-field acoustic resonance energy and nanoscale strain-gradient thermo-electromagnetic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftemr: float = 0.0        # time frequency for flexothermoelectromagnetic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoelectroacousticResonanceEnergy:
    """``/ENG/FLEXOTHERMOELECTROACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EA_RES_WORK`` (M339): Engine coupled flexothermal-flexoelectroacoustic full thermo-electro-acoustic resonance energy and nanoscale strain-gradient thermo-electroacoustic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftear: float = 0.0        # time frequency for flexothermoelectroacoustic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermomagnetoacousticResonanceEnergy:
    """``/ENG/FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MA_RES_WORK`` (M340): Engine coupled flexothermal-flexomagnetoacoustic full thermo-magneto-acoustic resonance energy and nanoscale strain-gradient thermo-magnetoacoustic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftmar: float = 0.0        # time frequency for flexothermomagnetoacoustic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoelectromagnetoacousticResonanceEnergy:
    """``/ENG/FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EMA_RES_WORK`` (M341): Engine coupled flexothermal-flexoelectro-flexomagneto-flexoacoustic full multi-field acoustic resonance energy and nanoscale strain-gradient thermo-electromagnetic-acoustic conversion dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftemmar: float = 0.0      # time frequency for flexothermoelectromagnetoacoustic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermophotonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPHOTONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHOTON_RES_WORK`` (M342): Engine coupled flexothermal-flexophotonic high-frequency nanoscale optical resonance energy and strain-gradient photon-polariton dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpr: float = 0.0         # time frequency for flexothermophotonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_RES_WORK`` (M343): Engine coupled flexothermal-flexoplasmonic nanoscale surface-plasmon polariton resonance energy and strain-gradient photothermal dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftplr: float = 0.0        # time frequency for flexothermoplasmonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoexcitonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOEXCITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_RES_WORK`` (M344): Engine coupled flexothermal-flexoexcitonic nanoscale exciton-polariton resonance energy and strain-gradient optoelectronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftexr: float = 0.0        # time frequency for flexothermoexcitonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermomagnonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAGNON_RES_WORK`` (M345): Engine coupled flexothermal-flexomagnonic nanoscale spin-wave magnon polariton resonance energy and strain-gradient spintronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftmr: float = 0.0         # time frequency for flexothermomagnonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermomagnonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAGNON_POLARITON_RES_WORK`` (M346): Engine coupled flexothermal-flexomagnonic-flexophotonic nanoscale magnon-polariton hybrid resonance energy and strain-gradient spintronic-photonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftmpr: float = 0.0        # time frequency for flexothermomagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_POLARITON_RES_WORK`` (M347): Engine coupled flexothermal-flexoplasmonic-flexophotonic nanoscale surface plasmon polariton resonance energy and strain-gradient electromagnetic-photothermal dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftppr: float = 0.0        # time frequency for flexothermoplasmonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoexcitonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_POLARITON_RES_WORK`` (M348): Engine coupled flexothermal-flexoexcitonic-flexophotonic nanoscale exciton-polariton hybrid resonance energy and strain-gradient optoelectronic-photothermal dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftepr: float = 0.0        # time frequency for flexothermoexcitonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermophononpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_POLARITON_RES_WORK`` (M349): Engine coupled flexothermal-flexophononic-flexophotonic nanoscale lattice phonon-polariton hybrid resonance energy and strain-gradient elastodynamic-photothermal dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftphpr: float = 0.0       # time frequency for flexothermophononpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonphononpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_PHONON_POLARITON_RES_WORK`` (M350): Engine coupled flexothermal-flexoplasmonic-flexophononic nanoscale surface plasmon phonon-polariton hybrid resonance energy and strain-gradient electromagnetic-elastodynamic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpppr: float = 0.0       # time frequency for flexothermoplasmonphononpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonexcitonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_POLARITON_RES_WORK`` (M351): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic nanoscale surface plasmon exciton-polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpepr: float = 0.0       # time frequency for flexothermoplasmonexcitonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonmagnonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_MAGNON_POLARITON_RES_WORK`` (M352): Engine coupled flexothermal-flexoplasmonic-flexomagnonic nanoscale surface plasmon magnon-polariton hybrid resonance energy and strain-gradient electromagnetic-spintronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpmpr: float = 0.0       # time frequency for flexothermoplasmonmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoexcitonphononpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_PHONON_POLARITON_RES_WORK`` (M353): Engine coupled flexothermal-flexoexcitonic-flexophononic nanoscale exciton phonon-polariton hybrid resonance energy and strain-gradient optoelectronic-elastodynamic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fteppr: float = 0.0       # time frequency for flexothermoexcitonphononpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoexcitonmagnonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_MAGNON_POLARITON_RES_WORK`` (M354): Engine coupled flexothermal-flexoexcitonic-flexomagnonic nanoscale exciton magnon-polariton hybrid resonance energy and strain-gradient optoelectronic-spintronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftempr: float = 0.0       # time frequency for flexothermoexcitonmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermophononmagnonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_MAGNON_POLARITON_RES_WORK`` (M355): Engine coupled flexothermal-flexophononic-flexomagnonic nanoscale lattice phonon magnon-polariton hybrid resonance energy and strain-gradient elastodynamic-spintronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpmpr: float = 0.0       # time frequency for flexothermophononmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonexcitonphononpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK`` (M356): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexophononic nanoscale plasmon exciton-phonon polariton multi-mode resonance energy and strain-gradient electromagnetic-optoelectronic-elastodynamic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpeppr: float = 0.0      # time frequency for flexothermoplasmonexcitonphononpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonexcitonmagnonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M357): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexomagnonic nanoscale surface plasmon exciton-magnon polariton multi-mode resonance energy and strain-gradient electromagnetic-optoelectronic-spintronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpempr: float = 0.0      # time frequency for flexothermoplasmonexcitonmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonphononmagnonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_PHONON_MAGNON_POLARITON_RES_WORK`` (M358): Engine coupled flexothermal-flexoplasmonic-flexophononic-flexomagnonic nanoscale surface plasmon phonon-magnon polariton multi-mode resonance energy and strain-gradient electromagnetic-elastodynamic-spintronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftppmpr: float = 0.0      # time frequency for flexothermoplasmonphononmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoexcitonphononmagnonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M359): Engine coupled flexothermal-flexoexcitonic-flexophononic-flexomagnonic nanoscale exciton phonon-magnon polariton multi-mode resonance energy and strain-gradient optoelectronic-elastodynamic-spintronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftepmpr: float = 0.0      # time frequency for flexothermoexcitonphononmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonexcitonphononmagnonpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M360): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexophononic-flexomagnonic nanoscale surface plasmon exciton-phonon-magnon polariton quad-hybrid multi-mode resonance energy and strain-gradient electromagnetic-optoelectronic-elastodynamic-spintronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpepmpr: float = 0.0     # time frequency for flexothermoplasmonexcitonphononmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_RES_WORK`` (M361): Engine coupled flexomagnetic-flexoplasmonic nanoscale surface magnetoplasmon polariton resonance energy and strain-gradient electromagnetic-magnetostatic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpr: float = 0.0         # time frequency for flexomagnetoplasmonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_RES_WORK`` (M362): Engine coupled flexomagnetic-flexophononic nanoscale acoustic phonon-magnon polariton resonance energy and strain-gradient magnetoelastic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpr: float = 0.0         # time frequency for flexomagnetophononic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoexcitonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOEXCITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_EXCITON_RES_WORK`` (M363): Engine coupled flexomagnetic-flexoexcitonic nanoscale exciton-magnon polariton resonance energy and strain-gradient optomagnetic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmer: float = 0.0         # time frequency for flexomagnetoexcitonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetopolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_POLARITON_RES_WORK`` (M364): Engine coupled flexomagnetic-flexopolaritonic nanoscale multi-mode magnon-polariton resonance energy and strain-gradient electromagnetic-magnetodynamic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpor: float = 0.0        # time frequency for flexomagnetopolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicphononResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICPHONON_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_PHONON_RES_WORK`` (M365): Engine coupled flexomagnetic-flexoplasmonic-flexophononic nanoscale surface magnetoplasmon-acoustic phonon polariton hybrid resonance energy and strain-gradient electromagnetic-elastodynamic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmppr: float = 0.0        # time frequency for flexomagnetoplasmonicphonon resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicexcitonResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_RES_WORK`` (M366): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic nanoscale surface magnetoplasmon-exciton polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmper: float = 0.0        # time frequency for flexomagnetoplasmonicexciton resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicmagnonResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_MAGNON_RES_WORK`` (M367): Engine coupled flexomagnetic-flexoplasmonic-flexomagnonic nanoscale surface magnetoplasmon-magnon polariton hybrid resonance energy and strain-gradient electromagnetic-magnetodynamic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpmr: float = 0.0        # time frequency for flexomagnetoplasmonicmagnon resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_POLARITON_RES_WORK`` (M368): Engine coupled flexomagnetic-flexoplasmonic-flexopolaritonic nanoscale surface magnetoplasmon-polariton hybrid resonance energy and strain-gradient electromagnetic-magnetodynamic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmppor: float = 0.0       # time frequency for flexomagnetoplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicexcitonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_EXCITON_RES_WORK`` (M369): Engine coupled flexomagnetic-flexophononic-flexoexcitonic nanoscale acoustic phonon exciton-magnon polariton hybrid resonance energy and strain-gradient optoacoustic-magnetoelastic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmper: float = 0.0        # time frequency for flexomagnetophononicexcitonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicmagnonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_MAGNON_RES_WORK`` (M370): Engine coupled flexomagnetic-flexophononic-flexomagnonic nanoscale acoustic phonon magnon-polariton hybrid resonance energy and strain-gradient elastomagnetic-magnetoelastic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpmr: float = 0.0        # time frequency for flexomagnetophononicmagnonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_POLARITON_RES_WORK`` (M371): Engine coupled flexomagnetic-flexophononic-flexopolaritonic nanoscale acoustic phonon polariton-magnon hybrid resonance energy and strain-gradient electromagnetic-optoacoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmppor: float = 0.0       # time frequency for flexomagnetophononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoexcitonicmagnonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_EXCITON_MAGNON_RES_WORK`` (M372): Engine coupled flexomagnetic-flexoexcitonic-flexomagnonic nanoscale exciton-magnon hybrid resonance energy and strain-gradient optomagnetic-magnetoelastic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmemr: float = 0.0        # time frequency for flexomagnetoexcitonicmagnonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoexcitonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_EXCITON_POLARITON_RES_WORK`` (M373): Engine coupled flexomagnetic-flexoexcitonic-flexopolaritonic nanoscale exciton polariton-magnon hybrid resonance energy and strain-gradient electromagnetic-optoelectronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmepor: float = 0.0       # time frequency for flexomagnetoexcitonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_RES_WORK`` (M374): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic nanoscale surface plasmon-exciton-magnon hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-magnetoelastic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpemr: float = 0.0       # time frequency for flexomagnetoplasmonicexcitonicmagnonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_POLARITON_RES_WORK`` (M375): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexopolaritonic nanoscale surface plasmon-exciton-polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-photonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpepr: float = 0.0       # time frequency for flexomagnetoplasmonicexcitonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_MAGNON_POLARITON_RES_WORK`` (M376): Engine coupled flexomagnetic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale surface plasmon-magnon-polariton hybrid resonance energy and strain-gradient electromagnetic-magnetophotonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpmpr: float = 0.0       # time frequency for flexomagnetoplasmonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicexcitonicmagnonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_EXCITON_MAGNON_RES_WORK`` (M377): Engine coupled flexomagnetic-flexophononic-flexoexcitonic-flexomagnonic nanoscale acoustic phonon exciton-magnon hybrid resonance energy and strain-gradient optoacoustic-magnetoelastic-electromagnetic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpemr: float = 0.0       # time frequency for flexomagnetophononicexcitonicmagnonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicexcitonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_EXCITON_POLARITON_RES_WORK`` (M378): Engine coupled flexomagnetic-flexophononic-flexoexcitonic-flexopolaritonic nanoscale acoustic phonon exciton-polariton hybrid resonance energy and strain-gradient optoacoustic-polaritonic-electromagnetic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpepr: float = 0.0       # time frequency for flexomagnetophononicexcitonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_MAGNON_POLARITON_RES_WORK`` (M379): Engine coupled flexomagnetic-flexophononic-flexomagnonic-flexopolaritonic nanoscale acoustic phonon magnon-polariton hybrid resonance energy and strain-gradient optoacoustic-magnetophotonic-electromagnetic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpmpr: float = 0.0       # time frequency for flexomagnetophononicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M380): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale surface plasmon exciton-magnon-polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-magnetophotonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpempr: float = 0.0      # time frequency for flexomagnetoplasmonicexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicplasmonicmagnonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_MAGNON_RES_WORK`` (M381): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexomagnonic nanoscale acoustic phonon surface plasmon-magnon hybrid resonance energy and strain-gradient optoacoustic-electromagnetic-magnetoelastic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpmr: float = 0.0        # time frequency for flexomagnetophononicplasmonicmagnonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_POLARITON_RES_WORK`` (M382): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-polariton hybrid resonance energy and strain-gradient optoacoustic-electromagnetic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpopr: float = 0.0       # time frequency for flexomagnetophononicplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_RES_WORK`` (M383): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic nanoscale surface plasmon exciton-magnon hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-magnetoelastic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpemr: float = 0.0       # time frequency for flexomagnetoplasmonicexcitonicmagnonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_POLARITON_RES_WORK`` (M384): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexopolaritonic nanoscale surface plasmon exciton-polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-photonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpepr: float = 0.0       # time frequency for flexomagnetoplasmonicexcitonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_MAGNON_POLARITON_RES_WORK`` (M385): Engine coupled flexomagnetic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale surface plasmon-magnon-polariton hybrid resonance energy and strain-gradient electromagnetic-magnetophotonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpmpr: float = 0.0       # time frequency for flexomagnetoplasmonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M386): Engine coupled flexomagnetic-flexophononic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale acoustic phonon exciton-magnon-polariton hybrid resonance energy and strain-gradient optoacoustic-magnetophotonic-electromagnetic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpempr: float = 0.0      # time frequency for flexomagnetophononicexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicplasmonicexcitonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_RES_WORK`` (M387): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexoexcitonic nanoscale acoustic phonon surface plasmon-exciton hybrid resonance energy and strain-gradient optoelectronic-electromagnetic-optoacoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmppre: float = 0.0       # time frequency for flexomagnetophononicplasmonicexcitonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_POLARITON_RES_WORK`` (M388): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-polariton hybrid resonance energy and strain-gradient optoacoustic-electromagnetic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmpppr: float = 0.0       # time frequency for flexomagnetophononicplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID

    @property
    def dt_fmpopr(self) -> float:
        return self.dt_fmpppr

    @dt_fmpopr.setter
    def dt_fmpopr(self, val: float) -> None:
        self.dt_fmpppr = val


@dataclass
class EngFlexomagnetophononicplasmonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_MAGNON_POLARITON_RES_WORK`` (M389): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-magnon-polariton hybrid resonance energy and strain-gradient optoacoustic-magnetophotonic-electromagnetic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmppmpr: float = 0.0      # time frequency for flexomagnetophononicplasmonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicplasmonicexcitonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_POLARITON_RES_WORK`` (M390): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexoexcitonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-exciton-polariton hybrid resonance energy and strain-gradient optoacoustic-electromagnetic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmppeopr: float = 0.0     # time frequency for flexomagnetophononicplasmonicexcitonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexomagnetophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M391): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-exciton-magnon-polariton hybrid resonance energy and strain-gradient optoacoustic-magnetophotonic-electromagnetic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_fmppempr: float = 0.0     # time frequency for flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M392): Engine coupled flexothermal-flexophononic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale phonon-plasmon-exciton-magnon-polariton multiphysics hybrid resonance energy and strain-gradient thermoelectric-optomagnetic-photonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftppempr: float = 0.0     # time frequency for flexothermophononicplasmonicexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M393): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton hybrid resonance energy and thermal-strain gradient optomagnetic-photonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpempr: float = 0.0      # time frequency for flexothermoplasmonicexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermophononicexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M394): Engine coupled flexothermal-flexophononic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale phonon-exciton-magnon-polariton hybrid resonance energy and thermal-strain gradient optomagnetic-photonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpempr: float = 0.0      # time frequency for flexothermophononicexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID



@dataclass
class EngFlexothermoplasmonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_MAGNON_POLARITON_RES_WORK`` (M395): Engine coupled flexothermal-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale surface plasmon-magnon-polariton hybrid resonance energy and thermal-gradient optomagnetic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpmp: float = 0.0        # time frequency for flexothermoplasmonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID



@dataclass
class EngFlexothermoplasmonicexcitonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_POLARITON_RES_WORK`` (M396): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexopolaritonic nanoscale surface plasmon-exciton-polariton hybrid resonance energy and thermal-strain gradient optophotonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpep: float = 0.0        # time frequency for flexothermoplasmonicexcitonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID



@dataclass
class EngFlexothermophononicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_MAGNON_POLARITON_RES_WORK`` (M397): Engine coupled flexothermal-flexophononic-flexomagnonic-flexopolaritonic nanoscale phonon-magnon-polariton hybrid resonance energy and thermal-gradient optomagnetic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpmp: float = 0.0        # time frequency for flexothermophononicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_POLARITON_RES_WORK`` (M398): Engine coupled flexothermal-flexoplasmonic-flexopolaritonic nanoscale surface plasmon-polariton hybrid resonance energy and thermal-gradient optophotonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpp: float = 0.0         # time frequency for flexothermoplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermophononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_POLARITON_RES_WORK`` (M399): Engine coupled flexothermal-flexophononic-flexopolaritonic nanoscale surface phonon-polariton hybrid resonance energy and thermal-gradient photonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpp_ph: float = 0.0      # time frequency for flexothermophononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoexcitonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_POLARITON_RES_WORK`` (M400): Engine coupled flexothermal-flexoexcitonic-flexopolaritonic nanoscale exciton-polariton hybrid resonance energy and thermal-gradient photonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftep: float = 0.0         # time frequency for flexothermoexcitonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermomagnonicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAGNON_POLARITON_RES_WORK`` (M401): Engine coupled flexothermal-flexomagnonic-flexopolaritonic nanoscale spin-wave magnon-polariton hybrid resonance energy and thermal-gradient optomagnetic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftmp: float = 0.0         # time frequency for flexothermomagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonicphononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_PHONON_POLARITON_RES_WORK`` (M402): Engine coupled flexothermal-flexoplasmonic-flexophononic-flexopolaritonic nanoscale surface plasmon-phonon-polariton hybrid resonance energy and thermal-gradient optoelectronic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftppp: float = 0.0        # time frequency for flexothermoplasmonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoexcitonicphononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_PHONON_POLARITON_RES_WORK`` (M403): Engine coupled flexothermal-flexoexcitonic-flexophononic-flexopolaritonic nanoscale exciton-phonon-polariton hybrid resonance energy and thermal-gradient optomechanical dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftepp: float = 0.0        # time frequency for flexothermoexcitonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermomagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAGNON_PHONON_POLARITON_RES_WORK`` (M404): Engine coupled flexothermal-flexomagnonic-flexophononic-flexopolaritonic nanoscale spin-wave magnon-phonon-polariton hybrid resonance energy and thermal-gradient spintronic-acousto-optic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftmpp: float = 0.0        # time frequency for flexothermomagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonicexcitonicphononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK`` (M405): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexophononic-flexopolaritonic nanoscale multi-quasiparticle plasmon-exciton-phonon-polariton hybrid resonance energy and thermal-gradient optoelectronic-quantum dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpepp: float = 0.0       # time frequency for flexothermoplasmonicexcitonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonicmagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK`` (M406): Engine coupled flexothermal-flexoplasmonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale surface plasmon-magnon-phonon-polariton hybrid resonance energy and thermal-gradient spintronic-acousto-optic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpmpp: float = 0.0       # time frequency for flexothermoplasmonicmagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoexcitonicmagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK`` (M407): Engine coupled flexothermal-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale exciton-magnon-phonon-polariton hybrid resonance energy and thermal-gradient optoelectronic-spintronic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftempp: float = 0.0       # time frequency for flexothermoexcitonicmagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngFlexothermoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMONIC_EXCITONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK`` (M408): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale 4-quasiparticle plasmon-exciton-magnon-phonon-polariton hybrid resonance energy and thermal-gradient spintronic-optoelectronic-photonic-polaritonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_ftpempp: float = 0.0      # time frequency for flexothermoplasmonicexcitonicmagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexoplasmonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_PLASMON_PHONON_POLARITON_RES_WORK`` (M409): Engine coupled electrothermal-flexoplasmonic-flexophononic-flexopolaritonic nanoscale surface plasmon-phonon-polariton hybrid resonance energy and multi-field dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfppp: float = 0.0       # time frequency for electrothermoflexoplasmonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexoexcitonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_EXCITON_PHONON_POLARITON_RES_WORK`` (M410): Engine coupled electrothermal-flexoexcitonic-flexophononic-flexopolaritonic nanoscale exciton-phonon-polariton hybrid resonance energy and optomechanical dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfepp: float = 0.0       # time frequency for electrothermoflexoexcitonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAGNON_PHONON_POLARITON_RES_WORK`` (M411): Engine coupled electrothermal-flexomagnonic-flexophononic-flexopolaritonic nanoscale magnon-phonon-polariton hybrid resonance energy and spintronic-acousto-optic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmpp: float = 0.0       # time frequency for electrothermoflexomagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexoplasmonicexcitonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK`` (M412): Engine coupled electrothermal-flexoplasmonic-flexoexcitonic-flexophononic-flexopolaritonic nanoscale plasmon-exciton-phonon-polariton hybrid resonance energy and opto-electro-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfpepp: float = 0.0      # time frequency for electrothermoflexoplasmonicexcitonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexoplasmonicmagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_PLASMON_MAGNON_PHONON_POLARITON_RES_WORK`` (M413): Engine coupled electrothermal-flexoplasmonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-magnon-phonon-polariton hybrid resonance energy and spintronic-plasmonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfpmpp: float = 0.0      # time frequency for electrothermoflexoplasmonicmagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_PLASMON_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK`` (M414): Engine coupled electrothermal-flexoplasmonic-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-exciton-magnon-phonon-polariton quintuple-hybrid resonance energy and multi-field opto-spintronic-plasmonic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfpempp: float = 0.0     # time frequency for electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_POLARITON_RES_WORK`` (M415): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexophononic-flexopolaritonic nanoscale plasmon-phonon-polariton hybrid resonance energy and multi-field opto-magneto-electro-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmppp: float = 0.0      # time frequency for electrothermoflexomagnetoplasmonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoexcitonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PHONON_POLARITON_RES_WORK`` (M416): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexophononic-flexopolaritonic nanoscale exciton-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmeppp: float = 0.0     # time frequency for electrothermoflexomagnetoexcitonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetomagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_PHONON_POLARITON_RES_WORK`` (M417): Engine coupled electrothermal-flexomagnetic-flexomagnonic-flexophononic-flexopolaritonic nanoscale magnon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmmppp: float = 0.0     # time frequency for electrothermoflexomagnetomagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoexcitonicmagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK`` (M418): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale exciton-magnon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmemppp: float = 0.0    # time frequency for electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicexcitonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK`` (M419): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexophononic-flexopolaritonic nanoscale plasmon-exciton-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmpeppp: float = 0.0    # time frequency for electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_PHONON_POLARITON_RES_WORK`` (M420/M426): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-magnon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmpmppp: float = 0.0    # time frequency for electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID
    dt_etfmpxmpp: float = 0.0

    def __post_init__(self):
        if self.dt_etfmpxmpp != 0.0 and self.dt_etfmpmppp == 0.0:
            self.dt_etfmpmppp = self.dt_etfmpxmpp
        elif self.dt_etfmpmppp != 0.0 and self.dt_etfmpxmpp == 0.0:
            self.dt_etfmpxmpp = self.dt_etfmpmppp


@dataclass
class EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK`` (M421): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-exciton-magnon-phonon-polariton heptuple-hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmpxmppp: float = 0.0   # time frequency for electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicmagnonpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_POLARITON_RES_WORK`` (M422): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale plasmon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmpmpp: float = 0.0     # time frequency for electrothermoflexomagnetoplasmonicmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoexcitonicmagnonpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_POLARITON_RES_WORK`` (M423): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmxmpp: float = 0.0     # time frequency for electrothermoflexomagnetoexcitonicmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetophononicmagnonpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_POLARITON_RES_WORK`` (M424): Engine coupled electrothermal-flexomagnetic-flexophononic-flexomagnonic-flexopolaritonic nanoscale phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmphmpp: float = 0.0    # time frequency for electrothermoflexomagnetophononicmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M425): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton 4-hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmpxmpp: float = 0.0    # time frequency for electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicphononicmagnonpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_MAGNON_POLARITON_RES_WORK`` (M427): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale plasmon-phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmppmpp: float = 0.0    # time frequency for electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoexcitonicphononicmagnonpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M428): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale exciton-phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmexpmpp: float = 0.0   # time frequency for electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M429): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmpexpmpp: float = 0.0  # time frequency for electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_POLARITON_RES_WORK`` (M430): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmexmnp: float = 0.0    # time frequency for electrothermoflexomagnetoexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_POLARITON_RES_WORK`` (M431): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale plasmon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfplmmnp: float = 0.0    # time frequency for electrothermoflexomagnetoplasmonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetophononicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_POLARITON_RES_WORK`` (M432): Engine coupled electrothermal-flexomagnetic-flexophononic-flexomagnonic-flexopolaritonic nanoscale phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfphmmnp: float = 0.0    # time frequency for electrothermoflexomagnetophononicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M433): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfplexmmnp: float = 0.0  # time frequency for electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M433): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfplexmmnp: float = 0.0    # time frequency for electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID


@dataclass
class EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_POLARITON_RES_WORK`` (M434): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexophononic-flexopolaritonic nanoscale plasmon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfplphnp: float = 0.0      # time frequency for electrothermoflexomagnetoplasmonicphononicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfmppp: float = 0.0        # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfmppp != 0.0 and self.dt_etfplphnp == 0.0:
            self.dt_etfplphnp = self.dt_etfmppp
        elif self.dt_etfplphnp != 0.0 and self.dt_etfmppp == 0.0:
            self.dt_etfmppp = self.dt_etfplphnp


@dataclass
class EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M435): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale multi-mode hybrid resonance energy and opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfplexphmnp: float = 0.0   # time frequency for electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfpepmp: float = 0.0       # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfpepmp != 0.0 and self.dt_etfplexphmnp == 0.0:
            self.dt_etfplexphmnp = self.dt_etfpepmp
        elif self.dt_etfplexphmnp != 0.0 and self.dt_etfpepmp == 0.0:
            self.dt_etfpepmp = self.dt_etfplexphmnp


@dataclass
class EngElectrothermoflexomagnetophononicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_POLARITON_RES_WORK`` (M436): Engine coupled electrothermal-flexomagnetic-flexophononic-flexopolaritonic nanoscale phonon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfphmnp: float = 0.0       # time frequency for electrothermoflexomagnetophononicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfphp: float = 0.0         # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfphp != 0.0 and self.dt_etfphmnp == 0.0:
            self.dt_etfphmnp = self.dt_etfphp
        elif self.dt_etfphmnp != 0.0 and self.dt_etfphp == 0.0:
            self.dt_etfphp = self.dt_etfphmnp


@dataclass
class EngElectrothermoflexomagnetoexcitonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_POLARITON_RES_WORK`` (M438): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexopolaritonic nanoscale exciton-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfexmnp: float = 0.0       # time frequency for electrothermoflexomagnetoexcitonicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfexp: float = 0.0         # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfexp != 0.0 and self.dt_etfexmnp == 0.0:
            self.dt_etfexmnp = self.dt_etfexp
        elif self.dt_etfexmnp != 0.0 and self.dt_etfexp == 0.0:
            self.dt_etfexp = self.dt_etfexmnp


@dataclass
class EngElectrothermoflexomagnetomagnonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_POLARITON_RES_WORK`` (M439): Engine coupled electrothermal-flexomagnetic-flexomagnonic-flexopolaritonic nanoscale magnon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmgmnp: float = 0.0       # time frequency for electrothermoflexomagnetomagnonicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfplp: float = 0.0         # alias field for plasmonic polaritonic compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfmgmnp == 0.0:
            self.dt_etfmgmnp = self.dt_etfplp
        elif self.dt_etfmgmnp != 0.0 and self.dt_etfplp == 0.0:
            self.dt_etfplp = self.dt_etfmgmnp

EngElectrothermoflexomagnetomagnonicpolaritonResonanceEnergy = EngElectrothermoflexomagnetomagnonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetoplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_POLARITON_RES_WORK`` (M440): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexopolaritonic nanoscale plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfplp: float = 0.0         # time frequency for electrothermoflexomagnetoplasmonicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfmgmnp: float = 0.0       # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfmgmnp != 0.0 and self.dt_etfplp == 0.0:
            self.dt_etfplp = self.dt_etfmgmnp
        elif self.dt_etfplp != 0.0 and self.dt_etfmgmnp == 0.0:
            self.dt_etfmgmnp = self.dt_etfplp


EngElectrothermoflexomagnetoplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetoplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetophononicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_PLASMON_POLARITON_RES_WORK`` (M441): Engine coupled electrothermal-flexomagnetic-flexophononic-flexoplasmonic-flexopolaritonic nanoscale phonon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfphplp: float = 0.0       # time frequency for electrothermoflexomagnetophononicplasmonicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfplp: float = 0.0         # alias field for compatibility
    dt_etfmppp: float = 0.0        # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfphplp == 0.0:
            self.dt_etfphplp = self.dt_etfplp
        elif self.dt_etfmppp != 0.0 and self.dt_etfphplp == 0.0:
            self.dt_etfphplp = self.dt_etfmppp
        elif self.dt_etfphplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfphplp
            if self.dt_etfmppp == 0.0:
                self.dt_etfmppp = self.dt_etfphplp


EngElectrothermoflexomagnetophononplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetophononicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetoexcitonicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PLASMON_POLARITON_RES_WORK`` (M442): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexoplasmonic-flexopolaritonic nanoscale exciton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfexplp: float = 0.0       # time frequency for electrothermoflexomagnetoexcitonicplasmonicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0       # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfexplp == 0.0:
            self.dt_etfexplp = self.dt_etfplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfexplp == 0.0:
            self.dt_etfexplp = self.dt_etfphplp
        elif self.dt_etfexplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfexplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfexplp


EngElectrothermoflexomagnetoexcitonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetoexcitonicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetomagnonicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_PLASMON_POLARITON_RES_WORK`` (M443): Engine coupled electrothermal-flexomagnetic-flexomagnonic-flexoplasmonic-flexopolaritonic nanoscale magnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfmagplp: float = 0.0      # time frequency for electrothermoflexomagnetomagnonicplasmonicpolaritonic resonance energy output
    sens_id: int = 0               # sensor activation ID
    dt_etfplp: float = 0.0         # alias field for compatibility
    dt_etfexplp: float = 0.0       # alias field for compatibility
    dt_etfphplp: float = 0.0       # alias field for compatibility
    dt_etfplmmnp: float = 0.0      # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfmagplp == 0.0:
            self.dt_etfmagplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfmagplp == 0.0:
            self.dt_etfmagplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfmagplp == 0.0:
            self.dt_etfmagplp = self.dt_etfphplp
        elif self.dt_etfplmmnp != 0.0 and self.dt_etfmagplp == 0.0:
            self.dt_etfmagplp = self.dt_etfplmmnp
        elif self.dt_etfmagplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfmagplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfmagplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfmagplp
            if self.dt_etfplmmnp == 0.0:
                self.dt_etfplmmnp = self.dt_etfmagplp


EngElectrothermoflexomagnetomagnonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetomagnonicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetophononicexcitonicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_EXCITON_PLASMON_POLARITON_RES_WORK`` (M444): Engine coupled electrothermal-flexomagnetic-flexophononic-flexoexcitonic-flexoplasmonic-flexopolaritonic nanoscale phonon-exciton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfpexplp: float = 0.0     # time frequency for electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic resonance energy output
    sens_id: int = 0              # sensor activation ID
    dt_etfplp: float = 0.0        # alias field for compatibility
    dt_etfexplp: float = 0.0      # alias field for compatibility
    dt_etfphplp: float = 0.0      # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfpexplp == 0.0:
            self.dt_etfpexplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfpexplp == 0.0:
            self.dt_etfpexplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfpexplp == 0.0:
            self.dt_etfpexplp = self.dt_etfphplp
        elif self.dt_etfpexplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfpexplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfpexplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfpexplp


EngElectrothermoflexomagnetophononexcitonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetophononicexcitonicplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoexcitonicphononicplasmonicpolaritonicResonanceEnergy = EngElectrothermoflexomagnetophononicexcitonicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetophononicmagnonicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_PLASMON_POLARITON_RES_WORK`` (M445): Engine coupled electrothermal-flexomagnetic-flexophononic-flexomagnonic-flexoplasmonic-flexopolaritonic nanoscale phonon-magnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfpmplp: float = 0.0      # time frequency for electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic resonance energy output
    sens_id: int = 0              # sensor activation ID
    dt_etfplp: float = 0.0        # alias field for compatibility
    dt_etfexplp: float = 0.0      # alias field for compatibility
    dt_etfphplp: float = 0.0      # alias field for compatibility
    dt_etfmagplp: float = 0.0     # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfpmplp == 0.0:
            self.dt_etfpmplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfpmplp == 0.0:
            self.dt_etfpmplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfpmplp == 0.0:
            self.dt_etfpmplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfpmplp == 0.0:
            self.dt_etfpmplp = self.dt_etfmagplp
        elif self.dt_etfpmplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfpmplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfpmplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfpmplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfpmplp


EngElectrothermoflexomagnetophononmagnonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetophononicmagnonicplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetomagnonicphononicplasmonicpolaritonicResonanceEnergy = EngElectrothermoflexomagnetophononicmagnonicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_EXCITON_MAGNON_PLASMON_POLARITON_RES_WORK`` (M446): Engine coupled electrothermal-flexomagnetic-flexophononic-flexoexcitonic-flexomagnonic-flexoplasmonic-flexopolaritonic nanoscale phonon-exciton-magnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfpexmplp: float = 0.0    # time frequency for electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic resonance energy output
    sens_id: int = 0              # sensor activation ID
    dt_etfplp: float = 0.0        # alias field for compatibility
    dt_etfexplp: float = 0.0      # alias field for compatibility
    dt_etfphplp: float = 0.0      # alias field for compatibility
    dt_etfmagplp: float = 0.0     # alias field for compatibility
    dt_etfpexplp: float = 0.0     # alias field for compatibility
    dt_etfpmplp: float = 0.0      # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfpexmplp == 0.0:
            self.dt_etfpexmplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfpexmplp == 0.0:
            self.dt_etfpexmplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfpexmplp == 0.0:
            self.dt_etfpexmplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfpexmplp == 0.0:
            self.dt_etfpexmplp = self.dt_etfmagplp
        elif self.dt_etfpexplp != 0.0 and self.dt_etfpexmplp == 0.0:
            self.dt_etfpexmplp = self.dt_etfpexplp
        elif self.dt_etfpmplp != 0.0 and self.dt_etfpexmplp == 0.0:
            self.dt_etfpexmplp = self.dt_etfpmplp
        elif self.dt_etfpexmplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfpexmplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfpexmplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfpexmplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfpexmplp
            if self.dt_etfpexplp == 0.0:
                self.dt_etfpexplp = self.dt_etfpexmplp
            if self.dt_etfpmplp == 0.0:
                self.dt_etfpmplp = self.dt_etfpexmplp


EngElectrothermoflexomagnetophononexcitonmagnonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetomagnonicexcitonicphononicplasmonicpolaritonicResonanceEnergy = EngElectrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PLASMON_POLARITON_RES_WORK`` (M447): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoplasmonic-flexopolaritonic nanoscale chiral-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcplp: float = 0.0      # time frequency for electrothermoflexomagnetochiralplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID
    dt_etfplp: float = 0.0       # alias field for compatibility
    dt_etfexplp: float = 0.0     # alias field for compatibility
    dt_etfphplp: float = 0.0     # alias field for compatibility
    dt_etfmagplp: float = 0.0    # alias field for compatibility
    dt_etfpexmplp: float = 0.0   # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcplp == 0.0:
            self.dt_etfcplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcplp == 0.0:
            self.dt_etfcplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcplp == 0.0:
            self.dt_etfcplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcplp == 0.0:
            self.dt_etfcplp = self.dt_etfmagplp
        elif self.dt_etfpexmplp != 0.0 and self.dt_etfcplp == 0.0:
            self.dt_etfcplp = self.dt_etfpexmplp
        elif self.dt_etfcplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcplp
            if self.dt_etfpexmplp == 0.0:
                self.dt_etfpexmplp = self.dt_etfcplp


EngElectrothermoflexomagnetochiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoplasmonicchiralpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralphononicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PHONON_PLASMON_POLARITON_RES_WORK`` (M448): Engine coupled electrothermal-flexomagnetic-flexochiral-flexophononic-flexoplasmonic-flexopolaritonic nanoscale chiral-phonon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcphplp: float = 0.0    # time frequency for electrothermoflexomagnetochiralphononicplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID
    dt_etfplp: float = 0.0       # alias field for compatibility
    dt_etfexplp: float = 0.0     # alias field for compatibility
    dt_etfphplp: float = 0.0     # alias field for compatibility
    dt_etfmagplp: float = 0.0    # alias field for compatibility
    dt_etfcplp: float = 0.0      # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcphplp == 0.0:
            self.dt_etfcphplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcphplp == 0.0:
            self.dt_etfcphplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcphplp == 0.0:
            self.dt_etfcphplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcphplp == 0.0:
            self.dt_etfcphplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcphplp == 0.0:
            self.dt_etfcphplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcphplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcphplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcphplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcphplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcphplp


EngElectrothermoflexomagnetochiralphononplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralphononicplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetophononicchiralplasmonicpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralphononicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralexcitonicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_EXCITON_PLASMON_POLARITON_RES_WORK`` (M449): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoexcitonic-flexoplasmonic-flexopolaritonic nanoscale chiral-exciton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcexplp: float = 0.0    # time frequency for electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID
    dt_etfplp: float = 0.0       # alias field for compatibility
    dt_etfexplp: float = 0.0     # alias field for compatibility
    dt_etfphplp: float = 0.0     # alias field for compatibility
    dt_etfmagplp: float = 0.0    # alias field for compatibility
    dt_etfcplp: float = 0.0      # alias field for compatibility
    dt_etfcphplp: float = 0.0    # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcexplp == 0.0:
            self.dt_etfcexplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcexplp == 0.0:
            self.dt_etfcexplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcexplp == 0.0:
            self.dt_etfcexplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcexplp == 0.0:
            self.dt_etfcexplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcexplp == 0.0:
            self.dt_etfcexplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcexplp == 0.0:
            self.dt_etfcexplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcexplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcexplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcexplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcexplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcexplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcexplp


EngElectrothermoflexomagnetochiralexcitonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralexcitonicplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoexcitonicchiralplasmonicpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralexcitonicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralmagnonicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MAGNON_PLASMON_POLARITON_RES_WORK`` (M450): Engine coupled electrothermal-flexomagnetic-flexochiral-flexomagnonic-flexoplasmonic-flexopolaritonic nanoscale chiral-magnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcmagplp: float = 0.0   # time frequency for electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID
    dt_etfplp: float = 0.0       # alias field for compatibility
    dt_etfexplp: float = 0.0     # alias field for compatibility
    dt_etfphplp: float = 0.0     # alias field for compatibility
    dt_etfmagplp: float = 0.0    # alias field for compatibility
    dt_etfcplp: float = 0.0      # alias field for compatibility
    dt_etfcphplp: float = 0.0    # alias field for compatibility
    dt_etfcexplp: float = 0.0    # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcmagplp == 0.0:
            self.dt_etfcmagplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcmagplp == 0.0:
            self.dt_etfcmagplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcmagplp == 0.0:
            self.dt_etfcmagplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcmagplp == 0.0:
            self.dt_etfcmagplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcmagplp == 0.0:
            self.dt_etfcmagplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcmagplp == 0.0:
            self.dt_etfcmagplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcmagplp == 0.0:
            self.dt_etfcmagplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcmagplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcmagplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcmagplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcmagplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcmagplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcmagplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcmagplp


EngElectrothermoflexomagnetochiralmagnonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralmagnonicplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetomagnonicchiralplasmonicpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralmagnonicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralspinplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPIN_PLASMON_POLARITON_RES_WORK`` (M451): Engine coupled electrothermal-flexomagnetic-flexochiral-flexospin-flexoplasmonic-flexopolaritonic nanoscale chiral-spin-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcspinplp: float = 0.0  # time frequency for electrothermoflexomagnetochiralspinplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID
    dt_etfplp: float = 0.0       # alias field for compatibility
    dt_etfexplp: float = 0.0     # alias field for compatibility
    dt_etfphplp: float = 0.0     # alias field for compatibility
    dt_etfmagplp: float = 0.0    # alias field for compatibility
    dt_etfcplp: float = 0.0      # alias field for compatibility
    dt_etfcphplp: float = 0.0    # alias field for compatibility
    dt_etfcexplp: float = 0.0    # alias field for compatibility
    dt_etfcmagplp: float = 0.0   # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcspinplp == 0.0:
            self.dt_etfcspinplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcspinplp == 0.0:
            self.dt_etfcspinplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcspinplp == 0.0:
            self.dt_etfcspinplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcspinplp == 0.0:
            self.dt_etfcspinplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcspinplp == 0.0:
            self.dt_etfcspinplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcspinplp == 0.0:
            self.dt_etfcspinplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcspinplp == 0.0:
            self.dt_etfcspinplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcspinplp == 0.0:
            self.dt_etfcspinplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcspinplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcspinplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcspinplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcspinplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcspinplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcspinplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcspinplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcspinplp


EngElectrothermoflexomagnetochiralspinplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralspinplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetospinonchiralplasmonicpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralspinplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralspinonplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPINON_PLASMON_POLARITON_RES_WORK`` (M452): Engine coupled electrothermal-flexomagnetic-flexochiral-flexospinon-flexoplasmonic-flexopolaritonic nanoscale chiral-spinon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcspinonplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralspinonplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID
    dt_etfplp: float = 0.0       # alias field for compatibility
    dt_etfexplp: float = 0.0     # alias field for compatibility
    dt_etfphplp: float = 0.0     # alias field for compatibility
    dt_etfmagplp: float = 0.0    # alias field for compatibility
    dt_etfcplp: float = 0.0      # alias field for compatibility
    dt_etfcphplp: float = 0.0    # alias field for compatibility
    dt_etfcexplp: float = 0.0    # alias field for compatibility
    dt_etfcmagplp: float = 0.0   # alias field for compatibility
    dt_etfcspinplp: float = 0.0  # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcspinonplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcspinonplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcspinonplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcspinonplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcspinonplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcspinonplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcspinonplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcspinonplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcspinonplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcspinonplp


EngElectrothermoflexomagnetochiralspinonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralspinonplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetospinonchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralspinonplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralholonplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HOLON_PLASMON_POLARITON_RES_WORK`` (M453): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoholon-flexoplasmonic-flexopolaritonic nanoscale chiral-holon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcholonplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralholonplasmonicpolaritonic resonance energy output
    sens_id: int = 0             # sensor activation ID
    dt_etfplp: float = 0.0       # alias field for compatibility
    dt_etfexplp: float = 0.0     # alias field for compatibility
    dt_etfphplp: float = 0.0     # alias field for compatibility
    dt_etfmagplp: float = 0.0    # alias field for compatibility
    dt_etfcplp: float = 0.0      # alias field for compatibility
    dt_etfcphplp: float = 0.0    # alias field for compatibility
    dt_etfcexplp: float = 0.0    # alias field for compatibility
    dt_etfcmagplp: float = 0.0   # alias field for compatibility
    dt_etfcspinplp: float = 0.0  # alias field for compatibility
    dt_etfcspinonplp: float = 0.0# alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcspinonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcholonplp == 0.0:
            self.dt_etfcholonplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcholonplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcholonplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcholonplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcholonplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcholonplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcholonplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcholonplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcholonplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcholonplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcholonplp


EngElectrothermoflexomagnetochiralholonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralholonplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoholonchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralholonplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralorbitonplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ORBITON_PLASMON_POLARITON_RES_WORK`` (M454): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoorbiton-flexoplasmonic-flexopolaritonic nanoscale chiral-orbiton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcorbitonplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralorbitonplasmonicpolaritonic resonance energy output
    sens_id: int = 0              # sensor activation ID
    dt_etfplp: float = 0.0        # alias field for compatibility
    dt_etfexplp: float = 0.0      # alias field for compatibility
    dt_etfphplp: float = 0.0      # alias field for compatibility
    dt_etfmagplp: float = 0.0     # alias field for compatibility
    dt_etfcplp: float = 0.0       # alias field for compatibility
    dt_etfcphplp: float = 0.0     # alias field for compatibility
    dt_etfcexplp: float = 0.0     # alias field for compatibility
    dt_etfcmagplp: float = 0.0    # alias field for compatibility
    dt_etfcspinplp: float = 0.0   # alias field for compatibility
    dt_etfcspinonplp: float = 0.0 # alias field for compatibility
    dt_etfcholonplp: float = 0.0  # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcorbitonplp == 0.0:
            self.dt_etfcorbitonplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcorbitonplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcorbitonplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcorbitonplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcorbitonplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcorbitonplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcorbitonplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcorbitonplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcorbitonplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcorbitonplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcorbitonplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcorbitonplp


EngElectrothermoflexomagnetochiralorbitonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralorbitonplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoorbitonchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralorbitonplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralplasmononplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PLASMONON_PLASMON_POLARITON_RES_WORK`` (M455): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoplasmonon-flexoplasmonic-flexopolaritonic nanoscale chiral-plasmonon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcplasmononplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralplasmononplasmonicpolaritonic resonance energy output
    sens_id: int = 0                # sensor activation ID
    dt_etfplp: float = 0.0          # alias field for compatibility
    dt_etfexplp: float = 0.0        # alias field for compatibility
    dt_etfphplp: float = 0.0        # alias field for compatibility
    dt_etfmagplp: float = 0.0       # alias field for compatibility
    dt_etfcplp: float = 0.0         # alias field for compatibility
    dt_etfcphplp: float = 0.0       # alias field for compatibility
    dt_etfcexplp: float = 0.0       # alias field for compatibility
    dt_etfcmagplp: float = 0.0      # alias field for compatibility
    dt_etfcspinplp: float = 0.0     # alias field for compatibility
    dt_etfcspinonplp: float = 0.0   # alias field for compatibility
    dt_etfcholonplp: float = 0.0    # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0  # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcplasmononplp == 0.0:
            self.dt_etfcplasmononplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcplasmononplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcplasmononplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcplasmononplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcplasmononplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcplasmononplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcplasmononplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcplasmononplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcplasmononplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcplasmononplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcplasmononplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcplasmononplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcplasmononplp


EngElectrothermoflexomagnetochiralplasmononplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralplasmononplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoplasmononchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralplasmononplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralparamagnonplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PARAMAGNON_PLASMON_POLARITON_RES_WORK`` (M456): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoparamagnon-flexoplasmonic-flexopolaritonic nanoscale chiral-paramagnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcparamagnonplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic resonance energy output
    sens_id: int = 0                  # sensor activation ID
    dt_etfplp: float = 0.0            # alias field for compatibility
    dt_etfexplp: float = 0.0          # alias field for compatibility
    dt_etfphplp: float = 0.0          # alias field for compatibility
    dt_etfmagplp: float = 0.0         # alias field for compatibility
    dt_etfcplp: float = 0.0           # alias field for compatibility
    dt_etfcphplp: float = 0.0         # alias field for compatibility
    dt_etfcexplp: float = 0.0         # alias field for compatibility
    dt_etfcmagplp: float = 0.0        # alias field for compatibility
    dt_etfcspinplp: float = 0.0       # alias field for compatibility
    dt_etfcspinonplp: float = 0.0     # alias field for compatibility
    dt_etfcholonplp: float = 0.0      # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0    # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0  # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcparamagnonplp == 0.0:
            self.dt_etfcparamagnonplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcparamagnonplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcparamagnonplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcparamagnonplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcparamagnonplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcparamagnonplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcparamagnonplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcparamagnonplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcparamagnonplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcparamagnonplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcparamagnonplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcparamagnonplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcparamagnonplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcparamagnonplp


EngElectrothermoflexomagnetochiralparamagnonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralparamagnonplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoparamagnonchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralparamagnonplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiraldyonicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_DYONIC_PLASMON_POLARITON_RES_WORK`` (M457): Engine coupled electrothermal-flexomagnetic-flexochiral-flexodyonic-flexoplasmonic-flexopolaritonic nanoscale chiral-dyonic-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcdyonicplp: float = 0.0    # time frequency for electrothermoflexomagnetochiraldyonicplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcdyonicplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcdyonicplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcdyonicplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcdyonicplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcdyonicplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcdyonicplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcdyonicplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcdyonicplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcdyonicplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcdyonicplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcdyonicplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcdyonicplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcdyonicplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcdyonicplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcdyonicplp


EngElectrothermoflexomagnetochiraldyonicplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiraldyonicplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetodyonicchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiraldyonicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralaxionicplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_AXIONIC_PLASMON_POLARITON_RES_WORK`` (M458): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoaxionic-flexoplasmonic-flexopolaritonic nanoscale chiral-axionic-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcaxionicplp: float = 0.0   # time frequency for electrothermoflexomagnetochiralaxionicplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcaxionicplp == 0.0:
            self.dt_etfcaxionicplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcaxionicplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcaxionicplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcaxionicplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcaxionicplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcaxionicplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcaxionicplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcaxionicplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcaxionicplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcaxionicplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcaxionicplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcaxionicplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcaxionicplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcaxionicplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcaxionicplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcaxionicplp


EngElectrothermoflexomagnetochiralaxionicplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralaxionicplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoaxionicchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralaxionicplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralmajoranaplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MAJORANA_PLASMON_POLARITON_RES_WORK`` (M459): Engine coupled electrothermal-flexomagnetic-flexochiral-flexomajorana-flexoplasmonic-flexopolaritonic nanoscale chiral-majorana-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcmajoranaplp: float = 0.0  # time frequency for electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcmajoranaplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcmajoranaplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcmajoranaplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcmajoranaplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcmajoranaplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcmajoranaplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcmajoranaplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcmajoranaplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcmajoranaplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcmajoranaplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcmajoranaplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcmajoranaplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcmajoranaplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcmajoranaplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcmajoranaplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcmajoranaplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcmajoranaplp


EngElectrothermoflexomagnetochiralmajoranaplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralmajoranaplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetomajoranachiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralmajoranaplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralanyonplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANYON_PLASMON_POLARITON_RES_WORK`` (M460): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoanyon-flexoplasmonic-flexopolaritonic nanoscale chiral-anyon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcanyonplp: float = 0.0     # time frequency for electrothermoflexomagnetochiralanyonplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcanyonplp == 0.0:
            self.dt_etfcanyonplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcanyonplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcanyonplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcanyonplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcanyonplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcanyonplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcanyonplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcanyonplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcanyonplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcanyonplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcanyonplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcanyonplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcanyonplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcanyonplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcanyonplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcanyonplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcanyonplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcanyonplp


EngElectrothermoflexomagnetochiralanyonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralanyonplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoanyonchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralanyonplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralskyrmionplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SKYRMION_PLASMON_POLARITON_RES_WORK`` (M461): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoskyrmion-flexoplasmonic-flexopolaritonic nanoscale chiral-skyrmion-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcskyrmionplp: float = 0.0  # time frequency for electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcskyrmionplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcskyrmionplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcskyrmionplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcskyrmionplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcskyrmionplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcskyrmionplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcskyrmionplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcskyrmionplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcskyrmionplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcskyrmionplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcskyrmionplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcskyrmionplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcskyrmionplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcskyrmionplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcskyrmionplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcskyrmionplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcskyrmionplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcskyrmionplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcskyrmionplp


EngElectrothermoflexomagnetochiralskyrmionplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralskyrmionplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoskyrmionchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralskyrmionplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralmeronplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MERON_PLASMON_POLARITON_RES_WORK`` (M462): Engine coupled electrothermal-flexomagnetic-flexochiral-flexomeron-flexoplasmonic-flexopolaritonic nanoscale chiral-meron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcmeronplp: float = 0.0     # time frequency for electrothermoflexomagnetochiralmeronplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcmeronplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcmeronplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcmeronplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcmeronplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcmeronplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcmeronplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcmeronplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcmeronplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcmeronplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcmeronplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcmeronplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcmeronplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcmeronplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcmeronplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcmeronplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcmeronplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcmeronplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcmeronplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcmeronplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcmeronplp


EngElectrothermoflexomagnetochiralmeronplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralmeronplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetomeronchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralmeronplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralbimeronplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BIMERON_PLASMON_POLARITON_RES_WORK`` (M463): Engine coupled electrothermal-flexomagnetic-flexochiral-flexobimeron-flexoplasmonic-flexopolaritonic nanoscale chiral-bimeron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcbimeronplp: float = 0.0   # time frequency for electrothermoflexomagnetochiralbimeronplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility
    dt_etfcmeronplp: float = 0.0     # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcbimeronplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcbimeronplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcbimeronplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcbimeronplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcbimeronplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcbimeronplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcbimeronplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcbimeronplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcbimeronplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcbimeronplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcbimeronplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcbimeronplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcbimeronplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcbimeronplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcbimeronplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcbimeronplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcbimeronplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcbimeronplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcbimeronplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcbimeronplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcbimeronplp


EngElectrothermoflexomagnetochiralbimeronplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralbimeronplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetobimeronchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralbimeronplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralinstantonplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_INSTANTON_PLASMON_POLARITON_RES_WORK`` (M464): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoinstanton-flexoplasmonic-flexopolaritonic nanoscale chiral-instanton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcinstantonplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralinstantonplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility
    dt_etfcmeronplp: float = 0.0     # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0   # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcinstantonplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcinstantonplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcinstantonplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcinstantonplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcinstantonplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcinstantonplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcinstantonplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcinstantonplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcinstantonplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcinstantonplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcinstantonplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcinstantonplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcinstantonplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcinstantonplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcinstantonplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcinstantonplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcinstantonplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcinstantonplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcinstantonplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcinstantonplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcinstantonplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcinstantonplp


EngElectrothermoflexomagnetochiralinstantonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralinstantonplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoinstantonchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralinstantonplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralsolitonplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SOLITON_PLASMON_POLARITON_RES_WORK`` (M465): Engine coupled electrothermal-flexomagnetic-flexochiral-flexosoliton-flexoplasmonic-flexopolaritonic nanoscale chiral-soliton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcsolitonplp: float = 0.0   # time frequency for electrothermoflexomagnetochiralsolitonplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility
    dt_etfcmeronplp: float = 0.0     # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0   # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0 # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcsolitonplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcsolitonplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcsolitonplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcsolitonplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcsolitonplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcphplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcsolitonplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcsolitonplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcsolitonplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcsolitonplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcsolitonplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcsolitonplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcsolitonplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcsolitonplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcsolitonplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcsolitonplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcsolitonplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcsolitonplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcsolitonplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcsolitonplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcsolitonplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcsolitonplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcsolitonplp


EngElectrothermoflexomagnetochiralsolitonplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralsolitonplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetosolitonchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralsolitonplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralvortexplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_VORTEX_PLASMON_POLARITON_RES_WORK`` (M466): Engine coupled electrothermal-flexomagnetic-flexochiral-flexovortex-flexoplasmonic-flexopolaritonic nanoscale chiral-vortex-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcvortexplp: float = 0.0   # time frequency for electrothermoflexomagnetochiralvortexplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility
    dt_etfcmeronplp: float = 0.0     # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0   # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0 # alias field for compatibility
    dt_etfcsolitonplp: float = 0.0   # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfcvortexplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcvortexplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcvortexplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcvortexplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcvortexplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcphplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcvortexplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcvortexplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcvortexplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcvortexplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcvortexplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcvortexplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcvortexplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcvortexplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcvortexplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcvortexplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcvortexplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcvortexplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcvortexplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcvortexplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcvortexplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcvortexplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcvortexplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcvortexplp


EngElectrothermoflexomagnetochiralvortexplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralvortexplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetovortexchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralvortexplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralhopfionplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HOPFION_PLASMON_POLARITON_RES_WORK`` (M467): Engine coupled electrothermal-flexomagnetic-flexochiral-flexohopfion-flexoplasmonic-flexopolaritonic nanoscale chiral-hopfion-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfchopfionplp: float = 0.0  # time frequency for electrothermoflexomagnetochiralhopfionplasmonicpolaritonic resonance energy output
    sens_id: int = 0                # sensor activation ID
    dt_etfplp: float = 0.0          # alias field for compatibility
    dt_etfexplp: float = 0.0        # alias field for compatibility
    dt_etfphplp: float = 0.0        # alias field for compatibility
    dt_etfmagplp: float = 0.0       # alias field for compatibility
    dt_etfcplp: float = 0.0         # alias field for compatibility
    dt_etfcphplp: float = 0.0       # alias field for compatibility
    dt_etfcexplp: float = 0.0       # alias field for compatibility
    dt_etfcmagplp: float = 0.0      # alias field for compatibility
    dt_etfcspinplp: float = 0.0     # alias field for compatibility
    dt_etfcspinonplp: float = 0.0   # alias field for compatibility
    dt_etfcholonplp: float = 0.0    # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0  # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0# alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0   # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0  # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0 # alias field for compatibility
    dt_etfcanyonplp: float = 0.0    # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0 # alias field for compatibility
    dt_etfcmeronplp: float = 0.0    # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0  # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0# alias field for compatibility
    dt_etfcsolitonplp: float = 0.0  # alias field for compatibility
    dt_etfcvortexplp: float = 0.0   # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfchopfionplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfchopfionplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfchopfionplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfchopfionplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfchopfionplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcphplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfchopfionplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfchopfionplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfchopfionplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfchopfionplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfchopfionplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfchopfionplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfchopfionplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfchopfionplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfchopfionplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfchopfionplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfchopfionplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfchopfionplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfchopfionplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfchopfionplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfchopfionplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfchopfionplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfchopfionplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfchopfionplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfchopfionplp


EngElectrothermoflexomagnetochiralhopfionplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralhopfionplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetohopfionchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralhopfionplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralmonopoleplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MONOPOLE_PLASMON_POLARITON_RES_WORK`` (M468): Engine coupled electrothermal-flexomagnetic-flexochiral-flexomonopole-flexoplasmonic-flexopolaritonic nanoscale chiral-monopole-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcmonopoleplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic resonance energy output
    sens_id: int = 0                # sensor activation ID
    dt_etfplp: float = 0.0          # alias field for compatibility
    dt_etfexplp: float = 0.0        # alias field for compatibility
    dt_etfphplp: float = 0.0        # alias field for compatibility
    dt_etfmagplp: float = 0.0       # alias field for compatibility
    dt_etfcplp: float = 0.0         # alias field for compatibility
    dt_etfcphplp: float = 0.0       # alias field for compatibility
    dt_etfcexplp: float = 0.0       # alias field for compatibility
    dt_etfcmagplp: float = 0.0      # alias field for compatibility
    dt_etfcspinplp: float = 0.0     # alias field for compatibility
    dt_etfcspinonplp: float = 0.0   # alias field for compatibility
    dt_etfcholonplp: float = 0.0    # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0  # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0# alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0   # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0  # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0 # alias field for compatibility
    dt_etfcanyonplp: float = 0.0    # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0 # alias field for compatibility
    dt_etfcmeronplp: float = 0.0    # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0  # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0# alias field for compatibility
    dt_etfcsolitonplp: float = 0.0  # alias field for compatibility
    dt_etfcvortexplp: float = 0.0   # alias field for compatibility
    dt_etfchopfionplp: float = 0.0  # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfcmonopoleplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfchopfionplp
        elif self.dt_etfcmonopoleplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcmonopoleplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcmonopoleplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcmonopoleplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcmonopoleplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcmonopoleplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcmonopoleplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcmonopoleplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcmonopoleplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcmonopoleplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcmonopoleplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcmonopoleplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcmonopoleplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcmonopoleplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcmonopoleplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcmonopoleplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcmonopoleplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcmonopoleplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcmonopoleplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcmonopoleplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcmonopoleplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcmonopoleplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcmonopoleplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcmonopoleplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfcmonopoleplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfcmonopoleplp


EngElectrothermoflexomagnetochiralmonopoleplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralmonopoleplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetomonopolechiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralmonopoleplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralsphaleronplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPHALERON_PLASMON_POLARITON_RES_WORK`` (M469): Engine coupled electrothermal-flexomagnetic-flexochiral-flexosphaleron-flexoplasmonic-flexopolaritonic nanoscale chiral-sphaleron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcsphaleronplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility
    dt_etfcmeronplp: float = 0.0     # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0   # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0 # alias field for compatibility
    dt_etfcsolitonplp: float = 0.0   # alias field for compatibility
    dt_etfcvortexplp: float = 0.0    # alias field for compatibility
    dt_etfchopfionplp: float = 0.0   # alias field for compatibility
    dt_etfcmonopoleplp: float = 0.0  # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfchopfionplp
        elif self.dt_etfcmonopoleplp != 0.0 and self.dt_etfcsphaleronplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcmonopoleplp
        elif self.dt_etfcsphaleronplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcsphaleronplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcsphaleronplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcsphaleronplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcsphaleronplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcsphaleronplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcsphaleronplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcsphaleronplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcsphaleronplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcsphaleronplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcsphaleronplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcsphaleronplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcsphaleronplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcsphaleronplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcsphaleronplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcsphaleronplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcsphaleronplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcsphaleronplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcsphaleronplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcsphaleronplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcsphaleronplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcsphaleronplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcsphaleronplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcsolitonplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfcsphaleronplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfcsphaleronplp
            if self.dt_etfcmonopoleplp == 0.0:
                self.dt_etfcmonopoleplp = self.dt_etfcsphaleronplp


EngElectrothermoflexomagnetochiralsphaleronplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralsphaleronplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetosphaleronchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralsphaleronplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiraltoronplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_TORON_PLASMON_POLARITON_RES_WORK`` (M470): Engine coupled electrothermal-flexomagnetic-flexochiral-flexotoron-flexoplasmonic-flexopolaritonic nanoscale chiral-toron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfctoronplp: float = 0.0     # time frequency for electrothermoflexomagnetochiraltoronplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility
    dt_etfcmeronplp: float = 0.0     # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0   # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0 # alias field for compatibility
    dt_etfcsolitonplp: float = 0.0   # alias field for compatibility
    dt_etfcvortexplp: float = 0.0    # alias field for compatibility
    dt_etfchopfionplp: float = 0.0   # alias field for compatibility
    dt_etfcmonopoleplp: float = 0.0  # alias field for compatibility
    dt_etfcsphaleronplp: float = 0.0 # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfchopfionplp
        elif self.dt_etfcmonopoleplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcmonopoleplp
        elif self.dt_etfcsphaleronplp != 0.0 and self.dt_etfctoronplp == 0.0:
            self.dt_etfctoronplp = self.dt_etfcsphaleronplp
        elif self.dt_etfctoronplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfctoronplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfctoronplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfctoronplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfctoronplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfctoronplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfctoronplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfctoronplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfctoronplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfctoronplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfctoronplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfctoronplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfctoronplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfctoronplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfctoronplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfctoronplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfctoronplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfctoronplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfctoronplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfctoronplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfctoronplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfctoronplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfctoronplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcsolitonplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfctoronplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfctoronplp
            if self.dt_etfcmonopoleplp == 0.0:
                self.dt_etfcmonopoleplp = self.dt_etfctoronplp
            if self.dt_etfcsphaleronplp == 0.0:
                self.dt_etfcsphaleronplp = self.dt_etfcsphaleronplp


EngElectrothermoflexomagnetochiraltoronplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiraltoronplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetotoronchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiraltoronplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralbobberplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBOBBERPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BOBBER_PLASMON_POLARITON_RES_WORK`` (M471): Engine coupled electrothermal-flexomagnetic-flexochiral-flexobobber-flexoplasmonic-flexopolaritonic nanoscale chiral-bobber-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcbobberplp: float = 0.0    # time frequency for electrothermoflexomagnetochiralbobberplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility
    dt_etfcmeronplp: float = 0.0     # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0   # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0 # alias field for compatibility
    dt_etfcsolitonplp: float = 0.0   # alias field for compatibility
    dt_etfcvortexplp: float = 0.0    # alias field for compatibility
    dt_etfchopfionplp: float = 0.0   # alias field for compatibility
    dt_etfcmonopoleplp: float = 0.0  # alias field for compatibility
    dt_etfcsphaleronplp: float = 0.0 # alias field for compatibility
    dt_etfctoronplp: float = 0.0     # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcdyonicplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcmajoranaplp = self.dt_etfcbobberplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcskyrmionplp = self.dt_etfcbobberplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcmeronplp = self.dt_etfcbobberplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbimeronplp = self.dt_etfcbobberplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcinstantonplp = self.dt_etfcbobberplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcsolitonplp = self.dt_etfcbobberplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcvortexplp = self.dt_etfcbobberplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfchopfionplp = self.dt_etfcbobberplp
        elif self.dt_etfcmonopoleplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcmonopoleplp = self.dt_etfcbobberplp
        elif self.dt_etfcsphaleronplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcsphaleronplp = self.dt_etfcbobberplp
        elif self.dt_etfctoronplp != 0.0 and self.dt_etfcbobberplp == 0.0:
            self.dt_etfcbobberplp = self.dt_etfctoronplp
        elif self.dt_etfcbobberplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcbobberplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcbobberplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcbobberplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcbobberplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcbobberplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcbobberplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcbobberplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcbobberplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcbobberplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcbobberplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcbobberplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcbobberplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcbobberplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcbobberplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcbobberplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcbobberplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcbobberplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcbobberplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcbobberplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcbobberplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcbobberplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcbobberplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcbobberplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfcbobberplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfcbobberplp
            if self.dt_etfcmonopoleplp == 0.0:
                self.dt_etfcmonopoleplp = self.dt_etfcbobberplp
            if self.dt_etfcsphaleronplp == 0.0:
                self.dt_etfcsphaleronplp = self.dt_etfcbobberplp
            if self.dt_etfctoronplp == 0.0:
                self.dt_etfctoronplp = self.dt_etfcbobberplp


EngElectrothermoflexomagnetochiralbobberplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralbobberplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetobobberchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralbobberplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BLOCH_POINT_PLASMON_POLARITON_RES_WORK`` (M472): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoblochpoint-flexoplasmonic-flexopolaritonic nanoscale chiral-blochpoint-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcblochpointplp: float = 0.0# time frequency for electrothermoflexomagnetochiralblochpointplasmonicpolaritonic resonance energy output
    sens_id: int = 0                 # sensor activation ID
    dt_etfplp: float = 0.0           # alias field for compatibility
    dt_etfexplp: float = 0.0         # alias field for compatibility
    dt_etfphplp: float = 0.0         # alias field for compatibility
    dt_etfmagplp: float = 0.0        # alias field for compatibility
    dt_etfcplp: float = 0.0          # alias field for compatibility
    dt_etfcphplp: float = 0.0        # alias field for compatibility
    dt_etfcexplp: float = 0.0        # alias field for compatibility
    dt_etfcmagplp: float = 0.0       # alias field for compatibility
    dt_etfcspinplp: float = 0.0      # alias field for compatibility
    dt_etfcspinonplp: float = 0.0    # alias field for compatibility
    dt_etfcholonplp: float = 0.0     # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0   # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0 # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0# alias field for compatibility
    dt_etfcdyonicplp: float = 0.0    # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0   # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0  # alias field for compatibility
    dt_etfcanyonplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0  # alias field for compatibility
    dt_etfcmeronplp: float = 0.0     # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0   # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0 # alias field for compatibility
    dt_etfcsolitonplp: float = 0.0   # alias field for compatibility
    dt_etfcvortexplp: float = 0.0    # alias field for compatibility
    dt_etfchopfionplp: float = 0.0   # alias field for compatibility
    dt_etfcmonopoleplp: float = 0.0  # alias field for compatibility
    dt_etfcsphaleronplp: float = 0.0 # alias field for compatibility
    dt_etfctoronplp: float = 0.0     # alias field for compatibility
    dt_etfcbobberplp: float = 0.0    # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfchopfionplp
        elif self.dt_etfcmonopoleplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcmonopoleplp
        elif self.dt_etfcsphaleronplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcsphaleronplp
        elif self.dt_etfctoronplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfctoronplp
        elif self.dt_etfcbobberplp != 0.0 and self.dt_etfcblochpointplp == 0.0:
            self.dt_etfcblochpointplp = self.dt_etfcbobberplp
        elif self.dt_etfcblochpointplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcblochpointplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcblochpointplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcblochpointplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcblochpointplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcblochpointplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcblochpointplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcblochpointplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcblochpointplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcblochpointplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcblochpointplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcblochpointplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcblochpointplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcblochpointplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcblochpointplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcblochpointplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcblochpointplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcblochpointplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcblochpointplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcblochpointplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcblochpointplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcblochpointplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcblochpointplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcblochpointplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfcblochpointplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfcblochpointplp
            if self.dt_etfcmonopoleplp == 0.0:
                self.dt_etfcmonopoleplp = self.dt_etfcblochpointplp
            if self.dt_etfcsphaleronplp == 0.0:
                self.dt_etfcsphaleronplp = self.dt_etfcblochpointplp
            if self.dt_etfctoronplp == 0.0:
                self.dt_etfctoronplp = self.dt_etfcblochpointplp
            if self.dt_etfcbobberplp == 0.0:
                self.dt_etfcbobberplp = self.dt_etfcblochpointplp


EngElectrothermoflexomagnetochiralblochpointplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoblochpointchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HEDGEHOG_PLASMON_POLARITON_RES_WORK`` (M473): Engine coupled electrothermal-flexomagnetic-flexochiral-flexohedgehog-flexoplasmonic-flexopolaritonic nanoscale chiral-hedgehog-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfchedgehogplp: float = 0.0  # time frequency for electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic energy output
    sens_id: int = 0                # optional sensor ID

    # Fallback / compatibility aliases
    dt_etfplp: float = 0.0
    dt_etfexplp: float = 0.0
    dt_etfphplp: float = 0.0
    dt_etfmagplp: float = 0.0
    dt_etfcplp: float = 0.0
    dt_etfcphplp: float = 0.0
    dt_etfcexplp: float = 0.0
    dt_etfcmagplp: float = 0.0
    dt_etfcspinplp: float = 0.0
    dt_etfcspinonplp: float = 0.0
    dt_etfcholonplp: float = 0.0
    dt_etfcorbitonplp: float = 0.0
    dt_etfcplasmononplp: float = 0.0
    dt_etfcparamagnonplp: float = 0.0
    dt_etfcdyonicplp: float = 0.0
    dt_etfcaxionicplp: float = 0.0
    dt_etfcmajoranaplp: float = 0.0
    dt_etfcanyonplp: float = 0.0
    dt_etfcskyrmionplp: float = 0.0
    dt_etfcmeronplp: float = 0.0
    dt_etfcbimeronplp: float = 0.0
    dt_etfcinstantonplp: float = 0.0
    dt_etfcsolitonplp: float = 0.0
    dt_etfcvortexplp: float = 0.0
    dt_etfchopfionplp: float = 0.0
    dt_etfcmonopoleplp: float = 0.0
    dt_etfcsphaleronplp: float = 0.0
    dt_etfctoronplp: float = 0.0
    dt_etfcbobberplp: float = 0.0
    dt_etfcblochpointplp: float = 0.0

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfchopfionplp
        elif self.dt_etfcmonopoleplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcmonopoleplp
        elif self.dt_etfcsphaleronplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcsphaleronplp
        elif self.dt_etfctoronplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfctoronplp
        elif self.dt_etfcbobberplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcbobberplp
        elif self.dt_etfcblochpointplp != 0.0 and self.dt_etfchedgehogplp == 0.0:
            self.dt_etfchedgehogplp = self.dt_etfcblochpointplp
        elif self.dt_etfchedgehogplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfchedgehogplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfchedgehogplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfchedgehogplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfchedgehogplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfchedgehogplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfchedgehogplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfchedgehogplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfchedgehogplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfchedgehogplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfchedgehogplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfchedgehogplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfchedgehogplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfchedgehogplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfchedgehogplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfchedgehogplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfchedgehogplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfchedgehogplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfchedgehogplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfchedgehogplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfchedgehogplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfchedgehogplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfchedgehogplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfchedgehogplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfchedgehogplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfchedgehogplp
            if self.dt_etfcmonopoleplp == 0.0:
                self.dt_etfcmonopoleplp = self.dt_etfchedgehogplp
            if self.dt_etfcsphaleronplp == 0.0:
                self.dt_etfcsphaleronplp = self.dt_etfchedgehogplp
            if self.dt_etfctoronplp == 0.0:
                self.dt_etfctoronplp = self.dt_etfchedgehogplp
            if self.dt_etfcbobberplp == 0.0:
                self.dt_etfcbobberplp = self.dt_etfchedgehogplp
            if self.dt_etfcblochpointplp == 0.0:
                self.dt_etfcblochpointplp = self.dt_etfchedgehogplp


EngElectrothermoflexomagnetochiralhedgehogplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetohedgehogchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SKYRMIONIUM_PLASMON_POLARITON_RES_WORK`` (M474): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoskyrmionium-flexoplasmonic-flexopolaritonic nanoscale chiral-skyrmionium-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcskyrmioniumplp: float = 0.0  # time frequency for electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic energy output
    sens_id: int = 0                   # optional sensor ID

    # Fallback / compatibility aliases
    dt_etfplp: float = 0.0
    dt_etfexplp: float = 0.0
    dt_etfphplp: float = 0.0
    dt_etfmagplp: float = 0.0
    dt_etfcplp: float = 0.0
    dt_etfcphplp: float = 0.0
    dt_etfcexplp: float = 0.0
    dt_etfcmagplp: float = 0.0
    dt_etfcspinplp: float = 0.0
    dt_etfcspinonplp: float = 0.0
    dt_etfcholonplp: float = 0.0
    dt_etfcorbitonplp: float = 0.0
    dt_etfcplasmononplp: float = 0.0
    dt_etfcparamagnonplp: float = 0.0
    dt_etfcdyonicplp: float = 0.0
    dt_etfcaxionicplp: float = 0.0
    dt_etfcmajoranaplp: float = 0.0
    dt_etfcanyonplp: float = 0.0
    dt_etfcskyrmionplp: float = 0.0
    dt_etfcmeronplp: float = 0.0
    dt_etfcbimeronplp: float = 0.0
    dt_etfcinstantonplp: float = 0.0
    dt_etfcsolitonplp: float = 0.0
    dt_etfcvortexplp: float = 0.0
    dt_etfchopfionplp: float = 0.0
    dt_etfcmonopoleplp: float = 0.0
    dt_etfcsphaleronplp: float = 0.0
    dt_etfctoronplp: float = 0.0
    dt_etfcbobberplp: float = 0.0
    dt_etfcblochpointplp: float = 0.0
    dt_etfchedgehogplp: float = 0.0

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfchopfionplp
        elif self.dt_etfcmonopoleplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcmonopoleplp
        elif self.dt_etfcsphaleronplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcsphaleronplp
        elif self.dt_etfctoronplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfctoronplp
        elif self.dt_etfcbobberplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcbobberplp
        elif self.dt_etfcblochpointplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfcblochpointplp
        elif self.dt_etfchedgehogplp != 0.0 and self.dt_etfcskyrmioniumplp == 0.0:
            self.dt_etfcskyrmioniumplp = self.dt_etfchedgehogplp
        elif self.dt_etfcskyrmioniumplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcmonopoleplp == 0.0:
                self.dt_etfcmonopoleplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcsphaleronplp == 0.0:
                self.dt_etfcsphaleronplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfctoronplp == 0.0:
                self.dt_etfctoronplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcbobberplp == 0.0:
                self.dt_etfcbobberplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfcblochpointplp == 0.0:
                self.dt_etfcblochpointplp = self.dt_etfcskyrmioniumplp
            if self.dt_etfchedgehogplp == 0.0:
                self.dt_etfchedgehogplp = self.dt_etfcskyrmioniumplp


EngElectrothermoflexomagnetochiralskyrmioniumplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoskyrmioniumchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANTISKYRMION_PLASMON_POLARITON_RES_WORK`` (M475): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoantiskyrmion-flexoplasmonic-flexopolaritonic nanoscale chiral-antiskyrmion-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcantiskyrmionplp: float = 0.0 # time frequency for electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic resonance energy output
    sens_id: int = 0                    # sensor activation ID
    dt_etfplp: float = 0.0              # alias field for compatibility
    dt_etfexplp: float = 0.0            # alias field for compatibility
    dt_etfphplp: float = 0.0            # alias field for compatibility
    dt_etfmagplp: float = 0.0           # alias field for compatibility
    dt_etfcplp: float = 0.0             # alias field for compatibility
    dt_etfcphplp: float = 0.0           # alias field for compatibility
    dt_etfcexplp: float = 0.0           # alias field for compatibility
    dt_etfcmagplp: float = 0.0          # alias field for compatibility
    dt_etfcspinplp: float = 0.0         # alias field for compatibility
    dt_etfcspinonplp: float = 0.0       # alias field for compatibility
    dt_etfcholonplp: float = 0.0        # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0      # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0    # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0   # alias field for compatibility
    dt_etfcdyonicplp: float = 0.0       # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0      # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0     # alias field for compatibility
    dt_etfcanyonplp: float = 0.0        # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0     # alias field for compatibility
    dt_etfcmeronplp: float = 0.0        # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0      # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0    # alias field for compatibility
    dt_etfcsolitonplp: float = 0.0      # alias field for compatibility
    dt_etfcvortexplp: float = 0.0       # alias field for compatibility
    dt_etfchopfionplp: float = 0.0      # alias field for compatibility
    dt_etfcmonopoleplp: float = 0.0     # alias field for compatibility
    dt_etfcsphaleronplp: float = 0.0    # alias field for compatibility
    dt_etfctoronplp: float = 0.0        # alias field for compatibility
    dt_etfcbobberplp: float = 0.0       # alias field for compatibility
    dt_etfcblochpointplp: float = 0.0   # alias field for compatibility
    dt_etfchedgehogplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmioniumplp: float = 0.0   # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfchopfionplp
        elif self.dt_etfcmonopoleplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcmonopoleplp
        elif self.dt_etfcsphaleronplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcsphaleronplp
        elif self.dt_etfctoronplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfctoronplp
        elif self.dt_etfcbobberplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcbobberplp
        elif self.dt_etfcblochpointplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcblochpointplp
        elif self.dt_etfchedgehogplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfchedgehogplp
        elif self.dt_etfcskyrmioniumplp != 0.0 and self.dt_etfcantiskyrmionplp == 0.0:
            self.dt_etfcantiskyrmionplp = self.dt_etfcskyrmioniumplp

        if self.dt_etfcantiskyrmionplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcmonopoleplp == 0.0:
                self.dt_etfcmonopoleplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcsphaleronplp == 0.0:
                self.dt_etfcsphaleronplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfctoronplp == 0.0:
                self.dt_etfctoronplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcbobberplp == 0.0:
                self.dt_etfcbobberplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcblochpointplp == 0.0:
                self.dt_etfcblochpointplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfchedgehogplp == 0.0:
                self.dt_etfchedgehogplp = self.dt_etfcantiskyrmionplp
            if self.dt_etfcskyrmioniumplp == 0.0:
                self.dt_etfcskyrmioniumplp = self.dt_etfcantiskyrmionplp


EngElectrothermoflexomagnetochiralantiskyrmionplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoantiskyrmionchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy


@dataclass
class EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANTIMERON_PLASMON_POLARITON_RES_WORK`` (M476): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoantimeron-flexoplasmonic-flexopolaritonic nanoscale chiral-antimeron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    id: int = 1
    title: str = ""
    dt_etfcantimeronplp: float = 0.0    # time frequency for electrothermoflexomagnetochiralantimeronplasmonicpolaritonic resonance energy output
    sens_id: int = 0                    # sensor activation ID
    dt_etfplp: float = 0.0              # alias field for compatibility
    dt_etfexplp: float = 0.0            # alias field for compatibility
    dt_etfphplp: float = 0.0            # alias field for compatibility
    dt_etfmagplp: float = 0.0           # alias field for compatibility
    dt_etfcplp: float = 0.0             # alias field for compatibility
    dt_etfcphplp: float = 0.0           # alias field for compatibility
    dt_etfcexplp: float = 0.0           # alias field for compatibility
    dt_etfcmagplp: float = 0.0          # alias field for compatibility
    dt_etfcspinplp: float = 0.0         # alias field for compatibility
    dt_etfcspinonplp: float = 0.0       # alias field for compatibility
    dt_etfcholonplp: float = 0.0        # alias field for compatibility
    dt_etfcorbitonplp: float = 0.0      # alias field for compatibility
    dt_etfcplasmononplp: float = 0.0    # alias field for compatibility
    dt_etfcparamagnonplp: float = 0.0   # alias field for compatibility
    dt_etfcdyonicplp: float = 0.0       # alias field for compatibility
    dt_etfcaxionicplp: float = 0.0      # alias field for compatibility
    dt_etfcmajoranaplp: float = 0.0     # alias field for compatibility
    dt_etfcanyonplp: float = 0.0        # alias field for compatibility
    dt_etfcskyrmionplp: float = 0.0     # alias field for compatibility
    dt_etfcmeronplp: float = 0.0        # alias field for compatibility
    dt_etfcbimeronplp: float = 0.0      # alias field for compatibility
    dt_etfcinstantonplp: float = 0.0    # alias field for compatibility
    dt_etfcsolitonplp: float = 0.0      # alias field for compatibility
    dt_etfcvortexplp: float = 0.0       # alias field for compatibility
    dt_etfchopfionplp: float = 0.0      # alias field for compatibility
    dt_etfcmonopoleplp: float = 0.0     # alias field for compatibility
    dt_etfcsphaleronplp: float = 0.0    # alias field for compatibility
    dt_etfctoronplp: float = 0.0        # alias field for compatibility
    dt_etfcbobberplp: float = 0.0       # alias field for compatibility
    dt_etfcblochpointplp: float = 0.0   # alias field for compatibility
    dt_etfchedgehogplp: float = 0.0     # alias field for compatibility
    dt_etfcskyrmioniumplp: float = 0.0   # alias field for compatibility
    dt_etfcantiskyrmionplp: float = 0.0 # alias field for compatibility
    dt_etfcbiskyrmionplp: float = 0.0   # alias field for compatibility

    def __post_init__(self):
        if self.dt_etfplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfplp
        elif self.dt_etfexplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfexplp
        elif self.dt_etfphplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfphplp
        elif self.dt_etfmagplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfmagplp
        elif self.dt_etfcplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcplp
        elif self.dt_etfcphplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcphplp
        elif self.dt_etfcexplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcexplp
        elif self.dt_etfcmagplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcmagplp
        elif self.dt_etfcspinplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcspinplp
        elif self.dt_etfcspinonplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcspinonplp
        elif self.dt_etfcholonplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcholonplp
        elif self.dt_etfcorbitonplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcorbitonplp
        elif self.dt_etfcplasmononplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcplasmononplp
        elif self.dt_etfcparamagnonplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcparamagnonplp
        elif self.dt_etfcdyonicplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcdyonicplp
        elif self.dt_etfcaxionicplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcaxionicplp
        elif self.dt_etfcmajoranaplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcmajoranaplp
        elif self.dt_etfcanyonplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcanyonplp
        elif self.dt_etfcskyrmionplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcskyrmionplp
        elif self.dt_etfcmeronplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcmeronplp
        elif self.dt_etfcbimeronplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcbimeronplp
        elif self.dt_etfcinstantonplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcinstantonplp
        elif self.dt_etfcsolitonplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcsolitonplp
        elif self.dt_etfcvortexplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcvortexplp
        elif self.dt_etfchopfionplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfchopfionplp
        elif self.dt_etfcmonopoleplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcmonopoleplp
        elif self.dt_etfcsphaleronplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcsphaleronplp
        elif self.dt_etfctoronplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfctoronplp
        elif self.dt_etfcbobberplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcbobberplp
        elif self.dt_etfcblochpointplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcblochpointplp
        elif self.dt_etfchedgehogplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfchedgehogplp
        elif self.dt_etfcskyrmioniumplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcskyrmioniumplp
        elif self.dt_etfcantiskyrmionplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcantiskyrmionplp
        elif self.dt_etfcbiskyrmionplp != 0.0 and self.dt_etfcantimeronplp == 0.0:
            self.dt_etfcantimeronplp = self.dt_etfcbiskyrmionplp

        if self.dt_etfcantimeronplp != 0.0:
            if self.dt_etfplp == 0.0:
                self.dt_etfplp = self.dt_etfcantimeronplp
            if self.dt_etfexplp == 0.0:
                self.dt_etfexplp = self.dt_etfcantimeronplp
            if self.dt_etfphplp == 0.0:
                self.dt_etfphplp = self.dt_etfcantimeronplp
            if self.dt_etfmagplp == 0.0:
                self.dt_etfmagplp = self.dt_etfcantimeronplp
            if self.dt_etfcplp == 0.0:
                self.dt_etfcplp = self.dt_etfcantimeronplp
            if self.dt_etfcphplp == 0.0:
                self.dt_etfcphplp = self.dt_etfcantimeronplp
            if self.dt_etfcexplp == 0.0:
                self.dt_etfcexplp = self.dt_etfcantimeronplp
            if self.dt_etfcmagplp == 0.0:
                self.dt_etfcmagplp = self.dt_etfcantimeronplp
            if self.dt_etfcspinplp == 0.0:
                self.dt_etfcspinplp = self.dt_etfcantimeronplp
            if self.dt_etfcspinonplp == 0.0:
                self.dt_etfcspinonplp = self.dt_etfcantimeronplp
            if self.dt_etfcholonplp == 0.0:
                self.dt_etfcholonplp = self.dt_etfcantimeronplp
            if self.dt_etfcorbitonplp == 0.0:
                self.dt_etfcorbitonplp = self.dt_etfcantimeronplp
            if self.dt_etfcplasmononplp == 0.0:
                self.dt_etfcplasmononplp = self.dt_etfcantimeronplp
            if self.dt_etfcparamagnonplp == 0.0:
                self.dt_etfcparamagnonplp = self.dt_etfcantimeronplp
            if self.dt_etfcdyonicplp == 0.0:
                self.dt_etfcdyonicplp = self.dt_etfcantimeronplp
            if self.dt_etfcaxionicplp == 0.0:
                self.dt_etfcaxionicplp = self.dt_etfcantimeronplp
            if self.dt_etfcmajoranaplp == 0.0:
                self.dt_etfcmajoranaplp = self.dt_etfcantimeronplp
            if self.dt_etfcanyonplp == 0.0:
                self.dt_etfcanyonplp = self.dt_etfcantimeronplp
            if self.dt_etfcskyrmionplp == 0.0:
                self.dt_etfcskyrmionplp = self.dt_etfcantimeronplp
            if self.dt_etfcmeronplp == 0.0:
                self.dt_etfcmeronplp = self.dt_etfcantimeronplp
            if self.dt_etfcbimeronplp == 0.0:
                self.dt_etfcbimeronplp = self.dt_etfcantimeronplp
            if self.dt_etfcinstantonplp == 0.0:
                self.dt_etfcinstantonplp = self.dt_etfcantimeronplp
            if self.dt_etfcsolitonplp == 0.0:
                self.dt_etfcsolitonplp = self.dt_etfcantimeronplp
            if self.dt_etfcvortexplp == 0.0:
                self.dt_etfcvortexplp = self.dt_etfcantimeronplp
            if self.dt_etfchopfionplp == 0.0:
                self.dt_etfchopfionplp = self.dt_etfcantimeronplp
            if self.dt_etfcmonopoleplp == 0.0:
                self.dt_etfcmonopoleplp = self.dt_etfcantimeronplp
            if self.dt_etfcsphaleronplp == 0.0:
                self.dt_etfcsphaleronplp = self.dt_etfcantimeronplp
            if self.dt_etfctoronplp == 0.0:
                self.dt_etfctoronplp = self.dt_etfcantimeronplp
            if self.dt_etfcbobberplp == 0.0:
                self.dt_etfcbobberplp = self.dt_etfcantimeronplp
            if self.dt_etfcblochpointplp == 0.0:
                self.dt_etfcblochpointplp = self.dt_etfcantimeronplp
            if self.dt_etfchedgehogplp == 0.0:
                self.dt_etfchedgehogplp = self.dt_etfcantimeronplp
            if self.dt_etfcskyrmioniumplp == 0.0:
                self.dt_etfcskyrmioniumplp = self.dt_etfcantimeronplp
            if self.dt_etfcantiskyrmionplp == 0.0:
                self.dt_etfcantiskyrmionplp = self.dt_etfcantimeronplp
            if self.dt_etfcbiskyrmionplp == 0.0:
                self.dt_etfcbiskyrmionplp = self.dt_etfcantimeronplp


EngElectrothermoflexomagnetochiralantimeronplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetoantimeronchiralplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetochiralbiskyrmionplasmonicpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy
EngElectrothermoflexomagnetochiralbiskyrmionplasmonpolaritonicResonanceEnergy = EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy
