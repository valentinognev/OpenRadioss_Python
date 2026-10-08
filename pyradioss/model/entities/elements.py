"""Parts, element properties (and their subclasses) and laminates.

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
# Properties  (Fortran: GEO array, starter/source/properties/...)
# ============================================================================

@dataclass
class Property:
    """One /PROP geometry set.

    ``type`` follows the Radioss numbering:
      1  SHELL   (thickness, n. of integration points, hourglass coeffs)
      2  TRUSS   (cross-section area)
      4  SPRING  (mass, stiffness, damping)
      14 SOLID   (quadratic/linear bulk viscosity, hourglass coeff)
    """

    id: int
    type: int
    title: str = ""
    params: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Parts  (Fortran: IPART, starter/source/model/sets/../hm_read_part.F)
# ============================================================================

@dataclass
class Part:
    """A /PART ties elements to one material + one property (like the
    Fortran IPART(1:4,:) = prop, mat, ... mapping). Elements reference the
    part; the part references material and property."""

    id: int
    prop_id: int
    mat_id: int
    title: str = ""


@dataclass
class Submodel:
    """``/SUBMODEL/submodel_ID`` container block (Fortran lecsubmod.F)."""
    id: int
    title: str = ""
    unit_id: int = 0
    off_def: int = 0
    off_nod: int = 0
    off_ele: int = 0
    off_part: int = 0
    off_mat: int = 0
    off_type: int = 0
    off_sub: int = 0


@dataclass
class Subdomain:
    """`/SUBDOMAIN/sub_id` domain partition for Rad2Rad coupling (lecextlnk.F).

    In the Fortran Starter, ``ISUBDOM(1,I)`` stores the count of parts,
    ``ISUBDOM(2,I)`` the user subdomain ID, and ``ISUBDOM_PART`` the
    flat list of internal part indices.  Here we store user-facing part
    IDs directly.
    """
    id: int
    title: str = ""
    part_ids: List[int] = field(default_factory=list)
    neg_part_ids: List[int] = field(default_factory=list)


@dataclass
class SolidPartPerturbation:
    """/PERTURB/PART/SOLID (M99): material/geometric perturbation on solid parts.

    Fortran origin: ``starter/source/model/perturbation/hm_read_perturb_solid.F``.
    """
    id: int
    title: str = ""
    f_mean: float = 0.0
    deviation: float = 0.0
    min_cut: float = 0.0
    max_cut: float = 0.0
    seed: int = 0
    idistri: int = 2
    grpart_id: int = 0
    var_name: str = ""


@dataclass
class Ply:
    """/PLY/ply_id (M100, M156): Composite ply definition.

    Fortran origin: ``starter/source/model/laminate/leclamply.F``.
    """
    id: int
    mat_id: int
    thick: float
    title: str = ""
    skew_id: int = 0
    orientangle: float = 0.0
    grsh4n_id: int = 0
    grsh3n_id: int = 0
    nip: int = 1
    orientangle2: float = 0.0


@dataclass
class LaminatePly:
    """Layer definition inside a /LAMINATE stack."""
    ply_id: int
    phi: float = 0.0
    zi: float = 0.0
    mat_interply: int = 0
    f_weight: float = 1.0


@dataclass
class Laminate:
    """/LAMINATE/laminate_id (M100): Composite laminate stack definition.

    Fortran origin: ``starter/source/model/laminate/leclam.F``.
    """
    id: int
    title: str = ""
    plies: List[LaminatePly] = field(default_factory=list)


@dataclass
class ShellPartPerturbation:
    """/PERTURB/PART/SHELL/perturb_ID (M101): Shell part perturbation.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_perturb_part_shell.F``.
    """
    id: int
    title: str = ""
    grpart_id: int = 0
    chvar: str = "THICK"
    f_mean: float = 0.0
    deviation: float = 0.0
    min_cut: float = 0.0
    max_cut: float = 0.0
    seed: int = 0
    idistri: int = 2


@dataclass
class Retractor:
    """/RETRACTOR (M106): Seatbelt retractor mechanism.

    Fortran origin: ``starter/source/seatbelts/retractor.F`` / CFG ``retractor.cfg``.
    """
    id: int
    title: str = ""
    subtype: str = "SPRING"
    el_id: int = 0
    node_id: int = 0
    elem_size: float = 0.0
    sens_id1: int = 0
    pullout: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    yscale1: float = 1.0
    xscale1: float = 1.0
    sens_id2: int = 0
    tens_typ: int = 0
    force: float = 0.0
    fct_id3: int = 0
    yscale2: float = 1.0
    xscale2: float = 1.0


@dataclass
class Slipring:
    """/SLIPRING (M106): Seatbelt slipring friction element.

    Fortran origin: ``starter/source/seatbelts/slipring.F`` / CFG ``slipring.cfg``, ``slipring_shell.cfg``.
    """
    id: int
    title: str = ""
    subtype: str = "SPRING"  # SPRING or SHELL
    el_id1: int = 0          # or EL_SET1 for SHELL
    el_id2: int = 0          # or EL_SET2 for SHELL
    node_id: int = 0         # or Node_SET for SHELL
    node_id2: int = 0
    sens_id: int = 0
    flow_flag: int = 0
    a: float = 0.0
    ed_factor: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fricd: float = 0.0
    xscale1: float = 1.0
    yscale2: float = 1.0
    xscale2: float = 1.0
    fct_id3: int = 0
    fct_id4: int = 0
    frics: float = 0.0
    xscale3: float = 1.0
    yscale4: float = 1.0
    xscale4: float = 1.0


@dataclass
class Pretensioner:
    """``/PRETENSIONER`` or ``/SEATBELT/PRETENSIONER`` (M197): Seatbelt pretensioner element.

    Fortran origin: ``starter/source/seatbelts/pretensioner.F`` / CFG ``pretensioner.cfg``.
    """
    id: int
    title: str = ""
    sens_id: int = 0
    fct_id: int = 0
    fscale: float = 1.0
    tstart: float = 0.0
    vmax: float = 0.0
    amax: float = 0.0
    reinf: float = 0.0
    i_type: int = 0
    retractor_id: int = 0
    slipring_id: int = 0
    element_ids: List[int] = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class SlipringShell:
    """/SLIPRING/SHELL (M123): Shell-to-shell seatbelt slipring connector."""
    id: int
    el_set1: int = 0
    el_set2: int = 0
    node_set: int = 0
    sens_id: int = 0
    flow_flag: int = 0
    a: float = 0.0
    ed_factor: float = 0.0
    fric_d: float = 0.0
    fric_s: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    fct_id4: int = 0
    xscale1: float = 1.0
    xscale2: float = 1.0
    yscale2: float = 1.0
    xscale3: float = 1.0
    xscale4: float = 1.0
    yscale4: float = 1.0
    title: str = ""

    @property
    def fricd(self) -> float:
        return self.fric_d

    @fricd.setter
    def fricd(self, val: float) -> None:
        self.fric_d = val

    @property
    def frics(self) -> float:
        return self.fric_s

    @frics.setter
    def frics(self, val: float) -> None:
        self.fric_s = val

    @property
    def sensor_id(self) -> int:
        return self.sens_id

    @sensor_id.setter
    def sensor_id(self, val: int) -> None:
        self.sens_id = val


@dataclass
class StackPly:
    """Layer definition inside a /STACK (M127)."""
    ply_id: int
    phi: float = 0.0
    zi: float = 0.0
    p_thick_fail: float = 0.0
    f_weight: float = 1.0


@dataclass
class Stack:
    """/STACK/stack_id (M127): Composite laminate stack definition.

    Fortran origin: ``starter/source/model/laminate/leclamply.F`` / CFG ``stack.cfg``.
    """
    id: int
    title: str = ""
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    z0: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    istrain: int = 0
    ashear: float = 0.833333
    iint: int = 0
    ithick: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    ip: int = 0
    plies: List[StackPly] = field(default_factory=list)


@dataclass
class PropRivet:
    """/PROP/TYPE5 or /PROP/RIVET (M143): Fastener / Rivet connector property.

    Fortran origin: ``starter/source/properties/rivet/hm_read_prop05.F``.
    """
    id: int
    title: str = ""
    mass: float = 0.0
    stiffness: float = 0.0
    fn_fail: float = 0.0
    ft_fail: float = 0.0


@dataclass
class PropXelem:
    """/PROP/TYPE28 or /PROP/XELEM (M143): X-FEM / cohesive element property.

    Fortran origin: ``starter/source/properties/xelem/hm_read_prop28.F``.
    """
    id: int
    title: str = ""
    itip: int = 0
    isurf: int = 0
    alpha: float = 0.0


@dataclass
class PerturbControl:
    """/PERTURB/PART/SHELL, /PERTURB/PART/SOLID, /PERTURB/FAIL (M151): Perturbation controls.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_perturb*.F``.
    """
    id: int
    title: str = ""
    subtype: str = "SHELL"
    grpart_id: int = 0
    ityp: int = 1
    fct_id: int = 0
    scale: float = 0.0
    seed: int = 0


@dataclass
class PropInject1Gas:
    """Gas component entry for /PROP/INJECT1."""
    mat_id: int = 0
    fun_id_m: int = 0
    fun_id_t: int = 0
    fscale_m: float = 1.0
    fscale_t: float = 1.0


@dataclass
class PropInject1:
    """/PROP/INJECT1 or /INJECT1 (M152): Gas injector property definition.

    Fortran origin: ``starter/source/properties/injector/hm_read_inject1.F``.
    """
    id: int
    title: str = ""
    n_gases: int = 1
    iflow: int = 0
    ascale_t: float = 1.0
    gases: List[PropInject1Gas] = field(default_factory=list)


@dataclass
class PropInject2Gas:
    """Gas mixture molar fraction entry for /PROP/INJECT2."""
    mat_id: int = 0
    molar_fraction: float = 1.0
    fun_id_mf: int = 0


@dataclass
class PropInject2:
    """/PROP/INJECT2 or /INJECT2 (M152): Multi-gas mixture injector property definition.

    Fortran origin: ``starter/source/properties/injector/hm_read_inject2.F``.
    """
    id: int
    title: str = ""
    n_gases: int = 1
    iflow: int = 0
    fun_id_m: int = 0
    fun_id_t: int = 0
    fscale_m: float = 1.0
    fscale_t: float = 1.0
    ascale_t: float = 1.0
    gases: List[PropInject2Gas] = field(default_factory=list)


@dataclass
class PropJoint:
    """/PROP/TYPE33 or /PROP/JOINT (M152): Specialized kinematic joints.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop33*.F``.
    """
    id: int
    title: str = ""
    joint_type: str = "SPH"
    skew_id: int = 0
    params: Dict = field(default_factory=dict)


@dataclass
class PropTorsion:
    """/PROP/TYPE35 or /PROP/TORSION (M152): Torsion bar spring property.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop35.F``.
    """
    id: int
    title: str = ""
    mass: float = 0.0
    k_elas: float = 0.0
    x_lim1: float = 0.0
    x_lim2: float = 0.0
    k_post: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    r_load: float = 0.0
    f_scal: float = 1.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    fct_id4: int = 0


@dataclass
class PropSpringElasPlas:
    """/PROP/SPR_ELAS_PLAS (M152): Elastic-plastic spring with kinematic hardening.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop_spr_ep.F``.
    """
    id: int
    title: str = ""
    skew_id: int = 0
    i_utyp: int = 1
    pid1: int = 0
    pid2: int = 0
    mid1: int = 0
    k_stiff: float = 0.0
    area: float = 0.0
    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    params: Dict = field(default_factory=dict)


@dataclass
class PropSpringBeam:
    """/PROP/TYPE44 (M152): Non-linear beam-spring connector.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop44.F``.
    """
    id: int
    title: str = ""
    skew_id: int = 0
    idamp: int = 0
    nc_filter: int = 0
    params: Dict = field(default_factory=dict)


@dataclass
class PropSpotweld:
    """/PROP/TYPE45 (M152): Spotweld connector beam.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop45.F``.
    """
    id: int
    title: str = ""
    skew_id: int = 0
    sensor_id: int = 0
    knn: float = 0.0
    cr: float = 0.0
    scf: float = 1.0
    params: Dict = field(default_factory=dict)


@dataclass
class PropBushing:
    """/PROP/TYPE46 (M152): Bushing connector spring.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop46.F``.
    """
    id: int
    title: str = ""
    mass: float = 0.0
    k_elas: float = 0.0
    x_lim1: float = 0.0
    x_lim2: float = 0.0
    k_post: float = 0.0
    damp: float = 0.0
    epsi: int = 0
    idens: int = 0
    params: Dict = field(default_factory=dict)

@dataclass
class SubLaminatePly:
    """Ply layer within a sub-laminate stack."""
    ply_id: int
    phi: float = 0.0
    zi: float = 0.0
    p_thick_fail: float = 0.0
    f_weight: float = 1.0


@dataclass
class SubLaminate:
    """/SUBLAMINATE or /STACK/SUB_LAMINATE (M165): Sub-laminate composite ply stack definition.

    Fortran origin: ``LAMINATE/sub_laminate_p51.cfg`` and ``LAMINATE/stack_sub_laminate.cfg``.
    """
    id: int
    title: str = ""
    plies: List[SubLaminatePly] = field(default_factory=list)


@dataclass
class PropType20:
    """/PROP/TYPE20 / /PROP/TSHELL (M180): Thick shell property."""
    id: int = 0
    isolid: int = 15
    ismstr: int = 0
    icpre: int = 0
    icstr: int = 0
    inpts_r: int = 2
    inpts_s: int = 2
    inpts_t: int = 2
    iint: int = 1
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    h: float = 0.1
    deltat_min: float = 0.0
    nbp: int = 0
    title: str = ""


PropTshell = PropType20
PropThickShell = PropType20


@dataclass
class PropType21:
    """/PROP/TYPE21 / /PROP/TSH_ORTH (M180): Orthotropic thick shell property."""
    id: int
    isolid: int = 15
    ismstr: int = 0
    icstr: int = 0
    inpts_r: int = 2
    inpts_s: int = 2
    inpts_t: int = 2
    iint: int = 1
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    phi: float = 0.0
    deltat_min: float = 0.0
    title: str = ""


@dataclass
class PropType22Layer:
    """Layer definition for /PROP/TYPE22 (/PROP/TSH_COMP)."""
    phi: float = 0.0
    thick: float = 0.0
    zi: float = 0.0
    mat_id: int = 0


@dataclass
class PropType22:
    """/PROP/TYPE22 / /PROP/TSH_COMP (M180): Composite layered thick shell property."""
    id: int
    isolid: int = 15
    ismstr: int = 0
    icstr: int = 0
    inpts_r: int = 2
    inpts_s: int = 2
    inpts_t: int = 2
    iint: int = 1
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    ashear: float = 0.0
    layers: List[PropType22Layer] = field(default_factory=list)
    deltat_min: float = 0.0
    title: str = ""


@dataclass
class PropType6:
    """/PROP/TYPE6 or /PROP/SOL_ORTH (M181): Solid orthotropic property."""
    id: int = 0
    isolid: int = 14
    ismstr: int = 0
    icpre: int = 0
    itetra10: int = 0
    inpts_r: int = 1
    inpts_s: int = 1
    inpts_t: int = 1
    itetra4: int = 0
    iframe: int = 0
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    h: float = 0.1
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    skew_csid: int = 0
    refplane: int = 0
    orthtrop: int = 0
    mat_beta: float = 0.0
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0
    deltat_min: float = 0.0
    vdef_min: float = 0.0
    vdef_max: float = 0.0
    asp_max: float = 0.0
    col_min: float = 0.0
    ndir: int = 0
    sphpart_id: int = 0
    istrain: int = 0
    ihkt: int = 0
    nbp: int = 0
    title: str = ""


PropSolOrth = PropType6
PropSolidOrth = PropType6


@dataclass
class PropType26Curve:
    fct_id: int = 0
    fscale: float = 1.0
    strain_rate: float = 0.0


@dataclass
class PropType26:
    """/PROP/TYPE26 or /PROP/SPR_TAB (M182): Tabulated nonlinear spring property."""
    id: int
    mass: float = 0.0
    sens_id: int = 0
    isflag: int = 0
    ileng: int = 0
    dmin: float = 0.0
    nfunc: int = 1
    nfund: int = 1
    lscale: float = 1.0
    kmax: float = 1.0
    dmax: float = 0.0
    alpha: float = 1.0
    loading_curves: list[PropType26Curve] = field(default_factory=list)
    unloading_curves: list[PropType26Curve] = field(default_factory=list)
    title: str = ""


PropSprTab = PropType26


@dataclass
class PropType27:
    """/PROP/TYPE27 or /PROP/SPR_BDAMP (M182): Spring with bilinear/barycentric damping."""
    id: int
    mass: float = 0.0
    sens_id: int = 0
    isflag: int = 0
    ileng: int = 0
    itens: int = 0
    ifail: int = 0
    stiff: float = 0.0
    damp: float = 0.0
    nexp: float = 1.0
    delta_min: float = 0.0
    delta_max: float = 0.0
    gap: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    ascale1: float = 1.0
    fscale1: float = 1.0
    ascale2: float = 1.0
    fscale2: float = 1.0
    title: str = ""


PropSprBdamp = PropType27


@dataclass
class PropSandwLayer:
    """Layer definition for /PROP/TYPE11 (SH_SANDW)."""
    phi: float = 0.0
    thick: float = 0.0
    z: float = 0.0
    mat_id: int = 0
    f_weight: float = 0.0


@dataclass
class PropType11:
    """/PROP/TYPE11 or /PROP/SH_SANDW (M184): Sandwich shell property."""
    id: int
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    p_thick_fail: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    nip: int = 0
    istrain: int = 0
    thick: float = 0.0
    ashear: float = 0.0
    ithick: int = 0
    iplas: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_csid: int = 0
    iorth: int = 0
    ipos: int = 0
    ip: int = 0
    layers: list[PropSandwLayer] = field(default_factory=list)
    title: str = ""


PropShSandw = PropType11
PropSandwich = PropType11


@dataclass
class PropFabricLayer:
    """Layer definition for /PROP/TYPE16 (SH_FABR)."""
    phi: float = 0.0
    alpha: float = 0.0
    thick: float = 0.0
    z: float = 0.0
    mat_id: int = 0


@dataclass
class PropType16:
    """/PROP/TYPE16 or /PROP/SH_FABR (M184): Fabric shell property."""
    id: int
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    p_thick_fail: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    nip: int = 0
    istrain: int = 0
    thick: float = 0.0
    ashear: float = 0.0
    ithick: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    ipos: int = 0
    ip: int = 0
    layers: list[PropFabricLayer] = field(default_factory=list)
    title: str = ""


PropShFabr = PropType16
PropFabricShell = PropType16
PropFabric = PropType16


@dataclass
class PropType17:
    """/PROP/TYPE17 or /PROP/STACK (M184): Composite ply stack property."""
    id: int
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    plyxfem: int = 0
    z0: float = 0.0
    vinterply: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    thick: float = 0.0
    ashear: float = 0.0
    ithick: int = 0
    iplas: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    refplane: int = 0
    title: str = ""


PropStack = PropType17
PropCompStack = PropType17


@dataclass
class PropType19:
    """/PROP/TYPE19 or /PROP/SPR_TORS: Torsion spring property."""
    id: int = 0
    mass: float = 0.0
    inertia: float = 0.0
    k_theta: float = 0.0
    c_theta: float = 0.0
    title: str = ""


PropSprTors = PropType19

@dataclass
class PropTorsion:
    """/PROP/TORSION or /PROP/TYPE35 (M152): Torsion bar spring property."""
    id: int = 0
    title: str = ""
    mass: float = 0.0
    k_elas: float = 0.0
    x_lim1: float = 0.0
    x_lim2: float = 0.0
    k_post: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    r_load: float = 0.0
    f_scal: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    fct_id4: int = 0
    inertia: float = 0.0
    k_theta: float = 0.0
    c_theta: float = 0.0



@dataclass
class PropType44:
    """/PROP/TYPE44 or /PROP/SPR_CRUS (M184): Crushing frame spring property."""
    id: int
    mass: float = 0.0
    inertia: float = 0.0
    stiff1: float = 0.0
    skew_csid: int = 0
    icoupling: int = 0
    ifiltr: int = 0
    k11: float = 0.0
    k44: float = 0.0
    k55: float = 0.0
    k66: float = 0.0
    idamp: int = 0
    k5b: float = 0.0
    k6c: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_a2: int = 0
    fscale11: float = 1.0
    fun_b2: int = 0
    fun_a3: int = 0
    fun_b3: int = 0
    fun_a4: int = 0
    fscale22: float = 1.0
    fun_b4: int = 0
    fun_a5: int = 0
    fun_b5: int = 0
    fun_a6: int = 0
    fscale33: float = 1.0
    fun_b6: int = 0
    fun_c1: int = 0
    fun_c2: int = 0
    fun_c3: int = 0
    fscale12: float = 1.0
    fun_c4: int = 0
    fun_c5: int = 0
    fun_c6: int = 0
    fun_d1: int = 0
    fscale23: float = 1.0
    fun_d2: int = 0
    fun_d3: int = 0
    fun_d4: int = 0
    fun_d5: int = 0
    fscale13: float = 1.0
    strain1: float = 0.0
    strain2: float = 0.0
    strain3: float = 0.0
    strain4: float = 0.0
    strain5: float = 0.0
    strain6: float = 0.0
    strain7: float = 0.0
    fct_d_x: int = 0
    dscale_x: float = 0.0
    f_x: float = 0.0
    fct_d_y: int = 0
    dscale_y: float = 0.0
    f_y: float = 0.0
    fct_d_z: int = 0
    dscale_z: float = 0.0
    f_z: float = 0.0
    fct_d_xx: int = 0
    dscale_xx: float = 0.0
    f_xx: float = 0.0
    fct_d_yy: int = 0
    dscale_yy: float = 0.0
    f_yy: float = 0.0
    fct_d_zz: int = 0
    dscale_zz: float = 0.0
    f_zz: float = 0.0
    title: str = ""


PropSprCrus = PropType44
PropCrushSpring = PropType44
PropSpringCrush = PropType44


@dataclass
class PropType12:
    """``/PROP/TYPE12`` or ``/PROP/SPR_PUL``: Pulley spring / sliding cable property."""
    id: int = 0
    mass: float = 0.0
    isensor: int = 0
    isflag: int = 0
    ileng: int = 0
    fric: float = 0.0
    stiff1: float = 0.0
    damp1: float = 0.0
    acoeft1: float = 1.0
    bcoeft1: float = 0.0
    dcoeft1: float = 1.0
    fun_a1: int = 0
    hflag1: int = 0
    fun_b1: int = 0
    fct_id31: int = 0
    fun_a2: int = 0
    min_rup1: float = -1.0e30
    max_rup1: float = 1.0e30
    prop_x_f: float = 1.0
    prop_x_e: float = 0.0
    scale1: float = 1.0
    h: float = 1.0
    funct_id: int = 0
    ifric: int = 0
    scale2: float = 1.0
    scale3: float = 1.0
    f_min: float = -1.0e30
    f_max: float = 1.0e30
    title: str = ""

    @property
    def k(self) -> float:
        return self.stiff1

    @property
    def c(self) -> float:
        return self.damp1

    @property
    def sensor_id(self) -> int:
        return self.isensor


PropSprPul = PropType12
PropPulley = PropType12


@dataclass
class PropType15:
    """``/PROP/TYPE15`` or ``/PROP/POROUS``: Porous solid property."""
    id: int = 0
    qa: float = 0.0
    qb: float = 0.0
    h: float = 0.1
    poros: float = 1.0
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    skew_csid: int = 0
    ihon: int = 0
    itu: int = 0
    alpha: float = 0.1
    l_mix: float = 0.0
    irby: int = 0
    title: str = ""

    @property
    def porosity(self) -> float:
        return self.poros

    @property
    def skew_id(self) -> int:
        return self.skew_csid


PropPorous = PropType15
PropSolidPorous = PropType15


@dataclass
class PropStrandLayer:
    """Layer definition for /PROP/TYPE28 (NSTRAND)."""
    type_name: str = ""
    k_id: int = 0
    mu: float = 0.0


@dataclass
class PropType28:
    """``/PROP/TYPE28`` or ``/PROP/NSTRAND``: Multi-strand cable / wire rope property."""
    id: int = 0
    mass: float = 0.0
    k: float = 0.0
    c: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    strain1: float = -1.0e30
    strain2: float = 1.0e30
    mu1: float = 0.0
    mu2: float = 0.0
    fscale11: float = 1.0
    fscale22: float = 1.0
    layers: list[PropStrandLayer] = field(default_factory=list)
    title: str = ""


PropNstrand = PropType28
PropStrand = PropType28


@dataclass
class PropType33:
    """``/PROP/TYPE33`` or ``/PROP/KJOINT`` / ``/PROP/KINEMATIC_JOINT``: 6-DOF kinematic joint property."""
    id: int = 0
    joint_type: int = 1
    skew_flag: int = 0
    id_sk1: int = 0
    id_sk2: int = 0
    xk: float = 0.0
    cr: float = 0.0
    kn: float = 0.0
    krx: float = 0.0
    kry: float = 0.0
    krz: float = 0.0
    ktx: float = 0.0
    kty: float = 0.0
    ktz: float = 0.0
    xr_fun: int = 0
    yr_fun: int = 0
    zr_fun: int = 0
    xt_fun: int = 0
    yt_fun: int = 0
    zt_fun: int = 0
    crx: float = 0.0
    cry: float = 0.0
    crz: float = 0.0
    ctx: float = 0.0
    cty: float = 0.0
    ctz: float = 0.0
    crx_fun: int = 0
    cry_fun: int = 0
    crz_fun: int = 0
    ctx_fun: int = 0
    cty_fun: int = 0
    ctz_fun: int = 0
    title: str = ""


PropKjoint = PropType33
PropKinematicJoint = PropType33


@dataclass
class PropType46:
    """``/PROP/TYPE46`` or ``/PROP/SPR_MUSCLE`` / ``/PROP/MUSCLE``: Hill-type muscle spring property."""
    id: int = 0
    mass: float = 0.0
    stiff0: float = 0.0
    vel_max: float = 0.0
    nforce: float = 0.0
    stiff1: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_c1: int = 0
    fun_d1: int = 0
    mat_imass: int = 0
    damp1: float = 0.0
    epsi: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    fscale21: float = 1.0
    fscale12: float = 1.0
    title: str = ""


PropSprMuscle = PropType46
PropMuscle = PropType46


@dataclass
class PropType35:
    """``/PROP/TYPE35`` or ``/PROP/STITCH`` / ``/PROP/SEW``: Stitch seam fastener property."""
    id: int = 0
    amas: float = 0.0
    elastif: float = 0.0
    xlim1: float = 0.0
    xk: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_c1: int = 0
    fun_d1: int = 0
    damg: float = 0.0
    fdelay: float = 0.0
    xlim2: float = 0.0
    rload: float = 0.0
    iload: int = 0
    fscal: float = 1.0
    title: str = ""


PropStitch = PropType35
PropSew = PropType35


@dataclass
class PropType45:
    """``/PROP/TYPE45`` or ``/PROP/KJOINT2`` / ``/PROP/KINEMATIC_JOINT2``: 6-DOF Kinematic Joint Type 2."""
    id: int = 0
    joint_type: int = 1
    kn: float = 0.0
    scale: float = 1.0
    cr: float = 0.0
    isensor: int = 0
    skew1: int = 0
    skew2: int = 0
    ktx: float = 0.0
    kty: float = 0.0
    ktz: float = 0.0
    xt_fun: int = 0
    yt_fun: int = 0
    zt_fun: int = 0
    xn: float = 0.0
    yn: float = 0.0
    zn: float = 0.0
    xc: float = 0.0
    yc: float = 0.0
    zc: float = 0.0
    ctx: float = 0.0
    cty: float = 0.0
    ctz: float = 0.0
    ctx_fun: int = 0
    cty_fun: int = 0
    ctz_fun: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    title: str = ""


PropKjoint2 = PropType45
PropKinematicJoint2 = PropType45


@dataclass
class PropType36:
    """``/PROP/TYPE36`` or ``/PROP/PREDIT``: Progressive damage interface spring property."""
    id: int = 0
    lutype: int = 1
    skew_csid: int = 0
    prop_id1: int = 0
    prop_id2: int = 0
    xk: float = 0.0
    mat_id: int = 0
    area: float = 0.0
    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    ray: float = 0.0
    # Direct / inherited material & failure parameters
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    g: float = 0.0
    sig0: float = 0.0
    hpla: float = 0.0
    m: float = 1.0
    sfac: float = 1.0
    ay: float = 1.0
    az: float = 1.0
    by: float = 1.0
    bz: float = 1.0
    cx: float = 1.0
    dc: float = 0.99999
    pr: float = 1.0e30
    ps: float = 1.0e30
    ifunc: int = 0
    title: str = ""

    @property
    def type(self) -> int:
        return 36


PropPredit = PropType36


@dataclass
class PropType9:
    """``/PROP/TYPE9`` or ``/PROP/SH_ORTH``: Orthotropic shell property."""
    id: int = 0
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    nip: int = 1
    istrain: int = 0
    thick: float = 0.0
    ashear: float = 0.833333
    ithick: int = 0
    iplas: int = 0
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    phi: float = 0.0
    title: str = ""


PropShOrth = PropType9
PropShellOrth = PropType9


@dataclass
class PropType10:
    """``/PROP/TYPE10`` or ``/PROP/SH_COMP``: Multi-layer composite shell property."""
    id: int = 0
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    nip: int = 1
    istrain: int = 0
    thick: float = 0.0
    ashear: float = 0.833333
    ithick: int = 0
    iplas: int = 0
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    phis: list[float] = field(default_factory=list)
    title: str = ""


PropShComp = PropType10
PropShellComp = PropType10


@dataclass
class PropType51:
    """``/PROP/TYPE51``, ``/PROP/P51`` or ``/PROP/SH_COH``: Cohesive shell / composite stack property."""
    id: int = 0
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    pthk: float = 0.0
    zshift: float = 0.0
    p_thick_fail: float = 0.0
    z0: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    istrain: int = 0
    ashear: float = 0.833333
    iint: int = 0
    ithick: int = 0
    failexp: float = 0.0
    fexp: float = 0.0
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    idsk: int = 0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    irp: int = 0
    refplane: int = 0
    layers: List[Any] = field(default_factory=list)
    title: str = ""


PropShCoh = PropType51
PropShellCoh = PropType51


@dataclass
class PropType5:
    """``/PROP/TYPE5`` or ``/PROP/RIVET``: Rivet connection property."""
    id: int = 0
    mass: float = 0.0
    stiffness: float = 0.0
    fn_fail: float = 0.0
    ft_fail: float = 0.0
    nforce: float = 0.0
    tforce: float = 0.0
    length: float = 0.0
    wflag: int = 0
    imod: int = 1
    title: str = ""

    @property
    def fn(self) -> float:
        return self.nforce or self.fn_fail

    @property
    def ft(self) -> float:
        return self.tforce or self.ft_fail



@dataclass
class PropType14:
    """``/PROP/TYPE14`` or ``/PROP/SOLID``: Generalized 3D solid property."""
    id: int = 0
    isolid: int = 14
    ismstr: int = 0
    icpre: int = 0
    inpts_r: int = 1
    inpts_s: int = 1
    inpts_t: int = 1
    i_rot: int = 0
    iframe: int = 0
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    h: float = 0.1
    deltat_min: float = 0.0
    istrain: int = 0
    qa_l: float = 0.0
    qb_l: float = 0.0
    h_l: float = 0.0
    iplas: int = 0
    icstr: int = 0
    title: str = ""


PropSolGene = PropType14
PropSolid = PropType14


@dataclass
class PropType8:
    """``/PROP/TYPE8`` or ``/PROP/SPR_GENE``: Generalized 6-DOF nonlinear spring property."""
    id: int = 0
    mass: float = 0.0
    inertia: float = 0.0
    skew_id: int = 0
    sensor_id: int = 0
    isflag: int = 0
    ifail: int = 0
    iequil: int = 0
    dofs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    title: str = ""


PropSprGene = PropType8
PropSpringGene = PropType8


@dataclass
class PropType25:
    """``/PROP/TYPE25`` or ``/PROP/SPR_AXI``: Axisymmetric nonlinear spring property."""
    id: int = 0
    mass: float = 0.0
    inertia: float = 0.0
    skew_id: int = 0
    sensor_id: int = 0
    isflag: int = 0
    ifail: int = 0
    ileng: int = 0
    ifail2: int = 0
    tension: Dict[str, Any] = field(default_factory=dict)
    shear: Dict[str, Any] = field(default_factory=dict)
    title: str = ""


PropSprAxi = PropType25
PropSpringAxi = PropType25


@dataclass
class PropType32:
    """``/PROP/TYPE32`` or ``/PROP/SPR_PRE``: Preloaded spring property."""
    id: int = 0
    mass: float = 0.0
    sensor_id: int = 0
    ilock: int = 0
    stiff0: float = 0.0
    f1: float = 0.0
    d1: float = 0.0
    e1: float = 0.0
    stiff1: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    scale_t: float = 1.0
    scale_d: float = 1.0
    scale_f: float = 1.0
    title: str = ""


PropSprPre = PropType32
PropSpringPre = PropType32


@dataclass
class PropType43:
    """``/PROP/TYPE43`` or ``/PROP/CONNECT``: Connector element property."""
    id: int = 0
    ismstr: int = 1
    thick: float = 0.0
    title: str = ""


PropConnect = PropType43
PropPropConnect = PropType43


@dataclass
class PropType34:
    """``/PROP/TYPE34`` or ``/PROP/SPH``: SPH particle property."""
    id: int = 0
    mass: float = 0.0
    h0: float = 0.0
    d0: float = 0.0
    qa: float = 2.0
    qb: float = 1.0
    alpha1: float = 0.0
    order: int = 0
    h: float = 0.0
    title: str = ""


PropSph = PropType34
PropPropSph = PropType34


@dataclass
class PropType29:
    """``/PROP/TYPE29``: User-defined property Type 29."""
    id: int = 0
    cards: List[str] = field(default_factory=list)
    title: str = ""


@dataclass
class PropType30:
    """``/PROP/TYPE30``: User-defined property Type 30."""
    id: int = 0
    cards: List[str] = field(default_factory=list)
    title: str = ""


@dataclass
class PropType31:
    """``/PROP/TYPE31``: User-defined property Type 31."""
    id: int = 0
    cards: List[str] = field(default_factory=list)
    title: str = ""




@dataclass
class PropIntBeamIP:
    y: float = 0.0
    z: float = 0.0
    area: float = 0.0


@dataclass
class PropType18:
    """``/PROP/TYPE18`` & ``/PROP/INT_BEAM``: Integrated beam property."""
    id: int = 0
    isflag: int = 0
    ismstr: int = 0
    dm: float = 0.0
    df: float = 0.0
    nip: int = 0
    iref: int = 0
    y0: float = 0.0
    z0: float = 0.0
    ips: List[PropIntBeamIP] = field(default_factory=list)
    nitrs: int = 0
    l1: float = 0.0
    l2: float = 0.0
    l3: float = 0.0
    l4: float = 0.0
    l5: float = 0.0
    l6: float = 0.0
    wx1: int = 0
    wy1: int = 0
    wz1: int = 0
    wx2: int = 0
    wy2: int = 0
    wz2: int = 0
    title: str = ""
    area: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    ixx: float = 0.0
    zy: float = 0.0
    zz: float = 0.0
    ishear: int = 0
    iform: int = 0
    params: Dict[str, Any] = field(default_factory=dict)


PropIntBeam = PropType18


@dataclass
class PropType13:
    """``/PROP/TYPE13`` or ``/PROP/SPR_PULL`` (M194): Pulling spring property."""
    id: int
    title: str = ""
    stiff: float = 0.0
    f_max: float = 0.0
    params: dict = field(default_factory=dict)
PropSprPull = PropType13



@dataclass
class PropType23:
    """``/PROP/TYPE23`` or ``/PROP/SPR_MAT`` (M195): Spring material property."""
    id: int
    type: int = 23
    title: str = ""
    mass: float = 0.0
    skew_id: int = 0
    isens: int = 0
    iflag: int = 0
    imass: int = 2
    area_or_volume: float = 0.0
    inertia: float = 0.0
    sensor_id: int = 0
    isflag: int = 0
    params: dict = field(default_factory=dict)
PropSprMat = PropType23


@dataclass
class PropPcompp:
    """``/PROP/PCOMPP/id`` (M201): Ply-based composite shell property."""
    id: int = 0
    title: str = ""
    laminate_id: int = 0


PropP51 = PropType51
PropTshP51 = PropType51


@dataclass
class PropSpringTors:
    """``/PROP/TYPE19`` or ``/PROP/SPR_TORS/id`` (M205): Torsional spring property."""
    id: int = 1
    title: str = ""
    mass: float = 0.0
    stiffness_k: float = 0.0
    damping_c: float = 0.0
    fcut: float = 0.0
    inertia: float = 0.0

    @property
    def k(self) -> float:
        return self.stiffness_k

    @property
    def c(self) -> float:
        return self.damping_c

    @property
    def k_theta(self) -> float:
        return self.stiffness_k

    @property
    def c_theta(self) -> float:
        return self.damping_c


@dataclass
class PropSpringBend:
    """``/PROP/TYPE20`` or ``/PROP/SPR_BEND/id`` (M205): Bending spring property."""
    id: int = 1
    title: str = ""
    mass: float = 0.0
    stiffness_k: float = 0.0
    damping_c: float = 0.0
    fcut: float = 0.0

    @property
    def k(self) -> float:
        return self.stiffness_k

    @property
    def c(self) -> float:
        return self.damping_c


PropType19 = PropSpringTors


@dataclass
class PropSpringPull:
    """``/PROP/TYPE47`` or ``/PROP/SPR_PULL/id`` (M207): Tension-only pulling spring property."""
    id: int = 1
    title: str = ""
    mass: float = 0.0
    stiffness_k: float = 0.0
    damping_c: float = 0.0
    fmax: float = 0.0
    fcut: float = 0.0

    @property
    def k(self) -> float:
        return self.stiffness_k

    @property
    def c(self) -> float:
        return self.damping_c


@dataclass
class PropSpringPush:
    """``/PROP/TYPE48`` or ``/PROP/SPR_PUSH/id`` (M207): Compression-only pushing spring property."""
    id: int = 1
    title: str = ""
    mass: float = 0.0
    stiffness_k: float = 0.0
    damping_c: float = 0.0
    fmax: float = 0.0
    fcut: float = 0.0

    @property
    def k(self) -> float:
        return self.stiffness_k

    @property
    def c(self) -> float:
        return self.damping_c


PropType47 = PropSpringPull
PropType48 = PropSpringPush


@dataclass
class PropType54Layer:
    """Layer definition for ``/PROP/TYPE54`` (M208)."""
    phi: float = 0.0
    thick: float = 0.0
    zi: float = 0.0
    mat_id: int = 0


@dataclass
class PropType54:
    """``/PROP/TYPE54`` or ``/PROP/TSH_P54/prop_ID`` (M208): Layered composite thick shell property."""
    id: int = 1
    isolid: int = 15
    ismstr: int = 0
    icstr: int = 0
    inpts_r: int = 2
    inpts_s: int = 2
    inpts_t: int = 2
    iint: int = 1
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    ashear: float = 5.0 / 6.0
    layers: List[PropType54Layer] = field(default_factory=list)
    deltat_min: float = 0.0
    title: str = ""


PropTshP54 = PropType54
