"""Monitored volumes, airbag chambers, jets and vents.

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
class MonitoredVolume:
    """A /MONVOL monitored volume (e.g., AIRBAG1)."""
    id: int
    title: str = ""
    vol_type: str = "AIRBAG1"  # "AIRBAG1", "COMMU", etc.
    surf_id: int = 0
    hconv: float = 0.0
    
    # Material properties
    matid: int = 0
    mu: float = 0.0
    pext: float = 0.0
    t_initial: float = 293.0
    iequil: int = 0
    ittf: int = 0
    
    # Scaling factors (typically 1.0)
    scale_t: float = 1.0
    scale_p: float = 1.0
    scale_s: float = 1.0
    scale_a: float = 1.0
    scale_d: float = 1.0

    # Injectors (jets)
    injectors: List[Dict] = field(default_factory=list)
    # Ventholes and porous surfaces
    vents: List[Dict] = field(default_factory=list)
    porous_surfaces: List[Dict] = field(default_factory=list)


@dataclass
class InivolContainer:
    """Container surface entry for /INIVOL (M94/M151)."""
    surf_id: int
    ale_phase: int = 1
    fill_opt: int = 0       # 0 = along normal, 1 = against normal (reversed)
    icumu: int = 0          # 0 = erase, 1 = additive, -1 = subtractive
    fill_ratio: float = 1.0 # filling volume fraction in [0, 1]

    @property
    def submat_id(self) -> int:
        return self.ale_phase

    @submat_id.setter
    def submat_id(self, val: int) -> None:
        self.ale_phase = val

    @property
    def ireversed(self) -> int:
        return self.fill_opt

    @ireversed.setter
    def ireversed(self, val: int) -> None:
        self.fill_opt = val

    @property
    def vfrac(self) -> float:
        return self.fill_ratio

    @vfrac.setter
    def vfrac(self, val: float) -> None:
        self.fill_ratio = val


@dataclass
class InitialVolume:
    """/INIVOL (M94/M151): initial volume fraction for multi-material fluid / ALE.

    Fortran origin: ``starter/source/initial_conditions/inivol/hm_read_inivol.F90``.
    """
    id: int
    part_id: int = 0
    title: str = ""
    containers: List[InivolContainer] = field(default_factory=list)


Inivol = InitialVolume


# ----------------------------------------------------------------------------
# Extended Control Volumes, Airbag Leakage & ALE Grid Controls (M105)
# ----------------------------------------------------------------------------

@dataclass
class MonvolPres:
    """/MONVOL/PRES (M105): Pressure monitored volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type1.F``.
    """
    id: int
    title: str = ""
    vol_type: str = "PRES"
    surf_id: int = 0
    fscale: float = 1.0
    p_ext: float = 0.0
    fct_id: int = 0


@dataclass
class MonvolGas:
    """/MONVOL/GAS (M105): Monitored gas control volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type10.F``.
    """
    id: int
    title: str = ""
    vol_type: str = "GAS"
    surf_id: int = 0
    heat_t0: float = 0.0
    scal_t: float = 1.0
    scal_p: float = 1.0
    scal_s: float = 1.0
    scal_a: float = 1.0
    scal_d: float = 1.0
    gamma: float = 1.4
    mu: float = 0.0
    trelax: float = 0.0
    tini: float = 293.15
    rho_gas: float = 1.2
    pext: float = 0.0
    pini: float = 0.0
    pmax: float = 0.0
    vinc: float = 0.0
    mini: float = 0.0


@dataclass
class MonvolCommu1:
    """/MONVOL/COMMU1 (M105): Communicating multi-chamber airbag volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type2.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    heat_t0: float = 0.0
    scal_t: float = 1.0
    scal_p: float = 1.0
    scal_s: float = 1.0
    scal_a: float = 1.0
    scal_d: float = 1.0
    mat_id: int = 0
    mu: float = 0.0
    pext: float = 0.0
    t_initial: float = 293.15
    iequil: int = 0
    ittf: int = 0


@dataclass
class MonvolLFluid:
    """/MONVOL/LFLUID (M105): Liquid fluid control volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type11.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    scal_t: float = 1.0
    scal_p: float = 1.0
    rho_fluid: float = 1000.0
    fct_k: int = 0
    fct_mtin: int = 0
    fscale_k: float = 1.0
    fscale_mtin: float = 1.0
    fct_mtout: int = 0
    fct_mpout: int = 0
    fscale_mtout: float = 1.0
    fscale_mpout: float = 1.0
    fct_padd: int = 0
    fct_pmax: int = 0
    fscale_padd: float = 1.0
    fscale_pmax: float = 1.0


@dataclass
class MonvolAirbagJet:
    """Injector jet specification for /MONVOL/AIRBAG or /MONVOL/COMMU."""
    gamma: float = 1.4
    cpa: float = 0.0
    cpb: float = 0.0
    cpc: float = 0.0
    fct_id_mass: int = 0
    iflow: int = 0
    fscale_mass: float = 1.0
    fct_id_t: int = 0
    fscale_t: float = 1.0
    sens_id: int = 0
    ijet: int = 0
    n1: int = 0
    n2: int = 0
    n3: int = 0


@dataclass
class MonvolAirbagVent:
    """Vent hole or porous surface specification for /MONVOL/AIRBAG or /MONVOL/COMMU."""
    surf_id_v: int = 0
    avent: float = 0.0
    bvent: float = 0.0
    tstop: float = 0.0
    tvent: float = 0.0
    dpdef: float = 0.0
    dtpdef: float = 0.0
    fct_id_v: int = 0
    fscale_v: float = 1.0


@dataclass
class MonvolAirbag:
    """/MONVOL/AIRBAG or /MONVOL/TYPE4 (M133): Multi-gas airbag control volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type4.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    scal_t: float = 1.0
    scal_p: float = 1.0
    scal_s: float = 1.0
    scal_a: float = 1.0
    scal_d: float = 1.0
    mu: float = 0.0
    pext: float = 0.0
    t0: float = 293.15
    iequi: int = 0
    ittf: int = 0
    gammai: float = 1.4
    cpai: float = 0.0
    cpbi: float = 0.0
    cpci: float = 0.0
    njet: int = 0
    jets: List[MonvolAirbagJet] = field(default_factory=list)
    nvent: int = 0
    vents: List[MonvolAirbagVent] = field(default_factory=list)


@dataclass
class MonvolCommu:
    """/MONVOL/COMMU or /MONVOL/TYPE5 (M133): Multi-chamber communicating volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type5.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    scal_t: float = 1.0
    scal_p: float = 1.0
    scal_s: float = 1.0
    scal_a: float = 1.0
    scal_d: float = 1.0
    mu: float = 0.0
    pext: float = 0.0
    t0: float = 293.15
    iequi: int = 0
    ittf: int = 0
    gammai: float = 1.4
    cpai: float = 0.0
    cpbi: float = 0.0
    cpci: float = 0.0
    njet: int = 0
    jets: List[MonvolAirbagJet] = field(default_factory=list)
    nvent: int = 0
    vents: List[MonvolAirbagVent] = field(default_factory=list)
    comm_ids: List[int] = field(default_factory=list)


@dataclass
class MonvolPart:
    """/MONVOL/PART (M133): Monitored volume defined by part ID / group."""
    id: int
    title: str = ""
    part_id: int = 0
    grpart_id: int = 0
    monvol_type: str = "PRES"
    pini: float = 0.0


@dataclass
class FvmChamber:
    id: int
    surf_id: int = 0
    mat_id: int = 0
    pext: float = 0.0
    t_initial: float = 293.15
    iequil: int = 0
    volume: float = 0.0
    volume_old: float = 0.0
    area: float = 0.0
    mass: float = 0.0
    energy: float = 0.0
    temperature: float = 293.15
    pressure: float = 0.0
    density: float = 0.0
    gamma: float = 1.4
    cpa: float = 1004.0
    cpb: float = 0.0
    cpc: float = 0.0
    cpd: float = 0.0
    cpe: float = 0.0
    cpf: float = 0.0
    r_spec: float = 287.0
    element_ids: List[int] = field(default_factory=list)


@dataclass
class FvmOrifice:
    id: int = 0
    surf_id: int = 0
    chamber1_id: int = 1
    chamber2_id: int = 2
    area: float = 0.0
    cd: float = 0.8
    pdef: float = 0.0
    dtpdef: float = 0.0
    tstart: float = 0.0
    tstop: float = 1e30
    is_open: bool = True
    fct_t: int = 0
    fct_p: int = 0
    title: str = ""


@dataclass
class FvmVent:
    id: int = 0
    surf_id: int = 0
    chamber_id: int = 1
    iform: int = 1
    avent: float = 1.0
    bvent: float = 0.0
    area: float = 0.0
    cd: float = 0.6
    tstart: float = 0.0
    tstop: float = 1e30
    dpdef: float = 0.0
    dtpdef: float = 0.0
    idtpdef: int = 0
    is_open: bool = True
    fct_t: int = 0
    fct_p: int = 0
    fct_a: int = 0
    fscale_t: float = 1.0
    fscale_p: float = 1.0
    fscale_a: float = 1.0
    fct_v: int = 0
    fscale_v: float = 1.0
    title: str = ""


@dataclass
class FvmPorousSurface:
    id: int = 0
    surf_id: int = 0
    chamber_id: int = 1
    iformps: int = 1
    iblockage: int = 0
    tstart: float = 0.0
    tstop: float = 1e30
    dpdef: float = 0.0
    dtpdef: float = 0.0
    idtpdef: int = 0
    is_open: bool = True
    fct_v: int = 0
    fscale_v: float = 1.0
    title: str = ""
    lr1: float = 0.0
    fthk: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0


@dataclass
class FvmInjector:
    id: int = 0
    inject_id: int = 0
    sens_id: int = 0
    surf_id: int = 0
    chamber_id: int = 1
    fct_vel: int = 0
    scale_vel: float = 1.0
    mass_flow: float = 0.0
    fct_mass: int = 0
    scale_mass: float = 1.0
    temperature: float = 293.15
    fct_temp: int = 0
    scale_temp: float = 1.0
    cpa: float = 1004.0
    cpb: float = 0.0
    cpc: float = 0.0
    cpd: float = 0.0
    cpe: float = 0.0
    cpf: float = 0.0
    r_spec: float = 287.0
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    area: float = 0.0
    is_active: bool = False


@dataclass
class MonvolFvmbag:
    """/MONVOL/FVMBAG, /MONVOL/FVMBAG1, /MONVOL/FVMBAG2 (M108, M111, M583):
    Finite Volume Method Airbag model with multiple chambers, internal orifices,
    external vents, fabric porosity, and gas injectors.
    """
    id: int
    title: str = ""
    vol_type: str = "FVMBAG1"
    surf_id: int = 0
    surf_id_ex: int = 0
    surf_id_in: int = 0
    hconv: float = 0.0
    ih3d: int = 0
    scale_t: float = 1.0
    scale_p: float = 1.0
    scale_s: float = 1.0
    scale_a: float = 1.0
    scale_d: float = 1.0
    mat_id: int = 0
    pext: float = 0.0
    ttot: float = 0.0
    t0: float = 0.0
    t_initial: float = 293.15
    iequil: int = 0
    i_ttf: int = 0
    injectors: List[FvmInjector] = field(default_factory=list)
    vents: List[FvmVent] = field(default_factory=list)
    porous_surfaces: List[FvmPorousSurface] = field(default_factory=list)
    orifices: List[FvmOrifice] = field(default_factory=list)
    chambers: Dict[int, FvmChamber] = field(default_factory=dict)
    kmesh: int = 1
    cgmerg: float = 0.0
    tswitch: float = 1e30
    iswitch: int = 0
    pswitch: float = 0.0
    dtsca: float = 0.9
    dtmin: float = 0.0
    qa: float = 0.0
    qb: float = 0.0
    hmin: float = 0.0
    params: Dict = field(default_factory=dict)
    volume: float = 0.0
    pressure: float = 0.0
    temperature: float = 293.15
    mass: float = 0.0
    energy: float = 0.0
    is_initialized: bool = False

    def __post_init__(self):
        if self.surf_id_ex == 0 and self.surf_id != 0:
            self.surf_id_ex = self.surf_id
        elif self.surf_id == 0 and self.surf_id_ex != 0:
            self.surf_id = self.surf_id_ex
        if self.t0 != 0.0 and self.t_initial == 293.15:
            self.t_initial = self.t0
        elif self.t_initial != 293.15 and self.t0 == 0.0:
            self.t0 = self.t_initial


# Aliases for backwards compatibility with earlier milestones
MonvolFvmBag1 = MonvolFvmbag
MonvolFvmBag2 = MonvolFvmbag


@dataclass
class MonvolArea:
    """/MONVOL/AREA (M115): Monitored volume surface area monitoring."""
    id: int
    title: str = ""
    surf_id_ext: int = 0
    scale_t: float = 1.0
    scale_p: float = 1.0
    scale_s: float = 1.0
    scale_a: float = 1.0
    scale_d: float = 1.0

    @property
    def surf_id(self) -> int:
        return self.surf_id_ext


@dataclass
class AirbagInjector:
    """/AIRBAG/INJECTOR or /INJECTOR (M142): Airbag jetting injector.

    Fortran origin: ``starter/source/airbag/hm_read_injector.F``.
    """
    id: int
    title: str = ""
    sensor_id: int = 0
    ijet: int = 0
    node1: int = 0
    node2: int = 0
    node3: int = 0
    fct_pt: int = 0
    fct_theta: int = 0
    fct_delta: int = 0
    fscale_pt: float = 1.0
    fscale_ptheta: float = 1.0
    fscale_pdelta: float = 1.0


@dataclass
class AirbagVenthole:
    """/AIRBAG/VENTHOLE or /VENTHOLE (M142): Airbag vent hole and membrane burst model.

    Fortran origin: ``starter/source/airbag/hm_read_venthole.F``.
    """
    id: int
    title: str = ""
    surf_vent: int = 0
    iform: int = 1
    avent: float = 0.0
    bvent: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    dpdef: float = 0.0
    dtpdef: float = 0.0
    idtpdef: int = 0
    fct_id_t: int = 0
    fct_id_p: int = 0
    fct_id_a: int = 0
    fscale_t: float = 1.0
    fscale_p: float = 1.0
    fscale_a: float = 1.0




@dataclass
class MonvolComm:
    """``/MONVOL/COMM`` or ``/MONVOL/COMMUNICATION`` (M198): Direct inter-chamber communication between monitored volumes."""
    id: int
    title: str = ""
    monvol1_id: int = 0
    monvol2_id: int = 0
    surface_id: int = 0
    cd: float = 0.0
    a_vent: float = 0.0
    fct_id: int = 0
    sens_id: int = 0
