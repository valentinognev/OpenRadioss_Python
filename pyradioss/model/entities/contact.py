"""Contact interfaces and their stiffness/gap descriptors.

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
class Interface:
    """One /INTER contact interface. ``type`` selects the mechanics:

    * **7**  — penalty node-to-surface (pyradioss/contact/inter_type7.py):
      secondary node group vs main surface; ``grnod_id = 0`` means
      *self-impact* — the secondary side defaults to the nodes of the main
      surface itself, the Radioss single-surface convention;
    * **2**  — tied/kinematic (inter_type2.py): the secondary nodes are
      glued to their main segment for the whole run;
    * **11** — penalty edge-to-edge (inter_type11.py): secondary /LINE
      edges vs main /LINE edges.

    Penalty options (types 7 and 11), following the Radioss cards:

    sens_id (M6): types 7/11 only — the interface is inactive (no
    forces, no dt claim) until /SENSOR sens_id fires.

    istf  : stiffness definition flag —
            0 = main-side element stiffness scaled by ``stfac`` (default),
            1 = ``stfac`` IS the stiffness (a constant spring value),
            2/3/4/5 = combine main-segment and secondary-node stiffness as
            average / max / min / series (K_m*K_s/(K_m+K_s)).
    igap  : 0 = constant gap (``gap``, auto-computed when 0),
            1 = variable gap per pair from element sizes:
            g = g_s(node) + g_m(segment), floored by ``gap`` (Gap_min)
            and optionally capped by ``gap_max``.
            2 = scaled variable gap (scaled by ``fscale_gap``).
            3 = mesh-size limited variable gap.
    stfac : stiffness scale factor — or the stiffness itself for istf=1.
    fric  : Coulomb friction coefficient.
    gap   : constant gap / Gap_min (0 = auto from main element sizes).

    Fortran origin: ``starter/source/interfaces/int07|02|11/hm_read_*.F``
    (the option cards) and the ``INTBUF_TAB`` interface buffers.
    """

    id: int
    type: int = 7
    grnod_id: int = 0     # secondary nodes (7: 0 = self-impact; 2: required; 24: node-to-surface; 16: secondary nodes)
    surf_id: int = 0      # main surface (types 7, 2, 24)
    surf_id1: int = 0     # secondary surface (type 24 surface-to-surface)
    surf_id2: int = 0     # main/secondary surface 2 (types 3, 20, 24)
    line_id1: int = 0     # secondary edges (type 11)
    line_id2: int = 0     # main edges (type 11)
    grbric_id1: int = 0   # secondary brick group (type 17) or main brick group (type 16)
    grbric_id2: int = 0   # main brick group (type 17)
    istf: int = 0
    itied: int = 0        # tied option flag (16, 17)
    lagmul: bool = False  # True if /INTER/LAGMUL
    hertz: bool = False   # True if /INTER/HERTZ
    igap: int = 0
    stfac: float = 1.0
    fric: float = 0.0
    gap: float = 0.0
    gap_max: float = 0.0  # igap=1 cap, 0 = no cap
    fscale_gap: float = 1.0   # Igap 2/3: gap scale factor (Fscale_gap, default 1.0)
    percent_mesh_size: float = 0.4  # Igap 3: mesh-size gap fraction (default 0.4)
    gap_max_m: float = 0.0 # gap_max_m for TYPE24
    dsearch: float = 0.0  # type 2: projection search distance (0 = auto)
    spotflag: int = 0     # type 2: tied rotational kinematics flag (1: tie rotations, 2: shell rotations)
    sens_id: int = 0      # M6: /SENSOR gating (types 7/11)
    multimp: int = 4      # type 10: max average number of impacted main segments
    idel10: int = 0       # type 10: node and segment deletion flag
    tstart: float = 0.0   # type 10: activation start time
    tstop: float = 1e30   # type 10: deactivation stop time
    inactiv: int = 0      # type 10: initial penetration deactivation flag
    stiff_dc: float = 0.05 # type 10: critical damping coefficient on interface stiffness (VISC)
    viss: float = 0.05     # types 7/24: viscous damping ratio (default 0.05 / 5%)
    sort_fact: float = 0.20 # type 10: bucket sorting search factor
    # ---- friction MODELS (M15, Ifric > 0 — contact/friction.py) ----------
    # mfrot = Ifric (the MFROT law), fric_c = C1..C6, ifq = Ifiltr and
    # xfiltr the reader-derived filter coefficient (hm_read_inter_type07.F:
    # IFQ=1 -> Xfreq, IFQ=2 -> 2*pi/Xfreq, IFQ=3 -> 2*pi*Xfreq). On TYPE11
    # these fields are a documented port EXTENSION (the original TYPE11
    # card carries none — see contact/friction.py).
    mfrot: int = 0
    ifq: int = 0
    iform: int = 0
    xfiltr: float = 0.0
    fric_c: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    title: str = ""
    # ---- TYPE18 (M60) / TYPE10 / TYPE19 / TYPE21 / GUIDED_CABLE ---------
    ibag: int = 0
    idel18: int = 0
    idel: int = 0         # type 19: deletion flag
    icurv: int = 0        # type 19: curve geometry flag
    iadm: int = 0         # type 21: admission flag
    grpart_id: int = 0    # guided cable: part group
    istiff: int = 1       # guided cable: stiffness formulation flag
    gap_scale: float = 1.0 # type 19/21/25: scale factor for gap
    gap_min: float = 0.0   # type 19: min gap
    isym: int = 0         # type 20: symmetric contact flag
    iedge: int = 0        # type 20: edge contact flag
    edge_angle: float = 0.0 # type 20: edge angle
    iload: int = 0        # type 14: load formulation flag
    fun_id1: int = 0      # type 14: fct_ID1
    fun_id2: int = 0      # type 14: fct_ID2
    tol: float = 0.0      # type 12: tolerance
    visc: float = 0.0     # type 9: damping viscosity
    radius: float = 0.0   # type 17: contact radius
    stmin: float = 0.0    # type 19/25: min stiffness
    stmax: float = 0.0    # type 19/25: max stiffness
    ifric: int = 0        # type 25: friction flag
    ifiltr: int = 0       # type 25: filter flag
    xfreq: float = 0.0    # type 25: filter cutoff frequency
    isensor: int = 0      # type 25: sensor ID
    fric_id: int = 0      # type 25: friction law ID
    c1: float = 0.0       # type 25: friction constant 1
    c2: float = 0.0       # type 25: friction constant 2
    c3: float = 0.0       # type 25: friction constant 3
    c4: float = 0.0       # type 25: friction constant 4
    c5: float = 0.0       # type 25: friction constant 5
    depth: float = 0.0    # type 21: drawbead depth
    pmax: float = 1e30    # type 21: maximum contact pressure / force limit
    itlim: int = 0        # type 21: tangential force limit flag (0=limited, 1=deactivated)
    fpenmax: float = 1.0  # type 23: max fraction of initial penetration
    params: dict = field(default_factory=dict)  # generic/extended interface parameters


@dataclass
class SubInterface:
    """/INTER/SUB/sub_inter_ID (M100): Contact sub-interface.

    Fortran origin: ``starter/source/interfaces/sub/hm_read_inter_sub.F``.
    """
    id: int
    inter_id: int
    main_id1: int
    second_id: int
    main_id2: int = 0
    title: str = ""


@dataclass
class GuidedCable:
    """/INTER/GUIDED_CABLE/cable_ID or /GUIDED_CABLE/cable_ID (M149): Guided cable sliding interface.

    Fortran origin: ``starter/source/tools/seatbelts/hm_read_guided_cable.F90`` / CFG ``inter_guided_cable.cfg``.
    """
    id: int
    grnod_id: int = 0
    grpart_id: int = 0
    istiff: int = 1
    stfac: float = 1.0
    fric: float = 0.0
    title: str = ""

    @property
    def grnod_main(self) -> int:
        return self.grnod_id

    @property
    def grnod_sub(self) -> int:
        return self.grpart_id

    @property
    def frad(self) -> float:
        return self.stfac


@dataclass
class FrictionPartPair:
    """Connected part pair friction specification for /FRICTION."""
    grpart_id1: int = 0
    grpart_id2: int = 0
    part_id1: int = 0
    part_id2: int = 0
    idir: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    fric: float = 0.0
    vis_f: float = 1.0
    c1_dir2: float = 0.0
    c2_dir2: float = 0.0
    c3_dir2: float = 0.0
    c4_dir2: float = 0.0
    c5_dir2: float = 0.0
    c6_dir2: float = 0.0
    fric_dir2: float = 0.0
    vis_f_dir2: float = 1.0


@dataclass
class FrictionModel:
    """/FRICTION/fric_id (M119): Generalized multi-part and orthotropic friction model."""
    id: int
    title: str = ""
    ifric: int = 0
    ifiltr: int = 0
    xfreq: float = 0.0
    iform: int = 1
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    fric: float = 0.0
    vis_f: float = 1.0
    pairs: list[FrictionPartPair] = field(default_factory=list)


@dataclass
class InterType18:
    """``/INTER/TYPE18``: Fluid-structure Lagrangian-Eulerian/ALE coupling contact interface."""
    id: int = 0
    title: str = ""
    grnod_id: int = 0
    surf_id: int = 0
    grbric_id: int = 0
    istf: int = 0
    igap: int = 0
    multimp: int = 4
    ibag: int = 0
    idel18: int = 0
    iauto: int = 0
    stfac: float = 1.0
    vref: float = 0.0
    gap: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    stiff_dc: float = 0.0
    sort_fact: float = 0.2
    params: dict = field(default_factory=dict)


@dataclass
class InterType10:
    """``/INTER/TYPE10`` (M199): Secondary node group to main surface contact interface."""
    id: int
    title: str = ""
    grnod_id: int = 0
    surf_id: int = 0
    multimp: int = 0
    idel: int = 0
    stfac: float = 1.0
    gap: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    itied: int = 0
    inactiv: int = 0
    stiff_dc: float = 0.0
    sort_fact: float = 0.2
    params: dict = field(default_factory=dict)


@dataclass
class InterType12:
    """``/INTER/TYPE12`` (M199): General sliding/tied surface-to-surface interface with interpolation."""
    id: int
    title: str = ""
    surf_ids: int = 0
    surf_idm: int = 0
    interpol: int = 0
    tol: float = 0.02
    tstart: float = 0.0
    tstop: float = 1.0e30
    itied: int = 0
    bcopt: int = 0
    skew_id: int = 0
    node_c: int = 0
    xc: float = 0.0
    yc: float = 0.0
    zc: float = 0.0
    theta: float = 0.0
    xn: float = 0.0
    yn: float = 0.0
    zn: float = 0.0
    xt: float = 0.0
    yt: float = 0.0
    zt: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.xc, self.yc, self.zc)

    @property
    def normal(self) -> tuple[float, float, float]:
        return (self.xn, self.yn, self.zn)

    @property
    def tangent(self) -> tuple[float, float, float]:
        return (self.xt, self.yt, self.zt)
InterType26 = GuidedCable
InterGuidedCable = GuidedCable


@dataclass
class InterType25:
    """``/INTER/TYPE25`` or ``/INTER/TIED_BREAK/id`` (M206): Tied breakable contact interface."""
    id: int = 1
    title: str = ""
    grnd_id: int = 0
    surf_id: int = 0
    fn_max: float = 0.0
    ft_max: float = 0.0
    wn: float = 0.0
    wt: float = 0.0
    gap: float = 0.0
    stiff: float = 0.0
    ifric: int = 0
    fric: float = 0.0
