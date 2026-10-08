# -*- coding: utf-8 -*-
"""
Property readers - /KEYWORD blocks -> Model.

/PROP/** - the element property cards.

Fortran origin: the ``hm_read_*.F`` routines under
``starter/source/**/reader/``, one routine per keyword, all driven from
``starter/source/starter/lectur.F``. Each reader's docstring names the
Fortran routine it mirrors; ``keywords/dispatch.py`` holds the keyword
-> parser table, and ``input/starter_keywords.py`` is a deprecated shim
over this package.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union
import os

import numpy as np

from ...common.messages import MessageLog
from ...common.tables import FunctTable
from ...model.entities import (
    AddedMass, BoundaryCondition, Box, ConcentratedLoad, Damping, Gravity,
    CentrifugalLoad, ImposedAcceleration, ImposedDisplacement, ImposedTemperature, ImposedVelocity, InitialVelocity, Interface, Line,
    Material, Mpc, NodeGroup, Part, PressureLoad, Property, Random, Rbe3, RigidBody,
    RigidWall, Section, MonitoredVolume, Sensor, Subdomain, Submodel, Surface, Table, THRequest, Xref,
    DetonatorPoint, DetonatorPlane,
    ConvectionLoad, InivolContainer, InitialVolume,
    RadiationLoad, ImposedFlux, InitialTemperature,
    InitialBrickState, InitialShellState,
    InitialTrussState, InitialBeamState, InitialSpringState,
    BcsNrf, BcsWall, RigidLink, CylJoint, GJoint, GeneralJoint,
    MergeNode, MergeRbody, IniCrack, IniCrackSegment, LaserLoad,
    PcylLoad, PfluidLoad, Preload, PreloadAxial, DampInter, DampRange,
    AnalyGlobal, UpwindGlobal, CaaControl,
    Gauge, Cluster, ExtLink, FxBody, IniGrav, IniMap1D, IniMap2D, IniStateFile,
    MonvolPres, MonvolGas, MonvolCommu1, MonvolLFluid, LeakMat,
    MonvolAirbag, MonvolAirbagJet, MonvolAirbagVent, MonvolCommu, MonvolPart,
    DampGlobal, DampPart, TransformProjection, TransformFrame,
    LoadGravity, LoadBody, LoadTherm, EulerBcs, HeatBcs,
    RwallBox, RwallCone, SectBox, SectCut,
    InivelPart, InivelSph, DetLine, DetCirc, IniMap3D, SetGeneric,
    AleGrid, AleLink, AleSolver, AleClose,
    Retractor, Slipring, UserWindow,
    Drape, IniBriEref, IncludeDyna, MonvolFvmBag1, MonvolFvmbag,
    FvmInjector, FvmVent, FvmOrifice, FvmPorousSurface, FvmChamber,
    GaugePoint, SphGlo, AnalyOptions, AleCfdSph,
    FailOrthBiquad, SlipringShell,
    Upbeam, RelaxSystem, MonvolComm,
)
from ...model.model import Model
from ...model.skew import SkewFrame
from .. import mat_reader
from ..card_layouts import LAYOUTS
from ..card_layouts import CARD_LAYOUTS
from ..card_layouts import split_fixed
from ..deck_reader import Card, KeywordBlock, _to_float, parse_fortran_float


# ----------------------------------------------------------------------------
# Small helpers shared by the parsers
# ----------------------------------------------------------------------------





def read_prop(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE<n>/prop_ID`` (aliases SHELL, TRUSS, SPRING, SOLID).

    TYPE1 / SHELL  (Fortran starter/source/properties/p01_shell)::

        card 1:  prop_title
        card 2:  Ishell  Ismstr  Ish3n  Idrill        (ints — only read,
                 the port always uses the Belytschko–Tsay formulation)
        card 3:  hm   hf   hr   dm   dn               (hourglass coefficients:
                 membrane, flexural, rotational; d* damping — dm/dn ignored)
        card 4:  N   Istrain   Thick                  (N = through-thickness
                 integration points, default 3; Thick = shell thickness)

      Cards 2 and 3 may be omitted **only together with everything after
      them**, so in practice give all 4 cards. To keep tiny decks easy, a
      block whose first data card holds a float that is not an int is
      interpreted as the short form:   card 2: Thick  [N]  [hm]

    TYPE2 / TRUSS::   card 1: title,  card 2: Area

    TYPE3 / BEAM  (Fortran starter/source/properties/p03_beam)::

        card 1:  prop_title
        card 2:  Ishear  dm  df       (flags/damping — read and ignored:
                 the port always includes Timoshenko shear, no damping)
        card 3:  Area   Iyy   Izz   Ixx

      Iyy/Izz = bending inertias about the local y/z axes, Ixx = torsion
      constant. Ixx = 0 defaults to Iyy + Izz (polar, exact for circular
      sections only — give the real torsion constant for others).
      Short form: a single data card 'Area Iyy Izz Ixx'.

    TYPE4 / SPRING:: card 1: title,  card 2: Mass  K  C
      (linear spring: F = K*dl + C*dl_dot; Mass is lumped half/half)

    TYPE14 / SOLID:: card 1: title,
        card 2:  Isolid  Ismstr  ...  (ints — read and ignored: 1-point +
                 Flanagan–Belytschko hourglass is the only ported option)
        card 3:  qa   qb   h          (bulk viscosity quadratic/linear,
                 hourglass coefficient; defaults 1.1 / 0.05 / 0.1)
      Short form: a single data card with 'qa qb h' floats, or none at all
      (all defaults).
    """
    from ...model.entities import Property
    if block.key0.startswith("PROP_") and len(block.key0) > 5:
        typename = block.key0[5:].upper()
    elif len(block.parts) > 1:
        typename = block.parts[1].upper()
    else:
        typename = ""
    if typename in ("INJECTOR", "INJECT", "JET"):
        read_airbag_injector(block, model, log)
        return
    if typename in ("VENTHOLE", "VENT", "VENT_POROUS"):
        read_airbag_venthole(block, model, log)
        return
    if typename in ("FASTENER",):
        read_prop_rivet(block, model, log)
        return
    if typename in ("XFEM",):
        read_prop_xelem(block, model, log)
        return
    if typename in ("TYPE26", "SPR_TAB", "PROP_TYPE26", "PROP_SPR_TAB", "P26_SPR_TAB"):
        read_prop_spr_tab(block, model, log)
        return
    if typename in ("TYPE27", "SPR_BDAMP", "PROP_TYPE27", "PROP_SPR_BDAMP", "P27_SPR_BDAMP"):
        read_prop_spr_bdamp(block, model, log)
        return
    if typename in ("TYPE11", "SH_SANDW", "SANDWICH", "PROP_TYPE11", "PROP_SH_SANDW", "P11_SH_SANDW"):
        read_prop_type11(block, model, log)
        return
    if typename in ("TYPE16", "SH_FABR", "FABRIC_SHELL", "FABRIC", "PROP_TYPE16", "PROP_SH_FABR", "P16_SH_FABR"):
        read_prop_type16(block, model, log)
        return
    if typename in ("TYPE17", "STACK", "COMP_STACK", "PROP_TYPE17", "PROP_STACK", "P17_STACK"):
        read_prop_type17(block, model, log)
        return
    if typename in ("TYPE18", "INT_BEAM", "BEAM_INT", "PROP_TYPE18", "PROP_INT_BEAM", "P18_INT_BEAM", "PROP_P18_INT_BEAM"):
        read_prop_type18(block, model, log)
        return
    if typename in ("TYPE19", "SPR_TORS", "TORSION", "PROP_TYPE19", "PROP_SPR_TORS", "P19_SPR_TORS"):
        read_prop_type19(block, model, log)
        return
    if typename in ("TYPE44", "SPR_CRUS", "CRUSH_SPRING", "SPRING_CRUSH", "PROP_TYPE44", "PROP_SPR_CRUS", "P44_SPR_CRUS"):
        read_prop_type44(block, model, log)
        return
    # M185: PROP_TYPE12 (SPR_PUL), PROP_TYPE15 (POROUS), PROP_TYPE28 (NSTRAND)
    if typename in ("TYPE12", "SPR_PUL", "PULLEY", "PROP_TYPE12", "PROP_SPR_PUL", "P12_SPR_PUL"):
        read_prop_type12(block, model, log)
        return
    if typename in ("SPR_PULL", "PROP_SPR_PULL", "P13_SPR_PULL"):
        read_prop_type13(block, model, log)
        return
    if typename in ("SPR_BEAM", "PROP_SPR_BEAM", "P13_SPR_BEAM"):
        from ..prop_reader import parse_spr_beam
        prop = parse_spr_beam(block, log)
        if prop is not None:
            model.properties[block.user_id] = prop
        return
    if typename in ("TYPE13", "PROP_TYPE13"):
        _t, _cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        _valid = [c for c in _cards if not c.is_blank and not c.raw.strip().startswith("#")]
        if len(_valid) <= 1:
            read_prop_type13(block, model, log)
            return
        from ..prop_reader import parse_spr_beam
        prop = parse_spr_beam(block, log)
        if prop is not None:
            model.properties[block.user_id] = prop
        return
    if typename in ("TYPE23", "SPR_MAT", "PROP_TYPE23", "PROP_SPR_MAT", "P23_SPR_MAT"):
        read_prop_type23(block, model, log)
        return
    if typename in ("TYPE15", "POROUS", "SOLID_POROUS", "PROP_TYPE15", "PROP_POROUS", "P15_POROUS"):
        read_prop_type15(block, model, log)
        return
    if typename in ("TYPE28", "NSTRAND", "STRAND", "PROP_TYPE28", "PROP_NSTRAND", "P28_NSTRAND"):
        read_prop_type28(block, model, log)
        return
    # M186: PROP_TYPE33 (KJOINT), PROP_TYPE46 (SPR_MUSCLE), PROP_TYPE35 (STITCH)
    if typename in ("TYPE33", "KJOINT", "KINEMATIC_JOINT", "PROP_TYPE33", "PROP_KJOINT", "PROP_KINEMATIC_JOINT", "P33_KJOINT"):
        read_prop_type33(block, model, log)
        return
    if typename in ("TYPE46", "SPR_MUSCLE", "MUSCLE", "PROP_TYPE46", "PROP_SPR_MUSCLE", "PROP_MUSCLE", "P46_SPR_MUSCLE"):
        read_prop_type46(block, model, log)
        return
    if typename in ("TYPE35", "STITCH", "SEW", "PROP_TYPE35", "PROP_STITCH", "PROP_SEW", "P35_STITCH"):
        read_prop_type35(block, model, log)
        return
    # M187: PROP_TYPE45 (KJOINT2), PROP_TYPE36 (PREDIT)
    if typename in ("TYPE45", "KJOINT2", "KINEMATIC_JOINT2", "PROP_TYPE45", "PROP_KJOINT2", "PROP_KINEMATIC_JOINT2", "P45_KJOINT2"):
        read_prop_type45(block, model, log)
        return
    if typename in ("TYPE36", "PREDIT", "PROP_TYPE36", "PROP_PREDIT", "P36_PREDIT", "DELAMINATION"):
        read_prop_type36(block, model, log)
        return
    # M188: PROP_TYPE9 (SH_ORTH), PROP_TYPE10 (SH_COMP), PROP_TYPE51 (SH_COH), PROP_TYPE5 (RIVET), PROP_TYPE6 (SOL_ORTH), PROP_TYPE20 (TSHELL)
    if typename in ("TYPE9", "SH_ORTH", "SHELL_ORTH", "PROP_TYPE9", "PROP_SH_ORTH", "PROP_SHELL_ORTH", "P9_SH_ORTH", "PROP_P9_SH_ORTH"):
        read_prop_type9(block, model, log)
        return
    if typename in ("TYPE10", "SH_COMP", "SHELL_COMP", "PROP_TYPE10", "PROP_SH_COMP", "PROP_SHELL_COMP", "P10_SH_COMP", "PROP_P10_SH_COMP"):
        read_prop_type10(block, model, log)
        return
    if typename in ("TYPE51", "SH_COH", "SHELL_COH", "COHESIVE", "PROP_TYPE51", "PROP_SH_COH", "PROP_SHELL_COH", "PROP_COHESIVE", "P51_SH_COH", "PROP_P51_SH_COH"):
        read_prop_type51(block, model, log)
        return
    if typename in ("TYPE5", "RIVET", "PROP_TYPE5", "PROP_RIVET", "P5_RIVET", "PROP_P5_RIVET"):
        read_prop_type5(block, model, log)
        return
    if typename in ("TYPE6", "SOL_ORTH", "SOLID_ORTH", "PROP_TYPE6", "PROP_SOL_ORTH", "PROP_SOLID_ORTH", "P6_SOL_ORTH", "PROP_P6_SOL_ORTH"):
        read_prop_type6(block, model, log)
        return
    # NOTE: INJECT1/2, JOINT, TORSION, SPR_ELAS_PLAS, SPR_BEAM, SPOTWELD,
    # BUSHING — all handled by the generic cfg-driven prop_reader below.
    # Dedicated readers (read_prop_inject1 etc.) exist but are not yet
    # wired here because existing tests expect model.properties[id].

    aliases = {"TYPE1": 1, "SHELL": 1, "PROP_P1_SHELL": 1, "P1_SHELL": 1,
               "TYPE2": 2, "TRUSS": 2,
               "TYPE3": 3, "BEAM": 3,
               "TYPE4": 4, "SPRING": 4, "TYPE14": 14, "SOLID": 14, "SOL_GENE": 14, "PROP_SOLID": 14, "PROP_SOL_GENE": 14, "PROP_P14_SOLID": 14, "P14_SOLID": 14,
               "TYPE5": 5, "RIVET": 5, "PROP_RIVET": 5, "PROP_P5_RIVET": 5,
               "TYPE28": 28, "XELEM": 28, "PROP_XELEM": 28, "PROP_P28_XELEM": 28,
               "TYPE9": 9, "SH_ORTH": 9, "PROP_P9_SH_ORTH": 9, "P9_SH_ORTH": 9, "PROP_SH_ORTH": 9,
               "TYPE10": 10, "SH_COMP": 10, "PROP_P10_SH_COMP": 10, "P10_SH_COMP": 10,
               "TYPE11": 11, "SH_SANDW": 11,
               "TYPE16": 16, "SH_FABR": 16,
               "TYPE6": 6, "SOL_ORTH": 6,
               "TYPE20": 20, "TSHELL": 20,
               "TYPE21": 21, "TSH_ORTH": 21,
               "TYPE22": 22, "TSH_COMP": 22,
               "TYPE18": 18, "INT_BEAM": 18, "PROP_P18_INT_BEAM": 18, "P18_INT_BEAM": 18, "BEAM_INT": 18,
               "TYPE19": 19, "SPR_TORS": 19, "PROP_SPR_TORS": 19, "PROP_P19_SPR_TORS": 19, "P19_SPR_TORS": 19,
               "TYPE8": 8, "SPR_GENE": 8, "SPRING_GENE": 8, "PROP_P8_SPR_GENE": 8, "P8_SPR_GENE": 8, "PROP_SPR_GENE": 8, "PROP_SPRING_GENE": 8,
               "TYPE12": 12, "SPR_PUL": 12, "PROP_P12_SPR_PUL": 12, "P12_SPR_PUL": 12, "PROP_SPR_PUL": 12, "SPR_PULL": 13, "PROP_SPR_PULL": 13,
               "TYPE13": 13, "SPR_BEAM": 13, "PROP_P13_SPR_BEAM": 13, "P13_SPR_BEAM": 13, "PROP_SPR_BEAM": 13,
               "TYPE15": 15, "POROUS": 15, "PROP_P15_POROUS": 15, "P15_POROUS": 15, "PROP_POROUS": 15,
               "TYPE23": 23, "SPR_MAT": 23, "PROP_P23_SPR_MAT": 23, "P23_SPR_MAT": 23, "PROP_SPR_MAT": 23,
               "TYPE25": 25, "SPR_AXI": 25, "SPRING_AXI": 25, "PROP_P25_SPR_AXI": 25, "P25_SPR_AXI": 25, "PROP_SPR_AXI": 25, "PROP_SPRING_AXI": 25,
               "TYPE32": 32, "SPR_PRE": 32, "SPRING_PRE": 32, "PROP_P32_SPR_PRE": 32, "P32_SPR_PRE": 32, "PROP_SPR_PRE": 32, "PROP_SPRING_PRE": 32,
               "TYPE26": 26, "SPR_TAB": 26, "PROP_P26_SPR_TAB": 26, "P26_SPR_TAB": 26, "PROP_SPR_TAB": 26,
               "TYPE27": 27, "SPR_BDAMP": 27, "PROP_P27_SPR_BDAMP": 27, "P27_SPR_BDAMP": 27, "PROP_SPR_BDAMP": 27,
               "TYPE34": 34, "SPH": 34, "USER_SOLID": 34, "PROP_USER_SOLID": 34, "PROP_P34_USER": 34, "PROP_SPH": 34,
               "USER_SPRING": 4, "PROP_USER_SPRING": 4, "PROP_P4_USER": 4,
               "TYPE43": 43, "CONNECT": 43, "PROP_CONNECT": 43, "PROP_TYPE43": 43, "PROP_P43_CONNECT": 43, "P43_CONNECT": 43,
               "TYPE17": 17, "STACK": 17, "PROP_STACK": 17,
               "TYPE51": 51, "P51": 51, "LAMINATE_P51": 51,
               "TYPE0": 0, "VOID": 0}
    if typename in ("PCOMPP", "PROP_PCOMPP", "PCOMP_P", "PROP_PCOMP_P"):
        read_prop_pcompp(block, model, log)
        return
    if typename in ("TYPE51", "P51", "PROP_P51", "PROP_TYPE51", "TSH_P51", "PROP_TSH_P51", "LAMINATE_P51"):
        read_prop_p51(block, model, log)
        return
    if typename not in aliases:
        if typename in ("INJECT1", "PROP_INJECT1", "INJECTOR1", "PROP_INJECTOR1"):
            read_prop_inject1(block, model, log)
            return
        elif typename in ("INJECT2", "PROP_INJECT2", "INJECTOR2", "PROP_INJECT2"):
            read_prop_inject2(block, model, log)
            return
        elif typename in ("TYPE29", "PROP_TYPE29"):
            from ...model.entities import PropType29
            pid = block.user_id or 1
            title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
            card_lines = [c.raw for c in cards if not c.is_blank]
            model.prop_type29s[pid] = PropType29(id=pid, cards=card_lines, title=title)
            from .. import prop_reader
            model.properties[pid] = prop_reader.InactiveProperty(id=pid, type=29, title=title, prop_name="TYPE29")
            return
        elif typename in ("TYPE30", "PROP_TYPE30"):
            from ...model.entities import PropType30
            pid = block.user_id or 1
            title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
            card_lines = [c.raw for c in cards if not c.is_blank]
            model.prop_type30s[pid] = PropType30(id=pid, cards=card_lines, title=title)
            from .. import prop_reader
            model.properties[pid] = prop_reader.InactiveProperty(id=pid, type=30, title=title, prop_name="TYPE30")
            return
        elif typename in ("TYPE31", "PROP_TYPE31"):
            from ...model.entities import PropType31
            pid = block.user_id or 1
            title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
            card_lines = [c.raw for c in cards if not c.is_blank]
            model.prop_type31s[pid] = PropType31(id=pid, cards=card_lines, title=title)
            from .. import prop_reader
            model.properties[pid] = prop_reader.InactiveProperty(id=pid, type=31, title=title, prop_name="TYPE31")
            return
        from .. import prop_reader
        prop = prop_reader.parse_property(block, log)
        if prop is not None:
            model.properties[block.user_id] = prop
        return
    ptype = aliases[typename]
    from ...model.entities import Property
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    params: Dict[str, float] = {}

    if ptype == 1:  # SHELL
        def_sh = getattr(model, "def_shell", {}) or {}
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01,
                  "ishell": def_sh.get("ishell", 0)}
        if "ish3n" in def_sh:
            params["ish3n"] = def_sh["ish3n"]
        if block.fixed:
            # REAL layout (cfg prop_p1_shell.cfg radioss2020; M37):
            # flags / Hm Hf Hr Dm Dn / N Istrain Thick Ashear Ithick
            # Iplas — column-cut: blank cards/fields are the defaults
            # (a token view read Thick from the wrong position when
            # Istrain was blank, or died on the 'thickness card
            # missing' guard when the hourglass card was blank)
            if cards and not cards[0].is_blank:
                f = cards[0].cut("PROP_SHELL_FLAGS")
                if f[0].strip():
                    params["ishell"] = _ival(f[0])
                if len(f) > 2 and f[2].strip():
                    params["ish3n"] = _ival(f[2])
            hm_d, hf_d, hr_d = _hourglass_defaults(params["ishell"])
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or hm_d
                params["hf"] = _fval(h[1]) or hf_d
                params["hr"] = _fval(h[2]) or hr_d
                # dn (5th field): the BATOZ-family numerical damping that
                # enters the dt claim (cncoef3.F AMU -> cndt3.F VISCMX;
                # zero -> the formulation default, see shell_bt4, M40)
                params["dn"] = _fval(h[4])
            else:
                params["hm"], params["hf"], params["hr"] = hm_d, hf_d, hr_d
            if len(cards) >= 3:
                f = cards[2].cut("PROP_SHELL_N")
                params["nip"] = _ival(f[0]) or 3
                thick = _fval(f[2])
                if thick <= 0.0:
                    log.error(f"/PROP/SHELL/{block.user_id}: Thick must "
                              f"be > 0 (blank = 0 in the real layout; "
                              f"per-/PART thickness is not ported)",
                              block.source)
                params["thick"] = thick
            else:
                log.error(f"/PROP/SHELL/{block.user_id}: thickness card "
                          f"missing", block.source)
            model.properties[block.user_id] = Property(
                id=block.user_id, type=ptype, title=title, params=params)
            return
        # Detect short form: single data card or first card contains a float.
        if cards and (len(cards) == 1 or any("." in tok or "e" in tok.lower()
                                             for tok in cards[0].tokens())):
            vals = _floats(cards[0], 3, defaults=[1.0, 3, 0.01])
            params["thick"], params["nip"], params["hm"] = \
                vals[0], int(vals[1]) if vals[1] else 3, vals[2] or 0.01
            params["hf"] = params["hr"] = params["hm"]
        else:
            # full form: flags card carries Ishell, then hourglass + N/Thick
            toks = cards[0].tokens() if cards else []
            if toks:
                try:
                    params["ishell"] = int(toks[0])
                except ValueError:
                    params["ishell"] = 0
                if len(toks) > 2:
                    try:
                        params["ish3n"] = int(toks[2])
                    except ValueError:
                        params["ish3n"] = 0
                else:
                    params["ish3n"] = 0
            else:
                params["ish3n"] = 0
            hm_d, hf_d, hr_d = _hourglass_defaults(params["ishell"])
            if len(cards) >= 2:
                hm, hf, hr = _floats(cards[1], 3,
                                     defaults=[hm_d, hf_d, hr_d])[:3]
                params["hm"], params["hf"], params["hr"] = \
                    hm or hm_d, hf or hf_d, hr or hr_d
                # dn (5th field) — BATOZ-family numerical damping (M40)
                params["dn"] = _floats(cards[1], 5)[4]
            else:
                params["hm"], params["hf"], params["hr"] = hm_d, hf_d, hr_d
            if len(cards) >= 3:
                v = _floats(cards[2], 3, defaults=[3, 0, 1.0])
                params["nip"] = int(v[0]) if v[0] else 3
                params["thick"] = v[2]
            else:
                log.error(f"/PROP/SHELL/{block.user_id}: thickness card "
                          f"missing", block.source)
    elif ptype == 2:  # TRUSS
        if not cards:
            log.error(f"/PROP/TRUSS/{block.user_id}: area card missing",
                      block.source)
            return
        fl = cards[0].floats()
        if not fl and block.fixed:
            fl = [f for f in _cut_floats(cards[0], "F20X2") if f is not None]
        if not fl:
            log.error(f"/PROP/TRUSS/{block.user_id}: area card missing",
                      block.source)
            return
        params = {"area": fl[0]}
    elif ptype == 3:  # BEAM
        if block.fixed:
            # REAL layout (cfg prop_p3_beam.cfg): title / Ismstr /
            # Dm Df / Area Iyy Izz Ixx / OmegaDof Ishear — the section
            # card is data card index 2.
            # If short form (1 data card), read card 0.
            sec = None
            if len(cards) == 1 and not cards[0].is_blank:
                sec = cards[0]
            elif len(cards) >= 3 and not cards[2].is_blank:
                sec = cards[2]
            elif len(cards) >= 2 and not cards[1].is_blank and (any("." in tok for tok in cards[1].tokens()) or len(cards[1].tokens()) >= 3):
                sec = cards[1]
            elif cards and not cards[0].is_blank:
                sec = cards[0]
            if sec is None:
                log.error(f"/PROP/BEAM/{block.user_id}: section card "
                          f"'Area Iyy Izz Ixx' missing", block.source)
                return
            a, iyy, izz, ixx = _cut_floats(sec, "F20X4")
        else:
            # Free format: support both short form (1 card: Area Iyy Izz Ixx)
            # and multi-card standard form (card with 3-4 section floats).
            sec_card = None
            if len(cards) == 1:
                sec_card = cards[0]
            else:
                candidates = [c for c in cards if len(c.tokens()) >= 3]
                if candidates:
                    sec_card = candidates[-1]
                else:
                    data = [c for c in cards if not all(tok.lstrip("+-").isdigit()
                                                        for tok in c.tokens())]
                    if data:
                        sec_card = data[0]
            if sec_card is None:
                log.error(f"/PROP/BEAM/{block.user_id}: section card "
                          f"'Area Iyy Izz Ixx' missing", block.source)
                return
            vals = _floats(sec_card, 4)
            a = vals[0] if len(vals) > 0 and vals[0] is not None else 0.0
            iyy = vals[1] if len(vals) > 1 and vals[1] is not None else 0.0
            izz = vals[2] if len(vals) > 2 and vals[2] is not None else 0.0
            ixx = vals[3] if len(vals) > 3 and vals[3] is not None else 0.0
        if a <= 0 or iyy <= 0 or izz <= 0:
            log.error(f"/PROP/BEAM/{block.user_id}: Area, Iyy and Izz "
                      f"must be > 0", block.source)
            return
        params = {"area": a, "iyy": iyy, "izz": izz,
                  "ixx": ixx if ixx > 0 else iyy + izz}
    elif ptype == 4:  # SPRING or USER_SPRING
        if typename in ("USER_SPRING", "PROP_USER_SPRING", "PROP_P4_USER"):
            nuvar = 0
            stif_inter = 0.0
            skew_id = 0
            if cards and not cards[0].is_blank:
                if block.fixed:
                    f = cards[0].cut("PROP_USER_SPRING_1")
                    nuvar = _ival(f[0]) if len(f) > 0 else 0
                    stif_inter = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                    skew_id = _ival(f[2]) if len(f) > 2 else 0
                else:
                    toks = cards[0].tokens()
                    nuvar = int(float(toks[0])) if len(toks) > 0 else 0
                    stif_inter = float(toks[1]) if len(toks) > 1 else 0.0
                    skew_id = int(float(toks[2])) if len(toks) > 2 else 0
            params = {"nuvar": nuvar, "stif_inter": stif_inter, "skew_id": skew_id}
        else:
            if not cards:
                log.error(f"/PROP/SPRING/{block.user_id}: data card missing",
                          block.source)
                return
            if len(cards) >= 2 and len(cards[0].tokens()) <= 2:
                m = _floats(cards[0], 1)[0]
                f1 = _floats(cards[1], 2)
                k = f1[0] if len(f1) > 0 else 0.0
                c = f1[1] if len(f1) > 1 else 0.0
            else:
                m, k, c = _floats(cards[0], 3)
            params = {"mass": m, "k": k, "c": c}
    elif ptype == 5:  # RIVET
        wflag = 0
        imod = 1
        fn = 0.0
        ft = 0.0
        dx = 0.0
        mass = 0.0
        stiffness = 0.0
        fn_fail = 0.0
        ft_fail = 0.0
        if len(cards) == 1 and len(cards[0].tokens()) >= 4:
            t = cards[0].tokens()
            mass = float(t[0])
            stiffness = float(t[1])
            fn_fail = float(t[2])
            ft_fail = float(t[3])
            fn = fn_fail
            ft = ft_fail
        elif block.fixed:
            if cards and not cards[0].is_blank:
                f1 = cards[0].cut("PROP_RIVET_1")
                wflag = _ival(f1[0]) if len(f1) > 0 else 0
                imod = _ival(f1[1], 1) if len(f1) > 1 else 1
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("PROP_RIVET_2")
                fn = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                ft = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                dx = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
                fn_fail = fn
                ft_fail = ft
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            wflag = int(float(t1[0])) if len(t1) > 0 else 0
            imod = int(float(t1[1])) if len(t1) > 1 else 1
            t2 = cards[1].tokens() if len(cards) > 1 else []
            fn = float(t2[0]) if len(t2) > 0 else 0.0
            ft = float(t2[1]) if len(t2) > 1 else 0.0
            dx = float(t2[2]) if len(t2) > 2 else 0.0
            fn_fail = fn
            ft_fail = ft
        params = {
            "wflag": wflag, "imod": imod, "fn": fn, "ft": ft, "dx": dx,
            "mass": mass, "stiffness": stiffness, "fn_fail": fn_fail, "ft_fail": ft_fail,
        }
    elif ptype == 28:  # XELEM
        mass = 0.0
        k = 0.0
        c = 0.0
        dmin = -1.0e30
        dmax = 1.0e30
        fun_k = 0
        fun_c = 0
        fscale_y = 1.0
        fscale_x = 1.0
        nip = 0
        mu1 = 0.0
        mu2 = 0.0
        itip = 0
        isurf = 0
        alpha = 0.0
        if len(cards) == 1 and len(cards[0].tokens()) == 3:
            t = cards[0].tokens()
            itip = int(float(t[0]))
            isurf = int(float(t[1]))
            alpha = float(t[2])
            fun_k = itip
            fun_c = isurf
            mu1 = alpha
        elif block.fixed:
            if cards and not cards[0].is_blank:
                f1 = cards[0].cut("PROP_XELEM_1")
                mass = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
                k = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                c = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                dmin = _fval(f1[3], -1.0e30) if len(f1) > 3 else -1.0e30
                dmax = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("PROP_XELEM_2")
                fun_k = _ival(f2[0]) if len(f2) > 0 else 0
                fun_c = _ival(f2[1]) if len(f2) > 1 else 0
                fscale_y = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
                fscale_x = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            if len(cards) > 2 and not cards[2].is_blank:
                f3 = cards[2].cut("PROP_XELEM_3")
                nip = _ival(f3[0]) if len(f3) > 0 else 0
                mu1 = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
                mu2 = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            mass = float(t1[0]) if len(t1) > 0 else 0.0
            k = float(t1[1]) if len(t1) > 1 else 0.0
            c = float(t1[2]) if len(t1) > 2 else 0.0
            dmin = float(t1[3]) if len(t1) > 3 else -1.0e30
            dmax = float(t1[4]) if len(t1) > 4 else 1.0e30
            t2 = cards[1].tokens() if len(cards) > 1 else []
            fun_k = int(float(t2[0])) if len(t2) > 0 else 0
            fun_c = int(float(t2[1])) if len(t2) > 1 else 0
            fscale_y = float(t2[2]) if len(t2) > 2 else 1.0
            fscale_x = float(t2[3]) if len(t2) > 3 else 1.0
            t3 = cards[2].tokens() if len(cards) > 2 else []
            nip = int(float(t3[0])) if len(t3) > 0 else 0
            mu1 = float(t3[1]) if len(t3) > 1 else 0.0
            mu2 = float(t3[2]) if len(t3) > 2 else 0.0
        params = {
            "mass": mass, "k": k, "c": c, "dmin": dmin, "dmax": dmax,
            "fun_k": fun_k, "fun_c": fun_c, "fscale_y": fscale_y,
            "fscale_x": fscale_x, "nip": nip, "mu1": mu1, "mu2": mu2,
            "itip": itip, "isurf": isurf, "alpha": alpha,
        }
    elif ptype == 19:  # SPR_TORS
        mass = 0.0
        k_tors = 0.0
        c_tors = 0.0
        fct_id_k = 0
        fct_id_c = 0
        fscale_k = 1.0
        fscale_c = 1.0
        if block.fixed:
            if cards and not cards[0].is_blank:
                f1 = cards[0].cut("PROP_SPR_TORS_1")
                mass = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
                k_tors = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                c_tors = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("PROP_SPR_TORS_2")
                fct_id_k = _ival(f2[0]) if len(f2) > 0 else 0
                fct_id_c = _ival(f2[1]) if len(f2) > 1 else 0
                fscale_k = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
                fscale_c = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            mass = float(t1[0]) if len(t1) > 0 else 0.0
            k_tors = float(t1[1]) if len(t1) > 1 else 0.0
            c_tors = float(t1[2]) if len(t1) > 2 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            fct_id_k = int(float(t2[0])) if len(t2) > 0 else 0
            fct_id_c = int(float(t2[1])) if len(t2) > 1 else 0
            fscale_k = float(t2[2]) if len(t2) > 2 else 1.0
            fscale_c = float(t2[3]) if len(t2) > 3 else 1.0
        params = {
            "mass": mass, "k_tors": k_tors, "c_tors": c_tors,
            "fct_id_k": fct_id_k, "fct_id_c": fct_id_c,
            "fscale_k": fscale_k, "fscale_c": fscale_c,
            "k": k_tors, "c": c_tors,
        }
    elif ptype == 14:  # SOLID
        from ...common.constants import DEFAULT_HOURGLASS, DEFAULT_QA, DEFAULT_QB
        def_so = getattr(model, "def_solid", {}) or {}
        params = {"qa": DEFAULT_QA, "qb": DEFAULT_QB, "h": DEFAULT_HOURGLASS,
                  "isolid": def_so.get("isolid", 14),
                  "ismstr": def_so.get("ismstr", 0)}
        if block.fixed:
            if cards and not cards[0].is_blank:
                c0 = [cards[0].raw[i:i+10] for i in range(0, min(80, len(cards[0].raw)), 10)]
                if len(c0) > 0 and c0[0].strip():
                    params["isolid"] = _ival(c0[0])
                if len(c0) > 1 and c0[1].strip():
                    params["ismstr"] = _ival(c0[1])
                params["icpre"] = _ival(c0[3]) if len(c0) > 3 and c0[3].strip() else 0
                nbp = _ival(c0[5]) if len(c0) > 5 and c0[5].strip() else 1
                params["inpts_r"] = nbp
                params["inpts_s"] = nbp
                params["inpts_t"] = nbp
                params["i_rot"] = _ival(c0[6]) if len(c0) > 6 and c0[6].strip() else 0
                params["iframe"] = _ival(c0[7]) if len(c0) > 7 and c0[7].strip() else 0
                if len(cards[0].raw) > 80:
                    params["dn"] = _fval(cards[0].raw[80:100])
            if len(cards) >= 2 and not cards[1].is_blank:
                f = cards[1].cut("F20X5")
                params["qa"] = _fval(f[0]) or DEFAULT_QA
                params["qb"] = _fval(f[1]) or DEFAULT_QB
                params["h"] = _fval(f[2]) or DEFAULT_HOURGLASS
            if len(cards) >= 3 and not cards[2].is_blank:
                c2 = cards[2].cut("PROP_TYPE14_3")
                params["deltat_min"] = _fval(c2[0]) if len(c2) > 0 else 0.0
                params["istrain"] = _ival(c2[1]) if len(c2) > 1 else 0
                params["qa_l"] = _fval(c2[2]) if len(c2) > 2 else 0.0
                params["qb_l"] = _fval(c2[3]) if len(c2) > 3 else 0.0
                params["h_l"] = _fval(c2[4]) if len(c2) > 4 else 0.0
                params["iplas"] = _ival(c2[5]) if len(c2) > 5 else 0
                params["icstr"] = _ival(c2[6]) if len(c2) > 6 else 0
        else:
            if len(cards) >= 3:
                t0 = cards[0].tokens()
                if len(t0) > 0:
                    params["isolid"] = int(float(t0[0]))
                if len(t0) > 1:
                    params["ismstr"] = int(float(t0[1]))
                params["icpre"] = int(float(t0[2])) if len(t0) > 2 else 0
                if len(t0) > 7:
                    params["inpts_r"] = int(float(t0[3]))
                    params["inpts_s"] = int(float(t0[4]))
                    params["inpts_t"] = int(float(t0[5]))
                    params["i_rot"] = int(float(t0[6]))
                    params["iframe"] = int(float(t0[7]))
                    params["dn"] = float(t0[8]) if len(t0) > 8 else 0.0
                else:
                    nbp = int(float(t0[3])) if len(t0) > 3 else 1
                    params["inpts_r"] = nbp
                    params["inpts_s"] = nbp
                    params["inpts_t"] = nbp
                    params["i_rot"] = int(float(t0[4])) if len(t0) > 4 else 0
                    params["iframe"] = int(float(t0[5])) if len(t0) > 5 else 0
                    params["dn"] = float(t0[6]) if len(t0) > 6 else 0.0

                t1 = cards[1].tokens()
                params["qa"] = float(t1[0]) if len(t1) > 0 else DEFAULT_QA
                params["qb"] = float(t1[1]) if len(t1) > 1 else DEFAULT_QB
                params["h"] = float(t1[2]) if len(t1) > 2 else DEFAULT_HOURGLASS

                t2 = cards[2].tokens()
                params["deltat_min"] = float(t2[0]) if len(t2) > 0 else 0.0
                params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
                params["qa_l"] = float(t2[2]) if len(t2) > 2 else 0.0
                params["qb_l"] = float(t2[3]) if len(t2) > 3 else 0.0
                params["h_l"] = float(t2[4]) if len(t2) > 4 else 0.0
                params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0
                params["icstr"] = int(float(t2[6])) if len(t2) > 6 else 0
            else:
                flags = [c for c in cards if all(tok.lstrip("+-").isdigit()
                                                 for tok in c.tokens())]
                if flags and flags[0].tokens():
                    toks = flags[0].tokens()
                    params["isolid"] = int(toks[0])
                    if len(toks) > 6:
                        params["itetra4"] = int(toks[6])

                data = [c for c in cards if not all(tok.lstrip("+-").isdigit()
                                                    for tok in c.tokens())]
                if data:
                    qa, qb, h = _floats(data[0], 3,
                                        defaults=[DEFAULT_QA, DEFAULT_QB,
                                                  DEFAULT_HOURGLASS])
                    params.update(qa=qa or DEFAULT_QA, qb=qb or DEFAULT_QB,
                                  h=h or DEFAULT_HOURGLASS)

    elif ptype == 10:  # SH_COMP
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "p_thick_fail": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "nip": 1, "istrain": 0, "thick": 1.0, "ashear": 0.833333, "ithick": 0, "iplas": 0,
                  "vx": 1.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "ip": 0, "phi_layers": []}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SH_COMP_FLAGS")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["p_thick_fail"] = _fval(f[5]) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3])
                params["dn"] = _fval(h[4])
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_SHELL_N")
                params["nip"] = _ival(f[0]) or 1
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["thick"] = _fval(f[2]) or 1.0
                params["ashear"] = _fval(f[3]) or 0.833333
                params["ithick"] = _ival(f[5]) if len(f) > 5 else 0
                params["iplas"] = _ival(f[6]) if len(f) > 6 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_SH_COMP_VEC")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["ip"] = _ival(v[5]) if len(v) > 5 else 0
            phi_list = []
            for c in cards[4:]:
                if c.is_blank:
                    continue
                vals = c.cut("F20X5")
                for val_str in vals:
                    if val_str.strip():
                        phi_list.append(_fval(val_str))
            params["phi_layers"] = phi_list
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["ish3n"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["idrill"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["p_thick_fail"] = float(t0[4]) if len(t0) > 4 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
            params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 1
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
            params["ip"] = int(float(t3[4])) if len(t3) > 4 else 0

            phi_list = []
            for c in cards[4:]:
                for tok in c.tokens():
                    phi_list.append(float(tok))
            params["phi_layers"] = phi_list

    elif ptype == 11:  # SH_SANDW
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "p_thick_fail": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "nip": 1, "istrain": 0, "thick": 1.0, "ashear": 0.833333, "ithick": 0, "iplas": 0,
                  "vx": 1.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "iorth": 0, "ipos": 0, "ip": 0,
                  "layers": []}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SH_COMP_FLAGS")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["p_thick_fail"] = _fval(f[5]) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3])
                params["dn"] = _fval(h[4])
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_SHELL_N")
                params["nip"] = _ival(f[0]) or 1
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["thick"] = _fval(f[2]) or 1.0
                params["ashear"] = _fval(f[3]) or 0.833333
                params["ithick"] = _ival(f[5]) if len(f) > 5 else 0
                params["iplas"] = _ival(f[6]) if len(f) > 6 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_SH_SANDW_VEC")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["iorth"] = _ival(v[4]) if len(v) > 4 else 0
                params["ipos"] = _ival(v[5]) if len(v) > 5 else 0
                params["ip"] = _ival(v[6]) if len(v) > 6 else 0
            layers = []
            for c in cards[4:]:
                if c.is_blank:
                    continue
                ly = c.cut("PROP_SH_SANDW_LAYER")
                layers.append({
                    "phi": _fval(ly[0]),
                    "thick": _fval(ly[1]),
                    "zi": _fval(ly[2]),
                    "mat_id": _ival(ly[3]),
                    "w_fi": _fval(ly[5]) if len(ly) > 5 else 1.0
                })
            params["layers"] = layers
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["ish3n"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["idrill"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["p_thick_fail"] = float(t0[4]) if len(t0) > 4 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
            params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 1
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
            params["iorth"] = int(float(t3[4])) if len(t3) > 4 else 0
            params["ipos"] = int(float(t3[5])) if len(t3) > 5 else 0
            params["ip"] = int(float(t3[6])) if len(t3) > 6 else 0

            layers = []
            for c in cards[4:]:
                toks = c.tokens()
                if not toks:
                    continue
                layers.append({
                    "phi": float(toks[0]) if len(toks) > 0 else 0.0,
                    "thick": float(toks[1]) if len(toks) > 1 else 0.0,
                    "zi": float(toks[2]) if len(toks) > 2 else 0.0,
                    "mat_id": int(float(toks[3])) if len(toks) > 3 else 0,
                    "w_fi": float(toks[4]) if len(toks) > 4 else 1.0
                })
            params["layers"] = layers

    elif ptype == 16:  # SH_FABR
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "p_thick_fail": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0,
                  "nip": 1, "istrain": 0, "thick": 1.0, "ashear": 0.833333, "ithick": 0,
                  "vx": 1.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "ipos": 0, "ip": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SH_FABR_FLAGS")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["p_thick_fail"] = _fval(f[4]) if len(f) > 4 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3])
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_SH_FABR_N")
                params["nip"] = _ival(f[0]) or 1
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["thick"] = _fval(f[2]) or 1.0
                params["ashear"] = _fval(f[3]) or 0.833333
                params["ithick"] = _ival(f[5]) if len(f) > 5 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_SH_FABR_VEC")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["ipos"] = _ival(v[4]) if len(v) > 4 else 0
                params["ip"] = _ival(v[6]) if len(v) > 6 else 0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["ish3n"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["p_thick_fail"] = float(t0[3]) if len(t0) > 3 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 1
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
            params["ipos"] = int(float(t3[4])) if len(t3) > 4 else 0
            params["ip"] = int(float(t3[5])) if len(t3) > 5 else 0

    elif ptype == 6:  # SOL_ORTH
        params = {"isolid": 14, "ismstr": 0, "icpre": 0, "itetra10": 0, "nbp": 0,
                  "inpts_r": 1, "inpts_s": 1, "inpts_t": 1, "itetra4": 0, "iframe": 0, "dn": 0.0,
                  "qa": 1.1, "qb": 0.05, "h": 0.1,
                  "vx": 0.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "ip": 0, "iorth": 0,
                  "phi": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0,
                  "deltat_min": 0.0, "vdef_min": 0.0, "vdef_max": 0.0, "asp_max": 0.0, "col_min": 0.0,
                  "ndir": 0, "sphpart_id": 0, "istrain": 0, "ihkt": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SOL_ORTH_1")
                params["isolid"] = _ival(f[0], 14) if len(f) > 0 and f[0].strip() else 14
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["icpre"] = _ival(f[3]) if len(f) > 3 else 0
                params["itetra10"] = _ival(f[4]) if len(f) > 4 else 0
                nbp = _ival(f[5]) if len(f) > 5 else 0
                params["nbp"] = nbp
                if nbp > 200:
                    params["inpts_r"] = nbp // 100
                    rem = nbp % 100
                    params["inpts_s"] = rem // 10
                    params["inpts_t"] = rem % 10
                elif nbp > 0:
                    params["inpts_r"] = nbp
                    params["inpts_s"] = nbp
                    params["inpts_t"] = nbp
                params["itetra4"] = _ival(f[6]) if len(f) > 6 else 0
                params["iframe"] = _ival(f[7]) if len(f) > 7 else 0
                params["dn"] = _fval(f[8]) if len(f) > 8 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("PROP_SOL_ORTH_2")
                params["qa"] = _fval(h[0], 1.1) if len(h) > 0 and h[0].strip() else 1.1
                params["qb"] = _fval(h[1], 0.05) if len(h) > 1 and h[1].strip() else 0.05
                params["h"] = _fval(h[2], 0.1) if len(h) > 2 and h[2].strip() else 0.1
            if len(cards) >= 3 and not cards[2].is_blank:
                v = cards[2].cut("PROP_SOL_ORTH_3")
                params["vx"] = _fval(v[0]) if len(v) > 0 else 0.0
                params["vy"] = _fval(v[1]) if len(v) > 1 else 0.0
                params["vz"] = _fval(v[2]) if len(v) > 2 else 0.0
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["ip"] = _ival(v[4]) if len(v) > 4 else 0
                params["iorth"] = _ival(v[5]) if len(v) > 5 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                ang = cards[3].cut("PROP_SOL_ORTH_4")
                params["phi"] = _fval(ang[0]) if len(ang) > 0 else 0.0
                params["px"] = _fval(ang[1]) if len(ang) > 1 else 0.0
                params["py"] = _fval(ang[2]) if len(ang) > 2 else 0.0
                params["pz"] = _fval(ang[3]) if len(ang) > 3 else 0.0
            if len(cards) >= 5 and not cards[4].is_blank:
                toks5 = cards[4].tokens()
                if len(toks5) <= 3 and len(cards[4].raw.rstrip()) <= 50:
                    dt = cards[4].cut("PROP_SOL_ORTH_DT")
                    params["deltat_min"] = _fval(dt[0]) if len(dt) > 0 else 0.0
                    params["istrain"] = _ival(dt[1]) if len(dt) > 1 else 0
                    params["ihkt"] = _ival(dt[2]) if len(dt) > 2 else 0
                else:
                    dt = cards[4].cut("PROP_SOL_ORTH_5")
                    params["deltat_min"] = _fval(dt[0]) if len(dt) > 0 else 0.0
                    params["vdef_min"] = _fval(dt[1]) if len(dt) > 1 else 0.0
                    params["vdef_max"] = _fval(dt[2]) if len(dt) > 2 else 0.0
                    params["asp_max"] = _fval(dt[3]) if len(dt) > 3 else 0.0
                    params["col_min"] = _fval(dt[4]) if len(dt) > 4 else 0.0
            if len(cards) >= 6 and not cards[5].is_blank:
                f6 = cards[5].cut("PROP_SOL_ORTH_6")
                params["ndir"] = _ival(f6[0]) if len(f6) > 0 else 0
                params["sphpart_id"] = _ival(f6[1]) if len(f6) > 1 else 0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["isolid"] = int(float(t0[0])) if len(t0) > 0 else 14
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["icpre"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["itetra10"] = int(float(t0[3])) if len(t0) > 3 else 0
            nbp = int(float(t0[4])) if len(t0) > 4 else 0
            params["nbp"] = nbp
            if nbp > 200:
                params["inpts_r"] = nbp // 100
                rem = nbp % 100
                params["inpts_s"] = rem // 10
                params["inpts_t"] = rem % 10
            elif nbp > 0:
                params["inpts_r"] = nbp
                params["inpts_s"] = nbp
                params["inpts_t"] = nbp
            params["itetra4"] = int(float(t0[5])) if len(t0) > 5 else 0
            params["iframe"] = int(float(t0[6])) if len(t0) > 6 else 0
            params["dn"] = float(t0[7]) if len(t0) > 7 else 0.0

            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                params["qa"] = float(t1[0]) if len(t1) > 0 else 1.1
                params["qb"] = float(t1[1]) if len(t1) > 1 else 0.05
                params["h"] = float(t1[2]) if len(t1) > 2 else 0.1

            if len(cards) > 2 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                params["vx"] = float(t2[0]) if len(t2) > 0 else 0.0
                params["vy"] = float(t2[1]) if len(t2) > 1 else 0.0
                params["vz"] = float(t2[2]) if len(t2) > 2 else 0.0
                params["skew_id"] = int(float(t2[3])) if len(t2) > 3 else 0
                params["ip"] = int(float(t2[4])) if len(t2) > 4 else 0
                params["iorth"] = int(float(t2[5])) if len(t2) > 5 else 0

            if len(cards) > 3 and not cards[3].is_blank:
                t3 = cards[3].tokens()
                params["phi"] = float(t3[0]) if len(t3) > 0 else 0.0
                params["px"] = float(t3[1]) if len(t3) > 1 else 0.0
                params["py"] = float(t3[2]) if len(t3) > 2 else 0.0
                params["pz"] = float(t3[3]) if len(t3) > 3 else 0.0

            if len(cards) > 4 and not cards[4].is_blank:
                t4 = cards[4].tokens()
                if len(t4) <= 3:
                    params["deltat_min"] = float(t4[0]) if len(t4) > 0 else 0.0
                    params["istrain"] = int(float(t4[1])) if len(t4) > 1 else 0
                    params["ihkt"] = int(float(t4[2])) if len(t4) > 2 else 0
                else:
                    params["deltat_min"] = float(t4[0]) if len(t4) > 0 else 0.0
                    params["vdef_min"] = float(t4[1]) if len(t4) > 1 else 0.0
                    params["vdef_max"] = float(t4[2]) if len(t4) > 2 else 0.0
                    params["asp_max"] = float(t4[3]) if len(t4) > 3 else 0.0
                    params["col_min"] = float(t4[4]) if len(t4) > 4 else 0.0

            if len(cards) > 5 and not cards[5].is_blank:
                t5 = cards[5].tokens()
                params["ndir"] = int(float(t5[0])) if len(t5) > 0 else 0
                params["sphpart_id"] = int(float(t5[1])) if len(t5) > 1 else 0

    elif ptype == 20:  # TSHELL
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01,
                  "itshell": 0, "ashear": 0.833333, "qa": 1.1, "qb": 0.05, "h": 0.1,
                  "deltat_min": 0.0}
        is_cfg = False
        if block.fixed:
            if len(cards) >= 2 and len(cards[1].raw.rstrip()) > 60:
                is_cfg = False
            else:
                is_cfg = True
        else:
            if len(cards) >= 2 and len(cards[1].tokens()) >= 4:
                is_cfg = False
            else:
                is_cfg = True

        if is_cfg:
            if block.fixed:
                if cards and not cards[0].is_blank:
                    f1 = cards[0].cut("PROP_TYPE20_1")
                    params["itshell"] = _ival(f1[0], 15) if len(f1) > 0 and f1[0].strip() else 15
                    params["isolid"] = params["itshell"]
                    params["ismstr"] = _ival(f1[1]) if len(f1) > 1 else 0
                    params["icpre"] = _ival(f1[2]) if len(f1) > 2 else 0
                    params["icstr"] = _ival(f1[3]) if len(f1) > 3 else 0
                    nbp = _ival(f1[4], 222) if len(f1) > 4 and f1[4].strip() else 222
                    params["nbp"] = nbp
                    if nbp > 200:
                        params["inpts_r"] = nbp // 100
                        rem = nbp % 100
                        params["inpts_s"] = rem // 10
                        params["inpts_t"] = rem % 10
                    else:
                        params["inpts_s"] = nbp
                    params["nip"] = params.get("inpts_t", 3)
                    params["iint"] = _ival(f1[5], 1) if len(f1) > 5 and f1[5].strip() else 1
                    params["dn"] = _fval(f1[6]) if len(f1) > 6 else 0.0
                if len(cards) > 1 and not cards[1].is_blank:
                    f2 = cards[1].cut("PROP_TYPE20_2")
                    params["qa"] = _fval(f2[0], 1.1) if len(f2) > 0 and f2[0].strip() else 1.1
                    params["qb"] = _fval(f2[1], 0.05) if len(f2) > 1 and f2[1].strip() else 0.05
                    params["h"] = _fval(f2[2], 0.1) if len(f2) > 2 and f2[2].strip() else 0.1
                if len(cards) > 2 and not cards[2].is_blank:
                    f3 = cards[2].cut("PROP_TYPE20_3")
                    params["deltat_min"] = _fval(f3[0]) if len(f3) > 0 else 0.0
            else:
                t0 = cards[0].tokens() if len(cards) > 0 else []
                params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 15
                params["isolid"] = params["itshell"]
                params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
                params["icpre"] = int(float(t0[2])) if len(t0) > 2 else 0
                params["icstr"] = int(float(t0[3])) if len(t0) > 3 else 0
                nbp = int(float(t0[4])) if len(t0) > 4 else 222
                params["nbp"] = nbp
                if nbp > 200:
                    params["inpts_r"] = nbp // 100
                    rem = nbp % 100
                    params["inpts_s"] = rem // 10
                    params["inpts_t"] = rem % 10
                else:
                    params["inpts_s"] = nbp
                params["nip"] = params.get("inpts_t", 3)
                params["iint"] = int(float(t0[5])) if len(t0) > 5 else 1
                params["dn"] = float(t0[6]) if len(t0) > 6 else 0.0
                if len(cards) > 1 and not cards[1].is_blank:
                    t1 = cards[1].tokens()
                    params["qa"] = float(t1[0]) if len(t1) > 0 else 1.1
                    params["qb"] = float(t1[1]) if len(t1) > 1 else 0.05
                    params["h"] = float(t1[2]) if len(t1) > 2 else 0.1
                if len(cards) > 2 and not cards[2].is_blank:
                    t2 = cards[2].tokens()
                    params["deltat_min"] = float(t2[0]) if len(t2) > 0 else 0.0
        else:
            if block.fixed:
                if cards and not cards[0].is_blank:
                    f = cards[0].cut("PROP_TSHELL_1")
                    params["itshell"] = _ival(f[0])
                    params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                    params["idrill"] = _ival(f[2]) if len(f) > 2 else 0
                    params["p_thick_fail"] = _fval(f[3]) if len(f) > 3 else 0.0
                if len(cards) >= 2 and not cards[1].is_blank:
                    h = cards[1].cut("PROP_TSHELL_2")
                    params["hm"] = _fval(h[0]) or 0.01
                    params["hf"] = _fval(h[1]) or 0.01
                    params["hr"] = _fval(h[2]) or 0.01
                    params["dm"] = _fval(h[3]) if len(h) > 3 else 0.0
                    params["dn"] = _fval(h[4]) if len(h) > 4 else 0.0
                if len(cards) >= 3 and not cards[2].is_blank:
                    t = cards[2].cut("PROP_TSHELL_3")
                    params["nip"] = _ival(t[0]) or 3
                    params["istrain"] = _ival(t[1]) if len(t) > 1 else 0
                    params["thick"] = _fval(t[2]) or 1.0
                    params["ashear"] = _fval(t[3]) or 0.833333
                    params["ithick"] = _ival(t[4]) if len(t) > 4 else 0
                    params["iplas"] = _ival(t[5]) if len(t) > 5 else 0
            else:
                t0 = cards[0].tokens() if len(cards) > 0 else []
                params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 0
                params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
                params["idrill"] = int(float(t0[2])) if len(t0) > 2 else 0
                params["p_thick_fail"] = float(t0[3]) if len(t0) > 3 else 0.0

                t1 = cards[1].tokens() if len(cards) > 1 else []
                params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
                params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
                params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
                params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
                params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

                t2 = cards[2].tokens() if len(cards) > 2 else []
                params["nip"] = int(float(t2[0])) if len(t2) > 0 else 3
                params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
                params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
                params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
                params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
                params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

    elif ptype == 21:  # TSH_ORTH
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01,
                  "itshell": 0, "ashear": 0.833333, "vx": 1.0, "vy": 0.0, "vz": 0.0,
                  "qa": 1.1, "qb": 0.05, "h": 0.1, "phi": 0.0, "deltat_min": 0.0}
        is_cfg = False
        if block.fixed:
            if len(cards) >= 2 and len(cards[1].raw.rstrip()) > 60:
                is_cfg = False
            else:
                is_cfg = True
        else:
            if len(cards) >= 2 and len(cards[1].tokens()) >= 4:
                is_cfg = False
            else:
                is_cfg = True

        if is_cfg:
            if block.fixed:
                if cards and not cards[0].is_blank:
                    f1 = cards[0].cut("PROP_TYPE21_1")
                    params["itshell"] = _ival(f1[0], 15) if len(f1) > 0 and f1[0].strip() else 15
                    params["isolid"] = params["itshell"]
                    params["ismstr"] = _ival(f1[1]) if len(f1) > 1 else 0
                    params["icpre"] = _ival(f1[3]) if len(f1) > 3 else 0
                    params["icstr"] = _ival(f1[4]) if len(f1) > 4 else 0
                    nbp = _ival(f1[5], 222) if len(f1) > 5 and f1[5].strip() else 222
                    params["nbp"] = nbp
                    if nbp > 200:
                        params["inpts_r"] = nbp // 100
                        rem = nbp % 100
                        params["inpts_s"] = rem // 10
                        params["inpts_t"] = rem % 10
                    else:
                        params["inpts_s"] = nbp
                    params["nip"] = params.get("inpts_t", 3)
                    params["iint"] = _ival(f1[6], 1) if len(f1) > 6 and f1[6].strip() else 1
                    params["dn"] = _fval(f1[8]) if len(f1) > 8 else 0.0
                if len(cards) > 1 and not cards[1].is_blank:
                    f2 = cards[1].cut("PROP_TYPE21_2")
                    params["qa"] = _fval(f2[0], 1.1) if len(f2) > 0 and f2[0].strip() else 1.1
                    params["qb"] = _fval(f2[1], 0.05) if len(f2) > 1 and f2[1].strip() else 0.05
                if len(cards) > 2 and not cards[2].is_blank:
                    f3 = cards[2].cut("PROP_TYPE21_3")
                    params["vx"] = _fval(f3[0], 1.0) if len(f3) > 0 and f3[0].strip() else 1.0
                    params["vy"] = _fval(f3[1]) if len(f3) > 1 else 0.0
                    params["vz"] = _fval(f3[2]) if len(f3) > 2 else 0.0
                    params["skew_id"] = _ival(f3[3]) if len(f3) > 3 else 0
                    params["iorth"] = _ival(f3[4]) if len(f3) > 4 else 0
                if len(cards) > 3 and not cards[3].is_blank:
                    f4 = cards[3].cut("PROP_TYPE21_4")
                    params["phi"] = _fval(f4[0]) if len(f4) > 0 else 0.0
                if len(cards) > 4 and not cards[4].is_blank:
                    f5 = cards[4].cut("PROP_TYPE21_5")
                    params["deltat_min"] = _fval(f5[0]) if len(f5) > 0 else 0.0
            else:
                t0 = cards[0].tokens() if len(cards) > 0 else []
                params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 15
                params["isolid"] = params["itshell"]
                params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
                params["icstr"] = int(float(t0[3])) if len(t0) > 3 else 0
                nbp = int(float(t0[4])) if len(t0) > 4 else 222
                params["nbp"] = nbp
                if nbp > 200:
                    params["inpts_r"] = nbp // 100
                    rem = nbp % 100
                    params["inpts_s"] = rem // 10
                    params["inpts_t"] = rem % 10
                else:
                    params["inpts_s"] = nbp
                params["nip"] = params.get("inpts_t", 3)
                params["iint"] = int(float(t0[5])) if len(t0) > 5 else 1
                params["dn"] = float(t0[6]) if len(t0) > 6 else 0.0
                if len(cards) > 1 and not cards[1].is_blank:
                    t1 = cards[1].tokens()
                    params["qa"] = float(t1[0]) if len(t1) > 0 else 1.1
                    params["qb"] = float(t1[1]) if len(t1) > 1 else 0.05
                if len(cards) > 2 and not cards[2].is_blank:
                    t2 = cards[2].tokens()
                    params["vx"] = float(t2[0]) if len(t2) > 0 else 1.0
                    params["vy"] = float(t2[1]) if len(t2) > 1 else 0.0
                    params["vz"] = float(t2[2]) if len(t2) > 2 else 0.0
                    params["skew_id"] = int(float(t2[3])) if len(t2) > 3 else 0
                    params["iorth"] = int(float(t2[4])) if len(t2) > 4 else 0
                if len(cards) > 3 and not cards[3].is_blank:
                    t3 = cards[3].tokens()
                    params["phi"] = float(t3[0]) if len(t3) > 0 else 0.0
                if len(cards) > 4 and not cards[4].is_blank:
                    t4 = cards[4].tokens()
                    params["deltat_min"] = float(t4[0]) if len(t4) > 0 else 0.0
        else:
            if block.fixed:
                if cards and not cards[0].is_blank:
                    f = cards[0].cut("PROP_TSHELL_1")
                    params["itshell"] = _ival(f[0])
                    params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                    params["idrill"] = _ival(f[2]) if len(f) > 2 else 0
                    params["p_thick_fail"] = _fval(f[3]) if len(f) > 3 else 0.0
                if len(cards) >= 2 and not cards[1].is_blank:
                    h = cards[1].cut("PROP_TSHELL_2")
                    params["hm"] = _fval(h[0]) or 0.01
                    params["hf"] = _fval(h[1]) or 0.01
                    params["hr"] = _fval(h[2]) or 0.01
                    params["dm"] = _fval(h[3]) if len(h) > 3 else 0.0
                    params["dn"] = _fval(h[4]) if len(h) > 4 else 0.0
                if len(cards) >= 3 and not cards[2].is_blank:
                    t = cards[2].cut("PROP_TSHELL_3")
                    params["nip"] = _ival(t[0]) or 3
                    params["istrain"] = _ival(t[1]) if len(t) > 1 else 0
                    params["thick"] = _fval(t[2]) or 1.0
                    params["ashear"] = _fval(t[3]) or 0.833333
                    params["ithick"] = _ival(t[4]) if len(t) > 4 else 0
                    params["iplas"] = _ival(t[5]) if len(t) > 5 else 0
                if len(cards) >= 4 and not cards[3].is_blank:
                    v = cards[3].cut("PROP_TSH_ORTH_1")
                    params["vx"] = _fval(v[0]) or 1.0
                    params["vy"] = _fval(v[1])
                    params["vz"] = _fval(v[2])
                    params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                    params["iorth"] = _ival(v[4]) if len(v) > 4 else 0
                    params["ipos"] = _ival(v[5]) if len(v) > 5 else 0
                    params["ip"] = _ival(v[6]) if len(v) > 6 else 0
            else:
                t0 = cards[0].tokens() if len(cards) > 0 else []
                params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 0
                params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
                params["idrill"] = int(float(t0[2])) if len(t0) > 2 else 0
                params["p_thick_fail"] = float(t0[3]) if len(t0) > 3 else 0.0

                t1 = cards[1].tokens() if len(cards) > 1 else []
                params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
                params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
                params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
                params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
                params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

                t2 = cards[2].tokens() if len(cards) > 2 else []
                params["nip"] = int(float(t2[0])) if len(t2) > 0 else 3
                params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
                params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
                params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
                params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
                params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

                t3 = cards[3].tokens() if len(cards) > 3 else []
                params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
                params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
                params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
                params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
                params["iorth"] = int(float(t3[4])) if len(t3) > 4 else 0
                params["ipos"] = int(float(t3[5])) if len(t3) > 5 else 0
                params["ip"] = int(float(t3[6])) if len(t3) > 6 else 0

    elif ptype == 22:  # TSH_COMP
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01,
                  "itshell": 0, "ashear": 0.833333, "vx": 1.0, "vy": 0.0, "vz": 0.0,
                  "qa": 1.1, "qb": 0.05, "h": 0.1, "phi": 0.0, "deltat_min": 0.0}
        is_cfg = False
        if block.fixed:
            if len(cards) >= 2 and len(cards[1].raw.rstrip()) > 60:
                is_cfg = False
            else:
                is_cfg = True
        else:
            if len(cards) >= 2 and len(cards[1].tokens()) >= 4:
                is_cfg = False
            else:
                is_cfg = True

        if is_cfg:
            if block.fixed:
                if cards and not cards[0].is_blank:
                    f1 = cards[0].cut("PROP_TYPE22_1")
                    params["itshell"] = _ival(f1[0], 15) if len(f1) > 0 and f1[0].strip() else 15
                    params["isolid"] = params["itshell"]
                    params["ismstr"] = _ival(f1[1]) if len(f1) > 1 else 0
                    params["icstr"] = _ival(f1[3]) if len(f1) > 3 else 0
                    nbp = _ival(f1[4], 222) if len(f1) > 4 and f1[4].strip() else 222
                    params["nbp"] = nbp
                    if nbp > 200:
                        params["inpts_r"] = nbp // 100
                        rem = nbp % 100
                        params["inpts_s"] = rem // 10
                        params["inpts_t"] = rem % 10
                    else:
                        params["inpts_s"] = nbp
                    params["nip"] = params.get("inpts_t", 3)
                    params["iint"] = _ival(f1[5], 1) if len(f1) > 5 and f1[5].strip() else 1
                    params["dn"] = _fval(f1[7]) if len(f1) > 7 else 0.0
                if len(cards) > 1 and not cards[1].is_blank:
                    f2 = cards[1].cut("PROP_TYPE22_2")
                    params["qa"] = _fval(f2[0], 1.1) if len(f2) > 0 and f2[0].strip() else 1.1
                    params["qb"] = _fval(f2[1], 0.05) if len(f2) > 1 and f2[1].strip() else 0.05
                if len(cards) > 2 and not cards[2].is_blank:
                    f3 = cards[2].cut("PROP_TYPE22_3")
                    params["vx"] = _fval(f3[0], 1.0) if len(f3) > 0 and f3[0].strip() else 1.0
                    params["vy"] = _fval(f3[1]) if len(f3) > 1 else 0.0
                    params["vz"] = _fval(f3[2]) if len(f3) > 2 else 0.0
                    params["skew_id"] = _ival(f3[3]) if len(f3) > 3 else 0
                    params["iorth"] = _ival(f3[4]) if len(f3) > 4 else 0
                    params["ipos"] = _ival(f3[5]) if len(f3) > 5 else 0
                if len(cards) > 3 and not cards[3].is_blank:
                    f4 = cards[3].cut("PROP_TYPE22_4")
                    params["ashear"] = _fval(f4[0], 0.833333) if len(f4) > 0 and f4[0].strip() else 0.833333
                layers = []
                last_is_deltat = len(cards) > 4 and len(cards[-1].tokens()) < 4
                layer_cards = cards[4:-1] if last_is_deltat else cards[4:]
                for c in layer_cards:
                    if c.is_blank:
                        continue
                    ly = c.cut("PROP_TYPE22_LAYER")
                    layers.append({
                        "phi": _fval(ly[0]),
                        "thick": _fval(ly[1]),
                        "zi": _fval(ly[2]),
                        "mat_id": _ival(ly[3]),
                    })
                if last_is_deltat and not cards[-1].is_blank:
                    f_last = cards[-1].cut("PROP_TYPE22_5")
                    params["deltat_min"] = _fval(f_last[0]) if len(f_last) > 0 else 0.0
                params["layers"] = layers
            else:
                t0 = cards[0].tokens() if len(cards) > 0 else []
                params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 15
                params["isolid"] = params["itshell"]
                params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
                params["icstr"] = int(float(t0[3])) if len(t0) > 3 else 0
                nbp = int(float(t0[4])) if len(t0) > 4 else 222
                params["nbp"] = nbp
                if nbp > 200:
                    params["inpts_r"] = nbp // 100
                    rem = nbp % 100
                    params["inpts_s"] = rem // 10
                    params["inpts_t"] = rem % 10
                else:
                    params["inpts_s"] = nbp
                params["nip"] = params.get("inpts_t", 3)
                params["iint"] = int(float(t0[5])) if len(t0) > 5 else 1
                params["dn"] = float(t0[6]) if len(t0) > 6 else 0.0
                if len(cards) > 1 and not cards[1].is_blank:
                    t1 = cards[1].tokens()
                    params["qa"] = float(t1[0]) if len(t1) > 0 else 1.1
                    params["qb"] = float(t1[1]) if len(t1) > 1 else 0.05
                if len(cards) > 2 and not cards[2].is_blank:
                    t2 = cards[2].tokens()
                    params["vx"] = float(t2[0]) if len(t2) > 0 else 1.0
                    params["vy"] = float(t2[1]) if len(t2) > 1 else 0.0
                    params["vz"] = float(t2[2]) if len(t2) > 2 else 0.0
                    params["skew_id"] = int(float(t2[3])) if len(t2) > 3 else 0
                    params["iorth"] = int(float(t2[4])) if len(t2) > 4 else 0
                    params["ipos"] = int(float(t2[5])) if len(t2) > 5 else 0
                if len(cards) > 3 and not cards[3].is_blank:
                    t3 = cards[3].tokens()
                    params["ashear"] = float(t3[0]) if len(t3) > 0 else 0.833333
                layers = []
                last_is_deltat = len(cards) > 4 and len(cards[-1].tokens()) < 4
                layer_cards = cards[4:-1] if last_is_deltat else cards[4:]
                for c in layer_cards:
                    toks = c.tokens()
                    if not toks:
                        continue
                    layers.append({
                        "phi": float(toks[0]) if len(toks) > 0 else 0.0,
                        "thick": float(toks[1]) if len(toks) > 1 else 1.0,
                        "zi": float(toks[2]) if len(toks) > 2 else 0.0,
                        "mat_id": int(float(toks[3])) if len(toks) > 3 else 0,
                    })
                if last_is_deltat and not cards[-1].is_blank:
                    t_last = cards[-1].tokens()
                    params["deltat_min"] = float(t_last[0]) if len(t_last) > 0 else 0.0
                params["layers"] = layers
        else:
            if block.fixed:
                if cards and not cards[0].is_blank:
                    f = cards[0].cut("PROP_TSHELL_1")
                    params["itshell"] = _ival(f[0])
                    params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                    params["idrill"] = _ival(f[2]) if len(f) > 2 else 0
                    params["p_thick_fail"] = _fval(f[3]) if len(f) > 3 else 0.0
                if len(cards) >= 2 and not cards[1].is_blank:
                    h = cards[1].cut("PROP_TSHELL_2")
                    params["hm"] = _fval(h[0]) or 0.01
                    params["hf"] = _fval(h[1]) or 0.01
                    params["hr"] = _fval(h[2]) or 0.01
                    params["dm"] = _fval(h[3]) if len(h) > 3 else 0.0
                    params["dn"] = _fval(h[4]) if len(h) > 4 else 0.0
                if len(cards) >= 3 and not cards[2].is_blank:
                    t = cards[2].cut("PROP_TSHELL_3")
                    params["nip"] = _ival(t[0]) or 3
                    params["istrain"] = _ival(t[1]) if len(t) > 1 else 0
                    params["thick"] = _fval(t[2]) or 1.0
                    params["ashear"] = _fval(t[3]) or 0.833333
                    params["ithick"] = _ival(t[4]) if len(t) > 4 else 0
                    params["iplas"] = _ival(t[5]) if len(t) > 5 else 0
                if len(cards) >= 4 and not cards[3].is_blank:
                    v = cards[3].cut("PROP_TSH_ORTH_1")
                    params["vx"] = _fval(v[0]) or 1.0
                    params["vy"] = _fval(v[1])
                    params["vz"] = _fval(v[2])
                    params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                    params["iorth"] = _ival(v[4]) if len(v) > 4 else 0
                    params["ipos"] = _ival(v[5]) if len(v) > 5 else 0
                    params["ip"] = _ival(v[6]) if len(v) > 6 else 0
                layers = []
                for c in cards[4:]:
                    if c.is_blank:
                        continue
                    ly = c.cut("PROP_SH_SANDW_LAYER")
                    layers.append({
                        "phi": _fval(ly[0]),
                        "thick": _fval(ly[1]),
                        "zi": _fval(ly[2]),
                        "mat_id": _ival(ly[3]),
                        "w_fi": _fval(ly[5]) if len(ly) > 5 else 1.0
                    })
                params["layers"] = layers
            else:
                t0 = cards[0].tokens() if len(cards) > 0 else []
                params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 0
                params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
                params["idrill"] = int(float(t0[2])) if len(t0) > 2 else 0
                params["p_thick_fail"] = float(t0[3]) if len(t0) > 3 else 0.0

                t1 = cards[1].tokens() if len(cards) > 1 else []
                params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
                params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
                params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
                params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
                params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

                t2 = cards[2].tokens() if len(cards) > 2 else []
                params["nip"] = int(float(t2[0])) if len(t2) > 0 else 3
                params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
                params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
                params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
                params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
                params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

                t3 = cards[3].tokens() if len(cards) > 3 else []
                params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
                params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
                params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
                params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
                params["iorth"] = int(float(t3[4])) if len(t3) > 4 else 0
                params["ipos"] = int(float(t3[5])) if len(t3) > 5 else 0
                params["ip"] = int(float(t3[6])) if len(t3) > 6 else 0
                layers = []
                for c in cards[4:]:
                    toks = c.tokens()
                    if not toks:
                        continue
                    layers.append({
                        "phi": float(toks[0]) if len(toks) > 0 else 0.0,
                        "thick": float(toks[1]) if len(toks) > 1 else 1.0,
                        "zi": float(toks[2]) if len(toks) > 2 else 0.0,
                        "mat_id": int(float(toks[3])) if len(toks) > 3 else 0,
                        "w_fi": float(toks[4]) if len(toks) > 4 else 1.0,
                    })
                params["layers"] = layers

    elif ptype == 34:  # SPH or USER_SOLID
        if typename in ("USER_SOLID", "PROP_USER_SOLID", "PROP_P34_USER"):
            nuvar = 0
            skew_id = 0
            if cards and not cards[0].is_blank:
                if block.fixed:
                    f = cards[0].cut("PROP_USER_SOLID_1")
                    nuvar = _ival(f[0]) if len(f) > 0 else 0
                    skew_id = _ival(f[1]) if len(f) > 1 else 0
                else:
                    toks = cards[0].tokens()
                    nuvar = int(float(toks[0])) if len(toks) > 0 else 0
                    skew_id = int(float(toks[1])) if len(toks) > 1 else 0
            params = {"nuvar": nuvar, "skew_id": skew_id}
        else:
            params = {"mass": 1.0, "h0": 1.0, "d0": 1.0, "alpha": 1.0, "beta": 1.0, "q0": 0.0, "gamma": 1.0}
            if block.fixed:
                if cards and not cards[0].is_blank:
                    c1 = cards[0].cut("PROP_SPH_1")
                    params["mass"] = _fval(c1[0]) or 1.0
                    params["h0"] = _fval(c1[1]) or 1.0
                    params["d0"] = _fval(c1[2]) or 1.0
                if len(cards) >= 2 and not cards[1].is_blank:
                    c2 = cards[1].cut("PROP_SPH_2")
                    params["alpha"] = _fval(c2[0]) or 1.0
                    params["beta"] = _fval(c2[1]) or 1.0
                    params["q0"] = _fval(c2[2])
                    params["gamma"] = _fval(c2[3]) or 1.0
            else:
                t0 = cards[0].tokens() if len(cards) > 0 else []
                params["mass"] = float(t0[0]) if len(t0) > 0 else 1.0
                params["h0"] = float(t0[1]) if len(t0) > 1 else 1.0
                params["d0"] = float(t0[2]) if len(t0) > 2 else 1.0

                t1 = cards[1].tokens() if len(cards) > 1 else []
                params["alpha"] = float(t1[0]) if len(t1) > 0 else 1.0
                params["beta"] = float(t1[1]) if len(t1) > 1 else 1.0
                params["q0"] = float(t1[2]) if len(t1) > 2 else 0.0
                params["gamma"] = float(t1[3]) if len(t1) > 3 else 1.0

    elif ptype == 0:  # VOID
        from ..prop_reader import _universal_geo_params
        params = _universal_geo_params()
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("F20X5")
                params["thick"] = _fval(f[0]) or 1.0
            else:
                t = cards[0].tokens()
                params["thick"] = float(t[0]) if t else 1.0

    elif ptype == 43:  # CONNECT / TYPE43
        ismstr = 0
        thick = 0.0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("PROP_CONNECT_1")
                ismstr = _ival(f[0]) if len(f) > 0 else 0
                thick = next((_fval(x) for x in reversed(f[1:]) if _fval(x) != 0.0), 0.0)
            else:
                toks = cards[0].tokens()
                ismstr = int(float(toks[0])) if len(toks) > 0 else 0
                thick = float(toks[1]) if len(toks) > 1 else 0.0
        params = {"ismstr": ismstr, "thick": thick}

    elif ptype == 17:  # STACK
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "z0": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "istrain": 0, "ashear": 0.833333, "iint": 0, "ithick": 0,
                  "vx": 0.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "iorth": 0, "ipos": 0, "ip": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("STACK_1")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["z0"] = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                f = cards[1].cut("STACK_2")
                params["hm"] = _fval(f[0], 0.01) if len(f) > 0 else 0.01
                params["hf"] = _fval(f[1], 0.01) if len(f) > 1 else 0.01
                params["hr"] = _fval(f[2], 0.01) if len(f) > 2 else 0.01
                params["dm"] = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                params["dn"] = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("STACK_3")
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["ashear"] = _fval(f[2], 0.833333) if len(f) > 2 else 0.833333
                params["iint"] = _ival(f[4]) if len(f) > 4 else 0
                params["ithick"] = _ival(f[6]) if len(f) > 6 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                f = cards[3].cut("STACK_4")
                params["vx"] = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                params["vy"] = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                params["vz"] = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                params["skew_id"] = _ival(f[3]) if len(f) > 3 else 0
                params["iorth"] = _ival(f[4]) if len(f) > 4 else 0
                params["ipos"] = _ival(f[5]) if len(f) > 5 else 0
                params["ip"] = _ival(f[6]) if len(f) > 6 else 0
        else:
            if len(cards) >= 1 and not cards[0].is_blank:
                t = cards[0].tokens()
                params["ishell"] = int(float(t[0])) if len(t) > 0 else 0
                params["ismstr"] = int(float(t[1])) if len(t) > 1 else 0
                params["ish3n"] = int(float(t[2])) if len(t) > 2 else 0
                params["idrill"] = int(float(t[3])) if len(t) > 3 else 0
                params["z0"] = float(t[4]) if len(t) > 4 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                t = cards[1].tokens()
                params["hm"] = float(t[0]) if len(t) > 0 else 0.01
                params["hf"] = float(t[1]) if len(t) > 1 else 0.01
                params["hr"] = float(t[2]) if len(t) > 2 else 0.01
                params["dm"] = float(t[3]) if len(t) > 3 else 0.0
                params["dn"] = float(t[4]) if len(t) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                t = cards[2].tokens()
                params["istrain"] = int(float(t[0])) if len(t) > 0 else 0
                params["ashear"] = float(t[1]) if len(t) > 1 else 0.833333
                params["iint"] = int(float(t[2])) if len(t) > 2 else 0
                params["ithick"] = int(float(t[3])) if len(t) > 3 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                t = cards[3].tokens()
                params["vx"] = float(t[0]) if len(t) > 0 else 0.0
                params["vy"] = float(t[1]) if len(t) > 1 else 0.0
                params["vz"] = float(t[2]) if len(t) > 2 else 0.0
                params["skew_id"] = int(float(t[3])) if len(t) > 3 else 0
                params["iorth"] = int(float(t[4])) if len(t) > 4 else 0
                params["ipos"] = int(float(t[5])) if len(t) > 5 else 0
                params["ip"] = int(float(t[6])) if len(t) > 6 else 0

    elif ptype == 51:  # P51
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "z0": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "istrain": 0, "ashear": 0.833333, "ithick": 0,
                  "vx": 0.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "iorth": 0, "ipos": 0,
                  "p_thick_fail": 0.0, "fexp": 0.0, "ip": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_P51_1")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["z0"] = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                f = cards[1].cut("PROP_P51_2")
                params["hm"] = _fval(f[0], 0.01) if len(f) > 0 else 0.01
                params["hf"] = _fval(f[1], 0.01) if len(f) > 1 else 0.01
                params["hr"] = _fval(f[2], 0.01) if len(f) > 2 else 0.01
                params["dm"] = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                params["dn"] = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_P51_3")
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["ashear"] = _fval(f[2], 0.833333) if len(f) > 2 else 0.833333
                params["ithick"] = _ival(f[4]) if len(f) > 4 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                f = cards[3].cut("PROP_P51_4")
                params["vx"] = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                params["vy"] = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                params["vz"] = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                params["skew_id"] = _ival(f[3]) if len(f) > 3 else 0
                params["iorth"] = _ival(f[4]) if len(f) > 4 else 0
                params["ipos"] = _ival(f[5]) if len(f) > 5 else 0
                params["p_thick_fail"] = _fval(f[6], 0.0) if len(f) > 6 else 0.0
                params["fexp"] = _fval(f[7], 0.0) if len(f) > 7 else 0.0
                params["ip"] = _ival(f[8]) if len(f) > 8 else 0
        else:
            if len(cards) >= 1 and not cards[0].is_blank:
                t = cards[0].tokens()
                params["ishell"] = int(float(t[0])) if len(t) > 0 else 0
                params["ismstr"] = int(float(t[1])) if len(t) > 1 else 0
                params["ish3n"] = int(float(t[2])) if len(t) > 2 else 0
                params["idrill"] = int(float(t[3])) if len(t) > 3 else 0
                params["z0"] = float(t[4]) if len(t) > 4 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                t = cards[1].tokens()
                params["hm"] = float(t[0]) if len(t) > 0 else 0.01
                params["hf"] = float(t[1]) if len(t) > 1 else 0.01
                params["hr"] = float(t[2]) if len(t) > 2 else 0.01
                params["dm"] = float(t[3]) if len(t) > 3 else 0.0
                params["dn"] = float(t[4]) if len(t) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                t = cards[2].tokens()
                params["istrain"] = int(float(t[0])) if len(t) > 0 else 0
                params["ashear"] = float(t[1]) if len(t) > 1 else 0.833333
                params["ithick"] = int(float(t[2])) if len(t) > 2 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                t = cards[3].tokens()
                params["vx"] = float(t[0]) if len(t) > 0 else 0.0
                params["vy"] = float(t[1]) if len(t) > 1 else 0.0
                params["vz"] = float(t[2]) if len(t) > 2 else 0.0
                params["skew_id"] = int(float(t[3])) if len(t) > 3 else 0
                params["iorth"] = int(float(t[4])) if len(t) > 4 else 0
                params["ipos"] = int(float(t[5])) if len(t) > 5 else 0
                params["p_thick_fail"] = float(t[6]) if len(t) > 6 else 0.0
                params["fexp"] = float(t[7]) if len(t) > 7 else 0.0
                params["ip"] = int(float(t[8])) if len(t) > 8 else 0

    elif ptype == 9:  # SH_ORTH
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "p_thick_fail": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "nip": 1, "istrain": 0, "thick": 1.0, "ashear": 0.833333, "skew_id": 0,
                  "ithick": 0, "iplas": 0, "vx": 1.0, "vy": 0.0, "vz": 0.0,
                  "mat_beta": 0.0, "phi": 0.0, "ipos": 0, "ip": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SH_ORTH_FLAGS")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["p_thick_fail"] = _fval(f[5]) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3])
                params["dn"] = _fval(h[4])
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_SH_ORTH_N")
                params["nip"] = _ival(f[0]) or 1
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["thick"] = _fval(f[2]) or 1.0
                params["ashear"] = _fval(f[3]) or 0.833333
                params["skew_id"] = _ival(f[4]) if len(f) > 4 else 0
                params["ithick"] = _ival(f[5]) if len(f) > 5 else 0
                params["iplas"] = _ival(f[6]) if len(f) > 6 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_SH_ORTH_VEC")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["mat_beta"] = _fval(v[3])
                params["phi"] = params["mat_beta"]
                params["ipos"] = _ival(v[4]) if len(v) > 4 else 0
                params["ip"] = _ival(v[5]) if len(v) > 5 else 0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["ish3n"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["idrill"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["p_thick_fail"] = float(t0[4]) if len(t0) > 4 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
            params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 1
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["skew_id"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["ithick"] = int(float(t2[5])) if len(t2) > 5 else 0
            params["iplas"] = int(float(t2[6])) if len(t2) > 6 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["mat_beta"] = float(t3[3]) if len(t3) > 3 else 0.0
            params["phi"] = params["mat_beta"]
            params["ipos"] = int(float(t3[4])) if len(t3) > 4 else 0
            params["ip"] = int(float(t3[5])) if len(t3) > 5 else 0

    elif ptype == 18:  # INT_BEAM
        read_prop_type18(block, model, log)
        return

    elif ptype == 26:  # SPR_TAB
        params = {
            "mass": 0.0, "isensor": 0, "sens_id": 0, "isflag": 0, "ileng": 0,
            "nfunc": 0, "nraten": 0, "scale": 1.0, "stiff0": 0.0, "k": 0.0, "dmax": 0.0, "alpha1": 0.0,
            "load_curves": [], "unload_curves": [],
        }
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                c0 = cards[0].cut("PROP_SPR_TAB_1")
                params["mass"] = _fval(c0[0])
                params["isensor"] = _ival(c0[2]) if len(c0) > 2 else 0
                params["sens_id"] = params["isensor"]
                params["isflag"] = _ival(c0[3]) if len(c0) > 3 else 0
                params["ileng"] = _ival(c0[4]) if len(c0) > 4 else 0
            if len(cards) >= 2 and not cards[1].is_blank:
                c1 = cards[1].cut("PROP_SPR_TAB_2")
                params["nfunc"] = _ival(c1[0])
                params["nraten"] = _ival(c1[1]) if len(c1) > 1 else 0
                params["scale"] = _fval(c1[2], 1.0) if len(c1) > 2 else 1.0
                params["stiff0"] = _fval(c1[3]) if len(c1) > 3 else 0.0
                params["k"] = params["stiff0"]
                params["alpha1"] = _fval(c1[5]) if len(c1) > 5 else 0.0

            nfunc = params["nfunc"]
            nraten = params["nraten"]
            idx = 2
            load_curves = []
            for _ in range(nfunc):
                if idx < len(cards) and not cards[idx].is_blank:
                    cc = cards[idx].cut("PROP_SPR_TAB_CARD")
                    load_curves.append({
                        "fun_load": _ival(cc[0]),
                        "scale_load": _fval(cc[1], 1.0) if len(cc) > 1 else 1.0,
                        "strainrate_load": _fval(cc[2]) if len(cc) > 2 else 0.0,
                    })
                    idx += 1
            params["load_curves"] = load_curves

            unload_curves = []
            for _ in range(nraten):
                if idx < len(cards) and not cards[idx].is_blank:
                    cc = cards[idx].cut("PROP_SPR_TAB_CARD")
                    unload_curves.append({
                        "fun_unload": _ival(cc[0]),
                        "scale_unload": _fval(cc[1], 1.0) if len(cc) > 1 else 1.0,
                        "strainrate_unload": _fval(cc[2]) if len(cc) > 2 else 0.0,
                    })
                    idx += 1
            params["unload_curves"] = unload_curves
        else:
            if len(cards) >= 1 and not cards[0].is_blank:
                t0 = cards[0].tokens()
                params["mass"] = float(t0[0]) if len(t0) > 0 else 0.0
                params["isensor"] = int(float(t0[1])) if len(t0) > 1 else 0
                params["sens_id"] = params["isensor"]
                params["isflag"] = int(float(t0[2])) if len(t0) > 2 else 0
                params["ileng"] = int(float(t0[3])) if len(t0) > 3 else 0
                if len(t0) >= 6:
                    params["nfunc"] = int(float(t0[4]))
                    params["nraten"] = int(float(t0[5]))
            if len(cards) >= 2 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                if len(cards[0].tokens()) >= 6:
                    params["scale"] = float(t1[0]) if len(t1) > 0 else 1.0
                    params["stiff0"] = float(t1[1]) if len(t1) > 1 else 0.0
                    params["k"] = params["stiff0"]
                    params["dmax"] = float(t1[2]) if len(t1) > 2 else 0.0
                    params["alpha1"] = float(t1[3]) if len(t1) > 3 else 0.0
                else:
                    params["nfunc"] = int(float(t1[0])) if len(t1) > 0 else 0
                    params["nraten"] = int(float(t1[1])) if len(t1) > 1 else 0
                    params["scale"] = float(t1[2]) if len(t1) > 2 else 1.0
                    params["stiff0"] = float(t1[3]) if len(t1) > 3 else 0.0
                    params["k"] = params["stiff0"]
                    params["alpha1"] = float(t1[4]) if len(t1) > 4 else 0.0

            nfunc = params["nfunc"]
            nraten = params["nraten"]
            idx = 2
            load_curves = []
            for _ in range(nfunc):
                if idx < len(cards) and not cards[idx].is_blank:
                    tc = cards[idx].tokens()
                    load_curves.append({
                        "fun_load": int(float(tc[0])) if len(tc) > 0 else 0,
                        "scale_load": float(tc[1]) if len(tc) > 1 else 1.0,
                        "strainrate_load": float(tc[2]) if len(tc) > 2 else 0.0,
                    })
                    idx += 1
            params["load_curves"] = load_curves

            unload_curves = []
            for _ in range(nraten):
                if idx < len(cards) and not cards[idx].is_blank:
                    tc = cards[idx].tokens()
                    unload_curves.append({
                        "fun_unload": int(float(tc[0])) if len(tc) > 0 else 0,
                        "scale_unload": float(tc[1]) if len(tc) > 1 else 1.0,
                        "strainrate_unload": float(tc[2]) if len(tc) > 2 else 0.0,
                    })
                    idx += 1
            params["unload_curves"] = unload_curves

    elif ptype == 27:  # SPR_BDAMP
        params = {
            "mass": 0.0, "isensor": 0, "sens_id": 0, "isflag": 0, "ileng": 0, "itens": 0, "ifail": 0,
            "stiff": 0.0, "k": 0.0, "damp": 0.0, "c": 0.0, "nexp": 1.0, "n": 1.0,
            "min_rup": 0.0, "delta_min": 0.0, "max_rup": 0.0, "delta_max": 0.0,
            "gap": 0.0, "fsmooth": 0, "fcut": 0.0,
            "fun1": 0, "fct1": 0, "fun2": 0, "fct2": 0,
            "ascale1": 1.0, "fscale1": 1.0, "ascale2": 1.0, "fscale2": 1.0,
        }
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                c0 = cards[0].cut("PROP_SPR_BDAMP_1")
                params["mass"] = _fval(c0[0])
                params["isensor"] = _ival(c0[2]) if len(c0) > 2 else 0
                params["sens_id"] = params["isensor"]
                params["isflag"] = _ival(c0[3]) if len(c0) > 3 else 0
                params["ileng"] = _ival(c0[4]) if len(c0) > 4 else 0
                params["itens"] = _ival(c0[5]) if len(c0) > 5 else 0
                params["ifail"] = _ival(c0[6]) if len(c0) > 6 else 0
            if len(cards) >= 2 and not cards[1].is_blank:
                c1 = cards[1].cut("PROP_SPR_BDAMP_2")
                params["stiff"] = _fval(c1[0])
                params["k"] = params["stiff"]
                params["damp"] = _fval(c1[1])
                params["c"] = params["damp"]
                params["nexp"] = _fval(c1[2], 1.0) if len(c1) > 2 else 1.0
                params["n"] = params["nexp"]
                params["min_rup"] = _fval(c1[3]) if len(c1) > 3 else 0.0
                params["delta_min"] = params["min_rup"]
                params["max_rup"] = _fval(c1[4]) if len(c1) > 4 else 0.0
                params["delta_max"] = params["max_rup"]
            if len(cards) >= 3 and not cards[2].is_blank:
                c2 = cards[2].cut("PROP_SPR_BDAMP_3")
                params["gap"] = _fval(c2[0])
                params["fsmooth"] = _ival(c2[2]) if len(c2) > 2 else 0
                params["fcut"] = _fval(c2[3]) if len(c2) > 3 else 0.0
            if len(cards) >= 4 and not cards[3].is_blank:
                c3 = cards[3].cut("PROP_SPR_BDAMP_4")
                params["fun1"] = _ival(c3[0])
                params["fct1"] = params["fun1"]
                params["fun2"] = _ival(c3[1]) if len(c3) > 1 else 0
                params["fct2"] = params["fun2"]
                params["ascale1"] = _fval(c3[2], 1.0) if len(c3) > 2 else 1.0
                params["fscale1"] = _fval(c3[3], 1.0) if len(c3) > 3 else 1.0
                params["ascale2"] = _fval(c3[4], 1.0) if len(c3) > 4 else 1.0
                params["fscale2"] = _fval(c3[5], 1.0) if len(c3) > 5 else 1.0
        else:
            if len(cards) >= 1 and not cards[0].is_blank:
                t0 = cards[0].tokens()
                params["mass"] = float(t0[0]) if len(t0) > 0 else 0.0
                params["isensor"] = int(float(t0[1])) if len(t0) > 1 else 0
                params["sens_id"] = params["isensor"]
                params["isflag"] = int(float(t0[2])) if len(t0) > 2 else 0
                params["ileng"] = int(float(t0[3])) if len(t0) > 3 else 0
                params["itens"] = int(float(t0[4])) if len(t0) > 4 else 0
                params["ifail"] = int(float(t0[5])) if len(t0) > 5 else 0
            if len(cards) >= 2 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                params["stiff"] = float(t1[0]) if len(t1) > 0 else 0.0
                params["k"] = params["stiff"]
                params["damp"] = float(t1[1]) if len(t1) > 1 else 0.0
                params["c"] = params["damp"]
                params["nexp"] = float(t1[2]) if len(t1) > 2 else 1.0
                params["n"] = params["nexp"]
                params["min_rup"] = float(t1[3]) if len(t1) > 3 else 0.0
                params["delta_min"] = params["min_rup"]
                params["max_rup"] = float(t1[4]) if len(t1) > 4 else 0.0
                params["delta_max"] = params["max_rup"]
            if len(cards) >= 3 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                params["gap"] = float(t2[0]) if len(t2) > 0 else 0.0
                params["fsmooth"] = int(float(t2[1])) if len(t2) > 1 else 0
                params["fcut"] = float(t2[2]) if len(t2) > 2 else 0.0
            if len(cards) >= 4 and not cards[3].is_blank:
                t3 = cards[3].tokens()
                params["fun1"] = int(float(t3[0])) if len(t3) > 0 else 0
                params["fct1"] = params["fun1"]
                params["fun2"] = int(float(t3[1])) if len(t3) > 1 else 0
                params["fct2"] = params["fun2"]
                params["ascale1"] = float(t3[2]) if len(t3) > 2 else 1.0
                params["fscale1"] = float(t3[3]) if len(t3) > 3 else 1.0
                params["ascale2"] = float(t3[4]) if len(t3) > 4 else 1.0
                params["fscale2"] = float(t3[5]) if len(t3) > 5 else 1.0

    elif ptype in (8, 13):  # SPR_GENE, SPR_BEAM
        pname = "SPR_GENE" if ptype == 8 else "SPR_BEAM"
        if not cards:
            log.error(f"/PROP/{pname}/{block.user_id}: missing cards", block.source)
            return
        if block.fixed:
            c0 = cards[0].cut("PROP_SPR_GENE_0")
            params["mass"] = _fval(c0[0]) if len(c0) > 0 else 0.0
            params["inertia"] = _fval(c0[1]) if len(c0) > 1 else 0.0
            params["skew_id"] = _ival(c0[2]) if len(c0) > 2 else 0
            params["isensor"] = _ival(c0[3]) if len(c0) > 3 else 0
            params["sens_id"] = params["isensor"]
            params["isflag"] = _ival(c0[4]) if len(c0) > 4 else 0
            params["ifail"] = _ival(c0[5]) if len(c0) > 5 else 0
            params["ifail2"] = _ival(c0[6]) if len(c0) > 6 else 0
            params["iequil"] = _ival(c0[7]) if len(c0) > 7 else 0

            cards_per_dof = 3 if (len(cards) - 1 >= 18) else 2
            c_idx = 1
            for dof_i in range(1, 7):
                k_val, c_val = 0.0, 0.0
                if c_idx < len(cards) and not cards[c_idx].is_blank:
                    c1 = cards[c_idx].cut("PROP_SPR_GENE_1")
                    k_val = _fval(c1[0]) if len(c1) > 0 else 0.0
                    c_val = _fval(c1[1]) if len(c1) > 1 else 0.0
                    params[f"a{dof_i}"] = _fval(c1[2]) if len(c1) > 2 else 0.0
                    params[f"b{dof_i}"] = _fval(c1[3]) if len(c1) > 3 else 0.0
                    params[f"d{dof_i}"] = _fval(c1[4]) if len(c1) > 4 else 0.0
                c_idx += 1
                if c_idx < len(cards) and not cards[c_idx].is_blank:
                    c2 = cards[c_idx].cut("PROP_SPR_GENE_2")
                    params[f"fun_a{dof_i}"] = _ival(c2[0]) if len(c2) > 0 else 0
                    params[f"hflag{dof_i}"] = _ival(c2[1]) if len(c2) > 1 else 0
                    params[f"fun_b{dof_i}"] = _ival(c2[2]) if len(c2) > 2 else 0
                    params[f"fun_c{dof_i}"] = _ival(c2[3]) if len(c2) > 3 else 0
                    params[f"fun_d{dof_i}"] = _ival(c2[4]) if len(c2) > 4 else 0
                    if len(c2) > 7:
                        params[f"min_rup{dof_i}"] = _fval(c2[6])
                        params[f"max_rup{dof_i}"] = _fval(c2[7])
                    elif len(c2) > 5:
                        params[f"min_rup{dof_i}"] = _fval(c2[4])
                        params[f"max_rup{dof_i}"] = _fval(c2[5])
                c_idx += 1
                if cards_per_dof >= 3:
                    if c_idx < len(cards) and not cards[c_idx].is_blank:
                        c3 = cards[c_idx].cut("PROP_SPR_GENE_3")
                        params[f"v0_{dof_i}"] = _fval(c3[0]) if len(c3) > 0 else 0.0
                        params[f"f0_{dof_i}"] = _fval(c3[1]) if len(c3) > 1 else 0.0
                        params[f"scale{dof_i}"] = _fval(c3[2], 1.0) if len(c3) > 2 else 1.0
                        params[f"hscale{dof_i}"] = _fval(c3[3], 1.0) if len(c3) > 3 else 1.0
                    c_idx += 1
                params[f"k{dof_i}"] = k_val
                params[f"c{dof_i}"] = c_val
                params[f"stiff{dof_i}"] = k_val
                params[f"damp{dof_i}"] = c_val
        else:
            t0 = cards[0].tokens()
            params["mass"] = float(t0[0]) if len(t0) > 0 else 0.0
            params["inertia"] = float(t0[1]) if len(t0) > 1 else 0.0
            params["skew_id"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["isensor"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["sens_id"] = params["isensor"]
            params["isflag"] = int(float(t0[4])) if len(t0) > 4 else 0
            params["ifail"] = int(float(t0[5])) if len(t0) > 5 else 0
            if len(t0) > 7:
                params["ifail2"] = int(float(t0[6]))
                params["iequil"] = int(float(t0[7]))
            elif len(t0) > 6:
                params["iequil"] = int(float(t0[6]))

            cards_per_dof = 3 if (len(cards) - 1 >= 18) else 2
            c_idx = 1
            for dof_i in range(1, 7):
                k_val, c_val = 0.0, 0.0
                if c_idx < len(cards) and not cards[c_idx].is_blank:
                    t1 = cards[c_idx].tokens()
                    k_val = float(t1[0]) if len(t1) > 0 else 0.0
                    c_val = float(t1[1]) if len(t1) > 1 else 0.0
                    params[f"a{dof_i}"] = float(t1[2]) if len(t1) > 2 else 0.0
                    params[f"b{dof_i}"] = float(t1[3]) if len(t1) > 3 else 0.0
                    params[f"d{dof_i}"] = float(t1[4]) if len(t1) > 4 else 0.0
                c_idx += 1
                if c_idx < len(cards) and not cards[c_idx].is_blank:
                    t2 = cards[c_idx].tokens()
                    params[f"fun_a{dof_i}"] = int(float(t2[0])) if len(t2) > 0 else 0
                    params[f"hflag{dof_i}"] = int(float(t2[1])) if len(t2) > 1 else 0
                    params[f"fun_b{dof_i}"] = int(float(t2[2])) if len(t2) > 2 else 0
                    params[f"fun_c{dof_i}"] = int(float(t2[3])) if len(t2) > 3 else 0
                    if len(t2) >= 8:
                        params[f"fun_d{dof_i}"] = int(float(t2[4]))
                        params[f"min_rup{dof_i}"] = float(t2[6])
                        params[f"max_rup{dof_i}"] = float(t2[7])
                    elif len(t2) >= 7:
                        params[f"fun_d{dof_i}"] = int(float(t2[4]))
                        params[f"min_rup{dof_i}"] = float(t2[5])
                        params[f"max_rup{dof_i}"] = float(t2[6])
                    elif len(t2) >= 6:
                        params[f"min_rup{dof_i}"] = float(t2[4])
                        params[f"max_rup{dof_i}"] = float(t2[5])
                c_idx += 1
                if cards_per_dof >= 3:
                    if c_idx < len(cards) and not cards[c_idx].is_blank:
                        t3 = cards[c_idx].tokens()
                        params[f"v0_{dof_i}"] = float(t3[0]) if len(t3) > 0 else 0.0
                        params[f"f0_{dof_i}"] = float(t3[1]) if len(t3) > 1 else 0.0
                        params[f"scale{dof_i}"] = float(t3[2]) if len(t3) > 2 else 1.0
                        params[f"hscale{dof_i}"] = float(t3[3]) if len(t3) > 3 else 1.0
                    c_idx += 1
                params[f"k{dof_i}"] = k_val
                params[f"c{dof_i}"] = c_val
                params[f"stiff{dof_i}"] = k_val
                params[f"damp{dof_i}"] = c_val
        params["k"] = params.get("k1", 0.0)
        params["c"] = params.get("c1", 0.0)

    elif ptype == 12:  # SPR_PUL
        if not cards:
            log.error(f"/PROP/SPR_PUL/{block.user_id}: missing cards", block.source)
            return
        if block.fixed:
            c0 = cards[0].cut("PROP_TYPE12_1")
            params["mass"] = _fval(c0[0]) if len(c0) > 0 else 0.0
            params["isensor"] = _ival(c0[2]) if len(c0) > 2 else 0
            params["sens_id"] = params["isensor"]
            params["isflag"] = _ival(c0[3]) if len(c0) > 3 else 0
            params["ileng"] = _ival(c0[4]) if len(c0) > 4 else 0
            params["fric"] = _fval(c0[5]) if len(c0) > 5 else 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                c1 = cards[1].cut("PROP_TYPE12_2")
                params["stiff1"] = _fval(c1[0]) if len(c1) > 0 else 0.0
                params["stiff"] = params["stiff1"]
                params["k"] = params["stiff1"]
                params["damp1"] = _fval(c1[1]) if len(c1) > 1 else 0.0
                params["damp"] = params["damp1"]
                params["c"] = params["damp1"]
                params["acoeft1"] = _fval(c1[2], 1.0) if len(c1) > 2 and c1[2].strip() else 1.0
                params["a"] = params["acoeft1"]
                params["bcoeft1"] = _fval(c1[3]) if len(c1) > 3 else 0.0
                params["b"] = params["bcoeft1"]
                params["dcoeft1"] = _fval(c1[4], 1.0) if len(c1) > 4 and c1[4].strip() else 1.0
                params["d"] = params["dcoeft1"]
            if len(cards) > 2 and not cards[2].is_blank:
                c2 = cards[2].cut("PROP_TYPE12_3")
                if len(c2) >= 8 and (c2[6].strip() or c2[7].strip()):
                    params["fun_a1"] = _ival(c2[0]) if len(c2) > 0 else 0
                    params["fun_a"] = params["fun_a1"]
                    params["fct_id1"] = params["fun_a1"]
                    params["hflag1"] = _ival(c2[1]) if len(c2) > 1 else 0
                    params["hflag"] = params["hflag1"]
                    params["fun_b1"] = _ival(c2[2]) if len(c2) > 2 else 0
                    params["fun_b"] = params["fun_b1"]
                    params["fct_id2"] = params["fun_b1"]
                    params["fct_id31"] = _ival(c2[3]) if len(c2) > 3 else 0
                    params["fun_a2"] = _ival(c2[4]) if len(c2) > 4 else 0
                    params["min_rup1"] = _fval(c2[6], -1.0e30) if len(c2) > 6 and c2[6].strip() else -1.0e30
                    params["min_rup"] = params["min_rup1"]
                    params["delta_min"] = params["min_rup1"]
                    params["max_rup1"] = _fval(c2[7], 1.0e30) if len(c2) > 7 and c2[7].strip() else 1.0e30
                    params["max_rup"] = params["max_rup1"]
                    params["delta_max"] = params["max_rup1"]
                else:
                    c2_old = cards[2].cut("PROP_TYPE12_3_OLD")
                    params["fun_a1"] = _ival(c2_old[0]) if len(c2_old) > 0 else 0
                    params["fun_a"] = params["fun_a1"]
                    params["fct_id1"] = params["fun_a1"]
                    params["hflag1"] = _ival(c2_old[1]) if len(c2_old) > 1 else 0
                    params["hflag"] = params["hflag1"]
                    params["fun_b1"] = _ival(c2_old[2]) if len(c2_old) > 2 else 0
                    params["fun_b"] = params["fun_b1"]
                    params["fct_id2"] = params["fun_b1"]
                    params["min_rup1"] = _fval(c2_old[4], -1.0e30) if len(c2_old) > 4 and c2_old[4].strip() else -1.0e30
                    params["min_rup"] = params["min_rup1"]
                    params["delta_min"] = params["min_rup1"]
                    params["max_rup1"] = _fval(c2_old[5], 1.0e30) if len(c2_old) > 5 and c2_old[5].strip() else 1.0e30
                    params["max_rup"] = params["max_rup1"]
                    params["delta_max"] = params["max_rup1"]
            if len(cards) > 3 and not cards[3].is_blank:
                c3 = cards[3].cut("PROP_TYPE12_4")
                params["prop_x_f"] = _fval(c3[0], 1.0) if len(c3) > 0 and c3[0].strip() else 1.0
                params["fscale"] = params["prop_x_f"]
                params["prop_x_e"] = _fval(c3[1]) if len(c3) > 1 else 0.0
                params["e"] = params["prop_x_e"]
                params["scale1"] = _fval(c3[2], 1.0) if len(c3) > 2 and c3[2].strip() else 1.0
                params["ascale"] = params["scale1"]
                params["h"] = _fval(c3[3], 1.0) if len(c3) > 3 and c3[3].strip() else 1.0
            if len(cards) > 4 and not cards[4].is_blank:
                c4 = cards[4].cut("PROP_TYPE12_5")
                params["funct_id"] = _ival(c4[0]) if len(c4) > 0 else 0
                params["fct_idfr"] = params["funct_id"]
                params["ifric"] = _ival(c4[1]) if len(c4) > 1 else 0
                params["scale2"] = _fval(c4[2], 1.0) if len(c4) > 2 and c4[2].strip() else 1.0
                params["yscale_f"] = params["scale2"]
                params["scale3"] = _fval(c4[3], 1.0) if len(c4) > 3 and c4[3].strip() else 1.0
                params["xscale_f"] = params["scale3"]
                params["f_min"] = _fval(c4[4], -1.0e30) if len(c4) > 4 and c4[4].strip() else -1.0e30
                params["f_max"] = _fval(c4[5], 1.0e30) if len(c4) > 5 and c4[5].strip() else 1.0e30
        else:
            t0 = cards[0].tokens()
            params["mass"] = float(t0[0]) if len(t0) > 0 else 0.0
            if len(t0) == 5:
                params["isensor"] = int(float(t0[1]))
                params["isflag"] = int(float(t0[2]))
                params["ileng"] = int(float(t0[3]))
                params["fric"] = float(t0[4])
            elif len(t0) >= 6:
                params["isensor"] = int(float(t0[2]))
                params["isflag"] = int(float(t0[3]))
                params["ileng"] = int(float(t0[4]))
                params["fric"] = float(t0[5])
            elif len(t0) == 2:
                params["fric"] = float(t0[1])
            params["sens_id"] = params.get("isensor", 0)

            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                params["stiff1"] = float(t1[0]) if len(t1) > 0 else 0.0
                params["stiff"] = params["stiff1"]
                params["k"] = params["stiff1"]
                params["damp1"] = float(t1[1]) if len(t1) > 1 else 0.0
                params["damp"] = params["damp1"]
                params["c"] = params["damp1"]
                params["acoeft1"] = float(t1[2]) if len(t1) > 2 else 1.0
                params["a"] = params["acoeft1"]
                params["bcoeft1"] = float(t1[3]) if len(t1) > 3 else 0.0
                params["b"] = params["bcoeft1"]
                params["dcoeft1"] = float(t1[4]) if len(t1) > 4 else 1.0
                params["d"] = params["dcoeft1"]
            if len(cards) > 2 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                params["fun_a1"] = int(float(t2[0])) if len(t2) > 0 else 0
                params["fun_a"] = params["fun_a1"]
                params["fct_id1"] = params["fun_a1"]
                params["hflag1"] = int(float(t2[1])) if len(t2) > 1 else 0
                params["hflag"] = params["hflag1"]
                params["fun_b1"] = int(float(t2[2])) if len(t2) > 2 else 0
                params["fun_b"] = params["fun_b1"]
                params["fct_id2"] = params["fun_b1"]
                if len(t2) >= 7:
                    params["fct_id31"] = int(float(t2[3]))
                    params["fun_a2"] = int(float(t2[4]))
                    params["min_rup1"] = float(t2[5])
                    params["max_rup1"] = float(t2[6])
                elif len(t2) >= 5:
                    params["min_rup1"] = float(t2[3])
                    params["max_rup1"] = float(t2[4])
                else:
                    params["min_rup1"] = -1.0e30
                    params["max_rup1"] = 1.0e30
                params["min_rup"] = params["min_rup1"]
                params["delta_min"] = params["min_rup1"]
                params["max_rup"] = params["max_rup1"]
                params["delta_max"] = params["max_rup1"]
            if len(cards) > 3 and not cards[3].is_blank:
                t3 = cards[3].tokens()
                params["prop_x_f"] = float(t3[0]) if len(t3) > 0 else 1.0
                params["fscale"] = params["prop_x_f"]
                params["prop_x_e"] = float(t3[1]) if len(t3) > 1 else 0.0
                params["e"] = params["prop_x_e"]
                params["scale1"] = float(t3[2]) if len(t3) > 2 else 1.0
                params["ascale"] = params["scale1"]
                params["h"] = float(t3[3]) if len(t3) > 3 else 1.0
            if len(cards) > 4 and not cards[4].is_blank:
                t4 = cards[4].tokens()
                params["funct_id"] = int(float(t4[0])) if len(t4) > 0 else 0
                params["fct_idfr"] = params["funct_id"]
                params["ifric"] = int(float(t4[1])) if len(t4) > 1 else 0
                params["scale2"] = float(t4[2]) if len(t4) > 2 else 1.0
                params["yscale_f"] = params["scale2"]
                params["scale3"] = float(t4[3]) if len(t4) > 3 else 1.0
                params["xscale_f"] = params["scale3"]
                params["f_min"] = float(t4[4]) if len(t4) > 4 else -1.0e30
                params["f_max"] = float(t4[5]) if len(t4) > 5 else 1.0e30

        from ...model.entities import PropType12
        p12 = PropType12(
            id=block.user_id,
            mass=params.get("mass", 0.0),
            isensor=params.get("isensor", 0),
            isflag=params.get("isflag", 0),
            ileng=params.get("ileng", 0),
            fric=params.get("fric", 0.0),
            stiff1=params.get("stiff1", 0.0),
            damp1=params.get("damp1", 0.0),
            acoeft1=params.get("acoeft1", 1.0),
            bcoeft1=params.get("bcoeft1", 0.0),
            dcoeft1=params.get("dcoeft1", 1.0),
            fun_a1=params.get("fun_a1", 0),
            hflag1=params.get("hflag1", 0),
            fun_b1=params.get("fun_b1", 0),
            fct_id31=params.get("fct_id31", 0),
            fun_a2=params.get("fun_a2", 0),
            min_rup1=params.get("min_rup1", -1.0e30),
            max_rup1=params.get("max_rup1", 1.0e30),
            prop_x_f=params.get("prop_x_f", 1.0),
            prop_x_e=params.get("prop_x_e", 0.0),
            scale1=params.get("scale1", 1.0),
            h=params.get("h", 1.0),
            funct_id=params.get("funct_id", 0),
            ifric=params.get("ifric", 0),
            scale2=params.get("scale2", 1.0),
            scale3=params.get("scale3", 1.0),
            f_min=params.get("f_min", -1.0e30),
            f_max=params.get("f_max", 1.0e30),
            title=title,
        )
        model.prop_type12s[block.user_id] = p12

    elif ptype == 15:  # POROUS
        from ...common.constants import DEFAULT_HOURGLASS, DEFAULT_QA, DEFAULT_QB
        params = {"qa": DEFAULT_QA, "qb": DEFAULT_QB, "h": DEFAULT_HOURGLASS, "por": 0.0}
        if block.fixed:
            c_idx = 1
            if c_idx < len(cards) and not cards[c_idx].is_blank:
                c1 = cards[c_idx].cut("PROP_POROUS_1")
                params["qa"] = _fval(c1[0], DEFAULT_QA) if len(c1) > 0 else DEFAULT_QA
                params["qb"] = _fval(c1[1], DEFAULT_QB) if len(c1) > 1 else DEFAULT_QB
                params["h"] = _fval(c1[2], DEFAULT_HOURGLASS) if len(c1) > 2 else DEFAULT_HOURGLASS
            c_idx += 1
            if c_idx < len(cards) and not cards[c_idx].is_blank:
                c2 = cards[c_idx].cut("PROP_POROUS_2")
                params["por"] = _fval(c2[0], 0.0) if len(c2) > 0 else 0.0
            c_idx += 1
            if c_idx < len(cards) and not cards[c_idx].is_blank:
                c3 = cards[c_idx].cut("PROP_POROUS_3")
                params["pdir1"] = _fval(c3[0], 0.0) if len(c3) > 0 else 0.0
                params["pdir2"] = _fval(c3[1], 0.0) if len(c3) > 1 else 0.0
                params["pdir3"] = _fval(c3[2], 0.0) if len(c3) > 2 else 0.0
            c_idx += 1
            if c_idx < len(cards) and not cards[c_idx].is_blank:
                c4 = cards[c_idx].cut("PROP_POROUS_4")
                params["skew_id"] = _ival(c4[0]) if len(c4) > 0 else 0
                params["iflag"] = _ival(c4[1]) if len(c4) > 1 else 0
            c_idx += 1
            if c_idx < len(cards) and not cards[c_idx].is_blank:
                c5 = cards[c_idx].cut("PROP_POROUS_5")
                params["i_th"] = _ival(c5[0]) if len(c5) > 0 else 0
                params["alpha"] = _fval(c5[1], 0.0) if len(c5) > 1 else 0.0
                params["thick"] = _fval(c5[2], 0.0) if len(c5) > 2 else 0.0
            c_idx += 1
            if c_idx < len(cards) and not cards[c_idx].is_blank:
                c6 = cards[c_idx].cut("PROP_POROUS_6")
                params["irby"] = _ival(c6[0]) if len(c6) > 0 else 0
        else:
            non_blank = [c for c in cards if not c.is_blank]
            if len(non_blank) >= 1:
                t0 = non_blank[0].tokens()
                params["qa"] = float(t0[0]) if len(t0) > 0 else DEFAULT_QA
                params["qb"] = float(t0[1]) if len(t0) > 1 else DEFAULT_QB
                params["h"] = float(t0[2]) if len(t0) > 2 else DEFAULT_HOURGLASS
            if len(non_blank) >= 2:
                t1 = non_blank[1].tokens()
                params["por"] = float(t1[0]) if len(t1) > 0 else 0.0
            if len(non_blank) >= 3:
                t2 = non_blank[2].tokens()
                params["pdir1"] = float(t2[0]) if len(t2) > 0 else 0.0
                params["pdir2"] = float(t2[1]) if len(t2) > 1 else 0.0
                params["pdir3"] = float(t2[2]) if len(t2) > 2 else 0.0
            if len(non_blank) >= 4:
                t3 = non_blank[3].tokens()
                params["skew_id"] = int(float(t3[0])) if len(t3) > 0 else 0
                params["iflag"] = int(float(t3[1])) if len(t3) > 1 else 0
            if len(non_blank) >= 5:
                t4 = non_blank[4].tokens()
                params["i_th"] = int(float(t4[0])) if len(t4) > 0 else 0
                params["alpha"] = float(t4[1]) if len(t4) > 1 else 0.0
                params["thick"] = float(t4[2]) if len(t4) > 2 else 0.0
            if len(non_blank) >= 6:
                t5 = non_blank[5].tokens()
                params["irby"] = int(float(t5[0])) if len(t5) > 0 else 0

    elif ptype == 23:  # SPR_MAT
        if not cards:
            log.error(f"/PROP/SPR_MAT/{block.user_id}: missing cards", block.source)
            return
        if block.fixed:
            c0 = cards[0].cut("PROP_SPR_MAT_0")
            params["imass"] = _ival(c0[0]) if len(c0) > 0 else 2
            av = _fval(c0[2]) if len(c0) > 2 else 0.0
            params["area_volume"] = av
            params["area"] = av if params["imass"] == 1 else 1.0
            params["volume"] = av if params["imass"] == 2 else 0.0
            params["inertia"] = _fval(c0[3]) if len(c0) > 3 else 0.0
            params["skew_id"] = _ival(c0[4]) if len(c0) > 4 else 0
            params["isensor"] = _ival(c0[5]) if len(c0) > 5 else 0
            params["sens_id"] = params["isensor"]
            params["isflag"] = _ival(c0[6]) if len(c0) > 6 else 0
        else:
            t0 = cards[0].tokens()
            params["imass"] = int(float(t0[0])) if len(t0) > 0 else 2
            av = float(t0[1]) if len(t0) > 1 else 0.0
            params["area_volume"] = av
            params["area"] = av if params["imass"] == 1 else 1.0
            params["volume"] = av if params["imass"] == 2 else 0.0
            params["inertia"] = float(t0[2]) if len(t0) > 2 else 0.0
            params["skew_id"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["isensor"] = int(float(t0[4])) if len(t0) > 4 else 0
            params["isflag"] = int(float(t0[5])) if len(t0) > 5 else 0
        from ...model.entities import PropType23
        model.prop_type23s[block.user_id] = PropType23(
            id=block.user_id,
            title=title,
            imass=params.get("imass", 2),
            area_or_volume=params.get("area_volume", 0.0),
            inertia=params.get("inertia", 0.0),
            skew_id=params.get("skew_id", 0),
            sensor_id=params.get("sensor_id", params.get("isensor", 0)),
            isflag=params.get("isflag", 0),
            params=params,
        )

    elif ptype == 25:  # SPR_AXI
        if not cards:
            log.error(f"/PROP/SPR_AXI/{block.user_id}: missing cards", block.source)
            return
        if block.fixed:
            c0 = cards[0].cut("PROP_SPR_AXI_0")
            params["mass"] = _fval(c0[0]) if len(c0) > 0 else 0.0
            params["inertia"] = _fval(c0[1]) if len(c0) > 1 else 0.0
            params["skew_id"] = _ival(c0[2]) if len(c0) > 2 else 0
            params["isensor"] = _ival(c0[3]) if len(c0) > 3 else 0
            params["sens_id"] = params["isensor"]
            params["isflag"] = _ival(c0[4]) if len(c0) > 4 else 0
            params["ifail"] = _ival(c0[5]) if len(c0) > 5 else 0
            params["ileng"] = _ival(c0[6]) if len(c0) > 6 else 0
            params["ifail2"] = _ival(c0[7]) if len(c0) > 7 else 0

            c_idx = 1
            for name in ("tens", "shear", "tors"):
                stiff, damp = 0.0, 0.0
                if c_idx < len(cards) and not cards[c_idx].is_blank:
                    c1 = cards[c_idx].cut("PROP_SPR_AXI_1")
                    stiff = _fval(c1[0]) if len(c1) > 0 else 0.0
                    damp = _fval(c1[1]) if len(c1) > 1 else 0.0
                    params[f"a_{name}"] = _fval(c1[2]) if len(c1) > 2 else 0.0
                    params[f"b_{name}"] = _fval(c1[3]) if len(c1) > 3 else 0.0
                    params[f"d_{name}"] = _fval(c1[4]) if len(c1) > 4 else 0.0
                c_idx += 1
                if c_idx < len(cards) and not cards[c_idx].is_blank:
                    c2 = cards[c_idx].cut("PROP_SPR_AXI_2")
                    params[f"fun_a_{name}"] = _ival(c2[0]) if len(c2) > 0 else 0
                    params[f"hflag_{name}"] = _ival(c2[1]) if len(c2) > 1 else 0
                    params[f"fun_b_{name}"] = _ival(c2[2]) if len(c2) > 2 else 0
                    params[f"fun_c_{name}"] = _ival(c2[3]) if len(c2) > 3 else 0
                    params[f"fscale_{name}"] = _fval(c2[4], 1.0) if len(c2) > 4 else 1.0
                    params[f"min_rup_{name}"] = _fval(c2[5]) if len(c2) > 5 else 0.0
                    params[f"max_rup_{name}"] = _fval(c2[6]) if len(c2) > 6 else 0.0
                    params[f"scale_{name}"] = _fval(c2[7], 1.0) if len(c2) > 7 else 1.0
                    params[f"e_{name}"] = _fval(c2[8]) if len(c2) > 8 else 0.0
                c_idx += 1
                params[f"stiff_{name}"] = stiff
                params[f"damp_{name}"] = damp
        else:
            t0 = cards[0].tokens()
            params["mass"] = float(t0[0]) if len(t0) > 0 else 0.0
            params["inertia"] = float(t0[1]) if len(t0) > 1 else 0.0
            params["skew_id"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["isensor"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["sens_id"] = params["isensor"]
            params["isflag"] = int(float(t0[4])) if len(t0) > 4 else 0
            params["ifail"] = int(float(t0[5])) if len(t0) > 5 else 0
            params["ileng"] = int(float(t0[6])) if len(t0) > 6 else 0
            params["ifail2"] = int(float(t0[7])) if len(t0) > 7 else 0

            c_idx = 1
            for name in ("tens", "shear", "tors"):
                stiff, damp = 0.0, 0.0
                if c_idx < len(cards) and not cards[c_idx].is_blank:
                    t1 = cards[c_idx].tokens()
                    stiff = float(t1[0]) if len(t1) > 0 else 0.0
                    damp = float(t1[1]) if len(t1) > 1 else 0.0
                    params[f"a_{name}"] = float(t1[2]) if len(t1) > 2 else 0.0
                    params[f"b_{name}"] = float(t1[3]) if len(t1) > 3 else 0.0
                    params[f"d_{name}"] = float(t1[4]) if len(t1) > 4 else 0.0
                c_idx += 1
                if c_idx < len(cards) and not cards[c_idx].is_blank:
                    t2 = cards[c_idx].tokens()
                    params[f"fun_a_{name}"] = int(float(t2[0])) if len(t2) > 0 else 0
                    params[f"hflag_{name}"] = int(float(t2[1])) if len(t2) > 1 else 0
                    params[f"fun_b_{name}"] = int(float(t2[2])) if len(t2) > 2 else 0
                    params[f"fun_c_{name}"] = int(float(t2[3])) if len(t2) > 3 else 0
                    params[f"fscale_{name}"] = float(t2[4]) if len(t2) > 4 else 1.0
                    params[f"min_rup_{name}"] = float(t2[5]) if len(t2) > 5 else 0.0
                    params[f"max_rup_{name}"] = float(t2[6]) if len(t2) > 6 else 0.0
                    params[f"scale_{name}"] = float(t2[7]) if len(t2) > 7 else 1.0
                    params[f"e_{name}"] = float(t2[8]) if len(t2) > 8 else 0.0
                c_idx += 1
                params[f"stiff_{name}"] = stiff
                params[f"damp_{name}"] = damp
        params["k"] = params.get("stiff_tens", 0.0)
        params["c"] = params.get("damp_tens", 0.0)

    elif ptype == 32:  # SPR_PRE
        from .. import prop_reader
        prop = prop_reader.parse_spr_pre(block, log)
        if prop is not None:
            model.properties[block.user_id] = prop
            from ...model.entities import PropType32
            model.prop_type32s[block.user_id] = PropType32(
                id=block.user_id, mass=float(prop.params.get("mass", 0.0)),
                sensor_id=int(prop.params.get("sensor_id", prop.params.get("sens_id", 0))),
                ilock=int(prop.params.get("ilock", 0)),
                stiff0=float(prop.params.get("stif0", prop.params.get("stiff0", 0.0))),
                f1=float(prop.params.get("f1", 0.0)),
                d1=float(prop.params.get("d1", 0.0)),
                e1=float(prop.params.get("e1", 0.0)),
                stiff1=float(prop.params.get("stif1", prop.params.get("stiff1", 0.0))),
                fun_a1=int(prop.params.get("fct_id1", prop.params.get("fun_a1", 0))),
                fun_b1=int(prop.params.get("fct_id2", prop.params.get("fun_b1", 0))),
                scale_t=float(prop.params.get("tscal", prop.params.get("scale_t", 1.0))),
                scale_d=float(prop.params.get("dscal", prop.params.get("scale_d", 1.0))),
                scale_f=float(prop.params.get("fscal", prop.params.get("scale_f", 1.0))),
                title=title,
            )
        return

    if ptype in (51, 17, 34, 12, 15, 23):
        from ..prop_reader import InactiveProperty, _universal_geo_params
        full_params = _universal_geo_params()
        full_params.update(params)
        model.properties[block.user_id] = InactiveProperty(
            id=block.user_id, type=ptype, title=title, params=full_params,
            prop_name=f"/PROP/{block.parts[1] if len(block.parts) > 1 else 'TYPE' + str(ptype)}"
        )
    elif ptype == 27:
        from ...model.entities import PropType27
        from ..prop_reader import _universal_geo_params
        p27 = PropType27(
            id=block.user_id,
            mass=float(params.get("mass", 0.0)),
            sens_id=int(params.get("sens_id", 0)),
            isflag=int(params.get("isflag", 0)),
            ileng=int(params.get("ileng", 0)),
            itens=int(params.get("itens", 0)),
            ifail=int(params.get("ifail", 0)),
            stiff=float(params.get("stiff", 0.0)),
            damp=float(params.get("damp", 0.0)),
            nexp=float(params.get("nexp", 1.0)),
            delta_min=float(params.get("delta_min", 0.0)),
            delta_max=float(params.get("delta_max", 0.0)),
            gap=float(params.get("gap", 0.0)),
            fsmooth=int(params.get("fsmooth", 0)),
            fcut=float(params.get("fcut", 0.0)),
            fct_id1=int(params.get("fun1", 0)),
            fct_id2=int(params.get("fun2", 0)),
            ascale1=float(params.get("ascale1", 1.0)),
            fscale1=float(params.get("fscale1", 1.0)),
            ascale2=float(params.get("ascale2", 1.0)),
            fscale2=float(params.get("fscale2", 1.0)),
            title=title,
        )
        model.prop_spr_bdamps[block.user_id] = p27
        full_params = _universal_geo_params()
        full_params.update(params)
        model.properties[block.user_id] = Property(
            id=block.user_id, type=27, title=title, params=full_params,
        )
        return
    elif ptype == 26:
        read_prop_spr_tab(block, model, log)
        return
    elif ptype == 25:
        from ...model.entities import PropType25
        tension = {
            "stiff": float(params.get("stiff_tens", 0.0)),
            "damp": float(params.get("damp_tens", 0.0)),
            "a": float(params.get("a_tens", 0.0)),
            "b": float(params.get("b_tens", 0.0)),
            "d": float(params.get("d_tens", 0.0)),
            "fun_a": int(params.get("fun_a_tens", 0)),
            "hflag": int(params.get("hflag_tens", 0)),
            "fun_b": int(params.get("fun_b_tens", 0)),
            "fun_c": int(params.get("fun_c_tens", 0)),
            "min_rup": float(params.get("min_rup_tens", 0.0)),
            "max_rup": float(params.get("max_rup_tens", 0.0)),
        }
        shear = {
            "stiff": float(params.get("stiff_shear", 0.0)),
            "damp": float(params.get("damp_shear", 0.0)),
            "a": float(params.get("a_shear", 0.0)),
            "b": float(params.get("b_shear", 0.0)),
            "d": float(params.get("d_shear", 0.0)),
            "fun_a": int(params.get("fun_a_shear", 0)),
            "hflag": int(params.get("hflag_shear", 0)),
            "fun_b": int(params.get("fun_b_shear", 0)),
            "fun_c": int(params.get("fun_c_shear", 0)),
            "min_rup": float(params.get("min_rup_shear", 0.0)),
            "max_rup": float(params.get("max_rup_shear", 0.0)),
        }
        params["tension"] = tension
        params["shear"] = shear
        model.prop_type25s[block.user_id] = PropType25(
            id=block.user_id, mass=float(params.get("mass", 0.0)),
            inertia=float(params.get("inertia", 0.0)),
            skew_id=int(params.get("skew_id", 0)),
            sensor_id=int(params.get("sensor_id", params.get("sens_id", 0))),
            isflag=int(params.get("isflag", 0)),
            ifail=int(params.get("ifail", 0)),
            ileng=int(params.get("ileng", 0)),
            ifail2=int(params.get("ifail2", 0)),
            tension=tension,
            shear=shear,
            title=title,
        )
    model.properties[block.user_id] = Property(
        id=block.user_id, type=ptype, title=title, params=params)
    if ptype == 5:
        from ...model.entities import PropRivet
        model.prop_rivets[block.user_id] = PropRivet(
            id=block.user_id, title=title,
            mass=params.get("mass", 0.0),
            stiffness=params.get("stiffness", params.get("fn", 0.0)),
            fn_fail=params.get("fn_fail", params.get("fn", 0.0)),
            ft_fail=params.get("ft_fail", params.get("ft", 0.0)),
        )
    elif ptype == 28:
        from ...model.entities import PropXelem
        model.prop_xelems[block.user_id] = PropXelem(
            id=block.user_id, title=title,
            itip=int(params.get("itip", params.get("fun_k", 0))),
            isurf=int(params.get("isurf", params.get("fun_c", 0))),
            alpha=params.get("alpha", params.get("mass", 0.0)),
        )
    elif ptype == 20:
        from ...model.entities import PropType20
        p20 = PropType20(
            id=block.user_id, isolid=int(params.get("itshell", params.get("isolid", 15))),
            ismstr=int(params.get("ismstr", 0)),
            icpre=int(params.get("icpre", params.get("idrill", 0))),
            icstr=int(params.get("icstr", params.get("istrain", 0))),
            inpts_r=int(params.get("inpts_r", params.get("nip", 2))),
            inpts_s=int(params.get("inpts_s", params.get("nip", 2))),
            inpts_t=int(params.get("inpts_t", params.get("nip", 2))),
            iint=int(params.get("iint", params.get("iplas", 1))),
            dn=float(params.get("dn", 0.0)),
            qa=float(params.get("qa", 1.1)),
            qb=float(params.get("qb", 0.05)),
            h=float(params.get("h", 0.1)),
            deltat_min=float(params.get("deltat_min", 0.0)),
            nbp=int(params.get("nbp", 0)),
            title=title,
        )
        model.prop_tshells[block.user_id] = p20
        model.prop_type20s[block.user_id] = p20
        model.props_type20[block.user_id] = p20
    elif ptype == 21:
        from ...model.entities import PropType21
        p21 = PropType21(
            id=block.user_id, isolid=int(params.get("itshell", 15)),
            ismstr=int(params.get("ismstr", 0)),
            icstr=int(params.get("icstr", params.get("istrain", 0))),
            inpts_r=int(params.get("inpts_r", params.get("nip", 2))),
            inpts_s=int(params.get("inpts_s", params.get("nip", 2))),
            inpts_t=int(params.get("inpts_t", params.get("nip", 2))),
            iint=int(params.get("iint", params.get("iplas", 1))),
            dn=float(params.get("dn", 0.0)),
            qa=float(params.get("qa", 1.1)),
            qb=float(params.get("qb", 0.05)),
            vx=float(params.get("vx", 1.0)),
            vy=float(params.get("vy", 0.0)),
            vz=float(params.get("vz", 0.0)),
            skew_id=int(params.get("skew_id", 0)),
            iorth=int(params.get("iorth", 0)),
            phi=float(params.get("phi", 0.0)),
            deltat_min=float(params.get("deltat_min", 0.0)),
            title=title,
        )
        model.prop_tsh_orths[block.user_id] = p21
        model.props_type21[block.user_id] = p21
    elif ptype == 22:
        from ...model.entities import PropType22, PropType22Layer
        ly_objs = []
        for ly in params.get("layers", []):
            ly_objs.append(PropType22Layer(
                phi=float(ly.get("phi", 0.0)),
                thick=float(ly.get("thick", 0.0)),
                zi=float(ly.get("zi", 0.0)),
                mat_id=int(ly.get("mat_id", 0)),
            ))
        p22 = PropType22(
            id=block.user_id, isolid=int(params.get("itshell", 15)),
            ismstr=int(params.get("ismstr", 0)),
            icstr=int(params.get("icstr", params.get("istrain", 0))),
            inpts_r=int(params.get("inpts_r", params.get("nip", 2))),
            inpts_s=int(params.get("inpts_s", params.get("nip", 2))),
            inpts_t=int(params.get("inpts_t", params.get("nip", 2))),
            iint=int(params.get("iint", params.get("iplas", 1))),
            dn=float(params.get("dn", 0.0)),
            qa=float(params.get("qa", 1.1)),
            qb=float(params.get("qb", 0.05)),
            vx=float(params.get("vx", 1.0)),
            vy=float(params.get("vy", 0.0)),
            vz=float(params.get("vz", 0.0)),
            skew_id=int(params.get("skew_id", 0)),
            iorth=int(params.get("iorth", 0)),
            ipos=int(params.get("ipos", 0)),
            ashear=float(params.get("ashear", 0.833333)),
            layers=ly_objs,
            deltat_min=float(params.get("deltat_min", 0.0)),
            title=title,
        )
        model.prop_tsh_comps[block.user_id] = p22
        model.props_type22[block.user_id] = p22
    elif ptype == 6:
        from ...model.entities import PropType6
        model.prop_sol_orths[block.user_id] = PropType6(
            id=block.user_id,
            isolid=int(params.get("isolid", 14)),
            ismstr=int(params.get("ismstr", 0)),
            icpre=int(params.get("icpre", 0)),
            itetra10=int(params.get("itetra10", 0)),
            inpts_r=int(params.get("inpts_r", 1)),
            inpts_s=int(params.get("inpts_s", 1)),
            inpts_t=int(params.get("inpts_t", 1)),
            itetra4=int(params.get("itetra4", 0)),
            iframe=int(params.get("iframe", 0)),
            dn=float(params.get("dn", 0.0)),
            qa=float(params.get("qa", 1.1)),
            qb=float(params.get("qb", 0.05)),
            h=float(params.get("h", 0.1)),
            vx=float(params.get("vx", 0.0)),
            vy=float(params.get("vy", 0.0)),
            vz=float(params.get("vz", 0.0)),
            skew_id=int(params.get("skew_id", 0)),
            refplane=int(params.get("ip", 0)),
            orthtrop=int(params.get("iorth", 0)),
            mat_beta=float(params.get("phi", 0.0)),
            px=float(params.get("px", 0.0)),
            py=float(params.get("py", 0.0)),
            pz=float(params.get("pz", 0.0)),
            deltat_min=float(params.get("deltat_min", 0.0)),
            vdef_min=float(params.get("vdef_min", 0.0)),
            vdef_max=float(params.get("vdef_max", 0.0)),
            asp_max=float(params.get("asp_max", 0.0)),
            col_min=float(params.get("col_min", 0.0)),
            ndir=int(params.get("ndir", 0)),
            sphpart_id=int(params.get("sphpart_id", 0)),
            istrain=int(params.get("istrain", 0)),
            ihkt=int(params.get("ihkt", 0)),
            title=title,
        )
    elif ptype == 14:
        from ...model.entities import PropType14
        model.prop_type14s[block.user_id] = PropType14(
            id=block.user_id, isolid=int(params.get("isolid", 14)),
            ismstr=int(params.get("ismstr", 0)),
            icpre=int(params.get("icpre", 0)),
            inpts_r=int(params.get("inpts_r", 1)),
            inpts_s=int(params.get("inpts_s", 1)),
            inpts_t=int(params.get("inpts_t", 1)),
            i_rot=int(params.get("i_rot", 0)),
            iframe=int(params.get("iframe", 0)),
            dn=float(params.get("dn", 0.0)),
            qa=float(params.get("qa", 1.1)),
            qb=float(params.get("qb", 0.05)),
            h=float(params.get("h", 0.1)),
            deltat_min=float(params.get("deltat_min", 0.0)),
            istrain=int(params.get("istrain", 0)),
            qa_l=float(params.get("qa_l", 0.0)),
            qb_l=float(params.get("qb_l", 0.0)),
            h_l=float(params.get("h_l", 0.0)),
            iplas=int(params.get("iplas", 0)),
            icstr=int(params.get("icstr", 0)),
            title=title,
        )
    elif ptype == 43:
        from ...model.entities import PropType43
        p43 = PropType43(
            id=block.user_id, title=title,
            ismstr=int(params.get("ismstr", 0)),
            thick=float(params.get("thick", 0.0)),
        )
        model.prop_type43s[block.user_id] = p43
        model.prop_connects[block.user_id] = p43
    elif ptype == 34:
        from ...model.entities import PropType34
        qa = float(params.get("qa", params.get("alpha", 1.0)))
        qb = float(params.get("qb", params.get("beta", 1.0)))
        alpha1 = float(params.get("alpha1", params.get("q0", 0.0)))
        order = int(params.get("order", params.get("gamma", 1)))
        h0 = float(params.get("h0", params.get("h", 0.0)))
        p34 = PropType34(
            id=block.user_id, mass=float(params.get("mass", 0.0)),
            h0=h0, d0=float(params.get("d0", 0.0)),
            qa=qa, qb=qb, alpha1=alpha1, order=order,
            h=h0, title=title
        )
        model.prop_type34s[block.user_id] = p34
        model.prop_sphs[block.user_id] = p34
        model.prop_prop_sphs[block.user_id] = p34
    elif ptype in (8, 12, 13):
        from ...model.entities import PropType8, PropType13
        dofs = {}
        for i, name in enumerate(["tx", "ty", "tz", "rx", "ry", "rz"], 1):
            dofs[name] = {
                "stiff": float(params.get(f"k{i}", params.get(f"stiff{i}", 0.0))),
                "damp": float(params.get(f"c{i}", params.get(f"damp{i}", 0.0))),
                "a": float(params.get(f"a{i}", 0.0)),
                "b": float(params.get(f"b{i}", 0.0)),
                "d": float(params.get(f"d{i}", 0.0)),
                "fun_a": int(params.get(f"fun_a{i}", 0)),
                "hflag": int(params.get(f"hflag{i}", 0)),
                "fun_b": int(params.get(f"fun_b{i}", 0)),
                "fun_c": int(params.get(f"fun_c{i}", 0)),
                "min_rup": float(params.get(f"min_rup{i}", 0.0)),
                "max_rup": float(params.get(f"max_rup{i}", 0.0)),
            }
        model.prop_type8s[block.user_id] = PropType8(
            id=block.user_id, mass=float(params.get("mass", 0.0)),
            inertia=float(params.get("inertia", 0.0)),
            skew_id=int(params.get("skew_id", 0)),
            sensor_id=int(params.get("sensor_id", params.get("sens_id", 0))),
            isflag=int(params.get("isflag", 0)),
            ifail=int(params.get("ifail", 0)),
            iequil=int(params.get("iequil", 0)),
            dofs=dofs,
            title=title,
        )
        stiff = float(params.get("stiff", params.get("k", params.get("k1", 0.0))))
        f_max = float(params.get("f_max", params.get("fn", params.get("fn_fail", 0.0))))
        p13 = PropType13(id=block.user_id, title=title, stiff=stiff, f_max=f_max, params=params)
        model.prop_spr_pulls[block.user_id] = p13




def read_prop_rivet(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE5`` or ``/PROP/RIVET`` (M143): Rivet / Fastener connector property."""
    from ...model.entities import PropRivet
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    mass, stiffness, fn_fail, ft_fail = 0.0, 0.0, 0.0, 0.0
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("PROP_RIVET_1")
            mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            stiffness = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            fn_fail = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            ft_fail = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        else:
            toks = cards[0].tokens()
            mass = float(toks[0]) if len(toks) > 0 else 0.0
            stiffness = float(toks[1]) if len(toks) > 1 else 0.0
            fn_fail = float(toks[2]) if len(toks) > 2 else 0.0
            ft_fail = float(toks[3]) if len(toks) > 3 else 0.0
    model.prop_rivets[pid] = PropRivet(
        id=pid, title=title, mass=mass, stiffness=stiffness, fn_fail=fn_fail, ft_fail=ft_fail
    )




def read_prop_xelem(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE28`` or ``/PROP/XELEM`` (M143): X-FEM / cohesive element property."""
    from ...model.entities import PropXelem
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    itip, isurf, alpha = 0, 0, 0.0
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("PROP_XELEM_1")
            itip = _ival(f[0]) if len(f) > 0 else 0
            isurf = _ival(f[1]) if len(f) > 1 else 0
            alpha = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        else:
            toks = cards[0].tokens()
            itip = int(float(toks[0])) if len(toks) > 0 else 0
            isurf = int(float(toks[1])) if len(toks) > 1 else 0
            alpha = float(toks[2]) if len(toks) > 2 else 0.0
    model.prop_xelems[pid] = PropXelem(
        id=pid, title=title, itip=itip, isurf=isurf, alpha=alpha
    )




def read_prop_inject1(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/INJECT1`` (M152): Gas injector property."""
    from ...model.entities import PropInject1, PropInject1Gas
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROP/INJECT1/{pid}: missing data card", block.source)
        return
    n_gases, iflow, ascale_t = 0, 0, 0.0
    if block.fixed:
        f = cards[0].cut("PROP_INJECT1_1")
        n_gases = _ival(f[0]) if len(f) > 0 else 0
        iflow = _ival(f[1]) if len(f) > 1 else 0
        ascale_t = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        n_gases = int(float(toks[0])) if len(toks) > 0 else 0
        iflow = int(float(toks[1])) if len(toks) > 1 else 0
        ascale_t = float(toks[2]) if len(toks) > 2 else 0.0

    gases = []
    for i in range(1, len(cards)):
        if cards[i].is_blank:
            continue
        if block.fixed:
            f = cards[i].cut("PROP_INJECT1_2")
            mat_id = _ival(f[0]) if len(f) > 0 else 0
            fun_id_m = _ival(f[1]) if len(f) > 1 else 0
            fun_id_t = _ival(f[2]) if len(f) > 2 else 0
            fscale_m = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            fscale_t = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            toks = cards[i].tokens()
            mat_id = int(float(toks[0])) if len(toks) > 0 else 0
            fun_id_m = int(float(toks[1])) if len(toks) > 1 else 0
            fun_id_t = int(float(toks[2])) if len(toks) > 2 else 0
            fscale_m = float(toks[3]) if len(toks) > 3 else 0.0
            fscale_t = float(toks[4]) if len(toks) > 4 else 0.0
        gases.append(PropInject1Gas(mat_id, fun_id_m, fun_id_t, fscale_m, fscale_t))

    model.prop_inject1s[pid] = PropInject1(
        id=pid, title=title, n_gases=n_gases, iflow=iflow, ascale_t=ascale_t, gases=gases
    )
    from .. import prop_reader
    model.properties[pid] = prop_reader.InactiveProperty(
        id=pid, type=0, title=title, prop_name="INJECT1",
        params={"thick": 1.0, "area": 1.0, "vol": 1.0}
    )




def read_prop_inject2(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/INJECT2`` (M152): Multi-gas mixture property."""
    from ...model.entities import PropInject2, PropInject2Gas
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or len(cards) < 2 or cards[0].is_blank:
        log.error(f"/PROP/INJECT2/{pid}: missing data cards", block.source)
        return
    n_gases, iflow = 0, 0
    fun_id_m, fun_id_t, fscale_m, fscale_t, ascale_t = 0, 0, 0.0, 0.0, 0.0
    if block.fixed:
        f = cards[0].cut("PROP_INJECT2_1")
        n_gases = _ival(f[0]) if len(f) > 0 else 0
        iflow = _ival(f[1]) if len(f) > 1 else 0
        f2 = cards[1].cut("PROP_INJECT2_2")
        fun_id_m = _ival(f2[0]) if len(f2) > 0 else 0
        fun_id_t = _ival(f2[1]) if len(f2) > 1 else 0
        fscale_m = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
        fscale_t = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        ascale_t = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
    else:
        toks = cards[0].tokens()
        n_gases = int(float(toks[0])) if len(toks) > 0 else 0
        iflow = int(float(toks[1])) if len(toks) > 1 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            fun_id_m = int(float(toks2[0])) if len(toks2) > 0 else 0
            fun_id_t = int(float(toks2[1])) if len(toks2) > 1 else 0
            fscale_m = float(toks2[2]) if len(toks2) > 2 else 0.0
            fscale_t = float(toks2[3]) if len(toks2) > 3 else 0.0
            ascale_t = float(toks2[4]) if len(toks2) > 4 else 0.0

    gases = []
    for i in range(2, len(cards)):
        if cards[i].is_blank:
            continue
        if block.fixed:
            f = cards[i].cut("PROP_INJECT2_3")
            mat_id = _ival(f[0]) if len(f) > 0 else 0
            molar_fraction = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            fun_id_mf = _ival(f[2]) if len(f) > 2 else 0
        else:
            toks = cards[i].tokens()
            mat_id = int(float(toks[0])) if len(toks) > 0 else 0
            molar_fraction = float(toks[1]) if len(toks) > 1 else 0.0
            fun_id_mf = int(float(toks[2])) if len(toks) > 2 else 0
        gases.append(PropInject2Gas(mat_id, molar_fraction, fun_id_mf))

    model.prop_inject2s[pid] = PropInject2(
        id=pid, title=title, n_gases=n_gases, iflow=iflow,
        fun_id_m=fun_id_m, fun_id_t=fun_id_t, fscale_m=fscale_m,
        fscale_t=fscale_t, ascale_t=ascale_t, gases=gases
    )
    from .. import prop_reader
    model.properties[pid] = prop_reader.InactiveProperty(
        id=pid, type=0, title=title, prop_name="INJECT2",
        params={"thick": 1.0, "area": 1.0, "vol": 1.0}
    )




def read_prop_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE33`` or ``/PROP/JOINT`` (M152): Kinematic joint property."""
    from ...model.entities import PropJoint
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROP/TYPE33/{pid}: missing data card", block.source)
        return
    joint_type, skew_id = 0, 0
    params = {}
    if block.fixed:
        f = cards[0].cut("PROP_TYPE33_1")
        joint_type = _ival(f[0]) if len(f) > 0 else 0
        skew_id = _ival(f[1]) if len(f) > 1 else 0
        if len(f) > 2: params["p1"] = _fval(f[2], 0.0)
        if len(f) > 3: params["p2"] = _fval(f[3], 0.0)
        if len(f) > 4: params["p3"] = _fval(f[4], 0.0)
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROP_TYPE33_2")
            if len(f2) > 0: params["p4"] = _fval(f2[0], 0.0)
            if len(f2) > 1: params["p5"] = _fval(f2[1], 0.0)
            if len(f2) > 2: params["p6"] = _fval(f2[2], 0.0)
            if len(f2) > 3: params["p7"] = _fval(f2[3], 0.0)
            if len(f2) > 4: params["p8"] = _fval(f2[4], 0.0)
    else:
        toks = cards[0].tokens()
        joint_type = int(float(toks[0])) if len(toks) > 0 else 0
        skew_id = int(float(toks[1])) if len(toks) > 1 else 0
        if len(toks) > 2: params["p1"] = float(toks[2])
        if len(toks) > 3: params["p2"] = float(toks[3])
        if len(toks) > 4: params["p3"] = float(toks[4])
        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            if len(toks2) > 0: params["p4"] = float(toks2[0])
            if len(toks2) > 1: params["p5"] = float(toks2[1])
            if len(toks2) > 2: params["p6"] = float(toks2[2])
            if len(toks2) > 3: params["p7"] = float(toks2[3])
            if len(toks2) > 4: params["p8"] = float(toks2[4])

    model.prop_joints[pid] = PropJoint(
        id=pid, title=title, joint_type=joint_type, skew_id=skew_id, params=params
    )




def read_prop_torsion(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE35`` or ``/PROP/TORSION`` (M152): Torsion bar spring property."""
    from ...model.entities import PropTorsion
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROP/TYPE35/{pid}: missing data card", block.source)
        return
    mass, k_elas, x_lim1, x_lim2, k_post = 0.0, 0.0, 0.0, 0.0, 0.0
    d1, d2, r_load, f_scal = 0.0, 0.0, 0.0, 0.0
    fct_id1, fct_id2, fct_id3, fct_id4 = 0, 0, 0, 0
    if block.fixed:
        f = cards[0].cut("PROP_TYPE35_1")
        mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        k_elas = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        x_lim1 = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        x_lim2 = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        k_post = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROP_TYPE35_2")
            d1 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            d2 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            r_load = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            f_scal = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("PROP_TYPE35_3")
            fct_id1 = _ival(f3[0]) if len(f3) > 0 else 0
            fct_id2 = _ival(f3[1]) if len(f3) > 1 else 0
            fct_id3 = _ival(f3[2]) if len(f3) > 2 else 0
            fct_id4 = _ival(f3[3]) if len(f3) > 3 else 0
    else:
        toks = cards[0].tokens()
        mass = float(toks[0]) if len(toks) > 0 else 0.0
        k_elas = float(toks[1]) if len(toks) > 1 else 0.0
        x_lim1 = float(toks[2]) if len(toks) > 2 else 0.0
        x_lim2 = float(toks[3]) if len(toks) > 3 else 0.0
        k_post = float(toks[4]) if len(toks) > 4 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            d1 = float(toks2[0]) if len(toks2) > 0 else 0.0
            d2 = float(toks2[1]) if len(toks2) > 1 else 0.0
            r_load = float(toks2[2]) if len(toks2) > 2 else 0.0
            f_scal = float(toks2[3]) if len(toks2) > 3 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            toks3 = cards[2].tokens()
            fct_id1 = int(float(toks3[0])) if len(toks3) > 0 else 0
            fct_id2 = int(float(toks3[1])) if len(toks3) > 1 else 0
            fct_id3 = int(float(toks3[2])) if len(toks3) > 2 else 0
            fct_id4 = int(float(toks3[3])) if len(toks3) > 3 else 0

    model.prop_torsions[pid] = PropTorsion(
        id=pid, title=title, mass=mass, k_elas=k_elas, x_lim1=x_lim1, x_lim2=x_lim2,
        k_post=k_post, d1=d1, d2=d2, r_load=r_load, f_scal=f_scal,
        fct_id1=fct_id1, fct_id2=fct_id2, fct_id3=fct_id3, fct_id4=fct_id4
    )




def read_prop_spring_elas_plas(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE36`` (M152): Spring elastic-plastic property."""
    from ...model.entities import PropSpringElasPlas
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROP/TYPE36/{pid}: missing data card", block.source)
        return
    skew_id, i_utyp, pid1, pid2, mid1 = 0, 0, 0, 0, 0
    k_stiff, area, ixx, iyy, izz = 0.0, 0.0, 0.0, 0.0, 0.0
    params = {}
    if block.fixed:
        f = cards[0].cut("PROP_TYPE36_1")
        skew_id = _ival(f[0]) if len(f) > 0 else 0
        i_utyp = _ival(f[1]) if len(f) > 1 else 0
        pid1 = _ival(f[2]) if len(f) > 2 else 0
        pid2 = _ival(f[3]) if len(f) > 3 else 0
        k_stiff = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROP_TYPE36_2")
            mid1 = _ival(f2[0]) if len(f2) > 0 else 0
            area = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            ixx = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            iyy = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
            izz = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
    else:
        toks = cards[0].tokens()
        skew_id = int(float(toks[0])) if len(toks) > 0 else 0
        i_utyp = int(float(toks[1])) if len(toks) > 1 else 0
        pid1 = int(float(toks[2])) if len(toks) > 2 else 0
        pid2 = int(float(toks[3])) if len(toks) > 3 else 0
        k_stiff = float(toks[4]) if len(toks) > 4 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            mid1 = int(float(toks2[0])) if len(toks2) > 0 else 0
            area = float(toks2[1]) if len(toks2) > 1 else 0.0
            ixx = float(toks2[2]) if len(toks2) > 2 else 0.0
            iyy = float(toks2[3]) if len(toks2) > 3 else 0.0
            izz = float(toks2[4]) if len(toks2) > 4 else 0.0

    model.prop_spring_elas_plas[pid] = PropSpringElasPlas(
        id=pid, title=title, skew_id=skew_id, i_utyp=i_utyp, pid1=pid1, pid2=pid2,
        mid1=mid1, k_stiff=k_stiff, area=area, ixx=ixx, iyy=iyy, izz=izz, params=params
    )




def read_prop_spring_beam(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE44`` (M152): Spring beam property."""
    from ...model.entities import PropSpringBeam
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROP/TYPE44/{pid}: missing data card", block.source)
        return
    skew_id, idamp, nc_filter = 0, 0, 0
    params = {}
    if block.fixed:
        f = cards[0].cut("PROP_SPRING_BEAM_1")
        skew_id = _ival(f[0]) if len(f) > 0 else 0
        idamp = _ival(f[1]) if len(f) > 1 else 0
        nc_filter = _ival(f[2]) if len(f) > 2 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROP_SPRING_BEAM_2")
            if len(f2) > 0: params["p1"] = _fval(f2[0], 0.0)
            if len(f2) > 1: params["p2"] = _fval(f2[1], 0.0)
            if len(f2) > 2: params["p3"] = _fval(f2[2], 0.0)
            if len(f2) > 3: params["p4"] = _fval(f2[3], 0.0)
    else:
        toks = cards[0].tokens()
        skew_id = int(float(toks[0])) if len(toks) > 0 else 0
        idamp = int(float(toks[1])) if len(toks) > 1 else 0
        nc_filter = int(float(toks[2])) if len(toks) > 2 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            if len(toks2) > 0: params["p1"] = float(toks2[0])
            if len(toks2) > 1: params["p2"] = float(toks2[1])
            if len(toks2) > 2: params["p3"] = float(toks2[2])
            if len(toks2) > 3: params["p4"] = float(toks2[3])

    model.prop_spring_beams[pid] = PropSpringBeam(
        id=pid, title=title, skew_id=skew_id, idamp=idamp, nc_filter=nc_filter, params=params
    )




def read_prop_spotweld(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE45`` (M152): Spotweld property."""
    from ...model.entities import PropSpotweld
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROP/TYPE45/{pid}: missing data card", block.source)
        return
    skew_id, sensor_id = 0, 0
    knn, cr, scf = 0.0, 0.0, 0.0
    params = {}
    if block.fixed:
        f = cards[0].cut("PROP_TYPE45_1")
        skew_id = _ival(f[0]) if len(f) > 0 else 0
        knn = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        cr = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        scf = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        sensor_id = _ival(f[4]) if len(f) > 4 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROP_TYPE45_2")
            if len(f2) > 0: params["p1"] = _fval(f2[0], 0.0)
            if len(f2) > 1: params["p2"] = _fval(f2[1], 0.0)
            if len(f2) > 2: params["p3"] = _fval(f2[2], 0.0)
            if len(f2) > 3: params["p4"] = _fval(f2[3], 0.0)
    else:
        toks = cards[0].tokens()
        skew_id = int(float(toks[0])) if len(toks) > 0 else 0
        knn = float(toks[1]) if len(toks) > 1 else 0.0
        cr = float(toks[2]) if len(toks) > 2 else 0.0
        scf = float(toks[3]) if len(toks) > 3 else 0.0
        sensor_id = int(float(toks[4])) if len(toks) > 4 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            if len(toks2) > 0: params["p1"] = float(toks2[0])
            if len(toks2) > 1: params["p2"] = float(toks2[1])
            if len(toks2) > 2: params["p3"] = float(toks2[2])
            if len(toks2) > 3: params["p4"] = float(toks2[3])

    model.prop_spotwelds[pid] = PropSpotweld(
        id=pid, title=title, skew_id=skew_id, sensor_id=sensor_id,
        knn=knn, cr=cr, scf=scf, params=params
    )




def read_prop_bushing(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE46`` (M152): Bushing property."""
    from ...model.entities import PropBushing
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROP/TYPE46/{pid}: missing data card", block.source)
        return
    mass, k_elas, x_lim1, x_lim2, k_post, damp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    epsi, idens = 0, 0
    params = {}
    if block.fixed:
        f = cards[0].cut("PROP_TYPE46_1")
        mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        k_elas = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        x_lim1 = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        x_lim2 = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        k_post = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROP_TYPE46_2")
            damp = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            epsi = _ival(f2[1]) if len(f2) > 1 else 0
            idens = _ival(f2[2]) if len(f2) > 2 else 0
    else:
        toks = cards[0].tokens()
        mass = float(toks[0]) if len(toks) > 0 else 0.0
        k_elas = float(toks[1]) if len(toks) > 1 else 0.0
        x_lim1 = float(toks[2]) if len(toks) > 2 else 0.0
        x_lim2 = float(toks[3]) if len(toks) > 3 else 0.0
        k_post = float(toks[4]) if len(toks) > 4 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            damp = float(toks2[0]) if len(toks2) > 0 else 0.0
            epsi = int(float(toks2[1])) if len(toks2) > 1 else 0
            idens = int(float(toks2[2])) if len(toks2) > 2 else 0

    model.prop_bushings[pid] = PropBushing(
        id=pid, title=title, mass=mass, k_elas=k_elas, x_lim1=x_lim1,
        x_lim2=x_lim2, k_post=k_post, damp=damp, epsi=epsi, idens=idens, params=params
    )




def read_prop_tshell(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE20/id`` or ``/PROP/TSHELL/id`` (M180): Thick shell property."""
    from ...model.entities import PropType20
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/TSHELL/{prop_id}: missing data cards", block.source)
        return

    isolid = 15
    ismstr, icpre, icstr = 0, 0, 0
    inpts_r, inpts_s, inpts_t, iint = 2, 2, 2, 1
    dn, qa, qb, h = 0.0, 1.1, 0.05, 0.1
    deltat_min = 0.0

    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE20_1")
        isolid = _ival(f1[0], 15) if len(f1) > 0 and f1[0].strip() else 15
        ismstr = _ival(f1[1]) if len(f1) > 1 else 0
        icpre = _ival(f1[2]) if len(f1) > 2 else 0
        icstr = _ival(f1[3]) if len(f1) > 3 else 0
        nbp = _ival(f1[4], 222) if len(f1) > 4 and f1[4].strip() else 222
        if nbp > 200:
            inpts_r = nbp // 100
            rem = nbp % 100
            inpts_s = rem // 10
            inpts_t = rem % 10
        else:
            inpts_s = nbp
        iint = _ival(f1[5], 1) if len(f1) > 5 and f1[5].strip() else 1
        dn = _fval(f1[6]) if len(f1) > 6 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("PROP_TYPE20_2")
            qa = _fval(f2[0], 1.1) if len(f2) > 0 and f2[0].strip() else 1.1
            qb = _fval(f2[1], 0.05) if len(f2) > 1 and f2[1].strip() else 0.05
            h = _fval(f2[2], 0.1) if len(f2) > 2 and f2[2].strip() else 0.1

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("PROP_TYPE20_3")
            deltat_min = _fval(f3[0]) if len(f3) > 0 else 0.0
    else:
        toks1 = valid_cards[0].tokens()
        isolid = int(float(toks1[0])) if len(toks1) > 0 else 15
        ismstr = int(float(toks1[1])) if len(toks1) > 1 else 0
        icpre = int(float(toks1[2])) if len(toks1) > 2 else 0
        icstr = int(float(toks1[3])) if len(toks1) > 3 else 0
        nbp = int(float(toks1[4])) if len(toks1) > 4 else 222
        if nbp > 200:
            inpts_r = nbp // 100
            rem = nbp % 100
            inpts_s = rem // 10
            inpts_t = rem % 10
        else:
            inpts_s = nbp
        iint = int(float(toks1[5])) if len(toks1) > 5 else 1
        dn = float(toks1[6]) if len(toks1) > 6 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            qa = float(toks2[0]) if len(toks2) > 0 else 1.1
            qb = float(toks2[1]) if len(toks2) > 1 else 0.05
            h = float(toks2[2]) if len(toks2) > 2 else 0.1

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            deltat_min = float(toks3[0]) if len(toks3) > 0 else 0.0

    p20 = PropType20(
        id=prop_id, isolid=isolid, ismstr=ismstr, icpre=icpre, icstr=icstr,
        inpts_r=inpts_r, inpts_s=inpts_s, inpts_t=inpts_t, iint=iint,
        dn=dn, qa=qa, qb=qb, h=h, deltat_min=deltat_min, title=title,
    )
    model.prop_tshells[prop_id] = p20
    if prop_id not in model.properties:
        model.properties[prop_id] = Property(
            id=prop_id, type=20, title=title,
            params={"Isolid": isolid, "Ismstr": ismstr, "Icpre": icpre, "Icstr": icstr, "qa": qa, "qb": qb, "h": h}
        )




def read_prop_tsh_orth(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE21/id`` or ``/PROP/TSH_ORTH/id`` (M180): Orthotropic thick shell property."""
    from ...model.entities import PropType21
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/TSH_ORTH/{prop_id}: missing data cards", block.source)
        return

    isolid = 15
    ismstr, icstr = 0, 0
    inpts_r, inpts_s, inpts_t, iint = 2, 2, 2, 1
    dn, qa, qb = 0.0, 1.1, 0.05
    vx, vy, vz = 0.0, 0.0, 0.0
    skew_id, iorth = 0, 0
    phi, deltat_min = 0.0, 0.0

    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE21_1")
        isolid = _ival(f1[0], 15) if len(f1) > 0 and f1[0].strip() else 15
        ismstr = _ival(f1[1]) if len(f1) > 1 else 0
        icstr = _ival(f1[3]) if len(f1) > 3 else 0
        nbp = _ival(f1[4], 222) if len(f1) > 4 and f1[4].strip() else 222
        if nbp > 200:
            inpts_r = nbp // 100
            rem = nbp % 100
            inpts_s = rem // 10
            inpts_t = rem % 10
        else:
            inpts_s = nbp
        iint = _ival(f1[5], 1) if len(f1) > 5 and f1[5].strip() else 1
        dn = _fval(f1[6]) if len(f1) > 6 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("PROP_TYPE21_2")
            qa = _fval(f2[0], 1.1) if len(f2) > 0 and f2[0].strip() else 1.1
            qb = _fval(f2[1], 0.05) if len(f2) > 1 and f2[1].strip() else 0.05

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("PROP_TYPE21_3")
            vx = _fval(f3[0]) if len(f3) > 0 else 0.0
            vy = _fval(f3[1]) if len(f3) > 1 else 0.0
            vz = _fval(f3[2]) if len(f3) > 2 else 0.0
            skew_id = _ival(f3[3]) if len(f3) > 3 else 0
            iorth = _ival(f3[4]) if len(f3) > 4 else 0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("PROP_TYPE21_4")
            phi = _fval(f4[0]) if len(f4) > 0 else 0.0

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("PROP_TYPE21_5")
            deltat_min = _fval(f5[0]) if len(f5) > 0 else 0.0
    else:
        toks1 = valid_cards[0].tokens()
        isolid = int(float(toks1[0])) if len(toks1) > 0 else 15
        ismstr = int(float(toks1[1])) if len(toks1) > 1 else 0
        icstr = int(float(toks1[2])) if len(toks1) > 2 else 0
        nbp = int(float(toks1[3])) if len(toks1) > 3 else 222
        if nbp > 200:
            inpts_r = nbp // 100
            rem = nbp % 100
            inpts_s = rem // 10
            inpts_t = rem % 10
        else:
            inpts_s = nbp
        iint = int(float(toks1[4])) if len(toks1) > 4 else 1
        dn = float(toks1[5]) if len(toks1) > 5 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            qa = float(toks2[0]) if len(toks2) > 0 else 1.1
            qb = float(toks2[1]) if len(toks2) > 1 else 0.05

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            vx = float(toks3[0]) if len(toks3) > 0 else 0.0
            vy = float(toks3[1]) if len(toks3) > 1 else 0.0
            vz = float(toks3[2]) if len(toks3) > 2 else 0.0
            skew_id = int(float(toks3[3])) if len(toks3) > 3 else 0
            iorth = int(float(toks3[4])) if len(toks3) > 4 else 0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            phi = float(toks4[0]) if len(toks4) > 0 else 0.0

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            deltat_min = float(toks5[0]) if len(toks5) > 0 else 0.0

    p21 = PropType21(
        id=prop_id, isolid=isolid, ismstr=ismstr, icstr=icstr,
        inpts_r=inpts_r, inpts_s=inpts_s, inpts_t=inpts_t, iint=iint,
        dn=dn, qa=qa, qb=qb, vx=vx, vy=vy, vz=vz, skew_id=skew_id,
        iorth=iorth, phi=phi, deltat_min=deltat_min, title=title,
    )
    model.prop_tsh_orths[prop_id] = p21
    if prop_id not in model.properties:
        model.properties[prop_id] = Property(
            id=prop_id, type=21, title=title,
            params={"Isolid": isolid, "Ismstr": ismstr, "Icstr": icstr, "qa": qa, "qb": qb, "Phi": phi, "skew_id": skew_id}
        )




def read_prop_tsh_comp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE22/id`` or ``/PROP/TSH_COMP/id`` (M180): Composite layered thick shell property."""
    from ...model.entities import PropType22, PropType22Layer
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/TSH_COMP/{prop_id}: missing data cards", block.source)
        return

    isolid = 15
    ismstr, icstr = 0, 0
    inpts_r, inpts_s, inpts_t, iint = 2, 2, 2, 1
    dn, qa, qb = 0.0, 1.1, 0.05
    vx, vy, vz = 0.0, 0.0, 0.0
    skew_id, iorth, ipos = 0, 0, 0
    ashear = 0.0
    layers: List[PropType22Layer] = []
    deltat_min = 0.0
    n_layers = 1

    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE22_1")
        isolid = _ival(f1[0], 15) if len(f1) > 0 and f1[0].strip() else 15
        ismstr = _ival(f1[1]) if len(f1) > 1 else 0
        icstr = _ival(f1[3]) if len(f1) > 3 else 0
        nbp = _ival(f1[4], 222) if len(f1) > 4 and f1[4].strip() else 222
        if nbp > 200:
            inpts_r = nbp // 100
            rem = nbp % 100
            inpts_s = rem // 10
            inpts_t = rem % 10
        else:
            inpts_s = nbp
        iint = _ival(f1[5], 1) if len(f1) > 5 and f1[5].strip() else 1
        dn = _fval(f1[6]) if len(f1) > 6 else 0.0

        if iint > 9:
            n_layers = iint
        else:
            n_layers = inpts_s

        card_idx = 1
        if card_idx < len(valid_cards):
            f2 = valid_cards[card_idx].cut("PROP_TYPE22_2")
            qa = _fval(f2[0], 1.1) if len(f2) > 0 and f2[0].strip() else 1.1
            qb = _fval(f2[1], 0.05) if len(f2) > 1 and f2[1].strip() else 0.05
            card_idx += 1

        if card_idx < len(valid_cards):
            f3 = valid_cards[card_idx].cut("PROP_TYPE22_3")
            vx = _fval(f3[0]) if len(f3) > 0 else 0.0
            vy = _fval(f3[1]) if len(f3) > 1 else 0.0
            vz = _fval(f3[2]) if len(f3) > 2 else 0.0
            skew_id = _ival(f3[3]) if len(f3) > 3 else 0
            iorth = _ival(f3[4]) if len(f3) > 4 else 0
            ipos = _ival(f3[5]) if len(f3) > 5 else 0
            card_idx += 1

        if card_idx < len(valid_cards):
            f4 = valid_cards[card_idx].cut("PROP_TYPE22_4")
            ashear = _fval(f4[0]) if len(f4) > 0 else 0.0
            card_idx += 1

        while card_idx < len(valid_cards) and len(layers) < n_layers:
            fl = valid_cards[card_idx].cut("PROP_TYPE22_LAYER")
            phi_l = _fval(fl[0]) if len(fl) > 0 else 0.0
            thick_l = _fval(fl[1]) if len(fl) > 1 else 0.0
            zi_l = _fval(fl[2]) if len(fl) > 2 else 0.0
            mat_l = _ival(fl[3]) if len(fl) > 3 else 0
            layers.append(PropType22Layer(phi=phi_l, thick=thick_l, zi=zi_l, mat_id=mat_l))
            card_idx += 1

        if card_idx < len(valid_cards):
            f5 = valid_cards[card_idx].cut("PROP_TYPE22_5")
            deltat_min = _fval(f5[0]) if len(f5) > 0 else 0.0
    else:
        toks1 = valid_cards[0].tokens()
        isolid = int(float(toks1[0])) if len(toks1) > 0 else 15
        ismstr = int(float(toks1[1])) if len(toks1) > 1 else 0
        icstr = int(float(toks1[2])) if len(toks1) > 2 else 0
        nbp = int(float(toks1[3])) if len(toks1) > 3 else 222
        if nbp > 200:
            inpts_r = nbp // 100
            rem = nbp % 100
            inpts_s = rem // 10
            inpts_t = rem % 10
        else:
            inpts_s = nbp
        iint = int(float(toks1[4])) if len(toks1) > 4 else 1
        dn = float(toks1[5]) if len(toks1) > 5 else 0.0

        if iint > 9:
            n_layers = iint
        else:
            n_layers = inpts_s

        card_idx = 1
        if card_idx < len(valid_cards):
            toks2 = valid_cards[card_idx].tokens()
            qa = float(toks2[0]) if len(toks2) > 0 else 1.1
            qb = float(toks2[1]) if len(toks2) > 1 else 0.05
            card_idx += 1

        if card_idx < len(valid_cards):
            toks3 = valid_cards[card_idx].tokens()
            vx = float(toks3[0]) if len(toks3) > 0 else 0.0
            vy = float(toks3[1]) if len(toks3) > 1 else 0.0
            vz = float(toks3[2]) if len(toks3) > 2 else 0.0
            skew_id = int(float(toks3[3])) if len(toks3) > 3 else 0
            iorth = int(float(toks3[4])) if len(toks3) > 4 else 0
            ipos = int(float(toks3[5])) if len(toks3) > 5 else 0
            card_idx += 1

        if card_idx < len(valid_cards):
            toks4 = valid_cards[card_idx].tokens()
            ashear = float(toks4[0]) if len(toks4) > 0 else 0.0
            card_idx += 1

        while card_idx < len(valid_cards) and len(layers) < n_layers:
            toksl = valid_cards[card_idx].tokens()
            phi_l = float(toksl[0]) if len(toksl) > 0 else 0.0
            thick_l = float(toksl[1]) if len(toksl) > 1 else 0.0
            zi_l = float(toksl[2]) if len(toksl) > 2 else 0.0
            mat_l = int(float(toksl[3])) if len(toksl) > 3 else 0
            layers.append(PropType22Layer(phi=phi_l, thick=thick_l, zi=zi_l, mat_id=mat_l))
            card_idx += 1

        if card_idx < len(valid_cards):
            toks5 = valid_cards[card_idx].tokens()
            deltat_min = float(toks5[0]) if len(toks5) > 0 else 0.0

    p22 = PropType22(
        id=prop_id, isolid=isolid, ismstr=ismstr, icstr=icstr,
        inpts_r=inpts_r, inpts_s=inpts_s, inpts_t=inpts_t, iint=iint,
        dn=dn, qa=qa, qb=qb, vx=vx, vy=vy, vz=vz, skew_id=skew_id,
        iorth=iorth, ipos=ipos, ashear=ashear, layers=layers,
        deltat_min=deltat_min, title=title,
    )
    model.prop_tsh_comps[prop_id] = p22
    if prop_id not in model.properties:
        model.properties[prop_id] = Property(
            id=prop_id, type=22, title=title,
            params={"Isolid": isolid, "Ismstr": ismstr, "Icstr": icstr, "qa": qa, "qb": qb, "N": len(layers), "skew_id": skew_id}
        )




def read_prop_spr_tab(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE26/id`` or ``/PROP/SPR_TAB/id`` (M182): Tabulated nonlinear spring property."""
    from ...model.entities import PropType26, PropType26Curve, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/SPR_TAB/{prop_id}: missing data cards", block.source)
        return

    mass = 0.0
    sens_id, isflag, ileng = 0, 0, 0
    dmin = 0.0
    nfunc, nfund = 1, 1
    lscale, kmax, dmax, alpha = 1.0, 1.0, 0.0, 1.0
    loading_curves: List[PropType26Curve] = []
    unloading_curves: List[PropType26Curve] = []

    card_idx = 0
    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE26_1")
        mass = _fval(f1[0]) if len(f1) > 0 else 0.0
        sens_id = _ival(f1[2]) if len(f1) > 2 else 0
        isflag = _ival(f1[3]) if len(f1) > 3 else 0
        ileng = _ival(f1[4]) if len(f1) > 4 else 0
        dmin = _fval(f1[5]) if len(f1) > 5 else 0.0
        card_idx = 1

        if card_idx < len(valid_cards):
            f2 = valid_cards[card_idx].cut("PROP_TYPE26_2")
            nfunc = _ival(f2[0], 1) if len(f2) > 0 and f2[0].strip() else 1
            nfund = _ival(f2[1], 1) if len(f2) > 1 and f2[1].strip() else 1
            lscale = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
            kmax = _fval(f2[3], 1.0) if len(f2) > 3 and f2[3].strip() else 1.0
            dmax = _fval(f2[4]) if len(f2) > 4 else 0.0
            alpha = _fval(f2[5], 1.0) if len(f2) > 5 and f2[5].strip() else 1.0
            card_idx += 1

        for _ in range(nfunc):
            if card_idx < len(valid_cards):
                fl = valid_cards[card_idx].cut("PROP_TYPE26_LOAD")
                fid = _ival(fl[0]) if len(fl) > 0 else 0
                fsc = _fval(fl[1], 1.0) if len(fl) > 1 and fl[1].strip() else 1.0
                sr = _fval(fl[2]) if len(fl) > 2 else 0.0
                loading_curves.append(PropType26Curve(fct_id=fid, fscale=fsc, strain_rate=sr))
                card_idx += 1

        for _ in range(nfund):
            if card_idx < len(valid_cards):
                ful = valid_cards[card_idx].cut("PROP_TYPE26_UNLOAD")
                fid = _ival(ful[0]) if len(ful) > 0 else 0
                fsc = _fval(ful[1], 1.0) if len(ful) > 1 and fl[1].strip() else 1.0
                sr = _fval(ful[2]) if len(ful) > 2 else 0.0
                unloading_curves.append(PropType26Curve(fct_id=fid, fscale=fsc, strain_rate=sr))
                card_idx += 1
    else:
        toks1 = valid_cards[0].tokens()
        mass = float(toks1[0]) if len(toks1) > 0 else 0.0
        sens_id = int(float(toks1[1])) if len(toks1) > 1 else 0
        isflag = int(float(toks1[2])) if len(toks1) > 2 else 0
        ileng = int(float(toks1[3])) if len(toks1) > 3 else 0
        dmin = 0.0
        card_idx = 1
        if len(toks1) >= 6:
            nfunc = int(float(toks1[4]))
            nfund = int(float(toks1[5]))
        elif len(toks1) == 5:
            dmin = float(toks1[4])

        if card_idx < len(valid_cards):
            toks2 = valid_cards[card_idx].tokens()
            if len(toks2) == 4:
                lscale = float(toks2[0]) if len(toks2) > 0 else 1.0
                kmax = float(toks2[1]) if len(toks2) > 1 else 1.0
                dmax = float(toks2[2]) if len(toks2) > 2 else 0.0
                alpha = float(toks2[3]) if len(toks2) > 3 else 1.0
                card_idx += 1
            elif len(toks2) == 5:
                nfunc = int(float(toks2[0])) if len(toks2) > 0 else 1
                nfund = int(float(toks2[1])) if len(toks2) > 1 else 1
                lscale = float(toks2[2]) if len(toks2) > 2 else 1.0
                kmax = float(toks2[3]) if len(toks2) > 3 else 1.0
                alpha = float(toks2[4]) if len(toks2) > 4 else 1.0
                card_idx += 1
            elif len(toks2) >= 6:
                nfunc = int(float(toks2[0])) if len(toks2) > 0 else 1
                nfund = int(float(toks2[1])) if len(toks2) > 1 else 1
                lscale = float(toks2[2]) if len(toks2) > 2 else 1.0
                kmax = float(toks2[3]) if len(toks2) > 3 else 1.0
                dmax = float(toks2[4]) if len(toks2) > 4 else 0.0
                alpha = float(toks2[5]) if len(toks2) > 5 else 1.0
                card_idx += 1

        for _ in range(nfunc):
            if card_idx < len(valid_cards):
                toksl = valid_cards[card_idx].tokens()
                fid = int(float(toksl[0])) if len(toksl) > 0 else 0
                fsc = float(toksl[1]) if len(toksl) > 1 else 1.0
                sr = float(toksl[2]) if len(toksl) > 2 else 0.0
                loading_curves.append(PropType26Curve(fct_id=fid, fscale=fsc, strain_rate=sr))
                card_idx += 1

        for _ in range(nfund):
            if card_idx < len(valid_cards):
                toksul = valid_cards[card_idx].tokens()
                fid = int(float(toksul[0])) if len(toksul) > 0 else 0
                fsc = float(toksul[1]) if len(toksul) > 1 else 1.0
                sr = float(toksul[2]) if len(toksul) > 2 else 0.0
                unloading_curves.append(PropType26Curve(fct_id=fid, fscale=fsc, strain_rate=sr))
                card_idx += 1

    p26 = PropType26(
        id=prop_id, mass=mass, sens_id=sens_id, isflag=isflag, ileng=ileng,
        dmin=dmin, nfunc=nfunc, nfund=nfund, lscale=lscale, kmax=kmax,
        dmax=dmax, alpha=alpha, loading_curves=loading_curves,
        unloading_curves=unloading_curves, title=title,
    )
    model.prop_spr_tabs[prop_id] = p26
    from ..prop_reader import _universal_geo_params
    p26_params = _universal_geo_params()
    p26_params.update({
        "mass": mass, "Mass": mass,
        "sens_id": sens_id, "isensor": sens_id,
        "isflag": isflag, "ileng": ileng,
        "dmin": dmin, "Dmin": dmin,
        "nfunc": nfunc, "Nfunc": nfunc,
        "nfund": nfund, "Nfund": nfund, "nraten": nfund, "Nraten": nfund,
        "scale": lscale, "lscale": lscale,
        "stiff0": kmax, "k": kmax, "kmax": kmax, "Kmax": kmax,
        "dmax": dmax, "Dmax": dmax,
        "alpha1": alpha, "alpha": alpha, "Alpha": alpha,
        "load_curves": [{"fun_load": c.fct_id, "scale_load": c.fscale, "strainrate_load": c.strain_rate} for c in loading_curves],
        "unload_curves": [{"fun_unload": c.fct_id, "scale_unload": c.fscale, "strainrate_unload": c.strain_rate} for c in unloading_curves],
    })
    model.properties[prop_id] = Property(
        id=prop_id, type=26, title=title,
        params=p26_params,
    )




def read_prop_spr_bdamp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE27/id`` or ``/PROP/SPR_BDAMP/id`` (M182): Spring with bilinear/barycentric damping."""
    from ...model.entities import PropType27, Property
    from ..prop_reader import _universal_geo_params
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/SPR_BDAMP/{prop_id}: missing data cards", block.source)
        return

    mass = 0.0
    sens_id, isflag, ileng, itens, ifail = 0, 0, 0, 0, 0
    stiff, damp, nexp, delta_min, delta_max = 0.0, 0.0, 1.0, 0.0, 0.0
    gap = 0.0
    fsmooth, fcut = 0, 0.0
    fct_id1, fct_id2 = 0, 0
    ascale1, fscale1, ascale2, fscale2 = 1.0, 1.0, 1.0, 1.0

    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE27_1")
        mass = _fval(f1[0]) if len(f1) > 0 else 0.0
        sens_id = _ival(f1[2]) if len(f1) > 2 else 0
        isflag = _ival(f1[3]) if len(f1) > 3 else 0
        ileng = _ival(f1[4]) if len(f1) > 4 else 0
        itens = _ival(f1[5]) if len(f1) > 5 else 0
        ifail = _ival(f1[6]) if len(f1) > 6 else 0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("PROP_TYPE27_2")
            stiff = _fval(f2[0]) if len(f2) > 0 else 0.0
            damp = _fval(f2[1]) if len(f2) > 1 else 0.0
            nexp = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
            delta_min = _fval(f2[3]) if len(f2) > 3 else 0.0
            delta_max = _fval(f2[4]) if len(f2) > 4 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("PROP_TYPE27_3")
            gap = _fval(f3[0]) if len(f3) > 0 else 0.0
            fsmooth = _ival(f3[2]) if len(f3) > 2 else 0
            fcut = _fval(f3[3]) if len(f3) > 3 else 0.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("PROP_TYPE27_4")
            fct_id1 = _ival(f4[0]) if len(f4) > 0 else 0
            fct_id2 = _ival(f4[1]) if len(f4) > 0 else 0
            ascale1 = _fval(f4[2], 1.0) if len(f4) > 2 and f4[2].strip() else 1.0
            fscale1 = _fval(f4[3], 1.0) if len(f4) > 3 and f4[3].strip() else 1.0
            ascale2 = _fval(f4[4], 1.0) if len(f4) > 4 and f4[4].strip() else 1.0
            fscale2 = _fval(f4[5], 1.0) if len(f4) > 5 and f4[5].strip() else 1.0
    else:
        toks1 = valid_cards[0].tokens()
        mass = float(toks1[0]) if len(toks1) > 0 else 0.0
        sens_id = int(float(toks1[1])) if len(toks1) > 1 else 0
        isflag = int(float(toks1[2])) if len(toks1) > 2 else 0
        ileng = int(float(toks1[3])) if len(toks1) > 3 else 0
        itens = int(float(toks1[4])) if len(toks1) > 4 else 0
        ifail = int(float(toks1[5])) if len(toks1) > 5 else 0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            stiff = float(toks2[0]) if len(toks2) > 0 else 0.0
            damp = float(toks2[1]) if len(toks2) > 1 else 0.0
            nexp = float(toks2[2]) if len(toks2) > 2 else 1.0
            delta_min = float(toks2[3]) if len(toks2) > 3 else 0.0
            delta_max = float(toks2[4]) if len(toks2) > 4 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            gap = float(toks3[0]) if len(toks3) > 0 else 0.0
            fsmooth = int(float(toks3[1])) if len(toks3) > 1 else 0
            fcut = float(toks3[2]) if len(toks3) > 2 else 0.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            fct_id1 = int(float(toks4[0])) if len(toks4) > 0 else 0
            fct_id2 = int(float(toks4[1])) if len(toks4) > 0 else 0
            ascale1 = float(toks4[2]) if len(toks4) > 2 else 1.0
            fscale1 = float(toks4[3]) if len(toks4) > 3 else 1.0
            ascale2 = float(toks4[4]) if len(toks4) > 4 else 1.0
            fscale2 = float(toks4[5]) if len(toks4) > 5 else 1.0

    p27 = PropType27(
        id=prop_id, mass=mass, sens_id=sens_id, isflag=isflag, ileng=ileng,
        itens=itens, ifail=ifail, stiff=stiff, damp=damp, nexp=nexp,
        delta_min=delta_min, delta_max=delta_max, gap=gap, fsmooth=fsmooth,
        fcut=fcut, fct_id1=fct_id1, fct_id2=fct_id2, ascale1=ascale1,
        fscale1=fscale1, ascale2=ascale2, fscale2=fscale2, title=title,
    )
    model.prop_spr_bdamps[prop_id] = p27
    p27_params = _universal_geo_params()
    p27_params.update({
        "mass": mass, "Mass": mass,
        "sens_id": sens_id, "isensor": sens_id,
        "isflag": isflag, "ileng": ileng,
        "itens": itens, "ifail": ifail,
        "stiff": stiff, "k": stiff, "K": stiff,
        "damp": damp, "c": damp, "C": damp,
        "nexp": nexp, "n": nexp,
        "delta_min": delta_min, "min_rup": delta_min, "Delta_min": delta_min,
        "delta_max": delta_max, "max_rup": delta_max, "Delta_max": delta_max,
        "gap": gap, "fsmooth": fsmooth, "fcut": fcut,
        "fun1": fct_id1, "fct1": fct_id1, "fct_id1": fct_id1,
        "fun2": fct_id2, "fct2": fct_id2, "fct_id2": fct_id2,
        "ascale1": ascale1, "fscale1": fscale1,
        "ascale2": ascale2, "fscale2": fscale2,
    })
    model.properties[prop_id] = Property(
        id=prop_id, type=27, title=title,
        params=p27_params,
    )




def read_prop_type11(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE11/id`` or ``/PROP/SH_SANDW/id`` (M184): Sandwich shell property."""
    from ...model.entities import PropType11, PropSandwLayer, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/TYPE11/{prop_id}: missing data cards", block.source)
        return

    ishell, ismstr, ish3n, idrill = 0, 0, 0, 0
    p_thick_fail = 0.0
    hm, hf, hr, dm, dn = 0.0, 0.0, 0.0, 0.0, 0.0
    nip, istrain = 0, 0
    thick, ashear = 0.0, 0.0
    ithick, iplas = 0, 0
    vx, vy, vz = 0.0, 0.0, 0.0
    skew_csid, iorth, ipos, ip = 0, 0, 0, 0
    layers: list[PropSandwLayer] = []

    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE11_1")
        ishell = _ival(f1[0]) if len(f1) > 0 else 0
        ismstr = _ival(f1[1]) if len(f1) > 1 else 0
        ish3n = _ival(f1[2]) if len(f1) > 2 else 0
        idrill = _ival(f1[3]) if len(f1) > 3 else 0
        p_thick_fail = _fval(f1[5]) if len(f1) > 5 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("PROP_TYPE11_2")
            hm = _fval(f2[0]) if len(f2) > 0 else 0.0
            hf = _fval(f2[1]) if len(f2) > 1 else 0.0
            hr = _fval(f2[2]) if len(f2) > 2 else 0.0
            dm = _fval(f2[3]) if len(f2) > 3 else 0.0
            dn = _fval(f2[4]) if len(f2) > 4 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("PROP_TYPE11_3")
            nip = _ival(f3[0]) if len(f3) > 0 else 0
            istrain = _ival(f3[1]) if len(f3) > 1 else 0
            thick = _fval(f3[2]) if len(f3) > 2 else 0.0
            ashear = _fval(f3[3]) if len(f3) > 3 else 0.0
            ithick = _ival(f3[5]) if len(f3) > 5 else 0
            iplas = _ival(f3[6]) if len(f3) > 6 else 0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("PROP_TYPE11_4")
            vx = _fval(f4[0]) if len(f4) > 0 else 0.0
            vy = _fval(f4[1]) if len(f4) > 1 else 0.0
            vz = _fval(f4[2]) if len(f4) > 2 else 0.0
            skew_csid = _ival(f4[3]) if len(f4) > 3 else 0
            iorth = _ival(f4[4]) if len(f4) > 4 else 0
            ipos = _ival(f4[5]) if len(f4) > 5 else 0
            ip = _ival(f4[6]) if len(f4) > 6 else 0

        for card in valid_cards[4:]:
            fl = card.cut("PROP_TYPE11_5")
            if fl and any(x.strip() for x in fl):
                phi = _fval(fl[0]) if len(fl) > 0 else 0.0
                thick_i = _fval(fl[1]) if len(fl) > 1 else 0.0
                z_i = _fval(fl[2]) if len(fl) > 2 else 0.0
                mat_i = _ival(fl[3]) if len(fl) > 3 else 0
                fw_i = _fval(fl[5]) if len(fl) > 5 else 0.0
                layers.append(PropSandwLayer(phi=phi, thick=thick_i, z=z_i, mat_id=mat_i, f_weight=fw_i))
    else:
        toks1 = valid_cards[0].tokens()
        ishell = int(float(toks1[0])) if len(toks1) > 0 else 0
        ismstr = int(float(toks1[1])) if len(toks1) > 1 else 0
        ish3n = int(float(toks1[2])) if len(toks1) > 2 else 0
        idrill = int(float(toks1[3])) if len(toks1) > 3 else 0
        p_thick_fail = float(toks1[4]) if len(toks1) > 4 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            hm = float(toks2[0]) if len(toks2) > 0 else 0.0
            hf = float(toks2[1]) if len(toks2) > 1 else 0.0
            hr = float(toks2[2]) if len(toks2) > 2 else 0.0
            dm = float(toks2[3]) if len(toks2) > 3 else 0.0
            dn = float(toks2[4]) if len(toks2) > 4 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            nip = int(float(toks3[0])) if len(toks3) > 0 else 0
            istrain = int(float(toks3[1])) if len(toks3) > 1 else 0
            thick = float(toks3[2]) if len(toks3) > 2 else 0.0
            ashear = float(toks3[3]) if len(toks3) > 3 else 0.0
            ithick = int(float(toks3[4])) if len(toks3) > 4 else 0
            iplas = int(float(toks3[5])) if len(toks3) > 5 else 0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            vx = float(toks4[0]) if len(toks4) > 0 else 0.0
            vy = float(toks4[1]) if len(toks4) > 1 else 0.0
            vz = float(toks4[2]) if len(toks4) > 2 else 0.0
            skew_csid = int(float(toks4[3])) if len(toks4) > 3 else 0
            iorth = int(float(toks4[4])) if len(toks4) > 4 else 0
            ipos = int(float(toks4[5])) if len(toks4) > 5 else 0
            ip = int(float(toks4[6])) if len(toks4) > 6 else 0

        for card in valid_cards[4:]:
            tl = card.tokens()
            if tl:
                phi = float(tl[0]) if len(tl) > 0 else 0.0
                thick_i = float(tl[1]) if len(tl) > 1 else 0.0
                z_i = float(tl[2]) if len(tl) > 2 else 0.0
                mat_i = int(float(tl[3])) if len(tl) > 3 else 0
                fw_i = float(tl[4]) if len(tl) > 4 else 0.0
                layers.append(PropSandwLayer(phi=phi, thick=thick_i, z=z_i, mat_id=mat_i, f_weight=fw_i))

    p11 = PropType11(
        id=prop_id, ishell=ishell, ismstr=ismstr, ish3n=ish3n, idrill=idrill,
        p_thick_fail=p_thick_fail, hm=hm, hf=hf, hr=hr, dm=dm, dn=dn,
        nip=nip, istrain=istrain, thick=thick, ashear=ashear, ithick=ithick,
        iplas=iplas, vx=vx, vy=vy, vz=vz, skew_csid=skew_csid, iorth=iorth,
        ipos=ipos, ip=ip, layers=layers, title=title,
    )
    model.prop_type11s[prop_id] = p11
    model.properties[prop_id] = Property(
        id=prop_id, type=11, title=title,
        params={
            "thick": thick, "ashear": ashear, "nip": nip or len(layers),
            "ishell": ishell, "ismstr": ismstr, "ish3n": ish3n, "idrill": idrill,
            "p_thick_fail": p_thick_fail, "hm": hm, "hf": hf, "hr": hr, "dm": dm, "dn": dn,
            "istrain": istrain, "ithick": ithick, "iplas": iplas, "vx": vx, "vy": vy, "vz": vz,
            "skew_csid": skew_csid, "iorth": iorth, "ipos": ipos, "ip": ip,
            "layers": [{"phi": l.phi, "thick": l.thick, "z": l.z, "mat_id": l.mat_id, "f_weight": l.f_weight} for l in layers],
        }
    )




def read_prop_type16(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE16/id`` or ``/PROP/SH_FABR/id`` (M184): Fabric shell property."""
    from ...model.entities import PropType16, PropFabricLayer, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/TYPE16/{prop_id}: missing data cards", block.source)
        return

    ishell, ismstr, ish3n = 0, 0, 0
    p_thick_fail = 0.0
    hm, hf, hr, dm, dn = 0.0, 0.0, 0.0, 0.0, 0.0
    nip, istrain = 0, 0
    thick, ashear = 0.0, 0.0
    ithick = 0
    vx, vy, vz = 0.0, 0.0, 0.0
    skew_id, ipos, ip = 0, 0, 0
    layers: list[PropFabricLayer] = []

    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE16_1")
        ishell = _ival(f1[0]) if len(f1) > 0 else 0
        ismstr = _ival(f1[1]) if len(f1) > 1 else 0
        ish3n = _ival(f1[2]) if len(f1) > 2 else 0
        p_thick_fail = _fval(f1[4]) if len(f1) > 4 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("PROP_TYPE16_2")
            hm = _fval(f2[0]) if len(f2) > 0 else 0.0
            hf = _fval(f2[1]) if len(f2) > 1 else 0.0
            hr = _fval(f2[2]) if len(f2) > 2 else 0.0
            dm = _fval(f2[3]) if len(f2) > 3 else 0.0
            dn = _fval(f2[4]) if len(f2) > 4 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("PROP_TYPE16_3")
            nip = _ival(f3[0]) if len(f3) > 0 else 0
            istrain = _ival(f3[1]) if len(f3) > 1 else 0
            thick = _fval(f3[2]) if len(f3) > 2 else 0.0
            ashear = _fval(f3[3]) if len(f3) > 3 else 0.0
            ithick = _ival(f3[5]) if len(f3) > 5 else 0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("PROP_TYPE16_4")
            vx = _fval(f4[0]) if len(f4) > 0 else 0.0
            vy = _fval(f4[1]) if len(f4) > 1 else 0.0
            vz = _fval(f4[2]) if len(f4) > 2 else 0.0
            skew_id = _ival(f4[3]) if len(f4) > 3 else 0
            ipos = _ival(f4[4]) if len(f4) > 4 else 0
            ip = _ival(f4[6]) if len(f4) > 6 else 0

        for card in valid_cards[4:]:
            fl = card.cut("PROP_TYPE16_5")
            if fl and any(x.strip() for x in fl):
                phi = _fval(fl[0]) if len(fl) > 0 else 0.0
                alpha = _fval(fl[1]) if len(fl) > 1 else 0.0
                thick_i = _fval(fl[2]) if len(fl) > 2 else 0.0
                z_i = _fval(fl[3]) if len(fl) > 3 else 0.0
                mat_i = _ival(fl[4]) if len(fl) > 4 else 0
                layers.append(PropFabricLayer(phi=phi, alpha=alpha, thick=thick_i, z=z_i, mat_id=mat_i))
    else:
        toks1 = valid_cards[0].tokens()
        ishell = int(float(toks1[0])) if len(toks1) > 0 else 0
        ismstr = int(float(toks1[1])) if len(toks1) > 1 else 0
        ish3n = int(float(toks1[2])) if len(toks1) > 2 else 0
        p_thick_fail = float(toks1[3]) if len(toks1) > 3 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            hm = float(toks2[0]) if len(toks2) > 0 else 0.0
            hf = float(toks2[1]) if len(toks2) > 1 else 0.0
            hr = float(toks2[2]) if len(toks2) > 2 else 0.0
            dm = float(toks2[3]) if len(toks2) > 3 else 0.0
            dn = float(toks2[4]) if len(toks2) > 4 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            nip = int(float(toks3[0])) if len(toks3) > 0 else 0
            istrain = int(float(toks3[1])) if len(toks3) > 1 else 0
            thick = float(toks3[2]) if len(toks3) > 2 else 0.0
            ashear = float(toks3[3]) if len(toks3) > 3 else 0.0
            ithick = int(float(toks3[4])) if len(toks3) > 4 else 0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            vx = float(toks4[0]) if len(toks4) > 0 else 0.0
            vy = float(toks4[1]) if len(toks4) > 1 else 0.0
            vz = float(toks4[2]) if len(toks4) > 2 else 0.0
            skew_id = int(float(toks4[3])) if len(toks4) > 3 else 0
            ipos = int(float(toks4[4])) if len(toks4) > 4 else 0
            ip = int(float(toks4[5])) if len(toks4) > 5 else 0

        for card in valid_cards[4:]:
            tl = card.tokens()
            if tl:
                phi = float(tl[0]) if len(tl) > 0 else 0.0
                alpha = float(tl[1]) if len(tl) > 1 else 0.0
                thick_i = float(tl[2]) if len(tl) > 2 else 0.0
                z_i = float(tl[3]) if len(tl) > 3 else 0.0
                mat_i = int(float(tl[4])) if len(tl) > 4 else 0
                layers.append(PropFabricLayer(phi=phi, alpha=alpha, thick=thick_i, z=z_i, mat_id=mat_i))

    p16 = PropType16(
        id=prop_id, ishell=ishell, ismstr=ismstr, ish3n=ish3n, p_thick_fail=p_thick_fail,
        hm=hm, hf=hf, hr=hr, dm=dm, dn=dn, nip=nip, istrain=istrain, thick=thick,
        ashear=ashear, ithick=ithick, vx=vx, vy=vy, vz=vz, skew_id=skew_id,
        ipos=ipos, ip=ip, layers=layers, title=title,
    )
    model.prop_type16s[prop_id] = p16
    model.properties[prop_id] = Property(
        id=prop_id, type=16, title=title,
        params={
            "thick": thick, "ashear": ashear, "nip": nip or len(layers),
            "ishell": ishell, "ismstr": ismstr, "ish3n": ish3n,
            "p_thick_fail": p_thick_fail, "hm": hm, "hf": hf, "hr": hr, "dm": dm, "dn": dn,
            "istrain": istrain, "ithick": ithick,
            "vx": vx, "vy": vy, "vz": vz, "skew_id": skew_id, "ipos": ipos, "ip": ip,
            "layers": [{"phi": l.phi, "alpha": l.alpha, "thick": l.thick, "z": l.z, "mat_id": l.mat_id} for l in layers],
        }
    )




def read_prop_type17(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE17/id`` or ``/PROP/STACK/id`` (M184): Composite ply stack property."""
    from ...model.entities import PropType17, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/TYPE17/{prop_id}: missing data cards", block.source)
        return

    ishell, ismstr, ish3n, idrill, plyxfem = 0, 0, 0, 0, 0
    z0, vinterply = 0.0, 0.0
    hm, hf, hr, dm, dn = 0.0, 0.0, 0.0, 0.0, 0.0
    thick, ashear = 0.0, 0.0
    ithick, iplas = 0, 0
    vx, vy, vz = 0.0, 0.0, 0.0
    skew_id, iorth, ipos, refplane = 0, 0, 0, 0

    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE17_1")
        ishell = _ival(f1[0]) if len(f1) > 0 else 0
        ismstr = _ival(f1[1]) if len(f1) > 1 else 0
        ish3n = _ival(f1[2]) if len(f1) > 2 else 0
        idrill = _ival(f1[3]) if len(f1) > 3 else 0
        plyxfem = _ival(f1[4]) if len(f1) > 4 else 0
        z0 = 0.0
        for slot in (6, 5):
            if len(f1) > slot and f1[slot].strip():
                z0 = _fval(f1[slot])
                break
        vinterply = _fval(f1[7]) if len(f1) > 7 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("PROP_TYPE17_2")
            hm = _fval(f2[0]) if len(f2) > 0 else 0.0
            hf = _fval(f2[1]) if len(f2) > 1 else 0.0
            hr = _fval(f2[2]) if len(f2) > 2 else 0.0
            dm = _fval(f2[3]) if len(f2) > 3 else 0.0
            dn = _fval(f2[4]) if len(f2) > 4 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("PROP_TYPE17_3")
            thick = _fval(f3[1]) if len(f3) > 1 else 0.0
            ashear = _fval(f3[2]) if len(f3) > 2 else 0.0
            ithick = _ival(f3[4]) if len(f3) > 4 else 0
            iplas = _ival(f3[5]) if len(f3) > 5 else 0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("PROP_TYPE17_4")
            vx = _fval(f4[0]) if len(f4) > 0 else 0.0
            vy = _fval(f4[1]) if len(f4) > 1 else 0.0
            vz = _fval(f4[2]) if len(f4) > 2 else 0.0
            skew_id = _ival(f4[3]) if len(f4) > 3 else 0
            iorth = _ival(f4[4]) if len(f4) > 4 else 0
            ipos = _ival(f4[5]) if len(f4) > 5 else 0
            refplane = _ival(f4[6]) if len(f4) > 6 else 0
    else:
        toks1 = valid_cards[0].tokens()
        ishell = int(float(toks1[0])) if len(toks1) > 0 else 0
        ismstr = int(float(toks1[1])) if len(toks1) > 1 else 0
        ish3n = int(float(toks1[2])) if len(toks1) > 2 else 0
        idrill = int(float(toks1[3])) if len(toks1) > 3 else 0
        if len(toks1) == 5:
            z0 = float(toks1[4])
        elif len(toks1) == 6:
            plyxfem = int(float(toks1[4]))
            z0 = float(toks1[5])
        elif len(toks1) >= 7:
            plyxfem = int(float(toks1[4]))
            z0 = float(toks1[5])
            vinterply = float(toks1[6])

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            hm = float(toks2[0]) if len(toks2) > 0 else 0.0
            hf = float(toks2[1]) if len(toks2) > 1 else 0.0
            hr = float(toks2[2]) if len(toks2) > 2 else 0.0
            dm = float(toks2[3]) if len(toks2) > 3 else 0.0
            dn = float(toks2[4]) if len(toks2) > 4 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            thick = float(toks3[0]) if len(toks3) > 0 else 0.0
            ashear = float(toks3[1]) if len(toks3) > 1 else 0.0
            ithick = int(float(toks3[2])) if len(toks3) > 2 else 0
            iplas = int(float(toks3[3])) if len(toks3) > 3 else 0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            vx = float(toks4[0]) if len(toks4) > 0 else 0.0
            vy = float(toks4[1]) if len(toks4) > 1 else 0.0
            vz = float(toks4[2]) if len(toks4) > 2 else 0.0
            skew_id = int(float(toks4[3])) if len(toks4) > 3 else 0
            iorth = int(float(toks4[4])) if len(toks4) > 4 else 0
            ipos = int(float(toks4[5])) if len(toks4) > 5 else 0
            refplane = int(float(toks4[6])) if len(toks4) > 6 else 0

    p17 = PropType17(
        id=prop_id, ishell=ishell, ismstr=ismstr, ish3n=ish3n, idrill=idrill,
        plyxfem=plyxfem, z0=z0, vinterply=vinterply, hm=hm, hf=hf, hr=hr,
        dm=dm, dn=dn, thick=thick, ashear=ashear, ithick=ithick, iplas=iplas,
        vx=vx, vy=vy, vz=vz, skew_id=skew_id, iorth=iorth, ipos=ipos,
        refplane=refplane, title=title,
    )
    model.prop_type17s[prop_id] = p17
    model.properties[prop_id] = Property(
        id=prop_id, type=17, title=title,
        params={
            "thick": thick, "ashear": ashear,
            "ishell": ishell, "ismstr": ismstr, "ish3n": ish3n, "idrill": idrill,
            "plyxfem": plyxfem, "z0": z0, "vinterply": vinterply,
            "hm": hm, "hf": hf, "hr": hr, "dm": dm, "dn": dn,
            "ithick": ithick, "iplas": iplas,
            "vx": vx, "vy": vy, "vz": vz, "skew_id": skew_id, "iorth": iorth,
            "ipos": ipos, "refplane": refplane,
        }
    )







def read_prop_type44(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE44/id`` or ``/PROP/SPR_CRUS/id`` (M184): Crushing frame spring property."""
    from ...model.entities import PropType44, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PROP/TYPE44/{prop_id}: missing data cards", block.source)
        return

    mass, inertia, stiff1 = 0.0, 0.0, 0.0
    skew_csid, icoupling, ifiltr = 0, 0, 0
    k11, k44, k55, k66 = 0.0, 0.0, 0.0, 0.0
    idamp = 0
    k5b, k6c = 0.0, 0.0
    fun_a1, fun_b1, fun_a2, fscale11 = 0, 0, 0, 1.0
    fun_b2, fun_a3, fun_b3, fun_a4, fscale22 = 0, 0, 0, 0, 1.0
    fun_b4, fun_a5, fun_b5, fun_a6, fscale33 = 0, 0, 0, 0, 1.0
    fun_b6, fun_c1, fun_c2, fun_c3, fscale12 = 0, 0, 0, 0, 1.0
    fun_c4, fun_c5, fun_c6, fun_d1, fscale23 = 0, 0, 0, 0, 1.0
    fun_d2, fun_d3, fun_d4, fun_d5, fscale13 = 0, 0, 0, 0, 1.0
    strain1, strain2, strain3 = 0.0, 0.0, 0.0
    strain4, strain5, strain6, strain7 = 0.0, 0.0, 0.0, 0.0
    fct_d_x, dscale_x, f_x = 0, 0.0, 0.0
    fct_d_y, dscale_y, f_y = 0, 0.0, 0.0
    fct_d_z, dscale_z, f_z = 0, 0.0, 0.0
    fct_d_xx, dscale_xx, f_xx = 0, 0.0, 0.0
    fct_d_yy, dscale_yy, f_yy = 0, 0.0, 0.0
    fct_d_zz, dscale_zz, f_zz = 0, 0.0, 0.0

    if block.fixed:
        f1 = valid_cards[0].cut("PROP_TYPE44_1")
        mass = _fval(f1[0]) if len(f1) > 0 else 0.0
        inertia = _fval(f1[1]) if len(f1) > 1 else 0.0
        stiff1 = _fval(f1[2]) if len(f1) > 2 else 0.0
        skew_csid = _ival(f1[3]) if len(f1) > 3 else 0
        icoupling = _ival(f1[4]) if len(f1) > 4 else 0
        ifiltr = _ival(f1[5]) if len(f1) > 5 else 0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("PROP_TYPE44_2")
            k11 = _fval(f2[0]) if len(f2) > 0 else 0.0
            k44 = _fval(f2[1]) if len(f2) > 1 else 0.0
            k55 = _fval(f2[2]) if len(f2) > 2 else 0.0
            k66 = _fval(f2[3]) if len(f2) > 3 else 0.0
            idamp = _ival(f2[4]) if len(f2) > 4 else 0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("PROP_TYPE44_3")
            k5b = _fval(f3[0]) if len(f3) > 0 else 0.0
            k6c = _fval(f3[1]) if len(f3) > 1 else 0.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("PROP_TYPE44_4")
            fun_a1 = _ival(f4[0]) if len(f4) > 0 else 0
            fun_b1 = _ival(f4[1]) if len(f4) > 1 else 0
            fun_a2 = _ival(f4[2]) if len(f4) > 2 else 0
            fscale11 = _fval(f4[4], 1.0) if len(f4) > 4 and f4[4].strip() else 1.0

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("PROP_TYPE44_5")
            fun_b2 = _ival(f5[0]) if len(f5) > 0 else 0
            fun_a3 = _ival(f5[1]) if len(f5) > 1 else 0
            fun_b3 = _ival(f5[2]) if len(f5) > 2 else 0
            fun_a4 = _ival(f5[3]) if len(f5) > 3 else 0
            fscale22 = _fval(f5[4], 1.0) if len(f5) > 4 and f5[4].strip() else 1.0

        if len(valid_cards) > 5:
            f6 = valid_cards[5].cut("PROP_TYPE44_6")
            fun_b4 = _ival(f6[0]) if len(f6) > 0 else 0
            fun_a5 = _ival(f6[1]) if len(f6) > 1 else 0
            fun_b5 = _ival(f6[2]) if len(f6) > 2 else 0
            fun_a6 = _ival(f6[3]) if len(f6) > 3 else 0
            fscale33 = _fval(f6[4], 1.0) if len(f6) > 4 and f6[4].strip() else 1.0

        if len(valid_cards) > 6:
            f7 = valid_cards[6].cut("PROP_TYPE44_7")
            fun_b6 = _ival(f7[0]) if len(f7) > 0 else 0
            fun_c1 = _ival(f7[1]) if len(f7) > 1 else 0
            fun_c2 = _ival(f7[2]) if len(f7) > 2 else 0
            fun_c3 = _ival(f7[3]) if len(f7) > 3 else 0
            fscale12 = _fval(f7[4], 1.0) if len(f7) > 4 and f7[4].strip() else 1.0

        if len(valid_cards) > 7:
            f8 = valid_cards[7].cut("PROP_TYPE44_8")
            fun_c4 = _ival(f8[0]) if len(f8) > 0 else 0
            fun_c5 = _ival(f8[1]) if len(f8) > 1 else 0
            fun_c6 = _ival(f8[2]) if len(f8) > 2 else 0
            fun_d1 = _ival(f8[3]) if len(f8) > 3 else 0
            fscale23 = _fval(f8[4], 1.0) if len(f8) > 4 and f8[4].strip() else 1.0

        if len(valid_cards) > 8:
            f9 = valid_cards[8].cut("PROP_TYPE44_9")
            fun_d2 = _ival(f9[0]) if len(f9) > 0 else 0
            fun_d3 = _ival(f9[1]) if len(f9) > 1 else 0
            fun_d4 = _ival(f9[2]) if len(f9) > 2 else 0
            fun_d5 = _ival(f9[3]) if len(f9) > 3 else 0
            fscale13 = _fval(f9[4], 1.0) if len(f9) > 4 and f9[4].strip() else 1.0

        if len(valid_cards) > 9:
            f10 = valid_cards[9].cut("PROP_TYPE44_10")
            strain1 = _fval(f10[0]) if len(f10) > 0 else 0.0
            strain2 = _fval(f10[1]) if len(f10) > 1 else 0.0
            strain3 = _fval(f10[2]) if len(f10) > 2 else 0.0

        if len(valid_cards) > 10:
            f11 = valid_cards[10].cut("PROP_TYPE44_11")
            strain4 = _fval(f11[0]) if len(f11) > 0 else 0.0
            strain5 = _fval(f11[1]) if len(f11) > 1 else 0.0
            strain6 = _fval(f11[2]) if len(f11) > 2 else 0.0
            strain7 = _fval(f11[3]) if len(f11) > 3 else 0.0

        if idamp == 1 and len(valid_cards) > 11:
            idx = 11
            if len(valid_cards) > idx:
                f12 = valid_cards[idx].cut("PROP_TYPE44_12")
                fct_d_x = _ival(f12[0]) if len(f12) > 0 else 0
                dscale_x = _fval(f12[1]) if len(f12) > 1 else 0.0
                f_x = _fval(f12[2]) if len(f12) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                f13 = valid_cards[idx].cut("PROP_TYPE44_13")
                fct_d_y = _ival(f13[0]) if len(f13) > 0 else 0
                dscale_y = _fval(f13[1]) if len(f13) > 1 else 0.0
                f_y = _fval(f13[2]) if len(f13) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                f14 = valid_cards[idx].cut("PROP_TYPE44_14")
                fct_d_z = _ival(f14[0]) if len(f14) > 0 else 0
                dscale_z = _fval(f14[1]) if len(f14) > 1 else 0.0
                f_z = _fval(f14[2]) if len(f14) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                f15 = valid_cards[idx].cut("PROP_TYPE44_15")
                fct_d_xx = _ival(f15[0]) if len(f15) > 0 else 0
                dscale_xx = _fval(f15[1]) if len(f15) > 1 else 0.0
                f_xx = _fval(f15[2]) if len(f15) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                f16 = valid_cards[idx].cut("PROP_TYPE44_16")
                fct_d_yy = _ival(f16[0]) if len(f16) > 0 else 0
                dscale_yy = _fval(f16[1]) if len(f16) > 1 else 0.0
                f_yy = _fval(f16[2]) if len(f16) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                f17 = valid_cards[idx].cut("PROP_TYPE44_17")
                fct_d_zz = _ival(f17[0]) if len(f17) > 0 else 0
                dscale_zz = _fval(f17[1]) if len(f17) > 1 else 0.0
                f_zz = _fval(f17[2]) if len(f17) > 2 else 0.0
    else:
        toks1 = valid_cards[0].tokens()
        mass = float(toks1[0]) if len(toks1) > 0 else 0.0
        inertia = float(toks1[1]) if len(toks1) > 1 else 0.0
        stiff1 = float(toks1[2]) if len(toks1) > 2 else 0.0
        skew_csid = int(float(toks1[3])) if len(toks1) > 3 else 0
        icoupling = int(float(toks1[4])) if len(toks1) > 4 else 0
        ifiltr = int(float(toks1[5])) if len(toks1) > 5 else 0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            k11 = float(toks2[0]) if len(toks2) > 0 else 0.0
            k44 = float(toks2[1]) if len(toks2) > 1 else 0.0
            k55 = float(toks2[2]) if len(toks2) > 2 else 0.0
            k66 = float(toks2[3]) if len(toks2) > 3 else 0.0
            idamp = int(float(toks2[4])) if len(toks2) > 4 else 0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            k5b = float(toks3[0]) if len(toks3) > 0 else 0.0
            k6c = float(toks3[1]) if len(toks3) > 1 else 0.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            fun_a1 = int(float(toks4[0])) if len(toks4) > 0 else 0
            fun_b1 = int(float(toks4[1])) if len(toks4) > 1 else 0
            fun_a2 = int(float(toks4[2])) if len(toks4) > 2 else 0
            fscale11 = float(toks4[3]) if len(toks4) > 3 else 1.0

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            fun_b2 = int(float(toks5[0])) if len(toks5) > 0 else 0
            fun_a3 = int(float(toks5[1])) if len(toks5) > 1 else 0
            fun_b3 = int(float(toks5[2])) if len(toks5) > 2 else 0
            fun_a4 = int(float(toks5[3])) if len(toks5) > 3 else 0
            fscale22 = float(toks5[4]) if len(toks5) > 4 else 1.0

        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            fun_b4 = int(float(toks6[0])) if len(toks6) > 0 else 0
            fun_a5 = int(float(toks6[1])) if len(toks6) > 1 else 0
            fun_b5 = int(float(toks6[2])) if len(toks6) > 2 else 0
            fun_a6 = int(float(toks6[3])) if len(toks6) > 3 else 0
            fscale33 = float(toks6[4]) if len(toks6) > 4 else 1.0

        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            fun_b6 = int(float(toks7[0])) if len(toks7) > 0 else 0
            fun_c1 = int(float(toks7[1])) if len(toks7) > 1 else 0
            fun_c2 = int(float(toks7[2])) if len(toks7) > 2 else 0
            fun_c3 = int(float(toks7[3])) if len(toks7) > 3 else 0
            fscale12 = float(toks7[4]) if len(toks7) > 4 else 1.0

        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            fun_c4 = int(float(toks8[0])) if len(toks8) > 0 else 0
            fun_c5 = int(float(toks8[1])) if len(toks8) > 1 else 0
            fun_c6 = int(float(toks8[2])) if len(toks8) > 2 else 0
            fun_d1 = int(float(toks8[3])) if len(toks8) > 3 else 0
            fscale23 = float(toks8[4]) if len(toks8) > 4 else 1.0

        if len(valid_cards) > 8:
            toks9 = valid_cards[8].tokens()
            fun_d2 = int(float(toks9[0])) if len(toks9) > 0 else 0
            fun_d3 = int(float(toks9[1])) if len(toks9) > 1 else 0
            fun_d4 = int(float(toks9[2])) if len(toks9) > 2 else 0
            fun_d5 = int(float(toks9[3])) if len(toks9) > 3 else 0
            fscale13 = float(toks9[4]) if len(toks9) > 4 else 1.0

        if len(valid_cards) > 9:
            toks10 = valid_cards[9].tokens()
            strain1 = float(toks10[0]) if len(toks10) > 0 else 0.0
            strain2 = float(toks10[1]) if len(toks10) > 1 else 0.0
            strain3 = float(toks10[2]) if len(toks10) > 2 else 0.0

        if len(valid_cards) > 10:
            toks11 = valid_cards[10].tokens()
            strain4 = float(toks11[0]) if len(toks11) > 0 else 0.0
            strain5 = float(toks11[1]) if len(toks11) > 1 else 0.0
            strain6 = float(toks11[2]) if len(toks11) > 2 else 0.0
            strain7 = float(toks11[3]) if len(toks11) > 3 else 0.0

        if idamp == 1 and len(valid_cards) > 11:
            idx = 11
            if len(valid_cards) > idx:
                t12 = valid_cards[idx].tokens()
                fct_d_x = int(float(t12[0])) if len(t12) > 0 else 0
                dscale_x = float(t12[1]) if len(t12) > 1 else 0.0
                f_x = float(t12[2]) if len(t12) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                t13 = valid_cards[idx].tokens()
                fct_d_y = int(float(t13[0])) if len(t13) > 0 else 0
                dscale_y = float(t13[1]) if len(t13) > 1 else 0.0
                f_y = float(t13[2]) if len(t13) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                t14 = valid_cards[idx].tokens()
                fct_d_z = int(float(t14[0])) if len(t14) > 0 else 0
                dscale_z = float(t14[1]) if len(t14) > 1 else 0.0
                f_z = float(t14[2]) if len(t14) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                t15 = valid_cards[idx].tokens()
                fct_d_xx = int(float(t15[0])) if len(t15) > 0 else 0
                dscale_xx = float(t15[1]) if len(t15) > 1 else 0.0
                f_xx = float(t15[2]) if len(t15) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                t16 = valid_cards[idx].tokens()
                fct_d_yy = int(float(t16[0])) if len(t16) > 0 else 0
                dscale_yy = float(t16[1]) if len(t16) > 1 else 0.0
                f_yy = float(t16[2]) if len(t16) > 2 else 0.0
                idx += 1
            if len(valid_cards) > idx:
                t17 = valid_cards[idx].tokens()
                fct_d_zz = int(float(t17[0])) if len(t17) > 0 else 0
                dscale_zz = float(t17[1]) if len(t17) > 1 else 0.0
                f_zz = float(t17[2]) if len(t17) > 2 else 0.0

    p44 = PropType44(
        id=prop_id, mass=mass, inertia=inertia, stiff1=stiff1,
        skew_csid=skew_csid, icoupling=icoupling, ifiltr=ifiltr,
        k11=k11, k44=k44, k55=k55, k66=k66, idamp=idamp, k5b=k5b, k6c=k6c,
        fun_a1=fun_a1, fun_b1=fun_b1, fun_a2=fun_a2, fscale11=fscale11,
        fun_b2=fun_b2, fun_a3=fun_a3, fun_b3=fun_b3, fun_a4=fun_a4, fscale22=fscale22,
        fun_b4=fun_b4, fun_a5=fun_a5, fun_b5=fun_b5, fun_a6=fun_a6, fscale33=fscale33,
        fun_b6=fun_b6, fun_c1=fun_c1, fun_c2=fun_c2, fun_c3=fun_c3, fscale12=fscale12,
        fun_c4=fun_c4, fun_c5=fun_c5, fun_c6=fun_c6, fun_d1=fun_d1, fscale23=fscale23,
        fun_d2=fun_d2, fun_d3=fun_d3, fun_d4=fun_d4, fun_d5=fun_d5, fscale13=fscale13,
        strain1=strain1, strain2=strain2, strain3=strain3,
        strain4=strain4, strain5=strain5, strain6=strain6, strain7=strain7,
        fct_d_x=fct_d_x, dscale_x=dscale_x, f_x=f_x,
        fct_d_y=fct_d_y, dscale_y=dscale_y, f_y=f_y,
        fct_d_z=fct_d_z, dscale_z=dscale_z, f_z=f_z,
        fct_d_xx=fct_d_xx, dscale_xx=dscale_xx, f_xx=f_xx,
        fct_d_yy=fct_d_yy, dscale_yy=dscale_yy, f_yy=f_yy,
        fct_d_zz=fct_d_zz, dscale_zz=dscale_zz, f_zz=f_zz,
        title=title,
    )
    model.prop_type44s[prop_id] = p44
    model.properties[prop_id] = Property(
        id=prop_id, type=44, title=title,
        params={
            "mass": mass, "inertia": inertia, "stiff1": stiff1, "k": k11,
            "skew_csid": skew_csid, "icoupling": icoupling, "ifiltr": ifiltr,
            "k11": k11, "k44": k44, "k55": k55, "k66": k66, "idamp": idamp,
            "k5b": k5b, "k6c": k6c,
            "fun_a1": fun_a1, "fun_b1": fun_b1, "fun_a2": fun_a2, "fscale11": fscale11,
            "fun_b2": fun_b2, "fun_a3": fun_a3, "fun_b3": fun_b3, "fun_a4": fun_a4, "fscale22": fscale22,
            "fun_b4": fun_b4, "fun_a5": fun_a5, "fun_b5": fun_b5, "fun_a6": fun_a6, "fscale33": fscale33,
            "fun_b6": fun_b6, "fun_c1": fun_c1, "fun_c2": fun_c2, "fun_c3": fun_c3, "fscale12": fscale12,
            "fun_c4": fun_c4, "fun_c5": fun_c5, "fun_c6": fun_c6, "fun_d1": fun_d1, "fscale23": fscale23,
            "fun_d2": fun_d2, "fun_d3": fun_d3, "fun_d4": fun_d4, "fun_d5": fun_d5, "fscale13": fscale13,
            "strain1": strain1, "strain2": strain2, "strain3": strain3,
            "strain4": strain4, "strain5": strain5, "strain6": strain6, "strain7": strain7,
            "fct_d_x": fct_d_x, "dscale_x": dscale_x, "f_x": f_x,
            "fct_d_y": fct_d_y, "dscale_y": dscale_y, "f_y": f_y,
            "fct_d_z": fct_d_z, "dscale_z": dscale_z, "f_z": f_z,
            "fct_d_xx": fct_d_xx, "dscale_xx": dscale_xx, "f_xx": f_xx,
            "fct_d_yy": fct_d_yy, "dscale_yy": dscale_yy, "f_yy": f_yy,
            "fct_d_zz": fct_d_zz, "dscale_zz": dscale_zz, "f_zz": f_zz,
        }
    )




def read_prop_type12(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE12`` or ``/PROP/SPR_PUL`` (M185): Pulley spring / sliding cable property."""
    from ...model.entities import PropType12, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    mass = 0.0
    isensor = 0
    isflag = 0
    ileng = 0
    fric = 0.0
    stiff1 = 0.0
    damp1 = 0.0
    acoeft1 = 1.0
    bcoeft1 = 0.0
    dcoeft1 = 1.0
    fun_a1 = 0
    hflag1 = 0
    fun_b1 = 0
    fct_id31 = 0
    fun_a2 = 0
    min_rup1 = -1.0e30
    max_rup1 = 1.0e30
    prop_x_f = 1.0
    prop_x_e = 0.0
    scale1 = 1.0
    h = 1.0
    funct_id = 0
    ifric = 0
    scale2 = 1.0
    scale3 = 1.0
    f_min = -1.0e30
    f_max = 1.0e30

    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE12_1")
            mass = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            isensor = _safe_int(c0[2]) if len(c0) > 2 else 0
            isflag = _safe_int(c0[3]) if len(c0) > 3 else 0
            ileng = _safe_int(c0[4]) if len(c0) > 4 else 0
            fric = _safe_float(c0[5]) if len(c0) > 5 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE12_2")
            stiff1 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            damp1 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            acoeft1 = _safe_float(c1[2]) if len(c1) > 2 and c1[2].strip() else 1.0
            bcoeft1 = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            dcoeft1 = _safe_float(c1[4]) if len(c1) > 4 and c1[4].strip() else 1.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE12_3")
            if len(c2) >= 8 and (c2[6].strip() or c2[7].strip()):
                fun_a1 = _safe_int(c2[0]) if len(c2) > 0 else 0
                hflag1 = _safe_int(c2[1]) if len(c2) > 1 else 0
                fun_b1 = _safe_int(c2[2]) if len(c2) > 2 else 0
                fct_id31 = _safe_int(c2[3]) if len(c2) > 3 else 0
                fun_a2 = _safe_int(c2[4]) if len(c2) > 4 else 0
                min_rup1 = _safe_float(c2[6]) if len(c2) > 6 and c2[6].strip() else -1.0e30
                max_rup1 = _safe_float(c2[7]) if len(c2) > 7 and c2[7].strip() else 1.0e30
            else:
                c2_old = valid_cards[2].cut("PROP_TYPE12_3_OLD")
                fun_a1 = _safe_int(c2_old[0]) if len(c2_old) > 0 else 0
                hflag1 = _safe_int(c2_old[1]) if len(c2_old) > 1 else 0
                fun_b1 = _safe_int(c2_old[2]) if len(c2_old) > 2 else 0
                min_rup1 = _safe_float(c2_old[4]) if len(c2_old) > 4 and c2_old[4].strip() else -1.0e30
                max_rup1 = _safe_float(c2_old[5]) if len(c2_old) > 5 and c2_old[5].strip() else 1.0e30
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_TYPE12_4")
            prop_x_f = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 1.0
            prop_x_e = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            scale1 = _safe_float(c3[2]) if len(c3) > 2 and c3[2].strip() else 1.0
            h = _safe_float(c3[3]) if len(c3) > 3 and c3[3].strip() else 1.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("PROP_TYPE12_5")
            funct_id = _safe_int(c4[0]) if len(c4) > 0 else 0
            ifric = _safe_int(c4[1]) if len(c4) > 1 else 0
            scale2 = _safe_float(c4[2]) if len(c4) > 2 and c4[2].strip() else 1.0
            scale3 = _safe_float(c4[3]) if len(c4) > 3 and c4[3].strip() else 1.0
            f_min = _safe_float(c4[4]) if len(c4) > 4 and c4[4].strip() else -1.0e30
            f_max = _safe_float(c4[5]) if len(c4) > 5 and c4[5].strip() else 1.0e30
    else:
        if len(valid_cards) > 0:
            toks = valid_cards[0].tokens()
            mass = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            if len(toks) == 5:
                isensor = _safe_int(toks[1])
                isflag = _safe_int(toks[2])
                ileng = _safe_int(toks[3])
                fric = _safe_float(toks[4])
            elif len(toks) >= 6:
                isensor = _safe_int(toks[2])
                isflag = _safe_int(toks[3])
                ileng = _safe_int(toks[4])
                fric = _safe_float(toks[5])
            elif len(toks) == 4:
                isensor = _safe_int(toks[1])
                isflag = _safe_int(toks[2])
                fric = _safe_float(toks[3])
            elif len(toks) == 2:
                fric = _safe_float(toks[1])
        if len(valid_cards) > 1:
            toks = valid_cards[1].tokens()
            stiff1 = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            damp1 = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            acoeft1 = _safe_float(toks[2]) if len(toks) > 2 else 1.0
            bcoeft1 = _safe_float(toks[3]) if len(toks) > 3 else 0.0
            dcoeft1 = _safe_float(toks[4]) if len(toks) > 4 else 1.0
        if len(valid_cards) > 2:
            toks = valid_cards[2].tokens()
            fun_a1 = _safe_int(toks[0]) if len(toks) > 0 else 0
            hflag1 = _safe_int(toks[1]) if len(toks) > 1 else 0
            fun_b1 = _safe_int(toks[2]) if len(toks) > 2 else 0
            if len(toks) >= 7:
                fct_id31 = _safe_int(toks[3])
                fun_a2 = _safe_int(toks[4])
                min_rup1 = _safe_float(toks[5])
                max_rup1 = _safe_float(toks[6])
            elif len(toks) >= 5:
                min_rup1 = _safe_float(toks[3])
                max_rup1 = _safe_float(toks[4])
        if len(valid_cards) > 3:
            toks = valid_cards[3].tokens()
            prop_x_f = _safe_float(toks[0]) if len(toks) > 0 else 1.0
            prop_x_e = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            scale1 = _safe_float(toks[2]) if len(toks) > 2 else 1.0
            h = _safe_float(toks[3]) if len(toks) > 3 else 1.0
        if len(valid_cards) > 4:
            toks = valid_cards[4].tokens()
            funct_id = _safe_int(toks[0]) if len(toks) > 0 else 0
            ifric = _safe_int(toks[1]) if len(toks) > 1 else 0
            scale2 = _safe_float(toks[2]) if len(toks) > 2 else 1.0
            scale3 = _safe_float(toks[3]) if len(toks) > 3 else 1.0
            f_min = _safe_float(toks[4]) if len(toks) > 4 else -1.0e30
            f_max = _safe_float(toks[5]) if len(toks) > 5 else 1.0e30

    p12 = PropType12(
        id=prop_id, mass=mass, isensor=isensor, isflag=isflag, ileng=ileng, fric=fric,
        stiff1=stiff1, damp1=damp1, acoeft1=acoeft1, bcoeft1=bcoeft1, dcoeft1=dcoeft1,
        fun_a1=fun_a1, hflag1=hflag1, fun_b1=fun_b1, fct_id31=fct_id31, fun_a2=fun_a2,
        min_rup1=min_rup1, max_rup1=max_rup1,
        prop_x_f=prop_x_f, prop_x_e=prop_x_e, scale1=scale1, h=h,
        funct_id=funct_id, ifric=ifric, scale2=scale2, scale3=scale3,
        f_min=f_min, f_max=f_max,
        title=title,
    )
    model.prop_type12s[prop_id] = p12
    model.properties[prop_id] = Property(
        id=prop_id, type=12, title=title,
        params={
            "mass": mass, "isensor": isensor, "sens_id": isensor, "isflag": isflag, "ileng": ileng, "fric": fric,
            "stiff1": stiff1, "stiff": stiff1, "k": stiff1, "damp1": damp1, "damp": damp1, "c": damp1,
            "acoeft1": acoeft1, "a": acoeft1, "bcoeft1": bcoeft1, "b": bcoeft1, "dcoeft1": dcoeft1, "d": dcoeft1,
            "fun_a1": fun_a1, "fun_a": fun_a1, "fun_k": fun_a1, "fct_id1": fun_a1,
            "hflag1": hflag1, "hflag": hflag1,
            "fun_b1": fun_b1, "fun_b": fun_b1, "fun_c": fun_b1, "fct_id2": fun_b1,
            "fct_id31": fct_id31, "fun_a2": fun_a2,
            "min_rup1": min_rup1, "min_rup": min_rup1, "delta_min": min_rup1, "dmin": min_rup1,
            "max_rup1": max_rup1, "max_rup": max_rup1, "delta_max": max_rup1, "dmax": max_rup1,
            "prop_x_f": prop_x_f, "fscale": prop_x_f, "prop_x_e": prop_x_e, "e": prop_x_e,
            "scale1": scale1, "ascale": scale1, "h": h,
            "funct_id": funct_id, "fct_idfr": funct_id, "ifric": ifric,
            "scale2": scale2, "yscale_f": scale2, "scale3": scale3, "xscale_f": scale3,
            "f_min": f_min, "f_max": f_max,
        }
    )




def read_prop_type15(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE15`` or ``/PROP/POROUS`` (M185): Porous solid property."""
    from ...model.entities import PropType15, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    qa = 0.0
    qb = 0.0
    h = 0.1
    poros = 1.0
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0
    skew_csid = 0
    ihon = 0
    itu = 0
    alpha = 0.1
    l_mix = 0.0
    irby = 0

    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE15_1")
            qa = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            qb = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            h = _safe_float(c0[2]) if len(c0) > 2 and c0[2].strip() else 0.1
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE15_2")
            poros = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 1.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE15_3")
            r1 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            r2 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            r3 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_TYPE15_4")
            skew_csid = _safe_int(c3[0]) if len(c3) > 0 else 0
            ihon = _safe_int(c3[1]) if len(c3) > 1 else 0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("PROP_TYPE15_5")
            itu = _safe_int(c4[0]) if len(c4) > 0 else 0
            alpha = _safe_float(c4[1]) if len(c4) > 1 and c4[1].strip() else 0.1
            l_mix = _safe_float(c4[2]) if len(c4) > 2 else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("PROP_TYPE15_6")
            irby = _safe_int(c5[0]) if len(c5) > 0 else 0
    else:
        if len(valid_cards) > 0:
            toks = valid_cards[0].tokens()
            qa = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            qb = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            h = _safe_float(toks[2]) if len(toks) > 2 else 0.1
        if len(valid_cards) > 1:
            toks = valid_cards[1].tokens()
            poros = _safe_float(toks[0]) if len(toks) > 0 else 1.0
        if len(valid_cards) > 2:
            toks = valid_cards[2].tokens()
            r1 = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            r2 = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            r3 = _safe_float(toks[2]) if len(toks) > 2 else 0.0
        if len(valid_cards) > 3:
            toks = valid_cards[3].tokens()
            skew_csid = _safe_int(toks[0]) if len(toks) > 0 else 0
            ihon = _safe_int(toks[1]) if len(toks) > 1 else 0
        if len(valid_cards) > 4:
            toks = valid_cards[4].tokens()
            itu = _safe_int(toks[0]) if len(toks) > 0 else 0
            alpha = _safe_float(toks[1]) if len(toks) > 1 else 0.1
            l_mix = _safe_float(toks[2]) if len(toks) > 2 else 0.0
        if len(valid_cards) > 5:
            toks = valid_cards[5].tokens()
            irby = _safe_int(toks[0]) if len(toks) > 0 else 0

    p15 = PropType15(
        id=prop_id, qa=qa, qb=qb, h=h, poros=poros,
        r1=r1, r2=r2, r3=r3, skew_csid=skew_csid, ihon=ihon,
        itu=itu, alpha=alpha, l_mix=l_mix, irby=irby,
        title=title,
    )
    model.prop_type15s[prop_id] = p15
    model.properties[prop_id] = Property(
        id=prop_id, type=15, title=title,
        params={
            "qa": qa, "qb": qb, "h": h, "poros": poros, "porosity": poros, "por": poros,
            "r1": r1, "pdir1": r1, "r2": r2, "pdir2": r2, "r3": r3, "pdir3": r3,
            "skew_csid": skew_csid, "skew_id": skew_csid,
            "ihon": ihon, "iflag": ihon,
            "itu": itu, "i_th": itu,
            "alpha": alpha,
            "l_mix": l_mix, "thick": l_mix,
            "irby": irby,
        }
    )




def read_prop_type28(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE28`` or ``/PROP/NSTRAND`` (M185): Multi-strand cable / wire rope property."""
    from ...model.entities import PropType28, PropStrandLayer, Property, PropXelem
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    mass = 0.0
    k = 0.0
    c = 0.0
    fun_a1 = 0
    fun_b1 = 0
    strain1 = -1.0e30
    strain2 = 1.0e30
    mu1 = 0.0
    mu2 = 0.0
    layers: list[PropStrandLayer] = []

    fscale11 = 1.0
    fscale22 = 1.0
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE28_1")
            mass = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            k = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            c = _safe_float(c0[2]) if len(c0) > 2 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE28_2")
            fun_a1 = _safe_int(c1[0]) if len(c1) > 0 else 0
            fun_b1 = _safe_int(c1[1]) if len(c1) > 1 else 0
            strain1 = _safe_float(c1[2]) if len(c1) > 2 and c1[2].strip() else -1.0e30
            strain2 = _safe_float(c1[3]) if len(c1) > 3 and c1[3].strip() else 1.0e30
            fscale11 = _safe_float(c1[4]) if len(c1) > 4 and c1[4].strip() else 1.0
            fscale22 = _safe_float(c1[5]) if len(c1) > 5 and c1[5].strip() else 1.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE28_3")
            mu1 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            mu2 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
        for line in valid_cards[3:]:
            cl = line.cut("PROP_TYPE28_4")
            tname = cl[0].strip() if len(cl) > 0 else ""
            kid = _safe_int(cl[1]) if len(cl) > 1 else 0
            mu = _safe_float(cl[2]) if len(cl) > 2 else 0.0
            if tname or kid != 0 or mu != 0.0:
                layers.append(PropStrandLayer(type_name=tname, k_id=kid, mu=mu))
    else:
        if len(valid_cards) > 0:
            toks = valid_cards[0].tokens()
            mass = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            k = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            c = _safe_float(toks[2]) if len(toks) > 2 else 0.0
        if len(valid_cards) > 1:
            toks = valid_cards[1].tokens()
            fun_a1 = _safe_int(toks[0]) if len(toks) > 0 else 0
            fun_b1 = _safe_int(toks[1]) if len(toks) > 1 else 0
            strain1 = _safe_float(toks[2]) if len(toks) > 2 else -1.0e30
            strain2 = _safe_float(toks[3]) if len(toks) > 3 else 1.0e30
            fscale11 = _safe_float(toks[4]) if len(toks) > 4 else 1.0
            fscale22 = _safe_float(toks[5]) if len(toks) > 5 else 1.0
        if len(valid_cards) > 2:
            toks = valid_cards[2].tokens()
            mu1 = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            mu2 = _safe_float(toks[1]) if len(toks) > 1 else 0.0
        for line in valid_cards[3:]:
            toks = line.tokens()
            if len(toks) >= 3:
                tname = toks[0]
                kid = _safe_int(toks[1])
                mu = _safe_float(toks[2])
                layers.append(PropStrandLayer(type_name=tname, k_id=kid, mu=mu))

    if fscale11 == 0.0:
        fscale11 = 1.0
    if fscale22 == 0.0:
        fscale22 = 1.0

    p28 = PropType28(
        id=prop_id, mass=mass, k=k, c=c, fun_a1=fun_a1, fun_b1=fun_b1,
        strain1=strain1, strain2=strain2, mu1=mu1, mu2=mu2,
        fscale11=fscale11, fscale22=fscale22,
        layers=layers, title=title,
    )
    model.prop_type28s[prop_id] = p28
    model.prop_xelems[prop_id] = PropXelem(
        id=prop_id, title=title, itip=int(mass) if mass else 0, isurf=int(k) if k else 0, alpha=c
    )
    model.properties[prop_id] = Property(
        id=prop_id, type=28, title=title,
        params={
            "mass": mass, "k": k, "c": c,
            "fun_a1": fun_a1, "fun_k": fun_a1, "fun_b1": fun_b1, "fun_c": fun_b1,
            "strain1": strain1, "dmin": strain1, "delta_min": strain1,
            "strain2": strain2, "dmax": strain2, "delta_max": strain2,
            "fscale11": fscale11, "ffac": fscale11,
            "fscale22": fscale22, "xfac": 1.0 / fscale22,
            "mu1": mu1, "mu2": mu2,
            "nip": len(layers),
            "layers": layers,
        }
    )




def read_prop_type33(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE33`` or ``/PROP/KJOINT`` / ``/PROP/KINEMATIC_JOINT`` (M186): 6-DOF kinematic joint property."""
    from ...model.entities import PropType33, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE33/{prop_id}: missing data card", block.source)
        return

    joint_type, skew_flag = 1, 0
    id_sk1, id_sk2 = 0, 0
    xk, cr, kn = 0.0, 0.0, 0.0
    krx, kry, krz, ktx, kty, ktz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xr_fun, yr_fun, zr_fun, xt_fun, yt_fun, zt_fun = 0, 0, 0, 0, 0, 0
    crx, cry, crz, ctx, cty, ctz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    crx_fun, cry_fun, crz_fun, ctx_fun, cty_fun, ctz_fun = 0, 0, 0, 0, 0, 0

    idx = 0
    if block.fixed:
        if idx < len(valid_cards):
            c0 = valid_cards[idx].cut("PROP_KJOINT_1")
            joint_type = _safe_int(c0[0]) if len(c0) > 0 else 1
            skew_flag = _safe_int(c0[1]) if len(c0) > 1 else 0
            idx += 1
        if idx < len(valid_cards):
            c1 = valid_cards[idx].cut("PROP_KJOINT_2")
            id_sk1 = _safe_int(c1[0]) if len(c1) > 0 else 0
            id_sk2 = _safe_int(c1[1]) if len(c1) > 1 else 0
            xk = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            cr = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            idx += 1
        if idx < len(valid_cards):
            c2 = valid_cards[idx].cut("PROP_KJOINT_3")
            kn = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            krx = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            kry = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            krz = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            idx += 1
        if idx < len(valid_cards):
            c3 = valid_cards[idx].cut("PROP_KJOINT_4")
            xr_fun = _safe_int(c3[0]) if len(c3) > 0 else 0
            yr_fun = _safe_int(c3[1]) if len(c3) > 1 else 0
            zr_fun = _safe_int(c3[2]) if len(c3) > 2 else 0
            idx += 1
        if idx < len(valid_cards):
            c4 = valid_cards[idx].cut("PROP_KJOINT_5")
            crx = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            cry = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            crz = _safe_float(c4[2]) if len(c4) > 2 else 0.0
            idx += 1
        if idx < len(valid_cards):
            c5 = valid_cards[idx].cut("PROP_KJOINT_6")
            crx_fun = _safe_int(c5[0]) if len(c5) > 0 else 0
            cry_fun = _safe_int(c5[1]) if len(c5) > 1 else 0
            crz_fun = _safe_int(c5[2]) if len(c5) > 2 else 0
            idx += 1
    else:
        if idx < len(valid_cards):
            t0 = valid_cards[idx].tokens()
            joint_type = _safe_int(t0[0]) if len(t0) > 0 else 1
            skew_flag = _safe_int(t0[1]) if len(t0) > 1 else 0
            idx += 1
        if idx < len(valid_cards):
            t1 = valid_cards[idx].tokens()
            id_sk1 = _safe_int(t1[0]) if len(t1) > 0 else 0
            id_sk2 = _safe_int(t1[1]) if len(t1) > 1 else 0
            xk = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            cr = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            idx += 1
        if idx < len(valid_cards):
            t2 = valid_cards[idx].tokens()
            kn = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            krx = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            kry = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            krz = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            idx += 1
        if idx < len(valid_cards):
            t3 = valid_cards[idx].tokens()
            xr_fun = _safe_int(t3[0]) if len(t3) > 0 else 0
            yr_fun = _safe_int(t3[1]) if len(t3) > 1 else 0
            zr_fun = _safe_int(t3[2]) if len(t3) > 2 else 0
            idx += 1
        if idx < len(valid_cards):
            t4 = valid_cards[idx].tokens()
            crx = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            cry = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            crz = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            idx += 1
        if idx < len(valid_cards):
            t5 = valid_cards[idx].tokens()
            crx_fun = _safe_int(t5[0]) if len(t5) > 0 else 0
            cry_fun = _safe_int(t5[1]) if len(t5) > 1 else 0
            crz_fun = _safe_int(t5[2]) if len(t5) > 2 else 0
            idx += 1

    p33 = PropType33(
        id=prop_id, joint_type=joint_type, skew_flag=skew_flag,
        id_sk1=id_sk1, id_sk2=id_sk2, xk=xk, cr=cr, kn=kn,
        krx=krx, kry=kry, krz=krz, ktx=ktx, kty=kty, ktz=ktz,
        xr_fun=xr_fun, yr_fun=yr_fun, zr_fun=zr_fun,
        xt_fun=xt_fun, yt_fun=yt_fun, zt_fun=zt_fun,
        crx=crx, cry=cry, crz=crz, ctx=ctx, cty=cty, ctz=ctz,
        crx_fun=crx_fun, cry_fun=cry_fun, crz_fun=crz_fun,
        ctx_fun=ctx_fun, cty_fun=cty_fun, ctz_fun=ctz_fun,
        title=title,
    )
    model.prop_type33s[prop_id] = p33
    model.properties[prop_id] = Property(
        id=prop_id, type=33, title=title,
        params={
            "joint_type": joint_type, "type": joint_type, "skew_flag": skew_flag,
            "id_sk1": id_sk1, "id_sk2": id_sk2, "xk": xk, "cr": cr, "kn": kn,
            "krx": krx, "kry": kry, "krz": krz, "ktx": ktx, "kty": kty, "ktz": ktz,
            "xr_fun": xr_fun, "yr_fun": yr_fun, "zr_fun": zr_fun,
            "xt_fun": xt_fun, "yt_fun": yt_fun, "zt_fun": zt_fun,
            "crx": crx, "cry": cry, "crz": crz, "ctx": ctx, "cty": cty, "ctz": ctz,
            "crx_fun": crx_fun, "cry_fun": cry_fun, "crz_fun": crz_fun,
            "ctx_fun": ctx_fun, "cty_fun": cty_fun, "ctz_fun": ctz_fun,
        }
    )




def read_prop_type46(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE46`` or ``/PROP/SPR_MUSCLE`` / ``/PROP/MUSCLE`` (M186): Hill-type muscle spring property."""
    from ...model.entities import PropType46, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE46/{prop_id}: missing data card", block.source)
        return

    mass, stiff0, vel_max, nforce, stiff1 = 0.0, 0.0, 0.0, 0.0, 0.0
    fun_a1, fun_b1, fun_c1, fun_d1 = 0, 0, 0, 0
    mat_imass = 0
    damp1 = 0.0
    epsi = 0
    fscale11, fscale22, fscale21, fscale12 = 1.0, 1.0, 1.0, 1.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_SPR_MUSCLE_1")
            mass = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            stiff0 = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            vel_max = _safe_float(c0[2]) if len(c0) > 2 else 0.0
            nforce = _safe_float(c0[3]) if len(c0) > 3 else 0.0
            stiff1 = _safe_float(c0[4]) if len(c0) > 4 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_SPR_MUSCLE_2")
            fun_a1 = _safe_int(c1[0]) if len(c1) > 0 else 0
            fun_b1 = _safe_int(c1[1]) if len(c1) > 1 else 0
            fun_c1 = _safe_int(c1[2]) if len(c1) > 2 else 0
            fun_d1 = _safe_int(c1[3]) if len(c1) > 3 else 0
            mat_imass = _safe_int(c1[5]) if len(c1) > 5 else 0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_SPR_MUSCLE_3")
            damp1 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            epsi = _safe_int(c2[1]) if len(c2) > 1 else 0
            fscale11 = _safe_float(c2[2]) if len(c2) > 2 and c2[2].strip() else 1.0
            fscale22 = _safe_float(c2[3]) if len(c2) > 3 and c2[3].strip() else 1.0
            fscale21 = _safe_float(c2[4]) if len(c2) > 4 and c2[4].strip() else 1.0
            fscale12 = _safe_float(c2[5]) if len(c2) > 5 and c2[5].strip() else 1.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            mass = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            stiff0 = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            vel_max = _safe_float(t0[2]) if len(t0) > 2 else 0.0
            nforce = _safe_float(t0[3]) if len(t0) > 3 else 0.0
            stiff1 = _safe_float(t0[4]) if len(t0) > 4 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            fun_a1 = _safe_int(t1[0]) if len(t1) > 0 else 0
            fun_b1 = _safe_int(t1[1]) if len(t1) > 1 else 0
            fun_c1 = _safe_int(t1[2]) if len(t1) > 2 else 0
            fun_d1 = _safe_int(t1[3]) if len(t1) > 3 else 0
            mat_imass = _safe_int(t1[4]) if len(t1) > 4 else 0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            damp1 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            epsi = _safe_int(t2[1]) if len(t2) > 1 else 0
            fscale11 = _safe_float(t2[2]) if len(t2) > 2 else 1.0
            fscale22 = _safe_float(t2[3]) if len(t2) > 3 else 1.0
            fscale21 = _safe_float(t2[4]) if len(t2) > 4 else 1.0
            fscale12 = _safe_float(t2[5]) if len(t2) > 5 else 1.0

    # M149 format compatibility fields
    itype_val, ifunc_ce, ifunc_see, ifunc_pe, ifunc_de, ifunc_act = 0, 0, 0, 0, 0, 0
    f_see0, iflag_val = 0.0, 0
    gamma_m, beta_m, act_init, tau_act = 0.0, 0.0, 0.0, 0.0
    if len(valid_cards) >= 4:
        t1 = valid_cards[1].tokens()
        if len(t1) >= 6:
            itype_val = _safe_int(t1[0])
            ifunc_ce = _safe_int(t1[1])
            ifunc_see = _safe_int(t1[2])
            ifunc_pe = _safe_int(t1[3])
            ifunc_de = _safe_int(t1[4])
            ifunc_act = _safe_int(t1[5])
        t2 = valid_cards[2].tokens()
        if len(t2) >= 2:
            f_see0 = _safe_float(t2[0])
            iflag_val = _safe_int(t2[1])
        t3 = valid_cards[3].tokens()
        if len(t3) >= 4:
            gamma_m = _safe_float(t3[0])
            beta_m = _safe_float(t3[1])
            act_init = _safe_float(t3[2])
            tau_act = _safe_float(t3[3])

    p46 = PropType46(
        id=prop_id, mass=mass, stiff0=stiff0, vel_max=vel_max, nforce=nforce, stiff1=stiff1,
        fun_a1=fun_a1, fun_b1=fun_b1, fun_c1=fun_c1, fun_d1=fun_d1,
        mat_imass=mat_imass, damp1=damp1, epsi=epsi,
        fscale11=fscale11, fscale22=fscale22, fscale21=fscale21, fscale12=fscale12,
        title=title,
    )
    model.prop_type46s[prop_id] = p46
    model.properties[prop_id] = Property(
        id=prop_id, type=46, title=title,
        params={
            "mass": mass, "stiff0": stiff0, "vel_max": vel_max, "nforce": nforce, "stiff1": stiff1,
            "fun_a1": fun_a1, "fun_b1": fun_b1, "fun_c1": fun_c1, "fun_d1": fun_d1,
            "mat_imass": mat_imass, "damp1": damp1, "epsi": epsi,
            "fscale11": fscale11, "fscale22": fscale22, "fscale21": fscale21, "fscale12": fscale12,
            # M149 compatibility aliases
            "f_max": stiff0, "l_opt": vel_max, "v_max": nforce, "k_pe": stiff1,
            "itype": itype_val, "ifunc_ce": ifunc_ce, "ifunc_see": ifunc_see,
            "ifunc_pe": ifunc_pe, "ifunc_de": ifunc_de, "ifunc_act": ifunc_act,
            "f_see0": f_see0, "iflag": iflag_val,
            "gamma_m": gamma_m, "beta_m": beta_m, "act_init": act_init, "tau_act": tau_act,
        }
    )




def read_prop_type35(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE35`` or ``/PROP/STITCH`` / ``/PROP/SEW`` (M186): Stitch seam fastener property."""
    from ...model.entities import PropType35, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE35/{prop_id}: missing data card", block.source)
        return

    amas, elastif, xlim1, xlim2, xk = 0.0, 0.0, 0.0, 0.0, 0.0
    fun_a1, fun_b1, fun_c1, fun_d1 = 0, 0, 0, 0
    damg, fdelay, rload, fscal = 0.0, 0.0, 0.0, 1.0
    # M149 format compatibility fields
    k_tens, k_comp, k_shear, f_tens, f_shear = 0.0, 0.0, 0.0, 0.0, 0.0
    skew_id, iflag, ipen, ifail, dist_max, area = 0, 0, 0, 0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) >= 3:
            # 3-card /PROP/TYPE35 layout (hm_read_prop35.F)
            c0 = valid_cards[0].cut("PROP_TYPE35_1")
            amas = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            elastif = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            xlim1 = _safe_float(c0[2]) if len(c0) > 2 else 0.0
            xlim2 = _safe_float(c0[3]) if len(c0) > 3 else 0.0
            xk = _safe_float(c0[4]) if len(c0) > 4 else 0.0

            c1 = valid_cards[1].cut("PROP_TYPE35_2")
            damg = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            fdelay = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            rload = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            fscal = _safe_float(c1[3], 1.0) if len(c1) > 3 and c1[3].strip() else 1.0

            c2 = valid_cards[2].cut("PROP_TYPE35_3")
            fun_a1 = _safe_int(c2[0]) if len(c2) > 0 else 0
            fun_b1 = _safe_int(c2[1]) if len(c2) > 1 else 0
            fun_c1 = _safe_int(c2[2]) if len(c2) > 2 else 0
            fun_d1 = _safe_int(c2[3]) if len(c2) > 3 else 0
        else:
            # 2-card /PROP/STITCH layout (prop_p35_stitch.cfg)
            if len(valid_cards) > 0:
                c0 = valid_cards[0].cut("PROP_STITCH_1")
                amas = _safe_float(c0[0]) if len(c0) > 0 else 0.0
                elastif = _safe_float(c0[1]) if len(c0) > 1 else 0.0
                xlim1 = _safe_float(c0[2]) if len(c0) > 2 else 0.0
                xk = _safe_float(c0[3]) if len(c0) > 3 else 0.0
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut("PROP_STITCH_2")
                fun_a1 = _safe_int(c1[0]) if len(c1) > 0 else 0
                fun_b1 = _safe_int(c1[1]) if len(c1) > 1 else 0
                fun_c1 = _safe_int(c1[2]) if len(c1) > 2 else 0
                fun_d1 = _safe_int(c1[3]) if len(c1) > 3 else 0
                damg = _safe_float(c1[4]) if len(c1) > 4 else 0.0
                fdelay = _safe_float(c1[5]) if len(c1) > 5 else 0.0
    else:
        if len(valid_cards) >= 3 and len(valid_cards[2].tokens()) == 4:
            # 3-card free format
            t0 = valid_cards[0].tokens()
            amas = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            elastif = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            xlim1 = _safe_float(t0[2]) if len(t0) > 2 else 0.0
            xlim2 = _safe_float(t0[3]) if len(t0) > 3 else 0.0
            xk = _safe_float(t0[4]) if len(t0) > 4 else 0.0

            t1 = valid_cards[1].tokens()
            damg = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            fdelay = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            rload = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            fscal = _safe_float(t1[3], 1.0) if len(t1) > 3 else 1.0

            t2 = valid_cards[2].tokens()
            fun_a1 = _safe_int(t2[0]) if len(t2) > 0 else 0
            fun_b1 = _safe_int(t2[1]) if len(t2) > 1 else 0
            fun_c1 = _safe_int(t2[2]) if len(t2) > 2 else 0
            fun_d1 = _safe_int(t2[3]) if len(t2) > 3 else 0
        elif len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            if len(t0) >= 5:
                # M149 format
                k_tens = _safe_float(t0[0])
                k_comp = _safe_float(t0[1])
                k_shear = _safe_float(t0[2])
                f_tens = _safe_float(t0[3])
                f_shear = _safe_float(t0[4])
                if len(valid_cards) > 1:
                    t1 = valid_cards[1].tokens()
                    skew_id = _safe_int(t1[0]) if len(t1) > 0 else 0
                    iflag = _safe_int(t1[1]) if len(t1) > 1 else 0
                    ipen = _safe_int(t1[2]) if len(t1) > 2 else 0
                    ifail = _safe_int(t1[3]) if len(t1) > 3 else 0
                    dist_max = _safe_float(t1[4]) if len(t1) > 4 else 0.0
                    area = _safe_float(t1[5]) if len(t1) > 5 else 0.0
            else:
                amas = _safe_float(t0[0]) if len(t0) > 0 else 0.0
                elastif = _safe_float(t0[1]) if len(t0) > 1 else 0.0
                xlim1 = _safe_float(t0[2]) if len(t0) > 2 else 0.0
                xk = _safe_float(t0[3]) if len(t0) > 3 else 0.0
            if len(valid_cards) > 1 and len(valid_cards[0].tokens()) < 5:
                t1 = valid_cards[1].tokens()
                fun_a1 = _safe_int(t1[0]) if len(t1) > 0 else 0
                fun_b1 = _safe_int(t1[1]) if len(t1) > 1 else 0
                fun_c1 = _safe_int(t1[2]) if len(t1) > 2 else 0
                fun_d1 = _safe_int(t1[3]) if len(t1) > 3 else 0
                damg = _safe_float(t1[4]) if len(t1) > 4 else 0.0
                fdelay = _safe_float(t1[5]) if len(t1) > 5 else 0.0

    if fscal == 0.0:
        fscal = 1.0

    p35 = PropType35(
        id=prop_id, amas=amas, elastif=elastif, xlim1=xlim1, xk=xk,
        fun_a1=fun_a1, fun_b1=fun_b1, fun_c1=fun_c1, fun_d1=fun_d1,
        damg=damg, fdelay=fdelay, xlim2=xlim2, rload=rload, iload=int(rload),
        fscal=fscal, title=title,
    )
    model.prop_type35s[prop_id] = p35
    model.properties[prop_id] = Property(
        id=prop_id, type=35, title=title,
        params={
            "mass": amas, "amas": amas, "elastif": elastif, "stiff": elastif, "k": elastif,
            "xlim1": xlim1, "x_lim1": xlim1, "xlim2": xlim2, "x_lim2": xlim2,
            "xk": xk, "k_post": xk,
            "fun_a1": fun_a1, "fun_b1": fun_b1, "fun_c1": fun_c1, "fun_d1": fun_d1,
            "damg": damg, "d1": damg, "fdelay": fdelay, "d2": fdelay,
            "rload": rload, "iload": int(rload), "fscal": fscal,
            # M149 compatibility aliases
            "k_tens": k_tens, "k_comp": k_comp, "k_shear": k_shear,
            "f_tens": f_tens, "f_shear": f_shear,
            "skew_id": skew_id, "iflag": iflag, "ipen": ipen, "ifail": ifail,
            "dist_max": dist_max, "area": area,
        }
    )




def read_prop_type45(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE45`` or ``/PROP/KJOINT2`` / ``/PROP/KINEMATIC_JOINT2`` (M187): 6-DOF Kinematic Joint Type 2."""
    from ...model.entities import PropType45, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE45/{prop_id}: missing data card", block.source)
        return

    joint_type = 1
    kn, scale, cr = 0.0, 1.0, 0.0
    isensor = 0
    skew1, skew2 = 0, 0
    ktx, kty, ktz = 0.0, 0.0, 0.0
    xt_fun, yt_fun, zt_fun = 0, 0, 0
    xn, yn, zn, xc, yc, zc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ctx, cty, ctz = 0.0, 0.0, 0.0
    ctx_fun, cty_fun, ctz_fun = 0, 0, 0
    vx, vy, vz = 0.0, 0.0, 0.0
    fx, fy, fz = 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_KJOINT2_1")
            joint_type = _safe_int(c0[0]) if len(c0) > 0 else 1
            kn = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            scale = _safe_float(c0[2]) if len(c0) > 2 and c0[2].strip() else 1.0
            cr = _safe_float(c0[3]) if len(c0) > 3 else 0.0
            isensor = _safe_int(c0[4]) if len(c0) > 4 else 0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_KJOINT2_2")
            skew1 = _safe_int(c1[0]) if len(c1) > 0 else 0
            skew2 = _safe_int(c1[1]) if len(c1) > 1 else 0
            ktx = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            kty = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            ktz = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_KJOINT2_3")
            xt_fun = _safe_int(c2[0]) if len(c2) > 0 else 0
            yt_fun = _safe_int(c2[1]) if len(c2) > 1 else 0
            zt_fun = _safe_int(c2[2]) if len(c2) > 2 else 0
            xn = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            yn = _safe_float(c2[4]) if len(c2) > 4 else 0.0
            zn = _safe_float(c2[5]) if len(c2) > 5 else 0.0
            xc = _safe_float(c2[6]) if len(c2) > 6 else 0.0
            yc = _safe_float(c2[7]) if len(c2) > 7 else 0.0
            zc = _safe_float(c2[8]) if len(c2) > 8 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_KJOINT2_4")
            ctx = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            cty = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            ctz = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            ctx_fun = _safe_int(c3[3]) if len(c3) > 3 else 0
            cty_fun = _safe_int(c3[4]) if len(c3) > 4 else 0
            ctz_fun = _safe_int(c3[5]) if len(c3) > 5 else 0
            vx = _safe_float(c3[6]) if len(c3) > 6 else 0.0
            vy = _safe_float(c3[7]) if len(c3) > 7 else 0.0
            vz = _safe_float(c3[8]) if len(c3) > 8 else 0.0
            fx = _safe_float(c3[9]) if len(c3) > 9 else 0.0
            fy = _safe_float(c3[10]) if len(c3) > 10 else 0.0
            fz = _safe_float(c3[11]) if len(c3) > 11 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            joint_type = _safe_int(t0[0]) if len(t0) > 0 else 1
            kn = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            scale = _safe_float(t0[2]) if len(t0) > 2 else 1.0
            cr = _safe_float(t0[3]) if len(t0) > 3 else 0.0
            isensor = _safe_int(t0[4]) if len(t0) > 4 else 0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            skew1 = _safe_int(t1[0]) if len(t1) > 0 else 0
            skew2 = _safe_int(t1[1]) if len(t1) > 1 else 0
            ktx = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            kty = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            ktz = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            xt_fun = _safe_int(t2[0]) if len(t2) > 0 else 0
            yt_fun = _safe_int(t2[1]) if len(t2) > 1 else 0
            zt_fun = _safe_int(t2[2]) if len(t2) > 2 else 0
            xn = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            yn = _safe_float(t2[4]) if len(t2) > 4 else 0.0
            zn = _safe_float(t2[5]) if len(t2) > 5 else 0.0
            xc = _safe_float(t2[6]) if len(t2) > 6 else 0.0
            yc = _safe_float(t2[7]) if len(t2) > 7 else 0.0
            zc = _safe_float(t2[8]) if len(t2) > 8 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            ctx = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            cty = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            ctz = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            ctx_fun = _safe_int(t3[3]) if len(t3) > 3 else 0
            cty_fun = _safe_int(t3[4]) if len(t3) > 4 else 0
            ctz_fun = _safe_int(t3[5]) if len(t3) > 5 else 0
            vx = _safe_float(t3[6]) if len(t3) > 6 else 0.0
            vy = _safe_float(t3[7]) if len(t3) > 7 else 0.0
            vz = _safe_float(t3[8]) if len(t3) > 8 else 0.0
            fx = _safe_float(t3[9]) if len(t3) > 9 else 0.0
            fy = _safe_float(t3[10]) if len(t3) > 10 else 0.0
            fz = _safe_float(t3[11]) if len(t3) > 11 else 0.0

    p45 = PropType45(
        id=prop_id, joint_type=joint_type, kn=kn, scale=scale, cr=cr, isensor=isensor,
        skew1=skew1, skew2=skew2, ktx=ktx, kty=kty, ktz=ktz,
        xt_fun=xt_fun, yt_fun=yt_fun, zt_fun=zt_fun,
        xn=xn, yn=yn, zn=zn, xc=xc, yc=yc, zc=zc,
        ctx=ctx, cty=cty, ctz=ctz, ctx_fun=ctx_fun, cty_fun=cty_fun, ctz_fun=ctz_fun,
        vx=vx, vy=vy, vz=vz, fx=fx, fy=fy, fz=fz, title=title
    )
    model.prop_type45s[prop_id] = p45
    model.properties[prop_id] = Property(
        id=prop_id, type=45, title=title,
        params={
            "joint_type": joint_type, "type": joint_type, "kn": kn, "scale": scale, "cr": cr,
            "isensor": isensor, "skew1": skew1, "skew2": skew2, "ktx": ktx, "kty": kty, "ktz": ktz,
            "xt_fun": xt_fun, "yt_fun": yt_fun, "zt_fun": zt_fun,
            "xn": xn, "yn": yn, "zn": zn, "xc": xc, "yc": yc, "zc": zc,
            "ctx": ctx, "cty": cty, "ctz": ctz, "ctx_fun": ctx_fun, "cty_fun": cty_fun, "ctz_fun": ctz_fun,
            "vx": vx, "vy": vy, "vz": vz, "fx": fx, "fy": fy, "fz": fz
        }
    )




def read_prop_type36(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE36`` or ``/PROP/PREDIT`` / ``/PROP/DELAMINATION`` (M187): Progressive delamination property."""
    from ...model.entities import PropType36, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE36/{prop_id}: missing data card", block.source)
        return

    lutype = 1
    skew_csid, prop_id1, prop_id2 = 0, 0, 0
    xk = 0.0
    mat_id = 0
    area, ixx, iyy, izz, ray = 0.0, 0.0, 0.0, 0.0, 0.0
    itype_val, itype_sub = 0, 0
    fct_id1, fct_id2, fct_id3 = 0, 0, 0
    k_init = 0.0
    p1, p2, p3, p4, p5 = 0.0, 0.0, 0.0, 0.0, 0.0

    # Determine layout: standard 3-card CFG format (Card 1: Iutype) or 2-card composite format
    t0 = valid_cards[0].tokens()
    if len(t0) == 1 or (block.fixed and len(valid_cards[0].raw.strip()) <= 10):
        # Standard OpenRadioss 3-card layout (hm_read_prop36.F, prop_p36_predit.cfg)
        lutype = _safe_int(t0[0]) if len(t0) > 0 else 1
        if lutype == 0:
            lutype = 1
        if lutype == 1:
            if len(valid_cards) > 1:
                if block.fixed:
                    c1 = valid_cards[1].cut("PROP_PREDIT_2A")
                    skew_csid = _safe_int(c1[0]) if len(c1) > 0 else 0
                    prop_id1 = _safe_int(c1[1]) if len(c1) > 1 else 0
                    prop_id2 = _safe_int(c1[2]) if len(c1) > 2 else 0
                else:
                    t1 = valid_cards[1].tokens()
                    skew_csid = _safe_int(t1[0]) if len(t1) > 0 else 0
                    prop_id1 = _safe_int(t1[1]) if len(t1) > 1 else 0
                    prop_id2 = _safe_int(t1[2]) if len(t1) > 2 else 0
            if len(valid_cards) > 2:
                if block.fixed:
                    c2 = valid_cards[2].cut("PROP_PREDIT_3A")
                    xk = _safe_float(c2[0]) if len(c2) > 0 else 0.0
                else:
                    t2 = valid_cards[2].tokens()
                    xk = _safe_float(t2[0]) if len(t2) > 0 else 0.0
        elif lutype == 2:
            if len(valid_cards) > 1:
                if block.fixed:
                    c1 = valid_cards[1].cut("PROP_PREDIT_2B")
                    mat_id = _safe_int(c1[0]) if len(c1) > 0 else 0
                else:
                    t1 = valid_cards[1].tokens()
                    mat_id = _safe_int(t1[0]) if len(t1) > 0 else 0
            if len(valid_cards) > 2:
                if block.fixed:
                    c2 = valid_cards[2].cut("PROP_PREDIT_3B")
                    area = _safe_float(c2[0]) if len(c2) > 0 else 0.0
                    ixx = _safe_float(c2[1]) if len(c2) > 1 else 0.0
                    iyy = _safe_float(c2[2]) if len(c2) > 2 else 0.0
                    izz = _safe_float(c2[3]) if len(c2) > 3 else 0.0
                    ray = _safe_float(c2[4]) if len(c2) > 4 else 0.0
                else:
                    t2 = valid_cards[2].tokens()
                    area = _safe_float(t2[0]) if len(t2) > 0 else 0.0
                    ixx = _safe_float(t2[1]) if len(t2) > 1 else 0.0
                    iyy = _safe_float(t2[2]) if len(t2) > 2 else 0.0
                    izz = _safe_float(t2[3]) if len(t2) > 3 else 0.0
                    ray = _safe_float(t2[4]) if len(t2) > 4 else 0.0
    elif block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_PREDIT_DELAM_1")
            lutype = _safe_int(c0[0]) if len(c0) > 0 else 1
            skew_csid = _safe_int(c0[1]) if len(c0) > 1 else 0
            prop_id1 = _safe_int(c0[2]) if len(c0) > 2 else 0
            prop_id2 = _safe_int(c0[3]) if len(c0) > 3 else 0
            xk = _safe_float(c0[4]) if len(c0) > 4 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_PREDIT_DELAM_2")
            mat_id = _safe_int(c1[0]) if len(c1) > 0 else 0
            area = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            ixx = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            iyy = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            izz = _safe_float(c1[4]) if len(c1) > 4 else 0.0
            ray = _safe_float(c1[5]) if len(c1) > 5 else 0.0
    else:
        # Free-format multi-token card 1
        lutype = _safe_int(t0[0]) if len(t0) > 0 else 1
        skew_csid = _safe_int(t0[1]) if len(t0) > 1 else 0
        prop_id1 = _safe_int(t0[2]) if len(t0) > 2 else 0
        prop_id2 = _safe_int(t0[3]) if len(t0) > 3 else 0
        xk = _safe_float(t0[4]) if len(t0) > 4 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            mat_id = _safe_int(t1[0]) if len(t1) > 0 else 0
            area = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            ixx = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            iyy = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            izz = _safe_float(t1[4]) if len(t1) > 4 else 0.0
            ray = _safe_float(t1[5]) if len(t1) > 5 else 0.0

    p36 = PropType36(
        id=prop_id, lutype=lutype, skew_csid=skew_csid, prop_id1=prop_id1, prop_id2=prop_id2,
        xk=xk, mat_id=mat_id, area=area, ixx=ixx, iyy=iyy, izz=izz, ray=ray, title=title
    )
    model.prop_type36s[prop_id] = p36
    model.properties[prop_id] = Property(
        id=prop_id, type=36, title=title,
        params={
            "lutype": lutype, "skew_csid": skew_csid, "skew_id": skew_csid,
            "prop_id1": prop_id1, "prop_id2": prop_id2,
            "xk": xk, "mat_id": mat_id, "area": area, "ixx": ixx, "iyy": iyy, "izz": izz, "ray": ray,
            # Compatibility aliases
            "itype": lutype, "itype_val": lutype, "itype_sub": itype_sub,
            "fct_id1": fct_id1, "fct_id2": fct_id2, "fct_id3": fct_id3, "k_init": k_init,
            "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
        }
    )




def read_prop_type9(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE9`` or ``/PROP/SH_ORTH`` (M188): Orthotropic shell property."""
    from ...model.entities import PropType9, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE9/{prop_id}: missing data card", block.source)
        return

    ishell, ismstr, ish3n, idrill = 0, 0, 0, 0
    hm, hf, hr, dm, dn = 0.0, 0.0, 0.0, 0.0, 0.0
    nip, istrain = 1, 0
    thick = 0.0
    ashear = 0.833333
    ithick, iplas = 0, 0
    skew_id, ip = 0, 0
    vx, vy, vz = 1.0, 0.0, 0.0
    phi = 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE9_1")
            ishell = _safe_int(c0[0]) if len(c0) > 0 else 0
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            ish3n = _safe_int(c0[2]) if len(c0) > 2 else 0
            idrill = _safe_int(c0[3]) if len(c0) > 3 else 0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE9_2")
            hm = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            hf = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            hr = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            dm = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            dn = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE9_3")
            nip = _safe_int(c2[0]) if len(c2) > 0 and c2[0].strip() else 1
            istrain = _safe_int(c2[1]) if len(c2) > 1 else 0
            thick = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            ashear = _safe_float(c2[3]) if len(c2) > 3 and c2[3].strip() else 0.833333
            if len(c2) > 4 and c2[4].strip():
                skew_id = _safe_int(c2[4])
            if len(c2) > 5 and c2[5].strip():
                ip = _safe_int(c2[5])
            ithick = _safe_int(c2[5]) if len(c2) > 5 else 0
            iplas = _safe_int(c2[6]) if len(c2) > 6 else 0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_TYPE9_4")
            vx = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 1.0
            vy = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            vz = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            phi = _safe_float(c3[3]) if len(c3) > 3 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            ishell = _safe_int(t0[0]) if len(t0) > 0 else 0
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            ish3n = _safe_int(t0[2]) if len(t0) > 2 else 0
            idrill = _safe_int(t0[3]) if len(t0) > 3 else 0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            hm = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            hf = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            hr = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            dm = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            dn = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            nip = _safe_int(t2[0]) if len(t2) > 0 else 1
            istrain = _safe_int(t2[1]) if len(t2) > 1 else 0
            thick = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            ashear = _safe_float(t2[3]) if len(t2) > 3 else 0.833333
            if len(t2) > 4:
                skew_id = _safe_int(t2[4])
            if len(t2) > 5:
                ip = _safe_int(t2[5])
            ithick = _safe_int(t2[4]) if len(t2) > 4 else 0
            iplas = _safe_int(t2[5]) if len(t2) > 5 else 0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            vx = _safe_float(t3[0]) if len(t3) > 0 else 1.0
            vy = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            vz = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            phi = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            if len(t3) > 4 and not skew_id:
                skew_id = _safe_int(t3[4])
            if len(t3) > 5:
                ip = _safe_int(t3[5])

    p9 = PropType9(
        id=prop_id, ishell=ishell, ismstr=ismstr, ish3n=ish3n, idrill=idrill,
        hm=hm, hf=hf, hr=hr, dm=dm, dn=dn, nip=nip, istrain=istrain,
        thick=thick, ashear=ashear, ithick=ithick, iplas=iplas,
        vx=vx, vy=vy, vz=vz, phi=phi, title=title
    )
    model.prop_type9s[prop_id] = p9
    model.properties[prop_id] = Property(
        id=prop_id, type=9, title=title,
        params={
            "ishell": ishell, "ismstr": ismstr, "ish3n": ish3n, "idrill": idrill,
            "hm": hm, "hf": hf, "hr": hr, "dm": dm, "dn": dn,
            "nip": nip, "istrain": istrain, "thick": thick, "ashear": ashear,
            "ithick": ithick, "iplas": iplas, "vx": vx, "vy": vy, "vz": vz,
            "phi": phi, "mat_beta": phi, "skew_id": skew_id, "ip": ip
        }
    )




def read_prop_type10(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE10`` or ``/PROP/SH_COMP`` (M188): Multi-layer composite shell property."""
    from ...model.entities import PropType10, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE10/{prop_id}: missing data card", block.source)
        return

    ishell, ismstr, ish3n, idrill = 0, 0, 0, 0
    p_thick_fail = 0.0
    hm, hf, hr, dm, dn = 0.01, 0.01, 0.01, 0.0, 0.0
    nip, istrain = 1, 0
    thick = 1.0
    ashear = 0.833333
    ithick, iplas = 0, 0
    vx, vy, vz = 1.0, 0.0, 0.0
    skew_id, ip = 0, 0
    phis: list[float] = []

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_SH_COMP_FLAGS") if len(valid_cards[0].raw.rstrip()) > 40 else valid_cards[0].cut("PROP_TYPE10_1")
            ishell = _safe_int(c0[0]) if len(c0) > 0 else 0
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            ish3n = _safe_int(c0[2]) if len(c0) > 2 else 0
            idrill = _safe_int(c0[3]) if len(c0) > 3 else 0
            if len(c0) > 5 and c0[5].strip():
                p_thick_fail = _safe_float(c0[5])
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE10_2")
            hm = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 0.01
            hf = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.01
            hr = _safe_float(c1[2]) if len(c1) > 2 and c1[2].strip() else 0.01
            dm = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            dn = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE10_3")
            nip = _safe_int(c2[0]) if len(c2) > 0 and c2[0].strip() else 1
            istrain = _safe_int(c2[1]) if len(c2) > 1 else 0
            thick = _safe_float(c2[2]) if len(c2) > 2 and c2[2].strip() else 1.0
            ashear = _safe_float(c2[3]) if len(c2) > 3 and c2[3].strip() else 0.833333
            ithick = _safe_int(c2[5]) if len(c2) > 5 else 0
            iplas = _safe_int(c2[6]) if len(c2) > 6 else 0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_SH_COMP_VEC") if len(valid_cards[3].raw.rstrip()) > 60 else valid_cards[3].cut("PROP_TYPE10_4")
            vx = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 1.0
            vy = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            vz = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            if len(c3) > 3 and c3[3].strip():
                skew_id = _safe_int(c3[3])
            if len(c3) > 5 and c3[5].strip():
                ip = _safe_int(c3[5])
        for card in valid_cards[4:]:
            for val in card.floats():
                phis.append(val)
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            ishell = _safe_int(t0[0]) if len(t0) > 0 else 0
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            ish3n = _safe_int(t0[2]) if len(t0) > 2 else 0
            idrill = _safe_int(t0[3]) if len(t0) > 3 else 0
            if len(t0) > 4:
                p_thick_fail = _safe_float(t0[4])
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            hm = _safe_float(t1[0]) if len(t1) > 0 else 0.01
            hf = _safe_float(t1[1]) if len(t1) > 1 else 0.01
            hr = _safe_float(t1[2]) if len(t1) > 2 else 0.01
            dm = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            dn = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            nip = _safe_int(t2[0]) if len(t2) > 0 else 1
            istrain = _safe_int(t2[1]) if len(t2) > 1 else 0
            thick = _safe_float(t2[2]) if len(t2) > 2 else 1.0
            ashear = _safe_float(t2[3]) if len(t2) > 3 else 0.833333
            ithick = _safe_int(t2[4]) if len(t2) > 4 else 0
            iplas = _safe_int(t2[5]) if len(t2) > 5 else 0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            vx = _safe_float(t3[0]) if len(t3) > 0 else 1.0
            vy = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            vz = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            if len(t3) > 3:
                skew_id = _safe_int(t3[3])
            if len(t3) > 4:
                ip = _safe_int(t3[4])
        for card in valid_cards[4:]:
            for val in card.floats():
                phis.append(val)

    p10 = PropType10(
        id=prop_id, ishell=ishell, ismstr=ismstr, ish3n=ish3n, idrill=idrill,
        hm=hm, hf=hf, hr=hr, dm=dm, dn=dn, nip=nip, istrain=istrain,
        thick=thick, ashear=ashear, ithick=ithick, iplas=iplas,
        vx=vx, vy=vy, vz=vz, phis=phis, title=title
    )
    model.prop_type10s[prop_id] = p10
    model.properties[prop_id] = Property(
        id=prop_id, type=10, title=title,
        params={
            "ishell": ishell, "ismstr": ismstr, "ish3n": ish3n, "idrill": idrill,
            "p_thick_fail": p_thick_fail, "hm": hm, "hf": hf, "hr": hr, "dm": dm, "dn": dn,
            "nip": nip, "istrain": istrain, "thick": thick, "ashear": ashear,
            "ithick": ithick, "iplas": iplas, "vx": vx, "vy": vy, "vz": vz,
            "skew_id": skew_id, "ip": ip, "phis": phis, "phi_layers": phis
        }
    )




def read_prop_type51(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE51`` or ``/PROP/SH_COH`` (M188): Cohesive shell / composite stack property."""
    from ...model.entities import PropType51, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE51/{prop_id}: missing data card", block.source)
        return

    ishell, ismstr, ish3n, idrill = 0, 0, 0, 0
    pthk, zshift = 0.0, 0.0
    hm, hf, hr, dm, dn = 0.0, 0.0, 0.0, 0.0, 0.0
    ashear = 0.833333
    iint, ithick = 0, 0
    failexp = 0.0
    p_thick_fail = 0.0
    vx, vy, vz = 1.0, 0.0, 0.0
    idsk, iorth, ipos, irp = 0, 0, 0, 0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE51_1")
            ishell = _safe_int(c0[0]) if len(c0) > 0 else 0
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            ish3n = _safe_int(c0[2]) if len(c0) > 2 else 0
            idrill = _safe_int(c0[3]) if len(c0) > 3 else 0
            pthk = _safe_float(c0[4]) if len(c0) > 4 else 0.0
            zshift = _safe_float(c0[5]) if len(c0) > 5 else (_safe_float(c0[4]) if len(c0) > 4 else 0.0)
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE51_2")
            hm = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            hf = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            hr = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            dm = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            dn = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE51_3")
            ashear = _safe_float(c2[0]) if len(c2) > 0 and c2[0].strip() else 0.833333
            iint = _safe_int(c2[1]) if len(c2) > 1 else 0
            ithick = _safe_int(c2[2]) if len(c2) > 2 else 0
            failexp = _safe_float(c2[3]) if len(c2) > 3 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_TYPE51_4")
            vx = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 1.0
            vy = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            vz = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            idsk = _safe_int(c3[3]) if len(c3) > 3 else 0
            iorth = _safe_int(c3[4]) if len(c3) > 4 else 0
            ipos = _safe_int(c3[5]) if len(c3) > 5 else 0
            irp = _safe_int(c3[6]) if len(c3) > 6 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            ishell = _safe_int(t0[0]) if len(t0) > 0 else 0
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            ish3n = _safe_int(t0[2]) if len(t0) > 2 else 0
            idrill = _safe_int(t0[3]) if len(t0) > 3 else 0
            if len(t0) > 5:
                pthk = _safe_float(t0[4])
                zshift = _safe_float(t0[5])
            elif len(t0) > 4:
                zshift = _safe_float(t0[4])
                pthk = zshift
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            hm = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            hf = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            hr = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            dm = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            dn = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            if len(t2) >= 4:
                ashear = _safe_float(t2[0])
                iint = _safe_int(t2[1])
                ithick = _safe_int(t2[2])
                failexp = _safe_float(t2[3])
            elif len(t2) >= 3 and _safe_float(t2[1]) < 1.0 and _safe_int(t2[0]) > 0:
                iint = _safe_int(t2[0])
                ashear = _safe_float(t2[1])
                ithick = _safe_int(t2[2])
            else:
                ashear = _safe_float(t2[0]) if len(t2) > 0 else 0.833333
                iint = _safe_int(t2[1]) if len(t2) > 1 else 0
                ithick = _safe_int(t2[2]) if len(t2) > 2 else 0
                failexp = _safe_float(t2[3]) if len(t2) > 3 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            vx = _safe_float(t3[0]) if len(t3) > 0 else 1.0
            vy = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            vz = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            idsk = _safe_int(t3[3]) if len(t3) > 3 else 0
            iorth = _safe_int(t3[4]) if len(t3) > 4 else 0
            ipos = _safe_int(t3[5]) if len(t3) > 5 else 0
            if len(t3) > 8:
                p_thick_fail = _safe_float(t3[6])
                failexp = _safe_float(t3[7])
                irp = _safe_int(t3[8])
            elif len(t3) > 6:
                irp = _safe_int(t3[6])

    p51 = PropType51(
        id=prop_id, ishell=ishell, ismstr=ismstr, ish3n=ish3n, idrill=idrill,
        pthk=pthk, zshift=zshift, hm=hm, hf=hf, hr=hr, dm=dm, dn=dn,
        ashear=ashear, iint=iint, ithick=ithick, failexp=failexp,
        vx=vx, vy=vy, vz=vz, idsk=idsk, iorth=iorth, ipos=ipos, irp=irp,
        title=title
    )
    model.prop_type51s[prop_id] = p51
    model.props_type51[prop_id] = p51
    model.properties[prop_id] = Property(
        id=prop_id, type=51, title=title,
        params={
            "ishell": ishell, "ismstr": ismstr, "ish3n": ish3n, "idrill": idrill,
            "pthk": pthk, "zshift": zshift, "z0": zshift,
            "hm": hm, "hf": hf, "hr": hr, "dm": dm, "dn": dn,
            "ashear": ashear, "iint": iint, "ithick": ithick,
            "failexp": failexp, "fexp": failexp, "p_thick_fail": p_thick_fail,
            "vx": vx, "vy": vy, "vz": vz, "idsk": idsk, "iorth": iorth, "ipos": ipos, "irp": irp
        }
    )




read_prop_p51 = read_prop_type51




def read_prop_pcompp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/PCOMPP/id`` (M201): Ply-based composite property."""
    from ...model.entities import PropPcompp, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    laminate_id = 0
    valid_cards = [c for c in cards if not c.is_blank]
    if valid_cards:
        if block.fixed:
            f = valid_cards[0].cut("PROP_PCOMPP_1")
            laminate_id = _ival(f[0]) if len(f) > 0 else 0
        else:
            toks = valid_cards[0].tokens()
            laminate_id = _safe_int(toks[0]) if toks else 0
    p = PropPcompp(id=prop_id, title=title, laminate_id=laminate_id)
    model.props_pcompp[prop_id] = p
    model.properties[prop_id] = Property(id=prop_id, type=51, title=title, params={"laminate_id": laminate_id})




def read_prop_type5(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE5`` or ``/PROP/RIVET`` (M188): Rivet connection property."""
    from ...model.entities import PropType5, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE5/{prop_id}: missing data card", block.source)
        return

    nforce, tforce, length = 0.0, 0.0, 0.0
    mass, stiffness, fn_fail, ft_fail = 0.0, 0.0, 0.0, 0.0
    wflag = 0
    imod = 1

    if block.fixed:
        if len(valid_cards) >= 2:
            c0 = valid_cards[0].tokens()
            wflag = _safe_int(c0[0]) if len(c0) > 0 else 0
            imod = _safe_int(c0[1]) if len(c0) > 1 else 1
            c1 = valid_cards[1].tokens()
            nforce = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            tforce = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            length = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            fn_fail, ft_fail = nforce, tforce
        else:
            c0 = valid_cards[0].cut("PROP_TYPE5_1")
            nforce = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            tforce = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            length = _safe_float(c0[2]) if len(c0) > 2 else 0.0
            wflag = _safe_int(c0[3]) if len(c0) > 3 else 0
            imod = _safe_int(c0[4]) if len(c0) > 4 and c0[4].strip() else 1
            fn_fail, ft_fail = nforce, tforce
    else:
        if len(valid_cards) >= 2:
            t0 = valid_cards[0].tokens()
            t1 = valid_cards[1].tokens()
            wflag = _safe_int(t0[0]) if len(t0) > 0 else 0
            imod = _safe_int(t0[1]) if len(t0) > 1 else 1
            nforce = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            tforce = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            length = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            fn_fail, ft_fail = nforce, tforce
        else:
            t0 = valid_cards[0].tokens()
            if len(t0) == 4:
                mass = _safe_float(t0[0])
                stiffness = _safe_float(t0[1])
                fn_fail = _safe_float(t0[2])
                ft_fail = _safe_float(t0[3])
                nforce, tforce = fn_fail, ft_fail
            elif len(t0) >= 5:
                nforce = _safe_float(t0[0])
                tforce = _safe_float(t0[1])
                length = _safe_float(t0[2])
                wflag = _safe_int(t0[3])
                imod = _safe_int(t0[4])
                fn_fail, ft_fail = nforce, tforce
            else:
                nforce = _safe_float(t0[0]) if len(t0) > 0 else 0.0
                tforce = _safe_float(t0[1]) if len(t0) > 1 else 0.0
                length = _safe_float(t0[2]) if len(t0) > 2 else 0.0
                fn_fail, ft_fail = nforce, tforce

    p5 = PropType5(
        id=prop_id, mass=mass, stiffness=stiffness, fn_fail=fn_fail, ft_fail=ft_fail,
        nforce=nforce, tforce=tforce, length=length,
        wflag=wflag, imod=imod, title=title
    )
    model.prop_type5s[prop_id] = p5
    model.prop_rivets[prop_id] = p5
    model.properties[prop_id] = Property(
        id=prop_id, type=5, title=title,
        params={
            "mass": mass, "stiffness": stiffness, "fn_fail": fn_fail, "ft_fail": ft_fail,
            "fn": nforce, "ft": tforce, "dx": length, "length": length,
            "nforce": nforce, "tforce": tforce,
            "wflag": wflag, "imod": imod
        }
    )




def read_prop_type6(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE6`` or ``/PROP/SOL_ORTH`` (M188): Orthotropic solid property."""
    from ...model.entities import PropType6, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE6/{prop_id}: missing data card", block.source)
        return

    isolid, ismstr, icpre, itetra10, nbp, itetra4, iframe = 14, 0, 0, 0, 0, 0, 0
    dn = 0.0
    qa, qb, h = 1.1, 0.05, 0.1
    vx, vy, vz = 0.0, 0.0, 0.0
    skew_id, ip, iorth = 0, 0, 0
    phi, px, py, pz = 0.0, 0.0, 0.0, 0.0
    mat_beta, deltat_min = 0.0, 0.0
    vdef_min, vdef_max, asp_max, col_min = 0.0, 0.0, 0.0, 0.0
    ndir, sphpart_id, istrain, ihkt = 0, 0, 0, 0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_SOL_ORTH_1")
            isolid = _safe_int(c0[0]) if len(c0) > 0 and c0[0].strip() else 14
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            icpre = _safe_int(c0[3]) if len(c0) > 3 and c0[3].strip() else (_safe_int(c0[2]) if len(c0) > 2 else 0)
            itetra10 = _safe_int(c0[4]) if len(c0) > 4 else 0
            nbp = _safe_int(c0[5]) if len(c0) > 5 and c0[5].strip() else (_safe_int(c0[3]) if len(c0) > 3 else 0)
            itetra4 = _safe_int(c0[6]) if len(c0) > 6 else 0
            iframe = _safe_int(c0[7]) if len(c0) > 7 and c0[7].strip() else (_safe_int(c0[4]) if len(c0) > 4 else 0)
            dn = _safe_float(c0[8]) if len(c0) > 8 and c0[8].strip() else (_safe_float(c0[5]) if len(c0) > 5 else 0.0)
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_SOL_ORTH_2")
            qa = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 1.1
            qb = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.05
            h = _safe_float(c1[2]) if len(c1) > 2 and c1[2].strip() else 0.1
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_SOL_ORTH_3")
            vx = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            vy = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            vz = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            skew_id = _safe_int(c2[3]) if len(c2) > 3 else 0
            ip = _safe_int(c2[4]) if len(c2) > 4 else 0
            iorth = _safe_int(c2[5]) if len(c2) > 5 else 0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_SOL_ORTH_4")
            phi = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            mat_beta = phi
            px = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            py = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            pz = _safe_float(c3[3]) if len(c3) > 3 else 0.0
        if len(valid_cards) > 4:
            toks4 = valid_cards[4].tokens()
            if len(toks4) <= 3 and len(valid_cards[4].raw.rstrip()) <= 50:
                c4 = valid_cards[4].cut("PROP_SOL_ORTH_DT")
                deltat_min = _safe_float(c4[0]) if len(c4) > 0 else 0.0
                istrain = _safe_int(c4[1]) if len(c4) > 1 else 0
                ihkt = _safe_int(c4[2]) if len(c4) > 2 else 0
            else:
                c4 = valid_cards[4].cut("PROP_SOL_ORTH_5")
                deltat_min = _safe_float(c4[0]) if len(c4) > 0 else 0.0
                vdef_min = _safe_float(c4[1]) if len(c4) > 1 else 0.0
                vdef_max = _safe_float(c4[2]) if len(c4) > 2 else 0.0
                asp_max = _safe_float(c4[3]) if len(c4) > 3 else 0.0
                col_min = _safe_float(c4[4]) if len(c4) > 4 else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("PROP_SOL_ORTH_6")
            ndir = _safe_int(c5[0]) if len(c5) > 0 else 0
            sphpart_id = _safe_int(c5[1]) if len(c5) > 1 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            isolid = _safe_int(t0[0]) if len(t0) > 0 else 14
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            if len(t0) <= 6:
                icpre = _safe_int(t0[2]) if len(t0) > 2 else 0
                nbp = _safe_int(t0[3]) if len(t0) > 3 else 0
                iframe = _safe_int(t0[4]) if len(t0) > 4 else 0
                dn = _safe_float(t0[5]) if len(t0) > 5 else 0.0
            else:
                icpre = _safe_int(t0[2]) if len(t0) > 2 else 0
                itetra10 = _safe_int(t0[3]) if len(t0) > 3 else 0
                nbp = _safe_int(t0[4]) if len(t0) > 4 else 0
                itetra4 = _safe_int(t0[5]) if len(t0) > 5 else 0
                iframe = _safe_int(t0[6]) if len(t0) > 6 else 0
                dn = _safe_float(t0[7]) if len(t0) > 7 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            qa = _safe_float(t1[0]) if len(t1) > 0 else 1.1
            qb = _safe_float(t1[1]) if len(t1) > 1 else 0.05
            h = _safe_float(t1[2]) if len(t1) > 2 else 0.1
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            vx = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            vy = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            vz = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            skew_id = _safe_int(t2[3]) if len(t2) > 3 else 0
            ip = _safe_int(t2[4]) if len(t2) > 4 else 0
            iorth = _safe_int(t2[5]) if len(t2) > 5 else 0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            phi = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            mat_beta = phi
            px = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            py = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            pz = _safe_float(t3[3]) if len(t3) > 3 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            deltat_min = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            if len(t4) > 4:
                vdef_min = _safe_float(t4[1])
                vdef_max = _safe_float(t4[2])
                asp_max = _safe_float(t4[3])
                col_min = _safe_float(t4[4])
            elif len(t4) > 1:
                istrain = _safe_int(t4[1])
                ihkt = _safe_int(t4[2]) if len(t4) > 2 else 0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            ndir = _safe_int(t5[0]) if len(t5) > 0 else 0
            sphpart_id = _safe_int(t5[1]) if len(t5) > 1 else 0

    p6 = PropType6(
        id=prop_id, isolid=isolid, ismstr=ismstr, icpre=icpre, nbp=nbp, iframe=iframe,
        inpts_r=nbp // 100 if nbp > 200 else (nbp if nbp > 0 else 1),
        inpts_s=(nbp % 100) // 10 if nbp > 200 else (nbp if nbp > 0 else 1),
        inpts_t=(nbp % 10) if nbp > 200 else (nbp if nbp > 0 else 1),
        dn=dn, qa=qa, qb=qb, h=h, vx=vx, vy=vy, vz=vz,
        skew_id=skew_id, skew_csid=skew_id, refplane=ip, orthtrop=iorth,
        mat_beta=mat_beta, px=px, py=py, pz=pz, deltat_min=deltat_min,
        vdef_min=vdef_min, vdef_max=vdef_max, asp_max=asp_max, col_min=col_min,
        ndir=ndir, sphpart_id=sphpart_id, istrain=istrain, ihkt=ihkt,
        title=title
    )
    model.prop_type6s[prop_id] = p6
    model.properties[prop_id] = Property(
        id=prop_id, type=6, title=title,
        params={
            "isolid": isolid, "ismstr": ismstr, "icpre": icpre, "itetra10": itetra10, "nbp": nbp,
            "inpts_r": p6.inpts_r, "inpts_s": p6.inpts_s, "inpts_t": p6.inpts_t,
            "itetra4": itetra4, "iframe": iframe, "dn": dn, "qa": qa, "qb": qb, "h": h,
            "vx": vx, "vy": vy, "vz": vz, "skew_id": skew_id, "ip": ip, "iorth": iorth,
            "phi": phi, "px": px, "py": py, "pz": pz,
            "deltat_min": deltat_min, "vdef_min": vdef_min, "vdef_max": vdef_max, "asp_max": asp_max, "col_min": col_min,
            "ndir": ndir, "sphpart_id": sphpart_id, "istrain": istrain, "ihkt": ihkt,
            "skew_csid": skew_id, "refplane": ip, "orthtrop": iorth, "mat_beta": mat_beta
        }
    )




def read_prop_type20(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE20`` or ``/PROP/TSHELL`` (M188): Thick shell property."""
    from ...model.entities import PropType20, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE20/{prop_id}: missing data card", block.source)
        return

    isolid, ismstr, icpre, icstr = 15, 0, 0, 0
    nbp = 0
    iint = 1
    dn = 0.1
    qa, qb, h = 1.1, 0.05, 0.1
    thick = 0.0
    nip = 0
    ashear = 0.833333
    deltat_min = 1.0e6

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE20_1")
            isolid = _safe_int(c0[0]) if len(c0) > 0 and c0[0].strip() else 15
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            if len(valid_cards[0].raw.rstrip()) <= 50:
                icpre = _safe_int(c0[2]) if len(c0) > 2 else 0
                dn = _safe_float(c0[3]) if len(c0) > 3 and c0[3].strip() else 0.1
            else:
                icpre = _safe_int(c0[3]) if len(c0) > 3 else 0
                icstr = _safe_int(c0[4]) if len(c0) > 4 else 0
                nbp = _safe_int(c0[5]) if len(c0) > 5 else 0
                iint = _safe_int(c0[6]) if len(c0) > 6 and c0[6].strip() else 1
                dn = _safe_float(c0[7]) if len(c0) > 7 and c0[7].strip() else 0.1
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE20_2")
            qa = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 1.1
            qb = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.05
            h = _safe_float(c1[2]) if len(c1) > 2 and c1[2].strip() else 0.1
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            if len(t2) > 1:
                nip = _safe_int(t2[0])
                thick = _safe_float(t2[2]) if len(t2) > 2 else 0.0
                ashear = _safe_float(t2[3]) if len(t2) > 3 else 0.833333
            else:
                c2 = valid_cards[2].cut("PROP_TYPE20_3")
                deltat_min = _safe_float(c2[0]) if len(c2) > 0 and c2[0].strip() else 1.0e6
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            isolid = _safe_int(t0[0]) if len(t0) > 0 else 15
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            if len(t0) == 4:
                icpre = _safe_int(t0[2])
                dn = _safe_float(t0[3])
            else:
                icpre = _safe_int(t0[2]) if len(t0) > 2 else 0
                icstr = _safe_int(t0[3]) if len(t0) > 3 else 0
                nbp = _safe_int(t0[4]) if len(t0) > 4 else 0
                iint = _safe_int(t0[5]) if len(t0) > 5 else 1
                dn = _safe_float(t0[6]) if len(t0) > 6 else 0.1
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            qa = _safe_float(t1[0]) if len(t1) > 0 else 1.1
            qb = _safe_float(t1[1]) if len(t1) > 1 else 0.05
            h = _safe_float(t1[2]) if len(t1) > 2 else 0.1
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            if len(t2) > 1:
                nip = _safe_int(t2[0])
                thick = _safe_float(t2[2]) if len(t2) > 2 else 0.0
                ashear = _safe_float(t2[3]) if len(t2) > 3 else 0.833333
            else:
                deltat_min = _safe_float(t2[0]) if len(t2) > 0 else 1.0e6

    p20 = PropType20(
        id=prop_id, isolid=isolid, ismstr=ismstr, icpre=icpre, icstr=icstr,
        nbp=nbp,
        inpts_r=nbp // 100 if nbp > 200 else (nbp if nbp > 0 else 2),
        inpts_s=(nbp % 100) // 10 if nbp > 200 else (nbp if nbp > 0 else 2),
        inpts_t=(nbp % 10) if nbp > 200 else (nbp if nbp > 0 else 2),
        iint=iint, dn=dn, qa=qa, qb=qb, h=h, deltat_min=deltat_min,
        title=title
    )
    model.prop_type20s[prop_id] = p20
    model.properties[prop_id] = Property(
        id=prop_id, type=20, title=title,
        params={
            "isolid": isolid, "ismstr": ismstr, "icpre": icpre, "icstr": icstr,
            "nbp": nbp, "inpts_r": p20.inpts_r, "inpts_s": p20.inpts_s, "inpts_t": p20.inpts_t,
            "iint": iint, "dn": dn, "qa": qa, "qb": qb, "h": h,
            "nip": nip if nip else nbp, "thick": thick, "ashear": ashear,
            "deltat_min": deltat_min
        }
    )




def read_prop_type19(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE19`` or ``/PROP/SPR_TORS/prop_ID`` (M205): Torsional spring property."""
    from ...model.entities import PropSpringTors, Property
    prop_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    mass, k, c, fcut = 0.0, 0.0, 0.0, 0.0
    inertia = 0.0
    if valid_cards:
        if block.fixed:
            f = valid_cards[0].cut("PROP_TYPE19_1")
            mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            k = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            c = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        else:
            t = valid_cards[0].tokens()
            if len(t) >= 4:
                mass = float(t[0])
                inertia = float(t[1])
                k = float(t[2])
                c = float(t[3])
            else:
                mass = float(t[0]) if len(t) > 0 else 0.0
                k = float(t[1]) if len(t) > 1 else 0.0
                c = float(t[2]) if len(t) > 2 else 0.0
    fct_id_k = 0
    fct_id_c = 0
    fscale_k = 1.0
    fscale_c = 1.0
    if len(valid_cards) > 1:
        if block.fixed:
            f2 = valid_cards[1].cut("PROP_SPR_TORS_2") if "PROP_SPR_TORS_2" in CARD_LAYOUTS else _fixed_vals(valid_cards[1], [10, 10, 20, 20])
            fct_id_k = _ival(f2[0]) if len(f2) > 0 else 0
            fct_id_c = _ival(f2[1]) if len(f2) > 1 else 0
            fscale_k = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            fscale_c = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
        else:
            t2 = valid_cards[1].tokens()
            fct_id_k = int(float(t2[0])) if len(t2) > 0 else 0
            fct_id_c = int(float(t2[1])) if len(t2) > 1 else 0
            fscale_k = float(t2[2]) if len(t2) > 2 else 1.0
            fscale_c = float(t2[3]) if len(t2) > 3 else 1.0
    prop = PropSpringTors(id=prop_id, title=title, mass=mass, stiffness_k=k, damping_c=c, fcut=fcut, inertia=inertia)
    model.props_type19[prop_id] = prop
    model.prop_type19s[prop_id] = prop
    model.properties[prop_id] = Property(
        id=prop_id, type=19, title=title,
        params={"mass": mass, "inertia": inertia, "stiffness_k": k, "damping_c": c, "fcut": fcut,
                "k_tors": k, "c_tors": c, "k_theta": k, "c_theta": c, "k": k, "c": c,
                "fct_id_k": fct_id_k, "fct_id_c": fct_id_c, "fscale_k": fscale_k, "fscale_c": fscale_c}
    )




def read_prop_type18(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE18`` or ``/PROP/INT_BEAM/prop_ID`` (M593): Integrated fiber beam property."""
    from ...model.entities import PropType18, PropIntBeamIP, Property
    from ...elements.beam_fiber import generate_fiber_section

    prop_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if block.fixed:
        cards = [c for c in cards if not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]
    else:
        cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]

    params = {
        "isflag": 0, "ismstr": 0, "dm": 0.0, "df": 0.0,
        "nip": 1, "iref": 0, "y0": 0.0, "z0": 0.0,
        "fibers": [], "nitrs": 0, "l_params": [0.0] * 6,
        "rot_dofs": (0, 0, 0, 0, 0, 0),
        "area": 1.0, "iyy": 1.0, "izz": 1.0, "ixx": 1.0,
        "zy": 0.0, "zz": 0.0,
        "ishear": 0, "iform": 0,
    }

    is_legacy = False
    if cards:
        toks0 = cards[0].tokens()
        toks1 = cards[1].tokens() if len(cards) > 1 else []
        if len(toks0) >= 3 or len(toks1) >= 4:
            is_legacy = True

    if is_legacy:
        if block.fixed:
            if len(cards) >= 1:
                f = cards[0].cut("PROP_INT_BEAM_1")
                params["ishear"] = _ival(f[0])
                params["iform"] = _ival(f[1]) if len(f) > 1 else 0
                params["nip"] = _ival(f[2]) if len(f) > 2 else 1
            if len(cards) >= 2:
                a = cards[1].cut("PROP_INT_BEAM_2")
                params["area"] = _fval(a[0]) or 1.0
                params["iyy"] = _fval(a[1]) or 1.0
                params["izz"] = _fval(a[2]) or 1.0
                params["ixx"] = _fval(a[3]) or 1.0
            if len(cards) >= 3:
                v = cards[2].cut("PROP_INT_BEAM_3")
                params["vy"] = _fval(v[0])
                params["vz"] = _fval(v[1])
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishear"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["iform"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["nip"] = int(float(t0[2])) if len(t0) > 2 else 1

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["area"] = float(t1[0]) if len(t1) > 0 else 1.0
            params["iyy"] = float(t1[1]) if len(t1) > 1 else 1.0
            params["izz"] = float(t1[2]) if len(t1) > 2 else 1.0
            params["ixx"] = float(t1[3]) if len(t1) > 3 else 1.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["vy"] = float(t2[0]) if len(t2) > 0 else 0.0
            params["vz"] = float(t2[1]) if len(t2) > 1 else 0.0
    else:
        if block.fixed:
            if len(cards) >= 1:
                f0 = cards[0].cut("PROP_INT_BEAM_FLAGS")
                params["isflag"] = _ival(f0[0])
                params["ismstr"] = _ival(f0[1]) if len(f0) > 1 else 0
            if len(cards) >= 2:
                f1 = cards[1].cut("PROP_INT_BEAM_DAMP")
                params["dm"] = _fval(f1[0])
                params["df"] = _fval(f1[1])
            if len(cards) >= 3:
                f2 = cards[2].cut("PROP_INT_BEAM_NIP")
                params["nip"] = _ival(f2[0]) or 1
                params["iref"] = _ival(f2[1]) if len(f2) > 1 else 0
                params["y0"] = _fval(f2[2]) if len(f2) > 2 else 0.0
                params["z0"] = _fval(f2[3]) if len(f2) > 3 else 0.0
            icard = 3
            if params["isflag"] == 0:
                fibers = []
                for k in range(params["nip"]):
                    if icard < len(cards):
                        fib = cards[icard].cut("PROP_INT_BEAM_FIBER")
                        y_ip = _fval(fib[0]) if len(fib) > 0 else 0.0
                        z_ip = _fval(fib[1]) if len(fib) > 1 else 0.0
                        a_ip = _fval(fib[2]) if len(fib) > 2 else 0.0
                        fibers.append((y_ip, z_ip, a_ip))
                    icard += 1
                params["fibers"] = fibers
            else:
                if icard < len(cards):
                    f3 = _fixed_vals(cards[icard], [10, 10, 20, 20, 20, 20])
                    params["nitrs"] = _ival(f3[0])
                    l1 = _fval(f3[2]) if len(f3) > 2 else 0.0
                    l2 = _fval(f3[3]) if len(f3) > 3 else 0.0
                    l3 = _fval(f3[4]) if len(f3) > 4 else 0.0
                    l4 = _fval(f3[5]) if len(f3) > 5 else 0.0
                    icard += 1
                    l5, l6 = 0.0, 0.0
                    if icard < len(cards):
                        f4 = _fixed_vals(cards[icard], [20, 20])
                        l5 = _fval(f4[0]) if len(f4) > 0 else 0.0
                        l6 = _fval(f4[1]) if len(f4) > 1 else 0.0
                        icard += 1
                    params["l_params"] = [l1, l2, l3, l4, l5, l6]
            if icard < len(cards):
                rw = _fixed_vals(cards[icard], [3, 1, 1, 1, 1, 1, 1, 1])
                params["rot_dofs"] = tuple(_ival(rw[i]) for i in range(1, 7) if i < len(rw))
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["isflag"] = _ival(t0[0]) if len(t0) > 0 else 0
            params["ismstr"] = _ival(t0[1]) if len(t0) > 1 else 0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["dm"] = _fval(t1[0]) if len(t1) > 0 else 0.0
            params["df"] = _fval(t1[1]) if len(t1) > 1 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = _ival(t2[0]) if len(t2) > 0 else 1
            params["iref"] = _ival(t2[1]) if len(t2) > 1 else 0
            params["y0"] = _fval(t2[2]) if len(t2) > 2 else 0.0
            params["z0"] = _fval(t2[3]) if len(t2) > 3 else 0.0

            icard = 3
            if params["isflag"] == 0:
                fibers = []
                for k in range(params["nip"]):
                    if icard < len(cards):
                        toks = cards[icard].tokens()
                        y_ip = _fval(toks[0]) if len(toks) > 0 else 0.0
                        z_ip = _fval(toks[1]) if len(toks) > 1 else 0.0
                        a_ip = _fval(toks[2]) if len(toks) > 2 else 0.0
                        fibers.append((y_ip, z_ip, a_ip))
                    icard += 1
                params["fibers"] = fibers
            else:
                if icard < len(cards):
                    toks = cards[icard].tokens()
                    params["nitrs"] = _ival(toks[0]) if len(toks) > 0 else 0
                    if len(toks) >= 6:
                        params["iref"] = _ival(toks[1])
                        l1_4 = [_fval(x) for x in toks[2:6]]
                    else:
                        l1_4 = [_fval(x) for x in toks[1:5]]
                    while len(l1_4) < 4:
                        l1_4.append(0.0)
                    icard += 1
                    l5_6 = [0.0, 0.0]
                    if icard < len(cards):
                        toks2 = cards[icard].tokens()
                        l5_6 = [_fval(x) for x in toks2[:2]]
                        while len(l5_6) < 2:
                            l5_6.append(0.0)
                        icard += 1
                    params["l_params"] = l1_4 + l5_6
            if icard < len(cards):
                toks = cards[icard].tokens()
                if len(toks) >= 6:
                    params["rot_dofs"] = tuple(_ival(x) for x in toks[:6])

    ips = [PropIntBeamIP(y=fib[0], z=fib[1], area=fib[2]) for fib in params.get("fibers", [])]
    rot = params.get("rot_dofs", (0, 0, 0, 0, 0, 0))
    lp = params.get("l_params", [0.0] * 6)
    while len(lp) < 6:
        lp.append(0.0)

    try:
        y_pts, z_pts, a_pts, sec_props = generate_fiber_section(
            isflag=params.get("isflag", 0),
            nitrs=params.get("nitrs", 0),
            l_params=lp,
            nip=len(params.get("fibers", [])),
            user_fibers=params.get("fibers", []),
            iref=params.get("iref", 0),
            y0=params.get("y0", 0.0),
            z0=params.get("z0", 0.0),
        )
        if not is_legacy:
            params["area"] = sec_props["area"]
            params["iyy"] = sec_props["iyy"]
            params["izz"] = sec_props["izz"]
            params["ixx"] = sec_props["ixx"]
            params["zy"] = sec_props["zy"]
            params["zz"] = sec_props["zz"]
            if not ips and len(y_pts) > 0:
                ips = [PropIntBeamIP(y=y_pts[i], z=z_pts[i], area=a_pts[i]) for i in range(len(y_pts))]
    except Exception:
        pass

    p18 = PropType18(
        id=prop_id,
        isflag=params.get("isflag", 0),
        ismstr=params.get("ismstr", 0),
        dm=params.get("dm", 0.0),
        df=params.get("df", 0.0),
        nip=params.get("nip", len(ips) if ips else 1),
        iref=params.get("iref", 0),
        y0=params.get("y0", 0.0),
        z0=params.get("z0", 0.0),
        ips=ips,
        nitrs=params.get("nitrs", 0),
        l1=lp[0], l2=lp[1], l3=lp[2], l4=lp[3], l5=lp[4], l6=lp[5],
        wx1=rot[0] if len(rot) > 0 else 0,
        wy1=rot[1] if len(rot) > 1 else 0,
        wz1=rot[2] if len(rot) > 2 else 0,
        wx2=rot[3] if len(rot) > 3 else 0,
        wy2=rot[4] if len(rot) > 4 else 0,
        wz2=rot[5] if len(rot) > 5 else 0,
        title=title,
        area=params.get("area", 1.0),
        iyy=params.get("iyy", 1.0),
        izz=params.get("izz", 1.0),
        ixx=params.get("ixx", 1.0),
        zy=params.get("zy", 0.0),
        zz=params.get("zz", 0.0),
        ishear=params.get("ishear", 0),
        iform=params.get("iform", 0),
        params=params,
    )
    model.prop_int_beams[prop_id] = p18
    model.prop_type18s[prop_id] = p18
    model.properties[prop_id] = Property(
        id=prop_id,
        type=18,
        title=title,
        params=params,
    )




def read_prop_spr_bend(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/SPR_BEND/prop_ID`` (M205): Bending spring property."""
    from ...model.entities import PropSpringBend, Property
    prop_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    mass, k, c, fcut = 0.0, 0.0, 0.0, 0.0
    if valid_cards:
        if block.fixed:
            f = valid_cards[0].cut("PROP_SPR_BEND_1")
            mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            k = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            c = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        else:
            t = valid_cards[0].tokens()
            mass = float(t[0]) if len(t) > 0 else 0.0
            k = float(t[1]) if len(t) > 1 else 0.0
            c = float(t[2]) if len(t) > 2 else 0.0
    prop = PropSpringBend(id=prop_id, title=title, mass=mass, stiffness_k=k, damping_c=c, fcut=fcut)
    model.props_type20[prop_id] = prop
    model.properties[prop_id] = Property(
        id=prop_id, type=20, title=title,
        params={"mass": mass, "stiffness_k": k, "damping_c": c, "fcut": fcut}
    )




def read_prop_type21(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE21`` or ``/PROP/TSH_ORTH`` (M206): Orthotropic thick shell property."""
    from ...model.entities import PropType21, Property
    prop_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    isolid, ismstr, icstr = 15, 0, 0
    inpts_r, inpts_s, inpts_t = 2, 2, 2
    iint = 1
    dn = 0.0
    qa, qb = 1.1, 0.05
    vx, vy, vz = 0.0, 0.0, 0.0
    skew_id, iorth = 0, 0
    phi, deltat_min = 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE21_1")
            isolid = _safe_int(c0[0]) if len(c0) > 0 and c0[0].strip() else 15
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            icstr = _safe_int(c0[2]) if len(c0) > 2 else 0
            inpts_r = _safe_int(c0[3]) if len(c0) > 3 and c0[3].strip() else 2
            inpts_s = _safe_int(c0[4]) if len(c0) > 4 and c0[4].strip() else 2
            inpts_t = _safe_int(c0[5]) if len(c0) > 5 and c0[5].strip() else 2
            iint = _safe_int(c0[6]) if len(c0) > 6 and c0[6].strip() else 1
            dn = _safe_float(c0[7]) if len(c0) > 7 and c0[7].strip() else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE21_2")
            qa = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 1.1
            qb = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.05
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE21_3")
            vx = _safe_float(c2[0]) if len(c2) > 0 and c2[0].strip() else 0.0
            vy = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            vz = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            skew_id = _safe_int(c2[3]) if len(c2) > 3 else 0
            iorth = _safe_int(c2[4]) if len(c2) > 4 else 0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_TYPE21_4")
            phi = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("PROP_TYPE21_5")
            deltat_min = _safe_float(c4[0]) if len(c4) > 0 and c4[0].strip() else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            isolid = _safe_int(t0[0]) if len(t0) > 0 else 15
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            icstr = _safe_int(t0[2]) if len(t0) > 2 else 0
            inpts_r = _safe_int(t0[3]) if len(t0) > 3 else 2
            inpts_s = _safe_int(t0[4]) if len(t0) > 4 else 2
            inpts_t = _safe_int(t0[5]) if len(t0) > 5 else 2
            iint = _safe_int(t0[6]) if len(t0) > 6 else 1
            dn = _safe_float(t0[7]) if len(t0) > 7 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            qa = _safe_float(t1[0]) if len(t1) > 0 else 1.1
            qb = _safe_float(t1[1]) if len(t1) > 1 else 0.05
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            vx = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            vy = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            vz = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            skew_id = _safe_int(t2[3]) if len(t2) > 3 else 0
            iorth = _safe_int(t2[4]) if len(t2) > 4 else 0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            phi = _safe_float(t3[0]) if len(t3) > 0 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            deltat_min = _safe_float(t4[0]) if len(t4) > 0 else 0.0

    p21 = PropType21(
        id=prop_id, isolid=isolid, ismstr=ismstr, icstr=icstr,
        inpts_r=inpts_r, inpts_s=inpts_s, inpts_t=inpts_t, iint=iint,
        dn=dn, qa=qa, qb=qb, vx=vx, vy=vy, vz=vz, skew_id=skew_id,
        iorth=iorth, phi=phi, deltat_min=deltat_min, title=title
    )
    model.props_type21[prop_id] = p21
    model.properties[prop_id] = Property(id=prop_id, type=21, title=title)




def read_prop_type22(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE22`` or ``/PROP/TSH_COMP`` (M206): Composite layered thick shell property."""
    from ...model.entities import PropType22, PropType22Layer, Property
    prop_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    isolid, ismstr, icstr = 15, 0, 0
    inpts_r, inpts_s, inpts_t = 2, 2, 2
    iint = 1
    dn = 0.0
    qa, qb = 1.1, 0.05
    vx, vy, vz = 0.0, 0.0, 0.0
    skew_id, iorth = 0, 0
    ipos, ashear, deltat_min = 0, 0.0, 0.0
    layers: List[PropType22Layer] = []

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE22_1")
            isolid = _safe_int(c0[0]) if len(c0) > 0 and c0[0].strip() else 15
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            icstr = _safe_int(c0[2]) if len(c0) > 2 else 0
            inpts_r = _safe_int(c0[3]) if len(c0) > 3 and c0[3].strip() else 2
            inpts_s = _safe_int(c0[4]) if len(c0) > 4 and c0[4].strip() else 2
            inpts_t = _safe_int(c0[5]) if len(c0) > 5 and c0[5].strip() else 2
            iint = _safe_int(c0[6]) if len(c0) > 6 and c0[6].strip() else 1
            dn = _safe_float(c0[7]) if len(c0) > 7 and c0[7].strip() else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE22_2")
            qa = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 1.1
            qb = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.05
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE22_3")
            vx = _safe_float(c2[0]) if len(c2) > 0 and c2[0].strip() else 0.0
            vy = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            vz = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            skew_id = _safe_int(c2[3]) if len(c2) > 3 else 0
            iorth = _safe_int(c2[4]) if len(c2) > 4 else 0
            ipos = _safe_int(c2[5]) if len(c2) > 5 else 0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_TYPE22_4")
            ashear = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 0.0
        for card in valid_cards[4:]:
            f = card.cut("PROP_TYPE22_LAYER")
            if len(f) >= 4 and any(x.strip() for x in f):
                layers.append(PropType22Layer(
                    phi=_safe_float(f[0]),
                    thick=_safe_float(f[1]),
                    zi=_safe_float(f[2]),
                    mat_id=_safe_int(f[3])
                ))
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            isolid = _safe_int(t0[0]) if len(t0) > 0 else 15
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            icstr = _safe_int(t0[2]) if len(t0) > 2 else 0
            inpts_r = _safe_int(t0[3]) if len(t0) > 3 else 2
            inpts_s = _safe_int(t0[4]) if len(t0) > 4 else 2
            inpts_t = _safe_int(t0[5]) if len(t0) > 5 else 2
            iint = _safe_int(t0[6]) if len(t0) > 6 else 1
            dn = _safe_float(t0[7]) if len(t0) > 7 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            qa = _safe_float(t1[0]) if len(t1) > 0 else 1.1
            qb = _safe_float(t1[1]) if len(t1) > 1 else 0.05
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            vx = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            vy = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            vz = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            skew_id = _safe_int(t2[3]) if len(t2) > 3 else 0
            iorth = _safe_int(t2[4]) if len(t2) > 4 else 0
            ipos = _safe_int(t2[5]) if len(t2) > 5 else 0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            ashear = _safe_float(t3[0]) if len(t3) > 0 else 0.0
        for card in valid_cards[4:]:
            toks = card.tokens()
            if len(toks) >= 4:
                layers.append(PropType22Layer(
                    phi=float(toks[0]),
                    thick=float(toks[1]),
                    zi=float(toks[2]),
                    mat_id=int(float(toks[3]))
                ))

    p22 = PropType22(
        id=prop_id, isolid=isolid, ismstr=ismstr, icstr=icstr,
        inpts_r=inpts_r, inpts_s=inpts_s, inpts_t=inpts_t, iint=iint,
        dn=dn, qa=qa, qb=qb, vx=vx, vy=vy, vz=vz, skew_id=skew_id,
        iorth=iorth, ipos=ipos, ashear=ashear, layers=layers,
        deltat_min=deltat_min, title=title
    )
    model.props_type22[prop_id] = p22
    model.properties[prop_id] = Property(id=prop_id, type=22, title=title)




def read_prop_type47(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE47`` or ``/PROP/SPR_PULL/prop_ID`` (M207): Tension-only pulling spring property."""
    from ...model.entities import PropSpringPull, PropType13, Property
    prop_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    mass, k, c, fmax, fcut = 0.0, 0.0, 0.0, 0.0, 0.0
    if valid_cards:
        if block.fixed:
            f = valid_cards[0].cut("PROP_TYPE47_1")
            mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            k = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            c = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        else:
            t = valid_cards[0].tokens()
            if len(t) == 2:
                k = float(t[0])
                fmax = float(t[1])
            else:
                mass = float(t[0]) if len(t) > 0 else 0.0
                k = float(t[1]) if len(t) > 1 else 0.0
                c = float(t[2]) if len(t) > 2 else 0.0
                fmax = float(t[3]) if len(t) > 3 else 0.0
    stiff = k if (k != 0.0 and fmax != 0.0) else (mass if mass != 0.0 else k)
    f_max = fmax if fmax != 0.0 else (k if (mass != 0.0 and k != 0.0) else fmax)
    prop = PropSpringPull(id=prop_id, title=title, mass=mass, stiffness_k=k, damping_c=c, fmax=fmax, fcut=fcut)
    p13 = PropType13(id=prop_id, title=title, stiff=stiff, f_max=f_max, params={})
    model.props_type47[prop_id] = prop
    model.props_spr_pull[prop_id] = prop
    model.prop_spr_pulls[prop_id] = p13
    model.properties[prop_id] = Property(
        id=prop_id, type=47, title=title,
        params={"mass": mass, "stiffness_k": k, "damping_c": c, "fmax": fmax, "fcut": fcut, "stiff": stiff, "f_max": f_max}
    )




def read_prop_type48(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE48`` or ``/PROP/SPR_PUSH/prop_ID`` (M207): Compression-only pushing spring property."""
    from ...model.entities import PropSpringPush, Property
    prop_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    mass, k, c, fmax, fcut = 0.0, 0.0, 0.0, 0.0, 0.0
    if valid_cards:
        if block.fixed:
            f = valid_cards[0].cut("PROP_TYPE48_1")
            mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            k = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            c = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        else:
            t = valid_cards[0].tokens()
            mass = float(t[0]) if len(t) > 0 else 0.0
            k = float(t[1]) if len(t) > 1 else 0.0
            c = float(t[2]) if len(t) > 2 else 0.0
            fmax = float(t[3]) if len(t) > 3 else 0.0
    prop = PropSpringPush(id=prop_id, title=title, mass=mass, stiffness_k=k, damping_c=c, fmax=fmax, fcut=fcut)
    model.props_type48[prop_id] = prop
    model.properties[prop_id] = Property(
        id=prop_id, type=48, title=title,
        params={"mass": mass, "stiffness_k": k, "damping_c": c, "fmax": fmax, "fcut": fcut}
    )




def read_prop_type54(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE54`` or ``/PROP/TSH_P54/prop_ID`` (M208): Layered composite thick shell property."""
    from ...model.entities import PropType54, PropType54Layer, Property
    prop_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE54/{prop_id}: missing data card", block.source)
        return

    isolid, ismstr, icstr, inpts_r, inpts_s, inpts_t, iint = 15, 0, 0, 2, 2, 2, 1
    dn, qa, qb, vx, vy, vz, skew_id, iorth, ipos, ashear, deltat_min = 0.0, 1.1, 0.05, 0.0, 0.0, 0.0, 0, 0, 0, 5.0/6.0, 0.0
    layers: List[PropType54Layer] = []

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE54_1")
            isolid = _safe_int(c0[0]) if len(c0) > 0 and c0[0].strip() else 15
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            icstr = _safe_int(c0[2]) if len(c0) > 2 else 0
            inpts_r = _safe_int(c0[3]) if len(c0) > 3 and c0[3].strip() else 2
            inpts_s = _safe_int(c0[4]) if len(c0) > 4 and c0[4].strip() else 2
            inpts_t = _safe_int(c0[5]) if len(c0) > 5 and c0[5].strip() else 2
            iint = _safe_int(c0[6]) if len(c0) > 6 and c0[6].strip() else 1
            dn = _safe_float(c0[7]) if len(c0) > 7 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE54_2")
            qa = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 1.1
            qb = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.05
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE54_3")
            vx = _safe_float(c2[0]) if len(c2) > 0 and c2[0].strip() else 0.0
            vy = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            vz = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            skew_id = _safe_int(c2[3]) if len(c2) > 3 else 0
            iorth = _safe_int(c2[4]) if len(c2) > 4 else 0
            ipos = _safe_int(c2[5]) if len(c2) > 5 else 0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_TYPE54_4")
            ashear = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 5.0/6.0
        for card in valid_cards[4:]:
            f = card.cut("PROP_TYPE54_LAYER")
            if len(f) >= 4 and any(x.strip() for x in f):
                layers.append(PropType54Layer(
                    phi=_safe_float(f[0]),
                    thick=_safe_float(f[1]),
                    zi=_safe_float(f[2]),
                    mat_id=_safe_int(f[3])
                ))
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            isolid = _safe_int(t0[0]) if len(t0) > 0 else 15
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            icstr = _safe_int(t0[2]) if len(t0) > 2 else 0
            inpts_r = _safe_int(t0[3]) if len(t0) > 3 else 2
            inpts_s = _safe_int(t0[4]) if len(t0) > 4 else 2
            inpts_t = _safe_int(t0[5]) if len(t0) > 5 else 2
            iint = _safe_int(t0[6]) if len(t0) > 6 else 1
            dn = _safe_float(t0[7]) if len(t0) > 7 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            qa = _safe_float(t1[0]) if len(t1) > 0 else 1.1
            qb = _safe_float(t1[1]) if len(t1) > 1 else 0.05
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            vx = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            vy = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            vz = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            skew_id = _safe_int(t2[3]) if len(t2) > 3 else 0
            iorth = _safe_int(t2[4]) if len(t2) > 4 else 0
            ipos = _safe_int(t2[5]) if len(t2) > 5 else 0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            ashear = _safe_float(t3[0]) if len(t3) > 0 else 5.0/6.0
        for card in valid_cards[4:]:
            toks = card.tokens()
            if len(toks) >= 4:
                layers.append(PropType54Layer(
                    phi=float(toks[0]),
                    thick=float(toks[1]),
                    zi=float(toks[2]),
                    mat_id=int(float(toks[3]))
                ))

    p54 = PropType54(
        id=prop_id, isolid=isolid, ismstr=ismstr, icstr=icstr,
        inpts_r=inpts_r, inpts_s=inpts_s, inpts_t=inpts_t, iint=iint,
        dn=dn, qa=qa, qb=qb, vx=vx, vy=vy, vz=vz, skew_id=skew_id,
        iorth=iorth, ipos=ipos, ashear=ashear, layers=layers,
        deltat_min=deltat_min, title=title
    )
    model.props_type54[prop_id] = p54
    model.properties[prop_id] = Property(id=prop_id, type=54, title=title)




def read_prop_type14(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE14`` or ``/PROP/SOLID`` (M189): Generalized 3D solid property."""
    from ...model.entities import PropType14, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE14/{prop_id}: missing data card", block.source)
        return

    isolid, ismstr, icpre = 14, 0, 0
    inpts_r, inpts_s, inpts_t = 1, 1, 1
    i_rot, iframe = 0, 0
    dn = 0.0
    qa, qb, h = 1.1, 0.05, 0.1
    deltat_min = 0.0
    istrain = 0
    qa_l, qb_l, h_l = 0.0, 0.0, 0.0
    iplas, icstr = 0, 0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE14_1")
            isolid = _safe_int(c0[0]) if len(c0) > 0 and c0[0].strip() else 14
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            icpre = _safe_int(c0[2]) if len(c0) > 2 else 0
            inpts_r = _safe_int(c0[3]) if len(c0) > 3 and c0[3].strip() else 1
            inpts_s = _safe_int(c0[4]) if len(c0) > 4 and c0[4].strip() else 1
            inpts_t = _safe_int(c0[5]) if len(c0) > 5 and c0[5].strip() else 1
            i_rot = _safe_int(c0[6]) if len(c0) > 6 else 0
            iframe = _safe_int(c0[7]) if len(c0) > 7 else 0
            dn = _safe_float(c0[8]) if len(c0) > 8 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE14_2")
            qa = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 1.1
            qb = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.05
            h = _safe_float(c1[2]) if len(c1) > 2 and c1[2].strip() else 0.1
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE14_3")
            deltat_min = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            istrain = _safe_int(c2[1]) if len(c2) > 1 else 0
            qa_l = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            qb_l = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            h_l = _safe_float(c2[4]) if len(c2) > 4 else 0.0
            iplas = _safe_int(c2[5]) if len(c2) > 5 else 0
            icstr = _safe_int(c2[6]) if len(c2) > 6 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            isolid = _safe_int(t0[0]) if len(t0) > 0 else 14
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            icpre = _safe_int(t0[2]) if len(t0) > 2 else 0
            if len(t0) > 5:
                inpts_r = _safe_int(t0[3])
                inpts_s = _safe_int(t0[4])
                inpts_t = _safe_int(t0[5])
                i_rot = _safe_int(t0[6]) if len(t0) > 6 else 0
                iframe = _safe_int(t0[7]) if len(t0) > 7 else 0
                dn = _safe_float(t0[8]) if len(t0) > 8 else 0.0
            elif len(t0) > 3:
                dn = _safe_float(t0[3])
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            qa = _safe_float(t1[0]) if len(t1) > 0 else 1.1
            qb = _safe_float(t1[1]) if len(t1) > 1 else 0.05
            h = _safe_float(t1[2]) if len(t1) > 2 else 0.1
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            deltat_min = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            istrain = _safe_int(t2[1]) if len(t2) > 1 else 0
            qa_l = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            qb_l = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            h_l = _safe_float(t2[4]) if len(t2) > 4 else 0.0
            iplas = _safe_int(t2[5]) if len(t2) > 5 else 0
            icstr = _safe_int(t2[6]) if len(t2) > 6 else 0

    p14 = PropType14(
        id=prop_id, isolid=isolid, ismstr=ismstr, icpre=icpre,
        inpts_r=inpts_r, inpts_s=inpts_s, inpts_t=inpts_t,
        i_rot=i_rot, iframe=iframe, dn=dn, qa=qa, qb=qb, h=h,
        deltat_min=deltat_min, istrain=istrain, qa_l=qa_l, qb_l=qb_l, h_l=h_l,
        iplas=iplas, icstr=icstr, title=title
    )
    model.prop_type14s[prop_id] = p14
    model.properties[prop_id] = Property(
        id=prop_id, type=14, title=title,
        params={
            "isolid": isolid, "ismstr": ismstr, "icpre": icpre,
            "inpts_r": inpts_r, "inpts_s": inpts_s, "inpts_t": inpts_t,
            "i_rot": i_rot, "iframe": iframe, "dn": dn, "qa": qa, "qb": qb, "h": h,
            "deltat_min": deltat_min, "istrain": istrain, "qa_l": qa_l, "qb_l": qb_l, "h_l": h_l,
            "iplas": iplas, "icstr": icstr
        }
    )




def read_prop_type8(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE8`` or ``/PROP/SPR_GENE`` (M189): Generalized 6-DOF nonlinear spring property."""
    from ...model.entities import PropType8, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE8/{prop_id}: missing data card", block.source)
        return

    mass, inertia = 0.0, 0.0
    skew_id, sensor_id, isflag, ifail, iequil = 0, 0, 0, 0, 0
    dofs = {}
    dof_names = ["tx", "ty", "tz", "rx", "ry", "rz"]

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE8_1")
            mass = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            inertia = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            skew_id = _safe_int(c0[2]) if len(c0) > 2 else 0
            sensor_id = _safe_int(c0[3]) if len(c0) > 3 else 0
            isflag = _safe_int(c0[4]) if len(c0) > 4 else 0
            ifail = _safe_int(c0[5]) if len(c0) > 5 else 0
            iequil = _safe_int(c0[6]) if len(c0) > 6 else 0
        idx = 1
        for name in dof_names:
            if idx >= len(valid_cards):
                break
            ca = valid_cards[idx].cut("PROP_TYPE8_A")
            idx += 1
            cb = valid_cards[idx].cut("PROP_TYPE8_B") if idx < len(valid_cards) else []
            idx += 1
            stiff = _safe_float(ca[0]) if len(ca) > 0 else 0.0
            damp = _safe_float(ca[1]) if len(ca) > 1 else 0.0
            a_coef = _safe_float(ca[2]) if len(ca) > 2 else 0.0
            b_coef = _safe_float(ca[3]) if len(ca) > 3 else 0.0
            d_coef = _safe_float(ca[4]) if len(ca) > 4 else 0.0
            fun_a = _safe_int(cb[0]) if len(cb) > 0 else 0
            hflag = _safe_int(cb[1]) if len(cb) > 1 else 0
            fun_b = _safe_int(cb[2]) if len(cb) > 2 else 0
            fun_c = _safe_int(cb[3]) if len(cb) > 3 else 0
            min_rup = _safe_float(cb[4]) if len(cb) > 4 else 0.0
            max_rup = _safe_float(cb[5]) if len(cb) > 5 else 0.0
            dofs[name] = {
                "stiff": stiff, "damp": damp, "a": a_coef, "b": b_coef, "d": d_coef,
                "fun_a": fun_a, "hflag": hflag, "fun_b": fun_b, "fun_c": fun_c,
                "min_rup": min_rup, "max_rup": max_rup
            }
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            mass = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            inertia = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            skew_id = _safe_int(t0[2]) if len(t0) > 2 else 0
            sensor_id = _safe_int(t0[3]) if len(t0) > 3 else 0
            isflag = _safe_int(t0[4]) if len(t0) > 4 else 0
            ifail = _safe_int(t0[5]) if len(t0) > 5 else 0
            iequil = _safe_int(t0[6]) if len(t0) > 6 else 0
        idx = 1
        for name in dof_names:
            if idx >= len(valid_cards):
                break
            ta = valid_cards[idx].tokens()
            idx += 1
            tb = valid_cards[idx].tokens() if idx < len(valid_cards) else []
            idx += 1
            stiff = _safe_float(ta[0]) if len(ta) > 0 else 0.0
            damp = _safe_float(ta[1]) if len(ta) > 1 else 0.0
            a_coef = _safe_float(ta[2]) if len(ta) > 2 else 0.0
            b_coef = _safe_float(ta[3]) if len(ta) > 3 else 0.0
            d_coef = _safe_float(ta[4]) if len(ta) > 4 else 0.0
            fun_a = _safe_int(tb[0]) if len(tb) > 0 else 0
            hflag = _safe_int(tb[1]) if len(tb) > 1 else 0
            fun_b = _safe_int(tb[2]) if len(tb) > 2 else 0
            fun_c = _safe_int(tb[3]) if len(tb) > 3 else 0
            min_rup = _safe_float(tb[4]) if len(tb) > 4 else 0.0
            max_rup = _safe_float(tb[5]) if len(tb) > 5 else 0.0
            dofs[name] = {
                "stiff": stiff, "damp": damp, "a": a_coef, "b": b_coef, "d": d_coef,
                "fun_a": fun_a, "hflag": hflag, "fun_b": fun_b, "fun_c": fun_c,
                "min_rup": min_rup, "max_rup": max_rup
            }

    p8 = PropType8(
        id=prop_id, mass=mass, inertia=inertia, skew_id=skew_id, sensor_id=sensor_id,
        isflag=isflag, ifail=ifail, iequil=iequil, dofs=dofs, title=title
    )
    model.prop_type8s[prop_id] = p8
    model.properties[prop_id] = Property(
        id=prop_id, type=8, title=title,
        params={
            "mass": mass, "inertia": inertia, "skew_id": skew_id, "sensor_id": sensor_id,
            "isflag": isflag, "ifail": ifail, "iequil": iequil, "dofs": dofs
        }
    )




def read_prop_type25(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE25`` or ``/PROP/SPR_AXI`` (M189): Axisymmetric nonlinear spring property."""
    from ...model.entities import PropType25, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE25/{prop_id}: missing data card", block.source)
        return

    mass, inertia = 0.0, 0.0
    skew_id, sensor_id, isflag, ifail, ileng, ifail2 = 0, 0, 0, 0, 0, 0
    tension = {}
    shear = {}

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE25_1")
            mass = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            inertia = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            skew_id = _safe_int(c0[2]) if len(c0) > 2 else 0
            sensor_id = _safe_int(c0[3]) if len(c0) > 3 else 0
            isflag = _safe_int(c0[4]) if len(c0) > 4 else 0
            ifail = _safe_int(c0[5]) if len(c0) > 5 else 0
            ileng = _safe_int(c0[6]) if len(c0) > 6 else 0
            ifail2 = _safe_int(c0[7]) if len(c0) > 7 else 0
        if len(valid_cards) > 1:
            ca = valid_cards[1].cut("PROP_TYPE25_A")
            cb = valid_cards[2].cut("PROP_TYPE25_B") if len(valid_cards) > 2 else []
            tension = {
                "stiff": _safe_float(ca[0]) if len(ca) > 0 else 0.0,
                "damp": _safe_float(ca[1]) if len(ca) > 1 else 0.0,
                "a": _safe_float(ca[2]) if len(ca) > 2 else 0.0,
                "b": _safe_float(ca[3]) if len(ca) > 3 else 0.0,
                "d": _safe_float(ca[4]) if len(ca) > 4 else 0.0,
                "fun_a": _safe_int(cb[0]) if len(cb) > 0 else 0,
                "hflag": _safe_int(cb[1]) if len(cb) > 1 else 0,
                "fun_b": _safe_int(cb[2]) if len(cb) > 2 else 0,
                "fun_c": _safe_int(cb[3]) if len(cb) > 3 else 0,
                "min_rup": _safe_float(cb[4]) if len(cb) > 4 else 0.0,
                "max_rup": _safe_float(cb[5]) if len(cb) > 5 else 0.0,
            }
        if len(valid_cards) > 3:
            ca = valid_cards[3].cut("PROP_TYPE25_A")
            cb = valid_cards[4].cut("PROP_TYPE25_B") if len(valid_cards) > 4 else []
            shear = {
                "stiff": _safe_float(ca[0]) if len(ca) > 0 else 0.0,
                "damp": _safe_float(ca[1]) if len(ca) > 1 else 0.0,
                "a": _safe_float(ca[2]) if len(ca) > 2 else 0.0,
                "b": _safe_float(ca[3]) if len(ca) > 3 else 0.0,
                "d": _safe_float(ca[4]) if len(ca) > 4 else 0.0,
                "fun_a": _safe_int(cb[0]) if len(cb) > 0 else 0,
                "hflag": _safe_int(cb[1]) if len(cb) > 1 else 0,
                "fun_b": _safe_int(cb[2]) if len(cb) > 2 else 0,
                "fun_c": _safe_int(cb[3]) if len(cb) > 3 else 0,
                "min_rup": _safe_float(cb[4]) if len(cb) > 4 else 0.0,
                "max_rup": _safe_float(cb[5]) if len(cb) > 5 else 0.0,
            }
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            mass = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            inertia = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            skew_id = _safe_int(t0[2]) if len(t0) > 2 else 0
            sensor_id = _safe_int(t0[3]) if len(t0) > 3 else 0
            isflag = _safe_int(t0[4]) if len(t0) > 4 else 0
            ifail = _safe_int(t0[5]) if len(t0) > 5 else 0
            ileng = _safe_int(t0[6]) if len(t0) > 6 else 0
            ifail2 = _safe_int(t0[7]) if len(t0) > 7 else 0
        if len(valid_cards) > 1:
            ta = valid_cards[1].tokens()
            tb = valid_cards[2].tokens() if len(valid_cards) > 2 else []
            tension = {
                "stiff": _safe_float(ta[0]) if len(ta) > 0 else 0.0,
                "damp": _safe_float(ta[1]) if len(ta) > 1 else 0.0,
                "a": _safe_float(ta[2]) if len(ta) > 2 else 0.0,
                "b": _safe_float(ta[3]) if len(ta) > 3 else 0.0,
                "d": _safe_float(ta[4]) if len(ta) > 4 else 0.0,
                "fun_a": _safe_int(tb[0]) if len(tb) > 0 else 0,
                "hflag": _safe_int(tb[1]) if len(tb) > 1 else 0,
                "fun_b": _safe_int(tb[2]) if len(tb) > 2 else 0,
                "fun_c": _safe_int(tb[3]) if len(tb) > 3 else 0,
                "min_rup": _safe_float(tb[4]) if len(tb) > 4 else 0.0,
                "max_rup": _safe_float(tb[5]) if len(tb) > 5 else 0.0,
            }
        if len(valid_cards) > 3:
            ta = valid_cards[3].tokens()
            tb = valid_cards[4].tokens() if len(valid_cards) > 4 else []
            shear = {
                "stiff": _safe_float(ta[0]) if len(ta) > 0 else 0.0,
                "damp": _safe_float(ta[1]) if len(ta) > 1 else 0.0,
                "a": _safe_float(ta[2]) if len(ta) > 2 else 0.0,
                "b": _safe_float(ta[3]) if len(ta) > 3 else 0.0,
                "d": _safe_float(ta[4]) if len(ta) > 4 else 0.0,
                "fun_a": _safe_int(tb[0]) if len(tb) > 0 else 0,
                "hflag": _safe_int(tb[1]) if len(tb) > 1 else 0,
                "fun_b": _safe_int(tb[2]) if len(tb) > 2 else 0,
                "fun_c": _safe_int(tb[3]) if len(tb) > 3 else 0,
                "min_rup": _safe_float(tb[4]) if len(tb) > 4 else 0.0,
                "max_rup": _safe_float(tb[5]) if len(tb) > 5 else 0.0,
            }

    p25 = PropType25(
        id=prop_id, mass=mass, inertia=inertia, skew_id=skew_id, sensor_id=sensor_id,
        isflag=isflag, ifail=ifail, ileng=ileng, ifail2=ifail2,
        tension=tension, shear=shear, title=title
    )
    model.prop_type25s[prop_id] = p25
    model.properties[prop_id] = Property(
        id=prop_id, type=25, title=title,
        params={
            "mass": mass, "inertia": inertia, "skew_id": skew_id, "sensor_id": sensor_id,
            "isflag": isflag, "ifail": ifail, "ileng": ileng, "ifail2": ifail2,
            "tension": tension, "shear": shear
        }
    )




def read_prop_type32(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE32`` or ``/PROP/SPR_PRE`` (M189): Preloaded spring property."""
    from ...model.entities import PropType32, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE32/{prop_id}: missing data card", block.source)
        return

    mass = 0.0
    sensor_id, ilock = 0, 0
    stiff0, f1, d1, e1, stiff1 = 0.0, 0.0, 0.0, 0.0, 0.0
    fun_a1, fun_b1 = 0, 0
    scale_t, scale_d, scale_f = 1.0, 1.0, 1.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE32_1")
            mass = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            sensor_id = _safe_int(c0[2]) if len(c0) > 2 else (_safe_int(c0[1]) if len(c0) > 1 else 0)
            ilock = _safe_int(c0[3]) if len(c0) > 3 else (_safe_int(c0[2]) if len(c0) > 2 else 0)
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE32_2")
            stiff0 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            f1 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            d1 = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            e1 = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            stiff1 = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE32_3")
            fun_a1 = _safe_int(c2[0]) if len(c2) > 0 else 0
            fun_b1 = _safe_int(c2[1]) if len(c2) > 1 else 0
            scale_t = _safe_float(c2[2], 1.0) if len(c2) > 2 and c2[2].strip() else 1.0
            scale_d = _safe_float(c2[3], 1.0) if len(c2) > 3 and c2[3].strip() else 1.0
            scale_f = _safe_float(c2[4], 1.0) if len(c2) > 4 and c2[4].strip() else 1.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            mass = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            sensor_id = _safe_int(t0[1]) if len(t0) > 1 else 0
            ilock = _safe_int(t0[2]) if len(t0) > 2 else 0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            stiff0 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            f1 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            d1 = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            e1 = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            stiff1 = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            fun_a1 = _safe_int(t2[0]) if len(t2) > 0 else 0
            fun_b1 = _safe_int(t2[1]) if len(t2) > 1 else 0
            scale_t = _safe_float(t2[2], 1.0) if len(t2) > 2 else 1.0
            scale_d = _safe_float(t2[3], 1.0) if len(t2) > 3 else 1.0
            scale_f = _safe_float(t2[4], 1.0) if len(t2) > 4 else 1.0

    p32 = PropType32(
        id=prop_id, mass=mass, sensor_id=sensor_id, ilock=ilock,
        stiff0=stiff0, f1=f1, d1=d1, e1=e1, stiff1=stiff1,
        fun_a1=fun_a1, fun_b1=fun_b1,
        scale_t=scale_t, scale_d=scale_d, scale_f=scale_f,
        title=title
    )
    model.prop_type32s[prop_id] = p32
    model.properties[prop_id] = Property(
        id=prop_id, type=32, title=title,
        params={
            "mass": mass, "sensor_id": sensor_id, "sens_id": sensor_id, "ilock": ilock,
            "stiff0": stiff0, "stif0": stiff0, "f1": f1, "d1": d1, "e1": e1,
            "stiff1": stiff1, "stif1": stiff1,
            "fun_a1": fun_a1, "fun_b1": fun_b1, "fct_id1": fun_a1, "fct_id2": fun_b1,
            "scale_t": scale_t, "scale_d": scale_d, "scale_f": scale_f,
            "tscal": scale_t, "dscal": scale_d, "fscal": scale_f
        }
    )




def read_prop_type43(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE43`` or ``/PROP/CONNECT`` (M189): Connector element property."""
    from ...model.entities import PropType43, Property
    from ..prop_reader import _universal_geo_params
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    params = _universal_geo_params()
    ismstr = 1
    thick = 0.0
    if valid_cards:
        if block.fixed:
            t = valid_cards[0].tokens()
            if len(t) > 0:
                ismstr = _safe_int(t[0], 1)
            if len(t) > 1:
                thick = _safe_float(t[1])
            elif len(valid_cards[0].raw) >= 80:
                thick = _safe_float(valid_cards[0].raw[80:].strip())
        else:
            t = valid_cards[0].tokens()
            if len(t) > 0:
                ismstr = _safe_int(t[0], 1)
            if len(t) > 1:
                thick = _safe_float(t[1])
    params["ismstr"] = ismstr
    params["thick"] = thick
    p43 = PropType43(id=prop_id, ismstr=ismstr, thick=thick, title=title)
    model.prop_type43s[prop_id] = p43
    model.properties[prop_id] = Property(id=prop_id, type=43, title=title, params=params)




def read_prop_type34(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE34`` or ``/PROP/SPH`` (M190): SPH particle property."""
    from ...model.entities import PropType34, Property
    from ..prop_reader import _universal_geo_params
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    mass, h0, d0 = 0.0, 0.0, 0.0
    qa, qb, alpha1 = 2.0, 1.0, 0.0
    order = 0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_SPH_1")
            mass = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            h0 = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            d0 = _safe_float(c0[2]) if len(c0) > 2 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_SPH_2")
            qa = _safe_float(c1[0], 2.0) if len(c1) > 0 and c1[0].strip() else 2.0
            qb = _safe_float(c1[1], 1.0) if len(c1) > 1 and c1[1].strip() else 1.0
            alpha1 = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            order = _safe_int(c1[3]) if len(c1) > 3 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            mass = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            h0 = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            d0 = _safe_float(t0[2]) if len(t0) > 2 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            qa = _safe_float(t1[0], 2.0) if len(t1) > 0 else 2.0
            qb = _safe_float(t1[1], 1.0) if len(t1) > 1 else 1.0
            alpha1 = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            order = _safe_int(t1[3]) if len(t1) > 3 else 0

    p34 = PropType34(id=prop_id, mass=mass, h0=h0, d0=d0, qa=qa, qb=qb, alpha1=alpha1, order=order, h=h0, title=title)
    model.prop_type34s[prop_id] = p34
    params = _universal_geo_params()
    params.update({"mass": mass, "h0": h0, "d0": d0, "qa": qa, "qb": qb, "alpha1": alpha1, "order": order, "h": h0})
    model.properties[prop_id] = Property(id=prop_id, type=34, title=title, params=params)




def read_prop_type29(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE29`` (M190): User property Type 29."""
    from ...model.entities import PropType29, Property
    from ..prop_reader import _universal_geo_params
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    raw_cards = [c.raw for c in valid_cards]
    p29 = PropType29(id=prop_id, cards=raw_cards, title=title)
    model.prop_type29s[prop_id] = p29
    params = _universal_geo_params()
    params["cards"] = raw_cards
    model.properties[prop_id] = Property(id=prop_id, type=29, title=title, params=params)




def read_prop_type30(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE30`` (M190): User property Type 30."""
    from ...model.entities import PropType30, Property
    from ..prop_reader import _universal_geo_params
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    raw_cards = [c.raw for c in valid_cards]
    p30 = PropType30(id=prop_id, cards=raw_cards, title=title)
    model.prop_type30s[prop_id] = p30
    params = _universal_geo_params()
    params["cards"] = raw_cards
    model.properties[prop_id] = Property(id=prop_id, type=30, title=title, params=params)




def read_prop_type31(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE31`` (M190): User property Type 31."""
    from ...model.entities import PropType31, Property
    from ..prop_reader import _universal_geo_params
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    raw_cards = [c.raw for c in valid_cards]
    p31 = PropType31(id=prop_id, cards=raw_cards, title=title)
    model.prop_type31s[prop_id] = p31
    params = _universal_geo_params()
    params["cards"] = raw_cards
    model.properties[prop_id] = Property(id=prop_id, type=31, title=title, params=params)




def read_prop_type13(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE13`` or ``/PROP/SPR_PULL`` (M194): Pulling spring property."""
    from ...model.entities import PropType13, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    stiff, f_max = 0.0, 0.0
    params = {}
    if len(valid_cards) > 0:
        c0 = valid_cards[0].tokens()
        stiff = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        f_max = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        for idx, val in enumerate(c0[2:], start=2):
            params[f"p_{idx}"] = _safe_float(val)
    prop = PropType13(id=prop_id, title=title, stiff=stiff, f_max=f_max, params=params)
    model.prop_spr_pulls[prop_id] = prop
    model.properties[prop_id] = Property(id=prop_id, type=13, title=title, params={"stiff": stiff, "f_max": f_max, **params})




def read_prop_type23(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE23`` or ``/PROP/SPR_MAT`` (M158/M195): Spring material property."""
    from ...model.entities import PropType23, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    imass = 0
    area_or_volume = 0.0
    mass = 0.0
    inertia = 0.0
    skew_id = 0
    sens_id = 0
    isflag = 0
    params = {}
    if valid_cards:
        c0 = valid_cards[0]
        if block.fixed:
            raw = c0.raw if hasattr(c0, "raw") else str(c0)
            f0 = raw[0:10].strip()
            f1 = raw[10:30].strip()
            f2 = raw[30:50].strip()
            f3 = raw[50:60].strip()
            f4 = raw[60:70].strip()
            f5 = raw[70:80].strip() if len(raw) > 70 else ""
            if len(f0) > 0 and "." not in f0 and (len(f1) > 0 or len(f2) > 0):
                imass = _safe_int(f0)
                area_or_volume = _safe_float(f1)
                mass = area_or_volume
                inertia = _safe_float(f2)
                skew_id = _safe_int(f3)
                sens_id = _safe_int(f4)
                isflag = _safe_int(f5)
            else:
                f_20 = raw[0:20].strip()
                f_skew = raw[20:30].strip()
                f_isens = raw[30:40].strip()
                f_iflag = raw[40:50].strip()
                mass = _safe_float(f_20)
                area_or_volume = mass
                skew_id = _safe_int(f_skew)
                sens_id = _safe_int(f_isens)
                isflag = _safe_int(f_iflag)
                imass = 1
        else:
            toks = c0.tokens()
            if len(toks) >= 6:
                imass = _safe_int(toks[0])
                area_or_volume = _safe_float(toks[1])
                mass = area_or_volume
                inertia = _safe_float(toks[2])
                skew_id = _safe_int(toks[3])
                sens_id = _safe_int(toks[4])
                isflag = _safe_int(toks[5])
            elif len(toks) == 4:
                mass = _safe_float(toks[0])
                area_or_volume = mass
                skew_id = _safe_int(toks[1])
                sens_id = _safe_int(toks[2])
                isflag = _safe_int(toks[3])
                imass = 1
            else:
                imass = _safe_int(toks[0]) if len(toks) > 0 else 0
                area_or_volume = _safe_float(toks[1]) if len(toks) > 1 else (_safe_float(toks[0]) if len(toks) > 0 else 0.0)
                mass = area_or_volume
                inertia = _safe_float(toks[2]) if len(toks) > 2 else 0.0
                skew_id = _safe_int(toks[3]) if len(toks) > 3 else 0
                sens_id = _safe_int(toks[4]) if len(toks) > 4 else 0
                isflag = _safe_int(toks[5]) if len(toks) > 5 else 0

    params = {
        "imass": imass,
        "area_or_volume": area_or_volume,
        "mass": mass,
        "inertia": inertia,
        "skew_id": skew_id,
        "sens_id": sens_id,
        "sensor_id": sens_id,
        "isens": sens_id,
        "isflag": isflag,
        "iflag": isflag,
    }
    if imass == 1:
        params["area"] = area_or_volume
    elif imass == 2:
        params["volume"] = area_or_volume

    prop = PropType23(
        id=prop_id, title=title, mass=mass, skew_id=skew_id,
        isens=sens_id, iflag=isflag, imass=imass, area_or_volume=area_or_volume,
        inertia=inertia, sensor_id=sens_id, isflag=isflag, params=params
    )
    model.prop_type23s[prop_id] = prop
    model.properties[prop_id] = Property(id=prop_id, type=23, title=title, params=params)
