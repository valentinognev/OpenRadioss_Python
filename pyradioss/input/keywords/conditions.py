# -*- coding: utf-8 -*-
"""
Boundary condition and initial-state readers - /KEYWORD blocks -> Model.

/BCS, /EBCS, /IMPDISP, /IMPVEL, /IMPACC, /IMPTEMP and the /INI*
initial-state family.

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





# ============================================================================
# Boundary conditions, initial conditions, loads
# ============================================================================

def read_bcs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BCS/bcs_ID`` and ``/BCS/CYCLIC/bcs_ID``::

        card 1:  title
        card 2:  Trarot   skew_ID   grnod_ID
        or (for CYCLIC):
        card 2:  skew_ID  grnd_ID1  grnd_ID2

    ``Trarot`` is the classic pair of 3-digit binary flags
    ``XYZ XYZ`` — first triple = translations, second = rotations,
    1 = fixed. Example: ``111 000`` clamps translations only.

    ``skew_ID`` (M39): the flags name the /SKEW's axes, not the global
    ones — the condensation rotates with the skew (and, for a /SKEW/MOV,
    every cycle).  See engine/kinematics.py and bcs1.F.

    Fixed dialect (cfg LOADS/bcs.cfg radioss51; M37): the six DOF flags
    live in ONE 10-character field (``   111 011``) with skew_ID and
    grnod_ID in the following %10d columns — a blank skew column made
    the token view miscount ('card 2 needs tra rot skew grnod').
    """
    from ...model.entities import CyclicBoundaryCondition

    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "NRF":
        read_bcs_nrf(block, model, log)
        return

    if sub == "WALL":
        read_bcs_wall(block, model, log)
        return

    if sub == "PROPELLANT":
        read_ebcs_propellant(block, model, log)
        return

    if sub in ("LAGMUL", "LAG"):
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or (block.fixed and cards[0].is_blank):
            log.error(f"/BCS/LAGMUL/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f1 = cards[0].cut("BCS_LAGMUL_1") if "BCS_LAGMUL_1" in CARD_LAYOUTS else _fixed_vals(cards[0], [10, 10, 10, 10])
            tra = f1[0].strip() if len(f1) > 0 else "111"
            rot = f1[1].strip() if len(f1) > 1 else "111"
            skew = _ival(f1[2]) if len(f1) > 2 else 0
            grnod = _ival(f1[3]) if len(f1) > 3 else 0
        else:
            t = cards[0].tokens()
            tra = t[0] if len(t) > 0 else "111"
            rot = t[1] if len(t) > 1 else "111"
            skew = int(float(t[2])) if len(t) > 2 else 0
            grnod = int(float(t[3])) if len(t) > 3 else 0
        from ...model.entities import BcsLagmul
        model.bcs_lagmuls[block.user_id] = BcsLagmul(
            id=block.user_id, title=title, tra=tra, rot=rot, skew_id=skew, grnod_id=grnod
        )
        return

    if sub in ("FLUX", "TEMP"):
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or (block.fixed and cards[0].is_blank):
            log.error(f"/BCS/{sub}/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("BCS_FLUX_1")
            grnod_id = _ival(f[0]) if len(f) > 0 else 0
            sens_id = _ival(f[1]) if len(f) > 1 else 0
            funct_id = _ival(f[2]) if len(f) > 2 else 0
            scale = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        else:
            t = cards[0].tokens()
            grnod_id = int(float(t[0])) if len(t) > 0 else 0
            sens_id = int(float(t[1])) if len(t) > 1 else 0
            funct_id = int(float(t[2])) if len(t) > 2 else 0
            scale = float(t[3]) if len(t) > 3 else 0.0
        model.heat_bcs[block.user_id] = HeatBcs(
            id=block.user_id,
            title=title,
            group_id=grnod_id,
            bcs_type=sub,
            sens_id=sens_id,
            funct_id=funct_id,
            scale=scale,
        )
        return

    if sub == "CYCLIC":
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or (block.fixed and cards[0].is_blank):
            log.error(f"/BCS/CYCLIC/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("BCS_CYCLIC")
            skew = _ival(f[0]) if len(f) > 0 else 0
            grnd1 = _ival(f[1]) if len(f) > 1 else 0
            grnd2 = _ival(f[2]) if len(f) > 2 else 0
        else:
            t = cards[0].tokens()
            skew = int(float(t[0])) if len(t) > 0 else 0
            grnd1 = int(float(t[1])) if len(t) > 1 else 0
            grnd2 = int(float(t[2])) if len(t) > 2 else 0
        from ...model.entities import BcsCyclic
        model.cyclic_bcs[block.user_id] = CyclicBoundaryCondition(
            id=block.user_id, title=title, skew_id=skew, grnod1_id=grnd1, grnod2_id=grnd2
        )
        model.bcs_cyclics[block.user_id] = BcsCyclic(
            id=block.user_id, title=title, skew_id=skew, grnd_id1=grnd1, grnd_id2=grnd2
        )
        return

    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    if not cards or (block.fixed and cards[0].is_blank):
        log.error(f"/BCS/{block.user_id}: missing data card", block.source)
        return

    if sub in ("TRA", "TRANS"):
        # /BCS/TRA (M138): pure translational constraints
        if block.fixed:
            f = cards[0].cut("BCS_TRA_1")
            tra = f[0].strip() if len(f) > 0 else "111"
            skew = _ival(f[1]) if len(f) > 1 else 0
            grnod = _ival(f[2]) if len(f) > 2 else 0
        else:
            t = cards[0].tokens()
            tra = t[0] if len(t) > 0 else "111"
            skew = int(float(t[1])) if len(t) > 1 else 0
            grnod = int(float(t[2])) if len(t) > 2 else 0
        fix_tra = np.array([ch == "1" for ch in tra.zfill(3)])
        fix_rot = np.array([False, False, False])
    elif sub in ("ROT", "ROTA"):
        # /BCS/ROT (M138): pure rotational constraints
        if block.fixed:
            f = cards[0].cut("BCS_ROT_1")
            rot = f[0].strip() if len(f) > 0 else "111"
            skew = _ival(f[1]) if len(f) > 1 else 0
            grnod = _ival(f[2]) if len(f) > 2 else 0
        else:
            t = cards[0].tokens()
            rot = t[0] if len(t) > 0 else "111"
            skew = int(float(t[1])) if len(t) > 1 else 0
            grnod = int(float(t[2])) if len(t) > 2 else 0
        fix_tra = np.array([False, False, False])
        fix_rot = np.array([ch == "1" for ch in rot.zfill(3)])
    else:
        if block.fixed:
            f = cards[0].cut("BCS")
            s_flags = f[0].strip() if f else ""
            raw_line = cards[0].raw.rstrip("\r\n") if hasattr(cards[0], "raw") else ""
            raw_f0 = raw_line[:10] if len(raw_line) >= 10 else (f[0] if f else "")
            if len(s_flags.split()) >= 2:
                tra, rot = s_flags.split()[:2]
            elif len(s_flags) == 6 and all(c in "012" for c in s_flags):
                tra, rot = s_flags[:3], s_flags[3:]
            elif len(raw_f0) >= 6 and any(
                len(raw_f0) > i and raw_f0[i] in ("1", "2")
                for i in (3, 4, 5, 7, 8, 9)
            ):
                dof1 = 1 if len(raw_f0) > 3 and raw_f0[3] in ("1", "2") else 0
                dof2 = 1 if len(raw_f0) > 4 and raw_f0[4] in ("1", "2") else 0
                dof3 = 1 if len(raw_f0) > 5 and raw_f0[5] in ("1", "2") else 0
                dof4 = 1 if len(raw_f0) > 7 and raw_f0[7] in ("1", "2") else 0
                dof5 = 1 if len(raw_f0) > 8 and raw_f0[8] in ("1", "2") else 0
                dof6 = 1 if len(raw_f0) > 9 and raw_f0[9] in ("1", "2") else 0
                tra = f"{dof1}{dof2}{dof3}"
                rot = f"{dof4}{dof5}{dof6}"
            elif s_flags and all(c in "012" for c in s_flags):
                padded = s_flags.zfill(6)
                tra, rot = padded[:3], padded[3:]
            else:
                log.error(f"/BCS/{block.user_id}: Trarot field needs 6 flags or 'TTT RRR', got '{f[0] if f else ''}'", block.source)
                return
            skew = _ival(f[1]) if len(f) > 1 else 0
            grnod = _ival(f[2]) if len(f) > 2 else 0
        else:
            t = cards[0].tokens()
            if len(t) < 4:
                log.error(f"/BCS/{block.user_id}: card 2 needs "
                          f"'tra rot skew grnod'", block.source)
                return
            tra, rot, skew, grnod = t[0], t[1], int(t[2]), int(t[3])
        fix_tra = np.array([ch in ("1", "2") for ch in tra.zfill(3)])
        fix_rot = np.array([ch in ("1", "2") for ch in rot.zfill(3)])

    model.bcs.append(BoundaryCondition(
        id=block.user_id, grnod_id=grnod, fix_tra=fix_tra, fix_rot=fix_rot,
        title=title, skew_id=skew))






def read_inivel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVEL/TRA/inivel_ID``::

        card 1:  title
        card 2:  Vx   Vy   Vz   grnod_ID

    ``/INIVEL/AXIS/inivel_ID`` (M5) — initial rotation about an axis::

        card 1:  title
        card 2:  omega   Dir(X|Y|Z)   grnod_ID   Xp   Yp   Zp

      every node of the group receives v += omega * d x (x0 - P), the
      velocity field of a rigid rotation at rate omega about the axis
      through P = (Xp,Yp,Zp) along Dir. This is how a spinning /RBODY is
      initialized. (The translational Vt fields of the full Radioss AXIS
      card are covered by adding a /INIVEL/TRA on the same group.)

    REAL dialects (M37):

    * TRA — cfg LOADS/inivel.cfg (radioss120)
      ``%20lg%20lg%20lg%10d%10d`` Vx Vy Vz Gnod_id Skew_id: column-cut
      (abutting/blank fields, &PARAMETER references), skew warned.
    * AXIS — cfg LOADS/inivel_axis.cfg (radioss120)::

          card 1:  DIR   FRAME_ID   GRNOD_ID        (%10s%10d%10d)
          card 2:  Vxt   Vyt   Vzt   VR             (4 x %20lg)

      rotation VR about the FRAME's DIR axis plus translation Vt.
      /FRAME is not ported: with FRAME_ID != 0 the axis is taken
      through the global origin along the GLOBAL Dir — warned loudly
      (physics deviates when the frame is not centred/aligned).  The
      card is told apart from the port's compact 'omega Dir grnod ...'
      card by its FIRST field: a direction letter, never a number.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "TRA"
    if kind.isdigit():
        kind = "TRA"
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    if not cards:
        log.error(f"/INIVEL/{block.user_id}: missing data card", block.source)
    if kind == "TRA":
        if block.fixed:
            f = cards[0].cut("INIVEL_TRA")
            v = [_fval(s) for s in f[:3]]
            grnod = _ival(f[3])
            if _ival(f[4]):
                log.warning(f"/INIVEL/TRA/{block.user_id}: skew frames "
                            f"not ported, Skew_id ignored", block.source)
        else:
            v = _floats(cards[0], 3)
            toks = cards[0].tokens()
            grnod = int(float(toks[3])) if len(toks) > 3 else 0
        model.inivel.append(InitialVelocity(
            id=block.user_id, grnod_id=grnod, v=np.array(v), title=title))
    elif kind in ("AXIS", "ROT", "ROTVEL", "ROT_VEL"):
        from ...model.entities import InivelAxis
        t = cards[0].tokens()
        real_layout = False
        if t:
            try:
                float(t[0].replace("D", "E").replace("d", "e"))
            except ValueError:
                real_layout = True         # first field is the DIR letter
        if real_layout:
            if block.fixed:
                f = cards[0].cut("INIVEL_AXIS_1")
                g = cards[1].cut("INIVEL_AXIS_2") if len(cards) > 1 else [""] * 4
                h = cards[2].cut("INIVEL_AXIS_3") if len(cards) > 2 and not cards[2].is_blank else []
            else:
                toks0 = cards[0].tokens()
                f = [toks0[0], toks0[1] if len(toks0) > 1 else "0", toks0[2] if len(toks0) > 2 else "0"]
                toks1 = cards[1].tokens() if len(cards) > 1 else []
                g = [toks1[i] if i < len(toks1) else "0.0" for i in range(4)]
                toks2 = cards[2].tokens() if len(cards) > 2 else []
                h = [toks2[i] if i < len(toks2) else "0.0" for i in range(2)]

            axis = _direction(f[0])
            frame, grnod = _ival(f[1]), _ival(f[2])
            vt = np.array([_fval(s) for s in g[:3]])
            omega = _fval(g[3])
            tstart = _fval(h[0], 0.0) if len(h) > 0 else 0.0
            sens_id = _ival(h[1]) if len(h) > 1 else 0

            model.inivel.append(InitialVelocity(
                id=block.user_id, grnod_id=grnod, v=vt, title=title,
                kind="AXIS", omega=omega, axis=axis, origin=np.zeros(3),
                frame_id=frame,
                dir={"X": 1, "Y": 2, "Z": 3}.get(f[0].strip().upper(), 3)))
            model.inivel_axes[block.user_id] = InivelAxis(
                id=block.user_id, title=title, dir=f[0].strip().upper() if f[0].strip() else "Z",
                frame_id=frame, grnod_id=grnod, vx=vt[0], vy=vt[1], vz=vt[2],
                vr=omega, tstart=tstart, sens_id=sens_id
            )
            return
        omega = float(t[0])
        axis = _direction(t[1]) if len(t) > 1 else np.array([0.0, 0.0, 1.0])
        grnod = int(float(t[2])) if len(t) > 2 else 0
        origin = np.array([float(x) for x in t[3:6]]) if len(t) >= 6 \
            else np.zeros(3)
        model.inivel.append(InitialVelocity(
            id=block.user_id, grnod_id=grnod, v=np.zeros(3), title=title,
            kind="AXIS", omega=omega, axis=axis, origin=origin))
        model.inivel_axes[block.user_id] = InivelAxis(
            id=block.user_id, title=title, dir=t[1].upper() if len(t) > 1 else "Z",
            frame_id=0, grnod_id=grnod, vx=0.0, vy=0.0, vz=0.0,
            vr=omega, tstart=0.0, sens_id=0
        )
    elif kind == "FVM":
        read_inivel_fvm(block, model, log)
    elif kind == "NODE":
        read_inivel_node(block, model, log)
    elif kind in ("T+G", "TG", "GRAD", "GRADIENT"):
        # /INIVEL/T+G (M168)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/INIVEL/T+G/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f1 = cards[0].cut("INIVEL_TG_1")
            vx = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
            vy = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
            vz = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            grnod_id = _ival(f1[3]) if len(f1) > 3 else 0
            skew_id = _ival(f1[4]) if len(f1) > 4 else 0
        else:
            t1 = cards[0].tokens()
            vx = float(t1[0]) if len(t1) > 0 else 0.0
            vy = float(t1[1]) if len(t1) > 1 else 0.0
            vz = float(t1[2]) if len(t1) > 2 else 0.0
            grnod_id = int(float(t1[3])) if len(t1) > 3 else 0
            skew_id = int(float(t1[4])) if len(t1) > 4 else 0

        model.inivel.append(InitialVelocity(
            id=block.user_id, grnod_id=grnod_id, v=np.array([vx, vy, vz]),
            title=title, kind="TG", frame_id=skew_id,
        ))
    elif kind == "PART":
        # /INIVEL/PART (M137)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/INIVEL/PART/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("INIVEL_PART_1")
            part_id = _ival(f[0]) if len(f) > 0 else 0
            vx = _fval(f[1]) if len(f) > 1 else 0.0
            vy = _fval(f[2]) if len(f) > 2 else 0.0
            vz = _fval(f[3]) if len(f) > 3 else 0.0
            vr = _fval(f[4]) if len(f) > 4 else 0.0
            skew_id = _ival(f[5]) if len(f) > 5 else 0
        else:
            t = cards[0].tokens()
            part_id = int(float(t[0])) if len(t) > 0 else 0
            vx = float(t[1]) if len(t) > 1 else 0.0
            vy = float(t[2]) if len(t) > 2 else 0.0
            vz = float(t[3]) if len(t) > 3 else 0.0
            vr = float(t[4]) if len(t) > 4 else 0.0
            skew_id = int(float(t[5])) if len(t) > 5 else 0
        tstart, sens_id = 0.0, 0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            tstart = float(t2[0]) if len(t2) > 0 else 0.0
            sens_id = int(float(t2[1])) if len(t2) > 1 else 0
        model.inivel_parts[block.user_id] = InivelPart(
            id=block.user_id, title=title, part_id=part_id,
            vx=vx, vy=vy, vz=vz, vr=vr, skew_id=skew_id,
            tstart=tstart, sens_id=sens_id
        )
    elif kind == "SPH":
        # /INIVEL/SPH (M137)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/INIVEL/SPH/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("INIVEL_SPH_1")
            grsph_id = _ival(f[0]) if len(f) > 0 else 0
            vx = _fval(f[1]) if len(f) > 1 else 0.0
            vy = _fval(f[2]) if len(f) > 2 else 0.0
            vz = _fval(f[3]) if len(f) > 3 else 0.0
            skew_id = _ival(f[4]) if len(f) > 4 else 0
        else:
            t = cards[0].tokens()
            grsph_id = int(float(t[0])) if len(t) > 0 else 0
            vx = float(t[1]) if len(t) > 1 else 0.0
            vy = float(t[2]) if len(t) > 2 else 0.0
            vz = float(t[3]) if len(t) > 3 else 0.0
            skew_id = int(float(t[4])) if len(t) > 4 else 0
        model.inivel_sphs[block.user_id] = InivelSph(
            id=block.user_id, title=title, grsph_id=grsph_id,
            vx=vx, vy=vy, vz=vz, skew_id=skew_id
        )
    else:
        log.warning(f"/INIVEL/{kind} not ported (TRA, AXIS, FVM, NODE, ROT, PART, SPH supported)",
                    block.source)




def read_inivel_fvm(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVEL/FVM/inivel_ID`` (M112): FVM airbag initial velocity::

        card 1:  title
        card 2:  Vx  Vy  Vz  grbric_ID  grqd_ID  grtria_ID  skew_ID
        card 3 (optional):  Tstart  sens_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIVEL/FVM/{block.user_id}: missing data card", block.source)
        return
    from ...model.entities import InivelFvm

    vx, vy, vz = 0.0, 0.0, 0.0
    grbric_id, grquad_id, grsh3n_id, skew_id = 0, 0, 0, 0
    tstart, sens_id = 0.0, 0

    if block.fixed:
        f1 = cards[0].cut("INIVEL_FVM_1")
        vx = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        vy = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        vz = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
        grbric_id = _ival(f1[3]) if len(f1) > 3 else 0
        grquad_id = _ival(f1[4]) if len(f1) > 4 else 0
        grsh3n_id = _ival(f1[5]) if len(f1) > 5 else 0
        skew_id = _ival(f1[6]) if len(f1) > 6 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("INIVEL_FVM_2")
            tstart = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            sens_id = _ival(f2[1]) if len(f2) > 1 else 0
    else:
        t1 = cards[0].tokens()
        vx = float(t1[0]) if len(t1) > 0 else 0.0
        vy = float(t1[1]) if len(t1) > 1 else 0.0
        vz = float(t1[2]) if len(t1) > 2 else 0.0
        grbric_id = int(float(t1[3])) if len(t1) > 3 else 0
        grquad_id = int(float(t1[4])) if len(t1) > 4 else 0
        grsh3n_id = int(float(t1[5])) if len(t1) > 5 else 0
        skew_id = int(float(t1[6])) if len(t1) > 6 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            tstart = float(t2[0]) if len(t2) > 0 else 0.0
            sens_id = int(float(t2[1])) if len(t2) > 1 else 0

    model.inivel_fvms[block.user_id] = InivelFvm(
        id=block.user_id, title=title, vx=vx, vy=vy, vz=vz,
        grbric_id=grbric_id, grquad_id=grquad_id, grsh3n_id=grsh3n_id,
        skew_id=skew_id, tstart=tstart, sens_id=sens_id
    )




def read_inivel_node(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVEL/NODE/inivel_ID`` (M112): Nodal vector initial velocities::

        card 1:  title
        card list:
            card a: Node_ID  Skew_ID  Vxt  Vyt  Vzt
            card b: (blank)  Vxr  Vyr  Vzr
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ...model.entities import InivelNode, InivelNodeItem

    items = []
    i = 0
    while i < len(cards):
        c1 = cards[i]
        if c1.is_blank:
            i += 1
            continue
        c2 = cards[i+1] if i+1 < len(cards) and not cards[i+1].is_blank else None
        if block.fixed:
            f1 = c1.cut("INIVEL_NODE_1")
            nid = _ival(f1[0]) if len(f1) > 0 else 0
            skw = _ival(f1[1]) if len(f1) > 1 else 0
            vxt = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            vyt = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
            vzt = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0

            vxr, vyr, vzr = 0.0, 0.0, 0.0
            if c2 is not None:
                f2 = c2.cut("INIVEL_NODE_2")
                vxr = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                vyr = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
                vzr = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
                i += 2
            else:
                i += 1
        else:
            t1 = c1.tokens()
            nid = int(float(t1[0])) if len(t1) > 0 else 0
            skw = int(float(t1[1])) if len(t1) > 1 else 0
            vxt = float(t1[2]) if len(t1) > 2 else 0.0
            vyt = float(t1[3]) if len(t1) > 3 else 0.0
            vzt = float(t1[4]) if len(t1) > 4 else 0.0

            vxr, vyr, vzr = 0.0, 0.0, 0.0
            if c2 is not None:
                t2 = c2.tokens()
                vxr = float(t2[0]) if len(t2) > 0 else 0.0
                vyr = float(t2[1]) if len(t2) > 1 else 0.0
                vzr = float(t2[2]) if len(t2) > 2 else 0.0
                i += 2
            else:
                i += 1
        items.append(InivelNodeItem(
            node_id=nid, skew_id=skw, vxt=vxt, vyt=vyt, vzt=vzt,
            vxr=vxr, vyr=vyr, vzr=vzr
        ))

    model.inivel_nodes[block.user_id] = InivelNode(id=block.user_id, title=title, items=items)




def read_imptemp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPTEMP/imptemp_ID`` (M93)::

        card 1:  title
        card 2:  func_IDT  sensor_ID  grnod_ID
        card 3:  Ascale_x  Fscale_y  T_start  T_stop
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/IMPTEMP/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("IMPTEMP_1")
        funct_id = _ival(f[0])
        sens_id = _ival(f[1]) if len(f) > 1 else 0
        grnod_id = _ival(f[2]) if len(f) > 2 else 0

        g = cards[1].cut("IMPTEMP_2") if len(cards) > 1 and not cards[1].is_blank else []
        xscale = _fval(g[0], 1.0) if len(g) > 0 else 1.0
        scale = _fval(g[1], 1.0) if len(g) > 1 else 1.0
        tstart = _fval(g[2], 0.0) if len(g) > 2 else 0.0
        tstop = _fval(g[3], 1.0e30) if len(g) > 3 else 1.0e30
    else:
        t0 = cards[0].tokens()
        funct_id = int(t0[0]) if len(t0) > 0 else 0
        sens_id = int(t0[1]) if len(t0) > 1 else 0
        grnod_id = int(t0[2]) if len(t0) > 2 else 0

        t1 = cards[1].tokens() if len(cards) > 1 and not cards[1].is_blank else []
        xscale = float(t1[0]) if len(t1) > 0 else 1.0
        scale = float(t1[1]) if len(t1) > 1 else 1.0
        tstart = float(t1[2]) if len(t1) > 2 else 0.0
        tstop = float(t1[3]) if len(t1) > 3 else 1.0e30

    if xscale == 0.0:
        xscale = 1.0
    if scale == 0.0:
        scale = 1.0
    if tstop == 0.0:
        tstop = 1.0e30

    it = ImposedTemperature(
        id=block.user_id, funct_id=funct_id, grnod_id=grnod_id,
        sens_id=sens_id, scale=scale, xscale=xscale,
        tstart=tstart, tstop=tstop, title=title,
    )
    model.imptemp.append(it)
    model.imptemps[block.user_id] = it




#: directions of the /IMPVEL & /IMPDISP cards, mapped to the 6-DOF index
#: (0..2 = translation X/Y/Z, 3..5 = rotation XX/YY/ZZ — the same ordering
#: the /MPC and implicit dofmap use). M39: the rotational directions are now
#: applied to the nodal / rigid-body angular velocity (they were parsed and
#: DISCARDED before, which left every RD-E-1000 Bending deck — an /IMPVEL/XX
#: on the /RBODY master — completely undriven).
_IMP_DOF = {"X": 0, "Y": 1, "Z": 2, "XX": 3, "YY": 4, "ZZ": 5}


_IMP_DIRS = tuple(_IMP_DOF)




def split_imposed_card(cards) -> Optional[dict]:
    """Field split shared by /IMPVEL and /IMPDISP (also used by the
    deck-writer round-trip). Returns a dict or None on an empty block.

    Official layout (cfg ``LOADS/impvel.cfg`` / ``impdisp.cfg``, FORMAT
    radioss51+, fixed 10-char columns; starter reader
    ``starter/source/constraints/general/impvel/read_impvel.F``)::

        card 1:  fct_ID    Dir   skew_ID  sensor_ID  grnod_ID  frame_ID  Icoor
        card 2:  Ascale_x  Fscale_Y  Tstart  Tstop

    with defaults Ascale_x 0 -> 1, Fscale_Y 0 -> 1, Tstop 0 -> infinity;
    v(t) = Fscale_Y * f(t / Ascale_x) inside [Tstart, Tstop] (fixvel.F
    stores FACX = 1/Ascale_x and skips outside STARTT/STOPT).

    The port's historic free-format card ``fct Dir grnod [scale]`` (still
    used by the bundled test decks) is kept as a fallback. Detection: an
    official card carries the direction ALONE in columns 11-20 and the
    group id in columns 41-50; a free-format card never populates
    column 41+ (pre-M37 the port read every deck free-format, which made
    grnod swallow the official card's skew_ID and the scale its
    sensor_ID — the RD-V-0200 'inert model' bug).
    """
    if not cards:
        return None
    f = cards[0].fields()
    official = False
    if f[1].upper() in _IMP_DIRS and f[4]:
        try:
            int(f[0]), int(f[4])
            official = True
        except ValueError:
            official = False
    out = {"skew": 0, "sens": 0, "frame": 0, "icoor": 0,
           "xscale": 1.0, "scale": 1.0, "tstart": 0.0, "tstop": 1.0e30}
    if official:
        try:
            out.update(fct=int(f[0]), dir=f[1].upper(), grnod=int(f[4]),
                       skew=int(f[2] or 0), sens=int(f[3] or 0),
                       frame=int(f[5] or 0), icoor=int(f[6] or 0))
        except ValueError:
            # a FREE-FORMAT card whose float scale spills across the
            # 10-char columns defeats the column detection ('...4
            # -0.2' puts '-0.' in the sensor column and '2' in the
            # grnod column): every official field is an integer, so an
            # int-parse failure identifies the compact dialect (M37)
            official = False
    if official:
        # card 2 is 4 x %20lg fixed columns (Ascale_x Fscale_Y Tstart
        # Tstop) — cut at the column widths, NOT tokenized: real decks
        # pack them with no whitespace ('0.01.00000000000000E+30' =
        # Tstart 0.0 abutting Tstop 1e30) and a blank column is the
        # field's default (M37, card_layouts 'IMP_2')
        vals = [_fval(s) for s in cards[1].cut("IMP_2")] \
            if len(cards) > 1 else []
        if len(vals) > 0 and vals[0] != 0.0:
            out["xscale"] = vals[0]
        if len(vals) > 1 and vals[1] != 0.0:
            out["scale"] = vals[1]
        if len(vals) > 2:
            out["tstart"] = vals[2]
        if len(vals) > 3 and vals[3] != 0.0:
            out["tstop"] = vals[3]
    else:
        raw0 = cards[0].raw.replace(",", " ")
        t = raw0.split()
        if len(t) >= 5 and t[1].upper() in _IMP_DIRS:
            out.update(fct=int(float(t[0])), dir=t[1].upper(),
                       skew=int(float(t[2])) if len(t) > 2 and t[2] else 0,
                       sens=int(float(t[3])) if len(t) > 3 and t[3] else 0,
                       grnod=int(float(t[4])) if len(t) > 4 and t[4] else 0,
                       frame=int(float(t[5])) if len(t) > 5 and t[5] else 0,
                       icoor=int(float(t[6])) if len(t) > 6 and t[6] else 0)
            if len(cards) > 1 and not cards[1].is_blank:
                raw1 = cards[1].raw.replace(",", " ")
                t2 = raw1.split()
                if len(t2) > 0 and float(t2[0]) != 0.0:
                    out["xscale"] = float(t2[0])
                if len(t2) > 1 and float(t2[1]) != 0.0:
                    out["scale"] = float(t2[1])
                if len(t2) > 2:
                    out["tstart"] = float(t2[2])
                if len(t2) > 3 and float(t2[3]) != 0.0:
                    out["tstop"] = float(t2[3])
        elif len(t) >= 3 and t[1].upper() in _IMP_DIRS:
            out.update(fct=int(float(t[0])), dir=t[1].upper(), grnod=int(float(t[2])),
                       scale=float(t[3]) if len(t) > 3 else 1.0)
            if len(cards) > 1 and not cards[1].is_blank:
                raw1 = cards[1].raw.replace(",", " ")
                t2 = raw1.split()
                if len(t2) > 0 and float(t2[0]) != 0.0:
                    out["xscale"] = float(t2[0])
                if len(t2) > 1 and float(t2[1]) != 0.0:
                    out["scale"] = float(t2[1])
                if len(t2) > 2:
                    out["tstart"] = float(t2[2])
                if len(t2) > 3 and float(t2[3]) != 0.0:
                    out["tstop"] = float(t2[3])
        elif len(t) > 0:
            out.update(fct=int(float(t[0])), dir=t[1].upper() if len(t) > 1 else "X",
                       grnod=int(float(t[2])) if len(t) > 2 else 0,
                       scale=float(t[3]) if len(t) > 3 else 1.0)
    return out




def read_impvel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPVEL/impvel_ID``::

        card 1:  title
        card 2:  fct_ID  Dir(X|Y|Z)  skew_ID  sensor_ID  grnod_ID  frame  Icoor
        card 3:  Ascale_x  Fscale_Y  Tstart  Tstop

      kinematic condition v(t) = Fscale_Y * f(t / Ascale_x) imposed on
      that DOF of the group's nodes inside [Tstart, Tstop] (overrides the
      equations of motion for that DOF). Format details + the legacy
      free-format fallback: :func:`split_imposed_card`.
    """
    if len(block.parts) > 1 and block.parts[1].upper() == "FGEO":
        read_impvel_fgeo(block, model, log)
        return
    _read_imposed(block, model, log, "IMPVEL", ImposedVelocity, model.impvel)




def read_impdisp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPDISP/impdisp_ID`` (M5, M112) — same cards as /IMPVEL (they share
    the cfg layout and the Fortran reader), but the curve is a
    *displacement*: d(t) = Fscale_Y * f(t / Ascale_x), the Engine sets the
    velocity each cycle so the node lands at x0 + d(t+dt). The curve
    should start at f(0) = 0 — a nonzero start makes the node JUMP in the
    first cycle.
    """
    if len(block.parts) > 1 and block.parts[1].upper() == "FGEO":
        read_impdisp_fgeo(block, model, log)
        return
    _read_imposed(block, model, log, "IMPDISP", ImposedDisplacement,
                  model.impdisp)




def read_impdisp_fgeo(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPDISP/FGEO/impdisp_ID`` (M112): Imposed final geometry displacement::

        card 1:  title
        card 2:  fct_ID  part_ID  (blank)  sens_ID
        card 3:  Ascale  (blank)  Tstart  Tstop
        card list: node_IDN  Xn  Yn  Zn
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/IMPDISP/FGEO/{block.user_id}: missing data card", block.source)
        return
    from ...model.entities import ImpdispFgeo

    if block.fixed:
        f1 = cards[0].cut("IMPDISP_FGEO_1")
        fct_id = _ival(f1[0]) if len(f1) > 0 else 0
        part_id = _ival(f1[1]) if len(f1) > 1 else 0
        sens_id = _ival(f1[3]) if len(f1) > 3 else 0

        ascale = 1.0
        tstart = 0.0
        tstop = 1.0e30
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("IMPDISP_FGEO_2")
            ascale = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            tstart = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            tstop = _fval(f2[3], 1.0e30) if len(f2) > 3 else 1.0e30

        nodes = []
        for c in cards[2:]:
            if c.is_blank:
                continue
            fl = c.cut("IMPDISP_FGEO_LIST")
            nodes.append({
                "node_id": _ival(fl[0]),
                "x": _fval(fl[1], 0.0) if len(fl) > 1 else 0.0,
                "y": _fval(fl[2], 0.0) if len(fl) > 2 else 0.0,
                "z": _fval(fl[3], 0.0) if len(fl) > 3 else 0.0,
            })
    else:
        t1 = cards[0].tokens()
        fct_id = int(float(t1[0])) if len(t1) > 0 else 0
        part_id = int(float(t1[1])) if len(t1) > 1 else 0
        sens_id = int(float(t1[2])) if len(t1) > 2 else 0

        ascale = 1.0
        tstart = 0.0
        tstop = 1.0e30
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            ascale = float(t2[0]) if len(t2) > 0 else 1.0
            tstart = float(t2[1]) if len(t2) > 1 else 0.0
            tstop = float(t2[2]) if len(t2) > 2 else 1.0e30

        nodes = []
        for c in cards[2:]:
            if c.is_blank:
                continue
            tl = c.tokens()
            nodes.append({
                "node_id": int(float(tl[0])) if len(tl) > 0 else 0,
                "x": float(tl[1]) if len(tl) > 1 else 0.0,
                "y": float(tl[2]) if len(tl) > 2 else 0.0,
                "z": float(tl[3]) if len(tl) > 3 else 0.0,
            })

    model.impdisp_fgeos[block.user_id] = ImpdispFgeo(
        id=block.user_id, title=title, fct_id=fct_id, part_id=part_id,
        sens_id=sens_id, ascale=ascale, tstart=tstart, tstop=tstop,
        nodes=nodes
    )




def read_impvel_fgeo(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPVEL/FGEO/impvel_ID`` (M112): Imposed final geometry velocity::

        card 1:  title
        card 2:  fct_ID  part_ID  fct_ID_L  sens_ID
        card 3:  Ascale  T0  Tstart  Fscale_L  Dmin
        card list: node_IDN  node_ID'N
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/IMPVEL/FGEO/{block.user_id}: missing data card", block.source)
        return
    from ...model.entities import ImpvelFgeo

    if block.fixed:
        f1 = cards[0].cut("IMPVEL_FGEO_1")
        fct_id = _ival(f1[0]) if len(f1) > 0 else 0
        part_id = _ival(f1[1]) if len(f1) > 1 else 0
        fct_l_id = _ival(f1[2]) if len(f1) > 2 else 0
        sens_id = _ival(f1[3]) if len(f1) > 3 else 0

        ascale = 1.0
        t0 = 0.0
        tstart = 0.0
        fscale_l = 1.0
        dmin = 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("IMPVEL_FGEO_2")
            ascale = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            t0 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            tstart = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            fscale_l = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            dmin = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0

        pairs = []
        for c in cards[2:]:
            if c.is_blank:
                continue
            fl = c.cut("IMPVEL_FGEO_LIST")
            pairs.append((_ival(fl[0]), _ival(fl[1]) if len(fl) > 1 else 0))
    else:
        t1 = cards[0].tokens()
        fct_id = int(float(t1[0])) if len(t1) > 0 else 0
        part_id = int(float(t1[1])) if len(t1) > 1 else 0
        fct_l_id = int(float(t1[2])) if len(t1) > 2 else 0
        sens_id = int(float(t1[3])) if len(t1) > 3 else 0

        ascale = 1.0
        t0 = 0.0
        tstart = 0.0
        fscale_l = 1.0
        dmin = 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            ascale = float(t2[0]) if len(t2) > 0 else 1.0
            t0 = float(t2[1]) if len(t2) > 1 else 0.0
            tstart = float(t2[2]) if len(t2) > 2 else 0.0
            fscale_l = float(t2[3]) if len(t2) > 3 else 1.0
            dmin = float(t2[4]) if len(t2) > 4 else 0.0

        pairs = []
        for c in cards[2:]:
            if c.is_blank:
                continue
            tl = c.tokens()
            pairs.append((int(float(tl[0])), int(float(tl[1])) if len(tl) > 1 else 0))

    model.impvel_fgeos[block.user_id] = ImpvelFgeo(
        id=block.user_id, title=title, fct_id=fct_id, part_id=part_id,
        fct_l_id=fct_l_id, sens_id=sens_id, ascale=ascale, t0=t0,
        tstart=tstart, fscale_l=fscale_l, dmin=dmin, pairs=pairs
    )





def read_impacc(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPACC/impacc_ID`` (M92) — imposed acceleration a(t) = scale * funct(t)
    on one DOF of a node group. Same cards and layout as /IMPVEL.
    """
    _read_imposed(block, model, log, "IMPACC", ImposedAcceleration,
                  model.impacc)




def read_ebcs_propellant(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/PROPELLANT/id`` or ``/BCS/PROPELLANT/id`` (M114): Solid propellant combustion boundary::

        card 1:  title
        card 2:  surf_ID  sensor_id  submat_id  ienthalpy
        card 3:  rho0s  Tburn
        card 4:  param_a  param_n
        card 5:  ffunc_id  (blank)  fscaleX  fscaleY
        card 6:  gfunc_id  (blank)  gscaleX  gscaleY
        card 7:  hfunc_id  (blank)  hscaleX  hscaleY
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/PROPELLANT/{block.user_id}: missing data card", block.source)
        return
    from ...model.entities import EbcsPropellant

    surf_id, sens_id, submat_id, ienthalpy = 0, 0, 1, 1
    rho0s, tburn = 0.0, 300.0
    param_a, param_n = 0.0, 0.0
    ffunc_id, fscaleX, fscaleY = 0, 1.0, 1.0
    gfunc_id, gscaleX, gscaleY = 0, 1.0, 1.0
    hfunc_id, hscaleX, hscaleY = 0, 1.0, 1.0

    if block.fixed:
        f1 = cards[0].cut("EBCS_PROPELLANT_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        sens_id = _ival(f1[1]) if len(f1) > 1 else 0
        submat_id = _ival(f1[2], default=1) if len(f1) > 2 else 1
        ienthalpy = _ival(f1[3], default=1) if len(f1) > 3 else 1

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("EBCS_PROPELLANT_2")
            rho0s = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            tburn = _fval(f2[1], 300.0) if len(f2) > 1 else 300.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("EBCS_PROPELLANT_3")
            param_a = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            param_n = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("EBCS_PROPELLANT_FUNC")
            ffunc_id = _ival(f4[0]) if len(f4) > 0 else 0
            fscaleX = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0
            fscaleY = _fval(f4[3], 1.0) if len(f4) > 3 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("EBCS_PROPELLANT_FUNC")
            gfunc_id = _ival(f5[0]) if len(f5) > 0 else 0
            gscaleX = _fval(f5[2], 1.0) if len(f5) > 2 else 1.0
            gscaleY = _fval(f5[3], 1.0) if len(f5) > 3 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("EBCS_PROPELLANT_FUNC")
            hfunc_id = _ival(f6[0]) if len(f6) > 0 else 0
            hscaleX = _fval(f6[2], 1.0) if len(f6) > 2 else 1.0
            hscaleY = _fval(f6[3], 1.0) if len(f6) > 3 else 1.0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        sens_id = int(float(t1[1])) if len(t1) > 1 else 0
        submat_id = int(float(t1[2])) if len(t1) > 2 else 1
        ienthalpy = int(float(t1[3])) if len(t1) > 3 else 1

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            rho0s = float(t2[0]) if len(t2) > 0 else 0.0
            tburn = float(t2[1]) if len(t2) > 1 else 300.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            param_a = float(t3[0]) if len(t3) > 0 else 0.0
            param_n = float(t3[1]) if len(t3) > 1 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            ffunc_id = int(float(t4[0])) if len(t4) > 0 else 0
            fscaleX = float(t4[1]) if len(t4) > 1 else 1.0
            fscaleY = float(t4[2]) if len(t4) > 2 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            gfunc_id = int(float(t5[0])) if len(t5) > 0 else 0
            gscaleX = float(t5[1]) if len(t5) > 1 else 1.0
            gscaleY = float(t5[2]) if len(t5) > 2 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            hfunc_id = int(float(t6[0])) if len(t6) > 0 else 0
            hscaleX = float(t6[1]) if len(t6) > 1 else 1.0
            hscaleY = float(t6[2]) if len(t6) > 2 else 1.0

    model.ebcs_propellants[block.user_id] = EbcsPropellant(
        id=block.user_id, title=title, surf_id=surf_id, sens_id=sens_id,
        submat_id=submat_id, ienthalpy=ienthalpy, rho0s=rho0s, tburn=tburn,
        param_a=param_a, param_n=param_n, f_func_id=ffunc_id, f_scale_x=fscaleX,
        f_scale_y=fscaleY, g_func_id=gfunc_id, g_scale_x=gscaleX, g_scale_y=gscaleY,
        h_func_id=hfunc_id, h_scale_x=hscaleX, h_scale_y=hscaleY
    )




def read_ebcs_nrf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/NRF`` or ``/EBCS/NON_REFLECT`` (M125): Non-reflecting frontier boundary condition."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/EBCS/NRF/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    tcar_p, tcar_vf = 0.0, 0.0
    from ...model.entities import EbcsNrf

    if block.fixed:
        f0 = cards[0].cut("EBCS_NRF_1")
        surf_id = _ival(f0[0]) if len(f0) > 0 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            f1 = cards[1].cut("EBCS_NRF_2")
            tcar_p = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
            tcar_vf = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
    else:
        t0 = cards[0].tokens()
        surf_id = int(float(t0[0])) if len(t0) > 0 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            t1 = cards[1].tokens()
            tcar_p = float(t1[0]) if len(t1) > 0 else 0.0
            tcar_vf = float(t1[1]) if len(t1) > 1 else 0.0

    ebcs_id = block.user_id if block.user_id is not None else 1
    model.ebcs_nrfs[ebcs_id] = EbcsNrf(
        id=ebcs_id,
        title=title,
        surf_id=surf_id,
        tcar_p=tcar_p,
        tcar_vf=tcar_vf,
    )




def read_ebcs_periodic(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/PERIODIC/id`` (M138): Eulerian periodic boundary condition."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/PERIODIC/{ebcs_id}: missing data card", block.source)
        return
    from ...model.entities import EbcsPeriodic
    if block.fixed:
        f = cards[0].cut("EBCS_PERIODIC_1")
        surf1 = _ival(f[0]) if len(f) > 0 else 0
        surf2 = _ival(f[1]) if len(f) > 1 else 0
        skew = _ival(f[2]) if len(f) > 2 else 0
        grpart = _ival(f[3]) if len(f) > 3 else 0
    else:
        toks = cards[0].tokens()
        surf1 = int(float(toks[0])) if len(toks) > 0 else 0
        surf2 = int(float(toks[1])) if len(toks) > 1 else 0
        skew = int(float(toks[2])) if len(toks) > 2 else 0
        grpart = int(float(toks[3])) if len(toks) > 3 else 0
    model.ebcs_periodics[ebcs_id] = EbcsPeriodic(
        id=ebcs_id, title=title, surf1_id=surf1, surf2_id=surf2, skew_id=skew, grpart_id=grpart
    )




def read_ebcs_cyclic(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/CYCLIC/id`` (M138, M200): Eulerian cyclic boundary condition."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/CYCLIC/{ebcs_id}: missing data card", block.source)
        return
    from ...model.entities import EbcsCyclic

    if len(cards) >= 2 and not cards[1].is_blank:
        if block.fixed:
            f1 = cards[0].cut("EBCS_CYCLIC_1") if "EBCS_CYCLIC_1" in CARD_LAYOUTS else cards[0].cut("EBCS_PERIODIC_1")
            f2 = cards[1].cut("EBCS_CYCLIC_2") if "EBCS_CYCLIC_2" in CARD_LAYOUTS else cards[1].cut("EBCS_PERIODIC_1")
            surf1 = _ival(f1[0]) if len(f1) > 0 else 0
            n1 = _ival(f1[1]) if len(f1) > 1 else 0
            n2 = _ival(f1[2]) if len(f1) > 2 else 0
            n3 = _ival(f1[3]) if len(f1) > 3 else 0
            surf2 = _ival(f2[0]) if len(f2) > 0 else 0
            n4 = _ival(f2[1]) if len(f2) > 1 else 0
            n5 = _ival(f2[2]) if len(f2) > 2 else 0
            n6 = _ival(f2[3]) if len(f2) > 3 else 0
        else:
            t1 = cards[0].tokens()
            t2 = cards[1].tokens()
            surf1 = int(float(t1[0])) if len(t1) > 0 else 0
            n1 = int(float(t1[1])) if len(t1) > 1 else 0
            n2 = int(float(t1[2])) if len(t1) > 2 else 0
            n3 = int(float(t1[3])) if len(t1) > 3 else 0
            surf2 = int(float(t2[0])) if len(t2) > 0 else 0
            n4 = int(float(t2[1])) if len(t2) > 1 else 0
            n5 = int(float(t2[2])) if len(t2) > 2 else 0
            n6 = int(float(t2[3])) if len(t2) > 3 else 0
        model.ebcs_cyclics[ebcs_id] = EbcsCyclic(
            id=ebcs_id, title=title, surf1_id=surf1, surf_id1=surf1,
            node_id1=n1, node_id2=n2, node_id3=n3,
            surf2_id=surf2, surf_id2=surf2,
            node_id4=n4, node_id5=n5, node_id6=n6
        )
    else:
        if block.fixed:
            f = cards[0].cut("EBCS_PERIODIC_1")
            surf1 = _ival(f[0]) if len(f) > 0 else 0
            surf2 = _ival(f[1]) if len(f) > 1 else 0
            skew = _ival(f[2]) if len(f) > 2 else 0
            grpart = _ival(f[3]) if len(f) > 3 else 0
        else:
            toks = cards[0].tokens()
            surf1 = int(float(toks[0])) if len(toks) > 0 else 0
            surf2 = int(float(toks[1])) if len(toks) > 1 else 0
            skew = int(float(toks[2])) if len(toks) > 2 else 0
            grpart = int(float(toks[3])) if len(toks) > 3 else 0
        model.ebcs_cyclics[ebcs_id] = EbcsCyclic(
            id=ebcs_id, title=title, surf1_id=surf1, surf2_id=surf2, skew_id=skew, grpart_id=grpart
        )





def read_ebcs_pres(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/PRES/id`` (M150): Eulerian imposed pressure boundary condition."""
    from ...model.entities import EbcsPres
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/PRES/{ebcs_id}: missing data card", block.source)
        return
    surf_id = 0
    c = 0.0
    fct_pres, scale_pres = 0, 1.0
    fct_rho, scale_rho = 0, 1.0
    fct_en, scale_en = 0, 1.0
    lcar, r1, r2 = 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("EBCS_PRES_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("EBCS_PRES_2")
            c = _fval(f2[0]) if len(f2) > 0 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("EBCS_PRES_3")
            fct_pres = _ival(f3[0]) if len(f3) > 0 else 0
            scale_pres = _fval(f3[1], 1.0) if len(f3) > 1 else 1.0
        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("EBCS_PRES_4")
            fct_rho = _ival(f4[0]) if len(f4) > 0 else 0
            scale_rho = _fval(f4[1], 1.0) if len(f4) > 1 else 1.0
        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("EBCS_PRES_5")
            fct_en = _ival(f5[0]) if len(f5) > 0 else 0
            scale_en = _fval(f5[1], 1.0) if len(f5) > 1 else 1.0
        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("EBCS_PRES_6")
            lcar = _fval(f6[0]) if len(f6) > 0 else 0.0
            r1 = _fval(f6[1]) if len(f6) > 1 else 0.0
            r2 = _fval(f6[2]) if len(f6) > 2 else 0.0
    else:
        all_tokens = []
        for card in cards:
            if not card.is_blank:
                all_tokens.extend(card.tokens())
        if len(all_tokens) >= 1:
            surf_id = int(float(all_tokens[0]))
        if len(all_tokens) >= 2:
            c = float(all_tokens[1])
        if len(all_tokens) >= 3:
            fct_pres = int(float(all_tokens[2]))
        if len(all_tokens) >= 4:
            scale_pres = float(all_tokens[3])
        if len(all_tokens) >= 5:
            fct_rho = int(float(all_tokens[4]))
        if len(all_tokens) >= 6:
            scale_rho = float(all_tokens[5])
        if len(all_tokens) >= 7:
            fct_en = int(float(all_tokens[6]))
        if len(all_tokens) >= 8:
            scale_en = float(all_tokens[7])
        if len(all_tokens) >= 9:
            lcar = float(all_tokens[8])
        if len(all_tokens) >= 10:
            r1 = float(all_tokens[9])
        if len(all_tokens) >= 11:
            r2 = float(all_tokens[10])

    model.ebcs_pres[ebcs_id] = EbcsPres(
        id=ebcs_id, title=title, surf_id=surf_id, c=c,
        fct_pres=fct_pres, scale_pres=scale_pres,
        fct_rho=fct_rho, scale_rho=scale_rho,
        fct_en=fct_en, scale_en=scale_en,
        lcar=lcar, r1=r1, r2=r2,
    )




def read_ebcs_vel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/VEL/id`` (M150): Eulerian imposed velocity boundary condition."""
    from ...model.entities import EbcsVel
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/VEL/{ebcs_id}: missing data card", block.source)
        return
    surf_id = 0
    c = 0.0
    fct_vx, scale_vx = 0, 0.0
    fct_vy, scale_vy = 0, 0.0
    fct_vz, scale_vz = 0, 0.0
    fct_rho, scale_rho = 0, 1.0
    fct_en, scale_en = 0, 1.0
    lcar, r1, r2 = 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("EBCS_VEL_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("EBCS_VEL_2")
            c = _fval(f2[0]) if len(f2) > 0 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("EBCS_VEL_3")
            fct_vx = _ival(f3[0]) if len(f3) > 0 else 0
            scale_vx = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("EBCS_VEL_4")
            fct_vy = _ival(f4[0]) if len(f4) > 0 else 0
            scale_vy = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("EBCS_VEL_5")
            fct_vz = _ival(f5[0]) if len(f5) > 0 else 0
            scale_vz = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("EBCS_VEL_6")
            fct_rho = _ival(f6[0]) if len(f6) > 0 else 0
            scale_rho = _fval(f6[1], 1.0) if len(f6) > 1 else 1.0
        if len(cards) > 6 and not cards[6].is_blank:
            f7 = cards[7].cut("EBCS_VEL_7") if len(cards) > 6 else []
            fct_en = _ival(f7[0]) if len(f7) > 0 else 0
            scale_en = _fval(f7[1], 1.0) if len(f7) > 1 else 1.0
        if len(cards) > 7 and not cards[7].is_blank:
            f8 = cards[7].cut("EBCS_VEL_8")
            lcar = _fval(f8[0]) if len(f8) > 0 else 0.0
            r1 = _fval(f8[1]) if len(f8) > 1 else 0.0
            r2 = _fval(f8[2]) if len(f8) > 2 else 0.0
    else:
        all_tokens = []
        for card in cards:
            if not card.is_blank:
                all_tokens.extend(card.tokens())
        if len(all_tokens) >= 1:
            surf_id = int(float(all_tokens[0]))
        if len(all_tokens) >= 2:
            c = float(all_tokens[1])
        if len(all_tokens) >= 3:
            fct_vx = int(float(all_tokens[2]))
        if len(all_tokens) >= 4:
            scale_vx = float(all_tokens[3])
        if len(all_tokens) >= 5:
            fct_vy = int(float(all_tokens[4]))
        if len(all_tokens) >= 6:
            scale_vy = float(all_tokens[5])
        if len(all_tokens) >= 7:
            fct_vz = int(float(all_tokens[6]))
        if len(all_tokens) >= 8:
            scale_vz = float(all_tokens[7])
        if len(all_tokens) >= 9:
            fct_rho = int(float(all_tokens[8]))
        if len(all_tokens) >= 10:
            scale_rho = float(all_tokens[9])
        if len(all_tokens) >= 11:
            fct_en = int(float(all_tokens[10]))
        if len(all_tokens) >= 12:
            scale_en = float(all_tokens[11])
        if len(all_tokens) >= 13:
            lcar = float(all_tokens[12])
        if len(all_tokens) >= 14:
            r1 = float(all_tokens[13])
        if len(all_tokens) >= 15:
            r2 = float(all_tokens[14])

    model.ebcs_vel[ebcs_id] = EbcsVel(
        id=ebcs_id, title=title, surf_id=surf_id, c=c,
        fct_vx=fct_vx, scale_vx=scale_vx,
        fct_vy=fct_vy, scale_vy=scale_vy,
        fct_vz=fct_vz, scale_vz=scale_vz,
        fct_rho=fct_rho, scale_rho=scale_rho,
        fct_en=fct_en, scale_en=scale_en,
        lcar=lcar, r1=r1, r2=r2,
    )




def read_ebcs_inlet(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/INLET/id`` (M150): Eulerian inflow boundary condition."""
    from ...model.entities import EbcsInlet
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/INLET/{ebcs_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("EBCS_INLET_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        rho = _fval(f[1]) if len(f) > 1 else 0.0
        vx = _fval(f[2]) if len(f) > 2 else 0.0
        vy = _fval(f[3]) if len(f) > 3 else 0.0
        vz = _fval(f[4]) if len(f) > 4 else 0.0
        en = _fval(f[5]) if len(f) > 5 else 0.0
        fct_id = _ival(f[6]) if len(f) > 6 else 0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        rho = float(toks[1]) if len(toks) > 1 else 0.0
        vx = float(toks[2]) if len(toks) > 2 else 0.0
        vy = float(toks[3]) if len(toks) > 3 else 0.0
        vz = float(toks[4]) if len(toks) > 4 else 0.0
        en = float(toks[5]) if len(toks) > 5 else 0.0
        fct_id = int(float(toks[6])) if len(toks) > 6 else 0
    model.ebcs_inlets[ebcs_id] = EbcsInlet(
        id=ebcs_id, title=title, surf_id=surf_id,
        density=rho, vx=vx, vy=vy, vz=vz, energy=en, fct_id=fct_id
    )




def read_ebcs_fluxout(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/FLUXOUT/id`` (M150): Eulerian mass outflow boundary condition."""
    from ...model.entities import EbcsFluxout
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/FLUXOUT/{ebcs_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("EBCS_FLUXOUT_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        p_ext = _fval(f[1]) if len(f) > 1 else 0.0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        p_ext = float(toks[1]) if len(toks) > 1 else 0.0
    model.ebcs_fluxouts[ebcs_id] = EbcsFluxout(
        id=ebcs_id, title=title, surf_id=surf_id, p_ext=p_ext
    )




def read_ebcs_gradp0(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/GRADP0/id`` (M150): Eulerian zero pressure gradient boundary condition."""
    from ...model.entities import EbcsGradp0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/GRADP0/{ebcs_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("EBCS_GRADP0_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
    model.ebcs_gradp0[ebcs_id] = EbcsGradp0(
        id=ebcs_id, title=title, surf_id=surf_id
    )




def read_ebcs_normv(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/NORMV/id`` (M150): Eulerian normal velocity constraint."""
    from ...model.entities import EbcsNormv
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/NORMV/{ebcs_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("EBCS_NORMV_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        vn = _fval(f[1]) if len(f) > 1 else 0.0
        fct_id = _ival(f[2]) if len(f) > 2 else 0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        vn = float(toks[1]) if len(toks) > 1 else 0.0
        fct_id = int(float(toks[2])) if len(toks) > 2 else 0
    model.ebcs_normv[ebcs_id] = EbcsNormv(
        id=ebcs_id, title=title, surf_id=surf_id, vn=vn, fct_id=fct_id
    )




def read_ebcs_valv(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/VALVIN`` or ``/EBCS/VALVOUT`` (M150): Eulerian valve boundary condition."""
    from ...model.entities import EbcsValv
    sub = block.parts[1].upper() if len(block.parts) > 1 else "VALVIN"
    is_out = "OUT" in sub
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/{sub}/{ebcs_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("EBCS_VALVIN_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        p_open = _fval(f[1]) if len(f) > 1 else 0.0
        p_close = _fval(f[2]) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        p_open = float(toks[1]) if len(toks) > 1 else 0.0
        p_close = float(toks[2]) if len(toks) > 2 else 0.0
    model.ebcs_valves[ebcs_id] = EbcsValv(
        id=ebcs_id, title=title, surf_id=surf_id, is_out=is_out,
        p_open=p_open, p_close=p_close
    )




def read_ebcs_monvol(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/MONVOL/id`` or ``/MONVOL/id`` (M150, M178): Eulerian monitored volume boundary connection."""
    from ...model.entities import EbcsMonvol
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/MONVOL/{ebcs_id}: missing data card", block.source)
        return
    surf_id = 0
    sens_id = 0
    monvol_id = 0
    fscale = 1.0
    if block.fixed:
        f = cards[0].cut("EBCS_MONVOL_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        if len(f) > 2 and f[2].strip():
            sens_id = _ival(f[1]) if len(f) > 1 else 0
            monvol_id = _ival(f[2]) if len(f) > 2 else 0
            fscale = _fval(f[3], 1.0) if len(f) > 3 and f[3].strip() else 1.0
        else:
            monvol_id = _ival(f[1]) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        if len(toks) >= 3:
            sens_id = int(float(toks[1])) if len(toks) > 1 else 0
            monvol_id = int(float(toks[2])) if len(toks) > 2 else 0
            fscale = float(toks[3]) if len(toks) > 3 else 1.0
        else:
            monvol_id = int(float(toks[1])) if len(toks) > 1 else 0
    model.ebcs_monvols[ebcs_id] = EbcsMonvol(
        id=ebcs_id, title=title, surf_id=surf_id, sens_id=sens_id, monvol_id=monvol_id, fscale=fscale
    )




def read_ebcs_inip(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/INIP/id`` (M152): Eulerian initial pressure boundary condition."""
    from ...model.entities import EbcsInip
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/INIP/{ebcs_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("EBCS_INIP_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        rho = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        c = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        lcar = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        rho = float(toks[1]) if len(toks) > 1 else 0.0
        c = float(toks[2]) if len(toks) > 2 else 0.0
        lcar = float(toks[3]) if len(toks) > 3 else 0.0
    model.ebcs_inips[ebcs_id] = EbcsInip(
        id=ebcs_id, title=title, surf_id=surf_id, rho=rho, c=c, lcar=lcar
    )




def read_ebcs_iniv(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/INIV/id`` (M152): Eulerian initial velocity boundary condition."""
    from ...model.entities import EbcsIniv
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    ebcs_id = block.user_id if block.user_id is not None else 1
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/INIV/{ebcs_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("EBCS_INIV_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        rho = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        c = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        lcar = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        rho = float(toks[1]) if len(toks) > 1 else 0.0
        c = float(toks[2]) if len(toks) > 2 else 0.0
        lcar = float(toks[3]) if len(toks) > 3 else 0.0
    model.ebcs_inivs[ebcs_id] = EbcsIniv(
        id=ebcs_id, title=title, surf_id=surf_id, rho=rho, c=c, lcar=lcar
    )




def read_ebcs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/<subtype>/id`` dispatcher (M114, M125, M138, M150)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "PROPELLANT":
        read_ebcs_propellant(block, model, log)
    elif sub in ("NRF", "NON_REFLECT", "NONREFLECT"):
        read_ebcs_nrf(block, model, log)
    elif sub == "PERIODIC":
        read_ebcs_periodic(block, model, log)
    elif sub == "CYCLIC":
        read_ebcs_cyclic(block, model, log)
    elif sub == "INIP":
        read_ebcs_inip(block, model, log)
    elif sub == "INIV":
        read_ebcs_iniv(block, model, log)
    elif sub == "PRES":
        read_ebcs_pres(block, model, log)
    elif sub == "VEL":
        read_ebcs_vel(block, model, log)
    elif sub == "INLET":
        read_ebcs_inlet(block, model, log)
    elif sub in ("FLUXOUT", "OUTLET"):
        read_ebcs_fluxout(block, model, log)
    elif sub == "GRADP0":
        read_ebcs_gradp0(block, model, log)
    elif sub == "NORMV":
        read_ebcs_normv(block, model, log)
    elif sub in ("VALVIN", "VALVOUT", "VALV_IN", "VALV_OUT"):
        read_ebcs_valv(block, model, log)
    elif sub == "MONVOL":
        read_ebcs_monvol(block, model, log)
    else:
        read_bcs(block, model, log)




# ============================================================================
# Boundary conditions & joints & special initial states (M102)
# ============================================================================

def read_bcs_nrf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BCS/NRF/id`` or ``/BCS_NRF/id`` (M102, M200): non-reflecting boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_nrf.F90`` and ``bcs_nrf.cfg``.
    Card 1: TITLE (%-100s)
    Card 2: grnod_ID, Iskep, frame_ID, Isurf, Ivel, Isub, Ityp, factor
            (%10d%10d%10d%10d%10d%10d%10d%20f)
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    grnod_id, iskep, frame_id, isurf, ivel, isub, ityp = 0, 0, 0, 0, 0, 0, 0
    factor = 0.0
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("BCS_NRF_1")
            grnod_id = _ival(f[0]) if len(f) > 0 else 0
            iskep = _ival(f[1]) if len(f) > 1 else 0
            frame_id = _ival(f[2]) if len(f) > 2 else 0
            isurf = _ival(f[3]) if len(f) > 3 else 0
            ivel = _ival(f[4]) if len(f) > 4 else 0
            isub = _ival(f[5]) if len(f) > 5 else 0
            ityp = _ival(f[6]) if len(f) > 6 else 0
            factor = _fval(f[7], 0.0) if len(f) > 7 else 0.0
        else:
            toks = cards[0].tokens()
            grnod_id = _ival(toks[0]) if len(toks) > 0 else 0
            iskep = _ival(toks[1]) if len(toks) > 1 else 0
            frame_id = _ival(toks[2]) if len(toks) > 2 else 0
            isurf = _ival(toks[3]) if len(toks) > 3 else 0
            ivel = _ival(toks[4]) if len(toks) > 4 else 0
            isub = _ival(toks[5]) if len(toks) > 5 else 0
            ityp = _ival(toks[6]) if len(toks) > 6 else 0
            factor = _fval(toks[7], 0.0) if len(toks) > 7 else 0.0
    bcs_obj = BcsNrf(
        id=block.user_id,
        title=title,
        grnod_id=grnod_id,
        set_id=grnod_id,
        iskep=iskep,
        frame_id=frame_id,
        isurf=isurf,
        ivel=ivel,
        isub=isub,
        ityp=ityp,
        factor=factor,
    )
    model.bcs_nrfs[block.user_id] = bcs_obj
    if hasattr(model, "bcs_nrf"):
        model.bcs_nrf[block.user_id] = bcs_obj
    if len(cards) > 1 and not cards[1].is_blank:
        from ...model.entities import EbcsNrf
        if block.fixed:
            f1 = cards[1].cut("EBCS_NRF_2")
            tcar_p = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
            tcar_vf = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        else:
            t1 = cards[1].tokens()
            tcar_p = _fval(t1[0], 0.0) if len(t1) > 0 else 0.0
            tcar_vf = _fval(t1[1], 0.0) if len(t1) > 1 else 0.0
        model.ebcs_nrfs[block.user_id] = EbcsNrf(
            id=block.user_id, title=title, surf_id=grnod_id, tcar_p=tcar_p, tcar_vf=tcar_vf
        )




def read_bcs_wall(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BCS/WALL/id`` (M102, M150): sliding wall boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_wall.F90``.
    Card 1: TITLE (%-100s)
    Card 2: grnod_ID, sensor_ID (%10d%10d)
    Card 3: Tstart, Tstop (%20lg%20lg)
    """
    from ...model.entities import BcsWall
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    grnod_id = 0
    sensor_id = 0
    tstart = 0.0
    tstop = 1.0e20
    if cards:
        c1 = cards[0]
        if block.fixed:
            f1 = c1.cut("BCS_WALL_1")
            grnod_id = _ival(f1[0]) if len(f1) > 0 else 0
            sensor_id = _ival(f1[1]) if len(f1) > 1 else 0
        else:
            toks1 = c1.tokens()
            grnod_id = int(float(toks1[0])) if len(toks1) > 0 else 0
            sensor_id = int(float(toks1[1])) if len(toks1) > 1 else 0
    if len(cards) > 1:
        c2 = cards[1]
        if block.fixed:
            f2 = c2.cut("BCS_WALL_2")
            tstart = _fval(f2[0]) if len(f2) > 0 else 0.0
            tstop = _fval(f2[1], 1.0e20) if len(f2) > 1 else 1.0e20
        else:
            toks2 = c2.tokens()
            tstart = float(toks2[0]) if len(toks2) > 0 else 0.0
            tstop = float(toks2[1]) if len(toks2) > 1 else 1.0e20
    model.bcs_walls[block.user_id] = BcsWall(
        id=block.user_id, title=title, set_id=grnod_id, sensor_id=sensor_id,
        tstart=tstart, tstop=tstop
    )




def read_bcs_cyclic(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BCS/CYCLIC/id`` or ``/CYCLIC/id`` (M178): cyclic symmetry boundary condition.

    Fortran origin: ``starter/source/loads/bcs_cyclic.cfg`` / ``hm_read_bcs.F``.
    Card 1: TITLE (%-100s)
    Card 2: skew_ID, grnd_ID1, grnd_ID2 (%10d%10d%10d)
    """
    from ...model.entities import BcsCyclic
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    skew_id, grnd_id1, grnd_id2 = 0, 0, 0
    if cards:
        c1 = cards[0]
        if block.fixed:
            f1 = c1.cut("BCS_CYCLIC")
            skew_id = _ival(f1[0]) if len(f1) > 0 else 0
            grnd_id1 = _ival(f1[1]) if len(f1) > 1 else 0
            grnd_id2 = _ival(f1[2]) if len(f1) > 2 else 0
        else:
            toks1 = c1.tokens()
            skew_id = int(float(toks1[0])) if len(toks1) > 0 else 0
            grnd_id1 = int(float(toks1[1])) if len(toks1) > 1 else 0
            grnd_id2 = int(float(toks1[2])) if len(toks1) > 2 else 0
    model.bcs_cyclics[block.user_id] = BcsCyclic(
        id=block.user_id, skew_id=skew_id, grnd_id1=grnd_id1, grnd_id2=grnd_id2, title=title
    )
    from ...model.entities import CyclicBoundaryCondition
    model.cyclic_bcs[block.user_id] = CyclicBoundaryCondition(
        id=block.user_id, title=title, skew_id=skew_id, grnod1_id=grnd_id1, grnod2_id=grnd_id2
    )




def read_inicrack(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INICRACK/id`` (M102, M143): initial crack definition.

    Fortran origin: ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F``.
    """
    from ...model.entities import IniCrack, IniCrackSegment
    cid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    segments = []
    grsh_id = 0
    p1 = [0.0, 0.0, 0.0]
    p2 = [0.0, 0.0, 0.0]
    norm = [0.0, 0.0, 0.0]
    open_flag = 0

    if cards:
        t0 = cards[0].tokens()
        if len(t0) <= 2 and len(cards) >= 4:
            grsh_id = int(float(t0[0])) if len(t0) > 0 else 0
            open_flag = int(float(t0[1])) if len(t0) > 1 else 0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                if len(t1) >= 3:
                    p1 = [float(x) for x in t1[:3]]
            if len(cards) > 2 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                if len(t2) >= 3:
                    p2 = [float(x) for x in t2[:3]]
            if len(cards) > 3 and not cards[3].is_blank:
                t3 = cards[3].tokens()
                if len(t3) >= 3:
                    norm = [float(x) for x in t3[:3]]
        else:
            start_idx = 0
            if len(t0) == 1:
                grsh_id = int(float(t0[0]))
                start_idx = 1
            for c in cards[start_idx:]:
                if c.is_blank:
                    continue
                if block.fixed:
                    f = c.cut("INICRACK_ITEM")
                    if len(f) >= 2 and any(f):
                        n1 = _ival(f[0])
                        n2 = _ival(f[1])
                        rat = _fval(f[2]) if len(f) > 2 else 0.0
                        segments.append(IniCrackSegment(node_id1=n1, node_id2=n2, ratio=rat))
                else:
                    toks = c.tokens()
                    if len(toks) >= 2:
                        n1 = int(float(toks[0]))
                        n2 = int(float(toks[1]))
                        rat = float(toks[2]) if len(toks) > 2 else 0.0
                        segments.append(IniCrackSegment(node_id1=n1, node_id2=n2, ratio=rat))
    model.inicracks[cid] = IniCrack(
        id=cid, title=title, segments=segments, grsh_id=grsh_id,
        p1=p1, p2=p2, norm=norm, open_flag=open_flag
    )
    model.ini_cracks[cid] = model.inicracks[cid]




def read_inigrav(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIGRAV/inigrav_ID`` (M104, M151)::

        card 1:  title
        card 2:  grpart_ID  surf_ID  grav_ID  [gap]  Pref  Bx  By  Bz
        (or 2 cards: card 2: grpart_ID surf_ID grav_ID; card 3: Pref Bx By Bz)
    """
    inigrav_id = block.user_id if block.user_id is not None else 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIGRAV/{inigrav_id}: missing data card", block.source)
        return

    grpart_id = 0
    surf_id = 0
    grav_id = 0
    pref = 0.0
    bx = 0.0
    by = 0.0
    bz = 0.0

    if block.fixed:
        if len(cards) >= 2:
            f1 = cards[0].cut("INIGRAV_1_SHORT")
            grpart_id = _ival(f1[0]) if len(f1) > 0 else 0
            surf_id = _ival(f1[1]) if len(f1) > 1 else 0
            grav_id = _ival(f1[2]) if len(f1) > 2 else 0
            f2 = cards[1].cut("INIGRAV_2")
            pref = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            bx = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            by = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            bz = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            f = cards[0].cut("INIGRAV_1")
            grpart_id = _ival(f[0]) if len(f) > 0 else 0
            surf_id = _ival(f[1]) if len(f) > 1 else 0
            grav_id = _ival(f[2]) if len(f) > 2 else 0
            pref = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            bx = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            by = _fval(f[6], 0.0) if len(f) > 6 else 0.0
            bz = _fval(f[7], 0.0) if len(f) > 7 else 0.0
    else:
        if len(cards) >= 2:
            t1 = cards[0].tokens()
            grpart_id = int(float(t1[0])) if len(t1) > 0 else 0
            surf_id = int(float(t1[1])) if len(t1) > 1 else 0
            grav_id = int(float(t1[2])) if len(t1) > 2 else 0
            t2 = cards[1].tokens()
            pref = float(t2[0]) if len(t2) > 0 else 0.0
            bx = float(t2[1]) if len(t2) > 1 else 0.0
            by = float(t2[2]) if len(t2) > 2 else 0.0
            bz = float(t2[3]) if len(t2) > 3 else 0.0
        else:
            toks = cards[0].tokens()
            grpart_id = int(float(toks[0])) if len(toks) > 0 else 0
            surf_id = int(float(toks[1])) if len(toks) > 1 else 0
            grav_id = int(float(toks[2])) if len(toks) > 2 else 0
            pref = float(toks[3]) if len(toks) > 3 else 0.0
            bx = float(toks[4]) if len(toks) > 4 else 0.0
            by = float(toks[5]) if len(toks) > 5 else 0.0
            bz = float(toks[6]) if len(toks) > 6 else 0.0

    model.ini_gravs[inigrav_id] = IniGrav(
        id=inigrav_id, title=title, grpart_id=grpart_id, surf_id=surf_id,
        grav_id=grav_id, pref=pref, bx=bx, by=by, bz=bz,
    )
    model.inigrav_loads[inigrav_id] = model.ini_gravs[inigrav_id]




def read_inimap(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIMAP/1D``, ``/INIMAP/2D``, or ``/INIMAP/3D`` dispatcher (M104, M137, M167)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "1D" or "1D" in block.parts[0].upper():
        read_inimap1d(block, model, log)
    elif sub == "2D" or "2D" in block.parts[0].upper():
        read_inimap2d(block, model, log)
    elif sub == "3D" or "3D" in block.parts[0].upper():
        read_inimap3d(block, model, log)
    else:
        log.warning(f"/INIMAP/{sub} not ported (1D, 2D, 3D supported)", block.source)




def read_inimap1d(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIMAP1D[/<formulation>]/map_ID`` or ``/INIMAP/1D[/<formulation>]/map_ID`` (M104, M167)::

        card 1:  title
        card 2:  type  node_ID1  node_ID2  grbric_ID  grquad_ID  grsh3n_ID  Fscale_V
        if FILE:
            card 3:  filename
        if VP or VE:
            card 3:  FUN_IDV  FSCALEV
            card 4:  Nb_integr
            cards 5+: fct_Idvfi  fct_IDri  Fscalerhoi  fct_IDpei  Fscalepei
    """
    map_id = block.user_id if block.user_id is not None else 1
    formulation = "FILE"
    for part in block.parts:
        p_up = part.upper()
        if p_up in ("VP", "VE", "FILE"):
            formulation = p_up
            break

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIMAP1D/{map_id}: missing data card", block.source)
        return

    filename = ""
    func_vel = 0
    fac_vel = 1.0
    nb_mat = 0
    func_alpha = []
    func_rho = []
    func_pres_ener = []
    fac_rho = []
    fac_pres_ener = []

    if block.fixed:
        f = cards[0].cut("INIMAP1D_1")
        map_type = _ival(f[0]) if len(f) > 0 else 0
        n1 = _ival(f[1]) if len(f) > 1 else 0
        n2 = _ival(f[2]) if len(f) > 2 else 0
        grbric = _ival(f[3]) if len(f) > 3 else 0
        grquad = _ival(f[4]) if len(f) > 4 else 0
        grsh3n = _ival(f[5]) if len(f) > 5 else 0
        fscale_v = _fval(f[6], 1.0) if len(f) > 6 else 1.0

        if formulation == "FILE" and len(cards) > 1:
            filename = cards[1].raw.strip()
        elif formulation in ("VP", "VE") and len(cards) > 1:
            fv = cards[1].cut("INIMAP1D_VP_1")
            func_vel = _ival(fv[0]) if len(fv) > 0 else 0
            fac_vel = _fval(fv[1], 1.0) if len(fv) > 1 else 1.0

            if len(cards) > 2:
                fnb = cards[2].cut("INIMAP1D_VP_2")
                nb_mat = _ival(fnb[0]) if len(fnb) > 0 else 0
                for c in cards[3:3 + nb_mat]:
                    if c.is_blank:
                        continue
                    fm = c.cut("INIMAP1D_VP_3")
                    func_alpha.append(_ival(fm[0]) if len(fm) > 0 else 0)
                    func_rho.append(_ival(fm[1]) if len(fm) > 1 else 0)
                    fac_rho.append(_fval(fm[2], 1.0) if len(fm) > 2 else 1.0)
                    func_pres_ener.append(_ival(fm[3]) if len(fm) > 3 else 0)
                    fac_pres_ener.append(_fval(fm[4], 1.0) if len(fm) > 4 else 1.0)
    else:
        toks = cards[0].tokens()
        map_type = int(float(toks[0])) if len(toks) > 0 else 0
        n1 = int(float(toks[1])) if len(toks) > 1 else 0
        n2 = int(float(toks[2])) if len(toks) > 2 else 0
        grbric = int(float(toks[3])) if len(toks) > 3 else 0
        grquad = int(float(toks[4])) if len(toks) > 4 else 0
        grsh3n = int(float(toks[5])) if len(toks) > 5 else 0
        fscale_v = float(toks[6]) if len(toks) > 6 else 1.0

        if formulation == "FILE" and len(cards) > 1:
            filename = cards[1].raw.strip()
        elif formulation in ("VP", "VE") and len(cards) > 1:
            tv = cards[1].tokens()
            func_vel = int(float(tv[0])) if len(tv) > 0 else 0
            fac_vel = float(tv[1]) if len(tv) > 1 else 1.0

            if len(cards) > 2:
                tnb = cards[2].tokens()
                nb_mat = int(float(tnb[0])) if len(tnb) > 0 else 0
                for c in cards[3:3 + nb_mat]:
                    if c.is_blank:
                        continue
                    tm = c.tokens()
                    func_alpha.append(int(float(tm[0])) if len(tm) > 0 else 0)
                    func_rho.append(int(float(tm[1])) if len(tm) > 1 else 0)
                    fac_rho.append(float(tm[2]) if len(tm) > 2 else 1.0)
                    func_pres_ener.append(int(float(tm[3])) if len(tm) > 3 else 0)
                    fac_pres_ener.append(float(tm[4]) if len(tm) > 4 else 1.0)

    model.ini_map1ds[map_id] = IniMap1D(
        id=map_id, title=title, formulation=formulation, map_type=map_type,
        node_id1=n1, node_id2=n2, grbric_id=grbric, grquad_id=grquad,
        grsh3n_id=grsh3n, fscale_v=fscale_v, func_vel=func_vel, fac_vel=fac_vel,
        nb_mat=nb_mat, func_alpha=func_alpha, func_rho=func_rho,
        func_pres_ener=func_pres_ener, fac_rho=fac_rho,
        fac_pres_ener=fac_pres_ener, filename=filename,
    )




def read_inimap2d(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIMAP2D[/<formulation>]/map_ID`` or ``/INIMAP/2D[/<formulation>]/map_ID`` (M104, M167)::

        card 1:  title
        card 2:  type  node_ID1  node_ID2  node_ID3  grbric_ID  Fscale_V
        if FILE:
            card 3:  filename
        if VP or VE:
            card 3:  FUN_IDV  FSCALEV
            card 4:  Nb_integr
            cards 5+: fct_Idvfi  fct_IDri  Fscalerhoi  fct_IDpei  Fscalepei
    """
    map_id = block.user_id if block.user_id is not None else 1
    formulation = "FILE"
    for part in block.parts:
        p_up = part.upper()
        if p_up in ("VP", "VE", "FILE"):
            formulation = p_up
            break

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIMAP2D/{map_id}: missing data card", block.source)
        return

    filename = ""
    func_vel = 0
    fac_vel = 1.0
    nb_mat = 0
    func_alpha = []
    func_rho = []
    func_pres_ener = []
    fac_rho = []
    fac_pres_ener = []

    if block.fixed:
        f = cards[0].cut("INIMAP2D_1")
        map_type = _ival(f[0]) if len(f) > 0 else 0
        n1 = _ival(f[1]) if len(f) > 1 else 0
        n2 = _ival(f[2]) if len(f) > 2 else 0
        n3 = _ival(f[3]) if len(f) > 3 else 0
        grbric = _ival(f[4]) if len(f) > 4 else 0
        fscale_v = _fval(f[5], 1.0) if len(f) > 5 else 1.0

        if formulation == "FILE" and len(cards) > 1:
            filename = cards[1].raw.strip()
        elif formulation in ("VP", "VE") and len(cards) > 1:
            fv = cards[1].cut("INIMAP1D_VP_1")
            func_vel = _ival(fv[0]) if len(fv) > 0 else 0
            fac_vel = _fval(fv[1], 1.0) if len(fv) > 1 else 1.0

            if len(cards) > 2:
                fnb = cards[2].cut("INIMAP1D_VP_2")
                nb_mat = _ival(fnb[0]) if len(fnb) > 0 else 0
                for c in cards[3:3 + nb_mat]:
                    if c.is_blank:
                        continue
                    fm = c.cut("INIMAP1D_VP_3")
                    func_alpha.append(_ival(fm[0]) if len(fm) > 0 else 0)
                    func_rho.append(_ival(fm[1]) if len(fm) > 1 else 0)
                    fac_rho.append(_fval(fm[2], 1.0) if len(fm) > 2 else 1.0)
                    func_pres_ener.append(_ival(fm[3]) if len(fm) > 3 else 0)
                    fac_pres_ener.append(_fval(fm[4], 1.0) if len(fm) > 4 else 1.0)
    else:
        toks = cards[0].tokens()
        map_type = int(float(toks[0])) if len(toks) > 0 else 0
        n1 = int(float(toks[1])) if len(toks) > 1 else 0
        n2 = int(float(toks[2])) if len(toks) > 2 else 0
        n3 = int(float(toks[3])) if len(toks) > 3 else 0
        grbric = int(float(toks[4])) if len(toks) > 4 else 0
        fscale_v = float(toks[5]) if len(toks) > 5 else 1.0

        if formulation == "FILE" and len(cards) > 1:
            filename = cards[1].raw.strip()
        elif formulation in ("VP", "VE") and len(cards) > 1:
            tv = cards[1].tokens()
            func_vel = int(float(tv[0])) if len(tv) > 0 else 0
            fac_vel = float(tv[1]) if len(tv) > 1 else 1.0

            if len(cards) > 2:
                tnb = cards[2].tokens()
                nb_mat = int(float(tnb[0])) if len(tnb) > 0 else 0
                for c in cards[3:3 + nb_mat]:
                    if c.is_blank:
                        continue
                    tm = c.tokens()
                    func_alpha.append(int(float(tm[0])) if len(tm) > 0 else 0)
                    func_rho.append(int(float(tm[1])) if len(tm) > 0 else 0)
                    fac_rho.append(float(tm[2]) if len(tm) > 2 else 1.0)
                    func_pres_ener.append(int(float(tm[3])) if len(tm) > 3 else 0)
                    fac_pres_ener.append(float(tm[4]) if len(tm) > 4 else 1.0)

    model.ini_map2ds[map_id] = IniMap2D(
        id=map_id, title=title, formulation=formulation, map_type=map_type,
        node_id1=n1, node_id2=n2, node_id3=n3, grbric_id=grbric,
        fscale_v=fscale_v, func_vel=func_vel, fac_vel=fac_vel,
        nb_mat=nb_mat, func_alpha=func_alpha, func_rho=func_rho,
        func_pres_ener=func_pres_ener, fac_rho=fac_rho,
        fac_pres_ener=fac_pres_ener, filename=filename,
    )




def read_inimap3d(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIMAP3D/map_ID`` or ``/INIMAP/3D/map_ID`` (M137)::

        card 1:  title
        card 2:  type  grbric_ID  grquad_ID  grsh3n_ID  skew_ID  Fscale_V
        card 3:  filename
    """
    map_id = block.user_id if block.user_id is not None else 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIMAP3D/{map_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("INIMAP3D_1")
        map_type = _ival(f[0]) if len(f) > 0 else 0
        grbric = _ival(f[1]) if len(f) > 1 else 0
        grquad = _ival(f[2]) if len(f) > 2 else 0
        grsh3n = _ival(f[3]) if len(f) > 3 else 0
        fscale_v = _fval(f[5], 1.0) if len(f) > 5 else 1.0
    else:
        toks = cards[0].tokens()
        map_type = int(float(toks[0])) if len(toks) > 0 else 0
        grbric = int(float(toks[1])) if len(toks) > 1 else 0
        grquad = int(float(toks[2])) if len(toks) > 2 else 0
        grsh3n = int(float(toks[3])) if len(toks) > 3 else 0
        fscale_v = float(toks[5]) if len(toks) > 5 else 1.0

    filename = cards[1].raw.strip() if len(cards) > 1 else ""

    model.ini_map3ds[map_id] = IniMap3D(
        id=map_id, title=title, map_type=map_type,
        grbric_id=grbric, grquad_id=grquad, grsh3n_id=grsh3n,
        filename=filename, fscale_v=fscale_v,
    )




def read_inista(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISTA``, ``/INISTATE``, ``/INISTATE/FILE`` (M104/M151), or ``/INISTA/<elem_type>/...`` (M116)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub in ("FILE", "") or sub.isdigit():
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            cards = [c for c in block.cards if not c.is_blank]
        if not cards or cards[0].is_blank:
            log.error("/INISTATE: missing data card", block.source)
            return

        inista_id = block.user_id if block.user_id is not None else 1
        filename = ""
        ibal = 1
        ioutyy = 0
        ioutynn = 0

        if block.fixed:
            # Case A: title extracted the filename (e.g. /INISTATE/FILE without title card)
            # and cards[0] contains the integer parameters
            first_raw = cards[0].raw.strip()
            first_toks = first_raw.split()
            if title and all(tok.lstrip("+-").isdigit() for tok in first_toks):
                filename = title
                f = cards[0].cut("INISTATE_1") if len(cards[0].raw) <= 20 else cards[0].cut("INISTA_1")
                ibal = _ival(f[0], 1) if len(f) > 0 else 1
                ioutyy = _ival(f[1], 0) if len(f) > 1 else 0
                ioutynn = _ival(f[2], 0) if len(f) > 2 else 0
            elif len(cards[0].raw) > 80:
                f = cards[0].cut("INISTA_1")
                filename = f[0].strip() if len(f) > 0 else ""
                ibal = _ival(f[1], 1) if len(f) > 1 else 1
                ioutyy = _ival(f[2], 0) if len(f) > 2 else 0
                ioutynn = _ival(f[3], 0) if len(f) > 3 else 0
            elif len(cards) > 1 and not cards[1].is_blank:
                filename = cards[0].raw.strip()
                f = cards[1].cut("INISTATE_1")
                ibal = _ival(f[0], 1) if len(f) > 0 else 1
                ioutyy = _ival(f[1], 0) if len(f) > 1 else 0
            else:
                f = cards[0].cut("INISTA_1")
                filename = f[0].strip() if len(f) > 0 else ""
                ibal = _ival(f[1], 1) if len(f) > 1 else 1
                ioutyy = _ival(f[2], 0) if len(f) > 2 else 0
                ioutynn = _ival(f[3], 0) if len(f) > 3 else 0
        else:
            tokens = cards[0].tokens()
            if len(tokens) >= 2 and not tokens[0].lstrip("+-").isdigit():
                filename = tokens[0]
                ibal = int(float(tokens[1])) if len(tokens) > 1 else 1
                ioutyy = int(float(tokens[2])) if len(tokens) > 2 else 0
                ioutynn = int(float(tokens[3])) if len(tokens) > 3 else 0
            elif title and all(tok.lstrip("+-").isdigit() for tok in tokens):
                filename = title
                ibal = int(float(tokens[0])) if len(tokens) > 0 else 1
                ioutyy = int(float(tokens[1])) if len(tokens) > 1 else 0
                ioutynn = int(float(tokens[2])) if len(tokens) > 2 else 0
            else:
                filename = cards[0].raw.strip()
                if len(cards) > 1 and not cards[1].is_blank:
                    toks = cards[1].tokens()
                    ibal = int(float(toks[0])) if len(toks) > 0 else 1
                    ioutyy = int(float(toks[1])) if len(toks) > 1 else 0

        from ...model.entities import IniStateFile, Inista
        model.ini_state_file = IniStateFile(filename=filename, isigi=ibal, ioutp_fmt=ioutyy)
        model.inistas[inista_id] = Inista(
            id=inista_id, title=title, filename=filename,
            ibal=ibal, ioutyy=ioutyy, ioutynn=ioutynn,
        )
        return

    if sub in ("SHE", "SHEL", "SHELL"):
        mod_block = KeywordBlock(
            keyword="/".join(["INISHE"] + block.parts[2:]),
            parts=["INISHE"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inishe(mod_block, model, log)
    elif sub in ("BRI", "BRIC", "BRICK", "SOLID"):
        mod_block = KeywordBlock(
            keyword="/".join(["INIBRI"] + block.parts[2:]),
            parts=["INIBRI"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inibri(mod_block, model, log)
    elif sub in ("SH3", "SH3N", "TRIA"):
        mod_block = KeywordBlock(
            keyword="/".join(["INISH3"] + block.parts[2:]),
            parts=["INISH3"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inish3(mod_block, model, log)
    elif sub in ("TRU", "TRUS", "TRUSS"):
        mod_block = KeywordBlock(
            keyword="/".join(["INITRU"] + block.parts[2:]),
            parts=["INITRU"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_initru(mod_block, model, log)
    elif sub in ("BEA", "BEAM"):
        mod_block = KeywordBlock(
            keyword="/".join(["INIBEA"] + block.parts[2:]),
            parts=["INIBEA"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inibea(mod_block, model, log)
    elif sub in ("SPR", "SPRI", "SPRING"):
        mod_block = KeywordBlock(
            keyword="/".join(["INISPR"] + block.parts[2:]),
            parts=["INISPR"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inispr(mod_block, model, log)
    elif sub in ("QUA", "QUAD"):
        mod_block = KeywordBlock(
            keyword="/".join(["INIQUA"] + block.parts[2:]),
            parts=["INIQUA"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_iniqua(mod_block, model, log)
    elif sub in ("PART", "PART_STRESS"):
        part_id = None
        if block.user_id is not None:
            part_id = block.user_id
        elif len(block.parts) > 2 and block.parts[2].isdigit():
            part_id = int(block.parts[2])

        from ...starter.inista import InistaRecord
        valid_cards = [c for c in block.cards if not c.is_blank and not c.raw.strip().startswith("#")]
        if not valid_cards:
            log.error(f"/INISTA/{sub}: missing data card", block.source)
            return

        c_idx = 0
        while c_idx < len(valid_cards):
            c = valid_cards[c_idx]
            toks = c.tokens() if not block.fixed or len(c.tokens()) > 1 else [f for f in c.fields(10) if f]
            if not toks:
                c_idx += 1
                continue

            cur_part = part_id
            t_vals = toks
            if cur_part is None:
                try:
                    cur_part = int(float(toks[0]))
                    t_vals = toks[1:]
                except ValueError:
                    c_idx += 1
                    continue

            vals = []
            for v in t_vals:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(0.0)

            sxx = vals[0] if len(vals) > 0 else 0.0
            syy = vals[1] if len(vals) > 1 else 0.0
            szz = vals[2] if len(vals) > 2 else 0.0
            sxy = vals[3] if len(vals) > 3 else 0.0
            syz = vals[4] if len(vals) > 4 else 0.0
            szx = vals[5] if len(vals) > 5 else 0.0
            epsp = vals[6] if len(vals) > 6 else 0.0

            sb_xx, sb_yy, sb_xy = 0.0, 0.0, 0.0
            c_idx += 1
            if c_idx < len(valid_cards):
                next_c = valid_cards[c_idx]
                next_toks = next_c.tokens() if not block.fixed or len(next_c.tokens()) > 1 else [f for f in next_c.fields(10) if f]
                if part_id is not None and len(next_toks) in (1, 2, 3):
                    b_vals = []
                    for v in next_toks:
                        try:
                            b_vals.append(float(v))
                        except ValueError:
                            b_vals.append(0.0)
                    sb_xx = b_vals[0] if len(b_vals) > 0 else 0.0
                    sb_yy = b_vals[1] if len(b_vals) > 1 else 0.0
                    sb_xy = b_vals[2] if len(b_vals) > 2 else 0.0
                    c_idx += 1

            rec = InistaRecord(
                part_id=cur_part,
                sigma_xx=sxx, sigma_yy=syy, sigma_zz=szz,
                sigma_xy=sxy, sigma_yz=syz, sigma_zx=szx,
                sigma_b_xx=sb_xx, sigma_b_yy=sb_yy, sigma_b_xy=sb_xy,
                epsp=epsp,
            )
            model.inista_records.append(rec)
        return

    elif sub in ("STRESS", "STRS"):
        from ...starter.inista import InistaRecord
        valid_cards = [c for c in block.cards if not c.is_blank and not c.raw.strip().startswith("#")]
        for c in valid_cards:
            toks = c.tokens() if not block.fixed or len(c.tokens()) > 1 else [f for f in c.fields(10) if f]
            if not toks:
                continue
            try:
                elem_id = int(float(toks[0]))
            except ValueError:
                continue
            vals = []
            for v in toks[1:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(0.0)
            sxx = vals[0] if len(vals) > 0 else 0.0
            syy = vals[1] if len(vals) > 1 else 0.0
            szz = vals[2] if len(vals) > 2 else 0.0
            sxy = vals[3] if len(vals) > 3 else 0.0
            syz = vals[4] if len(vals) > 4 else 0.0
            szx = vals[5] if len(vals) > 5 else 0.0
            epsp = vals[6] if len(vals) > 6 else None
            model.inista_records.append(InistaRecord(
                elem_id=elem_id,
                sigma_xx=sxx, sigma_yy=syy, sigma_zz=szz,
                sigma_xy=sxy, sigma_yz=syz, sigma_zx=szx,
                epsp=epsp,
            ))
        return

    elif sub in ("EPSP",):
        from ...starter.inista import InistaRecord
        valid_cards = [c for c in block.cards if not c.is_blank and not c.raw.strip().startswith("#")]
        for c in valid_cards:
            toks = c.tokens() if not block.fixed or len(c.tokens()) > 1 else [f for f in c.fields(10) if f]
            if not toks:
                continue
            try:
                elem_id = int(float(toks[0]))
                epsp = float(toks[1]) if len(toks) > 1 else 0.0
            except ValueError:
                continue
            model.inista_records.append(InistaRecord(
                elem_id=elem_id,
                epsp=epsp,
            ))
        return

    else:
        read_inishe(block, model, log)




def read_inibri_eref(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIBRI/EREF`` (M107): Initial brick reference element state."""
    sub_objects = []
    for card in block.cards:
        if card.raw.strip():
            toks = card.tokens()
            if len(toks) >= 2:
                try:
                    e1, e2 = int(toks[0]), int(toks[1])
                    sub_objects.append({"elem_id": e1, "ref_elem_id": e2})
                except ValueError:
                    sub_objects.append({"raw": card.raw.strip()})
            else:
                sub_objects.append({"raw": card.raw.strip()})
    model.inibri_erefs.append(IniBriEref(sub_objects=sub_objects))




def read_inivol(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVOL/[part_ID/]inivol_ID`` (M94/M151)::

        card 1 (optional title):  title
        card 2 (or 1): part_ID  NIP (or directly containers)
        cards 3+: surf_ID  ale_phase  fill_opt  icumu  fill_ratio
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    inivol_id = block.user_id if block.user_id is not None else 1
    part_id = 0
    if block.unit_id is not None:
        part_id, inivol_id = block.user_id or 0, block.unit_id

    containers: List[InivolContainer] = []
    non_blank = [c for c in cards if not c.is_blank] if cards else []
    if not non_blank:
        return

    start_idx = 0
    first_tokens = non_blank[0].tokens()
    if len(non_blank) > 1 and len(first_tokens) <= 2:
        try:
            part_id = int(float(first_tokens[0]))
            start_idx = 1
        except ValueError:
            pass

    for c in non_blank[start_idx:]:
        if block.fixed:
            f = c.cut("INIVOL")
            surf_id = _ival(f[0])
            ale_phase = _ival(f[1], default=1) if len(f) > 1 else 1
            fill_opt = _ival(f[2], default=0) if len(f) > 2 else 0
            icumu = _ival(f[3], default=0) if len(f) > 3 else 0
            fill_ratio = _fval(f[4], default=1.0) if len(f) > 4 else 1.0
        else:
            t = c.tokens()
            if not t:
                continue
            surf_id = int(float(t[0]))
            ale_phase = int(float(t[1])) if len(t) > 1 else 1
            fill_opt = int(float(t[2])) if len(t) > 2 else 0
            icumu = int(float(t[3])) if len(t) > 3 else 0
            fill_ratio = float(t[4]) if len(t) > 4 else 1.0

        if fill_ratio == 0.0:
            fill_ratio = 1.0

        containers.append(InivolContainer(
            surf_id=surf_id, ale_phase=ale_phase, fill_opt=fill_opt,
            icumu=icumu, fill_ratio=fill_ratio,
        ))

    iv = InitialVolume(id=inivol_id, part_id=part_id, title=title, containers=containers)
    model.inivol.append(iv)
    model.inivols[inivol_id] = iv




def read_initemp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INITEMP/initemp_ID`` (M95)::

        card 1:  title
        card 2:  T0  grnd_ID  [fld_type]
        cards 3+ (if fld_type==1): T0i  node_IDi
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INITEMP/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("INITEMP_1")
        t0 = _fval(f[0], 0.0)
        grnod_id = _ival(f[1]) if len(f) > 1 else 0
        fld_type = _ival(f[2], default=0) if len(f) > 2 else 0
    else:
        t = cards[0].tokens()
        t0 = float(t[0]) if len(t) > 0 else 0.0
        grnod_id = int(t[1]) if len(t) > 1 else 0
        fld_type = int(t[2]) if len(t) > 2 else 0

    nodal_temps: Dict[int, float] = {}
    if fld_type == 1 and len(cards) > 1:
        for c in cards[1:]:
            if c.is_blank:
                continue
            if block.fixed:
                sub = c.cut("INITEMP_SUB")
                t_val = _fval(sub[0], 0.0)
                n_id = _ival(sub[1]) if len(sub) > 1 else 0
            else:
                st = c.tokens()
                if not st:
                    continue
                t_val = float(st[0])
                n_id = int(st[1]) if len(st) > 1 else 0
            if n_id:
                nodal_temps[n_id] = t_val

    it = InitialTemperature(
        id=block.user_id, t0=t0, grnod_id=grnod_id,
        fld_type=fld_type, nodal_temps=nodal_temps, title=title,
    )
    model.initemp.append(it)
    model.initemps[block.user_id] = it




def read_inibri(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIBRI/{STRESS|EPSP|DENS|ENER|STRA_F|FAIL|AUX}[/id]`` (M96, M142, M195)::"""
    sub = block.parts[1].upper() if len(block.parts) > 1 else "STRESS"
    if sub.isdigit():
        sub = "STRESS"
    _, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INIBRI/{sub}: missing data card", block.source)
        return

    table_key = f"INIBRI_{sub}_{block.user_id or 1}"
    rows = []

    if sub in ("STRESS", "STRS_F", "STRS_FGLO", "STRS"):
        idx = 0
        while idx < len(cards):
            c1 = cards[idx]
            if block.fixed:
                f = c1.cut("INIBRI_STRESS_1")
                elem_id = _ival(f[0])
                s1 = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                s2 = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                s3 = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                idx += 1
                if idx < len(cards):
                    g = cards[idx].cut("INIBRI_STRESS_2")
                    s12 = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                    s23 = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                    s31 = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                    idx += 1
                else:
                    s12, s23, s31 = 0.0, 0.0, 0.0
            else:
                t = c1.tokens()
                elem_id = int(float(t[0]))
                if len(t) >= 7:
                    s1, s2, s3, s12, s23, s31 = [float(x) for x in t[1:7]]
                    idx += 1
                else:
                    s1 = float(t[1]) if len(t) > 1 else 0.0
                    s2 = float(t[2]) if len(t) > 2 else 0.0
                    s3 = float(t[3]) if len(t) > 3 else 0.0
                    idx += 1
                    if idx < len(cards):
                        t2 = cards[idx].tokens()
                        s12 = float(t2[0]) if len(t2) > 0 else 0.0
                        s23 = float(t2[1]) if len(t2) > 1 else 0.0
                        s31 = float(t2[2]) if len(t2) > 2 else 0.0
                        idx += 1
                    else:
                        s12, s23, s31 = 0.0, 0.0, 0.0

            st = model.ini_bricks.setdefault(elem_id, InitialBrickState(elem_id=elem_id))
            st.sigma = np.array([s1, s2, s3, s12, s23, s31], dtype=float)
            rows.append({
                "ELEM_ID": elem_id,
                "SIG_XX": s1, "SIG_YY": s2, "SIG_ZZ": s3,
                "SIG_XY": s12, "SIG_YZ": s23, "SIG_ZX": s31
            })

    elif sub in ("STRA_F", "STRA_FGLO", "STRA"):
        idx = 0
        while idx < len(cards):
            c1 = cards[idx]
            if block.fixed:
                f = c1.cut("INIBRI_STRA_1")
                elem_id = _ival(f[0])
                e1 = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                e2 = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                e3 = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                idx += 1
                if idx < len(cards):
                    g = cards[idx].cut("INIBRI_STRA_2")
                    e12 = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                    e23 = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                    e31 = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                    idx += 1
                else:
                    e12, e23, e31 = 0.0, 0.0, 0.0
            else:
                t = c1.tokens()
                elem_id = int(float(t[0]))
                if len(t) >= 7:
                    e1, e2, e3, e12, e23, e31 = [float(x) for x in t[1:7]]
                    idx += 1
                else:
                    e1 = float(t[1]) if len(t) > 1 else 0.0
                    e2 = float(t[2]) if len(t) > 2 else 0.0
                    e3 = float(t[3]) if len(t) > 3 else 0.0
                    idx += 1
                    if idx < len(cards):
                        t2 = cards[idx].tokens()
                        e12 = float(t2[0]) if len(t2) > 0 else 0.0
                        e23 = float(t2[1]) if len(t2) > 1 else 0.0
                        e31 = float(t2[2]) if len(t2) > 2 else 0.0
                        idx += 1
                    else:
                        e12, e23, e31 = 0.0, 0.0, 0.0

            st = model.ini_bricks.setdefault(elem_id, InitialBrickState(elem_id=elem_id))
            st.eps = np.array([e1, e2, e3, e12, e23, e31], dtype=float)
            rows.append({
                "ELEM_ID": elem_id,
                "EPS_XX": e1, "EPS_YY": e2, "EPS_ZZ": e3,
                "EPS_XY": e12, "EPS_YZ": e23, "EPS_ZX": e31
            })

    elif sub in ("EPSP", "DENS", "ENER", "TEMP", "PRES", "VOID", "FAIL", "AUX", "SCALE_YLD", "SCALE"):
        for c in cards:
            if block.fixed:
                f = c.cut("INIBRI_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_bricks.setdefault(elem_id, InitialBrickState(elem_id=elem_id))
            if sub == "EPSP":
                st.epsp = val
            elif sub == "DENS":
                st.rho = val
            elif sub == "ENER":
                st.ener = val
            elif sub == "TEMP":
                st.temp = val
            elif sub == "PRES":
                st.pres = val
            elif sub == "VOID":
                st.void = val
            elif sub == "FAIL":
                st.fail_flag = val
            elif sub == "AUX":
                st.aux = val
            elif sub in ("SCALE_YLD", "SCALE"):
                st.scale_yld = val
            rows.append({"ELEM_ID": elem_id, "VAL": val})
    elif sub == "EREF":
        read_inibri_eref(block, model, log)
    else:
        log.warning(f"/INIBRI/{sub} not ported — block skipped", block.source)

    if rows:
        from ...model.entities import IniStateTable
        model.ini_state_tables[table_key] = IniStateTable(
            keyword=f"/INIBRI/{sub}", id=block.user_id or 1, rows=rows
        )




def read_inishe(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISHE/{STRS_F|EPSP|THICK|STRA_F|EPSP_F|FAIL|AUX}[/id]`` (M96, M142, M195)::

        /INISHE/EPSP, /INISHE/THICK, /INISHE/FAIL, /INISHE/AUX:
          card 1: shell_ID  value
        /INISHE/STRS_F:
          card 1: shell_ID  nb_integr  npg  Thick
          card 2: Em  Eb  H1  H2  H3
          card 3: sigma_1  sigma_2  sigma_12  sigma_23  sigma_31
          card 4: eps_p  sigma_b1  sigma_b2  sigma_b12
        /INISHE/STRA_F:
          card 1: shell_ID  eps_1  eps_2  eps_12
        /INISHE/EPSP_F:
          card 1: shell_ID  epsp_1  epsp_2 ... (per layer)
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "STRS_F"
    if sub.isdigit():
        sub = "STRS_F"
    _, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INISHE/{sub}: missing data card", block.source)
        return

    table_key = f"INISHE_{sub}_{block.user_id or 1}"
    rows = []

    if sub in ("EPSP", "THICK", "TEMP", "ENER", "FAIL", "AUX", "SCALE_YLD", "SCALE", "DENS"):
        for c in cards:
            if block.fixed:
                f = c.cut("INISHE_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_shells.setdefault(elem_id, InitialShellState(elem_id=elem_id))
            if sub == "EPSP":
                st.epsp = val
            elif sub == "THICK":
                st.thick = val
            elif sub == "TEMP":
                st.temp = val
            elif sub == "DENS":
                st.rho = val
            elif sub == "ENER":
                st.em = val
            elif sub == "FAIL":
                st.fail_flag = val
            elif sub == "AUX":
                st.aux = val
            elif sub in ("SCALE_YLD", "SCALE"):
                st.scale_yld = val

            row_dict = {"ELEM_ID": elem_id, "VAL": val}
            if not block.fixed and len(c.tokens()) >= 3:
                row_dict["IFAIL"] = int(float(c.tokens()[1]))
                row_dict["VAL"] = float(c.tokens()[2])
            rows.append(row_dict)

    elif sub in ("EPSP_F", "EPSPF"):
        for c in cards:
            if c.is_blank:
                continue
            toks = c.tokens()
            if not toks:
                continue
            elem_id = int(float(toks[0]))
            layers = [float(x) for x in toks[1:]]
            st = model.ini_shells.setdefault(elem_id, InitialShellState(elem_id=elem_id))
            st.epsp_layers = layers
            rows.append({"ELEM_ID": elem_id, "LAYERS": layers})

    elif sub in ("STRA_F", "STRA", "STRA_FGLO", "STRA_F_GLOB", "STRA_F_GLO"):
        for c in cards:
            if c.is_blank:
                continue
            toks = c.tokens()
            if len(toks) >= 4:
                elem_id = int(float(toks[0]))
                e1 = float(toks[1])
                e2 = float(toks[2])
                e12 = float(toks[3])
                st = model.ini_shells.setdefault(elem_id, InitialShellState(elem_id=elem_id))
                st.eps = np.array([e1, e2, 0.0, e12, 0.0, 0.0], dtype=float)
                row_dict = {
                    "ELEM_ID": elem_id,
                    "EPS_XX": e1, "EPS_YY": e2, "EPS_ZZ": float(toks[3]) if len(toks) >= 7 else 0.0,
                    "EPS_XY": float(toks[4]) if len(toks) >= 7 else e12,
                    "EPS_YZ": float(toks[5]) if len(toks) >= 7 else 0.0,
                    "EPS_ZX": float(toks[6]) if len(toks) >= 7 else 0.0,
                }
                rows.append(row_dict)

    elif sub in ("STRS_F", "STRS_FGLO", "STRS_F/GLOB", "STRS_F_GLOB"):
        idx = 0
        while idx < len(cards):
            c0 = cards[idx]
            if block.fixed:
                f = c0.cut("INISHE_STRS_1")
                elem_id = _ival(f[0])
                thick = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                idx += 1

                g = cards[idx].cut("INISHE_STRS_2") if idx < len(cards) else []
                em = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                eb = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                h1 = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                h2 = _fval(g[3], 0.0) if len(g) > 3 else 0.0
                h3 = _fval(g[4], 0.0) if len(g) > 4 else 0.0
                idx += 1

                h = cards[idx].cut("INISHE_STRS_3") if idx < len(cards) else []
                s1 = _fval(h[0], 0.0) if len(h) > 0 else 0.0
                s2 = _fval(h[1], 0.0) if len(h) > 1 else 0.0
                s12 = _fval(h[2], 0.0) if len(h) > 2 else 0.0
                s23 = _fval(h[3], 0.0) if len(h) > 3 else 0.0
                s31 = _fval(h[4], 0.0) if len(h) > 4 else 0.0
                idx += 1

                k = cards[idx].cut("INISHE_STRS_4") if idx < len(cards) else []
                epsp = _fval(k[0], 0.0) if len(k) > 0 else 0.0
                sb1 = _fval(k[1], 0.0) if len(k) > 1 else 0.0
                sb2 = _fval(k[2], 0.0) if len(k) > 2 else 0.0
                sb12 = _fval(k[3], 0.0) if len(k) > 3 else 0.0
                idx += 1
            else:
                t0 = c0.tokens()
                elem_id = int(float(t0[0]))
                thick = float(t0[3]) if len(t0) > 3 else (float(t0[2]) if len(t0) > 2 else 0.0)
                idx += 1

                t1 = cards[idx].tokens() if idx < len(cards) else []
                em = float(t1[0]) if len(t1) > 0 else 0.0
                eb = float(t1[1]) if len(t1) > 1 else 0.0
                h1 = float(t1[2]) if len(t1) > 2 else 0.0
                h2 = float(t1[3]) if len(t1) > 3 else 0.0
                h3 = float(t1[4]) if len(t1) > 4 else 0.0
                idx += 1

                t2 = cards[idx].tokens() if idx < len(cards) else []
                s1 = float(t2[0]) if len(t2) > 0 else 0.0
                s2 = float(t2[1]) if len(t2) > 1 else 0.0
                s12 = float(t2[2]) if len(t2) > 2 else 0.0
                s23 = float(t2[3]) if len(t2) > 3 else 0.0
                s31 = float(t2[4]) if len(t2) > 4 else 0.0
                idx += 1

                t3 = cards[idx].tokens() if idx < len(cards) else []
                epsp = float(t3[0]) if len(t3) > 0 else 0.0
                sb1 = float(t3[1]) if len(t3) > 1 else 0.0
                sb2 = float(t3[2]) if len(t3) > 2 else 0.0
                sb12 = float(t3[3]) if len(t3) > 3 else 0.0
                idx += 1

            st = model.ini_shells.setdefault(elem_id, InitialShellState(elem_id=elem_id))
            st.thick = thick
            st.em = em
            st.eb = eb
            st.h_energy = np.array([h1, h2, h3], dtype=float)
            st.sigma = np.array([s1, s2, 0.0, s12, s23, s31], dtype=float)
            st.sigma_b = np.array([sb1, sb2, 0.0, sb12, 0.0, 0.0], dtype=float)
            st.epsp = epsp
            rows.append({
                "ELEM_ID": elem_id,
                "SIG_XX": s1, "SIG_YY": s2, "SIG_XY": s12, "SIG_YZ": s23, "SIG_ZX": s31,
                "THICK": thick, "EPSP": epsp
            })

    elif sub in ("ORTH_LOC", "ORTH_LOC_F", "ORTH"):
        idx = 0
        while idx < len(cards):
            c0 = cards[idx]
            if c0.is_blank:
                idx += 1
                continue
            if block.fixed:
                f0 = c0.cut("INISHE_ORTH_LOC_1")
                elem_id = _ival(f0[0])
                nb_lay = _ival(f0[1]) if len(f0) > 1 else 1
                npg = _ival(f0[2]) if len(f0) > 2 else 1
                ndir = _ival(f0[3]) if len(f0) > 3 else 1
                iunit = _ival(f0[4]) if len(f0) > 4 else 0
            else:
                t0 = c0.tokens()
                elem_id = int(float(t0[0]))
                nb_lay = int(float(t0[1])) if len(t0) > 1 else 1
                npg = int(float(t0[2])) if len(t0) > 2 else 1
                ndir = int(float(t0[3])) if len(t0) > 3 else 1
                iunit = int(float(t0[4])) if len(t0) > 4 else 0
            idx += 1

            phi_list = []
            alpha_list = []
            angles_list = []
            for _ in range(nb_lay):
                if idx >= len(cards):
                    break
                clay = cards[idx]
                idx += 1
                if clay.is_blank:
                    continue
                if block.fixed:
                    fl = clay.cut("INISHE_ORTH_LOC_2")
                    phi = _fval(fl[0], 0.0) if len(fl) > 0 else 0.0
                    alpha = _fval(fl[1], 0.0) if len(fl) > 1 else 0.0
                else:
                    tl = clay.tokens()
                    phi = float(tl[0]) if len(tl) > 0 else 0.0
                    alpha = float(tl[1]) if len(tl) > 1 else 0.0
                phi_list.append(phi)
                alpha_list.append(alpha)
                angles_list.append((phi, alpha))

            st = model.ini_shells.setdefault(elem_id, InitialShellState(elem_id=elem_id))
            st.orth_angles = angles_list
            st.orth_phi = phi_list
            st.orth_alpha = alpha_list
            rows.append({
                "ELEM_ID": elem_id,
                "NB_LAY": nb_lay,
                "NPG": npg,
                "NDIR": ndir,
                "IUNIT": iunit,
                "PHI": phi_list,
                "ALPHA": alpha_list,
            })
    else:
        log.warning(f"/INISHE/{sub} not ported — block skipped", block.source)


    if rows:
        from ...model.entities import IniStateTable
        model.ini_state_tables[table_key] = IniStateTable(
            keyword=f"/INISHE/{sub}", id=block.user_id or 1, rows=rows
        )




def read_inish3(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISH3/{STRS_F|EPSP|THICK|FAIL}[/id]`` (M96, M195):: initial state for 3-node shells."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else "STRS_F"
    if sub.isdigit():
        sub = "STRS_F"
    table_key = f"INISH3_{sub}_{block.user_id or 1}"
    read_inishe(block, model, log)
    she_key = f"INISHE_{sub}_{block.user_id or 1}"
    if she_key in model.ini_state_tables:
        model.ini_state_tables[table_key] = model.ini_state_tables[she_key]




def read_initru(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INITRU/{FULL|EPSP|FORCE|TENS}[/id]`` (M97)::

        /INITRU/FULL:
          card 1: truss_ID  prop_type  EINT  FOR  AREA  EPSP
        /INITRU/EPSP, /INITRU/FORCE, /INITRU/TENS:
          card 1: truss_ID  value
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "FULL"
    if sub.isdigit():
        sub = "FULL"
    title, cards = _title_and_data(block)
    cards = [c for c in cards if not c.is_blank]
    if not cards:
        log.error(f"/INITRU/{sub}: missing data card", block.source)
        return

    if sub in ("EPSP", "FORCE", "TENS", "TEMP", "STRA", "STRA_F"):
        for c in cards:
            if block.fixed:
                f = c.cut("INITRU_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_trusses.setdefault(elem_id, InitialTrussState(elem_id=elem_id))
            if sub == "EPSP":
                st.epsp = val
            elif sub in ("FORCE", "TENS"):
                st.force = val
            elif sub == "TEMP":
                st.temp = val
            elif sub.startswith("STRA"):
                st.epsp = val
    elif sub in ("FULL", "TRUSS"):
        for c in cards:
            if block.fixed:
                f = c.cut("INITRU_FULL")
                elem_id = _ival(f[0])
                ptype = _ival(f[1], 2) if len(f) > 1 else 2
                eint = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                force = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                area = _fval(f[4], 0.0) if len(f) > 4 else 0.0
                epsp = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                ptype = int(float(t[1])) if len(t) > 1 else 2
                eint = float(t[2]) if len(t) > 2 else 0.0
                force = float(t[3]) if len(t) > 3 else 0.0
                area = float(t[4]) if len(t) > 4 else 0.0
                epsp = float(t[5]) if len(t) > 5 else 0.0

            st = model.ini_trusses.setdefault(elem_id, InitialTrussState(elem_id=elem_id))
            st.prop_type = ptype
            st.eint = eint
            st.force = force
            st.area = area
            st.epsp = epssp if 'epssp' in locals() else epsp
    else:
        log.warning(f"/INITRU/{sub} not ported — block skipped", block.source)




def read_inibea(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIBEA/{FULL|FORCE|MOMENT|EPSP|STRA|STRA_F}[/id]`` (M97, M197)::

        /INIBEA/FORCE, /INIBEA/MOMENT, /INIBEA/EPSP, /INIBEA/STRA_F:
          card 1: beam_ID  value
        /INIBEA/FULL:
          card 1: beam_ID  nb_integr  prop_type
          card 2: EImemb  EIbend  F1  F2  F3  M1  M2  M3
          card 3: EpsilonP
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "FULL"
    if sub.isdigit():
        sub = "FULL"
    title, cards = _title_and_data(block)
    cards = [c for c in cards if not c.is_blank]
    if not cards:
        log.error(f"/INIBEA/{sub}: missing data card", block.source)
        return

    if sub in ("FORCE", "MOMENT", "EPSP", "TEMP", "STRA", "STRA_F"):
        for c in cards:
            if block.fixed:
                f = c.cut("INIBEA_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_beams.setdefault(elem_id, InitialBeamState(elem_id=elem_id))
            if sub == "FORCE":
                st.force[0] = val
            elif sub == "MOMENT":
                st.moment[0] = val
            elif sub == "EPSP":
                st.epsp = val
            elif sub == "TEMP":
                st.temp = val
            elif sub.startswith("STRA"):
                st.epsp = val
    elif sub in ("FULL", "BEAM"):
        idx = 0
        while idx < len(cards):
            c0 = cards[idx]
            if block.fixed:
                f = c0.cut("INIBEA_FULL_1")
                elem_id = _ival(f[0])
                nip = _ival(f[1], 0) if len(f) > 1 else 0
                ptype = _ival(f[2], 3) if len(f) > 2 else 3
                idx += 1

                g = cards[idx].cut("INIBEA_FULL_2") if idx < len(cards) else []
                eimemb = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                eibend = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                f1 = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                f2 = _fval(g[3], 0.0) if len(g) > 3 else 0.0
                f3 = _fval(g[4], 0.0) if len(g) > 4 else 0.0
                idx += 1

                h = cards[idx].cut("INIBEA_FULL_3") if idx < len(cards) else []
                m1 = _fval(h[0], 0.0) if len(h) > 0 else 0.0
                m2 = _fval(h[1], 0.0) if len(h) > 1 else 0.0
                m3 = _fval(h[2], 0.0) if len(h) > 2 else 0.0
                idx += 1

                k = cards[idx].cut("INIBEA_FULL_4") if idx < len(cards) else []
                epsp = _fval(k[0], 0.0) if len(k) > 0 else 0.0
                idx += 1
            else:
                t0 = c0.tokens()
                elem_id = int(float(t0[0]))
                nip = int(float(t0[1])) if len(t0) > 1 else 0
                ptype = int(float(t0[2])) if len(t0) > 2 else 3
                idx += 1

                t1 = cards[idx].tokens() if idx < len(cards) else []
                if len(t1) >= 8:
                    eimemb, eibend, f1, f2, f3, m1, m2, m3 = [float(x) for x in t1[:8]]
                    idx += 1
                else:
                    eimemb = float(t1[0]) if len(t1) > 0 else 0.0
                    eibend = float(t1[1]) if len(t1) > 1 else 0.0
                    f1 = float(t1[2]) if len(t1) > 2 else 0.0
                    f2 = float(t1[3]) if len(t1) > 3 else 0.0
                    f3 = float(t1[4]) if len(t1) > 4 else 0.0
                    idx += 1

                    t2 = cards[idx].tokens() if idx < len(cards) else []
                    m1 = float(t2[0]) if len(t2) > 0 else 0.0
                    m2 = float(t2[1]) if len(t2) > 1 else 0.0
                    m3 = float(t2[2]) if len(t2) > 2 else 0.0
                    idx += 1

                t3 = cards[idx].tokens() if idx < len(cards) else []
                epsp = float(t3[0]) if len(t3) > 0 else 0.0
                idx += 1

            st = model.ini_beams.setdefault(elem_id, InitialBeamState(elem_id=elem_id))
            st.prop_type = ptype
            st.nb_integr = nip
            st.eint_memb = eimemb
            st.eint_bend = eibend
            st.force = np.array([f1, f2, f3], dtype=float)
            st.moment = np.array([m1, m2, m3], dtype=float)
            st.epsp = epsp
    else:
        log.warning(f"/INIBEA/{sub} not ported — block skipped", block.source)




def read_inispr(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISPR/{FULL|DISP|FORCE}[/id]`` (M97)::

        /INISPR/DISP, /INISPR/FORCE:
          card 1: spring_ID  value
        /INISPR/FULL:
          card 1: spring_ID  prop_type  nvars
          card 2: F_X  D_X  FEP_X  DPL_XP  DPL_XM
          card 3: L_X  EI
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "FULL"
    if sub.isdigit():
        sub = "FULL"
    title, cards = _title_and_data(block)
    cards = [c for c in cards if not c.is_blank]
    if not cards:
        log.error(f"/INISPR/{sub}: missing data card", block.source)
        return

    if sub in ("DISP", "FORCE", "TEMP"):
        for c in cards:
            if block.fixed:
                f = c.cut("INISPR_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_springs.setdefault(elem_id, InitialSpringState(elem_id=elem_id))
            if sub == "DISP":
                st.disp = val
            elif sub == "FORCE":
                st.force = val
            elif sub == "TEMP":
                st.temp = val
    elif sub in ("FULL", "SPRING"):
        idx = 0
        while idx < len(cards):
            c0 = cards[idx]
            if block.fixed:
                f = c0.cut("INISPR_FULL_1")
                elem_id = _ival(f[0])
                ptype = _ival(f[1], 4) if len(f) > 1 else 4
                idx += 1

                g = cards[idx].cut("INISPR_FULL_2") if idx < len(cards) else []
                fx = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                dx = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                fep = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                dpl_pos = _fval(g[3], 0.0) if len(g) > 3 else 0.0
                dpl_neg = _fval(g[4], 0.0) if len(g) > 4 else 0.0
                idx += 1

                h = cards[idx].cut("INISPR_FULL_3") if idx < len(cards) else []
                lx = _fval(h[0], 0.0) if len(h) > 0 else 0.0
                ei = _fval(h[1], 0.0) if len(h) > 1 else 0.0
                idx += 1
            else:
                t0 = c0.tokens()
                elem_id = int(float(t0[0]))
                ptype = int(float(t0[1])) if len(t0) > 1 else 4
                idx += 1

                t1 = cards[idx].tokens() if idx < len(cards) else []
                fx = float(t1[0]) if len(t1) > 0 else 0.0
                dx = float(t1[1]) if len(t1) > 1 else 0.0
                fep = float(t1[2]) if len(t1) > 2 else 0.0
                dpl_pos = float(t1[3]) if len(t1) > 3 else 0.0
                dpl_neg = float(t1[4]) if len(t1) > 4 else 0.0
                idx += 1

                t2 = cards[idx].tokens() if idx < len(cards) else []
                lx = float(t2[0]) if len(t2) > 0 else 0.0
                ei = float(t2[1]) if len(t2) > 1 else 0.0
                idx += 1

            st = model.ini_springs.setdefault(elem_id, InitialSpringState(elem_id=elem_id))
            st.prop_type = ptype
            st.force = fx
            st.disp = dx
            st.fep = fep
            st.dpl_pos = dpl_pos
            st.dpl_neg = dpl_neg
            st.length = lx
            st.eint = ei
    else:
        log.warning(f"/INISPR/{sub} not ported — block skipped", block.source)




def read_iniqua(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIQUA/{STRS_F|EPSP|DENS|ENER}[/id]`` (M116): Initial state for quadrilateral shell elements."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else "STRS_F"
    if sub.isdigit():
        sub = "STRS_F"
    table_key = f"INIQUA_{sub}_{block.user_id or 1}"
    read_inishe(block, model, log)
    she_key = f"INISHE_{sub}_{block.user_id or 1}"
    if she_key in model.ini_state_tables:
        model.ini_state_tables[table_key] = model.ini_state_tables[she_key]












def read_init(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIT/<subtype>/id`` dispatcher (M110, M140)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    import dataclasses
    if sub.startswith("DET") or sub in ("POINT", "LINE", "PLAN", "CORD"):
        read_det(block, model, log)
    elif sub.startswith("VEL"):
        parts = ["INIVEL"] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_inivel(b, model, log)
    elif sub.startswith("CRACK"):
        parts = ["INICRACK"] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_inicrack(b, model, log)
    elif sub in ("BRI", "BRIC", "SOLID"):
        parts = ["INIBRI"] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_inibri(b, model, log)
    elif sub in ("SHE", "SHEL", "SH3", "SH3N", "QUAD", "TRIA"):
        parts = ["INISHE"] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_inishe(b, model, log)
    elif sub in ("TRU", "TRUS", "TRUSS"):
        parts = ["INITRU"] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_initru(b, model, log)
    elif sub in ("BEA", "BEAM"):
        parts = ["INIBEA"] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_inibea(b, model, log)
    elif sub in ("SPR", "SPRI", "SPRING"):
        parts = ["INISPR"] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_inispr(b, model, log)
    elif sub.startswith("GRAV"):
        parts = ["GRAV"] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_grav(b, model, log)
    elif sub in ("TEMP", "PRES", "DENS", "EPSP", "ENER", "VOID"):
        parts = ["INIBRI", sub] + block.parts[2:]
        b = dataclasses.replace(block, parts=parts)
        read_inibri(b, model, log)
    else:
        log.warning(f"/INIT/{sub} not ported", block.source)




def read_inisphcel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISPHCEL/part_id`` (M145/M195): SPH cell initial state."""
    from ...model.entities import IniSphCel, IniStateTable
    part_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    p, rho, e, vx, vy, vz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("INISPHCEL_1")
            p = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            rho = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            e = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            vx = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            vy = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            vz = _fval(f[5], 0.0) if len(f) > 5 else 0.0
        else:
            t = cards[0].tokens()
            p = float(t[0]) if len(t) > 0 else 0.0
            rho = float(t[1]) if len(t) > 1 else 0.0
            e = float(t[2]) if len(t) > 2 else 0.0
            vx = float(t[3]) if len(t) > 3 else 0.0
            vy = float(t[4]) if len(t) > 4 else 0.0
            vz = float(t[5]) if len(t) > 5 else 0.0
    model.ini_sphcels[part_id] = IniSphCel(
        part_id=part_id, p=p, rho=rho, e=e, vx=vx, vy=vy, vz=vz
    )
    model.ini_state_tables[f"INISPHCEL_{part_id}"] = IniStateTable(
        keyword="/INISPHCEL", id=part_id, title=title,
        rows=[{"P": p, "RHO": rho, "E": e, "VX": vx, "VY": vy, "VZ": vz}]
    )
