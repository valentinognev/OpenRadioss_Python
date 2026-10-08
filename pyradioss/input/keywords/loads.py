# -*- coding: utf-8 -*-
"""
Force and pressure load readers - /KEYWORD blocks -> Model.

/CLOAD, /GRAV, /PLOAD, /LOAD/** and the thermal surface loads
(convection, radiation, flux, cylindrical and fluid pressure).

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





def read_heat(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HEAT/MAT/mat_ID`` (M37, M171); ``/HEAT/BCS/bcs_ID`` (M135): Thermal boundary conditions."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub in ("MAT", "MATERIAL"):
        read_mat_heat(block, model, log)
    elif sub == "BCS":
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/HEAT/BCS/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("HEAT_BCS_1")
            group_id = _ival(f[0]) if len(f) > 0 else 0
            bcs_type = f[1].strip() if len(f) > 1 and f[1].strip() else "TEMP"
            tval = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            funct_id = _ival(f[3], 0) if len(f) > 3 else 0
            scale = _fval(f[4], 1.0) if len(f) > 4 else 1.0
            sens_id = _ival(f[5], 0) if len(f) > 5 else 0
        else:
            t = cards[0].tokens()
            group_id = int(float(t[0])) if len(t) > 0 else 0
            bcs_type = t[1].upper() if len(t) > 1 else "TEMP"
            tval = float(t[2]) if len(t) > 2 else 0.0
            funct_id = int(float(t[3])) if len(t) > 3 else 0
            scale = float(t[4]) if len(t) > 4 else 1.0
            sens_id = int(float(t[5])) if len(t) > 5 else 0
        model.heat_bcs[block.user_id] = HeatBcs(
            id=block.user_id, title=title, group_id=group_id,
            bcs_type=bcs_type, tval=tval, funct_id=funct_id,
            scale=scale, sens_id=sens_id
        )
    elif sub in ("CONVEC", "CONVECTION"):
        read_convec(block, model, log)
    elif sub in ("RADIATION", "RAD"):
        read_radiation(block, model, log)
    elif sub in ("FLUX",):
        read_flux(block, model, log)
    elif sub in ("SOLVER", "GLOBAL", "INIT"):
        from ...model.entities import HeatSolver
        hid = block.user_id or 1
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        isolv, itype = 1, 1
        ttol = 1e-3
        dttmax = 1.0
        dttmin = 1e-6
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("HEAT_SOLVER_1")
                isolv = _ival(f[0], 1)
                itype = _ival(f[1], 1)
                ttol = _fval(f[2], 1e-3)
                dttmax = _fval(f[3], 1.0)
                dttmin = _fval(f[4], 1e-6)
            else:
                toks = cards[0].tokens()
                isolv = int(float(toks[0])) if len(toks) > 0 else 1
                itype = int(float(toks[1])) if len(toks) > 1 else 1
                ttol = float(toks[2]) if len(toks) > 2 else 1e-3
                dttmax = float(toks[3]) if len(toks) > 3 else 1.0
                dttmin = float(toks[4]) if len(toks) > 4 else 1e-6
        model.heat_solvers[hid] = HeatSolver(
            id=hid, title=title, isolv=isolv, itype=itype,
            ttol=ttol, dttmax=dttmax, dttmin=dttmin
        )
    elif sub in ("RAD_CAV", "CAV", "CAVITY", "RADCAV") or (len(block.parts) > 2 and block.parts[1].upper() in ("RAD_CAV", "CAV", "RADCAV", "RAD")):
        read_heat_rad_cav(block, model, log)
    else:
        log.warning(f"/HEAT/{sub} not ported", block.source)




def read_flux(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HEAT/FLUX/id`` or ``/FLUX/id`` (M202): Thermal surface heat flux."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    fid = block.user_id if block.user_id is not None else len(model.heat_fluxes) + 1
    surf_id = 0
    funct_id = 0
    sensor_id = 0
    ascale = 1.0
    fscale = 1.0
    tstart = 0.0
    tstop = 1.0e30
    q = 0.0
    if cards:
        if block.fixed:
            f1 = cards[0].cut("HEAT_FLUX_1") if "HEAT_FLUX_1" in CARD_LAYOUTS else cards[0].cut("FLUX_1") if "FLUX_1" in CARD_LAYOUTS else _fixed_vals(cards[0], [10, 10, 10, 20, 20])
            surf_id = _ival(f1[0]) if len(f1) > 0 else 0
            funct_id = _ival(f1[1]) if len(f1) > 1 else 0
            sensor_id = _ival(f1[2]) if len(f1) > 2 else 0
            if len(f1) > 3 and f1[3].strip():
                ascale = _fval(f1[3], 1.0)
            if len(f1) > 4 and f1[4].strip():
                fscale = _fval(f1[4], 1.0)
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("HEAT_FLUX_2") if "HEAT_FLUX_2" in CARD_LAYOUTS else cards[1].cut("FLUX_2") if "FLUX_2" in CARD_LAYOUTS else _fixed_vals(cards[1], [20, 20, 20, 20, 20])
                if len(f2) >= 5:
                    ascale = _fval(f2[0], 1.0)
                    fscale = _fval(f2[1], 1.0)
                    tstart = _fval(f2[2], 0.0)
                    tstop = _fval(f2[3], 1.0e30)
                    q = _fval(f2[4], 0.0)
                elif len(f2) == 4:
                    ascale = _fval(f2[0], 1.0)
                    fscale = _fval(f2[1], 1.0)
                    tstart = _fval(f2[2], 0.0)
                    tstop = _fval(f2[3], 1.0e30)
                elif len(f2) >= 3:
                    tstart = _fval(f2[0], 0.0) if len(f2) > 0 and f2[0].strip() else 0.0
                    tstop = _fval(f2[1], 1.0e30) if len(f2) > 1 and f2[1].strip() else 1.0e30
                    q = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
                elif len(f2) == 1:
                    q = _fval(f2[0], 0.0) if f2[0].strip() else 0.0
        else:
            t1 = cards[0].tokens()
            surf_id = int(float(t1[0])) if len(t1) > 0 else 0
            funct_id = int(float(t1[1])) if len(t1) > 1 else 0
            sensor_id = int(float(t1[2])) if len(t1) > 2 else 0
            if len(t1) > 3:
                ascale = float(t1[3])
            if len(t1) > 4:
                fscale = float(t1[4])
            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                if len(t2) >= 5:
                    ascale = float(t2[0])
                    fscale = float(t2[1])
                    tstart = float(t2[2])
                    tstop = float(t2[3])
                    q = float(t2[4])
                elif len(t2) == 4:
                    ascale = float(t2[0])
                    fscale = float(t2[1])
                    tstart = float(t2[2])
                    tstop = float(t2[3])
                elif len(t2) >= 3:
                    tstart = float(t2[0])
                    tstop = float(t2[1])
                    q = float(t2[2])
                elif len(t2) == 1:
                    q = float(t2[0])

    from ...model.entities import HeatFlux
    hf = HeatFlux(
        id=fid, title=title, surf_id=surf_id, funct_id=funct_id, sensor_id=sensor_id,
        ascale=ascale, fscale=fscale, tstart=tstart, tstop=tstop, q=q
    )
    model.heat_fluxes[fid] = hf




def read_convec_heat_m202(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HEAT/CONVEC/id``, ``/HEAT/CONVECTION/id`` or ``/CONVEC/id`` (M202): Thermal surface convection."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    cid = block.user_id if block.user_id is not None else len(model.heat_convecs) + 1
    surf_id = 0
    funct_id = 0
    sensor_id = 0
    ascale = 1.0
    fscale = 1.0
    tstart = 0.0
    tstop = 1.0e30
    h = 0.0
    if cards:
        if block.fixed:
            f1 = cards[0].cut("HEAT_CONVEC_1") if "HEAT_CONVEC_1" in CARD_LAYOUTS else cards[0].cut("CONVEC_1") if "CONVEC_1" in CARD_LAYOUTS else _fixed_vals(cards[0], [10, 10, 10, 20, 20])
            surf_id = _ival(f1[0]) if len(f1) > 0 else 0
            funct_id = _ival(f1[1]) if len(f1) > 1 else 0
            sensor_id = _ival(f1[2]) if len(f1) > 2 else 0
            if len(f1) > 3 and f1[3].strip():
                ascale = _fval(f1[3], 1.0)
            if len(f1) > 4 and f1[4].strip():
                fscale = _fval(f1[4], 1.0)
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("HEAT_CONVEC_2") if "HEAT_CONVEC_2" in CARD_LAYOUTS else cards[1].cut("CONVEC_2") if "CONVEC_2" in CARD_LAYOUTS else _fixed_vals(cards[1], [20, 20, 20, 20, 20])
                if len(f2) >= 5:
                    ascale = _fval(f2[0], 1.0)
                    fscale = _fval(f2[1], 1.0)
                    tstart = _fval(f2[2], 0.0)
                    tstop = _fval(f2[3], 1.0e30)
                    h = _fval(f2[4], 0.0)
                elif len(f2) == 3:
                    tstart = _fval(f2[0], 0.0)
                    tstop = _fval(f2[1], 1.0e30)
                    h = _fval(f2[2], 0.0)
                elif len(f2) == 1:
                    h = _fval(f2[0], 0.0)
        else:
            t1 = cards[0].tokens()
            surf_id = int(float(t1[0])) if len(t1) > 0 else 0
            funct_id = int(float(t1[1])) if len(t1) > 1 else 0
            sensor_id = int(float(t1[2])) if len(t1) > 2 else 0
            if len(t1) > 3:
                ascale = float(t1[3])
            if len(t1) > 4:
                fscale = float(t1[4])
            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                if len(t2) >= 5:
                    ascale = float(t2[0])
                    fscale = float(t2[1])
                    tstart = float(t2[2])
                    tstop = float(t2[3])
                    h = float(t2[4])
                elif len(t2) == 3:
                    tstart = float(t2[0])
                    tstop = float(t2[1])
                    h = float(t2[2])
                elif len(t2) == 1:
                    h = float(t2[0])

    from ...model.entities import HeatConvec, ConvectionLoad
    hc = HeatConvec(
        id=cid, title=title, surf_id=surf_id, funct_id=funct_id, sensor_id=sensor_id,
        ascale=ascale, fscale=fscale, tstart=tstart, tstop=tstop, h=h
    )
    model.heat_convecs[cid] = hc
    cl = ConvectionLoad(
        id=cid, surf_id=surf_id, funct_id=funct_id,
        sens_id=sensor_id, xscale=ascale, scale=fscale,
        tstart=tstart, tstop=tstop, h=h, title=title,
    )
    model.convec_loads.append(cl)




def read_radiation_heat_m202(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HEAT/RADIATION/id``, ``/HEAT/RAD/id`` or ``/RADIATION/id`` (M202): Thermal surface radiation."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    rid = block.user_id if block.user_id is not None else len(model.heat_radiations) + 1
    surf_id = 0
    funct_id = 0
    sensor_id = 0
    ascale = 1.0
    fscale = 1.0
    tstart = 0.0
    tstop = 1.0e30
    emiss = 0.0
    if cards:
        if block.fixed:
            f1 = cards[0].cut("HEAT_RADIATION_1") if "HEAT_RADIATION_1" in CARD_LAYOUTS else cards[0].cut("RADIATION_1") if "RADIATION_1" in CARD_LAYOUTS else _fixed_vals(cards[0], [10, 10, 10, 20, 20])
            surf_id = _ival(f1[0]) if len(f1) > 0 else 0
            funct_id = _ival(f1[1]) if len(f1) > 1 else 0
            sensor_id = _ival(f1[2]) if len(f1) > 2 else 0
            if len(f1) > 3 and f1[3].strip():
                ascale = _fval(f1[3], 1.0)
            if len(f1) > 4 and f1[4].strip():
                fscale = _fval(f1[4], 1.0)
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("HEAT_RADIATION_2") if "HEAT_RADIATION_2" in CARD_LAYOUTS else cards[1].cut("RADIATION_2") if "RADIATION_2" in CARD_LAYOUTS else _fixed_vals(cards[1], [20, 20, 20, 20, 20])
                if len(f2) >= 5:
                    ascale = _fval(f2[0], 1.0)
                    fscale = _fval(f2[1], 1.0)
                    tstart = _fval(f2[2], 0.0)
                    tstop = _fval(f2[3], 1.0e30)
                    emiss = _fval(f2[4], 0.0)
                elif len(f2) == 3:
                    tstart = _fval(f2[0], 0.0)
                    tstop = _fval(f2[1], 1.0e30)
                    emiss = _fval(f2[2], 0.0)
                elif len(f2) == 1:
                    emiss = _fval(f2[0], 0.0)
        else:
            t1 = cards[0].tokens()
            surf_id = int(float(t1[0])) if len(t1) > 0 else 0
            funct_id = int(float(t1[1])) if len(t1) > 1 else 0
            sensor_id = int(float(t1[2])) if len(t1) > 2 else 0
            if len(t1) > 3:
                ascale = float(t1[3])
            if len(t1) > 4:
                fscale = float(t1[4])
            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                if len(t2) >= 5:
                    ascale = float(t2[0])
                    fscale = float(t2[1])
                    tstart = float(t2[2])
                    tstop = float(t2[3])
                    emiss = float(t2[4])
                elif len(t2) == 3:
                    tstart = float(t2[0])
                    tstop = float(t2[1])
                    emiss = float(t2[2])
                elif len(t2) == 1:
                    emiss = float(t2[0])

    from ...model.entities import HeatRadiation
    hr = HeatRadiation(
        id=rid, title=title, surf_id=surf_id, funct_id=funct_id, sensor_id=sensor_id,
        ascale=ascale, fscale=fscale, tstart=tstart, tstop=tstop, emissivity=emiss, emiss=emiss
    )
    model.heat_radiations[rid] = hr




def read_heat_rad_cav(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HEAT/RAD_CAV/id`` or ``/HEAT/CAV/id`` (M205): Cavity radiation surface coupling."""
    from ...model.entities import HeatRadCav
    hid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    surf_id1, surf_id2 = 0, 0
    emissivity1, emissivity2, view_factor = 1.0, 1.0, 1.0
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("HEAT_RAD_CAV_1")
            surf_id1 = _ival(f[0]) if len(f) > 0 else 0
            surf_id2 = _ival(f[1]) if len(f) > 1 else 0
            emissivity1 = _fval(f[2], 1.0) if len(f) > 2 and f[2].strip() else 1.0
            emissivity2 = _fval(f[3], 1.0) if len(f) > 3 and f[3].strip() else 1.0
            view_factor = _fval(f[4], 1.0) if len(f) > 4 and f[4].strip() else 1.0
        else:
            t = cards[0].tokens()
            surf_id1 = int(float(t[0])) if len(t) > 0 else 0
            surf_id2 = int(float(t[1])) if len(t) > 1 else 0
            emissivity1 = float(t[2]) if len(t) > 2 else 1.0
            emissivity2 = float(t[3]) if len(t) > 3 else 1.0
            view_factor = float(t[4]) if len(t) > 4 else 1.0
    hrc = HeatRadCav(
        id=hid, title=title, surf_id1=surf_id1, surf_id2=surf_id2,
        emissivity1=emissivity1, emissivity2=emissivity2, view_factor=view_factor
    )
    model.heat_rad_cavs[hid] = hrc





def read_grav(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRAV/grav_ID`` or ``/GRAV`` (M37, M151)::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   skew_ID   sens_ID   grnod_ID   <blank>   Ascale_x   Fscale_Y

      acceleration a(t) = Fscale * f(t) applied along Dir to the group
      (grnod_ID = 0 → all nodes). Fscale defaults to 1.
    """
    gid = block.user_id if block.user_id is not None else 1
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/GRAV/{gid}: missing data card", block.source)
            return
        f = cards[0].cut("GRAV")
        fct = _ival(f[0])
        direction = _direction(f[1])
        skew_id = _ival(f[2]) if len(f) > 2 else 0
        sens_id = _ival(f[3]) if len(f) > 3 else 0
        grnod = _ival(f[4]) if len(f) > 4 else 0
        scale_x = _fval(f[6], 1.0) if len(f) > 6 else 1.0
        scale_y = _fval(f[7], 1.0) if len(f) > 7 else 1.0
        if scale_x == 0.0:
            scale_x = 1.0
        if scale_y == 0.0:
            scale_y = 1.0
        model.gravity.append(Gravity(
            id=gid, grnod_id=grnod or None, funct_id=fct,
            direction=direction, scale=scale_y, title=title,
            skew_id=skew_id, sens_id=sens_id, scale_x=scale_x,
        ))
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/GRAV/{gid}: missing data card", block.source)
        return
    t = cards[0].tokens()
    fct = int(t[0]) if len(t) > 0 else 0
    direction = _direction(t[1]) if len(t) > 1 else np.array([0.0, 0.0, 1.0])
    grnod = int(t[2]) if len(t) > 2 else 0
    scale = float(t[3]) if len(t) > 3 else 1.0
    if scale == 0.0:
        scale = 1.0
    model.gravity.append(Gravity(
        id=gid, grnod_id=grnod or None, funct_id=fct,
        direction=direction, scale=scale, title=title,
    ))




def read_cload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CLOAD/cload_ID``::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale   [sens_ID]

      force F(t) = Fscale * f(t) applied along Dir to EVERY node of the
      group (Radioss semantics: per node, not divided among them).
      sens_ID (M6): the load waits for /SENSOR sens_ID and then follows
      f(t - t_fire) — the curve is the load's own history from activation.

    Fixed dialect (cfg LOADS/cload.cfg radioss51; M37): ``fct_IDT DIR
    skew_ID sens_ID grnod_ID <blank> Ascale_x Fscale_Y`` — same columns
    as /GRAV; the sensor column IS ported (M6 gating), skew/Ascale_x are
    warned when set.
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/CLOAD/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("CLOAD")
        scale = _fval(f[7], 1.0)
        _warn_ignored(log, f"/CLOAD/{block.user_id}", block.source,
                      [("skew_ID", f[2])])
        model.cloads.append(ConcentratedLoad(
            id=block.user_id, funct_id=_ival(f[0]),
            direction=_direction(f[1]), grnod_id=_ival(f[4]),
            scale=scale if scale != 0.0 else 1.0,
            time_scale=_fval(f[6], 1.0),
            sens_id=_ival(f[3]), title=title))
        from ...model.entities import LoadCload
        model.load_cloads[block.user_id] = LoadCload(
            id=block.user_id,
            curve_id=_ival(f[0]),
            dir=f[1].strip() or "X",
            skew_id=_ival(f[2]),
            sens_id=_ival(f[3]),
            grnod_id=_ival(f[4]),
            xscale=_fval(f[6], 1.0) if _fval(f[6], 1.0) != 0.0 else 1.0,
            magnitude=scale if scale != 0.0 else 1.0,
            title=title,
        )
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/CLOAD/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    if len(t) < 3:
        log.error(f"/CLOAD/{block.user_id}: data card requires at least funct_id, direction, grnod_id", block.source)
        return
    model.cloads.append(ConcentratedLoad(
        id=block.user_id, funct_id=_ival(t[0]), direction=_direction(t[1]),
        grnod_id=_ival(t[2]), scale=_fval(t[3], 1.0) if len(t) > 3 else 1.0,
        sens_id=_ival(t[4]) if len(t) > 4 else 0, title=title))
    from ...model.entities import LoadCload
    model.load_cloads[block.user_id] = LoadCload(
        id=block.user_id,
        curve_id=_ival(t[0]),
        dir=t[1].strip() if len(t) > 1 else "X",
        grnod_id=_ival(t[2]),
        magnitude=_fval(t[3], 1.0) if len(t) > 3 else 1.0,
        sens_id=_ival(t[4]) if len(t) > 4 else 0,
        title=title,
    )




def read_load_centri(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/CENTRI/load_ID`` (M93)::

        card 1:  title
        card 2:  funct_IDT  Dir  frame_ID  sensor_ID  grnod_ID  Ivar  Ascalex  Fscaley
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOAD/CENTRI/{block.user_id}: missing data card", block.source)
        return
    is_fixed = block.fixed and ("," not in cards[0].raw)
    if is_fixed:
        try:
            f = cards[0].cut("LOAD_CENTRI")
            funct_id = _ival(f[0])
            dir_str = f[1].strip().upper() if len(f) > 1 and f[1].strip() else "XX"
            frame_id = _ival(f[2]) if len(f) > 2 else 0
            sens_id = _ival(f[3]) if len(f) > 3 else 0
            grnod_id = _ival(f[4]) if len(f) > 4 else 0
            ivar = _ival(f[5], default=1) if len(f) > 5 else 1
            scale_x = _fval(f[6], default=1.0) if len(f) > 6 else 1.0
            scale_y = _fval(f[7], default=1.0) if len(f) > 7 else 1.0
        except ValueError:
            is_fixed = False
    if not is_fixed:
        raw = cards[0].raw.replace(",", " ")
        t = raw.split()
        funct_id = int(float(t[0])) if len(t) > 0 else 0
        dir_str = t[1].upper() if len(t) > 1 and t[1] else "XX"
        frame_id = int(float(t[2])) if len(t) > 2 else 0
        sens_id = int(float(t[3])) if len(t) > 3 else 0
        grnod_id = int(float(t[4])) if len(t) > 4 else 0
        ivar = int(float(t[5])) if len(t) > 5 else 1
        scale_x = float(t[6]) if len(t) > 6 else 1.0
        scale_y = float(t[7]) if len(t) > 7 else 1.0

    if scale_x == 0.0:
        scale_x = 1.0
    if scale_y == 0.0:
        scale_y = 1.0

    cl = CentrifugalLoad(
        id=block.user_id, funct_id=funct_id, dir=dir_str,
        frame_id=frame_id, sens_id=sens_id, grnod_id=grnod_id,
        ivar=ivar, scale_x=scale_x, scale_y=scale_y, title=title,
    )
    cl.grnd_id = grnod_id
    cl.fct_id = funct_id
    model.centri_loads.append(cl)
    from ...model.entities import LoadCentri
    model.load_centris[block.user_id] = LoadCentri(
        id=block.user_id, title=title, dir=dir_str,
        fct_id=funct_id, frame_id=frame_id, sens_id=sens_id,
        grnod_id=grnod_id, ivar=ivar, ascalex=scale_x, fscaley=scale_y,
    )
    model.centris[block.user_id] = cl




def read_load_pressure(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PRESSURE/load_ID`` or ``/LOAD/PFLUID/load_ID`` (M112/M163): Surface pressure loading.

    Fortran origin: ``starter/source/loads/general/load_pressure/hm_read_load_pressure.F``.
    Standard card format (M163 / radioss2022):
      Card 1:  surf_ID  Iload  sens_ID  Inorm  Direction  skew_ID
      Card 2:  fct_ID   (blank)  xscale_p  yscale_p
      Card 3+: Inter_ID (blank)  Gap_shift_i
    Legacy card format (M112):
      Card 1:  surf_ID  fct_ID  sens_ID
      Card 2:  Scale    Tstart  Tstop
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOAD/PRESSURE/{block.user_id}: missing data card", block.source)
        return
    from ...model.entities import LoadPressure

    inter_ids: List[int] = []
    gap_shifts: List[float] = []

    if block.fixed:
        f1 = cards[0].cut("LOAD_PRESSURE_1")
        is_legacy = False
        if len(cards) > 1 and not cards[1].is_blank:
            c1_raw20 = cards[1].raw[:20].strip()
            if "." in c1_raw20 and not f1[3].strip() and not f1[4].strip() and not f1[5].strip():
                is_legacy = True

        if is_legacy:
            surf_id = _ival(f1[0]) if len(f1) > 0 else 0
            fct_id = _ival(f1[1]) if len(f1) > 1 else 0
            sens_id = _ival(f1[2]) if len(f1) > 2 else 0
            scale = _fval(cards[1].raw[:20], 1.0)
            tstart = _fval(cards[1].raw[20:40], 0.0)
            tstop = _fval(cards[1].raw[40:60], 1.0e30)
            model.load_pressures[block.user_id] = LoadPressure(
                id=block.user_id, title=title, surf_id=surf_id, fct_id=fct_id,
                sens_id=sens_id, scale=scale, yscale_p=scale, tstart=tstart, tstop=tstop
            )
            return

        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        iload = _ival(f1[1], 1) if len(f1) > 1 and f1[1].strip() else 1
        sens_id = _ival(f1[2]) if len(f1) > 2 else 0
        inorm = _ival(f1[3], 1) if len(f1) > 3 and f1[3].strip() else 1
        direction = f1[4].strip() if len(f1) > 4 else ""
        skew_id = _ival(f1[5]) if len(f1) > 5 else 0

        fct_id = 0
        xscale_p = 1.0
        yscale_p = 1.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("LOAD_PRESSURE_2")
            fct_id = _ival(f2[0]) if len(f2) > 0 else 0
            xscale_p = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
            yscale_p = _fval(f2[3], 1.0) if len(f2) > 3 and f2[3].strip() else 1.0

        for c in cards[2:]:
            if c.is_blank:
                continue
            f3 = c.cut("LOAD_PRESSURE_3")
            iid = _ival(f3[0]) if len(f3) > 0 else 0
            gshift = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            if iid > 0:
                inter_ids.append(iid)
                gap_shifts.append(gshift)
    else:
        t1 = cards[0].tokens()
        if len(t1) == 3 and len(cards) > 1 and len(cards[1].tokens()) == 3 and "." in cards[1].tokens()[0]:
            surf_id = int(float(t1[0])) if len(t1) > 0 else 0
            fct_id = int(float(t1[1])) if len(t1) > 1 else 0
            sens_id = int(float(t1[2])) if len(t1) > 2 else 0
            t2 = cards[1].tokens()
            scale = float(t2[0]) if len(t2) > 0 else 1.0
            tstart = float(t2[1]) if len(t2) > 1 else 0.0
            tstop = float(t2[2]) if len(t2) > 2 else 1.0e30
            model.load_pressures[block.user_id] = LoadPressure(
                id=block.user_id, title=title, surf_id=surf_id, fct_id=fct_id,
                sens_id=sens_id, scale=scale, yscale_p=scale, tstart=tstart, tstop=tstop
            )
            return

        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        iload = int(float(t1[1])) if len(t1) > 1 else 1
        sens_id = int(float(t1[2])) if len(t1) > 2 else 0
        inorm = int(float(t1[3])) if len(t1) > 3 else 1
        direction = t1[4] if len(t1) > 4 else ""
        skew_id = int(float(t1[5])) if len(t1) > 5 else 0

        fct_id = 0
        xscale_p = 1.0
        yscale_p = 1.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            fct_id = int(float(t2[0])) if len(t2) > 0 else 0
            xscale_p = float(t2[1]) if len(t2) > 1 else 1.0
            yscale_p = float(t2[2]) if len(t2) > 2 else 1.0

        for c in cards[2:]:
            if c.is_blank:
                continue
            t3 = c.tokens()
            iid = int(float(t3[0])) if len(t3) > 0 else 0
            gshift = float(t3[1]) if len(t3) > 1 else 0.0
            if iid > 0:
                inter_ids.append(iid)
                gap_shifts.append(gshift)

    if xscale_p == 0.0:
        xscale_p = 1.0
    if yscale_p == 0.0:
        yscale_p = 1.0

    model.load_pressures[block.user_id] = LoadPressure(
        id=block.user_id, title=title, surf_id=surf_id, iload=iload,
        sens_id=sens_id, inorm=inorm, direction=direction, skew_id=skew_id,
        fct_id=fct_id, xscale_p=xscale_p, yscale_p=yscale_p,
        inter_ids=inter_ids, gap_shifts=gap_shifts,
        scale=yscale_p
    )






def read_pblast(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PBLAST/id`` or ``/PBLAST/id`` (M99): Blast pressure load.

    Fortran origin: ``starter/source/model/loads/hm_read_pblast.F``.
    Card format:
        card 1: title
        card 2: surf_ID  Exp_data  I_tshift  Ndt  IZ  Imodel  (3x blank)  Node_id
        card 3: Xdet  Ydet  Zdet  Tdet  WTNT
        card 4 (optional): PMIN
    """
    from ...model.entities import PBlastLoad
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/LOAD/PBLAST/{block.user_id}: missing data cards", block.source)
        return

    pmin = 0.0
    tstop = 1.0e30
    surf_ground_id = 0
    ishape = 0

    if block.fixed:
        c1 = cards[0].cut("LOAD_PBLAST_1")
        surf_id = _ival(c1[0]) if len(c1) > 0 else 0
        exp_data = _ival(c1[1], 1) if len(c1) > 1 else 1
        i_tshift = _ival(c1[2], 1) if len(c1) > 2 else 1
        ndt = _ival(c1[3]) if len(c1) > 3 else 0
        iz = _ival(c1[4], 2) if len(c1) > 4 else 2
        imodel = _ival(c1[5]) if len(c1) > 5 else 0
        node_id = _ival(c1[9]) if len(c1) > 9 else 0

        c2 = cards[1].cut("LOAD_PBLAST_2") if len(cards) > 1 else []
        xdet = _fval(c2[0]) if len(c2) > 0 else 0.0
        ydet = _fval(c2[1]) if len(c2) > 1 else 0.0
        zdet = _fval(c2[2]) if len(c2) > 2 else 0.0
        tdet = _fval(c2[3]) if len(c2) > 3 else 0.0
        wtnt = _fval(c2[4]) if len(c2) > 4 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            c3 = cards[2].cut("PBLAST_3")
            pmin = _fval(c3[0]) if len(c3) > 0 else 0.0
            tstop = _fval(c3[1], 1.0e30) if len(c3) > 1 and c3[1].strip() else 1.0e30

        if len(cards) > 3 and not cards[3].is_blank:
            c4 = cards[3].cut("PBLAST_4")
            surf_ground_id = _ival(c4[0]) if len(c4) > 0 else 0
            ishape = _ival(c4[1]) if len(c4) > 1 else 0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        exp_data = int(float(t1[1])) if len(t1) > 1 else 1
        i_tshift = int(float(t1[2])) if len(t1) > 2 else 1
        ndt = int(float(t1[3])) if len(t1) > 3 else 0
        iz = int(float(t1[4])) if len(t1) > 4 else 2
        imodel = int(float(t1[5])) if len(t1) > 5 else 0
        node_id = int(float(t1[6])) if len(t1) > 6 else 0

        t2 = cards[1].tokens() if len(cards) > 1 else []
        xdet = float(t2[0]) if len(t2) > 0 else 0.0
        ydet = float(t2[1]) if len(t2) > 1 else 0.0
        zdet = float(t2[2]) if len(t2) > 2 else 0.0
        tdet = float(t2[3]) if len(t2) > 3 else 0.0
        wtnt = float(t2[4]) if len(t2) > 4 else 0.0

        if len(cards) > 2 and cards[2].tokens():
            t3 = cards[2].tokens()
            pmin = float(t3[0]) if len(t3) > 0 else 0.0
            tstop = float(t3[1]) if len(t3) > 1 else 1.0e30

        if len(cards) > 3 and cards[3].tokens():
            t4 = cards[3].tokens()
            surf_ground_id = int(float(t4[0])) if len(t4) > 0 else 0
            ishape = int(float(t4[1])) if len(t4) > 1 else 0

    model.pblast_loads[block.user_id] = PBlastLoad(
        id=block.user_id, title=title, surf_id=surf_id, exp_data=exp_data,
        i_tshift=i_tshift, ndt=ndt, iz=iz, imodel=imodel, node_id=node_id,
        xdet=xdet, ydet=ydet, zdet=zdet, tdet=tdet, wtnt=wtnt, pmin=pmin,
        tstop=tstop, surf_ground_id=surf_ground_id, ishape=ishape,
    )




def read_load_pcyl(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PCYL/id`` or ``/PCYL/id`` (M178): Pressure load in cylindrical coordinates.

    Fortran origin: ``starter/source/loads/load_pcyl.cfg``.
    Card 1: TITLE (%-100s)
    Card 2: surf_ID, sens_ID, frame_ID (%10d%10d%10d)
    Card 3: table_ID, blank(10), xscale_r, xscale_t, yscale_p (%10d%10s%20lg%20lg%20lg)
    """
    from ...model.entities import PcylLoad
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    surf_id, sens_id, frame_id = 0, 0, 0
    table_id = 0
    xscale_r, xscale_t, yscale_p = 1.0, 1.0, 1.0
    if cards:
        c1 = cards[0]
        if block.fixed:
            f1 = c1.cut("LOAD_PCYL_1")
            surf_id = _ival(f1[0]) if len(f1) > 0 else 0
            sens_id = _ival(f1[1]) if len(f1) > 1 else 0
            frame_id = _ival(f1[2]) if len(f1) > 2 else 0
        else:
            toks1 = c1.tokens()
            surf_id = int(float(toks1[0])) if len(toks1) > 0 else 0
            sens_id = int(float(toks1[1])) if len(toks1) > 1 else 0
            frame_id = int(float(toks1[2])) if len(toks1) > 2 else 0
    if len(cards) > 1:
        c2 = cards[1]
        if block.fixed:
            f2 = c2.cut("LOAD_PCYL_2")
            table_id = _ival(f2[0]) if len(f2) > 0 else 0
            xscale_r = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
            xscale_t = _fval(f2[3], 1.0) if len(f2) > 3 and f2[3].strip() else 1.0
            yscale_p = _fval(f2[4], 1.0) if len(f2) > 4 and f2[4].strip() else 1.0
        else:
            toks2 = c2.tokens()
            table_id = int(float(toks2[0])) if len(toks2) > 0 else 0
            xscale_r = float(toks2[1]) if len(toks2) > 1 else 1.0
            xscale_t = float(toks2[2]) if len(toks2) > 2 else 1.0
            yscale_p = float(toks2[3]) if len(toks2) > 3 else 1.0
    model.pcyl_loads[block.user_id] = PcylLoad(
        id=block.user_id, surf_id=surf_id, sens_id=sens_id, frame_id=frame_id,
        table_id=table_id, xscale_r=xscale_r, xscale_t=xscale_t, yscale_p=yscale_p,
        title=title
    )




def read_load(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/<subtype>/load_ID`` dispatcher (M93, M99, M103, M112, M135)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub in ("CENTRI", "CENTRIF"):
        read_load_centri(block, model, log)
    elif sub == "CLOAD":
        read_cload(block, model, log)
    elif sub == "PLOAD":
        read_pload(block, model, log)
    elif sub == "PBLAST":
        read_pblast(block, model, log)
    elif sub == "PCYL":
        read_pcyl(block, model, log)
    elif sub == "PFLUID":
        read_pfluid(block, model, log)
    elif sub in ("PRESSURE", "PRESS"):
        read_load_pressure(block, model, log)
    elif sub == "LASER":
        read_laser(block, model, log)
    elif sub in ("PRELOAD_AXIAL", "PRELOAD"):
        read_preload_axial(block, model, log)
    elif sub == "BOLT":
        read_preload_bolt(block, model, log)
    elif sub in ("HYDRO", "HYDROSTATIC"):
        from ...model.entities import LoadHydro
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/LOAD/HYDRO/{block.user_id}: missing data card", block.source)
            return
        lid = block.user_id or 1
        if block.fixed:
            f = cards[0].cut("LOAD_HYDRO_1")
            surf_id = _ival(f[0]) if len(f) > 0 else 0
            sens_id = _ival(f[1]) if len(f) > 1 else 0
            density = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            z_free = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            gravity = _fval(f[4], 9.81) if len(f) > 4 and _fval(f[4]) > 0.0 else 9.81
        else:
            t = cards[0].tokens()
            surf_id = int(float(t[0])) if len(t) > 0 else 0
            sens_id = int(float(t[1])) if len(t) > 1 else 0
            density = float(t[2]) if len(t) > 2 else 0.0
            z_free = float(t[3]) if len(t) > 3 else 0.0
            gravity = float(t[4]) if len(t) > 4 else 9.81
        model.load_hydros[lid] = LoadHydro(
            id=lid, title=title, surf_id=surf_id, density=density,
            z_free=z_free, gravity=gravity, sens_id=sens_id
        )
    elif sub in ("GRAV", "GRAVITY"):
        # /LOAD/GRAV (M135)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/LOAD/GRAV/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("LOAD_GRAV_1")
            grnod_id = _ival(f[0]) if len(f) > 0 else 0
            dx = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            dy = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            dz = _fval(f[3], -1.0) if len(f) > 3 else -1.0
            funct_id = _ival(f[4], 0) if len(f) > 4 else 0
            scale = _fval(f[5], 1.0) if len(f) > 5 else 1.0
            sens_id = _ival(f[6], 0) if len(f) > 6 else 0
        else:
            t = cards[0].tokens()
            grnod_id = int(float(t[0])) if len(t) > 0 else 0
            dx = float(t[1]) if len(t) > 1 else 0.0
            dy = float(t[2]) if len(t) > 2 else 0.0
            dz = float(t[3]) if len(t) > 3 else -1.0
            funct_id = int(float(t[4])) if len(t) > 4 else 0
            scale = float(t[5]) if len(t) > 5 else 1.0
            sens_id = int(float(t[6])) if len(t) > 6 else 0
        model.load_gravities[block.user_id] = LoadGravity(
            id=block.user_id, title=title, grnod_id=grnod_id,
            dir_vector=(dx, dy, dz), funct_id=funct_id,
            scale=scale, sens_id=sens_id
        )
    elif sub == "BODY":
        # /LOAD/BODY (M135)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/LOAD/BODY/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("LOAD_BODY_1")
            grpart_id = _ival(f[0]) if len(f) > 0 else 0
            dx = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            dy = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            dz = _fval(f[3], -1.0) if len(f) > 3 else -1.0
            funct_id = _ival(f[4], 0) if len(f) > 4 else 0
            scale = _fval(f[5], 1.0) if len(f) > 5 else 1.0
            sens_id = _ival(f[6], 0) if len(f) > 6 else 0
        else:
            t = cards[0].tokens()
            grpart_id = int(float(t[0])) if len(t) > 0 else 0
            dx = float(t[1]) if len(t) > 1 else 0.0
            dy = float(t[2]) if len(t) > 2 else 0.0
            dz = float(t[3]) if len(t) > 3 else -1.0
            funct_id = int(float(t[4])) if len(t) > 4 else 0
            scale = float(t[5]) if len(t) > 5 else 1.0
            sens_id = int(float(t[6])) if len(t) > 6 else 0
        model.load_bodies[block.user_id] = LoadBody(
            id=block.user_id, title=title, grpart_id=grpart_id,
            dir_vector=(dx, dy, dz), funct_id=funct_id,
            scale=scale, sens_id=sens_id
        )
    elif sub in ("HEAT", "THERM"):
        # /LOAD/HEAT (M135)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/LOAD/HEAT/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("LOAD_THERM_1")
            group_id = _ival(f[0]) if len(f) > 0 else 0
            flux = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            funct_id = _ival(f[2], 0) if len(f) > 2 else 0
            scale = _fval(f[3], 1.0) if len(f) > 3 else 1.0
            sens_id = _ival(f[4], 0) if len(f) > 4 else 0
        else:
            t = cards[0].tokens()
            group_id = int(float(t[0])) if len(t) > 0 else 0
            flux = float(t[1]) if len(t) > 1 else 0.0
            funct_id = int(float(t[2])) if len(t) > 2 else 0
            scale = float(t[3]) if len(t) > 3 else 1.0
            sens_id = int(float(t[4])) if len(t) > 4 else 0
        model.load_therms[block.user_id] = LoadTherm(
            id=block.user_id, title=title, group_id=group_id,
            flux=flux, funct_id=funct_id, scale=scale, sens_id=sens_id
        )
    else:
        log.warning(f"/LOAD/{sub} not ported (CENTRI, PBLAST, PCYL, PFLUID, PRESSURE, LASER, PRELOAD, GRAV, BODY, HEAT supported)", block.source)




def read_pload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PLOAD/pload_ID`` (M5) or ``/PLOAD/PCYL/load_ID`` (M130)::

        card 1:  title
        card 2:  surf_ID   fct_ID   Fscale   [sens_ID]

      follower pressure p(t) = Fscale * f(t) on every segment of the
      surface, acting along the current segment normal (node ordering
      n1-n2-n3-n4, right-hand rule: positive p pushes along +n). The
      resultant p*A of each segment is lumped to its corners.
      sens_ID (M6): waits for /SENSOR sens_ID, then follows p(t - t_fire).

    Fixed dialect (cfg LOADS/pload.cfg radioss51; M37): ``surf_ID
    fct_IDT sens_ID <blank> Ascale_x Fscale_Y`` — the scale is column
    81-100's Fscale_Y (blank -> 1.0), Ascale_x warned when set.
    """
    if len(block.parts) > 1:
        sub = block.parts[1].upper()
        if sub == "PCYL":
            read_pcyl(block, model, log)
            return
        elif sub in ("PRESSURE", "PRESS", "PFLUID"):
            read_load_pressure(block, model, log)
            return

    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/PLOAD/{block.user_id}: missing data card",
                      block.source)
            return
        raw = cards[0].raw
        if len(cards[0].tokens()) >= 6 or (len(raw) >= 60 and raw[30:60].strip()):
            f = cards[0].cut("LOAD_PLOAD_2023")
            surf_id = _ival(f[0]) if len(f) > 0 else 0
            funct_id = _ival(f[1]) if len(f) > 1 else 0
            sens_id = _ival(f[2]) if len(f) > 2 else 0
            ipinch = _ival(f[3]) if len(f) > 3 else 0
            idel = _ival(f[4], 1) if len(f) > 4 and f[4].strip() else 1
            functype = _ival(f[5], 1) if len(f) > 5 and f[5].strip() else 1
            xscale = _fval(f[6], 1.0) if len(f) > 6 and f[6].strip() else 1.0
            magnitude = _fval(f[7], 1.0) if len(f) > 7 and f[7].strip() else 1.0
        else:
            f = cards[0].cut("PLOAD")
            surf_id = _ival(f[0]) if len(f) > 0 else 0
            funct_id = _ival(f[1]) if len(f) > 1 else 0
            sens_id = _ival(f[2]) if len(f) > 2 else 0
            ipinch = 0
            idel = 1
            functype = 1
            xscale = _fval(f[4], 1.0) if len(f) > 4 and f[4].strip() else 1.0
            magnitude = _fval(f[5], 1.0) if len(f) > 5 and f[5].strip() else 1.0
        scale = magnitude if magnitude != 0.0 else 1.0
        _warn_ignored(log, f"/PLOAD/{block.user_id}", block.source,
                      [("Ascale_x", str(xscale) if xscale not in (0.0, 1.0)
                        else "")])
        model.ploads.append(PressureLoad(
            id=block.user_id, surf_id=surf_id, funct_id=funct_id,
            scale=scale, sens_id=sens_id, title=title))
        from ...model.entities import LoadPload
        model.load_ploads[block.user_id] = LoadPload(
            id=block.user_id, surf_id=surf_id, curve_id=funct_id,
            sens_id=sens_id, ipinch=ipinch, idel=idel, functype=functype,
            xscale=xscale if xscale != 0.0 else 1.0,
            magnitude=scale, title=title,
        )
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/PLOAD/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    if len(t) >= 6:
        surf_id = int(float(t[0])) if len(t) > 0 else 0
        funct_id = int(float(t[1])) if len(t) > 1 else 0
        sens_id = int(float(t[2])) if len(t) > 2 else 0
        ipinch = int(float(t[3])) if len(t) > 3 else 0
        idel = int(float(t[4])) if len(t) > 4 else 1
        functype = int(float(t[5])) if len(t) > 5 else 1
        xscale = float(t[6]) if len(t) > 6 else 1.0
        magnitude = float(t[7]) if len(t) > 7 else 1.0
    else:
        surf_id = int(float(t[0])) if len(t) > 0 else 0
        funct_id = int(float(t[1])) if len(t) > 1 else 0
        magnitude = float(t[2]) if len(t) > 2 else 1.0
        sens_id = int(float(t[3])) if len(t) > 3 else 0
        ipinch = 0
        idel = 1
        functype = 1
        xscale = 1.0
    scale = magnitude if magnitude != 0.0 else 1.0
    model.ploads.append(PressureLoad(
        id=block.user_id, surf_id=surf_id, funct_id=funct_id,
        scale=scale, sens_id=sens_id, title=title))
    from ...model.entities import LoadPload
    model.load_ploads[block.user_id] = LoadPload(
        id=block.user_id, surf_id=surf_id, curve_id=funct_id,
        sens_id=sens_id, ipinch=ipinch, idel=idel, functype=functype,
        xscale=xscale if xscale != 0.0 else 1.0,
        magnitude=scale, title=title,
    )





def read_laser(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LASER/id`` or ``/DFS/LASER/id`` (M102): laser beam impact.

    Fortran origin: ``starter/source/loads/laser/leclas.F``.
    """
    from ...model.entities import LaserLoad
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    magnitude = 0.0
    curve_id = 0
    s_target = 0.0
    fct_id_target = 0
    hn = vcp = k0 = rd = ks = 0.0
    np = 0
    nc = 0
    plasma_elements = []

    if cards:
        c1 = cards[0]
        if block.fixed:
            f1 = c1.cut("LASER_1")
            magnitude = _fval(f1[0]) if len(f1) > 0 else 0.0
            curve_id = _ival(f1[1]) if len(f1) > 1 else 0
            s_target = _fval(f1[3]) if len(f1) > 3 else 0.0
            fct_id_target = _ival(f1[4]) if len(f1) > 4 else 0
        else:
            toks1 = c1.tokens()
            magnitude = float(toks1[0]) if len(toks1) > 0 else 0.0
            curve_id = int(float(toks1[1])) if len(toks1) > 1 else 0
            s_target = float(toks1[2]) if len(toks1) > 2 else 0.0
            fct_id_target = int(float(toks1[3])) if len(toks1) > 3 else 0

    if len(cards) > 1:
        c2 = cards[1]
        if block.fixed:
            f2 = c2.cut("LASER_2")
            hn = _fval(f2[0]) if len(f2) > 0 else 0.0
            vcp = _fval(f2[1]) if len(f2) > 1 else 0.0
            k0 = _fval(f2[2]) if len(f2) > 2 else 0.0
            rd = _fval(f2[3]) if len(f2) > 3 else 0.0
            ks = _fval(f2[4]) if len(f2) > 4 else 0.0
        else:
            toks2 = c2.tokens()
            hn = float(toks2[0]) if len(toks2) > 0 else 0.0
            vcp = float(toks2[1]) if len(toks2) > 1 else 0.0
            k0 = float(toks2[2]) if len(toks2) > 2 else 0.0
            rd = float(toks2[3]) if len(toks2) > 3 else 0.0
            ks = float(toks2[4]) if len(toks2) > 4 else 0.0

    if len(cards) > 2:
        c3 = cards[2]
        if block.fixed:
            f3 = c3.cut("LASER_3")
            np = _ival(f3[0]) if len(f3) > 0 else 0
            nc = _ival(f3[1]) if len(f3) > 1 else 0
        else:
            toks3 = c3.tokens()
            np = int(float(toks3[0])) if len(toks3) > 0 else 0
            nc = int(float(toks3[1])) if len(toks3) > 1 else 0

    for c in cards[3:]:
        for t in c.tokens():
            plasma_elements.append(int(float(t)))

    model.laser_loads[block.user_id] = LaserLoad(
        id=block.user_id, title=title, magnitude=magnitude, curve_id=curve_id,
        s_target=s_target, fct_id_target=fct_id_target,
        hn=hn, vcp=vcp, k0=k0, rd=rd, ks=ks,
        np=np, nc=nc, plasma_elements=plasma_elements,
    )




def read_pcyl(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PCYL/load_ID`` (M103)::

        card 1:  title
        card 2:  surf_ID  sens_ID  frame_ID
        card 3:  table_ID [gap]  xscale_r  xscale_t  yscale_p
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOAD/PCYL/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    sens_id = 0
    frame_id = 0
    table_id = 0
    xscale_r = 1.0
    xscale_t = 1.0
    yscale_p = 1.0

    if block.fixed:
        f1 = cards[0].cut("PCYL_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        sens_id = _ival(f1[1]) if len(f1) > 1 else 0
        frame_id = _ival(f1[2]) if len(f1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PCYL_2")
            table_id = _ival(f2[0]) if len(f2) > 0 else 0
            xscale_r = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            xscale_t = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            yscale_p = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        sens_id = int(float(t1[1])) if len(t1) > 1 else 0
        frame_id = int(float(t1[2])) if len(t1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            table_id = int(float(t2[0])) if len(t2) > 0 else 0
            xscale_r = float(t2[1]) if len(t2) > 1 else 1.0
            xscale_t = float(t2[2]) if len(t2) > 2 else 1.0
            yscale_p = float(t2[3]) if len(t2) > 3 else 1.0

    model.pcyl_loads[block.user_id] = PcylLoad(
        id=block.user_id, title=title, surf_id=surf_id, sens_id=sens_id,
        frame_id=frame_id, table_id=table_id, xscale_r=xscale_r,
        xscale_t=xscale_t, yscale_p=yscale_p,
    )




def read_pfluid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PFLUID/load_ID`` (M103)::

        card 1:  title
        card 2:  surf_ID  sens_ID
        card 3:  fct_ID_t [gap]  ascalex  fscaley
        card 4:  dir_p  frame_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOAD/PFLUID/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    sens_id = 0
    fct_id_t = 0
    ascalex = 1.0
    fscaley = 1.0
    dir_p = "Z"
    frame_id = 0
    fct_id_pc = 0
    ascalex_pc = 1.0
    fscaley_pc = 1.0
    fct_id_vel = 0
    ascalex_vel = 1.0
    fscaley_vel = 1.0
    dir_vel = "Z"
    frame_id_vel = 0

    if block.fixed:
        f1 = cards[0].cut("PFLUID_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        sens_id = _ival(f1[1]) if len(f1) > 1 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PFLUID_2")
            fct_id_t = _ival(f2[0]) if len(f2) > 0 else 0
            ascalex = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            fscaley = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("PFLUID_3")
            dir_p = f3[0].strip().upper() if len(f3) > 0 and f3[0].strip() else "Z"
            frame_id = _ival(f3[1]) if len(f3) > 1 else 0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("PFLUID_2")
            fct_id_pc = _ival(f4[0]) if len(f4) > 0 else 0
            ascalex_pc = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0
            fscaley_pc = _fval(f4[3], 1.0) if len(f4) > 3 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("PFLUID_2")
            fct_id_vel = _ival(f5[0]) if len(f5) > 0 else 0
            ascalex_vel = _fval(f5[2], 1.0) if len(f5) > 2 else 1.0
            fscaley_vel = _fval(f5[3], 1.0) if len(f5) > 3 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("PFLUID_3")
            dir_vel = f6[0].strip().upper() if len(f6) > 0 and f6[0].strip() else "Z"
            frame_id_vel = _ival(f6[1]) if len(f6) > 1 else 0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        sens_id = int(float(t1[1])) if len(t1) > 1 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            fct_id_t = int(float(t2[0])) if len(t2) > 0 else 0
            ascalex = float(t2[1]) if len(t2) > 1 else 1.0
            fscaley = float(t2[2]) if len(t2) > 2 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            dir_p = t3[0].strip().upper() if len(t3) > 0 else "Z"
            frame_id = int(float(t3[1])) if len(t3) > 1 else 0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            fct_id_pc = int(float(t4[0])) if len(t4) > 0 else 0
            ascalex_pc = float(t4[1]) if len(t4) > 1 else 1.0
            fscaley_pc = float(t4[2]) if len(t4) > 2 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            fct_id_vel = int(float(t5[0])) if len(t5) > 0 else 0
            ascalex_vel = float(t5[1]) if len(t5) > 1 else 1.0
            fscaley_vel = float(t5[2]) if len(t5) > 2 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            dir_vel = t6[0].strip().upper() if len(t6) > 0 else "Z"
            frame_id_vel = int(float(t6[1])) if len(t6) > 1 else 0

    model.pfluid_loads[block.user_id] = PfluidLoad(
        id=block.user_id, title=title, surf_id=surf_id, sens_id=sens_id,
        fct_id_t=fct_id_t, ascalex=ascalex, fscaley=fscaley, dir_p=dir_p,
        frame_id=frame_id, fct_id_pc=fct_id_pc, ascalex_pc=ascalex_pc,
        fscaley_pc=fscaley_pc, fct_id_vel=fct_id_vel, ascalex_vel=ascalex_vel,
        fscaley_vel=fscaley_vel, dir_vel=dir_vel, frame_id_vel=frame_id_vel,
    )




def read_preload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PRELOAD/preload_ID`` (M103, M144)::

        card 1:  title
        card 2:  sect_ID  sens_ID  Itype  fct_ID  Preload  Tstart  Tstop
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "AXIAL":
        read_preload_axial(block, model, log)
        return
    if sub in ("BOLT", "SECT_BOLT"):
        read_preload_bolt(block, model, log)
        return

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PRELOAD/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("PRELOAD_1")
        sect_id = _ival(f[0]) if len(f) > 0 else 0
        sens_id = _ival(f[1]) if len(f) > 1 else 0
        itype = _ival(f[2]) if len(f) > 2 else 0
        fct_id = _ival(f[3]) if len(f) > 3 else 0
        preload = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        tstart = _fval(f[5], 0.0) if len(f) > 5 else 0.0
        tstop = _fval(f[6], 1.0e30) if len(f) > 6 else 1.0e30
    else:
        toks = cards[0].tokens()
        sect_id = int(float(toks[0])) if len(toks) > 0 else 0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0
        itype = int(float(toks[2])) if len(toks) > 2 else 0
        fct_id = int(float(toks[3])) if len(toks) > 3 else 0
        preload = float(toks[4]) if len(toks) > 4 else 0.0
        tstart = float(toks[5]) if len(toks) > 5 else 0.0
        tstop = float(toks[6]) if len(toks) > 6 else 1.0e30

    model.preloads[block.user_id] = Preload(
        id=block.user_id, title=title, sect_id=sect_id, sens_id=sens_id,
        itype=itype, fct_id=fct_id, preload=preload, tstart=tstart, tstop=tstop,
    )




def read_preload_axial(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PRELOAD/AXIAL/preload_ID`` or ``/LOAD/PRELOAD_AXIAL/id`` (M103/M130)::

        card 1:  title
        card 2:  set_id  sens_id  curveid
        card 3:  Preload  Damp
      (also accepts 1-card format: set_id sens_id [gap] fct_id Preload Damp)
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PRELOAD/AXIAL/{block.user_id}: missing data card", block.source)
        return

    set_id = 0
    sens_id = 0
    fct_id = 0
    preload = 1.0
    damp = 0.0

    if block.fixed:
        if len(cards) >= 2 and not cards[1].is_blank:
            f1 = cards[0].cut("PRELOAD_AXIAL_1")
            set_id = _ival(f1[0]) if len(f1) > 0 else 0
            sens_id = _ival(f1[1]) if len(f1) > 1 else 0
            fct_id = _ival(f1[2]) if len(f1) > 2 else 0

            f2 = cards[1].cut("PRELOAD_AXIAL_2")
            preload = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            damp = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
        else:
            f = cards[0].cut("PRELOAD_AXIAL_LEGACY")
            set_id = _ival(f[0]) if len(f) > 0 else 0
            sens_id = _ival(f[1]) if len(f) > 1 else 0
            fct_id = _ival(f[3]) if len(f) > 3 else (_ival(f[2]) if len(f) > 2 else 0)
            preload = _fval(f[4], 1.0) if len(f) > 4 else (_fval(f[3], 1.0) if len(f) > 3 else 1.0)
            damp = _fval(f[5], 0.0) if len(f) > 5 else (_fval(f[4], 0.0) if len(f) > 4 else 0.0)
    else:
        toks1 = cards[0].tokens()
        if len(cards) >= 2 and not cards[1].is_blank and len(toks1) <= 3:
            set_id = int(float(toks1[0])) if len(toks1) > 0 else 0
            sens_id = int(float(toks1[1])) if len(toks1) > 1 else 0
            fct_id = int(float(toks1[2])) if len(toks1) > 2 else 0

            toks2 = cards[1].tokens()
            preload = float(toks2[0]) if len(toks2) > 0 else 1.0
            damp = float(toks2[1]) if len(toks2) > 1 else 0.0
        else:
            set_id = int(float(toks1[0])) if len(toks1) > 0 else 0
            sens_id = int(float(toks1[1])) if len(toks1) > 1 else 0
            fct_id = int(float(toks1[2])) if len(toks1) > 2 else 0
            preload = float(toks1[3]) if len(toks1) > 3 else 1.0
            damp = float(toks1[4]) if len(toks1) > 4 else 0.0

    if preload == 0.0:
        preload = 1.0

    model.preload_axials[block.user_id] = PreloadAxial(
        id=block.user_id, title=title, set_id=set_id, sens_id=sens_id,
        fun_id=fct_id, preload=preload, damp=damp,
    )




def read_preload_bolt(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PRELOAD/BOLT/preload_ID`` or ``/SECT/BOLT/id`` (M144): Bolt preload definition."""
    from ...model.entities import PreloadBolt
    bid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    sect_id, sens_id, fct_id = 0, 0, 0
    preload, tstart, tstop, torque, speed = 0.0, 0.0, 1.0e30, 0.0, 0.0
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("PRELOAD_BOLT_1")
            sect_id = _ival(f[0]) if len(f) > 0 else 0
            sens_id = _ival(f[1]) if len(f) > 1 else 0
            fct_id = _ival(f[2]) if len(f) > 2 else 0
            preload = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tstart = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            tstop = _fval(f[5], 1.0e30) if len(f) > 5 and _fval(f[5]) > 0.0 else 1.0e30
            torque = _fval(f[6], 0.0) if len(f) > 6 else 0.0
        else:
            toks = cards[0].tokens()
            sect_id = int(float(toks[0])) if len(toks) > 0 else 0
            sens_id = int(float(toks[1])) if len(toks) > 1 else 0
            fct_id = int(float(toks[2])) if len(toks) > 2 else 0
            preload = float(toks[3]) if len(toks) > 3 else 0.0
            tstart = float(toks[4]) if len(toks) > 4 else 0.0
            tstop = float(toks[5]) if len(toks) > 5 else 1.0e30
            torque = float(toks[6]) if len(toks) > 6 else 0.0
    model.preload_bolts[bid] = PreloadBolt(
        id=bid, title=title, sect_id=sect_id, sens_id=sens_id, fct_id=fct_id,
        preload=preload, tstart=tstart, tstop=tstop, torque=torque, speed=speed
    )




def read_convec(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONVEC/convec_ID`` (M94)::

        card 1:  title
        card 2:  surf_ID  funct_ID  sensor_ID
        card 3:  Ascale   Fscale    Tstart   Tstop   H
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONVEC/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("CONVEC_1")
        surf_id = _ival(f[0])
        funct_id = _ival(f[1]) if len(f) > 1 else 0
        sens_id = _ival(f[2]) if len(f) > 2 else 0

        g = cards[1].cut("CONVEC_2") if len(cards) > 1 and not cards[1].is_blank else []
        xscale = _fval(g[0], 1.0) if len(g) > 0 else 1.0
        scale = _fval(g[1], 1.0) if len(g) > 1 else 1.0
        tstart = _fval(g[2], 0.0) if len(g) > 2 else 0.0
        tstop = _fval(g[3], 1.0e30) if len(g) > 3 else 1.0e30
        h = _fval(g[4], 0.0) if len(g) > 4 else 0.0
    else:
        t0 = cards[0].tokens()
        surf_id = int(t0[0]) if len(t0) > 0 else 0
        funct_id = int(t0[1]) if len(t0) > 1 else 0
        sens_id = int(t0[2]) if len(t0) > 2 else 0

        t1 = cards[1].tokens() if len(cards) > 1 and not cards[1].is_blank else []
        xscale = float(t1[0]) if len(t1) > 0 else 1.0
        scale = float(t1[1]) if len(t1) > 1 else 1.0
        tstart = float(t1[2]) if len(t1) > 2 else 0.0
        tstop = float(t1[3]) if len(t1) > 3 else 1.0e30
        h = float(t1[4]) if len(t1) > 4 else 0.0

    if xscale == 0.0:
        xscale = 1.0
    if scale == 0.0:
        scale = 1.0
    if tstop == 0.0:
        tstop = 1.0e30

    cl = ConvectionLoad(
        id=block.user_id, surf_id=surf_id, funct_id=funct_id,
        sens_id=sens_id, xscale=xscale, scale=scale,
        tstart=tstart, tstop=tstop, h=h, title=title,
    )
    model.convec_loads.append(cl)
    from ...model.entities import HeatConvec
    model.heat_convecs[block.user_id] = HeatConvec(
        id=block.user_id, title=title, surf_id=surf_id, funct_id=funct_id,
        sensor_id=sens_id, ascale=xscale, fscale=scale, tstart=tstart, tstop=tstop, h=h
    )




def read_radiation(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RADIATION/rad_ID`` (M95)::

        card 1:  title
        card 2:  surf_ID  funct_ID  sensor_ID
        card 3:  Ascale   Fscale    Tstart   Tstop   E
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RADIATION/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("RADIATION_1")
        surf_id = _ival(f[0])
        funct_id = _ival(f[1]) if len(f) > 1 else 0
        sens_id = _ival(f[2]) if len(f) > 2 else 0

        g = cards[1].cut("RADIATION_2") if len(cards) > 1 and not cards[1].is_blank else []
        xscale = _fval(g[0], 1.0) if len(g) > 0 else 1.0
        scale = _fval(g[1], 1.0) if len(g) > 1 else 1.0
        tstart = _fval(g[2], 0.0) if len(g) > 2 else 0.0
        tstop = _fval(g[3], 1.0e30) if len(g) > 3 else 1.0e30
        emissivity = _fval(g[4], 0.0) if len(g) > 4 else 0.0
    else:
        t0 = cards[0].tokens()
        surf_id = int(t0[0]) if len(t0) > 0 else 0
        funct_id = int(t0[1]) if len(t0) > 1 else 0
        sens_id = int(t0[2]) if len(t0) > 2 else 0

        t1 = cards[1].tokens() if len(cards) > 1 and not cards[1].is_blank else []
        xscale = float(t1[0]) if len(t1) > 0 else 1.0
        scale = float(t1[1]) if len(t1) > 1 else 1.0
        tstart = float(t1[2]) if len(t1) > 2 else 0.0
        tstop = float(t1[3]) if len(t1) > 3 else 1.0e30
        emissivity = float(t1[4]) if len(t1) > 4 else 0.0

    if xscale == 0.0:
        xscale = 1.0
    if scale == 0.0:
        scale = 1.0
    if tstop == 0.0:
        tstop = 1.0e30

    rl = RadiationLoad(
        id=block.user_id, surf_id=surf_id, funct_id=funct_id,
        sens_id=sens_id, xscale=xscale, scale=scale,
        tstart=tstart, tstop=tstop, emissivity=emissivity, title=title,
    )
    model.radiation_loads.append(rl)
    from ...model.entities import HeatRadiation
    model.heat_radiations[block.user_id] = HeatRadiation(
        id=block.user_id, title=title, surf_id=surf_id, funct_id=funct_id,
        sensor_id=sens_id, ascale=xscale, fscale=scale, tstart=tstart, tstop=tstop,
        emissivity=emissivity, emiss=emissivity
    )




def read_impflux(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPFLUX/impflux_ID`` (M95)::

        card 1:  title
        card 2:  surf_ID  funct_ID  sensor_ID  grbric_ID
        card 3:  Ascale   Fscale    Tstart     Tstop
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/IMPFLUX/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("IMPFLUX_1")
        surf_id = _ival(f[0])
        funct_id = _ival(f[1]) if len(f) > 1 else 0
        sens_id = _ival(f[2]) if len(f) > 2 else 0
        grbric_id = _ival(f[3]) if len(f) > 3 else 0

        g = cards[1].cut("IMPFLUX_2") if len(cards) > 1 and not cards[1].is_blank else []
        xscale = _fval(g[0], 1.0) if len(g) > 0 else 1.0
        scale = _fval(g[1], 1.0) if len(g) > 1 else 1.0
        tstart = _fval(g[2], 0.0) if len(g) > 2 else 0.0
        tstop = _fval(g[3], 1.0e30) if len(g) > 3 else 1.0e30
    else:
        t0 = cards[0].tokens()
        surf_id = int(t0[0]) if len(t0) > 0 else 0
        funct_id = int(t0[1]) if len(t0) > 1 else 0
        sens_id = int(t0[2]) if len(t0) > 2 else 0
        grbric_id = int(t0[3]) if len(t0) > 3 else 0

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

    fl = ImposedFlux(
        id=block.user_id, surf_id=surf_id, funct_id=funct_id,
        sens_id=sens_id, grbric_id=grbric_id, xscale=xscale,
        scale=scale, tstart=tstart, tstop=tstop, title=title,
    )
    model.impflux_loads.append(fl)
    model.impfluxes[block.user_id] = fl




def read_centri(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CENTRI/id`` (M198): Centrifugal loading field."""
    centri_id = block.user_id or (len(model.centris) + 1)
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/CENTRI/{centri_id}: missing data card", block.source)
        return
    grnd_id, sens_id, fct_id, node_orig, node_axis = 0, 0, 0, 0, 0
    omega = 0.0
    scale_x, scale_y, scale_z = 1.0, 1.0, 1.0
    is_fixed = block.fixed and ("," not in cards[0].raw)
    if is_fixed:
        try:
            f1 = cards[0].cut("CENTRI_1")
            grnd_id = _ival(f1[0]) if len(f1) > 0 else 0
            sens_id = _ival(f1[1]) if len(f1) > 1 else 0
            fct_id = _ival(f1[2]) if len(f1) > 2 else 0
            node_orig = _ival(f1[3]) if len(f1) > 3 else 0
            node_axis = _ival(f1[4]) if len(f1) > 4 else 0
            omega = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0
            if len(cards) > 1 and ("," not in cards[1].raw):
                f2 = cards[1].cut("CENTRI_2")
                scale_x = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
                scale_y = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
                scale_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
        except ValueError:
            is_fixed = False
    if not is_fixed:
        t1 = cards[0].raw.replace(",", " ").split()
        if len(t1) > 0:
            grnd_id = _safe_int(t1[0])
        if len(t1) > 1:
            sens_id = _safe_int(t1[1])
        if len(t1) > 2:
            fct_id = _safe_int(t1[2])
        if len(t1) > 3:
            node_orig = _safe_int(t1[3])
        if len(t1) > 4:
            node_axis = _safe_int(t1[4])
        if len(t1) > 5:
            omega = _safe_float(t1[5])
        if len(cards) > 1:
            t2 = cards[1].raw.replace(",", " ").split()
            if len(t2) > 0:
                scale_x = _safe_float(t2[0])
            if len(t2) > 1:
                scale_y = _safe_float(t2[1])
            if len(t2) > 2:
                scale_z = _safe_float(t2[2])
    c = CentrifugalLoad(id=centri_id, title=title, grnd_id=grnd_id, sens_id=sens_id,
                        fct_id=fct_id, node_orig=node_orig, node_axis=node_axis,
                        omega=omega, scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)
    c.grnod_id = grnd_id
    c.funct_id = fct_id
    model.centris[centri_id] = c
    if not any(cl.id == centri_id for cl in model.centri_loads):
        model.centri_loads.append(c)
