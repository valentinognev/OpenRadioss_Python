# -*- coding: utf-8 -*-
"""
Output request and sensor readers - /KEYWORD blocks -> Model.

/TH, /ANIM, /TFILE, /H3D, /PRINT and the /SENSOR measurement family.

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





def read_sensor(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/{TIME|DISP|VEL|NOT|AND|OR|DIST|ENERGY|INTER|RBODY|TEMP}/sens_ID`` (M6, M84, M97)::

        /SENSOR/TIME:   card 1: title, card 2: Tdelay
        /SENSOR/DISP:   card 1: title, card 2: Tdelay, card 3: node_ID Dmin
        /SENSOR/VEL:    card 1: title, card 2: Tdelay, card 3: node_ID Vmax Fcut
        /SENSOR/NOT:    card 1: title, card 2: Tdelay, card 3: sens_ID1
        /SENSOR/AND:    card 1: title, card 2: Tdelay, card 3: sens_ID1 sens_ID2
        /SENSOR/OR:     card 1: title, card 2: Tdelay, card 3: sens_ID1 sens_ID2
        /SENSOR/DIST:   card 1: title, card 2: Tdelay, card 3: node_ID1 node_ID2 Dmin Dmax Tmin
        /SENSOR/ENERGY: card 1: title, card 2: Tdelay, card 3: part_ID subset_ID Iselect, card 4: IEmin IEmax KEmin KEmax Tmin
        /SENSOR/INTER:  card 1: title, card 2: Tdelay, card 3: int_ID DIR Fmin Fmax Tmin Fcut
        /SENSOR/RBODY:  card 1: title, card 2: Tdelay, card 3: rbody_ID DIR Fmin Fmax Tmin
        /SENSOR/TEMP:   card 1: title, card 2: Tdelay, card 3: Grnod_Id Tempmax Tempmin Tempmean Tmin
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if not kind or kind.isdigit():
        kind = "TIME"
    if kind == "NIC_NIJ":
        kind = "NIC"
    elif kind == "TYPE10":
        kind = "GAUGE"
    elif kind == "TYPE12":
        kind = "XSECTION"
    elif kind == "TYPE13":
        kind = "WORK"
    elif kind == "TYPE16":
        kind = "HIC"
    elif kind == "TYPE17":
        kind = "DIST_SURF"
    if kind in ("GAP", "TIME_GAP"):
        read_sensor_gap(block, model, log)
        return
    if kind in ("ENERGY_RATIO", "ENG_RATIO", "RATIO_E"):
        read_sensor_energy_ratio(block, model, log)
        return
    if kind in ("CROSSSECTION", "SEC_FORCE", "SECT_FORCE", "SECT"):
        read_sensor_cross_section(block, model, log)
        return
    if kind in ("RUPT", "RUPTURE", "SHELL_FAIL", "SOLID_FAIL", "ELEM_FAIL"):
        read_sensor_rupture(block, model, log)
        return
    if kind in ("SHEAR", "SHEAR_STRESS", "TAU"):
        read_sensor_shear(block, model, log)
        return
    if kind in ("PRESSURE", "PRESS", "P"):
        read_sensor_pressure(block, model, log)
        return
    if kind in ("MASS", "MASS_RATIO", "DMASS"):
        read_sensor_mass_ratio(block, model, log)
        return
    if kind in ("ENERGY_ERROR", "ENG_ERROR", "EERROR"):
        read_sensor_energy_error(block, model, log)
        return
    if kind in ("WORK_RATIO", "WRATIO"):
        read_sensor_work_ratio(block, model, log)
        return
    if kind in ("SPRING", "SPRING_FORCE"):
        read_sensor_spring(block, model, log)
        return
    if kind in ("SHELL_STRAIN", "STRAIN_SHELL", "EPS_SHELL"):
        read_sensor_shell_strain(block, model, log)
        return
    if kind in ("SOLID_STRAIN", "STRAIN_SOLID", "EPS_SOLID"):
        read_sensor_solid_strain(block, model, log)
        return
    if kind in ("BEAM_STRAIN", "STRAIN_BEAM", "EPS_BEAM"):
        read_sensor_beam_strain(block, model, log)
        return
    if kind in ("TRUSS_STRAIN", "STRAIN_TRUSS", "EPS_TRUSS"):
        read_sensor_truss_strain(block, model, log)
        return
    if kind in ("SHELL_FORCE", "FORCE_SHELL"):
        read_sensor_shell_force(block, model, log)
        return
    if kind in ("SOLID_FORCE", "FORCE_SOLID"):
        read_sensor_solid_force(block, model, log)
        return
    if kind in ("BEAM_FORCE", "FORCE_BEAM"):
        read_sensor_beam_force(block, model, log)
        return
    if kind in ("TRUSS_FORCE", "FORCE_TRUSS"):
        read_sensor_truss_force(block, model, log)
        return
    if kind in ("SPRING_ENERGY", "ENERGY_SPRING", "SPRING_ENER"):
        read_sensor_spring_energy(block, model, log)
        return
    if kind in ("SPRING_DEFL", "SPRING_DEF", "DEF_SPRING", "DEFL_SPRING"):
        read_sensor_spring_defl(block, model, log)
        return
    if kind in ("SPRING_ROT", "SPRING_ROTATION", "ROT_SPRING", "ROTATION_SPRING"):
        read_sensor_spring_rot(block, model, log)
        return
    if kind in ("SPRING_ROTV", "SPRING_ANGVEL", "ROTV_SPRING", "ANGVEL_SPRING"):
        read_sensor_spring_rotv(block, model, log)
        return
    if kind in ("SPRING_ROTA", "SPRING_ANGACC", "ROTA_SPRING", "ANGACC_SPRING"):
        read_sensor_spring_rota(block, model, log)
        return
    if kind in ("SPRING_AXIAL", "SPRING_AXIAL_FORCE", "AXIAL_SPRING", "SPRING_TENSION"):
        read_sensor_spring_axial(block, model, log)
        return
    if kind in ("SPRING_SHEAR", "SPRING_SHEAR_FORCE", "SHEAR_SPRING", "SPRING_TRANSVERSE_FORCE"):
        read_sensor_spring_shear(block, model, log)
        return
    if kind in ("SPRING_BEND", "SPRING_BENDING", "BEND_SPRING", "SPRING_MOMENT"):
        read_sensor_spring_bend(block, model, log)
        return
    if kind in ("SPRING_TORSION", "SPRING_TORSIONAL", "TORSION_SPRING", "SPRING_TORQUE"):
        read_sensor_spring_torsion(block, model, log)
        return
    if kind in ("SPRING_STRAIN_ENERGY", "SPRING_SE", "STRAIN_ENERGY_SPRING", "SPRING_INTERNAL_ENERGY"):
        read_sensor_spring_strain_energy(block, model, log)
        return
    if kind in ("SPRING_KINETIC_ENERGY", "SPRING_KE", "KINETIC_ENERGY_SPRING", "SPRING_KIN_ENERGY"):
        read_sensor_spring_kinetic_energy(block, model, log)
        return
    supported = ("TIME", "DISP", "VEL", "NOT", "AND", "OR", "SENS_AND_OR", "LOGIC", "DIST", "ENERGY", "INTER", "RBODY", "TEMP", "NIC", "NIC_NIJ", "GAUGE", "HIC", "WORK", "RWALL", "XSECTION", "CROSSSECTION", "SECT", "DIST_SURF", "ACCE", "ACC", "ACCEL", "TYPE1", "SENS", "TYPE3", "TYPE10", "TYPE12", "TYPE13", "TYPE16", "TYPE17", "PYTHON", "SPH", "AIRBAG", "MONVOL", "SHELL", "SOLID", "RWALL_CYL", "RWALL_PLANE", "PLANE", "BOX", "FORCE", "MOMENT", "GEOM", "REL", "RATIO", "ENERGY_RATIO", "SHEAR_LOCK", "GAP", "TIME_GAP")
    if kind not in supported:
        log.warning(f"/SENSOR/{kind} not ported ({', '.join(supported)} supported)",
                    block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/SENSOR/{block.user_id}: missing data card",
                  block.source)
        return

    tdelay = 0.0
    data_card_idx = 0
    if len(cards) > 1 and kind != "TIME":
        t0 = cards[0].tokens()
        if t0:
            try:
                tdelay = float(t0[0])
            except ValueError:
                pass
        data_card_idx = 1

    t = cards[data_card_idx].tokens()
    if kind == "TIME":
        tdelay = float(t[0]) if t else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="TIME", tdelay=tdelay, title=title))
    elif kind == "DISP":
        if len(t) < 2:
            log.error(f"/SENSOR/DISP/{block.user_id}: card needs "
                      f"'node_ID Dmin'", block.source)
            return
        dmin = float(t[1])
        if dmin <= 0.0:
            log.error(f"/SENSOR/DISP/{block.user_id}: Dmin must be > 0",
                      block.source)
            return
        model.sensors.append(Sensor(
            id=block.user_id, kind="DISP", tdelay=tdelay, node_id=int(t[0]), dmin=dmin,
            title=title))
    elif kind == "VEL":
        if len(t) < 2:
            log.error(f"/SENSOR/VEL/{block.user_id}: card needs "
                      f"'node_ID Vmax'", block.source)
            return
        node_id = int(t[0])
        vmax = float(t[1])
        fcut = float(t[2]) if len(t) > 2 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="VEL", tdelay=tdelay, node_id=node_id,
            vmax=vmax, fcut=fcut, title=title))
    elif kind == "NOT":
        if len(t) < 1:
            log.error(f"/SENSOR/NOT/{block.user_id}: card needs "
                      f"'sens_ID1'", block.source)
            return
        sens_id1 = int(t[0])
        model.sensors.append(Sensor(
            id=block.user_id, kind="NOT", tdelay=tdelay, sens_id1=sens_id1,
            title=title))
    elif kind in ("AND", "OR", "SENS_AND_OR", "SENS"):
        if len(t) < 2:
            log.error(f"/SENSOR/{kind}/{block.user_id}: card needs "
                      f"'sens_ID1 sens_ID2'", block.source)
            return
        sens_id1 = int(t[0])
        sens_id2 = int(t[1])
        s_kind = "AND" if "AND" in kind else "OR" if "OR" in kind else kind
        s_obj = Sensor(
            id=block.user_id, kind=s_kind, tdelay=tdelay, sens_id1=sens_id1,
            sens_id2=sens_id2, title=title)
        model.sensors.append(s_obj)
        from ...model.entities import SensorSensAndOr
        model.sensors_sens_and_or[block.user_id] = SensorSensAndOr(
            id=block.user_id, title=title, logic_type=s_kind,
            sensor_id1=sens_id1, sensor_id2=sens_id2,
            sens_id1=sens_id1, sens_id2=sens_id2,
            t_delay=tdelay, tdelay=tdelay
        )
    elif kind == "DIST":
        dflag = 0
        if block.fixed:
            raw = cards[data_card_idx].raw
            if len(raw) > 70:
                f = cards[data_card_idx].cut("SENSOR_DIST_22")
                n1 = _ival(f[0])
                n2 = _ival(f[1])
                dmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                dmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
                dflag = _ival(f[5]) if len(f) > 5 else 0
            else:
                f = cards[data_card_idx].cut("SENSOR_DIST_2")
                n1 = _ival(f[0])
                n2 = _ival(f[1])
                dmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                dmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            if len(t) < 2:
                log.error(f"/SENSOR/DIST/{block.user_id}: card needs 'node_ID1 node_ID2'", block.source)
                return
            n1, n2 = int(t[0]), int(t[1])
            dmin = float(t[2]) if len(t) > 2 else 0.0
            dmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
            dflag = int(float(t[5])) if len(t) > 5 else 0
        model.sensors.append(Sensor(
            id=block.user_id, kind="DIST", tdelay=tdelay, node_id1=n1, node_id2=n2,
            dmin=dmin, dmax=dmax, tmin=tmin, dflag=dflag, title=title))
    elif kind == "ENERGY":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_ENERGY_2")
            part_id = _ival(f[0])
            subset_id = _ival(f[1]) if len(f) > 1 else 0
            iselect = _ival(f[2], 1) if len(f) > 2 else 1
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].cut("SENSOR_ENERGY_3")
                iemin = _fval(g[0], -1e30) if len(g) > 0 else -1e30
                iemax = _fval(g[1], 1e30) if len(g) > 1 else 1e30
                kemin = _fval(g[2], -1e30) if len(g) > 2 else -1e30
                kemax = _fval(g[3], 1e30) if len(g) > 3 else 1e30
                tmin = _fval(g[4], 0.0) if len(g) > 4 else 0.0
            else:
                iemin, iemax, kemin, kemax, tmin = -1e30, 1e30, -1e30, 1e30, 0.0
        else:
            part_id = int(t[0]) if len(t) > 0 else 0
            subset_id = int(t[1]) if len(t) > 1 else 0
            iselect = int(t[2]) if len(t) > 2 else 1
            if data_card_idx + 1 < len(cards):
                t2 = cards[data_card_idx + 1].tokens()
                iemin = float(t2[0]) if len(t2) > 0 else -1e30
                iemax = float(t2[1]) if len(t2) > 1 else 1e30
                kemin = float(t2[2]) if len(t2) > 2 else -1e30
                kemax = float(t2[3]) if len(t2) > 3 else 1e30
                tmin = float(t2[4]) if len(t2) > 4 else 0.0
            else:
                iemin, iemax, kemin, kemax, tmin = -1e30, 1e30, -1e30, 1e30, 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="ENERGY", tdelay=tdelay, part_id=part_id, subset_id=subset_id,
            iselect=iselect, iemin=iemin, iemax=iemax, kemin=kemin, kemax=kemax, tmin=tmin, title=title))
    elif kind == "INTER":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_INTER_2")
            int_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            fmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            fcut = _fval(f[5], 0.0) if len(f) > 5 else 0.0
        else:
            int_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            fmin = float(t[2]) if len(t) > 2 else 0.0
            fmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
            fcut = float(t[5]) if len(t) > 5 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="INTER", tdelay=tdelay, int_id=int_id, dir=sdir,
            fmin=fmin, fmax=fmax, tmin=tmin, fcut=fcut, title=title))
    elif kind == "RBODY":
        if block.fixed:
            if data_card_idx + 1 < len(cards) and not cards[data_card_idx + 1].is_blank:
                f1 = cards[data_card_idx].cut("SENSOR_RBODY_1")
                node_id = _ival(f1[0]) if len(f1) > 0 else 0
                rb_id = _ival(f1[1]) if len(f1) > 1 else 0
                sdir = f1[3].strip() if len(f1) > 3 else ""

                f2 = cards[data_card_idx + 1].cut("SENSOR_RBODY_VAL")
                fmin = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                fmax = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                tmin = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            else:
                f = cards[data_card_idx].cut("SENSOR_RBODY_2")
                rb_id = _ival(f[0]) if len(f) > 0 else 0
                sdir = f[1].strip() if len(f) > 1 else ""
                fmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
                node_id = 0
        else:
            if data_card_idx + 1 < len(cards) and not cards[data_card_idx + 1].is_blank:
                node_id = int(float(t[0])) if len(t) > 0 else 0
                rb_id = int(float(t[1])) if len(t) > 1 else 0
                sdir = str(t[3]) if len(t) > 3 else ""
                t2 = cards[data_card_idx + 1].tokens()
                fmin = float(t2[0]) if len(t2) > 0 else 0.0
                fmax = float(t2[1]) if len(t2) > 1 else 0.0
                tmin = float(t2[2]) if len(t2) > 2 else 0.0
            else:
                rb_id = int(t[0]) if len(t) > 0 else 0
                sdir = t[1] if len(t) > 1 else ""
                fmin = float(t[2]) if len(t) > 2 else 0.0
                fmax = float(t[3]) if len(t) > 3 else 0.0
                tmin = float(t[4]) if len(t) > 4 else 0.0
                node_id = 0
        model.sensors.append(Sensor(
            id=block.user_id, kind="RBODY", tdelay=tdelay, node_id=node_id, rbody_id=rb_id, dir=sdir,
            fmin=fmin, fmax=fmax, tmin=tmin, title=title))
    elif kind == "TEMP":
        if block.fixed:
            raw = cards[data_card_idx].raw
            if len(raw) > 70:
                f = cards[data_card_idx].cut("SENSOR_TEMP_1")
                grnod_id = _ival(f[0])
                tempmax = _fval(f[2], 1e30) if len(f) > 2 and f[2].strip() else 1e30
                tempmin = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                tempmean = _fval(f[4], 1e30) if len(f) > 4 and f[4].strip() else 1e30
                tmin = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            else:
                f = cards[data_card_idx].cut("SENSOR_TEMP_2")
                grnod_id = _ival(f[0])
                tempmax = _fval(f[1], 1e30) if len(f) > 1 and f[1].strip() else 1e30
                tempmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                tempmean = _fval(f[3], 1e30) if len(f) > 3 and f[3].strip() else 1e30
                tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            grnod_id = int(float(t[0])) if len(t) > 0 else 0
            tempmax = float(t[1]) if len(t) > 1 else 1e30
            tempmin = float(t[2]) if len(t) > 2 else 0.0
            tempmean = float(t[3]) if len(t) > 3 else 1e30
            tmin = float(t[4]) if len(t) > 4 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="TEMP", tdelay=tdelay, grnod_id=grnod_id,
            tempmax=tempmax, tempmin=tempmin, tempmean=tempmean, tmin=tmin, title=title))
    elif kind == "NIC":
        nij_max, fint_tens, fint_comp, mint_flex, mint_ext = 0.0, 0.0, 0.0, 0.0, 0.0
        if len(cards) > data_card_idx and not cards[data_card_idx].is_blank:
            c2 = cards[data_card_idx].cut("SENSOR_NIC_2") if block.fixed else cards[data_card_idx].tokens()
            nij_max = _fval(c2[0]) if len(c2) > 0 else 0.0
            fint_tens = _fval(c2[1]) if len(c2) > 1 else 0.0
            fint_comp = _fval(c2[2]) if len(c2) > 2 else 0.0
            mint_flex = _fval(c2[3]) if len(c2) > 3 else 0.0
            mint_ext = _fval(c2[4]) if len(c2) > 4 else 0.0

        spring_id, skew_id, ax_dir, bend_dir = 0, 0, "", ""
        if len(cards) > data_card_idx + 1 and not cards[data_card_idx + 1].is_blank:
            c3 = cards[data_card_idx + 1].cut("SENSOR_NIC_3") if block.fixed else cards[data_card_idx + 1].tokens()
            spring_id = _ival(c3[0]) if len(c3) > 0 else 0
            skew_id = _ival(c3[1]) if len(c3) > 1 else 0
            ax_dir = c3[2].strip() if len(c3) > 2 else ""
            bend_dir = c3[3].strip() if len(c3) > 3 else ""

        tmin, alpha, cfc = 0.0, 0.0, 0.0
        if len(cards) > data_card_idx + 2 and not cards[data_card_idx + 2].is_blank:
            c4 = cards[data_card_idx + 2].cut("SENSOR_NIC_4") if block.fixed else cards[data_card_idx + 2].tokens()
            tmin = _fval(c4[0]) if len(c4) > 0 else 0.0
            alpha = _fval(c4[1]) if len(c4) > 1 else 0.0
            cfc = _fval(c4[2]) if len(c4) > 2 else 0.0

        model.sensors.append(Sensor(
            id=block.user_id, kind="NIC", tdelay=tdelay, nij_max=nij_max,
            fint_tens=fint_tens, fint_comp=fint_comp, mint_flex=mint_flex,
            mint_ext=mint_ext, spring_id=spring_id, skew_id=skew_id,
            ax_dir=ax_dir, bend_dir=bend_dir, tmin=tmin, alpha=alpha,
            cfc=cfc, title=title,
        ))
    elif kind == "GAUGE":
        entries = []
        if data_card_idx < len(cards):
            ngau_tok = cards[data_card_idx].tokens()
            ngau = int(ngau_tok[0]) if ngau_tok else 0
            for k in range(data_card_idx + 1, data_card_idx + 1 + ngau):
                if k < len(cards):
                    if block.fixed:
                        f = cards[k].cut("SENSOR_GAUGE_3")
                        gid = _ival(f[0])
                        fp = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                        ft = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                    else:
                        gt = cards[k].tokens()
                        gid = int(gt[0]) if len(gt) > 0 else 0
                        fp = float(gt[1]) if len(gt) > 1 else 0.0
                        ft = float(gt[2]) if len(gt) > 2 else 0.0
                    entries.append((gid, fp, ft))
        model.sensors.append(Sensor(
            id=block.user_id, kind="GAUGE", tdelay=tdelay, gauge_entries=entries, title=title))
    elif kind == "HIC":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_HIC_2")
            accel_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            hic_p = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            hic_v = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            grav = _fval(f[4], 9.81) if len(f) > 4 else 9.81
            tmin = _fval(f[5], 0.0) if len(f) > 5 else 0.0
        else:
            accel_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            hic_p = float(t[2]) if len(t) > 2 else 0.0
            hic_v = float(t[3]) if len(t) > 3 else 0.0
            grav = float(t[4]) if len(t) > 4 else 9.81
            tmin = float(t[5]) if len(t) > 5 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="HIC", tdelay=tdelay, accel_id=accel_id, dir=sdir,
            hic_period=hic_p, hic_val=hic_v, gravity=grav, tmin=tmin, title=title))
    elif kind == "WORK":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_WORK_2")
            n1 = _ival(f[0])
            n2 = _ival(f[1]) if len(f) > 1 else 0
            wmax = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            tmin = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            sect_id, int_id, rb_id, rw_id = 0, 0, 0, 0
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].cut("SENSOR_WORK_3")
                sect_id = _ival(g[0]) if len(g) > 0 else 0
                int_id = _ival(g[1]) if len(g) > 1 else 0
                rb_id = _ival(g[2]) if len(g) > 2 else 0
                rw_id = _ival(g[3]) if len(g) > 3 else 0
        else:
            n1 = int(t[0]) if len(t) > 0 else 0
            n2 = int(t[1]) if len(t) > 1 else 0
            wmax = float(t[2]) if len(t) > 2 else 0.0
            tmin = float(t[3]) if len(t) > 3 else 0.0
            sect_id, int_id, rb_id, rw_id = 0, 0, 0, 0
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].tokens()
                sect_id = int(g[0]) if len(g) > 0 else 0
                int_id = int(g[1]) if len(g) > 1 else 0
                rb_id = int(g[2]) if len(g) > 2 else 0
                rw_id = int(g[3]) if len(g) > 3 else 0
        model.sensors.append(Sensor(
            id=block.user_id, kind="WORK", tdelay=tdelay, node_id1=n1, node_id2=n2,
            work_max=wmax, tmin=tmin, sect_id=sect_id, int_id=int_id, rbody_id=rb_id,
            rwall_id=rw_id, title=title))
        from ...model.entities import SensorWork
        model.sensors_work[block.user_id] = SensorWork(
            id=block.user_id, title=title, object_id=n1, sens_type=n2,
            t_delay=tdelay, w_max=wmax, tmin=tmin, sect_id=sect_id,
            int_id=int_id, rbody_id=rb_id, rwall_id=rw_id,
            node_id1=n1, node_id2=n2
        )
    elif kind == "RWALL":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_RWALL_2")
            rwall_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            fmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            rwall_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            fmin = float(t[2]) if len(t) > 2 else 0.0
            fmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="RWALL", tdelay=tdelay, rwall_id=rwall_id, dir=sdir,
            fmin=fmin, fmax=fmax, tmin=tmin, title=title))
    elif kind in ("XSECTION", "CROSSSECTION", "SECT"):
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_XSECTION_2")
            sect_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            fmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            sect_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            fmin = float(t[2]) if len(t) > 2 else 0.0
            fmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="XSECTION", tdelay=tdelay, sect_id=sect_id, dir=sdir,
            fmin=fmin, fmax=fmax, tmin=tmin, title=title))
    elif kind == "DIST_SURF":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_DIST_SURF_2")
            n1 = _ival(f[0])
            surf_id = _ival(f[1]) if len(f) > 1 else 0
            n2 = _ival(f[2]) if len(f) > 2 else 0
            n3 = _ival(f[3]) if len(f) > 3 else 0
            n4 = _ival(f[4]) if len(f) > 4 else 0
            dmin, dmax, tmin = 0.0, 0.0, 0.0
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].cut("SENSOR_DIST_SURF_3")
                dmin = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                dmax = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                tmin = _fval(g[3], 0.0) if len(g) > 3 else 0.0
        else:
            if len(t) == 4:
                n1 = int(float(t[0]))
                surf_id = 0
                n2 = int(float(t[1]))
                n3 = int(float(t[2]))
                n4 = int(float(t[3]))
            else:
                n1 = int(float(t[0])) if len(t) > 0 else 0
                surf_id = int(float(t[1])) if len(t) > 1 else 0
                n2 = int(float(t[2])) if len(t) > 2 else 0
                n3 = int(float(t[3])) if len(t) > 3 else 0
                n4 = int(float(t[4])) if len(t) > 4 else 0
            dmin, dmax, tmin = 0.0, 0.0, 0.0
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].tokens()
                dmin = float(g[0]) if len(g) > 0 else 0.0
                dmax = float(g[1]) if len(g) > 1 else 0.0
                if len(g) >= 4:
                    tmin = float(g[3])
                elif len(g) >= 3:
                    tmin = float(g[2])
        model.sensors.append(Sensor(
            id=block.user_id, kind="DIST_SURF", tdelay=tdelay, node_id=n1, node_id1=n1, surf_id=surf_id,
            node_id2=n2, node_id3=n3, node_id4=n4, dmin=dmin, dmax=dmax, tmin=tmin, title=title))
        from ...model.entities import SensorDistSurf
        model.sensors_dist_surf[block.user_id] = SensorDistSurf(
            id=block.user_id, title=title, tdelay=tdelay, surf_id=surf_id,
            node_id=n1, surf_target_id=0, node_id1=n2, node_id2=n3, node_id3=n4,
            dist_min=dmin, dist_max=dmax, tmin=tmin
        )
    elif kind in ("ACCE", "ACC", "ACCEL", "TYPE1"):
        nacc = 1
        if block.fixed:
            f = cards[0].cut("SENSOR_ACCE_1")
            tdelay = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            nacc = _ival(f[1]) if len(f) > 1 and f[1].strip() else 1
        else:
            toks0 = cards[0].tokens()
            tdelay = float(toks0[0]) if len(toks0) > 0 else 0.0
            nacc = int(float(toks0[1])) if len(toks0) > 1 else 1

        acc_entries = []
        for c in cards[1: 1 + max(1, nacc)]:
            if c.is_blank:
                continue
            if block.fixed:
                f = c.cut("SENSOR_ACCE_ITEM")
                iacc = _ival(f[0]) if len(f) > 0 else 0
                sdir = f[1].strip() if len(f) > 1 else ""
                tomin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                tmin = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            else:
                toks = c.tokens()
                iacc = int(float(toks[0])) if len(toks) > 0 else 0
                sdir = toks[1] if len(toks) > 1 else ""
                tomin = float(toks[2]) if len(toks) > 2 else 0.0
                tmin = float(toks[3]) if len(toks) > 3 else 0.0
            acc_entries.append((iacc, sdir, tomin, tmin))

        model.sensors.append(Sensor(
            id=block.user_id, kind="ACCE", tdelay=tdelay, acc_entries=acc_entries, title=title
        ))
    elif kind in ("SENS", "TYPE3"):
        tdelay = 0.0
        s1, s2 = 0, 0
        if len(cards) >= 2:
            toks0 = cards[0].tokens()
            tdelay = float(toks0[0]) if toks0 else 0.0
            if block.fixed:
                f = cards[1].cut("SENSOR_SENS_2")
                s1 = _ival(f[0]) if len(f) > 0 else 0
                s2 = _ival(f[1]) if len(f) > 1 else 0
            else:
                toks1 = cards[1].tokens()
                s1 = int(float(toks1[0])) if len(toks1) > 0 else 0
                s2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        elif len(cards) == 1:
            toks0 = cards[0].tokens()
            if len(toks0) >= 3:
                tdelay = float(toks0[0])
                s1 = int(float(toks0[1]))
                s2 = int(float(toks0[2]))
            elif len(toks0) == 2:
                s1 = int(float(toks0[0]))
                s2 = int(float(toks0[1]))
        model.sensors.append(Sensor(
            id=block.user_id, kind="SENS", tdelay=tdelay, sens_id1=s1, sens_id2=s2, title=title
        ))
        from ...model.entities import SensorSensAndOr
        model.sensors_sens_and_or[block.user_id] = SensorSensAndOr(
            id=block.user_id, title=title, logic_type="AND",
            sensor_id1=s1, sensor_id2=s2, sens_id1=s1, sens_id2=s2,
            t_delay=tdelay, tdelay=tdelay
        )
    elif kind == "PYTHON":
        from ...model.entities import SensorPython
        tdelay = 0.0
        script_name, func_name = "", ""
        code_lines = []
        if cards:
            for c in cards:
                if not c.is_blank:
                    code_lines.append(c.raw.rstrip())
            toks0 = cards[0].tokens()
            if toks0 and not any(k in toks0[0] for k in ("def", "return", "import", "=")):
                try:
                    tdelay = float(toks0[0])
                except (ValueError, TypeError):
                    tdelay = 0.0
        if len(cards) > 1:
            toks1 = cards[1].tokens()
            if toks1 and not any(k in toks1[0] for k in ("def", "return", "import", "=")):
                script_name = toks1[0] if len(toks1) > 0 else ""
                func_name = toks1[1] if len(toks1) > 1 else ""
        code_str = "\n".join(code_lines)
        sp = SensorPython(
            id=block.user_id, title=title, script_name=script_name, func_name=func_name,
            code=code_str, t_delay=tdelay, tdelay=tdelay, sensor_type="PYTHON"
        )
        model.sensors_python[block.user_id] = sp
        model.sensors.append(Sensor(
            id=block.user_id, kind="PYTHON", tdelay=tdelay, script_name=script_name, func_name=func_name, title=title
        ))
    elif kind in ("SPH", "AIRBAG", "MONVOL", "SHELL", "SOLID"):
        # M136, M204 subsystem sensors
        target_id = int(float(t[0])) if len(t) > 0 and t[0].strip() else 0
        v1 = float(t[1]) if len(t) > 1 and t[1].strip() else 0.0
        v2 = float(t[2]) if len(t) > 2 and t[2].strip() else 0.0
        tmin = float(t[3]) if len(t) > 3 and t[3].strip() else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind=kind, tdelay=tdelay, target_id=target_id,
            dmin=v1, dmax=v2, tmin=tmin, title=title
        ))
        from ...model.entities import SensorSubsystem
        ss = SensorSubsystem(
            id=block.user_id, title=title, kind=kind,
            target_id=target_id, v1=v1, v2=v2, tmin=tmin, tdelay=tdelay
        )
        if kind in ("AIRBAG", "MONVOL"):
            model.sensors_airbag[block.user_id] = ss
        elif kind == "SHELL":
            model.sensors_shell[block.user_id] = ss
        elif kind == "SOLID":
            model.sensors_solid[block.user_id] = ss
        elif kind == "SPH":
            model.sensors_sph[block.user_id] = ss
    elif kind in ("FORCE", "MOMENT", "RWALL_CYL", "RWALL_PLANE", "PLANE", "BOX"):
        target_id = int(float(t[0])) if len(t) > 0 and t[0].strip() else 0
        v1 = float(t[1]) if len(t) > 1 and t[1].strip() else 0.0
        v2 = float(t[2]) if len(t) > 2 and t[2].strip() else 0.0
        tmin = float(t[3]) if len(t) > 3 and t[3].strip() else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind=kind, tdelay=tdelay, target_id=target_id,
            dmin=v1, dmax=v2, tmin=tmin, title=title
        ))
    elif kind == "GEOM":
        itype, n1, n2, n3 = 1, 0, 0, 0
        vmin, vmax, tmin = 0.0, 0.0, 0.0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f1 = cards[0].cut("SENSOR_GEOM_1")
                itype = _ival(f1[0], 1) if len(f1) > 0 else 1
                n1 = _ival(f1[1]) if len(f1) > 1 else 0
                n2 = _ival(f1[2]) if len(f1) > 2 else 0
                n3 = _ival(f1[3]) if len(f1) > 3 else 0
                vmin = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
                vmax = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0
            else:
                t1 = cards[0].tokens()
                itype = int(float(t1[0])) if len(t1) > 0 else 1
                n1 = int(float(t1[1])) if len(t1) > 1 else 0
                n2 = int(float(t1[2])) if len(t1) > 2 else 0
                n3 = int(float(t1[3])) if len(t1) > 3 else 0
                vmin = float(t1[4]) if len(t1) > 4 else 0.0
                vmax = float(t1[5]) if len(t1) > 5 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            if block.fixed:
                f2 = cards[1].cut("SENSOR_GEOM_2")
                tmin = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                tdelay = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            else:
                t2 = cards[1].tokens()
                tmin = float(t2[0]) if len(t2) > 0 else 0.0
                tdelay = float(t2[1]) if len(t2) > 1 else 0.0
        from ...model.entities import SensorGeom
        sg = SensorGeom(id=block.user_id, title=title, itype=itype, node1=n1, node2=n2, node3=n3,
                        val_min=vmin, val_max=vmax, tmin=tmin, tdelay=tdelay)
        model.sensors_geom[block.user_id] = sg
        model.sensors.append(Sensor(id=block.user_id, kind="GEOM", tdelay=tdelay, tmin=tmin, title=title))
    elif kind == "REL":
        n1, n2, idir, skew_id = 0, 0, 1, 0
        vmin, vmax, tmin = 0.0, 0.0, 0.0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f1 = cards[0].cut("SENSOR_REL_1")
                n1 = _ival(f1[0]) if len(f1) > 0 else 0
                n2 = _ival(f1[1]) if len(f1) > 1 else 0
                idir = _ival(f1[2], 1) if len(f1) > 2 else 1
                skew_id = _ival(f1[3]) if len(f1) > 3 else 0
                vmin = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
                vmax = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0
            else:
                t1 = cards[0].tokens()
                n1 = int(float(t1[0])) if len(t1) > 0 else 0
                n2 = int(float(t1[1])) if len(t1) > 1 else 0
                idir = int(float(t1[2])) if len(t1) > 2 else 1
                skew_id = int(float(t1[3])) if len(t1) > 3 else 0
                vmin = float(t1[4]) if len(t1) > 4 else 0.0
                vmax = float(t1[5]) if len(t1) > 5 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            if block.fixed:
                f2 = cards[1].cut("SENSOR_REL_2")
                tmin = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                tdelay = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            else:
                t2 = cards[1].tokens()
                tmin = float(t2[0]) if len(t2) > 0 else 0.0
                tdelay = float(t2[1]) if len(t2) > 1 else 0.0
        from ...model.entities import SensorRel
        sr = SensorRel(id=block.user_id, title=title, node1=n1, node2=n2, idir=idir, skew_id=skew_id,
                       val_min=vmin, val_max=vmax, tmin=tmin, tdelay=tdelay)
        model.sensors_rel[block.user_id] = sr
        model.sensors.append(Sensor(id=block.user_id, kind="REL", tdelay=tdelay, tmin=tmin, title=title))
    elif kind in ("RATIO", "ENERGY_RATIO"):
        ratio_type, vmin, vmax = 1, 0.0, 0.0
        tmin, tdelay = 0.0, 0.0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f1 = cards[0].cut("SENSOR_RATIO_1")
                ratio_type = _ival(f1[0], 1) if len(f1) > 0 else 1
                vmin = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                vmax = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tmin = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tdelay = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
            else:
                t1 = cards[0].tokens()
                ratio_type = int(float(t1[0])) if len(t1) > 0 else 1
                vmin = float(t1[1]) if len(t1) > 1 else 0.0
                vmax = float(t1[2]) if len(t1) > 2 else 0.0
                tmin = float(t1[3]) if len(t1) > 3 else 0.0
                tdelay = float(t1[4]) if len(t1) > 4 else 0.0
        from ...model.entities import SensorRatio
        s_rat = SensorRatio(id=block.user_id, title=title, ratio_type=ratio_type,
                            val_min=vmin, val_max=vmax, tmin=tmin, tdelay=tdelay)
        model.sensors_ratio[block.user_id] = s_rat
        model.sensors.append(Sensor(id=block.user_id, kind="RATIO", tdelay=tdelay, tmin=tmin, title=title))
    elif kind == "SHEAR_LOCK":
        part_id = 0
        vmax, tmin, tdelay = 0.0, 0.0, 0.0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f1 = cards[0].cut("SENSOR_SHEAR_LOCK_1")
                part_id = _ival(f1[0]) if len(f1) > 0 else 0
                vmax = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                tmin = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tdelay = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
            else:
                t1 = cards[0].tokens()
                part_id = int(float(t1[0])) if len(t1) > 0 else 0
                vmax = float(t1[1]) if len(t1) > 1 else 0.0
                tmin = float(t1[2]) if len(t1) > 2 else 0.0
                tdelay = float(t1[3]) if len(t1) > 3 else 0.0
        from ...model.entities import SensorShearLock
        s_sl = SensorShearLock(id=block.user_id, title=title, part_id=part_id,
                               val_max=vmax, tmin=tmin, tdelay=tdelay)
        model.sensors_shear_lock[block.user_id] = s_sl
        model.sensors.append(Sensor(id=block.user_id, kind="SHEAR_LOCK", tdelay=tdelay, tmin=tmin, title=title))






def read_gauge_point(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GAUGE/POINT/id`` (M121): Point gauge definition for spatial measurement."""
    title, cards = _title_and_data(block)
    points = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("GAUGE_POINT_1")
            xi = _fval(f[0]) if len(f) > 0 else 0.0
            yi = _fval(f[1]) if len(f) > 1 else 0.0
            zi = _fval(f[2]) if len(f) > 2 else 0.0
            dist = _fval(f[3]) if len(f) > 3 else 0.0
            sub = f[4].strip() if len(f) > 4 else ""
        else:
            t = c.tokens()
            xi = float(t[0]) if len(t) > 0 else 0.0
            yi = float(t[1]) if len(t) > 1 else 0.0
            zi = float(t[2]) if len(t) > 2 else 0.0
            dist = float(t[3]) if len(t) > 3 else 0.0
            sub = t[4] if len(t) > 4 else ""
        points.append((xi, yi, zi, dist, sub))
    gid = block.user_id or 1
    model.gauge_points[gid] = GaugePoint(id=gid, title=title, points=points)




# ============================================================================
# Time-history requests
# ============================================================================

def read_th(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TH/NODE|PART|SECT|RBODY|SHEL|SH3N|SPRING|BRIC|RWALL|SECTIO|INTER/th_ID``::

        card 1:  title
        card 2:  variable names (e.g. ``DX DY DZ VX VY VZ``) or ``DEF``
        card 3+: object IDs (any number per card)

      DEF expands to the Radioss default set for the object type.
      SECT (M5) variables: FX FY FZ MX MY MZ (section force/moment).

    REAL dialect (cfg OUTPUTBLOCK/th_node.cfg / th_part.cfg,
    radioss51+; M37):

    * the variable names are a FREE_CELL_LIST (``%-10s``, 10 per card)
      that CONTINUES onto further cards — every leading card whose
      first field is non-numeric is variables ('DEF HE IE ... XMOM' +
      'XXMOM YCG ...'); ``DEF`` expands to the default set and extra
      names ride along.  Variable names the port does not record are
      kept in the request (the writer ignores unknown columns).
    * NODE ids: ``CARD("%10d%10d%-80s", id, skew, name)`` — ONE node
      per card whose skew/name columns abut the id ('        2
      01x3' = node 2, skew 0, name '1x3'; pre-M37 the token view
      crashed on int('01x3')).  Other kinds pack plain %10d ids.

    M68: expanded from NODE/PART/SECT to all entity types.
    """
    _TH_KINDS = {
        "NODE", "PART", "SECT", "RBODY", "SHEL", "SH3N",
        "SPRING", "BRIC", "RWALL", "SECTIO", "INTER",
        "RETRACTOR", "SLIPRING", "TRIA", "TETRA4", "BEAM", "TRUSS",
        "SHELL", "SOLID", "QUAD", "SURF", "LINE", "ACCEL", "BOX",
        "NSTRAND", "STRAND", "SPHCEL", "SPH", "MODE", "CYL_JO", "CYL_JOINT",
        "FXBODY", "GAUGE", "GRSHEL", "GRBRIC", "GRQUAD", "GRSH3N",
        "GRBEAM", "GRTRUS", "GRSPRI", "SENSOR", "CLUSTER",
        "MONVOL", "MONV", "AIRBAG", "FVMBAG", "COMMU", "ALE", "ALEGRID", "ALECFD",
        "SUBS", "SUBDOMAIN", "SUBMODEL", "LAGMUL", "GEAR", "RACK", "DIFF",
        "IMPDISP", "IMPVEL", "PLOAD", "PROP", "MAT", "STACK", "PLY",
        "WAVE_SHAPER", "DET", "GUIDED_CABLE", "KJOINT", "SUBINTER",
        "EBCS", "SEATBELT", "SPH_FLOW", "HEAT", "TEMPER", "THERM",
        "INITEMP", "IMPTEMP", "INICRACK", "XFEM", "FRAME"
    }
    if block.key0.startswith("THPART_") or (block.key0 == "THPART" and len(block.parts) > 1 and block.parts[1].upper().startswith("GR")):
        read_thpart_group(block, model, log)
        return
    elif block.key0 == "THPART":
        kind = "PART"
    elif block.key0.startswith(("TH", "ATH", "BTH", "CTH", "DTH", "ETH", "FTH", "GTH", "HTH", "ITH")) and "_" in block.key0:
        kind = block.key0.split("_", 1)[1].upper()
    elif block.key0 in ("TH", "ATH", "BTH", "CTH", "DTH", "ETH", "FTH", "GTH", "HTH", "ITH"):
        kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    else:
        kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    if kind == "TITLE":
        read_th_title(block, model, log)
        return
    # /TH/SECTIO is the Fortran spelling; normalise to SECT for the model
    if kind == "SECTIO":
        kind = "SECT"
    elif kind == "SHELL":
        kind = "SHEL"
    elif kind == "SOLID":
        kind = "BRIC"
    elif kind == "TRIA":
        kind = "SH3N"
    elif kind == "STRAND":
        kind = "NSTRAND"
    elif kind == "SPH":
        kind = "SPHCEL"
    elif kind == "CYL_JOINT":
        kind = "CYL_JO"
    elif kind == "SUBINTER":
        kind = "SUBS"
    elif kind == "INTER_GUIDED_CABLE":
        kind = "GUIDED_CABLE"
    elif kind == "MONV":
        kind = "MONVOL"
    if kind not in _TH_KINDS:
        log.warning(f"/TH/{kind} not ported", block.source)
        return
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    if len(cards) < 2:
        log.error(f"/TH/{kind}/{block.user_id}: needs variables + ids cards",
                  block.source)
        return

    # -- variable cards: every leading card starting with a non-number ------
    # Variable cards use FREE_CELL_LIST("%-10s", VAR, 100) — left-justified
    # 10-char text cells (e.g. "DEF       EPSD      THIC      ").
    # ID cards use CARD("%10d ...") — right-justified integer in cols 1-10,
    # optionally followed by skew/name fields (e.g. "         1         0Name").
    # In fixed format, whitespace tokenisation mis-classifies ID cards when the
    # element name text happens to be the first whitespace token (the %10d ID
    # field is blank or zero-padded and the %-80s name dominates).  The fix:
    # in fixed format, check cols 1-10 as an integer field.
    def _is_var_card(card: Card) -> bool:
        if block.fixed:
            # Fixed format: first 10 chars = the %10d ID field.
            field0 = card.raw[:10].strip()
            if not field0:
                # Blank first field: could be a variable card with nothing in
                # the first cell, or a blank ID.  Check if rest of the card
                # has NON-numeric text (variable names) vs mostly blanks.
                rest = card.raw[10:].strip()
                if not rest:
                    return False          # fully blank → end of variables
                try:
                    int(rest.split()[0])
                    return False          # next field is numeric → ID card
                except (ValueError, IndexError):
                    return True           # text follows → variable card
            try:
                int(field0)
                return False              # numeric first 10 chars → ID card
            except ValueError:
                return True               # text first 10 chars → variable card
        # Free format: original logic
        toks = card.tokens()
        if not toks:
            return False
        try:
            float(toks[0].replace("D", "E").replace("d", "e"))
            return False
        except ValueError:
            return True

    variables: List[str] = []
    n_var_cards = 0
    for c in cards:
        if not _is_var_card(c):
            break
        variables.extend(v.upper() for v in c.tokens())
        n_var_cards += 1
    if n_var_cards == 0:
        log.error(f"/TH/{kind}/{block.user_id}: variables card missing",
                  block.source)
        return
    if "DEF" in variables:
        _TH_DEFAULTS = {
            "NODE":   ["DX", "DY", "DZ", "VX", "VY", "VZ"],
            "PART":   ["IE", "KE"],
            "SECT":   ["FX", "FY", "FZ", "MX", "MY", "MZ"],
            "RBODY":  ["DX", "DY", "DZ", "VX", "VY", "VZ"],
            "SHEL":   ["SIGXX", "SIGYY", "SIGXY", "SIGYZ", "SIGZX"],
            "SH3N":   ["SIGXX", "SIGYY", "SIGXY", "SIGYZ", "SIGZX"],
            "SPRING": ["FX", "FY", "FZ", "DX", "DY", "DZ"],
            "BRIC":   ["SIGXX", "SIGYY", "SIGZZ", "SIGXY", "SIGYZ", "SIGZX"],
            "RWALL":  ["FN", "FT"],
            "INTER":  ["FN", "FT"],
            "MONVOL": ["P", "VOL", "MASS", "TEMP"],
        }
        defaults = _TH_DEFAULTS.get(kind, ["DEF"])
        rest = [v for v in variables if v != "DEF"]
        variables = defaults + [v for v in rest if v not in defaults]

    # -- id cards ------------------------------------------------------------
    # Element-level kinds (SHEL, SH3N, BRIC) use the same card layout as
    # NODE: CARD("%10d%10d%-80s", Elid, Skew_ID, Elname) — one per card.
    # Aggregate kinds (PART, SECT, RBODY, ...) pack plain %10d IDs.
    _ONE_PER_CARD = {"NODE", "SHEL", "SH3N", "BRIC", "SPRING"}
    ids: List[Union[int, str]] = []
    for c in cards[n_var_cards:]:
        if c.is_blank:
            continue
        if block.fixed:
            if kind in _ONE_PER_CARD:
                # %10d%10d%-80s — id column only (skew/name informative)
                f = c.cut("TH_NODE_ID")
                if f[0]:
                    try:
                        ids.append(int(f[0]))
                    except ValueError:
                        ids.append(f[0].strip())
            else:
                for s in c.cut("IDS10"):
                    if s:
                        try:
                            ids.append(int(s))
                        except ValueError:
                            ids.append(s.strip())
        else:
            if kind in _ONE_PER_CARD:
                toks = c.tokens()
                if toks:
                    try:
                        ids.append(int(toks[0]))
                    except ValueError:
                        ids.append(toks[0])
            else:
                for t in c.tokens():
                    try:
                        ids.append(int(t))
                    except ValueError:
                        ids.append(t)
    model.th_requests.append(THRequest(
        id=block.user_id, kind=kind, ids=ids, variables=variables,
        title=title))

    if kind in ("SUBS", "SUBDOMAIN", "SUBMODEL"):
        from ...model.entities import ThSubs
        prefix = block.key0.split("_")[0].upper() if "_" in block.key0 else (block.parts[0].upper() if block.parts else "TH")
        subs_ids_int: List[int] = []
        for x in ids:
            try:
                subs_ids_int.append(int(x))
            except (ValueError, TypeError):
                pass
        model.th_subs[block.user_id] = ThSubs(
            id=block.user_id, title=title, prefix=prefix,
            vars=variables, subs_ids=subs_ids_int
        )




def read_thpart_group(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/THPART/GR.../id`` (M201): Group-based Time History part output block."""
    from ...model.entities import ThPartGroup, THRequest
    elem_type = "SHEL"
    for cand in ("BEAM", "BRIC", "QUAD", "SH3N", "SHEL", "SPRI", "TRUS"):
        if cand in block.key0.upper() or any(cand in p.upper() for p in block.parts):
            elem_type = cand
            break
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    grelem_id = 0
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("THPART_GR_1")
            grelem_id = _ival(f[0]) if len(f) > 0 else 0
        else:
            toks = cards[0].tokens()
            grelem_id = int(float(toks[0])) if toks else 0
    thp = ThPartGroup(id=block.user_id, title=title, elem_type=elem_type, grelem_id=grelem_id)
    model.th_part_groups[block.user_id] = thp
    model.th_requests.append(THRequest(
        id=block.user_id, kind=f"GR{elem_type}", ids=[grelem_id], variables=["DEF"], title=title
    ))




def read_th_subs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TH/SUBS/id`` (M201): Substructure Time History output block."""
    read_th(block, model, log)




def read_th_title(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TH/TITLE`` (M115): Time history descriptive title card."""
    for c in block.cards:
        if not c.is_blank:
            model.th_titles.append(c.raw.strip())




def read_gauge(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GAUGE[/<subtype>]/gauge_ID`` (M104)::

        card 1:  title
        card 2:  node_ID  [gap]  elem_ID  dist
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else ""
    if subtype == "POINT":
        return read_gauge_point(block, model, log)
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GAUGE/{block.user_id}: missing data card", block.source)
        return

    node_id = 0
    elem_id = 0
    dist = 0.0
    fcut = 0.0

    if block.fixed:
        if subtype == "SPH":
            f = cards[0].cut("GAUGE_SPH_1")
            node_id = _ival(f[0]) if len(f) > 0 else 0
            fcut = _fval(f[1]) if len(f) > 1 and f[1].strip() else (_fval(f[2], 0.0) if len(f) > 2 else 0.0)
            elem_id = _ival(f[3]) if len(f) > 3 else 0
            dist = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            f = cards[0].cut("GAUGE_1")
            node_id = _ival(f[0]) if len(f) > 0 else 0
            elem_id = _ival(f[2]) if len(f) > 2 else 0
            dist = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        node_id = int(float(toks[0])) if len(toks) > 0 else 0
        if subtype == "SPH":
            fcut = float(toks[1]) if len(toks) > 1 else 0.0
            elem_id = int(float(toks[2])) if len(toks) > 2 else 0
            dist = float(toks[3]) if len(toks) > 3 else 0.0
        else:
            elem_id = int(float(toks[1])) if len(toks) > 1 else 0
            dist = float(toks[2]) if len(toks) > 2 else 0.0

    model.gauges[block.user_id] = Gauge(
        id=block.user_id, subtype=subtype, title=title, node_id=node_id,
        elem_id=elem_id, dist=dist, fcut=fcut,
    )
    if subtype == "SPH":
        from ...model.entities import GaugeSph
        model.gauge_sphs[block.user_id] = GaugeSph(
            id=block.user_id, title=title, node_id=node_id,
            fcut=fcut, shell_id=elem_id, dist=dist,
        )




def read_cluster(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CLUSTER[/<subtype>]/cluster_ID`` (M104)::

        card 1:  title
        card 2:  group_ID  skew_ID  ifail
        card 3:  fn_fail  sca_a1  sca_b1
        card 4:  fs_fail  sca_a2  sca_b2
        card 5:  mt_fail  sca_a3  sca_b3
        card 6:  mb_fail  sca_a4  sca_b4
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else ""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CLUSTER/{block.user_id}: missing data card", block.source)
        return

    group_id = 0
    skew_id = 0
    ifail = 0
    fn_fail, sca_a1, sca_b1 = 0.0, 1.0, 1.0
    fs_fail, sca_a2, sca_b2 = 0.0, 1.0, 1.0
    mt_fail, sca_a3, sca_b3 = 0.0, 1.0, 1.0
    mb_fail, sca_a4, sca_b4 = 0.0, 1.0, 1.0

    if block.fixed:
        f1 = cards[0].cut("CLUSTER_1")
        group_id = _ival(f1[0]) if len(f1) > 0 else 0
        skew_id = _ival(f1[1]) if len(f1) > 1 else 0
        ifail = _ival(f1[2]) if len(f1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CLUSTER_2")
            fn_fail = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            sca_a1 = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
            sca_b1 = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("CLUSTER_2")
            fs_fail = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            sca_a2 = _fval(f3[1], 1.0) if len(f3) > 1 else 1.0
            sca_b2 = _fval(f3[2], 1.0) if len(f3) > 2 else 1.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("CLUSTER_2")
            mt_fail = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            sca_a3 = _fval(f4[1], 1.0) if len(f4) > 1 else 1.0
            sca_b3 = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("CLUSTER_2")
            mb_fail = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            sca_a4 = _fval(f5[1], 1.0) if len(f5) > 1 else 1.0
            sca_b4 = _fval(f5[2], 1.0) if len(f5) > 2 else 1.0
    else:
        t1 = cards[0].tokens()
        group_id = int(float(t1[0])) if len(t1) > 0 else 0
        skew_id = int(float(t1[1])) if len(t1) > 1 else 0
        ifail = int(float(t1[2])) if len(t1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            fn_fail = float(t2[0]) if len(t2) > 0 else 0.0
            sca_a1 = float(t2[1]) if len(t2) > 1 else 1.0
            sca_b1 = float(t2[2]) if len(t2) > 2 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fs_fail = float(t3[0]) if len(t3) > 0 else 0.0
            sca_a2 = float(t3[1]) if len(t3) > 1 else 1.0
            sca_b2 = float(t3[2]) if len(t3) > 2 else 1.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            mt_fail = float(t4[0]) if len(t4) > 0 else 0.0
            sca_a3 = float(t4[1]) if len(t4) > 1 else 1.0
            sca_b3 = float(t4[2]) if len(t4) > 2 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            mb_fail = float(t5[0]) if len(t5) > 0 else 0.0
            sca_a4 = float(t5[1]) if len(t5) > 1 else 1.0
            sca_b4 = float(t5[2]) if len(t5) > 2 else 1.0

    model.clusters[block.user_id] = Cluster(
        id=block.user_id, subtype=subtype, title=title, group_id=group_id,
        skew_id=skew_id, ifail=ifail, fn_fail=fn_fail, sca_a1=sca_a1,
        sca_b1=sca_b1, fs_fail=fs_fail, sca_a2=sca_a2, sca_b2=sca_b2,
        mt_fail=mt_fail, sca_a3=sca_a3, sca_b3=sca_b3, mb_fail=mb_fail,
        sca_a4=sca_a4, sca_b4=sca_b4,
    )




def read_anim_dt(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ANIM/DT`` or ``/ENG/ANIM/DT`` (M210): Animation output frequency and sensor gating."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_ANIM_DT_1")
            model.anim_tstart = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            model.anim_dt = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            model.anim_sens_id = _ival(f[2], 0) if len(f) > 2 else 0
        else:
            toks = cards[0].tokens()
            model.anim_tstart = float(toks[0]) if len(toks) > 0 else 0.0
            model.anim_dt = float(toks[1]) if len(toks) > 1 else 0.0
            if len(toks) > 2:
                model.anim_sens_id = int(float(toks[2]))




def read_sensor_gap(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/TIME_GAP`` or ``/SENSOR/GAP`` (M216): Relative distance gap trigger sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/TIME_GAP/{block.user_id}: missing data card", block.source)
        return

    node1, node2, d_gap, t_delay, isens_mode = 0, 0, 0.0, 0.0, 0
    if block.fixed:
        f = cards[0].cut("SENSOR_GAP_1")
        node1 = _ival(f[0], 0) if len(f) > 0 else 0
        node2 = _ival(f[1], 0) if len(f) > 1 else 0
        d_gap = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        isens_mode = _ival(f[4], 0) if len(f) > 4 else 0
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0])) if len(toks) > 0 else 0
        node2 = int(float(toks[1])) if len(toks) > 1 else 0
        d_gap = float(toks[2]) if len(toks) > 2 else 0.0
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0
        isens_mode = int(float(toks[4])) if len(toks) > 4 else 0

    from ...model.entities import SensorGap, Sensor
    sg = SensorGap(
        id=block.user_id or 1, title=title, node1=node1,
        node2=node2, d_gap=d_gap, t_delay=t_delay, isens_mode=isens_mode
    )
    model.sensor_gaps[sg.id] = sg
    model.sensors.append(Sensor(
        id=sg.id, kind="GAP", tdelay=t_delay
    ))




def read_sensor_energy_ratio(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/ENERGY_RATIO`` or ``/SENSOR/ENG_RATIO`` (M217): Energy ratio limit trigger sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/ENERGY_RATIO/{block.user_id}: missing data card", block.source)
        return

    ratio_max, ratio_min, t_delay = 1e30, 0.0, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_ENERGY_RATIO_1")
        ratio_max = _fval(f[0], 1e30) if len(f) > 0 else 1e30
        ratio_min = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        ratio_max = float(toks[0]) if len(toks) > 0 else 1e30
        ratio_min = float(toks[1]) if len(toks) > 1 else 0.0
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorEnergyRatio, Sensor
    ser = SensorEnergyRatio(
        id=block.user_id or 1, title=title,
        ratio_max=ratio_max, ratio_min=ratio_min, t_delay=t_delay
    )
    model.sensor_energy_ratios[ser.id] = ser
    model.sensors.append(Sensor(
        id=ser.id, kind="ENERGY_RATIO", tdelay=t_delay
    ))




def read_sensor_cross_section(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/CROSSSECTION`` or ``/SENSOR/SEC_FORCE`` (M218): Cross-section force/moment trigger sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/CROSSSECTION/{block.user_id}: missing data card", block.source)
        return

    sec_id, f_cut, m_cut, t_delay = 0, 1e30, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_CROSSSECTION_1")
        sec_id = _ival(f[0], 0) if len(f) > 0 else 0
        f_cut = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        m_cut = _fval(f[2], 1e30) if len(f) > 2 else 1e30
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        sec_id = int(float(toks[0])) if len(toks) > 0 else 0
        f_cut = float(toks[1]) if len(toks) > 1 else 1e30
        m_cut = float(toks[2]) if len(toks) > 2 else 1e30
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import SensorCrossSection, Sensor
    scs = SensorCrossSection(
        id=block.user_id or 1, title=title, sec_id=sec_id,
        f_cut=f_cut, m_cut=m_cut, t_delay=t_delay
    )
    model.sensor_cross_sections[scs.id] = scs
    model.sensors.append(Sensor(
        id=scs.id, kind="CROSSSECTION", tdelay=t_delay
    ))




def read_sensor_rupture(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/RUPT`` or ``/SENSOR/SHELL_FAIL`` (M219): Element failure / erosion trigger sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/RUPT/{block.user_id}: missing data card", block.source)
        return

    elem_id, itype, t_delay = 0, 1, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_RUPT_1")
        elem_id = _ival(f[0], 0) if len(f) > 0 else 0
        itype = _ival(f[1], 1) if len(f) > 1 else 1
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        elem_id = int(float(toks[0])) if len(toks) > 0 else 0
        itype = int(float(toks[1])) if len(toks) > 1 else 1
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorRupture, Sensor
    sr = SensorRupture(
        id=block.user_id or 1, title=title, elem_id=elem_id,
        itype=itype, t_delay=t_delay
    )
    model.sensor_ruptures[sr.id] = sr
    model.sensors.append(Sensor(
        id=sr.id, kind="RUPT", tdelay=t_delay
    ))




def read_sensor_shear(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SHEAR`` or ``/SENSOR/SHEAR_STRESS`` (M220): Shear stress threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SHEAR/{block.user_id}: missing data card", block.source)
        return

    elem_id, tau_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SHEAR_1")
        elem_id = _ival(f[0], 0) if len(f) > 0 else 0
        tau_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        elem_id = int(float(toks[0])) if len(toks) > 0 else 0
        tau_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorShearStress, Sensor
    sss = SensorShearStress(
        id=block.user_id or 1, title=title, elem_id=elem_id,
        tau_max=tau_max, t_delay=t_delay
    )
    model.sensor_shears[sss.id] = sss
    model.sensors.append(Sensor(
        id=sss.id, kind="SHEAR_STRESS", tdelay=t_delay
    ))




def read_sensor_pressure(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/PRESSURE`` or ``/SENSOR/PRESS`` (M221): Pressure threshold trigger sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/PRESSURE/{block.user_id}: missing data card", block.source)
        return

    elem_id, p_min, p_max, t_delay = 0, -1e30, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_PRESSURE_1")
        elem_id = _ival(f[0], 0) if len(f) > 0 else 0
        p_min = _fval(f[1], -1e30) if len(f) > 1 else -1e30
        p_max = _fval(f[2], 1e30) if len(f) > 2 else 1e30
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        elem_id = int(float(toks[0])) if len(toks) > 0 else 0
        p_min = float(toks[1]) if len(toks) > 1 else -1e30
        p_max = float(toks[2]) if len(toks) > 2 else 1e30
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import SensorPressure, Sensor
    sp = SensorPressure(
        id=block.user_id or 1, title=title, elem_id=elem_id,
        p_min=p_min, p_max=p_max, t_delay=t_delay
    )
    model.sensor_pressures[sp.id] = sp
    model.sensors.append(Sensor(
        id=sp.id, kind="PRESSURE", tdelay=t_delay
    ))




def read_sensor_mass_ratio(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/MASS`` or ``/SENSOR/MASS_RATIO`` (M222): Added mass ratio threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/MASS/{block.user_id}: missing data card", block.source)
        return

    part_id, dmass_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_MASS_1")
        part_id = _ival(f[0], 0) if len(f) > 0 else 0
        dmass_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        part_id = int(float(toks[0])) if len(toks) > 0 else 0
        dmass_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorMassRatio, Sensor
    sm = SensorMassRatio(
        id=block.user_id or 1, title=title, part_id=part_id,
        dmass_max=dmass_max, t_delay=t_delay
    )
    model.sensor_mass_ratios[sm.id] = sm
    model.sensors.append(Sensor(
        id=sm.id, kind="MASS_RATIO", tdelay=t_delay
    ))




def read_sensor_energy_error(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/ENERGY_ERROR`` or ``/SENSOR/ENG_ERROR`` (M223): Total energy error percentage sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/ENERGY_ERROR/{block.user_id}: missing data card", block.source)
        return

    err_max, t_delay = 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_ENERGY_ERROR_1")
        err_max = _fval(f[0], 1e30) if len(f) > 0 else 1e30
        t_delay = _fval(f[1], 0.0) if len(f) > 1 else 0.0
    else:
        toks = cards[0].tokens()
        err_max = float(toks[0]) if len(toks) > 0 else 1e30
        t_delay = float(toks[1]) if len(toks) > 1 else 0.0

    from ...model.entities import SensorEnergyError, Sensor
    see = SensorEnergyError(
        id=block.user_id or 1, title=title, err_max=err_max, t_delay=t_delay
    )
    model.sensor_energy_errors[see.id] = see
    model.sensors.append(Sensor(
        id=see.id, kind="ENERGY_ERROR", tdelay=t_delay
    ))




def read_sensor_work_ratio(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/WORK_RATIO`` or ``/SENSOR/WRATIO`` (M224): Work ratio threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/WORK_RATIO/{block.user_id}: missing data card", block.source)
        return

    w_ratio_max, t_delay = 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_WORK_RATIO_1")
        w_ratio_max = _fval(f[0], 1e30) if len(f) > 0 else 1e30
        t_delay = _fval(f[1], 0.0) if len(f) > 1 else 0.0
    else:
        toks = cards[0].tokens()
        w_ratio_max = float(toks[0]) if len(toks) > 0 else 1e30
        t_delay = float(toks[1]) if len(toks) > 1 else 0.0

    from ...model.entities import SensorWorkRatio, Sensor
    swr = SensorWorkRatio(
        id=block.user_id or 1, title=title, w_ratio_max=w_ratio_max, t_delay=t_delay
    )
    model.sensor_work_ratios[swr.id] = swr
    model.sensors.append(Sensor(
        id=swr.id, kind="WORK_RATIO", tdelay=t_delay
    ))




def read_sensor_spring(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING`` or ``/SENSOR/SPRING_FORCE`` (M225): Spring force/moment threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING/{block.user_id}: missing data card", block.source)
        return

    spring_id, f_max, m_max, t_delay = 0, 1e30, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        f_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        m_max = _fval(f[2], 1e30) if len(f) > 2 else 1e30
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        f_max = float(toks[1]) if len(toks) > 1 else 1e30
        m_max = float(toks[2]) if len(toks) > 2 else 1e30
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import SensorSpring, Sensor
    ss = SensorSpring(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        f_max=f_max, m_max=m_max, t_delay=t_delay
    )
    model.sensor_springs[ss.id] = ss
    model.sensors.append(Sensor(
        id=ss.id, kind="SPRING", tdelay=t_delay
    ))




def read_sensor_shell_strain(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SHELL_STRAIN`` or ``/SENSOR/STRAIN_SHELL`` (M226): Shell element strain threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SHELL_STRAIN/{block.user_id}: missing data card", block.source)
        return

    shell_id, eps_max, ip, t_delay = 0, 1e30, 1, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SHELL_STRAIN_1")
        shell_id = _ival(f[0], 0) if len(f) > 0 else 0
        eps_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        ip = _ival(f[2], 1) if len(f) > 2 else 1
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        shell_id = int(float(toks[0])) if len(toks) > 0 else 0
        eps_max = float(toks[1]) if len(toks) > 1 else 1e30
        ip = int(float(toks[2])) if len(toks) > 2 else 1
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import SensorShellStrain, Sensor
    sss = SensorShellStrain(
        id=block.user_id or 1, title=title, shell_id=shell_id,
        eps_max=eps_max, ip=ip, t_delay=t_delay
    )
    model.sensor_shell_strains[sss.id] = sss
    model.sensors.append(Sensor(
        id=sss.id, kind="SHELL_STRAIN", tdelay=t_delay
    ))




def read_sensor_solid_strain(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SOLID_STRAIN`` or ``/SENSOR/STRAIN_SOLID`` (M227): Solid element strain threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SOLID_STRAIN/{block.user_id}: missing data card", block.source)
        return

    solid_id, eps_max, ip, t_delay = 0, 1e30, 1, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SOLID_STRAIN_1")
        solid_id = _ival(f[0], 0) if len(f) > 0 else 0
        eps_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        ip = _ival(f[2], 1) if len(f) > 2 else 1
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        solid_id = int(float(toks[0])) if len(toks) > 0 else 0
        eps_max = float(toks[1]) if len(toks) > 1 else 1e30
        ip = int(float(toks[2])) if len(toks) > 2 else 1
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import SensorSolidStrain, Sensor
    sss = SensorSolidStrain(
        id=block.user_id or 1, title=title, solid_id=solid_id,
        eps_max=eps_max, ip=ip, t_delay=t_delay
    )
    model.sensor_solid_strains[sss.id] = sss
    model.sensors.append(Sensor(
        id=sss.id, kind="SOLID_STRAIN", tdelay=t_delay
    ))




def read_sensor_beam_strain(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/BEAM_STRAIN`` or ``/SENSOR/STRAIN_BEAM`` (M228): Beam element strain threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/BEAM_STRAIN/{block.user_id}: missing data card", block.source)
        return

    beam_id, eps_max, ip, t_delay = 0, 1e30, 1, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_BEAM_STRAIN_1")
        beam_id = _ival(f[0], 0) if len(f) > 0 else 0
        eps_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        ip = _ival(f[2], 1) if len(f) > 2 else 1
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        beam_id = int(float(toks[0])) if len(toks) > 0 else 0
        eps_max = float(toks[1]) if len(toks) > 1 else 1e30
        ip = int(float(toks[2])) if len(toks) > 2 else 1
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import SensorBeamStrain, Sensor
    sbs = SensorBeamStrain(
        id=block.user_id or 1, title=title, beam_id=beam_id,
        eps_max=eps_max, ip=ip, t_delay=t_delay
    )
    model.sensor_beam_strains[sbs.id] = sbs
    model.sensors.append(Sensor(
        id=sbs.id, kind="BEAM_STRAIN", tdelay=t_delay
    ))




def read_sensor_truss_strain(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/TRUSS_STRAIN`` or ``/SENSOR/STRAIN_TRUSS`` (M229): Truss element strain threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/TRUSS_STRAIN/{block.user_id}: missing data card", block.source)
        return

    truss_id, eps_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_TRUSS_STRAIN_1")
        truss_id = _ival(f[0], 0) if len(f) > 0 else 0
        eps_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        truss_id = int(float(toks[0])) if len(toks) > 0 else 0
        eps_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorTrussStrain, Sensor
    sts = SensorTrussStrain(
        id=block.user_id or 1, title=title, truss_id=truss_id,
        eps_max=eps_max, t_delay=t_delay
    )
    model.sensor_truss_strains[sts.id] = sts
    model.sensors.append(Sensor(
        id=sts.id, kind="TRUSS_STRAIN", tdelay=t_delay
    ))




def read_sensor_shell_force(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SHELL_FORCE`` or ``/SENSOR/FORCE_SHELL`` (M230): Shell element force/moment threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SHELL_FORCE/{block.user_id}: missing data card", block.source)
        return

    shell_id, f_max, m_max, t_delay = 0, 1e30, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SHELL_FORCE_1")
        shell_id = _ival(f[0], 0) if len(f) > 0 else 0
        f_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        m_max = _fval(f[2], 1e30) if len(f) > 2 else 1e30
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        shell_id = int(float(toks[0])) if len(toks) > 0 else 0
        f_max = float(toks[1]) if len(toks) > 1 else 1e30
        m_max = float(toks[2]) if len(toks) > 2 else 1e30
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import SensorShellForce, Sensor
    ssf = SensorShellForce(
        id=block.user_id or 1, title=title, shell_id=shell_id,
        f_max=f_max, m_max=m_max, t_delay=t_delay
    )
    model.sensor_shell_forces[ssf.id] = ssf
    model.sensors.append(Sensor(
        id=ssf.id, kind="SHELL_FORCE", tdelay=t_delay
    ))




def read_sensor_solid_force(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SOLID_FORCE`` or ``/SENSOR/FORCE_SOLID`` (M231): Solid element force threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SOLID_FORCE/{block.user_id}: missing data card", block.source)
        return

    solid_id, f_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SOLID_FORCE_1")
        solid_id = _ival(f[0], 0) if len(f) > 0 else 0
        f_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        solid_id = int(float(toks[0])) if len(toks) > 0 else 0
        f_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSolidForce, Sensor
    ssf = SensorSolidForce(
        id=block.user_id or 1, title=title, solid_id=solid_id,
        f_max=f_max, t_delay=t_delay
    )
    model.sensor_solid_forces[ssf.id] = ssf
    model.sensors.append(Sensor(
        id=ssf.id, kind="SOLID_FORCE", tdelay=t_delay
    ))




def read_sensor_beam_force(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/BEAM_FORCE`` or ``/SENSOR/FORCE_BEAM`` (M232): Beam element force/moment threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/BEAM_FORCE/{block.user_id}: missing data card", block.source)
        return

    beam_id, f_max, m_max, t_delay = 0, 1e30, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_BEAM_FORCE_1")
        beam_id = _ival(f[0], 0) if len(f) > 0 else 0
        f_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        m_max = _fval(f[2], 1e30) if len(f) > 2 else 1e30
        t_delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        beam_id = int(float(toks[0])) if len(toks) > 0 else 0
        f_max = float(toks[1]) if len(toks) > 1 else 1e30
        m_max = float(toks[2]) if len(toks) > 2 else 1e30
        t_delay = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import SensorBeamForce, Sensor
    sbf = SensorBeamForce(
        id=block.user_id or 1, title=title, beam_id=beam_id,
        f_max=f_max, m_max=m_max, t_delay=t_delay
    )
    model.sensor_beam_forces[sbf.id] = sbf
    model.sensors.append(Sensor(
        id=sbf.id, kind="BEAM_FORCE", tdelay=t_delay
    ))




def read_sensor_truss_force(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/TRUSS_FORCE`` or ``/SENSOR/FORCE_TRUSS`` (M233): Truss element force threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/TRUSS_FORCE/{block.user_id}: missing data card", block.source)
        return

    truss_id, f_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_TRUSS_FORCE_1")
        truss_id = _ival(f[0], 0) if len(f) > 0 else 0
        f_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        truss_id = int(float(toks[0])) if len(toks) > 0 else 0
        f_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorTrussForce, Sensor
    stf = SensorTrussForce(
        id=block.user_id or 1, title=title, truss_id=truss_id,
        f_max=f_max, t_delay=t_delay
    )
    model.sensor_truss_forces[stf.id] = stf
    model.sensors.append(Sensor(
        id=stf.id, kind="TRUSS_FORCE", tdelay=t_delay
    ))




def read_sensor_spring_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_ENERGY`` or ``/SENSOR/ENERGY_SPRING`` (M234): Spring element internal energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, e_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        e_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        e_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringEnergy, Sensor
    sse = SensorSpringEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        e_max=e_max, t_delay=t_delay
    )
    model.sensor_spring_energies[sse.id] = sse
    model.sensors.append(Sensor(
        id=sse.id, kind="SPRING_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_defl(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_DEFL`` or ``/SENSOR/DEF_SPRING`` (M235): Spring element deflection threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_DEFL/{block.user_id}: missing data card", block.source)
        return

    spring_id, defl_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_DEFL_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        defl_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        defl_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringDefl, Sensor
    ssd = SensorSpringDefl(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        defl_max=defl_max, t_delay=t_delay
    )
    model.sensor_spring_defls[ssd.id] = ssd
    model.sensors.append(Sensor(
        id=ssd.id, kind="SPRING_DEFL", tdelay=t_delay
    ))




def read_sensor_spring_rot(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_ROT`` or ``/SENSOR/ROT_SPRING`` (M236): Spring element rotation/twist threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_ROT/{block.user_id}: missing data card", block.source)
        return

    spring_id, rot_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_ROT_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        rot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        rot_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringRot, Sensor
    ssr = SensorSpringRot(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        rot_max=rot_max, t_delay=t_delay
    )
    model.sensor_spring_rots[ssr.id] = ssr
    model.sensors.append(Sensor(
        id=ssr.id, kind="SPRING_ROT", tdelay=t_delay
    ))




def read_sensor_spring_rotv(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_ROTV`` or ``/SENSOR/ROTV_SPRING`` (M237): Spring element rotational velocity threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_ROTV/{block.user_id}: missing data card", block.source)
        return

    spring_id, rotv_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_ROTV_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        rotv_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        rotv_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringRotv, Sensor
    ssrv = SensorSpringRotv(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        rotv_max=rotv_max, t_delay=t_delay
    )
    model.sensor_spring_rotvs[ssrv.id] = ssrv
    model.sensors.append(Sensor(
        id=ssrv.id, kind="SPRING_ROTV", tdelay=t_delay
    ))




def read_sensor_spring_rota(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_ROTA`` or ``/SENSOR/ROTA_SPRING`` (M238): Spring element rotational acceleration threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_ROTA/{block.user_id}: missing data card", block.source)
        return

    spring_id, rota_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_ROTA_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        rota_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        rota_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringRota, Sensor
    ssra = SensorSpringRota(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        rota_max=rota_max, t_delay=t_delay
    )
    model.sensor_spring_rotas[ssra.id] = ssra
    model.sensors.append(Sensor(
        id=ssra.id, kind="SPRING_ROTA", tdelay=t_delay
    ))




def read_sensor_spring_axial(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_AXIAL`` or ``/SENSOR/AXIAL_SPRING`` (M239): Spring element axial force threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_AXIAL/{block.user_id}: missing data card", block.source)
        return

    spring_id, fax_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_AXIAL_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        fax_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        fax_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringAxial, Sensor
    ssa = SensorSpringAxial(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        fax_max=fax_max, t_delay=t_delay
    )
    model.sensor_spring_axials[ssa.id] = ssa
    model.sensors.append(Sensor(
        id=ssa.id, kind="SPRING_AXIAL", tdelay=t_delay
    ))




def read_sensor_spring_shear(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_SHEAR`` or ``/SENSOR/SHEAR_SPRING`` (M240): Spring element shear force threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_SHEAR/{block.user_id}: missing data card", block.source)
        return

    spring_id, fsh_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_SHEAR_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        fsh_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        fsh_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringShear, Sensor
    sss = SensorSpringShear(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        fsh_max=fsh_max, t_delay=t_delay
    )
    model.sensor_spring_shears[sss.id] = sss
    model.sensors.append(Sensor(
        id=sss.id, kind="SPRING_SHEAR", tdelay=t_delay
    ))




def read_sensor_spring_bend(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BEND`` or ``/SENSOR/BEND_SPRING`` (M241): Spring element bending moment threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BEND/{block.user_id}: missing data card", block.source)
        return

    spring_id, mbend_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_BEND_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        mbend_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        mbend_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBend, Sensor
    ssb = SensorSpringBend(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        mbend_max=mbend_max, t_delay=t_delay
    )
    model.sensor_spring_bends[ssb.id] = ssb
    model.sensors.append(Sensor(
        id=ssb.id, kind="SPRING_BEND", tdelay=t_delay
    ))




def read_sensor_spring_torsion(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSION`` or ``/SENSOR/TORSION_SPRING`` (M242): Spring element torsional moment threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSION/{block.user_id}: missing data card", block.source)
        return

    spring_id, mtor_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_TORSION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        mtor_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        mtor_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsion, Sensor
    sst = SensorSpringTorsion(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        mtor_max=mtor_max, t_delay=t_delay
    )
    model.sensor_spring_torsions[sst.id] = sst
    model.sensors.append(Sensor(
        id=sst.id, kind="SPRING_TORSION", tdelay=t_delay
    ))




def read_sensor_spring_strain_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_STRAIN_ENERGY`` or ``/SENSOR/STRAIN_ENERGY_SPRING`` (M243): Spring element strain energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_STRAIN_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, estrain_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_STRAIN_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        estrain_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        estrain_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringStrainEnergy, SensorSpringTotalStrainEnergy, Sensor
    ssse = SensorSpringStrainEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        estrain_max=estrain_max, t_delay=t_delay
    )
    model.sensor_spring_strain_energies[ssse.id] = ssse
    sstse = SensorSpringTotalStrainEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_total_max=estrain_max, t_delay=t_delay
    )
    model.sensor_spring_total_strain_energies[sstse.id] = sstse
    model.sensors.append(Sensor(
        id=ssse.id, kind="SPRING_STRAIN_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_kinetic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_KINETIC_ENERGY`` or ``/SENSOR/KINETIC_ENERGY_SPRING`` (M244): Spring element kinetic energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_KINETIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, ekin_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_KINETIC_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        ekin_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        ekin_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringKineticEnergy, Sensor
    sske = SensorSpringKineticEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        ekin_max=ekin_max, t_delay=t_delay
    )
    model.sensor_spring_kinetic_energies[sske.id] = sske
    model.sensors.append(Sensor(
        id=sske.id, kind="SPRING_KINETIC_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_hourglass_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_HOURGLASS_ENERGY`` or ``/SENSOR/HOURGLASS_ENERGY_SPRING`` (M245): Spring element hourglass energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_HOURGLASS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, ehe_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_HOURGLASS_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        ehe_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        ehe_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringHourglassEnergy, Sensor
    sshe = SensorSpringHourglassEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        ehe_max=ehe_max, t_delay=t_delay
    )
    model.sensor_spring_hourglass_energies[sshe.id] = sshe
    model.sensors.append(Sensor(
        id=sshe.id, kind="SPRING_HOURGLASS_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_contact_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_CONTACT_ENERGY`` or ``/SENSOR/CONTACT_ENERGY_SPRING`` (M246): Spring element contact energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_CONTACT_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, ece_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_CONTACT_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        ece_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        ece_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringContactEnergy, Sensor
    ssce = SensorSpringContactEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        ece_max=ece_max, t_delay=t_delay
    )
    model.sensor_spring_contact_energies[ssce.id] = ssce
    model.sensors.append(Sensor(
        id=ssce.id, kind="SPRING_CONTACT_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_numerical_dissipation(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NUMERICAL_DISSIPATION`` or ``/SENSOR/SPRING_NUM_DISS`` (M247): Spring element numerical dissipation energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NUMERICAL_DISSIPATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, enum_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_NUMERICAL_DISSIPATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        enum_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        enum_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNumericalDissipation, Sensor
    ssnd = SensorSpringNumericalDissipation(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        enum_max=enum_max, t_delay=t_delay
    )
    model.sensor_spring_numerical_dissipations[ssnd.id] = ssnd
    model.sensors.append(Sensor(
        id=ssnd.id, kind="SPRING_NUMERICAL_DISSIPATION", tdelay=t_delay
    ))




def read_sensor_spring_ext_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_EXT_WORK`` or ``/SENSOR/SPRING_EXTERNAL_WORK`` (M248): Spring element external work threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_EXT_WORK/{block.user_id}: missing data card", block.source)
        return

    spring_id, ewext_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_EXT_WORK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        ewext_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        ewext_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringExtWork, Sensor
    ssew = SensorSpringExtWork(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        ewext_max=ewext_max, t_delay=t_delay
    )
    model.sensor_spring_ext_works[ssew.id] = ssew
    model.sensors.append(Sensor(
        id=ssew.id, kind="SPRING_EXT_WORK", tdelay=t_delay
    ))




def read_sensor_spring_tot_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOT_ENERGY`` or ``/SENSOR/SPRING_TOTAL_ENERGY`` (M249): Spring element total energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOT_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, etot_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_TOT_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        etot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        etot_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotEnergy, Sensor
    sste = SensorSpringTotEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        etot_max=etot_max, t_delay=t_delay
    )
    model.sensor_spring_tot_energies[sste.id] = sste
    model.sensors.append(Sensor(
        id=sste.id, kind="SPRING_TOT_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_mass_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_MASS_ENERGY`` or ``/SENSOR/SPRING_MASS_ENER`` (M250): Spring element added mass energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_MASS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, emass_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_MASS_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        emass_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        emass_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringMassEnergy, Sensor
    ssme = SensorSpringMassEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        emass_max=emass_max, t_delay=t_delay
    )
    model.sensor_spring_mass_energies[ssme.id] = ssme
    model.sensors.append(Sensor(
        id=ssme.id, kind="SPRING_MASS_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_mass_change(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_MASS_CHANGE`` or ``/SENSOR/SPRING_DMASS`` (M251): Spring element mass variation threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_MASS_CHANGE/{block.user_id}: missing data card", block.source)
        return

    spring_id, dmass_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_MASS_CHANGE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        dmass_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        dmass_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringMassChange, Sensor
    ssmc = SensorSpringMassChange(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        dmass_max=dmass_max, t_delay=t_delay
    )
    model.sensor_spring_mass_changes[ssmc.id] = ssmc
    model.sensors.append(Sensor(
        id=ssmc.id, kind="SPRING_MASS_CHANGE", tdelay=t_delay
    ))




def read_sensor_spring_pressure(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_PRESSURE`` or ``/SENSOR/SPRING_PRESS`` (M252): Spring element hydrostatic pressure threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_PRESSURE/{block.user_id}: missing data card", block.source)
        return

    spring_id, press_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_PRESSURE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        press_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        press_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringPressure, Sensor
    ssp = SensorSpringPressure(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        press_max=press_max, t_delay=t_delay
    )
    model.sensor_spring_pressures[ssp.id] = ssp
    model.sensors.append(Sensor(
        id=ssp.id, kind="SPRING_PRESSURE", tdelay=t_delay
    ))




def read_sensor_spring_temperature(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TEMPERATURE`` or ``/SENSOR/SPRING_TEMP`` (M253): Spring element temperature threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TEMPERATURE/{block.user_id}: missing data card", block.source)
        return

    spring_id, temp_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_TEMPERATURE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        temp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        temp_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTemperature, Sensor
    sst = SensorSpringTemperature(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        temp_max=temp_max, t_delay=t_delay
    )
    model.sensor_spring_temperatures[sst.id] = sst
    model.sensors.append(Sensor(
        id=sst.id, kind="SPRING_TEMPERATURE", tdelay=t_delay
    ))




def read_sensor_spring_volume(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_VOLUME`` or ``/SENSOR/SPRING_VOL`` (M254): Spring element volume threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_VOLUME/{block.user_id}: missing data card", block.source)
        return

    spring_id, vol_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_VOLUME_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        vol_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        vol_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringVolume, Sensor
    ssv = SensorSpringVolume(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        vol_max=vol_max, t_delay=t_delay
    )
    model.sensor_spring_volumes[ssv.id] = ssv
    model.sensors.append(Sensor(
        id=ssv.id, kind="SPRING_VOLUME", tdelay=t_delay
    ))




def read_sensor_spring_density(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_DENSITY`` or ``/SENSOR/SPRING_DENS`` (M255): Spring element material density threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_DENSITY/{block.user_id}: missing data card", block.source)
        return

    spring_id, dens_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_DENSITY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        dens_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        dens_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringDensity, Sensor
    ssd = SensorSpringDensity(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        dens_max=dens_max, t_delay=t_delay
    )
    model.sensor_spring_densities[ssd.id] = ssd
    model.sensors.append(Sensor(
        id=ssd.id, kind="SPRING_DENSITY", tdelay=t_delay
    ))




def read_sensor_spring_entropy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_ENTROPY`` or ``/SENSOR/SPRING_ENTR`` (M256): Spring element thermal entropy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_ENTROPY/{block.user_id}: missing data card", block.source)
        return

    spring_id, entr_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_ENTROPY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        entr_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        entr_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringEntropy, Sensor
    sse = SensorSpringEntropy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        entr_max=entr_max, t_delay=t_delay
    )
    model.sensor_spring_entropies[sse.id] = sse
    model.sensors.append(Sensor(
        id=sse.id, kind="SPRING_ENTROPY", tdelay=t_delay
    ))




def read_sensor_spring_sound_speed(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_SOUND_SPEED`` or ``/SENSOR/SPRING_SS`` (M257): Spring element acoustic sound speed threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_SOUND_SPEED/{block.user_id}: missing data card", block.source)
        return

    spring_id, sound_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_SOUND_SPEED_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        sound_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        sound_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringSoundSpeed, Sensor
    ssss = SensorSpringSoundSpeed(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        sound_max=sound_max, t_delay=t_delay
    )
    model.sensor_spring_sound_speeds[ssss.id] = ssss
    model.sensors.append(Sensor(
        id=ssss.id, kind="SPRING_SOUND_SPEED", tdelay=t_delay
    ))




def read_sensor_spring_yield_stress(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_YIELD_STRESS`` or ``/SENSOR/SPRING_YIELD`` (M259): Spring element yield stress threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_YIELD_STRESS/{block.user_id}: missing data card", block.source)
        return

    spring_id, sigy_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_YIELD_STRESS_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        sigy_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        sigy_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringYieldStress, Sensor
    ssys = SensorSpringYieldStress(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        sigy_max=sigy_max, t_delay=t_delay
    )
    model.sensor_spring_yield_stresses[ssys.id] = ssys
    model.sensors.append(Sensor(
        id=ssys.id, kind="SPRING_YIELD_STRESS", tdelay=t_delay
    ))




def read_sensor_spring_plastic_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_PLASTIC_WORK`` or ``/SENSOR/SPRING_WPLAS`` (M260): Spring element plastic work threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_PLASTIC_WORK/{block.user_id}: missing data card", block.source)
        return

    spring_id, wplas_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_PLASTIC_WORK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        wplas_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        wplas_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringPlasticWork, Sensor
    sspw = SensorSpringPlasticWork(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        wplas_max=wplas_max, t_delay=t_delay
    )
    model.sensor_spring_plastic_works[sspw.id] = sspw
    model.sensors.append(Sensor(
        id=sspw.id, kind="SPRING_PLASTIC_WORK", tdelay=t_delay
    ))




def read_sensor_spring_force_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_FORCE_RATE`` or ``/SENSOR/SPRING_DF`` (M261): Spring element force time-rate threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_FORCE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, df_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_FORCE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        df_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        df_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringForceRate, Sensor
    ssfr = SensorSpringForceRate(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        df_max=df_max, t_delay=t_delay
    )
    model.sensor_spring_force_rates[ssfr.id] = ssfr
    model.sensors.append(Sensor(
        id=ssfr.id, kind="SPRING_FORCE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_force_impulse(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_FORCE_IMPULSE`` or ``/SENSOR/SPRING_IMPULSE`` (M262): Spring element linear force impulse threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_FORCE_IMPULSE/{block.user_id}: missing data card", block.source)
        return

    spring_id, j_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_FORCE_IMPULSE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        j_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        j_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringForceImpulse, Sensor
    ssfi = SensorSpringForceImpulse(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        j_max=j_max, t_delay=t_delay
    )
    model.sensor_spring_force_impulses[ssfi.id] = ssfi
    model.sensors.append(Sensor(
        id=ssfi.id, kind="SPRING_FORCE_IMPULSE", tdelay=t_delay
    ))




def read_sensor_spring_moment_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_MOMENT_RATE`` or ``/SENSOR/SPRING_DM`` (M263): Spring element moment / torque time-rate threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_MOMENT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, dm_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_MOMENT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        dm_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        dm_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringMomentRate, Sensor
    ssmr = SensorSpringMomentRate(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        dm_max=dm_max, t_delay=t_delay
    )
    model.sensor_spring_moment_rates[ssmr.id] = ssmr
    model.sensors.append(Sensor(
        id=ssmr.id, kind="SPRING_MOMENT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_moment_impulse(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_MOMENT_IMPULSE`` or ``/SENSOR/SPRING_MOM_IMPULSE`` (M264): Spring element angular/moment impulse threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_MOMENT_IMPULSE/{block.user_id}: missing data card", block.source)
        return

    spring_id, h_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_MOMENT_IMPULSE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        h_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        h_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringMomentImpulse, Sensor
    ssmi = SensorSpringMomentImpulse(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        h_max=h_max, t_delay=t_delay
    )
    model.sensor_spring_moment_impulses[ssmi.id] = ssmi
    model.sensors.append(Sensor(
        id=ssmi.id, kind="SPRING_MOMENT_IMPULSE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_ENERGY`` or ``/SENSOR/SPRING_TOR_ENERGY`` (M265): Spring element torsional elastic deformation energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, e_tor_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        e_tor_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        e_tor_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalEnergy, Sensor
    sste = SensorSpringTorsionalEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        e_tor_max=e_tor_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_energies[sste.id] = sste
    model.sensors.append(Sensor(
        id=sste.id, kind="SPRING_TORSIONAL_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_bending_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_ENERGY`` or ``/SENSOR/SPRING_BEND_ENERGY`` (M266): Spring element bending/flexural elastic deformation energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, e_bend_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_BENDING_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        e_bend_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        e_bend_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingEnergy, Sensor
    ssbe = SensorSpringBendingEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        e_bend_max=e_bend_max, t_delay=t_delay
    )
    model.sensor_spring_bending_energies[ssbe.id] = ssbe
    model.sensors.append(Sensor(
        id=ssbe.id, kind="SPRING_BENDING_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_total_strain_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_STRAIN_ENERGY`` or ``/SENSOR/SPRING_STRAIN_ENERGY`` (M267): Spring element total elastic strain energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_STRAIN_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_total_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_STRAIN_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_total_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_total_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalStrainEnergy, SensorSpringStrainEnergy, Sensor
    sstse = SensorSpringTotalStrainEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_total_max=u_total_max, t_delay=t_delay
    )
    model.sensor_spring_total_strain_energies[sstse.id] = sstse
    ssse = SensorSpringStrainEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        estrain_max=u_total_max, t_delay=t_delay
    )
    model.sensor_spring_strain_energies[ssse.id] = ssse
    model.sensors.append(Sensor(
        id=sstse.id, kind="SPRING_TOTAL_STRAIN_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_volumetric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_VOLUMETRIC_ENERGY`` or ``/SENSOR/SPRING_VOL_ENERGY`` (M268): Spring element volumetric/hydrostatic elastic deformation energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_VOLUMETRIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_vol_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_VOLUMETRIC_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_vol_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_vol_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringVolumetricEnergy, Sensor
    ssve = SensorSpringVolumetricEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_vol_max=u_vol_max, t_delay=t_delay
    )
    model.sensor_spring_volumetric_energies[ssve.id] = ssve
    model.sensors.append(Sensor(
        id=ssve.id, kind="SPRING_VOLUMETRIC_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_shear_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_SHEAR_ENERGY`` or ``/SENSOR/SPRING_SH_ENERGY`` (M269): Spring element shear elastic deformation energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_SHEAR_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_shear_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_SHEAR_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_shear_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_shear_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringShearEnergy, Sensor
    ssse = SensorSpringShearEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_shear_max=u_shear_max, t_delay=t_delay
    )
    model.sensor_spring_shear_energies[ssse.id] = ssse
    model.sensors.append(Sensor(
        id=ssse.id, kind="SPRING_SHEAR_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_axial_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_AXIAL_ENERGY`` or ``/SENSOR/SPRING_AX_ENERGY`` (M270): Spring element axial elastic deformation energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_AXIAL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_axial_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_AXIAL_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_axial_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_axial_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringAxialEnergy, Sensor
    ssae = SensorSpringAxialEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_axial_max=u_axial_max, t_delay=t_delay
    )
    model.sensor_spring_axial_energies[ssae.id] = ssae
    model.sensors.append(Sensor(
        id=ssae.id, kind="SPRING_AXIAL_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_damping_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_DAMPING_ENERGY`` or ``/SENSOR/SPRING_DAMP_ENERGY`` (M271): Spring element damping dissipation energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_DAMPING_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_damp_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_DAMPING_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_damp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_damp_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringDampingEnergy, Sensor
    ssde = SensorSpringDampingEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_damp_max=u_damp_max, t_delay=t_delay
    )
    model.sensor_spring_damping_energies[ssde.id] = ssde
    model.sensors.append(Sensor(
        id=ssde.id, kind="SPRING_DAMPING_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_coupling_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_COUPLING_ENERGY`` or ``/SENSOR/SPRING_COUP_ENERGY`` (M272): Spring element coupling energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_COUPLING_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_coup_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_COUPLING_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_coup_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_coup_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringCouplingEnergy, Sensor
    ssce = SensorSpringCouplingEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_coup_max=u_coup_max, t_delay=t_delay
    )
    model.sensor_spring_coupling_energies[ssce.id] = ssce
    model.sensors.append(Sensor(
        id=ssce.id, kind="SPRING_COUPLING_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_torsional_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_ENERGY`` or ``/SENSOR/SPRING_TORS_ENERGY`` (M273): Spring element torsional energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_tors_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_tors_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_tors_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalEnergy, Sensor
    sste = SensorSpringTorsionalEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_tors_max=u_tors_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_energies[sste.id] = sste
    model.sensors.append(Sensor(
        id=sste.id, kind="SPRING_TORSIONAL_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_bending_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_ENERGY`` or ``/SENSOR/SPRING_BEND_ENERGY`` (M274): Spring element bending energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_bend_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_BENDING_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_bend_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_bend_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingEnergy, Sensor
    ssbe = SensorSpringBendingEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_bend_max=u_bend_max, t_delay=t_delay
    )
    model.sensor_spring_bending_energies[ssbe.id] = ssbe
    model.sensors.append(Sensor(
        id=ssbe.id, kind="SPRING_BENDING_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_pinching_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_PINCHING_ENERGY`` or ``/SENSOR/SPRING_PINCH_ENERGY`` (M275): Spring element transverse pinching / squeeze energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_PINCHING_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_pinch_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_PINCHING_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_pinch_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_pinch_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringPinchingEnergy, Sensor
    sspe = SensorSpringPinchingEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_pinch_max=u_pinch_max, t_delay=t_delay
    )
    model.sensor_spring_pinching_energies[sspe.id] = sspe
    model.sensors.append(Sensor(
        id=sspe.id, kind="SPRING_PINCHING_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_friction_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_FRICTION_ENERGY`` or ``/SENSOR/SPRING_FRICT_ENERGY`` (M276): Spring element frictional sliding dissipation energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_FRICTION_ENERGY/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_frict_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_FRICTION_ENERGY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_frict_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_frict_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringFrictionEnergy, Sensor
    ssfe = SensorSpringFrictionEnergy(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_frict_max=u_frict_max, t_delay=t_delay
    )
    model.sensor_spring_friction_energies[ssfe.id] = ssfe
    model.sensors.append(Sensor(
        id=ssfe.id, kind="SPRING_FRICTION_ENERGY", tdelay=t_delay
    ))




def read_sensor_spring_thermal_dissipation(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_THERMAL_DISSIPATION`` or ``/SENSOR/SPRING_THERM_DISS`` (M277): Spring element thermal dissipation / heat energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_THERMAL_DISSIPATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, u_therm_max, t_delay = 0, 1e30, 0.0
    if block.fixed:
        f = cards[0].cut("SENSOR_SPRING_THERMAL_DISSIPATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        u_therm_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0])) if len(toks) > 0 else 0
        u_therm_max = float(toks[1]) if len(toks) > 1 else 1e30
        t_delay = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringThermalDissipation, Sensor
    sstd = SensorSpringThermalDissipation(
        id=block.user_id or 1, title=title, spring_id=spring_id,
        u_therm_max=u_therm_max, t_delay=t_delay
    )
    model.sensor_spring_thermal_dissipations[sstd.id] = sstd
    model.sensors.append(Sensor(
        id=sstd.id, kind="SPRING_THERMAL_DISSIPATION", tdelay=t_delay
    ))




def read_sensor_spring_total_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_WORK`` or ``/SENSOR/SPRING_TOT_WORK`` (M278): Spring element total work energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_WORK/{block.user_id}: missing data card", block.source)
        return

    spring_id, w_tot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_WORK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        w_tot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        w_tot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalWork, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_works) + 1)
    sstw = SensorSpringTotalWork(
        id=s_id, title=title, spring_id=spring_id,
        w_tot_max=w_tot_max, t_delay=t_delay
    )
    model.sensor_spring_total_works[s_id] = sstw
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_WORK", tdelay=t_delay
    ))




def read_sensor_spring_rotational_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_ROTATIONAL_WORK`` or ``/SENSOR/SPRING_ROT_WORK`` (M279): Spring element rotational work energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_ROTATIONAL_WORK/{block.user_id}: missing data card", block.source)
        return

    spring_id, w_rot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_ROTATIONAL_WORK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        w_rot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        w_rot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringRotationalWork, Sensor
    s_id = block.user_id or (len(model.sensor_spring_rotational_works) + 1)
    ssrw = SensorSpringRotationalWork(
        id=s_id, title=title, spring_id=spring_id,
        w_rot_max=w_rot_max, t_delay=t_delay
    )
    model.sensor_spring_rotational_works[s_id] = ssrw
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_ROTATIONAL_WORK", tdelay=t_delay
    ))




def read_sensor_spring_translational_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSLATIONAL_WORK`` or ``/SENSOR/SPRING_TRANS_WORK`` (M280): Spring element translational work energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSLATIONAL_WORK/{block.user_id}: missing data card", block.source)
        return

    spring_id, w_trans_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSLATIONAL_WORK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        w_trans_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        w_trans_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTranslationalWork, Sensor
    s_id = block.user_id or (len(model.sensor_spring_translational_works) + 1)
    sstw = SensorSpringTranslationalWork(
        id=s_id, title=title, spring_id=spring_id,
        w_trans_max=w_trans_max, t_delay=t_delay
    )
    model.sensor_spring_translational_works[s_id] = sstw
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSLATIONAL_WORK", tdelay=t_delay
    ))




def read_sensor_spring_shear_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_SHEAR_WORK`` or ``/SENSOR/SPRING_SHR_WORK`` (M281): Spring element transverse shear work energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_SHEAR_WORK/{block.user_id}: missing data card", block.source)
        return

    spring_id, w_shear_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_SHEAR_WORK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        w_shear_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        w_shear_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringShearWork, Sensor
    s_id = block.user_id or (len(model.sensor_spring_shear_works) + 1)
    sssw = SensorSpringShearWork(
        id=s_id, title=title, spring_id=spring_id,
        w_shear_max=w_shear_max, t_delay=t_delay
    )
    model.sensor_spring_shear_works[s_id] = sssw
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_SHEAR_WORK", tdelay=t_delay
    ))




def read_sensor_spring_normal_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_WORK`` or ``/SENSOR/SPRING_NORM_WORK`` (M282): Spring element normal/axial work energy threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_WORK/{block.user_id}: missing data card", block.source)
        return

    spring_id, w_norm_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_WORK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        w_norm_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        w_norm_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalWork, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_works) + 1)
    ssnw = SensorSpringNormalWork(
        id=s_id, title=title, spring_id=spring_id,
        w_norm_max=w_norm_max, t_delay=t_delay
    )
    model.sensor_spring_normal_works[s_id] = ssnw
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_WORK", tdelay=t_delay
    ))




def read_sensor_spring_total_force(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_FORCE`` or ``/SENSOR/SPRING_TOT_FORCE`` (M283): Spring element resultant force magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_FORCE/{block.user_id}: missing data card", block.source)
        return

    spring_id, f_tot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_FORCE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        f_tot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        f_tot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalForce, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_forces) + 1)
    sstf = SensorSpringTotalForce(
        id=s_id, title=title, spring_id=spring_id,
        f_tot_max=f_tot_max, t_delay=t_delay
    )
    model.sensor_spring_total_forces[s_id] = sstf
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_FORCE", tdelay=t_delay
    ))




def read_sensor_spring_total_moment(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_MOMENT`` or ``/SENSOR/SPRING_TOT_MOMENT`` (M284): Spring element resultant torque/moment magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_MOMENT/{block.user_id}: missing data card", block.source)
        return

    spring_id, m_tot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_MOMENT_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        m_tot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        m_tot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalMoment, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_moments) + 1)
    sstm = SensorSpringTotalMoment(
        id=s_id, title=title, spring_id=spring_id,
        m_tot_max=m_tot_max, t_delay=t_delay
    )
    model.sensor_spring_total_moments[s_id] = sstm
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_MOMENT", tdelay=t_delay
    ))




def read_sensor_spring_angular_velocity(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_ANGULAR_VELOCITY`` or ``/SENSOR/SPRING_ANG_VEL`` (M285): Spring element relative rotational velocity / angular rate threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_ANGULAR_VELOCITY/{block.user_id}: missing data card", block.source)
        return

    spring_id, omega_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_ANGULAR_VELOCITY_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        omega_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        omega_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringAngularVelocity, Sensor
    s_id = block.user_id or (len(model.sensor_spring_angular_velocities) + 1)
    ssav = SensorSpringAngularVelocity(
        id=s_id, title=title, spring_id=spring_id,
        omega_max=omega_max, t_delay=t_delay
    )
    model.sensor_spring_angular_velocities[s_id] = ssav
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_ANGULAR_VELOCITY", tdelay=t_delay
    ))




def read_sensor_spring_angular_acceleration(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_ANGULAR_ACCELERATION`` or ``/SENSOR/SPRING_ANG_ACC`` (M286): Spring element relative angular acceleration threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_ANGULAR_ACCELERATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, alpha_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_ANGULAR_ACCELERATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        alpha_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        alpha_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringAngularAcceleration, Sensor
    s_id = block.user_id or (len(model.sensor_spring_angular_accelerations) + 1)
    ssaa = SensorSpringAngularAcceleration(
        id=s_id, title=title, spring_id=spring_id,
        alpha_max=alpha_max, t_delay=t_delay
    )
    model.sensor_spring_angular_accelerations[s_id] = ssaa
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_ANGULAR_ACCELERATION", tdelay=t_delay
    ))




def read_sensor_spring_torsional_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_RATE`` or ``/SENSOR/SPRING_TORS_RATE`` (M287): Spring element torque rate-of-change / torsional loading rate threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, mdot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        mdot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        mdot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_rates) + 1)
    sstr = SensorSpringTorsionalRate(
        id=s_id, title=title, spring_id=spring_id,
        mdot_max=mdot_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_rates[s_id] = sstr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_acceleration(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_ACCELERATION`` or ``/SENSOR/SPRING_NORM_ACC`` (M288): Spring element relative normal / axial acceleration threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_ACCELERATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, accn_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_ACCELERATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        accn_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        accn_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalAcceleration, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_accelerations) + 1)
    ssna = SensorSpringNormalAcceleration(
        id=s_id, title=title, spring_id=spring_id,
        accn_max=accn_max, t_delay=t_delay
    )
    model.sensor_spring_normal_accelerations[s_id] = ssna
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_ACCELERATION", tdelay=t_delay
    ))




def read_sensor_spring_shear_acceleration(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_SHEAR_ACCELERATION`` or ``/SENSOR/SPRING_SHEAR_ACC`` (M289): Spring element relative transverse shear acceleration threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_SHEAR_ACCELERATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, accs_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_SHEAR_ACCELERATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        accs_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        accs_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringShearAcceleration, Sensor
    s_id = block.user_id or (len(model.sensor_spring_shear_accelerations) + 1)
    sssa = SensorSpringShearAcceleration(
        id=s_id, title=title, spring_id=spring_id,
        accs_max=accs_max, t_delay=t_delay
    )
    model.sensor_spring_shear_accelerations[s_id] = sssa
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_SHEAR_ACCELERATION", tdelay=t_delay
    ))




def read_sensor_spring_resultant_acceleration(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_RESULTANT_ACCELERATION`` or ``/SENSOR/SPRING_RES_ACC`` (M290): Spring element relative 3D vector resultant translational acceleration magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_RESULTANT_ACCELERATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, accr_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_RESULTANT_ACCELERATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        accr_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        accr_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringResultantAcceleration, Sensor
    s_id = block.user_id or (len(model.sensor_spring_resultant_accelerations) + 1)
    ssra = SensorSpringResultantAcceleration(
        id=s_id, title=title, spring_id=spring_id,
        accr_max=accr_max, t_delay=t_delay
    )
    model.sensor_spring_resultant_accelerations[s_id] = ssra
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_RESULTANT_ACCELERATION", tdelay=t_delay
    ))




def read_sensor_spring_torsional_acceleration(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_ACCELERATION`` or ``/SENSOR/SPRING_TORS_ACC`` (M291): Spring element relative torsional angular acceleration threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_ACCELERATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, alphat_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_ACCELERATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        alphat_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        alphat_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalAcceleration, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_accelerations) + 1)
    ssta = SensorSpringTorsionalAcceleration(
        id=s_id, title=title, spring_id=spring_id,
        alphat_max=alphat_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_accelerations[s_id] = ssta
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_ACCELERATION", tdelay=t_delay
    ))




def read_sensor_spring_bending_acceleration(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_ACCELERATION`` or ``/SENSOR/SPRING_BEND_ACC`` (M292): Spring element relative transverse bending angular acceleration threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_ACCELERATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, alphab_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_ACCELERATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        alphab_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        alphab_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingAcceleration, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_accelerations) + 1)
    ssba = SensorSpringBendingAcceleration(
        id=s_id, title=title, spring_id=spring_id,
        alphab_max=alphab_max, t_delay=t_delay
    )
    model.sensor_spring_bending_accelerations[s_id] = ssba
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_ACCELERATION", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_acceleration(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION`` or ``/SENSOR/SPRING_TOT_ANG_ACC`` (M293): Spring element relative 3D resultant total angular acceleration magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION/{block.user_id}: missing data card", block.source)
        return

    spring_id, alpha_tot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_ACCELERATION_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        alpha_tot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        alpha_tot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularAcceleration, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_accelerations) + 1)
    sstaa = SensorSpringTotalAngularAcceleration(
        id=s_id, title=title, spring_id=spring_id,
        alpha_tot_max=alpha_tot_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_accelerations[s_id] = sstaa
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_ACCELERATION", tdelay=t_delay
    ))




def read_sensor_spring_normal_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_JERK`` or ``/SENSOR/SPRING_NORM_JERK`` (M294): Spring element relative normal / axial jerk (rate of change of linear acceleration) threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, jn_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jn_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jn_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_jerks) + 1)
    ssnj = SensorSpringNormalJerk(
        id=s_id, title=title, spring_id=spring_id,
        jn_max=jn_max, t_delay=t_delay
    )
    model.sensor_spring_normal_jerks[s_id] = ssnj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_JERK", tdelay=t_delay
    ))




def read_sensor_spring_shear_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_SHEAR_JERK`` or ``/SENSOR/SPRING_SHR_JERK`` (M295): Spring element relative transverse shear jerk (rate of change of transverse linear acceleration) threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_SHEAR_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, js_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_SHEAR_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        js_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        js_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringShearJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_shear_jerks) + 1)
    sssj = SensorSpringShearJerk(
        id=s_id, title=title, spring_id=spring_id,
        js_max=js_max, t_delay=t_delay
    )
    model.sensor_spring_shear_jerks[s_id] = sssj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_SHEAR_JERK", tdelay=t_delay
    ))




def read_sensor_spring_resultant_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_RESULTANT_JERK`` or ``/SENSOR/SPRING_RES_JERK`` (M296): Spring element relative 3D resultant linear jerk (rate of change of linear acceleration) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_RESULTANT_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, jres_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_RESULTANT_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jres_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jres_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringResultantJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_resultant_jerks) + 1)
    ssrj = SensorSpringResultantJerk(
        id=s_id, title=title, spring_id=spring_id,
        jres_max=jres_max, t_delay=t_delay
    )
    model.sensor_spring_resultant_jerks[s_id] = ssrj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_RESULTANT_JERK", tdelay=t_delay
    ))




def read_sensor_spring_torsional_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_JERK`` or ``/SENSOR/SPRING_TORS_JERK`` (M297): Spring element relative torsional jerk (rate of change of angular acceleration) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_jerks) + 1)
    sstj = SensorSpringTorsionalJerk(
        id=s_id, title=title, spring_id=spring_id,
        jtors_max=jtors_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_jerks[s_id] = sstj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_JERK", tdelay=t_delay
    ))




def read_sensor_spring_bending_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_JERK`` or ``/SENSOR/SPRING_BEND_JERK`` (M298): Spring element relative transverse bending angular jerk (rate of change of bending angular acceleration) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_jerks) + 1)
    ssbj = SensorSpringBendingJerk(
        id=s_id, title=title, spring_id=spring_id,
        jbend_max=jbend_max, t_delay=t_delay
    )
    model.sensor_spring_bending_jerks[s_id] = ssbj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_JERK", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_JERK`` or ``/SENSOR/SPRING_TOT_ANG_JERK`` (M299): Spring element relative 3D resultant total angular jerk (rate of change of total angular acceleration) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtota_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtota_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtota_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_jerks) + 1)
    sstaj = SensorSpringTotalAngularJerk(
        id=s_id, title=title, spring_id=spring_id,
        jtota_max=jtota_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_jerks[s_id] = sstaj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_JERK", tdelay=t_delay
    ))




def read_sensor_spring_total_acceleration_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ACCELERATION_RATE`` or ``/SENSOR/SPRING_TOT_ACC_RATE`` (M300): Spring element relative 3D resultant total acceleration rate-of-change (generalized resultant 6-DOF dynamic jerk) threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ACCELERATION_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jrate_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ACCELERATION_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jrate_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jrate_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAccelerationRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_acceleration_rates) + 1)
    sstarr = SensorSpringTotalAccelerationRate(
        id=s_id, title=title, spring_id=spring_id,
        jrate_max=jrate_max, t_delay=t_delay
    )
    model.sensor_spring_total_acceleration_rates[s_id] = sstarr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ACCELERATION_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_acceleration_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_ACCELERATION_RATE`` or ``/SENSOR/SPRING_NORM_ACC_RATE`` (M301): Spring element relative normal / axial acceleration rate-of-change (axial jerk) threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_ACCELERATION_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_rate_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_ACCELERATION_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_rate_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_rate_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalAccelerationRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_acceleration_rates) + 1)
    ssnarr = SensorSpringNormalAccelerationRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_rate_max=jnorm_rate_max, t_delay=t_delay
    )
    model.sensor_spring_normal_acceleration_rates[s_id] = ssnarr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_ACCELERATION_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_acceleration_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_ACCELERATION_RATE`` or ``/SENSOR/SPRING_TRANS_ACC_RATE`` (M302): Spring element relative transverse / shear acceleration rate-of-change (shear jerk) threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_ACCELERATION_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_rate_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_ACCELERATION_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_rate_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_rate_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseAccelerationRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_acceleration_rates) + 1)
    sstarr = SensorSpringTransverseAccelerationRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_rate_max=jtrans_rate_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_acceleration_rates[s_id] = sstarr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_ACCELERATION_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_acceleration_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_ACCELERATION_RATE`` or ``/SENSOR/SPRING_TORS_ACC_RATE`` (M303): Spring element relative torsional angular acceleration rate-of-change (torsional angular jerk) threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_ACCELERATION_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_rate_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_ACCELERATION_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_rate_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_rate_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalAccelerationRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_acceleration_rates) + 1)
    sstar = SensorSpringTorsionalAccelerationRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_rate_max=jtors_rate_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_acceleration_rates[s_id] = sstar
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_ACCELERATION_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_acceleration_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_ACCELERATION_RATE`` or ``/SENSOR/SPRING_BEND_ACC_RATE`` (M304): Spring element relative transverse bending angular acceleration rate-of-change (bending angular jerk) threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_ACCELERATION_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_rate_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_ACCELERATION_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_rate_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_rate_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingAccelerationRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_acceleration_rates) + 1)
    ssbar = SensorSpringBendingAccelerationRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_rate_max=jbend_rate_max, t_delay=t_delay
    )
    model.sensor_spring_bending_acceleration_rates[s_id] = ssbar
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_ACCELERATION_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_acceleration_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION_RATE`` or ``/SENSOR/SPRING_TOT_ANG_ACC_RATE`` (M305): Spring element relative 3D resultant total angular acceleration rate-of-change (total angular jerk) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_rate_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_ACCELERATION_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_rate_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_rate_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularAccelerationRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_acceleration_rates) + 1)
    sstaar = SensorSpringTotalAngularAccelerationRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_rate_max=jtot_ang_rate_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_acceleration_rates[s_id] = sstaar
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_ACCELERATION_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_acceleration_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ACCELERATION_JERK`` or ``/SENSOR/SPRING_TOT_ACC_JERK`` (M306): Spring element relative 3D resultant total linear and angular combined acceleration rate-of-change (generalized resultant 6-DOF jerk) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ACCELERATION_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_comb_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ACCELERATION_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_comb_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_comb_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAccelerationJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_acceleration_jerks) + 1)
    sstaj = SensorSpringTotalAccelerationJerk(
        id=s_id, title=title, spring_id=spring_id,
        jtot_comb_max=jtot_comb_max, t_delay=t_delay
    )
    model.sensor_spring_total_acceleration_jerks[s_id] = sstaj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ACCELERATION_JERK", tdelay=t_delay
    ))




def read_sensor_spring_normal_acceleration_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_ACCELERATION_JERK`` or ``/SENSOR/SPRING_NORM_ACC_JERK`` (M307): Spring element relative normal / axial acceleration rate-of-change (axial jerk) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_ACCELERATION_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_rate_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_ACCELERATION_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_rate_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_rate_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalAccelerationJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_acceleration_jerks) + 1)
    ssnaj = SensorSpringNormalAccelerationJerk(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_rate_max=jnorm_rate_max, t_delay=t_delay
    )
    model.sensor_spring_normal_acceleration_jerks[s_id] = ssnaj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_ACCELERATION_JERK", tdelay=t_delay
    ))




def read_sensor_spring_transverse_acceleration_jerk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_ACCELERATION_JERK`` or ``/SENSOR/SPRING_TRANS_ACC_JERK`` (M308): Spring element relative transverse / shear acceleration rate-of-change (shear jerk) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_ACCELERATION_JERK/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_rate_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_ACCELERATION_JERK_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_rate_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_rate_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseAccelerationJerk, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_acceleration_jerks) + 1)
    sstaj = SensorSpringTransverseAccelerationJerk(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_rate_max=jtrans_rate_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_acceleration_jerks[s_id] = sstaj
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_ACCELERATION_JERK", tdelay=t_delay
    ))




def read_sensor_spring_torsional_jerk_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_JERK_RATE`` or ``/SENSOR/SPRING_TORS_JERK_RATE`` (M309): Spring element relative torsional angular jerk rate-of-change (torsional angular snap/jounce) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_JERK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_snap_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_JERK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_snap_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_snap_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalJerkRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_jerk_rates) + 1)
    sstjr = SensorSpringTorsionalJerkRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_snap_max=jtors_snap_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_jerk_rates[s_id] = sstjr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_JERK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_jerk_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_JERK_RATE`` or ``/SENSOR/SPRING_BEND_JERK_RATE`` (M310): Spring element relative transverse bending angular jerk rate-of-change (bending angular snap/jounce) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_JERK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_snap_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_JERK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_snap_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_snap_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingJerkRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_jerk_rates) + 1)
    ssbjr = SensorSpringBendingJerkRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_snap_max=jbend_snap_max, t_delay=t_delay
    )
    model.sensor_spring_bending_jerk_rates[s_id] = ssbjr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_JERK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_jerk_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_JERK_RATE`` or ``/SENSOR/SPRING_TOT_JERK_RATE`` (M311): Spring element relative 3D resultant total linear and angular combined jerk rate-of-change (generalized resultant 6-DOF snap/jounce) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_JERK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_snap_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_JERK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_snap_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_snap_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalJerkRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_jerk_rates) + 1)
    sstjr = SensorSpringTotalJerkRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_snap_max=jtot_snap_max, t_delay=t_delay
    )
    model.sensor_spring_total_jerk_rates[s_id] = sstjr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_JERK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_jerk_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_JERK_RATE`` or ``/SENSOR/SPRING_NORM_JERK_RATE`` (M312): Spring element relative normal / axial acceleration rate-of-change rate (axial snap/jounce) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_JERK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_snap_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_JERK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_snap_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_snap_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalJerkRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_jerk_rates) + 1)
    ssnjr = SensorSpringNormalJerkRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_snap_max=jnorm_snap_max, t_delay=t_delay
    )
    model.sensor_spring_normal_jerk_rates[s_id] = ssnjr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_JERK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_jerk_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_JERK_RATE`` or ``/SENSOR/SPRING_TRANS_JERK_RATE`` (M313): Spring element relative transverse / shear acceleration rate-of-change rate (shear snap/jounce) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_JERK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_snap_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_JERK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_snap_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_snap_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseJerkRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_jerk_rates) + 1)
    sstjr = SensorSpringTransverseJerkRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_snap_max=jtrans_snap_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_jerk_rates[s_id] = sstjr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_JERK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M314): Spring element relative torsional angular acceleration 2nd rate-of-change (torsional angular snap/crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_crackle_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_crackle_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_crackle_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_snap_rates) + 1)
    sstsr = SensorSpringTorsionalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_crackle_max=jtors_crackle_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_snap_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_SNAP_RATE`` or ``/SENSOR/SPRING_BEND_SNAP_RATE`` (M315): Spring element relative transverse bending angular acceleration 2nd rate-of-change (bending angular snap/crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_crackle_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_crackle_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_crackle_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_snap_rates) + 1)
    ssbsr = SensorSpringBendingSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_crackle_max=jbend_crackle_max, t_delay=t_delay
    )
    model.sensor_spring_bending_snap_rates[s_id] = ssbsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_axial_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_AXIAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_AXIAL_CRK_RATE`` (M436): Spring element relative normal / axial acceleration 21st rate-of-change (axial crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_AXIAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jax_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_AXIAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jax_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jax_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringAxialCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_axial_crackle_rates) + 1)
    ssnsr = SensorSpringAxialCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jax_crk_max=jax_crk_max, t_delay=t_delay
    )
    model.sensor_spring_axial_crackle_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_AXIAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M458): Spring element relative resultant total angular acceleration 22nd rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_pop_rates) + 1)
    sstapr = SensorSpringTotalAngularPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_pop_max=jtot_ang_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_pop_rates[s_id] = sstapr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_POP_RATE", tdelay=t_delay
    ))



def read_sensor_spring_torsional_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORSION_POP_RATE`` (M457): Spring element relative torsional acceleration 22nd rate-of-change (torsional pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtor_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtor_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtor_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_pop_rates) + 1)
    sstpr = SensorSpringTorsionalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtor_pop_max=jtor_pop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_POP_RATE", tdelay=t_delay
    ))



def read_sensor_spring_coupled_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_COUPLED_POP_RATE`` or ``/SENSOR/SPRING_COUPLE_POP_RATE`` (M456): Spring element relative multi-axial coupled acceleration 22nd rate-of-change (coupled pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_COUPLED_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jcoup_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_COUPLED_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jcoup_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jcoup_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringCoupledPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_coupled_pop_rates) + 1)
    sscpr = SensorSpringCoupledPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jcoup_pop_max=jcoup_pop_max, t_delay=t_delay
    )
    model.sensor_spring_coupled_pop_rates[s_id] = sscpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_COUPLED_POP_RATE", tdelay=t_delay
    ))



def read_sensor_spring_transverse_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M455): Spring element relative transverse / shear acceleration 22nd rate-of-change (shear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransversePopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_pop_rates) + 1)
    sstpr = SensorSpringTransversePopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_pop_max=jtrans_pop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_POP_RATE", tdelay=t_delay
    ))



def read_sensor_spring_normal_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M454): Spring element relative normal / axial acceleration 22nd rate-of-change (axial pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_pop_rates) + 1)
    ssnpr = SensorSpringNormalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_pop_max=jnorm_pop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_pop_rates[s_id] = ssnpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_POP_RATE", tdelay=t_delay
    ))



def read_sensor_spring_total_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_CRACKLE_RATE`` (M453): Spring element relative 3D resultant total linear acceleration 21st rate-of-change (resultant total crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_crackle_rates) + 1)
    sstcr = SensorSpringTotalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_crk_max=jtot_crk_max, t_delay=t_delay
    )
    model.sensor_spring_total_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_CRACKLE_RATE", tdelay=t_delay
    ))



def read_sensor_spring_torsional_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TORS_CRACKLE_RATE`` (M452): Spring element relative torsional acceleration 21st rate-of-change (torsional crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_crackle_rates) + 1)
    sstcr = SensorSpringTorsionalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_crk_max=jtors_crk_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_CRACKLE_RATE", tdelay=t_delay
    ))



def read_sensor_spring_coupled_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_COUPLED_CRACKLE_RATE`` or ``/SENSOR/SPRING_COUPLE_CRACKLE_RATE`` (M451): Spring element relative coupled / mixed-mode acceleration 21st rate-of-change (coupled crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_COUPLED_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jcoup_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_COUPLED_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jcoup_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jcoup_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringCoupledCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_coupled_crackle_rates) + 1)
    ssccr = SensorSpringCoupledCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jcoup_crk_max=jcoup_crk_max, t_delay=t_delay
    )
    model.sensor_spring_coupled_crackle_rates[s_id] = ssccr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_COUPLED_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_SNAP_RATE`` or ``/SENSOR/SPRING_NORM_SNAP_RATE`` (M316): Spring element relative normal / axial acceleration 2nd rate-of-change (axial snap/crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_crackle_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_crackle_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_crackle_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_snap_rates) + 1)
    ssnsr = SensorSpringNormalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_crackle_max=jnorm_crackle_max, t_delay=t_delay
    )
    model.sensor_spring_normal_snap_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_SNAP_RATE`` or ``/SENSOR/SPRING_TRANS_SNAP_RATE`` (M317): Spring element relative transverse / shear acceleration 2nd rate-of-change (shear snap/crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_crackle_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_crackle_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_crackle_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_snap_rates) + 1)
    sstsr = SensorSpringTransverseSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_crackle_max=jtrans_crackle_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_snap_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_SNAP_RATE`` (M318): Spring element relative 3D resultant total linear and angular combined acceleration 2nd rate-of-change (generalized resultant 6-DOF snap/crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_crackle_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_crackle_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_crackle_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_snap_rates) + 1)
    sstsr = SensorSpringTotalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_crackle_max=jtot_crackle_max, t_delay=t_delay
    )
    model.sensor_spring_total_snap_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M433): Spring element relative torsional angular acceleration 20th rate-of-change (torsional angular snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_snap_rates) + 1)
    sstr = SensorSpringTorsionalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_snp_max=jtors_snp_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_snap_rates[s_id] = sstr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_NORM_CRACKLE_RATE`` (M319): Spring element relative normal / axial acceleration 3rd rate-of-change (axial crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_crackle_rates) + 1)
    ssncr = SensorSpringNormalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_pop_max=jnorm_pop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_crackle_rates[s_id] = ssncr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE`` or ``/SENSOR/SPRING_TRANS_CRACKLE_RATE`` (M320): Spring element relative transverse / shear acceleration 3rd rate-of-change (shear crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_crackle_rates) + 1)
    sstcr = SensorSpringTransverseCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_pop_max=jtrans_pop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_CRACKLE_RATE`` (M321): Spring element relative 3D resultant total linear and angular combined acceleration 3rd rate-of-change (generalized resultant 6-DOF crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_crackle_rates) + 1)
    sstcr = SensorSpringTotalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_pop_max=jtot_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TORS_CRACKLE_RATE`` (M322): Spring element relative torsional angular acceleration 3rd rate-of-change (torsional crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_crackle_rates) + 1)
    sstcr = SensorSpringTorsionalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_pop_max=jtors_pop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_CRACKLE_RATE`` or ``/SENSOR/SPRING_BEND_CRACKLE_RATE`` (M323): Spring element relative bending angular acceleration 3rd rate-of-change (bending crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_crackle_rates) + 1)
    ssbcr = SensorSpringBendingCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_pop_max=jbend_pop_max, t_delay=t_delay
    )
    model.sensor_spring_bending_crackle_rates[s_id] = ssbcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE`` (M324): Spring element relative 3D resultant total angular acceleration 3rd rate-of-change (total angular crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jang_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jang_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jang_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_crackle_rates) + 1)
    sstacr = SensorSpringTotalAngularCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jang_pop_max=jang_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_crackle_rates[s_id] = sstacr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M325): Spring element relative normal / axial acceleration 3rd rate-of-change (axial pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_pop_rates) + 1)
    ssnpr = SensorSpringNormalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_pop_max=jnorm_pop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_pop_rates[s_id] = ssnpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M326): Spring element relative transverse / shear acceleration 3rd rate-of-change (shear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransversePopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_pop_rates) + 1)
    sstpr = SensorSpringTransversePopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_pop_max=jtrans_pop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_POP_RATE`` or ``/SENSOR/SPRING_TOT_POP_RATE`` (M327): Spring element relative 3D resultant total linear acceleration 3rd rate-of-change (resultant total linear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_pop_rates) + 1)
    sstpr = SensorSpringTotalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_pop_max=jtot_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORS_POP_RATE`` (M328): Spring element relative torsional angular acceleration 3rd rate-of-change (torsional pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_pop_rates) + 1)
    sstpr = SensorSpringTorsionalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_pop_max=jtors_pop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_POP_RATE`` or ``/SENSOR/SPRING_BEND_POP_RATE`` (M329): Spring element relative bending angular acceleration 3rd rate-of-change (bending pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_pop_rates) + 1)
    ssbpr = SensorSpringBendingPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_pop_max=jbend_pop_max, t_delay=t_delay
    )
    model.sensor_spring_bending_pop_rates[s_id] = ssbpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M330): Spring element relative 3D resultant total angular acceleration 3rd rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_pop_rates) + 1)
    sstapr = SensorSpringTotalAngularPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_pop_max=jtot_ang_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_pop_rates[s_id] = sstapr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_LOCK_RATE`` or ``/SENSOR/SPRING_NORM_LOCK_RATE`` (M331): Spring element relative normal/axial acceleration 4th rate-of-change (axial lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_lock_rates) + 1)
    ssnlr = SensorSpringNormalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_lock_max=jnorm_lock_max, t_delay=t_delay
    )
    model.sensor_spring_normal_lock_rates[s_id] = ssnlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_LOCK_RATE`` or ``/SENSOR/SPRING_TRANS_LOCK_RATE`` (M332): Spring element relative transverse/shear acceleration 4th rate-of-change (shear lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_lock_rates) + 1)
    sstlr = SensorSpringTransverseLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_lock_max=jtrans_lock_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_LOCK_RATE`` (M333): Spring element relative 3D resultant total linear acceleration 4th rate-of-change (resultant total linear lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_lock_rates) + 1)
    sstlr = SensorSpringTotalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_lock_max=jtot_lock_max, t_delay=t_delay
    )
    model.sensor_spring_total_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_LOCK_RATE`` or ``/SENSOR/SPRING_TORS_LOCK_RATE`` (M334): Spring element relative torsional angular acceleration 4th rate-of-change (torsional angular lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_lock_rates) + 1)
    sstlr = SensorSpringTorsionalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_lock_max=jtors_lock_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_LOCK_RATE`` or ``/SENSOR/SPRING_BEND_LOCK_RATE`` (M335): Spring element relative transverse bending angular acceleration 4th rate-of-change (bending angular lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_lock_rates) + 1)
    ssblr = SensorSpringBendingLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_lock_max=jbend_lock_max, t_delay=t_delay
    )
    model.sensor_spring_bending_lock_rates[s_id] = ssblr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_ANG_LOCK_RATE`` (M336): Spring element relative 3D resultant total angular acceleration 4th rate-of-change (resultant total angular lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_lock_rates) + 1)
    sstar = SensorSpringTotalAngularLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_lock_max=jtot_ang_lock_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_lock_rates[s_id] = sstar
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_DROP_RATE`` or ``/SENSOR/SPRING_NORM_DROP_RATE`` (M337): Spring element relative normal / axial acceleration 5th rate-of-change (axial drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_drop_rates) + 1)
    ssndr = SensorSpringNormalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_drop_max=jnorm_drop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_drop_rates[s_id] = ssndr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_DROP_RATE`` or ``/SENSOR/SPRING_TRANS_DROP_RATE`` (M422): Spring element relative transverse / shear acceleration 18th rate-of-change (shear drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_drop_rates) + 1)
    sstdr = SensorSpringTransverseDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_drop_max=jtrans_drop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_DROP_RATE`` or ``/SENSOR/SPRING_TOT_DROP_RATE`` (M423): Spring element relative 3D resultant total linear acceleration 18th rate-of-change (resultant total linear drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_drop_rates) + 1)
    sstdr = SensorSpringTotalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_drop_max=jtot_drop_max, t_delay=t_delay
    )
    model.sensor_spring_total_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_DROP_RATE`` or ``/SENSOR/SPRING_TORS_DROP_RATE`` (M424): Spring element relative torsional angular acceleration 18th rate-of-change (torsional angular drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_drop_rates) + 1)
    sstdr = SensorSpringTorsionalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_drop_max=jtors_drop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_DROP_RATE`` or ``/SENSOR/SPRING_BEND_DROP_RATE`` (M425): Spring element relative transverse bending angular acceleration 18th rate-of-change (bending angular drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_drop_rates) + 1)
    ssbdr = SensorSpringBendingDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_drop_max=jbend_drop_max, t_delay=t_delay
    )
    model.sensor_spring_bending_drop_rates[s_id] = ssbdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_DROP_RATE`` (M426): Spring element relative 3D resultant total angular acceleration 18th rate-of-change (resultant total angular drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_drop_rates) + 1)
    sstdr = SensorSpringTotalAngularDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_drop_max=jtot_ang_drop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_rate_of_change(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_RATE_OF_CHANGE`` or ``/SENSOR/SPRING_TORS_RATE_OF_CHANGE`` (M427): Spring element relative torsional angular acceleration 19th rate-of-change (torsional angular rate of change) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_RATE_OF_CHANGE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_roc_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_RATE_OF_CHANGE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_roc_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_roc_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalRateOfChange, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_rate_of_changes) + 1)
    sstroc = SensorSpringTorsionalRateOfChange(
        id=s_id, title=title, spring_id=spring_id,
        jtors_roc_max=jtors_roc_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_rate_of_changes[s_id] = sstroc
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_RATE_OF_CHANGE", tdelay=t_delay
    ))




def read_sensor_spring_bending_rate_of_change(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_RATE_OF_CHANGE`` or ``/SENSOR/SPRING_BEND_RATE_OF_CHANGE`` (M428): Spring element relative transverse bending angular acceleration 19th rate-of-change (bending angular rate of change) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_RATE_OF_CHANGE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_roc_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_RATE_OF_CHANGE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_roc_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_roc_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingRateOfChange, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_rate_of_changes) + 1)
    ssbroc = SensorSpringBendingRateOfChange(
        id=s_id, title=title, spring_id=spring_id,
        jbend_roc_max=jbend_roc_max, t_delay=t_delay
    )
    model.sensor_spring_bending_rate_of_changes[s_id] = ssbroc
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_RATE_OF_CHANGE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_rate_of_change(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE`` or ``/SENSOR/SPRING_TOT_ANG_RATE_OF_CHANGE`` (M429): Spring element relative 3D resultant total angular acceleration 19th rate-of-change (resultant total angular rate of change) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_roc_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_roc_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_roc_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularRateOfChange, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_rate_of_changes) + 1)
    sstaroc = SensorSpringTotalAngularRateOfChange(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_roc_max=jtot_ang_roc_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_rate_of_changes[s_id] = sstaroc
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE", tdelay=t_delay
    ))




def read_sensor_spring_axial_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_AXIAL_SNAP_RATE`` or ``/SENSOR/SPRING_AXIAL_SNAP`` (M430): Spring element relative axial linear acceleration 20th rate-of-change (axial snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_AXIAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jax_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_AXIAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jax_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jax_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringAxialSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_axial_snap_rates) + 1)
    ssasr = SensorSpringAxialSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jax_snp_max=jax_snp_max, t_delay=t_delay
    )
    model.sensor_spring_axial_snap_rates[s_id] = ssasr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_AXIAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_SNAP_RATE`` or ``/SENSOR/SPRING_BEND_SNAP_RATE`` (M434): Spring element relative transverse bending angular acceleration 20th rate-of-change (bending angular snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbnd_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbnd_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbnd_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_snap_rates) + 1)
    sstr = SensorSpringBendingSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jbnd_snp_max=jbnd_snp_max, t_delay=t_delay
    )
    model.sensor_spring_bending_snap_rates[s_id] = sstr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M433): Spring element relative torsional angular acceleration 20th rate-of-change (torsional angular snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtor_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtor_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtor_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_snap_rates) + 1)
    sstr = SensorSpringTorsionalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtor_snp_max=jtor_snp_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_snap_rates[s_id] = sstr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_SNAP_RATE`` (M432): Spring element relative 3D resultant total linear acceleration 20th rate-of-change (resultant total linear snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_snap_rates) + 1)
    sstr = SensorSpringTotalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_snp_max=jtot_snp_max, t_delay=t_delay
    )
    model.sensor_spring_total_snap_rates[s_id] = sstr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M433): Spring element relative torsional angular acceleration 20th rate-of-change (torsional angular snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_snap_rates) + 1)
    sstr = SensorSpringTorsionalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_snp_max=jtors_snp_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_snap_rates[s_id] = sstr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_SNAP_RATE`` or ``/SENSOR/SPRING_TRANS_SNAP_RATE`` (M431): Spring element relative transverse / shear linear acceleration 20th rate-of-change (transverse snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtr_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtr_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtr_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_snap_rates) + 1)
    sstr = SensorSpringTransverseSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtr_snp_max=jtr_snp_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_snap_rates[s_id] = sstr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_axial_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_AXIAL_SNAP_RATE`` or ``/SENSOR/SPRING_AXIAL_SNAP`` (M430): Spring element relative axial linear acceleration 20th rate-of-change (axial snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_AXIAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jax_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_AXIAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jax_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jax_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringAxialSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_axial_snap_rates) + 1)
    ssasr = SensorSpringAxialSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jax_snp_max=jax_snp_max, t_delay=t_delay
    )
    model.sensor_spring_axial_snap_rates[s_id] = ssasr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_AXIAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_rate_of_change(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_RATE_OF_CHANGE`` or ``/SENSOR/SPRING_BEND_RATE_OF_CHANGE`` (M428): Spring element relative transverse bending angular acceleration 19th rate-of-change (bending angular rate of change) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_RATE_OF_CHANGE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_roc_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_RATE_OF_CHANGE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_roc_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_roc_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingRateOfChange, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_rate_of_changes) + 1)
    ssbroc = SensorSpringBendingRateOfChange(
        id=s_id, title=title, spring_id=spring_id,
        jbend_roc_max=jbend_roc_max, t_delay=t_delay
    )
    model.sensor_spring_bending_rate_of_changes[s_id] = ssbroc
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_RATE_OF_CHANGE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_DROP_RATE`` or ``/SENSOR/SPRING_TRANS_DROP_RATE`` (M338): Spring element relative transverse / shear acceleration 5th rate-of-change (shear drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_drop_rates) + 1)
    sstdr = SensorSpringTransverseDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_drop_max=jtrans_drop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_DROP_RATE`` or ``/SENSOR/SPRING_TOT_DROP_RATE`` (M339): Spring element relative 3D resultant total linear acceleration 5th rate-of-change (resultant total linear drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_drop_rates) + 1)
    sstdr = SensorSpringTotalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_drop_max=jtot_drop_max, t_delay=t_delay
    )
    model.sensor_spring_total_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_DROP_RATE`` or ``/SENSOR/SPRING_TORS_DROP_RATE`` (M340): Spring element relative torsional angular acceleration 5th rate-of-change (torsional angular drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_drop_rates) + 1)
    sstdr = SensorSpringTorsionalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_drop_max=jtors_drop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_DROP_RATE`` or ``/SENSOR/SPRING_BEND_DROP_RATE`` (M341): Spring element relative transverse bending angular acceleration 5th rate-of-change (bending angular drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_drop_rates) + 1)
    ssbdr = SensorSpringBendingDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_drop_max=jbend_drop_max, t_delay=t_delay
    )
    model.sensor_spring_bending_drop_rates[s_id] = ssbdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_DROP_RATE`` (M342): Spring element relative 3D resultant total angular acceleration 5th rate-of-change (resultant total angular drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_drop_rates) + 1)
    sstadr = SensorSpringTotalAngularDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_drop_max=jtot_ang_drop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_drop_rates[s_id] = sstadr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_drift_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_DRIFT_RATE`` or ``/SENSOR/SPRING_NORM_DRIFT_RATE`` (M343): Spring element relative normal / axial acceleration 6th rate-of-change (axial drift rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_DRIFT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_drift_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_DRIFT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_drift_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_drift_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalDriftRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_drift_rates) + 1)
    ssndr = SensorSpringNormalDriftRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_drift_max=jnorm_drift_max, t_delay=t_delay
    )
    model.sensor_spring_normal_drift_rates[s_id] = ssndr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_DRIFT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_drift_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_DRIFT_RATE`` or ``/SENSOR/SPRING_TRANS_DRIFT_RATE`` (M344): Spring element relative transverse / shear acceleration 6th rate-of-change (shear drift rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_DRIFT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_drift_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_DRIFT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_drift_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_drift_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseDriftRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_drift_rates) + 1)
    sstdr = SensorSpringTransverseDriftRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_drift_max=jtrans_drift_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_drift_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_DRIFT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_drift_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_DRIFT_RATE`` or ``/SENSOR/SPRING_TOT_DRIFT_RATE`` (M345): Spring element relative 3D resultant total linear acceleration 6th rate-of-change (resultant total linear drift rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_DRIFT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_drift_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_DRIFT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_drift_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_drift_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalDriftRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_drift_rates) + 1)
    sstdr = SensorSpringTotalDriftRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_drift_max=jtot_drift_max, t_delay=t_delay
    )
    model.sensor_spring_total_drift_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_DRIFT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_drift_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_DRIFT_RATE`` or ``/SENSOR/SPRING_TORS_DRIFT_RATE`` (M346): Spring element relative torsional angular acceleration 6th rate-of-change (torsional angular drift rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_DRIFT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_drift_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_DRIFT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_drift_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_drift_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalDriftRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_drift_rates) + 1)
    sstdr = SensorSpringTorsionalDriftRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_drift_max=jtors_drift_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_drift_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_DRIFT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_drift_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_DRIFT_RATE`` or ``/SENSOR/SPRING_BEND_DRIFT_RATE`` (M347): Spring element relative transverse bending angular acceleration 6th rate-of-change (bending angular drift rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_DRIFT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_drift_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_DRIFT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_drift_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_drift_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingDriftRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_drift_rates) + 1)
    sstdr = SensorSpringBendingDriftRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_drift_max=jbend_drift_max, t_delay=t_delay
    )
    model.sensor_spring_bending_drift_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_DRIFT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_drift_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_DRIFT_RATE`` or ``/SENSOR/SPRING_TOT_ANG_DRIFT_RATE`` (M348): Spring element relative 3D resultant total angular acceleration 6th rate-of-change (resultant total angular drift rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_DRIFT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_drift_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_DRIFT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_drift_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_drift_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularDriftRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_drift_rates) + 1)
    sstdr = SensorSpringTotalAngularDriftRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_drift_max=jtot_ang_drift_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_drift_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_DRIFT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_surge_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_SURGE_RATE`` or ``/SENSOR/SPRING_NORM_SURGE_RATE`` (M349): Spring element relative normal / axial acceleration 7th rate-of-change (axial surge rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_SURGE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_surge_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_SURGE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_surge_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_surge_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalSurgeRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_surge_rates) + 1)
    ssnsr = SensorSpringNormalSurgeRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_surge_max=jnorm_surge_max, t_delay=t_delay
    )
    model.sensor_spring_normal_surge_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_SURGE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_surge_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_SURGE_RATE`` or ``/SENSOR/SPRING_TRANS_SURGE_RATE`` (M350): Spring element relative transverse / shear acceleration 7th rate-of-change (shear surge rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_SURGE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_surge_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_SURGE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_surge_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_surge_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseSurgeRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_surge_rates) + 1)
    sstsr = SensorSpringTransverseSurgeRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_surge_max=jtrans_surge_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_surge_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_SURGE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_surge_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_SURGE_RATE`` or ``/SENSOR/SPRING_TOT_SURGE_RATE`` (M351): Spring element relative 3D resultant total linear acceleration 7th rate-of-change (resultant total linear surge rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_SURGE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_surge_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_SURGE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_surge_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_surge_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalSurgeRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_surge_rates) + 1)
    sstsr = SensorSpringTotalSurgeRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_surge_max=jtot_surge_max, t_delay=t_delay
    )
    model.sensor_spring_total_surge_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_SURGE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_surge_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_SURGE_RATE`` or ``/SENSOR/SPRING_TORS_SURGE_RATE`` (M352): Spring element relative torsional angular acceleration 7th rate-of-change (torsional angular surge rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_SURGE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jrot_surge_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_SURGE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jrot_surge_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jrot_surge_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalSurgeRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_surge_rates) + 1)
    sstsr = SensorSpringTorsionalSurgeRate(
        id=s_id, title=title, spring_id=spring_id,
        jrot_surge_max=jrot_surge_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_surge_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_SURGE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_surge_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_SURGE_RATE`` or ``/SENSOR/SPRING_BEND_SURGE_RATE`` (M353): Spring element relative transverse bending angular acceleration 7th rate-of-change (bending angular surge rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_SURGE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_surge_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_SURGE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_surge_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_surge_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingSurgeRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_surge_rates) + 1)
    sstsr = SensorSpringBendingSurgeRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_surge_max=jbend_surge_max, t_delay=t_delay
    )
    model.sensor_spring_bending_surge_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_SURGE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_surge_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_SURGE_RATE`` or ``/SENSOR/SPRING_TOT_ANG_SURGE_RATE`` (M354): Spring element relative 3D resultant total angular acceleration 7th rate-of-change (resultant total angular surge rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_SURGE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jrot_tot_surge_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_SURGE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jrot_tot_surge_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jrot_tot_surge_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularSurgeRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_surge_rates) + 1)
    sstasr = SensorSpringTotalAngularSurgeRate(
        id=s_id, title=title, spring_id=spring_id,
        jrot_tot_surge_max=jrot_tot_surge_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_surge_rates[s_id] = sstasr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_SURGE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M355): Spring element relative normal / axial acceleration 8th rate-of-change (axial pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_pop_rates) + 1)
    ssnpr = SensorSpringNormalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_pop_max=jnorm_pop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_pop_rates[s_id] = ssnpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M356): Spring element relative transverse / shear acceleration 8th rate-of-change (shear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransversePopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_pop_rates) + 1)
    sstpr = SensorSpringTransversePopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_pop_max=jtrans_pop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_POP_RATE`` or ``/SENSOR/SPRING_TOT_POP_RATE`` (M357): Spring element relative 3D resultant total linear acceleration 8th rate-of-change (resultant total linear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_pop_rates) + 1)
    sstpr = SensorSpringTotalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_pop_max=jtot_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORS_POP_RATE`` (M358): Spring element relative torsional angular acceleration 8th rate-of-change (torsional angular pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_pop_rates) + 1)
    sstpr = SensorSpringTorsionalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_pop_max=jtors_pop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_POP_RATE`` or ``/SENSOR/SPRING_BEND_POP_RATE`` (M359): Spring element relative transverse bending angular acceleration 8th rate-of-change (bending angular pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_pop_rates) + 1)
    ssbpr = SensorSpringBendingPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_pop_max=jbend_pop_max, t_delay=t_delay
    )
    model.sensor_spring_bending_pop_rates[s_id] = ssbpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M360): Spring element relative 3D resultant total angular acceleration 8th rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_pop_rates) + 1)
    sstapr = SensorSpringTotalAngularPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_pop_max=jtot_ang_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_pop_rates[s_id] = sstapr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_NORM_CRACKLE_RATE`` (M361): Spring element relative normal / axial acceleration 9th rate-of-change (axial crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_crackle_rates) + 1)
    ssncr = SensorSpringNormalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_crk_max=jnorm_crk_max, t_delay=t_delay
    )
    model.sensor_spring_normal_crackle_rates[s_id] = ssncr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE`` or ``/SENSOR/SPRING_TRANS_CRACKLE_RATE`` (M362): Spring element relative transverse / shear acceleration 9th rate-of-change (shear crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_crackle_rates) + 1)
    sstcr = SensorSpringTransverseCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_crk_max=jtrans_crk_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_CRACKLE_RATE`` (M363): Spring element relative 3D resultant total linear acceleration 9th rate-of-change (resultant total linear crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_crackle_rates) + 1)
    sstcr = SensorSpringTotalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_crk_max=jtot_crk_max, t_delay=t_delay
    )
    model.sensor_spring_total_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TORS_CRACKLE_RATE`` (M364): Spring element relative torsional angular acceleration 9th rate-of-change (torsional angular crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_crackle_rates) + 1)
    sstcr = SensorSpringTorsionalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_crk_max=jtors_crk_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_CRACKLE_RATE`` or ``/SENSOR/SPRING_BEND_CRACKLE_RATE`` (M365): Spring element relative transverse bending angular acceleration 9th rate-of-change (bending angular crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_crackle_rates) + 1)
    sstcr = SensorSpringBendingCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_crk_max=jbend_crk_max, t_delay=t_delay
    )
    model.sensor_spring_bending_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE`` (M366): Spring element relative 3D resultant total angular acceleration 9th rate-of-change (resultant total angular crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_crackle_rates) + 1)
    sstcr = SensorSpringTotalAngularCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_crk_max=jtot_ang_crk_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_crackle_rates[s_id] = sstcr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_SNAP_RATE`` or ``/SENSOR/SPRING_NORM_SNAP_RATE`` (M367): Spring element relative normal / axial acceleration 10th rate-of-change (axial snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_snap_rates) + 1)
    ssnsr = SensorSpringNormalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_snp_max=jnorm_snp_max, t_delay=t_delay
    )
    model.sensor_spring_normal_snap_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_SNAP_RATE`` or ``/SENSOR/SPRING_TRANS_SNAP_RATE`` (M368): Spring element relative transverse / shear acceleration 10th rate-of-change (shear snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_snap_rates) + 1)
    sstsr = SensorSpringTransverseSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_snp_max=jtrans_snp_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_snap_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_SNAP_RATE`` (M369): Spring element relative 3D resultant total linear acceleration 10th rate-of-change (resultant total linear snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_snap_rates) + 1)
    sstsr = SensorSpringTotalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_snp_max=jtot_snp_max, t_delay=t_delay
    )
    model.sensor_spring_total_snap_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M370): Spring element relative torsional angular acceleration 10th rate-of-change (torsional angular snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_snap_rates) + 1)
    sstsr = SensorSpringTorsionalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_snp_max=jtors_snp_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_snap_rates[s_id] = sstsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_SNAP_RATE`` or ``/SENSOR/SPRING_BEND_SNAP_RATE`` (M371): Spring element relative transverse bending angular acceleration 10th rate-of-change (bending angular snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_snap_rates) + 1)
    ssbsr = SensorSpringBendingSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_snp_max=jbend_snp_max, t_delay=t_delay
    )
    model.sensor_spring_bending_snap_rates[s_id] = ssbsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_SNAP_RATE`` (M372): Spring element relative 3D resultant total angular acceleration 10th rate-of-change (resultant total angular snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_snap_rates) + 1)
    sstarsr = SensorSpringTotalAngularSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_snp_max=jtot_ang_snp_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_snap_rates[s_id] = sstarsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M373): Spring element relative normal / axial acceleration 11th rate-of-change (axial pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_pop_rates) + 1)
    ssnpr = SensorSpringNormalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_pop_max=jnorm_pop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_pop_rates[s_id] = ssnpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M374): Spring element relative transverse / shear acceleration 11th rate-of-change (shear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransversePopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_pop_rates) + 1)
    sstpr = SensorSpringTransversePopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_pop_max=jtrans_pop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_POP_RATE`` or ``/SENSOR/SPRING_TOT_POP_RATE`` (M375): Spring element relative 3D resultant total linear acceleration 11th rate-of-change (resultant total linear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_pop_rates) + 1)
    sstpr = SensorSpringTotalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_pop_max=jtot_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORS_POP_RATE`` (M376): Spring element relative torsional angular acceleration 11th rate-of-change (torsional pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_pop_rates) + 1)
    sstpr = SensorSpringTorsionalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_pop_max=jtors_pop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_POP_RATE`` or ``/SENSOR/SPRING_BEND_POP_RATE`` (M377): Spring element relative transverse bending angular acceleration 11th rate-of-change (bending angular pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_pop_rates) + 1)
    sstpr = SensorSpringBendingPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_pop_max=jbend_pop_max, t_delay=t_delay
    )
    model.sensor_spring_bending_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M378): Spring element relative 3D resultant total angular acceleration 11th rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtang_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtang_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtang_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_pop_rates) + 1)
    sstapr = SensorSpringTotalAngularPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtang_pop_max=jtang_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_pop_rates[s_id] = sstapr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_LOCK_RATE`` or ``/SENSOR/SPRING_NORM_LOCK_RATE`` (M379): Spring element relative normal / axial acceleration 12th rate-of-change (axial lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_lock_rates) + 1)
    ssnlr = SensorSpringNormalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_lock_max=jnorm_lock_max, t_delay=t_delay
    )
    model.sensor_spring_normal_lock_rates[s_id] = ssnlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_LOCK_RATE`` or ``/SENSOR/SPRING_TRANS_LOCK_RATE`` (M380): Spring element relative transverse / shear acceleration 12th rate-of-change (shear lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_lock_rates) + 1)
    sstlr = SensorSpringTransverseLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_lock_max=jtrans_lock_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_LOCK_RATE`` (M381): Spring element relative 3D resultant total linear acceleration 12th rate-of-change (resultant total lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_lock_rates) + 1)
    sstlr = SensorSpringTotalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_lock_max=jtot_lock_max, t_delay=t_delay
    )
    model.sensor_spring_total_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_LOCK_RATE`` or ``/SENSOR/SPRING_TORS_LOCK_RATE`` (M382): Spring element relative torsional angular acceleration 12th rate-of-change (torsional lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_lock_rates) + 1)
    sstlr = SensorSpringTorsionalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_lock_max=jtors_lock_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_LOCK_RATE`` or ``/SENSOR/SPRING_BEND_LOCK_RATE`` (M383): Spring element relative transverse bending angular acceleration 12th rate-of-change (bending angular lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_lock_rates) + 1)
    ssblr = SensorSpringBendingLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_lock_max=jbend_lock_max, t_delay=t_delay
    )
    model.sensor_spring_bending_lock_rates[s_id] = ssblr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_ANG_LOCK_RATE`` (M384): Spring element relative 3D resultant total angular acceleration 12th rate-of-change (resultant total angular lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_lock_rates) + 1)
    sstalr = SensorSpringTotalAngularLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_lock_max=jtot_ang_lock_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_lock_rates[s_id] = sstalr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_DROP_RATE`` or ``/SENSOR/SPRING_NORM_DROP_RATE`` (M385): Spring element relative normal / axial acceleration 13th rate-of-change (axial drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_drop_rates) + 1)
    ssndr = SensorSpringNormalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_drop_max=jnorm_drop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_drop_rates[s_id] = ssndr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_DROP_RATE`` or ``/SENSOR/SPRING_TRANS_DROP_RATE`` (M386): Spring element relative transverse / shear acceleration 13th rate-of-change (shear drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_drop_rates) + 1)
    sstdr = SensorSpringTransverseDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_drop_max=jtrans_drop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_DROP_RATE`` or ``/SENSOR/SPRING_TOT_DROP_RATE`` (M387): Spring element relative 3D resultant total linear acceleration 13th rate-of-change (resultant total drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_drop_rates) + 1)
    sstdr = SensorSpringTotalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_drop_max=jtot_drop_max, t_delay=t_delay
    )
    model.sensor_spring_total_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_DROP_RATE`` or ``/SENSOR/SPRING_TORS_DROP_RATE`` (M388): Spring element relative torsional angular acceleration 13th rate-of-change (torsional drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_drop_rates) + 1)
    sstdr = SensorSpringTorsionalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_drop_max=jtors_drop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_DROP_RATE`` or ``/SENSOR/SPRING_BEND_DROP_RATE`` (M389): Spring element relative transverse bending angular acceleration 13th rate-of-change (bending drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_drop_rates) + 1)
    sstdr = SensorSpringBendingDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_drop_max=jbend_drop_max, t_delay=t_delay
    )
    model.sensor_spring_bending_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_DROP_RATE`` (M390): Spring element relative 3D resultant total angular acceleration 13th rate-of-change (resultant total angular drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jang_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jang_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jang_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_drop_rates) + 1)
    sstdr = SensorSpringTotalAngularDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jang_drop_max=jang_drop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_drop_rates[s_id] = sstdr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_DROP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_shot_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_SHOT_RATE`` or ``/SENSOR/SPRING_NORM_SHOT_RATE`` (M391): Spring element relative normal / axial acceleration 14th rate-of-change (axial shot rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_SHOT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_shot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_SHOT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_shot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_shot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalShotRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_shot_rates) + 1)
    ssnsr = SensorSpringNormalShotRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_shot_max=jnorm_shot_max, t_delay=t_delay
    )
    model.sensor_spring_normal_shot_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_SHOT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_shot_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_SHOT_RATE`` or ``/SENSOR/SPRING_TRANS_SHOT_RATE`` (M392): Spring element relative transverse / shear acceleration 14th rate-of-change (shear shot rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_SHOT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_shot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_SHOT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_shot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_shot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseShotRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_shot_rates) + 1)
    ssnsr = SensorSpringTransverseShotRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_shot_max=jtrans_shot_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_shot_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_SHOT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_shot_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_SHOT_RATE`` or ``/SENSOR/SPRING_TOT_SHOT_RATE`` (M393): Spring element relative 3D resultant total linear acceleration 14th rate-of-change (resultant total shot rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_SHOT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_shot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_SHOT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_shot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_shot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalShotRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_shot_rates) + 1)
    ssnsr = SensorSpringTotalShotRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_shot_max=jtot_shot_max, t_delay=t_delay
    )
    model.sensor_spring_total_shot_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_SHOT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_shot_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_SHOT_RATE`` or ``/SENSOR/SPRING_TORS_SHOT_RATE`` (M394): Spring element relative torsional angular acceleration 14th rate-of-change (torsional shot rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_SHOT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_shot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_SHOT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_shot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_shot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalShotRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_shot_rates) + 1)
    ssnsr = SensorSpringTorsionalShotRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_shot_max=jtors_shot_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_shot_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_SHOT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_shot_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_SHOT_RATE`` or ``/SENSOR/SPRING_BEND_SHOT_RATE`` (M395): Spring element relative transverse bending angular acceleration 14th rate-of-change (bending shot rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_SHOT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_shot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_SHOT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_shot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_shot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingShotRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_shot_rates) + 1)
    ssnsr = SensorSpringBendingShotRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_shot_max=jbend_shot_max, t_delay=t_delay
    )
    model.sensor_spring_bending_shot_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_SHOT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_shot_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_SHOT_RATE`` or ``/SENSOR/SPRING_TOT_ANG_SHOT_RATE`` (M396): Spring element relative 3D resultant total angular acceleration 14th rate-of-change (resultant total angular shot rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_SHOT_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_shot_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_SHOT_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_shot_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_shot_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularShotRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_shot_rates) + 1)
    ssnsr = SensorSpringTotalAngularShotRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_shot_max=jtot_ang_shot_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_shot_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_SHOT_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_SNAP_RATE`` or ``/SENSOR/SPRING_NORM_SNAP_RATE`` (M397): Spring element relative normal / axial acceleration 14th rate-of-change (axial snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_snap_rates) + 1)
    ssnsr = SensorSpringNormalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_snp_max=jnorm_snp_max, t_delay=t_delay
    )
    model.sensor_spring_normal_snap_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_SNAP_RATE`` or ``/SENSOR/SPRING_TRANS_SNAP_RATE`` (M398): Spring element relative transverse / shear acceleration 14th rate-of-change (shear snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_snap_rates) + 1)
    ssnsr = SensorSpringTransverseSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_snp_max=jtrans_snp_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_snap_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_SNAP_RATE`` (M399): Spring element relative 3D resultant total linear acceleration 14th rate-of-change (resultant total snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_snap_rates) + 1)
    ssnsr = SensorSpringTotalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_snp_max=jtot_snp_max, t_delay=t_delay
    )
    model.sensor_spring_total_snap_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_SNAP_RATE`` or ``/SENSOR/SPRING_TORS_SNAP_RATE`` (M400): Spring element relative torsional angular acceleration 14th rate-of-change (torsional snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_snap_rates) + 1)
    ssnsr = SensorSpringTorsionalSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_snp_max=jtors_snp_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_snap_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_SNAP_RATE`` or ``/SENSOR/SPRING_BEND_SNAP_RATE`` (M401): Spring element relative transverse bending angular acceleration 14th rate-of-change (bending snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_snap_rates) + 1)
    ssnsr = SensorSpringBendingSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_snp_max=jbend_snp_max, t_delay=t_delay
    )
    model.sensor_spring_bending_snap_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_snap_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_SNAP_RATE`` (M402): Spring element relative 3D resultant total angular acceleration 14th rate-of-change (resultant total angular snap rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_snp_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_SNAP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_snp_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_snp_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularSnapRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_snap_rates) + 1)
    ssnsr = SensorSpringTotalAngularSnapRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_snp_max=jtot_ang_snp_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_snap_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_SNAP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_NORM_CRACKLE_RATE`` (M403): Spring element relative normal / axial acceleration 15th rate-of-change (axial crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_crackle_rates) + 1)
    ssnsr = SensorSpringNormalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_crk_max=jnorm_crk_max, t_delay=t_delay
    )
    model.sensor_spring_normal_crackle_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE`` or ``/SENSOR/SPRING_TRANS_CRACKLE_RATE`` (M404): Spring element relative transverse / shear acceleration 15th rate-of-change (shear crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_crackle_rates) + 1)
    ssnsr = SensorSpringTransverseCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_crk_max=jtrans_crk_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_crackle_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_CRACKLE_RATE`` (M405): Spring element relative 3D resultant total linear acceleration 15th rate-of-change (resultant total crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_crackle_rates) + 1)
    ssnsr = SensorSpringTotalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_crk_max=jtot_crk_max, t_delay=t_delay
    )
    model.sensor_spring_total_crackle_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE`` or ``/SENSOR/SPRING_TORS_CRACKLE_RATE`` (M406): Spring element relative torsional angular acceleration 15th rate-of-change (torsional crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_crackle_rates) + 1)
    ssnsr = SensorSpringTorsionalCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_crk_max=jtors_crk_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_crackle_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_CRACKLE_RATE`` or ``/SENSOR/SPRING_BEND_CRACKLE_RATE`` (M407): Spring element relative transverse bending angular acceleration 15th rate-of-change (bending crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_crackle_rates) + 1)
    ssnsr = SensorSpringBendingCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_crk_max=jbend_crk_max, t_delay=t_delay
    )
    model.sensor_spring_bending_crackle_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_crackle_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE`` or ``/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE`` (M408): Spring element relative 3D resultant total angular acceleration 15th rate-of-change (resultant total angular crackle rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtota_crk_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_CRACKLE_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtota_crk_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtota_crk_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularCrackleRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_crackle_rates) + 1)
    ssnsr = SensorSpringTotalAngularCrackleRate(
        id=s_id, title=title, spring_id=spring_id,
        jtota_crk_max=jtota_crk_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_crackle_rates[s_id] = ssnsr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_CRACKLE_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_POP_RATE`` or ``/SENSOR/SPRING_NORM_POP_RATE`` (M409): Spring element relative normal / axial acceleration 16th rate-of-change (axial pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_pop_rates) + 1)
    ssnpr = SensorSpringNormalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_pop_max=jnorm_pop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_pop_rates[s_id] = ssnpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_POP_RATE`` or ``/SENSOR/SPRING_TRANS_POP_RATE`` (M410): Spring element relative transverse / shear acceleration 16th rate-of-change (shear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransversePopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_pop_rates) + 1)
    sstpr = SensorSpringTransversePopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_pop_max=jtrans_pop_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_POP_RATE`` or ``/SENSOR/SPRING_TOT_POP_RATE`` (M411): Spring element relative 3D resultant total linear acceleration 16th rate-of-change (resultant total linear pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_pop_rates) + 1)
    sstpr = SensorSpringTotalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_pop_max=jtot_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_POP_RATE`` or ``/SENSOR/SPRING_TORS_POP_RATE`` (M412): Spring element relative torsional angular acceleration 16th rate-of-change (torsional pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtor_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtor_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtor_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_pop_rates) + 1)
    sstpr = SensorSpringTorsionalPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtor_pop_max=jtor_pop_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_pop_rates[s_id] = sstpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_POP_RATE`` or ``/SENSOR/SPRING_BEND_POP_RATE`` (M413): Spring element relative transverse bending angular acceleration 16th rate-of-change (bending pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_pop_rates) + 1)
    ssbpr = SensorSpringBendingPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_pop_max=jbend_pop_max, t_delay=t_delay
    )
    model.sensor_spring_bending_pop_rates[s_id] = ssbpr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_pop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE`` or ``/SENSOR/SPRING_TOT_ANG_POP_RATE`` (M414): Spring element relative 3D resultant total angular acceleration 16th rate-of-change (resultant total angular pop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_pop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_POP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_pop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_pop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularPopRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_pop_rates) + 1)
    sstapr = SensorSpringTotalAngularPopRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_pop_max=jtot_ang_pop_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_pop_rates[s_id] = sstapr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_POP_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_LOCK_RATE`` or ``/SENSOR/SPRING_NORM_LOCK_RATE`` (M415): Spring element relative normal / axial acceleration 17th rate-of-change (axial lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_lock_rates) + 1)
    ssnlr = SensorSpringNormalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_lock_max=jnorm_lock_max, t_delay=t_delay
    )
    model.sensor_spring_normal_lock_rates[s_id] = ssnlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_transverse_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TRANSVERSE_LOCK_RATE`` or ``/SENSOR/SPRING_TRANS_LOCK_RATE`` (M416): Spring element relative transverse / shear acceleration 17th rate-of-change (shear lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtrans_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TRANSVERSE_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtrans_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtrans_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTransverseLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_transverse_lock_rates) + 1)
    sstlr = SensorSpringTransverseLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtrans_lock_max=jtrans_lock_max, t_delay=t_delay
    )
    model.sensor_spring_transverse_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TRANSVERSE_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_LOCK_RATE`` (M417): Spring element relative 3D resultant total linear acceleration 17th rate-of-change (resultant total linear lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_lock_rates) + 1)
    sstlr = SensorSpringTotalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_lock_max=jtot_lock_max, t_delay=t_delay
    )
    model.sensor_spring_total_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_torsional_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TORSIONAL_LOCK_RATE`` or ``/SENSOR/SPRING_TORS_LOCK_RATE`` (M418): Spring element relative torsional angular acceleration 17th rate-of-change (torsional angular lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TORSIONAL_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtors_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TORSIONAL_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtors_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtors_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTorsionalLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_torsional_lock_rates) + 1)
    sstlr = SensorSpringTorsionalLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtors_lock_max=jtors_lock_max, t_delay=t_delay
    )
    model.sensor_spring_torsional_lock_rates[s_id] = sstlr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TORSIONAL_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_bending_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_BENDING_LOCK_RATE`` or ``/SENSOR/SPRING_BEND_LOCK_RATE`` (M419): Spring element relative transverse bending angular acceleration 17th rate-of-change (bending angular lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_BENDING_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jbend_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_BENDING_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jbend_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jbend_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringBendingLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_bending_lock_rates) + 1)
    ssblr = SensorSpringBendingLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jbend_lock_max=jbend_lock_max, t_delay=t_delay
    )
    model.sensor_spring_bending_lock_rates[s_id] = ssblr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_BENDING_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_total_angular_lock_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE`` or ``/SENSOR/SPRING_TOT_ANG_LOCK_RATE`` (M420): Spring element relative 3D resultant total angular acceleration 17th rate-of-change (resultant total angular lock rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jtot_ang_lock_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_TOTAL_ANGULAR_LOCK_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jtot_ang_lock_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jtot_ang_lock_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringTotalAngularLockRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_total_angular_lock_rates) + 1)
    sstalr = SensorSpringTotalAngularLockRate(
        id=s_id, title=title, spring_id=spring_id,
        jtot_ang_lock_max=jtot_ang_lock_max, t_delay=t_delay
    )
    model.sensor_spring_total_angular_lock_rates[s_id] = sstalr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_TOTAL_ANGULAR_LOCK_RATE", tdelay=t_delay
    ))




def read_sensor_spring_normal_drop_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/SPRING_NORMAL_DROP_RATE`` or ``/SENSOR/SPRING_NORM_DROP_RATE`` (M421): Spring element relative normal / axial acceleration 18th rate-of-change (axial drop rate) magnitude threshold sensor."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SENSOR/SPRING_NORMAL_DROP_RATE/{block.user_id}: missing data card", block.source)
        return

    spring_id, jnorm_drop_max, t_delay = 0, 1e30, 0.0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("SENSOR_SPRING_NORMAL_DROP_RATE_1")
        spring_id = _ival(f[0], 0) if len(f) > 0 else 0
        jnorm_drop_max = _fval(f[1], 1e30) if len(f) > 1 else 1e30
        t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        spring_id = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        jnorm_drop_max = float(toks[1].rstrip(',')) if len(toks) > 1 else 1e30
        t_delay = float(toks[2].rstrip(',')) if len(toks) > 2 else 0.0

    from ...model.entities import SensorSpringNormalDropRate, Sensor
    s_id = block.user_id or (len(model.sensor_spring_normal_drop_rates) + 1)
    ssndr = SensorSpringNormalDropRate(
        id=s_id, title=title, spring_id=spring_id,
        jnorm_drop_max=jnorm_drop_max, t_delay=t_delay
    )
    model.sensor_spring_normal_drop_rates[s_id] = ssndr
    model.sensors.append(Sensor(
        id=s_id, kind="SPRING_NORMAL_DROP_RATE", tdelay=t_delay
    ))





























































































def read_h3d(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/H3D`` (M198): HyperView H3D file output format request."""
    # Stored for output configuration
    pass
