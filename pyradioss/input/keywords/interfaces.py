# -*- coding: utf-8 -*-
"""
Contact interface readers - /KEYWORD blocks -> Model.

/INTER/** including the LAGMUL variants, and /DEF_INTER defaults.

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






def read_inter(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTER/TYPE7|TYPE2|TYPE11/inter_ID`` — the three contact types.

    Fortran: ``starter/source/interfaces/int07|02|11/hm_read_inter*.F``.
    The port keeps a compact card layout (a strict subset of the Radioss
    fields, in the Radioss order where they exist):

    ``/INTER/TYPE7`` (penalty node-to-surface)::

        card 1:  title
        card 2:  grnod_ID  surf_ID  Istf  Igap  [sens_ID]  [Ifric]  [Ifiltr]
        card 3:  Stfac     Fric     Gapmin  Gapmax  [Xfreq]   (all optional)
        card 4:  C1  C2  C3  C4  C5  C6         (only read when Ifric > 0)

      sens_ID (M6): the interface stays inactive (no forces, no time-step
      claim) until /SENSOR sens_ID fires.

      grnod_ID = 0 → *self-impact*: the secondary nodes default to the
      nodes of the main surface itself (Radioss single-surface input).
      Istf 0..5 and Igap 0/1 as documented on
      :class:`pyradioss.model.entities.Interface`. For Istf=1, Stfac is
      the constant penalty stiffness itself (force/length); otherwise it
      scales the element-based stiffness (default 1.0).

      Ifric/Ifiltr/Xfreq/C1..C6 (M15 — the friction MODELS of
      ``hm_read_inter_type07.F``, mirrored field for field against the
      SOURCE): Ifric = MFROT 1..4 selects the mu(p, v) law (C1..C5 read
      for Ifric > 0, C6 for Ifric > 1 — the original's optional card 8);
      Ifiltr = IFQ 1/2/3 turns the tangential-force first-order filter
      on, with the coefficient derived from Xfreq exactly as the reader
      does (1: Xfreq itself, must be in [0, 1]; 2: 2*pi/Xfreq, Xfreq a
      period in cycles; 3: 2*pi*Xfreq, Xfreq a cutoff frequency —
      per-cycle alpha = XFILTR*dt). Xfreq = 0 switches the filter OFF
      whatever Ifiltr says — exactly the reference (``IF (ALPHA==0.)
      IFQ = 0`` in hm_read_inter_type07.F). IFQ >= 10 (the MODFR = 2
      incremental stiffness formulation) is refused loudly — deferred
      (see contact/friction.py; the implicit solver's return mapping IS
      that formulation). Laws and filter live in contact/friction.py.

      TYPE7 also accepts the REAL fixed-format layout (cfg
      ``inter_type7.cfg`` radioss2020+ / ``hm_read_inter_type07.F``),
      detected by its card count — 6+ data cards where the compact
      layout above has at most 3::

        card 2:  grnod_ID surf_ID Istf Ithe Igap __ Ibag Idel Icurv Iadm
        card 3:  Fscale_gap  Gap_max  Fpenmax  __  Itied
        card 4:  Stmin  Stmax  %mesh_size  dtmin  Irem_gap  Irem_i2
        [Icurv 1/2 only: node_ID1 node_ID2]
        card 5:  Stfac  Fric  Gapmin  Tstart  Tstop
        card 6:  IBC  __  Inacti  VisS  VisF  Bumult
        card 7:  Ifric Ifiltr Xfreq Iform sens_ID fct_IDF AscaleF fric_ID
        [Ifric > 0 only: C1..C5]  [Ifric > 1 only: C6]

      Real fields with no port equivalent (Ithe/Ibag/Idel/Icurv/Iadm,
      Fscale_gap/Fpenmax/Itied, Stmin/Stmax/%mesh_size/dtmin/Irem_*,
      Tstart/Tstop, IBC/Inacti/VisS/VisF/Bumult, fct_IDF/AscaleF/
      fric_ID) are accepted and, when set to a non-default value,
      reported in ONE warning. Iform = 2 (the incremental tangential
      formulation) is an ERROR when friction is actually active
      (Fric != 0 or Ifric > 0 — same machinery as the IFQ >= 10
      refusal); with no friction it is inert and only warned about.

    ``/INTER/TYPE2`` (tied, kinematic)::

        card 1:  title
        card 2:  grnod_ID  surf_ID  dsearch

      Every secondary node within ``dsearch`` of the main surface
      (0 → auto: twice the main segment size) is glued to its closest
      segment for the whole run. Not-found nodes are left free (warning).

    ``/INTER/TYPE11`` (penalty edge-to-edge)::

        card 1:  title
        card 2:  line_ID1  line_ID2  Istf  Igap  [sens_ID]  [Ifric]  [Ifiltr]
        card 3:  Stfac     Fric      Gapmin  Gapmax  [Xfreq]
        card 4:  C1  C2  C3  C4  C5  C6         (only read when Ifric > 0)

      The TYPE11 friction-model fields are a documented port EXTENSION:
      the ORIGINAL TYPE11 card has none and its engine never evaluates
      MFROT (i11mainf.F forces MFROT = 0 — checked; see
      contact/friction.py for the edge-pair pressure definition).

    Options NOT ported (accepted Radioss fields ignored elsewhere in the
    line): Inacti, Tstart/Tstop, thermal contact, the IFQ >= 10 / MODFR=2
    incremental tangential formulation (refused), Igap 2/3 mesh-size gap
    scaling.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind == "LAGDT" and len(block.parts) > 2:
        kind = block.parts[2].upper()
    if kind in ("GUIDED_CABLE", "CABLE", "TYPE26", "26"):
        read_guided_cable(block, model, log)
        return
    if kind in ("SUB", "SUBINTER"):
        read_subinter(block, model, log)
        return
    if kind == "HERTZ":
        subtype = block.parts[2].upper() if len(block.parts) > 2 else ""
        if subtype not in ("TYPE17", "17"):
            log.warning(f"/INTER/HERTZ/{subtype} not ported", block.source)
            return
        title, cards = _title_and_data(block)
        if not cards:
            log.error(f"/INTER/HERTZ/{subtype}/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f0 = cards[0].cut("INTER_HERTZ_17_1")
            grbric_id1 = _ival(f0[0]) if len(f0) > 0 else 0
            grbric_id2 = _ival(f0[1]) if len(f0) > 1 else 0
            fric = 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_HERTZ_17_2")
                fric = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        else:
            t0 = cards[0].tokens()
            grbric_id1 = int(float(t0[0])) if len(t0) > 0 else 0
            grbric_id2 = int(float(t0[1])) if len(t0) > 1 else 0
            fric = 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                fric = float(t1[0]) if len(t1) > 0 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=17, grbric_id1=grbric_id1, grbric_id2=grbric_id2,
            fric=fric, hertz=True, title=title
        ))
        return
    if kind == "LAGMUL":
        subtype = block.parts[2].upper() if len(block.parts) > 2 else ""
        if subtype not in ("TYPE2", "TYPE7", "TYPE11", "TYPE16", "TYPE17", "SPOTWELD", "SURF", "PART", "TIED", "BEAM", "EDGE"):
            log.warning(f"/INTER/LAGMUL/{subtype} not ported", block.source)
            return
        title, cards = _title_and_data(block)
        if not cards:
            log.error(f"/INTER/LAGMUL/{subtype}/{block.user_id}: missing data card", block.source)
            return

        if subtype in ("TYPE16", "SPOTWELD"):
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10])
                grnod_id, grbric_id = _ival(f[0]), _ival(f[1])
                itied = 0
                if len(cards) > 1:
                    f2 = _fixed_vals(cards[1], [20, 10])
                    itied = _ival(f2[1])
            else:
                toks = cards[0].tokens()
                grnod_id = int(toks[0]) if len(toks) > 0 else 0
                grbric_id = int(toks[1]) if len(toks) > 1 else 0
                itied = 0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    itied = int(t2[1]) if len(t2) > 1 else (int(t2[0]) if len(t2) > 0 else 0)
            model.interfaces.append(Interface(
                id=block.user_id, type=16, grnod_id=grnod_id, grbric_id1=grbric_id,
                itied=itied, lagmul=True, title=title))
            return
        elif subtype == "TYPE17":
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10])
                grbric_id1, grbric_id2 = _ival(f[0]), _ival(f[1])
                itied = 0
                if len(cards) > 1:
                    f2 = _fixed_vals(cards[1], [20, 10])
                    itied = _ival(f2[1])
            else:
                toks = cards[0].tokens()
                grbric_id1 = int(toks[0]) if len(toks) > 0 else 0
                grbric_id2 = int(toks[1]) if len(toks) > 1 else 0
                itied = 0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    itied = int(t2[0]) if len(t2) > 0 else 0
            model.interfaces.append(Interface(
                id=block.user_id, type=17, grbric_id1=grbric_id1, grbric_id2=grbric_id2,
                itied=itied, lagmul=True, title=title))
            return
        elif subtype in ("TYPE2", "PART", "TIED"):
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10, 30, 10, 20, 20])
                grnod_id, surf_id = _ival(f[0]), _ival(f[1])
                dsearch = _fval(f[5]) if len(f) > 5 else 0.0
            else:
                toks = cards[0].tokens()
                grnod_id = int(toks[0]) if len(toks) > 0 else 0
                surf_id = int(toks[1]) if len(toks) > 1 else 0
                dsearch = float(toks[3]) if len(toks) > 3 else (float(toks[2]) if len(toks) == 3 else 0.0)
            model.interfaces.append(Interface(
                id=block.user_id, type=2, grnod_id=grnod_id, surf_id=surf_id,
                dsearch=dsearch, lagmul=True, title=title))
            return
        elif subtype in ("TYPE7", "SURF"):
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10, 30, 10])
                grnod_id, surf_id = _ival(f[0]), _ival(f[1])
                gap_min = 0.0
                if len(cards) >= 2:
                    f4 = _fixed_vals(cards[1], [40, 20])
                    gap_min = _fval(f4[1])
            else:
                toks = cards[0].tokens()
                grnod_id = int(toks[0]) if len(toks) > 0 else 0
                surf_id = int(toks[1]) if len(toks) > 1 else 0
                gap_min = 0.0
                if len(cards) >= 2:
                    t4 = cards[1].tokens()
                    gap_min = float(t4[-1]) if len(t4) > 1 else (float(t4[0]) if len(t4) > 0 else 0.0)
            model.interfaces.append(Interface(
                id=block.user_id, type=7, grnod_id=grnod_id, surf_id=surf_id,
                gap=gap_min, lagmul=True, title=title))
            return
        elif subtype in ("TYPE11", "BEAM", "EDGE"):
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10])
                line_id1, line_id2 = _ival(f[0]), _ival(f[1])
            else:
                toks = cards[0].tokens()
                line_id1 = int(toks[0]) if len(toks) > 0 else 0
                line_id2 = int(toks[1]) if len(toks) > 1 else 0
            model.interfaces.append(Interface(
                id=block.user_id, type=11, line_id1=line_id1, line_id2=line_id2,
                lagmul=True, title=title))
    if kind in ("SUB_SURF", "SUB-SURF", "SUBSURF"):
        kind = "TYPE21"

    if kind not in ("TYPE1", "TYPE2", "TYPE3", "TYPE5", "TYPE6", "TYPE7", "TYPE8", "TYPE9", "TYPE10", "TYPE11",
                    "TYPE12", "TYPE14", "TYPE15", "TYPE16", "TYPE17", "TYPE18", "TYPE19", "TYPE20", "TYPE21", "TYPE22",
                    "TYPE23", "TYPE24", "TYPE25", "SUB", "GUIDED_CABLE"):
        log.warning(f"/INTER/{kind} not ported", block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INTER/{kind}/{block.user_id}: missing data card",
                  block.source)
        return

    if kind == "TYPE8":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE8_1")
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
            dbead_force, tstart, tstop = 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE8_2")
                dbead_force = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
            dbead_force, tstart, tstop = 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                dbead_force = float(t1[0]) if len(t1) > 0 else 0.0
                tstart = float(t1[1]) if len(t1) > 1 else 0.0
                tstop = float(t1[2]) if len(t1) > 2 else 1.0e30

        model.interfaces.append(Interface(
            id=block.user_id, type=8, grnod_id=grnod_id, surf_id=surf_id,
            stfac=dbead_force, tstart=tstart, tstop=tstop, title=title,
        ))
        return

    if kind == "TYPE18":
        grnod_id = 0
        surf_id = 0
        grbric_id = 0
        igap = 0
        ibag = 0
        idel18 = 0
        iauto = 0
        stfac = 1.0
        vref = 0.0
        gap = 0.0
        tstart = 0.0
        tstop = 1.0e30
        istf = 0
        multimp = 4
        stiff_dc = 0.0
        sort_fact = 0.2

        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                raw0 = cards[0].raw
                grnod_id = _ival(raw0[:10])
                surf_id = _ival(raw0[10:20])
                grbric_id = _ival(raw0[20:30])
                istf = grbric_id
                if len(raw0) > 30:
                    igap = _ival(raw0[30:50])
                if len(raw0) > 50:
                    ibag = _ival(raw0[50:70])
                if len(raw0) > 70:
                    idel18 = _ival(raw0[70:80])
                if len(raw0) > 80:
                    iauto = _ival(raw0[80:100])
            if len(cards) >= 2 and not cards[1].is_blank:
                raw1 = cards[1].raw
                stfac = _fval(raw1[:20], 1.0)
                vref = _fval(raw1[20:40], 0.0) if len(raw1) > 20 else 0.0
                gap = _fval(raw1[40:60], 0.0) if len(raw1) > 40 else 0.0
                tstart = _fval(raw1[60:80], 0.0) if len(raw1) > 60 else 0.0
                tstop = _fval(raw1[80:100], 1.0e30) if len(raw1) > 80 else 1.0e30
            if len(cards) >= 3 and not cards[2].is_blank:
                raw2 = cards[2].raw
                stiff_dc = _fval(raw2[40:60], 0.0) if len(raw2) > 40 else 0.0
                sf = _fval(raw2[80:100], 0.0) if len(raw2) > 80 else 0.0
                if sf == 0.0 and len(raw2) > 60:
                    sf = _fval(raw2[60:80], 0.0)
                sort_fact = sf if sf != 0.0 else 0.2
        else:
            if len(cards) >= 1 and not cards[0].is_blank:
                t0 = cards[0].tokens()
                grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
                surf_id = int(float(t0[1])) if len(t0) > 1 else 0
                grbric_id = int(float(t0[2])) if len(t0) > 2 else 0
                istf = grbric_id
                igap = int(float(t0[3])) if len(t0) > 3 else 0
                ibag = int(float(t0[4])) if len(t0) > 4 else 0
                idel18 = int(float(t0[5])) if len(t0) > 5 else 0
                iauto = int(float(t0[6])) if len(t0) > 6 else 0
            if len(cards) >= 2 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stfac = float(t1[0]) if len(t1) > 0 else 1.0
                vref = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                tstart = float(t1[3]) if len(t1) > 3 else 0.0
                tstop = float(t1[4]) if len(t1) > 4 else 1.0e30
            if len(cards) >= 3 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                stiff_dc = float(t2[0]) if len(t2) > 0 else 0.0
                sort_fact = float(t2[1]) if len(t2) > 1 else 0.2

        from ...model.entities import InterType18
        inter18 = InterType18(
            id=block.user_id, title=title, grnod_id=grnod_id, surf_id=surf_id, grbric_id=grbric_id,
            istf=istf, igap=igap, multimp=multimp, ibag=ibag, idel18=idel18, iauto=iauto,
            stfac=stfac, vref=vref, gap=gap, tstart=tstart, tstop=tstop, stiff_dc=stiff_dc, sort_fact=sort_fact,
            params={"grnod_id": grnod_id, "surf_id": surf_id, "grbric_id": grbric_id, "igap": igap, "ibag": ibag, "idel18": idel18, "iauto": iauto, "stfac": stfac, "vref": vref, "gap": gap, "tstart": tstart, "tstop": tstop, "stiff_dc": stiff_dc, "sort_fact": sort_fact}
        )
        model.inter_type18s[block.user_id] = inter18
        model.inter_type18s[block.user_id] = inter18
        model.interfaces.append(Interface(
            id=block.user_id, type=18, grnod_id=grnod_id, surf_id=surf_id,
            grbric_id1=grbric_id, igap=igap, ibag=ibag, idel=idel18, idel18=idel18,
            stfac=stfac, gap=gap, tstart=tstart, tstop=tstop,
            stiff_dc=stiff_dc, sort_fact=sort_fact, title=title,
        ))
        return

    if kind == "TYPE25":

        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE25_0") if len(cards) > 0 else []
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            istf = _ival(f0[2]) if len(f0) > 2 else 0
            igap = _ival(f0[4]) if len(f0) > 4 else 0
            idel = _ival(f0[6]) if len(f0) > 6 else 0

            grnod_id, prmesh_size, gap1, gap2 = 0, 0.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE25_1")
                grnod_id = _ival(f1[0]) if len(f1) > 0 else 0
                prmesh_size = _fval(f1[2]) if len(f1) > 2 else 0.0
                gap1 = _fval(f1[3]) if len(f1) > 3 else 0.0
                gap2 = _fval(f1[4]) if len(f1) > 4 else 0.0

            stmin, stmax, igap2 = 0.0, 0.0, 0
            if len(cards) > 2 and not cards[2].is_blank:
                f2 = cards[2].cut("INTER_TYPE25_2")
                stmin = _fval(f2[0]) if len(f2) > 0 else 0.0
                stmax = _fval(f2[1]) if len(f2) > 1 else 0.0
                igap2 = _ival(f2[2]) if len(f2) > 2 else 0

            stfac, fric, tstart, tstop = 1.0, 0.0, 0.0, 1.0e30
            if len(cards) > 3 and not cards[3].is_blank:
                f3 = cards[3].cut("INTER_TYPE25_3")
                stfac = _fval(f3[0], 1.0) if len(f3) > 0 else 1.0
                fric = _fval(f3[1]) if len(f3) > 1 else 0.0
                tstart = _fval(f3[3]) if len(f3) > 3 else 0.0
                tstop = _fval(f3[4], 1.0e30) if len(f3) > 4 else 1.0e30

            deact_x, deact_y, deact_z, inactiv, stiff_dc = 0, 0, 0, 0, 0.0
            if len(cards) > 4 and not cards[4].is_blank:
                f4 = cards[4].cut("INTER_TYPE25_4")
                deact_x = _ival(f4[1]) if len(f4) > 1 else 0
                deact_y = _ival(f4[2]) if len(f4) > 2 else 0
                deact_z = _ival(f4[3]) if len(f4) > 3 else 0
                inactiv = _ival(f4[5]) if len(f4) > 5 else 0
                stiff_dc = _fval(f4[6]) if len(f4) > 6 else 0.0

            ifric, ifiltr, xfreq, isensor, fric_id = 0, 0, 0.0, 0, 0
            if len(cards) > 5 and not cards[5].is_blank:
                f5 = cards[5].cut("INTER_TYPE25_5")
                ifric = _ival(f5[0]) if len(f5) > 0 else 0
                ifiltr = _ival(f5[1]) if len(f5) > 1 else 0
                xfreq = _fval(f5[2]) if len(f5) > 2 else 0.0
                isensor = _ival(f5[4]) if len(f5) > 4 else 0
                fric_id = _ival(f5[6]) if len(f5) > 6 else 0

            c1, c2, c3, c4, c5 = 0.0, 0.0, 0.0, 0.0, 0.0
            if len(cards) > 6 and not cards[6].is_blank:
                f6 = cards[6].cut("INTER_TYPE25_6")
                c1 = _fval(f6[0]) if len(f6) > 0 else 0.0
                c2 = _fval(f6[1]) if len(f6) > 1 else 0.0
                c3 = _fval(f6[2]) if len(f6) > 2 else 0.0
                c4 = _fval(f6[3]) if len(f6) > 3 else 0.0
                c5 = _fval(f6[4]) if len(f6) > 4 else 0.0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            istf = int(float(t0[2])) if len(t0) > 2 else 0
            if len(t0) >= 6:
                igap = int(float(t0[4]))
                idel = int(float(t0[5]))
            elif len(t0) == 5:
                igap = int(float(t0[3]))
                idel = int(float(t0[4]))
            else:
                igap = int(float(t0[3])) if len(t0) > 3 else 0
                idel = int(float(t0[4])) if len(t0) > 4 else 0

            grnod_id, prmesh_size, gap1, gap2 = 0, 0.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                grnod_id = int(float(t1[0])) if len(t1) > 0 else 0
                prmesh_size = float(t1[1]) if len(t1) > 1 else 0.0
                gap1 = float(t1[2]) if len(t1) > 2 else 0.0
                gap2 = float(t1[3]) if len(t1) > 3 else 0.0

            stmin, stmax, igap2 = 0.0, 0.0, 0
            if len(cards) > 2 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                stmin = float(t2[0]) if len(t2) > 0 else 0.0
                stmax = float(t2[1]) if len(t2) > 1 else 0.0
                igap2 = int(float(t2[2])) if len(t2) > 2 else 0

            stfac, fric, tstart, tstop = 1.0, 0.0, 0.0, 1.0e30
            if len(cards) > 3 and not cards[3].is_blank:
                t3 = cards[3].tokens()
                stfac = float(t3[0]) if len(t3) > 0 else 1.0
                fric = float(t3[1]) if len(t3) > 1 else 0.0
                tstart = float(t3[2]) if len(t3) > 2 else 0.0
                tstop = float(t3[3]) if len(t3) > 3 else 1.0e30

            deact_x, deact_y, deact_z, inactiv, stiff_dc = 0, 0, 0, 0, 0.0
            if len(cards) > 4 and not cards[4].is_blank:
                t4 = cards[4].tokens()
                if len(t4) == 3:
                    tok0 = t4[0]
                    deact_x = int(tok0[0]) if len(tok0) > 0 and tok0[0].isdigit() else 0
                    deact_y = int(tok0[1]) if len(tok0) > 1 and tok0[1].isdigit() else 0
                    deact_z = int(tok0[2]) if len(tok0) > 2 and tok0[2].isdigit() else 0
                    inactiv = int(float(t4[1]))
                    stiff_dc = float(t4[2])
                else:
                    deact_x = int(float(t4[0])) if len(t4) > 0 else 0
                    deact_y = int(float(t4[1])) if len(t4) > 1 else 0
                    deact_z = int(float(t4[2])) if len(t4) > 2 else 0
                    inactiv = int(float(t4[3])) if len(t4) > 3 else 0
                    stiff_dc = float(t4[4]) if len(t4) > 4 else 0.0

            ifric, ifiltr, xfreq, isensor, fric_id = 0, 0, 0.0, 0, 0
            if len(cards) > 5 and not cards[5].is_blank:
                t5 = cards[5].tokens()
                ifric = int(float(t5[0])) if len(t5) > 0 else 0
                ifiltr = int(float(t5[1])) if len(t5) > 1 else 0
                xfreq = float(t5[2]) if len(t5) > 2 else 0.0
                isensor = int(float(t5[3])) if len(t5) > 3 else 0
                fric_id = int(float(t5[4])) if len(t5) > 4 else 0

            c1, c2, c3, c4, c5 = 0.0, 0.0, 0.0, 0.0, 0.0
            if len(cards) > 6 and not cards[6].is_blank:
                t6 = cards[6].tokens()
                c1 = float(t6[0]) if len(t6) > 0 else 0.0
                c2 = float(t6[1]) if len(t6) > 1 else 0.0
                c3 = float(t6[2]) if len(t6) > 2 else 0.0
                c4 = float(t6[3]) if len(t6) > 3 else 0.0
                c5 = float(t6[4]) if len(t6) > 4 else 0.0

        model.interfaces.append(Interface(
            id=block.user_id, type=25, surf_id=surf1, surf_id1=surf2, grnod_id=grnod_id,
            istf=istf, igap=igap, idel=idel, stmin=stmin, stmax=stmax, stfac=stfac, fric=fric,
            gap=gap1, gap_max=gap2, tstart=tstart, tstop=tstop,
            inactiv=inactiv, stiff_dc=stiff_dc, ifric=ifric, ifiltr=ifiltr, xfreq=xfreq,
            isensor=isensor, fric_id=fric_id, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
            title=title
        ))
        from ...model.entities import InterType25
        model.inter_type25s[block.user_id] = InterType25(
            id=block.user_id, title=title, grnd_id=grnod_id, surf_id=surf1,
            fn_max=stmax, ft_max=stmin, wn=0.0, wt=0.0, gap=gap1, stiff=stfac,
            ifric=ifric, fric=fric
        )
        return

    if kind == "TYPE19":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE19_0") if len(cards) > 0 else []
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
            istf = _ival(f0[2]) if len(f0) > 2 else 0
            igap = _ival(f0[4]) if len(f0) > 4 else 0
            iedge = _ival(f0[5]) if len(f0) > 5 else 0
            ibag = _ival(f0[6]) if len(f0) > 6 else 0
            idel = _ival(f0[7]) if len(f0) > 7 else 0
            icurv = _ival(f0[8]) if len(f0) > 8 else 0

            gap_scale = 1.0
            gap_max = 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE19_1")
                gap_scale = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                gap_max = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

            stmin, stmax = 0.0, 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                f2 = cards[2].cut("INTER_TYPE19_2")
                stmin = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                stmax = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0

            n1, n2 = 0, 0
            if len(cards) > 3 and not cards[3].is_blank:
                f3 = cards[3].cut("INTER_TYPE19_3")
                n1 = _ival(f3[0]) if len(f3) > 0 else 0
                n2 = _ival(f3[1]) if len(f3) > 1 else 0

            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 4 and not cards[4].is_blank:
                f4 = cards[4].cut("INTER_TYPE19_4")
                stfac = _fval(f4[0], 1.0) if len(f4) > 0 else 1.0
                fric = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
                gap = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0
                tstart = _fval(f4[3], 0.0) if len(f4) > 3 else 0.0
                tstop = _fval(f4[4], 1.0e30) if len(f4) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
            istf = int(float(t0[2])) if len(t0) > 2 else 0
            igap = int(float(t0[3])) if len(t0) > 3 else 0
            iedge = int(float(t0[4])) if len(t0) > 4 else 0
            ibag = int(float(t0[5])) if len(t0) > 5 else 0
            idel = int(float(t0[6])) if len(t0) > 6 else 0
            icurv = int(float(t0[7])) if len(t0) > 7 else 0

            gap_scale, gap_max = 1.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                gap_scale = float(t1[0]) if len(t1) > 0 else 1.0
                gap_max = float(t1[1]) if len(t1) > 1 else 0.0

            stmin, stmax = 0.0, 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                stmin = float(t2[0]) if len(t2) > 0 else 0.0
                stmax = float(t2[1]) if len(t2) > 1 else 0.0

            n1, n2 = 0, 0
            if len(cards) > 3 and not cards[3].is_blank:
                t3 = cards[3].tokens()
                n1 = int(float(t3[0])) if len(t3) > 0 else 0
                n2 = int(float(t3[1])) if len(t3) > 1 else 0

            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 4 and not cards[4].is_blank:
                t4 = cards[4].tokens()
                stfac = float(t4[0]) if len(t4) > 0 else 1.0
                fric = float(t4[1]) if len(t4) > 1 else 0.0
                gap = float(t4[2]) if len(t4) > 2 else 0.0
                tstart = float(t4[3]) if len(t4) > 3 else 0.0
                tstop = float(t4[4]) if len(t4) > 4 else 1.0e30

        model.interfaces.append(Interface(
            id=block.user_id, type=19, grnod_id=grnod_id, surf_id=surf_id,
            line_id1=n1, line_id2=n2,
            istf=istf, igap=igap, multimp=iedge, ibag=ibag, idel=idel, icurv=icurv,
            gap_scale=gap_scale, gap_max=gap_max, gap=gap, stmin=stmin, stmax=stmax,
            stfac=stfac, fric=fric, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind in ("TYPE21", "21"):
        read_inter_type21(block, model, log)
        return

    if kind == "GUIDED_CABLE":
        if block.fixed:
            f0 = cards[0].cut("INTER_GUIDED_CABLE_1")
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            grpart_id = _ival(f0[1]) if len(f0) > 1 else 0
            istiff = _ival(f0[2]) if len(f0) > 2 else 1
            stfac = _fval(f0[3], 1.0) if len(f0) > 3 else 1.0
            fric = _fval(f0[4], 0.0) if len(f0) > 4 else 0.0
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            grpart_id = int(float(t0[1])) if len(t0) > 1 else 0
            istiff = int(float(t0[2])) if len(t0) > 2 else 1
            stfac = float(t0[3]) if len(t0) > 3 else 1.0
            fric = float(t0[4]) if len(t0) > 4 else 0.0

        model.interfaces.append(Interface(
            id=block.user_id, type=29, grnod_id=grnod_id, grpart_id=grpart_id,
            istiff=istiff, stfac=stfac, fric=fric, title=title
        ))
        return

    if kind == "TYPE1":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE1_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
        model.interfaces.append(Interface(
            id=block.user_id, type=1, surf_id=surf1, surf_id1=surf2, title=title
        ))
        return

    if kind == "TYPE3":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE3_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE3_2")
                stfac = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stfac = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                tstart = float(t1[3]) if len(t1) > 3 else 0.0
                tstop = float(t1[4]) if len(t1) > 4 else 1.0e30
        model.interfaces.append(Interface(
            id=block.user_id, type=3, surf_id=surf1, surf_id1=surf2,
            stfac=stfac, fric=fric, gap=gap, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind == "TYPE5":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE5_1")
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE5_2")
                stfac = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stfac = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                tstart = float(t1[3]) if len(t1) > 3 else 0.0
                tstop = float(t1[4]) if len(t1) > 4 else 1.0e30
        model.interfaces.append(Interface(
            id=block.user_id, type=5, grnod_id=grnod_id, surf_id=surf_id,
            stfac=stfac, fric=fric, gap=gap, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind == "TYPE6":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE6_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            scale, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE6_2")
                scale = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            scale, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                scale = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                tstart = float(t1[3]) if len(t1) > 3 else 0.0
                tstop = float(t1[4]) if len(t1) > 4 else 1.0e30
        model.interfaces.append(Interface(
            id=block.user_id, type=6, surf_id=surf1, surf_id1=surf2,
            stfac=scale, fric=fric, gap=gap, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind == "TYPE12":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE12_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            interpol = _ival(f0[2]) if len(f0) > 2 else 0
            itied = _ival(f0[3]) if len(f0) > 3 else 0
            bcopt = _ival(f0[4]) if len(f0) > 4 else 0
            skew_id = _ival(f0[5]) if len(f0) > 5 else 0
            node_c = _ival(f0[6]) if len(f0) > 6 else 0

            tol, tstart, tstop = 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE12_2")
                if len(f1) >= 4:
                    tol = _fval(f1[1], 0.0)
                    tstart = _fval(f1[2], 0.0)
                    tstop = _fval(f1[3], 1.0e30)
                elif len(f1) == 3:
                    tol = _fval(f1[0], 0.0)
                    tstart = _fval(f1[1], 0.0)
                    tstop = _fval(f1[2], 1.0e30)

            xc, yc, zc, theta = 0.0, 0.0, 0.0, 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                f2 = cards[2].cut("INTER_TYPE12_3")
                xc = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                yc = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                zc = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
                theta = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0

            xn, yn, zn = 0.0, 0.0, 0.0
            if len(cards) > 3 and not cards[3].is_blank:
                f3 = cards[3].cut("INTER_TYPE12_4")
                xn = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
                yn = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
                zn = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0

            xt, yt, zt = 0.0, 0.0, 0.0
            if len(cards) > 4 and not cards[4].is_blank:
                f4 = cards[4].cut("INTER_TYPE12_5")
                xt = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
                yt = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
                zt = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            interpol = int(float(t0[2])) if len(t0) > 2 else 0
            if len(t0) >= 6:
                tol = float(t0[3])
                tstart = float(t0[4])
                tstop = float(t0[5])
                itied = int(float(t0[6])) if len(t0) > 6 else 0
                bcopt = int(float(t0[7])) if len(t0) > 7 else 0
                skew_id = int(float(t0[8])) if len(t0) > 8 else 0
                node_c = int(float(t0[9])) if len(t0) > 9 else 0
                card_idx = 1
            else:
                itied = int(float(t0[3])) if len(t0) > 3 else 0
                bcopt = int(float(t0[4])) if len(t0) > 4 else 0
                skew_id = int(float(t0[5])) if len(t0) > 5 else 0
                node_c = int(float(t0[6])) if len(t0) > 6 else 0
                tol, tstart, tstop = 0.0, 0.0, 1.0e30
                if len(cards) > 1 and not cards[1].is_blank:
                    t1 = cards[1].tokens()
                    tol = float(t1[0]) if len(t1) > 0 else 0.0
                    tstart = float(t1[1]) if len(t1) > 1 else 0.0
                    tstop = float(t1[2]) if len(t1) > 2 else 1.0e30
                card_idx = 2

            xc, yc, zc, theta = 0.0, 0.0, 0.0, 0.0
            if len(cards) > card_idx and not cards[card_idx].is_blank:
                t1 = cards[card_idx].tokens()
                xc = float(t1[0]) if len(t1) > 0 else 0.0
                yc = float(t1[1]) if len(t1) > 1 else 0.0
                zc = float(t1[2]) if len(t1) > 2 else 0.0
                theta = float(t1[3]) if len(t1) > 3 else 0.0

            xn, yn, zn = 0.0, 0.0, 0.0
            if len(cards) > card_idx + 1 and not cards[card_idx + 1].is_blank:
                t2 = cards[card_idx + 1].tokens()
                xn = float(t2[0]) if len(t2) > 0 else 0.0
                yn = float(t2[1]) if len(t2) > 1 else 0.0
                zn = float(t2[2]) if len(t2) > 2 else 0.0

            xt, yt, zt = 0.0, 0.0, 0.0
            if len(cards) > card_idx + 2 and not cards[card_idx + 2].is_blank:
                t3 = cards[card_idx + 2].tokens()
                xt = float(t3[0]) if len(t3) > 0 else 0.0
                yt = float(t3[1]) if len(t3) > 1 else 0.0
                zt = float(t3[2]) if len(t3) > 2 else 0.0
        from ...model.entities import InterType12
        inter12 = InterType12(
            id=block.user_id, title=title, surf_ids=surf1, surf_idm=surf2,
            interpol=interpol, tol=tol, tstart=tstart, tstop=tstop,
            itied=itied, bcopt=bcopt, skew_id=skew_id, node_c=node_c,
            xc=xc, yc=yc, zc=zc, theta=theta,
            xn=xn, yn=yn, zn=zn, xt=xt, yt=yt, zt=zt,
            params={"surf_ids": surf1, "surf_idm": surf2, "interpol": interpol, "tol": tol}
        )
        model.inter_type12s[block.user_id] = inter12
        model.interfaces.append(Interface(
            id=block.user_id, type=12, surf_id=surf1, surf_id1=surf2,
            tol=tol, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind == "TYPE14":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE14_1")
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
            iload = _ival(f0[2]) if len(f0) > 2 else 0
            ifric = _ival(f0[3]) if len(f0) > 3 else 0
            fun_id1 = _ival(f0[4]) if len(f0) > 4 else 0
            fun_id2 = _ival(f0[5]) if len(f0) > 5 else 0

            stif, fric, gap = 1.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE14_2")
                stif = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
            iload = int(float(t0[2])) if len(t0) > 2 else 0
            ifric = int(float(t0[3])) if len(t0) > 3 else 0
            fun_id1 = int(float(t0[4])) if len(t0) > 4 else 0
            fun_id2 = int(float(t0[5])) if len(t0) > 5 else 0

            stif, fric, gap = 1.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stif = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[3]) if len(t1) > 3 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=14, grnod_id=grnod_id, surf_id=surf_id,
            iload=iload, mfrot=ifric, fun_id1=fun_id1, fun_id2=fun_id2,
            stfac=stif, fric=fric, gap=gap, title=title
        ))
        return

    if kind == "TYPE15":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE15_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            stif, fric = 1.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE15_2")
                stif = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            stif, fric = 1.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stif = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=15, surf_id=surf1, surf_id1=surf2,
            stfac=stif, fric=fric, title=title
        ))
        return

    if kind == "TYPE20":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE20_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            isym = _ival(f0[2]) if len(f0) > 2 else 0
            iedge = _ival(f0[3]) if len(f0) > 3 else 0
            grnod_id = _ival(f0[4]) if len(f0) > 4 else 0
            line_id1 = _ival(f0[5]) if len(f0) > 5 else 0
            line_id2 = _ival(f0[6]) if len(f0) > 6 else 0
            edge_angle = _fval(f0[8], 0.0) if len(f0) > 8 else 0.0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            isym = int(float(t0[2])) if len(t0) > 2 else 0
            iedge = int(float(t0[3])) if len(t0) > 3 else 0
            grnod_id = int(float(t0[4])) if len(t0) > 4 else 0
            line_id1 = int(float(t0[5])) if len(t0) > 5 else 0
            line_id2 = int(float(t0[6])) if len(t0) > 6 else 0
            edge_angle = float(t0[7]) if len(t0) > 7 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=20, surf_id=surf1, surf_id1=surf2,
            isym=isym, iedge=iedge, grnod_id=grnod_id, line_id1=line_id1,
            line_id2=line_id2, edge_angle=edge_angle, title=title
        ))
        return

    if kind == "TYPE22":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE22_1")
            grbric_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
        else:
            t0 = cards[0].tokens()
            grbric_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
        model.interfaces.append(Interface(
            id=block.user_id, type=22, grbric_id1=grbric_id, surf_id=surf_id, title=title
        ))
        return

    if kind in ("TYPE23", "23"):
        read_inter_type23(block, model, log)
        return

    if kind == "TYPE9":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE9_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            f1 = cards[1].cut("INTER_TYPE9_2") if len(cards) > 1 else []
            fric = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
            gap = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
            tstart = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            tstop = _fval(f1[3], 1.0e30) if len(f1) > 3 and _fval(f1[3]) > 0.0 else 1.0e30
            visc = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            t1 = cards[1].tokens() if len(cards) > 1 else []
            fric = float(t1[0]) if len(t1) > 0 else 0.0
            gap = float(t1[1]) if len(t1) > 1 else 0.0
            tstart = float(t1[2]) if len(t1) > 2 else 0.0
            tstop = float(t1[3]) if len(t1) > 3 else 1.0e30
            visc = float(t1[4]) if len(t1) > 4 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=9, surf_id=surf1, surf_id1=surf2,
            fric=fric, gap=gap, tstart=tstart, tstop=tstop, visc=visc, title=title
        ))
        return

    if kind == "TYPE16":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE16_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            stfac = _fval(f0[2], 1.0) if len(f0) > 2 and _fval(f0[2]) > 0.0 else 1.0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            stfac = float(t0[2]) if len(t0) > 2 else 1.0
        model.interfaces.append(Interface(
            id=block.user_id, type=16, surf_id=surf1, surf_id1=surf2,
            stfac=stfac, title=title
        ))
        return

    if kind == "TYPE17":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE17_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            stfac, fric, gap, radius = 1.0, 0.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE17_2")
                stfac = _fval(f1[0], 1.0) if len(f1) > 0 and _fval(f1[0]) > 0.0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                radius = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            stfac, fric, gap, radius = 1.0, 0.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stfac = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                radius = float(t1[3]) if len(t1) > 3 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=17, surf_id=surf1, surf_id1=surf2,
            stfac=stfac, fric=fric, gap=gap, radius=radius, title=title
        ))
        return

    toks = cards[0].tokens()

    if kind == "TYPE10":
        if block.fixed:
            f0 = _fixed_vals(cards[0], [10, 10, 10, 10, 10, 10, 10, 10])
            grnod_id = _ival(f0[0])
            surf_id = _ival(f0[1])
            multimp = _ival(f0[5])
            idel10 = _ival(f0[7])
            
            f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 20])
            stfac = _fval(f1[0]) if f1[0] else 1.0
            gap = _fval(f1[2])
            tstart = _fval(f1[3])
            tstop = _fval(f1[4]) if f1[4] else 1e30
            
            f2 = _fixed_vals(cards[2], [20, 10, 10, 20, 20, 20])
            itied = _ival(f2[1])
            inactiv = _ival(f2[2])
            stiff_dc = _fval(f2[3])
            sort_fact = _fval(f2[5]) if f2[5] else 0.2
        else:
            t0 = cards[0].tokens()
            grnod_id = int(t0[0]) if len(t0) > 0 else 0
            surf_id = int(t0[1]) if len(t0) > 1 else 0
            multimp = int(t0[2]) if len(t0) > 2 else 0
            idel10 = int(t0[3]) if len(t0) > 3 else 0
            
            t1 = cards[1].tokens()
            stfac = float(t1[0]) if len(t1) > 0 else 1.0
            gap = float(t1[1]) if len(t1) > 1 else 0.0
            tstart = float(t1[2]) if len(t1) > 2 else 0.0
            tstop = float(t1[3]) if len(t1) > 3 else 1e30
            
            t2 = cards[2].tokens()
            itied = int(t2[0]) if len(t2) > 0 else 0
            inactiv = int(t2[1]) if len(t2) > 1 else 0
            stiff_dc = float(t2[2]) if len(t2) > 2 else 0.0
            sort_fact = float(t2[3]) if len(t2) > 3 else 0.2
        from ...model.entities import InterType10
        inter10 = InterType10(
            id=block.user_id, title=title, grnod_id=grnod_id, surf_id=surf_id,
            multimp=multimp, idel=idel10, stfac=stfac, gap=gap, tstart=tstart,
            tstop=tstop, itied=itied, inactiv=inactiv, stiff_dc=stiff_dc,
            sort_fact=sort_fact,
            params={"grnod_id": grnod_id, "surf_id": surf_id, "stfac": stfac, "gap": gap, "itied": itied}
        )
        model.inter_type10s[block.user_id] = inter10
        model.interfaces.append(Interface(
            id=block.user_id, type=10, grnod_id=grnod_id, surf_id=surf_id,
            multimp=multimp, idel10=idel10, stfac=stfac, gap=gap, tstart=tstart,
            tstop=tstop, itied=itied, inactiv=inactiv, stiff_dc=stiff_dc,
            sort_fact=sort_fact, title=title))
        return

    if kind == "TYPE18":
        if block.fixed:
            f0 = _fixed_vals(cards[0], [10, 10, 10, 30, 10, 10])
            grnod_id = _ival(f0[0])
            surf_id = _ival(f0[1])
            grbric_id = _ival(f0[2])
            ibag = _ival(f0[4])
            idel18 = _ival(f0[5])
            
            f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 20])
            stfac = _fval(f1[0])
            gap = _fval(f1[2])
            
            f2 = _fixed_vals(cards[2], [40, 20, 20, 20])
            stiff_dc = _fval(f2[1])
            sort_fact = _fval(f2[3])
        else:
            t0 = cards[0].tokens()
            grnod_id = int(t0[0]) if len(t0) > 0 else 0
            surf_id = int(t0[1]) if len(t0) > 1 else 0
            grbric_id = int(t0[2]) if len(t0) > 2 else 0
            ibag = int(t0[3]) if len(t0) > 3 else 0
            idel18 = int(t0[4]) if len(t0) > 4 else 0
            
            t1 = cards[1].tokens()
            stfac = float(t1[0]) if len(t1) > 0 else 0.0
            gap = float(t1[2]) if len(t1) > 2 else 0.0
            
            t2 = cards[2].tokens()
            stiff_dc = float(t2[1]) if len(t2) > 1 else 0.0
            sort_fact = float(t2[3]) if len(t2) > 3 else 0.2
        
        model.interfaces.append(Interface(
            id=block.user_id, type=18, grnod_id=grnod_id, surf_id=surf_id,
            grbric_id1=grbric_id, ibag=ibag, idel18=idel18, stfac=stfac, gap=gap,
            stiff_dc=stiff_dc, sort_fact=sort_fact, title=title))
        return

    if kind == "TYPE2":
        if block.fixed:
            # cfg inter_type2.cfg (radioss2017): grnd_IDs surf_IDm
            # Ignore Spotflag Level Isearch Idel2 <blank> dsearch — the
            # option flags are accepted + warned, dsearch sits in
            # columns 81-100 (M37)
            f = cards[0].cut("INTER2")
            _warn_ignored(log, f"/INTER/TYPE2/{block.user_id}",
                          block.source,
                          [("Ignore", f[2]),
                           ("Level", f[4]), ("Isearch", f[5]),
                           ("Idel2", f[6])])
            model.interfaces.append(Interface(
                id=block.user_id, type=2, grnod_id=_ival(f[0]),
                surf_id=_ival(f[1]), dsearch=_fval(f[8]), spotflag=_ival(f[3]), title=title))
            return
        
        toks = cards[0].tokens()
        spotflag = int(toks[3]) if len(toks) > 3 else 0
        model.interfaces.append(Interface(
            id=block.user_id, type=2, grnod_id=int(toks[0]),
            surf_id=int(toks[1]), dsearch=float(toks[2]) if len(toks) > 2 else 0.0, spotflag=spotflag, title=title))
        return

    tstart, tstop, viss = 0.0, 1e30, 0.05
    if kind == "TYPE24" and len(cards) >= 6:
        ign: List[str] = []
        gap = 0.0
        f0 = _fixed_vals(cards[0], [10] * 9)
        id1, id2 = _ival(f0[0]), _ival(f0[1])
        istf = _ival(f0[2])
        idel = _ival(f0[6])
        for name, s in (("Irem_i2", f0[4]), ("IPSTIF", f0[8])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
                
        f1 = _fixed_vals(cards[1], [10, 20, 10, 20, 20, 20])
        grnod_id = _ival(f1[0])
        if _ival(f1[2]) != 0:
            ign.append(f"Iedge={f1[2]}")
        gap_max = _fval(f1[4])  # Gap_max_s
        gap_max_m = _fval(f1[5]) # Gap_max_m
        
        f2 = _fixed_vals(cards[2], [20, 20, 10, 10, 20, 20])
        igap = _ival(f2[2])
        for name, s in (("Stmin", f2[0]), ("Stmax", f2[1]), ("Ipen", f2[3]), ("Ipen_max", f2[4]), ("STFAC_MDT", f2[5])):
            if s and s.strip() and any(_to_float(tok) != 0.0 for tok in s.split()):
                ign.append(f"{name}={s.strip()}")
                
        f3 = _fixed_vals(cards[3], [20, 20, 20, 20, 20])
        stfac, fric = _fval(f3[0]), _fval(f3[1])
        tstart = _fval(f3[3])
        tstop = _fval(f3[4]) if _fval(f3[4]) > 0.0 else 1e30
                
        f4 = _fixed_vals(cards[4], [7, 1, 1, 1, 20, 10, 20, 20, 20])
        if _ival(f4[1]) or _ival(f4[2]) or _ival(f4[3]):
            ign.append(f"IBC={f4[1] or '0'}{f4[2] or '0'}{f4[3] or '0'}")
        viss = _fval(f4[6]) if _fval(f4[6]) > 0.0 else 0.05
        for name, s in (("Tpressfit", f4[8]),):
            if s and s.strip() and any(_to_float(tok) != 0.0 for tok in s.split()):
                ign.append(f"{name}={s.strip()}")
                
        f5 = _fixed_vals(cards[5], [10, 10, 20, 10, 10, 20, 10, 10])
        mfrot, ifq = _ival(f5[0]), _ival(f5[1])
        xfreq, sens = _fval(f5[2]), _ival(f5[4])
        for name, s in (("DTSTIF", f5[5]), ("Fric_ID", f5[7])):
            if s and s.strip() and any(_to_float(tok) != 0.0 for tok in s.split()):
                ign.append(f"{name}={s.strip()}")
                
        fric_c = (0.0,) * 6
        icard = 6
        if mfrot > 0:
            cc = [0.0] * 6
            if len(cards) > icard:
                cc[:5] = _floats(cards[icard], 5)
                icard += 1
                if mfrot > 1 and len(cards) > icard:
                    cc[5] = _floats(cards[icard], 1)[0]
                    icard += 1
            else:
                log.warning(f"/INTER/TYPE24/{block.user_id}: Ifric={mfrot} "
                            f"without a C1..C5 card — all coefficients 0",
                            block.source)
            fric_c = tuple(cc)
            
        if ign:
            log.warning(f"/INTER/TYPE24/{block.user_id}: real-format fields "
                        f"not ported — ignored: {'; '.join(ign)}",
                        block.source)

    if kind == "TYPE11" and (len(cards) >= 4 or (block.fixed and len(cards) == 3 and not (len(cards[0].ints()) >= 6 and len(cards[2].floats()) == 6))):
        # ==== the REAL fixed-format TYPE11 layout (inter_type11.cfg) ========
        ign: List[str] = []
        fscale_gap, percent_mesh_size = 1.0, 0.4
        f0 = _fixed_vals(cards[0], [10] * 8)
        id1, id2 = _ival(f0[0]), _ival(f0[1])
        istf, igap = _ival(f0[2]), _ival(f0[4])
        for name, s in (("Ithe", f0[3]), ("Multimp", f0[5]), ("Idel", f0[7])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
        if len(cards) >= 4:
            f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 10, 10])
            percent_mesh_size = _fval(f1[2], 0.4)
            sens = _ival(f1[5])
            for name, s in (("Stmin", f1[0]), ("Stmax", f1[1]), ("dtmin", f1[3]), ("Iform", f1[4])):
                if s and _to_float(s) != 0.0:
                    ign.append(f"{name}={s}")
            f2 = _fixed_vals(cards[2], [20] * 5)
            stfac, fric, gap = _fval(f2[0], 1.0), _fval(f2[1], 0.0), _fval(f2[2], 0.0)
            for name, s in (("Tstart", f2[3]), ("Tstop", f2[4])):
                if s and _to_float(s) != 0.0:
                    ign.append(f"{name}={s}")
            f3 = _fixed_vals(cards[3], [7, 1, 1, 1, 20, 10, 20, 20, 20])
            for name, s in (("Inacti", f3[5]), ("VIS_S", f3[6]), ("VIS_F", f3[7]), ("Bumult", f3[8])):
                if s and _to_float(s) != 0.0:
                    ign.append(f"{name}={s}")
        else:
            sens = 0
            f1 = _fixed_vals(cards[1], [20] * 5)
            stfac, fric, gap = _fval(f1[0], 1.0), _fval(f1[1], 0.0), _fval(f1[2], 0.0)
            for name, s in (("Tstart", f1[3]), ("Tstop", f1[4])):
                if s and _to_float(s) != 0.0:
                    ign.append(f"{name}={s}")
        gap_max = 0.0
        mfrot, ifq, xfreq = 0, 0, 0.0
        fric_c = (0.0,) * 6
        if ign:
            log.warning(f"/INTER/TYPE11/{block.user_id}: real-format fields "
                        f"not ported — ignored: {'; '.join(ign)}",
                        block.source)
    elif kind in ("TYPE7", "TYPE11", "TYPE24") and len(cards) < 6:
        # ==== the port's compact layout =====================================
        t = cards[0].ints()
        id1 = t[0]
        id2 = t[1]
        istf = t[2] if len(t) > 2 else 0
        igap = t[3] if len(t) > 3 else 0
        sens = t[4] if len(t) > 4 else 0
        mfrot = t[5] if len(t) > 5 else 0        # Ifric (M15)
        ifq = t[6] if len(t) > 6 else 0          # Ifiltr (M15)
        iform = t[7] if len(t) > 7 else 0
        idel = t[8] if len(t) > 8 else 0
        stfac, fric, gap, gap_max, xfreq = (1.0, 0.0, 0.0, 0.0, 0.0)
        gap_max_m = 0.0
        fscale_gap, percent_mesh_size = 1.0, 0.4
        tstart, tstop, viss = 0.0, 1.0e30, 0.05
        grnod_id = id1
        if kind == "TYPE24":
            id1 = 0 # surf_id1 is 0 when using node-to-surface
        
        if len(cards) > 1:
            c1_toks = cards[1].tokens()
            if len(c1_toks) > 5:
                vals = _floats(cards[1], 8, defaults=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e30, 0.05])
                stfac, fric, gap, gap_max, xfreq = vals[:5]
                tstart = vals[5]
                if len(c1_toks) > 6:
                    tstop = vals[6]
                if len(c1_toks) > 7:
                    viss = vals[7]
            else:
                stfac, fric, gap, gap_max, xfreq = _floats(
                    cards[1], 5, defaults=[1.0, 0.0, 0.0, 0.0, 0.0])
        # ---- optional C1..C6 card (the original's card 8, Ifric > 0) ------
        fric_c = (0.0,) * 6
        if mfrot in (1, 2, 3, 4):
            if len(cards) > 2:
                cc = _floats(cards[2], 6, defaults=[0.0] * 6)
                # C6 is only read for Ifric > 1 (hm_read_inter_type07.F)
                fric_c = tuple(cc[:5]) + ((cc[5],) if mfrot > 1 else (0.0,))
            else:
                log.warning(f"/INTER/{kind}/{block.user_id}: Ifric={mfrot} "
                            f"without a C1..C6 card — all coefficients 0",
                            block.source)

    elif kind == "TYPE7" and len(cards) >= 6:
        # ==== the REAL fixed-format TYPE7 layout (see docstring) ===========
        ign: List[str] = []            # non-default fields the port ignores
        fscale_gap, percent_mesh_size = 1.0, 0.4
        f0 = _fixed_vals(cards[0], [10] * 10)
        id1, id2 = _ival(f0[0]), _ival(f0[1])
        istf, igap = _ival(f0[2]), _ival(f0[4])
        idel = _ival(f0[7])
        for name, s in (("Ithe", f0[3]), ("Ibag", f0[6]), ("Idel", f0[7]),
                        ("Iadm", f0[9])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
        icurv = _ival(f0[8])
        f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 10])
        gap_max = _fval(f1[1])
        fscale_gap = _fval(f1[0])
        if fscale_gap == 0.0:
            fscale_gap = 1.0
        if igap < 2 and fscale_gap != 1.0:
            ign.append(f"Fscale_gap={f1[0]}")
        for name, s in (("Fpenmax", f1[2]), ("Itied", f1[4])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        f2 = _fixed_vals(cards[2], [20, 20, 20, 20, 10, 10])
        percent_mesh_size = _fval(f2[2])
        if percent_mesh_size == 0.0:
            percent_mesh_size = 0.4
        if igap != 3 and percent_mesh_size != 0.4:
            ign.append(f"%mesh_size={f2[2]}")
        for name, s in (("Stmin", f2[0]), ("Stmax", f2[1]),
                        ("dtmin", f2[3]),
                        ("Irem_gap", f2[4]), ("Irem_i2", f2[5])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        icard = 3
        if icurv in (1, 2):            # the optional curvature node card
            ign.append(f"Icurv={icurv} (node card skipped)")
            icard += 1
        elif icurv != 0:
            ign.append(f"Icurv={icurv}")
        if len(cards) < icard + 3:
            log.error(f"/INTER/TYPE7/{block.user_id}: real-format block "
                      f"needs 6 data cards (got {len(cards)})", block.source)
            return
        f3 = _fixed_vals(cards[icard], [20] * 5)
        stfac, fric, gap = _fval(f3[0]), _fval(f3[1]), _fval(f3[2])
        tstart = _fval(f3[3], 0.0)
        tstop = _fval(f3[4], 1.0e30)
        if tstop == 0.0:
            tstop = 1.0e30
        f4 = _fixed_vals(cards[icard + 1], [7, 1, 1, 1, 20, 10, 20, 20, 20])
        if _ival(f4[1]) or _ival(f4[2]) or _ival(f4[3]):
            ign.append(f"IBC={f4[1] or '0'}{f4[2] or '0'}{f4[3] or '0'}")
        viss = _fval(f4[6], 0.05)
        if viss == 0.0:
            viss = 0.05
        for name, s in (("VisF", f4[7]), ("Bumult", f4[8])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        f5 = _fixed_vals(cards[icard + 2],
                         [10, 10, 20, 10, 10, 10, 20, 10])
        mfrot, ifq = _ival(f5[0]), _ival(f5[1])
        xfreq, iform, sens = _fval(f5[2]), _ival(f5[3]), _ival(f5[4])
        for name, s in (("fct_IDF", f5[5]), ("fric_ID", f5[7])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
        if f5[6] and _to_float(f5[6]) not in (0.0, 1.0):
            ign.append(f"AscaleF={f5[6]}")
        # Iform = MODFR: 2 selects the incremental (stiffness) tangential
        # formulation (upstream turns it into IFQ >= 10). Without friction
        # it changes nothing.  The IFQ += 10 offset is applied AFTER the
        # xfreq/ALPHA mapping (shared validation section) — matching the
        # Fortran order where MODFR is applied last.
        _iform2_active = False
        if iform == 2 and fric == 0.0 and mfrot == 0:
            ign.append("Iform=2 (no friction defined — inert)")
        elif iform == 2 and (fric != 0.0 or mfrot != 0):
            _iform2_active = True
        # C1..C5 (Ifric > 0) and C6 (Ifric > 1) cards
        fric_c = (0.0,) * 6
        icard += 3
        if mfrot > 0:
            cc = [0.0] * 6
            if len(cards) > icard:
                cc[:5] = _floats(cards[icard], 5)
                icard += 1
                if mfrot > 1 and len(cards) > icard:
                    cc[5] = _floats(cards[icard], 1)[0]
                    icard += 1
            else:
                log.warning(f"/INTER/TYPE7/{block.user_id}: Ifric={mfrot} "
                            f"without a C1..C5 card — all coefficients 0",
                            block.source)
            fric_c = tuple(cc)
        if ign:
            log.warning(f"/INTER/TYPE7/{block.user_id}: real-format fields "
                        f"not ported — ignored: {'; '.join(ign)}",
                        block.source)
    elif kind == "TYPE7": # should never reach here since we handle len < 6 above
        pass
        # removed dup else block

    # ==== shared validation (both dialects) ================================
    if istf not in (0, 1, 2, 3, 4, 5):
        log.error(f"/INTER/{kind}/{block.user_id}: Istf={istf} (0..5)",
                  block.source)
    if igap not in (0, 1, 2, 3):
        log.error(f"/INTER/{kind}/{block.user_id}: Igap={igap} not ported "
                  f"(0 constant, 1 variable, 2 scaled, 3 mesh-size)", block.source)
    if mfrot not in (0, 1, 2, 3, 4):
        log.error(f"/INTER/{kind}/{block.user_id}: Ifric={mfrot} (0..4: "
                  f"Coulomb / generalized viscous / Darmstadt / Renard / "
                  f"exponential decay)", block.source)
        mfrot = 0
    if ifq >= 10:
        # MODFR = 2 / the incremental (stiffness) tangential formulation.
        if ifq not in (10, 11, 12, 13):
            log.error(f"/INTER/{kind}/{block.user_id}: Ifiltr={ifq} "
                      f"(incremental stiffness Ifiltr must be 10..13)", block.source)
            ifq = 0
    elif ifq not in (0, 1, 2, 3):
        log.error(f"/INTER/{kind}/{block.user_id}: Ifiltr={ifq} (0..3)",
                  block.source)
        ifq = 0
    if stfac == 0.0 and istf != 1:
        stfac = 1.0                # Radioss: Stfac = 0 -> default scale 1.0
    if istf == 1 and stfac <= 0.0:
        log.error(f"/INTER/{kind}/{block.user_id}: Istf=1 needs a "
                  f"positive Stfac (it IS the stiffness)", block.source)
    # ---- the XFILTR mapping of hm_read_inter_type07.F (M15, checked) ------
    # Xfreq (ALPHA) = 0 switches the filter OFF whatever Ifiltr says —
    # exactly the reference: IF (ALPHA==0.) IFQ = 0. Then IFQ=1: Xfreq IS
    # the coefficient; IFQ=2: 2*pi/Xfreq (a period in cycles); IFQ=3:
    # 2*pi*Xfreq (a cutoff frequency — alpha = XFILTR*dt per cycle). The
    # original's MSGID 554 errors are mirrored.
    if xfreq == 0.0:
        ifq = 0
    # Apply the Iform=2 → IFQ += 10 offset (real-format only) AFTER the
    # xfreq reset, matching Fortran order (MODFR applied last).
    try:
        if _iform2_active:
            ifq = ifq + 10
    except NameError:
        pass   # compact path — variable not defined
    xfiltr = 0.0
    if ifq > 0:
        if ifq == 10:
            xfiltr = 1.0                     # IFQ=10: constant alpha=1 (hm_read_inter_type07.F:623)
        elif ifq % 10 == 1:
            xfiltr = xfreq
        elif ifq % 10 == 2:
            xfiltr = (2.0 * np.pi / xfreq) if xfreq > 0.0 else -1.0
        elif ifq % 10 == 3:
            xfiltr = 2.0 * np.pi * xfreq
        if xfiltr < 0.0 or (xfiltr > 1.0 and ifq % 10 <= 2):
            log.error(f"/INTER/{kind}/{block.user_id}: friction filtering "
                      f"factor out of range (Xfreq={xfreq:g} -> "
                      f"XFILTR={xfiltr:g}, must be in [0,1] for "
                      f"Ifiltr 1/2)", block.source)
            ifq, xfiltr = 0, 0.0
    if kind == "TYPE7":
        model.interfaces.append(Interface(
            id=block.user_id, type=7, grnod_id=id1, surf_id=id2,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap,
            gap_max=gap_max, fscale_gap=fscale_gap, percent_mesh_size=percent_mesh_size,
            sens_id=sens, mfrot=mfrot, ifq=ifq,
            xfiltr=xfiltr, fric_c=fric_c, title=title,
            iform=iform if 'iform' in locals() else 0,
            idel=idel if 'idel' in locals() else 0,
            tstart=tstart if 'tstart' in locals() else 0.0,
            tstop=tstop if 'tstop' in locals() else 1.0e30,
            viss=viss if 'viss' in locals() else 0.05,
            stiff_dc=viss if 'viss' in locals() else 0.05))
    elif kind == "TYPE24":
        # For TYPE24, we pass gap so compact mode can explicitly set it for tests.
        model.interfaces.append(Interface(
            id=block.user_id, type=24, grnod_id=grnod_id, surf_id1=id1, surf_id=id2,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap,
            gap_max=gap_max, gap_max_m=gap_max_m, sens_id=sens, mfrot=mfrot, ifq=ifq,
            xfiltr=xfiltr, fric_c=fric_c, title=title,
            idel=idel if 'idel' in locals() else 0,
            tstart=tstart if 'tstart' in locals() else 0.0,
            tstop=tstop if 'tstop' in locals() else 1e30,
            stiff_dc=viss if ('viss' in locals() and viss > 0.0) else 0.05))
    else:                          # TYPE11
        if mfrot > 0 or ifq > 0:
            # the original TYPE11 has no friction models at all (checked:
            # i11mainf.F forces MFROT = 0) — the port extension is
            # announced so nobody mistakes it for Radioss behaviour
            log.info(f"     /INTER/TYPE11/{block.user_id}: FRICTION "
                     f"MODEL Ifric={mfrot} Ifiltr={ifq} — A PORT "
                     f"EXTENSION (the original TYPE11 never evaluates "
                     f"MFROT; see contact/friction.py)")
        model.interfaces.append(Interface(
            id=block.user_id, type=11, line_id1=id1, line_id2=id2,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap,
            gap_max=gap_max, fscale_gap=fscale_gap, percent_mesh_size=percent_mesh_size,
            sens_id=sens, mfrot=mfrot, ifq=ifq,
            xfiltr=xfiltr, fric_c=fric_c, title=title))




def read_subinter(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTER/SUB/sub_id`` or ``/SUBINTER/sub_id`` (M100/M149): Sub-interface for force/energy tracking.

    Fortran origin: ``starter/source/output/subinterface/hm_read_intsub.F`` / CFG ``inter_sub.cfg``.
    """
    from ...model.entities import SubInterface
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INTER/SUB/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("INTER_SUB_1")
        inter_id = _ival(f[0]) if len(f) > 0 else 0
        m1 = _ival(f[1]) if len(f) > 1 else 0
        s = _ival(f[2]) if len(f) > 2 else 0
        m2 = _ival(f[3]) if len(f) > 3 else 0
    else:
        toks = cards[0].tokens()
        inter_id = int(float(toks[0])) if len(toks) > 0 else 0
        m1 = int(float(toks[1])) if len(toks) > 1 else 0
        s = int(float(toks[2])) if len(toks) > 2 else 0
        m2 = int(float(toks[3])) if len(toks) > 3 else 0
    model.sub_interfaces.append(SubInterface(
        id=block.user_id, title=title, inter_id=inter_id, main_id1=m1, second_id=s, main_id2=m2
    ))




def read_guided_cable(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTER/GUIDED_CABLE/cable_id`` or ``/GUIDED_CABLE/cable_id`` (M149): Guided cable sliding interface.

    Fortran origin: ``starter/source/tools/seatbelts/hm_read_guided_cable.F90`` / CFG ``inter_guided_cable.cfg``.
    """
    from ...model.entities import GuidedCable
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INTER/GUIDED_CABLE/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("INTER_GUIDED_CABLE_1")
        grnod_id = _ival(f[0]) if len(f) > 0 else 0
        grpart_id = _ival(f[1]) if len(f) > 1 else 0
        istiff = _ival(f[2]) if len(f) > 2 else 1
        stfac = _fval(f[3], 1.0) if len(f) > 3 else 1.0
        fric = _fval(f[4], 0.0) if len(f) > 4 else 0.0
    else:
        toks = cards[0].tokens()
        grnod_id = int(float(toks[0])) if len(toks) > 0 else 0
        grpart_id = int(float(toks[1])) if len(toks) > 1 else 0
        istiff = int(float(toks[2])) if len(toks) > 2 else 1
        stfac = float(toks[3]) if len(toks) > 3 else 1.0
        fric = float(toks[4]) if len(toks) > 4 else 0.0
    gc = GuidedCable(
        id=block.user_id, grnod_id=grnod_id, grpart_id=grpart_id,
        istiff=istiff, stfac=stfac, fric=fric, title=title
    )
    model.guided_cables[block.user_id] = gc
    model.interfaces.append(Interface(
        id=block.user_id, type=29, grnod_id=grnod_id, grpart_id=grpart_id,
        istf=istiff, stfac=stfac, fric=fric, title=title
    ))




def read_inter_type21(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTER/TYPE21/inter_ID`` (M585): Drawbead contact interface.

    Fortran origin: ``starter/source/interfaces/int21/hm_read_inter_type21.F``
    and CFG ``inter_type21.cfg``.
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    is_fixed = block.fixed or (len(cards) > 0 and len(cards[0].raw) >= 60)
    if is_fixed and not block.fixed:
        title, cards = _fixed_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INTER/TYPE21/{block.user_id}: missing data card", block.source)
        return

    surf_s = 0
    surf_m = 0
    istf = 0
    igap = 0
    multimp = 4
    iadm = 0
    gap_scale = 1.0
    gap_max = 0.0
    depth = 0.0
    pmax = 1e30
    itlim = 0
    stmin = 0.0
    stmax = 1e30
    stfac = 1.0
    fric = 0.0
    gap_min = 0.0
    tstart = 0.0
    tstop = 1e30
    inactiv = 0
    viss = 0.05
    sort_fact = 0.2
    ifric = 0
    ifiltr = 0
    xfreq = 0.0
    sens_id = 0
    c1, c2, c3, c4, c5, c6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    is_20col = False
    if is_fixed:
        is_20col = (len(cards[0].raw) > 80 and cards[0].raw[:10].strip() == "" and len(cards[0].raw) >= 30 and cards[0].raw[20:30].strip() == "")
        if is_20col:
            f0 = cards[0].cut([20, 20, 20, 20, 20, 20])
            surf_s = _ival(f0[0]) if len(f0) > 0 else 0
            surf_m = _ival(f0[1]) if len(f0) > 1 else 0
            istf = _ival(f0[2]) if len(f0) > 2 else 0
            igap = _ival(f0[3]) if len(f0) > 3 else 0
            multimp = _ival(f0[4]) if len(f0) > 4 else 0
            iadm = _ival(f0[5]) if len(f0) > 5 else 0
        else:
            f0 = cards[0].cut("INTER_TYPE21_1")
            surf_s = _ival(f0[0]) if len(f0) > 0 else 0
            surf_m = _ival(f0[1]) if len(f0) > 1 else 0
            istf = _ival(f0[2]) if len(f0) > 2 else 0
            igap = _ival(f0[4]) if len(f0) > 4 else 0
            multimp = _ival(f0[5]) if len(f0) > 5 else 4
            iadm = _ival(f0[8]) if len(f0) > 8 else (_ival(f0[7]) if len(f0) > 7 else 0)

        if len(cards) > 1 and not cards[1].is_blank:
            f1 = cards[1].cut("INTER_TYPE21_2")
            gap_scale = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
            gap_max = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
            depth = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            pmax = _fval(f1[3], 1e30) if len(f1) > 3 and _fval(f1[3], 0.0) > 0.0 else 1e30
            itlim = _ival(f1[4]) if len(f1) > 4 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            if len(cards) == 3:
                # compact legacy format: Stmin, Stmax, Stfac, Fric
                f2 = _fixed_vals(cards[2], [20, 20, 20, 20, 20])
                if len(f2) >= 4 and (_fval(f2[2], 0.0) != 0.0 or _fval(f2[3], 0.0) != 0.0):
                    stfac = _fval(f2[2], 1.0)
                    fric = _fval(f2[3], 0.0)
                else:
                    stmin = _fval(f2[0], 0.0)
                    stmax = _fval(f2[1], 1e30) if _fval(f2[1], 0.0) > 0.0 else 1e30
            else:
                f2 = cards[2].cut("INTER_TYPE21_3")
                stmin = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                stmax = _fval(f2[1], 1e30) if len(f2) > 1 and _fval(f2[1], 0.0) > 0.0 else 1e30

        if len(cards) > 3 and not cards[3].is_blank:
            f3 = cards[3].cut("INTER_TYPE21_4")
            stfac = _fval(f3[0], 1.0) if len(f3) > 0 else 1.0
            fric = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            gap_min = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            tstart = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
            tstop = _fval(f3[4], 1e30) if len(f3) > 4 and _fval(f3[4], 0.0) > 0.0 else 1e30

        if len(cards) > 4 and not cards[4].is_blank:
            f4 = cards[4].cut("INTER_TYPE21_5")
            inactiv = _ival(f4[5]) if len(f4) > 5 else 0
            viss = _fval(f4[6], 0.05) if len(f4) > 6 else 0.05
            sort_fact = _fval(f4[8], 0.2) if len(f4) > 8 else 0.2

        if len(cards) > 5 and not cards[5].is_blank:
            f5 = cards[5].cut("INTER_TYPE21_6")
            ifric = _ival(f5[0]) if len(f5) > 0 else 0
            ifiltr = _ival(f5[1]) if len(f5) > 1 else 0
            xfreq = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0
            sens_id = _ival(f5[4]) if len(f5) > 4 else 0

        card_idx = 6
        if ifric > 0 and len(cards) > card_idx and not cards[card_idx].is_blank:
            fc1 = _fixed_vals(cards[card_idx], [20, 20, 20, 20, 20])
            c1 = _fval(fc1[0], 0.0)
            c2 = _fval(fc1[1], 0.0) if len(fc1) > 1 else 0.0
            c3 = _fval(fc1[2], 0.0) if len(fc1) > 2 else 0.0
            c4 = _fval(fc1[3], 0.0) if len(fc1) > 3 else 0.0
            c5 = _fval(fc1[4], 0.0) if len(fc1) > 4 else 0.0
            card_idx += 1
        if ifric > 1 and len(cards) > card_idx and not cards[card_idx].is_blank:
            fc2 = _fixed_vals(cards[card_idx], [20])
            c6 = _fval(fc2[0], 0.0)
    else:
        t0 = cards[0].tokens()
        surf_s = int(float(t0[0])) if len(t0) > 0 else 0
        surf_m = int(float(t0[1])) if len(t0) > 1 else 0
        istf = int(float(t0[2])) if len(t0) > 2 else 0
        igap = int(float(t0[3])) if len(t0) > 3 else 0
        multimp = int(float(t0[4])) if len(t0) > 4 else 4
        iadm = int(float(t0[5])) if len(t0) > 5 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t1 = cards[1].tokens()
            gap_scale = float(t1[0]) if len(t1) > 0 else 1.0
            gap_max = float(t1[1]) if len(t1) > 1 else 0.0
            depth = float(t1[2]) if len(t1) > 2 else 0.0
            pmax = float(t1[3]) if len(t1) > 3 and float(t1[3]) > 0.0 else 1e30
            itlim = int(float(t1[4])) if len(t1) > 4 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            t2 = cards[2].tokens()
            if len(cards) == 3:
                stfac = float(t2[2] if len(t2) > 2 else t2[0]) if len(t2) > 0 else 1.0
                fric = float(t2[3] if len(t2) > 3 else (t2[1] if len(t2) > 1 else 0.0)) if len(t2) > 1 else 0.0
            else:
                stmin = float(t2[0]) if len(t2) > 0 else 0.0
                stmax = float(t2[1]) if len(t2) > 1 and float(t2[1]) > 0.0 else 1e30

        if len(cards) > 3 and not cards[3].is_blank:
            t3 = cards[3].tokens()
            stfac = float(t3[0]) if len(t3) > 0 else 1.0
            fric = float(t3[1]) if len(t3) > 1 else 0.0
            gap_min = float(t3[2]) if len(t3) > 2 else 0.0
            tstart = float(t3[3]) if len(t3) > 3 else 0.0
            tstop = float(t3[4]) if len(t3) > 4 and float(t3[4]) > 0.0 else 1e30

        if len(cards) > 4 and not cards[4].is_blank:
            t4 = cards[4].tokens()
            inactiv = int(float(t4[1])) if len(t4) > 1 else 0
            viss = float(t4[2]) if len(t4) > 2 else 0.05
            sort_fact = float(t4[3]) if len(t4) > 3 else 0.2

        if len(cards) > 5 and not cards[5].is_blank:
            t5 = cards[5].tokens()
            ifric = int(float(t5[0])) if len(t5) > 0 else 0
            ifiltr = int(float(t5[1])) if len(t5) > 1 else 0
            xfreq = float(t5[2]) if len(t5) > 2 else 0.0
            sens_id = int(float(t5[4])) if len(t5) > 4 else 0

        card_idx = 6
        if ifric > 0 and len(cards) > card_idx and not cards[card_idx].is_blank:
            tc1 = cards[card_idx].tokens()
            c1 = float(tc1[0]) if len(tc1) > 0 else 0.0
            c2 = float(tc1[1]) if len(tc1) > 1 else 0.0
            c3 = float(tc1[2]) if len(tc1) > 2 else 0.0
            c4 = float(tc1[3]) if len(tc1) > 3 else 0.0
            c5 = float(tc1[4]) if len(tc1) > 4 else 0.0
            card_idx += 1
        if ifric > 1 and len(cards) > card_idx and not cards[card_idx].is_blank:
            tc2 = cards[card_idx].tokens()
            c6 = float(tc2[0]) if len(tc2) > 0 else 0.0

    is_sub = any("SUB" in str(p).upper() for p in block.parts)
    is_surf12 = (is_fixed and is_20col)
    assign_s_first = is_sub or is_surf12
    model.interfaces.append(Interface(
        id=block.user_id,
        type=21,
        surf_id=surf_s if assign_s_first else surf_m,
        surf_id1=surf_m if assign_s_first else surf_s,
        istf=istf,
        igap=igap,
        multimp=multimp,
        iadm=iadm,
        gap_scale=gap_scale,
        fscale_gap=gap_scale,
        gap_max=gap_max,
        depth=depth,
        dsearch=depth,
        pmax=pmax,
        itlim=itlim,
        stmin=stmin,
        stmax=stmax,
        stfac=stfac,
        fric=fric,
        gap=gap_min,
        gap_min=gap_min,
        tstart=tstart,
        tstop=tstop,
        inactiv=inactiv,
        viss=viss,
        stiff_dc=viss,
        sort_fact=sort_fact,
        sens_id=sens_id,
        ifric=ifric,
        mfrot=ifric,
        ifiltr=ifiltr,
        ifq=ifiltr,
        xfreq=xfreq,
        fric_c=(c1, c2, c3, c4, c5, c6),
        title=title,
    ))




def read_inter_type23(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTER/TYPE23/inter_ID`` (M585): Mortar segment-to-segment contact interface.

    Fortran origin: ``starter/source/interfaces/int23/hm_read_inter_type23.F``
    and CFG ``inter_type23.cfg``.
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    is_fixed = block.fixed or (len(cards) > 0 and len(cards[0].raw) >= 60)
    if is_fixed and not block.fixed:
        title, cards = _fixed_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INTER/TYPE23/{block.user_id}: missing data card", block.source)
        return

    surf_s = 0
    surf_m = 0
    istf = 0
    igap = 0
    ibag = 0
    idel = 0
    fscale_gap = 1.0
    gap_max = 0.0
    fpenmax = 1.0
    stmin = 0.0
    stmax = 1e30
    stfac = 1.0
    fric = 0.0
    gap_min = 0.0
    tstart = 0.0
    tstop = 1e30
    inactiv = 0
    viss = 0.05
    sort_fact = 0.2
    ifric = 0
    ifiltr = 0
    xfreq = 0.0
    c1, c2, c3, c4, c5, c6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if is_fixed:
        is_20col = (len(cards[0].raw) > 80 and cards[0].raw[:10].strip() == "" and len(cards[0].raw) >= 30 and cards[0].raw[20:30].strip() == "")
        if is_20col:
            f0 = cards[0].cut([20, 20, 20, 20, 20, 20])
            surf_s = _ival(f0[0]) if len(f0) > 0 else 0
            surf_m = _ival(f0[1]) if len(f0) > 1 else 0
            istf = _ival(f0[2]) if len(f0) > 2 else 0
            igap = _ival(f0[3]) if len(f0) > 3 else 0
            ibag = _ival(f0[4]) if len(f0) > 4 else 0
            idel = _ival(f0[5]) if len(f0) > 5 else 0
        else:
            f0 = cards[0].cut("INTER_TYPE23_1")
            surf_s = _ival(f0[0]) if len(f0) > 0 else 0
            surf_m = _ival(f0[1]) if len(f0) > 1 else 0
            istf = _ival(f0[2]) if len(f0) > 2 else 0
            igap = _ival(f0[4]) if len(f0) > 4 else 0
            ibag = _ival(f0[6]) if len(f0) > 6 else 0
            idel = _ival(f0[7]) if len(f0) > 7 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f1 = cards[1].cut("INTER_TYPE23_2")
            fscale_gap = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
            gap_max = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
            fpenmax = _fval(f1[2], 1.0) if len(f1) > 2 and _fval(f1[2], 0.0) > 0.0 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f2 = cards[2].cut("INTER_TYPE23_3")
            stmin = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            stmax = _fval(f2[1], 1e30) if len(f2) > 1 and _fval(f2[1], 0.0) > 0.0 else 1e30

        if len(cards) > 3 and not cards[3].is_blank:
            f3 = cards[3].cut("INTER_TYPE23_4")
            stfac = _fval(f3[0], 1.0) if len(f3) > 0 else 1.0
            fric = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            gap_min = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            tstart = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
            tstop = _fval(f3[4], 1e30) if len(f3) > 4 and _fval(f3[4], 0.0) > 0.0 else 1e30

        if len(cards) > 4 and not cards[4].is_blank:
            f4 = cards[4].cut("INTER_TYPE23_5")
            inactiv = _ival(f4[6]) if len(f4) > 6 else 0
            viss = _fval(f4[7], 0.05) if len(f4) > 7 else 0.05
            sort_fact = _fval(f4[9], 0.2) if len(f4) > 9 else 0.2

        if len(cards) > 5 and not cards[5].is_blank:
            f5 = cards[5].cut("INTER_TYPE23_6")
            ifric = _ival(f5[0]) if len(f5) > 0 else 0
            ifiltr = _ival(f5[1]) if len(f5) > 1 else 0
            xfreq = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0

        card_idx = 6
        if ifric > 0 and len(cards) > card_idx and not cards[card_idx].is_blank:
            fc1 = cards[card_idx].cut("INTER_TYPE23_7") if cards[card_idx].length >= 100 else _fixed_vals(cards[card_idx], [20, 20, 20, 20, 20])
            c1 = _fval(fc1[0], 0.0)
            c2 = _fval(fc1[1], 0.0) if len(fc1) > 1 else 0.0
            c3 = _fval(fc1[2], 0.0) if len(fc1) > 2 else 0.0
            c4 = _fval(fc1[3], 0.0) if len(fc1) > 3 else 0.0
            c5 = _fval(fc1[4], 0.0) if len(fc1) > 4 else 0.0
            card_idx += 1
        if ifric > 1 and len(cards) > card_idx and not cards[card_idx].is_blank:
            fc2 = cards[card_idx].cut("INTER_TYPE23_8") if cards[card_idx].length >= 20 else _fixed_vals(cards[card_idx], [20])
            c6 = _fval(fc2[0], 0.0)
    else:
        t0 = cards[0].tokens()
        surf_s = int(float(t0[0])) if len(t0) > 0 else 0
        surf_m = int(float(t0[1])) if len(t0) > 1 else 0
        istf = int(float(t0[2])) if len(t0) > 2 else 0
        igap = int(float(t0[3])) if len(t0) > 3 else 0
        ibag = int(float(t0[4])) if len(t0) > 4 else 0
        idel = int(float(t0[5])) if len(t0) > 5 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t1 = cards[1].tokens()
            fscale_gap = float(t1[0]) if len(t1) > 0 else 1.0
            gap_max = float(t1[1]) if len(t1) > 1 else 0.0
            fpenmax = float(t1[2]) if len(t1) > 2 and float(t1[2]) > 0.0 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t2 = cards[2].tokens()
            stmin = float(t2[0]) if len(t2) > 0 else 0.0
            stmax = float(t2[1]) if len(t2) > 1 and float(t2[1]) > 0.0 else 1e30

        if len(cards) > 3 and not cards[3].is_blank:
            t3 = cards[3].tokens()
            stfac = float(t3[0]) if len(t3) > 0 else 1.0
            fric = float(t3[1]) if len(t3) > 1 else 0.0
            gap_min = float(t3[2]) if len(t3) > 2 else 0.0
            tstart = float(t3[3]) if len(t3) > 3 else 0.0
            tstop = float(t3[4]) if len(t3) > 4 and float(t3[4]) > 0.0 else 1e30

        if len(cards) > 4 and not cards[4].is_blank:
            t4 = cards[4].tokens()
            inactiv = int(float(t4[1])) if len(t4) > 1 else 0
            viss = float(t4[2]) if len(t4) > 2 else 0.05
            sort_fact = float(t4[3]) if len(t4) > 3 else 0.2

        if len(cards) > 5 and not cards[5].is_blank:
            t5 = cards[5].tokens()
            ifric = int(float(t5[0])) if len(t5) > 0 else 0
            ifiltr = int(float(t5[1])) if len(t5) > 1 else 0
            xfreq = float(t5[2]) if len(t5) > 2 else 0.0

        card_idx = 6
        if ifric > 0 and len(cards) > card_idx and not cards[card_idx].is_blank:
            tc1 = cards[card_idx].tokens()
            c1 = float(tc1[0]) if len(tc1) > 0 else 0.0
            c2 = float(tc1[1]) if len(tc1) > 1 else 0.0
            c3 = float(tc1[2]) if len(tc1) > 2 else 0.0
            c4 = float(tc1[3]) if len(tc1) > 3 else 0.0
            c5 = float(tc1[4]) if len(tc1) > 4 else 0.0
            card_idx += 1
        if ifric > 1 and len(cards) > card_idx and not cards[card_idx].is_blank:
            tc2 = cards[card_idx].tokens()
            c6 = float(tc2[0]) if len(tc2) > 0 else 0.0

    model.interfaces.append(Interface(
        id=block.user_id,
        type=23,
        surf_id=surf_m,
        surf_id1=surf_s,
        istf=istf,
        igap=igap,
        ibag=ibag,
        idel=idel,
        fscale_gap=fscale_gap,
        gap_scale=fscale_gap,
        gap_max=gap_max,
        fpenmax=fpenmax,
        stmin=stmin,
        stmax=stmax,
        stfac=stfac,
        fric=fric,
        gap=gap_min,
        gap_min=gap_min,
        tstart=tstart,
        tstop=tstop,
        inactiv=inactiv,
        viss=viss,
        stiff_dc=viss,
        sort_fact=sort_fact,
        ifric=ifric,
        mfrot=ifric,
        ifiltr=ifiltr,
        ifq=ifiltr,
        xfreq=xfreq,
        fric_c=(c1, c2, c3, c4, c5, c6),
        title=title,
    ))




def read_def_inter(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DEF_INTER/type``, ``/DEFAULT/INTER/type`` (M99/M101) — Global contact defaults."""
    subtype = block.parts[-1].upper() if len(block.parts) > 1 else "TYPE25"
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        return
    c = cards[0]

    def _iv_from(vals, idx, default=0):
        try:
            return int(float(vals[idx]))
        except (IndexError, ValueError):
            return default

    if subtype == "TYPE2":
        vals = c.cut("DEF_INTER_2") if block.fixed else c.tokens()
        entry = {
            'idel': _iv_from(vals, 0), 'icurv': _iv_from(vals, 1), 'icurv_r': _iv_from(vals, 2),
            'icurv_s': _iv_from(vals, 3), 'ishape': _iv_from(vals, 4), 'iedge': _iv_from(vals, 5),
        }
    elif subtype == "TYPE7":
        vals = c.cut("DEF_INTER_7") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1), 'ibag': _iv_from(vals, 2),
            'idel7': _iv_from(vals, 3), 'ikrem': _iv_from(vals, 4), 'irem7i2': _iv_from(vals, 5),
            'inactiv': _iv_from(vals, 6), 'iform': _iv_from(vals, 7),
        }
    elif subtype == "TYPE11":
        vals0 = cards[0].cut("DEF_INTER_11") if block.fixed else cards[0].tokens()
        iform = _iv_from(vals0, 4, 1) if len(vals0) > 4 and str(vals0[4]).strip() else 1
        inactiv = _iv_from(vals0, 5, 1000) if len(vals0) > 5 and str(vals0[5]).strip() else 1000
        if len(cards) > 1 and not cards[1].is_blank:
            vals1 = cards[1].tokens()
            if vals1:
                iform = _iv_from(vals1, 0, iform)
        if len(cards) > 2 and not cards[2].is_blank:
            vals2 = cards[2].tokens()
            if vals2:
                inactiv = _iv_from(vals2, 0, inactiv)
        entry = {
            'istf': _iv_from(vals0, 0, 5), 'igap': _iv_from(vals0, 1, 1000),
            'ikrem': _iv_from(vals0, 2, 1), 'idel11': _iv_from(vals0, 3, 1000),
            'iform': iform, 'inactiv': inactiv,
        }
    elif subtype == "TYPE19":
        vals0 = cards[0].cut("DEF_INTER_19") if block.fixed else cards[0].tokens()
        inactiv = _iv_from(vals0, 6, 1000) if len(vals0) > 6 and str(vals0[6]).strip() else 1000
        iform = _iv_from(vals0, 7, 1) if len(vals0) > 7 and str(vals0[7]).strip() else 1
        if len(cards) > 1 and not cards[1].is_blank:
            vals1 = cards[1].tokens()
            if vals1:
                inactiv = _iv_from(vals1, 0, inactiv)
        if len(cards) > 2 and not cards[2].is_blank:
            vals2 = cards[2].tokens()
            if vals2:
                iform = _iv_from(vals2, 0, iform)
        entry = {
            'istf': _iv_from(vals0, 0, 1000), 'igap': _iv_from(vals0, 1, 1000),
            'iedge': _iv_from(vals0, 2, 2), 'ibag': _iv_from(vals0, 3, 2),
            'idel': _iv_from(vals0, 4, 1000), 'idel7': _iv_from(vals0, 4, 1000),
            'icurv': _iv_from(vals0, 5, 0),
            'inactiv': inactiv, 'iform': iform,
        }
    elif subtype == "TYPE18":
        vals = c.cut("DEF_INTER_18") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'multimp': _iv_from(vals, 1), 'ibag': _iv_from(vals, 2),
            'idel18': _iv_from(vals, 3), 'igap': _iv_from(vals, 4), 'iauto': _iv_from(vals, 5),
        }
    elif subtype == "TYPE8":
        vals = c.cut("DEF_INTER_8") if block.fixed else c.tokens()
        entry = {
            'iform1': _iv_from(vals, 0),
        }
    elif subtype == "TYPE24":
        vals = c.cut("DEF_INTER_24") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1), 'irem_i2': _iv_from(vals, 2),
            'idel': _iv_from(vals, 3), 'itied': _iv_from(vals, 4), 'ishape': _iv_from(vals, 5),
            'irs': _iv_from(vals, 6), 'iedge': _iv_from(vals, 7),
            'ipen': _iv_from(vals, 8) if len(vals) > 8 else 0,
        }
    elif subtype == "TYPE9":
        vals = c.cut("DEF_INTER_9") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1), 'ibag': _iv_from(vals, 2),
            'idel': _iv_from(vals, 3), 'inactiv': _iv_from(vals, 4),
        }
    elif subtype == "TYPE10":
        vals = c.cut("DEF_INTER_10") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'multimp': _iv_from(vals, 1), 'idel10': _iv_from(vals, 2),
            'itied': _iv_from(vals, 3),
        }
    elif subtype == "TYPE16":
        vals = c.cut("DEF_INTER_16") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1),
        }
    elif subtype == "TYPE17":
        vals = c.cut("DEF_INTER_17") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1), 'iform': _iv_from(vals, 2),
        }
    else:  # TYPE25 or default
        vals0 = cards[0].cut("DEF_INTER_25") if block.fixed else cards[0].tokens()
        itied = _iv_from(vals0, 4, 0)
        ishape = _iv_from(vals0, 5, 0)
        irs = _iv_from(vals0, 6, 1000)
        if len(cards) > 1 and not cards[1].is_blank:
            vals1 = cards[1].tokens()
            itied = _iv_from(vals1, 0, itied)
            ishape = _iv_from(vals1, 1, ishape)
        if len(cards) > 2 and not cards[2].is_blank:
            vals2 = cards[2].tokens()
            irs = _iv_from(vals2, 0, irs)
        entry = {
            'istf': _iv_from(vals0, 0, 0), 'igap': _iv_from(vals0, 1, 0), 'irem_i2': _iv_from(vals0, 2, 0),
            'idel': _iv_from(vals0, 3, 0), 'itied': itied, 'ishape': ishape,
            'irs': irs, 'iedge': _iv_from(vals0, 7, 1000) if len(vals0) > 7 else 1000,
        }
    model.def_inter[subtype] = entry
    if subtype == "TYPE25":
        model.def_inter.update(entry)
    if subtype == "TYPE11":
        from ...model.entities import DefInterType11
        model.def_inter_type11 = DefInterType11(
            istf=entry.get('istf', 5), igap=entry.get('igap', 1000),
            ikrem=entry.get('ikrem', 1), noddel11=entry.get('idel11', 1000),
            iform=entry.get('iform', 1), inactiv=entry.get('inactiv', 1000),
        )
    elif subtype == "TYPE19":
        from ...model.entities import DefInterType19
        model.def_inter_type19 = DefInterType19(
            istf=entry.get('istf', 1000), igap=entry.get('igap', 1000),
            iedge=entry.get('iedge', 2), ibag=entry.get('ibag', 2),
            idel7=entry.get('idel7', 1000), icurv=entry.get('icurv', 0),
            inactiv=entry.get('inactiv', 1000), iform=entry.get('iform', 1),
        )
    elif subtype == "TYPE2":
        from ...model.entities import DefInterType2
        model.def_inter_type2 = DefInterType2(
            istf=entry.get('istf', _iv_from(vals, 0)),
            igap=entry.get('igap', _iv_from(vals, 1)),
            iref=entry.get('iref', _iv_from(vals, 2)),
            params=entry,
        )
    elif subtype == "TYPE25":
        from ...model.entities import DefInterType25
        model.def_inter_type25 = DefInterType25(
            istf=entry.get('istf', 0), igap=entry.get('igap', 0),
            irem_i2=entry.get('irem_i2', 0), idel=entry.get('idel', 0),
            itied=entry.get('itied', 0), ishape=entry.get('ishape', 0),
            irs=entry.get('irs', 1000),
        )




def read_lagmul(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LAGMUL`` or ``/LAGMUL/OPTION`` (M131): Global Lagrange multiplier options.

    Fortran origin: ``starter/source/tools/lagmul/hm_read_lagmul.F``.
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "GEAR":
        read_gear(block, model, log)
        return
    elif sub == "RACK":
        read_rack(block, model, log)
        return
    elif sub == "DIFF":
        read_diff(block, model, log)
        return
    elif sub in ("BALL_JOINT", "BALL", "SPHERICAL", "SPHERICAL_JOINT"):
        read_ball_joint(block, model, log)
        return
    elif sub in ("PIN_JOINT", "PIN", "REVOLUTE", "HINGE", "HINGE_JOINT"):
        read_pin_joint(block, model, log)
        return
    elif sub in ("SLIDER", "SLIDE", "PRISMATIC", "TRANSLATIONAL", "TRANSLATIONAL_JOINT"):
        read_slider_joint(block, model, log)
        return
    elif sub in ("CYL_JOINT", "CYLINDER_JOINT", "CYL", "CYLINDER", "CYLINDRICAL", "CYLINDRICAL_JOINT"):
        read_cyl_joint(block, model, log)
        return
    elif sub in ("PLANAR", "PLANAR_JOINT", "PLANE"):
        read_planar_joint(block, model, log)
        return
    elif sub in ("CARDAN", "UNIVERSAL", "UNIVERSAL_JOINT", "CARDAN_JOINT"):
        read_cardan_joint(block, model, log)
        return
    elif sub in ("RIGID", "RIGID_JOINT", "RIGID_LINK"):
        read_rigid_joint(block, model, log)
        return
    elif sub in ("SCREW", "SCREW_JOINT", "HELICAL", "HELICAL_JOINT"):
        read_screw_joint(block, model, log)
        return
    elif sub in ("CV_JOINT", "CONSTANT_VELOCITY", "CV", "HOMOKINETIC"):
        read_cv_joint(block, model, log)
        return
    elif sub in ("TRIPOD", "TRIPOD_JOINT", "PLUNGING_CV"):
        read_tripod_joint(block, model, log)
        return
    elif sub in ("BEVEL_GEAR", "BEVEL_GEAR_JOINT", "BEVEL"):
        read_bevel_gear_joint(block, model, log)
        return
    elif sub in ("WORM_GEAR", "WORM_GEAR_JOINT", "WORM"):
        read_worm_gear_joint(block, model, log)
        return
    elif sub in ("HYPOID_GEAR", "HYPOID_GEAR_JOINT", "HYPOID"):
        read_hypoid_gear_joint(block, model, log)
        return
    elif sub in ("INLINE", "INLINE_JOINT", "LINE"):
        read_inline_joint(block, model, log)
        return
    elif sub in ("PARALLEL", "PARALLEL_JOINT"):
        read_parallel_joint(block, model, log)
        return
    elif sub in ("PERPENDICULAR", "PERPENDICULAR_JOINT", "ORTHOGONAL_JOINT", "PERP"):
        read_perpendicular_joint(block, model, log)
        return
    elif sub in ("GIMBAL", "GIMBAL_JOINT", "UNIVERSAL_GIMBAL"):
        read_gimbal_joint(block, model, log)
        return
    elif sub in ("DISTANCE", "DISTANCE_JOINT", "CONST_DIST", "CONST_DISTANCE"):
        read_distance_joint(block, model, log)
        return
    elif sub in ("SLOT", "SLOT_JOINT"):
        read_slot_joint(block, model, log)
        return

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    lagmod = 1
    lagopt = 1
    tol = 1e-11
    alpha = 5e-4
    alpha_s = 0.0

    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("LAGMUL_1")
            lagmod = _ival(f[0], 1)
            lagopt = _ival(f[1], 1)
            tol = _fval(f[2], 1e-11)
            alpha = _fval(f[3], 5e-4)
            alpha_s = _fval(f[4], 0.0)
        else:
            toks = cards[0].tokens()
            lagmod = int(float(toks[0])) if len(toks) > 0 else 1
            lagopt = int(float(toks[1])) if len(toks) > 1 else 1
            tol = float(toks[2]) if len(toks) > 2 else 1e-11
            alpha = float(toks[3]) if len(toks) > 3 else 5e-4
            alpha_s = float(toks[4]) if len(toks) > 4 else 0.0

    from ...model.entities import LagmulGlobal
    model.lagmul_global = LagmulGlobal(
        lagmod=lagmod, lagopt=lagopt, tol=tol, alpha=alpha, alpha_s=alpha_s
    )




def read_lagmul_harmonic_drive(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HARMONIC_DRIVE/id`` or ``/LAGMUL/HARMONIC_DRIVE/id`` (M259): Harmonic drive strain wave gear kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HARMONIC_DRIVE/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, ratio, stiff, skew_id, tol = 0, 0, 0, 100.0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("HARMONIC_DRIVE_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        ratio = _fval(f1[3], 100.0) if len(f1) > 3 and f1[3].strip() else 100.0
        stiff = _fval(f1[4], 1e6) if len(f1) > 4 and f1[4].strip() else 1e6
        skew_id = _ival(f1[5], 0) if len(f1) > 5 else 0
        tol = _fval(f1[6], 1e-6) if len(f1) > 6 and f1[6].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HARMONIC_DRIVE_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        ratio = float(toks1[3]) if len(toks1) > 3 else 100.0
        stiff = float(toks1[4]) if len(toks1) > 4 else 1e6
        skew_id = int(float(toks1[5])) if len(toks1) > 5 else 0
        tol = float(toks1[6]) if len(toks1) > 6 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulHarmonicDrive
    model.lagmul_harmonic_drives[block.user_id] = LagmulHarmonicDrive(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        ratio=ratio, stiff=stiff, axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
        skew_id=skew_id, tol=tol
    )




def read_lagmul_cycloidal_drive(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CYCLOIDAL_DRIVE/id`` or ``/LAGMUL/CYCLOIDAL_DRIVE/id`` (M260): Cycloidal speed reducer kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CYCLOIDAL_DRIVE/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, ratio, stiff, skew_id, tol = 0, 0, 0, 29.0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("CYCLOIDAL_DRIVE_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        ratio = _fval(f1[3], 29.0) if len(f1) > 3 and f1[3].strip() else 29.0
        stiff = _fval(f1[4], 1e6) if len(f1) > 4 and f1[4].strip() else 1e6
        skew_id = _ival(f1[5], 0) if len(f1) > 5 else 0
        tol = _fval(f1[6], 1e-6) if len(f1) > 6 and f1[6].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CYCLOIDAL_DRIVE_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        ratio = float(toks1[3]) if len(toks1) > 3 else 29.0
        stiff = float(toks1[4]) if len(toks1) > 4 else 1e6
        skew_id = int(float(toks1[5])) if len(toks1) > 5 else 0
        tol = float(toks1[6]) if len(toks1) > 6 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulCycloidalDrive
    model.lagmul_cycloidal_drives[block.user_id] = LagmulCycloidalDrive(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        ratio=ratio, stiff=stiff, axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
        skew_id=skew_id, tol=tol
    )




def read_lagmul_rack_pinion(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RACK_AND_PINION/id`` or ``/LAGMUL/RACK_AND_PINION/id`` (M261): Rack and pinion transmission kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RACK_AND_PINION/{block.user_id}: missing data card", block.source)
        return

    node1, node2, pitch_radius, stiff, skew_id, tol = 0, 0, 10.0, 1e6, 0, 1e-6
    axis_rot_x, axis_rot_y, axis_rot_z = 0.0, 0.0, 1.0
    axis_tra_x, axis_tra_y, axis_tra_z = 1.0, 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("RACK_PINION_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        pitch_radius = _fval(f1[2], 10.0) if len(f1) > 2 and f1[2].strip() else 10.0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("RACK_PINION_2")
            axis_rot_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_rot_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_rot_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            axis_tra_x = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            axis_tra_y = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
            axis_tra_z = _fval(f2[5], 0.0) if len(f2) > 5 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        pitch_radius = float(toks1[2]) if len(toks1) > 2 else 10.0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_rot_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_rot_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_rot_z = float(toks2[2]) if len(toks2) > 2 else 1.0
            axis_tra_x = float(toks2[3]) if len(toks2) > 3 else 1.0
            axis_tra_y = float(toks2[4]) if len(toks2) > 4 else 0.0
            axis_tra_z = float(toks2[5]) if len(toks2) > 5 else 0.0

    from ...model.entities import LagmulRackPinion
    model.lagmul_rack_pinions[block.user_id] = LagmulRackPinion(
        id=block.user_id, title=title, node1=node1, node2=node2,
        pitch_radius=pitch_radius, stiff=stiff, skew_id=skew_id, tol=tol,
        axis_rot_x=axis_rot_x, axis_rot_y=axis_rot_y, axis_rot_z=axis_rot_z,
        axis_tra_x=axis_tra_x, axis_tra_y=axis_tra_y, axis_tra_z=axis_tra_z
    )




def read_lagmul_screw_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCREW_JOINT/id`` or ``/LAGMUL/SCREW_JOINT/id`` (M262): Screw and leadscrew transmission kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCREW_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, lead_pitch, stiff, skew_id, tol = 0, 0, 5.0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("SCREW_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        lead_pitch = _fval(f1[2], 5.0) if len(f1) > 2 and f1[2].strip() else 5.0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SCREW_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        lead_pitch = float(toks1[2]) if len(toks1) > 2 else 5.0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulScrewJoint
    model.lagmul_screw_joints[block.user_id] = LagmulScrewJoint(
        id=block.user_id, title=title, node1=node1, node2=node2,
        lead_pitch=lead_pitch, stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_differential_gear(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIFFERENTIAL_GEAR/id`` or ``/LAGMUL/DIFFERENTIAL_GEAR/id`` (M263): Differential gear train kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIFFERENTIAL_GEAR/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, ratio, stiff, skew_id, tol = 0, 0, 0, 1.0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("DIFFERENTIAL_GEAR_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        ratio = _fval(f1[3], 1.0) if len(f1) > 3 and f1[3].strip() else 1.0
        stiff = _fval(f1[4], 1e6) if len(f1) > 4 and f1[4].strip() else 1e6
        skew_id = _ival(f1[5], 0) if len(f1) > 5 else 0
        tol = _fval(f1[6], 1e-6) if len(f1) > 6 and f1[6].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DIFFERENTIAL_GEAR_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        ratio = float(toks1[3]) if len(toks1) > 3 else 1.0
        stiff = float(toks1[4]) if len(toks1) > 4 else 1e6
        skew_id = int(float(toks1[5])) if len(toks1) > 5 else 0
        tol = float(toks1[6]) if len(toks1) > 6 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    if ratio == 0.0:
        ratio = 1.0

    from ...model.entities import LagmulDifferentialGear
    model.lagmul_differential_gears[block.user_id] = LagmulDifferentialGear(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        ratio=ratio, stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_transfer_case(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TRANSFER_CASE/id`` or ``/LAGMUL/TRANSFER_CASE/id`` (M264): 4WD/AWD Transfer case transmission kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TRANSFER_CASE/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, front_split, stiff, skew_id, tol = 0, 0, 0, 0.5, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("TRANSFER_CASE_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        front_split = _fval(f1[3], 0.5) if len(f1) > 3 and f1[3].strip() else 0.5
        stiff = _fval(f1[4], 1e6) if len(f1) > 4 and f1[4].strip() else 1e6
        skew_id = _ival(f1[5], 0) if len(f1) > 5 else 0
        tol = _fval(f1[6], 1e-6) if len(f1) > 6 and f1[6].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("TRANSFER_CASE_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        front_split = float(toks1[3]) if len(toks1) > 3 else 0.5
        stiff = float(toks1[4]) if len(toks1) > 4 else 1e6
        skew_id = int(float(toks1[5])) if len(toks1) > 5 else 0
        tol = float(toks1[6]) if len(toks1) > 6 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    if front_split == 0.0:
        front_split = 0.5

    from ...model.entities import LagmulTransferCase
    model.lagmul_transfer_cases[block.user_id] = LagmulTransferCase(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        front_split=front_split, stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_torque_split_gear(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TORQUE_SPLIT_GEAR/id`` or ``/LAGMUL/TORQUE_SPLIT_GEAR/id`` (M265): Dual-output torque splitter / PTO kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TORQUE_SPLIT_GEAR/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, split_ratio, stiff, skew_id, tol = 0, 0, 0, 0.5, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("TORQUE_SPLIT_GEAR_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        split_ratio = _fval(f1[3], 0.5) if len(f1) > 3 and f1[3].strip() else 0.5
        stiff = _fval(f1[4], 1e6) if len(f1) > 4 and f1[4].strip() else 1e6
        skew_id = _ival(f1[5], 0) if len(f1) > 5 else 0
        tol = _fval(f1[6], 1e-6) if len(f1) > 6 and f1[6].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("TORQUE_SPLIT_GEAR_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        split_ratio = float(toks1[3]) if len(toks1) > 3 else 0.5
        stiff = float(toks1[4]) if len(toks1) > 4 else 1e6
        skew_id = int(float(toks1[5])) if len(toks1) > 5 else 0
        tol = float(toks1[6]) if len(toks1) > 6 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    if split_ratio == 0.0:
        split_ratio = 0.5

    from ...model.entities import LagmulTorqueSplitGear
    model.lagmul_torque_split_gears[block.user_id] = LagmulTorqueSplitGear(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        split_ratio=split_ratio, stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_geneva_drive(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GENEVA_DRIVE/id`` or ``/LAGMUL/GENEVA_DRIVE/id`` (M266): Geneva drive intermittent rotary indexing kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GENEVA_DRIVE/{block.user_id}: missing data card", block.source)
        return

    node1, node2, num_slots, stiff, skew_id, tol = 0, 0, 4, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("GENEVA_DRIVE_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        num_slots = _ival(f1[2], 4) if len(f1) > 2 and f1[2].strip() else 4
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("GENEVA_DRIVE_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        num_slots = int(float(toks1[2])) if len(toks1) > 2 else 4
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    if num_slots < 3:
        num_slots = 4

    from ...model.entities import LagmulGenevaDrive
    model.lagmul_geneva_drives[block.user_id] = LagmulGenevaDrive(
        id=block.user_id, title=title, node1=node1, node2=node2,
        num_slots=num_slots, stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_scotch_yoke(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCOTCH_YOKE/id`` or ``/LAGMUL/SCOTCH_YOKE/id`` (M267): Scotch yoke pure harmonic rotary-to-linear conversion kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCOTCH_YOKE/{block.user_id}: missing data card", block.source)
        return

    node1, node2, crank_radius, stiff, skew_id, tol = 0, 0, 1.0, 1e6, 0, 1e-6
    rot_x, rot_y, rot_z = 0.0, 0.0, 1.0
    trans_x, trans_y, trans_z = 1.0, 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("SCOTCH_YOKE_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        crank_radius = _fval(f1[2], 1.0) if len(f1) > 2 and f1[2].strip() else 1.0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SCOTCH_YOKE_2")
            rot_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            rot_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            rot_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            trans_x = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            trans_y = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
            trans_z = _fval(f2[5], 0.0) if len(f2) > 5 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        crank_radius = float(toks1[2]) if len(toks1) > 2 else 1.0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            rot_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            rot_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            rot_z = float(toks2[2]) if len(toks2) > 2 else 1.0
            trans_x = float(toks2[3]) if len(toks2) > 3 else 1.0
            trans_y = float(toks2[4]) if len(toks2) > 4 else 0.0
            trans_z = float(toks2[5]) if len(toks2) > 5 else 0.0

    if crank_radius == 0.0:
        crank_radius = 1.0

    from ...model.entities import LagmulScotchYoke
    model.lagmul_scotch_yokes[block.user_id] = LagmulScotchYoke(
        id=block.user_id, title=title, node1=node1, node2=node2,
        crank_radius=crank_radius, stiff=stiff, skew_id=skew_id, tol=tol,
        rot_x=rot_x, rot_y=rot_y, rot_z=rot_z,
        trans_x=trans_x, trans_y=trans_y, trans_z=trans_z
    )




def read_lagmul_oldham_coupling(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/OLDHAM_COUPLING/id`` or ``/LAGMUL/OLDHAM_COUPLING/id`` (M268): Oldham coupling parallel offset shaft kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/OLDHAM_COUPLING/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_dir = 1
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        if len(cards) == 1:
            fj = cards[0].cut("OLDHAM_JOINT_1")
            node1 = _ival(fj[0]) if len(fj) > 0 else 0
            node2 = _ival(fj[1]) if len(fj) > 1 else 0
            axis_dir = _ival(fj[2], 1) if len(fj) > 2 else 1
            node3 = axis_dir
            skew_id = _ival(fj[3], 0) if len(fj) > 3 else 0
            tol = _fval(fj[4], 1e-6) if len(fj) > 4 else 1e-6
            stiff = 1e6
        else:
            f1 = cards[0].cut("OLDHAM_COUPLING_1")
            node1 = _ival(f1[0]) if len(f1) > 0 else 0
            node2 = _ival(f1[1]) if len(f1) > 1 else 0
            node3 = _ival(f1[2]) if len(f1) > 2 else 0
            axis_dir = node3 if node3 in (1, 2, 3) else 1
            stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
            skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
            tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

            if not cards[1].is_blank:
                f2 = cards[1].cut("OLDHAM_COUPLING_2")
                axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        if len(cards) == 1 and len(toks1) == 5:
            node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
            node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
            axis_dir = int(float(toks1[2])) if len(toks1) > 2 else 1
            node3 = axis_dir
            skew_id = int(float(toks1[3])) if len(toks1) > 3 else 0
            tol = float(toks1[4]) if len(toks1) > 4 else 1e-6
            stiff = 1e6
        else:
            node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
            node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
            node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
            axis_dir = node3 if node3 in (1, 2, 3) else 1
            stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
            skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
            tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

            if len(cards) > 1 and not cards[1].is_blank:
                toks2 = cards[1].tokens()
                axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
                axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
                axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulOldhamCoupling, OldhamJoint
    model.lagmul_oldham_couplings[block.user_id] = LagmulOldhamCoupling(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )
    model.oldham_joints[block.user_id] = OldhamJoint(
        id=block.user_id, title=title, node1=node1, node2=node2,
        axis_dir=axis_dir,
        skew_id=skew_id, tol=tol
    )




def read_lagmul_schmidt_coupling(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCHMIDT_COUPLING/id`` or ``/LAGMUL/SCHMIDT_COUPLING/id`` (M269): Schmidt (double-Cardan) coupling constant-velocity shaft joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCHMIDT_COUPLING/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("SCHMIDT_COUPLING_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SCHMIDT_COUPLING_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulSchmidtCoupling
    model.lagmul_schmidt_couplings[block.user_id] = LagmulSchmidtCoupling(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_rzeppa_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RZEPPA_JOINT/id`` or ``/LAGMUL/RZEPPA_JOINT/id`` (M270): Rzeppa constant-velocity ball joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RZEPPA_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("RZEPPA_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("RZEPPA_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulRzeppaJoint
    model.lagmul_rzeppa_joints[block.user_id] = LagmulRzeppaJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_birfield_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BIRFIELD_JOINT/id`` or ``/LAGMUL/BIRFIELD_JOINT/id`` (M271): Birfield (plunging CV) joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BIRFIELD_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("BIRFIELD_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BIRFIELD_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulBirfieldJoint
    model.lagmul_birfield_joints[block.user_id] = LagmulBirfieldJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_tripod_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TRIPOD_JOINT/id`` or ``/LAGMUL/TRIPOD_JOINT/id`` (M272): Tripod (tulip/spider) joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TRIPOD_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("TRIPOD_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("TRIPOD_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    plunge_limit = 0.0
    if len(cards) == 1 and stiff <= 1e5:
        axis_dir = node3 if node3 in (1, 2, 3) else 1
        real_skew = int(stiff)
        plunge_limit = float(skew_id) if skew_id != 0 else 0.0
        skew_id = real_skew
        stiff = 1e6
    else:
        axis_dir = node3 if node3 in (1, 2, 3) else 1

    from ...model.entities import LagmulTripodJoint, TripodJoint
    model.lagmul_tripod_joints[block.user_id] = LagmulTripodJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )
    model.tripod_joints[block.user_id] = TripodJoint(
        id=block.user_id, title=title, node1=node1, node2=node2,
        axis_dir=axis_dir,
        skew_id=skew_id, plunge_limit=plunge_limit, tol=tol
    )




def read_lagmul_hooke_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HOOKE_JOINT/id`` or ``/LAGMUL/HOOKE_JOINT/id`` (M273): Hooke (universal/Cardan) joint constraint."""
    title, cards = _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HOOKE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if len(cards) == 1 and not block.fixed:
        toks1 = cards[0].tokens()
        if len(toks1) == 5:
            node1 = int(float(toks1[0].rstrip(',')))
            node2 = int(float(toks1[1].rstrip(',')))
            node3 = int(float(toks1[2].rstrip(',')))
            skew_id = int(float(toks1[3].rstrip(',')))
            tol = float(toks1[4].rstrip(','))
        else:
            node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
            node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
            node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
            stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
            skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
            tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6
    elif block.fixed:
        if len(cards) == 1:
            f1 = cards[0].cut("CARDAN_1") if "CARDAN_1" in LAYOUTS else cards[0].fields(10, 5)
            node1 = _ival(f1[0]) if len(f1) > 0 else 0
            node2 = _ival(f1[1]) if len(f1) > 1 else 0
            node3 = _ival(f1[2]) if len(f1) > 2 else 0
            skew_id = _ival(f1[3]) if len(f1) > 3 else 0
            tol = _fval(f1[4], 1e-6) if len(f1) > 4 and f1[4].strip() else 1e-6
        else:
            f1 = cards[0].cut("HOOKE_JOINT_1")
            node1 = _ival(f1[0]) if len(f1) > 0 else 0
            node2 = _ival(f1[1]) if len(f1) > 1 else 0
            node3 = _ival(f1[2]) if len(f1) > 2 else 0
            stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
            skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
            tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("HOOKE_JOINT_2")
                axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulHookeJoint, CardanJoint
    model.lagmul_hooke_joints[block.user_id] = LagmulHookeJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )
    model.cardan_joints[block.user_id] = CardanJoint(
        id=block.user_id, title=title, node1=node1, node2=node2,
        axis_dir=node3 if node3 in (1, 2, 3) else 1,
        skew_id=skew_id, tol=tol
    )




def read_lagmul_tracta_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TRACTA_JOINT/id`` or ``/LAGMUL/TRACTA_JOINT/id`` (M274): Tracta (sliding-yoke) joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TRACTA_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("TRACTA_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("TRACTA_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulTractaJoint
    model.lagmul_tracta_joints[block.user_id] = LagmulTractaJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_thompson_coupling(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/THOMPSON_COUPLING/id`` or ``/LAGMUL/THOMPSON_COUPLING/id`` (M275): Thompson constant-velocity joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/THOMPSON_COUPLING/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("THOMPSON_COUPLING_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("THOMPSON_COUPLING_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 0 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulThompsonCoupling
    model.lagmul_thompson_couplings[block.user_id] = LagmulThompsonCoupling(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_weiss_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WEISS_JOINT/id`` or ``/LAGMUL/WEISS_JOINT/id`` (M276): Weiss constant-velocity ball-and-groove joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WEISS_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("WEISS_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WEISS_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 0 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulWeissJoint
    model.lagmul_weiss_joints[block.user_id] = LagmulWeissJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_tripod_ball_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TRIPOD_BALL_JOINT/id`` or ``/LAGMUL/TRIPOD_BALL_JOINT/id`` (M277): Tripod-ball kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TRIPOD_BALL_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("TRIPOD_BALL_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("TRIPOD_BALL_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2])) if len(toks1) > 2 else 0
        stiff = float(toks1[3]) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4])) if len(toks1) > 4 else 0
        tol = float(toks1[5]) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0]) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1]) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2]) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulTripodBallJoint
    model.lagmul_tripod_ball_joints[block.user_id] = LagmulTripodBallJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_clevis_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CLEVIS_JOINT/id`` or ``/LAGMUL/CLEVIS_JOINT/id`` (M278): Clevis pin / fork-and-tang kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CLEVIS_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CLEVIS_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CLEVIS_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulClevisJoint
    model.lagmul_clevis_joints[block.user_id] = LagmulClevisJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_pin_in_slot_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PIN_IN_SLOT_JOINT/id`` or ``/LAGMUL/PIN_IN_SLOT_JOINT/id`` (M279): Pin-in-slot planar mechanism kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PIN_IN_SLOT_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PIN_IN_SLOT_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PIN_IN_SLOT_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulPinInSlotJoint
    model.lagmul_pin_in_slot_joints[block.user_id] = LagmulPinInSlotJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_slider_slot_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SLIDER_SLOT_JOINT/id`` or ``/LAGMUL/SLIDER_SLOT_JOINT/id`` (M280): Slider-slot planar mechanism kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SLIDER_SLOT_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SLIDER_SLOT_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SLIDER_SLOT_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulSliderSlotJoint
    model.lagmul_slider_slot_joints[block.user_id] = LagmulSliderSlotJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_parallel_axis_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PARALLEL_AXIS_JOINT/id`` or ``/LAGMUL/PARALLEL_AXIS_JOINT/id`` (M281): Parallel-axis slider / Oldham coupling kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PARALLEL_AXIS_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PARALLEL_AXIS_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PARALLEL_AXIS_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 0 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulParallelAxisJoint
    model.lagmul_parallel_axis_joints[block.user_id] = LagmulParallelAxisJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_cam_follower_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CAM_FOLLOWER_JOINT/id`` or ``/LAGMUL/CAM_FOLLOWER_JOINT/id`` (M282): Cam and follower profile kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CAM_FOLLOWER_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    axis_x, axis_y, axis_z = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CAM_FOLLOWER_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CAM_FOLLOWER_JOINT_2")
            axis_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            axis_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            axis_x = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_y = float(toks2[1].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulCamFollowerJoint
    model.lagmul_cam_follower_joints[block.user_id] = LagmulCamFollowerJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z
    )




def read_lagmul_screw_nut_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCREW_NUT_JOINT/id`` or ``/LAGMUL/SCREW_NUT_JOINT/id`` (M283): Lead screw and nut helical rotary-to-linear conversion kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCREW_NUT_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    pitch, lead, axis_z = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SCREW_NUT_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SCREW_NUT_JOINT_2")
            pitch = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            lead = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            pitch = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            lead = float(toks2[1].rstrip(',')) if len(toks2) > 0 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulScrewNutJoint
    model.lagmul_screw_nut_joints[block.user_id] = LagmulScrewNutJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        pitch=pitch, lead=lead, axis_z=axis_z
    )




def read_lagmul_geneva_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GENEVA_JOINT/id`` or ``/LAGMUL/GENEVA_JOINT/id`` (M284): Geneva wheel / Maltese cross intermittent rotary indexing kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GENEVA_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    num_slots, crank_radius, axis_z = 4, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("GENEVA_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("GENEVA_JOINT_2")
            num_slots = _ival(f2[0], 4) if len(f2) > 0 else 4
            crank_radius = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            num_slots = int(float(toks2[0].rstrip(','))) if len(toks2) > 0 else 4
            crank_radius = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulGenevaJoint
    model.lagmul_geneva_joints[block.user_id] = LagmulGenevaJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        num_slots=num_slots, crank_radius=crank_radius, axis_z=axis_z
    )




def read_lagmul_cable_pulley_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CABLE_PULLEY_JOINT/id`` or ``/LAGMUL/CABLE_PULLEY_JOINT/id`` (M285): Flexible cable and pulley wrapping transmission kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CABLE_PULLEY_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    pulley_radius, wrap_angle, axis_z = 0.0, 180.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CABLE_PULLEY_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CABLE_PULLEY_JOINT_2")
            pulley_radius = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            wrap_angle = _fval(f2[1], 180.0) if len(f2) > 1 and f2[1].strip() else 180.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            pulley_radius = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            wrap_angle = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 180.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulCablePulleyJoint
    model.lagmul_cable_pulley_joints[block.user_id] = LagmulCablePulleyJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        pulley_radius=pulley_radius, wrap_angle=wrap_angle, axis_z=axis_z
    )




def read_lagmul_swash_plate_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SWASH_PLATE_JOINT/id`` or ``/LAGMUL/SWASH_PLATE_JOINT/id`` (M286): Swash plate cyclic tilting and rotating kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SWASH_PLATE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    plate_radius, tilt_angle, axis_z = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SWASH_PLATE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SWASH_PLATE_JOINT_2")
            plate_radius = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            tilt_angle = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            plate_radius = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            tilt_angle = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulSwashPlateJoint
    model.lagmul_swash_plate_joints[block.user_id] = LagmulSwashPlateJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        plate_radius=plate_radius, tilt_angle=tilt_angle, axis_z=axis_z
    )




def read_lagmul_scissor_mechanism_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCISSOR_MECHANISM_JOINT/id`` or ``/LAGMUL/SCISSOR_MECHANISM_JOINT/id`` (M287): Pantograph / scissor lift planar crossing linkage kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCISSOR_MECHANISM_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    arm_length, initial_angle, axis_z = 0.0, 45.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SCISSOR_MECHANISM_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SCISSOR_MECHANISM_JOINT_2")
            arm_length = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            initial_angle = _fval(f2[1], 45.0) if len(f2) > 1 and f2[1].strip() else 45.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            arm_length = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            initial_angle = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 45.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulScissorMechanismJoint
    model.lagmul_scissor_mechanism_joints[block.user_id] = LagmulScissorMechanismJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        arm_length=arm_length, initial_angle=initial_angle, axis_z=axis_z
    )




def read_lagmul_parallelogram_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PARALLELOGRAM_JOINT/id`` or ``/LAGMUL/PARALLELOGRAM_JOINT/id`` (M288): 4-bar parallelogram kinematic linkage joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PARALLELOGRAM_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_length, link_width, axis_z = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PARALLELOGRAM_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PARALLELOGRAM_JOINT_2")
            link_length = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_width = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            axis_z = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_length = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_width = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            axis_z = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulParallelogramJoint
    model.lagmul_parallelogram_joints[block.user_id] = LagmulParallelogramJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_length=link_length, link_width=link_width, axis_z=axis_z
    )




def read_lagmul_delta_robot_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DELTA_ROBOT_JOINT/id`` or ``/LAGMUL/DELTA_ROBOT_JOINT/id`` (M289): 3-DOF parallel delta robot spatial linkage kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DELTA_ROBOT_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    upper_arm_len, forearm_len, base_radius = 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DELTA_ROBOT_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DELTA_ROBOT_JOINT_2")
            upper_arm_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            forearm_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            base_radius = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            upper_arm_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            forearm_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            base_radius = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0

    from ...model.entities import LagmulDeltaRobotJoint
    model.lagmul_delta_robot_joints[block.user_id] = LagmulDeltaRobotJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        upper_arm_len=upper_arm_len, forearm_len=forearm_len, base_radius=base_radius
    )




def read_lagmul_spherical_wrist_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SPHERICAL_WRIST_JOINT/id`` or ``/LAGMUL/SPHERICAL_WRIST_JOINT/id`` (M290): 3-DOF robotic intersecting-axes spherical wrist kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SPHERICAL_WRIST_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    roll_limit, pitch_limit, yaw_limit = 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SPHERICAL_WRIST_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SPHERICAL_WRIST_JOINT_2")
            roll_limit = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            pitch_limit = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            yaw_limit = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            roll_limit = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            pitch_limit = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            yaw_limit = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0

    from ...model.entities import LagmulSphericalWristJoint
    model.lagmul_spherical_wrist_joints[block.user_id] = LagmulSphericalWristJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        roll_limit=roll_limit, pitch_limit=pitch_limit, yaw_limit=yaw_limit
    )




def read_lagmul_lead_screw_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LEAD_SCREW_JOINT/id`` or ``/LAGMUL/LEAD_SCREW_JOINT/id`` (M291): Helical lead screw and ball screw coupled linear-rotational kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LEAD_SCREW_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    pitch_lead, thread_angle, helix_efficiency = 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("LEAD_SCREW_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("LEAD_SCREW_JOINT_2")
            pitch_lead = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            thread_angle = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            helix_efficiency = _fval(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            pitch_lead = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            thread_angle = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            helix_efficiency = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 1.0

    from ...model.entities import LagmulLeadScrewJoint
    model.lagmul_lead_screw_joints[block.user_id] = LagmulLeadScrewJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        pitch_lead=pitch_lead, thread_angle=thread_angle, helix_efficiency=helix_efficiency
    )




def read_lagmul_hoeken_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HOEKEN_LINKAGE_JOINT/id`` or ``/LAGMUL/HOEKEN_LINKAGE_JOINT/id`` (M292): Hoecken 4-bar straight-line approximate planar kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HOEKEN_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    crank_len, rocker_len, coupler_len = 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HOEKEN_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HOEKEN_LINKAGE_JOINT_2")
            crank_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            rocker_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            coupler_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            crank_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            rocker_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            coupler_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0

    from ...model.entities import LagmulHoekenLinkageJoint
    model.lagmul_hoeken_linkage_joints[block.user_id] = LagmulHoekenLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        crank_len=crank_len, rocker_len=rocker_len, coupler_len=coupler_len
    )




def read_lagmul_chebyshev_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHEBYSHEV_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEBYSHEV_LINKAGE_JOINT/id`` (M293): Chebyshev 4-bar straight-line crossing linkage planar kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CHEBYSHEV_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    base_len, crank_len, coupler_len = 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CHEBYSHEV_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CHEBYSHEV_LINKAGE_JOINT_2")
            base_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            crank_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            coupler_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            base_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            crank_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            coupler_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0

    from ...model.entities import LagmulChebyshevLinkageJoint
    model.lagmul_chebyshev_linkage_joints[block.user_id] = LagmulChebyshevLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        base_len=base_len, crank_len=crank_len, coupler_len=coupler_len
    )




def read_lagmul_roberts_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ROBERTS_LINKAGE_JOINT/id`` or ``/LAGMUL/ROBERTS_LINKAGE_JOINT/id`` (M294): Roberts 4-bar straight-line symmetrical linkage planar kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ROBERTS_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    base_len, arm_len, coupler_height = 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ROBERTS_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("ROBERTS_LINKAGE_JOINT_2")
            base_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            arm_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            coupler_height = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            base_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            arm_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            coupler_height = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0

    from ...model.entities import LagmulRobertsLinkageJoint
    model.lagmul_roberts_linkage_joints[block.user_id] = LagmulRobertsLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        base_len=base_len, arm_len=arm_len, coupler_height=coupler_height
    )




def read_lagmul_evans_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EVANS_LINKAGE_JOINT/id`` or ``/LAGMUL/EVANS_LINKAGE_JOINT/id`` (M295): Evans (Grasshopper) 4-bar approximate straight-line linkage planar kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/EVANS_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    ground_len, crank_len, arm_len = 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("EVANS_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("EVANS_LINKAGE_JOINT_2")
            ground_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            crank_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            arm_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            ground_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            crank_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            arm_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0

    from ...model.entities import LagmulEvansLinkageJoint
    model.lagmul_evans_linkage_joints[block.user_id] = LagmulEvansLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        ground_len=ground_len, crank_len=crank_len, arm_len=arm_len
    )




def read_lagmul_watt_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WATT_LINKAGE_JOINT/id`` or ``/LAGMUL/WATT_LINKAGE_JOINT/id`` (M296): Watt 4-bar approximate straight-line linkage planar kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WATT_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    ground_len, link1_len, link2_len, coupler_len = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WATT_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WATT_LINKAGE_JOINT_2")
            ground_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link1_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            link2_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            coupler_len = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            ground_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link1_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            link2_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            coupler_len = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWattLinkageJoint
    model.lagmul_watt_linkage_joints[block.user_id] = LagmulWattLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        ground_len=ground_len, link1_len=link1_len, link2_len=link2_len, coupler_len=coupler_len
    )




def read_lagmul_hart_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HART_LINKAGE_JOINT/id`` or ``/LAGMUL/HART_LINKAGE_JOINT/id`` (M297): Hart 5-bar / 6-bar exact straight-line linkage planar kinematic inversor mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HART_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    base_len, short_link_len, long_link_len, coupler_ratio = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HART_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HART_LINKAGE_JOINT_2")
            base_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            short_link_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            long_link_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            coupler_ratio = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 0 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            base_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            short_link_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            long_link_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            coupler_ratio = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHartLinkageJoint
    model.lagmul_hart_linkage_joints[block.user_id] = LagmulHartLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        base_len=base_len, short_link_len=short_link_len, long_link_len=long_link_len, coupler_ratio=coupler_ratio
    )




def read_lagmul_peaucellier_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PEAUCELLIER_LINKAGE_JOINT/id`` or ``/LAGMUL/PEAUCELLIER_LINKAGE_JOINT/id`` (M298): Peaucellier–Lipkin 8-bar exact straight-line linkage planar kinematic inversor mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PEAUCELLIER_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    base_len, rhombus_len, long_link_len, inversor_k = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PEAUCELLIER_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PEAUCELLIER_LINKAGE_JOINT_2")
            base_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            rhombus_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            long_link_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            inversor_k = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            base_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            rhombus_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            long_link_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            inversor_k = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPeaucellierLinkageJoint
    model.lagmul_peaucellier_linkage_joints[block.user_id] = LagmulPeaucellierLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        base_len=base_len, rhombus_len=rhombus_len, long_link_len=long_link_len, inversor_k=inversor_k
    )




def read_lagmul_sarrus_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SARRUS_LINKAGE_JOINT/id`` or ``/LAGMUL/SARRUS_LINKAGE_JOINT/id`` (M299): Sarrus 6-bar spatial exact straight-line kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SARRUS_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    plate_len1, plate_len2, hinge_angle, guide_travel = 0.0, 0.0, 90.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SARRUS_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SARRUS_LINKAGE_JOINT_2")
            plate_len1 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            plate_len2 = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            hinge_angle = _fval(f2[2], 90.0) if len(f2) > 2 and f2[2].strip() else 90.0
            guide_travel = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            plate_len1 = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            plate_len2 = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            hinge_angle = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 90.0
            guide_travel = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSarrusLinkageJoint
    model.lagmul_sarrus_linkage_joints[block.user_id] = LagmulSarrusLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        plate_len1=plate_len1, plate_len2=plate_len2, hinge_angle=hinge_angle, guide_travel=guide_travel
    )




def read_lagmul_klann_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KLANN_LINKAGE_JOINT/id`` or ``/LAGMUL/KLANN_LINKAGE_JOINT/id`` (M300): Klann 6-bar mechanical walking linkage planar kinematic leg mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KLANN_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    crank_len, rocker_len, leg_upper_len, leg_lower_len = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KLANN_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KLANN_LINKAGE_JOINT_2")
            crank_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            rocker_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            leg_upper_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            leg_lower_len = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            crank_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            rocker_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            leg_upper_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            leg_lower_len = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKlannLinkageJoint
    model.lagmul_klann_linkage_joints[block.user_id] = LagmulKlannLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        crank_len=crank_len, rocker_len=rocker_len, leg_upper_len=leg_upper_len, leg_lower_len=leg_lower_len
    )




def read_lagmul_jansen_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/JANSEN_LINKAGE_JOINT/id`` or ``/LAGMUL/JANSEN_LINKAGE_JOINT/id`` (M301): Jansen 8-bar / 11-bar kinematic walking linkage (Theo Jansen Strandbeest leg mechanism) joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/JANSEN_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    crank_len, base_horizontal, base_vertical, leg_ratio = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("JANSEN_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("JANSEN_LINKAGE_JOINT_2")
            crank_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            base_horizontal = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            base_vertical = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            leg_ratio = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            crank_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            base_horizontal = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            base_vertical = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            leg_ratio = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulJansenLinkageJoint
    model.lagmul_jansen_linkage_joints[block.user_id] = LagmulJansenLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        crank_len=crank_len, base_horizontal=base_horizontal,
        base_vertical=base_vertical, leg_ratio=leg_ratio
    )




def read_lagmul_kempe_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KEMPE_LINKAGE_JOINT/id`` or ``/LAGMUL/KEMPE_LINKAGE_JOINT/id`` (M302): Kempe multi-bar kinematic linkage joint constraint (Kempe's exact straight-line and angle-multiplier mechanism)."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KEMPE_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    arm_len1, arm_len2, cross_len, travel_span = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KEMPE_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KEMPE_LINKAGE_JOINT_2")
            arm_len1 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            arm_len2 = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            cross_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            travel_span = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            arm_len1 = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            arm_len2 = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            cross_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            travel_span = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKempeLinkageJoint
    model.lagmul_kempe_linkage_joints[block.user_id] = LagmulKempeLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        arm_len1=arm_len1, arm_len2=arm_len2,
        cross_len=cross_len, travel_span=travel_span
    )




def read_lagmul_sylvester_kempe_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SYLVESTER_KEMPE_LINKAGE_JOINT/id`` or ``/LAGMUL/SYLVESTER_KEMPE_LINKAGE_JOINT/id`` (M303): Sylvester-Kempe 8-bar quad-inversor exact straight-line and circular-arc kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SYLVESTER_KEMPE_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    arm_len_a, arm_len_b, base_dist, angular_multiplier = 0.0, 0.0, 0.0, 1.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SYLVESTER_KEMPE_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SYLVESTER_KEMPE_LINKAGE_JOINT_2")
            arm_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            arm_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            base_dist = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            angular_multiplier = _fval(f2[3], 1.0) if len(f2) > 3 and f2[3].strip() else 1.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            arm_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            arm_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            base_dist = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            angular_multiplier = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 1.0

    from ...model.entities import LagmulSylvesterKempeLinkageJoint
    model.lagmul_sylvester_kempe_linkage_joints[block.user_id] = LagmulSylvesterKempeLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        arm_len_a=arm_len_a, arm_len_b=arm_len_b,
        base_dist=base_dist, angular_multiplier=angular_multiplier
    )




def read_lagmul_wobble_yoke_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WOBBLE_YOKE_JOINT/id`` or ``/LAGMUL/WOBBLE_YOKE_JOINT/id`` (M304): Wobble yoke / nutating spatial kinematic joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WOBBLE_YOKE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    nutation_angle, yoke_radius, stroke_travel, phase_offset = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WOBBLE_YOKE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WOBBLE_YOKE_JOINT_2")
            nutation_angle = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            yoke_radius = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            stroke_travel = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            phase_offset = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            nutation_angle = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            yoke_radius = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            stroke_travel = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            phase_offset = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWobbleYokeJoint
    model.lagmul_wobble_yoke_joints[block.user_id] = LagmulWobbleYokeJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        nutation_angle=nutation_angle, yoke_radius=yoke_radius,
        stroke_travel=stroke_travel, phase_offset=phase_offset
    )




def read_lagmul_hypocyclic_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HYPOCYCLIC_LINKAGE_JOINT/id`` or ``/LAGMUL/HYPOCYCLIC_LINKAGE_JOINT/id`` (M305): Hypocyclic / Tusi couple 2-gear straight-line kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HYPOCYCLIC_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    radius_outer, radius_inner, stroke_travel, phase_angle = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HYPOCYCLIC_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HYPOCYCLIC_LINKAGE_JOINT_2")
            radius_outer = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            radius_inner = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            stroke_travel = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            phase_angle = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            radius_outer = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            radius_inner = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            stroke_travel = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            phase_angle = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHypocyclicLinkageJoint
    model.lagmul_hypocyclic_linkage_joints[block.user_id] = LagmulHypocyclicLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        radius_outer=radius_outer, radius_inner=radius_inner,
        stroke_travel=stroke_travel, phase_angle=phase_angle
    )




def read_lagmul_pantograph_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PANTOGRAPH_LINKAGE_JOINT/id`` or ``/LAGMUL/PANTOGRAPH_LINKAGE_JOINT/id`` (M306): Pantograph parallelogram 4-bar / 5-bar kinematic motion scaling and copy joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PANTOGRAPH_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    scale_factor, arm_length_a, arm_length_b, cross_angle_0 = 2.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PANTOGRAPH_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PANTOGRAPH_LINKAGE_JOINT_2")
            scale_factor = _fval(f2[0], 2.0) if len(f2) > 0 and f2[0].strip() else 2.0
            arm_length_a = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            arm_length_b = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            cross_angle_0 = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            scale_factor = float(toks2[0].rstrip(',')) if len(toks2) > 0 and toks2[0].rstrip(',') else 2.0
            arm_length_a = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            arm_length_b = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            cross_angle_0 = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPantographLinkageJoint
    model.lagmul_pantograph_linkage_joints[block.user_id] = LagmulPantographLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        scale_factor=scale_factor, arm_length_a=arm_length_a,
        arm_length_b=arm_length_b, cross_angle_0=cross_angle_0
    )




def read_lagmul_watt_parallel_motion_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WATT_PARALLEL_MOTION_JOINT/id`` or ``/LAGMUL/WATT_PARALLEL_MOTION_JOINT/id`` (M307): Watt double-parallelogram parallel motion kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WATT_PARALLEL_MOTION_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    arm_length_main, arm_length_sub, stroke_travel, offset_dist = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WATT_PARALLEL_MOTION_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WATT_PARALLEL_MOTION_JOINT_2")
            arm_length_main = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            arm_length_sub = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            stroke_travel = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_dist = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            arm_length_main = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            arm_length_sub = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            stroke_travel = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_dist = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWattParallelMotionJoint
    model.lagmul_watt_parallel_motion_joints[block.user_id] = LagmulWattParallelMotionJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        arm_length_main=arm_length_main, arm_length_sub=arm_length_sub,
        stroke_travel=stroke_travel, offset_dist=offset_dist
    )




def read_lagmul_scott_russell_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCOTT_RUSSELL_LINKAGE_JOINT/id`` or ``/LAGMUL/SCOTT_RUSSELL_LINKAGE_JOINT/id`` (M308): Scott Russell exact straight-line rolling circle kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCOTT_RUSSELL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    crank_len, coupler_len, stroke_travel, slider_friction = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SCOTT_RUSSELL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SCOTT_RUSSELL_LINKAGE_JOINT_2")
            crank_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            coupler_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            stroke_travel = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            slider_friction = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            crank_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            coupler_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            stroke_travel = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            slider_friction = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulScottRussellLinkageJoint
    model.lagmul_scott_russell_linkage_joints[block.user_id] = LagmulScottRussellLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        crank_len=crank_len, coupler_len=coupler_len,
        stroke_travel=stroke_travel, slider_friction=slider_friction
    )




def read_lagmul_watt_beam_engine_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WATT_BEAM_ENGINE_JOINT/id`` or ``/LAGMUL/WATT_BEAM_ENGINE_JOINT/id`` (M309): Watt rocking walking-beam engine kinematic linkage joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WATT_BEAM_ENGINE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    beam_half_length_a, beam_half_length_b, stroke_travel, beam_tilt_max = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WATT_BEAM_ENGINE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WATT_BEAM_ENGINE_JOINT_2")
            beam_half_length_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            beam_half_length_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            stroke_travel = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            beam_tilt_max = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            beam_half_length_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            beam_half_length_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            stroke_travel = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            beam_tilt_max = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWattBeamEngineJoint
    model.lagmul_watt_beam_engine_joints[block.user_id] = LagmulWattBeamEngineJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        beam_half_length_a=beam_half_length_a, beam_half_length_b=beam_half_length_b,
        stroke_travel=stroke_travel, beam_tilt_max=beam_tilt_max
    )




def read_lagmul_stephenson_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STEPHENSON_LINKAGE_JOINT/id`` or ``/LAGMUL/STEPHENSON_LINKAGE_JOINT/id`` (M310): Stephenson 6-bar kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STEPHENSON_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, link_len_c, link_len_d = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("STEPHENSON_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("STEPHENSON_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            link_len_c = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            link_len_d = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            link_len_c = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            link_len_d = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulStephensonLinkageJoint
    model.lagmul_stephenson_linkage_joints[block.user_id] = LagmulStephensonLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        link_len_c=link_len_c, link_len_d=link_len_d
    )




def read_lagmul_wobble_plate_mechanism_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WOBBLE_PLATE_MECHANISM_JOINT/id`` or ``/LAGMUL/WOBBLE_PLATE_MECHANISM_JOINT/id`` (M311): Wobble plate / nutating axial kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WOBBLE_PLATE_MECHANISM_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    plate_radius, nutation_angle, stroke_travel, piston_count = 0.0, 0.0, 0.0, 1
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WOBBLE_PLATE_MECHANISM_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WOBBLE_PLATE_MECHANISM_JOINT_2")
            plate_radius = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nutation_angle = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            stroke_travel = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            piston_count = _ival(f2[3], 1) if len(f2) > 3 and f2[3].strip() else 1
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            plate_radius = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            nutation_angle = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            stroke_travel = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            piston_count = int(float(toks2[3].rstrip(','))) if len(toks2) > 3 else 1

    from ...model.entities import LagmulWobblePlateMechanismJoint
    model.lagmul_wobble_plate_mechanism_joints[block.user_id] = LagmulWobblePlateMechanismJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        plate_radius=plate_radius, nutation_angle=nutation_angle,
        stroke_travel=stroke_travel, piston_count=piston_count
    )




def read_lagmul_whitworth_quick_return_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WHITWORTH_QUICK_RETURN_JOINT/id`` or ``/LAGMUL/WHITWORTH_QUICK_RETURN_JOINT/id`` (M312): Whitworth quick-return slotted crank kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WHITWORTH_QUICK_RETURN_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    driving_crank_len, pivot_offset, slotted_arm_len, connecting_rod_len = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WHITWORTH_QUICK_RETURN_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WHITWORTH_QUICK_RETURN_JOINT_2")
            driving_crank_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            pivot_offset = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            slotted_arm_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            connecting_rod_len = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            driving_crank_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            pivot_offset = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            slotted_arm_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            connecting_rod_len = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWhitworthQuickReturnJoint
    model.lagmul_whitworth_quick_return_joints[block.user_id] = LagmulWhitworthQuickReturnJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        driving_crank_len=driving_crank_len, pivot_offset=pivot_offset,
        slotted_arm_len=slotted_arm_len, connecting_rod_len=connecting_rod_len
    )




def read_lagmul_chebyshev_lambda_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHEBYSHEV_LAMBDA_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEBYSHEV_LAMBDA_LINKAGE_JOINT/id`` (M313): Chebyshev lambda / walking mechanism 4-bar straight-line kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CHEBYSHEV_LAMBDA_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    ground_dist, crank_len, coupler_len, rocker_len = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CHEBYSHEV_LAMBDA_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CHEBYSHEV_LAMBDA_LINKAGE_JOINT_2")
            ground_dist = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            crank_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            coupler_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            rocker_len = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            ground_dist = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            crank_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            coupler_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            rocker_len = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulChebyshevLambdaLinkageJoint
    model.lagmul_chebyshev_lambda_linkage_joints[block.user_id] = LagmulChebyshevLambdaLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        ground_dist=ground_dist, crank_len=crank_len,
        coupler_len=coupler_len, rocker_len=rocker_len
    )




def read_lagmul_four_bar_crank_rocker_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FOUR_BAR_CRANK_ROCKER_JOINT/id`` or ``/LAGMUL/FOUR_BAR_CRANK_ROCKER_JOINT/id`` (M314): Grashof 4-bar crank-rocker kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FOUR_BAR_CRANK_ROCKER_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    crank_len, coupler_len, rocker_len, ground_len = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FOUR_BAR_CRANK_ROCKER_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("FOUR_BAR_CRANK_ROCKER_JOINT_2")
            crank_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            coupler_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            rocker_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            ground_len = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            crank_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            coupler_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            rocker_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            ground_len = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFourBarCrankRockerJoint
    model.lagmul_four_bar_crank_rocker_joints[block.user_id] = LagmulFourBarCrankRockerJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        crank_len=crank_len, coupler_len=coupler_len,
        rocker_len=rocker_len, ground_len=ground_len
    )




def read_lagmul_four_bar_double_crank_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FOUR_BAR_DOUBLE_CRANK_JOINT/id`` or ``/LAGMUL/FOUR_BAR_DOUBLE_CRANK_JOINT/id`` (M315): Grashof 4-bar double-crank (drag-link) kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FOUR_BAR_DOUBLE_CRANK_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    driving_crank_len, coupler_len, driven_crank_len, ground_len = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FOUR_BAR_DOUBLE_CRANK_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("FOUR_BAR_DOUBLE_CRANK_JOINT_2")
            driving_crank_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            coupler_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            driven_crank_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            ground_len = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            driving_crank_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            coupler_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            driven_crank_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            ground_len = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFourBarDoubleCrankJoint
    model.lagmul_four_bar_double_crank_joints[block.user_id] = LagmulFourBarDoubleCrankJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        driving_crank_len=driving_crank_len, coupler_len=coupler_len,
        driven_crank_len=driven_crank_len, ground_len=ground_len
    )




def read_lagmul_cohomological_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` (M435): Cohomological spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCohomologicalSpatialLinkageJoint
    model.lagmul_cohomological_spatial_linkage_joints[block.user_id] = LagmulCohomologicalSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_serre_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M476): Serre spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SERRE_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("SERRE_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSerreSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_serre_spinor_spatial_linkage_joints) + 1)
    model.lagmul_serre_spinor_spatial_linkage_joints[j_id] = LagmulSerreSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_spinorial_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SPINORIAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SPINORIAL_SPATIAL_LINKAGE_JOINT/id`` (M436): Spinorial spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SPINORIAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SPINORIAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SPINORIAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSpinorialSpatialLinkageJoint
    model.lagmul_spinorial_spatial_linkage_joints[block.user_id] = LagmulSpinorialSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_symplectic_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M438): Symplectic spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSymplecticSpinorSpatialLinkageJoint
    model.lagmul_symplectic_spinor_spatial_linkage_joints[block.user_id] = LagmulSymplecticSpinorSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_symplectic_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M438): Symplectic spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSymplecticSpinorSpatialLinkageJoint
    model.lagmul_symplectic_spinor_spatial_linkage_joints[block.user_id] = LagmulSymplecticSpinorSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_contact_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M439): Contact spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulContactSpinorSpatialLinkageJoint
    model.lagmul_contact_spinor_spatial_linkage_joints[block.user_id] = LagmulContactSpinorSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_conformal_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M440): Conformal spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulConformalSpinorSpatialLinkageJoint
    model.lagmul_conformal_spinor_spatial_linkage_joints[block.user_id] = LagmulConformalSpinorSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_projective_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M441): Projective spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulProjectiveSpinorSpatialLinkageJoint
    model.lagmul_projective_spinor_spatial_linkage_joints[block.user_id] = LagmulProjectiveSpinorSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_algebraic_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M442): Algebraic spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAlgebraicSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_algebraic_spinor_spatial_linkage_joints) + 1)
    model.lagmul_algebraic_spinor_spatial_linkage_joints[j_id] = LagmulAlgebraicSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_topological_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M443): Topological spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulTopologicalSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_topological_spinor_spatial_linkage_joints) + 1)
    model.lagmul_topological_spinor_spatial_linkage_joints[j_id] = LagmulTopologicalSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_homological_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M444): Homological spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHomologicalSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_homological_spinor_spatial_linkage_joints) + 1)
    model.lagmul_homological_spinor_spatial_linkage_joints[j_id] = LagmulHomologicalSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_cohomological_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M445): Cohomological spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCohomologicalSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_cohomological_spinor_spatial_linkage_joints) + 1)
    model.lagmul_cohomological_spinor_spatial_linkage_joints[j_id] = LagmulCohomologicalSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_sheaf_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M446): Sheaf spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSheafSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_sheaf_spinor_spatial_linkage_joints) + 1)
    model.lagmul_sheaf_spinor_spatial_linkage_joints[j_id] = LagmulSheafSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_stack_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STACK_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STACK_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M447): Stack spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STACK_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("STACK_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("STACK_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulStackSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_stack_spinor_spatial_linkage_joints) + 1)
    model.lagmul_stack_spinor_spatial_linkage_joints[j_id] = LagmulStackSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_topos_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M448): Topos spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulToposSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_topos_spinor_spatial_linkage_joints) + 1)
    model.lagmul_topos_spinor_spatial_linkage_joints[j_id] = LagmulToposSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_scheme_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M449): Scheme spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSchemeSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_scheme_spinor_spatial_linkage_joints) + 1)
    model.lagmul_scheme_spinor_spatial_linkage_joints[j_id] = LagmulSchemeSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_orbifold_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M450): Orbifold spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulOrbifoldSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_orbifold_spinor_spatial_linkage_joints) + 1)
    model.lagmul_orbifold_spinor_spatial_linkage_joints[j_id] = LagmulOrbifoldSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_foliation_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M451): Foliation spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFoliationSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_foliation_spinor_spatial_linkage_joints) + 1)
    model.lagmul_foliation_spinor_spatial_linkage_joints[j_id] = LagmulFoliationSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_stratification_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M452): Stratification spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulStratificationSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_stratification_spinor_spatial_linkage_joints) + 1)
    model.lagmul_stratification_spinor_spatial_linkage_joints[j_id] = LagmulStratificationSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_fibration_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M453): Fibration spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFibrationSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_fibration_spinor_spatial_linkage_joints) + 1)
    model.lagmul_fibration_spinor_spatial_linkage_joints[j_id] = LagmulFibrationSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_bundle_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M454): Bundle spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBundleSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_bundle_spinor_spatial_linkage_joints) + 1)
    model.lagmul_bundle_spinor_spatial_linkage_joints[j_id] = LagmulBundleSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_connection_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M455): Connection spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulConnectionSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_connection_spinor_spatial_linkage_joints) + 1)
    model.lagmul_connection_spinor_spatial_linkage_joints[j_id] = LagmulConnectionSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_curvature_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M456): Curvature spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCurvatureSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_curvature_spinor_spatial_linkage_joints) + 1)
    model.lagmul_curvature_spinor_spatial_linkage_joints[j_id] = LagmulCurvatureSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_torsion_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TORSION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TORSION_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M457): Torsion spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TORSION_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("TORSION_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("TORSION_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulTorsionSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_torsion_spinor_spatial_linkage_joints) + 1)
    model.lagmul_torsion_spinor_spatial_linkage_joints[j_id] = LagmulTorsionSpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_kostant_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M467): Kostant spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKostantSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_kostant_spinor_spatial_linkage_joints) + 1)
    model.lagmul_kostant_spinor_spatial_linkage_joints[j_id] = LagmulKostantSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_souriau_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M468): Souriau spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSouriauSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_souriau_spinor_spatial_linkage_joints) + 1)
    model.lagmul_souriau_spinor_spatial_linkage_joints[j_id] = LagmulSouriauSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_berezin_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M469): Berezin spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBerezinSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_berezin_spinor_spatial_linkage_joints) + 1)
    model.lagmul_berezin_spinor_spatial_linkage_joints[j_id] = LagmulBerezinSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_borel_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M470): Borel spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BOREL_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("BOREL_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBorelSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_borel_spinor_spatial_linkage_joints) + 1)
    model.lagmul_borel_spinor_spatial_linkage_joints[j_id] = LagmulBorelSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_bott_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BOTT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BOTT_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M471): Bott spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BOTT_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BOTT_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("BOTT_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBottSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_bott_spinor_spatial_linkage_joints) + 1)
    model.lagmul_bott_spinor_spatial_linkage_joints[j_id] = LagmulBottSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_chern_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M472): Chern spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CHERN_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("CHERN_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulChernSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_chern_spinor_spatial_linkage_joints) + 1)
    model.lagmul_chern_spinor_spatial_linkage_joints[j_id] = LagmulChernSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_atiyah_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M473): Atiyah spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAtiyahSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_atiyah_spinor_spatial_linkage_joints) + 1)
    model.lagmul_atiyah_spinor_spatial_linkage_joints[j_id] = LagmulAtiyahSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_hirzebruch_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M474): Hirzebruch spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHirzebruchSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_hirzebruch_spinor_spatial_linkage_joints) + 1)
    model.lagmul_hirzebruch_spinor_spatial_linkage_joints[j_id] = LagmulHirzebruchSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_grothendieck_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M475): Grothendieck spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulGrothendieckSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_grothendieck_spinor_spatial_linkage_joints) + 1)
    model.lagmul_grothendieck_spinor_spatial_linkage_joints[j_id] = LagmulGrothendieckSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_kahler_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M466): Kähler spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKahlerSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_kahler_spinor_spatial_linkage_joints) + 1)
    model.lagmul_kahler_spinor_spatial_linkage_joints[j_id] = LagmulKahlerSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_majorana_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M465): Majorana spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulMajoranaSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_majorana_spinor_spatial_linkage_joints) + 1)
    model.lagmul_majorana_spinor_spatial_linkage_joints[j_id] = LagmulMajoranaSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_cartan_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M464): Cartan spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCartanSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_cartan_spinor_spatial_linkage_joints) + 1)
    model.lagmul_cartan_spinor_spatial_linkage_joints[j_id] = LagmulCartanSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_jacobi_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M463): Jacobi spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulJacobiSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_jacobi_spinor_spatial_linkage_joints) + 1)
    model.lagmul_jacobi_spinor_spatial_linkage_joints[j_id] = LagmulJacobiSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_dirac_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M462): Dirac spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDiracSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_dirac_spinor_spatial_linkage_joints) + 1)
    model.lagmul_dirac_spinor_spatial_linkage_joints[j_id] = LagmulDiracSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_poisson_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M461): Poisson spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("POISSON_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("POISSON_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPoissonSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_poisson_spinor_spatial_linkage_joints) + 1)
    model.lagmul_poisson_spinor_spatial_linkage_joints[j_id] = LagmulPoissonSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_symplectic_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M460): Symplectic spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSymplecticSpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_symplectic_spinor_spatial_linkage_joints) + 1)
    model.lagmul_symplectic_spinor_spatial_linkage_joints[j_id] = LagmulSymplecticSpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_monodromy_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M459): Monodromy spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulMonodromySpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_monodromy_spinor_spatial_linkage_joints) + 1)
    model.lagmul_monodromy_spinor_spatial_linkage_joints[j_id] = LagmulMonodromySpinorSpatialLinkageJoint(
        id=j_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_holonomy_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M458): Holonomy spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card 1", block.source)
        return

    node1, node2, node3 = 0, 0, 0
    stiff, skew_id, tol = 1e6, 0, 1e-6
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0], 0) if len(f1) > 0 else 0
        node2 = _ival(f1[1], 0) if len(f1) > 1 else 0
        node3 = _ival(f1[2], 0) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0].rstrip(','))) if len(toks) > 0 else 0
        node2 = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0
        node3 = int(float(toks[2].rstrip(','))) if len(toks) > 2 else 0
        stiff = float(toks[3].rstrip(',')) if len(toks) > 3 else 1e6
        skew_id = int(float(toks[4].rstrip(','))) if len(toks) > 4 else 0
        tol = float(toks[5].rstrip(',')) if len(toks) > 5 else 1e-6

    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed and "," not in cards[1].raw:
            f2 = cards[1].cut("HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
        else:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHolonomySpinorSpatialLinkageJoint
    j_id = block.user_id or (len(model.lagmul_holonomy_spinor_spatial_linkage_joints) + 1)
    model.lagmul_holonomy_spinor_spatial_linkage_joints[j_id] = LagmulHolonomySpinorSpatialLinkageJoint(
        id=j_id,
        title=title,
        node1=node1,
        node2=node2,
        node3=node3,
        stiff=stiff,
        skew_id=skew_id,
        tol=tol,
        link_len_a=link_len_a,
        link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha,
        offset_distance_s=offset_distance_s
    )




def read_lagmul_projective_spinor_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/id`` (M441): Projective spinor spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulProjectiveSpinorSpatialLinkageJoint
    model.lagmul_projective_spinor_spatial_linkage_joints[block.user_id] = LagmulProjectiveSpinorSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_four_bar_double_rocker_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FOUR_BAR_DOUBLE_ROCKER_JOINT/id`` or ``/LAGMUL/FOUR_BAR_DOUBLE_ROCKER_JOINT/id`` (M316): Grashof / Non-Grashof 4-bar double-rocker (dual oscillating arm) kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FOUR_BAR_DOUBLE_ROCKER_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    input_rocker_len, coupler_len, output_rocker_len, ground_len = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FOUR_BAR_DOUBLE_ROCKER_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("FOUR_BAR_DOUBLE_ROCKER_JOINT_2")
            input_rocker_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            coupler_len = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            output_rocker_len = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            ground_len = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            input_rocker_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            coupler_len = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            output_rocker_len = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            ground_len = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFourBarDoubleRockerJoint
    model.lagmul_four_bar_double_rocker_joints[block.user_id] = LagmulFourBarDoubleRockerJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        input_rocker_len=input_rocker_len, coupler_len=coupler_len,
        output_rocker_len=output_rocker_len, ground_len=ground_len
    )




def read_lagmul_slider_rocker_inversion_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SLIDER_ROCKER_INVERSION_JOINT/id`` or ``/LAGMUL/SLIDER_ROCKER_INVERSION_JOINT/id`` (M317): Slider-rocker inverted kinematic mechanism (oscillating cylinder / swing engine) joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SLIDER_ROCKER_INVERSION_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    crank_len, frame_dist, piston_offset, stroke_limit = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SLIDER_ROCKER_INVERSION_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SLIDER_ROCKER_INVERSION_JOINT_2")
            crank_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            frame_dist = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            piston_offset = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            stroke_limit = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            crank_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            frame_dist = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            piston_offset = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            stroke_limit = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSliderRockerInversionJoint
    model.lagmul_slider_rocker_inversion_joints[block.user_id] = LagmulSliderRockerInversionJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        crank_len=crank_len, frame_dist=frame_dist,
        piston_offset=piston_offset, stroke_limit=stroke_limit
    )




def read_lagmul_scotch_yoke_mechanism_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCOTCH_YOKE_MECHANISM_JOINT/id`` or ``/LAGMUL/SCOTCH_YOKE_MECHANISM_JOINT/id`` (M318): Scotch yoke / slotted link reciprocating-to-rotary planar kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCOTCH_YOKE_MECHANISM_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    crank_radius, slot_width, stroke_limit, yoke_angle = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SCOTCH_YOKE_MECHANISM_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SCOTCH_YOKE_MECHANISM_JOINT_2")
            crank_radius = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            slot_width = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            stroke_limit = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            yoke_angle = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            crank_radius = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            slot_width = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            stroke_limit = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            yoke_angle = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulScotchYokeMechanismJoint
    model.lagmul_scotch_yoke_mechanism_joints[block.user_id] = LagmulScotchYokeMechanismJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        crank_radius=crank_radius, slot_width=slot_width,
        stroke_limit=stroke_limit, yoke_angle=yoke_angle
    )




def read_lagmul_differential_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/id`` (M433): Differential spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DIFFERENTIAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DIFFERENTIAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDifferentialSpatialLinkageJoint
    model.lagmul_differential_spatial_linkage_joints[block.user_id] = LagmulDifferentialSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_geneva_drive_mechanism_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GENEVA_DRIVE_MECHANISM_JOINT/id`` or ``/LAGMUL/GENEVA_DRIVE_MECHANISM_JOINT/id`` (M319): Geneva drive / Maltese cross intermittent rotary indexing kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GENEVA_DRIVE_MECHANISM_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    drive_radius, wheel_radius, num_slots, center_dist = 0.0, 0.0, 4, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("GENEVA_DRIVE_MECHANISM_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("GENEVA_DRIVE_MECHANISM_JOINT_2")
            drive_radius = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            wheel_radius = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            num_slots = _ival(f2[2], 4) if len(f2) > 2 and f2[2].strip() else 4
            center_dist = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            drive_radius = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            wheel_radius = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            num_slots = int(float(toks2[2].rstrip(','))) if len(toks2) > 2 else 4
            center_dist = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulGenevaDriveMechanismJoint
    model.lagmul_geneva_drive_mechanism_joints[block.user_id] = LagmulGenevaDriveMechanismJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        drive_radius=drive_radius, wheel_radius=wheel_radius,
        num_slots=num_slots, center_dist=center_dist
    )




def read_lagmul_double_cardan_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DOUBLE_CARDAN_JOINT/id`` or ``/LAGMUL/DOUBLE_CARDAN_JOINT/id`` (M320): Double Cardan / constant-velocity dual-universal kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DOUBLE_CARDAN_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    center_yoke_len, max_bend_angle, phase_offset, friction_coeff = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DOUBLE_CARDAN_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DOUBLE_CARDAN_JOINT_2")
            center_yoke_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            max_bend_angle = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            phase_offset = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            friction_coeff = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            center_yoke_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            max_bend_angle = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            phase_offset = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            friction_coeff = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDoubleCardanJoint
    model.lagmul_double_cardan_joints[block.user_id] = LagmulDoubleCardanJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        center_yoke_len=center_yoke_len, max_bend_angle=max_bend_angle,
        phase_offset=phase_offset, friction_coeff=friction_coeff
    )




def read_lagmul_bennett_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BENNETT_LINKAGE_JOINT/id`` or ``/LAGMUL/BENNETT_LINKAGE_JOINT/id`` (M321): Bennett's 4R spatial skewed-axis overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BENNETT_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, twist_angle_beta = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BENNETT_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BENNETT_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            twist_angle_beta = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            twist_angle_beta = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBennettLinkageJoint
    model.lagmul_bennett_linkage_joints[block.user_id] = LagmulBennettLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, twist_angle_beta=twist_angle_beta
    )




def read_lagmul_bricard_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BRICARD_LINKAGE_JOINT/id`` or ``/LAGMUL/BRICARD_LINKAGE_JOINT/id`` (M322): Bricard's 6R spatial line-symmetric/plane-symmetric overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BRICARD_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len, twist_angle, offset_dist, sym_angle = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BRICARD_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BRICARD_LINKAGE_JOINT_2")
            link_len = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            twist_angle = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            offset_dist = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            sym_angle = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            twist_angle = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            offset_dist = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            sym_angle = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBricardLinkageJoint
    model.lagmul_bricard_linkage_joints[block.user_id] = LagmulBricardLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len=link_len, twist_angle=twist_angle,
        offset_dist=offset_dist, sym_angle=sym_angle
    )




def read_lagmul_myard_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MYARD_LINKAGE_JOINT/id`` or ``/LAGMUL/MYARD_LINKAGE_JOINT/id`` (M323): Myard's 5R spatial plane-symmetric overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MYARD_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle, fold_angle = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("MYARD_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MYARD_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            fold_angle = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            fold_angle = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulMyardLinkageJoint
    model.lagmul_myard_linkage_joints[block.user_id] = LagmulMyardLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle=twist_angle, fold_angle=fold_angle
    )




def read_lagmul_goldberg_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GOLDBERG_LINKAGE_JOINT/id`` or ``/LAGMUL/GOLDBERG_LINKAGE_JOINT/id`` (M324): Goldberg 5R / 6R spatial overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GOLDBERG_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, skew_angle_alpha, offset_angle_beta = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("GOLDBERG_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("GOLDBERG_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            skew_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_angle_beta = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            skew_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_angle_beta = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulGoldbergLinkageJoint
    model.lagmul_goldberg_linkage_joints[block.user_id] = LagmulGoldbergLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        skew_angle_alpha=skew_angle_alpha, offset_angle_beta=offset_angle_beta
    )




def read_lagmul_waldron_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WALDRON_LINKAGE_JOINT/id`` or ``/LAGMUL/WALDRON_LINKAGE_JOINT/id`` (M325): Waldron 6R / hybrid spatial overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WALDRON_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WALDRON_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WALDRON_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWaldronLinkageJoint
    model.lagmul_waldron_linkage_joints[block.user_id] = LagmulWaldronLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_dietmaier_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIETMAIER_LINKAGE_JOINT/id`` or ``/LAGMUL/DIETMAIER_LINKAGE_JOINT/id`` (M326): Dietmaier 6R spatial overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIETMAIER_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, link_angle_theta = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DIETMAIER_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DIETMAIER_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            link_angle_theta = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            link_angle_theta = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDietmaierLinkageJoint
    model.lagmul_dietmaier_linkage_joints[block.user_id] = LagmulDietmaierLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, link_angle_theta=link_angle_theta
    )




def read_lagmul_baker_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BAKER_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_LINKAGE_JOINT/id`` (M327): Baker spatial overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BAKER_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_d = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BAKER_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BAKER_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_d = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_d = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBakerLinkageJoint
    model.lagmul_baker_linkage_joints[block.user_id] = LagmulBakerLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_d=offset_distance_d
    )




def read_lagmul_wohlhart_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WOHLHART_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_LINKAGE_JOINT/id`` (M328): Wohlhart spatial overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WOHLHART_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WOHLHART_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WOHLHART_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWohlhartLinkageJoint
    model.lagmul_wohlhart_linkage_joints[block.user_id] = LagmulWohlhartLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_altmann_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALTMANN_LINKAGE_JOINT/id`` or ``/LAGMUL/ALTMANN_LINKAGE_JOINT/id`` (M329): Altmann spatial 6R line-symmetric overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALTMANN_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_d = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ALTMANN_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("ALTMANN_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_d = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_d = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAltmannLinkageJoint
    model.lagmul_altmann_linkage_joints[block.user_id] = LagmulAltmannLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_d=offset_distance_d
    )




def read_lagmul_wunderlich_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WUNDERLICH_LINKAGE_JOINT/id`` or ``/LAGMUL/WUNDERLICH_LINKAGE_JOINT/id`` (M330): Wunderlich spatial 6R overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WUNDERLICH_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_e = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WUNDERLICH_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WUNDERLICH_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_e = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_e = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWunderlichLinkageJoint
    model.lagmul_wunderlich_linkage_joints[block.user_id] = LagmulWunderlichLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_e=offset_distance_e
    )




def read_lagmul_delassus_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DELASSUS_LINKAGE_JOINT/id`` or ``/LAGMUL/DELASSUS_LINKAGE_JOINT/id`` (M331): Delassus spatial 6R overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DELASSUS_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DELASSUS_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DELASSUS_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDelassusLinkageJoint
    model.lagmul_delassus_linkage_joints[block.user_id] = LagmulDelassusLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_schatz_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SCHATZ_LINKAGE_JOINT/id`` or ``/LAGMUL/SCHATZ_LINKAGE_JOINT/id`` (M332): Schatz spatial 6R / turbula inversor overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SCHATZ_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SCHATZ_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SCHATZ_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSchatzLinkageJoint
    model.lagmul_schatz_linkage_joints[block.user_id] = LagmulSchatzLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_franke_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FRANKE_LINKAGE_JOINT/id`` or ``/LAGMUL/FRANKE_LINKAGE_JOINT/id`` (M333): Franke spatial 6R overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FRANKE_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FRANKE_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("FRANKE_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFrankeLinkageJoint
    model.lagmul_franke_linkage_joints[block.user_id] = LagmulFrankeLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_krames_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KRAMES_LINKAGE_JOINT/id`` or ``/LAGMUL/KRAMES_LINKAGE_JOINT/id`` (M334): Krames spatial 6R symmetrical overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KRAMES_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KRAMES_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KRAMES_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKramesLinkageJoint
    model.lagmul_krames_linkage_joints[block.user_id] = LagmulKramesLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_borel_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BOREL_LINKAGE_JOINT/id`` or ``/LAGMUL/BOREL_LINKAGE_JOINT/id`` (M335): Borel spatial 6R spherical-bivector overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BOREL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BOREL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BOREL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBorelLinkageJoint
    model.lagmul_borel_linkage_joints[block.user_id] = LagmulBorelLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_herve_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HERVE_LINKAGE_JOINT/id`` or ``/LAGMUL/HERVE_LINKAGE_JOINT/id`` (M336): Hervé spatial 6R isocline overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HERVE_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HERVE_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HERVE_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHerveLinkageJoint
    model.lagmul_herve_linkage_joints[block.user_id] = LagmulHerveLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_kong_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KONG_LINKAGE_JOINT/id`` or ``/LAGMUL/KONG_LINKAGE_JOINT/id`` (M337): Kong spatial 6R multi-loop / line-symmetric overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KONG_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KONG_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KONG_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKongLinkageJoint
    model.lagmul_kong_linkage_joints[block.user_id] = LagmulKongLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_grassmann_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRASSMANN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/GRASSMANN_SPATIAL_LINKAGE_JOINT/id`` (M422): Grassmann spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GRASSMANN_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("GRASSMANN_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("GRASSMANN_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulGrassmannSpatialLinkageJoint
    model.lagmul_grassmann_spatial_linkage_joints[block.user_id] = LagmulGrassmannSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_minkowski_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MINKOWSKI_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MINKOWSKI_SPATIAL_LINKAGE_JOINT/id`` (M423): Minkowski spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MINKOWSKI_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("MINKOWSKI_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MINKOWSKI_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulMinkowskiSpatialLinkageJoint
    model.lagmul_minkowski_spatial_linkage_joints[block.user_id] = LagmulMinkowskiSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_lobachevsky_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/id`` (M424): Lobachevsky spatial 6R hyperbolic multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("LOBACHEVSKY_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("LOBACHEVSKY_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulLobachevskySpatialLinkageJoint
    model.lagmul_lobachevsky_spatial_linkage_joints[block.user_id] = LagmulLobachevskySpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_poincare_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/POINCARE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/POINCARE_SPATIAL_LINKAGE_JOINT/id`` (M425): Poincaré spatial 6R hyperbolic multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/POINCARE_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("POINCARE_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("POINCARE_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPoincareSpatialLinkageJoint
    model.lagmul_poincare_spatial_linkage_joints[block.user_id] = LagmulPoincareSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_beltrami_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BELTRAMI_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BELTRAMI_SPATIAL_LINKAGE_JOINT/id`` (M426): Beltrami spatial 6R hyperbolic multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BELTRAMI_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BELTRAMI_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BELTRAMI_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBeltramiSpatialLinkageJoint
    model.lagmul_beltrami_spatial_linkage_joints[block.user_id] = LagmulBeltramiSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_clifford_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CLIFFORD_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/id`` (M427): Clifford spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CLIFFORD_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CLIFFORD_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CLIFFORD_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCliffordSpatialLinkageJoint
    model.lagmul_clifford_spatial_linkage_joints[block.user_id] = LagmulCliffordSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_conformal_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONFORMAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONFORMAL_SPATIAL_LINKAGE_JOINT/id`` (M428): Conformal spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONFORMAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CONFORMAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CONFORMAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulConformalSpatialLinkageJoint
    model.lagmul_conformal_spatial_linkage_joints[block.user_id] = LagmulConformalSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_projective_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROJECTIVE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PROJECTIVE_SPATIAL_LINKAGE_JOINT/id`` (M429): Projective spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PROJECTIVE_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PROJECTIVE_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PROJECTIVE_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulProjectiveSpatialLinkageJoint
    model.lagmul_projective_spatial_linkage_joints[block.user_id] = LagmulProjectiveSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_symplectic_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/id`` (M430): Symplectic spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SYMPLECTIC_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SYMPLECTIC_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSymplecticSpatialLinkageJoint
    model.lagmul_symplectic_spatial_linkage_joints[block.user_id] = LagmulSymplecticSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_homological_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` (M434): Homological spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HOMOLOGICAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HOMOLOGICAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHomologicalSpatialLinkageJoint
    model.lagmul_homological_spatial_linkage_joints[block.user_id] = LagmulHomologicalSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_cohomological_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` (M435): Cohomological spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCohomologicalSpatialLinkageJoint
    model.lagmul_cohomological_spatial_linkage_joints[block.user_id] = LagmulCohomologicalSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_topological_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/id`` (M433): Topological spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("TOPOLOGICAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("TOPOLOGICAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulTopologicalSpatialLinkageJoint
    model.lagmul_topological_spatial_linkage_joints[block.user_id] = LagmulTopologicalSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_algebraic_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/id`` (M432): Algebraic spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ALGEBRAIC_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("ALGEBRAIC_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAlgebraicSpatialLinkageJoint
    model.lagmul_algebraic_spatial_linkage_joints[block.user_id] = LagmulAlgebraicSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_differential_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/id`` (M433): Differential spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DIFFERENTIAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DIFFERENTIAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDifferentialSpatialLinkageJoint
    model.lagmul_differential_spatial_linkage_joints[block.user_id] = LagmulDifferentialSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_contact_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONTACT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONTACT_SPATIAL_LINKAGE_JOINT/id`` (M431): Contact spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONTACT_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CONTACT_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CONTACT_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulContactSpatialLinkageJoint
    model.lagmul_contact_spatial_linkage_joints[block.user_id] = LagmulContactSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_symplectic_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/id`` (M430): Symplectic spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SYMPLECTIC_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SYMPLECTIC_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSymplecticSpatialLinkageJoint
    model.lagmul_symplectic_spatial_linkage_joints[block.user_id] = LagmulSymplecticSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_conformal_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONFORMAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CONFORMAL_SPATIAL_LINKAGE_JOINT/id`` (M428): Conformal spatial 6R multivector multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONFORMAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CONFORMAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CONFORMAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulConformalSpatialLinkageJoint
    model.lagmul_conformal_spatial_linkage_joints[block.user_id] = LagmulConformalSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_hunt_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HUNT_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_LINKAGE_JOINT/id`` (M338): Hunt spatial 6R screw-symmetric / line-symmetric overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HUNT_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HUNT_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HUNT_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHuntLinkageJoint
    model.lagmul_hunt_linkage_joints[block.user_id] = LagmulHuntLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_baker_line_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BAKER_LINE_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_LINE_LINKAGE_JOINT/id`` (M339): Baker spatial 6R line-symmetric overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BAKER_LINE_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BAKER_LINE_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BAKER_LINE_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBakerLineLinkageJoint
    model.lagmul_baker_line_linkage_joints[block.user_id] = LagmulBakerLineLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_baker_plane_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BAKER_PLANE_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_PLANE_LINKAGE_JOINT/id`` (M340): Baker spatial 6R plane-symmetric overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BAKER_PLANE_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_f = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BAKER_PLANE_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BAKER_PLANE_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_f = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_f = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBakerPlaneLinkageJoint
    model.lagmul_baker_plane_linkage_joints[block.user_id] = LagmulBakerPlaneLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_f=offset_distance_f
    )




def read_lagmul_wohlhart_hybrid_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WOHLHART_HYBRID_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_HYBRID_LINKAGE_JOINT/id`` (M341): Wohlhart 6R spatial hybrid / line-plane overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WOHLHART_HYBRID_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_h = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WOHLHART_HYBRID_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WOHLHART_HYBRID_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_h = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_h = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWohlhartHybridLinkageJoint
    model.lagmul_wohlhart_hybrid_linkage_joints[block.user_id] = LagmulWohlhartHybridLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_h=offset_distance_h
    )




def read_lagmul_chen_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHEN_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEN_LINKAGE_JOINT/id`` (M342): Chen's 6R spatial large-displacement overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CHEN_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_e = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CHEN_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CHEN_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_e = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_e = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulChenLinkageJoint
    model.lagmul_chen_linkage_joints[block.user_id] = LagmulChenLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_e=offset_distance_e
    )




def read_lagmul_baker_symmetric_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BAKER_SYMMETRIC_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_SYMMETRIC_LINKAGE_JOINT/id`` (M343): Baker spatial 6R fully-symmetric overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BAKER_SYMMETRIC_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_r = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BAKER_SYMMETRIC_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BAKER_SYMMETRIC_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_r = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_r = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBakerSymmetricLinkageJoint
    model.lagmul_baker_symmetric_linkage_joints[block.user_id] = LagmulBakerSymmetricLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_r=offset_distance_r
    )




def read_lagmul_altmann_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALTMANN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALTMANN_SPATIAL_LINKAGE_JOINT/id`` (M344): Altmann spatial 6R non-spherical multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALTMANN_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ALTMANN_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("ALTMANN_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAltmannSpatialLinkageJoint
    model.lagmul_altmann_spatial_linkage_joints[block.user_id] = LagmulAltmannSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_dietmaier_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIETMAIER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIETMAIER_SPATIAL_LINKAGE_JOINT/id`` (M345): Dietmaier spatial 6R variable-geometry overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIETMAIER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_t = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DIETMAIER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DIETMAIER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_t = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_t = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDietmaierSpatialLinkageJoint
    model.lagmul_dietmaier_spatial_linkage_joints[block.user_id] = LagmulDietmaierSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_t=offset_distance_t
    )




def read_lagmul_wohlhart_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WOHLHART_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_SPATIAL_LINKAGE_JOINT/id`` (M346): Wohlhart spatial 6R skew-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WOHLHART_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_u = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WOHLHART_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WOHLHART_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_u = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_u = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWohlhartSpatialLinkageJoint
    model.lagmul_wohlhart_spatial_linkage_joints[block.user_id] = LagmulWohlhartSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_u=offset_distance_u
    )




def read_lagmul_hunt_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HUNT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/id`` (M347): Hunt spatial 6R variable screw axis overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HUNT_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_v = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HUNT_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HUNT_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_v = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_v = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHuntSpatialLinkageJoint
    model.lagmul_hunt_spatial_linkage_joints[block.user_id] = LagmulHuntSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_v=offset_distance_v
    )




def read_lagmul_chen_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHEN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEN_SPATIAL_LINKAGE_JOINT/id`` (M348): Chen spatial 6R large-displacement multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CHEN_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_w = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CHEN_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CHEN_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_w = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_w = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulChenSpatialLinkageJoint
    model.lagmul_chen_spatial_linkage_joints[block.user_id] = LagmulChenSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_w=offset_distance_w
    )




def read_lagmul_baker_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BAKER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/id`` (M349): Baker spatial 6R variable-geometry multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BAKER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_r = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BAKER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BAKER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_r = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_r = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBakerSpatialLinkageJoint
    model.lagmul_baker_spatial_linkage_joints[block.user_id] = LagmulBakerSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_r=offset_distance_r
    )




def read_lagmul_waldron_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WALDRON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WALDRON_SPATIAL_LINKAGE_JOINT/id`` (M350): Waldron spatial 6R hybrid multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WALDRON_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WALDRON_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WALDRON_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWaldronSpatialLinkageJoint
    model.lagmul_waldron_spatial_linkage_joints[block.user_id] = LagmulWaldronSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_bricard_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BRICARD_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BRICARD_SPATIAL_LINKAGE_JOINT/id`` (M351): Bricard spatial 6R triaxial multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BRICARD_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BRICARD_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BRICARD_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBricardSpatialLinkageJoint
    model.lagmul_bricard_spatial_linkage_joints[block.user_id] = LagmulBricardSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_bennett_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BENNETT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BENNETT_SPATIAL_LINKAGE_JOINT/id`` (M352): Bennett spatial skew-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BENNETT_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BENNETT_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BENNETT_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBennettSpatialLinkageJoint
    model.lagmul_bennett_spatial_linkage_joints[block.user_id] = LagmulBennettSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_myard_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MYARD_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MYARD_SPATIAL_LINKAGE_JOINT/id`` (M353): Myard spatial 6R plane-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MYARD_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("MYARD_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MYARD_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulMyardSpatialLinkageJoint
    model.lagmul_myard_spatial_linkage_joints[block.user_id] = LagmulMyardSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_goldberg_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GOLDBERG_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/GOLDBERG_SPATIAL_LINKAGE_JOINT/id`` (M354): Goldberg spatial 6R variable-angle multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GOLDBERG_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("GOLDBERG_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("GOLDBERG_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulGoldbergSpatialLinkageJoint
    model.lagmul_goldberg_spatial_linkage_joints[block.user_id] = LagmulGoldbergSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_sarrus_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SARRUS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SARRUS_SPATIAL_LINKAGE_JOINT/id`` (M355): Sarrus spatial 6R rectilinear multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SARRUS_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SARRUS_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SARRUS_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSarrusSpatialLinkageJoint
    model.lagmul_sarrus_spatial_linkage_joints[block.user_id] = LagmulSarrusSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_delassus_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DELASSUS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DELASSUS_SPATIAL_LINKAGE_JOINT/id`` (M356): Delassus spatial 6R cylindrical-surfaced multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DELASSUS_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DELASSUS_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DELASSUS_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDelassusSpatialLinkageJoint
    model.lagmul_delassus_spatial_linkage_joints[block.user_id] = LagmulDelassusSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_wohlhart_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WOHLHART_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_SPATIAL_LINKAGE_JOINT/id`` (M357): Wohlhart spatial 6R hybrid-symmetry multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WOHLHART_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WOHLHART_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WOHLHART_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWohlhartSpatialLinkageJoint
    model.lagmul_wohlhart_spatial_linkage_joints[block.user_id] = LagmulWohlhartSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_altmann_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALTMANN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALTMANN_SPATIAL_LINKAGE_JOINT/id`` (M358): Altmann spatial 6R line-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALTMANN_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ALTMANN_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("ALTMANN_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAltmannSpatialLinkageJoint
    model.lagmul_altmann_spatial_linkage_joints[block.user_id] = LagmulAltmannSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_baker_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BAKER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/id`` (M359): Baker spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BAKER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BAKER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BAKER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBakerSpatialLinkageJoint
    model.lagmul_baker_spatial_linkage_joints[block.user_id] = LagmulBakerSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_dietmaier_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIETMAIER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIETMAIER_SPATIAL_LINKAGE_JOINT/id`` (M360): Dietmaier spatial 6R isomorphic multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIETMAIER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DIETMAIER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DIETMAIER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDietmaierSpatialLinkageJoint
    model.lagmul_dietmaier_spatial_linkage_joints[block.user_id] = LagmulDietmaierSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_waldron_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WALDRON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WALDRON_SPATIAL_LINKAGE_JOINT/id`` (M361): Waldron spatial 6R hybrid multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WALDRON_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WALDRON_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WALDRON_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWaldronSpatialLinkageJoint
    model.lagmul_waldron_spatial_linkage_joints[block.user_id] = LagmulWaldronSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_hunt_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HUNT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/id`` (M362): Hunt spatial 6R special-symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HUNT_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HUNT_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HUNT_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHuntSpatialLinkageJoint
    model.lagmul_hunt_spatial_linkage_joints[block.user_id] = LagmulHuntSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_chen_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHEN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEN_SPATIAL_LINKAGE_JOINT/id`` (M363): Chen spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CHEN_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CHEN_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CHEN_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulChenSpatialLinkageJoint
    model.lagmul_chen_spatial_linkage_joints[block.user_id] = LagmulChenSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_wunderlich_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WUNDERLICH_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WUNDERLICH_SPATIAL_LINKAGE_JOINT/id`` (M364): Wunderlich spatial 6R bistable multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WUNDERLICH_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WUNDERLICH_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WUNDERLICH_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWunderlichSpatialLinkageJoint
    model.lagmul_wunderlich_spatial_linkage_joints[block.user_id] = LagmulWunderlichSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_konnok_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KONNOK_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT/id`` (M365): Konnok spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KONNOK_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KONNOK_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KONNOK_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKonnokSpatialLinkageJoint
    model.lagmul_konnok_spatial_linkage_joints[block.user_id] = LagmulKonnokSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_pfurner_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PFURNER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT/id`` (M366): Pfurner spatial 6R parallel multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PFURNER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PFURNER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PFURNER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPfurnerSpatialLinkageJoint
    model.lagmul_pfurner_spatial_linkage_joints[block.user_id] = LagmulPfurnerSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_phillips_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PHILLIPS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/id`` (M367): Phillips spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PHILLIPS_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PHILLIPS_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PHILLIPS_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPhillipsSpatialLinkageJoint
    model.lagmul_phillips_spatial_linkage_joints[block.user_id] = LagmulPhillipsSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_stevens_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STEVENS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STEVENS_SPATIAL_LINKAGE_JOINT/id`` (M368): Stevens spatial 6R symmetric multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STEVENS_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("STEVENS_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("STEVENS_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulStevensSpatialLinkageJoint
    model.lagmul_stevens_spatial_linkage_joints[block.user_id] = LagmulStevensSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_baker_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M369): Baker hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BAKER_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BAKER_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBakerHybridSpatialLinkageJoint
    model.lagmul_baker_hybrid_spatial_linkage_joints[block.user_id] = LagmulBakerHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_waldron_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M370): Waldron hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWaldronHybridSpatialLinkageJoint
    model.lagmul_waldron_hybrid_spatial_linkage_joints[block.user_id] = LagmulWaldronHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_chen_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M371): Chen hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CHEN_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CHEN_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulChenHybridSpatialLinkageJoint
    model.lagmul_chen_hybrid_spatial_linkage_joints[block.user_id] = LagmulChenHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_wohlhart_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M372): Wohlhart hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 0 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWohlhartHybridSpatialLinkageJoint
    model.lagmul_wohlhart_hybrid_spatial_linkage_joints[block.user_id] = LagmulWohlhartHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_maverick_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M373): Maverick hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulMaverickHybridSpatialLinkageJoint
    model.lagmul_maverick_hybrid_spatial_linkage_joints[block.user_id] = LagmulMaverickHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_krause_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M374): Krause hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 0 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKrauseHybridSpatialLinkageJoint
    model.lagmul_krause_hybrid_spatial_linkage_joints[block.user_id] = LagmulKrauseHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_sturgess_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M375): Sturgess hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSturgessHybridSpatialLinkageJoint
    model.lagmul_sturgess_hybrid_spatial_linkage_joints[block.user_id] = LagmulSturgessHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_bevan_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M376): Bevan hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBevanHybridSpatialLinkageJoint
    model.lagmul_bevan_hybrid_spatial_linkage_joints[block.user_id] = LagmulBevanHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_heinrichs_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M377): Heinrichs hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHeinrichsHybridSpatialLinkageJoint
    model.lagmul_heinrichs_hybrid_spatial_linkage_joints[block.user_id] = LagmulHeinrichsHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_altmann_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M378): Altmann hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAltmannHybridSpatialLinkageJoint
    model.lagmul_altmann_hybrid_spatial_linkage_joints[block.user_id] = LagmulAltmannHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_kirkpatrick_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M379): Kirkpatrick hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKirkpatrickHybridSpatialLinkageJoint
    model.lagmul_kirkpatrick_hybrid_spatial_linkage_joints[block.user_id] = LagmulKirkpatrickHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_alexander_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M380): Alexander hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAlexanderHybridSpatialLinkageJoint
    model.lagmul_alexander_hybrid_spatial_linkage_joints[block.user_id] = LagmulAlexanderHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_chung_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M381): Chung hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulChungHybridSpatialLinkageJoint
    model.lagmul_chung_hybrid_spatial_linkage_joints[block.user_id] = LagmulChungHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_stevens_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M382): Stevens hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulStevensHybridSpatialLinkageJoint
    model.lagmul_stevens_hybrid_spatial_linkage_joints[block.user_id] = LagmulStevensHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_baker_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BAKER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/id`` (M383): Baker spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BAKER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BAKER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BAKER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBakerSpatialLinkageJoint
    model.lagmul_baker_spatial_linkage_joints[block.user_id] = LagmulBakerSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_dietmeier_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIETMEIER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIETMEIER_SPATIAL_LINKAGE_JOINT/id`` (M384): Dietmeier spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIETMEIER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DIETMEIER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DIETMEIER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDietmeierSpatialLinkageJoint
    model.lagmul_dietmeier_spatial_linkage_joints[block.user_id] = LagmulDietmeierSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_hunt_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HUNT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/id`` (M385): Hunt spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HUNT_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HUNT_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HUNT_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHuntSpatialLinkageJoint
    model.lagmul_hunt_spatial_linkage_joints[block.user_id] = LagmulHuntSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_pfurner_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PFURNER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT/id`` (M386): Pfurner spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PFURNER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PFURNER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PFURNER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPfurnerSpatialLinkageJoint
    model.lagmul_pfurner_spatial_linkage_joints[block.user_id] = LagmulPfurnerSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_phillips_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PHILLIPS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/id`` (M387): Phillips spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PHILLIPS_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PHILLIPS_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PHILLIPS_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPhillipsSpatialLinkageJoint
    model.lagmul_phillips_spatial_linkage_joints[block.user_id] = LagmulPhillipsSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_konnok_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KONNOK_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT/id`` (M388): Konnok spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KONNOK_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KONNOK_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KONNOK_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKonnokSpatialLinkageJoint
    model.lagmul_konnok_spatial_linkage_joints[block.user_id] = LagmulKonnokSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_wohlhart_hybrid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/id`` (M389): Wohlhart hybrid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulWohlhartHybridSpatialLinkageJoint
    model.lagmul_wohlhart_hybrid_spatial_linkage_joints[block.user_id] = LagmulWohlhartHybridSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_maverick_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAVERICK_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/MAVERICK_SPATIAL_LINKAGE_JOINT/id`` (M390): Maverick spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAVERICK_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("MAVERICK_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAVERICK_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 and f2[1].strip() else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulMaverickSpatialLinkageJoint
    model.lagmul_maverick_spatial_linkage_joints[block.user_id] = LagmulMaverickSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_krause_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KRAUSE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KRAUSE_SPATIAL_LINKAGE_JOINT/id`` (M391): Krause spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KRAUSE_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KRAUSE_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KRAUSE_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKrauseSpatialLinkageJoint
    model.lagmul_krause_spatial_linkage_joints[block.user_id] = LagmulKrauseSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_sturgess_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STURGESS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/STURGESS_SPATIAL_LINKAGE_JOINT/id`` (M392): Sturgess spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STURGESS_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("STURGESS_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("STURGESS_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSturgessSpatialLinkageJoint
    model.lagmul_sturgess_spatial_linkage_joints[block.user_id] = LagmulSturgessSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_hunt_special_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/id`` (M393): Hunt special spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHuntSpecialSpatialLinkageJoint
    model.lagmul_hunt_special_spatial_linkage_joints[block.user_id] = LagmulHuntSpecialSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_albrecht_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALBRECHT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/ALBRECHT_SPATIAL_LINKAGE_JOINT/id`` (M394): Albrecht spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALBRECHT_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("ALBRECHT_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("ALBRECHT_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulAlbrechtSpatialLinkageJoint
    model.lagmul_albrecht_spatial_linkage_joints[block.user_id] = LagmulAlbrechtSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_kirson_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KIRSON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/KIRSON_SPATIAL_LINKAGE_JOINT/id`` (M395): Kirson spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KIRSON_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("KIRSON_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("KIRSON_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulKirsonSpatialLinkageJoint
    model.lagmul_kirson_spatial_linkage_joints[block.user_id] = LagmulKirsonSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_diesel_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIESEL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DIESEL_SPATIAL_LINKAGE_JOINT/id`` (M396): Diesel spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIESEL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DIESEL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DIESEL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDieselSpatialLinkageJoint
    model.lagmul_diesel_spatial_linkage_joints[block.user_id] = LagmulDieselSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_tchebychev_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/id`` (M397): Chebyshev / Tchebychev spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("TCHEBYCHEV_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("TCHEBYCHEV_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulTchebychevSpatialLinkageJoint
    model.lagmul_tchebychev_spatial_linkage_joints[block.user_id] = LagmulTchebychevSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_sylvester_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SYLVESTER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SYLVESTER_SPATIAL_LINKAGE_JOINT/id`` (M398): Sylvester spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SYLVESTER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SYLVESTER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SYLVESTER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSylvesterSpatialLinkageJoint
    model.lagmul_sylvester_spatial_linkage_joints[block.user_id] = LagmulSylvesterSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_cauchy_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CAUCHY_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CAUCHY_SPATIAL_LINKAGE_JOINT/id`` (M399): Cauchy spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CAUCHY_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CAUCHY_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CAUCHY_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCauchySpatialLinkageJoint
    model.lagmul_cauchy_spatial_linkage_joints[block.user_id] = LagmulCauchySpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_cayley_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CAYLEY_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CAYLEY_SPATIAL_LINKAGE_JOINT/id`` (M400): Cayley spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CAYLEY_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CAYLEY_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CAYLEY_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCayleySpatialLinkageJoint
    model.lagmul_cayley_spatial_linkage_joints[block.user_id] = LagmulCayleySpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_euclid_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EUCLID_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/EUCLID_SPATIAL_LINKAGE_JOINT/id`` (M401): Euclid spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/EUCLID_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("EUCLID_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("EUCLID_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulEuclidSpatialLinkageJoint
    model.lagmul_euclid_spatial_linkage_joints[block.user_id] = LagmulEuclidSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_fermat_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FERMAT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FERMAT_SPATIAL_LINKAGE_JOINT/id`` (M402): Fermat spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FERMAT_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FERMAT_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("FERMAT_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFermatSpatialLinkageJoint
    model.lagmul_fermat_spatial_linkage_joints[block.user_id] = LagmulFermatSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_pascal_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PASCAL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/PASCAL_SPATIAL_LINKAGE_JOINT/id`` (M403): Pascal spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PASCAL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("PASCAL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PASCAL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPascalSpatialLinkageJoint
    model.lagmul_pascal_spatial_linkage_joints[block.user_id] = LagmulPascalSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_descartes_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DESCARTES_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/DESCARTES_SPATIAL_LINKAGE_JOINT/id`` (M404): Descartes spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DESCARTES_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("DESCARTES_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DESCARTES_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulDescartesSpatialLinkageJoint
    model.lagmul_descartes_spatial_linkage_joints[block.user_id] = LagmulDescartesSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_leibniz_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LEIBNIZ_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/LEIBNIZ_SPATIAL_LINKAGE_JOINT/id`` (M405): Leibniz spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LEIBNIZ_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("LEIBNIZ_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("LEIBNIZ_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulLeibnizSpatialLinkageJoint
    model.lagmul_leibniz_spatial_linkage_joints[block.user_id] = LagmulLeibnizSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_newton_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/NEWTON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/NEWTON_SPATIAL_LINKAGE_JOINT/id`` (M406): Newton spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/NEWTON_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("NEWTON_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("NEWTON_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulNewtonSpatialLinkageJoint
    model.lagmul_newton_spatial_linkage_joints[block.user_id] = LagmulNewtonSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_gauss_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GAUSS_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/GAUSS_SPATIAL_LINKAGE_JOINT/id`` (M407): Gauss spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GAUSS_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("GAUSS_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("GAUSS_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulGaussSpatialLinkageJoint
    model.lagmul_gauss_spatial_linkage_joints[block.user_id] = LagmulGaussSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_euler_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EULER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/EULER_SPATIAL_LINKAGE_JOINT/id`` (M408): Euler spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/EULER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("EULER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("EULER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulEulerSpatialLinkageJoint
    model.lagmul_euler_spatial_linkage_joints[block.user_id] = LagmulEulerSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_lagrange_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LAGRANGE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/LAGRANGE_SPATIAL_LINKAGE_JOINT/id`` (M409): Lagrange spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LAGRANGE_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("LAGRANGE_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("LAGRANGE_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulLagrangeSpatialLinkageJoint
    model.lagmul_lagrange_spatial_linkage_joints[block.user_id] = LagmulLagrangeSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_laplace_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LAPLACE_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/LAPLACE_SPATIAL_LINKAGE_JOINT/id`` (M410): Laplace spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LAPLACE_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("LAPLACE_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("LAPLACE_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulLaplaceSpatialLinkageJoint
    model.lagmul_laplace_spatial_linkage_joints[block.user_id] = LagmulLaplaceSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_fourier_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FOURIER_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FOURIER_SPATIAL_LINKAGE_JOINT/id`` (M411): Fourier spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FOURIER_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FOURIER_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("FOURIER_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFourierSpatialLinkageJoint
    model.lagmul_fourier_spatial_linkage_joints[block.user_id] = LagmulFourierSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_poisson_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/POISSON_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/POISSON_SPATIAL_LINKAGE_JOINT/id`` (M412): Poisson spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/POISSON_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("POISSON_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("POISSON_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulPoissonSpatialLinkageJoint
    model.lagmul_poisson_spatial_linkage_joints[block.user_id] = LagmulPoissonSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_bessel_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BESSEL_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BESSEL_SPATIAL_LINKAGE_JOINT/id`` (M413): Bessel spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BESSEL_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BESSEL_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BESSEL_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBesselSpatialLinkageJoint
    model.lagmul_bessel_spatial_linkage_joints[block.user_id] = LagmulBesselSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_riemann_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RIEMANN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/RIEMANN_SPATIAL_LINKAGE_JOINT/id`` (M414): Riemann spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RIEMANN_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("RIEMANN_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("RIEMANN_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulRiemannSpatialLinkageJoint
    model.lagmul_riemann_spatial_linkage_joints[block.user_id] = LagmulRiemannSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_hilbert_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HILBERT_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HILBERT_SPATIAL_LINKAGE_JOINT/id`` (M415): Hilbert spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HILBERT_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HILBERT_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HILBERT_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHilbertSpatialLinkageJoint
    model.lagmul_hilbert_spatial_linkage_joints[block.user_id] = LagmulHilbertSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_banach_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BANACH_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/BANACH_SPATIAL_LINKAGE_JOINT/id`` (M416): Banach spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/BANACH_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("BANACH_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("BANACH_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulBanachSpatialLinkageJoint
    model.lagmul_banach_spatial_linkage_joints[block.user_id] = LagmulBanachSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_sobolev_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SOBOLEV_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/SOBOLEV_SPATIAL_LINKAGE_JOINT/id`` (M417): Sobolev spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SOBOLEV_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("SOBOLEV_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SOBOLEV_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulSobolevSpatialLinkageJoint
    model.lagmul_sobolev_spatial_linkage_joints[block.user_id] = LagmulSobolevSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_frechet_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FRECHET_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/FRECHET_SPATIAL_LINKAGE_JOINT/id`` (M418): Fréchet spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FRECHET_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("FRECHET_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("FRECHET_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulFrechetSpatialLinkageJoint
    model.lagmul_frechet_spatial_linkage_joints[block.user_id] = LagmulFrechetSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_hausdorff_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HAUSDORFF_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/HAUSDORFF_SPATIAL_LINKAGE_JOINT/id`` (M419): Hausdorff spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HAUSDORFF_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("HAUSDORFF_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("HAUSDORFF_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulHausdorffSpatialLinkageJoint
    model.lagmul_hausdorff_spatial_linkage_joints[block.user_id] = LagmulHausdorffSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_cartan_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CARTAN_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CARTAN_SPATIAL_LINKAGE_JOINT/id`` (M420): Cartan spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CARTAN_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CARTAN_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CARTAN_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCartanSpatialLinkageJoint
    model.lagmul_cartan_spatial_linkage_joints[block.user_id] = LagmulCartanSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )




def read_lagmul_clifford_spatial_linkage_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CLIFFORD_SPATIAL_LINKAGE_JOINT/id`` or ``/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/id`` (M421): Clifford spatial 6R multi-loop overconstrained kinematic mechanism joint constraint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CLIFFORD_SPATIAL_LINKAGE_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, node3, stiff, skew_id, tol = 0, 0, 0, 1e6, 0, 1e-6
    link_len_a, link_len_b, twist_angle_alpha, offset_distance_s = 0.0, 0.0, 0.0, 0.0
    if block.fixed and "," not in cards[0].raw:
        f1 = cards[0].cut("CLIFFORD_SPATIAL_LINKAGE_JOINT_1")
        node1 = _ival(f1[0]) if len(f1) > 0 else 0
        node2 = _ival(f1[1]) if len(f1) > 1 else 0
        node3 = _ival(f1[2]) if len(f1) > 2 else 0
        stiff = _fval(f1[3], 1e6) if len(f1) > 3 and f1[3].strip() else 1e6
        skew_id = _ival(f1[4], 0) if len(f1) > 4 else 0
        tol = _fval(f1[5], 1e-6) if len(f1) > 5 and f1[5].strip() else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CLIFFORD_SPATIAL_LINKAGE_JOINT_2")
            link_len_a = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            link_len_b = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            twist_angle_alpha = _fval(f2[2], 0.0) if len(f2) > 2 and f2[2].strip() else 0.0
            offset_distance_s = _fval(f2[3], 0.0) if len(f2) > 3 and f2[3].strip() else 0.0
    else:
        toks1 = cards[0].tokens()
        node1 = int(float(toks1[0].rstrip(','))) if len(toks1) > 0 else 0
        node2 = int(float(toks1[1].rstrip(','))) if len(toks1) > 1 else 0
        node3 = int(float(toks1[2].rstrip(','))) if len(toks1) > 2 else 0
        stiff = float(toks1[3].rstrip(',')) if len(toks1) > 3 else 1e6
        skew_id = int(float(toks1[4].rstrip(','))) if len(toks1) > 4 else 0
        tol = float(toks1[5].rstrip(',')) if len(toks1) > 5 else 1e-6

        if len(cards) > 1 and not cards[1].is_blank:
            toks2 = cards[1].tokens()
            link_len_a = float(toks2[0].rstrip(',')) if len(toks2) > 0 else 0.0
            link_len_b = float(toks2[1].rstrip(',')) if len(toks2) > 1 else 0.0
            twist_angle_alpha = float(toks2[2].rstrip(',')) if len(toks2) > 2 else 0.0
            offset_distance_s = float(toks2[3].rstrip(',')) if len(toks2) > 3 else 0.0

    from ...model.entities import LagmulCliffordSpatialLinkageJoint
    model.lagmul_clifford_spatial_linkage_joints[block.user_id] = LagmulCliffordSpatialLinkageJoint(
        id=block.user_id, title=title, node1=node1, node2=node2, node3=node3,
        stiff=stiff, skew_id=skew_id, tol=tol,
        link_len_a=link_len_a, link_len_b=link_len_b,
        twist_angle_alpha=twist_angle_alpha, offset_distance_s=offset_distance_s
    )
