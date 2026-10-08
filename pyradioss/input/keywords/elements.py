# -*- coding: utf-8 -*-
"""
Element definition readers - /KEYWORD blocks -> Model.

/BRICK, /SHELL, /TETRA, /QUAD, /TRUSS, /SPRING, /BEAM, /SOLID,
/PENTA, /PYRA, /ENG and the ply/laminate/stack cards.

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





def read_parameter(block: KeywordBlock, model: Model,
                   log: MessageLog) -> None:
    """``/PARAMETER/<scope>/<REAL|INTEGER|TEXT>/param_ID`` (M37): the
    named value was already substituted into every ``&NAME`` reference by
    the deck reader (see deck_reader._finalize_deck — a textual
    substitution preserving the fixed columns, exactly like the
    reference reader).  Nothing further to do here; TEXT parameters are
    NOT substituted (an &NAME reference to one then surfaces as an
    unresolved token in its consumer — the honest failure)."""
    subtype = block.parts[2].upper() if len(block.parts) > 2 else ""
    if subtype not in ("REAL", "INTEGER", "INT") and block.cards:
        log.warning(f"/PARAMETER/{'/'.join(block.parts[1:])}: only "
                    f"REAL/INTEGER parameters are substituted — &"
                    f"references to this one stay unresolved",
                    block.source)




def read_quad(block, model, log):
    """``/QUAD``: 4-node 2D solid element."""
    return _read_elems(block, model, log, "QUAD", 4)




def read_penta6(block, model, log):
    """``/PENTA6/part_ID``, ``/PENTA/part_ID``, or ``/WEDGE/part_ID`` (M590):
    6-node wedge/prism solid element (elem_ID + 6 node IDs).
    """
    _read_elems(block, model, log, "PENTA6", 6)




def read_pyra(block, model, log):
    """``/PYRA5/part_ID`` or ``/PYRA/part_ID``: 5-node pyramid solid element (elem_ID + 5 node IDs)."""
    _read_elems(block, model, log, "PYRA5", 5)




def read_tria3(block, model, log):
    """``/TRIA3/part_ID``: 3-node 2D solid element (elem_ID + 3 node IDs)."""
    _read_elems(block, model, log, "TRIA3", 3)




def read_brick(block, model, log):
    """``/BRICK/part_ID``: 8-node solids (elem_ID + 8 node IDs).
    Degenerated bricks with 4 distinct nodes (the classic tetra-in-brick
    convention, e.g. n1 n2 n3 n3 n5 n5 n5 n5) are converted to /TETRA4
    elements by the Starter; other repeated-node patterns (penta/pyramid)
    are rejected with a clear error (see initialization.py)."""
    _read_elems(block, model, log, "BRICK", 8)




def read_tshell(block, model, log):
    """``/TSHELL/part_ID``: 8-node thick shell elements (elem_ID + 8 node IDs)."""
    _read_elems(block, model, log, "TSHELL", 8)




def read_tetra4(block, model, log):
    """``/TETRA4/part_ID``: 4-node solids (elem_ID + 4 node IDs, base
    triangle 1-2-3 counter-clockwise seen from node 4). Uses the same
    /PROP/TYPE14 (SOLID) property as bricks."""
    _read_elems(block, model, log, "TETRA4", 4)




def read_tetra10(block, model, log):
    """``/TETRA10/part_ID``: 10-node solids.
    Format is usually 2 cards per element:
    Card 1: elem_id
    Card 2: n1..n10
    (or free format equivalent).
    """
    part_id = block.user_id
    if part_id is None:
        log.error("/TETRA10 block without part id", block.source)
        return
    
    # In some decks, it's 2 cards per element. In others, it might be free format on 1 line.
    # We will gather all integers in the block and chunk them by 11 (1 ID + 10 nodes).
    ints = []
    for card in block.cards:
        ints.extend(card.ints())
            
    if len(ints) % 11 != 0:
        log.error(f"/TETRA10 block: expected multiple of 11 values (ID + 10 nodes), got {len(ints)}", block.source)
        # We will parse what we can
    
    for i in range(0, len(ints) - 10, 11):
        elem_id = ints[i]
        nodes = ints[i+1:i+11]
        model.raw_elems["TETRA10"].append((elem_id, part_id, nodes))





def read_shel16(block, model, log):
    """``/SHEL16/part_ID``: 16-node thick shells.
    Format is 3 cards per element:
    Card 1: id, n1..n8
    Card 2: n9..n12
    Card 3: n13..n16
    """
    part_id = block.user_id
    if part_id is None:
        log.error("/SHEL16 block without part id", block.source)
        return
    
    if len(block.cards) % 3 != 0:
        log.error(f"/SHEL16 block has {len(block.cards)} cards, expected a multiple of 3", block.source)
        return

    for i in range(0, len(block.cards), 3):
        c1 = block.cards[i]
        c2 = block.cards[i+1]
        c3 = block.cards[i+2]
        
        if block.fixed:
            f1 = c1.cut("ELEM_IDS")[:9]
            f2 = c2.cut("ELEM_IDS")[:4]
            f3 = c3.cut("ELEM_IDS")[:4]
            if not f1[0]:
                log.error("/SHEL16 card without an element id", c1.source)
                continue
            if any(not s for s in f1[1:]) or any(not s for s in f2) or any(not s for s in f3):
                log.error("/SHEL16 card needs 16 ids", c1.source)
                continue
            t = [int(s) for s in f1] + [int(s) for s in f2] + [int(s) for s in f3]
        else:
            t = c1.ints() + c2.ints() + c3.ints()
            if len(t) < 17:
                log.error(f"/SHEL16 card needs 17 ids, got {len(t)}", c1.source)
                continue
        model.raw_elems["SHEL16"].append((t[0], part_id, t[1:17]))




def read_bric20(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BRIC20/part_ID`` or ``/HEXA20/part_ID`` (M122): 20-node quadratic hexahedral solids.
    Format is 2 cards per element:
    Card 1: elem_id, n1..n10
    Card 2: n11..n20
    (or free-format stream of integers chunked by 21).
    """
    part_id = block.user_id
    if part_id is None:
        log.error(f"/{block.key0} block without part id", block.source)
        return

    if block.fixed and len(block.cards) % 3 == 0:
        for i in range(0, len(block.cards), 3):
            c1 = block.cards[i]
            c2 = block.cards[i+1]
            c3 = block.cards[i+2]
            f1 = c1.cut("ELEM_BRIC20_1")
            f2 = c2.cut("ELEM_BRIC20_2")
            f3 = c3.cut("ELEM_BRIC20_3")
            if not f1[0].strip():
                continue
            try:
                elem_id = _ival(f1[0])
                nodes = ([_ival(x) for x in f1[1:] if x.strip()] +
                         [_ival(x) for x in f2 if x.strip()] +
                         [_ival(x) for x in f3 if x.strip()])
                if len(nodes) == 20:
                    model.raw_elems["BRIC20"].append((elem_id, part_id, nodes))
                else:
                    log.error(f"/{block.key0} {elem_id}: expected 20 nodes, got {len(nodes)}", c1.source)
            except ValueError as e:
                log.error(f"/{block.key0}: {e}", c1.source)
    elif block.fixed and len(block.cards) % 2 == 0:
        for i in range(0, len(block.cards), 2):
            c1 = block.cards[i]
            c2 = block.cards[i+1]
            f1 = c1.cut("ELEM_BRIC20_1")
            f2 = c2.cut("ELEM_BRIC20_2")
            if not f1[0].strip():
                continue
            try:
                elem_id = _ival(f1[0])
                nodes = [_ival(x) for x in f1[1:] if x.strip()] + [_ival(x) for x in f2 if x.strip()]
                if len(nodes) == 20:
                    model.raw_elems["BRIC20"].append((elem_id, part_id, nodes))
                else:
                    log.error(f"/{block.key0} {elem_id}: expected 20 nodes, got {len(nodes)}", c1.source)
            except ValueError as e:
                log.error(f"/{block.key0}: {e}", c1.source)
    else:
        ints = []
        for card in block.cards:
            ints.extend(card.ints())
        if len(ints) % 21 != 0:
            log.warning(f"/{block.key0} block: expected multiple of 21 values (ID + 20 nodes), got {len(ints)}", block.source)
        for i in range(0, len(ints) - 20, 21):
            elem_id = ints[i]
            nodes = ints[i+1:i+21]
            model.raw_elems["BRIC20"].append((elem_id, part_id, nodes))






def read_shell(block, model, log):
    """``/SHELL/part_ID``: 4-node shells (elem_ID + 4 node IDs)."""
    _read_elems(block, model, log, "SHELL", 4)




def read_sh3n(block, model, log):
    """``/SH3N/part_ID``: 3-node shells (elem_ID + 3 node IDs). Uses the
    same /PROP/TYPE1 (SHELL) property as 4-node shells."""
    _read_elems(block, model, log, "SH3N", 3)




def read_truss(block, model, log):
    """``/TRUSS/part_ID``: 2-node trusses (elem_ID + 2 node IDs)."""
    _read_elems(block, model, log, "TRUSS", 2)




def read_spring(block, model, log):
    """``/SPRING/part_ID``: 2-node or 3-node springs (elem_ID + 2 or 3 node IDs)."""
    part_id = block.user_id
    if part_id is None:
        log.error(f"/{block.key0} block without part id", block.source)
        return
    if block.fixed:
        for card in block.cards:
            f = card.cut("ELEM_IDS")
            if not f or not f[0].strip():
                continue
            try:
                elem_id = int(f[0])
                n1 = int(f[1]) if len(f) > 1 and f[1].strip() else 0
                n2 = int(f[2]) if len(f) > 2 and f[2].strip() else 0
                n3 = int(f[3]) if len(f) > 3 and f[3].strip() else 0
                if n3 != 0:
                    model.raw_elems["SPRING"].append((elem_id, part_id, [n1, n2, n3]))
                else:
                    model.raw_elems["SPRING"].append((elem_id, part_id, [n1, n2]))
            except ValueError as e:
                log.error(f"/SPRING {f[0]}: {e}", card.source)
    else:
        for card in block.cards:
            t = card.ints()
            if len(t) < 3:
                log.error(f"/SPRING card needs at least 3 ids (elem, n1, n2), got {len(t)}", card.source)
                continue
            elem_id = t[0]
            n1 = t[1]
            n2 = t[2]
            if len(t) >= 4 and t[3] != 0:
                n3 = t[3]
                model.raw_elems["SPRING"].append((elem_id, part_id, [n1, n2, n3]))
            else:
                model.raw_elems["SPRING"].append((elem_id, part_id, [n1, n2]))




def read_beam(block, model, log):
    """``/BEAM/part_ID``: 2-node beams + orientation node (elem_ID + N1 N2
    N3). N3 orients the local y axis (in the N1-N2-N3 plane) and carries
    neither mass nor force — it may be any node, including a standalone
    one (which the mass check then freezes, harmlessly)."""
    _read_elems(block, model, log, "BEAM", 3)




def read_inertia_part(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INERTIA/PART/inertia_ID`` (M169): Part inertia modifier."""
    from ...model.entities import InertiaPart
    pid = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INERTIA/PART/{pid}: missing data card", block.source)
        return

    part_id = 0
    skew_id = 0
    iflag = 0
    mass = 0.0
    xg, yg, zg = 0.0, 0.0, 0.0
    ixx, iyy, izz, ixy, iyz, izx = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("INERTIA_PART_1")
        part_id = _ival(f1[0]) if len(f1) > 0 else 0
        skew_id = _ival(f1[1]) if len(f1) > 1 else 0
        iflag = _ival(f1[2]) if len(f1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("INERTIA_PART_2")
            mass = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            xg = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            yg = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            zg = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("INERTIA_PART_3")
            ixx = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            iyy = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            izz = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            ixy = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
            iyz = _fval(f3[4], 0.0) if len(f3) > 4 else 0.0
            izx = _fval(f3[5], 0.0) if len(f3) > 5 else 0.0
    else:
        t1 = cards[0].tokens()
        part_id = int(float(t1[0])) if len(t1) > 0 else 0
        skew_id = int(float(t1[1])) if len(t1) > 1 else 0
        iflag = int(float(t1[2])) if len(t1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            mass = float(t2[0]) if len(t2) > 0 else 0.0
            xg = float(t2[1]) if len(t2) > 1 else 0.0
            yg = float(t2[2]) if len(t2) > 2 else 0.0
            zg = float(t2[3]) if len(t2) > 3 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            ixx = float(t3[0]) if len(t3) > 0 else 0.0
            iyy = float(t3[1]) if len(t3) > 1 else 0.0
            izz = float(t3[2]) if len(t3) > 2 else 0.0
            ixy = float(t3[3]) if len(t3) > 3 else 0.0
            iyz = float(t3[4]) if len(t3) > 4 else 0.0
            izx = float(t3[5]) if len(t3) > 5 else 0.0

    model.inertia_parts[pid] = InertiaPart(
        id=pid, title=title, part_id=part_id, skew_id=skew_id, iflag=iflag,
        mass=mass, xg=xg, yg=yg, zg=zg,
        ixx=ixx, iyy=iyy, izz=izz, ixy=ixy, iyz=iyz, izx=izx,
    )




def read_ply(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PLY/ply_id`` (M100): Composite ply definition.

    Fortran origin: ``starter/source/model/laminate/leclamply.F``.
    Card format:
        card 1: title
        card 2: Mat_id  Thick  [skew_ID]
    """
    from ...model.entities import Ply
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/PLY/{block.user_id}: missing data cards", block.source)
        return

    if block.fixed:
        c = cards[0].cut("PLY_1")
        mat_id = _ival(c[0]) if len(c) > 0 else 0
        thick = _fval(c[1]) if len(c) > 1 else 0.0
        skew_id = 0
    else:
        toks = cards[0].tokens()
        mat_id = int(float(toks[0])) if len(toks) > 0 else 0
        thick = float(toks[1]) if len(toks) > 1 else 0.0
        skew_id = int(float(toks[2])) if len(toks) > 2 else 0

    model.plies[block.user_id] = Ply(
        id=block.user_id, title=title, mat_id=mat_id, thick=thick, skew_id=skew_id
    )




def read_laminate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LAMINATE/laminate_id`` (M100): Composite laminate stack definition.

    Fortran origin: ``starter/source/model/laminate/leclam.F``.
    Card format:
        card 1: title
        card 2: Ply_id  Phi  Zi
        card 3 (optional): Minterply
    """
    from ...model.entities import Laminate, LaminatePly
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/LAMINATE/{block.user_id}: missing data cards", block.source)
        return

    plies: List[LaminatePly] = []
    idx = 0
    while idx < len(cards):
        c = cards[idx]
        if c.is_blank:
            idx += 1
            continue
        if block.fixed:
            c1 = c.cut("LAMINATE_LAYER")
            ply_id = _ival(c1[0]) if len(c1) > 0 else 0
            phi = _fval(c1[1]) if len(c1) > 1 else 0.0
            zi = _fval(c1[2]) if len(c1) > 2 else 0.0
            idx += 1
            mat_inter = 0
            if idx < len(cards) and not cards[idx].is_blank:
                c2 = cards[idx].cut("LAMINATE_INTERPLY")
                mat_inter = _ival(c2[0]) if len(c2) > 0 else 0
                idx += 1
        else:
            toks = c.tokens()
            ply_id = int(float(toks[0])) if len(toks) > 0 else 0
            phi = float(toks[1]) if len(toks) > 1 else 0.0
            zi = float(toks[2]) if len(toks) > 2 else 0.0
            idx += 1
            mat_inter = 0
            if idx < len(cards) and not cards[idx].is_blank:
                t2 = cards[idx].tokens()
                if len(t2) == 1:
                    mat_inter = int(float(t2[0]))
                    idx += 1
        plies.append(LaminatePly(ply_id=ply_id, phi=phi, zi=zi, mat_interply=mat_inter))

    model.laminates[block.user_id] = Laminate(id=block.user_id, title=title, plies=plies)




def read_stack(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STACK/stack_ID`` (M127): Composite laminate stack definition."""
    from ...model.entities import Stack, StackPly
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/STACK/{block.user_id}: missing data card", block.source)
        return

    ishell, ismstr, ish3n, idrill, z0 = 0, 0, 0, 0, 0.0
    hm, hf, hr, dm, dn = 0.01, 0.01, 0.01, 0.0, 0.0
    istrain, ashear, iint, ithick = 0, 0.833333, 0, 0
    vx, vy, vz, skew_id, iorth, ipos, ip = 0.0, 0.0, 0.0, 0, 0, 0, 0

    if block.fixed:
        f1 = cards[0].cut("STACK_1")
        ishell = _ival(f1[0]) if len(f1) > 0 else 0
        ismstr = _ival(f1[1]) if len(f1) > 1 else 0
        ish3n = _ival(f1[2]) if len(f1) > 2 else 0
        idrill = _ival(f1[3]) if len(f1) > 3 else 0
        z0 = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("STACK_2")
            hm = _fval(f2[0], 0.01) if len(f2) > 0 else 0.01
            hf = _fval(f2[1], 0.01) if len(f2) > 1 else 0.01
            hr = _fval(f2[2], 0.01) if len(f2) > 2 else 0.01
            dm = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
            dn = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("STACK_3")
            istrain = _ival(f3[1]) if len(f3) > 1 else 0
            ashear = _fval(f3[2], 0.833333) if len(f3) > 2 else 0.833333
            iint = _ival(f3[4]) if len(f3) > 4 else 0
            ithick = _ival(f3[6]) if len(f3) > 6 else 0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("STACK_4")
            vx = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            vy = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            vz = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0
            skew_id = _ival(f4[3]) if len(f4) > 3 else 0
            iorth = _ival(f4[4]) if len(f4) > 4 else 0
            ipos = _ival(f4[5]) if len(f4) > 5 else 0
            ip = _ival(f4[6]) if len(f4) > 6 else 0

        ply_cards = cards[4:]
    else:
        t1 = cards[0].tokens()
        ishell = int(float(t1[0])) if len(t1) > 0 else 0
        ismstr = int(float(t1[1])) if len(t1) > 1 else 0
        ish3n = int(float(t1[2])) if len(t1) > 2 else 0
        idrill = int(float(t1[3])) if len(t1) > 3 else 0
        z0 = float(t1[4]) if len(t1) > 4 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            hm = float(t2[0]) if len(t2) > 0 else 0.01
            hf = float(t2[1]) if len(t2) > 1 else 0.01
            hr = float(t2[2]) if len(t2) > 2 else 0.01
            dm = float(t2[3]) if len(t2) > 3 else 0.0
            dn = float(t2[4]) if len(t2) > 4 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            istrain = int(float(t3[0])) if len(t3) > 0 else 0
            ashear = float(t3[1]) if len(t3) > 1 else 0.833333
            iint = int(float(t3[2])) if len(t3) > 2 else 0
            ithick = int(float(t3[3])) if len(t3) > 3 else 0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            vx = float(t4[0]) if len(t4) > 0 else 0.0
            vy = float(t4[1]) if len(t4) > 1 else 0.0
            vz = float(t4[2]) if len(t4) > 2 else 0.0
            skew_id = int(float(t4[3])) if len(t4) > 3 else 0
            iorth = int(float(t4[4])) if len(t4) > 4 else 0
            ipos = int(float(t4[5])) if len(t4) > 5 else 0
            ip = int(float(t4[6])) if len(t4) > 6 else 0

        ply_cards = cards[4:]

    plies = []
    current_sub_id = None
    sub_plies = []
    from ...model.entities import SubLaminate, SubLaminatePly

    for c in ply_cards:
        if c.is_blank:
            continue
        if block.fixed:
            raw_upper = c.raw.strip().upper()
            if raw_upper.startswith("SUB"):
                if current_sub_id is not None and sub_plies:
                    model.sub_laminates[current_sub_id] = SubLaminate(
                        id=current_sub_id, title=f"SubLaminate_{current_sub_id}", plies=sub_plies
                    )
                    sub_plies = []
                f_hdr = c.cut("SUB_LAMINATE_1")
                current_sub_id = _ival(f_hdr[1]) if len(f_hdr) > 1 and f_hdr[1].strip() else (block.user_id or 1)
                continue
            f = c.cut("STACK_PLY")
            pid = _ival(f[0])
            phi = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            zi = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            ptf = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            fw = _fval(f[4], 1.0) if len(f) > 4 else 1.0
        else:
            t = c.tokens()
            if not t:
                continue
            if t[0].upper() == "SUB":
                if current_sub_id is not None and sub_plies:
                    model.sub_laminates[current_sub_id] = SubLaminate(
                        id=current_sub_id, title=f"SubLaminate_{current_sub_id}", plies=sub_plies
                    )
                    sub_plies = []
                current_sub_id = int(float(t[1])) if len(t) > 1 else (block.user_id or 1)
                continue
            pid = int(float(t[0])) if len(t) > 0 else 0
            phi = float(t[1]) if len(t) > 1 else 0.0
            zi = float(t[2]) if len(t) > 2 else 0.0
            ptf = float(t[3]) if len(t) > 3 else 0.0
            fw = float(t[4]) if len(t) > 4 else 1.0

        if pid > 0:
            plies.append(StackPly(ply_id=pid, phi=phi, zi=zi, p_thick_fail=ptf, f_weight=fw))
            if current_sub_id is not None:
                sub_plies.append(SubLaminatePly(ply_id=pid, phi=phi, zi=zi, p_thick_fail=ptf, f_weight=fw))

    if current_sub_id is not None and sub_plies:
        model.sub_laminates[current_sub_id] = SubLaminate(
            id=current_sub_id, title=f"SubLaminate_{current_sub_id}", plies=sub_plies
        )

    stack_id = block.user_id if block.user_id is not None else 1
    model.stacks[stack_id] = Stack(
        id=stack_id, title=title, ishell=ishell, ismstr=ismstr, ish3n=ish3n, idrill=idrill,
        z0=z0, hm=hm, hf=hf, hr=hr, dm=dm, dn=dn, istrain=istrain, ashear=ashear,
        iint=iint, ithick=ithick, vx=vx, vy=vy, vz=vz, skew_id=skew_id, iorth=iorth,
        ipos=ipos, ip=ip, plies=plies,
    )




def read_sub_laminate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SUBLAMINATE/sub_ID`` or ``/STACK/SUB_LAMINATE/sub_ID`` (M165): Sub-laminate composite ply stack definition.

    Fortran origin: ``LAMINATE/sub_laminate_p51.cfg`` and ``LAMINATE/stack_sub_laminate.cfg``.
    """
    from ...model.entities import SubLaminate, SubLaminatePly
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    sub_id = block.user_id if block.user_id is not None else 1
    plies = []

    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            if c.raw[:3].upper() == "SUB" or len(c.raw) < 15:
                f_hdr = c.cut("SUB_LAMINATE_1")
                if len(f_hdr) > 1 and f_hdr[1].strip():
                    sub_id = _ival(f_hdr[1], sub_id)
                continue
            f = c.cut("SUB_LAMINATE_PLY")
            try:
                pid = int(f[0].strip()) if f and f[0].strip() else 0
            except ValueError:
                title = c.raw.strip()
                continue
            phi = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            zi = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            ptf = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            fw = _fval(f[4], 1.0) if len(f) > 4 else 1.0
        else:
            t = c.tokens()
            if not t:
                continue
            if t[0].upper() == "SUB":
                if len(t) > 1:
                    sub_id = int(float(t[1]))
                continue
            try:
                pid = int(float(t[0])) if len(t) > 0 else 0
            except ValueError:
                title = c.raw.strip()
                continue
            phi = float(t[1]) if len(t) > 1 else 0.0
            zi = float(t[2]) if len(t) > 2 else 0.0
            ptf = float(t[3]) if len(t) > 3 else 0.0
            fw = float(t[4]) if len(t) > 4 else 1.0
        if pid > 0:
            plies.append(SubLaminatePly(ply_id=pid, phi=phi, zi=zi, p_thick_fail=ptf, f_weight=fw))

    model.sub_laminates[sub_id] = SubLaminate(
        id=sub_id,
        title=title,
        plies=plies,
    )




def read_stamping_params_m58(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STAMPING/stamp_ID`` — Stamping simulation parameters."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or (block.fixed and cards[0].is_blank):
        log.error(f"/STAMPING/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f1 = cards[0].cut("STAMPING_1") if "STAMPING_1" in CARD_LAYOUTS else _fixed_vals(cards[0], [10, 10, 20, 20, 20])
        part_id = _ival(f1[0]) if len(f1) > 0 else 0
        tool_id = _ival(f1[1]) if len(f1) > 1 else 0
        gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
        fric = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
        vel = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
    else:
        t = cards[0].tokens()
        part_id = int(float(t[0])) if len(t) > 0 else 0
        tool_id = int(float(t[1])) if len(t) > 1 else 0
        gap = float(t[2]) if len(t) > 2 else 0.0
        fric = float(t[3]) if len(t) > 3 else 0.0
        vel = float(t[4]) if len(t) > 4 else 0.0
    from ...model.entities import Stamping
    model.stampings[block.user_id] = Stamping(
        id=block.user_id, title=title, part_id=part_id, tool_id=tool_id,
        gap=gap, fric=fric, vel=vel
    )




# ============================================================================
# Global defaults (M68)
# ============================================================================

def read_def_shell(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DEF_SHELL`` — global shell formulation defaults (M68/M101).

    Single data card (NO title card)::

        ISHELL  ISMSTR  ITHICK  IPLAS  ISTRAIN  (20-char gap)  ISH3N  IDRILL

    Mirrors ``hm_read_defshell.F``.  The values are stored on
    ``model.def_shell`` and consulted when /PROP/SHELL fields are 0."""
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        return
    c = cards[0]
    if block.fixed:
        vals = c.cut("DEF_SHELL_1")
    else:
        vals = c.tokens()
    def _iv(idx):
        try:
            return int(float(vals[idx]))
        except (IndexError, ValueError):
            return 0
    model.def_shell = {
        'ishell': _iv(0), 'ismstr': _iv(1), 'ithick': _iv(2),
        'iplas': _iv(3), 'istrain': _iv(4),
        # fixed: index 5 is 20-char gap, ISH3N at 6, IDRILL at 7
        # free:  no gap field, ISH3N at 5, IDRILL at 6
        'ish3n': _iv(6 if block.fixed else 5),
        'idrill': _iv(7 if block.fixed else 6),
    }




def read_def_solid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DEF_SOLID`` — global solid formulation defaults (M68/M101).

    Single data card (NO title card)::

        ISOLID  ISMSTR  ICPRE  (10-char gap)  ITETRA4  ITETRA10  IMAS  IFRAME

    Mirrors ``hm_read_defsolid.F``."""
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        return
    c = cards[0]
    if block.fixed:
        vals = c.cut("DEF_SOLID_1")
    else:
        vals = c.tokens()
    def _iv(idx):
        try:
            return int(float(vals[idx]))
        except (IndexError, ValueError):
            return 0
    model.def_solid = {
        'isolid': _iv(0), 'ismstr': _iv(1), 'icpre': _iv(2),
        # fixed: index 3 is 10-char gap, ITETRA4 at 4, ITETRA10 at 5, IMAS at 6, IFRAME at 7
        # free:  no gap field, ITETRA4 at 3, ITETRA10 at 4, IMAS at 5, IFRAME at 6
        'itetra4': _iv(4 if block.fixed else 3),
        'itetra10': _iv(5 if block.fixed else 4),
        'imas': _iv(6 if block.fixed else 5),
        'iframe': _iv(7 if block.fixed else 6),
    }




def read_tetra(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TETRA/part_ID`` (M194): 4-node or 10-node tetrahedral solids."""
    for card in block.cards:
        if not card.is_blank and not card.raw.strip().startswith("#"):
            t = card.ints()
            if len(t) >= 11:
                read_tetra10(block, model, log)
                return
            break
    read_tetra4(block, model, log)




def read_stamping(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STAMPING`` (M113): Sheet metal forming stamping history input::

        optional card 1:  #HF TIME SCALE <val>
        data line list
    """
    from ...model.entities import StampingInit

    time_scale = 1.0
    data_lines = []
    cards = block.fixed_cards() if block.fixed else block.cards
    for c in cards:
        line_str = c.raw.strip() if hasattr(c, "raw") else str(c).strip()
        if not line_str:
            continue
        if line_str.upper().startswith("#HF TIME SCALE"):
            parts = line_str.split()
            if len(parts) >= 4:
                try:
                    time_scale = float(parts[3])
                except ValueError:
                    pass
        else:
            data_lines.append(line_str)

    model.stamping_inits.append(StampingInit(time_scale=time_scale, data_lines=data_lines))




def read_retractor(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RETRACTOR[/<subtype>]/retractor_ID`` (M106)::

        card 1: title
        card 2: EL_ID  Node_ID  Elem_size
        card 3: Sens_ID1  Pullout  Fct_ID1  Fct_ID2  Yscale1  Xscale1
        card 4: Sens_ID2  Tens_typ  Force  Fct_ID3  Yscale2  Xscale2
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else "SPRING"
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RETRACTOR/{block.user_id}: missing data card", block.source)
        return

    el_id, node_id = 0, 0
    elem_size = 0.0
    sens_id1, fct_id1, fct_id2 = 0, 0, 0
    pullout = 0.0
    yscale1, xscale1 = 1.0, 1.0
    sens_id2, tens_typ, fct_id3 = 0, 0, 0
    force = 0.0
    yscale2, xscale2 = 1.0, 1.0

    if block.fixed:
        f1 = cards[0].cut("RETRACTOR_1")
        el_id = _ival(f1[0]) if len(f1) > 0 else 0
        node_id = _ival(f1[1]) if len(f1) > 1 else 0
        elem_size = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("RETRACTOR_2")
            sens_id1 = _ival(f2[0]) if len(f2) > 0 else 0
            pullout = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            fct_id1 = _ival(f2[2]) if len(f2) > 2 else 0
            fct_id2 = _ival(f2[3]) if len(f2) > 3 else 0
            yscale1 = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0
            xscale1 = _fval(f2[5], 1.0) if len(f2) > 5 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("RETRACTOR_3")
            sens_id2 = _ival(f3[0]) if len(f3) > 0 else 0
            tens_typ = _ival(f3[1]) if len(f3) > 1 else 0
            force = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            fct_id3 = _ival(f3[3]) if len(f3) > 3 else 0
            yscale2 = _fval(f3[4], 1.0) if len(f3) > 4 else 1.0
            xscale2 = _fval(f3[5], 1.0) if len(f3) > 5 else 1.0
    else:
        t1 = cards[0].tokens()
        el_id = int(float(t1[0])) if len(t1) > 0 else 0
        node_id = int(float(t1[1])) if len(t1) > 1 else 0
        elem_size = float(t1[2]) if len(t1) > 2 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            sens_id1 = int(float(t2[0])) if len(t2) > 0 else 0
            pullout = float(t2[1]) if len(t2) > 1 else 0.0
            fct_id1 = int(float(t2[2])) if len(t2) > 2 else 0
            fct_id2 = int(float(t2[3])) if len(t2) > 3 else 0
            yscale1 = float(t2[4]) if len(t2) > 4 else 1.0
            xscale1 = float(t2[5]) if len(t2) > 5 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            sens_id2 = int(float(t3[0])) if len(t3) > 0 else 0
            tens_typ = int(float(t3[1])) if len(t3) > 1 else 0
            force = float(t3[2]) if len(t3) > 2 else 0.0
            fct_id3 = int(float(t3[3])) if len(t3) > 3 else 0
            yscale2 = float(t3[4]) if len(t3) > 4 else 1.0
            xscale2 = float(t3[5]) if len(t3) > 5 else 1.0

    model.retractors[block.user_id] = Retractor(
        id=block.user_id, title=title, subtype=subtype, el_id=el_id,
        node_id=node_id, elem_size=elem_size, sens_id1=sens_id1,
        pullout=pullout, fct_id1=fct_id1, fct_id2=fct_id2,
        yscale1=yscale1, xscale1=xscale1, sens_id2=sens_id2,
        tens_typ=tens_typ, force=force, fct_id3=fct_id3,
        yscale2=yscale2, xscale2=xscale2,
    )




def read_slipring(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SLIPRING[/<subtype>]/slipring_ID`` (M106)::

        card 1: title
        card 2: El1_ID  El2_ID  Node_ID  Node_ID2  Sens_ID  Flow_flag  A  Ed_factor
        card 3: Fct_ID1  Fct_ID2  Fricd  Xscale1  Yscale2  Xscale2
        card 4: Fct_ID3  Fct_ID4  Frics  Xscale3  Yscale4  Xscale4
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else "SPRING"
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SLIPRING/{block.user_id}: missing data card", block.source)
        return

    el_id1, el_id2, node_id, node_id2 = 0, 0, 0, 0
    sens_id, flow_flag = 0, 0
    a, ed_factor = 0.0, 0.0
    fct_id1, fct_id2 = 0, 0
    fricd = 0.0
    xscale1, yscale2, xscale2 = 1.0, 1.0, 1.0
    fct_id3, fct_id4 = 0, 0
    frics = 0.0
    xscale3, yscale4, xscale4 = 1.0, 1.0, 1.0

    if block.fixed:
        if subtype == "SHELL":
            f1 = cards[0].cut("SLIPRING_SHELL_1")
            el_id1 = _ival(f1[0]) if len(f1) > 0 else 0
            el_id2 = _ival(f1[1]) if len(f1) > 1 else 0
            node_id = _ival(f1[2]) if len(f1) > 2 else 0
            sens_id = _ival(f1[3]) if len(f1) > 3 else 0
            flow_flag = _ival(f1[4]) if len(f1) > 4 else 0
            a = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0
            ed_factor = _fval(f1[6], 0.0) if len(f1) > 6 else 0.0
        else:
            f1 = cards[0].cut("SLIPRING_1")
            el_id1 = _ival(f1[0]) if len(f1) > 0 else 0
            el_id2 = _ival(f1[1]) if len(f1) > 1 else 0
            node_id = _ival(f1[2]) if len(f1) > 2 else 0
            node_id2 = _ival(f1[3]) if len(f1) > 3 else 0
            sens_id = _ival(f1[4]) if len(f1) > 4 else 0
            flow_flag = _ival(f1[5]) if len(f1) > 5 else 0
            a = _fval(f1[6], 0.0) if len(f1) > 6 else 0.0
            ed_factor = _fval(f1[7], 0.0) if len(f1) > 7 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SLIPRING_2")
            fct_id1 = _ival(f2[0]) if len(f2) > 0 else 0
            fct_id2 = _ival(f2[1]) if len(f2) > 1 else 0
            fricd = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            xscale1 = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            yscale2 = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0
            xscale2 = _fval(f2[5], 1.0) if len(f2) > 5 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("SLIPRING_3")
            fct_id3 = _ival(f3[0]) if len(f3) > 0 else 0
            fct_id4 = _ival(f3[1]) if len(f3) > 1 else 0
            frics = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            xscale3 = _fval(f3[3], 1.0) if len(f3) > 3 else 1.0
            yscale4 = _fval(f3[4], 1.0) if len(f3) > 4 else 1.0
            xscale4 = _fval(f3[5], 1.0) if len(f3) > 5 else 1.0
    else:
        t1 = cards[0].tokens()
        if subtype == "SHELL":
            el_id1 = int(float(t1[0])) if len(t1) > 0 else 0
            el_id2 = int(float(t1[1])) if len(t1) > 1 else 0
            node_id = int(float(t1[2])) if len(t1) > 2 else 0
            sens_id = int(float(t1[3])) if len(t1) > 3 else 0
            flow_flag = int(float(t1[4])) if len(t1) > 4 else 0
            a = float(t1[5]) if len(t1) > 5 else 0.0
            ed_factor = float(t1[6]) if len(t1) > 6 else 0.0
        else:
            el_id1 = int(float(t1[0])) if len(t1) > 0 else 0
            el_id2 = int(float(t1[1])) if len(t1) > 1 else 0
            node_id = int(float(t1[2])) if len(t1) > 2 else 0
            node_id2 = int(float(t1[3])) if len(t1) > 3 else 0
            sens_id = int(float(t1[4])) if len(t1) > 4 else 0
            flow_flag = int(float(t1[5])) if len(t1) > 5 else 0
            a = float(t1[6]) if len(t1) > 6 else 0.0
            ed_factor = float(t1[7]) if len(t1) > 7 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            fct_id1 = int(float(t2[0])) if len(t2) > 0 else 0
            fct_id2 = int(float(t2[1])) if len(t2) > 1 else 0
            fricd = float(t2[2]) if len(t2) > 2 else 0.0
            xscale1 = float(t2[3]) if len(t2) > 3 else 1.0
            yscale2 = float(t2[4]) if len(t2) > 4 else 1.0
            xscale2 = float(t2[5]) if len(t2) > 5 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fct_id3 = int(float(t3[0])) if len(t3) > 0 else 0
            fct_id4 = int(float(t3[1])) if len(t3) > 1 else 0
            frics = float(t3[2]) if len(t3) > 2 else 0.0
            xscale3 = float(t3[3]) if len(t3) > 3 else 1.0
            yscale4 = float(t3[4]) if len(t3) > 4 else 1.0
            xscale4 = float(t3[5]) if len(t3) > 5 else 1.0

    model.sliprings[block.user_id] = Slipring(
        id=block.user_id, title=title, subtype=subtype, el_id1=el_id1,
        el_id2=el_id2, node_id=node_id, node_id2=node_id2, sens_id=sens_id,
        flow_flag=flow_flag, a=a, ed_factor=ed_factor, fct_id1=fct_id1,
        fct_id2=fct_id2, fricd=fricd, xscale1=xscale1, yscale2=yscale2,
        xscale2=xscale2, fct_id3=fct_id3, fct_id4=fct_id4, frics=frics,
        xscale3=xscale3, yscale4=yscale4, xscale4=xscale4,
    )
    if subtype in ("SHELL", "SH"):
        model.slipring_shells[block.user_id] = SlipringShell(
            id=block.user_id, title=title, el_set1=el_id1, el_set2=el_id2,
            node_set=node_id, sens_id=sens_id, flow_flag=flow_flag,
            a=a, ed_factor=ed_factor, fric_d=fricd, fric_s=frics,
            fct_id1=fct_id1, fct_id2=fct_id2, fct_id3=fct_id3, fct_id4=fct_id4,
            xscale1=xscale1, xscale2=xscale2, yscale2=yscale2,
            xscale3=xscale3, xscale4=xscale4, yscale4=yscale4,
        )




def read_seatbelt(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SEATBELT/id`` (M150, M197): Complete seatbelt system assembly.

    Fortran origin: ``starter/source/tools/seatbelts/create_seatbelt.F``.
    """
    if len(block.parts) > 1:
        sub = block.parts[1].upper()
        if sub in ("PRETENSIONER", "PRETEN", "PRETENSION"):
            read_pretensioner(block, model, log)
            return
        elif sub in ("RETRACTOR", "RETRACT"):
            read_retractor(block, model, log)
            return
        elif sub in ("SLIPRING", "SLIP"):
            read_slipring(block, model, log)
            return
    from ...model.entities import SeatbeltSystem
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    retractor_ids = []
    slipring_ids = []
    element_ids = []
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("SEATBELT_1")
            vals = [_ival(x) for x in f if x.strip()]
        else:
            vals = [int(float(x)) for x in cards[0].tokens()]
        if len(vals) > 0 and vals[0] > 0:
            retractor_ids.append(vals[0])
        if len(vals) > 1 and vals[1] > 0:
            slipring_ids.append(vals[1])
        if len(vals) > 2:
            element_ids.extend([v for v in vals[2:] if v > 0])
    model.seatbelt_systems[block.user_id] = SeatbeltSystem(
        id=block.user_id, title=title,
        retractor_ids=retractor_ids, slipring_ids=slipring_ids, element_ids=element_ids
    )




def read_pretensioner(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PRETENSIONER[/<subtype>]/id`` or ``/SEATBELT/PRETENSIONER/id`` (M197): Seatbelt pretensioner."""
    from ...model.entities import Pretensioner
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/PRETENSIONER/{block.user_id}: missing data card", block.source)
        return

    sens_id, fct_id = 0, 0
    fscale = 1.0
    tstart = 0.0
    vmax, amax, reinf = 0.0, 0.0, 0.0
    i_type = 0
    retractor_id, slipring_id = 0, 0
    element_ids = []

    if block.fixed:
        f1 = valid_cards[0].cut("PRETENSIONER_1")
        sens_id = _ival(f1[0]) if len(f1) > 0 else 0
        fct_id = _ival(f1[1]) if len(f1) > 1 else 0
        fscale = _fval(f1[2], 1.0) if len(f1) > 2 else 1.0
        tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
        vmax = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
        amax = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0
        reinf = _fval(f1[6], 0.0) if len(f1) > 6 else 0.0
        i_type = _ival(f1[7]) if len(f1) > 7 else 0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("PRETENSIONER_2")
            vals = [_ival(x) for x in f2 if x.strip()]
            if len(vals) > 0:
                retractor_id = vals[0]
            if len(vals) > 1:
                slipring_id = vals[1]
            if len(vals) > 2:
                element_ids.extend([v for v in vals[2:] if v > 0])
    else:
        t1 = valid_cards[0].tokens()
        sens_id = int(float(t1[0])) if len(t1) > 0 else 0
        fct_id = int(float(t1[1])) if len(t1) > 1 else 0
        fscale = float(t1[2]) if len(t1) > 2 else 1.0
        tstart = float(t1[3]) if len(t1) > 3 else 0.0
        vmax = float(t1[4]) if len(t1) > 4 else 0.0
        amax = float(t1[5]) if len(t1) > 5 else 0.0
        reinf = float(t1[6]) if len(t1) > 6 else 0.0
        i_type = int(float(t1[7])) if len(t1) > 7 else 0

        if len(valid_cards) > 1:
            vals = [int(float(x)) for x in valid_cards[1].tokens()]
            if len(vals) > 0:
                retractor_id = vals[0]
            if len(vals) > 1:
                slipring_id = vals[1]
            if len(vals) > 2:
                element_ids.extend([v for v in vals[2:] if v > 0])

    pret = Pretensioner(
        id=block.user_id or 0, title=title, sens_id=sens_id, fct_id=fct_id,
        fscale=fscale, tstart=tstart, vmax=vmax, amax=amax, reinf=reinf,
        i_type=i_type, retractor_id=retractor_id, slipring_id=slipring_id,
        element_ids=element_ids,
        params={
            "sens_id": sens_id, "fct_id": fct_id, "fscale": fscale, "tstart": tstart,
            "vmax": vmax, "amax": amax, "reinf": reinf, "i_type": i_type,
            "retractor_id": retractor_id, "slipring_id": slipring_id, "element_ids": element_ids
        }
    )
    model.pretensioners[block.user_id or 0] = pret




def read_drape(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DRAPE/drape_ID`` (M107, M157): Composite fabric draping definition."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    did = block.user_id if block.user_id is not None else 1
    slices = []
    entity_keywords = {"SHELL", "SH3N", "GRSHEL", "GRSH3N", "ELEMS", "SETS", "PART", "GRPART"}
    i = 0
    while i < len(cards):
        card = cards[i]
        if not card.is_blank:
            if block.fixed:
                first_word = card.raw[:10].strip().upper()
                if first_word in entity_keywords:
                    hdr = card.cut("DRAPE_SLICE_HDR")
                    etype = hdr[0].strip().upper() if len(hdr) > 0 else ""
                    eid = _ival(hdr[1]) if len(hdr) > 1 else 0
                    thinning, theta_slice, mat_id, npt_slice = 1.0, 0.0, 0, 1
                    if len(card.raw) > 20 and card.raw[20:].strip():
                        from ..card_layouts import split_fixed
                        row = split_fixed(card.raw[20:], [20, 20, 10, 10])
                        thinning = _fval(row[0], 1.0) if len(row) > 0 else 1.0
                        theta_slice = _fval(row[1]) if len(row) > 1 else 0.0
                        mat_id = _ival(row[2]) if len(row) > 2 else 0
                        npt_slice = _ival(row[3], 1) if len(row) > 3 else 1
                    elif i + 1 < len(cards) and not cards[i+1].is_blank and cards[i+1].raw[:10].strip().upper() not in entity_keywords:
                        row = cards[i+1].cut("DRAPE_SLICE_ROW")
                        thinning = _fval(row[0], 1.0) if len(row) > 0 else 1.0
                        theta_slice = _fval(row[1]) if len(row) > 1 else 0.0
                        mat_id = _ival(row[2]) if len(row) > 2 else 0
                        npt_slice = _ival(row[3], 1) if len(row) > 3 else 1
                        i += 1
                    slices.append({
                        "entity_type": etype,
                        "entity_id": eid,
                        "thinning": thinning,
                        "theta_slice": theta_slice,
                        "mat_id": mat_id,
                        "npt_slice": npt_slice,
                        "raw": card.raw.strip(),
                    })
                else:
                    slices.append({"raw": card.raw.strip()})
            else:
                toks = card.tokens()
                if toks:
                    first_tok = toks[0].upper()
                    if first_tok in entity_keywords:
                        etype = first_tok
                        eid = int(float(toks[1])) if len(toks) > 1 else 0
                        thinning, theta_slice, mat_id, npt_slice = 1.0, 0.0, 0, 1
                        if len(toks) > 2:
                            thinning = float(toks[2])
                            theta_slice = float(toks[3]) if len(toks) > 3 else 0.0
                            mat_id = int(float(toks[4])) if len(toks) > 4 else 0
                            npt_slice = int(float(toks[5])) if len(toks) > 5 else 1
                        elif i + 1 < len(cards) and not cards[i+1].is_blank:
                            next_toks = cards[i+1].tokens()
                            if next_toks and next_toks[0].upper() not in entity_keywords:
                                thinning = float(next_toks[0]) if len(next_toks) > 0 else 1.0
                                theta_slice = float(next_toks[1]) if len(next_toks) > 1 else 0.0
                                mat_id = int(float(next_toks[2])) if len(next_toks) > 2 else 0
                                npt_slice = int(float(next_toks[3])) if len(next_toks) > 3 else 1
                                i += 1
                        slices.append({
                            "entity_type": etype,
                            "entity_id": eid,
                            "thinning": thinning,
                            "theta_slice": theta_slice,
                            "mat_id": mat_id,
                            "npt_slice": npt_slice,
                            "raw": card.raw.strip(),
                        })
                    else:
                        slices.append({"raw": card.raw.strip()})
        i += 1
    model.drapes[did] = Drape(id=did, title=title, slices=slices)




def read_shfra(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SHFRA/V4`` (M117): Shell local coordinate framing formulation flag."""
    model.shfra_v4 = True




def read_eng_damp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/DAMP`` or ``/DAMP`` (M208): Global Rayleigh damping controls."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_DAMP_1")
            model.damp_alpha = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            model.damp_beta = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            model.damp_tstart = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            model.damp_tstop = _fval(f[3], 1.0e30) if len(f) > 3 else 1.0e30
        else:
            toks = cards[0].tokens()
            model.damp_alpha = float(toks[0]) if len(toks) > 0 else 0.0
            model.damp_beta = float(toks[1]) if len(toks) > 1 else 0.0
            model.damp_tstart = float(toks[2]) if len(toks) > 2 else 0.0
            model.damp_tstop = float(toks[3]) if len(toks) > 3 else 1.0e30




def read_eng_sub_cycle(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/SUB_CYCLE`` or ``/SUB_CYCLE`` (M209): Element and contact time step sub-cycling controls."""
    model.sub_cycle_enabled = True
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_SUB_CYCLE_1")
            model.sub_cycle_ratio = _ival(f[0], 1) if len(f) > 0 else 1
            model.sub_cycle_inter = bool(_ival(f[2], 0)) if len(f) > 2 else False
        else:
            toks = cards[0].tokens()
            model.sub_cycle_ratio = int(float(toks[0])) if len(toks) > 0 else 1
            if len(toks) > 2:
                model.sub_cycle_inter = bool(int(float(toks[2])))




def read_eng_run(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RUN`` or ``/ENG/RUN`` (M211): Engine execution run title, stop time, and cycle controls."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    model.run_title = title
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_RUN_1")
            model.run_tstop = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            model.run_cycle_max = _ival(f[1], 0) if len(f) > 1 else 0
        else:
            toks = cards[0].tokens()
            model.run_tstop = float(toks[0]) if len(toks) > 0 else 0.0
            if len(toks) > 1:
                model.run_cycle_max = int(float(toks[1]))




def read_eng_stop(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STOP`` or ``/ENG/STOP`` (M212): Engine simulation sensor termination and cycle limit."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_STOP_1")
            model.stop_sens_id = _ival(f[0], 0) if len(f) > 0 else 0
            model.stop_cycle_max = _ival(f[1], 0) if len(f) > 1 else 0
            model.stop_time_max = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        else:
            toks = cards[0].tokens()
            model.stop_sens_id = int(float(toks[0])) if len(toks) > 0 else 0
            if len(toks) > 1:
                model.stop_cycle_max = int(float(toks[1]))
            if len(toks) > 2:
                model.stop_time_max = float(toks[2])




def read_eng_tfile(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TFILE`` or ``/ENG/TFILE`` (M212): Time-history output frequency and sensor gating."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_TFILE_1")
            model.tfile_dt = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            model.tfile_sens_id = _ival(f[1], 0) if len(f) > 1 else 0
        else:
            toks = cards[0].tokens()
            model.tfile_dt = float(toks[0]) if len(toks) > 0 else 0.0
            if len(toks) > 1:
                model.tfile_sens_id = int(float(toks[1]))




def read_eng_rfile(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RFILE`` or ``/ENG/RFILE`` (M213): Restart file output frequency and sensor gating."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_RFILE_1")
            model.rfile_dt = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            model.rfile_ncycle = _ival(f[1], 0) if len(f) > 1 else 0
            model.rfile_sens_id = _ival(f[2], 0) if len(f) > 2 else 0
        else:
            toks = cards[0].tokens()
            model.rfile_dt = float(toks[0]) if len(toks) > 0 else 0.0
            if len(toks) > 1:
                model.rfile_ncycle = int(float(toks[1]))
            if len(toks) > 2:
                model.rfile_sens_id = int(float(toks[2]))




def read_eng_print(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PRINT`` or ``/ENG/PRINT`` (M214): Engine terminal cycle output print frequency and sensor gating."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_PRINT_1")
            model.print_ncycle = _ival(f[0], 0) if len(f) > 0 else 0
            model.print_dt = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            model.print_sens_id = _ival(f[2], 0) if len(f) > 2 else 0
        else:
            toks = cards[0].tokens()
            model.print_ncycle = int(float(toks[0])) if len(toks) > 0 else 0
            if len(toks) > 1:
                model.print_dt = float(toks[1])
            if len(toks) > 2:
                model.print_sens_id = int(float(toks[2]))




def read_eng_parith(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PARITH``, ``/ENG/PARITH``, ``/PARITH/ON``, ``/PARITH/OFF`` (M215): Parallel arithmetic reproducibility toggle."""
    upper_parts = [p.upper() for p in block.parts]
    if "OFF" in upper_parts:
        model.parith_enabled = False
        return
    elif "ON" in upper_parts:
        model.parith_enabled = True

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_PARITH_1")
            val = _ival(f[0], 1) if len(f) > 0 else 1
            model.parith_enabled = bool(val)
        else:
            toks = cards[0].tokens()
            model.parith_enabled = bool(int(float(toks[0]))) if len(toks) > 0 else True




def read_eng_vers(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/VERS`` or ``/ENG/VERS`` (M215): Target OpenRadioss version and keyword syntax level."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_VERS_1")
            model.eng_version = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        else:
            toks = cards[0].tokens()
            model.eng_version = float(toks[0]) if len(toks) > 0 else 0.0




def read_eng_monitor(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONITOR`` or ``/ENG/MONITOR`` (M216): Engine runtime console monitor directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MONITOR/{block.user_id}: missing data card", block.source)
        return

    node_id, ivar_type, dt_print = 0, 1, 0.0
    if block.fixed:
        f = cards[0].cut("ENG_MONITOR_1")
        node_id = _ival(f[0], 0) if len(f) > 0 else 0
        ivar_type = _ival(f[1], 1) if len(f) > 1 else 1
        dt_print = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        node_id = int(float(toks[0])) if len(toks) > 0 else 0
        ivar_type = int(float(toks[1])) if len(toks) > 1 else 1
        dt_print = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import EngMonitor
    mon_id = block.user_id or (len(model.monitors) + 1)
    model.monitors[mon_id] = EngMonitor(
        id=mon_id, title=title, node_id=node_id,
        ivar_type=ivar_type, dt_print=dt_print
    )




def read_eng_nois(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/NOIS`` or ``/ENG/NOIS`` (M216): Engine high frequency noise filter control."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ENG_NOIS_1")
            model.nois_freq = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            model.nois_type = _ival(f[1], 0) if len(f) > 1 else 0
        else:
            toks = cards[0].tokens()
            model.nois_freq = float(toks[0]) if len(toks) > 0 else 0.0
            if len(toks) > 1:
                model.nois_type = int(float(toks[1]))




def read_eng_fxfreq(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FXFREQ`` or ``/ENG/FXFREQ`` (M217): Fast Fourier Transform frequency spectrum output."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FXFREQ/{block.user_id}: missing data card", block.source)
        return

    f_max, n_freq, t_start, t_end = 0.0, 100, 0.0, 0.0
    if block.fixed:
        f = cards[0].cut("ENG_FXFREQ_1")
        f_max = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        n_freq = _ival(f[1], 100) if len(f) > 1 else 100
        t_start = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        t_end = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        f_max = float(toks[0]) if len(toks) > 0 else 0.0
        n_freq = int(float(toks[1])) if len(toks) > 1 else 100
        t_start = float(toks[2]) if len(toks) > 2 else 0.0
        t_end = float(toks[3]) if len(toks) > 3 else 0.0

    from ...model.entities import EngFxfreq
    fxf_id = block.user_id or (len(model.eng_fxfreqs) + 1)
    model.eng_fxfreqs[fxf_id] = EngFxfreq(
        id=fxf_id, title=title, f_max=f_max,
        n_freq=n_freq, t_start=t_start, t_end=t_end
    )




def read_eng_track(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TRACK`` or ``/ENG/TRACK`` (M218): Nodal trajectory tracking output."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TRACK/{block.user_id}: missing data card", block.source)
        return

    node_id, skew_id, dt_track = 0, 0, 0.0
    if block.fixed:
        f = cards[0].cut("ENG_TRACK_1")
        node_id = _ival(f[0], 0) if len(f) > 0 else 0
        skew_id = _ival(f[1], 0) if len(f) > 1 else 0
        dt_track = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        node_id = int(float(toks[0])) if len(toks) > 0 else 0
        skew_id = int(float(toks[1])) if len(toks) > 1 else 0
        dt_track = float(toks[2]) if len(toks) > 2 else 0.0

    from ...model.entities import EngTrack
    tr_id = block.user_id or (len(model.eng_tracks) + 1)
    model.eng_tracks[tr_id] = EngTrack(
        id=tr_id, title=title, node_id=node_id,
        skew_id=skew_id, dt_track=dt_track
    )




def read_eng_helm(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HELM`` or ``/ENG/HELM`` (M219): Helmholtz acoustic frequency response directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HELM/{block.user_id}: missing data card", block.source)
        return

    freq_start, freq_end, n_step = 0.0, 0.0, 10
    if block.fixed:
        f = cards[0].cut("ENG_HELM_1")
        freq_start = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        freq_end = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        n_step = _ival(f[2], 10) if len(f) > 2 else 10
    else:
        toks = cards[0].tokens()
        freq_start = float(toks[0]) if len(toks) > 0 else 0.0
        freq_end = float(toks[1]) if len(toks) > 1 else 0.0
        n_step = int(float(toks[2])) if len(toks) > 2 else 10

    from ...model.entities import EngHelm
    h_id = block.user_id or (len(model.eng_helms) + 1)
    model.eng_helms[h_id] = EngHelm(
        id=h_id, title=title, freq_start=freq_start,
        freq_end=freq_end, n_step=n_step
    )




def read_eng_trunc(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TRUNC`` or ``/ENG/TRUNC`` (M220): Engine cycle truncation and tolerance directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TRUNC/{block.user_id}: missing data card", block.source)
        return

    tol_trunc, dt_min, n_cycle = 0.0, 0.0, 1
    if block.fixed:
        f = cards[0].cut("ENG_TRUNC_1")
        tol_trunc = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        dt_min = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        n_cycle = _ival(f[2], 1) if len(f) > 2 else 1
    else:
        toks = cards[0].tokens()
        tol_trunc = float(toks[0]) if len(toks) > 0 else 0.0
        dt_min = float(toks[1]) if len(toks) > 1 else 0.0
        n_cycle = int(float(toks[2])) if len(toks) > 2 else 1

    from ...model.entities import EngTrunc
    t_id = block.user_id or (len(model.eng_truncs) + 1)
    model.eng_truncs[t_id] = EngTrunc(
        id=t_id, title=title, tol_trunc=tol_trunc,
        dt_min=dt_min, n_cycle=n_cycle
    )




def read_eng_mass(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MASS`` or ``/ENG/MASS`` (M221): Engine mass summary output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MASS/{block.user_id}: missing data card", block.source)
        return

    dt_mass, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_MASS_1")
        dt_mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_mass = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngMass
    m_id = block.user_id or (len(model.eng_masses) + 1)
    model.eng_masses[m_id] = EngMass(
        id=m_id, title=title, dt_mass=dt_mass, sens_id=sens_id
    )




def read_eng_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENERGY`` or ``/ENG/ENERGY`` (M222): Engine energy balance tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_energy, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_ENERGY_1")
        dt_energy = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_energy = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngEnergy
    e_id = block.user_id or (len(model.eng_energies) + 1)
    model.eng_energies[e_id] = EngEnergy(
        id=e_id, title=title, dt_energy=dt_energy, sens_id=sens_id
    )




def read_eng_moment(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MOMENT`` or ``/ENG/MOMENT`` (M223): Engine momentum tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MOMENT/{block.user_id}: missing data card", block.source)
        return

    dt_mom, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_MOMENT_1")
        dt_mom = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_mom = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngMoment
    m_id = block.user_id or (len(model.eng_moments) + 1)
    model.eng_moments[m_id] = EngMoment(
        id=m_id, title=title, dt_mom=dt_mom, sens_id=sens_id
    )




def read_eng_state(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STATE`` or ``/ENG/STATE`` (M224): Engine state variable tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STATE/{block.user_id}: missing data card", block.source)
        return

    dt_state, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_STATE_1")
        dt_state = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_state = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngState
    s_id = block.user_id or (len(model.eng_states) + 1)
    model.eng_states[s_id] = EngState(
        id=s_id, title=title, dt_state=dt_state, sens_id=sens_id
    )




def read_eng_surf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SURF`` or ``/ENG/SURF`` (M225): Engine contact surface force tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SURF/{block.user_id}: missing data card", block.source)
        return

    dt_surf, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_SURF_1")
        dt_surf = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_surf = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngSurf
    s_id = block.user_id or (len(model.eng_surfs) + 1)
    model.eng_surfs[s_id] = EngSurf(
        id=s_id, title=title, dt_surf=dt_surf, sens_id=sens_id
    )




def read_eng_ale(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE`` or ``/ENG/ALE`` (M226): Engine ALE smoothing and advection directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ALE/{block.user_id}: missing data card", block.source)
        return

    dt_ale, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_ALE_1")
        dt_ale = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ale = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngAle
    a_id = block.user_id or (len(model.eng_ales) + 1)
    model.eng_ales[a_id] = EngAle(
        id=a_id, title=title, dt_ale=dt_ale, sens_id=sens_id
    )




def read_eng_sh_thick(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SH_THICK`` or ``/ENG/SH_THICK`` (M227): Engine shell thickness update directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SH_THICK/{block.user_id}: missing data card", block.source)
        return

    dt_thick, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_SH_THICK_1")
        dt_thick = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_thick = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngShThick
    t_id = block.user_id or (len(model.eng_sh_thicks) + 1)
    model.eng_sh_thicks[t_id] = EngShThick(
        id=t_id, title=title, dt_thick=dt_thick, sens_id=sens_id
    )




def read_eng_geo(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GEO`` or ``/ENG/GEO`` (M228): Engine nodal coordinate geometry update directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GEO/{block.user_id}: missing data card", block.source)
        return

    dt_geo, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_GEO_1")
        dt_geo = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_geo = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngGeo
    g_id = block.user_id or (len(model.eng_geos) + 1)
    model.eng_geos[g_id] = EngGeo(
        id=g_id, title=title, dt_geo=dt_geo, sens_id=sens_id
    )




def read_eng_tens(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TENS`` or ``/ENG/TENS`` (M229): Engine tensor output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TENS/{block.user_id}: missing data card", block.source)
        return

    dt_tens, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_TENS_1")
        dt_tens = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tens = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngTens
    t_id = block.user_id or (len(model.eng_tenses) + 1)
    model.eng_tenses[t_id] = EngTens(
        id=t_id, title=title, dt_tens=dt_tens, sens_id=sens_id
    )




def read_eng_stress(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STRESS`` or ``/ENG/STRESS`` (M230): Engine stress output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STRESS/{block.user_id}: missing data card", block.source)
        return

    dt_stress, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_STRESS_1")
        dt_stress = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_stress = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngStress
    s_id = block.user_id or (len(model.eng_stresses) + 1)
    model.eng_stresses[s_id] = EngStress(
        id=s_id, title=title, dt_stress=dt_stress, sens_id=sens_id
    )




def read_eng_strain(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STRAIN`` or ``/ENG/STRAIN`` (M231): Engine strain output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/STRAIN/{block.user_id}: missing data card", block.source)
        return

    dt_strain, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_STRAIN_1")
        dt_strain = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_strain = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngStrain
    s_id = block.user_id or (len(model.eng_strains) + 1)
    model.eng_strains[s_id] = EngStrain(
        id=s_id, title=title, dt_strain=dt_strain, sens_id=sens_id
    )




def read_eng_plastic(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PLASTIC`` or ``/ENG/PLASTIC`` (M232): Engine plastic strain output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PLASTIC/{block.user_id}: missing data card", block.source)
        return

    dt_plastic, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_PLASTIC_1")
        dt_plastic = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_plastic = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngPlastic
    p_id = block.user_id or (len(model.eng_plastics) + 1)
    model.eng_plastics[p_id] = EngPlastic(
        id=p_id, title=title, dt_plastic=dt_plastic, sens_id=sens_id
    )




def read_eng_velocity(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/VELOCITY`` or ``/ENG/VELOCITY`` (M233): Engine velocity output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/VELOCITY/{block.user_id}: missing data card", block.source)
        return

    dt_vel, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_VELOCITY_1")
        dt_vel = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_vel = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngVelocity
    v_id = block.user_id or (len(model.eng_velocities) + 1)
    model.eng_velocities[v_id] = EngVelocity(
        id=v_id, title=title, dt_vel=dt_vel, sens_id=sens_id
    )




def read_eng_accel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ACCEL`` or ``/ENG/ACCEL`` (M234): Engine acceleration output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ACCEL/{block.user_id}: missing data card", block.source)
        return

    dt_acc, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_ACCEL_1")
        dt_acc = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_acc = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngAccel
    a_id = block.user_id or (len(model.eng_accels) + 1)
    model.eng_accels[a_id] = EngAccel(
        id=a_id, title=title, dt_acc=dt_acc, sens_id=sens_id
    )




def read_eng_disp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DISP`` or ``/ENG/DISP`` (M235): Engine displacement output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DISP/{block.user_id}: missing data card", block.source)
        return

    dt_disp, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_DISP_1")
        dt_disp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_disp = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngDisp
    d_id = block.user_id or (len(model.eng_disps) + 1)
    model.eng_disps[d_id] = EngDisp(
        id=d_id, title=title, dt_disp=dt_disp, sens_id=sens_id
    )




def read_eng_rotc(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ROTC`` or ``/ENG/ROTC`` (M236): Engine rotational displacement output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ROTC/{block.user_id}: missing data card", block.source)
        return

    dt_rotc, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_ROTC_1")
        dt_rotc = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_rotc = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngRotc
    r_id = block.user_id or (len(model.eng_rotcs) + 1)
    model.eng_rotcs[r_id] = EngRotc(
        id=r_id, title=title, dt_rotc=dt_rotc, sens_id=sens_id
    )




def read_eng_rotv(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ROTV`` or ``/ENG/ROTV`` (M237): Engine rotational/angular velocity output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ROTV/{block.user_id}: missing data card", block.source)
        return

    dt_rotv, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_ROTV_1")
        dt_rotv = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_rotv = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngRotv
    r_id = block.user_id or (len(model.eng_rotvs) + 1)
    model.eng_rotvs[r_id] = EngRotv(
        id=r_id, title=title, dt_rotv=dt_rotv, sens_id=sens_id
    )




def read_eng_rota(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ROTA`` or ``/ENG/ROTA`` (M238): Engine rotational/angular acceleration output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ROTA/{block.user_id}: missing data card", block.source)
        return

    dt_rota, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_ROTA_1")
        dt_rota = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_rota = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngRota
    r_id = block.user_id or (len(model.eng_rotas) + 1)
    model.eng_rotas[r_id] = EngRota(
        id=r_id, title=title, dt_rota=dt_rota, sens_id=sens_id
    )




def read_eng_force(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FORCE`` or ``/ENG/FORCE`` (M239): Engine nodal resultant force output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FORCE/{block.user_id}: missing data card", block.source)
        return

    dt_force, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_FORCE_1")
        dt_force = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_force = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngForce
    r_id = block.user_id or (len(model.eng_forces) + 1)
    model.eng_forces[r_id] = EngForce(
        id=r_id, title=title, dt_force=dt_force, sens_id=sens_id
    )




def read_eng_volume(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/VOLUME`` or ``/ENG/VOLUME`` (M240): Engine element volume output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/VOLUME/{block.user_id}: missing data card", block.source)
        return

    dt_vol, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_VOLUME_1")
        dt_vol = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_vol = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngVolume
    r_id = block.user_id or (len(model.eng_volumes) + 1)
    model.eng_volumes[r_id] = EngVolume(
        id=r_id, title=title, dt_vol=dt_vol, sens_id=sens_id
    )




def read_eng_density(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DENSITY`` or ``/ENG/DENSITY`` (M241): Engine material density output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DENSITY/{block.user_id}: missing data card", block.source)
        return

    dt_dens, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_DENSITY_1")
        dt_dens = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_dens = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngDensity
    r_id = block.user_id or (len(model.eng_densities) + 1)
    model.eng_densities[r_id] = EngDensity(
        id=r_id, title=title, dt_dens=dt_dens, sens_id=sens_id
    )




def read_eng_internal_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTERNAL_ENERGY`` or ``/ENG/INTERNAL_ENERGY`` (M242): Engine internal energy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INTERNAL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ie, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_INTERNAL_ENERGY_1")
        dt_ie = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ie = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngInternalEnergy
    r_id = block.user_id or (len(model.eng_internal_energies) + 1)
    model.eng_internal_energies[r_id] = EngInternalEnergy(
        id=r_id, title=title, dt_ie=dt_ie, sens_id=sens_id
    )




def read_eng_kinetic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/KINETIC_ENERGY`` or ``/ENG/KINETIC_ENERGY`` (M243): Engine kinetic energy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/KINETIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ke, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_KINETIC_ENERGY_1")
        dt_ke = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ke = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngKineticEnergy
    r_id = block.user_id or (len(model.eng_kinetic_energies) + 1)
    model.eng_kinetic_energies[r_id] = EngKineticEnergy(
        id=r_id, title=title, dt_ke=dt_ke, sens_id=sens_id
    )




def read_eng_total_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TOTAL_ENERGY`` or ``/ENG/TOTAL_ENERGY`` (M244): Engine total energy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TOTAL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_te, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_TOTAL_ENERGY_1")
        dt_te = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_te = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngTotalEnergy
    r_id = block.user_id or (len(model.eng_total_energies) + 1)
    model.eng_total_energies[r_id] = EngTotalEnergy(
        id=r_id, title=title, dt_te=dt_te, sens_id=sens_id
    )




def read_eng_hourglass_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HOURGLASS_ENERGY`` or ``/ENG/HOURGLASS_ENERGY`` (M245): Engine hourglass energy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/HOURGLASS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_he, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_HOURGLASS_ENERGY_1")
        dt_he = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_he = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngHourglassEnergy
    r_id = block.user_id or (len(model.eng_hourglass_energies) + 1)
    model.eng_hourglass_energies[r_id] = EngHourglassEnergy(
        id=r_id, title=title, dt_he=dt_he, sens_id=sens_id
    )




def read_eng_contact_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONTACT_ENERGY`` or ``/ENG/CONTACT_ENERGY`` (M246): Engine contact energy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONTACT_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ce, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_CONTACT_ENERGY_1")
        dt_ce = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ce = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngContactEnergy
    r_id = block.user_id or (len(model.eng_contact_energies) + 1)
    model.eng_contact_energies[r_id] = EngContactEnergy(
        id=r_id, title=title, dt_ce=dt_ce, sens_id=sens_id
    )




def read_eng_numerical_dissipation(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/NUMERICAL_DISSIPATION`` or ``/ENG/NUMERICAL_DISSIPATION`` (M247): Engine numerical dissipation energy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/NUMERICAL_DISSIPATION/{block.user_id}: missing data card", block.source)
        return

    dt_num, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_NUMERICAL_DISSIPATION_1")
        dt_num = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_num = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngNumericalDissipation
    r_id = block.user_id or (len(model.eng_numerical_dissipations) + 1)
    model.eng_numerical_dissipations[r_id] = EngNumericalDissipation(
        id=r_id, title=title, dt_num=dt_num, sens_id=sens_id
    )




def read_eng_ext_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EXT_WORK`` or ``/ENG/EXT_WORK`` (M248): Engine external work output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/EXT_WORK/{block.user_id}: missing data card", block.source)
        return

    dt_wext, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_EXT_WORK_1")
        dt_wext = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_wext = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngExtWork
    r_id = block.user_id or (len(model.eng_ext_works) + 1)
    model.eng_ext_works[r_id] = EngExtWork(
        id=r_id, title=title, dt_wext=dt_wext, sens_id=sens_id
    )




def read_eng_tot_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TOT_ENERGY`` or ``/ENG/TOT_ENERGY`` (M249): Engine total system energy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/TOT_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etot, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_TOT_ENERGY_1")
        dt_etot = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etot = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngTotEnergy
    r_id = block.user_id or (len(model.eng_tot_energies) + 1)
    model.eng_tot_energies[r_id] = EngTotEnergy(
        id=r_id, title=title, dt_etot=dt_etot, sens_id=sens_id
    )




def read_eng_mass_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MASS_ENERGY`` or ``/ENG/MASS_ENERGY`` (M250): Engine added mass kinetic energy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MASS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_emass, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_MASS_ENERGY_1")
        dt_emass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_emass = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngMassEnergy
    r_id = block.user_id or (len(model.eng_mass_energies) + 1)
    model.eng_mass_energies[r_id] = EngMassEnergy(
        id=r_id, title=title, dt_emass=dt_emass, sens_id=sens_id
    )




def read_eng_mass_change(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MASS_CHANGE`` or ``/ENG/MASS_CHANGE`` (M251): Engine added/eroded mass delta variation output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MASS_CHANGE/{block.user_id}: missing data card", block.source)
        return

    dt_dmass, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_MASS_CHANGE_1")
        dt_dmass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_dmass = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngMassChange
    r_id = block.user_id or (len(model.eng_mass_changes) + 1)
    model.eng_mass_changes[r_id] = EngMassChange(
        id=r_id, title=title, dt_dmass=dt_dmass, sens_id=sens_id
    )




def read_eng_pressure(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PRESSURE`` or ``/ENG/PRESSURE`` (M252): Engine hydrostatic pressure output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PRESSURE/{block.user_id}: missing data card", block.source)
        return

    dt_press, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_PRESSURE_1")
        dt_press = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_press = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngPressure
    r_id = block.user_id or (len(model.eng_pressures) + 1)
    model.eng_pressures[r_id] = EngPressure(
        id=r_id, title=title, dt_press=dt_press, sens_id=sens_id
    )




def read_eng_temperature(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/TEMPERATURE`` or ``/ENG/TEMP`` (M253): Engine material temperature output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/TEMPERATURE/{block.user_id}: missing data card", block.source)
        return

    dt_temp, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_TEMPERATURE_1")
        dt_temp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_temp = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngTemperature
    r_id = block.user_id or (len(model.eng_temperatures) + 1)
    model.eng_temperatures[r_id] = EngTemperature(
        id=r_id, title=title, dt_temp=dt_temp, sens_id=sens_id
    )




def read_eng_volume(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/VOLUME`` or ``/ENG/VOL`` (M254): Engine element volume output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/VOLUME/{block.user_id}: missing data card", block.source)
        return

    dt_vol, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_VOLUME_1")
        dt_vol = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_vol = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngVolume
    r_id = block.user_id or (len(model.eng_volumes) + 1)
    model.eng_volumes[r_id] = EngVolume(
        id=r_id, title=title, dt_vol=dt_vol, sens_id=sens_id
    )




def read_eng_density(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/DENSITY`` or ``/ENG/RHO`` (M255): Engine material density output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/DENSITY/{block.user_id}: missing data card", block.source)
        return

    dt_dens, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_DENSITY_1")
        dt_dens = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_dens = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngDensity
    r_id = block.user_id or (len(model.eng_densities) + 1)
    model.eng_densities[r_id] = EngDensity(
        id=r_id, title=title, dt_dens=dt_dens, sens_id=sens_id
    )




def read_eng_entropy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ENTROPY`` or ``/ENG/THERMAL_ENTROPY`` (M256): Engine thermal entropy output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ENTROPY/{block.user_id}: missing data card", block.source)
        return

    dt_entr, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_ENTROPY_1")
        dt_entr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_entr = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngEntropy
    r_id = block.user_id or (len(model.eng_entropies) + 1)
    model.eng_entropies[r_id] = EngEntropy(
        id=r_id, title=title, dt_entr=dt_entr, sens_id=sens_id
    )




def read_eng_sound_speed(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/SOUND_SPEED`` or ``/ENG/C_SOUND`` (M257): Engine material sound speed output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/SOUND_SPEED/{block.user_id}: missing data card", block.source)
        return

    dt_sound, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_SOUND_SPEED_1")
        dt_sound = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_sound = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngSoundSpeed
    r_id = block.user_id or (len(model.eng_sound_speeds) + 1)
    model.eng_sound_speeds[r_id] = EngSoundSpeed(
        id=r_id, title=title, dt_sound=dt_sound, sens_id=sens_id
    )




def read_eng_yield_stress(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/YIELD_STRESS`` or ``/ENG/YIELD`` (M259): Engine material yield stress output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/YIELD_STRESS/{block.user_id}: missing data card", block.source)
        return

    dt_yield, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_YIELD_STRESS_1")
        dt_yield = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_yield = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngYieldStress
    r_id = block.user_id or (len(model.eng_yield_stresses) + 1)
    model.eng_yield_stresses[r_id] = EngYieldStress(
        id=r_id, title=title, dt_yield=dt_yield, sens_id=sens_id
    )




def read_eng_plastic_work(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PLASTIC_WORK`` or ``/ENG/WPLAS`` (M260): Engine plastic work dissipation output tracking directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PLASTIC_WORK/{block.user_id}: missing data card", block.source)
        return

    dt_wplas, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_PLASTIC_WORK_1")
        dt_wplas = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_wplas = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngPlasticWork
    r_id = block.user_id or (len(model.eng_plastic_works) + 1)
    model.eng_plastic_works[r_id] = EngPlasticWork(
        id=r_id, title=title, dt_wplas=dt_wplas, sens_id=sens_id
    )




def read_eng_temperature(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/TEMPERATURE`` or ``/ENG/TEMP`` (M261): Engine temperature field history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/TEMPERATURE/{block.user_id}: missing data card", block.source)
        return

    dt_temp, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_TEMPERATURE_1")
        dt_temp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_temp = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngTemperature
    r_id = block.user_id or (len(model.eng_temperatures) + 1)
    model.eng_temperatures[r_id] = EngTemperature(
        id=r_id, title=title, dt_temp=dt_temp, sens_id=sens_id
    )




def read_eng_stress_tri(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/STRESS_TRI`` or ``/ENG/TRIAXIALITY`` (M262): Engine stress triaxiality history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/STRESS_TRI/{block.user_id}: missing data card", block.source)
        return

    dt_triax, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_STRESS_TRI_1")
        dt_triax = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_triax = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngStressTri
    r_id = block.user_id or (len(model.eng_stress_tris) + 1)
    model.eng_stress_tris[r_id] = EngStressTri(
        id=r_id, title=title, dt_triax=dt_triax, sens_id=sens_id
    )




def read_eng_lode_angle(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/LODE_ANGLE`` or ``/ENG/LODE`` (M263): Engine normalized Lode angle parameter history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/LODE_ANGLE/{block.user_id}: missing data card", block.source)
        return

    dt_lode, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_LODE_ANGLE_1")
        dt_lode = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_lode = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngLodeAngle
    r_id = block.user_id or (len(model.eng_lode_angles) + 1)
    model.eng_lode_angles[r_id] = EngLodeAngle(
        id=r_id, title=title, dt_lode=dt_lode, sens_id=sens_id
    )




def read_eng_max_shear(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAX_SHEAR`` or ``/ENG/TMAX`` (M264): Engine maximum shear stress history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAX_SHEAR/{block.user_id}: missing data card", block.source)
        return

    dt_tmax, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_MAX_SHEAR_1")
        dt_tmax = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tmax = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngMaxShear
    r_id = block.user_id or (len(model.eng_max_shears) + 1)
    model.eng_max_shears[r_id] = EngMaxShear(
        id=r_id, title=title, dt_tmax=dt_tmax, sens_id=sens_id
    )




def read_eng_effective_stress(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/EFFECTIVE_STRESS`` or ``/ENG/SIG_EFF`` (M265): Engine von Mises equivalent / effective stress history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/EFFECTIVE_STRESS/{block.user_id}: missing data card", block.source)
        return

    dt_sigeff, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_EFFECTIVE_STRESS_1")
        dt_sigeff = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_sigeff = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngEffectiveStress
    r_id = block.user_id or (len(model.eng_effective_stresses) + 1)
    model.eng_effective_stresses[r_id] = EngEffectiveStress(
        id=r_id, title=title, dt_sigeff=dt_sigeff, sens_id=sens_id
    )




def read_eng_hydrostatic_pressure(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/HYDROSTATIC_PRESSURE`` or ``/ENG/HYDRO_PRES`` (M266): Engine hydrostatic pressure history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/HYDROSTATIC_PRESSURE/{block.user_id}: missing data card", block.source)
        return

    dt_phyd, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_HYDROSTATIC_PRESSURE_1")
        dt_phyd = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_phyd = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngHydrostaticPressure
    r_id = block.user_id or (len(model.eng_hydrostatic_pressures) + 1)
    model.eng_hydrostatic_pressures[r_id] = EngHydrostaticPressure(
        id=r_id, title=title, dt_phyd=dt_phyd, sens_id=sens_id
    )




def read_eng_octahedral_shear(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/OCTAHEDRAL_SHEAR`` or ``/ENG/OCT_SHEAR`` (M267): Engine octahedral shear stress history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/OCTAHEDRAL_SHEAR/{block.user_id}: missing data card", block.source)
        return

    dt_toct, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_OCTAHEDRAL_SHEAR_1")
        dt_toct = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_toct = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngOctahedralShear
    r_id = block.user_id or (len(model.eng_octahedral_shears) + 1)
    model.eng_octahedral_shears[r_id] = EngOctahedralShear(
        id=r_id, title=title, dt_toct=dt_toct, sens_id=sens_id
    )




def read_eng_deviatoric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/DEVIATORIC_ENERGY`` or ``/ENG/DEV_ENERGY`` (M268): Engine deviatoric strain energy history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/DEVIATORIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_wdev, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_DEVIATORIC_ENERGY_1")
        dt_wdev = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_wdev = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngDeviatoricEnergy
    r_id = block.user_id or (len(model.eng_deviatoric_energies) + 1)
    model.eng_deviatoric_energies[r_id] = EngDeviatoricEnergy(
        id=r_id, title=title, dt_wdev=dt_wdev, sens_id=sens_id
    )




def read_eng_strain_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/STRAIN_RATE`` or ``/ENG/EPSDOT`` (M269): Engine strain rate history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/STRAIN_RATE/{block.user_id}: missing data card", block.source)
        return

    dt_epsdot, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_STRAIN_RATE_1")
        dt_epsdot = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_epsdot = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngStrainRate
    r_id = block.user_id or (len(model.eng_strain_rates) + 1)
    model.eng_strain_rates[r_id] = EngStrainRate(
        id=r_id, title=title, dt_epsdot=dt_epsdot, sens_id=sens_id
    )




def read_eng_bulk_viscosity(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/BULK_VISCOSITY`` or ``/ENG/Q_VISC`` (M270): Engine bulk viscosity energy history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/BULK_VISCOSITY/{block.user_id}: missing data card", block.source)
        return

    dt_qvisc, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_BULK_VISCOSITY_1")
        dt_qvisc = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_qvisc = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngBulkViscosity
    r_id = block.user_id or (len(model.eng_bulk_viscosities) + 1)
    model.eng_bulk_viscosities[r_id] = EngBulkViscosity(
        id=r_id, title=title, dt_qvisc=dt_qvisc, sens_id=sens_id
    )




def read_eng_hourglass_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/HOURGLASS_ENERGY`` or ``/ENG/HG_ENERGY`` (M271): Engine hourglass energy history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/HOURGLASS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_hg, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_HOURGLASS_ENERGY_1")
        dt_hg = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_hg = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngHourglassEnergy
    r_id = block.user_id or (len(model.eng_hourglass_energies) + 1)
    model.eng_hourglass_energies[r_id] = EngHourglassEnergy(
        id=r_id, title=title, dt_hg=dt_hg, sens_id=sens_id
    )




def read_eng_contact_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/CONTACT_ENERGY`` or ``/ENG/CNT_ENERGY`` (M272): Engine contact energy history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/CONTACT_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_contact, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_CONTACT_ENERGY_1")
        dt_contact = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_contact = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngContactEnergy
    r_id = block.user_id or (len(model.eng_contact_energies) + 1)
    model.eng_contact_energies[r_id] = EngContactEnergy(
        id=r_id, title=title, dt_contact=dt_contact, sens_id=sens_id
    )




def read_eng_spring_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/SPRING_ENERGY`` or ``/ENG/SPR_ENERGY`` (M273): Engine spring energy history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/SPRING_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_spring, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_SPRING_ENERGY_1")
        dt_spring = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_spring = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngSpringEnergy
    r_id = block.user_id or (len(model.eng_spring_energies) + 1)
    model.eng_spring_energies[r_id] = EngSpringEnergy(
        id=r_id, title=title, dt_spring=dt_spring, sens_id=sens_id
    )




def read_eng_joint_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/JOINT_ENERGY`` or ``/ENG/JNT_ENERGY`` (M274): Engine joint energy history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/JOINT_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_joint, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_JOINT_ENERGY_1")
        dt_joint = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_joint = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngJointEnergy
    r_id = block.user_id or (len(model.eng_joint_energies) + 1)
    model.eng_joint_energies[r_id] = EngJointEnergy(
        id=r_id, title=title, dt_joint=dt_joint, sens_id=sens_id
    )




def read_eng_rwall_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/RWALL_ENERGY`` or ``/ENG/RWALL_WORK`` (M275): Engine rigid wall energy history tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/RWALL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_rwall, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_RWALL_ENERGY_1")
        dt_rwall = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_rwall = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngRwallEnergy
    r_id = block.user_id or (len(model.eng_rwall_energies) + 1)
    model.eng_rwall_energies[r_id] = EngRwallEnergy(
        id=r_id, title=title, dt_rwall=dt_rwall, sens_id=sens_id
    )




def read_eng_surf_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/SURF_ENERGY`` or ``/ENG/SURF_WORK`` (M276): Engine surface boundary pressure / traction work tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/SURF_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_surf, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_SURF_ENERGY_1")
        dt_surf = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_surf = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngSurfEnergy
    r_id = block.user_id or (len(model.eng_surf_energies) + 1)
    model.eng_surf_energies[r_id] = EngSurfEnergy(
        id=r_id, title=title, dt_surf=dt_surf, sens_id=sens_id
    )




def read_eng_heat_exchange(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/HEAT_EXCHANGE`` or ``/ENG/HEAT_ENERGY`` (M277): Engine heat exchange / thermal dissipation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/HEAT_EXCHANGE/{block.user_id}: missing data card", block.source)
        return

    dt_heat, sens_id = 0.0, 0
    if block.fixed:
        f = cards[0].cut("ENG_HEAT_EXCHANGE_1")
        dt_heat = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_heat = float(toks[0]) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0

    from ...model.entities import EngHeatExchange
    r_id = block.user_id or (len(model.eng_heat_exchanges) + 1)
    model.eng_heat_exchanges[r_id] = EngHeatExchange(
        id=r_id, title=title, dt_heat=dt_heat, sens_id=sens_id
    )




def read_eng_sph_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/SPH_ENERGY`` or ``/ENG/SPH_WORK`` (M278): Engine SPH particle internal/work energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/SPH_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_sph, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_SPH_ENERGY_1")
        dt_sph = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_sph = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngSphEnergy
    r_id = block.user_id or (len(model.eng_sph_energies) + 1)
    model.eng_sph_energies[r_id] = EngSphEnergy(
        id=r_id, title=title, dt_sph=dt_sph, sens_id=sens_id
    )




def read_eng_ale_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ALE_ENERGY`` or ``/ENG/ALE_WORK`` (M279): Engine ALE grid work/advection energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ALE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ale, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ALE_ENERGY_1")
        dt_ale = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ale = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngAleEnergy
    r_id = block.user_id or (len(model.eng_ale_energies) + 1)
    model.eng_ale_energies[r_id] = EngAleEnergy(
        id=r_id, title=title, dt_ale=dt_ale, sens_id=sens_id
    )




def read_eng_fsi_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FSI_ENERGY`` or ``/ENG/FSI_WORK`` (M280): Engine FSI interface work and energy transfer tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FSI_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fsi, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FSI_ENERGY_1")
        dt_fsi = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fsi = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFsiEnergy
    r_id = block.user_id or (len(model.eng_fsi_energies) + 1)
    model.eng_fsi_energies[r_id] = EngFsiEnergy(
        id=r_id, title=title, dt_fsi=dt_fsi, sens_id=sens_id
    )




def read_eng_xfem_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/XFEM_ENERGY`` or ``/ENG/XFEM_WORK`` (M281): Engine XFEM crack propagation and cohesive zone work tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/XFEM_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_xfem, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_XFEM_ENERGY_1")
        dt_xfem = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_xfem = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngXfemEnergy
    r_id = block.user_id or (len(model.eng_xfem_energies) + 1)
    model.eng_xfem_energies[r_id] = EngXfemEnergy(
        id=r_id, title=title, dt_xfem=dt_xfem, sens_id=sens_id
    )




# ============================================================================
# M282 Suite: Gene1 failure, EngHelmholtzEnergy, CardanJoint, SensorSpringNormalWork
# ============================================================================

def read_eng_helmholtz_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/HELMHOLTZ_ENERGY`` or ``/ENG/HELMHOLTZ_WORK`` (M282): Engine Helmholtz free energy and thermodynamic potential tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/HELMHOLTZ_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_helm, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_HELMHOLTZ_ENERGY_1")
        dt_helm = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_helm = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngHelmholtzEnergy
    r_id = block.user_id or (len(model.eng_helmholtz_energies) + 1)
    model.eng_helmholtz_energies[r_id] = EngHelmholtzEnergy(
        id=r_id, title=title, dt_helm=dt_helm, sens_id=sens_id
    )




def read_eng_entropy_production(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ENTROPY_PRODUCTION`` or ``/ENG/ENTROPY_PROD`` (M283): Engine irreversible entropy generation and dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ENTROPY_PRODUCTION/{block.user_id}: missing data card", block.source)
        return

    dt_entropy, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ENTROPY_PRODUCTION_1")
        dt_entropy = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_entropy = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngEntropyProduction
    r_id = block.user_id or (len(model.eng_entropy_productions) + 1)
    model.eng_entropy_productions[r_id] = EngEntropyProduction(
        id=r_id, title=title, dt_entropy=dt_entropy, sens_id=sens_id
    )




def read_eng_internal_pressure(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/INTERNAL_PRESSURE`` or ``/ENG/INT_PRESSURE`` (M284): Engine cavity internal pressure tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/INTERNAL_PRESSURE/{block.user_id}: missing data card", block.source)
        return

    dt_pres, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_INTERNAL_PRESSURE_1")
        dt_pres = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_pres = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngInternalPressure
    r_id = block.user_id or (len(model.eng_internal_pressures) + 1)
    model.eng_internal_pressures[r_id] = EngInternalPressure(
        id=r_id, title=title, dt_pres=dt_pres, sens_id=sens_id
    )




def read_eng_coriolis_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/CORIOLIS_ENERGY`` or ``/ENG/CORIOLIS_WORK`` (M285): Engine rotating frame Coriolis inertial force work tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/CORIOLIS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_coriolis, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_CORIOLIS_ENERGY_1")
        dt_coriolis = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_coriolis = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngCoriolisEnergy
    r_id = block.user_id or (len(model.eng_coriolis_energies) + 1)
    model.eng_coriolis_energies[r_id] = EngCoriolisEnergy(
        id=r_id, title=title, dt_coriolis=dt_coriolis, sens_id=sens_id
    )




def read_eng_magnetic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAGNETIC_ENERGY`` or ``/ENG/MAGNETIC_WORK`` (M286): Engine electromagnetic magnetic field energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAGNETIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_mag, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_MAGNETIC_ENERGY_1")
        dt_mag = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_mag = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngMagneticEnergy
    r_id = block.user_id or (len(model.eng_magnetic_energies) + 1)
    model.eng_magnetic_energies[r_id] = EngMagneticEnergy(
        id=r_id, title=title, dt_mag=dt_mag, sens_id=sens_id
    )




def read_eng_poynting_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/POYNTING_ENERGY`` or ``/ENG/POYNTING_WORK`` (M287): Engine electromagnetic Poynting flux vector and radiated energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/POYNTING_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_poynting, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_POYNTING_ENERGY_1")
        dt_poynting = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_poynting = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngPoyntingEnergy
    r_id = block.user_id or (len(model.eng_poynting_energies) + 1)
    model.eng_poynting_energies[r_id] = EngPoyntingEnergy(
        id=r_id, title=title, dt_poynting=dt_poynting, sens_id=sens_id
    )




def read_eng_maxwell_stress_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAXWELL_STRESS_ENERGY`` or ``/ENG/MAXWELL_WORK`` (M288): Engine Maxwell stress tensor mechanical work and electromagnetic field deformation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAXWELL_STRESS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_maxwell, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_MAXWELL_STRESS_ENERGY_1")
        dt_maxwell = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_maxwell = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngMaxwellStressEnergy
    r_id = block.user_id or (len(model.eng_maxwell_stress_energies) + 1)
    model.eng_maxwell_stress_energies[r_id] = EngMaxwellStressEnergy(
        id=r_id, title=title, dt_maxwell=dt_maxwell, sens_id=sens_id
    )




def read_eng_joule_heat_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/JOULE_HEAT_ENERGY`` or ``/ENG/JOULE_HEAT_WORK`` (M289): Engine electromagnetic resistive Joule heating dissipation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/JOULE_HEAT_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_joule, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_JOULE_HEAT_ENERGY_1")
        dt_joule = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_joule = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngJouleHeatEnergy
    r_id = block.user_id or (len(model.eng_joule_heat_energies) + 1)
    model.eng_joule_heat_energies[r_id] = EngJouleHeatEnergy(
        id=r_id, title=title, dt_joule=dt_joule, sens_id=sens_id
    )




def read_eng_lorentz_force_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/LORENTZ_FORCE_ENERGY`` or ``/ENG/LORENTZ_WORK`` (M290): Engine electromagnetic Lorentz force mechanical work and volume force energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/LORENTZ_FORCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_lorentz, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_LORENTZ_FORCE_ENERGY_1")
        dt_lorentz = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_lorentz = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngLorentzForceEnergy
    r_id = block.user_id or (len(model.eng_lorentz_force_energies) + 1)
    model.eng_lorentz_force_energies[r_id] = EngLorentzForceEnergy(
        id=r_id, title=title, dt_lorentz=dt_lorentz, sens_id=sens_id
    )




def read_eng_plasmonic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PLASMONIC_ENERGY`` or ``/ENG/PLASMONIC_WORK`` (M291): Engine surface plasmon polariton and resonant optical coupling dissipation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PLASMONIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_plasmon, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_PLASMONIC_ENERGY_1")
        dt_plasmon = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_plasmon = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngPlasmonicEnergy
    r_id = block.user_id or (len(model.eng_plasmonic_energies) + 1)
    model.eng_plasmonic_energies[r_id] = EngPlasmonicEnergy(
        id=r_id, title=title, dt_plasmon=dt_plasmon, sens_id=sens_id
    )




def read_eng_dielectric_loss_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/DIELECTRIC_LOSS_ENERGY`` or ``/ENG/DIELECTRIC_WORK`` (M292): Engine high-frequency dielectric permittivity loss and polarization dissipation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/DIELECTRIC_LOSS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_dielectric, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_DIELECTRIC_LOSS_ENERGY_1")
        dt_dielectric = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_dielectric = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngDielectricLossEnergy
    r_id = block.user_id or (len(model.eng_dielectric_loss_energies) + 1)
    model.eng_dielectric_loss_energies[r_id] = EngDielectricLossEnergy(
        id=r_id, title=title, dt_dielectric=dt_dielectric, sens_id=sens_id
    )




def read_eng_magnetic_hysteresis_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAGNETIC_HYSTERESIS_ENERGY`` or ``/ENG/MAG_HYST_WORK`` (M293): Engine ferromagnetic / magnetic hysteresis dissipation and core loss energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAGNETIC_HYSTERESIS_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_hysteresis, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_MAGNETIC_HYSTERESIS_ENERGY_1")
        dt_hysteresis = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_hysteresis = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngMagneticHysteresisEnergy
    r_id = block.user_id or (len(model.eng_magnetic_hysteresis_energies) + 1)
    model.eng_magnetic_hysteresis_energies[r_id] = EngMagneticHysteresisEnergy(
        id=r_id, title=title, dt_hysteresis=dt_hysteresis, sens_id=sens_id
    )




def read_eng_magnetostriction_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAGNETOSTRICTION_ENERGY`` or ``/ENG/MAG_STRICT_WORK`` (M294): Engine magnetostrictive strain deformation energy and magnetic-mechanical coupling work output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAGNETOSTRICTION_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_magnetostriction, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_MAGNETOSTRICTION_ENERGY_1")
        dt_magnetostriction = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_magnetostriction = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngMagnetostrictionEnergy
    r_id = block.user_id or (len(model.eng_magnetostriction_energies) + 1)
    model.eng_magnetostriction_energies[r_id] = EngMagnetostrictionEnergy(
        id=r_id, title=title, dt_magnetostriction=dt_magnetostriction, sens_id=sens_id
    )




def read_eng_electrocaloric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROCALORIC_ENERGY`` or ``/ENG/EC_WORK`` (M295): Engine electrocaloric reversible adiabatic thermal entropy change and polarization coupling energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROCALORIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_electrocaloric, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROCALORIC_ENERGY_1")
        dt_electrocaloric = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_electrocaloric = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrocaloricEnergy
    r_id = block.user_id or (len(model.eng_electrocaloric_energies) + 1)
    model.eng_electrocaloric_energies[r_id] = EngElectrocaloricEnergy(
        id=r_id, title=title, dt_electrocaloric=dt_electrocaloric, sens_id=sens_id
    )




def read_eng_magnetocaloric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAGNETOCALORIC_ENERGY`` or ``/ENG/MC_WORK`` (M296): Engine magnetocaloric reversible adiabatic temperature/magnetic entropy coupling energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAGNETOCALORIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_magnetocaloric, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_MAGNETOCALORIC_ENERGY_1")
        dt_magnetocaloric = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_magnetocaloric = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngMagnetocaloricEnergy
    r_id = block.user_id or (len(model.eng_magnetocaloric_energies) + 1)
    model.eng_magnetocaloric_energies[r_id] = EngMagnetocaloricEnergy(
        id=r_id, title=title, dt_magnetocaloric=dt_magnetocaloric, sens_id=sens_id
    )




def read_eng_thermoelectric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOELECTRIC_ENERGY`` or ``/ENG/TE_WORK`` (M297): Engine thermoelectric Seebeck/Peltier reversible thermal-electric energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOELECTRIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_thermoelectric, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOELECTRIC_ENERGY_1")
        dt_thermoelectric = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_thermoelectric = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermoelectricEnergy
    r_id = block.user_id or (len(model.eng_thermoelectric_energies) + 1)
    model.eng_thermoelectric_energies[r_id] = EngThermoelectricEnergy(
        id=r_id, title=title, dt_thermoelectric=dt_thermoelectric, sens_id=sens_id
    )




def read_eng_pyroelectric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PYROELECTRIC_ENERGY`` or ``/ENG/PYRO_WORK`` (M298): Engine pyroelectric thermal-polarization coupling and reversible temperature-induced electric energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PYROELECTRIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_pyroelectric, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_PYROELECTRIC_ENERGY_1")
        dt_pyroelectric = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_pyroelectric = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngPyroelectricEnergy
    r_id = block.user_id or (len(model.eng_pyroelectric_energies) + 1)
    model.eng_pyroelectric_energies[r_id] = EngPyroelectricEnergy(
        id=r_id, title=title, dt_pyroelectric=dt_pyroelectric, sens_id=sens_id
    )




def read_eng_thermomagnetic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOMAGNETIC_ENERGY`` or ``/ENG/TM_WORK`` (M299): Engine thermomagnetic Nernst/Ettingshausen reversible thermal-magnetic energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOMAGNETIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_thermomagnetic, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOMAGNETIC_ENERGY_1")
        dt_thermomagnetic = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_thermomagnetic = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermomagneticEnergy
    r_id = block.user_id or (len(model.eng_thermomagnetic_energies) + 1)
    model.eng_thermomagnetic_energies[r_id] = EngThermomagneticEnergy(
        id=r_id, title=title, dt_thermomagnetic=dt_thermomagnetic, sens_id=sens_id
    )




def read_eng_thermogalvanic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOGALVANIC_ENERGY`` or ``/ENG/TG_WORK`` (M300): Engine thermogalvanic electrochemical non-isothermal cell and temperature-induced redox reaction energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOGALVANIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_thermogalvanic, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOGALVANIC_ENERGY_1")
        dt_thermogalvanic = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_thermogalvanic = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermogalvanicEnergy
    r_id = block.user_id or (len(model.eng_thermogalvanic_energies) + 1)
    model.eng_thermogalvanic_energies[r_id] = EngThermogalvanicEnergy(
        id=r_id, title=title, dt_thermogalvanic=dt_thermogalvanic, sens_id=sens_id
    )




def read_eng_thermionic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMIONIC_ENERGY`` or ``/ENG/TI_WORK`` (M301): Engine thermionic emission electron thermal-field work and thermal-to-electric conversion energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMIONIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_thermionic, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMIONIC_ENERGY_1")
        dt_thermionic = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_thermionic = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermionicEnergy
    r_id = block.user_id or (len(model.eng_thermionic_energies) + 1)
    model.eng_thermionic_energies[r_id] = EngThermionicEnergy(
        id=r_id, title=title, dt_thermionic=dt_thermionic, sens_id=sens_id
    )




def read_eng_thermophotonic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOPHOTONIC_ENERGY`` or ``/ENG/TP_WORK`` (M302): Engine thermophotonic luminescence radiation and radiative thermal-to-electric photon conversion energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOPHOTONIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_thermophotonic, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOPHOTONIC_ENERGY_1")
        dt_thermophotonic = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_thermophotonic = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermophotonicEnergy
    r_id = block.user_id or (len(model.eng_thermophotonic_energies) + 1)
    model.eng_thermophotonic_energies[r_id] = EngThermophotonicEnergy(
        id=r_id, title=title, dt_thermophotonic=dt_thermophotonic, sens_id=sens_id
    )




def read_eng_electrostrictive_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROSTRICTIVE_ENERGY`` or ``/ENG/ES_WORK`` (M303): Engine electrostrictive non-linear quadratic electric polarization strain work and electrostrictive deformation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROSTRICTIVE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_electrostrictive, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROSTRICTIVE_ENERGY_1")
        dt_electrostrictive = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_electrostrictive = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrostrictiveEnergy
    r_id = block.user_id or (len(model.eng_electrostrictive_energies) + 1)
    model.eng_electrostrictive_energies[r_id] = EngElectrostrictiveEnergy(
        id=r_id, title=title, dt_electrostrictive=dt_electrostrictive, sens_id=sens_id
    )




def read_eng_photomagnetic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PHOTOMAGNETIC_ENERGY`` or ``/ENG/PM_WORK`` (M304): Engine photomagnetic magneto-optical resonant absorption and photon-spin polarization energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PHOTOMAGNETIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_photomagnetic, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_PHOTOMAGNETIC_ENERGY_1")
        dt_photomagnetic = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_photomagnetic = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngPhotomagneticEnergy
    r_id = block.user_id or (len(model.eng_photomagnetic_energies) + 1)
    model.eng_photomagnetic_energies[r_id] = EngPhotomagneticEnergy(
        id=r_id, title=title, dt_photomagnetic=dt_photomagnetic, sens_id=sens_id
    )




def read_eng_thermophotonic_emission_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOPHOTONIC_EMISSION_ENERGY`` or ``/ENG/TPE_WORK`` (M305): Engine thermophotonic non-equilibrium electroluminescent photon emission and optical extraction energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOPHOTONIC_EMISSION_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_tpe, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOPHOTONIC_EMISSION_ENERGY_1")
        dt_tpe = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tpe = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermophotonicEmissionEnergy
    r_id = block.user_id or (len(model.eng_thermophotonic_emission_energies) + 1)
    model.eng_thermophotonic_emission_energies[r_id] = EngThermophotonicEmissionEnergy(
        id=r_id, title=title, dt_tpe=dt_tpe, sens_id=sens_id
    )




def read_eng_phonon_polariton_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PHONON_POLARITON_ENERGY`` or ``/ENG/PP_WORK`` (M306): Engine surface phonon-polariton coupled infrared vibrational-electromagnetic resonance energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PHONON_POLARITON_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_pp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_PHONON_POLARITON_ENERGY_1")
        dt_pp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_pp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngPhononPolaritonEnergy
    r_id = block.user_id or (len(model.eng_phonon_polariton_energies) + 1)
    model.eng_phonon_polariton_energies[r_id] = EngPhononPolaritonEnergy(
        id=r_id, title=title, dt_pp=dt_pp, sens_id=sens_id
    )




def read_eng_exciton_polariton_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/EXCITON_POLARITON_ENERGY`` or ``/ENG/EP_WORK`` (M307): Engine quantum exciton-polariton cavity coupled light-matter hybridization and condensation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/EXCITON_POLARITON_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ep, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_EXCITON_POLARITON_ENERGY_1")
        dt_ep = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ep = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngExcitonPolaritonEnergy
    r_id = block.user_id or (len(model.eng_exciton_polariton_energies) + 1)
    model.eng_exciton_polariton_energies[r_id] = EngExcitonPolaritonEnergy(
        id=r_id, title=title, dt_ep=dt_ep, sens_id=sens_id
    )




def read_eng_magnon_polariton_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAGNON_POLARITON_ENERGY`` or ``/ENG/MP_WORK`` (M308): Engine quantum magnon-polariton coupled magnetic spin-wave electromagnetic cavity hybridization energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAGNON_POLARITON_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_mp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_MAGNON_POLARITON_ENERGY_1")
        dt_mp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_mp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngMagnonPolaritonEnergy
    r_id = block.user_id or (len(model.eng_magnon_polariton_energies) + 1)
    model.eng_magnon_polariton_energies[r_id] = EngMagnonPolaritonEnergy(
        id=r_id, title=title, dt_mp=dt_mp, sens_id=sens_id
    )




def read_eng_piezomagnetic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PIEZOMAGNETIC_ENERGY`` or ``/ENG/PZM_WORK`` (M309): Engine linear piezomagnetic magneto-mechanical coupled strain work and piezomagnetic polarization energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PIEZOMAGNETIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_pzm, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_PIEZOMAGNETIC_ENERGY_1")
        dt_pzm = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_pzm = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngPiezomagneticEnergy
    r_id = block.user_id or (len(model.eng_piezomagnetic_energies) + 1)
    model.eng_piezomagnetic_energies[r_id] = EngPiezomagneticEnergy(
        id=r_id, title=title, dt_pzm=dt_pzm, sens_id=sens_id
    )




def read_eng_barocaloric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/BAROCALORIC_ENERGY`` or ``/ENG/BCE_WORK`` (M310): Engine barocaloric pressure-induced thermal entropy change and reversible solid-state elastocaloric/barocaloric phase transformation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/BAROCALORIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_bce, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_BAROCALORIC_ENERGY_1")
        dt_bce = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_bce = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngBarocaloricEnergy
    r_id = block.user_id or (len(model.eng_barocaloric_energies) + 1)
    model.eng_barocaloric_energies[r_id] = EngBarocaloricEnergy(
        id=r_id, title=title, dt_bce=dt_bce, sens_id=sens_id
    )




def read_eng_thermomagnetoelectric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOMAGNETOELECTRIC_ENERGY`` or ``/ENG/TME_WORK`` (M311): Engine coupled thermomagnetoelectric multiferroic resonant energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOMAGNETOELECTRIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_tme, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOMAGNETOELECTRIC_ENERGY_1")
        dt_tme = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tme = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermomagnetoelectricEnergy
    r_id = block.user_id or (len(model.eng_thermomagnetoelectric_energies) + 1)
    model.eng_thermomagnetoelectric_energies[r_id] = EngThermomagnetoelectricEnergy(
        id=r_id, title=title, dt_tme=dt_tme, sens_id=sens_id
    )




def read_eng_electrohydrodynamic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROHYDRODYNAMIC_ENERGY`` or ``/ENG/EHD_WORK`` (M312): Engine electrohydrodynamic dielectric fluid pumping and space-charge coulombic body force transport energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROHYDRODYNAMIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ehd, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROHYDRODYNAMIC_ENERGY_1")
        dt_ehd = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ehd = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrohydrodynamicEnergy
    r_id = block.user_id or (len(model.eng_electrohydrodynamic_energies) + 1)
    model.eng_electrohydrodynamic_energies[r_id] = EngElectrohydrodynamicEnergy(
        id=r_id, title=title, dt_ehd=dt_ehd, sens_id=sens_id
    )




def read_eng_magnetogalvanic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAGNETOGALVANIC_ENERGY`` or ``/ENG/MGE_WORK`` (M313): Engine magnetogalvanic electromagnetic induction and lorentz force charge separation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAGNETOGALVANIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_mge, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_MAGNETOGALVANIC_ENERGY_1")
        dt_mge = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_mge = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngMagnetogalvanicEnergy
    r_id = block.user_id or (len(model.eng_magnetogalvanic_energies) + 1)
    model.eng_magnetogalvanic_energies[r_id] = EngMagnetogalvanicEnergy(
        id=r_id, title=title, dt_mge=dt_mge, sens_id=sens_id
    )




def read_eng_elastocaloric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELASTOCALORIC_ENERGY`` or ``/ENG/ELC_WORK`` (M314): Engine elastocaloric stress-induced martensitic entropy change and reversible solid-state superelastic heating/cooling energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELASTOCALORIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_elc, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELASTOCALORIC_ENERGY_1")
        dt_elc = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_elc = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElastocaloricEnergy
    r_id = block.user_id or (len(model.eng_elastocaloric_energies) + 1)
    model.eng_elastocaloric_energies[r_id] = EngElastocaloricEnergy(
        id=r_id, title=title, dt_elc=dt_elc, sens_id=sens_id
    )




def read_eng_thermophononic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOPHONONIC_ENERGY`` or ``/ENG/TPH_WORK`` (M315): Engine thermophononic lattice vibrational heat transport and phonon-scattering thermoelectric energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOPHONONIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_tph, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOPHONONIC_ENERGY_1")
        dt_tph = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tph = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermophononicEnergy
    r_id = block.user_id or (len(model.eng_thermophononic_energies) + 1)
    model.eng_thermophononic_energies[r_id] = EngThermophononicEnergy(
        id=r_id, title=title, dt_tph=dt_tph, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M435): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale multi-mode hybrid resonance energy and opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfplexphmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfplexphmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfplexphmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfplexphmnp=dt_etfplexphmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANTIMERON_PLASMON_POLARITON_RES_WORK`` (M476): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoantimeron-flexoplasmonic-flexopolaritonic nanoscale chiral-antimeron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcantimeronplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcantimeronplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcantimeronplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcantimeronplp=dt_etfcantimeronplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetophononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_POLARITON_RES_WORK`` (M436): Engine coupled electrothermal-flexomagnetic-flexophononic-flexopolaritonic nanoscale phonon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfphmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfphmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfphmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetophononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetophononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfphmnp=dt_etfphmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_POLARITON_RES_WORK`` (M438): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexopolaritonic nanoscale exciton-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfexmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfexmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfexmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfexmnp=dt_etfexmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_POLARITON_RES_WORK`` (M438): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexopolaritonic nanoscale exciton-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfexmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfexmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfexmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfexmnp=dt_etfexmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_POLARITON_RES_WORK`` (M439): Engine coupled electrothermal-flexomagnetic-flexomagnonic-flexopolaritonic nanoscale magnon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmgmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmgmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmgmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetomagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies) + 1)
    ee = EngElectrothermoflexomagnetomagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmgmnp=dt_etfmgmnp, sens_id=sens_id
    )
    model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies[r_id] = ee
    model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies[r_id] = ee




def read_eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_POLARITON_RES_WORK`` (M440): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexopolaritonic nanoscale plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies) + 1)
    ee = EngElectrothermoflexomagnetoplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfplp=dt_etfplp, sens_id=sens_id
    )
    model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies[r_id] = ee
    model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies[r_id] = ee




def read_eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_PLASMON_POLARITON_RES_WORK`` (M441): Engine coupled electrothermal-flexomagnetic-flexophononic-flexoplasmonic-flexopolaritonic nanoscale phonon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfphplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfphplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfphplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetophononicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies) + 1)
    ee = EngElectrothermoflexomagnetophononicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfphplp=dt_etfphplp, sens_id=sens_id
    )
    model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies[r_id] = ee
    model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies[r_id] = ee




def read_eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PLASMON_POLARITON_RES_WORK`` (M442): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexoplasmonic-flexopolaritonic nanoscale exciton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfexplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfexplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfexplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfexplp=dt_etfexplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_PLASMON_POLARITON_RES_WORK`` (M443): Engine coupled electrothermal-flexomagnetic-flexomagnonic-flexoplasmonic-flexopolaritonic nanoscale magnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmagplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmagplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmagplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetomagnonicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetomagnonicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmagplp=dt_etfmagplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_EXCITON_PLASMON_POLARITON_RES_WORK`` (M444): Engine coupled electrothermal-flexomagnetic-flexophononic-flexoexcitonic-flexoplasmonic-flexopolaritonic nanoscale phonon-exciton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfpexplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfpexplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfpexplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetophononicexcitonicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetophononicexcitonicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfpexplp=dt_etfpexplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_PLASMON_POLARITON_RES_WORK`` (M445): Engine coupled electrothermal-flexomagnetic-flexophononic-flexomagnonic-flexoplasmonic-flexopolaritonic nanoscale phonon-magnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfpmplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfpmplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfpmplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetophononicmagnonicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetophononicmagnonicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfpmplp=dt_etfpmplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_EXCITON_MAGNON_PLASMON_POLARITON_RES_WORK`` (M446): Engine coupled electrothermal-flexomagnetic-flexophononic-flexoexcitonic-flexomagnonic-flexoplasmonic-flexopolaritonic nanoscale phonon-exciton-magnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfpexmplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfpexmplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfpexmplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfpexmplp=dt_etfpexmplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PLASMON_POLARITON_RES_WORK`` (M447): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoplasmonic-flexopolaritonic nanoscale chiral-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcplp=dt_etfcplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PHONON_PLASMON_POLARITON_RES_WORK`` (M448): Engine coupled electrothermal-flexomagnetic-flexochiral-flexophononic-flexoplasmonic-flexopolaritonic nanoscale chiral-phonon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcphplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcphplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcphplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralphononicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralphononicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcphplp=dt_etfcphplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_EXCITON_PLASMON_POLARITON_RES_WORK`` (M449): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoexcitonic-flexoplasmonic-flexopolaritonic nanoscale chiral-exciton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcexplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcexplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcexplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralexcitonicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralexcitonicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcexplp=dt_etfcexplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MAGNON_PLASMON_POLARITON_RES_WORK`` (M450): Engine coupled electrothermal-flexomagnetic-flexochiral-flexomagnonic-flexoplasmonic-flexopolaritonic nanoscale chiral-magnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcmagplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcmagplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcmagplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralmagnonicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralmagnonicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcmagplp=dt_etfcmagplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPIN_PLASMON_POLARITON_RES_WORK`` (M451): Engine coupled electrothermal-flexomagnetic-flexochiral-flexospin-flexoplasmonic-flexopolaritonic nanoscale chiral-spin-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcspinplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcspinplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcspinplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralspinplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralspinplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcspinplp=dt_etfcspinplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPINON_PLASMON_POLARITON_RES_WORK`` (M452): Engine coupled electrothermal-flexomagnetic-flexochiral-flexospinon-flexoplasmonic-flexopolaritonic nanoscale chiral-spinon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcspinonplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcspinonplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcspinonplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralspinonplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralspinonplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcspinonplp=dt_etfcspinonplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HOLON_PLASMON_POLARITON_RES_WORK`` (M453): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoholon-flexoplasmonic-flexopolaritonic nanoscale chiral-holon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcholonplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcholonplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcholonplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralholonplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralholonplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcholonplp=dt_etfcholonplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ORBITON_PLASMON_POLARITON_RES_WORK`` (M454): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoorbiton-flexoplasmonic-flexopolaritonic nanoscale chiral-orbiton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcorbitonplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcorbitonplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcorbitonplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralorbitonplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralorbitonplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcorbitonplp=dt_etfcorbitonplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PLASMONON_PLASMON_POLARITON_RES_WORK`` (M455): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoplasmonon-flexoplasmonic-flexopolaritonic nanoscale chiral-plasmonon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcplasmononplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcplasmononplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcplasmononplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralplasmononplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralplasmononplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcplasmononplp=dt_etfcplasmononplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PARAMAGNON_PLASMON_POLARITON_RES_WORK`` (M456): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoparamagnon-flexoplasmonic-flexopolaritonic nanoscale chiral-paramagnon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcparamagnonplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcparamagnonplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcparamagnonplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralparamagnonplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralparamagnonplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcparamagnonplp=dt_etfcparamagnonplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_DYONIC_PLASMON_POLARITON_RES_WORK`` (M457): Engine coupled electrothermal-flexomagnetic-flexochiral-flexodyonic-flexoplasmonic-flexopolaritonic nanoscale chiral-dyonic-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcdyonicplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcdyonicplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcdyonicplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiraldyonicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiraldyonicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcdyonicplp=dt_etfcdyonicplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HOPFION_PLASMON_POLARITON_RES_WORK`` (M467): Engine coupled electrothermal-flexomagnetic-flexochiral-flexohopfion-flexoplasmonic-flexopolaritonic nanoscale chiral-hopfion-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfchopfionplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfchopfionplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfchopfionplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralhopfionplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralhopfionplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfchopfionplp=dt_etfchopfionplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MONOPOLE_PLASMON_POLARITON_RES_WORK`` (M468): Engine coupled electrothermal-flexomagnetic-flexochiral-flexomonopole-flexoplasmonic-flexopolaritonic nanoscale chiral-monopole-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcmonopoleplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcmonopoleplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcmonopoleplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralmonopoleplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralmonopoleplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcmonopoleplp=dt_etfcmonopoleplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPHALERON_PLASMON_POLARITON_RES_WORK`` (M469): Engine coupled electrothermal-flexomagnetic-flexochiral-flexosphaleron-flexoplasmonic-flexopolaritonic nanoscale chiral-sphaleron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcsphaleronplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcsphaleronplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcsphaleronplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralsphaleronplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralsphaleronplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcsphaleronplp=dt_etfcsphaleronplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_TORON_PLASMON_POLARITON_RES_WORK`` (M470): Engine coupled electrothermal-flexomagnetic-flexochiral-flexotoron-flexoplasmonic-flexopolaritonic nanoscale chiral-toron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfctoronplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfctoronplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfctoronplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiraltoronplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiraltoronplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfctoronplp=dt_etfctoronplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralbobberplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBOBBERPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BOBBER_PLASMON_POLARITON_RES_WORK`` (M471): Engine coupled electrothermal-flexomagnetic-flexochiral-flexobobber-flexoplasmonic-flexopolaritonic nanoscale chiral-bobber-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBOBBERPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcbobberplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALBOBBERPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcbobberplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcbobberplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralbobberplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralbobberplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralbobberplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralbobberplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcbobberplp=dt_etfcbobberplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BLOCH_POINT_PLASMON_POLARITON_RES_WORK`` (M472): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoblochpoint-flexoplasmonic-flexopolaritonic nanoscale chiral-bloch-point-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcblochpointplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcblochpointplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcblochpointplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcblochpointplp=dt_etfcblochpointplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HEDGEHOG_PLASMON_POLARITON_RES_WORK`` (M473): Engine coupled electrothermal-flexomagnetic-flexochiral-flexohedgehog-flexoplasmonic-flexopolaritonic nanoscale chiral-hedgehog-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfchedgehogplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfchedgehogplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfchedgehogplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfchedgehogplp=dt_etfchedgehogplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SKYRMIONIUM_PLASMON_POLARITON_RES_WORK`` (M474): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoskyrmionium-flexoplasmonic-flexopolaritonic nanoscale chiral-skyrmionium-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcskyrmioniumplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcskyrmioniumplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcskyrmioniumplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcskyrmioniumplp=dt_etfcskyrmioniumplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANTISKYRMION_PLASMON_POLARITON_RES_WORK`` (M475): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoantiskyrmion-flexoplasmonic-flexopolaritonic nanoscale chiral-antiskyrmion-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcantiskyrmionplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcantiskyrmionplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcantiskyrmionplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcantiskyrmionplp=dt_etfcantiskyrmionplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_VORTEX_PLASMON_POLARITON_RES_WORK`` (M466): Engine coupled electrothermal-flexomagnetic-flexochiral-flexovortex-flexoplasmonic-flexopolaritonic nanoscale chiral-vortex-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcvortexplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcvortexplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcvortexplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralvortexplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralvortexplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcvortexplp=dt_etfcvortexplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SOLITON_PLASMON_POLARITON_RES_WORK`` (M465): Engine coupled electrothermal-flexomagnetic-flexochiral-flexosoliton-flexoplasmonic-flexopolaritonic nanoscale chiral-soliton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcsolitonplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcsolitonplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcsolitonplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralsolitonplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralsolitonplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcsolitonplp=dt_etfcsolitonplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_INSTANTON_PLASMON_POLARITON_RES_WORK`` (M464): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoinstanton-flexoplasmonic-flexopolaritonic nanoscale chiral-instanton-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcinstantonplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcinstantonplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcinstantonplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralinstantonplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralinstantonplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcinstantonplp=dt_etfcinstantonplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BIMERON_PLASMON_POLARITON_RES_WORK`` (M463): Engine coupled electrothermal-flexomagnetic-flexochiral-flexobimeron-flexoplasmonic-flexopolaritonic nanoscale chiral-bimeron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcbimeronplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcbimeronplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcbimeronplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralbimeronplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralbimeronplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcbimeronplp=dt_etfcbimeronplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MERON_PLASMON_POLARITON_RES_WORK`` (M462): Engine coupled electrothermal-flexomagnetic-flexochiral-flexomeron-flexoplasmonic-flexopolaritonic nanoscale chiral-meron-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcmeronplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcmeronplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcmeronplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralmeronplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralmeronplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcmeronplp=dt_etfcmeronplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SKYRMION_PLASMON_POLARITON_RES_WORK`` (M461): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoskyrmion-flexoplasmonic-flexopolaritonic nanoscale chiral-skyrmion-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcskyrmionplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcskyrmionplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcskyrmionplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralskyrmionplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralskyrmionplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcskyrmionplp=dt_etfcskyrmionplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANYON_PLASMON_POLARITON_RES_WORK`` (M460): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoanyon-flexoplasmonic-flexopolaritonic nanoscale chiral-anyon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcanyonplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcanyonplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcanyonplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralanyonplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralanyonplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcanyonplp=dt_etfcanyonplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MAJORANA_PLASMON_POLARITON_RES_WORK`` (M459): Engine coupled electrothermal-flexomagnetic-flexochiral-flexomajorana-flexoplasmonic-flexopolaritonic nanoscale chiral-majorana-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcmajoranaplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcmajoranaplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcmajoranaplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralmajoranaplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralmajoranaplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcmajoranaplp=dt_etfcmajoranaplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_AXIONIC_PLASMON_POLARITON_RES_WORK`` (M458): Engine coupled electrothermal-flexomagnetic-flexochiral-flexoaxionic-flexoplasmonic-flexopolaritonic nanoscale chiral-axionic-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfcaxionicplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfcaxionicplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfcaxionicplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetochiralaxionicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetochiralaxionicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfcaxionicplp=dt_etfcaxionicplp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_PLASMON_POLARITON_RES_WORK`` (M441): Engine coupled electrothermal-flexomagnetic-flexophononic-flexoplasmonic-flexopolaritonic nanoscale phonon-plasmon-polariton hybrid resonance energy and opto-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfphplp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfphplp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfphplp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetophononicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies) + 1)
    ee = EngElectrothermoflexomagnetophononicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfphplp=dt_etfphplp, sens_id=sens_id
    )
    model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies[r_id] = ee
    model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies[r_id] = ee




def read_eng_thermoplasmonic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOPLASMONIC_ENERGY`` or ``/ENG/TPL_WORK`` (M316): Engine thermoplasmonic resonant metallic nanoparticle Joule dissipation and photothermal conversion energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOPLASMONIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_tpl, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOPLASMONIC_ENERGY_1")
        dt_tpl = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tpl = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermoplasmonicEnergy
    r_id = block.user_id or (len(model.eng_thermoplasmonic_energies) + 1)
    model.eng_thermoplasmonic_energies[r_id] = EngThermoplasmonicEnergy(
        id=r_id, title=title, dt_tpl=dt_tpl, sens_id=sens_id
    )




def read_eng_thermomagnetic_generator_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOMAGNETIC_GENERATOR_ENERGY`` or ``/ENG/TMG_WORK`` (M317): Engine thermomagnetic generator Curie-temperature phase-transition magnetization energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOMAGNETIC_GENERATOR_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_tmg, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOMAGNETIC_GENERATOR_ENERGY_1")
        dt_tmg = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tmg = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermomagneticGeneratorEnergy
    r_id = block.user_id or (len(model.eng_thermomagnetic_generator_energies) + 1)
    model.eng_thermomagnetic_generator_energies[r_id] = EngThermomagneticGeneratorEnergy(
        id=r_id, title=title, dt_tmg=dt_tmg, sens_id=sens_id
    )




def read_eng_magnetorheological_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/MAGNETORHEOLOGICAL_ENERGY`` or ``/ENG/MR_WORK`` (M318): Engine magnetorheological fluid yield stress activation and magnetic field controllable shear dissipation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/MAGNETORHEOLOGICAL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_mr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_MAGNETORHEOLOGICAL_ENERGY_1")
        dt_mr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_mr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngMagnetorheologicalEnergy
    r_id = block.user_id or (len(model.eng_magnetorheological_energies) + 1)
    model.eng_magnetorheological_energies[r_id] = EngMagnetorheologicalEnergy(
        id=r_id, title=title, dt_mr=dt_mr, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M433): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfplexmmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfplexmmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfplexmmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    eng_obj = EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfplexmmnp=dt_etfplexmmnp, sens_id=sens_id
    )
    model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = eng_obj
    if hasattr(model, 'eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies'):
        model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies[r_id] = eng_obj




def read_eng_electrorheological_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTRORHEOLOGICAL_ENERGY`` or ``/ENG/ER_WORK`` (M319): Engine electrorheological fluid electric-field induced fibrillated chain polarization and controllable shear yield dissipation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTRORHEOLOGICAL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_er, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTRORHEOLOGICAL_ENERGY_1")
        dt_er = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_er = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrorheologicalEnergy
    r_id = block.user_id or (len(model.eng_electrorheological_energies) + 1)
    model.eng_electrorheological_energies[r_id] = EngElectrorheologicalEnergy(
        id=r_id, title=title, dt_er=dt_er, sens_id=sens_id
    )




def read_eng_thermoacoustic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOACOUSTIC_ENERGY`` or ``/ENG/TA_WORK`` (M320): Engine thermoacoustic coupled acoustic wave resonance and oscillating thermal gradient heat pumping energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOACOUSTIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ta, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOACOUSTIC_ENERGY_1")
        dt_ta = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ta = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermoacousticEnergy
    r_id = block.user_id or (len(model.eng_thermoacoustic_energies) + 1)
    model.eng_thermoacoustic_energies[r_id] = EngThermoacousticEnergy(
        id=r_id, title=title, dt_ta=dt_ta, sens_id=sens_id
    )




def read_eng_ferroelectric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FERROELECTRIC_ENERGY`` or ``/ENG/FE_WORK`` (M321): Engine ferroelectric polarization switching, domain wall motion hysteresis, and electromechanical coupling dissipation energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FERROELECTRIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fe, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FERROELECTRIC_ENERGY_1")
        dt_fe = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fe = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFerroelectricEnergy
    r_id = block.user_id or (len(model.eng_ferroelectric_energies) + 1)
    model.eng_ferroelectric_energies[r_id] = EngFerroelectricEnergy(
        id=r_id, title=title, dt_fe=dt_fe, sens_id=sens_id
    )




def read_eng_flexoelectric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOELECTRIC_ENERGY`` or ``/ENG/FLEXO_WORK`` (M322): Engine flexoelectric strain-gradient induced electric polarization and nanoscale electromechanical energy tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOELECTRIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_flx, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOELECTRIC_ENERGY_1")
        dt_flx = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_flx = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexoelectricEnergy
    r_id = block.user_id or (len(model.eng_flexoelectric_energies) + 1)
    model.eng_flexoelectric_energies[r_id] = EngFlexoelectricEnergy(
        id=r_id, title=title, dt_flx=dt_flx, sens_id=sens_id
    )




def read_eng_pyromagnetic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PYROMAGNETIC_ENERGY`` or ``/ENG/PYROMAG_WORK`` (M323): Engine pyromagnetic temperature-dependent magnetization change, thermomagnetic entropy flux, and magnetic pyroelectric energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PYROMAGNETIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_pmg, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_PYROMAGNETIC_ENERGY_1")
        dt_pmg = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_pmg = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 0 else 0

    from ...model.entities import EngPyromagneticEnergy
    r_id = block.user_id or (len(model.eng_pyromagnetic_energies) + 1)
    model.eng_pyromagnetic_energies[r_id] = EngPyromagneticEnergy(
        id=r_id, title=title, dt_pmg=dt_pmg, sens_id=sens_id
    )




def read_eng_piezothermal_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PIEZOTHERMAL_ENERGY`` or ``/ENG/PIEZO_THERM_WORK`` (M324): Engine piezothermal coupled stress-temperature pyro-piezoelectric energy conversion and thermomechanical dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PIEZOTHERMAL_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_pzt, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_PIEZOTHERMAL_ENERGY_1")
        dt_pzt = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_pzt = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngPiezothermalEnergy
    r_id = block.user_id or (len(model.eng_piezothermal_energies) + 1)
    model.eng_piezothermal_energies[r_id] = EngPiezothermalEnergy(
        id=r_id, title=title, dt_pzt=dt_pzt, sens_id=sens_id
    )




def read_eng_thermoflexoelectric_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOFLEXOELECTRIC_ENERGY`` or ``/ENG/THERMOFLEXO_WORK`` (M325): Engine coupled temperature-gradient and strain-gradient induced electric polarization and nanoscale coupled thermal-flexoelectric energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOFLEXOELECTRIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_tfe, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOFLEXOELECTRIC_ENERGY_1")
        dt_tfe = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tfe = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermoflexoelectricEnergy
    r_id = block.user_id or (len(model.eng_thermoflexoelectric_energies) + 1)
    model.eng_thermoflexoelectric_energies[r_id] = EngThermoflexoelectricEnergy(
        id=r_id, title=title, dt_tfe=dt_tfe, sens_id=sens_id
    )




def read_eng_flexomagnetic_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETIC_ENERGY`` or ``/ENG/FLEXOMAG_WORK`` (M326): Engine flexomagnetic strain gradient-induced magnetic polarization and nanoscale coupled mechanical-magnetic energy conversion tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETIC_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fm, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETIC_ENERGY_1")
        dt_fm = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fm = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagneticEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetic_energies) + 1)
    model.eng_flexomagnetic_energies[r_id] = EngFlexomagneticEnergy(
        id=r_id, title=title, dt_fm=dt_fm, sens_id=sens_id
    )




def read_eng_pyroelectric_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/PYROELECTRIC_RESONANCE_ENERGY`` or ``/ENG/PYRO_RES_WORK`` (M327): Engine pyroelectric acoustic resonance energy and high-frequency thermal-dielectric polarization dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/PYROELECTRIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_pyr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_PYROELECTRIC_RESONANCE_ENERGY_1")
        dt_pyr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_pyr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngPyroelectricResonanceEnergy
    r_id = block.user_id or (len(model.eng_pyroelectric_resonance_energies) + 1)
    model.eng_pyroelectric_resonance_energies[r_id] = EngPyroelectricResonanceEnergy(
        id=r_id, title=title, dt_pyr=dt_pyr, sens_id=sens_id
    )




def read_eng_thermomagnetic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/THERMOMAGNETIC_RESONANCE_ENERGY`` or ``/ENG/THERMOMAG_RES_WORK`` (M328): Engine thermomagnetic acoustic resonance energy and high-frequency coupled magneto-caloric oscillation dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/THERMOMAGNETIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_tmr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_THERMOMAGNETIC_RESONANCE_ENERGY_1")
        dt_tmr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_tmr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngThermomagneticResonanceEnergy
    r_id = block.user_id or (len(model.eng_thermomagnetic_resonance_energies) + 1)
    model.eng_thermomagnetic_resonance_energies[r_id] = EngThermomagneticResonanceEnergy(
        id=r_id, title=title, dt_tmr=dt_tmr, sens_id=sens_id
    )




def read_eng_flexothermal_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMAL_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_RES_WORK`` (M329): Engine flexothermal acoustic resonance energy and nanoscale strain-gradient coupled thermoelastic resonance dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMAL_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMAL_RESONANCE_ENERGY_1")
        dt_ftr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermalResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermal_resonance_energies) + 1)
    model.eng_flexothermal_resonance_energies[r_id] = EngFlexothermalResonanceEnergy(
        id=r_id, title=title, dt_ftr=dt_ftr, sens_id=sens_id
    )




def read_eng_electromagnetomechanical_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROMAGNETOMECHANICAL_RESONANCE_ENERGY`` or ``/ENG/EMM_RES_WORK`` (M330): Engine multi-field coupled electromagnetomechanical acoustic resonance energy and high-frequency wave-matter polarization dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROMAGNETOMECHANICAL_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_emmr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROMAGNETOMECHANICAL_RESONANCE_ENERGY_1")
        dt_emmr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_emmr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectromagnetomechanicalResonanceEnergy
    r_id = block.user_id or (len(model.eng_electromagnetomechanical_resonance_energies) + 1)
    model.eng_electromagnetomechanical_resonance_energies[r_id] = EngElectromagnetomechanicalResonanceEnergy(
        id=r_id, title=title, dt_emmr=dt_emmr, sens_id=sens_id
    )




def read_eng_flexomagnetoelectric_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOELECTRIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAG_ELEC_RES_WORK`` (M331): Engine coupled flexomagnetic-flexoelectric acoustic resonance energy and nanoscale strain-gradient electromagnetic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOELECTRIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmer, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOELECTRIC_RESONANCE_ENERGY_1")
        dt_fmer = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmer = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoelectricResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoelectric_resonance_energies) + 1)
    model.eng_flexomagnetoelectric_resonance_energies[r_id] = EngFlexomagnetoelectricResonanceEnergy(
        id=r_id, title=title, dt_fmer=dt_fmer, sens_id=sens_id
    )




def read_eng_flexothermomagnetic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOMAGNETIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAG_RES_WORK`` (M332): Engine coupled flexomagnetic-flexothermal acoustic resonance energy and nanoscale strain-gradient magneto-caloric conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOMAGNETIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftmr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOMAGNETIC_RESONANCE_ENERGY_1")
        dt_ftmr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftmr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermomagneticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermomagnetic_resonance_energies) + 1)
    model.eng_flexothermomagnetic_resonance_energies[r_id] = EngFlexothermomagneticResonanceEnergy(
        id=r_id, title=title, dt_ftmr=dt_ftmr, sens_id=sens_id
    )




def read_eng_flexothermoelectric_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOELECTRIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_ELEC_RES_WORK`` (M333): Engine coupled flexoelectric-flexothermal acoustic resonance energy and nanoscale strain-gradient thermoelectric conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOELECTRIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fter, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOELECTRIC_RESONANCE_ENERGY_1")
        dt_fter = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fter = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoelectricResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoelectric_resonance_energies) + 1)
    model.eng_flexothermoelectric_resonance_energies[r_id] = EngFlexothermoelectricResonanceEnergy(
        id=r_id, title=title, dt_fter=dt_fter, sens_id=sens_id
    )




def read_eng_flexothermoacoustic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_AC_RES_WORK`` (M334): Engine coupled flexoacoustic-flexothermal acoustic resonance energy and nanoscale strain-gradient thermoacoustic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOACOUSTIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftar, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOACOUSTIC_RESONANCE_ENERGY_1")
        dt_ftar = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftar = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoacousticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoacoustic_resonance_energies) + 1)
    model.eng_flexothermoacoustic_resonance_energies[r_id] = EngFlexothermoacousticResonanceEnergy(
        id=r_id, title=title, dt_ftar=dt_ftar, sens_id=sens_id
    )




def read_eng_flexoelectromagnetic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOELECTROMAGNETIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOELEC_MAG_RES_WORK`` (M335): Engine coupled flexoelectric-flexomagnetic full electromagnetic acoustic resonance energy and nanoscale strain-gradient electromagnetic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOELECTROMAGNETIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_femr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOELECTROMAGNETIC_RESONANCE_ENERGY_1")
        dt_femr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_femr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexoelectromagneticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexoelectromagnetic_resonance_energies) + 1)
    model.eng_flexoelectromagnetic_resonance_energies[r_id] = EngFlexoelectromagneticResonanceEnergy(
        id=r_id, title=title, dt_femr=dt_femr, sens_id=sens_id
    )




def read_eng_flexoelectroacoustic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOELECTROACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOELEC_AC_RES_WORK`` (M336): Engine coupled flexoelectric-flexoacoustic acoustic resonance energy and nanoscale strain-gradient electroacoustic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOELECTROACOUSTIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fear, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOELECTROACOUSTIC_RESONANCE_ENERGY_1")
        dt_fear = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fear = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexoelectroacousticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexoelectroacoustic_resonance_energies) + 1)
    model.eng_flexoelectroacoustic_resonance_energies[r_id] = EngFlexoelectroacousticResonanceEnergy(
        id=r_id, title=title, dt_fear=dt_fear, sens_id=sens_id
    )




def read_eng_flexomagnetoacoustic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAG_AC_RES_WORK`` (M337): Engine coupled flexomagnetic-flexoacoustic acoustic resonance energy and nanoscale strain-gradient magneto-acoustic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOACOUSTIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmar, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOACOUSTIC_RESONANCE_ENERGY_1")
        dt_fmar = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmar = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoacousticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoacoustic_resonance_energies) + 1)
    model.eng_flexomagnetoacoustic_resonance_energies[r_id] = EngFlexomagnetoacousticResonanceEnergy(
        id=r_id, title=title, dt_fmar=dt_fmar, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_POLARITON_RES_WORK`` (M422): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale plasmon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmpmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmpmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmpmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmpmpp=dt_etfmpmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_POLARITON_RES_WORK`` (M423): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmxmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmxmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmxmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmxmpp=dt_etfmxmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_POLARITON_RES_WORK`` (M424): Engine coupled electrothermal-flexomagnetic-flexophononic-flexomagnonic-flexopolaritonic nanoscale phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmphmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmphmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmphmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetophononicmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetophononicmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmphmpp=dt_etfmphmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M425): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton 4-hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmpxmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmpxmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmpxmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmpxmpp=dt_etfmpxmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_PHONON_POLARITON_RES_WORK`` (M426): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-magnon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmpxmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmpxmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmpxmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmpxmpp=dt_etfmpxmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_MAGNON_POLARITON_RES_WORK`` (M427): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale plasmon-phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmppmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmppmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmppmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicphononicmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicphononicmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmppmpp=dt_etfmppmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M428): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale exciton-phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmexpmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmexpmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmexpmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicphononicmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicphononicmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmexpmpp=dt_etfmexpmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M429): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmpexpmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmpexpmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmpexpmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmpexpmpp=dt_etfmpexpmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_POLARITON_RES_WORK`` (M430): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmexmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmexmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmexmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmexmnp=dt_etfmexmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_POLARITON_RES_WORK`` (M434): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexophononic-flexopolaritonic nanoscale plasmon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfplphnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfplphnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfplphnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfplphnp=dt_etfplphnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M435): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale multi-mode hybrid resonance energy and opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfplexphmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfplexphmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfplexphmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfplexphmnp=dt_etfplexphmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M433): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfplexmmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfplexmmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfplexmmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    eng_obj = EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfplexmmnp=dt_etfplexmmnp, sens_id=sens_id
    )
    model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = eng_obj
    if hasattr(model, 'eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies'):
        model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies[r_id] = eng_obj




def read_eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_POLARITON_RES_WORK`` (M432): Engine coupled electrothermal-flexomagnetic-flexophononic-flexomagnonic-flexopolaritonic nanoscale phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfphmmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfphmmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfphmmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetophononicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetophononicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfphmmnp=dt_etfphmmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M433): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfplexmmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfplexmmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfplexmmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    eng_obj = EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfplexmmnp=dt_etfplexmmnp, sens_id=sens_id
    )
    model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = eng_obj
    if hasattr(model, 'eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies'):
        model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies[r_id] = eng_obj




def read_eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_POLARITON_RES_WORK`` (M431): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale plasmon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfplmmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfplmmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfplmmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfplmmnp=dt_etfplmmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_POLARITON_RES_WORK`` (M430): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale exciton-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmexmnp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmexmnp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmexmnp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmexmnp=dt_etfmexmnp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M428): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexophononic-flexomagnonic-flexopolaritonic nanoscale exciton-phonon-magnon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmexpmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmexpmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmexpmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicphononicmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicphononicmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmexpmpp=dt_etfmexpmpp, sens_id=sens_id
    )




def read_eng_flexothermoelectromagnetic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOELECTROMAGNETIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EM_RES_WORK`` (M338): Engine coupled flexothermal-flexoelectromagnetic full multi-field acoustic resonance energy and nanoscale strain-gradient thermo-electromagnetic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOELECTROMAGNETIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftemr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOELECTROMAGNETIC_RESONANCE_ENERGY_1")
        dt_ftemr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftemr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoelectromagneticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoelectromagnetic_resonance_energies) + 1)
    model.eng_flexothermoelectromagnetic_resonance_energies[r_id] = EngFlexothermoelectromagneticResonanceEnergy(
        id=r_id, title=title, dt_ftemr=dt_ftemr, sens_id=sens_id
    )




def read_eng_flexothermoelectroacoustic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOELECTROACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EA_RES_WORK`` (M339): Engine coupled flexothermal-flexoelectroacoustic full thermo-electro-acoustic resonance energy and nanoscale strain-gradient thermo-electroacoustic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOELECTROACOUSTIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftear, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOELECTROACOUSTIC_RESONANCE_ENERGY_1")
        dt_ftear = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftear = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoelectroacousticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoelectroacoustic_resonance_energies) + 1)
    model.eng_flexothermoelectroacoustic_resonance_energies[r_id] = EngFlexothermoelectroacousticResonanceEnergy(
        id=r_id, title=title, dt_ftear=dt_ftear, sens_id=sens_id
    )




def read_eng_flexothermomagnetoacoustic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MA_RES_WORK`` (M340): Engine coupled flexothermal-flexomagnetoacoustic full thermo-magneto-acoustic resonance energy and nanoscale strain-gradient thermo-magnetoacoustic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftmar, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_ENERGY_1")
        dt_ftmar = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftmar = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermomagnetoacousticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermomagnetoacoustic_resonance_energies) + 1)
    model.eng_flexothermomagnetoacoustic_resonance_energies[r_id] = EngFlexothermomagnetoacousticResonanceEnergy(
        id=r_id, title=title, dt_ftmar=dt_ftmar, sens_id=sens_id
    )




def read_eng_flexothermoelectromagnetoacoustic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EMA_RES_WORK`` (M341): Engine coupled flexothermal-flexoelectro-flexomagneto-flexoacoustic full multi-field acoustic resonance energy and nanoscale strain-gradient thermo-electromagnetic-acoustic conversion dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftemmar, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_ENERGY_1")
        dt_ftemmar = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftemmar = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoelectromagnetoacousticResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoelectromagnetoacoustic_resonance_energies) + 1)
    model.eng_flexothermoelectromagnetoacoustic_resonance_energies[r_id] = EngFlexothermoelectromagnetoacousticResonanceEnergy(
        id=r_id, title=title, dt_ftemmar=dt_ftemmar, sens_id=sens_id
    )




def read_eng_flexothermophotonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPHOTONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHOTON_RES_WORK`` (M342): Engine coupled flexothermal-flexophotonic high-frequency nanoscale optical resonance energy and strain-gradient photon-polariton dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPHOTONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPHOTONIC_RESONANCE_ENERGY_1")
        dt_ftpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermophotonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermophotonic_resonance_energies) + 1)
    model.eng_flexothermophotonic_resonance_energies[r_id] = EngFlexothermophotonicResonanceEnergy(
        id=r_id, title=title, dt_ftpr=dt_ftpr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_RES_WORK`` (M343): Engine coupled flexothermal-flexoplasmonic nanoscale surface-plasmon polariton resonance energy and strain-gradient photothermal dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftplr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONIC_RESONANCE_ENERGY_1")
        dt_ftplr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftplr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonic_resonance_energies[r_id] = EngFlexothermoplasmonicResonanceEnergy(
        id=r_id, title=title, dt_ftplr=dt_ftplr, sens_id=sens_id
    )




def read_eng_flexothermoexcitonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOEXCITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_RES_WORK`` (M344): Engine coupled flexothermal-flexoexcitonic nanoscale exciton-polariton resonance energy and strain-gradient optoelectronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOEXCITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftexr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOEXCITONIC_RESONANCE_ENERGY_1")
        dt_ftexr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftexr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoexcitonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoexcitonic_resonance_energies) + 1)
    model.eng_flexothermoexcitonic_resonance_energies[r_id] = EngFlexothermoexcitonicResonanceEnergy(
        id=r_id, title=title, dt_ftexr=dt_ftexr, sens_id=sens_id
    )




def read_eng_flexothermomagnonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAGNON_RES_WORK`` (M345): Engine coupled flexothermal-flexomagnonic nanoscale spin-wave magnon polariton resonance energy and strain-gradient spintronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOMAGNONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftmr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOMAGNONIC_RESONANCE_ENERGY_1")
        dt_ftmr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftmr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermomagnonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermomagnonic_resonance_energies) + 1)
    model.eng_flexothermomagnonic_resonance_energies[r_id] = EngFlexothermomagnonicResonanceEnergy(
        id=r_id, title=title, dt_ftmr=dt_ftmr, sens_id=sens_id
    )




def read_eng_flexothermomagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAGNON_POLARITON_RES_WORK`` (M346): Engine coupled flexothermal-flexomagnonic-flexophotonic nanoscale magnon-polariton hybrid resonance energy and strain-gradient spintronic-photonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermomagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermomagnonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermomagnonpolaritonic_resonance_energies[r_id] = EngFlexothermomagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftmpr=dt_ftmpr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_POLARITON_RES_WORK`` (M347): Engine coupled flexothermal-flexoplasmonic-flexophotonic nanoscale surface plasmon polariton resonance energy and strain-gradient electromagnetic-photothermal dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftppr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftppr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftppr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftppr=dt_ftppr, sens_id=sens_id
    )




def read_eng_flexothermoexcitonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_POLARITON_RES_WORK`` (M348): Engine coupled flexothermal-flexoexcitonic-flexophotonic nanoscale exciton-polariton hybrid resonance energy and strain-gradient optoelectronic-photothermal dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftepr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftepr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftepr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoexcitonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoexcitonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoexcitonpolaritonic_resonance_energies[r_id] = EngFlexothermoexcitonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftepr=dt_ftepr, sens_id=sens_id
    )




def read_eng_flexothermophononpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_POLARITON_RES_WORK`` (M349): Engine coupled flexothermal-flexophononic-flexophotonic nanoscale lattice phonon-polariton hybrid resonance energy and strain-gradient elastodynamic-photothermal dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftphpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftphpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftphpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermophononpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermophononpolaritonic_resonance_energies) + 1)
    model.eng_flexothermophononpolaritonic_resonance_energies[r_id] = EngFlexothermophononpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftphpr=dt_ftphpr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonphononpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_PHONON_POLARITON_RES_WORK`` (M350): Engine coupled flexothermal-flexoplasmonic-flexophononic nanoscale surface plasmon phonon-polariton hybrid resonance energy and strain-gradient electromagnetic-elastodynamic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpppr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpppr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpppr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonphononpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonphononpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonphononpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonphononpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpppr=dt_ftpppr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonexcitonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_POLARITON_RES_WORK`` (M351): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic nanoscale surface plasmon exciton-polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpepr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpepr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpepr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonexcitonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonexcitonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpepr=dt_ftpepr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_MAGNON_POLARITON_RES_WORK`` (M352): Engine coupled flexothermal-flexoplasmonic-flexomagnonic nanoscale surface plasmon magnon-polariton hybrid resonance energy and strain-gradient electromagnetic-spintronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpmpr=dt_ftpmpr, sens_id=sens_id
    )




def read_eng_flexothermoexcitonphononpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_PHONON_POLARITON_RES_WORK`` (M353): Engine coupled flexothermal-flexoexcitonic-flexophononic nanoscale exciton phonon-polariton hybrid resonance energy and strain-gradient optoelectronic-elastodynamic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fteppr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fteppr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fteppr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoexcitonphononpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoexcitonphononpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoexcitonphononpolaritonic_resonance_energies[r_id] = EngFlexothermoexcitonphononpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fteppr=dt_fteppr, sens_id=sens_id
    )




def read_eng_flexothermoexcitonmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_MAGNON_POLARITON_RES_WORK`` (M354): Engine coupled flexothermal-flexoexcitonic-flexomagnonic nanoscale exciton magnon-polariton hybrid resonance energy and strain-gradient optoelectronic-spintronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftempr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftempr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftempr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoexcitonmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies[r_id] = EngFlexothermoexcitonmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftempr=dt_ftempr, sens_id=sens_id
    )




def read_eng_flexothermophononmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_MAGNON_POLARITON_RES_WORK`` (M355): Engine coupled flexothermal-flexophononic-flexomagnonic nanoscale lattice phonon magnon-polariton hybrid resonance energy and strain-gradient elastodynamic-spintronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermophononmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermophononmagnonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermophononmagnonpolaritonic_resonance_energies[r_id] = EngFlexothermophononmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpmpr=dt_ftpmpr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK`` (M356): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexophononic nanoscale plasmon exciton-phonon polariton multi-mode resonance energy and strain-gradient electromagnetic-optoelectronic-elastodynamic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpeppr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpeppr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpeppr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonexcitonphononpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonexcitonphononpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpeppr=dt_ftpeppr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M357): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexomagnonic nanoscale surface plasmon exciton-magnon polariton multi-mode resonance energy and strain-gradient electromagnetic-optoelectronic-spintronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpempr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpempr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpempr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonexcitonmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonexcitonmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpempr=dt_ftpempr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_PHONON_MAGNON_POLARITON_RES_WORK`` (M358): Engine coupled flexothermal-flexoplasmonic-flexophononic-flexomagnonic nanoscale surface plasmon phonon-magnon polariton multi-mode resonance energy and strain-gradient electromagnetic-elastodynamic-spintronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftppmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftppmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftppmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonphononmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonphononmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftppmpr=dt_ftppmpr, sens_id=sens_id
    )




def read_eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M359): Engine coupled flexothermal-flexoexcitonic-flexophononic-flexomagnonic nanoscale exciton phonon-magnon polariton multi-mode resonance energy and strain-gradient optoelectronic-elastodynamic-spintronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftepmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftepmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftepmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoexcitonphononmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies[r_id] = EngFlexothermoexcitonphononmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftepmpr=dt_ftepmpr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK`` (M360): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexophononic-flexomagnonic nanoscale surface plasmon exciton-phonon-magnon polariton quad-hybrid multi-mode resonance energy and strain-gradient electromagnetic-optoelectronic-elastodynamic-spintronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpepmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpepmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpepmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonexcitonphononmagnonpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonexcitonphononmagnonpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpepmpr=dt_ftpepmpr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_RES_WORK`` (M361): Engine coupled flexomagnetic-flexoplasmonic nanoscale surface magnetoplasmon polariton resonance energy and strain-gradient electromagnetic-magnetostatic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONIC_RESONANCE_ENERGY_1")
        dt_fmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicResonanceEnergy(
        id=r_id, title=title, dt_fmpr=dt_fmpr, sens_id=sens_id
    )




def read_eng_flexomagnetophononic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_RES_WORK`` (M362): Engine coupled flexomagnetic-flexophononic nanoscale acoustic phonon-magnon polariton resonance energy and strain-gradient magnetoelastic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONIC_RESONANCE_ENERGY_1")
        dt_fmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononic_resonance_energies) + 1)
    model.eng_flexomagnetophononic_resonance_energies[r_id] = EngFlexomagnetophononicResonanceEnergy(
        id=r_id, title=title, dt_fmpr=dt_fmpr, sens_id=sens_id
    )




def read_eng_flexomagnetoexcitonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOEXCITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_EXCITON_RES_WORK`` (M363): Engine coupled flexomagnetic-flexoexcitonic nanoscale exciton-magnon polariton resonance energy and strain-gradient optomagnetic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOEXCITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmer, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOEXCITONIC_RESONANCE_ENERGY_1")
        dt_fmer = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmer = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoexcitonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoexcitonic_resonance_energies) + 1)
    model.eng_flexomagnetoexcitonic_resonance_energies[r_id] = EngFlexomagnetoexcitonicResonanceEnergy(
        id=r_id, title=title, dt_fmer=dt_fmer, sens_id=sens_id
    )




def read_eng_flexomagnetopolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_POLARITON_RES_WORK`` (M364): Engine coupled flexomagnetic-flexopolaritonic nanoscale multi-mode magnon-polariton resonance energy and strain-gradient electromagnetic-magnetodynamic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpor, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpor = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpor = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetopolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetopolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetopolaritonic_resonance_energies[r_id] = EngFlexomagnetopolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpor=dt_fmpor, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicphonon_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICPHONON_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_PHONON_RES_WORK`` (M365): Engine coupled flexomagnetic-flexoplasmonic-flexophononic nanoscale surface magnetoplasmon-acoustic phonon polariton hybrid resonance energy and strain-gradient electromagnetic-elastodynamic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICPHONON_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmppr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICPHONON_RESONANCE_ENERGY_1")
        dt_fmppr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmppr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicphononResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicphonon_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicphonon_resonance_energies[r_id] = EngFlexomagnetoplasmonicphononResonanceEnergy(
        id=r_id, title=title, dt_fmppr=dt_fmppr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicexciton_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_RES_WORK`` (M366): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic nanoscale surface magnetoplasmon-exciton polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmper, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_ENERGY_1")
        dt_fmper = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmper = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicexcitonResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicexciton_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicexciton_resonance_energies[r_id] = EngFlexomagnetoplasmonicexcitonResonanceEnergy(
        id=r_id, title=title, dt_fmper=dt_fmper, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicmagnon_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_MAGNON_RES_WORK`` (M367): Engine coupled flexomagnetic-flexoplasmonic-flexomagnonic nanoscale surface magnetoplasmon-magnon polariton hybrid resonance energy and strain-gradient electromagnetic-magnetodynamic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpmr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_ENERGY_1")
        dt_fmpmr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpmr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicmagnonResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicmagnon_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicmagnon_resonance_energies[r_id] = EngFlexomagnetoplasmonicmagnonResonanceEnergy(
        id=r_id, title=title, dt_fmpmr=dt_fmpmr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_POLARITON_RES_WORK`` (M368): Engine coupled flexomagnetic-flexoplasmonic-flexopolaritonic nanoscale surface magnetoplasmon-polariton hybrid resonance energy and strain-gradient electromagnetic-magnetodynamic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmppor, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmppor = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmppor = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmppor=dt_fmppor, sens_id=sens_id
    )




def read_eng_flexomagnetophononicexcitonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_EXCITON_RES_WORK`` (M369): Engine coupled flexomagnetic-flexophononic-flexoexcitonic nanoscale acoustic phonon exciton-magnon polariton hybrid resonance energy and strain-gradient optoacoustic-magnetoelastic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmper, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_ENERGY_1")
        dt_fmper = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmper = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicexcitonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicexcitonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicexcitonic_resonance_energies[r_id] = EngFlexomagnetophononicexcitonicResonanceEnergy(
        id=r_id, title=title, dt_fmper=dt_fmper, sens_id=sens_id
    )




def read_eng_flexomagnetophononicmagnonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_MAGNON_RES_WORK`` (M370): Engine coupled flexomagnetic-flexophononic-flexomagnonic nanoscale acoustic phonon magnon-polariton hybrid resonance energy and strain-gradient elastomagnetic-magnetoelastic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpmr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_ENERGY_1")
        dt_fmpmr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpmr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicmagnonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicmagnonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicmagnonic_resonance_energies[r_id] = EngFlexomagnetophononicmagnonicResonanceEnergy(
        id=r_id, title=title, dt_fmpmr=dt_fmpmr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_POLARITON_RES_WORK`` (M371): Engine coupled flexomagnetic-flexophononic-flexopolaritonic nanoscale acoustic phonon polariton-magnon hybrid resonance energy and strain-gradient electromagnetic-optoacoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmppor, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmppor = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmppor = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmppor=dt_fmppor, sens_id=sens_id
    )




def read_eng_flexomagnetoexcitonicmagnonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_EXCITON_MAGNON_RES_WORK`` (M372): Engine coupled flexomagnetic-flexoexcitonic-flexomagnonic nanoscale exciton-magnon hybrid resonance energy and strain-gradient optomagnetic-magnetoelastic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmemr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_ENERGY_1")
        dt_fmemr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmemr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoexcitonicmagnonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoexcitonicmagnonic_resonance_energies) + 1)
    model.eng_flexomagnetoexcitonicmagnonic_resonance_energies[r_id] = EngFlexomagnetoexcitonicmagnonicResonanceEnergy(
        id=r_id, title=title, dt_fmemr=dt_fmemr, sens_id=sens_id
    )




def read_eng_flexomagnetoexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_EXCITON_POLARITON_RES_WORK`` (M373): Engine coupled flexomagnetic-flexoexcitonic-flexopolaritonic nanoscale exciton polariton-magnon hybrid resonance energy and strain-gradient electromagnetic-optoelectronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmepor, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmepor = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmepor = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetoexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmepor=dt_fmepor, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_RES_WORK`` (M374): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic nanoscale surface plasmon-exciton-magnon hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-magnetoelastic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpemr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY_1")
        dt_fmpemr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpemr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy(
        id=r_id, title=title, dt_fmpemr=dt_fmpemr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_POLARITON_RES_WORK`` (M375): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexopolaritonic nanoscale surface plasmon-exciton-polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-photonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpepr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpepr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpepr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpepr=dt_fmpepr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_MAGNON_POLARITON_RES_WORK`` (M376): Engine coupled flexomagnetic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale surface plasmon-magnon-polariton hybrid resonance energy and strain-gradient electromagnetic-magnetophotonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpmpr=dt_fmpmpr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicexcitonicmagnonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_EXCITON_MAGNON_RES_WORK`` (M377): Engine coupled flexomagnetic-flexophononic-flexoexcitonic-flexomagnonic nanoscale acoustic phonon exciton-magnon hybrid resonance energy and strain-gradient optoacoustic-magnetoelastic-electromagnetic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpemr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_ENERGY_1")
        dt_fmpemr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpemr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicexcitonicmagnonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies[r_id] = EngFlexomagnetophononicexcitonicmagnonicResonanceEnergy(
        id=r_id, title=title, dt_fmpemr=dt_fmpemr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_EXCITON_POLARITON_RES_WORK`` (M378): Engine coupled flexomagnetic-flexophononic-flexoexcitonic-flexopolaritonic nanoscale acoustic phonon exciton-polariton hybrid resonance energy and strain-gradient optoacoustic-polaritonic-electromagnetic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpepr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpepr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpepr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpepr=dt_fmpepr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_MAGNON_POLARITON_RES_WORK`` (M379): Engine coupled flexomagnetic-flexophononic-flexomagnonic-flexopolaritonic nanoscale acoustic phonon magnon-polariton hybrid resonance energy and strain-gradient optoacoustic-magnetophotonic-electromagnetic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpmpr=dt_fmpmpr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M380): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale surface plasmon exciton-magnon-polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-magnetophotonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpempr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpempr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpempr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpempr=dt_fmpempr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicplasmonicmagnonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_MAGNON_RES_WORK`` (M381): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexomagnonic nanoscale acoustic phonon surface plasmon-magnon hybrid resonance energy and strain-gradient optoacoustic-electromagnetic-magnetoelastic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpmr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_ENERGY_1")
        dt_fmpmr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpmr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicplasmonicmagnonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies[r_id] = EngFlexomagnetophononicplasmonicmagnonicResonanceEnergy(
        id=r_id, title=title, dt_fmpmr=dt_fmpmr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_POLARITON_RES_WORK`` (M382): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-polariton hybrid resonance energy and strain-gradient optoacoustic-electromagnetic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpopr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpopr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpopr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpopr=dt_fmpopr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_RES_WORK`` (M383): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic nanoscale surface plasmon exciton-magnon hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-magnetoelastic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpemr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY_1")
        dt_fmpemr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpemr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy(
        id=r_id, title=title, dt_fmpemr=dt_fmpemr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_EXCITON_POLARITON_RES_WORK`` (M384): Engine coupled flexomagnetic-flexoplasmonic-flexoexcitonic-flexopolaritonic nanoscale surface plasmon exciton-polariton hybrid resonance energy and strain-gradient electromagnetic-optoelectronic-photonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpepr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpepr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpepr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpepr=dt_fmpepr, sens_id=sens_id
    )




def read_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PLASMON_MAGNON_POLARITON_RES_WORK`` (M385): Engine coupled flexomagnetic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale surface plasmon-magnon-polariton hybrid resonance energy and strain-gradient electromagnetic-magnetophotonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpmpr=dt_fmpmpr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M386): Engine coupled flexomagnetic-flexophononic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale acoustic phonon exciton-magnon-polariton hybrid resonance energy and strain-gradient optoacoustic-magnetophotonic-electromagnetic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpempr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpempr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpempr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpempr=dt_fmpempr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicplasmonicexcitonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_RES_WORK`` (M387): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexoexcitonic nanoscale acoustic phonon surface plasmon-exciton hybrid resonance energy and strain-gradient optoelectronic-electromagnetic-optoacoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmppre, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_ENERGY_1")
        dt_fmppre = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmppre = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicplasmonicexcitonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies[r_id] = EngFlexomagnetophononicplasmonicexcitonicResonanceEnergy(
        id=r_id, title=title, dt_fmppre=dt_fmppre, sens_id=sens_id
    )




def read_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_POLARITON_RES_WORK`` (M388): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-polariton hybrid resonance energy and strain-gradient optoacoustic-electromagnetic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmpppr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmpppr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmpppr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmpppr=dt_fmpppr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_MAGNON_POLARITON_RES_WORK`` (M389): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-magnon-polariton hybrid resonance energy and strain-gradient optoacoustic-magnetophotonic-electromagnetic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmppmpr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmppmpr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmppmpr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicplasmonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicplasmonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmppmpr=dt_fmppmpr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_POLARITON_RES_WORK`` (M390): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexoexcitonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-exciton-polariton hybrid resonance energy and strain-gradient optoacoustic-electromagnetic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmppeopr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmppeopr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmppeopr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicplasmonicexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicplasmonicexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmppeopr=dt_fmppeopr, sens_id=sens_id
    )




def read_eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M391): Engine coupled flexomagnetic-flexophononic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale acoustic phonon surface plasmon-exciton-magnon-polariton hybrid resonance energy and strain-gradient optoacoustic-magnetophotonic-electromagnetic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_fmppempr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_fmppempr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_fmppempr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexomagnetophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexomagnetophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_fmppempr=dt_fmppempr, sens_id=sens_id
    )




def read_eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M392): Engine coupled flexothermal-flexophononic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale phonon-plasmon-exciton-magnon-polariton multiphysics hybrid resonance energy and strain-gradient thermoelectric-optomagnetic-photonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftppempr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftppempr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftppempr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexothermophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftppempr=dt_ftppempr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M393): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale plasmon-exciton-magnon-polariton hybrid resonance energy and thermal-strain gradient optomagnetic-photonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpempr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpempr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpempr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpempr=dt_ftpempr, sens_id=sens_id
    )




def read_eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_EXCITON_MAGNON_POLARITON_RES_WORK`` (M394): Engine coupled flexothermal-flexophononic-flexoexcitonic-flexomagnonic-flexopolaritonic nanoscale phonon-exciton-magnon-polariton hybrid resonance energy and thermal-strain gradient optomagnetic-photonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpempr, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpempr = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpempr = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermophononicexcitonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexothermophononicexcitonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpempr=dt_ftpempr, sens_id=sens_id
    )




def read_eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_MAGNON_POLARITON_RES_WORK`` (M395): Engine coupled flexothermal-flexoplasmonic-flexomagnonic-flexopolaritonic nanoscale surface plasmon-magnon-polariton hybrid resonance energy and thermal-gradient optomagnetic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpmp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpmp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpmp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpmp=dt_ftpmp, sens_id=sens_id
    )




def read_eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_POLARITON_RES_WORK`` (M396): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexopolaritonic nanoscale surface plasmon-exciton-polariton hybrid resonance energy and thermal-strain gradient optophotonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpep, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpep = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpep = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonicexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpep=dt_ftpep, sens_id=sens_id
    )




def read_eng_flexothermophononicmagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_MAGNON_POLARITON_RES_WORK`` (M397): Engine coupled flexothermal-flexophononic-flexomagnonic-flexopolaritonic nanoscale phonon-magnon-polariton hybrid resonance energy and thermal-gradient optomagnetic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpmp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpmp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpmp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermophononicmagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies[r_id] = EngFlexothermophononicmagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpmp=dt_ftpmp, sens_id=sens_id
    )




def read_eng_flexothermoplasmonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_POLARITON_RES_WORK`` (M398): Engine coupled flexothermal-flexoplasmonic-flexopolaritonic nanoscale surface plasmon-polariton hybrid resonance energy and thermal-gradient optophotonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonicpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpp=dt_ftpp, sens_id=sens_id
    )




def read_eng_flexothermophononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PHONON_POLARITON_RES_WORK`` (M399): Engine coupled flexothermal-flexophononic-flexopolaritonic nanoscale surface phonon-polariton hybrid resonance energy and thermal-gradient photonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpp_ph, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpp_ph = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpp_ph = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermophononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermophononicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermophononicpolaritonic_resonance_energies[r_id] = EngFlexothermophononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpp_ph=dt_ftpp_ph, sens_id=sens_id
    )




def read_eng_flexothermoexcitonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_POLARITON_RES_WORK`` (M400): Engine coupled flexothermal-flexoexcitonic-flexopolaritonic nanoscale exciton-polariton hybrid resonance energy and thermal-gradient photonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftep, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftep = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftep = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoexcitonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoexcitonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoexcitonicpolaritonic_resonance_energies[r_id] = EngFlexothermoexcitonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftep=dt_ftep, sens_id=sens_id
    )




def read_eng_flexothermomagnonicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAGNON_POLARITON_RES_WORK`` (M401): Engine coupled flexothermal-flexomagnonic-flexopolaritonic nanoscale spin-wave magnon-polariton hybrid resonance energy and thermal-gradient optomagnetic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftmp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftmp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftmp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermomagnonicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermomagnonicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermomagnonicpolaritonic_resonance_energies[r_id] = EngFlexothermomagnonicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftmp=dt_ftmp, sens_id=sens_id
    )




def read_eng_flexothermoplasmonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_PHONON_POLARITON_RES_WORK`` (M402): Engine coupled flexothermal-flexoplasmonic-flexophononic-flexopolaritonic nanoscale surface plasmon-phonon-polariton hybrid resonance energy and thermal-gradient optoelectronic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftppp=dt_ftppp, sens_id=sens_id
    )




def read_eng_flexothermoexcitonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITON_PHONON_POLARITON_RES_WORK`` (M403): Engine coupled flexothermal-flexoexcitonic-flexophononic-flexopolaritonic nanoscale exciton-phonon-polariton hybrid resonance energy and thermal-gradient optomechanical dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftepp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftepp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftepp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoexcitonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies[r_id] = EngFlexothermoexcitonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftepp=dt_ftepp, sens_id=sens_id
    )




def read_eng_flexothermomagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_MAGNON_PHONON_POLARITON_RES_WORK`` (M404): Engine coupled flexothermal-flexomagnonic-flexophononic-flexopolaritonic nanoscale spin-wave magnon-phonon-polariton hybrid resonance energy and thermal-gradient spintronic-acousto-optic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermomagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies[r_id] = EngFlexothermomagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftmpp=dt_ftmpp, sens_id=sens_id
    )




def read_eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK`` (M405): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexophononic-flexopolaritonic nanoscale multi-quasiparticle plasmon-exciton-phonon-polariton hybrid resonance energy and thermal-gradient optoelectronic-quantum dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpepp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpepp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpepp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicexcitonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonicexcitonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpepp=dt_ftpepp, sens_id=sens_id
    )




def read_eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK`` (M406): Engine coupled flexothermal-flexoplasmonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale surface plasmon-magnon-phonon-polariton hybrid resonance energy and thermal-gradient spintronic-acousto-optic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpmpp=dt_ftpmpp, sens_id=sens_id
    )




def read_eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_EXCITONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK`` (M407): Engine coupled flexothermal-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale exciton-magnon-phonon-polariton hybrid resonance energy and thermal-gradient optoelectronic-spintronic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftempp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftempp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftempp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoexcitonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngFlexothermoexcitonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftempp=dt_ftempp, sens_id=sens_id
    )




def read_eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/FLEXOTHERM_PLASMONIC_EXCITONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK`` (M408): Engine coupled flexothermal-flexoplasmonic-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale 4-quasiparticle plasmon-exciton-magnon-phonon-polariton hybrid resonance energy and thermal-gradient spintronic-optoelectronic-photonic-polaritonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_ftpempp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_ftpempp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_ftpempp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngFlexothermoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngFlexothermoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_ftpempp=dt_ftpempp, sens_id=sens_id
    )




def read_eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_PLASMON_PHONON_POLARITON_RES_WORK`` (M409): Engine coupled electrothermal-flexoplasmonic-flexophononic-flexopolaritonic nanoscale surface plasmon-phonon-polariton hybrid resonance energy and multi-field dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexoplasmonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexoplasmonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfppp=dt_etfppp, sens_id=sens_id
    )




def read_eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_EXCITON_PHONON_POLARITON_RES_WORK`` (M410): Engine coupled electrothermal-flexoexcitonic-flexophononic-flexopolaritonic nanoscale exciton-phonon-polariton hybrid resonance energy and optomechanical dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfepp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfepp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfepp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexoexcitonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexoexcitonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfepp=dt_etfepp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAGNON_PHONON_POLARITON_RES_WORK`` (M411): Engine coupled electrothermal-flexomagnonic-flexophononic-flexopolaritonic nanoscale magnon-phonon-polariton hybrid resonance energy and spintronic-acousto-optic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmpp=dt_etfmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK`` (M412): Engine coupled electrothermal-flexoplasmonic-flexoexcitonic-flexophononic-flexopolaritonic nanoscale plasmon-exciton-phonon-polariton hybrid resonance energy and opto-electro-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfpepp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfpepp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfpepp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexoplasmonicexcitonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexoplasmonicexcitonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfpepp=dt_etfpepp, sens_id=sens_id
    )




def read_eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_PLASMON_MAGNON_PHONON_POLARITON_RES_WORK`` (M413): Engine coupled electrothermal-flexoplasmonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-magnon-phonon-polariton hybrid resonance energy and spintronic-plasmonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfpmpp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfpmpp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfpmpp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexoplasmonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexoplasmonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfpmpp=dt_etfpmpp, sens_id=sens_id
    )




def read_eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_PLASMON_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK`` (M414): Engine coupled electrothermal-flexoplasmonic-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-exciton-magnon-phonon-polariton quintuple-hybrid resonance energy and multi-field opto-spintronic-plasmonic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfpempp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfpempp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfpempp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfpempp=dt_etfpempp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_POLARITON_RES_WORK`` (M415): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexophononic-flexopolaritonic nanoscale plasmon-phonon-polariton hybrid resonance energy and multi-field opto-magneto-electro-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmppp=dt_etfmppp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PHONON_POLARITON_RES_WORK`` (M416): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexophononic-flexopolaritonic nanoscale exciton-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmeppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmeppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmeppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmeppp=dt_etfmeppp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_PHONON_POLARITON_RES_WORK`` (M417): Engine coupled electrothermal-flexomagnetic-flexomagnonic-flexophononic-flexopolaritonic nanoscale magnon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmmppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmmppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmmppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetomagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetomagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmmppp=dt_etfmmppp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK`` (M418): Engine coupled electrothermal-flexomagnetic-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale exciton-magnon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmemppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmemppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmemppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoexcitonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoexcitonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmemppp=dt_etfmemppp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK`` (M419): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexophononic-flexopolaritonic nanoscale plasmon-exciton-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmpeppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmpeppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmpeppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicexcitonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmpeppp=dt_etfmpeppp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_PHONON_POLARITON_RES_WORK`` (M420): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-magnon-phonon-polariton hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmpmppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmpmppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmpmppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmpmppp=dt_etfmpmppp, sens_id=sens_id
    )




def read_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY`` or ``/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK`` (M421): Engine coupled electrothermal-flexomagnetic-flexoplasmonic-flexoexcitonic-flexomagnonic-flexophononic-flexopolaritonic nanoscale plasmon-exciton-magnon-phonon-polariton heptuple-hybrid resonance energy and multi-field opto-spintronic-thermo-acoustic dissipation tracking output directive."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/{block.user_id}: missing data card", block.source)
        return

    dt_etfmpxmppp, sens_id = 0.0, 0
    if block.fixed and "," not in cards[0].raw:
        f = cards[0].cut("ENG_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY_1")
        dt_etfmpxmppp = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        sens_id = _ival(f[1], 0) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        dt_etfmpxmppp = float(toks[0].rstrip(',')) if len(toks) > 0 else 0.0
        sens_id = int(float(toks[1].rstrip(','))) if len(toks) > 1 else 0

    from ...model.entities import EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy
    r_id = block.user_id or (len(model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies) + 1)
    model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[r_id] = EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy(
        id=r_id, title=title, dt_etfmpxmppp=dt_etfmpxmppp, sens_id=sens_id
    )




def read_upbeam(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/UPBEAM`` or ``/UPBEAM/INT_BEAM/id`` (M198): Integrated beam cross-section update."""
    upbeam_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/UPBEAM/{upbeam_id}: missing data card", block.source)
        return
    grnd_id, i_updt, eps_max, npt_int = 0, 0, 0.0, 0
    if block.fixed:
        f = cards[0].cut("UPBEAM_1")
        grnd_id = _ival(f[0]) if len(f) > 0 else 0
        i_updt = _ival(f[1]) if len(f) > 1 else 0
        eps_max = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        npt_int = _ival(f[4]) if len(f) > 4 else 0
    else:
        toks = cards[0].tokens()
        if len(toks) > 0:
            grnd_id = _safe_int(toks[0])
        if len(toks) > 1:
            i_updt = _safe_int(toks[1])
        if len(toks) > 2:
            eps_max = _safe_float(toks[2])
        if len(toks) > 3:
            npt_int = _safe_int(toks[3])
    up = Upbeam(id=upbeam_id, title=title, grnd_id=grnd_id, i_updt=i_updt, eps_max=eps_max, npt_int=npt_int)
    model.upbeams[upbeam_id] = up
