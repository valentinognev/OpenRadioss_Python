# -*- coding: utf-8 -*-
"""
Geometry and coordinate systems - /KEYWORD blocks -> Model.

/NODE, /SURF, /BOX, /LINE, /GRID, /TRANSFORM, /FRAME, /SKEW and the
analytic solid primitives (sphere, cylinder, parallelogram).

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
# Mesh
# ============================================================================

def read_node(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/NODE`` — one card per node::

        node_ID   Xc   Yc   Zc

    Fixed dialect (cfg SETS/node.cfg ``%10d%20lg%20lg%20lg``): the
    coordinate columns may abut with no whitespace
    ('...2.01.11022303000000E-16') and a blank column means 0.0 — the
    card is cut at the column widths, not tokenized (M37).

    Fortran: starter/source/elements/reader/hm_read_node.F.
    """
    ids, xyz = [], []
    if block.fixed:
        for card in block.cards:
            f = card.cut("NODE")
            if not f[0]:
                log.error("/NODE card without a node id", card.source)
                continue
            ids.append(int(f[0]))
            xyz.append([_fval(s) for s in f[1:4]])
    else:
        for card in block.cards:
            t = card.tokens()
            if len(t) < 4:
                log.error(f"/NODE card needs 4 fields, got {len(t)}",
                          card.source)
                continue
            ids.append(int(t[0]))
            xyz.append([float(v) for v in card.floats()[1:4]])
    if ids:
        model.add_nodes(np.array(ids), np.array(xyz))




def read_box(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BOX/RECTA|CYLIN|SPHER/box_ID`` (M37 — the three real
    geometries; cfg BOX/recta|cylin|spher.cfg radioss110, Fortran
    rdbox.F)::

        RECTA:  card 1: title
                card 2: N1   N2   Iskew        (%10d x3; N1/N2 > 0 ->
                        corners taken from those NODES' positions)
                card 3: Xp1  Yp1  Zp1          (diagonal corner 1)
                card 4: Xp2  Yp2  Zp2          (diagonal corner 2)
        CYLIN:  card 2: Base_N  Dir_N  <blank>  Diameter
                card 3: Xp1 Yp1 Zp1            (axis base point)
                card 4: Xp2 Yp2 Zp2            (axis end point — the
                        cylinder is FINITE, capped at both points)
        SPHER:  card 2: N1  <blank>  Diameter
                card 3: Xp1 Yp1 Zp1            (center)

    Iskew is not ported (warned when set).  The port's historical
    compact RECTA dialect — title + 6 corner floats (one or two
    cards, no N-card) — is kept: it is what the M36 deck-writer's
    blank-N-card emission collapses to for non-fixed decks.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "RECTA"
    if kind in ("RECT", "RECTA"):
        kind = "RECTA"
    elif kind in ("CYL", "CYLIN"):
        kind = "CYLIN"
    elif kind in ("SPH", "SPHER"):
        kind = "SPHER"
    elif kind in ("BOX", "COMB"):
        kind = "BOX"
    if kind not in ("RECTA", "CYLIN", "SPHER", "BOX"):
        log.warning(f"/BOX/{kind} not ported (RECTA, CYLIN, SPHER, BOX "
                    f"supported)", block.source)
        return
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)

    if kind == "BOX":
        # /BOX/BOX/id or /BOX/COMB/id (M132): Box composed of other boxes
        if not cards:
            log.error(f"/BOX/BOX/{block.user_id}: missing data card", block.source)
            return
        nbox, nboxneg = 0, 0
        if block.fixed:
            f = cards[0].cut("BOX_BOX_1")
            nbox = _ival(f[0])
            nboxneg = _ival(f[1])
        else:
            toks = cards[0].tokens()
            nbox = int(float(toks[0])) if len(toks) > 0 else 0
            nboxneg = int(float(toks[1])) if len(toks) > 1 else 0

        box_ids = []
        idx = 1
        pos_read = 0
        while idx < len(cards) and pos_read < nbox:
            for tok in cards[idx].tokens():
                if pos_read < nbox:
                    box_ids.append(int(float(tok)))
                    pos_read += 1
            idx += 1

        neg_read = 0
        while idx < len(cards) and neg_read < nboxneg:
            for tok in cards[idx].tokens():
                if neg_read < nboxneg:
                    bid = int(float(tok))
                    box_ids.append(-abs(bid))
                    neg_read += 1
            idx += 1

        model.boxes[block.user_id] = Box(
            id=block.user_id, title=title, kind="BOX", box_ids=box_ids
        )
        return

    def _xyz(card):
        return np.array(_cut_floats(card, "XYZ20")[:3]) if block.fixed \
            else np.array(_floats(card, 3))

    if kind == "RECTA":
        # real 3-data-card layout vs the compact 6-float dialect: the
        # real N-card holds only integers (or is blank, kept in fixed
        # decks); compact corner cards always carry non-integer floats
        n1 = n2 = iskew = 0
        real = False
        if block.fixed:
            real = True
            if cards and not cards[0].is_blank:
                n1, n2, iskew = _cut_ints(cards[0], "BOX_RECTA_N")[:3]
            cards = cards[1:]
        elif len(cards) >= 3 and all(
                tok.lstrip("+-").isdigit() for tok in cards[0].tokens()):
            real = True
            t = cards[0].ints()
            n1 = t[0] if len(t) > 0 else 0
            n2 = t[1] if len(t) > 1 else 0
            iskew = t[2] if len(t) > 2 else 0
            cards = cards[1:]
        if real and (n1 or n2):
            if not (n1 and n2):
                log.error(f"/BOX/RECTA/{block.user_id}: corner nodes "
                          f"need BOTH N1 and N2", block.source)
                return
            model.boxes[block.user_id] = Box(
                id=block.user_id, title=title, kind="RECTA",
                node1=n1, node2=n2, iskew=iskew)
            return
        vals: List[float] = []
        for c in cards:
            if c.is_blank:
                continue
            vals.extend(_cut_floats(c, "XYZ20")[:3] if block.fixed
                        else c.floats())
        if len(vals) < 6:
            log.error(f"/BOX/RECTA/{block.user_id}: needs 6 coordinates",
                      block.source)
            return
        p1, p2 = np.array(vals[:3]), np.array(vals[3:6])
        model.boxes[block.user_id] = Box(
            id=block.user_id, corner_min=np.minimum(p1, p2),
            corner_max=np.maximum(p1, p2), title=title, kind="RECTA",
            iskew=iskew)
        return

    if len(cards) < (3 if kind == "CYLIN" else 2):
        log.error(f"/BOX/{kind}/{block.user_id}: missing geometry cards",
                  block.source)
        return
    if kind == "CYLIN":
        if block.fixed:
            f = cards[0].cut("BOX_CYLIN_N")
            n1, n2, diam = _ival(f[0]), _ival(f[1]), _fval(f[3])
        else:                    # free-format: Base_N Dir_N Diameter
            v = _floats(cards[0], 3)
            n1, n2, diam = int(v[0]), int(v[1]), v[2]
        p1, p2 = _xyz(cards[1]), _xyz(cards[2])
    else:                                     # SPHER
        if block.fixed:
            f = cards[0].cut("BOX_SPHER_N")
            n1, n2, diam = _ival(f[0]), 0, _fval(f[2])
        else:                    # free-format: N1 Diameter
            v = _floats(cards[0], 2)
            n1, n2, diam = int(v[0]), 0, v[1]
        p1, p2 = _xyz(cards[1]), None
    if diam <= 0.0:
        log.error(f"/BOX/{kind}/{block.user_id}: Diameter must be > 0",
                  block.source)
        return
    model.boxes[block.user_id] = Box(
        id=block.user_id, title=title, kind=kind, p1=p1, p2=p2,
        diameter=diam, node1=n1, node2=n2)




def read_surf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SURF/<subtype>/surf_ID``: title card, then

    * PART: part IDs (the Starter extracts the free outer faces / shell
      faces of those parts into segments). ``/SURF/PART/EXT`` (M37) is
      accepted as THE standard extraction: for solid parts the upstream
      EXT treatment (ssurftag.F) keeps exactly the faces not shared by
      another element of the tagged parts — the port's free-outer-face
      extraction computes the same set; shell parts contribute every
      element either way.  Other qualifiers (``ALL`` — all faces of all
      solids, an error upstream for plain /SURF/PART) stay warned.
    * SEG:  one segment per card, in either dialect —

        - port compact: ``n1 n2 n3 [n4]``  (3 ids = triangle),
        - REAL fixed format (``hm_read_surf.F`` 'SEG'):
          ``seg_ID n1 n2 n3 n4`` — 5 fields; the leading segment id is
          dropped, ``n4 = 0`` means a triangle (upstream: N4=0 -> N3).

      Cards with 4 ids are read as the compact quad ``n1..n4`` — a REAL
      triangle card that leaves N4 blank instead of writing 0 is
      ambiguous with it and would be misread (real writers, e.g. k2rad,
      write all 5 fields).
    * SURF (M37): surface-of-surfaces (hm_read_surfsurf.F) — the listed
      surfaces' segments concatenated, resolved by iterative fixpoint
      with cycle detection; a NEGATIVE id reverses the included
      segments' node order (the normal flips).
    * GRSHEL / GRSH3N (M37): every element of the /GRSHEL / /GRSH3N
      element group becomes a segment (hm_surfgr2 + surftage).
    """
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    sid = block.user_id
    if sid is None:
        for p in block.parts[1:]:
            try:
                sid = int(p)
                break
            except ValueError:
                pass
    if sid is None:
        sid = len(model.surfaces) + 1
    block.user_id = sid

    s = model.surfaces.setdefault(
        sid, Surface(id=sid, title=title))
    s.id = sid

    all_parts = [p.upper() for p in block.parts]
    modifier = ""
    for m in ("EXT", "ALL", "FREE"):
        if m in all_parts:
            modifier = m
            break
    s.modifier = modifier

    non_mods = [p for p in all_parts[1:] if p not in ("EXT", "ALL", "FREE") and not p.isdigit()]
    target = non_mods[0] if non_mods else "PART"

    if target == "PART":
        if modifier and modifier != "EXT":
            log.warning(f"/SURF/PART/{modifier}/{block.user_id}: the "
                        f"{modifier} qualifier is ignored — treated "
                        f"as plain /SURF/PART (the port extracts the free "
                        f"outer faces of the parts)", block.source)
        s.part_ids.extend(_id_list(block, cards))
    elif target == "SEG":
        for c in cards:
            if c.is_blank:
                continue
            t = [int(v) for v in c.cut("IDS10") if v] if block.fixed \
                else c.ints()
            if len(t) == 5:
                t = t[1:]                  # real dialect: drop seg_ID
            if len(t) == 3:
                t = t + [t[2]]
            if len(t) != 4:
                log.error(f"/SURF/SEG card needs 3 or 4 node ids, or "
                          f"seg_ID + 4 node ids (real format)", c.source)
                continue
            if t[3] == 0:
                t[3] = t[2]                # upstream: N4 = 0 -> triangle
            s.seg_nodes.append(t)
    elif target == "SURF":
        s.surf_ids.extend(_id_list(block, cards))
        from ...model.entities import SurfSurf
        ids = s.surf_ids
        surf1 = ids[0] if len(ids) > 0 else 0
        surf2 = ids[1] if len(ids) > 1 else 0
        model.surf_surfs[sid] = SurfSurf(
            id=sid, title=title, surf_ids=list(ids), surf1_id=surf1, surf2_id=surf2
        )
    elif target in ("GRSHEL", "GRSH3N", "GRTRIA", "GRBRIC"):
        fam = _GR_FAMILIES.get(target, target[2:])
        s.egroup_refs.extend((fam, i) for i in _id_list(block, cards))
    elif target == "PLANE":
        if len(cards) < 2:
            log.error(f"/SURF/PLANE/{block.user_id}: requires 2 data cards (P1, P2)", block.source)
        else:
            p1 = _cut_floats(cards[0], "SURF_PLANE") if block.fixed else _floats(cards[0], 3)
            p2 = _cut_floats(cards[1], "SURF_PLANE") if block.fixed else _floats(cards[1], 3)
            v = np.array(p2[:3]) - np.array(p1[:3])
            if np.linalg.norm(v) <= 1e-10:
                log.error(f"/SURF/PLANE/{block.user_id}: plane points P1 and P2 are identical (zero normal)", block.source)
            s.plane_p1 = np.array(p1[:3], dtype=float)
            s.plane_p2 = np.array(p2[:3], dtype=float)
    elif target == "MAT":
        s.mat_ids.extend(_id_list(block, cards))
    elif target == "PROP":
        s.prop_ids.extend(_id_list(block, cards))
    elif target == "BOX":
        s.box_ids.extend(_id_list(block, cards))
    elif target in ("ELLIPSE", "ELLIPSOID"):
        # /SURF/ELLIPSE (M132):
        # Card 1: Skew_ID, n
        # Card 2: Xc, Yc, Zc
        # Card 3: a, b, c (semi-axes)
        if len(cards) < 3:
            log.error(f"/SURF/ELLIPSE/{block.user_id}: requires 3 data cards (Skew/n, Center, Semiaxes)", block.source)
        else:
            if block.fixed:
                f1 = cards[0].cut("SURF_ELLIPSE_1")
                skew_id = _ival(f1[0]) if len(f1) > 0 else 0
                f2 = cards[1].cut("SURF_ELLIPSE_2")
                xc = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                yc = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                zc = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
                f3 = cards[2].cut("SURF_ELLIPSE_3")
                sa = _fval(f3[0], 1.0) if len(f3) > 0 else 1.0
                sb = _fval(f3[1], 1.0) if len(f3) > 1 else 1.0
                sc = _fval(f3[2], 1.0) if len(f3) > 2 else 1.0
            else:
                t1 = cards[0].tokens()
                skew_id = int(float(t1[0])) if len(t1) > 0 else 0
                t2 = cards[1].tokens()
                xc = float(t2[0]) if len(t2) > 0 else 0.0
                yc = float(t2[1]) if len(t2) > 1 else 0.0
                zc = float(t2[2]) if len(t2) > 2 else 0.0
                t3 = cards[2].tokens()
                sa = float(t3[0]) if len(t3) > 0 else 1.0
                sb = float(t3[1]) if len(t3) > 1 else 1.0
                sc = float(t3[2]) if len(t3) > 2 else 1.0

            s.ellipse_skew = skew_id
            s.ellipse_center = np.array([xc, yc, zc], dtype=float)
            s.ellipse_semiaxes = np.array([sa, sb, sc], dtype=float)
    elif target in ("CYL", "CYLIND"):
        # /SURF/CYL (M134):
        # Card 1: Skew_ID, Radius, Length
        # Card 2: X0, Y0, Z0
        # Card 3: Ax, Ay, Az
        if len(cards) < 3:
            log.error(f"/SURF/CYL/{block.user_id}: requires 3 data cards (Skew/R/L, Origin, Axis)", block.source)
        else:
            if block.fixed:
                f1 = cards[0].cut("SURF_CYL_1")
                rad = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                leng = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                f2 = cards[1].cut("SURF_CYL_2")
                x0 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                y0 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                z0 = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
                f3 = cards[2].cut("SURF_CYL_3")
                ax = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
                ay = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
                az = _fval(f3[2], 1.0) if len(f3) > 2 else 1.0
            else:
                t1 = cards[0].tokens()
                rad = float(t1[1]) if len(t1) > 1 else 0.0
                leng = float(t1[2]) if len(t1) > 2 else 0.0
                t2 = cards[1].tokens()
                x0 = float(t2[0]) if len(t2) > 0 else 0.0
                y0 = float(t2[1]) if len(t2) > 1 else 0.0
                z0 = float(t2[2]) if len(t2) > 2 else 0.0
                t3 = cards[2].tokens()
                ax = float(t3[0]) if len(t3) > 0 else 0.0
                ay = float(t3[1]) if len(t3) > 1 else 0.0
                az = float(t3[2]) if len(t3) > 2 else 1.0
            s.cyl_radius = rad
            s.cyl_length = leng
            s.cyl_center = np.array([x0, y0, z0], dtype=float)
            s.cyl_axis = np.array([ax, ay, az], dtype=float)
    elif target in ("SPHER", "SPHERE"):
        # /SURF/SPHER (M134):
        # Card 1: Skew_ID, Radius
        # Card 2: Xc, Yc, Zc
        if len(cards) < 2:
            log.error(f"/SURF/SPHER/{block.user_id}: requires 2 data cards (Skew/R, Center)", block.source)
        else:
            if block.fixed:
                f1 = cards[0].cut("SURF_SPHER_1")
                rad = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                f2 = cards[1].cut("SURF_SPHER_2")
                xc = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                yc = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                zc = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            else:
                t1 = cards[0].tokens()
                rad = float(t1[1]) if len(t1) > 1 else 0.0
                t2 = cards[1].tokens()
                xc = float(t2[0]) if len(t2) > 0 else 0.0
                yc = float(t2[1]) if len(t2) > 1 else 0.0
                zc = float(t2[2]) if len(t2) > 2 else 0.0
            s.spher_radius = rad
            s.spher_center = np.array([xc, yc, zc], dtype=float)
    elif target in ("SUB", "SUBSET"):
        sub_ids = _id_list(block, cards)
        s.subset_surf_ids.extend(sub_ids)
        s.surf_ids.extend(sub_ids)
    elif target in ("ALL", "EXT", "FREE"):
        s.modifier = target
        if cards and not cards[0].is_blank:
            s.part_ids.extend(_id_list(block, cards))
    else:
        log.warning(f"/SURF/{'/'.join(all_parts[1:])} not ported", block.source)




def read_skew(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SKEW/FIX``, ``/SKEW/MOV``, ``/SKEW/MOV2`` (M39) — see
    :func:`_read_reference_system`."""
    _read_reference_system(block, model, log, "SKEW")




def read_frame(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FRAME/FIX``, ``/FRAME/MOV``, ``/FRAME/MOV2``, ``/FRAME/NOD``
    (M39) — see :func:`_read_reference_system`.

    A /FRAME is built exactly like the matching /SKEW; what makes it a
    *reference* frame is the Engine's moving-frame formulation
    (``engine/source/tools/skew/movfram.F`` MOVFRA1/MOVFRA2, which also
    track the frame's velocity/acceleration so the relative-frame inertia
    terms of ``relfram.F`` can be added).  The port builds and uses the
    frame's GEOMETRY (the corpus's only frame consumer, /INIVEL/AXIS, is a
    Starter-time initial condition that needs the initial orientation and
    origin); a consumer that would need the moving-frame ENGINE update
    warns loudly where it is wired.
    """
    _read_reference_system(block, model, log, "FRAME")




def read_surf_surf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SURF_SURF/ID`` — Surface-to-surface interaction."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or (block.fixed and cards[0].is_blank):
        log.error(f"/SURF_SURF/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f1 = cards[0].cut("SURF_SURF_1") if "SURF_SURF_1" in CARD_LAYOUTS else _fixed_vals(cards[0], [10, 10, 10])
        surf1 = _ival(f1[0]) if len(f1) > 0 else 0
        surf2 = _ival(f1[1]) if len(f1) > 1 else 0
        iflag = _ival(f1[2]) if len(f1) > 2 else 0
        gap, fric = 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SURF_SURF_2") if "SURF_SURF_2" in CARD_LAYOUTS else _fixed_vals(cards[1], [20, 20])
            gap = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            fric = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
    else:
        t = cards[0].tokens()
        surf1 = int(float(t[0])) if len(t) > 0 else 0
        surf2 = int(float(t[1])) if len(t) > 1 else 0
        iflag = int(float(t[2])) if len(t) > 2 else 0
        gap, fric = 0.0, 0.0
        if len(cards) > 1:
            t2 = cards[1].tokens()
            gap = float(t2[0]) if len(t2) > 0 else 0.0
            fric = float(t2[1]) if len(t2) > 1 else 0.0
    from ...model.entities import SurfSurf
    model.surf_surfs[block.user_id] = SurfSurf(
        id=block.user_id, title=title, surf1_id=surf1, surf2_id=surf2,
        iflag=iflag, gap=gap, fric=fric
    )




def read_ale_grid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/GRID/...`` (M63, M105, M113): ALE grid formulation and damping controls."""
    if len(block.parts) > 1 and block.parts[1].upper() == "GRID":
        if len(block.parts) > 2 and not block.parts[2].isdigit():
            grid_sub = block.parts[2].upper()
        else:
            grid_sub = "STANDARD"
    elif len(block.parts) > 1 and not block.parts[1].isdigit():
        grid_sub = block.parts[1].upper()
    else:
        grid_sub = "STANDARD"
    gid = block.user_id if block.user_id is not None else 1
    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    title = ""
    dt_min, gamma, damp, nu_g = 0.0, 0.0, 0.0, 0.0
    from ...model.entities import (
        AleGridDonea, AleGridSpring, AleGridStandard, AleGridDisp,
        AleGridLaplacian, AleGridVolume
    )

    if grid_sub in ("DONE", "DONEA"):
        alpha, gamma, vx, vy, vz = 0.0, 100.0, 1.0, 1.0, 1.0
        vmin = -1e30
        if cards:
            if block.fixed:
                f1 = cards[0].cut("ALE_GRID_DONEA_1")
                alpha = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
                gamma = _fval(f1[1], 100.0) if len(f1) > 1 else 100.0
                vx = _fval(f1[2], 1.0) if len(f1) > 2 else 1.0
                vy = _fval(f1[3], 1.0) if len(f1) > 3 else 1.0
                vz = _fval(f1[4], 1.0) if len(f1) > 4 else 1.0
                if len(cards) > 1:
                    f2 = cards[1].cut("ALE_GRID_DONEA_2")
                    vmin = _fval(f2[0], -1e30) if len(f2) > 0 else -1e30
            else:
                t1 = cards[0].tokens()
                alpha = float(t1[0]) if len(t1) > 0 else 0.0
                gamma = float(t1[1]) if len(t1) > 1 else 100.0
                vx = float(t1[2]) if len(t1) > 2 else 1.0
                vy = float(t1[3]) if len(t1) > 3 else 1.0
                vz = float(t1[4]) if len(t1) > 4 else 1.0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    vmin = float(t2[0]) if len(t2) > 0 else -1e30
        model.ale_grid_donea = AleGridDonea(alpha=alpha, gamma=gamma, vel_x=vx, vel_y=vy, vel_z=vz, v_min=vmin)
    elif grid_sub == "SPRING":
        dt, gamma, damp, nu = 0.0, 0.0, 0.5, 1.0
        vmin = -1e30
        if cards:
            if block.fixed:
                f1 = cards[0].cut("ALE_GRID_SPRING_1")
                dt = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
                gamma = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                damp = _fval(f1[2], 0.5) if len(f1) > 2 else 0.5
                nu = _fval(f1[3], 1.0) if len(f1) > 3 else 1.0
                if len(cards) > 1:
                    f2 = cards[1].cut("ALE_GRID_SPRING_2")
                    vmin = _fval(f2[0], -1e30) if len(f2) > 0 else -1e30
            else:
                t1 = cards[0].tokens()
                dt = float(t1[0]) if len(t1) > 0 else 0.0
                gamma = float(t1[1]) if len(t1) > 1 else 0.0
                damp = float(t1[2]) if len(t1) > 2 else 0.5
                nu = float(t1[3]) if len(t1) > 3 else 1.0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    vmin = float(t2[0]) if len(t2) > 0 else -1e30
        model.ale_grid_spring = AleGridSpring(dt=dt, gamma=gamma, damp=damp, nu=nu, v_min=vmin)
    elif grid_sub in ("DISP", "VEL"):
        from ...model.entities import AleGridConstraint
        grnod_id, fun_id, skew_id = 0, 0, 0
        tra_code = ""
        scale, tstart, tstop = 1.0, 0.0, 1.0e30
        is_nodal_constraint = False
        if block.fixed:
            if cards and not cards[0].is_blank:
                f1 = cards[0].cut("ALE_GRID_1")
                if len(f1) >= 3 and _ival(f1[0]) > 0:
                    is_nodal_constraint = True
                    grnod_id = _ival(f1[0])
                    fun_id = _ival(f1[1])
                    skew_id = _ival(f1[2])
                    tra_code = f1[3].strip() if len(f1) > 3 else ""
                    if len(cards) > 1 and not cards[1].is_blank:
                        f2 = cards[1].cut("ALE_GRID_2")
                        scale = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
                        tstart = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                        tstop = _fval(f2[2], 1.0e30) if len(f2) > 2 and _fval(f2[2]) > 0.0 else 1.0e30
        else:
            t1 = cards[0].tokens() if cards else []
            if len(t1) >= 3:
                is_nodal_constraint = True
                grnod_id = int(float(t1[0]))
                fun_id = int(float(t1[1]))
                skew_id = int(float(t1[2]))
                tra_code = str(t1[3]) if len(t1) > 3 else ""
                if len(cards) > 1 and not cards[1].is_blank:
                    t2 = cards[1].tokens()
                    scale = float(t2[0]) if len(t2) > 0 else 1.0
                    tstart = float(t2[1]) if len(t2) > 1 else 0.0
                    tstop = float(t2[2]) if len(t2) > 2 and float(t2[2]) > 0.0 else 1.0e30
        if is_nodal_constraint:
            model.ale_grid_constraints[gid] = AleGridConstraint(
                id=gid, kind=grid_sub, title="", grnod_id=grnod_id, fun_id=fun_id,
                skew_id=skew_id, tra_code=tra_code, scale=scale, tstart=tstart, tstop=tstop,
            )
        else:
            umax, vmin = -1e30, -1e30
            if cards:
                if block.fixed:
                    f1 = cards[0].cut("ALE_GRID_DISP_1")
                    umax = _fval(f1[0], -1e30) if len(f1) > 0 else -1e30
                    if len(cards) > 1:
                        f2 = cards[1].cut("ALE_GRID_DISP_2")
                        vmin = _fval(f2[0], -1e30) if len(f2) > 0 else -1e30
                else:
                    t1 = cards[0].tokens()
                    umax = float(t1[0]) if len(t1) > 0 else -1e30
                    if len(cards) > 1:
                        t2 = cards[1].tokens()
                        vmin = float(t2[0]) if len(t2) > 0 else -1e30
            model.ale_grid_disp = AleGridDisp(u_max=umax, v_min=vmin)
    elif grid_sub == "LAPLACIAN":
        alpha, gamma, damp = 0.0, 0.0, 0.5
        if cards:
            if block.fixed:
                f = cards[0].cut("ALE_GRID_LAPLACIAN_1")
                alpha = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                gamma = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                damp = _fval(f[2], 0.5) if len(f) > 2 else 0.5
            else:
                t = cards[0].tokens()
                alpha = float(t[0]) if len(t) > 0 else 0.0
                gamma = float(t[1]) if len(t) > 1 else 0.0
                damp = float(t[2]) if len(t) > 2 else 0.5
        model.ale_grid_laplacian = AleGridLaplacian(alpha=alpha, gamma=gamma, damp=damp)
    elif grid_sub == "VOLUME":
        alpha, gamma = 0.0, 0.0
        if cards:
            if block.fixed:
                f = cards[0].cut("ALE_GRID_VOLUME_1")
                alpha = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                gamma = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = cards[0].tokens()
                alpha = float(t[0]) if len(t) > 0 else 0.0
                gamma = float(t[1]) if len(t) > 1 else 0.0
        model.ale_grid_volume = AleGridVolume(alpha=alpha, gamma=gamma)
    elif grid_sub in ("FLOW-TRACKING", "FLOW_TRACKING", "MASS-WEIGHTED-VEL", "MASS_WEIGHTED_VEL"):
        is_def, is_rot = 0, 0
        scale_def, scale_rot = 1.0, 1.0
        if cards:
            if block.fixed:
                f1 = cards[0].cut("ALE_GRID_FLOW_TRACK")
                is_def = _ival(f1[0]) if len(f1) > 0 else 0
                scale_def = _fval(f1[1], 1.0) if len(f1) > 1 else 1.0
                if len(cards) > 1:
                    f2 = cards[1].cut("ALE_GRID_FLOW_TRACK")
                    is_rot = _ival(f2[0]) if len(f2) > 0 else 0
                    scale_rot = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
            else:
                t1 = cards[0].tokens()
                is_def = int(float(t1[0])) if len(t1) > 0 else 0
                scale_def = float(t1[1]) if len(t1) > 1 else 1.0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    is_rot = int(float(t2[0])) if len(t2) > 0 else 0
                    scale_rot = float(t2[1]) if len(t2) > 1 else 1.0
        model.ale_grid_flow_tracking = {
            "is_def": is_def, "scale_def": scale_def,
            "is_rot": is_rot, "scale_rot": scale_rot,
        }
    elif grid_sub == "LAGRANGE":
        model.ale_grid_lagrange = True
    else:  # STANDARD
        alpha, gamma, damp, lc = 0.0, 0.0, 0.5, 1.0
        if cards:
            if block.fixed:
                f = cards[0].cut("ALE_GRID_STANDARD_1")
                alpha = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                gamma = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                damp = _fval(f[2], 0.5) if len(f) > 2 else 0.5
                lc = _fval(f[3], 1.0) if len(f) > 3 else 1.0
            else:
                t = cards[0].tokens()
                alpha = float(t[0]) if len(t) > 0 else 0.0
                gamma = float(t[1]) if len(t) > 1 else 0.0
                damp = float(t[2]) if len(t) > 2 else 0.5
                lc = float(t[3]) if len(t) > 3 else 1.0
        model.ale_grid_standard = AleGridStandard(alpha=alpha, gamma=gamma, damp=damp, l_c=lc)
        dt_min, nu_g = alpha, lc

    model.ale_grids[gid] = AleGrid(
        id=gid, subtype=grid_sub, dt_min=dt_min,
        gamma=gamma, damp=damp, nu_g=nu_g,
    )




def read_line(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LINE/<subtype>/line_ID`` — edge sets for /INTER/TYPE11
    (Fortran: hm_read_lines.F → IGRSLIN)::

        /LINE/SURF: card 1 = title, card 2+ = surf_IDs (any number/card)
                    → every unique edge of those surfaces' segments
        /LINE/EDGE: like SURF, but only the BORDER edges — edges used by
                    exactly ONE segment of the listed surfaces; interior
                    (shared) edges are removed entirely (linedge.F
                    'REMOVAL OF INTERNAL SEGMENTS EXCEPT BORDERS' — M37)
        /LINE/LINE: line-of-lines — the listed lines' edges concatenated
                    (hm_lines_of_lines.F, fixpoint + cycle detection)
        /LINE/PART: every 1-D element (truss/beam/spring) of the parts
                    becomes an edge (M37)
        /LINE/SEG:  card 1 = title, card 2+ = node_ID1 node_ID2 per card
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "SURF"
    if kind not in ("SURF", "SEG", "EDGE", "LINE", "PART", "BEAM", "TRUSS", "SPRING", "BOX", "CYL", "SPH", "CIRC", "ALL"):
        log.warning(f"/LINE/{kind} not ported (SURF, EDGE, LINE, PART, BEAM, TRUSS, SPRING, BOX, CIRC, ALL, "
                    f"SEG supported)", block.source)
        return
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    line = model.lines.setdefault(block.user_id,
                                  Line(id=block.user_id, title=title))
    if kind == "SURF":
        line.surf_ids.extend(_id_list(block, cards))
    elif kind == "EDGE":
        line.edge_surf_ids.extend(_id_list(block, cards))
    elif kind == "LINE":
        line.line_ids.extend(_id_list(block, cards))
    elif kind == "PART":
        line.part_ids.extend(_id_list(block, cards))
    elif kind == "BEAM":
        line.beam_ids.extend(_id_list(block, cards))
    elif kind == "TRUSS":
        line.truss_ids.extend(_id_list(block, cards))
    elif kind == "SPRING":
        line.spring_ids.extend(_id_list(block, cards))
    elif kind in ("BOX", "CYL", "SPH"):
        line.box_ids.extend(_id_list(block, cards))
    elif kind == "CIRC":
        # /LINE/CIRC (M134):
        # Card 1: Xc, Yc, Zc, Radius
        # Card 2: Nx, Ny, Nz
        if len(cards) < 2:
            log.error(f"/LINE/CIRC/{block.user_id}: requires 2 data cards (Center/Radius, Normal)", block.source)
        else:
            if block.fixed:
                f1 = cards[0].cut("LINE_CIRC_1")
                xc = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
                yc = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                zc = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                rad = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                f2 = cards[1].cut("LINE_CIRC_2")
                nx = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                ny = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                nz = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            else:
                t1 = cards[0].tokens()
                xc = float(t1[0]) if len(t1) > 0 else 0.0
                yc = float(t1[1]) if len(t1) > 1 else 0.0
                zc = float(t1[2]) if len(t1) > 2 else 0.0
                rad = float(t1[3]) if len(t1) > 3 else 0.0
                t2 = cards[1].tokens()
                nx = float(t2[0]) if len(t2) > 0 else 0.0
                ny = float(t2[1]) if len(t2) > 1 else 0.0
                nz = float(t2[2]) if len(t2) > 2 else 1.0
            line.circ_center = np.array([xc, yc, zc], dtype=float)
            line.circ_radius = rad
            line.circ_axis = np.array([nx, ny, nz], dtype=float)
    elif kind == "ALL":
        line.all_boundary = True
    else:
        for card in cards:
            if card.is_blank:
                continue
            t = [int(v) for v in card.cut("IDS10") if v] if block.fixed \
                else card.ints()
            if len(t) >= 3:
                t = t[1:]      # real dialect: 'seg_ID N1 N2' — drop the id
            if len(t) < 2:
                log.error(f"/LINE/SEG/{block.user_id}: a segment needs 2 "
                          f"node ids", card.source)
                continue
            line.seg_nodes.append(t[:2])




def read_cyl_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CYL_JOINT/id`` (M102, M209): cylindrical joint definition.

    Fortran origin: ``starter/source/constraints/general/cyl_joint/hm_read_cyljoint.F``.
    Card 1: TITLE (%-100s)
    Card 2: node_id1, node_id2, grnod_id (%10d%10d%10d) or node1, node2, axis_dir, skew_id, tol
    Card 3...: optional list of secondary nodes
    """
    from ...model.entities import CylJoint
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        model.cyl_joints[block.user_id] = CylJoint(id=block.user_id, title=title)
        return
    c = cards[0]
    axis_dir = 1
    skew_id = 0
    tol = 1e-6
    if block.fixed:
        f = c.cut("CYL_JOINT_1")
        n1 = _ival(f[0]) if len(f) > 0 else 0
        n2 = _ival(f[1]) if len(f) > 1 else 0
        gr = _ival(f[2]) if len(f) > 2 else 0
        axis_dir = _ival(f[2], 1) if len(f) > 2 else 1
        skew_id = _ival(f[3], 0) if len(f) > 3 else 0
        tol = _fval(f[4], 1e-6) if len(f) > 4 else 1e-6
    else:
        toks = c.tokens()
        n1 = int(float(toks[0])) if len(toks) > 0 else 0
        n2 = int(float(toks[1])) if len(toks) > 1 else 0
        gr = int(float(toks[2])) if len(toks) > 2 else 0
        axis_dir = int(float(toks[2])) if len(toks) > 2 else 1
        skew_id = int(float(toks[3])) if len(toks) > 3 else 0
        tol = float(toks[4]) if len(toks) > 4 else 1e-6
    secondary_nodes = []
    if len(cards) > 1:
        for card in cards[1:]:
            for tok in card.tokens():
                try:
                    nid = int(float(tok))
                    if nid > 0:
                        secondary_nodes.append(nid)
                except ValueError:
                    pass
    model.cyl_joints[block.user_id] = CylJoint(
        id=block.user_id, title=title, node_id1=n1, node_id2=n2, grnod_id=gr,
        node1=n1, node2=n2, axis_dir=axis_dir, skew_id=skew_id, tol=tol,
        secondary_nodes=secondary_nodes
    )







def read_transform(block: KeywordBlock, model: Model,
                   log: MessageLog) -> None:
    """`/TRANSFORM/{TRA|ROT|SYM|SCA|POS|POSITION}/transform_id` (M63, M85, M99) — Mesh transformations.

    Fortran origin: ``starter/source/model/transformation/lectrans.F``.
    Card format:
    * `/TRANSFORM/TRA`:
        card 1: title
        card 2: GR_NODE  TX  TY  TZ  node_ID1  node_ID2  sub_ID
        card 3 (optional): skew_ID
    * `/TRANSFORM/ROT`:
        card 1: title
        card 2: GR_NODE  X_p1  Y_p1  Z_p1  node_ID1  node_ID2  sub_ID
        card 3:          X_p2  Y_p2  Z_p2  Angle
    * `/TRANSFORM/SYM`:
        card 1: title
        card 2: GR_NODE  X_p1  Y_p1  Z_p1  node_ID1  node_ID2  sub_ID
        card 3:          X_p2  Y_p2  Z_p2
    * `/TRANSFORM/SCA`:
        card 1: title
        card 2: GR_NODE  Fscale_X  Fscale_Y  Fscale_Z  node_IDc  sub_ID
    * `/TRANSFORM/POS` or `/TRANSFORM/POSITION`:
        card 1: title
        card 2: GR_NODE  n1  n2  n3  n4  n5  n6  (blank)  (blank)  sub_ID
        cards 3..8 (optional): (blank)  X  Y  Z  for points 1..6
    * `/TRANSFORM/AUTOPOSITION` (M111):
        card 1: title
        card 2: GR_NODE  Surf_ID  skew_ID  Dir  Gap  Pflag
        card 3: Xpos  Ypos  Zpos  Xflag  Yflag  Zflag
    """
    if block.key0 in ("AUTOPOSITION", "AUTOPOS"):
        sub = "AUTOPOSITION"
    elif block.key0 in ("ROT", "ROTATE", "ROTATION"):
        sub = "ROT"
    elif block.key0 in ("TRA", "TRANSL", "TRANSLATION"):
        sub = "TRA"
    elif block.key0 in ("SCA", "SCALE"):
        sub = "SCA"
    elif block.key0 in ("SYM", "SYMET", "MIRROR", "PLANE"):
        sub = "SYM"
    elif block.key0 in ("MATRIX", "MATR"):
        sub = "MATRIX"
    elif block.key0 in ("POS", "POSITION"):
        sub = "POS"
    elif len(block.parts) > 1:
        sub = block.parts[1].upper()
    else:
        sub = ""

    # Normalization of aliases (M138, M199)
    if sub in ("SYMET", "MIRROR", "PLANE"):
        sub = "SYM"
    elif sub in ("SCALE",):
        sub = "SCA"
    elif sub in ("TRANSL", "TRANSLATION"):
        sub = "TRA"
    elif sub in ("ROTATE", "ROTATION"):
        sub = "ROT"
    elif sub in ("MATR",):
        sub = "MATRIX"

    if sub not in ("TRA", "ROT", "SYM", "SCA", "POS", "POSITION", "AUTOPOSITION", "AUTOPOS", "PROJ", "PROJECTION", "FRAME", "MATRIX"):
        log.warning(f"/TRANSFORM/{sub} not ported — block skipped "
                    f"(supported: TRA, ROT, SYM, SCA, POS, AUTOPOSITION, PROJ, FRAME, MATRIX)", block.source)
        return


    if block.fixed:
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/TRANSFORM/{sub}/{block.user_id}: missing data card",
                  block.source)
        return

    if sub in ("AUTOPOSITION", "AUTOPOS"):
        from ...model.entities import Autoposition
        if block.fixed:
            f1 = cards[0].cut("AUTOPOSITION_1")
            grnod_id = _ival(f1[0]) if len(f1) > 0 else 0
            surf_id = _ival(f1[1]) if len(f1) > 1 else 0
            skew_id = _ival(f1[2]) if len(f1) > 2 else 0
            dir_str = f1[3].strip() if len(f1) > 3 else ""
            gap = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
            pflag = _ival(f1[5], 0) if len(f1) > 5 else 0

            xpos, ypos, zpos, xflag, yflag, zflag = 0.0, 0.0, 0.0, 0, 0, 0
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("AUTOPOSITION_2")
                xpos = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                ypos = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                zpos = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
                xflag = _ival(f2[3], 0) if len(f2) > 3 else 0
                yflag = _ival(f2[4], 0) if len(f2) > 4 else 0
                zflag = _ival(f2[5], 0) if len(f2) > 5 else 0
        else:
            t1 = cards[0].tokens()
            grnod_id = int(float(t1[0])) if len(t1) > 0 else 0
            surf_id = int(float(t1[1])) if len(t1) > 1 else 0
            skew_id = int(float(t1[2])) if len(t1) > 2 else 0
            dir_str = t1[3] if len(t1) > 3 else ""
            gap = float(t1[4]) if len(t1) > 4 else 0.0
            pflag = int(float(t1[5])) if len(t1) > 5 else 0

            xpos, ypos, zpos, xflag, yflag, zflag = 0.0, 0.0, 0.0, 0, 0, 0
            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                xpos = float(t2[0]) if len(t2) > 0 else 0.0
                ypos = float(t2[1]) if len(t2) > 1 else 0.0
                zpos = float(t2[2]) if len(t2) > 2 else 0.0
                xflag = int(float(t2[3])) if len(t2) > 3 else 0
                yflag = int(float(t2[4])) if len(t2) > 4 else 0
                zflag = int(float(t2[5])) if len(t2) > 5 else 0

        model.autopositions.append(Autoposition(
            id=block.user_id, title=title, grnod_id=grnod_id, surf_id=surf_id,
            skew_id=skew_id, dir=dir_str, gap=gap, pflag=pflag,
            xpos=xpos, ypos=ypos, zpos=zpos, xflag=xflag, yflag=yflag, zflag=zflag
        ))
        return

    if sub in ("PROJ", "PROJECTION"):
        # /TRANSFORM/PROJ (M134):
        # Card 1: GRNOD_ID, Proj_type, Target_ID, Dist
        # Card 2: Dir_X, Dir_Y, Dir_Z
        if block.fixed:
            f1 = cards[0].cut("TRANSFORM_PROJ_1")
            grnod_id = _ival(f1[0]) if len(f1) > 0 else 0
            proj_type = f1[1].strip() if len(f1) > 1 and f1[1].strip() else "PLANE"
            target_id = _ival(f1[2]) if len(f1) > 2 else 0
            dist = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
            dir_x, dir_y, dir_z = 0.0, 0.0, 1.0
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("TRANSFORM_PROJ_2")
                dir_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                dir_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                dir_z = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
        else:
            t1 = cards[0].tokens()
            grnod_id = int(float(t1[0])) if len(t1) > 0 else 0
            proj_type = t1[1].upper() if len(t1) > 1 else "PLANE"
            target_id = int(float(t1[2])) if len(t1) > 2 else 0
            dist = float(t1[3]) if len(t1) > 3 else 0.0
            dir_x, dir_y, dir_z = 0.0, 0.0, 1.0
            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                dir_x = float(t2[0]) if len(t2) > 0 else 0.0
                dir_y = float(t2[1]) if len(t2) > 1 else 0.0
                dir_z = float(t2[2]) if len(t2) > 2 else 1.0

        model.transform_projections.append(TransformProjection(
            id=block.user_id, title=title, grnod_id=grnod_id,
            proj_type=proj_type, target_id=target_id,
            dir_vector=(dir_x, dir_y, dir_z), dist=dist
        ))
        return

    if sub == "FRAME":
        # /TRANSFORM/FRAME (M134):
        # Card 1: GRNOD_ID, Frame_orig, Frame_dest
        if block.fixed:
            f = cards[0].cut("TRANSFORM_FRAME_1")
            grnod_id = _ival(f[0]) if len(f) > 0 else 0
            frame_orig = _ival(f[1]) if len(f) > 1 else 0
            frame_dest = _ival(f[2]) if len(f) > 2 else 0
        else:
            t = cards[0].tokens()
            grnod_id = int(float(t[0])) if len(t) > 0 else 0
            frame_orig = int(float(t[1])) if len(t) > 1 else 0
            frame_dest = int(float(t[2])) if len(t) > 2 else 0

        model.transform_frames.append(TransformFrame(
            id=block.user_id, title=title, grnod_id=grnod_id,
            frame_orig=frame_orig, frame_dest=frame_dest
        ))
        return

    if not hasattr(model, 'transforms'):
        model.transforms = []

    if sub == "TRA":
        if block.fixed:
            f = cards[0].cut("TRANSFORM_TRA")
            grnod = _ival(f[0]) if len(f) > 0 else 0
            tx = _fval(f[1]) if len(f) > 1 else 0.0
            ty = _fval(f[2]) if len(f) > 2 else 0.0
            tz = _fval(f[3]) if len(f) > 3 else 0.0
            n1 = _ival(f[4]) if len(f) > 4 else 0
            n2 = _ival(f[5]) if len(f) > 5 else 0
            sub_id = _ival(f[6]) if len(f) > 6 else 0
        else:
            toks = cards[0].tokens()
            grnod = int(toks[0]) if len(toks) > 0 else 0
            tx = float(toks[1]) if len(toks) > 1 else 0.0
            ty = float(toks[2]) if len(toks) > 2 else 0.0
            tz = float(toks[3]) if len(toks) > 3 else 0.0
            n1 = int(toks[4]) if len(toks) > 4 else 0
            n2 = int(toks[5]) if len(toks) > 5 else 0
            sub_id = int(toks[6]) if len(toks) > 6 else 0

        skew_id = 0
        if len(cards) > 1:
            toks = cards[1].tokens()
            if toks:
                skew_id = int(toks[0])

        if skew_id > 0:
            log.warning(f"/TRANSFORM/TRA/{block.user_id}: local skew "
                        f"(skew_ID={skew_id}) not yet implemented — using "
                        f"global coordinates", block.source)

        model.transforms.append((block.user_id, "TRA", grnod, tx, ty, tz, n1, n2,
                                 sub_id, skew_id))

    elif sub == "ROT":
        if block.fixed:
            f1 = cards[0].cut("TRANSFORM_ROT_1")
            grnod = _ival(f1[0]) if len(f1) > 0 else 0
            x0 = _fval(f1[1]) if len(f1) > 1 else 0.0
            y0 = _fval(f1[2]) if len(f1) > 2 else 0.0
            z0 = _fval(f1[3]) if len(f1) > 3 else 0.0
            n1 = _ival(f1[4]) if len(f1) > 4 else 0
            n2 = _ival(f1[5]) if len(f1) > 5 else 0
            sub_id = _ival(f1[6]) if len(f1) > 6 else 0

            x1, y1, z1, angle = 0.0, 0.0, 0.0, 0.0
            if len(cards) > 1:
                f2 = cards[1].cut("TRANSFORM_ROT_2")
                x1 = _fval(f2[1]) if len(f2) > 1 else 0.0
                y1 = _fval(f2[2]) if len(f2) > 2 else 0.0
                z1 = _fval(f2[3]) if len(f2) > 3 else 0.0
                angle = _fval(f2[4]) if len(f2) > 4 else 0.0
        else:
            toks1 = cards[0].tokens()
            grnod = int(toks1[0]) if len(toks1) > 0 else 0
            x0 = float(toks1[1]) if len(toks1) > 1 else 0.0
            y0 = float(toks1[2]) if len(toks1) > 2 else 0.0
            z0 = float(toks1[3]) if len(toks1) > 3 else 0.0
            n1 = int(toks1[4]) if len(toks1) > 4 else 0
            n2 = int(toks1[5]) if len(toks1) > 5 else 0
            sub_id = int(toks1[6]) if len(toks1) > 6 else 0

            x1, y1, z1, angle = 0.0, 0.0, 0.0, 0.0
            if len(cards) > 1:
                toks2 = cards[1].tokens()
                x1 = float(toks2[0]) if len(toks2) > 0 else 0.0
                y1 = float(toks2[1]) if len(toks2) > 1 else 0.0
                z1 = float(toks2[2]) if len(toks2) > 2 else 0.0
                angle = float(toks2[3]) if len(toks2) > 3 else 0.0

        model.transforms.append((block.user_id, "ROT", grnod, (x0, y0, z0),
                                 (x1, y1, z1), angle, n1, n2, sub_id))

    elif sub == "SYM":
        if block.fixed:
            f1 = cards[0].cut("TRANSFORM_SYM_1")
            grnod = _ival(f1[0]) if len(f1) > 0 else 0
            x0 = _fval(f1[1]) if len(f1) > 1 else 0.0
            y0 = _fval(f1[2]) if len(f1) > 2 else 0.0
            z0 = _fval(f1[3]) if len(f1) > 3 else 0.0
            n1 = _ival(f1[4]) if len(f1) > 4 else 0
            n2 = _ival(f1[5]) if len(f1) > 5 else 0
            sub_id = _ival(f1[6]) if len(f1) > 6 else 0

            x1, y1, z1 = 0.0, 0.0, 0.0
            if len(cards) > 1:
                f2 = cards[1].cut("TRANSFORM_SYM_2")
                x1 = _fval(f2[1]) if len(f2) > 1 else 0.0
                y1 = _fval(f2[2]) if len(f2) > 2 else 0.0
                z1 = _fval(f2[3]) if len(f2) > 3 else 0.0
        else:
            toks1 = cards[0].tokens()
            grnod = int(toks1[0]) if len(toks1) > 0 else 0
            x0 = float(toks1[1]) if len(toks1) > 1 else 0.0
            y0 = float(toks1[2]) if len(toks1) > 2 else 0.0
            z0 = float(toks1[3]) if len(toks1) > 3 else 0.0
            n1 = int(toks1[4]) if len(toks1) > 4 else 0
            n2 = int(toks1[5]) if len(toks1) > 5 else 0
            sub_id = int(toks1[6]) if len(toks1) > 6 else 0

            x1, y1, z1 = 0.0, 0.0, 0.0
            if len(cards) > 1:
                toks2 = cards[1].tokens()
                x1 = float(toks2[0]) if len(toks2) > 0 else 0.0
                y1 = float(toks2[1]) if len(toks2) > 1 else 0.0
                z1 = float(toks2[2]) if len(toks2) > 2 else 0.0

        model.transforms.append((block.user_id, "SYM", grnod, (x0, y0, z0),
                                 (x1, y1, z1), n1, n2, sub_id))

    elif sub == "SCA":
        if block.fixed:
            f = cards[0].cut("TRANSFORM_SCA")
            grnod = _ival(f[0]) if len(f) > 0 else 0
            sx = _fval(f[1]) if len(f) > 1 else 1.0
            sy = _fval(f[2]) if len(f) > 2 else 1.0
            sz = _fval(f[3]) if len(f) > 3 else 1.0
            n1 = _ival(f[4]) if len(f) > 4 else 0
            sub_id = _ival(f[5]) if len(f) > 5 else 0
        else:
            toks = cards[0].tokens()
            grnod = int(toks[0]) if len(toks) > 0 else 0
            sx = float(toks[1]) if len(toks) > 1 else 1.0
            sy = float(toks[2]) if len(toks) > 2 else 1.0
            sz = float(toks[3]) if len(toks) > 3 else 1.0
            n1 = int(toks[4]) if len(toks) > 4 else 0
            sub_id = int(toks[5]) if len(toks) > 5 else 0

        model.transforms.append((block.user_id, "SCA", grnod, (sx, sy, sz),
                                 n1, sub_id))

    elif sub in ("POS", "POSITION"):
        if block.fixed:
            f = cards[0].cut("TRANSFORM_POS_1")
            grnod = _ival(f[0]) if len(f) > 0 else 0
            n1 = _ival(f[1]) if len(f) > 1 else 0
            n2 = _ival(f[2]) if len(f) > 2 else 0
            n3 = _ival(f[3]) if len(f) > 3 else 0
            n4 = _ival(f[4]) if len(f) > 4 else 0
            n5 = _ival(f[5]) if len(f) > 5 else 0
            n6 = _ival(f[6]) if len(f) > 6 else 0
            sub_id = _ival(f[9]) if len(f) > 9 else 0
            pts = []
            for i in range(1, 7):
                if i < len(cards) and not cards[i].is_blank:
                    p_cut = cards[i].cut("TRANSFORM_POS_PT")
                    pts.append([_fval(p_cut[1]), _fval(p_cut[2]), _fval(p_cut[3])])
                else:
                    pts.append([0.0, 0.0, 0.0])
        else:
            toks = cards[0].tokens()
            grnod = int(toks[0]) if len(toks) > 0 else 0
            n1 = int(toks[1]) if len(toks) > 1 else 0
            n2 = int(toks[2]) if len(toks) > 2 else 0
            n3 = int(toks[3]) if len(toks) > 3 else 0
            n4 = int(toks[4]) if len(toks) > 4 else 0
            n5 = int(toks[5]) if len(toks) > 5 else 0
            n6 = int(toks[6]) if len(toks) > 6 else 0
            sub_id = int(toks[7]) if len(toks) > 7 else 0
            pts = []
            for i in range(1, 7):
                if i < len(cards) and not cards[i].is_blank:
                    p_toks = cards[i].tokens()
                    pts.append([float(p_toks[0]) if len(p_toks) > 0 else 0.0,
                                float(p_toks[1]) if len(p_toks) > 1 else 0.0,
                                float(p_toks[2]) if len(p_toks) > 2 else 0.0])
                else:
                    pts.append([0.0, 0.0, 0.0])

        from ...model.entities import TransformPosition
        model.transform_positions[block.user_id] = TransformPosition(
            id=block.user_id,
            title=title,
            grnod_id=grnod,
            node_ids=(n1, n2, n3, n4, n5, n6),
            submodel=sub_id,
            points=tuple((p[0], p[1], p[2]) for p in pts),
        )
        model.transforms.append((block.user_id, "POS", grnod, (n1, n2, n3, n4, n5, n6),
                                 pts, sub_id))

    elif sub == "MATRIX":
        from ...model.entities import TransformMatrix
        m11, m12, m13, tx = 1.0, 0.0, 0.0, 0.0
        m21, m22, m23, ty = 0.0, 1.0, 0.0, 0.0
        m31, m32, m33, tz = 0.0, 0.0, 1.0, 0.0
        grnod = 0
        sub_id = 0
        if block.fixed:
            f1 = cards[0].cut("TRANSFORM_MATRIX_1")
            grnod = _ival(f1[0]) if len(f1) > 0 else 0
            m11 = _fval(f1[1], 1.0) if len(f1) > 1 else 1.0
            m12 = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            m13 = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
            tx = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
            sub_id = _ival(f1[5]) if len(f1) > 5 else 0

            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("TRANSFORM_MATRIX_2")
                m21 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                m22 = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
                m23 = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
                ty = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0

            if len(cards) > 2 and not cards[2].is_blank:
                f3 = cards[2].cut("TRANSFORM_MATRIX_3")
                m31 = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
                m32 = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
                m33 = _fval(f3[3], 1.0) if len(f3) > 3 else 1.0
                tz = _fval(f3[4], 0.0) if len(f3) > 4 else 0.0
        else:
            t1 = cards[0].tokens()
            grnod = int(float(t1[0])) if len(t1) > 0 else 0
            m11 = float(t1[1]) if len(t1) > 1 else 1.0
            m12 = float(t1[2]) if len(t1) > 2 else 0.0
            m13 = float(t1[3]) if len(t1) > 3 else 0.0
            tx = float(t1[4]) if len(t1) > 4 else 0.0
            sub_id = int(float(t1[5])) if len(t1) > 5 else 0

            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                m21 = float(t2[0]) if len(t2) > 0 else 0.0
                m22 = float(t2[1]) if len(t2) > 1 else 1.0
                m23 = float(t2[2]) if len(t2) > 2 else 0.0
                ty = float(t2[3]) if len(t2) > 3 else 0.0

            if len(cards) > 2 and not cards[2].is_blank:
                t3 = cards[2].tokens()
                m31 = float(t3[0]) if len(t3) > 0 else 0.0
                m32 = float(t3[1]) if len(t3) > 1 else 0.0
                m33 = float(t3[2]) if len(t3) > 2 else 1.0
                tz = float(t3[3]) if len(t3) > 3 else 0.0

        mat_3x3 = ((m11, m12, m13), (m21, m22, m23), (m31, m32, m33))
        trans_vec = (tx, ty, tz)
        tm = TransformMatrix(
            id=block.user_id or 1,
            title=title,
            grnod_id=grnod,
            matrix=mat_3x3,
            translation=trans_vec,
            sub_id=sub_id,
            submodel=sub_id,
        )
        model.transform_matrices[block.user_id or 1] = tm
        model.transforms.append((block.user_id or 1, "MATRIX", grnod, mat_3x3, trans_vec, sub_id))




def read_cyl_joint_m209(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CYL_JOINT/id`` or ``/LAGMUL/CYL_JOINT/id`` (M209): Cylindrical kinematic joint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CYL_JOINT/{block.user_id}: missing data card", block.source)
        return

    node1, node2, axis_dir, skew_id, tol = 0, 0, 1, 0, 1e-6
    if block.fixed:
        f = cards[0].cut("CYL_JOINT_1")
        node1 = _ival(f[0]) if len(f) > 0 else 0
        node2 = _ival(f[1]) if len(f) > 1 else 0
        axis_dir = _ival(f[2], 1) if len(f) > 2 else 1
        skew_id = _ival(f[3], 0) if len(f) > 3 else 0
        tol = _fval(f[4], 1e-6) if len(f) > 4 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0])) if len(toks) > 0 else 0
        node2 = int(float(toks[1])) if len(toks) > 1 else 0
        axis_dir = int(float(toks[2])) if len(toks) > 2 else 1
        skew_id = int(float(toks[3])) if len(toks) > 3 else 0
        tol = float(toks[4]) if len(toks) > 4 else 1e-6

    from ...model.entities import CylJoint
    model.cyl_joints[block.user_id] = CylJoint(
        id=block.user_id, title=title, node1=node1, node2=node2,
        axis_dir=axis_dir, skew_id=skew_id, tol=tol
    )




def read_parallel_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PARALLEL/id`` or ``/LAGMUL/PARALLEL/id`` (M213): Parallel axes kinematic joint."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PARALLEL/{block.user_id}: missing data card", block.source)
        return

    node1, node2, axis_dir, skew_id, tol = 0, 0, 1, 0, 1e-6
    if block.fixed:
        f = cards[0].cut("PARALLEL_JOINT_1")
        node1 = _ival(f[0]) if len(f) > 0 else 0
        node2 = _ival(f[1]) if len(f) > 1 else 0
        axis_dir = _ival(f[2], 1) if len(f) > 2 else 1
        skew_id = _ival(f[3], 0) if len(f) > 3 else 0
        tol = _fval(f[4], 1e-6) if len(f) > 4 else 1e-6
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0])) if len(toks) > 0 else 0
        node2 = int(float(toks[1])) if len(toks) > 1 else 0
        axis_dir = int(float(toks[2])) if len(toks) > 2 else 1
        skew_id = int(float(toks[3])) if len(toks) > 3 else 0
        tol = float(toks[4]) if len(toks) > 4 else 1e-6

    from ...model.entities import ParallelJoint
    model.parallel_joints[block.user_id] = ParallelJoint(
        id=block.user_id, title=title, node1=node1, node2=node2,
        axis_dir=axis_dir, skew_id=skew_id, tol=tol
    )






def read_frame_nod(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FRAME/NOD`` or ``/FRAME/NODE`` (M196): Nodal reference frame."""
    from ...model.entities import FrameNod
    frame_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    originnodeid, axisnodeid, planenodeid = 0, 0, 0
    displayaxis, displayplane = 0, 0
    globalyaxis = [0.0, 0.0, 0.0]
    globalzaxis = [0.0, 0.0, 0.0]
    params = {}

    if len(valid_cards) > 0:
        c0 = valid_cards[0].tokens()
        originnodeid = _safe_int(c0[0]) if len(c0) > 0 else 0
        axisnodeid = _safe_int(c0[1]) if len(c0) > 1 else 0
        planenodeid = _safe_int(c0[2]) if len(c0) > 2 else 0
    if len(valid_cards) > 1:
        c1 = valid_cards[1].tokens()
        for i in range(min(3, len(c1))):
            globalyaxis[i] = _safe_float(c1[i])
    if len(valid_cards) > 2:
        c2 = valid_cards[2].tokens()
        for i in range(min(3, len(c2))):
            globalzaxis[i] = _safe_float(c2[i])

    params.update({
        "originnodeid": originnodeid, "axisnodeid": axisnodeid, "planenodeid": planenodeid,
        "globalyaxis": globalyaxis, "globalzaxis": globalzaxis, "displayaxis": displayaxis, "displayplane": displayplane
    })
    fnod = FrameNod(
        id=frame_id, title=title, originnodeid=originnodeid, axisnodeid=axisnodeid, planenodeid=planenodeid,
        globalyaxis=globalyaxis, globalzaxis=globalzaxis, displayaxis=displayaxis, displayplane=displayplane, params=params
    )
    model.frame_nods[frame_id] = fnod
    _read_reference_system(block, model, log, "FRAME")
