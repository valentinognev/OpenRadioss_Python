# -*- coding: utf-8 -*-
"""
Entity-group readers - /KEYWORD blocks -> Model.

/PART, /GR**, /SET, /SUBMODEL, /SUBDOMAIN, /XREF, /RANDOM and the
/DFS detonation-ignition family.

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
# Part / material / property
# ============================================================================

def read_part(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PART/part_ID``::

        card 1:  part_title
        card 2:  prop_ID   mat_ID   [subset_ID  ignored]

    Fixed dialect (cfg PART/part.cfg ``%10d%10d%10d%20lg``): the title
    card is ALWAYS present — a purely numeric title ('1', written by
    HyperMesh) is still the title, and prop/mat/subset are cut at their
    columns (M37; the token view crashed on numeric titles and misread
    the mat column when fields abutted).

    Fortran: starter/source/model/assembling/hm_read_part.F.
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/PART/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("PART")           # prop mat subset (Thick %20lg)
        model.parts[block.user_id] = Part(
            id=block.user_id, prop_id=_ival(f[0]), mat_id=_ival(f[1]),
            title=title)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/PART/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    model.parts[block.user_id] = Part(
        id=block.user_id,
        prop_id=t[0] if len(t) > 0 else 0,
        mat_id=t[1] if len(t) > 1 else 0,
        title=title)




def read_random_stochastic_m37(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RANDOM/random_ID``: stochastic fields."""
    if block.fixed:
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)
    # Generic placeholder for /RANDOM
    model.randoms[block.user_id] = Random(block.user_id, {})




#: /GRNOD subtypes naming an ELEMENT-group family -> canonical family key
_GR_FAMILIES = {
    "GRSHEL": "SHEL", "GRSHELL": "SHEL",
    "GRSH3N": "SH3N", "GRTRIA": "SH3N",
    "GRBRIC": "BRIC", "GRBRICK": "BRIC", "GRHEXA": "BRIC", "GRHEX8": "BRIC",
    "GRBR20": "BRIC", "GRHEX20": "BRIC",
    "GRQUAD": "QUAD",
    "GRTRUS": "TRUS", "GRTRUSS": "TRUS",
    "GRBEAM": "BEAM",
    "GRSPRI": "SPRI", "GRSPRING": "SPRI",
    "GRTETRA4": "BRIC", "GRTET4": "BRIC",
    "GRTETRA10": "BRIC", "GRTET10": "BRIC",
}




def read_grnod(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRNOD/<subtype>/grnod_ID``: title card, then entity IDs (any
    number per card).  Ported subtypes (M37 — Fortran hm_lecgrn.F):

    * ``NODE`` — node ids;
    * ``PART`` — all nodes of the parts' elements;
    * ``BOX``  — all nodes inside the /BOX volumes (RECTA/CYLIN/SPHER);
    * ``SURF`` — all nodes of the surfaces' segments (hm_surfnod.F);
    * ``GRNOD`` — group of groups (hm_grogronod.F): RECURSIVE, resolved
      by iterative fixpoint with cycle detection; a NEGATIVE id REMOVES
      the referenced group's nodes (removal wins over addition whatever
      the order — the upstream BUFTMP = -1 convention);
    * ``GRSHEL|GRSH3N|GRBRIC|GRQUAD|GRTRUS|GRBEAM|GRSPRI`` — all nodes
      of the element groups' elements (hm_elngr/hm_elngrs);
    * ``GENE`` — generated ranges ``first_ID last_ID`` (pairs, 5 per
      card): every existing node with first <= id <= last;
      ``GEN_INCR`` — triplets ``first last incr`` (id stepping incr).
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    g = model.node_groups.setdefault(
        block.user_id, NodeGroup(id=block.user_id, title=title))
    if kind in ("GENE", "GEN_INCR"):
        step = 3 if kind == "GEN_INCR" else 2
        vals = _id_list(block, cards)
        for k in range(0, len(vals) - step + 1, step):
            first, last = vals[k], vals[k + 1]
            incr = vals[k + 2] if step == 3 else 1
            if first <= 0 or last < first or incr <= 0:
                log.error(f"/GRNOD/{kind}/{block.user_id}: bad range "
                          f"{vals[k:k + step]}", block.source)
                continue
            g.gene_ranges.append((first, last, incr))
        return
    ids = _id_list(block, cards)
    if kind in ("NODE", "NODENS"):
        g.node_ids.extend(ids)
    elif kind == "PART":
        g.part_ids.extend(ids)
    elif kind == "BOX":
        g.box_ids.extend(ids)
    elif kind == "SURF":
        g.surf_ids.extend(ids)
    elif kind == "LINE":
        g.line_ids.extend(ids)
    elif kind in ("GRNOD", "SUB", "SUBSET"):
        g.grnod_ids.extend(ids)
    elif kind in _GR_FAMILIES:
        g.egroup_refs.extend((_GR_FAMILIES[kind], i) for i in ids)
    else:
        log.warning(f"/GRNOD/{kind} not ported (NODE, NODENS, PART, BOX, SURF, LINE, "
                    f"GRNOD, GR<elem>, GENE supported)", block.source)




def read_gr_elem(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """Element groups ``/GRSHEL|GRSH3N|GRBRIC|GRQUAD|GRTRUS|GRBEAM|
    GRSPRI/<subtype>/id`` and part groups ``/GRPART/PART/id`` (M37, M136, M152).

    Fortran: hm_lecgre.F (direct lists + parts) and hm_grogro.F
    (recursive group-of-groups, same fixpoint/cycle/negative-id
    machinery as /GRNOD/GRNOD).  Ported subtypes::

        /GRSHEL/SHEL   /GRSH3N/SH3N   /GRBRIC/BRIC   ...  element ids
        /GR*/PART                       all the parts' elements of the
                                        family (GRBRIC = ALL solids)
        /GRSHEL/GRSHEL ...              group of groups (signed ids)
        /GRPART/PART                    part ids (the group IS parts)
        /GR*/BOX                        elements inside bounding box
        /GR*/SURF                       elements of contact surface

    Each family is its own id namespace (Model.egroups), exactly like
    the separate IGRSH4N/IGRSH3N/IGRBRIC arrays of groupdef_mod.F.
    """
    from ...model.entities import EntityGroup
    key0 = block.key0
    family = "PART" if key0 == "GRPART" else _GR_FAMILIES.get(key0)
    if family is None:
        log.warning(f"/{key0} not ported — block skipped", block.source)
        return
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    fam = model.egroups.setdefault(family, {})
    g = fam.setdefault(block.user_id, EntityGroup(
        id=block.user_id, family=family, title=title))
    ids = _id_list(block, cards)
    # the direct element-list subtype spelling per family (SHEL, SH3N,
    # BRIC, QUAD, TRUS, BEAM, SPRI — also accepted: ELEM)
    direct = {
        "SHEL": ("SHEL", "SHELL", "SH4N"),
        "SH3N": ("SH3N", "TRIA", "SH3"),
        "BRIC": ("BRIC", "BRICK", "HEXA", "HEX8", "HEXA8", "BR20", "HEX20",
                 "TETRA4", "TET4", "TETRA10", "TET10"),
        "QUAD": ("QUAD", "QUA4"),
        "TRUS": ("TRUS", "TRUSS"),
        "BEAM": ("BEAM",),
        "SPRI": ("SPRI", "SPRING"),
    }
    if key0 == "GRPART" and (kind in ("", "PART") or kind == str(block.user_id)):
        g.part_ids.extend(ids)                # the group IS a part list
    elif kind in direct.get(family, ()) or kind == "ELEM":
        g.elem_ids.extend(ids)
    elif kind == "PART":
        g.part_ids.extend(ids)
    elif kind == "BOX":
        g.box_ids.extend(ids)
    elif kind == "SURF":
        g.surf_ids.extend(ids)
    elif kind in (key0, "SUB", "SUBSET", family, f"GR{family}"):     # /GRSHEL/GRSHEL/... etc.
        g.group_ids.extend(ids)
    else:
        log.warning(f"/{key0}/{kind} not ported (direct ids, PART, BOX, SURF and "
                    f"/{key0}/{key0} group-of-groups supported)",
                    block.source)




def read_det(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIT/DET_POINT``, ``/INIT/DET_LINE``, ``/INIT/DET_PLAN``, ``/INIT/DET_CORD``,
    or ``/DET_POINT``, ``/DET_LINE``, ``/DET_PLAN``, ``/DET_CORD`` (M110):
    High-explosive detonation wavefront initialization.
    """
    if any(p.upper() in ("NODE", "SET", "GRNOD") for p in block.parts[1:] if not p.lstrip("-").isdigit()):
        read_dfs(block, model, log)
        return
    from ...model.entities import DetonationWave
    key0 = block.key0
    sub = ""
    if key0.startswith("DET_"):
        sub = key0[4:]
    elif key0 == "DET" and len(block.parts) > 1 and not block.parts[1].isdigit():
        sub = block.parts[1].upper()
    elif key0 in ("INIT", "LOAD") and len(block.parts) > 1:
        p1 = block.parts[1].upper()
        if p1.startswith("DET_"):
            sub = p1[4:]
        elif p1 == "DET" and len(block.parts) > 2 and not block.parts[2].isdigit():
            sub = block.parts[2].upper()
        elif p1 in ("POINT", "LINE", "PLAN", "CORD"):
            sub = p1
    if not sub:
        parts_str = "_".join(block.parts).upper()
        if "POINT" in parts_str:
            sub = "POINT"
        elif "LINE" in parts_str:
            sub = "LINE"
        elif "PLAN" in parts_str:
            sub = "PLAN"
        elif "CORD" in parts_str:
            sub = "CORD"
        else:
            sub = "POINT"

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/{key0}/{sub}/{block.user_id}: missing data cards", block.source)
        return

    det = DetonationWave(id=block.user_id, kind=sub, title=title)
    if sub == "POINT":
        if block.fixed:
            c = cards[0].cut("DET_POINT_1")
            det.x = _fval(c[0])
            det.y = _fval(c[1])
            det.z = _fval(c[2])
            det.tdet = _fval(c[3])
            det.mat_id = _ival(c[4])
        else:
            t = cards[0].tokens()
            det.x = float(t[0]) if len(t) > 0 else 0.0
            det.y = float(t[1]) if len(t) > 1 else 0.0
            det.z = float(t[2]) if len(t) > 2 else 0.0
            det.tdet = float(t[3]) if len(t) > 3 else 0.0
            det.mat_id = int(float(t[4])) if len(t) > 4 else 0
    elif sub == "LINE":
        if block.fixed:
            c = cards[0].cut("DET_LINE_1")
            det.x = _fval(c[0])
            det.y = _fval(c[1])
            det.z = _fval(c[2])
            det.x2 = _fval(c[3])
            det.y2 = _fval(c[4])
            det.z2 = _fval(c[5])
            det.tdet = _fval(c[6])
            det.mat_id = _ival(c[7])
            det.ddet = _fval(c[8])
        else:
            t = cards[0].tokens()
            det.x = float(t[0]) if len(t) > 0 else 0.0
            det.y = float(t[1]) if len(t) > 1 else 0.0
            det.z = float(t[2]) if len(t) > 2 else 0.0
            det.x2 = float(t[3]) if len(t) > 3 else 0.0
            det.y2 = float(t[4]) if len(t) > 4 else 0.0
            det.z2 = float(t[5]) if len(t) > 5 else 0.0
            det.tdet = float(t[6]) if len(t) > 6 else 0.0
            det.mat_id = int(float(t[7])) if len(t) > 7 else 0
            det.ddet = float(t[8]) if len(t) > 8 else 0.0
    elif sub == "PLAN":
        if block.fixed:
            c = cards[0].cut("DET_PLAN_1")
            det.x = _fval(c[0])
            det.y = _fval(c[1])
            det.z = _fval(c[2])
            det.x2 = _fval(c[3])
            det.y2 = _fval(c[4])
            det.z2 = _fval(c[5])
            det.tdet = _fval(c[6])
            det.mat_id = _ival(c[7])
            det.ddet = _fval(c[8])
        else:
            t = cards[0].tokens()
            det.x = float(t[0]) if len(t) > 0 else 0.0
            det.y = float(t[1]) if len(t) > 1 else 0.0
            det.z = float(t[2]) if len(t) > 2 else 0.0
            det.x2 = float(t[3]) if len(t) > 3 else 0.0
            det.y2 = float(t[4]) if len(t) > 4 else 0.0
            det.z2 = float(t[5]) if len(t) > 5 else 0.0
            det.tdet = float(t[6]) if len(t) > 6 else 0.0
            det.mat_id = int(float(t[7])) if len(t) > 7 else 0
            det.ddet = float(t[8]) if len(t) > 8 else 0.0
    elif sub == "CORD":
        if block.fixed:
            c = cards[0].cut("DET_CORD_1")
            det.ddet = _fval(c[0])
            det.iopt = _ival(c[1])
            det.tdet = _fval(c[2])
            det.mat_id = _ival(c[3])
        else:
            t = cards[0].tokens()
            det.ddet = float(t[0]) if len(t) > 0 else 0.0
            det.iopt = int(float(t[1])) if len(t) > 1 else 0
            det.tdet = float(t[2]) if len(t) > 2 else 0.0
            det.mat_id = int(float(t[3])) if len(t) > 3 else 0
    elif sub in ("WAVE_SHAPER", "WAVESHAPER"):
        from ...model.entities import WaveShaper
        surf_id, mat_id, thick, delay = 0, 0, 0.0, 0.0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("DFS_WAVE_SHAPER_1")
                surf_id = _ival(f[0]) if len(f) > 0 else 0
                mat_id = _ival(f[1]) if len(f) > 1 else 0
                thick = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            else:
                toks = cards[0].tokens()
                surf_id = int(float(toks[0])) if len(toks) > 0 else 0
                mat_id = int(float(toks[1])) if len(toks) > 1 else 0
                thick = float(toks[2]) if len(toks) > 2 else 0.0
                delay = float(toks[3]) if len(toks) > 3 else 0.0
        model.wave_shapers[block.user_id or 0] = WaveShaper(
            id=block.user_id or 0, title=title, surf_id=surf_id,
            mat_id=mat_id, thick=thick, delay=delay
        )
        return

    model.detonations.append(det)




def read_mid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MID/id`` (M194): Material ID assignment / mapping card."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ...model.entities import MidDirective
    mid = block.user_id if block.user_id is not None else 0
    mat_id = mid
    if cards and not cards[0].is_blank:
        toks = cards[0].tokens()
        if toks:
            try:
                mat_id = int(float(toks[0]))
            except ValueError:
                mat_id = mid
    model.mid_directives[mid] = MidDirective(id=mid, mat_id=mat_id, title=title)




def read_pid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PID/id`` (M194): Part/Property ID assignment / mapping card."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ...model.entities import PidDirective
    pid = block.user_id if block.user_id is not None else 0
    prop_id = pid
    if cards and not cards[0].is_blank:
        toks = cards[0].tokens()
        if toks:
            try:
                prop_id = int(float(toks[0]))
            except ValueError:
                prop_id = pid
    model.pid_directives[pid] = PidDirective(id=pid, prop_id=prop_id, title=title)




def read_random(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RANDOM`` or ``/RANDOM/GRNOD/grnod_ID`` (M113): Random vibration/noise parameters::

        card 1:  Xalea  Seed
    """
    from ...model.entities import RandomNoise
    grnod_id = 0
    if len(block.parts) > 1 and block.parts[1].upper() == "GRNOD":
        grnod_id = block.user_id if block.user_id is not None else 0

    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    xalea = 0.0
    seed = 0.0
    if cards:
        if block.fixed:
            f = cards[0].cut("RANDOM_1")
            xalea = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            seed = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        else:
            t = cards[0].tokens()
            xalea = float(t[0]) if len(t) > 0 else 0.0
            seed = float(t[1]) if len(t) > 1 else 0.0

    model.random_noises.append(RandomNoise(grnod_id=grnod_id, xalea=xalea, seed=seed))




def read_set(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SET/<subtype>/<id>`` or ``/SETS/<subtype>/<id>`` (M115): Entity sets."""
    stype = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    all_p = [p.upper() for p in block.parts]
    sub_qual = "PART" if "PART" in all_p else ("GENE" if "GENE" in all_p else ("GEN_INCR" if "GEN_INCR" in all_p else None))

    if stype in ("NODE", "NODENS", "GRNOD", "NS"):
        qual = "GEN_INCR" if "GEN_INCR" in all_p else ("GENE" if "GENE" in all_p else ("NODENS" if ("NODENS" in all_p or "NS" in all_p) else ("PART" if "PART" in all_p else ("BOX" if "BOX" in all_p else ("SURF" if "SURF" in all_p else ("GRNOD" if "GRNOD" in all_p else "NODE"))))))
        mod_block = KeywordBlock(
            keyword=f"/GRNOD/{qual}/{block.user_id}",
            parts=["GRNOD", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_grnod(mod_block, model, log)
    elif stype in ("PART", "GRPART"):
        mod_block = KeywordBlock(
            keyword=f"/GRPART/PART/{block.user_id}",
            parts=["GRPART", "PART", str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("SHELL", "SHEL", "GRSHEL"):
        qual = sub_qual or "SHEL"
        mod_block = KeywordBlock(
            keyword=f"/GRSHEL/{qual}/{block.user_id}",
            parts=["GRSHEL", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("SH3N", "TRIA", "GRSH3N", "GRTRIA"):
        qual = sub_qual or "SH3N"
        mod_block = KeywordBlock(
            keyword=f"/GRSH3N/{qual}/{block.user_id}",
            parts=["GRSH3N", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("BRIC", "SOLID", "GRBRIC", "GRBR20", "GRHEX20"):
        qual = sub_qual or "BRIC"
        mod_block = KeywordBlock(
            keyword=f"/GRBRIC/{qual}/{block.user_id}",
            parts=["GRBRIC", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("QUAD", "GRQUAD"):
        qual = sub_qual or "QUAD"
        mod_block = KeywordBlock(
            keyword=f"/GRQUAD/{qual}/{block.user_id}",
            parts=["GRQUAD", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("TRUS", "TRUSS", "GRTRUS"):
        qual = sub_qual or "TRUS"
        mod_block = KeywordBlock(
            keyword=f"/GRTRUS/{qual}/{block.user_id}",
            parts=["GRTRUS", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("BEAM", "GRBEAM"):
        qual = sub_qual or "BEAM"
        mod_block = KeywordBlock(
            keyword=f"/GRBEAM/{qual}/{block.user_id}",
            parts=["GRBEAM", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("SPRI", "SPRING", "GRSPRI"):
        qual = sub_qual or "SPRI"
        mod_block = KeywordBlock(
            keyword=f"/GRSPRI/{qual}/{block.user_id}",
            parts=["GRSPRI", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("SURF", "SURF_ALL", "SURF_EXT", "SURF_FREE"):
        read_surf(block, model, log)
    elif stype == "LINE":
        read_line(block, model, log)
    elif stype in ("MAT", "GRMAT", "MATERIAL"):
        # /SET/MAT/id (M137)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        ids = []
        for c in cards:
            if c.is_blank:
                continue
            for t in c.tokens():
                try:
                    ids.append(int(float(t)))
                except ValueError:
                    pass
        model.generic_sets.setdefault("MAT", {})[block.user_id] = SetGeneric(
            id=block.user_id, set_type="MAT", title=title, ids=ids
        )
    elif stype in ("PROP", "GRPROP", "PROPERTY"):
        # /SET/PROP/id (M137)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        ids = []
        for c in cards:
            if c.is_blank:
                continue
            for t in c.tokens():
                try:
                    ids.append(int(float(t)))
                except ValueError:
                    pass
        model.generic_sets.setdefault("PROP", {})[block.user_id] = SetGeneric(
            id=block.user_id, set_type="PROP", title=title, ids=ids
        )
    elif stype in ("SUB", "SUBSET"):
        # /SET/SUB/id (M137)
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        ids = []
        for c in cards:
            if c.is_blank:
                continue
            for t in c.tokens():
                try:
                    ids.append(int(float(t)))
                except ValueError:
                    pass
        model.generic_sets.setdefault("SUB", {})[block.user_id] = SetGeneric(
            id=block.user_id, set_type="SUB", title=title, ids=ids
        )
    elif stype in ("GENERAL", "GENE"):
        # /SET/GENERAL/id
        key = ""
        for p in block.parts[2:]:
            pu = p.upper()
            if pu in ("NODE", "NODENS", "SEG", "PART_E", "PART", "SOLID", "SHELL", "QUAD", "BEAM", "TRUSS", "SPRING"):
                key = pu
                break
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        ids = []
        seg_nodes = []
        for c in cards:
            if c.is_blank:
                continue
            toks = c.tokens()
            if not toks:
                continue
            if not key:
                t0 = toks[0].strip().upper()
                if t0 in ("NODE", "NODENS", "SEG", "PART_E", "PART", "SOLID", "SHELL", "QUAD", "BEAM", "TRUSS", "SPRING"):
                    key = t0
                    toks = toks[1:]
                elif any(t0.startswith(k) for k in ("NODE", "SEG", "PART_E", "PART")):
                    if t0.startswith("NODE"):
                        key = "NODE"
                    elif t0.startswith("SEG"):
                        key = "SEG"
                    elif t0.startswith("PART_E"):
                        key = "PART_E"
                    elif t0.startswith("PART"):
                        key = "PART"
                    else:
                        key = t0
                    toks = toks[1:]
                else:
                    try:
                        float(t0)
                    except ValueError:
                        key = t0
                        toks = toks[1:]

            if key == "SEG":
                card_ints = []
                for t in toks:
                    try:
                        card_ints.append(int(float(t)))
                    except ValueError:
                        pass
                if len(card_ints) == 5:
                    seg = card_ints[1:5]
                    seg_nodes.append(seg)
                    ids.extend(seg)
                elif len(card_ints) == 3:
                    seg = [card_ints[0], card_ints[1], card_ints[2], card_ints[2]]
                    seg_nodes.append(seg)
                    ids.extend(seg)
                elif len(card_ints) >= 4 and len(card_ints) % 4 == 0:
                    for i in range(0, len(card_ints), 4):
                        seg = list(card_ints[i:i+4])
                        if seg[3] == 0:
                            seg[3] = seg[2]
                        seg_nodes.append(seg)
                        ids.extend(seg)
                else:
                    for val in card_ints:
                        ids.append(val)
            else:
                for t in toks:
                    try:
                        ids.append(int(float(t)))
                    except ValueError:
                        pass
        model.generic_sets.setdefault("GENERAL", {})[block.user_id] = SetGeneric(
            id=block.user_id, set_type="GENERAL", title=title, ids=ids, key=key, seg_nodes=seg_nodes
        )
    else:
        log.warning(f"/SET/{stype} not ported", block.source)




def read_includedyna(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INCLUDE_DYNA``, ``/INCLUDE_LS-DYNA``, ``/INCL_DYNA`` (M107): LS-DYNA include file."""
    fname = block.cards[0].raw.strip() if block.cards else ""
    model.dyna_includes.append(IncludeDyna(filename=fname))






def read_submodel(block: KeywordBlock, model: Model,
                  log: MessageLog) -> None:
    """`/SUBMODEL/submodel_id[/unit_id]` — Sub-model container.

    Fortran origin: ``starter/source/model/submodel/lecsubmod.F`` and
    ``hm_cfg_files/config/CFG/radioss51/SUBMODEL/submodel.cfg``.
    In the real Starter, /SUBMODEL opens a container block whose entities
    (nodes, elements, etc.) are tagged with the submodel ID; /ENDSUB closes
    it.
    """
    sub_id = block.user_id if block.user_id is not None else 0
    unit_id = block.unit_id if block.unit_id is not None else 0
    title = ""
    off_def = off_nod = off_ele = off_part = off_mat = off_type = off_sub = 0
    if block.cards:
        if block.fixed:
            title, cards = _fixed_data(block)
            if cards:
                f = cards[0].cut("SUBMODEL")
                off_def = _ival(f[0]) if len(f) > 0 else 0
                off_nod = _ival(f[1]) if len(f) > 1 else 0
                off_ele = _ival(f[2]) if len(f) > 2 else 0
                off_part = _ival(f[3]) if len(f) > 3 else 0
                off_mat = _ival(f[4]) if len(f) > 4 else 0
                off_type = _ival(f[5]) if len(f) > 5 else 0
                off_sub = _ival(f[6]) if len(f) > 6 else 0
        else:
            title, cards = _title_and_data(block)
            if cards:
                t = cards[0].tokens()
                off_def = int(float(t[0])) if len(t) > 0 else 0
                off_nod = int(float(t[1])) if len(t) > 1 else 0
                off_ele = int(float(t[2])) if len(t) > 2 else 0
                off_part = int(float(t[3])) if len(t) > 3 else 0
                off_mat = int(float(t[4])) if len(t) > 4 else 0
                off_type = int(float(t[5])) if len(t) > 5 else 0
                off_sub = int(float(t[6])) if len(t) > 6 else 0

    sm = Submodel(
        id=sub_id, title=title, unit_id=unit_id,
        off_def=off_def, off_nod=off_nod, off_ele=off_ele,
        off_part=off_part, off_mat=off_mat, off_type=off_type,
        off_sub=off_sub,
    )
    model.submodels[sub_id] = sm
    model.active_submodels.append(sub_id)




def read_subdomain(block: KeywordBlock, model: Model,
                   log: MessageLog) -> None:
    """`/SUBDOMAIN/sub_id` — Domain partition for Rad2Rad coupling.

    Fortran origin: ``starter/source/coupling/rad2rad/lecextlnk.F``
    and ``hm_cfg_files/config/CFG/radioss2022/RAD2R/subdomain.cfg``.

    Card 1: title (100 chars).
    Cards 2+: free object list of part IDs — 10 per line in 10-column
    fixed format, or whitespace/comma separated in free format.
    Negative IDs mark exclusions (Fortran ``negativeIds``).
    """
    sub_id = block.user_id if block.user_id is not None else 0
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)

    # Collect all part IDs from the object list cards (reuse _id_list
    # which handles both fixed IDS10 layout and free-format tokens).
    all_ids = _id_list(block, cards)

    part_ids: List[int] = []
    neg_part_ids: List[int] = []
    for pid in all_ids:
        if pid > 0:
            part_ids.append(pid)
            if pid not in model.parts:
                log.warning(
                    f"/SUBDOMAIN/{sub_id}: part {pid} not found in model",
                    block.source,
                )
        elif pid < 0:
            neg_part_ids.append(abs(pid))

    sd = Subdomain(id=sub_id, title=title,
                   part_ids=part_ids, neg_part_ids=neg_part_ids)
    model.subdomains[sub_id] = sd




def read_xref(block: KeywordBlock, model: Model,
              log: MessageLog) -> None:
    """`/XREF/part_id` — Reference geometry (initial reference state).

    Fortran origin: ``starter/source/loads/reference_state/xref/hm_read_xref.F``
    and ``hm_cfg_files/config/CFG/radioss90/INITIAL_GEOMETRY/xref.cfg``.

    Card 1: title (100 chars).
    Card 2: nitrs (%10d) — number of steps from reference to initial state.
    Cards 3+: node coordinate table — node_ID X Y Z (%10d%20lg%20lg%20lg).
    """
    part_id = block.user_id if block.user_id is not None else 0
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)

    # Card 1 (after title): nitrs
    nitrs = 100  # Fortran default
    if cards:
        if block.fixed:
            f = cards[0].fields(10, 1)
            nitrs = _ival(f[0]) if f[0] else 100
        else:
            t = cards[0].tokens()
            nitrs = int(float(t[0])) if t else 100
        if nitrs == 0:
            nitrs = 100  # Fortran default when 0
        cards = cards[1:]

    # Remaining cards: node coordinate table
    node_ids: List[int] = []
    coords: List[List[float]] = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("NODE")  # [10, 20, 20, 20] layout
            nid = _ival(f[0])
            if nid == 0:
                continue
            x = _fval(f[1]) if len(f) > 1 else 0.0
            y = _fval(f[2]) if len(f) > 2 else 0.0
            z = _fval(f[3]) if len(f) > 3 else 0.0
        else:
            t = c.tokens()
            if not t:
                continue
            nid = int(float(t[0]))
            if nid == 0:
                continue
            x = float(t[1]) if len(t) > 1 else 0.0
            y = float(t[2]) if len(t) > 2 else 0.0
            z = float(t[3]) if len(t) > 3 else 0.0
        node_ids.append(nid)
        coords.append([x, y, z])

    xr = Xref(
        part_id=part_id, title=title, nitrs=nitrs,
        node_ids=np.array(node_ids, dtype=np.int32),
        coords=np.array(coords) if coords else np.zeros((0, 3)),
    )
    model.xrefs[part_id] = xr




def read_dfs(block: KeywordBlock, model: Model,
             log: MessageLog) -> None:
    """`/DFS/DETPOINT/det_id` and `/DFS/DETPLAN/det_id` — detonation ignition.

    Fortran origin: ``starter/source/initial_conditions/detonation/
    read_dfs_detpoint.F`` and ``read_dfs_detplan.F``.

    DETPOINT — point-source detonation::

        card 1:  XDET  YDET  ZDET  TDET  mat_IDDET   (%20lg*4 %10d)

    DETPLAN — planar detonation front::

        card 1:  XP  YP  ZP  TDET  mat_IDDET          (%20lg*4 %10d)
        card 2:  NX  NY  NZ                            (%20lg*3)
    """
    from ...model.entities import DetLine, DetPointNode, DetPointSet
    p0 = block.parts[0].upper()
    if p0 in ("DETPOINT", "DET_POINT", "DETPOIN"):
        sub = "DETPOINT"
        has_node = any(p.upper() in ("NODE",) for p in block.parts[1:] if not p.lstrip("-").isdigit())
        has_set = any(p.upper() in ("SET", "GRNOD") for p in block.parts[1:] if not p.lstrip("-").isdigit())
    elif p0 in ("DETPLAN", "DET_PLAN", "DETPLANE"):
        sub = "DETPLAN"
        has_node = any(p.upper() in ("NODE",) for p in block.parts[1:] if not p.lstrip("-").isdigit())
        has_set = any(p.upper() in ("SET", "GRNOD") for p in block.parts[1:] if not p.lstrip("-").isdigit())
    elif p0 in ("DETLINE", "DET_LINE"):
        sub = "DETLINE"
        has_node = any(p.upper() in ("NODE",) for p in block.parts[1:] if not p.lstrip("-").isdigit())
        has_set = any(p.upper() in ("SET", "GRNOD") for p in block.parts[1:] if not p.lstrip("-").isdigit())
    elif p0 in ("WAVE", "DFS_WAV_SHA", "DFS_WAVSHA"):
        sub = "WAV_SHA"
        has_node = False
        has_set = False
    else:
        sub = block.parts[1].upper() if len(block.parts) > 1 else ""
        if sub == "LASER":
            read_laser(block, model, log)
            return
        has_node = any(p.upper() in ("NODE",) for p in block.parts[2:] if not p.lstrip("-").isdigit())
        has_set = any(p.upper() in ("SET", "GRNOD") for p in block.parts[2:] if not p.lstrip("-").isdigit())

    det_id = block.user_id if block.user_id is not None else 0
    title, cards = _title_and_data(block)

    if not cards:
        log.error(f"/DFS/{sub}/{det_id}: missing data card", block.source)
        return

    if has_node:
        if sub in ("DETPOINT", "DETPOIN"):
            # /DFS/DETPOINT/NODE/det_id or /DETPOINT/NODE:
            # Card 1: ishadow, iframe1, iframe2, R0_shadow, [blank], TDET, mat_ID, node_ID1
            ishadow, iframe1, iframe2 = 0, 0, 0
            r0_shadow, tdet = 0.0, 0.0
            mat_id, node_id1 = 0, 0
            if block.fixed:
                f = cards[0].cut("DFS_DETPOINT_NODE")
                if len(f) >= 8:
                    ishadow = _ival(f[0])
                    iframe1 = _ival(f[1])
                    iframe2 = _ival(f[2])
                    r0_shadow = _fval(f[3], 0.0)
                    tdet = _fval(f[5], 0.0)
                    mat_id = _ival(f[6])
                    node_id1 = _ival(f[7])
                elif len(f) >= 4:
                    tdet = _fval(f[1], 0.0)
                    mat_id = _ival(f[2])
                    node_id1 = _ival(f[3])
            else:
                toks = cards[0].tokens()
                if len(toks) >= 7:
                    ishadow = int(float(toks[0]))
                    iframe1 = int(float(toks[1]))
                    iframe2 = int(float(toks[2]))
                    r0_shadow = float(toks[3])
                    tdet = float(toks[4])
                    mat_id = int(float(toks[5]))
                    node_id1 = int(float(toks[6]))
                elif len(toks) >= 3:
                    tdet = float(toks[0])
                    mat_id = int(float(toks[1]))
                    node_id1 = int(float(toks[2]))
            dp = DetonatorPoint(id=det_id, tdet=tdet, mat_id=mat_id, node_id=node_id1)
            model.det_points.append(dp)
            dp_node = DetPointNode(
                id=det_id, title=title, ishadow=ishadow, iframe1=iframe1, iframe2=iframe2,
                r0_shadow=r0_shadow, radius=r0_shadow, tdet=tdet, mat_id=mat_id,
                node_id1=node_id1, node_id=node_id1
            )
            model.detpoint_nodes[det_id] = dp_node
            return
        elif sub in ("DETPLAN", "DETPLANE"):
            # /DFS/DETPLAN/NODE/det_id:
            # Card 1: %60s%20lg%10d%10d: rad_det_time, rad_det_materialid, rad_det_node1
            # Card 2: %90s%10d: rad_det_node2
            if block.fixed:
                f1 = cards[0].cut("DFS_DETPLAN_NODE_1")
                tdet = _fval(f1[1]) if len(f1) > 1 else 0.0
                mat_id = _ival(f1[2]) if len(f1) > 2 else 0
                p_id = _ival(f1[3]) if len(f1) > 3 else 0
                n_id = 0
                if len(cards) > 1 and not cards[1].is_blank:
                    f2 = cards[1].cut("DFS_DETPLAN_NODE_2")
                    n_id = _ival(f2[1]) if len(f2) > 1 else 0
            else:
                toks1 = cards[0].tokens()
                tdet = float(toks1[0]) if len(toks1) > 0 else 0.0
                mat_id = int(float(toks1[1])) if len(toks1) > 1 else 0
                p_id = int(float(toks1[2])) if len(toks1) > 2 else 0
                n_id = 0
                if len(cards) > 1 and not cards[1].is_blank:
                    toks2 = cards[1].tokens()
                    n_id = int(float(toks2[0])) if len(toks2) > 0 else 0
            dp = DetonatorPlane(id=det_id, tdet=tdet, mat_id=mat_id, p_id=p_id, n_id=n_id)
            model.det_planes.append(dp)
            return
        elif sub in ("DETLINE",):
            # /DFS/DETLINE/NODE/det_id:
            # Card 1: %90s%10d: rad_det_node1
            # Card 2: %90s%10d: rad_det_node2
            # Card 3: %20lg%10d: rad_det_time, rad_det_materialid
            if block.fixed:
                f1 = cards[0].cut("DFS_DETLINE_NODE_1")
                node1 = _ival(f1[1]) if len(f1) > 1 else 0
                node2 = 0
                if len(cards) > 1 and not cards[1].is_blank:
                    f2 = cards[1].cut("DFS_DETLINE_NODE_2")
                    node2 = _ival(f2[1]) if len(f2) > 1 else 0
                tdet, mat_id = 0.0, 0
                if len(cards) > 2 and not cards[2].is_blank:
                    f3 = cards[2].cut("DFS_DETLINE_NODE_3")
                    tdet = _fval(f3[0]) if len(f3) > 0 else 0.0
                    mat_id = _ival(f3[1]) if len(f3) > 1 else 0
            else:
                toks1 = cards[0].tokens()
                node1 = int(float(toks1[0])) if len(toks1) > 0 else 0
                node2 = 0
                if len(cards) > 1 and not cards[1].is_blank:
                    toks2 = cards[1].tokens()
                    node2 = int(float(toks2[0])) if len(toks2) > 0 else 0
                tdet, mat_id = 0.0, 0
                if len(cards) > 2 and not cards[2].is_blank:
                    toks3 = cards[2].tokens()
                    tdet = float(toks3[0]) if len(toks3) > 0 else 0.0
                    mat_id = int(float(toks3[1])) if len(toks3) > 1 else 0
            dl = DetLine(id=det_id, node1=node1, node2=node2, t0=tdet, mat_id=mat_id)
            model.det_lines[det_id] = dl
            return
        elif sub in ("DETCORD",):
            # /DFS/DETCORD/NODE/det_id (M163): ordered list of node numbers (cards)
            from ...model.entities import DfsDetcord
            node_ids = []
            for c in cards:
                if c.is_blank:
                    continue
                if block.fixed:
                    for k in range(0, min(len(c.raw), 100), 10):
                        chunk = c.raw[k:k+10].strip()
                        if chunk:
                            try:
                                node_ids.append(int(float(chunk)))
                            except ValueError:
                                pass
                else:
                    for tok in c.tokens():
                        try:
                            node_ids.append(int(float(tok)))
                        except ValueError:
                            pass
            model.dfs_detcords[det_id] = DfsDetcord(id=det_id, title=title, nodes=node_ids)
            return

    if has_set:
        if sub in ("DETPOINT", "DETPOIN"):
            # /DFS/DETPOINT/SET/det_id or /DETPOINT/SET or /DETPOINT/GRNOD:
            # Card 1: ishadow, iframe1, iframe2, R0_shadow, [blank], TDET, mat_ID, grnod_ID1
            ishadow, iframe1, iframe2 = 0, 0, 0
            r0_shadow, tdet = 0.0, 0.0
            mat_id, grnod_id1 = 0, 0
            if block.fixed:
                f = cards[0].cut("DFS_DETPOINT_SET") if "DFS_DETPOINT_SET" in CARD_LAYOUTS else cards[0].cut("DFS_DETPOINT_NODE")
                if len(f) >= 8:
                    ishadow = _ival(f[0])
                    iframe1 = _ival(f[1])
                    iframe2 = _ival(f[2])
                    r0_shadow = _fval(f[3], 0.0)
                    tdet = _fval(f[5], 0.0)
                    mat_id = _ival(f[6])
                    grnod_id1 = _ival(f[7])
                elif len(f) >= 4:
                    tdet = _fval(f[1], 0.0)
                    mat_id = _ival(f[2])
                    grnod_id1 = _ival(f[3])
            else:
                toks = cards[0].tokens()
                if len(toks) >= 7:
                    ishadow = int(float(toks[0]))
                    iframe1 = int(float(toks[1]))
                    iframe2 = int(float(toks[2]))
                    r0_shadow = float(toks[3])
                    tdet = float(toks[4])
                    mat_id = int(float(toks[5]))
                    grnod_id1 = int(float(toks[6]))
                elif len(toks) >= 3:
                    tdet = float(toks[0])
                    mat_id = int(float(toks[1]))
                    grnod_id1 = int(float(toks[2]))
            dp_set = DetPointSet(
                id=det_id, title=title, ishadow=ishadow, iframe1=iframe1, iframe2=iframe2,
                r0_shadow=r0_shadow, radius=r0_shadow, tdet=tdet, mat_id=mat_id,
                grnod_id1=grnod_id1, grnod_id=grnod_id1
            )
            model.detpoint_sets[det_id] = dp_set
            return
        else:
            log.warning(f"/DFS/{sub}/SET not fully ported — node group "
                        f"expansion deferred", block.source)
            return

    if sub in ("DETPOINT", "DETPOIN"):
        # Card 1: XDET YDET ZDET TDET mat_IDDET
        if block.fixed:
            f = _cut_floats(cards[0], "DFS_DETPOINT")
        else:
            f = _floats(cards[0], 5)
        x = f[0] if len(f) > 0 else 0.0
        y = f[1] if len(f) > 1 else 0.0
        z = f[2] if len(f) > 2 else 0.0
        tdet = f[3] if len(f) > 3 else 0.0
        mat_id = int(f[4]) if len(f) > 4 and f[4] else 0
        dp = DetonatorPoint(id=det_id, x=x, y=y, z=z,
                            tdet=tdet, mat_id=mat_id)
        model.det_points.append(dp)

    elif sub in ("DETPLAN", "DETPLANE"):
        # Card 1: XP YP ZP TDET mat_IDDET
        if block.fixed:
            f = _cut_floats(cards[0], "DFS_DETPLAN_1")
        else:
            f = _floats(cards[0], 5)
        x = f[0] if len(f) > 0 else 0.0
        y = f[1] if len(f) > 1 else 0.0
        z = f[2] if len(f) > 2 else 0.0
        tdet = f[3] if len(f) > 3 else 0.0
        mat_id = int(f[4]) if len(f) > 4 and f[4] else 0

        # Card 2: NX NY NZ
        nx, ny, nz = 0.0, 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            if block.fixed:
                g = _cut_floats(cards[1], "DFS_DETPLAN_2")
            else:
                g = _floats(cards[1], 3)
            nx = g[0] if len(g) > 0 else 0.0
            ny = g[1] if len(g) > 1 else 0.0
            nz = g[2] if len(g) > 2 else 0.0

        if nx == 0.0 and ny == 0.0 and nz == 0.0:
            log.warning(f"/DFS/DETPLAN/{det_id}: direction vector is zero",
                        block.source)

        dp = DetonatorPlane(id=det_id, x=x, y=y, z=z, tdet=tdet,
                            mat_id=mat_id, nx=nx, ny=ny, nz=nz)
        model.det_planes.append(dp)

    elif sub in ("DETPOINTSET", "DETPOINSET", "DETSET") or block.key0 == "DETPOINTSET":
        # Card: %60s%20lg%10d%10d: rad_det_time, rad_det_materialid, rad_det_grnod1
        if block.fixed:
            f = _fixed_vals(cards[0], [60, 20, 10, 10])
            tdet = _fval(f[1]) if len(f) > 1 else 0.0
            mat_id = _ival(f[2]) if len(f) > 2 else 0
            grnod_id = _ival(f[3]) if len(f) > 3 else 0
        else:
            toks = cards[0].tokens()
            tdet = float(toks[0]) if len(toks) > 0 else 0.0
            mat_id = int(float(toks[1])) if len(toks) > 1 else 0
            grnod_id = int(float(toks[2])) if len(toks) > 2 else 0
        dp = DetonatorPoint(id=det_id, tdet=tdet, mat_id=mat_id, grnod_id=grnod_id)
        model.det_points.append(dp)

    elif sub in ("WAV_SHA", "WAVSHA", "WAVE"):
        # /DFS/WAV_SHA/id or /WAVE/id (M201)
        # Card 1: XDET, YDET, ZDET, TDET, mat_ID, grnod_ID (%20lg%20lg%20lg%20lg%10d%10d)
        from ...model.entities import DfsWavSha
        xdet, ydet, zdet, tdet = 0.0, 0.0, 0.0, 0.0
        mat_id, grnod_id = 0, 0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f6 = cards[0].cut("DFS_WAV_SHA_1")
                xdet = _fval(f6[0], 0.0) if len(f6) > 0 else 0.0
                ydet = _fval(f6[1], 0.0) if len(f6) > 1 else 0.0
                zdet = _fval(f6[2], 0.0) if len(f6) > 2 else 0.0
                tdet = _fval(f6[3], 0.0) if len(f6) > 3 else 0.0
                mat_id = _ival(f6[4]) if len(f6) > 4 else 0
                grnod_id = _ival(f6[5]) if len(f6) > 5 else 0
            else:
                toks = cards[0].tokens()
                xdet = float(toks[0]) if len(toks) > 0 else 0.0
                ydet = float(toks[1]) if len(toks) > 1 else 0.0
                zdet = float(toks[2]) if len(toks) > 2 else 0.0
                tdet = float(toks[3]) if len(toks) > 3 else 0.0
                mat_id = int(float(toks[4])) if len(toks) > 4 else 0
                grnod_id = int(float(toks[5])) if len(toks) > 5 else 0
        ws_obj = DfsWavSha(id=det_id, title=title, xdet=xdet, ydet=ydet, zdet=zdet, tdet=tdet, mat_id=mat_id, grnod_id=grnod_id)
        model.dfs_wav_shas[det_id] = ws_obj
        model.det_points.append(DetonatorPoint(id=det_id, tdet=tdet, mat_id=mat_id, grnod_id=grnod_id))

    elif sub in ("WAVE_SHAPER", "WAVESHAPER"):
        # /DFS/WAVE_SHAPER/id (M132)
        from ...model.entities import WaveShaper
        surf_id, mat_id, thick, delay = 0, 0, 0.0, 0.0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("DFS_WAVE_SHAPER_1")
                surf_id = _ival(f[0]) if len(f) > 0 else 0
                mat_id = _ival(f[1]) if len(f) > 1 else 0
                thick = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                delay = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            else:
                toks = cards[0].tokens()
                surf_id = int(float(toks[0])) if len(toks) > 0 else 0
                mat_id = int(float(toks[1])) if len(toks) > 1 else 0
                thick = float(toks[2]) if len(toks) > 2 else 0.0
                delay = float(toks[3]) if len(toks) > 3 else 0.0
        model.wave_shapers[det_id] = WaveShaper(
            id=det_id, title=title,
            surf_id=surf_id, mat_id=mat_id, thick=thick, delay=delay
        )

    elif sub == "DETLINE":
        # /DFS/DETLINE/id (M137)
        # Card 1: X1 Y1 Z1 X2 Y2 Z2 T0 Dvel
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("DFS_DETLINE_1")
                p1 = (_fval(f[0]), _fval(f[1]), _fval(f[2]))
                p2 = (_fval(f[3]), _fval(f[4]), _fval(f[5]))
                t0 = _fval(f[6]) if len(f) > 6 else 0.0
                dvel = _fval(f[7]) if len(f) > 7 else 0.0
            else:
                toks = cards[0].tokens()
                p1 = (float(toks[0]), float(toks[1]), float(toks[2])) if len(toks) >= 3 else (0.0, 0.0, 0.0)
                p2 = (float(toks[3]), float(toks[4]), float(toks[5])) if len(toks) >= 6 else (0.0, 0.0, 0.0)
                t0 = float(toks[6]) if len(toks) > 6 else 0.0
                dvel = float(toks[7]) if len(toks) > 7 else 0.0
        else:
            p1, p2, t0, dvel = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0
        model.det_lines[det_id] = DetLine(
            id=det_id, title=title, p1=p1, p2=p2, t0=t0, dvel=dvel
        )

    elif sub == "DETCIRC":
        # /DFS/DETCIRC/id (M137)
        # Card 1: Xc Yc Zc Nx Ny Nz R T0 Dvel
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("DFS_DETCIRC_1")
                center = (_fval(f[0]), _fval(f[1]), _fval(f[2]))
                axis = (_fval(f[3]), _fval(f[4]), _fval(f[5], 1.0))
                r = _fval(f[6]) if len(f) > 6 else 0.0
                t0 = _fval(f[7]) if len(f) > 7 else 0.0
                dvel = _fval(f[8]) if len(f) > 8 else 0.0
            else:
                toks = cards[0].tokens()
                center = (float(toks[0]), float(toks[1]), float(toks[2])) if len(toks) >= 3 else (0.0, 0.0, 0.0)
                axis = (float(toks[3]), float(toks[4]), float(toks[5])) if len(toks) >= 6 else (0.0, 0.0, 1.0)
                r = float(toks[6]) if len(toks) > 6 else 0.0
                t0 = float(toks[7]) if len(toks) > 7 else 0.0
                dvel = float(toks[8]) if len(toks) > 8 else 0.0
        else:
            center, axis, r, t0, dvel = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0, 0.0, 0.0
        model.det_circs[det_id] = DetCirc(
            id=det_id, title=title, center=center, axis=axis, radius=r, t0=t0, dvel=dvel
        )

    elif sub == "DETCORD":
        # /DFS/DETCORD/id (M163)
        # Card 1: grnd_ID T_det V_cj Iopt mat_ID
        from ...model.entities import DfsDetcord
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("DFS_DETCORD_1")
                grnd_id = _ival(f[0]) if len(f) > 0 else 0
                t_det = _fval(f[1]) if len(f) > 1 else 0.0
                v_cj = _fval(f[2]) if len(f) > 2 else 0.0
                iopt = _ival(f[3], 3) if len(f) > 3 and f[3].strip() else 3
                mat_id = _ival(f[4]) if len(f) > 4 else 0
            else:
                toks = cards[0].tokens()
                grnd_id = int(float(toks[0])) if len(toks) > 0 else 0
                t_det = float(toks[1]) if len(toks) > 1 else 0.0
                v_cj = float(toks[2]) if len(toks) > 2 else 0.0
                iopt = int(float(toks[3])) if len(toks) > 3 else 3
                mat_id = int(float(toks[4])) if len(toks) > 4 else 0
        else:
            grnd_id, t_det, v_cj, iopt, mat_id = 0, 0.0, 0.0, 3, 0
        if iopt == 0:
            iopt = 3
        model.dfs_detcords[det_id] = DfsDetcord(
            id=det_id, title=title, grnd_id=grnd_id, t_det=t_det, v_cj=v_cj, iopt=iopt, mat_id=mat_id
        )

    else:
        log.warning(f"/DFS/{sub} not ported (DETPOINT, DETPLAN, WAVE_SHAPER, DETLINE, DETCIRC, DETCORD supported)",
                    block.source)
