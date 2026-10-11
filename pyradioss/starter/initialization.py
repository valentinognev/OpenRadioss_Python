"""
Starter finalization: user IDs → indices, element groups, node groups,
contact surfaces, lumped mass.

Fortran origin: the Starter phases after reading — ``USR2SYS`` id
conversion, ``s/c/t/r-init3`` element initialization, ``inimass``/
``initwg`` mass building, the group and surface builders under
``starter/source/model/sets``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..common.messages import MessageLog
from ..elements import KERNELS
from ..input import prop_reader
from ..model.entities import Material
from ..model.model import ElementGroup, Model

# element type name -> (attr on Model, nodes per element, required prop type)
_ETYPES = {
    "BRICK": ("bricks", 8, 14),
    "PENTA6": ("penta6s", 6, 14),
    "QUAD": ("quads", 4, 14),
    "QUAD4": ("quads_full", 4, 14),
    "TETRA4": ("tetras", 4, 14),
    "TETRA10": ("tetra10s", 10, 14),
    "PYRA": ("pyra5s", 5, 14),
    "PYRA5": ("pyra5s", 5, 14),
    "TRIA": ("trias", 3, 14),
    "TRIA3": ("trias", 3, 14),
    "SHELL": ("shells", 4, 1),
    "SH3N": ("sh3n", 3, 1),
    "TRUSS": ("trusses", 2, 2),
    "SPRING": ("springs", 2, 4),
    "BEAM": ("beams", 3, 3),
    "TSHELL": ("tshells", 8, 20),
    "SHEL16": ("shel16s", 16, 20),
    "BRIC20": ("bric20s", 20, 23),
}

#: the "fictitious material law for spring elements" the reference assigns
#: to a /PART with mat_ID 0 on a spring property (hm_read_part.F): springs
#: read their mass/stiffness from the /PROP, never a material, so this is a
#: bare inert placeholder (never inactive — the Engine must not refuse it).
_SPRING_MAT_SINGLETON: Material = None       # type: ignore[assignment]


def _fictitious_spring_material() -> Material:
    global _SPRING_MAT_SINGLETON
    if _SPRING_MAT_SINGLETON is None:
        _SPRING_MAT_SINGLETON = Material(
            id=0, law=-1, rho0=0.0, title="fictitious spring material",
            params={"E": 0.0, "nu": 0.0})
    return _SPRING_MAT_SINGLETON


# ----------------------------------------------------------------------------
# Degenerated bricks: repeated nodes -> tetra conversion / clear rejection
# ----------------------------------------------------------------------------

def _convert_degenerated_bricks(model: Model, log: MessageLog) -> None:
    """Handle /BRICK cards with repeated node IDs (the classic Radioss way
    of writing lower-order solids in brick format).

    Fortran origin: the reference does NOT convert a degenerated brick into
    another element type — it runs it through the SAME 8-node hexa kernel
    with the repeated-node connectivity exactly as written, and merely
    COUNTS the collapsed nodes for a few local corrections.  That counter is
    ``engine/source/elements/solid/solide/degenes8.F`` (DEGENES8), whose
    whole body is "for each node, is it repeated elsewhere in IXS(1..8)?"
    reduced to ``IDEGE(I) = <repeats>/2``; its consumers are the solid
    drivers themselves (sforc3.F, s8eforc3.F, s8zforc3.F, szforc3.F ...),
    which use IDEGE only to pick the small-strain branch, to skip an
    optimisation, or to correct the characteristic length
    (``sdlen_dege.F``: "DELTAX correction for degenerated element").  The
    connectivity itself is never rewritten, and the Starter's element count
    is unchanged: RD-V-0700's HEXA_DEGE deck — 40 bricks, every one a
    6-distinct-node collapsed wedge — reports ``NUMELS: NUMBER OF 3D SOLID
    ELEMENTS . . . 40`` in the reference .out.

    So this port keeps a degenerated brick AS a brick (M39):

    * **8 distinct** — an ordinary hexa;
    * **5, 6 or 7 distinct** — a collapsed hexa: the connectivity is passed
      through UNCHANGED (repeats included) and the standard 8-node kernel
      runs it.  This is not an approximation of a penta/pyramid element —
      it *is* how the reference represents one.  The isoparametric map
      simply degenerates along the collapsed edge; the coincident nodes'
      shape functions sum, so the mass lumping ``mass/8`` per corner gives
      a repeated node its 2/8 share (exactly the reference's lumping) and
      the centroid Jacobian yields the wedge volume.  The classic pattern
      is the wedge ``n1 n1 n3 n4 n5 n5 n7 n8`` (bottom and top faces each
      collapsed along one edge), which is what HEXA_DEGE and RD_V_0240's
      Modele_HEXA_P14 are built from.  Before M39 these were REFUSED
      ("degenerated brick ... is not ported"), which cost the three
      official element-verification decks c12/c18/c27 their whole run —
      and, since every element in HEXA_DEGE is degenerate, also produced
      the follow-on "model has no elements";
    * **4 distinct** (e.g. ``n1 n2 n3 n3 n5 n5 n5 n5``) — promoted to a
      genuine /TETRA4.  Kept from the earlier port: for the FULLY collapsed
      pattern the constant-strain tetra is the better element (a collapsed
      hexa leaves zero-volume sub-shapes in the hourglass base vectors),
      and it is what the port's own /TETRA4 kernel is for.  Distinct nodes
      are taken in order of first appearance, which maps every standard
      collapse pattern onto the positively-oriented tetra (re-checked at
      element init);
    * **fewer than 4 distinct** — a brick collapsed past any 3-D shape has
      no volume to integrate; that stays a hard error.
    """
    kept, moved, degen = [], 0, 0
    for (eid, pid, nodes) in model.raw_elems["BRICK"]:
        uniq = list(dict.fromkeys(nodes))         # distinct, order preserved
        if len(uniq) == 8:
            kept.append((eid, pid, nodes))
        elif len(uniq) == 4:
            model.raw_elems["TETRA4"].append((eid, pid, uniq))
            moved += 1
        elif len(uniq) in (5, 6, 7):
            # collapsed hexa — connectivity AS WRITTEN, repeats included
            kept.append((eid, pid, nodes))
            degen += 1
        else:
            log.error(
                f"/BRICK {eid}: degenerated brick with only {len(uniq)} "
                f"distinct node(s) — collapsed past any 3-D shape, it has "
                f"no volume", "BRICK DEGEN")
    model.raw_elems["BRICK"] = kept
    if moved:
        log.info(f"     {moved} DEGENERATED /BRICK ELEMENT(S) CONVERTED "
                 f"TO /TETRA4")
    if degen:
        log.info(f"     {degen} DEGENERATED /BRICK ELEMENT(S) RUN AS "
                 f"COLLAPSED HEXA (penta/pyramid)")


def _convert_tetras(model: Model, log: MessageLog) -> None:
    """Convert /TETRA4 elements with nodal-pressure variants (Itetra4 = 1, 2)
    into /TETRA10 elements with zeroed mid-side nodes, maintaining Fortran
    physics parity (virtual slaved mid-side nodes).
    """
    kept, moved = [], 0
    for (eid, pid, nodes) in model.raw_elems["TETRA4"]:
        # part -> property -> itetra4 flag
        part = model.parts.get(pid)
        prop_id = part.prop_id if part is not None else None
        prop = model.properties.get(prop_id) if prop_id is not None else None
        if prop and prop.params.get("itetra4", 0) in (1, 2):
            # Pad with 6 zeros for the mid-side nodes
            nodes10 = list(nodes) + [0] * 6
            model.raw_elems["TETRA10"].append((eid, pid, nodes10))
            moved += 1
        else:
            kept.append((eid, pid, nodes))
    model.raw_elems["TETRA4"] = kept
    if moved:
        log.info(f"     {moved} /TETRA4 ELEMENT(S) CONVERTED TO /TETRA10 "
                 f"(Nodal-pressure variant Itetra4=1/2)")


# ----------------------------------------------------------------------------
# Elements: raw tuples -> ElementGroups with per-part slices
# ----------------------------------------------------------------------------

def build_element_groups(model: Model, log: MessageLog) -> None:
    """Convert the raw (id, part, nodes) tuples collected by the parsers
    into dense ElementGroups, **sorted by part** so that each part is a
    contiguous slice — the Python equivalent of the Fortran element
    *groups* (NGROUP blocks of same type/mat/prop), which lets the material
    law run vectorized on each slice."""
    _convert_degenerated_bricks(model, log)
    _convert_tetras(model, log)
    for etype, (attr, nnode, req_prop) in _ETYPES.items():
        raw = model.raw_elems[etype]
        if not raw:
            continue
        raw.sort(key=lambda t: t[1])                 # sort by part id
        ids = np.array([t[0] for t in raw], dtype=np.int64)
        part_ids = np.array([t[1] for t in raw], dtype=np.int64)

        # user node ids -> indices (USR2SYS)
        valid_raw = []
        valid_conn = []
        max_nodes = max(len(t[2]) for t in raw) if (etype == "SPRING" and raw) else nnode
        for eid, pid, nodes in raw:
            try:
                c_nodes = []
                for j, n in enumerate(nodes):
                    n_int = int(n)
                    if n_int == 0 or n_int == -1:
                        is_optional = (
                            (etype == "TETRA10" and j >= 4) or
                            (etype == "BRIC20" and j >= 8) or
                            (etype == "SHEL16" and j >= 8) or
                            (etype == "BEAM" and j >= 2) or
                            (etype == "SPRING" and j >= 2)
                        )
                        if is_optional:
                            c_nodes.append(-1)
                        elif 0 in model._id2idx:
                            c_nodes.append(model._id2idx[0])
                        else:
                            raise KeyError(0)
                    else:
                        c_nodes.append(model._id2idx[n_int])
                if len(c_nodes) < max_nodes:
                    c_nodes.extend([-1] * (max_nodes - len(c_nodes)))
                valid_conn.append(c_nodes)
                valid_raw.append((eid, pid, nodes))
            except KeyError as exc:
                log.error(f"/{etype} {eid}: unknown node id {exc}",
                          "ELEMENT CHECK")
        if not valid_raw:
            continue

        raw = valid_raw
        model.raw_elems[etype] = raw
        ids = np.array([t[0] for t in raw], dtype=np.int64)
        part_ids = np.array([t[1] for t in raw], dtype=np.int64)
        conn = np.array(valid_conn, dtype=np.int64)

        # part index + per-part slices with resolved (mat, prop)
        part_idx = np.zeros(len(raw), dtype=np.int64)
        slices = []
        start = 0
        for pid in np.unique(part_ids):
            sel = np.where(part_ids == pid)[0]
            end = start + len(sel)                   # contiguous after sort
            part = model.parts.get(int(pid))
            if part is None:
                log.error(f"/{etype}: part {pid} not defined", "PART CHECK")
                start = end
                continue
            if part not in model.parts_list:
                model.parts_list.append(part)
            mat = model.materials.get(part.mat_id)
            prop = model.properties.get(part.prop_id)
            if prop is None:
                log.error(f"/PART/{pid}: property {part.prop_id} not defined",
                          "PART CHECK")
            elif not prop_reader.prop_type_ok(req_prop, prop):
                log.error(f"/PART/{pid}: /{etype} elements need /PROP/TYPE"
                          f"{req_prop}, got TYPE{prop.type}", "PART CHECK")
            # material resolution incl. the mat_ID = 0 rule (hm_read_part.F,
            # M38): a spring property (TYPE4/8/13...) may legally carry
            # mat_ID 0 — a fictitious material is assigned for the spring
            # elements; every material-required property (solids, shells,
            # trusses, beams) still needs a defined material.
            if mat is None:
                if part.mat_id == 0 and prop is not None \
                        and not prop_reader.material_required(prop.type):
                    mat = _fictitious_spring_material()
                else:
                    log.error(f"/PART/{pid}: material {part.mat_id} not "
                              f"defined", "PART CHECK")
            if mat is not None and prop is not None:
                slices.append((slice(start, end), mat, prop))
            part_idx[sel] = model.parts_list.index(part) if part in \
                model.parts_list else 0
            start = end

        group = ElementGroup(ids=ids, conn=conn, part=part_idx)
        group.state["slices"] = slices
        group.state["part_ids"] = part_ids
        setattr(model, attr, group)

    _dispatch_solid_formulations(model, log)
    _dispatch_tetra_formulations(model, log)
    _dispatch_penta_formulations(model, log)
    _dispatch_quad_formulations(model, log)
    _dispatch_shell_formulations(model, log)
    _dispatch_sh3n_formulations(model, log)
    _dispatch_beam_formulations(model, log)



def _subset_element_group(src: ElementGroup, mask: np.ndarray) -> ElementGroup:
    """Row subset of an ElementGroup, slices rebuilt. Parts are
    contiguous after the part sort and each part routes WHOLLY to one
    formulation (the Ishell lives on the /PROP), so taking whole slices
    preserves both the ordering and the per-part contiguity."""
    g = ElementGroup(ids=src.ids[mask], conn=src.conn[mask],
                     part=src.part[mask])
    slices = []
    start = 0
    for sl, mat, prop in src.state["slices"]:
        cnt = int(mask[sl].sum())
        if cnt:
            slices.append((slice(start, start + cnt), mat, prop))
            start += cnt
    g.state["slices"] = slices
    g.state["part_ids"] = src.state["part_ids"][mask]
    return g


def _dispatch_solid_formulations(model: Model, log: MessageLog) -> None:
    """Solid element-technology dispatch (M64): split /BRICK parts whose
    /PROP/SOLID Isolid selects a dedicated formulation kernel out of the
    generic Belytschko-Tsay group, per elements.SOLID_ISOLID_GROUPS.
    Decks without such parts are left alone."""
    from ..elements import SOLID_ISOLID_GROUPS
    src = model.bricks
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        isolid = int(prop.params.get("isolid", 0) or 0)
        ptype = getattr(prop, "type", 14)
        gname = SOLID_ISOLID_GROUPS.get(isolid)
        if gname is None and ptype in (20, 21, 22):
            gname = "tshells"
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        kname = gname.split('_', 1)[1].upper() if '_' in gname else gname.upper()
        log.info(f"     {int(mask.sum())} /BRICK ELEMENT(S) ROUTED TO THE "
                 f"{kname} FORMULATION KERNEL "
                 f"(Isolid dispatch)")
    model.bricks = _subset_element_group(src, keep) if keep.any() else None


def _dispatch_tetra_formulations(model: Model, log: MessageLog) -> None:
    """Tetrahedral element-technology dispatch: split /TETRA4 parts whose
    property Itetra4 selects a dedicated formulation kernel (e.g. 3 = SFEM)."""
    from ..elements import TETRA4_ITETRA4_GROUPS
    src = model.tetras
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        itetra4 = int(prop.params.get("itetra4", 0) or 0)
        gname = TETRA4_ITETRA4_GROUPS.get(itetra4)
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        kname = gname.split('_', 1)[1].upper() if '_' in gname else gname.upper()
        log.info(f"     {int(mask.sum())} /TETRA4 ELEMENT(S) ROUTED TO THE "
                 f"{kname} FORMULATION KERNEL (Itetra4 dispatch)")
    model.tetras = _subset_element_group(src, keep) if keep.any() else None


def _dispatch_penta_formulations(model: Model, log: MessageLog) -> None:
    """Pentahedral element-technology dispatch: split /PENTA6 parts whose
    property Isolid selects HEPH physical stabilization (Isolid=24)."""
    from ..elements import PENTA_ISOLID_GROUPS
    src = model.penta6s
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        isolid = int(prop.params.get("isolid", 0) or 0)
        gname = PENTA_ISOLID_GROUPS.get(isolid)
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        kname = gname.split('_', 1)[1].upper() if '_' in gname else gname.upper()
        log.info(f"     {int(mask.sum())} /PENTA6 ELEMENT(S) ROUTED TO THE "
                 f"{kname} FORMULATION KERNEL (Isolid dispatch)")
    model.penta6s = _subset_element_group(src, keep) if keep.any() else None


def _dispatch_quad_formulations(model: Model, log: MessageLog) -> None:
    """Quad 2D element-technology dispatch: split /QUAD parts whose
    property Iquad selects full 2x2 Gauss integration (Iquad=2)."""
    from ..elements import QUAD_IQUAD_GROUPS
    src = model.quads
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        iquad = int(prop.params.get("iquad", 0) or prop.params.get("isolid", 0) or 0)
        gname = QUAD_IQUAD_GROUPS.get(iquad)
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        kname = gname.split('_', 1)[1].upper() if '_' in gname else gname.upper()
        log.info(f"     {int(mask.sum())} /QUAD ELEMENT(S) ROUTED TO THE "
                 f"{kname} FORMULATION KERNEL (Iquad dispatch)")
    model.quads = _subset_element_group(src, keep) if keep.any() else None



def _dispatch_shell_formulations(model: Model, log: MessageLog) -> None:
    """Shell element-technology dispatch (M41): split /SHELL parts whose
    /PROP/SHELL Ishell selects a dedicated formulation kernel out of the
    generic Belytschko-Tsay group, per elements.SHELL_ISHELL_GROUPS
    (12 = QBAT -> model.shells_qbat). Decks without such parts are left
    byte-identically alone (the split never runs)."""
    from ..elements import SHELL_ISHELL_GROUPS
    src = model.shells
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        ishell = int(prop.params.get("ishell", 0) or 0)
        gname = SHELL_ISHELL_GROUPS.get(ishell)
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        log.info(f"     {int(mask.sum())} /SHELL ELEMENT(S) ROUTED TO THE "
                 f"{gname.split('_', 1)[1].upper()} FORMULATION KERNEL "
                 f"(Ishell dispatch)")
    model.shells = _subset_element_group(src, keep) if keep.any() else None


def _dispatch_sh3n_formulations(model: Model, log: MessageLog) -> None:
    """SH3N element-technology dispatch: split /SH3N parts whose
    /PROP/SHELL Ish3n selects a dedicated formulation kernel out of the
    generic shell_tri3 group, per elements.SH3N_ISHELL_GROUPS.
    Decks without such parts are left alone."""
    from ..elements import SH3N_ISHELL_GROUPS
    src = model.sh3n
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        ish3n = int(prop.params.get("ish3n", 0) or 0)
        gname = SH3N_ISHELL_GROUPS.get(ish3n)
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        log.info(f"     {int(mask.sum())} /SH3N ELEMENT(S) ROUTED TO THE "
                 f"{gname.split('_', 1)[1].upper()} FORMULATION KERNEL "
                 f"(Ish3n dispatch)")
    model.sh3n = _subset_element_group(src, keep) if keep.any() else None


def _dispatch_beam_formulations(model: Model, log: MessageLog) -> None:
    """Beam element-technology dispatch (M593): split /BEAM parts whose
    property is /PROP/TYPE18 (/PROP/INT_BEAM) out of the generic beam group
    into model.beams_fiber (pyradioss.elements.beam_fiber)."""
    src = model.beams
    if src is None or not src.n:
        return
    mask = np.zeros(src.n, dtype=bool)
    slices = src.state.get("slices", [])
    for sl, mat, prop in slices:
        pt = getattr(prop, "type", 0)
        if pt == 18:
            mask[sl] = True
    if not mask.any():
        return
    keep = ~mask
    model.beams_fiber = _subset_element_group(src, mask)
    log.info(f"     {int(mask.sum())} /BEAM ELEMENT(S) ROUTED TO THE "
             f"INTEGRATED FIBER BEAM KERNEL (PROP/TYPE18 dispatch)")
    model.beams = _subset_element_group(src, keep) if keep.any() else None


# ----------------------------------------------------------------------------
# Material resolution: /FUNCT curve references, /FAIL attachment
# ----------------------------------------------------------------------------

def resolve_materials(model: Model, log: MessageLog) -> None:
    """Resolve everything a material references once the whole deck is
    read (deck order between /MAT, /FUNCT and /FAIL is free):

    * LAW36: pull the /FUNCT hardening curves into plain arrays in
      ``mat.params`` (curve_x/curve_y/curve_s + rates) so the Engine
      kernels never touch the function-table objects — the Fortran
      Starter does the same (curves are copied into the MLAW buffer).
      The per-curve Fscale_i (``params['yfac']``, M40) is baked into the
      copied ordinates AND slopes here: the reference applies YFAC at
      every engine evaluation (sigeps36.F ``Y1*YFAC(I,1)``,
      ``DYDX1*YFAC(I,1)`` — both value and derivative, before the
      strain-rate interpolation), which is algebraically identical to
      scaling the stored curve once;
    * /FAIL cards: attach each parsed FailureModel to its material.
    """
    for mat in model.materials.values():
        if mat.law != 36 or getattr(mat, "inactive", False):
            continue
        cxs, cys, css = [], [], []
        ok = True
        yfac = list(mat.params.get("yfac") or [])
        yfac += [1.0] * (len(mat.params["funct_ids"]) - len(yfac))
        for fid, yf in zip(mat.params["funct_ids"], yfac):
            fct = model.functions.get(fid)
            if fct is None:
                log.error(f"/MAT/LAW36/{mat.id}: function {fid} not defined",
                          "MAT CHECK")
                ok = False
                continue
            if np.any(fct.x < 0.0):
                log.error(f"/MAT/LAW36/{mat.id}: curve {fid} has negative "
                          f"plastic-strain abscissae", "MAT CHECK")
                ok = False
            if fct.eval(0.0) * yf <= 0.0:
                log.error(f"/MAT/LAW36/{mat.id}: curve {fid} gives a "
                          f"non-positive initial yield stress "
                          f"(Fscale={yf:g})", "MAT CHECK")
                ok = False
            cxs.append(fct.x.copy())
            cys.append(fct.y * yf)
            css.append(fct.slope * yf)
        if ok:
            mat.params["curve_x"] = cxs
            mat.params["curve_y"] = cys
            mat.params["curve_s"] = css
            mat.params["rates"] = np.asarray(mat.params["rates"], dtype=float)

    # LAW81 (M37 pack 2): resolve the four optional /FUNCT references
    # (K, G scale vs eps_p_vol; cohesion vs eps_p_dev; cap pressure Pb vs
    # eps_p_vol) into plain (x, y) arrays for the kernel, exactly like
    # LAW36 above.  Also surface the documented porosity cut.
    for mat in model.materials.values():
        if mat.law != 81 or getattr(mat, "inactive", False):
            continue
        if mat.params.pop("law81_porosity_ignored", False):
            log.warning(f"/MAT/LAW81/{mat.id}: pore-water block (Kw, P0r, "
                        f"sat0...) is not ported — porosity IGNORED "
                        f"(law81_druckerprager documented cut)",
                        "MAT CHECK")
        for fid, name in zip(mat.params.get("funct81_ids", []),
                             ("k", "g", "c", "pb")):
            if fid == 0:
                continue
            fct = model.functions.get(fid)
            if fct is None:
                log.error(f"/MAT/LAW81/{mat.id}: function {fid} not "
                          f"defined", "MAT CHECK")
                continue
            mat.params["curve81_" + name] = (fct.x.copy(), fct.y.copy())

    # M37 pack 1: LAW70 (loading/unloading tables + the law70_upd.F
    # derived constants), LAW35 (pressure curve) and LAW44 (tabulated
    # yield) resolve their /FUNCT references the same deck-order-free way
    for mat in model.materials.values():
        if getattr(mat, "inactive", False):
            continue
        if mat.law == 70:
            from ..materials import law70_tabfoam
            law70_tabfoam.resolve(mat, model, log)
        elif mat.law == 35:
            from ..materials import law35_kelvinmax
            law35_kelvinmax.resolve(mat, model, log)
        elif mat.law == 44:
            from ..materials import law44_cowper
            law44_cowper.resolve(mat, model, log)
        elif mat.law == 83:
            from ..materials import law83_spotweld
            law83_spotweld.resolve(mat, model, log)
        elif mat.law == 33:
            from ..materials import law33_foamplas
            law33_foamplas.resolve(mat, model, log)
        elif mat.law == 28:
            from ..materials import law28_honeycomb
            law28_honeycomb.resolve(mat, model, log)
        elif mat.law in (38, "38", "LAW38", "VISC_TAB") or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB"):
            from ..materials import law38_visc_tab
            if hasattr(law38_visc_tab, "resolve"):
                law38_visc_tab.resolve(mat, model, log)
        elif mat.law in (43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB"):
            from ..materials import law43_hill_tab
            if hasattr(law43_hill_tab, "resolve"):
                law43_hill_tab.resolve(mat, model, log)
        elif mat.law in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
            from ..materials import law60_plast3
            if hasattr(law60_plast3, "resolve"):
                law60_plast3.resolve(mat, model, log)
        elif mat.law in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
            from ..materials import law57_barlat
            if hasattr(law57_barlat, "resolve"):
                law57_barlat.resolve(mat, model, log)
        elif mat.law in (73, "73", "LAW73", "BARLAT2000", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_BARLAT2000", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "BARLAT2000", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_BARLAT2000", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
            from ..materials import law73_hill_therm
            if hasattr(law73_hill_therm, "resolve"):
                law73_hill_therm.resolve(mat, model, log)
        elif mat.law in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
            from ..materials import law66_plas_tab
            if hasattr(law66_plas_tab, "resolve"):
                law66_plas_tab.resolve(mat, model, log)
        elif mat.law in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D") or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D"):
            from ..materials import law74_hill_3d
            if hasattr(law74_hill_3d, "resolve"):
                law74_hill_3d.resolve(mat, model, log)
        elif mat.law in (88, "88", "LAW88", "HYP_TAB", "TAB_HYP", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TABULATED_HYP", "MAT_LAW88") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYP_TAB", "TAB_HYP", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TABULATED_HYP", "MAT_LAW88"):
            from ..materials import law88_tab_hyp
            if hasattr(law88_tab_hyp, "resolve"):
                law88_tab_hyp.resolve(mat, model, log)
        elif mat.law in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE"):
            from ..materials import law92_arruda_boyce
            if hasattr(law92_arruda_boyce, "resolve"):
                law92_arruda_boyce.resolve(mat, model, log)
        elif mat.law in (93, "93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
            from ..materials import law93_orth_hill
            if hasattr(law93_orth_hill, "resolve"):
                law93_orth_hill.resolve(mat, model, log)
        elif mat.law in (94, "94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH"):
            from ..materials import law94_yeoh
            if hasattr(law94_yeoh, "resolve"):
                law94_yeoh.resolve(mat, model, log)
        elif mat.law in (90, "90", "LAW90", "HYST_FOAM", "TAB_FOAM", "MAT_LAW90", "MAT_HYST_FOAM", "MAT_TAB_FOAM", "LAW90_HYST_FOAM") or getattr(mat, "law_name", None) in ("90", "LAW90", "HYST_FOAM", "TAB_FOAM", "MAT_LAW90", "MAT_HYST_FOAM", "MAT_TAB_FOAM", "LAW90_HYST_FOAM"):
            from ..materials import law90_foam
            if hasattr(law90_foam, "resolve"):
                law90_foam.resolve(mat, model, log)

    for mat_id, fm, source in model.raw_fails:
        mat = model.materials.get(mat_id)
        if mat is None:
            log.error(f"/FAIL/{fm.type}/{mat_id}: material {mat_id} not "
                      f"defined", source)
            continue
        if mat.law == 1:
            log.warning(f"/FAIL/{fm.type}/{mat_id}: attached to elastic "
                        f"LAW1 — no plastic strain ever accumulates, the "
                        f"criterion will never trigger", source)
        if mat.fail is not None:
            log.warning(f"/FAIL/{fm.type}/{mat_id}: material already has a "
                        f"/FAIL card — replaced", source)
        if fm.type == "TAB1":
            tid = fm.params.get("table1_id", 0)
            if tid not in model.tables:
                log.error(f"/FAIL/TAB1/{mat_id}: table {tid} not defined", source)
            else:
                fm.params["table"] = model.tables[tid]
        elif fm.type == "FLD":
            fid = fm.params.get("fct_id", 0)
            if fid not in model.functions:
                log.error(f"/FAIL/FLD/{mat_id}: function {fid} not defined", source)
            else:
                fm.params["function"] = model.functions[fid]
        elif fm.type == "TENSSTRAIN":
            fid = fm.params.get("fct_id", 0)
            if fid > 0 and fid not in model.functions:
                log.error(f"/FAIL/TENSSTRAIN/{mat_id}: function {fid} not defined", source)
            elif fid > 0:
                fm.params["function"] = model.functions[fid]
            fidel = fm.params.get("fct_idel", 0)
            if fidel > 0 and fidel not in model.functions:
                log.error(f"/FAIL/TENSSTRAIN/{mat_id}: function {fidel} not defined", source)
            elif fidel > 0:
                fm.params["function_el"] = model.functions[fidel]
            fidt = fm.params.get("fct_id_t", 0)
            if fidt > 0 and fidt not in model.functions:
                log.error(f"/FAIL/TENSSTRAIN/{mat_id}: function {fidt} not defined", source)
            elif fidt > 0:
                fm.params["function_t"] = model.functions[fidt]
        elif fm.type == "ORTHSTRAIN":
            fidel = fm.params.get("fct_idel", 0)
            if fidel > 0 and fidel not in model.functions:
                log.error(f"/FAIL/ORTHSTRAIN/{mat_id}: function {fidel} not defined", source)
            elif fidel > 0:
                fm.params["function_el"] = model.functions[fidel]
            for d in ("11", "22", "33", "12", "23", "31"):
                for suffix in ("t", "c"):
                    key = f"fct_id_{d}_{suffix}"
                    fid = fm.params.get(key, 0)
                    if fid > 0 and fid not in model.functions:
                        log.error(f"/FAIL/ORTHSTRAIN/{mat_id}: function {fid} not defined", source)
                    elif fid > 0:
                        fm.params[f"function_{d}_{suffix}"] = model.functions[fid]
                
        mat.fail = fm

    # /EOS attachment (M6, same free-order pattern as /FAIL): the EOS
    # replaces the law's pressure for SOLID elements — only meaningful
    # for laws whose deviatoric response is pressure-independent
    for mat_id, es, source in model.raw_eos:
        mat = model.materials.get(mat_id)
        if mat is None:
            log.error(f"/EOS/{es.kind}/{mat_id}: material {mat_id} not "
                      f"defined", source)
            continue
        if getattr(mat, "inactive", False):
            # M37: an /EOS on a parsed-but-not-implemented law is kept as
            # part of the record (parse-clean) — the Engine refuses the
            # material anyway, so the EOS never acts
            es.rho0 = mat.rho0
            mat.eos = es
            continue
        if mat.law not in (1, 2, 36, 44, 999):
            log.error(f"/EOS/{es.kind}/{mat_id}: an EOS can only attach "
                      f"to LAW1/LAW2/LAW36/LAW44/GAS (isotropic laws "
                      f"with a pressure-independent deviator), got "
                      f"LAW{mat.law}", source)
            continue
        if mat.eos is not None:
            log.warning(f"/EOS/{es.kind}/{mat_id}: material already has "
                        f"an /EOS card — replaced", source)
        if mat.law == 999 and mat.rho0 == 0.0 \
                and es.params.get("rho0_card", 0.0) > 0.0:
            # /MAT/GAS has no density card of its own — the /EOS/IDEAL-GAS
            # RHO_0 field supplies it (M37 pack 1, materials/mat_gas.py)
            mat.rho0 = es.params["rho0_card"]
        es.rho0 = mat.rho0
        mat.eos = es

    # M37 pack 1: /MAT/GAS element-path resolution — build the IDEAL-GAS
    # EOS from programmatic P0/T0/RHO0 params when no /EOS card gave one
    # (materials/mat_gas.resolve_gas; harmless for injector-only gases)
    for mat in model.materials.values():
        if mat.law == 999 and not getattr(mat, "inactive", False):
            from ..materials import mat_gas
            mat_gas.resolve_gas(mat, log)

    # /ALE/MAT, /EULER/MAT, /HEAT/MAT parse-only notes (M37): attach to
    # the material's params (deck order is free, like /FAIL and /EOS) —
    # accepted and remembered, no physics acts on them
    for kind, mat_id, params, source in getattr(model, "raw_mat_notes", []):
        mat = model.materials.get(mat_id)
        if mat is None:
            log.warning(f"/{kind}/{mat_id}: material {mat_id} not defined "
                        f"— note ignored", source)
            continue
        mat.params[kind.replace("/", "_").lower() + "_note"] = params

    # M147: /MAT/VISC_PLAS attachment (deck order free)
    for mat_id, vp in model.visc_plas_models.items():
        mat = model.materials.get(mat_id)
        if mat is not None:
            mat.visc_plas = vp

    # /FAIL/JOHNSON D5 needs the material's adiabatic temperature (M6)
    for mat in model.materials.values():
        if (mat.fail is not None and mat.fail.type == "JOHNSON"
                and mat.fail.params.get("D5", 0.0) != 0.0
                and "mT" not in mat.params):
            log.warning(f"/FAIL/JOHNSON/{mat.id}: D5 given but the "
                        f"material has no thermal card (LAW2 card 6) — "
                        f"D5 term dropped", "FAIL CHECK")
            mat.fail.params["D5"] = 0.0


# ----------------------------------------------------------------------------
# Node groups, element groups and surfaces
# ----------------------------------------------------------------------------

def _nodes_of_parts(model: Model, part_ids: List[int]) -> np.ndarray:
    """All node indices used by elements of the given parts."""
    out: List[np.ndarray] = []
    for _, group in model.element_groups():
        mask = np.isin(group.state["part_ids"], part_ids)
        if np.any(mask):
            u = np.unique(group.conn[mask])
            out.append(u[u >= 0])
    if not out:
        return np.zeros(0, dtype=np.int64)
    res = np.unique(np.concatenate(out))
    return res[res >= 0]


# element-group family key -> model attributes it spans (GRBRIC = ALL
# solids, like the Fortran IGRBRIC over the whole IXS; BEAM edges use
# the two END nodes only — the orientation node N3 is no geometry)
_EGROUP_FAMILIES = {
    "SHEL": ("shells", "shells_qbat", "shells_qeph", "shel16s"),
    "SH3N": ("sh3n", "sh3n_dkt18"),
    "BRIC": ("bricks", "bricks_heph", "tshells", "tetras", "tetra10s", "bric20s", "penta6s"),
    "TSHELL": ("tshells",),
    "QUAD": ("quads",),
    "TRUS": ("trusses",),
    "BEAM": ("beams", "beams_fiber"),
    "SPRI": ("springs",),
}


def _elem_row_lookup(model: Model, attr: str):
    """user element id -> row map for one element group (cached)."""
    g = getattr(model, attr, None)
    if g is None:
        return {}
    cache = g.state.get("_id2row")
    if cache is None:
        cache = {int(e): k for k, e in enumerate(g.ids)}
        g.state["_id2row"] = cache
    return cache


def _fixpoint(pending: dict, try_resolve, log: MessageLog, kind: str,
              on_cycle) -> None:
    """The iterative group-of-groups resolution of hm_grogro.F /
    hm_grogronod.F / hm_read_surfsurf.F: repeat passes resolving every
    item whose references are all resolved; when a full pass makes no
    progress the leftovers form reference CYCLES — reported as errors
    (the upstream 'ITER > N' MSGID 176/189 stop) and force-resolved
    empty so downstream consumers keep working."""
    guard = len(pending) + 1
    for _ in range(guard):
        progressed = False
        for gid in list(pending):
            if try_resolve(pending[gid]):
                del pending[gid]
                progressed = True
        if not pending:
            return
        if not progressed:
            break
    for gid, obj in pending.items():
        log.error(f"/{kind}/{gid}: circular {kind} group-of-groups "
                  f"reference (involves "
                  f"{sorted(pending)}) — group left empty", "GROUP CHECK")
        on_cycle(obj)


def resolve_generic_sets(model: Model, log: MessageLog) -> None:
    """Map /SET/GENERAL entity sets to model.node_groups and model.surfaces.

    Upstream Fortran origin: starter/source/model/sets/hm_set.F and fill_igr.F.
    /SET/GENERAL creates a node group (GRNOD) when KEY == 'NODE'.
    When KEY == 'SEG' or KEY == 'PART_E', it creates surfaces (SURF).
    """
    for set_id, gset in model.generic_sets.get("GENERAL", {}).items():
        gkey = (getattr(gset, "key", "") or "").strip().upper()
        if gkey in ("NODE", "NODENS"):
            if set_id not in model.node_groups:
                from ..model.entities import NodeGroup
                model.node_groups[set_id] = NodeGroup(
                    id=set_id,
                    title=gset.title or f"SET_{set_id}",
                    node_ids=np.array(gset.ids, dtype=np.int64),
                )
        elif gkey == "SEG":
            if set_id not in model.surfaces:
                from ..model.entities import Surface
                seg_nodes = [list(sn) for sn in getattr(gset, "seg_nodes", [])]
                if not seg_nodes and gset.ids:
                    if len(gset.ids) % 4 == 0:
                        seg_nodes = [list(gset.ids[i:i+4]) for i in range(0, len(gset.ids), 4)]
                    elif len(gset.ids) % 3 == 0:
                        seg_nodes = [[gset.ids[i], gset.ids[i+1], gset.ids[i+2], gset.ids[i+2]] for i in range(0, len(gset.ids), 3)]
                model.surfaces[set_id] = Surface(
                    id=set_id,
                    title=gset.title or f"SET_{set_id}",
                    seg_nodes=seg_nodes,
                )
        elif gkey == "PART_E":
            if set_id not in model.surfaces:
                from ..model.entities import Surface
                model.surfaces[set_id] = Surface(
                    id=set_id,
                    title=gset.title or f"SET_{set_id}",
                    part_ids=list(gset.ids),
                    modifier="EXT",
                )
        elif gkey == "PART":
            from ..model.entities import ElemGroup
            model.egroups.setdefault("PART", {})
            if set_id not in model.egroups["PART"]:
                model.egroups["PART"][set_id] = ElemGroup(
                    id=set_id,
                    family="PART",
                    title=gset.title or f"SET_{set_id}",
                    part_ids=list(gset.ids),
                )


def resolve_entity_groups(model: Model, log: MessageLog) -> None:
    """ELEMENT groups (/GRSHEL, /GRSH3N, /GRBRIC, ..., /GRPART) ->
    resolved member rows (Fortran hm_lecgre.F + hm_grogro.F).

    Each group ends with ``members`` = [(model attr, row ndarray)]
    (family 'PART': ``part_ids_resolved``).  Group-of-group references
    are signed: negative ids REMOVE the referenced group's members,
    and removal wins over addition (the upstream BUFTMP = -1 rule);
    cycles are detected by the fixpoint guard."""
    resolve_generic_sets(model, log)
    for family, groups in model.egroups.items():
        attrs = _EGROUP_FAMILIES.get(family, ())
        if family != "PART" and not attrs and any(
                g.elem_ids or g.part_ids for g in groups.values()):
            log.warning(f"/GR{family}: element family not ported — "
                        f"groups resolve empty", "GROUP CHECK")

        def _base(g):
            """direct + part content as {attr: set(rows)} / part set."""
            if family == "PART":
                for pid in g.part_ids:
                    if pid not in model.parts:
                        log.error(f"/GRPART/{g.id}: unknown part {pid}",
                                  "GROUP CHECK")
                return set(p for p in g.part_ids if p in model.parts)
            rows = {a: set() for a in attrs}
            if not attrs:            # family without a ported element
                return rows
            missing = []
            for eid in g.elem_ids:
                for a in attrs:
                    r = _elem_row_lookup(model, a).get(eid)
                    if r is not None:
                        rows[a].add(r)
                        break
                else:
                    missing.append(eid)
            if missing:
                log.error(f"/GR{family}/{g.id}: unknown element id(s) "
                          f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
                          "GROUP CHECK")
            if g.part_ids:
                for a in attrs:
                    eg = getattr(model, a, None)
                    if eg is None:
                        continue
                    mask = np.isin(eg.state["part_ids"], g.part_ids)
                    rows[a].update(np.where(mask)[0].tolist())
            return rows

        resolved: dict = {}
        base_cache: dict = {}

        def _try(g):
            if g.id not in base_cache:   # once — errors not duplicated
                base_cache[g.id] = _base(g)
            add = (set(base_cache[g.id]) if family == "PART"
                   else {a: set(v) for a, v in base_cache[g.id].items()})
            rem = set() if family == "PART" else {a: set() for a in attrs}
            for ref in g.group_ids:
                other = groups.get(abs(ref))
                if other is None:
                    log.warning(f"/GR{family}/{g.id}: unknown group "
                                f"{abs(ref)} — ignored", "GROUP CHECK")
                    continue
                if abs(ref) not in resolved:
                    return False                       # defer this pass
                if family == "PART":
                    (add if ref > 0 else rem).update(resolved[abs(ref)])
                else:
                    for a in attrs:
                        tgt = add if ref > 0 else rem
                        tgt[a].update(resolved[abs(ref)][a])
            if family == "PART":
                final = add - rem
                g.part_ids_resolved = sorted(final)
                resolved[g.id] = final
            else:
                final = {a: add[a] - rem[a] for a in attrs}
                g.members = [
                    (a, np.array(sorted(final[a]), dtype=np.int64))
                    for a in attrs if final[a]]
                resolved[g.id] = final
            return True

        def _on_cycle(g):
            if family == "PART":
                g.part_ids_resolved = []
            else:
                g.members = []

        _fixpoint(dict(groups), _try, log, f"GR{family}", _on_cycle)
        for g in groups.values():
            n = (len(g.part_ids_resolved) if family == "PART"
                 else sum(len(r) for _, r in (g.members or [])))
            if n == 0:
                log.warning(f"/GR{family}/{g.id} '{g.title}' is empty",
                            "GROUP CHECK")


def _nodes_in_box(model: Model, box, log: MessageLog,
                  who: str) -> np.ndarray:
    """Node indices inside one /BOX volume (rdbox.F membership tests,
    boundaries inclusive)."""
    x = model.x0

    def _pt(node_id, fallback):
        if node_id:
            try:
                return x[model.node_index(node_id)]
            except KeyError:
                log.error(f"{who}: /BOX/{box.id} references unknown "
                          f"node {node_id}", "GROUP CHECK")
                return None
        return fallback

    if box.kind == "RECTA":
        cmin, cmax = box.corner_min, box.corner_max
        if box.iskew:
            row = model.skews.index("SKEW", box.iskew)
            if row < 0:
                log.error(f"{who}: /BOX/{box.id} references unknown /SKEW {box.iskew}", "GROUP CHECK")
                return np.zeros(0, dtype=np.int64)
            axes = model.skews.axes[row]
            origin = model.skews.origins[row]
            x_test = (x - origin) @ axes.T
        else:
            axes = origin = None
            x_test = x
        if box.node1:
            p1 = _pt(box.node1, None)
            p2 = _pt(box.node2, None)
            if p1 is None or p2 is None:
                return np.zeros(0, dtype=np.int64)
            if box.iskew:
                p1 = (p1 - origin) @ axes.T
                p2 = (p2 - origin) @ axes.T
            cmin, cmax = np.minimum(p1, p2), np.maximum(p1, p2)
        inside = np.all((x_test >= cmin) & (x_test <= cmax), axis=1)
        return np.where(inside)[0]
    if box.kind == "SPHER":
        c = _pt(box.node1, box.p1)
        if c is None:
            return np.zeros(0, dtype=np.int64)
        d2 = np.einsum("nb,nb->n", x - c, x - c)
        return np.where(d2 <= (0.5 * box.diameter) ** 2)[0]
    # CYLIN: finite cylinder p1 -> p2, radius D/2 (INSIDE_CYLINDER)
    p1 = _pt(box.node1, box.p1)
    p2 = _pt(box.node2, box.p2)
    if p1 is None or p2 is None:
        return np.zeros(0, dtype=np.int64)
    axis = p2 - p1
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        log.error(f"{who}: /BOX/CYLIN/{box.id} has a zero-length axis",
                  "GROUP CHECK")
        return np.zeros(0, dtype=np.int64)
    a = axis / length
    d = (x - p1) @ a
    radial = x - p1 - d[:, None] * a
    r2 = np.einsum("nb,nb->n", radial, radial)
    inside = (d >= 0.0) & (d <= length) & (r2 <= (0.5 * box.diameter) ** 2)
    return np.where(inside)[0]


def resolve_node_group_base(model: Model, g, log: MessageLog) -> np.ndarray:
    """Evaluate the non-recursive content of a node group."""
    idx: List[np.ndarray] = []
    if g.node_ids is not None and len(g.node_ids) > 0:
        try:
            idx.append(model.node_indices(g.node_ids))
        except KeyError as exc:
            log.error(f"/GRNOD/{g.id}: unknown node id {exc}", "GROUP CHECK")
    if g.part_ids:
        idx.append(_nodes_of_parts(model, g.part_ids))
    for bid in g.box_ids:
        box = model.boxes.get(bid)
        if box is None:
            log.error(f"/GRNOD/{g.id}: unknown box {bid}", "GROUP CHECK")
            continue
        idx.append(_nodes_in_box(model, box, log, f"/GRNOD/{g.id}"))
    for first, last, incr in g.gene_ranges:
        uid = model.node_ids
        mask = (uid >= first) & (uid <= last)
        if incr > 1:
            mask &= (uid - first) % incr == 0
        idx.append(np.where(mask)[0])
    for sid in g.surf_ids:
        surf = model.surfaces.get(sid)
        if surf is None or surf.segments is None:
            log.error(f"/GRNOD/{g.id}: surface {sid} missing or unresolved", "GROUP CHECK")
            continue
        if surf.segments.size:
            u = np.unique(surf.segments)
            idx.append(u[u >= 0])
    for family, gid in g.egroup_refs:
        eg = model.egroups.get(family, {}).get(gid)
        if eg is None:
            log.error(f"/GRNOD/{g.id}: unknown /GR{family} group {gid}", "GROUP CHECK")
            continue
        for attr, rows in (eg.members or []):
            u = np.unique(getattr(model, attr).conn[rows])
            idx.append(u[u >= 0])
    if not idx:
        return np.zeros(0, dtype=np.int64)
    res = np.unique(np.concatenate(idx))
    return res[res >= 0]


def resolve_single_node_group(model: Model, g, log: MessageLog, visited: set = None) -> np.ndarray:
    """Evaluate a node group on-demand, resolving its base content and recursive references."""
    if getattr(g, "node_idx", None) is not None and g.node_idx.size > 0:
        return g.node_idx
        
    if visited is None:
        visited = set()
    if g.id in visited:
        return np.zeros(0, dtype=np.int64)  # cycle detected
    visited.add(g.id)
    
    add = [resolve_node_group_base(model, g, log)]
    rem = [np.zeros(0, dtype=np.int64)]
    
    for ref in g.grnod_ids:
        other = model.node_groups.get(abs(ref))
        if other is None:
            continue
        res = resolve_single_node_group(model, other, log, visited)
        (add if ref > 0 else rem).append(res)
        
    final = np.setdiff1d(np.unique(np.concatenate(add)),
                         np.unique(np.concatenate(rem)))
    return final


def resolve_node_groups(model: Model, log: MessageLog) -> None:
    """/GRNOD content -> dense node index arrays.

    M37: runs AFTER resolve_entity_groups and resolve_surfaces —
    /GRNOD/SURF takes the nodes of resolved surface segments
    (hm_surfnod.F) and /GRNOD/GR<elem> the nodes of resolved element
    groups (hm_elngr.F).  /GRNOD/GRNOD (hm_grogronod.F) resolves by
    iterative fixpoint with cycle detection; NEGATIVE references REMOVE
    the referenced group's nodes and removal wins over addition,
    whatever the order (the upstream BUFTMP = -1 convention).  Groups
    are stored sorted by node index, like the upstream sorted groups.
    """
    resolve_generic_sets(model, log)
    resolved: dict = {}
    base_cache: dict = {}

    def _try(g) -> bool:
        if g.id not in base_cache:       # once — errors not duplicated
            base_cache[g.id] = resolve_node_group_base(model, g, log)
        add = [base_cache[g.id]]
        rem = [np.zeros(0, dtype=np.int64)]
        for ref in g.grnod_ids:
            other = model.node_groups.get(abs(ref))
            if other is None:
                # upstream MSGID 174: unknown reference is a WARNING
                log.warning(f"/GRNOD/{g.id}: unknown node group "
                            f"{abs(ref)} — ignored", "GROUP CHECK")
                continue
            if abs(ref) not in resolved:
                return False                           # defer this pass
            (add if ref > 0 else rem).append(resolved[abs(ref)])
        final = np.setdiff1d(np.unique(np.concatenate(add)),
                             np.unique(np.concatenate(rem)))
        g.node_idx = final
        resolved[g.id] = final
        return True

    def _on_cycle(g):
        g.node_idx = np.zeros(0, dtype=np.int64)

    _fixpoint(dict(model.node_groups), _try, log, "GRNOD", _on_cycle)
    for g in model.node_groups.values():
        if g.node_idx is None or g.node_idx.size == 0:
            log.warning(f"/GRNOD/{g.id} '{g.title}' is empty", "GROUP CHECK")


def _free_faces_of_bricks(model: Model, part_ids: List[int], modifier: str = "EXT"):
    """Outer (free) faces of the given solid parts (or all faces if modifier='ALL').
    Supports bricks, bricks_heph, bric20s, and shel16s.
    Returns (faces (n,4), parent element rows (n,), element group names (n,))."""
    from ..elements.solid_hexa8 import _FACES
    all_faces: List[np.ndarray] = []
    all_owners: List[np.ndarray] = []
    all_attrs: List[np.ndarray] = []

    for attr in ("bricks", "bricks_heph", "tshells", "bric20s", "shel16s"):
        g = getattr(model, attr, None)
        if g is None:
            continue
        mask = np.isin(g.state["part_ids"], part_ids)
        if not np.any(mask):
            continue
        erow = np.where(mask)[0]
        # First 8 nodes define corner vertices
        conn = g.conn[mask, :8]
        faces = conn[:, _FACES.reshape(-1)].reshape(-1, 4)
        owner = np.repeat(erow, 6)
        key = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        free = (counts[inverse] == 1) if modifier != "ALL" else np.ones(len(faces), dtype=bool)
        if np.any(free):
            all_faces.append(faces[free])
            all_owners.append(owner[free])
            all_attrs.append(np.full(int(np.sum(free)), attr, dtype="<U16"))

    if not all_faces:
        return (np.zeros((0, 4), dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype="<U16"))
    return np.vstack(all_faces), np.concatenate(all_owners), np.concatenate(all_attrs)


def _free_faces_of_tetras(model: Model, part_ids: List[int], modifier: str = "EXT"):
    """Free triangular faces of /TETRA4 and /TETRA10 parts (or all faces if modifier='ALL'),
    as degenerate 4-node segments (3rd node repeated — Radioss triangle-segment convention).
    Returns (faces (n,4), parent element rows (n,), element group names (n,))."""
    from ..elements.solid_tetra4 import _FACES
    all_faces: List[np.ndarray] = []
    all_owners: List[np.ndarray] = []
    all_attrs: List[np.ndarray] = []

    for attr in ("tetras", "tetra10s"):
        g = getattr(model, attr, None)
        if g is None:
            continue
        mask = np.isin(g.state["part_ids"], part_ids)
        if not np.any(mask):
            continue
        erow = np.where(mask)[0]
        # First 4 nodes define corner vertices
        conn = g.conn[mask, :4]
        faces = conn[:, _FACES.reshape(-1)].reshape(-1, 3)
        owner = np.repeat(erow, 4)
        key = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        free = (counts[inverse] == 1) if modifier != "ALL" else np.ones(len(faces), dtype=bool)
        if np.any(free):
            ff = faces[free]
            all_faces.append(np.column_stack([ff, ff[:, 2]]))  # n4 = n3
            all_owners.append(owner[free])
            all_attrs.append(np.full(int(np.sum(free)), attr, dtype="<U16"))

    if not all_faces:
        return (np.zeros((0, 4), dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype="<U16"))
    return np.vstack(all_faces), np.concatenate(all_owners), np.concatenate(all_attrs)


def _free_faces_of_wedges(model: Model, part_ids: List[int], modifier: str = "EXT"):
    """Outer (free) faces of /PENTA6 wedge parts (or all faces if modifier='ALL'),
    as 4-node segments (triangles degenerate with 3rd node repeated).
    Returns (faces (n,4), parent element rows (n,), element group names (n,))."""
    from ..elements.solid_penta6 import _FACES
    all_faces: List[np.ndarray] = []
    all_owners: List[np.ndarray] = []
    all_attrs: List[np.ndarray] = []

    attr = "penta6s"
    g = getattr(model, attr, None)
    if g is not None:
        mask = np.isin(g.state["part_ids"], part_ids)
        if np.any(mask):
            erow = np.where(mask)[0]
            conn = g.conn[mask, :6]
            faces = conn[:, _FACES.reshape(-1)].reshape(-1, 4)
            owner = np.repeat(erow, 5)
            key = np.sort(faces, axis=1)
            _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
            free = (counts[inverse] == 1) if modifier != "ALL" else np.ones(len(faces), dtype=bool)
            if np.any(free):
                all_faces.append(faces[free])
                all_owners.append(owner[free])
                all_attrs.append(np.full(int(np.sum(free)), attr, dtype="<U16"))

    if not all_faces:
        return (np.zeros((0, 4), dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype="<U16"))
    return np.vstack(all_faces), np.concatenate(all_owners), np.concatenate(all_attrs)


def resolve_surfaces(model: Model, log: MessageLog) -> None:
    """/SURF content -> (nseg, 4) node-index arrays + per-segment
    provenance (parent element group/row, see Surface docstring).

    Supports shells, quads, triangles, solids (bricks, tetras, high-order),
    element groups (/SURF/GR*), selectors (/SURF/MAT, /PROP, /BOX, ALL),
    and surface-of-surfaces (/SURF/SURF)."""
    resolve_generic_sets(model, log)
    def _base(s):
        segs: List[np.ndarray] = []
        gtypes: List[np.ndarray] = []   # parallel provenance pieces
        elems: List[np.ndarray] = []

        def _add(seg_arr, gtype, elem_rows):
            segs.append(seg_arr)
            gtypes.append(np.full(len(seg_arr), gtype, dtype="<U16"))
            elems.append(np.asarray(elem_rows, dtype=np.int64))

        for row in s.seg_nodes:
            try:
                # explicit /SURF/SEG segments have no parent element: they
                # are never dropped by element deletion (gtype '')
                _add(model.node_indices(row)[None, :], "", [-1])
            except KeyError as exc:
                log.error(f"/SURF/{s.id}: unknown node id {exc}",
                          "SURFACE CHECK")

        # Resolve part selectors: s.part_ids, s.mat_ids, s.prop_ids, modifier == ALL
        parts_to_include = set(s.part_ids)
        if getattr(s, "mat_ids", None):
            for pid, part in model.parts.items():
                if part.mat_id in s.mat_ids:
                    parts_to_include.add(pid)
        if getattr(s, "prop_ids", None):
            for pid, part in model.parts.items():
                if part.prop_id in s.prop_ids:
                    parts_to_include.add(pid)
        if getattr(s, "modifier", "") == "ALL" and not parts_to_include and not s.seg_nodes and not s.egroup_refs and not s.surf_ids:
            parts_to_include = set(model.parts.keys())

        if parts_to_include:
            pids = list(parts_to_include)
            # shell parts
            for attr in ("shells", "shells_qbat", "shells_qeph"):
                g = getattr(model, attr, None)
                if g is not None:
                    mask = np.isin(g.state["part_ids"], pids)
                    if np.any(mask):
                        _add(g.conn[mask, :4], attr, np.where(mask)[0])
            # quads
            if getattr(model, "quads", None) is not None:
                mask = np.isin(model.quads.state["part_ids"], pids)
                if np.any(mask):
                    _add(model.quads.conn[mask, :4], "quads", np.where(mask)[0])
            # triangles
            for attr in ("sh3n", "sh3n_dkt18"):
                g = getattr(model, attr, None)
                if g is not None:
                    mask = np.isin(g.state["part_ids"], pids)
                    if np.any(mask):
                        c3 = g.conn[mask]
                        _add(np.column_stack([c3[:, :3], c3[:, 2]]), attr, np.where(mask)[0])
            # solid parts: bricks, tetras, bric20, shel16, tetra10
            mod = getattr(s, "modifier", "EXT") or "EXT"
            ff, fo, fa = _free_faces_of_bricks(model, pids, mod)
            if len(ff):
                for attr in np.unique(fa):
                    sel = (fa == attr)
                    _add(ff[sel], attr, fo[sel])
            ft, to, ta = _free_faces_of_tetras(model, pids, mod)
            if len(ft):
                for attr in np.unique(ta):
                    sel = (ta == attr)
                    _add(ft[sel], attr, to[sel])
            fw, wo, wa = _free_faces_of_wedges(model, pids, mod)
            if len(fw):
                for attr in np.unique(wa):
                    sel = (wa == attr)
                    _add(fw[sel], attr, wo[sel])

        # /SURF/GRSHEL | /SURF/GRSH3N | /SURF/GRBRIC
        for family, gid in s.egroup_refs:
            eg = model.egroups.get(family, {}).get(gid)
            if eg is None:
                log.error(f"/SURF/{s.id}: unknown /GR{family} group {gid}",
                          "SURFACE CHECK")
                continue
            for attr, rows in (eg.members or []):
                g = getattr(model, attr, None)
                if g is None:
                    continue
                conn = g.conn[rows]
                mod = getattr(s, "modifier", "EXT") or "EXT"
                if attr in ("tetras", "tetra10s"):
                    from ..elements.solid_tetra4 import _FACES
                    faces = conn[:, :4][:, _FACES.reshape(-1)].reshape(-1, 3)
                    owner = np.repeat(rows, 4)
                    key = np.sort(faces, axis=1)
                    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
                    free = (cnt[inv] == 1) if mod != "ALL" else np.ones(len(faces), dtype=bool)
                    if np.any(free):
                        ff = faces[free]
                        _add(np.column_stack([ff, ff[:, 2]]), attr, owner[free])
                elif attr == "penta6s":
                    from ..elements.solid_penta6 import _FACES
                    faces = conn[:, :6][:, _FACES.reshape(-1)].reshape(-1, 4)
                    owner = np.repeat(rows, 5)
                    key = np.sort(faces, axis=1)
                    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
                    free = (cnt[inv] == 1) if mod != "ALL" else np.ones(len(faces), dtype=bool)
                    if np.any(free):
                        _add(faces[free], attr, owner[free])
                elif attr in ("bricks", "bricks_heph", "tshells", "bric20s", "shel16s"):
                    from ..elements.solid_hexa8 import _FACES
                    faces = conn[:, :8][:, _FACES.reshape(-1)].reshape(-1, 4)
                    owner = np.repeat(rows, 6)
                    key = np.sort(faces, axis=1)
                    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
                    free = (cnt[inv] == 1) if mod != "ALL" else np.ones(len(faces), dtype=bool)
                    if np.any(free):
                        _add(faces[free], attr, owner[free])
                else:
                    if conn.shape[1] == 3:              # triangles: n4 = n3
                        c = np.column_stack([conn, conn[:, 2]])
                    else:
                        c = conn[:, :4]
                    _add(c, attr, rows)

        if getattr(s, "box_ids", None) and segs:
            box_nodes = set()
            for bid in s.box_ids:
                box = model.boxes.get(bid)
                if box:
                    bn = _nodes_in_box(model, box, log, f"/SURF/{s.id}")
                    box_nodes.update(bn)
            if not box_nodes:
                return (np.zeros((0, 4), dtype=np.int64),
                        np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))
            all_s = np.vstack(segs)
            all_gt = np.concatenate(gtypes)
            all_el = np.concatenate(elems)
            in_box = np.isin(all_s, list(box_nodes)).all(axis=1)
            return (all_s[in_box], all_gt[in_box], all_el[in_box])

        if segs:
            return (np.vstack(segs), np.concatenate(gtypes),
                    np.concatenate(elems))
        return (np.zeros((0, 4), dtype=np.int64),
                np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))

    resolved: dict = {}
    base_cache: dict = {}

    def _finish(s, triple):
        s.segments, s.seg_gtype, s.seg_elem = triple
        resolved[s.id] = triple

    def _try(s) -> bool:
        if s.id not in base_cache:       # once — errors not duplicated
            base_cache[s.id] = _base(s)
        pieces = [base_cache[s.id]]
        for ref in s.surf_ids:
            other = model.surfaces.get(abs(ref))
            if other is None:
                log.warning(f"/SURF/{s.id}: unknown surface {abs(ref)} — "
                            f"ignored", "SURFACE CHECK")
                continue
            if abs(ref) not in resolved:
                return False                        # defer this pass
            seg, gt, el = resolved[abs(ref)]
            if ref < 0:
                # negative id: node order reversed — the normal flips
                # (hm_read_surfsurf.F NODES(L,4..1))
                seg = seg[:, ::-1]
            pieces.append((seg, gt, el))
        _finish(s, (np.vstack([p[0] for p in pieces]),
                    np.concatenate([p[1] for p in pieces]),
                    np.concatenate([p[2] for p in pieces])))
        return True

    def _on_cycle(s):
        _finish(s, (np.zeros((0, 4), dtype=np.int64),
                    np.zeros(0, dtype="<U8"),
                    np.zeros(0, dtype=np.int64)))

    _fixpoint(dict(model.surfaces), _try, log, "SURF", _on_cycle)
    for s in model.surfaces.values():
        is_analytical = (
            getattr(s, "plane_p1", None) is not None or
            getattr(s, "ellipse_center", None) is not None or
            getattr(s, "cyl_center", None) is not None or
            getattr(s, "spher_center", None) is not None or
            getattr(s, "cyl_radius", 0.0) > 0.0 or
            getattr(s, "spher_radius", 0.0) > 0.0
        )
        if is_analytical:
            continue
        if s.segments.shape[0] == 0:
            log.warning(f"/SURF/{s.id} '{s.title}' has no segments",
                        "SURFACE CHECK")


# 4-node segment -> its 4 edges; a triangle segment (n4 = n3) yields the
# degenerate edge (n3, n3), filtered out below.
_SEG_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


def _surface_edges(model: Model, ln, surf_ids, log: MessageLog):
    """All edges of the listed surfaces' segments, with provenance —
    the raw union LINEDGE starts from (degenerate triangle edges
    dropped).  Returns (edges (n,2), gtype (n,), elem (n,))."""
    edges, gtypes, elems = [], [], []
    for sid in surf_ids:
        surf = model.surfaces.get(sid)
        if surf is None or surf.segments is None:
            log.error(f"/LINE/{ln.id}: surface {sid} not defined",
                      "LINE CHECK")
            continue
        e = surf.segments[:, _SEG_EDGES.reshape(-1)].reshape(-1, 2)
        own_g = np.repeat(surf.seg_gtype, 4)
        own_e = np.repeat(surf.seg_elem, 4)
        keep = e[:, 0] != e[:, 1]        # drop degenerate triangle edge
        edges.append(e[keep])
        gtypes.append(own_g[keep])
        elems.append(own_e[keep])
    if not edges:
        return (np.zeros((0, 2), dtype=np.int64),
                np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))
    return (np.vstack(edges), np.concatenate(gtypes),
            np.concatenate(elems))


def resolve_lines(model: Model, log: MessageLog) -> None:
    """/LINE content -> (nseg, 2) edge node-index arrays + provenance.
    Must run AFTER resolve_surfaces (LINE/SURF|EDGE read resolved
    segments).

    Fortran: hm_read_lines.F builds IGRSLIN from a surface (all unique
    edges — /LINE/SURF), from a surface's BORDER (/LINE/EDGE, M37:
    ``linedge.F`` keeps only edges used by exactly ONE segment of the
    union of the listed surfaces, removing interior edges entirely),
    from other lines (/LINE/LINE, M37: concatenation by fixpoint with
    cycle detection), from 1-D element parts (/LINE/PART, M37) and from
    explicit node pairs (/LINE/SEG)."""
    def _base(ln):
        edges: List[np.ndarray] = []
        gtypes: List[np.ndarray] = []
        elems: List[np.ndarray] = []
        if ln.surf_ids:
            # every unique edge: interior edges appear once per adjacent
            # segment — keep one copy (both copies are identical springs)
            e, gt, el = _surface_edges(model, ln, ln.surf_ids, log)
            if len(e):
                _, first = np.unique(np.sort(e, axis=1), axis=0,
                                     return_index=True)
                edges.append(e[first])
                gtypes.append(gt[first])
                elems.append(el[first])
        if ln.edge_surf_ids:
            # /LINE/EDGE: BORDER edges only — count over the union of
            # the listed surfaces' segments, keep count == 1 (linedge.F
            # 'removal of internal segments except borders'); stored
            # (lo, hi) like the upstream sorted pairs
            e, gt, el = _surface_edges(model, ln, ln.edge_surf_ids, log)
            if len(e):
                key = np.sort(e, axis=1)
                _, first, counts = np.unique(
                    key, axis=0, return_index=True, return_counts=True)
                border = first[counts == 1]
                edges.append(key[border])
                gtypes.append(gt[border])
                elems.append(el[border])
        for pid in ln.part_ids:
            # /LINE/PART: every 1-D element of the part is an edge
            # (beams: end nodes N1-N2 — the orientation node carries no
            # geometry)
            if pid not in model.parts:
                log.error(f"/LINE/{ln.id}: unknown part {pid}",
                          "LINE CHECK")
                continue
            found = False
            for attr in ("trusses", "springs", "beams"):
                eg = getattr(model, attr, None)
                if eg is None:
                    continue
                mask = np.isin(eg.state["part_ids"], [pid])
                if np.any(mask):
                    edges.append(eg.conn[mask][:, :2])
                    gtypes.append(np.full(int(mask.sum()), attr,
                                          dtype="<U8"))
                    elems.append(np.where(mask)[0])
                    found = True
            if not found:
                log.warning(f"/LINE/{ln.id}: part {pid} has no 1-D "
                            f"(truss/beam/spring) elements", "LINE CHECK")
        for row in ln.seg_nodes:
            try:
                edges.append(model.node_indices(row)[None, :])
                gtypes.append(np.array([""], dtype="<U8"))
                elems.append(np.array([-1], dtype=np.int64))
            except KeyError as exc:
                log.error(f"/LINE/{ln.id}: unknown node id {exc}",
                          "LINE CHECK")
        if edges:
            return (np.vstack(edges), np.concatenate(gtypes),
                    np.concatenate(elems))
        return (np.zeros((0, 2), dtype=np.int64),
                np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))

    resolved: dict = {}
    base_cache: dict = {}

    def _finish(ln, triple):
        ln.segments, ln.seg_gtype, ln.seg_elem = triple
        resolved[ln.id] = triple

    def _try(ln) -> bool:
        if ln.id not in base_cache:      # once — errors not duplicated
            base_cache[ln.id] = _base(ln)
        pieces = [base_cache[ln.id]]
        for ref in ln.line_ids:
            other = model.lines.get(abs(ref))
            if other is None:
                log.warning(f"/LINE/{ln.id}: unknown line {abs(ref)} — "
                            f"ignored", "LINE CHECK")
                continue
            if abs(ref) not in resolved:
                return False                        # defer this pass
            seg, gt, el = resolved[abs(ref)]
            if ref < 0:
                seg = seg[:, ::-1]                  # reversed direction
            pieces.append((seg, gt, el))
        _finish(ln, (np.vstack([p[0] for p in pieces]),
                     np.concatenate([p[1] for p in pieces]),
                     np.concatenate([p[2] for p in pieces])))
        return True

    def _on_cycle(ln):
        _finish(ln, (np.zeros((0, 2), dtype=np.int64),
                     np.zeros(0, dtype="<U8"),
                     np.zeros(0, dtype=np.int64)))

    _fixpoint(dict(model.lines), _try, log, "LINE", _on_cycle)
    for ln in model.lines.values():
        if ln.segments.shape[0] == 0:
            log.warning(f"/LINE/{ln.id} '{ln.title}' has no edges",
                        "LINE CHECK")


# ----------------------------------------------------------------------------
# Reference systems (/SKEW, /FRAME) and the consumers that name one  (M39)
# ----------------------------------------------------------------------------

def resolve_skews(model: Model, log: MessageLog) -> None:
    """Build every /SKEW and /FRAME from the initial geometry, then bind
    each consumer's ``skew_ID``/``frame_ID`` to its row.

    Fortran origin: ``starter/source/tools/skew/hm_read_skw.F`` +
    ``hm_read_frm.F`` build the frames; each consumer's reader then walks
    ``ISKN(4,*)`` to turn the USER id into the array index it stores (e.g.
    hm_read_rbody.F 246-256, hm_read_prop08.F 141-149,
    hm_read_inivel.F 415-448).  An unknown id is a hard error there
    (ANCMSG 184/490) and here.

    Must run after the nodes exist (the MOV frames read x0) and before
    anything consumes a frame.
    """
    model.skews.resolve(model, log)

    def _bind(kind: str, user_id: int, who: str) -> int:
        """USER skew/frame id -> row; -1 (and an error) when unknown."""
        row = model.skews.index(kind, user_id)
        if row < 0:
            log.error(f"{who}: unknown {kind.lower()}_ID {user_id}",
                      f"{kind} CHECK")
        return row

    # ---- /BCS: the fixed DOFs are the skew's axes (bcs1.F) --------------
    for bc in model.bcs:
        bc.skew_row = _bind("SKEW", bc.skew_id, f"/BCS/{bc.id}") \
            if bc.skew_id else 0

    # ---- /ALE/BCS: the fixed DOFs are the skew's axes (alewdx_grid_bcs.F) --
    for bc in model.ale_bcs:
        bc.skew_row = _bind("SKEW", bc.skew_id, f"/ALE/BCS/{bc.id}") \
            if bc.skew_id else 0

    # ---- /IMPVEL + /IMPDISP + /IMPACC: the imposed DOF is a skew axis (fixvel.F) --
    for im in list(model.impvel) + list(model.impdisp) + list(model.impacc):
        im.skew_row = _bind("SKEW", im.skew_id,
                            f"/IMP*/{im.id}") if im.skew_id else 0

    # ---- /RBODY: Jxx/Jyy/Jzz are written in the skew (inirby.F CHBAS) ---
    for rb in model.rbodies:
        rb.skew_row = _bind("SKEW", rb.skew_id,
                            f"/RBODY/{rb.id}") if rb.skew_id else 0

    # ---- /PROP TYPE8: the spring's 6 DOFs are the skew's axes (r2def3) --
    for prop in model.properties.values():
        sid = int(prop.params.get("skew_id", 0) or 0)
        if not sid:
            continue
        ptype = int(getattr(prop, "type", 0) or 0)
        who = f"/PROP/{prop.id}"
        row = _bind("SKEW", sid, who)
        prop.params["skew_row"] = max(row, 0)
        if ptype == 13:
            # TYPE13 (SPR_BEAM) uses its skew as the INITIAL frame of a
            # co-rotational beam, not as a fixed one — the port's TYPE13
            # frame is the element frame (spring_general.py) and no corpus
            # deck puts a skew on a TYPE13, so this stays a loud defer.
            log.warning(f"{who}: TYPE13 (SPR_BEAM) skew_ID={sid} NOT "
                        f"PORTED — the port uses the element frame "
                        f"(N1->N2); the physics DEVIATES when the skew is "
                        f"not aligned with it", "PROP CHECK")

    # ---- /INIVEL: skew transformation (M39) --------------------------
    for iv in model.inivel:
        if getattr(iv, "iskew", 0) and getattr(iv, "kind", "") in ("TRANS", "TRA") and not getattr(iv, "_v_is_global", False):
            row = model.skews.index("SKEW", iv.iskew)
            if row is not None and row >= 0:
                iv.v = model.skews.to_global(row, iv.v)
                iv._v_is_global = True

    # ---- /INIVEL/AXIS: the rotation axis + origin come from the /FRAME --
    for iv in model.inivel:
        if iv.kind != "AXIS" or not iv.frame_id:
            continue
        row = _bind("FRAME", iv.frame_id, f"/INIVEL/AXIS/{iv.id}")
        if row < 0:
            continue
        # hm_read_inivel.F 581-598: the axis is the frame's IDIR axis and
        # the rotation is about the FRAME ORIGIN; 437-439: the card's
        # Vxt/Vyt/Vzt are components IN the frame, rotated to global.
        iv.axis = model.skews.axes[row][iv.dir - 1].copy()
        iv.origin = model.skews.origins[row].copy()
        if not getattr(iv, "_v_is_global", False):
            iv.v = model.skews.to_global(row, iv.v)
            iv._v_is_global = True
        sf = next((s for s in model.skews.entries
                   if s.kind == "FRAME" and s.id == iv.frame_id), None)
        if sf is not None and sf.imov != 0:
            # a MOVING frame is rebuilt every cycle by movfram.F; as an
            # /INIVEL it is only ever read ONCE, at t=0, so the initial
            # orientation IS the whole story and nothing is lost here.
            log.info(f"     /INIVEL/AXIS/{iv.id}: /FRAME/{sf.subtype}/"
                     f"{sf.id} is a MOVING frame; an initial condition "
                     f"reads it only at t=0, so its initial orientation "
                     f"is used (exact)")


# ----------------------------------------------------------------------------
# Element initialization + lumped mass (inimass)
# ----------------------------------------------------------------------------

def initialize_elements_and_mass(model: Model, log: MessageLog) -> None:
    """Run every kernel's init_group and assemble the lumped nodal mass
    (and rotational inertia, from shells). Also applies /ADMAS and
    /INIVEL (TRA and AXIS)."""
    model.x = model.x0.copy()
    model.v = np.zeros((model.numnod, 3))
    model.vr = np.zeros((model.numnod, 3))
    model.mass = np.zeros(model.numnod)
    model.inertia = np.zeros(model.numnod)

    for name, group in model.element_groups():
        node_idx, mass_c, inertia_c = KERNELS[name].init_group(
            group, model, log)
        if len(node_idx) > 0:
            valid = (node_idx >= 0)
            np.add.at(model.mass, node_idx[valid], mass_c[valid])
            if inertia_c is not None:
                np.add.at(model.inertia, node_idx[valid], inertia_c[valid])
        from pyradioss.engine.coloring import compute_element_colors
        if group.conn is not None and len(group.conn) > 0:
            c_idx, c_off = compute_element_colors(group.conn, model.numnod)
            group.state["color_indices"], group.state["color_offsets"] = c_idx, c_off
    from pyradioss.elements import shell_nonlocal
    model.nloc_dmg = shell_nonlocal.setup(model, log)  # NLOC_DMG (None without
    # /NONLOCAL, which is what keeps IPARG(78) == 0)
    # /ADMAS (M5, M139): non-structural mass, added BEFORE the massless-node
    # check so a standalone node + /ADMAS is a legitimate free point mass
    for am in model.admas:
        if am.mass_type == 2:
            # Surface area distributed
            surf = model.surfaces.get(am.grnod_id) if hasattr(model, "surfaces") else None
            if surf is not None and surf.segments is not None and len(surf.segments) > 0:
                segs = np.asarray(surf.segments, dtype=np.int64)
                areas = np.zeros(len(segs), dtype=float)
                for si, seg in enumerate(segs):
                    n1, n2, n3 = seg[0], seg[1], seg[2]
                    n4 = seg[3] if len(seg) > 3 else seg[2]
                    if n4 == n3 or n4 < 0:
                        v1 = model.x0[n2] - model.x0[n1]
                        v2 = model.x0[n3] - model.x0[n1]
                        areas[si] = 0.5 * np.linalg.norm(np.cross(v1, v2))
                    else:
                        d1 = model.x0[n3] - model.x0[n1]
                        d2 = model.x0[n4] - model.x0[n2]
                        areas[si] = 0.5 * np.linalg.norm(np.cross(d1, d2))
                tot_area = np.sum(areas)
                if tot_area > 0.0:
                    for si, seg in enumerate(segs):
                        n1, n2, n3 = seg[0], seg[1], seg[2]
                        n4 = seg[3] if len(seg) > 3 else seg[2]
                        seg_m = am.mass * (areas[si] / tot_area)
                        if n4 == n3 or n4 < 0:
                            m_nod = seg_m / 3.0
                            for nid in (n1, n2, n3):
                                if 0 <= nid < model.numnod:
                                    model.mass[nid] += m_nod
                        else:
                            m_nod = seg_m / 4.0
                            for nid in (n1, n2, n3, n4):
                                if 0 <= nid < model.numnod:
                                    model.mass[nid] += m_nod
                    continue
        elif am.mass_type in (3, 4, 6, 7):
            if am.mass_type in (6, 7):
                pids = [am.grnod_id]
            else:
                # Part group
                grpart = model.egroups.get("PART", {}).get(am.grnod_id) if hasattr(model, "egroups") else None
                pids = getattr(grpart, "part_ids_resolved", None) if grpart else None
                if pids is None and grpart:
                    pids = getattr(grpart, "members", [])
                if not pids and hasattr(model, "part_groups") and am.grnod_id in model.part_groups:
                    pids = model.part_groups[am.grnod_id]
                if not pids and hasattr(model, "parts") and am.grnod_id in model.parts:
                    pids = [am.grnod_id]
            if pids:
                part_nodes = set()
                for _, grp in model.element_groups():
                    p_ids = grp.state.get("part_ids")
                    if p_ids is not None:
                        mask = np.isin(p_ids, pids)
                        if np.any(mask):
                            conn = grp.state.get("mass_conn", grp.conn)[mask]
                            valid = conn[conn >= 0]
                            part_nodes.update(valid.tolist())
                if part_nodes:
                    m_per_node = am.mass / len(part_nodes)
                    for n_idx in part_nodes:
                        if 0 <= n_idx < model.numnod:
                            model.mass[n_idx] += m_per_node
                    continue
            else:
                log.error(f"/ADMAS/{am.id}: unknown part or part group {am.grnod_id}", "ADMAS CHECK")
                continue

        g = model.node_groups.get(am.grnod_id)
        if g is None or g.node_idx is None:
            log.error(f"/ADMAS/{am.id}: unknown node group {am.grnod_id}",
                      "ADMAS CHECK")
            continue
        if am.mass_type == 1:
            m_per_node = am.mass / max(1, len(g.node_idx))
            model.mass[g.node_idx] += m_per_node
        else:
            model.mass[g.node_idx] += am.mass

    # /ADMAS/NON_UNIFORM (M114)
    for an in getattr(model, "admas_non_uniforms", {}).values():
        if an.kind == "NODE":
            for item in getattr(an, "items", []):
                try:
                    n_idx = model.node_index(item.entity_id)
                except (KeyError, ValueError):
                    continue
                if 0 <= n_idx < model.numnod:
                    model.mass[n_idx] += item.mass
        elif an.kind == "PART":
            for item in getattr(an, "items", []):
                part_id = item.entity_id
                part_nodes = set()
                for _, grp in model.element_groups():
                    p_ids = grp.state.get("part_ids")
                    if p_ids is not None:
                        mask = (p_ids == part_id)
                        if np.any(mask):
                            conn = grp.state.get("mass_conn", grp.conn)[mask]
                            valid = conn[conn >= 0]
                            part_nodes.update(valid.tolist())
                if part_nodes:
                    m_per_node = item.mass / len(part_nodes)
                    for n_idx in part_nodes:
                        if 0 <= n_idx < model.numnod:
                            model.mass[n_idx] += m_per_node

    # /INIVEL
    for iv in model.inivel:
        g = model.node_groups.get(iv.grnod_id)
        if g is None or g.node_idx is None:
            log.error(f"/INIVEL/{iv.id}: unknown node group {iv.grnod_id}",
                      "INIVEL CHECK")
            continue
        if iv.kind == "AXIS":
            # rigid-rotation velocity field: v += omega * d x (x0 - P),
            # plus the card's translational velocity Vt (the real AXIS
            # card carries Vxt/Vyt/Vzt next to VR — M37; zero for the
            # port's compact AXIS card, which has no Vt fields)
            r = model.x0[g.node_idx] - iv.origin
            model.v[g.node_idx] += iv.v + iv.omega * np.cross(iv.axis, r)
        else:
            model.v[g.node_idx] = iv.v

    # /GJOINT added mass & inertia (hm_read_gjoint.F 168-173, M594)
    if hasattr(model, "gjoints"):
        for gj in model.gjoints.values():
            for nid, m_add, i_add in [
                (gj.node_id0, gj.mass0, gj.inertia0),
                (gj.node_id1, gj.mass1, gj.inertia1),
                (gj.node_id2, gj.mass2, gj.inertia2),
                (getattr(gj, "node_id3", 0), getattr(gj, "mass3", 0.0), getattr(gj, "inertia3", 0.0)),
            ]:
                if nid and nid in model._id2idx:
                    idx = model.node_index(nid)
                    if m_add > 0.0:
                        model.mass[idx] += m_add
                    if i_add > 0.0:
                        model.inertia[idx] += i_add

    # massless nodes: harmless if nothing ever loads them, fatal otherwise.
    # The Engine divides force by mass, so give unreferenced nodes a tiny
    # mass and warn (the original errors out for loaded massless nodes).
    massless = model.mass <= 0.0
    if np.any(massless):
        log.warning(f"{int(massless.sum())} node(s) carry no mass "
                    f"(not referenced by any element); they are frozen.",
                    "MASS INIT")
        model.mass[massless] = 1e30  # infinite mass = frozen node

    # /INISTA initial stress, strain, and state mapping (hm_read_inista.F, lec_inistate_yfile.F)
    from .inista import apply_inista
    apply_inista(model, log=log)

    # /INITEMP initial temperature field (hm_read_initemp.F, initemp_shell.F90, sinit3.F)
    from .initemp import apply_initemp
    apply_initemp(model, log=log)



# ----------------------------------------------------------------------------
# Rigid bodies (/RBODY, /RBE2): mass, COG, inertia tensor  (M5)
# ----------------------------------------------------------------------------

def initialize_rigid_bodies(model: Model, log: MessageLog) -> None:
    """Compute each rigid body's total mass, center of gravity and inertia
    tensor from its slave nodes, and resolve the master node.

    Fortran origin: ``starter/source/constraints/general/rbody/rbyini.F``
    (and ``inirby``), which assemble exactly these quantities into the
    RBY buffer. The theory is the discrete rigid body: the slaves are
    point masses m_s at positions x_s (plus the isotropic lumped nodal
    inertias I_s that shells contribute), so

        M  = sum m_s (+ added mass)
        xg = sum m_s x_s / sum m_s
        J  = sum m_s (|r|^2 E - r r^T) + sum I_s E + diag(Jadd),
             r = x_s - xg

    Notes baked in:

    * an /RBE2 master is a structural node: its own mass/position joins
      the sums; an /RBODY master usually carries no mass (frozen by the
      mass check) and is EXCLUDED from them — with Icog=1 (default) it is
      *relocated* to the computed COG, the Radioss ICoG behaviour;
    * added mass/inertia act AT the COG (they shift nothing);
    * element deletion (/FAIL) never changes nodal masses (the deleted
      element's mass stays, see the /FAIL plumbing) — so M, xg, J are
      computed ONCE here and stay exact for the whole run;
    * slaves that are massless standalone nodes are tolerated (they are
      carried kinematically) but contribute nothing to M/J — a body needs
      at least some real mass;
    * a near-singular inertia tensor (all slave mass on one line) is
      regularized with a small isotropic term and flagged: the rotation
      rate about the mass line is then meaningless but stays bounded.
    """
    # one kinematic condition per DOF: a node SLAVE of two bodies (or a
    # node serving as MASTER twice) is an error. A node that is the MASTER
    # of one body and a SLAVE of another is a rigid-body CHAIN (M14): the
    # original starter resolves such nestings into a PARENT_OF hierarchy
    # (rbody_part_modif.F90); this port carries the bodies as written —
    # the IMPLICIT solver resolves the chain by transform substitution
    # (implicit/constraints.py), the EXPLICIT engine refuses it loudly
    # (engine/rigid_body.build_rigid_bodies — its per-body 6-DOF
    # integrator has no nesting order).
    seen_slave = np.zeros(model.numnod, dtype=bool)
    seen_master = np.zeros(model.numnod, dtype=bool)
    # which body claimed each node as a slave first (for the message)
    slave_owner: dict = {}
    for rb in model.rbodies:
        who = f"/{rb.kind}/{rb.id}"
        try:
            rb.master = model.node_index(rb.master_id)
        except KeyError:
            log.error(f"{who}: unknown master node {rb.master_id}",
                      "RBODY CHECK")
            continue
        g = model.node_groups.get(rb.grnod_id)
        if g is None or g.node_idx is None or g.node_idx.size == 0:
            log.error(f"{who}: slave node group {rb.grnod_id} is missing "
                      f"or empty", "RBODY CHECK")
            continue
        rb.slaves = g.node_idx[g.node_idx != rb.master]

        # ---- shared SECONDARY nodes (checkrby.F) --------------------------
        # A node claimed as a slave by two bodies is an over-constraint —
        # but ONLY when both claimants are ACTIVE.  The reference makes
        # exactly this distinction (checkrby.F, the "Report of secondary
        # double nodes" block): it walks the duplicated slave nodes,
        # collects the bodies sharing each one, and sets IACTI = 0 as soon
        # as ANY of them is inactive (NPBY(7,IRB) == 0, i.e. sens_ID != 0 —
        # hm_read_rbody.F: "IF(ISENS == 0) THEN NPBY(7,NRB)=1 ELSE
        # NPBY(7,NRB)=0").  Then
        #
        #     IF (IFOUND==0 .AND. IACTI> 0)  -> ERROR   (MSGID 3121)
        #     ELSEIF (IFOUND==0 .AND. IACTI==0) -> WARNING (MSGID 1026,
        #                          "sensor effect will be treated later")
        #
        # i.e. sharing is LEGAL between sensor-gated bodies: the sensors
        # choreograph which one owns the nodes at any instant.  That is a
        # real modelling technique — RD-E-1200's BIKERC defines overlapping
        # "all" / "all less wheels" / "rear wheel" bodies, every one gated
        # by a different /SENSOR, and the port's unconditional error
        # rejected the deck (M39 / M38-NEW-4).
        #
        # IFOUND is upstream's ifrbody_off.F90 probe for a /RBODY/OFF card
        # in the engine restart decks (a deactivated body cannot clash
        # either).  The port has no engine-deck /RBODY/OFF reader, so this
        # takes the IFOUND = 0 branch always — the STRICTER of the two, so
        # the cut can only ever over-report, never under-report.
        shared = rb.slaves[seen_slave[rb.slaves]] if len(rb.slaves) else \
            np.empty(0, dtype=np.int64)
        if len(shared) or seen_master[rb.master]:
            others = {}                       # (kind, id) -> body, deduped
            for n in shared:
                o = slave_owner.get(int(n))
                if o is not None:
                    others.setdefault((o.kind, o.id), o)
            others = [others[k] for k in sorted(others)]
            all_active = rb.sens_id == 0 and all(
                o.sens_id == 0 for o in others)
            if seen_master[rb.master] or all_active:
                log.error(f"{who}: node(s) already belong to another rigid "
                          f"body", "RBODY CHECK")
                continue
            names = ", ".join(f"/{o.kind}/{o.id}" for o in others)
            log.warning(f"{who}: shares slave node(s) with {names} — legal "
                        f"because the bodies are /SENSOR-gated (upstream "
                        f"resolves the overlap by activation, checkrby.F "
                        f"MSGID 1026).  NOTE the port does not gate rigid "
                        f"bodies by sensor (sens_ID is read but ignored), "
                        f"so the Engine would drive the shared nodes from "
                        f"every body at once", "RBODY CHECK")
        if seen_master[rb.slaves].any() or seen_slave[rb.master]:
            log.info(f"     {who}: rigid-body CHAIN (a master of one body "
                     f"is a slave of another) — supported by the IMPLICIT "
                     f"solver only (M14)")
        for n in rb.slaves:
            slave_owner.setdefault(int(n), rb)
        seen_slave[rb.slaves] = True
        seen_master[rb.master] = True

        # mass sums: slaves + the master if it is structural (real mass).
        # Frozen (1e30) masses are the mass-check placeholder for nodes no
        # element references — they carry NO physical mass.
        m = model.mass[rb.slaves].copy()
        m[m >= 1e29] = 0.0
        mm = model.mass[rb.master]
        m_master = mm if mm < 1e29 else 0.0
        msum = float(m.sum()) + m_master
        if msum + rb.added_mass <= 0.0:
            log.warning(f"{who}: rigid body has no mass — floored to 1e-20 (inirby.F)",
                        "RBODY CHECK")
            rb.added_mass = 1e-20
        if msum > 0.0:
            xg = (m[:, None] * model.x0[rb.slaves]).sum(axis=0)
            xg = (xg + m_master * model.x0[rb.master]) / msum
        else:
            if len(rb.slaves) == 0:
                continue
            xg = model.x0[rb.slaves].mean(axis=0)   # massless: geometric

        # inertia tensor about xg (point masses + isotropic nodal inertias)
        r = model.x0[rb.slaves] - xg
        r2 = np.einsum("nb,nb->n", r, r)
        J = (np.einsum("n,nb,nc->bc", m, r, r) * -1.0
             + np.eye(3) * float((m * r2).sum()))
        rm = model.x0[rb.master] - xg
        J += m_master * (np.eye(3) * float(rm @ rm) - np.outer(rm, rm))
        inert = model.inertia[rb.slaves]
        J += np.eye(3) * float(inert.sum())
        # the card's added inertia. With a Skew_ID (M39) Jxx/Jyy/Jzz are
        # written in the SKEW's axes, so the diagonal tensor is rotated
        # into the global frame before it is added — inirby.F's
        # ``IF(NOSKEW/=0) CALL CHBAS(SKEW(1,NOSKEW), RBY(1,NRB))``, which
        # is M = A M A^T with A's columns the skew axes (chbas.F).  Done
        # ONCE here: the body carries its own rotation afterwards, so even
        # a /SKEW/MOV only contributes its initial orientation.
        jadd = np.diag(rb.jadd if rb.jadd is not None else np.zeros(3))
        row = getattr(rb, "skew_row", 0)
        if row:
            jadd = model.skews.rotate_tensor(row, jadd)
        J += jadd
        if rb.ispher == 1:
            J = np.eye(3) * (np.trace(J) / 3.0)

        # regularize a singular tensor (collinear point masses): the spin
        # about the mass line has no physics — keep it bounded, warn once
        lam = np.linalg.eigvalsh(J)
        if lam[0] < 1e-8 * max(lam[2], 1e-30):
            J += np.eye(3) * max(1e-8 * lam[2], 1e-30)
            log.warning(f"{who}: (near-)singular inertia tensor — slave "
                        f"masses are collinear; the spin about that line "
                        f"is regularized", "RBODY CHECK")

        rb.mass_total = msum + rb.added_mass
        rb.xg = xg
        rb.J = J

        # ICoG = 1: relocate the master node to the COG (Radioss default)
        if rb.icog == 1 and rb.kind == "RBODY":
            model.x0[rb.master] = xg
            model.x[rb.master] = xg
        # the added mass physically rides the body: hang it on the master
        # so the Engine's nodal KE/momentum ledgers see it move
        if rb.added_mass > 0.0:
            if model.mass[rb.master] >= 1e29:      # was frozen-massless
                model.mass[rb.master] = rb.added_mass
            else:
                model.mass[rb.master] += rb.added_mass

        pj = np.linalg.eigvalsh(rb.J)
        log.info(f"     {who}: {len(rb.slaves)} SLAVE NODE(S), MASS = "
                 f"{rb.mass_total:12.5E}, COG = {xg[0]:12.5E} "
                 f"{xg[1]:12.5E} {xg[2]:12.5E}")
        log.info(f"       PRINCIPAL INERTIA . . . . : {pj[0]:12.5E} "
                 f"{pj[1]:12.5E} {pj[2]:12.5E}")
