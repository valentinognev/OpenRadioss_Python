"""
The Starter main program.

Fortran origin: ``starter/source/starter/starter0.F`` (program shell) and
``lectur.F`` (the giant reading/initialization driver). Sequence and the
produced files match the original:

    RunName_0000.rad  --Starter-->  RunName_0000.out  (listing)
                                    RunName_0000.rst  (restart for Engine)
"""

from __future__ import annotations

import os
import time

import numpy as np

from .. import banner
from ..common.messages import MessageLog, StarterError
from ..input.deck_reader import read_deck
from ..input.keywords import parse_starter_deck
from ..model.model import Model
from .checks import check_model
from ..input.units import apply_unit_conversions
from .airbag import initialize_monitored_volumes
from .initialization import (build_element_groups,
                             initialize_elements_and_mass,
                             initialize_rigid_bodies,
                             resolve_entity_groups,
                             resolve_generic_sets,
                             resolve_lines, resolve_materials,
                             resolve_node_groups, resolve_single_node_group,
                             resolve_skews,
                             resolve_surfaces)
from .restart import write_restart


def build_model(blocks_or_path, log: Optional[MessageLog] = None) -> Model:
    """Convenience helper to read/parse deck blocks and resolve materials into a Model."""
    if isinstance(blocks_or_path, (str, list, tuple)) or hasattr(blocks_or_path, "__fspath__"):
        blocks = read_deck(blocks_or_path)
    else:
        blocks = blocks_or_path
    model = Model()
    if log is None:
        log = MessageLog()
    parse_starter_deck(blocks, model, log)
    resolve_materials(model, log)
    return model


def run_name_from_input(path: str) -> str:
    """'MYRUN_0000.rad' -> 'MYRUN' (the Radioss run-name convention)."""
    base = os.path.basename(path)
    for suffix in ("_0000.rad", "_0000.RAD"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def _listing_summary(model: Model, log: MessageLog) -> None:
    """Model summary tables of the *_0000.out listing (mass, counts...)."""
    log.info("\n                       MODEL SUMMARY")
    log.info("                       -------------")
    log.info(f"     TITLE . . . . . . . . . . . . . . : {model.title}")
    log.info(f"     NUMBER OF NODES . . . . . . . . . : {model.numnod}")
    counts = {
        "bricks": "BRICK", "bricks_heph": "BRICK", "tshells": "TSHELL", "bric20s": "BRIC20",
        "penta6s": "PENTA6",
        "tetras": "TETRA4", "tetra10s": "TETRA10",
        "shells": "SHELL", "shells_qbat": "SHELL", "shells_qeph": "SHELL", "shel16s": "SHEL16",
        "quads": "QUAD", "sh3n": "SH3N", "sh3n_dkt18": "SH3N",
        "trusses": "TRUSS", "springs": "SPRING", "beams": "BEAM", "beams_fiber": "BEAM",
    }
    for attr, kw in counts.items():
        g = getattr(model, attr)
        if g is not None:
            log.info(f"     NUMBER OF /{kw:<6} ELEMENTS  . . . . : {g.n}")
    log.info(f"     NUMBER OF PARTS . . . . . . . . . : {len(model.parts)}")
    log.info(f"     NUMBER OF MATERIALS . . . . . . . : "
             f"{len(model.materials)}")

    # total mass & center of gravity (frozen nodes' 1e30 masses excluded)
    real = model.mass < 1e29
    m = model.mass[real]
    xm = model.x0[real]
    total = m.sum()
    if total > 0:
        cg = (m[:, None] * xm).sum(axis=0) / total
        log.info(f"     TOTAL MASS  . . . . . . . . . . . : {total:14.7E}")
        log.info(f"     CENTER OF GRAVITY . . . . . . . . : "
                 f"{cg[0]:14.7E} {cg[1]:14.7E} {cg[2]:14.7E}")

    log.info("\n     MATERIALS")
    for mat in model.materials.values():
        law = getattr(mat, "law_name", None) or f"LAW{mat.law}"
        tag = "  [parsed, physics not implemented (M37)]" \
            if getattr(mat, "inactive", False) else ""
        log.info(f"       MAT {mat.id:8d}  {law:<10s} RHO="
                 f"{mat.rho0:12.5E}  {mat.title}{tag}")
    log.info("\n     PARTS")
    for part in model.parts.values():
        log.info(f"       PART {part.id:8d}  PROP={part.prop_id:<8d} "
                 f"MAT={part.mat_id:<8d} {part.title}")


def _resolve_transform_nodes(model: Model, tr_id: int, tr_type: str,
                             grnod: int, sub_id: int, log: MessageLog):
    if sub_id > 0:
        idx = np.where(model.node_submodel == sub_id)[0]
        if len(idx) == 0:
            log.warning(f"/TRANSFORM/{tr_type}/{tr_id}: submodel {sub_id} has no nodes — skipped")
            return None
        return idx
    elif grnod > 0 and grnod in model.node_groups:
        g = model.node_groups[grnod]
        if getattr(g, "node_idx", None) is not None and len(g.node_idx) > 0:
            idx = g.node_idx
        else:
            idx = resolve_single_node_group(model, g, log)
        if len(idx) == 0:
            log.warning(f"/TRANSFORM/{tr_type}/{tr_id}: node group "
                        f"{grnod} evaluated to empty — skipped")
            return None
        return idx
    elif grnod > 0:
        log.warning(f"/TRANSFORM/{tr_type}/{tr_id}: node group "
                    f"{grnod} not found — skipped")
        return None
    return None


def apply_transforms(model: Model, log: MessageLog) -> None:
    """Apply /TRANSFORM transformations (TRA, ROT, SYM, SCA, POS - M63, M85) to node coordinates."""
    for tr in getattr(model, "transforms", []):
        tr_id = tr[0]
        if isinstance(tr[1], str):
            tr_type = tr[1]
            args = tr[2:]
        else:
            tr_type = "TRA"
            args = tr[1:]

        if tr_type == "TRA":
            grnod, tx, ty, tz, n1, n2, sub_id, skew_id = args
            idx = _resolve_transform_nodes(model, tr_id, tr_type, grnod, sub_id, log)
            if idx is None or len(idx) == 0:
                continue
            if n1 > 0 and n2 > 0:
                try:
                    idx1 = model.node_index(n1)
                    idx2 = model.node_index(n2)
                    v = model.x0[idx2] - model.x0[idx1]
                    tx = float(v[0])
                    ty = float(v[1])
                    tz = float(v[2])
                except KeyError as exc:
                    log.warning(f"/TRANSFORM/TRA/{tr_id}: node {exc} for "
                                f"node-pair vector not found")
                    continue
            model.x0[idx, 0] += tx
            model.x0[idx, 1] += ty
            model.x0[idx, 2] += tz

        elif tr_type == "ROT":
            grnod, p1, p2, angle_deg, n1, n2, sub_id = args
            idx = _resolve_transform_nodes(model, tr_id, tr_type, grnod, sub_id, log)
            if idx is None or len(idx) == 0:
                continue
            p1 = np.array(p1, dtype=float)
            p2 = np.array(p2, dtype=float)
            if n1 > 0 and n2 > 0:
                try:
                    p1 = model.x0[model.node_index(n1)].copy()
                    p2 = model.x0[model.node_index(n2)].copy()
                except KeyError as exc:
                    log.warning(f"/TRANSFORM/ROT/{tr_id}: node {exc} not found")
                    continue
            axis = p2 - p1
            norm_axis = np.linalg.norm(axis)
            if norm_axis > 1e-20 and abs(angle_deg) > 1e-12:
                u = axis / norm_axis
                theta = np.radians(angle_deg)
                v = model.x0[idx] - p1
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                dot = np.sum(v * u, axis=1, keepdims=True)
                cross = np.cross(u, v)
                v_rot = v * cos_t + cross * sin_t + u * dot * (1.0 - cos_t)
                model.x0[idx] = p1 + v_rot

        elif tr_type == "SYM":
            grnod, p1, p2, n1, n2, sub_id = args
            idx = _resolve_transform_nodes(model, tr_id, tr_type, grnod, sub_id, log)
            if idx is None or len(idx) == 0:
                continue
            p1 = np.array(p1, dtype=float)
            p2 = np.array(p2, dtype=float)
            if n1 > 0 and n2 > 0:
                try:
                    p1 = model.x0[model.node_index(n1)].copy()
                    p2 = model.x0[model.node_index(n2)].copy()
                except KeyError as exc:
                    log.warning(f"/TRANSFORM/SYM/{tr_id}: node {exc} not found")
                    continue
            normal = p2 - p1
            norm_n = np.linalg.norm(normal)
            if norm_n > 1e-20:
                n_unit = normal / norm_n
                v = model.x0[idx] - p1
                d = np.sum(v * n_unit, axis=1, keepdims=True)
                model.x0[idx] -= 2.0 * d * n_unit

        elif tr_type == "SCA":
            grnod, (sx, sy, sz), n1, sub_id = args
            idx = _resolve_transform_nodes(model, tr_id, tr_type, grnod, sub_id, log)
            if idx is None or len(idx) == 0:
                continue
            center = np.zeros(3, dtype=float)
            if n1 > 0:
                try:
                    center = model.x0[model.node_index(n1)].copy()
                except KeyError as exc:
                    log.warning(f"/TRANSFORM/SCA/{tr_id}: center node {exc} not found")
                    continue
            scale = np.array([sx if sx != 0.0 else 1.0,
                              sy if sy != 0.0 else 1.0,
                              sz if sz != 0.0 else 1.0], dtype=float)
            model.x0[idx] = center + (model.x0[idx] - center) * scale

        elif tr_type == "POS":
            grnod, nodes, pts, sub_id = args
            idx = _resolve_transform_nodes(model, tr_id, tr_type, grnod, sub_id, log)
            if idx is None or len(idx) == 0:
                continue
            p = np.array(pts, dtype=float)
            n1, n2, n3, n4, n5, n6 = nodes
            node_list = [n1, n2, n3, n4, n5, n6]
            skip = False
            for i, nid in enumerate(node_list):
                if nid > 0:
                    try:
                        p[i] = model.x0[model.node_index(nid)].copy()
                    except KeyError as exc:
                        log.warning(f"/TRANSFORM/POS/{tr_id}: node {exc} not found")
                        skip = True
                        break
            if skip:
                continue

            # Build orthonormal frame 1 (P1, P2, P3)
            v12 = p[1] - p[0]
            norm12 = np.linalg.norm(v12)
            if norm12 < 1e-20:
                log.warning(f"/TRANSFORM/POS/{tr_id}: source frame X-axis has zero length")
                continue
            ex1 = v12 / norm12
            v13 = p[2] - p[0]
            ez1_raw = np.cross(ex1, v13)
            normz1 = np.linalg.norm(ez1_raw)
            if normz1 < 1e-20:
                log.warning(f"/TRANSFORM/POS/{tr_id}: source frame points 1, 2, 3 are collinear")
                continue
            ez1 = ez1_raw / normz1
            ey1 = np.cross(ez1, ex1)
            R1 = np.column_stack([ex1, ey1, ez1])

            # Build orthonormal frame 2 (P4, P5, P6)
            v45 = p[4] - p[3]
            norm45 = np.linalg.norm(v45)
            if norm45 < 1e-20:
                log.warning(f"/TRANSFORM/POS/{tr_id}: target frame X-axis has zero length")
                continue
            ex2 = v45 / norm45
            v46 = p[5] - p[3]
            ez2_raw = np.cross(ex2, v46)
            normz2 = np.linalg.norm(ez2_raw)
            if normz2 < 1e-20:
                log.warning(f"/TRANSFORM/POS/{tr_id}: target frame points 4, 5, 6 are collinear")
                continue
            ez2 = ez2_raw / normz2
            ey2 = np.cross(ez2, ex2)
            R2 = np.column_stack([ex2, ey2, ez2])

            # Total rotation: R = R2 @ R1.T
            R = R2 @ R1.T
            O1 = p[0]
            O2 = p[3]
            model.x0[idx] = O2 + (model.x0[idx] - O1) @ R.T

        elif tr_type == "MATRIX":
            grnod, mat_3x3, trans_vec, sub_id = args
            idx = _resolve_transform_nodes(model, tr_id, tr_type, grnod, sub_id, log)
            if idx is None or len(idx) == 0:
                continue
            mat = np.asarray(mat_3x3, dtype=float)
            vec = np.asarray(trans_vec, dtype=float)
            model.x0[idx] = model.x0[idx] @ mat.T + vec


def run_starter(input_file: str, log: MessageLog | None = None,
                nspmd: int = 1) -> Model:
    """Run the full Starter on ``input_file``; returns the initialized
    model (and writes the .out listing and .rst restart next to it).

    ``nspmd > 1`` (``pyradioss-starter -np N``): the model is also
    decomposed into ``nspmd`` SPMD domains (pyradioss/spmd/domdec.py —
    domdec1.F / domdec2.F / ddsplit.F) and one restart per domain,
    ``RunName_0000_0001.rst`` ..., is written next to the global
    ``RunName_0000.rst``; the decomposition table goes to the listing.
    The returned model is the global one either way."""
    log = log or MessageLog()
    nspmd = int(nspmd or 1)
    run_name = run_name_from_input(input_file)
    out_dir = os.path.dirname(os.path.abspath(input_file))
    listing_path = os.path.join(out_dir, f"{run_name}_0000.out")

    with open(listing_path, "w") as listing:
        log.attach_listing(listing)
        log.info(banner())
        log.info(f" STARTER INPUT FILE . . . . . . . . . : {input_file}")
        log.info(f" RUN NAME . . . . . . . . . . . . . . : {run_name}")
        t0 = time.time()

        # 1. read + parse the deck (lectur.F)
        try:
            blocks = read_deck(input_file)
        except FileNotFoundError as e:
            log.error(f"LEXER CRASH: {e}", "LEXER")
            raise StarterError(f"Starter input file not found: {input_file}")
        model = Model()
        parse_starter_deck(blocks, model, log)

        # Apply /MOVE_FUNCT scale and shift transformations (M65)
        for funct_id, scx, scy, shx, shy in getattr(model, "move_functs", []):
            if funct_id in model.functions:
                model.functions[funct_id].transform(scx, scy, shx, shy)
            else:
                log.warning(f"/MOVE_FUNCT targets unknown function {funct_id}")
        # 2. finalize: ids->indices, element groups, node groups, surfaces,
        #    material curve/failure references.  Order matters (M37, the
        #    upstream two-pass resolve): local /UNIT conversion first
        #    (materials must be converted before their curves are
        #    copied), then elements -> element GROUPS -> surfaces (may
        #    reference element groups + other surfaces) -> lines (read
        #    surfaces) -> node groups (read surfaces + element groups).
        apply_unit_conversions(model, log)
        resolve_materials(model, log)
        build_element_groups(model, log)
        resolve_entity_groups(model, log)
        resolve_generic_sets(model, log)
        resolve_surfaces(model, log)
        resolve_lines(model, log)     # after surfaces: /LINE/SURF reads them
        # reference systems (M39): built from the node positions, then
        # bound to every consumer that names one (/BCS, /IMP*, /RBODY,
        # /PROP TYPE8, /INIVEL/AXIS) — before the checks so an unknown
        # skew_ID is reported with all the other model errors
        resolve_skews(model, log)
        # node groups evaluate /BOX which may need resolved skews
        resolve_node_groups(model, log)

        # Apply /TRANSFORM after node groups (including /GRNOD/PART, /GRNOD/SURF) are resolved (AUD-014)
        apply_transforms(model, log)
        if getattr(model, "transforms", None):
            resolve_skews(model, log)

        # 3. checks before any heavy work (fail early with ALL messages)
        check_model(model, log)

        # 4. element buffers + lumped mass + initial conditions, then the
        #    rigid bodies (they need the assembled nodal masses)
        if not log.errors:
            initialize_elements_and_mass(model, log)
            initialize_rigid_bodies(model, log)
            initialize_monitored_volumes(model)
            # reference (physical, pre-mass-scaling) nodal masses: the
            # Engine's init-time computations (interface dt bounds,
            # gravity) use these so a /DT/NODA/CST-grown model resumes
            # from a restart with IDENTICAL derived quantities (M6)
            model.mass0 = model.mass.copy()
            _listing_summary(model, log)

        log.info(log.summary())
        log.info(f" STARTER ELAPSED TIME . . . . . . . . : "
                 f"{time.time() - t0:10.3f} s")

        # 5. restart file — only for a clean model, like the original
        log.check()  # raises StarterError if errors were collected
        if nspmd > 1:
            # refuse the SPMD-unsupported features before any restart is
            # written (no restart for a model the -np run cannot take)
            from ..spmd.domdec import check_spmd_support
            check_spmd_support(model)
        rst_path = os.path.join(out_dir, f"{run_name}_0000.rst")
        write_restart(model, rst_path)
        log.info(f" RESTART FILE WRITTEN . . . . . . . . : {rst_path}")
        if nspmd > 1:
            # 6. SPMD domain decomposition + one restart per domain
            #    (domdec1.F, domdec2.F, ddsplit.F)
            from ..spmd.domdec import write_domain_restarts
            write_domain_restarts(model, nspmd, out_dir, run_name, log)
        log.info("\n     ------------------------------------------------")
        log.info("     STARTER TERMINATION : NORMAL")
        log.info("     ------------------------------------------------")

    return model
