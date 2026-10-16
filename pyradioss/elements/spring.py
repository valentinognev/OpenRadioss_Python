"""
2-node spring element (/SPRING + /PROP/SPRING, TYPE4).

Fortran origin: ``engine/source/elements/spring/`` (rforc3.F, TYPE4 branch).

The TYPE4 spring acts along the line joining its two nodes:

    F = K * (L - L0) + C * L_dot

with the property's mass M lumped half to each node (a spring with M = 0
would have no stable time step of its own — the Starter enforces M > 0).

Critical time step of the two-mass oscillator (masses M/2, stiffness K),
including the damping reduction used by the original:

    omega = 2 sqrt(K / M),  xi = C / sqrt(K M)
    dt    = (2/omega) * (sqrt(1 + xi^2) - xi)
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3
from . import spring_advanced, spring_beam, spring_general, spring_mat, spring_pretensioner

#: Spring property TYPE numbers whose /PROP card carries a mass that THIS
#: PORT actually reads AND whose Starter reader genuinely REQUIRES mass > 0.
#: The check below is a real requirement for an ACTIVE spring — a TYPE4
#: /SPRING with no mass has no stable time step of its own — but it must
#: mirror what the Fortran Starter actually enforces, card for card:
#:
#: * TYPE4 (/PROP/SPRING) — hand reader in starter_keywords.read_prop; the
#:   Fortran /PROP/TYPE4 reader/RINI4 needs a positive mass for the spring's
#:   own explicit step, so the port checks it.
#:
#: TYPE32 (/PROP/SPR_PRE) is NOT in this set (M40): the Fortran Starter does
#: NOT enforce MASS > 0 for the pretensioner.  ``hm_read_prop32.F`` reads the
#: card mass with ``HM_GET_FLOATV('MASS',AMAS,...)`` (a BLANK field gives
#: AMAS = 0), stores it with ``SET_U_GEO(1,AMAS)``, and its only checks are
#: MSGID 408 (F1/D1/E1/STIF1 force-curve over-specification) and MSGID 406
#: (zero spring LENGTH XL) — there is no mass check.  The ``MASS > 0`` in
#: ``prop_p32_spr_pre.cfg``'s CHECK block is a HyperMesh-GUI validation, not
#: a Starter one.  Verified by running the real ``starter_win64.exe`` on
#: RD-HWX-T-1010 cantilever_completed (whose SPR_PRE/2 card has a BLANK mass,
#: Stif0 = 13744.468): 0 ERRORS, the listing printing
#: ``MASS. . . = 0.000000000000`` — accepted.  So a blank/zero pretensioner
#: mass is legal Starter data; the port mirrors that and does not error.
#: (parse_spr_pre still READS the field so a real mass — e.g. RD-V-0031's
#: 1E-5 — feeds the nodal mass; the pretensioner physics is unported, so the
#: Engine refuses the group by :func:`prop_reader.refuse_inactive_properties`
#: — the honest ERROR->SKIPS the check must not pre-empt.  M39-BUG-SPRPRE.)
#:
#: TYPE8/TYPE13 are excluded because :mod:`spring_general` owns their mass.
#: Every OTHER spring spelling (SPR_PUL 12, SPR_MAT 23, SPR_AXI 25,
#: SPR_TAB 26, NSTRAND 28, KJOINT 33/45, SPR_CRUS 44, SPR_MUSCLE 46 ...)
#: parses to an InactiveProperty whose ``mass`` is the PLACEHOLDER 0.0 of
#: ``prop_reader._universal_geo_params`` — not a value off the card.  Mass-
#: checking that placeholder reports a deck error that does not exist, and
#: names the wrong card while doing it; the Engine already refuses those
#: groups (``prop_reader.refuse_inactive_properties``) and the Starter
#: already warns (checks.check_model's PROP CHECK), which is the honest
#: pair of messages.  Add a type here only together with a reader that fills
#: its mass AND a Fortran Starter that actually requires it:
#:   4  hm_read_prop04.F:114 reads MASS -> GEO(1); :137-143 MSGID 229 when
#:      GEO(1) <= EM15 (rmass.F:64 EMS = HALF*GEO(1)*XL feeds the dt).
#:   12 hm_read_prop12.F:114 reads MASS -> GEO(1); :150-156 MSGID 229 when
#:      GEO(1) <= EM15 (rinit3.F:537 routes TYPE12 with TYPE4 to RMASS).
#:   28 hm_read_prop28.F:139 reads MASS -> RHO; :262-268 MSGID 423 when
#:      RHO == 0 (mass per unit length of the strands).
#: An InactiveProperty instance of any of these types is skipped at the
#: use site (its mass is the placeholder, not a card value).
_MASS_REQUIRED_SPRING_TYPES = frozenset({4, 12, 28})

#: /PROP spelling per TYPE for the mass message (the card the user wrote)
_SPRING_PROP_SPELLING = {4: "SPRING", 12: "SPR_PUL", 19: "SPR_TORS", 25: "SPR_AXI", 26: "SPR_TAB", 27: "SPR_BDAMP", 28: "NSTRAND", 32: "SPR_PRE", 35: "STITCH", 36: "PREDIT", 44: "SPR_CRUS", 46: "SPR_MUSCLE"}

#: IGTYP -> owning kernel per rforc3.F ELSEIF branch, in ladder order (:243 4, :321 26,
#: :387 27, :452 8, :558 12, :626 13, :749 25, :862 29-31, :1061 32, :1160 33/45,
#: :1185 35/36, :1293 44, :1391 46; 23 via forint.F:1451; 33/45 live in engine.kjoint).
IGTYP_GROUPS: dict[int, str] = {
    4: "spring", 26: "spring_advanced", 27: "spring_advanced",
    8: "spring_general", 12: "spring_advanced", 13: "spring_beam",
    25: "spring_advanced", 29: "spring_user", 30: "spring_user",
    31: "spring_user", 32: "spring_pretensioner", 33: "kjoint",
    45: "kjoint", 35: "spring_advanced", 36: "spring_advanced",
    44: "spring_advanced", 46: "spring_advanced", 23: "spring_mat",
}


def _safe_param(params: dict, key: str, default: float = 0.0) -> float:
    """Safely extract float parameter from prop.params with fallback."""
    val = params.get(key)
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia.  A /SPRING group may mix the
    axial TYPE4 spring (this module) with the 6-DOF TYPE8/TYPE13 general
    springs (:mod:`spring_general`), one property per part slice — each
    element is classified by its slice's property type and the two paths
    write into the SAME per-(element,node) mass/inertia return."""
    st = group.state
    n = group.n
    if "off" not in st:
        st["off"] = np.ones(n, dtype=float)
    if n == 0 or len(group.conn) == 0:
        st.update(
            L0=np.empty(0),
            mass=np.empty(0),
            k=np.empty(0),
            cdamp=np.empty(0),
            force=np.empty(0),
            eint=np.empty(0),
            ehour=np.empty(0),
            idx4=np.empty(0, dtype=np.int64),
            idx6=np.empty(0, dtype=np.int64),
            idx12=np.empty(0, dtype=np.int64),
            idx23=np.empty(0, dtype=np.int64),
            idx32=np.empty(0, dtype=np.int64),
            idx19=np.empty(0, dtype=np.int64),
            idx44=np.empty(0, dtype=np.int64),
            idx46=np.empty(0, dtype=np.int64),
            model=model,
        )
        return np.empty(0, dtype=np.int64), np.empty(0), None

    if group.conn.shape[1] >= 2:
        xe = model.x0[group.conn[:, :2]]
        L0 = norm3(xe[:, 1] - xe[:, 0])
    else:
        L0 = np.zeros(n)
    mass = np.zeros(n)
    k = np.zeros(n)
    cdamp = np.zeros(n)
    kind = np.full(n, 4, dtype=np.int64)
    for sl, mat, prop in st["slices"]:
        pt = getattr(prop, "type", 4)
        if pt == 4:
            pname = type(prop).__name__
            if "36" in pname or "Predit" in pname:
                pt = 36
            elif "12" in pname or "Pulley" in pname or "SprPul" in pname:
                pt = 12
            elif "28" in pname or "Nstrand" in pname or "Type28" in pname:
                pt = 28
            elif "23" in pname or "SprMat" in pname or "Type23" in pname:
                pt = 23
        kind[sl] = pt
        p = getattr(prop, "params", {}) or {}
        if pt in (12, 28):
            mass[sl] = _safe_param(p, "mass", 0.0)
            continue
        if pt == 23:
            continue                       # SPR_MAT mass & inertia computed by spring_mat
        if pt in spring_general.SPRING_PROP_TYPES or pt in spring_advanced.ADVANCED_SPRING_PROP_TYPES:
            continue                       # 6-DOF and advanced springs built by their own modules
        mass[sl] = _safe_param(p, "mass", 0.0)
        k[sl] = _safe_param(p, "k", 0.0)
        cdamp[sl] = _safe_param(p, "c", 0.0)
    is6 = (kind == 8)
    is13 = (kind == 13)
    is12 = (kind == 12)
    is32 = (kind == 32)
    is19 = (kind == 19)
    is23 = (kind == 23)
    is25 = (kind == 25)
    is26 = (kind == 26)
    is27 = (kind == 27)
    is28 = (kind == 28)
    is35 = (kind == 35)
    is36 = (kind == 36)
    is44 = (kind == 44)
    is46 = (kind == 46)
    is_adv = is12 | is19 | is25 | is26 | is27 | is28 | is35 | is36 | is44 | is46
    is_kj = (kind == 33) | (kind == 45)
    idx4 = np.where(~is6 & ~is13 & ~is32 & ~is_adv & ~is_kj & ~is23)[0]
    idx6 = np.where(is6)[0]
    idx13 = np.where(is13)[0]
    idx12 = np.where(is12)[0]
    idx23 = np.where(is23)[0]
    idx32 = np.where(is32)[0]
    idx19 = np.where(is19)[0]
    idx25 = np.where(is25)[0]
    idx26 = np.where(is26)[0]
    idx27 = np.where(is27)[0]
    idx28 = np.where(is28)[0]
    idx35 = np.where(is35)[0]
    idx36 = np.where(is36)[0]
    idx44 = np.where(is44)[0]
    idx46 = np.where(is46)[0]
    idx_kj = np.where(is_kj)[0]

    st["idx23"] = idx23
    st["idx28"] = idx28

    # Initial length for 3-node pulley spring (r3buf3.F lines 97-99): L0 = L01 + L02
    if len(idx12) > 0 and group.conn.shape[1] >= 3:
        n1 = group.conn[idx12, 0]
        n2 = group.conn[idx12, 1]
        n3 = group.conn[idx12, 2]
        L01 = norm3(model.x0[n2] - model.x0[n1])
        L02 = norm3(model.x0[n2] - model.x0[n3])
        L0[idx12] = L01 + L02

    if len(idx32):
        spring_pretensioner.init_pretensioner_type32(group, model, log, idx32)

    # A spring with no mass has no stable time step of its own — but only
    # the property types whose mass this port actually READS may be checked
    # for it (see _MASS_REQUIRED_SPRING_TYPES; M39 / M38-NEW-1).  The
    # message names the property the user actually wrote, not TYPE4's card.
    for sl, mat, prop in st["slices"]:
        pt = getattr(prop, "type", 4)
        if pt not in _MASS_REQUIRED_SPRING_TYPES:
            continue
        if getattr(prop, "inactive", False):
            # InactiveProperty: its ``mass`` is _universal_geo_params()'s
            # placeholder 0.0, not a value read off the card -- checking it
            # would invent a deck error.  refuse_inactive_properties owns
            # the honest message for such a group.
            continue
        bad = np.zeros(n, dtype=bool)
        bad[sl] = mass[sl] <= 0.0
        if not bad.any():
            continue
        pn = getattr(prop, "prop_name", None)             or _SPRING_PROP_SPELLING.get(pt, f"TYPE{pt}")
        if log is not None:
            for eid in group.ids[bad]:
                log.error(f"/SPRING {eid}: /PROP/{pn}/{prop.id} mass must be "
                          f"> 0 (needed for the explicit time step)",
                          "SPRING INIT")

    st.update(L0=L0, mass=mass, k=k, cdamp=cdamp,
              force=np.zeros(n), eint=np.zeros(n), ehour=np.zeros(n),
              idx4=idx4, idx6=idx6, idx13=idx13, idx12=idx12, idx23=idx23, idx32=idx32,
              idx19=idx19, idx25=idx25, idx26=idx26, idx27=idx27, idx35=idx35, idx36=idx36, idx44=idx44, idx46=idx46, idx_kj=idx_kj, model=model)

    stride = group.conn.shape[1] if group.conn.ndim == 2 else 2
    massn = np.zeros(stride * n)
    inertn = np.zeros(stride * n)
    for i in range(n):
        if is12[i] or is28[i] or is23[i] or is6[i] or is13[i] or is19[i] or is36[i]:
            continue
        m = mass[i]
        massn[stride * i] = m / 2.0
        massn[stride * i + 1] = m / 2.0

    if len(idx6):
        spring_general.init6(group, model, log, idx6, massn, inertn)
    if len(idx13):
        spring_beam.init_spring_beam_type13(group, model, log, idx13, massn, inertn)
    if len(idx12) or len(idx19) or len(idx25) or len(idx26) or len(idx27) or len(idx35) or len(idx36) or len(idx44) or len(idx46):
        spring_advanced.init_advanced(group, model, log, idx12=idx12, idx19=idx19, idx25=idx25, idx26=idx26, idx27=idx27, idx35=idx35, idx36=idx36, idx44=idx44, idx46=idx46, massn=massn, inertn=inertn)
    if len(idx23):
        spring_mat.init_spring_mat_type23(group, model, log, idx23=idx23, massn=massn, inertn=inertn)
    if len(idx28):
        from . import nstrand
        nstrand.init_nstrand_type28(group, model, log, idx28=idx28, massn=massn, inertn=inertn)
    node_idx = group.conn.reshape(-1)
    valid = (node_idx >= 0)
    return node_idx[valid], massn[valid], (inertn[valid] if inertn.any() else None)


def _forces_axial(group, x, v, dt, fint, idx):
    """Axial TYPE4 spring forces for the elements ``idx`` (``slice(None)``
    for the whole group — the original vectorized path)."""
    st = group.state
    conn = group.conn[idx]
    if len(conn) == 0:
        return np.empty(0)
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])
    if v is None:
        Ldot = np.zeros(len(conn))
    else:
        Ldot = np.einsum("nb,nb->n",
                         v[conn[:, 1]] - v[conn[:, 0]], a)

    alive = st.get("off", np.ones(group.n, dtype=float))[idx] > 0.0

    F_old = st["force"][idx].copy()
    F = st["k"][idx] * (L - st["L0"][idx]) + st["cdamp"][idx] * Ldot
    F = np.where(alive, F, 0.0)
    st["force"][idx] = F

    fvec = F[:, None] * a          # tension pulls the nodes together
    if fint is not None:
        np.add.at(fint, conn[:, 0], fvec)
        np.add.at(fint, conn[:, 1], -fvec)

    # elastic part of the work goes to internal energy; damping work too
    # (the original books spring damping into internal energy as well).
    if dt is not None and dt > 0.0:
        st["eint"][idx] += np.where(alive, 0.5 * (F_old + F) * Ldot * dt, 0.0)

    mass = np.maximum(st["mass"][idx], EM20)
    k = np.maximum(st["k"][idx], 0.0)
    c_damp = st["cdamp"][idx]
    pos_k = (st["k"][idx] > 0.0) & (st["mass"][idx] > 0.0)
    pure_c = (st["k"][idx] <= 0.0) & (c_damp > 0.0) & (st["mass"][idx] > 0.0)

    omega = 2.0 * np.sqrt(np.where(pos_k, k / mass, 1.0))
    xi = np.where(pos_k, c_damp / np.sqrt(np.maximum(k * mass, EM20)), 0.0)
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    # OpenRadioss r1len3.F: when K=0 and C>0: dt = 0.5 * M / C
    dt_c = np.where(pure_c, 0.5 * mass / np.maximum(c_damp, EM20), EP30)
    dt_elem = np.where(pos_k, dt_crit, dt_c)
    return np.where(alive, dt_elem, EP30)


def _forces_axial_type32(group, x, v, dt, fint, idx):
    """TYPE32 pretensioner spring forces (delegated to spring_pretensioner)."""
    return spring_pretensioner.forces_pretensioner_type32(group, x, v, dt, fint, idx)



def forces(group, x, v, vr, dt, fint, mint):
    if group.n == 0 or len(group.conn) == 0:
        return np.empty(0)
    st = group.state
    alive = st.get("off", np.ones(group.n, dtype=float)) > 0.0
    idx6 = st.get("idx6")
    idx4 = st.get("idx4")
    idx12 = st.get("idx12")
    idx13 = st.get("idx13")
    idx23 = st.get("idx23")
    idx32 = st.get("idx32")
    idx19 = st.get("idx19")
    idx25 = st.get("idx25")
    idx26 = st.get("idx26")
    idx27 = st.get("idx27")
    idx28 = st.get("idx28")
    idx35 = st.get("idx35")
    idx36 = st.get("idx36")
    idx44 = st.get("idx44")
    idx46 = st.get("idx46")
    idx_kj = st.get("idx_kj")
    _branch_idx = (idx6, idx12, idx13, idx23, idx32, idx19, idx25, idx26,
                   idx27, idx28, idx35, idx36, idx44, idx46, idx_kj)
    if not any(idx is not None and len(idx) for idx in _branch_idx):
        # pure axial TYPE4 group
        dtc = _forces_axial(group, x, v, dt, fint, slice(None))
        if "force" in st:
            st["force"] = np.where(alive, st["force"], 0.0)
        return np.where(alive, dtc, EP30)
    dtc = np.full(group.n, EP30)
    if idx4 is not None and len(idx4):
        dtc[idx4] = _forces_axial(group, x, v, dt, fint, idx4)
    if idx12 is not None and len(idx12):
        dtc[idx12] = spring_advanced.forces_pulley_type12(group, x, v, dt, fint, idx12)
    if idx13 is not None and len(idx13):
        dtc[idx13] = spring_beam.forces_spring_beam_type13(group, x, v, vr, dt, fint, mint, idx13)
    if idx23 is not None and len(idx23):
        dtc[idx23] = spring_mat.forces_spring_mat_type23(group, x, v, vr, dt, fint, mint, idx23)
    if idx32 is not None and len(idx32):
        dtc[idx32] = _forces_axial_type32(group, x, v, dt, fint, idx32)
    if idx6 is not None and len(idx6):
        dtc[idx6] = spring_general.forces6(group, x, v, vr, dt, fint, mint, idx6)
    if idx19 is not None and len(idx19):
        dtc[idx19] = spring_advanced.forces_torsion_type19(group, x, v, vr, dt, fint, mint, idx19)
    if idx25 is not None and len(idx25):
        dtc[idx25] = spring_advanced.forces_axi_type25(group, x, v, dt, fint, idx25)
    if idx26 is not None and len(idx26):
        dtc[idx26] = spring_advanced.forces_tab_type26(group, x, v, dt, fint, idx26)
    if idx27 is not None and len(idx27):
        dtc[idx27] = spring_advanced.forces_bdamp_type27(group, x, v, dt, fint, idx27)
    if idx28 is not None and len(idx28):
        from . import nstrand
        dtc[idx28] = nstrand.forces_nstrand_type28(group, x, v, vr, dt, fint, mint, idx28)
    if idx35 is not None and len(idx35):
        dtc[idx35] = spring_advanced.forces_stitch_type35(group, x, v, dt, fint, idx35)
    if idx36 is not None and len(idx36):
        dtc[idx36] = spring_advanced.forces_predit_type36(group, x, v, vr, dt, fint, mint, idx36)
    if idx44 is not None and len(idx44):
        dtc[idx44] = spring_advanced.forces_crushing_type44(group, x, v, dt, fint, idx44, vr=vr, mint=mint)
    if idx46 is not None and len(idx46):
        dtc[idx46] = spring_advanced.forces_muscle_type46(group, x, v, dt, fint, idx46)
    if "force" in st:
        st["force"] = np.where(alive, st["force"], 0.0)
    return np.where(alive, dtc, EP30)


# ----------------------------------------------------------------------------
# Implicit tangent stiffness + residual (M11) — alongside forces()
# ----------------------------------------------------------------------------
# Fortran origin: the element-KE branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F``, spring stiffness) + its
# ``imp_kgeo`` geometric path.
#
# The TYPE4 spring is a TOTAL-form element: forces() rebuilds F = k(L - L0)
# + c*Ldot from the geometry every cycle instead of integrating a rate on
# committed state. That breaks the implicit drivers' pseudo-velocity trick
# (which relies on rate-form kernels accumulating C*grad(u) on the FROZEN
# committed frame): at the frozen geometry the elastic term k(L - L0) never
# feels the trial displacement at all, and the DAMPING term c*(a . du)
# would answer instead — a rate device masquerading as a stiffness. So the
# spring supplies its own implicit residual, ``implicit_internal_forces``,
# which the drivers call INSTEAD of forces() (both geometry modes):
#
# * linear geometry (M8 frozen frame): F = F_committed + k (a_ref . du_rel)
#   along the committed axis a_ref — the total form linearized at the
#   committed geometry, ACCUMULATED on the committed force state exactly
#   like the rate-form kernels accumulate stress (the driver feeds only
#   the CURRENT increment's displacement; the previous increments live in
#   the state, which the driver restores to the committed values before
#   every residual evaluation). Its derivative is EXACTLY the material
#   tangent k a a^T, so a linear step converges in one Newton iteration
#   (the M8 contract);
# * nonlinear geometry (M9 updated-Lagrangian): F = k (L(x_end) - L0) along
#   the CURRENT axis — the total form is exact at any configuration (no
#   midpoint objectivity step is needed: the elastic force is a state
#   function of the end geometry), and its exact derivative is the material
#   tangent PLUS the (F/L)(I - a a^T) geometric term of ``kgeo``.
#
# The DASHPOT (c > 0) is a rate device: like the LAW2 strain-rate term and
# the bulk viscosity (M10 convention), it is DISABLED under implicit with
# an explicit warning — never silently fed the pseudo-velocity du/1
# (deferred, PORTING_GUIDE M11). The stored force/energy state stays the
# state function F^2/2k of the elastic spring.

def _spring_axis(group, x):
    conn = group.conn
    if len(conn) == 0:
        return conn, np.empty(0), np.empty((0, 3))
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])
    return conn, L, a


def _spring_edofs(conn):
    if len(conn) == 0:
        return np.empty((0, 6), dtype=np.int64)
    edofs = np.empty((len(conn), 6), dtype=np.int64)
    for c in range(3):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 3 + c] = conn[:, 1] * 6 + c
    return edofs


def _blocks(kb):
    """(n,3,3) relative block -> (n,6,6) element [[kb,-kb],[-kb,kb]]."""
    n = len(kb)
    if n == 0:
        return np.empty((0, 6, 6))
    ke = np.empty((n, 6, 6))
    ke[:, :3, :3] = kb
    ke[:, 3:, 3:] = kb
    ke[:, :3, 3:] = -kb
    ke[:, 3:, :3] = -kb
    return ke


def tangent(group, x, epsp_incr=None):
    """Material element tangent k a a^T along the current axis at geometry
    ``x``. Returns (ke (n,6,6), edofs (n,6)). ``epsp_incr`` unused (the
    spring is elastic; the dashpot is disabled under implicit)."""
    if group.n == 0 or len(group.conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)
    st = group.state
    idx13 = st.get("idx13")
    if idx13 is not None and len(idx13) == group.n:
        return spring_beam.ke_spring_beam_type13(group, x, idx13)
    conn, L, a = _spring_axis(group, x)
    kb = st["k"][:, None, None] * np.einsum("ni,nj->nij", a, a)
    return _blocks(kb), _spring_edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) stiffness (F/L)(I - a a^T) from the
    current spring force at geometry ``x`` — the same taut-string operator
    as the truss. Identically zero at zero force."""
    if group.n == 0 or len(group.conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)
    st = group.state
    conn, L, a = _spring_axis(group, x)
    F_over_L = st["force"] / L
    eye = np.eye(3)
    kb = F_over_L[:, None, None] * (eye[None, :, :]
                                    - np.einsum("ni,nj->nij", a, a))
    return _blocks(kb), _spring_edofs(conn)


# ----------------------------------------------------------------------------
# Consistent (element) mass — M16, alongside the lumped mass of init_group.
# ----------------------------------------------------------------------------
# Fortran origin: the TYPE4 spring's mass is the /PROP/SPRING scalar mass M
# lumped half to each node (``rmass3.F`` / this file's ``init_group`` /2
# return). Unlike the truss/beam/solid, the spring carries NO distributed
# density field — its mass is a DISCRETE point property, not ∫ρ dV — so the
# "consistent" mass and the lumped mass COINCIDE exactly: M/2 as a point mass
# on each node's three translations, a diagonal 6×6. There is no shape-
# function integral to do (the spring has no interior), which is why this is
# the EXACT mass, not an approximation. Reported here so the modal assembler
# has a uniform ``consistent_mass()`` on every family; the value is identical
# to what the lumped path already puts on these DOFs, so a spring never shifts
# the consistent-vs-lumped spectrum.

def consistent_mass(group, x=None):
    """Exact element mass of the TYPE4 spring: the point mass M/2 on each
    node's translations (diagonal 6×6) — identical to the lumped mass (see
    the note above; the spring has no distributed density to integrate).

    Returns ``(me (n,6,6), edofs (n,6))``. ``x`` unused (a point mass is
    frame-invariant and configuration-independent)."""
    if group.n == 0 or len(group.conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)
    st = group.state
    conn = group.conn
    n = group.n
    half = st["mass"] / 2.0                            # M/2 per node
    me = np.zeros((n, 6, 6))
    for i in range(6):                                 # 3 trans on each node
        me[:, i, i] = half
    return me, _spring_edofs(conn)


# ----------------------------------------------------------------------------
# Viscous DAMPING matrix — M18, alongside the tangent / consistent mass.
# ----------------------------------------------------------------------------
# Fortran origin: the /PROP/SPRING dashpot ``c`` term of the TYPE4 force law
# F = K(L - L0) + C*L_dot (rforc3.F). Under the DIRECT explicit / implicit
# time march the dashpot is a rate device (it reads the nodal velocity, so it
# is disabled in the implicit Newton residual — see the note above and
# PORTING_GUIDE M11). M18 revives it as a genuine assembled DAMPING operator:
# the viscous force F_c = c*L_dot along the axis is EXACTLY the linear map
# f = C_e u_dot with the element damping matrix
#
#     C_e = c * [[ a a^T, -a a^T], [-a a^T, a a^T]]
#
# (the same relative-block structure as the elastic tangent k a a^T, with c in
# place of k — the dashpot resists the RATE of axial stretch just as the
# spring resists the axial stretch itself). A dashpot on one spring among many
# is the canonical source of NON-CLASSICAL damping: C is then NOT proportional
# to M or K, so the damped modes go complex (implicit/complex_modal.py). This
# operator never enters the M8-M17 residual/tangent paths — it feeds only the
# opt-in complex-eigenvalue / complex-mode-superposition solve.

def damping_matrix(group, x):
    """Viscous element damping matrix c a a^T along the current axis at
    geometry ``x`` — the /PROP/SPRING dashpot ``c`` term as a linear operator
    on the nodal velocities (see the note above). Returns ``(ce (n,6,6),
    edofs (n,6))``, the same shape/addressing as ``tangent()`` so the global
    C assembles through exactly the same COO->CSR scatter as K and M. Springs
    with c = 0 contribute a zero block (harmless)."""
    if group.n == 0 or len(group.conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)
    st = group.state
    conn, L, a = _spring_axis(group, x)
    cb = st["cdamp"][:, None, None] * np.einsum("ni,nj->nij", a, a)
    return _blocks(cb), _spring_edofs(conn)


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom):
    """The spring's own implicit residual (called by the drivers INSTEAD of
    ``forces()`` — see the note above): elastic total-form force at the
    trial configuration, linearized at the committed frame under linear
    geometry. Updates the force/energy state in place (trial values; the
    drivers' snapshot/commit machinery handles rollback exactly as for the
    rate-form kernels). ``ur``/``mint`` unused."""
    if group.n == 0 or len(group.conn) == 0:
        return
    st = group.state
    if nlgeom:
        # end configuration: the total form is exact there (x_ref advances
        # per increment under the updated-Lagrangian outer step, so
        # x_ref + u IS the trial configuration)
        conn, L, a = _spring_axis(group, x_ref + u)
        F = st["k"] * (L - st["L0"])
    else:
        # frozen committed frame (x_ref stays at x0 for the whole run and
        # u is THIS increment only): accumulate the linearized force on
        # the committed state — see the note above
        conn, L, a = _spring_axis(group, x_ref)
        du_rel = u[conn[:, 1]] - u[conn[:, 0]]
        F = st["force"] + st["k"] * np.einsum("nb,nb->n", du_rel, a)
    st["force"][...] = F
    # elastic state function (the dashpot is off): eint = F^2 / 2k
    st["eint"][...] = np.where(st["k"] > 0.0,
                               F * F / (2.0 * np.maximum(st["k"], EM20)),
                               0.0)
    fvec = F[:, None] * a
    if fint is not None:
        np.add.at(fint, conn[:, 0], fvec)
        np.add.at(fint, conn[:, 1], -fvec)
