"""DKT18 (Discrete Kirchhoff Triangle) 3-node shell element.

Fortran origin:
  Starter: starter/source/elements/sh3n/coquedk/
    - cdkinit3.F: group initialization
    - cmaini3.F: lumped mass and rotary inertia
  Engine: engine/source/elements/sh3n/coquedk/
    - cdkforc3.F: driver for internal forces and time step
    - cdkcoor3.F / clskew3.F: local frame and coordinates
    - cdkderic3.F: geometric derivatives and constants
    - cdkdefo3.F: membrane strain rates
    - cdkderi3.F: DKT bending shape function derivatives at 3 Hammer points
    - cdkcurv3.F: curvature rates
    - cdkfint3.F: internal forces/moments at Gauss points
    - cdkfcum3.F: accumulation and transformation to global coordinates
    - cndt3.F: critical Courant time step
    - cdkfint_reg.F / dtcdk_reg.F: the non-local (regularised) force and
      time step, reached only under INLOC = IPARG(78) > 0

The tangent, consistent mass, geometric stiffness and the implicit
residuals live in ``shell_dkt18_kinematics.py`` and are re-exported here;
they were moved out so this module stays inside its recorded size budget.

Kinematics & Implicit Extensions (M47, M511):
  - 3 Hammer integration points in-plane for DKT bending
  - 18 DOFs per element: [ux, uy, uz, thx, thy, thz] x 3 nodes
  - Full rank 12 tangent stiffness matrix (3 membrane + 6 bending + 3 drilling)
  - 6 exact rigid-body null modes (3 translations + 3 rotations)
  - Analytical consistent mass matrix (translational + rotary bending inertia)
  - Geometric (initial-stress) stiffness matrix K_geo
  - Static & implicit internal force evaluation for updated-Lagrangian solver
"""

import logging
import numpy as np
from ..common.constants import EM20, EP30
from .shell_bt4 import _init_material_state, _layer_extra
from .shell_tri3 import _local_geometry, _char_length, _exact_dt_factor
from . import shell_nonlocal
from .shell_dkt18_kinematics import (  # noqa: F401  (re-exported)
    _A_HAMMER, _edofs, tangent, consistent_mass, kgeo,
    static_internal_forces, implicit_internal_forces,
)
from ..accel.jit_kernels.shells_dkt18 import (
    cdkcoor3, cdkderic3, cdkdefo3, cdkderi3, cdkcurv3, cdkfint3, cdkfcum3
)
from .shell_bt4 import _layer_extra, _layer_failure
from .. import materials

log = logging.getLogger(__name__)


def _vertex_angle_fractions(xe):
    """Vertex-angle fractions PTG for the lumped-mass scatter.

    Starter ``c3inmas.F:1136-1150`` computes, in the element local frame,
    ``P1 = ACOS((A2+C2-B2)/(TWO*AA*CC))/PI`` (and cyclic) — the three
    interior angles over PI — and ``spmd_msin.F:557-558`` assembles the
    nodal mass and rotary inertia as ``MS += MSTG*PTG`` /
    ``IN += INTG*PTG``.  Edge lengths give the same angles without building
    the local frame (law of cosines).  Degenerate (zero-area) elements fall
    back to uniform thirds so no NaN is ever scattered; the kernel already
    logs those elements as errors.
    """
    x1 = xe[:, 0, :]
    x2 = xe[:, 1, :]
    x3 = xe[:, 2, :]
    a2 = ((x2 - x3) ** 2).sum(axis=1)  # squared edge opposite node 1
    b2 = ((x1 - x3) ** 2).sum(axis=1)  # squared edge opposite node 2
    c2 = ((x1 - x2) ** 2).sum(axis=1)  # squared edge opposite node 3
    a = np.sqrt(a2)
    b = np.sqrt(b2)
    c = np.sqrt(c2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = np.arccos(np.clip((b2 + c2 - a2) / np.maximum(2.0 * b * c, EM20),
                               -1.0, 1.0)) / np.pi
        p2 = np.arccos(np.clip((a2 + c2 - b2) / np.maximum(2.0 * a * c, EM20),
                               -1.0, 1.0)) / np.pi
        p3 = np.arccos(np.clip((a2 + b2 - c2) / np.maximum(2.0 * a * b, EM20),
                               -1.0, 1.0)) / np.pi
    ptg = np.stack([p1, p2, p3], axis=1)
    bad = ~(np.isfinite(ptg).all(axis=1))
    if np.any(bad):
        ptg[bad] = 1.0 / 3.0
    return ptg


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter c3init3/c3mass3)."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=float)

    xe = model.x0[group.conn]
    E, xl, area, B1, B2 = _local_geometry(xe)
    bad = area <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/SH3N_DKT18 {eid}: zero area (coincident nodes?)",
                      "DKT18 INIT")

    thick = np.zeros(n)
    rho0 = np.zeros(n)
    ssp0 = np.zeros(n)
    amu = np.zeros(n)
    mat_id = np.zeros(n, dtype=np.int64)
    nip_max = 1
    _DN_DEFAULT = 1e-3
    slices = group.state.get("slices", [])
    for sl, mat, prop in slices:
        t_val = prop.params.get("thick", getattr(prop, "thick", 1.0)) if hasattr(prop, "params") else getattr(prop, "thick", 1.0)
        nip_val = int(prop.params.get("nip", getattr(prop, "nip", 1))) if hasattr(prop, "params") else int(getattr(prop, "nip", 1))
        thick[sl] = t_val
        rho0[sl] = getattr(mat, "rho0", 0.0)
        ssp0[sl] = mat.sound_speed_shell() if (hasattr(mat, "sound_speed_shell") and getattr(mat, "law", 1) != 0) else 0.0
        amu_val = float(prop.params.get("dn", 0.0) or 0.0) if hasattr(prop, "params") else _DN_DEFAULT
        amu[sl] = amu_val if amu_val != 0.0 else _DN_DEFAULT
        mat_id[sl] = int(getattr(mat, "id", 0))
        nip_max = max(nip_max, nip_val)
    mass = rho0 * thick * area

    # Through-thickness Gauss stations per part slice (same as shell_bt4)
    zw = []
    for sl, mat, prop in slices:
        nip_val = int(prop.params.get("nip", getattr(prop, "nip", 1))) if hasattr(prop, "params") else int(getattr(prop, "nip", 1))
        gp, gw = np.polynomial.legendre.leggauss(nip_val)
        zw.append((gp * 0.5, gw * 0.5))
    nk = 3 * nip_max
    group.state.update(
        sig=np.zeros((n, nk, 3)),        # in-plane stress per layer x 3 GP
        qshear=np.zeros((n, 2)),         # transverse shear stress (elastic)
        epsp=np.zeros((n, nk)),
        thick=thick,
        area0=area.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),                  # 1 alive / 0 deleted
        zw=zw,
        dtfac=_exact_dt_factor(B1, B2, area, _char_length(xl, area),
                               thick, slices),
        ssp0=ssp0,
        amu=amu,
        nip_max=nip_max,
        mat_id=mat_id,
    )
    # The non-local (regularised) path needs the plane gradients at the
    # reference configuration, which cdkfint_reg_ini.F recomputes from the
    # corner coordinates via CDKEVEC3.  Same operators, same kernel; only the
    # PX/PY outputs are used (ALPE/ALDT depend on NU and are discarded, hence
    # the zero vector).
    _alpe, _aldt, _px2, _py2, _px3, _py3, _px, _py, _pxy, _pyy, _v = (
        cdkderic3(xl[:, 1, 0], xl[:, 1, 1], xl[:, 2, 0], xl[:, 2, 1],
                  np.maximum(2.0 * area, EM20), area * thick, np.zeros(n),
                  thick ** 2))
    group.state["nl_px2"] = _px2
    group.state["nl_py2"] = _py2
    group.state["nl_px3"] = _px3
    group.state["nl_py3"] = _py3
    _init_material_state(group, nk)
    # orthotropy fiber frame (/PROP/TYPE9 SH_ORTH, TYPE16) — see shell_bt4
    from . import shell_ortho
    group.state["ortho"] = shell_ortho.build_group_ortho(
        slices, E, n, log, group.ids)
    node_idx = group.conn.reshape(-1)
    # Lumped mass + rotary inertia (starter c3inmas.F / spmd_msin.F): each
    # triangle's mass MSTG and rotary inertia INTG scatter to its nodes by
    # vertex-angle fractions PTG, not uniform thirds.
    ptg = _vertex_angle_fractions(xe)
    group.state["ptg"] = ptg
    # element rotary inertia XI = EM*(AREA/4.5 + THK^2/12) (c3inmas.F:1116;
    # the /4.5 is NINE_OVER_2, kept for compatibility with 4-node shells)
    group.state["dt_iner"] = mass * (area / 4.5 + thick ** 2 / 12.0)
    mass_c = (mass[:, None] * ptg).reshape(-1)
    inertia_c = (group.state["dt_iner"][:, None] * ptg).reshape(-1)
    return node_idx, mass_c, inertia_c


def forces(group, x, v, vr, dt, fint, mint):
    """Compute DKT18 internal forces and critical time step."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)

    xe = x[conn]
    thick = st["thick"]
    nu = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        nu[sl] = getattr(mat, "nu", 0.3)

    # Cycle 0 or velocity-free evaluation
    if dt is None or dt <= 0.0 or v is None:
        ve0 = np.zeros((n, 3, 3), dtype=float)
        re0 = np.zeros((n, 3, 3), dtype=float)
        area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly, _e_frame = cdkcoor3(xe, ve0, re0, 0.0)
        area2 = np.maximum(area2, EM20)
        vol0 = 0.5 * area2 * thick
        alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _vol00 = cdkderic3(
            xl2, yl2, xl3, yl3, area2, vol0.copy(), nu, thick**2)
        viscdt = np.sqrt(1.0 + st["amu"]**2) - st["amu"]
        dt_e = st["dtfac"] * aldt * viscdt / np.maximum(st["ssp0"], EM20)
        alive = st["off"] > 0.0
        dt_e = np.where(alive & (st["ssp0"] > 0.0), dt_e, EP30)
        # DTCDK_REG (cdkforc3.F:573) runs on every cycle, the priming pass
        # included, so the non-local bound is in the initial step too.
        store = st.get("nlocal")
        if store is not None:
            dt_nl = shell_nonlocal.dtcdk_reg(
                store, thick, st["off"], aldt, int(st["nl_imat0"]),
                st["nl_nddl"], EP30, nodadt=store.nodadt,
                dt_fac1=store.dt_fac1)
            dt_e = np.where(alive, np.minimum(dt_e, dt_nl), dt_e)
        return dt_e

    ve = v[conn]
    re = vr[conn] if vr is not None else np.zeros((n, 3, 3), dtype=float)
    nip_max = st.get("nip_max", 1)

    # 1. Geometry and local frame
    (area2, xl2, yl2, xl3, yl3, vlx, vly, vlz, rlx, rly,
     e_frame) = cdkcoor3(xe, ve, re, dt)
    e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z = e_frame

    # 2. Derivatives and interpolation constants
    area = 0.5 * np.maximum(area2, EM20)
    vol0 = area * thick
    vol00 = vol0.copy()
    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, vol00 = cdkderic3(
        xl2, yl2, xl3, yl3, area2, vol00, nu, thick**2)

    # 3. Membrane rates
    exx = np.zeros(n)
    eyy = np.zeros(n)
    exy = np.zeros(n)
    exz = np.zeros(n)
    eyz = np.zeros(n)
    epsdot = np.zeros((3, n))
    gstr = np.zeros((n, 3))
    vdef = np.zeros((n, 3))

    cdkdefo3(vlx, vly, px2, py2, px3, py3, exx, eyy, exy, exz, eyz, dt, epsdot, 0, False, gstr, vdef, False)

    f11 = np.zeros(n)
    f12 = np.zeros(n)
    f13 = np.zeros(n)
    f21 = np.zeros(n)
    f22 = np.zeros(n)
    f23 = np.zeros(n)
    f31 = np.zeros(n)
    f32 = np.zeros(n)
    f33 = np.zeros(n)

    m11 = np.zeros(n)
    m12 = np.zeros(n)
    m13 = np.zeros(n)
    m21 = np.zeros(n)
    m22 = np.zeros(n)
    m23 = np.zeros(n)
    m31 = np.zeros(n)
    m32 = np.zeros(n)
    m33 = np.zeros(n)

    epsp_old = st["epsp"].copy() if st.get("chk_fail", False) else None
    nip_of = []

    # cdkforc3.F:392 ``INLOC = IPARG(78)``.  With no /NONLOCAL the group has no
    # store, the guard below never opens, and every line of the INLOC == 0
    # cycle is the code that was there before this path existed.
    nl_store = st.get("nlocal")
    nl_local = st.get("nl_local")

    for NG in range(3):
        eta, ksi = _A_HAMMER[NG]

        # VAR_REG at the Hammer point (cdkforc3.F:392-424): the local
        # variable to regularise, interpolated from the non-local increment
        # DNL through H = (1-ETA-KSI, ETA, KSI).
        var_reg = None
        if nl_store is not None:
            nddl = st["nl_nddl"]
            ks = np.arange(nddl)
            h = np.array([1.0 - eta - ksi, eta, ksi])
            var_reg = np.zeros((int(nl_local.sum()), nddl))
            for a, node in enumerate((st["nl_pos1"], st["nl_pos2"],
                                      st["nl_pos3"])):
                var_reg += nl_store.dnl[node[nl_local][:, None] + ks] * h[a]

        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)

        kxx = np.zeros(n)
        kyy = np.zeros(n)
        kxy = np.zeros(n)
        cdkcurv3(bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3, vlz, rlx, rly, kxx, kyy, kxy)

        dexx_bend = kxx * dt
        deyy_bend = kyy * dt
        dexy_bend = kxy * dt

        vol_gp = vol0 / 3.0
        force_pg = np.zeros((n, 3))
        mom_pg = np.zeros((n, 3))

        for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
            if NG == 0:
                if getattr(mat, "law", 1) == 0 or isl >= len(st.get("zw", [])):
                    nip_of.append(0)
                else:
                    nip_of.append(len(st["zw"][isl][0]))
            mask = sl
            if not np.any(mask):
                continue
            if getattr(mat, "law", 1) == 0:
                continue

            zrel, wrel = st["zw"][isl]
            nip = len(zrel)

            for il in range(nip):
                k = NG * nip_max + il
                z = zrel[il] * thick[mask]
                gw = wrel[il]

                # Strain increment at layer k
                deps = np.stack([
                    exx[mask] + z * dexx_bend[mask],
                    eyy[mask] + z * deyy_bend[mask],
                    exy[mask] + z * dexy_bend[mask]
                ], axis=1)

                st_sig_k = st["sig"][mask, k, :].copy()
                st_epsp_k = st["epsp"][mask, k].copy()

                sig_new, epsp_new = materials.shell_update(
                    mat, st_sig_k, deps, st_epsp_k, dt, _layer_extra(st, mask, k))

                st["epsp"][mask, k] = epsp_new
                if st.get("chk_fail", False):
                    _layer_failure(st, mask, mat, k, sig_new, epsp_old, deps, dt)

                st["sig"][mask, k, :] = sig_new

                force_pg[mask] += sig_new * gw
                mom_pg[mask] += sig_new * zrel[il] * gw

                dz_vol = gw * vol_gp[mask]
                st["eint"][mask] += np.sum(0.5 * (st_sig_k + sig_new) * deps, axis=1) * dz_vol

        cdkfint3(vol_gp, thick, force_pg, mom_pg, px2, py2, px3, py3,
                 bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3,
                 f11, f12, f13, f21, f22, f23, f32, f33,
                 m11, m12, m13, m21, m22, m23)

        # cdkforc3.F:498 — CDKFINT_REG, inside the Hammer-point loop
        if nl_store is not None:
            sel = nl_local
            shell_nonlocal.cdkfint_reg(
                nl_store,
                var_reg=var_reg,
                off=st["off"][sel],
                area=area[sel],
                thk=thick[sel],
                thk0=thick[sel],
                area0=st["area0"][sel],
                pos1=st["nl_pos1"][sel], pos2=st["nl_pos2"][sel],
                pos3=st["nl_pos3"][sel],
                px2=px2[sel], py2=py2[sel], px3=px3[sel], py3=py3[sel],
                ksi=ksi, eta=eta, imat=int(st["nl_imat0"]),
                nddl=st["nl_nddl"], dt2t=EP30, nodadt=nl_store.nodadt,
                bufnl=st["nl_bufnl"], dt12=0.5 * dt, dt1=dt,
                dt_fac1=nl_store.dt_fac1,
                dt_fac_noda=nl_store.dt_fac_noda,
                dt_min_noda=nl_store.dt_min_noda,
                idtn_min=nl_store.idtn_min)

    if st.get("chk_fail", False):
        from .shell_bt4 import _element_deletion
        alive = _element_deletion(st, nip_of)
        if not alive.all():
            dead = ~alive
            f11[dead] = 0.0
            f12[dead] = 0.0
            f13[dead] = 0.0
            f21[dead] = 0.0
            f22[dead] = 0.0
            f23[dead] = 0.0
            f31[dead] = 0.0
            f32[dead] = 0.0
            f33[dead] = 0.0
            m11[dead] = 0.0
            m12[dead] = 0.0
            m13[dead] = 0.0
            m21[dead] = 0.0
            m22[dead] = 0.0
            m23[dead] = 0.0
            m31[dead] = 0.0
            m32[dead] = 0.0
            m33[dead] = 0.0

    cdkfcum3(px2, py2, px3, py3, e1x, e2x, e3x, e1y, e2y, e3y, e1z, e2z, e3z,
             f11, f12, f13, f21, f22, f23, f31, f32, f33,
             m11, m12, m13, m21, m22, m23, m31, m32, m33)

    if fint is not None:
        np.subtract.at(fint[:, 0], conn[:, 0], f11)
        np.subtract.at(fint[:, 1], conn[:, 0], f21)
        np.subtract.at(fint[:, 2], conn[:, 0], f31)
        np.subtract.at(fint[:, 0], conn[:, 1], f12)
        np.subtract.at(fint[:, 1], conn[:, 1], f22)
        np.subtract.at(fint[:, 2], conn[:, 1], f32)
        np.subtract.at(fint[:, 0], conn[:, 2], f13)
        np.subtract.at(fint[:, 1], conn[:, 2], f23)
        np.subtract.at(fint[:, 2], conn[:, 2], f33)

    if mint is not None:
        np.subtract.at(mint[:, 0], conn[:, 0], m11)
        np.subtract.at(mint[:, 1], conn[:, 0], m21)
        np.subtract.at(mint[:, 2], conn[:, 0], m31)
        np.subtract.at(mint[:, 0], conn[:, 1], m12)
        np.subtract.at(mint[:, 1], conn[:, 1], m22)
        np.subtract.at(mint[:, 2], conn[:, 1], m32)
        np.subtract.at(mint[:, 0], conn[:, 2], m13)
        np.subtract.at(mint[:, 1], conn[:, 2], m23)
        np.subtract.at(mint[:, 2], conn[:, 2], m33)

    viscdt = np.sqrt(1.0 + st["amu"]**2) - st["amu"]
    dt_e = st["dtfac"] * aldt * viscdt / np.maximum(st["ssp0"], EM20)
    alive = st["off"] > 0.0
    dt_e = np.where(alive & (st["ssp0"] > 0.0), dt_e, EP30)
    # cdkforc3.F:573 — DTCDK_REG, after CNDT3
    if nl_store is not None:
        dt_nl = shell_nonlocal.dtcdk_reg(
            nl_store, thick, st["off"], aldt, int(st["nl_imat0"]),
            st["nl_nddl"], EP30, nodadt=nl_store.nodadt,
            dt_fac1=nl_store.dt_fac1)
        dt_e = np.where(alive, np.minimum(dt_e, dt_nl), dt_e)
    return dt_e


