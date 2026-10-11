"""DKT18 kinematics and implicit extensions — the non-explicit half of
:mod:`pyradioss.elements.shell_dkt18`.

Split out of ``shell_dkt18.py`` by the non-local (regularised) DKT18 task, which
had to add the ``INLOC > 0`` branch to ``forces()`` and could not do it inside
the line budget ``tools/validation_data/size_budget.json`` records for that
module (750 lines, "rounded down to that count, never up").  This is a pure
move: the five functions below are byte-identical to what ``shell_dkt18.py``
held, and ``shell_dkt18`` re-exports all five, so every consumer
(``pyradioss.implicit.assembly``, ``implicit.statics``, ``engine.py``) resolves
them exactly as before.

Fortran origin: the same DKT18 sources the explicit cycle cites — the tangent,
consistent mass and geometric stiffness are the M47/M511 implicit extensions
(layered plane-stress moduli per Hammer point and per integration layer over the
DK bending operators, the drilling penalty, and the local-to-global rotation by
the corotational triad).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from .shell_bt4 import _DRILL_COEF, _layer_extra
from ..accel.jit_kernels.shells_dkt18 import (
    cdkcoor3, cdkderic3, cdkderi3, cdkfint3, cdkfcum3,
)
from .. import materials

# the explicit cycle stays in shell_dkt18 (a local import would be circular:
# shell_dkt18 re-exports everything below)

#: 3 Hammer integration points in area coordinates (eta, ksi)
_A_HAMMER = [
    (0.166666666666667, 0.666666666666667),
    (0.666666666666667, 0.166666666666667),
    (0.166666666666667, 0.166666666666667),
]

#: CST shape function product integral S_ij = int N_i N_j dA / A
_S_TRI = (np.ones((3, 3)) + np.eye(3)) / 12.0


def _edofs(conn):
    """(n, 18) global scalar DOF slot ids, node-major [ux..rz] * 3 nodes."""
    n = len(conn)
    edofs = np.empty((n, 18), dtype=np.int64)
    for i in range(3):
        for c in range(6):
            edofs[:, i * 6 + c] = conn[:, i] * 6 + c
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole DKT18 group.

    Combines CST membrane, DKT bending at 3 Hammer points, and drilling penalty.
    Returns (ke, edofs): ke (n, 18, 18) in global coordinates, edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    ve0 = np.zeros((n, 3, 3), dtype=float)
    re0 = np.zeros((n, 3, 3), dtype=float)
    (area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly,
     e_frame) = cdkcoor3(xe, ve0, re0, 0.0)
    area = 0.5 * np.maximum(area2, EM20)
    thick = st["thick"]
    vol0 = area * thick
    nu = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        nu[sl] = getattr(mat, "nu", 0.3)

    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _vol00 = cdkderic3(
        xl2, yl2, xl3, yl3, area2, vol0.copy(), nu, thick**2)

    # Membrane operator Bm (n, 3, 6) on [ux0, uy0, ux1, uy1, ux2, uy2]
    px1 = -(px2 + px3)
    py1 = -(py2 + py3)

    Bm = np.zeros((n, 3, 6))
    Bm[:, 0, 0] = px1
    Bm[:, 0, 2] = px2
    Bm[:, 0, 4] = px3
    Bm[:, 1, 1] = py1
    Bm[:, 1, 3] = py2
    Bm[:, 1, 5] = py3
    Bm[:, 2, 0] = py1
    Bm[:, 2, 1] = px1
    Bm[:, 2, 2] = py2
    Bm[:, 2, 3] = px2
    Bm[:, 2, 4] = py3
    Bm[:, 2, 5] = px3

    # DKT Bending operators Bb (n, 3, 9) at 3 Hammer points
    # on [uz0, thx0, thy0, uz1, thx1, thy1, uz2, thx2, thy2]
    Bb_gp = []
    for NG in range(3):
        eta, ksi = _A_HAMMER[NG]
        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)
        B = np.zeros((n, 3, 9))
        B[:, 0, 0] = -(bz1[:, 0] + bz1[:, 1])
        B[:, 0, 1] = brx1[:, 0]
        B[:, 0, 2] = bry1[:, 0]
        B[:, 0, 3] = bz1[:, 0]
        B[:, 0, 4] = brx1[:, 1]
        B[:, 0, 5] = bry1[:, 1]
        B[:, 0, 6] = bz1[:, 1]
        B[:, 0, 7] = brx1[:, 2]
        B[:, 0, 8] = bry1[:, 2]

        B[:, 1, 0] = -(bz2[:, 0] + bz2[:, 1])
        B[:, 1, 1] = brx2[:, 0]
        B[:, 1, 2] = bry2[:, 0]
        B[:, 1, 3] = bz2[:, 0]
        B[:, 1, 4] = brx2[:, 1]
        B[:, 1, 5] = bry2[:, 1]
        B[:, 1, 6] = bz2[:, 1]
        B[:, 1, 7] = brx2[:, 2]
        B[:, 1, 8] = bry2[:, 2]

        B[:, 2, 0] = -(bz3[:, 0] + bz3[:, 1])
        B[:, 2, 1] = brx3[:, 0]
        B[:, 2, 2] = bry3[:, 0]
        B[:, 2, 3] = bz3[:, 0]
        B[:, 2, 4] = brx3[:, 1]
        B[:, 2, 5] = bry3[:, 1]
        B[:, 2, 6] = bz3[:, 1]
        B[:, 2, 7] = brx3[:, 2]
        B[:, 2, 8] = bry3[:, 2]

        Bb_gp.append(B)

    # Local stiffness accumulation (n, 18, 18)
    Kl = np.zeros((n, 18, 18))
    kdrill = np.zeros(n)
    mem_dofs = [0, 1, 6, 7, 12, 13]
    bend_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16]
    nip_max = st.get("nip_max", 1)

    for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
        if getattr(mat, "law", 1) == 0:
            continue
        t_sl = thick[sl]
        A_sl = area[sl]
        m_count = np.count_nonzero(sl) if isinstance(sl, np.ndarray) and sl.dtype == bool else (sl.stop - sl.start if isinstance(sl, slice) else len(sl))
        if m_count == 0:
            continue

        Bms = Bm[sl]
        if getattr(mat, "law", 1) == 1:
            C = materials.shell_membrane_tangent(mat)
            Km = (A_sl * t_sl)[:, None, None] * np.einsum("nai,ab,nbj->nij", Bms, C, Bms)
            Kl[sl, :, :][np.ix_(range(m_count), mem_dofs, mem_dofs)] += Km

            D_bend = (t_sl ** 3 / 12.0)[:, None, None] * C[None, :, :]
            for NG in range(3):
                Bbs = Bb_gp[NG][sl]
                Kb_ng = (A_sl / 3.0)[:, None, None] * np.einsum("nai,nab,nbj->nij", Bbs, D_bend, Bbs)
                Kl[sl, :, :][np.ix_(range(m_count), bend_dofs, bend_dofs)] += Kb_ng
        else:
            # Multi-layer elastoplastic integration
            zrel, wrel = st["zw"][isl]
            nip = len(zrel)
            for NG in range(3):
                Am_ng = np.zeros((m_count, 3, 3))
                Bm_ng = np.zeros((m_count, 3, 3))
                Dm_ng = np.zeros((m_count, 3, 3))
                for il in range(nip):
                    k = NG * nip_max + il
                    zk = zrel[il] * t_sl
                    wk = (wrel[il] * t_sl) / 3.0
                    dep_k = None if epsp_incr is None else epsp_incr[sl, k]
                    Dk = materials.shell_layer_tangent(
                        mat, st["sig"][sl, k, :], st["epsp"][sl, k], dep_k,
                        extra=_layer_extra(st, sl, k))
                    Am_ng += wk[:, None, None] * Dk
                    Bm_ng += (wk * zk)[:, None, None] * Dk
                    Dm_ng += (wk * zk * zk)[:, None, None] * Dk

                Bbs = Bb_gp[NG][sl]
                A3 = A_sl[:, None, None]
                # Membrane-membrane
                Km_ng = A3 * np.einsum("nai,nab,nbj->nij", Bms, Am_ng, Bms)
                Kl[sl, :, :][np.ix_(range(m_count), mem_dofs, mem_dofs)] += Km_ng
                # Bending-bending
                Kb_ng = A3 * np.einsum("nai,nab,nbj->nij", Bbs, Dm_ng, Bbs)
                Kl[sl, :, :][np.ix_(range(m_count), bend_dofs, bend_dofs)] += Kb_ng
                # Membrane-bending coupling
                Kmb_ng = A3 * np.einsum("nai,nab,nbj->nij", Bms, Bm_ng, Bbs)
                Kl[sl, :, :][np.ix_(range(m_count), mem_dofs, bend_dofs)] += Kmb_ng
                Kl[sl, :, :][np.ix_(range(m_count), bend_dofs, mem_dofs)] += np.swapaxes(Kmb_ng, 1, 2)

        kdrill[sl] = _DRILL_COEF * getattr(mat, "E", 0.0) * t_sl ** 3 * A_sl / 12.0

    # Drilling penalty coupled to continuum spin omega = 0.5 * (B1.vy - B2.vx)
    px_all = [px1, px2, px3]
    py_all = [py1, py2, py3]
    wrow = np.zeros((n, 18))
    for j in range(3):
        wrow[:, j * 6 + 0] = -0.5 * py_all[j]
        wrow[:, j * 6 + 1] = 0.5 * px_all[j]

    for i in range(3):
        gi = -wrow.copy()
        gi[:, i * 6 + 5] += 1.0
        Kl += kdrill[:, None, None] * np.einsum("ni,nj->nij", gi, gi)

    # Transform from local to global frame: R = [e1, e2, e3]
    e1 = np.stack([e_frame[0], e_frame[1], e_frame[2]], axis=1)  # (n, 3)
    e2 = np.stack([e_frame[3], e_frame[4], e_frame[5]], axis=1)  # (n, 3)
    e3 = np.stack([e_frame[6], e_frame[7], e_frame[8]], axis=1)  # (n, 3)
    R = np.stack([e1, e2, e3], axis=2)                            # (n, 3, 3)

    Kl_blocks = Kl.reshape(n, 6, 3, 6, 3)
    ke = np.einsum("nap,nIpJq,nbq->nIaJb", R, Kl_blocks, R).reshape(n, 18, 18)

    # Deactivated elements
    alive = st["off"] > 0.0
    ke[~alive] = 0.0

    return ke, _edofs(conn)


def consistent_mass(group, x=None):
    """Consistent element mass matrix for the DKT18 shell.

    Translational block: rho t S (x) I3
    Rotational block: rho t (t^2 / 12) S (x) I3
    Returns (me, edofs): me (n, 18, 18) strictly positive definite, edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    mass = st["mass"]
    thick = st["thick"]
    m_trans = mass
    m_rot = mass * thick ** 2 / 12.0

    me = np.zeros((n, 18, 18))
    for a in range(3):
        for b in range(3):
            s = _S_TRI[a, b]
            ft = m_trans * s
            fr = m_rot * s
            for c in range(3):
                me[:, a * 6 + c, b * 6 + c] = ft
                me[:, a * 6 + 3 + c, b * 6 + 3 + c] = fr

    alive = st.get("off")
    if alive is not None:
        me[alive <= 0.0] = 0.0

    return me, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the DKT18 group.

    Membrane-resultant von-Karman operator over translations.
    Returns (ke, edofs): ke (n, 18, 18), edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    ve0 = np.zeros((n, 3, 3), dtype=float)
    re0 = np.zeros((n, 3, 3), dtype=float)
    (area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly,
     _e_frame) = cdkcoor3(xe, ve0, re0, 0.0)
    area = 0.5 * np.maximum(area2, EM20)
    thick = st["thick"]
    vol0 = area * thick
    nu = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        nu[sl] = getattr(mat, "nu", 0.3)

    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _vol00 = cdkderic3(
        xl2, yl2, xl3, yl3, area2, vol0.copy(), nu, thick**2)

    # Membrane force resultants N = sum_g 1/3 sum_k w_k sigma_k (force/length)
    sig = st["sig"]
    Nres = np.zeros((n, 3))
    nip_max = st.get("nip_max", 1)
    for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        nip = len(zrel)
        for NG in range(3):
            for il in range(nip):
                k = NG * nip_max + il
                wk = (wrel[il] * t_sl) / 3.0
                Nres[sl] += wk[:, None] * sig[sl, k, :]

    px1 = -(px2 + px3)
    py1 = -(py2 + py3)
    B1 = np.stack([px1, px2, px3], axis=1)  # (n, 3)
    B2 = np.stack([py1, py2, py3], axis=1)  # (n, 3)

    # g_ab = A (B1a B1b Nxx + B2a B2b Nyy + (B1a B2b + B2a B1b) Nxy)
    g = area[:, None, None] * (
        Nres[:, 0, None, None] * B1[:, :, None] * B1[:, None, :]
        + Nres[:, 1, None, None] * B2[:, :, None] * B2[:, None, :]
        + Nres[:, 2, None, None] * (B1[:, :, None] * B2[:, None, :]
                                    + B2[:, :, None] * B1[:, None, :]))

    ke = np.zeros((n, 18, 18))
    ni = 6 * np.arange(3)
    for c in range(3):
        rows = (ni + c)[:, None]
        cols = (ni + c)[None, :]
        ke[:, rows, cols] += g

    alive = st["off"] > 0.0
    ke[~alive] = 0.0
    return ke, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal forces/moments at configuration x from CURRENT layer stresses."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return
    if fint is None and mint is None:
        return

    xe = x[conn]
    ve0 = np.zeros((n, 3, 3), dtype=float)
    re0 = np.zeros((n, 3, 3), dtype=float)
    (area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly,
     e_frame) = cdkcoor3(xe, ve0, re0, 0.0)
    e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z = e_frame

    thick = st["thick"]
    nu = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        nu[sl] = getattr(mat, "nu", 0.3)

    area = 0.5 * np.maximum(area2, EM20)
    vol0 = area * thick
    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _vol00 = cdkderic3(
        xl2, yl2, xl3, yl3, area2, vol0.copy(), nu, thick**2)

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

    nip_max = st.get("nip_max", 1)
    vol_gp = vol0 / 3.0

    for NG in range(3):
        eta, ksi = _A_HAMMER[NG]
        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)

        force_pg = np.zeros((n, 3))
        mom_pg = np.zeros((n, 3))

        for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
            mask = sl
            if not np.any(mask):
                continue
            if getattr(mat, "law", 1) == 0:
                continue
            zrel, wrel = st["zw"][isl]
            nip = len(zrel)
            for il in range(nip):
                k = NG * nip_max + il
                gw = wrel[il]
                sig_k = st["sig"][mask, k, :]
                force_pg[mask] += sig_k * gw
                mom_pg[mask] += sig_k * zrel[il] * gw

        cdkfint3(vol_gp, thick, force_pg, mom_pg, px2, py2, px3, py3,
                 bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3,
                 f11, f12, f13, f21, f22, f23, f32, f33,
                 m11, m12, m13, m21, m22, m23)

    alive = st["off"] > 0.0
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


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom=False):
    """Implicit residual internal forces and moments dispatch for DKT18."""
    if group.n == 0 or len(group.conn) == 0:
        return
    from .shell_dkt18 import forces

    if not nlgeom:
        forces(group, x_ref, u, ur, 1.0, fint, mint)
    else:
        x_mid = x_ref + 0.5 * u
        x_end = x_ref + u
        junk_f = np.zeros_like(fint) if fint is not None else None
        junk_m = np.zeros_like(mint) if mint is not None else None
        forces(group, x_mid, u, ur, 1.0, junk_f, junk_m)
        static_internal_forces(group, x_end, u, ur, fint, mint)
