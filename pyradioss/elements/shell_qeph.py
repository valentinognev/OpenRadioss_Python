"""
4-node QEPH shell element (/SHELL + /PROP/SHELL, Ishell=24 — the
physically-stabilized one-point quadrature shell; the starter folds the
legacy Ishell 22/23 into 24, hm_read_prop01.F line 185-192).

Fortran origin: ``engine/source/elements/shell/coquez/`` — the cycle path

    czforc3.F    driver (gather, frame, call chain, scatter)
    czcorc.F     CZCORC1: covariant frame (CLSKEW3 IREP=0), local corner
                 coordinates, warp Z1, the CZ geometry set (X13..MY34),
                 condensed characteristic length (FACDT=5/4), rotation /
                 velocity gather, the 2nd-order rigid-rotation correction
                 and the warped-element velocity projection (czcorp5.F)
    czdef.F      CZDEF: 8 generalized strain rates VDEF + 6 hourglass
                 rates VHG (the assumed-strain expansion point)
    czstra3.F    strain increments = VDEF * dt
    cmain3.F     plane-stress material laws (shared with BT — the port
                 reuses materials.shell_update per Gauss layer)
    cndt3.F      time step:  dt = (sqrt(1+dn^2)-dn) * LL / SSP  with LL
                 the czcorc condensed length (ported for the dt CLAIM in
                 M40 on the BT kernel; here it is the native claim)
    czfintce.F   CZFINTCE: one-point (constant) internal forces/moments
    czfintn.F    CZFINTN1: the PHYSICAL (stiffness) hourglass — see below
    czproj.F     CZPROJ1/CZPROJV: antisym/sym reconstruction, warped-
                 element force re-projection, local -> global transform

Physical stabilization (the QEPH point, czfintn.F)
--------------------------------------------------
Unlike the BT viscous hourglass (chvis3.F: rate dampers + small elastic
springs scaled by the user hm/hf/hr ~ 1e-2), QEPH stabilizes its four
in-plane hourglass modes with a STIFFNESS built from the material's OWN
plane-stress moduli each cycle (Belytschko-Bindeman-style assumed strain,
same family as the M38 brick treatment solid_hexa8._phys_hourglass_law70):

* hourglass strain rates (czdef.F lines 154-164) are the h=(1,-1,1,-1)
  modal rates ORTHOGONALIZED through the CZ geometry: with centered corner
  coordinates, MX13 = (x1+x3-x2-x4)/4 and MY13 likewise, so
  ``VHG1 = h.vx/4 - MX13*exx - MY13*(by.vx)`` vanishes IDENTICALLY on any
  linear velocity field (the Flanagan-Belytschko property, in covariant
  form) — no spurious stiffness on constant-strain states;
* the modal stress state VGLAS (GBUF%HOURG, 12 slots: eta/ksi families x
  membrane/bending/shear pairs) is integrated with moduli A11 = E/(1-nu2),
  A12 = nu*A11, G*SHF — scaled by CVIS = 1 (GEO(17) after the starter's
  hm_read_prop01.F lines 227-230 swap: the card's Dn moves to GEO(13) =
  the linear DAMPING coefficient, default 0.015 = ZEP015, and GEO(17)
  becomes CVIS = 1), NOT by the BT hm/hf knobs;
* a small LINEAR viscous damper rides on top (czfintn.F HVL =
  dn*sqrt(rho*A), with the SQRT moduli A11SR = sqrt(A11) etc. of
  resol_init.F lines 687-698) — ~1.5 % of critical;
* the ELASTIC stabilization work is booked into the element INTERNAL
  energy EINT (czfintn.F lines 297/449), and ONLY the damper work TESY
  goes to the part hourglass ledger EVIS(8) (lines 443-491) — this is why
  a healthy QEPH run reports near-zero HE where BT reports percent-level;
* under plasticity the modal stress increment is mostly RELAXED once the
  combined constant+hourglass von-Mises passes the yield (lines 301-351,
  COEF=0.85/COEFH=0.999 caps).  The port evaluates the same criterion
  with the current yield rebuilt from the slice's hardening data (LAW2/
  LAW36/LAW44) and ZCFAC = 1 — the (1-ETAN/E) floor of the Fortran
  needs the per-cycle material tangent ratio which the port's material
  layer does not surface; exact for elastic laws (documented cut).

Warped elements (Z1^2 >= LM*1e-8, LM = (L13+L24)/2 the mean squared
half-diagonal — czcorc.F 397 / czcorp5.F 84): the velocities receive the FULL
projection of czcorp5.F (rigid + nodal-drilling removal through the
nodal-normal frame VQN and the 3x3 inverse DI), and the assembled forces
receive the dual re-projection of czproj.F — both ported.  Flat elements
(the plate fast path, PLAT=.TRUE.) skip both, exactly as upstream.

Conventions shared with the rest of the port (see shell_bt4):
* ``forces()`` ACCUMULATES the NEGATED internal force into ``fint`` /
  ``mint`` (cupdtn3.F ``F(1,N) = F(1,N) - F11``);
* GBUF%FOR = N/t and GBUF%MOM = M/t^2 in the Fortran (sigeps01g.F:
  MOM increment = A1*THK0/12*KXX) — the port keeps the PHYSICAL
  resultants Nres = sum(w_k sig_k), Mres = sum(w_k z_k sig_k) and folds
  the THK/THK02 factors of czfintce.F into the formulas;
* FOR(4) pairs with EYZ and FOR(5) with EXZ (mulawc.F90 line 538) —
  i.e. VSTRE(4) = sig_yz = qshear[:,1], VSTRE(5) = sig_xz = qshear[:,0];
* lumped mass rho*t*A/4 and the generous rotational inertia
  m*(t^2+A)/12 (cinmas.F FAC=TWELVE for IHBE >= 11) — identical to BT.

Drilling stiffness ``Isrot`` (P3.5) IS ported on FLAT rows: ``CZDERIRZ``
+ ``CZDEFRZ`` on the strain side (``RLZ`` = nodal spin about the normal
over area, ``czcorc.F:518-534``), ``CZFINTCRZ`` on the force side
(``VSRZ(19)`` drill stress with ``KRZ = 1e-3 * G`` per
``cncoef3.F:86-91``, nodal drill moments ``VMZ``, in-plane force
corrections, trapezoidal ``EINT`` work), leaving through ``CZPROJN``'s
flat leg. Warped rows run WITHOUT drill stiffness (loud init warning).

Deliberate cuts (documented): the drill *hourglass* stabilization
(``CZFINTNRZ``'s ``VHGZK``/``VHGZE`` rows — the ``CZFINTN1`` stabilization
keeps running unchanged); ``ISMSTR=1/11`` frozen-
geometry small strain, thermal/XFEM/non-local, the NPT=0 "global
integration" flag (the port's prop reader resolves N=0 to 3 Gauss
stations — elastically identical, and only the plastic-relaxation
weight COEF1 16-vs-25 of czfintn.F line 92-96 differs).
Implicit (P3.10): ``tangent()``/``kgeo()`` are provided — the constant
part plus the ``shell_qeph_hg`` hourglass stiffness (``GAMA``/``CHM``,
shear-modal chain) and the spin-coupled drilling regularization, proven
by finite differences of the residual in
``tests/test_p3_qbat_qeph_tangent.py``.
"""

from __future__ import annotations

import numpy as np

from .. import materials
from . import shell_ortho, shell_qeph_hg, shell_qeph_ortho
from .shell_bt4 import (_element_deletion, _init_material_state,
                        _layer_extra, _layer_failure)
from .shell_drill import KRZ_FAC, derirz, group_isrot
from .shell_ithick import apply_qeph_thickness, thin_layer_increment
from ..common.constants import EM20, EP30, SHEAR_FACTOR
from ..common.fastmath import cross3, norm3, scatter_add3

# ---------------------------------------------------------------------------
# Fortran constants (constant_mod.F)
# ---------------------------------------------------------------------------
_TOL_PLAT = 1.0e-8      # czcorc.F line 135 (double precision; EM7 single)
#   the gate compares Z1^2 < LM*TOL with LM = (L13+L24)/2 the MEAN
#   squared half-diagonal (czcorc.F line 397, passed into CZCORP5's LL
#   dummy at line 581; czcorp5.F line 84) — a length^2, dimensionally
#   consistent with Z1^2.  The czcorc.F line 408 IMPL_S>0 floor
#   (LM >= 0.1*THK^2) is not ported (warped-implicit refinement).
_CVIS = 1.0             # GEO(17) after hm_read_prop01.F 197-230 (Ishell=24)
_DN_DEFAULT = 0.015     # ZEP015 — card Dn default (hm_read_prop01.F 199)
_COEF = 0.85            # ZEP85   (czfintn.F line 87)
_COEFH = 0.999          # ZEP999  (line 86)
_STIER = 16.0 / 3.0     # FIVEP333 (line 88)
_FBEND_V = 3.464        # THREEP464 (line 90) — viscous bending factor
_UNDOUZSR = np.sqrt(1.0 / 12.0)   # UNDOUZSR (line 103)
_TOL_PLAS = 1.0e-18     # TOL (line 89)
_C7 = 4.0 / 3.0         # C7 = FOUR_OVER_3 (line 85)
_FACDT = 1.25           # FIVE_OVER_4 (czcorc.F line 379, QEPH)


# ---------------------------------------------------------------------------
# geometry: covariant frame + CZ geometry set (czcorc.F CZCORC1)
# ---------------------------------------------------------------------------

def _geometry(xe):
    """Frame + the complete CZ geometry set for the group.

    Returns a dict G with (all (n,) unless noted):
    E (n,3,3) columns e1|e2|e3; area; a_i = 1/area; z1 (warp); corx/cory
    (n,4) centered corner coords; x13,x24,y13,y24 (half-diagonals);
    mx13,mx23,mx34,my13,my23,my34 (mid-sums); l13,l24; ll (condensed
    characteristic LENGTH, czcorc lines 379-403, FACDT=5/4)."""
    n = len(xe)
    # covariant vectors R = x2+x3-x1-x4, S = x3+x4-x1-x2 (czcorc 149-154)
    r = xe[:, 1] + xe[:, 2] - xe[:, 0] - xe[:, 3]
    s = xe[:, 2] + xe[:, 3] - xe[:, 0] - xe[:, 1]
    # CLSKEW3 (cdkcoor3.F 303-399, IREP=0 = the QEPH branch):
    # e3 = (R x S)/|R x S|; e1 ~ R*|S|/|R| + S x e3 (the symmetrized axis)
    e3 = cross3(r, s)
    det = norm3(e3)
    area = 0.25 * det
    bad_det = det <= EM20
    if np.any(bad_det):
        e3 = e3.copy()
        e3[bad_det] = np.array([0.0, 0.0, 1.0])
        norm_other = norm3(e3)
        good = ~bad_det
        if np.any(good):
            e3[good] = e3[good] / det[good, None]
    else:
        e3 = e3 / np.maximum(det, EM20)[:, None]
    c1c1 = np.einsum("ni,ni->n", r, r)
    c2c2 = np.einsum("ni,ni->n", s, s)
    c2_1 = np.where(c1c1 > 0.0,
                    np.sqrt(c2c2 / np.maximum(c1c1, EM20)), 1.0)
    c1_1 = np.where(c1c1 > 0.0, 1.0,
                    np.sqrt(c1c1 / np.maximum(c2c2, EM20)))
    e1 = r * c2_1[:, None] + cross3(s, e3) * c1_1[:, None]
    norm_e1 = norm3(e1)
    bad_e1 = norm_e1 <= EM20
    if np.any(bad_e1):
        e1 = e1.copy()
        cand = np.array([1.0, 0.0, 0.0])
        dot = np.abs(np.einsum("ni,i->n", e3, cand))
        cand_alt = np.where(dot[:, None] > 0.9, np.array([0.0, 1.0, 0.0]), cand)
        e1_alt = cross3(cand_alt, e3)
        e1[bad_e1] = e1_alt[bad_e1] / np.maximum(norm3(e1_alt[bad_e1]), EM20)[:, None]
        good_e1 = ~bad_e1
        if np.any(good_e1):
            e1[good_e1] = e1[good_e1] / norm_e1[good_e1, None]
    else:
        e1 = e1 / np.maximum(norm_e1, EM20)[:, None]
    e2 = cross3(e3, e1)
    E = np.stack([e1, e2, e3], axis=2)

    # local corner coordinates relative to NODE 1 (czcorc 200-227)
    d2 = xe[:, 1] - xe[:, 0]
    d3 = xe[:, 2] - xe[:, 0]
    d4 = xe[:, 3] - xe[:, 0]
    xl2 = np.einsum("ni,ni->n", d2, e1)
    yl2 = np.einsum("ni,ni->n", d2, e2)
    xl3 = np.einsum("ni,ni->n", d3, e1)
    yl3 = np.einsum("ni,ni->n", d3, e2)
    xl4 = np.einsum("ni,ni->n", d4, e1)
    yl4 = np.einsum("ni,ni->n", d4, e2)
    center = xe.mean(axis=1)
    z1 = np.einsum("ni,ni->n", xe[:, 0] - center, e3)

    # centered corner coordinates (czcorc 336-345)
    cx0 = 0.25 * (xl2 + xl3 + xl4)
    cy0 = 0.25 * (yl2 + yl3 + yl4)
    corx = np.stack([-cx0, xl2 - cx0, xl3 - cx0, xl4 - cx0], axis=1)
    cory = np.stack([-cy0, yl2 - cy0, yl3 - cy0, yl4 - cy0], axis=1)

    x13 = 0.5 * (corx[:, 0] - corx[:, 2])
    x24 = 0.5 * (corx[:, 1] - corx[:, 3])
    y13 = 0.5 * (cory[:, 0] - cory[:, 2])
    y24 = 0.5 * (cory[:, 1] - cory[:, 3])
    mx13 = 0.5 * (corx[:, 0] + corx[:, 2])
    mx23 = 0.5 * (corx[:, 1] + corx[:, 2])
    mx34 = 0.5 * (corx[:, 2] + corx[:, 3])
    my13 = 0.5 * (cory[:, 0] + cory[:, 2])
    my23 = 0.5 * (cory[:, 1] + cory[:, 2])
    my34 = 0.5 * (cory[:, 2] + cory[:, 3])
    l13 = x13 ** 2 + y13 ** 2
    l24 = x24 ** 2 + y24 ** 2

    a_i = 1.0 / np.maximum(area, EM20)
    # taper term HS = max(|c1|,|c2|)/A (czcorc 370-372)
    hs = np.maximum(
        np.abs(corx[:, 1] * cory[:, 3] - cory[:, 1] * corx[:, 3]),
        np.abs(corx[:, 0] * cory[:, 2] - cory[:, 0] * corx[:, 2])) * a_i

    # condensed characteristic length LL (czcorc 379-403, FACDT = 5/4)
    rx = xl2 + xl3 - xl4
    ry = yl2 + yl3 - yl4
    sx = -xl2 + xl3 + xl4
    sy = -yl2 + yl3 + yl4
    c1 = np.sqrt(rx ** 2 + ry ** 2)
    c2 = np.sqrt(sx ** 2 + sy ** 2)
    cmin = np.maximum(np.minimum(c1, c2), EM20)
    fac1 = np.minimum(0.5, 0.25 * (np.maximum(c1, c2) / cmin - 1.0)) + 1.0
    fac2 = 4.0 * area / np.maximum(c1 * c2, EM20)
    fac2 = 3.413 * np.maximum(0.0, fac2 - 0.7071)
    fac2 = 0.78 + 0.22 * fac2 ** 3
    faci = 2.0 * fac1 * fac2
    lldiag = np.maximum(l13, l24)
    s1 = np.maximum(np.sqrt(faci * (_FACDT + hs) * lldiag), 1.0e-10)
    ll = area / s1

    return dict(E=E, area=area, a_i=a_i, z1=z1, corx=corx, cory=cory,
                x13=x13, x24=x24, y13=y13, y24=y24,
                mx13=mx13, mx23=mx23, mx34=mx34,
                my13=my13, my23=my23, my34=my34,
                l13=l13, l24=l24, ll=ll,
                lm=0.5 * (l13 + l24))    # czcorc.F line 397 (plat gate)


# ---------------------------------------------------------------------------
# kinematics: local velocities, rigid correction, warp projection
# (czcorc.F lines 411-610 + czcorp5.F)
# ---------------------------------------------------------------------------

def _sym3_inv(d):
    """Inverse of the packed symmetric 3x3 [d1,d2,d3,d4,d5,d6] =
    [[1,4,5],[4,2,6],[5,6,3]] — czcorp5.F lines 269-284 (A3INVDP)."""
    abc = d[:, 0] * d[:, 1] * d[:, 2]
    xxyz2 = d[:, 0] * d[:, 5] ** 2
    yyxz2 = d[:, 1] * d[:, 4] ** 2
    zzxy2 = d[:, 2] * d[:, 3] ** 2
    deta = np.abs(abc + 2.0 * d[:, 3] * d[:, 4] * d[:, 5]
                  - xxyz2 - yyxz2 - zzxy2)
    deta = 1.0 / np.maximum(deta, EM20)
    di = np.empty_like(d)
    di[:, 0] = (abc - xxyz2) * deta / np.maximum(d[:, 0], EM20)
    di[:, 1] = (abc - yyxz2) * deta / np.maximum(d[:, 1], EM20)
    di[:, 2] = (abc - zzxy2) * deta / np.maximum(d[:, 2], EM20)
    di[:, 3] = (d[:, 4] * d[:, 5] - d[:, 3] * d[:, 2]) * deta
    di[:, 4] = (d[:, 5] * d[:, 3] - d[:, 4] * d[:, 1]) * deta
    di[:, 5] = (d[:, 3] * d[:, 4] - d[:, 5] * d[:, 0]) * deta
    return di


def _kinematics(G, ve, vre, dt, npt1):
    """Local velocity set of CZCORC1 + the czcorp5 projection.

    Returns (v13, v24, vhi, rl, plat, vqn, di, db) with the SCALED
    convention of czcorc.F lines 598-610: v13, v24 carry 1/A, vhi 1/4;
    ``rl`` (n,4,2) are the projected local nodal rotation rates.
    ``vqn/di/db`` are the warped-element projection operators (None if
    every element is flat), reused by the force re-projection."""
    E, area, a_i, z1 = G["E"], G["area"], G["a_i"], G["z1"]
    x13, x24, y13, y24 = G["x13"], G["x24"], G["y13"], G["y24"]
    mx13, my13 = G["mx13"], G["my13"]
    n = len(area)

    vloc = ve @ E                                    # (n, 4, 3)
    rloc = vre @ E                                   # (n, 4, 3)
    rl = rloc[:, :, :2].copy()                       # RLXYZ
    v13 = vloc[:, 0] - vloc[:, 2]
    v24 = vloc[:, 1] - vloc[:, 3]
    vhi = vloc[:, 0] - vloc[:, 1] + vloc[:, 2] - vloc[:, 3]

    # ---- 2nd-order rigid-rotation correction (czcorc 540-568) ----------
    # V is at t+dt/2 while X is at t: remove the half-step spin so a rigid
    # rotation produces no strain at finite dt.
    if dt != 0.0:
        dt05, dt025 = 0.5 * dt, 0.25 * dt
        exz = y24 * v13[:, 2] - y13 * v24[:, 2]
        eyz = -x24 * v13[:, 2] + x13 * v24[:, 2]
        ddry = dt05 * exz * a_i
        ddrx = dt05 * eyz * a_i
        v13x, v24x, vhix = v13[:, 0].copy(), v24[:, 0].copy(), \
            vhi[:, 0].copy()
        den1 = x13 - x24
        ddrz1 = np.where(np.abs(den1) < 1.0e-10, 0.0,
                         dt025 * (v13[:, 1] - v24[:, 1])
                         / np.where(np.abs(den1) < 1.0e-10, 1.0, den1))
        v13[:, 0] -= ddry * v13[:, 2] + ddrz1 * v13[:, 1]
        v24[:, 0] -= ddry * v24[:, 2] + ddrz1 * v24[:, 1]
        vhi[:, 0] -= ddry * vhi[:, 2] + ddrz1 * vhi[:, 1]
        den2 = y13 + y24
        ddrz2 = np.where(np.abs(den2) < 1.0e-10, 0.0,
                         dt025 * (v13x + v24x)
                         / np.where(np.abs(den2) < 1.0e-10, 1.0, den2))
        v13[:, 1] -= ddrx * v13[:, 2] + ddrz2 * v13x
        v24[:, 1] -= ddrx * v24[:, 2] + ddrz2 * v24x
        vhi[:, 1] -= ddrx * vhi[:, 2] + ddrz2 * vhix

    # ---- PLAT flag + warped-element projection (czcorp5.F line 84:
    # Z2 < LM*TOL with LM the mean squared half-diagonal, or NPT==1) ----
    plat = (z1 * z1 < G["lm"] * _TOL_PLAT) | npt1
    z1[plat] = 0.0                                   # czcorp5 line 85
    vqn = di = db = None
    if not plat.all():
        w = ~plat
        vqn = np.zeros((n, 4, 3))
        di = np.zeros((n, 6))
        db = np.zeros((n, 4, 3))
        corx, cory = G["corx"], G["cory"]
        z2 = z1 * z1
        a_4 = 0.25 * area
        # nodal normals VQN (czcorp5 96-130); node order (1,3) then (2,4)
        sz1 = mx13 * y24 - my13 * x24
        sz = z2 * G["l24"]
        sl = 1.0 / np.sqrt(np.maximum(sz + (a_4 + sz1) ** 2, EM20))
        vqn[:, 0, 0] = -z1 * y24 * sl
        vqn[:, 0, 1] = z1 * x24 * sl
        vqn[:, 0, 2] = (a_4 + sz1) * sl
        sl = 1.0 / np.sqrt(np.maximum(sz + (a_4 - sz1) ** 2, EM20))
        vqn[:, 2, 0] = z1 * y24 * sl
        vqn[:, 2, 1] = -z1 * x24 * sl
        vqn[:, 2, 2] = (a_4 - sz1) * sl
        sz1 = mx13 * y13 - my13 * x13
        sz = z2 * G["l13"]
        sl = 1.0 / np.sqrt(np.maximum(sz + (a_4 + sz1) ** 2, EM20))
        vqn[:, 1, 0] = -z1 * y13 * sl
        vqn[:, 1, 1] = z1 * x13 * sl
        vqn[:, 1, 2] = (a_4 + sz1) * sl
        sl = 1.0 / np.sqrt(np.maximum(sz + (a_4 - sz1) ** 2, EM20))
        vqn[:, 3, 0] = z1 * y13 * sl
        vqn[:, 3, 1] = -z1 * x13 * sl
        vqn[:, 3, 2] = (a_4 - sz1) * sl

        rr = np.concatenate([rl, rloc[:, :, 2:3]], axis=2)  # RRXYZ (n,4,3)
        # rigid + drilling residual (czcorp5 205-221) — UNSCALED v13 etc.
        ar = np.empty((n, 3))
        ar[:, 0] = (-z1 * vhi[:, 1] + y13 * v13[:, 2] + y24 * v24[:, 2]
                    + my13 * vhi[:, 2] + rr[:, :, 0].sum(axis=1))
        ar[:, 1] = (z1 * vhi[:, 0] - x13 * v13[:, 2] - x24 * v24[:, 2]
                    - mx13 * vhi[:, 2] + rr[:, :, 1].sum(axis=1))
        ar[:, 2] = (x13 * v13[:, 1] + x24 * v24[:, 1] + mx13 * vhi[:, 1]
                    - y13 * v13[:, 0] - y24 * v24[:, 0] - my13 * vhi[:, 0]
                    + rr[:, :, 2].sum(axis=1))
        ad = np.einsum("njk,njk->nj", vqn, rr)            # (n, 4)
        # projection matrix D (czcorp5 224-253)
        xx = (corx ** 2).sum(axis=1)
        yy = (cory ** 2).sum(axis=1)
        xy = (corx * cory).sum(axis=1)
        hpat = np.array([1.0, -1.0, 1.0, -1.0])
        xz = (corx @ hpat) * z1
        yz = (cory @ hpat) * z1
        zz = 4.0 * z2
        btb = np.einsum("njk,njl->nkl", vqn, vqn)
        d = np.empty((n, 6))
        d[:, 0] = yy + zz + 4.0 - btb[:, 0, 0]
        d[:, 1] = xx + zz + 4.0 - btb[:, 1, 1]
        d[:, 2] = xx + yy + 4.0 - btb[:, 2, 2]
        d[:, 3] = -xy - btb[:, 0, 1]
        d[:, 4] = -xz - btb[:, 0, 2]
        d[:, 5] = -yz - btb[:, 1, 2]
        di_w = _sym3_inv(d)
        # DB = DI . VQN per node (czcorp5 286-297)
        dimat = np.empty((n, 3, 3))
        dimat[:, 0, 0], dimat[:, 1, 1], dimat[:, 2, 2] = \
            di_w[:, 0], di_w[:, 1], di_w[:, 2]
        dimat[:, 0, 1] = dimat[:, 1, 0] = di_w[:, 3]
        dimat[:, 0, 2] = dimat[:, 2, 0] = di_w[:, 4]
        dimat[:, 1, 2] = dimat[:, 2, 1] = di_w[:, 5]
        db_w = np.einsum("nkl,njl->njk", dimat, vqn)      # (n, 4, 3)
        dbad = np.einsum("njk,nj->nk", db_w, ad)
        alr = np.einsum("nkl,nl->nk", dimat, ar) - dbad
        ald = (ad + np.einsum("njk,nk->nj", vqn, dbad)
               - np.einsum("njk,nk->nj", db_w, ar))
        # velocity corrections (czcorp5 325-334)
        c1 = 2.0 * alr[:, 2]
        dv13 = np.zeros((n, 3))
        dv24 = np.zeros((n, 3))
        dvhi = np.zeros((n, 3))
        dv13[:, 0] = c1 * y13
        dv24[:, 0] = c1 * y24
        dvhi[:, 0] = 4.0 * (alr[:, 2] * G["my13"] - z1 * alr[:, 1])
        dv13[:, 1] = -c1 * x13
        dv24[:, 1] = -c1 * x24
        dvhi[:, 1] = -4.0 * (alr[:, 2] * G["mx13"] - z1 * alr[:, 0])
        dv13[:, 2] = -2.0 * (y13 * alr[:, 0] - x13 * alr[:, 1])
        dv24[:, 2] = -2.0 * (y24 * alr[:, 0] - x24 * alr[:, 1])
        dvhi[:, 2] = 4.0 * (G["mx13"] * alr[:, 1] - G["my13"] * alr[:, 0])
        wc = w[:, None]
        v13 += np.where(wc, dv13, 0.0)
        v24 += np.where(wc, dv24, 0.0)
        vhi += np.where(wc, dvhi, 0.0)
        # rotation projection RLXYZ = RR - ALR - VQN*ALD (czcorp5 335-343)
        rl_w = (rr[:, :, :2] - alr[:, None, :2]
                - vqn[:, :, :2] * ald[:, :, None])
        rl[w] = rl_w[w]
        # zero the operators on flat elements (only warp rows are used)
        vqn[plat] = 0.0
        di[w] = di_w[w]
        db[w] = db_w[w]

    # ---- scaling (czcorc 598-610) --------------------------------------
    v13 *= a_i[:, None]
    v24 *= a_i[:, None]
    vhi *= 0.25
    G["v13s"], G["v24s"] = v13, v24   # scaled set, reused by _drill_strain
    return v13, v24, vhi, rl, plat, vqn, di, db


# ---------------------------------------------------------------------------
# strain + hourglass rates (czdef.F CZDEF)
# ---------------------------------------------------------------------------

def _rates(G, v13, v24, vhi, rl, alive):
    """VDEF (n,8) = [exx eyy exy gxz gyz kxx kyy kxy] and VHG (n,6) —
    czdef.F lines 114-173 verbatim (with the warp additions)."""
    area, a_i, z1 = G["area"], G["a_i"], G["z1"]
    x13, x24, y13, y24 = G["x13"], G["x24"], G["y13"], G["y24"]
    mx13, mx23, mx34 = G["mx13"], G["mx23"], G["mx34"]
    my13, my23, my34 = G["my13"], G["my23"], G["my34"]
    n = len(area)

    r13 = (rl[:, 0] - rl[:, 2]) * a_i[:, None]       # (n,2) [x,y]
    r24 = (rl[:, 1] - rl[:, 3]) * a_i[:, None]
    rsom = rl.sum(axis=1) * a_i[:, None]
    rhi = 0.25 * (rl[:, 0] - rl[:, 1] + rl[:, 2] - rl[:, 3])

    vdef = np.empty((n, 8))
    # membrane
    vdef[:, 0] = y24 * v13[:, 0] - y13 * v24[:, 0]
    vdef[:, 1] = -x24 * v13[:, 1] + x13 * v24[:, 1]
    bxv2 = y24 * v13[:, 1] - y13 * v24[:, 1]
    byv1 = -x24 * v13[:, 0] + x13 * v24[:, 0]
    vdef[:, 2] = bxv2 + byv1
    # flexion
    vdef[:, 5] = y24 * r13[:, 1] - y13 * r24[:, 1]
    vdef[:, 6] = x24 * r13[:, 0] - x13 * r24[:, 0]
    bxr1 = y13 * r24[:, 0] - y24 * r13[:, 0]
    byr2 = -x24 * r13[:, 1] + x13 * r24[:, 1]
    vdef[:, 7] = bxr1 + byr2
    # transverse shear
    bcxy = 0.25 * area
    bcx = v13[:, 2] - my13 * r13[:, 0] + mx13 * r13[:, 1]
    bcy = v24[:, 2] + my13 * r24[:, 0] - mx13 * r24[:, 1]
    vdef[:, 3] = y24 * bcx - y13 * bcy + bcxy * rsom[:, 1]
    vdef[:, 4] = x13 * bcy - x24 * bcx - bcxy * rsom[:, 0]
    # hourglass rates
    vhg = np.empty((n, 6))
    vhg[:, 0] = vhi[:, 0] - mx13 * vdef[:, 0] - my13 * byv1
    vhg[:, 1] = vhi[:, 1] - mx13 * bxv2 - my13 * vdef[:, 1]
    vhg[:, 2] = rhi[:, 1] - mx13 * vdef[:, 5] - my13 * byr2
    vhg[:, 3] = -rhi[:, 0] - mx13 * bxr1 - my13 * vdef[:, 6]
    vhg[:, 4] = (vhi[:, 2] * 4.0
                 - (my13 * rsom[:, 0] - my23 * (r13[:, 0] + r24[:, 0])
                    + mx23 * (r13[:, 1] + r24[:, 1])
                    - mx13 * rsom[:, 1]) * area) * 4.0
    vhg[:, 5] = (vhi[:, 2] * 4.0
                 - (my13 * rsom[:, 0] - my34 * (r13[:, 0] - r24[:, 0])
                    + mx34 * (r13[:, 1] - r24[:, 1])
                    - mx13 * rsom[:, 1]) * area) * 4.0
    vhg[:, 0] += (y24 * v13[:, 2] - y13 * v24[:, 2]) * z1
    vhg[:, 1] += (-x24 * v13[:, 2] + x13 * v24[:, 2]) * z1
    # warp additions to the curvatures (czdef 167-173)
    deta1 = z1 * 4.0 * a_i
    vdef[:, 5] += (x13 * v13[:, 0] - x24 * v24[:, 0]) * deta1
    vdef[:, 6] += (y13 * v13[:, 1] - y24 * v24[:, 1]) * deta1
    vdef[:, 7] += (x13 * v13[:, 1] - x24 * v24[:, 1]
                   + y13 * v13[:, 0] - y24 * v24[:, 0]) * deta1
    if not alive.all():
        vdef[~alive] = 0.0
        vhg[~alive] = 0.0
    return vdef, vhg


# ---------------------------------------------------------------------------
# Starter-side initialization
# ---------------------------------------------------------------------------

def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (cinit3/cinmas, IHBE=24).

    Mass and rotational inertia are the BT lumping (cinmas.F FAC=TWELVE
    for IHBE >= 11): m_i = rho t A/4, I_i = m_i (t^2 + A)/12 — identical
    arrays feed the /DT/NODA rotational claim (engine/mass_scaling)."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state.update(
            sig=np.empty((0, 1, 3), dtype=float),
            qshear=np.empty((0, 2), dtype=float),
            epsp=np.empty((0, 1), dtype=float),
            thick=np.empty(0, dtype=float),
            area0=np.empty(0, dtype=float),
            mass=np.empty(0, dtype=float),
            eint=np.empty(0, dtype=float),
            ehour=np.empty(0, dtype=float),
            hgstr=np.empty((0, 12), dtype=float),
            vsrz=np.empty(0, dtype=float),
            zw=[],
            a11=np.empty(0, dtype=float),
            a12=np.empty(0, dtype=float),
            gmod=np.empty(0, dtype=float),
            gs=np.empty(0, dtype=float),
            shf=np.empty(0, dtype=float),
            shfsr=np.empty(0, dtype=float),
            gsr=np.empty(0, dtype=float),
            a11sr=np.empty(0, dtype=float),
            a12sr=np.empty(0, dtype=float),
            amu=np.empty(0, dtype=float),
            cspd=np.empty(0, dtype=float),
            npt1=np.empty(0, dtype=bool),
            rho0=np.empty(0, dtype=float),
            dt_iner=np.empty(0, dtype=float),
            off=np.empty(0, dtype=float),
            chk_fail=False,
            slices=[],
        )
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float), np.empty(0, dtype=float)

    xe = model.x0[group.conn]
    G = _geometry(xe)
    area = G["area"]
    bad = area <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/SHELL {eid}: zero or negative area (QEPH)",
                      "SHELL INIT")

    thick = np.zeros(n)
    rho0 = np.zeros(n)
    nip_max = 1
    for sl, mat, prop in group.state["slices"]:
        params = getattr(prop, "params", {})
        thick[sl] = params.get("thick") if "thick" in params else getattr(prop, "thick", 0.0)
        rho0[sl] = getattr(mat, "rho0", 0.0)
        nip_val = int(params.get("nip", 1)) if "nip" in params else getattr(prop, "nip", 1)
        nip_max = max(nip_max, nip_val)
    mass = rho0 * thick * area

    # through-thickness Gauss stations per slice (shared convention)
    zw = []
    for sl, mat, prop in group.state["slices"]:
        params = getattr(prop, "params", {})
        nip = int(params.get("nip", 1)) if "nip" in params else getattr(prop, "nip", 1)
        gp, gw = np.polynomial.legendre.leggauss(nip)
        zw.append((gp * 0.5, gw * 0.5))

    # ---- per-element material/stabilization coefficients (cncoef3.F
    # CNCOEF3B standard branch lines 209-261 + resol_init.F 687-698) ----
    a11 = np.zeros(n)
    a12 = np.zeros(n)
    gmod = np.zeros(n)
    shf = np.zeros(n)
    amu = np.zeros(n)
    cspd = np.ones(n)
    npt1 = np.zeros(n, dtype=bool)
    for sl, mat, prop in group.state["slices"]:
        nu = getattr(mat, "nu", 0.3)
        E_mod = getattr(mat, "E", 0.0) or getattr(mat, "e1", 0.0)
        a11[sl] = E_mod / max(1.0 - nu * nu, EM20)
        a12[sl] = nu * a11[sl]
        gmod[sl] = getattr(mat, "G", 0.0) or getattr(mat, "g5", 0.0) or getattr(mat, "g0", 0.0) or (E_mod / max(2.0 * (1.0 + nu), EM20))
        params = getattr(prop, "params", {})
        nip = int(params.get("nip", 1)) if "nip" in params else getattr(prop, "nip", 1)
        one_pt = nip == 1
        npt1[sl] = one_pt
        shf[sl] = 0.0 if one_pt else SHEAR_FACTOR    # GEO(38) default 5/6
        dn_val = float(params.get("dn", 0.0)) if "dn" in params else float(getattr(prop, "dn", 0.0))
        amu[sl] = dn_val if dn_val > 0.0 else _DN_DEFAULT
        is_law52 = getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS")
        is_law58 = getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A")
        is_law57 = getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3")
        is_law73 = getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL")
        is_law87 = getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000")
        is_law88 = getattr(mat, "law", None) in (88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP")
        is_law92 = getattr(mat, "law", None) in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE")
        is_law93 = getattr(mat, "law", None) in (93, "93", "LAW93", "ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL")
        is_law94 = getattr(mat, "law", None) in (94, "94", "LAW94", "YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH")
        is_law66 = getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB")
        is_law95 = getattr(mat, "law", None) in (95, "95", "LAW95", "BERGSTROM_BOYCE") or getattr(mat, "law_name", None) in ("95", "LAW95", "BERGSTROM_BOYCE")
        is_law100_110 = getattr(mat, "law", None) in (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                                       "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
                                                       "LAW100", "LAW101", "LAW102", "LAW103", "LAW104", "LAW105", "LAW106", "LAW107", "LAW108", "LAW109", "LAW110") or \
                        getattr(mat, "law_name", None) in ("100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
                                                           "LAW100", "LAW101", "LAW102", "LAW103", "LAW104", "LAW105", "LAW106", "LAW107", "LAW108", "LAW109", "LAW110")
        is_recent_laws = is_law95 or is_law100_110
        has_stiff = getattr(mat, "E", 0.0) > 0.0 or getattr(mat, "e1", 0.0) > 0.0 or is_law58 or is_law52 or is_law57 or is_law73 or is_law66 or is_law87 or is_law88 or is_law92 or is_law93 or is_law94 or is_recent_laws
        if getattr(mat, "rho0", 0.0) > 0.0 and has_stiff and getattr(mat, "law", 1) != 0:
            if is_recent_laws:
                try:
                    cspd[sl] = materials.sound_speed(mat, is_shell=True)
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law52:
                try:
                    from ..materials import law52_gurson
                    cspd[sl] = law52_gurson.sound_speed_shell_law52(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law58:
                try:
                    from ..materials import law58_fabr_a
                    cspd[sl] = law58_fabr_a.sound_speed_shell_law58(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law57:
                try:
                    from ..materials import law57_barlat
                    cspd[sl] = law57_barlat.sound_speed_shell_law57(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law73:
                try:
                    from ..materials import law73_hill_therm
                    cspd[sl] = law73_hill_therm.sound_speed(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law87:
                try:
                    from ..materials import law87_barlat2000
                    cspd[sl] = law87_barlat2000.sound_speed(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law88:
                try:
                    from ..materials import law88_tab_hyp
                    cspd[sl] = law88_tab_hyp.sound_speed_shell(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law92:
                try:
                    from ..materials import law92_arruda_boyce
                    cspd[sl] = law92_arruda_boyce.sound_speed_shell(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law93:
                try:
                    from ..materials import law93_orth_hill
                    cspd[sl] = law93_orth_hill.sound_speed_shell(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law94:
                try:
                    from ..materials import law94_yeoh
                    cspd[sl] = law94_yeoh.sound_speed_shell(mat, getattr(mat, "rho0", None))
                except Exception:
                    cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law66:
                cspd[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif hasattr(mat, "sound_speed_shell"):
                cspd[sl] = mat.sound_speed_shell()
            else:
                cspd[sl] = 0.0
        else:
            cspd[sl] = 0.0            # /MAT/VOID skin: claims no dt
    gs = gmod * shf
    # sqrt moduli of the linear damper (resol_init.F 687-698: GSR=sqrt(G),
    # A11SR=sqrt(A11), A12SR=NUSR*A11SR with NUSR=sqrt(nu), SHFSR=sqrt(SHF))
    gsr = np.sqrt(gmod)
    a11sr = np.sqrt(a11)
    a12sr = np.zeros(n)
    shfsr = np.sqrt(shf)
    for sl, mat, prop in group.state["slices"]:
        nu = getattr(mat, "nu", 0.3)
        a12sr[sl] = np.sqrt(max(nu, 0.0)) * a11sr[sl]

    group.state.update(
        sig=np.zeros((n, nip_max, 3)),
        qshear=np.zeros((n, 2)),
        epsp=np.zeros((n, nip_max)),
        thick=thick,
        area0=area.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        # GBUF%HOURG: the 12 persistent hourglass stresses VGLAS
        # (czfintn.F: 1-6 the eta family, 7-12 the ksi family)
        hgstr=np.zeros((n, 12)),
        vsrz=np.zeros(n),             # CZFINTCRZ drill stress VSRZ(19)
        zw=zw,
        a11=a11, a12=a12, gmod=gmod, gs=gs, shf=shf, shfsr=shfsr,
        gsr=gsr, a11sr=a11sr, a12sr=a12sr, amu=amu, cspd=cspd,
        npt1=npt1, rho0=rho0,
        yld=np.zeros(n), fmat=np.zeros(n),
    )
    _init_material_state(group, nip_max)
    group.state["ortho"] = shell_ortho.build_group_ortho(
        group.state["slices"], G["E"], n, log, group.ids)
    # cmatc3.F:211-232 — IORTH=1 for MTN==19 builds the per-element HM/HC
    # here (GEPM_LC -> CCTOGLOB) and czfintn.F:1142-1147 re-derives the
    # damper moduli from it; an isotropic group keeps None and the loop
    # above.  Placed after build_group_ortho because CCTOGLOB needs the
    # element fibre cosines.
    hm, hc = shell_qeph_ortho.group_moduli(
        group.state["slices"], group.state["ortho"], shf, n)
    group.state["hm"] = hm
    group.state["hc"] = hc
    if hm is not None:
        gsr, a11sr, a12sr = shell_qeph_ortho.damper_moduli(hm)
        group.state["gsr"] = gsr
        group.state["a11sr"] = a11sr
        group.state["a12sr"] = a12sr
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    group.state["dt_iner"] = mass / 4.0 * (thick ** 2 + area) / 12.0
    inertia_c = np.repeat(group.state["dt_iner"], 4)
    group._model = model
    if group_isrot(group) > 0 and log is not None:
        # P3.5: drill stiffness runs on flat rows only (the CZPROJN
        # warped VQN leg is unported) — say so once, loudly.
        warped0 = ~(G["z1"] ** 2 < G["lm"] * _TOL_PLAT)
        if bool(np.any(warped0)):
            log.warning(
                f"QEPH Isrot>0: {int(warped0.sum())} warped row(s) run "
                "WITHOUT drill stiffness (warped CZPROJN leg unported; "
                "flat rows carry the full CZDEFRZ/CZFINTCRZ path)",
                "SHELL INIT")
    return node_idx, mass_c, inertia_c


# ---------------------------------------------------------------------------
# constant-part internal forces (czfintce.F CZFINTCE)
# ---------------------------------------------------------------------------

def _fint_const(G, thick, Nres, Mres, qres):
    """VF (n,3,4) / VM (n,2,4) with slots 0,1 = ANTISYM (node1/node2
    patterns: f1 = -f3, f2 = -f4) and slots 2,3 = SYM — czfintce.F with
    GBUF%FOR = N/t and GBUF%MOM = M/t^2 folded into physical resultants
    (THK*VSTRE -> Nres, THK02*MSTRE -> Mres)."""
    x13, x24, y13, y24 = G["x13"], G["x24"], G["y13"], G["y24"]
    mx13, my13 = G["mx13"], G["my13"]
    n = len(x13)
    VF = np.zeros((n, 3, 4))
    VM = np.zeros((n, 2, 4))
    qx, qy = qres[:, 0], qres[:, 1]          # t*sig_xz, t*sig_yz
    N1, N2, N3 = Nres[:, 0], Nres[:, 1], Nres[:, 2]
    M1, M2, M3 = Mres[:, 0], Mres[:, 1], Mres[:, 2]

    s1geo = G["my34"] * G["mx23"] - G["my23"] * G["mx34"]

    VF[:, 0, 0] = y24 * N1 - x24 * N3
    VF[:, 1, 0] = -x24 * N2 + y24 * N3
    VF[:, 2, 0] = -x24 * qy + y24 * qx       # VSTRE(4)=yz, VSTRE(5)=xz
    VM[:, 0, 0] = x24 * M2 - y24 * M3 - my13 * VF[:, 2, 0]
    VM[:, 1, 0] = y24 * M1 - x24 * M3 + mx13 * VF[:, 2, 0]
    VM[:, 0, 2] = -s1geo * qy                # -S52S
    VM[:, 1, 2] = s1geo * qx                 # +S42S
    VF[:, 0, 1] = -y13 * N1 + x13 * N3
    VF[:, 1, 1] = x13 * N2 - y13 * N3
    VF[:, 2, 1] = x13 * qy - y13 * qx
    VM[:, 0, 1] = -x13 * M2 + y13 * M3 + my13 * VF[:, 2, 1]
    VM[:, 1, 1] = -y13 * M1 + x13 * M3 - mx13 * VF[:, 2, 1]
    VM[:, 0, 3] = VM[:, 0, 2]
    VM[:, 1, 3] = VM[:, 1, 2]
    # warp coupling of the moments into in-plane forces (czfintce 95-100)
    c2 = G["z1"] * 4.0 * G["a_i"]
    VF[:, 0, 0] += c2 * (x13 * M1 + y13 * M3)
    VF[:, 1, 0] += c2 * (y13 * M2 + x13 * M3)
    VF[:, 0, 1] -= c2 * (x24 * M1 + y24 * M3)
    VF[:, 1, 1] -= c2 * (y24 * M2 + x24 * M3)
    return VF, VM


# ---------------------------------------------------------------------------
# physical stabilization (czfintn.F CZFINTN1) — the QEPH heart
# ---------------------------------------------------------------------------

def _current_yield(st, sl, mat):
    """Per-element current yield for the hourglass relaxation criterion —
    the port's stand-in for cmain3's SIGY (see module docstring).  EP30
    marks 'elastic: criterion inert' (czfintn.F line 302)."""
    nip = st["sig"].shape[1]
    if mat.law in (2, 44):
        ep = st["epsp"][sl, :nip].mean(axis=1)
        p = mat.params
        y = p["A"] + p["B"] * np.maximum(ep, 0.0) ** p["n"]
        return np.minimum(y, p.get("sig_max", EP30))
    if mat.law == 22:
        ep = st["epsp"][sl, :nip].mean(axis=1)
        p = mat.params
        a = p.get("A", p.get("a", 0.0))
        b = p.get("B", p.get("b", 0.0))
        n_exp = p.get("N", p.get("n", 1.0))
        y = a + b * np.maximum(ep, 0.0) ** n_exp
        y = np.minimum(y, p.get("sig_max", EP30))
        eps_dam = p.get("eps_dam", EP30)
        depsl = np.maximum(0.0, ep - eps_dam)
        y = np.minimum(y, p.get("YLDL", y) + p.get("HL", 0.0) * depsl)
        return np.maximum(y, 0.0)
    if mat.law == 36 and "curve_x" in mat.params:
        ep = st["epsp"][sl, :nip].mean(axis=1)
        return np.interp(ep, mat.params["curve_x"][0],
                         mat.params["curve_y"][0])
    if (mat.law in (43, "43", "LAW43", "HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB")) and "curve_x" in mat.params and len(mat.params["curve_x"]) > 0:
        ep = st["epsp"][sl, :nip].mean(axis=1)
        return np.interp(ep, mat.params["curve_x"][0],
                         mat.params["curve_y"][0])
    if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS"):
        if "sigm" in st.get("mat_extra", {}):
            return st["mat_extra"]["sigm"][sl, :nip].mean(axis=1)
        p = getattr(mat, "params", {}) or {}
        a = float(p.get("yield_a", p.get("a", p.get("A", 0.0))))
        b = float(p.get("hard_b", p.get("b", p.get("B", 0.0))))
        n_exp = float(p.get("hard_n", p.get("n", p.get("N", 1.0))))
        ep = st["epsp"][sl, :nip].mean(axis=1)
        return a + b * np.maximum(ep, 0.0) ** n_exp
    return None                                  # elastic — no relaxation


def _fint_stab(G, st, vhg, dt, alive, Nres, Mres, VF, VM, thick):
    """czfintn.F CZFINTN1 / CZFINTN_OR: elastic modal-stress update with
    the PHYSICAL moduli (CVIS = 1), plastic relaxation, linear damper,
    generalized force assembly, energy split EINT (elastic) / EHOUR
    (damper).  The DGLAS block selects the orthotropic row set whenever
    the group carries one (see shell_qeph_ortho)."""
    area, a_i, z1 = G["area"], G["a_i"], G["z1"]
    x13, x24, y13, y24 = G["x13"], G["x24"], G["y13"], G["y24"]
    mx13, mx23, mx34 = G["mx13"], G["mx23"], G["mx34"]
    my13, my23, my34 = G["my13"], G["my23"], G["my34"]
    vg = st["hgstr"]
    off = alive.astype(float)

    fbend = np.where(st["npt1"], 0.0, 1.0 / 12.0)    # czfintn 97-102
    fbend_v = np.where(st["npt1"], 0.0, _FBEND_V)
    c6 = thick ** 2 * fbend                          # C6 = THK02*FBEND

    dhg = vhg * dt

    # ---- elastic increments DGLAS (CZFINTN_OR czfintn.F:1164-1185 with
    # IORTH>0, CZFINTN1 czfintn.F:184-227 without — both branches in
    # shell_qeph_ortho, selected on st["hm"], which is None unless
    # cmatc3.F:211-217 raised IORTH) ----
    dg = shell_qeph_ortho.elastic_increment(G, st, vhg, dt, _CVIS)

    # ---- first (old-stress) half of the trapezoidal energy --------------
    ss1o = my34 * vg[:, 0] + my23 * vg[:, 6]
    ss2o = mx23 * vg[:, 7] + mx34 * vg[:, 1]
    sf1o = my34 * vg[:, 2] + my23 * vg[:, 8]
    sf2o = -mx23 * vg[:, 9] - mx34 * vg[:, 3]
    sc5o = my34 * vg[:, 4] + mx34 * vg[:, 5]
    sc6o = my23 * vg[:, 10] + mx23 * vg[:, 11]
    c5 = 0.5 * off * thick * _C7
    esx = ss1o * dhg[:, 0] + ss2o * dhg[:, 1]
    etmp1 = c5 * (esx + 0.25 * (sc5o * dhg[:, 4] + sc6o * dhg[:, 5]))
    emx = (sf1o * dhg[:, 2] - sf2o * dhg[:, 3]) * c6
    etmp2 = c5 * emx

    # ---- elastic update (czfintn 281-299) --------------------------------
    vg += dg

    # ---- plastic relaxation (czfintn 301-351; ZCFAC = 1 port cut) --------
    for sl, mat, prop in st["slices"]:
        sigy = _current_yield(st, sl, mat)
        if sigy is None:
            continue
        t_sl = thick[sl]
        sigy2 = np.maximum(sigy * sigy, _TOL_PLAS)
        nsl = Nres[sl] / np.maximum(t_sl, EM20)[:, None]      # VSTRE
        msl = Mres[sl] / np.maximum(t_sl ** 2, EM20)[:, None]  # MSTRE
        sxy0 = (nsl[:, 0] ** 2 + nsl[:, 1] ** 2 - nsl[:, 0] * nsl[:, 1]
                + 3.0 * nsl[:, 2] ** 2)
        mxy0 = (msl[:, 0] ** 2 + msl[:, 1] ** 2 - msl[:, 0] * msl[:, 1]
                + 3.0 * msl[:, 2] ** 2)
        cnn = _COEF
        cmm = _COEF * t_sl / 16.0
        cnnx = cnn * vg[sl, 0]
        cnny = cnn * vg[sl, 1]
        cnnx_k = cnn * vg[sl, 6]
        cnny_k = cnn * vg[sl, 7]
        cmmx = cmm * vg[sl, 2]
        cmmy = cmm * vg[sl, 3]
        cmmx_k = cmm * vg[sl, 8]
        cmmy_k = cmm * vg[sl, 9]
        sxy0 += cnnx ** 2 + cnny ** 2 - cnnx * cnny
        mxy0 += cmmx ** 2 + cmmy ** 2 - cmmx * cmmy
        sxy0 += cnnx_k ** 2 + cnny_k ** 2 - cnnx_k * cnny_k
        mxy0 += cmmx_k ** 2 + cmmy_k ** 2 - cmmx_k * cmmy_k
        sxy0 += np.abs(cnnx * (2.0 * cnnx_k - cnny_k)
                       + cnny * (2.0 * cnny_k - cnnx_k))
        mxy0 += np.abs(cmmx * (2.0 * cmmx_k - cmmy_k)
                       + cmmy * (2.0 * cmmy_k - cmmx_k))
        svm = sxy0 + 25.0 * mxy0                 # COEF1 (NPT>1 branch)
        yielding = svm > sigy2
        if not np.any(yielding):
            continue
        eh1 = np.minimum(sxy0 / sigy2, 1.0) * _COEFH
        eh2 = np.full(len(sigy2), _COEFH)
        eh1 = np.where(esx[sl] < 0.0, 0.0, eh1)   # unloading keeps elastic
        eh2 = np.where(emx[sl] < 0.0, 0.0, eh2)
        eh1 = np.where(yielding, eh1, 0.0)
        eh2 = np.where(yielding, eh2, 0.0)
        for k in (0, 1, 6, 7):
            vg[sl, k] -= eh1 * dg[sl, k]
        for k in (2, 3, 8, 9):
            vg[sl, k] -= eh2 * dg[sl, k]

    # ---- generalized stresses + linear damper (czfintn 353-396) ----------
    c8 = _C7 * off
    ss1 = (my34 * vg[:, 0] + my23 * vg[:, 6]) * c8
    ss2 = (mx23 * vg[:, 7] + mx34 * vg[:, 1]) * c8
    sf1 = (my34 * vg[:, 2] + my23 * vg[:, 8]) * c8
    sf2 = -(mx23 * vg[:, 9] + mx34 * vg[:, 3]) * c8
    hsura = thick * a_i
    c2t = c8 * thick
    sc5 = (my34 * vg[:, 4] + mx34 * vg[:, 5]) * c2t
    sc6 = (my23 * vg[:, 10] + mx23 * vg[:, 11]) * c2t
    ss3 = sc5 + sc6
    # linear viscous damper: HVL = AMU*sqrt(RHO*AREA*FAC1), FAC1 = CVIS = 1
    hvl = st["amu"] * np.sqrt(st["rho0"] * area * _CVIS) * off
    ssv0 = my23 ** 2
    ssv1 = my34 ** 2
    ssv2 = mx23 ** 2
    ssv3 = mx34 ** 2
    hxx_v = _STIER * (ssv1 + ssv0)
    hxy_v = -_STIER * (my34 * mx34 + my23 * mx23)
    hyy_v = _STIER * (ssv2 + ssv3)
    c2v = hvl * st["gsr"] * st["shfsr"] * _UNDOUZSR
    cxz_v = (ssv1 + ssv3) * c2v
    cyz_v = (ssv2 + ssv0) * c2v
    aux = a_i * hvl
    c1mv = st["a11sr"] * aux
    c2mv = st["a12sr"] * aux
    cxx_v = c1mv * hxx_v
    cyy_v = c1mv * hyy_v
    cxy_v = c2mv * hxy_v
    ss1_v = cxx_v * vhg[:, 0] + cxy_v * vhg[:, 1]
    ss2_v = cyy_v * vhg[:, 1] + cxy_v * vhg[:, 0]
    sf1_v = (cxx_v * vhg[:, 2] + cxy_v * vhg[:, 3]) * fbend_v
    sf2_v = (-cyy_v * vhg[:, 3] - cxy_v * vhg[:, 2]) * fbend_v
    sc5_v = cxz_v * vhg[:, 4] * hsura
    sc6_v = cyz_v * vhg[:, 5] * hsura
    ss1t = ss1 + ss1_v
    ss2t = ss2 + ss2_v
    sc5t = sc5 + sc5_v
    sc6t = sc6 + sc6_v
    ss3t = ss3 + sc5_v + sc6_v
    sf1t = sf1 + sf1_v
    sf2t = sf2 + sf2_v

    # ---- nodal assembly (czfintn 397-434) --------------------------------
    y13s = my13 * ss3t
    x13s = mx13 * ss3t
    y34s6 = my34 * sc6t
    y23s5 = my23 * sc5t
    x23s5 = mx23 * sc5t
    x34s6 = mx34 * sc6t
    c2n = 0.25 * thick
    b13 = (my13 * x24 - mx13 * y24) * hsura
    b24 = (mx13 * y13 - my13 * x13) * hsura
    VF[:, 0, 0] += b13 * ss1t
    VF[:, 0, 2] = c2n * ss1t
    VF[:, 1, 0] += b13 * ss2t
    VF[:, 1, 2] = c2n * ss2t
    VF[:, 2, 2] = ss3t
    VF[:, 0, 1] += b24 * ss1t
    VF[:, 0, 3] = -VF[:, 0, 2]
    VF[:, 1, 1] += b24 * ss2t
    VF[:, 1, 3] = -VF[:, 1, 2]
    VF[:, 2, 3] = -VF[:, 2, 2]
    c3a = c6 * b13
    c4a = c6 * c2n
    VM[:, 0, 0] += c3a * sf2t + y23s5 + y34s6
    VM[:, 0, 2] += c4a * sf2t - y13s
    VM[:, 1, 0] += c3a * sf1t - x23s5 - x34s6
    VM[:, 1, 2] += c4a * sf1t + x13s
    c3b = c6 * b24
    VM[:, 0, 1] += c3b * sf2t + y23s5 - y34s6
    VM[:, 0, 3] += -c4a * sf2t - y13s
    VM[:, 1, 1] += c3b * sf1t - x23s5 + x34s6
    VM[:, 1, 3] += -c4a * sf1t + x13s
    c2z = z1 * hsura
    VF[:, 2, 0] += c2z * (ss1t * y24 - ss2t * x24)
    VF[:, 2, 1] += c2z * (-ss1t * y13 + ss2t * x13)

    # ---- npt=1 slices: extra viscous-only transverse damper (CZFINTNM,
    # czfintn.F 502-558 — SHF = 0 killed the elastic shear stabilization,
    # this damper alone resists the transverse hourglass) -----------------
    if st["npt1"].any():
        m1 = st["npt1"]
        c2nm = (1.0 / 12.0) * st["gmod"] * st["rho0"] * area
        hvl_nm = 25.0 * st["amu"] * np.sqrt(np.maximum(c2nm, 0.0)) * off
        cxz_nm = (my34 ** 2 + mx34 ** 2) * hvl_nm
        cyz_nm = (my23 ** 2 + mx23 ** 2) * hvl_nm
        sc5_nm = np.where(m1, cxz_nm * vhg[:, 4] * hsura, 0.0)
        sc6_nm = np.where(m1, cyz_nm * vhg[:, 5] * hsura, 0.0)
        ss3_nm = sc5_nm + sc6_nm
        VF[:, 2, 2] += ss3_nm
        VF[:, 2, 3] -= ss3_nm
        st["ehour"] += (sc5_nm * vhg[:, 4] + sc6_nm * vhg[:, 5]) * dt

    # ---- second (new-stress) half of the trapezoidal energy (czfintn
    # 437-441; note SS1t - SS1_V = the ELASTIC generalized stress) ---------
    esy = ((ss1 * dhg[:, 0] + ss2 * dhg[:, 1]) * thick
           + 0.25 * (sc5 * dhg[:, 4] + sc6 * dhg[:, 5]))
    etmp1 = etmp1 + 0.5 * esy
    emy = sf1 * dhg[:, 2] - sf2 * dhg[:, 3]
    etmp2 = etmp2 + 0.5 * c6 * emy * thick
    st["eint"] += etmp1 + etmp2
    # damper work -> the part hourglass ledger (czfintn 443-445, EVIS(8))
    tesy = ((ss1_v * dhg[:, 0] + ss2_v * dhg[:, 1]) * thick
            + (sf1_v * dhg[:, 2] - sf2_v * dhg[:, 3]) * thick * c6
            + 0.25 * (sc5_v * dhg[:, 4] + sc6_v * dhg[:, 5]))
    st["ehour"] += tesy


# ---------------------------------------------------------------------------
# reconstruction + projection to global (czproj.F CZPROJV, IFINI = 0)
# ---------------------------------------------------------------------------

def _project(G, VF, VM, plat, vqn, di, db, vmz=None):
    """FL/ML from the antisym/sym slots, the warped-element force
    re-projection, and the local -> global transform.  Returns the
    INTERNAL nodal forces/moments (n,4,3) in global axes (czproj.F).
    ``vmz`` (n,4) is the P3.5 drill moment about the local normal —
    nonzero only on flat drill rows (``CZPROJN`` flat leg)."""
    E, z1 = G["E"], G["z1"]
    n = len(z1)
    FL = np.empty((n, 3, 4))
    ML3 = np.zeros((n, 3, 4))
    for c in range(3):
        FL[:, c, 0] = VF[:, c, 0] + VF[:, c, 2]
        FL[:, c, 1] = VF[:, c, 1] + VF[:, c, 3]
        FL[:, c, 2] = -VF[:, c, 0] + VF[:, c, 2]
        FL[:, c, 3] = -VF[:, c, 1] + VF[:, c, 3]
    for c in range(2):
        ML3[:, c, 0] = VM[:, c, 0] + VM[:, c, 2]
        ML3[:, c, 1] = VM[:, c, 1] + VM[:, c, 3]
        ML3[:, c, 2] = -VM[:, c, 0] + VM[:, c, 2]
        ML3[:, c, 3] = -VM[:, c, 1] + VM[:, c, 3]
    if vmz is not None:
        ML3[:, 2, :] += vmz

    if not plat.all():
        # full re-projection (czproj.F 480-556): remove the net torque and
        # the nodal-normal (drilling) moment components consistently.
        w = ~plat
        corx, cory = G["corx"], G["cory"]
        hpat = np.array([1.0, -1.0, 1.0, -1.0])
        ar = np.empty((n, 3))
        ar[:, 0] = (-z1 * (FL[:, 1] @ hpat)
                    + np.einsum("nj,nj->n", cory, FL[:, 2])
                    + ML3[:, 0].sum(axis=1))
        ar[:, 1] = (z1 * (FL[:, 0] @ hpat)
                    - np.einsum("nj,nj->n", corx, FL[:, 2])
                    + ML3[:, 1].sum(axis=1))
        ar[:, 2] = (np.einsum("nj,nj->n", corx, FL[:, 1])
                    - np.einsum("nj,nj->n", cory, FL[:, 0]))
        ad = np.einsum("njk,nkj->nj", vqn, ML3)          # VQN . ML per node
        dbad = np.einsum("njk,nj->nk", db, ad)
        dimat = np.empty((n, 3, 3))
        dimat[:, 0, 0], dimat[:, 1, 1], dimat[:, 2, 2] = \
            di[:, 0], di[:, 1], di[:, 2]
        dimat[:, 0, 1] = dimat[:, 1, 0] = di[:, 3]
        dimat[:, 0, 2] = dimat[:, 2, 0] = di[:, 4]
        dimat[:, 1, 2] = dimat[:, 2, 1] = di[:, 5]
        alr = np.einsum("nkl,nl->nk", dimat, ar) - dbad
        ald = (ad + np.einsum("njk,nk->nj", vqn, dbad)
               - np.einsum("njk,nk->nj", db, ar))
        c1 = z1 * alr[:, 1]
        dF0 = (-c1[:, None] * hpat[None, :]
               + cory * alr[:, 2][:, None])
        c1 = z1 * alr[:, 0]
        dF1 = (c1[:, None] * hpat[None, :]
               - corx * alr[:, 2][:, None])
        dF2 = -cory * alr[:, 0][:, None] + corx * alr[:, 1][:, None]
        MM = np.empty((n, 3, 4))
        MM[:, 0] = ML3[:, 0] - alr[:, 0][:, None] - vqn[:, :, 0] * ald
        MM[:, 1] = ML3[:, 1] - alr[:, 1][:, None] - vqn[:, :, 1] * ald
        MM[:, 2] = -alr[:, 2][:, None] - vqn[:, :, 2] * ald
        wm = w[:, None]
        FL[:, 0] += np.where(wm, dF0, 0.0)
        FL[:, 1] += np.where(wm, dF1, 0.0)
        FL[:, 2] += np.where(wm, dF2, 0.0)
        ML3[w] = MM[w]

    # local -> global: F_g[n,j,b] = sum_a E[n,b,a] * FL[n,a,j]
    fg = np.einsum("nba,naj->njb", E, FL)
    mg = np.einsum("nba,naj->njb", E, ML3)
    return fg, mg


# ---------------------------------------------------------------------------
# Engine-side forces (czforc3.F)
# ---------------------------------------------------------------------------

def _pre(x_conn, v_conn, vr_conn, dt, st_npt1, alive):
    G = _geometry(x_conn)
    v13, v24, vhi, rl, plat, vqn, di, db = _kinematics(G, v_conn, vr_conn, dt, st_npt1)
    vdef, vhg = _rates(G, v13, v24, vhi, rl, alive)
    return G, vdef, vhg, plat, vqn, di, db

# ---------------------------------------------------------------------------
# Drilling stiffness (P3.5): CZDEFRZ + CZFINTCRZ, flat rows
# ---------------------------------------------------------------------------

def _drill_strain(G, vdef, vre, thick, dt, alive, plat, st):
    """CZDEFRZ (``czdef.F:212``) with the ``CZDERIRZ`` operators.

    ``rlz`` is the nodal spin about the element normal over the area
    (``czcorc.F:518-534``); the membrane augment and ``VRLZ`` use the
    scaled ``v13s``/``v24s`` set, matching ``_rates``' convention. Also
    carries the ``CZFINTCRZ`` constant-part update (``VSRZ(19)`` drill
    stress + trapezoidal ``EINT`` work — needs only ``dt``, no
    resultants). Stashes ``rlz``/``vrlz``/``bm0rz``/``vmz`` inputs on
    ``G`` for ``_drill_moments``/``_post``. Flat live rows only.
    """
    n = len(G["area"])
    flat = np.asarray(plat) & np.asarray(alive)
    rlz = np.zeros((n, 4))
    vrlz = np.zeros(n)
    b0 = np.zeros((n, 4, 4))
    if np.any(flat):
        rr = np.einsum("njk,nka->nja", vre, G["E"])
        rlz[flat] = rr[flat][:, :, 2] * G["a_i"][flat, None]
        b0, _bk, _be = derirz(G["x13"], G["x24"], G["y13"], G["y24"],
                              G["mx13"], G["mx23"], G["mx34"],
                              G["my13"], G["my23"], G["my34"],
                              G["area"])
        v13s, v24s = G["v13s"], G["v24s"]
        bxv2 = G["y24"] * v13s[:, 1] - G["y13"] * v24s[:, 1]
        byv1 = -G["x24"] * v13s[:, 0] + G["x13"] * v24s[:, 0]
        rl = rlz[flat]
        vdef[flat, 0] += (b0[flat, 0] * rl).sum(axis=1)
        vdef[flat, 1] += (b0[flat, 1] * rl).sum(axis=1)
        vdef[flat, 2] += (b0[flat, 2] * rl).sum(axis=1)
        vrlz[flat] = 0.5 * ((bxv2 - byv1)[flat]
                            + (b0[flat, 3] * rl).sum(axis=1))
        vol = G["area"][flat] * thick[flat]
        erz = vrlz[flat] * dt
        off01 = alive[flat].astype(float)
        off05 = 0.5 * off01 * vol
        vs = st["vsrz"]
        krz = KRZ_FAC * st["gmod"][flat]      # cncoef3.F: KRZ = KFAC * G
        st["eint"][flat] += vs[flat] * erz * off05
        vs[flat] = vs[flat] + krz * erz * off01
        st["eint"][flat] += vs[flat] * erz * off05
    G["rlz"], G["vrlz"], G["bm0rz"] = rlz, vrlz, b0


def _drill_moments(G, Nres, thick, alive, plat, st):
    """Nodal drill moments ``VMZ`` (``CZFINTCRZ`` force part)."""
    n = len(G["area"])
    vmz = np.zeros((n, 4))
    flat = np.asarray(plat) & np.asarray(alive)
    if np.any(flat):
        b0 = G["bm0rz"]
        vs = np.where(np.asarray(alive), st["vsrz"], 0.0)
        c2 = thick * vs * 0.5
        n1, n2, n3 = Nres[:, 0], Nres[:, 1], Nres[:, 2]
        for j in range(4):
            vmz[flat, j] = (b0[flat, 0, j] * n1[flat]
                            + b0[flat, 1, j] * n2[flat]
                            + b0[flat, 2, j] * n3[flat]) * thick[flat] \
                + b0[flat, 3, j] * c2[flat]
    return vmz


def _post(G, thick, Nres, Mres, qres, st, vhg, dt, alive, plat, vqn, di, db):
    VF, VM = _fint_const(G, thick, Nres, Mres, qres)
    if not alive.all():
        VF[~alive] = 0.0
        VM[~alive] = 0.0
    _fint_stab(G, st, vhg, dt, alive, Nres, Mres, VF, VM, thick)
    vmz = G.get("vmz")
    if vmz is not None:
        # CZFINTCRZ in-plane force corrections (nodes 1, 2 slots).
        vs = np.where(np.asarray(alive), st["vsrz"], 0.0)
        c2 = thick * vs * 0.5
        VF[:, 0, 0] += c2 * G["x24"]
        VF[:, 1, 0] += c2 * G["y24"]
        VF[:, 0, 1] -= c2 * G["x13"]
        VF[:, 1, 1] -= c2 * G["y13"]
    fg, mg = _project(G, VF, VM, plat, vqn, di, db, vmz=vmz)

    visc = np.sqrt(1.0 + st["amu"] ** 2) - st["amu"]
    dt_e = np.where(st["cspd"] > 0.0,
                    visc * G["ll"] / np.maximum(st["cspd"], EM20), EP30)
    return fg, mg, dt_e

def forces(group, x, v, vr, dt, fint, mint):
    """Compute QEPH internal forces and critical time step."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)

    xe = x[conn]
    thick = st["thick"]
    alive = st["off"] > 0.0

    # Cycle 0 or velocity-free evaluation (e.g. initial time step estimate)
    if dt is None or dt <= 0.0 or v is None:
        G = _geometry(xe)
        viscdt = np.sqrt(1.0 + st["amu"] ** 2) - st["amu"]
        # cndt3.F: dt = (sqrt(1+dn^2)-dn) * LL / SSP
        cspd = np.maximum(st["cspd"], EM20)
        dt_e = viscdt * G["ll"] / cspd
        return np.where(alive & (st["cspd"] > 0.0), dt_e, EP30)

    ve = v[conn]
    vre = vr[conn] if vr is not None else np.zeros_like(ve)

    # P3.5 drilling: group-level ISROT = IPARG(41). The numba mirrors do
    # not know the drill path, so drill groups stay on numpy.
    drill = group_isrot(group) > 0

    from pyradioss.accel import get as accel_get
    jit_pre = accel_get("qeph_pre") if not drill else None
    if jit_pre is not None:
        vdef, vhg, plat, vqn, di, db, E, area, a_i, z1, corx, cory, x13, x24, y13, y24, mx13, mx23, mx34, my13, my23, my34, l13, l24, ll, lm = jit_pre(xe, ve, vre, dt, st["npt1"], alive)
        G = dict(E=E, area=area, a_i=a_i, z1=z1, corx=corx, cory=cory, x13=x13, x24=x24, y13=y13, y24=y24, mx13=mx13, mx23=mx23, mx34=mx34, my13=my13, my23=my23, my34=my34, l13=l13, l24=l24, ll=ll, lm=lm)
    else:
        G, vdef, vhg, plat, vqn, di, db = _pre(xe, ve, vre, dt, st["npt1"], alive)

    if drill:
        _drill_strain(G, vdef, vre, thick, dt, alive, plat, st)

    dm = vdef[:, 0:3] * dt
    gsr2 = vdef[:, 3:5]
    kap = vdef[:, 5:8] * dt

    sig = st["sig"]
    if hasattr(group, "_model") and hasattr(group._model, "t"):
        st["time"] = group._model.t
    epsp_old = st["epsp"].copy() if st["chk_fail"] else None
    Nres = np.zeros((n, 3))
    Mres = np.zeros((n, 3))
    de_layers = np.zeros(n)
    dthk = np.zeros(n)                       # ITHK=1 thinning increment
    nip_of = []
    ortho_all = st.get("ortho")
    area = G["area"]
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if getattr(mat, "law", 1) == 0:
            continue
        zrel, wrel = st["zw"][isl]
        nip_of.append(len(zrel))
        t_sl = thick[sl]
        cs = ortho_all[sl] if (ortho_all is not None and getattr(
            prop, "type", 0) in shell_ortho.ORTHO_PROP_TYPES) else None
        for k in range(len(zrel)):
            zk = zrel[k] * t_sl
            wk = wrel[k] * t_sl
            deps = dm[sl] + zk[:, None] * kap[sl]
            if cs is not None:
                deps = shell_ortho.rot_strain_e2m(deps, cs)
            s_old = sig[sl, k, :].copy()
            s_new, ep_new = materials.shell_update(
                mat, sig[sl, k, :], deps, st["epsp"][sl, k], dt,
                _layer_extra(st, sl, k, area=area))
            if ep_new is not None:
                st["epsp"][sl, k] = ep_new
            if st["chk_fail"]:
                _layer_failure(st, sl, mat, k, s_new, epsp_old, deps, dt)
            sig[sl, k, :] = s_new
            s_mid = 0.5 * (s_old + s_new)
            de_layers[sl] += wk * np.einsum("nk,nk->n", s_mid, deps)
            s_res = shell_ortho.rot_stress_m2e(s_new, cs) if cs is not None else s_new
            Nres[sl] += wk[:, None] * s_res
            Mres[sl] += (wk * zk)[:, None] * s_res
            dthk[sl] += thin_layer_increment(deps, wk, getattr(mat, "nu", 0.3), st["off"][sl])
        qold = st["qshear"][sl].copy()
        st["qshear"][sl] += st["gs"][sl][:, None] * gsr2[sl] * dt
        de_layers[sl] += t_sl * np.einsum(
            "nk,nk->n", 0.5 * (qold + st["qshear"][sl]), gsr2[sl] * dt)
        if "uvar66" in st and "uvar66" in st.get("mat_extra", {}):
            st["uvar66"][sl] = st["mat_extra"]["uvar66"][sl, 0]
        if "uvar87" in st and "uvar87" in st.get("mat_extra", {}):
            u87 = st["mat_extra"]["uvar87"]
            st["uvar87"][sl] = u87[sl, 0] if u87.ndim == 3 else u87[sl]
        if "uvar88" in st.get("mat_extra", {}):
            u88 = st["mat_extra"]["uvar88"]
            if "uvar88" not in st:
                st["uvar88"] = np.zeros((n, 30))
            st["uvar88"][sl] = u88[sl, 0] if u88.ndim == 3 else u88[sl]

    if st["chk_fail"]:
        alive = _element_deletion(st, nip_of)
        if not alive.all():
            dead = ~alive
            Nres[dead] = 0.0
            Mres[dead] = 0.0
            sig[dead] = 0.0
            st["qshear"][dead] = 0.0
            st["hgstr"][dead] = 0.0
            if "mat_extra" in st:
                for name in st["mat_extra"]:
                    if name.startswith("off"):
                        st["mat_extra"][name][dead] = 0.0
    qres = st["qshear"] * thick[:, None]
    st["eint"] += area * de_layers

    if drill:
        G["vmz"] = _drill_moments(G, Nres, thick, alive, plat, st)

    jit_post = accel_get("qeph_post")
    # P3.3: the numba mirror only knows CZFINTN1's isotropic rows.  An
    # orthotropic (hm-carrying) group stays on the numpy _post — which
    # selects the CZFINTN_OR branch — rather than silently running the
    # isotropic mirror.  Porting the mirror is Phase 3 task P3.13's job.
    # P3.5: drill groups likewise stay on numpy (no drill mirror yet).
    if jit_post is not None and "E" in G and st.get("hm") is None and not drill:
        fg, mg, dt_e = jit_post(thick, Nres, Mres, qres, st["amu"], st["cspd"], st["yld"], st["fmat"], vhg, dt, alive, plat, vqn, di, db, G["E"], G["area"], G["a_i"], G["z1"], G["corx"], G["cory"], G["x13"], G["x24"], G["y13"], G["y24"], G["mx13"], G["mx23"], G["mx34"], G["my13"], G["my23"], G["my34"], G["l13"], G["l24"], G["ll"], G["lm"], st["a11"], st["a12"], st["npt1"], st["gs"], st["hgstr"])
    else:
        fg, mg, dt_e = _post(G, thick, Nres, Mres, qres, st, vhg, dt, alive, plat, vqn, di, db)
    apply_qeph_thickness(st, dthk, alive)

    flat = conn.reshape(-1)
    if fint is not None:
        scatter_add3(fint, flat, -fg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
    if mint is not None:
        scatter_add3(mint, flat, -mg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))

    return np.where(alive & (st["cspd"] > 0.0), dt_e, EP30)


# ---------------------------------------------------------------------------
# Global DOF indexing
# ---------------------------------------------------------------------------

def _edofs(conn):
    """Return global DOF indices (n, 24) for 4-node 6-DOF shell elements."""
    conn = np.asarray(conn, dtype=np.int64)
    if len(conn) == 0:
        return np.empty((0, 24), dtype=np.int64)
    base = conn[:, :, None] * 6 + np.arange(6)[None, None, :]
    return base.reshape(len(conn), 24)


# ---------------------------------------------------------------------------
# Exact analytical consistent mass matrix
# ---------------------------------------------------------------------------

def consistent_mass(group, x=None):
    """Exact analytical consistent mass matrix for 4-node QEPH shell element.

    Bilinear quad shape function integral S_ab = integral(N_a N_b dA):
        S = (A / 36) * [
            [4, 2, 1, 2],
            [2, 4, 2, 1],
            [1, 2, 4, 2],
            [2, 1, 2, 4]
        ]
    Translational block: rho * t * S (x) I_3
    Rotational block: rho * t * (t^2 / 12) * S (x) I_3

    Returns:
        M_e (n, 24, 24): Strictly positive-definite consistent mass matrix.
        edofs (n, 24): Global DOF indices.
    """
    n = group.n
    conn = group.conn
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24), dtype=float), np.empty((0, 24), dtype=np.int64)

    st = group.state
    thick = st["thick"]
    rho0 = st["rho0"]
    area = st.get("area0")
    if area is None or x is not None:
        xe = x[conn] if x is not None else group.model.x0[conn]
        G = _geometry(xe)
        area = G["area"]

    # (4, 4) reference Gram matrix
    S_ref = np.array([
        [4.0, 2.0, 1.0, 2.0],
        [2.0, 4.0, 2.0, 1.0],
        [1.0, 2.0, 4.0, 2.0],
        [2.0, 1.0, 2.0, 4.0]
    ], dtype=float) / 36.0

    Me = np.zeros((n, 24, 24), dtype=float)
    m_elem = rho0 * thick * area
    j_elem = m_elem * (thick ** 2 / 12.0)

    for a in range(4):
        for b in range(4):
            coeff_trans = (m_elem)[:, None, None] * S_ref[a, b] * np.eye(3)
            coeff_rot = (j_elem)[:, None, None] * S_ref[a, b] * np.eye(3)
            Me[:, 6 * a:6 * a + 3, 6 * b:6 * b + 3] += coeff_trans
            Me[:, 6 * a + 3:6 * a + 6, 6 * b + 3:6 * b + 6] += coeff_rot

    off = st.get("off")
    if off is not None:
        dead = off <= 0.0
        if np.any(dead):
            Me[dead] = 0.0

    return Me, _edofs(conn)


# ---------------------------------------------------------------------------
# Assumed-strain 24-DOF tangent stiffness matrix
# ---------------------------------------------------------------------------

def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for 4-node QEPH shell group.

    1-point constant membrane/bending/shear plus the hourglass stiffness
    of ``shell_qeph_hg`` and the spin-coupled drilling regularization.
    Proven by finite differences of the residual (P3.10 gate).

    Returns:
        ke (n, 24, 24): Dense element tangent matrices.
        edofs (n, 24): Global DOF indices.
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24), dtype=float), np.empty((0, 24), dtype=np.int64)

    xe = x[conn]
    G = _geometry(xe)
    R = G["E"]  # (n, 3, 3) columns e1, e2, e3
    area = np.maximum(G["area"], EM20)
    thick = st["thick"]
    corx = G["corx"]
    cory = G["cory"]

    # Flanagan-Belytschko gradient operators
    a_inv = 0.5 / area
    B1 = np.empty((n, 4), dtype=float)
    B1[:, 0] = a_inv * (cory[:, 1] - cory[:, 3])
    B1[:, 1] = a_inv * (cory[:, 2] - cory[:, 0])
    B1[:, 2] = a_inv * (cory[:, 3] - cory[:, 1])
    B1[:, 3] = a_inv * (cory[:, 0] - cory[:, 2])

    B2 = np.empty((n, 4), dtype=float)
    B2[:, 0] = a_inv * (corx[:, 3] - corx[:, 1])
    B2[:, 1] = a_inv * (corx[:, 0] - corx[:, 2])
    B2[:, 2] = a_inv * (corx[:, 1] - corx[:, 3])
    B2[:, 3] = a_inv * (corx[:, 2] - corx[:, 0])

    # 24 local DOFs: for node a in 0..3: [u_a, v_a, w_a, th_xa, th_ya, th_za]
    Bm = np.zeros((n, 3, 24), dtype=float)
    for a in range(4):
        Bm[:, 0, 6 * a] = B1[:, a]
        Bm[:, 1, 6 * a + 1] = B2[:, a]
        Bm[:, 2, 6 * a] = B2[:, a]
        Bm[:, 2, 6 * a + 1] = B1[:, a]

    Bb = np.zeros((n, 3, 24), dtype=float)
    for a in range(4):
        Bb[:, 0, 6 * a + 4] = B1[:, a]
        Bb[:, 1, 6 * a + 3] = -B2[:, a]
        Bb[:, 2, 6 * a + 3] = -B1[:, a]
        Bb[:, 2, 6 * a + 4] = B2[:, a]

    Bs = np.zeros((n, 2, 24), dtype=float)
    hpat = np.array([1.0, -1.0, 1.0, -1.0])
    mx13 = 0.5 * (corx[:, 0] + corx[:, 2])
    my13 = 0.5 * (cory[:, 0] + cory[:, 2])
    Bs[:, 0, 2::6] = B1
    Bs[:, 1, 2::6] = B2
    # shear rotation rows (czdef.F via _rates — h-weighted, not 0.25)
    Bs[:, 0, 3::6] = -my13[:, None] * (hpat * B1)
    Bs[:, 0, 4::6] = mx13[:, None] * (hpat * B1) + 0.25
    Bs[:, 1, 3::6] = -my13[:, None] * (hpat * B2) - 0.25
    Bs[:, 1, 4::6] = mx13[:, None] * (hpat * B2)

    Kl = np.zeros((n, 24, 24), dtype=float)

    for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
        if getattr(mat, "law", 1) == 0:
            continue
        t_sl = thick[sl]
        A_sl = area[sl]
        nu = getattr(mat, "nu", 0.3)
        E_mod = getattr(mat, "E", 2.1e11)
        G_mod = getattr(mat, "G", E_mod / (2.0 * (1.0 + nu)))
        kGt = SHEAR_FACTOR * G_mod * t_sl

        Bms, Bbs, Bss = Bm[sl], Bb[sl], Bs[sl]
        if getattr(mat, "law", 1) == 1:
            C = materials.shell_membrane_tangent(mat)
            Kl[sl] += (A_sl * t_sl)[:, None, None] * np.einsum("nai,ab,nbj->nij", Bms, C, Bms)
            Kl[sl] += (A_sl * t_sl ** 3 / 12.0)[:, None, None] * np.einsum("nai,ab,nbj->nij", Bbs, C, Bbs)
        else:
            zrel, wrel = st["zw"][isl]
            m = sl.stop - sl.start
            Am_ = np.zeros((m, 3, 3), dtype=float)
            Bm_ = np.zeros((m, 3, 3), dtype=float)
            Dm_ = np.zeros((m, 3, 3), dtype=float)
            for k in range(len(zrel)):
                zk = zrel[k] * t_sl
                wk = wrel[k] * t_sl
                dep_k = None if epsp_incr is None else epsp_incr[sl, k]
                Dk = materials.shell_layer_tangent(
                    mat, st["sig"][sl, k, :], st["epsp"][sl, k], dep_k,
                    extra=_layer_extra(st, sl, k))
                Am_ += wk[:, None, None] * Dk
                Bm_ += (wk * zk)[:, None, None] * Dk
                Dm_ += (wk * zk * zk)[:, None, None] * Dk
            Kl[sl] += A_sl[:, None, None] * (
                np.einsum("nai,nab,nbj->nij", Bms, Am_, Bms)
                + np.einsum("nai,nab,nbj->nij", Bms, Bm_, Bbs)
                + np.einsum("nai,nab,nbj->nij", Bbs, Bm_, Bms)
                + np.einsum("nai,nab,nbj->nij", Bbs, Dm_, Bbs))

        Kl[sl] += (A_sl * kGt)[:, None, None] * np.einsum("nai,naj->nij", Bss, Bss)

        # Spin-coupled drilling regularization (implicit-only, like the
        # BT4 rotational-hourglass term): the explicit Idrill = 0 path has
        # no spin response, so this block is outside the FD gate by design,
        # but it keeps the assembled K at rank 18 (the nullspace test).
        kdrill = (1e-3 * E_mod * t_sl ** 3 * A_sl / 12.0)[:, None, None]
        for i in range(4):
            gi = np.zeros((len(A_sl), 24), dtype=float)
            gi[:, 6 * i + 5] = 1.0
            for b in range(4):
                gi[:, 6 * b + 0] += 0.5 * B2[sl, b]
                gi[:, 6 * b + 1] += -0.5 * B1[sl, b]
            Kl[sl] += kdrill * np.einsum("ni,nj->nij", gi, gi)

    # Hourglass stiffness (shell_qeph_hg.hg_tangent: CZLKEN3 membrane/
    # bending, the shear-modal chain, and the DHZ spin term).
    Kl += shell_qeph_hg.hg_tangent(G, st["a11"], st["a12"], st["gs"],
                                   thick, area)

    # Local to global transformation via triad R = [e1, e2, e3]
    # u_global = R u_local, th_global = R th_local
    # Ke_global = Tg Kl Tg^T where Tg = blkdiag(R, R, R, R, R, R, R, R)
    Kl_blocks = Kl.reshape(n, 8, 3, 8, 3)
    ke = np.einsum("nap,nIpJq,nbq->nIaJb", R, Kl_blocks, R).reshape(n, 24, 24)

    off = st.get("off")
    if off is not None:
        dead = off <= 0.0
        if np.any(dead):
            ke[dead] = 0.0

    return ke, _edofs(conn)


# ---------------------------------------------------------------------------
# Initial-stress geometric stiffness matrix
# ---------------------------------------------------------------------------

def kgeo(group, x):
    """Initial-stress geometric stiffness matrix for 4-node QEPH shell element.

    Couples translational displacements through in-plane membrane force resultants:
        g_ab = A * (B1_a * B1_b * Nxx + B2_a * B2_b * Nyy + (B1_a * B2_b + B2_a * B1_b) * Nxy)
    Replicated over (x, y, z) translational DOFs of each node pair (a, b).

    Returns:
        k_geo (n, 24, 24): Symmetric initial stress geometric stiffness.
        edofs (n, 24): Global DOF indices.
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24), dtype=float), np.empty((0, 24), dtype=np.int64)

    xe = x[conn]
    G = _geometry(xe)
    area = np.maximum(G["area"], EM20)
    thick = st["thick"]
    corx = G["corx"]
    cory = G["cory"]

    a_inv = 0.5 / area
    B1 = np.empty((n, 4), dtype=float)
    B1[:, 0] = a_inv * (cory[:, 1] - cory[:, 3])
    B1[:, 1] = a_inv * (cory[:, 2] - cory[:, 0])
    B1[:, 2] = a_inv * (cory[:, 3] - cory[:, 1])
    B1[:, 3] = a_inv * (cory[:, 0] - cory[:, 2])

    B2 = np.empty((n, 4), dtype=float)
    B2[:, 0] = a_inv * (corx[:, 3] - corx[:, 1])
    B2[:, 1] = a_inv * (corx[:, 0] - corx[:, 2])
    B2[:, 2] = a_inv * (corx[:, 1] - corx[:, 3])
    B2[:, 3] = a_inv * (corx[:, 2] - corx[:, 0])

    Nres = np.zeros((n, 3), dtype=float)
    for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
        if getattr(mat, "law", 1) == 0:
            continue
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        for k in range(len(zrel)):
            wk = wrel[k] * t_sl
            Nres[sl] += wk[:, None] * st["sig"][sl, k, :3]

    Nxx = Nres[:, 0]
    Nyy = Nres[:, 1]
    Nxy = Nres[:, 2]

    k_geo = np.zeros((n, 24, 24), dtype=float)
    I3 = np.eye(3, dtype=float)

    for a in range(4):
        for b in range(4):
            gab = area * (
                B1[:, a] * B1[:, b] * Nxx
                + B2[:, a] * B2[:, b] * Nyy
                + (B1[:, a] * B2[:, b] + B2[:, a] * B1[:, b]) * Nxy
            )
            k_geo[:, 6 * a:6 * a + 3, 6 * b:6 * b + 3] += gab[:, None, None] * I3

    # local to global (same Tg rotation as tangent(); kgeo() previously
    # returned the local frame — wrong on rotated elements, P3.10)
    E = G["E"]
    k_geo = np.einsum("nap,nIpJq,nbq->nIaJb",
                      E, k_geo.reshape(n, 8, 3, 8, 3), E).reshape(n, 24, 24)

    off = st.get("off")
    if off is not None:
        dead = off <= 0.0
        if np.any(dead):
            k_geo[dead] = 0.0

    return k_geo, _edofs(conn)


# ---------------------------------------------------------------------------
# Static & Implicit Internal Forces
# ---------------------------------------------------------------------------

def static_internal_forces(group, x, u, ur, fint, mint):
    """End-configuration static assembly from the CURRENT resultants.

    Zero-rate evaluation of the explicit assembly (``forces()`` with zero
    velocities): strain increments vanish, so the updates are exact no-ops
    and the state is bit-preserved (asserted by the P3.10 test), while the
    resultant transpose assembles on THIS geometry. ``u``/``ur`` unused —
    the state was advanced by the caller's midpoint ``forces()`` (M9)."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        return
    if fint is None and mint is None:
        return
    forces(group, x, np.zeros_like(x), np.zeros_like(x), 1.0, fint, mint)


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom=False):
    """Implicit residual: linear K.u, or midpoint-advance + static (M9)."""
    n = group.n
    conn = group.conn
    if n == 0 or len(conn) == 0:
        return

    if nlgeom:
        nn = len(x_ref)
        forces(group, x_ref + 0.5 * u, u, ur, 1.0,
               np.zeros((nn, 3)), np.zeros((nn, 3)))
        static_internal_forces(group, x_ref + u, u, ur, fint, mint)
        return
    ke, edofs = tangent(group, x_ref)
    u_elem = np.zeros((n, 24), dtype=float)
    for i in range(4):
        c_nodes = conn[:, i]
        u_elem[:, 6 * i:6 * i + 3] = u[c_nodes]
        if ur is not None:
            u_elem[:, 6 * i + 3:6 * i + 6] = ur[c_nodes]

    f_elem = np.einsum("nij,nj->ni", ke, u_elem)
    for i in range(4):
        c_nodes = conn[:, i]
        if fint is not None:
            np.add.at(fint, c_nodes, -f_elem[:, 6 * i:6 * i + 3])
        if mint is not None:
            np.add.at(mint, c_nodes, -f_elem[:, 6 * i + 3:6 * i + 6])

