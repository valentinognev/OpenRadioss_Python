"""
4-node QBAT shell element (/SHELL + /PROP/SHELL, **Ishell=12** — the
Batoz-Dhatt fully integrated quadrilateral, engine numbering IHBE=11).

Fortran origin: ``engine/source/elements/shell/coqueba/`` — the cycle path

    cbaforc3.F  driver (frame, Gauss loop, material, forces, dt)
    cbacoor.F   corotational frame + flat/warped geometry, the nodal and
                Gauss-point frames of the warped element, the condensed
                characteristic length (M40 already ported the dt CLAIM;
                this module now carries the FORCE physics that goes with it)
    cbadef.F    CBADEF   in-plane + transverse-shear [B] operators and
                velocity strains per Gauss point (warped element),
                CBADEF1-equivalent flat branch (the JFT..NPLAT loops),
                CBADEFSH assumed CONSTANT membrane shear (Idrill = 0)
    cbastra3.F  strain increments  EXX..KXY = VDEF * dt
    cmain3.F -> per-layer plane-stress material laws (the port reuses
                materials.shell_update — the same plumbing as shell_bt4)
    cbavisc.F   CBAVISC numerical-damping (dn) viscous stresses
    cbaener.F   CBAENER/CBAENERS assumed-shear energy corrections
    cbafori.F   CBAFORI internal force assembly, CBAFORCT constant-shear
                membrane force from the MEAN resultant
    cbaproj.F   local 5-dof forces -> global 6-dof nodal forces/moments
                (+ the free-rigid-mode projection of the warped element)
    cndt3.F     dt claim from the condensed length (via cbacoor's LL)
    cbacoorpinch.F  CBACOORPINCH through-thickness stretch DOFs, local transform & thickness scaling
    cbadefpinch.F   CBADEFPINCH pinching transverse shear & through-thickness normal stretch rates
    cbastra3pinch.F CBASTRA3PINCH strain increments EPINCHXZ, EPINCHYZ, EZZ
    cbaforipinch.F  CBAFORIPINCH internal pinching force assembly
    cbapinchproj.F  CBAPINCHPROJ local pinching force -> global 3D nodal forces along normal
    cbapinchthk.F   CBAPINCHTHK dynamic thickness update from through-thickness strain
    cndt3pinch.F    CNDT3PINCH 3D solid sound speed & dt claim with pinching
    cupdtn3pinch.F  CUPDTN3PINCH accumulation of global pinching forces
    mulawglcpinch.F / sigeps01gpinch.F 3D Hooke's law with through-thickness stretch coupling

Formulation (Batoz & Dhatt, "Modélisation des structures par éléments
finis", vol. 3; the Q4gamma24 / DKQ12-style shell):

* **2x2 in-plane Gauss integration** — NO hourglass control of any kind:
  every mode of the bilinear quad is integrated, so the hourglass energy
  HE of a QBAT deck is IDENTICALLY zero on the element side (the only HE
  this module books is cbavisc.F's numerical-damping work, which upstream
  books to the same PARTSAV(8) hourglass slot — chvis3.F line 366 for BT,
  cbavisc.F lines 74/88 for QBAT; with the default dn = 1e-3 it is
  invisible next to IE).  Before M41 the port mapped Ishell=12 decks onto
  the BT 1-point kernel, which stored the RD-E-1000 bending energy partly
  as hourglass (HE up to 40 % of EW) — unreachable MATCH by construction.

* **Corotational frame** (clskew3.F, the K=0 branch): e3 = (r x s)/|r x s|
  from the covariant diagonal sums r = x2+x3-x1-x4, s = x3+x4-x1-x2;
  e1 bisects r and s (e1 ~ r*|s| + (s x e3)*|r|, normalized); e2 = e3 x e1.
  AREA = |r x s| / 4.

* **Flat / warped split** (cbacoor.F lines 442-451): an element whose
  center-plane offset ZL1 satisfies ZL1^2 < 1e-12 * max(L13,L24) (or a
  1-integration-point property) takes the FLAT operators — the reduced
  13/24/HI velocity system with the analytic Jacobians JAC(4) and the
  hourglass-coordinate gradients HX/HY; anything else takes the WARPED
  machinery: per-NODE frames VQN (t1/t2/normal, built from the half-edge
  directions and the warp z1), per-GAUSS-POINT frames VQG, the inverse
  in-plane Jacobian columns VJFI, and a 3x3 'free rigid mode' projection
  (matrix DI) that removes the rigid rotation from the local velocity
  field exactly (cbacoor.F lines 984-1075) and from the assembled forces
  (cbaproj.F lines 202-240).

* **Assumed strains** — the two classic Batoz treatments:
  - transverse shear: 4 mid-edge samples of the edge-projected shear,
    interpolated in (ksi, eta) (the gamma-24 field; flat: the BC(24)
    block of cbadef.F lines 108-161, warped: the BC(40) block lines
    577-712 with the edge normals VNRM and rotation arms VASTN);
  - membrane shear (Idrill = 0): one CONSTANT assumed exy over the
    element (CBADEFSH), whose force is assembled from the MEAN of the 4
    Gauss-point resultants (CBAFORCT) with the energy corrections
    CBAENER (per-GP removal) / CBAENERS (mean re-addition).

* **Through-thickness**: NIP Gauss layers per Gauss point reusing the
  established material plumbing (LAW1/2/27/36 shells + /FAIL). The layer
  state is stored (n, 4*nip, 3) with k = GP*nip_max + layer, so the
  anim/failure plumbing written for (n, nip) shells applies unchanged.
  N = 0 decks (upstream 'global integration', NPTTOT = 0 running the
  resultant law mulawglc.F/sigeps01g.F) are read as nip = 3 by the deck
  reader: for LAW1 the Gauss-layer integration reproduces the global
  N = t C e, M = t^3/12 C kappa EXACTLY (2+ point Gauss integrates z^0
  and z^2 exactly); for elastoplastic laws the layered stack is the
  DOCUMENTED approximation of the unported resultant-plasticity law.
  nip = 1 runs the single-point branch (P3.7): CBADEF1 membrane-only
  velocity strains, the distinct CBAFORI1 force assembly, SHF = 0
  (cncoef3.F zeroes the shear factor at NPT==1, so the transverse shear
  vanishes identically like upstream), forced-flat (cbacoor.F 'OR NPT==1'),
  and the CBAVISNP1 rotation damping (opposite z-forces on slots 3/4, TESY
  work into the hourglass ledger); the group NIP is stored in
  ``state["NIP"]`` (scalar when uniform). No official deck runs QBAT at
  N=1, so this branch is covered by synthetic one-element tests.

* **Numerical damping** (cbavisc.F): membrane FOR += 1.414*dn*rho*ssp*
  sqrt(CDET) * C_nu(rates), bending MOM += 0.3*t*that; dn defaults to
  1e-3 (cncoef3.F lines 426-431, IHBE==11). The damping work is booked
  to ``ehour`` — upstream's EVIS(8,MX) PARTSAV hourglass slot — with the
  SAME component bookkeeping as upstream (the membrane FXY work is not
  booked there either; the engine's EN ledger absorbs it, as upstream's
  balance does).

* **dt claim** (cndt3.F): dt_e = LC * (sqrt(1+dn^2)-dn) / ssp with LC the
  condensed characteristic length recomputed from the CURRENT geometry
  every cycle (cbacoor.F lines 1080-1099, FACDT = 4/3) — the same claim
  M40 grafted onto the BT kernel for Ishell 12 decks, now computed by the
  owning kernel (and, for warped elements, with cbacoor's exact area
  bookkeeping: FAC2 and the final LL use the sum-of-Gauss-Jacobians area,
  the LM term keeps the FLAT-area inverse — lines 926 vs 1089/1096).

* **Ipinch (pinching DOFs)**: through-thickness stretch DOFs (Ipinch=1 on
  /PROP/SHELL) with full 3D Hooke's law coupling, dynamic thickness update,
  and normal force projection along nodal normals (cbacoorpinch.F, cbadefpinch.F,
  cbastra3pinch.F, cbaforipinch.F, cbapinchproj.F, cbapinchthk.F, cndt3pinch.F).

* **Ithick**: thickness change without pinching is constant THK0 = deck value,
  matching the c04-family decks (Ithick = 0). Idrill > 0 (ISROT) is NOT
  ported — the deck reader does not surface Idrill; the Idrill=0 branch
  (assumed constant membrane shear, no drilling stiffness) is what every
  RD-E-1000 QBAT deck runs.

Implicit: ``tangent()``/``kgeo()`` are DELIBERATELY not provided — a QBAT
group in an /IMPL deck is refused by implicit/assembly.py's supported-
kernel gate with the explicit supported list (the documented refusal;
consistent_mass() is provided for completeness of the kernel contract).

Sign convention: like every kernel, forces are ACCUMULATED NEGATED into
``fint``/``mint`` (cupdtn3.F line 133: F(1,N1) = F(1,N1) - F11).
"""

from __future__ import annotations

import numpy as np

from .. import materials
from ..common.constants import EM20, EP30, SHEAR_FACTOR
from ..common.fastmath import scatter_add3
from .shell_bt4 import _init_material_state, _layer_extra, _layer_failure
from .shell_qbat_nip1 import _cbavisnp1, _fori_flat, _fori_np1, _nip_array

#: 1/sqrt(3) — the Gauss abscissa PG of cbacoor.F/cbadef.F
_PG = 0.577350269189626

#: Gauss-point (ksi, eta) coordinates, cbadef.F DATA VPG (column-major):
#: GP order 1..4 = (-,-), (+,-), (+,+), (-,+)
_VPG = np.array([[-_PG, -_PG], [_PG, -_PG], [_PG, _PG], [-_PG, _PG]])

#: bilinear corner signs
_KSI_N = np.array([-1.0, 1.0, 1.0, -1.0])
_ETA_N = np.array([-1.0, -1.0, 1.0, 1.0])

#: VKSI[j, ng] = dN_j/dksi at GP ng ; VETA[j, ng] = dN_j/deta — exactly
#: the cbacoor.F lines 167-198 table (checked term by term against the
#: closed form dN_j/dksi = ksi_j (1 + eta eta_j)/4).
_VKSI = 0.25 * _KSI_N[:, None] * (1.0 + _VPG[None, :, 1] * _ETA_N[:, None])
_VETA = 0.25 * _ETA_N[:, None] * (1.0 + _VPG[None, :, 0] * _KSI_N[:, None])

#: Shape function values N_j at Gauss point ng (cbacoorpinch.F lines 65-90)
#: _TNPG[j, ng] = 0.25 * (1 + ksi_j * ksi_g) * (1 + eta_j * eta_g)
_TNPG = 0.25 * (1.0 + _KSI_N[:, None] * _VPG[None, :, 0]) * (1.0 + _ETA_N[:, None] * _VPG[None, :, 1])

#: flat/warped split tolerance (cbacoor.F TOL=EM12, ISROT=0)
_TOL_FLAT = 1.0e-12

#: cbavisc.F constants: ONEP414 and the 0.3 bending factor (ZEP3)
_ONEP414 = 1.414
_ZEP3 = 0.3

#: QBAT condensed-length FACDT = 4/3 (cbacoor.F line 1096 FOUR_OVER_3)
_FACDT = 4.0 / 3.0

#: QBAT default numerical damping dn (cncoef3.F lines 426-431, IHBE==11)
_DN_DEFAULT = 1.0e-3

#: consistent-mass bilinear area matrix  S_ab = (1/A) * int N_a N_b dA
#: (same dimensionless factor as shell_bt4's; rows sum to 1/4)
_S_QUAD = np.array([[4.0, 2.0, 1.0, 2.0],
                    [2.0, 4.0, 2.0, 1.0],
                    [1.0, 2.0, 4.0, 2.0],
                    [2.0, 1.0, 2.0, 4.0]]) / 36.0


# ----------------------------------------------------------------------------
# corotational frame (clskew3.F, K = 0 branch)
# ----------------------------------------------------------------------------

def _frame(xe: np.ndarray):
    """QBAT corotational triad from the covariant diagonal sums.

    Returns (E (n,3,3) with COLUMNS e1,e2,e3; det = |r x s| = 4*AREA).
    clskew3.F lines 336-401 (IREP=0): e3 = normalize(r x s), e1 =
    r*sqrt(|s|^2/|r|^2) + (s x e3), normalized; e2 = e3 x e1."""
    n = len(xe)
    if n == 0:
        return np.empty((0, 3, 3), dtype=float), np.empty(0, dtype=float)
    r = xe[:, 1] + xe[:, 2] - xe[:, 0] - xe[:, 3]
    s = xe[:, 2] + xe[:, 3] - xe[:, 0] - xe[:, 1]
    e3 = np.cross(r, s)
    det = np.sqrt(np.einsum("nk,nk->n", e3, e3))
    bad_det = det <= EM20
    if np.any(bad_det):
        e3 = e3.copy()
        e3[bad_det] = np.array([0.0, 0.0, 1.0])
        good = ~bad_det
        if np.any(good):
            e3[good] = e3[good] / det[good, None]
    else:
        e3 = e3 / np.maximum(det, EM20)[:, None]

    c1c1 = np.einsum("nk,nk->n", r, r)
    c2c2 = np.einsum("nk,nk->n", s, s)
    c21 = np.where(c1c1 > 0.0, np.sqrt(c2c2 / np.maximum(c1c1, EM20)), 1.0)
    e1 = r * c21[:, None] + np.cross(s, e3)
    norm_e1 = np.sqrt(np.einsum("nk,nk->n", e1, e1))
    bad_e1 = norm_e1 <= EM20
    if np.any(bad_e1):
        e1 = e1.copy()
        cand = np.array([1.0, 0.0, 0.0])
        dot = np.abs(np.einsum("ni,i->n", e3, cand))
        cand_alt = np.where(dot[:, None] > 0.9, np.array([0.0, 1.0, 0.0]), cand)
        e1_alt = np.cross(cand_alt, e3)
        e1[bad_e1] = e1_alt[bad_e1] / np.maximum(np.sqrt(np.einsum("nk,nk->n", e1_alt[bad_e1], e1_alt[bad_e1])), EM20)[:, None]
        good_e1 = ~bad_e1
        if np.any(good_e1):
            e1[good_e1] = e1[good_e1] / norm_e1[good_e1, None]
    else:
        e1 = e1 / np.maximum(norm_e1, EM20)[:, None]
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], axis=2), det


# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter cinit3/cmass3 —
    identical lumping to shell_bt4: m_i = rho t A/4, I_i = m_i (t^2+A)/12,
    the cinmas.F FAC=TWELVE convention of the IHBE>=11 family that the
    M40 STIFR work validated against the c04 /RBODY gather)."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state.update(
            sig=np.empty((0, 4, 3), dtype=float),
            qshear=np.empty((0, 4, 2), dtype=float),
            epsp=np.empty((0, 4), dtype=float),
            forpg=np.empty((0, 4, 5), dtype=float),
            mompg=np.empty((0, 4, 3), dtype=float),
            for_mean=np.empty((0, 5), dtype=float),
            thick=np.empty(0, dtype=float),
            area0=np.empty(0, dtype=float),
            mass=np.empty(0, dtype=float),
            rho0=np.empty(0, dtype=float),
            nu0=np.empty(0, dtype=float),
            E_mod=np.empty(0, dtype=float),
            ssp0=np.empty(0, dtype=float),
            amu=np.empty(0, dtype=float),
            eint=np.empty(0, dtype=float),
            ehour=np.empty(0, dtype=float),
            zw=[],
            nip_max=1,
            slices=[],
            off=np.empty(0, dtype=float),
            chk_fail=False,
            ipinch=False,
            vpinch=np.empty((0, 4, 3), dtype=float),
            fpinch=np.empty((0, 4, 3), dtype=float),
            thk0=np.empty(0, dtype=float),
            ezzpg=np.empty((0, 4), dtype=float),
            forpg_pinch=np.empty((0, 4), dtype=float),
            mompg_pinch=np.empty((0, 4, 2), dtype=float),
            epg_pinch_xz=np.empty((0, 4), dtype=float),
            epg_pinch_yz=np.empty((0, 4), dtype=float),
            epg_pinch_zz=np.empty((0, 4), dtype=float),
        )
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float), np.empty(0, dtype=float)

    xe = model.x0[group.conn]
    E, det = _frame(xe)
    area = 0.25 * det
    bad = area <= EM20
    if np.any(bad) and log is not None:
        for eid in group.ids[bad]:
            log.error(f"/SHELL {eid}: zero or negative area", "SHELL INIT")

    thick = np.zeros(n)
    rho0 = np.zeros(n)
    nu = np.zeros(n)
    E_mod = np.zeros(n)
    amu = np.zeros(n)
    ssp = np.zeros(n)
    nip_max = 1
    nip_vals = []
    for sl, mat, prop in group.state["slices"]:
        params = getattr(prop, "params", {})
        thick[sl] = params.get("thick") if "thick" in params else getattr(prop, "thick", 0.0)
        rho0[sl] = getattr(mat, "rho0", 0.0)
        nu[sl] = getattr(mat, "nu", 0.3)
        E_mod[sl] = getattr(mat, "E", getattr(mat, "young", 0.0))
        if getattr(mat, "rho0", 0.0) > 0.0 and getattr(mat, "law", 1) != 0:
            try:
                ssp[sl] = materials.sound_speed(mat, is_shell=True)
            except Exception:
                if hasattr(mat, "sound_speed_shell"):
                    ssp[sl] = mat.sound_speed_shell()
                else:
                    ssp[sl] = 0.0
        else:
            ssp[sl] = 0.0
        dn_val = float(params.get("dn", 0.0)) if "dn" in params else float(getattr(prop, "dn", 0.0))
        amu[sl] = dn_val if dn_val > 0.0 else _DN_DEFAULT
        nip_val = int(params.get("nip", 1)) if "nip" in params else getattr(prop, "nip", 1)
        nip_max = max(nip_max, nip_val)
        nip_vals.append(nip_val)

    # QBAT pinching detection (cbacoorpinch.F / sigeps01gpinch.F)
    ipinch = bool(group.state.get("ipinch", False) or any(
        getattr(prop, "ipinch", 0) == 1
        or getattr(prop, "params", {}).get("ipinch", 0) == 1
        for _, mat, prop in group.state["slices"]
    ))
    if ipinch:
        # Recompute sound speed as solid P-wave speed: ssp = sqrt(PA1 / rho) (sigeps01gpinch.F line 133)
        for sl, mat, prop in group.state["slices"]:
            if getattr(mat, "rho0", 0.0) > 0.0 and getattr(mat, "law", 1) != 0:
                E_mod = getattr(mat, "E", getattr(mat, "young", 0.0))
                nu_val = nu[sl]
                if E_mod > 0.0 and (1.0 - 2.0 * nu_val) > EM20:
                    pa1 = E_mod * (1.0 - nu_val) / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
                    ssp[sl] = np.sqrt(pa1 / rho0[sl])

    mass = rho0 * thick * area

    zw = []
    for sl, mat, prop in group.state["slices"]:
        params = getattr(prop, "params", {})
        nip = int(params.get("nip", 1)) if "nip" in params else getattr(prop, "nip", 1)
        gp, gw = np.polynomial.legendre.leggauss(nip)
        zw.append((gp * 0.5, gw * 0.5))          # relative to thickness

    nk = 4 * nip_max                             # GP-major flat layer axis
    group.state.update(
        sig=np.zeros((n, nk, 3)),      # per-GP per-layer in-plane stress
        qshear=np.zeros((n, 4, 2)),    # per-GP transverse shear [xz, yz]
        epsp=np.zeros((n, nk)),
        forpg=np.zeros((n, 4, 5)),     # GBUF%FORPG (stress units, incl. dn
        mompg=np.zeros((n, 4, 3)),     # viscosity) / GBUF%MOMPG (M/t^2)
        for_mean=np.zeros((n, 5)),     # GBUF%FOR = 4-GP mean, previous cycle
        thick=thick,
        thk0=thick.copy(),
        area0=area.copy(),
        mass=mass,
        rho0=rho0,
        nu0=nu,
        E_mod=E_mod,
        ssp0=ssp,
        amu=amu,
        eint=np.zeros(n),
        ehour=np.zeros(n),             # cbavisc.F work -> PARTSAV(8) slot
        zw=zw,
        nip_max=nip_max,
        NIP=(int(nip_vals[0]) if nip_vals and all(
            v == nip_vals[0] for v in nip_vals) else _nip_array(
                n, group.state["slices"], nip_vals)),
        ipinch=ipinch,
        vpinch=np.zeros((n, 4, 3)),
        fpinch=np.zeros((n, 4, 3)),
        ezzpg=np.zeros((n, 4)),
        forpg_pinch=np.zeros((n, 4)),
        mompg_pinch=np.zeros((n, 4, 2)),
        epg_pinch_xz=np.zeros((n, 4)),
        epg_pinch_yz=np.zeros((n, 4)),
        epg_pinch_zz=np.zeros((n, 4)),
    )
    _init_material_state(group, nk)
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    group.state["dt_iner"] = mass / 4.0 * (thick ** 2 + area) / 12.0
    inertia_c = np.repeat(group.state["dt_iner"], 4)
    return node_idx, mass_c, inertia_c


# ----------------------------------------------------------------------------
# geometry + kinematics (cbacoor.F)
# ----------------------------------------------------------------------------

def _cbacoor(xe, ve, vre, off, dt, force_flat):
    """The full CBACOOR pass, vectorized. Returns a dict ``g`` holding the
    frame, the flat/warped index sets and every geometric operand the
    Gauss loop needs, in cbacoor.F's own naming."""
    n = len(xe)
    E, det = _frame(xe)
    area = 0.25 * det
    area_i = 1.0 / np.maximum(area, EM20)

    # local corner coordinates relative to node 1, then centered
    d = xe - xe[:, 0:1, :]                       # (n,4,3), row 0 = 0
    xl = np.einsum("njk,nka->nja", d, E)         # (n,4,3) local
    cx = xl[:, :, 0] - xl[:, :, 0].mean(axis=1)[:, None]
    cy = xl[:, :, 1] - xl[:, :, 1].mean(axis=1)[:, None]
    # ZL1 = e3 . (x1 - center): the corner z alternates +-ZL1 exactly
    zl1 = -xl[:, :, 2].mean(axis=1)

    x13 = 0.5 * (cx[:, 0] - cx[:, 2])
    x24 = 0.5 * (cx[:, 1] - cx[:, 3])
    y13 = 0.5 * (cy[:, 0] - cy[:, 2])
    y24 = 0.5 * (cy[:, 1] - cy[:, 3])
    l13 = x13 ** 2 + y13 ** 2
    l24 = x24 ** 2 + y24 ** 2
    ll = np.maximum(l13, l24)
    lm = np.maximum(np.abs(cx[:, 1] * cy[:, 3] - cy[:, 1] * cx[:, 3]),
                    np.abs(cx[:, 0] * cy[:, 2] - cy[:, 0] * cx[:, 2]))

    # velocity differences in the local frame (V13, V24, VHI)
    vg = np.empty((n, 3, 3))
    vg[:, 0] = ve[:, 0] - ve[:, 2]
    vg[:, 1] = ve[:, 1] - ve[:, 3]
    vg[:, 2] = ve[:, 0] - ve[:, 1] + ve[:, 2] - ve[:, 3]
    vloc = np.einsum("nsk,nka->nsa", vg, E)      # (n, slot, comp)
    v13, v24, vhi = vloc[:, 0].copy(), vloc[:, 1].copy(), vloc[:, 2].copy()

    # ---- explicit spin correction (cbacoor.F lines 415-437) --------------
    dt05, dt025 = 0.5 * dt, 0.25 * dt
    exz = y24 * v13[:, 2] - y13 * v24[:, 2]
    eyz = -x24 * v13[:, 2] + x13 * v24[:, 2]
    ddry = dt05 * exz * area_i
    ddrx = dt05 * eyz * area_i
    v13x, v24x, vhix = v13[:, 0].copy(), v24[:, 0].copy(), vhi[:, 0].copy()
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

    # nodal rotation rates in the local frame (3 components, warp needs z)
    rr = np.einsum("njk,nka->nja", vre, E)       # (n, node, comp)

    # ---- flat / warped split (cbacoor.F 442-451) -------------------------
    flat = (zl1 ** 2 < _TOL_FLAT * ll) | force_flat
    i_f = np.where(flat)[0]
    i_w = np.where(~flat)[0]

    g = dict(E=E, area=area, area_i=area_i, cx=cx, cy=cy, zl1=zl1,
             x13=x13, x24=x24, y13=y13, y24=y24, l13=l13, l24=l24,
             ll=ll, lm=lm, i_f=i_f, i_w=i_w, n=n)

    mx13 = 0.5 * (cx[:, 0] + cx[:, 2])
    my13 = 0.5 * (cy[:, 0] + cy[:, 2])
    mx23 = 0.5 * (cx[:, 1] + cx[:, 2])
    my23 = 0.5 * (cy[:, 1] + cy[:, 2])
    mx34 = 0.5 * (cx[:, 2] + cx[:, 3])
    my34 = 0.5 * (cy[:, 2] + cy[:, 3])
    g.update(mx13=mx13, my13=my13, mx23=mx23, my23=my23,
             mx34=mx34, my34=my34)
    # normalized half-diagonal operators (both branches use them:
    # CBADEFSH / CBAFORCT and the flat membrane rows)
    x13n = x13 * area_i
    x24n = x24 * area_i
    y13n = y13 * area_i
    y24n = y24 * area_i
    g.update(x13n=x13n, x24n=x24n, y13n=y13n, y24n=y24n)
    gama1 = -mx13 * y24 + my13 * x24
    gama2 = mx13 * y13 - my13 * x13
    g.update(gama1=gama1, gama2=gama2,
             gama1n=gama1 * area_i, gama2n=gama2 * area_i)

    # ---- FLAT pack: analytic Gauss Jacobians + hourglass gradients -------
    # (cbacoor.F lines 455-547)
    jac = np.empty((n, 4))
    j1 = (mx23 * my13 - mx13 * my23) * _PG
    j2 = -(mx13 * my34 - mx34 * my13) * _PG
    j0 = 0.25 * area
    jac[:, 0] = np.abs(j0 + j2 - j1)
    jac[:, 1] = np.abs(j0 + j2 + j1)
    jac[:, 2] = np.abs(j0 - j2 + j1)
    jac[:, 3] = np.abs(j0 - j2 - j1)
    jac = np.maximum(jac, EM20)
    hx = np.empty((n, 4))
    hy = np.empty((n, 4))
    j1h = (my23 - my34) * _PG
    j2h = -(my23 + my34) * _PG
    hx[:, 0] = j1h / jac[:, 0]
    hx[:, 1] = j2h / jac[:, 1]
    hx[:, 2] = -j1h / jac[:, 2]
    hx[:, 3] = -j2h / jac[:, 3]
    j1y = (mx34 - mx23) * _PG
    j2y = (mx34 + mx23) * _PG
    hy[:, 0] = j1y / jac[:, 0]
    hy[:, 1] = j2y / jac[:, 1]
    hy[:, 2] = -j1y / jac[:, 2]
    hy[:, 3] = -j2y / jac[:, 3]
    g.update(jac=jac, hx=hx, hy=hy)

    # flat reduced velocities/rotations: VXYZ slots (13, 24, HI),
    # RXYZ slots (13, 24, HI, TI) x (x, y)
    vxyz_f = np.stack([v13, v24, vhi], axis=1)   # (n, 3, 3) [slot, comp]
    rxyz = np.empty((n, 4, 2))
    rxyz[:, 0] = rr[:, 0, :2] - rr[:, 2, :2]
    rxyz[:, 1] = rr[:, 1, :2] - rr[:, 3, :2]
    rxyz[:, 2] = rr[:, 0, :2] - rr[:, 1, :2] + rr[:, 2, :2] - rr[:, 3, :2]
    rxyz[:, 3] = rr[:, 0, :2] + rr[:, 1, :2] + rr[:, 2, :2] + rr[:, 3, :2]
    g.update(vxyz_f=vxyz_f, rxyz=rxyz)

    # ---- WARPED pack (cbacoor.F 549-927 + 945-1075), compact on i_w ------
    if len(i_w):
        _warp_geometry(g, rr, v13, v24, vhi)

    # ---- condensed characteristic length (cbacoor.F 1080-1099) -----------
    # rx/sx sums over node-1-relative coords == centered coords (translation
    # invariant); AREA is the WARP-UPDATED area for warped elements (line
    # 926 executes before this block), AREA_I stays the flat-area inverse.
    rx = cx[:, 1] + cx[:, 2] - cx[:, 3] - cx[:, 0]
    ry = cy[:, 1] + cy[:, 2] - cy[:, 3] - cy[:, 0]
    sx = -cx[:, 1] + cx[:, 2] + cx[:, 3] - cx[:, 0]
    sy = -cy[:, 1] + cy[:, 2] + cy[:, 3] - cy[:, 0]
    c1 = np.sqrt(rx ** 2 + ry ** 2)
    c2 = np.sqrt(sx ** 2 + sy ** 2)
    cmax = np.maximum(c1, c2)
    cmin = np.maximum(np.minimum(c1, c2), EM20)
    fac1 = np.minimum(0.5, 0.25 * (cmax / cmin - 1.0)) + 1.0
    fac2 = 4.0 * g["area"] / np.maximum(c1 * c2, EM20)
    fac2 = 3.413 * np.maximum(0.0, fac2 - 0.7071)
    fac2 = 0.78 + 0.22 * fac2 ** 3
    faci = 2.0 * fac1 * fac2
    s1 = np.maximum(np.sqrt(faci * (_FACDT + lm * area_i) * ll), EM20)
    g["lc"] = g["area"] / s1

    # element OFF: zero the strain sources of dead elements (cbacoor.F
    # lines 1101-1134 zero VXYZ/RXYZ for OFFG<0; the port zeroes for
    # every dead element — its forces get the OFF factor anyway)
    dead = off <= 0.0
    if np.any(dead):
        g["vxyz_f"][dead] = 0.0
        g["rxyz"][dead] = 0.0
        if len(i_w):
            wdead = dead[i_w]
            g["vxyz_w"][wdead] = 0.0
            g["rxyz_w"][wdead] = 0.0
    return g


def _warp_geometry(g, rr, v13, v24, vhi):
    """The warped-element geometry: nodal frames VQN, edge normals VNRM +
    rotation arms VASTN, Gauss frames VQG + inverse-Jacobian columns VJFI,
    Gauss Jacobians, the warped AREA, the DI rigid-projection matrix and
    the projected nodal velocities/rotations (cbacoor.F 549-927/945-1075).
    All arrays compact over ``i_w``; results stored into ``g``."""
    iw = g["i_w"]
    m = len(iw)
    cx, cy = g["cx"][iw], g["cy"][iw]
    z1 = g["zl1"][iw]
    z2 = z1 * z1
    area = g["area"][iw]
    x13, x24 = g["x13"][iw], g["x24"][iw]
    y13, y24 = g["y13"][iw], g["y24"][iw]
    l13, l24 = g["l13"][iw], g["l24"][iw]
    mx13, my13 = g["mx13"][iw], g["my13"][iw]
    mx23, my23 = g["mx23"][iw], g["my23"][iw]
    mx34, my34 = g["mx34"][iw], g["my34"][iw]
    gama1, gama2 = g["gama1"][iw], g["gama2"][iw]

    # half-edge vectors (with warp z)
    x21 = mx23 - mx13
    x34 = 0.5 * (cx[:, 2] - cx[:, 3])
    y21 = my23 - my13
    y34 = 0.5 * (cy[:, 2] - cy[:, 3])
    l12 = np.sqrt(x21 ** 2 + y21 ** 2 + z2)
    l34 = np.sqrt(x34 ** 2 + y34 ** 2 + z2)
    x41 = mx34 - mx13
    x32 = 0.5 * (cx[:, 2] - cx[:, 1])
    y41 = my34 - my13
    y32 = 0.5 * (cy[:, 2] - cy[:, 1])
    a4 = 0.25 * area

    # ---- nodal frames VQN (t1 | t2 | n) ---------------------------------
    vqn = np.empty((m, 4, 9))

    def _t2(nrm, t1):
        return np.cross(nrm, t1)

    sl = 1.0 / np.maximum(l12, EM20)
    t1_12 = np.stack([x21 * sl, y21 * sl, -z1 * sl], axis=1)
    sz24 = z2 * l24
    sz2 = a4 - gama1
    sln = 1.0 / np.sqrt(np.maximum(sz24 + sz2 ** 2, EM20))
    n1 = np.stack([-z1 * y24 * sln, z1 * x24 * sln, sz2 * sln], axis=1)
    vqn[:, 0, 0:3] = t1_12
    vqn[:, 0, 6:9] = n1
    vqn[:, 0, 3:6] = _t2(n1, t1_12)

    sl = 1.0 / np.maximum(l34, EM20)
    t1_34 = np.stack([x34 * sl, y34 * sl, z1 * sl], axis=1)
    sz2 = a4 + gama1
    sln = 1.0 / np.sqrt(np.maximum(sz24 + sz2 ** 2, EM20))
    n3 = np.stack([z1 * y24 * sln, -z1 * x24 * sln, sz2 * sln], axis=1)
    vqn[:, 2, 0:3] = t1_34
    vqn[:, 2, 6:9] = n3
    vqn[:, 2, 3:6] = _t2(n3, t1_34)

    sz13 = z2 * l13
    sz2 = a4 + gama2
    sln = 1.0 / np.sqrt(np.maximum(sz13 + sz2 ** 2, EM20))
    n2 = np.stack([-z1 * y13 * sln, z1 * x13 * sln, sz2 * sln], axis=1)
    vqn[:, 1, 0:3] = t1_12
    vqn[:, 1, 6:9] = n2
    vqn[:, 1, 3:6] = _t2(n2, t1_12)

    sz2 = a4 - gama2
    sln = 1.0 / np.sqrt(np.maximum(sz13 + sz2 ** 2, EM20))
    n4 = np.stack([z1 * y13 * sln, -z1 * x13 * sln, sz2 * sln], axis=1)
    vqn[:, 3, 0:3] = t1_34
    vqn[:, 3, 6:9] = n4
    vqn[:, 3, 3:6] = _t2(n4, t1_34)

    # ---- edge normals VNRM + rotation arms VASTN ------------------------
    vnrm = np.empty((m, 4, 3))
    for j, (a, b) in enumerate(((0, 1), (3, 2), (0, 3), (1, 2))):
        s = vqn[:, a, 6:9] + vqn[:, b, 6:9]
        vnrm[:, j] = s / np.maximum(
            np.sqrt(np.einsum("nk,nk->n", s, s)), EM20)[:, None]
    vastn = np.zeros((m, 4, 4))
    vastn[:, 0, 1] = l12
    vastn[:, 0, 3] = l12
    vastn[:, 1, 1] = l34
    vastn[:, 1, 3] = l34
    e41 = np.stack([x41, y41, -z1], axis=1)
    vastn[:, 2, 0] = -np.einsum("nk,nk->n", e41, vqn[:, 0, 3:6])
    vastn[:, 2, 1] = np.einsum("nk,nk->n", e41, vqn[:, 0, 0:3])
    vastn[:, 2, 2] = -np.einsum("nk,nk->n", e41, vqn[:, 3, 3:6])
    vastn[:, 2, 3] = np.einsum("nk,nk->n", e41, vqn[:, 3, 0:3])
    e32 = np.stack([x32, y32, z1], axis=1)
    vastn[:, 3, 0] = -np.einsum("nk,nk->n", e32, vqn[:, 1, 3:6])
    vastn[:, 3, 1] = np.einsum("nk,nk->n", e32, vqn[:, 1, 0:3])
    vastn[:, 3, 2] = -np.einsum("nk,nk->n", e32, vqn[:, 2, 3:6])
    vastn[:, 3, 3] = np.einsum("nk,nk->n", e32, vqn[:, 2, 0:3])

    # ---- Gauss frames VQG + VJFI + Jacobians ----------------------------
    a4pg = a4 / _PG
    jmx13 = mx13 * _PG
    jmy13 = my13 * _PG
    jmz13 = z1 * _PG
    j2myz = jmz13 ** 2
    g1x1 = mx23 - jmx13
    g1y1 = my23 - jmy13
    c1g = np.sqrt(g1x1 ** 2 + g1y1 ** 2 + j2myz)
    g2x1 = mx34 - jmx13
    g2y1 = my34 - jmy13
    c2g = np.sqrt(g2x1 ** 2 + g2y1 ** 2 + j2myz)
    g1x3 = mx23 + jmx13
    g1y3 = my23 + jmy13
    j1g = np.sqrt(g1x3 ** 2 + g1y3 ** 2 + j2myz)
    g2x2 = mx34 + jmx13
    g2y2 = my34 + jmy13
    j2g = np.sqrt(g2x2 ** 2 + g2y2 ** 2 + j2myz)

    vqg = np.empty((m, 4, 9))
    vjfi = np.empty((m, 4, 6))
    jac_w = np.empty((m, 4))

    # the four in-plane covariant vectors of the warped mid-surface, as
    # FULL 3-vectors (z = -+jmz13). Expanding cbacoor.F's hand-written
    # blocks (lines 773-925) against these shows the exact structure
    #   VJFI(1..3,NG) = (g2 x n) / (JAC)      [after the SL/PG scale]
    #   VJFI(4..6,NG) = (n x g1) / (JAC)
    #   t1_raw        = g1 * |g2| + (g2 x n) * |g1| ,  t2 = n x t1
    # with (g1, g2) per GP:  NG=1 (g1m, g2m), NG=3 (g1p, g2p),
    # NG=2 (g1m, g2p), NG=4 (g1p, g2m) — the 'm/p' vectors carry
    # z = -jmz13 / +jmz13 respectively.
    g1m = np.stack([g1x1, g1y1, -jmz13], axis=1)
    g1p = np.stack([g1x3, g1y3, jmz13], axis=1)
    g2m = np.stack([g2x1, g2y1, -jmz13], axis=1)
    g2p = np.stack([g2x2, g2y2, jmz13], axis=1)

    def _gp(ng, sz, sz2, nraw, g1v, g2v, ca, cb):
        """One Gauss point: nraw = unnormalized in-plane normal part;
        g1v/g2v the covariant vectors; ca = |g1v|, cb = |g2v|."""
        slr = np.sqrt(sz + sz2 ** 2)
        jac_w[:, ng] = slr * _PG
        sli = 1.0 / np.maximum(slr, EM20)
        nrm = np.empty((m, 3))
        nrm[:, 0] = nraw[:, 0] * sli
        nrm[:, 1] = nraw[:, 1] * sli
        nrm[:, 2] = sz2 * sli
        vqg[:, ng, 6:9] = nrm
        v1 = np.cross(g2v, nrm)                  # VJFI rows 1-3 (raw scale)
        t1 = g1v * cb[:, None] + v1 * ca[:, None]
        sl2 = (sli / _PG)[:, None]
        vjfi[:, ng, 0:3] = v1 * sl2
        vjfi[:, ng, 3:6] = np.cross(nrm, g1v) * sl2
        nt = np.sqrt(np.einsum("nk,nk->n", t1, t1))
        t1 /= np.where(nt == 0.0, 1.0, nt)[:, None]
        vqg[:, ng, 0:3] = t1
        vqg[:, ng, 3:6] = np.cross(nrm, t1)

    nraw1 = np.stack([-z1 * y24, z1 * x24], axis=1)
    nraw2 = np.stack([-z1 * y13, z1 * x13], axis=1)
    _gp(0, sz24, a4pg - gama1, nraw1, g1m, g2m, c1g, c2g)
    _gp(2, sz24, a4pg + gama1, -nraw1, g1p, g2p, j1g, j2g)
    _gp(1, sz13, a4pg + gama2, nraw2, g1m, g2p, c1g, j2g)
    _gp(3, sz13, a4pg - gama2, -nraw2, g1p, g2m, j1g, c2g)

    # warped element AREA := sum of Gauss Jacobians (cbacoor.F line 926)
    g["area"][iw] = jac_w.sum(axis=1)
    g["jac"][iw] = jac_w

    # ---- rigid-mode projection of the velocities (945-1075) -------------
    rr_w = rr[iw]                                # (m, node, comp)
    v13w, v24w, vhiw = v13[iw], v24[iw], vhi[iw]
    ar = np.empty((m, 3))
    ar[:, 0] = (-z1 * vhiw[:, 1] + y13 * v13w[:, 2] + y24 * v24w[:, 2]
                + my13 * vhiw[:, 2] + rr_w[:, :, 0].sum(axis=1))
    ar[:, 1] = (z1 * vhiw[:, 0] - x13 * v13w[:, 2] - x24 * v24w[:, 2]
                - mx13 * vhiw[:, 2] + rr_w[:, :, 1].sum(axis=1))
    ar[:, 2] = (x13 * v13w[:, 1] + x24 * v24w[:, 1] + mx13 * vhiw[:, 1]
                - y13 * v13w[:, 0] - y24 * v24w[:, 0] - my13 * vhiw[:, 0]
                + rr_w[:, :, 2].sum(axis=1))
    xx1 = (cx ** 2).sum(axis=1)
    yy = (cy ** 2).sum(axis=1)
    xy = (cx * cy).sum(axis=1)
    hi_x = cx[:, 0] - cx[:, 1] + cx[:, 2] - cx[:, 3]
    hi_y = cy[:, 0] - cy[:, 1] + cy[:, 2] - cy[:, 3]
    xz1 = hi_x * z1
    yz = hi_y * z1
    zz = 4.0 * z2
    d1 = yy + zz + 4.0
    d2 = xx1 + zz + 4.0
    d3 = xx1 + yy + 4.0
    d4 = -xy
    d5 = -xz1
    d6 = -yz
    abc = d1 * d2 * d3
    xxyz2 = d1 * d6 * d6
    yyxz2 = d2 * d5 * d5
    zzxy2 = d3 * d4 * d4
    deta = abc + 2.0 * d4 * d5 * d6 - xxyz2 - yyxz2 - zzxy2
    deta = np.where(deta < EM20, 1.0, 1.0 / np.where(deta < EM20, 1.0, deta))
    di = np.empty((m, 6))
    di[:, 0] = (abc - xxyz2) * deta / d1
    di[:, 1] = (abc - yyxz2) * deta / d2
    di[:, 2] = (abc - zzxy2) * deta / d3
    di[:, 3] = (d5 * d6 - d4 * d3) * deta
    di[:, 4] = (d6 * d4 - d5 * d2) * deta
    di[:, 5] = (d4 * d5 - d6 * d1) * deta
    alr = np.empty((m, 3))
    alr[:, 0] = di[:, 0] * ar[:, 0] + di[:, 3] * ar[:, 1] + di[:, 4] * ar[:, 2]
    alr[:, 1] = di[:, 3] * ar[:, 0] + di[:, 1] * ar[:, 1] + di[:, 5] * ar[:, 2]
    alr[:, 2] = di[:, 4] * ar[:, 0] + di[:, 5] * ar[:, 1] + di[:, 2] * ar[:, 2]

    v13p = np.empty((m, 3))
    v24p = np.empty((m, 3))
    vhip = np.empty((m, 3))
    v13p[:, 0] = 0.5 * v13w[:, 0] + alr[:, 2] * y13
    v24p[:, 0] = 0.5 * v24w[:, 0] + alr[:, 2] * y24
    vhip[:, 0] = 0.25 * vhiw[:, 0] + (alr[:, 2] * my13 - z1 * alr[:, 1])
    v13p[:, 1] = 0.5 * v13w[:, 1] - alr[:, 2] * x13
    v24p[:, 1] = 0.5 * v24w[:, 1] - alr[:, 2] * x24
    vhip[:, 1] = 0.25 * vhiw[:, 1] - (alr[:, 2] * mx13 - z1 * alr[:, 0])
    v13p[:, 2] = 0.5 * v13w[:, 2] - (y13 * alr[:, 0] - x13 * alr[:, 1])
    v24p[:, 2] = 0.5 * v24w[:, 2] - (y24 * alr[:, 0] - x24 * alr[:, 1])
    vhip[:, 2] = 0.25 * vhiw[:, 2] + (mx13 * alr[:, 1] - my13 * alr[:, 0])

    vxyz_w = np.empty((m, 4, 3))                 # per-NODE local velocities
    vxyz_w[:, 0] = v13p + vhip
    vxyz_w[:, 1] = v24p - vhip
    vxyz_w[:, 2] = -v13p + vhip
    vxyz_w[:, 3] = -v24p - vhip

    rrc = rr_w - alr[:, None, :]
    rxyz_w = np.empty((m, 4, 2))                 # nodal 2-dof rotations
    for j in range(4):
        rxyz_w[:, j, 0] = np.einsum("nk,nk->n", vqn[:, j, 0:3], rrc[:, j])
        rxyz_w[:, j, 1] = np.einsum("nk,nk->n", vqn[:, j, 3:6], rrc[:, j])

    corel = np.empty((m, 3, 4))                  # cbaproj COREL (comp,node)
    corel[:, 0, :] = cx
    corel[:, 1, :] = cy
    corel[:, 2, 0] = z1
    corel[:, 2, 1] = -z1
    corel[:, 2, 2] = z1
    corel[:, 2, 3] = -z1

    g.update(vqn=vqn, vnrm=vnrm, vastn=vastn, vqg=vqg, vjfi=vjfi,
             vxyz_w=vxyz_w, rxyz_w=rxyz_w, di=di, corel=corel)


# ----------------------------------------------------------------------------
# per-Gauss-point operators + velocity strains (cbadef.F)
# ----------------------------------------------------------------------------

def _flat_gp(g, ng):
    """FLAT branch of CBADEF for Gauss point ``ng``: the 8-entry membrane/
    bending row operators BM, the 24-entry transverse-shear operator BC
    and the velocity strains VDEF(1,2,4,5,6,7,8) (cbadef.F lines 79-162).
    Compact over i_f."""
    i = g["i_f"]
    hxg = g["hx"][i, ng]
    hyg = g["hy"][i, ng]
    bm = np.empty((len(i), 8))
    bm[:, 0] = g["y24n"][i] + hxg * g["gama1n"][i]
    bm[:, 1] = -g["y13n"][i] + hxg * g["gama2n"][i]
    bm[:, 2] = 0.25 * hxg
    bm[:, 3] = -bm[:, 2]
    bm[:, 4] = -g["x24n"][i] + hyg * g["gama1n"][i]
    bm[:, 5] = g["x13n"][i] + hyg * g["gama2n"][i]
    bm[:, 6] = 0.25 * hyg
    bm[:, 7] = -bm[:, 6]

    v = g["vxyz_f"][i]                           # (m, slot, comp)
    r = g["rxyz"][i]                             # (m, slot4, xy)
    vdef = np.zeros((len(i), 8))
    vdef[:, 0] = bm[:, 0] * v[:, 0, 0] + bm[:, 1] * v[:, 1, 0] \
        + bm[:, 2] * v[:, 2, 0]
    vdef[:, 1] = bm[:, 4] * v[:, 0, 1] + bm[:, 5] * v[:, 1, 1] \
        + bm[:, 6] * v[:, 2, 1]
    vdef[:, 5] = bm[:, 0] * r[:, 0, 1] + bm[:, 1] * r[:, 1, 1] \
        + bm[:, 2] * r[:, 2, 1]
    vdef[:, 6] = -(bm[:, 4] * r[:, 0, 0] + bm[:, 5] * r[:, 1, 0]
                   + bm[:, 6] * r[:, 2, 0])
    vdef[:, 7] = -(bm[:, 0] * r[:, 0, 0] + bm[:, 1] * r[:, 1, 0]
                   + bm[:, 2] * r[:, 2, 0]) \
        + bm[:, 4] * r[:, 0, 1] + bm[:, 5] * r[:, 1, 1] \
        + bm[:, 6] * r[:, 2, 1]

    # transverse shear BC(24) (cbadef.F 106-161)
    cdet = g["jac"][i, ng]
    ksi, eta = _VPG[ng]
    a_1 = 0.25 / np.maximum(cdet, EM20)
    my34, my23, my13 = g["my34"][i], g["my23"][i], g["my13"][i]
    mx34, mx23, mx13 = g["mx34"][i], g["mx23"][i], g["mx13"][i]
    c11 = (my34 + my13 * ksi) * a_1
    c12 = -(my23 + my13 * eta) * a_1
    c21 = -(mx34 + mx13 * ksi) * a_1
    c22 = (mx23 + mx13 * eta) * a_1
    beta1 = my13 + my23 * eta
    ksi1 = my13 + my34 * ksi
    beta2 = mx13 + mx23 * eta
    ksi2 = mx13 + mx34 * ksi
    bc = np.empty((len(i), 24))
    bc[:, 0] = -c11 - c12
    bc[:, 1] = -c21 - c22
    bc[:, 2] = beta1 * c11 + ksi1 * c12
    bc[:, 3] = beta1 * c21 + ksi1 * c22
    bc[:, 4] = -beta2 * c11 - ksi2 * c12
    bc[:, 5] = -beta2 * c21 - ksi2 * c22
    bc[:, 6] = c11 - c12
    bc[:, 7] = c21 - c22
    bc[:, 8] = beta1 * c11 - ksi1 * c12
    bc[:, 9] = beta1 * c21 - ksi1 * c22
    bc[:, 10] = -beta2 * c11 + ksi2 * c12
    bc[:, 11] = -beta2 * c21 + ksi2 * c22
    beta1 = my23 + my13 * eta
    ksi1 = my34 + my13 * ksi
    beta2 = mx23 + mx13 * eta
    ksi2 = mx34 + mx13 * ksi
    bc[:, 12] = c11 * eta + c12 * ksi
    bc[:, 13] = c21 * eta + c22 * ksi
    bc[:, 14] = -beta1 * c11 - ksi1 * c12
    bc[:, 15] = -beta1 * c21 - ksi1 * c22
    bc[:, 16] = c11 * beta2 + c12 * ksi2
    bc[:, 17] = c21 * beta2 + c22 * ksi2
    # (BC 19-24 of the Fortran are the node-2 copies with sign flips,
    # only used through the closed forms below — not materialized)

    vdef[:, 3] = (bc[:, 0] * v[:, 0, 2] + bc[:, 6] * v[:, 1, 2]
                  + bc[:, 12] * v[:, 2, 2]
                  + bc[:, 2] * r[:, 0, 0] + bc[:, 8] * r[:, 1, 0]
                  + bc[:, 14] * r[:, 3, 0]
                  + bc[:, 4] * r[:, 0, 1] + bc[:, 10] * r[:, 1, 1]
                  + bc[:, 16] * r[:, 3, 1])
    vdef[:, 4] = (bc[:, 1] * v[:, 0, 2] + bc[:, 7] * v[:, 1, 2]
                  + bc[:, 13] * v[:, 2, 2]
                  + bc[:, 3] * r[:, 0, 0] + bc[:, 9] * r[:, 1, 0]
                  + bc[:, 15] * r[:, 3, 0]
                  + bc[:, 5] * r[:, 0, 1] + bc[:, 11] * r[:, 1, 1]
                  + bc[:, 17] * r[:, 3, 1])
    return bm, bc, vdef


def _warp_gp(g, ng):
    """WARPED branch of CBADEF for Gauss point ``ng``: the per-node
    membrane rows BM (m,4,3,2), bending rows BMF (m,4,3,3) and BF
    (m,4,2,3), the shear operator BCQ (m,4,5,2), the 2x2 TC map and the
    velocity strains (cbadef.F lines 176-714). Compact over i_w."""
    iw = g["i_w"]
    m = len(iw)
    vqn = g["vqn"]
    vqg = g["vqg"][:, ng]
    vjfi = g["vjfi"][:, ng]
    v = g["vxyz_w"]
    r = g["rxyz_w"]

    tfn = np.zeros((m, 3, 2))
    for j in range(4):
        tfn[:, :, 0] += _VKSI[j, ng] * vqn[:, j, 6:9]
        tfn[:, :, 1] += _VETA[j, ng] * vqn[:, j, 6:9]
    f1, f2 = vjfi[:, 0:3], vjfi[:, 3:6]
    tbi = np.empty((m, 2, 2))
    tbi[:, 1, 1] = np.einsum("nk,nk->n", f1, tfn[:, :, 0])
    tbi[:, 1, 0] = np.einsum("nk,nk->n", f2, tfn[:, :, 0])
    tbi[:, 0, 1] = np.einsum("nk,nk->n", f1, tfn[:, :, 1])
    tbi[:, 0, 0] = np.einsum("nk,nk->n", f2, tfn[:, :, 1])
    thk = -(tbi[:, 0, 0] + tbi[:, 1, 1])
    tbi[:, 0, 1] = -tbi[:, 0, 1]
    tbi[:, 1, 0] = -tbi[:, 1, 0]

    t1g, t2g = vqg[:, 0:3], vqg[:, 3:6]
    tc = np.empty((m, 2, 2))
    tc[:, 0, 0] = np.einsum("nk,nk->n", f1, t1g)
    tc[:, 1, 0] = np.einsum("nk,nk->n", f2, t1g)
    tc[:, 0, 1] = np.einsum("nk,nk->n", f1, t2g)
    tc[:, 1, 1] = np.einsum("nk,nk->n", f2, t2g)
    tbc = np.einsum("nab,nbc->nac", tbi, tc)

    bmw = np.empty((m, 4, 3, 2))
    bmfw = np.empty((m, 4, 3, 3))
    bfw = np.empty((m, 4, 2, 3))
    vdef = np.zeros((m, 8))
    for j in range(4):
        vt1 = np.einsum("nk,nk->n", t1g, v[:, j])
        vt2 = np.einsum("nk,nk->n", t2g, v[:, j])
        c1 = _VKSI[j, ng] * tc[:, 0, 0] + _VETA[j, ng] * tc[:, 1, 0]
        c2 = _VKSI[j, ng] * tc[:, 0, 1] + _VETA[j, ng] * tc[:, 1, 1]
        bc1 = _VKSI[j, ng] * tbc[:, 0, 0] + _VETA[j, ng] * tbc[:, 1, 0]
        bc2 = _VKSI[j, ng] * tbc[:, 0, 1] + _VETA[j, ng] * tbc[:, 1, 1]
        bmw[:, j, :, 0] = t1g * c1[:, None]
        bmw[:, j, :, 1] = t2g * c2[:, None]
        bmfw[:, j, :, 0] = thk[:, None] * bmw[:, j, :, 0] + t1g * bc1[:, None]
        bmfw[:, j, :, 1] = thk[:, None] * bmw[:, j, :, 1] + t2g * bc2[:, None]
        bmfw[:, j, :, 2] = t1g * bc2[:, None] + t2g * bc1[:, None]
        vdef[:, 0] += c1 * vt1
        vdef[:, 1] += c2 * vt2
        vdef[:, 5] += bc1 * vt1
        vdef[:, 6] += bc2 * vt2
        vdef[:, 7] += bc1 * vt2 + bc2 * vt1
        v12 = np.einsum("nk,nk->n", t1g, vqn[:, j, 0:3])   # V1(2)
        v11 = -np.einsum("nk,nk->n", t1g, vqn[:, j, 3:6])  # V1(1)
        v22 = np.einsum("nk,nk->n", t2g, vqn[:, j, 0:3])   # V2(2)
        v21 = -np.einsum("nk,nk->n", t2g, vqn[:, j, 3:6])  # V2(1)
        rv1 = v11 * r[:, j, 0] + v12 * r[:, j, 1]
        rv2 = v21 * r[:, j, 0] + v22 * r[:, j, 1]
        bfw[:, j, 0, 0] = v11 * c1
        bfw[:, j, 0, 1] = v21 * c2
        bfw[:, j, 0, 2] = v11 * c2 + v21 * c1
        bfw[:, j, 1, 0] = v12 * c1
        bfw[:, j, 1, 1] = v22 * c2
        bfw[:, j, 1, 2] = v12 * c2 + v22 * c1
        vdef[:, 5] += c1 * rv1
        vdef[:, 6] += c2 * rv2
        vdef[:, 7] += c1 * rv2 + c2 * rv1
    # closing 2H*[B0] contribution: uses the CONSTANT assumed membrane
    # shear vdef3 (set by CBADEFSH before the GP loop — cbadef.F line 569)
    vdef[:, 5] += thk * vdef[:, 0]
    vdef[:, 6] += thk * vdef[:, 1]
    vdef[:, 7] += thk * g["vdef3"][iw]

    # ---- assumed transverse shear BCQ (m, node, dof(x,y,z,r1,r2), row) --
    bcq = np.zeros((m, 4, 5, 2))
    v11e = np.array([_VKSI[1, ng], _VKSI[2, ng], _VETA[3, ng], _VETA[2, ng]])
    edges = ((0, 1, 0, 0), (3, 2, 1, 0), (0, 3, 2, 1), (1, 2, 3, 1))
    for (na, nb, je, row) in edges:
        w = v11e[je]
        for c in range(3):
            coef = w * g["vnrm"][:, je, c]
            bcq[:, na, c, row] = -coef
            bcq[:, nb, c, row] = coef
        bcq[:, na, 3, row] = w * g["vastn"][:, je, 0]
        bcq[:, na, 4, row] = w * g["vastn"][:, je, 1]
        bcq[:, nb, 3, row] = w * g["vastn"][:, je, 2]
        bcq[:, nb, 4, row] = w * g["vastn"][:, je, 3]
    bcx = np.zeros(m)
    bcy = np.zeros(m)
    for j in range(4):
        bcx += np.einsum("nk,nk->n", bcq[:, j, 0:3, 0], v[:, j]) \
            + bcq[:, j, 3, 0] * r[:, j, 0] + bcq[:, j, 4, 0] * r[:, j, 1]
        bcy += np.einsum("nk,nk->n", bcq[:, j, 0:3, 1], v[:, j]) \
            + bcq[:, j, 3, 1] * r[:, j, 0] + bcq[:, j, 4, 1] * r[:, j, 1]
    vdef[:, 3] = tc[:, 0, 0] * bcx + tc[:, 1, 0] * bcy
    vdef[:, 4] = tc[:, 0, 1] * bcx + tc[:, 1, 1] * bcy
    return bmw, bmfw, bfw, bcq, tc, vdef


# ----------------------------------------------------------------------------
# QBAT Pinching (Through-Thickness Stretch) Routines
# ----------------------------------------------------------------------------

def _compute_vqn(g):
    """Nodal frames VQN = [t1 (3) | t2 (3) | n (3)] for all elements.

    # Ported from engine/source/elements/shell/coqueba/cbacoor.F lines 614-650
    """
    n = g["n"]
    vqn = np.zeros((n, 4, 9))
    if len(g.get("i_w", [])):
        vqn[g["i_w"]] = g["vqn"]
    if len(g.get("i_f", [])):
        i = g["i_f"]
        mx13, my13 = g["mx13"][i], g["my13"][i]
        mx23, my23 = g["mx23"][i], g["my23"][i]
        cx, cy = g["cx"][i], g["cy"][i]
        x21 = mx23 - mx13
        y21 = my23 - my13
        l12 = np.sqrt(x21 ** 2 + y21 ** 2)
        sl12 = 1.0 / np.maximum(l12, EM20)
        t1_12 = np.stack([x21 * sl12, y21 * sl12, np.zeros_like(x21)], axis=1)
        t2_12 = np.stack([-y21 * sl12, x21 * sl12, np.zeros_like(x21)], axis=1)

        x34 = 0.5 * (cx[:, 2] - cx[:, 3])
        y34 = 0.5 * (cy[:, 2] - cy[:, 3])
        l34 = np.sqrt(x34 ** 2 + y34 ** 2)
        sl34 = 1.0 / np.maximum(l34, EM20)
        t1_34 = np.stack([x34 * sl34, y34 * sl34, np.zeros_like(x34)], axis=1)
        t2_34 = np.stack([-y34 * sl34, x34 * sl34, np.zeros_like(x34)], axis=1)

        nrm = np.zeros((len(i), 3))
        nrm[:, 2] = 1.0

        for j in (0, 1):
            vqn[i, j, 0:3] = t1_12
            vqn[i, j, 3:6] = t2_12
            vqn[i, j, 6:9] = nrm
        for j in (2, 3):
            vqn[i, j, 0:3] = t1_34
            vqn[i, j, 3:6] = t2_34
            vqn[i, j, 6:9] = nrm
    return vqn


def _cbacoorpinch(E, vqn, vpe, thick, dt, lc):
    """Transform global pinching velocities into local frame and scale by thickness.

    # Ported from engine/source/elements/shell/coqueba/cbacoorpinch.F lines 93-200
    E     : (n, 3, 3) element corotational triad (columns e1, e2, e3)
    vqn   : (n, 4, 9) nodal frames [t1, t2, n]
    vpe   : (n, 4, 3) nodal pinching velocities in global coordinates
    thick : (n,) shell thickness
    dt    : float, time step
    lc    : (n,) characteristic length
    Returns: (vp_xyz, vp_t1, vp_t2, facp, ezzavg, avgthk)
    """
    # BETABETA = VQ^T @ VPINCH (lines 102-125): E has columns e1, e2, e3
    betabeta = np.einsum("nka,njk->nja", E, vpe)  # (n, 4, 3)

    # Project on t1, t2, normal (lines 129-168)
    vp_t1 = np.einsum("nja,nja->nj", vqn[:, :, 0:3], betabeta)   # (n, 4)
    vp_t2 = np.einsum("nja,nja->nj", vqn[:, :, 3:6], betabeta)   # (n, 4)
    vp_xyz = np.einsum("nja,nja->nj", vqn[:, :, 6:9], betabeta)  # (n, 4)

    # Average thickness (lines 170-176)
    thkn = thick[:, None] * (1.0 + 2.0 * vp_xyz * dt)
    avgthk = thkn.mean(axis=1)
    elthkinv = 2.0 / np.maximum(avgthk, EM20)

    # Divide by thickness (lines 179-194)
    vp_xyz = vp_xyz * elthkinv[:, None]
    vp_t1 = vp_t1 * elthkinv[:, None]
    vp_t2 = vp_t2 * elthkinv[:, None]

    # Stiffness scaling factor for dynamic condensation (line 198)
    facp = (lc / np.maximum(avgthk, EM20)) ** 2
    ezzavg = 0.25 * vp_xyz.sum(axis=1) * dt
    return vp_xyz, vp_t1, vp_t2, facp, ezzavg, avgthk


def _compute_tc_flat(g, ng):
    """Compute 2x2 inverse Jacobian TC for flat elements at GP ng.

    # Ported from engine/source/elements/shell/coqueba/cbadef.F lines 212-219
    """
    i = g["i_f"]
    cx, cy = g["cx"][i], g["cy"][i]
    m = len(i)
    j00 = np.sum(_VKSI[:, ng][None, :] * cx, axis=1)
    j01 = np.sum(_VKSI[:, ng][None, :] * cy, axis=1)
    j10 = np.sum(_VETA[:, ng][None, :] * cx, axis=1)
    j11 = np.sum(_VETA[:, ng][None, :] * cy, axis=1)
    detj = j00 * j11 - j01 * j10
    detj_safe = np.where(np.abs(detj) < EM20, 1.0, detj)
    tc = np.empty((m, 2, 2))
    tc[:, 0, 0] = j11 / detj_safe
    tc[:, 0, 1] = -j01 / detj_safe
    tc[:, 1, 0] = -j10 / detj_safe
    tc[:, 1, 1] = j00 / detj_safe
    return tc


def _cbadefpinch(tc, vqg_11, vqg_12, vp_xyz, vp_t1, vp_t2, ng):
    """Compute pinching transverse-shear, thickness stretch, and gradient strain rates.

    # Ported from engine/source/elements/shell/coqueba/cbadefpinch.F lines 65-181
    tc     : (n, 2, 2) inverse in-plane Jacobian at Gauss point ng
    vqg_11 : (n,) x-component of Gauss point triad vector 1
    vqg_12 : (n,) x-component of Gauss point triad vector 2
    vp_xyz : (n, 4) normal pinching velocity rates
    vp_t1  : (n, 4) tangent 1 pinching velocity rates
    vp_t2  : (n, 4) tangent 2 pinching velocity rates
    ng     : int, Gauss point index 0..3
    Returns:
        vdefpinch : (n, 3) [rate_xz, rate_yz, rate_zz]
        bcp       : (n, 4, 2) matrix [BCP] for transverse shear due to pinching
        bp        : (4,) shape functions at Gauss point ng
        dbetadxy  : (n, 3) in-plane derivatives of beta
    """
    n = len(tc)
    bcp = np.empty((n, 4, 2))
    for j in range(4):
        d1 = _VKSI[j, ng] * tc[:, 0, 0] + _VETA[j, ng] * tc[:, 1, 0]
        d2 = _VKSI[j, ng] * tc[:, 0, 1] + _VETA[j, ng] * tc[:, 1, 1]
        bcp[:, j, 0] = vqg_11 * d1
        bcp[:, j, 1] = vqg_12 * d2

    # Transverse shear rate due to pinching (lines 108-128)
    bcx = np.sum(bcp[:, :, 0] * vp_xyz, axis=1)
    bcy = np.sum(bcp[:, :, 1] * vp_xyz, axis=1)

    vdefpinch = np.empty((n, 3))
    vdefpinch[:, 0] = tc[:, 0, 0] * bcx + tc[:, 1, 0] * bcx
    vdefpinch[:, 1] = tc[:, 0, 1] * bcx + tc[:, 1, 1] * bcy

    # Through-thickness stretch rate (lines 136-144)
    bp = _TNPG[:, ng]  # (4,)
    vdefpinch[:, 2] = np.sum(bp[None, :] * vp_xyz, axis=1)

    # In-plane derivatives of beta (lines 150-170)
    dbetadxy = np.empty((n, 3))
    dn_dx = _VKSI[:, ng][None, :] * tc[:, 0:1, 0] + _VETA[:, ng][None, :] * tc[:, 1:2, 0]  # (n, 4)
    dn_dy = _VKSI[:, ng][None, :] * tc[:, 0:1, 1] + _VETA[:, ng][None, :] * tc[:, 1:2, 1]  # (n, 4)

    dbetadxy[:, 0] = np.sum(dn_dx * vp_t1, axis=1)
    dbetadxy[:, 1] = np.sum(dn_dy * vp_t2, axis=1)
    dbetadxy[:, 2] = 0.5 * (np.sum(dn_dy * vp_t1, axis=1) + np.sum(dn_dx * vp_t2, axis=1))

    return vdefpinch, bcp, bp, dbetadxy


def _cbastra3pinch(vdefpinch, dt):
    """Compute strain increments due to pinching.

    # Ported from engine/source/elements/shell/coqueba/cbastra3pinch.F lines 65-74
    vdefpinch : (n, 3) [rate_xz, rate_yz, rate_zz]
    dt        : float, time step
    Returns: (epinch_xz, epinch_yz, ezz)
    """
    epinch_xz = vdefpinch[:, 0] * dt
    epinch_yz = vdefpinch[:, 1] * dt
    ezz = vdefpinch[:, 2] * dt
    return epinch_xz, epinch_yz, ezz


def _cbaforipinch(cdet, thick, bcp, bp, sig_zz, mom_p):
    """Compute internal pinching forces at the 4 nodes.

    # Ported from engine/source/elements/shell/coqueba/cbaforipinch.F lines 60-72
    cdet   : (n,) Gauss Jacobian determinant
    thick  : (n,) shell thickness
    bcp    : (n, 4, 2) pinching transverse shear operator
    bp     : (4,) shape functions at Gauss point
    sig_zz : (n,) through-thickness normal stress resultant (FF)
    mom_p  : (n, 2) pinching shear moments [M_xz^p / t^2, M_yz^p / t^2] (MM)
    Returns:
        vfpinch : (n, 4) internal pinching generalized force at 4 nodes
    """
    c1 = (thick ** 2 / 12.0) * cdet
    c2 = thick * cdet
    vfpinch = np.empty((len(cdet), 4))
    for j in range(4):
        vfpinch[:, j] = (
            c1 * (bcp[:, j, 0] * mom_p[:, 0] + bcp[:, j, 1] * mom_p[:, 1])
            + c2 * bp[j] * sig_zz
        )
    return vfpinch


def _cbapinchproj(E, vqn, vfpinch, thick0):
    """Project local nodal pinching force along normal into global 3D forces.

    # Ported from engine/source/elements/shell/coqueba/cbapinchproj.F lines 59-133
    E       : (n, 3, 3) element corotational triad
    vqn     : (n, 4, 9) nodal frames (normals at 6:9)
    vfpinch : (n, 4) local pinching force
    thick0  : (n,) initial thickness
    Returns:
        fp : (n, 4, 3) global nodal pinching force vectors
    """
    elthkinv = 1.0 / np.maximum(thick0, EM20)
    # FPP = (vfpinch / thick0) * normal
    fpp = (vfpinch * elthkinv[:, None])[:, :, None] * vqn[:, :, 6:9]  # (n, 4, 3) in local frame
    # Global projection: FP = VQ @ FPP = E @ FPP
    fp = np.einsum("nka,nja->njk", E, fpp)
    return fp


def _cbapinchthk(thick, ezzpg):
    """Update shell thickness based on through-thickness normal strain.

    # Ported from engine/source/elements/shell/coqueba/cbapinchthk.F lines 60-69
    thick : (n,) shell thickness
    ezzpg : (n, 4) normal strain at the 4 Gauss points
    Returns:
        thick_new : (n,) updated thickness
    """
    thkn = thick[:, None] * (1.0 + ezzpg)  # (n, 4)
    thkavg = thkn.mean(axis=1)             # (n,)
    return thkavg


def _cndt3pinch(lc, amu, rho, E_mod, nu):
    """Critical time step using 3D solid P-wave sound speed for pinched shells.

    # Ported from engine/source/elements/shell/coqueba/cndt3pinch.F & sigeps01gpinch.F line 133
    lc    : (n,) characteristic length
    amu   : (n,) numerical damping
    rho   : (n,) density
    E_mod : (n,) Young's modulus
    nu    : (n,) Poisson's ratio
    Returns:
        dt_e : (n,) critical time step
        ssp  : (n,) 3D solid P-wave speed
    """
    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    denom = np.where(np.abs(denom) < EM20, 1.0, denom)
    pa1 = E_mod * (1.0 - nu) / denom
    ssp = np.sqrt(np.maximum(pa1 / np.maximum(rho, EM20), EM20))
    viscdt = np.sqrt(1.0 + amu ** 2) - amu
    dt_e = lc * viscdt / ssp
    return dt_e, ssp


# ----------------------------------------------------------------------------
# Engine-side forces (cbaforc3.F)
# ----------------------------------------------------------------------------

def _element_deletion_gpmajor(st):
    """Element OFF from the layer flags — shell_bt4._element_deletion's
    RULE (/FAIL Ifail_sh 1 = one broken layer kills the element, the
    Radioss default also used for eps_p_max; Ifail_sh 2 / LAW27 = ALL
    layers) re-derived for THIS kernel's GP-major layer axis
    k = GP*nip_max + layer: a slice whose nip is below the group's
    nip_max leaves its PAD slots permanently intact, so the BT-style
    'first nip columns' count would make Ifail_sh=2 / LAW27 elements of
    a mixed-nip group immortal. Counts the REAL slots only (4 Gauss
    points x the slice's own layer count)."""
    off = st["off"]
    layfail = st["layfail"]
    nip_max = st["nip_max"]
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if not (mat.fail is not None or mat.law == 27
                or mat.params.get("eps_p_max", EP30) < 1e30):
            continue
        nip = len(st["zw"][isl][0])
        cols = np.array([ng * nip_max + il
                         for ng in range(4) for il in range(nip)])
        nbroken = (layfail[sl][:, cols] == 0.0).sum(axis=1)
        if mat.law == 27 or (mat.fail is not None
                             and mat.fail.ifail_sh == 2):
            dead = nbroken == len(cols)
        else:
            dead = nbroken >= 1
        off[sl][dead] = 0.0
    return off > 0.0


def forces(group, x, v, vr, dt, fint, mint, vpinch=None, fpinch=None):
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)

    thick = st["thick"]
    off = st.get("off", np.ones(n))
    alive = off > 0.0
    nip_max = st["nip_max"]

    # Cycle 0 or velocity-free forces evaluation:
    if dt is None or dt <= 0.0 or v is None or vr is None:
        xe = x[conn]
        E, det = _frame(xe)
        area = 0.25 * det
        area_i = 1.0 / np.maximum(area, EM20)
        d = xe - xe[:, 0:1, :]
        xl = np.einsum("njk,nka->nja", d, E)
        cx = xl[:, :, 0] - xl[:, :, 0].mean(axis=1)[:, None]
        cy = xl[:, :, 1] - xl[:, :, 1].mean(axis=1)[:, None]
        x13 = 0.5 * (cx[:, 0] - cx[:, 2])
        x24 = 0.5 * (cx[:, 1] - cx[:, 3])
        y13 = 0.5 * (cy[:, 0] - cy[:, 2])
        y24 = 0.5 * (cy[:, 1] - cy[:, 3])
        l13 = x13 ** 2 + y13 ** 2
        l24 = x24 ** 2 + y24 ** 2
        ll = np.maximum(l13, l24)
        lm = np.maximum(np.abs(cx[:, 1] * cy[:, 3] - cy[:, 1] * cx[:, 3]),
                        np.abs(cx[:, 0] * cy[:, 2] - cy[:, 0] * cx[:, 2]))
        rx = cx[:, 1] + cx[:, 2] - cx[:, 3] - cx[:, 0]
        ry = cy[:, 1] + cy[:, 2] - cy[:, 3] - cy[:, 0]
        sx = -cx[:, 1] + cx[:, 2] + cx[:, 3] - cx[:, 0]
        sy = -cy[:, 1] + cy[:, 2] + cy[:, 3] - cy[:, 0]
        c1 = np.sqrt(rx ** 2 + ry ** 2)
        c2 = np.sqrt(sx ** 2 + sy ** 2)
        cmax = np.maximum(c1, c2)
        cmin = np.maximum(np.minimum(c1, c2), EM20)
        fac1 = np.minimum(0.5, 0.25 * (cmax / cmin - 1.0)) + 1.0
        fac2 = 4.0 * area / np.maximum(c1 * c2, EM20)
        fac2 = 3.413 * np.maximum(0.0, fac2 - 0.7071)
        fac2 = 0.78 + 0.22 * fac2 ** 3
        faci = 2.0 * fac1 * fac2
        s1 = np.maximum(np.sqrt(faci * (_FACDT + lm * area_i) * ll), EM20)
        lc = area / s1
        viscdt = np.sqrt(1.0 + st["amu"] ** 2) - st["amu"]
        dt_e = lc * viscdt / np.maximum(st["ssp0"], EM20)
        is_void = np.zeros(n, dtype=bool)
        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "law", 1) == 0:
                is_void[sl] = True
        return np.where(alive & (~is_void), dt_e, EP30)

    # a group whose every slice runs one integration point is forced FLAT
    # (cbacoor.F line 444 'OR NPT==1')
    force_flat = np.zeros(n, dtype=bool)
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if len(st["zw"][isl][0]) == 1:
            force_flat[sl] = True

    has_pinch = bool(st.get("ipinch", False) or (vpinch is not None))

    from pyradioss.accel import get as accel_get
    jit_pre = accel_get("qbat_pre") if not has_pinch else None
    jit_post = accel_get("qbat_post") if not has_pinch else None
    if jit_pre is None or jit_post is None:
        jit_pre = jit_post = None
    if jit_pre is not None:
        E, area, lc, vdef3, cdet, vdef, i_f, i_w, bm_f, bc_f, bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w, x13n_f, x24n_f, y13n_f, y24n_f, x13n_w, x24n_w, y13n_w, y24n_w = jit_pre(x[conn], v[conn], vr[conn], off, dt, force_flat)
        g = {"lc": lc}
    else:
        g = _cbacoor(x[conn], v[conn], vr[conn], off, dt, force_flat)
        i_f, i_w = g["i_f"], g["i_w"]
        area = g["area"]

        # ---- CBADEFSH: constant assumed membrane shear ------------------------
        vdef3 = np.zeros(n)
        if len(i_f):
            vf_ = g["vxyz_f"][i_f]
            vdef3[i_f] = (g["y24n"][i_f] * vf_[:, 0, 1]
                          - g["y13n"][i_f] * vf_[:, 1, 1]
                          - g["x24n"][i_f] * vf_[:, 0, 0]
                          + g["x13n"][i_f] * vf_[:, 1, 0])
        if len(i_w):
            vw_ = g["vxyz_w"]
            vdef3[i_w] = (g["y24n"][i_w] * (vw_[:, 0, 1] - vw_[:, 2, 1])
                          + g["y13n"][i_w] * (-vw_[:, 1, 1] + vw_[:, 3, 1])
                          - g["x24n"][i_w] * (vw_[:, 0, 0] - vw_[:, 2, 0])
                          + g["x13n"][i_w] * (vw_[:, 1, 0] - vw_[:, 3, 0]))
        vdef3[~alive] = 0.0
        g["vdef3"] = vdef3
    volg = area * thick

    if has_pinch:
        if vpinch is not None:
            if vpinch.ndim == 2:
                vpe = vpinch[conn]
            else:
                vpe = vpinch
        elif "vpinch" in st:
            vpe = st["vpinch"]
            if vpe.ndim == 2:
                vpe = vpe[conn]
        else:
            vpe = np.zeros((n, 4, 3))

        vqn = _compute_vqn(g)
        vp_xyz, vp_t1, vp_t2, facp, ezzavg, avgthk = _cbacoorpinch(
            g["E"], vqn, vpe, thick, dt, g["lc"]
        )
        vfpinch = np.zeros((n, 4))
        ezzpg = np.zeros((n, 4))

    # CBAENERS (pre): + FOR3_mean_old * vdef3 * A*t*dt/2  (cbaforc3 l.566)
    st["eint"] += off * volg * dt * 0.5 * st["for_mean"][:, 2] * vdef3

    # per-slice constants
    gs_mod = np.zeros(n)                          # GS = G*SHF (0 if nip==1)
    bend_visc = np.ones(n)                        # cbavisc.F 'NPT /= 1' gate
    nip_e = np.zeros(n, dtype=np.int64)           # through-thickness points
    gshear = np.zeros(n)                          # G for CBAVISNP1 (NPT==1)
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if getattr(mat, "law", 1) == 0:
            continue
        nip = len(st["zw"][isl][0])
        nip_e[sl] = nip
        gshear[sl] = (getattr(mat, "G", 0.0) or getattr(mat, "g5", 0.0)
                      or getattr(mat, "g0", 0.0))
        gs_mod[sl] = 0.0 if nip == 1 else SHEAR_FACTOR * gshear[sl]
        if nip == 1:
            bend_visc[sl] = 0.0

    # single-point rows: NPT==1 elements are forced FLAT (cbacoor.F line 444
    # 'OR NPT==1'), so nip1 is a subset of i_f on the numpy path.
    nip1 = nip_e == 1
    has_nip1 = bool(np.any(nip1))

    sig = st["sig"]
    qsh = st["qshear"]
    forpg = st["forpg"]
    mompg = st["mompg"]
    epsp_old = st["epsp"].copy() if st["chk_fail"] else None

    vf = np.zeros((n, 3, 4))                     # local generalized forces
    vm = np.zeros((n, 2, 4))
    de = np.zeros(n)                             # eint increment
    dehg = np.zeros(n)                           # ehour increment (cbavisc)

    ops_f = []
    ops_w = []
    for ng in range(4):
        if jit_pre is not None:
            cdet_ = cdet[:, ng]
            vdef_ = vdef[:, ng, :]
        else:
            cdet_ = g["jac"][:, ng]
            vdef_ = np.zeros((n, 8))
            if len(i_f):
                bm, bc, vd = _flat_gp(g, ng)
                vd[~alive[i_f]] = 0.0
                vdef_[i_f] = vd
                ops_f.append((bm, bc))
            else:
                ops_f.append(None)
            if len(i_w):
                bmw, bmfw, bfw, bcq, tc, vd = _warp_gp(g, ng)
                vd[~alive[i_w]] = 0.0
                vdef_[i_w] = vd
                ops_w.append((bmw, bmfw, bfw, bcq, tc))
            else:
                ops_w.append(None)
            vdef_[:, 2] = vdef3

        if has_nip1:
            # CBADEF1 (cbadef.F, the NPT==1 leg): membrane-only velocity
            # strains — VDEF(4:8) (transverse shear + bending) are zero and
            # VDEF(3) keeps the CBADEFSH constant assumed shear from vdef3.
            vdef_[nip1, 3:8] = 0.0

        # strains (cbastra3.F): EXZ=VDEF4, EYZ=VDEF5
        exx = vdef_[:, 0] * dt
        eyy = vdef_[:, 1] * dt
        exy = vdef_[:, 2] * dt
        exz = vdef_[:, 3] * dt
        eyz = vdef_[:, 4] * dt
        kxx = vdef_[:, 5] * dt
        kyy = vdef_[:, 6] * dt
        kxy = vdef_[:, 7] * dt

        # CBAENER (pre): remove the per-GP old-stress shear work
        de -= 0.5 * off * thick * cdet_ * forpg[:, ng, 2] * exy

        # ---- layer stress updates + resultants ---------------------------
        npg_ = np.zeros((n, 3))                  # membrane N (force/length)
        mpg_ = np.zeros((n, 3))                  # moment M (moment/length)
        for isl, (sl, mat, prop) in enumerate(st["slices"]):
            zrel, wrel = st["zw"][isl]
            t_sl = thick[sl]
            dm = np.stack([exx[sl], eyy[sl], exy[sl]], axis=1)
            kap = np.stack([kxx[sl], kyy[sl], kxy[sl]], axis=1)
            for il in range(len(zrel)):
                k = ng * nip_max + il
                zk = zrel[il] * t_sl
                wk = wrel[il] * t_sl
                deps = dm + zk[:, None] * kap
                s_old = sig[sl, k, :].copy()
                s_new, _ = materials.shell_update(
                    mat, sig[sl, k, :], deps, st["epsp"][sl, k], dt,
                    _layer_extra(st, sl, k))
                if st["chk_fail"]:
                    _layer_failure(st, sl, mat, k, s_new, epsp_old,
                                   deps, dt)
                sig[sl, k, :] = s_new
                s_mid = 0.5 * (s_old + s_new)
                de[sl] += cdet_[sl] * wk * np.einsum("nk,nk->n", s_mid, deps)
                npg_[sl] += wk[:, None] * s_new
                mpg_[sl] += (wk * zk)[:, None] * s_new
            # elastic transverse shear (per GP)
            qold = qsh[sl, ng].copy()
            dq = np.stack([exz[sl], eyz[sl]], axis=1)
            qsh[sl, ng] += gs_mod[sl][:, None] * dq
            de[sl] += cdet_[sl] * t_sl * np.einsum(
                "nk,nk->n", 0.5 * (qold + qsh[sl, ng]), dq)

        if has_pinch:
            tc_all = np.empty((n, 2, 2))
            vqg_11 = np.ones(n)
            vqg_12 = np.ones(n)
            if len(i_f):
                tc_all[i_f] = _compute_tc_flat(g, ng)
            if len(i_w):
                tc_all[i_w] = ops_w[ng][4]
                vqg_11[i_w] = g["vqg"][:, ng, 0]
                vqg_12[i_w] = g["vqg"][:, ng, 3]

            vdefp, bcp, bp, dbetadxy = _cbadefpinch(
                tc_all, vqg_11, vqg_12, vp_xyz, vp_t1, vp_t2, ng
            )
            epinch_xz, epinch_yz, ezz = _cbastra3pinch(vdefp, dt)
            ezzpg[:, ng] = ezz
            st["epg_pinch_xz"][:, ng] = epinch_xz
            st["epg_pinch_yz"][:, ng] = epinch_yz
            st["epg_pinch_zz"][:, ng] = ezz

            # 3D Hooke's law coupling for through-thickness stress (sigeps01gpinch.F)
            for sl, mat, prop in st["slices"]:
                if getattr(mat, "law", 1) == 0:
                    continue
                E_mod = getattr(mat, "E", getattr(mat, "young", 0.0))
                nu_val = st["nu0"][sl]
                denom = (1.0 + nu_val) * (1.0 - 2.0 * nu_val)
                denom = np.where(np.abs(denom) < EM20, 1.0, denom)
                pa1 = E_mod * (1.0 - nu_val) / denom
                pa2 = E_mod * nu_val / denom
                pa3 = E_mod / (2.0 * (1.0 + nu_val))
                b3 = pa3 * thick[sl] / 12.0

                sig_zz_old = st["forpg_pinch"][sl, ng].copy()
                d_sig_zz = pa2 * (exx[sl] + eyy[sl]) + pa1 * ezz[sl]
                sig_zz_new = sig_zz_old + d_sig_zz
                st["forpg_pinch"][sl, ng] = sig_zz_new

                mom_p_old = st["mompg_pinch"][sl, ng].copy()
                mom_p_new = np.empty_like(mom_p_old)
                mom_p_new[:, 0] = mom_p_old[:, 0] + b3 * epinch_xz[sl]
                mom_p_new[:, 1] = mom_p_old[:, 1] + b3 * epinch_yz[sl]
                st["mompg_pinch"][sl, ng] = mom_p_new

                npg_[sl, 0] += thick[sl] * pa2 * ezz[sl]
                npg_[sl, 1] += thick[sl] * pa2 * ezz[sl]

                sig_zz_mid = 0.5 * (sig_zz_old + sig_zz_new)
                de[sl] += cdet_[sl] * thick[sl] * sig_zz_mid * ezz[sl]
                mom_p_mid = 0.5 * (mom_p_old + mom_p_new)
                de[sl] += cdet_[sl] * (thick[sl] ** 2 / 12.0) * (
                    mom_p_mid[:, 0] * epinch_xz[sl] + mom_p_mid[:, 1] * epinch_yz[sl]
                )

            vfpinch += _cbaforipinch(
                cdet_, thick, bcp, bp, st["forpg_pinch"][:, ng], st["mompg_pinch"][:, ng]
            )

        # CBAENER (post): remove the per-GP NEW-stress (pre-viscous) work
        de -= 0.5 * off * thick * cdet_ * (npg_[:, 2] / np.maximum(
            thick, EM20)) * exy

        # ---- CBAVISC: dn numerical damping ------------------------------
        visc = _ONEP414 * off * st["amu"] * st["rho0"] * st["ssp0"] \
            * np.sqrt(np.maximum(cdet_, 0.0))
        nu = st["nu0"]
        gg = 0.5 / (1.0 + nu)
        fx = visc * (vdef_[:, 0] + nu * vdef_[:, 1])
        fy = visc * (vdef_[:, 1] + nu * vdef_[:, 0])
        fxy = visc * vdef_[:, 2] * gg
        npg_v = npg_.copy()
        npg_v[:, 0] += fx * thick
        npg_v[:, 1] += fy * thick
        npg_v[:, 2] += fxy * thick
        dv = cdet_ * thick * dt
        dehg += (fx * vdef_[:, 0] + fy * vdef_[:, 1]) * dv
        viscb = _ZEP3 * thick * visc * bend_visc
        mvx = viscb * (vdef_[:, 5] + nu * vdef_[:, 6])
        mvy = viscb * (vdef_[:, 6] + nu * vdef_[:, 5])
        mvxy = viscb * vdef_[:, 7] * gg
        mpg_v = mpg_.copy()
        t2 = thick ** 2
        mpg_v[:, 0] += mvx * t2
        mpg_v[:, 1] += mvy * t2
        mpg_v[:, 2] += mvxy * t2
        dehg += (mvx * vdef_[:, 5] + mvy * vdef_[:, 6]
                 + mvxy * vdef_[:, 7]) * dv * thick

        # persist the GBUF%FORPG / MOMPG state (stress / M/t^2 units)
        t_i = 1.0 / np.maximum(thick, EM20)
        forpg[:, ng, 0:3] = npg_v * t_i[:, None]
        forpg[:, ng, 3] = qsh[:, ng, 1]          # FOR(4) = sig_yz
        forpg[:, ng, 4] = qsh[:, ng, 0]          # FOR(5) = sig_zx
        mompg[:, ng] = mpg_v * (t_i ** 2)[:, None]

        # ---- CBAFORI / CBAFORI1: internal force assembly ------------------
        q_pg = qsh[:, ng] * thick[:, None]       # physical [q_xz, q_yz]
        if jit_pre is None:
            if len(i_f):
                if has_nip1:
                    # NPT==1 rows take the distinct CBAFORI1 membrane-only
                    # assembly; the rest take the standard CBAFORI leg.
                    m1 = nip1[i_f]
                    if np.any(m1):
                        _fori_np1(vf, g, ops_f[ng][0], cdet_, npg_v, m1)
                    m2 = ~m1
                    if np.any(m2):
                        _fori_flat(vf, vm, dict(g, i_f=i_f[m2]),
                                   ops_f[ng][0][m2], ops_f[ng][1][m2],
                                   cdet_, npg_v, mpg_v, q_pg)
                else:
                    _fori_flat(vf, vm, g, ops_f[ng][0], ops_f[ng][1],
                               cdet_, npg_v, mpg_v, q_pg)
            if len(i_w):
                _fori_warp(vf, vm, g, ops_w[ng], cdet_, npg_v, mpg_v, q_pg)

    # ---- after the Gauss loop --------------------------------------------
    for_mean = forpg.mean(axis=1)                # GBUF%FOR (cbaforc3 962)
    st["for_mean"] = for_mean

    # CBAFORCT: constant membrane shear force from the MEAN resultant
    thoff = volg * for_mean[:, 2] * off
    if jit_pre is None:
        if len(i_f):
            th_f = thoff[i_f]
            vf[i_f, 0, 0] += -th_f * g["x24n"][i_f]
            vf[i_f, 1, 0] += th_f * g["y24n"][i_f]
            vf[i_f, 0, 1] += th_f * g["x13n"][i_f]
            vf[i_f, 1, 1] += -th_f * g["y13n"][i_f]
        if len(i_w):
            th_w = thoff[i_w]
            sx1 = -th_w * g["x24n"][i_w]
            sy1 = th_w * g["y24n"][i_w]
            sx2 = th_w * g["x13n"][i_w]
            sy2 = -th_w * g["y13n"][i_w]
            vf[i_w, 0, 0] += sx1
            vf[i_w, 1, 0] += sy1
            vf[i_w, 0, 1] += sx2
            vf[i_w, 1, 1] += sy2
            vf[i_w, 0, 2] -= sx1
            vf[i_w, 1, 2] -= sy1
            vf[i_w, 0, 3] -= sx2
            vf[i_w, 1, 3] -= sy2

    # CBAENERS (post): + FOR3_mean_new * vdef3 * A*t*dt/2
    de += off * volg * dt * 0.5 * for_mean[:, 2] * vdef3

    # CBAVISNP1 (cbaforc3 line ~999): rotation-damping z-forces + TESY work,
    # NPT==1 only. Runs after CBAFORCT/CBAENERS like upstream, before CBAPROJ.
    if has_nip1 and jit_pre is None:
        _cbavisnp1(vf, g, nip1, gshear, st["rho0"], st["amu"], off,
                   thick, area, dt, dehg)

    # ---- element deletion from the layer flags ---------------------------
    if st["chk_fail"]:
        alive = _element_deletion_gpmajor(st)
        off = st["off"]
        if not alive.all():
            dead = ~alive
            sig[dead] = 0.0
            qsh[dead] = 0.0
            forpg[dead] = 0.0
            mompg[dead] = 0.0
            st["for_mean"][dead] = 0.0

    st["eint"] += de
    st["ehour"] += dehg

    # ---- CBAPROJ: local -> global, rigid projection, OFF -----------------
    jit_post = accel_get("qbat_post")
    if jit_post is not None:
        fg, mg = jit_post(n, E, off, thick, volg, forpg, mompg, for_mean, cdet, i_f, i_w, bm_f, bc_f, bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w, x13n_f, x24n_f, y13n_f, y24n_f, x13n_w, x24n_w, y13n_w, y24n_w)
    else:
        fg, mg = _cbaproj(g, vf, vm, off)

    # accumulate NEGATED (cupdtn3.F: F -= F11)
    flat_idx = conn.reshape(-1)
    if fint is not None:
        scatter_add3(fint, flat_idx, -fg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
    if mint is not None:
        scatter_add3(mint, flat_idx, -mg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))

    if has_pinch:
        st["thick"] = _cbapinchthk(thick, ezzpg)
        thk0_ref = st.get("thk0", thick)
        fp = _cbapinchproj(g["E"], vqn, vfpinch, thk0_ref)
        st["fpinch"] = -fp
        if fpinch is not None:
            scatter_add3(fpinch, flat_idx, -fp.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))

    # ---- dt claim (cndt3.F / cndt3pinch.F): condensed LC * (sqrt(1+dn^2)-dn) / ssp ------
    viscdt = np.sqrt(1.0 + st["amu"] ** 2) - st["amu"]
    dt_e = g["lc"] * viscdt / np.maximum(st["ssp0"], EM20)
    if has_pinch:
        dt_p, _ = _cndt3pinch(g["lc"], st["amu"], st["rho0"], st.get("E_mod", np.zeros(n)), st["nu0"])
        dt_e = np.minimum(dt_e, dt_p)
    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    return np.where(alive & (~is_void) & (st["ssp0"] > 0.0), dt_e, EP30)


def _fori_warp(vf, vm, g, ops, cdet, npg, mpg, q_pg):
    """CBAFORI warped branch (cbafori.F lines 120-272)."""
    iw = g["i_w"]
    bmw, bmfw, bfw, bcq, tc = ops
    c = cdet[iw]
    nn = npg[iw]
    mm = mpg[iw]
    qx, qy = q_pg[iw, 0], q_pg[iw, 1]
    bcx = tc[:, 0, 0] * qx + tc[:, 0, 1] * qy
    bcy = tc[:, 1, 0] * qx + tc[:, 1, 1] * qy
    for j in range(4):
        for comp in range(3):
            vf[iw, comp, j] += c * (
                bmw[:, j, comp, 0] * nn[:, 0] + bmw[:, j, comp, 1] * nn[:, 1]
                + bcq[:, j, comp, 0] * bcx + bcq[:, j, comp, 1] * bcy
                + bmfw[:, j, comp, 0] * mm[:, 0]
                + bmfw[:, j, comp, 1] * mm[:, 1]
                + bmfw[:, j, comp, 2] * mm[:, 2])
        for a in range(2):
            vm[iw, a, j] += c * (
                bcq[:, j, 3 + a, 0] * bcx + bcq[:, j, 3 + a, 1] * bcy
                + bfw[:, j, a, 0] * mm[:, 0] + bfw[:, j, a, 1] * mm[:, 1]
                + bfw[:, j, a, 2] * mm[:, 2])


def _cbaproj(g, vf, vm, off):
    """CBAPROJ: decode the reduced slots (flat) / project out the free
    rigid modes (warped), rotate to global, apply OFF. Returns global
    per-node forces fg (n,4,3) and moments mg (n,4,3)."""
    n = g["n"]
    E = g["E"]
    fg = np.zeros((n, 4, 3))
    mg = np.zeros((n, 4, 3))

    i = g["i_f"]
    if len(i):
        fl = np.empty((len(i), 3, 4))
        ml2 = np.empty((len(i), 2, 4))
        for comp in range(3):
            fl[:, comp, 0] = vf[i, comp, 0] + vf[i, comp, 2]
            fl[:, comp, 1] = vf[i, comp, 1] + vf[i, comp, 3]
            fl[:, comp, 2] = -vf[i, comp, 0] + vf[i, comp, 2]
            fl[:, comp, 3] = -vf[i, comp, 1] + vf[i, comp, 3]
        for a in range(2):
            ml2[:, a, 0] = vm[i, a, 0] + vm[i, a, 2]
            ml2[:, a, 1] = vm[i, a, 1] + vm[i, a, 3]
            ml2[:, a, 2] = -vm[i, a, 0] + vm[i, a, 2]
            ml2[:, a, 3] = -vm[i, a, 1] + vm[i, a, 3]
        Ei = E[i]
        # F_global[b] = sum_a E[b,a] * FL[a]; moments have no local z
        fg[i] = np.einsum("nba,naj->njb", Ei, fl)
        mg[i, :, :] = np.einsum("nba,naj->njb", Ei[:, :, 0:2], ml2)

    iw = g["i_w"]
    if len(iw):
        vqn = g["vqn"]
        corel = g["corel"]
        di = g["di"]
        vfw = np.stack([vf[iw, :, j] for j in range(4)], axis=2)  # (m,3,4)
        vmw = vm[iw]                                             # (m,2,4)
        mm = np.empty((len(iw), 3, 4))
        for j in range(4):
            mm[:, :, j] = vqn[:, j, 0:3] * vmw[:, 0, j][:, None] \
                + vqn[:, j, 3:6] * vmw[:, 1, j][:, None]
        z1 = corel[:, 2, 0]
        ar = np.empty((len(iw), 3))
        ar[:, 0] = (-z1 * (vfw[:, 1, 0] - vfw[:, 1, 1]
                           + vfw[:, 1, 2] - vfw[:, 1, 3])
                    + (corel[:, 1, :] * vfw[:, 2, :]).sum(axis=1)
                    + mm[:, 0, :].sum(axis=1))
        ar[:, 1] = (z1 * (vfw[:, 0, 0] - vfw[:, 0, 1]
                          + vfw[:, 0, 2] - vfw[:, 0, 3])
                    - (corel[:, 0, :] * vfw[:, 2, :]).sum(axis=1)
                    + mm[:, 1, :].sum(axis=1))
        ar[:, 2] = ((-corel[:, 1, :] * vfw[:, 0, :]
                     + corel[:, 0, :] * vfw[:, 1, :]).sum(axis=1)
                    + mm[:, 2, :].sum(axis=1))
        alr = np.empty((len(iw), 3))
        alr[:, 0] = di[:, 0] * ar[:, 0] + di[:, 3] * ar[:, 1] \
            + di[:, 4] * ar[:, 2]
        alr[:, 1] = di[:, 3] * ar[:, 0] + di[:, 1] * ar[:, 1] \
            + di[:, 5] * ar[:, 2]
        alr[:, 2] = di[:, 4] * ar[:, 0] + di[:, 5] * ar[:, 1] \
            + di[:, 2] * ar[:, 2]
        c1 = z1 * alr[:, 1]
        sgn = np.array([1.0, -1.0, 1.0, -1.0])
        for j in range(4):
            vfw[:, 0, j] += -sgn[j] * c1 + corel[:, 1, j] * alr[:, 2]
        c1 = z1 * alr[:, 0]
        for j in range(4):
            vfw[:, 1, j] += sgn[j] * c1 - corel[:, 0, j] * alr[:, 2]
        for j in range(4):
            vfw[:, 2, j] += -corel[:, 1, j] * alr[:, 0] \
                + corel[:, 0, j] * alr[:, 1]
            mm[:, :, j] -= alr
        Ew = E[iw]
        fg[iw] = np.einsum("nba,naj->njb", Ew, vfw)
        mg[iw] = np.einsum("nba,naj->njb", Ew, mm)

    fg *= off[:, None, None]
    mg *= off[:, None, None]
    return fg, mg


def _edofs(conn):
    """Global degree-of-freedom indices for 4-node shell (24 DOFs)."""
    n = len(conn)
    if n == 0:
        return np.empty((0, 24), dtype=np.int64)
    edofs = np.empty((n, 24), dtype=np.int64)
    for i in range(4):
        for c in range(6):
            edofs[:, i * 6 + c] = conn[:, i] * 6 + c
    return edofs


# ----------------------------------------------------------------------------
# Consistent element mass matrix
# ----------------------------------------------------------------------------

def consistent_mass(group, x=None):
    """rho*t*S x I3 on translations, rho*t^3/12*S x I3 on rotations —
    analytical bilinear quad shape function integral."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24), dtype=float), np.empty((0, 24), dtype=np.int64)
    mass = st["mass"]
    thick = st["thick"]
    m_rot = mass * thick ** 2 / 12.0
    me = np.zeros((n, 24, 24), dtype=float)
    for a in range(4):
        for b in range(4):
            s = _S_QUAD[a, b]
            for c in range(3):
                me[:, a * 6 + c, b * 6 + c] = mass * s
                me[:, a * 6 + 3 + c, b * 6 + 3 + c] = m_rot * s
    off = st.get("off")
    if off is not None:
        dead = off <= 0.0
        if np.any(dead):
            me[dead] = 0.0
    return me, _edofs(conn)


# ----------------------------------------------------------------------------
# 24-DOF Material Tangent Stiffness Matrix
# ----------------------------------------------------------------------------

def tangent(group, x, epsp_incr=None):
    """24-DOF material tangent stiffness matrix for 4-node fully-integrated QBAT shell.

    Evaluates 2x2 in-plane Gauss numerical integration combining:
      1. Membrane 2x2 Gauss integration: sum_g dA_g * t * B_m^T C_m B_m
      2. Bending 2x2 Gauss integration: sum_g dA_g * (t^3 / 12) * B_b^T C_b B_b
      3. Transverse shear 2x2 Gauss integration: sum_g dA_g * k_s G t * B_s^T B_s (k_s = 5/6)
      4. Drilling penalty coupled to continuum spin omega_z:
         g_i = theta_zi - 0.5 * sum_b (B1_b v_b - B2_b u_b), k_drill * sum_i g_i x g_i
         strictly preserving exact 6 rigid-body null modes.
      5. Frame transformation via orthonormal triad R = [e1, e2, e3]:
         ke = np.einsum("nap,nIpJq,nbq->nIaJb", R, Kl.reshape(n, 8, 3, 8, 3), R).reshape(n, 24, 24)

    Returns:
        ke (n, 24, 24): Symmetric tangent stiffness matrix.
        edofs (n, 24): Global DOF indices.
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24), dtype=float), np.empty((0, 24), dtype=np.int64)

    xe = x[conn]
    R, det = _frame(xe)
    area = np.maximum(0.25 * det, EM20)
    thick = st["thick"]

    # Local centered coordinates
    d = xe - xe[:, 0:1, :]
    xl = np.einsum("njk,nka->nja", d, R)
    cx = xl[:, :, 0] - xl[:, :, 0].mean(axis=1)[:, None]
    cy = xl[:, :, 1] - xl[:, :, 1].mean(axis=1)[:, None]

    # 2x2 Gauss points
    gp = np.array([-_PG, _PG])
    xi_pts = np.array([gp[0], gp[1], gp[1], gp[0]])
    eta_pts = np.array([gp[0], gp[0], gp[1], gp[1]])
    weights = np.ones(4)

    xi_n = _KSI_N
    eta_n = _ETA_N

    Kl = np.zeros((n, 24, 24), dtype=float)

    for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
        if getattr(mat, "law", 1) == 0:
            continue
        m = sl.stop - sl.start
        t_sl = thick[sl]
        nu = getattr(mat, "nu", 0.3)
        E_mod = getattr(mat, "E", 2.1e11)
        G_mod = getattr(mat, "G", E_mod / max(2.0 * (1.0 + nu), EM20))
        ks = SHEAR_FACTOR  # 5/6

        is_elastic = getattr(mat, "law", 1) == 1
        if is_elastic:
            Cm = materials.shell_membrane_tangent(mat)
            Cb = Cm
        else:
            zrel, wrel = st["zw"][isl]
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

        for g in range(4):
            xi_g, eta_g = xi_pts[g], eta_pts[g]
            wg = weights[g]
            N = 0.25 * (1.0 + xi_n * xi_g) * (1.0 + eta_n * eta_g)
            dN_dxi = 0.25 * xi_n * (1.0 + eta_n * eta_g)
            dN_deta = 0.25 * eta_n * (1.0 + xi_n * xi_g)

            cx_sl = cx[sl]
            cy_sl = cy[sl]

            J11 = np.sum(dN_dxi[None, :] * cx_sl, axis=1)
            J12 = np.sum(dN_dxi[None, :] * cy_sl, axis=1)
            J21 = np.sum(dN_deta[None, :] * cx_sl, axis=1)
            J22 = np.sum(dN_deta[None, :] * cy_sl, axis=1)
            detJ = np.maximum(J11 * J22 - J12 * J21, EM20)

            invJ11 = J22 / detJ
            invJ12 = -J12 / detJ
            invJ21 = -J21 / detJ
            invJ22 = J11 / detJ

            b1 = invJ11[:, None] * dN_dxi[None, :] + invJ12[:, None] * dN_deta[None, :]
            b2 = invJ21[:, None] * dN_dxi[None, :] + invJ22[:, None] * dN_deta[None, :]

            Bm = np.zeros((m, 3, 24), dtype=float)
            Bb = np.zeros((m, 3, 24), dtype=float)
            Bs = np.zeros((m, 2, 24), dtype=float)

            for a in range(4):
                # Membrane
                Bm[:, 0, 6 * a + 0] = b1[:, a]
                Bm[:, 1, 6 * a + 1] = b2[:, a]
                Bm[:, 2, 6 * a + 0] = b2[:, a]
                Bm[:, 2, 6 * a + 1] = b1[:, a]
                # Bending
                Bb[:, 0, 6 * a + 4] = b1[:, a]
                Bb[:, 1, 6 * a + 3] = -b2[:, a]
                Bb[:, 2, 6 * a + 3] = -b1[:, a]
                Bb[:, 2, 6 * a + 4] = b2[:, a]
                # Shear
                Bs[:, 0, 6 * a + 2] = b1[:, a]
                Bs[:, 0, 6 * a + 4] = N[a]
                Bs[:, 1, 6 * a + 2] = b2[:, a]
                Bs[:, 1, 6 * a + 3] = -N[a]

            dA = (wg * detJ)[:, None, None]

            if is_elastic:
                Kl[sl] += (dA * t_sl[:, None, None]) * np.einsum("mai,ab,mbj->mij", Bm, Cm, Bm)
                Kl[sl] += (dA * (t_sl ** 3 / 12.0)[:, None, None]) * np.einsum("mai,ab,mbj->mij", Bb, Cb, Bb)
            else:
                Kl[sl] += dA * (
                    np.einsum("mai,mab,mbj->mij", Bm, Am_, Bm)
                    + np.einsum("mai,mab,mbj->mij", Bm, Bm_, Bb)
                    + np.einsum("mai,mab,mbj->mij", Bb, Bm_, Bm)
                    + np.einsum("mai,mab,mbj->mij", Bb, Dm_, Bb)
                )

            Kl[sl] += (dA * (ks * G_mod * t_sl)[:, None, None]) * np.einsum("mai,maj->mij", Bs, Bs)

        # Spin-coupled drilling penalty
        a_inv = (0.5 / area[sl])[:, None]
        B1_sl = np.empty((m, 4), dtype=float)
        B1_sl[:, 0] = a_inv[:, 0] * (cy_sl[:, 1] - cy_sl[:, 3])
        B1_sl[:, 1] = a_inv[:, 0] * (cy_sl[:, 2] - cy_sl[:, 0])
        B1_sl[:, 2] = a_inv[:, 0] * (cy_sl[:, 3] - cy_sl[:, 1])
        B1_sl[:, 3] = a_inv[:, 0] * (cy_sl[:, 0] - cy_sl[:, 2])

        B2_sl = np.empty((m, 4), dtype=float)
        B2_sl[:, 0] = a_inv[:, 0] * (cx_sl[:, 3] - cx_sl[:, 1])
        B2_sl[:, 1] = a_inv[:, 0] * (cx_sl[:, 0] - cx_sl[:, 2])
        B2_sl[:, 2] = a_inv[:, 0] * (cx_sl[:, 1] - cx_sl[:, 3])
        B2_sl[:, 3] = a_inv[:, 0] * (cx_sl[:, 2] - cx_sl[:, 0])

        kdrill = (1e-3 * E_mod * t_sl ** 3 * area[sl] / 12.0)[:, None, None]
        for i in range(4):
            gi = np.zeros((m, 24), dtype=float)
            gi[:, 6 * i + 5] = 1.0
            for b in range(4):
                gi[:, 6 * b + 0] += 0.5 * B2_sl[:, b]
                gi[:, 6 * b + 1] += -0.5 * B1_sl[:, b]
            Kl[sl] += kdrill * np.einsum("mi,mj->mij", gi, gi)

    # Local to global transformation via triad R = [e1, e2, e3]
    Kl_blocks = Kl.reshape(n, 8, 3, 8, 3)
    ke = np.einsum("nap,nIpJq,nbq->nIaJb", R, Kl_blocks, R).reshape(n, 24, 24)

    off = st.get("off")
    if off is not None:
        dead = off <= 0.0
        if np.any(dead):
            ke[dead] = 0.0

    return ke, _edofs(conn)


# ----------------------------------------------------------------------------
# 24-DOF Geometric Stiffness Matrix
# ----------------------------------------------------------------------------

def kgeo(group, x):
    """24-DOF initial-stress geometric stiffness matrix for 4-node QBAT shell element.

    Couples translational displacements through 2x2 Gauss integrated in-plane membrane forces:
        k_geo,ab = sum_g dA_g * [ b1_a b1_b Nxx + b2_a b2_b Nyy + (b1_a b2_b + b2_a b1_b) Nxy ] * I3
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
    R, det = _frame(xe)
    thick = st["thick"]

    # Local centered coordinates
    d = xe - xe[:, 0:1, :]
    xl = np.einsum("njk,nka->nja", d, R)
    cx = xl[:, :, 0] - xl[:, :, 0].mean(axis=1)[:, None]
    cy = xl[:, :, 1] - xl[:, :, 1].mean(axis=1)[:, None]

    gp = np.array([-_PG, _PG])
    xi_pts = np.array([gp[0], gp[1], gp[1], gp[0]])
    eta_pts = np.array([gp[0], gp[0], gp[1], gp[1]])
    weights = np.ones(4)

    xi_n = _KSI_N
    eta_n = _ETA_N

    k_geo = np.zeros((n, 24, 24), dtype=float)
    I3 = np.eye(3, dtype=float)

    for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
        if getattr(mat, "law", 1) == 0:
            continue
        m = sl.stop - sl.start
        t_sl = thick[sl]
        zrel, wrel = st["zw"][isl]
        nip = len(zrel)
        nip_max = st["nip_max"]

        # 4-GP membrane resultants: shape (m, 4, 3)
        Nres = np.zeros((m, 4, 3), dtype=float)
        for g in range(4):
            for k in range(nip):
                idx = g * nip_max + k
                wk = wrel[k] * t_sl
                Nres[:, g, :] += wk[:, None] * st["sig"][sl, idx, :3]

        cx_sl = cx[sl]
        cy_sl = cy[sl]

        for g in range(4):
            xi_g, eta_g = xi_pts[g], eta_pts[g]
            wg = weights[g]
            dN_dxi = 0.25 * xi_n * (1.0 + eta_n * eta_g)
            dN_deta = 0.25 * eta_n * (1.0 + xi_n * xi_g)

            J11 = np.sum(dN_dxi[None, :] * cx_sl, axis=1)
            J12 = np.sum(dN_dxi[None, :] * cy_sl, axis=1)
            J21 = np.sum(dN_deta[None, :] * cx_sl, axis=1)
            J22 = np.sum(dN_deta[None, :] * cy_sl, axis=1)
            detJ = np.maximum(J11 * J22 - J12 * J21, EM20)

            invJ11 = J22 / detJ
            invJ12 = -J12 / detJ
            invJ21 = -J21 / detJ
            invJ22 = J11 / detJ

            b1 = invJ11[:, None] * dN_dxi[None, :] + invJ12[:, None] * dN_deta[None, :]
            b2 = invJ21[:, None] * dN_dxi[None, :] + invJ22[:, None] * dN_deta[None, :]

            Nxx = Nres[:, g, 0]
            Nyy = Nres[:, g, 1]
            Nxy = Nres[:, g, 2]
            dA = wg * detJ

            for a in range(4):
                for b in range(4):
                    gab = dA * (
                        b1[:, a] * b1[:, b] * Nxx
                        + b2[:, a] * b2[:, b] * Nyy
                        + (b1[:, a] * b2[:, b] + b2[:, a] * b1[:, b]) * Nxy
                    )
                    k_geo[sl, 6 * a:6 * a + 3, 6 * b:6 * b + 3] += gab[:, None, None] * I3

    off = st.get("off")
    if off is not None:
        dead = off <= 0.0
        if np.any(dead):
            k_geo[dead] = 0.0

    return k_geo, _edofs(conn)


# ----------------------------------------------------------------------------
# Static & Implicit Internal Forces
# ----------------------------------------------------------------------------

def static_internal_forces(group, x, u, ur, fint, mint):
    """Evaluate and scatter static internal forces and moments in deformed state x + u."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        return
    ke, edofs = tangent(group, x + u)
    conn = group.conn
    u_el = np.zeros((n, 24), dtype=float)
    for a in range(4):
        u_el[:, 6 * a:6 * a + 3] = u[conn[:, a]]
        u_el[:, 6 * a + 3:6 * a + 6] = ur[conn[:, a]]
    f_el = np.einsum("nij,nj->ni", ke, u_el)
    for a in range(4):
        scatter_add3(fint, conn[:, a], -f_el[:, 6 * a:6 * a + 3])
        scatter_add3(mint, conn[:, a], -f_el[:, 6 * a + 3:6 * a + 6])


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom=False):
    """Assemble implicit internal forces into fint and mint."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        return
    x_curr = x_ref + u if nlgeom else x_ref
    ke, edofs = tangent(group, x_curr)
    conn = group.conn
    u_el = np.zeros((n, 24), dtype=float)
    for a in range(4):
        u_el[:, 6 * a:6 * a + 3] = u[conn[:, a]]
        u_el[:, 6 * a + 3:6 * a + 6] = ur[conn[:, a]]
    f_el = np.einsum("nij,nj->ni", ke, u_el)
    for a in range(4):
        scatter_add3(fint, conn[:, a], -f_el[:, 6 * a:6 * a + 3])
        scatter_add3(mint, conn[:, a], -f_el[:, 6 * a + 3:6 * a + 6])

