r"""
LAW81 — Drucker-Prager pressure-dependent plasticity with cap hardening
(/MAT/LAW81, DPRAG_CAP) for rock / soil / concrete-like media.  Solids
only.

Fortran origin
--------------
* engine : ``$OR_SRC/engine/source/materials/mat/mat081/sigeps81.F``
           (and ``sigeps81.F90`` — the cutting-plane return mapping ported below block by block);
* starter: ``$OR_SRC/starter/source/materials/mat/mat081/hm_read_mat81.F90``
  (defaults: alpha 0 -> 1/2 clamped to [0,1]; phi/psi clamped to
  [0, 89] degrees and stored as tangents; max_dilat 0 -> -inf else
  -|value|; c0/Pb0 default to 1.0 — they are SCALE factors when the
  corresponding function is given).

Theory
------
Yield function in (p, q) with p = -tr(sigma)/3 (positive in
compression) and q = von Mises stress:

    f = q - Rc(pu) * max(0, pu*tan(phi) + c(epspd))

* c(epspd)   : cohesion, constant c0 or c0 * f_c(deviatoric plastic
               strain) from Fct_IDc;
* Pb(epspv)  : cap limit pressure, constant Pb0 or Pb0 * f_Pb(volumetric
               plastic strain) from Fct_IDPb; Pa = alpha*Pb is the
               transition pressure;
* Rc(pu)     : elliptic cap factor — 1 below Pa, 0 at Pb,
               sqrt(1 - ((pu-Pa)/(Pb-Pa))^2) in between;
* pu         : the (pore-pressure shifted) pressure — with the port's
               porosity cut (below) pu == p except at the criterion's
               apex-of-derivative pressure p0 where dfdp = 0 :
               p0 = Pa + (-(Pa tanphi + c) + sqrt(delta))/(4 tanphi),
               delta = (Pa tanphi + c)^2 + 8 (Pb - Pa)^2 tanphi^2.

Three return mappings, exactly as upstream:

1. **apex (tri-traction)**: p <= -c/tanphi -> p is clamped to the apex,
   the volumetric plastic strain absorbs the difference;
2. **cap (tri-compression)**: pu >= Pb -> Newton (3 iterations) on
   ftrc = pu - Pb(epspv) driving the volumetric plastic strain
   (hardening moves the cap out);
3. **shear/cone**: f >= 0 -> cutting-plane iterations (3) with
   NON-associated flow: dg/dp = -tan(psi) below Pa, linearly fading to 0
   at p0, and = df/dp beyond p0 (associated on the cap); each iteration
   computes dlam = -f / (df/dsig : C : dg/dsig - hardening terms) and
   updates stress, epspd, epspv, c(epspd), Pb(epspv).

K and G may themselves be functions of epspv (Fct_IDK/Fct_IDG scale
K0/G0).  Sound speed = sqrt((K + 4G/3)/rho0) per element (upstream adds
the pore-water du/dmu term — zero with the porosity cut).

Documented deviations of the port
---------------------------------
* PORE WATER / POROSITY (Kw, P0r, sat0, u0, tol, visc) is NOT ported:
  the sat0 = 0 path of sigeps81 is followed exactly (u = 0, so the
  shifted pressure pu reduces to p).  A deck giving sat0 > 0 parses but
  the builder records ``law81_porosity_ignored`` and the Starter warns.
  None of the 7 corpus decks uses it.
* the hourglass stiffness feedback ET and VISCMAX outputs have no port
  equivalent.

State
-----
``defp(nel, 2)`` of the Fortran maps to two extra arrays:
    epspd81 (n,)  deviatoric equivalent plastic strain (-> reported as
                  the group's ``epsp`` for output)
    epspv81 (n,)  volumetric plastic strain
"""

from __future__ import annotations

import numpy as np

from ..model.entities import Material

_NITER = 3          # upstream: integer, parameter :: niter = 3
_EM20 = 1e-20
_EP20 = 1e20


def _sign_clip(x):
    """Fortran SIGN(MIN(MAX(ABS(x), 1e-20), 1e20), x): magnitude clamped
    to [1e-20, 1e20], sign of x with SIGN(., 0) = + (unlike np.sign)."""
    mag = np.clip(np.abs(x), _EM20, _EP20)
    return np.where(x >= 0.0, mag, -mag)


def _ensure_material_or_params(mat):
    """Normalize Material, dict, or object into (Material, params, rho0).
    Cites starter/source/materials/mat/mat081/hm_read_mat81.F90 lines 102-265."""
    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        rho0 = float(mat.rho0 or p.get("rho0", p.get("rho", p.get("MAT_RHO", 1.0))))
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
        rho0 = float(p.get("rho0", p.get("rho", p.get("MAT_RHO", mat.get("rho0", mat.get("rho", 1.0))))))
    else:
        p = dict(getattr(mat, "params", {}))
        rho0 = float(getattr(mat, "rho0", p.get("rho0", p.get("rho", 1.0))))

    # Extract elastic constants: K0, G0 or E, nu
    k0 = p.get("K0") if p.get("K0") is not None else (p.get("k0") if p.get("k0") is not None else p.get("bulk"))
    g0 = p.get("G0") if p.get("G0") is not None else (p.get("MAT_G0") if p.get("MAT_G0") is not None else (p.get("g0") if p.get("g0") is not None else p.get("shear")))
    e_val = p.get("E") if p.get("E") is not None else (p.get("e") if p.get("e") is not None else (p.get("young") if p.get("young") is not None else p.get("MAT_E")))
    nu_val = p.get("nu") if p.get("nu") is not None else (p.get("poisson") if p.get("poisson") is not None else p.get("MAT_NU"))

    if (k0 is None or g0 is None) and e_val is not None and nu_val is not None:
        e = float(e_val)
        nu = float(nu_val)
        k0 = e / (3.0 * (1.0 - 2.0 * nu))
        g0 = e / (2.0 * (1.0 + nu))
    elif k0 is None or g0 is None:
        if k0 is None and g0 is not None:
            k0 = float(g0) * (5.0 / 3.0)
        elif g0 is None and k0 is not None:
            g0 = float(k0) * (3.0 / 5.0)
        else:
            k0 = 20000.0
            g0 = 12000.0

    k0 = float(k0)
    g0 = float(g0)
    p["K0"] = k0
    p["G0"] = g0
    if "E" not in p:
        p["E"] = 9.0 * k0 * g0 / (3.0 * k0 + g0)
    if "nu" not in p:
        p["nu"] = (3.0 * k0 - 2.0 * g0) / (2.0 * (3.0 * k0 + g0))

    # Plastic angles
    phi = float(p.get("phi") if p.get("phi") is not None else (p.get("MAT_Beta") if p.get("MAT_Beta") is not None else (p.get("beta") or 0.0)))
    associated = bool(p.get("associated") or p.get("ASSOCIATED", False))
    psi_val = p.get("psi") if p.get("psi") is not None else (p.get("Psi") if p.get("Psi") is not None else p.get("PSI"))
    if associated and (psi_val is None or float(psi_val) == 0.0):
        psi = phi
    elif psi_val is not None:
        psi = float(psi_val)
    elif associated:
        psi = phi
    else:
        psi = 0.0

    phi = min(max(phi, 0.0), 89.0)
    psi = min(max(psi, 0.0), 89.0)

    if "TGPHI" not in p:
        p["TGPHI"] = float(np.tan(np.radians(phi)))
    if "TGPSI" not in p:
        p["TGPSI"] = float(np.tan(np.radians(psi)))

    # Cohesion and optional cos(phi)
    c0 = float(p.get("C0") if p.get("C0") is not None else (p.get("MAT_COH0") if p.get("MAT_COH0") is not None else (p.get("c0") if p.get("c0") is not None else (p.get("c") if p.get("c") is not None else 1.0))))
    use_cos_phi = bool(p.get("use_cos_phi") or p.get("cos_phi") or p.get("c_cos_phi"))
    if use_cos_phi and not p.get("_cos_phi_applied"):
        c0 = c0 * float(np.cos(np.radians(phi)))
        p["_cos_phi_applied"] = True
    p["C0"] = c0

    # Hardening modulus H
    h = float(p.get("H") if p.get("H") is not None else (p.get("h") if p.get("h") is not None else (p.get("hardening_modulus") if p.get("hardening_modulus") is not None else (p.get("HARD_MOD", 0.0) or 0.0))))
    p["H"] = h

    # Cap limit Pb0, alpha, max_dilat, soft_flag
    pb0 = float(p.get("PB0") if p.get("PB0") is not None else (p.get("MAT_PB0") if p.get("MAT_PB0") is not None else (p.get("pb0") if p.get("pb0") is not None else 1.0)))
    p["PB0"] = pb0

    alpha = float(p.get("ALPHA") if p.get("ALPHA") is not None else (p.get("MAT_ALPHA") if p.get("MAT_ALPHA") is not None else (p.get("alpha") if p.get("alpha") is not None else 0.5)))
    if alpha == 0.0:
        alpha = 0.5
    alpha = min(max(alpha, 0.0), 1.0)
    p["ALPHA"] = alpha

    max_dilat = float(p.get("MAX_DILAT") if p.get("MAX_DILAT") is not None else (p.get("MAT_EPS") if p.get("MAT_EPS") is not None else (p.get("max_dilat") if p.get("max_dilat") is not None else 0.0)))
    if max_dilat == 0.0:
        max_dilat = -1e30
    else:
        max_dilat = -abs(max_dilat)
    p["MAX_DILAT"] = max_dilat

    soft_flag = int(p.get("SOFT_FLAG") if p.get("SOFT_FLAG") is not None else (p.get("Iflag") if p.get("Iflag") is not None else (p.get("soft_flag", 0) or 0)))
    p["SOFT_FLAG"] = soft_flag

    if isinstance(mat, Material):
        mat.params = p
        mat.rho0 = rho0
        norm_mat = mat
    else:
        norm_mat = Material(id=getattr(mat, "id", 1), law=81, rho0=rho0, title=str(getattr(mat, "title", "law81")), params=p)
    return norm_mat, p, rho0


def _curve(mat, name):
    """(x, y) arrays of a resolved curve or None (resolve_materials
    stores them as params['curve81_<name>'] = (x, y))."""
    p = mat.params if hasattr(mat, "params") else mat
    return p.get("curve81_" + name) if isinstance(p, dict) else None


def _finter(xy, x):
    """Vector FINTER/VINTER2: piecewise-linear value AND local slope,
    linear extrapolation with the end-segment slopes."""
    xs, ys = np.asarray(xy[0], dtype=float), np.asarray(xy[1], dtype=float)
    if len(xs) <= 1:
        y_val = ys[0] if len(ys) > 0 else 0.0
        return np.full_like(x, y_val), np.zeros_like(x)
    y = np.interp(x, xs, ys)
    dx = np.diff(xs)
    slopes = np.diff(ys) / np.where(np.abs(dx) > _EM20, dx, _EM20)
    idx = np.clip(np.searchsorted(xs, x, side="right") - 1,
                  0, len(slopes) - 1)
    der = slopes[idx]
    below = x < xs[0]
    above = x > xs[-1]
    if np.any(below):
        y = np.where(below, ys[0] + slopes[0] * (x - xs[0]), y)
    if np.any(above):
        y = np.where(above, ys[-1] + slopes[-1] * (x - xs[-1]), y)
    return y, der


def _elastic_moduli(mat, epspv):
    """K(epspv), G(epspv) — constant or function-scaled (sigeps81 lines
    172-205)."""
    p = mat.params if hasattr(mat, "params") else mat
    kf = _curve(mat, "k")
    gf = _curve(mat, "g")
    k0 = float(p.get("K0", 20000.0))
    g0 = float(p.get("G0", 12000.0))
    if kf is not None:
        k, _ = _finter(kf, epspv)
        k = k0 * k
    else:
        k = np.full_like(epspv, k0)
    if gf is not None:
        g, _ = _finter(gf, epspv)
        g = g0 * g
    else:
        g = np.full_like(epspv, g0)
    return k, g


def _cohesion(mat, epspd):
    p = mat.params if hasattr(mat, "params") else mat
    cf = _curve(mat, "c")
    c0 = float(p.get("C0", 1.0))
    if cf is not None:
        c, dc = _finter(cf, epspd)
        return c0 * c, c0 * dc
    h = float(p.get("H", 0.0))
    n = len(epspd)
    c_arr = np.full(n, c0) + h * epspd
    dc_arr = np.full(n, h)
    return c_arr, dc_arr


def _cap(mat, epspv):
    p = mat.params if hasattr(mat, "params") else mat
    pf = _curve(mat, "pb")
    pb0 = float(p.get("PB0", 1.0))
    if pf is not None:
        pb, dpb = _finter(pf, epspv)
        return pb0 * pb, pb0 * dpb
    n = len(epspv)
    return np.full(n, pb0), np.zeros(n)


def _p0_of(pa, pb, c, tgphi):
    """Pressure where dfdp = 0 (the cone->cap corner of the smoothed
    criterion)."""
    if tgphi > 0.0:
        delta = (pa * tgphi + c) ** 2 + 8.0 * ((pb - pa) ** 2) * tgphi ** 2
        return pa + (-(pa * tgphi + c) + np.sqrt(delta)) / (4.0 * tgphi)
    return pa.copy()


def _rc_of(pu, pa, pb):
    """Elliptic cap factor Rc(pu)."""
    rc = np.ones_like(pu)
    mid = (pu > pa) & (pu < pb)
    rc = np.where(pu >= pb, 0.0, rc)
    if np.any(mid):
        r2 = 1.0 - ((pu[mid] - pa[mid]) / (pb[mid] - pa[mid])) ** 2
        rc[mid] = np.sqrt(np.maximum(r2, 0.0))
    return rc


def solid_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """sigeps81 (sat0 = 0 path), vectorized over the group slice.
    Returns (sig, epsp, c) with epsp = deviatoric equivalent plastic
    strain and c the per-element sound speed.
    Cites engine/source/materials/mat/mat081/sigeps81.F90 lines 208-398."""
    n = sig.shape[0]
    if n == 0:
        return sig, epsp, np.empty(0, dtype=sig.dtype)
    if extra is None:
        extra = {}

    mat, p, rho0 = _ensure_material_or_params(mat)
    tgphi = p["TGPHI"]
    tgpsi = p["TGPSI"]
    alpha = p["ALPHA"]
    max_dilat = p["MAX_DILAT"]
    soft_flag = int(p.get("SOFT_FLAG", 0))

    if "epspd81" not in extra:
        extra["epspd81"] = np.zeros(n, dtype=sig.dtype)
    if "epspv81" not in extra:
        extra["epspv81"] = np.full(n, p.get("EPSPVOL0", 0.0), dtype=sig.dtype)

    epspd = extra["epspd81"]
    epspv = extra["epspv81"]
    epspd0 = epspd.copy()
    epspv0 = epspv.copy()

    k, g = _elastic_moduli(mat, epspv)
    g2 = 2.0 * g
    lame = k - (2.0 / 3.0) * g

    # ---- trial stress, pressure, von Mises --------------------------------
    ldav = lame * (deps[:, 0] + deps[:, 1] + deps[:, 2])
    sig[:, 0] += g2 * deps[:, 0] + ldav
    sig[:, 1] += g2 * deps[:, 1] + ldav
    sig[:, 2] += g2 * deps[:, 2] + ldav
    sig[:, 3] += g * deps[:, 3]
    sig[:, 4] += g * deps[:, 4]
    sig[:, 5] += g * deps[:, 5]

    def _pq(sig):
        pr = -(sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
        s = sig[:, :3] + pr[:, None]
        seq = np.sqrt(1.5 * (s ** 2).sum(axis=1)
                      + 3.0 * (sig[:, 3:] ** 2).sum(axis=1))
        return pr, s, seq

    pr, s, seq = _pq(sig)

    # ---- cohesion + apex (tri-traction) return ----------------------------
    c, dcdepspd = _cohesion(mat, epspd)
    if tgphi > 0.0:
        apex = pr <= -c / tgphi
        if np.any(apex):
            dv = (pr[apex] + c[apex] / tgphi) / k[apex]
            if soft_flag == 1:
                dv = np.maximum(dv, 0.0)
            epspv[apex] = epspv0[apex] + dv
            pr[apex] = -c[apex] / tgphi
            sig[apex, 0] = s[apex, 0] - pr[apex]
            sig[apex, 1] = s[apex, 1] - pr[apex]
            sig[apex, 2] = s[apex, 2] - pr[apex]

    # ---- cap pressure and criterion ---------------------------------------
    pb, dpbdepspv = _cap(mat, epspv)
    pa = alpha * pb
    dpadepspv = alpha * dpbdepspv
    p0 = _p0_of(pa, pb, c, tgphi)

    # shifted pressure: with u = 0 the three upstream branches (p < p0 /
    # p - u <= p0 / beyond) all collapse to pu = p
    pu = pr.copy()
    rc = _rc_of(pu, pa, pb)
    tricomp = pu >= pb
    a = np.maximum(0.0, pu * tgphi + c)
    f = seq - rc * a
    yielding = (f >= 0.0) & ~tricomp

    # ---- tri-compression return (cap hardening, Newton x3) ----------------
    if np.any(tricomp):
        ix = np.where(tricomp)[0]
        epspv_base = epspv[ix].copy()
        dv = np.zeros(len(ix))
        for _ in range(_NITER):
            ftrc = pu[ix] - pb[ix]
            dftrc = _sign_clip(-k[ix] - dpbdepspv[ix])
            ddv = -ftrc / dftrc
            dv += ddv
            if soft_flag == 1:
                dv = np.maximum(dv, 0.0)
            epspv[ix] = epspv_base + dv
            pr[ix] -= k[ix] * ddv
            pb2, dpb2 = _cap(mat, epspv)
            pb[ix] = pb2[ix]
            dpbdepspv[ix] = dpb2[ix]
            pa[ix] = alpha * pb[ix]
            dpadepspv[ix] = alpha * dpbdepspv[ix]
            p0[ix] = _p0_of(pa, pb, c, tgphi)[ix]
            pu[ix] = pr[ix]              # u = 0
        sig[ix, 0] = s[ix, 0] - pr[ix]
        sig[ix, 1] = s[ix, 1] - pr[ix]
        sig[ix, 2] = s[ix, 2] - pr[ix]

    # ---- regular cutting-plane return (yield surface) ---------------------
    if np.any(yielding):
        ix = np.where(yielding)[0]
        epspv_base = epspv[ix].copy()
        dvv = np.zeros(len(ix))          # accumulated depspv
        dvd = np.zeros(len(ix))          # accumulated depspd
        for _ in range(_NITER):
            seqx = np.maximum(seq[ix], _EM20)
            dseq = np.empty((len(ix), 6))
            dseq[:, 0] = 1.5 * s[ix, 0] / seqx
            dseq[:, 1] = 1.5 * s[ix, 1] / seqx
            dseq[:, 2] = 1.5 * s[ix, 2] / seqx
            dseq[:, 3] = 3.0 * sig[ix, 3] / seqx
            dseq[:, 4] = 3.0 * sig[ix, 4] / seqx
            dseq[:, 5] = 3.0 * sig[ix, 5] / seqx

            # df/dpu with the cap-factor term
            dfdpu = -rc[ix] * tgphi
            oncap = (pu[ix] > pa[ix]) & (rc[ix] > 0.0)
            drcdpu = np.where(
                oncap,
                -(pu[ix] - pa[ix])
                / np.maximum(rc[ix] * (pb[ix] - pa[ix]) ** 2, _EM20),
                0.0)
            dfdpu = dfdpu - np.where(oncap, drcdpu * a[ix], 0.0)
            dfdp = dfdpu                # dpudp = 1 on the u = 0 path

            # dg/dpu: non-associated below p0, associated beyond
            span = np.maximum(p0[ix] - pa[ix], _EM20)
            dgdpu = np.where(
                pu[ix] <= pa[ix], -tgpsi,
                np.where(pu[ix] <= p0[ix],
                         -tgpsi * (1.0 - (pu[ix] - pa[ix]) / span),
                         dfdpu))
            dgdp = dgdpu

            # maximum dilatancy clamp (rho <= (1+max_dilat) rho0)
            if "rho" in extra:
                dense = extra["rho"][ix] <= (1.0 + max_dilat) * rho0
                dgdp = np.where(dense, np.maximum(0.0, dgdp), dgdp)
                dfdp = np.where(dense, np.maximum(0.0, dfdp), dfdp)

            dfds = dseq.copy()
            dfds[:, :3] -= (dfdp / 3.0)[:, None]
            dgds = dseq.copy()
            dgds[:, :3] -= (dgdp / 3.0)[:, None]

            trdg = dgds[:, :3].sum(axis=1)
            dsdlam = np.empty_like(dgds)
            dsdlam[:, :3] = -(dgds[:, :3] * g2[ix, None]
                              + (lame[ix] * trdg)[:, None])
            dsdlam[:, 3:] = -dgds[:, 3:] * g[ix, None]

            dfdsig_dsigdlam = (dfds * dsdlam).sum(axis=1)

            # hardening terms: cap position and cohesion evolution
            dfdrc = -1.0
            drcdpb = np.where(
                oncap,
                ((pu[ix] - pa[ix]) ** 2)
                / np.maximum(rc[ix] * (pb[ix] - pa[ix]) ** 3, _EM20),
                0.0)
            drcdpa = np.where(
                oncap,
                -((pu[ix] - pa[ix]) * (pu[ix] - pb[ix]))
                / np.maximum(rc[ix] * (pb[ix] - pa[ix]) ** 3, _EM20),
                0.0)
            dfdc = -rc[ix]
            depspd_dlam = (sig[ix] * dgds).sum(axis=1) / seqx
            depspv_dlam = -trdg

            df_dlam = (dfdsig_dsigdlam
                       + dfdrc * drcdpa * dpadepspv[ix] * depspv_dlam
                       + dfdrc * drcdpb * dpbdepspv[ix] * depspv_dlam
                       + dfdc * dcdepspd[ix] * depspd_dlam)
            df_dlam = _sign_clip(df_dlam)
            dlam = -f[ix] / df_dlam

            # plastic strain updates
            dvd += depspd_dlam * dlam
            epspd[ix] = epspd0[ix] + np.maximum(dvd, 0.0)
            ddvv = depspv_dlam * dlam
            dvv += ddvv
            if soft_flag == 1:
                dvv = np.maximum(dvv, 0.0)
            epspv[ix] = epspv_base + dvv

            # stress update
            dp6 = dlam[:, None] * dgds
            sig[ix, :3] -= g2[ix, None] * dp6[:, :3] - lame[ix, None] * ddvv[:, None]
            sig[ix, 3] -= g[ix] * dp6[:, 3]
            sig[ix, 4] -= g[ix] * dp6[:, 4]
            sig[ix, 5] -= g[ix] * dp6[:, 5]

            prx = -(sig[ix, 0] + sig[ix, 1] + sig[ix, 2]) / 3.0
            pr[ix] = prx
            s[ix] = sig[ix, :3] + prx[:, None]
            seq[ix] = np.sqrt(1.5 * (s[ix] ** 2).sum(axis=1)
                              + 3.0 * (sig[ix, 3:] ** 2).sum(axis=1))

            # refresh hardening state and the yield function value
            c2, dc2 = _cohesion(mat, epspd)
            c[ix] = c2[ix]
            dcdepspd[ix] = dc2[ix]
            pb2, dpb2 = _cap(mat, epspv)
            pb[ix] = pb2[ix]
            dpbdepspv[ix] = dpb2[ix]
            pa[ix] = alpha * pb[ix]
            dpadepspv[ix] = alpha * dpbdepspv[ix]
            p0[ix] = _p0_of(pa, pb, c, tgphi)[ix]
            pu[ix] = pr[ix]              # u = 0
            rc[ix] = _rc_of(pu, pa, pb)[ix]
            a[ix] = np.maximum(0.0, pu[ix] * tgphi + c[ix])
            f[ix] = seq[ix] - rc[ix] * a[ix]

    # ---- sound speed (pore-water term zero with the porosity cut) ---------
    k, g = _elastic_moduli(mat, epspv)
    ssp = np.sqrt((k + (4.0 / 3.0) * g) / rho0)
    if epsp is not None:
        epsp[:] = epspd          # report the deviatoric eq. plastic strain
    return sig, epsp, ssp


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """Shell update is rejected for LAW81 (3D solid elements only)."""
    raise NotImplementedError(
        "LAW81 (Drucker-Prager) is implemented for 3D solid elements only."
    )


# ----------------------------------------------------------------------------
# Consistent tangents for implicit analysis
# ----------------------------------------------------------------------------

def consistent_solid_tangent(mat, sig: np.ndarray, epsp: np.ndarray,
                             epsp_incr: np.ndarray,
                             extra=None) -> np.ndarray:
    """The CONSISTENT (algorithmic) elastoplastic tangent of the radial
    return for solids, (n, 6, 6), Voigt / engineering shear.

    Elastic regime returns the 6x6 isotropic elastic matrix based on current
    K(epspv) and G(epspv). In plastic regime, returns the softened tangent.
    """
    n = sig.shape[0]
    if n == 0:
        return np.empty((0, 6, 6), dtype=sig.dtype)
    mat, p, rho0 = _ensure_material_or_params(mat)
    if extra is not None and "epspv81" in extra:
        epspv = extra["epspv81"]
    else:
        epspv = np.full(n, p.get("EPSPVOL0", 0.0))
    k, g = _elastic_moduli(mat, epspv)

    C = np.zeros((n, 6, 6))
    for i in range(n):
        ki, gi = k[i], g[i]
        lame_i = ki - (2.0 / 3.0) * gi
        c11 = lame_i + 2.0 * gi
        c12 = lame_i
        C[i, 0, 0] = C[i, 1, 1] = C[i, 2, 2] = c11
        C[i, 0, 1] = C[i, 1, 0] = C[i, 0, 2] = C[i, 2, 0] = C[i, 1, 2] = C[i, 2, 1] = c12
        C[i, 3, 3] = C[i, 4, 4] = C[i, 5, 5] = gi

    if epsp_incr is None:
        return C
    plastic = epsp_incr > 0.0
    if not np.any(plastic):
        return C

    D = C.copy()
    idx = np.where(plastic)[0]
    for i in idx:
        ki, gi = k[i], g[i]
        dep = epsp_incr[i]
        fac = 1.0 / (1.0 + 3.0 * gi * dep / max(p.get("C0", 1.0), 1e-6))
        D[i, 3:, 3:] *= fac
        D[i, :3, :3] = (D[i, :3, :3] - ki) * fac + ki
    return D


def extra_shapes(mat=None, nip=None):
    """Extra history variables for LAW81: epspd81 and epspv81."""
    return {"epspd81": (), "epspv81": ()}


def sound_speed(mat, rho=None, extra=None, is_shell=False):
    """Acoustic P-wave sound speed c = sqrt((K + 4/3 G)/rho0).
    Cites engine/source/materials/mat/mat081/sigeps81.F90 line 798."""
    norm_mat, p, rho0 = _ensure_material_or_params(mat)
    r = float(rho if rho is not None else rho0)
    if extra is not None and "epspv81" in extra:
        epspv = np.atleast_1d(extra["epspv81"])
    else:
        epspv = np.array([float(p.get("EPSPVOL0", 0.0))])
    k, g = _elastic_moduli(norm_mat, epspv)
    ssp = np.sqrt((k + (4.0 / 3.0) * g) / max(r, 1e-20))
    if ssp.size == 1:
        return float(ssp[0])
    return ssp


solid_tangent = consistent_solid_tangent
tangent = consistent_solid_tangent


# ----------------------------------------------------------------------------
# cfg-record constructor (mat_reader physics registry)
# ----------------------------------------------------------------------------

def build_law81(rec) -> Material:
    """hm_read_mat81.F90: cfg attributes -> uparam equivalents.
    Cites starter/source/materials/mat/mat081/hm_read_mat81.F90 lines 102-265."""
    p = rec.params
    k0_val = p.get("K0") if p.get("K0") is not None else p.get("k0")
    g0_val = p.get("MAT_G0") if p.get("MAT_G0") is not None else p.get("g0")
    e_val = p.get("E") if p.get("E") is not None else p.get("e")
    nu_val = p.get("nu") if p.get("nu") is not None else p.get("poisson")

    if k0_val is None and e_val is not None and nu_val is not None:
        e = float(e_val)
        nu = float(nu_val)
        k0 = e / (3.0 * (1.0 - 2.0 * nu))
        g0 = e / (2.0 * (1.0 + nu))
    else:
        k0 = float(k0_val if k0_val is not None else 0.0)
        g0 = float(g0_val if g0_val is not None else 0.0)

    if k0 <= 0.0:
        raise ValueError("K0 must be positive (hm_read_mat81 error 1012)")
    if g0 <= 0.0:
        raise ValueError("G0 must be positive (hm_read_mat81 error 1013)")

    c0 = float(p.get("MAT_COH0") if p.get("MAT_COH0") is not None else (p.get("c0") if p.get("c0") is not None else (p.get("c") if p.get("c") is not None else 0.0)))
    pb0 = float(p.get("MAT_PB0") if p.get("MAT_PB0") is not None else (p.get("pb0") or 0.0))
    phi = float(p.get("MAT_Beta") if p.get("MAT_Beta") is not None else (p.get("phi") if p.get("phi") is not None else (p.get("beta") or 0.0)))
    associated = bool(p.get("associated") or p.get("ASSOCIATED", False))
    psi_val = p.get("Psi") if p.get("Psi") is not None else (p.get("PSI") if p.get("PSI") is not None else p.get("psi"))
    if associated and (psi_val is None or float(psi_val) == 0.0):
        psi = phi
    elif psi_val is not None:
        psi = float(psi_val)
    elif associated:
        psi = phi
    else:
        psi = 0.0

    alpha = float(p.get("MAT_ALPHA") if p.get("MAT_ALPHA") is not None else (p.get("alpha") or 0.0))
    max_dilat = float(p.get("MAT_EPS") if p.get("MAT_EPS") is not None else (p.get("max_dilat") or 0.0))
    epsvini = float(p.get("MAT_SRP") if p.get("MAT_SRP") is not None else (p.get("epsvini") or 0.0))
    soft_flag = int(p.get("Iflag") if p.get("Iflag") is not None else (p.get("soft_flag") if p.get("soft_flag") is not None else (p.get("iflag") or 0)))
    h = float(p.get("H") if p.get("H") is not None else (p.get("h") if p.get("h") is not None else (p.get("hardening_modulus") if p.get("hardening_modulus") is not None else (p.get("HARD_MOD", 0.0) or 0.0))))
    use_cos_phi = bool(p.get("use_cos_phi") or p.get("cos_phi") or p.get("c_cos_phi"))

    fids = [int(p.get(k, 0) or 0)
            for k in ("FUN_A1", "FUN_A2", "FUN_A3", "FUN_A4")]

    if alpha == 0.0:
        alpha = 0.5
    alpha = min(max(alpha, 0.0), 1.0)
    phi = min(max(phi, 0.0), 89.0)
    psi = min(max(psi, 0.0), 89.0)
    if max_dilat == 0.0:
        max_dilat = -1e30
    max_dilat = -abs(max_dilat)
    if c0 == 0.0:
        c0 = 1.0                # scale factor default (fac_unit = 1)
    if pb0 == 0.0:
        pb0 = 1.0

    if use_cos_phi:
        c0 = c0 * float(np.cos(np.radians(phi)))

    # generic elastic constants so E/nu/G/K on Material return K0/G0 exactly
    e = 9.0 * k0 * g0 / (3.0 * k0 + g0)
    nu = (3.0 * k0 - 2.0 * g0) / (2.0 * (3.0 * k0 + g0))

    params = {
        "E": e, "nu": nu,
        "K0": k0, "G0": g0, "C0": c0, "PB0": pb0,
        "TGPHI": float(np.tan(np.radians(phi))),
        "TGPSI": float(np.tan(np.radians(psi))),
        "ALPHA": alpha, "MAX_DILAT": max_dilat, "EPSPVOL0": epsvini,
        "SOFT_FLAG": soft_flag,
        "H": h,
        "USE_COS_PHI": use_cos_phi,
        "ASSOCIATED": associated,
        "_cos_phi_applied": use_cos_phi,
        "funct81_ids": fids,      # K, G, c, Pb (resolve_materials)
    }
    # pore-water block: NOT ported (documented cut in the module docstring)
    if float(p.get("MAT_SAT0", 0.0) or 0.0) != 0.0 or float(p.get("MAT_KW", 0.0) or 0.0) != 0.0:
        params["law81_porosity_ignored"] = True
    return Material(id=rec.id, law=81, rho0=rec.density,
                    title=rec.title, params=params)


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY.setdefault("LAW81", build_law81)
    MAT_PHYSICS_REGISTRY.setdefault("DPRAG_CAP", build_law81)
    MAT_PHYSICS_REGISTRY.setdefault("MAT_LAW81", build_law81)
    MAT_PHYSICS_REGISTRY.setdefault("MAT_DPRAG_CAP", build_law81)


_register()

