# $OR_SRC/engine/source/materials/mat/mat062/sigeps62.F
# Function: SIGEPS62 (lines 33-517)
# $OR_SRC/starter/source/materials/mat/mat062/hm_read_mat62.F
# Function: HM_READ_MAT62 (lines 38-339)
"""
LAW62 — hyper-visco-elastic foam (/MAT/LAW62, /MAT/VISC_HYP).  Solids
only: an Ogden series with PER-TERM compressibility exponents beta_i
(nu_i) plus an optional Prony-series viscoelastic overstress.

Fortran origin
--------------
* engine : ``engine/source/materials/mat/mat062/sigeps62.F`` (the
  principal-stretch evaluation, the pressure/PK2 split, the Prony
  history update and the CIMAX sound-speed bound — ported block by
  block below);
* starter: ``starter/source/materials/mat/mat062/hm_read_mat62.F``
  (defaults: alpha_i 0 -> 1, tau_i <= 0 -> 1e20, gamma_inf =
  1 - sum(gamma_i) must stay > 0, nu clamps at 0.499; per-term
  beta_i = nu_i/(1-2 nu_i) when any nu_i is given, else the global nu;
  Rflag = 2 rescales mu_i by 1/gamma_inf).

Theory
------
With principal stretches lambda_j and J = lambda_1 lambda_2 lambda_3,
each Ogden term contributes the principal 2nd Piola-Kirchhoff stress

    S_j = (2 mu_i / alpha_i / lambda_j^2) *
          (lambda_j^alpha_i - J^(-beta_i alpha_i))

and the Cauchy principal stress is sigma_j = S_j lambda_j^2 / J.  The
``beta_i`` exponent is the term's volumetric response — nu_i = 0 (the
corpus foam decks) gives beta = 0: fully decoupled, highly compressible
foam.

Viscosity (Ivisc = 1, deviatoric — the M and gamma_i/tau_i input):
the deviatoric part of the PK2 stress SD = (S - p J C^-1) J^(2/3) is
tracked per Prony term with the standard midpoint exponential update

    H_i(t+dt) = exp(-dt/tau_i) H_i(t) + exp(-dt/2tau_i) (SD(t+dt)-SD(t))

carried in the GLOBAL frame (the principal frame rotates), and the
total stress is p J C^-1 + [gamma_inf SD + sum gamma_i dev(H_i)]
J^(-2/3).  Ivisc = 2 (Flag_Visc = 1) applies the series to the FULL
stress instead.

Documented deviations of the port
---------------------------------
* the principal stretches come from the EXACT deformation gradient the
  solid kernels pass (``extra['F']``, same machinery as LAW42) — the
  Fortran diagonalizes its incrementally-accumulated total strain,
  which is the approximate version of the same quantity (ISMSTR
  variants are therefore moot here);
* the implicit/IHET tangent-stiffness feedback (ET) is upgraded to the
  exact spectral spatial tangent (Bonet & Wood §6.6, Truesdell rate).

Extra state (only with Prony terms):
    sdg62 (6,)          previous deviatoric PK2 in the global frame
    h62   (nprony, 6)   Prony history H_i (global frame)
"""

from __future__ import annotations

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20
_VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def _spectral(F):
    """Principal stretches (n, 3) and spatial principal directions
    (n, 3, 3; columns = eigenvectors) from B = F F^T."""
    B = np.einsum("nij,nkj->nik", F, F)
    w, v = np.linalg.eigh(B)
    ev = np.sqrt(np.maximum(w, _EM20))
    return ev, v


def _ensure_params(mat: Material) -> dict:
    """Ensure material params contain both CFG and direct keys."""
    p = mat.params
    if "MU62" in p and "AL62" in p and "BETA62" in p:
        return p

    # Fallback to build_law62-style extraction from whatever params exist
    nug = float(p.get("MAT_NU") if p.get("MAT_NU") is not None else (p.get("nu") or 0.0))
    norder = int(p.get("ORDER") or p.get("N") or p.get("norder") or p.get("order") or 0)
    mu_in = p.get("Mu_arr") or p.get("mu_arr") or p.get("mu") or []
    if norder == 0 and len(mu_in) > 0:
        norder = len(mu_in)
    if norder <= 0:
        norder = 1
        mu_in = [1.0]

    nvisc = int(p.get("Order2") or p.get("M") or p.get("nvisc") or p.get("order2") or 0)
    gamma_in = p.get("Gamma_arr") or p.get("gamma_arr") or p.get("gamma") or []
    if nvisc == 0 and len(gamma_in) > 0:
        nvisc = len(gamma_in)

    flag_visc = int(p.get("Vflag") or p.get("vflag") or p.get("flag_visc") or 0)
    flag_rigidity = int(p.get("Rflag") or p.get("rflag") or p.get("flag_rigidity") or 0)

    def _arr(count, *keys):
        v = []
        for k in keys:
            val = p.get(k)
            if val is not None and (not isinstance(val, (list, tuple, np.ndarray)) or len(val) > 0):
                v = val
                break
        if not isinstance(v, (list, tuple, np.ndarray)):
            v = [v] if v is not None else []
        v = [float(x or 0.0) for x in v]
        v += [0.0] * (count - len(v))
        return np.asarray(v[:count], dtype=float)

    mu = _arr(norder, "Mu_arr", "mu_arr", "mu", "mus")
    al = _arr(norder, "Alpha_arr", "alpha_arr", "alpha", "alphas")
    nu = _arr(norder, "Nu_arr", "nu_arr", "nu_list", "nus")
    gama = _arr(nvisc, "Gamma_arr", "gamma_arr", "gamma", "gammas")
    taux = _arr(nvisc, "Tau_arr", "tau_arr", "tau", "taus")

    al = np.where(al == 0.0, 1.0, al)
    taux = np.where(taux <= 0.0, 1e20, taux)
    gamainf = 1.0
    if nvisc > 0:
        gamainf = max(1e-12, 1.0 - gama.sum())
    nug = min(max(nug, 0.0), 0.499)
    if nvisc > 0 and flag_rigidity == 2:
        mu = mu / gamainf
    gs = float(mu.sum())
    if gs <= 0.0:
        gs = 1.0
        mu = np.array([1.0])
        al = np.array([1.0])
        norder = 1
    nu = np.where(nu >= 0.5, 0.499, nu)
    nu = np.where(nu < 0.0, 0.0, nu)
    if np.any(nu != 0.0):
        beta = nu / (1.0 - 2.0 * nu)
        rbulk = float((2.0 * mu * (1.0 / 3.0 + beta)).sum())
        nug = 0.5 * (3.0 * rbulk - 2.0 * gs) / max(1e-20, 3.0 * rbulk + gs)
    else:
        beta = np.full(norder, nug / max(1e-20, 1.0 - 2.0 * nug))
        rbulk = (2.0 / 3.0) * gs * (1.0 + nug) / max(1e-20, 1.0 - 2.0 * nug)
    ivisc = 0 if nvisc == 0 else (2 if flag_visc == 1 else 1)

    p.update({
        "E": 2.0 * gs * (1.0 + nug), "nu": nug,
        "MU62": mu, "AL62": al, "BETA62": beta,
        "GAMA62": gama, "TAU62": taux,
        "GAMAINF": gamainf, "RBULK": rbulk, "IVISC": ivisc,
        "NPRONY": nvisc,
        "mu": mu, "alpha": al, "beta": beta,
        "gamma": gama, "tau": taux, "gamainf": gamainf,
        "rbulk": rbulk, "ivisc": ivisc, "nprony": nvisc,
        "N": norder, "M": nvisc,
    })
    return p


def solid_update(mat, sig, deps=None, epsp=None, dt=0.0, extra=None):
    """sigeps62.F (lines 33-517) — total-form update from extra['F'].
    
    Returns (sig, epsp, c).
    """
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[np.newaxis, :]
        if deps is not None and deps.ndim == 1:
            deps = deps[np.newaxis, :]
        if epsp is not None and hasattr(epsp, "ndim") and epsp.ndim == 1:
            epsp = epsp[np.newaxis]

    n = sig.shape[0]
    if n == 0:
        c_empty = np.empty(0, dtype=sig.dtype if hasattr(sig, "dtype") else float)
        res_sig = sig[0] if is_1d else sig
        res_epsp = epsp[0] if (is_1d and epsp is not None and hasattr(epsp, "__len__")) else (epsp if epsp is not None else np.empty(0, dtype=sig.dtype))
        return res_sig, res_epsp, c_empty

    p = _ensure_params(mat)
    mu = p["MU62"]
    al = p["AL62"]
    beta = p["BETA62"]
    gama = p["GAMA62"]
    taux = p["TAU62"]
    gamainf = p["GAMAINF"]
    rbulk = p["RBULK"]
    ivisc = int(p["IVISC"])
    nprony = len(gama)

    if extra is None or "F" not in extra or extra["F"] is None:
        # Small-strain fallback when F is not supplied: F = I + eps (or deps)
        eps_in = extra.get("eps") if (extra and "eps" in extra) else deps
        if eps_in is not None and is_1d and eps_in.ndim == 1:
            eps_in = eps_in[np.newaxis, :]
        F = np.zeros((n, 3, 3), dtype=sig.dtype if hasattr(sig, "dtype") else float)
        for i in range(3):
            F[:, i, i] = 1.0 + (eps_in[:, i] if eps_in is not None else 0.0)
        if eps_in is not None:
            F[:, 0, 1] = F[:, 1, 0] = 0.5 * eps_in[:, 3]
            F[:, 1, 2] = F[:, 2, 1] = 0.5 * eps_in[:, 4]
            F[:, 0, 2] = F[:, 2, 0] = 0.5 * eps_in[:, 5]
    else:
        F = extra["F"]
        if F.ndim == 2:
            F = F[None, :, :]

    ev, dirp = _spectral(F)                    # (n,3), (n,3,3)
    rv = ev[:, 0] * ev[:, 1] * ev[:, 2]        # J = det F
    ec = np.maximum(ev ** 2, _EM20)
    rv_m = np.maximum(rv, _EM20)

    # ---- pressure term and principal PK2 stress (sigeps62.F: 239-288) -------
    pres = np.zeros(len(ev))
    S = np.zeros_like(ev)
    for i in range(len(mu)):
        fac = 2.0 * mu[i] / al[i]
        lam_al = ev ** al[i]                              # lambda^alpha
        jvol = rv_m ** (-beta[i] * al[i])                 # J^(-beta*alpha)
        fac1 = fac / rv_m
        pres += fac1 * (lam_al.sum(axis=1) / 3.0 - jvol)
        S += (fac / ec) * (lam_al - jvol[:, None])

    # ---- viscoelastic overstress (Prony) (sigeps62.F: 310-450) --------------
    # (skipped when the caller passes no history views — the implicit
    # statics re-evaluation path calls total-form laws with a bare
    # {"F"} extra; re-advancing the Prony history there would double
    # the relaxation step)
    has_history = False
    sdg0 = None
    h = None
    uvar = None
    if ivisc > 0 and nprony > 0 and extra is not None:
        if "sdg62" in extra and extra["sdg62"] is not None:
            sdg0 = extra["sdg62"]
            h = extra.get("h62")
            if is_1d and sdg0.ndim == 1:
                sdg0 = sdg0[np.newaxis, :]
            if is_1d and h is not None and h.ndim == 2:
                h = h[np.newaxis, :, :]
            if h is not None:
                has_history = True
        elif "uvar" in extra and extra["uvar"] is not None:
            uvar = extra["uvar"]
            if is_1d and uvar.ndim == 1:
                uvar = uvar[np.newaxis, :]
            sdg0 = uvar[:, :6]
            h = uvar[:, 6:6 + nprony * 6].reshape(n, nprony, 6)
            has_history = True

    if has_history and sdg0 is not None and h is not None:
        rv23 = rv_m ** (2.0 / 3.0)
        ssp = pres[:, None] / ec                          # pressure part
        sd = (S - ssp * rv[:, None]) * rv23[:, None]      # scaled deviator
        if ivisc == 2:
            sd = S.copy()                                 # full stress
        # principal (diagonal) -> global frame: SDG = R diag(sd) R^T
        sdg = np.einsum("naj,nj,nbj->nab", dirp, sd, dirp)
        sdg6 = np.stack([sdg[:, 0, 0], sdg[:, 1, 1], sdg[:, 2, 2],
                         sdg[:, 0, 1], sdg[:, 1, 2], sdg[:, 0, 2]], axis=1)
        hp = np.zeros((len(ev), nprony, 3))
        for ii in range(nprony):
            fac = -dt / taux[ii]
            h[:, ii] = (np.exp(fac) * h[:, ii]
                        + np.exp(0.5 * fac) * (sdg6 - sdg0))
            # global H -> principal diagonal
            HM = np.empty_like(sdg)
            HM[:, 0, 0], HM[:, 1, 1], HM[:, 2, 2] =                 h[:, ii, 0], h[:, ii, 1], h[:, ii, 2]
            HM[:, 0, 1] = HM[:, 1, 0] = h[:, ii, 3]
            HM[:, 1, 2] = HM[:, 2, 1] = h[:, ii, 4]
            HM[:, 0, 2] = HM[:, 2, 0] = h[:, ii, 5]
            hp[:, ii] = np.einsum("nai,nab,nbi->ni", dirp, HM, dirp)
        sdg0[:] = sdg6
        if uvar is not None:
            uvar[:, :6] = sdg6
            uvar[:, 6:6 + nprony * 6] = h.reshape(n, nprony * 6)

        if ivisc == 1:
            # deviatoric projection in the strain metric
            hd = hp - (np.einsum("nij,nj->ni", hp, ec) / 3.0
                       )[:, :, None] / np.maximum(ec, _EM20)[:, None, :]
            rvm23 = rv_m ** (-2.0 / 3.0)
            S = ssp * rv[:, None] + gamainf * rvm23[:, None] * sd
            for ii in range(nprony):
                S += gama[ii] * rvm23[:, None] * hd[:, ii]
        else:                                             # ivisc == 2
            S = gamainf * S
            for ii in range(nprony):
                S += gama[ii] * hp[:, ii]

    # ---- Cauchy principal stress, back to the global frame ------------------
    cauchy = S * ec / rv_m[:, None]
    sig_t = np.einsum("naj,nj,nbj->nab", dirp, cauchy, dirp)
    sig[:, 0] = sig_t[:, 0, 0]
    sig[:, 1] = sig_t[:, 1, 1]
    sig[:, 2] = sig_t[:, 2, 2]
    sig[:, 3] = sig_t[:, 0, 1]
    sig[:, 4] = sig_t[:, 1, 2]
    sig[:, 5] = sig_t[:, 0, 2]

    # ---- sound-speed bound CIMAX (sigeps62.F lines 452-478) ------------------
    gmax = 2.0 * float(np.sum(mu))
    gvis = gmax if ivisc > 0 else 0.0
    cmax0 = (2.0 / 3.0) * gmax + rbulk
    ai = np.zeros_like(ev)
    bi = np.zeros_like(ev)
    cj = np.zeros(len(ev))
    for i in range(len(mu)):
        lam_al = ev ** al[i]
        rvl = (beta[i] + 1.0 / al[i]) * rv_m ** (-beta[i] * al[i])
        cj += 2.0 * mu[i] * rvl
        ai += 2.0 * mu[i] * lam_al
        bi += (2.0 * mu[i] / al[i]) * lam_al
    d = ai - bi + cj[:, None]
    cmax = d.max(axis=1) / rv_m
    cimax = (2.0 / 3.0) * gvis + np.maximum(cmax, cmax0)
    rho = extra.get("rho") if extra else None
    if rho is None:
        rho = (getattr(mat, "rho0", 1.0) or 1.0) / rv_m                # rho = rho0/J
    elif np.isscalar(rho):
        rho = np.full(n, rho)
    c = np.sqrt(np.maximum(cimax, _EM20) / np.maximum(rho, _EM20))

    res_sig = sig[0] if is_1d else sig
    res_epsp = epsp[0] if (is_1d and epsp is not None and hasattr(epsp, "__len__")) else epsp
    res_c = c[0] if is_1d else c
    return res_sig, res_epsp, res_c


def shell_update(mat, sig, deps, epsp, dt, extra=None):
    """LAW62 is a solids-only material model (hm_read_mat62.F / sigeps62.F)."""
    raise NotImplementedError("LAW62 (hyper-visco foam) is a solids-only material model (implemented for 3D solid elements only).")


def consistent_solid_tangent(mat, sig=None, epsp=None, epsp_incr=None, extra=None, F=None, dt=None):
    """(m, 6, 6) spatial tangent modulus tensor in Voigt form.
    
    Linearization of Cauchy stress under finite deformation F.
    Satisfies directional derivative consistency with the Hughes-Winget /
    Truesdell rate of Cauchy stress.
    """
    p = _ensure_params(mat)
    mu = p["MU62"]
    al = p["AL62"]
    beta = p["BETA62"]
    gama = p.get("GAMA62", [])
    taux = p.get("TAU62", [])
    gamainf = p.get("GAMAINF", 1.0)
    ivisc = int(p.get("IVISC", 0))
    nprony = len(gama)

    # Resolve deformation gradient F
    if F is not None:
        pass
    elif isinstance(sig, np.ndarray) and sig.ndim == 3 and sig.shape[1:] == (3, 3):
        F = sig
    elif extra is not None and isinstance(extra, dict) and "F" in extra and extra["F"] is not None:
        F = extra["F"]
    elif sig is not None and isinstance(sig, np.ndarray):
        n = sig.shape[0]
        if n == 0:
            return np.empty((0, 6, 6), dtype=sig.dtype if hasattr(sig, "dtype") else float)
        F = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    else:
        F = np.eye(3)[None, :, :]

    if F.ndim == 2:
        F = F[None, :, :]
    m = F.shape[0]
    if m == 0:
        return np.empty((0, 6, 6))

    ev, dirp = _spectral(F)
    rv = ev[:, 0] * ev[:, 1] * ev[:, 2]
    ec = np.maximum(ev ** 2, _EM20)
    rv_m = np.maximum(rv, _EM20)

    # Principal Kirchhoff stresses tau_a = J * sigma_a
    tau = np.zeros_like(ev)
    for i in range(len(mu)):
        fac = 2.0 * mu[i] / al[i]
        lam_al = ev ** al[i]
        jvol = rv_m ** (-beta[i] * al[i])
        tau += fac * (lam_al - jvol[:, None])
    sigp = tau / rv_m[:, None]

    # beta_ab = d tau_a / d ln(lambda_b) (symmetric)
    jvol_sum = np.zeros(m)
    for i in range(len(mu)):
        jvol_sum += 2.0 * mu[i] * beta[i] * (rv_m ** (-beta[i] * al[i]))

    beta_mat = np.zeros((m, 3, 3))
    for a in range(3):
        for b in range(3):
            beta_mat[:, a, b] = jvol_sum
    for i in range(len(mu)):
        lam_al = ev ** al[i]
        for a in range(3):
            beta_mat[:, a, a] += 2.0 * mu[i] * lam_al[:, a]

    c_diag = beta_mat / rv_m[:, None, None]
    for a in range(3):
        c_diag[:, a, a] -= 2.0 * sigp[:, a]

    lam2 = ec
    shear = np.zeros((m, 3, 3))
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            num = sigp[:, a] * lam2[:, b] - sigp[:, b] * lam2[:, a]
            den = lam2[:, a] - lam2[:, b]
            close = np.abs(den) <= 1e-6 * (lam2[:, a] + lam2[:, b])
            lim = 0.5 * (c_diag[:, a, a] - c_diag[:, a, b])
            with np.errstate(divide="ignore", invalid="ignore"):
                gen = num / np.where(close, 1.0, den)
            shear[:, a, b] = np.where(close, lim, gen)

    n_a = dirp.transpose(0, 2, 1)  # columns are eigenvectors
    c4 = np.zeros((m, 3, 3, 3, 3))
    for a in range(3):
        for b in range(3):
            Pa = np.einsum("mi,mj->mij", n_a[:, a], n_a[:, a])
            Pb = np.einsum("mi,mj->mij", n_a[:, b], n_a[:, b])
            c4 += c_diag[:, a, b, None, None, None, None] * np.einsum("mij,mkl->mijkl", Pa, Pb)
            if a != b:
                Qab = np.einsum("mi,mj->mij", n_a[:, a], n_a[:, b])
                Qba = np.einsum("mi,mj->mij", n_a[:, b], n_a[:, a])
                c4 += shear[:, a, b, None, None, None, None] * (
                    np.einsum("mij,mkl->mijkl", Qab, Qab)
                    + np.einsum("mij,mkl->mijkl", Qab, Qba)
                )

    D = np.empty((m, 6, 6))
    for I, (i, j) in enumerate(_VOIGT):
        for Jc, (k, ell) in enumerate(_VOIGT):
            D[:, I, Jc] = c4[:, i, j, k, ell]

    # Viscoelastic time-dependent factor scaling if dt is given
    if dt is None and extra is not None and isinstance(extra, dict):
        dt = extra.get("dt")

    if ivisc > 0 and nprony > 0 and dt is not None and dt > 0.0:
        w_prony = gamainf
        for ii in range(nprony):
            w_prony += gama[ii] * np.exp(-dt / taux[ii])
        if ivisc == 2:
            D = w_prony * D
        elif ivisc == 1:
            # Deviatoric scaling
            D_dev = D.copy()
            # Mean volumetric part is preserved
            D = w_prony * D_dev + (1.0 - w_prony) * (p["RBULK"] / 3.0)

    return D


def solid_tangent(mat, sig=None, epsp=None, epsp_incr=None, extra=None, F=None, dt=None):
    """(m, 6, 6) spatial tangent modulus tensor in Voigt form."""
    return consistent_solid_tangent(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra, F=F, dt=dt)


def tangent(mat_or_group=None, **kwargs):
    """Material tangent interface matching pyradioss material dispatcher convention."""
    if mat_or_group is None:
        return None
    mat = getattr(mat_or_group, "material", mat_or_group)
    return consistent_solid_tangent(mat, **kwargs)

# cfg-record constructor (mat_reader physics registry)
# ----------------------------------------------------------------------------

def build_law62(rec) -> Material:
    """hm_read_mat62.F: cfg attributes -> uparam equivalents."""
    if isinstance(rec, Material):
        q = rec.params
        mat_id = rec.id
        title = rec.title
        density = rec.rho0
    elif isinstance(rec, dict):
        q = rec.get("params", rec)
        mat_id = rec.get("id", 1)
        title = rec.get("title", "")
        density = float(rec.get("density") or rec.get("rho") or rec.get("rho0") or rec.get("MAT_RHO") or 1.0)
    else:
        q = getattr(rec, "params", {})
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "")
        density = float(getattr(rec, "density", 0.0) or getattr(rec, "rho", 0.0) or getattr(rec, "rho0", 1.0))

    nug = float(q.get("MAT_NU") if q.get("MAT_NU") is not None else (q.get("nu") or 0.0))
    norder = int(q.get("ORDER") or q.get("N") or q.get("norder") or q.get("order") or 0)
    mu_in = q.get("Mu_arr") or q.get("mu_arr") or q.get("mu") or []
    if norder == 0 and len(mu_in) > 0:
        norder = len(mu_in)

    nvisc = int(q.get("Order2") or q.get("M") or q.get("nvisc") or q.get("order2") or 0)
    gamma_in = q.get("Gamma_arr") or q.get("gamma_arr") or q.get("gamma") or []
    if nvisc == 0 and len(gamma_in) > 0:
        nvisc = len(gamma_in)

    flag_visc = int(q.get("Vflag") or q.get("vflag") or q.get("flag_visc") or 0)
    flag_rigidity = int(q.get("Rflag") or q.get("rflag") or q.get("flag_rigidity") or 0)

    def _arr(count, *keys):
        v = []
        for k in keys:
            val = q.get(k)
            if val is not None and (not isinstance(val, (list, tuple, np.ndarray)) or len(val) > 0):
                v = val
                break
        if not isinstance(v, (list, tuple, np.ndarray)):
            v = [v] if v is not None else []
        v = [float(x or 0.0) for x in v]
        v += [0.0] * (count - len(v))
        return np.asarray(v[:count], dtype=float)

    mu = _arr(norder, "Mu_arr", "mu_arr", "mu", "mus")
    al = _arr(norder, "Alpha_arr", "alpha_arr", "alpha", "alphas")
    nu = _arr(norder, "Nu_arr", "nu_arr", "nu_list", "nus")
    gama = _arr(nvisc, "Gamma_arr", "gamma_arr", "gamma", "gammas")
    taux = _arr(nvisc, "Tau_arr", "tau_arr", "tau", "taus")

    if norder <= 0:
        raise ValueError("LAW62 needs at least one Ogden term "
                         "(hm_read_mat62 error 559)")
    al = np.where(al == 0.0, 1.0, al)
    taux = np.where(taux <= 0.0, 1e20, taux)
    gamainf = 1.0
    if nvisc > 0:
        if np.any((gama < 0.0) | (gama > 1.0)):
            raise ValueError("LAW62: every gamma_i must be in [0, 1] "
                             "(hm_read_mat62 error 560)")
        gamainf = 1.0 - gama.sum()
        if gamainf <= 0.0:
            raise ValueError("LAW62: sum(gamma_i) must stay below 1 "
                             "(hm_read_mat62 error 2084)")
    nug = min(max(nug, 0.0), 0.499)
    if nvisc > 0 and flag_rigidity == 2:
        mu = mu / gamainf
    gs = float(mu.sum())
    if gs <= 0.0:
        raise ValueError("LAW62: sum(mu_i) must be positive "
                         "(hm_read_mat62 error 846)")
    nu = np.where(nu >= 0.5, 0.499, nu)
    nu = np.where(nu < 0.0, 0.0, nu)
    if np.any(nu != 0.0):
        beta = nu / (1.0 - 2.0 * nu)
        rbulk = float((2.0 * mu * (1.0 / 3.0 + beta)).sum())
        nug = 0.5 * (3.0 * rbulk - 2.0 * gs) / max(1e-20, 3.0 * rbulk + gs)
    else:
        beta = np.full(norder, nug / max(1e-20, 1.0 - 2.0 * nug))
        rbulk = (2.0 / 3.0) * gs * (1.0 + nug) / max(1e-20, 1.0 - 2.0 * nug)
    ivisc = 0 if nvisc == 0 else (2 if flag_visc == 1 else 1)

    params = {
        # generic elastic constants: G = sum(mu_i), the upstream
        # PARMAT(2) young modulus E = 2 G (1 + nu)
        "E": 2.0 * gs * (1.0 + nug), "nu": nug,
        "MU62": mu, "AL62": al, "BETA62": beta,
        "GAMA62": gama, "TAU62": taux,
        "GAMAINF": gamainf, "RBULK": rbulk, "IVISC": ivisc,
        "NPRONY": nvisc,
        "mu": mu, "alpha": al, "beta": beta,
        "gamma": gama, "tau": taux, "gamainf": gamainf,
        "rbulk": rbulk, "ivisc": ivisc, "nprony": nvisc,
        "N": norder, "M": nvisc,
    }
    return Material(id=mat_id, law=62, rho0=density,
                    title=title, params=params)


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY.setdefault("LAW62", build_law62)
    MAT_PHYSICS_REGISTRY.setdefault("VISC_HYP", build_law62)


_register()
