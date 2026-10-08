r"""
LAW163 — Crushable Foam Material (/MAT/LAW163, /MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM).

Upstream Fortran reference:
- Engine physics: ``$OR_SRC/engine/source/materials/mat/mat163/sigeps163.F90``
  Subroutine SIGEPS163 (lines 53-309)
- Starter card reader: ``$OR_SRC/starter/source/materials/mat/mat163/hm_read_mat163.F90``
- Starter property update: ``$OR_SRC/starter/source/materials/mat/mat163/law163_upd.F90``

Theory & Algorithm (sigeps163.F90)
----------------------------------
1. Elastic trial stress:
   Stiffness coefficients:
     g = E / (2 * (1 + nu))
     bulk = E / (3 * (1 - 2 * nu))
     cii = bulk + 4/3 * g
     cij = bulk - 2/3 * g
   Trial stress increment:
     st_xx = sig_xx + cii*deps_xx + cij*deps_yy + cij*deps_zz
     st_yy = sig_yy + cij*deps_xx + cii*deps_yy + cij*deps_zz
     st_zz = sig_zz + cij*deps_xx + cij*deps_yy + cii*deps_zz
     st_xy = sig_xy + g*deps_xy
     st_yz = sig_yz + g*deps_yz
     st_zx = sig_zx + g*deps_zx

2. Principal stress decomposition:
   Eigenvalues sigp = (sig1, sig2, sig3) and eigenvectors dirp = (n1, n2, n3).

3. Volumetric strain & strain rate filtering:
   gama = 1 - rho0 / rho
   If nrs == 1:
     dgamdt = (gama - uvar1) / max(dt, 1e-20)
   Else (nrs == 0):
     dgamdt = -(deps_xx + deps_yy + deps_zz) / max(dt, 1e-20)
   alpha = 2 * pi / (2 * pi + ncycle)
   dgamdt = alpha * dgamdt + (1 - alpha) * epsd_old
   If abs(dgamdt - epsd_old) > srclmt * dt:
     dgamdt = epsd_old + sign(dgamdt - epsd_old) * srclmt * dt
   uvar1 = gama
   epsd = dgamdt
   plas = log(max(rho0 / rho, 1e-20))

4. Tabulated yield stress:
   Lookup with (max(gama, 0), max(dgamdt, 0)) -> sigy_val, dsdgam
   sigy = -abs(sigy_val) * fscale
   dsdgam = max(dsdgam, 0.0)

5. Principal stress return & tensile cutoff:
   For j in (1, 2, 3):
     if sigp[j] < sigy: sigp[j] = sigy
     elif sigp[j] > tsc: sigp[j] = tsc

6. Reconstruct Cauchy stress:
   sig_new = sum_j (sigp[j] * (n_j (x) n_j))

7. Viscous damping:
   denom = copysign(max(abs(1 + gama), 1e-20), 1 + gama)
   ssp0 = sqrt((max(bulk, dsdgam) + 4/3*g) / rho)
   a = ssp0 * rho * damp * le / denom
   ldav = (deps_xx + deps_yy + deps_zz) / (3.0 * max(dt, 1e-20))
   sigv_xx = a * ((epsp_xx - ldav)/(1 + nu) + ldav/(1 - 2*nu))
   ...
   sigv_xy = a * epsp_xy / (2 * (1 + nu))
   Total stress: sig_tot = sig_new + sigv

8. Sound speed:
   If dt > 0:
     c = sqrt((max(bulk, dsdgam) + 4/3*g + abs(a)/max(dt, 1e-20)) / rho)
   Else:
     c = sqrt((max(bulk, dsdgam) + 4/3*g) / rho)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..common.tables import FunctTable, SmoothFunctTable
from ..model.entities import Material, MatLaw163

_EM20 = 1.0e-20
_TWO_PI = 2.0 * math.pi
_FOUR_THIRD = 4.0 / 3.0
_TWO_THIRD = 2.0 / 3.0


@dataclass
class Law163Params:
    """Parameters for /MAT/LAW163 (/MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM).

    Matches ``hm_read_mat163.F90`` and ``law163_upd.F90``.
    """

    rho0: float = 1.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    tsc: float = 0.0            # Tensile stress cutoff
    damp: float = 0.10          # Damping coefficient (default 0.10)
    ncycle: int = 12            # Number of cycles for strain-rate filtering (default 12)
    srclmt: float = 1.0e20      # Strain-rate change limit (default 1e20)
    tab_id: int = 0             # Table ID for yield stress vs volumetric strain
    epsd_ref: float = 0.0       # Reference strain rate
    fscale: float = 1.0         # Scale factor for yield stress table
    nrs: int = 0                # Strain rate flag: 0 = true rate, 1 = engineering rate
    table: Any = None           # Resolved yield table, curve, or function
    title: str = ""

    # Derived elastic stiffness properties
    g: float = 0.0              # Shear modulus
    bulk: float = 0.0           # Bulk modulus
    cii: float = 0.0            # Diagonal stiffness (bulk + 4/3*g)
    cij: float = 0.0            # Off-diagonal stiffness (bulk - 2/3*g)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        # Upstream hm_read_mat163: nu = min(max(nu, 0.0), 0.499)
        self.nu = min(max(float(self.nu), 0.0), 0.499)
        if self.damp < 0.0:
            self.damp = 0.10
        if self.ncycle <= 0:
            self.ncycle = 12
        if self.srclmt <= 0.0:
            self.srclmt = 1.0e20
        if self.fscale == 0.0:
            self.fscale = 1.0
        self.nrs = max(min(int(self.nrs), 1), 0)

        # Derived elastic moduli
        if self.e > 0.0:
            if self.g == 0.0:
                self.g = self.e / (2.0 * (1.0 + self.nu))
            if self.bulk == 0.0:
                self.bulk = self.e / (3.0 * (1.0 - 2.0 * self.nu))
            if self.cii == 0.0:
                self.cii = self.bulk + _FOUR_THIRD * self.g
            if self.cij == 0.0:
                self.cij = self.bulk - _TWO_THIRD * self.g
        elif self.bulk > 0.0 or self.g > 0.0:
            if self.cii == 0.0:
                self.cii = self.bulk + _FOUR_THIRD * self.g
            if self.cij == 0.0:
                self.cij = self.bulk - _TWO_THIRD * self.g


def _extract_params(mat: Any) -> Law163Params:
    """Helper to get a Law163Params instance from any supported input."""
    if isinstance(mat, Law163Params):
        return mat

    if isinstance(mat, MatLaw163):
        tbl = getattr(mat, "table", mat.params.get("table", None) if hasattr(mat, "params") else None)
        return Law163Params(
            rho0=mat.rho if mat.rho > 0 else 1.0,
            refer_rho=mat.rho if mat.rho > 0 else 1.0,
            e=mat.e,
            nu=mat.nu,
            tsc=mat.tsc,
            damp=mat.damp if mat.damp > 0 else 0.10,
            ncycle=mat.ncycle if mat.ncycle > 0 else 12,
            srclmt=mat.srclmt if mat.srclmt > 0 else 1.0e20,
            tab_id=mat.tab_id,
            epsd_ref=mat.epsd_ref,
            fscale=mat.fscale if mat.fscale != 0.0 else 1.0,
            nrs=mat.nrs,
            table=tbl,
            title=getattr(mat, "title", ""),
        )

    if isinstance(mat, Material):
        p = mat.params if mat.params is not None else {}
        if "law163_params" in p and isinstance(p["law163_params"], Law163Params):
            return p["law163_params"]
        rho0 = float(mat.rho0 if mat.rho0 > 0 else p.get("rho0", p.get("rho", p.get("MAT_RHO", 1.0))))
        e = float(p.get("e", p.get("MAT_E", p.get("E", 0.0))))
        nu = float(p.get("nu", p.get("MAT_NU", p.get("Nu", p.get("pr", 0.0)))))
        tsc = float(p.get("tsc", p.get("LSDYNA_TSC", p.get("sigt_cutoff", 0.0))))
        damp = float(p.get("damp", p.get("LSD_MAT_DAMP", 0.10)))
        ncycle = int(p.get("ncycle", p.get("LSD_NCYCLE", 12)))
        srclmt = float(p.get("srclmt", p.get("LSD_SRCLMT", 1.0e20)))
        tab_id = int(p.get("tab_id", p.get("LSD_TID", p.get("tid", 0))))
        epsd_ref = float(p.get("epsd_ref", p.get("EPSD_REF", 0.0)))
        fscale = float(p.get("fscale", p.get("FSCALE", 1.0)))
        nrs = int(p.get("nrs", p.get("NRSFlag", 0)))
        tbl = p.get("table", p.get("yield_table", p.get("curve", p.get("yield_curve"))))
        return Law163Params(
            rho0=rho0,
            refer_rho=float(p.get("refer_rho", rho0)),
            e=e,
            nu=nu,
            tsc=tsc,
            damp=damp,
            ncycle=ncycle,
            srclmt=srclmt,
            tab_id=tab_id,
            epsd_ref=epsd_ref,
            fscale=fscale,
            nrs=nrs,
            table=tbl,
            title=getattr(mat, "title", ""),
        )

    if isinstance(mat, dict):
        p = mat.get("params", mat)
        rho0 = float(p.get("rho0", p.get("rho", p.get("density", p.get("MAT_RHO", 1.0)))))
        e = float(p.get("e", p.get("MAT_E", p.get("E", 0.0))))
        nu = float(p.get("nu", p.get("MAT_NU", p.get("Nu", p.get("pr", 0.0)))))
        tsc = float(p.get("tsc", p.get("LSDYNA_TSC", p.get("sigt_cutoff", 0.0))))
        damp = float(p.get("damp", p.get("LSD_MAT_DAMP", 0.10)))
        ncycle = int(p.get("ncycle", p.get("LSD_NCYCLE", 12)))
        srclmt = float(p.get("srclmt", p.get("LSD_SRCLMT", 1.0e20)))
        tab_id = int(p.get("tab_id", p.get("LSD_TID", p.get("tid", 0))))
        epsd_ref = float(p.get("epsd_ref", p.get("EPSD_REF", 0.0)))
        fscale = float(p.get("fscale", p.get("FSCALE", 1.0)))
        nrs = int(p.get("nrs", p.get("NRSFlag", 0)))
        tbl = p.get("table", p.get("yield_table", p.get("curve", p.get("yield_curve"))))
        return Law163Params(
            rho0=rho0,
            refer_rho=float(p.get("refer_rho", rho0)),
            e=e,
            nu=nu,
            tsc=tsc,
            damp=damp,
            ncycle=ncycle,
            srclmt=srclmt,
            tab_id=tab_id,
            epsd_ref=epsd_ref,
            fscale=fscale,
            nrs=nrs,
            table=tbl,
            title=str(p.get("title", "")),
        )

    # Fallback to duck-typing
    p = getattr(mat, "params", {})
    rho0 = float(getattr(mat, "rho0", getattr(mat, "rho", p.get("rho0", 1.0))))
    e = float(getattr(mat, "e", p.get("e", 0.0)))
    nu = float(getattr(mat, "nu", p.get("nu", 0.0)))
    tsc = float(getattr(mat, "tsc", p.get("tsc", 0.0)))
    damp = float(getattr(mat, "damp", p.get("damp", 0.10)))
    ncycle = int(getattr(mat, "ncycle", p.get("ncycle", 12)))
    srclmt = float(getattr(mat, "srclmt", p.get("srclmt", 1.0e20)))
    tab_id = int(getattr(mat, "tab_id", p.get("tab_id", 0)))
    epsd_ref = float(getattr(mat, "epsd_ref", p.get("epsd_ref", 0.0)))
    fscale = float(getattr(mat, "fscale", p.get("fscale", 1.0)))
    nrs = int(getattr(mat, "nrs", p.get("nrs", 0)))
    tbl = getattr(mat, "table", p.get("table", None))
    return Law163Params(
        rho0=rho0,
        refer_rho=rho0,
        e=e,
        nu=nu,
        tsc=tsc,
        damp=damp,
        ncycle=ncycle,
        srclmt=srclmt,
        tab_id=tab_id,
        epsd_ref=epsd_ref,
        fscale=fscale,
        nrs=nrs,
        table=tbl,
    )


def build_law163(mat_def: Any) -> Material:
    """Factory creating a Material object configured for LAW163.

    Supports dict, MatLaw163, Material, or Law163Params.
    """
    params_obj = _extract_params(mat_def)
    mat_id = 1
    title = params_obj.title
    if isinstance(mat_def, MatLaw163):
        mat_id = mat_def.id
    elif isinstance(mat_def, Material):
        mat_id = mat_def.id
        title = mat_def.title
    elif isinstance(mat_def, dict):
        mat_id = int(mat_def.get("id", mat_def.get("mat_id", 1)))
        title = str(mat_def.get("title", title))

    params_dict = asdict(params_obj)
    # Extra convenient aliases
    params_dict.update({
        "E": params_obj.e,
        "Nu": params_obj.nu,
        "nu": params_obj.nu,
        "rho": params_obj.rho0,
        "rho0": params_obj.rho0,
        "G": params_obj.g,
        "bulk": params_obj.bulk,
        "cii": params_obj.cii,
        "cij": params_obj.cij,
        "law163_params": params_obj,
    })

    return Material(
        id=mat_id,
        law=163,
        rho0=params_obj.rho0,
        title=title,
        params=params_dict,
    )


def resolve(mat: Material, model: Any, log: Any = None) -> None:
    """Resolve table / function curves for /MAT/LAW163 from model definitions."""
    p = mat.params
    tab_id = p.get("tab_id", 0)
    if tab_id and tab_id != 0:
        if hasattr(model, "tables") and tab_id in model.tables:
            p["table"] = model.tables[tab_id]
        elif hasattr(model, "functions") and tab_id in model.functions:
            p["table"] = model.functions[tab_id]
        elif log is not None and hasattr(log, "warning"):
            log.warning(f"/MAT/LAW163/{mat.id}: table/curve ID {tab_id} not found in model", "MAT CHECK")


def _lookup_yield(
    table: Any,
    gama: np.ndarray,
    dgamdt: np.ndarray,
    fscale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lookup yield stress and volumetric slope (TABLE_MAT_VINTERP equivalent).

    Parameters
    ----------
    table : Any
        Yield table, curve, callable, or None.
    gama : (n,) ndarray
        Clamped volumetric strain (max(gamma, 0)).
    dgamdt : (n,) ndarray
        Clamped volumetric strain rate (max(dgamdt, 0)).
    fscale : float
        Scale factor for yield stress.

    Returns
    -------
    sigy : (n,) ndarray
        Negative compressive yield stress: -abs(sigy_val) * fscale.
    dsdgam : (n,) ndarray
        Non-negative derivative with respect to volumetric strain: max(dsdgam * fscale, 0.0).
    """
    n = gama.shape[0]

    # 1. No table specified -> default high elastic yield limit
    if table is None:
        sigy_val = np.full(n, 1.0e20, dtype=gama.dtype)
        dsdgam_val = np.zeros(n, dtype=gama.dtype)
        return -np.abs(sigy_val) * fscale, np.maximum(dsdgam_val * fscale, 0.0)

    # 2. Constant scalar
    if isinstance(table, (int, float, np.floating, np.integer)):
        sigy_val = np.full(n, float(table), dtype=gama.dtype)
        dsdgam_val = np.zeros(n, dtype=gama.dtype)
        return -np.abs(sigy_val) * fscale, np.maximum(dsdgam_val * fscale, 0.0)

    # 3. FunctTable or object with x, y, slope attributes (1D curve)
    if isinstance(table, (FunctTable, SmoothFunctTable)) or (hasattr(table, "x") and hasattr(table, "y")):
        xs = np.asarray(table.x, dtype=float)
        ys = np.asarray(table.y, dtype=float)
        if hasattr(table, "slope"):
            slopes = np.asarray(table.slope, dtype=float)
        else:
            slopes = np.diff(ys) / np.maximum(np.diff(xs), _EM20)

        # Vectorized lookup with end-slope extrapolation
        idx = np.clip(np.searchsorted(xs, gama, side="right"), 1, len(xs) - 1)
        dsdgam_val = slopes[idx - 1]
        sigy_val = ys[idx - 1] + dsdgam_val * (gama - xs[idx - 1])
        return -np.abs(sigy_val) * fscale, np.maximum(dsdgam_val * fscale, 0.0)

    # 4. 2-tuple (x, y)
    if isinstance(table, (tuple, list)) and len(table) == 2 and isinstance(table[0], (np.ndarray, list, tuple)):
        xs = np.asarray(table[0], dtype=float)
        ys = np.asarray(table[1], dtype=float)
        slopes = np.diff(ys) / np.maximum(np.diff(xs), _EM20)
        idx = np.clip(np.searchsorted(xs, gama, side="right"), 1, len(xs) - 1)
        dsdgam_val = slopes[idx - 1]
        sigy_val = ys[idx - 1] + dsdgam_val * (gama - xs[idx - 1])
        return -np.abs(sigy_val) * fscale, np.maximum(dsdgam_val * fscale, 0.0)

    # 5. 3-tuple (xg, rates, Y) -> 2D table lookup (strain x strain rate)
    if isinstance(table, (tuple, list)) and len(table) == 3:
        xg = np.asarray(table[0], dtype=float)
        rates = np.asarray(table[1], dtype=float)
        Y = np.asarray(table[2], dtype=float)  # shape (len(xg), len(rates))

        idx_x = np.clip(np.searchsorted(xg, gama, side="right"), 1, len(xg) - 1)
        dx = np.maximum(xg[idx_x] - xg[idx_x - 1], _EM20)
        t = (gama - xg[idx_x - 1]) / dx

        if len(rates) == 1 or Y.shape[1] == 1:
            y0 = Y[idx_x - 1, 0]
            y1 = Y[idx_x, 0]
            dsdgam_val = (y1 - y0) / dx
            sigy_val = y0 + t * (y1 - y0)
        else:
            idx_r = np.clip(np.searchsorted(rates, dgamdt, side="right"), 1, len(rates) - 1)
            dr = np.maximum(rates[idx_r] - rates[idx_r - 1], _EM20)
            u = (dgamdt - rates[idx_r - 1]) / dr

            # 4 knot points
            y00 = Y[idx_x - 1, idx_r - 1]
            y10 = Y[idx_x, idx_r - 1]
            y01 = Y[idx_x - 1, idx_r]
            y11 = Y[idx_x, idx_r]

            slope_r0 = (y10 - y00) / dx
            slope_r1 = (y11 - y01) / dx
            dsdgam_val = slope_r0 + u * (slope_r1 - slope_r0)

            val_r0 = y00 + t * (y10 - y00)
            val_r1 = y01 + t * (y11 - y01)
            sigy_val = val_r0 + u * (val_r1 - val_r0)

        return -np.abs(sigy_val) * fscale, np.maximum(dsdgam_val * fscale, 0.0)

    # 6. Dict with table keys (e.g. {'xg': ..., 'rates': ..., 'Y': ...})
    if isinstance(table, dict):
        if "xg" in table and "rates" in table and "Y" in table:
            return _lookup_yield((table["xg"], table["rates"], table["Y"]), gama, dgamdt, fscale=fscale)
        if "x" in table and "y" in table:
            return _lookup_yield((table["x"], table["y"]), gama, dgamdt, fscale=fscale)

    # 7. Callable function: f(gama, dgamdt) or f(gama)
    if callable(table):
        try:
            res = table(gama, dgamdt)
        except TypeError:
            res = table(gama)
        if isinstance(res, tuple) and len(res) == 2:
            sigy_val = np.asarray(res[0], dtype=gama.dtype)
            dsdgam_val = np.asarray(res[1], dtype=gama.dtype)
        else:
            sigy_val = np.asarray(res, dtype=gama.dtype)
            dsdgam_val = np.zeros(n, dtype=gama.dtype)
        return -np.abs(sigy_val) * fscale, np.maximum(dsdgam_val * fscale, 0.0)

    # Fallback
    sigy_val = np.full(n, 1.0e20, dtype=gama.dtype)
    dsdgam_val = np.zeros(n, dtype=gama.dtype)
    return -np.abs(sigy_val) * fscale, np.maximum(dsdgam_val * fscale, 0.0)


def solid_update(
    mat: Any = None,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    return_tuple: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Any:
    r"""Vectorized constitutive update for /MAT/LAW163 Crushable Foam solid elements.

    Function: SIGEPS_163 (lines 53-309) in
    $OR_SRC/engine/source/materials/mat/mat163/sigeps163.F90

    Parameters
    ----------
    mat : Material, Law163Params, MatLaw163, or dict
        Material model definition.
    sig : (n, 6) or (6,) ndarray
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx].
    deps : (n, 6) or (6,) ndarray
        Strain increment tensor (engineering shear).
    epsp : (n,) or scalar ndarray, optional
        Accumulated effective volumetric plastic strain (plas = log(rho0/rho)).
    dt : float, default 0.0
        Time step increment.
    extra : dict, optional
        Per-element state arrays.
    return_tuple : bool, default False
        If True, returns (sign, epsp, c). If False, returns sign.

    Returns
    -------
    (sig_tot, epsp, c) or sig_tot
    """
    if len(args) >= 2 or "fint" in kwargs or (mat is not None and hasattr(mat, "mat") and sig is None):
        # Template call: solid_update(group, x, u, ur, dt, fint, mint)
        return None

    if sig is None or deps is None:
        return None

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, 6)
        deps = deps.reshape(1, 6)
        if epsp is not None and np.ndim(epsp) == 0:
            epsp = np.array([epsp])

    n = sig.shape[0]
    p = _extract_params(mat)

    rho0 = float(p.rho0 if p.rho0 > 0 else 1.0)
    nu = p.nu
    g = p.g
    bulk = p.bulk
    cii = p.cii
    cij = p.cij
    tsc = float(p.tsc)
    damp = float(p.damp)
    ncycle = int(p.ncycle)
    srclmt = float(p.srclmt)
    nrs = int(p.nrs)
    fscale = float(p.fscale)
    epsd_ref = float(p.epsd_ref)

    # 1. Density & element length determination
    if extra is not None and "rho" in extra:
        rho_curr = np.atleast_1d(np.asarray(extra["rho"], dtype=sig.dtype))
    elif extra is not None and "amu" in extra:
        amu = np.atleast_1d(np.asarray(extra["amu"], dtype=sig.dtype))
        rho_curr = rho0 * (1.0 + amu)
    elif extra is not None and "mu" in extra:
        mu = np.atleast_1d(np.asarray(extra["mu"], dtype=sig.dtype))
        rho_curr = rho0 * (1.0 + mu)
    else:
        rho_curr = np.full(n, rho0, dtype=sig.dtype)

    # Element characteristic length le (sigeps163.F90 lines 156-161)
    if extra is not None and "le" in extra:
        le = np.atleast_1d(np.asarray(extra["le"], dtype=sig.dtype))
    elif extra is not None and "aldt" in extra:
        le = np.atleast_1d(np.asarray(extra["aldt"], dtype=sig.dtype))
    elif extra is not None and "deltax" in extra:
        le = np.atleast_1d(np.asarray(extra["deltax"], dtype=sig.dtype))
    elif extra is not None and "lc" in extra:
        le = np.atleast_1d(np.asarray(extra["lc"], dtype=sig.dtype))
    elif extra is not None and "char_len" in extra:
        le = np.atleast_1d(np.asarray(extra["char_len"], dtype=sig.dtype))
    else:
        le = np.ones(n, dtype=sig.dtype)
    if le.size == 1 and n > 1:
        le = np.full(n, float(le[0]), dtype=sig.dtype)

    # History: uvar1 = old volumetric strain, uvar2 = initial length
    uvar = None
    if extra is not None:
        if "uvar163" in extra:
            uvar = extra["uvar163"]
        elif "uvar" in extra:
            uvar = extra["uvar"]
        elif "uvar1" in extra:
            uvar = np.zeros((n, 2), dtype=sig.dtype)
            uvar[:, 0] = np.atleast_1d(np.asarray(extra["uvar1"], dtype=sig.dtype))
            uvar[:, 1] = le

    if uvar is None:
        uvar = np.zeros((n, 2), dtype=sig.dtype)
        uvar[:, 1] = le
        if extra is not None:
            extra["uvar163"] = uvar
    elif uvar.ndim == 1:
        uvar = uvar.reshape(1, -1)

    if np.all(uvar[:, 1] == 0.0):
        uvar[:, 1] = le
    le = uvar[:, 1]
    uvar1_old = uvar[:, 0].copy()

    # History: old filtered volumetric strain rate epsd
    if extra is not None and "epsd163" in extra:
        epsd_old = np.atleast_1d(np.asarray(extra["epsd163"], dtype=sig.dtype))
    elif extra is not None and "epsd" in extra:
        epsd_old = np.atleast_1d(np.asarray(extra["epsd"], dtype=sig.dtype))
    else:
        epsd_old = np.full(n, epsd_ref, dtype=sig.dtype)
    if epsd_old.size == 1 and n > 1:
        epsd_old = np.full(n, float(epsd_old[0]), dtype=sig.dtype)

    # Optional explicit strain rate tensor (epsp) from extra
    rates = None
    if extra is not None:
        if "rates" in extra and extra["rates"] is not None:
            rates = np.atleast_2d(np.asarray(extra["rates"], dtype=sig.dtype))
        elif "strain_rate" in extra and extra["strain_rate"] is not None:
            rates = np.atleast_2d(np.asarray(extra["strain_rate"], dtype=sig.dtype))
        elif "epsp" in extra and extra["epsp"] is not None:
            arr = np.asarray(extra["epsp"], dtype=sig.dtype)
            if arr.ndim == 2 and arr.shape[-1] == 6:
                rates = arr

    # =========================================================================
    # 1. Computation of trial stress tensor (sigeps163.F90 lines 169-176)
    # =========================================================================
    st_xx = sig[:, 0] + cii * deps[:, 0] + cij * deps[:, 1] + cij * deps[:, 2]
    st_yy = sig[:, 1] + cij * deps[:, 0] + cii * deps[:, 1] + cij * deps[:, 2]
    st_zz = sig[:, 2] + cij * deps[:, 0] + cij * deps[:, 1] + cii * deps[:, 2]
    st_xy = sig[:, 3] + g * deps[:, 3]
    st_yz = sig[:, 4] + g * deps[:, 4]
    st_zx = sig[:, 5] + g * deps[:, 5]

    # =========================================================================
    # 2. Principal stresses and directions (sigeps163.F90 lines 178-188)
    # =========================================================================
    S_trial = np.zeros((n, 3, 3), dtype=sig.dtype)
    S_trial[:, 0, 0] = st_xx
    S_trial[:, 1, 1] = st_yy
    S_trial[:, 2, 2] = st_zz
    S_trial[:, 0, 1] = st_xy
    S_trial[:, 1, 0] = st_xy
    S_trial[:, 1, 2] = st_yz
    S_trial[:, 2, 1] = st_yz
    S_trial[:, 2, 0] = st_zx
    S_trial[:, 0, 2] = st_zx

    sigp, dirp = np.linalg.eigh(S_trial)  # sigp: (n, 3), dirp: (n, 3, 3)

    # =========================================================================
    # 3. Volumetric strain & strain rate filtering (sigeps163.F90 lines 193-214)
    # =========================================================================
    rho_safe = np.maximum(rho_curr, _EM20)
    gama = 1.0 - rho0 / rho_safe

    dt_safe = max(float(dt), _EM20)
    if nrs == 1:
        # Engineering strain rate
        dgamdt = (gama - uvar1_old) / dt_safe
    else:
        # True strain rate
        if rates is not None:
            dgamdt = -(rates[:, 0] + rates[:, 1] + rates[:, 2])
        else:
            dgamdt = -(deps[:, 0] + deps[:, 1] + deps[:, 2]) / dt_safe

    # Volumetric strain rate filtering
    alpha = _TWO_PI / (_TWO_PI + float(ncycle))
    dgamdt = alpha * dgamdt + (1.0 - alpha) * epsd_old

    # Cap the change of volumetric strain rate
    if dt > 0.0 and srclmt > 0.0:
        max_change = srclmt * dt
        diff = dgamdt - epsd_old
        exceed = np.abs(diff) > max_change
        if np.any(exceed):
            dgamdt = np.where(exceed, epsd_old + np.sign(diff) * max_change, dgamdt)

    # Effective volumetric true strain & strain rate
    plas = np.log(np.maximum(rho0 / rho_safe, _EM20))
    epsd_new = dgamdt.copy()

    # Update history in uvar
    uvar[:, 0] = gama
    if extra is not None:
        extra["epsd163"] = epsd_new
        extra["epsd"] = epsd_new
        extra["uvar163"] = uvar
        extra["uvar"] = uvar
        extra["uvar1"] = uvar[:, 0]
        extra["plas"] = plas

    # =========================================================================
    # 4. Yield stress computation (sigeps163.F90 lines 219-228)
    # =========================================================================
    gama_pos = np.maximum(gama, 0.0)
    dgamdt_pos = np.maximum(dgamdt, 0.0)

    tbl = extra.get("table", p.table) if extra is not None else p.table
    sigy, dsdgam = _lookup_yield(tbl, gama_pos, dgamdt_pos, fscale=fscale)

    # =========================================================================
    # 5. Stress scaling procedure (sigeps163.F90 lines 232-240)
    # =========================================================================
    sigp_scaled = np.empty_like(sigp)
    for j in range(3):
        sp = sigp[:, j]
        sigp_scaled[:, j] = np.where(sp < sigy, sigy, np.where(sp > tsc, tsc, sp))

    # =========================================================================
    # 6. Reconstruct Cauchy stress (sigeps163.F90 lines 242-260)
    # =========================================================================
    S_new = np.einsum("nik,nk,njk->nij", dirp, sigp_scaled, dirp)
    sign_xx = S_new[:, 0, 0]
    sign_yy = S_new[:, 1, 1]
    sign_zz = S_new[:, 2, 2]
    sign_xy = S_new[:, 0, 1]
    sign_yz = S_new[:, 1, 2]
    sign_zx = S_new[:, 2, 0]

    # =========================================================================
    # 7. Viscous damping (sigeps163.F90 lines 279-307)
    # =========================================================================
    # Fortran sigeps163.F90 lines 267-271: sound speed denominator is rho0 if rho <= em20
    rho_ssp = np.where(rho_curr > _EM20, rho_curr, rho0)
    denom = np.copysign(np.maximum(np.abs(1.0 + gama), _EM20), 1.0 + gama)
    mod_eff = np.maximum(bulk, dsdgam) + _FOUR_THIRD * g
    ssp0 = np.sqrt(mod_eff / rho_ssp)
    a = ssp0 * rho_safe * damp * le / denom

    if rates is not None:
        epsp_xx = rates[:, 0]
        epsp_yy = rates[:, 1]
        epsp_zz = rates[:, 2]
        epsp_xy = rates[:, 3]
        epsp_yz = rates[:, 4]
        epsp_zx = rates[:, 5]
        ldav = (epsp_xx + epsp_yy + epsp_zz) / 3.0

        one_p_nu = 1.0 + nu
        one_m_2nu = 1.0 - 2.0 * nu
        sigv_xx = a * ((epsp_xx - ldav) / one_p_nu + ldav / one_m_2nu)
        sigv_yy = a * ((epsp_yy - ldav) / one_p_nu + ldav / one_m_2nu)
        sigv_zz = a * ((epsp_zz - ldav) / one_p_nu + ldav / one_m_2nu)
        sigv_xy = a * epsp_xy / (2.0 * one_p_nu)
        sigv_yz = a * epsp_yz / (2.0 * one_p_nu)
        sigv_zx = a * epsp_zx / (2.0 * one_p_nu)
    elif dt > 0.0:
        epsp_xx = deps[:, 0] / dt
        epsp_yy = deps[:, 1] / dt
        epsp_zz = deps[:, 2] / dt
        epsp_xy = deps[:, 3] / dt
        epsp_yz = deps[:, 4] / dt
        epsp_zx = deps[:, 5] / dt
        ldav = (epsp_xx + epsp_yy + epsp_zz) / 3.0

        one_p_nu = 1.0 + nu
        one_m_2nu = 1.0 - 2.0 * nu
        sigv_xx = a * ((epsp_xx - ldav) / one_p_nu + ldav / one_m_2nu)
        sigv_yy = a * ((epsp_yy - ldav) / one_p_nu + ldav / one_m_2nu)
        sigv_zz = a * ((epsp_zz - ldav) / one_p_nu + ldav / one_m_2nu)
        sigv_xy = a * epsp_xy / (2.0 * one_p_nu)
        sigv_yz = a * epsp_yz / (2.0 * one_p_nu)
        sigv_zx = a * epsp_zx / (2.0 * one_p_nu)
    else:
        sigv_xx = np.zeros(n, dtype=sig.dtype)
        sigv_yy = np.zeros(n, dtype=sig.dtype)
        sigv_zz = np.zeros(n, dtype=sig.dtype)
        sigv_xy = np.zeros(n, dtype=sig.dtype)
        sigv_yz = np.zeros(n, dtype=sig.dtype)
        sigv_zx = np.zeros(n, dtype=sig.dtype)

    if extra is not None:
        extra["sigv"] = np.column_stack([sigv_xx, sigv_yy, sigv_zz, sigv_xy, sigv_yz, sigv_zx])
        extra["a"] = a
        extra["dsdgam"] = dsdgam

    # Total stress tensor = reconstructed inviscid stress + viscous stress
    sig_tot = np.column_stack([
        sign_xx + sigv_xx,
        sign_yy + sigv_yy,
        sign_zz + sigv_zz,
        sign_xy + sigv_xy,
        sign_yz + sigv_yz,
        sign_zx + sigv_zx,
    ])

    # =========================================================================
    # 8. Sound speed computation (sigeps163.F90 lines 292-307)
    # =========================================================================
    if dt > 0.0:
        c = np.sqrt((mod_eff + np.abs(a) / dt_safe) / rho_ssp)
    else:
        c = np.sqrt(mod_eff / rho_ssp)

    # Return processing
    epsp_out = plas
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = epsp_out
        except Exception:
            pass

    if is_1d:
        sig_tot = sig_tot[0]
        epsp_out = epsp_out[0]
        c = float(c[0])

    if return_tuple:
        return sig_tot, epsp_out, c
    return sig_tot


def sound_speed_solid(mat: Any, rho: Any = None, extra: dict | None = None, dt: float = 0.0, **kwargs) -> Any:
    """Sound speed for LAW163 solid elements.

    Matches ``sigeps163.F90`` lines 267-271, 292-306.
    """
    p = _extract_params(mat)
    rho0 = float(p.rho0 if p.rho0 > 0 else 1.0)
    bulk = p.bulk
    g = p.g

    dsdgam = float(kwargs.get("slope", kwargs.get("dsdgam", 0.0)))
    if extra is not None and "dsdgam" in extra:
        dsdgam = max(dsdgam, float(np.max(extra["dsdgam"])))

    mod_base = max(bulk, dsdgam) + _FOUR_THIRD * g

    if rho is not None:
        r = np.asarray(rho, dtype=float)
        r_eff = np.where(r > _EM20, r, rho0)
        mod_eff = np.full_like(r_eff, mod_base)

        if dt > 0.0 and extra is not None:
            if "a" in extra and extra["a"] is not None:
                a_val = np.asarray(extra["a"], dtype=float)
                mod_eff = mod_eff + np.abs(a_val) / max(float(dt), _EM20)
            elif "damp" in extra or p.damp > 0.0:
                damp = float(extra.get("damp", p.damp))
                le = np.asarray(extra.get("le", extra.get("aldt", 1.0)), dtype=float)
                gama = np.asarray(extra.get("gama", 0.0), dtype=float)
                ssp0 = np.sqrt(np.maximum(mod_base, 0.0) / r_eff)
                denom = np.copysign(np.maximum(np.abs(1.0 + gama), _EM20), 1.0 + gama)
                r_safe = np.maximum(r, _EM20)
                a_val = ssp0 * r_safe * damp * le / denom
                mod_eff = mod_eff + np.abs(a_val) / max(float(dt), _EM20)

        c = np.sqrt(np.maximum(mod_eff, 0.0) / r_eff)
        return float(c) if np.ndim(rho) == 0 else c

    if dt > 0.0 and extra is not None and "a" in extra and extra["a"] is not None:
        a_val = float(np.max(np.abs(extra["a"])))
        mod_base = mod_base + a_val / max(float(dt), _EM20)

    return math.sqrt(max(mod_base, 0.0) / max(rho0, _EM20))


def consistent_solid_tangent(
    mat: Any,
    sig: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    epsp_incr: np.ndarray | None = None,
    extra: dict | None = None,
) -> np.ndarray:
    """Algorithmic consistent tangent stiffness tensor (n, 6, 6) for LAW163 solid elements.

    For the elastic predictor regime, the tangent is the anisotropic stiffness matrix
    formed by cii, cij, and g.
    """
    if sig is not None and hasattr(sig, "shape") and sig.ndim > 1:
        n = sig.shape[0]
    else:
        n = 1

    p = _extract_params(mat)
    cii = p.cii
    cij = p.cij
    g = p.g

    dtype = sig.dtype if (sig is not None and hasattr(sig, "dtype")) else np.float64
    D = np.zeros((n, 6, 6), dtype=dtype)
    D[:, 0, 0] = cii
    D[:, 1, 1] = cii
    D[:, 2, 2] = cii

    D[:, 0, 1] = cij
    D[:, 0, 2] = cij
    D[:, 1, 0] = cij
    D[:, 1, 2] = cij
    D[:, 2, 0] = cij
    D[:, 2, 1] = cij

    D[:, 3, 3] = g
    D[:, 4, 4] = g
    D[:, 5, 5] = g

    return D


def extra_shapes(mat: Any = None, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Per-element persistent state required by LAW163."""
    return {
        "uvar163": (2,),
        "epsd163": (),
        "sigv": (6,),
    }


def shell_update(*args: Any, **kwargs: Any) -> None:
    """LAW163 is defined for solid elements only."""
    raise NotImplementedError("LAW163 (/MAT/CRUSHABLE_FOAM) is implemented for solid elements only.")


def tangent(*args: Any, **kwargs: Any) -> Any:
    r"""Consistent solid tangent stiffness for LAW163 per template.

    Function: SIGEPS_163 (lines 53-309) in
    $OR_SRC/engine/source/materials/mat/mat163/sigeps163.F90
    """
    if len(args) == 1 and hasattr(args[0], "mat"):
        # Template call: tangent(group)
        return None
    mat = args[0] if args else kwargs.get("mat")
    return consistent_solid_tangent(mat, *args[1:], **kwargs)


# Aliases for consistent naming
solid_tangent = consistent_solid_tangent
tangent_law163_solid = consistent_solid_tangent
solid_update_law163 = solid_update
shell_update_law163 = shell_update
sound_speed = sound_speed_solid
sound_speed_solid_law163 = sound_speed_solid


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW163 uses an incremental hypoelastic rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Persistent history variables for LAW163 (sigeps163.F90 lines 62, 70-75)."""
    return {
        "uvar163": (nip, 2) if nip is not None else (2,),
        "epsd163": (nip,) if nip is not None else (),
        "sigv": (nip, 6) if nip is not None else (6,),
    }
