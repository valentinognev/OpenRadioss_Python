# pyradioss/engine/ale_turbulence.py
"""
k-epsilon Turbulence Model for ALE / Eulerian Flows.

Ported from OpenRadioss Fortran sources:
- engine/source/ale/turbulence/akturb.F: Turbulent kinetic energy transport & eddy viscosity
- engine/source/ale/turbulence/aeturb.F: Dissipation transport & turbulent Prandtl scaling
- engine/source/ale/turbulence/aturbn.F: Source terms (production, dissipation, compression, SGS floor)
- engine/source/ale/ale2d/adiff2.F & engine/source/ale/ale3d/adiff3.F: Harmonic diffusion
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple, Dict, Set, List, Any


# Standard Launder-Sharma / Jones-Launder k-epsilon empirical constants
# Fortran PM(81..86, MAT) defaults in OpenRadioss:
DEFAULT_CMU = 0.09       # PM(81, MAT): turbulent viscosity coefficient C_mu
DEFAULT_C1 = 1.44        # PM(82, MAT): production coefficient C_eps1
DEFAULT_C2 = 1.92        # PM(83, MAT): destruction coefficient C_eps2
DEFAULT_C3 = -0.33       # PM(84, MAT): compressibility coefficient C_eps3
DEFAULT_SIGMA_K = 1.0    # PM(85, MAT): turbulent Prandtl number for k, sigma_k
DEFAULT_SIGMA_E = 1.3    # PM(86, MAT): turbulent Prandtl number for eps, sigma_eps


def compute_eddy_viscosity(k: np.ndarray,
                           eps: np.ndarray,
                           rho: np.ndarray,
                           c_mu: float = DEFAULT_CMU) -> np.ndarray:
    """Compute turbulent eddy viscosity mu_t.

    Ported from $OR_SRC/engine/source/ale/turbulence/akturb.F
    lines 109-111:
      RK = GBUF%RK(I)   ! rho * k
      RE = GBUF%RE(I)   ! rho * eps
      XMT = PM(81, MT) * RK * RK / max(EM15, RE)

    Formula:
      mu_t = C_mu * (rho * k)^2 / max(1e-15, rho * eps) = C_mu * rho * k^2 / eps

    Args:
        k: (n_elem,) specific turbulent kinetic energy (J/kg or m^2/s^2).
        eps: (n_elem,) specific dissipation rate (W/kg or m^2/s^3).
        rho: (n_elem,) fluid density (kg/m^3).
        c_mu: empirical constant (default 0.09).

    Returns:
        mu_t: (n_elem,) turbulent dynamic eddy viscosity (Pa.s).
    """
    rk = np.maximum(rho * k, 0.0)
    re = np.maximum(rho * eps, 1e-15)
    mu_t = c_mu * (rk * rk) / re
    return np.maximum(mu_t, 0.0)


def compute_turbulent_pressure(k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Compute isotropic turbulent pressure P_turb = 2/3 * rho * k.

    Ported from $OR_SRC/engine/source/ale/turbulence/aturbn.F
    line 64:
      Pturb(I) = TWO * RK(I) / (VNEW(I) * THREE)

    Args:
        k: (n_elem,) specific turbulent kinetic energy.
        rho: (n_elem,) fluid density.

    Returns:
        p_turb: (n_elem,) isotropic turbulent pressure (Pa).
    """
    return (2.0 / 3.0) * rho * np.maximum(k, 0.0)


def update_turbulence_sources(k: np.ndarray,
                              eps: np.ndarray,
                              rho: np.ndarray,
                              vol: np.ndarray,
                              dvol: np.ndarray,
                              e_inc: np.ndarray,
                              mu_lam: np.ndarray,
                              dt: float,
                              c_mu: float = DEFAULT_CMU,
                              c1: float = DEFAULT_C1,
                              c2: float = DEFAULT_C2,
                              c3: float = DEFAULT_C3,
                              sigma_e: float = DEFAULT_SIGMA_E,
                              sgsl: Optional[np.ndarray] = None,
                              off: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Advance turbulent kinetic energy k and dissipation eps with production and destruction.

    Ported from $OR_SRC/engine/source/ale/turbulence/aturbn.F
    lines 60-98:
      XFAC = TMU(I) / VIS(I)
      Pturb(I) = 2/3 * RK(I) / VNEW(I)
      EI = max(0, XFAC * EINC(I) - DVOL(I) * Pturb(I))
      RK_new = RK + (EI - RE * DT1) * OFF
      C1 = PM(82, MAT) / max(RK, EM15)
      C2 = PM(83, MAT) / max(RK, EM15)
      C3 = PM(84, MAT)
      RE_new = RE * (1 + (C1*EI - C2*RE*DT1 + C3*DVOL/VNEW) * OFF)
      Subgrid scale (SGS) floor:
      FAC = sqrt(CMU / (SE * (C2 - C1))) / SGSL
      RESGS = RHO * FAC * (RK / RHO)**(1.5)
      RE_new = max(RE_new, RESGS)

    Energy Accounting:
      The turbulent dissipation rate converts mechanical turbulent kinetic energy
      into thermodynamic internal energy:
        Q_diss = sum(rho * eps * vol * dt)
      This dissipated thermal work is returned for booking into the engine energy ledger.

    Args:
        k: (n_elem,) specific turbulent kinetic energy.
        eps: (n_elem,) specific dissipation rate.
        rho: (n_elem,) fluid density.
        vol: (n_elem,) element volume.
        dvol: (n_elem,) element volume change over time step.
        e_inc: (n_elem,) laminar viscous internal energy increment.
        mu_lam: (n_elem,) laminar dynamic viscosity.
        dt: explicit time step duration.
        c_mu, c1, c2, c3, sigma_e: model empirical constants.
        sgsl: optional (n_elem,) subgrid length scale filter width (PM(46, MAT)).
        off: element active flag (1.0 = active).

    Returns:
        k_new: (n_elem,) updated specific turbulent kinetic energy.
        eps_new: (n_elem,) updated specific dissipation rate.
        dissipated_energy: float, total energy dissipated to heat (Joules).
    """
    n_elem = len(k)
    if n_elem == 0:
        return k.copy(), eps.copy(), 0.0

    mu_t = compute_eddy_viscosity(k, eps, rho, c_mu=c_mu)
    xfac = mu_t / np.maximum(mu_lam, 1e-20)
    p_turb = compute_turbulent_pressure(k, rho)

    # Volumetric extensive fields
    rk_old = rho * np.maximum(k, 1e-15) * vol
    re_old = rho * np.maximum(eps, 1e-15) * vol

    # Shear production and compression work
    ei = xfac * e_inc - dvol * p_turb
    ei = np.maximum(0.0, ei)

    # k update (aturbn.F line 67)
    rk_new = rk_old + (ei - re_old * dt) * off
    ark = np.maximum(rk_new, 1e-15)

    # eps update (aturbn.F lines 69-74)
    c1_local = c1 / ark
    c2_local = c2 / ark
    vol_safe = np.maximum(vol, 1e-30)
    eps_factor = 1.0 + (c1_local * ei - c2_local * re_old * dt + c3 * dvol / vol_safe) * off
    eps_factor = np.maximum(eps_factor, 0.01)  # prevent numerical collapse
    re_new = re_old * eps_factor

    # Convert back to specific values
    mass_elem = np.maximum(rho * vol, 1e-30)
    k_res = np.maximum(1e-15, rk_new / mass_elem)
    eps_res = np.maximum(1e-15, re_new / mass_elem)

    # Subgrid Scale (SGS) / Pope floor limiter (aturbn.F lines 77-86)
    if sgsl is not None:
        denom = sigma_e * max(1e-6, c2 - c1)
        fac = np.sqrt(c_mu / denom) / np.maximum(sgsl, 1e-15)
        eps_sgs = fac * np.maximum(k_res, 0.0)**1.5
        eps_res = np.maximum(eps_res, eps_sgs)

    # Energy ledger booking: dissipated turbulent kinetic energy converts to heat
    dissipated_energy = float(np.sum(rho * eps_res * vol * dt))

    return k_res, eps_res, dissipated_energy


def diffuse_turbulence_fields(phi_field: np.ndarray,
                              diffusivity: np.ndarray,
                              volumes: np.ndarray,
                              geom_grad: np.ndarray,
                              neighbor_elem: np.ndarray,
                              dt: float) -> np.ndarray:
    """Diffuse a turbulent scalar field (k or eps) using harmonic finite-volume diffusion.

    Ported from $OR_SRC/engine/source/ale/turbulence/akturb.F
    lines 111-122, aeturb.F lines 109-120, and engine/source/ale/ale3d/adiff3.F:
      Harmonic face interpolation of diffusivity:
        AA_face_k = (alpha_e * alpha_nbr) / max(1e-20, alpha_e + alpha_nbr)
      Time advance:
        DPHI_e = sum_k AA_face_k * (phi_nbr - phi_e) * GRAD(e, k)
        phi_new_e = phi_e + 2 * DPHI_e * dt / max(VOL_e, 1e-20)

    Args:
        phi_field: (n_elem,) scalar field (k or eps).
        diffusivity: (n_elem,) turbulent diffusion coefficient (mu_t / sigma_k or mu_t / sigma_eps).
        volumes: (n_elem,) element volumes.
        geom_grad: (n_elem, n_faces) geometric gradient factors from agrad/egrad.
        neighbor_elem: (n_elem, n_faces) neighbor element connectivity.
        dt: explicit time step duration.

    Returns:
        phi_new: (n_elem,) diffused field.
    """
    n_elem, n_faces = geom_grad.shape
    dphi = np.zeros(n_elem, dtype=np.float64)

    for e in range(n_elem):
        ae = diffusivity[e]
        qe = phi_field[e]
        for k in range(n_faces):
            nbr = neighbor_elem[e, k]
            if nbr >= 0:
                anbr = diffusivity[nbr]
                qnbr = phi_field[nbr]
            else:
                anbr = ae
                qnbr = qe

            # Harmonic mean across face (adiff.F)
            aa_face = (ae * anbr) / max(1e-20, ae + anbr)
            dphi[e] += aa_face * (qnbr - qe) * geom_grad[e, k]

    delta = 2.0 * dphi * dt / np.maximum(volumes, 1e-20)
    phi_new = np.maximum(1e-15, phi_field + delta)
    return phi_new


class KEpsilonTurbulenceModel:
    """k-epsilon Turbulence Model Orchestrator.

    Integrates:
      1. Eddy viscosity computation (akturb.F)
      2. Production, destruction, and compression source terms (aturbn.F)
      3. Turbulent diffusion of k and eps (akturb.F, aeturb.F, adiff2/3.F)
      4. Subgrid length scale limiter (aturbn.F)
      5. Strict thermodynamic energy accounting for dissipated turbulent work
    """

    def __init__(self,
                 n_elem: int,
                 c_mu: float = DEFAULT_CMU,
                 c1: float = DEFAULT_C1,
                 c2: float = DEFAULT_C2,
                 c3: float = DEFAULT_C3,
                 sigma_k: float = DEFAULT_SIGMA_K,
                 sigma_e: float = DEFAULT_SIGMA_E,
                 sgsl: Optional[Union[float, np.ndarray]] = None) -> None:
        """Initialize k-epsilon turbulence model.

        Args:
            n_elem: number of elements in the fluid domain.
            c_mu: eddy viscosity constant (default 0.09).
            c1: production constant (default 1.44).
            c2: destruction constant (default 1.92).
            c3: compression constant (default -0.33).
            sigma_k: turbulent Prandtl number for k (default 1.0).
            sigma_e: turbulent Prandtl number for eps (default 1.3).
            sgsl: optional subgrid filter length scale.
        """
        self.n_elem = n_elem
        self.c_mu = c_mu
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.sigma_k = sigma_k
        self.sigma_e = sigma_e

        if sgsl is not None:
            if np.isscalar(sgsl):
                self.sgsl = np.full(n_elem, float(sgsl), dtype=np.float64)
            else:
                self.sgsl = np.asarray(sgsl, dtype=np.float64)
        else:
            self.sgsl = None

        # State fields
        self.k = np.full(n_elem, 1e-4, dtype=np.float64)    # specific turbulent kinetic energy
        self.eps = np.full(n_elem, 1e-4, dtype=np.float64)  # specific dissipation rate
        self.mu_t = np.zeros(n_elem, dtype=np.float64)      # eddy viscosity

        # Energy tracking
        self.cumulative_dissipated_energy: float = 0.0

    def compute_mu_t(self, rho: np.ndarray) -> np.ndarray:
        """Compute turbulent dynamic eddy viscosity for current state."""
        self.mu_t = compute_eddy_viscosity(self.k, self.eps, rho, c_mu=self.c_mu)
        return self.mu_t

    def step(self,
             rho: np.ndarray,
             vol: np.ndarray,
             dvol: np.ndarray,
             e_inc: np.ndarray,
             mu_lam: np.ndarray,
             dt: float,
             geom_grad: Optional[np.ndarray] = None,
             neighbor_elem: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform a complete turbulence step (sources + diffusion + energy accounting).

        Args:
            rho: (n_elem,) fluid density.
            vol: (n_elem,) element volume.
            dvol: (n_elem,) element volume change.
            e_inc: (n_elem,) viscous strain increment work.
            mu_lam: (n_elem,) laminar dynamic viscosity.
            dt: time step duration.
            geom_grad: optional (n_elem, n_faces) geometric gradient factors.
            neighbor_elem: optional (n_elem, n_faces) neighbor connectivity.

        Returns:
            dict with updated 'k', 'eps', 'mu_t', 'p_turb', and 'dissipated_energy'.
        """
        # 1. Update source terms (production, destruction, SGS floor)
        k_src, eps_src, diss_energy = update_turbulence_sources(
            k=self.k,
            eps=self.eps,
            rho=rho,
            vol=vol,
            dvol=dvol,
            e_inc=e_inc,
            mu_lam=mu_lam,
            dt=dt,
            c_mu=self.c_mu,
            c1=self.c1,
            c2=self.c2,
            c3=self.c3,
            sigma_e=self.sigma_e,
            sgsl=self.sgsl,
        )
        self.k[:] = k_src
        self.eps[:] = eps_src
        self.cumulative_dissipated_energy += diss_energy

        # 2. Diffusive transport (if connectivity and gradients are provided)
        mu_t = self.compute_mu_t(rho)
        if geom_grad is not None and neighbor_elem is not None:
            # Diffuse k with diffusivity = mu_t / sigma_k (akturb.F lines 110-123)
            alpha_k = mu_t / self.sigma_k
            self.k[:] = diffuse_turbulence_fields(
                self.k, alpha_k, vol, geom_grad, neighbor_elem, dt
            )

            # Diffuse eps with diffusivity = mu_t / sigma_e (aeturb.F lines 109-120)
            alpha_e = mu_t / self.sigma_e
            self.eps[:] = diffuse_turbulence_fields(
                self.eps, alpha_e, vol, geom_grad, neighbor_elem, dt
            )

            # Recompute mu_t after diffusion
            mu_t = self.compute_mu_t(rho)

        p_turb = compute_turbulent_pressure(self.k, rho)

        return {
            "k": self.k,
            "eps": self.eps,
            "mu_t": mu_t,
            "p_turb": p_turb,
            "dissipated_energy": diss_energy,
            "cumulative_dissipated_energy": self.cumulative_dissipated_energy,
        }
