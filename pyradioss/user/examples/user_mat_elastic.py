"""
Example: Linear Elastic User-Defined Material Law for pyradioss.

Demonstrates how to write a custom material law conforming to the `UserMaterial` ABC.

Upstream OpenRadioss Fortran references:
- Ported from $OR_SRC/engine/source/materials/mat/mat001/sigeps01.F (solid elasticity)
- Ported from $OR_SRC/engine/source/materials/mat/mat001/sigeps01c.F (shell plane stress)
- Ported from $OR_SRC/engine/source/materials/mat_share/usermat_solid.F
- Ported from $OR_SRC/engine/source/materials/mat_share/usermat_shell.F
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import numpy as np

from pyradioss.user.user_material import MaterialState, UserMaterial, register_user_material


class UserElasticMaterial(UserMaterial):
    """Linear isotropic hypoelastic material implemented as a UserMaterial.

    Governing equations:
    - 3D Continuum (solids):
        dsigma_ij = lambda * tr(deps) * delta_ij + 2 * G * deps_ij
        tau_ij = G * dgamma_ij  (engineering shear: dgamma = 2 * deps)
    - 2D Shell (plane stress):
        c = E / (1 - nu^2)
        dsigma_xx = c * (deps_xx + nu * deps_yy)
        dsigma_yy = c * (deps_yy + nu * deps_xx)
        dsigma_xy = G * dgamma_xy
    - Sound speed:
        c = sqrt((K + 4/3 * G) / rho)
    """

    def __init__(
        self,
        E: float = 210000.0,
        nu: float = 0.3,
        rho0: float = 7.85e-9,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        merged_params = {"E": E, "nu": nu, "rho0": rho0}
        if params:
            merged_params.update(params)
        merged_params.update(kwargs)
        super().__init__(params=merged_params)

        self.E: float = float(self.params["E"])
        self.nu: float = float(self.params["nu"])
        self.rho0: float = float(self.params.get("rho0", self.params.get("rho", 1.0)))

        # Lame parameters and bulk modulus
        self.G: float = self.E / (2.0 * (1.0 + self.nu))
        self.K: float = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        self.lam: float = self.K - (2.0 / 3.0) * self.G

    def _unpack_state(
        self, state: Union[MaterialState, Any]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Extract sig, deps, and rho from state."""
        if isinstance(state, dict):
            sig = state["sig"]
            deps = state["deps"]
            rho = float(state.get("rho", self.rho0))
        elif isinstance(state, MaterialState):
            sig = state.sig
            deps = state.deps
            rho = float(state.rho) if state.rho > 0.0 else self.rho0
        else:
            sig = getattr(state, "sig")
            deps = getattr(state, "deps")
            rho = float(getattr(state, "rho", self.rho0))
        return sig, deps, rho

    def solid_update(self, state: Union[MaterialState, Any], dt: float) -> np.ndarray:
        """3D stress update for continuum solid elements.

        Ported from sigeps01.F and usermat_solid.F.
        """
        sig, deps, _ = self._unpack_state(state)
        sig_arr = np.asarray(sig, dtype=float)
        deps_arr = np.asarray(deps, dtype=float)

        single_element = (sig_arr.ndim == 1)
        if single_element:
            sig_work = sig_arr.reshape(1, -1)
            deps_work = deps_arr.reshape(1, -1)
        else:
            sig_work = sig_arr
            deps_work = deps_arr

        if sig_work.shape[0] == 0:
            return sig_arr

        # Volumetric trace
        tr = deps_work[:, 0] + deps_work[:, 1] + deps_work[:, 2]

        # Normal stress update
        sig_work[:, 0] += self.lam * tr + 2.0 * self.G * deps_work[:, 0]
        sig_work[:, 1] += self.lam * tr + 2.0 * self.G * deps_work[:, 1]
        sig_work[:, 2] += self.lam * tr + 2.0 * self.G * deps_work[:, 2]

        # Engineering shear stress update
        sig_work[:, 3:] += self.G * deps_work[:, 3:]

        if single_element:
            sig_arr[:] = sig_work.reshape(-1)
        else:
            sig_arr[:] = sig_work
        return sig_arr

    def shell_update(self, state: Union[MaterialState, Any], dt: float) -> np.ndarray:
        """Plane stress update for 2D shell elements.

        Ported from sigeps01c.F and usermat_shell.F.
        """
        sig, deps, _ = self._unpack_state(state)
        sig_arr = np.asarray(sig, dtype=float)
        deps_arr = np.asarray(deps, dtype=float)

        single_element = (sig_arr.ndim == 1)
        if single_element:
            sig_work = sig_arr.reshape(1, -1)
            deps_work = deps_arr.reshape(1, -1)
        else:
            sig_work = sig_arr
            deps_work = deps_arr

        if sig_work.shape[0] == 0:
            return sig_arr

        c = self.E / (1.0 - self.nu * self.nu)
        dxx = deps_work[:, 0]
        dyy = deps_work[:, 1]

        sig_work[:, 0] += c * (dxx + self.nu * dyy)
        sig_work[:, 1] += c * (dyy + self.nu * dxx)
        sig_work[:, 2] += self.G * deps_work[:, 2]

        if single_element:
            sig_arr[:] = sig_work.reshape(-1)
        else:
            sig_arr[:] = sig_work
        return sig_arr

    def sound_speed(self, state: Union[MaterialState, Any]) -> float:
        """Acoustic wave speed c = sqrt((K + 4/3 G) / rho)."""
        _, _, rho = self._unpack_state(state)
        r = max(rho, 1.0e-20)
        c2 = (self.K + 4.0 / 3.0 * self.G) / r
        return math.sqrt(max(c2, 0.0))

    def tangent_modulus(self, state: Union[MaterialState, Any]) -> np.ndarray:
        """(6, 6) isotropic elastic tangent matrix in Voigt ordering."""
        return np.array([
            [self.lam + 2.0 * self.G, self.lam, self.lam, 0.0, 0.0, 0.0],
            [self.lam, self.lam + 2.0 * self.G, self.lam, 0.0, 0.0, 0.0],
            [self.lam, self.lam, self.lam + 2.0 * self.G, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, self.G, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, self.G, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, self.G],
        ], dtype=float)


# Example registration hook: register for custom law 90 or 29
register_user_material(90, UserElasticMaterial)
