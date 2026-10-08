"""
User-defined material subroutine framework for pyradioss (/MAT/LAW29, /MAT/USER*, /MAT/LAW99).

Upstream OpenRadioss Fortran references:
- Ported from $OR_SRC/engine/source/user_interface/nolib_usermat99.F
- Ported from $OR_SRC/engine/source/materials/mat_share/usermat_solid.F
- Ported from $OR_SRC/engine/source/materials/mat_share/usermat_shell.F
- Ported from $OR_SRC/engine/source/user_interface/dyn_userlib.c

In OpenRadioss, user-defined materials (LAW29 through LAW99) allow custom stress integration
routines dynamically linked via dyn_userlib.c and called from usermat_solid.F (3D solid elements)
and usermat_shell.F (2D shell elements). When no dynamic library is linked, OpenRadioss routes
the call to nolib_usermat99.F which issues an error (MSGID 257) and aborts (ARRET).

In pyradioss, users can implement custom material laws in pure Python or NumPy by subclassing
`UserMaterial` and registering the class via `register_user_material(law_number, material_class)`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

import numpy as np


@dataclass
class MaterialState:
    """State bundle passed to user material update routines.

    Attributes:
        sig: Stress tensor array. For 3D solids: (n, 6) or (6,) Voigt [xx, yy, zz, xy, yz, zx].
             For shells: (n, 3) or (3,) plane stress Voigt [xx, yy, xy].
        deps: Strain increment array (engineering shear).
        rho: Current density (default 1.0).
        uvar: User state variables / history array (analogous to UVAR in OpenRadioss).
        extra: Additional kinematics or context dictionary (e.g., deformation gradient F, temp).
        strain_rate: Optional strain rate tensor array (deps / dt).
        epsp: Optional equivalent plastic strain array.
    """

    sig: np.ndarray
    deps: np.ndarray
    rho: float = 1.0
    uvar: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    extra: Dict[str, Any] = field(default_factory=dict)
    strain_rate: Optional[np.ndarray] = None
    epsp: Optional[np.ndarray] = None


class UserMaterial(ABC):
    """Abstract base class for user-defined constitutive material laws.

    Subclasses must implement:
    - `solid_update`: 3D stress integration from strain rate / increment.
    - `shell_update`: Plane stress integration for shells.

    Subclasses may optionally override:
    - `sound_speed`: Acoustic wave speed for Courant time step control.
    - `tangent_modulus`: Consistent or elastic tangent stiffness matrix for implicit solvers.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self.params.update(kwargs)

    @abstractmethod
    def solid_update(self, state: Union[MaterialState, Any], dt: float) -> np.ndarray:
        """Compute 3D stress update from strain rate / increment.

        Ported from:
        $OR_SRC/engine/source/materials/mat_share/usermat_solid.F

        Args:
            state: MaterialState instance, dictionary, or object providing `sig`, `deps`,
                   `rho`, `uvar`, and optional parameters.
            dt: Time step duration.

        Returns:
            Updated Cauchy stress array `sig` (n, 6) or (6,).
        """
        ...

    @abstractmethod
    def shell_update(self, state: Union[MaterialState, Any], dt: float) -> np.ndarray:
        """Compute plane stress update for 2D shell elements.

        Ported from:
        $OR_SRC/engine/source/materials/mat_share/usermat_shell.F

        Args:
            state: MaterialState instance, dictionary, or object providing `sig`, `deps`,
                   `rho`, `uvar`, and optional parameters.
            dt: Time step duration.

        Returns:
            Updated plane stress array `sig` (n, 3) or (3,).
        """
        ...

    def sound_speed(self, state: Union[MaterialState, Any]) -> float:
        """Compute acoustic wave speed for critical time step calculation.

        Ported from:
        $OR_SRC/engine/source/materials/mat_share/usermat_solid.F
        (SOUNDSP calculation)

        Returns:
            Acoustic sound speed c = sqrt((K + 4/3 G) / rho). Default implementation
            extracts E, nu, rho from parameters/state or returns 0.0 if not specified.
        """
        params = getattr(self, "params", {})
        rho = getattr(state, "rho", None)
        if rho is None and isinstance(state, dict):
            rho = state.get("rho")
        if rho is None:
            rho = params.get("rho0", params.get("rho", 1.0))
        rho = max(float(rho), 1.0e-20)

        E = params.get("E", params.get("young", 0.0))
        nu = params.get("nu", params.get("poisson", 0.0))
        if E > 0.0 and -1.0 < nu < 0.5:
            G = E / (2.0 * (1.0 + nu))
            K = E / (3.0 * (1.0 - 2.0 * nu))
            c2 = (K + 4.0 / 3.0 * G) / rho
            return math.sqrt(max(c2, 0.0))
        return 0.0

    def tangent_modulus(self, state: Union[MaterialState, Any]) -> np.ndarray:
        """Compute consistent or elastic constitutive tangent matrix C.

        For 3D solids, returns a (6, 6) matrix in Voigt ordering [xx, yy, zz, xy, yz, zx].
        For shells, returns a (3, 3) plane-stress matrix [xx, yy, xy].

        Default implementation builds the standard isotropic elastic tangent if E and nu
        are present in `self.params`.
        """
        params = getattr(self, "params", {})
        E = float(params.get("E", params.get("young", 0.0)))
        nu = float(params.get("nu", params.get("poisson", 0.0)))
        if E > 0.0 and -1.0 < nu < 0.5:
            G = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return np.array([
                [lam + 2.0 * G, lam, lam, 0.0, 0.0, 0.0],
                [lam, lam + 2.0 * G, lam, 0.0, 0.0, 0.0],
                [lam, lam, lam + 2.0 * G, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, G, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, G, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, G],
            ], dtype=float)
        return np.zeros((6, 6), dtype=float)


class UserMaterialLaw29(UserMaterial):
    """Stub implementation for /MAT/LAW29 (/MAT/USER_LAW29).

    Ported from:
    - $OR_SRC/engine/source/user_interface/nolib_usermat99.F
    - $OR_SRC/engine/source/materials/mat_share/usermat_solid.F (lines 985-995)
    - $OR_SRC/engine/source/materials/mat_share/usermat_shell.F (lines 943-953)

    In Fortran OpenRadioss, selecting LAW29 without a dynamic user library produces:
        OPTION='/MAT/USER29 - SOLID' (or SHELL)
        CALL ANCMSG(MSGID=257, C1=OPTION)
        CALL ARRET(2)

    This stub raises NotImplementedError with clear instructions for implementing
    a custom Python material law.
    """

    def solid_update(self, state: Union[MaterialState, Any], dt: float) -> np.ndarray:
        raise NotImplementedError(
            "User material LAW29 (/MAT/USER29) is a stub. "
            "To define custom material behavior, create a subclass of `UserMaterial` "
            "implementing `solid_update` and `shell_update`, then register it via:\n"
            "    from pyradioss.user import register_user_material\n"
            "    register_user_material(29, MyCustomMaterial)"
        )

    def shell_update(self, state: Union[MaterialState, Any], dt: float) -> np.ndarray:
        raise NotImplementedError(
            "User material LAW29 (/MAT/USER29 - SHELL) is a stub. "
            "To define custom material behavior, create a subclass of `UserMaterial` "
            "implementing `solid_update` and `shell_update`, then register it via:\n"
            "    from pyradioss.user import register_user_material\n"
            "    register_user_material(29, MyCustomMaterial)"
        )


class UserMaterialLaw99(UserMaterial):
    """Stub implementation for /MAT/LAW99 (/MAT/USER99).

    Ported from:
    - $OR_SRC/engine/source/user_interface/nolib_usermat99.F
    """

    def solid_update(self, state: Union[MaterialState, Any], dt: float) -> np.ndarray:
        raise NotImplementedError(
            "User material LAW99 (/MAT/USER99) is a stub. "
            "To define custom material behavior, create a subclass of `UserMaterial` "
            "and register it via `register_user_material(99, MyCustomMaterial)`."
        )

    def shell_update(self, state: Union[MaterialState, Any], dt: float) -> np.ndarray:
        raise NotImplementedError(
            "User material LAW99 (/MAT/USER99 - SHELL) is a stub. "
            "To define custom material behavior, create a subclass of `UserMaterial` "
            "and register it via `register_user_material(99, MyCustomMaterial)`."
        )


# Global user material registry: law_number (int) -> UserMaterial class
_USER_MATERIALS: Dict[int, Type[UserMaterial]] = {
    29: UserMaterialLaw29,
    99: UserMaterialLaw99,
}


def register_user_material(law_number: int, material_class: Type[UserMaterial]) -> None:
    """Register a custom user material class for a specific LAW number.

    Args:
        law_number: Integer material law ID (e.g. 29, 99, or custom 1..99).
        material_class: Class inheriting from `UserMaterial`.

    Raises:
        TypeError: If `material_class` is not a subclass of `UserMaterial`.
    """
    if not isinstance(material_class, type) or not issubclass(material_class, UserMaterial):
        raise TypeError(
            f"Expected a subclass of UserMaterial, got {material_class!r}"
        )
    _USER_MATERIALS[int(law_number)] = material_class


def get_user_material(law_number: int) -> Optional[Type[UserMaterial]]:
    """Look up a registered user material class by law number.

    Args:
        law_number: Integer material law ID.

    Returns:
        The registered UserMaterial subclass, or None if not registered.
    """
    return _USER_MATERIALS.get(int(law_number))


def unregister_user_material(law_number: int) -> None:
    """Unregister a user material class for the given law number."""
    _USER_MATERIALS.pop(int(law_number), None)


def list_user_materials() -> Dict[int, Type[UserMaterial]]:
    """Return a shallow copy of all registered user material classes."""
    return dict(_USER_MATERIALS)
