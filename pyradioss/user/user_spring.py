"""
User-defined spring subroutine framework (/PROP/USER1, /PROP/USER2, /PROP/USER3).

Upstream OpenRadioss Fortran references:
- Ported from $OR_SRC/engine/source/elements/spring/rforc3.F (lines 917-970)
- Ported from $OR_SRC/engine/source/user_interface/dyn_userlib.c (eng_userlib_ruser_)

In OpenRadioss:
- IGTYP == 29: /PROP/USER1 - SPRING
- IGTYP == 30: /PROP/USER2 - SPRING
- IGTYP == 31: /PROP/USER3 - SPRING

When USERL_AVAIL == 0 (no dynamic library loaded), OpenRadioss calls:
    OPTION='PROP/USER* - SPRING'
    CALL ANCMSG(MSGID=257, C1=OPTION)
    CALL ARRET(2)

In pyradioss, users can subclass `UserSpring`, implement `forces` and `stiffness`,
and register their implementation via `register_user_spring(type_number, spring_class)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np


@dataclass
class SpringState:
    """State of a spring element passed to user spring subroutines.

    Attributes:
        disp: Elongation (L - L0) for 1D springs, or displacement vector (dx, dy, dz).
        vel: Relative velocity along spring axis dL/dt, or velocity vector.
        length: Current spring length L.
        length0: Initial spring length L0.
        rot: Relative angular displacement (rx, ry, rz) for rotational/6-DOF springs.
        rot_vel: Relative angular velocity (vrx, vry, vrz).
        uvar: User state variables / history array (analogous to USER_UVAR).
        extra: Additional property context and parameters.
    """

    disp: float = 0.0
    vel: float = 0.0
    length: float = 0.0
    length0: float = 0.0
    rot: Optional[np.ndarray] = None
    rot_vel: Optional[np.ndarray] = None
    uvar: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    extra: Dict[str, Any] = field(default_factory=dict)


class UserSpring(ABC):
    """Abstract base class for user-defined spring elements.

    Subclasses must implement:
    - `forces(state, dt)`: Compute axial force (or force/moment components) and update state.

    Subclasses may optionally implement:
    - `stiffness(state)`: Return tangent stiffness for time step and implicit solutions.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self.params.update(kwargs)

    @abstractmethod
    def forces(
        self, state: Union[SpringState, Any], dt: float
    ) -> Union[float, Tuple[float, ...], np.ndarray]:
        """Compute spring forces from current kinematic state.

        Ported from:
        $OR_SRC/engine/source/elements/spring/rforc3.F (lines 920-962)

        Args:
            state: SpringState or equivalent object containing `disp`, `vel`, `length`, etc.
            dt: Time increment.

        Returns:
            Scalar axial force F, or 6-component vector [Fx, Fy, Fz, Mx, My, Mz].
        """
        ...

    def stiffness(self, state: Union[SpringState, Any]) -> float:
        """Return tangent stiffness K for time step calculation and implicit solver.

        Default implementation extracts 'stiffness' or 'k' from self.params.
        """
        k = self.params.get("stiffness", self.params.get("k", 0.0))
        return float(k)


class UserSpringType29(UserSpring):
    """Stub for /PROP/USER1 (Spring Property Type 29).

    Ported from:
    $OR_SRC/engine/source/elements/spring/rforc3.F (lines 917-935)
    """

    def forces(
        self, state: Union[SpringState, Any], dt: float
    ) -> Union[float, Tuple[float, ...], np.ndarray]:
        raise NotImplementedError(
            "User spring TYPE29 (/PROP/USER1 - SPRING) is a stub. "
            "To define custom spring behavior, subclass `UserSpring` and register it via:\n"
            "    from pyradioss.user import register_user_spring\n"
            "    register_user_spring(29, MyCustomSpring)"
        )


class UserSpringType30(UserSpring):
    """Stub for /PROP/USER2 (Spring Property Type 30).

    Ported from:
    $OR_SRC/engine/source/elements/spring/rforc3.F (lines 936-953)
    """

    def forces(
        self, state: Union[SpringState, Any], dt: float
    ) -> Union[float, Tuple[float, ...], np.ndarray]:
        raise NotImplementedError(
            "User spring TYPE30 (/PROP/USER2 - SPRING) is a stub. "
            "To define custom spring behavior, subclass `UserSpring` and register it via:\n"
            "    from pyradioss.user import register_user_spring\n"
            "    register_user_spring(30, MyCustomSpring)"
        )


class UserSpringType31(UserSpring):
    """Stub for /PROP/USER3 (Spring Property Type 31).

    Ported from:
    $OR_SRC/engine/source/elements/spring/rforc3.F (lines 954-970)
    """

    def forces(
        self, state: Union[SpringState, Any], dt: float
    ) -> Union[float, Tuple[float, ...], np.ndarray]:
        raise NotImplementedError(
            "User spring TYPE31 (/PROP/USER3 - SPRING) is a stub. "
            "To define custom spring behavior, subclass `UserSpring` and register it via:\n"
            "    from pyradioss.user import register_user_spring\n"
            "    register_user_spring(31, MyCustomSpring)"
        )


# Global user spring registry: type_number (int) -> UserSpring class
_USER_SPRINGS: Dict[int, Type[UserSpring]] = {
    29: UserSpringType29,
    30: UserSpringType30,
    31: UserSpringType31,
}


def register_user_spring(type_number: int, spring_class: Type[UserSpring]) -> None:
    """Register a custom user spring class for a given spring property type.

    Args:
        type_number: Integer spring property type (e.g. 29, 30, 31).
        spring_class: Subclass of `UserSpring`.

    Raises:
        TypeError: If `spring_class` is not a subclass of `UserSpring`.
    """
    if not isinstance(spring_class, type) or not issubclass(spring_class, UserSpring):
        raise TypeError(f"Expected a subclass of UserSpring, got {spring_class!r}")
    _USER_SPRINGS[int(type_number)] = spring_class


def get_user_spring(type_number: int) -> Optional[Type[UserSpring]]:
    """Look up a registered user spring class by type number.

    Args:
        type_number: Integer spring property type.

    Returns:
        The registered UserSpring subclass, or None if not registered.
    """
    return _USER_SPRINGS.get(int(type_number))


def unregister_user_spring(type_number: int) -> None:
    """Unregister a user spring class for the given type number."""
    _USER_SPRINGS.pop(int(type_number), None)


def list_user_springs() -> Dict[int, Type[UserSpring]]:
    """Return a shallow copy of all registered user spring classes."""
    return dict(_USER_SPRINGS)
