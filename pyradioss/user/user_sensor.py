"""
User-defined sensor subroutine framework (/SENSOR/USER, /SENSOR/TYPE29, TYPE30, TYPE31).

Upstream OpenRadioss Fortran references:
- Ported from $OR_SRC/engine/source/tools/sensor/sensor_base.F (lines 232-283)
- Ported from $OR_SRC/engine/source/user_interface/usensor.F
- Ported from $OR_SRC/engine/source/user_interface/dyn_userlib.c (eng_user_sens, ENG_USERLIB_USER_SENS)

In OpenRadioss:
- TYP == 29: /SENSOR/USER 29
- TYP == 30: /SENSOR/USER 30
- TYP == 31: /SENSOR/USER 31

When USERL_AVAIL == 0, OpenRadioss issues:
    OPTION='USER SENSOR <TYP>'
    CALL ANCMSG(MSGID=257, C1=OPTION)
    CALL ARRET(2)

In pyradioss, users can subclass `UserSensor`, implement `evaluate(self, model, time)`,
and register their sensor class with `register_user_sensor(sensor_class)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union


class UserSensor(ABC):
    """Abstract base class for user-defined event sensors.

    Subclasses must implement:
    - `evaluate(model, time) -> bool`: Return True when triggered/activated.
    """

    def __init__(self, sensor_id: int = 0, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.sensor_id = sensor_id
        self.params: Dict[str, Any] = dict(params or {})
        self.params.update(kwargs)
        self.active: bool = False
        self.fire_time: Optional[float] = None

    @abstractmethod
    def evaluate(self, model: Any, time: float) -> bool:
        """Evaluate custom trigger condition.

        Ported from:
        $OR_SRC/engine/source/tools/sensor/sensor_base.F (lines 232-283)

        Args:
            model: Radioss Model instance providing access to nodal coordinates,
                   displacements, velocities, energies, etc.
            time: Current simulation time.

        Returns:
            True if the sensor should trigger/fire at this time, False otherwise.
        """
        ...


class UserSensorType29(UserSensor):
    """Stub for user sensor TYPE 29.

    Ported from:
    $OR_SRC/engine/source/tools/sensor/sensor_base.F (lines 232-248)
    """

    def evaluate(self, model: Any, time: float) -> bool:
        raise NotImplementedError(
            "User sensor TYPE 29 ('USER SENSOR 29') is a stub. "
            "To define custom sensor logic, subclass `UserSensor` and register it via:\n"
            "    from pyradioss.user import register_user_sensor\n"
            "    register_user_sensor(MyCustomSensor, type_number=29)"
        )


class UserSensorType30(UserSensor):
    """Stub for user sensor TYPE 30.

    Ported from:
    $OR_SRC/engine/source/tools/sensor/sensor_base.F (lines 250-266)
    """

    def evaluate(self, model: Any, time: float) -> bool:
        raise NotImplementedError(
            "User sensor TYPE 30 ('USER SENSOR 30') is a stub. "
            "To define custom sensor logic, subclass `UserSensor` and register it via:\n"
            "    from pyradioss.user import register_user_sensor\n"
            "    register_user_sensor(MyCustomSensor, type_number=30)"
        )


class UserSensorType31(UserSensor):
    """Stub for user sensor TYPE 31.

    Ported from:
    $OR_SRC/engine/source/tools/sensor/sensor_base.F (lines 268-283)
    """

    def evaluate(self, model: Any, time: float) -> bool:
        raise NotImplementedError(
            "User sensor TYPE 31 ('USER SENSOR 31') is a stub. "
            "To define custom sensor logic, subclass `UserSensor` and register it via:\n"
            "    from pyradioss.user import register_user_sensor\n"
            "    register_user_sensor(MyCustomSensor, type_number=31)"
        )


# Global user sensor registry: type_number (int) -> UserSensor class
_USER_SENSORS: Dict[int, Type[UserSensor]] = {
    29: UserSensorType29,
    30: UserSensorType30,
    31: UserSensorType31,
}


def register_user_sensor(
    sensor_class: Type[UserSensor], type_number: int = 29
) -> None:
    """Register a custom user sensor class.

    Args:
        sensor_class: Class inheriting from `UserSensor`.
        type_number: Sensor type number (default: 29).

    Raises:
        TypeError: If `sensor_class` is not a subclass of `UserSensor`.
    """
    if not isinstance(sensor_class, type) or not issubclass(sensor_class, UserSensor):
        raise TypeError(f"Expected a subclass of UserSensor, got {sensor_class!r}")
    _USER_SENSORS[int(type_number)] = sensor_class


def get_user_sensor(type_number: int = 29) -> Optional[Type[UserSensor]]:
    """Look up a registered user sensor class by type number.

    Args:
        type_number: Integer sensor type (default: 29).

    Returns:
        The registered UserSensor subclass, or None if not found.
    """
    return _USER_SENSORS.get(int(type_number))


def unregister_user_sensor(type_number: int = 29) -> None:
    """Unregister a user sensor class for the given type number."""
    _USER_SENSORS.pop(int(type_number), None)


def list_user_sensors() -> Dict[int, Type[UserSensor]]:
    """Return a shallow copy of all registered user sensor classes."""
    return dict(_USER_SENSORS)
