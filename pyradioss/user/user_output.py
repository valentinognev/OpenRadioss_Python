"""
User-defined output subroutine framework.

Upstream OpenRadioss Fortran references:
- Ported from $OR_SRC/engine/source/user_interface/user_output.F (USER_OUTPUT)
- Ported from $OR_SRC/engine/source/user_interface/dyn_userlib.c

In OpenRadioss, USER_OUTPUT opens and processes user-generated scratch files
(e.g., SO<root>_<claw>.scr) and writes user output blocks into listing files.

In pyradioss, users can subclass `UserOutput` and implement `write(self, model, time, cycle)`
to export custom diagnostic tables, VTK data, JSON telemetry, or arbitrary metrics
during the simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union


class UserOutput(ABC):
    """Abstract base class for user-defined simulation output routines.

    Subclasses must implement:
    - `write(model, time, cycle)`: Called at user-requested output intervals.
    """

    def __init__(self, name: str = "", params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.name = name or self.__class__.__name__
        self.params: Dict[str, Any] = dict(params or {})
        self.params.update(kwargs)

    @abstractmethod
    def write(self, model: Any, time: float, cycle: int) -> None:
        """Execute user output logic.

        Ported from:
        $OR_SRC/engine/source/user_interface/user_output.F

        Args:
            model: Current simulation Model instance.
            time: Current simulation time.
            cycle: Current cycle number.
        """
        ...


# Global user output registry
_USER_OUTPUT_CLASSES: Dict[str, Type[UserOutput]] = {}
_ACTIVE_USER_OUTPUTS: List[UserOutput] = []


def register_user_output(
    output_item: Union[Type[UserOutput], UserOutput], name: Optional[str] = None
) -> None:
    """Register a custom user output class or active instance.

    Args:
        output_item: Subclass of UserOutput, or an instance of UserOutput.
        name: Optional custom identifier.

    Raises:
        TypeError: If output_item is neither a UserOutput subclass nor instance.
    """
    if isinstance(output_item, type) and issubclass(output_item, UserOutput):
        key = name or output_item.__name__
        _USER_OUTPUT_CLASSES[key] = output_item
    elif isinstance(output_item, UserOutput):
        key = name or output_item.name or output_item.__class__.__name__
        _USER_OUTPUT_CLASSES[key] = type(output_item)
        if output_item not in _ACTIVE_USER_OUTPUTS:
            _ACTIVE_USER_OUTPUTS.append(output_item)
    else:
        raise TypeError(
            f"Expected a UserOutput subclass or instance, got {output_item!r}"
        )


def get_user_output(name: str) -> Optional[Type[UserOutput]]:
    """Look up a registered user output class by name."""
    return _USER_OUTPUT_CLASSES.get(name)


def get_active_user_outputs() -> List[UserOutput]:
    """Return all active UserOutput instances."""
    return list(_ACTIVE_USER_OUTPUTS)


def clear_user_outputs() -> None:
    """Clear all registered user output classes and instances."""
    _USER_OUTPUT_CLASSES.clear()
    _ACTIVE_USER_OUTPUTS.clear()


def list_user_outputs() -> Dict[str, Type[UserOutput]]:
    """Return a shallow copy of all registered user output classes."""
    return dict(_USER_OUTPUT_CLASSES)
