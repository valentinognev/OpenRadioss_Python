"""
pyradioss.user — User-Defined Subroutine Framework.

Upstream OpenRadioss Fortran and C references:
- Materials:
  - $OR_SRC/engine/source/user_interface/nolib_usermat99.F
  - $OR_SRC/engine/source/materials/mat_share/usermat_solid.F
  - $OR_SRC/engine/source/materials/mat_share/usermat_shell.F
- Springs:
  - $OR_SRC/engine/source/elements/spring/rforc3.F (lines 917-970)
- Sensors:
  - $OR_SRC/engine/source/tools/sensor/sensor_base.F (lines 232-283)
  - $OR_SRC/engine/source/user_interface/usensor.F
- Outputs:
  - $OR_SRC/engine/source/user_interface/user_output.F
- Dynamic Library Loading:
  - $OR_SRC/engine/source/user_interface/dyn_userlib.c
  - $OR_SRC/engine/source/user_interface/dyn_userlib_callback.c
  - $OR_SRC/engine/source/user_interface/user_windows.F

How to write custom routines in pyradioss
==========================================

1. Custom Material Laws (/MAT/LAW29, /MAT/LAW99, /MAT/USER*)
------------------------------------------------------------
Subclass `UserMaterial`, implement `solid_update` (for 3D continuum elements)
and `shell_update` (for 2D shell elements), and register your class:

    import numpy as np
    from pyradioss.user import UserMaterial, register_user_material

    class MyPlasticMaterial(UserMaterial):
        def solid_update(self, state, dt):
            # state.sig: (n, 6) Cauchy stress in Voigt ordering [xx, yy, zz, xy, yz, zx]
            # state.deps: (n, 6) strain increment
            # Return updated stress array
            ...
            return state.sig

        def shell_update(self, state, dt):
            # state.sig: (n, 3) Plane stress in Voigt ordering [xx, yy, xy]
            # state.deps: (n, 3) plane strain increment
            ...
            return state.sig

    register_user_material(29, MyPlasticMaterial)

2. Custom Spring Elements (/PROP/USER1, /PROP/USER2, /PROP/USER3)
-----------------------------------------------------------------
Subclass `UserSpring`, implement `forces(self, state, dt)`:

    from pyradioss.user import UserSpring, register_user_spring

    class MyNonlinearSpring(UserSpring):
        def forces(self, state, dt):
            # state.disp: axial elongation (L - L0)
            # state.vel: relative velocity dL/dt
            k = self.params.get("k", 1000.0)
            return k * state.disp

    register_user_spring(29, MyNonlinearSpring)

3. Custom Event Sensors (/SENSOR/USER)
--------------------------------------
Subclass `UserSensor`, implement `evaluate(self, model, time) -> bool`:

    from pyradioss.user import UserSensor, register_user_sensor

    class MyDisplacementSensor(UserSensor):
        def evaluate(self, model, time):
            # Check displacement or energy from model state
            return time > 0.05

    register_user_sensor(MyDisplacementSensor, type_number=29)

4. Custom Output Routines (USER_OUTPUT)
---------------------------------------
Subclass `UserOutput`, implement `write(self, model, time, cycle)`:

    from pyradioss.user import UserOutput, register_user_output

    class MyCsvLogger(UserOutput):
        def write(self, model, time, cycle):
            print(f"Cycle {cycle}: time={time:.6e}")

    register_user_output(MyCsvLogger)

5. Dynamic Module Loading
-------------------------
Custom user scripts can be loaded dynamically from standalone files without modifying
pyradioss code:

    from pyradioss.user import load_user_library, auto_discover_user_modules

    # Load a single file
    load_user_library("path/to/my_custom_routines.py")

    # Or auto-discover all user_*.py files in a directory
    auto_discover_user_modules("path/to/user_plugins_dir")
"""

from .loader import (
    auto_discover_user_modules,
    load_from_environment,
    load_user_library,
)
from .user_material import (
    MaterialState,
    UserMaterial,
    UserMaterialLaw29,
    UserMaterialLaw99,
    get_user_material,
    list_user_materials,
    register_user_material,
    unregister_user_material,
)
from .user_output import (
    UserOutput,
    clear_user_outputs,
    get_active_user_outputs,
    get_user_output,
    list_user_outputs,
    register_user_output,
)
from .user_sensor import (
    UserSensor,
    UserSensorType29,
    UserSensorType30,
    UserSensorType31,
    get_user_sensor,
    list_user_sensors,
    register_user_sensor,
    unregister_user_sensor,
)
from .user_spring import (
    SpringState,
    UserSpring,
    UserSpringType29,
    UserSpringType30,
    UserSpringType31,
    get_user_spring,
    list_user_springs,
    register_user_spring,
    unregister_user_spring,
)

__all__ = [
    # Material
    "MaterialState",
    "UserMaterial",
    "UserMaterialLaw29",
    "UserMaterialLaw99",
    "register_user_material",
    "get_user_material",
    "unregister_user_material",
    "list_user_materials",
    # Spring
    "SpringState",
    "UserSpring",
    "UserSpringType29",
    "UserSpringType30",
    "UserSpringType31",
    "register_user_spring",
    "get_user_spring",
    "unregister_user_spring",
    "list_user_springs",
    # Sensor
    "UserSensor",
    "UserSensorType29",
    "UserSensorType30",
    "UserSensorType31",
    "register_user_sensor",
    "get_user_sensor",
    "unregister_user_sensor",
    "list_user_sensors",
    # Output
    "UserOutput",
    "register_user_output",
    "get_user_output",
    "get_active_user_outputs",
    "clear_user_outputs",
    "list_user_outputs",
    # Loader
    "load_user_library",
    "auto_discover_user_modules",
    "load_from_environment",
]
