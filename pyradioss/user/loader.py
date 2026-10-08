"""
Dynamic loader for user-defined Python subroutines and libraries.

Upstream OpenRadioss C/Fortran references:
- Ported from $OR_SRC/engine/source/user_interface/dyn_userlib.c (DYN_USERLIB_INIT)
- Environment variable RAD_USERLIB_LIBPATH used in dyn_userlib.c (lines 105-112)

In OpenRadioss C, DYN_USERLIB_INIT uses `LoadLibrary` (Windows) or `dlopen` (Linux)
to link user `.dll` or `.so` libraries, scanning the directory pointed to by
`RAD_USERLIB_LIBPATH` or the current working directory.

In pyradioss, this loader provides equivalent dynamic loading for Python user modules
(`user_*.py`) using `importlib.util.spec_from_file_location`, allowing custom material
laws, springs, sensors, and output hooks to be loaded at runtime without modifying pyradioss.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Union


def load_user_library(path: Union[str, Path]) -> ModuleType:
    """Dynamically load a Python module from a file path.

    Ported from dynamic library loader in:
    $OR_SRC/engine/source/user_interface/dyn_userlib.c

    Args:
        path: Path to the .py user module file.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ImportError: If the module cannot be imported or executed.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"User library file not found: {p}")

    # Generate a unique module name based on file stem and resolved path hash
    module_name = f"pyradioss_user_{p.stem}_{abs(hash(str(p))) % 1000000:06d}"
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification for: {p}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ImportError(f"Failed to execute user library module {p}: {exc}") from exc

    return mod


def auto_discover_user_modules(search_dir: Union[str, Path]) -> List[ModuleType]:
    """Find and dynamically load all `user_*.py` modules in a directory.

    Corresponds to the directory scan mechanism in dyn_userlib.c where
    libraries in the current directory or `RAD_USERLIB_LIBPATH` are discovered.

    Args:
        search_dir: Directory path to scan for `user_*.py` files.

    Returns:
        List of successfully loaded module objects.
    """
    d = Path(search_dir).resolve()
    if not d.is_dir():
        return []

    loaded: List[ModuleType] = []
    # Discover all user_*.py files, excluding __init__.py or temporary files
    for file_path in sorted(d.glob("user_*.py")):
        if file_path.is_file() and not file_path.name.startswith((".", "__")):
            try:
                mod = load_user_library(file_path)
                loaded.append(mod)
            except Exception:
                # Re-raise to ensure user knows if a discovered module fails
                raise

    return loaded


def load_from_environment(env_var: str = "RAD_USERLIB_LIBPATH") -> List[ModuleType]:
    """Load user libraries found in the directory specified by an environment variable.

    Ported from dyn_userlib.c lines 105-112:
    `dllpath_size = GetEnvironmentVariable("RAD_USERLIB_LIBPATH", dllpath, 10240);`

    Args:
        env_var: Environment variable name (default: "RAD_USERLIB_LIBPATH").

    Returns:
        List of loaded modules.
    """
    lib_path = os.environ.get(env_var)
    if not lib_path:
        return []
    return auto_discover_user_modules(lib_path)
