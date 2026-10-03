"""
Post-processing converters for the pyradioss GUI — no ``tkinter`` import.

No Fortran counterpart (see the package docstring). This module wraps three
result converters so a finished run can be opened in the usual downstream
viewers, and — like :mod:`pyradioss.gui.runner` — is kept free of any
``tkinter`` import so it is fully unit-testable head-less and can be driven
by a scripted caller.

The three converters
--------------------
1. **anim -> d3plot** (the headline): the open-source *Vortex-Radioss*
   library (Vortex-CAE) reads OpenRadioss animation files (``…A001``,
   ``…A002`` …) and writes an LS-Dyna ``.d3plot`` family so results open in
   LS-PrePost. Run in a subprocess of the *same* interpreter (mirroring the
   user's own ``oropt`` d3plot bridge) with ``use_shell_mask=True`` /
   ``use_solid_mask=False``.

   **Version pin (VORTEX_PIN):** ``v1.021``. Vortex-Radioss is *not* on PyPI,
   so the pin is a git tag. The pin is chosen for the **effective-plastic-
   strain** conversion: the original ``Official`` tag carried *no* effective-
   plastic-strain handling at all (the array is simply absent from the
   d3plot); the mapping was introduced in Aug-2024 (commit 2c199e9) and the
   v1.02/v1.021 generation additionally added the *solid* ``EPSFLG`` array,
   the shell stress/strain local->global tensor transforms and the v1.021
   "incorrect assignment of arrays to data fields" global-history bugfix.
   Empirically verified on a real Johnson-Cook run (ball impact, A001-A011):
   ``element_shell_effective_plastic_strain`` is present and non-zero at the
   final state (max ~0.0554). This is also the generation the user already
   vendors for their oropt / openradioss_fatigue tooling.

2. **anim -> VTK**: the OpenRadioss ``anim_to_vtk_win64.exe`` converter — one
   ``.vtk`` per animation file (the exe writes VTK to stdout, redirected to
   ``<anim>.vtk``).

3. **TH (binary Txx) -> CSV**: the OpenRadioss ``th_to_csv_win64.exe``
   converter — the Fortran time-history binary ``…T01`` -> ``…T01.csv``.
   (pyradioss runs already emit the T01 as CSV natively, so this is only for
   a *Fortran* binary time-history file.)

Where the binary converters live
-------------------------------
The two Fortran converter exes are located through
:mod:`pyradioss.paths` — the project's single resource resolver — and never
through a path hardcoded here: the GUI JSON config (``exec_dir``) wins, else
the install prefix the resolver accepts is searched for a ``bin/`` or ``exec/``
subdirectory that actually carries a converter, else the Windows compatibility
path ``C:\\OpenRadioss\\exec``.  When nothing carries one, the converters say
where they looked and what to export (see :func:`exec_dir_error`) instead of
reporting a directory that does not exist.

Streaming
---------
Each streaming converter takes an optional ``emit`` callback receiving event
tuples compatible with :class:`pyradioss.gui.runner.JobRunner`'s queue:

* ``("line", "postproc", text)`` — one progress/output line;
* ``("post_done", kind, ok, outputs, message)`` — a converter finished, where
  ``kind`` is ``"d3plot"`` / ``"vtk"`` / ``"th_csv"``, ``ok`` a bool,
  ``outputs`` the list of produced file paths and ``message`` a short summary.

so the existing Tk log-pane drain handles them unchanged.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from .. import paths

# ---------------------------------------------------------------------------
# Constants (pin + tool locations)
# ---------------------------------------------------------------------------

#: git tag of Vortex-Radioss to pin (not on PyPI). See the module docstring
#: for the effective-plastic-strain rationale.
VORTEX_PIN = "v1.021"

#: the one-liner that installs the pinned Vortex-Radioss into the GUI's env.
VORTEX_INSTALL_HINT = (
    'pip install "git+https://github.com/Vortex-CAE/Vortex-Radioss.git@'
    + VORTEX_PIN + '"  (also needs: pip install lasso-python)')

#: the documented Windows install location of the OpenRadioss Fortran
#: converter exes.
#: Rule 3 of :mod:`pyradioss.paths`' order — the Windows compatibility path,
#: and the documented fallback when no configured prefix carries a converter.
#: It is a *fallback*, not the answer: :func:`resolve_exec_dir` searches the
#: install prefix first, so a user on Linux is never handed a ``C:\`` path when
#: ``$OR_ROOT`` resolves.  See :func:`default_exec_dir`.
_WIN_COMPAT_EXEC_DIR = r"C:\OpenRadioss\exec"
ANIM_TO_VTK_EXE = "anim_to_vtk_win64.exe"
TH_TO_CSV_EXE = "th_to_csv_win64.exe"

#: subdirectories of an OpenRadioss install prefix that carry executables, in
#: the order the rest of the repository searches them: ``bin`` first (where
#: ``tools/oracle/oracle_env.sh`` puts the built ``starter_linux64_gf`` /
#: ``engine_linux64_gf``, and what ``paths.or_starter`` tries first), then
#: ``exec`` — upstream's pre-cmake install layout, which
#: ``$OR_SRC/RELEASES.md:29,52`` names ``exec/th_to_csv_linux64_gf`` for and
#: ``paths.or_starter`` falls back to as ``exec/starter_win64.exe``.
#: ``tools/validate_vs_fortran.py``'s ``th_to_csv`` candidates use these two
#: subdirectories too.
#: A directory counts only if it really carries a converter exe (see
#: :func:`_carries_converter`).
_EXEC_SUBDIRS = ("bin", "exec")

#: the converters this module actually runs, as basenames.  A directory counts
#: as an exec dir only if it holds one of *these* files, because accepting a
#: differently-spelled binary would select a directory this module then cannot
#: run anything from — the same silent degradation as selecting the wrong
#: directory.  For the record, the per-platform Linux spellings upstream also
#: ships (``th_to_csv_linux64_gf``, ``$OR_SRC/RELEASES.md:29,52``) are named in
#: ``tools/validate_vs_fortran.py``'s ``_TH_TO_CSV_NAMES``; running those is a
#: different decision from finding the directory, and is not made here.
_CONVERTER_EXES = (ANIM_TO_VTK_EXE, TH_TO_CSV_EXE)

#: how this resource is named in :func:`paths.missing_resource` messages.
_EXEC_DIR_RESOURCE = ("the OpenRadioss converter exec dir "
                      "(anim_to_vtk / th_to_csv)")

#: subprocess driver that runs the Vortex conversion (mirrors oropt's bridge).
_VORTEX_DRIVER = (
    "import sys; "
    "from vortex_radioss.animtod3plot.Anim_to_D3plot import readAndConvert; "
    "readAndConvert(sys.argv[1], use_shell_mask=True, use_solid_mask=False)"
)

# animation file basename: ``<stem>A001`` … (>=3 digits, no extension).
_ANIM_RE = re.compile(r"^(?P<stem>.+?)A(?P<num>\d{3,})$")
# binary time-history basename: ``<stem>T01`` … (exactly 2 digits, no ext).
_THBIN_RE = re.compile(r"^(?P<stem>.+?)T(?P<num>\d{2})$")

EmitFn = Callable[[tuple], None]


# ---------------------------------------------------------------------------
# Artifact detection
# ---------------------------------------------------------------------------

def find_anim_files(run_dir: str) -> List[str]:
    """Sorted list of OpenRadioss animation files in ``run_dir``.

    Matches basenames of the form ``<stem>A<nnn>`` (>= 3 digits, no
    extension), so ``…A001`` matches while ``…A001.vtk`` / ``…A001.d3plot``
    do not. When several stems are present the largest family wins.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return []
    families: Dict[str, List[str]] = {}
    for name in os.listdir(run_dir):
        full = os.path.join(run_dir, name)
        if not os.path.isfile(full):
            continue
        m = _ANIM_RE.match(name)
        if m:
            families.setdefault(m.group("stem"), []).append(full)
    if not families:
        return []
    best = max(families.values(), key=len)
    best.sort(key=_natural_key)
    return best


def anim_file_stem(run_dir: str) -> Optional[str]:
    """The Vortex ``file_stem`` (full path prefix before ``A<nnn>``) of the
    animation family in ``run_dir``, or ``None`` when there is none. This is
    exactly what ``readAndConvert`` expects (it globs ``stem + 'A*[0-9]'``)."""
    files = find_anim_files(run_dir)
    if not files:
        return None
    m = _ANIM_RE.match(os.path.basename(files[0]))
    if not m:
        return None
    return os.path.join(run_dir, m.group("stem"))


def find_th_binary(run_dir: str) -> Optional[str]:
    """The Fortran binary time-history file ``<stem>T01`` (no extension) in
    ``run_dir``, or ``None``. Prefers ``T01`` if several ``Tnn`` exist."""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    hits: List[str] = []
    for name in os.listdir(run_dir):
        full = os.path.join(run_dir, name)
        if os.path.isfile(full) and _THBIN_RE.match(name):
            hits.append(full)
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.basename(p))
    return hits[0]


def find_port_csv(run_dir: str) -> List[str]:
    """pyradioss-native T01 CSV files (``<stem>T01.csv``) already in
    ``run_dir`` — the port writes these directly, no converter needed."""
    if not run_dir or not os.path.isdir(run_dir):
        return []
    return sorted(glob.glob(os.path.join(run_dir, "*T[0-9][0-9].csv")))


def find_port_vtk(run_dir: str) -> List[str]:
    """VTK files already in ``run_dir`` (the port can emit these natively)."""
    if not run_dir or not os.path.isdir(run_dir):
        return []
    return sorted(glob.glob(os.path.join(run_dir, "*.vtk")))


def find_d3plot(run_dir: str) -> Optional[str]:
    """The master ``.d3plot`` file in ``run_dir`` (the family root, i.e. the
    one whose name ends exactly ``.d3plot``), or ``None``."""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    for name in sorted(os.listdir(run_dir)):
        if name.endswith(".d3plot"):
            full = os.path.join(run_dir, name)
            if os.path.isfile(full):
                return full
    return None


def detect_artifacts(run_dir: str) -> Dict[str, object]:
    """Summarise what is present in ``run_dir`` so the GUI can enable/disable
    each converter. Never raises.

    Keys: ``run_dir``, ``anim_files`` (list), ``anim_stem`` (str|None),
    ``th_binary`` (str|None), ``port_csv`` (list), ``port_vtk`` (list),
    ``d3plot`` (str|None), plus the booleans ``can_d3plot`` (anim present),
    ``can_vtk`` (anim present) and ``can_th_csv`` (a binary Tnn present).
    """
    anim = find_anim_files(run_dir)
    th_bin = find_th_binary(run_dir)
    return {
        "run_dir": run_dir,
        "anim_files": anim,
        "anim_stem": anim_file_stem(run_dir),
        "th_binary": th_bin,
        "port_csv": find_port_csv(run_dir),
        "port_vtk": find_port_vtk(run_dir),
        "d3plot": find_d3plot(run_dir),
        "can_d3plot": bool(anim),
        "can_vtk": bool(anim),
        "can_th_csv": bool(th_bin),
    }


# ---------------------------------------------------------------------------
# Tool-path resolution + command construction (pure, testable)
# ---------------------------------------------------------------------------

def _carries_converter(directory: str) -> bool:
    """True when ``directory`` really holds one of the converter exes.

    The existence predicate every candidate must pass — ``pyradioss.paths``
    holds its own candidates to a resource-specific check for the same reason:
    a wrong-but-existing directory is worse than a missing one, because it
    turns "nothing is configured" into a confident run of the wrong binary.
    Here that would be ``$OR_ROOT/bin`` holding the starter and the engine and
    no converter at all."""
    if not os.path.isdir(directory):
        return False
    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    return any(name in _CONVERTER_EXES for name in entries)


def _or_root_soft() -> Tuple[Optional[str], str]:
    """``(prefix, report)`` from :func:`paths.or_root`; never raises, because
    these candidates must be buildable in order to *report* a failure."""
    try:
        return str(paths.or_root()), ""
    except FileNotFoundError as exc:
        return None, str(exc)


def exec_dir_candidates() -> List[Tuple[str, str]]:
    """``(origin, path)`` for every location searched, in the order
    :mod:`pyradioss.paths` and ``tools/validate_vs_fortran.py`` already use:
    the environment variable first, then the install prefix the resolver
    accepts (sibling-of-build / dev-box / Windows compatibility) in its
    ``bin`` / ``exec`` subdirectories, and the Windows compatibility ``exec/``
    last because it is rule 3 and may never shadow a configured prefix.

    A pair whose origin *is* its path is a candidate that could not even be
    built — the variable is unset, or ``$OR_ROOT`` is unresolved — which is how
    :func:`paths.missing_resource` renders "this one was never on the table".

    Nothing here is a new mechanism: the prefix is :func:`paths.or_root`, and
    every entry is existence-checked before it is used."""
    tried: List[Tuple[str, str]] = []
    env_root = os.environ.get("OR_ROOT")
    if env_root:
        tried.extend((f"env OR_ROOT/{sub}", os.path.join(env_root, sub))
                     for sub in _EXEC_SUBDIRS)
    else:
        label = "env OR_ROOT (not set)"
        tried.append((label, label))
    root, report = _or_root_soft()
    if root is not None:
        tried.extend((f"$OR_ROOT/{sub}", os.path.join(root, sub))
                     for sub in _EXEC_SUBDIRS)
    else:
        head = str(report).splitlines()[0] if report else "no candidate"
        label = f"$OR_ROOT/{{{','.join(_EXEC_SUBDIRS)}}} — OR_ROOT unresolved"
        tried.append((f"{label} ({head})", f"{label} ({head})"))
    tried.append(("Windows compatibility path (pyradioss.paths rule 3)",
                  _WIN_COMPAT_EXEC_DIR))
    return tried


def resolve_exec_dir(explicit: Optional[str] = None) -> Optional[str]:
    """The converter directory to use, or ``None`` when none was found.

    ``explicit`` (the GUI's ``exec_dir`` entry, or a caller's argument) is
    returned verbatim and unvalidated — the override has always been
    authoritative, and the converters report a missing exe for it.  Otherwise
    the candidates of :func:`exec_dir_candidates` are tried in order and the
    first one that carries a converter wins; ``None`` means the caller should
    report :func:`exec_dir_error` rather than silently run a path that is not
    there.

    Never raises: the GUI entry has to render even with nothing configured, and
    :mod:`pyradioss.paths` fails loudly by *listing* what it tried."""
    if explicit:
        return explicit
    for origin, path in exec_dir_candidates():
        if origin == path:
            continue                     # a candidate that was never buildable
        if _carries_converter(path):
            return path
    return None


def default_exec_dir(explicit: Optional[str] = None) -> str:
    """:func:`resolve_exec_dir`, falling back to the documented Windows
    compatibility path — the last-resort string a command is built from when
    no converter was found.  The exe is never executed without an existence
    check, so the fallback can only ever appear in a diagnostic."""
    return resolve_exec_dir(explicit) or _WIN_COMPAT_EXEC_DIR


def exec_dir_error(explicit: Optional[str] = None) -> FileNotFoundError:
    """Build and **return** the diagnostic for an unresolvable converter
    directory: every location :func:`exec_dir_candidates` searched, rendered by
    :func:`paths.missing_resource` exactly as the rest of the repository
    renders a failed resolution, plus the file names looked for and the two
    things a user can set.

    Returned, not raised — the converters never raise (they stream a
    ``post_done`` event instead), so they put ``str(...)`` in the message they
    emit."""
    if explicit:
        return FileNotFoundError(
            f"converter not found under {explicit}: neither "
            + " nor ".join(_CONVERTER_EXES) + " is in that directory.")
    lines = [str(paths.missing_resource(_EXEC_DIR_RESOURCE,
                                        exec_dir_candidates()))]
    lines.append("Each candidate is accepted only if it holds one of "
                 + " / ".join(_CONVERTER_EXES) + " (the exes this module "
                 "runs; see ANIM_TO_VTK_EXE / TH_TO_CSV_EXE).")
    lines.append("Set one of:")
    lines.append("  export OR_ROOT=<the OpenRadioss install prefix> — the "
                 "converters are in its bin/ or exec/ subdirectory;")
    lines.append("  or the exec dir in the GUI config (Post-processing tab).")
    return FileNotFoundError("\n".join(lines))


def exec_path(exec_dir: Optional[str], exe: str) -> str:
    """Full path to a converter exe under ``exec_dir`` (default
    :func:`default_exec_dir`: the resolved directory, else the documented
    Windows compatibility path)."""
    return os.path.join(default_exec_dir(exec_dir), exe)


def __getattr__(name: str):
    """``postproc.DEFAULT_EXEC_DIR`` — kept as this module's public default,
    now *resolved* on every access (PEP 562).

    The GUI pre-fills its exec-dir entry from this attribute
    (``gui/app.py:121-123``), so it must not be a Windows path on a box where
    ``$OR_ROOT`` resolves.  Resolution is deliberately lazy rather than done
    once at import: :mod:`pyradioss.paths` resolves nothing on import (its
    stated import contract), an import-time answer would be frozen before a
    caller could export a variable or call :func:`paths.reload`, and reading
    the attribute is cheap enough for a GUI entry.  Every *other* missing
    attribute still raises ``AttributeError``, as it must."""
    if name == "DEFAULT_EXEC_DIR":
        return default_exec_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def vortex_available() -> bool:
    """True when the Vortex-Radioss package imports in *this* interpreter
    (the same one the GUI launches subprocesses with)."""
    try:
        return importlib.util.find_spec(
            "vortex_radioss.animtod3plot.Anim_to_D3plot") is not None
    except (ImportError, ValueError):
        return False


def d3plot_command(run_dir: str,
                   python_exe: Optional[str] = None) -> Optional[List[str]]:
    """The subprocess command that converts the animation family in
    ``run_dir`` to a d3plot, or ``None`` when there is no family.

    ``[python, -u, -c, <driver>, <stem>]`` — the driver imports Vortex's
    ``readAndConvert`` and runs it on the stem (mirrors oropt's bridge)."""
    stem = anim_file_stem(run_dir)
    if stem is None:
        return None
    return [python_exe or sys.executable, "-u", "-c", _VORTEX_DRIVER, stem]


def anim_to_vtk_commands(
        run_dir: str, exec_dir: Optional[str] = None
) -> List[Tuple[List[str], str]]:
    """One ``(command, output_vtk_path)`` per animation file in ``run_dir``.

    The exe writes VTK to stdout; the runner redirects it to ``<anim>.vtk``.
    """
    exe = exec_path(exec_dir, ANIM_TO_VTK_EXE)
    out: List[Tuple[List[str], str]] = []
    for anim in find_anim_files(run_dir):
        out.append(([exe, anim], anim + ".vtk"))
    return out


def th_to_csv_command(
        run_dir: str, exec_dir: Optional[str] = None
) -> Optional[Tuple[List[str], str]]:
    """``(command, output_csv_path)`` for the binary ``Tnn`` in ``run_dir``,
    or ``None`` when there is no binary time-history. The exe is run *in*
    ``run_dir`` on the basename and writes ``<Tnn>.csv`` next to it."""
    th = find_th_binary(run_dir)
    if th is None:
        return None
    exe = exec_path(exec_dir, TH_TO_CSV_EXE)
    return ([exe, os.path.basename(th)], th + ".csv")


# ---------------------------------------------------------------------------
# Streaming converters
# ---------------------------------------------------------------------------

def _noop_emit(_event: tuple) -> None:
    pass


def _line(emit: EmitFn, text: str) -> None:
    emit(("line", "postproc", text))


def _search_report(explicit: Optional[str]) -> str:
    """The resolution diagnostic to append to a "converter not found"
    message.

    Only when the caller configured *nothing*: with an explicit ``exec_dir``
    the message already names the directory that was asked for, and repeating
    the machine-wide search would only bury it.  This is what turns "no such
    file on this machine" into "here is the search, and here is what to
    export"."""
    if explicit:
        return ""
    return " " + str(exec_dir_error())


def convert_to_d3plot(run_dir: str, emit: Optional[EmitFn] = None,
                      python_exe: Optional[str] = None,
                      timeout: float = 1800.0) -> Dict[str, object]:
    """Convert the animation family in ``run_dir`` to an LS-Dyna d3plot with
    the pinned Vortex-Radioss. Streams progress via ``emit`` and returns a
    result dict ``{ok, outputs, message}``. Never raises.

    Graceful degradation: a missing Vortex install yields ``ok=False`` and a
    message carrying :data:`VORTEX_INSTALL_HINT` (not a traceback).
    """
    emit = emit or _noop_emit
    cmd = d3plot_command(run_dir, python_exe=python_exe)
    if cmd is None:
        return _finish(emit, "d3plot", False, [],
                       "no animation files (…A001, …A002 …) found in "
                       + run_dir)
    if not vortex_available():
        _line(emit, " ** Vortex-Radioss is not installed in this Python.")
        _line(emit, "    Install the pinned version:")
        _line(emit, "    " + VORTEX_INSTALL_HINT)
        return _finish(emit, "d3plot", False, [],
                       "Vortex-Radioss not installed — " + VORTEX_INSTALL_HINT)

    stem = anim_file_stem(run_dir)
    _line(emit, f" anim -> d3plot (Vortex-Radioss {VORTEX_PIN})")
    _line(emit, f"   stem: {stem}")
    rc = _stream_subprocess(cmd, run_dir, emit, timeout)
    d3 = find_d3plot(run_dir)
    outputs = sorted(glob.glob(stem + ".d3plot*")) if stem else []
    ok = rc == 0 and d3 is not None
    msg = (f"wrote {os.path.basename(d3)} (+{max(0, len(outputs) - 1)} state "
           f"files)" if ok else f"conversion failed (rc={rc})")
    return _finish(emit, "d3plot", ok, outputs, msg)


def convert_anim_to_vtk(run_dir: str, exec_dir: Optional[str] = None,
                        emit: Optional[EmitFn] = None,
                        timeout: float = 600.0) -> Dict[str, object]:
    """Convert every animation file in ``run_dir`` to ``<anim>.vtk`` with the
    OpenRadioss ``anim_to_vtk`` exe. Streams a line per file; never raises."""
    emit = emit or _noop_emit
    cmds = anim_to_vtk_commands(run_dir, exec_dir)
    if not cmds:
        return _finish(emit, "vtk", False, [],
                       "no animation files found in " + run_dir)
    exe = exec_path(exec_dir, ANIM_TO_VTK_EXE)
    if not os.path.exists(exe):
        return _finish(emit, "vtk", False, [],
                       f"converter not found: {exe} — set the exec dir in "
                       f"the config" + _search_report(exec_dir))
    outputs: List[str] = []
    _line(emit, f" anim -> VTK ({len(cmds)} file(s), {exe})")
    for cmd, out_path in cmds:
        try:
            with open(out_path, "wb") as fh:
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.PIPE,
                                        cwd=run_dir)
                with _active_procs_lock:
                    _active_procs.append(proc)
                try:
                    _, err_data = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    raise
                finally:
                    with _active_procs_lock:
                        if proc in _active_procs:
                            _active_procs.remove(proc)
        except (OSError, subprocess.SubprocessError) as exc:
            _line(emit, f"   ** {os.path.basename(cmd[-1])}: {exc}")
            continue
        if proc.returncode == 0 and os.path.exists(out_path) \
                and os.path.getsize(out_path) > 0:
            outputs.append(out_path)
            _line(emit, f"   {os.path.basename(cmd[-1])} -> "
                        f"{os.path.basename(out_path)} "
                        f"({os.path.getsize(out_path)} bytes)")
        else:
            err = (err_data or b"").decode("utf-8", "replace").strip()
            _line(emit, f"   ** {os.path.basename(cmd[-1])} failed "
                        f"(rc={proc.returncode}) {err[:200]}")
    ok = len(outputs) == len(cmds)
    return _finish(emit, "vtk", ok, outputs,
                   f"wrote {len(outputs)}/{len(cmds)} VTK file(s)")


def convert_th_to_csv(run_dir: str, exec_dir: Optional[str] = None,
                      emit: Optional[EmitFn] = None,
                      timeout: float = 300.0) -> Dict[str, object]:
    """Convert the binary time-history ``Tnn`` in ``run_dir`` to CSV with the
    OpenRadioss ``th_to_csv`` exe. Streams the exe output; never raises.

    Note: pyradioss runs already emit the T01 as CSV natively, so this only
    applies to a *Fortran* binary time-history file.
    """
    emit = emit or _noop_emit
    built = th_to_csv_command(run_dir, exec_dir)
    if built is None:
        return _finish(emit, "th_csv", False, [],
                       "no binary time-history (…T01) in " + run_dir
                       + " (pyradioss runs already emit T01 as CSV)")
    cmd, out_path = built
    if not os.path.exists(cmd[0]):
        return _finish(emit, "th_csv", False, [],
                       f"converter not found: {cmd[0]} — set the exec dir in "
                       f"the config" + _search_report(exec_dir))
    _line(emit, f" TH -> CSV ({cmd[0]})")
    rc = _stream_subprocess(cmd, run_dir, emit, timeout)
    ok = rc == 0 and os.path.exists(out_path)
    msg = (f"wrote {os.path.basename(out_path)}" if ok
           else f"conversion failed (rc={rc})")
    return _finish(emit, "th_csv", ok, [out_path] if ok else [], msg)


# ---------------------------------------------------------------------------
# Runner (worker thread + queue) for the Post-processing tab
# ---------------------------------------------------------------------------

_active_procs: List[subprocess.Popen] = []
_active_procs_lock = threading.Lock()


class PostProcRunner:
    """Runs a selection of converters on a worker thread, streaming events to
    ``self.queue`` (a :class:`queue.Queue`) so the Tk UI drains them exactly
    as it drains a :class:`~pyradioss.gui.runner.JobRunner`.

    ``actions`` is any subset of ``("d3plot", "vtk", "th_csv")``.
    """

    ALL_ACTIONS = ("d3plot", "vtk", "th_csv")

    def __init__(self, run_dir: str, actions, exec_dir: Optional[str] = None,
                 python_exe: Optional[str] = None, event_queue=None):
        import queue as _queue
        self.run_dir = run_dir
        self.actions = [a for a in actions if a in self.ALL_ACTIONS]
        self.exec_dir = default_exec_dir(exec_dir)
        self.python_exe = python_exe or sys.executable
        self.queue = event_queue or _queue.Queue()
        self.results: Dict[str, Dict[str, object]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("post-processing already running")
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="pyradioss-postproc")
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Terminate child subprocesses and signal worker to stop."""
        self._stop.set()
        with _active_procs_lock:
            procs = list(_active_procs)
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except OSError:
                    pass

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        emit = self.queue.put
        for action in self.actions:
            if self._stop.is_set():
                break
            run_post_actions(self.run_dir, [action], exec_dir=self.exec_dir,
                             emit=emit, python_exe=self.python_exe,
                             results=self.results)
        emit(("post_all_done", dict(self.results)))

    def run_to_completion(self, timeout: float = 1800.0
                          ) -> Dict[str, Dict[str, object]]:
        """Head-less convenience: run the actions on this thread and return
        the per-action result dicts (for tests / scripted callers)."""
        run_post_actions(self.run_dir, self.actions, exec_dir=self.exec_dir,
                         emit=self.queue.put, python_exe=self.python_exe,
                         results=self.results)
        return self.results


def run_post_actions(run_dir: str, actions, exec_dir: Optional[str] = None,
                     emit: Optional[EmitFn] = None,
                     python_exe: Optional[str] = None,
                     results: Optional[Dict[str, Dict[str, object]]] = None
                     ) -> Dict[str, Dict[str, object]]:
    """Run the selected converters in order, collecting per-action results.

    Shared by :class:`PostProcRunner` (the Post-processing tab) and the
    :class:`~pyradioss.gui.runner.JobRunner` auto-convert-after-run hook, so
    both paths stream identical events. Never raises.
    """
    emit = emit or _noop_emit
    results = results if results is not None else {}
    for action in actions:
        if action == "d3plot":
            results["d3plot"] = convert_to_d3plot(
                run_dir, emit=emit, python_exe=python_exe)
        elif action == "vtk":
            results["vtk"] = convert_anim_to_vtk(
                run_dir, exec_dir=exec_dir, emit=emit)
        elif action == "th_csv":
            results["th_csv"] = convert_th_to_csv(
                run_dir, exec_dir=exec_dir, emit=emit)
    return results


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _natural_key(path: str):
    """Sort key that orders ``A2`` before ``A10`` (natural numeric order)."""
    base = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok
            for tok in re.split(r"(\d+)", base)]


def _finish(emit: EmitFn, kind: str, ok: bool, outputs: List[str],
            message: str) -> Dict[str, object]:
    emit(("post_done", kind, ok, list(outputs), message))
    return {"ok": ok, "outputs": list(outputs), "message": message}


def _stream_subprocess(cmd: List[str], cwd: str, emit: EmitFn,
                       timeout: float) -> int:
    """Run ``cmd`` streaming combined stdout/stderr line-by-line via ``emit``.
    Returns the exit code (or a negative sentinel on a launch/timeout error).
    """
    try:
        # decode as UTF-8, never fatally: the Vortex bridge's tqdm bar emits
        # UTF-8 block characters (▍ = ..0x8d) that the Windows locale codec
        # (cp1252) cannot decode — with text=True alone the stream loop died
        # in UnicodeDecodeError mid-conversion whenever a fractional block
        # happened to be on screen at a flush
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=cwd)
    except OSError as exc:
        _line(emit, f"   ** failed to launch: {exc}")
        return -1

    with _active_procs_lock:
        _active_procs.append(proc)

    q: queue.Queue[Optional[str]] = queue.Queue()

    def _reader():
        try:
            if proc.stdout is not None:
                for raw in proc.stdout:
                    q.put(raw)
        finally:
            q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    deadline = time.time() + timeout
    timed_out = False

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            timed_out = True
            break
        try:
            item = q.get(timeout=min(remaining, 0.1))
            if item is None:
                break
            line = item.rstrip("\n")
            if line.strip():
                _line(emit, "   " + line)
        except queue.Empty:
            if proc.poll() is not None:
                # Process terminated, drain queue
                while True:
                    try:
                        item = q.get_nowait()
                        if item is None:
                            break
                        line = item.rstrip("\n")
                        if line.strip():
                            _line(emit, "   " + line)
                    except queue.Empty:
                        break
                break

    if timed_out:
        proc.kill()
        _line(emit, "   ** timed out")
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
        rc = -9
    else:
        try:
            rc = proc.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            _line(emit, "   ** timed out")
            rc = -9

    try:
        t.join(timeout=2.0)
    except Exception:
        pass
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except OSError:
        pass

    with _active_procs_lock:
        if proc in _active_procs:
            _active_procs.remove(proc)

    return rc
