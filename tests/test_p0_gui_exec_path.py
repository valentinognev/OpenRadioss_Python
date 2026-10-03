"""Task P0.18 — the GUI's Fortran-converter exec dir must resolve through
``pyradioss.paths``, not through a Windows install that is not here.

``pyradioss.gui.postproc.DEFAULT_EXEC_DIR`` was the literal
``C:\\OpenRadioss\\exec``: a directory that **does not exist on this box**, and
the module offered no way out.  A user here who pressed *anim -> VTK* got
``converter not found: C:\\OpenRadioss\\exec\\anim_to_vtk_win64.exe — set the
exec dir in the config``, which names a directory on a machine they are not on
and does not say where the exes actually are.  The same default is what
``pyradioss/gui/app.py:121-123`` pre-fills into the GUI entry, so the broken
value was visible before any button was pressed.

The project already solved this problem once, properly, in
``pyradioss.paths`` (task P0.6, ``plan/00_ORCHESTRATION.md`` §4.1): env
variable if set and existing -> sibling-of-build layout -> the Windows
compatibility path -> **fail loudly** through ``paths.missing_resource``, which
lists every attempted location.  This module must *consume* that resolver, not
add a fourth mechanism of its own.  Three consequences are pinned here:

* the converter directory is found under a resolved install prefix
  (``$OR_ROOT/bin`` — where ``tools/oracle/oracle_env.sh`` puts the built
  binaries — and ``$OR_ROOT/exec``, upstream's pre-cmake install layout), so a
  configured Linux box stops being told about ``C:\\``;
* a prefix that exists but carries **no** converter exe is rejected, because
  a wrong-but-existing directory is worse than a missing one (the rule
  ``pyradioss.paths`` applies to every candidate);
* the Windows form still wins when it is the only thing that exists — it is
  rule 3 of the resolver's order, and the maintainer has a Windows box —
  while nothing resolving at all fails with a message that names what to set.

Deliberately head-less and cheap: ``pyradioss.gui.postproc`` imports no
``tkinter`` (that is the module's stated design rule, and ``pyradioss/gui/
__init__.py`` imports none either), so the GUI-facing default —
``postproc.DEFAULT_EXEC_DIR``, the attribute ``app.py`` reads — is assertable
without constructing a ``Tk()`` root or a display.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyradioss import paths
from pyradioss.gui import postproc as PP

_SCRUBBED = (
    "OR_SRC", "OR_ROOT", "OR_BUILD", "OR_STARTER", "OR_ENGINE",
    "PYRADIOSS_HM_CFG", "PYRADIOSS_RD_DECKS", "RAD_CFG_PATH",
    "OPENRADIOSS_PATH",
)


def _fake_converter(directory: Path, name: str = "anim_to_vtk_win64.exe",
                    body: str = "") -> Path:
    """A runnable stand-in for one of the converter exes in ``directory``.

    The name is a pinned Windows exe by default because that is what this
    module runs (upstream ships the converters prebuilt); a POSIX shell script
    wearing the name is the only way to exercise the resolution end-to-end
    here, and the module never inspects anything but the basename."""
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / name
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    return exe


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No inherited OpenRadioss environment, an empty resolver cache and a
    neutral dev-box prefix — so no machine this repo does not control can
    decide the outcome (same discipline as ``tests/test_p0_paths.py``)."""
    for var in _SCRUBBED:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(paths, "_dev_or_root",
                        lambda: Path("/nonexistent/prefix"))
    paths.reload()
    yield
    paths.reload()


# ---------------------------------------------------------------------------
# (1) a configured install prefix is used instead of the Windows default
# ---------------------------------------------------------------------------

def test_a_configured_prefix_resolves_the_converter_dir(monkeypatch, tmp_path):
    """``OR_ROOT`` set to a prefix whose ``exec/`` carries the converter: the
    module must build its command there, not under ``C:\\OpenRadioss\\exec``.

    This is the finding: on this box the Windows directory does not exist, and
    before the fix the command was built there regardless of what the
    environment said."""
    root = tmp_path / "prefix"
    _fake_converter(root / "exec")
    monkeypatch.setenv("OR_ROOT", str(root))

    assert PP.resolve_exec_dir() == str(root / "exec")
    assert PP.exec_path(None, PP.ANIM_TO_VTK_EXE) == str(
        root / "exec" / PP.ANIM_TO_VTK_EXE)
    assert PP.exec_path(None, PP.TH_TO_CSV_EXE) == str(
        root / "exec" / PP.TH_TO_CSV_EXE)


def test_the_bin_subdirectory_of_the_prefix_is_searched_too(
        monkeypatch, tmp_path):
    """``tools/oracle/oracle_env.sh`` puts the built binaries in
    ``$OR_ROOT/bin``; an installation that carries the converters beside them
    must resolve just as the pre-cmake ``exec/`` layout does."""
    root = tmp_path / "prefix"
    _fake_converter(root / "bin", name=PP.TH_TO_CSV_EXE)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert PP.resolve_exec_dir() == str(root / "bin")


def test_the_gui_default_entry_is_the_resolved_directory(monkeypatch,
                                                         tmp_path):
    """``pyradioss/gui/app.py:121-123`` pre-fills its exec-dir entry with
    ``postproc.DEFAULT_EXEC_DIR``, so that attribute IS the user-facing
    default. It must carry the resolved directory, not a path on another
    machine."""
    root = tmp_path / "prefix"
    _fake_converter(root / "exec")
    monkeypatch.setenv("OR_ROOT", str(root))
    assert PP.DEFAULT_EXEC_DIR == str(root / "exec")


# ---------------------------------------------------------------------------
# (2) a wrong-but-existing directory is worse than a missing one
# ---------------------------------------------------------------------------

def test_a_prefix_without_the_converter_exe_is_rejected(monkeypatch, tmp_path):
    """``$OR_ROOT/bin`` holding only the starter and the engine is *not* an
    exec dir. Selecting it would report the same "converter not found" while
    claiming a resolution had happened — the silent-degradation failure
    ``pyradioss.paths`` exists to remove."""
    root = tmp_path / "prefix"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "starter_linux64_gf").write_text("")
    (root / "bin" / "engine_linux64_gf").write_text("")
    monkeypatch.setenv("OR_ROOT", str(root))

    assert PP.resolve_exec_dir() is None
    assert PP.resolve_exec_dir(str(root / "bin")) == str(root / "bin"), (
        "an explicitly configured directory stays authoritative and "
        "unvalidated — the GUI's own exec_dir entry has always been honoured "
        "verbatim, and the converters report a missing exe for it")


def test_an_explicitly_configured_directory_wins_over_resolution(tmp_path):
    """The GUI config / caller override is rule 1 of the search and is not
    re-validated (unchanged behaviour: ``test_m41_gui_post.py`` passes
    ``exec_dir="D:/x"`` and expects that string in the command)."""
    assert PP.resolve_exec_dir("D:/tools") == "D:/tools"
    assert PP.exec_path("D:/tools", PP.ANIM_TO_VTK_EXE) == os.path.join(
        "D:/tools", PP.ANIM_TO_VTK_EXE)


# ---------------------------------------------------------------------------
# (3) Windows compatibility: rule 3, and it must still win
# ---------------------------------------------------------------------------

def test_the_windows_compatibility_path_still_resolves(monkeypatch, tmp_path):
    """The maintainer's Windows box has ``C:\\OpenRadioss\\exec`` and nothing
    else; that is exactly rule 3 of ``pyradioss.paths``'s order. Point the
    compatibility candidate at a real directory and it must be selected — the
    check is done against the filesystem, never against the drive letter, so
    it is testable here."""
    win_exec = tmp_path / "OpenRadioss" / "exec"
    _fake_converter(win_exec)
    monkeypatch.setattr(PP, "_WIN_COMPAT_EXEC_DIR", str(win_exec))

    assert PP.resolve_exec_dir() == str(win_exec)
    assert PP.exec_path(None, PP.ANIM_TO_VTK_EXE) == str(
        win_exec / PP.ANIM_TO_VTK_EXE)
    assert PP.DEFAULT_EXEC_DIR == str(win_exec)


def test_the_windows_compatibility_prefix_resolves_through_paths_or_root(
        monkeypatch, tmp_path):
    """Rule 3 the way the resolver serves it: with nothing exported,
    ``paths.or_root()`` returns the Windows install prefix itself, and the
    converters must be found under its ``exec/``.  ``paths._WIN_ROOT`` is the
    documented seam for exercising that rule off Windows (``pyradioss.paths``
    says so on the constant), which is why this case is testable here at all —
    without it, "Windows keeps working" would be an unverifiable claim."""
    win_root = tmp_path / "OpenRadioss"
    _fake_converter(win_root / "exec")
    monkeypatch.setattr(paths, "_WIN_ROOT", win_root)

    assert paths.or_root() == win_root, "the rule-3 seam did not take"
    assert PP.resolve_exec_dir() == str(win_root / "exec")
    assert PP.DEFAULT_EXEC_DIR == str(win_root / "exec")


def test_a_configured_prefix_wins_over_the_windows_compatibility_path(
        monkeypatch, tmp_path):
    """Order is the resolver's: a configured prefix is rule 1, the Windows
    candidate is rule 3, so it can never shadow it."""
    win_exec = tmp_path / "OpenRadioss" / "exec"
    _fake_converter(win_exec)
    monkeypatch.setattr(PP, "_WIN_COMPAT_EXEC_DIR", str(win_exec))
    root = tmp_path / "prefix"
    _fake_converter(root / "exec")
    monkeypatch.setenv("OR_ROOT", str(root))
    assert PP.resolve_exec_dir() == str(root / "exec")


# ---------------------------------------------------------------------------
# (4) nothing resolves -> fail loudly, naming what to set
# ---------------------------------------------------------------------------

def test_nothing_resolving_raises_nothing_and_says_what_to_set(
        monkeypatch, tmp_path):
    """``resolve_exec_dir`` returns ``None`` (it never raises: the GUI entry
    must still render) and ``exec_dir_error`` carries the diagnostic built by
    ``paths.missing_resource`` — every attempted location plus a hint."""
    root = tmp_path / "prefix"
    root.mkdir()
    monkeypatch.setenv("OR_ROOT", str(root))

    assert PP.resolve_exec_dir() is None
    message = str(PP.exec_dir_error())
    assert "OR_ROOT" in message, message
    assert str(root / "exec") in message, message
    assert str(root / "bin") in message, message
    assert isinstance(PP.exec_dir_error(), FileNotFoundError)


def test_the_converter_reports_the_search_not_a_windows_path_only(
        monkeypatch, tmp_path):
    """The end the user actually reaches: *anim -> VTK* with nothing
    configured must say where it looked and what to set, while keeping the
    existing "converter not found" wording other callers rely on."""
    root = tmp_path / "prefix"
    root.mkdir()
    monkeypatch.setenv("OR_ROOT", str(root))

    run = tmp_path / "run"
    run.mkdir()
    (run / "MODELA001").write_bytes(b"fake-anim")

    res = PP.convert_anim_to_vtk(str(run))
    assert res["ok"] is False
    assert "converter not found" in res["message"]
    assert "OR_ROOT" in res["message"]
    assert str(root / "exec") in res["message"]


def test_the_converter_runs_the_resolved_executable(monkeypatch, tmp_path):
    """The strongest form: the command the module builds is the one that gets
    executed. The stand-in converter copies its input to stdout, which the
    runner redirects to ``<anim>.vtk`` — so a ``ok=True`` here cannot happen
    unless the resolved exe was really invoked."""
    root = tmp_path / "prefix"
    _fake_converter(root / "exec", name=PP.ANIM_TO_VTK_EXE, body='cat "$1"\n')
    monkeypatch.setenv("OR_ROOT", str(root))

    run = tmp_path / "run"
    run.mkdir()
    (run / "MODELA001").write_bytes(b"fake-anim")

    res = PP.convert_anim_to_vtk(str(run))
    assert res["ok"] is True, res["message"]
    assert (run / "MODELA001.vtk").read_bytes() == b"fake-anim"