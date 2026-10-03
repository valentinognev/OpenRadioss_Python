"""Task P0.9 — ``tools/validate_vs_fortran.py`` must run on a Linux dev box.

The differential-validation harness is the evidence channel every later phase
needs ("numerics changes REQUIRE validation evidence"), and it was pinned to
one Windows machine: the oracle prefix, the Intel oneAPI runtime and the three
executables were module-scope string constants, so importing the module named
a path that does not exist anywhere else.  Nothing about the *comparison* is
changed here — the verdict thresholds, the significance rule and every verdict
string in ``tools/validation_data/parity_m41.json`` are untouched, because
that file's comparability depends on them.

Upstream Fortran / environment origins cited by the harness and pinned here
(repo-relative under the ``OpenCourant`` tree, i.e. ``$OR_SRC``):

* ``INSTALL.md:34-42`` — "Environment variables settings under Linux":
  ``OPENRADIOSS_PATH``, ``RAD_CFG_PATH=$OPENRADIOSS_PATH/hm_cfg_files``,
  ``RAD_H3D_PATH``, ``OMP_STACKSIZE=400m`` and
  ``LD_LIBRARY_PATH=$OPENRADIOSS_PATH/extlib/hm_reader/linux64/``.  This is
  the environment block the harness has to reproduce on whatever platform it
  runs on; ``tools/oracle/oracle_env.sh`` is the Linux spelling that was
  measured on this box.
* ``INSTALL.md:110-111`` — the invocation: ``./starter_linux64_gf -i deck
  -np 1`` and ``./engine_linux64_gf -i deck``.  The engine line carries **no**
  ``-np``; measured on this box the engine's argument parser does not accept
  it (usage dump, then SIGSEGV), so the harness passes ``-nt 1`` and nothing
  else.  Recorded in
  ``tools/validation_data/oracle_provenance.json`` ``invocation.warning``.
* ``RELEASES.md:29,52`` / ``:72,106`` — the installed names of the T01
  converter, ``exec/th_to_csv_linux64_gf`` and ``exec/th_to_csv_win64.exe``.
* ``tools/th_to_csv/README.md:1-7`` — the pinned tree carries only a README:
  the converter's **source** moved to the separate ``OpenRadioss/Tools``
  repository, and no ``starter``/``engine`` CMake target builds it
  (``Apptainer/openradioss.def:31-32`` builds it from its own checkout).
  So it is resolved as an OPTIONAL key and reported honestly when absent.
* ``tools/validation_data/oracle_provenance.json``
  ``admissible_parity_evidence`` — which channels a parity claim may rest on
  (T01, A-files, RESTART, the starter ``.out``, the energy ledger) and which
  it may not (H3D, native ``.k`` reading, ``/ALE/STRUCTURED_MESH``,
  ``/CHECKSUM_REPORT`` over H3D).  H3D is *refused* rather than merely
  excluded, which is why ``fortran_env()`` deletes ``RAD_H3D_PATH`` instead
  of setting it.

What is asserted, and why each shape
------------------------------------
1. **import is free** — on Linux, with every ``OR_*`` / ``RAD_*`` /
   ``PYRADIOSS_*`` variable scrubbed, and with ``pathlib.Path``'s
   existence checks booby-trapped, so "importing the harness resolves
   nothing" is enforced rather than asserted in prose.  The module-scope
   toolchain constants must be gone (``OR_ROOT``, ``ONEAPI``,
   ``STARTER_EXE``, ``ENGINE_EXE``, ``TH2CSV_EXE``).
2. **no Windows-only default survives** — the harness names **no drive
   letter at all**; and no ``tools/*.py`` source keeps the oracle install
   prefix or the Intel oneAPI runtime as a live literal.  Deliberately *not*
   asserted over all of ``tools/``: the recorded evidence under
   ``tools/validation_data/`` is a historical Windows-box record and must not
   be rewritten, ``tools/run_reference_or.ps1`` is the Windows reference-box
   runner whose defaults *are* that box, and ``tools/lspp_check.py`` lists
   LS-PrePost install paths — a product with no Linux build, already resolved
   through ``LSPP_EXE`` first.  The exclusions are named here so the scope of
   the claim is visible rather than implied.
3. **resolution works where the oracle is** — ``oracle_paths()`` returns the
   five keys, and on a box with the oracle built it returns the real
   binaries (existence *and* the executable bit).  Skips when the oracle is
   absent; fails under ``PYRADIOSS_ORACLE_REQUIRED=1`` (the Phase 0 gate).
4. **honest degradation** — with an empty install prefix, ``oracle_paths()``
   returns ``None`` for the starter, warns with
   ``pyradioss.paths.missing_resource``'s full candidate list, and
   ``strict=True`` raises it; ``parity()`` then returns 2 having written no
   ``parity_results.json``, so a missing oracle can never masquerade as an
   empty comparison.
5. **the invocation is right** — the argv the starter is given carries
   ``-np 1``, the argv the engine is given does not, and
   :func:`run_fortran` is pinned end-to-end (with the subprocess boundary
   stubbed) so the asymmetry cannot be reintroduced in the driver.
6. **the runtime environment is upstream's** — ``RAD_CFG_PATH``,
   ``LD_LIBRARY_PATH`` (the native-``.k`` reader and its APR dependency, which
   the oracle cannot start without), ``OMP_STACKSIZE`` and ``OMP_NUM_THREADS``
   are set; the path separator is the platform's; and ``RAD_H3D_PATH`` is
   absent **even when the caller's environment exports it**, because the
   reachable writer is ABI-incompatible and a run with it set writes silently
   wrong H3D files.
7. **the manifest API is untouched** — task P0.8 owns
   ``load_manifest`` / ``resolve_manifest_record`` and this task must not
   change them; their signatures are pinned so a refactor here cannot quietly
   move them.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from pyradioss import paths

REPO = Path(paths.__file__).resolve().parents[1]
HARNESS = REPO / "tools" / "validate_vs_fortran.py"
TOOLS = REPO / "tools"

#: Every variable the resolvers consult; scrubbed for the bare-import test so
#: "works with nothing configured" means literally nothing.
ORACLE_ENV_VARS = (
    "OR_SRC", "OR_ROOT", "OR_BUILD", "OR_STARTER", "OR_ENGINE",
    "PYRADIOSS_HM_CFG", "PYRADIOSS_RD_DECKS", "RAD_CFG_PATH",
    "RAD_H3D_PATH", "OPENRADIOSS_PATH", "OR_TH_TO_CSV",
)

ORACLE_REQUIRED = os.environ.get("PYRADIOSS_ORACLE_REQUIRED") == "1"
ORACLE_DISABLED = os.environ.get("PYRADIOSS_ORACLE_DISABLED") == "1"

_HINT = ("run tools/oracle/mirror_and_fetch.sh then "
         "tools/oracle/build_oracle.sh")

#: The five keys the brief's interface fixes.
ORACLE_KEYS = ("starter", "engine", "th_to_csv", "h3d_lib", "hm_reader_lib")

#: Live Windows defaults that made the harness Windows-only.  Matched
#: case-insensitively, with either separator, because the old module mixed
#: ``C:\``, ``C:/`` and the escaped docstring spelling.
WINDOWS_ORACLE_LITERALS = (
    r"c:[\\/]openradioss",
    r"intel[\\/]oneapi",
    r"program files \(x86\)",
)

#: The exact assertions the brief's Step 1 makes, kept verbatim in spirit.
FORBIDDEN_IN_HARNESS = (r"C:\OpenRadioss", "Intel\\\\oneAPI")


def _require_oracle():
    """Skip when the oracle is absent; fail when the Phase 0 gate demands it."""
    from tools import validate_vs_fortran as V
    resolved = V.oracle_paths()
    if resolved["starter"] and resolved["engine"]:
        return
    if ORACLE_DISABLED:
        pytest.skip("PYRADIOSS_ORACLE_DISABLED=1")
    if ORACLE_REQUIRED:
        pytest.fail(f"oracle binaries missing and PYRADIOSS_ORACLE_REQUIRED=1 "
                     f"is set: {_HINT}")
    pytest.skip(f"oracle not built ({_HINT}); set "
                f"PYRADIOSS_ORACLE_REQUIRED=1 to enforce")


@pytest.fixture
def bare_env(monkeypatch):
    """No oracle/corpus configuration at all, and an empty resolver cache.

    The deletions go through ``monkeypatch`` so the caller's environment is
    restored — writing ``os.environ`` here would leak a dead ``OR_ROOT`` into
    every later test in the session, and the next test would then "prove" the
    oracle is missing for the wrong reason.
    """
    for var in ORACLE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    paths.reload()
    yield
    paths.reload()


# --------------------------------------------------------------------------
# 1. importing the harness resolves nothing
# --------------------------------------------------------------------------

def test_harness_imports_with_no_environment_set():
    """A fresh interpreter, every variable scrubbed, and the import succeeds.

    Run in a subprocess so the assertion is about *import* and not about what
    another test already put in ``sys.modules``.
    """
    env = {k: v for k, v in os.environ.items() if k not in ORACLE_ENV_VARS}
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [sys.executable, "-c", "import tools.validate_vs_fortran as V; "
                               "print(V.__name__)"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "tools.validate_vs_fortran" in proc.stdout


def test_harness_import_touches_no_filesystem():
    """Import must not stat a resource — resolution is lazy and per-call.

    ``pathlib``'s existence predicates are replaced by a raiser, so any
    resolver work at import time fails the test instead of merely being slow.
    numpy and ``pyradioss.paths`` are imported first: both are imported by the
    harness at module scope and do their own (unavoidable) ``__file__``
    bookkeeping, which is not what this test is about.
    """
    env = {k: v for k, v in os.environ.items() if k not in ORACLE_ENV_VARS}
    env["PYTHONPATH"] = str(REPO)
    code = textwrap.dedent(
        """
        import numpy                       # noqa: F401
        import pyradioss.paths             # noqa: F401
        import pathlib

        class Touched(Exception):
            pass

        def _boom(*a, **k):
            raise Touched("the harness touched the filesystem at import")

        for _name in ("exists", "is_file", "is_dir", "stat", "iterdir", "glob"):
            setattr(pathlib.Path, _name, _boom)

        import tools.validate_vs_fortran as V
        print("imported", V.__name__)
        """)
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "imported tools.validate_vs_fortran" in proc.stdout


def test_harness_has_no_module_scope_toolchain_constants():
    """``OR_ROOT`` / ``ONEAPI`` / the three exes must be gone, not renamed.

    A module-scope constant is resolved once, at import, on whatever box
    imported first; that is exactly the failure this task removes.
    """
    from tools import validate_vs_fortran as V
    for name in ("OR_ROOT", "ONEAPI", "STARTER_EXE", "ENGINE_EXE", "TH2CSV_EXE",
                 "OR_SRC", "ONEAPI_ROOT"):
        assert not hasattr(V, name), (
            f"{name} is still a module-scope constant of the harness")


# --------------------------------------------------------------------------
# 2. no Windows-only default survives
# --------------------------------------------------------------------------

def test_harness_names_no_drive_letter():
    """No drive letter anywhere in the harness — code, f-string or comment.

    Stronger than the brief's two literals on purpose: a default hidden in a
    comment or an f-string is still a default somebody will believe.
    """
    src = HARNESS.read_text(encoding="utf-8")
    for lit in FORBIDDEN_IN_HARNESS:
        assert lit not in src, f"{lit!r} survived in {HARNESS}"
    hits = sorted(set(re.findall(r"[A-Za-z]:[\\/]", src)))
    assert hits == [], (
        f"{HARNESS.name} still spells a Windows drive letter at {hits}; resolve "
        "it through pyradioss.paths instead")


def test_no_tools_python_source_keeps_a_windows_oracle_or_toolchain_default():
    """No ``tools/*.py`` may keep the oracle prefix / oneAPI as a live literal.

    Scoped to the oracle and the Intel toolchain — see the module docstring
    for what is deliberately out of scope (recorded evidence, the PowerShell
    reference-box runner, the LS-PrePost candidate list).
    """
    offenders = []
    for src in sorted(TOOLS.rglob("*.py")):
        text = src.read_text(encoding="utf-8", errors="replace")
        for pattern in WINDOWS_ORACLE_LITERALS:
            if re.search(pattern, text, re.I):
                offenders.append(f"{src.relative_to(REPO)}: {pattern}")
    assert offenders == [], (
        "Windows oracle/toolchain literals left in tools/: " + "; ".join(offenders))


# --------------------------------------------------------------------------
# 3. resolution works where the oracle is
# --------------------------------------------------------------------------

def test_oracle_paths_returns_the_built_binaries():
    _require_oracle()
    from tools import validate_vs_fortran as V
    resolved = V.oracle_paths()
    assert tuple(resolved) == ORACLE_KEYS
    for key in ("starter", "engine"):
        got = resolved[key]
        assert got is not None, f"{key} did not resolve: {V.ORACLE_KEYS}"
        assert Path(got).is_file(), f"{key} -> {got} is not a file"
        assert os.access(got, os.X_OK), f"{key} -> {got} is not executable"
    assert Path(resolved["starter"]).name.startswith(
        ("starter_linux64", "starter_win64"))
    assert Path(resolved["engine"]).name.startswith(
        ("engine_linux64", "engine_win64"))
    # hm_reader is what the binaries cannot start without
    # (tools/oracle/oracle_env.sh: LD_LIBRARY_PATH is load-bearing).
    assert Path(resolved["hm_reader_lib"]).is_file()


def test_oracle_paths_follows_the_paths_resolver_not_its_own_default():
    """``starter``/``engine`` come from ``pyradioss.paths``, not a second copy.

    Two independent resolvers would be two orderings to keep in step; the
    harness must be a consumer of ``pyradioss.paths``, never a rival.
    """
    _require_oracle()
    from tools import validate_vs_fortran as V
    resolved = V.oracle_paths()
    assert Path(resolved["starter"]) == paths.or_starter()
    assert Path(resolved["engine"]) == paths.or_engine()


# --------------------------------------------------------------------------
# 4. honest degradation
# --------------------------------------------------------------------------

def test_oracle_paths_says_loudly_when_a_binary_is_missing(tmp_path, bare_env,
                                                           monkeypatch):
    """Absent starter -> ``None`` + every attempted location, not an exception.

    The diagnostic has to be ``pyradioss.paths.missing_resource``'s, because
    that is the one that knows the contract's candidate order; a bespoke
    "not found" string here would be a second, drifting version of it.
    """
    from tools import validate_vs_fortran as V
    empty = tmp_path / "empty_prefix"
    empty.mkdir()
    monkeypatch.setenv("OR_ROOT", str(empty))
    monkeypatch.setenv("OR_STARTER", str(empty / "bin" / "starter_linux64_gf"))
    monkeypatch.setenv("OR_ENGINE", str(empty / "bin" / "engine_linux64_gf"))
    paths.reload()

    with pytest.warns(RuntimeWarning) as caught:
        resolved = V.oracle_paths()
    assert resolved["starter"] is None
    assert resolved["engine"] is None
    messages = "\n".join(str(w.message) for w in caught)
    assert "OR_STARTER not found" in messages
    assert "Tried:" in messages
    assert str(empty) in messages, "the diagnostic does not name what it tried"

    with pytest.raises(FileNotFoundError) as raised:
        V.oracle_paths(strict=True)
    assert "OR_STARTER not found" in str(raised.value)


def test_parity_refuses_to_report_without_an_oracle(tmp_path, bare_env,
                                                    capsys, monkeypatch):
    """No oracle -> exit 2, loud message, and NO ``parity_results.json``.

    This is the "must not silently produce an empty comparison" rule: a
    results file whose rows carry no comparison would read as evidence.
    """
    from tools import validate_vs_fortran as V
    empty = tmp_path / "empty_prefix"
    empty.mkdir()
    monkeypatch.setenv("OR_ROOT", str(empty))
    monkeypatch.setenv("OR_STARTER", str(empty / "bin" / "starter_linux64_gf"))
    monkeypatch.setenv("OR_ENGINE", str(empty / "bin" / "engine_linux64_gf"))
    paths.reload()

    args = argparse_namespace(workdir=str(tmp_path / "wd"), only=None,
                              budget=10.0, timeout=10.0, tol=0.05,
                              shim="translate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = V.parity(args)
    captured = capsys.readouterr()
    assert rc == 2
    assert "OR_STARTER not found" in captured.err
    assert not (tmp_path / "wd" / "parity_results.json").exists()


# --------------------------------------------------------------------------
# 5. the invocation
# --------------------------------------------------------------------------

def test_starter_takes_np_and_the_engine_does_not():
    """``INSTALL.md:110`` gives the starter ``-np 1`` and the engine nothing.

    The engine *does* take ``-nt``; what it must never be given is ``-np``,
    which its argument parser rejects with a usage dump and then a SIGSEGV —
    indistinguishable, in a batch log, from a broken oracle.
    """
    from tools import validate_vs_fortran as V
    starter = V.starter_argv("/opt/or/bin/starter_linux64_gf", "CASE_0000.rad")
    engine = V.engine_argv("/opt/or/bin/engine_linux64_gf", "CASE_0001.rad")
    assert starter == ["/opt/or/bin/starter_linux64_gf", "-i", "CASE_0000.rad",
                       "-np", "1", "-nt", "1"]
    assert engine == ["/opt/or/bin/engine_linux64_gf", "-i", "CASE_0001.rad",
                      "-nt", "1"]
    assert "-np" not in engine


def test_run_fortran_uses_those_argv(monkeypatch, tmp_path, bare_env):
    """Pin the driver, not just the helpers: starter ``-np 1``, engine no ``-np``.

    The subprocess boundary is stubbed, so no solver runs; the fake
    ``run_cmd`` creates exactly the artefacts each stage looks for, which is
    what lets the whole three-stage chain be walked.
    """
    from tools import validate_vs_fortran as V
    case = tmp_path / "case"
    case.mkdir()
    deck0 = case / "CASE_0000.rad"
    deck1 = case / "CASE_0001.rad"
    deck0.write_text("#RADIOSS STARTER\n/TITLE\nCASE\n", encoding="utf-8")
    deck1.write_text("#RADIOSS ENGINE\n/STOP\n", encoding="utf-8")

    oracle = {"starter": str(tmp_path / "starter"), "engine": str(tmp_path / "engine"),
              "th_to_csv": str(tmp_path / "th_to_csv"),
              "h3d_lib": None, "hm_reader_lib": None}
    monkeypatch.setattr(V, "oracle_paths", lambda *a, **k: dict(oracle))
    for value in (oracle["starter"], oracle["engine"], oracle["th_to_csv"]):
        p = Path(value)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    seen = []

    def fake_run_cmd(cmd, cwd, timeout, env=None):
        seen.append(list(cmd))
        run = "CASE"
        if cmd[0] == oracle["starter"]:
            (Path(cwd) / f"{run}_0000.out").write_text(
                "     NUMNOD: NUMBER OF NODAL POINTS. . .    8\n"
                "     ELAPSED TIME...........=  0.10 s\n", encoding="utf-8")
            (Path(cwd) / f"{run}_0000_0001.rst").write_text("rst", encoding="utf-8")
        elif cmd[0] == oracle["engine"]:
            (Path(cwd) / f"{run}_0001.out").write_text(
                "  TOTAL NUMBER OF CYCLES  :    100\n"
                "  ELAPSED TIME...........=  0.20 s\n"
                "  NORMAL TERMINATION\n", encoding="utf-8")
            (Path(cwd) / f"{run}T01").write_bytes(b"\x00\x01")
        else:                                     # th_to_csv
            (Path(cwd) / f"{run}T01.csv").write_text(
                "TIME,INTERNAL ENERGY\n0.0,1.0\n", encoding="utf-8")
        return 0, "NORMAL TERMINATION", 0.01

    monkeypatch.setattr(V, "run_cmd", fake_run_cmd)
    info = V.run_fortran("case", "CASE", str(deck0), str(deck1),
                         str(tmp_path / "wd"), "none")

    assert info["status"] == "ok", info
    assert len(seen) == 3, seen
    assert seen[0][:2] == [oracle["starter"], "-i"]
    assert seen[0][-4:] == ["-np", "1", "-nt", "1"]
    assert seen[1][:2] == [oracle["engine"], "-i"]
    assert "-np" not in seen[1]
    assert seen[1][-2:] == ["-nt", "1"]
    assert seen[2] == [oracle["th_to_csv"], "CASET01"]
    assert info["n_nodes"] == 8 and info["n_cycles"] == 100


# --------------------------------------------------------------------------
# 6. the runtime environment is upstream's
# --------------------------------------------------------------------------

def test_fortran_env_is_upstreams_block_and_refuses_h3d(monkeypatch):
    """``INSTALL.md:34-42`` reproduced, minus the one variable that lies.

    ``RAD_H3D_PATH`` is not merely left alone: it is **removed**.  The
    reachable ``libh3dwriter.so`` is one parameter short of what the pinned
    source calls, so a run that sets it reaches NORMAL TERMINATION and writes
    silently wrong H3D files
    (``tools/validation_data/oracle_provenance.json``,
    ``extlib.version_gaps.enforcement``).  An inherited value from the
    caller's shell would reintroduce the hazard, so the harness deletes it
    even when the caller exported it.
    """
    _require_oracle()
    from tools import validate_vs_fortran as V
    monkeypatch.setenv("RAD_H3D_PATH", "/should/not/be/here")
    env = V.fortran_env()

    assert "RAD_H3D_PATH" not in env
    assert env["RAD_CFG_PATH"] == str(paths.hm_cfg_dir()) or \
        Path(env["RAD_CFG_PATH"]).is_dir()
    assert "400m" in env["OMP_STACKSIZE"]
    assert env["OMP_NUM_THREADS"] == "1"
    reader_dir = str(Path(V.oracle_paths()["hm_reader_lib"]).parent)
    assert reader_dir in env["LD_LIBRARY_PATH"]
    assert os.pathsep in env["PATH"] or env["PATH"] == ""


def test_fortran_env_uses_the_platform_path_separator():
    """``;`` is a Windows separator; a POSIX loader would not split on it."""
    from tools import validate_vs_fortran as V
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = V.fortran_env()
    for var in ("LD_LIBRARY_PATH", "PATH"):
        if os.pathsep == ":":
            assert ";" not in env.get(var, ""), (
                f"{var} is joined with the Windows separator: {env.get(var)!r}")


# --------------------------------------------------------------------------
# 7. the P0.8 manifest API is untouched
# --------------------------------------------------------------------------

def test_manifest_api_is_unchanged():
    """Task P0.8 owns these; this task must not move them.

    ``tools/validation_data/rd_decks_manifest.json`` is written against them,
    and ``resolve_manifest_record`` is the corpus binding that keeps a verdict
    joined to the bytes it was measured on.
    """
    from tools import validate_vs_fortran as V
    # the module uses ``from __future__ import annotations``, so the
    # annotations are the strings a caller sees in a signature — pin those.
    expected = {
        "load_manifest_doc": "(path: 'Optional[str]' = None) -> 'dict'",
        "load_manifest": "(path: 'Optional[str]' = None) -> 'List[dict]'",
        "manifest_corpus_root": "(doc: 'Optional[dict]' = None) -> 'str'",
        "corpus_fingerprint": "(root: 'str') -> 'str'",
        "resolve_manifest_record":
            "(rec: 'dict', root: 'Optional[str]' = None, *, verify: 'bool' = "
            "True, require_fingerprint: 'bool' = True) -> 'str'",
    }
    for name, signature in expected.items():
        fn = getattr(V, name)
        assert str(inspect.signature(fn)) == signature, name


def test_manifest_still_loads_every_record():
    """A smoke test on the API the harness's own corpus work depends on."""
    from tools.validate_vs_fortran import (load_manifest, load_manifest_doc,
                                           resolve_manifest_record)
    doc = load_manifest_doc()
    records = load_manifest()
    assert records and records == doc["decks"]
    assert all(r["case_id"] for r in records)
    first = sorted(records, key=lambda r: r["deck"])[0]
    resolved = resolve_manifest_record(first)
    assert Path(resolved).is_file()


def argparse_namespace(**kwargs):
    """The attribute bag ``main()`` hands to ``parity()``."""
    import argparse
    return argparse.Namespace(**kwargs)
