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

import ast
import inspect
import os
import re
import shutil
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

#: A Windows drive letter in **source**: one letter, a colon, a separator —
#: and not the tail of a longer word, so prose like ``"instead:\n"`` inside a
#: string literal is not a false positive.
DRIVE_LETTER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")

#: ``tools/*.py`` files that legitimately keep a Windows literal, and why.
#: The sweep fails on any file NOT listed here, so the list is the whole
#: exemption surface and cannot grow by accident.  Each entry is also checked
#: to still *contain* the literal it excuses: delete the literal and the
#: entry has to be deleted with it, which stops an exemption from silently
#: becoming a permanent hole.
DELIBERATE_WINDOWS_LITERALS = {
    "lspp_check.py":
        "LS-PrePost ships no Linux build, so its install paths can only be "
        "Windows; find_lsprepost() honours $LSPP_EXE before this list and the "
        "function returns None when neither exists",
}

#: Non-Python sources and recorded data, deliberately outside the sweep.
#: Named here so the scope of the claim is visible, not implied.
DELIBERATE_NON_SWEEP = {
    "tools/run_reference_or.ps1":
        "the Windows reference-box launcher: a PowerShell script whose "
        "defaults are that box's OpenRadioss install and Intel oneAPI MPI, "
        "the layout tools/oracle_env.sh replaces on the Linux oracle",
    "tools/validation_data/**":
        "recorded evidence from the historical Windows sweeps - a JSON "
        "record of what ran where; rewriting it would falsify the provenance",
}

#: The drive-letter sweep above is *shape* shaped, so a machine-specific
#: POSIX path slips through it.  These are the literals that exist today, both
#: predating this task and both in files it does not own.  The table is a
#: **ratchet**, not an amnesty: any *other* literal in those files — or in any
#: other file — fails the test below.  The exemption is per literal, not per
#: file, so a new path added to an excused file cannot hide behind it.
KNOWN_POSIX_MACHINE_DEFAULTS = {
    "oracle/toolchain_probe.py": {
        "reason": "$OR_SRC falls back to a fixed checkout path; owned by the "
                  "toolchain task, under tools/oracle/ which P0.9 may not edit",
        "paths": ("/home/valentin",),
    },
    "profile_cycle.py": {
        "reason": "a profiling scratch directory composed from TEMP plus a "
                  "mangled Windows profile name and a session UUID; a per-run "
                  "artefact of the M39 profiling session, not a repository "
                  "resource",
        "paths": ("C--Users-pmqua-PycharmProjects-OpenRadioss-Python",),
    },
}

#: What counts as a machine-specific POSIX default in a ``tools/*.py`` source:
#: an absolute home directory (matched to the user, not the whole path — the
#: user *is* the machine), or an encoded Windows profile directory.
POSIX_MACHINE_PATH = re.compile(
    r"/(?:home|Users)/[A-Za-z0-9_.-]+"         # /home/someone, /Users/someone
    r"|C--Users-[A-Za-z0-9_.-]+")               # "C--Users-…" as a path part

#: The exact assertions the brief's Step 1 makes, kept verbatim in spirit.
FORBIDDEN_IN_HARNESS = (r"C:\OpenRadioss", "Intel\\\\oneAPI")


# ---------------------------------------------------------------------------
# Oracle discovery -- pyradioss.paths is the single resolver (plan 00 §4.1);
# the live-oracle gate itself lives in tests/test_p0_oracle_build.py (one
# definition, imported by every oracle test module, so the three cannot drift).
# ---------------------------------------------------------------------------

from tests.test_p0_oracle_build import _require_live_oracle  # noqa: E402


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
    comment or an f-string is still a default somebody will believe.  The
    harness is held to this rule **instead of** the ``DELIBERATE_WINDOWS_LITERALS``
    exemption every other ``tools/*.py`` may claim, so its historical
    docstring (the Ryan Lee corpus paragraph) and its ``h3d_dl.c`` citations
    are demonstrably literal-free rather than merely excused.
    """
    src = HARNESS.read_text(encoding="utf-8")
    for lit in FORBIDDEN_IN_HARNESS:
        assert lit not in src, f"{lit!r} survived in {HARNESS}"
    hits = sorted(set(DRIVE_LETTER.findall(src)))
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


def test_no_tools_python_source_names_a_windows_drive_letter():
    """No ``tools/**/*.py`` may spell a Windows drive letter, period.

    The reviewer's check for this task, made mechanical: every Python source
    under ``tools/`` must be literal-free except the files listed in
    :data:`DELIBERATE_WINDOWS_LITERALS`, whose entries state why.  Those
    files are additionally required to still contain the literal they excuse,
    so an exemption cannot outlive its reason.
    """
    excused, offenders = set(DELIBERATE_WINDOWS_LITERALS), []
    for src in sorted(TOOLS.rglob("*.py")):
        name = src.name
        if name == HARNESS.name:
            continue                      # held to the stricter rule above
        text = src.read_text(encoding="utf-8", errors="replace")
        hits = sorted(set(DRIVE_LETTER.findall(text)))
        if not hits:
            assert name not in excused, (
                f"{name} no longer needs its DELIBERATE_WINDOWS_LITERALS entry "
                f"({DELIBERATE_WINDOWS_LITERALS.get(name)}); delete the entry so "
                f"the exemption surface stays honest")
            continue
        if name in excused:
            continue
        offenders.append(f"{src.relative_to(REPO)}: {hits}")
    assert offenders == [], (
        "Windows drive letters left in tools/*.py — resolve them through the "
        "environment, or add a documented entry to "
        "DELIBERATE_WINDOWS_LITERALS: " + "; ".join(offenders))


def test_deliberate_windows_literal_exemptions_are_still_needed():
    """Every exemption names a real file and still has work to do.

    Two failure modes this closes: an exemption for a file that no longer
    exists (the sweep would silently stop covering a *new* file of the same
    name), and one whose literal has been removed.
    """
    for name, reason in DELIBERATE_WINDOWS_LITERALS.items():
        src = TOOLS / name
        assert src.is_file(), f"{name} does not exist but is excused: {reason}"
        assert reason.strip(), f"{name} is excused without a reason"
        text = src.read_text(encoding="utf-8", errors="replace")
        assert DRIVE_LETTER.search(text), (
            f"{name} is excused from the drive-letter sweep but spells none")
    for path in DELIBERATE_NON_SWEEP:
        head = path.split("*")[0].rstrip("/")
        assert (REPO / head).exists(), f"{path} is declared out of scope but absent"


def _machine_paths_in_code(src: Path) -> set:
    """The machine-specific path literals in ``src``'s *code* (not prose).

    A docstring naming a measured layout is context, not a default, so the
    docstring line ranges are dropped before scanning — with ``ast``, which
    knows where they are instead of guessing from the quotes.  Comments are
    dropped for the same reason.
    """
    text = src.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:                           # not our file to judge
        return set()
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            skip.update(range(node.lineno, node.value.end_lineno + 1))
    code_only = "\n".join(line for i, line in enumerate(text.splitlines(), 1)
                          if i not in skip and not line.lstrip().startswith("#"))
    return set(POSIX_MACHINE_PATH.findall(code_only))


def test_no_new_machine_specific_posix_path_in_tools():
    """Ratchet: no machine path in ``tools/*.py`` beyond the named literals.

    The drive-letter sweep is shape shaped, so a POSIX absolute path walks
    straight past it — which is how ``tools/oracle/toolchain_probe.py:41`` and
    ``tools/profile_cycle.py:102`` survived round 1.  Both are in files this
    task does not own, so their exact literals are *named* in
    :data:`KNOWN_POSIX_MACHINE_DEFAULTS` rather than edited.  The point of this
    test is that the next one fails, in the same file or a new one.  Delete a
    literal and its entry together.
    """
    problems = []
    for src in sorted(TOOLS.rglob("*.py")):
        relative = str(src.relative_to(TOOLS))
        found = _machine_paths_in_code(src)
        known = KNOWN_POSIX_MACHINE_DEFAULTS.get(relative)
        if found and known is None:
            problems.append(f"{src.relative_to(REPO)}: unrecorded {sorted(found)}")
            continue
        if known is None:
            continue
        extra = sorted(found - set(known["paths"]))
        if extra:
            problems.append(
                f"{src.relative_to(REPO)}: new machine path(s) {extra} in a "
                f"file that already has a recorded one — resolve them through "
                f"the environment, or extend the entry with a reason")
        gone = sorted(set(known["paths"]) - found)
        if gone:
            problems.append(
                f"{src.relative_to(REPO)}: recorded machine path(s) {gone} are "
                f"gone — delete the KNOWN_POSIX_MACHINE_DEFAULTS entry so the "
                f"ratchet stays honest")
    assert problems == [], ("machine-specific absolute paths in tools/*.py: "
                           + "; ".join(problems))


def test_known_posix_machine_defaults_are_still_real():
    """Each named exception names a file that exists and still carries it."""
    for name, entry in KNOWN_POSIX_MACHINE_DEFAULTS.items():
        src = TOOLS / name
        assert src.is_file(), f"{name} is named as a known offender but absent"
        assert entry["reason"].strip(), f"{name} is named without a reason"
        assert entry["paths"], f"{name} is named without a path"
        found = _machine_paths_in_code(src)
        for path in entry["paths"]:
            assert path in found, (
                f"{name} was recorded as carrying {path!r} but does not")


# --------------------------------------------------------------------------
# 3. resolution works where the oracle is
# --------------------------------------------------------------------------

def test_oracle_paths_returns_the_built_binaries():
    _require_live_oracle()
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
    _require_live_oracle(runtime_env=False)
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
    _require_live_oracle()
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


def test_fortran_env_closes_every_h3d_dlopen_route(monkeypatch):
    """``h3dlib_load_`` has FOUR routes to the writer; all four are closed.

    ``$OR_SRC/common_source/output/h3d/h3d_build_cpp/h3d_dl.c:616-923``
    (``h3dlib_load_``; ``h3dlib`` is ``libh3dwriter.so``, ``:63``) tries, in
    order: ``$RAD_H3D_PATH`` (``:623-632``), ``getcwd()`` (``:634-644``),
    ``$ALTAIR_HOME/hwsolvers/common/bin/$ARCH`` (``:647-658``) and a bare
    ``dlopen`` fed by the loader's search path (``:660-666``).  Deleting only
    the first leaves three live routes, and any of them loads the writer that
    ``oracle_provenance.json`` calls one parameter short of the pinned API —
    a run that does reaches NORMAL TERMINATION and writes silently wrong H3D
    files.  The first three are environment variables, so they are dropped
    here even when inherited; the fourth and the cwd route are covered by
    :func:`test_fortran_env_drops_a_poisoned_loader_path_entry` and
    :func:`test_run_fortran_refuses_a_workdir_holding_the_h3d_writer`.
    """
    from tools import validate_vs_fortran as V
    monkeypatch.setenv("RAD_H3D_PATH", "/should/not/be/here")
    monkeypatch.setenv("ALTAIR_HOME", "/opt/altair")
    monkeypatch.setenv("ARCH", "linux64")
    env = V.fortran_env(base={"RAD_H3D_PATH": "/x", "ALTAIR_HOME": "/y",
                              "ARCH": "z", "PATH": ""})
    for var in ("RAD_H3D_PATH", "ALTAIR_HOME", "ARCH"):
        assert var not in env, f"{var} reaches the solver, so h3d_dl.c can " \
                               f"still dlopen the incompatible writer"


def test_fortran_env_drops_a_poisoned_loader_path_entry(tmp_path, monkeypatch):
    """An ``LD_LIBRARY_PATH`` entry holding the writer is dropped AND named.

    ``h3d_dl.c:660-666`` is a bare ``dlopen("libh3dwriter.so")``, which the
    loader resolves through ``LD_LIBRARY_PATH``.  A caller who has a stale
    writer on that path would otherwise get silently wrong H3D output from an
    oracle run; a silent *drop* is no better, so the warning says which
    directory went and why.
    """
    from tools import validate_vs_fortran as V
    poisoned = tmp_path / "poisoned"
    poisoned.mkdir()
    (poisoned / "libh3dwriter.so").write_bytes(b"\x7fELF not really")
    honest = tmp_path / "honest"
    honest.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH",
                       os.pathsep.join([str(honest), str(poisoned)]))
    with pytest.warns(RuntimeWarning) as caught:
        env = V.fortran_env()
    entries = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(poisoned) not in entries
    assert str(honest) in entries, "an unrelated entry must survive"
    message = "\n".join(str(w.message) for w in caught)
    assert str(poisoned) in message and "h3d" in message.lower()


def test_h3d_search_path_vars_covers_the_windows_path_trial():
    """On Windows the fourth trial reads ``PATH`` — so ``PATH`` is scrubbed.

    ``h3d_dl.c``'s Windows ``h3dlib_load_`` (``:313``) has the same four trials
    as the POSIX one, and its fourth takes the search path straight out of the
    environment: ``GetEnvironmentVariable("PATH", …)`` at ``:356``,
    ``SetDllDirectory`` at ``:357``, ``LoadLibrary`` at ``:358``.  The
    platform is a **parameter**, not a ``skip``: a Windows-only route that
    cannot be exercised from a POSIX box is a route nobody tests.
    """
    from tools import validate_vs_fortran as V
    assert V.h3d_search_path_vars("nt")[0] == "PATH"
    assert "PATH" in V.h3d_search_path_vars("nt")
    assert "PATH" not in V.h3d_search_path_vars("posix")
    assert V.h3d_search_path_vars() == V.h3d_search_path_vars(os.name)


def test_fortran_env_scrubs_a_poisoned_path_element_on_windows(tmp_path,
                                                                monkeypatch):
    """A hazardous ``PATH`` element is dropped; the rest of ``PATH`` survives.

    Policy, stated because it is a judgement: ``PATH`` also carries every
    ordinary tool a run may shell out to, so stripping it wholesale would
    break the run in a way that looks like a broken oracle.  Only the
    elements that actually hold the writer go, and the warning names them.
    """
    from tools import validate_vs_fortran as V
    poisoned = tmp_path / "poisoned"
    poisoned.mkdir()
    (poisoned / "h3dwriter.dll").write_bytes(b"MZ not really")
    honest = tmp_path / "honest"
    honest.mkdir()
    (honest / "where.exe").write_bytes(b"MZ not really")
    with pytest.warns(RuntimeWarning) as caught:
        env = V.fortran_env(base={"PATH": os.pathsep.join(
            [str(honest), str(poisoned)])}, platform="nt")
    entries = env["PATH"].split(os.pathsep)
    assert str(poisoned) not in entries, "h3d_dl.c:356 would still find it"
    assert str(honest) in entries, "an ordinary tool directory must survive"
    message = "\n".join(str(w.message) for w in caught)
    assert str(poisoned) in message and "PATH" in message


def test_fortran_env_refuses_a_reader_dir_that_holds_the_h3d_writer(tmp_path):
    """The reader directory the harness *adds* is checked too.

    Prepending ``$OR_BUILD/extlib/hm_reader/linux64`` is the harness's own
    doing, so if that directory ever carried the writer the harness would be
    creating the ``:660-666`` route itself.  It is then not added, and the
    run fails loudly instead of writing wrong H3D files.
    """
    from tools import validate_vs_fortran as V
    reader_dir = tmp_path / "lib"
    reader_dir.mkdir()
    (reader_dir / "libh3dwriter.so").write_bytes(b"\x7fELF not really")
    (reader_dir / "libhm_reader_linux64.so").write_bytes(b"\x7fELF not really")
    oracle = {"hm_reader_lib": str(reader_dir / "libhm_reader_linux64.so")}
    with pytest.warns(RuntimeWarning) as caught:
        env = V.fortran_env(oracle=oracle, base={"LD_LIBRARY_PATH": ""})
    assert str(reader_dir) not in env.get("LD_LIBRARY_PATH", "")
    assert "h3d" in "\n".join(str(w.message) for w in caught).lower()


def test_run_fortran_refuses_a_workdir_holding_the_h3d_writer(tmp_path,
                                                              monkeypatch,
                                                              bare_env):
    """The scratch directory IS the solvers' cwd, so it is checked too.

    ``h3d_dl.c:634-644`` dlopens ``getcwd() + "/" + h3dlib``, and
    ``run_fortran`` copies the deck's own directory into the scratch
    directory — so a writer sitting next to a deck would be copied in and
    then loaded.  The run is refused before anything is launched.
    """
    from tools import validate_vs_fortran as V
    case = tmp_path / "case"
    case.mkdir()
    deck0 = case / "CASE_0000.rad"
    deck1 = case / "CASE_0001.rad"
    deck0.write_text("#RADIOSS STARTER\n", encoding="utf-8")
    deck1.write_text("#RADIOSS ENGINE\n", encoding="utf-8")
    (case / "libh3dwriter.so").write_bytes(b"\x7fELF not really")

    launched = []
    monkeypatch.setattr(V, "oracle_paths", lambda *a, **k: {
        "starter": "/opt/or/bin/starter_linux64_gf",
        "engine": "/opt/or/bin/engine_linux64_gf", "th_to_csv": None,
        "h3d_lib": None, "hm_reader_lib": None})
    monkeypatch.setattr(V, "run_cmd",
                        lambda *a, **k: launched.append(a) or (0, "", 0.0))
    info = V.run_fortran("case", "CASE", str(deck0), str(deck1),
                         str(tmp_path / "wd"), "none")
    assert info["status"] == "h3d-writer-in-workdir", info
    assert "h3d_dl.c:634-644" in info["error"]
    assert launched == [], "a solver was launched from a poisoned directory"


def _synthetic_elf(path, rpath_entries, tag=15):
    """A minimal but *real* 64-bit little-endian ELF carrying an RPATH.

    Built by hand because the point is that ``elf_search_paths`` reads the
    dynamic section correctly — a fixture that merely reuses the parser would
    prove nothing, and depending on a built oracle would make the test skip on
    most boxes.  One ``PT_LOAD`` covering the whole file, one ``PT_DYNAMIC``
    at 0x800, the string table at 0x1000, and ``DT_STRTAB`` / ``DT_STRSZ`` /
    one ``DT_RPATH``/``DT_RUNPATH`` entry.
    """
    import struct
    dynamic_off, strtab_off = 0x800, 0x1000
    load_vaddr = 0x400000
    # one DT_RPATH string holding a colon-separated list, as the format is
    # defined — not several strings
    body = ":".join(rpath_entries).encode() + b"\0" + b"NEEDED-name\0"
    file_size = strtab_off + len(body)
    dyn = b"".join(struct.pack("<QQ", t, v) for t, v in
                   ((5, load_vaddr + strtab_off), (10, len(body)), (tag, 0)))
    dyn += struct.pack("<QQ", 0, 0)                 # DT_NULL terminates
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    header = ident + struct.pack(
        "<HHIQQQIHHHHHH", 2, 0x3E, 1, 0, 64, 0, 0, 64, 56, 2, 0, 0, 0)
    phdrs = (struct.pack("<IIQQQQQQ", 1, 5, 0, load_vaddr, load_vaddr,
                         file_size, file_size, 0x1000)
             + struct.pack("<IIQQQQQQ", 2, 6, dynamic_off,
                           load_vaddr + dynamic_off, 0, len(dyn), len(dyn), 8))
    blob = bytearray(header + phdrs)
    blob += b"\0" * (dynamic_off - len(blob))
    blob += dyn
    blob += b"\0" * (strtab_off - len(blob))
    blob += body
    assert len(blob) == file_size
    Path(path).write_bytes(bytes(blob))
    return str(path)


def test_elf_search_paths_reads_the_dynamic_section(tmp_path):
    """RPATH and RUNPATH, ``$ORIGIN`` expanded, junk answered with ``[]``.

    Parsed straight from the ELF rather than by shelling out to ``readelf``:
    a validation run must not depend on binutils being installed, and a
    process spawn per case is a hot path this does not need (a few seeks).
    """
    from tools import validate_vs_fortran as V
    binary = tmp_path / "oracle_bin"
    _synthetic_elf(binary, ["/opt/conda/lib", "/opt/other/lib"])
    assert V.elf_search_paths(binary) == ["/opt/conda/lib", "/opt/other/lib"]

    runpath = tmp_path / "runpath_bin"
    _synthetic_elf(runpath, ["/opt/runpath"], tag=29)
    assert V.elf_search_paths(runpath) == ["/opt/runpath"]

    origin = tmp_path / "origin_bin"
    _synthetic_elf(origin, ["$ORIGIN/../lib", "${ORIGIN}/lib"])
    origin_dir = str(origin.parent)
    assert V.elf_search_paths(origin) == [
        os.path.normpath(os.path.join(origin_dir, "..", "lib")),
        os.path.normpath(os.path.join(origin_dir, "lib"))]

    # no dynamic section, not an ELF, unreadable: a probe that raises would
    # be worse than one that reports nothing
    assert V.elf_search_paths(str(tmp_path / "missing")) == []
    assert V.elf_search_paths(__file__) == []
    (tmp_path / "plain").write_text("not an elf\n", encoding="utf-8")
    assert V.elf_search_paths(str(tmp_path / "plain")) == []


def test_binary_rpath_hazards_names_a_poisoned_rpath(tmp_path):
    """A writer visible in the binary's RPATH is reported, and the run refused.

    glibc searches ``DT_RPATH`` **before** ``LD_LIBRARY_PATH`` — the reviewer
    proved it with a purpose-built ELF pair — so a stale
    ``libh3dwriter.so`` in a prefix the binary carries is dlopen'd by the bare
    trial (``h3d_dl.c:660-666``) whatever the harness exports.  Measured on
    the oracle built for this box, both binaries carry
    ``DT_RPATH=/home/valentin/anaconda/lib``: a conda prefix that has nothing
    to do with the oracle, i.e. exactly the machine-specific leakage this
    task exists to remove.
    """
    from tools import validate_vs_fortran as V
    prefix = tmp_path / "conda_lib"
    prefix.mkdir()
    starter = _synthetic_elf(tmp_path / "starter_linux64_gf", [str(prefix)])
    engine = _synthetic_elf(tmp_path / "engine_linux64_gf", [str(prefix)])
    oracle = {"starter": starter, "engine": engine, "th_to_csv": None,
              "h3d_lib": None, "hm_reader_lib": None}
    assert V.binary_rpath_hazards(oracle) == [], (
        "an empty RPATH directory is not a hazard")

    (prefix / "libh3dwriter.so").write_bytes(b"\x7fELF not really")
    hazards = V.binary_rpath_hazards(oracle)
    assert len(hazards) == 2, "both binaries carry the hazard"
    assert str(prefix) in hazards[0] and "DT_RPATH" in hazards[0]

    with pytest.warns(RuntimeWarning) as caught:
        V.fortran_env(oracle=oracle)
    assert any("DT_RPATH" in str(w.message) for w in caught)

    launched = []
    monkey_run = V.run_cmd
    try:
        V.run_cmd = lambda *a, **k: launched.append(a)
        info = V.run_fortran("case", "CASE", __file__, __file__,
                             str(tmp_path / "wd"), "none", oracle)
    finally:
        V.run_cmd = monkey_run
    assert info["status"] == "h3d-writer-in-rpath", info
    assert launched == [], "a solver was launched with an RPATH hazard"


def test_the_real_oracle_binaries_rpath_is_read_correctly():
    """Cross-check the parser against ``readelf`` on the actual binaries.

    Only meaningful when the oracle exists and ``readelf`` is installed; the
    parser itself is covered hermetically by the synthetic-ELF test above.
    """
    _require_live_oracle(runtime_env=False)
    from tools import validate_vs_fortran as V
    if not shutil.which("readelf"):
        pytest.skip("readelf not installed; the synthetic-ELF test covers the "
                    "parser")
    for key in ("starter", "engine"):
        binary = V.oracle_paths()[key]
        proc = subprocess.run(["readelf", "-d", binary], capture_output=True,
                              text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        expected = []
        for line in proc.stdout.splitlines():
            if "(RPATH)" in line or "(RUNPATH)" in line:
                expected += re.findall(r"\[([^\]]*)\]", line)[0].split(":")
        assert V.elf_search_paths(binary) == [e for e in expected if e], (
            f"{binary}: parser disagrees with readelf")


def test_fortran_env_uses_the_platform_path_separator():
    from tools import validate_vs_fortran as V
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = V.fortran_env()
    for var in ("LD_LIBRARY_PATH", "PATH"):
        if os.pathsep == ":":
            assert ";" not in env.get(var, ""), (
                f"{var} is joined with the Windows separator: {env.get(var)!r}")


# --------------------------------------------------------------------------
# 6b. import-time laziness, and the subtag convention
# --------------------------------------------------------------------------

def test_default_workdir_is_lazy():
    """``tempfile.gettempdir()`` must not run at import.

    It creates and deletes a probe file when the platform's temp location is
    not already known, so evaluating it in a module constant would make
    "importing the harness touches no filesystem" false with an exception in
    it — and the pathlib booby-trap above cannot see it, because the probe
    goes through ``os``, not ``pathlib``.
    """
    from tools import validate_vs_fortran as V
    assert not hasattr(V, "DEFAULT_WORKDIR")
    assert callable(V.default_workdir)


def test_import_needs_no_tempdir_probe():
    """Import the module with ``tempfile.gettempdir`` booby-trapped."""
    env = {k: v for k, v in os.environ.items() if k not in ORACLE_ENV_VARS}
    env["PYTHONPATH"] = str(REPO)
    code = textwrap.dedent(
        """
        import numpy                       # noqa: F401
        import pyradioss.paths             # noqa: F401
        import tempfile

        def _boom(*a, **k):
            raise AssertionError("import asked for the platform temp dir")

        tempfile.gettempdir = _boom
        tempfile.mkdtemp = _boom
        tempfile.mkstemp = _boom
        import tools.validate_vs_fortran as V
        print("imported", V.__name__)
        """)
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "imported tools.validate_vs_fortran" in proc.stdout


def test_read_deck_is_not_imported_at_module_scope():
    """Only ``coverage()`` reads decks, so only ``coverage()`` imports the reader.

    A module-scope ``from pyradioss.input.deck_reader import read_deck`` pulls
    numpy and the whole keyword table into every parity run — work a parity
    run never uses, imported before anything can report it.
    """
    from tools import validate_vs_fortran as V
    assert not hasattr(V, "read_deck")
    assert "read_deck" in V.coverage.__code__.co_varnames, (
        "coverage() must import read_deck locally")


def test_a_missing_converter_is_not_reported_as_a_failing_solver(tmp_path,
                                                                  monkeypatch,
                                                                  capsys):
    """``th2csv-missing`` gets its own class subtag; the old classes do not move.

    The console table collapsed every Fortran-side problem into one bare
    ``FORTRAN-FAIL``, so a reader could not tell "the engine died" from "the
    converter was never installed" — and the second is a *tooling* fact, not
    a physics verdict.  ``tools/validation_data/parity_m41.json`` is keyed on
    the bare string, so the two pre-existing statuses must keep producing it.
    """
    from tools import validate_vs_fortran as V
    assert V.FORTRAN_FAIL_SUBTAGS["engine-fail"] is None
    assert V.FORTRAN_FAIL_SUBTAGS["th2csv-fail"] is None
    assert V.FORTRAN_FAIL_SUBTAGS["th2csv-missing"] == "th2csv-missing"

    def run(status, workdir):
        monkeypatch.setattr(V, "oracle_paths", lambda *a, **k: {
            key: "/opt/or/bin/x" for key in V.ORACLE_KEYS})
        monkeypatch.setattr(V, "run_fortran",
                            lambda *a, **k: {"status": status, "mode": "none"})
        monkeypatch.setattr(V, "run_pyradioss",
                            lambda *a, **k: {"status": "ok", "csv": None})
        args = argparse_namespace(workdir=str(workdir), only="tensile_bar",
                                  budget=10.0, timeout=10.0, tol=0.05,
                                  shim="translate")
        assert V.parity(args) == 0
        import json
        rows = json.loads((Path(workdir) / "parity_results.json")
                          .read_text(encoding="utf-8"))
        return rows[0]["class"]

    assert run("th2csv-missing", tmp_path / "a") == \
        "FORTRAN-FAIL(th2csv-missing)"
    assert run("engine-fail", tmp_path / "b") == "FORTRAN-FAIL"
    printed = capsys.readouterr().out
    assert "FORTRAN-FAIL(th2csv-missing)" in printed


# --------------------------------------------------------------------------
# 6c. the other two tools: no machine named in the source
# --------------------------------------------------------------------------

def test_benchmark_rad_db_corpus_root_follows_the_environment(tmp_path,
                                                               monkeypatch):
    """The harvested corpus is named by ``$RAD_EXAMPLES_DB``, or loudly not.

    ``tools/benchmark_rad_db.py`` used to hardcode one developer's home
    directory; ``load_gold_set()`` then died with a bare
    ``FileNotFoundError`` on a path no reader could act on.  A candidate must
    also *carry the gold set*, so a wrong-but-existing directory is refused
    rather than half-read.
    """
    from tools import benchmark_rad_db as B
    corpus = tmp_path / "rad_examples_db"
    (corpus / "candidates" / "bench").mkdir(parents=True)
    (corpus / "candidates" / "bench" / "gold_set.json").write_text("[]",
                                                                  encoding="utf-8")
    monkeypatch.setenv(B.CORPUS_ROOT_ENV, str(corpus))
    assert B.corpus_root() == corpus
    assert B.gold_set_path().is_file()
    assert B.load_gold_set() == []

    monkeypatch.setenv(B.CORPUS_ROOT_ENV, str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError) as raised:
        B.corpus_root()
    assert B.CORPUS_ROOT_ENV in str(raised.value)

    monkeypatch.delenv(B.CORPUS_ROOT_ENV)
    with pytest.raises(FileNotFoundError) as raised:
        B.corpus_root()
    assert B.CORPUS_ROOT_ENV in str(raised.value)


def test_compare_t01_tab1_finds_its_inputs(tmp_path, monkeypatch, capsys):
    """No scratchpad path: ``$TAB1_BASE_DIR``, else the working directory.

    And a missing input is reported by name — both files, the directory
    searched, and the variable to set — instead of a bare numpy
    ``FileNotFoundError``.
    """
    from tools import compare_t01_tab1 as T
    monkeypatch.setenv(T.BASE_DIR_ENV, str(tmp_path))
    assert T.default_base_dir() == tmp_path
    monkeypatch.delenv(T.BASE_DIR_ENV)
    monkeypatch.chdir(tmp_path)
    assert T.default_base_dir() == tmp_path

    assert T.main([]) == 2
    err = capsys.readouterr().err
    assert T.FORTRAN_CSV_NAME in err and T.PORT_CSV_NAME in err
    assert T.BASE_DIR_ENV in err

    (tmp_path / T.FORTRAN_CSV_NAME).write_text("t,IE\n0,1\n1,2\n", encoding="utf-8")
    (tmp_path / T.PORT_CSV_NAME).write_text("t,IE\n0,1\n1,2\n", encoding="utf-8")
    assert T.main([]) == 0
    assert "Match (rtol=0.05" in capsys.readouterr().out


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
