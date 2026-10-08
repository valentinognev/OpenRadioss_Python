"""Task P1.6 — the frozen surface of the quarantined ``pyradioss/implicit/``.

Why this file exists
--------------------
``pyradioss/implicit/`` carries ~25k LOC that upstream OpenRadioss does **not**
have: implicit Newton/arc-length (M8–M15), modal + complex-modal superposition
(M16–M18), random/PSD response and response spectra (M19), spectral fatigue
(M20–M34). The maintainer decision of **2026-10-02**, recorded in
``plan/00_ORCHESTRATION.md`` §3, is *keep, but quarantine*: no new features
there except what Phase 15 needs for parity with upstream
``engine/source/implicit``, and every module must stay importable and green for
the whole program.

That decision is only real if the *surface* is pinned. Nothing stopped an agent
from adding or deleting a file in that package, and both would silently change
what the project is. ``plan/00_ORCHESTRATION.md`` §3 names the mechanism:
``pyradioss/implicit/_frozen_surface.py`` enumerates the modules, and a test
pins the enumeration — **adding** a module becomes a deliberate act with a
review gate, **removing** one is a hard failure.

The invariants here, and what each can actually detect
------------------------------------------------------
1. :func:`test_implicit_surface_is_frozen` — the live ``pkgutil`` scan, minus
   the two names in ``_CENSUS_EXCLUDED``, equals ``FROZEN_MODULES``, reported in
   BOTH directions (``added`` / ``removed``) so the failure names the culprit
   instead of just saying "sets differ". Detects: a module added, a module
   deleted, a module renamed — **including a ``_``-prefixed one**, which an
   earlier revision of this rule (exclude anything starting with ``_``) let
   through untouched. Cannot detect: a change *inside* a module (that is the
   milestone suite's job, not this file's).

2. :func:`test_the_census_excludes_only_the_two_named_modules` and
   :func:`test_the_census_cannot_be_hiding_a_non_module_artifact` — the census
   rule itself is pinned, from two sides: the exclusion list is exactly the two
   names it is documented to be, and the ``pkgutil`` scan accounts for every
   ``.py`` file on disk and nothing else. Without these, widening the exclusion
   (or a sub-package appearing) would be a way to make verdict 1 read green.

3. :func:`test_frozen_surface_sha256_is_recomputable` — the pinned digest is
   recomputed here from the live scan, by an implementation independent of the
   record's, and compared. Without it the digest is decoration: a constant
   nobody recomputes cannot detect a wrong constant. It bites alone: perturbing
   the constant fails this test while the set comparison still passes.

4. :func:`test_frozen_implicit_modules_still_import` — every frozen module
   imports **in a cold interpreter**, i.e. in a subprocess, so no earlier test
   in the session can have put it into ``sys.modules`` and turned the check
   into a cache hit. This is strictly stronger than importing in-process, and
   it also checks each module's origin file really sits in the package
   directory (a name that resolves to a stray module elsewhere on ``sys.path``
   is a failure, not a pass). The subprocess is **backend-pinned and
   time-boxed**: see ``_probe_env`` and ``_PROBE_TIMEOUT_S``.

5. :func:`test_optional_linsolve_backends_still_fall_back` and
   :func:`test_the_optional_linsolve_wrappers_execute_against_stand_ins` — the
   CHOLMOD/MUMPS seam, described precisely in §4 below.

6. :func:`test_frozen_surface_records_its_provenance` — the record says where it
   came from (Phase 0's standard: a record that does not is a defect).

What test 4 CANNOT detect, stated plainly
-----------------------------------------
Importability is not solvability. Every implicit module obtains SciPy through
``pyradioss.implicit.require_scipy()`` (a *call*, ``__init__.py:241-260``)
rather than a top-level ``import scipy``, so a broken SciPy path is invisible
to an import probe — by design, because the base explicit install is NumPy-only.
There is **no** ``pytest.importorskip`` inside the package: the guards are in
the *tests* (``tests/test_m8_implicit.py:30``, ``test_m10_impdyn.py:42``, and
nine siblings) and in ``require_scipy``.

§4 — the CHOLMOD/MUMPS seam: exactly what is and is not covered
----------------------------------------------------------------
``sksparse`` and ``mumps`` are **absent from this box and from CI**
(``find_spec`` → absent), and no test in this repository installs them. What the
two tests below actually assert, in full:

* **Asserted, always, on every box:**
  - ``LinearSolver._solve_cholmod`` / ``._solve_mumps`` still exist (presence);
  - ``select_linsolve``'s contract for both, on whichever branch the box is
    on: absent library → ``"superlu"`` **and a warning**; present library → the
    requested name (``test_optional_linsolve_backends_still_fall_back``);
  - **the bodies themselves run.** ``test_the_optional_linsolve_wrappers_
    execute_against_stand_ins`` installs stand-ins for ``sksparse.cholmod`` and
    ``mumps`` in ``sys.modules`` and drives ``LinearSolver.solve`` through the
    CHOLMOD and MUMPS dispatch, asserting that each wrapper converts its matrix
    with ``.tocsc()``, calls the library with ``R``, and returns what the library
    returned; and that the ``CholmodNotPositiveDefiniteError`` path **warns and
    delegates to ``_solve_superlu``** instead of raising or silently continuing.
    That fallback is project-owned code that no other test executes, because no
    other test can get past the ``import``.
* **NOT asserted, and not assertable here:** that the real ``scikit-sparse`` /
  ``python-mumps`` APIs match the stand-ins. A stand-in proves *this project's*
  wrapper logic; it cannot prove the wrapped library's signature. **So a
  regression in ``cholesky(K.tocsc())`` against the real library, or in
  ``mumps.spsolve``'s signature, would still ship silently.**
* **What a maintainer must do when the libraries are available** — this is the
  maintenance debt, stated so it cannot be forgotten: install
  ``scikit-sparse`` and ``python-mumps`` on a box that has them, then
  ``test_optional_linsolve_backends_still_fall_back`` automatically switches
  branch and asserts the real activation. Add, in that same environment, a test
  that calls ``LinearSolver(name="cholmod").solve(K, R)`` and
  ``LinearSolver(name="mumps").solve(K, R)`` on a real SPD ``K`` and asserts
  ``numpy.allclose(solver.solve(K, R), R)`` against a direct solve. Until that
  test exists, the CHOLMOD/MUMPS numerical paths are covered by review, not by
  a test — and the stand-in test must not be read as closing that gap.

No Fortran is involved (project structure only), so nothing here is cited from
``$OR_SRC``; ``$OR_SRC`` is READ-ONLY and untouched by this task.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The two names the census leaves out, written out HERE as an independent
#: literal rather than imported from the record — so widening the record's
#: ``EXCLUDED_FROM_CENSUS`` cannot quietly widen this scan. The two must agree;
#: ``test_the_census_excludes_only_the_two_named_modules`` is what holds them to
#: it, and that test pins the set to exactly these two names.
_CENSUS_EXCLUDED = frozenset({"__init__", "_frozen_surface"})

#: Wall-clock ceiling for the cold-import probe. The probe imports all 33
#: modules of the tower in one fresh interpreter (measured: ~1.5 s), so this is
#: roughly two orders of magnitude of headroom; it exists so that a module whose
#: *import hangs* fails the gate instead of hanging the gate. The probe prints a
#: line per module as it goes, so the timeout message names the culprit.
_PROBE_TIMEOUT_S = 300

REMEDIATION = (
    "Adding a name to pyradioss/implicit/ is a deliberate act that needs the "
    "maintainer's approval (plan/00_ORCHESTRATION.md §3: no new features there "
    "except what Phase 15 needs for parity with upstream "
    "engine/source/implicit). With that approval, regenerate FROZEN_MODULES in "
    "pyradioss/implicit/_frozen_surface.py from a live pkgutil scan (minus "
    "EXCLUDED_FROM_CENSUS) and update FROZEN_SURFACE_SHA256 with the digest "
    "that scan prints. Removing a name is a hard failure and is never 'fixed' "
    "by editing this list — and moving a name into EXCLUDED_FROM_CENSUS is the "
    "same deliberate act, pinned to the same two entries."
)


def _live_surface() -> set[str]:
    """The module names currently in ``pyradioss/implicit/``, minus the two
    excluded by name. A ``_`` prefix is deliberately NOT the test — see the
    scope-rule section of ``_frozen_surface.py``."""
    import pyradioss.implicit as implicit_pkg

    return {
        info.name
        for info in pkgutil.iter_modules(implicit_pkg.__path__)
        if info.name not in _CENSUS_EXCLUDED
    }


def _sha256_of_surface(names) -> str:
    """The documented digest recipe, implemented independently of the record.

    ``sha256("\\n".join(sorted(names)) + "\\n")`` — POSIX-text shaped (one name
    per line, trailing newline) so the digest is stable and reproducible by hand
    from a terminal. Deliberately NOT shared with
    ``pyradioss/implicit/_frozen_surface.py``: the record pins a literal and
    states the recipe in prose, so this recipe and the pinned constant are two
    independent things that must agree. A shared helper would make the test an
    echo of the record rather than a check on it.
    """
    payload = "\n".join(sorted(names)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# 1. the surface itself
# --------------------------------------------------------------------------


def test_implicit_surface_is_frozen():
    """Live ``pkgutil`` scan == ``FROZEN_MODULES``, reported both ways."""
    from pyradioss.implicit import _frozen_surface as frozen

    present = _live_surface()
    pinned = set(frozen.FROZEN_MODULES)

    assert present == pinned, {
        "added": sorted(present - pinned),
        "removed": sorted(pinned - present),
        "frozen_count": len(pinned),
        "live_count": len(present),
        "remediation": REMEDIATION,
    }


def test_the_census_excludes_only_the_two_named_modules():
    """The exclusion is an explicit two-name list, not a rule about ``_``.

    Two directions have to hold. The list must be exactly the two names the
    record documents (``__init__``, the facade; ``_frozen_surface``, the record
    itself) — so it cannot grow to hide a new module. And no frozen name may be
    ``_``-prefixed: private modules are INSIDE the census, which is the whole
    point of the change, because a blanket prefix rule is what let
    ``_review_private.py`` land in a quarantined package with all six tests
    reporting green.
    """
    from pyradioss.implicit import _frozen_surface as frozen

    assert set(frozen.EXCLUDED_FROM_CENSUS) == set(_CENSUS_EXCLUDED) == {
        "__init__",
        "_frozen_surface",
    }, (
        "the census exclusion list was widened or narrowed: "
        f"record={sorted(frozen.EXCLUDED_FROM_CENSUS)} test={sorted(_CENSUS_EXCLUDED)}. "
        "It is pinned to exactly two names because a '_' prefix is not a "
        "criterion — a new private module must show up as 'added' like any other."
    )
    assert len(frozen.FROZEN_MODULES) >= 30, (
        "the frozen implicit surface collapsed to "
        f"{len(frozen.FROZEN_MODULES)} modules; a census that small is a "
        "census that lost the tree, not a decision"
    )
    private = sorted(n for n in frozen.FROZEN_MODULES if n.startswith("_"))
    assert not private, (
        f"private names belong in the frozen surface, not outside it: {private}"
    )
    for excluded in sorted(_CENSUS_EXCLUDED):
        assert excluded not in frozen.FROZEN_MODULES, (
            f"{excluded!r} is excluded from the census by name; it must not also "
            "be a pinned member"
        )


def test_the_census_cannot_be_hiding_a_non_module_artifact():
    """``pkgutil`` must see everything importable in the package directory.

    The census is a ``pkgutil.iter_modules`` scan, and a scanner that silently
    skips something is a scanner whose silence reads as agreement. Two halves:

    * every name it reports is a plain ``.py`` file directly in the package
      directory — a sub-package, an extension module or a stray loader would
      show up as a name with no ``.py`` file behind it;
    * every ``.py`` file in the directory is accounted for. The one legitimate
      exception is ``__init__``: ``pkgutil._iter_file_finder_modules`` skips it
      **by design**, so the facade is invisible to any ``pkgutil``-based census
      (which is why the exclusion list names it defensively rather than because
      the scan needs it). Anything else that is present and unscanned is a hole.
    """
    import pyradioss.implicit as implicit_pkg

    pkg_dir = Path(implicit_pkg.__file__).resolve().parent
    scanned = {info.name for info in pkgutil.iter_modules(implicit_pkg.__path__)}
    on_disk = {path.stem for path in pkg_dir.glob("*.py")}

    unscanned = on_disk - scanned
    assert unscanned == {"__init__"}, {
        "on_disk_but_unscanned": sorted(unscanned),
        "note": (
            "pkgutil skips __init__ by design; every other .py file in the "
            "package must appear in the census or the scan is hiding something"
        ),
    }
    assert not scanned - on_disk, {
        "scanned_but_not_a_py_file": sorted(scanned - on_disk),
        "note": (
            "a name pkgutil reports that has no .py file in the package "
            "directory is a sub-package, an extension module or a stray loader; "
            "any of those changes the surface and must be named explicitly"
        ),
    }
    unexpected = sorted(
        entry.name
        for entry in pkg_dir.iterdir()
        if entry.suffix != ".py" and entry.name != "__pycache__"
    )
    assert not unexpected, (
        f"pyradioss/implicit/ holds non-module entries {unexpected}; if one of "
        "them can become an importable module the census has to account for it"
    )


# --------------------------------------------------------------------------
# 2. the pinned digest must be verifiable
# --------------------------------------------------------------------------


def test_frozen_surface_sha256_is_recomputable():
    """Recompute ``FROZEN_SURFACE_SHA256`` from the LIVE scan and compare.

    A constant nobody recomputes is decoration. This test is deliberately
    independent of the set comparison above: with the module set left untouched
    and one hex character of the pinned constant flipped, THIS test fails and
    the set test passes — which is the evidence that the digest carries a check
    the set cannot.
    """
    from pyradioss.implicit import _frozen_surface as frozen

    live = _live_surface()
    recomputed = _sha256_of_surface(live)

    assert recomputed == frozen.FROZEN_SURFACE_SHA256, {
        "recomputed_from_live_scan": recomputed,
        "pinned": frozen.FROZEN_SURFACE_SHA256,
        "live_count": len(live),
        "note": (
            "the digest is sha256 over the SORTED census names, newline-joined "
            "with a trailing newline; the recipe is stated in the record's "
            "docstring and reimplemented independently here"
        ),
    }
    # the pinned digest must also be a digest OF the pinned set, otherwise it
    # tracks the tree instead of pinning it
    assert recomputed == _sha256_of_surface(frozen.FROZEN_MODULES)
    assert len(frozen.FROZEN_SURFACE_SHA256) == 64
    int(frozen.FROZEN_SURFACE_SHA256, 16)  # raises if it is not hex


# --------------------------------------------------------------------------
# 3. every frozen module still imports — in a COLD interpreter
# --------------------------------------------------------------------------

#: The child program. Two things to note:
#:  * it receives the exclusion list on argv, not from the record, so the scan
#:    it probes is the test's own;
#:  * it prints one ``PYRADIOSS_FROZEN_PROBE_STEP:`` line per module before
#:    importing it, so a probe killed by the parent's timeout can still say
#:    which module it was inside.
_COLD_IMPORT_PROBE = r'''
import importlib, json, os, pkgutil, sys, traceback

root = sys.argv[1]
excluded = frozenset(json.loads(sys.argv[2]))
sys.path.insert(0, root)

import pyradioss.implicit as implicit_pkg

pkg_dir = os.path.dirname(os.path.abspath(implicit_pkg.__file__))
names = sorted(
    info.name
    for info in pkgutil.iter_modules(implicit_pkg.__path__)
    if info.name not in excluded
)

def step(name, status):
    sys.stdout.write("PYRADIOSS_FROZEN_PROBE_STEP:%s %s\n" % (name, status))
    sys.stdout.flush()

report = {"pkg_dir": pkg_dir, "names": names, "failures": [],
          "wrong_origin": [], "preloaded": [], "executed": [],
          "evict_failed": []}
for name in names:
    dotted = "pyradioss.implicit." + name
    # Importing module N often drags in module N+1 as a side effect, so a
    # plain import_module() is a CACHE HIT for every such sibling and proves
    # nothing. Evict first, then import, then check the object changed.
    was_present = dotted in sys.modules
    if was_present:
        report["preloaded"].append(dotted)
    old = sys.modules.pop(dotted, None)
    if dotted in sys.modules:
        report["evict_failed"].append(dotted)
    step(name, "importing")
    try:
        module = importlib.import_module(dotted)
    except BaseException:
        step(name, "failed")
        report["failures"].append(
            {"name": name, "traceback": traceback.format_exc()})
        continue
    new = sys.modules.get(dotted)
    # `new is not old` is re-REGISTRATION, which is what we can observe from
    # here; the body is proven to have run by the fact that import_module()
    # returned without raising. A module whose body is sound on first execution
    # and raises on re-execution is caught by that, which is the case a plain
    # cache-hit probe misses.
    if new is not old:
        report["executed"].append(dotted)
    origin = os.path.abspath(getattr(module, "__file__", None) or "")
    if os.path.dirname(origin) != pkg_dir:
        report["wrong_origin"].append(
            {"name": name, "file": origin or "<none>", "expected_dir": pkg_dir})
    step(name, "ok")

print("PYRADIOSS_FROZEN_PROBE:" + json.dumps(report))
'''

_REPORT_MARKER = "PYRADIOSS_FROZEN_PROBE:"
_STEP_MARKER = "PYRADIOSS_FROZEN_PROBE_STEP:"


def _probe_env() -> dict:
    """The child environment: backend-pinned, following the house rule.

    ``AGENTS.md`` §Domain rules: "For before/after comparisons pin the backend:
    ``PYRADIOSS_BACKEND=numpy``". This probe is not comparing two runs, but it
    IS a comparison whose verdict must not depend on the ambient session — a
    frozen module that imports under ``numba`` and not under ``numpy`` (or the
    reverse) would otherwise pass or fail depending on who launched pytest. So
    the probe pins ``numpy``, and — the same rule the regression ledger's
    ``child_command`` uses — **an existing pin wins**, because the probe must
    describe the run that actually happened rather than silently relabelling it.
    """
    env = dict(os.environ)
    if "PYRADIOSS_BACKEND" not in env:
        env["PYRADIOSS_BACKEND"] = "numpy"
    return env


def _last_step(stdout: str) -> str:
    steps = [line[len(_STEP_MARKER):] for line in stdout.splitlines()
             if line.startswith(_STEP_MARKER)]
    return steps[-1] if steps else "<none>"


def _probe_report() -> dict:
    argv = [
        sys.executable,
        "-c",
        _COLD_IMPORT_PROBE,
        str(REPO_ROOT),
        json.dumps(sorted(_CENSUS_EXCLUDED)),
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=_probe_env(),
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        raise AssertionError(
            f"the cold-import probe did not finish within {_PROBE_TIMEOUT_S}s "
            f"— the last module it reported was {_last_step(partial)!r}. A "
            "frozen module whose import hangs (an import-time read, a lock, a "
            "network call) must fail the gate, not hang it.\n"
            f"--- partial stdout ---\n{partial}\n--- partial stderr ---\n{exc.stderr}"
        ) from exc
    for line in proc.stdout.splitlines():
        if line.startswith(_REPORT_MARKER):
            return json.loads(line[len(_REPORT_MARKER):])
    raise AssertionError(
        "the cold-import probe produced no report "
        f"(returncode {proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def test_the_cold_import_probe_is_pinned_and_time_boxed(monkeypatch):
    """The probe's own contract, asserted on the real subprocess call.

    Two properties that are otherwise true only by construction and can be
    broken silently: the child gets an explicit ``env`` carrying the backend
    pin, and ``subprocess.run`` carries a ``timeout``. Both are checked against
    the call that actually happens (the probe is spied on, then really run), not
    against a restatement of the intent.
    """
    captured: dict = {}
    real_run = subprocess.run

    def spy(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    report = _probe_report()

    assert report["names"] == sorted(_live_surface()), (
        "the spied probe must still be the real probe; a fake report would make "
        "this test prove nothing"
    )
    kwargs = captured["kwargs"]
    assert kwargs.get("timeout") == _PROBE_TIMEOUT_S, (
        f"the probe subprocess has no usable timeout ({kwargs.get('timeout')!r}) "
        "— a module whose import hangs would hang the gate forever"
    )
    assert kwargs.get("env") is not None, (
        "the probe subprocess must be given an explicit env, or it inherits the "
        "ambient session and its verdict depends on who launched pytest"
    )
    expected = os.environ.get("PYRADIOSS_BACKEND", "numpy")
    assert kwargs["env"]["PYRADIOSS_BACKEND"] == expected, (
        f"the probe must run under the session's backend pin, or numpy when "
        f"there is none; expected {expected!r}, child had "
        f"{kwargs['env'].get('PYRADIOSS_BACKEND')!r}"
    )

    # ... and the "respect an existing pin" half, which is a property of
    # _probe_env rather than of this process's environment.
    saved = os.environ.pop("PYRADIOSS_BACKEND", None)
    try:
        assert _probe_env()["PYRADIOSS_BACKEND"] == "numpy"
        os.environ["PYRADIOSS_BACKEND"] = "numba"
        assert _probe_env()["PYRADIOSS_BACKEND"] == "numba", (
            "an existing pin wins — the probe must describe the run that "
            "happened, not relabel it (same rule as the regression ledger's "
            "child_command)"
        )
    finally:
        os.environ.pop("PYRADIOSS_BACKEND", None)
        if saved is not None:
            os.environ["PYRADIOSS_BACKEND"] = saved


def test_frozen_implicit_modules_still_import():
    """Every frozen module's body executes from a cold ``sys.modules``, from its own file.

    Three hardening details, all learned from a first version of this test that
    passed for the wrong reason:

    * the probe runs in a **subprocess**, because in-process
      ``importlib.import_module`` returns the cached module when an earlier test
      in the session already imported it;
    * the probe **evicts** each name from ``sys.modules`` before importing it,
      because importing one module routinely drags in a later sibling — nine of
      the 33 (``damping_matrix``, ``dofmap``, ``joint_evolutionary_fatigue``,
      ``linesearch``, ``linsolve``, ``multi_input_response``,
      ``multiaxial_fatigue``, ``spectral_nonproportional_fatigue``,
      ``statics``) arrive that way, and a cache hit for those would have made
      the test a no-op for a third of the surface;
    * the probe is **time-boxed and backend-pinned** (see ``_probe_env`` and
      ``_PROBE_TIMEOUT_S``), so a hanging import fails the gate and the verdict
      does not depend on the ambient session.

    So a failure here means the module's own body could not run — a real
    failure, with the traceback attached — not a naming accident. A module whose
    body is sound on first execution and raises on re-execution is caught: the
    eviction guarantees the body runs again.
    """
    from pyradioss.implicit import _frozen_surface as frozen

    report = _probe_report()

    assert not report["evict_failed"], (
        f"sys.modules eviction did not take effect: {report['evict_failed']}"
    )
    assert not report["failures"], (
        "frozen implicit module(s) failed to import:\n"
        + "\n".join(f"  {f['name']}:\n{f['traceback']}" for f in report["failures"])
    )
    assert not report["wrong_origin"], (
        "frozen name(s) resolved outside the implicit package:\n"
        + "\n".join(
            f"  {f['name']} -> {f['file']} (expected a file in {f['expected_dir']})"
            for f in report["wrong_origin"]
        )
    )
    # closes the hole a trivially-empty or fully-cached scan would leave open:
    # the probe must have scanned exactly the frozen set and re-executed all of it
    assert report["names"] == sorted(frozen.FROZEN_MODULES), {
        "probed": report["names"],
        "frozen": sorted(frozen.FROZEN_MODULES),
        "note": (
            "the probe scans with the test's own exclusion list, so a module "
            "the census does not know about shows up here as a mismatch"
        ),
    }
    assert sorted(report["executed"]) == sorted(
        f"pyradioss.implicit.{n}" for n in frozen.FROZEN_MODULES
    ), {
        "not_re_executed": sorted(
            f"pyradioss.implicit.{n}"
            for n in set(frozen.FROZEN_MODULES) - set(report["executed"])
        ),
        "note": (
            "a frozen module that did not re-register itself in sys.modules is "
            "unverified — the import test must not pass by cache hit"
        ),
    }


# --------------------------------------------------------------------------
# 4. the optional-dependency seam: presence + selection, and the wrappers
#    themselves — see the module docstring §4 for what this still cannot cover
# --------------------------------------------------------------------------


def _importable(dotted: str) -> bool:
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def test_optional_linsolve_backends_still_fall_back():
    """CHOLMOD/MUMPS stay OPTIONAL and stay wrapped — asserted, not skipped.

    ``plan/00_ORCHESTRATION.md`` §3: "The extras keep their existing
    ``pytest.importorskip`` guards for optional dependencies (CHOLMOD, MUMPS).
    Do not promote them to hard deps." Importing ``linsolve`` proves nothing
    about that, because the probes live inside functions. This test drives the
    selection API and checks both branches:

    * library absent  -> ``select_linsolve`` returns ``"superlu"`` and WARNS;
    * library present -> ``select_linsolve`` returns the requested name.

    A box with neither library installed is the common case and still gets a
    real assertion; nothing here skips. The *bodies* of the two backends are
    covered separately, by
    :func:`test_the_optional_linsolve_wrappers_execute_against_stand_ins`.
    """
    from pyradioss.implicit import linsolve

    solver_cls = linsolve.LinearSolver
    assert hasattr(solver_cls, "_solve_cholmod"), (
        "linsolve.LinearSolver lost its CHOLMOD path — the wrapper is the code "
        "this project owns (the library itself is not ported)"
    )
    assert hasattr(solver_cls, "_solve_mumps"), (
        "linsolve.LinearSolver lost its MUMPS path"
    )

    state_before = dict(linsolve._state)
    for requested, library in (
        ("cholmod", "sksparse.cholmod"),
        ("mumps", "mumps"),
        ("superlu-nonsense", None),  # typo / unknown name -> same fallback
    ):
        saved = dict(linsolve._state)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                active = linsolve.select_linsolve(requested)
            messages = [str(entry.message) for entry in caught]
        finally:
            linsolve._state.clear()
            linsolve._state.update(saved)

        if library is not None and _importable(library):
            assert active == requested, (
                f"{library} IS installed, so requesting {requested!r} must "
                f"activate it, got {active!r}"
            )
        else:
            assert active == "superlu", (
                f"select_linsolve({requested!r}) must fall back to superlu when "
                f"the optional library ({library}) is absent, got {active!r}"
            )
            assert messages, (
                f"the fallback for {requested!r} must WARN — a silent "
                "downgrade is how a run ends up on the wrong solver unnoticed"
            )

    assert dict(linsolve._state) == state_before, (
        "the selection probe left the resolved backend state mutated: "
        f"{state_before} -> {linsolve._state}"
    )


class _FakeMatrix:
    """A ``K`` that records the layout conversion without needing SciPy.

    ``_solve_cholmod`` / ``_solve_mumps`` only ever call ``K.tocsc()`` on their
    way to the library, so this is enough to pin *that they convert to CSC* on
    any box, including the NumPy-only one the implicit branch is designed to
    keep working on.
    """

    def __init__(self, tag: str):
        self.tag = tag

    def tocsc(self) -> str:
        return f"CSC({self.tag})"


def _stand_in(name: str, attrs: dict) -> types.ModuleType:
    """A module object that stands in for an uninstalled optional library.

    ``__spec__`` is set (with a null loader) so that any ``find_spec`` call
    elsewhere in the session behaves as it would for a real module rather than
    raising on a spec-less stand-in.
    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_the_optional_linsolve_wrappers_execute_against_stand_ins():
    """The CHOLMOD/MUMPS wrapper BODIES run here, against stand-in libraries.

    ``linsolve.py:457-474`` — eleven lines of project code that no test on this
    box can reach, because reaching them requires importing ``sksparse.cholmod``
    or ``mumps``, and neither library is installed here or in CI. So the test
    installs stand-ins for both in ``sys.modules``, removes them again, and
    drives the real dispatch (``LinearSolver.solve``) over them:

    * **CHOLMOD, factorable** — the wrapper hands the library a CSC conversion of
      ``K``, calls the factor with ``R``, and returns what the factor returned.
    * **CHOLMOD, not positive definite** — the library raises
      ``CholmodNotPositiveDefiniteError``; the wrapper must WARN and delegate to
      ``_solve_superlu`` rather than propagate the exception (that would abort a
      Newton step at a softening limit point) and must return the delegate's
      result. This fallback is the reason the wrapper exists and was previously
      executed nowhere.
    * **MUMPS** — the wrapper hands the library a CSC conversion of ``K`` with
      ``R`` and returns the result.

    **What this does not prove, and the debt it leaves.** The stand-ins are
    written to the shape this wrapper uses, so they cannot detect a mismatch
    against the *real* ``scikit-sparse`` / ``python-mumps`` APIs. Compatibility
    with those libraries is verified by review and by the maintainer action
    spelled out in this file's docstring §4, not here. What this test does prove
    is the half the project owns: the wrappers convert, call, propagate and
    degrade the way they are supposed to.
    """
    from pyradioss.implicit import linsolve

    seen: dict = {}

    class CholmodNotPositiveDefiniteError(Exception):
        """Stand-in for the exception ``sksparse.cholmod`` raises."""

    def cholesky(matrix):
        seen["cholmod_matrix"] = matrix
        if seen.get("cholmod_not_pd"):
            raise CholmodNotPositiveDefiniteError("stand-in: not positive definite")

        def factor(rhs):
            seen["cholmod_rhs"] = rhs
            return rhs * 3.0 + 1.0  # distinguishable from any real solve

        return factor

    def spsolve(matrix, rhs):
        seen["mumps_matrix"] = matrix
        seen["mumps_rhs"] = rhs
        return rhs * 5.0 - 2.0  # distinguishable from any real solve

    stand_ins = {
        "sksparse": _stand_in("sksparse", {}),
        "sksparse.cholmod": _stand_in(
            "sksparse.cholmod",
            {
                "cholesky": cholesky,
                "CholmodNotPositiveDefiniteError": CholmodNotPositiveDefiniteError,
            },
        ),
        "mumps": _stand_in("mumps", {"spsolve": spsolve}),
    }
    stand_ins["sksparse"].cholmod = stand_ins["sksparse.cholmod"]

    saved_modules = {name: sys.modules.get(name) for name in stand_ins}
    saved_state = dict(linsolve._state)
    try:
        sys.modules.update(stand_ins)

        solver = linsolve.LinearSolver(name="superlu")

        # -- CHOLMOD, factorable ------------------------------------------
        K = _FakeMatrix("K")
        R = np.array([1.0, 2.0, 3.0])
        seen.pop("cholmod_not_pd", None)
        solver.name = "cholmod"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = solver.solve(K, R)
        assert seen.get("cholmod_matrix") == "CSC(K)", (
            "the CHOLMOD wrapper must hand the library a CSC conversion of K, "
            f"got {seen.get('cholmod_matrix')!r} (linsolve.py:457-474)"
        )
        assert np.array_equal(np.asarray(seen.get("cholmod_rhs")), R), (
            "the CHOLMOD wrapper must call the factor with R unchanged, got "
            f"{seen.get('cholmod_rhs')!r}"
        )
        assert np.allclose(out, R * 3.0 + 1.0), (
            f"the CHOLMOD wrapper must return what the factor returned, got {out!r}"
        )
        assert not caught, (
            "a factorable CHOLMOD solve must not warn; it warned "
            f"{[str(w.message) for w in caught]}"
        )

        # -- CHOLMOD, tangent lost positive-definiteness ---------------------
        seen["cholmod_not_pd"] = True
        delegated: dict = {}
        real_superlu = solver._solve_superlu

        def recording_superlu(k, rhs):
            delegated["args"] = (k, rhs)
            return "SUPERLU-RESULT"

        solver._solve_superlu = recording_superlu
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = solver.solve(K, R)
        finally:
            solver._solve_superlu = real_superlu

        messages = [str(entry.message) for entry in caught]
        assert any("positive definite" in message for message in messages), (
            "a non-positive-definite CHOLMOD tangent must WARN before falling "
            f"back — a silent switch of solver is how a Newton step goes wrong "
            f"undetected; warnings were {messages}"
        )
        assert delegated.get("args") == (K, R), (
            "the CHOLMOD wrapper must delegate to _solve_superlu with the same "
            f"(K, R); it called {delegated.get('args')!r}"
        )
        assert out == "SUPERLU-RESULT", (
            "the CHOLMOD wrapper must return the SuperLU fallback's result, "
            f"got {out!r}"
        )

        # -- MUMPS ----------------------------------------------------------
        solver.name = "mumps"
        out = solver.solve(K, R)
        assert seen.get("mumps_matrix") == "CSC(K)", (
            "the MUMPS wrapper must hand the library a CSC conversion of K, got "
            f"{seen.get('mumps_matrix')!r} (linsolve.py:470-474)"
        )
        assert np.array_equal(np.asarray(seen.get("mumps_rhs")), R), (
            "the MUMPS wrapper must pass R unchanged, got "
            f"{seen.get('mumps_rhs')!r}"
        )
        assert np.allclose(out, R * 5.0 - 2.0), (
            f"the MUMPS wrapper must return what spsolve returned, got {out!r}"
        )
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        linsolve._state.clear()
        linsolve._state.update(saved_state)

    assert dict(linsolve._state) == saved_state, (
        "the stand-in probe left the resolved backend state mutated: "
        f"{saved_state} -> {linsolve._state}"
    )


# --------------------------------------------------------------------------
# 5. the record must carry its provenance (Phase 0 standard: a record that
#    does not say where it came from is a defect)
# --------------------------------------------------------------------------


def test_frozen_surface_records_its_provenance():
    from pyradioss.implicit import _frozen_surface as frozen

    doc = frozen.__doc__ or ""
    for needed, why in (
        ("pkgutil", "how the list was generated, not typed"),
        ("census", "the record must name the act that produced it"),
        ("2026-10-02", "the maintainer decision date this record encodes"),
        ("plan/00_ORCHESTRATION.md", "the binding decision being encoded"),
        ("maintainer", "ADDING requires approval; REMOVING is a hard failure"),
        ("hard failure", "same"),
        ("sha256", "how the pinned digest is recomputed"),
        ("EXCLUDED_FROM_CENSUS", "the scope rule must be named, not implied"),
        ("denylist", "the scope rule must say it is an explicit list"),
    ):
        assert needed in doc, (
            f"_frozen_surface.py's docstring must mention {needed!r} — {why}"
        )
    assert len(doc) > 800, (
        "the provenance record is a stub; Phase 0's standard is that a record "
        "which does not say where it came from is a defect"
    )
