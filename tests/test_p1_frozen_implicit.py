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

The three invariants here, and what each can actually detect
-------------------------------------------------------------
1. :func:`test_implicit_surface_is_frozen` — the live ``pkgutil`` scan equals
   ``FROZEN_MODULES``, reported in BOTH directions (``added`` /
   ``removed``) so the failure names the culprit instead of just saying "sets
   differ". Detects: a module added, a module deleted, a module renamed.
   Cannot detect: a change *inside* a module (that is the milestone suite's
   job, not this file's).

2. :func:`test_frozen_surface_sha256_is_recomputable` — the pinned digest is
   recomputed here from the live scan and compared. Without it the digest is
   decoration: a constant nobody recomputes cannot detect a wrong constant.

3. :func:`test_frozen_implicit_modules_still_import` — every frozen module
   imports **in a cold interpreter**, i.e. in a subprocess, so no earlier test
   in the session can have put it into ``sys.modules`` and turned the check
   into a cache hit. This is strictly stronger than importing in-process, and
   it also checks each module's origin file really sits in the package
   directory (a name that resolves to a stray module elsewhere on ``sys.path``
   is a failure, not a pass).

   **What test 3 CANNOT detect, stated plainly.** Importability is not
   solvability. Every implicit module obtains SciPy through
   ``pyradioss.implicit.require_scipy()`` (a *call*, ``__init__.py:241-260``)
   rather than a top-level ``import scipy``, so a broken SciPy path is invisible
   to an import probe — by design, because the base explicit install is
   NumPy-only. The same is true of the two optional linear-solver backends:
   CHOLMOD (``sksparse.cholmod``) and MUMPS (``mumps``) are probed *inside*
   ``linsolve.select_linsolve`` (``linsolve.py:72-87``) and imported inside
   ``LinearSolver._solve_cholmod`` / ``._solve_mumps``
   (``linsolve.py:457-474``). There is **no** ``pytest.importorskip`` inside the
   package — the guards are in the *tests* (``tests/test_m8_implicit.py:30``,
   ``test_m10_impdyn.py:42``, and nine siblings) and in ``require_scipy``.

   So "import every module" would pass vacuously with respect to those optional
   dependencies — importing ``linsolve`` says nothing about whether its CHOLMOD
   path still works. :func:`test_optional_linsolve_backends_still_fall_back`
   is what closes that hole: it asserts the optional-backend code is still
   *present and reachable* (``_solve_cholmod`` / ``_solve_mumps`` exist) and
   that selecting an unavailable backend **warns and falls back to SuperLU**
   instead of raising — the exact contract ``plan/00_ORCHESTRATION.md`` §3
   protects ("The extras keep their existing ``pytest.importorskip`` guards for
   optional dependencies (CHOLMOD, MUMPS). Do not promote them to hard
   deps"). Where the optional library *is* installed, the same test asserts the
   backend really activates, so it can never pass by skipping.

   The honest residual limit: with ``sksparse``/``mumps`` absent, the numerical
   CHOLMOD/MUMPS kernels themselves are unexecuted here. They are wrapped, not
   ported (``linsolve.py:24-31``), so the wrapper contract is what this project
   owns — and that contract is now asserted.

No Fortran is involved (project structure only), so nothing here is cited from
``$OR_SRC``; ``$OR_SRC`` is READ-ONLY and untouched by this task.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pkgutil
import subprocess
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REMEDIATION = (
    "Adding a name to pyradioss/implicit/ is a deliberate act that needs the "
    "maintainer's approval (plan/00_ORCHESTRATION.md §3: no new features there "
    "except what Phase 15 needs for parity with upstream "
    "engine/source/implicit). With that approval, regenerate FROZEN_MODULES in "
    "pyradioss/implicit/_frozen_surface.py from a live pkgutil scan and update "
    "FROZEN_SURFACE_SHA256 with the digest that scan prints. Removing a name is "
    "a hard failure and is never 'fixed' by editing this list."
)


def _live_surface() -> set[str]:
    """The public module names currently in ``pyradioss/implicit/``."""
    import pyradioss.implicit as implicit_pkg

    return {
        info.name
        for info in pkgutil.iter_modules(implicit_pkg.__path__)
        if not info.name.startswith("_")
    }


def _sha256_of_surface(names) -> str:
    """The documented digest recipe, implemented independently of the module.

    ``sha256("\\n".join(sorted(names)) + "\\n")`` — POSIX-text shaped (one name
    per line, trailing newline) so the digest is stable and reproducible by
    hand from a terminal. It must match ``_surface_digest`` in
    ``pyradioss/implicit/_frozen_surface.py``.
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


def test_frozen_surface_is_non_empty_and_free_of_private_names():
    """A census that silently shrank to nothing must not read as agreement."""
    from pyradioss.implicit import _frozen_surface as frozen

    assert len(frozen.FROZEN_MODULES) >= 30, (
        "the frozen implicit surface collapsed to "
        f"{len(frozen.FROZEN_MODULES)} modules; a census that small is a "
        "census that lost the tree, not a decision"
    )
    private = sorted(n for n in frozen.FROZEN_MODULES if n.startswith("_"))
    assert not private, (
        f"private names are outside the census by rule: {private}"
    )
    assert "_frozen_surface" not in frozen.FROZEN_MODULES, (
        "this module is private and excluded from its own census; a name "
        "inside it would make the digest self-referential"
    )


# --------------------------------------------------------------------------
# 2. the pinned digest must be verifiable
# --------------------------------------------------------------------------


def test_frozen_surface_sha256_is_recomputable():
    """Recompute ``FROZEN_SURFACE_SHA256`` from the LIVE scan and compare.

    A constant nobody recomputes is decoration. This test is deliberately
    downstream of test 1 — if the tree and the pinned set disagree, both fail,
    which is the intended behaviour (a changed tree is wrong whether or not
    its digest happens to match).
    """
    from pyradioss.implicit import _frozen_surface as frozen

    live = _live_surface()
    recomputed = _sha256_of_surface(live)

    assert recomputed == frozen.FROZEN_SURFACE_SHA256, {
        "recomputed_from_live_scan": recomputed,
        "pinned": frozen.FROZEN_SURFACE_SHA256,
        "live_count": len(live),
        "note": (
            "the digest is over the SORTED names, newline-joined with a "
            "trailing newline; see _surface_digest in _frozen_surface.py"
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

_COLD_IMPORT_PROBE = r'''
import importlib, json, os, pkgutil, sys, traceback

root = sys.argv[1]
sys.path.insert(0, root)

import pyradioss.implicit as implicit_pkg

pkg_dir = os.path.dirname(os.path.abspath(implicit_pkg.__file__))
names = sorted(
    info.name
    for info in pkgutil.iter_modules(implicit_pkg.__path__)
    if not info.name.startswith("_")
)

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
    try:
        module = importlib.import_module(dotted)
    except BaseException:
        report["failures"].append(
            {"name": name, "traceback": traceback.format_exc()})
        continue
    new = sys.modules.get(dotted)
    if new is not old:
        report["executed"].append(dotted)
    origin = os.path.abspath(getattr(module, "__file__", None) or "")
    if os.path.dirname(origin) != pkg_dir:
        report["wrong_origin"].append(
            {"name": name, "file": origin or "<none>", "expected_dir": pkg_dir})

print("PYRADIOSS_FROZEN_PROBE:" + json.dumps(report))
'''


def _probe_report() -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _COLD_IMPORT_PROBE, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    marker = "PYRADIOSS_FROZEN_PROBE:"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise AssertionError(
        "the cold-import probe produced no report "
        f"(returncode {proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def test_frozen_implicit_modules_still_import():
    """Every frozen module's body executes from a cold ``sys.modules``, from its own file.

    Two hardening details, both learned from a first version of this test that
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
      the test a no-op for a third of the surface.

    So a failure here means the module's own body could not run — a real
    failure, with the traceback attached — not a naming accident.
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
    }
    assert sorted(report["executed"]) == sorted(
        f"pyradioss.implicit.{n}" for n in frozen.FROZEN_MODULES
    ), {
        "not_re_executed": sorted(
            f"pyradioss.implicit.{n}"
            for n in set(frozen.FROZEN_MODULES) - set(report["executed"])
        ),
        "note": (
            "a frozen module whose body did not actually execute in this probe "
            "is unverified — the import test must not pass by cache hit"
        ),
    }


# --------------------------------------------------------------------------
# 4. the optional-dependency seam must not pass by skipping
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
    real assertion; nothing here skips.
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
    ):
        assert needed in doc, (
            f"_frozen_surface.py's docstring must mention {needed!r} — {why}"
        )
    assert len(doc) > 800, (
        "the provenance record is a stub; Phase 0's standard is that a record "
        "which does not say where it came from is a defect"
    )