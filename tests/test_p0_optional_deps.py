"""Task P0.7 — the optional (extra) dependency contract of this repo.

Three separate invariants live here, because "numba / mpi4py are optional" is
easy to state and easy to break:

1. **They stay OPTIONAL.**  ``pyproject.toml`` must keep the base install
   NumPy-only (``dependencies = ["numpy>=1.22"]``) — ``README.md`` promises a
   working NumPy-only install and ``numba``/``mpi4py`` may only appear under
   ``[project.optional-dependencies]``.  Promoting either to a base dep is a
   regression *even though it would fix the skips below*, so it is pinned by
   reading the TOML rather than by trying an install.

2. **The box reports honestly.**  ``test_optional_backends_import_or_report``
   SKIPS (never fails) when a dep cannot be imported, so a NumPy-only box can
   still run the suite; setting ``PYRADIOSS_ALLOW_MISSING_DEPS=0`` turns the
   same check into a hard gate (the companion
   ``test_allow_missing_deps_zero_is_a_hard_gate`` drives that path directly,
   so the gate is exercised even on a box where nothing is missing).
   A dep that is installed but *unusable* — ``mpi4py`` whose ``libmpi.so.12``
   cannot be resolved, say — is a FAILURE, not a skip: ``pyradioss/spmd/comm.py``
   guards its optional import with ``except ImportError``, so an ``OSError``
   leaking out of ``import mpi4py`` would escape every seam.

3. **The lock cannot rot silently.**  The resolved pins recorded in
   ``requirements-lock.txt`` (section [B], the shared interpreter P0.7 set up)
   must match what is actually importable, and a pinned-but-absent dep must
   carry an explicit ``NOT INSTALLED`` annotation so the gap stays documented
   instead of drifting into an undocumented lie.

No Fortran is involved (environment only), so there is nothing to cite from
``$OR_SRC``; the two upstream-adjacent claims this file rests on are instead
cited where they are enforced:

* ``pyradioss/accel/__init__.py:158-166`` — ``_load_numba_module`` imports
  numba *inside* the function, which is why the base install stays importable
  without it and why "numba missing" is a fallback, not a crash.
* ``pyradioss/spmd/comm.py:401-419`` — ``Mpi4pyComm.__init__`` does
  ``from mpi4py import MPI`` lazily; until P0.7 fix round 1 that bare import
  had NO handler at all, so any failure escaped to the driver;
  ``pyradioss/spmd/comm.py:495-509`` (``mpi_world_size``) had a bare
  ``except ImportError``.  Both now handle the whole "no usable MPI" family.
  That is why "installed but unloadable" is a test FAILURE here and not a
  skip: the port has to survive it, and the test is what proves it does.

Two further facts this file pins, both measured on this box (2026-10-03) by
downloading wheels WITHOUT installing them:

* **mpi4py is lazy.**  ``import mpi4py`` succeeds with no MPI runtime at all
  (4.1.2 ``__init__.py`` only installs an ABI finder); libmpi is resolved on
  the first ``from mpi4py import MPI``, which raises **RuntimeError**
  ("cannot load MPI library") on 4.x — NOT ``OSError``, and not at
  ``import mpi4py``.  (mpi4py 3.x links MPI at build time; its equivalent
  failure is an ``ImportError`` — not verified on this box.)
* **``pyradioss.accel.jit_kernels`` imports fine WITHOUT numba**
  (``jit_kernels/__init__.py:49-58`` degrades ``njit`` to an identity
  decorator).  So ``import pyradioss.accel.jit_kernels`` is NOT a numba
  availability probe — the real one is ``accel._load_numba_module``.

The tests are deliberately fast (a few milliseconds; one tiny njit compile)
and never touch the network — the resolved versions are read from files on
disk.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK = REPO_ROOT / "requirements-lock.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The optional (extra) dependencies this task is responsible for.  A pinned
#: module in this set that is NOT importable must carry a `NOT INSTALLED`
#: annotation in the lock — that is how the mpi4py gap stays documented.
OPTIONAL_MODULES = ("numba", "llvmlite", "mpi4py")

#: The base/environment pins the lock records as machine-verified too, so a
#: `pip install -U numpy` cannot leave the lock's prose quietly wrong.
BASE_MODULES = ("numpy", "scipy", "pytest")

#: Every module whose `# pin:` line the lock must carry and the test checks.
PINNED_MODULES = OPTIONAL_MODULES + BASE_MODULES

#: The interpreter the lock's `# pin: python==` line must agree with.
PINNED_PYTHON = "python"

#: The two the plan named explicitly (plan/00_ORCHESTRATION.md §4.3).
PLAN_MODULES = ("numba", "mpi4py")

GATE_ENV = "PYRADIOSS_ALLOW_MISSING_DEPS"


# ---------------------------------------------------------------------------
# probe helper
# ---------------------------------------------------------------------------
def probe(module: str) -> tuple[str, str]:
    """Import ``module`` and classify the outcome.

    Returns ``(state, detail)`` with ``state`` in:

    * ``"importable"``  — it imported (detail = ``__version__``);
    * ``"absent"``      — genuinely not installed (``ModuleNotFoundError``);
    * ``"unloadable"``  — installed but broken (e.g. mpi4py whose ``MPI``
      submodule raises ``RuntimeError: cannot load MPI library``); this is a
      failure of the install, not an absence.
    """
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # ModuleNotFoundError of the module ITSELF means "not installed";
        # one of its own imports missing means a broken install.
        if exc.name == module:
            return "absent", f"{module} is not installed"
        return "unloadable", f"{module} is installed but a dependency is missing: {exc}"
    except Exception as exc:  # OSError, ImportError, RuntimeError, ...
        return "unloadable", f"{module} is installed but does not import: {exc!r}"
    return "importable", getattr(mod, "__version__", "?")


def _gated() -> bool:
    """True when missing optional deps must FAIL instead of skipping."""
    return os.environ.get(GATE_ENV, "1").strip() in ("0", "false", "False")


# ---------------------------------------------------------------------------
# 1. reporting / gating
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", PLAN_MODULES)
def test_optional_backends_import_or_report(module):
    """The brief's test: report rather than fail, so a NumPy-only box stays
    green — and gate hard when ``PYRADIOSS_ALLOW_MISSING_DEPS=0``."""
    state, detail = probe(module)
    if state == "importable":
        return
    if state == "unloadable":
        # never skippable: a half-installed dep is exactly what the guarded
        # import seams in pyradioss/spmd/comm.py must survive
        pytest.fail(f"{detail} — install it properly or uninstall it")
    if _gated():
        pytest.fail(f"{detail} — but {GATE_ENV}=0 gates on it "
                    f"(see .superpowers/sdd/01_phase0_oracle_and_licensing/"
                    f"task-7-report.md)")
    pytest.skip(f"{detail} — set {GATE_ENV}=0 to gate")


def test_allow_missing_deps_zero_is_a_hard_gate(monkeypatch):
    """The companion to the reporting test: with ``PYRADIOSS_ALLOW_MISSING_DEPS=0``
    a missing dep must FAIL.  Driven through the probe + gate helper rather
    than by uninstalling a package, so the gate is covered on a box where
    everything happens to be installed."""
    monkeypatch.setenv(GATE_ENV, "0")
    assert _gated() is True
    monkeypatch.setenv(GATE_ENV, "1")
    assert _gated() is False
    monkeypatch.delenv(GATE_ENV)
    assert _gated() is False


def test_plan_modules_are_reported_in_the_session(monkeypatch):
    """With the gate OFF a missing dep produces a SKIP that names the module
    and the switch, so `pytest -rs` is the honest environment report the
    brief asks for.  Simulated by probing a module that cannot exist."""
    monkeypatch.delenv(GATE_ENV, raising=False)
    assert _gated() is False
    state, detail = probe("pyradioss_absent_optional_dep_probe")
    assert state == "absent"
    assert "not installed" in detail


# ---------------------------------------------------------------------------
# 2. what the extra must actually enable
# ---------------------------------------------------------------------------
def test_accel_kernels_compile_and_run_when_numba_present():
    """``import numba`` succeeding is necessary but not sufficient: the M7/M40
    backend only works if ``pyradioss.accel.jit_kernels`` loads and a kernel
    COMPILES and runs.  This is the one-second proof that the accel path is
    testable again (plan/00_ORCHESTRATION.md §4.3)."""
    state, detail = probe("numba")
    if state != "importable":
        if state == "unloadable":
            pytest.fail(detail)
        pytest.skip(f"{detail} — set {GATE_ENV}=0 to gate")

    from pyradioss import accel

    try:
        accel.select_backend("numba")
    except Exception as exc:  # pragma: no cover - only on a broken install
        pytest.fail(f"numba imported but accel.select_backend('numba') failed: {exc!r}")
    try:
        assert accel._state["name"] == "numba", "a pinned numba request must stick"
        assert accel.get("hexa_pre") is not None, "kernels are not served"
        import numpy as np

        # a real call at the signature solid_hexa8.py:727 uses
        # (_pre: srcoor3/sdefo3/srota3) — one element, compiles end-to-end
        jit = accel.get("hexa_pre")
        xi = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 1.0, -1.0],
                       [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                       [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        xe = xi[None, :, :]                     # 1 element, reference cube
        ve = np.zeros((1, 8, 3))
        sig = np.zeros((1, 8, 6))
        off = np.ones(1)
        dndx, vol, lc, deps, trD = jit(xe, ve, sig, 1e-5, off, np.ones(1))
        assert dndx.shape == (1, 8, 3)
        assert float(vol[0]) == pytest.approx(2.0)   # the 1x1x2 reference prism
        assert float(lc[0]) > 0.0
    finally:
        accel.select_backend("numpy")


def test_mpi_fallback_is_honest_about_the_gap():
    """Without a usable mpi4py the port must fall back to the in-process
    ``ThreadComm`` path (``pyradioss/spmd/comm.py``) rather than pretend to
    be SPMD: ``mpi_world_size()`` is 1 and no launcher env var is required.
    If mpi4py *does* import, this still holds off ``mpirun`` (that is the
    documented contract of the helper)."""
    from pyradioss.spmd import comm

    assert comm.mpi_world_size() == 1

    state, detail = probe("mpi4py")
    if state == "unloadable":
        pytest.fail(detail)
    if state == "absent" and _gated():
        pytest.fail(f"{detail} — but {GATE_ENV}=0 gates on it")


def test_mpi_world_size_falls_back_under_a_launcher_env_var(monkeypatch):
    """The launcher env var is present but MPI is not usable — the state this
    box would be in after a bare ``pip install mpi4py``.  ``mpi_world_size``
    must still answer 1 (so ``driver.run_engine_spmd`` takes the ThreadComm
    branch) instead of leaking mpi4py's own exception: 3.x raises
    ImportError, 4.x raises RuntimeError("cannot load MPI library"), and a
    libmpi that will not dlopen raises OSError.  Before P0.7 fix round 1 only
    ImportError was caught, so this test fails on the 4.x shape."""
    from pyradioss.spmd import comm

    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert comm.mpi_world_size() == 1


def test_mpi4py_comm_reports_an_unusable_mpi_instead_of_a_bare_traceback():
    """``Mpi4pyComm.__init__`` had a bare ``from mpi4py import MPI`` with no
    handler at all, so on a box with mpi4py-but-no-MPI the driver surfaced a
    raw RuntimeError from deep inside the import machinery.  It must now
    raise ONE actionable error naming the real fix."""
    from pyradioss.spmd import comm

    try:
        from mpi4py import MPI  # noqa: F401
    except Exception:
        pass          # no usable MPI here -> the guard path is what we test
    else:
        pytest.skip("a working MPI is present: Mpi4pyComm must succeed")

    with pytest.raises(RuntimeError) as excinfo:
        comm.Mpi4pyComm()
    msg = str(excinfo.value)
    assert "mpi4py's MPI module is not usable here" in msg
    assert "mpich" in msg.lower()          # the actionable half


def test_jit_kernels_are_not_a_numba_availability_probe():
    """``pyradioss/accel/jit_kernels/__init__.py:49-58`` degrades ``njit``
    to an identity decorator when numba is absent, so the module IMPORTS
    fine without numba.  Any ``HAS_NUMBA = importlib.util.find_spec(...)``
    style probe built on it therefore lies (the M7/M40 suites' ``needs_numba``
    marker never skipped because of this).  The honest probe is
    ``accel._load_numba_module``, which imports numba itself; pin both halves
    in a subprocess where numba is genuinely unimportable."""
    script = textwrap.dedent(
        """
        import sys
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "numba" or name.startswith("numba."):
                    raise ImportError("simulated: numba not installed")
                return None
        sys.meta_path.insert(0, _Block())
        import pyradioss.accel.jit_kernels as jk      # must NOT raise
        assert callable(jk.njit)
        from pyradioss import accel
        try:
            accel._load_numba_module()
        except ImportError:
            print("PROBE-OK")
        else:
            raise SystemExit("the probe loaded numba mirrors without numba")
        """
    )
    out = subprocess.run([sys.executable, "-c", script],
                         cwd=str(REPO_ROOT), capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "PROBE-OK" in out.stdout


# ---------------------------------------------------------------------------
# 3. pyproject: optional means optional
# ---------------------------------------------------------------------------
def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib", reason="needs Python >= 3.11")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("module", PLAN_MODULES)
def test_numba_and_mpi4py_are_only_optional_extras(module):
    """The base dependency list must stay NumPy-only: ``README.md`` promises
    a working NumPy-only install, and the brief forbids promoting these."""
    pyp = _pyproject()
    base = " ".join(pyp["project"]["dependencies"]).lower()
    assert module not in base, (
        f"{module} must NOT be a base dependency (README promises a "
        f"NumPy-only base install); it belongs in an extra")

    extras = pyp["project"]["optional-dependencies"]
    holders = [name for name, reqs in extras.items()
               if any(module in r.lower() for r in reqs)]
    assert holders, f"{module} is declared in no optional extra of pyproject.toml"


def test_base_install_is_numpy_only_and_declares_numpy():
    pyp = _pyproject()
    deps = pyp["project"]["dependencies"]
    assert len(deps) == 1, f"base deps drifted from NumPy-only: {deps}"
    assert deps[0].lower().startswith("numpy"), deps


def test_optional_extras_exist_for_accel_and_mpi():
    """The two extras the brief installs through must exist and be named."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "accel" in extras and "mpi" in extras


# ---------------------------------------------------------------------------
# 4. the lock file cannot rot
# ---------------------------------------------------------------------------
_PIN = re.compile(r"^#\s*pin:\s*([A-Za-z0-9_.\-]+)==([^\s#]+)(.*)$")


def _recorded_pins() -> dict[str, tuple[str, str]]:
    """The pins recorded in the lock's [B] section: ``name -> (version, note)``.

    Only lines of the form ``# pin: name==version[ # note]`` are read; the
    flat ``name==version`` lines of section [A] are the canonical Windows
    .venv lock (AGENTS.md) and are NOT this box's resolution.
    """
    pins = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        m = _PIN.match(line)
        if m:
            pins[m.group(1).lower()] = (m.group(2), m.group(3).strip())
    return pins


def test_every_pinned_module_is_recorded_in_the_lock():
    """Every module whose version the lock claims must carry a `# pin:`
    line — the optional deps (installed or not) AND the base ones.  Prose
    version numbers are not acceptable in [B]: they are the rot this test
    exists to kill, so the file may only state a version it can prove."""
    pins = _recorded_pins()
    for module in PINNED_MODULES + (PINNED_PYTHON,):
        assert module in pins, (
            f"{module} has no `# pin:` line in requirements-lock.txt; record the "
            f"resolved version (or an explicit NOT INSTALLED note)")


@pytest.mark.parametrize("module", PINNED_MODULES)
def test_recorded_pin_matches_what_is_importable(module):
    """The recorded version must be the importable one.  A pin that has
    drifted (upgrade, conda vs pip) fails here instead of quietly misleading
    the next agent.  For a pinned module that is ABSENT the lock must say so
    explicitly (``NOT INSTALLED``), which is what keeps the mpi4py gap a
    documented gap instead of a silent one."""
    pins = _recorded_pins()
    assert module in pins, (
        f"{module} has no `# pin:` line in requirements-lock.txt — record the "
        f"resolved version (or an explicit NOT INSTALLED note)")
    recorded, note = pins[module]
    state, detail = probe(module)
    if state == "absent":
        assert "NOT INSTALLED" in note, (
            f"{module} is not importable here ({detail}) but the lock records "
            f"{recorded} with no NOT INSTALLED annotation — document the gap")
    else:
        assert state == "importable", detail
        assert recorded == detail, (
            f"requirements-lock.txt records {module}=={recorded} but the "
            f"importable version is {detail}")


def test_recorded_python_pin_matches_the_running_interpreter():
    """The lock's ``# pin: python==`` line must be the interpreter actually
    running the suite — the version everything else in [B] was resolved on."""
    pins = _recorded_pins()
    assert PINNED_PYTHON in pins, "no `# pin: python==` line in the lock"
    recorded, _ = pins[PINNED_PYTHON]
    running = platform.python_version()
    assert recorded == running, (
        f"the lock records {PINNED_PYTHON}=={recorded} but the suite is "
        f"running {running}")