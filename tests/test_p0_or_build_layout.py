"""The writable mirror's documented location must resolve, and its absence must
be loud.

``pyradioss.paths.or_build()`` listed two candidate locations: the environment
variable and ``$OR_ROOT/source``.  Neither is where the mirror is on this box.
``$OR_ROOT/source`` (``/home/valentin/OpenRadioss_or/source``) does not exist --
``tools/validation_data/oracle_provenance.json`` already records that the
pre-migration path was *removed*, and the cmake evidence for the location that
replaced it is ``build/starter/CMakeCache.txt:CMAKE_HOME_DIRECTORY``, which
names ``/home/valentin/OpenRadioss_build``.  The project's own documents agree:

* ``plan/01_phase0_oracle_and_licensing.md:669`` -- "``upstream.mirror_path``
  corrected to ``$OR_BUILD`` (``= /home/valentin/OpenRadioss_build``)";
* ``plan/01_phase0_oracle_and_licensing.md:829`` and ``docs/STATE.md:54,835`` --
  the same value as *the* ``OR_BUILD`` of this project;
* the exit gate at ``plan/01_phase0_oracle_and_licensing.md:826-830`` exports it
  explicitly, so a configured run was never in doubt.

So the resolver was incomplete for the documented dev-box layout while the
*other* dev-box prefixes were not: ``or_root()`` has carried a
``~/OpenRadioss_or`` default since P0.6 and ``hm_cfg_dir()`` has two measured
candidates (``paths.py`` module docstring, "Candidates beyond the contract"), on
the stated principle that a fresh shell with *no* exported variable resolves
instead of failing.  The mirror was the one gap, and it cost real coverage:
with nothing exported, ``oracle_paths()`` could not find ``libhm_reader`` and
seven tests skipped on a resource this box has.

What this file pins, so the gap cannot reopen silently:

* **A configured mirror is found.**  Rule 1 (§4.1) and the dev-box default both
  resolve; the contract candidate still outranks the extra one, so an added
  candidate can never shadow ``$OR_ROOT/source``
  (:func:`test_the_contract_candidate_outranks_the_dev_box_default`).
* **An absent mirror is loud.**  §4.1 rule 4 requires the failure to list every
  attempted location: :func:`test_an_absent_mirror_is_reported_loudly_naming_
  every_candidate` checks all three origins, their resolved paths and the hint,
  and :func:`test_the_candidate_list_is_pinned` freezes the count so a future
  edit cannot quietly drop a location nobody notices missing.
* **The extra candidate cannot find the wrong directory.**  ``~/OpenRadioss_build``
  is a *discovered* location, not a configured one, so it must prove it is a
  mirror before it is accepted: a ``CMakeLists.txt`` and an
  ``extlib/hm_reader`` directory, the two things ``tools/oracle/build_oracle.sh``
  itself gates on (``:169``, ``:175``).  Silently binding the build to a
  same-named directory that is not a mirror would be worse than not resolving,
  which is why :func:`test_a_directory_that_is_not_a_mirror_is_refused` and its
  missing-extlib sibling exist.
* **The coverage win is measured, not asserted.**  :func:`test_or_build_
  resolves_in_a_bare_shell` runs a *fresh interpreter* with the oracle
  variables scrubbed and requires the mirror to resolve there -- the difference
  between this and a unit test of the candidate list is that it is the state a
  contributor's shell is actually in.

Nothing here edits an existing test.  ``tests/test_p0_paths.py`` keeps its own
copy of the isolation it needs: its autouse fixture neutralises the dev-box
seams (``_dev_or_root``), and this change adds the second seam it must also
neutralise.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pyradioss import paths

REPO_ROOT = Path(paths.__file__).resolve().parents[1]

#: The *unpatched* dev-box seam, captured at import -- before the isolation
#: fixture below replaces it.  :func:`test_or_build_is_found_in_a_bare_shell`
#: needs the real default and no machine path written out as a literal.
_REAL_DEV_OR_BUILD = paths._dev_or_build

#: The variables that name an oracle resource; scrubbed so no test here can
#: pass on the contributor's shell instead of on the code.
_SCRUBBED = (
    "OR_SRC", "OR_ROOT", "OR_BUILD", "OR_STARTER", "OR_ENGINE",
    "PYRADIOSS_HM_CFG", "RAD_CFG_PATH", "OPENRADIOSS_PATH",
)

#: The two markers ``tools/oracle/build_oracle.sh`` refuses to go without
#: (``:169`` "no CMakeLists.txt", ``:175`` the reader tree).
MIRROR_MARKER = "CMakeLists.txt"
MIRROR_EXTLIB = ("extlib", "hm_reader")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No exported variable, no machine candidate, empty resolver cache.

    Mirrors ``tests/test_p0_paths.py``'s own isolation: the candidate seams are
    the machine-dependent part of this resolver, and a test that leaves the
    real ``~/OpenRadioss_or`` or ``~/OpenRadioss_build`` in play proves nothing
    about the list under test.
    """
    for var in _SCRUBBED:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(paths, "_dev_or_root",
                        lambda: Path("/nonexistent/prefix"))
    monkeypatch.setattr(paths, "_dev_or_build",
                        lambda: Path("/nonexistent/mirror"))
    paths.reload()
    yield
    paths.reload()


def _mirror(base: Path) -> Path:
    """A directory that *is* a mirror, by ``build_oracle.sh``'s own two gates.

    ``git archive``d tree + a fetched ``extlib``: ``CMakeLists.txt`` at the top
    and ``extlib/hm_reader`` beneath it.  Built for real rather than mocked, so
    the predicate is exercised against the filesystem it will be run against.
    """
    base.mkdir(parents=True, exist_ok=True)
    (base / MIRROR_MARKER).write_text("# mirror\n")
    (base.joinpath(*MIRROR_EXTLIB)).mkdir(parents=True, exist_ok=True)
    return base


def _candidates() -> list[str]:
    """The symbolic origins ``or_build()`` reports, by asking for a failure.

    Read out of the loud failure rather than out of the source: the message is
    the contract (§4.1 rule 4), so pinning it pins both.  Only the top-level
    entries are collected -- ``missing_resource`` renders them with a two-space
    indent and nests a candidate list four deep, so the indentation is what
    separates "a location ``or_build`` tried" from "a location ``OR_ROOT``
    tried on its way to being unresolvable".
    """
    with pytest.raises(FileNotFoundError) as exc:
        paths.or_build()
    return re.findall(r"^  \[(.+?)\]", str(exc.value), re.M)


# ---------------------------------------------------------------------------
# 1. a configured mirror is found
# ---------------------------------------------------------------------------

def test_or_build_finds_a_configured_mirror(monkeypatch, tmp_path):
    """Rule 1: the environment variable wins, and wins by existing.

    §4.1 rule 1 is "set **and** existing", so the variable is honoured even
    when the directory does not look like a mirror -- a configured location is
    the maintainer's decision, not this resolver's to second-guess.
    """
    build = tmp_path / "configured-mirror"
    build.mkdir()
    monkeypatch.setenv("OR_BUILD", str(build))
    paths.reload()
    assert paths.or_build() == build


def test_or_build_finds_the_dev_box_default(monkeypatch, tmp_path):
    """With nothing exported the measured dev-box mirror resolves.

    This is the gap: before the candidate existed, a bare shell raised here and
    every consumer that needs the mirror's ``extlib`` (th_to_csv, the h3d
    writer, ``libhm_reader``) degraded to a skip.
    """
    mirror = _mirror(tmp_path / "OpenRadioss_build")
    monkeypatch.setattr(paths, "_dev_or_build", lambda: mirror)
    paths.reload()
    assert paths.or_build() == mirror


def test_the_contract_candidate_outranks_the_dev_box_default(monkeypatch,
                                                            tmp_path):
    """``$OR_ROOT/source`` still wins over the extra candidate.

    The order rule the module states for every candidate beyond §4.1: placed
    **after** rule 3, so no extra can shadow a contract candidate.  Adding a
    fourth candidate must not quietly move it to the front.
    """
    root = tmp_path / "prefix"
    contract = _mirror(root / "source")
    dev = _mirror(tmp_path / "OpenRadioss_build")
    monkeypatch.setenv("OR_ROOT", str(root))
    monkeypatch.setattr(paths, "_dev_or_build", lambda: dev)
    paths.reload()
    assert paths.or_build() == contract


def test_or_build_is_found_in_a_bare_shell(monkeypatch):
    """The end-to-end claim: a *fresh interpreter*, no exported variable.

    Every other test here patches the seams; this one restores the real
    dev-box default, so it measures the state a contributor's shell is in --
    which is where the coverage was being lost.  Skipped loudly, never
    silently, when the mirror is not on this box: there is no candidate to find
    and nothing to prove.
    """
    mirror = _REAL_DEV_OR_BUILD()
    if not (mirror / MIRROR_MARKER).is_file() \
            or not mirror.joinpath(*MIRROR_EXTLIB).is_dir():
        pytest.skip(f"the dev-box mirror is not at {mirror} "
                    f"(needs {MIRROR_MARKER} and "
                    f"{'/'.join(MIRROR_EXTLIB)}); nothing to resolve")
    monkeypatch.setattr(paths, "_dev_or_build", _REAL_DEV_OR_BUILD)
    paths.reload()
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED}
    env["PYTHONPATH"] = str(REPO_ROOT)
    done = subprocess.run(
        [sys.executable, "-c",
         "from pyradioss import paths; print(paths.or_build())"],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        timeout=120)
    assert done.returncode == 0, (
        f"or_build() does not resolve with nothing exported "
        f"(exit {done.returncode}):\n{done.stderr}")
    assert Path(done.stdout.strip()) == mirror, (
        f"a bare shell resolved the mirror to {done.stdout.strip()!r}, not "
        f"the measured {mirror}")


# ---------------------------------------------------------------------------
# 2. the extra candidate may not bind the wrong directory
# ---------------------------------------------------------------------------

def test_a_directory_that_is_not_a_mirror_is_refused(monkeypatch, tmp_path):
    """``~/OpenRadioss_build``-shaped, but no ``CMakeLists.txt``: refused.

    ``build_oracle.sh:169`` refuses such a directory outright, so accepting it
    here would resolve to a tree the build then rejects -- a "found" mirror that
    cannot be built is exactly the silent wrong path §4.1 rule 4 exists to
    prevent.
    """
    impostor = tmp_path / "OpenRadioss_build"
    impostor.mkdir()
    monkeypatch.setattr(paths, "_dev_or_build", lambda: impostor)
    paths.reload()
    with pytest.raises(FileNotFoundError) as exc:
        paths.or_build()
    assert str(impostor) in str(exc.value)


def test_a_mirror_whose_extlib_was_never_fetched_is_refused(monkeypatch,
                                                             tmp_path):
    """``CMakeLists.txt`` present, ``extlib/hm_reader`` absent: still refused.

    The other half of ``build_oracle.sh``'s two gates (``:175``): a tree that
    carries no reader cannot start either binary, and resolving it would let
    ``oracle_paths()`` report a mirror whose ``libhm_reader`` does not exist.
    """
    half = tmp_path / "OpenRadioss_build"
    half.mkdir()
    (half / MIRROR_MARKER).write_text("# mirror\n")
    monkeypatch.setattr(paths, "_dev_or_build", lambda: half)
    paths.reload()
    with pytest.raises(FileNotFoundError):
        paths.or_build()


# ---------------------------------------------------------------------------
# 3. absence is loud -- §4.1 rule 4
# ---------------------------------------------------------------------------

def test_an_absent_mirror_is_reported_loudly_naming_every_candidate(
        monkeypatch, tmp_path):
    """A set-but-nonexistent ``OR_BUILD``: every location, every reason."""
    bogus = tmp_path / "not-a-mirror"
    monkeypatch.setenv("OR_BUILD", str(bogus))
    root = tmp_path / "prefix"           # exists, but holds no `source/`
    root.mkdir()
    monkeypatch.setenv("OR_ROOT", str(root))
    paths.reload()
    with pytest.raises(FileNotFoundError) as exc:
        paths.or_build()
    message = str(exc.value)
    assert message.startswith("OR_BUILD not found:"), message
    for origin in ("env OR_BUILD", "$OR_ROOT/source (writable mirror)",
                   "~/OpenRadioss_build"):
        assert origin in message, f"{origin} missing from:\n{message}"
    assert str(bogus) in message          # the configured-but-absent path
    assert str(root / "source") in message
    assert "/nonexistent/mirror" in message
    assert "mirror_and_fetch.sh" in message, message      # the actionable hint
    assert "3 candidate locations" in message, message


def test_the_candidate_list_is_pinned():
    """Exactly three origins, in order: env, contract, dev box.

    The count and the order are the contract a maintainer matches against
    §4.1.  Pinning them means a later edit that adds, drops or reorders a
    location fails here instead of turning into a skip nobody reads.
    """
    assert _candidates() == [
        "env OR_BUILD",
        "$OR_ROOT/source (writable mirror)",
        "~/OpenRadioss_build (dev-box default)",
    ]


def test_a_set_but_missing_or_build_is_refused_not_stepped_over(monkeypatch,
                                                                 tmp_path):
    """Stale export: loud, never a silent skip.

    Realigned by the P0 whole-branch review (Finding 2, the Critical behind
    Finding 1): this test required the *warn-then-step-over* behaviour, which is
    the silently-degrading path ``plan/00_ORCHESTRATION.md`` §9.1 item 9
    rejects and ``plan/01_phase0_oracle_and_licensing.md:720`` contradicts ("a
    *stale export* -> fail, not skip").  With ``LD_LIBRARY_PATH`` exported by
    ``tools/oracle/oracle_env.sh`` the oracle then *ran* against the mirror
    substituted here, so a bogus ``OR_BUILD`` produced a green run.

    Nothing was dropped: the mirror below is still the answer for a shell that
    exports nothing, which is now asserted explicitly instead of by accident.
    """
    mirror = _mirror(tmp_path / "OpenRadioss_build")
    monkeypatch.setattr(paths, "_dev_or_build", lambda: mirror)
    monkeypatch.setenv("OR_BUILD", str(tmp_path / "gone"))
    paths.reload()
    with pytest.warns(RuntimeWarning, match="OR_BUILD"):
        with pytest.raises(FileNotFoundError) as exc:
            paths.or_build()
    message = str(exc.value)
    assert message.startswith("OR_BUILD not found:"), message
    assert "[env OR_BUILD]" in message and str(tmp_path / "gone") in message
    # the mirror a silent fallthrough would have used is named as NOT taken
    assert str(mirror) in message and "not tried" in message
    # ... and an unset variable still resolves through the same candidate
    monkeypatch.delenv("OR_BUILD")
    paths.reload()
    assert paths.or_build() == mirror


def test_or_build_never_resolves_to_the_read_only_source_tree(monkeypatch,
                                                               tmp_path):
    """``$OR_SRC`` is READ-ONLY (§1.2) and the build writes into the mirror.

    With a real source tree as the only tree around, resolution must still
    fail rather than hand the upstream tree back.
    """
    src = tmp_path / "OpenCourant"
    src.mkdir()
    src.joinpath(*MIRROR_EXTLIB).mkdir(parents=True)
    (src / MIRROR_MARKER).write_text("# not a mirror of anything\n")
    monkeypatch.setenv("OR_SRC", str(src))
    paths.reload()
    with pytest.raises(FileNotFoundError):
        paths.or_build()


def test_the_dev_box_seam_is_a_seam_and_can_be_patched():
    """``_dev_or_build`` mirrors ``_dev_or_root``: a function, not a constant.

    A constant here would make the candidate untestable without the real mirror
    present, which is how machine-dependent candidates rot into skips.
    """
    assert callable(paths._dev_or_build)
    assert not isinstance(paths._dev_or_build, (str, Path))