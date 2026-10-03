"""Task P0.6 — ``pyradioss.paths``, the single resource resolver.

Contract / upstream references pinned here:

* ``plan/00_ORCHESTRATION.md`` §4.1 — the environment variables and the
  four-step resolution order (env → sibling-of-build → Windows compat →
  **fail loudly**), and §1.2 — ``$OR_SRC`` is READ-ONLY.
* ``$OR_SRC/INSTALL.md:34-42`` — upstream's own Linux environment block
  (``OPENRADIOSS_PATH`` / ``RAD_CFG_PATH`` / ``RAD_H3D_PATH``); upstream
  spells the cfg tree ``RAD_CFG_PATH`` and ``$OPENRADIOSS_PATH/hm_cfg_files``
  — a **tree root**, which the resolver accepts as the documented spelling.
* ``.github/workflows/ci.yml:100,161`` — exports
  ``PYRADIOSS_HM_CFG=$GITHUB_WORKSPACE/or_cfg/hm_cfg_files/config/CFG``, the
  **schema directory** itself.  Both spellings are supported; a directory
  named ``CFG`` that carries no ``radioss<version>`` schemas is rejected.
* ``$OR_SRC/INSTALL.md:110`` — the binary name ``starter_linux64_gf``.
* ``docs/OPEN_BUGS.md`` item 6 — the LAW4 cfg-path bug this resolver
  closes: the reader used to fall through silently to a directory that
  does not exist, so ``/MAT/LAW4`` lost ``E`` and the test failed.

Convention chosen for ``missing_resource``: it **returns** a
``FileNotFoundError`` *instance* (it never raises itself), and every
resolver does ``raise missing_resource(...)``.  Reason: the brief types
it ``-> FileNotFoundError``, and a returned exception object composes —
the caller chooses whether to raise it or to log it (the ``/MAT`` reader
logs it instead of aborting a deck).  Both halves are pinned below: the
return-instance half here, the raising half through every resolver's
failure path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyradioss import paths

REPO_ROOT = Path(paths.__file__).resolve().parents[1]

_SCRUBBED = (
    "OR_SRC", "OR_ROOT", "OR_BUILD", "OR_STARTER", "OR_ENGINE",
    "PYRADIOSS_HM_CFG", "PYRADIOSS_RD_DECKS", "RAD_CFG_PATH",
    "OPENRADIOSS_PATH",
)


def _cfg_tree(base: Path, version: str = "radioss2022") -> Path:
    """A minimal but *real* cfg tree: ``<base>/config/CFG/radioss2022/MAT``.

    Real matters: the resolver accepts a directory only when it actually
    carries the incremental ``radioss<version>`` schema subdirectories, so a
    test that made an empty directory would prove nothing.
    """
    mat = base / "config" / "CFG" / version / "MAT"
    mat.mkdir(parents=True)
    (mat / "matl4_hyd_jcook.cfg").write_text("")
    return base


def _cfg_schema_dir(base: Path, version: str = "radioss2022") -> Path:
    """The schema directory itself — the spelling ``ci.yml`` exports."""
    root = base / "config" / "CFG"
    (root / version / "MAT").mkdir(parents=True)
    return root


def _prefix(tmp_path: Path, name: str = "prefix") -> Path:
    """An existing ``$OR_ROOT`` with nothing beside it unless a test says
    otherwise (so the OR_ROOT-derived tiers cannot fire by accident)."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Every test starts and ends with a clean environment, an empty
    resolver cache and a neutral install prefix, so no monkeypatched path
    can leak into another test file that runs after this one."""
    for var in _SCRUBBED:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(paths, "_dev_or_root", lambda: Path("/nonexistent/prefix"))
    # Same for the writable mirror's dev-box seam (``~/OpenRadioss_build``,
    # added by the P0 mirror-layout fix): without this the two or_build()
    # failure-path tests below would resolve against the real mirror on this
    # box and pass or fail depending on where it lives.  The candidate itself
    # is pinned, with its mirror predicate and its ordering, in
    # tests/test_p0_or_build_layout.py.
    monkeypatch.setattr(paths, "_dev_or_build",
                        lambda: Path("/nonexistent/mirror"))
    paths.reload()
    yield
    paths.reload()


# ---------------------------------------------------------------------------
# import-time contract
# ---------------------------------------------------------------------------

def test_import_never_raises_with_nothing_configured(tmp_path):
    """``import pyradioss.paths`` resolves nothing and cannot fail: the
    module must be importable with a scrubbed environment, from an
    unrelated working directory, before anything is resolved."""
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED}
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import pyradioss.paths as p; print(p.or_src.__module__)"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "pyradioss.paths" in proc.stdout


# ---------------------------------------------------------------------------
# (1) the environment variable wins
# ---------------------------------------------------------------------------

def test_or_src_from_env(monkeypatch, tmp_path):
    src = tmp_path / "OpenCourant"
    src.mkdir()
    monkeypatch.setenv("OR_SRC", str(src))
    assert paths.or_src() == src


def test_or_root_from_env(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_root() == root


def test_or_build_from_env(monkeypatch, tmp_path):
    build = tmp_path / "mirror"
    build.mkdir()
    monkeypatch.setenv("OR_BUILD", str(build))
    assert paths.or_build() == build


def test_or_starter_honours_the_env_override(monkeypatch, tmp_path):
    """``tools/oracle/oracle_env.sh`` exports ``OR_STARTER``/``OR_ENGINE``;
    an override that exists must win over ``$OR_ROOT/bin``."""
    exe = tmp_path / "starter_elsewhere"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("OR_STARTER", str(exe))
    assert paths.or_starter() == exe


def test_or_engine_honours_the_env_override(monkeypatch, tmp_path):
    exe = tmp_path / "engine_elsewhere"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("OR_ENGINE", str(exe))
    assert paths.or_engine() == exe


def test_hm_cfg_from_env(monkeypatch, tmp_path):
    cfg = _cfg_tree(tmp_path / "hm_cfg_files")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(cfg))
    assert paths.hm_cfg_dir() == cfg


def test_hm_cfg_accepts_the_cfg_directory_spelling(monkeypatch, tmp_path):
    """``PYRADIOSS_HM_CFG=.../hm_cfg_files/config/CFG`` — what
    ``.github/workflows/ci.yml:100,161`` exports.  The resolver accepts it
    as-is; ``mat_reader._find_cfg_root`` recognises the shape."""
    cfg = _cfg_schema_dir(tmp_path / "hm_cfg_files")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(cfg))
    assert paths.hm_cfg_dir() == cfg
    assert paths.is_cfg_schema_dir(paths.hm_cfg_dir())


def test_hm_cfg_accepts_the_upstream_rad_cfg_path_spelling(monkeypatch, tmp_path):
    """``$OR_SRC/INSTALL.md:39`` — upstream exports ``RAD_CFG_PATH``; a
    shell that sourced the upstream block instead of this repository's
    block must still resolve."""
    cfg = _cfg_tree(tmp_path / "hm_cfg_files")
    monkeypatch.setenv("RAD_CFG_PATH", str(cfg))
    assert paths.hm_cfg_dir() == cfg


def test_pyriass_hm_cfg_wins_over_rad_cfg_path(monkeypatch, tmp_path):
    ours = _cfg_tree(tmp_path / "ours")
    theirs = _cfg_tree(tmp_path / "theirs")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(ours))
    monkeypatch.setenv("RAD_CFG_PATH", str(theirs))
    assert paths.hm_cfg_dir() == ours


def test_rd_decks_from_env(monkeypatch, tmp_path):
    decks = tmp_path / "rd_decks"
    decks.mkdir()
    monkeypatch.setenv("PYRADIOSS_RD_DECKS", str(decks))
    assert paths.rd_decks_dir() == decks


# ---------------------------------------------------------------------------
# (2) sibling-of-build layout
# ---------------------------------------------------------------------------

def test_or_starter_defaults_to_the_install_prefix(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    exe = root / "bin" / "starter_linux64_gf"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_starter() == exe


def test_or_engine_defaults_to_the_install_prefix(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    exe = root / "bin" / "engine_linux64_gf"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_engine() == exe


def test_or_build_defaults_to_the_mirror_beside_the_install_prefix(
        monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    (root / "source").mkdir(parents=True)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_build() == root / "source"


def test_or_src_falls_back_to_the_sibling_of_the_build(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    upstream = tmp_path / "OpenCourant"
    upstream.mkdir()
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_src() == upstream


def test_hm_cfg_falls_back_to_the_sibling_of_the_build(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    cfg = _cfg_tree(tmp_path / "OpenCourant" / "hm_cfg_files")
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.hm_cfg_dir() == cfg


def test_hm_cfg_defaults_to_the_sibling_tree(monkeypatch, tmp_path):
    """Straight from the task brief: ``$OR_ROOT/OpenCourant/hm_cfg_files``
    is the brief's sibling tree and must be found without any cfg env var
    set."""
    monkeypatch.setenv("OR_ROOT", str(tmp_path))
    _cfg_tree(tmp_path / "OpenCourant" / "hm_cfg_files")
    paths.reload()
    assert paths.hm_cfg_dir() == tmp_path / "OpenCourant" / "hm_cfg_files"


def test_hm_cfg_falls_back_to_the_upstream_checkout_beside_the_repo(
        monkeypatch, tmp_path):
    """Last resort before failing: the upstream tree checked out next to
    this repository (``<repo>/../OpenCourant/hm_cfg_files``).  This is what
    makes a fresh shell with no exported variable work on the dev box."""
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr(paths, "_REPO_ROOT", fake_repo)
    monkeypatch.setenv("OR_ROOT", str(_prefix(tmp_path)))
    cfg = _cfg_tree(fake_repo.parent / "OpenCourant" / "hm_cfg_files")
    assert paths.hm_cfg_dir() == cfg


def test_or_root_falls_back_to_the_home_prefix(monkeypatch, tmp_path):
    """``~/OpenRadioss_or`` is the measured dev-box install prefix; it is
    the only candidate that can resolve ``OR_ROOT`` with no env at all."""
    monkeypatch.setattr(paths, "_dev_or_root",
                        lambda: tmp_path / "OpenRadioss_or")
    (tmp_path / "OpenRadioss_or").mkdir()
    assert paths.or_root() == tmp_path / "OpenRadioss_or"


def test_env_variable_pointing_at_a_missing_path_is_skipped(monkeypatch, tmp_path):
    """Rule 1 is *set AND existing*: a stale env var must not win — and it
    must say so."""
    root = _prefix(tmp_path)
    upstream = tmp_path / "OpenCourant"
    upstream.mkdir()
    monkeypatch.setenv("OR_SRC", str(tmp_path / "gone"))
    monkeypatch.setenv("OR_ROOT", str(root))
    with pytest.warns(RuntimeWarning, match="OR_SRC"):
        assert paths.or_src() == upstream


def test_a_set_but_missing_variable_is_reported_not_silently_skipped(
        monkeypatch, tmp_path):
    """Minor 7: a stale ``PYRADIOSS_RD_DECKS`` used to vanish without a
    trace, so a validation run could cover the small vendored corpus while
    the log claimed the full extract."""
    monkeypatch.setenv("PYRADIOSS_RD_DECKS", str(tmp_path / "gone"))
    with pytest.warns(RuntimeWarning, match="PYRADIOSS_RD_DECKS"):
        resolved = paths.rd_decks_dir()
    assert resolved == REPO_ROOT / "tests" / "data" / "rd_decks"


def test_a_set_cfg_variable_with_no_schemas_is_rejected_and_reported(
        monkeypatch, tmp_path):
    """An existing directory is not enough: it must carry the schemas, or
    accepting it would resolve a tree the reader can parse nothing from.
    Every other candidate is neutralised so this holds on any box (the
    repo-adjacent checkout exists on the dev box and nowhere else)."""
    bogus = tmp_path / "not_a_cfg_tree"
    (bogus / "config" / "CFG").mkdir(parents=True)
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(bogus))
    with pytest.warns(RuntimeWarning, match="PYRADIOSS_HM_CFG"):
        with pytest.raises(FileNotFoundError) as e:
            paths.hm_cfg_dir()
    assert str(bogus) in str(e.value)          # named as tried, never accepted


def test_or_build_never_resolves_to_the_read_only_source_tree(
        monkeypatch, tmp_path):
    """``$OR_SRC`` is READ-ONLY (``plan/00_ORCHESTRATION.md`` §1.2) and the
    build writes into the mirror, so ``or_build()`` must refuse to hand the
    upstream tree back even when it is the only tree around."""
    src = tmp_path / "OpenCourant"
    src.mkdir()
    monkeypatch.setenv("OR_SRC", str(src))
    with pytest.raises(FileNotFoundError):
        paths.or_build()


# ---------------------------------------------------------------------------
# precedence — §4.1 is an ORDER, so every neighbouring pair is pinned
# ---------------------------------------------------------------------------

def test_rule1_env_beats_rule2_sibling_for_or_src(monkeypatch, tmp_path):
    """§4.1 rule 1 (env) outranks rule 2 (sibling-of-build).  Transposing
    the two in ``or_src()`` must break this test."""
    root = _prefix(tmp_path)
    sibling = tmp_path / "OpenCourant"
    sibling.mkdir()
    exported = tmp_path / "exported"
    exported.mkdir()
    monkeypatch.setenv("OR_ROOT", str(root))
    monkeypatch.setenv("OR_SRC", str(exported))
    assert paths.or_src() == exported


def test_rule2_sibling_beats_rule3_windows_for_or_src(monkeypatch, tmp_path):
    """§4.1 rule 2 outranks rule 3 (Windows compat).  The Windows candidate
    is patched into existence because ``C:\\OpenRadioss`` cannot exist on
    Linux; it is a module attribute precisely so this order is testable."""
    root = _prefix(tmp_path)
    sibling = tmp_path / "OpenCourant"
    sibling.mkdir()
    win = tmp_path / "C_OpenRadioss"
    win.mkdir()
    monkeypatch.setattr(paths, "_WIN_ROOT", win)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_src() == sibling


def test_rule3_windows_compat_beats_the_dev_box_default_for_or_src(
        monkeypatch, tmp_path):
    """…and rule 3 outranks ``~/OpenRadioss_or``.  Deleting the Windows
    candidate from ``or_src()`` must break this test."""
    win = tmp_path / "C_OpenRadioss"
    win.mkdir()
    monkeypatch.setattr(paths, "_WIN_ROOT", win)
    monkeypatch.setattr(paths, "_dev_or_root", lambda: tmp_path / "OpenRadioss_or")
    (tmp_path / "OpenRadioss_or").mkdir()
    assert paths.or_root() == win


def test_rule1_env_beats_rule2_sibling_for_hm_cfg(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    _cfg_tree(tmp_path / "OpenCourant" / "hm_cfg_files")
    exported = _cfg_tree(tmp_path / "exported")
    monkeypatch.setenv("OR_ROOT", str(root))
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(exported))
    assert paths.hm_cfg_dir() == exported


def test_rule3_windows_compat_beats_the_extra_sibling_candidate(
        monkeypatch, tmp_path):
    """Important 2: the brief-only ``$OR_ROOT/OpenCourant/hm_cfg_files``
    must sit AFTER the §4.1 rule-3 Windows candidate.  The first version of
    this module had it before, which made the rule-3 path unreachable."""
    root = _prefix(tmp_path)
    _cfg_tree(root / "OpenCourant" / "hm_cfg_files")          # the extra
    win_cfg = _cfg_tree(tmp_path / "C_OpenRadioss" / "hm_cfg_files")
    monkeypatch.setattr(paths, "_WIN_CFG", win_cfg)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.hm_cfg_dir() == win_cfg


def test_rule2_sibling_beats_rule3_windows_for_hm_cfg(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    sibling_cfg = _cfg_tree(tmp_path / "OpenCourant" / "hm_cfg_files")
    win_cfg = _cfg_tree(tmp_path / "C_OpenRadioss" / "hm_cfg_files")
    monkeypatch.setattr(paths, "_WIN_CFG", win_cfg)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.hm_cfg_dir() == sibling_cfg


def test_the_repo_adjacent_checkout_never_shadows_the_env(monkeypatch, tmp_path):
    """The dev-box convenience tier is last: an exported variable, the
    sibling-of-build layout and the Windows path all outrank it.  Every
    competing candidate is built, so only the ordering decides."""
    fake_repo = tmp_path / "wt" / "repo"
    fake_repo.mkdir(parents=True)
    adjacent = _cfg_tree(fake_repo.parent / "OpenCourant" / "hm_cfg_files")
    root = _prefix(tmp_path)
    sibling = _cfg_tree(tmp_path / "OpenCourant" / "hm_cfg_files")
    win_cfg = _cfg_tree(tmp_path / "C_OpenRadioss" / "hm_cfg_files")
    monkeypatch.setattr(paths, "_REPO_ROOT", fake_repo)
    monkeypatch.setattr(paths, "_WIN_CFG", win_cfg)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.hm_cfg_dir() == sibling
    assert paths.hm_cfg_dir() not in (adjacent, win_cfg)


def test_the_repo_adjacent_checkout_never_shadows_the_exported_cfg(
        monkeypatch, tmp_path):
    fake_repo = tmp_path / "wt" / "repo"
    fake_repo.mkdir(parents=True)
    _cfg_tree(fake_repo.parent / "OpenCourant" / "hm_cfg_files")
    monkeypatch.setattr(paths, "_REPO_ROOT", fake_repo)
    exported = _cfg_tree(tmp_path / "exported")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(exported))
    assert paths.hm_cfg_dir() == exported


def test_or_starter_env_beats_the_install_prefix(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    exe = root / "bin" / "starter_linux64_gf"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("#!/bin/sh\n")
    monkeypatch.setenv("OR_ROOT", str(root))
    monkeypatch.setenv("OR_STARTER", str(elsewhere))
    assert paths.or_starter() == elsewhere


# ---------------------------------------------------------------------------
# (4) fail loudly — never silently degrade to a wrong directory
# ---------------------------------------------------------------------------

def test_missing_resource_names_every_attempted_location(monkeypatch):
    """Straight from the task brief."""
    monkeypatch.delenv("OR_SRC", raising=False)
    with pytest.raises(FileNotFoundError) as e:
        paths.reload()
        paths.or_src()
    msg = str(e.value)
    assert "OR_SRC" in msg and "$OR_ROOT/../OpenCourant" in msg


def test_or_src_failure_enumerates_all_candidates(monkeypatch, tmp_path):
    """Every candidate is listed *with its symbolic origin*, so the
    message names the contract location ($OR_ROOT/../OpenCourant) and not
    only the resolved path."""
    root = _prefix(tmp_path)              # resolvable, but has no sibling
    monkeypatch.setenv("OR_ROOT", str(root))
    with pytest.raises(FileNotFoundError) as e:
        paths.or_src()
    msg = str(e.value)
    assert "OR_SRC not found" in msg
    assert "[env OR_SRC]" in msg
    assert "not set in the environment" in msg
    assert "[$OR_ROOT/../OpenCourant]" in msg
    assert "C:\\OpenRadioss" in msg
    # the resolved path is printed too when it differs from the symbolic one
    assert str(tmp_path / "OpenCourant") in msg


def test_hm_cfg_failure_enumerates_all_candidates(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    monkeypatch.setenv("OR_ROOT", str(root))
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    with pytest.raises(FileNotFoundError) as e:
        paths.hm_cfg_dir()
    msg = str(e.value)
    assert "PYRADIOSS_HM_CFG" in msg
    assert "[env PYRADIOSS_HM_CFG]" in msg
    assert "[env RAD_CFG_PATH]" in msg
    assert "[$OR_ROOT/../OpenCourant/hm_cfg_files]" in msg
    assert "[C:\\OpenRadioss\\hm_cfg_files]" in msg
    # every candidate, contract first, extras last
    assert msg.index("[env PYRADIOSS_HM_CFG]") < \
        msg.index("[$OR_ROOT/../OpenCourant/hm_cfg_files]") < \
        msg.index("[C:\\OpenRadioss\\hm_cfg_files]") < \
        msg.index("$OR_ROOT/OpenCourant/hm_cfg_files")
    assert str(root) in msg                # the resolved sibling is printed


def test_or_root_failure_enumerates_all_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_dev_or_root", lambda: tmp_path / "absent")
    with pytest.raises(FileNotFoundError) as e:
        paths.or_root()
    msg = str(e.value)
    assert "[env OR_ROOT]" in msg
    assert "C:\\OpenRadioss" in msg
    assert "~/OpenRadioss_or (dev-box default)" in msg


def test_or_build_failure_enumerates_all_candidates(monkeypatch, tmp_path):
    """Nothing pinned the build-mirror failure before; it is the resource
    whose absence aborts every oracle build."""
    root = _prefix(tmp_path)              # no <root>/source
    monkeypatch.setenv("OR_ROOT", str(root))
    with pytest.raises(FileNotFoundError) as e:
        paths.or_build()
    msg = str(e.value)
    assert "OR_BUILD not found" in msg
    assert "[env OR_BUILD]" in msg
    assert "$OR_ROOT/source (writable mirror)" in msg
    assert str(root / "source") in msg
    assert "$OR_SRC" not in msg.splitlines()[-1]   # hint, not a candidate


def test_or_starter_failure_names_the_prefix_and_the_binary(monkeypatch, tmp_path):
    root = _prefix(tmp_path)
    (root / "bin").mkdir(parents=True)
    monkeypatch.setenv("OR_ROOT", str(root))
    with pytest.raises(FileNotFoundError) as e:
        paths.or_starter()
    msg = str(e.value)
    assert "OR_STARTER not found" in msg
    assert str(root / "bin" / "starter_linux64_gf") in msg
    assert "$OR_ROOT/exec/starter_win64.exe" in msg


def test_or_engine_failure_enumerates_all_candidates(monkeypatch, tmp_path):
    """Nothing pinned the engine-binary failure before."""
    root = _prefix(tmp_path)
    (root / "bin").mkdir(parents=True)
    monkeypatch.setenv("OR_ROOT", str(root))
    with pytest.raises(FileNotFoundError) as e:
        paths.or_engine()
    msg = str(e.value)
    assert "OR_ENGINE not found" in msg
    assert "[env OR_ENGINE]" in msg
    assert str(root / "bin" / "engine_linux64_gf") in msg
    assert "$OR_ROOT/exec/engine_win64.exe" in msg


def test_rd_decks_failure_is_loud_when_nothing_is_vendored(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    with pytest.raises(FileNotFoundError) as e:
        paths.rd_decks_dir()
    assert "PYRADIOSS_RD_DECKS not found" in str(e.value)


def test_unresolvable_or_root_nests_its_own_candidate_list(monkeypatch, tmp_path):
    """Minor 6: when ``OR_ROOT`` cannot be resolved the subordinate
    candidates must show *its* search, not a placeholder."""
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    with pytest.raises(FileNotFoundError) as e:
        paths.hm_cfg_dir()
    msg = str(e.value)
    assert "OR_ROOT unresolved; its own candidates:" in msg
    assert "OR_ROOT not found" in msg
    assert "[env OR_ROOT]" in msg
    assert "C:\\OpenRadioss" in msg
    assert "not set in the environment" in msg


def test_a_set_but_missing_or_root_is_not_reported_as_unset(monkeypatch, tmp_path):
    """Minor 6 again: "unset" was printed for a variable that was set and
    merely pointed at nothing."""
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    monkeypatch.setenv("OR_ROOT", str(tmp_path / "gone"))
    with pytest.warns(RuntimeWarning, match="OR_ROOT"):
        with pytest.raises(FileNotFoundError) as e:
            paths.hm_cfg_dir()
    nested = str(e.value)
    assert "OR_ROOT is unset" not in nested
    assert "not set in the environment" in nested    # RAD_CFG_PATH, not OR_ROOT
    assert f"{tmp_path / 'gone'}" in nested


# ---------------------------------------------------------------------------
# missing_resource itself: returns an instance, never raises
# ---------------------------------------------------------------------------

def test_missing_resource_returns_a_file_not_found_error_instance(tmp_path):
    tried = [tmp_path / "nope", tmp_path / "also-nope"]
    exc = paths.missing_resource("THING", tried)
    assert isinstance(exc, FileNotFoundError)
    assert not isinstance(exc, type)        # an instance, not the class
    msg = str(exc)
    assert msg.startswith("THING not found")
    for p in tried:
        assert str(p) in msg


def test_missing_resource_accepts_str_and_path_candidates(tmp_path):
    exc = paths.missing_resource("THING", [str(tmp_path / "a"), tmp_path / "b"])
    assert isinstance(exc, FileNotFoundError)
    assert str(tmp_path / "a") in str(exc)
    assert str(tmp_path / "b") in str(exc)


def test_missing_resource_accepts_an_origin_path_pair(tmp_path):
    exc = paths.missing_resource("THING", [("$SOMEWHERE/x", tmp_path / "x")])
    assert "$SOMEWHERE/x" in str(exc)
    assert str(tmp_path / "x") in str(exc)


def test_missing_resource_message_is_never_empty():
    exc = paths.missing_resource("THING", [])
    assert isinstance(exc, FileNotFoundError)
    assert "THING not found" in str(exc)


def test_missing_resource_docstring_states_the_convention():
    """Minor 8: later phases must not guess whether it raises."""
    doc = (paths.missing_resource.__doc__ or "").lower()
    assert "return" in doc and "never raise" in doc


# ---------------------------------------------------------------------------
# caching + reload()
# ---------------------------------------------------------------------------

def test_reload_picks_up_a_changed_environment(monkeypatch, tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    monkeypatch.setenv("OR_SRC", str(first))
    assert paths.or_src() == first

    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.setenv("OR_SRC", str(second))
    assert paths.or_src() == first           # still cached
    assert paths.reload() is None
    assert paths.or_src() == second          # re-read from the environment


def test_resolved_values_are_cached(monkeypatch, tmp_path):
    src = tmp_path / "OpenCourant"
    src.mkdir()
    monkeypatch.setenv("OR_SRC", str(src))
    assert paths.or_src() is paths.or_src()


def test_failures_are_not_cached(monkeypatch, tmp_path):
    """A resolver that failed must not freeze the failure: after the
    resource appears (a late build, a re-exported variable) it resolves."""
    empty = tmp_path / "empty-repo"
    empty.mkdir()
    monkeypatch.setattr(paths, "_REPO_ROOT", empty)
    with pytest.raises(FileNotFoundError):
        paths.rd_decks_dir()
    later = tmp_path / "later-repo"
    (later / "tests" / "data" / "rd_decks").mkdir(parents=True)
    monkeypatch.setattr(paths, "_REPO_ROOT", later)
    assert paths.rd_decks_dir() == later / "tests" / "data" / "rd_decks"


# ---------------------------------------------------------------------------
# the vendored deck corpus must resolve from any working directory
# ---------------------------------------------------------------------------

def test_rd_decks_resolves_the_vendored_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert paths.rd_decks_dir() == REPO_ROOT / "tests" / "data" / "rd_decks"


def test_rd_decks_is_independent_of_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    here = paths.rd_decks_dir()
    monkeypatch.chdir(REPO_ROOT)
    assert paths.rd_decks_dir() == here


def test_vendored_deck_corpus_really_exists():
    """Guards the resolver against silently pointing at an empty tree."""
    assert (REPO_ROOT / "tests" / "data" / "rd_decks").is_dir()


# ---------------------------------------------------------------------------
# the /MAT reader's cfg root: BOTH spellings, decided by the filesystem
# ---------------------------------------------------------------------------

def test_find_cfg_root_accepts_the_tree_root_spelling(monkeypatch, tmp_path):
    from pyradioss.input import mat_reader
    tree = _cfg_tree(tmp_path / "hm_cfg_files")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(tree))
    assert mat_reader._find_cfg_root() == str(tree / "config" / "CFG")


def test_find_cfg_root_accepts_the_cfg_directory_spelling(monkeypatch, tmp_path):
    """Critical 1: the exact spelling of ``.github/workflows/ci.yml:100``
    (``PYRADIOSS_HM_CFG=$GITHUB_WORKSPACE/or_cfg/hm_cfg_files/config/CFG``)
    must keep working — before this it was the pre-fix RED, 2 failed."""
    from pyradioss.input import mat_reader
    schema = _cfg_schema_dir(tmp_path / "or_cfg" / "hm_cfg_files")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(schema))
    root = mat_reader._find_cfg_root()
    assert root == str(schema)
    assert paths.is_cfg_schema_dir(root)


def test_find_cfg_root_rejects_a_cfg_directory_without_schemas(
        monkeypatch, tmp_path):
    """A directory named ``CFG`` that carries no ``radioss<version>``
    schemas must be rejected, not accepted — the reader would otherwise
    resolve a tree it can parse nothing from."""
    from pyradioss.input import mat_reader
    empty_cfg = tmp_path / "or_cfg" / "hm_cfg_files" / "config" / "CFG"
    empty_cfg.mkdir(parents=True)
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(empty_cfg))
    with pytest.warns(RuntimeWarning, match="PYRADIOSS_HM_CFG"):
        assert mat_reader._find_cfg_root() is None
    assert str(empty_cfg) in mat_reader._CFG_SEARCH_FAILED


def test_find_cfg_root_rejects_a_tree_with_an_empty_cfg_directory(
        monkeypatch, tmp_path):
    from pyradioss.input import mat_reader
    tree = tmp_path / "hm_cfg_files"
    (tree / "config" / "CFG").mkdir(parents=True)
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(tree))
    with pytest.warns(RuntimeWarning, match="PYRADIOSS_HM_CFG"):
        assert mat_reader._find_cfg_root() is None
    assert str(tree) in mat_reader._CFG_SEARCH_FAILED


def test_catalogue_follows_a_changed_environment_after_reload(
        monkeypatch, tmp_path):
    """Important 4: ``paths.reload()`` must reach the reader's catalogue
    singleton, or the first ``catalogue()`` call — which several test
    modules make at *collection* time — freezes the root for the session."""
    from pyradioss.input import mat_reader
    first = _cfg_tree(tmp_path / "first")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(first))
    paths.reload()
    cat1 = mat_reader.catalogue()
    assert cat1.root == str(first / "config" / "CFG")
    assert mat_reader.catalogue() is cat1          # stable while unchanged

    second = _cfg_tree(tmp_path / "second")
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(second))
    paths.reload()
    cat2 = mat_reader.catalogue()
    assert cat2.root == str(second / "config" / "CFG")
    assert cat2 is not cat1
    assert paths.hm_cfg_dir() == second


def test_catalogue_with_no_cfg_tree_has_no_root(monkeypatch, tmp_path):
    """No tree anywhere: the resolver raises, the reader catches it and the
    catalogue is built with ``root=None`` (the degradation path)."""
    from pyradioss.input import mat_reader
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    with pytest.raises(FileNotFoundError):
        paths.hm_cfg_dir()
    assert mat_reader.catalogue().root is None
    assert "PYRADIOSS_HM_CFG not found" in mat_reader._CFG_SEARCH_FAILED


def test_catalogue_explicit_root_is_not_re_resolved(tmp_path):
    """``CfgCatalogue(root)`` keeps the root it was given — ``None`` means
    "no tree", not "resolve it now"."""
    from pyradioss.input import mat_reader
    empty = mat_reader.CfgCatalogue(None)
    assert empty.root is None