"""Task P0.6 — ``pyradioss.paths``, the single resource resolver.

Contract / upstream references pinned here:

* ``plan/00_ORCHESTRATION.md`` §4.1 — the environment variables and the
  four-step resolution order (env → sibling-of-build → Windows compat →
  **fail loudly**).
* ``$OR_SRC/INSTALL.md:34-42`` — upstream's own Linux environment block
  (``OPENRADIOSS_PATH`` / ``RAD_CFG_PATH`` / ``RAD_H3D_PATH``); upstream
  spells the cfg tree ``RAD_CFG_PATH``, which the resolver accepts as an
  alias for ``PYRADIOSS_HM_CFG``.
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


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Every test starts and ends with a clean environment and an empty
    resolver cache, so no monkeypatched path can leak into another test
    file that runs after this one."""
    for var in _SCRUBBED:
        monkeypatch.delenv(var, raising=False)
    paths.reload()
    yield
    paths.reload()


# ---------------------------------------------------------------------------
# import-time contract
# ---------------------------------------------------------------------------

def test_import_never_raises_with_nothing_configured(tmp_path):
    """``import pyradioss.paths`` does no filesystem access and cannot
    fail: the module must be importable with a scrubbed environment, from
    an unrelated working directory, before anything is resolved."""
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
    root = tmp_path / "prefix"
    root.mkdir()
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
    cfg = tmp_path / "hm_cfg_files"
    cfg.mkdir()
    monkeypatch.setenv("PYRADIOSS_HM_CFG", str(cfg))
    assert paths.hm_cfg_dir() == cfg


def test_hm_cfg_accepts_the_upstream_rad_cfg_path_spelling(monkeypatch, tmp_path):
    """``$OR_SRC/INSTALL.md:39`` — upstream exports ``RAD_CFG_PATH``; a
    shell that sourced the upstream block instead of this repository's
    block must still resolve."""
    cfg = tmp_path / "hm_cfg_files"
    cfg.mkdir()
    monkeypatch.setenv("RAD_CFG_PATH", str(cfg))
    assert paths.hm_cfg_dir() == cfg


def test_pyriass_hm_cfg_wins_over_rad_cfg_path(monkeypatch, tmp_path):
    ours = tmp_path / "ours"
    ours.mkdir()
    theirs = tmp_path / "theirs"
    theirs.mkdir()
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

def test_or_build_never_resolves_to_the_read_only_source_tree(
        monkeypatch, tmp_path):
    """``$OR_SRC`` is READ-ONLY (``plan/00_ORCHESTRATION.md`` §1.2) and the
    build writes into the mirror, so ``or_build()`` must refuse to hand the
    upstream tree back even when it is the only tree around."""
    src = tmp_path / "OpenCourant"
    src.mkdir()
    monkeypatch.setenv("OR_SRC", str(src))
    monkeypatch.setattr(paths, "_dev_or_root", lambda: tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        paths.or_build()


def test_or_starter_defaults_to_the_install_prefix(monkeypatch, tmp_path):
    root = tmp_path / "prefix"
    (root / "bin").mkdir(parents=True)
    exe = root / "bin" / "starter_linux64_gf"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_starter() == exe


def test_or_engine_defaults_to_the_install_prefix(monkeypatch, tmp_path):
    root = tmp_path / "prefix"
    (root / "bin").mkdir(parents=True)
    exe = root / "bin" / "engine_linux64_gf"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_engine() == exe


def test_or_build_defaults_to_the_mirror_beside_the_install_prefix(
        monkeypatch, tmp_path):
    root = tmp_path / "prefix"
    (root / "source").mkdir(parents=True)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_build() == root / "source"


def test_or_src_falls_back_to_the_sibling_of_the_build(monkeypatch, tmp_path):
    root = tmp_path / "prefix"
    root.mkdir()
    upstream = tmp_path / "OpenCourant"
    upstream.mkdir()
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_src() == upstream


def test_hm_cfg_falls_back_to_the_sibling_of_the_build(monkeypatch, tmp_path):
    root = tmp_path / "prefix"
    root.mkdir()
    cfg = tmp_path / "OpenCourant" / "hm_cfg_files"
    cfg.mkdir(parents=True)
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.hm_cfg_dir() == cfg


def test_hm_cfg_defaults_to_the_sibling_tree(monkeypatch, tmp_path):
    """Straight from the task brief: ``$OR_ROOT/OpenCourant/hm_cfg_files``
    is the dev-box/prefix layout and must be found without any cfg env
    var set."""
    monkeypatch.setenv("OR_ROOT", str(tmp_path))
    (tmp_path / "OpenCourant" / "hm_cfg_files").mkdir(parents=True)
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
    prefix = tmp_path / "prefix"
    prefix.mkdir()                       # OR_ROOT exists but has no sibling
    monkeypatch.setenv("OR_ROOT", str(prefix))
    cfg = fake_repo.parent / "OpenCourant" / "hm_cfg_files"
    cfg.mkdir(parents=True)
    assert paths.hm_cfg_dir() == cfg


def test_or_root_falls_back_to_the_home_prefix(monkeypatch, tmp_path):
    """``~/OpenRadioss_or`` is the measured dev-box install prefix; it is
    the only candidate that can resolve ``OR_ROOT`` with no env at all."""
    monkeypatch.setattr(paths, "_dev_or_root",
                        lambda: tmp_path / "OpenRadioss_or")
    (tmp_path / "OpenRadioss_or").mkdir()
    assert paths.or_root() == tmp_path / "OpenRadioss_or"


def test_env_variable_pointing_at_a_missing_path_is_skipped(monkeypatch, tmp_path):
    """Rule 1 is *set AND existing*: a stale env var must not win."""
    root = tmp_path / "prefix"
    root.mkdir()
    upstream = tmp_path / "OpenCourant"
    upstream.mkdir()
    monkeypatch.setenv("OR_SRC", str(tmp_path / "gone"))
    monkeypatch.setenv("OR_ROOT", str(root))
    assert paths.or_src() == upstream


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
    root = tmp_path / "prefix"          # resolvable, but has no sibling
    root.mkdir()
    monkeypatch.setenv("OR_ROOT", str(root))
    with pytest.raises(FileNotFoundError) as e:
        paths.or_src()
    msg = str(e.value)
    assert "OR_SRC not found" in msg
    assert "[env OR_SRC]" in msg
    assert "<not set>" in msg
    assert "[$OR_ROOT/../OpenCourant]" in msg
    assert "C:\\OpenRadioss" in msg
    # the resolved path is printed too when it differs from the symbolic one
    assert str(tmp_path / "OpenCourant") in msg


def test_hm_cfg_failure_enumerates_all_candidates(monkeypatch, tmp_path):
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    monkeypatch.setenv("OR_ROOT", str(prefix))
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    with pytest.raises(FileNotFoundError) as e:
        paths.hm_cfg_dir()
    msg = str(e.value)
    assert "PYRADIOSS_HM_CFG" in msg
    assert "[env PYRADIOSS_HM_CFG]" in msg
    assert "[env RAD_CFG_PATH]" in msg
    assert "[$OR_ROOT/../OpenCourant/hm_cfg_files]" in msg
    assert "[C:\\OpenRadioss\\hm_cfg_files]" in msg
    assert str(prefix) in msg          # the resolved sibling path is printed


def test_or_root_failure_enumerates_all_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_dev_or_root", lambda: tmp_path / "absent")
    with pytest.raises(FileNotFoundError) as e:
        paths.or_root()
    msg = str(e.value)
    assert "[env OR_ROOT]" in msg
    assert "C:\\OpenRadioss" in msg


def test_or_starter_failure_names_the_prefix_and_the_binary(monkeypatch, tmp_path):
    root = tmp_path / "prefix"
    (root / "bin").mkdir(parents=True)
    monkeypatch.setenv("OR_ROOT", str(root))
    with pytest.raises(FileNotFoundError) as e:
        paths.or_starter()
    msg = str(e.value)
    assert "OR_STARTER not found" in msg
    assert str(root / "bin" / "starter_linux64_gf") in msg


def test_rd_decks_failure_is_loud_when_nothing_is_vendored(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-repo-here")
    with pytest.raises(FileNotFoundError) as e:
        paths.rd_decks_dir()
    assert "PYRADIOSS_RD_DECKS not found" in str(e.value)


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


def test_missing_resource_message_is_never_empty():
    exc = paths.missing_resource("THING", [])
    assert isinstance(exc, FileNotFoundError)
    assert "THING not found" in str(exc)


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