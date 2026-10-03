"""The single resource resolver: every external path pyradioss needs.

Upstream Fortran / environment origin
-------------------------------------
This module ports **no formula**; it resolves the *installation layout*
upstream documents, so its authority is upstream's own environment block:

* ``$OR_SRC/INSTALL.md:34-42`` — "Environment variables settings under
  Linux": ``OPENRADIOSS_PATH`` (the install prefix), ``RAD_CFG_PATH=
  $OPENRADIOSS_PATH/hm_cfg_files`` (the CFG card-schema tree this port's
  ``/MAT``-``/PROP`` readers parse), ``RAD_H3D_PATH``,
  ``LD_LIBRARY_PATH=$OPENRADIOSS_PATH/extlib/hm_reader/linux64/``.
  ``$OR_SRC/INSTALL.md:44-50`` is the Windows spelling of the same block,
  which is why the Windows candidates below exist at all.
* ``$OR_SRC/INSTALL.md:110`` — the installed executable name
  ``starter_linux64_gf`` (``engine_linux64_gf`` beside it).

This repository's own contract (``plan/00_ORCHESTRATION.md`` §4.1) renames
and extends those variables to ``OR_SRC`` / ``OR_ROOT`` / ``OR_STARTER`` /
``OR_ENGINE`` / ``PYRADIOSS_HM_CFG`` / ``PYRADIOSS_RD_DECKS``.  That table,
not any hardcoded string, is what this module implements, in this order:

1. the environment variable, **if set and the path exists**;
2. ``$OR_ROOT/../OpenCourant`` and ``$OR_ROOT/../OpenCourant/hm_cfg_files``
   (sibling-of-build layout);
3. the Windows compatibility paths ``C:\\OpenRadioss`` and
   ``C:\\OpenRadioss\\hm_cfg_files``;
4. **fail loudly** — :func:`missing_resource` builds a
   ``FileNotFoundError`` listing every candidate.

Why rule 4 is the whole point
-----------------------------
``docs/OPEN_BUGS.md`` item 6: ``pyradioss/input/mat_reader.py`` looked the
cfg tree up in a tuple built from ``os.environ.get("PYRADIOSS_HM_CFG")`` at
**import time** plus two invented paths.  With none of them present it
returned ``None`` and the reader silently degraded to a heuristic parse, so
``/MAT/LAW4`` lost ``E`` and
``tests/test_m535_law04.py::test_direct_read_generic_mat_law4`` failed with
no clue why.  A wrong-but-existing directory is worse than a missing one;
therefore every candidate is *existence-checked*, every failure names every
location, and nothing ever falls through to a directory nobody verified.

Two candidates extend the contract above; both are marked in the failure
message and both are existence-checked like the rest:

* ``$OR_ROOT/OpenCourant/hm_cfg_files`` — the brief's sibling tree
  (``task-6-brief.md`` ``test_hm_cfg_defaults_to_the_sibling_tree``);
* ``<repo>/../OpenCourant/hm_cfg_files`` and ``~/OpenRadioss_or`` — the
  layout measured on the Linux dev box on 2026-10-03, so that a fresh
  shell with *no* exported variable resolves instead of failing.  They are
  placed **after** every contract candidate, so they can never override a
  rule-1/2/3 answer, and they are skipped entirely if the tree is not there.

Import contract: this module does **no** filesystem access at import time
and cannot fail at import — ``import pyradioss.paths`` works with nothing
configured.  Resolution is lazy, memoised per resource, and re-read after
:func:`reload`.

Convention: :func:`missing_resource` **returns** a ``FileNotFoundError``
instance and never raises it itself; every resolver does
``raise missing_resource(...)``.  A returned exception lets a caller that
must not abort (the ``/MAT`` reader logs it and keeps parsing) reuse the
exact same diagnostic the raising callers surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "or_src",
    "or_root",
    "or_build",
    "or_starter",
    "or_engine",
    "hm_cfg_dir",
    "rd_decks_dir",
    "missing_resource",
    "reload",
]

#: The package directory and the repository root, both derived from
#: ``__file__`` so resolution never depends on the working directory (a
#: test may run from anywhere).  ``_REPO_ROOT`` is the seam the tests
#: monkeypatch to simulate a checkout with no vendored corpus.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent

#: Upstream install prefix as shipped on Windows (``$OR_SRC/INSTALL.md:44-50``
#: and the pre-cmake ``exec/`` layout that ``tools/validate_vs_fortran.py:106``
#: hardcoded).  On POSIX these are inert single-component names and simply
#: never exist; they stay in the candidate list so the failure message is
#: the same on both platforms.
_WIN_ROOT = Path(r"C:\OpenRadioss")
_WIN_CFG = Path(r"C:\OpenRadioss\hm_cfg_files")

#: Name of the upstream checkout directory (``$OR_SRC``'s basename).
_UPSTREAM_DIRNAME = "OpenCourant"


# ---------------------------------------------------------------------------
# Candidate bookkeeping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Candidate:
    """One location a resource may live at.

    ``origin`` is the *symbolic* location from the contract
    (``$OR_ROOT/../OpenCourant``), printed verbatim in failure messages so
    a maintainer can match it against ``plan/00_ORCHESTRATION.md`` §4.1.
    ``path`` is the resolved absolute path, or ``None`` when the candidate
    could not even be constructed (the env var is unset, or ``OR_ROOT``
    itself is unresolvable).  ``note`` explains that case in the message.
    """

    origin: str
    path: Optional[Path] = None
    note: str = ""


Tried = Union[str, Path, _Candidate, Tuple[str, Path], Sequence[object]]


def _env_candidate(var: str) -> _Candidate:
    """Rule-1 candidate: ``var`` if set (existence is checked later)."""
    value = os.environ.get(var)
    if not value:
        return _Candidate(f"env {var}", None, "not set")
    return _Candidate(f"env {var}", Path(value))


def _dev_or_root() -> Path:
    """Dev-box install prefix, measured 2026-10-03: ``~/OpenRadioss_or``.

    A seam for the tests; the only candidate that can name ``OR_ROOT``
    when nothing is exported.
    """
    return Path(os.path.expanduser("~")) / "OpenRadioss_or"


def _checkout_hm_cfg() -> Path:
    """``<repo>/../OpenCourant/hm_cfg_files`` — the upstream checkout
    sitting beside this repository (the measured dev-box layout)."""
    return _REPO_ROOT.parent / _UPSTREAM_DIRNAME / "hm_cfg_files"


def _norm(path: Path) -> Path:
    """Collapse ``..`` lexically so a returned path compares equal to the
    location a maintainer would type."""
    return Path(os.path.normpath(str(path)))


def _resolve(name: str, candidates: Sequence[_Candidate],
             exists: Callable[[Path], bool]) -> Path:
    """Return the first candidate that exists; raise loudly if none does."""
    cached = _CACHE.get(name)
    if cached is not None:
        return cached
    tried: List[_Candidate] = []
    for cand in candidates:
        tried.append(cand)
        if cand.path is not None and exists(cand.path):
            resolved = _norm(cand.path)
            _CACHE[name] = resolved
            return resolved
    raise missing_resource(name, tried)


def _under_or_root(origin: str, *parts: str) -> _Candidate:
    """``$OR_ROOT/<parts...>``, listed even when ``OR_ROOT`` itself cannot
    be resolved — the symbolic origin is the useful diagnostic there."""
    root = _or_root_maybe()
    if root is None:
        return _Candidate(origin, None,
                          "OR_ROOT is unset and no install prefix resolved")
    return _Candidate(origin, _norm(root.joinpath(*parts)))


def _or_root_maybe() -> Optional[Path]:
    """``or_root()`` without the exception, for use inside a candidate
    list that must be buildable in order to *report* a failure."""
    try:
        return or_root()
    except FileNotFoundError:
        return None


def _or_src_maybe() -> Optional[Path]:
    try:
        return or_src()
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Public resolvers
# ---------------------------------------------------------------------------

def or_src() -> Path:
    """The read-only upstream OpenRadioss source tree ($OR_SRC)."""
    return _resolve("OR_SRC", [
        _env_candidate("OR_SRC"),
        _under_or_root("$OR_ROOT/../OpenCourant", "..", _UPSTREAM_DIRNAME),
        _Candidate(str(_WIN_ROOT), _WIN_ROOT),
    ], Path.is_dir)


def or_root() -> Path:
    """The out-of-tree build/install prefix of the oracle ($OR_ROOT)."""
    dev = _dev_or_root()
    return _resolve("OR_ROOT", [
        _env_candidate("OR_ROOT"),
        _Candidate(str(_WIN_ROOT), _WIN_ROOT),
        _Candidate("~/OpenRadioss_or (dev-box default)", dev),
    ], Path.is_dir)


def or_build() -> Path:
    """The writable mirror of the source tree ($OR_BUILD) — the only
    tree the build may write into; see ``tools/oracle/mirror_and_fetch.sh``.

    Deliberately never falls back to ``$OR_SRC``: the build writes into
    this tree (``load_extlib.py`` extracts there, and the starter/engine
    POST_BUILD rules create ``../exec``), and ``$OR_SRC`` is READ-ONLY by
    rule (``plan/00_ORCHESTRATION.md`` §1.2).
    """
    return _resolve("OR_BUILD", [
        _env_candidate("OR_BUILD"),
        _under_or_root("$OR_ROOT/source (writable mirror)", "source"),
    ], Path.is_dir)


def or_starter() -> Path:
    """The Fortran starter executable ($OR_STARTER,
    ``starter_linux64_gf`` per ``$OR_SRC/INSTALL.md:110``)."""
    return _resolve("OR_STARTER", [
        _env_candidate("OR_STARTER"),
        _under_or_root("$OR_ROOT/bin/starter_linux64_gf",
                       "bin", "starter_linux64_gf"),
        _under_or_root("$OR_ROOT/exec/starter_win64.exe "
                       "(pre-cmake Windows install)",
                       "exec", "starter_win64.exe"),
    ], Path.is_file)


def or_engine() -> Path:
    """The Fortran engine executable ($OR_ENGINE)."""
    return _resolve("OR_ENGINE", [
        _env_candidate("OR_ENGINE"),
        _under_or_root("$OR_ROOT/bin/engine_linux64_gf",
                       "bin", "engine_linux64_gf"),
        _under_or_root("$OR_ROOT/exec/engine_win64.exe "
                       "(pre-cmake Windows install)",
                       "exec", "engine_win64.exe"),
    ], Path.is_file)


def hm_cfg_dir() -> Path:
    """The ``hm_cfg_files`` CFG card-schema tree (PYRADIOSS_HM_CFG).

    Upstream reaches the same tree through ``RAD_CFG_PATH``
    (``$OR_SRC/INSTALL.md:39``), which is accepted as an alias.
    """
    derived = _or_src_maybe()
    return _resolve("PYRADIOSS_HM_CFG", [
        _env_candidate("PYRADIOSS_HM_CFG"),
        _env_candidate("RAD_CFG_PATH"),
        _under_or_root("$OR_ROOT/../OpenCourant/hm_cfg_files",
                       "..", _UPSTREAM_DIRNAME, "hm_cfg_files"),
        _under_or_root("$OR_ROOT/OpenCourant/hm_cfg_files",
                       _UPSTREAM_DIRNAME, "hm_cfg_files"),
        _Candidate(str(_WIN_CFG), _WIN_CFG),
        _Candidate("$OR_SRC/hm_cfg_files (from $OR_SRC)",
                   derived / "hm_cfg_files" if derived else None,
                   "" if derived else "OR_SRC is unset and did not resolve"),
        _Candidate("$OR_SRC/hm_cfg_files (upstream checkout beside this repo)",
                   _checkout_hm_cfg()),
    ], Path.is_dir)


def rd_decks_dir() -> Path:
    """The official RD-* deck corpus: PYRADIOSS_RD_DECKS if set, else the
    vendored ``tests/data/rd_decks`` (resolved from the package location,
    never from the working directory)."""
    return _resolve("PYRADIOSS_RD_DECKS", [
        _env_candidate("PYRADIOSS_RD_DECKS"),
        _Candidate("tests/data/rd_decks (vendored)",
                   _REPO_ROOT / "tests" / "data" / "rd_decks"),
    ], Path.is_dir)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_HINTS = {
    "OR_SRC": "export OR_SRC=<the OpenRadioss source tree> "
              "(read-only; see plan/00_ORCHESTRATION.md §4.1)",
    "OR_ROOT": "export OR_ROOT=<out-of-tree install prefix> "
               "(see tools/oracle/build_oracle.sh)",
    "OR_BUILD": "export OR_BUILD=<writable mirror> "
                "(see tools/oracle/mirror_and_fetch.sh)",
    "OR_STARTER": "export OR_STARTER, or build the oracle with "
                  "tools/oracle/build_oracle.sh",
    "OR_ENGINE": "export OR_ENGINE, or build the oracle with "
                 "tools/oracle/build_oracle.sh",
    "PYRADIOSS_HM_CFG": "export PYRADIOSS_HM_CFG=<hm_cfg_files>, or upstream's "
                         "RAD_CFG_PATH ($OR_SRC/INSTALL.md:39)",
    "PYRADIOSS_RD_DECKS": "export PYRADIOSS_RD_DECKS=<corpus>, or keep the "
                          "vendored tests/data/rd_decks in the checkout",
}


def _as_candidates(tried: Tried) -> List[_Candidate]:
    """Accept the documented ``list[Path]`` as well as the internal
    ``_Candidate`` / ``(origin, path)`` forms."""
    out: List[_Candidate] = []
    if tried is None:
        return out
    if isinstance(tried, (_Candidate, str, Path)):
        tried = [tried]
    for item in tried:
        if isinstance(item, _Candidate):
            out.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            out.append(_Candidate(str(item[0]), Path(item[1])))
        elif isinstance(item, (str, Path)):
            p = Path(item)
            out.append(_Candidate(str(p), p))
        else:                               # defensive: keep the text
            out.append(_Candidate(str(item), None, "unusable candidate"))
    return out


def _render(cand: _Candidate) -> str:
    if cand.path is None:
        return f"[{cand.origin}] <{cand.note or 'unresolved'}>"
    resolved = _norm(cand.path)
    if str(resolved) == cand.origin:
        return f"[{cand.origin}]"
    return f"[{cand.origin}] -> {resolved}"


def missing_resource(name: str, tried: Tried) -> FileNotFoundError:
    """Build — **return, do not raise** — the loud failure for ``name``.

    Every attempted location is listed with its symbolic origin from the
    contract and, when it differs, its resolved absolute path.  Resolvers
    call ``raise missing_resource(...)``; a caller that must not abort logs
    ``str(exc)`` instead.  (Convention chosen because the interface is
    typed ``-> FileNotFoundError``: returning keeps one diagnostic object
    usable both ways.)
    """
    candidates = _as_candidates(tried)
    lines = [f"{name} not found: none of the {len(candidates)} "
             f"candidate location{'s' if len(candidates) != 1 else ''} exists."]
    if candidates:
        lines.append("Tried:")
        lines.extend(f"  {_render(c)}" for c in candidates)
    lines.append(_HINTS.get(name, "set the environment variable, or fix "
                                  "the layout (plan/00_ORCHESTRATION.md §4.1)"))
    return FileNotFoundError("\n".join(lines))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def reload() -> None:
    """Drop the memoised resolutions and re-read the environment.

    Tests monkeypatch environment variables after import; the cache holds
    resolved paths, so it must be cleared for a changed variable to be
    observed.  Import itself resolves nothing, so ``reload()`` is a no-op
    on a fresh interpreter.
    """
    _CACHE.clear()