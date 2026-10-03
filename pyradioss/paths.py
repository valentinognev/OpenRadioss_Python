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

Every candidate is existence-checked against a **resource-specific
predicate**, never against "the string looks like a path": for
:func:`hm_cfg_dir` a directory only counts if it really carries the cfg
schemas (see :func:`is_cfg_tree`).  A wrong-but-existing directory is worse
than a missing one, and a *set but missing* environment variable is
reported with a :func:`warnings.warn` so a validation run cannot quietly
cover a smaller corpus than it claims.

Candidates beyond the contract, and their exact position
-------------------------------------------------------
Three candidates are not in §4.1.  All three are placed **after rule 3**,
so no extra can ever shadow a contract candidate (a reviewer's Important-2
finding — the first version of this module put the brief's
``$OR_ROOT/OpenCourant/hm_cfg_files`` *before* the Windows candidate and
the rule-3 path was therefore unreachable on a box where both existed):

* ``$OR_ROOT/OpenCourant/hm_cfg_files`` — the brief's sibling tree
  (``task-6-brief.md`` ``test_hm_cfg_defaults_to_the_sibling_tree``);
* ``$OR_SRC/hm_cfg_files`` derived from a resolved ``$OR_SRC``;
* ``<repo>/../OpenCourant/hm_cfg_files`` and ``~/OpenRadioss_or`` — the
  layout measured on the Linux dev box on 2026-10-03, so that a fresh shell
  with *no* exported variable resolves instead of failing.  They are skipped
  entirely when the tree is not there.

So the candidate order for ``hm_cfg_dir`` is: ``$PYRADIOSS_HM_CFG``,
``$RAD_CFG_PATH`` (upstream's own spelling, ``INSTALL.md:39``),
``$OR_ROOT/../OpenCourant/hm_cfg_files``, ``C:\\OpenRadioss\\hm_cfg_files``,
``$OR_ROOT/OpenCourant/hm_cfg_files``, ``$OR_SRC/hm_cfg_files``,
``<repo>/../OpenCourant/hm_cfg_files``.  ``tests/test_p0_paths.py`` pins
every one of those precedence relations by name.

Import contract
---------------
Importing this module touches only ``__file__`` (one ``Path.resolve()``,
i.e. the ``stat``/``readlink`` syscalls that entails) and the process
environment; it never *resolves* a resource, never walks a tree, and cannot
fail — ``import pyradioss.paths`` works with nothing configured.  Resolution
is lazy, memoised per resource, and re-read after :func:`reload`.

Convention for :func:`missing_resource`
---------------------------------------
It **returns** a ``FileNotFoundError`` instance; it does **not** raise.  Every
resolver here does ``raise missing_resource(...)``.  Keep it that way: the
brief types it ``-> FileNotFoundError``, and a returned exception object
composes — a caller that must not abort (the ``/MAT`` reader logs it and
keeps parsing) reuses the exact diagnostic the raising callers surface.
Calling ``missing_resource(...)`` and *expecting* a raise is a bug; the
return value has to be raised explicitly.
"""

from __future__ import annotations

import os
import re
import warnings
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
    "is_cfg_schema_dir",
    "is_cfg_tree",
    "CFG_VERSION_DIR_RE",
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
#: the same on both platforms.  Both are monkeypatched by the precedence
#: tests, which is the only way to exercise rule 3 off Windows.
_WIN_ROOT = Path(r"C:\OpenRadioss")
_WIN_CFG = Path(r"C:\OpenRadioss\hm_cfg_files")

#: Name of the upstream checkout directory (``$OR_SRC``'s basename).
_UPSTREAM_DIRNAME = "OpenCourant"

#: A CFG schema directory holds one subdirectory per incremental format
#: version (``radioss2022``, ``radioss110``, ...); ``mat_reader``'s
#: ``CfgCatalogue._scan`` reads exactly those, newest first.
CFG_VERSION_DIR_RE = re.compile(r"radioss\d+")


def is_cfg_schema_dir(path) -> bool:
    """True when ``path`` is a CFG **schema** directory — it holds at
    least one ``radiossNNN`` version subdirectory.

    A directory merely *named* ``CFG`` is not accepted: an empty or partial
    sparse checkout would otherwise resolve and then parse nothing, which is
    the silent-degradation failure mode this module exists to remove.
    """
    try:
        entries = os.listdir(str(path))
    except OSError:
        return False
    return any(CFG_VERSION_DIR_RE.fullmatch(name) for name in entries)


def is_cfg_tree(path) -> bool:
    """True for **either** spelling of the cfg location:

    * the documented one, a tree root whose ``config/CFG`` holds the
      schemas (``$OR_SRC/hm_cfg_files``, ``plan/00_ORCHESTRATION.md`` §4.1,
      upstream's ``RAD_CFG_PATH`` = ``$OPENRADIOSS_PATH/hm_cfg_files``);
    * the schema directory itself (``.../hm_cfg_files/config/CFG``) — what
      ``.github/workflows/ci.yml`` has always exported, and what the old
      ``mat_reader._DEFAULT_CFG_ROOTS`` tuple accepted.

    Which of the two a caller got is decided by inspecting the filesystem
    (``mat_reader._find_cfg_root``), never by the shape of the string.
    """
    p = Path(str(path))
    return is_cfg_schema_dir(p / "config" / "CFG") or is_cfg_schema_dir(p)


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
    itself is unresolvable).  ``env_var`` names the variable when this
    candidate came from one, so a *set but missing* variable can be warned
    about instead of skipped in silence.  ``note`` explains an unconstructed
    candidate in the failure message.
    """

    origin: str
    path: Optional[Path] = None
    note: str = ""
    env_var: str = ""


@dataclass(frozen=True)
class _Tried:
    """A candidate plus why it was rejected."""

    candidate: _Candidate
    reason: str = ""


Tried = Union[str, Path, _Candidate, _Tried, Tuple[str, Path],
              Sequence[object]]


def _env_candidate(var: str) -> _Candidate:
    """Rule-1 candidate: ``var`` if set (the predicate is checked later)."""
    value = os.environ.get(var)
    if not value:
        return _Candidate(f"env {var}", None, "not set in the environment",
                          env_var=var)
    return _Candidate(f"env {var}", Path(value), env_var=var)


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
             exists: Callable[[Path], bool],
             kind: str = "directory") -> Path:
    """Return the first candidate the predicate accepts; else raise loudly.

    A variable that *is* set but points at nothing usable is announced with
    :func:`warnings.warn` — otherwise a stale ``PYRADIOSS_RD_DECKS`` lets a
    validation run cover the small vendored corpus while the log claims the
    full extract.
    """
    cached = _CACHE.get(name)
    if cached is not None:
        return cached
    tried: List[_Tried] = []
    for cand in candidates:
        if cand.path is None:
            tried.append(_Tried(cand, cand.note or "unresolved"))
            continue
        if not exists(cand.path):
            reason = f"{_norm(cand.path)}: {kind} does not exist"
            if cand.env_var:
                warnings.warn(
                    f"{cand.env_var} is set to {cand.path} but that {kind} "
                    f"does not exist — ignored; pyradioss is falling "
                    f"through to the next candidate location",
                    RuntimeWarning, stacklevel=3)
            tried.append(_Tried(cand, reason))
            continue
        resolved = _norm(cand.path)
        _CACHE[name] = resolved
        return resolved
    raise missing_resource(name, tried)


def _under_or_root(origin: str, *parts: str) -> _Candidate:
    """``$OR_ROOT/<parts...>``.

    When ``OR_ROOT`` itself cannot be resolved the candidate is still listed
    — its symbolic origin is the useful diagnostic — and it *nests*
    ``OR_ROOT``'s own candidate list so the message shows the whole search,
    not a placeholder.
    """
    root, nested = _or_root_soft()
    if root is None:
        return _Candidate(origin, None,
                          "OR_ROOT unresolved; its own candidates:\n" + nested)
    return _Candidate(origin, _norm(root.joinpath(*parts)))


def _or_root_soft() -> Tuple[Optional[Path], str]:
    """``(root, nested-report)`` — never raises, because these candidates
    must be buildable in order to *report* a failure of something else."""
    try:
        return or_root(), ""
    except FileNotFoundError as exc:
        return None, str(exc)


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
    ], Path.is_file, kind="file")


def or_engine() -> Path:
    """The Fortran engine executable ($OR_ENGINE)."""
    return _resolve("OR_ENGINE", [
        _env_candidate("OR_ENGINE"),
        _under_or_root("$OR_ROOT/bin/engine_linux64_gf",
                       "bin", "engine_linux64_gf"),
        _under_or_root("$OR_ROOT/exec/engine_win64.exe "
                       "(pre-cmake Windows install)",
                       "exec", "engine_win64.exe"),
    ], Path.is_file, kind="file")


def hm_cfg_dir() -> Path:
    """The ``hm_cfg_files`` CFG card-schema tree (PYRADIOSS_HM_CFG).

    Upstream reaches the same tree through ``RAD_CFG_PATH``
    (``$OR_SRC/INSTALL.md:39``), accepted here as an alias.  Accepts either
    the tree root or the ``config/CFG`` schema directory inside it — see
    :func:`is_cfg_tree`; ``mat_reader._find_cfg_root`` decides which one it
    got by inspecting the filesystem.
    """
    derived = _or_src_maybe()
    return _resolve("PYRADIOSS_HM_CFG", [
        # -- rule 1: the environment variable (set AND carrying schemas) --
        _env_candidate("PYRADIOSS_HM_CFG"),
        _env_candidate("RAD_CFG_PATH"),
        # -- rule 2: sibling-of-build --
        _under_or_root("$OR_ROOT/../OpenCourant/hm_cfg_files",
                       "..", _UPSTREAM_DIRNAME, "hm_cfg_files"),
        # -- rule 3: Windows compatibility path (never shadowed) --
        _Candidate(str(_WIN_CFG), _WIN_CFG),
        # -- beyond the contract, strictly after rules 1-3 --
        _under_or_root("$OR_ROOT/OpenCourant/hm_cfg_files (brief's sibling "
                       "tree, not §4.1)",
                       _UPSTREAM_DIRNAME, "hm_cfg_files"),
        _Candidate("$OR_SRC/hm_cfg_files (from a resolved $OR_SRC)",
                   derived / "hm_cfg_files" if derived else None,
                   "" if derived else "$OR_SRC did not resolve"),
        _Candidate("$OR_SRC/hm_cfg_files (upstream checkout beside this repo)",
                   _checkout_hm_cfg()),
    ], is_cfg_tree)


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
    "PYRADIOSS_HM_CFG": "export PYRADIOSS_HM_CFG=<hm_cfg_files> (the tree "
                         "root) or upstream's RAD_CFG_PATH "
                         "($OR_SRC/INSTALL.md:39)",
    "PYRADIOSS_RD_DECKS": "export PYRADIOSS_RD_DECKS=<corpus>, or keep the "
                          "vendored tests/data/rd_decks in the checkout",
}


def _as_tried(tried: Tried) -> List[_Tried]:
    """Accept the documented ``list[Path]``, a bare path, a
    ``(origin, path)`` pair, and the internal ``_Candidate`` / ``_Tried``
    forms."""
    out: List[_Tried] = []
    if tried is None:
        return out
    if isinstance(tried, (_Tried, _Candidate, str, Path)):
        tried = [tried]
    for item in tried:
        if isinstance(item, _Tried):
            out.append(item)
        elif isinstance(item, _Candidate):
            out.append(_Tried(item, item.note))
        elif isinstance(item, tuple) and len(item) == 2 \
                and isinstance(item[0], _Candidate):
            out.append(_Tried(item[0], str(item[1])))
        elif isinstance(item, tuple) and len(item) == 2:
            out.append(_Tried(_Candidate(str(item[0]), Path(item[1])), ""))
        elif isinstance(item, (str, Path)):
            p = Path(item)
            out.append(_Tried(_Candidate(str(p), p), ""))
        else:                              # defensive: keep the text
            out.append(_Tried(_Candidate(str(item), None,
                                         "unusable candidate"), ""))
    return out


def _render(entry: _Tried) -> str:
    cand = entry.candidate
    reason = entry.reason or cand.note
    if cand.path is not None:
        resolved = _norm(cand.path)
        head = (f"[{cand.origin}]" if str(resolved) == cand.origin
                else f"[{cand.origin}] -> {resolved}")
    else:
        head = f"[{cand.origin}]"
    if not reason:
        return head
    lines = reason.splitlines()
    if len(lines) == 1:
        return f"{head} — {lines[0]}"
    body = "\n".join("    " + line for line in lines[1:])
    return f"{head} — {lines[0]}\n{body}"


def missing_resource(name: str, tried: Tried) -> FileNotFoundError:
    """Build and **return** (never raise) the loud failure for ``name``.

    Every attempted location is listed with its symbolic origin from the
    contract and, when it differs, its resolved absolute path, plus the
    reason it was rejected.  Resolvers call
    ``raise missing_resource(...)``; a caller that must not abort logs
    ``str(exc)`` instead.  (Convention: the interface is typed
    ``-> FileNotFoundError``, so the object is returned and the caller
    decides — this function never raises by itself.)
    """
    entries = _as_tried(tried)
    lines = [f"{name} not found: none of the {len(entries)} "
             f"candidate location{'s' if len(entries) != 1 else ''} "
             f"passed the check."]
    if entries:
        lines.append("Tried:")
        lines.extend(f"  {_render(e)}" for e in entries)
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