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
than a missing one, and a *set but wrong* environment variable is **not**
stepped over: it is announced with :func:`warnings.warn` and the resolution
aborts, naming the variable, its value, and every location it did *not* take
(:func:`_unreached`).  Stepping over it was the last silently-degrading path
left here — ``plan/00_ORCHESTRATION.md`` §9.1 item 9 rejects it,
``plan/01_phase0_oracle_and_licensing.md:720`` states the rule ("a *stale
export* -> fail, not skip") — and it had teeth: with
``tools/oracle/oracle_env.sh`` sourced (it exports ``LD_LIBRARY_PATH``) a
stale ``OR_BUILD`` gave a **green** oracle run against a mirror the operator
had not asked for.  An *unset* variable is unchanged: it takes no part in the
search, so a fresh shell with nothing configured still resolves through the
candidate chain.

Candidates beyond the contract, and their exact position
-------------------------------------------------------
The ``hm_cfg_dir`` candidates listed here are not in §4.1.  All of them are
placed **after rule 3**, so no extra can ever shadow a contract candidate (a
reviewer's Important-2 finding — the first version of this module put the
brief's
``$OR_ROOT/OpenCourant/hm_cfg_files`` *before* the Windows candidate and
the rule-3 path was therefore unreachable on a box where both existed):

* ``$OR_ROOT/OpenCourant/hm_cfg_files`` — the brief's sibling tree
  (``task-6-brief.md`` ``test_hm_cfg_defaults_to_the_sibling_tree``);
* ``$OR_SRC/hm_cfg_files`` derived from a resolved ``$OR_SRC``;
* ``<repo>/../OpenCourant/hm_cfg_files`` and ``~/OpenRadioss_or`` — the
  layout measured on the Linux dev box on 2026-10-03, so that a fresh shell
  with *no* exported variable resolves instead of failing.  They are skipped
  entirely when the tree is not there.

``or_src()`` carries the matching fourth candidate ``<repo>/../OpenCourant``
(after rule 3, for the same reason and the same measured layout): the resolver
already knew this checkout for the cfg tree, while ``$OR_SRC`` — the resource
the checkout *is* — did not resolve in a bare shell, which cost two test skips
(``tests/test_p0_oracle_env_script.py:619`` and
``tests/test_p0_oracle_provenance.py:636``).  The directory it names is exactly
the one §4.1's own dev-box column records as ``OR_SRC`` —
``/home/valentin/Projects/OpenRadioss/OpenCourant`` — which is a sibling of the
**repository checkout**, not of ``$OR_ROOT``, so rule 2 cannot reach it here.
It is checked like the cfg-tree sibling of the same name **and**, being a
candidate this module *guessed* rather than was told, like the mirror:
:func:`is_or_source_tree` requires the two files tracked at the root of every
OpenRadioss checkout (``INSTALL.md``, the root ``CMakeLists.txt``), so a
directory that merely carries the name ``OpenCourant`` — an empty one, a
half-made one — is refused instead of bound.  An empty directory that binds as
``$OR_SRC`` is worse than no candidate at all: it converts a loud, actionable
"OR_SRC not found" into a mysterious failure inside whatever reads the tree.
``tests/test_p0_paths.py`` pins that it resolves when it carries the markers,
that rules 1/2/3 each outrank it, that an empty one does not satisfy the
resolver, and that the rule-4 failure still enumerates all four origins in
order.

``$OR_BUILD`` (the writable mirror) is itself absent from §4.1's table, so its
whole candidate list is "beyond the contract"; it carries the same dev-box
default for the same reason.  ``~/OpenRadioss_build`` is a top-level path — the
pre-migration ``$OR_ROOT/source`` location was removed, so the install
prefix's ``source/`` is not where the mirror is on this box, and without the
default the mirror resolved nowhere and every consumer of its ``extlib``
(``th_to_csv``, the h3d writer, ``libhm_reader``) degraded to a skip.  Because
it is the one candidate here that the module *guessed* rather than was told, it
is checked with :func:`is_or_mirror` — the two gates
``tools/oracle/build_oracle.sh:169,175`` enforces — and a directory that
merely shares the name is rejected rather than returned.  The two shape checks
are the same mechanism, for the same reason, on the two guessed candidates:
a configured location (``$OR_SRC``/``$OR_BUILD`` set by hand) is only checked
for existence, because §4.1 rule 1 makes it the maintainer's decision.

So the candidate order for ``hm_cfg_dir`` is: ``$PYRADIOSS_HM_CFG``,
``$RAD_CFG_PATH`` (upstream's own spelling, ``INSTALL.md:39``),
``$OR_ROOT/../OpenCourant/hm_cfg_files``, ``C:\\OpenRadioss\\hm_cfg_files``,
``$OR_ROOT/OpenCourant/hm_cfg_files``, ``$OR_SRC/hm_cfg_files``,
``<repo>/../OpenCourant/hm_cfg_files``.  For ``or_src`` it is:
``$OR_SRC``, ``$OR_ROOT/../OpenCourant``, ``C:\\OpenRadioss``,
``<repo>/../OpenCourant``.  For ``or_build`` it is:
``$OR_BUILD``, ``$OR_ROOT/source``, ``~/OpenRadioss_build``.
``tests/test_p0_paths.py`` pins the ``hm_cfg_dir`` precedence relations by
name and ``tests/test_p0_or_build_layout.py`` pins the mirror's (the count and
the order, the loud failure that names all three, and the mirror predicate).

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
    "is_or_mirror",
    "is_or_source_tree",
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

    ``predicate`` is for a candidate that must prove what it is before it is
    accepted — a *discovered* location, never a configured one.  It defaults to
    ``None`` (the resolver's resource-level ``exists`` check applies), because
    §4.1 rule 1 is deliberately permissive about a variable the maintainer set
    by hand: second-guessing a configured path is this resolver's opportunity to
    be wrong in a way that is much harder to see than refusing a directory it
    merely guessed at.  ``why`` is the human reason a ``predicate`` refuses,
    quoted verbatim in the failure and in the warning; it must say what the
    location *would* have to carry, never "does not exist".
    """

    origin: str
    path: Optional[Path] = None
    note: str = ""
    env_var: str = ""
    predicate: Optional[Callable[[Path], bool]] = None
    why: str = ""


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


def _dev_or_build() -> Path:
    """Dev-box writable mirror, measured 2026-10-03 and recorded in
    ``plan/01_phase0_oracle_and_licensing.md:669,829`` /
    ``docs/STATE.md:54,835``: ``~/OpenRadioss_build``.

    A top-level path in its own right — a sibling of nothing — because the
    pre-migration ``$OR_ROOT/source`` location was *removed* rather than moved
    (the cmake evidence is
    ``$OR_BUILD/build/starter/CMakeCache.txt:CMAKE_HOME_DIRECTORY``, recorded
    in ``tools/validation_data/oracle_provenance.json`` under
    ``mirror_path_note``).  So the install prefix's own ``source/`` is no
    longer where this tree lives, and the seam exists for the same reason
    ``_dev_or_root`` does: to be patched out by the resolver tests.
    """
    return Path(os.path.expanduser("~")) / "OpenRadioss_build"


def _checkout_upstream() -> Path:
    """``<repo>/../OpenCourant`` — the upstream checkout sitting **beside the
    repository** (the measured dev-box layout, and the one
    ``plan/00_ORCHESTRATION.md`` §4.1 records as ``OR_SRC``).

    A *sibling of the checkout*, not of ``$OR_ROOT``: rule 2's
    ``$OR_ROOT/../OpenCourant`` cannot reach it on this box, because
    ``$OR_ROOT`` is ``~/OpenRadioss_or`` while the checkout pair lives under
    ``~/Projects/OpenRadioss``.  ``hm_cfg_dir()`` has resolved through this
    layout since P0.6; ``or_src()`` did not, so ``$OR_SRC`` was the one
    resource of the pair that a bare shell could not find.  A seam for the
    tests, which patch ``_REPO_ROOT``.
    """
    return _REPO_ROOT.parent / _UPSTREAM_DIRNAME


def _checkout_hm_cfg() -> Path:
    """``<repo>/../OpenCourant/hm_cfg_files`` — the cfg tree inside the
    upstream checkout that sits beside this repository."""
    return _checkout_upstream() / "hm_cfg_files"


def is_or_mirror(path) -> bool:
    """True when ``path`` really is a writable OpenRadioss **mirror**.

    A mirror is the ``git archive``d tree plus a fetched ``extlib``
    (``tools/oracle/mirror_and_fetch.sh``), and ``build_oracle.sh`` refuses to
    go on without both of them: ``:169`` requires ``CMakeLists.txt`` ("no
    CMakeLists.txt" is not a mirror) and ``:175`` requires
    ``extlib/hm_reader``.  Those two gates are the predicate, unchanged,
    because they are what the build itself enforces — a directory that fails
    them resolves to a mirror that cannot be built, which is a worse outcome
    than not resolving at all.

    Applied only to **discovered** locations.  ``$OR_BUILD`` set by hand stays
    a plain "does it exist" check (§4.1 rule 1): a configured path is the
    maintainer's decision.
    """
    p = Path(str(path))
    if not (p / "CMakeLists.txt").is_file():
        return False
    return p.joinpath("extlib", "hm_reader").is_dir()


#: The verbatim reason each shape check reports, quoted in the failure and in
#: the warning — so the message says what the location would have had to carry
#: rather than that it "does not exist" (it does; it is just not the resource).
MIRROR_WHY = ("an OpenRadioss mirror (needs CMakeLists.txt and "
              "extlib/hm_reader; the two gates build_oracle.sh:169,175 "
              "enforces)")
SOURCE_TREE_WHY = ("an OpenRadioss source tree (needs INSTALL.md and the "
                   "root CMakeLists.txt; the two files tracked at the root of "
                   "every checkout, git ls-files INSTALL.md CMakeLists.txt)")


def is_or_source_tree(path) -> bool:
    """True when ``path`` really is the upstream OpenRadioss **source tree**.

    Two markers, both tracked at the root of every OpenRadioss checkout
    (``git ls-files INSTALL.md CMakeLists.txt`` in ``$OR_SRC``):

    * ``INSTALL.md`` — the file this module already quotes for upstream's own
      environment block (``$OR_SRC/INSTALL.md:34-42``, the Windows spelling at
      ``:44-50``, and the installed binary names at ``:110``).  A tree without
      it is not the source tree those citations describe.
    * the root ``CMakeLists.txt`` — ``project (OpenRadioss)`` plus the
      ``add_subdirectory`` gate for ``starter``/``engine``
      (``$OR_SRC/CMakeLists.txt:5-6,19-31``), i.e. the tree is buildable.

    Applied only to a *discovered* location, the same rule
    :func:`is_or_mirror` follows: ``$OR_SRC`` set by hand stays a plain "does it
    exist" check (§4.1 rule 1), a configured path is the maintainer's decision,
    while a directory **this module guessed** (``<repo>/../OpenCourant``) must
    prove what it is.  An empty or same-named directory is worse than no
    candidate at all: it turns a loud, actionable "OR_SRC not found" into a
    mysterious failure inside whatever reads the tree.
    """
    p = Path(str(path))
    return (p / "INSTALL.md").is_file() and (p / "CMakeLists.txt").is_file()


def _norm(path: Path) -> Path:
    """Collapse ``..`` lexically so a returned path compares equal to the
    location a maintainer would type."""
    return Path(os.path.normpath(str(path)))


def _resolve(name: str, candidates: Sequence[_Candidate],
             exists: Callable[[Path], bool],
             kind: str = "directory",
             needs: str = "") -> Path:
    """Return the first candidate the predicate accepts; else raise loudly.

    A candidate may carry its own ``predicate`` (a *discovered* location that
    must prove what it is -- see :func:`is_or_mirror` and
    :func:`is_or_source_tree`); it replaces the resource-level ``exists`` for
    that candidate alone.  ``needs`` names what the *resource-level* check
    demands, so a rejection says why rather than claiming a directory that is
    really there "does not exist".

    **A variable that is set but resolves to nothing is terminal** (§4.1 rule 1
    read together with rule 4, and ``plan/01_phase0_oracle_and_licensing.md:720``
    "a *stale export* -> fail, not skip").  It is announced with
    :func:`warnings.warn` -- a caller that swallows the exception still sees the
    cause -- and the resolution then raises instead of stepping over it: a stale
    ``PYRADIOSS_RD_DECKS`` used to leave a validation run covering the small
    vendored corpus while the log claimed the full extract, and a stale
    ``OR_BUILD`` let a parity run validate whatever dev-box mirror happened to
    be lying next to the checkout.  The abort still names **every** candidate,
    the unreached ones marked "not tried", so rule 4 loses nothing.  An
    *unset* variable takes no part in the search and the chain falls through
    normally: a fresh shell with nothing configured resolves, as §4.1 intends.
    """
    cached = _CACHE.get(name)
    if cached is not None:
        return cached
    tried: List[_Tried] = []
    for pos, cand in enumerate(candidates):
        if cand.path is None:
            tried.append(_Tried(cand, cand.note or "unresolved"))
            continue
        accepted = cand.predicate or exists
        if not accepted(cand.path):
            tried.append(_Tried(cand, _why_not(cand, kind, needs)))
            if cand.env_var:
                warnings.warn(
                    f"{cand.env_var} is set to {cand.path} but that is not "
                    f"{_what_it_must_be(cand, kind, needs)} — refusing to "
                    f"resolve {name} from another location; unset "
                    f"{cand.env_var} or point it at the right one",
                    RuntimeWarning, stacklevel=3)
                raise missing_resource(
                    name, tried + _unreached(candidates[pos + 1:], cand.env_var))
            continue
        resolved = _norm(cand.path)
        _CACHE[name] = resolved
        return resolved
    raise missing_resource(name, tried)


def _what_it_must_be(cand: _Candidate, kind: str, needs: str) -> str:
    """The phrase a rejection reason and its warning share: what the location
    has to be for the resource to be found there — the candidate's own ``why``
    when it carries a shape check, else the resource's ``needs``, else the
    plain ``kind``."""
    return cand.why or needs or f"a {kind}"


def _why_not(cand: _Candidate, kind: str, needs: str) -> str:
    """Why this candidate was rejected.

    No path prefix — :func:`_render` already prints the resolved location — and
    never "does not exist" about a directory that does exist: the misleading
    diagnostic this exists to prevent is a wrong reason, not a missing one.
    """
    return f"not {_what_it_must_be(cand, kind, needs)}"


def _unreached(rest: Sequence[_Candidate], env_var: str) -> List[_Tried]:
    """The candidates an abort never looked at, marked as such.

    §4.1 rule 4 is "name every attempted location"; refusing to continue leaves
    locations *un*attempted, and saying which is what keeps the diagnostic
    complete instead of leaving the operator to guess what was left out.
    """
    stop = f"not tried: {env_var} was exported and did not resolve"
    return [_Tried(c, c.note or stop) if c.path is None
            else _Tried(c, stop) for c in rest]


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
    """The read-only upstream OpenRadioss source tree ($OR_SRC).

    Candidates, in order:

    1. ``$OR_SRC`` if set and the path exists (§4.1 rule 1);
    2. ``$OR_ROOT/../OpenCourant`` — the sibling-of-build spelling rule 2 gives;
    3. ``C:\\OpenRadioss`` — the Windows compatibility path (§4.1 rule 3);
    4. ``<repo>/../OpenCourant`` — the upstream checkout sitting beside *this
       repository*, which is where §4.1's own dev-box column puts ``OR_SRC``
       (``/home/valentin/Projects/OpenRadioss/OpenCourant``, a sibling of the
       checkout rather than of ``$OR_ROOT``).  Beyond the contract, and last,
       so it can never shadow a contract candidate — the same placement
       ``hm_cfg_dir()`` has had for this layout since P0.6.  It is the one
       location here **this module guessed**, so it must prove what it is
       (:func:`is_or_source_tree`; the same mechanism, and the same reason, as
       :func:`is_or_mirror` on the mirror): a directory that merely shares the
       name is refused, because binding one turns a loud "OR_SRC not found"
       into a mysterious failure inside whatever reads the tree.  Rule 4 still
       fires, naming all four, when none of them does.
    """
    return _resolve("OR_SRC", [
        _env_candidate("OR_SRC"),
        _under_or_root("$OR_ROOT/../OpenCourant", "..", _UPSTREAM_DIRNAME),
        _Candidate(str(_WIN_ROOT), _WIN_ROOT),
        _Candidate("<repo>/../OpenCourant (upstream checkout beside this repo)",
                   _checkout_upstream(), predicate=is_or_source_tree,
                   why=SOURCE_TREE_WHY),
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

    Candidates, in order:

    1. ``$OR_BUILD`` if set and the path exists (§4.1 rule 1);
    2. ``$OR_ROOT/source`` — the sibling-of-build spelling rule 2 gives, and
       still the right answer for a layout that puts the mirror inside the
       install prefix;
    3. ``~/OpenRadioss_build`` — the dev-box location measured 2026-10-03 and
       recorded in ``plan/01_phase0_oracle_and_licensing.md:669,829``.  It is a
       top-level path in its own right: the pre-migration
       ``$OR_ROOT/source`` was removed, so on this box candidate 2 cannot fire
       and the mirror was invisible to every resolver.  It is **verified, not
       assumed** (:func:`is_or_mirror`), because it is the one candidate here
       that this module guessed rather than was told.

    Deliberately never falls back to ``$OR_SRC``: the build writes into
    this tree (``load_extlib.py`` extracts there, and the starter/engine
    POST_BUILD rules create ``../exec``), and ``$OR_SRC`` is READ-ONLY by
    rule (``plan/00_ORCHESTRATION.md`` §1.2).
    """
    dev = _dev_or_build()
    return _resolve("OR_BUILD", [
        _env_candidate("OR_BUILD"),
        _under_or_root("$OR_ROOT/source (writable mirror)", "source"),
        _Candidate("~/OpenRadioss_build (dev-box default)", dev,
                   predicate=is_or_mirror, why=MIRROR_WHY),
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
    ], is_cfg_tree, needs="CFG tree (config/CFG/radioss<version>)")


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