#!/usr/bin/env python
"""
Differential validation of the pyradioss port against the Fortran
OpenRadioss binaries (the RESEARCH_AUDIT's #1 recommendation).

Two modes
=========

parity (default)
    For each example under ``examples/`` (or ``--only name1,name2``):

    1. run the real Fortran Starter + Engine on the example's deck pair
       in a scratch directory, convert the binary T01 time-history with
       upstream's ``th_to_csv`` converter (resolved by :func:`oracle_paths`;
       an optional external tool — see the note there);
    2. run the pyradioss Starter + Engine on the *original* deck pair in
       a sibling scratch directory (the port writes its T01 as CSV
       directly);
    3. compare every overlapping time-history channel (global internal /
       kinetic / hourglass / contact energies, external work, mass,
       momentum components, and /TH/PART energies where unambiguous) and
       report the relative RMS deviation and the final-time deviation —
       the actual numbers are always printed, the tolerance only sets
       the MATCH/DEVIATION label.

    Classification per example:

    - ``MATCH``          all compared channels within ``--tol`` (default 5 %)
    - ``DEVIATION``      Fortran and pyradioss both ran; some channel above tol
    - ``PORT-ONLY``      the Fortran chain cannot run this deck. Subtags:
                         ``(implicit)`` engine deck uses the port's /IMPL
                         cards; ``(dialect)`` the real Starter rejects the
                         port's deck dialect (see the DIALECT NOTE below);
                         ``(starter-reject)`` translated deck still refused.
    - ``FORTRAN-FAIL``   Fortran starter passed but engine/converter failed
    - ``PYRADIOSS-FAIL`` the port itself failed on its own example
    - ``SKIPPED-SLOW``   pyradioss side exceeded ``--timeout`` (default 240 s)
                         or the global ``--budget`` (default 2700 s)

    DIALECT NOTE (M36 update): the bundled example decks are now
    generated in the **real fixed 2022 format** directly, by
    ``pyradioss/input/deck_writer.py`` — the M35 per-keyword translator
    below was PROMOTED into that module and extended to every supported
    keyword family; the copy kept here is only a **fallback for
    old-dialect decks** (user decks written in the port's historical
    free-format style).  The harness auto-detects the format
    (:func:`deck_is_real_format` looks for the /BEGIN input-version
    card): real-format decks skip translation entirely and only receive
    :func:`real_deck_fixups` — the deck_writer's *documented residue
    fields* mapped to their real meaning for the Fortran run
    (/RWALL blank search distance -> 1e30; /INTER/TYPE7|11 port gap_max
    in the real Tstart column -> moved to the real GAPMAX field; the
    port-only /TH/SECT block stripped).  See the deck_writer module
    docstring for why those residues exist.

    For OLD-dialect decks the historical behaviour is unchanged: feeding
    them to the Fortran Starter unmodified fails at /BEGIN
    ("INPUT FORMAT 0 NOT SUPPORTED"), so ``--shim translate`` (default)
    re-emits a fixed-format copy per keyword; decks using keywords
    without a verified translator are classified PORT-ONLY(dialect)
    (``--shim begin`` inserts only the /BEGIN version card so the
    per-deck real error is visible; ``--shim none`` runs the deck raw).

coverage
    Run ONLY the pyradioss Starter on the given native ``.rad`` deck(s)
    (e.g. the Ryan Lee k2rad conversions) and tabulate every keyword the
    port skipped, warned about, or errored on — a keyword-coverage
    census of the port against real-world decks.

Examples
========
    python tools/validate_vs_fortran.py parity
    python tools/validate_vs_fortran.py parity --only tensile_bar,box_beam_impact
    python tools/validate_vs_fortran.py coverage <path>/W12_k2rad/W12_0000.rad

    (the Ryan Lee k2rad corpus of the historical sweeps lived on the
    original Windows reference box, under its ``openradioss_run`` share;
    on any other box pass whatever deck path you have — the harness never
    assumes a corpus location.)

Results land in ``<workdir>/parity_results.json`` (parity) /
``<workdir>/coverage_<deck>.json`` (coverage) plus a console table.

Where the oracle comes from (this module is platform-neutral)
============================================================
Nothing here names a machine.  The oracle location is resolved **per call**
through :func:`pyradioss.paths` — env var, then sibling-of-build, then the
Windows compatibility rule inside ``paths`` itself, then a loud failure
listing every candidate — and never at import time, so importing this module
touches no filesystem and works with nothing configured
(``plan/00_ORCHESTRATION.md`` §4.1).  :func:`oracle_paths` reports each of the
five resources the run needs; when the starter or the engine cannot be
resolved, ``parity`` stops with exit code 2 rather than reporting an empty
comparison.

The runtime environment the harness builds is upstream's own
(``$OR_SRC/INSTALL.md:34-42`` — ``OPENRADIOSS_PATH``, ``RAD_CFG_PATH``,
``OMP_STACKSIZE``, ``LD_LIBRARY_PATH`` for the native-``.k`` reader), with one
deliberate omission and one deliberate deletion, both recorded in
``tools/validation_data/oracle_provenance.json``:

* the h3d writer is kept out of reach **as far as an environment can do
  it**.  ``RAD_H3D_PATH`` is *deleted* rather than set, and so are the other
  places ``h3dlib_load_`` looks: ``$ALTAIR_HOME/hwsolvers/common/bin/$ARCH``
  (``$OR_SRC/common_source/output/h3d/h3d_build_cpp/h3d_dl.c:647-658``; the
  Windows branch at ``:339-353`` reads the same two variables) and the loader
  search path (``:660-666`` POSIX, and on Windows ``PATH`` at ``:356``); the
  scratch directory is checked too, because it is the solvers' cwd
  (``:634-644``).  With the reachable writer reachable by any of them, a run
  reaches NORMAL TERMINATION and writes silently wrong H3D files; with them
  closed ``h3d_dl.c:920-922`` sets ``*IERROR = 1`` and
  ``engine/source/output/h3d/h3d_results/genh3d.F:728-732`` aborts with
  MSGID 274.  Two routes are **not** closeable and are reported rather than
  claimed closed: a system-wide install (``ld.so`` cache / default
  directories) and the binaries' own ``DT_RPATH``, which glibc searches
  *before* ``LD_LIBRARY_PATH``.  The harness reads the RPATH out of the ELF
  and refuses the run when the writer is visible there.  T01, A-files,
  RESTART and the listing are unaffected either way, and only those are
  admissible parity evidence (``oracle_provenance.json``
  ``admissible_parity_evidence``).  See :func:`fortran_env` for the route
  table and :func:`run_fortran` for the two refusals.
* the Intel-MPI / oneAPI entries of the historical Windows launch are gone:
  the Linux oracle is the OpenMP build, and an inherited ``KMP_*`` from the
  caller's shell still reaches the solver because the environment is copied.

Invocation (``$OR_SRC/INSTALL.md:110-111``, and what was measured on the box):
the starter is given ``-np 1 -nt 1``; the engine is given ``-nt 1`` and
**never** ``-np`` — its argument parser does not accept it, prints its usage
and dies with a SIGSEGV, which reads like a broken oracle but is not.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pyradioss import paths  # noqa: E402

#: Where a run's scratch directories go.  A **function**, not a constant:
#: ``tempfile.gettempdir()`` creates and deletes a probe file when the
#: platform's temp location is not already known, and doing that at import
#: time would make "importing this harness touches no filesystem" a claim
#: with an exception in it (the guard is
#: tests/test_p0_harness_portable.py::test_harness_import_touches_no_filesystem).
def default_workdir() -> str:
    """``$VALRUNS_DIR``, else ``<platform temp>/valruns``.

    ``TEMP`` is the Windows spelling and ``TMPDIR`` the POSIX one; a POSIX
    box with neither falls back to :func:`tempfile.gettempdir`, which is why
    this cannot be a module constant.  Resolved per call so the value always
    reflects the environment at run time.
    """
    override = os.environ.get("VALRUNS_DIR")
    if override:
        return override
    base = os.environ.get("TEMP") or os.environ.get("TMPDIR")
    return os.path.join(base or tempfile.gettempdir(), "valruns")


# ----------------------------------------------------------------------------
# Oracle resources (P0.9) — resolution is lazy, per call, and machine-neutral
# ----------------------------------------------------------------------------
#
# Upstream / environment origin of every name below (repo-relative under the
# read-only ``OpenCourant`` tree, i.e. $OR_SRC):
#
#   * INSTALL.md:110          starter_linux64_gf        ($OR_ROOT/bin here;
#                             the modern upstream install prefix is `exec`, see
#                             RELEASES.md:22-41 — paths.or_starter() lists both)
#   * INSTALL.md:111          engine_linux64_gf
#   * INSTALL.md:39,42        RAD_CFG_PATH=$OPENRADIOSS_PATH/hm_cfg_files and
#                             LD_LIBRARY_PATH=$OPENRADIOSS_PATH/extlib/
#                             hm_reader/linux64/  -> hm_reader_lib
#   * INSTALL.md:40           RAD_H3D_PATH=$OPENRADIOSS_PATH/extlib/h3d/lib/
#                             linux64            -> h3d_lib (resolved, NOT used)
#   * RELEASES.md:29,52       exec/th_to_csv_linux64_gf  ("Time History file to
#                             Paraview CSV format converter")
#   * RELEASES.md:72,106      exec/th_to_csv_win64.exe
#   * tools/th_to_csv/README.md:1-7  the converter's SOURCE lives in the
#                             separate OpenRadioss/Tools repository; the pinned
#                             tree carries only this README, and neither
#                             starter/CMakeLists.txt nor engine/CMakeLists.txt
#                             builds it (Apptainer/openradioss.def:31-32 builds
#                             it from its own checkout).  It is therefore an
#                             OPTIONAL resource: resolved when present, and
#                             reported with its candidate list when not.
#
# Nothing below is a module-scope path.  Importing this module must resolve
# nothing (a test booby-traps pathlib to prove it), because a module-scope
# constant is frozen against whichever box imported the module first.

#: The resources a parity run needs, in the order they are reported.
ORACLE_KEYS = ("starter", "engine", "th_to_csv", "h3d_lib", "hm_reader_lib")

#: ``th_to_csv`` file names, per platform (RELEASES.md:29,52,72,106).
_TH_TO_CSV_NAMES = ("th_to_csv_linux64_gf", "th_to_csv_win64.exe")

#: The h3d writer's file names — ``h3d_dl.c:63`` (POSIX) / ``:58`` (Windows)
#: initialise ``h3dlib`` to exactly these, and ``h3dlib_load_``
#: (``h3d_dl.c:616-923``) dlopens them from four different places.
_H3D_WRITER_NAMES = ("libh3dwriter.so", "h3dwriter.dll")

#: The h3d writer and the native-.k reader, per platform, exactly as
#: ``$OR_SRC/INSTALL.md:40,42`` spells them under ``$OPENRADIOSS_PATH``.
_H3D_LIB_NAMES = ("extlib/h3d/lib/linux64/libh3dwriter.so",
                  "extlib/h3d/lib/win64/h3dwriter.dll")
_HM_READER_LIB_NAMES = (
    "extlib/hm_reader/linux64/libhm_reader_linux64.so",
    "extlib/hm_reader/win64/hm_reader_win64.dll")


def _first_file(tried: List[Tuple[str, Path]]) -> Optional[str]:
    """The first candidate that exists, as a string; ``None`` if none does."""
    for _origin, candidate in tried:
        if os.path.isfile(candidate):
            return str(candidate)
    return None


def _build_candidates(relative_paths) -> List[Tuple[str, Path]]:
    """``$OR_BUILD/<rel>`` candidates, or an explanation when it is unresolved.

    ``pyradioss.paths`` owns the install prefixes; the extlib payloads below
    the writable mirror are the remaining piece, and they are named exactly as
    ``$OR_SRC/INSTALL.md:39-42`` names them.
    """
    try:
        root = paths.or_build()
    except FileNotFoundError as exc:
        first_line = str(exc).splitlines()[0]
        return [(f"$OR_BUILD/{rel} — OR_ROOT unresolved ({first_line})",
                 f"$OR_BUILD/{rel} — OR_ROOT unresolved ({first_line})")
                for rel in relative_paths]
    return [(f"$OR_BUILD/{rel}", root.joinpath(*rel.split("/")))
            for rel in relative_paths]


def _th_to_csv_candidates() -> List[Tuple[str, Path]]:
    """``$OR_TH_TO_CSV`` first, then the two documented install spellings.

    The override follows the ``OR_*`` naming the rest of the contract uses;
    the two locations come from ``RELEASES.md:29,52,72,106`` (the modern
    install prefix and the pre-cmake ``exec/`` one that ``paths.or_starter``
    also lists).
    """
    var = "OR_TH_TO_CSV"
    tried: List[Tuple[str, Path]] = []
    value = os.environ.get(var)
    if value:
        tried.append((f"env {var}", Path(value)))
    else:
        # an unconstructed candidate: origin == path, which is how
        # paths.missing_resource renders "this one could not even be built"
        label = f"env {var} (not set)"
        tried.append((label, label))
    try:
        root = paths.or_root()
    except FileNotFoundError as exc:
        label = (f"$OR_ROOT/{{bin,exec}}/{_TH_TO_CSV_NAMES[0]} — OR_ROOT "
                 f"unresolved ({str(exc).splitlines()[0]})")
        tried.append((label, label))
        return tried
    for sub in ("bin", "exec"):
        for filename in _TH_TO_CSV_NAMES:
            tried.append((f"$OR_ROOT/{sub}/{filename}", root / sub / filename))
    return tried


def _resolve_oracle_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    """``(path, None)`` when resolved, ``(None, diagnostic)`` when not.

    The diagnostic is ``str(paths.missing_resource(...))`` — the one that knows
    the contract's candidate order, which a bespoke message here would only
    drift away from.  A *set but missing* override is part of it, so a stale
    ``OR_TH_TO_CSV`` cannot masquerade as "not built here".
    """
    if key in ("starter", "engine"):
        try:
            return str(getattr(paths, "or_" + key)()), None
        except FileNotFoundError as exc:
            return None, str(exc)
    if key == "th_to_csv":
        tried = _th_to_csv_candidates()
    elif key == "h3d_lib":
        tried = _build_candidates(_H3D_LIB_NAMES)
    else:
        tried = _build_candidates(_HM_READER_LIB_NAMES)
    return _first_file(tried), str(paths.missing_resource(key, tried))


def oracle_paths(strict: bool = False) -> Dict[str, Optional[str]]:
    """Resolve every resource a parity run needs; ``key -> path or None``.

    Keys: ``starter``, ``engine``, ``th_to_csv``, ``h3d_lib``,
    ``hm_reader_lib`` (:data:`ORACLE_KEYS`).  ``starter``/``engine`` come from
    :func:`pyradioss.paths.or_starter` / ``or_engine`` — this harness is a
    consumer of the one resolver, never a rival with a second ordering.

    **Honest degradation.**  An unresolved resource is reported as ``None``
    and announced with a :class:`RuntimeWarning` carrying every location that
    was tried; it is never silently turned into an empty string, and never
    into an empty comparison — :func:`parity` stops instead (exit code 2).
    The caller decides what an unresolved resource means: ``strict=True``
    raises the first one (``ORACLE_KEYS`` order, so the starter's own
    diagnostic first).  ``th_to_csv`` and ``h3d_lib`` are optional by
    construction — see the block above — so their absence is a fact to
    report, not a reason to refuse to run.

    **Caching is ``paths``' business, not this function's.**  Nothing is
    memoised here, but :func:`pyradioss.paths.or_starter` and friends *are*:
    their answers live in ``paths._CACHE`` until :func:`pyradioss.paths.reload`
    clears it.  So moving ``OR_ROOT`` mid-process still hands back the old
    starter from here — the answer is per *call* but the resolver underneath
    is not per *call* — and a test that changes the environment must call
    ``paths.reload()`` first.  That asymmetry is stated rather than papered
    over with a second cache that could disagree with the first.
    """
    resolved: Dict[str, Optional[str]] = {}
    missing: Dict[str, str] = {}
    for key in ORACLE_KEYS:
        value, why = _resolve_oracle_key(key)
        resolved[key] = value
        if value is None:
            missing[key] = (f"{key} is not available.\n{why}\n"
                            f"Set the variable named above, or fix the install "
                            f"layout (plan/00_ORCHESTRATION.md §4.1); the "
                            f"harness will not substitute a default.")
    for key in ORACLE_KEYS:
        if key in missing:
            warnings.warn(missing[key], RuntimeWarning, stacklevel=2)
    if strict and missing:
        first = next(k for k in ORACLE_KEYS if k in missing)
        raise FileNotFoundError(missing[first])
    return resolved


def oracle_report(oracle: Optional[Dict[str, Optional[str]]] = None) -> str:
    """The loud text for every unresolved resource; ``""`` when all resolve.

    :func:`oracle_paths` announces the same failures as warnings; a driver
    that is about to *refuse to run* should print the reason itself rather
    than point at a warning a caller may have filtered away.
    """
    if oracle is None:
        oracle = oracle_paths()
    blocks = []
    for key in ORACLE_KEYS:
        if oracle.get(key) is not None:
            continue
        _value, why = _resolve_oracle_key(key)
        blocks.append(f"{key} is not available.\n{why}")
    return "\n\n".join(blocks)


# ----------------------------------------------------------------------------
# Solver invocation
# ----------------------------------------------------------------------------

def starter_argv(starter: str, deck: str, np: int = 1,
                 nt: int = 1) -> List[str]:
    """``$OR_SRC/INSTALL.md:110`` — the starter accepts ``-np``.

    Both counts are passed: ``-np`` sizes the (SMP) process pool and ``-nt``
    the OpenMP threads, and the historical Windows launch used both.
    """
    return [str(starter), "-i", os.path.basename(deck),
            "-np", str(np), "-nt", str(nt)]


def engine_argv(engine: str, deck: str, nt: int = 1) -> List[str]:
    """The engine takes ``-nt`` and **must not** be given ``-np``.

    Measured on the oracle box and recorded in
    ``tools/validation_data/oracle_provenance.json`` ``invocation``: the
    engine's argument parser does not accept ``-np``; it prints its usage and
    then dies with a SIGSEGV, which in a batch log is indistinguishable from
    a broken oracle.  ``INSTALL.md:111`` shows the bare invocation for the same
    reason.  Upstream's behaviour is deliberately not "fixed" here —
    ``execargcheck.F`` is upstream source.
    """
    return [str(engine), "-i", os.path.basename(deck), "-nt", str(nt)]


def th_to_csv_argv(converter: str, t01: str) -> List[str]:
    """``th_to_csv <T01>`` — one positional argument, no flags."""
    return [str(converter), os.path.basename(t01)]


def fortran_env(oracle: Optional[Dict[str, Optional[str]]] = None,
                base: Optional[Dict[str, str]] = None,
                platform: Optional[str] = None) -> Dict[str, str]:
    """The oracle's runtime environment: ``$OR_SRC/INSTALL.md:34-42``.

    ``base`` (default: this process's environment) is copied first, so a
    caller-supplied ``LD_LIBRARY_PATH``, ``KMP_*`` or locale survives — this
    function only *adds* what upstream requires and *removes* what must never
    be inherited.

    * ``OPENRADIOSS_PATH`` — upstream's name for the prefix; the solvers and
      the cfg lookup key off it (``INSTALL.md:38``).
    * ``RAD_CFG_PATH`` — the ``hm_cfg_files`` tree (``INSTALL.md:39``).  The
      mirror's own copy wins when it is a real cfg tree, because that is the
      environment the oracle was proved in
      (``tests/test_p0_oracle_build.py::test_oracle_*``); ``paths.hm_cfg_dir``
      is the fallback, so a box without the mirror still resolves.
    * ``LD_LIBRARY_PATH`` (POSIX) or ``PATH`` (Windows) — the native ``.k``
      reader and its APR dependency.  Not cosmetic: without it the binaries
      die on the first message call with an unresolved ``libhm_reader``
      (``tools/oracle/oracle_env.sh``).
    * ``OMP_STACKSIZE`` / ``OMP_NUM_THREADS`` — upstream's 400 m headroom
      (``INSTALL.md:41``) and the single-thread cap that keeps a validation
      run's wall clock meaningful.

    **The h3d writer: what the harness closes, and what it cannot.**  Upstream
    looks for the writer in more than one place, and the two branches differ.
    ``$OR_SRC/common_source/output/h3d/h3d_build_cpp/h3d_dl.c`` has two
    definitions of ``h3dlib_load_`` — ``#ifdef _WIN32`` at ``:312`` and
    ``#elif 1`` at ``:615`` — each with four trials:

    =========  ======================================================  ==========
    lines      route                                                 closed by
    =========  ======================================================  ==========
    ``:623-632``  ``$RAD_H3D_PATH/<h3dlib>`` (Windows: ``:320-328``)    popped
    ``:634-644``  ``getcwd()/<h3dlib>`` (Windows: ``:329-337``)        run_fortran
    ``:647-658``  ``$ALTAIR_HOME/hwsolvers/common/bin/$ARCH/<h3dlib>``  popped
                  (Windows: ``:338-353``, same two variables)
    ``:660-666``  bare ``dlopen(<h3dlib>)`` from the loader path;       scrubbed
                  Windows reads ``PATH`` explicitly at ``:356``
                  (``SetDllDirectory`` ``:357``, ``LoadLibrary`` ``:358``,
                  trial ``:354-359``)
    =========  ======================================================  ==========

    (``h3dlib`` is ``libh3dwriter.so`` on POSIX, ``h3dwriter.dll`` on
    Windows — ``h3d_dl.c:63`` / ``:58``.)  Note that the Windows fourth trial
    is *commented* ``$LD_LIBRARY_PATH settings`` at ``:355`` while the code
    below it reads ``PATH``: the label is upstream's, the behaviour is not,
    which is why scrubbing ``PATH`` is not optional on that platform.

    **Policy on a search path: only the hazardous elements are dropped.**
    ``PATH`` also carries every ordinary tool the run may need, so stripping
    it wholesale would break the run in a way that looks like a broken
    oracle; the rest of the path is kept untouched and the warning names
    exactly what went.

    **Two routes remain open, and no environment variable can close either.**
    (1) A ``libh3dwriter.so`` installed **system-wide** — the ``ld.so`` cache
    or a default directory — is found by ``:660-666`` regardless.  (2) The
    binaries' own ``DT_RPATH``/``DT_RUNPATH``: glibc searches ``DT_RPATH``
    **before** ``LD_LIBRARY_PATH``, so a writer reachable through one of those
    entries is dlopen'd by ``:660-666`` whatever this harness exports.
    Measured on the oracle built for this box (2026-10-03, ``readelf -d`` on
    both binaries, cross-read by :func:`elf_search_paths`): **neither binary
    carries a ``DT_RPATH`` or a ``DT_RUNPATH``** — the conda prefix that
    ``DT_RPATH`` used to name is gone with the pre-migration conda-forge
    toolchain, and these link the system one (ldd resolves ``libgfortran.so.5``
    from ``/lib/x86_64-linux-gnu``).  So route (2) is closed by the BUILD, not
    by the environment, and the mirror image of that fact is why
    ``LD_LIBRARY_PATH`` is not optional here either: with no RPATH and nothing
    in the ``ld.so`` cache (``ldconfig -p`` lists no ``libhm_reader``), the
    starter resolves ``libhm_reader_linux64.so`` from ``LD_LIBRARY_PATH`` and
    from nowhere else (``tools/oracle/oracle_env.sh``, "LD_LIBRARY_PATH IS
    required, not cosmetic").  The route is therefore **probed, never assumed**:
    :func:`binary_rpath_hazards` reads it out of the ELF on every call and
    reports any entry holding the writer; :func:`run_fortran` then refuses the
    run, which is the only honest outcome for a hazard that cannot be scrubbed.

    With every closeable route closed the writer stays unreachable,
    ``h3dhandle`` remains NULL, ``*IERROR = 1`` (``:920-922``), and
    ``$OR_SRC/engine/source/output/h3d/h3d_results/genh3d.F:728-732`` turns
    that into MSGID 274 + ``ARRET(2)`` — h3d refused loudly, while T01,
    A-files, RESTART and the listing stay admissible
    (``oracle_provenance.json`` ``admissible_parity_evidence``).

    ``platform`` selects which loader semantics to assume (``"nt"`` or
    ``"posix"``), defaulting to the running one.  It exists because the
    Windows route cannot be exercised from a POSIX box otherwise — and
    ``os.name`` is not monkeypatchable in a test, because ``pathlib`` builds
    its class from it at import time.
    """
    if oracle is None:
        oracle = oracle_paths()
    env = dict(os.environ if base is None else base)
    # h3d_dl.c:623-632 (trial 1), :647-658 (trial 3) — see the docstring.
    env.pop("RAD_H3D_PATH", None)
    env.pop("ALTAIR_HOME", None)
    env.pop("ARCH", None)
    # h3d_dl.c:660-666 (POSIX) / :355-359 (Windows, which reads PATH) —
    # drop the search-path entries that hold the writer, saying which,
    # instead of quietly running with a live route.
    for var in h3d_search_path_vars(platform):
        _scrub_h3d_search_path(env, var)
    for hazard in binary_rpath_hazards(oracle):
        warnings.warn(hazard, RuntimeWarning, stacklevel=2)
    build = None
    try:
        build = paths.or_build()
    except FileNotFoundError:
        pass
    if build is not None:
        env["OPENRADIOSS_PATH"] = str(build)
        cfg = build / "hm_cfg_files"
        if paths.is_cfg_tree(cfg):
            env["RAD_CFG_PATH"] = str(cfg)
    if "RAD_CFG_PATH" not in env:
        try:
            env["RAD_CFG_PATH"] = str(paths.hm_cfg_dir())
        except FileNotFoundError as exc:
            warnings.warn(f"no hm_cfg_files tree for RAD_CFG_PATH: {exc}",
                          RuntimeWarning, stacklevel=2)
    env["OMP_STACKSIZE"] = "400m"          # INSTALL.md:41
    env["OMP_NUM_THREADS"] = "1"
    reader = oracle.get("hm_reader_lib")
    if reader:
        reader_dir = str(Path(reader).parent)
        if _dirs_holding_h3d_writer([reader_dir]):
            warnings.warn(
                f"{reader_dir} holds the ABI-incompatible h3d writer, so it is "
                f"NOT added to the loader path (h3d_dl.c:660-666 would dlopen "
                f"it from there); the oracle run will fail loudly instead of "
                f"writing wrong H3D files",
                RuntimeWarning, stacklevel=2)
        elif (os.name if platform is None else platform) == "nt":
            env["PATH"] = os.pathsep.join(
                [reader_dir, env.get("PATH", "")]).strip(os.pathsep)
        else:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [reader_dir, env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    return env


def h3d_search_path_vars(platform: Optional[str] = None) -> Tuple[str, ...]:
    """The environment variables the platform's loader search is taken from.

    Upstream's fourth h3d trial is a plain ``dlopen(h3dlib)`` on POSIX
    (``h3d_dl.c:660-666``), which the loader answers from
    ``LD_LIBRARY_PATH``; on Windows the same trial reads ``PATH`` explicitly
    (``h3d_dl.c:356`` ``GetEnvironmentVariable("PATH", …)``, ``:357``
    ``SetDllDirectory``, ``:358`` ``LoadLibrary``).  So the variable to scrub
    is platform-dependent, and that is a parameter rather than a ``skip``:
    the Windows route has to be testable from a POSIX box.
    """
    name = os.name if platform is None else platform
    if name == "nt":
        return ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
    return ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")


#: ``DT_RPATH`` / ``DT_RUNPATH`` dynamic tags (ELF gABI; ``<elf.h>``).
_DT_NULL, _DT_RPATH, _DT_RUNPATH = 0, 15, 29


def elf_search_paths(path) -> List[str]:
    """The ``DT_RPATH`` / ``DT_RUNPATH`` directories of an ELF binary.

    Parsed straight from the file — the program headers give the dynamic
    section, the dynamic section gives the string-table address, and a
    ``PT_LOAD`` maps that virtual address to a file offset.  A handful of
    seeks, no subprocess, so it is safe in a per-case code path; shelling out
    to ``readelf`` per run would make the harness depend on binutils being
    installed and would put a process spawn on the hot path.

    ``$ORIGIN`` (the binary's own directory, glibc's dynamic token) is expanded
    and the result normalised, because an unexpanded ``$ORIGIN`` — or an
    un-normalised ``$ORIGIN/../lib`` — would hide the very directory a reader
    needs to check and print a path nobody would type.  Returns ``[]`` for a
    non-ELF or unreadable file rather than raising: this is a hazard probe,
    and a probe that crashes the run is worse than one that reports nothing.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(64)
            if len(header) < 64 or header[:4] != b"\x7fELF":
                return []
            is64 = header[4] == 2
            endian = "little" if header[5] == 1 else "big"
            if is64:
                phoff = int.from_bytes(header[32:40], endian)
                phentsize = int.from_bytes(header[54:56], endian)
                phnum = int.from_bytes(header[56:58], endian)
            else:
                phoff = int.from_bytes(header[28:32], endian)
                phentsize = int.from_bytes(header[42:44], endian)
                phnum = int.from_bytes(header[44:46], endian)
            if not phoff or not phnum or phentsize < (56 if is64 else 32):
                return []
            fh.seek(phoff)
            table = fh.read(phnum * phentsize)
            loads, dynamic = [], None
            for i in range(phnum):
                ph = table[i * phentsize:(i + 1) * phentsize]
                if len(ph) < phentsize:
                    return []
                p_type = int.from_bytes(ph[0:4], endian)
                if is64:
                    p_offset = int.from_bytes(ph[8:16], endian)
                    p_vaddr = int.from_bytes(ph[16:24], endian)
                    p_filesz = int.from_bytes(ph[32:40], endian)
                else:
                    p_offset = int.from_bytes(ph[4:8], endian)
                    p_vaddr = int.from_bytes(ph[8:12], endian)
                    p_filesz = int.from_bytes(ph[16:20], endian)
                if p_type == 1:                      # PT_LOAD
                    loads.append((p_vaddr, p_offset, p_filesz))
                elif p_type == 2:                    # PT_DYNAMIC
                    dynamic = (p_offset, p_filesz)
            if dynamic is None:
                return []
            step = 16 if is64 else 8
            fh.seek(dynamic[0])
            raw = fh.read(dynamic[1])
            strtab_vaddr, strsz, wanted = None, None, []
            for off in range(0, len(raw) - step + 1, step):
                width = 8 if is64 else 4
                tag = int.from_bytes(raw[off:off + width], endian,
                                     signed=False)
                if tag == 0:                          # DT_NULL: end of array
                    break
                val = int.from_bytes(raw[off + width:off + 2 * width], endian)
                if tag == 5:                          # DT_STRTAB
                    strtab_vaddr = val
                elif tag == 10:                       # DT_STRSZ
                    strsz = val
                elif tag in (_DT_RPATH, _DT_RUNPATH):
                    wanted.append(val)
            if not wanted or strtab_vaddr is None:
                return []
            strtab_off = None
            for vaddr, offset, filesz in loads:
                if vaddr <= strtab_vaddr < vaddr + filesz:
                    strtab_off = offset + (strtab_vaddr - vaddr)
                    break
            if strtab_off is None:
                return []
            fh.seek(strtab_off)
            blob = fh.read(strsz or 4096)
    except OSError:
        return []
    out: List[str] = []
    origin = str(Path(path).resolve().parent)
    for index in wanted:
        start = index
        while 0 <= start < len(blob) and blob[start] != 0:
            start += 1
        if not 0 <= start <= len(blob):
            continue
        for element in blob[index:start].decode("utf-8", "replace").split(":"):
            if not element:
                continue
            for token in ("${ORIGIN}", "$ORIGIN"):
                element = element.replace(token, origin)
            out.append(os.path.normpath(element))
    return out


def binary_rpath_hazards(oracle: Optional[Dict[str, Optional[str]]] = None
                         ) -> List[str]:
    """Hazard lines for any oracle binary whose RPATH holds the h3d writer.

    The fifth route, and the one no environment variable can close: glibc
    searches a ``DT_RPATH`` **before** ``LD_LIBRARY_PATH``, so a stale
    ``libh3dwriter.so`` dropped into a prefix the binaries carry in their
    RPATH is found by the bare ``dlopen(h3dlib)`` trial
    (``h3d_dl.c:660-666``) no matter what this harness exports.  Measured on
    the oracle built for this box (2026-10-03, ``readelf -d`` on both
    binaries): neither of them carries a ``DT_RPATH`` or a ``DT_RUNPATH``, so
    this probe reports nothing today.  It stays because that absence is a
    property of THAT build, not of the harness: these binaries link the system
    toolchain (``/usr/bin/gfortran`` 13.3.0), while the pre-migration ones,
    built with a conda-forge toolchain, carried a conda prefix in their RPATH —
    the machine-specific leakage this harness exists to remove, and the reason
    the probe reads the ELF instead of trusting the link line.  The reader
    library is found the same honest way, through ``LD_LIBRARY_PATH`` alone
    (``tools/oracle/oracle_env.sh``).

    Policy: **report and let the driver refuse.** :func:`fortran_env` warns
    (an environment builder must not raise); :func:`run_fortran` refuses the
    run, because unlike a stray ``PATH`` entry this one cannot be scrubbed —
    the honest outcome is a loud stop, not a run that silently writes wrong
    H3D files.
    """
    if oracle is None:
        oracle = oracle_paths()
    hazards = []
    for key in ("starter", "engine"):
        binary = oracle.get(key)
        if not binary:
            continue
        poisoned = _dirs_holding_h3d_writer(elf_search_paths(binary))
        if poisoned:
            hazards.append(
                f"{binary} carries DT_RPATH/DT_RUNPATH entry "
                f"{', '.join(poisoned)}, which holds the ABI-incompatible h3d "
                f"writer; glibc searches DT_RPATH before LD_LIBRARY_PATH, so "
                f"h3d_dl.c:660-666 would dlopen it and a run would write "
                f"silently wrong H3D files. No environment change can close "
                f"this — remove the writer from that prefix, or rebuild the "
                f"oracle without that RPATH.")
    return hazards


def _dirs_holding_h3d_writer(entries) -> List[str]:
    """Which of ``entries`` contain an h3d writer ``h3dlib`` file.

    ``h3d_dl.c:58,63`` spell the two names; a directory is only listed when a
    read actually finds one, so an unreadable entry is not accused.
    """
    found = []
    for entry in entries:
        directory = entry or os.curdir
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        if any(name in names for name in _H3D_WRITER_NAMES):
            found.append(directory)
    return found


def _scrub_h3d_search_path(env: Dict[str, str], var: str) -> None:
    """Drop every ``var`` entry that holds the h3d writer; warn about each.

    The warning is not decoration: a caller who put that directory on the
    loader path expects it to be there, and silently dropping it would trade
    a visible hazard for an invisible change.
    """
    value = env.get(var)
    if not value:
        return
    entries = value.split(os.pathsep)
    poisoned = _dirs_holding_h3d_writer(entries)
    if not poisoned:
        return
    kept = [e for e in entries if e not in poisoned]
    env[var] = os.pathsep.join(kept)
    warnings.warn(
        f"{var} entries dropped because they hold the ABI-incompatible h3d "
        f"writer and h3d_dl.c:660-666 would dlopen it from there: "
        f"{', '.join(poisoned)}",
        RuntimeWarning, stacklevel=2)


# ----------------------------------------------------------------------------
# Corpus manifest (Phase 0 / task P0.8)
# ----------------------------------------------------------------------------
#
# The manifest is the generated, hashed answer to "which deck files does the
# differential-validation evidence actually cover, and what are their bytes?".
# It replaces the hand-kept case list that tools/validation_data/inventory.json
# and parity_m41.json are keyed by; those two files stay the authority for
# *ids*, and the manifest carries them over on `case_id` so the three files
# remain joinable.
#
# The loader is deliberately lazy and read-only: it reads the JSON inside the
# function and opens NO deck.  A validation run may legitimately happen with
# the corpus unmounted, and the recorded hashes must still be readable; it
# also means importing this module touches no resource — which is why the
# oracle toolchain lives in oracle_paths() above (P0.9) and not in module-scope
# constants.
#
# A record's `deck` is RELATIVE to a corpus, and the corpus is chosen by the
# environment (`PYRADIOSS_RD_DECKS`), so a bare `rd_decks_dir() / rec["deck"]`
# can silently join a verdict to a *different* file that happens to sit at the
# same relative path.  Two things prevent that: every record carries the
# `corpus_fingerprint` it was hashed from, and `resolve_manifest_record()`
# refuses to resolve a record against a corpus whose fingerprint differs.

MANIFEST_SCHEMA = "pyradioss/rd-decks-manifest/1"
MANIFEST_PATH = os.path.join(REPO, "tools", "validation_data",
                             "rd_decks_manifest.json")

#: Fields every record must carry — the record schema, in full.  A record
#: missing one is a broken manifest, not a deck the harness may silently skip;
#: a record carrying an *extra* field is drift the tests must see too.
MANIFEST_FIELDS = (
    # what was covered, and which bytes
    "case_id", "deck", "hashed_file", "sha256", "size_bytes",
    # which corpus the bytes belong to
    "corpus_fingerprint",
    # identity, carried over from inventory.json (never invented)
    "category", "package", "inventory_classification",
    "inventory_classification_strict",
    # the envelope claim and exactly what it rests on
    "in_envelope", "in_envelope_source", "in_envelope_reason",
    # the measured verdict (parity_m41.json) and its provenance
    "parity_case", "parity_class", "parity_max_rel_rms", "parity_provenance",
    # the reader census (coverage_results_m41.json): the families the port
    # still skips on this deck qualify every verdict recorded above
    "coverage_verdict", "skipped_families", "coverage_hard_skips",
    "coverage_blockers", "coverage_degrade_warnings",
    # whether the verdict above was measured on THESE bytes (false for every
    # M41 row: the sweep ran from a scratchpad extract that no longer exists)
    "parity_run_deck_bytes_verified", "coverage_run_deck_bytes_verified",
)


def load_manifest_doc(path: Optional[str] = None) -> dict:
    """Return the whole manifest document: header + ``decks``.

    The header is not decoration: ``corpus_root.fingerprint`` is what binds a
    record's relative ``deck`` to one corpus, ``counts`` is the coverage
    summary, and ``envelope_rule`` / ``notes`` state what ``in_envelope`` is
    allowed to mean.  :func:`load_manifest` returns the records alone for the
    common case.

    Raises ``FileNotFoundError`` when the manifest has not been generated (run
    ``tools/build_rd_decks_manifest.py``) and ``ValueError`` when the file is
    present but is not a manifest this loader understands.
    """
    target = path or MANIFEST_PATH
    with open(target, encoding="utf-8") as fh:
        doc = json.load(fh)
    schema = doc.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise ValueError(
            f"{target}: schema {schema!r} is not {MANIFEST_SCHEMA!r} — "
            "regenerate it with tools/build_rd_decks_manifest.py")
    records = doc.get("decks")
    if not isinstance(records, list):
        raise ValueError(f"{target}: no 'decks' list")
    documented = set(MANIFEST_FIELDS)
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError(f"{target}: deck record is not an object: {rec!r}")
        missing = sorted(documented - set(rec))
        extra = sorted(set(rec) - documented)
        if missing or extra:
            raise ValueError(
                f"{target}: record {rec.get('deck')!r} does not match the "
                f"record schema (missing {missing}, undocumented {extra}) — "
                "regenerate it with tools/build_rd_decks_manifest.py")
    return doc


def load_manifest(path: Optional[str] = None) -> List[dict]:
    """Return the corpus manifest records, one per starter deck.

    ``path`` defaults to :data:`MANIFEST_PATH`.  Each record names the deck
    (``deck``, a POSIX path relative to the corpus it was hashed from), the
    file its ``sha256`` was computed over (``hashed_file`` — equal to ``deck``
    unless a record says otherwise, so the hashed bytes are never a guess),
    the inventory ``case_id`` it corresponds to (``null`` when the corpus
    holds a deck inventory.json never saw), and the evidence-derived
    ``in_envelope`` flag together with the source of that flag and the
    families the port skips on the deck.

    A record is bound to the corpus named by its ``corpus_fingerprint``, NOT
    to whatever ``paths.rd_decks_dir()`` happens to resolve to; use
    :func:`resolve_manifest_record` to turn one into a path.
    """
    return load_manifest_doc(path)["decks"]


def manifest_corpus_root(doc: Optional[dict] = None) -> str:
    """The corpus the manifest was hashed from, as an absolute path.

    Prefers the checkout-relative ``corpus_root.vendored`` (so the answer does
    not depend on where the manifest was generated) and falls back to the
    absolute ``resolved_at_generation`` recorded at generation time.  This is
    NOT ``paths.rd_decks_dir()``: an env override must not silently re-point a
    record at another extract.
    """
    doc = doc if doc is not None else load_manifest_doc()
    root = doc.get("corpus_root") or {}
    vendored = root.get("vendored")
    if vendored:
        return os.path.abspath(os.path.join(REPO, vendored))
    resolved = root.get("resolved_at_generation")
    if resolved:
        return os.path.abspath(resolved)
    raise ValueError(f"{MANIFEST_PATH}: corpus_root names no location")


def corpus_fingerprint(root: str) -> str:
    """``sha256:<hex>`` over every starter deck in ``root``.

    Path-independent by construction: the digest covers the POSIX-relative
    deck paths and their content hashes, never the root itself, so the same
    corpus at another absolute path (another checkout, another machine, a
    copy in a scratch dir) fingerprints identically — which is what makes it
    usable to *accept* a moved corpus and to *reject* a different one.
    """
    root = os.path.abspath(root)
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.endswith("_0000.rad"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            digest = hashlib.sha256()
            with open(full, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            lines.append(f"{rel}\t{digest.hexdigest()}\n")
    if not lines:
        raise ValueError(f"{root}: no *_0000.rad starter deck to fingerprint")
    lines.sort()
    return "sha256:" + hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def resolve_manifest_record(rec: dict, root: Optional[str] = None, *,
                            verify: bool = True,
                            require_fingerprint: bool = True) -> str:
    """Resolve one record's ``hashed_file`` inside a corpus, refusing a corpus
    it was not hashed from.

    ``root`` defaults to :func:`manifest_corpus_root` — the corpus the
    manifest names — so the result is independent of ``PYRADIOSS_RD_DECKS``.
    A corpus whose :func:`corpus_fingerprint` differs from the record's
    ``corpus_fingerprint`` raises ``ValueError`` rather than returning a path
    to a different file that happens to share the relative name.  With
    ``verify`` (the default) the resolved file is also re-hashed and must match
    the record's ``sha256``.

    A record with **no** ``corpus_fingerprint`` is an error, not a bypass: it
    is the one input shape for which the corpus binding cannot be checked, so
    it has to be said out loud.  ``require_fingerprint=False`` is the explicit
    opt-out for a hand-built record, and it waives the corpus BINDING only —
    ``verify`` still re-hashes the file.
    """
    expected = rec.get("corpus_fingerprint")
    if not expected and require_fingerprint:
        raise ValueError(
            f"record {rec.get('deck')!r} carries no corpus_fingerprint, so the "
            "corpus it was hashed from cannot be checked — pass "
            "require_fingerprint=False only for a record you built yourself, "
            "knowing the binding is then unchecked")
    target_root = os.path.abspath(root) if root else manifest_corpus_root()
    if expected:
        actual = corpus_fingerprint(target_root)
        if actual != expected:
            raise ValueError(
                f"record {rec.get('deck')!r} was hashed from corpus "
                f"{expected}, but {target_root} is {actual} — resolve it "
                "against the corpus the manifest names, or re-hash this one "
                "with tools/build_rd_decks_manifest.py")
    path = os.path.join(target_root, rec["hashed_file"].replace("/", os.sep))
    if verify:
        if not os.path.isfile(path):
            raise ValueError(f"record {rec.get('deck')!r}: {path} is missing")
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != rec["sha256"]:
            raise ValueError(
                f"record {rec.get('deck')!r}: {path} hashes to "
                f"{digest.hexdigest()}, the manifest says {rec['sha256']}")
    return os.path.abspath(path)


def run_cmd(cmd: List[str], cwd: str, timeout: float,
            env: Optional[Dict[str, str]] = None) -> Tuple[int, str, float]:
    """Run a command, return (rc, tail-of-output, elapsed). rc=-9 on timeout."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace")
        out = p.stdout or ""
        return p.returncode, out[-4000:], time.time() - t0
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT", time.time() - t0


# ----------------------------------------------------------------------------
# Listing (.out) harvesting — model size, cycle count, self-reported elapsed
# (M36: per-run wall-clock timing is a first-class deliverable; the harness
# wall-clocks each subprocess AND harvests the solvers' own counters so the
# perf records can carry cycles/s and elements*cycles/s.)
# ----------------------------------------------------------------------------

def harvest_fortran_out(starter_out: str, engine_out: str) -> Dict:
    """Counts from the real Starter/Engine listings.

    Starter: ``NUMNOD: NUMBER OF NODAL POINTS. . .   99`` and the
    ``NUMEL*:`` family (summed); ``ELAPSED TIME...........=  1.43 s``.
    Engine: ``TOTAL NUMBER OF CYCLES  :  1420`` and its ELAPSED TIME.
    """
    info: Dict = {}
    if os.path.exists(starter_out):
        txt = open(starter_out, errors="replace").read()
        m = re.search(r"NUMNOD\s*:\s*NUMBER OF NODAL POINTS[ .]*(\d+)", txt)
        if m:
            info["n_nodes"] = int(m.group(1))
        nel = 0
        seen = set()
        for m in re.finditer(r"^\s*(NUMEL\w*)\s*:[^\n]*?(\d+)\s*$",
                             txt, re.M):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                nel += int(m.group(2))
        if seen:
            info["n_elements"] = nel
        m = re.search(r"ELAPSED TIME[ .]*=\s*([\d.]+)", txt)
        if m:
            info["starter_elapsed_self"] = float(m.group(1))
    if os.path.exists(engine_out):
        txt = open(engine_out, errors="replace").read()
        m = re.search(r"TOTAL NUMBER OF CYCLES\s*:\s*(\d+)", txt)
        if m:
            info["n_cycles"] = int(m.group(1))
        m = re.search(r"ELAPSED TIME[ .]*=\s*([\d.]+)", txt)
        if m:
            info["engine_elapsed_self"] = float(m.group(1))
    return info


def harvest_port_out(starter_out: str, engine_out: str) -> Dict:
    """Counts from the pyradioss listings.

    Starter: ``NUMBER OF NODES . . : 99`` / ``NUMBER OF /BRICK  ELEMENTS``
    (summed) / ``STARTER ELAPSED TIME . . :  0.031 s``.
    Engine: ``CYCLES . . : 1847`` / ``ELAPSED TIME . . :  3.034 s``.

    Plus the compute backend the run actually resolved to:
    ``COMPUTE BACKEND . . . : numba (auto: 40 elements >= 32)``, emitted by
    ``pyradioss/accel/__init__.py`` ``_log_backend`` (``:260-262``) through
    :func:`auto_select_backend` (``:265-268``).  It has to be read back from the
    listing rather than assumed from ``PYRADIOSS_BACKEND``, because ``auto``
    resolves per model (``auto: 40 elements >= 32``) — and a parity row that
    does not say which backend produced its numbers cannot be compared with
    another one (``plan/00_ORCHESTRATION.md`` §1 item 6: *pin the backend for
    before/after comparisons*, ``PYRADIOSS_BACKEND=numpy``).
    """
    info: Dict = {}
    if os.path.exists(starter_out):
        txt = open(starter_out, errors="replace").read()
        m = re.search(r"NUMBER OF NODES[ .]*:\s*(\d+)", txt)
        if m:
            info["n_nodes"] = int(m.group(1))
        nel = sum(int(m.group(2)) for m in re.finditer(
            r"NUMBER OF /(\w+)\s*ELEMENTS[ .]*:\s*(\d+)", txt))
        if re.search(r"NUMBER OF /(\w+)\s*ELEMENTS", txt):
            info["n_elements"] = nel
        m = re.search(r"ELAPSED TIME[ .]*:\s*([\d.]+)", txt)
        if m:
            info["starter_elapsed_self"] = float(m.group(1))
    if os.path.exists(engine_out):
        txt = open(engine_out, errors="replace").read()
        m = re.search(r"^\s*CYCLES[ .]*:\s*(\d+)", txt, re.M)
        if m:
            info["n_cycles"] = int(m.group(1))
        m = re.search(r"ELAPSED TIME[ .]*:\s*([\d.]+)", txt)
        if m:
            info["engine_elapsed_self"] = float(m.group(1))
        m = re.search(r"COMPUTE BACKEND\s*\. \. .*:\s*([A-Za-z0-9_+-]+)"
                      r"\s*\(([^)]*)\)", txt)
        if m:
            info["compute_backend_used"] = m.group(1)
            info["compute_backend_reason"] = m.group(2).strip()
    return info


# ----------------------------------------------------------------------------
# Real-format deck detection + residue fixups (M36)
# ----------------------------------------------------------------------------

def deck_is_real_format(path: str) -> bool:
    """True when the starter deck already carries the fixed-2022 /BEGIN
    input-version card (decks emitted by pyradioss/input/deck_writer.py).
    Such decks are fed to the Fortran Starter without translation."""
    lines = open(path, errors="replace").read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("/BEGIN"):
            for j in range(i + 1, min(i + 4, len(lines))):
                toks = lines[j].split()
                if toks and toks[0].isdigit() and len(toks[0]) >= 3:
                    return True
            return False
    return False


def real_deck_fixups(path: str) -> str:
    """Map the deck_writer's DOCUMENTED RESIDUE fields to their real
    meaning in the Fortran-side deck copy (values preserved; see the
    deck_writer module docstring for why the residues exist):

    * /RWALL/* — the blank d/fric card (the port reads the next card as
      the wall point) becomes d = 1e30, the real equivalent of the
      port's "track all nodes";
    * /INTER/TYPE7 — a value in the Stfac card's 4th field (real Tstart)
      is the PORT's gap_max: moved to the real GAPMAX field (card B,
      2nd field); /INTER/TYPE11 — same field blanked (the real TYPE11
      2020 layout has no gap-cap field);
    * /TH/SECT — the port's section-output request (real Radioss spells
      it /TH/SECTIO and defines sections differently): stripped.
    """
    return real_deck_fixups_text(open(path, errors="replace").read())


def real_deck_fixups_text(text: str) -> str:
    """Text-input variant of :func:`real_deck_fixups` (same mapping)."""
    lines = text.splitlines()
    out: List[str] = []
    i, n = 0, len(lines)

    def is_comment(s: str) -> bool:
        return s.lstrip().startswith(("#", "$"))

    while i < n:
        ln = lines[i]
        s = ln.strip().upper()
        if s.startswith("/TH/SECT/"):
            i += 1
            while i < n and not lines[i].lstrip().startswith("/"):
                i += 1
            out.append("# (/TH/SECT port block stripped for the Fortran "
                       "run — real Radioss spells it /TH/SECTIO)")
            continue
        if s.startswith("/RBODY/"):
            out.append(ln)
            i += 1
            if i < n and not is_comment(lines[i]):
                out.append(lines[i]) # title
                i += 1
                if i < n and not lines[i].lstrip().startswith("/"):
                    card = lines[i]
                    if len(card) <= 50 and card[10:20].strip() and not card[50:].strip():
                        # Port dialect format: master(10) grnod(10) mass(20) icog(10)
                        master = card[:10]
                        grnod = card[10:20]
                        mass = card[20:40] if len(card) >= 40 else " " * 20
                        icog = card[40:50] if len(card) == 50 else " " * 10
                        out.append(master + " " * 30 + mass + grnod + " " * 10 + icog)
                        i += 1
                        if i < n and not lines[i].lstrip().startswith("/") and not is_comment(lines[i]):
                            out.append(lines[i]) # jadd
                            i += 1
                    else:
                        out.append(lines[i])
                        i += 1
                        if i < n and not lines[i].lstrip().startswith("/") and not is_comment(lines[i]):
                            out.append(lines[i]) # jadd
                            i += 1
            continue
        if s.startswith("/RWALL/"):
            out.append(ln)
            i += 1
            replaced = False
            while i < n and not lines[i].lstrip().startswith("/"):
                if (not replaced and lines[i].strip() == ""
                        and not is_comment(lines[i])):
                    out.append(_f20(1e30))     # d: search distance
                    replaced = True
                else:
                    out.append(lines[i])
                i += 1
            continue
        if s.startswith("/INTER/TYPE7/") or s.startswith("/INTER/TYPE11/"):
            is7 = s.startswith("/INTER/TYPE7/")
            blk = [ln]
            i += 1
            while i < n and not lines[i].lstrip().startswith("/"):
                blk.append(lines[i])
                i += 1
            data = [k for k in range(1, len(blk)) if not is_comment(blk[k])]
            nonblank = [k for k in data if blk[k].strip()]
            # nonblank: [title, cardA, cardD(Stfac), ...]
            if len(nonblank) >= 3:
                kd = nonblank[2]
                card = blk[kd].ljust(100)
                tstart = card[60:80].strip()
                if tstart:
                    blk[kd] = card[:60].rstrip()
                    if is7:
                        blanks = [k for k in data if not blk[k].strip()]
                        if blanks:      # card B: Fscalegap | GAPMAX | ...
                            blk[blanks[0]] = " " * 20 + f"{tstart:>20}"
            out.extend(blk)
            continue
        out.append(ln)
        i += 1
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Old-dialect fallback translation
#
# Since M36 the translator lives in pyradioss/input/deck_writer.py (the
# promotion of the M35 per-keyword translator that used to be duplicated
# here — single source of truth).  This wrapper remains as the FALLBACK
# for user decks still written in the port's historical free-format
# dialect: it converts them with the package writer and then applies the
# same documented residue fixups as real-format decks.
# ----------------------------------------------------------------------------

def _f20(v) -> str:
    return f"{float(v):>20.10G}"


def translate_starter_deck(path: str, runname: str) -> Tuple[Optional[str],
                                                             List[str]]:
    """Return (translated text, failure notes).  None text on failure."""
    from pyradioss.input import deck_writer as _dw
    lines = open(path, errors="replace").read().splitlines()
    try:
        text = _dw.starter_deck_from_lines(lines, runname).render()
    except Exception as exc:                      # noqa: BLE001 — reported
        return None, [f"{type(exc).__name__}: {exc}"]
    return real_deck_fixups_text(text), []


def shim_begin_only(path: str) -> str:
    """Insert the 2022 version + unit cards after the /BEGIN title card
    (the minimum to get past the real reader's front door)."""
    lines = open(path, errors="replace").read().splitlines()
    out, i = [], 0
    if not (lines and lines[0].startswith("#RADIOSS STARTER")):
        # the real reader demands '#RADIOSS STARTER' as the first card
        # (ERROR 100201) — two example decks omit it
        out.append("#RADIOSS STARTER")
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip().upper().startswith("/BEGIN"):
            if i + 1 < len(lines) and not lines[i + 1].lstrip().startswith("/"):
                i += 1
                out.append(lines[i])          # keep the run-name card
            out += ["      2022         0",
                    "                  Mg                  mm                   s",
                    "                  Mg                  mm                   s"]
        i += 1
    return "\n".join(out) + "\n"


def strip_engine_stop(path: str) -> str:
    """Drop the port's /STOP block from the engine deck copy: the real
    Engine's reader hits EOF on it (measured: forrtl severe(24) on unit 30);
    everything else passes through unchanged."""
    lines = open(path, errors="replace").read().splitlines()
    out, skip = [], False
    for ln in lines:
        if ln.strip().upper().startswith("/STOP"):
            skip = True
            continue
        if skip and not ln.lstrip().startswith("/"):
            continue                     # /STOP data card(s)
        skip = False
        out.append(ln)
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Time-history CSV parsing + comparison
# ----------------------------------------------------------------------------

# Fortran th_to_csv global column -> port column (pyradioss T01 CSV)
GLOBAL_MAP = [
    ("INTERNAL ENERGY", "IE"),
    ("KINETIC ENERGY", "KE"),
    ("HOURGLASS ENERGY", "HE"),
    ("CONTACT ENERGY", "CE"),
    ("EXTERNAL WORK", "EW"),
    ("MASS", "MASS"),
    ("X-MOMENTUM", "MOMX"),
    ("Y-MOMENTUM", "MOMY"),
    ("Z-MOMENTUM", "MOMZ"),
]


def read_csv_columns(path: str) -> Tuple[List[str], "np.ndarray"]:
    import numpy as np
    rows = []
    header: List[str] = []
    with open(path, newline="", errors="replace") as fh:
        for rec in csv.reader(fh):
            rec = [c.strip().strip('"') for c in rec if c.strip() != ""]
            if not rec:
                continue
            if rec[0].startswith("#"):
                continue
            if not header:
                header = rec
                continue
            try:
                rows.append([float(x) for x in rec])
            except ValueError:
                continue
    n = min(len(r) for r in rows) if rows else 0
    data = np.array([r[:n] for r in rows]) if rows else np.zeros((0, 0))
    return header[:n] if n else header, data


def compare_channels(f_hdr, f_dat, p_hdr, p_dat, part_titles=None):
    """Return list of dicts: channel, rel_rms, final_dev (fractions)."""
    import numpy as np
    res = []
    if f_dat.size == 0 or p_dat.size == 0:
        return res
    tf, tp = f_dat[:, 0], p_dat[:, 0]
    tend = min(tf[-1], tp[-1])
    grid = tp[tp <= tend + 1e-12]
    if len(grid) < 3:
        return res

    def col(hdr, dat, name, exact=True):
        for j, h in enumerate(hdr):
            if (h == name) if exact else (name in h):
                return dat[:, j]
        return None

    pairs = []
    for fname, pname in GLOBAL_MAP:
        a, b = col(f_hdr, f_dat, fname), col(p_hdr, p_dat, pname)
        if a is not None and b is not None:
            pairs.append((pname, a, b))
    # derived total energy
    fie, fke = col(f_hdr, f_dat, "INTERNAL ENERGY"), col(f_hdr, f_dat,
                                                         "KINETIC ENERGY")
    pie, pke = col(p_hdr, p_dat, "IE"), col(p_hdr, p_dat, "KE")
    if all(x is not None for x in (fie, fke, pie, pke)):
        pairs.append(("IE+KE", fie + fke, pie + pke))
    # /TH/PART energies: fortran header 'title ... IE'; port 'P<id>_IE'
    for suffix in ("IE", "KE"):
        fcols = [j for j, h in enumerate(f_hdr)
                 if re.search(rf"\s{suffix}\s*$", h)]
        pcols = [j for j, h in enumerate(p_hdr)
                 if re.fullmatch(rf"P\d+_{suffix}", h)]
        if len(fcols) == 1 and len(pcols) == 1:
            pairs.append((f"part_{suffix}", f_dat[:, fcols[0]],
                          p_dat[:, pcols[0]]))

    # group reference scales: a channel that is numerically ~zero on BOTH
    # sides relative to its group's dominant channel (e.g. the transverse
    # momentum of a uniaxial test) is noise, not physics — it is still
    # reported, but marked insignificant and excluded from MATCH/DEVIATION.
    def group_of(nm):
        if nm.startswith("MOM"):
            return "momentum"
        if nm == "MASS":
            return "mass"
        return "energy"

    interp = {}
    for nm, a, b in pairs:
        interp[nm] = (np.interp(grid, tf, a), np.interp(grid, tp, b))
    gref: Dict[str, float] = {}
    for nm, (ai, bi) in interp.items():
        g = group_of(nm)
        gref[g] = max(gref.get(g, 0.0), float(np.max(np.abs(ai))),
                      float(np.max(np.abs(bi))))

    for nm, a, b in pairs:
        ai, bi = interp[nm]
        denom = max(float(np.max(np.abs(ai))), float(np.max(np.abs(bi))))
        ref = gref[group_of(nm)]
        if denom <= 0 or (ref > 0 and denom < 1e-9 * ref):
            continue    # empty on both sides — nothing to compare
        rel_rms = float(np.sqrt(np.mean((ai - bi) ** 2)) / denom)
        final = float(abs(ai[-1] - bi[-1]) / denom)
        # significance rule (documented in VALIDATION.md): a channel drives
        # the MATCH/DEVIATION class only when it carries at least 1 % of its
        # group's dominant scale — e.g. the transverse momentum of an
        # axially loaded mast (0.1 % of the axial momentum) is numerical
        # noise on both sides; it is still printed, prefixed '~'.
        res.append({"channel": nm, "rel_rms": rel_rms, "final_dev": final,
                    "scale": denom,
                    "significant": bool(denom >= 1e-2 * ref)})
    return res


# ----------------------------------------------------------------------------
# Admissibility — what a parity row is allowed to claim
# ----------------------------------------------------------------------------
#
# The oracle's own record decides this, not this module's judgement:
# ``tools/validation_data/oracle_provenance.json``
# ``admissible_parity_evidence`` marks the T01, the A-files, the restart and
# the starter listing admissible, and H3D files, native ``.k`` reading,
# ``/ALE/STRUCTURED_MESH`` and ``/CHECKSUM_REPORT`` over H3D inadmissible.
# A row therefore names the evidence it rests on and says whether the record
# admits it; if the record cannot be read, the row says *that* rather than
# assuming admissibility.

#: The provenance record, relative to the repo root.
PROVENANCE_REL = os.path.join("tools", "validation_data", "oracle_provenance.json")

#: The evidence channel a T01 comparison rests on, spelled as the provenance
#: spells it, because the key is looked up in the record.
T01_EVIDENCE_KEY = "T01 (binary results table)"

#: The features that record marks inadmissible, with the deck keyword that
#: requests each.  ``/H3D`` is the engine's h3d output family —
#: ``engine/source/input/freform.F:2680`` and ``:2696`` dispatch ``KEY3=='H3D'``
#: with ``NSLASH(KH3D) /= 0``; ``/ALE/STRUCTURED_MESH`` is the S-ALE mesh the
#: harvested reader cannot create
#: (``starter/source/ale/s_ale_message.F90:210-215``, MSGERROR 3153);
#: ``/CHECKSUM_REPORT`` over H3D needs ``libh3dreader.so``, which no reachable
#: extlib contains (``starter/source/output/checksum/checksum_list.cpp:640-687``).
INADMISSIBLE_KEYWORDS = {
    "H3D animation files": ("/H3D",),
    "/ALE/STRUCTURED_MESH (S-ALE)": ("/ALE/STRUCTURED_MESH",),
    "/CHECKSUM_REPORT over H3D files": ("/CHECKSUM_REPORT",),
}

_PROVENANCE_CACHE: Optional[Dict] = None


def provenance_record(refresh: bool = False) -> Optional[Dict]:
    """The oracle provenance record, or ``None`` if it cannot be read.

    Lazy and cached: it is read once per process, and ``refresh=True`` re-reads
    it (a test that edits the record needs that).  ``None`` is a real answer,
    not an error to swallow — the caller then refuses to *claim* admissibility.
    """
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is None or refresh:
        try:
            with open(os.path.join(REPO, PROVENANCE_REL), encoding="utf-8") as fh:
                _PROVENANCE_CACHE = json.load(fh)
        except (OSError, ValueError):
            return None
    return _PROVENANCE_CACHE


def evidence_admissibility(channel: str = T01_EVIDENCE_KEY,
                           record: Optional[Dict] = None) -> Tuple[bool, str]:
    """``(admissible, why)`` for one evidence channel, per the provenance record.

    A channel the record does not mention is **not** admissible: the record is
    the authority on which channels this oracle's numbers may be read from, and
    an unlisted channel is an unknown one, not a permitted one.
    """
    record = record if record is not None else provenance_record()
    if record is None:
        return False, (f"{PROVENANCE_REL} could not be read, so admissibility "
                       f"is unknown and nothing is claimed")
    census = record.get("admissible_parity_evidence") or {}
    verdict = census.get(channel)
    if verdict is None:
        listed = ", ".join(sorted(census)) or "nothing"
        return False, (f"the provenance record does not list {channel!r} as an "
                       f"evidence channel (it lists: {listed})")
    admissible = str(verdict).strip().lower().startswith("yes")
    why = (str(verdict).strip() if admissible
           else str(record.get("inadmissible_parity_evidence_reasons", {})
                    .get(channel, verdict)).strip())
    return admissible, why


def deck_inadmissible_features(deck_text: str) -> List[str]:
    """Which inadmissible features this deck *asks for*, by provenance name.

    A keyword scan, deliberately shallow: it does not claim to be a reader, it
    states which of the record's inadmissible channels this run would have
    produced, so a reader of a parity row can see what the row does **not**
    cover.  Matching is on the keyword at the start of a card or in a comment,
    case-insensitively.
    """
    flagged = []
    upper = deck_text.upper()
    for feature, keywords in INADMISSIBLE_KEYWORDS.items():
        if any(re.search(rf"(^|[\s/]){re.escape(kw.upper())}($|[\s/])", upper)
               for kw in keywords):
            flagged.append(feature)
    return flagged


def evidence_record(deck1_text: Optional[str] = None) -> Dict:
    """The ``evidence`` block every parity row carries.

    Names the channel, the record that admits it, and the inadmissible
    features the deck asks for.  A row with this block can always answer "what
    is this claim standing on, and what did it not look at?".
    """
    admissible, why = evidence_admissibility()
    features = deck_inadmissible_features(deck1_text or "")
    return {"channel": T01_EVIDENCE_KEY, "admissible": admissible,
            "why": why, "source": PROVENANCE_REL,
            "inadmissible_features_in_deck": features}


# ----------------------------------------------------------------------------
# The binary-T01 comparison route (tools.compare_t01, commit c679734)
# ----------------------------------------------------------------------------
#
# ``th_to_csv`` is upstream's own renderer and it is not obtainable on this box
# (``tools/th_to_csv/README.md:1-7`` names a separate repository,
# ``OpenRadioss/Tools``, that this machine cannot reach — the same class of
# blockage as the extlib releases recorded in ``oracle_provenance.json``).
# ``tools/compare_t01.py`` reads the very same binary file
# (``engine/source/output/th/hist1.F:201-316`` for the header, ``hist2.F:302-477``
# for the per-step records) and scores it against the port's own T01 CSV with
# the same 5 % tolerance the recorded sweeps used
# (``tools/validation_data/parity_m41.json`` ``tolerance_rel_rms``).
#
# Two things this route must never do, and does not:
#   * compare raw bytes — the header carries ``ctime()``
#     (``hist1.F:210-234`` via ``engine/source/system/timer_c.c:30-40``), so two
#     runs of one deck differ in 24 bytes and nowhere else;
#   * claim a channel without significant samples — ``score().worst`` is
#     ``NODATA`` when nothing carries signal, and that case keeps the harness's
#     own "no overlapping channels" failure rather than becoming a MATCH.

#: Recorded per row so a reader of ``parity_results.json`` knows which route
#: produced the numbers.
CSV_ROUTE = "th_to_csv CSV"
BINARY_ROUTE = "binary T01 (tools.compare_t01.read_t01)"


def compare_binary_t01(fortran_t01: str, port_csv: str) -> Tuple[List[Dict],
                                                                  Dict]:
    """Channel rows + the roll-up, from the Fortran **binary** T01.

    Returns ``(rows, summary)``.  Each row is
    ``{"channel", "rel_rms", "max_abs", "n", "verdict", "significant"}`` — the
    ``significant`` flag is the scorer's own
    (:attr:`tools.compare_t01.Score.significant`), so a reader sees *why* a
    0.558 next to a ``MATCH`` did not count, exactly as the CSV route's ``~``
    marker explains it there.

    ``summary`` separates the two counts that an earlier revision conflated:

    * ``n_compared`` — channels both sides produced (a verdict != ``NODATA``);
    * ``n_signal`` — of those, the ones that carry signal and therefore decided
      ``worst`` (``Score.significant``), with ``significant_channels`` and
      ``noise_channels`` naming them.

    On ``examples/tensile_bar`` the two differ (11 compared, 5 signal): ``CE``
    is identically zero, ``YMOM``/``ZMOM`` are 1e-16 against an axial momentum of
    2e-4, and ``HE``/``KE``/``P1_2`` sit at 0.02-0.05 % of the dominant energy
    channel — below the 1 %-of-group rule the harness has always applied.  A
    summary that called 11 "significant" overstated the evidence, which is why
    the two counts are separate fields and never one.

    The row shape is otherwise **not** the CSV path's: that one also carries
    ``final_dev`` and ``scale``, which the scorer does not compute
    (``tools/compare_t01.py`` is another task's file and was not modified).
    What carries over is the roll-up, so ``row["max_rel_rms"]`` means what it
    means in ``parity_m41.json`` whichever route produced it.  Consumers that
    read ``final_dev`` off every row must therefore check
    ``row["comparison_route"]`` — which is why the route is recorded on the row
    rather than inferred.
    """
    from tools import compare_t01          # local: keeps import-time work lazy

    reference = compare_t01.read_t01(fortran_t01)
    port = compare_t01.read_port_csv(port_csv)
    result = compare_t01.score(reference, port)
    signal = set(result.significant)
    rows = [{"channel": name, "rel_rms": score.rel_rms,
             "max_abs": score.max_abs, "n": score.n, "verdict": score.verdict,
             "significant": name in signal}
            for name, score in sorted(result.per_channel.items())]
    compared = [r["channel"] for r in rows if r["verdict"] != "NODATA"]
    noise = sorted(set(compared) - signal)
    summary = {"worst_rel_rms": result.worst.rel_rms,
               "worst_verdict": result.worst.verdict,
               "n_compared": len(compared), "n_signal": len(signal),
               "n_channels": len(rows),
               "significant_channels": sorted(signal), "noise_channels": noise,
               "n_samples_reference": int(reference.times.size),
               "n_samples_port": int(port.times.size),
               "tolerance": compare_t01.MATCH_RMS}
    return rows, summary


def measured_step_block(t01_path: str, limit: int = 6) -> Optional[Dict]:
    """Measure the T01's real per-step record block, or ``None``.

    A stopgap diagnostic, and it exists because a refusal must be *informative*:
    ``tools.oracle.oracle_selftest.parse_t01`` fixes the per-step stride at four
    records (``hist2.F`` writes one record per ``/TH`` family that has curves,
    so the shape is **variable**), and when it refuses a file this reports what
    the file actually contains.  Measured on
    ``rd_e/RD-E-1000_Bending/10_Bending/BATOZ/Sf_0.6/ROLLING``: six records per
    step, ``[4, 92, 36, 64, 264, 36]`` bytes, 1605 steps.  Note that the
    hierarchy record's ``NSUBS`` is **not** evidence of a ``/TH/SUBSET``
    request — ``starter/source/starter/contrl.F:671-673`` counts the option and
    then adds one "for global subset", so ``NSUBS >= 1`` always.

    Delete this with the walk fix it works around: once ``parse_t01`` derives
    the block from the header, the reader stops refusing and this has no
    caller.
    """
    try:
        from tools.oracle.oracle_selftest import t01_records
        records = t01_records(Path(t01_path).read_bytes())
    except Exception:                          # noqa: BLE001 — a diagnostic
        return None
    # the per-step time record is one value: 4 bytes (wrtdes.F:121-133)
    starts = [i for i, (_, payload) in enumerate(records) if len(payload) == 4]
    if len(starts) < limit + 1:
        return None
    gaps: Dict[int, int] = {}
    for before, after in zip(starts, starts[1:]):
        gaps[after - before] = gaps.get(after - before, 0) + 1
    stride, count = max(gaps.items(), key=lambda kv: kv[1])
    if stride < 1:
        return None
    # Only the starts whose gap to the next 4-byte record IS the modal stride
    # are real steps; a 4-byte record elsewhere (a parameter record before the
    # data section) is excluded by exactly that test, which is why the shapes
    # below come out as one entry rather than two.
    shapes: Dict[Tuple[int, ...], int] = {}
    for before, after in zip(starts, starts[1:]):
        if after - before != stride:
            continue
        shape = tuple(len(records[before + j][1]) for j in range(stride)
                      if before + j < len(records))
        shapes[shape] = shapes.get(shape, 0) + 1
    return {"stride_records": stride,
            # steps = the first block plus one per modal gap.  Counting the
            # 4-byte records instead would add the pre-data one (a parameter
            # record, not a step) — the gap histogram is what separates them.
            "steps": 1 + count,
            "block_byte_lengths": [list(shape) for shape in sorted(shapes)],
            "records": len(records)}


# ----------------------------------------------------------------------------
# Fortran + pyradioss single-example drivers
# ----------------------------------------------------------------------------

def first_starter_error(out_file: str, log_tail: str) -> str:
    if os.path.exists(out_file):
        txt = open(out_file, errors="replace").read()
        m = re.search(r"ERROR ID\s*:\s*\S+\n\*\*[^\n]*\nDESCRIPTION[^\n]*\n"
                      r"((?:--[^\n]*\n|[^\n]*\n){0,3})", txt)
        if m:
            desc = " | ".join(l.strip() for l in m.group(0).splitlines()[:6]
                              if l.strip())
            return desc[:300]
    tail = " | ".join(l.strip() for l in log_tail.splitlines()[-4:]
                      if l.strip())
    return tail[:300]


def run_fortran(name: str, runname: str, deck0: str, deck1: str,
                workdir: str, shim: str,
                oracle: Optional[Dict[str, Optional[str]]] = None) -> Dict:
    """Run starter+engine+th_to_csv. Returns dict with status/csv/error.

    ``oracle`` defaults to :func:`oracle_paths`, resolved per call; pass it in
    to run several examples against one resolution.  A missing starter/engine
    is an ``oracle-unavailable`` status with the loud diagnostic attached —
    never an empty comparison (see :func:`parity`, which refuses to start
    without them at all).

    The two solvers are invoked through :func:`starter_argv` /
    :func:`engine_argv`, which is where the ``-np`` asymmetry lives
    (``$OR_SRC/INSTALL.md:110-111``).

    The scratch directory is also the solvers' working directory, which is
    upstream's **second** h3d dlopen trial (``h3d_dl.c:634-644``,
    ``getcwd()`` + ``/`` + ``h3dlib``).  So a deck directory that carries a
    writer would be copied into it, and the run would then load the
    ABI-incompatible writer and write silently wrong H3D files.  That is
    refused here — before anything is launched — as an
    ``h3d-writer-in-workdir`` status; :func:`fortran_env` closes the other
    three routes.

    A **fifth** route is checked before that and cannot be closed at all:
    the binaries' own ``DT_RPATH``/``DT_RUNPATH`` (:func:`binary_rpath_hazards`
    reads it out of the ELF; glibc searches it *before*
    ``LD_LIBRARY_PATH``).  A poisoned entry there is an
    ``h3d-writer-in-rpath`` refusal, because no environment change reaches it.
    """
    if oracle is None:
        oracle = oracle_paths()
    starter, engine = oracle.get("starter"), oracle.get("engine")
    if not starter or not engine:
        absent = "starter" if not starter else "engine"
        return {"mode": shim, "dir": None, "status": "oracle-unavailable",
                "error": f"the oracle {absent} is not available; see "
                         f"pyradioss.paths and oracle_paths() for every "
                         f"location that was tried"}
    # The one route no environment can close: glibc searches DT_RPATH before
    # LD_LIBRARY_PATH, so a stale writer in a prefix the binary carries in its
    # RPATH is dlopen'd regardless of what we export.  Refuse before a single
    # file is written.
    rpath_hazards = binary_rpath_hazards(oracle)
    if rpath_hazards:
        return {"mode": shim, "dir": None, "status": "h3d-writer-in-rpath",
                "error": " | ".join(rpath_hazards)}
    rd = os.path.join(workdir, "fortran", name)
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(rd)
    shutil.copytree(os.path.dirname(deck0), rd, dirs_exist_ok=True)
    poisoned = _dirs_holding_h3d_writer([rd])
    if poisoned:
        return {"mode": shim, "dir": rd, "status": "h3d-writer-in-workdir",
                "error": f"{rd} holds an h3d writer "
                         f"({', '.join(_H3D_WRITER_NAMES)}), and it is the "
                         f"solvers' working directory — h3d_dl.c:634-644 "
                         f"dlopens the writer from getcwd(), so this run is "
                         f"refused rather than allowed to write silently "
                         f"wrong H3D files. Move the deck out of that "
                         f"directory and re-run."}
    d0 = os.path.join(rd, os.path.basename(deck0))
    d1 = os.path.join(rd, os.path.basename(deck1))
    info: Dict = {"mode": shim, "dir": rd}

    if shim != "none" and deck_is_real_format(deck0):
        # M36: the deck is already in the real fixed 2022 format (emitted
        # by pyradioss/input/deck_writer.py) — no translation, only the
        # documented residue fixups for the Fortran side.
        info["mode"] = "real-format"
        open(d0, "w").write(real_deck_fixups(deck0))
    elif shim == "translate":
        text, missing = translate_starter_deck(deck0, runname)
        if text is None:
            info.update(status="untranslatable", missing=missing)
            # still probe with the begin shim to capture the real error
            open(d0, "w").write(shim_begin_only(deck0))
            env = fortran_env(oracle)
            rc, tail, dt = run_cmd(starter_argv(starter, d0), rd, 180, env)
            info["probe_error"] = first_starter_error(
                os.path.join(rd, f"{runname}_0000.out"), tail)
            return info
        open(d0, "w").write(text)
    elif shim == "begin":
        open(d0, "w").write(shim_begin_only(deck0))
    else:
        shutil.copy(deck0, d0)
    open(d1, "w").write(strip_engine_stop(deck1)
                        if shim != "none" else open(deck1).read())

    env = fortran_env(oracle)
    rc, tail, dt = run_cmd(starter_argv(starter, d0), rd, 300, env)
    info["starter_rc"] = rc
    info["starter_time"] = round(dt, 2)
    out0 = os.path.join(rd, f"{runname}_0000.out")
    nerr = 0
    if os.path.exists(out0):
        nerr = len(re.findall(r"^ERROR ID", open(out0, errors="replace")
                              .read(), re.M))
    rst = os.path.join(rd, f"{runname}_0000_0001.rst")
    if rc != 0 or nerr or not os.path.exists(rst):
        info.update(status="starter-reject", nerr=nerr,
                    error=first_starter_error(out0, tail))
        return info

    rc, tail, dt = run_cmd(engine_argv(engine, d1), rd, 900, env)
    info["engine_rc"] = rc
    info["engine_time"] = round(dt, 2)
    info.update(harvest_fortran_out(out0,
                                    os.path.join(rd, f"{runname}_0001.out")))
    t01 = os.path.join(rd, f"{runname}T01")
    normal = "NORMAL TERMINATION" in tail or (
        os.path.exists(os.path.join(rd, f"{runname}_0001.out")) and
        "NORMAL TERMINATION" in open(os.path.join(rd, f"{runname}_0001.out"),
                                     errors="replace").read())
    if not os.path.exists(t01):
        info.update(status="engine-fail",
                    error=tail.splitlines()[-1] if tail else f"rc={rc}")
        return info
    converter = oracle.get("th_to_csv")
    if not converter:
        # The engine DID write an admissible T01 (oracle_provenance.json
        # admissible_parity_evidence); what is missing is only the converter
        # that renders it as CSV.  So the T01 itself is handed on: parity
        # compares it with tools.compare_t01.read_t01 (the binary reader
        # landed in c679734 for exactly this), and the record says which
        # route produced its numbers.  Never "compare something else".
        info.update(status="th2csv-missing", t01=t01,
                    error="the engine wrote its T01, but no th_to_csv "
                          "converter is installed (it is a separate upstream "
                          "tool: $OR_SRC/tools/th_to_csv/README.md:1-7 points "
                          "at the OpenRadioss/Tools repository); the binary "
                          "T01 is at " + os.path.basename(t01) + " and will be "
                          "read by tools.compare_t01.read_t01")
        return info
    rc2, tail2, _ = run_cmd(th_to_csv_argv(converter, t01), rd, 120, env)
    csvp = t01 + ".csv"
    if not os.path.exists(csvp):
        info.update(status="th2csv-fail", error=tail2[-200:])
        return info
    info.update(status="ok" if normal else "engine-partial", csv=csvp, t01=t01)
    return info


def run_pyradioss(name: str, runname: str, deck0: str, deck1: str,
                  workdir: str, timeout: float) -> Dict:
    rd = os.path.join(workdir, "pyradioss", name)
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(rd)
    shutil.copytree(os.path.dirname(deck0), rd, dirs_exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    # The compute backend is part of the evidence, so it is recorded twice: what
    # was asked for (``PYRADIOSS_BACKEND``, default ``auto`` — see
    # plan/00_ORCHESTRATION.md §1 item 6, "pin the backend for before/after
    # comparisons") and, from the run's own listing, what ``auto`` resolved to
    # (harvest_port_out reads the COMPUTE BACKEND line the engine logs).  A row
    # that cannot say which backend produced its numbers cannot be compared with
    # another row, and ``auto`` picks per model.
    info: Dict = {"dir": rd,
                  "backend_requested": env.get("PYRADIOSS_BACKEND", "auto")}
    rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.starter", "-i",
                            os.path.basename(deck0)], rd, timeout, env)
    info["starter_rc"] = rc
    info["starter_time"] = round(dt, 2)
    if rc == -9:
        info["status"] = "timeout"
        return info
    if rc != 0:
        info.update(status="starter-fail", error=tail[-300:])
        return info
    remain = max(20.0, timeout - dt)
    rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.engine", "-i",
                            os.path.basename(deck1)], rd, remain, env)
    info["engine_rc"] = rc
    info["engine_time"] = round(dt, 2)
    info.update(harvest_port_out(
        os.path.join(rd, f"{runname}_0000.out"),
        os.path.join(rd, f"{runname}_0001.out")))
    if rc == -9:
        info["status"] = "timeout"
        return info
    csvp = os.path.join(rd, f"{runname}T01.csv")
    if rc != 0:
        info.update(status="engine-fail", error=tail[-300:])
        return info
    if not os.path.exists(csvp):
        # normal termination without a T-file (e.g. /IMPL runs write no
        # explicit time history) — success, but nothing to compare.
        info.update(status="ok-no-th")
        return info
    info.update(status="ok", csv=csvp)
    return info


# ----------------------------------------------------------------------------
# parity mode
# ----------------------------------------------------------------------------

#: Fortran-side statuses that end in ``FORTRAN-FAIL``, mapped to the subtag
#: the class carries.  ``None`` means the BARE class.
#:
#: The invariant, stated correctly (an earlier version of this comment claimed
#: ``parity_m41.json`` carries ``FORTRAN-FAIL`` rows — it does not): **the class
#: vocabulary of a published ``parity_m<NN>.json`` must not shift.**  Its
#: census over M36..M41 is ``MATCH, DEVIATION, PORT-ONLY(implicit),
#: PORT-ONLY(starter-reject), PYRADIOSS-FAIL, SKIPPED-SLOW, NO-CHANNELS`` —
#: **zero ``FORTRAN-FAIL``**.  So:
#:
#: * ``engine-fail`` / ``th2csv-fail`` keep the bare ``FORTRAN-FAIL`` they have
#:   always printed.  They are the two pre-existing emitters and are named here
#:   so a third cannot join them by accident; they remain the one documented
#:   deviation from the recorded vocabulary (an earlier revision of this file
#:   also emitted bare ``FORTRAN-FAIL`` for "nothing comparable", which is now
#:   ``NO-CHANNELS`` — the recorded class for that condition, 4-9 rows a sweep);
#: * every other status gets an ADDITIVE subtag, so ``PORT-ONLY(implicit)``-style
#:   consumers keep working and the base string never appears on its own.
FORTRAN_FAIL_SUBTAGS = {
    "engine-fail": None,
    "th2csv-fail": None,
    "th2csv-missing": "th2csv-missing",
    "oracle-unavailable": "oracle-unavailable",
    "h3d-writer-in-workdir": "h3d-writer-in-workdir",
    "h3d-writer-in-rpath": "h3d-writer-in-rpath",
}


def find_examples(only: Optional[List[str]]) -> List[Tuple[str, str, str, str]]:
    """[(name, runname, deck0, deck1)] sorted explicit-first."""
    exdir = os.path.join(REPO, "examples")
    items = []
    for name in sorted(os.listdir(exdir)):
        d = os.path.join(exdir, name)
        if not os.path.isdir(d):
            continue
        if only and name not in only:
            continue
        d0 = [f for f in os.listdir(d) if f.endswith("_0000.rad")]
        if not d0:
            continue
        runname = d0[0][:-len("_0000.rad")]
        deck0 = os.path.join(d, d0[0])
        deck1 = os.path.join(d, f"{runname}_0001.rad")
        if not os.path.exists(deck1):
            continue
        implicit = "/IMPL" in open(deck1, errors="replace").read()
        items.append((implicit, name, runname, deck0, deck1))
    items.sort(key=lambda it: (it[0], it[1]))     # explicit first
    return [(n, r, a, b) for _, n, r, a, b in items]


def channel_text(entry: Dict) -> str:
    """One channel cell of the console table, for either comparison route.

    A ``NODATA`` channel prints as ``name=-``: the reader route reports one
    (with ``rel_rms = inf``, deliberately, so a stray arithmetic use is loud)
    where the CSV route omits it, and ``inf`` in a table of deviations would
    read as a catastrophic mismatch rather than "not compared".  A channel that
    was compared but **did not carry signal** prints with the CSV route's ``~``
    marker — so a 0.558 never sits next to a ``MATCH`` unexplained, on either
    route.
    """
    if entry.get("verdict") == "NODATA":
        return f"{entry['channel']}=-"
    marker = "" if entry.get("significant", True) else "~"
    return f"{marker}{entry['channel']}={entry['rel_rms']:.3G}"


def parity(args) -> int:
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    only = args.only.split(",") if args.only else None
    examples = find_examples(only)
    if not examples:
        print("no examples found", file=sys.stderr)
        return 2
    # Refuse to start without the oracle.  Every row of a parity table is a
    # comparison against the Fortran solvers, so a run that cannot launch
    # them has nothing to report — and a results file full of uncompared
    # rows would read as evidence.  The diagnostic is the one
    # pyradioss.paths.missing_resource builds, so it lists every candidate.
    oracle = oracle_paths()
    if not (oracle["starter"] and oracle["engine"]):
        absent = [k for k in ("starter", "engine") if not oracle[k]]
        print(f"parity needs the reference (Fortran) solvers; "
              f"{', '.join(absent)} did not resolve. Nothing was run and no "
              f"results file was written.\n\n"
              f"{oracle_report(oracle)}\n", file=sys.stderr)
        return 2
    budget_left = args.budget
    results = []
    retry_queue = []

    def one(name, runname, deck0, deck1, budget_left):
        implicit = "/IMPL" in open(deck1, errors="replace").read()
        row = {"example": name, "runname": runname, "implicit": implicit}

        # ---- Fortran side ------------------------------------------------
        if implicit:
            # the engine controls (/IMPL, /IMPL/FATIG/...) are port
            # extensions; the Fortran chain cannot run them.  Probe the
            # starter anyway so the table carries the real error message.
            f = run_fortran(name, runname, deck0, deck1, workdir,
                            "begin" if args.shim != "none" else "none", oracle)
            f["status"] = "implicit-port-card"
            row["class"] = "PORT-ONLY(implicit)"
        else:
            f = run_fortran(name, runname, deck0, deck1, workdir, args.shim,
                            oracle)
            if f["status"] == "untranslatable":
                row["class"] = "PORT-ONLY(dialect)"
            elif f["status"] == "starter-reject":
                row["class"] = ("PORT-ONLY(dialect)" if args.shim != "translate"
                                else "PORT-ONLY(starter-reject)")
            elif f["status"] in FORTRAN_FAIL_SUBTAGS:
                # ``engine-fail`` and ``th2csv-fail`` keep the BARE class they
                # have always had: a published parity_m<NN>.json is joined on
                # its class vocabulary, which must not shift (see
                # FORTRAN_FAIL_SUBTAGS).  The statuses this harness could not
                # produce before (the converter absent, a solver
                # unresolvable, an h3d writer reachable through the working
                # directory or the binaries' RPATH) get an ADDITIVE subtag,
                # exactly like the existing ``PORT-ONLY(implicit)``
                # convention, so a reader of the console table is not left to
                # guess whether a solver failed or the converter was never
                # installed.
                subtag = FORTRAN_FAIL_SUBTAGS[f["status"]]
                row["class"] = f"FORTRAN-FAIL({subtag})" if subtag \
                    else "FORTRAN-FAIL"
        row["fortran"] = {k: v for k, v in f.items() if k != "dir"}
        # What this row is allowed to claim, decided by the oracle's own
        # provenance record rather than by this module's judgement.
        row["evidence"] = evidence_record(
            Path(deck1).read_text(errors="replace"))

        # ---- pyradioss side ----------------------------------------------
        if budget_left <= 0:
            row["pyradioss"] = {"status": "skipped-budget"}
            row.setdefault("class", "SKIPPED-SLOW")
            return row, budget_left
        t0 = time.time()
        p = run_pyradioss(name, runname, deck0, deck1, workdir,
                          min(args.timeout, max(30, budget_left)))
        budget_left -= time.time() - t0
        row["pyradioss"] = {k: v for k, v in p.items() if k != "dir"}
        # Which compute backend produced the port numbers, requested and
        # resolved (run_pyradioss records both).  plan/00_ORCHESTRATION.md §1
        # item 6 pins numpy for before/after comparisons, and ``auto`` picks
        # numba per model, so a row that does not say which is not comparable
        # with another row.
        row["compute_backend"] = {
            "requested": p.get("backend_requested", "auto"),
            "used": p.get("compute_backend_used"),
            "reason": p.get("compute_backend_reason"),
            "pinned_for_comparison":
                p.get("compute_backend_used") == "numpy"}
        if p["status"] == "timeout":
            row["class"] = "SKIPPED-SLOW"
            return row, budget_left
        if p["status"] not in ("ok", "ok-no-th"):
            row["class"] = "PYRADIOSS-FAIL"

        # ---- comparison ----------------------------------------------------
        # The tolerance the CLASS was decided at, recorded so a row can never
        # contradict itself (a DEVIATION that satisfies the tolerance printed
        # beside it).  ``tolerance`` inside channel_summary stays the historical
        # constant, parity_m41.json's own.
        row["tolerance_used"] = args.tol
        if f.get("csv") and p.get("csv"):
            # The historical route, unchanged: upstream's own converter rendered
            # the Fortran T01 as CSV.  It wins whenever it exists.
            row["comparison_route"] = CSV_ROUTE
            row["fortran"]["comparison_route"] = CSV_ROUTE
            fh, fd = read_csv_columns(f["csv"])
            ph, pd = read_csv_columns(p["csv"])
            ch = compare_channels(fh, fd, ph, pd)
            row["channels"] = ch
            sig = [c for c in ch if c.get("significant", True)]
            if sig:
                worst = max(c["rel_rms"] for c in sig)
                row["max_rel_rms"] = worst
                row["class"] = "MATCH" if worst <= args.tol else "DEVIATION"
            else:
                # No significant channel -> the guard that predates this route
                # and that a validation harness may never drop: a MATCH derived
                # from nothing comparable is the worst thing this file could
                # print.  The class is the recorded one for exactly this
                # condition -- NO-CHANNELS, 4 rows in parity_m41.json -- and
                # NOT the bare FORTRAN-FAIL an earlier revision emitted, which
                # no recorded sweep carries.
                row["class"] = "NO-CHANNELS"
                row["fortran"]["error"] = (
                    f"no overlapping channels: {len(ch)} channel(s) compared, "
                    f"none carrying signal")
        elif f.get("t01") and p.get("csv") and row["evidence"]["admissible"]:
            # The route this task adds: read the Fortran side's own binary T01
            # with tools.compare_t01 (commit c679734) instead of shelling out
            # to th_to_csv, which is unobtainable on this box.  Same tolerance,
            # same significance-filtered roll-up, so max_rel_rms still means
            # what parity_m41.json's means.
            row["comparison_route"] = BINARY_ROUTE
            row["fortran"]["comparison_route"] = BINARY_ROUTE
            row["fortran"]["reader"] = "tools.compare_t01.read_t01"
            try:
                ch, summary = compare_binary_t01(f["t01"], p["csv"])
            except Exception as exc:                # noqa: BLE001 — reported
                # Anything the reader raises (a missing file, a truncated
                # record, a format the walk refuses) is a refusal with the
                # reason quoted — never an empty comparison and never a MATCH.
                row["channels"] = []
                row["class"] = "FORTRAN-FAIL(t01-unreadable)"
                detail = ""
                block = measured_step_block(f["t01"])
                if block:
                    detail = (f"; measured on that file: {block['steps']} "
                              f"steps of {block['stride_records']} records, "
                              f"block byte lengths "
                              f"{block['block_byte_lengths']}, because "
                              f"hist2.F writes one record per /TH family that "
                              f"has curves, so the per-step block is variable "
                              f"while tools.oracle.oracle_selftest.parse_t01 "
                              f"assumes four")
                row["fortran"]["error"] = (
                    f"tools.compare_t01.read_t01 could not use the Fortran T01 "
                    f"at {f['t01']}: {type(exc).__name__}: {exc}{detail}")
                return row, budget_left
            row["channels"] = ch
            row["channel_summary"] = summary
            if summary["worst_verdict"] == "NODATA":
                # Same guard as the CSV route above, and the same recorded
                # class: nothing comparable is NO-CHANNELS.
                row["class"] = "NO-CHANNELS"
                row["fortran"]["error"] = (
                    f"no comparable channel: {summary['n_compared']} of "
                    f"{summary['n_channels']} channels compared, 0 carrying "
                    f"signal (reference {summary['n_samples_reference']} "
                    f"samples, port {summary['n_samples_port']} samples)")
            else:
                worst = summary["worst_rel_rms"]
                row["max_rel_rms"] = worst
                # The scorer's own verdict, not a re-derivation of it: its
                # MATCH_RMS is parity_m41.json's tolerance, and its
                # significance rule is the harness's.  ``--tol`` stays
                # authoritative downward (it can tighten, never loosen past
                # MATCH_RMS), and tolerance_used above says which was applied.
                row["class"] = ("MATCH" if worst <= args.tol
                                and summary["worst_verdict"] == "MATCH"
                                else "DEVIATION")
        elif "class" not in row:
            # Nothing to compare at all (no Fortran CSV and no readable T01, or
            # no port history).  The recorded class for "no comparison" is
            # NO-CHANNELS; a bare FORTRAN-FAIL appears in no recorded sweep.
            row["class"] = "NO-CHANNELS"
        return row, budget_left

    for name, runname, deck0, deck1 in examples:
        print(f"=== {name} ...", flush=True)
        row, budget_left = one(name, runname, deck0, deck1, budget_left)
        if row.get("pyradioss", {}).get("status") in ("starter-fail",
                                                      "engine-fail"):
            retry_queue.append((name, runname, deck0, deck1))
        results.append(row)
        print(f"    -> {row.get('class')}", flush=True)

    # one retry for pyradioss failures (concurrent-edit protection)
    for name, runname, deck0, deck1 in retry_queue:
        if budget_left <= 0:
            break
        print(f"=== retry {name} ...", flush=True)
        time.sleep(30)
        row, budget_left = one(name, runname, deck0, deck1, budget_left)
        for i, r in enumerate(results):
            if r["example"] == name:
                results[i] = row
        print(f"    -> {row.get('class')} (retry)", flush=True)

    out = os.path.join(workdir, "parity_results.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nresults JSON: {out}\n")

    # ---- console table ------------------------------------------------------
    hdr = f"{'example':<34} {'class':<26} {'maxRMS':>8}  channels"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        chs = r.get("channels", [])
        chtxt = " ".join(channel_text(c) for c in chs)
        rms = f"{r.get('max_rel_rms', float('nan')):.3G}" \
            if "max_rel_rms" in r else "-"
        print(f"{r['example']:<34} {r.get('class', '?'):<26} {rms:>8}  "
              f"{chtxt}")
    if any(r.get("comparison_route") == BINARY_ROUTE for r in results):
        from tools import compare_t01      # already imported by the route used
        print("\nchannels read from the Fortran binary T01 by "
              "tools.compare_t01.read_t01 (no th_to_csv on this box):\n"
              "  '-'      not compared — NODATA: only one side asked for the "
              "channel, or there were\n"
              "           too few samples to compare it\n"
              "  maxRMS    the scorer's worst over the channels that carry "
              "signal\n"
              "            (compare_t01.SIGNIFICANCE_FRACTION, the same "
              "1 %-of-group rule the CSV route\n"
              "             applies), so it means what parity_m41.json's "
              "max_rel_rms means\n"
              f"  tolerance {compare_t01.MATCH_RMS} — parity_m41.json's own "
              f"tolerance_rel_rms")
    return 0


# ----------------------------------------------------------------------------
# coverage mode
# ----------------------------------------------------------------------------

def coverage(args) -> int:
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    # Both imports are local: coverage mode is the only consumer of either,
    # and a module-scope import of the reader is module-scope work a parity
    # run never needs.
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import KEYWORD_PARSERS
    rc_all = 0
    for deck in args.decks:
        deck = os.path.abspath(deck)
        name = os.path.splitext(os.path.basename(deck))[0]
        rd = os.path.join(workdir, "coverage", name)
        shutil.rmtree(rd, ignore_errors=True)
        os.makedirs(rd)
        shutil.copytree(os.path.dirname(deck), rd, dirs_exist_ok=True)

        # 1. keyword census straight from the lexer
        census: Dict[str, int] = {}
        for b in read_deck(deck):
            census[b.keyword] = census.get(b.keyword, 0) + 1

        # 2. run the port starter, harvest its messages
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO
        rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.starter",
                                "-i", os.path.basename(deck)], rd,
                               args.timeout, env)
        listing = os.path.join(
            rd, re.sub(r"_0000$", "", name) + "_0000.out")
        text = tail
        if os.path.exists(listing):
            text = open(listing, errors="replace").read() + "\n" + tail
        msgs: Dict[str, List[str]] = {}
        for m in re.finditer(r"\*\*\s*(WARNING|ERROR)[^\n]*\n?([^\n]*)", text):
            line = (m.group(0).replace("\n", " ").strip())[:200]
            kw = "?"
            km = re.search(r"/([A-Z0-9_/\-]+)", line)
            if km:
                kw = km.group(1).split("/")[0]
            msgs.setdefault(kw, []).append(line)

        rows = []
        for kw in sorted(census):
            k0 = kw.split("/")[0]
            ported = k0 in KEYWORD_PARSERS
            note = ""
            for mk, ml in msgs.items():
                for line in ml:
                    if f"/{kw}" in line or (not ported and f"/{k0}" in line):
                        note = line
                        break
                if note:
                    break
            rows.append({"keyword": kw, "blocks": census[kw],
                         "dispatch": "ported" if ported else "UNKNOWN",
                         "message": note})
        result = {"deck": deck, "starter_rc": rc, "elapsed": round(dt, 1),
                  "rows": rows,
                  "other_messages": {k: v[:3] for k, v in msgs.items()},
                  "tail": tail[-1500:]}
        outp = os.path.join(workdir, f"coverage_{name}.json")
        json.dump(result, open(outp, "w"), indent=1)
        print(f"\n### coverage {name}  (starter rc={rc}, {dt:.0f}s)"
              f"  -> {outp}")
        print(f"{'keyword':<28} {'blocks':>6} {'dispatch':<9} message")
        for r in rows:
            print(f"{r['keyword']:<28} {r['blocks']:>6} {r['dispatch']:<9} "
                  f"{r['message'][:90]}")
        if rc not in (0,):
            rc_all = 1
    return rc_all


# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")

    pp = sub.add_parser("parity", help="Fortran vs pyradioss on examples/")
    pp.add_argument("--only", help="comma-separated example names")
    pp.add_argument("--tol", type=float, default=0.05,
                    help="rel RMS tolerance for MATCH (default 0.05)")
    pp.add_argument("--timeout", type=float, default=240,
                    help="per-example pyradioss wall clock cap (s)")
    pp.add_argument("--budget", type=float, default=2700,
                    help="total pyradioss wall clock budget (s)")
    pp.add_argument("--shim", choices=["none", "begin", "translate"],
                    default="translate",
                    help="Fortran-side deck handling (see module docstring)")
    pp.add_argument("--workdir", default=default_workdir())

    cp = sub.add_parser("coverage", help="pyradioss starter keyword census")
    cp.add_argument("decks", nargs="+", help=".rad starter decks")
    cp.add_argument("--timeout", type=float, default=900)
    cp.add_argument("--workdir", default=default_workdir())

    args = ap.parse_args(argv)
    if args.mode == "coverage":
        return coverage(args)
    if args.mode is None:
        args = ap.parse_args(["parity"] + (argv or sys.argv[1:]))
    return parity(args)


if __name__ == "__main__":
    sys.exit(main())
