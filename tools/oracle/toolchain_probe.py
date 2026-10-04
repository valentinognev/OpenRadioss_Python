#!/usr/bin/env python
"""Toolchain probe for building/running the Fortran OpenRadioss "oracle".

Phase 0 (P0.1) of the oracle effort: before any Fortran build is attempted we
record what this Linux box actually provides -- gfortran (its own version and
the whole ``--version`` banner it printed), cmake, make, a *working* OpenMP
runtime, python3, docker -- plus TWO reachability answers, because they are two
different facts and one key was claiming to be both: ``network_ok`` (can this box
reach a known-good host at all -- a machine fact) and ``extlib_url_reachable``
(does the one extlib asset URL still answer -- a fact about the release, which
answers 404 on a box whose network is fine).  The result lands in
``tools/validation_data/toolchain_probe.json`` so a later failure to build can be
attributed to the toolchain instead of guessed at.

Every recorded value is a property of the MACHINE: a tool resolved on PATH, a
capability the probe exercised, or a reachability answer.  None of them is a
property of the checkout or of the interpreter that happened to run the probe,
because a record that changes when the worktree moves is a record of the
worktree, not of the build machine.

``openmp_ok`` is decided by COMPILING AND RUNNING a minimal OpenMP Fortran
program, never by grepping ``-fopenmp`` out of a makefile: a compiler can
accept the flag and still lack a usable runtime, and only a running program
proves ``libgomp`` (and the GOMP/OMP env contract) works.

Mirrors the Linux environment contract documented in
``$OR_SRC/INSTALL.md:34-42`` (OPENRADIOSS_PATH, RAD_CFG_PATH, RAD_H3D_PATH,
OMP_STACKSIZE=400m, LD_LIBRARY_PATH on extlib/hm_reader/linux64).

THE RECORD IS A GATED FACT, NOT A CACHE
----------------------------------------
The JSON is committed, so it is a *claim* about a build machine, and a claim
only means something if something checks it.  Two rules keep it honest, and
both are load-bearing:

* **Writing is an explicit act.**  :func:`main` is the only function here that
  touches the file, and it is only reached from the command line.  Reading the
  record (:func:`read_record`), comparing it to a fresh probe (:func:`differences`)
  and probing itself (:func:`probe`) are all pure -- so running the test suite
  cannot rewrite history, and ``git status`` staying clean is evidence that the
  suite verified the record instead of refreshing it.
* **A difference is reported with the repair, never alone.**  :func:`explain`
  names every differing key with both values and quotes
  :data:`REFRESH_COMMAND`.  The check itself is
  ``tests/test_p0_toolchain.py``; this module measures, it does not rule.

Usage::

    python -m tools.oracle.toolchain_probe            # probe + REFRESH the JSON
    python -m tools.oracle.toolchain_probe --quiet    # refresh, no summary
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "tools" / "validation_data" / "toolchain_probe.json"


def _or_src_default():
    """Where ``$OR_SRC`` is when the environment does not say -- resolved,
    never hardcoded.

    The default used to be one developer's absolute checkout path.  That made
    ``EXTLIB_VERSION.json`` resolvable on exactly one machine: everywhere else
    the manifest read failed and ``extlib_url_reachable`` recorded ``false``
    -- a false record produced by a missing default rather than by a missing
    release.  Order, each step derivable from something portable:

    1. ``$OR_SRC`` when set, honoured verbatim.  A *stale* one is not this
       probe's business to reinterpret: a set variable is the operator's
       decision (``plan/00_ORCHESTRATION.md`` §4.1 rule 1) and
       :func:`pyradioss.paths` already fails it loudly.
    2. :func:`pyradioss.paths.or_src` -- the project's single resolver, whose
       own last candidate is the upstream checkout sitting **beside this
       repository**.  It is derived from ``__file__``, so it follows the
       worktree to whatever directory the checkout lives in.
    3. :data:`OR_SRC_UNRESOLVED` -- a single component that cannot exist, so
       the manifest read raises :class:`OSError` and ``_extlib_url`` reports
       the honest "no url" for a box that genuinely has no source tree.

    Never raises: this runs at import time, and a probe that cannot start
    cannot record why.
    """
    configured = os.environ.get("OR_SRC")
    if configured:
        return pathlib.Path(configured)
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from pyradioss import paths

        return paths.or_src()
    except (ImportError, OSError, ValueError, Warning) as exc:
        # FileNotFoundError from the resolver is an OSError; a stale export
        # turns its RuntimeWarning into an exception under -W error, hence the
        # Warning.  Either way the answer is the same: nothing to read.
        sys.stderr.write(
            f"toolchain probe: $OR_SRC is unset and did not resolve ({exc}); "
            "extlib_url_reachable will be measured as false\n")
        return pathlib.Path(OR_SRC_UNRESOLVED)


#: The stand-in for "$OR_SRC points nowhere resolvable".  A single relative
#: component, so it can never exist and never names a machine.
OR_SRC_UNRESOLVED = "<OR_SRC unset and unresolved>"

#: The upstream source tree, from :func:`_or_src_default`.
OR_SRC = _or_src_default()
EXTLIB_MANIFEST = OR_SRC / "EXTLIB_VERSION.json"
NET_TIMEOUT = 5.0
COMPILE_TIMEOUT = 120

#: A known-good host for the *machine* network fact, and the reason it is not the
#: extlib asset URL: that asset answers ``404`` on this box (the organisation
#: moved the release), while this host answers ``200``, and a record key called
#: ``network_ok`` must never claim a dead network because one download URL moved.
#: It is the operator that serves the asset, so answering here is the precondition
#: for fetching it.  MEASURED on every run, never assumed: :func:`probe_network`
#: HEADs it and records what came back.
NETWORK_HOST_URL = "https://github.com/"

#: A parenthesised group of a ``--version`` banner, with its contents: the
#: *packaging* of a compiler, never its own version.  Matched so the whole group
#: goes, contents included -- deleting only the brackets would leave the vendor's
#: version behind as an ordinary token.
_PARENTHESISED = re.compile(r"\([^()]*\)")

#: The one command that refreshes the committed record.  It lives here so that
#: a drift failure can *quote* the fix instead of merely reporting the
#: difference -- a failure that does not say how to repair itself sends the
#: next agent looking for the writer instead of running the refresh.
REFRESH_COMMAND = ".venv/bin/python -m tools.oracle.toolchain_probe"

#: Every key the record is required to carry.  A record missing one cannot be
#: compared key-by-key, so it is rejected outright rather than half-verified.
#: The shape is ``<tool>`` + ``<tool>_version`` + (for the compiler)
#: ``<tool>_banner``: the path the build invokes, the version it reports, and
#: for gfortran the whole first line it printed, because the version alone does
#: not say which *packaging* of the compiler produced it.  The last two keys are
#: the two reachability facts, kept apart on purpose (see :data:`NETWORK_HOST_URL`).
RECORD_KEYS = (
    "gfortran",
    "gfortran_version",
    "gfortran_banner",
    "cmake",
    "cmake_version",
    "make",
    "openmp_ok",
    "python3",
    "python3_version",
    "docker",
    "network_ok",
    "extlib_url_reachable",
)

#: Every key is a property of the MACHINE -- a tool resolved on PATH, a
#: capability, or a reachability answer -- and none is a property of the
#: checkout or of the interpreter that ran the probe.  That is load-bearing:
#: ``python3`` used to hold ``sys.executable``, which made the record
#: checkout-local and made the gate fail whenever the suite ran under a
#: different venv, i.e. it reported "the toolchain moved" when nothing the build
#: uses had moved.  tests/test_p0_toolchain.py holds the invariant; do not add a
#: key whose value names the repository, the virtualenv or ``sys.executable``.

OMP_SOURCE = """program t
integer i
!$omp parallel do private(i)
do i=1,10
enddo
!$omp end parallel do
end program t
"""


def _which(tool):
    return shutil.which(tool)


def _run(cmd, timeout=30):
    """Run ``cmd``; return (returncode, stdout+stderr).  Never raises."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _version(exe, flag="--version"):
    """First-line version string of ``exe``, or "" when it cannot run."""
    if not exe:
        return ""
    rc, out = _run([exe, flag])
    if rc != 0 or not out:
        return ""
    return out.strip().splitlines()[0].strip()


def _gcc_version(banner):
    """The compiler's OWN version out of a ``--version`` first line.

    GCC prints the *packaging* in parentheses and its own version outside them,
    and the two do not always carry the same numbers -- so the parenthesised
    groups are removed **whole** before anything is searched, and only what the
    compiler said about itself is left.  The banner shapes this parser reads,
    one row per distro, look like this::

        GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0  Debian/Ubuntu
        GNU Fortran (GCC) 13.2.0                             upstream GCC
        gfortran (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)     Red Hat
        gfortran-13 (Homebrew 13.1) 13.2.0                   Homebrew
        GNU Fortran (GCC-14.2.0) 14.2.0                     Fedora/openSUSE
        gfortran-13                                            bare program name
        13.2.0                                                bare version

    Rules, in order:

    1. the first remaining token that is *entirely* dot-separated digits with at
       least two parts.  Requiring every part to be numeric is what rejects the
       Debian package string ``13.3.0-6ubuntu2~24.04.1``; taking the groups
       with their contents is what rejects the vendor version *inside* them --
       the Homebrew row above would otherwise yield ``13.1``, which is the
       packaging's number, not the compiler's.  Red Hat's bare date
       ``20150623`` has no dot and is skipped for the same reason.
    2. otherwise a ``-<major>`` suffix on the program name -- the only version
       the bare-program-name row carries.
    3. otherwise the whole first line, verbatim: an exotic banner then records
       what the compiler actually said instead of an empty string, which the
       gate can still compare.

    One rule per shape above is a test row in tests/test_p0_toolchain.py, so a
    parser repaired against the banner of whichever box ran it cannot quietly
    regress on the next distro.
    """
    unvendored = " ".join(_PARENTHESISED.sub(" ", banner).split())
    tokens = unvendored.split()
    for token in tokens:
        parts = token.split(".")
        if len(parts) >= 2 and all(part.isdigit() for part in parts):
            return token
    if tokens:  # rule 2: the version a bare program name carries
        program, sep, suffix = tokens[0].rpartition("-")
        if sep and program and all(part.isdigit() for part in suffix.split(".")):
            return suffix
    return banner  # rule 3: the banner verbatim


def _gfortran_version(exe):
    """``13.3.0`` out of ``GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0``."""
    first = _version(exe)
    return _gcc_version(first) if first else ""


def probe_openmp(gfortran):
    """True only if ``gfortran -fopenmp`` compiles AND the binary exits 0."""
    if not gfortran:
        return False
    with tempfile.TemporaryDirectory(prefix="openradioss_omp_") as tmp:
        src = pathlib.Path(tmp) / "omp_probe.f90"
        exe = pathlib.Path(tmp) / "omp_probe"
        src.write_text(OMP_SOURCE, encoding="utf-8")
        rc, out = _run([gfortran, "-fopenmp", str(src), "-o", str(exe)],
                       timeout=COMPILE_TIMEOUT)
        if rc != 0:
            sys.stderr.write(f"openmp probe: compile failed\n{out}\n")
            return False
        rc, out = _run([str(exe)], timeout=COMPILE_TIMEOUT)
        if rc != 0:
            sys.stderr.write(f"openmp probe: run failed (rc={rc})\n{out}\n")
            return False
    return True


def probe_network():
    """The MACHINE fact: is a known-good host reachable at all?  HEAD only.

    Deliberately NOT the extlib asset URL.  That URL answers ``404`` here -- the
    release moved -- and a key called ``network_ok`` reporting ``false`` on a
    box whose network demonstrably works is a false record.  What this box can
    reach is what this key says; where the release lives is
    :func:`probe_extlib_url`'s to say.
    """
    return _head_ok(NETWORK_HOST_URL)


def probe_extlib_url():
    """The RELEASE fact: does the extlib asset URL in the manifest answer?

    Kept as its own key (recorded as ``extlib_url_reachable``) so the honest
    ``network_ok`` above does not cost the information Task 0.3 needs: a 404
    here says the asset moved, which is a fact about the release and not about
    the machine.
    """
    url = _extlib_url()
    return _head_ok(url) if url else False


def _head_ok(url):
    """True if ``url`` answers a HEAD with 2xx/3xx.  Nothing is downloaded.

    A ``404`` is an *answer*, so it raises :class:`urllib.error.HTTPError` (a
    :class:`urllib.error.URLError`) and lands in the ``False`` branch together
    with a real connectivity failure -- the caller decides what that False
    means, which is exactly why the two facts have two keys.
    """
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as resp:
            return 200 <= resp.status < 400
    except (urllib.error.URLError, OSError, ValueError) as exc:
        sys.stderr.write(f"network probe: {url} unreachable: {exc}\n")
        return False


def _extlib_url():
    """The extlib asset URL declared by ``$OR_SRC/EXTLIB_VERSION.json`` ("" if none)."""
    try:
        return json.loads(EXTLIB_MANIFEST.read_text(encoding="utf-8"))["url"]
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write(f"extlib probe: no url in {EXTLIB_MANIFEST}: {exc}\n")
        return ""


def probe():
    """Return the toolchain inventory of this machine.

    ``python3`` is the interpreter the ORACLE BUILD invokes -- ``shutil.which``
    of the name upstream hardcodes on non-Windows
    (``starter/CMakeLists.txt:18`` / ``engine/CMakeLists.txt``: ``set(PYTHON_EXEC
    "python3")``), resolved from PATH exactly as cmake resolves it.  It is
    deliberately NOT ``sys.executable``: this probe can be run by any
    interpreter, and recording the one that happened to run it would put the
    checkout and its virtualenv into a record whose job is to describe the
    build machine.
    """
    gfortran = _which("gfortran")
    cmake = _which("cmake")
    python3 = _which("python3")
    return {
        "gfortran": gfortran or "",
        "gfortran_version": _gfortran_version(gfortran),
        "gfortran_banner": _version(gfortran),
        "cmake": cmake or "",
        "cmake_version": _version(cmake),
        "make": _which("make") or "",
        "openmp_ok": probe_openmp(gfortran),
        "python3": python3 or "",
        "python3_version": _version(python3),
        "docker": _which("docker") or "",
        "network_ok": probe_network(),
        "extlib_url_reachable": probe_extlib_url(),
    }


def _fmt(report):
    return "\n".join(f"  {k:<18} {v}" for k, v in report.items())


# ---------------------------------------------------------------------------
# Reading and comparing the committed record.  Nothing below writes: the record
# is a fact to be checked, and the only thing allowed to change it is the
# operator running REFRESH_COMMAND on purpose.
# ---------------------------------------------------------------------------

def read_record(path=None):
    """Return the committed record as a dict.

    Pure read -- no probing, no writing, no network.  Raises ``ValueError``
    with the path in the message when the file is missing or is not a JSON
    object, because "the record is gone" and "the record is a list" are both
    drift an agent has to be told about, not tracebacks to guess from.
    """
    path = pathlib.Path(path) if path is not None else JSON_PATH
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"no toolchain record at {path}: {exc}") from exc
    try:
        record = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(
            f"{path} holds a {type(record).__name__}, not a JSON object")
    return record


def differences(recorded, fresh):
    """Every way ``recorded`` disagrees with a fresh ``fresh`` probe.

    Returns an ordered list of human-readable lines, empty when the two agree.
    Compares the *union* of both key sets, so a key the probe stopped
    measuring is reported as loudly as one whose value moved.
    """
    lines = []
    for key in sorted(set(recorded) | set(fresh)):
        if key not in recorded:
            lines.append(f"{key}: absent from the record, measured {fresh[key]!r}")
        elif key not in fresh:
            lines.append(f"{key}: recorded {recorded[key]!r}, not measured")
        elif recorded[key] != fresh[key]:
            lines.append(
                f"{key}: recorded {recorded[key]!r}, measured {fresh[key]!r}")
    return lines


def explain(diffs):
    """Render the drift, then the command that repairs it.

    No per-key classification: every key is a machine fact now (see the note
    beside ``RECORD_KEYS``), so a difference is a difference and the list of
    them is the whole diagnosis.
    """
    if not diffs:
        return ""
    return "\n".join(f"  {line}" for line in diffs + [
        "",
        f"Refresh the record with:  {REFRESH_COMMAND}",
    ])


def main(argv=None):
    """Probe the toolchain, write the report, print a summary.  -> rc.

    This is the ONLY writer of the record, and it is reached from the command
    line or from an operator who means to refresh.  Nothing in ``tests/`` may
    call it against :data:`JSON_PATH`: a suite that rewrites the record it is
    meant to check cannot fail, and it leaves the worktree dirty on every run.
    (The suite may call it against a temporary path, to prove the refresh act
    still writes -- that is a different claim and writes nothing committed.)

    Refreshing is an explicit act because the record is a committed claim, and
    a claim rewritten as a side effect of running the suite is not a claim.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    report = probe()
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    if "--quiet" not in argv:
        print("toolchain probe -- refreshed the record")
        print(_fmt(report))
        print(f"  {'json':<18} {JSON_PATH}")
    return 0


# the public entry point is both callable -- probe() -> dict -- and the CLI
# holder -- probe.main(argv) -> rc; a function attribute keeps both spellings.
probe.main = main


if __name__ == "__main__":
    raise SystemExit(main())
