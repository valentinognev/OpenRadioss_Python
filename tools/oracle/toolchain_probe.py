#!/usr/bin/env python
"""Toolchain probe for building/running the Fortran OpenRadioss "oracle".

Phase 0 (P0.1) of the oracle effort: before any Fortran build is attempted we
record what this Linux box actually provides -- gfortran (and its version),
cmake, make, a *working* OpenMP runtime, python3, docker -- plus whether the
extlib download host is reachable.  The result lands in
``tools/validation_data/toolchain_probe.json`` so a later failure to build can
be attributed to the toolchain instead of guessed at.

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
  names every differing key, classifies it (toolchain vs run-local), and
  quotes :data:`REFRESH_COMMAND`.  The check itself is
  ``tests/test_p0_toolchain.py``; this module measures, it does not rule.

Usage::

    python -m tools.oracle.toolchain_probe            # probe + REFRESH the JSON
    python -m tools.oracle.toolchain_probe --quiet    # refresh, no summary
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "tools" / "validation_data" / "toolchain_probe.json"
OR_SRC = pathlib.Path(
    os.environ.get("OR_SRC", "/home/valentin/Projects/OpenRadioss/OpenCourant")
)
EXTLIB_MANIFEST = OR_SRC / "EXTLIB_VERSION.json"
NET_TIMEOUT = 5.0
COMPILE_TIMEOUT = 120

#: The one command that refreshes the committed record.  It lives here so that
#: a drift failure can *quote* the fix instead of merely reporting the
#: difference -- a failure that does not say how to repair itself sends the
#: next agent looking for the writer instead of running the refresh.
REFRESH_COMMAND = ".venv/bin/python -m tools.oracle.toolchain_probe"

#: Every key the record is required to carry.  A record missing one cannot be
#: compared key-by-key, so it is rejected outright rather than half-verified.
RECORD_KEYS = (
    "gfortran",
    "gfortran_version",
    "cmake",
    "cmake_version",
    "make",
    "openmp_ok",
    "python3",
    "docker",
    "network_ok",
)

#: Keys whose value is a property of *the interpreter that ran the probe*
#: rather than of the machine's toolchain.  ``python3`` is ``sys.executable``:
#: it names the venv the probe ran under, which moves with the checkout.  It is
#: still compared like every other key -- the split only decides how a drift is
#: *explained*, so an agent can tell "my compiler moved" from "I ran the suite
#: under a different interpreter" without reading the diff.
RUN_LOCAL_KEYS = ("python3",)

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


def _gfortran_version(exe):
    """'15.2.0' out of 'GNU Fortran (GCC) 15.2.0 ...' -- else the raw 1st line."""
    first = _version(exe)
    if not first:
        return ""
    for tok in first.replace("(", " ").replace(")", " ").split():
        parts = tok.split(".")
        if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
            return ".".join(parts[:3]) if len(parts) >= 3 else ".".join(parts[:2])
    return first


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
    """True if the extlib URL answers a HEAD request.  Nothing is downloaded."""
    url = ""
    try:
        url = json.loads(EXTLIB_MANIFEST.read_text(encoding="utf-8"))["url"]
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write(f"network probe: no url in {EXTLIB_MANIFEST}: {exc}\n")
        return False
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as resp:
            return 200 <= resp.status < 400
    except (urllib.error.URLError, OSError, ValueError) as exc:
        sys.stderr.write(f"network probe: {url} unreachable: {exc}\n")
        return False


def probe():
    """Return the toolchain inventory of this machine."""
    gfortran = _which("gfortran")
    cmake = _which("cmake")
    return {
        "gfortran": gfortran or "",
        "gfortran_version": _gfortran_version(gfortran),
        "cmake": cmake or "",
        "cmake_version": _version(cmake),
        "make": _which("make") or "",
        "openmp_ok": probe_openmp(gfortran),
        "python3": sys.executable,
        "docker": _which("docker") or "",
        "network_ok": probe_network(),
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
    """Render drift lines with each key's class named, plus the fix."""
    out = []
    for line in diffs:
        key = line.split(":", 1)[0]
        kind = ("run-local, moves with the interpreter that ran the probe"
                if key in RUN_LOCAL_KEYS else "toolchain")
        out.append(f"  [{kind}] {line}")
    if out:
        out.append("")
        out.append(f"Refresh the record with:  {REFRESH_COMMAND}")
    return "\n".join(out)


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