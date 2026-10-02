#!/usr/bin/env python
"""Toolchain probe for building/running the Fortran OpenRadioss "oracle".

Phase 0 (P0.1) of the oracle effort: before any Fortran build is attempted we
record what this Linux box actually provides -- gfortran (and its version),
cmake, make, a *working* OpenMP runtime, python3, docker -- plus whether the
extlib download host is reachable.  The result is written to
``tools/validation_data/toolchain_probe.json`` so a later failure to build can
be attributed to the toolchain instead of guessed at.

``openmp_ok`` is decided by COMPILING AND RUNNING a minimal OpenMP Fortran
program, never by grepping ``-fopenmp`` out of a makefile: a compiler can
accept the flag and still lack a usable runtime, and only a running program
proves ``libgomp`` (and the GOMP/OMP env contract) works.

Mirrors the Linux environment contract documented in
``$OR_SRC/INSTALL.md:34-42`` (OPENRADIOSS_PATH, RAD_CFG_PATH, RAD_H3D_PATH,
OMP_STACKSIZE=400m, LD_LIBRARY_PATH on extlib/hm_reader/linux64).

Usage::

    python tools/oracle/toolchain_probe.py            # probe + write the JSON
    python tools/oracle/toolchain_probe.py --quiet    # no human summary
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


def main(argv=None):
    """Probe the toolchain, write the JSON report, print a summary.  -> rc."""
    argv = list(sys.argv[1:] if argv is None else argv)
    report = probe()
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    if "--quiet" not in argv:
        print("toolchain probe")
        print(_fmt(report))
        print(f"  {'json':<18} {JSON_PATH}")
    return 0


# the public entry point is both callable -- probe() -> dict -- and the CLI
# holder -- probe.main(argv) -> rc; a function attribute keeps both spellings.
probe.main = main


if __name__ == "__main__":
    raise SystemExit(main())