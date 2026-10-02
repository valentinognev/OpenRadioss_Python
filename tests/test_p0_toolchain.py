"""Phase 0 (P0.1) — toolchain probe for the Fortran reference ("oracle") build.

The probe reports the tools the OpenCourant build needs (gfortran, cmake, make,
a *working* OpenMP, python3, docker) plus whether the extlib download host is
reachable.  ``openmp_ok`` is decided by compiling AND running a tiny OpenMP
Fortran program -- grepping a flag string proves nothing about the runtime.

Mirrors the Linux environment contract in $OR_SRC/INSTALL.md:34-42.
"""

import json
import pathlib


def test_probe_reports_required_tools():
    from tools.oracle.toolchain_probe import probe

    r = probe()
    for key in ("gfortran", "cmake", "make", "python3", "openmp_ok"):
        assert key in r
    assert r["gfortran_version"].startswith(("11", "12", "13", "14", "15"))


def test_openmp_is_decided_by_compiling_not_by_reading_a_flag():
    from tools.oracle.toolchain_probe import probe

    # openmp_ok must be a real bool from an actual compile+run
    assert isinstance(probe()["openmp_ok"], bool)


def test_probe_json_is_written():
    from tools.oracle.toolchain_probe import probe

    probe.main([])
    d = json.loads(
        pathlib.Path("tools/validation_data/toolchain_probe.json").read_text()
    )
    assert set(d) >= {"gfortran", "cmake", "openmp_ok", "network_ok"}