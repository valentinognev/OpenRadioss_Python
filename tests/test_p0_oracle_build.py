"""Gate for P0.4 -- the native cmake build of the reference (Fortran) OpenRadioss.

The binaries under test are produced by ``tools/oracle/build_oracle.sh`` from the
writable mirror ``$OR_BUILD`` and installed into ``$OR_ROOT/bin``.  Every later
phase produces differential parity evidence against them, so this gate asserts
two things:

1. both executables exist and are executable (brief Step 3, install step), and
2. both actually RUN -- ``-v`` reaches ``PREXECINFO`` and exits 0.  Existence
   alone proves nothing: the starter's ``-v`` path calls ``HM_BUILD_ID`` from
   ``libhm_reader`` (starter/source/starter/execargcheck.F:1150-1229), so a
   missing ``LD_LIBRARY_PATH`` kills the very first call.  A stub or a
   zero-byte file would pass an existence check and fail here.

Upstream Fortran origins (repo-relative under the ``OpenCourant`` tree,
i.e. ``$OR_SRC`` = /home/valentin/Projects/OpenRadioss/OpenCourant):

* ``CMakeLists.txt:5-30``         -- ``-Dbuild=starter|engine|both`` switch and
  the ``set (EXEC_NAME ${starter})`` / ``set (EXEC_NAME ${engine})`` hand-off.
  EXEC_NAME is the *target name*, i.e. it is only defined if the caller passes
  ``-Dstarter=<name>`` / ``-Dengine=<name>``; ``build_windows.bat:86-87,146``
  is the upstream precedent for the exact names, and ``INSTALL.md:110-111``
  documents them (``./starter_linux64_gf -i ... -np 1``).
* ``starter/CMakeLists.txt:136-149`` -- ``arch`` default and the
  ``CMake_Compilers/cmake_${arch}.txt`` include (which owns every compile and
  link flag, including the extlib library paths).
* ``starter/CMakeLists.txt:189-193`` -- the ``extlib`` custom target that runs
  ``load_extlib.py``; ``engine/CMakeLists.txt:267-269`` is the same target.
* ``starter/CMakeLists.txt:242-250`` / ``engine/CMakeLists.txt:347-355`` --
  ``add_executable``, ``add_dependencies(... extlib)`` and the POST_BUILD copy
  into ``${source_directory}/../exec`` (the trap: NOT into ``$OR_ROOT``).
* ``engine/CMakeLists.txt:335-340`` -- the CUDA ``*.cu`` glob is discarded when
  no ``gpu_cc`` is defined, so the list stays empty without an NVIDIA SDK.
* ``INSTALL.md:38-42`` -- ``RAD_CFG_PATH`` / ``RAD_H3D_PATH`` /
  ``OMP_STACKSIZE`` / ``LD_LIBRARY_PATH``: the runtime environment the built
  binaries need.
* ``starter/source/starter/execargcheck.F:200-217`` (UPCASE, so lowercase
  ``-v`` is accepted), ``:1150-1163`` (``PEXECINFO`` -> ``MY_EXIT(0)``) and
  ``:1203-1229`` (the banner text asserted below);
  ``engine/source/engine/execargcheck.F:648-692`` and ``:726-742`` are the
  engine equivalents.

These tests resolve the binaries themselves (env override, then the documented
default prefix) because pytest is not guaranteed to have ``oracle_env.sh``
sourced.  They FAIL -- never skip -- when the binaries are absent, so a missing
oracle can never masquerade as a green suite.  ``PYRADIOSS_ORACLE_DISABLED=1``
is the only way to switch them off.
"""

import os
import pathlib
import subprocess

import pytest

DEFAULT_OR_ROOT = pathlib.Path("/home/valentin/OpenRadioss_or")
DEFAULT_OR_BUILD = DEFAULT_OR_ROOT / "source"

OR_ROOT = pathlib.Path(os.environ.get("OR_ROOT") or DEFAULT_OR_ROOT)
OR_BUILD = pathlib.Path(os.environ.get("OR_BUILD") or DEFAULT_OR_BUILD)

STARTER = pathlib.Path(
    os.environ.get("OR_STARTER") or OR_ROOT / "bin" / "starter_linux64_gf"
)
ENGINE = pathlib.Path(
    os.environ.get("OR_ENGINE") or OR_ROOT / "bin" / "engine_linux64_gf"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PYRADIOSS_ORACLE_DISABLED") == "1",
    reason="PYRADIOSS_ORACLE_DISABLED=1",
)


def _runtime_env():
    """INSTALL.md:38-42, retargeted from OPENRADIOSS_PATH to the mirror."""
    env = dict(os.environ)
    env["OPENRADIOSS_PATH"] = str(OR_BUILD)
    env["RAD_CFG_PATH"] = str(OR_BUILD / "hm_cfg_files")
    env["RAD_H3D_PATH"] = str(OR_BUILD / "extlib" / "h3d" / "lib" / "linux64")
    env["LD_LIBRARY_PATH"] = (
        str(OR_BUILD / "extlib" / "hm_reader" / "linux64")
        + os.pathsep
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["OMP_STACKSIZE"] = "400m"
    return env


def test_oracle_binaries_exist():
    for binary in (STARTER, ENGINE):
        assert binary.is_file(), (
            f"oracle binary missing: {binary} -- run tools/oracle/"
            "mirror_and_fetch.sh then tools/oracle/build_oracle.sh"
        )
        assert os.access(binary, os.X_OK), f"oracle binary not executable: {binary}"


@pytest.mark.parametrize(
    "binary,banner",
    [
        (STARTER, "OpenRadioss Starter"),
        (ENGINE, "OpenRadios Engine"),  # upstream spelling, execargcheck.F:729
    ],
    ids=["starter", "engine"],
)
def test_oracle_binary_runs(binary, banner, tmp_path):
    assert binary.is_file(), f"oracle binary missing: {binary}"
    proc = subprocess.run(
        [str(binary), "-v"],
        cwd=tmp_path,
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{binary.name} -v exited {proc.returncode}:\n{out}"
    assert banner in out, f"{binary.name} -v printed no version banner:\n{out}"
    # build_info.inc is generated with -arch=${arch}
    # (starter/CMakeLists.txt:210-217, Compiling_tools/script/or_build_info.py:40-47)
    assert "Platform release : linux64_gf" in out, (
        f"{binary.name} -v did not report the linux64_gf platform:\n{out}"
    )