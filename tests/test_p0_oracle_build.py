"""Gate for P0.4 -- the native cmake build of the reference (Fortran) OpenRadioss.

Three layers, in decreasing order of "always applies":

1. Binary assertions (``test_oracle_binaries_exist``, ``test_oracle_binary_runs``)
   SKIP with an explicit reason while the oracle has not been built, and FAIL
   when ``PYRADIOSS_ORACLE_REQUIRED=1`` (the Phase 0 exit gate sets it).  They
   used to fail unconditionally, which turned the whole fast tier red for
   every other agent in the repo.
2. Acquisition assertions (``test_build_script_is_valid_bash``,
   ``test_provenance_names_the_extlib_source``) always run: the build script
   and the provenance record must exist whether or not the binaries do.
3. ``test_upstream_source_is_untouched`` always runs: ``$OR_SRC`` is read-only
   and must never be dirtied by any of this.

The binaries are resolved here rather than through ``oracle_env.sh`` because
pytest is not guaranteed to have it sourced.  Both exist AND run: the starter's
``-v`` path calls ``HM_BUILD_ID`` from ``libhm_reader``
(``starter/source/starter/execargcheck.F:1200-1201``), so an existence-only
check would pass for a binary that dies on the first call.

Upstream Fortran origins (repo-relative under the ``OpenCourant`` tree,
i.e. ``$OR_SRC`` = /home/valentin/Projects/OpenRadioss/OpenCourant):

* ``CMakeLists.txt:5-30``         -- ``-Dbuild=starter|engine|both`` switch and
  the ``set (EXEC_NAME ${starter})`` / ``set (EXEC_NAME ${engine})`` hand-off
  (a CMake variable, so EXEC_NAME is only set if the caller passes
  ``-Dstarter=``/``-Dengine=``; upstream precedent ``build_windows.bat:86-87,146``).
* ``starter/CMakeLists.txt:136-149`` -- ``arch`` default and the
  ``CMake_Compilers/cmake_${arch}.txt`` include that owns every compile/link
  flag, including the extlib library paths.
* ``starter/CMakeLists.txt:189-193`` -- the ``extlib`` custom target running
  ``load_extlib.py``; ``engine/CMakeLists.txt:266-270`` is the same target.
* ``starter/CMakeLists.txt:242-250`` / ``engine/CMakeLists.txt:347-355`` --
  ``add_executable``, ``add_dependencies(... extlib)`` and the POST_BUILD copy
  into ``${source_directory}/../exec`` (NOT into ``$OR_ROOT``).
* ``engine/CMakeLists.txt:335-340`` -- the CUDA ``*.cu`` glob is discarded when
  ``gpu_cc`` is undefined, so it stays empty without an NVIDIA SDK.
* ``INSTALL.md:34-42`` -- ``RAD_CFG_PATH`` / ``RAD_H3D_PATH`` /
  ``OMP_STACKSIZE`` / ``LD_LIBRARY_PATH``, the runtime environment.
* ``INSTALL.md:110-111`` -- the invocation and the binary names
  (``./starter_linux64_gf -i ... -np 1``) this test asserts.
* ``starter/source/starter/execargcheck.F:200`` (UPCASE, so lowercase ``-v``
  works), ``:1150-1163`` (``PEXECINFO`` -> ``MY_EXIT(0)``), ``:1203-1229`` (the
  starter banner); ``engine/source/engine/execargcheck.F:648-692`` and
  ``:726-742`` are the engine equivalents (upstream spells it "OpenRadios
  Engine").
* ``starter/CMakeLists.txt:210-217`` + ``Compiling_tools/script/or_build_info.py:40-47``
  -- ``build_info.inc`` is generated with ``-arch=${arch}``, hence the
  ``Platform release : linux64_gf`` assertion below.
"""

import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_OR_ROOT = pathlib.Path("/home/valentin/OpenRadioss_or")
DEFAULT_OR_BUILD = DEFAULT_OR_ROOT / "source"
DEFAULT_OR_SRC = pathlib.Path("/home/valentin/Projects/OpenRadioss/OpenCourant")

OR_ROOT = pathlib.Path(os.environ.get("OR_ROOT") or DEFAULT_OR_ROOT)
OR_BUILD = pathlib.Path(os.environ.get("OR_BUILD") or DEFAULT_OR_BUILD)
OR_SRC = pathlib.Path(os.environ.get("OR_SRC") or DEFAULT_OR_SRC)

STARTER = pathlib.Path(
    os.environ.get("OR_STARTER") or OR_ROOT / "bin" / "starter_linux64_gf"
)
ENGINE = pathlib.Path(
    os.environ.get("OR_ENGINE") or OR_ROOT / "bin" / "engine_linux64_gf"
)

BUILD_SCRIPT = REPO / "tools" / "oracle" / "build_oracle.sh"
PROVENANCE = REPO / "tools" / "validation_data" / "oracle_provenance.json"

ORACLE_DISABLED = os.environ.get("PYRADIOSS_ORACLE_DISABLED") == "1"
ORACLE_REQUIRED = os.environ.get("PYRADIOSS_ORACLE_REQUIRED") == "1"

_HINT = (
    "run tools/oracle/mirror_and_fetch.sh (acquires extlib) then "
    "tools/oracle/build_oracle.sh"
)


def _require_oracle():
    """Skip when the oracle is absent, fail when the gate demands it."""
    if STARTER.is_file() and ENGINE.is_file():
        return
    if ORACLE_DISABLED:
        pytest.skip("PYRADIOSS_ORACLE_DISABLED=1")
    if ORACLE_REQUIRED:
        pytest.fail(
            f"oracle binaries missing ({STARTER}, {ENGINE}) and "
            f"PYRADIOSS_ORACLE_REQUIRED=1 is set: {_HINT}"
        )
    pytest.skip(f"oracle not built ({_HINT}); set PYRADIOSS_ORACLE_REQUIRED=1 to enforce")


def _runtime_env():
    """INSTALL.md:34-42, retargeted from OPENRADIOSS_PATH to the mirror."""
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
    _require_oracle()
    for binary in (STARTER, ENGINE):
        assert binary.is_file(), f"oracle binary missing: {binary}"
        assert os.access(binary, os.X_OK), f"oracle binary not executable: {binary}"


@pytest.mark.parametrize(
    "binary,banner",
    [
        (STARTER, "OpenRadioss Starter"),
        (ENGINE, "OpenRadios Engine"),  # upstream spelling, execargcheck.F:727
    ],
    ids=["starter", "engine"],
)
def test_oracle_binary_runs(binary, banner, tmp_path):
    _require_oracle()
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
    assert "Platform release : linux64_gf" in out, (
        f"{binary.name} -v did not report the linux64_gf platform:\n{out}"
    )


def test_build_script_is_valid_bash():
    assert BUILD_SCRIPT.is_file(), f"missing build script: {BUILD_SCRIPT}"
    proc = subprocess.run(
        ["bash", "-n", str(BUILD_SCRIPT)], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"bash -n {BUILD_SCRIPT} failed:\n{proc.stderr}"


def test_provenance_names_the_extlib_source():
    assert PROVENANCE.is_file(), f"missing provenance record: {PROVENANCE}"
    data = json.loads(PROVENANCE.read_text())

    assert data.get("extlib"), "provenance must record an extlib entry"
    extlib = data["extlib"]
    for key in ("kind", "reason_upstream_url_unreachable", "sources"):
        assert key in extlib, f"provenance: extlib.{key} missing"
    # Every source must be pinned: a tag, a commit or a digest -- never "latest".
    for src in extlib["sources"]:
        assert isinstance(src, dict), f"provenance: bad source entry {src!r}"
        assert src.get("uri"), f"provenance: source without uri: {src!r}"
        pinned = src.get("tag") or src.get("commit") or src.get("digest")
        assert pinned, (
            f"provenance: source {src.get('uri')} is not pinned "
            "(need one of tag/commit/digest)"
        )
        assert "latest" not in str(pinned), f"provenance: unpinned tag {pinned!r}"

    # The fields every later parity claim cites. null is allowed, absent is not,
    # and a null must carry its reason (never invented).
    for key in ("upstream_git_sha", "toolchain", "built_on", "solvers"):
        assert key in data, f"provenance: top-level '{key}' missing"
    assert data["upstream_git_sha"], "provenance: upstream_git_sha must be filled"
    assert data["toolchain"].get("gfortran"), "provenance: toolchain.gfortran missing"
    assert data["toolchain"].get("cmake"), "provenance: toolchain.cmake missing"
    if data["built_on"] is None:
        assert data.get("not_built_reason"), (
            "provenance: built_on is null but not_built_reason is missing"
        )
    else:
        assert isinstance(data["built_on"], str), "provenance: built_on must be a date"
    for name, entry in data["solvers"].items():
        assert isinstance(entry, dict), f"provenance: solvers.{name} must be an object"
        assert "built" in entry, f"provenance: solvers.{name}.built missing"
        if entry["built"] is not True:
            assert entry.get("reason"), (
                f"provenance: solvers.{name} is not built, so it needs a reason"
            )


def test_upstream_source_is_untouched():
    proc = subprocess.run(
        ["git", "-C", str(OR_SRC), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"git status failed on {OR_SRC}: {proc.stderr}"
    assert proc.stdout.strip() == "", f"$OR_SRC was modified:\n{proc.stdout}"
