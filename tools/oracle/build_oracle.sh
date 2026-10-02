#!/usr/bin/env bash
# Build the reference (Fortran) OpenRadioss starter/engine with cmake and
# install both executables into $OR_ROOT/bin.
#
# ---------------------------------------------------------------------------
# THREE UPSTREAM TRAPS THIS SCRIPT EXISTS TO HANDLE (none of them are
# mentioned in the two-line snippet in the task brief; all were found by
# reading the upstream CMake files and confirmed against this box)
# ---------------------------------------------------------------------------
#
#  1. EXEC_NAME must be passed in, and it is a *target* name, not a path.
#     The top-level CMakeLists.txt:18-30 does
#        set (EXEC_NAME ${starter})     # <- ${starter}: a CMake VARIABLE,
#        set (EXEC_NAME ${engine})      #    not the string "starter"
#     so `add_executable(${EXEC_NAME} ...)` (starter/CMakeLists.txt:242,
#     engine/CMakeLists.txt:347) gets an EMPTY target name unless the caller
#     passes -Dstarter=... / -Dengine=....  build_windows.bat:86-87,146 is the
#     upstream precedent.  The value must also stay path-free: a name with a
#     '/' is rejected by add_executable() ("reserved or not valid for certain
#     CMake features"), so the documented binary name starter_linux64_gf
#     (INSTALL.md:110-111) canNOT be the target name here -- see trap 3 for how
#     the documented names are still produced.
#
#  2. `-Dbuild=both` in ONE cmake run is broken: both component files declare
#     a custom target literally named `extlib`
#     (starter/CMakeLists.txt:189-193, engine/CMakeLists.txt:266-270) and CMake
#     refuses the second one:
#         CMake Error at engine/CMakeLists.txt:266 (add_custom_target):
#           add_custom_target cannot create target "extlib" because another
#           target with the same name already exists.
#     One cmake run per component is also what upstream does
#     (build_windows.bat cds into starter/ or engine/ first).
#
#  3. `-Dbuild=<c>` cannot be combined with `-Dstarter=starter_linux64_gf`.
#     CMakeLists.txt:18 expands to
#         if (starter STREQUAL "starter" OR starter STREQUAL "both")
#     and `if()` DEREFERENCES a defined variable: with
#     -Dstarter=starter_linux64_gf the left side becomes the string
#     "starter_linux64_gf", which matches neither "starter" nor "both", so the
#     subdirectory is silently NOT added (no error, an empty build system).
#     Same trap for the engine.  Therefore: -Dstarter=starter,
#     -Dengine=engine, -Dbuild=starter / -Dbuild=engine, one component per
#     build directory, and the link products are renamed to the documented
#     names on install.
#
#  4. The POST_BUILD copy does NOT install into $OR_ROOT.  Both component
#     files copy the link product into ${source_directory}/../exec, i.e.
#     $OR_BUILD/exec (starter/CMakeLists.txt:246-250, engine/CMakeLists.txt:
#     351-355 -- verified in the generated
#     build/<c>/<c>/CMakeFiles/<name>.dir/build.make).  `cmake --install` is
#     useless here: install(TARGETS ${EXECUTABLES}) refers to a variable
#     upstream never sets (last 5 lines of both component files).  So the move
#     into $OR_ROOT/bin is done explicitly at the end.
#
# Additional flags that are NOT in the brief's snippet but are needed for the
# tree to build what the oracle is supposed to be (all read off upstream):
#   -Dprecision=dp  starter/CMakeLists.txt:101-119 and
#                   CMake_Compilers/cmake_linux64_gf.txt:60-64 branch on it;
#                   it also selects share/r4 vs share/r8 include dirs.
#   -DMPI=smp       engine/CMake_Compilers/cmake_linux64_gf.txt:21-65 -- the
#                   only branch that needs no MPI installation and adds no
#                   -DMPI to the compile line (OpenMP-only build).
#   -Ddebug=0       starter/CMakeLists.txt:113-117 + 189-193 and the engine
#                   arch file only install the -O0/-O1/-O2 per-file overrides
#                   when debug STREQUAL "0".
#   -Dstatic_link=0 picks the non-static link line (-static-libgfortran &
#                   friends are the "1" branch).
#   -DCMAKE_BUILD_TYPE=Release  starter/CMakeLists.txt:233-237 forces RELEASE
#                   when unset anyway; passed explicitly so the log is honest.
#
# Architecture flags are NOT taken from here: each component includes its own
# CMake_Compilers/cmake_${arch}.txt (starter/CMakeLists.txt:136-149), which is
# authoritative and supplies every compile/link flag, including the extlib
# include and library paths.  tools/oracle/cmake_linux64_gf.txt (P0.2) is only
# the top-level declaration of the same toolchain.
#
# Usage:
#   OR_BUILD=...  OR_ROOT=...  tools/oracle/build_oracle.sh [-j N] [--clean]
#                                              [-build starter|engine|both]
#
set -euo pipefail

: "${OR_BUILD:?set OR_BUILD to the writable mirror prefix (see mirror_and_fetch.sh)}"
: "${OR_ROOT:?set OR_ROOT to the writable install prefix}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="linux64_gf"
BUILD_ROOT="${OR_BUILD}/build"

# cmake: prefer the upstream-era 3.28 over the conda 4.x, which rejects the
# `cmake_minimum_required(VERSION 3.15)` compatibility declarations
# (starter/CMakeLists.txt:5, engine/CMakeLists.txt:5, CMakeLists.txt:5).
CMAKE="${CMAKE:-/usr/bin/cmake}"
if [ ! -x "$CMAKE" ]; then
  CMAKE="$(command -v cmake)"
fi

# Compilers must be passed explicitly: the arch file's fallback
# (CMake_Compilers/cmake_linux64_gf.txt:4-10) sets CMAKE_<LANG>_COMPILER only
# when it is empty, but it is included AFTER enable_language()
# (starter/CMakeLists.txt:23-25), i.e. never early enough to matter.
FORTRAN_COMPILER="${FORTRAN_COMPILER:-$(command -v gfortran || true)}"
if [ -z "$FORTRAN_COMPILER" ]; then
  echo "!!! build_oracle.sh: no gfortran on PATH; install it or set FORTRAN_COMPILER" >&2
  exit 1
fi
C_COMPILER="${C_COMPILER:-$(command -v gcc || true)}"
CXX_COMPILER="${CXX_COMPILER:-$(command -v g++ || true)}"

JOBS=""
CLEAN=0
WHICH="both"
while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    -j*) JOBS="${1#-j}"; shift ;;
    -build) WHICH="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) sed -n '2,60p' "$0" | sed -n '/^# Usage/,$p'; exit 0 ;;
    *) echo "build_oracle.sh: unknown option '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$JOBS" ] || JOBS="$(nproc 2>/dev/null || echo 4)"

# "both" is a request for two components, not a component name.
case "$WHICH" in
  both)   COMPONENTS="starter engine" ;;
  starter|engine) COMPONENTS="$WHICH" ;;
  *) echo "build_oracle.sh: -build must be starter|engine|both" >&2; exit 2 ;;
esac

echo "cmake    : $CMAKE ($("$CMAKE" --version | head -1))"
echo "Fortran  : $FORTRAN_COMPILER ($("$FORTRAN_COMPILER" --version | head -1))"
echo "C / C++  : $C_COMPILER / $CXX_COMPILER"
echo "OR_BUILD : $OR_BUILD   (writable mirror; build root $BUILD_ROOT)"
echo "OR_ROOT  : $OR_ROOT"
echo "components: $WHICH   arch=$ARCH  -j $JOBS"
echo

# ------------------------------------------------------------------ guards --
if [ ! -f "$OR_BUILD/CMakeLists.txt" ]; then
  echo "!!! $OR_BUILD is not an OpenRadioss mirror (no CMakeLists.txt)." >&2
  echo "    Run tools/oracle/mirror_and_fetch.sh first." >&2
  exit 1
fi

if [ ! -d "$OR_BUILD/extlib/hm_reader" ]; then
  cat >&2 <<'EOF'
!!! extlib is missing.
    Both arch files hardcode -I${source_directory}/../extlib/{h3d/includes,
    md5/include,zlib/linux64/include} and link -lhm_reader_linux64 / lapack /
    metis (starter/CMake_Compilers/cmake_linux64_gf.txt:25,30-33,36-44;
    engine/CMake_Compilers/cmake_linux64_gf.txt:73-81), and the extlib custom
    target (starter/CMakeLists.txt:189-193) re-downloads it at every build.
    Without it nothing compiles and nothing links.
    Run tools/oracle/mirror_and_fetch.sh (network required).
EOF
  exit 1
fi

# The extlib gates (all required paths present AND API-current for the pinned
# source) live in mirror_and_fetch.sh; reuse them instead of duplicating them,
# so the build refuses a stale extlib in seconds instead of failing ten minutes
# into the C/C++ h3d sources or at the link step.
if [ "${OR_SKIP_EXTLIB_API_CHECK:-0}" != "1" ]; then
  echo "--- Gate: extlib completeness and API currency"
  OR_SRC="${OR_SRC:-$OR_BUILD}" "$SCRIPT_DIR/mirror_and_fetch.sh" --check-only || exit 1
fi

# ------------------------------------------------------ extlib short-circuit --
# starter/CMakeLists.txt:189-193 (engine/CMakeLists.txt:266-270) declares
#     add_custom_target(extlib ALL COMMAND ${PYTHON_EXEC} .../load_extlib.py)
# A custom target with ALL is rebuilt on EVERY build, so the build dies at 0%
# with "Download failed" even when a complete extlib is already on disk.  The
# only honest way out without network access is to make that command a no-op
# when -- and only when -- mirror_and_fetch.sh harvested the tree.
#
# The patch is therefore:
#   * applied here, reproducibly, on every run (idempotent),
#   * gated on the marker file extlib/P0_HARVESTED, which only the harvest
#     step writes, so an upstream-downloaded extlib is never short-circuited,
#   * limited to the MIRROR ($OR_BUILD) -- $OR_SRC is never touched,
#   * announced on stdout, and
#   * recorded in tools/validation_data/oracle_provenance.json.
LOAD_EXTLIB="$OR_BUILD/Compiling_tools/script/load_extlib.py"
if [ -f "$OR_BUILD/extlib/P0_HARVESTED" ]; then
  echo "--- Short-circuiting the extlib download target (pre-harvested extlib)"
  echo "    patch: $LOAD_EXTLIB (mirror copy only, see provenance json)"
  python3 - "$LOAD_EXTLIB" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
anchor = '   source_root =  os.path.dirname(os.path.abspath(__file__))+"/../.."'
patch = (
    anchor + "\n"
    "   # --- pyradioss P0.4 short-circuit -------------------------------------\n"
    "   # extlib was harvested by tools/oracle/mirror_and_fetch.sh from pinned\n"
    "   # sources (see tools/validation_data/oracle_provenance.json); the\n"
    "   # upstream download URL is unreachable from this machine.  Without this\n"
    "   # guard the `extlib` custom target re-downloads at every build and the\n"
    "   # build dies at 0%.  Gated on the marker file so a genuine upstream\n"
    "   # extlib is never skipped.\n"
    "   if os.path.isdir(source_root+\"/extlib/hm_reader\") and \\\n"
    "      os.path.isfile(source_root+\"/extlib/P0_HARVESTED\"):\n"
    "      print(\"extlib pre-harvested (tools/oracle/mirror_and_fetch.sh); \"\n"
    "            \"skipping download\")\n"
    "      exit(0)\n"
    "   # --- end pyradioss P0.4 short-circuit ---------------------------------"
)
if "pyradioss P0.4 short-circuit" in src:
    print("    already patched")
elif anchor not in src:
    sys.exit("load_extlib.py: anchor line not found; refusing to guess")
else:
    open(path, "w").write(src.replace(anchor, patch, 1))
    print("    patched")
PY
fi

if [ "$CLEAN" = "1" ] && [ -d "$BUILD_ROOT" ]; then
  echo "--- Removing $BUILD_ROOT"
  rm -rf "$BUILD_ROOT"
  # The POST_BUILD output dir must go too, otherwise a stale `starter` from an
  # older mirror would be picked up as a successful install.
  rm -rf "$OR_BUILD/exec"
fi

mkdir -p "$OR_ROOT/bin"

# ------------------------------------------------------------ configure+build
# Trap 2 + 3: one project per component, EXEC_NAME kept equal to the component
# name so the -Dbuild dereference trap is not triggered.
for component in $COMPONENTS; do
  bdir="$BUILD_ROOT/$component"
  echo "=================================================================="
  echo "--- $component : configure ($bdir)"
  echo "=================================================================="
  # The CUDA *.cu glob of engine/CMakeLists.txt:57 is emptied at lines 335-340
  # because gpu_cc is not defined, so no NVIDIA SDK is required.
  "$CMAKE" -S "$OR_BUILD" -B "$bdir" \
    -Dbuild="$component" \
    -Darch="$ARCH" \
    -Dstarter=starter \
    -Dengine=engine \
    -Dprecision=dp \
    -DMPI=smp \
    -Ddebug=0 \
    -Dstatic_link=0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_COMPILER="$FORTRAN_COMPILER" \
    -DCMAKE_C_COMPILER="$C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER"

  echo "--- $component : build"
  "$CMAKE" --build "$bdir" --parallel "$JOBS"
done

# -------------------------------------------------------------- install ----
# Trap 4: the POST_BUILD step put the link products in $OR_BUILD/exec under
# their target names; move them to $OR_ROOT/bin under the documented names.
echo
echo "--- Installing into $OR_ROOT/bin"
for component in $COMPONENTS; do
  src="$OR_BUILD/exec/$component"
  dst="$OR_ROOT/bin/${component}_${ARCH}"
  if [ ! -f "$src" ]; then
    echo "!!! expected $src to exist (POST_BUILD copy target)" >&2
    exit 1
  fi
  cp -f "$src" "$dst"
  chmod +x "$dst"
  echo "    $src -> $dst"
done

# ---------------------------------------------------------------- smoke ----
# INSTALL.md:110-111 documents the invocation; execargcheck.F PREXECINFO
# (starter :1150-1163, engine :648-692) prints the banner and MY_EXIT(0)s.
# The starter's -v path calls HM_BUILD_ID from libhm_reader, so this also
# proves LD_LIBRARY_PATH/RAD_H3D_PATH are right (INSTALL.md:38-42).
echo
echo "--- Smoke check: version banner"
export OPENRADIOSS_PATH="$OR_BUILD"
export RAD_CFG_PATH="$OR_BUILD/hm_cfg_files"
export RAD_H3D_PATH="$OR_BUILD/extlib/h3d/lib/linux64"
export LD_LIBRARY_PATH="$OR_BUILD/extlib/hm_reader/linux64/:${LD_LIBRARY_PATH:-}"
export OMP_STACKSIZE=400m
rc=0
for component in $COMPONENTS; do
  bin="$OR_ROOT/bin/${component}_${ARCH}"
  echo "--- $bin -v"
  "$bin" -v | head -8 || rc=1
done
[ "$rc" = 0 ] || { echo "!!! smoke check failed" >&2; exit 1; }

echo
echo "--- Done."
echo "    OR_STARTER=$OR_ROOT/bin/starter_${ARCH}"
echo "    OR_ENGINE =$OR_ROOT/bin/engine_${ARCH}"
echo "    runtime env: source $SCRIPT_DIR/oracle_env.sh"