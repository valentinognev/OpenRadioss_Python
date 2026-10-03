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
# Upstream declares `cmake_minimum_required (VERSION 3.15)`
# (CMakeLists.txt:5, starter/CMakeLists.txt:5, engine/CMakeLists.txt:5).  CMake 4
# removed the compatibility for < 3.10-era declarations and refuses such a
# project outright, so the first cmake on PATH is NOT a safe fallback here (on
# this box `command -v cmake` is conda cmake 4.4.3, which fails the configure
# with an upstream-looking error).  Require 3.15 <= version < 4 explicitly.
CMAKE="${CMAKE:-/usr/bin/cmake}"
if [ ! -x "$CMAKE" ]; then
  CMAKE="$(command -v cmake || true)"
fi
[ -n "$CMAKE" ] && [ -x "$CMAKE" ] || {
  echo "!!! build_oracle.sh: no cmake found; set CMAKE=/path/to/cmake" >&2
  exit 1
}
cmake_version="$("$CMAKE" --version | head -1 | sed -E 's/.*version ([0-9]+\.[0-9]+(\.[0-9]+)?).*/\1/')"
cmake_major="${cmake_version%%.*}"
cmake_rest="${cmake_version#*.}"
cmake_minor="${cmake_rest%%.*}"
if [ -z "$cmake_version" ] || [ "$cmake_major" -lt 3 ] \
   || { [ "$cmake_major" -eq 3 ] && [ "${cmake_minor:-0}" -lt 15 ]; } \
   || [ "$cmake_major" -ge 4 ]; then
  echo "!!! build_oracle.sh: cmake >= 3.15 and < 4 is required (upstream uses" >&2
  echo "    cmake_minimum_required (VERSION 3.15), which CMake 4 rejects);" >&2
  echo "    found $CMAKE -> $cmake_version" >&2
  exit 1
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

# A failed rebuild must not leave last week's binary behind for the gate (and for
# any parity run) to find: drop both before configuring.
for component in starter engine; do
  stale="$OR_ROOT/bin/${component}_${ARCH}"
  if [ -f "$stale" ]; then
    echo "--- Removing stale $stale (it will be reinstalled only if this build succeeds)"
    rm -f "$stale"
  fi
done

# ------------------------------------------------------------- extlib shims --
# The reachable extlib (v59) predates the pinned source's reader and h3d API.
# tools/oracle/extlib_shims/ holds SOURCE adapters for the missing surface; see
# each file's header comment for the upstream citations and the exact contract
# each caller relies on.  Nothing in the harvested tree is modified and no
# harvested binary is touched: the shims are copied into $OR_BUILD/p0_shims
# (idempotent, so the build is reproducible from a fresh mirror) and injected
# without editing a single upstream file:
#
#   * the h3d header surface is a compile-time gap, so it is injected as
#     `-include <p0_h3d_api_shim.h> -I<p0_shims>` on the C and CXX lines.  The
#     -I is what lets `#include <h3dpublic_import.h>`
#     (starter/source/output/checksum/checksum_list.cpp:27) find our stand-in;
#     it is only added when the harvested extlib has no file of that name, so it
#     can never shadow the real header;
#   * the three missing reader symbols need to be LINKED.  They go in through
#     `-Dflexpipe_lib=<archive>`, an unused hook in
#     starter/CMake_Compilers/cmake_linux64_gf.txt:177
#     (`set (LINK "dl ${flexpipe_lib} ...")`), so upstream's CMake is untouched.
SHIM_SRC="$SCRIPT_DIR/extlib_shims"
SHIM_DIR="$OR_BUILD/p0_shims"
SHIM_LIB=""
mkdir -p "$SHIM_DIR"
cp "$SHIM_SRC/p0_h3d_api_shim.h" "$SHIM_SRC/h3dpublic_import.h" \
   "$SHIM_SRC/h3dpublic_export.h" "$SHIM_SRC/p0_hm_reader_adapter.c" "$SHIM_DIR/"
echo "--- extlib shims installed in $SHIM_DIR"

# -I<shim dir> must come BEFORE the extlib include dir so that the shim
# headers win; it does, because CMAKE_C_FLAGS/CMAKE_CXX_FLAGS precede the
# per-source COMPILE_FLAGS that carry -I${source_directory}/../extlib/h3d/includes
# (engine/CMake_Compilers/cmake_linux64_gf.txt:73,73-77 and
#  starter/CMake_Compilers/cmake_linux64_gf.txt:36).  The two shim headers that
# must shadow a real one are only put on the path when the harvested extlib has
# no file of that name, so a current extlib is never shadowed.
shim_flags="-include $SHIM_DIR/p0_h3d_api_shim.h -I$SHIM_DIR"
h3d_inc="$OR_BUILD/extlib/h3d/includes"

# h3dpublic_import.h: the shim stands in only when there is no real header.
if [ -f "$h3d_inc/h3dpublic_import.h" ]; then
  echo "    NOTE: harvested extlib HAS h3dpublic_import.h -- shim copy dropped"
  rm -f "$SHIM_DIR/h3dpublic_import.h"
else
  echo "    injecting h3dpublic_import.h shim (extlib has none)"
fi

# h3dpublic_export.h: the real header exists but may be too old.  It is current
# only if it already mentions H3D_NF_FORMAT (the parameter the pinned source
# passes); otherwise the shim replaces exactly the two prototypes.
if [ -f "$h3d_inc/h3dpublic_export.h" ] && grep -q "H3D_NF_FORMAT" "$h3d_inc/h3dpublic_export.h"; then
  echo "    NOTE: harvested h3dpublic_export.h already declares H3D_NF_FORMAT"
  echo "          -- shim copy dropped, the real header is used"
  rm -f "$SHIM_DIR/h3dpublic_export.h"
else
  echo "    injecting h3dpublic_export.h shim (2 prototypes lack H3D_NF_FORMAT)"
fi

# The adapters define symbols the reachable libraries do not export.  Link them
# ONLY when that is actually true, so a current extlib never gets them
# interposed over the real implementations.
HM="$OR_BUILD/extlib/hm_reader/linux64/libhm_reader_linux64.so"
need_shim_lib=0
for s in cpp_get_include_file_by_index cpp_sale_mesh_create_ \
         cpp_is_part_with_elements_; do
  nm -D --defined-only "$HM" 2>/dev/null | grep -q "[[:space:]]$s\$" || need_shim_lib=1
done

# Only the hm_reader adapter is compiled.  There is deliberately no adapter for
# Hyper3DExportLibraryVersion / Hyper3DCompressionLevel: the pinned source
# defines both itself (common_source/output/h3d/h3d_build_cpp/h3d_dl.c:984-1000),
# so an archive member defining them was never pulled in -- a "safety net" that
# was dead code.  What refuses h3d output is oracle_env.sh, which does not put
# the ABI-incompatible writer on RAD_H3D_PATH (h3d_dl.c:920-921 -> *IERROR=1 ->
# genh3d.F:729-731 ARRET(2)).
if [ "$need_shim_lib" = "1" ]; then
  echo "    compiling the extlib shim archive (missing hm_reader entry points)"
  rm -f "$SHIM_LIB"
  "$C_COMPILER" -c -O2 -fPIC -Wall -Wextra -o "$SHIM_DIR/p0_hm_reader_adapter.o" \
                "$SHIM_DIR/p0_hm_reader_adapter.c"
  ar rcs "$SHIM_DIR/libp0extlibshims.a" "$SHIM_DIR/p0_hm_reader_adapter.o"
  SHIM_LIB="$SHIM_DIR/libp0extlibshims.a"
  echo "    -> $SHIM_LIB"
else
  echo "    harvested hm_reader exports every entry point: no shim archive linked"
fi

# ---------------------------------------------------------------- libcrypt ---
# The harvested libapr-1.so has a real NEEDED entry on libcrypt.so.1
# (`objdump -p extlib/hm_reader/linux64/libapr-1.so.0 | grep NEEDED`).  ld only
# warns about it ("needed by .../libapr-1.so, not found (try using -rpath or
# -rpath-link)") because it does not search the default library directories for
# the dependencies of a shared object it found through -L.  The runtime loader
# does search them and does find it, so this is not merely cosmetic: give ld the
# same information instead of muting the warning, and prove the resolution below
# with ldd.
CRYPT_DIR=""
for d in /lib/x86_64-linux-gnu /lib64 /usr/lib/x86_64-linux-gnu /lib /usr/lib; do
  if [ -e "$d/libcrypt.so.1" ]; then CRYPT_DIR="$d"; break; fi
done
LINK_EXTRA="-Wl,-rpath-link,$OR_BUILD/extlib/hm_reader/linux64"
[ -n "$CRYPT_DIR" ] && LINK_EXTRA="$LINK_EXTRA -Wl,-rpath-link,$CRYPT_DIR"
echo "--- libcrypt.so.1 found in '${CRYPT_DIR:-<none>}' -> $LINK_EXTRA"

# ------------------------------------------------------------ configure+build
# Trap 2 + 3: one project per component, EXEC_NAME kept equal to the component
# name so the -Dbuild dereference trap is not triggered.
for component in $COMPONENTS; do
  bdir="$BUILD_ROOT/$component"
  echo "=================================================================="
  echo "--- $component : configure ($bdir)"
  echo "=================================================================="
  # ${flexpipe_lib} is an unused hook in the arch file's link line
  # (starter/CMake_Compilers/cmake_linux64_gf.txt:177), so it is where both the
  # shim archive and the -rpath-link fragments go; the engine arch file has no
  # such hook and needs neither.  The whole value must be ONE quoted argument:
  # `-Dflexpipe_lib=a b` unquoted makes cmake eat `b` as an unknown option.
  flexpipe=()
  if [ "$component" = "starter" ]; then
    flexpipe=("-Dflexpipe_lib=$SHIM_LIB $LINK_EXTRA")
  fi
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
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
    -DCMAKE_C_FLAGS="$shim_flags" \
    -DCMAKE_CXX_FLAGS="$shim_flags" \
    "${flexpipe[@]+"${flexpipe[@]}"}"

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

# ------------------------------------------------- runtime library check ----
# The install is only usable if every NEEDED library resolves under the oracle
# environment; check it instead of trusting the link.
echo
echo "--- Checking runtime library resolution"
rc=0
for component in $COMPONENTS; do
  bin="$OR_ROOT/bin/${component}_${ARCH}"
  missing="$(ldd "$bin" 2>&1 | grep 'not found' || true)"
  if [ -n "$missing" ]; then
    echo "!!! $bin has unresolved libraries:" >&2
    echo "$missing" | sed 's/^/    /' >&2
    rc=1
  else
    echo "    $bin: all libraries resolved"
    echo "        hm_reader: $(ldd "$bin" | grep hm_reader | sed 's/^ *//')"
  fi
done
[ "$rc" = 0 ] || { echo "!!! runtime library check failed" >&2; exit 1; }

echo
echo "--- Done."
echo "    OR_STARTER=$OR_ROOT/bin/starter_${ARCH}"
echo "    OR_ENGINE =$OR_ROOT/bin/engine_${ARCH}"
echo "    runtime env: source $SCRIPT_DIR/oracle_env.sh"
