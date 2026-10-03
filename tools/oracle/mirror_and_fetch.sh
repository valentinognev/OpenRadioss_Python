#!/usr/bin/env bash
# Mirror the READ-ONLY OpenRadioss source tree into a writable prefix and
# fetch extlib there.
#
# WHY A MIRROR IS MANDATORY (not a convenience):
#
#   1. Compiling_tools/script/load_extlib.py computes
#          source_root = dirname(abspath(__file__)) + "/../.."
#      downloads extlib.zip there and extracts it into <source_root>/extlib,
#      leaving extlib.zip behind.  It ALWAYS writes into the source tree.
#   2. starter/CMakeLists.txt:247 and engine/CMakeLists.txt:352 carry a
#      POST_BUILD rule that creates ${source_directory}/../exec and copies
#      the linked executable into it.  That also writes into the source tree.
#
# $OR_SRC is therefore never written to; every write lands in $OR_BUILD.
#
# `git archive` is used instead of `cp -r` because it:
#   - omits .git entirely (no giant object store in the mirror),
#   - honours .gitattributes export-ignore/export-subst rules,
#   - reproduces exactly the tracked tree at HEAD, and
#   - is idempotent, so re-running refreshes the mirror in place.
#
# Usage:
#   OR_SRC=...  OR_BUILD=...  OR_ROOT=...  tools/oracle/mirror_and_fetch.sh
#
set -euo pipefail

: "${OR_SRC:?set OR_SRC to the READ-ONLY OpenRadioss source tree}"
: "${OR_BUILD:?set OR_BUILD to the writable mirror prefix (build happens here)}"
: "${OR_ROOT:?set OR_ROOT to the writable install prefix (binaries land here)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --check-only runs the two verification gates (extlib paths, extlib API
# currency) and stops.  build_oracle.sh calls it so the gates live in exactly
# one place and the build refuses a stale extlib in seconds, not after ten
# minutes of compiling.
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    -h|--help) sed -n '/^# Usage/,$p' "$0" | head -8; exit 0 ;;
    *) echo "usage: mirror_and_fetch.sh [--check-only]" >&2; exit 2 ;;
  esac
done

# ============================================================================
# PINNED EXTLIB SOURCES  (see tools/validation_data/oracle_provenance.json)
# ============================================================================
# WHY NOT Compiling_tools/script/load_extlib.py:
#
#   EXTLIB_VERSION.json in the source tree still points at
#     https://github.com/OpenRadioss/OpenRadioss_extlib/releases/download/v82/extlib.zip
#   which is a pre-move pointer to the old `OpenRadioss` organisation.  From
#   this machine the whole organisation is unreachable -- every repo under it
#   answers 404, including the main one, while github.com itself answers 200 --
#   and GitHub *release-asset* downloads are blocked generally (a known-good
#   asset, psf/numpy v2.4.6, also 404s).  So the canonical download cannot
#   succeed by any URL, from any org, and there is no mirror of the asset.
#
#   load_extlib.py is therefore NOT run by default.  Set OR_USE_UPSTREAM_EXTLIB=1
#   to use it anyway once the network is fixed; it is idempotent and writes only
#   into the mirror.
#
# WHAT IS HARVESTED INSTEAD, both pinned:
#
#   1. The extlib repository itself -- h3d, lapack-3.10.0, metis, zlib, md5 and
#      all their prebuilt linux64 libraries and headers ARE tracked in that
#      repository's git tree.  It is cloned at a fixed commit below.  This is a
#      fork/mirror of OpenRadioss/OpenRadioss_extlib, because the original
#      organisation is unreachable; its own EXTLIB_VERSION.json says version 59,
#      i.e. these libraries predate the v82 pointer but are the same upstream
#      artefacts (reference LAPACK 3.10.0, upstream zlib/metis/md5 builds).
#
#   2. hm_reader + apr-1 -- these two exist ONLY as release assets of the extlib
#      repository ("the hm_reader library is added in the assets during its
#      creation", extlib/README.md); no git mirror of them exists anywhere.  They
#      are copied out of a pinned public Docker Hub image that carries a full
#      OpenRadioss installation, by `docker create` + `docker cp` (the container
#      is never started and is removed again).
#
# Neither source substitutes a library with something unrelated: every file below
# is the genuine upstream artifact, and the verification step re-checks all of
# them against the paths the two cmake arch files actually ask for.
# ============================================================================
EXTLIB_GIT_URI="https://github.com/xupeiwust/OpenRadioss_extlib.git"
EXTLIB_GIT_COMMIT="145c7b61031266d7d69543cca6133bd4dd9e32b7"   # extlib version 59
EXTLIB_HM_IMAGE="ale10tech/openradioss-core:ubuntu24.04"
EXTLIB_HM_IMAGE_DIGEST="sha256:15c2c76f6380f3650cfd629ea5c7fdb75a5f9e7c2c3f2f4da73114f0974c97a5"
EXTLIB_HM_CONTAINER="or_extlib_harvest_$$"

# Source adapters for the API the reachable (older) extlib does not have.
SHIM_SRC="$SCRIPT_DIR/extlib_shims"

# The oracle is only meaningful for the exact source tree it was reviewed
# against: every path:line citation in tools/validation_data/
# oracle_provenance.json and all five shim headers point into that commit.
# Override with OR_ALLOW_ANY_OR_SRC=1 to build a different one on purpose.
EXPECTED_OR_SRC_SHA="1a0d691b82bd4f94a3bc2399682551e84e1d5fa4"

# Local-extlib escape hatch, for boxes that cannot reach Docker Hub or the extlib
# git mirror: point OR_EXTLIB_LOCAL at a directory containing an `extlib/`
# (or at the extlib/ directory itself) and it is copied in instead of harvested.
OR_EXTLIB_LOCAL="${OR_EXTLIB_LOCAL:-}"

# The exact set the build needs, read off
#   starter/CMake_Compilers/cmake_linux64_gf.txt:25,30-33,36-44
#   engine/CMake_Compilers/cmake_linux64_gf.txt:73-81
REQUIRED_EXTLIB_PATHS=(
  "hm_reader/linux64/libhm_reader_linux64.so"
  "hm_reader/linux64/libapr-1.so"
  "hm_reader/linux64/libapr-1.so.0"
  "h3d/includes/h3dpublic_defs.h"
  "h3d/lib/linux64/libh3dwriter.so"
  "lapack-3.10.0/lib_linux64_gf/liblapack.a"
  "lapack-3.10.0/lib_linux64_gf/librefblas.a"
  "lapack-3.10.0/lib_linux64_gf/libtmglib.a"
  "metis/linux64/libmetis_linux64_gcc.a"
  "zlib/linux64/include/zlib.h"
  "zlib/linux64/lib/libz.a"
  "md5/include/md5.h"
  "md5/linux64/libmd5.a"
  "license/hm_reader_license.txt"
)

# load_extlib.py swallows the underlying network error and only prints
# "Download failed".  Trap ERR so the real cause is visible, and report which
# half of the job succeeded.
trap 'rc=$?; echo; echo "!!! mirror_and_fetch.sh FAILED (exit $rc)"; if [ -d "$OR_BUILD/extlib/hm_reader" ]; then echo "    mirror OK, extlib OK"; else echo "    mirror OK at $OR_BUILD, but extlib was NOT fetched"; echo "    -> the build cannot link or run until extlib/hm_reader exists"; fi; exit $rc' ERR

echo "OR_SRC   = $OR_SRC   (read-only, never written)"
echo "OR_BUILD = $OR_BUILD (mirror)"
echo "OR_ROOT  = $OR_ROOT  (install prefix)"
echo

# ------------------------------------------------------- source revision ----
# Mirroring HEAD of whatever $OR_SRC happens to be would silently invalidate
# every citation in the provenance file, so check it.
OR_SRC_SHA="$(git -C "$OR_SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "OR_SRC HEAD = $OR_SRC_SHA (expected $EXPECTED_OR_SRC_SHA)"
if [ "$OR_SRC_SHA" != "$EXPECTED_OR_SRC_SHA" ] && [ "${OR_ALLOW_ANY_OR_SRC:-0}" != "1" ]; then
  cat >&2 <<EOF
!!! $OR_SRC is at $OR_SRC_SHA but this oracle was reviewed against
    $EXPECTED_OR_SRC_SHA.  Every path:line citation in
    tools/validation_data/oracle_provenance.json and every shim header refers
    to that commit.  Re-review them, or set OR_ALLOW_ANY_OR_SRC=1 to build this
    tree on purpose.
EOF
  exit 1
fi

# --------------------------------------------------------------- gates ------
# Gate 1: every path the two component arch flag files ask for must exist
# (starter/CMake_Compilers/cmake_linux64_gf.txt:25,30-33,36-44 and
#  engine/CMake_Compilers/cmake_linux64_gf.txt:73-81).  A missing one is a hard
# error, not a warning, because it surfaces much later as a link failure.
verify_extlib() {
  local missing=0 p
  for p in "${REQUIRED_EXTLIB_PATHS[@]}"; do
    if [ ! -e "$OR_BUILD/extlib/$p" ]; then
      echo "    MISSING extlib/$p" >&2
      missing=1
    fi
  done
  return "$missing"
}

# Gate 2: a *file-complete* extlib is not necessarily a *version-current* one,
# and the difference only shows up as a compile error ten minutes into a build.
# The missing API facts the pinned source depends on:
#   H3D_NF_FORMAT in h3dpublic_defs.h   (common_source/output/h3d/h3d_build_cpp/
#       h3d_dl.c:258-281,1370-1416 and 77 call sites in
#       engine/source/output/h3d/h3d_build_cpp/*.cpp)
#   h3dpublic_import.h                   (starter/source/output/checksum/
#       checksum_list.cpp:27)
#   the cpp_* entry points libhm_reader_linux64.so must export for
#       starter/source/devtools/hm_reader/*.F90
#
# A stale tree is only tolerated when EVERY missing fact has a declared source
# adapter in tools/oracle/extlib_shims/ -- see check_shims_cover_stale below.
STALE_ITEMS=()
check_extlib_api() {
  STALE_ITEMS=()
  local inc="$OR_BUILD/extlib/h3d/includes"
  local hm="$OR_BUILD/extlib/hm_reader/linux64/libhm_reader_linux64.so"
  local s
  if ! grep -q "H3D_NF_FORMAT" "$inc/h3dpublic_defs.h" 2>/dev/null; then
    echo "    STALE h3d: h3dpublic_defs.h does not define H3D_NF_FORMAT" >&2
    STALE_ITEMS+=("H3D_NF_FORMAT")
  fi
  if [ ! -f "$inc/h3dpublic_import.h" ]; then
    echo "    STALE h3d: h3dpublic_import.h is missing" >&2
    STALE_ITEMS+=("h3dpublic_import.h")
  fi
  if command -v nm >/dev/null 2>&1; then
    for s in cpp_get_include_file_by_index cpp_sale_mesh_create_ \
             cpp_is_part_with_elements_; do
      if ! nm -D --defined-only "$hm" 2>/dev/null | grep -q "[[:space:]]$s\$"; then
        echo "    STALE hm_reader: libhm_reader_linux64.so does not export $s" >&2
        STALE_ITEMS+=("$s")
      fi
    done
  else
    echo "    (nm not available: skipping the hm_reader symbol check)"
  fi
  [ "${#STALE_ITEMS[@]}" -eq 0 ]
}

# Every stale item must be named by an adapter, otherwise the build would either
# fail ten minutes in or -- worse -- silently do something other than what the
# adapter documents.
check_shims_cover_stale() {
  local s provider ok=0
  echo "    each missing entry point needs a declared adapter in extlib_shims/:" >&2
  for s in "${STALE_ITEMS[@]}"; do
    provider="$(grep -rl -- "$s" "$SHIM_SRC" 2>/dev/null | tr '\n' ' ')"
    if [ -n "$provider" ]; then
      echo "      adapter  $s  <-  $provider" >&2
    else
      echo "      NO ADAPTER  $s" >&2
      ok=1
    fi
  done
  return "$ok"
}

report_stale() {
  echo "!!! extlib is STALE: it predates the pinned OpenRadioss source." >&2
  echo "    $OR_SRC/EXTLIB_VERSION.json asks for version" \
       "$(sed -n 's/.*"version"[^0-9]*\([0-9]*\).*/\1/p' "$OR_BUILD/EXTLIB_VERSION.json" 2>/dev/null || echo '?')," >&2
  echo "    the libraries on disk are older.  The build therefore runs with the" >&2
  echo "    source adapters in tools/oracle/extlib_shims/; every one of them is" >&2
  echo "    declared in tools/validation_data/oracle_provenance.json with the" >&2
  echo "    output paths it makes untrustworthy." >&2
}

if [ "$CHECK_ONLY" = "1" ]; then
  echo "--- Verifying $OR_BUILD/extlib against the cmake arch files"
  verify_extlib || { echo "!!! extlib is incomplete (see above)" >&2; exit 1; }
  echo "--- Checking that the extlib is API-current for the pinned source"
  if ! check_extlib_api; then
    report_stale
    check_shims_cover_stale || exit 1
    echo "--- extlib OK (stale, but fully covered by declared adapters)"
    exit 0
  fi
  echo "--- extlib OK"
  exit 0
fi

# ---------------------------------------------------------------- mirror ----
mkdir -p "$OR_BUILD"
mkdir -p "$OR_ROOT/bin"

# Remove stale mirror content that git archive will not overwrite cleanly
# (e.g. extlib.zip left by a previous run, or files dropped upstream).
rm -rf "$OR_BUILD/.git"

echo "--- Mirroring tracked tree at HEAD into $OR_BUILD"
git -C "$OR_SRC" archive --format=tar HEAD | tar -x -C "$OR_BUILD"

# ------------------------------------------------------- arch flags file ----
# Upstream has no top-level CMake_Compilers/cmake_linux64_gf.txt (only the
# per-component ones under starter/ and engine/).  Ship ours so the mirror has
# a top-level arch definition for the linux64 gfortran toolchain.
echo "--- Installing top-level arch flags (linux64_gf)"
mkdir -p "$OR_BUILD/CMake_Compilers"
cp "$SCRIPT_DIR/cmake_linux64_gf.txt" "$OR_BUILD/CMake_Compilers/cmake_linux64_gf.txt"

# --------------------------------------------------------------- extlib -----
if [ -n "$OR_EXTLIB_LOCAL" ]; then
  # Maintainer-supplied extlib: no network, no docker.
  src_extlib="$OR_EXTLIB_LOCAL"
  [ -d "$src_extlib/extlib" ] && src_extlib="$src_extlib/extlib"
  [ -d "$src_extlib" ] || {
    echo "!!! OR_EXTLIB_LOCAL=$OR_EXTLIB_LOCAL holds no extlib/ directory" >&2
    exit 1
  }
  echo "--- Installing extlib from OR_EXTLIB_LOCAL=$src_extlib"
  rm -rf "$OR_BUILD/extlib"
  cp -a "$src_extlib" "$OR_BUILD/extlib"
  printf 'supplied locally via OR_EXTLIB_LOCAL=%s
' "$OR_EXTLIB_LOCAL" \
    > "$OR_BUILD/extlib/UPSTREAM_EXTLIB_VERSION.json"
  printf 'supplied locally via OR_EXTLIB_LOCAL=%s
see tools/validation_data/oracle_provenance.json
' \
    "$OR_EXTLIB_LOCAL" > "$OR_BUILD/extlib/P0_HARVESTED"
elif [ "${OR_USE_UPSTREAM_EXTLIB:-0}" = "1" ]; then
  echo "--- Fetching extlib with the upstream script (OR_USE_UPSTREAM_EXTLIB=1)"
  # Must be python3: the component CMakeLists hardcode python3 on non-Windows.
  # cwd is irrelevant -- load_extlib.py derives source_root from its own path, so
  # it unzips into the mirror and never touches $OR_SRC.
  python3 "$OR_BUILD/Compiling_tools/script/load_extlib.py"
  # The script leaves the downloaded archive behind on purpose; drop it so a
  # re-run does not carry a duplicate of the whole extlib tree.
  rm -f "$OR_BUILD/extlib.zip"
elif verify_extlib 2>/dev/null; then
  echo "--- extlib already complete in $OR_BUILD/extlib (nothing to harvest)"
else
  echo "--- Acquiring extlib from the pinned sources"

  # (1) the extlib repository: h3d, lapack-3.10.0, metis, zlib, md5, licenses
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/or_extlib.XXXXXX")"
  echo "    clone $EXTLIB_GIT_URI @ ${EXTLIB_GIT_COMMIT:0:12}"
  git clone --quiet "$EXTLIB_GIT_URI" "$tmp/repo"
  git -C "$tmp/repo" checkout --quiet "$EXTLIB_GIT_COMMIT"
  rm -rf "$OR_BUILD/extlib"
  cp -a "$tmp/repo/extlib" "$OR_BUILD/extlib"
  # Keep the upstream version marker next to the tree: it is what makes the
  # difference from EXTLIB_VERSION.json's v82 auditable instead of silent.
  cp "$tmp/repo/EXTLIB_VERSION.json" "$OR_BUILD/extlib/UPSTREAM_EXTLIB_VERSION.json"
  rm -rf "$tmp"

  # (2) hm_reader + apr-1: release-asset-only upstream, harvested from a pinned
  #     public image.  `docker create` never starts the container, and it is
  #     removed again immediately after the copy.
  echo "    pull  $EXTLIB_HM_IMAGE"
  docker pull --quiet "$EXTLIB_HM_IMAGE" >/dev/null
  digest="$(docker image inspect "$EXTLIB_HM_IMAGE" --format '{{index .RepoDigests 0}}')"
  # RepoDigests are reported as <repo>@<digest>, without the tag.
  want="${EXTLIB_HM_IMAGE%:*}@${EXTLIB_HM_IMAGE_DIGEST}"
  if [ "$digest" != "$want" ]; then
    echo "!!! image digest mismatch" >&2
    echo "    got      $digest" >&2
    echo "    expected $want" >&2
    exit 1
  fi
  echo "    create (not run) throw-away container $EXTLIB_HM_CONTAINER"
  docker create --name "$EXTLIB_HM_CONTAINER" "$EXTLIB_HM_IMAGE" >/dev/null
  # shellcheck disable=SC2064
  trap "docker rm -f '$EXTLIB_HM_CONTAINER' >/dev/null 2>&1 || true" EXIT
  rm -rf "$OR_BUILD/extlib/hm_reader"
  mkdir -p "$OR_BUILD/extlib/hm_reader"
  docker cp "$EXTLIB_HM_CONTAINER:/opt/OpenRadioss/extlib/hm_reader/linux64" \
            "$OR_BUILD/extlib/hm_reader/"
  docker rm -f "$EXTLIB_HM_CONTAINER" >/dev/null
  trap - EXIT
fi

echo "--- Verifying the harvested extlib against the cmake arch files"
if ! verify_extlib; then
  echo "!!! extlib is incomplete; the build cannot link.  See the list above." >&2
  exit 1
fi

echo "--- Checking that the extlib is API-current for the pinned source"
if check_extlib_api; then
  echo "    h3d and hm_reader expose the APIs the pinned source needs"
elif [ "${OR_SKIP_EXTLIB_API_CHECK:-0}" = "1" ]; then
  echo "!!! extlib is STALE (see above) -- continuing anyway because" >&2
  echo "    OR_SKIP_EXTLIB_API_CHECK=1; no adapter coverage is checked." >&2
else
  report_stale
  check_shims_cover_stale || {
    echo "    Obtain the extlib version named above, or teach" >&2
    echo "    tools/oracle/extlib_shims/ to cover the rest." >&2
    exit 1
  }
fi

# Marker read by build_oracle.sh before it short-circuits the upstream `extlib`
# download target (starter/CMakeLists.txt:189-193, engine/CMakeLists.txt:266-270),
# which re-downloads on every single build.  Its presence means "these files were
# NOT fetched from the URL in $OR_SRC/EXTLIB_VERSION.json".  Written whenever
# the tree carries the harvest's own version marker, so it survives re-runs.
if [ -f "$OR_BUILD/extlib/UPSTREAM_EXTLIB_VERSION.json" ]; then
  printf 'harvested by tools/oracle/mirror_and_fetch.sh\nsee tools/validation_data/oracle_provenance.json\n' \
    > "$OR_BUILD/extlib/P0_HARVESTED"
fi

echo
echo "--- Mirror ready"
echo "    OR_SRC sha : $OR_SRC_SHA"
echo "    extlib     : $OR_BUILD/extlib ($(du -sh "$OR_BUILD/extlib" | cut -f1))"
echo "    extlib ver : $( [ -f "$OR_BUILD/extlib/UPSTREAM_EXTLIB_VERSION.json" ] \
                     && tr -d ' \n' < "$OR_BUILD/extlib/UPSTREAM_EXTLIB_VERSION.json" \
                     || echo 'from upstream load_extlib.py' )"
echo "    needed by  : version $(sed -n 's/.*"version"[^0-9]*\([0-9]*\).*/\1/p' "$OR_BUILD/EXTLIB_VERSION.json" 2>/dev/null || echo '?') of \$OR_SRC/EXTLIB_VERSION.json"
echo "    exec prefix: $OR_BUILD/exec"
echo "    install to : $OR_ROOT/bin"
echo "    next       : tools/oracle/build_oracle.sh"
