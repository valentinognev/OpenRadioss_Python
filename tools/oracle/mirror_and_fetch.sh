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

# load_extlib.py swallows the underlying network error and only prints
# "Download failed".  Trap ERR so the real cause is visible, and report which
# half of the job succeeded.
trap 'rc=$?; echo; echo "!!! mirror_and_fetch.sh FAILED (exit $rc)"; if [ -d "$OR_BUILD/extlib/hm_reader" ]; then echo "    mirror OK, extlib OK"; else echo "    mirror OK at $OR_BUILD, but extlib was NOT fetched"; echo "    -> the build cannot link or run until extlib/hm_reader exists"; fi; exit $rc' ERR

echo "OR_SRC   = $OR_SRC   (read-only, never written)"
echo "OR_BUILD = $OR_BUILD (mirror)"
echo "OR_ROOT  = $OR_ROOT  (install prefix)"
echo

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
# Must be python3: the component CMakeLists hardcode python3 on non-Windows.
# cwd is irrelevant -- load_extlib.py derives source_root from its own path,
# so it unzips into the mirror and never touches $OR_SRC.
echo "--- Fetching extlib (network required)"
python3 "$OR_BUILD/Compiling_tools/script/load_extlib.py"

echo
echo "--- Mirror ready"
echo "    extlib     : $( [ -d "$OR_BUILD/extlib" ] && echo present || echo MISSING )"
echo "    hm_reader  : $( [ -d "$OR_BUILD/extlib/hm_reader" ] && echo present || echo MISSING )"
echo "    exec prefix: $OR_BUILD/exec"
echo "    install to : $OR_ROOT/bin"

# The script leaves the downloaded archive behind on purpose; drop it so a
# re-run does not carry a duplicate of the whole extlib tree.
rm -f "$OR_BUILD/extlib.zip"