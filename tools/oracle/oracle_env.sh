# Runtime environment for the reference (Fortran) OpenRadioss binaries built
# from the writable mirror.  Source it, do not execute it:
#
#   source tools/oracle/oracle_env.sh
#
# RAD_H3D_PATH and LD_LIBRARY_PATH are REQUIRED, not cosmetic.  The binaries
# start without them and then die on the first H3D / message call with an
# unresolved libh3d or libhm_reader.  Both come straight from
# $OR_SRC/INSTALL.md:34-42 ("Environment variables settings under Linux"),
# retargeted from OPENRADIOSS_PATH to the mirror prefix $OR_BUILD.
#
# Expects the mirror to already exist (see tools/oracle/mirror_and_fetch.sh);
# source order matters, so this file derives $OR_BUILD from $OR_ROOT's parent
# only as a fallback.

: "${OR_BUILD:?source oracle_env.sh after exporting OR_BUILD (see mirror_and_fetch.sh)}"
: "${OR_ROOT:?set OR_ROOT to the writable install prefix}"

export OR_BUILD
export OR_ROOT

# OPENRADIOSS_PATH is the upstream name for the prefix; some helper scripts and
# the hm_cfg_files lookup still key off it.
export OPENRADIOSS_PATH="$OR_BUILD"
export RAD_CFG_PATH="$OR_BUILD/hm_cfg_files"

# H3D reader + message library lookup at runtime.
export RAD_H3D_PATH="$OR_BUILD/extlib/h3d/lib/linux64"
export LD_LIBRARY_PATH="$OR_BUILD/extlib/hm_reader/linux64/:$LD_LIBRARY_PATH"

# OpenRadioss stacks aggressively; the Intel/OpenMP runtimes need headroom.
export OMP_STACKSIZE=400m

# Put the built starter/engine at the front of PATH.
export PATH="$OR_ROOT/bin:$PATH"