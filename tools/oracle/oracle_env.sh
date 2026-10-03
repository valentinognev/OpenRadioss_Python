# Runtime environment for the reference (Fortran) OpenRadioss binaries built
# from the writable mirror.  Source it, do not execute it:
#
#   source tools/oracle/oracle_env.sh
#
# RAD_H3D_PATH is deliberately NOT exported, and that is a safety decision, not
# an omission.  INSTALL.md:38-42 tells you to set it, and with a current extlib
# you must.  The extlib reachable on this machine (v59) carries a
# libh3dwriter.so whose writer API is one parameter short of what the pinned
# source calls, so a run that wrote H3D files through it would silently produce
# wrong ones.  Instead of pointing RAD_H3D_PATH at that library, this file
# leaves it unset: common_source/output/h3d/h3d_build_cpp/h3d_dl.c:920-921 then
# fails to dlopen the writer in any of its four trials and returns *IERROR = 1,
# which engine/source/output/h3d/h3d_results/genh3d.F:729-731 turns into
# MSGID 274 + ARRET(2).  h3d output is refused loudly; T01, A-files, RESTART and
# the .out listing are unaffected.  The second and third dlopen trials look in
# the working directory and in $ALTAIR_HOME/$ARCH, so keep the stale
# libh3dwriter.so out of both.
#
# LD_LIBRARY_PATH IS required, not cosmetic: the binaries start without it and
# then die on the first H3D / message call with an unresolved libhm_reader or
# libapr-1.  Both come straight from $OR_SRC/INSTALL.md:34-42 ("Environment
# variables settings under Linux"), retargeted from OPENRADIOSS_PATH to the
# mirror prefix $OR_BUILD.
#
# Expects the mirror to already exist (see tools/oracle/mirror_and_fetch.sh) and
# the oracle to be built (see tools/oracle/build_oracle.sh).

: "${OR_BUILD:?source oracle_env.sh after exporting OR_BUILD (see mirror_and_fetch.sh)}"
: "${OR_ROOT:?set OR_ROOT to the writable install prefix}"

export OR_BUILD
export OR_ROOT

# OPENRADIOSS_PATH is the upstream name for the prefix; some helper scripts and
# the hm_cfg_files lookup still key off it.
export OPENRADIOSS_PATH="$OR_BUILD"
export RAD_CFG_PATH="$OR_BUILD/hm_cfg_files"

# The native-.k reader and its APR dependency.  libapr-1.so.0 has a real
# NEEDED entry on the system libcrypt.so.1, which the runtime loader finds in
# the default search path (verified by the ldd gate in build_oracle.sh).
export LD_LIBRARY_PATH="$OR_BUILD/extlib/hm_reader/linux64/:${LD_LIBRARY_PATH:-}"

# OpenRadioss stacks aggressively; the OpenMP runtime needs headroom.
export OMP_STACKSIZE=400m

# The two oracle binaries themselves.  Names come from
# $OR_SRC/INSTALL.md:110-111 and from build_oracle.sh (the cmake TARGET names
# are only `starter`/`engine` -- see the trap notes in build_oracle.sh).
export OR_STARTER="$OR_ROOT/bin/starter_linux64_gf"
export OR_ENGINE="$OR_ROOT/bin/engine_linux64_gf"

# Put the built starter/engine at the front of PATH.
export PATH="$OR_ROOT/bin:$PATH"
