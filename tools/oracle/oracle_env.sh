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
# LD_LIBRARY_PATH IS required, not cosmetic -- and it is the STARTER that needs
# it.  readelf -d on the installed binaries: starter_linux64_gf carries a NEEDED
# entry for libhm_reader_linux64.so, engine_linux64_gf carries none, and neither
# binary carries a DT_RPATH or a DT_RUNPATH entry, so this export is the only
# route to that library.  The failure is therefore NOT a run-time one: with
# LD_LIBRARY_PATH unset the starter never starts at all.  The dynamic loader
# rejects it before a single instruction of its own code has run, and it exits
# 127 with this on stderr:
#
#     starter_linux64_gf: error while loading shared libraries:
#     libhm_reader_linux64.so: cannot open shared object file: No such file or
#     directory
#
# (the loader prefixes whatever path it was handed).  So a 127 here is a loader
# problem, never an H3D or message-subsystem one: no H3D call is reached,
# because nothing runs.  Do not go hunting an h3d bug on the strength of this
# export being absent -- that is the misdiagnosis this paragraph used to carry,
# and it sent an operator to the wrong subsystem entirely.  The engine needs no
# reader library and prints its -v banner with LD_LIBRARY_PATH unset (exit 0),
# which is the whole of why the two binaries look inconsistent here.  libapr-1
# is not a second symptom to go looking for either: libapr-1.so.0 is a NEEDED
# entry of libhm_reader_linux64.so itself and ships in the very directory this
# export names, so one entry covers it, and ldconfig lists no libapr on this box
# at all, so the loader would find it nowhere else.
#
# The requirement is upstream's, and this paragraph is the retarget of it:
# $OR_SRC/INSTALL.md:34-42 ("Environment variables settings under Linux"), whose
# line 42 is the export
#
#     export LD_LIBRARY_PATH=$OPENRADIOSS_PATH/extlib/hm_reader/linux64/:$LD_LIBRARY_PATH
#
# with OPENRADIOSS_PATH moved to the mirror prefix $OR_BUILD.
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

# RAD_CFG_PATH is upstream's spelling of the CFG card-schema tree
# ($OR_SRC/INSTALL.md:39); pyradioss/paths.py accepts it as an alias of
# $PYRADIOSS_HM_CFG (plan/00_ORCHESTRATION.md 4.1, rule 1).
#
# It is exported ONLY when the directory is really there, and that guard is
# load-bearing, not tidiness.  Since the "a stale export is terminal" change, a
# variable that is set but does not resolve makes the resolver raise instead of
# warning -- so exporting a path that is not there POISONS the one variable that
# would otherwise have let hm_cfg_dir() fall through to a real tree (a sibling
# OpenCourant/hm_cfg_files, $OR_SRC/hm_cfg_files, the Windows prefix).  Measured
# cost of the unconditional form, on this box with a mirror that has no
# hm_cfg_files: tests/test_m539_law34_input_audit.py and
# tests/test_m540_law37_input_audit.py give 57 passed, 14 skipped -- the 7 LAW34
# and 7 LAW37 CFG-schema audits all skip -- against 71 passed with no
# environment at all.  Leave it unset when the mirror has no tree; the resolver
# then searches, which is what 4.1 rule 1 means by "if set".
if [ -d "$OR_BUILD/hm_cfg_files" ]; then
  export RAD_CFG_PATH="$OR_BUILD/hm_cfg_files"
fi

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
