/* ===========================================================================
 * h3dpublic_export.h -- shim that adds the two newer h3d writer prototypes to
 * the reachable (older) h3dpublic_export.h, for the reference (Fortran)
 * OpenRadioss oracle built against a stale extlib (task P0.4).
 *
 * It is put on the include path ONLY when the harvested extlib has no
 * h3dpublic_import.h (see tools/oracle/build_oracle.sh), so it can never shadow
 * a real header.  The real header is still used for everything else: it is
 * pulled in with #include_next and only the two prototypes below are replaced.
 *
 * WHY TWO PROTOTYPES HAVE TO CHANGE
 * ---------------------------------
 * The pinned source (OpenCourant/OpenCourant @ 1a0d691b8) calls, in all 77
 * h3d datatype/dataset creation files under
 * engine/source/output/h3d/h3d_build_cpp/, e.g.
 *   c_h3d_create_nodal_scalar_datatype.cpp:139-140
 *       rc = Hyper3DDatatypeWrite(h3d_file, edata_type, *cpt_data,
 *                                  H3D_DS_SCALAR, H3D_DS_NODE,
 *                                  H3D_NF_REAL, pool_count);
 *   c_h3d_eroded_oned.cpp:106-108
 *       rc = Hyper3DDatasetBegin(h3d_file, *NUMELT, sim_idx, subcase_id,
 *                                 H3D_DS_ELEM, H3D_DS_EROSION, H3D_NF_REAL,
 *                                 num_corners, num_modes, *CPT_DATATYPE, 0,
 *                                 truss_poolname_id);
 * i.e. H3D_NF_FORMAT sits immediately after (type, format), and the trailing
 * `bool complex` of Hyper3DDatasetBegin is gone.
 *
 * The reachable h3dpublic_export.h (extlib v59) declares both without the
 * nf_format argument, which is the compile error
 *   c_h3d_create_1d_tensor_datatype.cpp:149:34: error: too many arguments to
 *   function 'bool Hyper3DDatatypeWrite(H3DFileInfo*, const char*,
 *   unsigned int, H3D_DS_FORMAT, H3D_DS_TYPE, unsigned int)'
 * The declarations below are the pinned source's own call sites' shape, and they
 * are the only ones that can compile in C++ (an `int` such as `pool_count`
 * cannot be passed implicitly to an enum parameter, so the older
 * nf_format-last shape is rejected outright).
 *
 * WHAT THE OLDER LIBRARY DOES WITH THEM
 * -------------------------------------
 * Both entry points keep their names in the reachable library
 * (libh3dwriter.so: Hyper3DDatatypeWrite@@H3D_EXPORT_11.0,
 * Hyper3DDatasetBegin@@H3D_EXPORT_11.0) and are reached through the dlopen
 * shim in common_source/output/h3d/h3d_build_cpp/h3d_dl.c:878,894.  The
*   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:878,894.  The
 * prototype below is what those C++ call sites bind to; the definition they
 * actually run is the shim's exported wrapper, h3d_dl.c:1371-1376 and
 * :1409-1420, which takes H3D_NF_FORMAT as its LAST parameter and forwards it
 * to the dlsym'd library pointer.  The pinned tree therefore disagrees with
 * itself about the position of that argument (call sites: 6th/7th; wrapper:
 * last), and no choice of header can reconcile the two: either way the wrapper
 * receives H3D_NF_REAL where it expects num_pools.  That is an upstream
 * inconsistency, present with a current extlib too, and it is NOT repaired
 * here -- upstream source is not modified.  Its consequence is recorded in
 * tools/validation_data/oracle_provenance.json: h3d output from this oracle is
 * inadmissible for parity, and it is additionally REFUSED at run time because
 * oracle_env.sh does not put the stale writer on RAD_H3D_PATH (see the note on
 * Hyper3DExportLibraryVersion below and h3d_dl.c:920-921 / genh3d.F:729-731).
 * =========================================================================*/
#ifndef P0_H3DPUBLIC_EXPORT_SHIM_H
#define P0_H3DPUBLIC_EXPORT_SHIM_H

#include "p0_h3d_api_shim.h"   /* H3D_NF_FORMAT, see that file */

/* Hide the outdated prototypes while the real header is parsed. */
#define Hyper3DDatatypeWrite   P0_old_Hyper3DDatatypeWrite
#define Hyper3DDatasetBegin    P0_old_Hyper3DDatasetBegin
#define Hyper3DElementBegin    P0_old_Hyper3DElementBegin
#define Hyper3DElement2Begin   P0_old_Hyper3DElement2Begin
#include_next <h3dpublic_export.h>
#undef Hyper3DDatatypeWrite
#undef Hyper3DDatasetBegin
#undef Hyper3DElementBegin
#undef Hyper3DElement2Begin

#include <stdint.h>
#include <h3dpublic_defs.h>

/* The reachable h3dpublic_defs.h has no H3D_NOZLIB / H3D_NOSORT: that h3d build
 * can neither disable zlib nor skip the sort, so the flags are 0 here.  Zero is
 * the honest value (the requested mode is unavailable and the library's default
 * applies); inventing bit patterns would silently set unrelated mode bits in a
 * library whose H3D_FileMode layout is not ours to guess
 * (extlib/h3d/includes/h3dpublic_defs.h:171-176 shows the reachable layout:
 * H3D_SINGLEFILE 0, H3D_MULTIFILEBYSIM 1, H3D_APPEND 4).  Upstream uses them at
 * engine/source/output/h3d/h3d_build_cpp/c_h3d_open_file.cpp:240-241. */
#ifndef H3D_NOZLIB
#define H3D_NOZLIB 0
#endif
#ifndef H3D_NOSORT
#define H3D_NOSORT 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

DllExport bool Hyper3DDatatypeWrite(H3DFileInfo* h3d_file, const char* label,
                    H3D_ID dt_id, H3D_DS_FORMAT format, H3D_DS_TYPE type,
                    H3D_NF_FORMAT nf_format, unsigned int num_pools);

DllExport bool Hyper3DDatasetBegin(H3DFileInfo* h3d_file, unsigned int count,
                    H3D_SIM_IDX idx, H3D_ID subcase_id,
                    H3D_DS_TYPE type, H3D_DS_FORMAT format,
                    H3D_NF_FORMAT nf_format,
                    unsigned int num_corners, unsigned int num_modes,
                    H3D_ID dt_id, int layer_idx, H3D_ID data_poolname_id);

/* Hyper3DElementBegin / Hyper3DElement2Begin gained an H3D_ID type_id right
 * after `config`; the shape below is the pinned source's own, and here the
 * call sites and the shim agree (common_source/output/h3d/h3d_build_cpp/
 * h3d_dl.c:172-179 typedefs and :1192-1223 wrappers, versus e.g.
 * engine/source/output/h3d/h3d_build_cpp/c_h3d_create_beams.cpp:97-99 and
 * c_h3d_create_rbe2.cpp:94-96).  The reachable library takes seven parameters,
 * so the eighth travels in a register it never reads -- which shifts
 * parent_id/parent_poolname_id/node_poolname_id inside the old callee.  That is
 * a real ABI break, and it is one more reason h3d output is recorded as
 * inadmissible for parity. */
DllExport bool Hyper3DElementBegin(H3DFileInfo* h3d_file, unsigned int count,
                    H3D_ID poolname_id, H3D_ElementConfig config,
                    H3D_ID type_id, H3D_ID parent_id,
                    H3D_ID parent_poolname_id, H3D_ID node_poolname_id);

DllExport bool Hyper3DElement2Begin(H3DFileInfo* h3d_file, unsigned int count,
                    H3D_ID poolname_id, H3D_ElementConfig config,
                    H3D_ID type_id, H3D_ID parent_id,
                    H3D_ID parent_poolname_id, H3D_ID node_poolname_id);

/* Declared by the reachable libh3dwriter.so?  No -- neither
 * Hyper3DExportLibraryVersion nor Hyper3DCompressionLevel exists there
 * (`nm -D --defined-only extlib/h3d/lib/linux64/libh3dwriter.so | grep -c
 * Hyper3DExportLibraryVersion` -> 0).  They are DEFINED by the pinned source
 * itself, in the dlopen shim every binary links:
 *   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:984-989  Hyper3DCompressionLevel
 *   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:991-1000  Hyper3DExportLibraryVersion
 *     (with the "function not available -> 0.0" fallback at :996-1000)
 * and the C++ callers bind to those definitions, so only the DECLARATIONS are
 * needed here.  There is deliberately no adapter .c for them: an earlier
 * revision had one, it was never linked (h3d_dl.o already provides both
 * symbols, so the archive member was never pulled in and `strings` found no
 * trace of it), and it advertised a safety net that did not exist.
 *
 * What actually stops an h3d run on this box is stated in
 * tools/validation_data/oracle_provenance.json and enforced by
 * tools/oracle/oracle_env.sh: the ABI-incompatible libh3dwriter.so is NOT put
 * on RAD_H3D_PATH, so h3d_dl.c:920-921 returns *IERROR = 1 and
 * engine/source/output/h3d/h3d_results/genh3d.F:729-731 aborts the run with
 * MSGID 274 instead of letting a one-parameter-short ABI write H3D files. */
DllExport uint32_t Hyper3DExportLibraryVersion(uint32_t* majorVersion,
                                               uint32_t* minorVersion);
DllExport bool Hyper3DCompressionLevel(H3DFileInfo* h3d_file,
                                       unsigned int level);

#ifdef __cplusplus
}
#endif

#endif /* P0_H3DPUBLIC_EXPORT_SHIM_H */
