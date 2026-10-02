/* ===========================================================================
 * p0_h3d_api_shim.h -- the missing piece of the h3d SDK header surface, for the
 * reference (Fortran) OpenRadioss oracle built against a stale extlib
 * (task P0.4).  Force-included with `gcc -include` by tools/oracle/build_oracle.sh;
 * it is never copied over a harvested header and never modifies a binary.
 *
 * WHAT IS MISSING
 * ---------------
 * The pinned source (OpenCourant/OpenCourant @ 1a0d691b8, 2026-09-26) passes one
 * extra argument of type H3D_NF_FORMAT to the h3d datatype/dataset entry
 * points:
 *
 *   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:258-260
 *       bool (*DLHyper3DDatatypeWrite)(H3DFileInfo*, const char*, H3D_ID,
 *           H3D_DS_FORMAT, H3D_DS_TYPE, unsigned int, H3D_NF_FORMAT nf_format);
 *   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:276-281
 *       bool (*DLHyper3DDatasetBegin)(..., H3D_ID data_poolname_id,
 *           H3D_NF_FORMAT nf_format);
 *   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:1370-1376, 1409-1417
 *       the exported wrappers Hyper3DDatatypeWrite / Hyper3DDatasetBegin that
 *       forward nf_format to them.
 *
 *   The 77 call sites are in engine/source/output/h3d/h3d_build_cpp/*.cpp, e.g.
 *   c_h3d_create_nodal_scalar_datatype.cpp:139-140
 *       rc = Hyper3DDatatypeWrite(h3d_file, edata_type, *cpt_data, H3D_DS_SCALAR,
 *                                 H3D_DS_NODE, H3D_NF_REAL, pool_count);
 *   and every one of them passes the single enumerator H3D_NF_REAL
 *   (`grep -rho 'H3D_NF_[A-Z_]*' engine starter common_source | sort -u`
 *    -> H3D_NF_REAL, H3D_NF_FORMAT).
 *
 * The reachable extlib (v59, 2026-05-28) predates that API:
 *   h3d_dl.c:260:36: error: unknown type name 'H3D_NF_FORMAT';
 *                            did you mean 'H3D_DS_FORMAT'?
 *
 * WHY THIS TYPE IS ADMISSIBLE (and what it does NOT buy)
 * -----------------------------------------------------
 * Both entry points are resolved at RUN time by dlsym in the same file
 * (h3d_dl.c:568, 584, 878, 894), they keep their names in the reachable
 * library, and the reachable libh3dwriter.so exports them with version tag
 * H3D_EXPORT_11.0:
 *
 *   nm -D --defined-only extlib/h3d/lib/linux64/libh3dwriter.so
 *     00000000000518d0 T Hyper3DDatatypeWrite@@H3D_EXPORT_11.0
 *     0000000000051e30 T Hyper3DDatasetBegin@@H3D_EXPORT_11.0
 *
 * The extra argument therefore travels in a register that the older callee
 * never reads: the call is ABI-compatible, and the older writer simply produces
 * the pre-node-force output.
 *
 * WHAT REMAINS UNSUPPORTED, stated plainly: node-force ("NF") datasets cannot
 * be written by this oracle, because no reachable libh3dwriter.so exports a
 * node-force entry point
 * (`nm -D --defined-only ... | grep -ci node` -> 0, on every candidate image as
 * well).  That is a missing FEATURE, not a broken call, and it is why
 * tools/validation_data/oracle_provenance.json records h3d output as
 * inadmissible for parity.
 *
 * The numeric value of H3D_NF_REAL is not recoverable from any reachable
 * source; it is 0 here so that the code compiles, and it is ignored by the
 * reachable library.  The name, its use and its position in the argument list
 * are all taken from the upstream sources cited above.
 * =========================================================================*/
#ifndef P0_H3D_API_SHIM_H
#define P0_H3D_API_SHIM_H

typedef enum _H3D_NF_FORMAT {
  H3D_NF_REAL = 0
} H3D_NF_FORMAT;

#endif /* P0_H3D_API_SHIM_H */
