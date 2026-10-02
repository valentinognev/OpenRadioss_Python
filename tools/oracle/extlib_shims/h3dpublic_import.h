/* ===========================================================================
 * h3dpublic_import.h -- stand-in for the h3d SDK header of the same name, which
 * exists only in extlib releases newer than the reachable v59 and is required
 * by the pinned OpenRadioss source (task P0.4).
 *
 * UPSTREAM USER (the only one in the whole tree):
 *   starter/source/output/checksum/checksum_list.cpp:27
 *       #include <h3dpublic_import.h>
 *   ...:640-687  List_checksum::parse_h3d_files()
 *       int ierror = 0;
 *       h3dreaderlib_load_(&ierror);
 *       if (ierror != 0) { ...H3D checksum parsing skipped...; return; }
 *       for (...) {
 *           H3DFileInfo* h3d = Hyper3DImportOpen(h3d_path.c_str(), nullptr, nullptr);
 *           while (Hyper3DLookupString(h3d, string_id, &value)) { ... }
 *           Hyper3DImportClose(h3d);
 *           ...
 *       }
 *
 * The three prototypes are transcribed from the pinned source tree itself,
 * which declares the same reader API for its own dynamic loader:
 *   common_source/output/h3d/h3d_build_cpp/h3dreader_dl.c:53,56-57,78-84
 *       typedef void H3DReaderInfo;
 *       H3DReaderInfo* (*DLHyper3DImportOpen)(const char* filename, ...);
 *       bool (*DLHyper3DImportClose)(H3DReaderInfo* h3d_file);
 *       bool (*DLHyper3DLookupString)(H3DReaderInfo* h3d_file, uint32_t str_id,
 *                                    const char** string);
 *
 * WHY THE DECLARATIONS ARE WEAK
 * -----------------------------
 * These three functions live in libh3dreader.so, which the starter does NOT
 * link: h3dreader_dl.c dlopens it (dlsym at h3dreader_dl.c:226-231) with
 * RTLD_GLOBAL, and checksum_list.cpp:645-652 refuses to call anything if that
 * load reported an error.  A strong declaration would make the link fail with
 * three more undefined references; a weak one leaves the symbols to be resolved
 * by the dynamic linker from the dlopen'ed library at first call, and stays
 * harmlessly null when libh3dreader.so is absent -- in which case the caller's
 * own error check has already returned.  Nothing here implements those
 * functions: the reachable libh3dwriter.so does not export them
 * (`nm -D --defined-only | grep -c Hyper3DImport` -> 0), and faking an H3D
 * string-table reader would make /CHECKSUM_REPORT report wrong checksums.
 *
 * Because this file is only put on the include path when the harvested extlib
 * has no h3dpublic_import.h of its own, it can never shadow the real header.
 * =========================================================================*/
#ifndef P0_H3DPUBLIC_IMPORT_H
#define P0_H3DPUBLIC_IMPORT_H

#include <stdint.h>
#include <stdbool.h>

#include <h3dpublic_defs.h>   /* H3DFileInfo, as the real header does */
#include "p0_h3d_api_shim.h"  /* H3D_NF_FORMAT, see that file */

#if defined(__GNUC__) || defined(__clang__)
#  define P0_WEAK __attribute__((weak))
#else
#  define P0_WEAK
#endif

#ifdef __cplusplus
extern "C" {
#endif

P0_WEAK H3DFileInfo* Hyper3DImportOpen(const char* filename, const char* password,
                                        void* comm);
P0_WEAK bool Hyper3DImportClose(H3DFileInfo* h3d_file);
P0_WEAK bool Hyper3DLookupString(H3DFileInfo* h3d_file, uint32_t str_id,
                                 const char** string);

#ifdef __cplusplus
}
#endif

#endif /* P0_H3DPUBLIC_IMPORT_H */
