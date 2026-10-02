/* ===========================================================================
 * p0_h3d_writer_adapter.c -- the two h3d writer entry points that the reachable
 * (older) libh3dwriter.so does not export, for the reference (Fortran)
 * OpenRadioss oracle (task P0.4).
 *
 * This is deliberately NOT a re-implementation of the h3d writer.  Both
 * functions below are absent from the reachable library
 * (libh3dwriter.so dated 2026-01-21, export version H3D_EXPORT_11.0):
 *
 *   nm -D --defined-only extlib/h3d/lib/linux64/libh3dwriter.so \
 *       | grep -E 'Hyper3DExportLibraryVersion|Hyper3DCompressionLevel'
 *   -> nothing
 *
 * and the pinned source calls them directly (not through the dlopen shim), so
 * they must exist at link time.
 *
 * 1. Hyper3DExportLibraryVersion
 *
 *    Upstream caller:
 *      engine/source/output/h3d/h3d_build_cpp/c_h3d_export_library_version.cpp:64-66
 *          Hyper3DExportLibraryVersion((uint32_t*)major_version,
 *                                     (uint32_t*)minor_version);
 *      engine/source/output/h3d/h3d_results/genh3d.F:734-746
 *          REQUESTED_H3D_VERSION_MAJOR = 2612
 *          REQUESTED_H3D_VERSION_MINOR = 0
 *          CALL C_H3D_EXPORT_LIBRARY_VERSION(H3D_VERSION_MAJOR, H3D_VERSION_MINOR)
 *          IF (H3D_VERSION_MAJOR < REQUESTED_H3D_VERSION_MAJOR .AND. ...
 *              CALL ANCMSG(MSGID=324,ANMODE=ANINFO,...)
 *              CALL ARRET(2)
 *
 *    That is upstream's own version gate, and it is the correct thing to let it
 *    see: the reachable library reports its real export version
 *    (`readelf -V ... | grep H3D_EXPORT` -> H3D_EXPORT_11.0), so a run that asks
 *    for h3d output is aborted by upstream with MSGID 324 instead of this oracle
 *    silently pretending the library is new enough.  Reporting anything larger
 *    here would defeat a safety check the solver relies on.
 *
 * 2. Hyper3DCompressionLevel
 *
 *    Upstream caller:
 *      engine/source/output/h3d/h3d_build_cpp/c_h3d_open_file.cpp:245
 *          Hyper3DCompressionLevel(h3d_file, *comp_level);
 *    It sits behind the version gate above (genh3d.F opens the file only after
 *    the check), so with this library the call is unreachable.  The definition
 *    below therefore does nothing but say so once, on stderr; it is not a
 *    silent no-op, and it never runs.
 *
 * The other two h3d gaps of the stale extlib are compile-time only and live in
 * the header shims: H3D_NF_FORMAT (p0_h3d_api_shim.h) and the four writer
 * prototypes plus H3D_NOZLIB/H3D_NOSORT (h3dpublic_export.h).
 * =========================================================================*/

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* The export version of the reachable libh3dwriter.so, read from its ELF
 * symbol version table: `readelf -V ... | grep H3D_EXPORT` ->
 *   0x001c: Rev: 1 Flags: none Index: 2 Cnt: 1 Name: H3D_EXPORT_11.0
 * Upstream compares (major, minor) against 2612/0 (genh3d.F:735-736). */
#define P0_H3D_LIBRARY_VERSION_MAJOR 11
#define P0_H3D_LIBRARY_VERSION_MINOR 0

/* Signature transcribed from the pinned source:
 *   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:85
 *       uint32_t (*DLHyper3DExportLibraryVersion)(uint32_t* majorVersion,
 *                                                 uint32_t* minorVersion);
 *   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:938-939
 *       if (DLHyper3DExportLibraryVersion != NULL)
 *           return DLHyper3DExportLibraryVersion(majorVersion, minorVersion);
 * and from the call site at c_h3d_export_library_version.cpp:65. */
uint32_t Hyper3DExportLibraryVersion(uint32_t *majorVersion, uint32_t *minorVersion)
{
    if (majorVersion != NULL) *majorVersion = P0_H3D_LIBRARY_VERSION_MAJOR;
    if (minorVersion != NULL) *minorVersion = P0_H3D_LIBRARY_VERSION_MINOR;
    return P0_H3D_LIBRARY_VERSION_MAJOR * 1000u + P0_H3D_LIBRARY_VERSION_MINOR;
}

/* Signature transcribed from the pinned source:
 *   common_source/output/h3d/h3d_build_cpp/h3d_dl.c:83
 *       bool (*DLHyper3DCompressionLevel)(H3DFileInfo* h3d_file,
 *                                         unsigned int level);
 * called at engine/source/output/h3d/h3d_build_cpp/c_h3d_open_file.cpp:245.
 * H3DFileInfo is only used as an opaque pointer here, so it is not needed. */
bool Hyper3DCompressionLevel(void *h3d_file, unsigned int level)
{
    static int warned = 0;
    (void)h3d_file;
    (void)level;
    if (!warned) {
        warned = 1;
        fprintf(stderr,
                "[p0_h3d_writer_adapter] NOTE: this libh3dwriter.so has no "
                "compression-level API; the level requested by OpenRadioss "
                "cannot be applied.  Unreachable in practice: upstream's h3d "
                "version gate (genh3d.F:734-746) aborts first.\n");
    }
    return false;
}
