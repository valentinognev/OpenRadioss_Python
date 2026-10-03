/* ===========================================================================
 * p0_hm_reader_adapter.c -- source-level adapter for the reference (Fortran)
 * OpenRadioss oracle (task P0.4).
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The oracle is built from the pinned source tree
 *   OpenCourant/OpenCourant @ 1a0d691b82bd4f94a3bc2399682551e84e1d5fa4
 * whose EXTLIB_VERSION.json asks for extlib v82.  The newest extlib that is
 * reachable from this machine is v59 (see tools/validation_data/
 * oracle_provenance.json), so three reader entry points the starter links
 * against do not exist in libhm_reader_linux64.so:
 *
 *   ld: undefined reference to `cpp_is_part_with_elements_'
 *   ld: undefined reference to `cpp_sale_mesh_create_'
 *   ld: undefined reference to `cpp_get_include_file_by_index'
 *
 * This file implements them against the v59 API.  It is ordinary source: it is
 * compiled, its behaviour is auditable, and it is declared symbol-by-symbol in
 * the provenance file.  The harvested binaries are NOT modified, and nothing
 * here pretends to have read a deck it could not read.
 *
 * WHERE THE v59 PRIMITIVES COME FROM
 * ----------------------------------
 * libhm_reader_linux64.so exports the reader's C++ implementations directly:
 *
 *   _Z31GlobalModelSDICountIncludeFilesPi     GlobalModelSDICountIncludeFiles(int*)
 *   _Z29GlobalModelSDIGetIncludesListPPc      GlobalModelSDIGetIncludesList(char**)
 *   _Z34GlobalEntitySDICountElementsInPartPi  GlobalEntitySDICountElementsInPart(int*)
 *   _Z20GlobalEntitySDIGetIdPiPb              GlobalEntitySDIGetId(int*, bool*)
 *
 * The same tree that defines the *new* call sites also contains an
 * implementation of two of them for OpenRadioss' own reader, which is the
 * authority for their semantics:
 *   reader/source/solver_interface/source/cfg_reading/cpp_get_number_of_include_files.cpp:31-43
 *   reader/source/solver_interface/source/cfg_reading/cpp_get_include_files_list.cpp:33-45
 *   reader/source/solver_interface/source/cfg_reading/cpp_count_elements_in_part.cpp:36-49
 *   reader/source/solver_interface/source/cfg_reading/GlobalModelSdi.cpp:2561-2585
 *   reader/source/solver_interface/source/cfg_reading/GlobalModelSdi.cpp:2666-2693
 *
 * The mangled implementation symbols are used deliberately instead of the
 * `cpp_*` Fortran-linkage thunks in the same .so.  An earlier version of this
 * comment claimed those thunks point at unrelated code; that was WRONG and has
 * been removed.  What objdump actually shows is:
 *
 *   00000000003942a0 <cpp_count_elements_in_part_>:
 *     3942a0:  jmp  296990 <_Z34GlobalEntitySDICountElementsInPartPi@plt>
 *   00000000003961b0 <cpp_get_number_of_include_files_>:
 *     jmp  2960e0 <cpp_get_number_of_include_files_@plt>  (self-thunk chain)
 *   00000000003952b0 <cpp_get_include_files_list_>:
 *     3952b0:  jmp  294680 <_Z29GlobalModelSDIGetIncludesListPPc@plt>
 *
 * i.e. the thunks resolve to the very implementations used below.  Two reasons
 * remain for addressing the mangled symbols directly: (1) it names the exact
 * implementation instead of relying on an alias table that upstream itself
 * generates, and (2) cpp_get_number_of_include_files is the one symbol where
 * the thunk CANNOT be used, because its target takes ONE argument while the
 * pinned source calls it with two (see that function's comment).  Every
 * primitive below is resolved by dlsym from its own mangled name, and each is
 * checked: if one is missing the adapter reports the failure instead of
 * inventing a value.
 *
 * HOW IT IS BUILT IN
 * ------------------
 * tools/oracle/build_oracle.sh compiles this into a static archive and passes
 * it to cmake as -Dflexpipe_lib=<archive>.  `${flexpipe_lib}` is an unused hook
 * in starter/CMake_Compilers/cmake_linux64_gf.txt:177 (`set (LINK "dl
 * ${flexpipe_lib} ...")`, intended for the FlexiPE coupling), so this needs no
 * edit to the mirror's CMake files.
 *
 * WHAT IS *NOT* ADAPTED HERE
 * --------------------------
 * cpp_sale_mesh_create_: the v59 reader has no structured-ALE-mesh creator at
 * all (`nm -D --defined-only | grep -ci sale` finds only unrelated
 * RadiossAleCfdSph_Card symbols), so the adapter reports the reader's own
 * "mesh not created" error and lets the caller's error path raise it.  Decks
 * without /ALE/STRUCTURED_MESH never call it.
 * =========================================================================*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* --------------------------------------------------------------------------
 * v59 primitives, resolved lazily by their true (mangled) symbol names.
 * -------------------------------------------------------------------------- */
typedef void (*fn_count_include_files)(int *n);
typedef void (*fn_get_includes_list)(char **list);
typedef void (*fn_count_elements_in_part)(int *n);
typedef void (*fn_get_current_id)(int *id, bool *ok);

static void *p0_sym(const char *name)
{
    /* RTLD_DEFAULT: the adapter is linked into the executable, and
     * libhm_reader_linux64.so is a DT_NEEDED of it (starter arch file:25), so
     * its symbols are already in the global scope. */
    void *p = dlsym(RTLD_DEFAULT, name);
    if (p == NULL) {
        fprintf(stderr,
                "[p0_hm_reader_adapter] FATAL: v59 primitive '%s' not found (%s)\n"
                "[p0_hm_reader_adapter] the oracle must not be run with this extlib\n",
                name, dlerror());
    }
    return p;
}

static fn_count_include_files p0_count_include_files(void)
{
    static fn_count_include_files f = NULL;
    static int tried = 0;
    if (!tried) { tried = 1; f = (fn_count_include_files)p0_sym("_Z31GlobalModelSDICountIncludeFilesPi"); }
    return f;
}

static fn_get_includes_list p0_get_includes_list(void)
{
    static fn_get_includes_list f = NULL;
    static int tried = 0;
    if (!tried) { tried = 1; f = (fn_get_includes_list)p0_sym("_Z29GlobalModelSDIGetIncludesListPPc"); }
    return f;
}

static fn_count_elements_in_part p0_count_elements_in_part(void)
{
    static fn_count_elements_in_part f = NULL;
    static int tried = 0;
    if (!tried) { tried = 1; f = (fn_count_elements_in_part)p0_sym("_Z34GlobalEntitySDICountElementsInPartPi"); }
    return f;
}

static fn_get_current_id p0_get_current_id(void)
{
    static fn_get_current_id f = NULL;
    static int tried = 0;
    if (!tried) { tried = 1; f = (fn_get_current_id)p0_sym("_Z20GlobalEntitySDIGetIdPiPb"); }
    return f;
}

/* ===========================================================================
 * 1. cpp_get_number_of_include_files
 *
 * Upstream caller (newer contract, TWO arguments):
 *   starter/source/devtools/hm_reader/write_include_files_list.F90:42-47
 *       subroutine cpp_get_number_of_include_files(is_dyna, num_includes)
 *         bind(C, name='cpp_get_number_of_include_files')
 *         integer(c_int), intent(in)  :: is_dyna
 *         integer(c_int), intent(out) :: num_includes
 *   ...:86-87   num_includes = 0 ; call cpp_get_number_of_include_files(...)
 *   ...:99-111 the loop over 1..num_includes calls the adapter below
 *
 * The v59 library exports the same symbol with the OLD, ONE-argument contract
 * (reader/source/solver_interface/source/cfg_reading/
 * cpp_get_number_of_include_files.cpp:31-33), and its thunk jumps straight to
 * GlobalModelSDICountIncludeFiles(int*).  Left alone, upstream's two-argument
 * call would write the count through the FIRST argument -- clobbering the
 * caller's `is_dyna` -- and leave num_includes at the 0 it was just given, so
 * the include-file block of the starter listing would silently disappear.
 * Defining the symbol here fixes that: it is the only caller of it in the
 * starter/engine/common_source trees, and the .so's own thunks call the
 * single-underscore name, which is left untouched.
 *
 * is_dyna (0 = Radioss-format deck, 1 = native LS-DYNA .k, see
 * starter/source/starter/starter0.F:412-415,697) has no counterpart in the v59
 * model view, which always answers for the deck it loaded; that is what the
 * v59-era single-argument function did as well.
 * =========================================================================*/
void cpp_get_number_of_include_files(const int *is_dyna, int *num_includes)
{
    fn_count_include_files count = p0_count_include_files();
    (void)is_dyna;
    *num_includes = 0;
    if (count != NULL) count(num_includes);
}

/* ===========================================================================
 * 2. cpp_get_include_file_by_index   (MISSING from v59 -- adapted)
 *
 * Upstream caller:
 *   starter/source/devtools/hm_reader/write_include_files_list.F90:49-56
 *       subroutine cpp_get_include_file_by_index(is_dyna, include_index,
 *                                                 file_name, bufsize)
 *         bind(C, name='cpp_get_include_file_by_index')
 *         integer(c_int), intent(in)  :: is_dyna
 *         integer(c_int), intent(in)  :: include_index
 *         character(kind=c_char), intent(out) :: file_name(*)
 *         integer(c_int), intent(in)  :: bufsize
 *   ...:79-81  bufsize = size(c_file_name) = 512 ; c_file_name(512)
 *   ...:105-111
 *       do include_index = 1, num_includes
 *         c_file_name = ' '
 *         call cpp_get_include_file_by_index(is_dyna, include_index,
 *                                           c_file_name, bufsize)
 *         file_name = transfer(c_file_name, file_name)
 *         write(iout,'(6X,A)') trim(file_name)
 *       end do
 *
 * Contract the caller relies on, and nothing more:
 *   - include_index is 1-based and bounded by the count from
 *     cpp_get_number_of_include_files;
 *   - on return file_name holds a NUL-terminated C string of at most bufsize
 *     bytes (it is transferred straight into a Fortran CHARACTER(len=512) and
 *     printed), and the caller has pre-blanked the buffer;
 *   - there is no error channel: nothing is reported to the caller, so an
 *     unavailable name must come back as the blank string the caller already
 *     put there, never as a fabricated path.
 *
 * v59 has no by-index accessor, only the whole-list one
 * (GlobalModelSDIGetIncludesList(char**), semantics per
 * reader/source/solver_interface/source/cfg_reading/GlobalModelSdi.cpp:2666-2693:
 * fills entries 0..n-1 with individually malloc'd names).  So the adapter asks
 * for the list and picks the requested entry.
 *
 * The list is intentionally NOT free()d: those pointers are the reader's own
 * allocations (the reference implementation uses malloc per entry) and the
 * reader may still own them; leaking a few hundred bytes once per starter run
 * is the safe choice, a double free would abort the run.
 * =========================================================================*/
void cpp_get_include_file_by_index(const int *is_dyna, const int *include_index,
                                   char *file_name, const int *bufsize)
{
    fn_count_include_files count = p0_count_include_files();
    fn_get_includes_list list = p0_get_includes_list();
    int n = 0;
    char **entries = NULL;
    int i = (include_index != NULL) ? *include_index : 0;
    int cap = (bufsize != NULL && *bufsize > 0) ? *bufsize : 0;

    if (file_name == NULL || cap <= 1) return;
    file_name[0] = '\0';                    /* blank: the documented fallback */
    if (count == NULL || list == NULL) return;
    (void)is_dyna;                          /* no counterpart in the v59 view */

    count(&n);
    if (n <= 0 || i < 1 || i > n) return;

    entries = (char **)calloc((size_t)n, sizeof(char *));
    if (entries == NULL) return;
    list(entries);
    if (entries[i - 1] != NULL) {
        strncpy(file_name, entries[i - 1], (size_t)cap - 1);
        file_name[cap - 1] = '\0';
    }
    free(entries);                          /* the strings themselves stay */
}

/* ===========================================================================
 * 3. cpp_is_part_with_elements_   (MISSING from v59 -- adapted)
 *
 * Upstream caller:
 *   starter/source/devtools/hm_reader/hm_is_part_with_elements.F90:44-62
 *       subroutine hm_is_part_with_elements(part_id, is_part_with_elements)
 *         logical(c_bool), intent(inout) :: is_part_with_elements
 *         ...
 *         call cpp_is_part_with_elements(part_id, is_part_with_elements)
 *     No interface block: a plain Fortran external call, so the symbol is
 *     `cpp_is_part_with_elements_` (lowercase + one underscore) and both
 *     arguments arrive by reference.  `logical(c_bool)` is a 4-byte integer
 *     holding 0/1.
 *
 *   starter/source/model/assembling/hm_read_part.F:210-232
 *       IS_FILLED = .FALSE.
 *       call hm_is_part_with_elements(ID, IS_FILLED)      ! ID = current /PART
 *       ... MSGID 178 as ERROR if IS_FILLED else WARNING ...
 *       IF(IS_FILLED .OR. IGTYP == 34) THEN               ! element-stack path
 *
 * Contract: answer "does the /PART just read contain at least one element".
 * The caller has just done HM_OPTION_READ_KEY for that part, so the reader's
 * current entity IS the part in question -- which is exactly what the v59
 * primitive counts (GlobalEntitySDICountElementsInPart, reference semantics at
 * reader/source/solver_interface/source/cfg_reading/GlobalModelSdi.cpp:2561-2585:
 * `SelectionElementRead elems(*g_pEntity); *NB_ELEMS = elems.Count();`).
 *
 * part_id is therefore only used as a consistency check: if the reader's current
 * option id disagrees with it the situation is one the caller never produces,
 * so it is reported once on stderr and the count of the current entity is still
 * answered (that is the entity the caller is asking about).
 *
 * The flag is always written, so the caller's intent(inout) contract holds; on
 * any internal failure it is left FALSE, which is the value the caller had just
 * set (hm_read_part.F:210) -- i.e. "no elements", never a fabricated "yes".
 *
 * ONE BYTE, NOT AN int -- this is the whole subtlety of this function.
 * `logical(c_bool)` is a FOUR-byte... no: gfortran gives `logical(c_bool)`
 * KIND = 1 and STORAGE_SIZE = 1, verified with
 *   print *, kind(b), storage_size(b)/8      ->  1  1
 * So the caller's IS_FILLED (hm_read_part.F:131) occupies ONE byte, and the
 * adapter must write exactly one byte through the pointer.  An earlier version
 * of this file stored an `int`, which overwrote the three bytes above
 * IS_FILLED: in that frame they are the low bytes of the part id ID, so every
 * /PART came out as 0 and the starter aborted with MSGERROR 494 + 402
 * (reproduced on tests/data/oracle/part_smoke_0000.rad, gated by
 * test_oracle_starter_reads_a_multi_part_deck).  Every other argument of the
 * three other adapted entry points is a genuine 4-byte integer or a character
 * array, and is audited in the width table below.
 * =========================================================================*/

/* Width audit of every pointer the four adapted entry points receive.
 *
 *   entry point                          argument   upstream declaration
 *   ------------------------------------  ---------  ----------------------------
 *   cpp_get_number_of_include_files      is_dyna    integer(c_int)   -> 4 bytes
 *                                         num        integer(c_int)   -> 4 bytes
 *   cpp_get_include_file_by_index        is_dyna    integer(c_int)   -> 4 bytes
 *                                         index      integer(c_int)   -> 4 bytes
 *                                         file_name  character(kind=c_char)(*) -> 1 byte/char, 512 of them
 *                                         bufsize    integer(c_int)   -> 4 bytes
 *   cpp_is_part_with_elements_           part_id    integer           -> 4 bytes
 *                                         flag      logical(c_bool)   -> 1 byte  <-- the trap
 *   cpp_sale_mesh_create_                message    integer(45)       -> 4 bytes/element
 *
 * Sources: starter/source/devtools/hm_reader/write_include_files_list.F90:42-56
 *          starter/source/devtools/hm_reader/hm_is_part_with_elements.F90:44-56
 *          starter/source/devtools/hm_reader/hm_s_ale.F90:79-80,133
 *          starter/source/model/assembling/hm_read_part.F:131,211
 * =========================================================================*/
void cpp_is_part_with_elements_(const int *part_id, int *is_part_with_elements)
{
    fn_count_elements_in_part count = p0_count_elements_in_part();
    fn_get_current_id get_id = p0_get_current_id();
    int n = 0;
    /* One byte only: the caller passes a `logical(c_bool)`, which gfortran
     * stores in ONE byte (kind 1).  Writing an int here corrupts the caller's
     * frame -- in hm_read_part.F the part id itself. */
    unsigned char *flag = (unsigned char *)is_part_with_elements;

    if (flag == NULL) return;
    *flag = 0;
    if (count == NULL) return;

    if (get_id != NULL && part_id != NULL) {
        int cur = 0;
        bool ok = false;
        get_id(&cur, &ok);
        static int warned = 0;
        if (ok && cur != *part_id && !warned) {
            warned = 1;
            fprintf(stderr,
                    "[p0_hm_reader_adapter] WARNING: current option id %d != "
                    "requested part id %d; answering for the current entity\n",
                    cur, *part_id);
        }
    }

    count(&n);
    *flag = (n > 0) ? 1u : 0u;
}

/* Non-underscored alias, for callers written in C (upstream's own reader
 * exports the same four spellings for every cpp_* entry point, see
 * reader/source/solver_interface/source/cfg_reading/cpp_count_elements_in_part.cpp:42-49). */
void cpp_is_part_with_elements(const int *part_id, int *is_part_with_elements)
{
    cpp_is_part_with_elements_(part_id, is_part_with_elements);
}

/* ===========================================================================
 * 4. cpp_sale_mesh_create_   (MISSING from v59 -- NOT implementable)
 *
 * Upstream caller:
 *   starter/source/devtools/hm_reader/hm_s_ale.F90:24-162, call at :133
 *       message_value(1:45) = 0 ; message_value(1) = 0
 *       ...
 *       call cpp_sale_mesh_create(message_value)
 *     no interface block -> symbol `cpp_sale_mesh_create_`, one argument by
 *     reference: 45 integers, pre-zeroed by the caller, filled in by the reader
 *     with (per the comment block at hm_s_ale.F90:100-124) the mesh id, part id,
 *     control points, kind of mesh, trimming, created element/node counts and
 *     offsets, boundary conditions, an error code, material and offsets, the
 *     reference node and its /ALE/LINK/VEL id.
 *
 *   The consumer is starter/source/ale/s_ale_message.F90:104-229 and it keys
 *   entirely off field 38:
 *     ...:107   if(input_modification%s_ale(i)%error < 0) error_check(i)=.true.
 *     ...:116   if(.not.error_check(i)) then  <report the mesh>
 *     ...:209-227  else  -> error_id == -1        => MSGERROR 3153
 *                             -2..-4                => MSGERROR 3154
 *                             -5..-7                => MSGERROR 3155
 *                             -39                   => MSGERROR 3156
 *   Any other negative code would leave `message_id` unset and then be passed
 *   to ANCMSG (a bug at :228), so only codes the caller knows may be returned.
 *
 * The v59 reader has no structured-ALE-mesh creator: nothing in
 * libhm_reader_linux64.so matches sale/structured_mesh (the only "ale" hits are
 * the unrelated RadiossAleCfdSph_Card reader classes), so there is nothing to
 * forward to.  This adapter therefore does NOT fake a mesh: it sets the
 * reader's own -1 ("the mesh could not be created", reported by the caller as
 * MSGERROR 3153 with an ANMODE error, i.e. the starter run fails loudly) and
 * leaves every other field at the caller's zeros.  A deck that uses
 * /ALE/STRUCTURED_MESH therefore fails with an explicit, traceable error
 * instead of silently getting a mesh of zeros; a deck that does not use the
 * keyword never calls this function at all (hm_s_ale.F90:106-108 counts the
 * options first, and the call sits inside `if(len_trim(key3)==0)`), so no other
 * deck is affected.
 *
 * message_value is 0-based here: Fortran message_value(38) is index 37.
 * =========================================================================*/
#define P0_SALE_MESH_ERROR_INDEX 37   /* Fortran message_value(38) = error   */
#define P0_SALE_MESH_ERROR_CODE (-1) /* reader's own "mesh not created" code */

void cpp_sale_mesh_create_(int *message_value)
{
    if (message_value == NULL) return;
    message_value[P0_SALE_MESH_ERROR_INDEX] = P0_SALE_MESH_ERROR_CODE;
}

void cpp_sale_mesh_create(int *message_value)
{
    cpp_sale_mesh_create_(message_value);
}
