# Updates

## 1.3.0 - P0.10: the binary T01 reader, and one comparable number

- **New** `tools/compare_t01.py` — `read_t01(path) -> T01` decodes the Fortran
  engine's binary time-history file (`ITTYP==3` Radioss IEEE: big-endian 4-byte
  record markers, header int32, single-precision channel values) into named
  channels, sample times and an `(n_times, n_channels)` matrix, plus the header
  facts (format code, title width, unit-scaling triples, the located 24-byte
  `ctime` run stamp). Every structural constant is listed in a machine-readable
  `LAYOUT` table with the upstream `file:line` range and a pattern that must
  still be found there, so a constant cannot drift from `$OR_SRC` unnoticed.
  The record framing is **imported** from `tools/oracle/oracle_selftest`, not
  re-implemented: one record walk in the program.
- **New** `score(ref, port) -> Score` — per-channel
  `rel_rms = sqrt(mean((port-ref)^2)) / max(|ref|)` with the `MATCH_RMS = 0.05`
  threshold pinned to `parity_m41.json`'s `tolerance_rel_rms`, so a verdict here
  is comparable with the M36..M41 tables. Channels are matched **by name**; a
  channel only one side has is `NODATA` and never drives `worst`, and neither
  does a channel below 1 % of its group's dominant reference peak (the
  historical harness's own significance rule — without it the transverse
  momentum of a uniaxial test reads as `rel_rms ~ 0.5`).
- **New** `read_port_csv(path) -> T01` — the port's ASCII T01 as the same shape,
  with its column names folded onto the upstream short names (`MOMX -> XMOM`,
  `EW -> EFW`, `P1_IE -> P1_1`) so the two series can be matched by name.
- **New** `tests/test_p0_compare_t01.py` — 20 tests. The reader is validated
  three ways, because upstream's own converter (`th_to_csv`) is unobtainable on
  this box: against the cited upstream sources, against the committed golden
  T01 of the P0.5 reference run (channel set, 100 samples, per-channel maxima,
  all exactly), and against the port's own CSV for the same deck (significant
  channels agree to ≤ 0.2 % rel-RMS; the round-off channels to an absolute
  bound). A fourth test runs the brief's `th_to_csv` cross-check **if** the
  converter ever appears and skips with a reason naming every place tried.
- **Known limitation, stated not hidden:** the shared record walk fixes the
  per-step record count at four, so a deck that also requests `/TH/SUBSET`
  curves is refused loudly rather than mis-parsed; a test pins that refusal.

## 1.2.1 - P0.9: the validation harness runs on any box, and cannot be fooled into writing wrong H3D

- **Fixed** `tools/validate_vs_fortran.py` no longer names one machine:
  `OR_ROOT` / `ONEAPI` / the three executable paths are gone, replaced by
  `oracle_paths() -> {starter, engine, th_to_csv, h3d_lib, hm_reader_lib}`,
  resolved **per call** through `pyradioss.paths` — so importing the module
  touches no filesystem and works with nothing configured. An unresolved
  resource is reported as `None` + a warning listing every candidate
  (`paths.missing_resource`), and `parity` exits 2 with the full diagnostic
  rather than writing a results file with nothing compared in it.
- **Fixed** all **four** of upstream's h3d dlopen routes are now closed, not
  one: `h3dlib_load_` (`h3d_dl.c:616-923`) tries `$RAD_H3D_PATH` (`:623-632`),
  the working directory (`:634-644`), `$ALTAIR_HOME/hwsolvers/common/bin/$ARCH`
  (`:647-658`) and a bare `dlopen` fed by the loader path (`:660-666`). The
  harness drops the three variables, drops (and loudly names) any
  `LD_LIBRARY_PATH` entry holding the writer, refuses to prepend a reader
  directory that holds it, and refuses a scratch directory that holds it.
  With all four closed the writer stays unreachable and `genh3d.F:728-732`
  turns `*IERROR = 1` (`:920-922`) into the loud MSGID 274 refusal.
- **Fixed** `tools/benchmark_rad_db.py` and `tools/compare_t01_tab1.py` no
  longer hardcode a harvested-corpus path or a per-session scratchpad; both
  take an environment variable (`RAD_EXAMPLES_DB`, `TAB1_BASE_DIR`) and fail
  loudly, naming what was tried. `benchmark_rad_db.py` also got the
  `sys.path` seam its CLI needed.
- **Changed** the console table distinguishes a missing `th_to_csv` from a
  failing solver (`FORTRAN-FAIL(th2csv-missing)`), additively: the two
  pre-existing statuses still print the bare `FORTRAN-FAIL` string that
  `tools/validation_data/parity_m41.json` is keyed on.
- **New** `tests/test_p0_harness_portable.py` — 27 tests (from 15): import
  resolves nothing and needs no temp-dir probe, `read_deck` and the workdir
  default are lazy, the four h3d routes stay closed, the loader-path scrub and
  the working-directory guard bite, the invocation asymmetry (`-np 1` starter,
  `-nt 1` engine, never `-np`) is pinned at argv *and* driver level, and no
  `tools/*.py` spells a Windows drive letter outside a named, justified
  exemption. Oracle-dependent tests skip when it is absent and fail under
  `PYRADIOSS_ORACLE_REQUIRED=1`.

## 1.2.0 - P0.5: golden reference run + the oracle's determinism, proved

- **New** `tools/oracle/oracle_selftest.py` — runs the oracle built by
  `tools/oracle/build_oracle.sh` on `examples/tensile_bar` (starter `-np 1`,
  engine `-nt 1`, one thread, `RAD_H3D_PATH` unset), parses
  `<run>_0001.out` for the `ENGINE TERMINATION` banner and
  `TOTAL NUMBER OF CYCLES`, md5s `<run>T01` and decodes it. Ships a pure-Python
  `ITTYP==3` T01 reader (big-endian Radioss IEEE records, float32 values) so the
  stored numbers can be re-derived with no oracle installed.
  `--write` regenerates the record and the golden artefacts.
- **New** `tools/validation_data/oracle_smoke.json` — the seed of every later
  parity comparison: deck path + sha256, oracle binary sha256s and the
  `oracle_provenance.json` pointer, the starter/engine argv, `NORMAL
  TERMINATION`, 1420 cycles, T01 size + md5, per-channel maxima for the 23
  global channels (`write_thnms1.F90:228-250`) and the 2 part channels
  (`varpa_title`), wall time, the `RAD_H3D_PATH`-unset fact, and a
  `not_established` block where every null carries a reason (th_to_csv is not
  built on this box; H3D, the starter include-file list, native `.k` reading,
  `/ALE/STRUCTURED_MESH` and `/CHECKSUM_REPORT` over H3D are inadmissible).
- **New** `tests/data/oracle_smoke/{TENSILET01,TENSILE_0001.out}` — the golden
  run's admissible artefacts, committed so the structural tests can verify the
  record without a solver.
- **New** `tests/test_p0_oracle_selftest.py` — 7 tests: 4 always-run structural
  ones (record completeness, deck bytes, maxima re-derived from the committed
  T01, determinism evidence) and 3 oracle ones (reproduction, bit
  reproducibility, differing-bytes-inside-the-run-stamp). Oracle absent → skip
  with an actionable reason; `PYRADIOSS_ORACLE_DISABLED=1` /
  `PYRADIOSS_ORACLE_REQUIRED=1` behave exactly as in
  `tests/test_p0_oracle_build.py`. ~1.3 s, so default tier, no `slow` marker.
- **Finding — the raw T01 is not byte-reproducible, and that is upstream, not a
  defect.** `hist1.F:211` writes `ctime()` (`timer_c.c:30-40`) into the T01
  header unconditionally; no keyword and no environment variable suppresses it.
  Measured: runs sharing a wall-clock second have an identical raw md5, a run
  one second away differs in **one** byte (the seconds digit), and with those
  24 bytes zeroed the digest is identical every time
  (`e3688899358f35e825cd640f9bd94964`). The anchor is therefore
  `deterministic_md5()` — the stamp-normalised digest — and the gate asserts
  *both* halves (equal normalised digests **and** every varying raw byte inside
  the 24-byte window), failing rather than skipping when it is violated.
- **Deck choice** `examples/tensile_bar`, not a vendored RD-* deck: it is the
  only small deck runnable with one thread that does not request `/H3D` (the
  oracle refuses H3D by design, so every `/H3D/DT` deck aborts with MSGID 274),
  it costs 0.30 s + 0.08 s per run, and `oracle_provenance.json` already records
  it as the measured evidence for the T01 / A-file / RESTART / energy channels.

## 1.1.3 - P0.8 fix round 3: the remedy must not damage the artifact

- **Fix (I1a)** the manifest's corpus-dependent assertions now read
  `tools.validate_vs_fortran.manifest_corpus_root()` — the corpus **the
  manifest describes** (the in-tree vendored `tests/data/rd_decks`) — instead
  of `paths.rd_decks_dir()`, which the environment selects. Consequences:
  **0 skipped tests in every configuration**, including the sanctioned
  `PYRADIOSS_RD_DECKS=<full E: extract>` setup, and the live-vs-described
  relationship stays visible as a **report, never a failure**
  (`test_live_corpus_report_*`, emitted as a `RuntimeWarning` on every run).
  Every record's `sha256` is still re-hashed from the described corpus on every
  run, so the "does the described corpus still hash to the record?" property is
  kept, not traded away. Pinned by `test_described_corpus_is_the_vendored_one`
  (asserted *under* an override) and `test_this_module_has_no_skip_markers`
  (AST check — no conditional skip may be reintroduced).
- **Fix (I1b)** the loud-failure text no longer prescribes a harmful command.
  It states the committed manifest's corpus (path + `sha256:…`), the live
  corpus (path + `sha256:…`) and how many of the live corpus' decks have no
  record, then **branches**: the vendored corpus changed → re-hash *it* with
  `PYRADIOSS_RD_DECKS` unset (`--root tests/data/rd_decks`); a wider extract is
  wanted → write it to an explicit `--out <private path>` and do **not** commit
  it over `tools/validation_data/rd_decks_manifest.json`.
- **Fix (I1c)** `tools/build_rd_decks_manifest.py` now **REFUSES** (exit 2) to
  write the committed manifest from any non-vendored root — `write_refusal()`
  fires for both the bare command under `PYRADIOSS_RD_DECKS` and
  `--root <live dir>`; an explicit `--out` or `--allow-nonportable` is required.
  `--check` is unaffected (verifying a foreign corpus is harmless). Without this
  the prescribed fix was a dead end *and* it turned 1 failure into 6–8.
- **Fix (N8)** the `--check` corpus-fingerprint message derives its record count
  instead of hardcoding "not 75".
- **Fix (N6/N7/N9)** `UPDATES.md`'s "46 (was 30)" corrected to the collected 47
  (the round-3 count is 56, measured); the test module's docstring rewritten —
  it described a vendored-path skip predicate and repeated the harmful remedy;
  the round-2 report's "7 tests" corrected to the measured 8.
- **Tests** `tests/test_p0_manifest.py` 56 (was 47), **0 skipped** in the
  default, byte-identical-override and foreign-extract configurations.
  Mutation-checked on copies: `write_refusal()` neutered, `main()` ignoring the
  refusal, the hardcoded "not 75", the report prescribing the live-corpus
  re-hash, and `_described_corpus()` following `PYRADIOSS_RD_DECKS` — all five
  killed.
- **Process note** writing the refusal tests *before* the refusal existed made
  one of them clobber the committed manifest with a synthetic corpus (restored
  from git immediately). Those tests now snapshot and restore the committed
  file through a `committed_manifest` fixture, so a regression of the guard
  cannot leave the artifact damaged.

## 1.1.2 - P0.8 fix round 2: the manifest says only what it can support

- **Docs (M5)** `tools/build_rd_decks_manifest.py`'s header claim "the manifest
  closes that gap" was **false** and contradicted its own `PROVENANCE_CAVEAT`:
  the manifest binds *future* runs to bytes; the M41 verdicts stay bound to a
  scratchpad extract that no longer exists (hence
  `parity_run_deck_bytes_verified: false` everywhere). Rewritten to say
  exactly that. `UPDATES.md`'s I2 bullet no longer claims "all six" in-envelope
  decks skip the five control families (5 do; the 6th skips `/TH/BRIC` alone,
  and none reports `ERROR`), and "21-field" became 24.
- **Fix (N1)** the loader's *undocumented extra field* rejection is now covered:
  `test_loader_rejects_an_undocumented_record_field` (+ the missing-field and
  unknown-schema mirrors). Neutering the branch used to leave the suite green.
- **Fix (N2)** `--check` now compares the header prose — `hash_algorithm`,
  `hashed_file_rule`, `envelope_rule`, `bytes_verified_rule`, `joins`, `notes`
  (`PROSE_KEYS`), including the envelope-qualification note generated from the
  records. Deleting or rewording a rule a reader quotes is drift; it used to
  pass silently.
- **Fix (N3)** `resolve_manifest_record(rec)` **requires**
  `corpus_fingerprint`: a hand-built record without it used to bypass the
  corpus guard in silence — the one input shape where the I1 protection
  switched off unnoticed. It is now an explicit `ValueError`, with an explicit
  `require_fingerprint=False` opt-out that waives the corpus *binding* only
  (the file is still re-hashed).
- **Fix (N4)** a one-byte corpus change no longer prints 75 `deck changed`
  lines: the corpus-level fact is stated once ("corpus fingerprint differs:
  X -> Y … every record's `corpus_fingerprint` moves with it, which is this one
  fact, not 75") and per-deck lines list only genuine per-deck fields.
- **Behaviour change (R1)** the whole-corpus tests now skip only when
  `rd_decks_dir()` is a corpus the manifest does **not** describe. A
  **byte-identical** copy elsewhere (`PYRADIOSS_RD_DECKS`) is recognised by the
  fingerprint, so the tests RUN and pass there. A *different* extract skips
  them but fails the never-skipped
  `test_live_corpus_is_the_corpus_the_manifest_describes`, which names both
  fingerprints and the command that re-hashes — eight silent skips are no
  longer possible.
- **Tests** `tests/test_p0_manifest.py` 47 (was 30; the 46 I first wrote
  here was not the collected count — run the number, do not assert it).
  copies in `/tmp/opencode`: an extra record key, a deleted
  `bytes_verified_rule`/`envelope_rule`/`notes`, and a record built without
  `corpus_fingerprint` are each rejected.

## 1.1.1 - P0.8 fix round 1: records bound to their corpus, envelope qualified

- **Fix (I1)** a manifest record is now bound to the corpus it was hashed from:
  every record carries `corpus_fingerprint`, the header is reachable through the
  new `load_manifest_doc()`, and `resolve_manifest_record(rec, root=None)`
  resolves against the manifest's **own** declared corpus (never
  `rd_decks_dir()`) and raises `ValueError` when the corpus fingerprint differs
  — so a `PYRADIOSS_RD_DECKS` override can no longer join a verdict to a
  different file at the same relative path. New `corpus_fingerprint(root)`:
  `sha256:<hex>` over the sorted `<relpath>\t<sha256>` lines of every
  `*_0000.rad`, path-independent by construction.
- **Fix (I2)** records now carry `skipped_families` (with each family's class),
  `coverage_hard_skips`, `coverage_blockers` and `coverage_degrade_warnings`
  from `coverage_results_m41.json`. This matters because `coverage_verdict`
  counts only NON-control skips: a `SKIPS(2)` row hides five control-class
  families. Five of the six in-envelope decks (BATOZ Sf_0.6/0.8/0.9, QEPH
  Sf_0.8/0.9) report `SKIPS(2)` while skipping **seven** families —
  `/ANALY`, `/DEF_SHELL`, `/DEF_SOLID`, `/IOFLAG`, `/SPMD` (class `control`)
  plus `/TH/RBODY`, `/TH/SHEL` (soft); the sixth
  (`RD-V-0700/…/HEXA_ELEM_SOLID_18`) reports `SKIPS(1)` and skips `/TH/BRIC`
  alone. No in-envelope deck reports `ERROR`. `in_envelope_reason` now names
  each deck's skips, so the envelope claim cannot be read unqualified.
- **Fix (I3)** new per-record `parity_run_deck_bytes_verified` /
  `coverage_run_deck_bytes_verified`, **false on all 75**: the M41 sweep ran
  from a session scratchpad that no longer exists
  (`.agents/skills/validation-compare/SKILL.md:56-59`) and the official parity
  rows carry no deck path, so each record pairs the hash of the re-vendored
  copy with a verdict measured on the scratchpad copy. Both flags are
  **derived** (`deck_bytes_match()`: does the evidence row name an existing
  file whose bytes hash to ours?) so a future in-place run flips them itself;
  the manifest notes and `bytes_verified_rule` say so.
- **Fix (M1)** `build_rd_decks_manifest.py --check` now compares what the corpus
  **contains** (fingerprint, counts, records) and ignores this invocation's
  provenance (`generated`, `corpus_root.source` / `.vendored` /
  `.resolved_at_generation`), so it is portable: a byte-identical corpus at any
  path verifies clean, while a changed deck still fails.
- **Fix (M2)** `MANIFEST_FIELDS` is now the complete record schema —
  24 fields (was 15: it omitted `parity_provenance` and
  `inventory_classification_strict`, and the loader only complained about
  *missing* fields, never about *extra* ones).
- **Fix (M3)** the manifest tests' skip condition is derived from
  `paths.rd_decks_dir()` vs this checkout's `tests/data/rd_decks`, not from the
  manifest header, plus a never-skipped `test_manifest_file_exists()` — a
  missing manifest now fails instead of skipping five tests with a misleading
  reason.
- **Fix (M4)** `counts` declares `measured` (24) and `unmeasured` (51)
  explicitly, next to `in_envelope_with_control_skips` and the two
  `*_deck_bytes_verified` totals, instead of leaving them to subtraction.
- **Tests** `tests/test_p0_manifest.py` 30 (was 15). Mutation-checked against
  copies in `/tmp/opencode`: emptying an in-envelope record's
  `skipped_families` (killed), claiming
  `parity_run_deck_bytes_verified: true` (killed), making
  `resolve_manifest_record` follow `PYRADIOSS_RD_DECKS` (killed) and also
  dropping the fingerprint guard (killed, 2 tests), restoring the wholesale
  `corpus_root` comparison in `--check` (killed).

## 1.1.0 - Hashed corpus manifest (P0.8)

- **New** `tools/validation_data/rd_decks_manifest.json` — one record per
  starter deck (`*_0000.rad`) under `paths.rd_decks_dir()`: `deck`,
  `hashed_file`, `sha256`, `size_bytes`, `case_id`/`category`/`package`
  (carried over from `inventory.json`, `null` when it never catalogued the
  deck — ids are never invented), the `parity_m41.json` /
  `coverage_results_m41.json` verdicts, and `in_envelope` **with**
  `in_envelope_source` + `in_envelope_reason`.
- **`in_envelope` is measured, never predicted**: `true` only where
  `parity_m41.json` carries a `MATCH` verdict for that exact `case_id` (the
  port reproduced the oracle). `inventory.json`'s static `IN_ENVELOPE`
  classification (a *reading* prediction) is recorded as context and never
  flips the flag. Every `false` carries its reason.
- **New** `tools/validate_vs_fortran.load_manifest()` → `list[dict]`; lazy and
  read-only (reads the JSON inside the function, opens no deck), so it works
  with no environment exported and no corpus mounted. Raises `ValueError` on an
  unknown schema or a record missing a documented field.
- **New** `tools/build_rd_decks_manifest.py` — regenerates the manifest from
  the corpus, and `--check` verifies the committed file against the decks on
  disk without writing.
- **Tests** `tests/test_p0_manifest.py` (15): one record per corpus starter
  deck, **every** recorded `sha256` re-hashed from disk on every run (75 decks
  / 8.8 MB, ~10 ms — the whole corpus, not a sample), unique ids that match
  `inventory.json`, the envelope flag only ever backed by a measured `MATCH`,
  non-empty joins with `parity_m41.json` and `coverage_results_m41.json`, the
  loader working with nothing exported, and a synthetic uncatalogued deck
  coming out `case_id: null` / `in_envelope: false`.

## 1.0.2 - P0.6 fix round 1: CI cfg spelling, candidate order, pinned precedence

- **Fix** `PYRADIOSS_HM_CFG` now accepts **both** spellings — the documented
  tree root (`…/hm_cfg_files`, §4.1 and upstream's `RAD_CFG_PATH`) and the
  schema directory itself (`…/hm_cfg_files/config/CFG`, what
  `.github/workflows/ci.yml` exported). `mat_reader._find_cfg_root` decides by
  inspecting the filesystem (`<cfg>/config/CFG` exists?), never by string
  shape; a directory named `CFG` with no `radioss<version>` schemas is
  rejected loudly instead of accepted.
- **Out-of-map file edit** `.github/workflows/ci.yml` re-points
  `PYRADIOSS_HM_CFG` at the tree root (both jobs), so the workflow exports
  what §4.1 says the variable means. The old spelling still works in code.
- **Fix** the brief-only candidate `$OR_ROOT/OpenCourant/hm_cfg_files` no
  longer sits *before* the §4.1 rule-3 Windows candidate — every
  non-contract candidate is now strictly after rules 1–3.
- **Fix** `pyradioss.paths.is_cfg_tree` / `is_cfg_schema_dir`: a cfg
  candidate must carry the incremental `radioss<version>` subdirectories, so
  an empty or partial sparse checkout fails loudly instead of resolving.
- **Fix** `mat_reader.catalogue()` is keyed on the freshly resolved cfg root
  and rebuilds when it moves, so `paths.reload()` reaches the singleton that
  several test modules freeze at collection time.
- **Diagnostics** a *set but missing* variable now emits a `RuntimeWarning`
  (a stale `PYRADIOSS_RD_DECKS` could silently downgrade a validation run);
  an unresolvable `OR_ROOT` nests its own candidate list instead of printing
  a placeholder, and "unset" is no longer claimed for a variable that was set
  and merely absent.
- **Tests** `tests/test_p0_paths.py` 60 tests (was 35): every neighbouring
  precedence pair, loud failures for `or_build()`/`or_engine()`, both cfg
  spellings, the empty-`CFG` rejection, and the catalogue-follows-`reload`
  invariant. 7/7 mutants killed by a harness in `/tmp/opencode`
  (extras-first, rule2-before-rule1, Windows-candidate-deleted, no-op
  `reload()`, cfg-root-appends-`config/CFG`, frozen catalogue root,
  any-directory-is-a-cfg-tree).

## 1.0.1 - Single resource resolver (P0.6), LAW4 cfg bug closed

- **New** `pyradioss/paths.py` — the one place that resolves every external
  path (`or_src`, `or_root`, `or_build`, `or_starter`, `or_engine`,
  `hm_cfg_dir`, `rd_decks_dir`, `missing_resource`, `reload`), implementing
  the order of `plan/00_ORCHESTRATION.md` §4.1: env var (if set **and**
  existing) → sibling-of-build → Windows compat → **fail loudly** with every
  candidate listed. Upstream authority: `$OR_SRC/INSTALL.md:34-42` (the Linux
  env block: `OPENRADIOSS_PATH` / `RAD_CFG_PATH` / `RAD_H3D_PATH` /
  `LD_LIBRARY_PATH`) and `:110` (the `starter_linux64_gf` name). No import-time
  filesystem access; `import pyradioss.paths` cannot fail.
- **Convention** `missing_resource(name, tried)` *returns* a
  `FileNotFoundError` instance; resolvers `raise` it. Keeps one diagnostic
  usable by callers that must not abort (the `/MAT` reader logs it).
- **Fix (OPEN_BUGS item 6)** `pyradioss/input/mat_reader.py` no longer probes
  an import-time tuple of `os.environ` plus two hardcoded roots; the cfg tree
  comes from `paths.hm_cfg_dir()`. `tests/test_m535_law04.py` now passes on
  Linux (76 passed) — before: `no cfg schema found`, `E must be > 0`. The
  heuristic degradation is kept (a deck must still parse) but its warning now
  carries the full list of locations searched.
- **New** `tests/test_p0_paths.py` (35 tests) — every resolver per tier, the
  stale-env skip, the loud failure message, `reload()`, the vendored deck
  corpus resolved from any CWD, and `or_build()` never returning the
  read-only `$OR_SRC`.
- 22 cfg-dependent test modules that silently skipped on Linux now execute
  (previously `mat_reader.catalogue().schema("FABRI") is None`): 535 passed.

## 1.0.0 - Full-port program plan

The first structured plan for taking `pyradioss` from its current state to a
literal line-for-line port of OpenRadioss. 20 files in `plan/`, 200 tasks,
18 phases, sized against a measured upstream census.

**Scope decision (maintainer, 2026-10-02):** *literal line-for-line port* —
every element family, material law, failure model, EOS, contact type, the
binary output formats (T01, A-files, H3D, PARITH, restart), the full MPI/SPMD
exchange layer, and the `reader/` SDI layer. Not "parity on what the corpus
exercises".

**Findings that shaped the plan (measured 2026-10-02):**

- Breadth is nearly done, depth is not: 571 material registry keys but ~59
  numeric laws; 26 contact modules; 752 Python lines of output against 180,559
  Fortran lines; 2,582 of MPI against 112,000.
- **`pyradioss/elements/solid_tria3.py` cites `engine/source/elements/solid_2d/tria/` and four `.F` files that do not exist upstream** — the 2-D solid triangle is a port extension with a fabricated citation. Phase 2 Task P2.1.
- **Six `/FAIL` directories upstream are empty** (`changchang`, `composite`, `hashin`, `lemaitre`, `puck`, `spalling`) while the port has six modules for them. Phase 7 Task P7.1.
- **`failwave` is a failure-wave *propagation* model, not a damage-combination rule** — `MAXLEV_STACK`, `FWAVE_NOD_STACK`, `ERROR IN FAILWAVE PROPAGATION`. Phase 7 Task P7.8.
- **`/INTER/TYPE19` is real** but read by the *generic* reader `interf1/definter.F:190`, not by an `int19` directory — the directory count is evidence, not proof. Phase 8 Task P8.1.
- `mulawc.F90` names **51 laws with an upstream shell kernel**; the port's
  shell-capable set is a small fraction of it. Phase 6 Task P6.0.
- `starter/source/interfaces/inter3d1/` is **59,973 LOC** — the largest
  directory in the solver — and the port has no dedicated module for it.
  Phase 8 Task P8.7.

**Blocking issue found while planning — the licence contradiction:**

- `$OR_SRC/LICENSE.md` is **AGPL-3.0-or-later** (Siemens 2026 headers).
- `pyradioss/LICENSE` is **MIT** (© 2026 Minh Quang Pham).
- `README.md` claims **GPL-3.0**.

A literal port of AGPL code cannot ship under MIT. This is **Task P0.0 and it
blocks Phase 1 onward**; it enumerates four lawful resolutions and does not
choose one — that is a maintainer/legal decision.

**Environment change:** the working machine is now Linux with the upstream
source at `/home/valentin/Projects/OpenRadioss/OpenCourant`, **unbuilt**
(no extlib, no `exec/`, no `cmake_linux64_gf.txt` compiler flags). `AGENTS.md`
and `tools/validate_vs_fortran.py` are Windows-specific. Phase 0 builds the
oracle; Phase 1 Task P1.0 rewrites `AGENTS.md` platform-neutral.

**Parallel-execution design:** wave graph with one git worktree per wave, a
declared single-owner map for the 16 contended files, one reviewer per task,
one per phase, and a whole-program reviewer at Phase 17. All subagents and
reviewers run on **space-bunny**; no Fast variant, ever.

**Also frozen:** `pyradioss/implicit/` (~25k LOC the upstream Fortran does not
have — modal, complex-modal, PSD, spectral fatigue, NORTA, evolutionary
fatigue) is kept but quarantined behind extras, pinned by
`tests/test_p1_frozen_implicit.py`.

**Nothing in `pyradioss/` was changed.** The plan is the only artefact.