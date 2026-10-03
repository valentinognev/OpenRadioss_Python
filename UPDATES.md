# Updates

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