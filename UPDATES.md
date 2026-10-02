# Updates

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