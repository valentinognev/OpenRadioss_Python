# pyradioss full-port program — plan index

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:subagent-driven-development` to execute these phase files
> task-by-task. Steps use checkbox (`- [ ]`) syntax. **Model: space-bunny.
> Never dispatch a Fast variant** — see `00_ORCHESTRATION.md` §2.

**Goal:** take `pyradioss` from its current state (breadth-complete, depth
unproven) to a **literal, line-for-line Python port of OpenRadioss**, with every
ported formula traceable to the exact upstream Fortran file and every physics
path backed by differential evidence against the real Fortran solver.

**Architecture:** 18 phases in a dependency-ordered wave graph. Phase 0 builds
the Fortran oracle (nothing downstream can claim parity without it); Phase 1
clears the structural blockers (licence, platform, monoliths); Phases 2–16 port
one upstream subsystem each; Phase 17 is the convergence gate — the 529-deck
official corpus and the parity ledger.

**Tech stack:** Python ≥ 3.9 (dev box: 3.14.6), NumPy ≥ 1.22 (dev box 2.5.2),
SciPy (implicit/linear algebra), optional numba / mpi4py / matplotlib,
pytest 9.x + pytest-xdist. Fortran oracle: gfortran + cmake + silecc + extlib v82.

**Upstream reference (read-only):**
`/home/valentin/Projects/OpenRadioss/OpenCourant` — **AGPL-3.0-or-later**.
Called `$OR_SRC` below.

---

## 0. Read this first

| File | Purpose |
|---|---|
| **`plan/00_ORCHESTRATION.md`** | The contract every phase obeys: global constraints, environment variables, wave + worktree + reviewer protocol, the shared-file ownership map, Definition of Done, Review Focus. **Read fully before executing any phase.** |
| `AGENTS.md` | Repository rules. **Rewritten platform-neutral by Task P1.0** — until that lands, prefer this plan's environment contract (§2 of `00_ORCHESTRATION.md`). |
| `docs/STATE.md` | Current status + milestone history. Authoritative on *what exists*, stale on *what is validated*. |
| `VALIDATION.md` | Parity/coverage/perf evidence per milestone. **The M41 edition (§3.6, §4.10, §6.5, §7) is the last complete measurement**; anything after M41 is unmeasured. |
| `docs/OPEN_BUGS.md` | Known-unfixed defects. Several are Phase-gated inputs. |

## 1. Scope decision (recorded 2026-10-02, maintainer)

"Full port" = **literal line-for-line port** of OpenCourant's `starter/source`,
`engine/source`, `common_source` and `reader/`. That includes:

- every element family and every per-family routine set, not only the ones the
  corpus happens to exercise;
- every material law, failure model, equation of state, viscosity model;
- every contact interface type, including the ones no deck currently uses;
- **binary output formats** — T01, A-files, H3D, PARITH, RES, restart — because
  a literal port that only emits CSV/VTK is not a literal port;
- the full MPI/SPMD exchange layer;
- the `reader/` SDI layer used by LS-PrePost.

**Explicitly in scope but currently absent upstream:** the port's own implicit /
modal / spectral-fatigue tower is *kept and frozen* (see `00_ORCHESTRATION.md`
§3). It is not extended, and it is not the parity target; it must simply stay
green. Phase 15 exists only to bring the implicit branch to parity with the
upstream `engine/source/implicit`, and to give the frozen tower tangent coverage
for the element families Phase 15's new kernels introduce.

## 2. Size and honesty

| Upstream area | Files | Fortran LOC |
|---|---:|---:|
| `engine/source/elements` | 858 | 251,191 |
| `engine/source/interfaces` | 439 | 211,380 |
| `engine/source/output` | 423 | 180,559 |
| `engine/source/materials` | 401 | 153,327 |
| `engine/source/mpi` | 257 | 111,955 |
| `engine/source/implicit` | 38 | 51,269 |
| `starter/source` (all) | ~1,100 | ~380,000 |
| `reader/` (SDI, C++) | 368 | 93,360 |
| `common_source` (all) | ~400 | ~30,000 |
| `hm_cfg_files` (input schemas) | 2,485 | 819,434 (data, not code) |
| **Total code** | **≈ 4,300** | **≈ 1,513,000** |

`pyradioss` today is 459,569 Python LOC across 432 modules, with 14,267
non-slow tests collected. **Breadth is close to done; depth is not.** 571
material registry keys exist but only ~59 numeric laws are registered; 22 of 25
contact types have a module; output is 752 LOC against 180,559 Fortran lines.

This is a **multi-year program**. The plan is honest about that: each phase
carries a size band, and Phases 2–5 (elements), 8 (contact), 12 (output) and 16
(SDI) are the long poles. Re-ordering is a legitimate maintainer decision —
`00_ORCHESTRATION.md` §7 documents which phases are independent and can be
swapped.

## 3. Phase index

| # | Phase | File | Upstream scope | Band | Depends on |
|---|---|---|---|---|---|
| 0 | Oracle + licensing | `01_phase0_oracle_and_licensing.md` | build infra, licence | S | — |
| 1 | Foundation | `02_phase1_foundation.md` | structural blockers | M | 0 |
| 2 | Elements — solids | `03_phase2_elements_solid.md` | `elements/solid*`, `solid_2d`, `thickshell` | **XL** | 1 |
| 3 | Elements — shells | `04_phase3_elements_shell.md` | `elements/shell`, `sh3n` | **L** | 1 |
| 4 | Elements — 1-D | `05_phase4_elements_1d.md` | `truss`, `beam`, `spring`, `joint`, `rivet`, `xelem` | L | 1 |
| 5 | Elements — special | `06_phase5_elements_special.md` | `sph`, `xfem`, `ige3d`, `elbuf` | L | 1, 2 |
| 6 | Materials | `07_phase6_materials.md` | `materials/mat/*`, `mat_share` | **XL** | 1, 2, 3 |
| 7 | Failure + EOS + visco | `08_phase7_fail_eos_visc.md` | `materials/fail`, `eos`, `visc`, `tools` | L | 6 |
| 8 | Contact | `09_phase8_contact.md` | `interfaces/*` both trees | **XL** | 1, 2, 3 |
| 9 | Constraints, loads, properties | `10_phase9_constraints_loads_props.md` | `constraints`, `loads`, `properties`, `boundary_conditions` | L | 1, 2–5 |
| 10 | Starter pipeline | `11_phase10_starter_pipeline.md` | `starter/source/model`, `initial_conditions`, `restart`, `spmd`, `elbuf_init` | **L** | 1, 9 |
| 11 | ALE, airbag, multifluid, FSI | `12_phase11_ale_airbag_fsi.md` | `ale`, `airbag`, `multifluid`, `ams`, `fluid` | **L** | 2, 3, 6, 7, 8, 10 |
| 12 | Output | `13_phase12_output.md` | `output/*` both trees | **XL** | 1, 2, 3, 6, 7, 8, 9, 10 (+11 for PARITH) |
| 13 | SPMD / MPI | `14_phase13_spmd_mpi.md` | `engine/source/mpi`, `starter/source/spmd` | **L** | 2–5, 8, 12 |
| 14 | Implicit parity | `15_phase14_implicit_parity.md` | `engine/source/implicit` | M | 2–5, 8 |
| 15 | Common modules | `16_phase15_common_modules.md` | `common_source/*` | M | 2–14 (callers exist) |
| 16 | SDI reader, tools, QA | `17_phase16_sdi_tools_qa.md` | `reader/`, `tools/`, `qa-tests/` | **XL** | 0 |
| 17 | Convergence gate | `18_phase17_convergence.md` | — (measurement) | L | all |

Bands: **S** ≤ 1 week · **M** 2–4 weeks · **L** 1–3 months · **XL** 3–9 months.
Bands are per-phase totals for **one** implementer; a phase's waves are sized so
3–5 agents can run concurrently inside a wave.

## 4. Dependency graph

```
                          ┌──────────────────┐
                          │ P0 oracle+licence│  (gate: nothing starts before this)
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │ P1 foundation    │  (gate: platform-neutral, monoliths split)
                          └──┬────┬────┬─────┘
       ┌────────────┬─────────┘    │     │         ┌──────────┐
       │            │              │     │         │          │
  ┌────▼────┐  ┌────▼────┐  ┌──────▼──┐  │    ┌────▼────┐ ┌───▼─────┐
  │ P2 solid│  │ P3 shell│  │ P4 1-D  │  │    │ P6 mats │ │ P9 cons │
  └────┬────┘  └────┬────┘  └────┬────┘  │    └────┬────┘ └────┬────┘
       │            │            │       │         │           │
       │            │            │  ┌────▼────┐    │           │
       │            │            │  │P5 special   │           │
       │            │            │  └────┬────┘    │           │
       └────────┬───┴────────────┴───────┼─────────┴───────────┘
                │                        │
        ┌───────▼────────┐      ┌────────▼───────┐
        │ P8 contact     │◄─────┤ P7 fail/eos    │◄──── P6
        └───────┬────────┘      └────────────────┘
                │
   ┌────────────┼────────────┬────────────────┐
   │            │            │                │
┌──▼───────┐ ┌──▼──────┐ ┌───▼─────┐  ┌───────▼──────┐
│P11 ale/  │ │P13 spmd │ │P12 out  │  │P15 common    │
│airbag/fsi│ │  /mpi   │ │         │  │  modules     │
└──────────┘ └─────────┘ └─────────┘  └──────────────┘
                             │              ▲
                             ▼              │
                      ┌──────────────┐       │
                      │ P10 starter  │───────┘
                      │   pipeline   │
                      └──────────────┘

  P14 implicit parity depends on P2–P5 + P8 (needs all tangents)
  P16 SDI/tools/QA depends only on P0
  P17 convergence depends on ALL
```

**Independent tracks that can be re-ordered or run concurrently** (each in its
own worktree): P2/P3/P4/P6/P9 all start together after P1. P16 depends only on
P0, so it can run from day one. P5 depends on P2. P13 is last among the heavy
ones because it must exchange every per-family array.

## 5. How a phase file is structured

Every one of the 18 phase files uses the identical skeleton, so an agent that
has read one has read all of them:

```
Header (goal / architecture / tech stack / upstream ref)
Scope table        — exact upstream paths, file counts, LOC, Idispatch values
Gap analysis       — what pyradioss already has, what is missing, what is stubbed
Wave graph         — which tasks may run concurrently, in what order
Tasks              — numbered, each with Files / Interfaces / checkbox Steps
Reviewer checklist — what the per-task reviewer must actually verify
Parallelisation    — the ownership map slice for this phase
Exit gate          — the exact command(s) that must pass before the phase merges
```

## 6. Conventions used throughout

- **Task IDs** are `P<phase>.<task>` — e.g. `P2.7`. Test files follow the
  existing convention `tests/test_p2_<slug>.py` (phase-prefixed, replacing the
  milestone-prefixed `test_mNN_<slug>.py` for program-era work; the milestone
  number is retained in the docstring).
- **Milestones.** The existing series ends at M620. Program-era milestones are
  reserved as **M700 +**, one per green task that a maintainer may want to
  release independently. Milestone number and phase task number are unrelated.
- **Fortran citations** are repo-relative paths under `$OR_SRC`, with a line
  range and the routine name, e.g.
  `` `engine/source/elements/solid/solide/sforc3.F:1-400` ``. Never a bare
  filename, never a Windows path.
- **Every ported formula cites upstream.** A formula without a citation is a
  review rejection, not a nit.
- **Numerics changes require evidence**: the targeted tests **plus** a parity run
  against the Fortran oracle (Phase 0 output). "The suite is green" is not
  evidence for a physics change.
- **Never weaken a test to make a change land.** A realigned test names itself
  and the numbers in the commit message.
- **Energy accounting is load-bearing.** Every new force path books its work in
  the EN ledger; the energy-balance tests are the leak detector.
- **Mark any test ≥ 60 s serial with `@pytest.mark.slow`.**
- **Backend pinning.** Every before/after comparison pins
  `PYRADIOSS_BACKEND=numpy`. numba parity is proven separately and must stay
  bit-identical where it already is.

## 7. Re-ordering, deferring, cutting

`00_ORCHESTRATION.md` §7 records which phases are independent. If the
maintainer wants a cheaper milestone, the levers in order of value-per-effort
are:

1. **P0 + P1** — small, and they unblock everything else. Do not skip.
2. **P8 §Wave 3 (interface types with no corpus deck)** — 22 types have a module
   today; the missing TYPE4/13/19 and the `intsort` broad-phase parity are the
   cheapest correctness wins.
3. **P6 restricted to laws with a corpus deck** — the ~59 registered laws before
   the full 121-directory sweep.
4. **P12 restricted to T01 binary + PARITH** — the two formats external tooling
   actually reads; H3D can follow.

What must **not** be cut: P0 (no parity evidence without it), P1 Task **P1.0**
(licence), P17 (otherwise there is no definition of "done").

## 8. Known status inputs this plan is built on

These were measured on 2026-10-02 and are the plan's baseline:

- `python -m pytest --collect-only -m "not slow"` → **14,267 collected**
  (20 deselected).
- Dev box: Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, pytest 9.1.1,
  matplotlib 3.11.1. **numba and mpi4py are NOT installed** on this box — every
  numba/mpi code path is therefore currently untested locally and is a Phase
  0 install item.
- Corpus coverage at M41 (last full sweep, **stale**):
  529 decks → 13 CLEAN / 440 SKIPS / 76 ERROR / 0 CRASH / 0 TIMEOUT.
- Parity at M41: five RD-E-1000 shell cases MATCH; the BT family is DEVIATION
  0.138–0.257 and is characterised as *unreachable by comparison* because the
  reference itself diverges; DKT18 is 0.2581–0.4556 and is the cleanest
  remaining RD-E-1000 lead.
- **13 tests were red at handover** (M614, `test_m6`/`m12`/`m14`/`numpy_compat`)
  and are counted as pre-existing, not regressions.
- `pyradioss/input/starter_keywords.py` is **97,102 lines in one file** — a
  structural blocker, split by Task P1.5.
- Dozens of `SyntaxWarning: "\O" is an invalid escape sequence` warnings fire at
  import from docstrings citing `C:\OpenRadioss\...` — fixed by Task P1.1.

## 9. Where the machine-generated census lives

**The LOC and file counts in these phase files were measured on 2026-10-02 by
direct inspection of `$OR_SRC`**, not by `tools/census.py` — that tool does not
exist yet. They were taken with `find`/`wc -l` over the upstream tree and are
reproducible; where a count here disagrees with the tree, **the tree wins and the
phase file gets corrected.**

**Task P1.3** then makes the census permanent and machine-readable, so the
counts stop being a snapshot. That task emits:

- `tools/validation_data/census.json` — per upstream file: path, LOC, `!||`
  call-graph edges, owning Python module (or `null`), port status
  (`ported` / `partial` / `stub` / `missing`).
- `plan/CENSUS.md` — the rendered table, regenerated on demand.

Until P1.3 lands, **`plan/CENSUS.md` does not exist** and the phase files'
tables are the only census.

Phase files cite the census for the *shape* of a subsystem and then name the
individual files a task must port. If the census and a phase file disagree, the
census wins and the phase file gets corrected — that correction is itself a
task.