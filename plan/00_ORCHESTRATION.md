# 00 — Orchestration contract

> **For agentic workers:** this file is binding on every phase. Read it in full
> before executing `01_phase0_*` … `18_phase17_*`. **Model: space-bunny. Never
> dispatch a Fast variant** (see §2).

**Goal:** one place that states the rules every phase and every agent obeys —
environment, wave isolation, reviewer protocol, file ownership, and what
"finished" means — so that 18 phases executed by parallel agents cannot corrupt
each other.

**Architecture:** a wave graph of git worktrees, one owner per contended file,
one reviewer per task, one gate per phase. Every rule here is checkable; the
ones that are not checkable are marked *(unenforceable — reviewer must assert)*.

---

## 1. Global constraints

These are copied from the maintainer and the repository's own rules. Every
task's requirements implicitly include this section.

### 1.1 Domain rules (non-negotiable)

1. **Never guess Fortran semantics.** Every ported formula cites the exact
   upstream file under `$OR_SRC`, with a line range. Open the Fortran; do not
   recall it. **If the Fortran disagrees with your expectation, the Fortran
   wins.**
2. **Numerics changes require validation evidence**: the task's own tests
   **plus** a parity run against the real Fortran solver. "The suite is green"
   is not evidence for a physics change.
3. **Never weaken a test.** If a phase must realign a pre-existing test,
   name the test and justify it with numbers in the commit message and the
   walkthrough.
4. **Energy accounting is load-bearing.** Every new force path books its work
   in the EN ledger. The energy-balance tests catch silent leaks; a new force
   path that does not book is a review rejection.
5. **Mark any test ≥ 60 s serial with `@pytest.mark.slow`.**
6. **Pin the backend for before/after comparisons**: `PYRADIOSS_BACKEND=numpy`.
   numba parity is proven separately and must stay bit-identical where it
   already is.
7. **Commit at every green sub-step.** Session and quota interruptions are
   normal; committed work survives, uncommitted work may not.

### 1.2 Repository rules

- **READ-ONLY:** `$OR_SRC` (`OpenCourant`). Never write, patch, or generate
  inside it. Never run a build *in* it — build out-of-tree into `$OR_ROOT`.
- Every new Python module's docstring names its upstream Fortran origin
  (`engine/source/.../file.F`, with line ranges for the formulas ported).
- All element/material/contact kernels are **vectorised over element groups**,
  matching how the Fortran loops over groups. A new kernel that loops in Python
  per element is a review rejection unless it is a control-flow routine.
- Unknown keywords must degrade gracefully: warn, record the reason, skip. They
  must never abort the Starter on their own.
- Tests live in `tests/`, one file per phase task, named `test_p<phase>_<slug>.py`.

### 1.3 Licensing — **GATING, UNRESOLVED**

`$OR_SRC/LICENSE.md` is **GNU AGPL-3.0-or-later**; every source file carries a
Siemens 2026 AGPL header. The port's own state contradicts this three ways:

| Artefact | Says |
|---|---|
| `$OR_SRC/LICENSE.md` | AGPL-3.0-or-later (Siemens) |
| `pyradioss/LICENSE` | **MIT** (© 2026 Minh Quang Pham) |
| `pyradioss/README.md` | "GPL-3.0 … inherited from files derived from OpenRadioss, © Altair Engineering Inc." |

A literal port copies AGPL-covered expression into this repository. Shipping it
under MIT is not defensible. This is **Task P0.0 and it blocks Phase 1 onward.**
The task enumerates the four lawful resolutions; it does **not** pick one —
that is a maintainer/legal decision. Until it lands, agents may not add new
upstream-derived code beyond what already exists.

## 2. Model policy

- All implementers and reviewers run on **space-bunny** (the session model).
- **A Fast variant is forbidden for every role** — implementer, reviewer,
  explore subagent, whole-program reviewer. No exceptions, no fallbacks.
- If the only available subagent type is a Fast agent (`explore`), **do not
  dispatch it.** Report the intended model to the maintainer and ask. This has
  happened before in this repository and the correct answer was "do the
  research in the parent session".
- Name the intended model on every dispatch.

## 3. Frozen subsystem — `pyradioss/implicit/`

The port carries ~25k LOC of machinery that **upstream OpenRadioss does not
have**: implicit Newton/arc-length (M8–M15), modal + complex-modal
superposition (M16–M18), random/PSD response and response spectra (M19),
spectral fatigue (M20–M34), plus the GUI.

**Decision (maintainer, 2026-10-02): keep, but quarantine behind extras.**

Consequences, binding on every phase:

- **No new features** are added under `pyradioss/implicit/` except what Phase 15
  (`15_phase14_implicit_parity.md`) needs for parity with upstream
  `engine/source/implicit`.
- Every module there must remain **importable and green** through the whole
  program. A phase that breaks it is a phase that does not merge.
- The frozen surface is enumerated by name in
  `pyradioss/implicit/_frozen_surface.py` (created by Task P1.6). A test pins the
  list, so *adding* a module to the implicit package is a deliberate act with a
  review gate, and *deleting* one is a hard failure.
- The extras keep their existing `pytest.importorskip` guards for optional
  dependencies (CHOLMOD, MUMPS). Do not promote them to hard deps.

## 4. Environment contract (platform-neutral)

The repository's `AGENTS.md` was written for Windows 11 + a prebuilt
`C:\OpenRadioss`. Task P1.0 rewrites it. Until then, this section is
authoritative. **Linux-first; Windows is a compatibility layer, never the
assumption.**

### 4.1 Environment variables — the single source of truth

| Variable | Meaning | Dev-box value (2026-10-02) |
|---|---|---|
| `OR_SRC` | Read-only upstream source tree | `/home/valentin/Projects/OpenRadioss/OpenCourant` |
| `OR_ROOT` | Out-of-tree build/install prefix for the oracle | `/home/valentin/OpenRadioss_or` |
| `OR_STARTER` | Fortran starter binary | `$OR_ROOT/bin/starter_linux64_gf` |
| `OR_ENGINE` | Fortran engine binary | `$OR_ROOT/bin/engine_linux64_gf` |
| `PYRADIOSS_HM_CFG` | CFG card-schema tree | `$OR_SRC/hm_cfg_files` |
| `PYRADIOSS_RD_DECKS` | Full official deck corpus | vendored `tests/data/rd_decks` |
| `PYRADIOSS_BACKEND` | `numpy` \| `numba` \| `auto` | pin `numpy` for comparisons |
| `PYRADIOSS_SPMD_LOG_ALL` | Per-domain listings under `-np` | unset |
| `PYTHON` | Interpreter | `.venv/bin/python` (Linux) / `.venv\Scripts\python.exe` (Windows) |

Resolution order, implemented once in `pyradioss/paths.py` (Task P0.6):
1. the environment variable, if set and **existing**;
2. `$OR_ROOT/../OpenCourant` and `$OR_ROOT/../OpenCourant/hm_cfg_files`
   (sibling-of-build layout);
3. the Windows compat paths `C:\OpenRadioss` / `C:\OpenRadioss\hm_cfg_files`;
4. **fail loudly** — `pyradioss.paths.missing_resource(name: str, tried: list[Path])` raises with the
   four attempted locations, rather than degrading to a silent wrong path.

### 4.2 Commands

One flat command per call. No `&&`, no pipes, no redirection.

```bash
# edit loop — one phase task's tests
$PYTHON -m pytest -q tests/test_p3_bt_hourglass.py

# fast tier — the pre-commit gate for EVERY task
$PYTHON -m pytest -q -m "not slow"

# full suite — final gate only
$PYTHON -m pytest -q

# parity evidence — required for any numerics change
$PYTHON tools/validate_vs_fortran.py --mode parity --cases <list> --out tools/validation_data/parity_p<N>.json
$PYTHON tools/validate_vs_fortran.py --mode coverage --out tools/validation_data/coverage_p<N>.json

# census regeneration
$PYTHON tools/census.py --render plan/CENSUS.md
```

### 4.3 Missing optional dependencies on this box

`numba` and `mpi4py` are **not installed** (2026-10-02). Consequences:

- any numba JIT kernel added in this program is **untestable locally** until
  Task P0.7 lands, which is why P0.7 runs before any accel work.
- every `-np N` / MPI task is untestable until then; Phase 13 has a hard entry
  criterion on Task P0.7 landing.
- Do **not** silently skip these. A phase whose entry criterion is unmet is
  *blocked*, not *started*.

## 5. Wave protocol

A **wave** is a set of tasks that can run concurrently. A **phase** is a
sequence of waves with one gate between them.

```
phase P
  ├── Wave 0   tasks touching only NEW files        → fully parallel, N agents
  ├──  gate    targeted tests green for Wave 0
  ├── Wave 1   tasks touching shared files          → serialised, owner per file
  ├──  gate    targeted tests + fast tier green
  ├── …
  └──  exit gate   (the phase's full exit gate, §8)
```

Rules:

1. **Wave 0 is where parallelism lives.** If a phase's tasks are not splittable
   into disjoint new files, the phase is under-decomposed — fix the plan, not
   the protocol.
2. **Within a wave, a task that needs a contended file (§6) is not in that
   wave.** Contended-file edits are always their own serial wave.
3. **One worktree per wave**, not per task:
   ```bash
   git worktree add ../wt-p<N>-w<M> -b plan/p<N>-w<M> main
   ```
   All Wave-M agents work in that worktree. Merging several agents' commits out
   of one worktree is what makes the ownership map enforceable.
4. **Commit at every green sub-step** inside the worktree; the phase lead
   merges the wave branch into the phase branch after the gate.
5. **After every merge**: `git log --oneline -1` and `git status` — verify the
   commit landed on the branch you think it did.
6. **Phase branches** are named `plan/p<N>`; they merge to `main` at the exit
   gate, one commit-squash per task so history stays bisectable.

## 6. Shared-file ownership map (the contention registry)

These files are rewritten by many phases. **One owner at a time.** The owner is
declared in the phase file's Parallelisation section before the wave starts.

| File | LOC (2026-10-02) | Touched by | Notes |
|---|---:|---|---|
| `pyradioss/engine/engine.py` | 1,689 | P2–P5, P8–P13 | the cycle. Serialised always. |
| `pyradioss/model/entities.py` | **53,286** | P2–P12 | the biggest monolith. Split by Task P1.4. |
| `pyradioss/model/model.py` | 3,955 | P1–P12 | serialised always |
| `pyradioss/input/starter_keywords.py` | **97,102** | P1, P9, P10, P11 | split by Task P1.4/P1.5 |
| `pyradioss/materials/__init__.py` | 5,401 | P6, P7 | the registry |
| `pyradioss/input/engine_keywords.py` | 2,107 | P10, P12, P14 | |
| `pyradioss/input/mat_reader.py` | 1,399 | P6, P7 | |
| `pyradioss/input/prop_reader.py` | 1,647 | P9 | |
| `pyradioss/input/checks.py` | 58 | P2–P9 | the refusal gates; small, high-traffic |
| `pyradioss/starter/initialization.py` | 2,106 | P2–P5, P9, P10 | |
| `pyradioss/elements/__init__.py` | 151 | P2–P5 | the dispatch table |
| `pyradioss/contact/__init__.py` | 119 | P8 | |
| `pyradioss/accel/__init__.py`, `jit_kernels/` | ~5,000 | P2–P5, P8 | numba mirrors |
| `tools/validate_vs_fortran.py` | 917 | P0, all | harness |
| `tests/conftest.py` | 74 | all | fixtures |
| `pyproject.toml` | — | P0, P1 | deps, markers, entry points |
| `AGENTS.md`, `docs/STATE.md` | — | P1, each phase lead | docs |

**Enforcement:** before a wave starts, the phase lead writes the owner map for
that wave into the wave branch as `plan/waves/p<N>-w<M>.owners`. A reviewer
whose task touched an unowned contended file rejects the task. *(Unenforceable
mechanically — the reviewer must assert it.)*

## 7. Task contract

Every task in every phase file has this shape. If a task in a phase file does
not, the phase file is wrong.

```
### Task P<N>.<k>: <name>

**Fortran:** <exact $OR_SRC paths, with the routine names>
**Files:** Create/Modify <exact paths with line ranges>
**Interfaces:**
  - Consumes: <exact signatures from earlier tasks>
  - Produces: <exact signatures later tasks rely on>
- [ ] Step 1  Write the failing test (name + assertions, as code)
- [ ] Step 2  Run it → expect FAIL, with the message quoted
- [ ] Step 3  Implement (signature + file + the pinned values; body only if the
             signature and test leave a genuine choice)
- [ ] Step 4  Run it → expect PASS
- [ ] Step 5  Run the fast tier → expect 0 new failures
- [ ] Step 6  Run parity evidence (numerics tasks only)
- [ ] Step 7  Commit
```

Rules that make a task reviewable:

- One deliverable per task. If the name needs "and", split it.
- Setup, scaffolding and doc steps fold into the task whose deliverable needs
  them.
- **No step may contain a line that decides nothing** ("handle edge cases",
  "add validation", "write tests"). If you find one, the plan is underspecified
  — fix the plan.
- **No step may contain a function body** that the signature and the test
  already determine. A plan longer than the code it describes has written the
  code instead of the plan.
- Test steps are **code** with the exact assertions and the exact expected
  values.
- A task that cannot be shown by a test it wrote first is not a task; it is a
  refactor, and it says so.

## 8. Definition of Done

### 8.1 Per task

- [ ] Its tests were written first and observed failing (the failure message
      is quoted in the commit body).
- [ ] Its tests pass.
- [ ] `pytest -q -m "not slow"` shows **no new failure** relative to the branch's
      recorded baseline.
- [ ] Numerics tasks have a `tools/validation_data/parity_p<N>.json` entry.
- [ ] Every new/edited Python module docstring cites its upstream Fortran with
      line ranges.
- [ ] The new force path books into the EN ledger (force tasks).
- [ ] No contended file edited without being the declared owner.
- [ ] Committed. `git status` clean on the wave branch.

### 8.2 Per phase

- [ ] Every task in the phase meets §8.1.
- [ ] The phase's **exit gate** (§ its own file) passes.
- [ ] The census (`plan/CENSUS.md`, Task P1.3) is regenerated and every upstream file in
      the phase's scope is `ported` or has an explicit, cited, maintainer-
      accepted `partial` entry.
- [ ] `pytest -q -m "not slow"` green against the phase's baseline.
- [ ] A **phase reviewer** has reviewed the phase branch as a whole (§9.3).
- [ ] `docs/STATE.md` gains the phase row; `UPDATES.md` gains a top entry.
- [ ] The phase branch merges to `main` as one commit per task.

### 8.3 Per program (Phase 17 only)

- [ ] All 18 phases merged.
- [ ] Whole-program reviewer passed.
- [ ] 529-deck corpus: every verdict is CLEAN or a cited, accepted SKIP.
- [ ] Parity ledger: every element family, every registered material law, every
      contact type has a measured rel-RMS with a stated tolerance.

## 9. Reviewer protocol

### 9.1 Per task (default)

A **fresh** reviewer — new context, has not seen the implementation. Given:
the task text, the diff, and the test command + its output. It must answer:

1. Does every ported formula cite upstream with a line range? List any that
   does not.
2. Was the test written first and observed failing? (commit body evidence)
3. Does the test actually pin the claimed behaviour — would it fail if the
   implementation were replaced by a plausible wrong one?
4. Is the kernel vectorised over element groups?
5. Does the force path book into the EN ledger?
6. Was the backend pinned for any before/after claim?
7. Did any contended file get touched without ownership?
8. Any place where the implementation chose **not** to mirror the Fortran —
   is it cited, documented, and justified?
9. Any new silently-degrading path (a warning where the Fortran would abort)?

### 9.2 Adversarial verification

Any claim that **matches the Fortran** must be checked by a second route, not
by reading the code: a hand-computed value on a paper case, an analytic
invariant (energy, momentum, symmetry, limiting case), or an md5 comparison
against the oracle. "The reviewer read it and it looks right" is not evidence.

### 9.3 Per phase

A reviewer with fresh context reviews the phase branch end-to-end: the
census delta, the fast-tier result, the parity JSON, and the three highest-risk
tasks re-derived adversarially. Reports: what is genuinely done, what is
claimed-but-weak, what regressed.

### 9.4 Whole program

At Phase 17: a fresh reviewer re-derives the completion claims — re-runs the
corpus sweep, re-runs the parity ledger, and reads the census — rather than
trusting 18 phase reports.

## 10. Review Focus

The five input classes most likely to bite a user of this software that no
task's tests obviously exercise. Each phase that owns the relevant code must
pin its line with a test.

1. **A deck with a keyword the port parses but has no physics for.** The Starter
   accepts it (generic CFG reader, `InactiveMaterial`/`InactiveProperty`), the
   Engine is supposed to refuse it. *Expected:* a named refusal naming the law,
   not a zero-stress run and not a crash. Owned by P6/P9 → test
   `refuse_inactive_materials` / `refuse_inactive_properties` per-law.
2. **Degenerate or pathological geometry** — a `/BRICK` with repeated nodes, a
   zero-volume `/TETRA4`, a `/QUAD` with a collapsed edge, an inverted shell.
   *Expected:* the Fortran's degeneracy path (`degenes8`, `sdlen_dege`,
   `idege*`) produces a defined dt and finite forces, never a NaN and never a
   silent pass-through. Owned by P2.
3. **A run that diverges mid-simulation.** *Expected:* the `/STOP/ENERGY` and
   `/STOP/TIME` guards fire and are **reported**, with KE *and* IE *and* HE
   checked for finiteness — note the current backstop only checks KE
   (`VALIDATION.md` §7 item 10). Owned by P12.
4. **Restart across a mid-run state change.** *Expected:* `_0002.rad` resumes
   bit-identically to the unchained run, including for the new element
   families and the binary restart format of Phase 12. Owned by P12.
5. **An interface or material activated on an element family it does not
   support.** *Expected:* a loud refusal citing the upstream refusal, never a
   silent no-op force. Owned by P8 and P6.

## 11. Where the size bands are honest, and where they are not

- **Honest:** the upstream LOC counts. They are measured, not estimated.
- **Honest:** the port/LOC ratio for output (752 Python LOC vs 180,559 Fortran)
  and for MPI (2,582 vs 111,955) — these are the emptiest areas and their bands
  are the least likely to be underestimates.
- **Not honest:** the band for P6 (materials). 121 upstream law directories,
  ~59 registered, ranging from 133 lines (`mat133`) to 7,824 (`mat104`). A
  "port the material phase" is not a task; each law is a task, and the phase
  file sequences them by *corpus usage frequency × implementation risk*.
- **Not honest:** P2 and P3 element bands, which assume the existing kernels
  are structurally right and only need their missing routine sets filled in.
  Phase 8 item 1 (BT family unreachable by comparison) is evidence that this
  assumption can fail. The first wave of P2/P3 is therefore a **reconciliation
  wave**: audit the existing kernel against its upstream routine set and report
  divergences before porting anything new.