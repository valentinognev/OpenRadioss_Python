# pyradioss (OpenRadioss_Python) — agent instructions

Python port of the OpenRadioss explicit FEM solver: Starter + Engine, an
implicit branch, a spectral-fatigue tower, optional numba backend, tkinter GUI.
The value of this repo is readable, **Fortran-faithful** physics — porting, not
inventing.

Linux-first. Windows 11 is a supported compatibility target, not the assumption
— see §Windows compatibility. The environment contract is
`plan/00_ORCHESTRATION.md` §4, and its §4.1 table is the single source of truth
for every variable named below.

## STOP — licensing gate: no new upstream-derived code

**Unresolved. Do not add new upstream-derived code** — no ported routine, kernel,
material law, contact path or output path taken from `$OR_SRC` — until a
maintainer records a decision. Upstream is **AGPL-3.0-or-later**; this
repository declares MIT (`LICENSE`) *and* GPL-3.0 (`README.md`), so a literal
port is an AGPL-covered derivative work: `plan/00_ORCHESTRATION.md`
**§1.3, lines 58-73** (GATING, UNRESOLVED).

- **No decision is recorded, and an agent may not record one**
  (`docs/LICENSING.md` §Decision, lines 136-156 — a recommendation only). This
  notice records none.
- **What unblocks it:** the maintainer records one of the four options in that
  §Decision and applies it, so `pyproject.toml`, `LICENSE` and `README.md` agree;
  `tests/test_p0_licensing.py` is then updated by hand.
- **The enforcement is documentary, not mechanical:** the only enforcement is
  `tests/test_p0_licensing.py::test_declared_licence_is_consistent`, an
  `xfail(strict=True)` that **PASSES**. Nothing stops you except this notice.
- Working text: `docs/STATE.md` §Licensing gate, lines 11-56.

## Environment (do not deviate)

- **Interpreter: `$PYTHON`** — never a bare `python`/`pip`, never a path
  hardcoded to one platform. Resolve it from the repo root
  (`export PYTHON="$PWD/.venv/bin/python"`) and put it where it survives the
  tool you drive: your shell profile is the durable answer; without one,
  repeat the `export` at the head of **every** call that uses it (§Terminal
  rules — an export from an earlier call is not inherited). The interpreter's
  own version is the `python` pin in `requirements-lock.txt` §[B]; do not
  repeat it in prose.
- Every dependency is **already installed**. Do NOT create a venv, do NOT
  `pip install`. Pins live in `requirements-lock.txt` §[B] (this Linux `.venv`),
  and every one of them is asserted against the importable module on every run
  by `tests/test_p0_optional_deps.py`.
- The other variables — `$OR_SRC`, `$OR_ROOT`, `$OR_STARTER`, `$OR_ENGINE`,
  `$PYRADIOSS_HM_CFG`, `$PYRADIOSS_RD_DECKS`, `$PYRADIOSS_BACKEND` — are
  defined in §4.1; read it before setting one. `$OR_BUILD` is the one exception,
  and §4.1 does not define it: its home is `tools/oracle/oracle_env.sh`, which
  hard-fails (`:?`) unless it is exported.
  `$PYRADIOSS_HM_CFG` (the CFG card-schema tree the `/MAT`–`/PROP` readers
  parse) and `$PYRADIOSS_RD_DECKS` (a fuller deck corpus than the vendored
  `tests/data/rd_decks`) both resolve unaided on a dev box: set them only to
  point somewhere else.
- **Resolution lives in code, not in your head.** `pyradioss/paths.py` resolves
  every external resource in the §4.1 order and **fails loudly** rather than
  degrading to a silent wrong path. If a path is wrong, fix the variable — do
  not work around the resolver, and do not "fix" it by setting a variable that
  happens to exist.
- `mpi4py` is **not installed** in this `.venv` (`requirements-lock.txt` §[B]
  records the candidate pin and the measurement): every `-np N` / SPMD path runs
  single-rank — `comm.mpi_world_size()` returns 1 — so MPI work is *blocked*
  until the `mpi` extra is installed, not merely untested.
- **Never write into the oracle's trees.** `$OR_SRC` (upstream source) is
  **READ-ONLY**: no edits, no generated files, no builds inside it.
  `$OR_ROOT` (the install prefix `tools/oracle/build_oracle.sh` installs the two
  binaries into) is likewise not yours to hand-edit. `$OR_BUILD` is the only
  tree a build may write into.

## The oracle and parity evidence

- The oracle is the real Fortran build: `$OR_STARTER` and `$OR_ENGINE` out of
  `$OR_ROOT/bin`. Its runtime comes from **`source tools/oracle/oracle_env.sh`**,
  which sets `LD_LIBRARY_PATH`, `OR_STARTER` and `OR_ENGINE` and refuses to run
  unless `OR_BUILD` and `OR_ROOT` are already exported. The dev box's values —
  and `pyradioss/paths.py` finds this layout unaided, so exporting them only
  pins a *different* oracle — are:

  ```bash
  export OR_SRC="$HOME/Projects/OpenRadioss/OpenCourant"   # READ-ONLY
  export OR_BUILD="$HOME/OpenRadioss_build"               # the writable mirror
  export OR_ROOT="$HOME/OpenRadioss_or"                   # the install prefix
  source tools/oracle/oracle_env.sh
  ```

  A `starter` exiting **127** is a loader problem, never an H3D or
  message-subsystem one; that script's header explains why, and it is right.
- **A numerics change requires a parity run against that oracle.** The targeted
  tests are necessary and not sufficient for anything touching element, material,
  contact or dt code: run `tools/validate_vs_fortran.py parity` (the full
  invocation is in §Commands) and quote the numbers in the commit. See §Domain
  rules and the `validation-compare` skill.
- `PYRADIOSS_ORACLE_REQUIRED=1` (what the phase gates set) makes a *missing*
  oracle fail instead of skipping. Do not silently skip an absent oracle.
- The oracle's tree must stay pristine: `git -C "$OR_SRC" status --porcelain`
  is empty at every gate. If a step needs a write there, the step is wrong.

## Commands

Written as `$PYTHON …`; resolve `$PYTHON` first (§Environment).

- Edit loop — one task's tests: `$PYTHON -m pytest -q tests/test_p<N>_<slug>.py`
- Fast tier — the pre-commit gate for **every** task:
  `$PYTHON -m pytest -q -m "not slow"`
- Full suite (adds the `slow` tests) — final gate only, not for iteration:
  `$PYTHON -m pytest -q`
- Parity evidence — required for any numerics change. `parity` is a
  **subcommand**, `--only` takes comma-separated `examples/` names, and the
  rows land in `<workdir>/parity_results.json`:
  `$PYTHON tools/validate_vs_fortran.py parity --only <names>`
- Run the port on an example (from `examples/tensile_bar`):
  `$PYTHON -m pyradioss.starter -i TENSILE_0000.rad`, then
  `$PYTHON -m pyradioss.engine -i TENSILE_0001.rad`
- **Do not quote a suite size out of a record.** Re-measure it
  (`$PYTHON -m pytest -q --collect-only -m "not slow"`): a bare count rots, and
  `tests/test_p0_record_suite_counts.py` reports any count not bound to the
  date/commit it was measured at. `docs/STATE.md` §Baseline is the model — it
  states the command, not a promise.

## Terminal rules

- One flat command per call. Do NOT wrap whole commands in outer quotes.
- No `&&`, no redirection — one command, or another call. Pipes are fine on
  POSIX; the rule is that what you ran stays copy-pasteable as a single line.
- Never start a long-running or watch process (`pyradioss-gui`, a `-Wait` loop);
  ask the user instead.
- One call, one environment: do not rely on a variable exported in an earlier
  call. Anything the Fortran solver needs comes from
  `source tools/oracle/oracle_env.sh` **in the same call** that runs the oracle;
  the same is true of `$PYTHON` unless your shell profile already resolves it
  (§Environment).

## Project

- START HERE: `docs/STATE.md` — §What is implemented (the milestone table) and
  §Baseline (the commands, not the numbers). Then `plan/README.md` for the
  roadmap and `plan/00_ORCHESTRATION.md` for the binding contract: environment,
  file ownership, reviewer protocol, definition of done.
- **Vocabulary.** The program runs `P<phase>.<task>` tasks (e.g. `P1.0`) in
  **waves** of parallel agents on one shared worktree, with one **reviewer** per
  task and the phase lead merging at the wave gate. Pre-program milestone
  numbering continues at **M700+**; the `test_mNN_*.py` files predate it. Work
  **ONE task per conversation**.
- **Commit at every green sub-step** — session and quota interruptions are
  normal; committed work survives, uncommitted work may not. **Stage by path**
  (`git add AGENTS.md`), never `git add -A`: several agents share one wave
  worktree. Then `git log --oneline -1` and `git status`, to confirm the commit
  landed on the branch you think it did.
- `PORTING_GUIDE.md` (535 KB) and `VALIDATION.md` (233 KB) are REFERENCE, not
  reading material — NEVER read them linearly. Grep for the section you need and
  read that range only. VALIDATION.md's LATEST §3.x parity / §4.x coverage /
  §6.x perf edition is authoritative.
- Skills: `.agents/skills/run-reference-openradioss/` (launch the real Fortran
  solver), `.agents/skills/validation-compare/` (parity/coverage vs Fortran),
  `.agents/skills/lspp-check/` (verify a d3plot in LS-PrePost batch mode).
- Layout: `pyradioss/` (common, input, model, starter, engine, elements,
  materials, failure, contact, output, implicit, accel, gui), `tests/`
  (+ `tests/data/rd_decks` vendored corpus decks), `tools/`
  (validate_vs_fortran.py, oracle/, validation_data/), `examples/` (runnable
  decks). `tools/benchmark_rad_db.py` benchmarks against the **external**
  harvested `rad_examples_db` corpus, which is not in this checkout: it is
  located solely by `$RAD_EXAMPLES_DB` and has no default.

## Model policy

- Every implementer and every reviewer runs on **space-bunny**.
- **A Fast variant is forbidden for every role** — implementer, reviewer,
  explore subagent, whole-program reviewer. No exceptions, no fallbacks.
- If the only subagent available is a Fast one, **do not dispatch it**: report
  the intended model to the maintainer and ask, or do the work in the parent
  session. Name the intended model on every dispatch.

## Domain rules

- NEVER guess Fortran semantics. Every ported formula cites the exact upstream
  file under `$OR_SRC/{starter,engine,common_source}/...` — open
  and read the Fortran, don't recall it. If the Fortran disagrees with your
  expectation, the Fortran wins.
- Numerics changes REQUIRE validation evidence: the targeted tests, plus — for
  anything touching element/material/contact/dt code — a parity run against
  the real Fortran solver (validation-compare skill). "The suite is green"
  alone is not evidence for a physics change.
- Never weaken a test. If a milestone must realign a pre-existing test
  (phase-sensitive thresholds shift when physics is corrected), name the test
  and justify the change with numbers in the commit message and walkthrough.
- Energy accounting is load-bearing: every new force path must book its work
  in the EN ledger; the energy-balance tests catch silent leaks.
- For before/after comparisons pin the backend: `PYRADIOSS_BACKEND=numpy`
  (numba is proven separately via T01 md5 equality — see validation-compare).
- Mark any test ≥60 s serial with `@pytest.mark.slow`.

## Windows compatibility

The maintainer's Windows 11 box is a supported target, not the assumption — and
nothing above is written for it. When a task must run there:

- **Interpreter:** the venv's `Scripts` variant. §4.1's `PYTHON` row carries
  both spellings; set `$env:PYTHON` to it in PowerShell, exactly as you set
  `$PYTHON` in a POSIX shell, and every command in §Commands works unchanged.
- **Pins:** `requirements-lock.txt` **§[A]** is that box's pin set, and the only
  place its versions are recorded (numpy 2.4.6, scipy 1.18.0, numba 0.66.0,
  pytest 9.1.1). §[B] is the Linux `.venv`'s, and is the only section anything
  machine-checks.
- **Paths:** the Windows prefix is still the **last** candidate in
  `pyradioss/paths.py`'s resolution order (rule 3), so a Windows checkout needs
  no configuration — that is the whole of the compatibility story in code.
- §Terminal rules apply unchanged: one command per call, no `&&`, no
  redirection.
