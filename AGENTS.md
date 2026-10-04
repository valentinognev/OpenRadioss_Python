# pyradioss (OpenRadioss_Python) — agent instructions

Python port of the OpenRadioss explicit FEM solver: Starter + Engine, an
implicit branch, a spectral-fatigue tower, optional numba backend, tkinter GUI.
The value of this repo is readable, **Fortran-faithful** physics — porting, not
inventing.

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

- Windows 11, PowerShell. No WSL, no Git Bash, no Unix commands (ls/rm/cat/grep).
- Python 3.12.3 in `.venv` — this box's interpreter, pinned in
  `requirements-lock.txt` §[B]. NEVER bare `python` or `pip`. Always exact
  paths:
  - `.venv\Scripts\python.exe`
  - `.venv\Scripts\pip.exe`
- All dependencies are preinstalled (numpy 2.4.6, scipy 1.18.0, numba 0.66.0,
  pytest 9.1.1 — exact pins in `requirements-lock.txt`). Do NOT create a venv,
  do NOT run pip install, unless the user explicitly asks.
- The generic `/MAT`–`/PROP` readers need the hm_cfg_files CFG tree; it is
  auto-found at `C:\OpenRadioss\hm_cfg_files` (env `PYRADIOSS_HM_CFG` overrides).
- Corpus-deck tests use `tests\data\rd_decks` (vendored); env
  `PYRADIOSS_RD_DECKS` points at a fuller extract (see the README there).
- External harvested reference corpus for benchmarking & validation:
  `C:\Users\pmqua\PycharmProjects\rad_examples_db` (decks, manifest, benchmarks).
- READ-ONLY locations — never write, delete, or extract in place:
  `C:\OpenRadioss` (reference Fortran install + source), `C:\OpenRadioss_old`,
  and everything under `E:\` (`E:\openradioss_run`, `E:\foxcore_data`).

## Commands

- Edit loop (one milestone's tests):
  `.venv\Scripts\python.exe -m pytest -q tests\test_m42_<slug>.py`
- Fast tier — the pre-commit gate:
  `.venv\Scripts\python.exe -m pytest -q -m "not slow"`
  (~1105 tests; ~12–50 min depending on machine load — 30 min at handover)
- Full suite (adds the 16 `slow` tests, ~25 min): `.venv\Scripts\python.exe -m pytest -q`
  — final gate only, not for iteration.
- Run the port on an example (from inside `examples\tensile_bar`):
  `.venv\Scripts\python.exe -m pyradioss.starter -i TENSILE_0000.rad` then
  `.venv\Scripts\python.exe -m pyradioss.engine -i TENSILE_0001.rad`
- Fast-tier baseline at handover (2026-08-02, this venv): see docs/STATE.md §Baseline.

## Terminal rules

- One flat command per call. Do NOT wrap whole commands in outer quotes.
- No `&&`, no pipes, no redirection — separate commands.
- Never start long-running/watch processes (`pyradioss-gui`, `-Wait` loops);
  ask the user instead.
- Windows backslash paths everywhere.
- Do not rely on env vars set in an earlier terminal call persisting; scripts
  that need env (the Fortran solver!) go through the provided .ps1 helpers.

## Project

- START HERE: @docs/STATE.md — what is implemented (M1–M41), the current
  baseline numbers, the repo map, and the re-sliced milestone roadmap with a
  per-milestone acceptance command.
- Work ONE milestone per conversation. **Commit at every green sub-step**
  (`git add -A`, `git commit`) — session/quota interruptions are normal here;
  committed work survives, uncommitted work may not.
- PORTING_GUIDE.md (535 KB) and VALIDATION.md (233 KB) are REFERENCE, not
  reading material — NEVER read them linearly. Grep for the milestone/section
  heading you need, read that range only. PORTING_GUIDE §5 = per-milestone
  roadmap history; VALIDATION.md = validation reports (the LATEST §3.x parity /
  §4.x coverage / §6.x perf edition is authoritative).
- Skills: `.agents/skills/run-reference-openradioss/` (launch the real Fortran
  solver), `.agents/skills/validation-compare/` (parity/coverage vs Fortran),
  `.agents/skills/lspp-check/` (verify a d3plot in LS-PrePost batch mode).
- Layout: `pyradioss/` (common, input, model, starter, engine, elements,
  materials, failure, contact, output, implicit, accel, gui), `tests/`
  (+ `tests/data/rd_decks` vendored corpus decks), `tools/`
  (validate_vs_fortran.py, validation_data/, lspp_check.py, benchmark.py),
  `examples/` (runnable decks).

## Domain rules

- NEVER guess Fortran semantics. Every ported formula cites the exact upstream
  file under `C:\OpenRadioss\source\{starter,engine,common_source}\...` — open
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
- After committing: `git log --oneline -1` and `git status` — verify the
  commit actually landed on the branch you think it did.
