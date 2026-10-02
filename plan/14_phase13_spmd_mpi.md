# Phase 13 — SPMD / MPI

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/mpi/` (112,000 LOC, 257 files) and the Starter
half a literal port, and **lift the three refusals** `docs/OPEN_BUGS.md` records
as interim measures.

**Architecture:** MPI is a **single-collective-per-arrays** design: 21
subdirectories, each owning the exchange for one array family. The port's
`pyradioss/spmd/` is 2,582 lines with one communicator abstraction
(`comm.py`: `ThreadComm`, `Mpi4pyComm`) and a frontier force exchange. A literal
port needs **the whole array family inventory**, because every array that lives
per-element or per-node is an exchange.

**Tech stack:** NumPy; optional `mpi4py`; the port must keep working
single-process with the domains as threads (`README.md`'s documented mode).
**Spec:** `docs/OPEN_BUGS.md` SPMD-1/2/3; `plan/00_ORCHESTRATION.md` §5.
**Entry criteria:** Phases 2–5, 8, 10, 12 exit gates; **Phase 0 Task P0.7**
  (`mpi4py` must be installed or the phase is *blocked*, not skipped).
**Band: L.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `engine/source/mpi/interfaces` | — | **47,831** | `spmd/exchange.py` (frontier forces only) |
| `engine/source/mpi/implicit` | — | 16,421 | — |
| `engine/source/mpi/forces` | — | 9,536 | `spmd/exchange.py` |
| `engine/source/mpi/anim` | — | 6,813 | — |
| `engine/source/mpi/elements` | — | 4,651 | — |
| `engine/source/mpi/kinematic_conditions` | — | 3,357 | — |
| `engine/source/mpi/generic` | — | 3,462 | — |
| `engine/source/mpi/output` | — | 3,363 | — |
| `engine/source/mpi/ams` | — | 2,986 | — |
| `engine/source/mpi/r2r` | — | 2,511 | — |
| `engine/source/mpi/fluid` | — | 2,487 | — |
| `engine/source/mpi/sph` | — | 1,286 | — |
| `engine/source/mpi/sections` | — | 1,549 | — |
| `engine/source/mpi/airbags` | — | 1,359 | — |
| `engine/source/mpi/lag_multipliers` | — | 913 | — |
| `engine/source/mpi/nodes` | — | 1,556 | — |
| `engine/source/mpi/init` | — | 937 | — |
| `engine/source/mpi/{ale,seatbelts,user_interface}` | — | 283 | — |
| `starter/source/spmd/` | 37 | 18,608 | `spmd/domdec.py` (Phase 10 Task P10.10) |
| `engine/source/mpi/` (top level) | — | 654 | `spmd/__init__.py` |
| **Total** | **294** | **≈ 165,000** | **2,582** |

## Gap analysis

### 1. `interfaces/` — 47,831 LOC, and the three refusals

`docs/OPEN_BUGS.md` records three interim **refusals** rather than ports:

| Refusal | Upstream file it cites | What is missing |
|---|---|---|
| SPMD-1 `/GJOINT` under `-np > 1` | `engine/source/tools/lagmul/lag_mult.F`, `LAG_MULTP` ~683-687 | **nothing in serial** — the joint is replicated and its non-linear branch selection diverges. The upstream fix is `SPMD_EXCH` on the joint forces. |
| SPMD-2 ghost `off` never refreshed | `interfaces/interf/chkstfn3.F` (`TAGOFF3N` ~572, `SPMD_EXCH_IDEL` 1129-1136, `SPMD_INIT_IDEL`/`SPMD_EXCHMSR_IDEL` 1807-1808, `CHK2MSR3N` ~3158 with its 3658-3668 remote query) | `engine/source/mpi/interfaces/spmd_exch_idel.F` |
| SPMD-3 `/KJOINT` counted once per replica | `elements/joint/ruser33.F`, `rgjoint.F` (computed once on the owning domain, then frontier-summed) | the `WEIGHT` convention on the additive joint forces |

**`SPMD_EXCH_IDEL` is the whole of SPMD-2 and it is the port's own listed
follow-up.** Its fix is spelled out in `docs/OPEN_BUGS.md`:

1. Domdec keeps a global-id array per ghost group.
2. Once per cycle, on every rank, after the element loop and before the contact
   / TYPE2 `transfer_forces`, all-gather the `elem_glob` rows whose `off` went
   to 0 this cycle.
3. Each rank sets `off = 0` on the matching rows of
   `model.spmd_ghost[g].state["off"]` — **on the same cycle everywhere**.
4. Remove the "Known limitation" paragraph from `domdec.py`.
5. Add a `-np 2` vs serial parity test with a TYPE2 tie on a failing shell.

**This phase does that verbatim.** The port's own bug report is the
specification.

### 2. The array inventory

Every `engine/source/mpi/<family>/` is a set of arrays exchanged each cycle.
The census (`tools/census.py`, Phase 1 Task P1.3) already extracts the `!||`
call graph, so **the inventory is derivable mechanically**: for each
`spmd_*.F` in `engine/source/mpi/`, record the routine, the arrays it touches
(read from the routine's argument list), and whether the port exchanges it.

### 3. The `interf/` MPI half

`interfaces/interf/` has both a local half and an MPI half
(`find_surface_from_remote_proc.F`, `find_edge_from_remote_proc.F`,
`get_neighbour_surface_from_remote_proc.F`, `check_remote_surface_state.F`).
The port's `contact/state.py` (Phase 8 Task P8.6) is deliberately
**communicator-free** so this phase can wrap it. That design decision is what
makes this phase tractable.

### 4. What the port already gets right

`README.md`: frontier forces and stiffnesses are exchanged and summed
(`spmd_exch_a.F`), the global time step is agreed (`spmd_glob_min5.F`), the
energy/momentum ledgers reduce with the `WEIGHT` convention (`ecrit.F`), and
domain 0 gathers the global view and writes the only listing/T01/ANIM
(`spmd_chkw.F`, `PYRADIOSS_SPMD_LOG_ALL=1` for per-domain listings). **The port
reproduces serial results up to floating-point summation order.** That is the
parity contract this phase must extend, not invent.

## Wave graph

```
Wave 0  P13.0 the MPI array inventory (serial — the checklist)
   ── gate: every spmd_*.F has an array list and a port status
Wave 1 (parallel — new modules)
  P13.1 comm abstraction completion (mpi4py + threads + the WEIGHT convention)
  P13.2 forces exchange (spmd_exch_a family)
  P13.3 nodes exchange                   P13.4 kinematic_conditions exchange
  P13.5 elements exchange (incl. deletion) P13.6 interfaces exchange
  P13.7 implicit exchange                P13.8 anim/output exchange
  P13.9 airbags / fluid / sph / sections / r2r / ams exchanges
   ── gate per family: -np N reproduces serial
Wave 2 (serial)
  P13.10 SPMD_EXCH_IDEL — lift the SPMD-2 refusal
  P13.11 lift the SPMD-1 (/GJOINT) and SPMD-3 (/KJOINT) refusals
  P13.12 remove the domdec.py "Known limitation" paragraph
  P13.13 -np N parity sweep across the corpus
```

---

### Task P13.0: The MPI array inventory

**Fortran:** every `engine/source/mpi/**/spmd_*.F` and `*_exch*.F`; the
  `engine/source/mpi/*/` subdirectory structure is itself the inventory.
**Files:** Create `tools/validation_data/mpi_status.json`,
`tests/test_p13_mpi_census.py`.
**Interfaces:**
- Produces: one record per MPI routine:
  `{"fortran": "engine/source/mpi/forces/spmd_exch_a.F", "family": "forces",
    "arrays": ["FIINT", "FMOMT", "EINT"], "frequency": "cycle",
    "reduction": "sum|min|max|none", "port": "exchanged|missing|not-needed",
    "port_module": "spmd/exchange.py"}`
  plus `"families"` (the 21 subdirectory names).

- [ ] **Step 1** — write the failing test:

```python
def test_every_mpi_family_is_in_the_inventory():
    st = json.loads(Path("tools/validation_data/mpi_status.json").read_text())
    src = paths.or_src() / "engine/source/mpi"
    families = {d.name for d in src.iterdir() if d.is_dir()}
    assert {r["family"] for r in st} == families
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — build it. **The `!||` call graph from `tools/census.py` is
  the input**, not a re-read of 257 files: for each routine, its callers tell
  you which arrays matter, and its argument list gives the names. Mark
  `not-needed` only with a reason (e.g. a family that is empty in the open tree
  like `ale`, `seatbelts`, `user_interface` at 283 LOC total).
- [ ] **Step 4** — run → PASS; commit `feat(census): the MPI exchange inventory`.

---

### Task P13.1: Complete the communicator abstraction

**Fortran:** `engine/source/mpi/init/` (937 LOC) — the communicator setup,
  `ISPMD`, `NSPR` and the MPI handle plumbing.
**Files:** Modify `pyradioss/spmd/comm.py`, `pyradioss/spmd/__init__.py`.
**Interfaces:**
- Consumes: existing `ThreadComm`, `Mpi4pyComm`.
- Produces: a uniform `Comm` protocol with
  `rank`, `size`, `bcast(array, root=0)`, `allreduce(array, op)`,
  `allgather(arrays)`, `exchange(per_rank_dict) -> list[dict]`,
  `barrier()`, `weight(node_ids) -> np.ndarray` — **the `WEIGHT` convention**
  (`ecrit.F`: shared nodes booked once).

- [ ] **Step 1** — write the failing test:

```python
def test_weight_is_one_on_a_frontier_node_and_shared_arrays_are_booked_once():
    comm = _comm(n=3)
    w = comm.weight(_frontier_node_ids())
    assert w.max() == 1.0
    f = np.ones((len(w), 3))
    assert comm.allreduce_weighted(f).sum() == pytest.approx(_serial_force_sum())
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The `WEIGHT` convention is the subtle part**: a
  frontier node is held by two domains, and summing both gives twice the force.
  `ecrit.F` handles this by weighting the *ledger* contributions, not the
  forces. Read it — implementing it as `allreduce(sum)` on the forces is a
  factor-of-two bug on every frontier node, which the test above catches.
  `ThreadComm` must remain the default when `mpi4py` is absent.
- [ ] **Step 4** — run → PASS under both `ThreadComm` and `Mpi4pyComm`;
  commit `feat(spmd): unified communicator with the WEIGHT convention`.

---

### Tasks P13.2 – P13.9: The exchange families

**Eight tasks, one per family group. The specification below is shared; only
the `Fortran:` line differs.** Each is written into its own file, so an agent
that opens this file knows exactly which one it owns.

| Task | Family | Fortran | LOC |
|---|---|---|---:|
| **P13.2** | `forces` | `engine/source/mpi/forces/` (`spmd_exch_a.F`) | 9,536 |
| **P13.3** | `nodes` | `engine/source/mpi/nodes/` | 1,556 |
| **P13.4** | `kinematic_conditions` | `engine/source/mpi/kinematic_conditions/` | 3,357 |
| **P13.5** | `elements` (incl. deletion) | `engine/source/mpi/elements/` | 4,651 |
| **P13.6** | `interfaces` | `engine/source/mpi/interfaces/` — **the big one** | 47,831 |
| **P13.7** | `implicit` | `engine/source/mpi/implicit/` | 16,421 |
| **P13.8** | `anim`, `output` | `engine/source/mpi/anim/`, `.../output/` | 10,176 |
| **P13.9** | `airbags`, `fluid`, `sph`, `sections`, `r2r`, `ams`, `generic`, `lag_multipliers`, `init` | the remaining nine directories | 15,988 |

**Common specification (every task in this group):**

**Files:** Create `pyradioss/spmd/exch_<family>.py`,
`tests/test_p13_exch_<family>.py`.
**Interfaces:**
- Consumes: Task P13.1's `Comm`; the family's arrays from Task P13.0.
- Produces: `exch_<family>.exchange(comm, model, state) -> None` — in place.

- [ ] **Step 1** — write the failing test, always the same shape:

```python
def test_<family>_exchange_reproduces_serial():
    a = _run(np=1); b = _run(np=4)
    assert np.allclose(a.T01["IE"], b.T01["IE"], rtol=1e-8)
    assert np.allclose(a.T01["MOMX"], b.T01["MOMX"], rtol=1e-8)
```

  **The force and the momentum channels are the specification** — a
  decomposition that exchanges forces but not momentum is wrong in a way the
  energy total hides.
- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The reduction op is per family and per array**
  (`sum` for forces and masses, `min` for the global time step
  (`spmd_glob_min5.F`), `max` for some status flags, gather for the output
  family). Read each routine; do not default everything to `sum`.
- [ ] **Step 4** — run → PASS; commit.

**Per-family notes the implementer must not skip:**

- **P13.2 `forces`** — this is the exchange the port already has (frontier
  forces and stiffnesses, `spmd_exch_a.F`). The task is **completeness**, not
  existence: the family's other arrays.
- **P13.4 `kinematic_conditions`** — `/BCS`, `/IMPVEL`, `/IMPDISP` and the
  rigid-wall velocities. A boundary condition that is not exchanged produces a
  **half-constrained** boundary node, which is silent.
- **P13.5 `elements`** — includes the deletion-related arrays, and is the
  prerequisite for Task P13.10's `SPMD_EXCH_IDEL`.
- **P13.6 `interfaces`** — the largest family. Consumes Phase 8 Task P8.6's
  communicator-free `contact/state.py` **by design**; that design is what makes
  this task tractable.
- **P13.7 `implicit`** — the constraint and tangent exchanges; depends on Phase
  14, so it may need to run after Phase 14 merges.
- **P13.8 `anim`/`output`** — the gather side: **domain 0 writes, everyone
  else contributes** (`spmd_chkw.F`; only P0 prints).
- **P13.9** — the nine small families; `ale`, `seatbelts` and `user_interface`
  are **283 LOC in total** and may be empty in the open tree, which is a
  legitimate `not-needed` per Task P13.0.

---

### Task P13.10: `SPMD_EXCH_IDEL` — lift the SPMD-2 refusal

**Fortran:** `engine/source/mpi/interfaces/spmd_exch_idel.F` (the routine
  `docs/OPEN_BUGS.md` names as "tracked for future SPMD enhancement");
  `engine/source/interfaces/interf/chkstfn3.F` — `TAGOFF3N` (~line 572),
  `SPMD_EXCH_IDEL` calls (~1129-1136), `SPMD_INIT_IDEL`/`SPMD_EXCHMSR_IDEL`
  (~1807-1808), `CHK2MSR3N` (~3158) and its remote query (~3658-3668);
  `spmd_exchmsr_idel.F`.
**Files:** Create `pyradioss/spmd/exch_idel.py`,
`tests/test_p13_exch_idel.py`; Modify `pyradioss/spmd/domdec.py`
(remove the "Known limitation" paragraph), `pyradioss/contact/state.py`
(the hook).
**Interfaces:**
- Consumes: `model.spmd_ghost[g].elem_glob` (global ids per ghost group).
- Produces: `exch_idel.sync_deletions(comm, model, cycle) -> int` (the number
  of newly-deleted elements), called from `engine.py` **after the element loop
  and before `transfer_forces`**, exactly as `docs/OPEN_BUGS.md` specifies.

- [ ] **Step 1** — write the failing test — the port's own failure cases:

```python
def test_a_ghost_off_flag_is_refreshed_on_the_same_cycle():
    """OPEN_BUGS SPMD-2 failure case A: a TYPE2 tie whose main-segment parent
    shell is native on rank A and a ghost on rank B must go inactive on BOTH."""
    model = _model_with_type2_tie_on_a_failable_shell(np=2)
    model.groups["shells"].state["off"][_native_index(model)] = 1
    n = exch_idel.sync_deletions(_comm(model), model, cycle=9)
    assert n == 1
    ghost_rows = model.spmd_ghost["shells"].state["off"]
    assert ghost_rows[_ghost_row(model)] == 0        # NOT stale 1
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, in the five steps `docs/OPEN_BUGS.md` spells out
  (they are reproduced in §Gap analysis 1 above). **Step 3's "same cycle
  everywhere" is the whole point**: the exchange must be collective and
  unconditional, not guarded by a local `any(off == 0)` test, or the ranks
  disagree and deadlock.
- [ ] **Step 4** — run → PASS; the `-np 2` vs serial parity test for a TYPE2
  tie on a failing shell is green; commit
  `feat(spmd): SPMD_EXCH_IDEL — ghost deletion is now domain-consistent`.

---

### Task P13.11: Lift the `/GJOINT` and `/KJOINT` refusals

**Fortran:** `engine/source/tools/lagmul/lag_mult.F` `LAG_MULTP`
  (SPMD-1's citation), `engine/source/elements/joint/ruser33.F`,
  `rgjoint.F` (SPMD-3's citation).
**Files:** Modify `pyradioss/spmd/domdec.py` (remove the refusal),
`pyradioss/engine/engine.py`, `tests/test_p13_gjoint.py`.
**Interfaces:**
- Consumes: Task P13.2's force exchange.
- Produces: `gj.transfer_forces(...)` moved **after** the frontier sum, with the
  joint's non-linear branch resolved on the **merged** partial torques.

- [ ] **Step 1** — write the failing test — the port's own analysis:

```python
def test_gjoint_replica_does_not_double_count_the_reaction():
    """OPEN_BUGS SPMD-1: with T1 on rank A and T2 on rank B, each replica sees
    one non-zero partial torque and takes a different reaction branch than the
    serial run's carrier-only branch 3."""
    serial = _run_gjoint(np=1)
    decomposed = _run_gjoint(np=2)
    assert np.allclose(serial.T01["IE"], decomposed.T01["IE"], rtol=1e-8)
    assert np.allclose(serial.gear_torques, decomposed.gear_torques, rtol=1e-8)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **There are two honest resolutions** and the task
  must pick the one the physics supports:
  - **(a) exchange the joint's partial torques** so every replica sees both
    `T1` and `T2` and takes the same branch as serial. This is a small exchange
    on a tiny array and it is what the upstream refusal implies is needed.
  - **(b) compute the joint on the owning domain only**, the way
    `docs/OPEN_BUGS.md` records upstream doing TYPE33/45 ("computed once on the
    owning domain and then frontier-summed by `spmd_exch_a.F`. They are never
    replicated").
  **(b) is what upstream does and is therefore the answer**; (a) is listed
  because it is the smaller change and the reviewer should check that the
  implementer chose (b) for the cited reason and not out of convenience.
- [ ] **Step 4** — run → PASS; the `kjoints` refusal is lifted with the `WEIGHT`
  convention applied to the additive terms; commit
  `feat(spmd): /GJOINT and /KJOINT under decomposition (lag_mult.LAG_MULTP, ruser33)`.

---

### Task P13.12: Remove the `domdec.py` limitation note

**Fortran:** n/a (documentation).
**Files:** Modify `pyradioss/spmd/domdec.py`.
**Interfaces:**
- Produces: the "Known limitation: a ghost element's `off` flag is never
  updated during the run" paragraph (at `domdec.py:~72`) is **deleted**.

- [ ] **Step 1** — write the failing test:

```python
def test_domdec_no_longer_documents_the_stale_ghost_flag():
    doc = (paths.or_src_root() / "pyradioss/spmd/domdec.py").read_text()
    assert "never updated during the run" not in doc
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — delete the paragraph. **The test is the guard against it
  coming back**, and `docs/OPEN_BUGS.md` SPMD-2's step 4 asks for exactly this.
- [ ] **Step 4** — run → PASS; commit `docs(spmd): remove the stale-ghost-flag limitation note`.

---

### Task P13.13: The `-np N` parity sweep

**Fortran:** `$OR_SRC/qa-tests/`'s parallel-arithmetic check (`test_script.bash`
  `-type=pon`, the "check parallel arithmetic" mode).
**Files:** Modify `tools/validate_vs_fortran.py`, Create
  `tools/validation_data/spmd_parity.json`.
**Interfaces:**
- Produces: for N ∈ {1, 2, 4}, for every deck in the corpus that runs, the
  rel-RMS of the T01 channels against the serial run, and the max over decks.

- [ ] **Step 1** — write the failing test:

```python
def test_spmd_parity_is_within_the_summation_tolerance():
    d = json.loads(Path("tools/validation_data/spmd_parity.json").read_text())
    worst = max(r["rel_rms"] for r in d["cases"])
    assert worst < 1e-8, f"worst -np 4 vs serial rel-RMS {worst}"
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The tolerance is not a free choice.** The port's
  documented contract is "the serial results up to floating-point summation
  order", which for a 4-domain decomposition means differences at the
  ~1e-15 level for O(1e6) accumulations. Set `1e-8` because it is the number in
  the test above and it is loose enough to be about summation and tight enough
  to catch a missing exchange. **If a deck exceeds it, that is a bug, not a
  tolerance to raise.**
- [ ] **Step 4** — run → PASS; commit
  `test(spmd): -np 1/2/4 parity sweep against serial`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P13.0** — the inventory is derived from the call graph, not re-read by hand;
  `not-needed` entries carry reasons.
- **P13.1** — the `WEIGHT` convention is implemented in the **ledger**, per
  `ecrit.F`, not as a force average; `ThreadComm` remains the default.
- **P13.2–P13.9** — the test checks **both** an energy channel and a **momentum**
  channel; the reduction op is per array, not per family.
- **P13.10** — the exchange is **collective and unconditional**; the "same cycle
  everywhere" requirement is met.
- **P13.11** — resolution **(b)** (owning-domain-only) is chosen, with the
  `ruser33.F` citation, and the branch analysis from `docs/OPEN_BUGS.md` is
  resolved rather than sidestepped.
- **P13.13** — the tolerance is not raised to make a failure pass.

## Parallelisation

- **Wave 0** — P13.0 serial.
- **Wave 1** — Task P13.1 first (it defines `Comm`), then the eight family tasks
  in parallel. P13.6 is by far the largest and starts first.
- **Wave 2** — serial; P13.10 is the highest-value task in the phase.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p13_exch_idel.py tests/test_p13_gjoint.py tests/test_p13_mpi_census.py
python -m pytest -q -k spmd
python tools/validate_vs_fortran.py --mode parity --np 4 --out tools/validation_data/spmd_parity.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
```

The phase reviewer confirms, in writing:

1. **`docs/OPEN_BUGS.md` SPMD-1, SPMD-2 and SPMD-3 are all closed**, each with a
   named test;
2. the `domdec.py` limitation paragraph is gone and its test is green;
3. the `-np 4` parity sweep is within `1e-8` on every deck;
4. both `ThreadComm` and `Mpi4pyComm` paths were exercised.