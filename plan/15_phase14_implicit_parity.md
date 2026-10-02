# Phase 14 — Implicit branch parity

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/implicit/` (38 files, 51,269 LOC) a literal
port, **and keep the port's own beyond-upstream tower frozen and green**.

**Architecture:** two halves with opposite rules.
- **The upstream half** (`imp_solv.F`, `imp_dyna.F`, `imp_buck.F`, `imp_dsolv.F`,
  `imp_dt.F`, `imp_int_k.F`, `imp_glob_k.F`, `ind_glob_k.F`, `upd_glob_k.F`,
  `produt_v.F`, the `assem_*` family, `integrator.F`, `lin_solv.F`,
  `nl_solv.F`, `prec_solv.F`, `imp_bfgs.F`, `imp_lanz.F`, `imp_mumps.F`,
  `imp_pcg.F`, `imp_pc_inv.F`, `imp_fsa_inv.F`, …) is **ported and matched**.
- **The port's own half** (`pyradioss/implicit/`'s modal, complex-modal,
  random-response, spectral-fatigue, NORTA and evolutionary-fatigue modules —
  ~25,000 LOC, `00_ORCHESTRATION.md` §3) is **frozen**: no new features, must
  stay green, pinned by `tests/test_p1_frozen_implicit.py`.

**Tech stack:** SciPy (required for implicit; SuperLU default, CHOLMOD/MUMPS
optional); `scipy.sparse.linalg`.
**Spec:** `plan/00_ORCHESTRATION.md` §3.
**Entry criteria:** Phases 2–5 exit gates (every element family has a tangent),
Phase 8 (contact in the Newton loop), Phase 3 Task P3.10 (QBAT/QEPH tangents).
**Band: M.**

---

## Scope

| Upstream file | LOC | Port | Port state |
|---|---:|---|---|
| `imp_solv.F` (the Newton driver) | ~4,500 | `implicit/statics.py` | `Newton-Raphson` with load stepping; arc-length is a *documented deviation* (the spherical Riks metric is computed in the reduced space, M14) |
| `imp_dyna.F` (Newmark/HHT-α) | ~4,000 | `implicit/dynamics.py` | ported (M10); rate devices disabled explicitly |
| `imp_dt.F` (step control) | ~900 | `implicit/statics.py::StepControl` | ported (M11); `IDTC` 2/3 deferred |
| `imp_buck.F` (linearised buckling) | ~1,500 | `implicit/buckling.py` | ported (M14); **documented deviation**: upstream omits contact, the port includes it |
| `imp_dsolv.F` (direct solver) | ~1,200 | `implicit/linsolve.py` | SuperLU + optional CHOLMOD/MUMPS |
| `imp_int_k.F` (contact in Newton) | ~1,800 | `implicit/contact.py` | ported (M12–M14), with **documented deviations** (consistent rather than upstream's diagonal-only friction tangent) |
| `imp_glob_k.F` / `ind_glob_k.F` / `upd_glob_k.F` | ~4,500 | `implicit/assembly.py`, `dofmap.py` | ported (M8–M14) |
| `assem_s8.F`, `assem_s4.F`, `assem_s6.F`, `assem_s20.F`, `assem_r3.F`, `assem_c3.F`, `assem_c4.F`, `assem_q4.F`, `assem_p.F`, `assem_int.F` | ~9,000 | element `tangent()` methods | **incomplete** — the port's element tangents are the M11 deliverable and Phase 3 Task P3.10 adds QBAT/QEPH |
| `imp_init.F`, `imp_sol_init.F`, `imp_setb.F`, `imp_fac_ic.F` | ~2,000 | `implicit/*` | partial |
| `recudis.F`, `produt_v.F`, `cgshell.F`, `imp_dsfext.F`, `imp_bfgs.F`, `imp_lanz.F`, `imp_mumps.F`, `imp_pcg.F`, `imp_pc_inv.F` | ~8,000 | `implicit/linesearch.py`, `bfgs.py` | partial |
| `integrator.F`, `lin_solv.F`, `nl_solv.F`, `prec_solv.F` | ~4,500 | `implicit/statics.py`, `linsolve.py` | partial |
| **Total** | **51,269** | **~21,000 (upstream half) + 25,000 (frozen tower)** | |

## Gap analysis

### 1. Documented deviations — keep or reconcile?

The port records several **deliberate deviations** in the implicit branch. For
a literal port each one is a decision:

| Deviation | Where recorded | Decision needed |
|---|---|---|
| Arc-length metric computed in the **reduced** space, not the full recovered field | `PORTING_GUIDE.md:125` | Both are valid Crisfield parametrisations (they differ by the fixed SPD reweighting `TᵀT`). **Keep and document** — it is more correct under constraints. |
| Friction tangent is the **consistent** one, not `i7keg3.F`'s always-stick `µK` | `PORTING_GUIDE.md:119,129` | Keep. Consistency matters for Newton convergence; the port's rationale is stated. |
| TYPE11 near-parallel edges get a **two-point trapezoid quadrature** | `PORTING_GUIDE.md:120` | Keep — the single closest point of parallel edges is non-unique and flips ends, causing a period-2 Newton cycle (an observed M13 lesson). |
| TYPE11 tangential spring is **capped**; upstream's `i11keg3.F` is uncapped | `PORTING_GUIDE.md:124` | Keep, with the citation. |
| Buckling **includes** contact; upstream omits it entirely | `PORTING_GUIDE.md:126` | Keep — a column resting on a stop would otherwise report the free-column factor. |
| `ZCFAC=1` plastic relaxation **not ported** in QEPH | `VALIDATION.md` §7 item 5 | Phase 3 Task P3.10's scope; exact for elastic laws. |

**None of these is a bug and none should be "fixed" to match upstream.** The
task is to make each one a **tested, cited, permanent decision** rather than a
prose note.

### 2. `IDTC = 2` and `IDTC = 3` deferred

`PORTING_GUIDE.md:114`: "IDTC 2/3 deferred". Read `imp_dt.F`'s `IMP_DTN` and
port them.

### 3. Element tangent completeness

`assem_*.F` has **one assembly per element family**: `s8` (hexa8), `s4`
(tetra4), `s6` (wedge), `s20` (bric20), `r3` (beam), `c3`/`c4` (triangle /
quad shells), `q4` (2-D quad), `p` (thick shell), `int` (contact). The port's
tangents are per-element-module; **the assembly itself** (`assem_*.F`) has no
port counterpart. That is this phase's deliverable.

### 4. The frozen tower's tangent obligation

`00_ORCHESTRATION.md` §3 freezes the tower but requires it to stay green. Phase
3 Task P3.10 adds QBAT/QEPH tangents; **this phase must make sure the frozen
tower still works with them**, or the freeze is violated by omission.

## Wave graph

```
Wave 0  P14.0 implicit reconciliation audit (serial)
   ── gate: every upstream routine classified; every deviation enumerated
Wave 1 (parallel — new modules)
  P14.1 IDTC 2/3 step control
  P14.2 the assem_* element assembly family (one module per family)
  P14.3 recudis / produt_v / cgshell / dsfext (residual and projection helpers)
  P14.4 imp_bfgs / imp_lanz / imp_pcg / imp_pc_inv (solvers)
  P14.5 integrator.F / nl_solv.F / prec_solv.F
  P14.6 the frozen tower's regression suite (characterise it before touching it)
   ── gate per module: Newton convergence + a known-solution test
Wave 2 (serial)
  P14.7 convert each documented deviation into a pinned test
  P14.8 parity + convergence re-measure
```

---

### Task P14.0: Implicit reconciliation audit

**Fortran:** all 38 files in `engine/source/implicit/`.
**Files:** Create `tools/validation_data/implicit_status.json`,
`tests/test_p14_reconciliation.py`.
**Interfaces:**
- Produces: one record per upstream routine; plus a `deviations` list — each
  with `name`, `port_behaviour`, `upstream_behaviour`, `citation`,
  `decision` (`keep` | `reconcile` | `decline`), `reason`.

- [ ] **Step 1** — write the failing test:

```python
def test_every_documented_deviation_has_a_decision_and_a_citation():
    st = json.loads(Path("tools/validation_data/implicit_status.json").read_text())
    for d in st["deviations"]:
        assert d["decision"] in {"keep", "reconcile", "decline"}
        assert d["citation"].startswith(("engine/source/", "common_source/"))
        assert d["reason"]
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit. The six deviations in §Gap analysis must all appear;
  any deviation recorded in `PORTING_GUIDE.md` and **not** in
  `implicit_status.json` is a finding. Cross-check the two sources
  mechanically — that cross-check is the deliverable.
- [ ] **Step 4** — run → PASS; commit
  `docs(implicit): audit + the deviation register with citations`.

---

### Task P14.1: `IDTC = 2` and `IDTC = 3`

**Fortran:** `engine/source/implicit/imp_dt.F` — `IMP_DTN` (cut on `IMCONV < 0`
  bounded by `DT_MIN`; `IDTC = 1` grows toward `DT_MAX` after ≤ `NL_DTP`
  iterations), and the `IDTC = 3` Riks machinery `PORTING_GUIDE.md:125`
  references.
**Files:** Modify `pyradioss/implicit/statics.py`,
`tests/test_p14_stepcontrol.py`.
**Interfaces:**
- Consumes: `implicit.statics.StepControl`.
- Produces: `StepControl.update(converged: bool, iters: int, dt: float,
  arc: bool) -> tuple[float, int]` handling `IDTC ∈ {1,2,3}`.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("idtc,expect", [
    (1, "grow toward DT_MAX"),
    (2, "grow on a second criterion"),
    (3, "Riks arc control"),
])
def test_step_control_distinguishes_the_three_modes(idtc, expect):
    sc = implicit.statics.StepControl(idtc=idtc, dt=1e-6, dt_max=1e-4, dt_min=1e-9)
    new_dt, mode = sc.update(converged=True, iters=3, dt=1e-6, arc=(idtc == 3))
    assert mode == expect
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement from `imp_dt.F`. **Read the `IDTC = 3` branch
  before writing `IDTC = 2`** — in upstream they are not independent: `IDTC = 3`
  enables the Riks machinery that `IDTC = 2` interacts with. `PORTING_GUIDE.md:125`
  says "`IDTC = 3` of `imp_dt.F` + the BFAC load rescaling", so the coupling is
  real.
- [ ] **Step 4** — run → PASS; commit `feat(implicit): IDTC 2 and 3 step control (imp_dt.IMP_DTN)`.

---

### Task P14.2: The `assem_*` element assembly family

**Fortran:** `engine/source/implicit/assem_s8.F` (hexa8), `assem_s4.F`
  (tetra4), `assem_s6.F` (wedge), `assem_s20.F` (bric20), `assem_r3.F` (beam),
  `assem_c3.F` (triangle shell), `assem_c4.F` (quad shell), `assem_q4.F` (2-D
  quad), `assem_p.F` (thick shell), `assem_int.F` (contact).
**Files:** Create `pyradioss/implicit/element_assembly.py` (one function per
  family), `tests/test_p14_element_assembly.py`.
**Interfaces:**
- Consumes: each element module's `tangent(group) -> (triplets, n_dof)`.
- Produces: `element_assembly.assemble(groups, dofmap) -> (rows, cols, vals)`
  — one COO triplet stream for the whole model, which is what
  `implicit/assembly.py` consumes.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("family", ["s8", "s4", "s6", "s20", "r3", "c3", "c4",
                                    "q4", "p", "int"])
def test_assembled_matrix_matches_a_dense_reference(family):
    m = _small_model(family)
    K = element_assembly.to_dense(element_assembly.assemble(m.groups, m.dofmap))
    assert np.allclose(K, _dense_reference(m), rtol=1e-10, atol=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, one function per family, mirroring the `assem_*.F`
  triplet loop **order by order** — the ordering determines the CSR fill-in
  pattern, which determines the factorisation time and, for an iterative solver,
  the iteration count. `assemble_int.F` is the contact one and its triplets come
  from `implicit/contact.py`'s `_friction_state`/`triplets`.
- [ ] **Step 4** — run → PASS; the dense comparison is `1e-10`; commit per family.

---

### Task P14.3: Residual and projection helpers

**Fortran:** `engine/source/implicit/recudis.F`, `produt_v.F`, `cgshell.F`,
  `imp_dsfext.F`.
**Files:** Create `pyradioss/implicit/projection.py`, `tests/test_p14_projection.py`.
**Interfaces:**
- Produces: `projection.shell_projection(dofmap) -> np.ndarray`,
  `projection.reduced_dot(a, b, T) -> float` (`produt_v.F`: the full-field dot
  after dependent-motion recovery), `projection.deflation(Q) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_reduced_dot_equals_the_full_field_dot_after_recovery():
    rng = np.random.default_rng(0)
    u = rng.normal(size=20); v = rng.normal(size=20)
    T = _rigid_transform(6)
    up, vp = T @ u, T @ v
    assert projection.reduced_dot(up, vp, T) == pytest.approx(u @ v, rel=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **`produt_v.F` is the metric `PORTING_GUIDE.md:125`
  says the port computes in the reduced space instead** — so this task ports the
  upstream full-field version too, and the two become a **cross-check**: for an
  unconstrained problem they must agree exactly. That is a stronger test than
  either alone and it validates the port's documented deviation.
- [ ] **Step 4** — run → PASS; commit `feat(implicit): residual/projection helpers with a cross-check`.

---

### Task P14.4: The alternative solvers

**Fortran:** `engine/source/implicit/imp_bfgs.F`, `imp_lanz.F` (Lanczos),
  `imp_pcg.F`, `imp_pc_inv.F`, `imp_mumps.F`, `imp_fsa_inv.F`, `imp_dsolv.F`.
**Files:** Create `pyradioss/implicit/solvers/` package
(`bfgs.py`, `lanczos.py`, `pcg.py`, `mumps.py`, `superlu.py`),
`tests/test_p14_solvers.py`.
**Interfaces:**
- Produces: `solvers.SOLVERS: dict[str, LinearSolver]` with
  `LinearSolver.solve(K, b, x0=None) -> np.ndarray`,
  `LinearSolver.solve_deflation(K, b, Q) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("name", sorted(solvers.SOLVERS))
def test_every_solver_solves_a_known_spd_system(name):
    A = _spd(40, seed=1); b = _rhs(40)
    x = solvers.SOLVERS[name].solve(A, b)
    assert np.allclose(A @ x, b, rtol=1e-8)
```

- [ ] **Step 2** — run → FAIL (only SuperLU + CHOLMOD/MUMPS wrappers exist).
- [ ] **Step 3** — implement. **`imp_mumps.F` and `imp_fsa_inv.F` wrap external
  libraries**; the port's existing optional-dependency pattern
  (`python-mumps`, `scikit-sparse`) is the model, and the fallback must warn
  and continue. **Do not promote either to a base dependency.**
- [ ] **Step 4** — run → PASS; commit per solver.

---

### Task P14.5: `integrator.F`, `nl_solv.F`, `prec_solv.F`

**Fortran:** `engine/source/implicit/integrator.F` (the driver above
  `imp_solv.F`), `nl_solv.F` (the nonlinear loop), `prec_solv.F` (the
  preconditioner setup), `imp_init.F`, `imp_sol_init.F`, `imp_setb.F`,
  `imp_fac_ic.F`.
**Files:** Create `pyradioss/implicit/integrator.py`, `tests/test_p14_integrator.py`.
**Interfaces:**
- Produces: `integrator.run(model, ctx) -> ImplicitResult`, the single entry
  point that Phase 15's element tangents and Phase 8's contact feed.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("card", ["/IMPL/NONLIN", "/IMPL/ARCL", "/IMPL/DYNA",
                                  "/IMPL/BUCKL"])
def test_each_implicit_mode_converges_on_the_cantilever(card):
    r = _run_implicit(card, _cantilever_deck())
    assert r.converged
    assert r.iterations <= r.max_iterations
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **This task must not change any convergence
  behaviour** — it is a restructure into a single entry point. The four tests
  above are the guard: any change in iteration count is a regression, and the
  reviewer should see the before/after iteration counts.
- [ ] **Step 4** — run → PASS; iteration counts unchanged; commit
  `refactor(implicit): one integrator entry point, behaviour unchanged`.

---

### Task P14.6: Characterise the frozen tower before touching it

**Fortran:** none (the tower is the port's own).
**Files:** Create `tests/test_p14_frozen_regression.py`,
`tools/validation_data/frozen_baseline.json`.
**Interfaces:**
- Produces: a baseline of the frozen tower's outputs on the bundled implicit
  examples (`examples/{implicit_cantilever, implicit_ringdown, modal_mast,
  random_vibration, spectral_fatigue, multiaxial_fatigue, nongaussian_fatigue,
  evolutionary_fatigue, complex_modes, modal_frf, …}`) — convergence
  tolerances, mode frequencies, damage values — recorded **before** any change.

- [ ] **Step 1** — write the failing test:

```python
def test_frozen_tower_baseline_is_recorded_for_every_implicit_example():
    base = json.loads(Path("tools/validation_data/frozen_baseline.json").read_text())
    for d in sorted((paths.root() / "examples").iterdir()):
        if d.name.startswith(("implicit_", "modal_", "random_", "spectral_",
                              "fatigue", "multiaxial", "nongaussian",
                              "evolutionary", "complex_", "exact_covariance",
                              "wigner", "multi_input", "freq_", "ks2")):
            assert d.name in base, d.name
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **This is the freeze's enforcement mechanism.**
  Every subsequent task in this phase re-runs it, and any change to a frozen
  value requires an explicit maintainer decision. Taking this baseline **after**
  a change would be useless, so it lands first.
- [ ] **Step 4** — run → PASS; commit
  `test(implicit): baseline the frozen tower before any change`.

---

### Task P14.7: Convert each deviation into a pinned test

**Fortran:** the citations in `implicit_status.json`.
**Files:** Create `tests/test_p14_deviations.py`.
**Interfaces:**
- Produces: one test per deviation, asserting the **port's** behaviour and
  citing the upstream behaviour it deliberately differs from.

- [ ] **Step 1** — write the failing test:

```python
def test_arc_length_metric_is_the_reduced_space_one():
    """PORTING_GUIDE.md:125 — the reduced metric differs from upstream's full
    recovered field by the fixed SPD reweighting T^T T; both are valid."""
    r = _run_arc_length(_snap_through_deck())
    assert r.metric_space == "reduced"
    assert r.converged

def test_buckling_includes_contact_which_upstream_omits():
    """PORTING_GUIDE.md:126 — a column on a stop must not report the free
    column's factor."""
    a = _run_buckling(_column_on_a_stop())
    b = _run_buckling(_free_column())
    assert a.factors[0] != pytest.approx(b.factors[0], rel=1e-3)
```

- [ ] **Step 2** — run → FAIL (some may pass; the deviation list drives it).
- [ ] **Step 3** — implement, one test per deviation. **A deviation test asserts
  the port's behaviour, not upstream's.** The point is to make the difference
  permanent and visible, so that nobody later "fixes" it by accident.
- [ ] **Step 4** — run → PASS; `implicit_status.json`'s deviations all have a
  test; commit `test(implicit): pin every documented deviation`.

---

### Task P14.8: Parity and convergence re-measure

**Fortran:** the implicit decks in the corpus that upstream runs.
**Files:** Modify `tools/validate_vs_fortran.py`,
`docs/OUTPUT_FORMATS.md` (implicit listing section).
**Interfaces:**
- Produces: `/IMPL/*` parity cases and convergence statistics.

- [ ] **Step 1** — write the failing test:

```python
def test_implicit_cases_have_parity_records():
    d = load_parity("tools/validation_data/parity_p14.json")
    assert d["cases"], "no implicit parity cases recorded"
    assert all("rel_rms" in c for c in d["cases"])
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Implicit parity is harder than explicit parity**
  and the reason must be recorded: an implicit solution is the root of a
  nonlinear system, so a small formulation difference shows as a *different
  converged state* rather than a growing error. Use Newton **iteration counts**
  as the primary comparison and the converged state as the secondary.
- [ ] **Step 4** — run → PASS; commit
  `docs(validation): implicit parity — iteration counts first, converged state second`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P14.0** — every deviation in `PORTING_GUIDE.md` appears in the register, and
  the cross-check is mechanical.
- **P14.1** — `IDTC = 2`/`3` are read together, not independently.
- **P14.2** — the triplet order mirrors `assem_*.F`, and the dense comparison is
  `1e-10`.
- **P14.3** — the port's reduced metric and upstream's full-field metric are
  cross-checked to agree on an unconstrained problem.
- **P14.4** — the optional dependencies stay optional and warn on absence.
- **P14.5** — iteration counts are quoted before and after; no behaviour change.
- **P14.6** — the baseline lands **before** any change, and every implicit
  example is in it.
- **P14.7** — deviation tests assert the **port's** behaviour with the upstream
  difference cited.
- **P14.8** — iteration counts are the primary metric, and that choice is
  justified in the report.

## Parallelisation

- **Wave 0** — P14.0 serial.
- **Wave 1** — 6 tasks. **P14.6 must be first** (the baseline is worthless
  after a change). P14.2 is internally ten sub-tasks, one per element family.
- **Wave 2** — serial.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p1_frozen_implicit.py tests/test_p14_frozen_regression.py
python -m pytest -q tests/test_p14_reconciliation.py tests/test_p14_deviations.py
python -m pytest -q tests/test_p14_element_assembly.py
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area engine/source/implicit
```

The phase reviewer confirms, in writing:

1. **the frozen tower's baseline is unchanged** — every value in
   `frozen_baseline.json` matches, or the commit says which and why;
2. every upstream `implicit/` routine is `ported` or has a cited reason;
3. every documented deviation has a test;
4. no frozen module was added or removed (Task P1.6's test is green).