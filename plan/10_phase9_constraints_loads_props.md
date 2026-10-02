# Phase 9 — Constraints, loads, properties, boundary conditions

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `starter/source/{constraints,properties,loads,boundary_conditions}`
and `engine/source/{constraints,loads,properties,boundary_conditions,coupling}`
a literal port. This phase owns the **glue layer**: everything that turns a
parsed card into a force, a mass, or a stiffness contribution, plus every
`/PROP` family.

**Architecture:** split by *card family*, not by upstream directory, because
the upstream directories mix reader and physics and the port needs them
separated. The phase's central deliverable is a **`/PROP` census** — 51 property
types upstream (`starter/source/properties/` has beam, composite_options,
injector, rivet, shell, solid, sph, spring, thickshell, truss,
user_spring_solid, void, xelem) against a port that has hand readers for
TYPE1/2/3/4/14 and a cfg-driven generic reader for the rest.

**Tech stack:** NumPy, SciPy (for `/MPC` condensation, already in the implicit
branch).
**Spec:** `plan/00_ORCHESTRATION.md` §10 item 5.
**Entry criteria:** Phase 1 exit gate; Phases 2–5 exit gates (the properties
attach to elements).
**Band: L.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `starter/source/properties/**` (13 dirs) | 57 | 26,572 | `input/prop_reader.py` (1,647) + `input/keywords/properties.py` + `prop_composite.py`, `prop_rivet.py`, `prop_sandwich.py`, `prop_shell_type16.py` |
| `starter/source/constraints/**` (6 dirs) | 74 | 26,984 | `engine/rigid_body.py`, `rbe3.py`, `mpc.py`, `gjoint.py`, `kjoint.py`, `cyl_joint.py`, `rlink.py`, `flex_body.py`, `nbcs.py`, `bcs_cyclic.py`, `bcs_nrf.py`, `bcs_wall.py`, `engine/kinematics.py` |
| `engine/source/constraints/**` (3 dirs) | 69 | 34,596 | as above + `engine/rigid_wall.py`, `rwall_thermal.py` |
| `starter/source/loads/**` (8 dirs) | 40 | 10,214 | `input/keywords/loads.py`, `engine/kinematics.py`, `load_centri.py`, `load_pcyl.py`, `pblast.py`, `pfluid.py`, `laser.py`, `impacc.py`, `centri.py`, `bolt_preload.py` |
| `engine/source/loads/**` (3 dirs) | 18 | 6,666 | as above |
| `starter/source/boundary_conditions` | 20 | 4,459 | `input/keywords/boundary_conditions.py` |
| `engine/source/boundary_conditions` | 14 | 3,089 | `engine/kinematics.py`, `engine/bcs_*.py` |
| `engine/source/coupling` | 4 | 2,173 | `engine/fluid.py`? — verify |
| **Total** | **296** | **115,000** | **~12,000** |

### Upstream card families (from the directories)

**Constraints** — `starter/source/constraints/general/`:
`bcs`, `cyl_joint`, `gjoint`, `impvel`, `kinchk.F`, `kinini.F`, `kinset.F`,
`merge`, `mpc`, `rbe2`, `rbe3`, `rbody`, `rwall` (engine adds `rlink`,
`kinini.F`).
`starter/source/constraints/fxbody/` (6,186 LOC) — the **flexible-body**
constraint family, with 11 engine routines (`fxbdispl.F`, `fxbodfp.F`,
`fxbodv.F`, `fxbodvp.F`, `fxbsgmaj.F`, `fxbsys.F`, `fxbyfor.F`, `fxbypid.F`,
`fxbyvit.F`, `fxgrvcor.F`, `fxbody`).
`starter/source/constraints/{ale,sph,thermic,rigidlink}` — the ALE/SPH/thermal
constraint variants (353/192/718/345).

**Loads** — `starter/source/loads/{general,laser,pblast,bem,bolt,
reference_state,sph,thermic}`, `engine/source/loads/{general,laser,pblast}`.

**Properties** — `starter/source/properties/`:
`beam` (2,266), `composite_options`, `injector` (932), `rivet` (135),
`shell` (4,409), `solid` (1,876), `sph` (318), `spring` (**11,665**),
`thickshell` (1,430), `truss` (137), `user_spring_solid` (381), `void` (157),
`xelem` (417), plus 1,398 of top-level readers.

## Gap analysis

### 1. The `/PROP` surface

The port's `input/prop_reader.py` is a cfg-driven generic reader producing real
parameters for a handful of types and `InactiveProperty` for the rest
(`PORTING_GUIDE.md:39`). `docs/STATE.md` M153–M195 record a long tail of
readers added per milestone (TYPE5/6/8/9/10/11/12/13/14/15/16/17/18/19/20/21/
22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/43/44/45/46/51, PCOMPP, TSHELL,
TSH_ORTH, TSH_COMP, SH_SANDW, SH_FABR, STACK, SPR_PRED/PRE/PUL/BEAM/MAT/GENE/
TORS/AXI/MUSC/TAB/BDAMP/CRUS, INJECT1/2, RIVET, XELEM, NSTRAND, SEW, SPH,
CONNECT, PRELOAD, PCOMPP).

What is missing is **the census**: which of those produce *physics the engine
uses*, versus which parse into a record nothing reads. `tests/test_mat_all_135_census.py`
does the analogous job for materials; there is no property equivalent. **That is
Task P9.0.**

### 2. `flex_body` (`/FXBODY`) — 11 engine routines, 2,734 + 6,186 LOC

The port has `engine/flex_body.py` — **736 lines** against 2,734 (engine) +
6,186 (Starter) = 8,920 Fortran lines, a 1:12 ratio — cited at
`starter/source/constraints/fxbody/ini_fxbody.F`. Which of the 11 engine
routines are ported is the audit question.

### 3. Constraints with a refusal in the port

`docs/OPEN_BUGS.md` records three **refusals** rather than ports, all citing
upstream:

| Refusal | Upstream citation |
|---|---|
| `/GJOINT` under `-np > 1` | `engine/source/tools/lagmul/lag_mult.F`, `LAG_MULTP` ~683-687 |
| `/INTER/TYPE2` with element deletion under `-np > 1` | `interfaces/interf/chkstfn3.F`, `SPMD_EXCH_IDEL` 1129-1136 |
| `model.kjoints` under `-np > 1` | `elements/joint/ruser33.F`, `spmd_exch_a.F` |

**All three refusals are SPMD-scoped, not physics-scoped.** The physics exists
in serial. Phase 13's job is to lift them, not Phase 9's. **This phase must
verify that** and record it — a refusal that is accidentally permanent would be
a silent capability loss.

### 4. Loose threads in the port

- `engine/nbcs.py` (`/NBCS`, M119) — no upstream directory of that name;
  probably a Simcenter-only feature. Audit.
- `engine/laser.py`, `engine/pfluid.py`, `engine/impacc.py`, `engine/centri.py`,
  `engine/load_pcyl.py` — all present; each needs its audit.
- `starter/source/loads/bem/` and `engine/…/bem` — `/BEM/FLOW`, `/BEM/DAA`
  (M119); the port has `engine/bem_flow.py` and `engine/daa.py`.
- `starter/source/loads/reference_state/` — `/REFSTA`, `/EREF` (M119); the port
  has `model/refsta.py`.
- `starter/source/constraints/general/merge.F` — `/MERGE` (M102,
  `/MERGE/NODE` M130); the port has `fix_entities.py`-era code.

## Wave graph

```
Wave 0  P9.0 property census (serial — the checklist)
        P9.1 constraint-family census + verify the three refusals are SPMD-only
        P9.2 load-family census
   ── gate: every /PROP type and every constraint card classified
Wave 1 (parallel — new modules)
  P9.3 flex_body (fxbody) — the largest unmeasured constraint family
  P9.4 fxbody Starter assembly + mass/inertia
  P9.5 the constraint-set builder (kinset/kinchk/kinini)
  P9.6 thermal constraints + RBCs
  P9.7 load families: laser, pblast, bem, reference_state
  P9.8 ALE/SPH constraint variants
   ── gate per module: analytic test + one oracle case
Wave 2 (serial)
  P9.9 wire every property type into its element group; refuse the rest
  P9.10 coverage re-measure
```

---

### Task P9.0: The `/PROP` census

**Fortran:** `starter/source/properties/` — every `hm_read_prop*.F`, the
  `shell/hm_read_prop01/09/10/11/17/51.F` variants (all reading `Ishell`),
  `spring/hm_read_prop*.F` (24 files incl. the nine `prop33_*` joint variants),
  `solid/`, `beam/`, `thickshell/`, `truss/`, `rivet/`, `injector/`, `sph/`,
  `void/`, `xelem/`, `user_spring_solid/`; the void reader
  (`hm_read_prop00` or equivalent) that produces the no-stiffness placeholder.
**Files:** Create `tools/validation_data/prop_status.json`,
`tests/test_p9_prop_census.py`.
**Interfaces:**
- Produces: one record per property type:
  `{"type": 51, "name": "PCOMPP|SH_COH", "starter_loc": 4409,
    "shell_ishell_ok": True, "port_reader": "prop_composite.py",
    "port_consumer": "shell_qbat._material_layers" | null,
    "status": "live|parsed-unused|refused|missing",
    "citation": "starter/source/properties/shell/hm_read_prop51.F"}`
  plus `"types_by_family"`.

- [ ] **Step 1** — write the failing test — the one that matters:

```python
def test_no_property_type_is_parsed_and_never_consumed():
    st = json.loads(Path("tools/validation_data/prop_status.json").read_text())
    silent = [r["type"] for r in st if r["status"] == "parsed-unused"]
    assert silent == [], f"{len(silent)} property types parse into nothing: {silent}"
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit. For each type: does anything in `pyradioss/elements/`
  or `pyradioss/engine/` read the parsed record? **`parsed-unused` is the defect
  class this census exists to find** — it is the reader-side twin of the 440
  corpus SKIPS, and it is silent: the deck parses, the model builds, the run
  completes, and the property does nothing.
- [ ] **Step 4** — run → PASS; commit `feat(census): property-type status and consumer census`.

---

### Task P9.1: Constraint-family census and refusal verification

**Fortran:** `starter/source/constraints/general/*`, `engine/source/constraints/
{general,fxbody,thermic}/`.
**Files:** Create `tools/validation_data/constraint_status.json`,
`tests/test_p9_constraint_census.py`.
**Interfaces:**
- Produces: one record per constraint card, plus
  `constraint_status.refusals: dict[str, str]` — each of the three
  `docs/OPEN_BUGS.md` refusals with its upstream citation and a flag
  `spmd_only: bool`.

- [ ] **Step 1** — write the failing test:

```python
def test_known_refusals_are_spmd_only_not_physics_refusals():
    """docs/OPEN_BUGS.md refuses these three under -np; they must still WORK in
    serial. A permanent refusal would be a silent capability loss."""
    for name, cite in checks.refusals().items():
        assert cite.startswith(("engine/source/", "starter/source/")), name
    assert checks.refusals_are_spmd_only()

def test_gjoint_runs_in_serial():
    s, e = make_deck("GJT", _gjoint_starter_deck(), _gjoint_engine_deck())
    r = _run(s, e)
    assert r.termination == "NORMAL"
```

- [ ] **Step 2** — run → FAIL (no `checks.refusals()`).
- [ ] **Step 3** — implement. **Run the serial GJOINT test first** and record
  whether it passes today. If it does not, the refusal is *not* SPMD-only and
  the finding goes in `docs/OPEN_BUGS.md` with the reproduction, because that
  is a live capability gap rather than a Phase 13 item.
- [ ] **Step 4** — run → PASS; commit
  `test(constraints): census, and the three refusals proven serial-clean`.

---

### Task P9.2: Load-family census

**Fortran:** `starter/source/loads/{general,laser,pblast,bem,bolt,
reference_state,sph,thermic}`, `engine/source/loads/{general,laser,pblast}`.
**Files:** Create `tools/validation_data/load_status.json`,
`tests/test_p9_load_census.py`.
**Interfaces:**
- Produces: one record per load card with `booked: bool` — **whether the load
  books work into the EN ledger**. A load that applies force and books no work
  is an energy leak the domain rules forbid.

- [ ] **Step 1** — write the failing test:

```python
def test_every_load_that_applies_force_books_its_work():
    st = json.loads(Path("tools/validation_data/load_status.json").read_text())
    unbooked = [r["name"] for r in st if r["applies_force"] and not r["booked"]]
    assert unbooked == [], unbooked
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit `engine/kinematics.py`'s `external_forces` (which holds
  `/PLOAD` and the concentrated loads), `load_centri.py`, `load_pcyl.py`,
  `pblast.py`, `pfluid.py`, `laser.py`, `impacc.py`, `bolt_preload.py`,
  `bem_flow.py`, `daa.py`, and the thermal loads
  (`thermal_loads.py`, `thermal_solver.py`).
- [ ] **Step 4** — run → PASS; commit `feat(census): load-family status with energy-booking audit`.

---

### Task P9.3: `flex_body` (`/FXBODY`) — the engine half

**Fortran:** `engine/source/constraints/fxbody/` — `fxbyfor.F` (the force),
  `fxbyvit.F`, `fxbodv.F`, `fxbodvp.F`, `fxbdispl.F`, `fxbodfp.F`,
  `fxbsgmaj.F` (the segment-major assembly), `fxbsys.F`, `fxbypid.F`,
  `fxgrvcor.F` (the gravity correction).
**Files:** Create `pyradioss/engine/fxbody_forces.py`,
`tests/test_p9_fxbody.py`.
**Interfaces:**
- Consumes: the `/FXBODY` geometry from Task P9.4.
- Produces: `fxbody_forces.forces(fxbodies, x, v, dt, fint, mint) -> np.ndarray`
  — and it **books work**.

- [ ] **Step 1** — write the failing test:

```python
def test_flexible_body_carries_load_without_rigid_arms():
    """A /FXBODY is a deformable body: its stiffness is finite, so under a
    nodal load its nodes deflect. A rigid-body port would give zero."""
    fx = _flexible_body(n_nodes=12, stiffness=1e6)
    fx.v[0] = 1.0
    f = fxbody_forces.forces(fx, fx.x, fx.v, dt=1e-6, fint=np.zeros((12,3)),
                             mint=np.zeros((12,3)))
    assert np.abs(f[0]).sum() > 0.0

def test_flexible_body_conserves_momentum():
    fx = _flexible_body(n_nodes=12, stiffness=1e6)
    f = fxbody_forces.forces(fx, fx.x, fx.v, dt=1e-6, fint=np.zeros((12,3)),
                             mint=np.zeros((12,3)))
    assert np.allclose(f.sum(0), 0.0, atol=1e-12)
```

- [ ] **Step 2** — run → FAIL (or, if `flex_body.py` already does this, the
  audit task records it and the test passes — **which is a legitimate outcome
  and must be recorded as such, not forced**).
- [ ] **Step 3** — implement. `fxbsgmaj.F`'s "segment-major" assembly is the
  structural difference from `/RBODY`: the body is a set of *interconnected
  segments* with local stiffness, not a 6-DOF rigid frame. Reading `fxbyfor.F`
  and mistaking it for `rbyfor.F` is the obvious error; the deflect-under-load
  test is what distinguishes the two.
- [ ] **Step 4** — run → PASS; work booked; commit
  `feat(constraints): flexible-body engine forces (fxbody)`.

---

### Task P9.4: `/FXBODY` Starter assembly

**Fortran:** `starter/source/constraints/fxbody/` (6,186 LOC, the largest
  constraint directory on the Starter side) — including `ini_fxbody.F`, which
  the port's `flex_body.py` docstring already cites.
**Files:** Create `pyradioss/starter/fxbody_init.py`,
`tests/test_p9_fxbody_init.py`.
**Interfaces:**
- Produces: `fxbody_init.initialize(model, log) -> np.ndarray` — the body
  connectivity, local frames, mass and rotational inertia, matching the pattern
  `starter/initialization.py::initialize_rigid_bodies` already uses for
  `/RBODY`.

- [ ] **Step 1** — write the failing test:

```python
def test_fxbody_init_matches_the_rbodymass_invariant():
    fx = fxbody_init.initialize(_model_with_fxbody(), MessageLog())
    assert fx.mass.sum() == pytest.approx(_deck_total_mass(), rel=1e-12)
    assert np.linalg.norm(fx.centre_of_mass - _deck_com()) < 1e-12
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Use the same invariants `/RBODY` already
  satisfies** (total mass, centre of mass, the inertia tensor about the COM) —
  they are checkable without the oracle and they catch most assembly errors.
  `ini_fxbody.F` is 6,186 LOC of directory; audit it directory by directory as
  Phase 8 Task P8.7 does for `inter3d1`.
- [ ] **Step 4** — run → PASS; commit `feat(constraints): /FXBODY Starter assembly and inertia`.

---

### Task P9.5: The constraint-set builder

**Fortran:** `starter/source/constraints/general/{kinset.F, kinchk.F,
  kinini.F}` — the set/master-slave resolution pass shared by `/RBE2`, `/RBE3`,
  `/MPC`, `/RBE2` pendulums.
**Files:** Create `pyradioss/starter/constraint_set.py`,
`tests/test_p9_constraint_set.py`.
**Interfaces:**
- Produces: `constraint_set.build(model, log) -> ConstraintSet` with
  `ConstraintSet.masters: np.ndarray`, `.slaves: np.ndarray`,
  `.weights: np.ndarray` (CSR), `.chains: list[list[int]]`,
  `.circular: list[list[int]]`.

- [ ] **Step 1** — write the failing test — the chain machinery is the point
  (`docs/STATE.md` M14's implicit constraint chains):

```python
def test_chains_resolve_by_substitution_in_topological_order():
    cs = constraint_set.build(_model_with_rigid_on_rigid(), MessageLog())
    assert cs.circular == []
    assert cs.chains, "a rigid body whose master is an RBE3 slave is a chain"
    assert len(cs.chains[0]) >= 2

def test_circular_chains_are_refused_not_silently_dropped():
    with pytest.raises(CircularConstraintError):
        constraint_set.build(_model_with_circular_chain(), MessageLog())
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The implicit branch (`pyradioss/implicit/
  constraints.py::_fully_resolve`, M14) already implements topological chain
  resolution for the *implicit* system. **This task extracts that logic into
  the shared builder and makes the explicit path use it** — a duplicated
  resolution algorithm in two branches is a divergence waiting to happen.
  `CircularConstraintError` is the named refusal.
- [ ] **Step 4** — run → PASS; the implicit branch's chain tests still pass
  (they now exercise the shared code); commit
  `refactor(constraints): one constraint-set builder shared by explicit and implicit`.

---

### Task P9.6: Thermal constraints and RBCs

**Fortran:** `starter/source/constraints/thermic/` (718),
  `engine/source/constraints/thermic/` (1,786), plus the port's
  `engine/thermal_loads.py`, `engine/thermal_solver.py`, `engine/rwall_thermal.py`.
**Files:** Create `pyradioss/engine/thermal_constraints.py`,
`tests/test_p9_thermal_constraints.py`.
**Interfaces:**
- Produces: `thermal_constraints.apply(model, cycle) -> np.ndarray` — the
  thermal boundary conditions and the temperature–displacement coupling force,
  booked to the thermal ledger.

- [ ] **Step 1** — write the failing test:

```python
def test_a_fixed_temperature_holds_its_node():
    n = _single_node_with_temperature(t0=300.0)
    for _ in range(50):
        thermal_constraints.apply(_model(n), cycle=_)
    assert n.T[0] == pytest.approx(300.0)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Thermal loads expand the state vector**
  (temperature is an extra DOF per node), which interacts with the frozen
  implicit tower's `DofMap` and with Phase 13's SPMD exchanges. Declare both
  dependencies in the module docstring; do not try to solve them here.
- [ ] **Step 4** — run → PASS; commit `feat(constraints): thermal boundary conditions and coupling`.

---

### Task P9.7: Load families

**Fortran:** `starter/source/loads/{laser,pblast,bem,bolt,reference_state}`,
  `engine/source/loads/{laser,pblast}`.
**Files:** Create `pyradioss/engine/loads/` package
(`laser.py`, `pblast.py`, `bem_flow.py`, `daa.py`, `preload.py`,
`refsta.py`, `centrifugal.py`, `pcyl.py`, `pfluid.py`, `impacc.py`),
`tests/test_p9_loads_*.py`.
**Interfaces:**
- Consumes: nothing new (each load reads a parsed card).
- Produces: one module per family with `forces(model, t, dt) -> np.ndarray`
  and `booked_work(...) -> float`.

- [ ] **Step 1** — write the failing test per family. The `/LOAD/PBLAST`
  example, because a blast pressure profile has a definite analytic form:

```python
def test_pblast_profile_matches_the_closed_form():
    t = np.linspace(0, 1e-3, 100)
    p = load_pblast.profile(t, p_ref=1e9, t_rise=1e-5, t_fall=5e-4)
    assert p[0] == 0.0
    assert p.max() > 0.0
    assert np.all(np.diff(p[p.argmax():]) <= 1e-6)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, one family per task. **`/LOAD/PBLAST` and
  `/LASER` are the two with real time histories**; both must book their work at
  the leapfrog-consistent midstep velocity like the rest of the engine.
- [ ] **Step 4** — run → PASS; commit per family.

---

### Task P9.8: ALE / SPH constraint variants

**Fortran:** `starter/source/constraints/{ale,sph}` (353 + 192),
  `engine/source/constraints/thermic`; the `/ALE/GRID/*` card set
  (`/ALE/GRID/DONEA`, `/ALE/GRID/SPRING`, `/ALE/GRID/STANDARD`, `/ALE/GRID/DISP`,
  `/ALE/GRID/LAPLACIAN`, `/ALE/GRID/VOLUME` per M113) and
  `/SPHBCS`, `/SPH/INOUT`.
**Files:** Create `pyradioss/engine/ale_grid_constraints.py`,
`tests/test_p9_ale_grid.py`.
**Interfaces:**
- Consumes: Phase 11's ALE grid; Phase 5's SPH boundary.
- Produces: `ale_grid_constraints.apply(grid, model, cycle) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_ale_grid_donea_moves_the_grid_by_the_mesh_velocity():
    g = _ale_grid()
    v = _mesh_velocity_field(g)
    before = g.x.copy()
    ale_grid_constraints.apply(g, _model_with(v), cycle=1, dt=1e-6)
    assert not np.allclose(g.x, before)
    assert np.allclose(g.x - before, v * 1e-6, atol=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **This task depends on Phase 11's grid, so it is
  scheduled in Phase 11's tail, not here.** This phase's contribution is the
  *census* of the `/ALE/GRID/*` and `/SPHBCS` cards (Task P9.2's table) so
  Phase 11 knows what it inherits.
- [ ] **Step 4** — run → PASS; commit `feat(constraints): ALE grid and SPH boundary constraints`.

---

### Task P9.9: Wire every property type into its element group

**Fortran:** `starter/source/elements/reader/` (the per-element property
  dispatch), `engine/source/properties/`.
**Files:** Modify `pyradioss/elements/__init__.py`, `pyradioss/input/checks.py`,
  `pyradioss/starter/initialization.py`.
**Interfaces:**
- Produces: for every type in `prop_status.json`, either a consumer in an
  element kernel or an entry in `checks.unsupported_prop_type(n)` with its
  citation. **Exhaustiveness is the deliverable.**

- [ ] **Step 1** — write the failing test:

```python
def test_every_property_type_is_consumed_or_cited_refused():
    st = json.loads(Path("tools/validation_data/prop_status.json").read_text())
    from input.checks import unsupported_prop_type
    unhandled = [r["type"] for r in st
                 if r["type"] not in unsupported_prop_type()
                 and not _has_consumer(r["type"])]
    assert unhandled == [], unhandled
```

- [ ] **Step 2** — run → FAIL (`parsed-unused` types surface here).
- [ ] **Step 3** — implement. **This is the task that closes the
  `parsed-unused` class.** Each such type gets either a kernel consumer (a task
  in Phases 2–5, raised as a blocker) or a cited refusal. A reviewer who accepts
  "it parses" as resolution has missed the point.
- [ ] **Step 4** — run → PASS; commit
  `feat(properties): every /PROP type consumed or cited-refused`.

---

### Task P9.10: Coverage re-measure

**Files:** Modify `tools/validate_vs_fortran.py` (add the property/constraint
  blocker families to the ranking).
**Interfaces:**
- Produces: `prop_status` and `constraint_status` families appear in
  `ranked_gaps` so they are visible next to `INTER/*`.

- [ ] **Step 1** — write the failing test:

```python
def test_ranked_gaps_include_property_and_constraint_families():
    g = load_coverage("tools/validation_data/coverage_p9.json")["ranked_gaps"]
    fams = {r["family"] for r in g}
    assert any(f.startswith("PROP/") for f in fams)
    assert any(f.startswith("RBE") or f.startswith("FXBODY") for f in fams)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, by attributing each `SKIPS`/`ERROR` deck's
  blocker to a property or constraint family where one applies.
- [ ] **Step 4** — run → PASS; commit `feat(validation): property and constraint blocker ranking`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P9.0** — `parsed-unused` is empty, or each remaining entry has a filed
  blocker task id. **This is Review Focus item 1 for `/PROP`**: a deck with a
  property the port parses but has no physics for must produce a *named refusal*,
  never a silent no-op.
- **P9.1** — the three `docs/OPEN_BUGS.md` refusals are demonstrated to work in
  **serial**, with the run output quoted.
- **P9.2** — every force-applying load books work; the energy-balance tests back
  the claim.
- **P9.3** — the flex-body deflects under load (a rigid implementation fails);
  momentum is conserved.
- **P9.4** — mass and COM match the deck; the assembly is transcribed from
  `ini_fxbody.F`.
- **P9.5** — the explicit path now uses the shared builder; the implicit branch's
  chain tests still pass unmodified.
- **P9.9** — exhaustiveness holds; "it parses" is not accepted as a resolution.

## Parallelisation

- **Wave 0** — P9.0 serial (the checklist). P9.1 and P9.2 parallel.
- **Wave 1** — 6 tasks. **P9.5 must land before P9.3/P9.4 if those consume the
  builder** — they do not (fxbody is its own family), so all six can run in
  parallel. P9.8's dependency on Phase 11 is declared and its Phase-9
  contribution is the census only.
- **Wave 2** — serial.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p9_prop_census.py tests/test_p9_load_census.py tests/test_p9_constraint_census.py
python tools/validate_vs_fortran.py --mode coverage --out tools/validation_data/coverage_p9.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area starter/source/properties
```

The phase reviewer reports: the number of `/PROP` types live / refused (with
`parsed-unused` = 0), the number of load families with unbooked work (= 0), and
the corpus coverage delta.