# Phase 11 — ALE, Euler, airbags, multifluid, FSI, advanced mass scaling

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/{ale,airbag,ams,fluid,multifluid,coupling}` and
their Starter counterparts a literal port. These are the **moving-mesh and
flow** subsystems: they own the grid, the mass conservation, and the largest
single coverage blocker in the corpus after contact (`ALE/BCS`: 12 decks, **7
sole blockers** — the highest sole-blocker ratio of any family).

**Architecture:** five independent subsystems with one shared obligation:
**every one of them must book its work in the EN ledger and conserve mass**.
Phase 5 supplies SPH (46 files) and XFEM (40); this phase supplies the ALE grid
(19 files) and the flow solvers. The port's four existing modules
(`sph_engine.py` 1,681, `ale_engine.py` 2,080, `fsi_coupling.py`,
`airbag.py`) are the entry points and are **kept** — the new modules are called
from inside them.

**Tech stack:** NumPy, `scipy.sparse` for the AMS preconditioner graph,
optional numba for the advection and MUSCL kernels.
**Spec:** `docs/STATE.md` §"Multiphysics Engine" (the 26 multiphysics tests are
the current floor — this phase must not weaken them).
**Entry criteria:** Phases 2, 3, 6, 7, 8 exit gates; Phase 10 Task P10.7
(general controls, which carry `/ALE`, `/EULER`, `/MULTIFLID`, `/MONVOL`).
**Band: L.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `engine/source/ale/` (17 entries + 6 subdirs) | 127 | 30,824 | `engine/ale_engine.py` (2,080), `ale_2d.py`, `ale_bimat.py`, `ale_cut_cells.py`, `ale_fvm.py`, `ale_multimaterial.py`, `ale_porous.py`, `ale_turbulence.py` |
| — `ale/grid` | 19 | 3,992 | — |
| — `ale/ale3d` | 10 | 2,254 | `ale_engine.py` |
| — `ale/bimat` | 11 | 3,245 | `ale_bimat.py` |
| — `ale/alemuscl` | 14 | 2,357 | — |
| — `ale/ale2d` | 13 | 1,914 | `ale_2d.py` |
| — `ale/euler3d` | 2 | 603 | `engine/euler.py` |
| `engine/source/airbag` | 40 | 24,631 | `engine/airbag.py`, `airbag_fvm.py`, `airbag_mesh.py`, `airbag_commu.py`, `airbag_implicit.py` |
| `starter/source/airbag` | 45 | 26,020 | `starter/airbag.py` |
| `engine/source/ams` | 24 | 20,803 | `engine/ams.py`, `mass_scaling.py` (which covers `/DT/NODA` only) |
| `starter/source/ams` | 2 | 2,661 | — |
| `engine/source/fluid` | 11 | 5,555 | `engine/daa.py`, `bem_flow.py` |
| `engine/source/multifluid` | 31 | 9,050 | `engine/multifluid.py` |
| `starter/source/multifluid` | 9 | 1,667 | — |
| `engine/source/coupling` | 4 | 1,032 | `engine/fsi_coupling.py` |
| `starter/source/coupling` | 19 | 8,912 | — |
| `starter/source/ale` | 25 | 3,116 | `input/keywords/ale.py`? |
| `common_source/modules/ale`, `aleanim_mod.F` | — | ~2,000 | — |
| `starter/source/model/remesh` | — | 2,553 | Phase 10 Task P10.2's consumer |
| **Total** | **~350** | **~145,000** | **~12,000** |

## Gap analysis

### 1. `ALE/BCS` — 12 decks, **7 sole blockers**

The single highest-value target in the corpus by ratio. Upstream: the
`/ALE/BCS` card family (`hm_read_ale_close.F` on the Starter) and
`engine/source/ale/alemain.F` + the boundary-condition handling in `ale3d/` and
`grid/`. `docs/STATE.md` records M57 "Parse `/ALE/BCS` to unblock tests" — parse
only. The port has `engine/ale_engine.py` and 7 tests
(`test_ale_engine.py`, `test_ale_extended.py`, …) proving mass conservation
`<1e-12` and energy `<1e-14`.

Those two numbers are **conservation properties on a simple case**, not
parity. A mass-conserving ALE scheme can still be the wrong scheme.

### 2. `MONVOL/AIRBAG1` — 16 decks, 2 sole blockers

`docs/STATE.md` records M51–M53 "AIRBAG1 fluid-structure coupling &
MONVOL/AIRBAG1 starter volume/area calculation". Upstream is substantial:
`engine/source/airbag/airbag1.F`, `airbag2.F`, `airbaga1.F`, `airbagb1.F`,
`fvbag*`, `fvmesh*`, `fvinjt*`, and a Starter side with **45 files** including
`Connectivity.cpp`, `facepoly.F`, `fvelarea.F`, `fvelinte.F`, `fvelsurf.F`,
`fvinject*`, `fvmbag1.F`.

The port has five `engine/airbag*.py` modules and 6 tests proving `p·V = const`
(isothermal), `T·V^(γ-1) = const` (adiabatic), prescribed-curve tracking, and
Saint-Venant choked orifice outflow. Those are **the right invariants** for an
airbag; whether the scheme matches `fvbag1.F` is unmeasured.

### 3. `AMS` — 5 decks, **5 sole blockers**

`engine/source/ams/` has **24 files / 20,803 LOC** — a full AMG-style
preconditioned-CG mass solver (`sms_pcg.F`, `sms_fsa_inv.F`, `sms_proj.F`,
`sms_build_mat_2.F`, `sms_build_diag.F`, `sms_mass_scale_2.F`,
`sms_admesh.F`, `sms_encin_2.F`), plus the constraint-aware variants
(`sms_cjoint.F`, `sms_rbe2.F`, `sms_rbe3.F`, `sms_rgwall.F`, `sms_fixvel.F`,
`sms_gravit.F`) and the boundary-condition family (`sms_bcs*.F`).

The port has `engine/ams.py` + `mass_scaling.py`, and the latter implements
**`/DT/NODA[/CST]` only** — the classic mass-scaling heuristic, not AMS. AMS is
a different algorithm: it solves a constrained optimisation for added mass so
that a target dt is met while minimising added inertia. **5 decks are blocked
because the port does not have it.**

### 4. `SPHCEL`, `SPHGLO` — 4 and 3 decks

`/SPHCEL` (SPH cell-based output/sensor) and `/SPHGLO` (global SPH variables).
Phase 5's SPH state handler is the prerequisite.

### 5. Multifluid

`engine/source/multifluid/` (31 files, 9,050): `multi_allocate.F`,
`multi_bilan.F`, `multi_compute_dt.F`, `multi_ebcs.F`, `multi_inlet_ebcs.F`,
`multi_nrf_ebcs.F`, `multi_fluxes_computation.F`, `multi_fvm2fem.F`,
`multi_muscl_*.F`, `multi_globalize.F`, `multi_evolve_{global,partial}.F`,
`multi_face_data_elem.F`, `multi_face_data_fem.F`, `multi_unplug_neighbors.F`.
The port has `engine/multifluid.py`.

### 6. The remeshing consumer

`starter/source/model/remesh/` (2,553) is Phase 10's; its **engine** consumers
are here (ALE/Euler remesh). Phase 2 Task P2.7 declared Phase 11 as the consumer
of the tensor remap — this phase must actually call it, or that task's code is
dead.

## Wave graph

```
Wave 0  P11.0 multiphysics reconciliation audit (serial — the checklist)
        P11.1 assess the 26 existing multiphysics tests: what do they pin?
   ── gate: the checklist exists; the existing tests are characterised
Wave 1 (parallel — new modules)
  P11.2 the ALE grid (19 files)          P11.3 ALE boundary conditions (ALE/BCS)
  P11.4 ALE 2-D + Euler 3-D              P11.5 ALE bimat (multi-material)
  P11.6 MUSCL advection                  P11.7 ALE porous + turbulence
  P11.8 airbag FV mesh + connectivity    P11.9 airbag injection + vent
  P11.10 AMS: the PCG preconditioner      P11.11 AMS: constraint-aware variants
  P11.12 multifluid cell/flux set        P11.13 FSI coupling (int18/intal1)
  P11.14 SPH cell/global output (/SPHCEL, /SPHGLO)
   ── gate per module: conservation invariant + one oracle case
Wave 2 (serial)
  P11.15 wire the remesher (P2.7's consumer) + coverage re-measure
```

---

### Task P11.0: Multiphysics reconciliation audit

**Fortran:** every directory in §Scope, plus `common_source/modules/ale*`.
**Files:** Create `tools/validation_data/multiphysics_status.json`,
`tests/test_p11_reconciliation.py`.
**Interfaces:**
- Produces: one record per routine with `subsystem`
  (`ale`, `ale_grid`, `ale_muscl`, `ale_bimat`, `ale2d`, `euler3d`, `airbag`,
  `airbag_starter`, `ams`, `fluid`, `multifluid`, `coupling`, `remesh`).

- [ ] **Step 1** — write the failing test:

```python
def test_every_multiphysics_subsystem_is_populated():
    st = json.loads(Path("tools/validation_data/multiphysics_status.json").read_text())
    assert {r["subsystem"] for r in st} >= {
        "ale","ale_grid","ale_muscl","ale_bimat","ale2d","euler3d",
        "airbag","airbag_starter","ams","fluid","multifluid","coupling","remesh"}
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit, with attention to the ALE sub-directory split:
  `ale3d` (2,254) vs `ale2d` (1,914) vs `euler3d` (603) are three distinct
  formulations and the port's single `ale_engine.py` may implement one of them
  well and the others not at all.
- [ ] **Step 4** — run → PASS; commit
  `docs(multiphysics): per-routine audit across all subsystems`.

---

### Task P11.1: Characterise the existing 26 multiphysics tests

**Fortran:** none — this task reads the port's tests against the Fortran's
  invariants.
**Files:** Create `docs/MULTIPHYSICS_TEST_MAP.md`,
`tests/test_p11_test_map.py`.
**Interfaces:**
- Produces: a table mapping each of the 26 tests
  (`test_ale_*`, `test_airbag_*`, `test_ale_coupling.py`,
  `test_ale_fvm.py`, `test_ale_multimaterial.py`) to: the property it pins, the
  upstream file that guarantees that property, and **whether the property is
  actually claimed by that upstream file**.

- [ ] **Step 1** — write the failing test:

```python
def test_every_multiphysics_test_maps_to_an_upstream_file():
    import re, pathlib, yaml
    m = yaml.safe_load(Path("docs/MULTIPHYSICS_TEST_MAP.md").read_text().split("```yaml",1)[1].split("```",1)[0])
    tests = [p.name for p in pathlib.Path("tests").glob("test_ale*.py")] + \
            [p.name for p in pathlib.Path("tests").glob("test_airbag*.py")]
    for t in tests:
        assert any(r["test"] == t for r in m), t
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — do it, honestly. The mapping will show that several tests pin
  a property upstream does *not* claim (e.g. a mass-conservation tolerance that
  is stronger than the upstream scheme guarantees). **Those tests must be
  weakened to upstream's guarantee, with the number quoted** — the alternative is
  keeping a test the port passes for the wrong reason, which is the
  "documented deliberate cut" pattern in reverse.
- [ ] **Step 4** — run → PASS; the weakener commits quote old and new numbers;
  commit `test(multiphysics): map every test to the property upstream claims`.

---

### Task P11.2: The ALE grid

**Fortran:** `engine/source/ale/grid/` (19 files, 3,992 LOC) — `alew5.F`
  (the weighted-median smoothing), `grid/` helpers, the `DONEA` /
  `LAPLACIAN` / `VOLUME` grid operators; `engine/source/ale/alemain.F` is the
  ALE driver the port's `ale_engine.py` mirrors.
**Files:** Create `pyradioss/engine/ale_grid.py`, `tests/test_p11_ale_grid.py`.
**Interfaces:**
- Produces: `ale_grid.Grid` with `.x`, `.v`, `.mass`, `.faces`, `.centroids`,
  `.volume`, `.smooth(h)`, `.donem(v_mesh)`, `.laplacian()`,
  `.volume_operator()`; `ale_grid.Grid.build(nodes, faces) -> Grid`.

- [ ] **Step 1** — write the failing test:

```python
def test_grid_volume_is_positive_and_summed_consistently():
    g = ale_grid.Grid.build(_tetra_mesh())
    assert (g.volume > 0).all()
    assert g.volume.sum() == pytest.approx(_tetra_mesh().volume_sum(), rel=1e-12)

def test_grid_smoothing_conserves_volume():
    g = ale_grid.Grid.build(_tetra_mesh())
    V0 = g.volume.copy()
    g.smooth(h=0.1)
    assert g.volume.sum() == pytest.approx(V0.sum(), rel=1e-10)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **`alew5.F`'s five-point weighted median is the
  algorithm**; a Laplacian-smoothed grid is a different scheme and will produce
  a different ALE field. Read it before writing anything. The
  volume-conservation test is the specification.
- [ ] **Step 4** — run → PASS; commit `feat(ale): the ALE grid and its operators`.

---

### Task P11.3: ALE boundary conditions — the 12-deck blocker

**Fortran:** the `/ALE/BCS` card family; `starter/source/ale/hm_read_ale_close.F`
  (the reader, ported by M57), `starter/source/ale/{alelec.F, alesop.F,
  athlen.F, atheri.F, pornod.F, ale_check_lag.F}`, and the engine's boundary
  application in `engine/source/ale/alemain.F` + `grid/`.
**Files:** Create `pyradioss/engine/ale_bcs.py`, `tests/test_p11_ale_bcs.py`.
**Interfaces:**
- Consumes: Task P11.2's `Grid`.
- Produces: `ale_bcs.apply(grid, model, cycle) -> None` — the ALE grid's own
  boundary conditions (inflow, outflow, symmetry, wall, roller).

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("card,expect", [
    ("/ALE/BCS\nbcs1\ngrnod1\n/vvel1\n", "grid nodes follow the mesh velocity"),
    ("/ALE/BCS\nbcs1\ngrnod1\n/vvel0\n",  "grid nodes are fixed"),
])
def test_ale_bcs_moves_or_fixes_grid_nodes(card, expect):
    ...
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. This is the **highest-value task in the phase by
  corpus ratio (12 decks, 7 sole blockers)**. Two things must be right:
  the `grnod` entity resolution (Phase 10 Task P10.1's `Sets.node_groups`) and
  the velocity source — `/vvel1` follows the **material** velocity, and reading
  it as the grid velocity is the natural mistake.
- [ ] **Step 4** — run → PASS; **the 12 decks' verdicts re-measured** and
  reported individually; commit `feat(ale): ALE boundary conditions`.

---

### Task P11.4: ALE 2-D and Euler 3-D

**Fortran:** `engine/source/ale/ale2d/` (13 files, 1,914),
  `engine/source/ale/euler3d/` (2 files, 603), `engine/source/ale/euler2d/`.
**Files:** Create `pyradioss/engine/ale_euler.py`,
`tests/test_p11_ale_euler.py`.
**Interfaces:**
- Produces: `ale_euler.solve(grid, model, mode="ale"|"euler") -> np.ndarray`
  (Euler = ALE with the material velocity ignored; the grid follows the
  prescribed velocity field only).

- [ ] **Step 1** — write the failing test:

```python
def test_euler_mode_does_not_follow_the_material():
    g, m = _grid_with_material_velocity()
    x_ale = ale_euler.solve(g, m, mode="ale").x.copy()
    x_eul = ale_euler.solve(g, m, mode="euler").x
    assert not np.allclose(x_ale, x_eul)
    assert np.allclose(x_eul, g.x0)      # Euler: the grid stays put
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `euler3d/` being only 2 files / 603 LOC means
  upstream's Euler mode is *thin* — it is a flag in the ALE solver, not a
  separate formulation. Read it before assuming a second solver is needed.
- [ ] **Step 4** — run → PASS; commit `feat(ale): ALE 2-D and the Euler mode`.

---

### Task P11.5: ALE bimat (multi-material)

**Fortran:** `engine/source/ale/bimat/` (11 files, 3,245) — the
  multi-material ALE interface tracking (`arezon.F90`, the bimat cut-cell
  construction).
**Files:** Create `pyradioss/engine/ale_bimat2.py`,
`tests/test_p11_ale_bimat.py`.
**Interfaces:**
- Produces: `bimat.track_interfaces(materials, grid) -> InterfaceTrack`.

- [ ] **Step 1** — write the failing test:

```python
def test_bimat_interface_volume_is_conserved():
    t = bimat.track_interfaces(_two_material_model(), _grid())
    v0 = t.interface_volume.copy()
    for _ in range(10):
        t.advance(dt=1e-5)
    assert t.interface_volume.sum() == pytest.approx(v0.sum(), rtol=1e-10)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `arezon.F90` (`arezo3` per `docs/STATE.md`) is a
  **Reale-Zone explicit interface reconstruction**; the port's `ale_bimat.py`
  exists — the task is depth, and `bimat2.py` is deliberately a *new* module so
  the audit can compare rather than overwrite.
- [ ] **Step 4** — run → PASS; commit `feat(ale): Reale-Zone multi-material interfaces`.

---

### Task P11.6: MUSCL advection

**Fortran:** `engine/source/ale/alemuscl/` (14 files, 2,357) —
  `aMUSCL*.F`, the advection scheme with a compression factor; the
  `/ALE/SOLVER/MUSCL` card (M113) and `docs/STATE.md`'s `ALE/MUSCL` (3 decks).
**Files:** Create `pyradioss/engine/ale_muscl.py`,
`tests/test_p11_ale_muscl.py`.
**Interfaces:**
- Produces: `muscl.fluxes(field, grid, wavespeeds, comp_factor) -> np.ndarray`,
  `muscl.gradients(field, grid) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_muscl_reduces_to_first_order_at_zero_compression():
    f = _smooth_scalar_field()
    hi = muscl.gradients(f, _grid(), scheme="muscl")
    lo = muscl.gradients(f, _grid(), scheme="first_order")
    assert np.abs(hi).sum() < np.abs(lo).sum()

def test_muscl_conserves_total_mass():
    f = _uniform_field(value=1.0)
    fl = muscl.fluxes(f, _grid(), wavespeeds=_ac_speeds(), comp_factor=1.0)
    assert fl.sum() == pytest.approx(0.0, atol=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The compression-factor semantics are the trap**:
  `COEF1`-style factors below 1.0 are *anti-diffusive by design* (that is the
  point of a compressive scheme), so a naive "higher order = more dissipative"
  assumption gives the wrong sign of the error. Read `aMUSCL*.F` for the
  limiter.
- [ ] **Step 4** — run → PASS; the `ALE/MUSCL` decks re-measured; commit
  `feat(ale): MUSCL advection with compression factor`.

---

### Task P11.7: ALE porous media and turbulence

**Fortran:** `engine/source/ale/porous/` (`aleflow*.F`, `alefvm_*` — the
  `sforc3.F` `!||` block shows `sforc3` calls nine `alefvm_*` routines, so the
  porous coupling is wired into the **solid** force path),
  `engine/source/ale/aleturb*.F` (`nsvis_stab11.F` in `solidez` also carries
  `nsvis`); the port has `ale_porous.py`, `ale_turbulence.py`.
**Files:** Create `pyradioss/engine/ale_porous2.py`,
`tests/test_p11_ale_porous.py`.
**Interfaces:**
- Produces: `porous.darcy_source(field, permeability, porosity) -> np.ndarray`
  (wired into the solid kernel per `sforc3.F`'s call graph), and
  `turbulence.smagorinsky(...)`.

- [ ] **Step 1** — write the failing test:

```python
def test_porous_source_opposes_the_flow():
    src = porous.darcy_source(velocity=np.array([1.0, 0.0, 0.0]),
                              permeability=np.array([1e-3, 1e-3, 1e-3]))
    assert src[0] < 0.0     # a Darcy drag opposes the velocity

def test_zero_permeability_means_no_drag():
    src = porous.darcy_source(velocity=np.array([1.0, 0.0, 0.0]),
                              permeability=np.array([0.0, 0.0, 0.0]))
    assert np.allclose(src, 0.0)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **This task wires a source into
  `solid_hexa8.forces`**, so it touches Phase 2's kernel. The rule is the same
  as `prelecflow.F` / `fe_close.F` / `mod_close.F` / `vfluid.F` in `solide/`
  (Phase 2's audit lists them): the source is added to the internal force **and
  booked**. Assert the bookkeeping.
- [ ] **Step 4** — run → PASS; the `PYRADIOSS_BACKEND=numpy` T01 md5 for the
  non-porous examples is unchanged; commit `feat(ale): porous drag and turbulence in the solid path`.

---

### Task P11.8: The airbag finite-volume mesh

**Fortran:** `starter/source/airbag/` (45 files, 26,020) — `fvmesh.F`,
  `fvmesh0.F`, `fvbric*.F` (the brick decomposition), `fvelarea.F`,
  `fvelinte.F`, `fvelsurf.F`, `fvlength.F`, `facepoly.F`, `Connectivity.cpp`,
  `fvmbag1.F`; `engine/source/airbag/fvmesh.F`, `fvmesh0.F`, `fvbric.F`,
  `fvdim.F`, `fvtemp.F`, `fvstats*.F`.
**Files:** Create `pyradioss/starter/airbag_mesh2.py`,
`pyradioss/engine/airbag_fvm2.py`, `tests/test_p11_airbag_mesh.py`.
**Interfaces:**
- Produces: `airbag_mesh2.mesh(part, connectivity, log) -> FVMesh` with
  `.cells`, `.faces`, `.areas`, `.volumes`, `.face_areas`, `.face_centroids`.

- [ ] **Step 1** — write the failing test:

```python
def test_fv_cell_volumes_sum_to_the_bag_volume():
    m = fvm2.mesh(_airbag_part(), _connectivity(), MessageLog())
    assert m.volumes.sum() == pytest.approx(_bag_volume(), rel=1e-10)
    assert (m.volumes > 0).all()

def test_face_areas_are_consistent_with_the_volumes_by_the_divergence_theorem():
    m = fvm2.mesh(_airbag_part(), _connectivity(), MessageLog())
    f = np.ones_like(m.face_centroids)
    assert _divergence_theorem(m, f) == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The divergence-theorem test is the strong one**:
  it checks that the face areas and volumes are mutually consistent, which is
  where an FV mesh goes wrong silently (the run looks fine and conserves the
  wrong amount). `Connectivity.cpp` is C++ and defines the face extraction.
- [ ] **Step 4** — run → PASS; commit `feat(airbag): the finite-volume bag mesh and connectivity`.

---

### Task P11.9: Airbag injection and venting

**Fortran:** `engine/source/airbag/fvinjt6.F`, `fvinjt8.F`,
  `fvinjt8_1.F`, `fvvent0.F`, `fvbag1.F`, `fvbag2.F`, `fvtemp.F`, `fvrezone.F`;
  `starter/source/airbag/fvinject.F`, `fvinjectint.F`, `fvinjnormal.F`.
**Files:** Create `pyradioss/engine/airbag_inject.py`,
`pyradioss/starter/airbag_inject_init.py`, `tests/test_p11_airbag_inject.py`.
**Interfaces:**
- Produces: `inject.mass_flow(t, spec) -> np.ndarray`,
  `inject.vent_flow(p_internal, spec) -> np.ndarray` (Saint-Venant choked
  orifice, per `docs/STATE.md`).

- [ ] **Step 1** — write the failing test:

```python
def test_choked_flow_reaches_sonic_and_then_plateaus():
    p = np.linspace(1e3, 1e9, 500)
    q = inject.vent_flow(p, spec=_vent_spec(critical_ratio=0.528))
    assert np.all(np.diff(q[q > 0]) >= -1e-12)
    assert q[-1] == pytest.approx(q[-2], rel=1e-3)    # plateau at choking
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The critical ratio is gas-dependent**
  (`((k-1)/(k+2))^(k/(k-1))`, ≈0.528 for air) and upstream computes it from the
  EOS gamma — hardcoding 0.528 is wrong for a different gas. Read `fvvent0.F`.
- [ ] **Step 4** — run → PASS; commit `feat(airbag): injection and choked venting`.

---

### Task P11.10: AMS — the preconditioned-CG mass solver

**Fortran:** `engine/source/ams/` (24 files, 20,803 LOC) — `sms_pcg.F` (the CG
  solver), `sms_fsa_inv.F` (the fast-solve inverse), `sms_proj.F` (the
  preconditioner projection), `sms_build_mat_2.F`, `sms_build_diag.F`,
  `sms_mass_scale_2.F`, `sms_init.F`, `sms_admesh.F`, `sms_encin_2.F`,
  `sms_fixvel.F`, `sms_gravit.F`, `sms_auto_dt.F`; Starter `starter/source/ams/`
  (2,661 LOC).
**Files:** Create `pyradioss/engine/ams_pcg.py`, `tests/test_p11_ams.py`.
**Interfaces:**
- Produces: `ams_pcg.solve(model, target_dt) -> np.ndarray` (the added mass per
  node), `ams_pcg.preconditioner(model) -> Callable`,
  `ams_pcg.residual(model, added_mass) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_ams_reaches_the_target_dt_with_the_minimum_added_mass():
    m = _model(n_nodes=500, seed=0)
    added = ams_pcg.solve(m, target_dt=1e-5)
    dt = _recompute_dt(m, added)
    assert dt.min() >= 1e-5 * 0.999
    # and the unconstrained minimum: no node gets more than the constraint needs
    per_node_min = _minimum_added_mass_for_dt(m, 1e-5)
    assert added.sum() <= per_node_min.sum() * 1.05 + 1e-12
```

- [ ] **Step 2** — run → FAIL (the port has `/DT/NODA` only).
- [ ] **Step 3** — implement. **AMS is not mass scaling.** `/DT/NODA` adds mass
  at the critical node; AMS *solves* for an added-mass distribution that meets a
  target dt subject to constraints (fixed DOFs, rigid bodies, walls) while
  minimising added mass. The second assertion is the specification: a solver
  that just adds mass until the dt is met passes the first test and fails the
  second, and is not AMS. **5 corpus decks are blocked on this.**
- [ ] **Step 4** — run → PASS; the added mass, momentum and energy are tracked
  and reported in the ledger (the `/DT/NODA` precedent); commit
  `feat(ams): the preconditioned-CG advanced mass scaling solver`.

---

### Task P11.11: AMS constraint-aware variants

**Fortran:** `sms_cjoint.F`, `sms_rbe2.F`, `sms_rbe3.F`, `sms_rgwall.F`,
  `sms_rgwal0.F`, `sms_rgwalc.F`, `sms_bcs.F`, `sms_bcs1th.F`, `sms_bcscyc.F`.
**Files:** Modify `pyradioss/engine/ams_pcg.py`, `tests/test_p11_ams_constraints.py`.
**Interfaces:**
- Produces: `ams_pcg.constrained_dofs(model) -> np.ndarray` (bool mask),
  `ams_pcg.project_out(preconditioner, mask)`.

- [ ] **Step 1** — write the failing test:

```python
def test_ams_never_adds_mass_to_a_fixed_dof():
    m = _model_with_fixed_dofs()
    added = ams_pcg.solve(m, target_dt=1e-5)
    assert np.allclose(added[m.fixed_nodes], 0.0)

def test_ams_leaves_a_rigid_body_rigid():
    m = _model_with_rigid_body()
    added = ams_pcg.solve(m, target_dt=1e-5)
    assert _rigid_body_stays_rigid(m, added)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Each constraint type is a **projection out of the
  solution space**, and the projection must come after the CG solve and before
  the mass scaling — read `sms_proj.F` for the order. Adding mass that breaks a
  rigid body produces a body that is no longer rigid, which is a silent
  constraint violation.
- [ ] **Step 4** — run → PASS; commit `feat(ams): constraint projection for fixed DOFs, rigid bodies and walls`.

---

### Task P11.12: Multifluid cell and flux set

**Fortran:** `engine/source/multifluid/` (31 files, 9,050) —
  `multi_allocate.F`, `multi_bilan.F`, `multi_compute_dt.F`, `multi_ebcs.F`,
  `multi_inlet_ebcs.F`, `multi_nrf_ebcs.F`, `multi_fluxes_computation.F`,
  `multi_muscl_{compute_pressure,fluxes_computation,gradients}.F`,
  `multi_fvm2fem.F`, `multi_globalize.F`, `multi_evolve_{global,partial}.F`,
  `multi_face_data_{elem,fem}.F`, `multi_unplug_neighbors.F`,
  `multi_computevolume.F`, `multi_buf2var.F`, `multi_ebcs.F`;
  `starter/source/multifluid/` (9 files, 1,667).
**Files:** Create `pyradioss/engine/multifluid_cells.py`,
`pyradioss/engine/multifluid_ebcs.py`, `tests/test_p11_multifluid.py`.
**Interfaces:**
- Produces: `multifluid_cells.allocate(model) -> MultiCells` (one cell per
  material per element), `multifluid_cells.globalize(cells) -> np.ndarray`,
  `multifluid_ebcs.apply_boundary(cells, model, cycle) -> None`.

- [ ] **Step 1** — write the failing test:

```python
def test_each_material_conserves_its_own_mass():
    c = multifluid_cells.allocate(_three_material_model())
    m0 = c.mass.copy()
    for _ in range(20):
        c.advance(dt=1e-6)
    assert np.allclose(c.mass.sum(1), m0.sum(1), rtol=1e-10)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `multi_unplug_neighbors.F` is the subtle one: a
  neighbour pair that stops sharing a face must be disconnected from the flux
  exchange, and forgetting it leaks material across a moving interface.
- [ ] **Step 4** — run → PASS; commit `feat(multifluid): cell allocation, fluxes and Eulerian BCS`.

---

### Task P11.13: Fluid-structure coupling

**Fortran:** `engine/source/coupling/` (4 files, 1,032), `engine/source/fluid/`
  (11 files, 5,555: `flow0.F`, `flow1.F`, `fluxsw.F`, `incpflow.F`,
  `lecflsw.F`, `nintrn.F`, `bemsolv.F`, `daasolv.F`);
  `engine/source/interfaces/int18/` (the FSI interface — Phase 8);
  `docs/STATE.md`: the port's origin is `inter/intal1.F`, `int18/i18for3.F`,
  `ale/inter/iqela2.F`.
**Files:** Create `pyradioss/engine/fsi.py`, `tests/test_p11_fsi.py`.
**Interfaces:**
- Produces: `fsi.exchange_faces(grid, mesh, areas) -> Exchange`,
  `fsi.interface_force(exchange) -> np.ndarray` (booked),
  `fsi.transfer(momentum, interface_force) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_fsi_exchange_is_antisymmetric():
    e = fsi.exchange_faces(_fluid_grid(), _solid_mesh(), _overlap_areas())
    f = fsi.interface_force(e)
    assert np.allclose(f.fluid.sum(0) + f.solid.sum(0), 0.0, atol=1e-10)
    assert e.energy_balance() == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `docs/STATE.md` records the port's current FSI as
  verifying Newton's third law and the `p·dV` balance — **both are the right
  invariants and both are already tested**. This task's job is the *scheme*:
  the coupling coefficient (a relaxation/stabilisation factor) and its default,
  which upstream takes from the interface card. The antisymmetry test is the
  specification; a coupling that leaks momentum passes a "does it run" test.
- [ ] **Step 4** — run → PASS; commit `feat(fsi): the fluid-structure coupling scheme`.

---

### Task P11.14: `/SPHCEL` and `/SPHGLO`

**Fortran:** `engine/source/elements/sph/spgauge.F`, `sppas2t.F`-family cell
  output; the `/SPHCEL` and `/SPHGLO` readers (M106, M113);
  `common_source/modules/aleanim_mod.F`.
**Files:** Create `pyradioss/engine/sph_output.py`,
`tests/test_p11_sph_output.py`.
**Interfaces:**
- Produces: `sph_output.cells(model, cycle) -> np.ndarray` (per-SPH-cell fields
  for `/SPHCEL` and `/TH/SPHCEL`), `sph_output.globals(model, cycle) -> dict`
  (for `/SPHGLO`).

- [ ] **Step 1** — write the failing test:

```python
def test_sphcel_cells_partition_the_particle_volume():
    out = sph_output.cells(_model_with_sph(), cycle=1)
    assert out.volume.sum() == pytest.approx(_mesh_volume(), rel=1e-9)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `spgauge.F` computes the **numerical pressure**
  from the SPH density field, which is not the material pressure — do not
  substitute.
- [ ] **Step 4** — run → PASS; the `SPHCEL` (4 decks) and `SPHGLO` (3 decks)
  blockers re-measured; commit `feat(sph): /SPHCEL cell output and /SPHGLO globals`.

---

### Task P11.15: Wire the remesher, then re-measure coverage

**Fortran:** `starter/source/model/remesh/` (2,553), its engine consumers, and
  the ALE/Euler remesh loop.
**Files:** Modify `pyradioss/engine/ale_engine.py`,
`pyradioss/elements/solid_remesh.py` (Phase 2 Task P2.7's module),
`tools/validate_vs_fortran.py`.
**Interfaces:**
- Produces: the ALE/Euler remesh cycle calls `solid_remesh.remap_state`;
  `tools/validation_data/coverage_p11.json`.

- [ ] **Step 1** — write the failing test:

```python
def test_remesh_actually_calls_the_tensor_remap():
    """Phase 2 Task P2.7 declared this phase the consumer; if nothing calls
    solid_remesh.remap_state, that task's code is dead."""
    calls = []
    monkeypatch.setattr(solid_remesh, "remap_state", lambda *a: calls.append(1))
    _run_an_ale_remesh_case()
    assert calls
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **This is the task that makes Phase 2 Task P2.7
  non-dead.** A reviewer of Phase 2 should have flagged a port-extension with no
  caller; this task closes that loop.
- [ ] **Step 4** — run → PASS; coverage re-measured; commit
  `feat(ale): wire the remesher; record the multiphysics coverage delta`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P11.0** — the ALE sub-directory split (`ale3d`/`ale2d`/`euler3d`) is
  distinguished; one module does not silently cover three formulations.
- **P11.1** — tests that pin a property upstream does not claim are **weakened
  to upstream's guarantee with both numbers quoted**, not deleted and not kept.
- **P11.2** — the smoothing is `alew5.F`'s weighted median, and volume is
  conserved.
- **P11.3** — `/vvel1` follows the **material** velocity; the 12 decks are
  re-measured individually.
- **P11.6** — the limiter's sign convention is read, not assumed.
- **P11.7** — the porous source is added **and booked**; non-porous examples are
  md5-unchanged.
- **P11.8** — the divergence-theorem consistency check is in the test.
- **P11.9** — the critical ratio is gas-dependent, not hardcoded.
- **P11.10** — the **minimum-added-mass** assertion is present; a solver that
  just adds mass until the dt is met is rejected.
- **P11.11** — fixed DOFs get zero added mass and rigid bodies stay rigid.
- **P11.13** — antisymmetry is proven, not asserted.
- **P11.15** — `solid_remesh.remap_state` is genuinely called.

## Parallelisation

- **Wave 0** — P11.0 serial; P11.1 parallel.
- **Wave 1** — 13 tasks, 13 agents, disjoint new modules. **Two are effectively
  serial within the wave**: P11.3 consumes P11.2's `Grid`, and P11.5/P11.7
  consume P11.2 as well. Schedule P11.2 first (it is small), then the three
  consumers in parallel.
- **Wave 2** — serial.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_ale_*.py tests/test_airbag_*.py   # the 26 existing tests, unchanged in count
python -m pytest -q tests/test_p11_reconciliation.py tests/test_p11_ams.py
python tools/validate_vs_fortran.py --mode coverage --out tools/validation_data/coverage_p11.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
```

The phase reviewer reports, as numbers: the corpus coverage delta with
`ALE/BCS` (12/7), `MONVOL/AIRBAG1` (16/2), `AMS` (5/5), `ALE/MUSCL` (3/0),
`SPHCEL` (4/0), `SPHGLO` (3/0) each called out; and that the 26 existing
multiphysics tests are **still green, still 26, and were not rewritten to pass**.