# Phase 5 — Elements: special (SPH, XFEM, IGA, element buffers)

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/elements/{sph,xfem,ige3d,elbuf}` a literal port.
These are the four families whose *state model*, not whose formula, is the hard
part: each keeps per-crack / per-particle / per-control-point history that the
element buffer (`ELBUF_TAB`) must serialise into the restart file.

**Architecture:** a reconciliation audit first (these four families have the
worst port/upstream ratio in the project), then one task per family's missing
capability, then the element-buffer task that makes the restart round-trip work
— which is the cross-cutting deliverable of this phase.

**Tech stack:** NumPy; `scipy.spatial` for the SPH neighbour search; numba for
the SPH density loop if it pays.
**Spec:** `plan/00_ORCHESTRATION.md` §1.1; `docs/STATE.md` §"Multiphysics
Engine" (the 26 multiphysics tests are the current floor).
**Entry criteria:** Phase 2's `init_group`/`forces` contract is final;
Phase 0 oracle.
**Band: L.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `engine/source/elements/sph` | 46 | 18,822 | `engine/sph_engine.py` (1,681), `engine/sph_boundary.py` |
| `engine/source/elements/xfem` | 40 | 11,889 | `elements/xfem_shell.py` (778), `elements/xfem_crack.py` (889), `failure/inicrack.py`, `failure/fractal.py` |
| `engine/source/elements/ige3d` | 15 | 3,347 | `elements/iga3d.py` (694) |
| `engine/source/elements/elbuf` | 6 | 5,762 | `starter/initialization.py` (the element state arrays) |
| `starter/source/elements/{sph,ige3d,xfem}` | ~55 | 14,069 | `input/keywords/elements.py`, `starter/initialization.py` |
| `common_source/modules/{xfem2def_mod.F90,nlocal_reg_mod.F90}` | 2 | ~600 | — |
| **Total** | **164** | **54,489** | **~4,100** |

The **ratio is the finding**: 4,100 Python lines against 54,489 Fortran lines
(1:13), and `sph_engine.py` is a *single* module where upstream has 46 files with
per-routine separation (`spdens.F`, `sppro3.F`, `spdefo3.F`, `spforcp.F`,
`sphreq.F`, `spreploc.F`, `spstab.F`, `spsym.F`, `spmall3.F`, `splissv.F`,
`sph_crit_voxel.F90`, …). The 26 multiphysics tests in `docs/STATE.md` are
honest about scope — they prove conservation properties, not routine-by-routine
fidelity.

## Gap analysis

### SPH — the largest single gap in the element tree

| Upstream routine | What it does | Port |
|---|---|---|
| `spdens.F` | density summation | in `sph_engine.py` |
| `sppro3.F` | stress | in `sph_engine.py` |
| `spdefo3.F` | deformation gradient | in `sph_engine.py` |
| `sphreq.F` | the equation-of-state / material request | partial |
| `spreploc.F` | **replacement / re-spawn of particles** | **gap** |
| `spstab.F`, `spsym.F`, `spsym_alloc.F` | **stability and symmetry of the particle set** | **gap** |
| `spmall3.F` | the small-mass / particle-merge criterion | **gap** |
| `splissv.F` | the **Lissajous / trajectory integrator** (sliding integration) | **gap** |
| `sph_crit_voxel.F90` | the critical-voxel criterion | **gap** |
| `spbuc3.F`, `spmall3.F`, `spclasv.F` | voxel/bucket bookkeeping | partial |
| `spgauge.F` | the SPH numerical pressure gauge | `engine/gauge`-adjacent |
| `spcompl.F`, `spstres.F`, `spreploc.F` | stress states, replication | partial |
| `spn*` (`sponfprs`, `sponfro`, `sponfv`, `sponof1`, `sponof2`) | SPH boundary/interface conditions | `engine/sph_boundary.py` |
| `spadah.F`, `spadasm.F`, `spback3.F`, `spoff3.F` | damage, assembly, **background-grid coupling**, deletion | partial |
| `spst*.F`, `spechan.F` | SPH state persistence and the state handler | **gap** |
| `sphprep.F`, `sph_nodseg.F`, `sphres44b.F`, `sphtri.F`, `sphtri0.F` | prep, node-segment search, results, triangulation | partial |

`spreploc` (particle replacement), `splissv` (the sliding-Lissajous integrator)
and `spmall3` (particle merging) are the three that change results the most and
are the three with no port counterpart.

### XFEM — the enrichment machinery

Upstream has **40 files** in three groups:
- **Crack geometry**: `crk_coord_ini.F`, `crk_velocity.F`, `crk_velocity2.F`,
  `xfem_crk_dir.F`, `crklayer3n_ini/adv`, `crklayer4n_ini/adv`,
  `crklen3n_adv`, `crklen4n_adv`, `crk_tagxp3/4`, `prec rklay.F`, `xfemfsky.F`
- **Enrichment**: `enrichc_ini.F`, `enrichtg_ini.F`, `upenr_crk.F`,
  `upenric1_n3/n4`, `upenric2_n3/n4`, `upenric3_nx`, `upenritg_last.F`,
  `upenric_last.F`, `upxfem1.F`, `upxfem2.F`, `upxfem_tagxp.F`, `activ_xfem.F`,
  `inixfem.F`, `accele_crk.F`
- **Enriched force/coordinate**: `ccoor3_crk.F`, `ccoor3z_crk.F`,
  `cforc3_crk.F`, `czforc3_crk.F`, `c3coor3_crk.F`, `c3forc3_crk.F`,
  `asspar_crk.F`, `xfeoff.F`, `upoffc.F`, `upofftg.F`

The port's `xfem_shell.py` + `xfem_crack.py` (1,667 lines) plus
`failure/inicrack.py` cover the *initiation* side (which `/FAIL/XFEM/*` model
fires) and a basic enriched force. **The enrichment-basis assembly, the crack
front advance, the level-set tagging and the crack-growth history are not
ported.**

### IGA (`ige3d`) — the smallest and the most complete

`elements/iga3d.py` (694) against 15 files / 3,347. Unclaimed:
`onebasisfun.F` (the **tensor-product NURBS basis**, the heart of it),
`dersbasisfuns.F`, `dersonebasisfun.F`, `ig3dderishap.F`,
`ig3donebasis.F`, `ig3donederiv.F`, `ig3daverage.F`, `ig3daire.F`.
A 694-line module that omits `onebasisfun.F` is not an isogeometric kernel; it
is a polynomial NURBS evaluator. **Verify before assuming.**

### Element buffers (`elbuf`) — the cross-cutting piece

`elbuf_ini.F`, `allocbuf_auto.F`, `alloc_elbuf_imp.F`, `copy_elbuf.F`,
`copy_elbuf_1.F`, `w_elbuf_str.F` (5,762 LOC) define how an element's state is
allocated, grown, copied between buffers and written to the restart file. The
port has the state arrays inline in `starter/initialization.py` and pickles the
model. For a literal port this phase delivers `elbuf` as a real abstraction, so
that the SPH/XFEM state handlers of `spechan.F` and the crack history have a
place to live.

## Wave graph

```
Wave 0  P5.0 special-elements reconciliation audit (serial)
   ── gate: every routine in sph/xfem/ige3d/elbuf classified
Wave 1 (parallel — new modules)
  P5.1 SPH particle replacement (spreploc)     P5.2 SPH sliding integrator (splissv)
  P5.3 SPH particle merging (spmall3)          P5.4 SPH stability/symmetry (spstab/spsym)
  P5.5 SPH state handler + bucket bookkeeping  P5.6 XFEM enrichment basis (upenr*/upenric*)
  P5.7 XFEM crack growth + level-set tagging   P5.8 IGA NURBS basis (onebasisfun + ders*)
   ── gate: each has an analytic test; conservation tests stay green
Wave 2 (serial)
  P5.9 elbuf: allocation, growth, copy, restart serialisation
  P5.10 wire into engine.py + starter/initialization.py
  P5.11 numba mirror for the SPH density loop (if it pays)
```

---

### Task P5.0: Reconciliation audit

**Fortran:** `sph/*` (46), `xfem/*` (40), `ige3d/*` (15), `elbuf/*` (6), plus
`starter/source/elements/{sph,ige3d,xfem}`.
**Files:** Create `tools/validation_data/special_routine_status.json`,
`tests/test_p5_reconciliation.py`.
**Interfaces:**
- Produces: the standard record shape, plus a `subfamily` field
  (`geometry`, `stress`, `enrichment`, `crack`, `state`, `bucket`) so Phase 17
  can report per-capability coverage rather than per-file.

- [ ] **Step 1** — write the failing test:

```python
def test_sph_subfamilies_are_all_covered():
    st = json.loads(Path("tools/validation_data/special_routine_status.json").read_text())
    sph = [r for r in st if r["family"] == "sph"]
    assert {r["subfamily"] for r in sph} >= {
        "geometry","stress","enrichment","state","bucket","integrator"}
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit. **Two specific claims to verify or refute, not to
  assume:** (a) `iga3d.py`'s basis is genuinely `onebasisfun.F` and not a
  polynomial substitute; (b) the SPH module's conservation tests
  (`docs/STATE.md`: antisymmetric forces, momentum conservation) are satisfied
  *because* the physics is right and not because the test is weak. Read the
  26 multiphysics tests and state what they do and do not pin.
- [ ] **Step 4** — run → PASS; commit `docs(special): audit of sph/xfem/ige3d/elbuf`.

---

### Task P5.1: SPH particle replacement (`spreploc.F`)

**Fortran:** `engine/source/elements/sph/spreploc.F`, `soltosph.F`,
`soltospha.F`, `solsphp`/`solsph_hour.F` (the hourly-hourglass state).
**Files:** Create `pyradioss/engine/sph_replace.py`,
`tests/test_p5_sph_replace.py`.
**Interfaces:**
- Consumes: `sph_engine.ParticleSet` and its state arrays.
- Produces: `sph_replace.replace(particles, boundary, model) -> np.ndarray`
  (the surviving particle ids), `sph_replace.carry_state(old, new)`,
  `sph_replace.hourglass_state(old, new)`.

- [ ] **Step 1** — write the failing test:

```python
def test_replacement_preserves_total_mass_and_momentum():
    ps = _particle_set(n=1000, seed=0)
    keep = sph_replace.replace(ps, boundary=_open_box(), model=None)
    after = ps.take(keep)
    assert after.mass.sum() == pytest.approx(ps.mass.sum(), rel=1e-12)
    assert (after.mass[:, None] * after.vel).sum(0) == pytest.approx(
        (ps.mass[:, None] * ps.vel).sum(0), rel=1e-12)

def test_replacement_conserves_the_hourglass_mode_mass():
    ps = _particle_set(n=1000, seed=0, hourglass=True)
    keep = sph_replace.replace(ps, boundary=_open_box(), model=None)
    assert np.isclose(sph_replace.hg_mass(ps, keep), sph_replace.hg_mass(ps),
                      rtol=1e-10)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Replacement is a **volume- and momentum-conserving**
  filter: upstream resamples the density field onto the surviving particle set.
  The conservation tests are the specification; a replacement that merely drops
  particles passes neither.
- [ ] **Step 4** — run → PASS; the 26 existing multiphysics tests stay green;
  commit `feat(sph): particle replacement with hourglass-state carry (spreploc)`.

---

### Task P5.2: SPH sliding/Lissajous integrator (`splissv.F`)

**Fortran:** `engine/source/elements/sph/splissv.F`, its caller in
`sphprep.F`/`spreploc.F`.
**Files:** Modify `pyradioss/engine/sph_engine.py`,
`tests/test_p5_sph_lissajous.py`.
**Interfaces:**
- Consumes: the SPH transport kernel (`sotosph`).
- Produces: `sph_engine.transport(stencil, dt, mode="euler"|"lissajous")`.

- [ ] **Step 1** — write the failing test:

```python
def test_lissajous_reduces_to_euler_at_infinite_particles():
    s = _stencil(n=10**6, seed=0)
    assert np.allclose(sph_engine.transport(s, dt=1e-6, mode="lissajous"),
                       sph_engine.transport(s, dt=1e-6, mode="euler"),
                       rtol=1e-4, atol=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The limiting test is the specification: the
  Lissajous sliding integrator converges to the Euler advection of the density
  field as the particle count goes up. A kernel that does not is not the
  upstream integrator.
- [ ] **Step 4** — run → PASS; commit `feat(sph): sliding-Lissajous transport (splissv)`.

---

### Task P5.3: SPH particle merging (`spmall3.F`)

**Fortran:** `engine/source/elements/sph/spmall3.F`, `spbuc3.F`,
`sph_crit_voxel.F90`.
**Files:** Create `pyradioss/engine/sph_merge.py`,
`tests/test_p5_sph_merge.py`.
**Interfaces:**
- Produces: `sph_merge.merge(particles, tol) -> tuple[np.ndarray, np.ndarray]`
  (merged id per particle, group index), `sph_merge.conserved_fields(...)`.

- [ ] **Step 1** — write the failing test:

```python
def test_merge_is_momentum_and_energy_conserving():
    ps = _particle_set(n=5000, seed=1, mass_spread=1e6)
    gid, groups = sph_merge.merge(ps, tol=1e-3)
    merged = sph_merge.conserved_fields(ps, gid, groups)
    assert merged.mass.sum() == pytest.approx(ps.mass.sum(), rel=1e-12)
    assert np.allclose(merged.mass[:, None]*merged.vel, (ps.mass[:, None]*ps.vel).sum(0),
                       rtol=1e-10)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Merging two particles into one changes the
  neighbour count, which changes the SPH density normalisation — a merge that
  conserves mass but not the density kernel will pass the test above and still
  be wrong. **Also verify `sph_crit_voxel.F90`'s criterion is implemented**, or
  record it as `missing` with its citation; do not approximate it.
- [ ] **Step 4** — run → PASS; commit `feat(sph): particle merging with the critical-voxel criterion`.

---

### Task P5.4: SPH stability and symmetry (`spstab.F`, `spsym.F`)

**Fortran:** `engine/source/elements/sph/spstab.F`, `spsym.F`,
`spsym_alloc.F`, `spclasv.F`.
**Files:** Create `pyradioss/engine/sph_stability.py`,
`tests/test_p5_sph_stability.py`.
**Interfaces:**
- Produces: `sph_stability.repartition(particles) -> np.ndarray`,
  `sph_stability.enforce_symmetry(particles, image_particles) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_a_particle_and_its_image_get_identical_densities():
    p = _particle_set(n=400, seed=2)
    img = p.mirror()
    both = sph_stability.enforce_symmetry(np.concatenate([p, img]))
    assert np.allclose(both.dens[:400], both.dens[400:], rtol=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. This is a **symmetry-preserving boundary**
  condition for periodic/image particle sets; getting it wrong shows up as a
  slowly drifting total mass at a symmetry plane, which is why the test asserts
  *equality of density*, not equality of mass.
- [ ] **Step 4** — run → PASS; commit `feat(sph): particle-set stability and symmetry (spstab/spsym)`.

---

### Task P5.5: SPH state handler and bucket bookkeeping

**Fortran:** `engine/source/elements/sph/spechan.F`, `sphprep.F`,
`spbuc3.F`, `spbilan.F`, `spgauge.F`, `sph_nodseg.F`, `sphres44b.F`.
**Files:** Create `pyradioss/engine/sph_state.py`,
`tests/test_p5_sph_state.py`.
**Interfaces:**
- Consumes: the Phase 0 pickle restart path.
- Produces: `sph_state.capture(model) -> dict`, `sph_state.restore(model, blob)`,
  `sph_state.neighbour_stencil(particles, radius) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_sph_state_survives_a_restart_roundtrip():
    ps = _particle_set(n=2000, seed=3)
    blob = sph_state.capture(_model_with(ps))
    m2 = _model_with(_particle_set(n=1))          # a different model entirely
    sph_state.restore(m2, blob)
    assert np.array_equal(m2.sph.particle_ids, ps.ids)
    assert np.array_equal(m2.sph.mass, ps.mass)

def test_neighbour_stencil_is_symmetric_and_complete():
    ps = _particle_set(n=500, seed=4)
    s = sph_state.neighbour_stencil(ps, radius=ps.h)
    assert (s.T[s != -1] == np.flatnonzero(np.arange(len(s))[:, None].repeat(len(s), 1)
            [s != -1])).mean() > 0.99     # neighbour relation is symmetric
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The restart round-trip is the point: Review Focus
  item 4 requires that a mid-run state change resumes bit-identically, and SPH
  particle state is exactly the kind of history that a naive pickle drops.
  The stencil must be **symmetric** — an asymmetric neighbour list is the
  classic SPH force-nonreciprocity bug and the conservation tests will not
  necessarily catch it.
- [ ] **Step 4** — run → PASS; restart-chaining md5 unchanged for the SPH
  examples; commit `feat(sph): state handler, bucket bookkeeping and restart round-trip`.

---

### Task P5.6: XFEM enrichment basis

**Fortran:** `engine/source/elements/xfem/upenr_crk.F`, `upenric1_n3.F`,
`upenric1_n4.F`, `upenric2_n3.F`, `upenric2_n4.F`, `upenric3_nx.F`,
`upenritg_last.F`, `upenric_last.F`, `upxfem1.F`, `upxfem2.F`,
`upxfem_tagxp.F`, `enrichc_ini.F`, `enrichtg_ini.F`, `activ_xfem.F`,
`inixfem.F`, `accele_crk.F`; the module
`common_source/modules/xfem2def_mod.F90`.
**Files:** Create `pyradioss/elements/xfem_enrichment.py`,
`tests/test_p5_xfem_enrichment.py`.
**Interfaces:**
- Consumes: the crack geometry from Task P5.7.
- Produces: `xfem_enrichment.build(elements, cracks, level_sets) -> Enrichment`
  with `Enrichment.enriched_ids`, `Enrichment.weights` (a CSR-like
  `rows/cols/data`), `Enrichment.phantom_nodes`.

- [ ] **Step 1** — write the failing test:

```python
def test_a_crack_crossing_an_element_adds_phantom_nodes_only_there():
    mesh = _unit_quad_mesh(n=4)
    crk = _straight_crack(mesh, from_=(1.0, 0.0), to_=(3.0, 4.0))
    enr = xfem_enrichment.build(mesh, [crk], level_sets=_sign_functions(mesh, [crk]))
    crossed = mesh.element_ids_with_segment_intersection([crk])
    assert set(enr.enriched_ids) == set(crossed)
    # a crack in the element's own plane must not enrich
    flat = _crack_in_element_plane(mesh, crossed[0])
    enr0 = xfem_enrichment.build(mesh, [flat], level_sets=_sign_functions(mesh, [flat]))
    assert not enr0.enriched_ids
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The enrichment weights must be **partition of
  unity** on the element and must sum to zero for the constant mode; a kernel
  that satisfies neither produces a spurious constant strain field — a real
  force that looks like a load. Add the partition-of-unity assertion to the
  test. `upenric1`/`upenric2`/`upenric3_nx` are three different enrichment
  orders; port all three, not just the simplest.
- [ ] **Step 4** — run → PASS; commit `feat(xfem): enrichment basis assembly (upenr/upenric/upxfem family)`.

---

### Task P5.7: XFEM crack growth and level-set tagging

**Fortran:** `crk_coord_ini.F`, `crk_velocity.F`, `crk_velocity2.F`,
`xfem_crk_dir.F`, `crklayer3n_ini/adv.F`, `crklayer4n_ini/adv.F`,
`crklen3n_adv.F`, `crklen4n_adv.F`, `crk_tagxp3.F`, `crk_tagxp4.F`,
`precrklay.F`, `xfemfsky.F`, `upoffc.F`, `upofftg.F`, `xfeoff.F`.
**Files:** Create `pyradioss/elements/xfem_growth.py`,
`tests/test_p5_xfem_growth.py`.
**Interfaces:**
- Consumes: the `/FAIL/XFEM/*` initiation models the port already has.
- Produces: `xfem_growth.advance(crack, dt, rate_model) -> Crack`,
  `xfem_growth.tag(elements, crack) -> np.ndarray` (the level-set side tags),
  `xfem_growth.grow_front(crack, advancing_elements) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_a_crack_advances_only_where_the_criterion_fires():
    crack = _straight_crack(_unit_quad_mesh(n=8), from_=(2., 0.), to_=(6., 8.))
    tip_before = crack.tip.copy()
    xfem_growth.advance(crack, dt=1e-5, rate_model=_zero_rate())
    assert np.allclose(crack.tip, tip_before)

    xfem_growth.advance(crack, dt=1e-5, rate_model=_constant_rate(1e-3))
    assert np.linalg.norm(crack.tip - tip_before) > 0.0

def test_crack_length_is_conserved_when_growing_from_both_ends():
    crack = _two_tip_crack(_unit_quad_mesh(n=8))
    L0 = xfem_growth.length(crack)
    xfem_growth.advance(crack, dt=1e-5, rate_model=_both_ends_rate(1e-3))
    assert xfem_growth.length(crack) >= L0 - 1e-12
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Crack growth consumes **element deletion state**
  (`xfeoff.F`, `upoffc.F`): an element that has failed stops advancing the crack.
  Wiring this to the `/FAIL` deletion plumbing is part of this task, not a
  follow-up, or the crack will grow through failed material.
- [ ] **Step 4** — run → PASS; commit `feat(xfem): crack growth, layer bookkeeping and level-set tagging`.

---

### Task P5.8: IGA NURBS basis (`onebasisfun.F` and friends)

**Fortran:** `engine/source/elements/ige3d/onebasisfun.F` (the tensor-product
NURBS basis), `dersbasisfuns.F`, `dersonebasisfun.F`, `ig3donebasis.F`,
`ig3donederiv.F`, `ig3dderishap.F`, `ig3daverage.F`, `ig3daire.F`,
`ige3ddefo.F`, `ig3dfint.F`, `ig3duforc3.F`, `ig3dcumu3.F`, `ige3dbilan.F`,
`ige3dzero.F`, `projecig3d.F`.
**Files:** Create `pyradioss/elements/iga_basis.py`,
`tests/test_p5_iga_basis.py`.
**Interfaces:**
- Produces: `iga_basis.basis(degree, knots, u) -> np.ndarray` (the univariate
  NURBS basis), `iga_basis.derivatives(degree, knots, u, order) -> np.ndarray`,
  `iga_basis.tensor_product(Bu, Bv, Bw) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_partition_of_unity_and_krondelta_endpoints():
    deg, knots = 2, np.array([0,0,0, 0.5, 1,1,1], float)
    u = np.linspace(0, 1, 11)
    B = iga_basis.basis(deg, knots, u)
    assert np.allclose(B.sum(1), 1.0)
    assert np.allclose(iga_basis.basis(deg, knots, np.array([1.0]))[0],
                       np.array([0., 0., 1.]))

def test_second_derivative_matches_a_central_difference():
    deg, knots = 3, np.array([0]*4 + [0.3, 0.7] + [1]*4, float)
    h = 1e-6
    fd = (iga_basis.derivatives(deg, knots, np.array([0.4]), 1)
          - iga_basis.derivatives(deg, knots, np.array([0.4-2*h]), 0)) / (2*h)
    assert np.allclose(fd, iga_basis.derivatives(deg, knots, np.array([0.4]), 2),
                       atol=1e-5)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The knot vector conventions are the risk:**
  open vs uniform vs unclamped, and whether the basis is defined on the *span*
  or the *whole* knot vector. Read `onebasisfun.F` and `idersbasisfuns.F`
  together; a wrong span convention gives a basis that sums to 1 and is still
  wrong. The finite-difference derivative test catches the second-order term;
  a hand-computed cubic Bernstein case must also be in the test.
- [ ] **Step 4** — run → PASS; `iga3d.py` refactored to consume this module (a
  behaviour-preserving change, quoted by test counts); commit
  `feat(iga): tensor-product NURBS basis and derivatives (onebasisfun family)`.

---

### Task P5.9: `elbuf` — allocation, growth, copy, restart

**Fortran:** `engine/source/elements/elbuf/elbuf_ini.F`, `allocbuf_auto.F`,
`alloc_elbuf_imp.F`, `copy_elbuf.F`, `copy_elbuf_1.F`, `w_elbuf_str.F`;
Starter side `starter/source/elements/elbuf_init/` (7,209 LOC).
**Files:** Create `pyradioss/model/elbuf.py`, `pyradioss/starter/elbuf_init.py`,
`tests/test_p5_elbuf.py`.
**Interfaces:**
- Consumes: every element group's `state` dictionary.
- Produces: `elbuf.ElBuf` with `fields: dict[str, np.ndarray]`,
  `ElBuf.grow(n: int)`, `ElBuf.copy(ids: np.ndarray) -> ElBuf`,
  `ElBuf.write_restart(fh)`, `elbuf.ElBuf.read_restart(fh) -> ElBuf`,
  `elbuf.build_buffers(model) -> dict[str, ElBuf]` — one buffer per group.

- [ ] **Step 1** — write the failing test:

```python
def test_grow_and_copy_preserve_every_field():
    b = elbuf.ElBuf({"off": np.zeros(4, np.int8), "sig": np.zeros((4, 6))})
    b.grow(6)
    assert b.fields["sig"].shape == (6, 6)
    assert np.all(b.fields["sig"][4:] == 0.0)

    ids = np.array([0, 2, 5])
    c = b.copy(ids)
    assert np.array_equal(c.fields["off"], b.fields["off"][ids])

def test_restart_roundtrip_is_byte_identical():
    b = _random_elbuf(n=17, seed=7)
    with io.BytesIO() as fh:
        b.write_restart(fh); fh.seek(0)
        got = elbuf.ElBuf.read_restart(fh)
    for k, v in b.fields.items():
        assert got.fields[k].tobytes() == v.tobytes()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement as the **real abstraction**, then migrate the
  existing per-module inline `state` dictionaries to it. The migration is a
  separate commit per group so a failure bisects. `w_elbuf_str.F` is the
  restart writer and its field order is the on-disk format — copy it exactly,
  or the restart files of an already-running project break.
- [ ] **Step 4** — run → PASS; full suite counts identical; commit per group.

---

### Task P5.10: Wire into the engine and starter

**Fortran:** `engine/source/elements/forint.F` (the group loop),
`engine/source/engine/resol.F`.
**Files:** Modify `pyradioss/engine/engine.py`,
`pyradioss/starter/initialization.py`, `pyradioss/elements/__init__.py`.
**Interfaces:**
- Consumes: Tasks P5.1–P5.9.
- Produces: the `sph`, `xfem`, `iga` groups reachable from a deck; the `elbuf`
  state threaded through the restart.

- [ ] **Step 1** — write the failing test:

```python
def test_sph_deck_runs_end_to_end():
    s, e = make_deck("SPHBOX", _sph_starter_deck(), _sph_engine_deck())
    subprocess.run([sys.executable, "-m", "pyradioss.starter", "-i", s], check=True)
    r = subprocess.run([sys.executable, "-m", "pyradioss.engine", "-i", e],
                       capture_output=True, text=True)
    assert "ENGINE TERMINATION" in r.stdout
    assert "NORMAL" in r.stdout
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement the wiring. **Keep the two SPH engines the port
  already has (`engine/sph_engine.py`, `engine/ale_engine.py`) as the public
  entry points**; the new modules are called from inside them. Do not rename or
  re-export them, or the 26 existing multiphysics tests break for no reason.
- [ ] **Step 4** — run → PASS; commit `feat(special): wire sph/xfem/iga through the engine and starter`.

---

### Task P5.11: numba mirror for the SPH density loop

**Fortran:** none. **Files:** Create `pyradioss/accel/jit_kernels/sph.py`.
**Interfaces:**
- Consumes: `sph_state.neighbour_stencil`.

- [ ] **Step 1** — write the failing test — a timing test **and** a bitwise test:

```python
def test_sph_density_mirror_is_bitwise_and_faster():
    ref = accel.reference("sph_density")
    assert accel.get("sph_density")(*ref.inputs).tobytes() == ref.expected.tobytes()
    assert bench("sph", backend="numba") < bench("sph", backend="numpy")
```

- [ ] **Step 2** — run → SKIP until P0.7, then FAIL.
- [ ] **Step 3** — implement. The SPH density loop is the one place in this phase
  where a numba mirror is likely to pay (it is a tight gather-sum over a
  neighbour list). If the timing does not clear, **exclude it from `auto` and
  say so** — the same rule as Phase 3 Task P3.13.
- [ ] **Step 4** — run → PASS; conservation tests green under both backends;
  commit `perf(sph): numba density mirror (or an auto-rule exclusion)`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P5.0** — the two "verify, do not assume" claims (IGA basis, SPH test
  strength) are resolved with evidence in the audit record.
- **P5.1 / P5.3** — mass, momentum **and** the hourglass/density kernel are
  conserved; a mass-only test is a rejection.
- **P5.2** — the infinite-particle limit reduces to Euler.
- **P5.5** — the stencil is symmetric; restart round-trip is byte-identical.
- **P5.6** — partition of unity is asserted, and the three enrichment orders are
  all present.
- **P5.7** — crack growth is gated on element deletion state.
- **P5.8** — the knot convention is read from the source, and a hand-computed
  cubic Bernstein case is in the test alongside the finite difference.
- **P5.9** — the restart field order matches `w_elbuf_str.F` exactly.
- **P5.10** — the existing multiphysics test modules are unchanged in count.
- **P5.11** — mirror or exclusion; never neither.

## Parallelisation

- **Wave 0** — P5.0 serial.
- **Wave 1** — 8 tasks, 8 agents, disjoint new modules. P5.6 depends on P5.7's
  crack geometry interface, so P5.7 defines the `Crack` dataclass **first** in a
  short preceding task (folded into P5.7's Step 3); P5.6 consumes it read-only.
- **Wave 2** — P5.9 → P5.10 → P5.11, serial (P5.9 is a contended migration).

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p5_reconciliation.py tests/test_p5_sph_state.py tests/test_p5_elbuf.py
python -m pytest -q tests/test_ale_*.py tests/test_airbag_*.py   # the 26 multiphysics tests
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area engine/source/elements/sph
```

The phase reviewer confirms the 26 multiphysics tests still pass **unchanged in
count and identity** (the Phase 5 SPH tasks must not have quietly rewritten them
to pass), and that no `missing` census row in `sph/`, `xfem/`, `ige3d/`,
`elbuf/` lacks a citation.