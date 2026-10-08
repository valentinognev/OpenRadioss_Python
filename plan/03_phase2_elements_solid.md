# Phase 2 — Elements: solids

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/elements/solid/**`, `solid_2d/**` and
`thickshell/**` a literal port — every family present, every routine set
complete, every claim verified against the Fortran.

**Architecture:** OpenRadioss gives every element family a **routine set**
(`Xcoor3`, `Xdefo3`, `Xderi3`, `Xke3`, `Xfint3`, `Xforc3`, `Xbilan`,
`Xcumu3`, `Xcumg3`, `Xlen3`/`Xdlensh`, `Xvis3`, `Xtherm`, `Xrot`,
`Xcumualpha`, `Xzero3`, `Xeoff`). A family is ported when **all** of its
routines exist. This phase is organised as *missing families first, missing
routines second*, with a reconciliation wave first of all.

**Tech stack:** NumPy (vectorised over element groups), optional numba mirrors
in `pyradioss/accel/jit_kernels/`, the Phase 0 oracle for parity.
**Spec:** `plan/00_ORCHESTRATION.md` §1.1 (Fortran wins), §10 item 2
(degenerate geometry).
**Entry criteria:** Phase 1 exit gate; Phase 0 oracle available.
**Band: XL** (≈107,000 Fortran lines of scope).

---

## Scope

| Upstream directory | Files | LOC | Port claim today |
|---|---:|---:|---|
| `engine/source/elements/solid/solide` (hexa8 1-pt) | 101 | 22,441 | `solid_hexa8.py` (1,423), `solid_heph.py` (1,133) |
| `engine/source/elements/solid/solide8` (2×2×2 Gauss, general basis) | 14 | 2,454 | `solid_hexa8_full.py` (578) |
| `engine/source/elements/solid/solide8e` (EAS) | 44 | 9,886 | `solid_hexa8_eas.py` (507) |
| `engine/source/elements/solid/solide8z` (hourglass-assumed strain) | 48 | 13,860 | `solid_hexa8z.py` (439) |
| `engine/source/elements/solid/solidez` (orthotropic solid) | 38 | 7,799 | only `szforc3` (cohesive) |
| `engine/source/elements/solid/solide6z` (wedge) | 20 | 7,321 | `solid_penta6.py`, `solid_penta6_heph.py` |
| `engine/source/elements/solid/solide4` (tetra4) | 42 | 7,274 | `solid_tetra4.py` (859) |
| `engine/source/elements/solid/solide4_sfem` | 12 | 2,786 | `solid_tetra4_sfem.py` (414) |
| `engine/source/elements/solid/solide10` (tetra10) | 50 | 13,529 | `solid_tetra10.py` (722) |
| `engine/source/elements/solid/solide20` (bric20) | 16 | 8,188 | `solid_bric20.py` (733) |
| `engine/source/elements/solid/solide8s` (crimp / imperfection) | 17 | 5,995 | `solid_shell_ha8.py` (HA8 only) |
| `engine/source/elements/solid/sconnect` (spotweld / connector) | 9 | 1,966 | `solid_connect.py`, `solid_cohesive.py` |
| `engine/source/elements/solid/solid_q1np` | 3 | 2,318 | **none** |
| `engine/source/elements/solid_2d/quad`, `quad4` | 46 | 7,450 | `solid_quad.py` (634), `solid_quad4_full.py` (527) |
| `engine/source/elements/thickshell/{solidec,solide6c,solide8c,solide16}` | 74 | 20,540 | `solid_tshell8.py`, `shell_thick16.py`, `thickshell_wedge6.py`, `thickshell_composite.py`, `thickshell_props.py` |
| `starter/source/elements/{reader,elbuf_init,initia,solid}` | ~90 | 21,900 | `input/keywords/elements.py`, `starter/initialization.py` |

**Dispatch map** (`pyradioss/elements/__init__.py`, the authority for what the
Starter can route): `SOLID_ISOLID_GROUPS` covers `Idsolid` 2, 14, 15, 16, 17,
21, 24, 43. `TETRA4_ITETRA4_GROUPS` covers `Itetra4` 3.
`PENTA_ISOLID_GROUPS` covers `Idsolid` 24 for wedges. `QUAD_IQUAD_GROUPS` covers
`Iquad` 2.

## Gap analysis

### Missing families (no port module at all)

1. **`solidez` — the orthotropic solid family (7,799 LOC).** Only the cohesive
   `szforc3.F` is claimed. Unclaimed and substantial:
   `sortho3.F`, `sortho12.F`, `sortho31.F`, `sorth3.F`, `sordef3.F`,
   `sordeft3.F`, `sordeft12.F`, `szderi3.F`, `szsvm.F`, `szsvm_or.F`,
   `szsigpara.F`, `szstraingps.F`, `szstrainhg.F`, `sz_dt1.F90`, `szetfac.F`,
   `c33stif2el.F`, `cbatran3v.F`, `gettransv.F`, `mmodul.F`, `mmodul24c.F`,
   `mmod_norm.F`, `mstiforthv.F`, `mdama24.F`, `m1tot_stab24.F`,
   `visc_et.F`, `scoor_cp2sp.F`, `sdlen8.F`, `sdlen_sms.F`, `gfhour_or.F`,
   `szhour3_or.F`, `nsvis_stab11.F`. This is the **only** 3-D orthotropic
   solid path upstream and the port has none: `/PROP/SOLID` orthotropic
   (`TYPE6`/`/PROP/SOL_ORTH`) solids run the isotropic moduli today.
2. **`solid_q1np` (2,318 LOC).** `q1np_forc3.F90`, `q1np_nurbs_surface_eval_mod.F90`,
   `q1np_dump_hist_state.F90` — hexahedra constrained onto a NURBS surface.
3. **`solide8s` beyond HA8.** `crframe_imp.F`, `crtrans_imp.F`, `transk.F`,
   `getuloc.F`, `s8xref_imp.F`, `srcoor3_imp.F`, `srcoork_imp.F`, `s8sav3_imp.F`,
   `s8sfint3_crimp.F` — the geometrical-imperfection / crimp frame path.
4. **`solid_2d/tria` — the 2-D solid triangle. It does not exist upstream.**
   See the audit finding below.

### Missing routines inside claimed families

`engine/source/elements/solid/solide/` (101 files, 22,441 LOC) claims a large
share through `solid_hexa8.py`, but these have no port counterpart:

| Fortran | LOC group | What it is |
|---|---|---|
| `sfor_n2s4.F`, `sfor_n2stria.F`, `sfor_ns2s4.F90`, `sfor_4n2s4.F90` | ~1,900 | solid→2-node line/segment degenerations (called by `s6for_distor`, `s8for_distor` per their provenance blocks) |
| `upwind.F`, `upwind_v.F`, `schkjab3.F`, `s_hg5.F` | ~2,300 | the volume-upwind strain filter and the 5-node hourglass control |
| `degenes8.F`, `deges4v.F`, `idege.F`, `idege8.F`, `nodedege.F`, `sldege.F`, `tetra4v.F` | ~1,900 | degeneracy detection and element-deletion plumbing |
| `sreploc3.F`, `srepiso3.F`, `srepiso12.F`, `srepisot3.F`, `sreploc3.F` | ~1,400 | remeshing remap (local + isotropic + total Lagrangian) |
| `stherm.F`, `stherm.F`, `s_sav*` | ~700 | solid thermal strain coupling |
| `sboltlaw.F`, `boltst.F`, `preload_solid_ini.F90`, `s11defo3.F`, `s11fx3.F` | ~1,200 | bolt preload inside the solid |
| `scre_sig3.F`, `sortho3.F`, `sgeodel3.F`, `sgparav3.F` | ~1,100 | orthotropic/crane-signal geometry |
| `fe_close.F`, `mod_close.F`, `vfluid.F`, `prelecflow.F` | ~1,200 | porous-media and fluid-element coupling hooks |
| `s8get_x3.F`, `s8jac_i.F`, `s8sav12.F`, `slena.F`, `small*.F`, `sorthdir3.F` | ~1,600 | auxiliary geometry/inertia |
| `sfor_4n2s4.F90`, `sfor_visn8.F`, `sfor_distor.F` | ~800 | distortion visibility + 4-node-to-2-node |

### Audit findings (Wave 0 must resolve these before anything else)

- **A fabricated citation.** `pyradioss/elements/solid_tria3.py` names
  `engine/source/elements/solid_2d/tria/` and files `t3forc2.F`, `t3deri2.F`,
  `t3rota2.F`, `t3dlen2.F`. **That directory does not exist** — upstream
  `solid_2d/` contains only `quad` and `quad4`. The 2-D solid triangle in the
  port is therefore a port invention whose docstring attributes it to Fortran
  that is not there. This is the exact failure the domain rules exist to
  prevent, and it invalidates every formula in that module as "Fortran-faithful".
- **Unverified depth.** `solid_hexa8.py` is 1,423 lines against 22,441 Fortran
  lines across 101 files. `shell_bt4.py` (1,913) and `shell_qbat.py` (2,166)
  have the same shape. **A 1:15 line ratio is not evidence of a port.** The
  reconciliation wave exists to establish, per routine, whether the kernel is
  complete, approximate, or absent.
- `pyradioss/elements/solid_hexa8z.py` cites `solide8z/s8zforc3.F` etc. in
  Windows-path form — legitimate content, fixed by Task P1.2.

## Wave graph

```
Wave 0  P2.0 reconciliation audit (serial, produces the per-routine status table)
        P2.1 resolve the solid_tria3 citation          (independent)
        P2.7 thromb/porter the QA census refresh        (independent)
   ── gate: audit table complete; every solidez/solid_q1np/solide8s routine has a status
Wave 1  (parallel — new files only)
        P2.2 solidez orthotropic family        -> pyradioss/elements/solid_orthotropic.py
        P2.3 solid->2-node degenerations       -> pyradioss/elements/solid_degenerate.py
        P2.4 upwinding filter                  -> pyradioss/elements/solid_upwind.py
        P2.5 5-node hourglass control          -> pyradioss/elements/solid_hourglass5.py
        P2.8 degeneracy detection/deletion     -> pyradioss/elements/solid_degeneracy.py
        P2.9 remeshing remap                   -> pyradioss/elements/solid_remesh.py
        P2.10 solid thermal strain             -> pyradioss/elements/solid_thermal.py
        P2.11 solid_q1np NURBS coupling        -> pyradioss/elements/solid_q1np.py
   ── gate: each new module has an analytic test + one oracle parity case
Wave 2  (SERIAL — shared files)
        P2.11 wire the dispatch + refusal gates   (elements/__init__.py, checks.py, engine.py)
        P2.12 numba mirrors for the Wave-1 kernels
        P2.13 close the audit table's `missing` rows
   ── gate: fast tier + corpus coverage delta + BT/DKT parity spot-check
```

---

### Task P2.0: Reconciliation audit of the claimed solid kernels

**Fortran:** all of `engine/source/elements/solid/solide/`,
`solide8/`, `solide8e/`, `solide8z/`, `solide6z/`, `solide4/`, `solide4_sfem/`,
`solide10/`, `solide20/`, `solid_2d/`, `thickshell/`.
**Files:** Create `tools/validation_data/solid_routine_status.json`,
`tests/test_p2_reconciliation.py`.
**Interfaces:**
- Produces: `solid_routine_status.json` — one record per upstream routine:
  `{"fortran": "engine/source/elements/solid/solide/sforc3.F",
    "routine": "sforc3", "port_module": "pyradioss.elements.solid_hexa8",
    "port_symbol": "forces", "status": "ported|partial|approximate|missing",
    "deviation_note": "..." , "parity_case": "RD-E-1000/c13" | null}`.
  Phase 12 and Phase 17 read this file.

- [ ] **Step 1** — write the failing test:

```python
def test_every_upstream_solid_routine_has_a_status():
    st = json.loads(Path("tools/validation_data/solid_routine_status.json").read_text())
    census = json.loads(Path("tools/validation_data/census.json").read_text())
    missing = [f for f in census["files"]
               if f.startswith("engine/source/elements/solid/")
               and f not in {r["fortran"] for r in st}]
    assert missing == [], f"{len(missing)} unclassified, e.g. {missing[:5]}"
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — do the audit. **This is reading, not coding.** For each
  routine, open both files and record one of four statuses. The rule that
  decides `approximate`: the port computes the same quantity by a *different*
  formulation (the port's own Ogden tangent, its own orthotropic stress, its
  own generalised strain), not merely a shorter derivation of the same formula.
  Every `approximate` needs a `deviation_note` that names the difference and
  whether it has a measured parity consequence. **Do not fix anything here.**
- [ ] **Step 4** — run → PASS; the test additionally asserts that every
  `missing` row is referenced by some Wave-1/Wave-2 task id in this phase file;
  commit `docs(solids): per-routine reconciliation audit of the claimed kernels`.

---

### Task P2.1: Resolve the `solid_tria3` fabricated citation

**Fortran:** none — `$OR_SRC/engine/source/elements/solid_2d/` contains only
`quad/` and `quad4/`, verified 2026-10-02.
**Files:** Modify `pyradioss/elements/solid_tria3.py`; Create
`tests/test_p2_tria3_provenance.py`, `docs/PORT_EXTENSIONS.md`.
**Interfaces:**
- Consumes: nothing.
- Produces: `docs/PORT_EXTENSIONS.md` — the single registry of code with **no**
  upstream counterpart, with a justification and a test that keeps it honest.

- [ ] **Step 1** — write the failing test:

```python
def test_every_cited_fortran_file_exists():
    import re, pathlib
    bad = []
    for p in pathlib.Path("pyradioss/elements").glob("*.py"):
        for m in re.finditer(r"elements/[a-z0-9_]+/[a-z0-9_0-9]+\.F", p.read_text()):
            rel = "engine/source/elements/" + m.group(0)
            if not (paths.or_src() / rel).is_file():
                bad.append((p.name, rel))
    assert bad == [], bad
```

- [ ] **Step 2** — run → FAIL, naming `solid_tria3.py` and `solid_2d/tria/*.F`.
- [ ] **Step 3** — decide, in this order, and record the decision in
  `docs/PORT_EXTENSIONS.md`:
  1. **The 2-D solid triangle may exist in a closed-source Radioss tree.** If so,
     the honest citation is `Simcenter Radioss (closed source)`, the module
     becomes a declared port extension, and its docstring says so instead of
     naming a file that is not there.
  2. If it is genuinely absent upstream, it is a **port extension** and is
     removed from any claim of being a port.
  Either way, **`t3forc2.F` and friends stop appearing as citations.** A
     reviewer must not accept "leave it, upstream might have it".
- [ ] **Step 4** — run → PASS; run the generalise the test to the whole
  `pyradioss/` tree and commit whatever additional honest failures it finds (each
  as its own follow-up commit); commit
  `fix(elements): replace the nonexistent solid_2d/tria citation with a declared port extension`.

---

### Task P2.2: The `solidez` orthotropic solid family

**Fortran:**
- Kinematics/strain: `sordef3.F`, `sordeft3.F`, `sordeft12.F`, `szderi3.F`,
  `szstraingps.F`, `szstrainhg.F`, `sz_dt1.F90`
- Material moduli: `mmodul.F`, `mmodul24c.F`, `mmod_norm.F`, `mstiforthv.F`,
  `mdama24.F`, `c33stif2el.F`, `m1tot_stab24.F`, `cbatran3v.F`, `gettransv.F`
- Stress/measurement: `szforc3.F`, `szsvm.F`, `szsvm_or.F`, `szsigpara.F`,
  `sztorth3.F`, `szhour3.F`, `szhour3_or.F`, `szhour_ctl.F`, `shour_ctl.F90`,
  `gfhour_or.F`, `nsvis_stab11.F`, `visc_et.F`
- Geometry/length: `scoor_cp2sp.F`, `sdlen8.F`, `sdlen_sms.F`, `szetfac.F`
**Files:** Create `pyradioss/elements/solid_orthotropic.py`,
`tests/test_p2_solid_orthotropic.py`.
**Interfaces:**
- Consumes: `pyradioss.elements.KERNELS` contract —
  `init_group(group, model, log) -> np.ndarray` (nodal mass+inertia),
  `forces(group, x, v, vr, dt, fint, mint) -> np.ndarray` (per-element dt).
- Produces: `solid_orthotropic.init_group`, `solid_orthotropic.forces`,
  `solid_orthotropic.dt_claim`, `solid_orthotropic.tangent`,
  `solid_orthotropic.kgeo`; registered as the group `"solids_ortho"`.

- [ ] **Step 1** — write the failing test:

```python
def test_orthotropic_solid_matches_the_isotropic_limit():
    """With C11=C22=C33=E/(1-2nu) and all 12 G=0 the orthotropic kernel must
    reproduce solid_hexa8.forces bit-for-bit on the same state."""
    import numpy as np
    from pyradioss.elements import solid_orthotropic, solid_hexa8
    g = _unit_hexa8_group(orthotropic=None)
    ref = solid_hexa8.forces(g, *g._cycle_inputs(dt=1e-3))
    got = solid_orthotropic.forces(g, *g._cycle_inputs(dt=1e-3))
    assert np.array_equal(ref["fint"], got["fint"])

def test_orthotropic_uniaxial_stiffness_is_C11():
    g = _unit_hexa8_group(C11=2.0e11, nu12=0.3, nu13=0.3, nu23=0.3, G=0.0)
    e = _uniaxial_strain(g, eps=1e-3)
    got = solid_orthotropic.stress_from_strain(g, e)
    assert np.allclose(got[0], 2.0e11 * 1e-3, rtol=1e-6)   # sigma_11
    assert np.allclose(got[1:], 0.0, atol=1e-3)            # lateral free
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Read `mmodul.F` first**: the compliance
  construction and its singular cases decide everything downstream, and
  `mmodul24c.F` is the Isolid=24 (HEPH) variant with different defaults. The
  `scoor_cp2sp.F` call is the *strain-rate-to-cotangent* transform; getting its
  transpose/rotation order wrong produces a kernel that passes an isotropic
  test and is wrong for every real orthotropic part. `tangent`/`kgeo` are
  required because Phase 15 needs them for the frozen implicit tower; if the
  consistent tangent is genuinely out of reach, raise through the existing
  `checks.py` refusal mechanism with the upstream file cited — **never** return
  a zero tangent.
- [ ] **Step 4** — run → PASS; parity on the RD-E-2100 orthotropic deck and one
  hand-computed single-element uniaxial case; commit
  `feat(solids): solidez orthotropic solid family (mmodul/sortho/szforc3)`.

---

### Task P2.3: Solid → 2-node degenerations

**Fortran:** `sfor_n2s4.F`, `sfor_n2stria.F`, `sfor_ns2s4.F90`,
`sfor_4n2s4.F90` (per their `!||` blocks: `sfor_n2s4` is called by
`s6for_distor` and `s8for_distor` and calls `ssort_n4`).
**Files:** Create `pyradioss/elements/solid_degenerate.py`,
`tests/test_p2_solid_degenerate.py`.
**Interfaces:**
- Produces: `solid_degenerate.six_to_two(state, nodes) -> (ids, weights)`,
  `solid_degenerate.eight_to_two(...)`, `solid_degenerate.four_to_two(...)`,
  `solid_degenerate.four_to_four_striking(...)`.

- [ ] **Step 1** — write the failing test:

```python
def test_hexa_to_line_preserves_mass_and_first_moment():
    """A hexa squeezed onto a line must reduce to a 2-node segment with the
    same total mass and the same centroid (sfor_ns2s4.F)."""
    ids, w, m, com = solid_degenerate.hexa_to_two(_collapsed_hexa8())
    assert m == pytest.approx(1.0, rel=1e-12)
    assert np.allclose(com, _collapsed_hexa8().centroid(), atol=1e-12)
    assert ids.shape == (2,)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The `ssort_n4` call inside `sfor_n2s4` is a
  **node re-ordering** (sorting which pair survives the degeneration), and
  getting it wrong gives a segment with swapped ends — an infinite energy
  engine. Transcribe the sort, do not re-derive it.
- [ ] **Step 4** — run → PASS; parity on a deck that collapses a solid onto a
  line; commit `feat(solids): solid->2-node degeneration paths (sfor_n2s4 family)`.

---

### Task P2.4: The volume-upwind strain filter

**Fortran:** `upwind.F`, `upwind_v.F`, `schkjab3.F` (the strain-rate filter and
its switching predicate), the `UF_INT`/`IFUPM` flags read by the Starter.
**Files:** Create `pyradioss/elements/solid_upwind.py`,
`tests/test_p2_solid_upwind.py`.
**Interfaces:**
- Produces: `solid_upwind.filter_rate(group, dstra, dt) -> np.ndarray`,
  `solid_upwind.should_upwind(group, dstra, dt) -> np.ndarray` (bool mask).

- [ ] **Step 1** — write the failing test:

```python
def test_upwind_is_a_no_op_on_a_smooth_strain_field():
    g = _unit_hexa8_group()
    d = np.tile(np.array([1e-3, 0.0, 0.0]), (8, 1))     # uniform tension
    assert np.array_equal(solid_upwind.filter_rate(g, d, 1e-6), d)

def test_upwind_activates_on_a_volumetric_spike():
    g = _unit_hexa8_group()
    d = np.zeros((8, 6)); d[0, :3] = 1e6                 # one node explodes
    out = solid_upwind.filter_rate(g, d, 1e-6)
    assert np.abs(out).max() < np.abs(d).max()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Read `upwind.F` for the `IFUPM` ladder before
  writing anything:** upstream selects among several filtering strength levels
  and the default is *off*; a port that always filters changes every
  hourglass-on deck's results. Gate the default on the property read.
- [ ] **Step 4** — run → PASS; commit `feat(solids): volume-upwind strain filter`.

---

### Task P2.5: The 5-node hourglass control

**Fortran:** `s_hg5.F`, `shvis3.F` (the existing viscous counterpart the port
does implement), `szhour_ctl.F90`/`shour_ctl.F90` for the coefficient ladder.
**Files:** Create `pyradioss/elements/solid_hourglass5.py`,
`tests/test_p2_hourglass5.py`.
**Interfaces:**
- Consumes: the hourglass mode matrix `A1`/`Q` already built by
  `solid_hexa8` (do **not** recompute it — read its provenance first).
- Produces: `solid_hourglass5.hg5_forces(group, hg_mode, rot) -> np.ndarray`
  (adds into `fint`), `solid_hourglass5.hg5_energy(...) -> float` booked to
  `EVIS(8)`/the hourglass ledger.

- [ ] **Step 1** — write the failing test:

```python
def test_hg5_dissipates_a_zero_energy_mode():
    g = _unit_hexa8_group(hourglass="5")
    f = solid_hourglass5.hg5_forces(g, hg_mode="zx", rot=np.eye(3))
    assert np.abs(f).sum() > 0.0

def test_hg5_energy_lands_in_the_hourglass_ledger():
    g = _unit_hexa8_group(hourglass="5")
    e = solid_hourglass5.hg5_energy(g)
    assert e >= 0.0
    assert g.state["evis"][8] == pytest.approx(e, rel=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The energy test is the point.** The M36 finding
  was that the port's hourglass dissipation was far above the reference
  (box_beam HE 0.584 → 0.094 after the `chvis3.F` fix). A new hourglass mode
  that does not book its work into the ledger will pass a force test and leak
  energy silently — the energy-balance tests are the detector.
- [ ] **Step 4** — run → PASS; energy-balance test green; commit
  `feat(solids): 5-mode hourglass control with ledger booking`.

---

### Task P2.6: Degeneracy detection and element deletion

**Fortran:** `degenes8.F`, `deges4v.F`, `idege.F`, `idege8.F`, `nodedege.F`,
`sldege.F`, `tetra4v.F`, `dim_tshedg.F`, `ind_tshedg.F`, `tshcdcom_ini.F`,
`tshcdcom_dim.F`.
**Files:** Create `pyradioss/elements/solid_degeneracy.py`,
`tests/test_p2_solid_degeneracy.py`.
**Interfaces:**
- Produces: `solid_degeneracy.detect(group) -> np.ndarray` (bool per element),
  `solid_degeneracy.degeneracy_factors(group) -> tuple[np.ndarray, np.ndarray]`,
  `solid_degeneracy.assemble_isotropic(mesh, ids) -> np.ndarray` (the
  `dim_tshedg`/`ind_tshedg` mesh used to find the shortest path between the
  degenerate nodes).

- [ ] **Step 1** — write the failing test — this is Review Focus item 2:

```python
@pytest.mark.parametrize("coords", [
    [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,1,0],[0,1,0],[0,0,1]],  # repeated nodes
    [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,0,0],[0,0,1]],  # a fully flat face
])
def test_degenerate_solid_gets_a_defined_dt_and_finite_forces(coords):
    g = _hexa8_group(coords=coords)
    degen = solid_degeneracy.detect(g)
    assert degen.any()
    dt = g.forces(dt=1e-3)
    assert np.all(np.isfinite(dt)) and np.all(dt > 0.0)
    assert np.all(np.isfinite(g.state["fint"]))
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Note the port already accepts a 5/6/7-distinct-node
  brick as a "COLLAPSED HEXA" (`starter/initialization.py`, per
  `PORTING_GUIDE.md:46`), so the *Starter* half exists and the *Engine* half
  (`degenes8`, the zero-energy-mode detection, the assembly remesh) does not.
  Those two halves must agree; a Starter that marks an element collapsed while
  the Engine never sees the flag is a silent inconsistency.
- [ ] **Step 4** — run → PASS; commit
  `feat(solids): degeneracy detection, deletion and the dim/ind assembly remesh`.

---

### Task P2.7: Remeshing remap

**Fortran:** `sreploc3.F` (local), `srepiso3.F` (isotropic), `srepiso12.F`
(12-node isotropic), `srepisot3.F` (isotropic with rotation).
**Files:** Create `pyradioss/elements/solid_remesh.py`,
`tests/test_p2_solid_remesh.py`.
**Interfaces:**
- Produces: `solid_remesh.remap_state(old, new, nodes_old, nodes_new) -> dict`,
  `solid_remesh.remap_isotropic(tensor, rotation) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_isotropic_remap_is_rotation_covariant():
    th = 0.7
    R = _rot_x(th)
    sig = np.diag([1e8, 2e8, 3e8])
    got = solid_remesh.remap_isotropic(sig, R)
    assert np.allclose(got, R @ sig @ R.T, atol=1e-3 * np.abs(sig).max())
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Remeshing remap only matters for the adaptive
  mesh path; Phase 11 reaches the ALE/Euler remeshers. **This task ports the
  tensor transforms and the state remap, and gates the caller** — a remap with
  no caller is dead code and the reviewer should reject it unless Phase 11 is
  declared as its consumer in this file. *(That consumer declaration is
  mandatory here, not optional.)*
- [ ] **Step 4** — run → PASS; commit
  `feat(solids): remeshing state remap (sreploc3/srepiso3 family)`.

---

### Task P2.8: Solid thermal strain

**Fortran:** `stherm.F`, `s4therm.F`, `s4therm-itet1.F`, `stherm` in the
thickshell families (`sctherm.F`, `s6ctherm.F`).
**Files:** Create `pyradioss/elements/solid_thermal.py`,
`tests/test_p2_solid_thermal.py`.
**Interfaces:**
- Consumes: `materials.thermal_expansion(law) -> np.ndarray` (α per material),
  which Phase 6 produces; until then the module takes α as an argument.
- Produces: `solid_thermal.thermal_strain(group, dT, alpha) -> np.ndarray`,
  `solid_thermal.energy(...) -> float` (booked to the thermal ledger).

- [ ] **Step 1** — write the failing test:

```python
def test_isotropic_heating_produces_free_expansion():
    g = _unit_hexa8_group()
    d = solid_thermal.thermal_strain(g, dT=100.0, alpha=np.array([12e-6]*3))
    assert np.allclose(d, 12e-6 * 100.0)

def test_constrained_heating_produces_stress():
    g = _unit_hexa8_group()
    locked = g.copy(); locked.nodes[:] = 0.0      # all DOFs fixed
    f = solid_thermal.forces(locked, dT=100.0, alpha=np.array([12e-6]*3), dt=1e-3)
    assert np.abs(f).sum() > 0.0
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Read `stherm.F` for the **order**: upstream applies
  the thermal strain *before* the constitutive update and subtracts it from the
  strain rate, so the energy booked differs between the two orders by
  `σ·α·ΔT`. The second test is what pins it.
- [ ] **Step 4** — run → PASS; commit `feat(solids): thermal strain coupling (stherm)`.

---

### Task P2.9: `solid_q1np` — hexahedra on a NURBS surface

**Fortran:** `q1np_forc3.F90`, `q1np_nurbs_surface_eval_mod.F90`,
`q1np_dump_hist_state.F90`; Starter side
`starter/source/model/remesh/` and `common_source/modules/q1np_geom_mod.F90`,
`q1np_restart_mod.F90`.
**Files:** Create `pyradioss/elements/solid_q1np.py`,
`tests/test_p2_solid_q1np.py`.
**Interfaces:**
- Consumes: a NURBS surface description, which `$OR_SRC` gets from an external
  CAD bridge. **There is no deck syntax for it in the open source tree.**
- Produces: `solid_q1np.evaluate_surface(ctrl, knots, degree, u, v) -> np.ndarray`
  (position + first derivatives), `solid_q1np.forces(...)` projecting the
  element nodes onto the surface.

- [ ] **Step 1** — write the failing test:

```python
def test_bilinear_nurbs_surface_evaluates_and_differentiates():
    ctrl = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], float)
    knots = np.array([0,0,1,1], float)
    P = solid_q1np.evaluate_surface(ctrl, knots, degree=(1,1), u=0.25, v=0.5)
    assert np.allclose(P.x, 0.25)
    d = solid_q1np.evaluate_surface(ctrl, knots, degree=(1,1), u=0.25, v=0.5, derivs=(1,0))
    assert np.allclose(d.dx_du, 1.0) and np.allclose(d.dy_du, 0.0)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement the surface evaluator and its derivatives, and
  **stop there**. The projection path (`q1np_forc3.F90`) cannot be exercised
  without a deck that specifies a surface, and no such deck exists in the open
  source. The honest delivery is: evaluator ported and tested, projection ported
  and **marked unreferenced with the reason recorded** in
  `docs/PORT_EXTENSIONS.md`, and `census.json` marking `q1np_forc3.F90` as
  `ported-unreachable` rather than `ported`. Claiming a reachability the deck
  format cannot deliver is the failure mode this task is designed to avoid.
- [ ] **Step 4** — run → PASS; commit
  `feat(solids): Q1NP NURBS surface evaluator; projection marked unreachable`.

---

### Task P2.10: Crimp / geometrical imperfection (`solide8s`)

**Fortran:** `crframe_imp.F`, `crtrans_imp.F`, `transk.F`, `getuloc.F`,
`s8xref_imp.F`, `srcoor3_imp.F`, `srcoork_imp.F`, `s8sav3_imp.F`,
`s8sfint3_crimp.F`, `s8sk*`/`s8slke3.F` (the imperfection stiffness and mass
operators), `crframe`/`crtrans` starter readers.
**Files:** Create `pyradioss/elements/solid_crimp.py`,
`tests/test_p2_solid_crimp.py`, and a keyword reader in
`pyradioss/input/keywords/elements.py`.
**Interfaces:**
- Produces: `solid_crimp.imperfection_frame(group, model) -> np.ndarray`
  (the 3×3 local frame per element), `solid_crimp.reference_state(...)`.

- [ ] **Step 1** — write the failing test:

```python
def test_crimp_frame_is_a_orthonormal_right_handed_basis():
    R = solid_crimp.imperfection_frame(_hexa8_group(), reference="undeformed")
    for e in range(R.shape[0]):
        assert np.allclose(R[e] @ R[e].T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R[e]) == pytest.approx(1.0, abs=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Read `crtrans_imp.F` and `transk.F` together** —
  the pair is a two-stage transform and porting only one produces a frame that
  is orthogonal but wrong. The keyword reader is in this task because the
  property card and the kernel are one deliverable: a kernel no deck can reach
  is not a port.
- [ ] **Step 4** — run → PASS; commit `feat(solids): crimp/imperfection frame (solide8s)`.

---

### Task P2.11: Wire the dispatch, refusal gates and reader

**Fortran:** `starter/source/elements/reader/hm_read_solid.F` (the `Idsolid`
card), `$OR_SRC/engine/source/elements/forint.F` (the group loop the port's
`engine.py` mirrors).
**Files:** Modify `pyradioss/elements/__init__.py`,
`pyradioss/input/keywords/elements.py`, `pyradioss/input/checks.py`,
`pyradioss/engine/engine.py` (one dispatch line per new group).
**Interfaces:**
- Consumes: every `init_group`/`forces` from Tasks P2.2–P2.10.
- Produces: new `SOLID_ISOLID_GROUPS` entries and `KERNELS` keys; every
  `Idsolid` value the Fortran accepts is either routed or **refused with the
  upstream file cited**.

- [ ] **Step 1** — write the failing test — exhaustiveness, not a sample:

```python
def test_every_idsolid_value_is_routed_or_refused():
    from pyradioss.elements import SOLID_ISOLID_GROUPS
    from pyradioss.input.checks import unsupported_isolid
    for idsolid in range(0, 50):
        if idsolid in SOLID_ISOLID_GROUPS:
            assert KERNELS[SOLID_ISOLID_GROUPS[idsolid]]
        else:
            assert idsolid in unsupported_isolid()   # cited refusal, not silence
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The test is the deliverable's real content: it
  forces the port to be **exhaustive** over `Idsolid`, matching
  `hm_read_solid.F`'s card set. Anything not routed must appear in
  `checks.py::unsupported_isolid()` with its upstream citation — a silent skip
  is what made 440 corpus decks "SKIPS" instead of "CLEAN".
- [ ] **Step 4** — run → PASS; corpus coverage delta measured
  (`tools/validate_vs_fortran.py --mode coverage`); commit
  `feat(solids): route and refuse every Isolid value exhaustively`.

---

### Task P2.12: numba mirrors for the new solid kernels

**Fortran:** none (port-side performance only).
**Files:** Modify `pyradioss/accel/jit_kernels/`, `pyradioss/accel/__init__.py`.
**Interfaces:**
- Consumes: the Wave-1 kernels.
- Produces: numba mirrors dispatched through `accel.get(name)` so the NumPy
  default install is untouched; each mirror is **bitwise** equal to its NumPy
  twin.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("name", ["hexa_hg5", "solid_upwind_filter", "ortho_moduli"])
def test_numba_mirror_is_bitwise_identical(name):
    import numpy as np, os
    os.environ["PYRADIOSS_BACKEND"] = "numba"
    import pyradioss.accel as accel; accel.reload()
    ref = accel.reference(name)
    got = accel.get(name)(*ref.inputs)
    assert got.tobytes() == ref.expected.tobytes()
```

- [ ] **Step 2** — run → SKIP until P0.7; then FAIL if the mirror is absent.
- [ ] **Step 3** — implement **one kernel per commit**, mirroring the
  reference expression **statement by statement** so the bitwise property holds;
  a reassociation that is faster but not bitwise is rejected. Follow the
  existing pattern in `jit_kernels` (see the `scatter3` and `hexa_hgphys`
  entries, `PORTING_GUIDE.md:99-100`).
- [ ] **Step 4** — run → PASS under both backends; the T01 md5 must be
  unchanged; commit `perf(accel): numba mirrors for the new solid kernels`.

---

### Task P2.13: Close the reconciliation audit

**Fortran:** every `missing` row of `tools/validation_data/solid_routine_status.json`.

The P2.0 audit found that no earlier task in this phase dispatches the
remaining `missing` rows. **P2.13 owns all of them**, and the families to
name per row are: `solide`, `solide8`, `solide8e`, `solide8z`, `solide6z`,
`solide4`, `solide4_sfem`, `solide10`, `solide20`, `sconnect`, and the
standalone `srotorth.F`. (`solidez`, `solid_q1np` and `solide8s` are *not*
listed here: P2.2, P2.9 and P2.10 own those families, and P2.3 … P2.8 own the
individual files they name.) Each of those 213 rows becomes `ported`,
`approximate` (with a deviation note) or `refused` here — see
`tools/validation_data/solid_routine_status.json` and Step 3 for what
"worked" means per row.
**Files:** Modify `tools/validation_data/solid_routine_status.json`,
  `pyradioss/input/checks.py`.
**Interfaces:**
- Consumes: the P2.0 table and every Wave-1/Wave-2 kernel.
- Produces: a table with **zero `missing` rows**; each becomes `ported`,
  `approximate` (with a deviation note) or `refused` (with the upstream file
  that lacks the routine).

- [ ] **Step 1** — write the failing test:

```python
def test_no_solid_routine_is_left_unclassified():
    st = json.loads(Path("tools/validation_data/solid_routine_status.json").read_text())
    left = [r["fortran"] for r in st if r["status"] == "missing"]
    assert left == [], left
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — work the list. **A row may legitimately end `refused`**, and a
  refusal must name the upstream file or the fact that the routine does not
  exist (`degenes8`'s successors, the 2-D triangle, the CUDA kernels). What is
  not acceptable is leaving a row open. For each `approximate` row, record the
  measured parity consequence — if none was measured, say so and open a
  follow-up task id in the row.
- [ ] **Step 4** — run → PASS; commit
  `docs(solids): close the reconciliation audit`.

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P2.0** — the status table covers **every** routine in scope; `approximate`
  rows carry a deviation note with a measured consequence.
- **P2.1** — no citation in `pyradioss/` names a file that is not in `$OR_SRC`;
  the reparented test covers the whole package.
- **P2.2** — the isotropic-limit test passes *bitwise*, not just within
  tolerance. An orthotropic kernel that matches only approximately in the
  isotropic limit has a bug in the rotation or the compliance assembly.
- **P2.3** — the node sort from `ssort_n4` is transcribed, not re-derived.
- **P2.4** — the default is off; a deck that does not request upwinding is
  byte-identical before and after.
- **P2.5** — the energy lands in the hourglass ledger; the energy-balance test
  is quoted in the commit body.
- **P2.6** — degenerate geometry yields a finite, positive dt and finite forces
  (Review Focus item 2), never a NaN and never a pass-through.
- **P2.7** — the consumer (Phase 11) is named, or the task is rejected as dead code.
- **P2.9** — reachability is honestly recorded; no `ported` claim for code no
  deck can reach.
- **P2.11** — `Idsolid` is exhaustive: every value routed or cited-refused.
- **P2.12** — bitwise, not `allclose`.
- **P2.13** — the audit table has no `missing` row left without a citation, and
  the `Refused` rows name the upstream file.

## Parallelisation

- **Wave 0** — P2.0 (serial, produces the table everything else needs), P2.1 and
  P2.7 independent.
- **Wave 1** — 8 tasks, 8 agents, all **new files**. No two touch the same
  module. They read `$OR_SRC` and the P2.0 table; they do not write to shared
  files. This is where the parallelism is.
- **Wave 2** — P2.11, P2.12, P2.13 are all contended-file tasks; serial, one
  owner each. P2.11 must land before P2.13 (the latter closes the audit's
  `missing` rows, which are dispatched into groups by P2.11).

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p2_reconciliation.py   # every routine classified
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area engine/source/elements/solid
python tools/validate_vs_fortran.py --mode parity --cases RD-E-1000 --out tools/validation_data/parity_p2.json
```

Additionally the phase reviewer confirms, by reading the P2.0 table:

1. no row remains `missing` without a cited reason;
2. every `approximate` row has a parity measurement attached;
3. `pyradioss/elements/solid_tshell8.py`, `shell_thick16.py`,
   `thickshell_wedge6.py` and `thickshell_composite.py` between them cover all
   four upstream thickshell directories — the phase's thickshell obligation is
   to make that statement **provable from the census**, not to expand them.