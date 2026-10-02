# Phase 3 — Elements: shells

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/elements/shell/**` and `sh3n/**` a literal port,
and close the three measured parity gaps the port itself documents: the BT
family, DKT18, and the orthotropic QEPH stabilisation.

**Architecture:** the same routine-set criterion as Phase 2. The phase is
ordered by **measured deviation**, not by code size — the small `cdefo3.F`
branches that `VALIDATION.md` §7 item 1 names as "the two concrete targets" come
before the 16,000-line QBAT polish, because they are the only places where a
known-wrong result is already quantified.

**Tech stack:** NumPy, numba mirrors (`jit_kernels/shells_dkt18.py`,
`shells_qbat.py`, `shells_qeph.py` already exist), optional CuPy GPU path
(`accel/gpu_kernels`).
**Spec:** `VALIDATION.md` §3.6 (M41 shell-element-technology parity),
§7 items 1, 2, 3, 5, 6.
**Entry criteria:** Phase 1 exit gate; Phase 0 oracle.
**Band: L.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `engine/source/elements/shell/coque` (BT + GPU split) | 26 | 10,439 | `shell_bt4.py` (1,913) |
| `engine/source/elements/shell/coqueba` (QBAT) | 30 | 16,403 | `shell_qbat.py` (2,166) |
| `engine/source/elements/shell/coquez` (QEPH) | 18 | 15,747 | `shell_qeph.py` (1,600) |
| `engine/source/elements/shell/` (common) | 4 | 804 | none |
| `engine/source/elements/sh3n/coque3n` (C0 triangle) | 25 | 7,570 | `shell_tri3.py` (1,017) |
| `engine/source/elements/sh3n/coquedk` (DKT18) | 11 | 3,613 | `shell_dkt18.py` (750) |
| `engine/source/elements/sh3n/coquedk6` (DKT6) | 10 | 2,746 | `shell_dkt6.py` (459) |
| **Total** | **124** | **57,322** | **7,905** |

Empty upstream directories (0 files — verify, do not assume):
`engine/source/interfaces/ists/`, `.../ists_q1np/`, `.../shell_offset/`.

**Dispatch** (`pyradioss/elements/__init__.py`): `SHELL_ISHELL_GROUPS` routes
`Ishell` 12 → QBAT, 22/23/24 → QEPH (22 and 23 folded into 24 by the Starter per
`hm_read_prop01.F:186-192`). `SH3N_ISHELL_GROUPS` routes `Ish3n` 2 → DKT18,
3 → DKT6. Everything else goes to `shells` (BT).

## Gap analysis

### The three measured gaps (`VALIDATION.md` §7)

1. **The BT family is DEVIATION 0.138–0.257** (c40–c45, 88–96 % of `/RUN`,
   worst channel HE or MOMZ). §7 item 1 is explicit that this is *not*
   "unfinished" but "unreachable by comparison", because the reference itself
   diverges in its final window (Fortran HE = 76 % of its own IE at TSTOP,
   printed ENERGY ERROR −40.7 %). It nevertheless names **two concrete
   unported targets**:
   - **`cdefo3.F` for c43 (BT type 3)** — the node-1-relative velocity form. It
     currently takes **no** `rot2` correction and gains only via the FAC=9 dt.
   - **`cdefo3.F` for c45 (BT type 4)** — the Z2 warp correction.
   `PORTING_GUIDE.md:60` records the M41 fix for the general branch:
   `TMP1A=(dt/4)(VZ13−VZ24)²/(PY1+PY2)` into `VX13`/`VX24`,
   `TMP2B=(dt/4)(VZ13+VZ24)²/(PX2−PX1)` into `VY13`/`VY24`, with
   `SIGN(MAX(ABS,EM20))` guards, disabled when `IMPL_S>0`. The 43/45 branches are
   the same correction in a different velocity frame.
2. **DKT18 is 0.2581–0.4556** (c06 0.2813, c07 0.2581, worst channel MOMX on
   both; c00 twisted beam 0.4556). §7 item 2 calls it "the only unported shell
   formulation in the family" as of M41 — it *has* a kernel (`shell_dkt18.py`,
   750 lines) and 3 numba/jit files, so the claim in §7 is stale and the real
   work is **depth**, not existence. The first job is to find out what.
3. **The orthotropic QEPH stabilisation (`czfintn_or`, the HM/HF path) is NOT
   ported** — LAW19 shells run the isotropic `czfintn` moduli, flagged in
   `checks.py` (§7 item 5).

### Missing routines / features

| Item | Fortran | LOC | Note |
|---|---|---:|---|
| Shell basis table `Z01` (11×11) | `shell/coqini.F` | 130 | built at startup by `radioss2.F`; no port counterpart. Drives the higher-order shell formulations. |
| Thick-shell error check | `shell/err_thk.F` | ~80 | |
| Force equilibrium | `shell/fequilibre.F` | ~250 | |
| Shell offset / warping-membrane init | `shell/shell_offset_wm_ini.F90` | ~250 | also `interfaces/shell_offset` (empty dir) |
| Orthotropic QEPH | `shell/coquez/czfintn.F`'s `_or` branch | — | see gap 3 |
| Drilling stiffness `Idrill` | `coque/cdrill*`, `coqueba`, `coquez` | — | `Idrill>0` documented as a cut in **both** QBAT and QEPH; `IGEO(20)=ISROT` in `hm_read_prop01.F` is the reader side |
| `Ithick=1` thickness update | `coqueba`, `coquez` | — | documented cut in QBAT |
| QBAT `nip=1` branch | `coqueba/cbafori1`, `cbavisnp1` | — | documented cut |
| QBAT/QEPH `tangent()`/`kgeo()` | `imp_int_k` counterparts | — | currently a **loud refusal**; Phase 15 needs them |
| `real_type.h` precision selection | `coque/real_type.h` | ~80 | |
| CUDA kernels | `shell_geometry_kernel.cu`, `shell_strain_material_kernel.cu`, `shell_force_assembly_kernel.cu`, `shell_gpu_driver.cu` | ~3,000 | `shell_gpu_mod.F90` dispatch is called from `resol.F`; the port has `accel/gpu_kernels` (1,157 lines) but no claim on these four |
| `coque` auxiliary routines | `ccurv3.F`, `cortdir3.F`, `c4eoff.F`, `layini.F`, `thickvar.F`, `csens3.F`, `cevec3.F`, `cnvec3.F`, `cstra3.F`, `cupdt3.F`, `cupdtn3.F`, `shell_internal_forces.F90` | ~4,000 | partially present |

### Documented deliberate cuts that this phase must keep *visible*

`PORTING_GUIDE.md:61-62` and `VALIDATION.md` §7 item 5 list, per kernel:
QBAT `nip=1` (`CBAFORI1`/`CBAVISNP1`), `Idrill>0`, `Ithick=1`, no tangent/kgeo,
XFEM/thermal; QEPH `ZCFAC=1` in the plastic relaxation (exact for elastic
laws), `NPT=0` resolved to 3 Gauss stations, `Idrill=1`, `ISMSTR=1/11`,
XFEM/thermal, no implicit tangent, no orthotropic path. **Each cut must remain
either ported or explicitly refused with its upstream citation. "Silently
missing" is not an acceptable state for any of them.**

## Wave graph

```
Wave 0 (parallel)
  P3.0 shell reconciliation audit (serial — produces the status table)
  P3.1 cdefo3 c43 branch        P3.2 cdefo3 c45 branch
  P3.3 QEPH orthotropic path    P3.4 coqini.Z01 basis + err_thk + fequilibre
   ── gate: the two cdefo3 branches re-measure c43/c45; audit table complete
Wave 1 (parallel — new files)
  P3.5 Idrill drilling stiffness across BT/QBAT/QEPH
  P3.6 Ithick=1 thickness update
  P3.7 QBAT nip=1 branch (CBAFORI1/CBAVISNP1)
  P3.8 shell_offset_wm_ini
  P3.9 DKT18 depth (conditional on P3.0's finding)
  P3.10 QBAT/QEPH tangent + kgeo      (consumed by Phase 15)
   ── gate: each has an analytic test + one oracle parity case
Wave 2 (serial)
  P3.11 CUDA kernels: port or formally decline
  P3.12 reconcile _CONDENSED_FACDT ownership in shell_bt4.py (§7 item 6)
  P3.13 numba mirrors for QBAT/QEPH (already have files; make them earn it)
```

---

### Task P3.0: Shell reconciliation audit

**Fortran:** all seven directories in §Scope.
**Files:** Create `tools/validation_data/shell_routine_status.json`,
`tests/test_p3_reconciliation.py`.
**Interfaces:**
- Produces: the same record shape as `solid_routine_status.json` (Task P2.0),
  one row per upstream shell routine, plus a `parity` sub-record per family:
  `{"family": "dkt18", "cases": {"c06": 0.2813, "c07": 0.2581},
    "worst_channel": "MOMX"}` seeded from `parity_m41.json`.

- [ ] **Step 1** — write the failing test:

```python
def test_dkt18_parity_gap_is_localised_not_global():
    st = json.loads(Path("tools/validation_data/shell_routine_status.json").read_text())
    dk = [r for r in st if r["family"] == "dkt18"]
    missing = [r["fortran"] for r in dk if r["status"] == "missing"]
    assert missing == [], f"DKT18 has {len(missing)} unported routines: {missing}"
```

- [ ] **Step 2** — run → FAIL (no table).
- [ ] **Step 3** — do the audit. For DKT18 specifically, establish **which**
  routine explains a 0.2581 MOMX deviation. `cdkcoor3.F`, `cdkdefo3.F`,
  `cdkderi3.F`, `cdkstra3.F`, `cdkfint3.F`, `cdkforc3.F`, `cncoef3.F`,
  `cdkfcum3.F`, `cndt3.F`, `dtcdk_reg.F`, `cdkfint_reg.F` — the *force* being
  wrong while the *strain* is right points at `cdkfint3.F`; both wrong points at
  `cdkdefo3.F`/`cdkderi3.F`. Record the finding as a named hypothesis per
  routine, not as a conclusion. **Do not fix anything here.**
- [ ] **Step 4** — run → PASS; the test additionally asserts every
  `partial`/`approximate` row has a `deviation_note`; commit
  `docs(shells): per-routine reconciliation audit + DKT18 deviation localisation`.

---

### Task P3.1: `cdefo3.F` — BT type 3 (c43) node-1-relative velocity form

**Fortran:** `engine/source/elements/shell/coque/cdefo3.F` (the M41 branch is
recorded at `PORTING_GUIDE.md:60`; the c43 branch is the one that currently takes
no `rot2` correction), `cdt3.F` for the condensation length.
**Files:** Modify `pyradioss/elements/shell_bt4.py`, `pyradioss/accel/jit_kernels/shells_bt4.py`.
**Interfaces:**
- Consumes: `shell_bt4.forces(group, x, v, vr, dt, fint, mint)`;
  `rot2_mask` (the existing `Ishell` cards 0/1/2 → engine `IHBE≤1` mapping).
- Produces: `shell_bt4._rot2_correction(px, py, vx, vy, dt, kind: int) -> tuple[np.ndarray, np.ndarray]`
  with `kind ∈ {0: none, 1: general (exists), 3: node-1-relative, 4: Z2 warp}`.

- [ ] **Step 1** — write the failing test:

```python
def test_bt3_takes_the_node1_relative_velocity_correction():
    """A rolled BT element whose mid-step frame lags must have its out-of-plane
    velocity leak removed. Identity: with VZ13 = VZ24 the correction vanishes."""
    px, py = np.array([1e5, 1e5]), np.array([1e5, 1e5])
    vx = np.array([0.3, 0.3]); vy = np.array([0.0, 0.0])
    dx, dy = shell_bt4._rot2_correction(px, py, vx, vy, dt=1e-4, kind=3)
    assert np.abs(dx).max() == 0.0 and np.abs(dy).max() == 0.0

    vx = np.array([0.3, 0.3])
    dx3, _ = shell_bt4._rot2_correction(px, py, vx, vy, dt=1e-4, kind=3)
    dx1, _ = shell_bt4._rot2_correction(px, py, vx, vy, dt=1e-4, kind=1)
    assert np.abs(dx3).max() != np.abs(dx1).max()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement by reading the c43 branch of `cdefo3.F` directly.
  **Do not extrapolate from the general branch.** The M41 fix deliberately
  mirrored an upstream asymmetry (the static bias is `+ω²dt/2` raw and
  `−ω²dt/2` corrected — upstream over-corrects 2×; `IHBE==2/3`'s velocity form
  is the exact cancellation) — that decision applies to the general branch and
  does **not** license the same choice on c43. Read which one c43 uses.
  The `SIGN(MAX(ABS,EM20))` guards and the `IMPL_S>0` disable must be copied.
  Update the numba mirror in the same commit, bitwise.
- [ ] **Step 4** — run → PASS; **re-measure the c43 parity case** and record the
  new rel-RMS in `tools/validation_data/parity_p3.json`; commit
  `fix(shells): port the cdefo3 BT type-3 node-1-relative velocity branch`.

---

### Task P3.2: `cdefo3.F` — BT type 4 (c45) Z2 warp correction

**Fortran:** the c45 branch of `cdefo3.F`, plus `ccurv3.F` (the Z2 curvature it
consumes).
**Files:** Modify `pyradioss/elements/shell_bt4.py`, `ccurv3` equivalent,
`pyradioss/accel/jit_kernels/shells_bt4.py`.
**Interfaces:**
- Consumes: `shell_bt4` element curvature state.
- Produces: `shell_bt4._rot2_correction(..., kind=4)`; extends the curvature
  state to carry `Z2` when `kind=4`.

- [ ] **Step 1** — write the failing test:

```python
def test_bt4_z2_warp_correction_is_zero_on_a_flat_element():
    g = _flat_bt4_group()
    dx, dy = shell_bt4._rot2_correction(g.px, g.py, g.vx, g.vy, dt=1e-4, kind=4)
    assert np.abs(dx).max() == 0.0 and np.abs(dy).max() == 0.0

def test_bt4_z2_is_nonzero_on_a_warped_element():
    g = _warped_bt4_group(warp=0.02)
    dx, dy = shell_bt4._rot2_correction(g.px, g.py, g.vx, g.vy, dt=1e-4, kind=4)
    assert np.abs(dx).max() > 0.0
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `Z2` is the second surface curvature mode; getting
  it from `ccurv3.F` rather than recomputing it is the point — the port's
  existing curvature state is BT-type-specific and may not carry `Z2`.
- [ ] **Step 4** — run → PASS; re-measure c45; commit
  `fix(shells): port the cdefo3 BT type-4 Z2 warp-correction branch`.

---

### Task P3.3: Orthotropic QEPH (`czfintn_or`)

**Fortran:** the `_or` branch of
`engine/source/elements/shell/coquez/czfintn.F` (HM/HF moduli) and its caller in
`czdef.F`/`czforc3.F`; the LAW19 shell modulus source
`engine/source/materials/mat/mat019/sigeps19c.F`.
**Files:** Modify `pyradioss/elements/shell_qeph.py`, `pyradioss/materials/law19_fabric.py`,
`pyradioss/input/checks.py`.
**Interfaces:**
- Consumes: `materials.shell_moduli(layer, law)` — currently returning isotropic
  moduli for LAW19.
- Produces: `materials.shell_moduli(layer, law) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  returning `(A11, A12, G)` per Gauss point, anisotropic when the law supplies
  them; `shell_qeph._stabilisation(..., moduli)` selecting the `_or` branch.

- [ ] **Step 1** — write the failing test:

```python
def test_law19_shell_no_longer_reports_the_isotropic_refusal():
    g = _qeph_group(law=19)
    from pyradioss.input.checks import warnings_for
    assert not any("isotropic" in w for w in warnings_for(g))

def test_qeph_anisotropic_stabilisation_differs_from_isotropic():
    g = _qeph_group(law=19)
    A, h = shell_qeph._stabilisation(g, anisotropic=True)
    Ai, hi = shell_qeph._stabilisation(g, anisotropic=False)
    assert not np.allclose(A, Ai)
    assert h.energy >= 0.0
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Read the `_or` branch's **which moduli and in
  which order** — the HM/HF path substitutes the *material's own* plane-stress
  moduli for the isotropic ones, and the substitution list is not symmetric.
  Phase 6 owns `law19_fabric.py`; this task touches it, so it takes Phase 6's
  entry criterion (a frozen law19 API) and gives back a documented extension.
  Remove the `checks.py` isotropic warning **only** when this path is live.
- [ ] **Step 4** — run → PASS; parity on a LAW19 QEPH deck; commit
  `feat(shells): orthotropic QEPH stabilisation (czfintn_or HM/HF path)`.

---

### Task P3.4: `coqini.Z01`, `err_thk.F`, `fequilibre.F`

**Fortran:** `engine/source/elements/shell/coqini.F` (the `DATA Z01/` 11×11
block, read off lines 24-60 of that file), `err_thk.F`, `fequilibre.F`.
**Files:** Create `pyradioss/elements/shell_basis.py`,
`tests/test_p3_shell_basis.py`.
**Interfaces:**
- Produces: `shell_basis.Z01() -> np.ndarray` (11×11), `shell_basis.WF1()`,
  `shell_basis.WM1()`, `shell_basis.ZN1()` — the startup tables `coqini.F`
  builds; `shell_basis.check_thickness(thick) -> list[str]` (the `err_thk`
  errors); `shell_basis.force_equilibrium(fint, fint_ref, tol) -> np.ndarray`
  (the `fequilibre` diagnostic).

- [ ] **Step 1** — write the failing test:

```python
def test_z01_is_transcribed_not_derived():
    Z = shell_basis.Z01()
    assert Z.shape == (11, 11)
    # spot values read off coqini.F DATA block, rows 3,4,5 (0-based 2,3,4)
    assert Z[2, 0] == -0.5 and Z[2, 1] == 0.5
    assert Z[3, 0] == 0.0 and Z[3, 1] == 0.0 and Z[3, 2] == 0.0
    assert Z[4, 0] == -0.5 and Z[4, 1] == 0.0 and Z[4, 2] == 0.5

def test_force_equilibrium_flags_a_net_force_on_a_free_element():
    f = np.zeros((4, 3)); f[0, 0] = 1.0
    assert shell_basis.force_equilibrium(f, np.zeros((4, 3)), tol=1e-9).any()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The `DATA` block is data, not a formula: it is
  transcribed digit-for-digit and the test pins the spot values so a
  transcription slip is caught.** `err_thk.F` produces *errors* (aborts) in
  upstream — port the abort, do not soften it to a warning, and cite the file.
  `fequilibre.F` is a diagnostic the Engine calls; it must be reachable from
  `engine.py` and must be cheap enough to leave on.
- [ ] **Step 4** — run → PASS; commit `feat(shells): coqini basis tables, err_thk, fequilibre`.

---

### Task P3.5: Drilling stiffness `Idrill`

**Fortran:** the drilling terms in `coque` (`IGEO(20)=ISROT` and the drill
moment in `cforc3.F`/`cfint3.F`), `coqueba`, `coquez`; the reader
`hm_read_prop01.F:221-224` (`IF(ISROT==0)ISROT=IDRIL_D`, `IF(ISROT==2)ISROT=0`).
**Files:** Modify `pyradioss/elements/shell_bt4.py`, `shell_qbat.py`,
`shell_qeph.py`, `pyradioss/input/keywords/properties.py`.
**Interfaces:**
- Consumes: `prop.Isrot` / `prop.Idrill` on the shell property (currently not
  surfaced — a reader gap, cited at `PORTING_GUIDE.md:61`).
- Produces: `Idrill` and `Isrot` on the shell property object; a drill-moment
  term in each of the three `forces()` paths.

- [ ] **Step 1** — write the failing test:

```python
def test_idrill2_is_mapped_to_zero_isrot():
    """hm_read_prop01.F:223  IF(ISROT==2) ISROT=0"""
    prop = _shell_prop(isrot=2)
    assert prop.isrot == 0

def test_drilling_stiffness_resists_a_node_spin():
    g = _bt4_group(idrill=True)
    f0 = g.forces(dt=1e-3)                     # at rest
    g.vr[0] = 0.01                             # spin node 0 about its normal
    f1 = g.forces(dt=1e-3)
    assert np.abs(f1 - f0).sum() > 0.0
```

- [ ] **Step 2** — run → FAIL (`Idrill` not surfaced).
- [ ] **Step 3** — implement the reader half first (it is the smaller half and it
  unblocks the kernel half). **`Idrill` is not a plain stiffness**: upstream
  applies it through the drilling-degree of freedom bookkeeping, and the
  `IF(ISROT==2) ISROT=0` mapping is a *non-obvious* reader rule that the test
  pins. Do not guess its magnitude; read `cfint3.F`.
- [ ] **Step 4** — run → PASS; parity on an `Idrill>0` deck; commit
  `feat(shells): drilling stiffness across BT/QBAT/QEPH (Idrill, Isrot)`.

---

### Task P3.6: `Ithick=1` thickness update

**Fortran:** the `Ithick` branch in `coqueba` and `coquez`.
**Files:** Modify `pyradioss/elements/shell_qbat.py`, `shell_qeph.py`.
**Interfaces:**
- Produces: `Ithick` on the shell property; a per-cycle thickness update in
  both `forces()` paths.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("group_name", ["shells_qbat", "shells_qeph"])
def test_ithick_one_updates_thickness(group_name):
    g = _shell_group(group_name, ithick=1)
    t0 = g.thickness.copy()
    g.forces(dt=1e-3)
    assert not np.array_equal(g.thickness, t0)

def test_ithick_zero_leaves_thickness_alone():
    g = _shell_group("shells_qbat", ithick=0)
    t0 = g.thickness.copy(); g.forces(dt=1e-3)
    assert np.array_equal(g.thickness, t0)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The `Ithick=0` no-op test is the important one:
  `Ithick` is 0 on every deck the corpus runs today, so a kernel that updates
  thickness unconditionally would silently change every existing result.
- [ ] **Step 4** — run → PASS; commit `feat(shells): Ithick=1 thickness update`.

---

### Task P3.7: QBAT `nip=1` branch

**Fortran:** `engine/source/elements/shell/coqueba/cbafori1`, `cbavisnp1`
(the single-Gaffer-point path with `SHF=0`).
**Files:** Modify `pyradioss/elements/shell_qbat.py`.
**Interfaces:**
- Produces: `shell_qbat.NIP` from the group; a single-point branch in
  `forces()`; the Starter warning at `PORTING_GUIDE.md:61` is **removed** once
  this lands.

- [ ] **Step 1** — write the failing test:

```python
def test_qbat_nip1_reduces_to_the_single_point_path():
    g = _qbat_group(nip=1)
    out = g.forces(dt=1e-3)
    assert np.all(np.isfinite(out)) and np.all(out > 0)
    assert not any("nip" in w.lower() for w in g.log.warnings)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The two branches are not the same computation with
  a different loop count — `CBAFORI1` is a distinct force assembly. Read both
  before writing either.
- [ ] **Step 4** — run → PASS; parity on an `N=1` QBAT deck (the port's note says
  no official deck runs it, so this may be a synthetic deck — **label it as
  synthetic in the test docstring**); commit
  `feat(shells): QBAT nip=1 single-point branch (CBAFORI1/CBAVISNP1)`.

---

### Task P3.8: Shell offset / warping-membrane initialisation

**Fortran:** `engine/source/elements/shell/shell_offset_wm_ini.F90`,
`common_source/modules/elements/element_mod.F90` (the offset fields), the
`interfaces/shell_offset/` directory (**empty upstream — verify**).
**Files:** Create `pyradioss/elements/shell_offset.py`,
`tests/test_p3_shell_offset.py`.
**Interfaces:**
- Produces: `shell_offset.initialise(group, model) -> np.ndarray` (the
  warping-membrane reference offsets), consumed by the Starter's element init.

- [ ] **Step 1** — write the failing test:

```python
def test_offset_initialisation_is_zero_without_the_keyword():
    g = _bt4_group(offset=None)
    assert np.array_equal(shell_offset.initialise(g, None), np.zeros((4, 3)))

def test_offset_shifts_the_membrane_stiffness_not_the_moment():
    g = _bt4_group(offset=0.1)
    before = g._membrane_stiffness()
    shell_offset.initialise(g, model=None)
    assert not np.allclose(before, g._membrane_stiffness())
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Confirm the empty upstream directory before
  claiming anything about it** — `$OR_SRC/engine/source/interfaces/shell_offset/`
  has 0 files as of 2026-10-02, so the whole obligation rests on
  `shell_offset_wm_ini.F90`. If the function's only caller is absent from the
  open tree, record it in `docs/PORT_EXTENSIONS.md` as unreachable rather than
  claiming a port.
- [ ] **Step 4** — run → PASS; commit `feat(shells): shell offset / warping-membrane init`.

---

### Task P3.9: DKT18 depth (scope set by P3.0)

**Fortran:** `engine/source/elements/sh3n/coquedk/*` (11 files, 3,613 LOC).
**Files:** Modify `pyradioss/elements/shell_dkt18.py`.
**Interfaces:**
- Consumes: the P3.0 hypothesis table for the `dkt18` family.
- Produces: whatever routine P3.0 identified.

- [ ] **Step 1** — the test named by P3.0's row for the identified routine. If
  P3.0 localised the deviation to `cdkfint3.F`, the test is a single-element
  bending case with a hand-computed moment; if to `cdkdefo3.F`, it is a
  constant-curvature case. **This step may not be written generically** — the
  whole point of P3.0 is that it names the routine, and this task must follow
  it.
- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, per P3.0's finding.
- [ ] **Step 4** — run → PASS; **re-measure c06, c07 and c00** and report the
  three new rel-RMS values in the commit message. If the deviation does not
  move, that is a finding, not a failure — record it and hand the next
  hypothesis to the following task; do **not** tune a threshold to make it move.

---

### Task P3.10: QBAT/QEPH tangents and geometric stiffness

**Fortran:** the `imp_int_k`/`imp_kgeo` counterparts for the Batoz family; the
refusal the port currently raises is documented at `PORTING_GUIDE.md:61-62`.
**Files:** Modify `pyradioss/elements/shell_qbat.py`, `shell_qeph.py`,
`pyradioss/input/checks.py`.
**Interfaces:**
- Produces: `shell_qbat.tangent(group) -> (triplets, n_dof)`,
  `shell_qbat.kgeo(group) -> (triplets, n_dof)`; the same for `shell_qeph`.
  Both satisfy the implicit assembly contract in
  `pyradioss/implicit/assembly.py`.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("mod", [shell_qbat, shell_qeph])
def test_tangent_matches_a_finite_difference_of_the_residual(mod):
    g = _qbat_like_group()
    T = mod.tangent(g)
    h = 1e-8
    for dof in (0, 3, 7, 11):
        plus, minus = g.copy(), g.copy()
        plus.u[dof] += h; minus.u[dof] -= h
        fd = (mod.static_internal_forces(plus) - mod.static_internal_forces(minus)) / (2*h)
        assert np.allclose(T.apply(g, dof), fd, rtol=2e-5, atol=1e-6)
```

- [ ] **Step 2** — run → FAIL (currently a loud refusal).
- [ ] **Step 3** — implement. **Phase 15 is the declared consumer**
  (`plan/15_phase14_implicit_parity.md`); if that phase is deferred, this task
  still lands but the refusal gate stays for the *nonlinear-geometry* case.
  The finite-difference test is the specification: a tangent that is merely
  symmetric-positive will fail it.
- [ ] **Step 4** — run → PASS; commit `feat(shells): QBAT/QEPH tangent and geometric stiffness`.

---

### Task P3.11: The CUDA kernels — port or formally decline

**Fortran:** `engine/source/elements/shell/coque/shell_geometry_kernel.cu`,
`shell_strain_material_kernel.cu`, `shell_force_assembly_kernel.cu`,
`shell_gpu_driver.cu`, `shell_gpu_mod.F90`, `shell_gpu_data.h`.
**Files:** Modify `pyradioss/accel/gpu_kernels/__init__.py`, `docs/PORT_EXTENSIONS.md`.
**Interfaces:**
- Produces: a decision record. Either (a) a CuPy port of the three kernels with
  the host-side `shell_gpu_mod.F90` dispatch in
  `pyradioss/accel/gpu_kernels`, or (b) a `docs/PORT_EXTENSIONS.md` entry
  recording them as declined with the reason.

- [ ] **Step 1** — write the failing test:

```python
def test_gpu_shell_path_is_either_ported_or_declined():
    text = Path("docs/PORT_EXTENSIONS.md").read_text()
    either = all(k in text for k in ("shell_geometry_kernel",
                                     "shell_strain_material_kernel",
                                     "shell_force_assembly_kernel"))
    assert either or accel.gpu_shell_available()
```

- [ ] **Step 2** — run → FAIL (neither).
- [ ] **Step 3** — decide and implement. **Option (b) is acceptable and is the
  recommended outcome**: the port's stated goal is readability, not GPU
  throughput; the NumPy and numba paths already cover performance. What is *not*
  acceptable is silence — a `.cu` file upstream with no port and no record reads
  as an oversight. The port must keep the **host-side dispatch semantics**
  (`shell_gpu_mod.F90` is called from `resol.F`, so it decides *when* a GPU path
  is taken and what the fallback is), because that is solver logic, not
  throughput.
- [ ] **Step 4** — run → PASS; commit
  `docs(extensions): record the shell CUDA kernels as declined (with the host dispatch ported)`.

---

### Task P3.12: Reconcile `_CONDENSED_FACDT` ownership in `shell_bt4.py`

**Fortran:** `coque/cdt3.F`, `coqueba/cndt3.F`, `coquez/`'s `FACDT=5/4`
condensation (per `PORTING_GUIDE.md:61-62`).
**Files:** Modify `pyradioss/elements/shell_bt4.py`.
**Interfaces:**
- Produces: `shell_bt4._CONDENSED_FACDT` deleted; the condensation owned solely
  by the kernel that computes it.

- [ ] **Step 1** — write the failing test:

```python
def test_bt4_owns_no_dt_claim_for_qbat_or_qeph_groups():
    assert not hasattr(shell_bt4, "_CONDENSED_FACDT")
    g = _group(ishell=12)
    assert g._owns_dt_claim is False
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `VALIDATION.md` §7 item 6 records this precisely:
  the M40 branches are now **shadowed** for every deck the new starter dispatch
  routes to QBAT/QEPH, so the code is harmless but is duplicated Batoz-family
  claim logic in a file that no longer owns it. Deleting it is the fix; the test
  is what stops it drifting back.
- [ ] **Step 4** — run → PASS; **the BT, QBAT and QEPH dt claims must be
  unchanged** — this is a refactor with a physics-adjacent blast radius, so the
  commit quotes the three dt values before and after; commit
  `refactor(shells): remove shadowed Batoz dt-claim logic from shell_bt4`.

---

### Task P3.13: numba mirrors for QBAT/QEPH

**Fortran:** none. **Files:** Modify
`pyradioss/accel/jit_kernels/shells_qbat.py`, `shells_qeph.py`.
**Interfaces:**
- Consumes: the kernels from Tasks P3.3, P3.6, P3.7, P3.10.
- Produces: working mirrors, so the `auto` rule stops picking the slower path.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("case", ["c04", "c08", "c09"])
def test_qbat_qeph_numba_beats_numpy(case):
    ref_t = bench(case, backend="numpy"); num_t = bench(case, backend="numba")
    assert num_t < ref_t, f"{case}: numba {num_t:.1f}s vs numpy {ref_t:.1f}s"
```

- [ ] **Step 2** — run → FAIL (VALIDATION.md §6.5(c): c04 is numba 395 s vs numpy
  330 s — **numba is currently the slower path** because dispatch overhead is
  added with nothing compiled).
- [ ] **Step 3** — implement. Two acceptable outcomes, both fine:
  (a) real mirrors that are both **bitwise identical** and faster; or
  (b) the `auto` rule in `pyradioss/accel/__init__.py` **excludes**
  `shells_qbat`/`shells_qeph` until mirrors exist, and the test above is
  replaced by one asserting the exclusion. Option (b) is a smaller change with
  the same user-visible effect and is the one to prefer if the mirrors would not
  clear the speed bar. **Do not do neither.**
- [ ] **Step 4** — run → PASS; T01 md5 unchanged for c04/c08/c09; commit
  `perf(accel): QBAT/QEPH numba mirrors or an auto-rule exclusion`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P3.0** — DKT18's deviation is localised to named routines with hypotheses,
  not attributed vaguely.
- **P3.1 / P3.2** — the branch is read from `cdefo3.F`, not extrapolated from the
  general branch; the `SIGN(MAX(ABS,EM20))` guards and the `IMPL_S>0` disable are
  present; the new parity number is in the commit body.
- **P3.3** — the `checks.py` isotropic warning is removed only when the path is
  live.
- **P3.4** — the `DATA` block is transcribed digit-for-digit and the spot-value
  test proves it; `err_thk` still **aborts**.
- **P3.5** — `IF(ISROT==2) ISROT=0` is honoured; the magnitude comes from
  `cfint3.F`, not from a plausible formula.
- **P3.6 / P3.7** — the `Ithick=0` / `nip=2` no-op cases are green, so every
  existing deck is unchanged.
- **P3.9** — if the deviation did not move, the commit says so with numbers and
  hands over the next hypothesis. **A threshold was not tuned.**
- **P3.10** — the tangent passes a finite-difference of the residual, not a
  symmetry check.
- **P3.12** — the three dt values are quoted before and after.
- **P3.13** — outcome (a) or (b); never neither.

## Parallelisation

- **Wave 0** — P3.0 serial; P3.1, P3.2, P3.3, P3.4 are new-code or
  single-module tasks and run in parallel with it. P3.3 touches `law19_fabric.py`
  and therefore claims Phase 6's frozen-law19 window explicitly.
- **Wave 1** — 6 tasks, 6 agents. P3.5 is the contended one (`shell_bt4`,
  `shell_qbat`, `shell_qeph`, plus a reader) and is scheduled **last** in the wave
  because P3.1/P3.2 also edit `shell_bt4.py`.
- **Wave 2** — serial, contended.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p3_reconciliation.py
python tools/validate_vs_fortran.py --mode parity --cases RD-E-1000 --out tools/validation_data/parity_p3.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
```

The phase reviewer additionally reports, in measured values, the new c43 / c45 /
c06 / c07 / c00 rel-RMS and states plainly whether the BT-family conclusion
changed. **If the BT family is still DEVIATION, that is an acceptable outcome
of this phase** provided the two named branches are ported and the remaining
gap is characterised — §7 item 1's framing stands, and the reviewer must not
accept a weakened guard threshold as progress.