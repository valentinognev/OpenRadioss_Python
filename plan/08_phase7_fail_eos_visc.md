# Phase 7 — Failure models, equations of state, viscoelasticity, material tools

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/materials/{fail,eos,visc,tools}` and
`common_source/eos` a literal port. Failure models are the port's deletion
mechanism, so this phase is load-bearing for **contact** (Phase 8): a deleted
element must drop out of every interface, and that only works if every `/FAIL`
criterion sets the same flag in the same way.

**Architecture:** one task per failure model (40 upstream directories), plus
the EOS set, plus the viscoelastic set, plus the shared table/interpolation
tools Phase 6 also consumes. **Deletion semantics are fixed once, up front, by
Task P7.0** and every subsequent model must satisfy it — this is the one place
where an inconsistent convention would produce silent, model-dependent contact
divergence.

**Tech stack:** NumPy; the EOS set is the largest pure-math group in the
program and needs no element machinery at all, so it parallelises perfectly.
**Spec:** `plan/00_ORCHESTRATION.md` §10 item 5 (a `/FAIL` on an unsupported
family must refuse loudly).
**Entry criteria:** Phase 6's `Law` contract is fixed; Phase 0 oracle.
**Band: L.**

---

## Scope

### Failure models — 40 upstream directories + 3 top-level routines

| Directory | LOC | Port |
|---|---:|---|
| `failwave` (6 files: `seg_intersect.F`, `set_failwave_nod3/nod4.F`, `update_failwave.F`, `upd_failwave_sh3n/sh4n.F`) | 1,661 | `failure/failwave.py` |
| `tabulated` | 3,164 | `failure/tab.py`, `tab1.py`, `tab2.py` |
| `gene1` | 1,703 | `failure/gene1.py` |
| `tensstrain` | 1,971 | `failure/tensstrain.py` |
| `fld` (forming limit diagram) | 1,360 | `failure/fld.py` |
| `orthstrain` | 1,409 | `failure/orthstrain.py` |
| `johnson_cook` | 1,093 | `failure/johnson.py` |
| `biquad` | 1,204 | `failure/biquad.py` |
| `alter` | 1,249 | `failure/alter.py` |
| `inievo` | 1,002 | `failure/inievo.py` |
| `orthenerg` | 915 | `failure/orthenerg.py` |
| `tuler_butcher` | 955 | `failure/tbutcher.py` |
| `cockroft_latham` | 614 | `failure/cockcroft.py` |
| `energy` | 646 | `failure/energy.py` |
| `visual` | 660 | `failure/visual.py` |
| `syazwan` | 529 | `failure/sahraei.py`, `syazwan.py` |
| `wilkins` | 505 | `failure/wilkins.py` |
| `connect` | 352 | `failure/connect.py` |
| `hoffman` | 390 | `failure/hoffman.py` |
| `max_strain` | 381 | `failure/max_strain.py` |
| `tsaihill` | 374 | `failure/tsaihill.py` |
| `tsaiwu` | 390 | `failure/tsaiwu.py` |
| `wierzbicki` | 394 | `failure/mmc.py` |
| `snconnect` | 276 | `failure/snconnect.py` |
| `rtcl` | 270 | `failure/fail_rtcl.py` |
| `nxt` | 226 | `failure/nxt.py` |
| `emc` | 211 | `failure/emc.py` |
| `fabric` | 205 | `failure/fabric.py` |
| `mullins_or` | 202 | `failure/mullins.py` |
| `hc_dsse` | 277 | `failure/hc_dsse.py` |
| `ladeveze` | 410 | `failure/fail_ladeveze.py` |
| `sahraei` | 522 | `failure/sahraei.py` |
| `orthbiquad` (3 files: `_c`, `_s`) | 720 | `failure/orthbiquad.py` |
| `brokmann` (Starter) | — | `failure/brokmann.py` |
| `fail_setoff_c.F`, `fail_setoff_npg_c.F`, `fail_setoff_wind_frwave.F` (top level) | ~400 | `failure/failwave.py` |
| **`changchang`, `composite`, `hashin`, `lemaitre`, `puck`, `spalling` — 0 files** | **0** | port modules exist: `chang.py`, `fail_composite.py`, `hashin.py`, `lemaitre.py`, `puck.py`, `spalling.py` |

That last row is **the single most important finding in this phase**: six
upstream failure directories contain **zero files**, and the port has modules for
all six. Those modules are either ports of a closed-source tree, or inventions.
Either way they must be reclassified in `docs/PORT_EXTENSIONS.md` exactly as
Phase 2 Task P2.1 does for `solid_tria3.py`. **Do not assume; verify.**

### Equations of state

`engine/source/materials/eos` + `common_source/eos` (4,898 LOC) + the EOS
readers in `starter/source/materials/eos/` (5,021 LOC). The port has
`pyradioss/materials/eos.py` and readers for LINEAR, IDEAL-GAS, NASG, OSBORNE,
PUFF, SESAME, STIFF-GAS. `docs/STATE.md` records
`/EOS/LINEAR` as a **coverage blocker on 6 corpus decks** and `/EOS/GRUNEISEN`,
`/EOS/STIFF-GAS` (4 decks), plus M110's GRUNEISEN/PUFF/TILLOTSON/MURNAGHAN/
OSBORNE/LSZK/NOBLE-ABEL/STIFF-GAS set and M166's POWDER-BURN/COMPACTION/
EXPONENTIAL/IDEAL-GAS-VT.

### Viscoelasticity

`engine/source/materials/visc/` (5 files, 1,655 LOC): `viscmain.F`,
`prony_modelc.F`, `visc_plas.F90`, `visc_prony.F`, `visc_prony_lstrain.F`.
Plus `mat_share/mqvisc8.F`, `mqviscb.F`, `mqvisc26.F`, `mnsvis.F`, `nsvisul.F`,
`fmqviscb.F` and the Starter's `starter/source/materials/visc/` (1,655 LOC).
The port has `/VISC/LPRONY` (M141) and LAW35/LAW38/LAW40/LAW62/LAW70 as
per-law implementations — but `viscmain.F` is the **shared** driver, and having
five per-law Prony implementations instead of one shared one is a divergence risk
of the same kind Phase 6 Task P6.3 fixes for strain rate.

### Material tools

`engine/source/materials/tools/` (14 files, 6,714 LOC) — shared with Phase 6
Task P6.3; `engine/source/materials/mat_share/{thermc.F, thermexpc.F,
tempcg.F, meint.F, meos8.F, mdtsph.F, mreploc.F}`; and
`starter/source/materials/nonlocal/` (196 LOC), `therm/` (575 LOC),
`time_step/` (2,663 LOC).

## Gap analysis

1. **Six failure directories upstream are empty; the port has six modules.**
   Highest-priority item in the phase (Task P7.1).
2. **`failwave` is not a damage-combination rule — it is a failure-wave
   PROPAGATION model.** `failwave/update_failwave.F` + `set_failwave_nod3.F` /
   `set_failwave_nod4.F` maintain `FAILWAVE%FWAVE_NOD_STACK` and
   `FAILWAVE%MAXLEV_STACK`, propagate a wave level through the mesh after a
   local failure, and abort with
   `ERROR IN FAILWAVE PROPAGATION: ELEMENT =` when `MAXLEV > FAILWAVE%SIZE`
   (`set_failwave_nod3.F:236-246`). It uses `seg_intersect.F` for the
   topology. The port's `failure/failwave.py` is a single module against six
   upstream files; whether it models propagation at all is unverified. **This is
   a distinct model family from the damage criteria and needs its own task.**

3. **`/FAIL/*` models set deletion in several different ways.** `docs/STATE.md`
   records per-layer bookkeeping for shells and `GBUF%OFF` for solids; Phase 8's
   contact deletion reads it. **One convention, pinned by one test.**
4. **`/EOS/LINEAR` blocks 6 corpus decks** — the highest-value single keyword in
   this phase by coverage-per-line.
5. **No shared `viscmain`.** Five per-law Prony implementations will diverge.
6. **`/FAIL/XFEM/*`** (`FLD`, `JOHNS`, `TBUTC` per M159) couples to Phase 5's
   XFEM growth; the models exist upstream in `fail/` but the port's
   `failure/fld.py` is the *forming-limit-diagram* model, not the XFEM one.
7. **Non-local damage** (`starter/source/materials/nonlocal/`, plus
   `common_source/modules/nlocal_reg_mod.F90`) is unported, and `/MAT/NONLOCAL`
   is M171.
8. `docs/OPEN_BUGS.md` item 4's `_SOLID_ONLY_LAWS` audit must be reconciled
   against `mulawc.F90` (Phase 6 Task P6.0) — that is Phase 6's job, but its
   **failure** counterpart (which `/FAIL` models have a shell kernel) is this
   phase's, and it is not yet enumerated anywhere.

## Wave graph

```
Wave 0  P7.0 deletion-semantics contract (serial — everything depends on it)
        P7.1 reclassify the six empty-upstream failure modules
        P7.2 shell/solid failure-kernel census
   ── gate: one deletion convention, pinned; no unclassified module
Wave 1 (parallel — one model per task)
  P7.3 failure models, banded by LOC (40 dirs → ~30 live models)
  P7.4 equations of state (12 EOS families)
  P7.5 shared viscoelastic driver (viscmain + Prony)
  P7.6 material tools: thermal, non-local, time-step sound speed
   ── gate per model: monotonic-damage test + deletion test + oracle case
Wave 2 (serial)
  P7.7 XFEM failure models (consumes Phase 5)
  P7.8 failwave propagation model
  P7.9 registry + coverage re-measure
```

---

### Task P7.0: The deletion-semantics contract

**Fortran:** `engine/source/materials/fail/` dispatch, the `GBUF%OFF` flag
(`common_source/modules/elements/element_mod.F90`, and the port's
`pyradioss/model/entities.py` element-state layout), the per-layer shell
deletion the port already implements (`docs/STATE.md` §Failure), and
`engine/source/interfaces/interf/chkstfn3.F`'s `IDEL` bookkeeping (which Phase 8
owns but which reads this flag).
**Files:** Create `pyradioss/failure/base.py`, `tests/test_p7_deletion.py`.
**Interfaces:**
- Produces — **the contract every later task in this phase implements**:

```python
class FailureModel(Protocol):
    law: int
    def damage(self, state: ElementState, props: FailProps,
               model: Model) -> np.ndarray:
        """Per-integration-point damage in [0, 1]; monotone non-decreasing."""

    def should_delete(self, state: ElementState, props: FailProps) -> np.ndarray:
        """Per-ELEMENT bool. The single authoritative deletion predicate."""
```

  plus
  - `failure.base.apply(model, log) -> np.ndarray` — the one pass that calls
    every registered model and writes `state["off"]`;
  - `failure.base.layer_mask(group) -> np.ndarray` — the per-layer mask for
    shells (already the port's convention; now pinned);
  - `failure.base.DELETION_WITNESS: str` — the upstream file that owns the flag,
    for the docstring.

- [ ] **Step 1** — write the failing test:

```python
def test_deletion_is_monotone_and_irreversible():
    st = _shell_state(n_layers=5)
    st.eps_p[0, 4] = 0.5
    m = johnson.Johnson(_props(eps_p_max=0.3))
    first = m.should_delete(st, _props(eps_p_max=0.3))
    st.eps_p[0, 4] = 0.0                     # unload completely
    assert m.should_delete(st, _props(eps_p_max=0.3))[0] >= first[0]

def test_one_pass_writes_off_and_nothing_else_does():
    model = _model_with_all_fail_models()
    failure.base.apply(model, MessageLog())
    assert "off" in model.groups["shells"].state
    assert model.groups["shells"].state["off"].dtype == np.int8

def test_deletion_is_all_or_nothing_per_element():
    """A shell with 4 damaged layers and 1 intact layer still deletes the
    element — or it does not; the criterion must be one value per element."""
    g = _shell_group(n_layers=5)
    g.state["eps_p"][0, :4] = 9.9
    assert _delete_flag(g).shape == (g.n,)
```

- [ ] **Step 2** — run → FAIL (no single `apply`, no shared base).
- [ ] **Step 3** — implement. **Two decisions must be read from the Fortran, not
  chosen:** (a) whether a partly-damaged multi-layer shell deletes as a whole
  element (upstream `/FAIL` on shells deletes the whole element — confirm in
  `failwave.F`/`tabulated.F`); (b) whether `off` is `int8` or `logical` and how
  it interacts with `GBUF%OFF` vs the contact `IDEL` flag. Getting (a) wrong
  means a shell that should tear keeps carrying load, which is the most
  consequential single-line error available in this phase.
- [ ] **Step 4** — run → PASS; commit
  `refactor(failure): one deletion convention with a pinned contract`.

---

### Task P7.1: Reclassify the six empty-upstream failure modules

**Fortran:** `$OR_SRC/engine/source/materials/fail/{changchang,composite,hashin,
lemaitre,puck,spalling}/` — **each 0 files**, verified 2026-10-02.
**Files:** Modify `pyradioss/failure/{chang.py, fail_composite.py, hashin.py,
lemaitre.py, puck.py, spalling.py}`; Create `docs/PORT_EXTENSIONS.md` entries;
Create `tests/test_p7_failure_provenance.py`.
**Interfaces:**
- Consumes: the same test shape as Phase 2 Task P7.1 (`test_every_cited_fortran_file_exists`).
- Produces: each module's docstring states its true provenance — a closed-source
  Radioss tree, a literature re-derivation, or a port invention — and a
  `PORT_EXTENSIONS.md` entry says which.

- [ ] **Step 1** — write the failing test — the generalised citation checker:

```python
def test_every_failing_citation_names_a_file_that_exists():
    import re, pathlib, paths
    bad = []
    for p in pathlib.Path("pyradioss").rglob("*.py"):
        for m in re.finditer(r"(?:fail|mat|elements)/[a-z0-9_]+/[a-z0-9_]+\.[Ff]90?", p.read_text()):
            for top in ("engine/source/materials", "engine/source/elements"):
                if (paths.or_src() / top / m.group(0)).is_file():
                    break
            else:
                bad.append((p.name, m.group(0)))
    assert bad == [], bad
```

- [ ] **Step 2** — run → FAIL, naming the six.
- [ ] **Step 3** — for each: search the whole tree for the routine
  (`grep -rl 'PUCK' $OR_SRC`), including `common_source/fail/` (104 LOC — check
  it) and the Starter's `starter/source/materials/fail/` (2,732 LOC — the
  readers may exist even when the engine kernel does not). Record what is
  actually found. **If nothing is found, the model is a port extension, its
  docstring must say so, and it must carry the three standard tests** (rigid →
  no damage, monotone damage, correct element flag) so it is at least
  self-consistent.
- [ ] **Step 4** — run → PASS; commit
  `fix(failure): reclassify six failure models with no upstream kernel`.

---

### Task P7.2: The failure-model census

**Fortran:** every `fail/*/` directory and `fail/*.F` at that level; the shell
vs solid split (which `/FAIL` models have a shell path); the readers
`starter/source/materials/fail/*`.
**Files:** Create `tools/validation_data/fail_status.json`,
`tests/test_p7_fail_census.py`.
**Interfaces:**
- Produces: one record per model: `{"model": "johnson_cook", "loc": 1093,
  "solid": true, "shell": true, "port": "failure/johnson.py",
  "status": "ported|partial|missing", "parity": {...} | null}` — and the
  shell-capable set, which **nothing currently enumerates**.

- [ ] **Step 1** — write the failing test:

```python
def test_shell_capable_failure_set_is_enumerated():
    st = json.loads(Path("tools/validation_data/fail_status.json").read_text())
    shell = {r["model"] for r in st if r["shell"]}
    assert shell, "no shell-capable failure model recorded"
    for m in shell:
        rec = next(r for r in st if r["model"] == m)
        assert rec["status"] in {"ported", "partial"}, m
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit. Read each `fail/*/` directory for a `*c` / shell
  variant; the shell support may live in `mulawc`-adjacent code rather than in
  the failure directory itself, so **check both** and record where each shell
  path actually lives.
- [ ] **Step 4** — run → PASS; commit `feat(census): failure-model status and shell-capable set`.

---

### Task P7.3: Port the missing failure models (one per task)

**Fortran:** the `fail/*/` directory of the model (see §Scope table).
**Files:** Create `pyradioss/failure/<model>.py`, `tests/test_p7_<model>.py`.
**Interfaces:**
- Consumes: the `FailureModel` protocol from Task P7.0.
- Produces: one registered model per task.

- [ ] **Step 1** — write the failing test, always the same three:

```python
def test_<model>_damage_is_monotone_and_bounded():
    m, props, st = _setup(_increasing_load(20))
    d = m.damage(st, props, None)
    assert np.all((d >= 0.0) & (d <= 1.0))
    assert np.all(np.diff(d) >= -1e-12)          # monotone

def test_<model>_does_not_delete_under_rigid_motion():
    m, props, st = _setup(_rigid_motion())
    assert not m.should_delete(st, props).any()

def test_<model>_deletes_at_its_calibration_point():
    m, props, st = _setup(_at_calibration_point())
    assert m.should_delete(st, props).all()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement from the model's own Fortran. **The third test is
  the calibration test and it is the strongest single check available**: a
  `/FAIL` criterion has documented calibration values, and reproducing them
  proves the port read the same table. Transcribe the calibration from the
  reader, do not tune it.
- [ ] **Step 4** — run → PASS; one oracle case where a deck exists; commit per
  model.

---

### Task P7.4: Equations of state

**Fortran:** `common_source/eos/` (4,898 LOC), the engine's
`materials/eos/eosmain.F`, and the Starter readers
`starter/source/materials/eos/` (5,021 LOC). EOS families the port's `docs/STATE.md`
records: LINEAR (M59, a 6-deck coverage blocker), IDEAL-GAS, NASG, OSBORNE,
PUFF, SESAME (M185), STIFF-GAS (M110, 4 decks), GRUNEISEN, TILLOTSON,
MURNAGHAN, LSZK, NOBLE-ABEL (M110), POWDER-BURN, COMPACTION, EXPONENTIAL,
IDEAL-GAS-VT (M166), JWL (M140), LEE-TARVER (M180/192).
**Files:** Create `pyradioss/materials/eos/` package
(`linear.py`, `ideal_gas.py`, `polynomials.py`, `gruneisen.py`, `sesame.py`,
`powder_burn.py`, `compaction.py`, `lee_tarver.py`, `base.py`),
`tests/test_p7_eos_*.py`.
**Interfaces:**
- Produces: `eos.EOS` protocol —
  `pressure(rho, e) -> np.ndarray`, `soundsp(rho, e) -> np.ndarray`,
  `energy(rho, p) -> np.ndarray`, `temperature(rho, e) -> np.ndarray` —
  consumed by the solid kernels' `/EOS` block and by the dt path.

- [ ] **Step 1** — write the failing test — a shared conformance suite run over
  **every** registered EOS:

```python
@pytest.mark.parametrize("name", EOS_NAMES)
def test_eos_is_monotone_in_density_at_fixed_energy(name):
    eos = eos.build(name)
    rho = np.linspace(0.5, 2.0, 50)
    p = eos.pressure(rho, np.full_like(rho, 1e6))
    assert np.all(np.diff(p) >= -1e-6 * np.abs(p).max())

@pytest.mark.parametrize("name", EOS_NAMES)
def test_eos_is_thermodynamically_consistent(name):
    """p = rho*dE/drho|E by the Maxwell construction — the identity every
    consistent EOS must satisfy numerically."""
    eos = eos.build(name)
    h = 1e-6
    dEdrho = (eos.energy((np.array([1.0 + h]), np.array([1e6]))
              - eos.energy((np.array([1.0 - h]), np.array([1e6])))) / (2*h)
    assert eos.pressure(np.array([1.0]), np.array([1e6]))[0] == pytest.approx(
        dEdrho[0], rel=2e-4)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, one module per family. **The Maxwell identity test
  is the workhorse**: an EOS whose `pressure` and `energy` are transcribed
  independently will fail it if upstream's are consistent, and pass only if the
  port got both right. Where upstream defines only one of the two (some tabulated
  EOS are pressure-only), the identity test is skipped for that family **with a
  citation**, not disabled.
- [ ] **Step 4** — run → PASS; **`/EOS/LINEAR` corpus coverage re-measured** and
  the six decks' verdict change reported; commit per family.

---

### Task P7.5: The shared viscoelastic driver

**Fortran:** `engine/source/materials/visc/{viscmain.F, prony_modelc.F,
visc_plas.F90, visc_prony.F, visc_prony_lstrain.F}`, plus
`mat_share/{mqvisc8.F, mqviscb.F, mqvisc26.F, mnsvis.F, nsvisul.F, fmqviscb.F}`
and the Starter's `starter/source/materials/visc/`.
**Files:** Create `pyradioss/materials/visco.py`, `tests/test_p7_visco.py`.
**Interfaces:**
- Produces: `visco.PronySeries(branches) -> PronySeries` with
  `.relaxation_modulus(t)`, `.internal_variables`, `.update(dstra, dt)`,
  `.viscous_stress()`, `.tangent()`; `visco.plastic_viscosity(model)` for
  `visc_plas.F90`.

- [ ] **Step 1** — write the failing test:

```python
def test_prony_relaxation_matches_the_analytical_laplace_transform():
    """A single Maxwell branch has G(t) = G_inf + G_0*exp(-t/tau)."""
    p = visco.PronySeries([(1.0, 1e-3)])          # (G_i, tau_i)
    assert p.relaxation_modulus(0.0) == pytest.approx(1.0)
    assert p.relaxation_modulus(1e-3) == pytest.approx(1.0 + np.exp(-1.0), rel=1e-6)

def test_single_branch_agrees_with_the_numerical_constitutive_response():
    from scipy.integrate import solve_ivp
    p = visco.PronySeries([(1.0, 1e-3)])
    eps = 0.01
    ref = solve_ivp(lambda t, y: (y[1], -y[1]/1e-3), [0, 1e-2],
                    [eps, p.update_state(0.0, eps)], rtol=1e-9, atol=1e-12).y[1]
    assert p.internal_stress() == pytest.approx(ref[-1], rel=1e-4)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The exact integration scheme matters** —
  `visc_prony.F` uses an exact/propagator integration while `prony_modelc.F`
  uses a Crank–Nicolson step; LAW35's port already notes "exact
  Crank–Nicolson MIDSTEP" (`PORTING_GUIDE.md:75`). Read which is which and do
  not unify them. Then **migrate the five per-law Prony implementations onto
  this driver**, one commit per law, and add a test asserting the shared and
  per-law paths agree bitwise.
- [ ] **Step 4** — run → PASS; commit per law for the migration, plus the driver.

---

### Task P7.6: Material tools — thermal, non-local, time step

**Fortran:** `engine/source/materials/mat_share/{thermc.F, thermexpc.F,
tempcg.F, meint.F, meos8.F, mdtsph.F, mreploc.F}`,
`starter/source/materials/nonlocal/` (196 LOC) +
`common_source/modules/nlocal_reg_mod.F90`,
`starter/source/materials/time_step/` (2,663 LOC),
`starter/source/materials/therm/` (575 LOC).
**Files:** Create `pyradioss/materials/nonlocal.py`,
`pyradioss/materials/time_step.py`, `tests/test_p7_material_tools.py`.
**Interfaces:**
- Produces: `nonlocal.regularise(field, length, kernel) -> np.ndarray` (the
  non-local length-scale operator for `/MAT/NONLOCAL` and `/FAIL/BROKMANN`),
  `time_step.sound_speed(model, group) -> np.ndarray` (the per-element
  critical speed feeding the dt kernel).

- [ ] **Step 1** — write the failing test:

```python
def test_nonlocal_regularisation_is_a_convolution_that_conserves_the_mean():
    f = np.random.default_rng(0).normal(size=(64, 64))
    g = nonlocal_.regularise(f, length=3.0, kernel="brokmann")
    assert g.mean() == pytest.approx(f.mean(), rel=1e-12)

def test_nonlocal_reduces_to_local_when_length_is_zero():
    f = np.random.default_rng(1).normal(size=(32, 32))
    assert np.allclose(nonlocal_.regularise(f, length=0.0), f)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `nlocal_reg_mod.F90` defines the **stored variable**
  set, not just the convolution — a non-local model needs its history in the
  element buffer, which is Phase 5's `ElBuf`. Declare that dependency
  explicitly; if `ElBuf` is not landed, the history lives in the element state
  dict and the migration is a note.
- [ ] **Step 4** — run → PASS; commit
  `feat(materials): non-local regularisation, thermal tools and the sound-speed path`.

---

### Task P7.7: XFEM failure models

**Fortran:** `/FAIL/XFEM/FLD`, `/FAIL/XFEM/JOHNS`, `/FAIL/XFEM/TBUTC` (M159) —
their readers under `starter/source/materials/fail/` and their kernels
(`fail/fld/` at 1,360 LOC is the *forming limit diagram* model, **not** the XFEM
one — do not confuse the two).
**Files:** Create `pyradioss/failure/xfem_models.py`,
`tests/test_p7_xfem_failure.py`.
**Interfaces:**
- Consumes: Phase 5 Task P5.7's `xfem_growth.advance`.
- Produces: three models that gate crack advance rather than element deletion.

- [ ] **Step 1** — write the failing test:

```python
def test_xfem_failure_gates_crack_advance_not_element_deletion():
    m = xfem_models.XFEMJohnson(_props(epsmax=0.2))
    crack = _straight_crack(_mesh(), rate=_constant_rate(1e-3))
    before = crack.tip.copy()
    m.advance(crack, dt=1e-5)
    assert np.allclose(crack.tip, before)   # criterion not met
    m.advance(crack, dt=1e-5, overstressed=True)
    assert np.linalg.norm(crack.tip - before) > 0.0
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Do not reuse `failure/fld.py`** — that is a
  different model (M86, forming limit diagram for shells). Read the three
  `/FAIL/XFEM/*` readers and their kernels by their own names.
- [ ] **Step 4** — run → PASS; commit `feat(failure): the three XFEM crack-advance models`.

---

### Task P7.8: `failwave` — failure-wave propagation

**Fortran:** `engine/source/materials/fail/failwave/` — `seg_intersect.F`,
`set_failwave_nod3.F`, `set_failwave_nod4.F`, `update_failwave.F`,
`upd_failwave_sh3n.F`, `upd_failwave_sh4n.F` — plus the three top-level
routines `fail_setoff_c.F`, `fail_setoff_npg_c.F`,
`fail_setoff_wind_frwave.F`.
**Files:** Create `pyradioss/failure/failwave.py` (replacing the current single
module in place, same public name), `tests/test_p7_failwave.py`.
**Interfaces:**
- Consumes: the `off` flag from Task P7.0.
- Produces: `failure.failwave.propagate(off, mesh, size) -> np.ndarray` (the
  per-node wave **level**, `MAXLEV_STACK`), `failure.failwave.overflow(nodes) ->
  list[int]` (the elements that exceeded `FAILWAVE%SIZE`),
  `failure.failwave.segment_intersects(mesh, a, b) -> bool`
  (`seg_intersect.F`).

- [ ] **Step 1** — write the failing tests:

```python
def test_failwave_propagates_one_level_from_a_failed_element():
    mesh = _quad_grid(n=5)
    off = np.zeros(len(mesh.elements), bool); off[12] = True
    lev = failure.failwave.propagate(off, mesh, size=3)
    assert lev.max() == 1
    assert np.all(lev[mesh.elements[12]] >= 1)     # neighbours are reached

def test_failwave_reports_overflow_rather_than_silently_capping():
    mesh = _quad_grid(n=9)
    off = np.zeros(len(mesh.elements), bool); off[40] = True
    over = failure.failwave.overflow(off, mesh, size=2)
    assert over, "an over-propagating wave must be reported, like failwave.F:242"
```

- [ ] **Step 2** — run → FAIL (or — if the existing `failure/failwave.py`
  already implements propagation — run the audit below first and only then the
  test).
- [ ] **Step 3** — implement. **First establish what the port's existing
  `failure/failwave.py` does**; if it is a damage-criterion model rather than a
  propagation model, it is misnamed and this task replaces it, and the commit
  says so explicitly. Read `set_failwave_nod3.F:236-246` for the overflow
  behaviour: upstream **aborts** with a written error, and the port must abort
  too — a capped wave is a silently wrong fracture path.
- [ ] **Step 4** — run → PASS; commit
  `refactor(failure): failwave as a propagation model with overflow abort (failwave/)`.

### Task P7.9: Registry and coverage re-measure

**Fortran:** the `/FAIL` dispatch, and the reader-side model names.
**Files:** Modify `pyradioss/failure/__init__.py`,
`pyradioss/input/keywords/materials.py`.
**Interfaces:**
- Produces: `failure.REGISTRY: dict[str, type[FailureModel]]` keyed by the
  `/FAIL` sub-card name (`JOHNSON`, `BIQUAD`, `TAB1`, `PUCK`, `BROKMANN`,
  `XFEM/JOHNS`, …); `failure.build(name, props) -> FailureModel`.

- [ ] **Step 1** — write the failing test — exhaustive over the reader set:

```python
def test_every_fail_subcard_is_registered_or_cited_refused():
    import yaml
    names = {r["name"] for r in yaml.safe_load(
        Path("tools/validation_data/fail_status.json").read_text())}
    known = set(failure.REGISTRY) | set(failure.refused())
    assert names <= known, sorted(names - known)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Same rationale as Phases 2 and 4: convert every
  silent skip into an explicit routing decision with a citation. This is the
  task that turns the corpus's "SKIPS" into "CLEAN or refused".
- [ ] **Step 4** — run → PASS; coverage re-measured; commit
  `feat(failure): exhaustive /FAIL registry with cited refusals`.

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P7.0** — the two Fortran decisions (whole-element shell deletion; the flag
  type and its `IDEL` interaction) are quoted from the source.
- **P7.1** — each of the six modules states its true provenance; none still
  claims an upstream file that does not exist.
- **P7.2** — the shell-capable failure set is enumerated from the source, not
  assumed.
- **P7.3** — every model has all three tests, and the calibration value is
  transcribed from the reader rather than tuned to pass.
- **P7.4** — the Maxwell identity is the spec; where upstream is pressure-only,
  the skip carries a citation.
- **P7.5** — the shared and per-law Prony paths agree bitwise; the two upstream
  integration schemes are **not** unified.
- **P7.7** — `failure/fld.py` was not reused for the XFEM models.
- **P7.8** — the overflow **aborts** as `set_failwave_nod3.F:242` does; a
  capped wave is a rejection.

## Parallelisation

- **Wave 0** — P7.0 serial (it defines the contract). P7.1 and P7.2 are
  independent of it and run in parallel with it.
- **Wave 1** — the highest concurrency in the program after Phase 6: ~30 failure
  models + 16 EOS families + 3 shared-machinery tasks, all new files. **Each
  agent creates exactly one `pyradioss/failure/<model>.py` or
  `pyradioss/materials/eos/<eos>.py` and one test file, and edits nothing else.**
- **Wave 2** — serial.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p7_deletion.py tests/test_p7_fail_census.py
python tools/validate_vs_fortran.py --mode coverage --out tools/validation_data/coverage_p7.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area engine/source/materials/fail
```

The phase reviewer reports: the number of failure models live vs the 44 upstream
directories; the `/EOS/LINEAR` coverage delta (6 decks at baseline); and that
the energy-balance tests are green, since `/FAIL` deletion changes which
elements book work and a leak here is silent.