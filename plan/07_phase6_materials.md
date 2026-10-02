# Phase 6 — Materials

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/materials/mat/**` and `mat_share/**` a literal
port — every one of the 121 upstream law directories present, every law's
**3-D and shell** kernels where upstream has them, every documented cut closed
or formally refused.

**Architecture:** the phase is organised around three upstream artefacts that
are the port's real specification, and which no amount of reading prose
substitutes for:

1. `mat_share/mmulaw.F90` — the **3-D** constitutive dispatch.
2. `mat_share/mmulawc.F90` — the **shell** constitutive dispatch, which names
   **51 laws with a shell kernel**.
3. `mat_share/mmmain.F90` / `mmain.F90` — the entry point and the per-element
   state machine.

**Tech stack:** NumPy; `pyradioss/materials/` is the largest package after
`input/` (119,971 LOC across 142 modules).
**Spec:** `plan/00_ORCHESTRATION.md` §1.1, §10 item 1 (inactive material
refusal).
**Entry criteria:** Phase 1 exit gate; Phase 0 oracle. **Phase 2's
`init_group`/`forces` contract and Phase 3's material hook (`shell_moduli`) are
frozen for the duration of this phase** — this phase extends them only by
declared signature.
**Band: XL.**

---

## Scope

### Upstream law directories (121), with file count and LOC

```
001|13|2318  002|9|3197  003|2|535   004|1|245   005|2|375   006|1|147
010|1|220   011|3|1078  012|1|680   013|5|790   014|4|997   015|3|611
016|5|768   017|1|180   018|2|223   019|1|180   021|1|270   022|4|1211
024|18|3770 025|11|5166 026|6|696   027|4|717   028|1|338   032|3|672
033|1|2922 034|4|589   035|2|668   036|5|4258  037|1|397   038|1|1207
040|1|538   041|2|825   042|2|936   043|2|937   044|5|1693  045|2|739
046|2|489   048|2|857   049|1|212   050|1|407   051|12|5708 052|2|2260
053|1|415   055|1|463   056|3|1777  057|1|623   058|1|1774  059|1|269
060|3|2522  062|2|1097  063|1|452   064|1|458   065|2|1244  066|2|1532
068|1|474   069|2|610   070|1|713   071|3|1362  072|2|773   073|1|739
074|1|956   075|1|494   076|7|2911  077|1|631   078|2|1292  079|1|391
080|4|2740  081|1|803   082|2|649   083|1|468   084|1|469   085|1|169
086|2|1328  087|5|3561  088|2|2268  090|1|841   091|1|264   092|1|324
093|2|969   094|1|308   095|1|548   096|1|517   097|1|422   100|8|1896
101|1|2195  102|1|163   103|1|242   104|14|7824 105|1|661   106|2|1050
107|6|2534  109|2|892   110|5|4600  111|1|378   112|6|3456  115|3|927
116|1|385   117|1|452   119|2|551   120|10|3740 121|6|2286  122|6|2980
123|6|2194  124|1|981   125|4|1675  126|1|533   127|2|1033  128|2|828
129|1|579   130|1|1277  131|16|1567 132|2|925   133|1|210   134|1|190
136|1|857   137|2|1382  158|1|460   163|1|311   169|1|290   187|1|1021
190|3|1595
```

`engine/source/materials/mat/` = **≈ 110,000 LOC**. Plus `mat_share` (24 files,
≈ 8,000), `tools` (14 files, 6,714), `visc` (5 files, 1,655), `eos` (Phase 7),
`fail` (Phase 7).

### The three dispatch artefacts

`mat_share/mmulawc.F90` names **51 laws with a shell kernel** (transcribed from
its 141 `sigeps` references):

```
01 02 15 19 22 25 27 32 34 35 36 42 43 44 45 48 52 55 56 57 58 60 62 63 64
65 66 69 71 72 73 76 78 80 82 85 86 87 88 93 104 106 107 109 110 112 119 121
122 123 125 127 128 131 132 137 158
```

`mat_share/` also holds the shared machinery: `mmain.F90`, `mmain8.F`,
`mulaw.F90`, `mulaw8.F90`, `mulawc.F90`, `mulawglc.F`, `cmain3.F`,
`mqvisc8.F`/`mqviscb.F`/`mqvisc26.F` (viscosity), `mnsvis.F`, `nsvisul.F`,
`meint.F`, `meos8.F`, `mdtsph.F`, `mreploc.F`, `mrotens.F`, `mstrain_rate.F`,
`rotos4.F`, `tempcg.F`, `thermc.F`, `thermexpc.F`, `fmqviscb.F`,
`jacobview_v.F`, `usermat_shell.F`, `usermat_solid.F`.

### Port state today

- `pyradioss/materials/` — 142 modules, 119,971 LOC.
- `MAT_PHYSICS_REGISTRY` — **571 keys**, of which **59 are numeric law numbers**:
  `5,10,12,14,15,21,22,24,25,28,32,34,37,38,43,48,49,50,52,57,58,60,66,69,71,73,
  74,76,79,82,87,88,92,93,94,95,100,101,102,103,104,105,106,107,109,110,114,117,
  119,120,121,123,124,126,132,163,169,187,190`.
- `docs/STATE.md` §"What the port can do today" (dated M41) claims ~16 laws
  carry real physics; `docs/STATE.md` §"Material Law & Failure Model Porting
  Session" (2026-09-22) claims ~36 more were ported in a batch.
- `pyradioss/input/mat_reader.py` (1,399 lines) resolves all 204 law spellings
  from the CFG tree, producing either a real material or an `InactiveMaterial`,
  and `refuse_inactive_materials()` refuses the latter at run time.

## Gap analysis

**The gap is not "which laws parse" — it is which laws have *physics*, and in
which element context.**

1. **51 laws have an upstream shell kernel; the port's shell-capable set is a
   small fraction of that.** `mulawc.F90` is the checklist, and
   `tests/test_mat_all_135_census.py`'s `_SOLID_ONLY_LAWS` is the port's
   *opposite* assertion for a handful of laws. Those two must be reconciled
   against `mulawc.F90`, which is the authority.
2. **51 upstream law directories have no registered physics at all.** These are
   the numeric IDs present in `$OR_SRC/engine/source/materials/mat/` and absent
   from both `MAT_PHYSICS_REGISTRY`'s 59 numeric keys and the 12 hand-registered
   base laws (`0/1/2/19/27/35/36/40/42/44/62/70/81`, i.e. VOID, LAW1, LAW2,
   LAW19, LAW27, LAW35, LAW36, LAW40, LAW42, LAW44, LAW62, LAW70, LAW81):

   ```
   3  4  6 11 13 16 17 18 26 33 41 45 46 51 53 55 56 59 63 64 65 68 72 75 77
   78 80 83 84 85 86 90 91 96 97 111 112 115 116 122 125 127 128 129 130 131
   133 134 136 137 158
   ```

   **Each is a task.** P6.0 re-derives this list from the tree rather than
   trusting this paragraph; if they disagree, the census wins.
3. **The batch of 2026-09-22 (`docs/STATE.md`) has no parity evidence.** It
   landed ~36 laws in one session with no `VALIDATION.md` entry. Under the
   domain rule "numerics changes require validation evidence", every one of
   them is now **unproven** and must be re-verified against its `sigeps##.F`
   before it can be called ported. This is Task P6.1 and it is the most
   important task in the phase.
4. **A 142-module package with a 5,401-line `__init__.py` registry.** The
   registry is contended (every law task touches it).
5. **Hyperelastic/viscoelastic laws return "their own SOUNDSP"** (LAW42 per
   `README.md`), which feeds the stable time step — so a wrong sound speed is a
   dt error, which is a *stability* error, not a small accuracy error.
6. **User materials** `usermat_solid.F` / `usermat_shell.F` and the
   `ENG_USERLIB` hook are unported (Phase 4 Task P4.2 touches the spring side).

## Wave graph

```
Wave 0  P6.0 law census vs mulaw.F90/mulawc.F90   (serial — the checklist)
        P6.1 re-verify the 2026-09-22 batch against its sigeps (per-law agents)
   ── gate: every law has a status; every "ported" has a parity number or a
              citation for why none is possible
Wave 1 (parallel — one law per task, banded by measured upstream LOC)
  Band A (<= 400 LOC,   11 laws)   11 tasks, 11 agents
  Band B (400-1200,     27 laws)   27 tasks, ~7 agents x 4 laws
  Band C (1200-3200,    11 laws)   11 tasks, ~4 agents x 3 laws
  Band D (> 3200,        2 laws)    2 tasks, 1 agent each, started FIRST
   ── gate per law: limiting-case test + tangent FD test + one oracle case

Band membership, computed from the §Scope table (2026-10-02):
  A: 4(245) 6(147) 17(180) 18(223) 59(269) 85(169) 91(264) 111(378)
     116(385) 133(210) 134(190)
  B: 3(535) 11(1078) 13(790) 16(768) 26(696) 41(825) 45(739) 46(489)
     53(415) 55(463) 63(452) 64(458) 68(474) 72(775) 75(494) 77(631)
     83(468) 84(469) 90(841) 96(517) 97(422) 115(927) 127(1033) 128(828)
     129(579) 136(857) 158(460)
  C: 33(2922) 56(1777) 65(1244) 78(1292) 80(2740) 86(1328) 122(2984)
     125(1675) 130(1277) 131(1567) 137(1382)
  D: 51(5708) 112(3456)
Wave 2 (serial)
  P6.3 mat_share shared machinery (mmain/mulaw*/strain-rate/rotation/thermal)
  P6.4 shell-kernel coverage: every mulawc.F90 law live or refused
  P6.5 registry + refusal gates + SOUNDSP plumbing
  P6.6 numba mirrors for the hot leaves (mmain's inner loop)
```

---

### Task P6.0: The law census — build the checklist

**Fortran:** `engine/source/materials/mat_share/mmulaw.F90` (3-D dispatch),
`mmulawc.F90` (shell dispatch, 51 `sigeps##c`), `mmulaw8.F90`, `mmulawglc.F`,
`mmain.F90`, `mmain8.F`; every `mat*/` directory; the Starter readers
`starter/source/materials/mat/mat###/hm_read_mat##.F`.
**Files:** Create `tools/validation_data/law_status.json`,
`tests/test_p6_law_census.py`.
**Interfaces:**
- Produces: one record per upstream law:
  `{"law": 104, "dir": "engine/source/materials/mat/mat104", "loc": 7824,
    "has_solid_kernel": true, "has_shell_kernel": true,
    "port_module": "pyradioss/materials/law104_drucker.py" | null,
    "solid": "ported|partial|missing|refused",
    "shell": "ported|partial|missing|refused|not-applicable",
    "parity": {"case": "RD-E-...", "rel_rms": 0.03} | null,
    "cuts": ["..."]}`
  plus `"shell_laws"` — the exact 51-ID list read out of `mulawc.F90`.

- [ ] **Step 1** — write the failing test:

```python
def test_shell_law_list_matches_mulawc_exactly():
    """mulawc.F90 names 51 shell kernels; the port must not claim a
    different set."""
    st = json.loads(Path("tools/validation_data/law_status.json").read_text())
    upstream = set(st["shell_laws"])
    assert len(upstream) == 51
    port_claims = {r["law"] for r in st if r["shell"] == "ported"}
    assert port_claims <= upstream, sorted(port_claims - upstream)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — build the table by reading `mulaw.F90`, `mulawc.F90` and the
  121 directories. **`mulawc.F90` is the authority on which laws have a shell
  kernel** — do not infer it from the law's `/MAT` subtype, and do not trust
  `tests/test_mat_all_135_census.py`'s `_SOLID_ONLY_LAWS` where they disagree.
- [ ] **Step 4** — run → PASS; commit `feat(census): law status table from mulaw/mulawc`.

---

### Task P6.1: Re-verify the 2026-09-22 batch

**Fortran:** each law's own `sigeps##.F` / `sigeps##.F90` and its
`hm_read_mat##.F`.
**Files:** Modify the corresponding `pyradioss/materials/law*.py`;
Create `tests/test_p6_batch_verify.py`.
**Interfaces:**
- Consumes: `law_status.json`.
- Produces: for each law in the 2026-09-22 batch, a `parity` entry or a
  `corrections` list. **This task is expected to find bugs** — that is its
  purpose, and a batch that produces zero findings is itself a finding (it means
  the verification was shallow).

- [ ] **Step 1** — write the standard per-law verification test. The shape is
  fixed for every law and is the deliverable that makes this task mechanical:

```python
def _verify_law(module, law, is_shell):
    """Three checks every constitutive law must pass."""
    # 1. limiting case: rigid -> zero stress
    assert _rigid_strain_stress(module, is_shell) == approx(0, abs=1e-9)
    # 2. tangent consistency: d(stress)/d(strain) == the module's tangent
    assert _fd_vs_tangent(module, is_shell, tol=2e-4)
    # 3. sound speed positive and finite where the law defines one
    assert _soundsp(module) > 0.0
```

- [ ] **Step 2** — run → the first law fails, showing how far the batch is from
  its own claims. **Quote that failure count in the task's report; do not tune
  the tolerances to make it zero.**
- [ ] **Step 3** — one agent per law. Read the law's `sigeps##.F`; correct the
  port; record the parity case. Laws whose Fortran needs a reader the port does
  not have get a reader in the same task — a kernel with no reachable deck is
  not a port.
- [ ] **Step 4** — run → PASS per law; commit per law.

---

### Task P6.2: The 51 laws with no registered physics (Band A–D)

**Fortran:** each law's own directory (§Scope). A law task is one task; there
are 51; they are grouped by size band in §Wave 1 so that a reviewer rejects one
law without rejecting the band.
**Files:** Create `pyradioss/materials/law<NNN>_<name>.py` per law; Create
`tests/test_p6_law<NNN>.py` per law. **Registration into
`pyradioss/materials/__init__.py` is the phase lead's job at the end of the
wave — see Parallelisation.**
**Interfaces:**
- Consumes: the law contract, which this task **fixes once and for all**:

```python
class Law(Protocol):
    law_id: int
    def solid_update(self, state: ElementState, props: MatProps,
                     dstra: np.ndarray, dt: float) -> None: ...
    def shell_update(self, state: ElementState, props: MatProps,
                     dstra: np.ndarray, dt: float) -> None: ...   # or NotImplementedError
    def tangent(self, props: MatProps, dstra: np.ndarray) -> np.ndarray: ...
    def soundsp(self, props: MatProps, rho0: float) -> float: ...
```

  `shell_update` must **raise `NotImplementedError` with a citation** when the
  law has no upstream shell kernel — the pattern already established by
  `law14_compso.py` (`tests/test_m547_law14_compso.py::test_shell_update_not_implemented`).
- Produces: one registered entry per law, plus a `law_status.json` update.

- [ ] **Step 1** — write the failing test, per law:

```python
def test_law104_drucker_solid_limit_case():
    m = law104_drucker.Law104()
    st = _one_gauss_point_state()
    m.solid_update(st, _props_104(), np.zeros((1, 6)), dt=1e-6)
    assert np.allclose(st.sigma[0], 0.0, atol=1e-9)

def test_law104_has_no_shell_kernel():
    with pytest.raises(NotImplementedError, match="mulawc.F90"):
        law104_drucker.Law104().shell_update(...)
```

  The second test's `match` is the citation — **it is what stops a "port" from
  being a shell stub that silently returns zeros**, the exact failure
  `docs/OPEN_BUGS.md` item 4 records.
- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Read the law's `sigeps##.F` in full. **For a
  return-mapping law, port the actual algorithm** — the correct yield surface
  with a projected return is *not* equivalent to a radial return, and the
  tangent test is what distinguishes them.
- [ ] **Step 4** — run → PASS; one oracle parity case per law where a deck
  exists; commit per law.

---

### Task P6.3: `mat_share` — the shared constitutive machinery

**Fortran:** `mat_share/mmain.F90`, `mmain8.F`, `mulaw.F90`, `mulaw8.F90`,
`mmulawc.F90`, `mmulawglc.F`, `cmain3.F`, `mstrain_rate.F`, `mrotens.F`,
`rotos4.F`, `tempcg.F`, `thermc.F`, `thermexpc.F`, `meint.F`, `meos8.F`,
`mreploc.F`, `mdtsph.F`, `usermat_solid.F`, `usermat_shell.F`;
`materials/tools/{kmatinv.F, prodAAT.F, prodATA.F, prodmat.F, read_mat_table.F,
roto_tens2d.F, roto_tens2d_aniso.F, uroto_tens2d.F, uroto_tens2d_aniso.F,
table_mat_vinterp.F, table_mat_vinterp_c1.F90, table_mat_vinterp_inv.F90,
table_rresti_mat.F, write_mat_table.F}`.
**Files:** Create `pyradioss/materials/share/` package
(`mmain.py`, `mulaw.py`, `strain_rate.py`, `rotation.py`, `thermal.py`,
`table.py`, `tools.py`), Modify `pyradioss/materials/__init__.py`.
**Interfaces:**
- Produces:
  - `share.mmain.update_solid(group, model, dt) -> np.ndarray` (the per-element
    3-D driver; the analogue of the port's current `materials.solid_update`);
  - `share.mmulaw.update_shell(group, model, dt) -> np.ndarray`;
  - `share.strain_rate.rate(props, laws, tensor) -> np.ndarray`
    (`mstrain_rate.F` — the rate-measure selection VP=1/2/3 the port's LAW44
    already implements *per law*; this is the shared one);
  - `share.rotation.rotate_to_local(tensor, frame) -> np.ndarray`
    (`mrotens.F`, `rotos4.F`);
  - `share.thermal.{expansion, conduction}` (`thermc.F`, `thermexpc.F`,
    `tempcg.F`);
  - `share.table.{interp, interp_inverse, table_to_matrix}` (`table_mat_vinterp*`,
    `kmatinv.F`, `prodAAT/ATA/prodmat.F`).

- [ ] **Step 1** — write the failing tests:

```python
def test_strain_rate_measures_differ_and_are_labelled():
    L = np.diag([1e8, -4e7, -6e7])
    for vp in (1, 2, 3):
        assert share.strain_rate.rate(_props(vp=vp), None, L) is not None
    assert (share.strain_rate.rate(_props(vp=1), None, L)
            != share.strain_rate.rate(_props(vp=2), None, L))

def test_rotos4_is_orthogonal_and_determinant_plus_one():
    for th in np.linspace(0, np.pi, 9):
        R = share.rotation.rotos4(th)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

def test_table_interp_inverse_round_trips():
    t = np.array([[0.0, 0.0], [1.0, 100.0], [2.0, 60.0]])
    x = 1.37
    assert share.table.interp_inverse(t)(share.table.interp(t)(x)) == pytest.approx(x)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **`table_mat_vinterp_inv.F90` is the subtle one**:
  the inverse interpolation (a *strain → curve-parameter* inversion used by
  tabulated laws) is what makes a tabulated curve differentiable, and getting
  the monotonicity handling wrong silently extrapolates. `mstrain_rate.F`'s VP
  selection is the other: LAW44's per-law implementation and this shared one
  must agree **bitwise**, or the same law behaves differently depending on
  which call site runs it — a test asserts they agree.
- [ ] **Step 4** — run → PASS; commit `feat(materials): mat_share shared constitutive machinery`.

---

### Task P6.4: Shell-kernel coverage for the 51 `mulawc.F90` laws

**Fortran:** `mulawc.F90`'s 51 `sigeps##c.F` files.
**Files:** Create `pyradioss/materials/law<NNN>_shell.py` (or extend the law
module), Modify `pyradioss/materials/shell_update` plumbing.
**Interfaces:**
- Produces: for each of the 51, `shell_update` that either works or raises with
  a citation. `law_status.json`'s `shell` field becomes `ported` or `refused`.

- [ ] **Step 1** — write the failing test — exhaustiveness again:

```python
@pytest.mark.parametrize("law", MULAWC_SHELL_LAWS)   # the exact 51
def test_every_mulawc_law_has_a_shell_kernel_or_a_cited_refusal(law):
    m = materials.build(law)
    try:
        m.shell_update(_shell_state(), _props(law), np.zeros((1, 8)), 1e-6)
    except NotImplementedError as e:
        assert "mulawc.F90" in str(e) or "no upstream" in str(e)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Where a law's shell kernel is a thin
  plane-stress reduction of its 3-D kernel, that reduction is a real piece of
  physics and must be read from `sigeps##c.F`, not derived by assuming
  `σ₃=0`.** The four QBAT/QEPH/BT laws in `PORTING_GUIDE.md:61-62` are gated to
  `{0,1,2,27,36,44}` and that gate is a real restriction to lift here.
- [ ] **Step 4** — run → PASS; commit per law.

---

### Task P6.5: Registry, refusal gates and SOUNDSP plumbing

**Fortran:** `starter/source/materials/mat/mat000/hm_read_mat00.F` (the VOID
law, no engine kernel — "that is the point", `PORTING_GUIDE.md:73`), the
`refuse_inactive` pattern, and each law's `SOUNDSP` return feeding
`engine/source/time_step/`.
**Files:** Modify `pyradioss/materials/__init__.py` (5,401 lines),
`pyradioss/input/mat_reader.py`, `pyradioss/input/checks.py`,
`pyradioss/engine/mass_scaling.py` (the dt path).
**Interfaces:**
- Produces:
  - `materials.REGISTRY: dict[int, type[Law]]`;
  - `materials.MAT_PHYSICS_REGISTRY` unchanged in shape (Wave 1 fills it);
  - `materials.soundsp(group) -> np.ndarray` (per-element critical wave speed,
    consumed by the dt kernel — the contract LAW42/LAW62/LAW70 already return);
  - `checks.refuse_inactive_materials(model, log)` extended to name the element
    family too, not just the law (Review Focus item 1).

- [ ] **Step 1** — write the failing test:

```python
def test_refusal_names_the_law_and_the_family():
    model = _model_with_part(lawname="/MAT/LAW104/DRUCKER", pname="P_SHELL")
    with pytest.raises(InactiveMaterialError) as e:
        checks.refuse_inactive_materials(model)
    msg = str(e.value)
    assert "LAW104" in msg and "SHELL" in msg
```

- [ ] **Step 2** — run → FAIL (the refusal names the law, not the family).
- [ ] **Step 3** — implement. The **SOUNDSP** half is the one with teeth: a
  hyperelastic law whose sound speed is computed from the *current* tangent
  rather than the initial one changes the stable dt mid-run, and the port must
  do whichever the Fortran does. Read each law's SOUNDSP block; do not assume
  "sound speed from E0".
- [ ] **Step 4** — run → PASS; **the `PYRADIOSS_BACKEND=numpy` T01 md5 for the
  bundled examples must be unchanged** — this is a registry/plumbing task and
  must not move a single cycle; commit
  `feat(materials): registry, named refusals and the SOUNDSP dt path`.

---

### Task P6.6: numba mirrors for the constitutive inner loop

**Fortran:** none. **Files:** Create `pyradioss/accel/jit_kernels/materials.py`.
**Interfaces:**
- Consumes: `Law.solid_update` / `shell_update` leaves.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("law", [1, 2, 36, 44, 70, 104])
def test_material_leaf_is_bitwise_under_numba(law):
    ref = accel.reference(f"mat_leaf_{law}")
    assert accel.get(f"mat_leaf_{law}")(*ref.inputs).tobytes() == ref.expected.tobytes()
```

- [ ] **Step 2** — run → SKIP until P0.7, then FAIL.
- [ ] **Step 3** — implement **only the leaves that pay**. Return-mapping laws
  with table lookups are usually *not* worth mirroring — the gather dominates
  and NumPy's `take` is already good. Record a measured decision per law; where
  the mirror does not clear the speed bar, exclude that law from `auto` and
  write the measurement down. The existing `law70_tab2d`/`snorm`/`enorm`/
  `elastic_stress` mirrors (`PORTING_GUIDE.md:99`) are the precedent.
- [ ] **Step 4** — run → PASS; T01 md5 unchanged; commit per law.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P6.0** — `mulawc.F90` was read, not inferred; the port does not claim a
  shell kernel `mulawc.F90` does not name.
- **P6.1** — the batch verification **found something**, or the report explains
  why zero findings is itself the finding. Tolerances were not widened.
- **P6.2** — every law has the rigid → zero-stress limiting test, an
  FD-consistent tangent, and a `NotImplementedError`-with-citation shell refusal
  where applicable. A law whose shell update returns zeros is a hard rejection.
- **P6.3** — the per-law and shared strain-rate paths agree bitwise; the inverse
  table interpolation handles non-monotone curves as `table_mat_vinterp_inv.F90`
  does, not by assuming monotonicity.
- **P6.4** — the plane-stress reductions are transcribed from `sigeps##c.F`.
- **P6.5** — the refusal names law **and** family; the SOUNDSP convention is
  read per law; the bundled examples' T01 md5 is unchanged.
- **P6.6** — measured decision per law; mirror or exclusion, never neither.

## Parallelisation

- **Wave 0** — P6.0 serial (it is the checklist). P6.1 fans out to **one agent
  per law** across the ~36 batch laws — this is the highest-concurrency task in
  the whole program.
- **Wave 1** — 51 law tasks, banded as tabled above. **Band D (laws 51 and 112)
  gets one agent each and starts first**, because they dominate the phase's
  critical path.
- **Wave 2** — serial, contended.

**File ownership in Wave 1:** each agent creates exactly one
`pyradioss/materials/law<NNN>_*.py` and one `tests/test_p6_law<NNN>.py`. **No
agent edits `pyradioss/materials/__init__.py` during Wave 1** — the phase lead
collects the registrations in one commit at the end of the wave, after
`git diff --stat` shows only additions across the 51 files. This is the single
most important ownership rule in the phase.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p6_law_census.py
python -m pytest -q tests/test_mat_all_135_census.py
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/validate_vs_fortran.py --mode coverage --out tools/validation_data/coverage_p6.json
python tools/census.py --render plan/CENSUS.md --area engine/source/materials/mat
```

The phase reviewer reports, as numbers: how many of the 121 law directories are
`ported` / `partial` / `refused` / `missing`; how many of the 51 `mulawc.F90`
shell kernels are live; and the corpus coverage delta from
`coverage_results_m41.json` (13 CLEAN / 440 SKIPS / 76 ERROR at baseline).
**A law marked `ported` with no parity number and no citation is the rejection
condition for the whole phase.**