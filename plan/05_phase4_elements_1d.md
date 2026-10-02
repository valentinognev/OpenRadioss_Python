# Phase 4 — Elements: 1-D (truss, beam, spring, joint, rivet, xelem)

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/elements/{truss,beam,spring,joint,rivet,xelem}` a
literal port. This is the smallest element phase by line count and the one with
the **largest dispatch surface** — `spring/rforc3.F` alone branches over 16
distinct `IGTYP` values with per-branch kernels, force laws and user hooks.

**Architecture:** one kernel module per `IGTYP` branch (so a branch is a
reviewable unit), one shared 1-D driver, and an **exhaustive `IGTYP` coverage
test** modelled on the one Phase 2 Task P2.11 builds for `Idsolid`.

**Tech stack:** NumPy, numba where the kernel is vectorised over groups.
**Spec:** `rforc3.F`'s `IGTYP` ladder (`rforc3.F:243,321,387,452,558,626,749,
862,1061,1160,1185,1293,1391`).
**Entry criteria:** Phase 1 exit gate; Phase 0 oracle.
**Band: L.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `engine/source/elements/truss` | 11 | 1,389 | `truss.py` (445) |
| `engine/source/elements/beam` (type 3 + type 18) | 25 | 4,455 | `beam_type3.py` (1,204), `beam_fiber.py` (1,015) |
| `engine/source/elements/spring` | 78 | 22,747 | `spring.py` (598), `spring_advanced.py` (3,429), `spring_mat.py` (869), `spring_beam.py` (764), `spring_general.py` (321), `spring_pretensioner.py` (537) |
| `engine/source/elements/joint` | 10 | 2,643 | `engine/gjoint.py`, `engine/kjoint.py`, `engine/lagmul.py` |
| `engine/source/elements/rivet` | 1 | 420 | `rivet.py` (502) |
| `engine/source/elements/xelem` | 10 | 1,941 | `nstrand.py` (622) |
| `starter/source/elements/{truss,beam,spring,joint,xelem}` | ~50 | ~10,400 | `input/keywords/elements.py`, `starter/initialization.py` |
| **Total** | **185** | **43,997** | **~10,700** |

## Gap analysis

### The authoritative `IGTYP` ladder

`engine/source/elements/spring/rforc3.F` dispatches on `IGTYP = IGEO(11,I0)`
(a `/PROP` type number). Read off the file, this is the complete branch list:

| `IGTYP` | Branch | Upstream kernel | Port status |
|---:|---|---|---|
| 1,2,3 | axial / generic / torsional | `r1def3.F`, `r2def3.F`, `r3def3.F` | `spring.py` |
| 4 | — | `r4def3.F` | `spring.py` |
| 5 | — | `r5def3.F` (also reused for 32) | partial |
| 6 | — | `r6def3.F` (calls `redef3.F90`, `repla3.F`) | **gap** |
| 8 | `SPR_GENE` 6-DOF | `r8ke3.F`, `r8sumg3.F` | `spring_general.py` — **documented cuts** (`PORTING_GUIDE.md:67`: fct_IDji force functions, `Hi/IECROU` hardening, rupture, rate smoothing, sensor activation, `skew_ID`, TYPE13 co-rotational frame) |
| 12 | `SPR_PUL` pulley | `r12ke3.F`, `r12mat3.F`, `r12sumg3.F` | `spring_advanced.py` |
| 13 | `SPR_BEAM` 6-DOF | `r13ke3.F`, `r13mat3.F`, `r13sumg3.F` | `spring_beam.py` — documented cut (implicit path TYPE4-only) |
| 19 | `SPR_TORS` | `r3tors.F` family | `spring_advanced.py` |
| 23 | `SPR_MAT` | `r23*` (11 files) + `r23law108/113/114/135` | `spring_mat.py` — needs audit |
| 25 | `SPR_AXI` | `r25*` (via `r5def3.F` calls at 1410) | `spring_advanced.py` — needs audit |
| 26 | `SPR_TAB` | `r26def3.F`, `r26sig.F` (**uses `python_funct_mod`**) | `spring_advanced.py` |
| 27 | `SPR_BDAMP` | `r27def3.F` (**calls `finter`**) | `spring_advanced.py` |
| 29,30,31 | user springs | `ENG_USERLIB_RUSER` (`common_source`) | **gap** |
| 32 | `SPR_PRE` pretensioner | `r5def3.F` + `ruser32*.F` | `spring_pretensioner.py` — `InactiveProperty` per `VALIDATION.md` §7 item 11 ("assessed tractable, left InactiveProperty, c52 stays SKIPS") |
| 33, 45 | joints | `rgjoint.F`, `joint_block_stiffness.F` | `engine/gjoint.py`, `kjoint.py` |
| 35, 36 | `STITCH` / `PREDIT` user materials | `ruser35.F`, `ruser36.F` | `spring_advanced.py` |
| 44 | `SPR_CRUS` crushing | `r5def3.F` branch at 1312 | `spring_advanced.py` |
| 46 | `SPR_MUSCLE` | `r5def3.F` branch at 1410 | `spring_advanced.py` |

### The gaps that matter

1. **`IGTYP` 6, 29, 30, 31** — TYPE6 has no kernel; the three user-spring types
   have no dispatch at all.
2. **Every `r2xdef3.F` branch that `uses python_funct_mod`** — TYPE26 and
   TYPE27 call back into Python. The port has `/FUNCT_PYTHON` (M119) but the
   spring-side `python_funct_mod` integration is not wired, so those two
   branches are silently curve-only.
3. **`spring.py` is 598 lines against 22,747 Fortran lines**, and the cut list at
   `PORTING_GUIDE.md:67` names **eight** unported features of TYPE8 alone. The
   1:38 ratio is the same warning sign Phase 2 found for solids.
4. **TYPE32 pretensioner physics** is `InactiveProperty` (`VALIDATION.md`
   §7 item 11). The Fortran is `ruser32.F` + `ruser32ke3.F` + `ruser32mat3.F` +
   `preload_axial.F90` — tractable, and the port's own audit says so.
5. **`/PROP/TYPE33` has 10 reader variants upstream**
   (`hm_read_prop33_cyl_jnt.F`, `_fix_jnt`, `_free_jnt`, `_old_jnt`, `_plan_jnt`,
   `_rev_jnt`, `_sph_jnt`, `_trans_jnt`, `_univ_jnt`, plus `_cyl_joint.py` and
   `_cyl_joint` in the port). **Nine joint topologies**; the port has
   `engine/cyl_joint.py`. The joint *element* phase obligation is that every
   topology is either computed or refused with its reader cited.
6. **`/PROP/TYPE35` STITCH and `/PROP/TYPE46` MUSCLE** exist on both sides but
   need the audit, not the port.
7. **`rivet/rivet1.F`** is 420 lines against the port's 502 — the only 1-D
   family where the port is *longer* than the Fortran, which usually means the
   port absorbed reader code into the kernel module. Verify and split.

## Wave graph

```
Wave 0  P4.0 1-D reconciliation audit (serial — produces the status table)
   ── gate: every IGTYP branch classified
Wave 1 (parallel — new files, one branch per module)
  P4.1 IGTYP 6 (redef3/repla3)     P4.2 IGTYP 29/30/31 user springs
  P4.3 TYPE8 cuts (fct_IDji, hardening, rupture, rate, sensor, skew)
  P4.4 TYPE32 pretensioner         P4.5 python_funct_mod spring callback
  P4.6 /PROP/TYPE33 topology matrix
   ── gate: each branch has an analytic test + one oracle parity case
Wave 2 (serial)
  P4.7 rivet kernel/reader split
  P4.8 exhaustive IGTYP coverage + refusal gates
  P4.9 numba mirrors for the vectorised 1-D branches
```

---

### Task P4.0: 1-D reconciliation audit

**Fortran:** `truss/*`, `beam/*`, `spring/*`, `joint/*`, `rivet/*`, `xelem/*`.
**Files:** Create `tools/validation_data/w1d_routine_status.json`,
`tests/test_p4_reconciliation.py`.
**Interfaces:**
- Produces: the `solid_routine_status.json` shape, plus an `igtyp` field on
  every spring row so the coverage test of Task P4.8 can consume it.

- [ ] **Step 1** — write the failing test:

```python
def test_every_igtyp_branch_has_a_status():
    st = json.loads(Path("tools/validation_data/w1d_routine_status.json").read_text())
    have = {r["igtyp"] for r in st if r.get("igtyp") is not None}
    for igtyp in (1,2,3,4,5,6,8,12,13,19,23,25,26,27,29,30,31,32,33,35,36,44,45,46):
        assert igtyp in have, f"IGTYP {igtyp} unclassified"
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit every routine; for each of the eight TYPE8 cuts named
  at `PORTING_GUIDE.md:67`, record `missing` with that citation. Read the
  `PORTING_GUIDE` claim about TYPE13 ("implicit-spring path stays TYPE4-only")
  and confirm it against `r13ke3.F`/`r2def3.F`.
- [ ] **Step 4** — run → PASS; commit `docs(1d): per-routine audit of the 1-D families`.

---

### Task P4.1: `IGTYP` 6 — `r6def3.F`

**Fortran:** `engine/source/elements/spring/r6def3.F` (its `!||` block shows it
calls `redef3.F90` and `repla3.F`, and `uses python_funct_mod`),
`engine/source/elements/spring/redef3.F90`, `repla3.F`.
**Files:** Create `pyradioss/elements/spring_type6.py`,
`tests/test_p4_spring_type6.py`.
**Interfaces:**
- Consumes: the shared `spring.driver` contract:
  `driver(group, model, ctx) -> np.ndarray` (returns per-element dt).
- Produces: `spring_type6.driver`, `spring_type6.tangent`,
  `spring_type6.kgeo`, `spring_type6.energy`.

- [ ] **Step 1** — write the failing test:

```python
def test_type6_zero_stiffness_is_a_free_segment():
    g = _type6_group(k=0.0)
    g.v[0] = 1.0
    f = spring_type6.driver(g, _ctx(g))
    assert np.abs(f).max() == 0.0

def test_type6_linear_spring_matches_analytic():
    g = _type6_group(k=1000.0, length=2.0)
    g.v[1] = 1.0
    f = spring_type6.driver(g, _ctx(g))
    assert f[0] == pytest.approx(-1000.0 * 2.0 * 0.5 / 2.0, rel=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `redef3.F90` is a **shared** routine also used by
  other `IGTYP` values — factor it once into `pyradioss/elements/spring_common.py`
  and have both branches call it. Do not copy it twice; the two copies will
  drift and the drift will be invisible.
- [ ] **Step 4** — run → PASS; commit `feat(1d): IGTYP 6 spring branch (r6def3/redef3/repla3)`.

---

### Task P4.2: `IGTYP` 29 / 30 / 31 — user springs

**Fortran:** `engine/source/elements/spring/rforc3.F:862-1059` (the
`IGTYP >= 29 .AND. IGTYP <= 31` block, calling `ENG_USERLIB_RUSER`),
`ENG_USERLIB_RUSER` in `common_source/`, and the `/PROP/TYPE29..31` user
property readers in `starter/source/properties/user_spring_solid/`
(`PORTING_GUIDE.md:190` records `/PROP/TYPE29/30/31` as user property readers).
**Files:** Create `pyradioss/elements/spring_user.py`,
`tests/test_p4_spring_user.py`.
**Interfaces:**
- Produces: `spring_user.register_user_spring(igtyp, fn) -> None`
  (the port's own extension point, mirroring `ENG_USERLIB_RUSER`),
  `spring_user.driver(group, model, ctx)` refusing with a **named** error when
  no callback is registered.

- [ ] **Step 1** — write the failing test — Review Focus item 5:

```python
def test_unregistered_user_spring_is_refused_not_silently_skipped():
    g = _user_spring_group(igtyp=29)
    with pytest.raises(UserSpringNotRegistered, match="IGTYP 29"):
        spring_user.driver(g, _ctx(g))

def test_registered_user_spring_is_called_once_per_element():
    calls = []
    spring_user.register_user_spring(29, lambda el, ctx: calls.append(el) or np.zeros(3))
    g = _user_spring_group(igtyp=29, n=4)
    spring_user.driver(g, _ctx(g))
    assert len(calls) == 4
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Upstream `ENG_USERLIB_RUSER` resolves a symbol
  from a user object library; the open tree has no such library, so the port's
  Python-callback registration is the honest equivalent. **Refusing loudly is
  mandatory** — a silent zero force would be an unbounded energy leak with no
  warning.
- [ ] **Step 4** — run → PASS; commit `feat(1d): user spring types 29/30/31 with a loud refusal default`.

---

### Task P4.3: The eight documented TYPE8 cuts

**Fortran:** `engine/source/elements/spring/r2def3.F`,
`r2coor3.F`, `r2len3.F`, `r2len3law135.F90`, `r8ke3.F`, `r8sumg3.F`,
`r23sens3.F`, `r1sens3.F`; the reader `hm_read_prop08.F`.
**Files:** Modify `pyradioss/elements/spring_general.py`,
`pyradioss/input/keywords/properties.py`.
**Interfaces:**
- Consumes: `prop.fct_IDji`, `prop.Hi`, `prop.IECROU`, rupture params,
  `prop.ISRATE`, `prop.Fcut`, `prop.skew_ID`, `prop.sensor_ID` on
  `/PROP/TYPE8`.
- Produces: all eight features live; the `PORTING_GUIDE.md:67` cut list shrinks
  to only what is genuinely left.

- [ ] **Step 1** — write eight failing tests, one per cut. The force-function
  one:

```python
def test_type8_force_function_uses_the_requested_curve():
    """fct_IDji overrides the closed form with an /FUNCT curve (finter.F)."""
    g = _type8_group(fct_id=7)
    g.funct7 = np.array([[0.0, 0.0], [1.0, 100.0]])
    g.v[1] = 0.01
    got = spring_general.driver(g, _ctx(g))
    assert got is not None
    # and: with fct_IDji=0 the closed form is used instead
    g0 = _type8_group(fct_id=0); g0.v[1] = 0.01
    assert spring_general.driver(g0, _ctx(g0)) != got
```

  and one each for hardening, rupture, rate smoothing, sensor activation, skew
  frame, and the co-rotational beam frame.
- [ ] **Step 2** — run → FAIL on all eight.
- [ ] **Step 3** — implement, **one feature per commit** (the name cannot contain
  "and"). Update `PORTING_GUIDE.md:67`'s row in the same commit as the last one.
- [ ] **Step 4** — run → PASS; the pre-existing TYPE8 decks stay byte-identical
  (all eight default to off); commit per feature.

---

### Task P4.4: TYPE32 pretensioner physics

**Fortran:** `engine/source/elements/spring/ruser32.F`, `ruser32ke3.F`,
`ruser32mat3.F`, `preload_axial.F90`; the branch at `rforc3.F:1061`;
the reader `starter/source/properties/spring/hm_read_prop32.F`.
**Files:** Modify `pyradioss/elements/spring_pretensioner.py`,
`pyradioss/input/keywords/properties.py`.
**Interfaces:**
- Produces: real physics instead of `InactiveProperty`; the Starter no longer
  produces an `InactiveProperty` for TYPE32, and the c52 deck moves out of SKIPS.

- [ ] **Step 1** — write the failing test:

```python
def test_type32_no_longer_refuses_at_the_starter():
    model = _starter(_deck("/SPRING\nSPRINGA\n/PROP/TYPE32\n"))
    assert not isinstance(model.prop_type32s[1], InactiveProperty)

def test_pretensioner_imposes_its_locked_length():
    g = _type32_group(locked_length=0.01, natural_length=0.02)
    f = spring_pretensioner.driver(g, _ctx(g))
    assert abs(f).sum() > 0.0     # a preload force exists at rest
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `VALIDATION.md` §7 item 11 already assessed this as
  tractable; **read `ruser32.F` before writing anything** — the pretensioner
  solves for a locked length at initialisation and then behaves as a spring about
  *that* length, which is not a stiffness the Engine can evaluate ad hoc.
  The initialisation half belongs to the Starter (`hm_read_prop32.F` +
  `starter/source/elements/…`), the force half to the Engine.
- [ ] **Step 4** — run → PASS; **the c52 corpus case must change verdict** —
  measured, not assumed; commit `feat(1d): TYPE32 pretensioner physics (ruser32)`.

---

### Task P4.5: The `python_funct_mod` spring callback

**Fortran:** `common_source/modules/python_mod.F90`,
`common_source/modules/cpp_python_funct.cpp`, the `PYTHON` first argument that
`r1def3/r2def3/r3def3/r6def3/r26def3/r27def3` all receive
(`common_source/modules/python_signal.h`).
**Files:** Create `pyradioss/elements/spring_python_funct.py`,
`tests/test_p4_spring_python_funct.py`.
**Interfaces:**
- Consumes: `pyradioss.input.functions` (the `/FUNCT` tables the port already
  has) and the `/FUNCT_PYTHON` registry from M119.
- Produces: `spring_python_funct.resolve(fct_id, model) -> Callable[[np.ndarray], np.ndarray]`
  — one curve evaluator shared by every branch that needs it, plus
  `spring_python_funct.PYTHON_FUNCT_ENABLED` (default **on** for local curves,
  off for arbitrary user code).

- [ ] **Step 1** — write the failing test:

```python
def test_type26_python_curve_matches_a_hand_written_funct():
    g = _type26_group(fct_id=3)
    g.functs[3] = np.array([[0.0, 0.0], [0.5, 10.0], [1.0, 0.0]])
    out = spring_python_funct.resolve(3, g.model)(np.array([0.5]))
    assert out == pytest.approx([10.0])
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Security and determinism:** upstream's
  `python_funct_mod` executes user Python. The port must (a) never execute a
  deck-supplied string by default, (b) route deck `/FUNCT_PYTHON` calls through
  an explicit registry the user populated in their own process, and (c) record
  that policy in the module docstring. A solver that `exec()`s a deck line is a
  remote-code-execution vector; this is not optional diligence.
- [ ] **Step 4** — run → PASS; commit
  `feat(1d): python_funct_mod callback for TYPE26/27 curves (registry-based, no deck exec)`.

---

### Task P4.6: `/PROP/TYPE33` topology matrix

**Fortran:** the nine joint readers under
`$OR_SRC/starter/source/properties/spring/`:
`hm_read_prop33_cyl_jnt.F`, `hm_read_prop33_fix_jnt.F`,
`hm_read_prop33_free_jnt.F`, `hm_read_prop33_old_jnt.F`,
`hm_read_prop33_plan_jnt.F`, `hm_read_prop33_rev_jnt.F`,
`hm_read_prop33_sph_jnt.F`, `hm_read_prop33_trans_jnt.F`,
`hm_read_prop33_univ_jnt.F`; the element side
`engine/source/elements/joint/*` (`rbilan33.F`, `rcum33.F`, `rcum33p.F`,
`ranim33.F`, `rdtime33.F`, `rskew33.F`, `joint_block_stiffness.F`,
`joint_elem_timestep.F`, `rgjoint.F`, `ruser33.F`).
**Files:** Create `pyradioss/elements/joint_topologies.py`,
`tests/test_p4_joint_topologies.py`, and the nine readers in
`pyradioss/input/keywords/properties.py`.
**Interfaces:**
- Produces: `joint_topologies.TOPOLOGIES: dict[str, JointTopology]` with one
  entry per joint kind (`fixed`, `free`, `planar`, `revolute`, `spherical`,
  `cylindrical`, `universal`, `translation`, `old`), each with
  `dofs: int`, `free_dofs: tuple[int, ...]`, `ncmax: int`.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("kind,dofs", [("fixed",0),("planar",1),("revolute",1),
                                        ("spherical",3),("cylindrical",2),
                                        ("universal",2),("translation",3)])
def test_joint_topology_frees_the_right_dofs(kind, dofs):
    t = joint_topologies.TOPOLOGIES[kind]
    assert len(t.free_dofs) == dofs

def test_every_topology_is_routed_or_refused():
    for kind in joint_topologies.TOPOLOGIES:
        assert kind in joint_topologies.routed() or \
               kind in joint_topologies.refused()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **A joint's free-DOF count is the whole physics**
  — get it from each reader's `HM_GET_INTV` calls, not from the name. The
  `!||` block of `ruser33.F` states TYPE33/45 springs are computed **once on the
  owning domain and frontier-summed** by `spmd_exch_a.F`, never replicated —
  which is the fact `docs/OPEN_BUGS.md` SPMD-3 records the port getting wrong
  before it was refused. Phase 13 needs this matrix.
- [ ] **Step 4** — run → PASS; commit `feat(1d): the nine TYPE33 joint topologies`.

---

### Task P4.7: Split `rivet.py`

**Fortran:** `engine/source/elements/rivet/rivet1.F` (420 lines);
`starter/source/properties/rivet/` (135 lines).
**Files:** Create `pyradioss/elements/rivet.py` (kernel only),
`pyradioss/input/keywords/properties.py` (reader).
**Interfaces:**
- Produces: `rivet.init_group`, `rivet.forces`, `rivet.dt_claim`,
  `rivet.tangent`, `rivet.kgeo`, `rivet.energy` — the standard kernel contract
  and nothing else.

- [ ] **Step 1** — write the failing test:

```python
def test_rivet_module_has_no_reader_code():
    src = Path("pyradioss/elements/rivet.py").read_text()
    assert "HM_GET" not in src and "hm_read" not in src
    assert rivet.forces and rivet.init_group
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — split. A 502-line Python module against a 420-line Fortran
  file means the port folded reader or material code into the kernel; find it
  and move it. This is a **pure move with an import fix**, and the fixer's
  justification for putting it there originally must be recorded in the commit
  (if there was no good reason, say so).
- [ ] **Step 4** — run → PASS; full suite counts identical; commit
  `refactor(1d): separate the rivet kernel from its reader`.

---

### Task P4.8: Exhaustive `IGTYP` coverage and refusal gates

**Fortran:** `rforc3.F`'s `IGTYP` ladder (cited in §Scope), plus
`rforc3.F`'s `ELSEIF` chain — **an `IGTYP` with no branch upstream raises**,
so the port must match that: exhaustive over the documented set, loud outside it.
**Files:** Modify `pyradioss/elements/spring.py`, `spring_advanced.py`,
`pyradioss/input/checks.py`.
**Interfaces:**
- Produces: `pyradioss.elements.IGTYP_GROUPS: dict[int, str]` mapping every
  `IGTYP` to its kernel group; `checks.unsupported_igtyp() -> dict[int, str]`
  (value → the upstream file that lacks the branch).

- [ ] **Step 1** — write the failing test:

```python
def test_igtyp_is_exhaustive():
    known = set(IGTYP_GROUPS) | set(checks.unsupported_igtyp())
    for igtyp in range(0, 50):
        assert igtyp in known, f"IGTYP {igtyp} silently unhandled"

def test_unsupported_igtyp_cites_upstream():
    for igtyp, cite in checks.unsupported_igtyp().items():
        assert cite.startswith("engine/source/elements/spring/")
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Same rationale as Task P2.11: the port's 440
  corpus "SKIPS" are largely silent skips, and this test converts each into an
  explicit, cited routing decision.
- [ ] **Step 4** — run → PASS; corpus coverage delta measured; commit
  `feat(1d): exhaustive IGTYP routing with cited refusals`.

---

### Task P4.9: numba mirrors for the 1-D branches

**Fortran:** none.
**Files:** Create `pyradioss/accel/jit_kernels/spring.py`, `beam.py`.
**Interfaces:**
- Consumes: the vectorised branch drivers from Tasks P4.1–P4.5.
- Produces: bitwise-identical mirrors for the branches that iterate over
  *elements*; branches that iterate over *curves* are not worth mirroring and
  are not.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("name", ["spring_linear", "spring_curve", "beam_type3"])
def test_1d_mirror_is_bitwise(name):
    ref = accel.reference(name)
    assert accel.get(name)(*ref.inputs).tobytes() == ref.expected.tobytes()
```

- [ ] **Step 2** — run → SKIP until P0.7, then FAIL.
- [ ] **Step 3** — implement one kernel per commit, statement-for-statement as
  in Phase 2 Task P2.12. Where a mirror would be slower than NumPy, **do not
  add it** and exclude the branch from the `auto` rule instead; record the
  measurement.
- [ ] **Step 4** — run → PASS; restart md5 unchanged; commit per kernel.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P4.0** — all 24 `IGTYP` values classified; the eight TYPE8 cuts are recorded
  with citations, not omitted.
- **P4.1** — `redef3.F90` is factored once, not copied per branch.
- **P4.2** — an unregistered user spring **raises by name**; a silent zero force
  is an automatic rejection.
- **P4.3** — one feature per commit; existing TYPE8 decks byte-identical.
- **P4.4** — the c52 verdict change is measured, not assumed.
- **P4.5** — no `exec`/`eval` of deck content anywhere; the registry policy is in
  the docstring.
- **P4.6** — free-DOF counts are read from the readers, not from the joint's
  name; the "computed once on the owning domain" fact is carried forward for
  Phase 13.
- **P4.8** — exhaustive over `0..49`, with citations on every refusal.

## Parallelisation

- **Wave 0** — P4.0 serial.
- **Wave 1** — 6 tasks, 6 agents, disjoint modules. P4.3 is internally six
  sequential commits by one agent (it edits one module eight times); it is the
  long pole of the wave and starts first.
- **Wave 2** — serial.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p4_reconciliation.py tests/test_p4_joint_topologies.py
python tools/validate_vs_fortran.py --mode parity --cases SEATBELT,RD-E-5200 --out tools/validation_data/parity_p4.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area engine/source/elements/spring
```

The phase reviewer confirms the census shows no `missing` row in
`truss/`, `beam/`, `spring/`, `joint/`, `rivet/`, `xelem/` without a cited
reason, and that `docs/STATE.md` no longer claims TYPE32 is `InactiveProperty`.