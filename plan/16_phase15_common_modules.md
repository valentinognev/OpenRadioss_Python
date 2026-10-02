# Phase 15 — Common modules, tools, linear algebra, communications

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `common_source/**` a literal port. These are the modules every
other phase calls, so they are **last in dependency order for physics and first
in quality order**: a wrong `array_mod.F` corrupts everything, and a wrong
`precision_mod.F90` corrupts it silently.

**Architecture:** one Python module per Fortran module file. `pyradioss/common/`
is **1,373 lines** against ≈ 70,500 Fortran lines (a 1:51 ratio), so most of this
phase is new code, not a port of existing code.

**Tech stack:** NumPy; SciPy where a Fortran file calls LAPACK.
**Spec:** `plan/00_ORCHESTRATION.md` §1.1.
**Entry criteria:** Phases 2–14 (the callers exist, so their requirements are
  known).
**Band: M.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `common_source/modules/` | 87 | **32,850** | `common/npcompat.py`, `fastmath.py`, `units.py` (in `input/`) |
| `common_source/tools/` | 48 | **17,259** | `common/fastmath.py` (partly) |
| `common_source/interf/` | 9 | 6,763 | `contact/stiffness.py` (partly) |
| `common_source/eos/` | 26 | 4,898 | Phase 7 Task P7.4 |
| `common_source/output/` | 8 | 2,300 | Phase 12 |
| `common_source/comm/` | 9 | 1,563 | Phase 13 Task P13.1 |
| `common_source/qa/` | 2 | 1,194 | Phase 16 |
| `common_source/sortie/`, `linearalgebra/`, `fail/`, `fxbody/`, `input/`, `includes/` | 9 | 1,005 | — |
| **Total** | **199** | **≈ 69,832** | **1,373** |

### `common_source/modules/` — the 87 files that matter most

`airbag/`, `ale/`, `boundary_conditions/`, `constraints/`, `elements/`,
`loads/`, `mat_elem/`, `interfaces/`, plus the top-level files:

| Module | LOC | What it is |
|---|---:|---|
| `precision_mod.F90` | ~200 | `my_real`, the precision selection (`sp`/`dp`) — **a silent-corruption module** |
| `array_mod.F` | ~4,000 | dynamic array growth, `ALLOC`, `FREED`, the memory pool |
| `nodal_arrays.F90` | ~3,500 | the node array registry (`NODA`, the `NODA_r`, `NODA_c`, `NODA_t` sets) |
| `constant_mod.F` | ~1,200 | `ZERO`, `ONE`, `HALF`, `EM01`, `EM02`, `EM20` and the rest of the constants table |
| `element_mod.F90` / `connectivity.F90` | ~3,000 each | element state, `ELBUF`, connectivity |
| `check_mod.F` | ~1,500 | the `ANCMSG`/`ANCRES`/`ARRET` message system |
| `names_and_titles_mod.F` | ~800 | the model name/title tables |
| `setdef_mod.F`, `groupdef_mod.F`, `skew_mod.F`, `state_mod.F`, `sensor_mod.F`, `seatbelt_mod.F`, `inoutfile_mod.F`, `outmax_mod.F` | ~1,000 each | the keyword-entity registries |
| `error_mod`, `memstat`, `optiondef_mod.F90`, `unitab_mod.F`, `table4d_mod.F90`, `cast_mod.F90`, `plot_curve_mod.F90`, `root_finding_algo_mod.F90` | ~1,000 each | |

## Gap analysis

1. **`precision_mod.F90` and `constant_mod.F` are silent-corruption modules.**
   A wrong `EM20` shifts an hourglass force; a wrong `ZERO` shifts a sign
   convention. **These are the first two tasks.**
2. **`array_mod.F` (~4,000 LOC)** — Fortran's dynamic array growth with
   `ALLOC`/`FREED` and a memory pool. The port uses NumPy arrays sized from the
   deck, which is *sufficient* but not *faithful*. **Decide:** port `array_mod`
   for the cases where upstream's growth semantics are observable (e.g. a
   `/SURF/GRNOD` that grows as it resolves), or record the deviation.
3. **`nodal_arrays.F90` (~3,500)** — the node-array registry with its
   `NODA_r`/`NODA_c`/`NODA_t` groupings and their offsets. The port has
   `model/entities.py` (53,286 lines, before Phase 1 Task P1.4 splits it). The
   registry's *offsets and lengths* are what `model.nb_nodes`-style accessors
   read; getting them wrong is a silent data mix-up.
4. **`check_mod.F` (~1,500)** — the `ANCMSG`/`ANCRES`/`ARRET` system, with the
   message catalogue in `hm_cfg_files/messages/CONFIG/*.txt` (Phase 10 Task
   P10.6's message-ID mapping needs this).
5. **`common_source/tools/` (48 files, 17,259)** — the shared computational
   toolbox: curve interpolation (`finter.F`), sorting, search, linear algebra,
   the `table4d` interpolator, root finding. Many are already used implicitly by
   the port's NumPy equivalents; the phase's job is to make each one explicit
   and cited.
6. **`common_source/linearalgebra/` (343 LOC, 2 files)** — a dense LAPACK
   wrapper. The port uses SciPy. Decide and record.
7. **`common_source/interf/` (9 files, 6,763)** — the shared interface
   machinery (`interf_generic.F`, the sort criteria, the voxel definitions).
   Phase 8 Task P8.3 owns the port; this phase provides the shared base it
   should have used.

## Wave graph

```
Wave 0 (serial — everything depends on these)
  P15.0 module audit + the port/decline decision per file
  P15.1 precision + constants         P15.2 the message system (check_mod)
   ── gate: constants pinned by value; every message id resolvable
Wave 1 (parallel — new modules)
  P15.3 array_mod / memory pool       P15.4 nodal array registry
  P15.5 element_mod / connectivity   P15.6 setdef/groupdef/skew registries
  P15.7 the tools toolbox (finter, sort, search, root finding, table4d)
  P15.8 interf shared base            P15.9 linear algebra decision
   ── gate per module: analytic test + a caller
Wave 2 (serial)
  P15.10 swap the call sites onto the new modules; re-measure
```

---

### Task P15.0: Module audit and decisions

**Fortran:** every file in `common_source/` except `eos/` (Phase 7),
  `comm/` (Phase 13), `output/` (Phase 12), `qa/` (Phase 16).
**Files:** Create `tools/validation_data/common_status.json`,
`tests/test_p15_common_census.py`.
**Interfaces:**
- Produces: one record per file with `port` (`pyradioss.common.<name>` |
  `pyradioss.<other>` | null), `decision` (`port` | `substitute` | `decline`),
  `decision_reason`, `citation`.

- [ ] **Step 1** — write the failing test:

```python
def test_every_common_source_file_has_a_decision():
    st = json.loads(Path("tools/validation_data/common_status.json").read_text())
    for r in st:
        assert r["decision"] in {"port", "substitute", "decline"}
        assert r["decision_reason"], r["fortran"]
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit. **The `substitute` decision is legitimate and common
  here**: a Fortran `DOUBLE PRECISION` loop over a dense matrix becomes a SciPy
  call. `substitute` requires the *name* of the replacement and a statement of
  whether the results are bit-identical, order-preserving, or merely
  equivalent. A `decline` requires a reason. A `port` requires the target
  module name.
- [ ] **Step 4** — run → PASS; commit `feat(census): common_source audit with port decisions`.

---

### Task P15.1: Precision and constants

**Fortran:** `common_source/modules/precision_mod.F90` (`my_real`,
  `ONE`/`ZERO`/`HALF` in single and double, the `my_real` kind selection),
  `common_source/modules/constant_mod.F` (the constants table: `ZERO`, `ONE`,
  `HALF`, `THIRD`, `EM01`, `EM02`, `EM04`, `EM05`, `EM06`, `EM10`, `EM12`,
  `EM14`, `EM16`, `EM20`, `EM24`, `EM40`, `PI`, `TINY`, …).
**Files:** Create `pyradioss/common/precision.py`, `pyradioss/common/constant.py`,
`tests/test_p15_constants.py`.
**Interfaces:**
- Produces: `precision.my_real` (an alias for `np.float64`),
  `precision.SINGLE = False` (the port's precision flag),
  `precision.set_precision(single: bool) -> None`,
  and `constant.ZERO`, `.ONE`, `.HALF`, `.EM01`, … as module-level floats.

- [ ] **Step 1** — write the failing test — **value-pinned, not derived**:

```python
def test_constants_are_the_fortran_values():
    from pyradioss.common import constant as C
    assert C.EM01 == 1e-2
    assert C.EM02 == 1e-2
    assert C.EM20 == 1e-20
    assert C.EM06 == 1e-6
    assert C.HALF == 0.5
    assert C.PI == pytest.approx(math.pi, rel=1e-15)

def test_switching_precision_raises_under_the_flag():
    from pyradioss.common import precision
    assert precision.my_real is np.float64          # dp
    precision.set_precision(True)
    assert precision.my_real is np.float32
    precision.set_precision(False)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement by **transcribing the `DATA` block**, not by typing
  plausible values. The test pins the ones the port actually uses; a reviewer
  should diff the full table against `constant_mod.F`. **The precision flag is a
  global**: setting it must be centralised here and nowhere else, or two parts of
  the engine will disagree about `my_real` and produce a type error at the
  boundary.
- [ ] **Step 4** — run → PASS; `md5` of every bundled example unchanged;
  commit `feat(common): precision selection and the constants table (constant_mod)`.

---

### Task P15.2: The message system

**Fortran:** `common_source/modules/check_mod.F` (~1,500 LOC) —
  `ANCMSG`, `ANCRES`, `ARRET`, `ANINFO`, the message levels, the output unit
  routing; the catalogue in
  `$OR_SRC/hm_cfg_files/messages/CONFIG/msg_hw_radioss_reader.txt` and
  `starter/source/output/message/starter_message_description.txt`.
**Files:** Create `pyradioss/common/messages.py`, `tests/test_p15_messages.py`.
**Interfaces:**
- Consumes: the CFG message catalogue (Phase 0 Task P0.6's `hm_cfg_dir()`).
- Produces: `messages.ANCMSG(msgid, type, **fmt) -> Message`,
  `messages.ARRET(code) -> NoReturn`,
  `messages.ANCRES(...) -> Message`,
  `messages.CATALOGUE: dict[int, str]` (id → message template),
  `messages.MSG_FATAL / MSG_ERROR / MSGWARNING / MSGWARNING / MSGINFO`.

- [ ] **Step 1** — write the failing test:

```python
def test_every_msgid_used_in_the_port_resolves_to_an_upstream_template():
    import re, pathlib
    used = set()
    for p in pathlib.Path("pyradioss").rglob("*.py"):
        used |= {int(m) for m in re.findall(r"MSGID\s*=\s*(\d+)", p.read_text())}
    missing = [i for i in used if i not in messages.CATALOGUE]
    assert missing == [], missing

def test_arret_code_follows_the_upstream_convention():
    assert messages.ARRET.__doc__
    with pytest.raises(SystemExit) as e:
        messages.ARRET(2)
    assert e.value.code == 2
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The port already emits `MSGID=NNN` in many
  places (the `/SUBMODEL` balance check, `hm_read_properties.F:798 VDOUBLE`, the
  `ANCMSG 305` LAW14 refusal) — **those ids must now resolve to the upstream
  template**, which means a port message becomes traceable to the Fortran
  message it mirrors. A `MSGID` used with a template the catalogue does not
  contain is a citation error.
- [ ] **Step 4** — run → PASS; commit `feat(common): the ANCMSG/ANCRES/ARRET message system`.

---

### Task P15.3: `array_mod` and the memory pool

**Fortran:** `common_source/modules/array_mod.F` (~4,000 LOC) — `ALLOC`,
  `FREED`, the segment allocator, the memory pool, `BLOCK`/`BLOCKMIN`,
  `SETALLOC`/`TESTALLOC`.
**Files:** Create `pyradioss/common/arrays.py`, `tests/test_p15_arrays.py`.
**Interfaces:**
- Produces: `arrays.Segment(name, dtype)` with `.data: np.ndarray`,
  `.capacity`, `.n`, `.grow(n)`, `.shrink(n)`, `.view(shape)`;
  `arrays.Pool` with `.alloc(name, n)`, `.free(name)`, `.usage() -> dict`.

- [ ] **Step 1** — write the failing test:

```python
def test_grow_preserves_contents_and_shrink_frees():
    s = arrays.Segment("FINT", np.float64)
    s.grow(4); s.data[:] = [1, 2, 3, 4]
    s.grow(6)
    assert list(s.data[:4]) == [1, 2, 3, 4]
    assert s.data[4:].tolist() == [0.0, 0.0]
    assert s.capacity >= 6

def test_pool_accounting_is_exact():
    p = arrays.Pool()
    p.alloc("a", 100); p.alloc("b", 50)
    assert p.usage()["a"] == 100
    p.free("a")
    assert "a" not in p.usage()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement **only where upstream's growth semantics are
  observable.** The port pre-sizes arrays from the deck counts, which is
  equivalent for almost everything; the exceptions are the resolvers that grow
  as they resolve (`/GRNOD/GRNOD` fixpoint, `/SURF/GRNOD`, `/LINE/EDGE`).
  **Porting `array_mod` wholesale would be busywork**; this task ports the
  growth-tracking allocator those resolvers need and records the rest as
  `substitute`.
- [ ] **Step 4** — run → PASS; commit
  `feat(common): the growth-tracking array allocator (array_mod subset)`.

---

### Task P15.4: The node array registry

**Fortran:** `common_source/modules/nodal_arrays.F90` (~3,500 LOC) — `NODA`,
  the `NODA_r`/`NODA_c`/`NODA_t` groupings, the offset/length arrays, and the
  per-array registration.
**Files:** Create `pyradioss/common/nodal_arrays.py`,
`tests/test_p15_nodal_arrays.py`.
**Interfaces:**
- Produces: `nodal_arrays.Registry` with `.register(name, kind) -> int`,
  `.offset(name) -> int`, `.length(name) -> int`, `.view(name) -> np.ndarray`,
  `.as_dict() -> dict[str, np.ndarray]`.

- [ ] **Step 1** — write the failing test:

```python
def test_registered_arrays_are_disjoint_and_cover_the_registry():
    r = nodal_arrays.Registry()
    for name in ("NODA_VEL", "NODA_COI", "NODA_RFR"):
        r.register(name, kind="r")
    views = [r.view(n) for n in ("NODA_VEL", "NODA_COI", "NODA_RFR")]
    assert sum(v.nbytes for v in views) == r.usage_bytes()

def test_unknown_name_raises_rather_than_returning_a_new_array():
    r = nodal_arrays.Registry()
    with pytest.raises(KeyError):
        r.view("NODA_NOPE")
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The `model` package already has a parallel
  structure in `model/entities.py`; **the registry is where they must meet**, so
  this task also replaces the ad-hoc `getattr(model, ...)` accesses with
  registry views. That is a **large, mechanical refactor** — do it group by group
  (`pyradioss/engine/`, then `elements/`, then `contact/`) so a failure bisects.
- [ ] **Step 4** — run → PASS; the fast tier counts unchanged; commit per group.

---

### Task P15.5: `element_mod` and connectivity

**Fortran:** `common_source/modules/element_mod.F90` (~3,000 LOC),
  `connectivity.F90` (~3,000).
**Files:** Create `pyradioss/common/connectivity.py`,
`tests/test_p15_connectivity.py`.
**Interfaces:**
- Produces: `connectivity.table(isolid, ish3n, itetra4) -> FamilySpec` with
  `FamilySpec.n_nodes`, `.n_layers`, `.element_type`,
  `.connectivity_shape`, `.corners()`.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("idsolid,n", [(0, 8), (2, 8), (14, 8), (17, 8),
                                        (24, 8), (43, 3), (1, 8)])
def test_family_spec_node_counts_match_the_upstream_table(idsolid, n):
    assert connectivity.table(idsolid, 0, 0).n_nodes == n
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The `Idsolid` → node-count table is exactly the
  dispatch Phase 2 Task P2.11's exhaustiveness test needs**, so the two must
  agree. Read the table from `element_mod.F90` and make Phase 2's
  `SOLID_ISOLID_GROUPS` a consumer of it rather than an independent list — one
  source of truth.
- [ ] **Step 4** — run → PASS; Phase 2's test still green; commit
  `refactor(common): one element-family table, consumed by the dispatch`.

---

### Task P15.6: The keyword-entity registries

**Fortran:** `setdef_mod.F`, `groupdef_mod.F`, `skew_mod.F`, `state_mod.F`,
  `sensor_mod.F`, `seatbelt_mod.F`, `names_and_titles_mod.F`,
  `inoutfile_mod.F`, `outmax_mod.F`, `optiondef_mod.F90`.
**Files:** Create `pyradioss/common/registry.py`, `tests/test_p15_registry.py`.
**Interfaces:**
- Produces: `registry.Registry(kind)` with `.declare(id, spec)`,
  `.resolve(id) -> spec`, `.all()`, `.require(id, citation) -> spec` — where
  `require` **raises a named error naming the upstream file** for an undeclared
  id.

- [ ] **Step 1** — write the failing test:

```python
def test_require_names_the_upstream_citation_for_an_undeclared_id():
    r = registry.Registry("sensor")
    with pytest.raises(UndeclaredEntityError) as e:
        r.require(999, "engine/source/tools/sensor/sensor.F")
    assert "999" in str(e.value) and "sensor.F" in str(e.value)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. This is the module that makes Review Focus item 1
  and item 5 mechanical: an undeclared `/SENSOR`, `/SET`, `/SKEW` or `/STATE`
  **must raise with a citation** rather than creating an empty record.
- [ ] **Step 4** — run → PASS; commit `feat(common): the keyword-entity registries`.

---

### Task P15.7: The tools toolbox

**Fortran:** `common_source/tools/` (48 files, 17,259) — `finter.F` (curve
  interpolation, called by `r27def3.F` per its `!||` block),
  `table4d_mod.F90`, `root_finding_algo_mod.F90`, `plot_curve_mod.F90`,
  `cast_mod.F90`, `unitab_mod.F`, the sort and search routines, `kbmatrix.F`,
  `kee.F` and the rest.
**Files:** Create `pyradioss/common/tools/` package
(`finter.py`, `table4d.py`, `root_finding.py`, `plot_curve.py`, `cast.py`,
`units.py`, `sorting.py`, `search.py`, `kbmatrix.py`),
`tests/test_p15_tools.py`.
**Interfaces:**
- Produces: `tools.finter(x, curve) -> float` with upstream's **linear
  extrapolation past the last point** and **clamping behaviour**;
  `tools.table4d.interp(...)`; `tools.root_finding.{brent, bisect}`;
  `tools.plot_curve.render(...)`.

- [ ] **Step 1** — write the failing test — the extrapolation behaviour is the
  specification:

```python
def test_finter_extrapolates_linearly_past_the_last_point():
    c = np.array([[0.0, 0.0], [1.0, 10.0]])
    assert tools.finter(0.5, c) == pytest.approx(5.0)
    assert tools.finter(2.0, c) == pytest.approx(20.0)      # linear beyond

def test_finter_clamps_below_the_first_point():
    c = np.array([[1.0, 3.0], [2.0, 5.0]])
    assert tools.finter(0.0, c) == pytest.approx(3.0)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **`finter.F`'s out-of-range behaviour is
  load-bearing**: it feeds the TYPE27 spring (`r27def3.F` calls `finter`) and
  several material curves, and an implementation that clamps where upstream
  extrapolates changes the force on every off-table value. Read it before
  writing either test's expectation into code — if the tests above disagree with
  the Fortran, **the tests are wrong** and get corrected with the citation.
- [ ] **Step 4** — run → PASS; commit per tool.

---

### Task P15.8: The interface shared base

**Fortran:** `common_source/interf/` (9 files, 6,763) — the shared interface
  definitions (`ITYPE` values, the `I2`/`I10`/`I24` common parameters,
  `istori`/`istfin`/`ISTF`), the generic interface utilities.
**Files:** Create `pyradioss/common/interf_base.py`,
`tests/test_p15_interf_base.py`.
**Interfaces:**
- Consumes: Phase 8 Task P8.3's `broadphase` (which should have used this).
- Produces: `interf_base.ITYPE: frozenset[int]`,
  `interf_base.common_params(itype) -> Params` with `istf`, `igap`, `ign`,
  `idure`, `ifric`, `itynod`, `itiedt`.

- [ ] **Step 1** — write the failing test:

```python
def test_common_params_cover_every_interface_type():
    for it in sorted(interf_base.ITYPE):
        p = interf_base.common_params(it)
        assert p.istf in range(0, 6)
        assert p.igap in range(0, 5)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, and **re-point Phase 8's per-type modules at it**,
  so the `ISTF` ladder is defined once. `ISTF` (0–5) is the interface-stiffness
  variant ladder the port already implements for TYPE7
  (`PORTING_GUIDE.md:85`); every other type should read it from here.
- [ ] **Step 4** — run → PASS; Phase 8's tests still green; commit
  `refactor(contact): the common interface-parameter base (common_source/interf)`.

---

### Task P15.9: The linear-algebra decision

**Fortran:** `common_source/linearalgebra/` (2 files, 343 LOC) — a dense LAPACK
  wrapper.
**Files:** Modify `pyradioss/implicit/linsolve.py`, Create
`docs/LINEAR_ALGEBRA.md`.
**Interfaces:**
- Produces: a documented decision: `substitute` (SciPy) with a statement of
  which LAPACK routine replaces which Fortran call and whether the results are
  bit-identical, order-preserving, or merely equivalent.

- [ ] **Step 1** — write the failing test:

```python
def test_linear_algebra_decision_is_recorded():
    doc = Path("docs/LINEAR_ALGEBRA.md").read_text()
    for r in ("dgefa", "dgesv", "dgbtrf", "dsytrf", "dpocon"):
        assert r in doc or r.upper() in doc
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **A sparse-versus-dense substitution is a real
  behaviour difference** for a badly conditioned system: a dense Cholesky on an
  SPD matrix is backward-stable in a way a sparse iterative solve is not. The
  decision must say which solver is used for which matrix class and what the
  consequence is.
- [ ] **Step 4** — run → PASS; commit `docs(linear-algebra): the SciPy-for-LAPACK substitution decision`.

---

### Task P15.10: Swap the call sites and re-measure

**Files:** Modify the calling modules across `pyradioss/`.
**Interfaces:**
- Produces: no caller uses an ad-hoc constant or a bare `getattr` for a
  registered entity; `tools/validation_data/perf_p15.json`; a speed comparison
  (the new modules should not be slower).

- [ ] **Step 1** — write the failing test:

```python
def test_no_module_defines_its_own_copy_of_a_constant():
    import re, pathlib
    bad = []
    for p in pathlib.Path("pyradioss").rglob("*.py"):
        if p.name in ("constant.py", "precision.py"):
            continue
        if re.search(r"^\s*EM20\s*=", p.read_text(), re.M):
            bad.append(str(p))
    assert bad == [], bad
```

- [ ] **Step 2** — run → FAIL (many modules define their own).
- [ ] **Step 3** — swap, module by module, in separate commits. **Measure
  before and after**: a central registry adds a lookup on a hot path, and if it
  costs more than 5 % of the cycle the design needs a cached accessor rather
  than a raw lookup. The `perf_p15.json` comparison is the deliverable.
- [ ] **Step 4** — run → PASS; the perf comparison is within the stated budget;
  commit per module group.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P15.1** — the constants table is **transcribed**, and a reviewer diffs the
  full table against `constant_mod.F`. Precision selection is centralised.
- **P15.2** — every `MSGID` the port emits resolves to an upstream template; a
  port message is traceable to the Fortran message it mirrors.
- **P15.3** — `array_mod` is ported **only where growth semantics are
  observable**, and the rest is `substitute` with a reason.
- **P15.5** — there is **one** `Idsolid` table, not two.
- **P15.7** — the extrapolation/clamp tests were **checked against
  `finter.F`**; if the test disagreed with the Fortran, the test was corrected
  with the citation, not the code.
- **P15.10** — the perf comparison is reported; a >5 % regression forces a
  cached accessor rather than being accepted.

## Parallelisation

- **Wave 0** — P15.0 serial; P15.1 and P15.2 parallel with it.
- **Wave 1** — 7 tasks. P15.8 depends on Phase 8's `broadphase` interface but
  not on its implementation; P15.4 and P15.5 touch `pyradioss/model/` and are
  the contended ones — schedule them **last** in the wave.
- **Wave 2** — serial (a repo-wide refactor).

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q
python -m pytest -q tests/test_p15_constants.py tests/test_p15_messages.py tests/test_p15_tools.py
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area common_source
```

The phase reviewer confirms:

1. the fast tier **and the full suite** are unchanged — this phase is a
   refactor, not a behaviour change;
2. no `common_source` file is `missing` without a decision and a reason;
3. the perf comparison is reported against `perf_m41.json`.