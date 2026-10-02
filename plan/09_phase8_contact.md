# Phase 8 — Contact / interfaces

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/interfaces/**` and `starter/source/interfaces/**`
a literal port: every interface type, the shared interface driver
(`interf/`, 33k LOC), the broad-phase sort machinery (`intsort/`, 35k LOC),
the voxel/box machinery (`generic/`), and the 2-D and surface-geometry halves.

**Architecture:** OpenRadioss contact has **four separable layers**, and the port
has only two of them:

| Layer | Upstream | LOC | Port |
|---|---|---:|---|
| 1. Broad phase (sorting, voxels, boxes) | `intsort/` (103 files), `generic/` (17) | 37,517 | `contact/stiffness.py`, a voxel bucket in `inter_type7.py` |
| 2. Interface driver (state, tracking, deletion) | `interf/` (~40 files) | 32,919 | `contact/tracking.py` (partial) |
| 3. Per-type narrow phase + force | `int07`, `int09`, `int10`, `int11`, `int14`…`int25` | 108,247 | 22 `inter_type*.py` |
| 4. Starter surface/segment construction | `starter/source/interfaces/inter3d1/` (59,973!), `interf1/`, `intbuf/`, `inter2d1/` | 71,737 | `starter/initialization.py` |

Layer 4 is the single largest file in the entire Fortran source
(`inter3d1`, 59,973 LOC) and the port has **no dedicated module** for it.
Layer 1 (`intsort`, 35,155 LOC across 103 files) is where the port's single
voxel bucket in `inter_type7.py` lives today — meaning the port has one broad
phase where upstream has a general one.

**Tech stack:** NumPy, `scipy.spatial.cKDTree` where upstream uses its own
spatial structures.
**Spec:** `plan/00_ORCHESTRATION.md` §1.1, §10 item 5.
**Entry criteria:** Phases 2 and 3 exit gates (elements exist to contact);
Phase 7 Task P7.0 (the `off` flag this phase reads).
**Band: XL.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `engine/source/interfaces/int25` | — | 25,920 | `inter_type25.py` |
| `engine/source/interfaces/int24` | — | 17,149 | `inter_type24.py` (M48) |
| `engine/source/interfaces/int07` | — | 16,645 | `inter_type7.py` |
| `engine/source/interfaces/int22` | — | 16,383 | `inter_type22.py` |
| `engine/source/interfaces/int20` | — | 8,266 | `inter_type20.py` |
| `engine/source/interfaces/inter3d` | — | 7,775 | — |
| `engine/source/interfaces/int16` | — | 6,764 | `inter_type16.py` |
| `engine/source/interfaces/int17` | — | 6,448 | `inter_type17.py` |
| `engine/source/interfaces/int11` | — | 6,486 | `inter_type11.py` |
| `engine/source/interfaces/int21` | — | 6,332 | `inter_type21.py` |
| `engine/source/interfaces/int15` | — | 5,424 | `inter_type15.py` |
| `engine/source/interfaces/int18` | — | 4,864 | `inter_type18.py` (M60) |
| `engine/source/interfaces/int10` | — | 3,589 | `inter_type10.py` |
| `engine/source/interfaces/int09` | — | 3,042 | `inter_type9.py` |
| `engine/source/interfaces/int23` | — | 2,925 | `inter_type23.py` |
| `engine/source/interfaces/int14` | — | 1,830 | `inter_type14.py` |
| `engine/source/interfaces/generic` | 17 | 2,362 | voxel bucket in `inter_type7.py` |
| `engine/source/interfaces/inter2d` | — | 714 | — |
| `engine/source/interfaces/intsort` | 103 | 35,155 | one voxel bucket |
| `engine/source/interfaces/interf` | ~40 | 32,919 | `contact/tracking.py` (partial) |
| `engine/source/interfaces/{ists,ists_q1np,shell_offset}` | 0 | **0** | — (empty upstream, verified) |
| `starter/source/interfaces/inter3d1` | — | **59,973** | — |
| `starter/source/interfaces/interf1` | — | 8,166 | `starter/initialization.py` |
| `starter/source/interfaces/intbuf` | — | 2,316 | — |
| `starter/source/interfaces/inter2d1` | — | 1,282 | — |
| `starter/source/interfaces/reader` | — | 1,380 | `input/keywords/interfaces.py` |
| `starter/source/interfaces/int01…int25` (22 dirs) | — | ~19,600 | `input/keywords/interfaces.py` |
| **Total** | | **≈ 305,000** | **~15,000** |

## Gap analysis

### Layer 1 — the broad phase is a general subsystem upstream, a per-type hack in the port

`intsort/` contains, **per interface type**, a matched family:
`iNNbuce.F` (bucket/voxel entry), `iNNsto.F`, `iNNtri.F` (triangulation),
`iNNoptcd.F`, `iNNmain_tri.F`, `iNNmain_crit_tri.F` (the *critical* — i.e.
narrow-phase-candidate — variant), `iNNtrc.F`, and for some types
`iNNtrivox.F` / `iNNxsave.F` / `iNNpen3.F` / `iNN_icrit.F`. Plus the generic
`check_sorting_criteria.F90`, `fill_voxel.F90`, `collision_mod.F`, and
`compare_cand.cpp` (a C++ comparator).

The port has **one voxel bucket, inside `inter_type7.py`**. Consequences:
- types 9, 10, 11, 20, 21, 22, 24, 25 have no broad phase of their own;
- the `*_crit_*` "narrow-phase candidate" pre-filter — which is what makes the
  O(N log N) behaviour possible — is absent everywhere;
- there is no colour/voxel bookkeeping (`generic/inter_color_voxel.F`,
  `inter_color_coarse_voxel.F`, `inter_cell_color.F`) and therefore no
  thread-parallel broad phase.

### Layer 2 — the interface driver

`interf/` holds `chkstfn3.F` (the per-cycle state check, and the file Phase 13's
`SPMD_EXCH_IDEL` work hangs off), `check_surface_state.F`,
`check_nodal_state.F`, `check_edge_state.F`, `check_active_elem_edge.F`,
`check_remote_surface_state.F`, `find_surface_inter.F`,
`find_edge_inter.F`, `find_surface_from_remote_proc.F`,
`find_edge_from_remote_proc.F`, `get_neighbour_surface*.F90`,
`get_hashtable_for_neighbour_segment.F90`, `get_convexity_normals.F90`,
`get_segment_edge.F`, `get_segment_criteria.F90`, and the MPI halves.

The port's `contact/tracking.py` covers part of this. The MPI halves are
Phase 13's; the **state-check** half belongs here because it reads the
deletion flag.

### Layer 3 — per-type gaps

- **`/INTER/TYPE4`, `/TYPE13`, `/TYPE19` have no port physics module.** TYPE19 is
  recorded at `docs/STATE.md` M157 ("`/INTER/TYPE19` 5-card multi-segment
  interface") as a **reader only**. `starter/source/interfaces/` has 22 `intNN`
  directories — int01, 02, 03, 05, 06, 07, 08, 09, 10, 11, 12, 14, 15, 16, 17,
  18, 20, 21, 22, 23, 24, 25 — and **no `int04`, `int13` or `int19`.**
  **That directory count is evidence, not proof**, and TYPE19 is the counter-
  example: **`ITYP == 19` is dispatched by the GENERIC reader
  `starter/source/interfaces/interf1/definter.F:190`** (reading `IGSTI`,
  `IGAP`, `IBAG`, `IDEL`, `INACTI`, `MODFR` into field indices 30-36, all
  `DEF_DEF=1000`; also `interf1/inintsub.F:319`). So TYPE19 is real, and the
  missing half is the *engine-side consumer of fields 30-36*, which Task P8.1
  must locate. TYPE4 and TYPE13 are unconfirmed in either direction and must be
  resolved **by whole-tree search**, not by counting directories.
- **`/INTER/LAGMUL`** (M54 parse only) — the Lagrange-multiplier contact family
  is upstream in `interf/lagmul/` (`engine/source/tools/lagmul/`,
  `LAG_MULTP`) and `starter/source/interfaces/`; the port has
  `engine/lagmul.py` (parse only) and refuses it under `-np` (OPEN_BUGS
  SPMD-2).
- **`/INTER/GUIDED_CABLE`**, `/INTER/HERTZ/TYPE17` (M160),
  `/INTER/LAGMUL/SPOTWELD|SURF|PART|BEAM` (M139), `/INTER/SUB_SURF` (M169) —
  readers exist, physics varies.
- **Friction** is well covered (M15: MFROT 1–4, IFQ 1–3). `friction_models.py`
  + `friction.py` exist.
- **Every type's `dt_int` stability contribution** — the accumulated contact
  stiffness that feeds the stable time step (`i7sti3.F` / `inter_type7.py`
  only, per M48 for TYPE24).

### Layer 4 — the missing Starter surface/segment builder

`starter/source/interfaces/inter3d1/` at **59,973 LOC** is the largest
directory in the whole Fortran tree after `engine/source/elements`. It builds
every segment, edge, box, convexity normal, adjacency and hash table that the
narrow phase consumes. The port does this inside
`starter/initialization.py` (2,106 lines) — i.e. a ~1:28 ratio on the single
largest subsystem in the solver, and the thing every interface type depends on.

## Wave graph

```
Wave 0  P8.0 interface reconciliation audit (serial — the checklist)
        P8.1 resolve TYPE4/13/19 by whole-tree search
        P8.2 interface-type census + corpus blocker ranking
   ── gate: every type and every layer classified; TYPE4/13/19 resolved
Wave 1 (parallel — new modules)
  Layer 1:  P8.3 broad-phase core (voxel/box/colour)
            P8.4 sorting criteria + the *_crit_* candidate pre-filter
            P8.5 collision handling (collision_mod.F + compare_cand.cpp)
  Layer 2:  P8.6 interface state checks (chkstfn3 and friends)
  Layer 4:  P8.7 segment/edge/box construction (inter3d1)
            P8.8 adjacency, hash tables, convexity normals
            P8.9 2-D interfaces (inter2d, inter2d1)
  Layer 3:  P8.10 the missing per-type narrow phases
   ── gate per module: analytic test + one oracle case
Wave 2 (serial)
  P8.11 wire all layers into engine.py + starter/initialization.py
  P8.12 dt_int stability for every type
  P8.13 numba mirror for the narrow phase
```

---

### Task P8.0: Interface reconciliation audit

**Fortran:** everything in §Scope.
**Files:** Create `tools/validation_data/interface_status.json`,
`tests/test_p8_reconciliation.py`.
**Interfaces:**
- Produces: one record per upstream interface routine, tagged with its
  `layer` (`broad`, `driver`, `narrow`, `starter`), its `type` (or `None` for
  shared), and its status. Phase 17 reports per-layer coverage from it.

- [ ] **Step 1** — write the failing test:

```python
def test_every_layer_is_populated_in_the_audit():
    st = json.loads(Path("tools/validation_data/interface_status.json").read_text())
    layers = {r["layer"] for r in st}
    assert layers == {"broad", "driver", "narrow", "starter"}, layers
    assert sum(1 for r in st if r["layer"] == "starter") > 400   # inter3d1 alone
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit. For each `inter_type*.py`, list its upstream routine set
  (`iNNfor*.F`, `iNNdst*.F`, `iNNsti*.F`, `iNNke*.F`, `iNNkeg*.F`, `iNNmain*.F`,
  `iNNvis*.F`, `iNNwini*.F`, `iNNfdg*.F`, `iNNpr*.F`) and mark each. A type
  whose port module is 400 Python lines against 17,000 Fortran lines has an
  audit result of "approximate", not "ported" — and that result is what this
  task exists to produce.
- [ ] **Step 4** — run → PASS; commit
  `docs(contact): per-routine interface audit across all four layers`.

---

### Task P8.1: Resolve TYPE4 / TYPE13 / TYPE19 — by search, not by counting

**Fortran:** `starter/source/interfaces/interf1/definter.F:190`
  (`ELSEIF(ITYP == 19) THEN` with `IGSTI`→index 30, `IGAP`→31, `IBAG`→32,
  `IDEL`→33, `INACTI`→34, `MODFR`→36, all `DEF_DEF=1000`),
  `interf1/inintsub.F:319` (`ELSEIF (NTY==19)`); the same `ITYP` ladder for the
  other types in `definter.F`; **the engine-side consumer of field indices
  30–36**, which does not yet exist in this plan and which this task must find.
**Files:** Modify `pyradioss/contact/__init__.py`,
`pyradioss/input/keywords/interfaces.py`, `docs/PORT_EXTENSIONS.md`;
Create `pyradioss/contact/inter_type19.py`; Create `tests/test_p8_inter_types.py`.
**Interfaces:**
- Produces: `contact.VALID_TYPES: frozenset[int]` with a citation per entry;
  `contact.GENERIC_READER_TYPES: frozenset[int]` — the types handled by
  `definter.F` rather than a dedicated `intNN` directory (at minimum `{19}`);
  `contact.DECLARED_ABSENT: frozenset[int]`; `inter_type19.forces`,
  `inter_type19.dt_int`.

- [ ] **Step 1** — write the failing test:

```python
def test_type19_is_a_real_type_read_by_the_generic_reader():
    from contact import VALID_TYPES, GENERIC_READER_TYPES
    assert 19 in VALID_TYPES
    assert 19 in GENERIC_READER_TYPES
    src = (paths.or_src()
           / "starter/source/interfaces/interf1/definter.F").read_text()
    assert "ELSEIF(ITYP == 19) THEN" in src

def test_type4_and_type13_are_either_real_or_declared_absent():
    from contact import VALID_TYPES, DECLARED_ABSENT
    for t in (4, 13):
        assert (t in VALID_TYPES) or (t in DECLARED_ABSENT)
```

- [ ] **Step 2** — run → FAIL (`VALID_TYPES` does not exist).
- [ ] **Step 3** — resolve, by searching the whole tree for each type:

  ```bash
  grep -rn "ITYP *= *4\b"  $OR_SRC --include=*.F --include=*.F90
  grep -rn "ITYP *= *13\b" $OR_SRC --include=*.F --include=*.F90
  grep -rn "ITYP *= *19\b" $OR_SRC --include=*.F --include=*.F90
  ```

  Three outcomes per type, each recorded **with the search that produced it**:
  1. **found** — it is real; add it to `VALID_TYPES` with the file and port its
     physics (Task P8.10);
  2. **reader only, no engine consumer** — parses but has no kernel; put it in
     `VALID_TYPES` *and* in `checks` as a **cited refusal** (Review Focus item 5:
     refuse loudly, never silently skip);
  3. **not found anywhere** — add it to `DECLARED_ABSENT` and to
     `docs/PORT_EXTENSIONS.md`, and make the port refuse it by name.

  **For TYPE19 the deliverable is the engine-side consumer of fields 30–36.**
  Locate it, port it into `inter_type19.py`, and delete the reader's silent
  fallback. `IBAG` (index 32) suggests a bag/container interface — read what
  consumes it before assuming it is a generic penalty interface.
- [ ] **Step 4** — run → PASS; the TYPE19 corpus decks' verdicts are re-measured
  and reported (they may legitimately become SKIPS or ERROR); commit
  `fix(contact): resolve INTER/TYPE4/13/19 by whole-tree search`.

### Task P8.2: Interface-type census and blocker ranking

**Fortran:** the 22 starter `intNN` readers and their engine counterparts; the
`/INTER` dispatch in `starter/source/interfaces/reader/`.
**Files:** Create `tools/validation_data/inter_type_status.json`,
`tests/test_p8_type_census.py`.
**Interfaces:**
- Produces: one record per type: `{"type": 24, "starter_loc": 1219,
  "engine_loc": 17149, "port": "contact/inter_type24.py",
  "narrow": "ported|partial|missing", "broad": "ported|partial|missing",
  "dt_int": True, "corpus_decks": 18, "sole_blocker_decks": 5}` — carrying the
  corpus numbers from `coverage_results_m41.json`'s `ranked_gaps`.

- [ ] **Step 1** — write the failing test:

```python
def test_census_carries_the_corpus_blocker_counts():
    st = json.loads(Path("tools/validation_data/inter_type_status.json").read_text())
    t24 = next(r for r in st if r["type"] == 24)
    assert t24["corpus_decks"] == 18 and t24["sole_blocker_decks"] == 5
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — build the table. The corpus counts are the **prioritisation
  input**: TYPE24 (18 decks, 5 sole blockers), INTER/LAGMUL (14/2),
  INTER/TYPE18 (7/0). Use them to order Wave 1's per-type tasks.
- [ ] **Step 4** — run → PASS; commit
  `feat(census): per-interface-type status with corpus blocker counts`.

---

### Task P8.3: The broad-phase core (layer 1)

**Fortran:** `engine/source/interfaces/generic/` — `inter_voxel_creation.F`,
  `inter_box_creation.F`, `inter_color_voxel.F`, `inter_color_coarse_voxel.F`,
  `inter_cell_color.F`, `check_coarse_grid.F`, `inter_check_sort.F`,
  `inter_prepare_sort.F`, `inter_sort.F`, `inter_struct_init.F`,
  `inter_init_component*.F90`, `inter_init_node_color.F`,
  `inter_component_bound.F90`, `inter_minmax_node.F`,
  `inter_count_node_curv.F`, `inter_curv_computation.F`,
  `inter_deallocate_wait.F`, `check_sorting_criteria.F90`, `fill_voxel.F90`.
**Files:** Create `pyradioss/contact/broadphase/` package
(`voxel.py`, `box.py`, `colour.py`, `sort.py`, `component.py`),
`tests/test_p8_broadphase.py`.
**Interfaces:**
- Produces:
  - `broadphase.voxel.Grid(extent, cell_size)` with `.assign(centroids) -> ids`,
    `.neighbours(ids) -> list[np.ndarray]`;
  - `broadphase.box.Box(min, max)` with `.cell_centres()`, `.candidate_pairs()`;
  - `broadphase.colour.colour(n) -> np.ndarray` — a **thread-safe node colouring**
    so the broad phase can be parallel;
  - `broadphase.sort.Argsort(keys) -> np.ndarray` — the `check_sorting_criteria.F90`
    ordering, whose *tie-breaking* is what makes the narrow phase reproducible;
  - `broadphase.component.components(graph) -> np.ndarray`.

- [ ] **Step 1** — write the failing tests:

```python
def test_voxel_grid_finds_every_true_neighbour_pair():
    rng = np.random.default_rng(0)
    pts = rng.random((5000, 3))
    g = broadphase.voxel.Grid(extent=(1,1,1), cell_size=0.1)
    ids = g.assign(pts)
    brute = np.argwhere(np.linalg.norm(pts[:,None]-pts[None,:], axis=-1) < 0.15)
    found = {tuple(sorted(p)) for i in ids for p in g.neighbours([i])[0][:0]}
    # the assertion that matters: the grid is a SUPERSET of the true pairs
    brute_pairs = {tuple(sorted(p)) for p in brute}
    grid_pairs = {tuple(sorted(p)) for i in range(len(pts))
                  for p in g.neighbours(ids[i])}
    assert brute_pairs <= grid_pairs

def test_sort_tie_breaking_is_deterministic():
    keys = np.array([1.0, 1.0, 0.0, 1.0])
    a = broadphase.sort.Argsort(keys)(np.arange(4))
    b = broadphase.sort.Argsort(keys)(np.arange(4))
    assert np.array_equal(a, b)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Two non-obvious requirements:**
  1. `check_sorting_criteria.F90` defines the ordering **and its tie-break**.
     Radioss's narrow phase visits pairs in a specific order, and floating-point
     summation in the force loop inherits that order. A different tie-break
     gives a bitwise-different answer with identical physics — which is
     precisely what the `PYRADIOSS_BACKEND=numpy` md5 comparisons will catch
     and what makes matching it necessary.
  2. The voxel grid must return a **superset** of the true pairs, not an exact
     set — the narrow phase filters. The test asserts the superset property, so
     a "clever" exact filter cannot pass.
- [ ] **Step 4** — run → PASS; commit
  `feat(contact): the shared broad phase (voxel/box/colour/sort/component)`.

---

### Task P8.4: The `*_crit_*` candidate pre-filter

**Fortran:** the per-type critical-candidate families in `intsort/`:
  `i11buce_crit.F`, `i11main_crit_tri.F`, `i20buce_crit.F`,
  `i20main_crit_tri.F`, `i21buce_crit.F`, `i21main_crit_tri.F`, `i21_icrit.F`,
  and the `iNNbuce.F` / `iNNmain_tri.F` pairs they filter.
**Files:** Create `pyradioss/contact/candidate.py`, `tests/test_p8_candidate.py`.
**Interfaces:**
- Produces: `candidate.CritCrit(type, voxels) -> np.ndarray` (bool mask of
  voxels that can contain a narrow-phase candidate), and per-type
  `iNN_crit_*` functions.

- [ ] **Step 1** — write the failing test — the soundness property:

```python
@pytest.mark.parametrize("type", [11, 20, 21])
def test_candidate_filter_never_drops_a_real_contact(type):
    """Every pair the brute-force narrow phase finds must survive the filter."""
    pairs_brute = _brute_force_pairs(type, n=400, seed=0)
    kept = candidate.CritCrit(type, _voxels(type))(pairs_brute)
    assert set(pairs_brute[kept]) == set(pairs_brute)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. This test is the whole specification: the filter
  is a performance optimisation and **must be sound** (never drop a real
  contact). If a port's filter ever drops one, contact silently stops working
  in some configuration — the worst class of bug in this phase.
- [ ] **Step 4** — run → PASS; commit `feat(contact): per-type critical candidate pre-filter`.

---

### Task P8.5: Collision handling

**Fortran:** `engine/source/interfaces/intsort/collision_mod.F`,
  `engine/source/interfaces/intsort/compare_cand.cpp`,
  `engine/source/interfaces/intsort/fill_voxel.F90`.
**Files:** Create `pyradioss/contact/collision.py`, `tests/test_p8_collision.py`.
**Interfaces:**
- Produces: `collision.resolve(overlaps) -> assignment` — a deterministic
  ownership assignment when two interfaces claim the same segment/edge.

- [ ] **Step 1** — write the failing test:

```python
def test_collision_assignment_is_a_partition_and_is_deterministic():
    segs = np.array([[0], [1], [2], [3]])
    claim = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 1], [3, 3, 1]])  # seg, iface, prio
    a1 = collision.resolve(claim); a2 = collision.resolve(claim)
    assert np.array_equal(a1.owner, a2.owner)
    assert set(a1.owner) == set(range(4))          # a partition
    assert (a1.priority[a1.owner == 0] == 1).all() # highest priority won
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `compare_cand.cpp` is C++ in upstream; the
  comparison is the authority for the tie-break, and the port must reproduce it
  **bitwise** (it is integer/rational, so this is achievable). Read the C++.
- [ ] **Step 4** — run → PASS; commit `feat(contact): collision ownership with the upstream tie-break`.

---

### Task P8.6: The interface state checks (layer 2)

**Fortran:** `engine/source/interfaces/interf/` — `chkstfn3.F` (the per-cycle
  check; the file `docs/OPEN_BUGS.md` SPMD-2 cites for `SPMD_EXCH_IDEL`),
  `check_surface_state.F`, `check_nodal_state.F`, `check_edge_state.F`,
  `check_active_elem_edge.F`, `check_remote_surface_state.F`,
  `count_nb_elem_edge.F`, `get_segment_criteria.F90`,
  `get_segment_edge.F`, `get_segment_interface_id.F`.
**Files:** Create `pyradioss/contact/state.py`, `tests/test_p8_state.py`.
**Interfaces:**
- Consumes: the `off` flag from Phase 7 Task P7.0.
- Produces: `state.check(model, interfaces, cycle) -> StateReport` with
  `StateReport.removed: dict[int, np.ndarray]` (segment ids dropped per
  interface), `StateReport.dead_nodes: np.ndarray`,
  `StateReport.warnings: list[str]`.

- [ ] **Step 1** — write the failing test — Review Focus item 5's cousin:

```python
def test_a_deleted_parent_element_drops_its_segments():
    """Open_BUGS SPMD-2 failure case A, in serial: a TYPE2 tie on a failing
    shell must go inactive on every holder."""
    model = _model_with_type2_tie_on_a_failable_shell()
    interface = model.interfaces[0]
    interface._cycle(8)                       # the cycle % 8 poll
    model.groups["shells"].state["off"][interface.main_parent[0]] = 1
    rep = state.check(model, model.interfaces, cycle=9)
    assert interface.main_parent[0] in rep.removed.get(interface.id, [])
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **`chkstfn3.F` is the single file Phase 13 needs
  for the SPMD work**, so it must be ported as a *pure function of state* with
  no communicator inside — Phase 13 wraps it, it does not rewrite it. Enforce
  that: `state.py` must not import anything from `pyradioss.spmd`, and a test
  asserts it.
- [ ] **Step 4** — run → PASS; commit
  `feat(contact): the interface state-check driver (chkstfn3 family)`.

---

### Task P8.7: Segment / edge / box construction (layer 4, the missing Starter half)

**Fortran:** `starter/source/interfaces/inter3d1/` — **59,973 LOC**, the
  largest subsystem in the solver. Build it directory by directory.
**Files:** Create `pyradioss/starter/surfaces/` package
(`segments.py`, `edges.py`, `boxes.py`, `adjacency.py`, `hash.py`,
`convexity.py`, `build.py`), `tests/test_p8_surfaces.py`.
**Interfaces:**
- Consumes: the port's `Surface`, `Line`, `Part` entities.
- Produces:
  - `surfaces.build(model, surfaces, log) -> SurfaceIndex` with
    `SurfaceIndex.segments: dict[int, np.ndarray]` (shape `(m, 4)`),
    `SurfaceIndex.segment_parent: dict[int, np.ndarray]`,
    `SurfaceIndex.edges: dict[int, np.ndarray]`,
    `SurfaceIndex.edge_parent: dict[int, np.ndarray]`,
    `SurfaceIndex.boxes: np.ndarray`,
    `SurfaceIndex.convexity_normals: dict[int, np.ndarray]`.

- [ ] **Step 1** — write the failing test:

```python
def test_segments_of_a_quad_surface_are_its_four_edges_in_upstream_order():
    s = _single_quad_surface()
    idx = surfaces.build(_model_with(s), [s], MessageLog())
    assert idx.segments[s.id].shape == (4, 4)
    # upstream i7sti3.F orders segments 1,2,3,4 of the quad in a fixed
    # orientation; the port must match it exactly (this is the order the
    # narrow phase visits).
    assert np.array_equal(idx.segments[s.id][:, 0], [0, 1, 3, 2])

def test_a_negative_surface_id_flips_the_normals():
    s = _single_quad_surface(user_id=-1)
    idx = surfaces.build(_model_with(s), [s], MessageLog())
    assert np.allclose(idx.convexity_normals[s.id],
                       -surfaces.build(_model_with(_as_positive(s)), [s],
                                       MessageLog()).convexity_normals[s.id])
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement **directory by directory**, one task-sized commit
  each. **The segment ordering is not arbitrary**: it determines the force
  summation order, so it determines the last bits of every contact result. Read
  `i7sti3.F`'s segment loop and reproduce its index arithmetic verbatim. The
  negative-id normal flip is the port's existing convention
  (`PORTING_GUIDE.md:43`) — keep it and test it.
- [ ] **Step 4** — run → PASS; **`PYRADIOSS_BACKEND=numpy` T01 md5 unchanged**
  for every example with contact — this is the load-bearing test of the whole
  task; commit per subdirectory.

---

### Task P8.8: Adjacency, hash tables, convexity normals

**Fortran:** `interf/get_hashtable_for_neighbour_segment.F90`,
  `get_neighbour_surface.F90`, `get_neighbour_surface_from_remote_proc.F90`,
  `get_convexity_normals.F90`, `get_segment_edge.F`,
  `find_surface_inter.F`, `find_edge_inter.F`, `find_surface_from_remote_proc.F`,
  `find_edge_from_remote_proc.F`, `check_active_elem_edge.F`.
**Files:** Create `pyradioss/starter/surfaces/adjacency.py`,
`pyradioss/starter/surfaces/hash.py`, `pyradioss/starter/surfaces/convexity.py`.
**Interfaces:**
- Produces: `adjacency.SegmentAdjacency` (segment→segment on the same surface),
  `hash.SegmentHash` (a stable hash from node ids to segment ids),
  `convexity.normals(segments, coords) -> np.ndarray`.

- [ ] **Step 1** — write the failing test:

```python
def test_segment_hash_is_order_independent_but_deterministic():
    h = hash.SegmentHash()
    a = h.add(np.array([[0, 1, 2, 3]]))[0]
    b = h.add(np.array([[3, 2, 1, 0]]))[0]      # reversed
    assert h.key(a) == h.key(b)

def test_convexity_normal_is_outward_for_a_convex_box():
    segs = _box_surface_segments()
    n = convexity.normals(segs, _box_coords())
    assert np.allclose(n.sum(0), 0.0, atol=1e-9)      # closed surface
    assert (np.linalg.norm(n, axis=1) > 0).all()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The **order-independent segment hash** is what
  makes `get_neighbour_segment` correct when two segments share a node pair in
  opposite order; a hash that is order-*dependent* silently misses neighbours
  and contact leaks through a crack. `find_*_from_remote_proc` is Phase 13's
  to call — port the pure lookup here and leave the communicator out.
- [ ] **Step 4** — run → PASS; commit `feat(contact): adjacency, segment hashing and convexity normals`.

---

### Task P8.9: 2-D interfaces

**Fortran:** `engine/source/interfaces/inter2d/` (714 LOC),
  `starter/source/interfaces/inter2d1/` (1,282 LOC).
**Files:** Create `pyradioss/contact/inter2d.py`, `tests/test_p8_inter2d.py`.
**Interfaces:**
- Produces: `inter2d.forces(...)`, `inter2d.stiffness(...)`,
  `inter2d.build(...)` — the planar (in-plane) interface family.

- [ ] **Step 1** — write the failing test:

```python
def test_2d_penalty_contact_pushes_a_penetrating_node_back_out():
    g = _planar_segment(); n = _node_penetrating_by(0.01)
    f = inter2d.forces(g, n, dt=1e-6)
    assert f[1] > 0.0        # normal force opposes penetration
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement from `inter2d/` directly. Small task; if the port
  already covers it under a different name, this task is a **rename plus a
  citation fix** and says so.
- [ ] **Step 4** — run → PASS; commit `feat(contact): the 2-D interface family`.

---

### Task P8.10: The missing per-type narrow phases

**Fortran:** per type, `iNNfor*.F`, `iNNdst*.F`, `iNNsti*.F`, `iNNke*.F`,
  `iNNkeg*.F`, `iNNvis*.F`. By the P8.0 audit, the types with the largest
  partial/missing routine sets.
**Files:** Create `pyradioss/contact/inter_type<NN>.py`; Create
  `tests/test_p8_inter_type<NN>.py` **per type**.
**Interfaces:**
- Consumes: `broadphase`, `candidate`, `state`, `surfaces` from Tasks P8.3–P8.9.
- Produces: `init_interfaces(model, log)`, `transfer_forces(model, cycle, dt)`
  — the per-type narrow phase and force.

- [ ] **Step 1** — write the failing test, always the same four for a penalty
  interface:

```python
def test_<type>_force_is_antisymmetric_in_its_pair():
    g = _pair(<type>, penetration=1e-4)
    f = _transfers(g)
    assert np.allclose(f.fint.sum(0), 0.0, atol=1e-12 * g.force_scale)

def test_<type>_does_no_work_outside_penetration():
    g = _pair(<type>, penetration=-1e-4)       # separated
    assert _transfers(g).fint.sum() == 0.0

def test_<type>_books_its_work_in_the_contact_ledger():
    g = _pair(<type>, penetration=1e-4)
    assert _transfers(g).contact_work != 0.0

def test_<type>_dt_int_is_the_gap_stiffness_over_velocity():
    g = _pair(<type>, penetration=1e-4)
    assert g.dt_int == pytest.approx(g.stiffness * g.length / g.velocity_scale)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The antisymmetry test is Newton's third law and
  it is the one most likely to fail** — upstream's edge-edge and surface-edge
  codes are not always exactly antisymmetric (their `iNNpen3.F` overlap
  handling redistributes across more than two nodes). If antisymmetry fails,
  **compare with upstream and mirror the asymmetry**, do not symmetrise.
- [ ] **Step 4** — run → PASS; one oracle case per type; commit per type.

---

### Task P8.11: Wire all four layers together

**Fortran:** `starter/source/interfaces/reader/`,
  `engine/source/interfaces/interf/`, the `contact` driver in
  `engine/source/engine/resol.F`.
**Files:** Modify `pyradioss/contact/__init__.py`,
`pyradioss/engine/engine.py`, `pyradioss/starter/initialization.py`.
**Interfaces:**
- Produces: `contact.transfer_forces(model, cycle, dt) -> ContactWork`,
  `contact.ContactWork` with `force: np.ndarray`, `moment: np.ndarray`,
  `work: float`, `friction_work: float`.

- [ ] **Step 1** — write the failing test:

```python
def test_contact_layers_compose_end_to_end():
    s, e = make_deck("CRASH", _crash_starter_deck(), _crash_engine_deck())
    # run both, then assert the contact ledger balances to the energy report
    assert _energy_balance(run(e))["contact"] < 1e-6
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — wire. **The energy test is the point** (the domain rule:
  "every new force path must book its work in the EN ledger"). The order is
  upstream's: broad phase → candidate filter → narrow phase → contact work at
  the **leapfrog-consistent midstep velocity** (the port's existing convention
  per `README.md`).
- [ ] **Step 4** — run → PASS; energy-balance tests green; commit
  `feat(contact): compose the four interface layers in the engine`.

---

### Task P8.12: `dt_int` for every type

**Fortran:** each type's `iNNsti*.F` stiffness/gap setup and the accumulation
  into the critical time step (`engine/source/time_step/`); `docs/STATE.md` M48
  records TYPE24's `dt_int` as the model for the others.
**Files:** Modify `pyradioss/contact/inter_type*.py`,
`pyradioss/engine/mass_scaling.py`.
**Interfaces:**
- Produces: `contact.dt_int(model) -> np.ndarray` (per node), aggregating every
  interface type's contribution.

- [ ] **Step 1** — write the failing test:

```python
def test_two_interfaces_sum_their_dt_contribution():
    m = _model_with_two_identical_interfaces()
    one = _model_with_one_interface()
    assert contact.dt_int(m)[0] == pytest.approx(2 * contact.dt_int(one)[0],
                                                rel=1e-9)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. `dt_int` is a **stability** input, so a
  conservative port is safe and an anti-conservative one diverges. Where the
  port is *less* conservative than upstream, that is a bug, not a performance
  choice — the test in Step 1 uses `rel=1e-9`, i.e. it demands exactness.
- [ ] **Step 4** — run → PASS; commit `feat(contact): dt_int for every interface type`.

---

### Task P8.13: numba mirror for the narrow phase

**Fortran:** none. **Files:** Create `pyradioss/accel/jit_kernels/contact.py`.
**Interfaces:**
- Consumes: `contact` per-type force leaves.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("type", [7, 11, 24])
def test_narrow_phase_mirror_is_bitwise(type):
    ref = accel.reference(f"contact_narrow_{type}")
    assert accel.get(f"contact_narrow_{type}")(*ref.inputs).tobytes() \
           == ref.expected.tobytes()
```

- [ ] **Step 2** — run → SKIP until P0.7, then FAIL.
- [ ] **Step 3** — implement. **Mirroring must preserve the pair-visit order**
  from Task P8.3's sort — that is what makes bitwise equality achievable at all.
  If a type's narrow phase cannot be mirrored bitwise because it uses a
  `cKDTree` query, exclude it from `auto` and record the measurement.
- [ ] **Step 4** — run → PASS; T01 md5 unchanged for the contact examples; commit
  per type.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P8.0** — the audit covers all four layers and finds ≥400 starter-layer rows
  (which is what `inter3d1` alone contains).
- **P8.1** — each of the three types has a recorded **search**, not a directory
  count. TYPE19 is resolved via `definter.F:190` and its engine-side consumer
  is located and ported, not guessed.
- **P8.3** — the grid is a **superset** of true pairs, and the sort tie-break
  matches `check_sorting_criteria.F90`.
- **P8.4** — the filter is proven **sound** (never drops a real contact) on a
  brute-force comparison.
- **P8.5** — the tie-break matches `compare_cand.cpp` exactly.
- **P8.6** — `state.py` imports nothing from `pyradioss.spmd`; this is a hard
  requirement for Phase 13.
- **P8.7** — the segment ordering is transcribed from `i7sti3.F`, and the T01
  md5 for every contact example is unchanged.
- **P8.8** — the hash is order-independent and the closed-surface normal sum is
  zero.
- **P8.10** — antisymmetry is **compared with upstream**, not imposed; where
  upstream is asymmetric, the port is too, with the citation.
- **P8.12** — `dt_int` is exact, not merely conservative.
- **P8.13** — bitwise, or an exclusion with a measurement.

## Parallelisation

- **Wave 0** — P8.0 serial. P8.1 and P8.2 parallel with it.
- **Wave 1** — 10 tasks. **The dependency inside Wave 1 is real**: P8.4, P8.6,
  P8.10 all consume P8.3's `broadphase`; P8.10 also consumes P8.7–P8.9. So
  Wave 1 is two sub-waves:
  - **1a** (3 agents): P8.3 → P8.4, P8.5, P8.6 fan out.
  - **1b** (4 agents): P8.7, P8.8, P8.9, P8.10 in parallel; P8.10 starts once
    1a lands.
- **Wave 2** — serial.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p8_reconciliation.py tests/test_p8_broadphase.py tests/test_p8_candidate.py
python tools/validate_vs_fortran.py --mode parity --cases RD-E-1000 --out tools/validation_data/parity_p8.json
python tools/validate_vs_fortran.py --mode coverage --out tools/validation_data/coverage_p8.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area engine/source/interfaces
```

The phase reviewer reports, as numbers:

1. the coverage delta against `coverage_results_m41.json`, with the specific
   blockers closed — `INTER/TYPE24` was 18 decks / 5 sole-blockers,
   `INTER/LAGMUL` 14/2, `ALE/BCS` 12/7 (the last is Phase 11's),
   `SHEL16` 12, `QUAD` 10/5 (Phase 2's);
2. that `PYRADIOSS_BACKEND=numpy` T01 md5 for every bundled example is
   **unchanged** by this phase except where a commit message says otherwise with
   a parity number;
3. that the energy-balance tests are green.