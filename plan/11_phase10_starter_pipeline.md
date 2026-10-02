# Phase 10 — Starter pipeline

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `starter/source/{model,initial_conditions,restart,spmd,elements,
starter,system,general_controls}` a literal port. The Starter is where a model
becomes a solver state; every gap here is a **coverage blocker** (a deck the
Starter refuses is a deck that can never run) and a **parity blocker** (a
Starter that builds a different model produces a different run from cycle 1).

**Architecture:** the Starter has four separable concerns, and the port has
one module (`starter/initialization.py`, 2,106 lines) plus
`starter/starter.py` (407) for all four:

| Concern | Upstream | LOC | Port |
|---|---|---:|---|
| Sets and groups (`/GRNOD`, `/SURF`, `/LINE`, `/SET`) | `starter/source/model/sets/` (41 files) + `starter/source/groups/` (41) | 26,460 | `input/keywords/` + `initialization.py` |
| Model assembly (parts, meshes, boxes, transformations, submodels) | `model/{assembling,box,group,mesh,remesh,submodel,transformation}` | 14,120 | `initialization.py` |
| Initial conditions | `starter/source/initial_conditions/` (8 dirs, 51 files) | 14,218 | `starter/inista.py`, `initemp.py`, `inivel.py`, `initialization.py` |
| Restart / domain-split serialization | `starter/source/restart/ddsplit/` (**165 files**) | 45,501 | `starter/restart.py` (pickle) + `restart_binary.py` |
| SPMD decomposition (Starter half) | `starter/source/spmd/` (37 files) | 18,608 | `pyradioss/spmd/domdec.py` |
| **Total** | | **≈ 118,900** | **~4,000** |

The two largest ratios: `restart/ddsplit` at **165 files / 45,501 lines**
against a pickle, and `model/sets` at **26,460 lines** against a shared
`resolve_*` block.

**Tech stack:** NumPy; the restart format becomes binary (Phase 12's
`wrrestp.F` counterpart), which is the single largest format-fidelity
deliverable in the program.
**Spec:** `plan/00_ORCHESTRATION.md` §10 item 4 (restart continuity).
**Entry criteria:** Phase 8 exit gate (surfaces exist),
Phase 9 exit gate (properties attach), Phase 0 oracle.
**Band: L.**

---

## Scope

| Concern | Upstream | Files | LOC | Port |
|---|---|---:|---:|---|
| Sets and groups | `starter/source/model/sets/` + `starter/source/groups/` | 82 | 26,460 | `input/keywords/` + `starter/initialization.py` |
| Model assembly | `starter/source/model/{assembling,box,group,mesh,remesh,submodel,transformation}` | ~55 | 14,120 | `starter/initialization.py` |
| Initial conditions | `starter/source/initial_conditions/` (8 dirs) | 51 | 14,218 | `starter/{inista,initemp,inivel}.py`, `initialization.py` |
| Restart / domain-split | `starter/source/restart/ddsplit/` | 165 | **45,501** | `starter/restart.py` (pickle) + `restart_binary.py` |
| SPMD decomposition (Starter) | `starter/source/spmd/` | 37 | 18,608 | `pyradioss/spmd/domdec.py` |
| The driver | `starter/source/starter/` (9 files) | 9 | 17,792 | `starter/starter.py` (407) |
| System | `starter/source/system/` | 16 | 6,631 | `MessageLog`, `input/units.py` |
| General controls | `starter/source/general_controls/` + `engine/source/general_controls/` | 42 | 9,164 | `input/engine_keywords.py` (2,107) |
| **Total** | | **457** | **≈ 152,494** | **~4,000** |

Per-directory LOC, measured 2026-10-02: `sets` 15,057, `box` 5,109,
`submodel` 3,161, `remesh` 2,553, `assembling` 1,360, `transformation` 1,101,
`group` 912, `mesh` 815; `restart/ddsplit` 45,501; `spmd` 5,869 + `spmd/node`
1,844 + `spmd/tools` 359 + `spmd/domain_decomposition` 10,536.

## Gap analysis

### 1. `restart/ddsplit` — 165 files, 45,501 LOC

Every persisted array gets a writer and a reader here: `c_bufel.F` (element
buffers), `c_crkadd.F`/`c_crkxfem.F` (XFEM cracks), `c_dampvrel.F`,
`c_drape.F`, `c_eig.F`, `c_elig3d.F` (IGA control nets), and ~150 more. The
port pickles the model object.

That is a **deliberate, documented** decision (`PORTING_GUIDE.md:47`: "pickle
instead of binary"), and for a *literal* port it must change. It is also the
task with the highest blast radius: a restart file written by the old code must
either still load or be explicitly invalidated with a version tag.

### 2. `model/sets` — the corpus's #1–#10 blocker family

`PORTING_GUIDE.md:43` records the M37 "groups-sets builder" closing the ranked
gaps 1–10, and lists the readers it covers: `hm_lecgrn.F` (node groups),
`hm_surfnod.F`, `hm_grogronod.F` (group-of-groups), `hm_elngr*.F`,
`hm_read_surfsurf.F`, `hm_surfgr2`/`surftage`, `linedge.F`, `hm_lines_of_lines.F`.
`model/sets/` is **15,057 lines in one directory** — the port has no dedicated
module for it.

### 3. `model/box` — 5,109 lines

`/BOX`, `/BOX/BOX`, `/SURF/PLANE`, `/SURF/ELLIPSE`, `/LOAD/PCYL` and the
analytic-surface geometry live here (M132). The port has partial support. The
volume is mostly **analytic surface evaluation**, which is exact-able.

### 4. `model/remesh` — 2,553 lines

`/REMESH`, the Starter's mesh-refinement set-up, plus the consumers in Phase 11
(ALE/Euler remesh). The port's `solid_remesh.py` (Phase 2 Task P2.7) has the
tensor remap but not the mesh machinery.

### 5. `model/submodel` — 3,161 lines

`/SUBMODEL` and `/ENDSUB` (M87) — a container architecture with its own
scoping, transformations and units. The port has readers (M87) and the
`LSUBMODEL` argument appears throughout the readers, but the scoping model is a
Starter concern. Two corpus decks are blocked on `SUBMODEL`/`ENDSUB`.

### 6. `spmd/` — 18,608 LOC on the Starter side alone

`spmd/domain_decomposition/` (23 files, 10,536), `spmd/node` (3, 1,844),
`cpp_reorder_elements.cpp`, `cpp_split_tool.cpp`, `prepare_split_i25e2e.F`,
`split_cfd_solide.F`, `igrsurf_split.F`, `globvars.F`, `domdec2.F`.
The port has `spmd/domdec.py` (weighted recursive bisection per
`initwg.F`/`domdec1.F`/`domdec2.F`, frontier sharing, per-domain restart files).
Phase 13 owns the engine half; this phase owns the **Starter half plus the
`cpp_*` element-reordering**, which affects the *element ordering in the restart
file* and therefore bitwise reproducibility under `-np`.

### 7. Loose threads

- `starter/source/starter/` (9 files, 17,792 LOC) — the driver: `lectur.F`,
  `lecfun.F`, the initial-condition passes, the mass/energy pre-check,
  the message assembly. The port has `starter/starter.py` (407).
- `starter/source/general_controls/` (37 files, 7,742) — `/RUN`, `/VERS`,
  `/PRINT`, `/CHECKSUM`, `/ALTDOCTAG`, `/DEBUG`, `/ENG/STATE`, `/ENG/DYNAIN`
  and the rest of the engine-control card set. Phase 12's listing work depends
  on it.
- `starter/source/system/` (16 files, 6,631) — messages, the log, unit handling,
  the random-number generator.

## Wave graph

```
Wave 0  P10.0 Starter reconciliation audit (serial — the checklist)
   ── gate: every starter routine classified
Wave 1 (parallel — new modules)
  P10.2 sets/groups as a dedicated module (26k LOC)
  P10.3 boxes and analytic surfaces (5k LOC)
  P10.4 submodel scoping (3k LOC)
  P10.5 initial conditions (14k LOC, 8 dirs)
  P10.6 model assembling + mesh + transformations
  P10.7 the Starter driver: lectur/lecfun/message assembly (18k LOC)
  P10.8 general controls card set (8k LOC)
   ── gate per module: analytic test + one oracle case
Wave 2 (serial)
  P10.8 the binary restart/ddsplit format (45k LOC, 165 files) — the big one
  P10.9 restart compatibility (version tag + migration)
  P10.10 SPMD Starter half + element reordering
  P10.11 coverage re-measure
```

---

### Task P10.0: Starter reconciliation audit

**Fortran:** every `starter/source/` directory except `materials/`,
`interfaces/`, `properties/` (Phases 6, 8, 9).
**Files:** Create `tools/validation_data/starter_status.json`,
`tests/test_p10_reconciliation.py`.
**Interfaces:**
- Produces: one record per starter routine with `area` (`sets`, `assembly`,
  `initcond`, `restart`, `spmd`, `driver`, `controls`, `system`) and status.

- [ ] **Step 1** — write the failing test:

```python
def test_every_starter_area_is_populated():
    st = json.loads(Path("tools/validation_data/starter_status.json").read_text())
    assert {r["area"] for r in st} == {
        "sets","assembly","initcond","restart","spmd","driver","controls","system"}
    assert sum(1 for r in st if r["area"] == "restart") > 150   # ddsplit alone
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit. For `restart/ddsplit` specifically, produce the
  **array inventory**: the 165 files between them write every persisted array,
  and the binary restart format of Task P10.9 needs that inventory as its
  specification. Emit it as `tools/validation_data/restart_fields.json`:
  `{"array": "GBUF%SIG", "writer": "c_sig.F", "reader": "r_sig.F",
    "dtype": "float32", "shape_fn": "n_elem_groups"}`.
- [ ] **Step 4** — run → PASS; commit
  `docs(starter): per-routine audit + the restart array inventory`.

---

### Task P10.1: Sets and groups as a dedicated module

**Fortran:** `starter/source/model/sets/` (41 files, 15,057 LOC) +
  `starter/source/groups/` (41 files, 11,443) — `hm_lecgrn.F`, `hm_lecgre.F`,
  `hm_surfnod.F`, `hm_grogronod.F`, `hm_elngr.F`, `hm_elngrr.F`,
  `hm_linengr.F`, `hm_lines_of_lines.F`, `hm_read_grpart.F`, `hm_read_lines.F`,
  `hm_read_surf.F`, `hm_bigsbox.F`, `hm_admlist.F`, `hm_admlistcnt.F`,
  `hm_prelecgrns.F`, `hm_grogro.F`, `elegror.F`, `elegror_seatbelt.F`,
  `check_surf.F`, `groups_get_elem_list.F`, `groups_get_nentity.F`.
**Files:** Create `pyradioss/starter/sets/` package
(`node_groups.py`, `element_groups.py`, `surfaces.py`, `lines.py`,
`group_of_groups.py`, `seatsbelt.py`, `resolver.py`, `census.py`),
`tests/test_p10_sets.py`.
**Interfaces:**
- Consumes: parsed `NodeGroup`, `Surface`, `Line`, `EntityGroup`, `Part`.
- Produces: `sets.resolver.build(model, log) -> Sets` with
  `Sets.node_groups: dict[int, np.ndarray]`,
  `Sets.element_groups: dict[str, np.ndarray]`,
  `Sets.surfaces: dict[int, np.ndarray]` (nodes),
  `Sets.lines: dict[int, np.ndarray]` (edges),
  `Sets.group_of_groups: dict[int, np.ndarray]`,
  and — critically —
  `Sets.unresolved: list[UnresolvedRef]` with a **reason per entry**.

- [ ] **Step 1** — write the failing test:

```python
def test_group_of_groups_resolves_transitively_with_a_fixpoint():
    g = _model_with_chain(a=["b", "c"], b=["c"], c=["leaf"])
    s = sets.resolver.build(g, MessageLog())
    assert set(s.node_groups["a"]) == set(s.node_groups["leaf"])

def test_every_unresolved_reference_carries_a_reason():
    g = _model_with_missing_targets()      # /GRNOD/D nogroup9
    s = sets.resolver.build(g, MessageLog())
    assert s.unresolved
    for u in s.unresolved:
        assert u.reason and u.keyword and u.card
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, reader by reader. **Three upstream conventions
  are easy to lose and must be transcribed** (they are recorded in
  `PORTING_GUIDE.md:43` for the port's existing work, and this task verifies
  them rather than re-deriving them):
  1. `/GRNOD/GRNOD` resolves by **fixpoint iteration**, not recursion, and
     **negative ids win** in a conflict;
  2. cycle detection is **an error with a named message**, not an infinite loop;
  3. `/LINE/EDGE` selects **border edges only** (`linedge.F` semantics) — a
     `/LINE` over an interior edge set is a different thing.
- [ ] **Step 4** — run → PASS; **the `SUBMODEL`/`ENDSUB`, `SUBDOMAIN`,
  `GRNOD/NODENS` and `XREF` corpus blockers are re-measured** (5, 4, 1, 1 decks
  at baseline); commit per reader group.

---

### Task P10.2: Boxes and analytic surfaces

**Fortran:** `starter/source/model/box/` (5,109 LOC) — the `/BOX`,
  `/BOX/BOX`, `/SURF/PLANE`, `/SURF/ELLIPSE` and analytic-cylinder/sphere
  geometry; consumers in `starter/source/loads/` (`/LOAD/PCYL`) and
  `engine/source/constraints/general/rwall`.
**Files:** Create `pyradioss/starter/geometry.py`,
`tests/test_p10_geometry.py`.
**Interfaces:**
- Produces: `geometry.Box(min, max)`,
  `geometry.analytic_surface(spec) -> Surface` with `.distance(p) -> np.ndarray`,
  `.normal(p) -> np.ndarray`, `.contains(p) -> np.ndarray`,
  `geometry.mesh(nodes, surfaces) -> np.ndarray` (the surface triangulation,
  consuming Phase 8 Task P8.7's `surfaces.build`).

- [ ] **Step 1** — write the failing test:

```python
def test_plane_surface_distance_and_normal_are_exact():
    s = geometry.analytic_surface({"type": "PLANE", "p": [0, 0, 1],
                                    "n": [0, 0, 2]})
    p = np.array([[0.0, 0.0, 3.0], [1.0, 0.0, -1.0]])
    assert np.allclose(np.abs(s.distance(p)), [2.0, 2.0])
    assert np.allclose(np.abs(s.normal(p)), [[0, 0, 1], [0, 0, 1]])

def test_ellipse_contains_its_own_boundary():
    s = geometry.analytic_surface({"type": "ELLIPSE", "a": 2.0, "b": 1.0,
                                   "c": [0, 0, 0]})
    th = np.linspace(0, 2*np.pi, 50)
    pts = np.stack([2*np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    assert s.contains(pts).all()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **`/SURF/PLANE` is infinite** and is a coverage
  blocker (1 deck), so the port needs both the analytic distance/normal *and*
  the bounding box used by the broad phase (Phase 8 Task P8.3). The analytic
  form is exact-able, so the analytic tests above must be to the last bit.
- [ ] **Step 4** — run → PASS; the `SURF/PLANE` and `BOX/BOX` blockers
  re-measured; commit `feat(starter): boxes and analytic surfaces`.

---

### Task P10.3: `/SUBMODEL` scoping

**Fortran:** `starter/source/model/submodel/` (3,161 LOC) — the
  `/SUBMODEL`/`/ENDSUB` container, `LSUBMODEL` propagation, per-submodel
  units and transformations.
**Files:** Create `pyradioss/starter/submodel.py`,
`tests/test_p10_submodel.py`.
**Interfaces:**
- Produces: `submodel.Scope` with `.enter(block)`, `.exit(block)`,
  `.lsubmodel: int`, `.transform(index) -> np.ndarray`,
  `.units(index) -> UnitSystem`, and `submodel.scoped(model, blocks, log) -> list[Block]`.

- [ ] **Step 1** — write the failing test:

```python
def test_submodel_scopes_nodes_transforms_and_units():
    b = _blocks_with_submodel(nodes=[1, 2], tf="/TRANSFORM/POS", unit=3)
    scoped = submodel.scoped(_model(), b, MessageLog())
    assert all(x.lsubmodel == 0 for x in scoped[:3])
    assert any(x.lsubmodel == 1 for x in scoped)

def test_a_submodel_block_without_endsub_is_refused():
    with pytest.raises(UnbalancedSubmodelError, match="ENDSUB"):
        submodel.scoped(_model(), ["/SUBMODEL", "/NODE"], MessageLog())
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The `LSUBMODEL` argument already threads through
  the port's readers** (`hm_get_intv(..., LSUBMODEL)` appears throughout the
  upstream readers), so the readers are mostly ready and what is missing is the
  Starter's scope stack. `SUBMODEL`/`ENDSUB` block **5 corpus decks with 5 sole
  blockers** — the highest sole-blocker ratio of any keyword family in the
  corpus, so this is worth doing carefully rather than quickly.
- [ ] **Step 4** — run → PASS; **the 5 decks' verdicts re-measured**; commit
  `feat(starter): /SUBMODEL scope stack and scoping`.

---

### Task P10.4: Initial conditions

**Fortran:** `starter/source/initial_conditions/` (8 directories, 51 files,
  14,218 LOC): `detonation`, `general`, `inicrack`, `inigrav`, `inimap`,
  `inista`, `inivol`, `thermic`.
**Files:** Create `pyradioss/starter/initcond/` package
(`velocity.py`, `gravity.py`, `temperature.py`, `inista.py`, `inivol.py`,
`inimap.py`, `inicrack.py`, `detonation.py`, `assemble.py`),
`tests/test_p10_initcond.py`.
**Interfaces:**
- Produces: `initcond.apply(model, log) -> np.ndarray` — one pass that applies
  every initial condition **in upstream's order**, returning the resulting total
  momentum (a checkable invariant).

- [ ] **Step 1** — write the failing test:

```python
def test_initial_conditions_compose_to_the_expected_total_momentum():
    m = _model_with_all_initcond_cards()
    p = initcond.apply(m, MessageLog())
    assert np.allclose(p, _expected_total_momentum(), rtol=1e-10)

def test_order_matters_and_matches_upstream():
    """INIVEL and INIGRAV compose; upstream applies gravity first."""
    a = _run_initcond(["/INIVEL", "/INIGRAV"])
    b = _run_initcond(["/INIGRAV", "/INIVEL"])
    assert np.allclose(a.p, b.p)          # same result, different bookkeeping
    assert a.order[-1] != b.order[-1]     # but the recorded order differs
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, directory by directory. **The total-momentum
  invariant is the check that catches most of these**: `INIVEL`, `INIVEL/AXIS`,
  `INIVEL/T+G`, `INIVEL/ROT`, `INIVEL/NODE`, `INIVEL/FVM`, `INIGRAV`,
  `INIVEL/PART`, `INIVEL/SPH` all contribute momentum and the sum is
  computable from the deck. `inista`/`inistate` is the big dispatcher (M96,
  M97, M140's `/INIT/*` aliases).
- [ ] **Step 4** — run → PASS; commit per initial-condition directory.

---

### Task P10.5: Model assembling, mesh, transformations

**Fortran:** `starter/source/model/{assembling,mesh,transformation,group}` —
  `assembling` (1,360), `mesh` (815), `transformation` (1,101), `group` (912);
  the transformations the port has as `model/skew.py` (M39) plus
  `/TRANSFORM/POS`, `/TRANSFORM/PROJ`, `/TRANSFORM/FRAME`, `/TRANSFORM/AUTOPOSITION`,
  `/TRANSFORM/TRA` (M99, M112, M134, M180).
**Files:** Create `pyradioss/starter/assembly.py`, `pyradioss/starter/mesh.py`,
`tests/test_p10_assembly.py`.
**Interfaces:**
- Produces: `assembly.build(model, log) -> Assembly` with
  `Assembly.part_of_node`, `.part_of_elem`, `.element_groups`,
  `Assembly.mass`, `Assembly.inertia`, `Assembly.centre_of_mass`.

- [ ] **Step 1** — write the failing test:

```python
def test_total_mass_equals_the_sum_of_part_masses():
    a = assembly.build(_model_with_parts(), MessageLog())
    assert a.mass.sum() == pytest.approx(
        sum(p.density * _part_volume(p) for p in a.parts), rel=1e-12)

def test_transform_pos_translates_and_rotates_the_referenced_nodes():
    m = _model_with_transform_pos()
    x0 = m.nodes[:, :3].copy()
    assembly.build(m, MessageLog())
    assert not np.allclose(m.nodes[:, :3], x0)
```

- [ ] **Step 3** — implement. **`/TRANSFORM/TRA` is a 5-deck coverage blocker**
  and `MOVE_FUNCT` a 6-deck one; both live here or in the loads. The Starter
  applies transforms **before** mass assembly — an ordering error changes every
  mass in the model and is invisible except through the total-mass invariant.
- [ ] **Step 4** — run → PASS; commit per transformation family.

---

### Task P10.6: The Starter driver

**Fortran:** `starter/source/starter/` (9 files, **17,792 LOC**) — `lectur.F`
  (the deck-reading driver, the port's `input/keywords/dispatch.py` mirrors it),
  `lecfun.F`, the initial-condition pass, the mass/energy pre-check, the message
  assembly, the `/VERS` banner, the exit-code handling.
**Files:** Create `pyradioss/starter/lectur.py`, `pyradioss/starter/precheck.py`,
`tests/test_p10_lectur.py`.
**Interfaces:**
- Produces: `lectur.run(path, log) -> StarterResult` with
  `StarterResult.model`, `.messages`, `.exit_code`, `.banner`.
  `precheck.check(model, log) -> list[Problem]` — the Starter's own validation
  (`/CHECK` style), producing the same message numbers as upstream.

- [ ] **Step 1** — write the failing test:

```python
def test_starter_exit_code_follows_the_upstream_convention():
    """Upstream: 0 = clean, 1 = warnings only, 2 = errors (ARRET(2))."""
    assert _run_starter(_clean_deck()).exit_code == 0
    assert _run_starter(_warning_deck()).exit_code == 1
    assert _run_starter(_error_deck()).exit_code == 2
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The exit-code convention is a scriptable
  contract** that CI and the coverage harness depend on; the port's current
  behaviour must be checked against `lectur.F`'s `ARRET` calls before changing
  it. The message-number mapping (`MSGID`) is a second contract: the port has
  its own strings, and upstream's `hm_cfg_files/messages/CONFIG/*.txt` is the
  catalogue — **map to it**, because it is what makes a port message
  identifiable.
- [ ] **Step 4** — run → PASS; commit
  `feat(starter): the lectur driver, pre-check and upstream exit codes`.

---

### Task P10.7: General controls

**Fortran:** `starter/source/general_controls/` (37 files, 7,742) plus
  `engine/source/general_controls/` (5 files, 1,422) — `/RUN`, `/VERS`,
  `/PRINT`, `/CHECKSUM`, `/ALTDOCTAG`, `/DEBUG`, `/ENG/STATE`, `/ENG/DYNAIN`,
  `/STOP/*`, `/PARITH`, `/UPWIND`, `/TH`, `/TFILE`, `/H3D`, `/ANIM`, `/MON`,
  `/RFILE`.
**Files:** Create `pyradioss/input/controls.py`, `tests/test_p10_controls.py`.
**Interfaces:**
- Produces: `controls.Controls` — one typed record per card, read once and
  consumed by Phase 12 (output) and Phase 11 (multiphysics).

- [ ] **Step 1** — write the failing test:

```python
def test_every_control_card_in_the_cfg_tree_is_read_or_refused():
    names = _cfg_control_keywords()
    known = set(controls.CONTROL_READERS) | set(controls.refused())
    assert names <= known, sorted(names - known)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement, exhaustively over `$OR_SRC/hm_cfg_files/CFG/**`
  control cards. **Phase 12 cannot be specified without this**, so this task
  lands before Phase 12 even though Phase 12 was listed first in the index.
- [ ] **Step 4** — run → PASS; commit
  `feat(input): the general-controls card set, read exhaustively`.

---

### Task P10.8: The binary restart format (`restart/ddsplit`)

**Fortran:** `starter/source/restart/ddsplit/` — **165 files, 45,501 LOC**.
  Writer/reader pairs `c_*.F` / `r_*.F` for every persisted array:
  `c_bufel.F` (element buffers), `c_crkadd.F`, `c_crkedge.F`, `c_crkxfem.F`
  (XFEM), `c_dampvrel.F`, `c_drape.F`, `c_eig.F`, `c_elig3d.F` (IGA), and ~150
  more. The layout is documented by the record order in the writer; the
  Starter's own header (`or_build_info.py`-generated `build_info.inc`) is
  embedded.
**Files:** Create `pyradioss/restart/` package
(`format.py`, `fields.py`, `write.py`, `read.py`, `version.py`),
`tests/test_p10_restart_binary.py`.
**Interfaces:**
- Produces:
  - `restart.format.Header` with `magic: bytes`, `version: int`,
    `build_info: str`, `arch: str`, `precision: str`, `n_parts: int`;
  - `restart.fields.FIELDS: tuple[Field, ...]` — the ordered field table from
    Task P10.0's `restart_fields.json`, each `Field` with `name`, `dtype`,
    `nbytes_fn(model) -> int`;
  - `restart.write.write(model, path) -> None`;
  - `restart.read.read(path) -> Model`;
  - `restart.version.CURRENT = 1`.

- [ ] **Step 1** — write the failing test:

```python
def test_binary_restart_roundtrip_is_byte_identical(tmp_path):
    m = _rich_model(seed=0)
    restart.write.write(m, tmp_path / "a.rst")
    m2 = restart.read.read(tmp_path / "a.rst")
    restart.write.write(m2, tmp_path / "b.rst")
    assert (tmp_path / "a.rst").read_bytes() == (tmp_path / "b.rst").read_bytes()

def test_old_pickle_restart_is_refused_with_a_clear_version_error(tmp_path):
    import pickle
    (tmp_path / "old.rst").write_bytes(pickle.dumps(_model()))
    with pytest.raises(RestartVersionError, match="pickle"):
        restart.read.read(tmp_path / "old.rst")
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Four decisions the implementer must not make
  alone:**
  1. **The magic and version come first**, and a version mismatch is a **named
     error**, not a best-effort read. Review Focus item 4 requires restart
     continuity, and a silent misread is worse than a refusal.
  2. **Field order is the upstream writer's order** — read
     `starter/source/restart/ddsplit/` and follow the write sequence. Reordering
     fields changes the bytes without changing the meaning.
  3. **Precision**: the Fortran build is `dp`/`sp` selectable
     (`starter/CMakeLists.txt:88-101`); the port is single-precision NumPy. The
     header records the precision, and a `sp` Fortran restart read by the port
     must be explicitly supported or explicitly refused — **not** silently
     widened.
  4. **Every array in `restart_fields.json` must be written.** A test walks the
     field table and asserts each one appears in the output, so a forgotten
     field is a test failure rather than a lost array three milestones later.
- [ ] **Step 4** — run → PASS; the restart-chaining md5 test (`_0002.rad`
  reproduces the unchained run) is green under the binary format; commit
  `feat(restart): the binary restart format (ddsplit, 165 field writers)`.

---

### Task P10.9: Restart compatibility and migration

**Fortran:** `starter/source/restart/ddsplit/`'s reader set; the engine's
  `engine/source/output/restart/` (`wrrestp.F`, `rdresb.F`) — Phase 12's.
**Files:** Modify `pyradioss/starter/restart.py`,
`pyradioss/output/` (Phase 12 lands the engine half).
**Interfaces:**
- Produces: `restart.migration.convert_pickle(path, to_binary: Path) -> None`,
  and `restart.read_legacy(path) -> Model` for the transition window.

- [ ] **Step 1** — write the failing test:

```python
def test_a_pickle_restart_converts_to_binary_and_gives_the_same_model(tmp_path):
    m = _rich_model(seed=1)
    pickle_path = _write_pickle_restart(m, tmp_path)
    restart.migration.convert_pickle(pickle_path, tmp_path / "b.rst")
    assert _fingerprint(restart.read.read(tmp_path / "b.rst")) == _fingerprint(m)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **Decide the transition policy explicitly and
  write it in `docs/STATE.md`:** either (a) pickle reads keep working for one
  release with a `DeprecationWarning`, or (b) they stop. Option (a) costs one
  small module and avoids breaking anyone's chained run; option (b) is cleaner
  and is the maintainer's call. **The task does not choose — it implements both
  paths and the warning.**
- [ ] **Step 4** — run → PASS; commit
  `feat(restart): pickle-to-binary migration with an explicit transition policy`.

---

### Task P10.10: SPMD Starter half and element reordering

**Fortran:** `starter/source/spmd/` (37 files, 18,608 LOC) —
  `spmd/domain_decomposition/` (23 files, 10,536), `spmd/node` (3, 1,844),
  `cpp_reorder_elements.cpp`, `cpp_split_tool.cpp`, `domdec2.F`,
  `igrsurf_split.F`, `prepare_split_i25e2e.F`, `split_cfd_solide.F`,
  `globvars.F`, `deallocate_igrsurf_split.F`, `get_size_tag.F`,
  `spmd_anim_ply_init.F`.
**Files:** Modify `pyradioss/spmd/domdec.py`, `pyradioss/spmd/`,
`tests/test_p10_spmd_starter.py`.
**Interfaces:**
- Consumes: Phase 8 Task P8.7's `SurfaceIndex` (the interfaces must be split
  per domain: `igrsurf_split.F`).
- Produces: `spmd.domdec.reorder_elements(model, domains) -> np.ndarray`
  (the `cpp_reorder_elements.cpp` pass), `spmd.domdec.split_surfaces(...)`,
  `spmd.domdec.write_per_domain_restarts(model, path) -> list[Path]`.

- [ ] **Step 1** — write the failing test:

```python
def test_named_restart_reproduces_the_serial_result():
    serial = _run(np=1)
    decomposed = _run(np=4)
    assert np.allclose(serial.T01["IE"], decomposed.T01["IE"], rtol=1e-9)

def test_element_reordering_is_deterministic():
    a = spmd.domdec.reorder_elements(_model(seed=3), n=4)
    b = spmd.domdec.reorder_elements(_model(seed=3), n=4)
    assert np.array_equal(a, b)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The `-np 4` run reproducing serial is the port's
  own parity contract** (`README.md`: "The results are the serial results up to
  floating-point summation order"). The element reordering is what makes that
  true: a domain-local element order changes the force summation order and
  therefore the last bits. Read `cpp_reorder_elements.cpp` — it is C++, and its
  comparator defines the order.
- [ ] **Step 4** — run → PASS; commit
  `feat(spmd): the Starter decomposition half and element reordering`.

---

### Task P10.11: Coverage re-measure

**Files:** Modify `tools/validate_vs_fortran.py`.
**Interfaces:**
- Produces: the `SUBMODEL`/`ENDSUB` (5/5), `SUBDOMAIN` (4/1), `GRNOD/NODENS`
  (1/1), `XREF` (1/1), `TRANSFORM/TRA` (5/0), `TRANSFORM/ROT` (3/0),
  `TRANSFORM/SYM` (2/0), `MOVE_FUNCT` (6/6), `SPHGLO` (3/0) blocker rows updated.

- [ ] **Step 1** — write the failing test:

```python
def test_starter_coverage_delta_is_recorded():
    d = load_coverage("tools/validation_data/coverage_p10.json")
    assert d["delta_vs_p9"]["CLEAN"] > 0
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement the measurement and the report table in
  `VALIDATION.md` §4.x.
- [ ] **Step 4** — run → PASS; commit `docs(validation): Starter coverage delta (M-series → P-series)`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P10.0** — the restart array inventory covers all 165 `ddsplit` files; an
  inventory that misses a writer is an incomplete specification.
- **P10.1** — fixpoint resolution, negative-id precedence and cycle detection
  are transcribed, not re-invented.
- **P10.2** — the analytic surfaces are exact to the last bit.
- **P10.3** — the unbalanced-`/SUBMODEL` case **raises**; 5 decks re-measured.
- **P10.4** — the total-momentum invariant is the specification.
- **P10.5** — transforms are applied **before** mass assembly; the total-mass
  invariant catches an ordering error.
- **P10.6** — the exit codes match upstream's `ARRET` convention and the message
  IDs map to `hm_cfg_files/messages/CONFIG/`.
- **P10.7** — exhaustive over the CFG control cards.
- **P10.8** — the version check is a named error; field order is upstream's;
  **every** field in the inventory is written (proven by a test that walks the
  table); precision is recorded, not assumed.
- **P10.9** — both transition paths exist; the policy is the maintainer's, and
  the task says so in the commit.
- **P10.10** — the `-np 4` run reproduces serial within the stated summation
  tolerance; the reordering comparator is read from the C++.

## Parallelisation

- **Wave 0** — P10.0 serial; it produces `restart_fields.json`, which Task P10.8
  is specified against.
- **Wave 1** — 6 tasks, 6 agents, disjoint modules. P10.7 lands before Phase 12
  and is small, so it starts first.
- **Wave 2** — serial; P10.8 is the largest single task in the phase.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p10_restart_binary.py tests/test_p10_sets.py
python tools/validate_vs_fortran.py --mode coverage --out tools/validation_data/coverage_p10.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area starter/source
```

The phase reviewer confirms:

1. the restart-chaining test (Review Focus item 4) is green under the **binary**
   format, not only under pickle;
2. every Starter-owned coverage blocker from the list in Task P10.11 is either
   closed or has a filed blocker;
3. the Starter exit codes match upstream.