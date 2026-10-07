# Phase 12 — Output

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `engine/source/output/**` and `starter/source/output/**` a
literal port — **binary T01, binary A-files, H3D, the listing, PARITH, and the
restart pair**. The port today writes CSV and legacy VTK; that is a *deliberate,
documented* deviation (`README.md`) and a literal port replaces it.

**Architecture:** output is the phase with the **worst ratio in the program**:
**752 Python lines** (`output/time_history.py` 346 + `output/anim_vtk.py` 406)
against **180,559 Fortran lines**. It is also the phase that unlocks the other
ones: a binary T01 is what makes `tools/compare_t01.py` (Phase 0 Task P0.10)
read the reference, and a binary restart pair is what Phase 10 Task P10.8
writes.

**Tech stack:** NumPy + `struct`/`np.frombuffer` for the binary formats; the
port already depends on neither VTK nor any proprietary library for its current
output, and the new formats must keep it that way.
**Spec:** `plan/00_ORCHESTRATION.md` §10 items 3 and 4.
**Entry criteria:** Phases 2, 3, 6, 7, 8, 9, 10 exit gates; Phase 0 Task P0.10
(the T01 reader). **Task P12.9 (PARITH) additionally needs Phase 11 Task P11.15
(the remesher)**; every other task in this phase is independent of Phase 11.
**Band: XL.**

---

## Scope

| Upstream directory | Files | LOC | Port |
|---|---:|---:|---|
| `engine/source/output/h3d/` | 201 | **70,363** | — (VTK only) |
| `engine/source/output/anim/` | 126 | **52,458** | `output/anim_vtk.py` (406) |
| `engine/source/output/restart/` | 46 | 21,763 | pickle + Phase 10's binary |
| `engine/source/output/sta/` | 51 | 21,367 | — |
| `engine/source/output/th/` | 45 | 13,707 | `output/time_history.py` (346) |
| `engine/source/output/dynain/` | 9 | 4,916 | `engine/engine_controls.py` |
| `engine/source/output/sty/` | 14 | 7,444 | — |
| `engine/source/output/report/` | 2 | 3,175 | — |
| `engine/source/output/qaprint/` | 4 | 879 | — |
| `engine/source/output/message/`, `outfile/`, `tools/`, `cluster/`, `csv/` | 20 | 3,343 | `input/`, `starter/` |
| `starter/source/output/qaprint/` | 35 | 10,356 | — |
| `starter/source/output/th/` | 24 | 11,547 | — |
| `starter/source/output/anim/` | 65 | 12,248 | `output/anim_vtk.py` |
| `starter/source/output/message/` | 9 | 3,493 | `MessageLog` |
| `starter/source/output/subinterface/` | 5 | 4,016 | — |
| `starter/source/output/{analyse,checksum,cluster,gauge,outp,stat,thpart,tools}` | 38 | 2,573 | — |
| **Total** | **≈ 697** | **≈ 245,000** | **752** |

**1:326.** This is the number that sizes the phase.

## Gap analysis

### The formats a literal port must write

| Format | Upstream | Why it matters |
|---|---|---|
| **T01 (binary time history)** | `engine/source/output/th/` (45 files) | the reference's own format; `th_to_csv` converts it; every parity comparison reads it |
| **A-files (binary animation)** | `engine/source/output/anim/` (126 files) | third-party post-processors read them (Vortex-Radioss, `anim_to_vtk`) |
| **H3D** | `engine/source/output/h3d/` (201 files, 70,363 — **the largest subsystem in the program**) | the Altair viewer format; needs `extlib/h3d` |
| **Listing** (`.out`) | `sta/` + `sty/` + `report/` (67 files, 32,986) | the human-readable run report and its final energy balance |
| **PARITH** (adaptive remeshing history) | `engine/source/model/remesh` + `parith_on_mod.F90` | the adaptive-mesh log |
| **Restart** (engine half) | `engine/source/output/restart/` (46 files, 21,763) — `wrrestp.F`, `rdresb.F`, `read_ale_grid.F90`, … | the engine's own checkpoint (`_0002.rad` chaining) |
| **QAPRINT** | `starter/source/output/qaprint/` (35 files, 10,356) | the QA comparison dumps used by `qa-tests` |

### The port's current state, honestly

- `time_history.py` (346 lines) writes **CSV** with named channels. It is a good
  module. It is not the T01 format.
- `anim_vtk.py` (406 lines) writes **legacy VTK**, already optimised (M39's
  `_write_block`, 2.75× on c37, byte-identical output).
- Neither is wrong for its purpose — they are simply not the upstream formats.
- **The VTK path must not be deleted.** It is the port's user-facing output and
  the `README.md`'s promise. The correct end state is **both**: upstream
  formats *and* the documented ASCII/VTK, selected by a flag.

### Highest-value items, in order

1. **T01 binary** (Phase 0 Task P0.10 already landed the *reader*). This is
   small and everything else depends on it.
2. **The engine restart pair** (`wrrestp.F`/`rdresb.F`) — Phase 10 Task P10.8
   writes the *Starter* restart; this is the *Engine's*.
3. **The listing** (`sta`/`sty`/`report`) — 32,986 LOC, and it is where the
   energy balance, the cycle table and the final verdict live. It is also where
   Review Focus item 3's guard reporting happens.
4. **A-files** — 52,458 LOC, needed for third-party post-processing.
5. **H3D** — 70,363 LOC and needs `extlib/h3d`, which is a *proprietary*
   Altair binary format. **This is the one item where "port" may mean
   "interoperate": decide with the maintainer whether to write H3D natively or
   to route it through `extlib/h3d` via a C shim, or to declare it out of
   scope with the reason recorded.** Task P12.8.
6. **PARITH** — small, and gated on Phase 11's remesher.

## Wave graph

```
Wave 0  P12.0 output reconciliation audit + the format decision (serial)
   ── gate: every format has a decision; H3D decided
Wave 1 (parallel — new modules)
  P12.1 binary T01 writer            P12.2 the /TH channel set
  P12.3 the listing: cycle table     P12.4 the listing: energy balance + verdict
  P12.5 A-file binary format         P12.6 the engine restart pair (wrrestp/rdresb)
  P12.7 QAPRINT dumps                 P12.8 H3D: write / interop / decline
  P12.9 PARITH adaptive-mesh history
   ── gate per format: roundtrip + byte-level + oracle comparison
Wave 2 (serial)
  P12.10 keep CSV/VTK alongside, with a flag
  P12.11 coverage/perf re-measure
      (the Review Focus item 3 guard extension is Task P12.4, in Wave 1)
```

---

### Task P12.0: Output audit and format decisions

**Fortran:** every directory in §Scope.
**Files:** Create `tools/validation_data/output_status.json`,
`docs/OUTPUT_FORMATS.md`, `tests/test_p12_output_census.py`.
**Interfaces:**
- Produces: one record per output artefact with `format`, `upstream_writer`,
  `port_module`, `decision` (`write-native` | `interop` | `decline`),
  `decision_reason`, `status`.

- [ ] **Step 1** — write the failing test:

```python
def test_every_output_format_has_a_decision_and_a_reason():
    st = json.loads(Path("tools/validation_data/output_status.json").read_text())
    for r in st:
        assert r["decision"] in {"write-native", "interop", "decline"}, r["format"]
        assert r["decision_reason"], r["format"]
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — audit and decide. **The H3D decision is the one that needs
  the maintainer**, and it has three defensible answers:
  - **write-native**: reimplement the Altair H3D binary from `extlib/h3d`'s
    observed layout. Highest effort, no proprietary dependency, and **legal
    exposure** if the format is undocumented proprietary work. That exposure is
    a question about `extlib/h3d`'s own terms, not about this repository's:
    Task P0.0 settled this repository's licence (AGPL-3.0-or-later, 2026-10-07,
    `docs/LICENSING.md` §Decision) and does not settle H3D's;
  - **interop**: keep VTK/CSV and declare H3D out of scope, recording that the
    port's animation output is readable in ParaView (`README.md`'s stated
    position);
  - **decline with a C shim to `extlib/h3d`**: possible in principle, adds a
    compiled dependency to a pure-Python project.
  **The task presents the three with their costs and does not choose.**
- [ ] **Step 4** — run → PASS; commit
  `docs(output): per-format audit and the write/interop/decline decisions`.

---

### Task P12.1: The binary T01 writer

**Fortran:** `engine/source/output/th/` — the record layout, channel naming and
  the cycle-block structure; `$OR_SRC/tools/th_to_csv/` (upstream's own
  binary→ASCII converter, the **reference** for the layout).
**Files:** Create `pyradioss/output/t01.py`, `tests/test_p12_t01.py`.
**Interfaces:**
- Consumes: `tools.compare_t01.T01` from Phase 0 Task P0.10.
- Produces:
  - `t01.write(path, header, channels, blocks) -> None`
  - `t01.Header` with `version`, `nchannels`, `nprops`, `title`, `date`,
    `radioss_version`;
  - `t01.Block` with `cycle`, `time`, `step`, `values` (one per channel).

- [ ] **Step 1** — write the failing test:

```python
def test_t01_roundtrips_through_the_reference_reader(tmp_path):
    p = tmp_path / "xT01"
    t01.write(p, _header(), ["T", "IE", "KE"], [_block(n=5)])
    got = tools.compare_t01.read_t01(p)
    assert got.channels == ["T", "IE", "KE"]
    assert np.allclose(got.values, _expected_values(), rtol=1e-12)

def test_th_to_csv_agrees_with_our_reader(tmp_path):
    """The strongest available check: upstream's own converter must produce
    the same numbers."""
    p = _write_t01(tmp_path)
    csv = _run_th_to_csv(p, tmp_path)
    assert _parse_csv(csv) == pytest.approx(tools.compare_t01.read_t01(p).values,
                                            rel=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The second test is the specification and the
  reason Phase 0 put `th_to_csv` in the resolution order.** Run upstream's
  converter on the port's file: if the numbers disagree, the layout is wrong.
  Do not "fix" the test. Note that `$OR_ROOT/extlib/hm_reader/linux64` must be on
  `LD_LIBRARY_PATH` for `th_to_csv` to run (Task P0.4).
- [ ] **Step 4** — run → PASS; a full example's T01 written by the port is
  **byte-identical** to the reference's for the same run when the physics
  matches; commit `feat(output): the binary T01 writer`.

---

### Task P12.2: The `/TH` channel set

**Fortran:** `engine/source/output/th/` — 45 files: `init_th.F`,
  `init_th0.F`, `init_th_group.F`, `thbcs.F`, `thbcs_imp.F`, `bcs1th.F`,
  `bcs1th_imp.F`, `hist1.F`, `hist2.F`, `hist13.F`, `thkin.F`, `thcoq.F`,
  `thcluster.F`, `thchecksum.F90`, `read_th_restart.F`, `surf_area.F`,
  `surf_mass.F`, `grelem_sav.F`, `init_reac_nod.F`; the Starter's
  `starter/source/output/th/` (24 files, 11,547).
**Files:** Create `pyradioss/output/th_channels.py`, `tests/test_p12_th_channels.py`.
**Interfaces:**
- Produces: `th_channels.CHANNELS: dict[str, Channel]` with
  `Channel.compute(model, group, t, step) -> np.ndarray`,
  `Channel.scaling: tuple[int,int,int]` (the Fortran `SCALING` triple).

- [ ] **Step 1** — write the failing test:

```python
def test_every_declared_th_card_has_a_channel_or_a_cited_refusal():
    cards = {r["card"] for r in load_th_cards()}
    known = set(th_channels.CHANNELS) | set(th_channels.refused())
    assert cards <= known, sorted(cards - known)

def test_energy_channels_are_budget_consistent():
    vals = th_channels.sample(_finished_model())
    ie, ke, xk, ext = (vals[c] for c in ("IE", "KE", "XK", "EXT"))
    assert abs(ie - (ke + xk + ext)) / max(abs(ie), 1.0) < 1e-3
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The energy-consistency test is the one that
  matters** — it ties the output layer to the EN ledger, so a channel that
  double-counts or drops a term fails here. The `SCALING` triple matters for
  the binary format (Phase 12 Task P12.1) and must be read from the Fortran, not
  inferred.
- [ ] **Step 4** — run → PASS; commit `feat(output): the /TH channel set with scaling triples`.

---

### Task P12.3: The listing — cycle table

**Fortran:** `engine/source/output/sta/` (51 files, 21,367) — `genstat.F`,
  `stat_brick_{mp,spmd}.F`, `stat_beam_{mp,spmd}.F`, `sta_c_*` families, and
  `engine/source/output/sty/` (14 files, 7,444) — the styling/format layer;
  `engine/source/output/outfile/`, `engine/source/output/tools/`.
**Files:** Create `pyradioss/output/listing.py`, `tests/test_p12_listing.py`.
**Interfaces:**
- Produces: `listing.CycleRow` with `cycle, time, dt, epdt, ie, ke, xk, ext,
  err, momx, momy, momz`; `listing.render_cycle_table(rows) -> str`;
  `listing.CYCLE_EVERY: int`.

- [ ] **Step 1** — write the failing test:

```python
def test_cycle_table_matches_the_reference_column_for_column():
    rows = listing.run_example(_example_dir("tensile_bar"))
    ref = _reference_listing(_example_dir("tensile_bar"))
    assert listing.render_cycle_table(rows) == _strip_banner(ref)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **A literal port means the listing text matches
  upstream**, column widths included (`sty/` is the formatting layer and it is
  7,444 lines of it). The test compares against the reference listing with the
  banner and the build-info block stripped. If exact matching proves
  unattainable (Fortran format-specifier edge cases), **record the residual
  differences explicitly in `docs/OUTPUT_FORMATS.md`** rather than accepting a
  loose comparison.
- [ ] **Step 4** — run → PASS; commit `feat(output): the cycle-table listing (sta/sty)`.

---

### Task P12.4: The listing — energy balance and verdict

**Fortran:** `engine/source/output/report/` (2 files, 3,175),
  `engine/source/output/sta/`'s final block; `engine/source/interfaces/interf/`
  no — `ecrit.F` is in `engine/source/output/sta/` per `README.md`'s
  `ecrit.F` citation.
**Files:** Modify `pyradioss/output/listing.py`, `pyradioss/engine/engine.py`.
**Interfaces:**
- Produces: `listing.final_energy_balance(model) -> Balance` with
  `Balance.ie, .ke, .xk, .ew, .ed, .ext, .error_pct, .balance_ok`;
  `listing.verdict(model) -> Literal["NORMAL", "ERROR", "ABORT"]`.

- [ ] **Step 1** — write the failing test — **Review Focus item 3**:

```python
def test_divergence_guard_checks_ie_and_he_not_only_ke():
    """VALIDATION.md §7 item 10: the backstop only checks KE. NaN comparisons
    are False, so an IE that goes NaN while KE stays finite slips every
    percentage guard."""
    e = {"KE": 1.0, "IE": float("nan"), "HE": 0.1}
    assert not listing.energy_ok(e, limits=_limits())
    assert not listing.energy_ok({"KE": float("nan"), "IE": 1.0, "HE": 0.1}, _limits())
    assert listing.energy_ok({"KE": 1.0, "IE": 1.0, "HE": 0.01}, _limits())
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **`VALIDATION.md` §7 item 10 is explicit that this
  was deliberately not changed while fixing an unrelated startup-conditioning
  issue, and that widening a guard's reach is "a separate change with its own
  risk". This is that separate change.** Read `ecrit.F` for the upstream guard
  set (`DEMXS`/`EP30`) and port **all** of it — and note the upstream default is
  `EP30`, i.e. **no energy stop at all**, which is a reporting choice and not a
  physics endorsement (`VALIDATION.md` §7 item 1). Do not "fix" that.
- [ ] **Step 4** — run → PASS; commit
  `fix(output): extend the divergence backstop from KE to IE and HE (VALIDATION §7 item 10)`.

---

### Task P12.5: The A-file binary format

**Fortran:** `engine/source/output/anim/` (126 files, 52,458) — the record
  layout, the per-group field tables, the resizing/allocation model;
  `starter/source/output/anim/` (65 files, 12,248).
**Files:** Create `pyradioss/output/anim_binary.py`,
`tests/test_p12_anim_binary.py`.
**Interfaces:**
- Produces: `anim_binary.write(path, frame) -> None`,
  `anim_binary.Frame` with `cycle, time, scale, translation, groups`,
  `anim_binary.GROUP_TABLES: dict[str, FieldTable]`.

- [ ] **Step 1** — write the failing test:

```python
def test_anim_file_is_readable_by_the_upstream_converter(tmp_path):
    """anim_to_vtk_win64.exe (or the Vortex-Radioss reader) must be able to
    read what the port writes."""
    p = anim_binary.write(_one_frame_model(), tmp_path)
    vtks = _run_anim_to_vtk(p, tmp_path)
    assert vtks, "anim_to_vtk produced nothing"
    assert "effective_plastic_strain" in _fields_of(vtks[0])
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The converter is the oracle**: `$OR_SRC/tools/
  anim_to_vtk/` is upstream's own tool, and the port's existing
  `lspp_check.py` skill (`.agents/skills/lspp-check/`) verifies d3plot
  round-trips. The `EFFECTIVE PLASTIC STRAIN` array is the one the port's
  `pyproject.toml` already pins `vortex_radioss@v1.021` for; the binary writer
  must place it where that reader expects.
- [ ] **Step 4** — run → PASS; **the existing VTK output stays byte-identical**;
  commit `feat(output): the binary A-file writer`.

---

### Task P12.6: The engine restart pair

**Fortran:** `engine/source/output/restart/` (46 files, 21,763) —
  `wrrestp.F`, `rdresb.F`, `rdresa.F`, `rdcomm.F`, `arralloc.F`, `fillxdp.F`,
  `read_ale_grid.F90`, `read_ale_rezoning_param.F90`, `read_bcs_nrf.F90`,
  `read_bcs_wall.F90`, and ~36 more.
**Files:** Create `pyradioss/output/restart_pair.py`,
`tests/test_p12_restart_pair.py`.
**Interfaces:**
- Produces: `restart_pair.write(model, path) -> None`,
  `restart_pair.read(path) -> Model`; the engine-side counterpart of Phase 10
  Task P10.8's Starter-side `restart.write`.

- [ ] **Step 1** — write the failing test — **Review Focus item 4**:

```python
def test_engine_restart_chaining_reproduces_the_unchained_run(tmp_path):
    """_0002.rad resumes the previous run and must reproduce it exactly."""
    a = _run_chain([_engine_deck()], tmp_path / "a")
    b = _run_chain([_engine_deck(), _engine_deck()], tmp_path / "b")
    assert np.allclose(a.T01["IE"], b.T01["IE"][:len(a.T01["IE"])], rtol=0, atol=0)
```

- [ ] **Step 2** — run → FAIL (the port has restart chaining via pickle; the
  binary pair does not exist).
- [ ] **Step 3** — implement. The test compares the **first N cycles of the
  chained run against the unchained run with `atol=0`** — i.e. bit-identical.
  That is the port's existing contract (`README.md`: restart chaining "reproduces
  the unchained run exactly") and it must survive the format change. The
  `read_ale_grid.F90` / `read_bcs_nrf.F90` / `read_bcs_wall.F90` readers mean the
  restart carries **ALE and non-reflecting-BC state**, which Phase 11 owns;
  declare the dependency and refuse if it is absent rather than reading garbage.
- [ ] **Step 4** — run → PASS under **both** backends; commit
  `feat(output): the engine restart pair (wrrestp/rdresb)`.

---

### Task P12.7: QAPRINT dumps

**Fortran:** `starter/source/output/qaprint/` (35 files, 10,356) +
  `engine/source/output/qaprint/` (4 files, 879).
**Files:** Create `pyradioss/output/qaprint.py`, `tests/test_p12_qaprint.py`.
**Interfaces:**
- Produces: `qaprint.dump(model, results) -> Path` — the reference-value files
  `qa-tests/scripts/verify_results.py` and `qa_system.py` compare against.

- [ ] **Step 1** — write the failing test:

```python
def test_qaprint_output_is_readable_by_the_upstream_verifier(tmp_path):
    p = qaprint.dump(_model_with_known_results(), tmp_path)
    assert _run_or_verify_results(p, tmp_path).returncode == 0
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **This is the highest-leverage single task for
  automated validation after T01**: it makes the port's output consumable by
  upstream's own QA tooling (`$OR_SRC/qa-tests/qa_tools/verify_results.py`),
  which converts the whole program's validation story from "we compare
  ourselves" to "upstream's verifier accepts us".
- [ ] **Step 4** — run → PASS; commit `feat(output): QAPRINT reference dumps readable by verify_results.py`.

---

### Task P12.8: H3D — the decision's implementation

**Fortran:** `engine/source/output/h3d/` (201 files, **70,363 lines**) —
  `h3d_build_cpp/`, `h3d_build_fortran/`, `h3d_results/`, `input_list/`,
  `spmd/`; the runtime is `extlib/h3d/lib/linux64` (`INSTALL.md:42`).
**Files:** per the Task P12.0 decision.
**Interfaces:**
- Produces: for `write-native`, `pyradioss/output/h3d.py` with
  `h3d.write(path, frame) -> None`; for `interop`, a `docs/OUTPUT_FORMATS.md`
  entry plus a VTK→H3D conversion note; for `decline`, the same entry.

- [ ] **Step 1** — write the failing test:

```python
def test_h3d_decision_is_implemented_or_declared():
    st = json.loads(Path("tools/validation_data/output_status.json").read_text())
    h3d = next(r for r in st if r["format"] == "h3d")
    if h3d["decision"] == "write-native":
        assert h3d["status"] == "ported"
    else:
        assert "h3d" in Path("docs/OUTPUT_FORMATS.md").read_text().lower()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement the decision Task P12.0 recorded. **If
  `write-native` was chosen, this is the largest task in the program** and it
  deserves its own phase rather than a task; say so and stop, because
  pretending it fits here would be the plan lying about its own size.
- [ ] **Step 4** — run → PASS; commit per the decision.

---

### Task P12.9: PARITH adaptive-mesh history

**Fortran:** `common_source/modules/parith_on_mod.F90`,
  `starter/source/model/remesh/`'s parith output,
  `engine/source/tools/`'s remesh statistics.
**Files:** Create `pyradioss/output/parith.py`, `tests/test_p12_parith.py`.
**Interfaces:**
- Produces: `parith.write(model, history) -> None`,
  `parith.History` with `.cycles`, `.elements_added`, `.elements_removed`.

- [ ] **Step 1** — write the failing test:

```python
def test_parith_history_is_monotone_in_element_count():
    h = _run_a_remeshing_case()
    counts = np.array([r.n_elements for r in h])
    assert np.all(np.diff(counts) >= 0) or _removes_explicitly(h)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. Gated on Phase 11's remesher being live (Task
  P11.15). If the remesher is not landed, this task records the dependency and
  lands the format with a synthetic-history test.
- [ ] **Step 4** — run → PASS; commit `feat(output): PARITH adaptive-mesh history`.

---

### Task P12.10: Keep CSV and VTK alongside the binary formats

**Fortran:** none (this is the port's documented deviation, being *preserved*
  rather than replaced).
**Files:** Modify `pyradioss/output/time_history.py`, `anim_vtk.py`,
  `pyradioss/engine/engine.py`.
**Interfaces:**
- Produces: `-output {binary,csv,vtk,both}` on the engine CLI and
  `PYRADIOSS_OUTPUT`; default `both`. The `README.md`'s promise ("legacy VTK,
  directly readable in ParaView") is preserved and pinned by a test.

- [ ] **Step 1** — write the failing test:

```python
def test_vtk_output_is_still_the_default_and_unchanged():
    r = _run_engine("-output vtk")
    assert list(r.dir.glob("*.vtk"))
    assert not list(r.dir.glob("*A00*"))
    assert r.t01_csv.exists()

def test_both_writes_both_and_the_csv_is_byte_identical():
    a = _run_engine("-output both"); b = _run_engine("-output csv")
    assert _md5(a.t01_csv) == _md5(b.t01_csv)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The CSV must be byte-identical whatever else is
  enabled** — it is the port's documented, stable, human-readable output and
  changing it would break every consumer. The VTK path must keep its M39
  byte-identical property.
- [ ] **Step 4** — run → PASS; the VTK md5 unchanged from M41; commit
  `feat(output): -output flag; binary formats alongside the documented CSV/VTK`.

---

### Task P12.11: Coverage and performance re-measure

**Files:** Modify `tools/validate_vs_fortran.py`, `tools/benchmark.py`.
**Interfaces:**
- Produces: parity measured on the **binary** T01 rather than CSV;
  `tools/validation_data/perf_p12.json`.

- [ ] **Step 1** — write the failing test:

```python
def test_parity_mode_reads_the_binary_t01():
    assert load_parity_config()["t01_format"] == "binary"
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The write side is the expensive part** —
  `anim/` (52k) and `h3d/` (70k) dominate a run's I/O. Measure before
  optimising, per the M39 profiling discipline
  (`tools/profile_cycle.py`), and record the numbers.
- [ ] **Step 4** — run → PASS; commit `feat(validation): binary-T01 parity and output perf`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P12.0** — every format has a decision **and** a reason; the H3D decision is
  the maintainer's and the task does not make it.
- **P12.1** — `th_to_csv` agrees with the port's reader to `1e-12`. A test that
  was "fixed" to pass is an automatic rejection.
- **P12.2** — the energy channels are budget-consistent with the EN ledger.
- **P12.3** — the listing comparison is against the **reference**, with residual
  differences recorded rather than tolerated.
- **P12.4** — the guard set is upstream's, including `EP30` as the default;
  Review Focus item 3 is closed with a test.
- **P12.5** — upstream's own `anim_to_vtk` reads the port's file, and the VTK
  path stays byte-identical.
- **P12.6** — the chained run is **bit-identical** (`atol=0`).
- **P12.7** — upstream's `verify_results.py` accepts the port's dump.
- **P12.10** — CSV and VTK are byte-identical whatever else is enabled.

## Parallelisation

- **Wave 0** — P12.0 serial; it decides H3D, which sizes Task P12.8.
- **Wave 1** — 9 tasks. **P12.4 and P12.10 touch `engine.py`** (contended) and
  are scheduled last in the wave; the other seven are independent new modules.
- **Wave 2** — serial.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p12_t01.py tests/test_p12_restart_pair.py tests/test_p12_listing.py
python tools/validate_vs_fortran.py --mode parity --out tools/validation_data/parity_p12.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md --area engine/source/output
```

The phase reviewer confirms:

1. **both** output modes work and CSV/VTK are unchanged;
2. the restart-chaining test is bit-identical under the **binary** restart;
3. `th_to_csv` and `verify_results.py` accept the port's output;
4. Review Focus items 3 and 4 have a test each, and both are green.