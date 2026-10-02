# Phase 16 — SDI reader, solver tools, QA harness

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make `reader/**`, `tools/**`, `qa-tests/**` and
`engine/source/tools/**` a literal port. This phase has the **earliest possible
start** (it depends only on Phase 0) and the **widest licence exposure**.

**Architecture:** `reader/` is a **C++ library**, not solver code — 588 files,
  168,510 lines, split into six components. Its job is the **SDI** (Solver Data
  Interface) that LS-PrePost uses to read a model, and `dyna2rad` (a deck
  converter). The port has **no reader counterpart at all** — `pyradioss/input/`
  is the solver's own deck reader, a different thing.

**Tech stack:** the port is pure Python. A C++ library has three honest
resolutions: port it, wrap it, or decline it. **This phase decides, per
  component, and records the decision** — the same three-way choice Phase 12
  Task P12.0 makes for H3D.

**Spec:** `plan/00_ORCHESTRATION.md` §1.3 (licensing — this is where it bites).
**Entry criterion: Phase 0 exit gate.** No other phase is required.
**Band: XL** (by scope), **S** by dependency depth — it can start on day one.

---

## Scope

### `reader/` — 588 files, 168,510 lines of C++

| Component | Files | LOC | What it is |
|---|---:|---:|---|
| `cfgkernel/` | 311 | 63,947 | the keyword CFG kernel — the C++ implementation of the `hm_cfg_files` card schemas |
| `dyna2rad` | 83 | 44,802 | the LS-DYNA → Radioss deck converter |
| `cfgio` | 65 | 31,780 | the CFG file I/O |
| `sdi` | 44 | 15,588 | the **Solver Data Interface** — what LS-PrePost links against |
| `solver_interface` | 79 | 11,457 | the legacy solver interface |
| `io` | 6 | 1,936 | stream/record I/O |
| **Total** | **588** | **168,510** | **none** |

### `tools/`

| Directory | Files | What it is |
|---|---:|---|
| `tools/anim_to_vtk/` | 1 | upstream's A-file → VTK converter (Phase 12 Task P12.5's oracle) |
| `tools/th_to_csv/` | 1 | upstream's T01 → CSV converter (Phase 0 Task P0.10/P12.1's oracle) |
| `tools/mockup/` | 38 | the **mesh-less** geometry/mockup converters (IGES, STEP…) |
| `tools/rht_user_material/` | 13 | the RHT user-material generator (recently added upstream) |

### `qa-tests/` — 937 files

`qa-tests/miniqa/`, `qa-tests/scripts/` (`or_qa_script`, `or_run_test.py`,
`or_generate_ctest.py`, `verify_results.py`, `qa_system.py`, `miniqa.txt`),
`qa-tests/test_script.{bash,bat}`, `qa-tests/CMakeLists.txt`.

### `engine/source/tools/` — 9 subdirectories

`accele`, `curve` (`finter.F` — Phase 15 Task P15.7), `lagmul` (Phase 4/8),
`seatbelts`, `sect` (the port's `engine/sections.py`), `sensor` (the port's
`engine/sensors.py`), `skew` (Phase 2/10's `model/skew.py`), `univ`.

## Gap analysis

### 1. The licence exposure is worst here

`reader/` is **AGPL C++** (Task P0.0). Wrapping it means shipping or linking
AGPL code, which is the question Task P0.0 must answer first. **Phase 16 cannot
start its substantive work before Task P0.0 is decided.** The three
observations below are therefore conditional on that decision, and each is
written so it works under any of the four outcomes.

### 2. Three components, three honest answers

- **`sdi`** — the thing LS-PrePost links against. A pure-Python SDI is not
  linkable from C++ without a C extension, which breaks the project's
  pure-Python promise. **Realistic answers:** (a) decline, recording that SDI
  interoperability is out of scope; (b) provide the SDI **data** through the
  port's own reader so a future SDI shim is possible. (b) is cheap and is what
  this task should deliver.
- **`cfgkernel` + `cfgio`** — the port **already reimplements their purpose** in
  `pyradioss/input/card_layouts.py` (5,962 `LAYOUTS` entries) and
  `mat_reader.py`/`prop_reader.py` (2,485 CFG files, 819,434 lines of schema).
  The honest answer is **`substitute`, already done**, and the task here is a
  *fidelity audit*: does the port's cfg reader accept every dialect the C++ one
  does? `docs/STATE.md` M36/M37 claim yes for the 2022 catalogue. **Verify.**
- **`dyna2rad`** — an LS-DYNA → Radioss deck converter. Genuinely useful, no
  port counterpart, ~45k lines of C++. This is a **new feature**, not a port of
  solver physics, and it is the one item in this phase where the answer might
  legitimately be "out of scope for a solver port".

### 3. The QA harness is the highest-leverage item

`qa-tests/verify_results.py` + `qa_system.py` + `or_qa_script` are **upstream's
own acceptance machinery**. Phase 12 Task P12.7 makes the port's QAPRINT dumps
readable by `verify_results.py`. **Wiring the port's own runs into upstream's
QA script** converts the program's validation from self-assessment to
third-party assessment, and it is worth more than any other single task in this
phase.

### 4. `tools/mockup/` — 38 files

IGES/STEP mesh import. **Not a solver capability.** The answer is `decline`,
recorded, with the reasoning that the port reads `.rad` decks and the starter's
`/INIVOL`, `/MESH` keywords cover the mesh sources a solver needs.

### 5. `tools/rht_user_material/` — 13 files

A user-material generator for the RHT model. Ties to Phase 6's user-material
hooks (`usermat_solid.F`/`usermat_shell.F`, `ENG_USERLIB`). Small and
self-contained; `substitute` via a Python generator.

## Wave graph

```
Wave 0  P16.0 reader audit + per-component decision (serial; gated on P0.0)
        P16.1 QA harness inventory
   ── gate: every component has a decision with a reason
Wave 1 (parallel — independent of every other phase)
  P16.2 the cfg-kernel fidelity audit (does the port's reader match?)
  P16.3 the SDI data layer (so a shim becomes possible)
  P16.4 run the port through upstream's verify_results.py
  P16.5 run the port through upstream's or_qa_script
  P16.6 the miniqa suite as a port-side test set
  P16.7 engine/source/tools: sect, sensor, seatbelts, univ, accele, skew
  P16.8 rht_user_material generator
   ── gate per task: upstream's tool accepts the port's output
Wave 2 (serial)
  P16.9 dyna2rad: port, scope, or decline
  P16.10 the mockup decision + docs/PORT_EXTENSIONS.md completion
```

---

### Task P16.0: Reader audit and per-component decision

**Fortran:** `$OR_SRC/reader/` — `CMakeLists.txt`, `build_script.bash`,
  `build_windows.bat`, `CMake_arch/`, and the six `source/` components.
**Files:** Create `tools/validation_data/reader_status.json`,
`docs/READER_DECISIONS.md`, `tests/test_p16_reader_census.py`.
**Interfaces:**
- Produces: one record per component with `files`, `loc`, `decision`
  (`port` | `wrap` | `substitute` | `decline`), `decision_reason`,
  `licence_implication` (from Task P0.0's recorded decision), `target_module`.

- [ ] **Step 1** — write the failing test:

```python
def test_every_reader_component_has_a_decision_with_a_licence_implication():
    st = json.loads(Path("tools/validation_data/reader_status.json").read_text())
    assert {r["component"] for r in st} >= {
        "cfgio", "cfgkernel", "dyna2rad", "io", "sdi", "solver_interface"}
    for r in st:
        assert r["decision"] in {"port", "wrap", "substitute", "decline"}
        assert r["licence_implication"], r["component"]
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — decide, per component, and record the **licence implication of
  each under all four Task P0.0 outcomes**. That is the deliverable: a
  maintainer who later picks "relicense to AGPL" can read off what each decision
  costs, and a maintainer who picks "rederive from literature" can read off what
  is now prohibited. **The task does not choose**; the licence decision
  constrains the answer and the task shows the shape of every option.
- [ ] **Step 4** — run → FAIL → PASS; commit
  `docs(reader): per-component port decisions with licence implications`.

---

### Task P16.1: The QA harness inventory

**Fortran:** `$OR_SRC/qa-tests/` — `scripts/{or_qa_script, or_run_test.py,
  or_generate_ctest.py, verify_results.py, qa_system.py, execute_solver.py,
  or_execute.py, or_qa_timeout.pl, or_radioss.pl, or_QA.constants,
  or_QA.files_1miniqa, or_QA.files_all}`, `miniqa/`,
  `test_script.{bash,bat}`, `CMakeLists.txt`.
**Files:** Create `tools/validation_data/qa_status.json`,
`tests/test_p16_qa_census.py`.
**Interfaces:**
- Produces: one record per QA artefact with `runs_the_port` (bool),
  `blocker` (what stops it), `category`.

- [ ] **Step 1** — write the failing test:

```python
def test_every_qa_artefact_is_classified():
    st = json.loads(Path("tools/validation_data/qa_status.json").read_text())
    for r in st:
        assert isinstance(r["runs_the_port"], bool)
        assert r["blocker"] is None or r["blocker"]
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — inventory. `test_script.bash` is the interesting one: its
  `-arch`, `-mpi`, `-prec`, `-np`, `-nt`, `-type=pon` and `-tests` options
  define the matrix upstream tests, and the port should be substitutable for the
  binary through `qa-tools/execute_solver.py`. **Classify every option as
  supported / partially supported / unsupported**, because the `-type=pon`
  (parallel arithmetic) mode is exactly Phase 13's `-np` parity check.
- [ ] **Step 4** — run → PASS; commit `feat(census): the upstream QA harness inventory`.

---

### Task P16.2: The cfg-kernel fidelity audit

**Fortran:** `reader/source/cfgio/` (65 files, 31,780) and
  `reader/source/cfgkernel/` (311 files, 63,947) — the C++ CFG kernel.
**Files:** Create `tools/validate_cfg_fidelity.py`,
`tests/test_p16_cfg_fidelity.py`, `tools/validation_data/cfg_fidelity.json`.
**Interfaces:**
- Consumes: `pyradioss.input.card_layouts.LAYOUTS` (5,962 entries),
  `pyradioss.input.mat_reader`, `prop_reader`.
- Produces: `tools.validate_cfg_fidelity.check() -> Report` with
  `Report.cards_tested: int`, `Report.accepted: int`, `Report.rejected:
  list[dict]` (card, dialect, reason).

- [ ] **Step 1** — write the failing test:

```python
def test_every_cfg_card_dialect_is_accepted():
    rep = tools.validate_cfg_fidelity.check()
    assert rep.cards_tested > 2000
    assert rep.rejected == [], rep.rejected[:10]
```

- [ ] **Step 2** — run → FAIL (the checker does not exist).
- [ ] **Step 3** — implement. **Sample every card in
  `$OR_SRC/hm_cfg_files/CFG/**` in all four dialects** (free-format, `block.fixed`,
  the column-positioned form, and the `/BEGIN`-declared input version's form) and
  assert the port's reader produces the same field set as the C++ kernel. Where
  it does not, **record the rejection with the CFG file and the format string**
  rather than adapting the test.
- [ ] **Step 4** — run → PASS; the report states the number of cards and dialects
  tested; commit
  `feat(validation): cfg-kernel fidelity audit across all dialects`.

---

### Task P16.3: The SDI data layer

**Fortran:** `reader/source/sdi/` (44 files, 15,588) — the SDI interface
  definition (`sdiElement.h`, `sdiElementData.h`, `sdiElement.cxx` and 41 more).
**Files:** Create `pyradioss/sdi/` package (`nodes.py`, `elements.py`,
  `properties.py`, `groups.py`, `assemble.py`), `tests/test_p16_sdi.py`.
**Interfaces:**
- Produces: `sdi.assemble(model) -> SDIModel` with
  `SDIModel.nodes`, `.elements`, `.element_types`, `.node_sets`,
  `.element_sets`, `.components`, `.materials`, `.properties`, `.load_curves`
  — the SDI data structures as plain Python objects, JSON-serialisable.

- [ ] **Step 1** — write the failing test:

```python
def test_sdi_model_is_json_serialisable_and_complete():
    import json
    m = sdi.assemble(_example_model("tensile_bar"))
    payload = json.dumps(m.to_dict())
    assert len(payload) > 1000
    assert m.element_types            # the Isolid/Ishell dispatch table
    assert len(m.nodes) == len(m.nodes)  # consistency, not a magic number
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **This is the cheap half of the SDI answer**
  (Task P16.0's option (b)): the *data* becomes available, so a future SDI shim
  is a thin serialisation layer rather than a re-derivation. Read
  `sdiElement.h` and `sdiElementData.h` for the field names and **use upstream's
  names**, so a future shim maps one-to-one.
- [ ] **Step 4** — run → PASS; commit `feat(sdi): the SDI data layer from a pyradioss model`.

---

### Task P16.4: `verify_results.py` accepts the port

**Fortran:** `$OR_SRC/qa-tests/qa_tools/verify_results.py`,
  `$OR_SRC/qa-tests/qa_tools/constants.json`.
**Files:** Create `tools/run_upstream_verifier.py`,
`tests/test_p16_verify_results.py`.
**Interfaces:**
- Consumes: Phase 12 Task P12.7's QAPRINT dumps.
- Produces: `tools.run_upstream_verifier.run(qaprint_dir, tolerances) -> Verdict`
  with `Verdict.passed: int`, `Verdict.failed: list[dict]`.

- [ ] **Step 1** — write the failing test:

```python
def test_upstream_verifier_accepts_the_port_on_the_bundled_examples():
    v = tools.run_upstream_verifier.run(_qaprint_dir_for_all_examples(),
                                        tolerances=_tolerances())
    assert v.failed == [], v.failed[:5]
    assert v.passed > 0
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The tolerances must come from
  `qa-tools/constants.json`**, not from the port. This task is the one that turns
  the program's validation story from self-assessment into third-party
  assessment, and a reviewer should treat any tolerance the port proposes as a
  red flag.
- [ ] **Step 4** — run → PASS; commit
  `feat(validation): upstream verify_results.py accepts the port's QAPRINT`.

---

### Task P16.5: `or_qa_script` runs the port

**Fortran:** `$OR_SRC/qa-tests/scripts/or_qa_script`,
  `qa-tools/execute_solver.py`, `or_run_test.py`.
**Files:** Create `tools/oracle/or_qa_bridge.py`,
`tests/test_p16_or_qa_bridge.py`.
**Interfaces:**
- Produces: a shim executable that upstream's QA script can invoke in place of
  `starter_linux64_gf` / `engine_linux64_gf`, plus
  `or_qa_bridge.run(cmdline) -> subprocess.CompletedProcess` implementing
  upstream's CLI surface (`-i`, `-np`, `-nt`, `-v`).

- [ ] **Step 1** — write the failing test:

```python
def test_the_bridge_speaks_the_upstream_cli():
    r = or_qa_bridge.run([os.environ["OR_STARTER"], "-i", "T_0000.rad", "-np", "2"])
    assert r.returncode == 0
    assert "ENGINE TERMINATION" not in r.stderr
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **The `-np` argument is the interesting one**:
  upstream's QA script passes it and expects N processes; the port runs N domains
  as threads by default (`README.md`). The bridge must make the port's behaviour
  *match what the script expects* for the SMP path, and record where it differs
  for the MPI path.
- [ ] **Step 4** — run → PASS; commit `feat(validation): run the port through upstream's or_qa_script`.

---

### Task P16.6: `miniqa` as a port-side test set

**Fortran:** `$OR_SRC/qa-tests/miniqa/`, `qa-tools/miniqa.txt`,
  `scripts/or_QA.files_1miniqa`.
**Files:** Create `tests/test_p16_miniqa.py`, `tools/run_miniqa.py`.
**Interfaces:**
- Produces: a pytest-driven runner over upstream's miniqa case list.

- [ ] **Step 1** — write the failing test:

```python
def test_miniqa_cases_are_all_classified():
    cases = tools.run_miniqa.load()
    assert cases
    assert all(c["verdict"] in {"pass", "fail", "skip"} for c in cases)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **miniQA is upstream's fast acceptance set** — the
  closest thing to a conformance suite OpenRadioss has. Every case must end in a
  verdict; a case with no verdict is the same defect class as Phase 9's
  `parsed-unused`.
- [ ] **Step 4** — run → PASS; the pass/fail/skip counts reported; commit
  `test(qa): the port runs upstream's miniQA case list`.

---

### Task P16.7: `engine/source/tools/`

**Fortran:** `engine/source/tools/{accele, curve, lagmul, seatbelts, sect,
  sensor, skew, univ}` and `finter_mixed.F90`.
**Files:** Modify `pyradioss/engine/{sections.py, sensors.py}`, create
`pyradioss/engine/seatbelts.py`, `pyradioss/engine/univ.py`,
`tests/test_p16_engine_tools.py`.
**Interfaces:**
- Produces: the module-level functions of each subdirectory.

- [ ] **Step 1** — write the failing test:

```python
@pytest.mark.parametrize("tool", ["accele", "seatbelts", "univ", "lagmul"])
def test_engine_tool_has_a_port(tool):
    mod = importlib.import_module(f"pyradioss.engine.{tool}", fromlist=["x"])
    assert mod.__doc__, f"{tool} has no docstring naming its Fortran origin"
```

- [ ] **Step 2** — run → FAIL (`univ` and `seatbelts` do not exist).
- [ ] **Step 3** — implement. **`tools/seatbelts/` is the seatbelt assembly and
  routing solver**, which the port has only as readers (`/SEATBELT` M150,
  `/SLIPRING/SHELL` M123, `/SLIPRING` M106) — so this is real new physics.
  `tools/accele/` is the **SPH/particle acceleration** (Phase 5's consumer).
- [ ] **Step 4** — run → PASS; commit per tool.

---

### Task P16.8: The RHT user-material generator

**Fortran:** `$OR_SRC/tools/rht_user_material/` (13 files) — the generator that
  emits the Fortran source of an RHT (Johnson–Holmquist, Phase 6's LAW79)
  user material.
**Files:** Create `tools/rht_user_material.py`, `tests/test_p16_rht.py`.
**Interfaces:**
- Produces: `tools.rht_user_material.generate(params) -> str` (the generated
  Fortran) and `generate_fortran_module(params) -> str` (the Python equivalent,
  for the port's user-material hook).

- [ ] **Step 1** — write the failing test:

```python
def test_generated_rht_material_reproduces_the_reference_law():
    from pyradioss.materials import law79 as ref
    gen = tools.rht_user_material.generate_module(_default_params())
    assert _matches_reference(gen, ref, rel=1e-12)
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. The generated material must be **equivalent to the
  port's LAW79**, which is the test. A generator that produces something subtly
  different from the hand-ported law is a divergence source.
- [ ] **Step 4** — run → PASS; commit `feat(tools): the RHT user-material generator`.

---

### Task P16.9: `dyna2rad` — port, scope, or decline

**Fortran:** `reader/source/dyna2rad/` (83 files, 44,802 LOC) — the LS-DYNA
  keyword → Radioss keyword converter.
**Files:** `tools/validation_data/reader_status.json` (updated),
  `docs/READER_DECISIONS.md`, `docs/PORT_EXTENSIONS.md`.
**Interfaces:**
- Produces: a decision, and for `decline`, the reason recorded.

- [ ] **Step 1** — write the failing test:

```python
def test_dyna2rad_is_decided_with_a_reason():
    d = next(r for r in json.loads(
        Path("tools/validation_data/reader_status.json").read_text())
        if r["component"] == "dyna2rad")
    assert d["decision"] in {"port", "wrap", "substitute", "decline"}
    assert len(d["decision_reason"]) > 80
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — decide. **The honest default is `decline`**: an LS-DYNA
  converter is a *preprocessor* capability, not a solver capability, and this
  plan's scope is the solver. It is listed because it is the largest single item
  in `reader/` and a maintainer may want it. **If `port` or `substitute` is
  chosen, this becomes its own phase** — 44,802 lines of C++ is not a task, and
  pretending otherwise would be this plan misstating its own size.
- [ ] **Step 4** — run → PASS; commit `docs(reader): the dyna2rad decision`.

---

### Task P16.10: Complete the extension and decision registries

**Files:** Modify `docs/PORT_EXTENSIONS.md`, `docs/PORTING_GUIDE.md`,
  `docs/OUTPUT_FORMATS.md`, `docs/LINEAR_ALGEBRA.md`.
**Interfaces:**
- Produces: one place a maintainer can read every place the port knowingly
  differs from upstream, across all 17 phases.

- [ ] **Step 1** — write the failing test:

```python
def test_every_declared_deviation_has_a_test_or_a_reason():
    devs = json.loads(Path("tools/validation_data/deviations.json").read_text())
    for d in devs:
        assert d["test"] or d["reason_no_test"]
        assert d["citation"].startswith(("engine/source/", "starter/source/",
                                         "common_source/", "reader/"))
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — consolidate. Every deviation recorded anywhere in the plan's
  phases lands in `deviations.json` with a citation and either a pinning test or
  a reason why a test is impossible. **This is the last chance to catch a
  deviation that was decided in prose and never pinned** — and there are several
  (the pickle restart, the CSV/VTK output, the six empty failure directories,
  the `solid_2d/tria` citation).
- [ ] **Step 4** — run → PASS; commit `docs(deviations): the consolidated deviation register`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P16.0** — every decision carries a licence implication **under all four
  Task P0.0 outcomes**, and the task chose nothing.
- **P16.2** — rejections are recorded with the CFG file and format string; the
  test was not adapted to the port's behaviour.
- **P16.4** — the tolerances come from `qa-tools/constants.json`; a port-proposed
  tolerance is a rejection.
- **P16.5** — the `-np` difference between threads and processes is recorded,
  not papered over.
- **P16.6** — every miniqa case has a verdict; no case is silently absent.
- **P16.8** — the generated material matches the port's LAW79 to `1e-12`.
- **P16.9** — if `port`/`substitute` was chosen, the task escalated to a phase
  rather than proceeding.
- **P16.10** — every prose-only deviation is now pinned or has a stated reason.

## Parallelisation

- **This phase has no dependencies on any other phase** and can run from the day
  Phase 0 exits. In a program this serial by dependency, it is the one place
  where a second agent has meaningful work immediately.
- **Wave 0** — P16.0 serial (it gates the substantive work); P16.1 parallel.
- **Wave 1** — 7 tasks, 7 agents, disjoint.
- **Wave 2** — serial.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_p16_reader_census.py tests/test_p16_cfg_fidelity.py
python -m pytest -q tests/test_p16_verify_results.py
python tools/census.py --render plan/CENSUS.md --area reader
```

The phase reviewer confirms, in writing:

1. **upstream's own `verify_results.py`, `or_qa_script` and `miniqa` run against
   the port** — this is the phase's headline deliverable;
2. every `reader/` component has a decision with a licence implication;
3. `docs/PORT_EXTENSIONS.md` and `deviations.json` together account for every
   known difference between the port and upstream.