# Phase 1 — Foundation

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** remove the structural blockers that would make every later phase
unmaintainable or unverifiable — the platform lock, the two 50k–97k-line
monoliths, the silent path fallbacks, the frozen-subsurface marker, and the
absence of a machine-readable statement of *what is actually ported*.

**Architecture:** split the two monoliths into packages with unchanged public
API; route every external path through `pyradioss.paths`; generate
`tools/census.py` → `plan/CENSUS.md` so every phase knows its true scope;
freeze the implicit tower behind a pinned surface test; add a module-size
linter and a regression ledger so "no new failures" is measurable.

**Tech stack:** Python 3.12 dev box (`requirements-lock.txt` §[B]), pytest,
`tomllib`, `pathlib`.
**Spec:** `plan/00_ORCHESTRATION.md` §3, §5, §6, §11.
**Entry criterion: Phase 0 exit gate passed.**
**Band: M** — small, but nothing downstream is safe without it.

---

## Scope

| Item | Value |
|---|---|
| Upstream paths cited | `$OR_SRC` docstrings and `INSTALL.md` only — **no upstream code is ported in this phase** |
| Python produced | ~1,900 LOC (mostly moved, not new) |
| Tests | ~60 |

## Gap analysis

Measured 2026-10-02:

1. **`pyradioss/input/starter_keywords.py` is 97,102 lines with 1,522 `def`s**,
   ending in a `KEYWORD_PARSERS` dispatch table whose `parse_starter_deck`
   sits at line 96,983. It is unmergeable-by-review and un-navigable.
2. **`pyradioss/model/entities.py` is 53,286 lines with 3,045 classes /
   dataclasses.** A dataclass change to one entity cannot be reviewed in
   isolation.
3. **`pyradioss/materials/__init__.py` is 5,401 lines** — the law registry.
4. **Dozens of `SyntaxWarning: "\O" is an invalid escape sequence` fire at
   import**, from docstrings citing `C:\OpenRadioss\source\...`. These are
   latent errors, not cosmetic.
5. **`AGENTS.md` is Windows-only**: `.venv\Scripts\python.exe`,
   `C:\OpenRadioss`, `.ps1` helpers, "no Unix commands".
6. **`docs/STATE.md`'s baseline is internally inconsistent** — it reports the
   fast tier as `11182 passed` (2026-09-12), then
   `13030 passed / 13 failed` (2026-09-21), then a full-suite figure of
   `9893 passed` (2026-09-11), against `11205 collected`. Meanwhile the tree
   collects **14,267** non-slow tests today. There is no single trustworthy
   regression ledger.
7. **No census.** `plan/CENSUS.md` does not exist; "what is ported" is known
   only from prose in `docs/STATE.md` and `PORTING_GUIDE.md`.
8. **The frozen implicit surface is unenforced** — nothing prevents an agent
   from adding or deleting a module under `pyradioss/implicit/`.
9. **CI is Windows-shaped**: `.github/workflows/ci.yml` has a
   `numpy-only smoke` job and dropped one; `OPEN_BUGS.md` item 6 records that
   two LAW4 cfg tests pass in CI but fail in the Linux dev container — i.e.
   the Linux path is *less* tested than the Windows path.

## Wave graph

```
Wave 0 (parallel — disjoint new files)
  P1.0 AGENTS.md rewrite      P1.1 escape-sequence sweep
  P1.2 citation normalisation P1.3 census generator
  P1.6 frozen implicit surface P1.7 module-size linter
  P1.9 CI Linux matrix        P1.8 regression ledger
   ── gate: fast tier has the same known failures and no more
Wave 1 (SERIAL — shared files; one owner each)
  P1.4 split model/entities.py        (internally: 8 agents, disjoint new files)
  P1.5 split input/starter_keywords.py(internally: 4 agents + lead owns dispatch.py)
  P1.10 documentation contract
   ── gate: full suite byte-identical to the pre-split baseline
```

Task numbering within this file: **P1.0 – P1.10**. Wave 0 is P1.0, P1.1, P1.2,
P1.3, P1.6, P1.7, P1.8, P1.9. Wave 1 is P1.4, P1.5, P1.10.

---

### Task P1.0: Make `AGENTS.md` platform-neutral

**Fortran:** none. **Files:** Modify `AGENTS.md` (whole file).
**Interfaces:**
- Consumes: `plan/00_ORCHESTRATION.md` §4.
- Produces: the repository rule file, matching `00_ORCHESTRATION.md` §4 on every
  point, with the Windows forms kept as a labelled compatibility section.

- [ ] **Step 1** — write the failing test:

```python
def test_agents_md_has_no_windows_only_assumptions():
    text = Path("AGENTS.md").read_text()
    for stale in (r".venv\Scripts\python.exe", r"C:\OpenRadioss",
                  "No WSL, no Git Bash, no Unix commands"):
        assert stale not in text
    for required in ("OR_SRC", "OR_ROOT", "PYRADIOSS_HM_CFG", "-m \"not slow\""):
        assert required in text
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — rewrite. Required content:
  - **Environment:** Linux-first. `$PYTHON` is `.venv/bin/python` on POSIX and
    `.venv\Scripts\python.exe` on Windows. Commands are given as `$PYTHON …` and
    resolved by the shell profile, never hardcoded.
  - **READ-ONLY list**, now `$OR_SRC` and `$OR_ROOT` — replacing the old
    `C:\OpenRadioss`, `C:\OpenRadioss_old`, `E:\` entries.
  - **Terminal rules:** keep "one flat command per call"; replace "no `&&`, no
    pipes" with the platform-neutral version (no `&&`, no redirection; pipes are
    fine on POSIX and the rule exists so a command stays copy-pasteable).
  - **New section: the oracle.** `$OR_STARTER` / `$OR_ENGINE` come from
    `tools/oracle/oracle_env.sh`; a numerics change requires a parity run.
  - **New section: milestones.** "Work ONE milestone per conversation" becomes
    "work ONE phase task per conversation"; milestone numbering is M700+.
  - **Model policy:** space-bunny, never a Fast variant, for every subagent and
    reviewer.
  - Keep the domain rules verbatim (Fortran wins, energy ledger, no weakened
    tests, `@pytest.mark.slow` at 60 s, pin `PYRADIOSS_BACKEND=numpy`).
- [ ] **Step 4** — run → PASS; run the fast tier; commit
  `docs(agents): make the agent contract platform-neutral (Linux-first)`.

---

### Task P1.1: Escape-sequence sweep

**Files:** Modify every `pyradioss/**/*.py` docstring that contains a Windows
path; Create `tests/test_p1_no_windows_escapes.py`.
**Interfaces:**
- Consumes: nothing. **Files:** grep-driven; the test is the deliverable that
  makes it stay fixed.

- [ ] **Step 1** — write the failing test:

```python
import warnings, py_compile, pathlib

def test_no_module_emits_an_escape_syntaxwarning():
    offenders = []
    for p in pathlib.Path("pyradioss").rglob("*.py"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            py_compile.compile(str(p), doraise=True, cfile="/tmp/x.pyc")
        offenders += [(str(p), str(x.message)) for x in w
                      if "escape sequence" in str(x.message)]
    assert offenders == [], offenders[:5]
```

- [ ] **Step 2** — run → FAIL; it will name `law119_seatbelt.py`, `law87_barlat2000.py`,
  `flex_body.py` and others (observed at import on 2026-10-02).
- [ ] **Step 3** — convert every affected docstring to a **raw** string
  (`r"""…"""`). **Do not** rewrite the citations to Linux paths in this task —
  the citations still name the Windows reference tree they were written against,
  and normalising them to `$OR_SRC` is Task 2.3's job. Converting
  `C:\OpenRadioss\source\OpenRadioss-latest-20260520\…` to `$OR_SRC/…` here
  would be a second change in one commit.
- [ ] **Step 4** — run → PASS; `python -W error::SyntaxWarning -c "import pyradioss.starter"`
  → clean; commit `fix(docstrings): raw-string the Windows-path citations`.

---

### Task P1.2: Registry-led `C:\` → `$OR_SRC` citation normalisation

**Files:** Modify docstrings across `pyradioss/`; Create
`tools/normalize_citations.py`, `tests/test_p1_citations.py`.
**Interfaces:**
- Consumes: `pyradioss.paths.or_src()`.
- Produces: `tools.normalize_citations.rewrite(text: str) -> str`, and the
  invariant that **no** module docstring names a path outside `$OR_SRC`
  (excluding the Windows-compat note `00_ORCHESTRATION.md` §4.1 mentions).

- [ ] **Step 1** — write the failing test:

```python
def test_no_citation_names_a_windows_reference_tree():
    bad = []
    for p in pathlib.Path("pyradioss").rglob("*.py"):
        for m in re.finditer(r"[A-Z]:\\\\?OpenRadioss[^`\"'\s]*", p.read_text()):
            bad.append((str(p), m.group(0)))
    assert bad == [], bad[:10]
```

- [ ] **Step 2** — run → FAIL (hundreds of hits).
- [ ] **Step 3** — implement `rewrite()` to map
  `C:\OpenRadioss\source\OpenRadioss-latest-20260520\<rel>` →
  `$OR_SRC/<rel>` and `C:\OpenRadioss\hm_cfg_files` → `$OR_SRC/hm_cfg_files`,
  and to flag (not rewrite) any path it does not recognise. Apply it in one
  commit per `pyradioss/` subpackage so the diff stays reviewable.
- [ ] **Step 4** — run → PASS; commit
  `refactor(docs): normalise every Fortran citation to the $OR_SRC form`.

---

### Task P1.3: The machine-readable census

**Fortran:** every `$OR_SRC` source file, via the embedded provenance block —
the `!||` header that names the routine, its own path, `--- called by ---` and
`--- calls ---` (format read off
`$OR_SRC/engine/source/elements/solid/solide/sforc3.F:24-60`).
**Files:** Create `tools/census.py`, `tests/test_p0_census.py`,
`tools/validation_data/census.json` (generated), `plan/CENSUS.md` (generated).

**Interfaces** — this is the single most-reused interface in the program:

- `tools.census.scan() -> Census` with fields `files: dict[str, FileRecord]`.
- `tools.census.FileRecord` fields: `path: str` (relative to `$OR_SRC`),
  `lines: int`, `calls: tuple[str, ...]`, `called_by: tuple[str, ...]`,
  `area: str` (the top-level subsystem), `python_module: str | None`,
  `status: Literal["ported","partial","stub","missing"]`.
- `tools.census.render(c: Census) -> str` — the markdown table.
- `tools.census.main(argv) -> int` — `--render PATH`, `--json PATH`,
  `--area AREA`, `--status-missing`.

- [ ] **Step 1** — write the failing test:

```python
def test_census_finds_every_fortran_source_file():
    c = census.scan()
    assert len(c.files) > 5000
    assert c.files["engine/source/engine/resol.F"].lines > 100
    assert "forint" in c.files["engine/source/elements/solid/solide/sforc3.F"].called_by

def test_census_classifies_the_output_tree_as_missing():
    c = census.scan()
    anim = c.files["engine/source/output/th/th1t.F90"] if "engine/source/output/th/th1t.F90" in c.files else None
    assert anim is not None and anim.status in {"missing", "partial"}
```

- [ ] **Step 2** — run → FAIL (module absent).
- [ ] **Step 3** — implement `scan()`. Parse the `!||` block with the layout
  read off `sforc3.F` above: line 1 is the routine name, line 2 the
  self-path, then `--- called by ---` and `--- calls ---` sections of
  `name<spaces>path`. Map `python_module` by scanning the port's own docstrings
  for the same `$OR_SRC` path — this is why Task 1.2 must land first. Derive
  `status` from a per-module allowlist file `tools/validation_data/port_status.json`
  that later phases update; **an unlisted file is `missing`, never `ported`.**
- [ ] **Step 4** — run → PASS; render `plan/CENSUS.md`; commit
  `feat(census): machine-readable Fortran<->Python coverage census`.

---

### Task P1.4: Split `pyradioss/model/entities.py`

**Files:** Create `pyradioss/model/entities/` package with
`__init__.py`, `nodes.py`, `elements.py`, `materials.py`, `loads.py`,
`constraints.py`, `groups.py`, `contact.py`, `airbag.py`, `misc.py`;
Modify `pyradioss/model/model.py` imports. Delete `entities.py`.
**Interfaces:**
- Consumes: `pyradioss.model.entities` — its ~300 exported class names.
- Produces: `pyradioss.model.entities` with **byte-identical** exported names,
  via a re-export in `__init__.py`. `from pyradioss.model.entities import Part`
  must keep working, and `pyradioss.model.entities.Part.__module__` is allowed
  to change (nothing may depend on it).

- [ ] **Step 1** — write the failing test that **fails today** by construction:

```python
def test_entities_is_a_package_not_a_file():
    import pyradioss.model.entities as E
    assert E.__path__, "entities must be a package so it can be split"
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — the split. **Three constraints the implementer must respect:**
  1. **This is a pure move.** No logic change, no renaming, no reordering of
     fields. Dataclass field *order* changes `__init__` positional semantics.
  2. Watch for **import cycles**: the file's classes reference each other
     freely (e.g. `Part` → `Material`, `Property`). Break cycles with
     `if TYPE_CHECKING:` imports plus string annotations, not by reordering.
  3. Capture the export list **before** moving:
     `python -c "import pyradioss.model.entities as E; print(sorted(n for n in dir(E) if not n.startswith('_')))" > /tmp/exports.txt`,
     and assert it is unchanged afterwards.
  Suggested ownership split: `nodes` (NodeGroup, Surface, Line, Box, EntityGroup,
  InitialVelocity), `elements` (Part, Property and its subclasses), `materials`
  (Material, FailureModel, EquationOfState), `constraints` (BoundaryCondition,
  Mpc, RigidBody, Rbe3, RigidWall), `loads`, `contact`, `airbag`, `misc`.
- [ ] **Step 4** — run → PASS; `diff` the two export lists → empty; run the
  **full** suite (not just fast tier) → identical pass/skip/fail counts;
  commit `refactor(model): split entities.py into a package, API unchanged`.

---

### Task P1.5: Split `pyradioss/input/starter_keywords.py`

**Fortran:** the per-keyword readers under `$OR_SRC/starter/source/**/reader/`
(29 starter element/interface/material/property reader directories, 4,970 LOC
in `starter/source/elements/reader` alone, 59,973 in
`starter/source/interfaces/inter3d1`). Each Python `read_*` function's
docstring already names its `hm_read_*.F`; that mapping drives the split.
**Files:** Create `pyradioss/input/keywords/` with
`__init__.py`, `dispatch.py`, `geometry.py`, `elements.py`, `materials.py`,
`properties.py`, `interfaces.py`, `constraints.py`, `loads.py`, `conditions.py`,
`groups.py`, `output.py`, `misc.py`; delete `starter_keywords.py`.
**Interfaces:**
- Consumes: `pyradioss.input.starter_keywords`' public surface —
  `parse_starter_deck(blocks, model=None, log=None) -> Model`,
  `read_starter_deck(...)`, `KEYWORD_PARSERS`, and every public `read_*`.
- Produces: `pyradioss.input.keywords` exporting the same names;
  `pyradioss.input.starter_keywords` becomes a thin re-export shim for one
  release cycle, emitting `DeprecationWarning` on import.

- [ ] **Step 1** — write the failing test:

```python
def test_starter_keywords_is_split():
    import pyradioss.input.keywords as K
    assert len(K.KEYWORD_PARSERS) > 900        # 1,522 defs today; count is exact, see below
    assert callable(K.parse_starter_deck)
```

- [ ] **Step 2** — run → FAIL (`pyradioss.input.keywords` absent).
- [ ] **Step 3** — split. **The dispatcher is the hard part and it is its own
  file.** `KEYWORD_PARSERS` is keyed by the underscore-joined keyword
  (`"_".join(block.parts).upper()`) with two fallbacks (`parts[:2]`, then
  `key0` alone), then the `ENGINE_KEYWORDS_IGNORE` bypass, then the
  "not ported — block skipped" warning. **Copy that resolution logic verbatim**
  (it is at `starter_keywords.py:96983-97040`); the split must not change which
  keyword resolves to which parser. To prove it, capture the mapping before and
  after:

```python
before = {k: f"{v.__module__}.{v.__qualname__}" for k, v in KEYWORD_PARSERS.items()}
# after the split, compare by QUALNAME only — the module path legitimately changed
```

  Split by **keyword family**, assigned so no two agents touch the same file;
  the four agents in this task own `geometry+elements`, `materials+properties`,
  `interfaces+constraints`, `loads+conditions+groups+output+misc`.
- [ ] **Step 4** — run → PASS; assert `set(before) == set(after)` and
  `set(before.values()) == set(after.values())` on qualnames; full suite counts
  identical; commit
  `refactor(input): split starter_keywords.py into a keywords package`.

---

### Task P1.6: Freeze the implicit tower's surface

**Files:** Create `pyradioss/implicit/_frozen_surface.py`,
`tests/test_p1_frozen_implicit.py`.
**Interfaces:**
- Produces: `pyradioss.implicit._frozen_surface.FROZEN_MODULES: frozenset[str]`
  — the 37 module names currently in `pyradioss/implicit/` — and
  `FROZEN_SURFACE_SHA256: str` over the sorted names.

- [ ] **Step 1** — write the failing test:

```python
def test_implicit_surface_is_frozen():
    from pyradioss.implicit import _frozen_surface as F
    import pkgutil, pyradioss.implicit as I
    present = {m.name for m in pkgutil.iter_modules(I.__path__)
               if not m.name.startswith("_")}
    assert present == set(F.FROZEN_MODULES), {
        "added": sorted(present - set(F.FROZEN_MODULES)),
        "removed": sorted(set(F.FROZEN_MODULES) - present)}

def test_frozen_implicit_modules_still_import():
    import importlib, pytest
    for name in sorted(__import__(
            "pyradioss.implicit._frozen_surface", fromlist=["x"]).FROZEN_MODULES):
        importlib.import_module(f"pyradioss.implicit.{name}")
```

- [ ] **Step 2** — run → FAIL (`_frozen_surface` absent).
- [ ] **Step 3** — generate `FROZEN_MODULES` from
  `pkgutil.iter_modules`, and record in the module docstring that the list came
  from the 2026-10-02 census, that adding a name is a deliberate act requiring
  the maintainer's approval, and that removing one is a hard failure.
- [ ] **Step 4** — run → PASS; commit
  `test(implicit): pin the frozen tower surface`.

---

### Task P1.7: Module-size linter

**Files:** Create `tests/test_p1_module_size.py`,
`tools/validation_data/size_budget.json`.
**Interfaces:**
- Produces: `tools.validation_data.size_budget.json` — `{"<path>": <max_lines>}`
  with an entry for **every** current `pyradioss/**/*.py`, plus the rule
  `new_files_default_max_lines = 1200`.

- [ ] **Step 1** — write the failing test:

```python
def test_no_module_grew_past_its_budget():
    budget = json.loads(Path("tools/validation_data/size_budget.json").read_text())
    over = {}
    for rel, limit in budget.items():
        n = len(Path(rel).read_text().splitlines())
        if n > limit:
            over[rel] = (n, limit)
    assert over == {}, over
```

- [ ] **Step 2** — run → FAIL (no budget file).
- [ ] **Step 3** — generate the budget at the current sizes, **rounded down to
  the current line count** (never up — a budget may only shrink), and add a
  second test that any **new** module under 1,200 lines is rejected by a
  `test_no_oversized_new_module` scan of git-tracked files against
  `git ls-files --diff-filter=A`. The two remaining monsters
  (`input/starter_keywords.py` at 97,102 and `model/entities.py` at 53,286) get
  their **post-split** sizes once Tasks 1.4/1.5 land.
- [ ] **Step 4** — run → PASS; commit
  `test(quality): per-module line budget that can only shrink`.

---

### Task P1.8: Regression ledger

**Files:** Create `tools/regression_ledger.py`, `tests/test_p1_regression_ledger.py`,
`tools/validation_data/baseline.json`.
**Interfaces:**
- Produces:
  - `tools.regression_ledger.collect() -> Ledger` with
    `Ledger.passed: int`, `.failed: tuple[str, ...]` (node ids),
    `.skipped: int`, `.errors: tuple[str, ...]`, `.xfailed: int`.
  - `tools.regression_ledger.compare(base: Ledger, now: Ledger) -> Delta` with
    `Delta.new_failures`, `Delta.fixed`, `Delta.new_errors`,
    `Delta.unknown` (a node id present in neither — a rename).
  - `tools.regression_ledger.main(argv) -> int` — `--record PATH`,
    `--check PATH`, exit 1 on any new failure.

- [ ] **Step 1** — write the failing test:

```python
def test_ledger_flags_a_new_failure():
    base = Ledger(passed=10, failed=("tests/test_a.py::t1",), skipped=0,
                  errors=(), xfailed=0)
    now = Ledger(passed=10, failed=("tests/test_a.py::t1",
                                    "tests/test_b.py::t2"), skipped=0,
                 errors=(), xfailed=0)
    d = compare(base, now)
    assert d.new_failures == ("tests/test_b.py::t2",)
    assert d.fixed == ()

def test_ledger_distinguishes_a_rename_from_a_new_failure():
    base = Ledger(passed=1, failed=("tests/test_a.py::old_name",), skipped=0, errors=(), xfailed=0)
    now  = Ledger(passed=1, failed=("tests/test_a.py::new_name",), skipped=0, errors=(), xfailed=0)
    assert compare(base, now).unknown == ()
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — implement. **A rename must not read as a new failure and must
  not read as a fix** — that is the `unknown` bucket, and an `unknown` in the
  middle of a path's node ids is a rename; a genuinely deleted test is a
  deliberate act that needs its own ledger entry. Then record the real
  baseline with `--record`: the current tree's fast-tier result, which the
  ledger must capture **exactly**, including the 13 known failures documented in
  `docs/STATE.md:34` and `tests/conftest.py`'s `_KNOWN_XFAIL`.
- [ ] **Step 4** — run → PASS; commit
  `feat(qa): regression ledger so "no new failures" is measurable`.

---

### Task P1.9: CI on Linux

**Files:** Modify `.github/workflows/ci.yml`, `.github/workflows/parity.yml` (new).
**Interfaces:**
- Produces: a Linux job that runs `pytest -q -m "not slow"` and **fails on any
  new failure** via `tools/regression_ledger.py --check`; and a parity job that
  runs only when `OR_STARTER`/`OR_ENGINE` secrets/paths are configured (skipped
  otherwise, with the skip reason visible in the log).

- [ ] **Step 1** — write the failing test:

```python
def test_ci_has_a_linux_job_that_gates_on_the_ledger():
    text = Path(".github/workflows/ci.yml").read_text()
    assert "runs-on: ubuntu" in text
    assert "regression_ledger.py" in text
    assert "-m \"not slow\"" in text
```

- [ ] **Step 2** — run → FAIL (current CI has no ledger gate).
- [ ] **Step 3** — rewrite the workflow. **The parity job must not be a
  lie:** it is `if: ${{ vars.OR_STARTER != '' }}`, and when skipped it prints
  `::notice::parity job skipped — OR_STARTER not configured`. The LAW4 cfg
  discrepancy (`docs/OPEN_BUGS.md` item 6 — passes in CI, fails in the Linux dev
  container) is the reason Linux CI is a gate rather than a convenience.
- [ ] **Step 4** — run → PASS; commit `ci: Linux job gated on the regression ledger`.

---

### Task P1.10: Documentation contract

**Files:** Modify `docs/STATE.md`, `README.md`; Create `UPDATES.md`.
**Interfaces:**
- Produces: `UPDATES.md` (the mandatory agent changelog, newest entry on top);
  `docs/STATE.md` gains a `## Program status` section pointing at `plan/README.md`
  and stating which phase is current, and its `## Baseline` section is corrected
  against the Task 1.8 ledger rather than its three mutually inconsistent
  historical numbers.

- [ ] **Step 1** — write the failing test:

```python
def test_updates_md_is_the_newest_first_changelog():
    text = Path("UPDATES.md").read_text()
    vers = re.findall(r"^## (\d+\.\d+\.\d+)", text, re.M)
    assert vers, "no versioned entries"
    assert vers == sorted(vers, key=lambda v: [int(x) for x in v.split(".")], reverse=True)
```

- [ ] **Step 2** — run → FAIL (no `UPDATES.md`).
- [ ] **Step 3** — write `UPDATES.md` with a `1.0.0` entry summarising Phase 0
  and 1; correct `docs/STATE.md`'s baseline to the ledger's numbers and delete
  the stale 11,182/13,030/9,893 trio, replacing it with one measured line plus a
  pointer to `tools/validation_data/baseline.json`.
- [ ] **Step 4** — run → PASS; commit
  `docs(state): single measured baseline + program changelog`.

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P1.1 / P1.2** — no logic changed; the diff is docstrings only.
- **P1.4 / P1.5** — **byte-identical behaviour**: full suite pass/skip/fail
  counts unchanged, and the exported-name / `KEYWORD_PARSERS` snapshot diffs are
  empty. A reviewer who cannot produce that diff rejects the task.
- **P1.3** — an unlisted upstream file is `missing`, never `ported`; the
  `python_module` mapping is derived from citations, not hand-maintained.
- **P1.6** — the frozen list matches `pkgutil` exactly, both directions.
- **P1.7** — no budget entry was increased.
- **P1.8** — a renamed test lands in `unknown`, not in `new_failures`.
- **P1.9** — the parity job is honestly skipped, not silently green.

## Parallelisation

- **Wave 0** is 7 independent tasks on disjoint files → 7 agents.
- **Wave 1** is serial by construction: P1.3 → (P1.4, then P1.5, each with its
  own internal agent split). P1.4 and P1.5 both move code that other tasks read,
  so they do not overlap. P1.4's and P1.5's *own* internal splits are safe:
  P1.4's agents own disjoint new files and only the phase lead edits
  `model/model.py`; P1.5's four agents own disjoint modules and only the lead
  edits `dispatch.py`.
- `starter_keywords.py` and `entities.py` are contended until their split
  lands; after Task 1.5, `pyradioss/input/keywords/**` is new-file territory
  and Phases 9–12 get parallelism back.

## Exit gate

```bash
python -m pytest -q -m "not slow"
python -m pytest -q
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python tools/census.py --render plan/CENSUS.md
python -m pytest -q tests/test_p0_citation*.py tests/test_p1_*.py
git -C "$OR_SRC" status --porcelain        # still empty
```

The phase reviewer additionally re-runs `git ls-files | wc -l`-based diffs to
confirm the two splits lost no code: `git show --stat HEAD~2..HEAD` must show
moves, not deletions, and a **round-trip check** — re-splitting is idempotent —
must be attempted and its failure explained.