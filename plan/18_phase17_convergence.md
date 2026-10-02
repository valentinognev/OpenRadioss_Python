# Phase 17 — Convergence gate

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**
> This phase is mostly **measurement and judgement**, not implementation.

**Goal:** decide, with evidence, whether `pyradioss` is a full port of
OpenRadioss — and say plainly what it is not.

**Architecture:** six independent measurements, then a whole-program reviewer
who **re-derives** the completion claims rather than trusting the 17 phase
reports. The gate is deliberately designed so that it can fail: a program that
cannot say "not done, here is why" is not being measured.

**Tech stack:** the Phase 0 harness, the census, the regression ledger, the
deviation register.
**Spec:** `plan/README.md` §1 (the scope decision), `plan/00_ORCHESTRATION.md`
  §8.3 (Definition of Done, per program).
**Entry criteria: all 17 prior phases merged.**
**Band: L.**

---

## Scope

**No upstream code is ported in this phase.** Its scope is the six
measurements in §"The gate" below, plus the judgement they support. The
measurement inputs are:

| Input | Produced by | Used for |
|---|---|---|
| `plan/CENSUS.md`, `tools/validation_data/census.json` | Task P1.3, refreshed per phase | G1 |
| `tools/validation_data/coverage_final.json` | `tools/validate_vs_fortran.py --mode coverage` | G2 |
| `tools/validation_data/parity_ledger.json` | `tools/validate_vs_fortran.py --mode parity` | G3 |
| `tools/validation_data/upstream_acceptance.json` | Phase 16 Tasks P16.4–P16.6 | G4 |
| `tools/validation_data/deviations.json` | Phase 16 Task P16.10, consolidated | G5 |
| `tools/validation_data/baseline.json`, `frozen_baseline.json` | Tasks P1.8, P14.6 | G6 |

Baseline for G2 (the last full sweep, **M41, stale**): 529 decks →
13 CLEAN / 440 SKIPS / 76 ERROR / 0 CRASH / 0 TIMEOUT.

## Gap analysis

The gap this phase closes is the one the whole program has: **there is no
single, measured, current statement of what is ported.** `docs/STATE.md`'s
three baseline figures are mutually inconsistent, its "What is implemented"
narrative is dated M41, and `PORTING_GUIDE.md`'s coverage is per-milestone.
Six measurements plus one honest reviewer answer the question; nothing else
does.

## The gate

A full port is claimed **if and only if** all six hold:

| # | Gate | Measurement |
|---|---|---|
| G1 | **Census** | every upstream `.F/.F90/.f90/.c/.cpp` in `starter/source`, `engine/source`, `common_source`, `reader` is `ported`, or has a cited, maintainer-accepted `partial`/`decline` |
| G2 | **Coverage** | every deck in the 529-deck corpus is `CLEAN`, or a `SKIP` with a cited reason and no `ERROR` |
| G3 | **Parity ledger** | every element family, every registered material law and every interface type has a measured rel-RMS against the oracle, with a stated tolerance |
| G4 | **Third-party acceptance** | upstream's `verify_results.py`, `or_qa_script` and `miniqa` run against the port |
| G5 | **Deviation register** | every difference is recorded with a citation, a pinning test or a stated reason |
| G6 | **Regression ledger** | no new failure; the frozen implicit tower's baseline is unchanged |

## Wave graph

```
Wave 0 (parallel — six independent measurements)
  P17.1 G1 census closure        P17.2 G2 corpus sweep
  P17.3 G3 parity ledger         P17.4 G4 upstream acceptance
  P17.5 G5 deviation register    P17.6 G6 regression + freeze check
   ── gate: all six reports produced; disagreements are recorded, not averaged
Wave 1
  P17.7 reconcile the disagreements
  P17.8 the whole-program reviewer re-derives every claim
   ── gate: the reviewer's verdict, recorded verbatim
Wave 2
  P17.9 the published status: docs/STATE.md, README.md, UPDATES.md
```

---

### Task P17.1: G1 — census closure

**Fortran:** every source file in `$OR_SRC/{starter,engine,common_source,reader}`
  (≈ 4,300 files).
**Files:** Create `tools/validate_census_closure.py`,
`tools/validation_data/census_closed.json`, `tests/test_p17_census_closure.py`.
**Interfaces:**
- Consumes: `tools.census.scan()` (Phase 1 Task P1.3) and
  `tools/validation_data/port_status.json`.
- Produces: `closure.report() -> Report` with
  `Report.total: int`, `Report.ported: int`, `Report.partial: list[dict]`,
  `Report.missing: list[dict]`, `Report.declined: list[dict]`,
  `Report.coverage: float`.

- [ ] **Step 1** — write the failing test:

```python
def test_no_upstream_file_is_missing_or_partial_without_acceptance():
    rep = tools.validate_census_closure.report()
    assert rep.missing == [], rep.missing[:20]
    accepted = json.loads(Path("tools/validation_data/accepted_partials.json")
                          .read_text())
    for r in rep.partial:
        assert r["fortran"] in accepted and accepted[r["fortran"]]["who"], r
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — run the closure, then **chase every remaining row to a
  decision.** This is the task where the plan's own optimism meets the tree, and
  the expected outcome is a long list. **Each row needs one of:**
  - `ported` — with the port module and, where physics was involved, a parity
    number;
  - `partial` — with **what** is missing, **why**, and a maintainer's name
    accepting it (the `accepted_partials.json` requirement above is what makes
    this non-negotiable);
  - `decline` — with a reason (e.g. the empty upstream directories, `silecc`,
    the CUDA kernels);
  - `not-source` — a file that is a stub, a `.h` include, or generated.
  **A row that cannot reach any of these four is the finding of this task, and it
  is reported as such.**
- [ ] **Step 4** — run → PASS; commit
  `feat(validation): census closure with an accepted-partials register`.

---

### Task P17.2: G2 — the corpus sweep

**Fortran:** the corpus in `PYRADIOSS_RD_DECKS` / `tests/data/rd_decks`.
**Files:** Modify `tools/validate_vs_fortran.py`; Create
  `tools/validation_data/coverage_final.json`, `tests/test_p17_coverage.py`.
**Interfaces:**
- Produces: the final `coverage_final.json` with the same schema as
  `coverage_results_m41.json` plus a `delta_since_m41` block.

- [ ] **Step 1** — write the failing test:

```python
def test_no_deck_errors_or_crashes():
    d = json.loads(Path("tools/validation_data/coverage_final.json").read_text())
    assert d["summary"]["verdicts"].get("ERROR", 0) == 0
    assert d["summary"]["verdicts"].get("CRASH", 0) == 0
    assert d["summary"]["verdicts"].get("TIMEOUT", 0) == 0

def test_every_skip_has_a_cited_reason():
    d = json.loads(Path("tools/validation_data/coverage_final.json").read_text())
    for c in d["cases"]:
        if c["verdict"] == "SKIPS":
            assert c["skip_reasons"], c["case_id"]
```

- [ ] **Step 2** — run → FAIL (baseline: 13 CLEAN / 440 SKIPS / 76 ERROR).
- [ ] **Step 3** — run the sweep. **Two cautions the report must carry:**
  1. **A `SKIP` is not a pass.** The baseline's 440 SKIPS include families like
     `DEF_SOLID`, `IOFLAG`, `ANALY` that are *benign* — they do not affect the
     physics — and families like `INTER/TYPE24` that are **blockers**. The
     final report separates them by whether the skip has a `hard_skip`.
  2. **`CLEAN` means the Starter accepted the deck**, not that the Engine ran it
     correctly. G2 is a *coverage* gate; G3 is the *physics* gate. Do not let a
     CLEAN count stand in for correctness.
- [ ] **Step 4** — run → PASS; commit
  `docs(validation): the final corpus sweep`.

---

### Task P17.3: G3 — the parity ledger

**Fortran:** every element family, every registered material law, every
  interface type.
**Files:** Create `tools/validation_data/parity_ledger.json`,
`tests/test_p17_parity_ledger.py`.
**Interfaces:**
- Produces: one row per subject:
  `{"subject": "shell_qbat", "kind": "element", "cases": 3,
    "worst_rel_rms": 0.000006, "tolerance": 0.05, "verdict": "MATCH",
    "verdict_basis": "reference did not diverge"}`.

- [ ] **Step 1** — write the failing test:

```python
def test_every_subject_has_a_measured_verdict_and_a_basis():
    led = json.loads(Path("tools/validation_data/parity_ledger.json").read_text())
    for r in led["rows"]:
        assert r["verdict"] in {"MATCH", "DEVIATION", "UNREACHABLE", "NO-CASE"}
        assert r["verdict_basis"], r["subject"]
        if r["verdict"] == "DEVIATION":
            assert r["worst_rel_rms"] > r["tolerance"], r["subject"]
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — build the ledger. **This phase must not hide behind the
  ledger's own escape hatches.** In particular:
  - `UNREACHABLE` is legitimate **only** where the reference itself diverged —
    the BT-family argument `VALIDATION.md` §7 item 1 makes, which requires the
    reference's own final-window divergence to be **shown with numbers**, per
    case. A blanket `UNREACHABLE` across a family is a rejection.
  - `NO-CASE` is legitimate **only** where no corpus deck exercises the subject,
    and it must name the deck that would. "No deck available" without a search
    is a rejection.
  - A `MATCH` needs a case and a number. "Matches by construction" is not a
    measurement.
- [ ] **Step 4** — run → PASS; commit
  `docs(validation): the parity ledger`.

---

### Task P17.4: G4 — upstream's own acceptance

**Fortran:** `$OR_SRC/qa-tests/`.
**Files:** Modify `tools/run_upstream_verifier.py`, `tools/run_miniqa.py`.
**Interfaces:**
- Produces: `tools/validation_data/upstream_acceptance.json` with one row per
  upstream QA artefact: `{"artefact": "verify_results.py", "runs": true,
  "passed": 11, "failed": 0, "blocker": null}`.

- [ ] **Step 1** — write the failing test:

```python
def test_upstream_qa_accepts_the_port_on_every_artefact():
    d = json.loads(Path("tools/validation_data/upstream_acceptance.json").read_text())
    not_runs = [r["artefact"] for r in d["rows"] if not r["runs"]]
    assert not_runs == [], not_runs
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — run every upstream QA artefact. **This gate is the one that
  cannot be satisfied by self-assessment**, which is why it is a gate. Where an
  artefact still fails, the blocker is fixed here or recorded with a citation.
- [ ] **Step 4** — run → PASS; commit
  `docs(validation): upstream QA acceptance of the port`.

---

### Task P17.5: G5 — the deviation register

**Files:** Modify `tools/validation_data/deviations.json` (Phase 16 Task P16.10),
`docs/PORT_EXTENSIONS.md`.
**Interfaces:**
- Produces: the consolidated register, one row per difference.

- [ ] **Step 1** — write the failing test:

```python
def test_every_deviation_has_a_citation_and_a_test_or_a_reason():
    for d in json.loads(Path("tools/validation_data/deviations.json").read_text()):
        assert d["citation"]
        assert d["test"] or d["reason_no_test"]
        assert d["category"] in {"format", "algorithm", "scope", "environment",
                                 "licence", "performance"}
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — consolidate all 17 phases' deviations into one register,
  sorted by category, and check each is either pinned by a test or has a stated
  reason why it cannot be. **The known prose-only deviations that must appear:**
  the pickle restart (superseded by Phase 10 Task P10.8), CSV/VTK output (kept by
  Phase 12 Task P12.10), the six empty failure directories (Phase 7 Task P7.1),
  the `solid_2d/tria` citation (Phase 2 Task P2.1), the CUDA shell kernels
  (Phase 3 Task P3.11), the mockup tools (Phase 16 Task P16.9).
- [ ] **Step 4** — run → PASS; commit
  `docs(deviations): the consolidated register, pinned or reasoned`.

---

### Task P17.6: G6 — regression and freeze

**Files:** Modify `tools/validation_data/baseline.json`,
`tools/validation_data/frozen_baseline.json`.
**Interfaces:**
- Produces: the final ledger comparison.

- [ ] **Step 1** — write the failing test:

```python
def test_no_new_failures_and_the_frozen_tower_is_unchanged():
    now = tools.regression_ledger.collect()
    delta = compare(load("baseline.json"), now)
    assert delta.new_failures == (), delta.new_failures
    frozen = load("frozen_baseline.json")
    for k, v in frozen.items():
        assert _current(k) == pytest.approx(v), k
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — run. **The 13 pre-existing M614 failures must still be there,
  and must still be the same 13.** If any of them is now fixed, the baseline is
  updated deliberately (that is good news and should be recorded); if any *new*
  one appeared, that is the finding.
- [ ] **Step 4** — run → PASS; commit
  `docs(validation): the final regression and freeze check`.

---

### Task P17.7: Reconcile the disagreements

**Files:** the phases' own files, wherever a disagreement needs a fix.
**Interfaces:**
- Produces: a `plan/RECONCILIATION.md` recording every disagreement between
  G1–G6 and how each was resolved.

- [ ] **Step 1** — write the failing test:

```python
def test_every_disagreement_is_recorded_and_resolved():
    rec = yaml.safe_load(Path("plan/RECONCILIATION.md").read_text()
                         .split("```yaml", 1)[1].split("```", 1)[0])
    for d in rec:
        assert d["status"] in {"resolved", "accepted", "reopened"}
        assert d["detail"]
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — reconcile. **The disagreements are the point of running the
  six gates in parallel.** Expected examples: G2 says a deck is `CLEAN` but G1
  says a routine behind it is `partial`; G3 says an element is `MATCH` but G6
  says a test is newly failing. Each gets a status:
  - `resolved` — one gate's finding was wrong and was corrected;
  - `accepted` — both are right and the disagreement is inherent, with the
    reason;
  - `reopened` — a phase task is reopened, and its id is named.
- [ ] **Step 4** — run → PASS; commit `docs: the gate reconciliation record`.

---

### Task P17.8: The whole-program reviewer

**Files:** none (this is a review, not an implementation).
**Interfaces:**
- Produces: `plan/FINAL_REVIEW.md` — the reviewer's verdict, recorded
  **verbatim**, including anything it rejects.

- [ ] **Step 1** — the review brief:

```
Re-derive, do not read. For each of G1..G6, re-run the measurement
yourself and compare your number to tools/validation_data/*.json.

Then answer, with evidence:

1. Which claims in the 17 phase reports are overstated?
2. Which census rows are marked `ported` that a re-read of the Fortran
   does not support?
3. Which parity `MATCH` verdicts rest on a case that does not exercise
   the claimed behaviour?
4. Is there any place where a deviation was decided in prose and never
   pinned?
5. Does the licence position hold? (Task P0.0's recorded decision, and
   whether the code now matches it.)
6. What would you tell a user who asked "is this a full port of
   OpenRadioss?" — in one paragraph, with no hedging.
```

- [ ] **Step 2** — a fresh reviewer, **space-bunny, never Fast**, with no
  history from the phases.
- [ ] **Step 3** — the review runs. **Its rejections are the deliverable.** A
  review that finds nothing is a sign the brief was too narrow, not that the
  program is complete.
- [ ] **Step 4** — record the verdict verbatim; any rejection reopens the named
  phase task; commit `docs: the whole-program review`.

---

### Task P17.9: Publish the status

**Files:** Modify `docs/STATE.md`, `README.md`, `UPDATES.md`.
**Interfaces:**
- Produces: the authoritative status, measured rather than asserted.

- [ ] **Step 1** — write the failing test:

```python
def test_state_md_numbers_match_the_final_measurements():
    state = Path("docs/STATE.md").read_text()
    led = json.loads(Path("tools/validation_data/parity_ledger.json").read_text())
    cov = json.loads(Path("tools/validation_data/coverage_final.json").read_text())
    assert f"{cov['summary']['verdicts']['CLEAN']} CLEAN" in state
    assert f"{sum(1 for r in led['rows'] if r['verdict'] == 'MATCH')}" in state
```

- [ ] **Step 2** — run → FAIL.
- [ ] **Step 3** — publish. **`docs/STATE.md`'s "What is implemented" section is
  the port's public claim and it must be written from the measurements, not from
  the phase reports' optimism.** If the answer is "a full port of the explicit
  solver, with N deviations recorded and M families declined", that is what it
  says. `README.md`'s "What is implemented" narrative (currently M1–M11) is
  finally replaced with the measured statement.
- [ ] **Step 4** — run → PASS; commit
  `docs(state): the measured status, from the six gates`.

---

## Reviewer checklist

Beyond `00_ORCHESTRATION.md` §9.1 and §9.4:

- **P17.1** — no row is left without one of the four decisions; every
  `partial` has a **named acceptor**.
- **P17.2** — benign and blocking skips are separated; `CLEAN` is not presented
  as correctness.
- **P17.3** — every `UNREACHABLE` shows the reference's own divergence with
  numbers; every `NO-CASE` names the deck that would provide one.
- **P17.4** — upstream's own tools ran; no self-assessment substitute.
- **P17.5** — the six known prose-only deviations are all present.
- **P17.6** — the same 13 pre-existing failures, no more, no fewer.
- **P17.7** — every disagreement has a status; `reopened` entries name a task.
- **P17.8** — the reviewer's verdict is **verbatim**, rejections included.
- **P17.9** — `docs/STATE.md`'s numbers are the measured ones.

## Parallelisation

- **Wave 0** — six independent measurements, six agents. This is the widest
  parallelism in the program.
- **Wave 1** — P17.7 needs all six reports; P17.8 is one reviewer.
- **Wave 2** — P17.9.

## Exit gate — the program's

```bash
python -m pytest -q -m "not slow"
python -m pytest -q
python tools/census.py --render plan/CENSUS.md
python tools/validate_census_closure.py
python tools/validate_vs_fortran.py --mode coverage --out tools/validation_data/coverage_final.json
python tools/validate_vs_fortran.py --mode parity --all --out tools/validation_data/parity_final.json
python tools/regression_ledger.py --check tools/validation_data/baseline.json
python -m pytest -q tests/test_p17_*.py
```

**And then the question the whole program exists to answer**, in
`plan/FINAL_REVIEW.md`, in one paragraph, with no hedging: is `pyradioss` a full
port of OpenRadioss? If the answer is qualified, the qualification is the next
milestone.