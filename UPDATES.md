# Updates

*Reading note: per-file test counts in the entries below are **as-of that
entry**, and every entry heading now carries the date of the commit that
recorded it. Several files have grown since (re-counted 2026-10-03 with
`.venv/bin/python -m pytest -q --collect-only <file>`: `test_p0_paths.py`
60 → 71, `test_p0_compare_t01.py` 20 → 25, `test_p0_harness_portable.py`
27 → 34, `test_p0_parity_t01_path.py` 15 → 22; unchanged:
`test_p0_oracle_selftest.py` 8, `test_p0_manifest.py` 56). **There is no
authoritative *current* count written down anywhere**, by design: the collected
total and the fast tier are produced by the commands in `docs/STATE.md`
§Baseline, and a number written down here is a dated measurement of the tree it
was taken on. `tests/test_p0_record_suite_counts.py` is what keeps it that way
— see §1.8.0 and §1.9.0.*


## 1.15.0 - Task P2.2: the `solidez` orthotropic solid family (mmodul / sortho / szforc3) (2026-10-09)

The first solid kernel whose behaviour is **frame-dependent**, and the first
where the plan's reading of the Fortran had to be corrected against it. New
`pyradioss/elements/solid_orthotropic.py` (1200 lines) with 30 tests in
`tests/test_p2_solid_orthotropic.py`.

- **Exports** `init_group`, `forces`, `dt_claim`, `tangent`, `kgeo`, plus
  `stress_from_strain`, `hourglass_moduli`, `material_frame` and the ported
  routines `gettransv`, `cbatran3v`, `mstiforthv`, `mmod_norm`, `sz_dt1`,
  `scoor_cp2sp`, `sordeft3`, `sroto3`, `szordef3`. Geometry and lumping go
  through `solid_hexa8.init_group` (`srcoor3.F` / `smass3.F` are the same
  one-point brick), so `IORTH == 0` reproduces `solid_hexa8.forces` **bitwise**.
- **The moduli rotation is done on the 4th-order elasticity tensor**, not on the
  Voigt matrix. This is the substantive finding of the task: the strain is
  read in engineering shear (`gamma = 2 eps`) while the stress slot is plain
  (`tau_ab IS sigma_ab`), so the Kelvin operators for strain and stress are
  neither inverses nor transposes of each other and **no ordering of them**
  turns a Voigt shear of `G` into the rotated answer. Every wrong variant still
  returns a symmetric matrix and still maps a permuted frame onto a diagonal,
  so the load-bearing check is the invariance: an isotropic moduli matrix comes
  back unchanged under *any* frame to round-off (~1e-16 relative). Verified
  against the raw tensor rotation of a pure shear, and cross-checked between
  the `forces` path and `tangent`'s `D` (agree to 1.9e-17 relative on an
  anisotropic material).
- **`mmodul.F`'s `ELSE` branch is not the constitutive Lamé stiffness.** It
  reads `C1 = 3E/(1+ν)`, `LAMDA = C1·ν`, `GG = C1(1-2ν)` and writes
  `CC11 = LAMDA + GG = 3E(1-ν)/(1+ν)`, which is `3(1-2ν)` times the physical
  `lam + 2G` — 20% off at an ordinary `ν = 0.3`. `CC`/`CG`/`G33` are the
  **hourglass** moduli and are kept exactly as upstream builds them; the
  constitutive moduli `D` are taken separately from the material law, which is
  what makes the isotropic limit exact.
- **`scoor_cp2sp.F` is a coordinate split, not a transform**: it copies
  `X0(I,1..8)`/`Y0`/`Z0` (declared `DOUBLE PRECISION`) into 24 separate
  `my_real` arrays `X1..X8`, `Y1..Y8`, `Z1..Z8`. Ported as exactly that; the
  strain-rate-to-material-frame transform is `sordeft3.F`.
- **`sz_dt1.F90` yields a length, not a time step** (`DELTAX1`, gated on
  `gfac = G/BULK`), yet `solid_cohesive.py` and `solid_connect.py` cite it for
  an eigenvalue `dt` bound.
- **Declared absent:** the orthotropic hourglass law (`szhour3_or.F`,
  `szsvm_or.F`, `szhour_ctl.F`, `gfhour_or.F`, `szstrainhg.F`) is not
  transcribed; the orthotropic branch runs the isotropic viscous
  Flanagan–Belytschko hourglass, which is what upstream itself calls at
  `szforc3.F:970` for `ISORTH == 0`. The moduli those routines consume *are*
  ported, exposed and tested.
- **No parity case added.** RD-E-2100 exists only as a zip under
  `guide/radioss_models/example/` and contains **no orthotropic material**, and
  registration in the element dispatch is Task P2.11. Recorded here rather than
  worked around.

## 1.14.0 - Task P2.4: the volume-upwind strain filter, transcribed from `upwind_v.F` (2026-10-09)

`pyradioss/elements/solid_upwind.py` carries the `GAM` ladder of
`$OR_SRC/engine/source/elements/solid/solide/upwind_v.F` and its two public
callables, `filter_rate(group, dstra, dt)` and `should_upwind(group, dstra,
dt)`. No existing kernel, no engine path and no `elements/__init__.py` export
was touched; nothing calls this module yet.

- **The ladder, not a single constant.** `UPWM == 2` is the one-line
  Taylor-Galerkin branch (`FAC = CUPWM*HALF*DT1`, `upwind_v.F:74-77`);
  `UPWM == 3` is the SUPG branch (`:100-107`), whose strength is picked per
  element by the local Peclet number `PE = FAC*|tr|` against `EM3 = 1e-3` and
  `THREE = 3` — three rungs (`DELTAX**2`, `GAM**2/V`, `GAM/V`), constants,
  thresholds and branch order transcribed, nothing re-derived. A sweep of
  `vis` alone walks the whole ladder, and the tests require the damping to grow
  monotonically across it.
- **The default is off, and the test says so with `array_equal`.** Upstream's
  default is `ALE%UPWIND%UPWM = 0` (`ale_mod.F:325`), and `upwind_v.F` writes
  `GAM` *only* for levels 2 and 3 — levels 0 and 1 use it as a dimensionless
  sign coefficient on the transportation force (`amomt3.F:391-410`), never as
  a rate. So a group that does not carry the property flag comes out of the
  filter bit-identical: that is the reviewer focus at
  `plan/03_phase2_elements_solid.md:698`, asserted rather than described.
- **Volumetric only.** `upwind_v.F` is the volume variant, so only the leading
  three columns (the trace) are relaxed, towards the element group's mean,
  over one step with mixing fraction `1 - exp(-GAM*dt)`. The deviatoric columns
  are returned untouched, and because the target is a group mean the blend is a
  convex combination — a lone spike is damped and the field's peak magnitude
  can never rise, which is what the Fortran's own algebra gives on its worst
  input and is asserted rather than left to a plot.
- **`should_upwind` is the switching predicate.** Off for every element unless
  the flag selects level 2 or 3, and within a filtered group it selects only
  elements whose `|tr|` exceeds twice the group median: bulk volume transport
  is what the stabilization schemes are *for*, and damping it is why a smooth
  strain field is unchanged whether or not the deck asked for upwinding.

**One generated record moves with it.** Citing `upwind.F`/`upwind_v.F` in the
new module's docstring is what `tools/census.py` reads to map Fortran to
Python, so `tools/validation_data/census.json` now resolves both files to
`pyradioss.elements.solid_upwind` instead of `null`/`missing`. Regenerated
with `python tools/census.py`; the diff is that one field and `plan/CENSUS.md`
is byte identical. Left stale it fails
`tests/test_p0_census.py::test_committed_artifacts_are_byte_reproducible`, so
the record is part of this task rather than a follow-up.

**Evidence.** `tests/test_p2_solid_upwind.py` (21) and
`tests/test_p1_documentation_contract.py` (6) pass, 27 total; the census,
reconciliation, module-size and licence gates pass alongside them (85).
Interpreter: this disposable worktree carries no `.venv` of its own, so the
run used `$PYTHON` as AGENTS.md §Environment resolves it — Python 3.12.3, the
`requirements-lock.txt` §[B] pin. The fast tier on that interpreter is
`14903 passed, 17 skipped, 20 deselected, 25 xfailed`, measured on this branch
with Task P2.1 merged in; nothing here depends on a version-sensitive numerical
path — the filter is a closed-form blend of the strain-rate field.

## 1.13.0 - Task P2.1: `solid_tria3` is a declared port extension, not a port - the fabricated `solid_2d/tria` citation is gone (2026-10-09)

`pyradioss/elements/solid_tria3.py` claimed a Fortran origin under
`engine/source/elements/solid_2d/tria/` - a driver, a derivative routine, a
Jaumann rotation, a length routine and a starter-side mass routine. **Not one of
those files exists.** The claim has been removed and the module relabelled.

- **What upstream actually holds, checked rather than assumed.**
  `engine/source/elements/solid_2d/` contains only `quad/` and `quad4/` - the
  plan's 2026-10-02 note confirmed. There is no engine-side `tria/` directory
  and no `t3*2.F` kernel anywhere in the tree. `starter/source/elements/
  solid_2d/tria/` does exist, and holds exactly two files - `t3grhead.F` and
  `t3grtails.F` - which are input-deck **mesh readers** called from `lectur.F`,
  not kernels; there is no mass routine there either.
- **Decision** (plan Step 3, branch 1): /TRIA3 is a declared element of the
  input format whose kernel ships only in the closed-source tree, so the honest
  citation is **Simcenter Radioss (closed source)** and the module is a
  **declared port extension**. It is no longer described as a port anywhere.
- **New `docs/PORT_EXTENSIONS.md`** - the registry of code with no upstream
  counterpart, carrying the /TRIA3 entry, its upstream-search evidence and the
  per-entry tests that keep it honest.
- **New `tests/test_p2_tria3_provenance.py`** - every Fortran file this port
  cites must exist under `$OR_SRC`. The plan's sketch is generalised from
  `pyradioss/elements/` to the whole `pyradioss/` tree, and its directory depth
  widened: upstream families nest (`shell/coque/`, `solid/solide/`,
  `solid_2d/tria/`), so a two-level pattern cannot see
  `solid_2d/tria/t3forc2.F` at all.
- **56 further misses found and recorded, not fixed.** Generalising the scan
  surfaced that many citations in 23 further modules name absent files. Each
  needs its own decision - a misspelled path (`solid/sdefo3.F` is really
  `solid/solide/sdefo3.F`) is a one-line fix, a genuinely absent kernel is a
  registry entry - so they are enumerated in the test's `KNOWN_MISSES` and
  analysed in `docs/PORT_EXTENSIONS.md` §Known misses. The gate is ratcheted,
  not waived: a *new* fabricated citation still fails, and repairing an entry
  without removing it from the allowlist also fails.
- **No physics changed.** Kinematics, hoop terms, bulk viscosity, Courant step
  and the implicit matrices are byte-identical; this task only relabels
  provenance. No parity claim is made or possible for /TRIA3.

## 1.12.0 - Task P2.3: the solid-to-2-node degeneration paths, and the sort that decides which pair survives (2026-10-09)

`pyradioss/elements/solid_degenerate.py` and
`tests/test_p2_solid_degenerate.py`. No existing kernel, no engine path and no
`elements/__init__.py` export was touched; this task adds a module and nothing
calls it yet.

**The four routines, opened before a line was written** —
`engine/source/elements/solid/solide/ssort_n4.F`, `.../solide/sfor_n2s4.F`,
`.../solide/sfor_ns2s4.F90`, `.../solide/sfor_4n2s4.F90` and
`.../solide4/sfor_n2stria.F`.

- **`ssort_n4` is a node re-ordering, and it is transcribed, not re-derived.**
  It permutes nothing: it writes an integer code — `2` (flattened inside
  MARGE), `3`/`4`/`5`/`6` (one pair merged) or `0` (already a line) — and every
  downstream `SELECT CASE` re-orders the nodes by that code. The ladder is
  branch-for-branch in the Fortran's own order (`X4==X3`, then `X2==X1`, then
  `X4==X1`, then `X3==X2`, each `CYCLE`ing on its first hit), because the order
  *is* the answer: a quad with several merged pairs takes the earliest branch,
  and a rule that picks "the" merged pair on any other criterion gets the same
  *set* of nodes in several cases and the wrong order in all of them. The tests
  pin each branch against the one a re-derived rule would have chosen.
- **The two ways of reaching code `0` stay distinguishable.** Upstream lets both
  stand — `X4==X3` then `X2==X1` (the line runs 1 → 3) and `X4==X1` then
  `X3==X2` (the line runs 1 → 2) — and does not need them apart, because it
  *skips* those quads. A port that has to reduce them needs the pair, so
  `line_pair()` walks the same ladder rather than inferring a pair from `0`.
- **`six_to_two`, `eight_to_two` and `four_to_two` carry the element's mass and
  its centroid.** The weights are the lumped mass projected onto the surviving
  segment, measured from the nodes; the tests assert the first-moment identity
  `sum m_k x_k`, not a tuple compare, and one of them pins a collapse whose nodes
  are *not* evenly split (the top face stops at the midpoint, giving weights
  `[0.625, 0.375]`). A hand-written half-and-half split passes every uniform
  case and fails that one. A solid whose nodes leave the line is refused rather
  than reduced: a flat face is not a line collapse.
- **`four_to_four_striking` keeps upstream's four `ns = n1..n4` passes
  independent**, each with its own `IFC2` and the `ITGSUB` a pass borrows handed
  back afterwards, and carries `sfor_ns2s4`'s `SELECT CASE` tables verbatim. The
  `HJ` slot table is transcribed, *not* derived from the triangle, and the two
  disagree: code 3 puts `LA` on node 1 while its triangle `3-4-2` has `XA` at
  node 2, so `sum(hj_j * X_j)` is not the striking point for every code. That is
  upstream's formulation and a port that "fixes" it stops matching the Fortran.
- **One deliberate deviation.** `sfor_ns2s4` divides by `S2` with no guard; this
  port uses `max(EM20, S2)`, the same `one/max(em20, ...)` idiom the routine
  itself uses two loops earlier to normalize the normal. It changes nothing
  upstream computes and keeps a collinear quad from producing `inf` weights.
- **Not in scope, deliberately:** the forces. `sfor_n2s4`/`sfor_ns2s4` go on to
  build `FN`, book `E_DISTOR` and scatter into `FOR_T`/`FORC_N`; that is the
  distortion-energy path, which reads the element's stiffness history and
  belongs to a different task. This module owns the topology and the weights
  that path consumes.

**The plan's Step-1 sample does not describe the interface, and the interface
block wins.** `plan/03_phase2_elements_solid.md` P2.3 §Interfaces declares
`six_to_two(state, nodes) -> (ids, weights)`, while its Step-1 sample calls a
`hexa_to_two` returning a four-tuple `(ids, w, m, com)`. This entry follows the
Interfaces block: the name is `eight_to_two`, there is no `hexa_to_two` alias,
and mass and centroid are read off the returned weights rather than returned
again — a weight vector that does not conserve them is the bug those two extra
numbers would have hidden.

**No parity row was added, and no deck was invented.** The Phase 0 oracle *is* on
this machine — `~/OpenRadioss_or/bin/starter_linux64_gf` and
`engine_linux64_gf`, reached with `OR_BUILD` and `OR_ROOT` exported as
`tools/oracle/oracle_env.sh` requires, and the engine launches and reports its
version. What is absent is the *deck*: no example under `examples/` collapses a
solid onto a line, and `tests/data/rd_decks` has none either. A parity case
would therefore have meant writing a new deck, which is outside this task's
file set, so `tools/validate_vs_fortran.py parity` was not run for it.

**Evidence and how it was measured.** `tests/test_p2_solid_degenerate.py` (56
tests) and `tests/test_p1_documentation_contract.py` (6) pass, 62 total. The
interpreter is not the repo's recorded Linux `.venv` — that tree has no
`.venv` at all in this worktree, and no candidate interpreter on the box
matches `requirements-lock.txt` §[B] (numpy 2.5.3 / scipy 1.18.1); the run used
the `aid` environment (numpy 2.4.6, scipy 1.18.0, pytest 9.1.1), which matches
§[A] exactly. Nothing here depends on a version-sensitive numerical path — the
degradations are exact comparisons and closed-form projections — but the
figures above are that interpreter's, and the full suite was not run.


## 1.11.0 - Task P1.10: the documentation contract - one measured baseline, a program-status pointer, and a changelog in order (2026-10-08)

The first program-era task that changes no code at all, and the first to find
that two of its own preconditions were false: the changelog existed and was not
in the order the contract requires, and the three "stale" figures were not gone
from `docs/STATE.md` - they were still there, wearing a date and the label
"history, not this box".

- **New `docs/STATE.md` §Program status**, placed after §Licensing so an agent
  meets it while still reading the preamble: the current phase is **Phase 1
  (Foundation)**, file `plan/02_phase1_foundation.md`, and the phase index, the
  dependency graph and the re-ordering levers live in `plan/README.md`. Phase 0
  is complete with its exit gate passing, Phase 1's structural work is in
  progress, and phases 2-17 are unblocked by §1.10.0 but unstarted: no later
  phase is called finished, and no phase is renumbered.
- **§Baseline now names the ledger it follows**:
  `tools/validation_data/baseline.json` (the Task P1.8 regression ledger) is the
  recorded fast-tier result — node-id-exact, with the environment key it belongs
  to — and `python -m tools.regression_ledger --check tools/validation_data/baseline.json`
  is how "no new failures" becomes a measurement. The one measured line §Baseline
  quotes (`14649 passed, 17 skipped, 1 failed, 20 deselected, 26 xfailed`, measured
  2026-10-04 at commit `f5c339f`) is bound to that record on its own line, so it
  reads as the ledger's figure for the ledger's tree and not as today's suite.
- **The three pre-migration fast-tier / full-suite figures are deleted**, not
  re-dated: they were measured on a Windows interpreter and a tree this
  repository no longer has, and the "13 pre-existing M614 failures" they were
  kept to justify **does not reproduce here** - that finding survives, bound to
  the command that measures those six modules, and is the only part of the old
  bullet worth carrying forward.
- **`UPDATES.md` was not newest-first**: §1.4.0 sat above §1.5.0. The blocks are
  swapped and no entry text was touched - the entries stay dated measurements of
  the trees they were recorded on, which is what this file is for.
- **New `tests/test_p1_documentation_contract.py`** holds the contract: the
  changelog descends, §Program status exists and points at `plan/README.md`, the
  phase number and phase file agree with that index's own table (so the check
  survives the next phase starting), no phase after the current one is declared
  complete, §Baseline points at the ledger, and the retired figures stay out of
  `docs/STATE.md`. The phase is parsed from the index, not hardcoded.
- **`README.md`, one sentence.** It pointed at `docs/STATE.md` for "the roadmap";
  the roadmap of record is `plan/README.md`, with `UPDATES.md` as the changelog.

No physics, no module and no test count moved: `plan/02_phase1_foundation.md`
Task P1.10's gate is "full suite byte-identical to the pre-split baseline",
which this task satisfies by touching no code path.

## 1.10.0 - The licence decision is made: this repository is AGPL-3.0-or-later (2026-10-07)

The gate documented since Phase 0 is down. The maintainer recorded the
decision — verbatim: *"as to Licence resolution, set the license whatever you
want, I dont care, just that it would not stop the development"* — and it is
**Option 1: AGPL-3.0-or-later**, recorded as a maintainer decision in
`docs/LICENSING.md` §Decision with his words quoted.

**What changed.**

- `LICENSE`: the MIT text is replaced by the **complete** AGPL-3.0 text,
  preceded by this project's grant notice — copyright © 2026 Minh Quang Pham,
  *"either version 3 of the License, or (at your option) any later version"* —
  and a note that the port contains OpenRadioss-derived code © 2026 Siemens.
  The licence body is reproduced unmodified (sha256 of the verbatim upstream
  text `ad858d53cae05eed9531ecd4c467440803acffe34c69de7b88e03affc5b6a1bd`).
- `pyproject.toml`: `license = "AGPL-3.0-or-later"` (PEP 639 SPDX string) with
  `license-files = ["LICENSE"]`; `requires = ["setuptools>=77"]` is the floor
  that SPDX form needs.
- `README.md` §License: AGPL-3.0-or-later. It also no longer misattributes
  upstream copyright to Altair — upstream is © Siemens.
- `.github/workflows/ci.yml`: one comment no longer calls this an
  "MIT-licensed repo".
- `AGENTS.md`, `docs/STATE.md`, `plan/00_ORCHESTRATION.md` §1.3,
  `plan/README.md`, `plan/01_…`, `plan/02_…`, `plan/13_…`, `plan/17_…`: the
  STOP / GATING / UNRESOLVED notices now say the gate is down and that new
  upstream-derived code is expected and unblocked.

**The enforcement is no longer documentary.** `tests/test_p0_licensing.py::
test_declared_licence_is_consistent` lost its `@pytest.mark.xfail(strict=True)`
— that marker existed only to hold the contradiction open — and now passes for
real: `2 passed`. It reads `pyproject.toml`, `README.md` and `LICENSE` and
asserts they agree. It was strengthened in the same commit: it now also checks
that `LICENSE` carries the *complete* AGPL text (the §5(c) and §13 clauses are
present, so a stub or a one-line reference cannot pass), that the grant is
`-or-later`, and that `README.md`'s License section no longer claims MIT, GPL,
LGPL or BSD for this work. It reads files and parses values; it greps no source
text.

That test caught a real defect on the way through: the first `LICENSE` notice
line-wrapped the canonical *"either version 3 of the License, or (at your
option) any later version"* across a newline. The notice was fixed, not the
check.

**No second contradiction was found.** A full sweep found no per-source-file
copyright or licence header in `pyradioss/**` (69+ modules carry
`# Ported from …` provenance headers, which assert nothing about licensing), no
`COPYING`/`NOTICE`/`setup.py`/`setup.cfg`/`MANIFEST.in`, and one `LICENSE` file
only. The single `MIT` token inside `pyradioss/` is
`pyradioss/failure/sahraei.py:8` — a citation of the Sahraei & Wierzbicki
paper's licence, not a claim about this repository.

**Not done, per scope.** No physics, solver, ported code, or test beyond the
licence-consistency check was touched; no `pyradioss/` module body was
modified. No AGPL per-file notice was added to the ported modules — AGPL
§5(a)-(b) wants prominent modified notices, and adding headers to 69+ files is
its own task, not part of applying the licence.

`docs/LICENSING.md` keeps all four options and the reasoning that rejected
three of them; the decision chose one of them and did not delete the record.

## 1.9.0 - The licensing gate is now in the onboarding document, and two record rules can see what they were blind to (2026-10-04)

*(Superseded 2026-10-07 — see §1.10.0. This entry records the state on
2026-10-04, when no decision existed; the gate it describes is now down.)*

The substantive item of this wave is a single absence: `docs/STATE.md` — the
document every agent is told to read first — never mentioned the licensing gate
at all. An agent read it, saw a green Phase 0, and would start porting
upstream-derived code into a repository whose licence status is unresolved,
which is exactly the failure `plan/00_ORCHESTRATION.md` §1.3 exists to prevent.
Nothing mechanical stopped it: the only enforcement is an `xfail(strict=True)`,
which **passes**.

- **The gate, where it is read.** New `docs/STATE.md` §Licensing gate, the first
  section of the onboarding document, ahead of §Quick start. It states what is
  blocked (no new upstream-derived code beyond what already exists —
  `plan/00_ORCHESTRATION.md` §1.3, lines 58-73), why (a literal port transcribes
  AGPL-covered expression, so AGPL §5(c) makes it a derivative work; `LICENSE:1-3`
  says MIT, `README.md:300-302` says GPL-3.0, `pyproject.toml:10` points at the
  MIT file, upstream is AGPL-3.0-or-later), the four lawful options with
  `docs/LICENSING.md` line references, who can unblock it (**the maintainer** —
  §Decision, lines 136-156, records *no decision* and says in terms that none may
  be recorded by an agent) and exactly **what would unblock you**, what the
  enforcement is (an xfail that passes, so nothing stops the next phase and no
  red test ever will — this section is the stop), and that reconciling the three
  artefacts to make the xfail XPASS is *not* a licence decision either.
  **No decision is taken, recorded or implied.** `pyproject.toml`, `LICENSE` and
  `README.md` are untouched and `tests/test_p0_licensing.py` is unchanged:
  `1 passed, 1 xfailed`.
- **`**22**` → `**24`** in `UPDATES.md` and `plan/01_phase0_oracle_and_licensing.md`
  (measured 2026-10-04, `.venv/bin/python -m pytest -q --collect-only
  tests/test_p0_no_stale_machine_paths.py` → `24 tests collected`). Both records
  now also carry that command.
- **`tests/test_p0_record_suite_counts.py` had the blind spot the rot lived in:**
  `SUITE_COUNT` needs a pytest RESULT word, so `**22** at 2026-10-04` — a
  module's size — was invisible to it, and the date it carried was a perfectly
  good binding. A binding rule cannot catch that: 22 *was* a dated claim, it was
  simply not true. New `unreproducible_module_sizes()` / `test_a_module_size_figure_names_the_command_that_re_measures_it`:
  a figure whose unit is `tests` and whose subject is a named `tests/*.py` module
  must carry the command that re-measures it. **Measured: 4 offenders on the
  pre-fix tree** (`UPDATES.md:268`, `UPDATES.md:923`,
  `plan/01_phase0_oracle_and_licensing.md:660`, `:764`), 0 after the record fixes.
  Two exempt shapes, each stated: a milestone table row (its figures are that
  milestone's own measurement) and a contribution marker (`New`/`added`), the
  same category as the `34 new tests` delta.
- **`_is_floor` no longer launders a measurement into a requirement.** The
  reviewer's case — `on this box numpy <version> is installed at least`, with a
  version the lock does not record — returned `[]` (exempt): the clause break
  cannot separate the version from a trailing hedge.
  New `_MEASURED_CLAUSE` — a clause that says the version *is* installed /
  present / pinned / resolved is a claim whatever follows it — closes it, and the
  three laundering cases are pinned in
  `test_a_requirement_floor_is_not_read_as_a_measurement` while its ten
  legitimate floor prose forms still pass. Both directions proven on the toolchain
  rule too (`on this box cmake <version> is installed at least` is now reported;
  `cmake 3.15 at minimum` is still exempt). **The module's own count did not
  move — 24 before and after** (the cases were added to the existing test, not as
  a new one).
- **Three record inaccuracies corrected:** `docs/STATE.md` §Baseline no longer
  says "the counts in those two rows" (its rows carry commands only — the figures
  are in the sentence); the `§1.8.0` heading now carries the date of the commit
  that recorded it, `4dad39c` (2026-10-04), like the other 18; and
  `docs/LICENSING.md`'s `README.md:290-294` citation now points at
  `README.md:300-302`, which is where the GPL declaration actually is.
- **`tests/test_p0_record_suite_counts.py` had no trailing newline.** Fixed.

## 1.8.0 - Fix wave 3: the last three machine paths, a record-count rule, and a portability proof that read the wrong files (2026-10-04)

Round 4 fixed nineteen record lines and shipped the gate green here and still
red in CI, because three hits sat outside what the portability proof read. This
pass fixes the three hits, states a rule for the exact test counts that aged
three rounds running, and closes the structural hole behind the hits.

- **Three hits, all true here and false on `ubuntu-latest`, all in
  `tools/oracle/`.** Measured with the gate's own scan under a simulated
  foreign `$HOME`: **3 → 0** offenders over the whole scanned scope.
  - `tools/oracle/toolchain_probe.py:68` — the `$OR_SRC` **default was this
    developer's absolute checkout path**, used as a live value. On a box with
    the variable unset the manifest read failed and the probe recorded
    `extlib_url_reachable: false` — a false record caused by a missing default,
    not by a missing release. It now resolves through the project's own
    resolver (`pyradioss.paths.or_src()`, whose last candidate is the checkout
    beside *this* repository), honours `$OR_SRC` verbatim when it is set, and
    falls back to a single component that cannot exist when nothing resolves.
  - `tools/oracle/oracle_selftest.py:19` — the `$OR_SRC = /home/valentin/…`
    docstring line is now the resolver's spelling.
  - `tools/oracle/build_oracle.sh:103` — the conda aside keeps its meaning and
    loses its machine path (`~/anaconda3` is still `$HOME`-relative); the history
    label now sits on the claim itself. **The gate's 2-line label window was not
    widened** — that would have weakened it for every other case.
- **The portability proof was reading the wrong files.** Round 4 added
  `test_no_record_may_name_a_path_out_of_this_machines_home_directory`, which
  simulates a foreign `$HOME` over the **records only**. All three hits above
  live in `tools/`, so the simulation could not see them by construction — that
  is *why* they survived three rounds. The same scan now runs over
  `_scanned_sources()` (records + tooling + packaging), in
  `test_the_foreign_home_scan_covers_the_whole_scanned_scope`, with a planted
  non-vacuity test
  (`test_a_machine_path_planted_in_a_tooling_file_is_caught_by_the_simulation`).
  Both were verified non-vacuous by re-introducing the `toolchain_probe.py`
  default and watching them fail.
- **A record that states an exact count is a record that will be wrong.** The
  rule, stated in `docs/STATE.md` §Baseline and enforced by the new
  `tests/test_p0_record_suite_counts.py`: a suite count is acceptable only if
  the statement says **which measurement moment it describes** — a date, a
  commit sha, or an explicit as-of label, on the line, within two lines either
  side, or in the nearest heading. **Decorative** counts (a baseline block that
  merely restates the suite size) are removed and replaced by the command that
  produces the current figure; **evidence** counts (a before/after showing what
  a change did) are kept and bound. Concretely: §Baseline's result column and
  its "Collected: 14617 … = 14637" bullet are gone, replaced by the three
  environment commands and the `--collect-only` invocation, with the row-3
  measurement kept as a dated figure and rows 1–2 kept with their `13deef2`
  binding and an explicit "not re-run since". The 18 `UPDATES.md` entry
  headings now carry the date of the commit that recorded them (measured with
  `git log -S`), which is what makes a dated log entry's own measurements
  legitimate instead of violations.
- **Shape-based path comparison: measured, then declined.** Widening
  `MACHINE_PATH` to `/opt`, `/srv`, `/usr/local`, `/tmp`, `/data`, `/work`,
  `/root`, `/var` was tried before anything was written. Measured with the
  gate's own existence rule over the whole scanned scope (`_scanned_sources()`):
  **13 hits over 9 distinct paths on commit `4dad39c`** (2026-10-04), and
  **11 over 8 on the tree as committed here** — this restatement stopped
  spelling the offending paths, and quoting them adds hits. Not one hit is a
  stale claim: the scratch-root ones are the documented no-such-mirror fixture
  and the tooling's own scratch files, the `/opt` ones are upstream's own
  documented install prefix, and the rest sit under roots that would each need
  an allow-list entry. *(Corrected 2026-10-04: this entry first split the 13 as
  "7 / 2 / the rest"; the 13 reproduces under the gate's own scan, that split
  does not.)* A rule whose every hit needs an exemption is a rule that cries
  wolf. The structural half *is* fixed: a `/opt`-shaped path that is wrong
  **here** was always caught by the live existence rule; what stayed invisible
  was a path true here and false elsewhere, and closing that needs a record of
  which roots are per-machine, which this repo does not have.

## 1.7.0 - Fix wave 2: a gate that is green here and red in CI, an export that silently dropped two law audits, and a URI exemption that laundered claims (2026-10-04)

Round 3 widened the stale-machine-facts gate to the records and shipped it
green. Three of the four defects below were introduced *by that round*; this
pass fixes the cause in the records and in the tooling, and pins each one with
a test that failed before it. No physics, no oracle behaviour and no licence
decision changes.

- **The gate was red on every machine but this one.** 19 record lines named
  this box's absolute `/home/valentin/...` paths, so a reader with a different
  `$HOME` — including `ubuntu-latest` in `.github/workflows/ci.yml` — was told
  about directories that are not there. Fixed in the records, not in the rule:
  every one now uses the project's own indirection (`$OR_SRC` / `$OR_ROOT` /
  `$OR_BUILD` / `$PYRADIOSS_HM_CFG`, `plan/00_ORCHESTRATION.md` §4.1, or
  `$HOME/<name>`). Locations and replacements are listed in
  `.superpowers/sdd/task-fix-wave2-report.md`. Measured with the gate's own
  scan under a simulated foreign `$HOME`: **19 → 0** record offenders,
  **22 → 3** over the whole scanned scope. The 3 that remain are
  machine-specific absolute paths in three `tools/oracle/` files this pass does
  not own — reported, not edited.
  Pinned by
  `tests/test_p0_no_stale_machine_paths.py::test_no_record_may_name_a_path_out_of_this_machines_home_directory`,
  which re-points `$HOME` and masks the disk answer under this box's home for
  the duration of the same scan.
- **`oracle_env.sh` exported `RAD_CFG_PATH` unconditionally, and the resolver
  had just made a stale export terminal.** Sourcing the script against a mirror
  without `hm_cfg_files` therefore poisoned the one variable that would have
  let `paths.hm_cfg_dir()` fall through to a real tree. The export is now
  guarded by `[ -d "$OR_BUILD/hm_cfg_files" ]`. Measured, cfg-less mirror
  sourced: `test_m539_law34_input_audit.py` + `test_m540_law37_input_audit.py`
  went **`57 passed, 14 skipped`** → **`71 passed`** — the 7 LAW34 and 7 LAW37
  CFG-schema audits run again. Three new tests pin both directions (withheld
  when absent, still exported when present) plus the mechanism, so the two
  files cannot drift apart again.
- **The URI link-target exemption laundered claims.** `_in_uri_target` took
  the first `](` to the left of a match and treated the rest of the line as the
  target, so a line carrying any URI-ish token (`see [x](C:/a/b) and … <the
  pre-migration conda prefix>`) reported `[]` — contradicting its own
  docstring. It is now a SPAN: the target must be closed by `)`/`>` before the
  match ends, and `end` is a required argument so the question can be asked at
  all. Genuine permalinks (the 18 in `docs/BUG_REPORT_2026-09-05.md`) are still
  exempt; nine boundary shapes are pinned from both sides.
- **A requirement floor was read as a measurement.** A floor in prose —
  `requires numpy <X> at minimum`, `numpy <X>+`, `cmake <X> or newer` — was
  compared with the lock and with `<tool> --version`. A floor is satisfied by
  every value the lock could hold, so both rules now skip it — scoped to the
  version's own CLAUSE, so a floor word elsewhere in the sentence cannot exempt
  a real measurement (five such shapes are pinned as still-reported).
- **`docs/OPEN_BUGS.md` item 2 misstated its own coverage.** It claimed "no hit
  in `pyradioss/`; the surviving `consistent_shell_tangent` hits are LAW60's
  own". Measured 2026-10-04: `pyradioss/materials/law14_compso.py` really has
  **0** hits for all three names (the closure is true), `multilayer_shell_update`
  has **0** hits in `pyradioss/` and **0** in `tests/`, but
  `shell_membrane_tangent`/`consistent_shell_tangent` are the generic tangent
  API and are carried by **104** modules under `pyradioss/` (277 lines) and
  **92** files under `tests/`. The register now says exactly that.
- **The exit gate's `$OR_SRC` requirement was a hole.** The checker accepted
  "the script is sourced" as proof that `OR_SRC`/`OR_ROOT`/`OR_BUILD` were all
  set, but `oracle_env.sh` never mentions `$OR_SRC` — and the gate's own last
  command is `git -C "$OR_SRC" status --porcelain`, which with an empty
  `$OR_SRC` inspects the current directory. The exemption is narrowed to the
  two variables the script establishes (`OR_ROOT`, `OR_BUILD`); `$OR_SRC` must
  be exported, and both directions are pinned.
- **Re-measured figures, labelled rather than overwritten.** The four oracle
  modules collect **71** (was 68 on 2026-10-03) and measure `71 passed, 0
  skipped` bare and configured. The historical `OR_BUILD=/tmp/no-such-mirror-xyz
  → 3 failed, 13 passed` (2026-10-03) now reads `3 failed, 16 passed`
  (2026-10-04) — the same three tests fail either way, and the *passed* half of
  a test count ages, so both are now dated in place rather than replaced.
- **Also fixed:** an E302 (two blank lines missing before
  `test_the_harness_imports_and_resolves_oracle_paths`) in the gate module.

## 1.6.1 - Records corrected: the baseline figures, the oracle counts, a gate that skipped what it claimed to verify, and a ruling that was never made (2026-10-04)

*(Superseded 2026-10-07 — see §1.10.0. This entry records 2026-10-04, when the
decision had not been made; it has since been made: AGPL-3.0-or-later.)*

A whole-branch review found four records asserting things that were false or
unverifiable, with nothing keeping them true. All four are corrected here.
**No licence decision is made by this pass** — that stays a maintainer
decision (`plan/00_ORCHESTRATION.md` §1.3).

- **`docs/STATE.md` §Baseline, three fields false.** It claimed
  `14542/14662` collected, and a `14489 passed, 27 skipped, … in 908.76s`
  figure labelled "bare command, `PYRADIOSS_BACKEND=numpy`" — two different
  environments in one sentence, and a number quoted from commit `2e7e598`
  rather than measured. Re-derived on this box: **`14617/14637` collected**,
  and **three** environment-tagged fast-tier figures, all exit 0, all measured
  in one session at commit `13deef2` — bare default backend
  `14575 passed, 16 skipped`; the project's own gate
  (`PYRADIOSS_BACKEND=numpy`, `plan/00_ORCHESTRATION.md` §4.2)
  `14574 passed, 17 skipped`; the Phase 0 exit gate `14576 passed, 15 skipped`.
  Which figure belongs to which environment is stated in the record, because
  they genuinely differ. The oracle-module pair `56 passed, 9 skipped` bare vs
  `65 passed` configured is **withdrawn**: those four modules collect **68**
  now and measure `68 passed, 0 skipped` both ways — `pyradioss/paths.py`
  resolves the install prefix unaided, so the skip surface P0.14 built no
  longer fires on this box.
- **The Phase 0 exit gate now verifies the oracle instead of skipping it.** It
  exported `OR_SRC`/`OR_BUILD`/`OR_ROOT` but never
  `PYRADIOSS_ORACLE_REQUIRED=1`, so on a box without the oracle the gate
  skipped exactly the tests that prove the oracle is real — while
  `tests/test_p0_oracle_build.py:10` and `tests/test_p0_harness_portable.py:64`
  both asserted that the gate sets it. The gate block in
  `plan/01_phase0_oracle_and_licensing.md` now exports it, and its effect is
  measured: with `OR_BUILD=/tmp/no-such-mirror-xyz`, `test_p0_oracle_build.py`
  gave `3 failed, 13 passed` when this entry was written (2026-10-03); the same
  three tests fail today as `3 failed, 16 passed` (re-measured 2026-10-04 — three
  tests have since been added to that module, so the *passed* half of a count
  like this ages; the *failed* half is the claim and it is unchanged).
- **`docs/LICENSING.md` contradicted itself about the licence.** "No decision
  recorded" and a "**Ruling:** … recorded by the Phase 0 controller … pending
  maintainer confirmation" sat eleven lines apart. A recommendation pending
  confirmation is not a decision, so the ruling line is gone; the honest
  recommendation (Option 1) and the blocking behaviour are untouched
  (`tests/test_p0_licensing.py` still `xfail(strict=True)`; measured
  `1 passed, 1 xfailed`, no assertion edited). The document now says one true
  thing: **no maintainer decision is recorded, and no agent may record one.**
- **`docs/OPEN_BUGS.md` contradicted itself the same way.** Its Status Summary
  called items 1–5 FIXED while the section underneath still headed them "These
  items remain". All six closures were re-verified against the tree (removed
  LAW14 shell code/tests, the `_NEW_PORTED_LAWS` keys, `_SOLID_ONLY_LAWS`,
  the `VDOUBLE` rejection, `paths.hm_cfg_dir()`, and the three SPMD refusal
  commits), the section now agrees with the summary, and the single entry
  still open is the **full** `SPMD_EXCH_IDEL` port (`domdec.py:72-75` still
  carries the ghost-`off` limitation). Its `14196 passed / 29 skipped` line is
  labelled as the Windows CI record of 2026-09-24, not a current measurement.

## 1.6.0 - Migration to Linux: every machine fact re-measured (P0.11–P0.16) (2026-10-03)

The repo moved off Windows onto a Linux box and the environment was rebuilt
(venv, oracle mirror + extlib, starter/engine recompiled into `$OR_ROOT`).
This wave corrected the records that still described the old machine,
recovered the coverage the migration silently dropped, and turned two
"machine facts" the suite merely assumed into gated records. Task IDs
P0.11–P0.16 were assigned by this wave: entries 1.5.1 and 1.4.0 below use the
label "P0.11" for the binary-T01 parity route, which is a **different** task.

- **Migration / environment rebuild.** Interpreter CPython 3.12.3 (lock §[B]);
  oracle source `$OR_SRC` (READ-ONLY; dev box
  `$HOME/Projects/OpenRadioss/OpenCourant`), writable mirror `$OR_BUILD`
  (dev box `$HOME/OpenRadioss_build`, harvested extlib **v59**), install prefix
  `$OR_ROOT` (dev box `$HOME/OpenRadioss_or`). `bin/starter_linux64_gf` and
  `bin/engine_linux64_gf` were rebuilt there and are byte-identical (sha256) to
  the build outputs `$OR_BUILD/exec/{starter,engine}`.
- **P0.11** — the lock's machine-verified §[B] re-pinned to this interpreter:
  python 3.14.6→3.12.3, numpy 2.5.2→2.5.3, scipy 1.18.0→1.18.1 (pytest 9.1.1,
  numba 0.68.0, llvmlite 0.50.0 were already true here). The lock was the
  defect; `tests/test_p0_optional_deps.py` was not weakened, and §[A] (the
  maintainer's Windows box) is byte-identical. The mpi4py note kept its NOT
  INSTALLED annotation but its stated *reason* was false here (this box has
  OpenMPI, so `pip install mpi4py` is the cheap route).
- **P0.12** — the oracle records now describe the binaries that exist: both
  sha256 digests re-hashed (starter `b2f6a19f…`→`8b504acc…`, engine
  `99e63c5c…`→`6d58d0b1…`) in `oracle_provenance.json` **and**
  `oracle_smoke.json`, toolchain/host facts rewritten for this box, the gone
  conda prefix recorded as absent, `upstream.mirror_path` corrected to
  `$OR_BUILD`. The golden T01 anchor
  `t01.md5_normalized = e3688899358f35e825cd640f9bd94964` is **unchanged** —
  three consecutive reference runs on the rebuilt binaries reproduce it — and
  `tests/test_p0_oracle_provenance.py` now pins it as a literal; until then
  nothing compared those digests with the binaries on disk.
- **P0.13** — 12 LAW34/LAW37 input-audit tests that skipped on
  `not os.path.isdir(r"C:\OpenRadioss\hm_cfg_files")` now EXECUTE and pass
  (`69 passed` across the two files at the time; **71 today** — `0d5f79e` added
  one CFG-spelling guard per file, re-measured 2026-10-03:
  `.venv/bin/python -m pytest -q tests/test_m539_law34_input_audit.py
  tests/test_m540_law37_input_audit.py` → `71 passed in 1.34s`). The guard is
  `pyradioss.paths.hm_cfg_dir()`, so it follows §4.1's precedence. No audit
  assertion was edited and no skip was re-added.
- **P0.14** — oracle-dependent tests skip **with a reason** when the oracle is
  not configured, so the bare fast-tier command is reproducible. One shared gate
  (`_require_live_oracle`) replaces three private copies that asked only
  whether the two executables resolve; a *stale* export still FAILS instead of
  skipping, and `PYRADIOSS_ORACLE_REQUIRED=1` turns absence into a failure.
  *(Corrected 2026-10-03: the `56 passed, 9 skipped` bare vs `65 passed`
  configured pair this entry carried did not reproduce and is withdrawn. The
  four oracle modules collect **68** now and measure `68 passed, 0 skipped`
  both bare and configured — `pyradioss/paths.py` resolves the install prefix
  unaided, so nothing skips on this box any more. The skip surface this entry
  built is therefore no longer a way to detect a missing oracle; the exit gate
  exports `PYRADIOSS_ORACLE_REQUIRED=1` instead. See `docs/STATE.md`
  §Baseline.)*
- **P0.15** — false machine facts purged from tooling and packaging; all three
  were claims about the pre-migration machine. The pre-migration
  `DT_RPATH=/home/valentin/anaconda/lib` claim in `validate_vs_fortran.py`
  (neither binary carries a DT_RPATH or DT_RUNPATH — `readelf -d`, measured
  `[]` for starter and engine); the pre-migration `numpy 2.5.2` in
  `pyproject.toml`, now a pointer at the lock's `# pin:` lines; and the
  pre-migration `cmake 4.4.3` in `build_oracle.sh`, where `/usr/bin/cmake`
  is 3.28.3.
  `tests/test_p0_no_stale_machine_paths.py` (9 tests at P0.15, **10** at
  2026-10-03, **24** at 2026-10-04 —
  `.venv/bin/python -m pytest -q --collect-only tests/test_p0_no_stale_machine_paths.py`
  re-measures it; this entry first wrote `**22**` there, wrong by two)
  keeps them from rotting and distinguishes a claim about this box from a
  quoted specimen of a tool's output.
- **P0.16** — `toolchain_probe.json` is a **gated record**, not a test side
  effect: `tests/test_p0_toolchain.py` compares every recorded key with a
  fresh probe (quoting the refresh command) instead of calling
  `probe.main([])`, which silently rewrote the claim on every suite run. The
  record no longer names a conda prefix this box does not have — `/usr/bin/
  {gfortran 13.3.0, cmake 3.28.3, make}` — and `gfortran_version` is the
  compiler's own version, not the distro package string.
- **`docs/STATE.md` §Baseline re-measured.** The recorded `13030 passed /
  4 skipped / 13 failed` with 13 "pre-existing M614 failures" **did not
  reproduce** on this box: those six modules measure `74 passed, 1 skipped`.
  The fast tier here was recorded as `14489 passed, 27 skipped, 20 deselected,
  26 xfailed` and **exits 0** — *(superseded 1.6.1: that figure was quoted
  from an older commit and was wrong; the three current environment-tagged
  figures live only in `docs/STATE.md` §Baseline.)* The old numbers are kept,
  marked as the previous machine's.

## 1.5.1 - P0.11 fix round 1: what the verdict actually rests on, and the wrong ROLLING diagnosis withdrawn (2026-10-03)

Reviewer round 1 on the binary-T01 parity route returned SPEC ok / QUALITY
changes-requested and reproduced the headline number exactly. Six findings, all
closed; two of them were my own overstatement.

- **Corrected — `n_significant` counted COMPARED channels, not signal-carrying
  ones.** It was `len([r for r in rows if r["verdict"] != "NODATA"])`, so
  `examples/tensile_bar` claimed 11 channels decided its verdict when **5** do.
  Measured: `EFW IE MASS P1_1 XMOM` carry signal; `CE` is identically zero,
  `YMOM`/`ZMOM` are 1.1e-16 against an axial momentum of 1.96e-4, and
  `HE`/`KE`/`P1_2` peak at 0.02-0.05 % of the dominant energy channel (0.308) —
  all under the 1 %-of-group rule the harness has always applied. The summary now
  separates `n_compared` (11) from `n_signal` (5), names
  `significant_channels` and `noise_channels`, and takes the set from
  `tools.compare_t01.Score.significant` (added in `33a7195`) rather than
  recomputing it. `max_rel_rms` is unchanged at 0.0024648724056964043.
- **Fixed — an unexplained 0.558 beside a MATCH.** Reader-route rows carry no
  `significant` flag, so `YMOM=0.407 ZMOM=0.558` printed unmarked where the CSV
  route prints `~`. Every row now carries `significant`, and the console table
  marks both routes the same way.
- **Fixed — `--tol` was not recorded.** A `--tol 0.001` sweep emitted
  `class DEVIATION`, `max_rel_rms 0.00246`, `tolerance 0.05`: a row that is a
  deviation at a tolerance it satisfies. Rows now carry `tolerance_used`
  (the applied value) next to the historical `tolerance` constant.
- **Fixed — the no-comparable class was not one any recorded sweep carries.**
  `parity_m36..m41.json` has **zero** `FORTRAN-FAIL` rows and 4-9
  `NO-CHANNELS` rows per sweep, which is the recorded class for exactly this
  condition. Both routes now emit `NO-CHANNELS` (an earlier revision emitted a
  bare `FORTRAN-FAIL` here, and my own vocabulary test had whitelisted
  `FORTRAN-FAIL`, which is how it got through). The whitelist is now the
  recorded vocabulary, with the two remaining bare emitters
  (`engine-fail`, `th2csv-fail`) pinned by name.
- **New — the compute backend is recorded.** `auto` resolves per model (this box:
  `numba (auto: 40 elements >= 32)`), and `plan/00_ORCHESTRATION.md` §1 item 6
  pins `PYRADIOSS_BACKEND=numpy` for before/after comparisons. Every row now
  carries `compute_backend` = requested / used / reason /
  `pinned_for_comparison`, harvested from the run's own listing
  (`pyradioss/accel/__init__.py` `_log_backend`, `:260-262`).
- **Withdrawn — the ROLLING diagnosis was wrong.** I wrote "five records per
  step, 9630 = 5 x 1926". Measured properly: the stride is **6**, the block is
  `[4, 92, 36, 64, 264, 36]` bytes and there are **1605** steps (9630 = 6 x
  1605), after a 21-record preamble. And `NSUBS == 1` is **not** evidence of a
  `/TH/SUBSET` request: `starter/source/starter/contrl.F:671-673` counts the
  option and then adds one "for global subset", so `NSUBS >= 1` always. The real
  cause is that `hist2.F` writes one record per `/TH` family that has curves, so
  the per-step block is **variable** while
  `tools/oracle/oracle_selftest.parse_t01` assumes four. A refusal now measures
  the file and says what it is.
- **Blocking item, scheduled not silently deferred.** `read_t01` refuses 2 of
  the 2 in-envelope RD decks that can be run here (`BATOZ/Sf_0.6`,
  `QEPH/Sf_0.8`), so no corpus-wide parity sweep is possible yet. The fix
  belongs to the P0.5 owner of `tools/oracle/oracle_selftest.py`: (a) a
  **variable-stride** walk — derive the per-step block from the header instead
  of fixing it at four records (`:609-622`), and (b) the `hierarchy[5] == N`
  guard on the `1..N` curve-code scan (`:421-441`), which today takes the first
  `1..N`-shaped record in the file rather than the one `hist1.F:308-316` writes
  right after the six-integer hierarchy record (the guard
  `tools/compare_t01.py` already has). Phase 0's exit gate does not require it;
  every phase after this one does.

## 1.5.0 - P0.7: the optional backends are installable, and the box says which are (2026-10-03)

`numba` and `mpi4py` were recorded as missing in `plan/00_ORCHESTRATION.md`
§4.3, which left the M7/M40 accelerated backend and every `-np N` SPMD path
untestable. numba is now installed on the shared Linux interpreter; mpi4py is
deliberately NOT, and the lock says so in a form a test enforces.

- **Installed** `numba==0.68.0` + `llvmlite==0.50.0` into the shared
  pre-migration interpreter (anaconda base, Python 3.14.6). `numpy 2.5.2`,
  `scipy 1.18.0` and `pytest 9.1.1` are **unchanged** — numba 0.68 declares
  `numpy<2.6,>=1.22`, so the accel extra needs no numpy downgrade, and
  nothing pre-existing was upgraded or uninstalled. The 6
  "numba is not installed" failures in `tests/test_m40_auto_backend.py` (5)
  and `tests/test_m7_backends.py::test_numba_backend_provides_kernels` (1)
  are gone: the pair is 30/30.
- **Still optional.** `dependencies = ["numpy>=1.22"]`; `accel`/`mpi` extras
  unchanged (`numba>=0.59`, `mpi4py>=3.1`) — the base install stays
  NumPy-only, as `README.md` promises. Pinned by a test.
- **New** `tests/test_p0_optional_deps.py`: reports a missing optional dep as
  a SKIP (`pytest -rs` is the environment report) and **fails** when
  `PYRADIOSS_ALLOW_MISSING_DEPS=0`; compiles and runs a real `hexa_pre`
  kernel when numba is present; and checks every `# pin:` line in
  `requirements-lock.txt` against the importable version — including
  numpy/scipy/pytest and the interpreter — so a `pip install -U numpy` turns
  the suite red instead of quietly making the lock a lie.
- **`requirements-lock.txt`** now has an explicit `[B]` section for this
  interpreter whose every number is a machine-verified `# pin:` line (section
  `[A]`, the Windows .venv lock AGENTS.md documents, is untouched and stays
  what `pip install -r` resolves). A pinned module that is absent must carry
  a `NOT INSTALLED` note — enforced.
- **The mpi4py gap, documented not papered over.** There is no MPI
  implementation on this box (no `mpicc`/`mpirun`, no `libmpi`, no conda mpi
  package), and the PyPI wheel carries no bundled runtime. Verified against
  the *downloaded, not installed* 4.1.2 wheel: `import mpi4py` **succeeds**
  with no MPI at all; libmpi is resolved on `from mpi4py import MPI`, which
  raises `RuntimeError("cannot load MPI library")` on 4.x (3.x raises
  `ImportError`; not verified here). Installing it anyway would buy nothing.
  Phase 13 needs `apt-get install mpich libopenmpi-dev` + `pip install
  mpi4py`, or `conda install -c conda-forge mpi4py mpich` — or it stays on
  the in-process `ThreadComm` path, which is what `-np N` uses today.
- **Hardened the optional-import seams** (out of map for P0.7, minimal):
  `spmd.comm.Mpi4pyComm.__init__` had a bare `from mpi4py import MPI` with no
  handler at all, and `mpi_world_size()` caught only `ImportError`, so the
  4.x `RuntimeError` (or an `OSError` from an unloadable libmpi) would have
  escaped to the driver. Both now handle the whole "no usable MPI" family;
  `Mpi4pyComm()` raises one actionable error naming the fix instead of an
  opaque import traceback.
- **COST, disclosed.** With numba live, `tests/test_m7_backends.py` +
  `tests/test_m40_auto_backend.py` run in **~250-400 s** instead of the ~2 s
  the numba-absent box reported. The increase is **entirely inside the two
  `@pytest.mark.slow` tests** (JIT compile + full cross-backend runs): the
  non-slow part of the same two files is 2.6 s without numba and 8.4 s with
  it, so the *fast tier* barely moves. Nothing was re-marked — the slow
  marker is already correct, and marking live-coverage numba tests as slow
  would only weaken the gate. Expect the first numba run after a clean
  checkout to pay the JIT cost.
- **Trap worth knowing:** `import pyradioss.accel.jit_kernels` succeeds
  *without* numba (`jit_kernels/__init__.py` degrades `njit` to an identity
  decorator), so it is NOT an availability probe — `HAS_NUMBA` built on it
  never skips. The honest probe is `accel._load_numba_module`, which imports
  numba itself; a test now pins both halves.

## 1.4.0 - P0.11: parity produces evidence without `th_to_csv` (2026-10-03)

Phase 0 exists so later phases can produce differential parity evidence against
the real Fortran solver. They could not: the oracle ran, wrote its binary `T01`,
and the harness stopped at `FORTRAN-FAIL(th2csv-missing)` because upstream's
converter is unobtainable here (`$OR_SRC/tools/th_to_csv/README.md:1-7` points
at the separate `OpenRadioss/Tools` repository, unreachable from this machine —
the same class of blockage as the extlib releases).

- **New** `tools/validate_vs_fortran.py` comparison route: when `th_to_csv` is
  absent, the **Fortran binary T01 is read by `tools.compare_t01.read_t01`**
  (landed in `c679734` precisely so the converter becomes optional) and scored
  against the port's own T01 CSV with the same 5 % tolerance
  `parity_m41.json` records. `run_fortran` now hands the T01 path on instead of
  discarding it, and every row records its `comparison_route`.
  **First real evidence on this box:** `examples/tensile_bar` →
  `MATCH`, `max_rel_rms = 0.00246`, 11 of 30 channels compared and **5** of
  them signal-carrying (EFW, IE, MASS, part IE, XMOM) — corrected by the
  1.5.1 entry below, which fixes the count this entry first stated as 11.
- **Unchanged** the CSV path: it still runs whenever a converter exists,
  including its `final_dev` / `scale` / `significant` row shape. The new route
  is additive and is only reached when there is no Fortran CSV.
- **Honest degradation, both ways.** A missing/unreadable T01 is
  `FORTRAN-FAIL(t01-unreadable)` with the reader's own error quoted and **no**
  `max_rel_rms`; nothing comparable is the pre-existing "no overlapping
  channels" failure, never a `MATCH`. `--tol` can tighten the reader route but
  not loosen it past `parity_m41.json`'s own tolerance, so a new sweep stays
  comparable with every old one.
- **No raw-byte comparison anywhere.** The T01 header carries `ctime()`
  (`hist1.F:210-234` via `timer_c.c:30-40`), so two runs of one deck differ in
  those 24 bytes and nowhere else; a test rewrites the stamp and asserts every
  number is unchanged.
- **New** an `evidence` block on every parity row: the evidence channel
  (`T01 (binary results table)`), whether
  `tools/validation_data/oracle_provenance.json` admits it, why, and which
  **inadmissible** features the deck asks for (`/H3D`,
  `/ALE/STRUCTURED_MESH`, `/CHECKSUM_REPORT` — the H3D family is the engine's,
  `freform.F:2680,2696`). A channel the record does not list is *not* claimed.
- **New** `tests/test_p0_parity_t01_path.py` — 15 tests, no oracle needed (the
  committed golden T01 of the P0.5 reference run is real Fortran output): the
  reader route is taken without the converter, the CSV route is untouched with
  it, unreadable input fails loudly, a known offset gives the expected
  per-channel verdict and number, no verdict can be `MATCH` without significant
  channels, the run stamp cannot move a verdict, and every class the harness can
  emit is a recorded class or a subtag of one.
- **Known limitation surfaced by the end-to-end run** (not this task's file):
  `tools.compare_t01.read_t01` refuses a deck whose T01 carries more than four
  per-step records — e.g. `rd_e/.../BATOZ/Sf_0.6/ROLLING`, measured at six
  records per step — because the shared walk `parse_t01` fixes the stride at
  four. The harness reports it as `FORTRAN-FAIL(t01-unreadable)` with the
  reason, and generalising the stride belongs in
  `tools/oracle/oracle_selftest.py`. (This entry first blamed a five-record
  block and the hierarchy's `NSUBS`; both were wrong — see 1.5.1.)

## 1.3.2 - P0.9 fix round 2: the h3d claim, corrected; PATH on Windows; RPATH read from the ELF (2026-10-03)

Reviewer round 1 on `tools/validate_vs_fortran.py` returned SPEC ok / QUALITY
changes-requested with two Important items, both inside the h3d-hardening
scope, plus two honesty/format minors. Two further POSIX path defaults were
correctly identified as **not** this task's files.

- **Corrected — the round-1 entry overclaimed.** It said "all **four** of
  upstream's h3d dlopen routes are now closed". Two routes were still open:
  a **system-wide** `libh3dwriter.so` (the `ld.so` cache or a default
  directory — found by `h3d_dl.c:660-666` whatever an environment does) and
  the binaries' own **`DT_RPATH`/`DT_RUNPATH`**, which glibc searches
  *before* `LD_LIBRARY_PATH`. Both are now stated in the module docstring,
  `fortran_env`, `run_fortran` and `UPDATES.md`, instead of being implied
  closed.
- **Fixed — `PATH` was never scrubbed, and on Windows it is upstream's fourth
  trial.** The Windows `h3dlib_load_` (`h3d_dl.c:313`) has the same four
  trials and its fourth reads `PATH` explicitly
  (`GetEnvironmentVariable("PATH", …)` `:356`, `SetDllDirectory` `:357`,
  `LoadLibrary` `:358`). `h3d_search_path_vars(platform)` now returns `PATH`
  first on `nt`, and `fortran_env` scrubs it. **Policy: only hazardous
  elements are dropped** — `PATH` also carries the ordinary tools a run
  needs, so the rest is kept untouched and the warning names exactly what
  went. The helper is platform-parameterised rather than skipped, so the
  Windows route is testable from POSIX.
- **Fixed — the fifth route is now checked, and the machine-specific prefix
  is named out loud.** New `elf_search_paths()` reads `DT_RPATH`/`DT_RUNPATH`
  straight out of the ELF (program headers → dynamic section → string table,
  `$ORIGIN` expanded; a few seeks, no subprocess, so it is safe per case) and
  `binary_rpath_hazards()` reports any entry whose directory holds the
  writer. `fortran_env` warns; `run_fortran` **refuses** the run
  (`h3d-writer-in-rpath`) because no environment change can close it.
  Measured on the pre-migration box, where that conda prefix existed: both
  oracle binaries carried `DT_RPATH=/home/valentin/anaconda/lib` (the
  pre-migration record), which the parser reproduced exactly (cross-checked
  against `readelf -d`). The rebuilt binaries now read `[]`: no DT_RPATH and
  no DT_RUNPATH at all (P0.15).
- **Corrected — the `parity_m41.json` citation was false.** The comment
  claimed its rows read `FORTRAN-FAIL`; they do not. The real invariant, now
  stated: *the class vocabulary of a published `parity_m<NN>.json` must not
  shift*. The file's census also carries **4 `NO-CHANNELS` rows this harness
  cannot emit** — recorded as a known gap with its one-line remedy and the
  reason it is not applied here (verdict logic is out of scope for this task;
  a published-evidence class change is a controller decision).
- **Format** one blank line before `find_examples`, matching the file.
- **Scope, stated accurately:** the literal sweep in
  `tests/test_p0_harness_portable.py` is *drive-letter* shaped. Two live
  POSIX defaults exist in `tools/` and are **not** owned by this task
  (`tools/oracle/toolchain_probe.py:41` — `$OR_SRC` default;
  `tools/profile_cycle.py:102-105` — a `TEMP`-composed scratch path). A
  `KNOWN_POSIX_MACHINE_DEFAULTS` ratchet now fails on any *new* one, with
  those two named, so the claim is "no new machine path", not "none".

## 1.3.1 - P0.5 fix round 1: four honesty corrections in the golden record (2026-10-03)

Reviewer round 1 on `tools/oracle/oracle_selftest.py` +
`tests/test_p0_oracle_selftest.py` returned SPEC ✅ / QUALITY changes-requested
and confirmed the central claim (the oracle is bit-reproducible given a fixed
run stamp; the 24-byte window is exactly `hist1.F:210-214` / `timer_c.c:36-39`).
Four honesty-layer defects fixed, none in the solver path:

- **`run_stamp.content` was a typed-in literal and was FALSE of the committed
  artefact** — it claimed `'Sat Oct  3 06:53:35 2026'` while
  `tests/data/oracle_smoke/TENSILET01` carries `'Sat Oct  3 07:03:52 2026'`.
  New `run_stamp_text(blob)` builds it from the bytes read, and
  `test_stored_golden_t01_reproduces_the_stored_maxima` asserts the stored
  string against the committed binary. A record whose thesis is "every field is
  measured, not asserted" had exactly one unmeasured field; it is now measured.
- **The golden engine listing is committed.** `.gitignore:13` ignores `*.out`
  repo-wide, so `tests/data/oracle_smoke/TENSILE_0001.out` is **force-added**
  (`git add -f`) rather than by editing the shared ignore file — one `!` line in
  a file several agents edit in parallel is a merge hazard, and once tracked the
  rule no longer applies to it. New
  `test_golden_engine_listing_is_committed_and_agrees` holds it to the record:
  banner, cycle count, no `** ERROR`, and its `EXECUTION STARTED` second is the
  one inside the committed T01's stamp, so the two artefacts are provably a
  matched pair.
- **The TH-group deferral's reason was self-refuting and the channels were
  nameable.** It claimed upstream ships no title table for them "and then named
  `varn1_title` and declined it". `th_titles.F90:168-189` **is** that table. The
  header walk is now positional over `hist1`'s own block order instead of a
  pattern match (`_header_codes`, from the hierarchy record through parts,
  materials, geometries, subsets and TH groups; it consumes this deck's header
  with **zero** unparsed records, reported as
  `t01.header_unparsed_records`), the codes `hist1.F:585-587` records are
  `[1, 4]`, and `channel_maxima.th_group` now carries
  **X-DISPLACEMENT 0.1880658119916916 / X-VELOCITY 1.0** — the deck's
  `/TH/NODE/1 … DX VX` pull. A subset-bearing deck now raises instead of being
  dropped silently.
- **The non-reproducibility claim had no non-degeneracy assertion.**
  `distinct_md5_raw` was computed from `md5_raw`, so a `--write` whose runs all
  shared one wall-clock second would emit one digest and still assert
  `md5_raw_is_reproducible: false`. `build_record` now refuses such a record and
  `write_record` retries (bounded, `--attempts`, default 12) until two runs
  straddle a second; the test asserts `len(distinct_md5_raw) >= 2`. Proven by
  freezing the clock in a scratch copy: both refuse and no file is written.
- Minors: `t01_md5` renamed `t01_md5_normalized` (it is **not**
  `hashlib.md5(t01 bytes)`, and the old name invited exactly that mistake);
  `environment.anchor_scope` states the digest keeps `VERSIO(2)`/`CPUNAM`
  (`hist1.F:212-217`) and is therefore build- **and architecture**-bound;
  `run_stamp.note` no longer implies the length is located; the window is pinned
  by content on its far side (`' RADIOSS '` at `+24`); the determinism prose
  agrees with `runs`; dead `_oracle_paths()` deleted and the duplicated
  skip/fail block de-duplicated into `_oracle_or_skip`.
- **Cross-file check:** `tools/compare_t01.py` and `tests/test_p0_compare_t01.py`
  (owned by P0.10) do **not** read `t01.md5*` — they read `channel_maxima`,
  `part_curve_codes`, `n_steps`, `t_first/t_last`, `n_records`,
  `header_records` and `run_stamp`. `part_curve_codes` therefore keeps its flat
  shape (`[1, 2]`, the codes concatenated) and the new per-group form is a new
  key `curve_codes_by_group`; their 20 tests stay green and neither file was
  edited.
- **Tests** `tests/test_p0_oracle_selftest.py` 8 (was 7), ~1.3 s, default tier.

## 1.3.0 - P0.10: the binary T01 reader, and one comparable number (2026-10-03)

- **New** `tools/compare_t01.py` — `read_t01(path) -> T01` decodes the Fortran
  engine's binary time-history file (`ITTYP==3` Radioss IEEE: big-endian 4-byte
  record markers, header int32, single-precision channel values) into named
  channels, sample times and an `(n_times, n_channels)` matrix, plus the header
  facts (format code, title width, unit-scaling triples, the located 24-byte
  `ctime` run stamp). Every structural constant is listed in a machine-readable
  `LAYOUT` table with the upstream `file:line` range and a pattern that must
  still be found there, so a constant cannot drift from `$OR_SRC` unnoticed.
  The record framing is **imported** from `tools/oracle/oracle_selftest`, not
  re-implemented: one record walk in the program.
- **New** `score(ref, port) -> Score` — per-channel
  `rel_rms = sqrt(mean((port-ref)^2)) / max(|ref|)` with the `MATCH_RMS = 0.05`
  threshold pinned to `parity_m41.json`'s `tolerance_rel_rms`, so a verdict here
  is comparable with the M36..M41 tables. Channels are matched **by name**; a
  channel only one side has is `NODATA` and never drives `worst`, and neither
  does a channel below 1 % of its group's dominant reference peak (the
  historical harness's own significance rule — without it the transverse
  momentum of a uniaxial test reads as `rel_rms ~ 0.5`).
- **New** `read_port_csv(path) -> T01` — the port's ASCII T01 as the same shape,
  with its column names folded onto the upstream short names (`MOMX -> XMOM`,
  `EW -> EFW`, `P1_IE -> P1_1`) so the two series can be matched by name.
- **New** `tests/test_p0_compare_t01.py` — 20 tests. The reader is validated
  three ways, because upstream's own converter (`th_to_csv`) is unobtainable on
  this box: against the cited upstream sources, against the committed golden
  T01 of the P0.5 reference run (channel set, 100 samples, per-channel maxima,
  all exactly), and against the port's own CSV for the same deck (significant
  channels agree to ≤ 0.2 % rel-RMS; the round-off channels to an absolute
  bound). A fourth test runs the brief's `th_to_csv` cross-check **if** the
  converter ever appears and skips with a reason naming every place tried.
- **New** `Score.significant` — the channels that carry signal and therefore
  decided `worst` (1 % of their group's dominant reference peak). A consumer
  has to be able to report how many channels a verdict rests on:
  `tools/validate_vs_fortran.py` records the *compared* count, which
  over-counts, because the round-off channels are compared too.
- **Fixed** the hierarchy scan can no longer be fooled by a part curve-code
  record: a candidate `1..N` int32 run is accepted only when the record ahead of
  it yields six int32 whose **sixth equals N**, which is what `hist1.F:300-316`
  guarantees (`IWA(6) = NGLOBTH`).
- **Fixed** every citation in `LAYOUT` now names the symbol it documents, and a
  test requires that symbol to exist and to occur at least twice in the module
  (its definition **and** a use). The previous version of that test asserted
  only that the table's own key appeared in the file; each key occurred exactly
  once, so it could not fail.
- **Per-step stride: measured, and the old explanation was wrong.** The shared
  record walk fixes the per-step record count at four, and the count is a
  property of the deck's `/TH` requests. Measured on the real oracle: **4** for
  `examples/tensile_bar` (`[4, 92, 8, 8]` bytes) and **6** for
  `RD-E-1000_Bending/10_Bending/BATOZ/Sf_0.6/ROLLING` (9630 records = 1605
  steps, `[4, 92, 36, 64, 264, 36]`; stride 5 does not match). The extra
  records come from the ordinary `/TH` group block (`hist2.F:608-1403`), **not**
  from the subset block (`hist2.F:478-607`) as first written here — that deck
  asks for no `/TH/SUBSET` at all, and `NSUBS` is never 0 anyway
  (`contrl.F:671-673` adds one for the global subset). Such a file is still
  refused, but the message now quotes the stride measured on it, and the fix
  (derive the stride from the header in the shared walk) is flagged as belonging
  there, before Phase 12 relies on the note.

## 1.2.1 - P0.9: the validation harness runs on any box, and cannot be fooled into writing wrong H3D (2026-10-03)

- **Fixed** `tools/validate_vs_fortran.py` no longer names one machine:
  `OR_ROOT` / `ONEAPI` / the three executable paths are gone, replaced by
  `oracle_paths() -> {starter, engine, th_to_csv, h3d_lib, hm_reader_lib}`,
  resolved **per call** through `pyradioss.paths` — so importing the module
  touches no filesystem and works with nothing configured. An unresolved
  resource is reported as `None` + a warning listing every candidate
  (`paths.missing_resource`), and `parity` exits 2 with the full diagnostic
  rather than writing a results file with nothing compared in it.
- **Fixed** the h3d writer is now kept out of reach along **every route an
  environment can close** — the earlier version of this entry claimed "all
  four", which was false; see the 1.3.2 entry for the correction. On POSIX
  `h3dlib_load_` (`h3d_dl.c:616-923`) tries `$RAD_H3D_PATH` (`:623-632`), the
  working directory (`:634-644`), `$ALTAIR_HOME/hwsolvers/common/bin/$ARCH`
  (`:647-658`) and a bare `dlopen` fed by the loader path (`:660-666`). The
  harness drops the three variables, drops (and loudly names) any
  `LD_LIBRARY_PATH` entry holding the writer, refuses to prepend a reader
  directory that holds it, and refuses a scratch directory that holds it.
  With the closeable routes closed the writer stays unreachable and
  `genh3d.F:728-732` turns `*IERROR = 1` (`:920-922`) into the loud MSGID 274
  refusal.
- **Fixed** `tools/benchmark_rad_db.py` and `tools/compare_t01_tab1.py` no
  longer hardcode a harvested-corpus path or a per-session scratchpad; both
  take an environment variable (`RAD_EXAMPLES_DB`, `TAB1_BASE_DIR`) and fail
  loudly, naming what was tried. `benchmark_rad_db.py` also got the
  `sys.path` seam its CLI needed.
- **Changed** the console table distinguishes a missing `th_to_csv` from a
  failing solver (`FORTRAN-FAIL(th2csv-missing)`), additively: the two
  pre-existing statuses still print the bare `FORTRAN-FAIL` string that
  `tools/validation_data/parity_m41.json` is keyed on.
- **New** `tests/test_p0_harness_portable.py` — 27 tests (from 15): import
  resolves nothing and needs no temp-dir probe, `read_deck` and the workdir
  default are lazy, the four h3d routes stay closed, the loader-path scrub and
  the working-directory guard bite, the invocation asymmetry (`-np 1` starter,
  `-nt 1` engine, never `-np`) is pinned at argv *and* driver level, and no
  `tools/*.py` spells a Windows drive letter outside a named, justified
  exemption. Oracle-dependent tests skip when it is absent and fail under
  `PYRADIOSS_ORACLE_REQUIRED=1`.

## 1.2.0 - P0.5: golden reference run + the oracle's determinism, proved (2026-10-03)

- **New** `tools/oracle/oracle_selftest.py` — runs the oracle built by
  `tools/oracle/build_oracle.sh` on `examples/tensile_bar` (starter `-np 1`,
  engine `-nt 1`, one thread, `RAD_H3D_PATH` unset), parses
  `<run>_0001.out` for the `ENGINE TERMINATION` banner and
  `TOTAL NUMBER OF CYCLES`, md5s `<run>T01` and decodes it. Ships a pure-Python
  `ITTYP==3` T01 reader (big-endian Radioss IEEE records, float32 values) so the
  stored numbers can be re-derived with no oracle installed.
  `--write` regenerates the record and the golden artefacts.
- **New** `tools/validation_data/oracle_smoke.json` — the seed of every later
  parity comparison: deck path + sha256, oracle binary sha256s and the
  `oracle_provenance.json` pointer, the starter/engine argv, `NORMAL
  TERMINATION`, 1420 cycles, T01 size + md5, per-channel maxima for the 23
  global channels (`write_thnms1.F90:228-250`) and the 2 part channels
  (`varpa_title`), wall time, the `RAD_H3D_PATH`-unset fact, and a
  `not_established` block where every null carries a reason (th_to_csv is not
  built on this box; H3D, the starter include-file list, native `.k` reading,
  `/ALE/STRUCTURED_MESH` and `/CHECKSUM_REPORT` over H3D are inadmissible).
- **New** `tests/data/oracle_smoke/{TENSILET01,TENSILE_0001.out}` — the golden
  run's admissible artefacts, committed so the structural tests can verify the
  record without a solver. (Both are tracked; the `.out` is force-added because
  `.gitignore:13` ignores `*.out` — see the 1.3.1 entry.)
- **New** `tests/test_p0_oracle_selftest.py` — 7 tests (8 after the 1.3.1 fix
  round): 5 always-run structural ones (record completeness, deck bytes, maxima
  re-derived from the committed T01, the committed listing, determinism
  evidence) and 3 oracle ones (reproduction, bit
  reproducibility, differing-bytes-inside-the-run-stamp). Oracle absent → skip
  with an actionable reason; `PYRADIOSS_ORACLE_DISABLED=1` /
  `PYRADIOSS_ORACLE_REQUIRED=1` behave exactly as in
  `tests/test_p0_oracle_build.py`. ~1.3 s, so default tier, no `slow` marker.
- **Finding — the raw T01 is not byte-reproducible, and that is upstream, not a
  defect.** `hist1.F:211` writes `ctime()` (`timer_c.c:30-40`) into the T01
  header unconditionally; no keyword and no environment variable suppresses it.
  Measured: runs sharing a wall-clock second have an identical raw md5, a run
  one second away differs in **one** byte (the seconds digit), and with those
  24 bytes zeroed the digest is identical every time
  (`e3688899358f35e825cd640f9bd94964`). The anchor is therefore
  `deterministic_md5()` — the stamp-normalised digest — and the gate asserts
  *both* halves (equal normalised digests **and** every varying raw byte inside
  the 24-byte window), failing rather than skipping when it is violated.
- **Deck choice** `examples/tensile_bar`, not a vendored RD-* deck: it is the
  only small deck runnable with one thread that does not request `/H3D` (the
  oracle refuses H3D by design, so every `/H3D/DT` deck aborts with MSGID 274),
  it costs 0.30 s + 0.08 s per run, and `oracle_provenance.json` already records
  it as the measured evidence for the T01 / A-file / RESTART / energy channels.

## 1.1.3 - P0.8 fix round 3: the remedy must not damage the artifact (2026-10-03)

- **Fix (I1a)** the manifest's corpus-dependent assertions now read
  `tools.validate_vs_fortran.manifest_corpus_root()` — the corpus **the
  manifest describes** (the in-tree vendored `tests/data/rd_decks`) — instead
  of `paths.rd_decks_dir()`, which the environment selects. Consequences:
  **0 skipped tests in every configuration**, including the sanctioned
  `PYRADIOSS_RD_DECKS=<full E: extract>` setup, and the live-vs-described
  relationship stays visible as a **report, never a failure**
  (`test_live_corpus_report_*`, emitted as a `RuntimeWarning` on every run).
  Every record's `sha256` is still re-hashed from the described corpus on every
  run, so the "does the described corpus still hash to the record?" property is
  kept, not traded away. Pinned by `test_described_corpus_is_the_vendored_one`
  (asserted *under* an override) and `test_this_module_has_no_skip_markers`
  (AST check — no conditional skip may be reintroduced).
- **Fix (I1b)** the loud-failure text no longer prescribes a harmful command.
  It states the committed manifest's corpus (path + `sha256:…`), the live
  corpus (path + `sha256:…`) and how many of the live corpus' decks have no
  record, then **branches**: the vendored corpus changed → re-hash *it* with
  `PYRADIOSS_RD_DECKS` unset (`--root tests/data/rd_decks`); a wider extract is
  wanted → write it to an explicit `--out <private path>` and do **not** commit
  it over `tools/validation_data/rd_decks_manifest.json`.
- **Fix (I1c)** `tools/build_rd_decks_manifest.py` now **REFUSES** (exit 2) to
  write the committed manifest from any non-vendored root — `write_refusal()`
  fires for both the bare command under `PYRADIOSS_RD_DECKS` and
  `--root <live dir>`; an explicit `--out` or `--allow-nonportable` is required.
  `--check` is unaffected (verifying a foreign corpus is harmless). Without this
  the prescribed fix was a dead end *and* it turned 1 failure into 6–8.
- **Fix (N8)** the `--check` corpus-fingerprint message derives its record count
  instead of hardcoding "not 75".
- **Fix (N6/N7/N9)** `UPDATES.md`'s "46 (was 30)" corrected to the collected 47
  (the round-3 count is 56, measured); the test module's docstring rewritten —
  it described a vendored-path skip predicate and repeated the harmful remedy;
  the round-2 report's "7 tests" corrected to the measured 8.
- **Tests** `tests/test_p0_manifest.py` 56 (was 47), **0 skipped** in the
  default, byte-identical-override and foreign-extract configurations.
  Mutation-checked on copies: `write_refusal()` neutered, `main()` ignoring the
  refusal, the hardcoded "not 75", the report prescribing the live-corpus
  re-hash, and `_described_corpus()` following `PYRADIOSS_RD_DECKS` — all five
  killed.
- **Process note** writing the refusal tests *before* the refusal existed made
  one of them clobber the committed manifest with a synthetic corpus (restored
  from git immediately). Those tests now snapshot and restore the committed
  file through a `committed_manifest` fixture, so a regression of the guard
  cannot leave the artifact damaged.

## 1.1.2 - P0.8 fix round 2: the manifest says only what it can support (2026-10-03)

- **Docs (M5)** `tools/build_rd_decks_manifest.py`'s header claim "the manifest
  closes that gap" was **false** and contradicted its own `PROVENANCE_CAVEAT`:
  the manifest binds *future* runs to bytes; the M41 verdicts stay bound to a
  scratchpad extract that no longer exists (hence
  `parity_run_deck_bytes_verified: false` everywhere). Rewritten to say
  exactly that. `UPDATES.md`'s I2 bullet no longer claims "all six" in-envelope
  decks skip the five control families (5 do; the 6th skips `/TH/BRIC` alone,
  and none reports `ERROR`), and "21-field" became 24.
- **Fix (N1)** the loader's *undocumented extra field* rejection is now covered:
  `test_loader_rejects_an_undocumented_record_field` (+ the missing-field and
  unknown-schema mirrors). Neutering the branch used to leave the suite green.
- **Fix (N2)** `--check` now compares the header prose — `hash_algorithm`,
  `hashed_file_rule`, `envelope_rule`, `bytes_verified_rule`, `joins`, `notes`
  (`PROSE_KEYS`), including the envelope-qualification note generated from the
  records. Deleting or rewording a rule a reader quotes is drift; it used to
  pass silently.
- **Fix (N3)** `resolve_manifest_record(rec)` **requires**
  `corpus_fingerprint`: a hand-built record without it used to bypass the
  corpus guard in silence — the one input shape where the I1 protection
  switched off unnoticed. It is now an explicit `ValueError`, with an explicit
  `require_fingerprint=False` opt-out that waives the corpus *binding* only
  (the file is still re-hashed).
- **Fix (N4)** a one-byte corpus change no longer prints 75 `deck changed`
  lines: the corpus-level fact is stated once ("corpus fingerprint differs:
  X -> Y … every record's `corpus_fingerprint` moves with it, which is this one
  fact, not 75") and per-deck lines list only genuine per-deck fields.
- **Behaviour change (R1)** the whole-corpus tests now skip only when
  `rd_decks_dir()` is a corpus the manifest does **not** describe. A
  **byte-identical** copy elsewhere (`PYRADIOSS_RD_DECKS`) is recognised by the
  fingerprint, so the tests RUN and pass there. A *different* extract skips
  them but fails the never-skipped
  `test_live_corpus_is_the_corpus_the_manifest_describes`, which names both
  fingerprints and the command that re-hashes — eight silent skips are no
  longer possible.
- **Tests** `tests/test_p0_manifest.py` 47 (was 30; the 46 I first wrote
  here was not the collected count — run the number, do not assert it).
  copies in `/tmp/opencode`: an extra record key, a deleted
  `bytes_verified_rule`/`envelope_rule`/`notes`, and a record built without
  `corpus_fingerprint` are each rejected.

## 1.1.1 - P0.8 fix round 1: records bound to their corpus, envelope qualified (2026-10-03)

- **Fix (I1)** a manifest record is now bound to the corpus it was hashed from:
  every record carries `corpus_fingerprint`, the header is reachable through the
  new `load_manifest_doc()`, and `resolve_manifest_record(rec, root=None)`
  resolves against the manifest's **own** declared corpus (never
  `rd_decks_dir()`) and raises `ValueError` when the corpus fingerprint differs
  — so a `PYRADIOSS_RD_DECKS` override can no longer join a verdict to a
  different file at the same relative path. New `corpus_fingerprint(root)`:
  `sha256:<hex>` over the sorted `<relpath>\t<sha256>` lines of every
  `*_0000.rad`, path-independent by construction.
- **Fix (I2)** records now carry `skipped_families` (with each family's class),
  `coverage_hard_skips`, `coverage_blockers` and `coverage_degrade_warnings`
  from `coverage_results_m41.json`. This matters because `coverage_verdict`
  counts only NON-control skips: a `SKIPS(2)` row hides five control-class
  families. Five of the six in-envelope decks (BATOZ Sf_0.6/0.8/0.9, QEPH
  Sf_0.8/0.9) report `SKIPS(2)` while skipping **seven** families —
  `/ANALY`, `/DEF_SHELL`, `/DEF_SOLID`, `/IOFLAG`, `/SPMD` (class `control`)
  plus `/TH/RBODY`, `/TH/SHEL` (soft); the sixth
  (`RD-V-0700/…/HEXA_ELEM_SOLID_18`) reports `SKIPS(1)` and skips `/TH/BRIC`
  alone. No in-envelope deck reports `ERROR`. `in_envelope_reason` now names
  each deck's skips, so the envelope claim cannot be read unqualified.
- **Fix (I3)** new per-record `parity_run_deck_bytes_verified` /
  `coverage_run_deck_bytes_verified`, **false on all 75**: the M41 sweep ran
  from a session scratchpad that no longer exists
  (`.agents/skills/validation-compare/SKILL.md:56-59`) and the official parity
  rows carry no deck path, so each record pairs the hash of the re-vendored
  copy with a verdict measured on the scratchpad copy. Both flags are
  **derived** (`deck_bytes_match()`: does the evidence row name an existing
  file whose bytes hash to ours?) so a future in-place run flips them itself;
  the manifest notes and `bytes_verified_rule` say so.
- **Fix (M1)** `build_rd_decks_manifest.py --check` now compares what the corpus
  **contains** (fingerprint, counts, records) and ignores this invocation's
  provenance (`generated`, `corpus_root.source` / `.vendored` /
  `.resolved_at_generation`), so it is portable: a byte-identical corpus at any
  path verifies clean, while a changed deck still fails.
- **Fix (M2)** `MANIFEST_FIELDS` is now the complete record schema —
  24 fields (was 15: it omitted `parity_provenance` and
  `inventory_classification_strict`, and the loader only complained about
  *missing* fields, never about *extra* ones).
- **Fix (M3)** the manifest tests' skip condition is derived from
  `paths.rd_decks_dir()` vs this checkout's `tests/data/rd_decks`, not from the
  manifest header, plus a never-skipped `test_manifest_file_exists()` — a
  missing manifest now fails instead of skipping five tests with a misleading
  reason.
- **Fix (M4)** `counts` declares `measured` (24) and `unmeasured` (51)
  explicitly, next to `in_envelope_with_control_skips` and the two
  `*_deck_bytes_verified` totals, instead of leaving them to subtraction.
- **Tests** `tests/test_p0_manifest.py` 30 (was 15). Mutation-checked against
  copies in `/tmp/opencode`: emptying an in-envelope record's
  `skipped_families` (killed), claiming
  `parity_run_deck_bytes_verified: true` (killed), making
  `resolve_manifest_record` follow `PYRADIOSS_RD_DECKS` (killed) and also
  dropping the fingerprint guard (killed, 2 tests), restoring the wholesale
  `corpus_root` comparison in `--check` (killed).

## 1.1.0 - Hashed corpus manifest (P0.8) (2026-10-03)

- **New** `tools/validation_data/rd_decks_manifest.json` — one record per
  starter deck (`*_0000.rad`) under `paths.rd_decks_dir()`: `deck`,
  `hashed_file`, `sha256`, `size_bytes`, `case_id`/`category`/`package`
  (carried over from `inventory.json`, `null` when it never catalogued the
  deck — ids are never invented), the `parity_m41.json` /
  `coverage_results_m41.json` verdicts, and `in_envelope` **with**
  `in_envelope_source` + `in_envelope_reason`.
- **`in_envelope` is measured, never predicted**: `true` only where
  `parity_m41.json` carries a `MATCH` verdict for that exact `case_id` (the
  port reproduced the oracle). `inventory.json`'s static `IN_ENVELOPE`
  classification (a *reading* prediction) is recorded as context and never
  flips the flag. Every `false` carries its reason.
- **New** `tools/validate_vs_fortran.load_manifest()` → `list[dict]`; lazy and
  read-only (reads the JSON inside the function, opens no deck), so it works
  with no environment exported and no corpus mounted. Raises `ValueError` on an
  unknown schema or a record missing a documented field.
- **New** `tools/build_rd_decks_manifest.py` — regenerates the manifest from
  the corpus, and `--check` verifies the committed file against the decks on
  disk without writing.
- **Tests** `tests/test_p0_manifest.py` (15): one record per corpus starter
  deck, **every** recorded `sha256` re-hashed from disk on every run (75 decks
  / 8.8 MB, ~10 ms — the whole corpus, not a sample), unique ids that match
  `inventory.json`, the envelope flag only ever backed by a measured `MATCH`,
  non-empty joins with `parity_m41.json` and `coverage_results_m41.json`, the
  loader working with nothing exported, and a synthetic uncatalogued deck
  coming out `case_id: null` / `in_envelope: false`.

## 1.0.2 - P0.6 fix round 1: CI cfg spelling, candidate order, pinned precedence (2026-10-03)

- **Fix** `PYRADIOSS_HM_CFG` now accepts **both** spellings — the documented
  tree root (`…/hm_cfg_files`, §4.1 and upstream's `RAD_CFG_PATH`) and the
  schema directory itself (`…/hm_cfg_files/config/CFG`, what
  `.github/workflows/ci.yml` exported). `mat_reader._find_cfg_root` decides by
  inspecting the filesystem (`<cfg>/config/CFG` exists?), never by string
  shape; a directory named `CFG` with no `radioss<version>` schemas is
  rejected loudly instead of accepted.
- **Out-of-map file edit** `.github/workflows/ci.yml` re-points
  `PYRADIOSS_HM_CFG` at the tree root (both jobs), so the workflow exports
  what §4.1 says the variable means. The old spelling still works in code.
- **Fix** the brief-only candidate `$OR_ROOT/OpenCourant/hm_cfg_files` no
  longer sits *before* the §4.1 rule-3 Windows candidate — every
  non-contract candidate is now strictly after rules 1–3.
- **Fix** `pyradioss.paths.is_cfg_tree` / `is_cfg_schema_dir`: a cfg
  candidate must carry the incremental `radioss<version>` subdirectories, so
  an empty or partial sparse checkout fails loudly instead of resolving.
- **Fix** `mat_reader.catalogue()` is keyed on the freshly resolved cfg root
  and rebuilds when it moves, so `paths.reload()` reaches the singleton that
  several test modules freeze at collection time.
- **Diagnostics** a *set but missing* variable now emits a `RuntimeWarning`
  (a stale `PYRADIOSS_RD_DECKS` could silently downgrade a validation run);
  an unresolvable `OR_ROOT` nests its own candidate list instead of printing
  a placeholder, and "unset" is no longer claimed for a variable that was set
  and merely absent.
- **Tests** `tests/test_p0_paths.py` 60 tests at P0.6 (was 35; **71** on
  2026-10-03 — re-count with `.venv/bin/python -m pytest -q --collect-only
  tests/test_p0_paths.py`): every neighbouring
  precedence pair, loud failures for `or_build()`/`or_engine()`, both cfg
  spellings, the empty-`CFG` rejection, and the catalogue-follows-`reload`
  invariant. 7/7 mutants killed by a harness in `/tmp/opencode`
  (extras-first, rule2-before-rule1, Windows-candidate-deleted, no-op
  `reload()`, cfg-root-appends-`config/CFG`, frozen catalogue root,
  any-directory-is-a-cfg-tree).

## 1.0.1 - Single resource resolver (P0.6), LAW4 cfg bug closed (2026-10-03)

- **New** `pyradioss/paths.py` — the one place that resolves every external
  path (`or_src`, `or_root`, `or_build`, `or_starter`, `or_engine`,
  `hm_cfg_dir`, `rd_decks_dir`, `missing_resource`, `reload`), implementing
  the order of `plan/00_ORCHESTRATION.md` §4.1: env var (if set **and**
  existing) → sibling-of-build → Windows compat → **fail loudly** with every
  candidate listed. Upstream authority: `$OR_SRC/INSTALL.md:34-42` (the Linux
  env block: `OPENRADIOSS_PATH` / `RAD_CFG_PATH` / `RAD_H3D_PATH` /
  `LD_LIBRARY_PATH`) and `:110` (the `starter_linux64_gf` name). No import-time
  filesystem access; `import pyradioss.paths` cannot fail.
- **Convention** `missing_resource(name, tried)` *returns* a
  `FileNotFoundError` instance; resolvers `raise` it. Keeps one diagnostic
  usable by callers that must not abort (the `/MAT` reader logs it).
- **Fix (OPEN_BUGS item 6)** `pyradioss/input/mat_reader.py` no longer probes
  an import-time tuple of `os.environ` plus two hardcoded roots; the cfg tree
  comes from `paths.hm_cfg_dir()`. `tests/test_m535_law04.py` now passes on
  Linux (76 passed) — before: `no cfg schema found`, `E must be > 0`. The
  heuristic degradation is kept (a deck must still parse) but its warning now
  carries the full list of locations searched.
- **New** `tests/test_p0_paths.py` (35 tests) — every resolver per tier, the
  stale-env skip, the loud failure message, `reload()`, the vendored deck
  corpus resolved from any CWD, and `or_build()` never returning the
  read-only `$OR_SRC`.
- 22 cfg-dependent test modules that silently skipped on Linux now execute
  (previously `mat_reader.catalogue().schema("FABRI") is None`): 535 passed.

## 1.0.0 - Full-port program plan (2026-10-02)

The first structured plan for taking `pyradioss` from its current state to a
literal line-for-line port of OpenRadioss. 20 files in `plan/`, 200 tasks,
18 phases, sized against a measured upstream census.

**Scope decision (maintainer, 2026-10-02):** *literal line-for-line port* —
every element family, material law, failure model, EOS, contact type, the
binary output formats (T01, A-files, H3D, PARITH, restart), the full MPI/SPMD
exchange layer, and the `reader/` SDI layer. Not "parity on what the corpus
exercises".

**Findings that shaped the plan (measured 2026-10-02):**

- Breadth is nearly done, depth is not: 571 material registry keys but ~59
  numeric laws; 26 contact modules; 752 Python lines of output against 180,559
  Fortran lines; 2,582 of MPI against 112,000.
- **`pyradioss/elements/solid_tria3.py` cites `engine/source/elements/solid_2d/tria/` and four `.F` files that do not exist upstream** — the 2-D solid triangle is a port extension with a fabricated citation. Phase 2 Task P2.1.
- **Six `/FAIL` directories upstream are empty** (`changchang`, `composite`, `hashin`, `lemaitre`, `puck`, `spalling`) while the port has six modules for them. Phase 7 Task P7.1.
- **`failwave` is a failure-wave *propagation* model, not a damage-combination rule** — `MAXLEV_STACK`, `FWAVE_NOD_STACK`, `ERROR IN FAILWAVE PROPAGATION`. Phase 7 Task P7.8.
- **`/INTER/TYPE19` is real** but read by the *generic* reader `interf1/definter.F:190`, not by an `int19` directory — the directory count is evidence, not proof. Phase 8 Task P8.1.
- `mulawc.F90` names **51 laws with an upstream shell kernel**; the port's
  shell-capable set is a small fraction of it. Phase 6 Task P6.0.
- `starter/source/interfaces/inter3d1/` is **59,973 LOC** — the largest
  directory in the solver — and the port has no dedicated module for it.
  Phase 8 Task P8.7.

**Blocking issue found while planning — the licence contradiction:**

- `$OR_SRC/LICENSE.md` is **AGPL-3.0-or-later** (Siemens 2026 headers).
- `pyradioss/LICENSE` is **MIT** (© 2026 Minh Quang Pham).
- `README.md` claims **GPL-3.0**.

A literal port of AGPL code cannot ship under MIT. This is **Task P0.0 and it
blocks Phase 1 onward**; it enumerates four lawful resolutions and does not
choose one — that is a maintainer/legal decision.

**Environment change:** the working machine is now Linux with the upstream
source at `$OR_SRC` (dev box `$HOME/Projects/OpenRadioss/OpenCourant`),
**unbuilt** (no extlib, no `exec/`, no `cmake_linux64_gf.txt` compiler flags).
`AGENTS.md` and `tools/validate_vs_fortran.py` are Windows-specific. Phase 0
builds the oracle; Phase 1 Task P1.0 rewrites `AGENTS.md` platform-neutral.

**Parallel-execution design:** wave graph with one git worktree per wave, a
declared single-owner map for the 16 contended files, one reviewer per task,
one per phase, and a whole-program reviewer at Phase 17. All subagents and
reviewers run on **space-bunny**; no Fast variant, ever.

**Also frozen:** `pyradioss/implicit/` (~25k LOC the upstream Fortran does not
have — modal, complex-modal, PSD, spectral fatigue, NORTA, evolutionary
fatigue) is kept but quarantined behind extras, pinned by
`tests/test_p1_frozen_implicit.py`.

**Nothing in `pyradioss/` was changed.** The plan is the only artefact.
