# Open bugs — found, not yet fixed

*Recorded 2026-09-24 at the end of the M615 (SPMD) bug-fix pass, PR #51.
Update (2026-09-24): Items fixed in PR #47 / branch `feat/vtk-full-tensors`.*

## Status Summary

- **SPMD-1 (`/GJOINT` under `-np > 1`)**: **FIXED** (commit `ecff9bc`). Refused in `check_spmd_support` citing `lag_mult.F LAG_MULTP L683`.
- **SPMD-2 (ghost `off` deletion / interface divergence)**: **FIXED (interim refusal)** (commit `4219bdb`). Refused `/INTER/TYPE2` with failure and penalty contacts with `idel >= 1` / `idel10 >= 1` citing `chkstfn3.F SPMD_EXCH_IDEL`. Full port of `SPMD_EXCH_IDEL` tracked for future SPMD enhancement.
- **SPMD-3 (`/KJOINT` penalty replica double-counting)**: **FIXED** (commit `4c4e80b`). Refused `model.kjoints` in `check_spmd_support` citing `ruser33.F` / `spmd_exch_a.F`.
- **Item 1 (Dead LAW14 shell tests)**: **FIXED** (commit `e1dbee6`). Removed dead shell tests from `tests/test_law14_compso.py`.
- **Item 2 (Dead LAW14 shell code)**: **FIXED** (commit `e1dbee6`). Removed `multilayer_shell_update`, `shell_membrane_tangent`, `consistent_shell_tangent` from `pyradioss/materials/law14_compso.py` citing `mulawc.F90:1125-1307` and `hm_read_mat14.F:151-158`.
- **Item 3 (Duplicate law registration)**: **FIXED** (commit `1f680b5`). Removed duplicate LAW24, LAW37, LAW90 from `_NEW_PORTED_LAWS` in `pyradioss/materials/__init__.py`.
- **Item 4 (Solid-only audit)**: **FIXED** (commit `1f680b5`). Audited against `mulawc.F90` and added 11, 13, 51, 54, 151 to `_SOLID_ONLY_LAWS` in `tests/test_mat_all_135_census.py`.
- **Item 5 (Duplicate `/PROP` IDs not rejected at parse time)**: **FIXED** (commit `c6c3b3c`). Added duplicate ID rejection in `pyradioss/input/starter_keywords.py:parse_starter_deck` citing `hm_read_properties.F:798 VDOUBLE`.
- **Item 6 (LAW4 cfg lookup in Linux dev container)**: **FIXED** (Task P0.6,
  `pyradioss/paths.py`). The cfg tree is now resolved by the single resolver
  `pyradioss.paths.hm_cfg_dir()` instead of an import-time tuple of
  `os.environ` + two hardcoded roots; the `/MAT` reader keeps its heuristic
  degradation but now logs every location that was searched.

---

## SPMD-1 — `/GJOINT` under `-np > 1` gives wrong gear torques (severity: high)

- **Where:** `pyradioss/engine/engine.py:1103` calls `gj.transfer_forces(...)`
  *before* the frontier sum `spmd.exch_forces(...)` at `engine.py:1133`. The
  joint is replicated on every domain holding one of its nodes
  (`spmd/domdec.py:663-666`), and `check_spmd_support`
  (`spmd/domdec.py:468`) does not refuse it.
- **Why it is wrong:** `GJoint.transfer_forces` is not linear in `mint`. For
  GEAR (`engine/gjoint.py:579`, also `:616`) it picks one of three branches
  from which of the partial torques T1 and T2 is non-zero. Suppose gear nodes
  idx1 and idx2 are native on different domains. Domain 0 sees only T2 and
  applies reaction branch 1; domain 1 sees only T1 and applies branch 2.
  After the sum, both gears carry a reaction, whereas the serial run (T1 and
  T2 both non-zero) takes the carrier-only branch 3. The result is wrong
  torques and energy injection. CV (joint type 4) has the same problem; RACK
  does too near zero torque.
- **Fortran:** `engine/source/tools/lagmul/lag_mult.F`, `LAG_MULTP` (the SPMD
  variant), around lines 683-687: `IF(ISPMD==0 .AND. NGJOINT>0) CALL
  ANCMSG(MSGID=113,...,C1='JOINT'); CALL ARRET(2)`. OpenRadioss aborts a
  decomposed run that contains GJOINTs.
- **Fix (Fortran-faithful, small):** add `("gjoints", "/GJOINT")` to the
  refusal tuple in `check_spmd_support`, next to the other
  `_nonempty(getattr(model, attr, None))` checks, and cite `LAG_MULTP`.
  Add a `test_spmd_domdec.py` case asserting that the Starter refuses
  `-np 2` with a `/GJOINT`.

## SPMD-2 — a ghost element's `off` is never refreshed, so deletion splits replicated interfaces (severity: high)

This merges two confirmed findings (anchored at `contact/inter_type2.py:325`
and `spmd/domdec.py:468`).

- **Where:** the `domdec.py` docstring (around line 72) already admits it:
  *"Known limitation: a ghost element's `off` flag is never updated during
  the run."* Nothing in `pyradioss/engine/` or `spmd/exchange.py` writes to
  `model.spmd_ghost`. `contact/tracking.py` resolves `ghost:<group>`
  provenance to those frozen copies.
- **Failure case A (TYPE2):** a `/INTER/TYPE2` tie whose main-segment parent
  shell carries `/FAIL`. The tie is replicated (`domdec.py` closure). The
  shell is native on rank A and a ghost on rank B. When it fails, the
  `cycle % 8` poll at `inter_type2.py:325` runs `_release` on rank A only:
  the tie goes inactive and the mass is removed from `mass_eff`. Rank B still
  reads `off = 1` and keeps the tie. From then on the two holders redistribute
  the frontier forces differently. `mass_eff` differs between holders, and the
  replicated velocities diverge permanently (velocities are never exchanged;
  `exchange.py` relies on every holder computing the same ones). There is no
  warning, and the gathered output and energy ledger are wrong.
  `tracked_node_mask` has the same problem, because it counts stale ghost
  elements as alive.
- **Failure case B (penalty contact with deletion):** on the owning rank of a
  TYPE7/10/11/24 interface with `Idel >= 1` (for TYPE10 the field is
  `idel10`), segments whose parent element is on another domain never die, so
  the contact keeps pushing on crack faces. Secondary nodes whose remote
  elements have all died also stay tracked.
- **Fortran:** `engine/source/interfaces/interf/chkstfn3.F`:
  - `TAGOFF3N` (around line 572) exchanges node deletion tags through
    `SPMD_EXCH_IDEL` (lines 1129-1136; `engine/source/mpi/interfaces/spmd_exch_idel.F`).
  - Lines 1807-1808 call `SPMD_INIT_IDEL` / `SPMD_EXCHMSR_IDEL`
    (`spmd_exchmsr_idel.F`).
  - `CHK2MSR3N` (TYPE2 main-segment check, around line 3158) queries other
    processors at lines 3658-3668.

  The upstream code gives every domain the same deletion state.
- **Fix, interim (refusal):** in `check_spmd_support`, if any element group
  has deletion enabled (`state["chk_fail"]`, the test that
  `tracking.any_deletable` uses), refuse `/INTER/TYPE2` and every penalty
  interface with `idel >= 1` / `idel10 >= 1`. The message should cite
  `SPMD_EXCH_IDEL`.
- **Fix, full (port `SPMD_EXCH_IDEL`):**
  1. Domdec keeps a global-id array per ghost group.
  2. Once per cycle, on every rank, after the element loop and before the
     contact / TYPE2 `transfer_forces`, all-gather the `elem_glob` rows whose
     `off` went to 0 this cycle.
  3. Each rank sets `off = 0` on the matching rows of
     `model.spmd_ghost[g].state["off"]`. This must happen on the same cycle
     everywhere, so all TYPE2 holders release identically.
  4. Then remove the "Known limitation" paragraph from `domdec.py`.
  5. Add a `-np 2` vs serial parity test: a TYPE2 tie on a `/FAIL` shell, cut
     by the decomposition.

## SPMD-3 — `/KJOINT` penalty forces counted once per replica (severity: low, latent)

- **Where:** `engine/engine.py:1105` calls `kj.transfer_forces(...)` before
  `exch_forces`. `engine/kjoint.py:498-503` *adds* the full penalty and
  damping forces (`fint[self.idx1] += F1`, and so on), with no SPMD `WEIGHT`.
  `domdec.py:667-669` replicates `model.kjoints` onto every holder. On a
  frontier node, the joint force is applied k times.
- **Why it is latent:** no input reader fills `model.kjoints` yet (it is
  declared at `model.py:1694`; only `tests/test_m602_kjoint.py` sets it, in
  serial). The TYPE33/45 spring path cannot double-count, because a spring
  element lives on one domain and non-TYPE4 spring buffers are already
  refused (`domdec.py:531-537`).
- **Fortran:** `engine/source/elements/joint/ruser33.F` (and `rgjoint.F`).
  TYPE33/45 joints are spring elements, computed once on the owning domain
  and then frontier-summed by `spmd_exch_a.F`. They are never replicated.
- **Fix:** refuse `model.kjoints` in `check_spmd_support`. The alternative is
  to weight the additive terms (`fint[idx] += W[idx]*F`, and the same for
  `mint`), including any additive kjoint loops at `engine.py` around
  1229/1366/1387.

---

## Left over from the red-test fix pass (PR #51)

The five clusters landed in commits `41240aa` and `cf065dc`. Each one was
adversarially verified against the Fortran, and every realigned test cites
its source. These items remain:

1. **PR #51 CI is red: 8 failures in `tests/test_law14_compso.py`.** They are
   expected collateral. The fix made `law14_compso.shell_update` raise, as the
   Fortran does: no `sigeps14c.F` exists upstream, `mulawc.F90:1125-1307`
   dispatches no LAW14 shell kernel, and `hm_read_mat14.F:151-158` raises
   ANCMSG 305. The 8 shell tests exercise a kernel with no Fortran
   counterpart, and their docstring cites the non-existent `sigeps14c.F`.
   They are: `shell_elastic_response`, `shell_layer_orientation`,
   `multilayer_shell_cross_ply`, `shell_directional_damage`,
   `shell_crack_closure`, `shell_tsai_wu_plasticity`, `shell_tangents`,
   `5_component_shell`.
   **Fix:** delete them, or replace them with one assertion that
   `shell_update` raises `NotImplementedError`, as
   `tests/test_m547_law14_compso.py::test_shell_update_not_implemented`
   already does. In the last CI run everything else passed
   (14196 passed / 29 skipped).
2. **Dead LAW14 shell code:** `pyradioss/materials/law14_compso.py` still
   has `multilayer_shell_update` and the `shell_membrane_tangent` /
   `consistent_shell_tangent` helpers, whose docstrings cite `sigeps14c.F`.
   Remove them together with item 1.
3. **Duplicate law registration:** LAW24, LAW37 and LAW90 are registered both
   as dedicated laws and in `_NEW_PORTED_LAWS` in
   `pyradioss/materials/__init__.py`.
4. **Solid-only audit:** `tests/test_mat_all_135_census.py` now skips shell
   checks for `_SOLID_ONLY_LAWS = {24, 37, 90}`. Other laws in
   `_NEW_PORTED_LAWS` (e.g. 11, 13, 51, 54, 151) pass the shell census only
   because their modules carry shell stubs. They should be audited against
   `mulawc.F90` and added to the set where there is no upstream shell kernel.
5. **Duplicate `/PROP` IDs are not rejected.** Upstream
   `hm_read_properties.F:798-802` calls `VDOUBLE`; the port only fails later,
   at PART CHECK.
6. **LAW4 cfg lookup — FIXED (Task P0.6).**
   `tests/test_m535_law04.py::test_direct_read_generic_mat_law4` and
   `::test_direct_read_generic_mat_hyd_jcook` failed in the Linux dev
   container. `read_generic_mat` reported "no cfg schema found" for
   `matl4_hyd_jcook.cfg` even when `PYRADIOSS_HM_CFG` was set, because
   `pyradioss/input/mat_reader.py` built its candidate list from
   `os.environ.get("PYRADIOSS_HM_CFG")` **at import time** plus two
   hardcoded roots (`C:/OpenRadioss/...`, `/opt/OpenRadioss/...`) — none of
   which exist on Linux — and returned `None` in silence, so E was never
   parsed. `pyradioss/paths.py` now owns the lookup: resolution is lazy,
   existence-checked, memoised, re-read by `paths.reload()`, and a missing
   tree raises with every candidate location listed (symbolic origin +
   resolved path). Fix commit: see Task P0.6 in
   `.superpowers/sdd/01_phase0_oracle_and_licensing/task-6-report.md`.
